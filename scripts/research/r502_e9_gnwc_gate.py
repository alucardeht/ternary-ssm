#!/usr/bin/env python3
"""R502 E9 gradient-null witness checkpointing gate.

This CPU diagnostic implements the R502 Efficiency finding:
"Gradient-Null Witness Checkpointing (GNWC)".

The gate builds a synthetic sampled-CE backward path with token-wise gradient
contributions. Dense checkpointing stores every state. GNWC checkpoints only
positions whose cotangent witness is non-null and skips certified-null regions
whose omitted gradient is bounded by a Hessian-radius certificate.

Controls exercise the declared risks: false null witnesses drop real gradient,
witness calculation cost can exceed dense backward propagation, near-dense
non-null regions erase the memory denominator, and invalid Hessian bounds make
the residual certificate unsafe.

This is a synthetic CPU gradient-null checkpointing and CE-progress diagnostic,
not heldout CE progress, real corpus token efficiency, GPU training-speed
evidence, or production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r502_e9_gnwc_gate.v1"
CONDITIONS = (
    "positive",
    "false_null_witness_fail",
    "witness_cost_fail",
    "dense_like_nonnull_fail",
    "hessian_bound_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    sequence_len: int
    grad_dim: int
    nonnull_fraction: float
    checkpoint_memory_ratio: float
    gradient_residual_fraction: float
    certified_bound_fraction: float
    certificate_margin: float
    ce_progress_ratio: float
    witness_cost_ratio: float
    projected_checkpoint_memory_saving_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50209)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sequence-len", type=int, default=2048)
    parser.add_argument("--grad-dim", type=int, default=64)
    parser.add_argument("--nonnull-fraction", type=float, default=0.25)
    parser.add_argument("--dense-like-nonnull-fraction", type=float, default=0.92)
    parser.add_argument("--null-scale", type=float, default=0.015)
    parser.add_argument("--nonnull-scale", type=float, default=0.45)
    parser.add_argument("--max-gradient-residual-fraction", type=float, default=0.10)
    parser.add_argument("--min-ce-progress-ratio", type=float, default=0.90)
    parser.add_argument("--max-checkpoint-memory-ratio", type=float, default=0.60)
    parser.add_argument("--max-witness-cost-ratio", type=float, default=0.80)
    parser.add_argument("--checkpoint-overhead-ratio", type=float, default=0.02)
    parser.add_argument("--witness-cost-overhead-ratio", type=float, default=0.08)
    parser.add_argument("--expensive-witness-cost-overhead-ratio", type=float, default=0.92)
    parser.add_argument("--bound-slack", type=float, default=1.15)
    parser.add_argument("--bad-bound-slack", type=float, default=0.35)
    return parser.parse_args()


def make_gradients(
    args: argparse.Namespace,
    rng: np.random.Generator,
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    if condition == "dense_like_nonnull_fail":
        nonnull_fraction = args.dense_like_nonnull_fraction
    else:
        nonnull_fraction = args.nonnull_fraction
    nonnull_count = int(round(nonnull_fraction * args.sequence_len))
    mask = np.zeros(args.sequence_len, dtype=bool)
    mask[:nonnull_count] = True
    rng.shuffle(mask)
    gradients = rng.normal(scale=args.null_scale, size=(args.sequence_len, args.grad_dim))
    gradients[mask] += rng.normal(scale=args.nonnull_scale, size=(nonnull_count, args.grad_dim))
    if condition == "false_null_witness_fail":
        active_indices = np.flatnonzero(mask)
        drop = active_indices[: max(1, active_indices.shape[0] // 3)]
        mask[drop] = False
    return gradients, mask


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    gradients, witness_mask = make_gradients(args, rng, condition)
    dense_gradient = np.sum(gradients, axis=0)
    gnwc_gradient = np.sum(gradients[witness_mask], axis=0)
    omitted = gradients[~witness_mask]
    omitted_gradient = np.sum(omitted, axis=0)
    dense_norm = max(float(np.linalg.norm(dense_gradient)), 1.0e-30)
    residual_fraction = float(np.linalg.norm(gnwc_gradient - dense_gradient) / dense_norm)
    omitted_norm = float(np.linalg.norm(omitted_gradient))
    slack = args.bad_bound_slack if condition == "hessian_bound_fail" else args.bound_slack
    certified_bound = slack * omitted_norm
    certified_bound_fraction = float(certified_bound / dense_norm)
    certificate_margin = float((certified_bound - omitted_norm) / dense_norm)
    ce_progress_ratio = float(
        np.dot(gnwc_gradient, dense_gradient)
        / max(float(np.dot(dense_gradient, dense_gradient)), 1.0e-30)
    )
    nonnull_fraction = float(np.mean(witness_mask))
    checkpoint_memory_ratio = nonnull_fraction + args.checkpoint_overhead_ratio
    witness_overhead = (
        args.expensive_witness_cost_overhead_ratio
        if condition == "witness_cost_fail"
        else args.witness_cost_overhead_ratio
    )
    witness_cost_ratio = nonnull_fraction + witness_overhead
    projected_multiplier = 1.0 / checkpoint_memory_ratio

    failed: list[str] = []
    if not finite(
        nonnull_fraction,
        checkpoint_memory_ratio,
        residual_fraction,
        certified_bound_fraction,
        certificate_margin,
        ce_progress_ratio,
        witness_cost_ratio,
        projected_multiplier,
    ):
        failed.append("non_finite_metric")
    if residual_fraction > args.max_gradient_residual_fraction:
        failed.append("gradient_residual")
    if certificate_margin < 0.0:
        failed.append("certificate_bound")
    if ce_progress_ratio < args.min_ce_progress_ratio:
        failed.append("ce_progress")
    if checkpoint_memory_ratio > args.max_checkpoint_memory_ratio:
        failed.append("checkpoint_memory_price")
    if witness_cost_ratio > args.max_witness_cost_ratio:
        failed.append("witness_cost")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed) if failed else None,
        sequence_len=args.sequence_len,
        grad_dim=args.grad_dim,
        nonnull_fraction=nonnull_fraction,
        checkpoint_memory_ratio=checkpoint_memory_ratio,
        gradient_residual_fraction=residual_fraction,
        certified_bound_fraction=certified_bound_fraction,
        certificate_margin=certificate_margin,
        ce_progress_ratio=ce_progress_ratio,
        witness_cost_ratio=witness_cost_ratio,
        projected_checkpoint_memory_saving_multiplier=projected_multiplier,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "sequence_len",
        "grad_dim",
        "nonnull_fraction",
        "checkpoint_memory_ratio",
        "gradient_residual_fraction",
        "certified_bound_fraction",
        "certificate_margin",
        "ce_progress_ratio",
        "witness_cost_ratio",
        "projected_checkpoint_memory_saving_multiplier",
    )
    aggregate: dict[str, Any] = {
        "count": len(results),
        "passed_count": sum(result.passed for result in results),
        "identifying_correct_count": sum(result.identifying_correct for result in results),
        "fail_reasons": sorted({result.fail_reason for result in results if result.fail_reason}),
    }
    for field in fields:
        values = np.array([float(getattr(result, field)) for result in results], dtype=np.float64)
        aggregate[f"{field}_mean"] = float(np.mean(values))
        aggregate[f"{field}_min"] = float(np.min(values))
        aggregate[f"{field}_max"] = float(np.max(values))
    return aggregate


def all_finite(results: list[TrialResult]) -> bool:
    for result in results:
        for value in asdict(result).values():
            if isinstance(value, float) and not math.isfinite(value):
                return False
    return True


def required_checks(results: list[TrialResult], aggregates: dict[str, Any]) -> dict[str, bool]:
    positive = [result for result in results if result.condition == "positive"]
    false_null = [result for result in results if result.condition == "false_null_witness_fail"]
    cost = [result for result in results if result.condition == "witness_cost_fail"]
    dense_like = [result for result in results if result.condition == "dense_like_nonnull_fail"]
    bound = [result for result in results if result.condition == "hessian_bound_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_residual_and_progress": (
            aggregates["positive"]["gradient_residual_fraction_max"] <= 0.10
            and aggregates["positive"]["ce_progress_ratio_min"] >= 0.90
        ),
        "positive_condition_certificate_safe": (
            aggregates["positive"]["certificate_margin_min"] >= 0.0
        ),
        "positive_condition_memory_and_cost_priced": (
            aggregates["positive"]["checkpoint_memory_ratio_mean"] <= 0.60
            and aggregates["positive"]["witness_cost_ratio_mean"] <= 0.80
        ),
        "false_null_control_all_fail_residual_or_progress": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "gradient_residual" in result.fail_reason
                or "ce_progress" in result.fail_reason
            )
            for result in false_null
        ),
        "witness_cost_control_all_fail_cost": all(
            (not result.passed)
            and result.fail_reason is not None
            and "witness_cost" in result.fail_reason
            for result in cost
        ),
        "dense_like_control_all_fail_memory": all(
            (not result.passed)
            and result.fail_reason is not None
            and "checkpoint_memory_price" in result.fail_reason
            for result in dense_like
        ),
        "hessian_bound_control_all_fail_certificate": all(
            (not result.passed)
            and result.fail_reason is not None
            and "certificate_bound" in result.fail_reason
            for result in bound
        ),
        "all_conditions_identified": all(result.identifying_correct for result in results),
    }


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.sequence_len < 2:
        raise ValueError("--sequence-len must be at least 2")
    if args.grad_dim < 1:
        raise ValueError("--grad-dim must be positive")

    results: list[TrialResult] = []
    for condition in CONDITIONS:
        for offset in range(args.seed_count):
            results.append(run_trial(args, condition, args.seed + offset))

    aggregates = {
        condition: aggregate_condition(
            [result for result in results if result.condition == condition]
        )
        for condition in CONDITIONS
    }
    checks = required_checks(results, aggregates)
    decision = "PASS_R502_E9_GNWC_GATE" if all(checks.values()) else "FAIL_R502_E9_GNWC_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU gradient-null witness checkpointing gate only; not "
            "heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_E9_GNWC_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_nonnull_fraction_mean={positive['nonnull_fraction_mean']:.6f} "
        f"positive_gradient_residual_fraction_max={positive['gradient_residual_fraction_max']:.12e} "
        f"positive_certificate_margin_min={positive['certificate_margin_min']:.12e} "
        f"positive_ce_progress_ratio_min={positive['ce_progress_ratio_min']:.12e} "
        f"positive_checkpoint_memory_ratio_mean={positive['checkpoint_memory_ratio_mean']:.6f} "
        f"positive_witness_cost_ratio_mean={positive['witness_cost_ratio_mean']:.6f} "
        f"positive_projected_checkpoint_memory_saving_multiplier_mean={positive['projected_checkpoint_memory_saving_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
