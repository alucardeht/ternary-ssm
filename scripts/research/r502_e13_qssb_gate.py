#!/usr/bin/env python3
"""R502 E13 quotient-space sparse backward gate.

This CPU diagnostic implements the R502 Efficiency finding:
"Quotient-Space Sparse Backward (QSSB)".

The gate builds a synthetic recurrent cotangent field, prunes cotangents below
per-time thresholds, prices the retained backward reads, and checks that the
paid descent per unit work clears the stated gate while an error-debt certificate
and martingale-style verifier alignment remain safe.

Controls exercise the declared risks: over-pruning creates unpaid error debt,
under-pruning fails the paid-work denominator, unstable transition debt exceeds
budget, verifier-noise breaks martingale alignment, and reading dense work
despite sparse cotangents erases the speed denominator.

This is a synthetic CPU quotient-space sparse-backward diagnostic, not heldout
CE progress, real corpus token efficiency, GPU training-speed evidence, Vulkan
shader evidence, or production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r502_e13_qssb_gate.v1"
CONDITIONS = (
    "positive",
    "over_prune_noise_fail",
    "under_prune_price_fail",
    "debt_budget_fail",
    "martingale_alignment_fail",
    "dense_work_leak_fail",
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
    state_dim: int
    retained_fraction: float
    paid_work_ratio: float
    residual_fraction: float
    certified_debt_fraction: float
    debt_margin: float
    descent_ratio: float
    descent_per_paid_work: float
    martingale_alignment: float
    projected_backward_work_saving_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50213)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sequence-len", type=int, default=512)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--active-fraction", type=float, default=0.38)
    parser.add_argument("--positive-prune-quantile", type=float, default=0.44)
    parser.add_argument("--over-prune-quantile", type=float, default=0.86)
    parser.add_argument("--under-prune-quantile", type=float, default=0.05)
    parser.add_argument("--access-overhead-ratio", type=float, default=0.035)
    parser.add_argument("--dense-work-leak-ratio", type=float, default=1.0)
    parser.add_argument("--transition-radius", type=float, default=0.97)
    parser.add_argument("--unstable-transition-radius", type=float, default=1.18)
    parser.add_argument("--debt-slack", type=float, default=1.15)
    parser.add_argument("--max-certified-debt-fraction", type=float, default=0.10)
    parser.add_argument("--min-descent-ratio", type=float, default=0.90)
    parser.add_argument("--min-descent-per-paid-work", type=float, default=1.111)
    parser.add_argument("--min-martingale-alignment", type=float, default=0.95)
    parser.add_argument("--max-paid-work-ratio", type=float, default=0.85)
    return parser.parse_args()


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0e-30)
    return float(np.dot(a, b) / denom)


def make_cotangents(args: argparse.Namespace, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    active_mask = rng.random((args.sequence_len, args.state_dim)) < args.active_fraction
    cotangent = rng.normal(0.0, 0.018, size=(args.sequence_len, args.state_dim))
    cotangent[active_mask] += rng.normal(
        0.0,
        0.44,
        size=int(np.sum(active_mask)),
    )
    features = rng.normal(0.0, 0.35, size=(args.sequence_len, args.state_dim))
    return cotangent, features


def threshold_for_condition(args: argparse.Namespace, condition: str) -> float:
    if condition == "over_prune_noise_fail":
        return args.over_prune_quantile
    if condition == "under_prune_price_fail":
        return args.under_prune_quantile
    return args.positive_prune_quantile


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    cotangent, features = make_cotangents(args, rng)
    abs_cotangent = np.abs(cotangent)
    threshold = float(np.quantile(abs_cotangent, threshold_for_condition(args, condition)))
    retained = abs_cotangent > threshold
    pruned_cotangent = np.where(retained, cotangent, 0.0)

    if condition == "martingale_alignment_fail":
        retained_indices = np.flatnonzero(retained.reshape(-1))
        flip_count = max(1, retained_indices.shape[0] // 3)
        flat = pruned_cotangent.reshape(-1)
        flat[retained_indices[:flip_count]] *= -1.0
        pruned_cotangent = flat.reshape(pruned_cotangent.shape)

    dense_update = np.sum(cotangent * features, axis=0)
    sparse_update = np.sum(pruned_cotangent * features, axis=0)
    dense_norm = max(float(np.linalg.norm(dense_update)), 1.0e-30)
    residual_fraction = float(np.linalg.norm(sparse_update - dense_update) / dense_norm)
    descent_ratio = float(np.dot(sparse_update, dense_update) / max(np.dot(dense_update, dense_update), 1.0e-30))

    dense_time_witness = np.sum(cotangent * features, axis=1)
    sparse_time_witness = np.sum(pruned_cotangent * features, axis=1)
    martingale_alignment = cosine(dense_time_witness, sparse_time_witness)

    retained_fraction = float(np.mean(retained))
    if condition == "dense_work_leak_fail":
        paid_work_ratio = args.dense_work_leak_ratio
    else:
        paid_work_ratio = min(1.0, retained_fraction + args.access_overhead_ratio)
    descent_per_paid_work = descent_ratio / max(paid_work_ratio, 1.0e-30)
    projected_multiplier = 1.0 / paid_work_ratio

    radius = (
        args.unstable_transition_radius
        if condition == "debt_budget_fail"
        else args.transition_radius
    )
    power_norm = radius ** np.arange(args.sequence_len - 1, -1, -1, dtype=np.float64)
    pruned_abs_by_time = np.sum(np.abs(cotangent) * (~retained), axis=1)
    dense_abs = max(float(np.sum(np.abs(cotangent) * np.abs(features))), 1.0e-30)
    certified_debt_fraction = float(
        args.debt_slack
        * np.sum(pruned_abs_by_time * power_norm)
        / dense_abs
    )
    debt_margin = args.max_certified_debt_fraction - certified_debt_fraction

    failed: list[str] = []
    if not finite(
        retained_fraction,
        paid_work_ratio,
        residual_fraction,
        certified_debt_fraction,
        debt_margin,
        descent_ratio,
        descent_per_paid_work,
        martingale_alignment,
        projected_multiplier,
    ):
        failed.append("non_finite_metric")
    if certified_debt_fraction > args.max_certified_debt_fraction:
        failed.append("error_debt")
    if residual_fraction > args.max_certified_debt_fraction:
        failed.append("residual")
    if descent_ratio < args.min_descent_ratio:
        failed.append("descent")
    if descent_per_paid_work < args.min_descent_per_paid_work:
        failed.append("descent_per_paid_work")
    if martingale_alignment < args.min_martingale_alignment:
        failed.append("martingale_alignment")
    if paid_work_ratio > args.max_paid_work_ratio:
        failed.append("paid_work_price")

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
        state_dim=args.state_dim,
        retained_fraction=retained_fraction,
        paid_work_ratio=paid_work_ratio,
        residual_fraction=residual_fraction,
        certified_debt_fraction=certified_debt_fraction,
        debt_margin=debt_margin,
        descent_ratio=descent_ratio,
        descent_per_paid_work=descent_per_paid_work,
        martingale_alignment=martingale_alignment,
        projected_backward_work_saving_multiplier=projected_multiplier,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "sequence_len",
        "state_dim",
        "retained_fraction",
        "paid_work_ratio",
        "residual_fraction",
        "certified_debt_fraction",
        "debt_margin",
        "descent_ratio",
        "descent_per_paid_work",
        "martingale_alignment",
        "projected_backward_work_saving_multiplier",
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
    over_prune = [result for result in results if result.condition == "over_prune_noise_fail"]
    under_prune = [result for result in results if result.condition == "under_prune_price_fail"]
    debt = [result for result in results if result.condition == "debt_budget_fail"]
    martingale = [result for result in results if result.condition == "martingale_alignment_fail"]
    dense_leak = [result for result in results if result.condition == "dense_work_leak_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_debt_descent_alignment": (
            aggregates["positive"]["certified_debt_fraction_max"] <= 0.10
            and aggregates["positive"]["descent_ratio_min"] >= 0.90
            and aggregates["positive"]["martingale_alignment_min"] >= 0.95
        ),
        "positive_condition_paid_work_gate": (
            aggregates["positive"]["paid_work_ratio_mean"] <= 0.85
            and aggregates["positive"]["descent_per_paid_work_min"] > 1.111
        ),
        "over_prune_control_all_fail_debt_or_descent": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "error_debt" in result.fail_reason
                or "residual" in result.fail_reason
                or "descent" in result.fail_reason
            )
            for result in over_prune
        ),
        "under_prune_control_all_fail_paid_work": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "descent_per_paid_work" in result.fail_reason
                or "paid_work_price" in result.fail_reason
            )
            for result in under_prune
        ),
        "debt_budget_control_all_fail_debt": all(
            (not result.passed)
            and result.fail_reason is not None
            and "error_debt" in result.fail_reason
            for result in debt
        ),
        "martingale_control_all_fail_alignment": all(
            (not result.passed)
            and result.fail_reason is not None
            and "martingale_alignment" in result.fail_reason
            for result in martingale
        ),
        "dense_work_leak_control_all_fail_paid_work": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "descent_per_paid_work" in result.fail_reason
                or "paid_work_price" in result.fail_reason
            )
            for result in dense_leak
        ),
        "all_conditions_identified": all(result.identifying_correct for result in results),
    }


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.sequence_len < 2:
        raise ValueError("--sequence-len must be at least 2")
    if args.state_dim < 1:
        raise ValueError("--state-dim must be positive")

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
    decision = "PASS_R502_E13_QSSB_GATE" if all(checks.values()) else "FAIL_R502_E13_QSSB_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU quotient-space sparse-backward gate only; not "
            "heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, not Vulkan shader evidence, and not production "
            "v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_E13_QSSB_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_retained_fraction_mean={positive['retained_fraction_mean']:.6f} "
        f"positive_paid_work_ratio_mean={positive['paid_work_ratio_mean']:.6f} "
        f"positive_certified_debt_fraction_max={positive['certified_debt_fraction_max']:.12e} "
        f"positive_descent_ratio_min={positive['descent_ratio_min']:.12e} "
        f"positive_descent_per_paid_work_min={positive['descent_per_paid_work_min']:.12e} "
        f"positive_martingale_alignment_min={positive['martingale_alignment_min']:.12e} "
        f"positive_projected_backward_work_saving_multiplier_mean={positive['projected_backward_work_saving_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
