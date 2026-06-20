#!/usr/bin/env python3
"""R501 S73 martingale-preserving local contrastive credit gate.

This CPU diagnostic implements the R501 Quality Math 10 finding:
"Martingale-Preserving Local Contrastive Credit (MP-LCC)".

The gate constructs a low-rank hidden-state trajectory and a local contrastive
credit term that pushes states away from negative samples. A positive case must
strictly increase the effective singular-value rank while keeping the expected
contrastive gradient centered so the CE martingale direction is preserved.

Controls exercise the declared risks: an uncentered contrastive estimator biases
the CE objective, weak local negatives fail to increase rank, and an expensive
sliding-window contrastive pass erases the priced denominator.

This is a synthetic CPU state-spectrum and gradient-bias diagnostic, not
heldout CE progress, real corpus token efficiency, GPU training-speed evidence,
or production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s73_mp_lcc_gate.v1"
CONDITIONS = (
    "positive",
    "uncentered_contrastive_fail",
    "weak_negative_fail",
    "contrastive_cost_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    token_count: int
    state_dim: int
    baseline_effective_rank: float
    mp_lcc_effective_rank: float
    effective_rank_gain: float
    effective_rank_gain_fraction: float
    contrastive_gradient_bias_norm: float
    ce_gradient_cosine: float
    contrastive_mean_norm: float
    priced_work_ratio: float
    projected_token_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50173)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--token-count", type=int, default=160)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--baseline-rank", type=int, default=3)
    parser.add_argument("--contrastive-scale", type=float, default=0.65)
    parser.add_argument("--max-gradient-bias", type=float, default=0.025)
    parser.add_argument("--min-rank-gain-fraction", type=float, default=0.50)
    parser.add_argument("--min-ce-gradient-cosine", type=float, default=0.995)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.85)
    parser.add_argument("--priced-work-ratio", type=float, default=0.78)
    parser.add_argument("--expensive-priced-work-ratio", type=float, default=0.91)
    return parser.parse_args()


def orthonormal(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    matrix = rng.normal(size=(rows, cols))
    q, _r = np.linalg.qr(matrix)
    return q[:, :cols]


def effective_rank(states: np.ndarray) -> float:
    centered = states - np.mean(states, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = singular_values * singular_values
    total = max(float(np.sum(energy)), 1.0e-30)
    probabilities = energy / total
    probabilities = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(np.exp(entropy))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0e-30)
    return float(np.dot(a, b) / denom)


def make_states(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    basis = orthonormal(rng, args.state_dim, args.baseline_rank)
    coefficients = rng.normal(size=(args.token_count, args.baseline_rank))
    baseline = coefficients @ basis.T
    baseline += rng.normal(scale=0.025, size=baseline.shape)

    residual = rng.normal(size=(args.token_count, args.state_dim))
    residual = residual - residual @ basis @ basis.T
    residual -= np.mean(residual, axis=0, keepdims=True)
    residual /= max(float(np.sqrt(np.mean(residual * residual))), 1.0e-30)

    if condition == "weak_negative_fail":
        contrastive = 0.05 * (coefficients @ basis.T)
        contrastive -= np.mean(contrastive, axis=0, keepdims=True)
    else:
        contrastive = residual

    ce_base_direction = rng.normal(size=args.state_dim)
    ce_base_direction /= max(float(np.linalg.norm(ce_base_direction)), 1.0e-30)
    ce_gradients = ce_base_direction[None, :] + rng.normal(scale=0.03, size=baseline.shape)

    if condition == "uncentered_contrastive_fail":
        contrastive = contrastive + 0.40 * ce_base_direction[None, :]

    return baseline, contrastive, ce_gradients, ce_base_direction


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    baseline, contrastive, ce_gradients, _ce_direction = make_states(rng, args, condition)
    candidate = baseline + args.contrastive_scale * contrastive
    baseline_rank = effective_rank(baseline)
    candidate_rank = effective_rank(candidate)
    rank_gain = candidate_rank - baseline_rank
    rank_gain_fraction = rank_gain / max(baseline_rank, 1.0e-30)

    contrastive_gradients = args.contrastive_scale * contrastive
    total_gradients = ce_gradients + contrastive_gradients
    ce_mean = np.mean(ce_gradients, axis=0)
    total_mean = np.mean(total_gradients, axis=0)
    contrastive_mean = np.mean(contrastive_gradients, axis=0)
    bias_norm = float(np.linalg.norm(total_mean - ce_mean)) / max(float(np.linalg.norm(ce_mean)), 1.0e-30)
    ce_cosine = cosine(total_mean, ce_mean)
    contrastive_mean_norm = float(np.linalg.norm(contrastive_mean))
    priced_work_ratio = (
        args.expensive_priced_work_ratio
        if condition == "contrastive_cost_fail"
        else args.priced_work_ratio
    )
    projected_multiplier = 1.0 / priced_work_ratio

    failed: list[str] = []
    if not finite(
        baseline_rank,
        candidate_rank,
        rank_gain,
        rank_gain_fraction,
        bias_norm,
        ce_cosine,
        contrastive_mean_norm,
        priced_work_ratio,
        projected_multiplier,
    ):
        failed.append("non_finite_metric")
    if rank_gain_fraction < args.min_rank_gain_fraction:
        failed.append("effective_rank")
    if bias_norm > args.max_gradient_bias:
        failed.append("gradient_bias")
    if ce_cosine < args.min_ce_gradient_cosine:
        failed.append("ce_direction")
    if priced_work_ratio > args.max_priced_work_ratio:
        failed.append("price")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed) if failed else None,
        token_count=args.token_count,
        state_dim=args.state_dim,
        baseline_effective_rank=float(baseline_rank),
        mp_lcc_effective_rank=float(candidate_rank),
        effective_rank_gain=float(rank_gain),
        effective_rank_gain_fraction=float(rank_gain_fraction),
        contrastive_gradient_bias_norm=float(bias_norm),
        ce_gradient_cosine=float(ce_cosine),
        contrastive_mean_norm=float(contrastive_mean_norm),
        priced_work_ratio=float(priced_work_ratio),
        projected_token_efficiency_multiplier=float(projected_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "token_count",
        "state_dim",
        "baseline_effective_rank",
        "mp_lcc_effective_rank",
        "effective_rank_gain",
        "effective_rank_gain_fraction",
        "contrastive_gradient_bias_norm",
        "ce_gradient_cosine",
        "contrastive_mean_norm",
        "priced_work_ratio",
        "projected_token_efficiency_multiplier",
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
    uncentered = [result for result in results if result.condition == "uncentered_contrastive_fail"]
    weak = [result for result in results if result.condition == "weak_negative_fail"]
    cost = [result for result in results if result.condition == "contrastive_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_rank_increased": (
            aggregates["positive"]["effective_rank_gain_fraction_mean"] >= 0.50
            and aggregates["positive"]["mp_lcc_effective_rank_mean"]
            > aggregates["positive"]["baseline_effective_rank_mean"]
        ),
        "positive_condition_martingale_preserved": (
            aggregates["positive"]["contrastive_gradient_bias_norm_max"] <= 0.025
            and aggregates["positive"]["ce_gradient_cosine_min"] >= 0.995
        ),
        "positive_condition_priced": (
            aggregates["positive"]["priced_work_ratio_mean"] <= 0.85
        ),
        "uncentered_control_all_fail_bias": all(
            (not result.passed)
            and result.fail_reason is not None
            and ("gradient_bias" in result.fail_reason or "ce_direction" in result.fail_reason)
            for result in uncentered
        ),
        "weak_negative_control_all_fail_rank": all(
            (not result.passed)
            and result.fail_reason is not None
            and "effective_rank" in result.fail_reason
            for result in weak
        ),
        "cost_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "price" in result.fail_reason
            for result in cost
        ),
        "all_conditions_identified": all(result.identifying_correct for result in results),
    }


def main() -> int:
    args = parse_args()
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
    decision = (
        "PASS_R501_S73_MP_LCC_GATE"
        if all(checks.values())
        else "FAIL_R501_S73_MP_LCC_GATE"
    )
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU martingale-preserving local contrastive credit gate "
            "only; not heldout CE progress, not real corpus token efficiency, "
            "not GPU training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S73_MP_LCC_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_baseline_effective_rank_mean={positive['baseline_effective_rank_mean']:.6f} "
        f"positive_mp_lcc_effective_rank_mean={positive['mp_lcc_effective_rank_mean']:.6f} "
        f"positive_effective_rank_gain_fraction_mean={positive['effective_rank_gain_fraction_mean']:.6f} "
        f"positive_gradient_bias_norm_max={positive['contrastive_gradient_bias_norm_max']:.6f} "
        f"positive_ce_gradient_cosine_min={positive['ce_gradient_cosine_min']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_token_efficiency_multiplier_mean={positive['projected_token_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
