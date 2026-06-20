#!/usr/bin/env python3
"""R501 S80 recurrent target-containing subspace certificate gate.

This CPU diagnostic implements the R501 Quality Paradigm 2 finding:
"Recurrent Target-Containing Subspace Certificate (RTCSC)".

The gate constructs a parameter-update target whose useful capability gain is
claimed to lie in the span of recent hidden states. The candidate projects the
target into that span and then realizes the projected coefficients through a
ternary coefficient alphabet. A positive case must certify containment,
preserve the target loss under projection, preserve the ternary realization,
and keep the subspace search denominator below the dense search denominator.

Controls exercise the declared risks: the target can sit outside the hidden
span, the hidden span can be stale, a large span can erase the denominator, the
continuous certificate can fail under ternary realization, and an expensive
certificate can recover dense cost.

This is a synthetic CPU subspace-certificate diagnostic, not heldout CE
progress, real corpus token efficiency, GPU training-speed evidence, or
production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s80_rtcsc_gate.v1"
CONDITIONS = (
    "positive",
    "target_outside_span_fail",
    "stale_hidden_span_fail",
    "rank_too_large_fail",
    "ternary_resolution_fail",
    "certificate_cost_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    param_dim: int
    active_rank: int
    active_rank_fraction: float
    containment_residual_ratio: float
    continuous_loss_ratio: float
    ternary_realization_error_ratio: float
    ternary_loss_ratio: float
    descent_cosine: float
    dense_work_units: float
    subspace_work_units: float
    certificate_work_units: float
    priced_work_ratio: float
    projected_token_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50180)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--param-dim", type=int, default=96)
    parser.add_argument("--active-rank", type=int, default=12)
    parser.add_argument("--ternary-scale", type=float, default=0.18)
    parser.add_argument("--max-active-rank-fraction", type=float, default=0.30)
    parser.add_argument("--max-containment-residual-ratio", type=float, default=0.08)
    parser.add_argument("--max-continuous-loss-ratio", type=float, default=0.01)
    parser.add_argument("--max-ternary-error-ratio", type=float, default=0.12)
    parser.add_argument("--max-ternary-loss-ratio", type=float, default=0.015)
    parser.add_argument("--min-descent-cosine", type=float, default=0.995)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.45)
    parser.add_argument("--min-token-efficiency-multiplier", type=float, default=2.0)
    parser.add_argument("--certificate-overhead", type=float, default=0.08)
    parser.add_argument("--expensive-certificate-overhead", type=float, default=0.40)
    return parser.parse_args()


def orthonormal_basis(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    matrix = rng.normal(size=(rows, cols))
    q, _ = np.linalg.qr(matrix, mode="reduced")
    return q[:, :cols]


def orthogonal_residual(
    rng: np.random.Generator,
    basis: np.ndarray,
    norm: float,
) -> np.ndarray:
    vector = rng.normal(size=basis.shape[0])
    vector = vector - basis @ (basis.T @ vector)
    vector_norm = max(float(np.linalg.norm(vector)), 1.0e-30)
    return vector * (norm / vector_norm)


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0e-30)
    return float(np.dot(a, b) / denom)


def quantize_ternary(coefficients: np.ndarray, scale: float) -> np.ndarray:
    return np.where(np.abs(coefficients) >= 0.5 * scale, np.sign(coefficients) * scale, 0.0)


def ternary_friendly_coefficients(
    rng: np.random.Generator,
    rank: int,
    scale: float,
) -> np.ndarray:
    coefficients = rng.choice(np.array([-scale, 0.0, scale]), size=rank, p=[0.42, 0.16, 0.42])
    if float(np.linalg.norm(coefficients)) < scale:
        coefficients[0] = scale
    return coefficients


def make_target(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
    true_basis: np.ndarray,
) -> np.ndarray:
    rank = true_basis.shape[1]
    if condition == "ternary_resolution_fail":
        coefficients = rng.uniform(-0.45 * args.ternary_scale, 0.45 * args.ternary_scale, size=rank)
        coefficients[::3] += 0.62 * args.ternary_scale
    else:
        coefficients = ternary_friendly_coefficients(rng, rank, args.ternary_scale)
    in_span = true_basis @ coefficients
    in_span_norm = max(float(np.linalg.norm(in_span)), 1.0e-30)

    if condition == "target_outside_span_fail":
        return in_span + orthogonal_residual(rng, true_basis, 0.90 * in_span_norm)
    return in_span + orthogonal_residual(rng, true_basis, 0.010 * in_span_norm)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    active_rank = args.active_rank
    if condition == "rank_too_large_fail":
        active_rank = int(round(0.72 * args.param_dim))

    true_basis = orthonormal_basis(rng, args.param_dim, active_rank)
    target = make_target(rng, args, condition, true_basis)
    candidate_basis = true_basis
    if condition == "stale_hidden_span_fail":
        candidate_basis = orthonormal_basis(rng, args.param_dim, active_rank)

    coefficients = candidate_basis.T @ target
    continuous_update = candidate_basis @ coefficients
    ternary_update = candidate_basis @ quantize_ternary(coefficients, args.ternary_scale)

    target_norm = max(float(np.linalg.norm(target)), 1.0e-30)
    initial_loss = 0.5 * target_norm * target_norm
    continuous_loss = 0.5 * float(np.linalg.norm(target - continuous_update) ** 2)
    ternary_loss = 0.5 * float(np.linalg.norm(target - ternary_update) ** 2)
    continuous_loss_ratio = continuous_loss / max(initial_loss, 1.0e-30)
    ternary_loss_ratio = ternary_loss / max(initial_loss, 1.0e-30)
    containment_residual = float(np.linalg.norm(target - continuous_update)) / target_norm
    ternary_error = float(np.linalg.norm(target - ternary_update)) / target_norm
    descent_direction = cosine(ternary_update, target)

    dense_work = float(args.param_dim)
    subspace_work = float(active_rank)
    overhead = (
        args.expensive_certificate_overhead
        if condition == "certificate_cost_fail"
        else args.certificate_overhead
    )
    certificate_work = overhead * dense_work
    priced_work_ratio = (subspace_work + certificate_work) / dense_work
    quality_retained = max(0.0, 1.0 - ternary_loss_ratio)
    projected_multiplier = quality_retained / max(priced_work_ratio, 1.0e-30)
    active_rank_fraction = active_rank / args.param_dim

    failed: list[str] = []
    if not finite(
        active_rank_fraction,
        containment_residual,
        continuous_loss_ratio,
        ternary_error,
        ternary_loss_ratio,
        descent_direction,
        dense_work,
        subspace_work,
        certificate_work,
        priced_work_ratio,
        projected_multiplier,
    ):
        failed.append("non_finite_metric")
    if active_rank_fraction > args.max_active_rank_fraction:
        failed.append("rank_not_condensed")
    if containment_residual > args.max_containment_residual_ratio:
        failed.append("target_not_contained")
    if continuous_loss_ratio > args.max_continuous_loss_ratio:
        failed.append("continuous_loss_gap")
    if ternary_error > args.max_ternary_error_ratio:
        failed.append("ternary_realization_error")
    if ternary_loss_ratio > args.max_ternary_loss_ratio:
        failed.append("ternary_loss_gap")
    if descent_direction < args.min_descent_cosine:
        failed.append("descent_direction")
    if priced_work_ratio > args.max_priced_work_ratio:
        failed.append("price")
    if projected_multiplier < args.min_token_efficiency_multiplier:
        failed.append("token_efficiency")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed) if failed else None,
        param_dim=args.param_dim,
        active_rank=active_rank,
        active_rank_fraction=float(active_rank_fraction),
        containment_residual_ratio=float(containment_residual),
        continuous_loss_ratio=float(continuous_loss_ratio),
        ternary_realization_error_ratio=float(ternary_error),
        ternary_loss_ratio=float(ternary_loss_ratio),
        descent_cosine=float(descent_direction),
        dense_work_units=float(dense_work),
        subspace_work_units=float(subspace_work),
        certificate_work_units=float(certificate_work),
        priced_work_ratio=float(priced_work_ratio),
        projected_token_efficiency_multiplier=float(projected_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "param_dim",
        "active_rank",
        "active_rank_fraction",
        "containment_residual_ratio",
        "continuous_loss_ratio",
        "ternary_realization_error_ratio",
        "ternary_loss_ratio",
        "descent_cosine",
        "dense_work_units",
        "subspace_work_units",
        "certificate_work_units",
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
    outside = [result for result in results if result.condition == "target_outside_span_fail"]
    stale = [result for result in results if result.condition == "stale_hidden_span_fail"]
    large_rank = [result for result in results if result.condition == "rank_too_large_fail"]
    ternary = [result for result in results if result.condition == "ternary_resolution_fail"]
    cost = [result for result in results if result.condition == "certificate_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_rank_condensed": (
            aggregates["positive"]["active_rank_fraction_mean"] <= 0.30
        ),
        "positive_condition_target_contained": (
            aggregates["positive"]["containment_residual_ratio_max"] <= 0.08
            and aggregates["positive"]["continuous_loss_ratio_max"] <= 0.01
        ),
        "positive_condition_ternary_realized": (
            aggregates["positive"]["ternary_realization_error_ratio_max"] <= 0.12
            and aggregates["positive"]["ternary_loss_ratio_max"] <= 0.015
            and aggregates["positive"]["descent_cosine_min"] >= 0.995
        ),
        "positive_condition_priced": (
            aggregates["positive"]["priced_work_ratio_mean"] <= 0.45
            and aggregates["positive"]["projected_token_efficiency_multiplier_mean"] >= 2.0
        ),
        "outside_control_all_fail_containment": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "target_not_contained" in result.fail_reason
                or "continuous_loss_gap" in result.fail_reason
            )
            for result in outside
        ),
        "stale_control_all_fail_containment": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "target_not_contained" in result.fail_reason
                or "continuous_loss_gap" in result.fail_reason
            )
            for result in stale
        ),
        "large_rank_control_all_fail_price_or_rank": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "rank_not_condensed" in result.fail_reason
                or "price" in result.fail_reason
            )
            for result in large_rank
        ),
        "ternary_control_all_fail_realization": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "ternary_realization_error" in result.fail_reason
                or "ternary_loss_gap" in result.fail_reason
            )
            for result in ternary
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
        "PASS_R501_S80_RTCSC_GATE"
        if all(checks.values())
        else "FAIL_R501_S80_RTCSC_GATE"
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
            "Synthetic CPU recurrent target-containing subspace certificate "
            "gate only; not heldout CE progress, not real corpus token "
            "efficiency, not GPU training speed, and not production v12 "
            "integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S80_RTCSC_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_active_rank_fraction_mean={positive['active_rank_fraction_mean']:.6f} "
        f"positive_containment_residual_ratio_max={positive['containment_residual_ratio_max']:.6f} "
        f"positive_continuous_loss_ratio_max={positive['continuous_loss_ratio_max']:.6f} "
        f"positive_ternary_realization_error_ratio_max={positive['ternary_realization_error_ratio_max']:.6f} "
        f"positive_ternary_loss_ratio_max={positive['ternary_loss_ratio_max']:.6f} "
        f"positive_descent_cosine_min={positive['descent_cosine_min']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_token_efficiency_multiplier_mean={positive['projected_token_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
