#!/usr/bin/env python3
"""R502-S27 Quotient Space Gradient Embedding gate.

This synthetic CPU gate checks whether a recurrent cotangent can be projected
onto a quotient-space active subspace while preserving the parameter-gradient
descent direction at a lower priced backward cost. Controls cover full-rank
active spaces, non-negligible null cotangents, misaligned bases, expensive
projection, and null pullback amplification.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


CONDITIONS = [
    "positive",
    "full_rank_fail",
    "null_cotangent_fail",
    "basis_misalignment_fail",
    "projection_cost_fail",
    "null_pullback_fail",
]


@dataclass
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str
    state_dim: int
    active_rank: int
    param_dim: int
    sample_count: int
    active_rank_ratio: float
    null_energy_fraction: float
    gradient_relative_error: float
    max_abs_gradient_error: float
    descent_cosine: float
    basis_alignment_min: float
    null_pullback_gain: float
    dense_work_units: float
    projection_work_units: float
    reduced_work_units: float
    priced_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=50227)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--active-rank", type=int, default=8)
    parser.add_argument("--param-dim", type=int, default=128)
    parser.add_argument("--sample-count", type=int, default=96)
    parser.add_argument("--null-scale", type=float, default=0.01)
    parser.add_argument("--null-sensitivity", type=float, default=0.15)
    parser.add_argument("--projection-cost-multiplier", type=float, default=1.0)
    parser.add_argument("--expensive-projection-cost-multiplier", type=float, default=16.0)
    parser.add_argument("--max-active-rank-ratio", type=float, default=0.25)
    parser.add_argument("--max-null-energy-fraction", type=float, default=0.02)
    parser.add_argument("--max-gradient-relative-error", type=float, default=0.05)
    parser.add_argument("--min-descent-cosine", type=float, default=0.995)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.75)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def orthonormal_matrix(
    rng: np.random.Generator,
    rows: int,
    cols: int,
) -> np.ndarray:
    matrix = rng.normal(size=(rows, cols))
    q, _ = np.linalg.qr(matrix, mode="reduced")
    return q[:, :cols]


def make_state_basis(
    rng: np.random.Generator,
    state_dim: int,
    active_rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    q = orthonormal_matrix(rng, state_dim, state_dim)
    return q[:, :active_rank], q[:, active_rank:]


def make_misaligned_basis(
    rng: np.random.Generator,
    active_basis: np.ndarray,
    null_basis: np.ndarray,
    angle: float,
) -> np.ndarray:
    null_rank = active_basis.shape[1]
    null_mix = null_basis @ orthonormal_matrix(rng, null_basis.shape[1], null_rank)
    mixed = math.cos(angle) * active_basis + math.sin(angle) * null_mix
    q, _ = np.linalg.qr(mixed, mode="reduced")
    return q[:, :null_rank]


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    numerator = float(np.vdot(left, right).real)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 1.0 if np.linalg.norm(left - right) == 0.0 else 0.0
    return numerator / denominator


def condition_config(args: argparse.Namespace, condition: str) -> dict[str, float | int]:
    active_rank = args.active_rank
    null_scale = args.null_scale
    null_sensitivity = args.null_sensitivity
    projection_cost = args.projection_cost_multiplier
    misalignment_angle = 0.0

    if condition == "full_rank_fail":
        active_rank = int(round(args.state_dim * 0.875))
    elif condition == "null_cotangent_fail":
        null_scale = 0.80
    elif condition == "basis_misalignment_fail":
        misalignment_angle = 0.75
    elif condition == "projection_cost_fail":
        projection_cost = args.expensive_projection_cost_multiplier
    elif condition == "null_pullback_fail":
        null_scale = 0.03
        null_sensitivity = 25.0

    return {
        "active_rank": active_rank,
        "null_scale": null_scale,
        "null_sensitivity": null_sensitivity,
        "projection_cost": projection_cost,
        "misalignment_angle": misalignment_angle,
    }


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    cfg = condition_config(args, condition)
    state_dim = args.state_dim
    active_rank = int(cfg["active_rank"])
    active_basis, null_basis = make_state_basis(rng, state_dim, active_rank)
    if float(cfg["misalignment_angle"]) > 0.0:
        projection_basis = make_misaligned_basis(
            rng,
            active_basis,
            null_basis,
            float(cfg["misalignment_angle"]),
        )
    else:
        projection_basis = active_basis

    active_coeff = rng.normal(size=(args.sample_count, active_rank))
    null_coeff = rng.normal(size=(args.sample_count, state_dim - active_rank))
    cotangent = (
        active_coeff @ active_basis.T
        + float(cfg["null_scale"]) * null_coeff @ null_basis.T
    )

    active_jacobian = rng.normal(size=(active_rank, args.param_dim))
    null_jacobian = rng.normal(size=(state_dim - active_rank, args.param_dim))
    dense_jacobian = (
        active_basis @ active_jacobian
        + float(cfg["null_sensitivity"]) * null_basis @ null_jacobian
    )

    dense_gradient = cotangent @ dense_jacobian
    projected_coeff = cotangent @ projection_basis
    reduced_jacobian = projection_basis.T @ dense_jacobian
    reduced_gradient = projected_coeff @ reduced_jacobian

    active_projection = cotangent @ active_basis @ active_basis.T
    null_residual = cotangent - active_projection
    null_energy = float(np.sum(null_residual * null_residual))
    total_energy = float(np.sum(cotangent * cotangent))
    null_fraction = null_energy / max(total_energy, 1e-12)

    gradient_delta = reduced_gradient - dense_gradient
    gradient_error = float(np.linalg.norm(gradient_delta) / max(np.linalg.norm(dense_gradient), 1e-12))
    max_abs_error = float(np.max(np.abs(gradient_delta)))
    descent_cosine = cosine_similarity(dense_gradient, reduced_gradient)
    singular_values = np.linalg.svd(active_basis.T @ projection_basis, compute_uv=False)
    basis_alignment_min = float(np.min(singular_values)) if len(singular_values) else 1.0

    null_pullback = null_basis.T @ dense_jacobian
    active_pullback = active_basis.T @ dense_jacobian
    null_pullback_gain = float(
        np.linalg.norm(null_pullback) / max(np.linalg.norm(active_pullback), 1e-12)
    )

    dense_work = float(args.sample_count * state_dim * args.param_dim)
    projection_work = float(args.sample_count * state_dim * active_rank * float(cfg["projection_cost"]))
    reduced_work = float(args.sample_count * active_rank * args.param_dim)
    price = (projection_work + reduced_work) / max(dense_work, 1e-12)
    speedup = 1.0 / max(price, 1e-12)
    active_ratio = active_rank / state_dim

    failed: list[str] = []
    if not np.isfinite(
        [
            null_fraction,
            gradient_error,
            max_abs_error,
            descent_cosine,
            basis_alignment_min,
            null_pullback_gain,
            price,
            speedup,
        ]
    ).all():
        failed.append("non_finite_metric")
    if active_ratio > args.max_active_rank_ratio:
        failed.append("active_rank_ratio")
    if null_fraction > args.max_null_energy_fraction:
        failed.append("null_energy_fraction")
    if gradient_error > args.max_gradient_relative_error:
        failed.append("gradient_relative_error")
    if descent_cosine < args.min_descent_cosine:
        failed.append("descent_cosine")
    if price > args.max_priced_work_ratio:
        failed.append("priced_work")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed),
        state_dim=state_dim,
        active_rank=active_rank,
        param_dim=args.param_dim,
        sample_count=args.sample_count,
        active_rank_ratio=active_ratio,
        null_energy_fraction=null_fraction,
        gradient_relative_error=gradient_error,
        max_abs_gradient_error=max_abs_error,
        descent_cosine=descent_cosine,
        basis_alignment_min=basis_alignment_min,
        null_pullback_gain=null_pullback_gain,
        dense_work_units=dense_work,
        projection_work_units=projection_work,
        reduced_work_units=reduced_work,
        priced_work_ratio=price,
        projected_speedup=speedup,
    )


NUMERIC_FIELDS = [
    "state_dim",
    "active_rank",
    "param_dim",
    "sample_count",
    "active_rank_ratio",
    "null_energy_fraction",
    "gradient_relative_error",
    "max_abs_gradient_error",
    "descent_cosine",
    "basis_alignment_min",
    "null_pullback_gain",
    "dense_work_units",
    "projection_work_units",
    "reduced_work_units",
    "priced_work_ratio",
    "projected_speedup",
]


def aggregate(results: list[TrialResult]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition in CONDITIONS:
        subset = [result for result in results if result.condition == condition]
        summary: dict[str, Any] = {
            "count": len(subset),
            "passed_count": sum(result.passed for result in subset),
            "identifying_correct_count": sum(result.identifying_correct for result in subset),
            "fail_reasons": sorted({result.fail_reason for result in subset if result.fail_reason}),
        }
        for field in NUMERIC_FIELDS:
            values = [float(getattr(result, field)) for result in subset]
            summary[f"{field}_mean"] = float(np.mean(values))
            summary[f"{field}_max"] = float(np.max(values))
            summary[f"{field}_min"] = float(np.min(values))
        output[condition] = summary
    return output


def required_checks(results: list[TrialResult], aggregates: dict[str, Any], args: argparse.Namespace) -> dict[str, bool]:
    by_condition = {
        condition: [result for result in results if result.condition == condition]
        for condition in CONDITIONS
    }
    return {
        "all_metrics_finite": all(
            math.isfinite(float(value))
            for result in results
            for value in asdict(result).values()
            if isinstance(value, (int, float, np.floating))
        ),
        "all_conditions_identified": all(result.identifying_correct for result in results),
        "positive_condition_all_pass": all(result.passed for result in by_condition["positive"]),
        "positive_condition_projection_preserves_gradient_and_price": (
            aggregates["positive"]["active_rank_ratio_max"] <= args.max_active_rank_ratio
            and aggregates["positive"]["null_energy_fraction_max"] <= args.max_null_energy_fraction
            and aggregates["positive"]["gradient_relative_error_max"] <= args.max_gradient_relative_error
            and aggregates["positive"]["descent_cosine_min"] >= args.min_descent_cosine
            and aggregates["positive"]["priced_work_ratio_max"] <= args.max_priced_work_ratio
        ),
        "full_rank_control_all_fail_price_or_rank": all(
            (not result.passed)
            and (
                "active_rank_ratio" in result.fail_reason
                or "priced_work" in result.fail_reason
            )
            for result in by_condition["full_rank_fail"]
        ),
        "null_cotangent_control_all_fail_quality": all(
            (not result.passed)
            and (
                "null_energy_fraction" in result.fail_reason
                or "gradient_relative_error" in result.fail_reason
                or "descent_cosine" in result.fail_reason
            )
            for result in by_condition["null_cotangent_fail"]
        ),
        "basis_misalignment_control_all_fail_quality": all(
            (not result.passed)
            and (
                "gradient_relative_error" in result.fail_reason
                or "descent_cosine" in result.fail_reason
            )
            for result in by_condition["basis_misalignment_fail"]
        ),
        "projection_cost_control_all_fail_price": all(
            (not result.passed) and "priced_work" in result.fail_reason
            for result in by_condition["projection_cost_fail"]
        ),
        "null_pullback_control_all_fail_gradient_despite_small_null": all(
            (not result.passed)
            and "gradient_relative_error" in result.fail_reason
            and result.null_energy_fraction <= args.max_null_energy_fraction
            for result in by_condition["null_pullback_fail"]
        ),
    }


def main() -> int:
    args = parse_args()
    results = [
        run_trial(args, condition, args.seed + index)
        for index in range(args.seed_count)
        for condition in CONDITIONS
    ]
    aggregates = aggregate(results)
    checks = required_checks(results, aggregates, args)
    decision = "PASS_R502_S27_QSGE_GATE" if all(checks.values()) else "FAIL_R502_S27_QSGE_GATE"
    payload = {
        "round": "R502",
        "canonical_id": "R502-S27",
        "aspect": "Speed",
        "finding": "Quotient Space Gradient Embedding",
        "decision": decision,
        "disposition": "synthetic_positive_control_identified_not_training_speed_claim",
        "args": vars(args) | {"output": str(args.output) if args.output else None},
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU quotient-space gradient embedding gate only; not "
            "heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_S27_QSGE_GATE] "
        f"active=1 decision={decision} "
        f"disposition={payload['disposition']} "
        f"positive_active_rank_ratio_max={positive['active_rank_ratio_max']:.12f} "
        f"positive_null_energy_fraction_max={positive['null_energy_fraction_max']:.12f} "
        f"positive_gradient_relative_error_max={positive['gradient_relative_error_max']:.12f} "
        f"positive_descent_cosine_min={positive['descent_cosine_min']:.12f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.12f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.12f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
