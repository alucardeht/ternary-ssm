#!/usr/bin/env python3
"""R501 S50 Stokes boundary-integral CE gate.

This CPU diagnostic implements the R501 Speed Paradigm 2 finding:
"Stokes' Theorem Boundary-Integral CE (STBICE)".

The gate builds a discrete 2D phase patch whose CE density is represented as
the curl of an edge field. By discrete Stokes/Green telescoping, the dense
volume sum of that curl must equal the oriented boundary circulation. The
candidate pays only boundary evaluation plus a small audit cost.

The finding survives only when the boundary integral matches the dense volume
integral, no hidden non-boundary CE source is present, and boundary plus audit
work stays below the dense volume computation.

Controls exercise the declared risks: an internal non-exact CE source cannot be
recovered from the boundary, wrong boundary orientation breaks exactness, and
small/high-audit patches can erase the boundary-pricing advantage.

This is a synthetic CPU diagnostic, not heldout CE progress, GPU training-speed
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


SCHEMA = "r501_s50_stbice_gate.v1"
CONDITIONS = (
    "positive",
    "non_exact_source_fail",
    "orientation_fail",
    "boundary_cost_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    grid_size: int
    dense_volume_integral: float
    boundary_integral: float
    relative_integral_error: float
    unrepresented_source_fraction: float
    boundary_to_volume_work_ratio: float
    audit_work_ratio: float
    priced_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50150)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--small-grid-size", type=int, default=8)
    parser.add_argument("--max-relative-error", type=float, default=1.0e-10)
    parser.add_argument("--max-unrepresented-source-fraction", type=float, default=0.02)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.50)
    parser.add_argument("--audit-work-ratio", type=float, default=0.035)
    parser.add_argument("--high-audit-work-ratio", type=float, default=0.12)
    return parser.parse_args()


def smooth_edge_fields(
    rng: np.random.Generator,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    xh = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    yh = np.linspace(0.0, 1.0, grid_size + 1, dtype=np.float64)
    xv = np.linspace(0.0, 1.0, grid_size + 1, dtype=np.float64)
    yv = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    xh_grid, yh_grid = np.meshgrid(xh, yh)
    xv_grid, yv_grid = np.meshgrid(xv, yv)
    phase = rng.uniform(-0.3, 0.3)
    h = (
        0.7 * np.sin(2.0 * math.pi * (xh_grid + phase))
        + 0.3 * np.cos(3.0 * math.pi * yh_grid)
        + 0.2 * xh_grid * (1.0 - yh_grid)
    )
    v = (
        0.6 * np.cos(2.0 * math.pi * (yv_grid - phase))
        + 0.25 * np.sin(3.0 * math.pi * xv_grid)
        + 0.2 * yv_grid * (1.0 - xv_grid)
    )
    h += rng.normal(scale=0.002, size=h.shape)
    v += rng.normal(scale=0.002, size=v.shape)
    return h.astype(np.float64), v.astype(np.float64)


def discrete_curl(horizontal: np.ndarray, vertical: np.ndarray) -> np.ndarray:
    return (vertical[:, 1:] - vertical[:, :-1]) - (horizontal[1:, :] - horizontal[:-1, :])


def boundary_integral(horizontal: np.ndarray, vertical: np.ndarray, orientation: str) -> float:
    if orientation == "correct":
        return float(
            np.sum(horizontal[0, :])
            + np.sum(vertical[:, -1])
            - np.sum(horizontal[-1, :])
            - np.sum(vertical[:, 0])
        )
    if orientation == "wrong_top":
        return float(
            np.sum(horizontal[0, :])
            + np.sum(vertical[:, -1])
            + np.sum(horizontal[-1, :])
            - np.sum(vertical[:, 0])
        )
    raise ValueError(f"unknown orientation: {orientation}")


def interior_source(grid_size: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    x, y = np.meshgrid(axis, axis)
    source = np.exp(-8.0 * (x * x + y * y))
    source *= 0.35 / max(float(np.mean(source)), 1.0e-30)
    return source


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    grid_size = args.small_grid_size if condition == "boundary_cost_fail" else args.grid_size
    horizontal, vertical = smooth_edge_fields(rng, grid_size)
    curl = discrete_curl(horizontal, vertical)
    hidden_source = np.zeros_like(curl)
    orientation = "correct"
    audit_ratio = args.audit_work_ratio
    if condition == "non_exact_source_fail":
        hidden_source = interior_source(grid_size)
    elif condition == "orientation_fail":
        orientation = "wrong_top"
    elif condition == "boundary_cost_fail":
        audit_ratio = args.high_audit_work_ratio

    dense_volume = float(np.sum(curl + hidden_source))
    boundary = boundary_integral(horizontal, vertical, orientation)
    denominator = max(abs(dense_volume), 1.0e-30)
    relative_error = abs(boundary - dense_volume) / denominator
    source_fraction = abs(float(np.sum(hidden_source))) / denominator
    boundary_work = float(4 * grid_size)
    volume_work = float(grid_size * grid_size)
    boundary_ratio = boundary_work / volume_work
    priced_work = boundary_ratio + audit_ratio
    projected_speedup = 1.0 / priced_work if priced_work > 0.0 else math.inf

    exact_pass = relative_error <= args.max_relative_error
    source_pass = source_fraction <= args.max_unrepresented_source_fraction
    price_pass = priced_work <= args.max_priced_work_ratio
    passed = bool(exact_pass and source_pass and price_pass)
    expected_pass = condition == "positive"
    identifying_correct = passed == expected_pass

    fail_reason = None
    if not passed:
        if condition == "non_exact_source_fail" and (not exact_pass or not source_pass):
            fail_reason = "ce_density_contains_non_boundary_source"
        elif condition == "orientation_fail" and not exact_pass:
            fail_reason = "boundary_orientation_breaks_stokes_identity"
        elif condition == "boundary_cost_fail" and not price_pass:
            fail_reason = "boundary_evaluation_recovers_dense_volume_cost"
        elif not exact_pass:
            fail_reason = "boundary_orientation_breaks_stokes_identity"
        elif not source_pass:
            fail_reason = "ce_density_contains_non_boundary_source"
        elif not price_pass:
            fail_reason = "boundary_evaluation_recovers_dense_volume_cost"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=identifying_correct,
        fail_reason=fail_reason,
        grid_size=grid_size,
        dense_volume_integral=dense_volume,
        boundary_integral=boundary,
        relative_integral_error=relative_error,
        unrepresented_source_fraction=source_fraction,
        boundary_to_volume_work_ratio=boundary_ratio,
        audit_work_ratio=float(audit_ratio),
        priced_work_ratio=priced_work,
        projected_speedup=float(projected_speedup),
    )


def finite_trial(trial: TrialResult) -> bool:
    values = (
        trial.dense_volume_integral,
        trial.boundary_integral,
        trial.relative_integral_error,
        trial.unrepresented_source_fraction,
        trial.boundary_to_volume_work_ratio,
        trial.audit_work_ratio,
        trial.priced_work_ratio,
        trial.projected_speedup,
    )
    return all(math.isfinite(float(value)) for value in values)


def aggregate(trials: list[TrialResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[TrialResult]] = {}
    for trial in trials:
        grouped.setdefault(trial.condition, []).append(trial)
    out: dict[str, dict[str, float]] = {}
    for condition, rows in grouped.items():
        out[condition] = {
            "count": float(len(rows)),
            "pass_count": float(sum(row.passed for row in rows)),
            "grid_size_mean": float(np.mean([row.grid_size for row in rows])),
            "relative_integral_error_mean": float(
                np.mean([row.relative_integral_error for row in rows])
            ),
            "relative_integral_error_max": float(
                np.max([row.relative_integral_error for row in rows])
            ),
            "unrepresented_source_fraction_mean": float(
                np.mean([row.unrepresented_source_fraction for row in rows])
            ),
            "boundary_to_volume_work_ratio_mean": float(
                np.mean([row.boundary_to_volume_work_ratio for row in rows])
            ),
            "audit_work_ratio_mean": float(np.mean([row.audit_work_ratio for row in rows])),
            "priced_work_ratio_mean": float(np.mean([row.priced_work_ratio for row in rows])),
            "projected_speedup_mean": float(
                np.mean([row.projected_speedup for row in rows])
            ),
        }
    return out


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.grid_size < 4 or args.small_grid_size < 4:
        raise ValueError("--grid-size and --small-grid-size must be at least 4")

    trials: list[TrialResult] = []
    for seed_offset in range(args.seed_count):
        seed = args.seed + seed_offset
        for condition in CONDITIONS:
            trials.append(run_trial(args, condition, seed))

    required_passes = {
        "all_metrics_finite": all(finite_trial(trial) for trial in trials),
        "positive_condition_all_pass": all(
            trial.passed for trial in trials if trial.condition == "positive"
        ),
        "positive_condition_boundary_equals_volume": all(
            trial.relative_integral_error <= args.max_relative_error
            for trial in trials
            if trial.condition == "positive"
        ),
        "non_exact_source_control_all_fail_source": all(
            (not trial.passed)
            and trial.fail_reason == "ce_density_contains_non_boundary_source"
            for trial in trials
            if trial.condition == "non_exact_source_fail"
        ),
        "orientation_control_all_fail_exactness": all(
            (not trial.passed)
            and trial.fail_reason == "boundary_orientation_breaks_stokes_identity"
            for trial in trials
            if trial.condition == "orientation_fail"
        ),
        "boundary_cost_control_all_fail_price": all(
            (not trial.passed)
            and trial.fail_reason == "boundary_evaluation_recovers_dense_volume_cost"
            for trial in trials
            if trial.condition == "boundary_cost_fail"
        ),
        "all_conditions_identified": all(trial.identifying_correct for trial in trials),
    }
    decision = (
        "PASS_R501_S50_STBICE_GATE"
        if all(required_passes.values())
        else "FAIL_R501_S50_STBICE_GATE"
    )
    disposition = (
        "synthetic_positive_control_identified_not_training_speed_claim"
        if decision.startswith("PASS_")
        else "stbice_gate_failed_or_non_identifying"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "CPU synthetic Stokes boundary-integral CE gate. It checks dense "
            "curl-volume versus oriented boundary circulation, hidden non-"
            "boundary CE source mass, and priced boundary plus audit work. It "
            "is not heldout CE progress or GPU training-speed evidence."
        ),
        "finding": {
            "round": "R501",
            "aspect": "Speed",
            "id": "R501-S50",
            "title": "Stokes' Theorem Boundary-Integral CE",
        },
        "thresholds": {
            "max_relative_error": args.max_relative_error,
            "max_unrepresented_source_fraction": args.max_unrepresented_source_fraction,
            "max_priced_work_ratio": args.max_priced_work_ratio,
            "audit_work_ratio": args.audit_work_ratio,
        },
        "shape": {
            "seed": args.seed,
            "seed_count": args.seed_count,
            "grid_size": args.grid_size,
            "small_grid_size": args.small_grid_size,
        },
        "required_passes": required_passes,
        "aggregate": aggregate(trials),
        "trials": [asdict(trial) for trial in trials],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    positive = result["aggregate"]["positive"]
    print(
        "[R501_S50_STBICE_GATE] "
        f"active=1 decision={decision} disposition={disposition} "
        f"positive_relative_integral_error_mean={positive['relative_integral_error_mean']:.6e} "
        f"positive_unrepresented_source_fraction_mean={positive['unrepresented_source_fraction_mean']:.6f} "
        f"positive_boundary_to_volume_work_ratio_mean={positive['boundary_to_volume_work_ratio_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
