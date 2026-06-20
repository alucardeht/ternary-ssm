#!/usr/bin/env python3
"""R501 S38 temporal multigrid CE gradient coarsening gate.

This CPU diagnostic implements the R501 Speed Math 5 finding:
"Temporal Multigrid CE Gradient Coarsening".

The gate groups timesteps into blocks, computes exact cotangent values only at
block boundaries through a coarsened block recurrence, and linearly interpolates
the internal cotangent trajectory. The finding survives only when:

1. block-boundary cotangents match the exact dense recurrence;
2. interpolation residual debt stays below a fixed budget;
3. the K^2 second-difference bound is not violated; and
4. the priced multigrid route is below 0.50x dense backward work.

Controls exercise the declared kill criteria: high-frequency gradient impulses,
smoothness violations, and too-small block sizes that cannot amortize boundary
work.

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


SCHEMA = "r501_s38_temporal_multigrid_ce_gate.v1"
CONDITIONS = (
    "positive",
    "impulse_fail",
    "curvature_fail",
    "pricing_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    segment_length: int
    state_dim: int
    block_size: int
    boundary_relative_error: float
    mean_relative_interp_error: float
    max_relative_interp_error: float
    residual_debt_ratio: float
    second_difference_bound_violation_fraction: float
    max_error_to_bound_ratio: float
    priced_work_ratio: float
    projected_speedup: float
    price_components: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50138)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--segment-length", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=24)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--pricing-fail-block-size", type=int, default=2)
    parser.add_argument("--max-boundary-relative-error", type=float, default=1.0e-10)
    parser.add_argument("--max-relative-interp-error", type=float, default=8.0e-2)
    parser.add_argument("--max-residual-debt-ratio", type=float, default=3.0e-2)
    parser.add_argument("--max-bound-violation-fraction", type=float, default=0.0)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.50)
    parser.add_argument("--interp-cost-ratio", type=float, default=0.020)
    parser.add_argument("--audit-cost-ratio", type=float, default=0.045)
    parser.add_argument("--residual-settlement-multiplier", type=float, default=1.25)
    return parser.parse_args()


def orthogonal_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def transition_matrix(rng: np.random.Generator, state_dim: int) -> np.ndarray:
    q = orthogonal_matrix(rng, state_dim)
    eigs = np.linspace(0.05, 0.62, state_dim)
    return (q @ np.diag(eigs) @ q.T).astype(np.float64)


def forcing_sequence(
    rng: np.random.Generator,
    segment_length: int,
    state_dim: int,
    condition: str,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, segment_length, dtype=np.float64)
    columns = [
        np.ones_like(t),
        t,
        t * t,
        np.sin(2.0 * math.pi * t),
        np.cos(2.0 * math.pi * t),
        np.sin(4.0 * math.pi * t + 0.2),
        np.cos(6.0 * math.pi * t - 0.5),
    ]
    if condition == "curvature_fail":
        columns.extend(
            [
                np.sin(25.0 * math.pi * t),
                np.cos(31.0 * math.pi * t),
                np.sign(np.sin(37.0 * math.pi * t)),
            ]
        )
    x = np.stack(columns, axis=1)
    weights = rng.normal(scale=0.055, size=(x.shape[1], state_dim))
    g = x @ weights
    g += 0.0025 * rng.normal(size=g.shape)
    if condition == "impulse_fail":
        for offset in (segment_length // 3, 2 * segment_length // 3):
            width = 1
            impulse = rng.normal(scale=1.45, size=(2 * width + 1, state_dim))
            start = max(0, offset - width)
            end = min(segment_length, offset + width + 1)
            g[start:end] += impulse[: end - start]
    return g.astype(np.float64)


def exact_adjoint_with_terminal(g: np.ndarray, a: np.ndarray) -> np.ndarray:
    out = np.zeros((g.shape[0] + 1, g.shape[1]), dtype=np.float64)
    carry = np.zeros(g.shape[1], dtype=np.float64)
    out[-1] = carry
    for t in range(g.shape[0] - 1, -1, -1):
        carry = g[t] + carry @ a
        out[t] = carry
    return out


def coarse_boundaries(g: np.ndarray, a: np.ndarray, block_size: int) -> np.ndarray:
    segment_length, state_dim = g.shape
    boundaries = list(range(0, segment_length, block_size))
    if boundaries[-1] != segment_length:
        boundaries.append(segment_length)
    else:
        boundaries.append(segment_length)
    out = np.zeros((segment_length + 1, state_dim), dtype=np.float64)
    out[segment_length] = 0.0
    for start in reversed(boundaries[:-1]):
        end = min(segment_length, start + block_size)
        carry = np.array(out[end], copy=True)
        for t in range(end - 1, start - 1, -1):
            carry = g[t] + carry @ a
        out[start] = carry
    return out


def interpolate_from_boundaries(
    boundary_values: np.ndarray,
    segment_length: int,
    block_size: int,
) -> np.ndarray:
    out = np.zeros_like(boundary_values)
    for start in range(0, segment_length, block_size):
        end = min(segment_length, start + block_size)
        left = boundary_values[start]
        right = boundary_values[end]
        span = max(1, end - start)
        for t in range(start, end + 1):
            alpha = (t - start) / float(span)
            out[t] = (1.0 - alpha) * left + alpha * right
    return out


def relative_rows(error: np.ndarray, exact: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm(error, axis=1)
    denominator = np.maximum(np.linalg.norm(exact, axis=1), 1.0e-30)
    return numerator / denominator


def second_difference_bounds(exact: np.ndarray, block_size: int) -> np.ndarray:
    segment_length = exact.shape[0] - 1
    second = np.zeros(segment_length + 1, dtype=np.float64)
    for t in range(1, segment_length):
        second[t] = float(np.linalg.norm(exact[t + 1] - 2.0 * exact[t] + exact[t - 1]))
    bounds = np.zeros(segment_length + 1, dtype=np.float64)
    for start in range(0, segment_length, block_size):
        end = min(segment_length, start + block_size)
        local = float(np.max(second[max(1, start) : max(1, end)])) if end > start else 0.0
        bounds[start : end + 1] = (block_size * block_size) * local
    return bounds


def price_route(
    block_size: int,
    residual_debt_ratio: float,
    args: argparse.Namespace,
) -> dict[str, float]:
    return {
        "coarse_boundary_dense_work": 1.0 / float(block_size),
        "linear_interpolation": float(args.interp_cost_ratio),
        "curvature_audit": float(args.audit_cost_ratio),
        "residual_settlement": float(args.residual_settlement_multiplier * residual_debt_ratio),
    }


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    block_size = args.pricing_fail_block_size if condition == "pricing_fail" else args.block_size
    a = transition_matrix(rng, args.state_dim)
    g = forcing_sequence(rng, args.segment_length, args.state_dim, condition)
    exact = exact_adjoint_with_terminal(g, a)
    boundary = coarse_boundaries(g, a, block_size)
    approx = interpolate_from_boundaries(boundary, args.segment_length, block_size)

    boundary_mask = np.zeros(args.segment_length + 1, dtype=bool)
    boundary_mask[0 : args.segment_length + 1 : block_size] = True
    boundary_mask[args.segment_length] = True
    boundary_error = boundary[boundary_mask] - exact[boundary_mask]
    boundary_relative_error = float(
        np.linalg.norm(boundary_error.ravel())
        / max(np.linalg.norm(exact[boundary_mask].ravel()), 1.0e-30)
    )

    interp_error = approx[:-1] - exact[:-1]
    relative = relative_rows(interp_error, exact[:-1])
    mean_relative = float(np.mean(relative))
    max_relative = float(np.max(relative))
    residual_debt = float(
        np.mean(np.linalg.norm(interp_error, axis=1))
        / max(np.mean(np.linalg.norm(exact[:-1], axis=1)), 1.0e-30)
    )

    bounds = second_difference_bounds(exact, block_size)[:-1]
    error_norm = np.linalg.norm(interp_error, axis=1)
    violations = error_norm > bounds + 1.0e-12
    bound_violation_fraction = float(np.mean(violations.astype(np.float64)))
    max_error_to_bound_ratio = float(
        np.max(error_norm / np.maximum(bounds, 1.0e-30))
    )

    price_components = price_route(block_size, residual_debt, args)
    priced_work_ratio = float(sum(price_components.values()))
    projected_speedup = 1.0 / priced_work_ratio if priced_work_ratio > 0.0 else math.inf

    boundary_pass = boundary_relative_error <= args.max_boundary_relative_error
    debt_pass = residual_debt <= args.max_residual_debt_ratio
    bound_pass = bound_violation_fraction <= args.max_bound_violation_fraction
    price_pass = priced_work_ratio <= args.max_priced_work_ratio
    passed = bool(boundary_pass and debt_pass and bound_pass and price_pass)
    expected_pass = condition == "positive"
    identifying_correct = passed == expected_pass

    fail_reason = None
    if not passed:
        if not boundary_pass:
            fail_reason = "coarse_boundary_recurrence_not_exact"
        elif not bound_pass:
            fail_reason = "second_difference_bound_violated"
        elif not debt_pass:
            fail_reason = "interpolation_residual_debt_exceeds_budget"
        elif not price_pass:
            fail_reason = "temporal_multigrid_not_below_dense_budget"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=identifying_correct,
        fail_reason=fail_reason,
        segment_length=args.segment_length,
        state_dim=args.state_dim,
        block_size=block_size,
        boundary_relative_error=boundary_relative_error,
        mean_relative_interp_error=mean_relative,
        max_relative_interp_error=max_relative,
        residual_debt_ratio=residual_debt,
        second_difference_bound_violation_fraction=bound_violation_fraction,
        max_error_to_bound_ratio=max_error_to_bound_ratio,
        priced_work_ratio=priced_work_ratio,
        projected_speedup=float(projected_speedup),
        price_components=price_components,
    )


def finite_trial(trial: TrialResult) -> bool:
    values = (
        trial.boundary_relative_error,
        trial.mean_relative_interp_error,
        trial.max_relative_interp_error,
        trial.residual_debt_ratio,
        trial.second_difference_bound_violation_fraction,
        trial.max_error_to_bound_ratio,
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
            "block_size_mean": float(np.mean([row.block_size for row in rows])),
            "boundary_relative_error_max": float(
                np.max([row.boundary_relative_error for row in rows])
            ),
            "mean_relative_interp_error_mean": float(
                np.mean([row.mean_relative_interp_error for row in rows])
            ),
            "max_relative_interp_error_max": float(
                np.max([row.max_relative_interp_error for row in rows])
            ),
            "residual_debt_ratio_mean": float(
                np.mean([row.residual_debt_ratio for row in rows])
            ),
            "bound_violation_fraction_mean": float(
                np.mean([row.second_difference_bound_violation_fraction for row in rows])
            ),
            "max_error_to_bound_ratio_max": float(
                np.max([row.max_error_to_bound_ratio for row in rows])
            ),
            "priced_work_ratio_mean": float(
                np.mean([row.priced_work_ratio for row in rows])
            ),
            "projected_speedup_mean": float(
                np.mean([row.projected_speedup for row in rows])
            ),
        }
    return out


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.segment_length < 8:
        raise ValueError("--segment-length must be at least 8")
    if args.state_dim < 2:
        raise ValueError("--state-dim must be at least 2")
    if args.block_size < 2 or args.pricing_fail_block_size < 2:
        raise ValueError("block sizes must be at least 2")

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
        "positive_condition_boundaries_exact": all(
            trial.boundary_relative_error <= args.max_boundary_relative_error
            for trial in trials
            if trial.condition == "positive"
        ),
        "positive_condition_residual_debt_below_budget": all(
            trial.residual_debt_ratio <= args.max_residual_debt_ratio
            for trial in trials
            if trial.condition == "positive"
        ),
        "impulse_control_all_fail_debt": all(
            (not trial.passed)
            and trial.fail_reason == "interpolation_residual_debt_exceeds_budget"
            for trial in trials
            if trial.condition == "impulse_fail"
        ),
        "curvature_control_all_fail_debt": all(
            (not trial.passed)
            and trial.fail_reason == "interpolation_residual_debt_exceeds_budget"
            for trial in trials
            if trial.condition == "curvature_fail"
        ),
        "pricing_control_all_fail_price": all(
            (not trial.passed)
            and trial.fail_reason == "temporal_multigrid_not_below_dense_budget"
            for trial in trials
            if trial.condition == "pricing_fail"
        ),
        "all_conditions_identified": all(trial.identifying_correct for trial in trials),
    }
    decision = (
        "PASS_R501_S38_TEMPORAL_MULTIGRID_CE_GATE"
        if all(required_passes.values())
        else "FAIL_R501_S38_TEMPORAL_MULTIGRID_CE_GATE"
    )
    disposition = (
        "synthetic_positive_control_identified_not_training_speed_claim"
        if decision.startswith("PASS_")
        else "temporal_multigrid_ce_gate_failed_or_non_identifying"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "CPU synthetic temporal multigrid cotangent coarsening gate. It "
            "checks exact block-boundary recurrence, interpolation residual "
            "debt, second-difference bound violations, and priced work. It is "
            "not heldout CE progress or GPU training-speed evidence."
        ),
        "finding": {
            "round": "R501",
            "aspect": "Speed",
            "id": "R501-S38",
            "title": "Temporal Multigrid CE Gradient Coarsening",
        },
        "thresholds": {
            "max_boundary_relative_error": args.max_boundary_relative_error,
            "max_relative_interp_error": args.max_relative_interp_error,
            "max_residual_debt_ratio": args.max_residual_debt_ratio,
            "max_bound_violation_fraction": args.max_bound_violation_fraction,
            "max_priced_work_ratio": args.max_priced_work_ratio,
            "required_projected_speedup": 1.0 / args.max_priced_work_ratio,
        },
        "shape": {
            "seed": args.seed,
            "seed_count": args.seed_count,
            "segment_length": args.segment_length,
            "state_dim": args.state_dim,
            "block_size": args.block_size,
            "pricing_fail_block_size": args.pricing_fail_block_size,
        },
        "required_passes": required_passes,
        "aggregate": aggregate(trials),
        "trials": [asdict(trial) for trial in trials],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    positive = result["aggregate"]["positive"]
    print(
        "[R501_S38_TEMPORAL_MULTIGRID_CE_GATE] "
        f"active=1 decision={decision} disposition={disposition} "
        f"positive_boundary_error_max={positive['boundary_relative_error_max']:.6e} "
        f"positive_residual_debt_mean={positive['residual_debt_ratio_mean']:.6f} "
        f"positive_max_interp_error_max={positive['max_relative_interp_error_max']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
