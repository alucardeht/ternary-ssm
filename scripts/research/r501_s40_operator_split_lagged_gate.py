#!/usr/bin/env python3
"""R501 S40 operator-split lagged dense corrections gate.

This CPU diagnostic implements the R501 Speed Math 7 finding:
"Operator-Split Lagged Dense Corrections".

The gate splits the recurrent cotangent operator into a block-diagonal part D
and an off-block residual R. D is applied every timestep. The dense R correction
is estimated from the previous block-diagonal pass and sampled every M steps,
then interpolated as a lagged correction.

The finding survives only when R does not carry most credit assignment, lagged
correction residual debt is bounded, and the priced route
C_diag + (1/M) C_dense + C_residual stays below dense BPTT work.

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


SCHEMA = "r501_s40_operator_split_lagged_gate.v1"
CONDITIONS = (
    "positive",
    "residual_dominant_fail",
    "high_frequency_residual_fail",
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
    lag_interval: int
    residual_operator_fraction: float
    correction_variation_ratio: float
    mean_relative_solution_error: float
    max_relative_solution_error: float
    residual_debt_ratio: float
    priced_work_ratio: float
    projected_speedup: float
    price_components: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50140)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--segment-length", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--lag-interval", type=int, default=8)
    parser.add_argument("--pricing-fail-lag-interval", type=int, default=1)
    parser.add_argument("--max-residual-operator-fraction", type=float, default=0.35)
    parser.add_argument("--max-residual-debt-ratio", type=float, default=4.0e-2)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.90)
    parser.add_argument("--audit-cost-ratio", type=float, default=0.040)
    parser.add_argument("--residual-settlement-multiplier", type=float, default=1.35)
    return parser.parse_args()


def orthogonal_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def split_transition(
    rng: np.random.Generator,
    state_dim: int,
    block_size: int,
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    d = np.zeros((state_dim, state_dim), dtype=np.float64)
    for start in range(0, state_dim, block_size):
        end = min(state_dim, start + block_size)
        width = end - start
        q = orthogonal_matrix(rng, width)
        eigs = np.linspace(0.08, 0.52, width)
        d[start:end, start:end] = q @ np.diag(eigs) @ q.T

    if condition == "residual_dominant_fail":
        scale = 0.105
    else:
        scale = 0.018
    r = rng.normal(scale=scale, size=(state_dim, state_dim))
    for start in range(0, state_dim, block_size):
        end = min(state_dim, start + block_size)
        r[start:end, start:end] = 0.0
    spectral = max(abs(np.linalg.eigvals(d + r)))
    if spectral >= 0.92:
        r *= 0.92 / float(spectral)
    return d, r.astype(np.float64)


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
        np.sin(4.0 * math.pi * t + 0.4),
    ]
    if condition == "high_frequency_residual_fail":
        columns.extend(
            [
                np.sin(27.0 * math.pi * t),
                np.cos(35.0 * math.pi * t),
                np.sign(np.sin(41.0 * math.pi * t)),
            ]
        )
    x = np.stack(columns, axis=1)
    w = rng.normal(scale=0.060, size=(x.shape[1], state_dim))
    g = x @ w
    g += 0.003 * rng.normal(size=g.shape)
    return g.astype(np.float64)


def recurrent_adjoint(g: np.ndarray, a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(g, dtype=np.float64)
    carry = np.zeros(g.shape[1], dtype=np.float64)
    for t in range(g.shape[0] - 1, -1, -1):
        carry = g[t] + carry @ a
        out[t] = carry
    return out


def sample_interpolate_correction(correction: np.ndarray, lag_interval: int) -> np.ndarray:
    segment_length, state_dim = correction.shape
    out = np.zeros_like(correction)
    points = list(range(0, segment_length, lag_interval))
    if points[-1] != segment_length - 1:
        points.append(segment_length - 1)
    for left_idx, right_idx in zip(points[:-1], points[1:]):
        left = correction[left_idx]
        right = correction[right_idx]
        span = max(1, right_idx - left_idx)
        for t in range(left_idx, right_idx + 1):
            alpha = (t - left_idx) / float(span)
            out[t] = (1.0 - alpha) * left + alpha * right
    return out


def lagged_split_adjoint(
    g: np.ndarray,
    d: np.ndarray,
    r: np.ndarray,
    lag_interval: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = recurrent_adjoint(g, d)
    next_base = np.vstack([base[1:], np.zeros((1, base.shape[1]), dtype=np.float64)])
    correction = next_base @ r
    correction_hat = sample_interpolate_correction(correction, lag_interval)
    out = np.zeros_like(g, dtype=np.float64)
    carry = np.zeros(g.shape[1], dtype=np.float64)
    for t in range(g.shape[0] - 1, -1, -1):
        carry = g[t] + carry @ d + correction_hat[t]
        out[t] = carry
    return out, correction, correction_hat


def relative_rows(error: np.ndarray, exact: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm(error, axis=1)
    denominator = np.maximum(np.linalg.norm(exact, axis=1), 1.0e-30)
    return numerator / denominator


def correction_variation(correction: np.ndarray) -> float:
    diffs = np.diff(correction, axis=0)
    numerator = float(np.mean(np.linalg.norm(diffs, axis=1)))
    denominator = max(float(np.mean(np.linalg.norm(correction, axis=1))), 1.0e-30)
    return numerator / denominator


def price_route(
    args: argparse.Namespace,
    lag_interval: int,
    residual_debt_ratio: float,
) -> dict[str, float]:
    blocks = math.ceil(args.state_dim / args.block_size)
    diag_ratio = float(blocks * args.block_size * args.block_size) / float(
        args.state_dim * args.state_dim
    )
    return {
        "block_diagonal_apply": diag_ratio,
        "lagged_dense_correction": 1.0 / float(lag_interval),
        "residual_audit": float(args.audit_cost_ratio),
        "residual_settlement": float(args.residual_settlement_multiplier * residual_debt_ratio),
    }


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    lag_interval = (
        args.pricing_fail_lag_interval if condition == "pricing_fail" else args.lag_interval
    )
    d, r = split_transition(rng, args.state_dim, args.block_size, condition)
    g = forcing_sequence(rng, args.segment_length, args.state_dim, condition)
    exact = recurrent_adjoint(g, d + r)
    approx, correction, correction_hat = lagged_split_adjoint(g, d, r, lag_interval)
    error = approx - exact
    relative = relative_rows(error, exact)
    residual_debt = float(
        np.mean(np.linalg.norm(error, axis=1))
        / max(np.mean(np.linalg.norm(exact, axis=1)), 1.0e-30)
    )
    residual_fraction = float(
        np.linalg.norm(r) / max(np.linalg.norm(d + r), 1.0e-30)
    )
    correction_var = correction_variation(correction)
    price_components = price_route(args, lag_interval, residual_debt)
    priced_work_ratio = float(sum(price_components.values()))
    projected_speedup = 1.0 / priced_work_ratio if priced_work_ratio > 0.0 else math.inf

    residual_operator_pass = residual_fraction <= args.max_residual_operator_fraction
    debt_pass = residual_debt <= args.max_residual_debt_ratio
    price_pass = priced_work_ratio <= args.max_priced_work_ratio
    passed = bool(residual_operator_pass and debt_pass and price_pass)
    expected_pass = condition == "positive"
    identifying_correct = passed == expected_pass

    fail_reason = None
    if not passed:
        if not residual_operator_pass:
            fail_reason = "residual_operator_carries_majority_credit"
        elif not debt_pass:
            fail_reason = "lagged_dense_correction_residual_debt_exceeds_budget"
        elif not price_pass:
            fail_reason = "operator_split_not_below_dense_budget"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=identifying_correct,
        fail_reason=fail_reason,
        segment_length=args.segment_length,
        state_dim=args.state_dim,
        block_size=args.block_size,
        lag_interval=lag_interval,
        residual_operator_fraction=residual_fraction,
        correction_variation_ratio=correction_var,
        mean_relative_solution_error=float(np.mean(relative)),
        max_relative_solution_error=float(np.max(relative)),
        residual_debt_ratio=residual_debt,
        priced_work_ratio=priced_work_ratio,
        projected_speedup=float(projected_speedup),
        price_components=price_components,
    )


def finite_trial(trial: TrialResult) -> bool:
    values = (
        trial.residual_operator_fraction,
        trial.correction_variation_ratio,
        trial.mean_relative_solution_error,
        trial.max_relative_solution_error,
        trial.residual_debt_ratio,
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
            "lag_interval_mean": float(np.mean([row.lag_interval for row in rows])),
            "residual_operator_fraction_mean": float(
                np.mean([row.residual_operator_fraction for row in rows])
            ),
            "correction_variation_ratio_mean": float(
                np.mean([row.correction_variation_ratio for row in rows])
            ),
            "mean_relative_solution_error_mean": float(
                np.mean([row.mean_relative_solution_error for row in rows])
            ),
            "max_relative_solution_error_max": float(
                np.max([row.max_relative_solution_error for row in rows])
            ),
            "residual_debt_ratio_mean": float(
                np.mean([row.residual_debt_ratio for row in rows])
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
    if args.state_dim < 2:
        raise ValueError("--state-dim must be at least 2")
    if args.block_size < 1 or args.state_dim % args.block_size != 0:
        raise ValueError("--block-size must evenly divide --state-dim")
    if args.lag_interval < 1 or args.pricing_fail_lag_interval < 1:
        raise ValueError("lag intervals must be at least 1")

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
        "positive_condition_residual_debt_below_budget": all(
            trial.residual_debt_ratio <= args.max_residual_debt_ratio
            for trial in trials
            if trial.condition == "positive"
        ),
        "residual_dominant_control_all_fail_operator": all(
            (not trial.passed)
            and trial.fail_reason == "residual_operator_carries_majority_credit"
            for trial in trials
            if trial.condition == "residual_dominant_fail"
        ),
        "high_frequency_control_all_fail_debt": all(
            (not trial.passed)
            and trial.fail_reason == "lagged_dense_correction_residual_debt_exceeds_budget"
            for trial in trials
            if trial.condition == "high_frequency_residual_fail"
        ),
        "pricing_control_all_fail_price": all(
            (not trial.passed)
            and trial.fail_reason == "operator_split_not_below_dense_budget"
            for trial in trials
            if trial.condition == "pricing_fail"
        ),
        "all_conditions_identified": all(trial.identifying_correct for trial in trials),
    }
    decision = (
        "PASS_R501_S40_OPERATOR_SPLIT_LAGGED_GATE"
        if all(required_passes.values())
        else "FAIL_R501_S40_OPERATOR_SPLIT_LAGGED_GATE"
    )
    disposition = (
        "synthetic_positive_control_identified_not_training_speed_claim"
        if decision.startswith("PASS_")
        else "operator_split_lagged_gate_failed_or_non_identifying"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "CPU synthetic operator-split lagged dense correction gate. It "
            "checks residual-operator credit share, lagged correction residual "
            "debt, and priced block-diagonal plus delayed dense work. It is not "
            "heldout CE progress or GPU training-speed evidence."
        ),
        "finding": {
            "round": "R501",
            "aspect": "Speed",
            "id": "R501-S40",
            "title": "Operator-Split Lagged Dense Corrections",
        },
        "thresholds": {
            "max_residual_operator_fraction": args.max_residual_operator_fraction,
            "max_residual_debt_ratio": args.max_residual_debt_ratio,
            "max_priced_work_ratio": args.max_priced_work_ratio,
            "required_projected_speedup": 1.0 / args.max_priced_work_ratio,
        },
        "shape": {
            "seed": args.seed,
            "seed_count": args.seed_count,
            "segment_length": args.segment_length,
            "state_dim": args.state_dim,
            "block_size": args.block_size,
            "lag_interval": args.lag_interval,
            "pricing_fail_lag_interval": args.pricing_fail_lag_interval,
        },
        "required_passes": required_passes,
        "aggregate": aggregate(trials),
        "trials": [asdict(trial) for trial in trials],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    positive = result["aggregate"]["positive"]
    print(
        "[R501_S40_OPERATOR_SPLIT_LAGGED_GATE] "
        f"active=1 decision={decision} disposition={disposition} "
        f"positive_residual_operator_fraction_mean={positive['residual_operator_fraction_mean']:.6f} "
        f"positive_residual_debt_ratio_mean={positive['residual_debt_ratio_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
