#!/usr/bin/env python3
"""R501 S35 spectral-radius certified horizon truncation gate.

This CPU diagnostic implements the R501 Speed Math 2 finding:
"Spectral-Radius Certified Horizon Truncation (Cotangent Escrow)".

For a stable recurrent transition A with spectral radius rho < 1, the omitted
tail of recurrent cotangent credit after horizon H is bounded by a geometric
series. The gate chooses a per-position truncation horizon only when the bound
is smaller than the local ternary flip-margin slack, then checks that truncated
and exact recurrent adjoints produce identical ternary flip decisions.

The result is a synthetic gate. It is not a measured GPU training-speed claim.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s35_spectral_horizon_gate.v1"
CONDITIONS = ("positive", "rho_fail", "margin_fail", "optimistic_bound_control")


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    actual_spectral_radius: float
    bound_spectral_radius: float
    accepted_fraction: float
    mean_horizon: float
    max_horizon: int
    dense_horizon: int
    priced_work_ratio: float
    projected_speedup: float
    flip_mismatch_fraction: float
    tail_bound_violation_fraction: float
    max_actual_tail_inf: float
    max_certified_tail_bound: float
    min_certified_slack: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50135)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--segment-length", type=int, default=96)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.50)
    parser.add_argument("--min-accepted-fraction", type=float, default=0.90)
    parser.add_argument("--margin-check-overhead", type=float, default=0.035)
    parser.add_argument("--max-flip-mismatch-fraction", type=float, default=0.0)
    parser.add_argument("--max-tail-bound-violation-fraction", type=float, default=0.0)
    parser.add_argument("--tail-bound-eps", type=float, default=1.0e-12)
    return parser.parse_args()


def orthogonal_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def transition_matrix(
    rng: np.random.Generator,
    state_dim: int,
    condition: str,
) -> tuple[np.ndarray, float, float]:
    if condition == "rho_fail":
        actual_rho = 0.985
        bound_rho = actual_rho
    elif condition == "optimistic_bound_control":
        actual_rho = 0.86
        bound_rho = 0.42
    elif condition == "margin_fail":
        actual_rho = 0.52
        bound_rho = actual_rho
    else:
        actual_rho = 0.42
        bound_rho = actual_rho

    q = orthogonal_matrix(rng, state_dim)
    eigs = np.linspace(0.05, actual_rho, state_dim)
    a = q @ np.diag(eigs) @ q.T
    return a.astype(np.float64), float(actual_rho), float(bound_rho)


def forcing_sequence(
    rng: np.random.Generator,
    segment_length: int,
    state_dim: int,
    condition: str,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, segment_length, dtype=np.float64)
    columns = [
        np.ones_like(t),
        np.sin(2.0 * math.pi * t),
        np.cos(2.0 * math.pi * t),
        np.sin(4.0 * math.pi * t + 0.3),
        np.cos(6.0 * math.pi * t - 0.2),
    ]
    if condition in ("rho_fail", "optimistic_bound_control"):
        columns.extend(
            [
                np.sin(13.0 * math.pi * t),
                np.cos(17.0 * math.pi * t),
            ]
        )
    x = np.stack(columns, axis=1)
    w = rng.normal(scale=0.075, size=(x.shape[1], state_dim))
    g = x @ w
    noise = 0.006 if condition == "positive" else 0.012
    g += noise * rng.normal(size=g.shape)
    return g.astype(np.float64)


def exact_adjoint(g: np.ndarray, a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(g, dtype=np.float64)
    carry = np.zeros(g.shape[1], dtype=np.float64)
    for t in range(g.shape[0] - 1, -1, -1):
        carry = g[t] + carry @ a
        out[t] = carry
    return out


def truncated_adjoint_at(g: np.ndarray, a: np.ndarray, start: int, horizon: int) -> np.ndarray:
    carry = np.zeros(g.shape[1], dtype=np.float64)
    end = min(g.shape[0] - 1, start + horizon)
    for t in range(end, start - 1, -1):
        carry = g[t] + carry @ a
    return carry


def build_margins(
    rng: np.random.Generator,
    exact: np.ndarray,
    condition: str,
) -> np.ndarray:
    abs_exact = np.abs(exact) + 1.0e-6
    if condition == "margin_fail":
        jitter = rng.uniform(-0.008, 0.008, size=exact.shape)
        return abs_exact * (1.0 + jitter)

    flip_mask = rng.random(size=exact.shape) < 0.28
    high_margin = abs_exact * rng.uniform(1.65, 2.10, size=exact.shape) + 0.020
    low_margin = abs_exact * rng.uniform(0.48, 0.72, size=exact.shape)
    return np.where(flip_mask, low_margin, high_margin)


def geometric_tail_bound(
    future_l2_max: float,
    rho: float,
    horizon: int,
    remaining_after_start: int,
) -> float:
    tail_terms = remaining_after_start - horizon
    if tail_terms <= 0:
        return 0.0
    if rho >= 1.0:
        return math.inf
    first = rho ** (horizon + 1)
    finite_sum = (1.0 - rho ** tail_terms) / max(1.0e-30, 1.0 - rho)
    return float(future_l2_max * first * finite_sum)


def future_l2_suffix(g: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(g, axis=1)
    suffix = np.zeros_like(norms)
    current = 0.0
    for idx in range(len(norms) - 1, -1, -1):
        current = max(current, float(norms[idx]))
        suffix[idx] = current
    return suffix


def choose_horizon(
    g: np.ndarray,
    a: np.ndarray,
    margins: np.ndarray,
    start: int,
    bound_rho: float,
    suffix_l2: np.ndarray,
) -> tuple[int, np.ndarray, float, float, bool]:
    dense_horizon = g.shape[0] - 1 - start
    for horizon in range(dense_horizon + 1):
        approx = truncated_adjoint_at(g, a, start, horizon)
        future_idx = min(g.shape[0] - 1, start + horizon + 1)
        future_max = float(suffix_l2[future_idx]) if horizon < dense_horizon else 0.0
        bound = geometric_tail_bound(future_max, bound_rho, horizon, dense_horizon)
        slack = float(np.min(np.abs(np.abs(approx) - margins[start])))
        if bound + 1.0e-15 < slack:
            return horizon, approx, bound, slack, horizon < dense_horizon
    approx = truncated_adjoint_at(g, a, start, dense_horizon)
    return dense_horizon, approx, 0.0, float("inf"), False


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    a, actual_rho, bound_rho = transition_matrix(rng, args.state_dim, condition)
    g = forcing_sequence(rng, args.segment_length, args.state_dim, condition)
    exact = exact_adjoint(g, a)
    margins = build_margins(rng, exact, condition)
    suffix_l2 = future_l2_suffix(g)

    chosen_horizons: list[int] = []
    accepted_count = 0
    mismatch_count = 0
    total_decisions = args.segment_length * args.state_dim
    violation_count = 0
    max_actual_tail = 0.0
    max_bound = 0.0
    min_slack = math.inf

    for start in range(args.segment_length):
        horizon, approx, bound, slack, accepted = choose_horizon(
            g, a, margins, start, bound_rho, suffix_l2
        )
        chosen_horizons.append(horizon)
        accepted_count += int(accepted)
        exact_row = exact[start]
        actual_tail = float(np.max(np.abs(exact_row - approx)))
        max_actual_tail = max(max_actual_tail, actual_tail)
        max_bound = max(max_bound, float(bound))
        min_slack = min(min_slack, float(slack))
        if actual_tail > bound + args.tail_bound_eps:
            violation_count += 1

        exact_flips = np.abs(exact_row) >= margins[start]
        approx_flips = np.abs(approx) >= margins[start]
        mismatch_count += int(np.sum(exact_flips != approx_flips))

    dense_horizon = args.segment_length - 1
    mean_horizon = float(np.mean(np.array(chosen_horizons, dtype=np.float64)))
    accepted_fraction = accepted_count / float(args.segment_length)
    horizon_ratio = (mean_horizon + 1.0) / float(args.segment_length)
    priced_work_ratio = float(horizon_ratio + args.margin_check_overhead)
    projected_speedup = 1.0 / priced_work_ratio if priced_work_ratio > 0.0 else math.inf
    flip_mismatch_fraction = mismatch_count / float(total_decisions)
    tail_violation_fraction = violation_count / float(args.segment_length)

    accepted_pass = accepted_fraction >= args.min_accepted_fraction
    price_pass = priced_work_ratio <= args.max_priced_work_ratio
    mismatch_pass = flip_mismatch_fraction <= args.max_flip_mismatch_fraction
    bound_pass = tail_violation_fraction <= args.max_tail_bound_violation_fraction
    passed = bool(accepted_pass and price_pass and mismatch_pass and bound_pass)
    expected_pass = condition == "positive"
    identifying_correct = passed == expected_pass

    fail_reason = None
    if not passed:
        if not bound_pass:
            fail_reason = "certified_tail_bound_underestimated_actual_tail"
        elif not mismatch_pass:
            fail_reason = "truncated_flip_decisions_do_not_match_exact_adjoint"
        elif not accepted_pass:
            fail_reason = "insufficient_certified_truncation_mass"
        elif not price_pass:
            fail_reason = "priced_horizon_not_below_target_budget"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=identifying_correct,
        fail_reason=fail_reason,
        actual_spectral_radius=actual_rho,
        bound_spectral_radius=bound_rho,
        accepted_fraction=float(accepted_fraction),
        mean_horizon=mean_horizon,
        max_horizon=int(max(chosen_horizons)),
        dense_horizon=dense_horizon,
        priced_work_ratio=priced_work_ratio,
        projected_speedup=float(projected_speedup),
        flip_mismatch_fraction=float(flip_mismatch_fraction),
        tail_bound_violation_fraction=float(tail_violation_fraction),
        max_actual_tail_inf=max_actual_tail,
        max_certified_tail_bound=max_bound,
        min_certified_slack=float(min_slack),
    )


def finite_trial(trial: TrialResult) -> bool:
    values = (
        trial.actual_spectral_radius,
        trial.bound_spectral_radius,
        trial.accepted_fraction,
        trial.mean_horizon,
        trial.priced_work_ratio,
        trial.projected_speedup,
        trial.flip_mismatch_fraction,
        trial.tail_bound_violation_fraction,
        trial.max_actual_tail_inf,
        trial.max_certified_tail_bound,
        trial.min_certified_slack,
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
            "accepted_fraction_mean": float(np.mean([row.accepted_fraction for row in rows])),
            "mean_horizon_mean": float(np.mean([row.mean_horizon for row in rows])),
            "max_horizon_max": float(np.max([row.max_horizon for row in rows])),
            "priced_work_ratio_mean": float(np.mean([row.priced_work_ratio for row in rows])),
            "projected_speedup_mean": float(np.mean([row.projected_speedup for row in rows])),
            "flip_mismatch_fraction_mean": float(
                np.mean([row.flip_mismatch_fraction for row in rows])
            ),
            "tail_bound_violation_fraction_mean": float(
                np.mean([row.tail_bound_violation_fraction for row in rows])
            ),
            "max_actual_tail_inf_max": float(np.max([row.max_actual_tail_inf for row in rows])),
            "max_certified_tail_bound_max": float(
                np.max([row.max_certified_tail_bound for row in rows])
            ),
        }
    return out


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.segment_length < 4:
        raise ValueError("--segment-length must be at least 4")
    if args.state_dim < 2:
        raise ValueError("--state-dim must be at least 2")

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
        "positive_condition_no_flip_mismatch": all(
            trial.flip_mismatch_fraction <= args.max_flip_mismatch_fraction
            for trial in trials
            if trial.condition == "positive"
        ),
        "positive_condition_no_tail_bound_violation": all(
            trial.tail_bound_violation_fraction <= args.max_tail_bound_violation_fraction
            for trial in trials
            if trial.condition == "positive"
        ),
        "rho_fail_control_all_fail": all(
            (not trial.passed)
            and trial.fail_reason
            in {
                "insufficient_certified_truncation_mass",
                "priced_horizon_not_below_target_budget",
            }
            for trial in trials
            if trial.condition == "rho_fail"
        ),
        "margin_fail_control_all_fail": all(
            (not trial.passed)
            and trial.fail_reason
            in {
                "insufficient_certified_truncation_mass",
                "truncated_flip_decisions_do_not_match_exact_adjoint",
                "priced_horizon_not_below_target_budget",
            }
            for trial in trials
            if trial.condition == "margin_fail"
        ),
        "optimistic_bound_control_all_fail": all(
            (not trial.passed)
            and trial.fail_reason == "certified_tail_bound_underestimated_actual_tail"
            for trial in trials
            if trial.condition == "optimistic_bound_control"
        ),
        "all_conditions_identified": all(trial.identifying_correct for trial in trials),
    }
    decision = (
        "PASS_R501_S35_SPECTRAL_HORIZON_GATE"
        if all(required_passes.values())
        else "FAIL_R501_S35_SPECTRAL_HORIZON_GATE"
    )
    disposition = (
        "synthetic_positive_control_identified_not_training_speed_claim"
        if decision.startswith("PASS_")
        else "spectral_horizon_gate_failed_or_non_identifying"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "CPU synthetic spectral-radius horizon truncation gate. It checks "
            "certified omitted-tail bounds, exact/truncated ternary flip "
            "agreement, accepted truncation mass, and priced horizon work. It "
            "is not heldout CE progress or GPU training-speed evidence."
        ),
        "finding": {
            "round": "R501",
            "aspect": "Speed",
            "id": "R501-S35",
            "title": "Spectral-Radius Certified Horizon Truncation (Cotangent Escrow)",
        },
        "thresholds": {
            "max_priced_work_ratio": args.max_priced_work_ratio,
            "min_accepted_fraction": args.min_accepted_fraction,
            "max_flip_mismatch_fraction": args.max_flip_mismatch_fraction,
            "max_tail_bound_violation_fraction": args.max_tail_bound_violation_fraction,
            "tail_bound_eps": args.tail_bound_eps,
        },
        "shape": {
            "seed": args.seed,
            "seed_count": args.seed_count,
            "segment_length": args.segment_length,
            "state_dim": args.state_dim,
        },
        "required_passes": required_passes,
        "aggregate": aggregate(trials),
        "trials": [asdict(trial) for trial in trials],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    positive = result["aggregate"]["positive"]
    print(
        "[R501_S35_SPECTRAL_HORIZON_GATE] "
        f"active=1 decision={decision} disposition={disposition} "
        f"positive_accepted_fraction_mean={positive['accepted_fraction_mean']:.6f} "
        f"positive_mean_horizon_mean={positive['mean_horizon_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"positive_flip_mismatch_fraction_mean={positive['flip_mismatch_fraction_mean']:.6f} "
        f"positive_tail_bound_violation_fraction_mean={positive['tail_bound_violation_fraction_mean']:.6f} "
        f"seed_count={args.seed_count} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
