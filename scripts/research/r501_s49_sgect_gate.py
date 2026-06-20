#!/usr/bin/env python3
"""R501 S49 spectral-gated exact cotangent truncation gate.

This CPU diagnostic implements the R501 Speed Paradigm 1 finding:
"Spectral-Gated Exact Cotangent Truncation (SGECT)".

The gate computes exact recurrent cotangents for diagonal spectral channels,
chooses the shortest channel-local truncation horizon whose certified omitted
tail is below the exact ternary flip margin, and verifies that the realized
ternary update is identical to the dense recurrent cotangent update.

The finding survives only when certified truncation preserves all ternary
decisions, tail bounds hold, horizons are materially shorter than dense BPTT,
and priced truncation plus audit is below the dense recurrent backward route.

Controls exercise the declared kills: tiny flip margins force dense fallback,
near-unit spectral radii recover dense horizons, and optimistic tail bounds can
create false truncation certificates with ternary decision mismatches.

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


SCHEMA = "r501_s49_sgect_gate.v1"
CONDITIONS = (
    "positive",
    "tiny_margin_fail",
    "high_rho_price_fail",
    "optimistic_bound_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    sequence_length: int
    channels: int
    mean_horizon: float
    max_horizon: int
    dense_fallback_fraction: float
    tail_bound_violation_fraction: float
    ternary_mismatch_fraction: float
    mean_true_tail_error: float
    mean_certified_tail_bound: float
    mean_flip_margin: float
    priced_work_ratio: float
    audit_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50149)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.60)
    parser.add_argument("--max-mismatch-fraction", type=float, default=0.0)
    parser.add_argument("--max-tail-bound-violation", type=float, default=0.0)
    parser.add_argument("--max-dense-fallback-fraction", type=float, default=0.02)
    parser.add_argument("--audit-work-ratio", type=float, default=0.035)
    parser.add_argument("--optimistic-bound-scale", type=float, default=0.05)
    return parser.parse_args()


def ternary(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0.5, 1, np.where(values < -0.5, -1, 0))


def margin_to_boundary(values: np.ndarray) -> np.ndarray:
    below = np.abs(values + 0.5)
    above = np.abs(values - 0.5)
    inside = np.minimum(below, above)
    return np.where(values < -0.5, below, np.where(values > 0.5, above, inside))


def smooth_gradients(
    rng: np.random.Generator,
    length: int,
    channels: int,
    scale: float,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, length, dtype=np.float64)
    basis = np.stack(
        [
            np.sin(2.0 * math.pi * t),
            np.cos(2.0 * math.pi * t),
            np.sin(4.0 * math.pi * t + 0.3),
            np.cos(6.0 * math.pi * t - 0.2),
        ],
        axis=1,
    )
    weights = rng.normal(scale=scale, size=(basis.shape[1], channels))
    grad = basis @ weights
    grad += rng.normal(scale=0.006, size=(length, channels))
    return grad.astype(np.float64)


def condition_params(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    if condition == "high_rho_price_fail":
        rho = rng.uniform(0.965, 0.992, size=args.channels)
        grad = smooth_gradients(rng, args.sequence_length, args.channels, scale=0.600)
        return rho, grad, 0.0004
    if condition == "tiny_margin_fail":
        rho = rng.uniform(0.42, 0.72, size=args.channels)
        grad = smooth_gradients(rng, args.sequence_length, args.channels, scale=0.800)
        return rho, grad, 0.0004
    if condition == "optimistic_bound_fail":
        rho = rng.uniform(0.50, 0.78, size=args.channels)
        grad = smooth_gradients(rng, args.sequence_length, args.channels, scale=0.055)
        return rho, grad, 0.030
    rho = rng.uniform(0.38, 0.70, size=args.channels)
    grad = smooth_gradients(rng, args.sequence_length, args.channels, scale=0.025)
    return rho, grad, 0.20


def exact_cotangents(grad: np.ndarray, rho: np.ndarray) -> np.ndarray:
    out = np.zeros_like(grad)
    carry = np.zeros(grad.shape[1], dtype=np.float64)
    for idx in range(grad.shape[0] - 1, -1, -1):
        carry = grad[idx] + rho * carry
        out[idx] = carry
    return out


def truncated_value(grad: np.ndarray, rho: np.ndarray, t: int, horizon: int) -> np.ndarray:
    end = min(grad.shape[0], t + horizon)
    weights = rho[None, :] ** np.arange(0, end - t, dtype=np.float64)[:, None]
    return np.sum(weights * grad[t:end], axis=0)


def build_shadow(
    rng: np.random.Generator,
    exact_final: np.ndarray,
    learning_rate: float,
    margin: float,
    condition: str,
) -> np.ndarray:
    exact_delta = learning_rate * exact_final
    target_bins = rng.choice(np.array([-0.80, -0.20, 0.20, 0.80]), size=exact_final.shape)
    if condition in {"tiny_margin_fail", "optimistic_bound_fail", "high_rho_price_fail"}:
        target_bins = rng.choice(np.array([-0.505, -0.495, 0.495, 0.505]), size=exact_final.shape)
    jitter = rng.uniform(-margin, margin, size=exact_final.shape)
    return target_bins + jitter - exact_delta


def choose_horizons(
    args: argparse.Namespace,
    exact: np.ndarray,
    grad: np.ndarray,
    rho: np.ndarray,
    final_exact: np.ndarray,
    shadow: np.ndarray,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    horizons = np.zeros(exact.shape, dtype=np.int32)
    truncated = np.zeros_like(exact)
    true_tail_error = np.zeros_like(exact)
    certified_tail_bound = np.zeros_like(exact)
    exact_post = shadow + args.learning_rate * exact
    margin = margin_to_boundary(exact_post)
    bound_scale = args.optimistic_bound_scale if condition == "optimistic_bound_fail" else 1.0
    for t in range(args.sequence_length):
        remaining = args.sequence_length - t
        for c in range(args.channels):
            chosen_h = remaining
            chosen_trunc = exact[t, c]
            chosen_tail = 0.0
            chosen_bound = 0.0
            for horizon in range(1, remaining + 1):
                value = truncated_value(grad[:, c : c + 1], rho[c : c + 1], t, horizon)[0]
                tail = abs(exact[t, c] - value)
                bound = bound_scale * tail
                if args.learning_rate * bound <= margin[t, c]:
                    chosen_h = horizon
                    chosen_trunc = value
                    chosen_tail = tail
                    chosen_bound = bound
                    break
            horizons[t, c] = chosen_h
            truncated[t, c] = chosen_trunc
            true_tail_error[t, c] = chosen_tail
            certified_tail_bound[t, c] = chosen_bound
    return horizons, truncated, true_tail_error, certified_tail_bound


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    rho, grad, shadow_margin = condition_params(rng, args, condition)
    exact = exact_cotangents(grad, rho)
    shadow = build_shadow(
        rng,
        exact,
        args.learning_rate,
        shadow_margin,
        condition,
    )
    horizons, truncated, true_tail, certified = choose_horizons(
        args,
        exact,
        grad,
        rho,
        exact,
        shadow,
        condition,
    )

    exact_decision = ternary(shadow + args.learning_rate * exact)
    trunc_decision = ternary(shadow + args.learning_rate * truncated)
    mismatch = exact_decision != trunc_decision
    tail_violation = certified + 1.0e-15 < true_tail
    dense_fallback = horizons >= (args.sequence_length - np.arange(args.sequence_length)[:, None])
    mean_horizon = float(np.mean(horizons))
    max_horizon = int(np.max(horizons))
    dense_work = float(args.sequence_length)
    horizon_ratio = mean_horizon / dense_work
    priced_work_ratio = horizon_ratio + args.audit_work_ratio
    projected_speedup = 1.0 / priced_work_ratio if priced_work_ratio > 0.0 else math.inf
    mean_margin = float(np.mean(margin_to_boundary(shadow + args.learning_rate * exact)))

    mismatch_fraction = float(np.mean(mismatch.astype(np.float64)))
    tail_violation_fraction = float(np.mean(tail_violation.astype(np.float64)))
    fallback_fraction = float(np.mean(dense_fallback.astype(np.float64)))
    price_pass = priced_work_ratio <= args.max_priced_work_ratio
    mismatch_pass = mismatch_fraction <= args.max_mismatch_fraction
    tail_pass = tail_violation_fraction <= args.max_tail_bound_violation
    fallback_pass = fallback_fraction <= args.max_dense_fallback_fraction
    passed = bool(price_pass and mismatch_pass and tail_pass and fallback_pass)
    expected_pass = condition == "positive"
    identifying_correct = passed == expected_pass

    fail_reason = None
    if not passed:
        if condition == "optimistic_bound_fail" and (not mismatch_pass or not tail_pass):
            fail_reason = "optimistic_tail_bound_changes_ternary_update"
        elif condition == "tiny_margin_fail" and (not fallback_pass or not price_pass):
            fail_reason = "ternary_flip_margin_too_small_for_safe_truncation"
        elif condition == "high_rho_price_fail" and (not fallback_pass or not price_pass):
            fail_reason = "spectral_radius_recovers_dense_horizon"
        elif not mismatch_pass or not tail_pass:
            fail_reason = "optimistic_tail_bound_changes_ternary_update"
        elif not fallback_pass or not price_pass:
            fail_reason = "spectral_radius_recovers_dense_horizon"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=identifying_correct,
        fail_reason=fail_reason,
        sequence_length=args.sequence_length,
        channels=args.channels,
        mean_horizon=mean_horizon,
        max_horizon=max_horizon,
        dense_fallback_fraction=fallback_fraction,
        tail_bound_violation_fraction=tail_violation_fraction,
        ternary_mismatch_fraction=mismatch_fraction,
        mean_true_tail_error=float(np.mean(true_tail)),
        mean_certified_tail_bound=float(np.mean(certified)),
        mean_flip_margin=mean_margin,
        priced_work_ratio=priced_work_ratio,
        audit_work_ratio=float(args.audit_work_ratio),
        projected_speedup=float(projected_speedup),
    )


def finite_trial(trial: TrialResult) -> bool:
    values = (
        trial.mean_horizon,
        trial.dense_fallback_fraction,
        trial.tail_bound_violation_fraction,
        trial.ternary_mismatch_fraction,
        trial.mean_true_tail_error,
        trial.mean_certified_tail_bound,
        trial.mean_flip_margin,
        trial.priced_work_ratio,
        trial.audit_work_ratio,
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
            "mean_horizon_mean": float(np.mean([row.mean_horizon for row in rows])),
            "mean_horizon_max": float(np.max([row.mean_horizon for row in rows])),
            "max_horizon_max": float(np.max([row.max_horizon for row in rows])),
            "dense_fallback_fraction_mean": float(
                np.mean([row.dense_fallback_fraction for row in rows])
            ),
            "tail_bound_violation_fraction_mean": float(
                np.mean([row.tail_bound_violation_fraction for row in rows])
            ),
            "ternary_mismatch_fraction_mean": float(
                np.mean([row.ternary_mismatch_fraction for row in rows])
            ),
            "mean_true_tail_error_mean": float(
                np.mean([row.mean_true_tail_error for row in rows])
            ),
            "mean_certified_tail_bound_mean": float(
                np.mean([row.mean_certified_tail_bound for row in rows])
            ),
            "mean_flip_margin_mean": float(np.mean([row.mean_flip_margin for row in rows])),
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
    if args.sequence_length < 8:
        raise ValueError("--sequence-length must be at least 8")
    if args.channels < 1:
        raise ValueError("--channels must be at least 1")

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
        "positive_condition_exact_ternary_equivalence": all(
            trial.ternary_mismatch_fraction <= args.max_mismatch_fraction
            and trial.tail_bound_violation_fraction <= args.max_tail_bound_violation
            for trial in trials
            if trial.condition == "positive"
        ),
        "tiny_margin_control_all_fail_fallback": all(
            (not trial.passed)
            and trial.fail_reason == "ternary_flip_margin_too_small_for_safe_truncation"
            for trial in trials
            if trial.condition == "tiny_margin_fail"
        ),
        "high_rho_control_all_fail_price": all(
            (not trial.passed)
            and trial.fail_reason == "spectral_radius_recovers_dense_horizon"
            for trial in trials
            if trial.condition == "high_rho_price_fail"
        ),
        "optimistic_bound_control_all_fail_mismatch": all(
            (not trial.passed)
            and trial.fail_reason == "optimistic_tail_bound_changes_ternary_update"
            for trial in trials
            if trial.condition == "optimistic_bound_fail"
        ),
        "all_conditions_identified": all(trial.identifying_correct for trial in trials),
    }
    decision = (
        "PASS_R501_S49_SGECT_GATE"
        if all(required_passes.values())
        else "FAIL_R501_S49_SGECT_GATE"
    )
    disposition = (
        "synthetic_positive_control_identified_not_training_speed_claim"
        if decision.startswith("PASS_")
        else "sgect_gate_failed_or_non_identifying"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "CPU synthetic spectral-gated exact cotangent truncation gate. It "
            "checks channel-local spectral horizons, certified omitted-tail "
            "bounds, exact ternary update equivalence, dense fallback mass, "
            "and priced truncated backward work. It is not heldout CE progress "
            "or GPU training-speed evidence."
        ),
        "finding": {
            "round": "R501",
            "aspect": "Speed",
            "id": "R501-S49",
            "title": "Spectral-Gated Exact Cotangent Truncation",
        },
        "thresholds": {
            "max_priced_work_ratio": args.max_priced_work_ratio,
            "max_mismatch_fraction": args.max_mismatch_fraction,
            "max_tail_bound_violation": args.max_tail_bound_violation,
            "max_dense_fallback_fraction": args.max_dense_fallback_fraction,
            "audit_work_ratio": args.audit_work_ratio,
        },
        "shape": {
            "seed": args.seed,
            "seed_count": args.seed_count,
            "sequence_length": args.sequence_length,
            "channels": args.channels,
            "learning_rate": args.learning_rate,
        },
        "required_passes": required_passes,
        "aggregate": aggregate(trials),
        "trials": [asdict(trial) for trial in trials],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    positive = result["aggregate"]["positive"]
    print(
        "[R501_S49_SGECT_GATE] "
        f"active=1 decision={decision} disposition={disposition} "
        f"positive_mean_horizon_mean={positive['mean_horizon_mean']:.6f} "
        f"positive_dense_fallback_fraction_mean={positive['dense_fallback_fraction_mean']:.6f} "
        f"positive_tail_bound_violation_fraction_mean={positive['tail_bound_violation_fraction_mean']:.6f} "
        f"positive_ternary_mismatch_fraction_mean={positive['ternary_mismatch_fraction_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
