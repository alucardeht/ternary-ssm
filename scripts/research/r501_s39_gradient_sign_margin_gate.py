#!/usr/bin/env python3
"""R501 S39 contrastive gradient-sign margin certificate gate.

This CPU diagnostic implements the R501 Speed Math 6 finding:
"Contrastive Gradient-Sign Margin Certificate".

Ternary updates only need to know whether an accumulated latent gradient crosses
the negative or positive margin. The gate replaces dense cotangent evaluation
with a cheap interval certificate. If the interval lies wholly in one decision
region, the ternary update is certified; otherwise the gate prices an exact
dense fallback for that item.

The finding survives only when certified decisions match the exact dense
gradient signs, ambiguous fallback mass is low, and interval plus fallback cost
is below the dense baseline.

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


SCHEMA = "r501_s39_gradient_sign_margin_gate.v1"
CONDITIONS = (
    "positive",
    "loose_bounds_fail",
    "near_margin_fail",
    "optimistic_false_certificate_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    sample_count: int
    threshold: float
    certified_fraction: float
    dense_fallback_fraction: float
    certified_mismatch_fraction: float
    final_decision_mismatch_fraction: float
    priced_work_ratio: float
    projected_speedup: float
    mean_interval_width: float
    mean_margin_distance: float
    exact_positive_fraction: float
    exact_negative_fraction: float
    exact_zero_fraction: float
    price_components: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50139)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.45)
    parser.add_argument("--max-dense-fallback-fraction", type=float, default=0.30)
    parser.add_argument("--max-certified-mismatch-fraction", type=float, default=0.0)
    parser.add_argument("--interval-cost-ratio", type=float, default=0.070)
    parser.add_argument("--certificate-audit-ratio", type=float, default=0.025)
    return parser.parse_args()


def exact_gradients(
    rng: np.random.Generator,
    condition: str,
    sample_count: int,
    threshold: float,
) -> np.ndarray:
    if condition == "near_margin_fail":
        signs = rng.choice(np.array([-1.0, 1.0]), size=sample_count)
        near = threshold + rng.normal(scale=0.055, size=sample_count)
        zeros = rng.random(sample_count) < 0.28
        near[zeros] = threshold + rng.normal(scale=0.035, size=int(np.sum(zeros)))
        return signs * near
    if condition == "optimistic_false_certificate_fail":
        signs = rng.choice(np.array([-1.0, 1.0]), size=sample_count)
        near = threshold - 0.045 + rng.normal(scale=0.075, size=sample_count)
        return signs * near

    regions = rng.choice(
        np.array(["negative", "zero", "positive"]),
        size=sample_count,
        p=np.array([0.29, 0.42, 0.29]),
    )
    values = np.zeros(sample_count, dtype=np.float64)
    neg = regions == "negative"
    pos = regions == "positive"
    zero = regions == "zero"
    values[neg] = -(
        threshold
        + 0.42
        + rng.gamma(shape=2.0, scale=0.22, size=int(np.sum(neg)))
    )
    values[pos] = (
        threshold
        + 0.42
        + rng.gamma(shape=2.0, scale=0.22, size=int(np.sum(pos)))
    )
    values[zero] = rng.normal(scale=0.28, size=int(np.sum(zero)))
    return values


def interval_estimate(
    rng: np.random.Generator,
    exact: np.ndarray,
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    if condition == "loose_bounds_fail":
        center = exact + rng.normal(scale=0.025, size=exact.shape)
        radius = rng.uniform(1.15, 1.75, size=exact.shape)
    elif condition == "near_margin_fail":
        center = exact + rng.normal(scale=0.020, size=exact.shape)
        radius = rng.uniform(0.10, 0.18, size=exact.shape)
    elif condition == "optimistic_false_certificate_fail":
        center = exact + np.sign(exact + 1.0e-9) * rng.uniform(0.14, 0.22, size=exact.shape)
        radius = rng.uniform(0.010, 0.018, size=exact.shape)
    else:
        center = exact + rng.normal(scale=0.018, size=exact.shape)
        radius = rng.uniform(0.035, 0.070, size=exact.shape)
    lower = center - radius
    upper = center + radius
    return lower.astype(np.float64), upper.astype(np.float64)


def exact_decision(values: np.ndarray, threshold: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.int8)
    out[values >= threshold] = 1
    out[values <= -threshold] = -1
    return out


def certified_decision(lower: np.ndarray, upper: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    decision = np.zeros_like(lower, dtype=np.int8)
    certified = np.zeros_like(lower, dtype=bool)
    negative = upper < -threshold
    positive = lower > threshold
    zero = (lower > -threshold) & (upper < threshold)
    decision[negative] = -1
    decision[positive] = 1
    decision[zero] = 0
    certified[negative | positive | zero] = True
    return decision, certified


def margin_distance(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.minimum(np.abs(values - threshold), np.abs(values + threshold))


def price_route(args: argparse.Namespace, fallback_fraction: float) -> dict[str, float]:
    return {
        "interval_graph": float(args.interval_cost_ratio),
        "certificate_audit": float(args.certificate_audit_ratio),
        "dense_fallback": float(fallback_fraction),
    }


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    exact = exact_gradients(rng, condition, args.sample_count, args.threshold)
    lower, upper = interval_estimate(rng, exact, condition)
    exact_update = exact_decision(exact, args.threshold)
    cert_update, certified = certified_decision(lower, upper, args.threshold)
    final_update = np.where(certified, cert_update, exact_update)

    certified_count = int(np.sum(certified))
    certified_fraction = certified_count / float(args.sample_count)
    fallback_fraction = 1.0 - certified_fraction
    if certified_count > 0:
        certified_mismatch = float(
            np.mean((cert_update[certified] != exact_update[certified]).astype(np.float64))
        )
    else:
        certified_mismatch = 0.0
    final_mismatch = float(np.mean((final_update != exact_update).astype(np.float64)))
    price_components = price_route(args, fallback_fraction)
    priced_work_ratio = float(sum(price_components.values()))
    projected_speedup = 1.0 / priced_work_ratio if priced_work_ratio > 0.0 else math.inf
    exact_positive = float(np.mean((exact_update == 1).astype(np.float64)))
    exact_negative = float(np.mean((exact_update == -1).astype(np.float64)))
    exact_zero = float(np.mean((exact_update == 0).astype(np.float64)))

    mismatch_pass = certified_mismatch <= args.max_certified_mismatch_fraction
    fallback_pass = fallback_fraction <= args.max_dense_fallback_fraction
    price_pass = priced_work_ratio <= args.max_priced_work_ratio
    final_pass = final_mismatch <= args.max_certified_mismatch_fraction
    passed = bool(mismatch_pass and fallback_pass and price_pass and final_pass)
    expected_pass = condition == "positive"
    identifying_correct = passed == expected_pass

    fail_reason = None
    if not passed:
        if not mismatch_pass or not final_pass:
            fail_reason = "certified_interval_decision_mismatches_exact_gradient_sign"
        elif not fallback_pass:
            fail_reason = "intervals_too_ambiguous_dense_fallback_mass_high"
        elif not price_pass:
            fail_reason = "interval_plus_fallback_not_below_dense_budget"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=identifying_correct,
        fail_reason=fail_reason,
        sample_count=args.sample_count,
        threshold=args.threshold,
        certified_fraction=certified_fraction,
        dense_fallback_fraction=fallback_fraction,
        certified_mismatch_fraction=certified_mismatch,
        final_decision_mismatch_fraction=final_mismatch,
        priced_work_ratio=priced_work_ratio,
        projected_speedup=float(projected_speedup),
        mean_interval_width=float(np.mean(upper - lower)),
        mean_margin_distance=float(np.mean(margin_distance(exact, args.threshold))),
        exact_positive_fraction=exact_positive,
        exact_negative_fraction=exact_negative,
        exact_zero_fraction=exact_zero,
        price_components=price_components,
    )


def finite_trial(trial: TrialResult) -> bool:
    values = (
        trial.certified_fraction,
        trial.dense_fallback_fraction,
        trial.certified_mismatch_fraction,
        trial.final_decision_mismatch_fraction,
        trial.priced_work_ratio,
        trial.projected_speedup,
        trial.mean_interval_width,
        trial.mean_margin_distance,
        trial.exact_positive_fraction,
        trial.exact_negative_fraction,
        trial.exact_zero_fraction,
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
            "certified_fraction_mean": float(np.mean([row.certified_fraction for row in rows])),
            "dense_fallback_fraction_mean": float(
                np.mean([row.dense_fallback_fraction for row in rows])
            ),
            "certified_mismatch_fraction_mean": float(
                np.mean([row.certified_mismatch_fraction for row in rows])
            ),
            "final_decision_mismatch_fraction_mean": float(
                np.mean([row.final_decision_mismatch_fraction for row in rows])
            ),
            "priced_work_ratio_mean": float(np.mean([row.priced_work_ratio for row in rows])),
            "projected_speedup_mean": float(np.mean([row.projected_speedup for row in rows])),
            "mean_interval_width_mean": float(np.mean([row.mean_interval_width for row in rows])),
            "mean_margin_distance_mean": float(
                np.mean([row.mean_margin_distance for row in rows])
            ),
        }
    return out


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.sample_count < 64:
        raise ValueError("--sample-count must be at least 64")
    if args.threshold <= 0.0:
        raise ValueError("--threshold must be positive")

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
        "positive_condition_no_certified_mismatch": all(
            trial.certified_mismatch_fraction <= args.max_certified_mismatch_fraction
            for trial in trials
            if trial.condition == "positive"
        ),
        "positive_condition_low_fallback_mass": all(
            trial.dense_fallback_fraction <= args.max_dense_fallback_fraction
            for trial in trials
            if trial.condition == "positive"
        ),
        "loose_bounds_control_all_fail_fallback": all(
            (not trial.passed)
            and trial.fail_reason == "intervals_too_ambiguous_dense_fallback_mass_high"
            for trial in trials
            if trial.condition == "loose_bounds_fail"
        ),
        "near_margin_control_all_fail_fallback": all(
            (not trial.passed)
            and trial.fail_reason == "intervals_too_ambiguous_dense_fallback_mass_high"
            for trial in trials
            if trial.condition == "near_margin_fail"
        ),
        "optimistic_control_all_fail_mismatch": all(
            (not trial.passed)
            and trial.fail_reason
            == "certified_interval_decision_mismatches_exact_gradient_sign"
            for trial in trials
            if trial.condition == "optimistic_false_certificate_fail"
        ),
        "all_conditions_identified": all(trial.identifying_correct for trial in trials),
    }
    decision = (
        "PASS_R501_S39_GRADIENT_SIGN_MARGIN_GATE"
        if all(required_passes.values())
        else "FAIL_R501_S39_GRADIENT_SIGN_MARGIN_GATE"
    )
    disposition = (
        "synthetic_positive_control_identified_not_training_speed_claim"
        if decision.startswith("PASS_")
        else "gradient_sign_margin_gate_failed_or_non_identifying"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "CPU synthetic contrastive gradient-sign interval certificate gate. "
            "It checks certified ternary update agreement, dense fallback mass, "
            "and interval-plus-fallback pricing. It is not heldout CE progress "
            "or GPU training-speed evidence."
        ),
        "finding": {
            "round": "R501",
            "aspect": "Speed",
            "id": "R501-S39",
            "title": "Contrastive Gradient-Sign Margin Certificate",
        },
        "thresholds": {
            "threshold": args.threshold,
            "max_priced_work_ratio": args.max_priced_work_ratio,
            "max_dense_fallback_fraction": args.max_dense_fallback_fraction,
            "max_certified_mismatch_fraction": args.max_certified_mismatch_fraction,
            "required_projected_speedup": 1.0 / args.max_priced_work_ratio,
        },
        "shape": {
            "seed": args.seed,
            "seed_count": args.seed_count,
            "sample_count": args.sample_count,
        },
        "required_passes": required_passes,
        "aggregate": aggregate(trials),
        "trials": [asdict(trial) for trial in trials],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    positive = result["aggregate"]["positive"]
    print(
        "[R501_S39_GRADIENT_SIGN_MARGIN_GATE] "
        f"active=1 decision={decision} disposition={disposition} "
        f"positive_certified_fraction_mean={positive['certified_fraction_mean']:.6f} "
        f"positive_fallback_fraction_mean={positive['dense_fallback_fraction_mean']:.6f} "
        f"positive_certified_mismatch_fraction_mean={positive['certified_mismatch_fraction_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
