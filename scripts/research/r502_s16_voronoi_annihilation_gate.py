#!/usr/bin/env python3
"""R502 S16 ternary Voronoi-cell gradient annihilation gate.

This CPU diagnostic implements the R502 Speed finding:
"Ternary Voronoi-Cell Gradient Annihilation".

The gate builds synthetic sequence segments with true maximum ternary-weight
gradient magnitudes and forward-computed cotangent-energy upper bounds. A
segment skips dense backward only when the certified bound remains inside the
Voronoi cell of the current ternary weight, so no active ternary flip can be
missed while the paid backward work drops.

Controls exercise the declared risks: unsafe bounds can skip real flips, low
skip rate fails the denominator, expensive bound checks erase the gain, a
miscalibrated Voronoi threshold skips active updates, and overly loose bounds
become safe but too conservative to matter.

This is a synthetic CPU ternary Voronoi-cell skip diagnostic, not heldout CE
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


SCHEMA = "r502_s16_voronoi_annihilation_gate.v1"
CONDITIONS = (
    "positive",
    "unsafe_bound_fail",
    "low_skip_rate_fail",
    "bound_overhead_fail",
    "threshold_miscalibration_fail",
    "loose_bound_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    segment_count: int
    true_flip_fraction: float
    skipped_fraction: float
    missed_flip_fraction: float
    false_skip_fraction: float
    bound_violation_fraction: float
    certified_inside_fraction: float
    max_skipped_true_gradient: float
    true_voronoi_threshold: float
    decision_threshold: float
    bound_check_overhead_ratio: float
    priced_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50216)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--segment-count", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--active-fraction", type=float, default=0.30)
    parser.add_argument("--low-skip-active-fraction", type=float, default=0.90)
    parser.add_argument("--bound-check-overhead-ratio", type=float, default=0.05)
    parser.add_argument("--expensive-bound-check-overhead-ratio", type=float, default=0.75)
    parser.add_argument("--min-skip-fraction", type=float, default=0.20)
    parser.add_argument("--max-missed-flip-fraction", type=float, default=0.0)
    parser.add_argument("--max-bound-violation-fraction", type=float, default=0.0)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.90)
    return parser.parse_args()


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def make_true_gradients(
    args: argparse.Namespace,
    rng: np.random.Generator,
    condition: str,
) -> np.ndarray:
    active_fraction = (
        args.low_skip_active_fraction
        if condition == "low_skip_rate_fail"
        else args.active_fraction
    )
    active_count = int(round(active_fraction * args.segment_count))
    active = np.zeros(args.segment_count, dtype=bool)
    active[:active_count] = True
    rng.shuffle(active)
    gradients = rng.uniform(0.05, 0.25, size=args.segment_count)
    gradients[active] = rng.uniform(0.65, 1.20, size=active_count)
    return gradients


def make_bounds(true_gradients: np.ndarray, condition: str) -> np.ndarray:
    if condition == "unsafe_bound_fail":
        return true_gradients * 0.30
    if condition == "loose_bound_fail":
        return true_gradients * 4.00 + 0.25
    return true_gradients * 1.12 + 0.02


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    true_threshold = 0.5 / args.learning_rate
    decision_threshold = 1.0 if condition == "threshold_miscalibration_fail" else true_threshold
    true_gradients = make_true_gradients(args, rng, condition)
    bounds = make_bounds(true_gradients, condition)
    true_flips = true_gradients >= true_threshold
    skipped = bounds < decision_threshold
    missed_flips = skipped & true_flips
    false_skips = skipped & (bounds < true_threshold) & (true_gradients >= true_threshold)
    bound_violations = bounds < true_gradients

    skipped_fraction = float(np.mean(skipped))
    true_flip_fraction = float(np.mean(true_flips))
    missed_flip_fraction = float(np.mean(missed_flips))
    false_skip_fraction = float(np.mean(false_skips))
    bound_violation_fraction = float(np.mean(bound_violations))
    certified_inside_fraction = float(np.mean(bounds < true_threshold))
    if np.any(skipped):
        max_skipped_true_gradient = float(np.max(true_gradients[skipped]))
    else:
        max_skipped_true_gradient = 0.0
    overhead = (
        args.expensive_bound_check_overhead_ratio
        if condition == "bound_overhead_fail"
        else args.bound_check_overhead_ratio
    )
    priced_work_ratio = (1.0 - skipped_fraction) + overhead
    projected_speedup = 1.0 / priced_work_ratio

    failed: list[str] = []
    if not finite(
        skipped_fraction,
        true_flip_fraction,
        missed_flip_fraction,
        false_skip_fraction,
        bound_violation_fraction,
        certified_inside_fraction,
        max_skipped_true_gradient,
        true_threshold,
        decision_threshold,
        overhead,
        priced_work_ratio,
        projected_speedup,
    ):
        failed.append("non_finite_metric")
    if bound_violation_fraction > args.max_bound_violation_fraction:
        failed.append("unsafe_bound")
    if missed_flip_fraction > args.max_missed_flip_fraction:
        failed.append("missed_flip")
    if skipped_fraction < args.min_skip_fraction:
        failed.append("skip_rate")
    if priced_work_ratio > args.max_priced_work_ratio:
        failed.append("priced_work")
    if max_skipped_true_gradient >= true_threshold:
        failed.append("voronoi_cell_violation")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed) if failed else None,
        segment_count=args.segment_count,
        true_flip_fraction=true_flip_fraction,
        skipped_fraction=skipped_fraction,
        missed_flip_fraction=missed_flip_fraction,
        false_skip_fraction=false_skip_fraction,
        bound_violation_fraction=bound_violation_fraction,
        certified_inside_fraction=certified_inside_fraction,
        max_skipped_true_gradient=max_skipped_true_gradient,
        true_voronoi_threshold=true_threshold,
        decision_threshold=decision_threshold,
        bound_check_overhead_ratio=overhead,
        priced_work_ratio=priced_work_ratio,
        projected_speedup=projected_speedup,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "segment_count",
        "true_flip_fraction",
        "skipped_fraction",
        "missed_flip_fraction",
        "false_skip_fraction",
        "bound_violation_fraction",
        "certified_inside_fraction",
        "max_skipped_true_gradient",
        "true_voronoi_threshold",
        "decision_threshold",
        "bound_check_overhead_ratio",
        "priced_work_ratio",
        "projected_speedup",
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
    unsafe = [result for result in results if result.condition == "unsafe_bound_fail"]
    low_skip = [result for result in results if result.condition == "low_skip_rate_fail"]
    overhead = [result for result in results if result.condition == "bound_overhead_fail"]
    threshold = [result for result in results if result.condition == "threshold_miscalibration_fail"]
    loose = [result for result in results if result.condition == "loose_bound_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_no_missed_flips": (
            aggregates["positive"]["missed_flip_fraction_max"] == 0.0
            and aggregates["positive"]["bound_violation_fraction_max"] == 0.0
            and aggregates["positive"]["max_skipped_true_gradient_max"] < aggregates["positive"]["true_voronoi_threshold_min"]
        ),
        "positive_condition_skip_rate_and_cost": (
            aggregates["positive"]["skipped_fraction_mean"] >= 0.20
            and aggregates["positive"]["priced_work_ratio_mean"] <= 0.90
        ),
        "unsafe_bound_control_all_fail_safety": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "unsafe_bound" in result.fail_reason
                or "missed_flip" in result.fail_reason
                or "voronoi_cell_violation" in result.fail_reason
            )
            for result in unsafe
        ),
        "low_skip_control_all_fail_skip_or_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "skip_rate" in result.fail_reason
                or "priced_work" in result.fail_reason
            )
            for result in low_skip
        ),
        "bound_overhead_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "priced_work" in result.fail_reason
            for result in overhead
        ),
        "threshold_control_all_fail_missed_flip": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "missed_flip" in result.fail_reason
                or "voronoi_cell_violation" in result.fail_reason
            )
            for result in threshold
        ),
        "loose_bound_control_all_fail_skip_or_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "skip_rate" in result.fail_reason
                or "priced_work" in result.fail_reason
            )
            for result in loose
        ),
        "all_conditions_identified": all(result.identifying_correct for result in results),
    }


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.segment_count < 1:
        raise ValueError("--segment-count must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")

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
    decision = "PASS_R502_S16_VORONOI_ANNIHILATION_GATE" if all(checks.values()) else "FAIL_R502_S16_VORONOI_ANNIHILATION_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU ternary Voronoi-cell gradient-annihilation gate only; "
            "not heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_S16_VORONOI_ANNIHILATION_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_skipped_fraction_mean={positive['skipped_fraction_mean']:.6f} "
        f"positive_missed_flip_fraction_max={positive['missed_flip_fraction_max']:.12e} "
        f"positive_bound_violation_fraction_max={positive['bound_violation_fraction_max']:.12e} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
