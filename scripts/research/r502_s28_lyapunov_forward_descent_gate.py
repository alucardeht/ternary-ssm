#!/usr/bin/env python3
"""R502-S28 Lyapunov-Certified Forward-Only Descent gate.

This synthetic CPU gate checks whether a cheap forward-computed Lyapunov
certificate can skip recurrent BPTT while using a heuristic gradient that stays
in the same descent half-space as the exact gradient. Controls cover overly
strict certificates, unstable energy, heuristic misalignment, expensive
certificate checks, and local CE-bound failures.
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
    "strict_certificate_fail",
    "unstable_energy_fail",
    "heuristic_misalignment_fail",
    "expensive_certificate_fail",
    "ce_bound_fail",
]


@dataclass
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str
    sample_count: int
    param_dim: int
    certified_fraction: float
    delta_v_negative_fraction: float
    ce_within_bound_fraction: float
    descent_halfspace_fraction: float
    aggregate_descent_cosine: float
    quality_progress_ratio: float
    dense_work_units: float
    lyapunov_check_units: float
    heuristic_work_units: float
    fallback_backward_units: float
    priced_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=50228)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--param-dim", type=int, default=64)
    parser.add_argument("--backward-cost-multiplier", type=float, default=16.0)
    parser.add_argument("--lyapunov-check-cost-multiplier", type=float, default=1.0)
    parser.add_argument("--heuristic-cost-multiplier", type=float, default=1.0)
    parser.add_argument("--expensive-check-cost-multiplier", type=float, default=15.0)
    parser.add_argument("--lyapunov-margin", type=float, default=0.05)
    parser.add_argument("--ce-bound", type=float, default=0.8)
    parser.add_argument("--min-certified-fraction", type=float, default=0.65)
    parser.add_argument("--min-halfspace-fraction", type=float, default=0.98)
    parser.add_argument("--min-descent-cosine", type=float, default=0.98)
    parser.add_argument("--min-quality-progress-ratio", type=float, default=0.90)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.75)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    numerator = float(np.vdot(left, right).real)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 1.0 if np.linalg.norm(left - right) == 0.0 else 0.0
    return numerator / denominator


def make_condition_arrays(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    true_gradient = rng.normal(size=(args.sample_count, args.param_dim))
    if condition == "heuristic_misalignment_fail":
        heuristic_gradient = -true_gradient + 0.05 * rng.normal(size=true_gradient.shape)
    else:
        heuristic_gradient = true_gradient + 0.04 * rng.normal(size=true_gradient.shape)

    if condition == "strict_certificate_fail":
        delta_v = rng.uniform(-0.20, -0.06, size=args.sample_count)
        lyapunov_margin = 0.50
    elif condition == "unstable_energy_fail":
        delta_v = rng.uniform(0.01, 0.30, size=args.sample_count)
        lyapunov_margin = args.lyapunov_margin
    else:
        delta_v = rng.uniform(-0.40, -0.08, size=args.sample_count)
        lyapunov_margin = args.lyapunov_margin

    if condition == "ce_bound_fail":
        local_ce = rng.uniform(1.05, 1.50, size=args.sample_count)
    else:
        local_ce = rng.uniform(0.10, 0.55, size=args.sample_count)

    check_cost = (
        args.expensive_check_cost_multiplier
        if condition == "expensive_certificate_fail"
        else args.lyapunov_check_cost_multiplier
    )
    return true_gradient, heuristic_gradient, delta_v, local_ce, lyapunov_margin, check_cost


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    true_gradient, heuristic_gradient, delta_v, local_ce, margin, check_cost = make_condition_arrays(
        rng,
        args,
        condition,
    )
    certified = (delta_v < -margin) & (local_ce <= args.ce_bound)
    used_gradient = true_gradient.copy()
    used_gradient[certified] = heuristic_gradient[certified]

    true_sum = np.sum(true_gradient, axis=0)
    used_sum = np.sum(used_gradient, axis=0)
    aggregate_cosine = cosine_similarity(true_sum, used_sum)
    quality_progress = float(np.vdot(used_sum, true_sum).real / max(np.vdot(true_sum, true_sum).real, 1e-12))

    certified_dot = np.sum(true_gradient[certified] * heuristic_gradient[certified], axis=1)
    if certified_dot.size:
        halfspace_fraction = float(np.mean(certified_dot > 0.0))
    else:
        halfspace_fraction = 0.0

    dense_per_sample = args.param_dim * args.backward_cost_multiplier
    check_per_sample = args.param_dim * check_cost
    heuristic_per_sample = args.param_dim * args.heuristic_cost_multiplier
    dense_work = float(args.sample_count * dense_per_sample)
    check_work = float(args.sample_count * check_per_sample)
    heuristic_work = float(np.sum(certified) * heuristic_per_sample)
    fallback_work = float((args.sample_count - np.sum(certified)) * dense_per_sample)
    price = (check_work + heuristic_work + fallback_work) / max(dense_work, 1e-12)
    speedup = 1.0 / max(price, 1e-12)

    certified_fraction = float(np.mean(certified))
    delta_v_negative = float(np.mean(delta_v < -args.lyapunov_margin))
    ce_within_bound = float(np.mean(local_ce <= args.ce_bound))

    failed: list[str] = []
    if not np.isfinite(
        [
            certified_fraction,
            delta_v_negative,
            ce_within_bound,
            halfspace_fraction,
            aggregate_cosine,
            quality_progress,
            price,
            speedup,
        ]
    ).all():
        failed.append("non_finite_metric")
    if certified_fraction < args.min_certified_fraction:
        failed.append("certified_fraction")
    if halfspace_fraction < args.min_halfspace_fraction:
        failed.append("descent_halfspace_fraction")
    if aggregate_cosine < args.min_descent_cosine:
        failed.append("aggregate_descent_cosine")
    if quality_progress < args.min_quality_progress_ratio:
        failed.append("quality_progress_ratio")
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
        sample_count=args.sample_count,
        param_dim=args.param_dim,
        certified_fraction=certified_fraction,
        delta_v_negative_fraction=delta_v_negative,
        ce_within_bound_fraction=ce_within_bound,
        descent_halfspace_fraction=halfspace_fraction,
        aggregate_descent_cosine=aggregate_cosine,
        quality_progress_ratio=quality_progress,
        dense_work_units=dense_work,
        lyapunov_check_units=check_work,
        heuristic_work_units=heuristic_work,
        fallback_backward_units=fallback_work,
        priced_work_ratio=price,
        projected_speedup=speedup,
    )


NUMERIC_FIELDS = [
    "sample_count",
    "param_dim",
    "certified_fraction",
    "delta_v_negative_fraction",
    "ce_within_bound_fraction",
    "descent_halfspace_fraction",
    "aggregate_descent_cosine",
    "quality_progress_ratio",
    "dense_work_units",
    "lyapunov_check_units",
    "heuristic_work_units",
    "fallback_backward_units",
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
        "positive_condition_certificate_alignment_and_price": (
            aggregates["positive"]["certified_fraction_min"] >= args.min_certified_fraction
            and aggregates["positive"]["descent_halfspace_fraction_min"] >= args.min_halfspace_fraction
            and aggregates["positive"]["aggregate_descent_cosine_min"] >= args.min_descent_cosine
            and aggregates["positive"]["quality_progress_ratio_min"] >= args.min_quality_progress_ratio
            and aggregates["positive"]["priced_work_ratio_max"] <= args.max_priced_work_ratio
        ),
        "strict_certificate_control_all_fail_certificate": all(
            (not result.passed) and "certified_fraction" in result.fail_reason
            for result in by_condition["strict_certificate_fail"]
        ),
        "unstable_energy_control_all_fail_certificate": all(
            (not result.passed) and "certified_fraction" in result.fail_reason
            for result in by_condition["unstable_energy_fail"]
        ),
        "heuristic_misalignment_control_all_fail_quality": all(
            (not result.passed)
            and (
                "descent_halfspace_fraction" in result.fail_reason
                or "aggregate_descent_cosine" in result.fail_reason
                or "quality_progress_ratio" in result.fail_reason
            )
            for result in by_condition["heuristic_misalignment_fail"]
        ),
        "expensive_certificate_control_all_fail_price": all(
            (not result.passed) and "priced_work" in result.fail_reason
            for result in by_condition["expensive_certificate_fail"]
        ),
        "ce_bound_control_all_fail_certificate": all(
            (not result.passed) and "certified_fraction" in result.fail_reason
            for result in by_condition["ce_bound_fail"]
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
    decision = "PASS_R502_S28_LCFD_GATE" if all(checks.values()) else "FAIL_R502_S28_LCFD_GATE"
    payload = {
        "round": "R502",
        "canonical_id": "R502-S28",
        "aspect": "Speed",
        "finding": "Lyapunov-Certified Forward-Only Descent",
        "decision": decision,
        "disposition": "synthetic_positive_control_identified_not_training_speed_claim",
        "args": vars(args) | {"output": str(args.output) if args.output else None},
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU Lyapunov-certified forward-only descent gate only; "
            "not heldout CE progress, not real corpus token efficiency, not "
            "GPU training speed, and not production v12 integration."
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_S28_LCFD_GATE] "
        f"active=1 decision={decision} "
        f"disposition={payload['disposition']} "
        f"positive_certified_fraction_min={positive['certified_fraction_min']:.12f} "
        f"positive_descent_halfspace_fraction_min={positive['descent_halfspace_fraction_min']:.12f} "
        f"positive_aggregate_descent_cosine_min={positive['aggregate_descent_cosine_min']:.12f} "
        f"positive_quality_progress_ratio_min={positive['quality_progress_ratio_min']:.12f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.12f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.12f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
