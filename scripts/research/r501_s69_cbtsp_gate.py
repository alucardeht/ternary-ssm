#!/usr/bin/env python3
"""R501 S69 curvature-bounded target-subspace projection gate.

This CPU diagnostic implements the R501 Quality Math 6 finding:
"Curvature-Bounded Target-Subspace Projection (CBTSP)".

The gate constructs a quadratic CE local model with low-curvature
capability-bearing coordinates and high-curvature rejected coordinates. A
positive case must project the update into the certified target subspace,
preserve sufficient gradient energy, make the accepted update decrease loss,
and show that the rejected orthogonal update would be a false descent.

Controls exercise the declared risks: stale curvature accepts high-curvature
coordinates, over-rejection drops valid capability-bearing gradient, and an
expensive subspace audit erases the denominator.

This is a synthetic CPU curvature-subspace diagnostic, not heldout CE progress,
GPU training-speed evidence, or production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s69_cbtsp_gate.v1"
CONDITIONS = (
    "positive",
    "stale_curvature_fail",
    "overreject_fail",
    "subspace_cost_fail",
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
    true_safe_dim: int
    estimated_safe_dim: int
    gradient_capture_fraction: float
    valid_rejected_energy_fraction: float
    accepted_loss_delta: float
    rejected_loss_delta: float
    dense_loss_delta: float
    accepted_descent_certificate: float
    rejected_false_descent_margin: float
    priced_work_ratio: float
    projected_token_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50169)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--param-dim", type=int, default=24)
    parser.add_argument("--safe-dim", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    parser.add_argument("--min-gradient-capture", type=float, default=0.45)
    parser.add_argument("--max-valid-rejected-energy", type=float, default=0.10)
    parser.add_argument("--min-descent-certificate", type=float, default=0.05)
    parser.add_argument("--min-false-descent-margin", type=float, default=0.02)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.85)
    return parser.parse_args()


def normalized(rng: np.random.Generator, size: int, scale: float) -> np.ndarray:
    vector = rng.normal(size=size)
    vector /= max(float(np.linalg.norm(vector)), 1.0e-30)
    return vector * scale


def make_problem(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    curvature = np.empty(args.param_dim, dtype=np.float64)
    curvature[: args.safe_dim] = rng.uniform(0.25, 0.45, size=args.safe_dim)
    curvature[args.safe_dim :] = rng.uniform(7.0, 9.0, size=args.param_dim - args.safe_dim)
    gradient = np.zeros(args.param_dim, dtype=np.float64)
    gradient[: args.safe_dim] = normalized(rng, args.safe_dim, scale=1.0)
    unsafe_scale = 0.90
    if condition == "stale_curvature_fail":
        unsafe_scale = 2.20
    gradient[args.safe_dim :] = normalized(
        rng,
        args.param_dim - args.safe_dim,
        scale=unsafe_scale,
    )
    true_safe = np.zeros(args.param_dim, dtype=bool)
    true_safe[: args.safe_dim] = True
    estimated_safe = true_safe.copy()
    if condition == "stale_curvature_fail":
        estimated_safe[:] = True
    elif condition == "overreject_fail":
        estimated_safe[:] = False
        estimated_safe[:2] = True
    return curvature, gradient, true_safe, estimated_safe


def loss_delta(gradient: np.ndarray, curvature: np.ndarray, update: np.ndarray) -> float:
    return float(np.dot(gradient, update) + 0.5 * np.dot(curvature * update, update))


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    curvature, gradient, true_safe, estimated_safe = make_problem(rng, args, condition)
    accepted_gradient = np.where(estimated_safe, gradient, 0.0)
    rejected_gradient = np.where(~estimated_safe, gradient, 0.0)
    accepted_update = -args.learning_rate * accepted_gradient
    rejected_update = -args.learning_rate * rejected_gradient
    dense_update = -args.learning_rate * gradient

    accepted_delta = loss_delta(gradient, curvature, accepted_update)
    rejected_delta = loss_delta(gradient, curvature, rejected_update)
    dense_delta = loss_delta(gradient, curvature, dense_update)
    total_energy = max(float(np.dot(gradient, gradient)), 1.0e-30)
    capture = float(np.dot(accepted_gradient, accepted_gradient)) / total_energy
    true_safe_energy = max(float(np.dot(gradient[true_safe], gradient[true_safe])), 1.0e-30)
    valid_rejected = float(np.dot(gradient[true_safe & ~estimated_safe], gradient[true_safe & ~estimated_safe]))
    valid_rejected_fraction = valid_rejected / true_safe_energy
    descent_certificate = -accepted_delta
    false_descent_margin = rejected_delta
    priced_work_ratio = 0.78
    if condition == "subspace_cost_fail":
        priced_work_ratio = 0.91
    projected_multiplier = 1.0 / priced_work_ratio

    failed: list[str] = []
    if not finite(
        capture,
        valid_rejected_fraction,
        accepted_delta,
        rejected_delta,
        dense_delta,
        descent_certificate,
        false_descent_margin,
        priced_work_ratio,
        projected_multiplier,
    ):
        failed.append("non_finite_metric")
    if capture < args.min_gradient_capture:
        failed.append("gradient_capture")
    if valid_rejected_fraction > args.max_valid_rejected_energy:
        failed.append("capability_rejected")
    if descent_certificate < args.min_descent_certificate:
        failed.append("descent_certificate")
    if false_descent_margin < args.min_false_descent_margin:
        failed.append("false_descent")
    if priced_work_ratio > args.max_priced_work_ratio:
        failed.append("price")

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
        true_safe_dim=args.safe_dim,
        estimated_safe_dim=int(np.count_nonzero(estimated_safe)),
        gradient_capture_fraction=float(capture),
        valid_rejected_energy_fraction=float(valid_rejected_fraction),
        accepted_loss_delta=float(accepted_delta),
        rejected_loss_delta=float(rejected_delta),
        dense_loss_delta=float(dense_delta),
        accepted_descent_certificate=float(descent_certificate),
        rejected_false_descent_margin=float(false_descent_margin),
        priced_work_ratio=float(priced_work_ratio),
        projected_token_efficiency_multiplier=float(projected_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "param_dim",
        "true_safe_dim",
        "estimated_safe_dim",
        "gradient_capture_fraction",
        "valid_rejected_energy_fraction",
        "accepted_loss_delta",
        "rejected_loss_delta",
        "dense_loss_delta",
        "accepted_descent_certificate",
        "rejected_false_descent_margin",
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
    stale = [result for result in results if result.condition == "stale_curvature_fail"]
    overreject = [result for result in results if result.condition == "overreject_fail"]
    cost = [result for result in results if result.condition == "subspace_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_safe_descent": (
            aggregates["positive"]["gradient_capture_fraction_mean"] >= 0.45
            and aggregates["positive"]["accepted_descent_certificate_min"] >= 0.05
            and aggregates["positive"]["rejected_false_descent_margin_min"] >= 0.02
        ),
        "positive_condition_no_capability_rejection": (
            aggregates["positive"]["valid_rejected_energy_fraction_max"] <= 0.10
        ),
        "positive_condition_priced": (
            aggregates["positive"]["priced_work_ratio_mean"] <= 0.85
        ),
        "stale_control_all_fail_descent": all(
            (not result.passed)
            and result.fail_reason is not None
            and "descent_certificate" in result.fail_reason
            for result in stale
        ),
        "overreject_control_all_fail_capability": all(
            (not result.passed)
            and result.fail_reason is not None
            and "gradient_capture" in result.fail_reason
            and "capability_rejected" in result.fail_reason
            for result in overreject
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
        "PASS_R501_S69_CBTSP_GATE"
        if all(checks.values())
        else "FAIL_R501_S69_CBTSP_GATE"
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
            "Synthetic CPU curvature-bounded target-subspace projection gate "
            "only; not heldout CE progress, not real corpus token efficiency, "
            "not GPU training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S69_CBTSP_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_gradient_capture_fraction_mean={positive['gradient_capture_fraction_mean']:.6f} "
        f"positive_accepted_descent_certificate_min={positive['accepted_descent_certificate_min']:.6f} "
        f"positive_rejected_false_descent_margin_min={positive['rejected_false_descent_margin_min']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_token_efficiency_multiplier_mean={positive['projected_token_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
