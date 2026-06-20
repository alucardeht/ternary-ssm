#!/usr/bin/env python3
"""R501 S68 sequence-quotient capability substrate gate.

This CPU diagnostic implements the R501 Quality Math 5 finding:
"Sequence-Quotient Capability Substrate (SQCS)".

The gate groups repeated state-transition signatures into quotient classes and
checks whether dense credit can be replaced by n_C times a representative
gradient plus a small residual r_C. A positive case must show high exact
multiplicity, low quotient residual, and a priced denominator below the dense
per-token path.

Controls exercise the declared risks: continuous context drift makes r_C
dominate, low multiplicity erases the denominator, and hash collisions merge
nonidentical transitions into false credit.

This is a synthetic CPU transition-signature quotient diagnostic, not heldout
CE progress, GPU training-speed evidence, or production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s68_sqcs_gate.v1"
CONDITIONS = (
    "positive",
    "drift_residual_fail",
    "low_multiplicity_fail",
    "hash_collision_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    transition_count: int
    class_count: int
    quotient_bucket_count: int
    mean_multiplicity: float
    max_multiplicity: float
    collision_fraction: float
    residual_ratio: float
    quotient_gradient_relative_error: float
    priced_work_ratio: float
    projected_token_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50168)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--transition-count", type=int, default=512)
    parser.add_argument("--signature-dim", type=int, default=6)
    parser.add_argument("--gradient-dim", type=int, default=10)
    parser.add_argument("--max-residual-ratio", type=float, default=0.05)
    parser.add_argument("--max-gradient-error", type=float, default=0.05)
    parser.add_argument("--min-mean-multiplicity", type=float, default=2.0)
    parser.add_argument("--max-collision-fraction", type=float, default=0.0)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.60)
    return parser.parse_args()


def condition_shape(condition: str) -> tuple[int, int | None, float]:
    if condition == "low_multiplicity_fail":
        return 430, None, 0.002
    if condition == "hash_collision_fail":
        return 64, 16, 0.002
    if condition == "drift_residual_fail":
        return 32, None, 0.085
    return 32, None, 0.002


def make_assignments(total: int, class_count: int) -> np.ndarray:
    base = np.arange(class_count)
    repeats = int(np.ceil(total / class_count))
    assignments = np.tile(base, repeats)[:total]
    return assignments


def make_fixture(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    class_count, bucket_count, noise_scale = condition_shape(condition)
    assignments = make_assignments(args.transition_count, class_count)
    rng.shuffle(assignments)
    signatures = rng.normal(size=(class_count, args.signature_dim))
    representative_gradients = rng.normal(size=(class_count, args.gradient_dim))
    gradients = representative_gradients[assignments].copy()
    if condition == "drift_residual_fail":
        drift_axis = rng.normal(size=args.gradient_dim)
        drift_axis /= max(float(np.linalg.norm(drift_axis)), 1.0e-30)
        phase = np.linspace(-1.0, 1.0, args.transition_count)
        gradients += noise_scale * phase[:, None] * drift_axis[None, :]
    else:
        gradients += rng.normal(scale=noise_scale, size=gradients.shape)
    return assignments, signatures, gradients, bucket_count or class_count


def quotient_gradient(
    assignments: np.ndarray,
    gradients: np.ndarray,
    class_count: int,
    bucket_count: int,
) -> tuple[np.ndarray, int, float, float, float]:
    dense = np.sum(gradients, axis=0)
    quotient = np.zeros_like(dense)
    collision_buckets = 0
    multiplicities: list[int] = []
    for bucket in range(bucket_count):
        classes = np.flatnonzero(np.arange(class_count) % bucket_count == bucket)
        indices = np.flatnonzero(np.isin(assignments, classes))
        if indices.size == 0:
            continue
        if classes.size > 1:
            collision_buckets += 1
        representative_class = classes[0]
        representative_index = int(np.flatnonzero(assignments == representative_class)[0])
        quotient += indices.size * gradients[representative_index]
        multiplicities.append(int(indices.size))
    residual = dense - quotient
    residual_ratio = float(np.linalg.norm(residual)) / max(float(np.linalg.norm(quotient)), 1.0e-30)
    gradient_error = float(np.linalg.norm(residual)) / max(float(np.linalg.norm(dense)), 1.0e-30)
    collision_fraction = collision_buckets / max(bucket_count, 1)
    mean_multiplicity = float(np.mean(multiplicities)) if multiplicities else 0.0
    max_multiplicity = float(np.max(multiplicities)) if multiplicities else 0.0
    return residual, bucket_count, mean_multiplicity, max_multiplicity, collision_fraction, residual_ratio, gradient_error


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    assignments, _signatures, gradients, bucket_count = make_fixture(rng, args, condition)
    class_count = int(np.max(assignments)) + 1
    (
        _residual,
        quotient_bucket_count,
        mean_multiplicity,
        max_multiplicity,
        collision_fraction,
        residual_ratio,
        gradient_error,
    ) = quotient_gradient(assignments, gradients, class_count, bucket_count)
    overhead = 0.10
    priced_work_ratio = quotient_bucket_count / args.transition_count + overhead
    projected_multiplier = 1.0 / priced_work_ratio

    failed: list[str] = []
    if not finite(
        mean_multiplicity,
        max_multiplicity,
        collision_fraction,
        residual_ratio,
        gradient_error,
        priced_work_ratio,
        projected_multiplier,
    ):
        failed.append("non_finite_metric")
    if mean_multiplicity < args.min_mean_multiplicity:
        failed.append("multiplicity")
    if collision_fraction > args.max_collision_fraction:
        failed.append("hash_collision")
    if residual_ratio > args.max_residual_ratio:
        failed.append("quotient_residual")
    if gradient_error > args.max_gradient_error:
        failed.append("gradient_error")
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
        transition_count=args.transition_count,
        class_count=class_count,
        quotient_bucket_count=quotient_bucket_count,
        mean_multiplicity=float(mean_multiplicity),
        max_multiplicity=float(max_multiplicity),
        collision_fraction=float(collision_fraction),
        residual_ratio=float(residual_ratio),
        quotient_gradient_relative_error=float(gradient_error),
        priced_work_ratio=float(priced_work_ratio),
        projected_token_efficiency_multiplier=float(projected_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "transition_count",
        "class_count",
        "quotient_bucket_count",
        "mean_multiplicity",
        "max_multiplicity",
        "collision_fraction",
        "residual_ratio",
        "quotient_gradient_relative_error",
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
    drift = [result for result in results if result.condition == "drift_residual_fail"]
    low = [result for result in results if result.condition == "low_multiplicity_fail"]
    collision = [result for result in results if result.condition == "hash_collision_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_quotient_exact": (
            aggregates["positive"]["mean_multiplicity_mean"] >= 2.0
            and aggregates["positive"]["residual_ratio_max"] <= 0.05
            and aggregates["positive"]["quotient_gradient_relative_error_max"] <= 0.05
            and aggregates["positive"]["collision_fraction_max"] == 0.0
        ),
        "positive_condition_priced": (
            aggregates["positive"]["priced_work_ratio_mean"] <= 0.60
        ),
        "drift_control_all_fail_residual": all(
            (not result.passed)
            and result.fail_reason is not None
            and "quotient_residual" in result.fail_reason
            for result in drift
        ),
        "low_multiplicity_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "multiplicity" in result.fail_reason
            and "price" in result.fail_reason
            for result in low
        ),
        "collision_control_all_fail_collision": all(
            (not result.passed)
            and result.fail_reason is not None
            and "hash_collision" in result.fail_reason
            and "quotient_residual" in result.fail_reason
            for result in collision
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
        "PASS_R501_S68_SQCS_GATE"
        if all(checks.values())
        else "FAIL_R501_S68_SQCS_GATE"
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
            "Synthetic CPU transition-signature quotient gate only; not "
            "heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S68_SQCS_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_mean_multiplicity_mean={positive['mean_multiplicity_mean']:.6f} "
        f"positive_residual_ratio_max={positive['residual_ratio_max']:.6e} "
        f"positive_gradient_error_max={positive['quotient_gradient_relative_error_max']:.6e} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_token_efficiency_multiplier_mean={positive['projected_token_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
