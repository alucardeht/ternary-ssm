#!/usr/bin/env python3
"""R501 S84 orthogonalized credit trajectory gate.

This CPU diagnostic implements the R501 Quality Paradigm 6 finding:
"Orthogonalized Credit Trajectory (OCT)".

The gate constructs credit vectors with a new capability component plus a
harmful component along a recently used capability subspace. The candidate
subtracts the projection onto the recent credit basis, so the new update becomes
orthogonal to the recent trajectory. A positive case must reduce prior-overwrite
energy while retaining the new capability direction and matching the oracle
orthogonal residual.

Controls exercise the declared kill criteria: the new target may not be
independent, the recent basis can be stale or contaminated with the new
capability, no interference may exist to justify the projection, or the
projection machinery can be too expensive.

This is a synthetic CPU credit-trajectory diagnostic, not heldout CE progress,
real corpus token efficiency, GPU training-speed evidence, or production v12
integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s84_oct_gate.v1"
CONDITIONS = (
    "positive",
    "nonorthogonal_target_fail",
    "stale_basis_fail",
    "no_interference_fail",
    "noisy_basis_fail",
    "projection_cost_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    credit_count: int
    param_dim: int
    recent_rank: int
    new_rank: int
    prior_overlap_before_fraction: float
    prior_overlap_after_fraction: float
    overwrite_reduction_fraction: float
    new_capability_retention_fraction: float
    future_descent_cosine: float
    oracle_relative_error: float
    observed_basis_condition_proxy: float
    projection_overhead_ratio: float
    baseline_repair_units: float
    oct_repair_units: float
    priced_work_ratio: float
    projected_capability_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50184)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--credit-count", type=int, default=160)
    parser.add_argument("--param-dim", type=int, default=96)
    parser.add_argument("--recent-rank", type=int, default=8)
    parser.add_argument("--new-rank", type=int, default=6)
    parser.add_argument("--interference-scale", type=float, default=0.9)
    parser.add_argument("--noise-scale", type=float, default=0.01)
    parser.add_argument("--projection-overhead", type=float, default=0.05)
    parser.add_argument("--expensive-projection-overhead", type=float, default=0.85)
    parser.add_argument("--repair-multiplier", type=float, default=4.0)
    parser.add_argument("--min-baseline-prior-overlap", type=float, default=0.2)
    parser.add_argument("--max-prior-overlap-after", type=float, default=0.025)
    parser.add_argument("--min-overwrite-reduction", type=float, default=0.92)
    parser.add_argument("--min-new-retention", type=float, default=0.96)
    parser.add_argument("--min-future-cosine", type=float, default=0.995)
    parser.add_argument("--max-oracle-error", type=float, default=0.05)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.55)
    parser.add_argument("--min-capability-efficiency-multiplier", type=float, default=1.5)
    return parser.parse_args()


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def orthonormalize(matrix: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(matrix)
    keep = np.abs(np.diag(r)) > 1.0e-10
    if not np.any(keep):
        return q[:, :1] * 0.0
    return q[:, keep]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0e-30)
    return float(np.dot(a, b) / denom)


def projection(basis: np.ndarray, values: np.ndarray) -> np.ndarray:
    return (basis @ (basis.T @ values.T)).T


def make_bases(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full_q = orthonormalize(rng.normal(size=(args.param_dim, args.param_dim)))
    prior_basis = full_q[:, : args.recent_rank]
    new_basis = full_q[:, args.recent_rank : args.recent_rank + args.new_rank]
    observed_basis = prior_basis

    if condition == "nonorthogonal_target_fail":
        new_basis = prior_basis[:, : args.new_rank]
    elif condition == "stale_basis_fail":
        observed_basis = full_q[
            :,
            args.recent_rank
            + args.new_rank : args.recent_rank
            + args.new_rank
            + args.recent_rank,
        ]
    elif condition == "noisy_basis_fail":
        polluted = np.concatenate(
            [
                prior_basis[:, : args.recent_rank - args.new_rank],
                new_basis[:, : args.new_rank],
            ],
            axis=1,
        )
        observed_basis = orthonormalize(polluted)

    return prior_basis, new_basis, orthonormalize(observed_basis)


def make_credits(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
    prior_basis: np.ndarray,
    new_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    new_coeff = rng.normal(size=(args.credit_count, new_basis.shape[1]))
    new_component = new_coeff @ new_basis.T

    prior_coeff = rng.normal(size=(args.credit_count, prior_basis.shape[1]))
    if condition == "no_interference_fail":
        interference_scale = 0.0
    else:
        interference_scale = args.interference_scale
    prior_component = interference_scale * (prior_coeff @ prior_basis.T)

    noise = rng.normal(size=(args.credit_count, args.param_dim))
    noise = noise - projection(prior_basis, noise)
    if new_basis.shape[1] > 0:
        noise = noise - projection(orthonormalize(new_basis), noise)
    noise_norm = np.maximum(np.linalg.norm(noise, axis=1, keepdims=True), 1.0e-30)
    noise = args.noise_scale * noise / noise_norm

    return new_component + prior_component + noise, new_component


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    prior_basis, new_basis, observed_basis = make_bases(rng, args, condition)
    credits, new_component = make_credits(rng, args, condition, prior_basis, new_basis)

    oracle = credits - projection(prior_basis, credits)
    candidate = credits - projection(observed_basis, credits)

    before_prior = projection(prior_basis, credits)
    after_prior = projection(prior_basis, candidate)
    before_prior_energy = float(np.sum(before_prior * before_prior))
    after_prior_energy = float(np.sum(after_prior * after_prior))
    credit_energy = max(float(np.sum(credits * credits)), 1.0e-30)

    before_overlap = before_prior_energy / credit_energy
    after_overlap = after_prior_energy / max(float(np.sum(candidate * candidate)), 1.0e-30)
    overwrite_reduction = 1.0 - after_prior_energy / max(before_prior_energy, 1.0e-30)

    new_energy = max(float(np.sum(new_component * new_component)), 1.0e-30)
    new_retention = float(np.sum(candidate * new_component)) / new_energy
    future_cosine = cosine(candidate.reshape(-1), new_component.reshape(-1))
    oracle_error = float(np.linalg.norm(candidate - oracle)) / max(
        float(np.linalg.norm(oracle)),
        1.0e-30,
    )

    singular_values = np.linalg.svd(observed_basis, compute_uv=False)
    condition_proxy = float(np.max(singular_values) / max(np.min(singular_values), 1.0e-30))

    projection_overhead = (
        args.expensive_projection_overhead
        if condition == "projection_cost_fail"
        else args.projection_overhead
    )
    baseline_units = 1.0 + args.repair_multiplier * before_overlap
    oct_units = 1.0 + projection_overhead + args.repair_multiplier * after_overlap
    priced_work_ratio = oct_units / max(baseline_units, 1.0e-30)
    capability_multiplier = max(overwrite_reduction, 0.0) * max(new_retention, 0.0)
    capability_multiplier /= max(priced_work_ratio, 1.0e-30)

    failed: list[str] = []
    if not finite(
        before_overlap,
        after_overlap,
        overwrite_reduction,
        new_retention,
        future_cosine,
        oracle_error,
        condition_proxy,
        projection_overhead,
        baseline_units,
        oct_units,
        priced_work_ratio,
        capability_multiplier,
    ):
        failed.append("non_finite_metric")
    if before_overlap < args.min_baseline_prior_overlap:
        failed.append("baseline_prior_overlap")
    if after_overlap > args.max_prior_overlap_after:
        failed.append("residual_prior_overlap")
    if overwrite_reduction < args.min_overwrite_reduction:
        failed.append("overwrite_reduction")
    if new_retention < args.min_new_retention:
        failed.append("new_capability_retention")
    if future_cosine < args.min_future_cosine:
        failed.append("future_descent_direction")
    if oracle_error > args.max_oracle_error:
        failed.append("oracle_residual_match")
    if priced_work_ratio > args.max_priced_work_ratio:
        failed.append("price")
    if capability_multiplier < args.min_capability_efficiency_multiplier:
        failed.append("capability_efficiency")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed) if failed else None,
        credit_count=args.credit_count,
        param_dim=args.param_dim,
        recent_rank=args.recent_rank,
        new_rank=args.new_rank,
        prior_overlap_before_fraction=float(before_overlap),
        prior_overlap_after_fraction=float(after_overlap),
        overwrite_reduction_fraction=float(overwrite_reduction),
        new_capability_retention_fraction=float(new_retention),
        future_descent_cosine=float(future_cosine),
        oracle_relative_error=float(oracle_error),
        observed_basis_condition_proxy=float(condition_proxy),
        projection_overhead_ratio=float(projection_overhead),
        baseline_repair_units=float(baseline_units),
        oct_repair_units=float(oct_units),
        priced_work_ratio=float(priced_work_ratio),
        projected_capability_efficiency_multiplier=float(capability_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "credit_count",
        "param_dim",
        "recent_rank",
        "new_rank",
        "prior_overlap_before_fraction",
        "prior_overlap_after_fraction",
        "overwrite_reduction_fraction",
        "new_capability_retention_fraction",
        "future_descent_cosine",
        "oracle_relative_error",
        "observed_basis_condition_proxy",
        "projection_overhead_ratio",
        "baseline_repair_units",
        "oct_repair_units",
        "priced_work_ratio",
        "projected_capability_efficiency_multiplier",
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
    nonorthogonal = [
        result for result in results if result.condition == "nonorthogonal_target_fail"
    ]
    stale = [result for result in results if result.condition == "stale_basis_fail"]
    no_interference = [
        result for result in results if result.condition == "no_interference_fail"
    ]
    noisy = [result for result in results if result.condition == "noisy_basis_fail"]
    cost = [result for result in results if result.condition == "projection_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_has_overwrite_to_remove": (
            aggregates["positive"]["prior_overlap_before_fraction_min"] >= 0.2
        ),
        "positive_condition_removes_recent_overlap": (
            aggregates["positive"]["prior_overlap_after_fraction_max"] <= 0.025
            and aggregates["positive"]["overwrite_reduction_fraction_min"] >= 0.92
        ),
        "positive_condition_preserves_new_capability": (
            aggregates["positive"]["new_capability_retention_fraction_min"] >= 0.96
            and aggregates["positive"]["future_descent_cosine_min"] >= 0.995
            and aggregates["positive"]["oracle_relative_error_max"] <= 0.05
        ),
        "positive_condition_priced": (
            aggregates["positive"]["priced_work_ratio_mean"] <= 0.55
            and aggregates["positive"]["projected_capability_efficiency_multiplier_mean"] >= 1.5
        ),
        "nonorthogonal_target_control_all_fail_retention": all(
            (not result.passed)
            and result.fail_reason is not None
            and "new_capability_retention" in result.fail_reason
            for result in nonorthogonal
        ),
        "stale_basis_control_all_fail_overlap_or_oracle": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "residual_prior_overlap" in result.fail_reason
                or "oracle_residual_match" in result.fail_reason
            )
            for result in stale
        ),
        "no_interference_control_all_fail_baseline_or_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "baseline_prior_overlap" in result.fail_reason
                or "price" in result.fail_reason
                or "capability_efficiency" in result.fail_reason
            )
            for result in no_interference
        ),
        "noisy_basis_control_all_fail_retention_or_oracle": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "new_capability_retention" in result.fail_reason
                or "oracle_residual_match" in result.fail_reason
            )
            for result in noisy
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
    decision = "PASS_R501_S84_OCT_GATE" if all(checks.values()) else "FAIL_R501_S84_OCT_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU orthogonalized credit trajectory gate only; not "
            "heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S84_OCT_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_prior_overlap_before_min={positive['prior_overlap_before_fraction_min']:.6f} "
        f"positive_prior_overlap_after_max={positive['prior_overlap_after_fraction_max']:.6f} "
        f"positive_overwrite_reduction_min={positive['overwrite_reduction_fraction_min']:.6f} "
        f"positive_new_retention_min={positive['new_capability_retention_fraction_min']:.6f} "
        f"positive_future_descent_cosine_min={positive['future_descent_cosine_min']:.6f} "
        f"positive_oracle_relative_error_max={positive['oracle_relative_error_max']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_capability_efficiency_multiplier_mean={positive['projected_capability_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
