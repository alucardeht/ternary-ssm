#!/usr/bin/env python3
"""R501 S63 exact matrix-fraction cotangent reconstruction gate.

This CPU diagnostic implements the R501 Speed Paradigm 15 finding:
"Exact Matrix-Fraction Cotangent Reconstruction (EMFCR)".

The gate compares a dense gradient update against a fractional matrix update
that contains only the cotangent component required to cross ternary
quantization boundaries. A positive case must preserve the exact post-update
ternary matrix and a witness objective while admitting a low-rank fractional
reconstruction priced below the dense backward path.

Controls exercise the declared risks: an overly strict boundary epsilon misses
real ternary crossings, too many fractional crossings erase the speed case, and
a high-rank crossing pattern cannot be reconstructed as a cheap matrix fraction.

This is a synthetic CPU diagnostic, not heldout CE progress, GPU training-speed
evidence, Vulkan benchmark evidence, or production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s63_emfcr_gate.v1"
CONDITIONS = (
    "positive",
    "epsilon_miss_fail",
    "dense_fraction_fail",
    "high_rank_fraction_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    rows: int
    cols: int
    fractional_boundary_fraction: float
    fraction_rank: int
    rank_fraction: float
    ternary_mismatch_fraction: float
    objective_equivalence_error: float
    rank_limited_reconstruction_error: float
    scan_work_ratio: float
    priced_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50163)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.10)
    parser.add_argument("--ternary-threshold", type=float, default=0.50)
    parser.add_argument("--boundary-margin", type=float, default=0.012)
    parser.add_argument("--max-rank", type=int, default=4)
    parser.add_argument("--max-ternary-mismatch", type=float, default=0.0)
    parser.add_argument("--max-objective-error", type=float, default=1.0e-12)
    parser.add_argument("--max-rank-reconstruction-error", type=float, default=1.0e-10)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.70)
    parser.add_argument("--scan-work-ratio", type=float, default=0.08)
    parser.add_argument("--crossing-work-multiplier", type=float, default=1.60)
    parser.add_argument("--rank-work-multiplier", type=float, default=0.50)
    return parser.parse_args()


def quantize(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(values > threshold, 1, np.where(values < -threshold, -1, 0))


def low_rank_block_fixture(
    rng: np.random.Generator,
    args: argparse.Namespace,
    row_count: int,
    col_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    shadow = rng.uniform(-0.25, 0.25, size=(args.rows, args.cols))
    dense_gradient = rng.normal(scale=0.01, size=(args.rows, args.cols))
    rows = rng.choice(args.rows, size=row_count, replace=False)
    cols = rng.choice(args.cols, size=col_count, replace=False)
    row_gap = rng.uniform(0.010, 0.020, size=row_count)
    col_gap = rng.uniform(0.005, 0.012, size=col_count)
    gap = row_gap[:, None] + col_gap[None, :]
    target = args.ternary_threshold + args.boundary_margin
    shadow[np.ix_(rows, cols)] = args.ternary_threshold - gap
    dense_gradient[np.ix_(rows, cols)] = (
        shadow[np.ix_(rows, cols)] - target
    ) / args.learning_rate
    return shadow, dense_gradient


def random_high_rank_fixture(
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    shadow = rng.uniform(-0.25, 0.25, size=(args.rows, args.cols))
    dense_gradient = rng.normal(scale=0.01, size=(args.rows, args.cols))
    mask = rng.uniform(size=(args.rows, args.cols)) < 0.32
    gap = rng.uniform(0.006, 0.030, size=(args.rows, args.cols))
    target = args.ternary_threshold + args.boundary_margin
    shadow[mask] = args.ternary_threshold - gap[mask]
    dense_gradient[mask] = (shadow[mask] - target) / args.learning_rate
    return shadow, dense_gradient


def make_fixture(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    if condition == "dense_fraction_fail":
        return low_rank_block_fixture(rng, args, row_count=24, col_count=24)
    if condition == "high_rank_fraction_fail":
        return random_high_rank_fixture(rng, args)
    return low_rank_block_fixture(rng, args, row_count=4, col_count=8)


def fractional_gradient(
    shadow: np.ndarray,
    dense_after: np.ndarray,
    args: argparse.Namespace,
    condition: str,
) -> np.ndarray:
    start_q = quantize(shadow, args.ternary_threshold)
    dense_q = quantize(dense_after, args.ternary_threshold)
    crossing = dense_q != start_q
    if condition == "epsilon_miss_fail":
        strict_margin = args.boundary_margin + 0.060
        upward = dense_q > start_q
        downward = dense_q < start_q
        crossing = crossing & (
            (upward & (dense_after > args.ternary_threshold + strict_margin))
            | (downward & (dense_after < -args.ternary_threshold - strict_margin))
        )

    target = shadow.copy()
    upward = crossing & (dense_q > start_q)
    downward = crossing & (dense_q < start_q)
    target[upward] = args.ternary_threshold + args.boundary_margin
    target[downward] = -args.ternary_threshold - args.boundary_margin
    gradient = np.zeros_like(shadow)
    gradient[crossing] = (shadow[crossing] - target[crossing]) / args.learning_rate
    return gradient


def matrix_rank_and_error(matrix: np.ndarray, max_rank: int) -> tuple[int, float]:
    norm = float(np.linalg.norm(matrix))
    if norm <= 1.0e-30:
        return 0, 0.0
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * float(np.max(singular))
    rank = int(np.count_nonzero(singular > tolerance))
    kept = min(max_rank, singular.size)
    limited = (u[:, :kept] * singular[:kept]) @ vt[:kept, :]
    error = float(np.linalg.norm(matrix - limited)) / norm
    return rank, error


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    shadow, dense_gradient = make_fixture(rng, args, condition)
    dense_after = shadow - args.learning_rate * dense_gradient
    dense_q = quantize(dense_after, args.ternary_threshold)
    start_q = quantize(shadow, args.ternary_threshold)
    gradient_fraction = fractional_gradient(shadow, dense_after, args, condition)
    fractional_after = shadow - args.learning_rate * gradient_fraction
    fractional_q = quantize(fractional_after, args.ternary_threshold)
    witness = rng.normal(size=shadow.shape)

    crossing_fraction = float(np.count_nonzero(gradient_fraction)) / gradient_fraction.size
    fraction_rank, rank_error = matrix_rank_and_error(gradient_fraction, args.max_rank)
    rank_fraction = fraction_rank / min(args.rows, args.cols)
    mismatch_fraction = float(np.mean(fractional_q != dense_q))
    objective_error = float(np.sum(np.abs((fractional_q - dense_q) * witness))) / max(
        float(np.sum(np.abs(dense_q * witness))),
        1.0,
    )
    priced_work_ratio = (
        args.scan_work_ratio
        + args.crossing_work_multiplier * crossing_fraction
        + args.rank_work_multiplier * rank_fraction
    )
    projected_speedup = 1.0 / priced_work_ratio

    failed: list[str] = []
    if not finite(
        crossing_fraction,
        rank_fraction,
        mismatch_fraction,
        objective_error,
        rank_error,
        priced_work_ratio,
        projected_speedup,
    ):
        failed.append("non_finite_metric")
    if mismatch_fraction > args.max_ternary_mismatch:
        failed.append("ternary_equivalence")
    if objective_error > args.max_objective_error:
        failed.append("objective_equivalence")
    if fraction_rank > args.max_rank or rank_error > args.max_rank_reconstruction_error:
        failed.append("low_rank_reconstruction")
    if priced_work_ratio > args.max_priced_work_ratio:
        failed.append("price")
    if condition == "positive" and np.array_equal(start_q, dense_q):
        failed.append("no_boundary_crossing")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed) if failed else None,
        rows=args.rows,
        cols=args.cols,
        fractional_boundary_fraction=float(crossing_fraction),
        fraction_rank=fraction_rank,
        rank_fraction=float(rank_fraction),
        ternary_mismatch_fraction=float(mismatch_fraction),
        objective_equivalence_error=float(objective_error),
        rank_limited_reconstruction_error=float(rank_error),
        scan_work_ratio=float(args.scan_work_ratio),
        priced_work_ratio=float(priced_work_ratio),
        projected_speedup=float(projected_speedup),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "rows",
        "cols",
        "fractional_boundary_fraction",
        "fraction_rank",
        "rank_fraction",
        "ternary_mismatch_fraction",
        "objective_equivalence_error",
        "rank_limited_reconstruction_error",
        "scan_work_ratio",
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
    epsilon = [result for result in results if result.condition == "epsilon_miss_fail"]
    dense = [result for result in results if result.condition == "dense_fraction_fail"]
    high_rank = [result for result in results if result.condition == "high_rank_fraction_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_boundary": (
            aggregates["positive"]["ternary_mismatch_fraction_max"] == 0.0
            and aggregates["positive"]["objective_equivalence_error_max"] <= 1.0e-12
        ),
        "positive_condition_low_rank_and_priced": (
            aggregates["positive"]["fraction_rank_max"] <= 4.0
            and aggregates["positive"]["rank_limited_reconstruction_error_max"] <= 1.0e-10
            and aggregates["positive"]["priced_work_ratio_mean"] <= 0.70
        ),
        "epsilon_control_all_fail_equivalence": all(
            (not result.passed)
            and result.fail_reason is not None
            and "ternary_equivalence" in result.fail_reason
            and "objective_equivalence" in result.fail_reason
            for result in epsilon
        ),
        "dense_fraction_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "price" in result.fail_reason
            for result in dense
        ),
        "high_rank_control_all_fail_low_rank": all(
            (not result.passed)
            and result.fail_reason is not None
            and "low_rank_reconstruction" in result.fail_reason
            for result in high_rank
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
        "PASS_R501_S63_EMFCR_GATE"
        if all(checks.values())
        else "FAIL_R501_S63_EMFCR_GATE"
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
            "Synthetic CPU matrix-fraction ternary-boundary reconstruction "
            "gate only; not heldout CE progress, not GPU training speed, not "
            "Vulkan benchmark evidence, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S63_EMFCR_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_fractional_boundary_fraction_mean={positive['fractional_boundary_fraction_mean']:.6f} "
        f"positive_fraction_rank_max={positive['fraction_rank_max']:.6f} "
        f"positive_ternary_mismatch_fraction_max={positive['ternary_mismatch_fraction_max']:.6e} "
        f"positive_rank_limited_error_max={positive['rank_limited_reconstruction_error_max']:.6e} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
