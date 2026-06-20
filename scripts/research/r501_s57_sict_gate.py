#!/usr/bin/env python3
"""R501 S57 spline-interpolated cotangent trajectory gate.

This CPU diagnostic implements the R501 Speed Paradigm 9 finding:
"Spline-Interpolated Cotangent Trajectory (SICT)".

For a linear SSM backward recurrence

    lambda_t = g_t + A^T lambda_{t+1},

the candidate computes exact dense cotangents only at knot states and reconstructs
interior cotangents with a cubic interpolant. It survives only when the dense
cotangent trajectory is low-curvature enough that interpolation residual debt is
below the descent budget and the knot/interpolation/verifier route is cheaper
than dense per-step backward evaluation.

Controls exercise the declared risks: high curvature invalidates the spline
residual bound, a hidden interior impulse invalidates sparse verification, and
too many dense knots recover dense cost even when interpolation is exact.

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


SCHEMA = "r501_s57_sict_gate.v1"
CONDITIONS = (
    "positive",
    "high_curvature_fail",
    "interior_impulse_fail",
    "dense_knot_price_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    steps: int
    state_dim: int
    knot_stride: int
    knot_count: int
    residual_debt_ratio: float
    max_point_residual_ratio: float
    sampled_residual_ratio: float
    fourth_difference_ratio: float
    curvature_budget_ratio: float
    descent_alignment_ratio: float
    knot_work_ratio: float
    interpolation_work_ratio: float
    verifier_work_ratio: float
    priced_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50157)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=12)
    parser.add_argument("--knot-stride", type=int, default=8)
    parser.add_argument("--max-residual-debt-ratio", type=float, default=1.0e-10)
    parser.add_argument("--max-sampled-residual-ratio", type=float, default=1.0e-10)
    parser.add_argument("--max-fourth-difference-ratio", type=float, default=1.0e-10)
    parser.add_argument("--min-descent-alignment-ratio", type=float, default=0.999999)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.55)
    parser.add_argument("--interpolation-cost-ratio", type=float, default=0.080)
    parser.add_argument("--verifier-work-ratio", type=float, default=0.075)
    parser.add_argument("--sample-count", type=int, default=16)
    return parser.parse_args()


def condition_knot_stride(args: argparse.Namespace, condition: str) -> int:
    if condition == "dense_knot_price_fail":
        return 2
    return args.knot_stride


def make_transition(rng: np.random.Generator, dim: int) -> np.ndarray:
    raw = rng.normal(size=(dim, dim))
    q, _ = np.linalg.qr(raw)
    eig = rng.uniform(0.35, 0.72, size=dim) * rng.choice(np.array([-1.0, 1.0]), size=dim)
    return (q @ np.diag(eig) @ q.T).astype(np.float64)


def make_cotangent_trajectory(
    rng: np.random.Generator,
    steps: int,
    dim: int,
    condition: str,
) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, steps + 1, dtype=np.float64)
    coeff = rng.normal(scale=np.array([0.42, 0.25, 0.12, 0.045])[:, None], size=(4, dim))
    trajectory = (
        coeff[0][None, :]
        + coeff[1][None, :] * x[:, None]
        + coeff[2][None, :] * (x[:, None] ** 2)
        + coeff[3][None, :] * (x[:, None] ** 3)
    )
    if condition == "high_curvature_fail":
        freq = rng.integers(10, 15)
        phase = rng.uniform(-math.pi, math.pi, size=dim)
        trajectory += 0.18 * np.sin(freq * math.pi * x[:, None] + phase[None, :])
    elif condition == "interior_impulse_fail":
        center = steps // 2 + 3
        width = 1.7
        direction = rng.normal(size=dim)
        direction /= max(float(np.linalg.norm(direction)), 1.0e-30)
        bump = np.exp(-0.5 * ((np.arange(steps + 1) - center) / width) ** 2)
        trajectory += 0.55 * bump[:, None] * direction[None, :]
    return trajectory.astype(np.float64)


def make_sources(a: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    steps = lambdas.shape[0] - 1
    source = np.zeros_like(lambdas, dtype=np.float64)
    source[steps] = lambdas[steps]
    for t in range(steps):
        source[t] = lambdas[t] - a.T @ lambdas[t + 1]
    return source


def dense_backward(a: np.ndarray, source: np.ndarray) -> np.ndarray:
    steps = source.shape[0] - 1
    lambdas = np.zeros_like(source, dtype=np.float64)
    lambdas[steps] = source[steps]
    for t in range(steps - 1, -1, -1):
        lambdas[t] = source[t] + a.T @ lambdas[t + 1]
    return lambdas


def knot_indices(steps: int, stride: int) -> np.ndarray:
    knots = list(range(0, steps + 1, stride))
    if knots[-1] != steps:
        knots.append(steps)
    return np.array(knots, dtype=np.int64)


def cubic_lagrange(xs: np.ndarray, ys: np.ndarray, x: float) -> np.ndarray:
    out = np.zeros(ys.shape[1], dtype=np.float64)
    for i in range(4):
        basis = 1.0
        for j in range(4):
            if i == j:
                continue
            basis *= (x - xs[j]) / (xs[i] - xs[j])
        out += basis * ys[i]
    return out


def interpolate_from_knots(
    dense_lambdas: np.ndarray,
    knots: np.ndarray,
) -> np.ndarray:
    steps = dense_lambdas.shape[0] - 1
    if knots.size < 4:
        raise ValueError("at least four knots are required for cubic interpolation")
    interpolated = np.zeros_like(dense_lambdas, dtype=np.float64)
    knot_values = dense_lambdas[knots]
    for t in range(steps + 1):
        if t in set(knots.tolist()):
            interpolated[t] = dense_lambdas[t]
            continue
        pos = int(np.searchsorted(knots, t))
        start = min(max(pos - 2, 0), knots.size - 4)
        xs = knots[start : start + 4].astype(np.float64)
        ys = knot_values[start : start + 4]
        interpolated[t] = cubic_lagrange(xs, ys, float(t))
    return interpolated


def fourth_difference_ratio(values: np.ndarray) -> float:
    fourth = np.diff(values, n=4, axis=0)
    numerator = float(np.linalg.norm(fourth))
    denominator = max(float(np.linalg.norm(values)), 1.0e-30)
    return numerator / denominator


def residual_ratios(
    dense: np.ndarray,
    reconstructed: np.ndarray,
    samples: np.ndarray,
) -> tuple[float, float, float, float]:
    residual = reconstructed - dense
    residual_debt = float(np.linalg.norm(residual)) / max(float(np.linalg.norm(dense)), 1.0e-30)
    point_norm = np.linalg.norm(dense, axis=1)
    point_residual = np.linalg.norm(residual, axis=1) / np.maximum(point_norm, 1.0e-30)
    sampled = float(np.linalg.norm(residual[samples])) / max(float(np.linalg.norm(dense[samples])), 1.0e-30)
    alignment = float(np.sum(reconstructed * dense)) / max(float(np.sum(dense * dense)), 1.0e-30)
    return residual_debt, float(np.max(point_residual)), sampled, alignment


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    stride = condition_knot_stride(args, condition)
    a = make_transition(rng, args.state_dim)
    target_lambdas = make_cotangent_trajectory(rng, args.steps, args.state_dim, condition)
    source = make_sources(a, target_lambdas)
    dense_lambdas = dense_backward(a, source)
    knots = knot_indices(args.steps, stride)
    reconstructed = interpolate_from_knots(dense_lambdas, knots)
    sample_count = min(args.sample_count, args.steps + 1)
    sample_pool = np.setdiff1d(np.arange(args.steps + 1), knots, assume_unique=False)
    if sample_pool.size < sample_count:
        samples = sample_pool
    else:
        samples = np.sort(rng.choice(sample_pool, size=sample_count, replace=False))

    residual_debt, max_point_residual, sampled_residual, descent_alignment = residual_ratios(
        dense_lambdas,
        reconstructed,
        samples,
    )
    curvature = fourth_difference_ratio(dense_lambdas)
    knot_work_ratio = knots.size / (args.steps + 1)
    interpolation_work_ratio = args.interpolation_cost_ratio * (
        (args.steps + 1 - knots.size) / (args.steps + 1)
    )
    verifier_work_ratio = args.verifier_work_ratio
    priced_work_ratio = knot_work_ratio + interpolation_work_ratio + verifier_work_ratio
    projected_speedup = 1.0 / priced_work_ratio if priced_work_ratio > 0.0 else math.inf

    failed: list[str] = []
    if not all(
        math.isfinite(value)
        for value in (
            residual_debt,
            max_point_residual,
            sampled_residual,
            curvature,
            descent_alignment,
            priced_work_ratio,
        )
    ):
        failed.append("non_finite_metric")
    if residual_debt > args.max_residual_debt_ratio:
        failed.append("residual_debt")
    if sampled_residual > args.max_sampled_residual_ratio:
        failed.append("sampled_verifier_residual")
    if curvature > args.max_fourth_difference_ratio:
        failed.append("curvature_budget")
    if descent_alignment < args.min_descent_alignment_ratio:
        failed.append("descent_alignment")
    if priced_work_ratio > args.max_priced_work_ratio:
        failed.append("price")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=(passed == expected_pass),
        fail_reason=";".join(failed) if failed else None,
        steps=args.steps,
        state_dim=args.state_dim,
        knot_stride=stride,
        knot_count=int(knots.size),
        residual_debt_ratio=float(residual_debt),
        max_point_residual_ratio=float(max_point_residual),
        sampled_residual_ratio=float(sampled_residual),
        fourth_difference_ratio=float(curvature),
        curvature_budget_ratio=float(curvature / max(args.max_fourth_difference_ratio, 1.0e-30)),
        descent_alignment_ratio=float(descent_alignment),
        knot_work_ratio=float(knot_work_ratio),
        interpolation_work_ratio=float(interpolation_work_ratio),
        verifier_work_ratio=float(verifier_work_ratio),
        priced_work_ratio=float(priced_work_ratio),
        projected_speedup=float(projected_speedup),
    )


def aggregate(results: list[TrialResult]) -> dict[str, Any]:
    by_condition: dict[str, list[TrialResult]] = {
        condition: [result for result in results if result.condition == condition]
        for condition in CONDITIONS
    }
    numeric_fields = (
        "knot_stride",
        "knot_count",
        "residual_debt_ratio",
        "max_point_residual_ratio",
        "sampled_residual_ratio",
        "fourth_difference_ratio",
        "curvature_budget_ratio",
        "descent_alignment_ratio",
        "knot_work_ratio",
        "interpolation_work_ratio",
        "verifier_work_ratio",
        "priced_work_ratio",
        "projected_speedup",
    )
    aggregates: dict[str, Any] = {}
    for condition, condition_results in by_condition.items():
        aggregates[condition] = {
            "count": len(condition_results),
            "passed_count": sum(result.passed for result in condition_results),
            "identifying_correct_count": sum(
                result.identifying_correct for result in condition_results
            ),
            "fail_reasons": sorted(
                {result.fail_reason for result in condition_results if result.fail_reason}
            ),
        }
        for field in numeric_fields:
            values = [float(getattr(result, field)) for result in condition_results]
            if not values:
                continue
            aggregates[condition][f"{field}_mean"] = float(np.mean(values))
            aggregates[condition][f"{field}_max"] = float(np.max(values))
            aggregates[condition][f"{field}_min"] = float(np.min(values))
    return aggregates


def all_finite(results: list[TrialResult]) -> bool:
    for result in results:
        for value in asdict(result).values():
            if isinstance(value, float) and not math.isfinite(value):
                return False
    return True


def required_checks(results: list[TrialResult], aggregates: dict[str, Any]) -> dict[str, bool]:
    positive = [result for result in results if result.condition == "positive"]
    high_curvature = [result for result in results if result.condition == "high_curvature_fail"]
    impulse = [result for result in results if result.condition == "interior_impulse_fail"]
    dense_knot = [result for result in results if result.condition == "dense_knot_price_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_spline": (
            aggregates["positive"]["residual_debt_ratio_max"] <= 1.0e-10
            and aggregates["positive"]["sampled_residual_ratio_max"] <= 1.0e-10
            and aggregates["positive"]["fourth_difference_ratio_max"] <= 1.0e-10
        ),
        "positive_condition_price": aggregates["positive"]["priced_work_ratio_mean"] <= 0.55,
        "high_curvature_control_all_fail_curvature": all(
            (not result.passed)
            and result.fail_reason is not None
            and "curvature_budget" in result.fail_reason
            and "residual_debt" in result.fail_reason
            for result in high_curvature
        ),
        "interior_impulse_control_all_fail_verifier": all(
            (not result.passed)
            and result.fail_reason is not None
            and "sampled_verifier_residual" in result.fail_reason
            and "curvature_budget" in result.fail_reason
            for result in impulse
        ),
        "dense_knot_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "price" in result.fail_reason
            for result in dense_knot
        ),
        "all_conditions_identified": all(result.identifying_correct for result in results),
    }


def main() -> int:
    args = parse_args()
    results: list[TrialResult] = []
    for condition in CONDITIONS:
        for offset in range(args.seed_count):
            results.append(run_trial(args, condition, args.seed + 1000 * offset))

    aggregates = aggregate(results)
    checks = required_checks(results, aggregates)
    decision = "PASS_R501_S57_SICT_GATE" if all(checks.values()) else "FAIL_R501_S57_SICT_GATE"
    payload = {
        "schema": SCHEMA,
        "decision": decision,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "conditions": CONDITIONS,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU spline-interpolated cotangent trajectory gate only; "
            "not heldout CE progress, not GPU training speed, and not "
            "production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S57_SICT_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_knot_stride_mean={positive['knot_stride_mean']:.6f} "
        f"positive_residual_debt_max={positive['residual_debt_ratio_max']:.6e} "
        f"positive_sampled_residual_max={positive['sampled_residual_ratio_max']:.6e} "
        f"positive_fourth_difference_max={positive['fourth_difference_ratio_max']:.6e} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
