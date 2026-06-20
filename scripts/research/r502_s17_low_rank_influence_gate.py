#!/usr/bin/env python3
"""R502-S17 Forward-Propagated Low-Rank Influence Matrix gate.

This synthetic CPU gate checks whether an online low-rank influence sketch can
track exact recurrent parameter influence well enough to replace dense BPTT
work in the constructed regime. It also includes controls for high-rank
influence, biased gradients, expensive sketch rank, stale basis transfer, and
unstable updates.
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
    "high_rank_fail",
    "biased_gradient_fail",
    "rank_cost_fail",
    "stale_basis_fail",
    "unstable_update_fail",
]


@dataclass
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str
    sequence_len: int
    state_dim: int
    param_dim: int
    sketch_rank: int
    exact_update_rank: int
    gradient_relative_error_max: float
    heldout_gradient_relative_error_max: float
    matrix_residual_fraction_max: float
    heldout_matrix_residual_fraction_max: float
    descent_cosine_min: float
    heldout_descent_cosine_min: float
    descent_ratio_min: float
    descent_per_paid_work_min: float
    priced_work_ratio: float
    memory_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=50217)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sequence-len", type=int, default=192)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--param-dim", type=int, default=64)
    parser.add_argument("--sketch-rank", type=int, default=4)
    parser.add_argument("--high-rank", type=int, default=16)
    parser.add_argument("--expensive-sketch-rank", type=int, default=28)
    parser.add_argument("--stable-decay", type=float, default=0.82)
    parser.add_argument("--unstable-decay", type=float, default=1.04)
    parser.add_argument("--unstable-noise-scale", type=float, default=0.22)
    parser.add_argument("--sketch-overhead-ratio", type=float, default=0.04)
    parser.add_argument("--max-gradient-relative-error", type=float, default=0.05)
    parser.add_argument("--max-heldout-gradient-relative-error", type=float, default=0.08)
    parser.add_argument("--max-matrix-residual-fraction", type=float, default=0.08)
    parser.add_argument("--min-descent-cosine", type=float, default=0.98)
    parser.add_argument("--min-heldout-descent-cosine", type=float, default=0.95)
    parser.add_argument("--min-descent-per-paid-work", type=float, default=1.0)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.90)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def orthonormal(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    raw = rng.normal(size=(rows, cols))
    q, _ = np.linalg.qr(raw)
    return q[:, :cols]


def truncate_rank(matrix: np.ndarray, rank: int) -> np.ndarray:
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    keep = min(rank, s.size)
    return (u[:, :keep] * s[:keep]) @ vh[:keep, :]


def project_to_basis(matrix: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left @ (left.T @ matrix @ right) @ right.T


def relative_norm(numerator: np.ndarray, denominator: np.ndarray) -> float:
    denom = float(np.linalg.norm(denominator))
    if denom == 0.0:
        return 0.0 if float(np.linalg.norm(numerator)) == 0.0 else math.inf
    return float(np.linalg.norm(numerator) / denom)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def make_update(
    rng: np.random.Generator,
    left: np.ndarray,
    right: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    rank = left.shape[1]
    coeff = rng.normal(scale=scale, size=rank)
    return (left * coeff[None, :]) @ right.T


def priced_work_ratio(args: argparse.Namespace, rank: int) -> float:
    dense_units = args.state_dim * args.param_dim
    sketch_units = rank * (args.state_dim + args.param_dim) + rank * rank
    return float(sketch_units / dense_units + args.sketch_overhead_ratio)


def memory_ratio(args: argparse.Namespace, rank: int) -> float:
    dense_units = args.state_dim * args.param_dim
    sketch_units = rank * (args.state_dim + args.param_dim)
    return float(sketch_units / dense_units)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    base_rank = args.sketch_rank
    sketch_rank = args.expensive_sketch_rank if condition == "rank_cost_fail" else base_rank
    update_rank = args.high_rank if condition == "high_rank_fail" else base_rank
    decay = args.unstable_decay if condition == "unstable_update_fail" else args.stable_decay

    left = orthonormal(rng, args.state_dim, max(update_rank, sketch_rank))
    right = orthonormal(rng, args.param_dim, max(update_rank, sketch_rank))
    shifted_left = orthonormal(rng, args.state_dim, base_rank)
    shifted_right = orthonormal(rng, args.param_dim, base_rank)
    noise_left = orthonormal(rng, args.state_dim, args.high_rank)
    noise_right = orthonormal(rng, args.param_dim, args.high_rank)

    exact = np.zeros((args.state_dim, args.param_dim), dtype=np.float64)
    sketch = np.zeros_like(exact)
    split = args.sequence_len // 2

    grad_errors: list[float] = []
    heldout_grad_errors: list[float] = []
    residuals: list[float] = []
    heldout_residuals: list[float] = []
    cosines: list[float] = []
    heldout_cosines: list[float] = []
    descent_ratios: list[float] = []

    for t in range(args.sequence_len):
        if condition == "stale_basis_fail" and t >= split:
            update = make_update(rng, shifted_left, shifted_right)
        else:
            update = make_update(rng, left[:, :update_rank], right[:, :update_rank])

        if condition == "unstable_update_fail":
            update = update + make_update(
                rng,
                noise_left,
                noise_right,
                scale=args.unstable_noise_scale,
            )

        exact = decay * exact + update
        raw_sketch = decay * sketch + update
        if condition == "stale_basis_fail" and t >= split:
            sketch = project_to_basis(raw_sketch, left[:, :base_rank], right[:, :base_rank])
        else:
            sketch = truncate_rank(raw_sketch, sketch_rank)

        loss_grad = rng.normal(size=args.state_dim)
        grad_exact = exact.T @ loss_grad
        sketch_for_grad = -0.40 * sketch if condition == "biased_gradient_fail" else sketch
        grad_sketch = sketch_for_grad.T @ loss_grad

        grad_error = relative_norm(grad_exact - grad_sketch, grad_exact)
        residual = relative_norm(exact - sketch_for_grad, exact)
        direction = cosine(grad_sketch, grad_exact)
        denom = float(np.dot(grad_exact, grad_exact))
        descent_ratio = 0.0 if denom == 0.0 else float(np.dot(grad_sketch, grad_exact) / denom)

        grad_errors.append(grad_error)
        residuals.append(residual)
        cosines.append(direction)
        descent_ratios.append(descent_ratio)
        if t >= split:
            heldout_grad_errors.append(grad_error)
            heldout_residuals.append(residual)
            heldout_cosines.append(direction)

    price = priced_work_ratio(args, sketch_rank)
    min_descent_ratio = min(descent_ratios)
    descent_per_paid_work = min_descent_ratio / price

    gradient_error_max = max(grad_errors)
    heldout_gradient_error_max = max(heldout_grad_errors)
    residual_max = max(residuals)
    heldout_residual_max = max(heldout_residuals)
    descent_cosine_min = min(cosines)
    heldout_descent_cosine_min = min(heldout_cosines)

    failed: list[str] = []
    if not np.isfinite(
        [
            gradient_error_max,
            heldout_gradient_error_max,
            residual_max,
            heldout_residual_max,
            descent_cosine_min,
            heldout_descent_cosine_min,
            min_descent_ratio,
            descent_per_paid_work,
            price,
        ]
    ).all():
        failed.append("non_finite_metric")
    if gradient_error_max > args.max_gradient_relative_error:
        failed.append("gradient_error")
    if heldout_gradient_error_max > args.max_heldout_gradient_relative_error:
        failed.append("heldout_gradient_error")
    if residual_max > args.max_matrix_residual_fraction:
        failed.append("sketch_residual")
    if descent_cosine_min < args.min_descent_cosine:
        failed.append("descent_alignment")
    if heldout_descent_cosine_min < args.min_heldout_descent_cosine:
        failed.append("heldout_descent_alignment")
    if descent_per_paid_work < args.min_descent_per_paid_work:
        failed.append("online_descent_per_paid_work")
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
        sequence_len=args.sequence_len,
        state_dim=args.state_dim,
        param_dim=args.param_dim,
        sketch_rank=sketch_rank,
        exact_update_rank=update_rank,
        gradient_relative_error_max=gradient_error_max,
        heldout_gradient_relative_error_max=heldout_gradient_error_max,
        matrix_residual_fraction_max=residual_max,
        heldout_matrix_residual_fraction_max=heldout_residual_max,
        descent_cosine_min=descent_cosine_min,
        heldout_descent_cosine_min=heldout_descent_cosine_min,
        descent_ratio_min=min_descent_ratio,
        descent_per_paid_work_min=descent_per_paid_work,
        priced_work_ratio=price,
        memory_ratio=memory_ratio(args, sketch_rank),
    )


NUMERIC_FIELDS = [
    "sequence_len",
    "state_dim",
    "param_dim",
    "sketch_rank",
    "exact_update_rank",
    "gradient_relative_error_max",
    "heldout_gradient_relative_error_max",
    "matrix_residual_fraction_max",
    "heldout_matrix_residual_fraction_max",
    "descent_cosine_min",
    "heldout_descent_cosine_min",
    "descent_ratio_min",
    "descent_per_paid_work_min",
    "priced_work_ratio",
    "memory_ratio",
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


def required_checks(results: list[TrialResult], aggregates: dict[str, Any]) -> dict[str, bool]:
    by_condition = {
        condition: [result for result in results if result.condition == condition]
        for condition in CONDITIONS
    }
    positive = by_condition["positive"]
    high_rank = by_condition["high_rank_fail"]
    biased = by_condition["biased_gradient_fail"]
    rank_cost = by_condition["rank_cost_fail"]
    stale = by_condition["stale_basis_fail"]
    unstable = by_condition["unstable_update_fail"]

    return {
        "all_metrics_finite": all(
            math.isfinite(float(value))
            for result in results
            for value in asdict(result).values()
            if isinstance(value, (int, float, np.floating))
        ),
        "all_conditions_identified": all(result.identifying_correct for result in results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_gradient_bound": (
            aggregates["positive"]["gradient_relative_error_max_max"] <= 0.05
            and aggregates["positive"]["heldout_gradient_relative_error_max_max"] <= 0.08
            and aggregates["positive"]["matrix_residual_fraction_max_max"] <= 0.08
        ),
        "positive_condition_descent_and_price": (
            aggregates["positive"]["descent_cosine_min_min"] >= 0.98
            and aggregates["positive"]["heldout_descent_cosine_min_min"] >= 0.95
            and aggregates["positive"]["descent_per_paid_work_min_min"] >= 1.0
            and aggregates["positive"]["priced_work_ratio_max"] <= 0.90
        ),
        "high_rank_control_all_fail_residual": all(
            (not result.passed)
            and (
                "gradient_error" in result.fail_reason
                or "sketch_residual" in result.fail_reason
                or "descent_alignment" in result.fail_reason
            )
            for result in high_rank
        ),
        "biased_gradient_control_all_fail_descent": all(
            (not result.passed)
            and (
                "descent_alignment" in result.fail_reason
                or "online_descent_per_paid_work" in result.fail_reason
                or "gradient_error" in result.fail_reason
            )
            for result in biased
        ),
        "rank_cost_control_all_fail_price": all(
            (not result.passed) and "priced_work" in result.fail_reason
            for result in rank_cost
        ),
        "stale_basis_control_all_fail_heldout": all(
            (not result.passed)
            and (
                "heldout_gradient_error" in result.fail_reason
                or "heldout_descent_alignment" in result.fail_reason
                or "sketch_residual" in result.fail_reason
            )
            for result in stale
        ),
        "unstable_update_control_all_fail_residual": all(
            (not result.passed)
            and (
                "gradient_error" in result.fail_reason
                or "sketch_residual" in result.fail_reason
                or "heldout_gradient_error" in result.fail_reason
            )
            for result in unstable
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
    checks = required_checks(results, aggregates)
    decision = "PASS_R502_S17_LOW_RANK_INFLUENCE_GATE" if all(checks.values()) else "FAIL_R502_S17_LOW_RANK_INFLUENCE_GATE"

    payload = {
        "round": "R502",
        "canonical_id": "R502-S17",
        "aspect": "Speed",
        "finding": "Forward-Propagated Low-Rank Influence Matrix",
        "decision": decision,
        "disposition": "synthetic_positive_control_identified_not_training_speed_claim",
        "args": vars(args) | {"output": str(args.output) if args.output else None},
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU low-rank influence-sketch gate only; not heldout CE "
            "progress, not real corpus token efficiency, not GPU training speed, "
            "and not production v12 integration."
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_S17_LOW_RANK_INFLUENCE_GATE] "
        f"active=1 decision={decision} "
        f"disposition={payload['disposition']} "
        f"positive_gradient_error_max={positive['gradient_relative_error_max_max']:.12e} "
        f"positive_heldout_gradient_error_max={positive['heldout_gradient_relative_error_max_max']:.12e} "
        f"positive_descent_cosine_min={positive['descent_cosine_min_min']:.12e} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.12f} "
        f"positive_descent_per_paid_work_min={positive['descent_per_paid_work_min_min']:.12f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
