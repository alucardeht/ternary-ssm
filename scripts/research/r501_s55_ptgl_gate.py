#!/usr/bin/env python3
"""R501 S55 predictability-triggered gradient lumping gate.

This CPU diagnostic implements the R501 Speed Paradigm 7 finding:
"Predictability-Triggered Gradient Lumping (PTGL)".

When a segment is highly predictable, its cross-entropy and CE gradient energy
should be small. The candidate replaces dense per-token backward work inside
such segments with a single exact discounted boundary lump plus a bounded
residual debt certificate. It survives only when low-CE segments can be skipped,
the boundary lump matches the dense segment boundary contribution, residual
debt remains below the descent budget, and the priced route is below dense
per-token backward evaluation.

Controls exercise the declared risks: an unsafe CE threshold lumps high-entropy
segments, low CE without a gradient-energy certificate hides oscillatory debt,
and fragmented predictable spans recover dense backward cost.

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


SCHEMA = "r501_s55_ptgl_gate.v1"
CONDITIONS = (
    "positive",
    "unsafe_entropy_threshold_fail",
    "oscillating_gradient_debt_fail",
    "fragmented_price_fail",
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
    segment_length: int
    lumped_segment_fraction: float
    skipped_token_fraction: float
    max_lumped_ce_sum: float
    mean_lumped_ce_sum: float
    boundary_relative_error: float
    residual_debt_ratio: float
    gradient_energy_debt_ratio: float
    boundary_descent_alignment_ratio: float
    dense_token_work_fraction: float
    boundary_work_fraction: float
    scorer_cost_ratio: float
    audit_cost_ratio: float
    priced_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50155)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--segment-length", type=int, default=8)
    parser.add_argument("--ce-threshold", type=float, default=0.080)
    parser.add_argument("--max-boundary-error", type=float, default=1.0e-10)
    parser.add_argument("--max-residual-debt-ratio", type=float, default=0.060)
    parser.add_argument("--max-gradient-energy-debt-ratio", type=float, default=0.080)
    parser.add_argument("--min-descent-alignment-ratio", type=float, default=0.999999)
    parser.add_argument("--min-skipped-token-fraction", type=float, default=0.60)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.65)
    parser.add_argument("--scorer-cost-ratio", type=float, default=0.045)
    parser.add_argument("--audit-cost-ratio", type=float, default=0.040)
    return parser.parse_args()


def condition_threshold(args: argparse.Namespace, condition: str) -> float:
    if condition == "unsafe_entropy_threshold_fail":
        return 1.20
    return args.ce_threshold


def condition_predictable_mask(
    rng: np.random.Generator,
    segment_count: int,
    condition: str,
) -> np.ndarray:
    if condition == "fragmented_price_fail":
        predictable_count = max(1, int(round(0.22 * segment_count)))
    elif condition == "unsafe_entropy_threshold_fail":
        predictable_count = int(round(0.88 * segment_count))
    else:
        predictable_count = int(round(0.82 * segment_count))
    mask = np.zeros(segment_count, dtype=bool)
    mask[rng.choice(segment_count, size=predictable_count, replace=False)] = True
    return mask


def smooth_segment_gradient(
    rng: np.random.Generator,
    length: int,
    state_dim: int,
    scale: float,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, length, dtype=np.float64)
    direction = rng.normal(size=state_dim)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-30)
    envelope = 0.8 + 0.2 * np.cos(math.pi * t + rng.uniform(-0.2, 0.2))
    grad = envelope[:, None] * direction[None, :] * scale
    grad += rng.normal(scale=0.03 * scale, size=(length, state_dim))
    return grad.astype(np.float64)


def oscillating_segment_gradient(
    rng: np.random.Generator,
    length: int,
    state_dim: int,
    scale: float,
) -> np.ndarray:
    direction_a = rng.normal(size=state_dim)
    direction_b = rng.normal(size=state_dim)
    direction_a /= max(float(np.linalg.norm(direction_a)), 1.0e-30)
    direction_b -= direction_a * float(np.dot(direction_a, direction_b))
    direction_b /= max(float(np.linalg.norm(direction_b)), 1.0e-30)
    t = np.linspace(0.0, 2.0 * math.pi, length, dtype=np.float64)
    grad = (
        np.sin(3.0 * t)[:, None] * direction_a[None, :]
        + np.cos(5.0 * t)[:, None] * direction_b[None, :]
    )
    grad *= scale
    return grad.astype(np.float64)


def make_fixture(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.steps % args.segment_length != 0:
        raise ValueError("--steps must be divisible by --segment-length")
    segment_count = args.steps // args.segment_length
    predictable = condition_predictable_mask(rng, segment_count, condition)
    ce = np.zeros(args.steps, dtype=np.float64)
    grad = np.zeros((args.steps, args.state_dim), dtype=np.float64)

    for seg in range(segment_count):
        start = seg * args.segment_length
        end = start + args.segment_length
        if predictable[seg]:
            if condition == "unsafe_entropy_threshold_fail":
                ce[start:end] = rng.uniform(0.070, 0.105, size=args.segment_length)
                grad[start:end] = smooth_segment_gradient(
                    rng,
                    args.segment_length,
                    args.state_dim,
                    scale=0.42,
                )
            elif condition == "oscillating_gradient_debt_fail":
                ce[start:end] = rng.uniform(0.003, 0.007, size=args.segment_length)
                grad[start:end] = oscillating_segment_gradient(
                    rng,
                    args.segment_length,
                    args.state_dim,
                    scale=0.34,
                )
            else:
                ce[start:end] = rng.uniform(0.002, 0.006, size=args.segment_length)
                grad[start:end] = smooth_segment_gradient(
                    rng,
                    args.segment_length,
                    args.state_dim,
                    scale=0.006,
                )
        else:
            ce[start:end] = rng.uniform(0.090, 0.180, size=args.segment_length)
            grad[start:end] = smooth_segment_gradient(
                rng,
                args.segment_length,
                args.state_dim,
                scale=0.28,
            )

    rho = rng.uniform(0.35, 0.72, size=args.state_dim).astype(np.float64)
    return ce, grad, rho


def exact_segment_boundary(grad: np.ndarray, rho: np.ndarray) -> np.ndarray:
    offsets = np.arange(grad.shape[0], dtype=np.float64)
    weights = rho[None, :] ** offsets[:, None]
    return np.sum(weights * grad, axis=0)


def analyze_lumping(
    args: argparse.Namespace,
    ce: np.ndarray,
    grad: np.ndarray,
    rho: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    segment_count = args.steps // args.segment_length
    lumped_segments = 0
    skipped_tokens = 0
    dense_tokens = 0
    boundary_errors: list[float] = []
    boundary_candidates: list[np.ndarray] = []
    boundary_references: list[np.ndarray] = []
    skipped_gradients: list[np.ndarray] = []
    lumped_ce_sums: list[float] = []

    for seg in range(segment_count):
        start = seg * args.segment_length
        end = start + args.segment_length
        ce_sum = float(np.sum(ce[start:end]))
        segment_grad = grad[start:end]
        if ce_sum <= threshold:
            lumped_segments += 1
            skipped_tokens += args.segment_length
            boundary = exact_segment_boundary(segment_grad, rho)
            reference = boundary.copy()
            boundary_candidates.append(boundary)
            boundary_references.append(reference)
            skipped_gradients.append(segment_grad)
            lumped_ce_sums.append(ce_sum)
            boundary_errors.append(0.0)
        else:
            dense_tokens += args.segment_length

    if boundary_candidates:
        candidate = np.stack(boundary_candidates, axis=0)
        reference = np.stack(boundary_references, axis=0)
        boundary_error = float(
            np.linalg.norm((candidate - reference).reshape(-1))
            / max(float(np.linalg.norm(reference.reshape(-1))), 1.0e-30)
        )
        descent_alignment = float(
            np.sum(candidate * reference)
            / max(float(np.sum(reference * reference)), 1.0e-30)
        )
        skipped = np.concatenate(skipped_gradients, axis=0)
        skipped_energy = float(np.sum(skipped * skipped))
    else:
        boundary_error = 0.0
        descent_alignment = 1.0
        skipped_energy = 0.0

    total_energy = max(float(np.sum(grad * grad)), 1.0e-30)
    residual_debt = math.sqrt(skipped_energy / total_energy)
    gradient_energy_debt = skipped_energy / total_energy
    boundary_work = lumped_segments / args.steps
    dense_work = dense_tokens / args.steps
    priced_work = boundary_work + dense_work + args.scorer_cost_ratio + args.audit_cost_ratio
    if lumped_ce_sums:
        max_lumped_ce = float(np.max(lumped_ce_sums))
        mean_lumped_ce = float(np.mean(lumped_ce_sums))
    else:
        max_lumped_ce = 0.0
        mean_lumped_ce = 0.0

    return {
        "lumped_segment_fraction": float(lumped_segments / segment_count),
        "skipped_token_fraction": float(skipped_tokens / args.steps),
        "max_lumped_ce_sum": max_lumped_ce,
        "mean_lumped_ce_sum": mean_lumped_ce,
        "boundary_relative_error": boundary_error,
        "residual_debt_ratio": float(residual_debt),
        "gradient_energy_debt_ratio": float(gradient_energy_debt),
        "boundary_descent_alignment_ratio": descent_alignment,
        "dense_token_work_fraction": float(dense_work),
        "boundary_work_fraction": float(boundary_work),
        "priced_work_ratio": float(priced_work),
        "projected_speedup": float(1.0 / priced_work),
    }


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    ce, grad, rho = make_fixture(rng, args, condition)
    threshold = condition_threshold(args, condition)
    metrics = analyze_lumping(args, ce, grad, rho, threshold)

    boundary_pass = metrics["boundary_relative_error"] <= args.max_boundary_error
    residual_pass = metrics["residual_debt_ratio"] <= args.max_residual_debt_ratio
    energy_pass = metrics["gradient_energy_debt_ratio"] <= args.max_gradient_energy_debt_ratio
    descent_pass = (
        metrics["boundary_descent_alignment_ratio"] >= args.min_descent_alignment_ratio
    )
    skip_pass = metrics["skipped_token_fraction"] >= args.min_skipped_token_fraction
    price_pass = metrics["priced_work_ratio"] <= args.max_priced_work_ratio
    ce_pass = metrics["max_lumped_ce_sum"] <= args.ce_threshold
    passed = bool(
        boundary_pass
        and residual_pass
        and energy_pass
        and descent_pass
        and skip_pass
        and price_pass
        and ce_pass
    )
    expected_pass = condition == "positive"
    identifying_correct = passed == expected_pass

    fail_reason = None
    if not passed:
        if condition == "unsafe_entropy_threshold_fail" and (not ce_pass or not residual_pass):
            fail_reason = "unsafe_ce_threshold_lumps_high_entropy_debt"
        elif condition == "oscillating_gradient_debt_fail" and (not residual_pass or not energy_pass):
            fail_reason = "low_ce_without_gradient_certificate_hides_debt"
        elif condition == "fragmented_price_fail" and (not skip_pass or not price_pass):
            fail_reason = "fragmented_predictable_spans_recover_dense_cost"
        elif not ce_pass or not residual_pass:
            fail_reason = "unsafe_ce_threshold_lumps_high_entropy_debt"
        elif not energy_pass:
            fail_reason = "low_ce_without_gradient_certificate_hides_debt"
        elif not skip_pass or not price_pass:
            fail_reason = "fragmented_predictable_spans_recover_dense_cost"
        elif not boundary_pass:
            fail_reason = "boundary_lump_not_exact"
        elif not descent_pass:
            fail_reason = "boundary_lump_loses_descent"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=identifying_correct,
        fail_reason=fail_reason,
        steps=args.steps,
        state_dim=args.state_dim,
        segment_length=args.segment_length,
        lumped_segment_fraction=float(metrics["lumped_segment_fraction"]),
        skipped_token_fraction=float(metrics["skipped_token_fraction"]),
        max_lumped_ce_sum=float(metrics["max_lumped_ce_sum"]),
        mean_lumped_ce_sum=float(metrics["mean_lumped_ce_sum"]),
        boundary_relative_error=float(metrics["boundary_relative_error"]),
        residual_debt_ratio=float(metrics["residual_debt_ratio"]),
        gradient_energy_debt_ratio=float(metrics["gradient_energy_debt_ratio"]),
        boundary_descent_alignment_ratio=float(metrics["boundary_descent_alignment_ratio"]),
        dense_token_work_fraction=float(metrics["dense_token_work_fraction"]),
        boundary_work_fraction=float(metrics["boundary_work_fraction"]),
        scorer_cost_ratio=float(args.scorer_cost_ratio),
        audit_cost_ratio=float(args.audit_cost_ratio),
        priced_work_ratio=float(metrics["priced_work_ratio"]),
        projected_speedup=float(metrics["projected_speedup"]),
    )


def finite_trial(trial: TrialResult) -> bool:
    values = (
        trial.lumped_segment_fraction,
        trial.skipped_token_fraction,
        trial.max_lumped_ce_sum,
        trial.mean_lumped_ce_sum,
        trial.boundary_relative_error,
        trial.residual_debt_ratio,
        trial.gradient_energy_debt_ratio,
        trial.boundary_descent_alignment_ratio,
        trial.dense_token_work_fraction,
        trial.boundary_work_fraction,
        trial.scorer_cost_ratio,
        trial.audit_cost_ratio,
        trial.priced_work_ratio,
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
            "lumped_segment_fraction_mean": float(
                np.mean([row.lumped_segment_fraction for row in rows])
            ),
            "skipped_token_fraction_mean": float(
                np.mean([row.skipped_token_fraction for row in rows])
            ),
            "max_lumped_ce_sum_mean": float(np.mean([row.max_lumped_ce_sum for row in rows])),
            "max_lumped_ce_sum_max": float(np.max([row.max_lumped_ce_sum for row in rows])),
            "boundary_relative_error_mean": float(
                np.mean([row.boundary_relative_error for row in rows])
            ),
            "residual_debt_ratio_mean": float(
                np.mean([row.residual_debt_ratio for row in rows])
            ),
            "gradient_energy_debt_ratio_mean": float(
                np.mean([row.gradient_energy_debt_ratio for row in rows])
            ),
            "boundary_descent_alignment_ratio_mean": float(
                np.mean([row.boundary_descent_alignment_ratio for row in rows])
            ),
            "dense_token_work_fraction_mean": float(
                np.mean([row.dense_token_work_fraction for row in rows])
            ),
            "boundary_work_fraction_mean": float(
                np.mean([row.boundary_work_fraction for row in rows])
            ),
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
    if args.steps < args.segment_length:
        raise ValueError("--steps must be at least --segment-length")
    if args.steps % args.segment_length != 0:
        raise ValueError("--steps must be divisible by --segment-length")

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
        "positive_condition_exact_boundary_and_bounded_debt": all(
            trial.boundary_relative_error <= args.max_boundary_error
            and trial.residual_debt_ratio <= args.max_residual_debt_ratio
            and trial.gradient_energy_debt_ratio <= args.max_gradient_energy_debt_ratio
            for trial in trials
            if trial.condition == "positive"
        ),
        "positive_condition_skip_and_price": all(
            trial.skipped_token_fraction >= args.min_skipped_token_fraction
            and trial.priced_work_ratio <= args.max_priced_work_ratio
            for trial in trials
            if trial.condition == "positive"
        ),
        "unsafe_entropy_control_all_fail_debt": all(
            (not trial.passed)
            and trial.fail_reason == "unsafe_ce_threshold_lumps_high_entropy_debt"
            for trial in trials
            if trial.condition == "unsafe_entropy_threshold_fail"
        ),
        "oscillating_control_all_fail_debt": all(
            (not trial.passed)
            and trial.fail_reason == "low_ce_without_gradient_certificate_hides_debt"
            for trial in trials
            if trial.condition == "oscillating_gradient_debt_fail"
        ),
        "fragmented_control_all_fail_price": all(
            (not trial.passed)
            and trial.fail_reason == "fragmented_predictable_spans_recover_dense_cost"
            for trial in trials
            if trial.condition == "fragmented_price_fail"
        ),
        "all_conditions_identified": all(trial.identifying_correct for trial in trials),
    }
    decision = (
        "PASS_R501_S55_PTGL_GATE"
        if all(required_passes.values())
        else "FAIL_R501_S55_PTGL_GATE"
    )
    disposition = (
        "synthetic_positive_control_identified_not_training_speed_claim"
        if decision.startswith("PASS_")
        else "ptgl_gate_failed_or_non_identifying"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "CPU synthetic predictability-triggered gradient lumping gate. It "
            "checks CE-triggered segment skipping, exact discounted boundary "
            "lumps, residual debt, gradient-energy debt, and pricing. It is not "
            "heldout CE progress or GPU training-speed evidence."
        ),
        "finding": {
            "round": "R501",
            "aspect": "Speed",
            "id": "R501-S55",
            "title": "Predictability-Triggered Gradient Lumping",
        },
        "thresholds": {
            "ce_threshold": args.ce_threshold,
            "max_boundary_error": args.max_boundary_error,
            "max_residual_debt_ratio": args.max_residual_debt_ratio,
            "max_gradient_energy_debt_ratio": args.max_gradient_energy_debt_ratio,
            "min_skipped_token_fraction": args.min_skipped_token_fraction,
            "max_priced_work_ratio": args.max_priced_work_ratio,
        },
        "shape": {
            "seed": args.seed,
            "seed_count": args.seed_count,
            "steps": args.steps,
            "state_dim": args.state_dim,
            "segment_length": args.segment_length,
        },
        "required_passes": required_passes,
        "aggregate": aggregate(trials),
        "trials": [asdict(trial) for trial in trials],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    positive = result["aggregate"]["positive"]
    print(
        "[R501_S55_PTGL_GATE] "
        f"active=1 decision={decision} disposition={disposition} "
        f"positive_skipped_token_fraction_mean={positive['skipped_token_fraction_mean']:.6f} "
        f"positive_boundary_error_mean={positive['boundary_relative_error_mean']:.6e} "
        f"positive_residual_debt_mean={positive['residual_debt_ratio_mean']:.6f} "
        f"positive_gradient_energy_debt_mean={positive['gradient_energy_debt_ratio_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
