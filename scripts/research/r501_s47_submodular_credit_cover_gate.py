#!/usr/bin/env python3
"""R501 S47 submodular trajectory credit cover gate.

This CPU diagnostic implements the R501 Speed Math 14 finding:
"Submodular Trajectory Credit Cover".

The gate treats per-token surprise peaks as candidate credit origins. A greedy
submodular cover selects a small set of trajectory positions, covers nearby CE
and gradient energy through a local influence kernel, and prices exact BPTT only
through selected origins plus peak-finder overhead.

The finding survives only when the selected peaks cover the target loss mass,
cover the actual gradient-energy mass, leave bounded residual debt, and keep
the priced exact-credit route below half the dense trajectory cost.

Controls exercise the declared kills: uniformly distributed credit cannot be
explained by a few peaks, high peak-finder overhead erases the priced gain, and
loss peaks can be misleading when gradient energy lives elsewhere.

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


SCHEMA = "r501_s47_submodular_credit_cover_gate.v1"
CONDITIONS = (
    "positive",
    "distributed_credit_fail",
    "peak_finder_cost_fail",
    "loss_gradient_mismatch_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    sequence_length: int
    max_peaks: int
    selected_count: int
    selected_fraction: float
    loss_cover_fraction: float
    gradient_cover_fraction: float
    selected_raw_gradient_fraction: float
    residual_debt_ratio: float
    peak_find_cost_ratio: float
    cover_accounting_cost_ratio: float
    priced_work_ratio: float
    projected_speedup: float
    selected_positions: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50147)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--max-peaks", type=int, default=8)
    parser.add_argument("--target-loss-cover", type=float, default=0.95)
    parser.add_argument("--min-gradient-cover", type=float, default=0.90)
    parser.add_argument("--max-residual-debt", type=float, default=0.10)
    parser.add_argument("--max-selected-fraction", type=float, default=0.10)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.50)
    parser.add_argument("--coverage-sigma", type=float, default=1.45)
    parser.add_argument("--peak-find-cost-ratio", type=float, default=0.06)
    parser.add_argument("--cover-accounting-cost-ratio", type=float, default=0.01)
    parser.add_argument("--high-peak-find-cost-ratio", type=float, default=0.55)
    return parser.parse_args()


def gaussian_positions(length: int, centers: np.ndarray, sigma: float) -> np.ndarray:
    positions = np.arange(length, dtype=np.float64)
    out = np.zeros(length, dtype=np.float64)
    for center in centers:
        out += np.exp(-0.5 * ((positions - float(center)) / sigma) ** 2)
    return out


def separated_centers(
    rng: np.random.Generator,
    length: int,
    count: int,
) -> np.ndarray:
    base = np.linspace(18, length - 19, count, dtype=np.float64)
    jitter = rng.integers(-4, 5, size=count)
    centers = np.clip(np.rint(base + jitter), 4, length - 5).astype(int)
    return np.unique(centers)


def make_condition_series(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    length = args.sequence_length
    peak_cost = args.peak_find_cost_ratio
    if condition in {"positive", "peak_finder_cost_fail", "loss_gradient_mismatch_fail"}:
        centers = separated_centers(rng, length, args.max_peaks)
        loss = 0.001 + gaussian_positions(length, centers, sigma=0.55)
        if condition == "loss_gradient_mismatch_fail":
            shifted = (centers + max(9, length // (2 * args.max_peaks))) % length
            gradient = 0.001 + gaussian_positions(length, shifted, sigma=0.65)
        else:
            gradient = 0.001 + gaussian_positions(length, centers, sigma=0.65)
        loss *= rng.uniform(0.90, 1.10, size=length)
        gradient *= rng.uniform(0.92, 1.08, size=length)
        if condition == "peak_finder_cost_fail":
            peak_cost = args.high_peak_find_cost_ratio
        return loss, gradient, peak_cost

    if condition == "distributed_credit_fail":
        loss = 1.0 + rng.uniform(-0.04, 0.04, size=length)
        gradient = 1.0 + rng.uniform(-0.04, 0.04, size=length)
        return loss, gradient, peak_cost

    raise ValueError(f"unknown condition: {condition}")


def influence_matrix(length: int, sigma: float) -> np.ndarray:
    positions = np.arange(length, dtype=np.float64)
    distance = positions[:, None] - positions[None, :]
    coverage = np.exp(-0.5 * (distance / sigma) ** 2)
    coverage[coverage < 0.02] = 0.0
    return np.clip(coverage * 1.15, 0.0, 1.0)


def greedy_cover(
    loss: np.ndarray,
    coverage: np.ndarray,
    max_peaks: int,
    target_loss_cover: float,
) -> tuple[list[int], np.ndarray]:
    selected: list[int] = []
    covered = np.zeros_like(loss)
    total_loss = max(float(np.sum(loss)), 1.0e-30)
    for _ in range(max_peaks):
        best_idx = -1
        best_gain = -math.inf
        for candidate in range(loss.shape[0]):
            if candidate in selected:
                continue
            candidate_cover = np.maximum(covered, coverage[candidate])
            gain = float(np.sum((candidate_cover - covered) * loss))
            if gain > best_gain:
                best_gain = gain
                best_idx = candidate
        if best_idx < 0 or best_gain <= 0.0:
            break
        selected.append(int(best_idx))
        covered = np.maximum(covered, coverage[best_idx])
        if float(np.sum(covered * loss)) / total_loss >= target_loss_cover:
            break
    return selected, covered


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    loss, gradient, peak_cost = make_condition_series(rng, args, condition)
    coverage = influence_matrix(args.sequence_length, args.coverage_sigma)
    selected, covered = greedy_cover(
        loss,
        coverage,
        args.max_peaks,
        args.target_loss_cover,
    )

    total_loss = max(float(np.sum(loss)), 1.0e-30)
    total_gradient = max(float(np.sum(gradient)), 1.0e-30)
    selected_positions = np.asarray(selected, dtype=int)
    loss_cover = float(np.sum(covered * loss) / total_loss)
    gradient_cover = float(np.sum(covered * gradient) / total_gradient)
    raw_gradient = (
        float(np.sum(gradient[selected_positions]) / total_gradient)
        if selected_positions.size
        else 0.0
    )
    residual_debt = max(0.0, 1.0 - gradient_cover)
    selected_fraction = float(len(selected) / args.sequence_length)
    priced_work = selected_fraction + peak_cost + args.cover_accounting_cost_ratio
    projected_speedup = 1.0 / priced_work if priced_work > 0.0 else math.inf

    loss_pass = loss_cover >= args.target_loss_cover
    gradient_pass = gradient_cover >= args.min_gradient_cover
    debt_pass = residual_debt <= args.max_residual_debt
    fraction_pass = selected_fraction <= args.max_selected_fraction
    price_pass = priced_work <= args.max_priced_work_ratio
    passed = bool(loss_pass and gradient_pass and debt_pass and fraction_pass and price_pass)
    expected_pass = condition == "positive"
    identifying_correct = passed == expected_pass

    fail_reason = None
    if not passed:
        if condition == "peak_finder_cost_fail" and not price_pass:
            fail_reason = "peak_finder_recovers_dense_cost"
        elif condition == "loss_gradient_mismatch_fail" and loss_pass and (not gradient_pass or not debt_pass):
            fail_reason = "loss_peaks_do_not_cover_gradient_energy"
        elif not loss_pass or not gradient_pass or not debt_pass:
            fail_reason = "trajectory_credit_is_dense_distributed"
        elif not fraction_pass or not price_pass:
            fail_reason = "peak_finder_recovers_dense_cost"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=identifying_correct,
        fail_reason=fail_reason,
        sequence_length=args.sequence_length,
        max_peaks=args.max_peaks,
        selected_count=len(selected),
        selected_fraction=selected_fraction,
        loss_cover_fraction=loss_cover,
        gradient_cover_fraction=gradient_cover,
        selected_raw_gradient_fraction=raw_gradient,
        residual_debt_ratio=residual_debt,
        peak_find_cost_ratio=float(peak_cost),
        cover_accounting_cost_ratio=float(args.cover_accounting_cost_ratio),
        priced_work_ratio=priced_work,
        projected_speedup=float(projected_speedup),
        selected_positions=[int(position) for position in selected],
    )


def finite_trial(trial: TrialResult) -> bool:
    values = (
        trial.selected_fraction,
        trial.loss_cover_fraction,
        trial.gradient_cover_fraction,
        trial.selected_raw_gradient_fraction,
        trial.residual_debt_ratio,
        trial.peak_find_cost_ratio,
        trial.cover_accounting_cost_ratio,
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
            "selected_count_mean": float(np.mean([row.selected_count for row in rows])),
            "selected_count_max": float(np.max([row.selected_count for row in rows])),
            "selected_fraction_mean": float(np.mean([row.selected_fraction for row in rows])),
            "loss_cover_fraction_mean": float(
                np.mean([row.loss_cover_fraction for row in rows])
            ),
            "loss_cover_fraction_min": float(np.min([row.loss_cover_fraction for row in rows])),
            "gradient_cover_fraction_mean": float(
                np.mean([row.gradient_cover_fraction for row in rows])
            ),
            "gradient_cover_fraction_min": float(
                np.min([row.gradient_cover_fraction for row in rows])
            ),
            "selected_raw_gradient_fraction_mean": float(
                np.mean([row.selected_raw_gradient_fraction for row in rows])
            ),
            "residual_debt_ratio_mean": float(
                np.mean([row.residual_debt_ratio for row in rows])
            ),
            "residual_debt_ratio_max": float(
                np.max([row.residual_debt_ratio for row in rows])
            ),
            "peak_find_cost_ratio_mean": float(
                np.mean([row.peak_find_cost_ratio for row in rows])
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
    if args.sequence_length < 32:
        raise ValueError("--sequence-length must be at least 32")
    if args.max_peaks < 1:
        raise ValueError("--max-peaks must be at least 1")

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
        "positive_condition_loss_and_gradient_covered": all(
            trial.loss_cover_fraction >= args.target_loss_cover
            and trial.gradient_cover_fraction >= args.min_gradient_cover
            for trial in trials
            if trial.condition == "positive"
        ),
        "distributed_credit_control_all_fail_dense_credit": all(
            (not trial.passed)
            and trial.fail_reason == "trajectory_credit_is_dense_distributed"
            for trial in trials
            if trial.condition == "distributed_credit_fail"
        ),
        "peak_finder_cost_control_all_fail_price": all(
            (not trial.passed)
            and trial.fail_reason == "peak_finder_recovers_dense_cost"
            for trial in trials
            if trial.condition == "peak_finder_cost_fail"
        ),
        "loss_gradient_mismatch_control_all_fail_debt": all(
            (not trial.passed)
            and trial.fail_reason == "loss_peaks_do_not_cover_gradient_energy"
            for trial in trials
            if trial.condition == "loss_gradient_mismatch_fail"
        ),
        "all_conditions_identified": all(trial.identifying_correct for trial in trials),
    }
    decision = (
        "PASS_R501_S47_SUBMODULAR_CREDIT_COVER_GATE"
        if all(required_passes.values())
        else "FAIL_R501_S47_SUBMODULAR_CREDIT_COVER_GATE"
    )
    disposition = (
        "synthetic_positive_control_identified_not_training_speed_claim"
        if decision.startswith("PASS_")
        else "submodular_credit_cover_gate_failed_or_non_identifying"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "CPU synthetic submodular trajectory credit-cover gate. It checks "
            "CE loss cover, actual gradient-energy cover, residual non-peak "
            "debt, selected credit-origin fraction, and priced exact-credit "
            "work. It is not heldout CE progress or GPU training-speed evidence."
        ),
        "finding": {
            "round": "R501",
            "aspect": "Speed",
            "id": "R501-S47",
            "title": "Submodular Trajectory Credit Cover",
        },
        "thresholds": {
            "target_loss_cover": args.target_loss_cover,
            "min_gradient_cover": args.min_gradient_cover,
            "max_residual_debt": args.max_residual_debt,
            "max_selected_fraction": args.max_selected_fraction,
            "max_priced_work_ratio": args.max_priced_work_ratio,
            "coverage_sigma": args.coverage_sigma,
        },
        "shape": {
            "seed": args.seed,
            "seed_count": args.seed_count,
            "sequence_length": args.sequence_length,
            "max_peaks": args.max_peaks,
        },
        "required_passes": required_passes,
        "aggregate": aggregate(trials),
        "trials": [asdict(trial) for trial in trials],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    positive = result["aggregate"]["positive"]
    print(
        "[R501_S47_SUBMODULAR_CREDIT_COVER_GATE] "
        f"active=1 decision={decision} disposition={disposition} "
        f"positive_selected_count_mean={positive['selected_count_mean']:.6f} "
        f"positive_loss_cover_fraction_mean={positive['loss_cover_fraction_mean']:.6f} "
        f"positive_gradient_cover_fraction_mean={positive['gradient_cover_fraction_mean']:.6f} "
        f"positive_residual_debt_ratio_mean={positive['residual_debt_ratio_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
