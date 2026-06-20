#!/usr/bin/env python3
"""R501 S43 sparsified adjoint support masking gate.

This CPU diagnostic implements the R501 Speed Math 10 finding:
"Sparsified Adjoint Support Masking".

The gate masks cotangent coordinates whose magnitude is below a threshold before
applying a ternary transition matrix. The finding survives only when the support
is genuinely sparse, the omitted tail has bounded transition debt, and the
priced mask plus sparse-matmul plus residual-repair route stays below dense
work.

Controls exercise the declared kills: CE cotangents becoming dense, small
below-threshold coordinates carrying hidden transition debt, and sparse-kernel
overhead dominating even when support is sparse.

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


SCHEMA = "r501_s43_sparse_adjoint_support_gate.v1"
CONDITIONS = (
    "positive",
    "dense_gradient_fail",
    "hidden_tail_debt_fail",
    "sparse_overhead_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    segment_length: int
    state_dim: int
    support_density: float
    zero_fraction: float
    residual_debt_ratio: float
    dropped_lambda_energy_ratio: float
    masked_transition_relative_error: float
    priced_work_ratio: float
    projected_speedup: float
    sparse_kernel_overhead: float
    price_components: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50143)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--segment-length", type=int, default=192)
    parser.add_argument("--state-dim", type=int, default=384)
    parser.add_argument("--threshold", type=float, default=0.035)
    parser.add_argument("--max-support-density", type=float, default=0.15)
    parser.add_argument("--max-residual-debt-ratio", type=float, default=0.030)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.90)
    parser.add_argument("--mask-cost-ratio", type=float, default=0.020)
    parser.add_argument("--sparse-kernel-overhead", type=float, default=1.60)
    parser.add_argument("--bad-sparse-kernel-overhead", type=float, default=10.00)
    parser.add_argument("--residual-repair-multiplier", type=float, default=2.00)
    return parser.parse_args()


def ternary_transition(
    rng: np.random.Generator,
    state_dim: int,
    condition: str,
    active_columns: np.ndarray,
) -> np.ndarray:
    density = 0.12
    signs = rng.choice(np.array([-1.0, 1.0]), size=(state_dim, state_dim))
    keep = rng.random(size=(state_dim, state_dim)) < density
    a = signs * keep.astype(np.float64) / math.sqrt(max(density * state_dim, 1.0))
    if condition in {"positive", "sparse_overhead_fail"}:
        inactive = ~active_columns
        a[inactive, :] *= 0.010
    if condition == "hidden_tail_debt_fail":
        inactive = ~active_columns
        a[inactive, :] *= 1.80
    return a.astype(np.float64)


def cotangent_sequence(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    if condition == "dense_gradient_fail":
        active_fraction = 0.62
    else:
        active_fraction = 0.10
    active_count = max(1, int(round(active_fraction * args.state_dim)))
    active_columns = np.zeros(args.state_dim, dtype=bool)
    active_columns[rng.choice(args.state_dim, size=active_count, replace=False)] = True

    lam = rng.normal(scale=0.006, size=(args.segment_length, args.state_dim))
    if condition == "hidden_tail_debt_fail":
        inactive = ~active_columns
        lam[:, inactive] = rng.choice(np.array([-1.0, 1.0]), size=(args.segment_length, int(np.sum(inactive)))) * (
            0.92 * args.threshold
        )
        active_values = 0.090 + np.abs(
            rng.normal(scale=0.045, size=(args.segment_length, int(np.sum(active_columns))))
        )
        active_signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=(args.segment_length, int(np.sum(active_columns))),
        )
        lam[:, active_columns] = active_signs * active_values
    elif condition == "dense_gradient_fail":
        lam[:, active_columns] = rng.normal(
            scale=0.13,
            size=(args.segment_length, int(np.sum(active_columns))),
        )
        inactive = ~active_columns
        lam[:, inactive] = rng.normal(
            scale=0.025,
            size=(args.segment_length, int(np.sum(inactive))),
        )
    else:
        active_values = 0.090 + np.abs(
            rng.normal(scale=0.045, size=(args.segment_length, int(np.sum(active_columns))))
        )
        active_signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=(args.segment_length, int(np.sum(active_columns))),
        )
        lam[:, active_columns] = active_signs * active_values
    return lam.astype(np.float64), active_columns


def relative_norm(error: np.ndarray, exact: np.ndarray) -> float:
    return float(np.linalg.norm(error.ravel()) / max(np.linalg.norm(exact.ravel()), 1.0e-30))


def price_route(
    args: argparse.Namespace,
    condition: str,
    support_density: float,
    residual_debt_ratio: float,
) -> tuple[dict[str, float], float]:
    overhead = (
        args.bad_sparse_kernel_overhead
        if condition == "sparse_overhead_fail"
        else args.sparse_kernel_overhead
    )
    components = {
        "mask_scan": float(args.mask_cost_ratio),
        "sparse_transition_apply": float(overhead * support_density),
        "residual_repair": float(args.residual_repair_multiplier * residual_debt_ratio),
    }
    return components, float(overhead)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    lam, active_columns = cotangent_sequence(rng, args, condition)
    a = ternary_transition(rng, args.state_dim, condition, active_columns)
    mask = np.abs(lam) > args.threshold
    masked_lam = lam * mask
    exact = lam @ a
    masked = masked_lam @ a
    residual = masked - exact
    residual_debt_ratio = relative_norm(residual, exact)
    support_density = float(np.mean(mask.astype(np.float64)))
    zero_fraction = 1.0 - support_density
    dropped_lambda = lam - masked_lam
    dropped_lambda_energy_ratio = relative_norm(dropped_lambda, lam)
    masked_transition_relative_error = residual_debt_ratio
    price_components, overhead = price_route(
        args,
        condition,
        support_density,
        residual_debt_ratio,
    )
    priced_work_ratio = float(sum(price_components.values()))
    projected_speedup = 1.0 / priced_work_ratio if priced_work_ratio > 0.0 else math.inf

    support_pass = support_density <= args.max_support_density
    debt_pass = residual_debt_ratio <= args.max_residual_debt_ratio
    price_pass = priced_work_ratio <= args.max_priced_work_ratio
    passed = bool(support_pass and debt_pass and price_pass)
    expected_pass = condition == "positive"
    identifying_correct = passed == expected_pass

    fail_reason = None
    if not passed:
        if not support_pass:
            fail_reason = "cotangent_support_not_sparse_enough"
        elif not debt_pass:
            fail_reason = "masked_tail_residual_debt_exceeds_budget"
        elif not price_pass:
            fail_reason = "sparse_kernel_overhead_exceeds_dense_budget"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=identifying_correct,
        fail_reason=fail_reason,
        segment_length=args.segment_length,
        state_dim=args.state_dim,
        support_density=support_density,
        zero_fraction=zero_fraction,
        residual_debt_ratio=residual_debt_ratio,
        dropped_lambda_energy_ratio=dropped_lambda_energy_ratio,
        masked_transition_relative_error=masked_transition_relative_error,
        priced_work_ratio=priced_work_ratio,
        projected_speedup=float(projected_speedup),
        sparse_kernel_overhead=overhead,
        price_components=price_components,
    )


def finite_trial(trial: TrialResult) -> bool:
    values = (
        trial.support_density,
        trial.zero_fraction,
        trial.residual_debt_ratio,
        trial.dropped_lambda_energy_ratio,
        trial.masked_transition_relative_error,
        trial.priced_work_ratio,
        trial.projected_speedup,
        trial.sparse_kernel_overhead,
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
            "support_density_mean": float(np.mean([row.support_density for row in rows])),
            "support_density_max": float(np.max([row.support_density for row in rows])),
            "zero_fraction_mean": float(np.mean([row.zero_fraction for row in rows])),
            "residual_debt_ratio_mean": float(
                np.mean([row.residual_debt_ratio for row in rows])
            ),
            "residual_debt_ratio_max": float(
                np.max([row.residual_debt_ratio for row in rows])
            ),
            "dropped_lambda_energy_ratio_mean": float(
                np.mean([row.dropped_lambda_energy_ratio for row in rows])
            ),
            "masked_transition_relative_error_mean": float(
                np.mean([row.masked_transition_relative_error for row in rows])
            ),
            "priced_work_ratio_mean": float(
                np.mean([row.priced_work_ratio for row in rows])
            ),
            "projected_speedup_mean": float(
                np.mean([row.projected_speedup for row in rows])
            ),
            "sparse_kernel_overhead_mean": float(
                np.mean([row.sparse_kernel_overhead for row in rows])
            ),
        }
    return out


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.segment_length < 1 or args.state_dim < 4:
        raise ValueError("--segment-length must be positive and --state-dim must be >= 4")
    if args.threshold <= 0.0:
        raise ValueError("--threshold must be positive")

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
        "positive_condition_sparse_enough": all(
            trial.support_density <= args.max_support_density
            for trial in trials
            if trial.condition == "positive"
        ),
        "positive_condition_debt_below_budget": all(
            trial.residual_debt_ratio <= args.max_residual_debt_ratio
            for trial in trials
            if trial.condition == "positive"
        ),
        "dense_gradient_control_all_fail_density": all(
            (not trial.passed)
            and trial.fail_reason == "cotangent_support_not_sparse_enough"
            for trial in trials
            if trial.condition == "dense_gradient_fail"
        ),
        "hidden_tail_control_all_fail_debt": all(
            (not trial.passed)
            and trial.fail_reason == "masked_tail_residual_debt_exceeds_budget"
            for trial in trials
            if trial.condition == "hidden_tail_debt_fail"
        ),
        "sparse_overhead_control_all_fail_price": all(
            (not trial.passed)
            and trial.fail_reason == "sparse_kernel_overhead_exceeds_dense_budget"
            for trial in trials
            if trial.condition == "sparse_overhead_fail"
        ),
        "all_conditions_identified": all(trial.identifying_correct for trial in trials),
    }
    decision = (
        "PASS_R501_S43_SPARSE_ADJOINT_SUPPORT_GATE"
        if all(required_passes.values())
        else "FAIL_R501_S43_SPARSE_ADJOINT_SUPPORT_GATE"
    )
    disposition = (
        "synthetic_positive_control_identified_not_training_speed_claim"
        if decision.startswith("PASS_")
        else "sparse_adjoint_support_gate_failed_or_non_identifying"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "CPU synthetic sparse cotangent support gate. It checks support "
            "density, omitted-tail transition debt, and priced mask plus sparse "
            "apply plus residual repair. It is not heldout CE progress or GPU "
            "training-speed evidence."
        ),
        "finding": {
            "round": "R501",
            "aspect": "Speed",
            "id": "R501-S43",
            "title": "Sparsified Adjoint Support Masking",
        },
        "thresholds": {
            "threshold": args.threshold,
            "max_support_density": args.max_support_density,
            "max_residual_debt_ratio": args.max_residual_debt_ratio,
            "max_priced_work_ratio": args.max_priced_work_ratio,
            "required_zero_fraction": 1.0 - args.max_support_density,
            "required_projected_speedup": 1.0 / args.max_priced_work_ratio,
        },
        "shape": {
            "seed": args.seed,
            "seed_count": args.seed_count,
            "segment_length": args.segment_length,
            "state_dim": args.state_dim,
        },
        "required_passes": required_passes,
        "aggregate": aggregate(trials),
        "trials": [asdict(trial) for trial in trials],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    positive = result["aggregate"]["positive"]
    print(
        "[R501_S43_SPARSE_ADJOINT_SUPPORT_GATE] "
        f"active=1 decision={decision} disposition={disposition} "
        f"positive_support_density_mean={positive['support_density_mean']:.6f} "
        f"positive_zero_fraction_mean={positive['zero_fraction_mean']:.6f} "
        f"positive_residual_debt_ratio_mean={positive['residual_debt_ratio_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
