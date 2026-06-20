#!/usr/bin/env python3
"""R501 S52 ternary-constrained time-to-flip escrow gate.

This CPU diagnostic implements the R501 Speed Paradigm 4 finding:
"Ternary-Constrained Time-To-Flip Escrow (TCTFE)".

The gate keeps dense continuous shadow parameters as the reference, but asks a
ternary artifact question: can an escrowed CE-potential estimate sleep a
parameter until the exact distance-to-flip is reached while preserving every
{-1, 0, 1} realized parameter seen by the forward path?

The positive fixture uses monotone single-boundary motion. Parameters whose
integrated CE potential never reaches their flip margin are never evaluated, and
parameters that do cross are evaluated exactly at the trigger. The candidate
passes only if the realized ternary artifact is identical to dense shadow
updates, the sparse backward fraction remains small, and escrow audit work
keeps the priced route materially below dense per-step backward evaluation.

Controls exercise the declared risks: tiny margins recover dense fallback,
under-covering potential creates false sleep and ternary mismatches, and high
audit cost erases the speed advantage.

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


SCHEMA = "r501_s52_tctfe_gate.v1"
CONDITIONS = (
    "positive",
    "tiny_margin_fail",
    "false_sleep_fail",
    "audit_cost_fail",
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
    parameters: int
    dense_flip_fraction: float
    sparse_backward_fraction: float
    sleeping_parameter_fraction: float
    ternary_mismatch_fraction: float
    missed_flip_fraction: float
    false_event_fraction: float
    zero_artifact_residual_fraction: float
    mean_time_to_flip: float
    min_trigger_slack: float
    estimator_cost_ratio: float
    audit_cost_ratio: float
    priced_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50152)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--parameters", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--epsilon-safe", type=float, default=1.0e-12)
    parser.add_argument("--max-ternary-mismatch-fraction", type=float, default=0.0)
    parser.add_argument("--max-missed-flip-fraction", type=float, default=0.0)
    parser.add_argument("--max-sparse-backward-fraction", type=float, default=0.25)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.60)
    parser.add_argument("--min-dense-flip-fraction", type=float, default=0.05)
    parser.add_argument("--estimator-cost-ratio", type=float, default=0.085)
    parser.add_argument("--audit-cost-ratio", type=float, default=0.045)
    parser.add_argument("--high-audit-cost-ratio", type=float, default=0.58)
    return parser.parse_args()


def ternary(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0.5, 1, np.where(values < -0.5, -1, 0))


def make_initial_shadow(
    rng: np.random.Generator,
    parameters: int,
    active_fraction: float,
    tiny_margins: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active_count = max(1, int(round(parameters * active_fraction)))
    active = np.zeros(parameters, dtype=bool)
    active[rng.choice(parameters, size=active_count, replace=False)] = True

    if tiny_margins:
        margins = rng.uniform(0.003, 0.028, size=parameters)
    else:
        margins = rng.uniform(0.090, 0.260, size=parameters)

    boundary_choice = rng.integers(0, 4, size=parameters)
    boundary = np.empty(parameters, dtype=np.float64)
    direction = np.empty(parameters, dtype=np.float64)
    theta0 = np.empty(parameters, dtype=np.float64)

    for idx, choice in enumerate(boundary_choice):
        margin = margins[idx]
        if choice == 0:
            boundary[idx] = -0.5
            direction[idx] = 1.0
            theta0[idx] = boundary[idx] - margin
        elif choice == 1:
            boundary[idx] = -0.5
            direction[idx] = -1.0
            theta0[idx] = boundary[idx] + margin
        elif choice == 2:
            boundary[idx] = 0.5
            direction[idx] = 1.0
            theta0[idx] = boundary[idx] - margin
        else:
            boundary[idx] = 0.5
            direction[idx] = -1.0
            theta0[idx] = boundary[idx] + margin

    return theta0, direction, margins, active


def smooth_profile(rng: np.random.Generator, steps: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    profile = 1.0 + 0.25 * np.sin(2.0 * math.pi * t + rng.uniform(-0.3, 0.3))
    profile += 0.12 * np.cos(6.0 * math.pi * t + rng.uniform(-0.5, 0.5))
    profile = np.maximum(profile, 0.05)
    return profile / float(np.sum(profile))


def condition_config(condition: str) -> tuple[float, bool, float, float]:
    if condition == "tiny_margin_fail":
        return 0.78, True, 1.0, 0.045
    if condition == "false_sleep_fail":
        return 0.14, False, 0.48, 0.045
    if condition == "audit_cost_fail":
        return 0.12, False, 1.0, 0.58
    return 0.12, False, 1.0, 0.045


def make_delta(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active_fraction, tiny_margins, _, _ = condition_config(condition)
    theta0, direction, margins, active = make_initial_shadow(
        rng,
        args.parameters,
        active_fraction,
        tiny_margins,
    )
    total_potential = np.empty(args.parameters, dtype=np.float64)
    total_potential[active] = margins[active] + rng.uniform(0.035, 0.125, size=int(active.sum()))
    total_potential[~active] = margins[~active] * rng.uniform(
        0.18,
        0.72,
        size=int((~active).sum()),
    )

    if condition == "tiny_margin_fail":
        total_potential[active] = margins[active] + rng.uniform(
            0.006,
            0.030,
            size=int(active.sum()),
        )
    if condition == "false_sleep_fail":
        total_potential[active] = margins[active] + rng.uniform(
            0.018,
            0.055,
            size=int(active.sum()),
        )

    profile = smooth_profile(rng, args.steps)
    delta = direction[None, :] * total_potential[None, :] * profile[:, None]
    return theta0, direction, margins, active, delta.astype(np.float64)


def run_escrow(
    args: argparse.Namespace,
    theta0: np.ndarray,
    direction: np.ndarray,
    margins: np.ndarray,
    delta: np.ndarray,
    estimator_scale: float,
) -> dict[str, Any]:
    dense_theta = theta0.copy()
    candidate_theta = theta0.copy()
    initial_ternary = ternary(theta0)
    event_done = np.zeros(theta0.shape[0], dtype=bool)
    event_step = np.full(theta0.shape[0], -1, dtype=np.int32)
    actual_potential = np.zeros(theta0.shape[0], dtype=np.float64)
    estimated_potential = np.zeros(theta0.shape[0], dtype=np.float64)
    trigger_slack: list[float] = []
    mismatch_total = 0
    comparisons = 0

    for step in range(delta.shape[0]):
        signed_delta = args.learning_rate * delta[step]
        dense_theta += signed_delta
        actual_potential += np.abs(signed_delta)
        estimated_potential += estimator_scale * np.abs(signed_delta)

        trigger = (~event_done) & (
            estimated_potential + args.epsilon_safe >= margins
        )
        if np.any(trigger):
            candidate_theta[trigger] = theta0[trigger] + direction[trigger] * actual_potential[trigger]
            event_step[trigger] = step + 1
            event_done[trigger] = True
            trigger_slack.extend((estimated_potential[trigger] - margins[trigger]).tolist())

        dense_ternary = ternary(dense_theta)
        candidate_ternary = ternary(candidate_theta)
        mismatch_total += int(np.count_nonzero(dense_ternary != candidate_ternary))
        comparisons += int(dense_ternary.size)

    dense_final = ternary(dense_theta)
    candidate_final = ternary(candidate_theta)
    dense_flipped = dense_final != initial_ternary
    candidate_flipped = candidate_final != initial_ternary
    missed_flip = dense_flipped & (candidate_final != dense_final)
    false_event = event_done & (~dense_flipped)

    event_count = int(np.count_nonzero(event_done))
    flip_count = int(np.count_nonzero(dense_flipped))
    if event_count:
        mean_time_to_flip = float(np.mean(event_step[event_done]))
        min_trigger_slack = float(np.min(np.array(trigger_slack, dtype=np.float64)))
    else:
        mean_time_to_flip = float(delta.shape[0] + 1)
        min_trigger_slack = float(np.max(estimated_potential - margins))

    return {
        "event_count": event_count,
        "flip_count": flip_count,
        "dense_flip_fraction": float(flip_count / theta0.size),
        "sparse_backward_fraction": float(event_count / theta0.size),
        "sleeping_parameter_fraction": float(1.0 - event_count / theta0.size),
        "ternary_mismatch_fraction": float(mismatch_total / max(comparisons, 1)),
        "missed_flip_fraction": float(np.count_nonzero(missed_flip) / theta0.size),
        "false_event_fraction": float(np.count_nonzero(false_event) / theta0.size),
        "zero_artifact_residual_fraction": float(
            1.0 - mismatch_total / max(comparisons, 1)
        ),
        "mean_time_to_flip": mean_time_to_flip,
        "min_trigger_slack": min_trigger_slack,
    }


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    theta0, direction, margins, _, delta = make_delta(rng, args, condition)
    _, _, estimator_scale, audit_cost = condition_config(condition)
    escrow = run_escrow(args, theta0, direction, margins, delta, estimator_scale)
    priced_work = (
        args.estimator_cost_ratio
        + float(escrow["sparse_backward_fraction"])
        + audit_cost
    )
    projected_speedup = 1.0 / priced_work if priced_work > 0.0 else math.inf

    mismatch_pass = (
        escrow["ternary_mismatch_fraction"] <= args.max_ternary_mismatch_fraction
        and escrow["missed_flip_fraction"] <= args.max_missed_flip_fraction
    )
    sparse_pass = escrow["sparse_backward_fraction"] <= args.max_sparse_backward_fraction
    price_pass = priced_work <= args.max_priced_work_ratio
    flip_presence_pass = escrow["dense_flip_fraction"] >= args.min_dense_flip_fraction
    passed = bool(mismatch_pass and sparse_pass and price_pass and flip_presence_pass)
    expected_pass = condition == "positive"
    identifying_correct = passed == expected_pass

    fail_reason = None
    if not passed:
        if condition == "false_sleep_fail" and not mismatch_pass:
            fail_reason = "escrow_false_sleep_creates_ternary_mismatch"
        elif condition == "tiny_margin_fail" and (not sparse_pass or not price_pass):
            fail_reason = "tiny_margins_recover_dense_backward_evaluation"
        elif condition == "audit_cost_fail" and not price_pass:
            fail_reason = "escrow_audit_recovers_dense_backward_cost"
        elif not mismatch_pass:
            fail_reason = "escrow_false_sleep_creates_ternary_mismatch"
        elif not sparse_pass:
            fail_reason = "tiny_margins_recover_dense_backward_evaluation"
        elif not price_pass:
            fail_reason = "escrow_audit_recovers_dense_backward_cost"
        elif not flip_presence_pass:
            fail_reason = "no_material_flip_surface"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=identifying_correct,
        fail_reason=fail_reason,
        steps=args.steps,
        parameters=args.parameters,
        dense_flip_fraction=float(escrow["dense_flip_fraction"]),
        sparse_backward_fraction=float(escrow["sparse_backward_fraction"]),
        sleeping_parameter_fraction=float(escrow["sleeping_parameter_fraction"]),
        ternary_mismatch_fraction=float(escrow["ternary_mismatch_fraction"]),
        missed_flip_fraction=float(escrow["missed_flip_fraction"]),
        false_event_fraction=float(escrow["false_event_fraction"]),
        zero_artifact_residual_fraction=float(escrow["zero_artifact_residual_fraction"]),
        mean_time_to_flip=float(escrow["mean_time_to_flip"]),
        min_trigger_slack=float(escrow["min_trigger_slack"]),
        estimator_cost_ratio=float(args.estimator_cost_ratio),
        audit_cost_ratio=float(audit_cost),
        priced_work_ratio=float(priced_work),
        projected_speedup=float(projected_speedup),
    )


def finite_trial(trial: TrialResult) -> bool:
    values = (
        trial.dense_flip_fraction,
        trial.sparse_backward_fraction,
        trial.sleeping_parameter_fraction,
        trial.ternary_mismatch_fraction,
        trial.missed_flip_fraction,
        trial.false_event_fraction,
        trial.zero_artifact_residual_fraction,
        trial.mean_time_to_flip,
        trial.min_trigger_slack,
        trial.estimator_cost_ratio,
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
            "dense_flip_fraction_mean": float(np.mean([row.dense_flip_fraction for row in rows])),
            "sparse_backward_fraction_mean": float(
                np.mean([row.sparse_backward_fraction for row in rows])
            ),
            "sparse_backward_fraction_max": float(
                np.max([row.sparse_backward_fraction for row in rows])
            ),
            "sleeping_parameter_fraction_mean": float(
                np.mean([row.sleeping_parameter_fraction for row in rows])
            ),
            "ternary_mismatch_fraction_mean": float(
                np.mean([row.ternary_mismatch_fraction for row in rows])
            ),
            "missed_flip_fraction_mean": float(
                np.mean([row.missed_flip_fraction for row in rows])
            ),
            "false_event_fraction_mean": float(
                np.mean([row.false_event_fraction for row in rows])
            ),
            "zero_artifact_residual_fraction_mean": float(
                np.mean([row.zero_artifact_residual_fraction for row in rows])
            ),
            "mean_time_to_flip_mean": float(np.mean([row.mean_time_to_flip for row in rows])),
            "min_trigger_slack_min": float(np.min([row.min_trigger_slack for row in rows])),
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
    if args.steps < 4:
        raise ValueError("--steps must be at least 4")
    if args.parameters < 8:
        raise ValueError("--parameters must be at least 8")

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
        "positive_condition_zero_ternary_residual": all(
            trial.ternary_mismatch_fraction <= args.max_ternary_mismatch_fraction
            and trial.missed_flip_fraction <= args.max_missed_flip_fraction
            and trial.zero_artifact_residual_fraction == 1.0
            for trial in trials
            if trial.condition == "positive"
        ),
        "positive_condition_sparse_and_priced": all(
            trial.sparse_backward_fraction <= args.max_sparse_backward_fraction
            and trial.priced_work_ratio <= args.max_priced_work_ratio
            for trial in trials
            if trial.condition == "positive"
        ),
        "tiny_margin_control_all_fail_price": all(
            (not trial.passed)
            and trial.fail_reason == "tiny_margins_recover_dense_backward_evaluation"
            for trial in trials
            if trial.condition == "tiny_margin_fail"
        ),
        "false_sleep_control_all_fail_mismatch": all(
            (not trial.passed)
            and trial.fail_reason == "escrow_false_sleep_creates_ternary_mismatch"
            for trial in trials
            if trial.condition == "false_sleep_fail"
        ),
        "audit_cost_control_all_fail_price": all(
            (not trial.passed)
            and trial.fail_reason == "escrow_audit_recovers_dense_backward_cost"
            for trial in trials
            if trial.condition == "audit_cost_fail"
        ),
        "all_conditions_identified": all(trial.identifying_correct for trial in trials),
    }
    decision = (
        "PASS_R501_S52_TCTFE_GATE"
        if all(required_passes.values())
        else "FAIL_R501_S52_TCTFE_GATE"
    )
    disposition = (
        "synthetic_positive_control_identified_not_training_speed_claim"
        if decision.startswith("PASS_")
        else "tctfe_gate_failed_or_non_identifying"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "CPU synthetic ternary-constrained time-to-flip escrow gate. It "
            "checks exact realized {-1,0,1} artifact equivalence under a "
            "monotone single-boundary distance-to-flip fixture with sparse "
            "backward evaluation and audit pricing. It is not heldout CE "
            "progress or GPU training-speed evidence."
        ),
        "finding": {
            "round": "R501",
            "aspect": "Speed",
            "id": "R501-S52",
            "title": "Ternary-Constrained Time-To-Flip Escrow",
        },
        "thresholds": {
            "max_ternary_mismatch_fraction": args.max_ternary_mismatch_fraction,
            "max_missed_flip_fraction": args.max_missed_flip_fraction,
            "max_sparse_backward_fraction": args.max_sparse_backward_fraction,
            "max_priced_work_ratio": args.max_priced_work_ratio,
            "min_dense_flip_fraction": args.min_dense_flip_fraction,
            "epsilon_safe": args.epsilon_safe,
        },
        "shape": {
            "seed": args.seed,
            "seed_count": args.seed_count,
            "steps": args.steps,
            "parameters": args.parameters,
        },
        "required_passes": required_passes,
        "aggregate": aggregate(trials),
        "trials": [asdict(trial) for trial in trials],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    positive = result["aggregate"]["positive"]
    print(
        "[R501_S52_TCTFE_GATE] "
        f"active=1 decision={decision} disposition={disposition} "
        f"positive_ternary_mismatch_fraction_mean={positive['ternary_mismatch_fraction_mean']:.6e} "
        f"positive_dense_flip_fraction_mean={positive['dense_flip_fraction_mean']:.6f} "
        f"positive_sparse_backward_fraction_mean={positive['sparse_backward_fraction_mean']:.6f} "
        f"positive_sleeping_parameter_fraction_mean={positive['sleeping_parameter_fraction_mean']:.6f} "
        f"positive_zero_artifact_residual_fraction_mean={positive['zero_artifact_residual_fraction_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
