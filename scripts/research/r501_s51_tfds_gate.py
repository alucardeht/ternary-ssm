#!/usr/bin/env python3
"""R501 S51 target-forced dual-state forward-flow gate.

This CPU diagnostic implements the R501 Speed Paradigm 3 finding:
"Target-Forced Dual-State Forward Flow (TFDS)".

The gate runs two linear SSM forward trajectories: an autonomous predicted-input
trajectory and a target-forced trajectory. In the positive linear fixture, their
state difference is exactly the forward-propagated target residual, so the
candidate can recover the dual credit signal without a recurrent backward pass.

The finding survives only when the dual-state difference matches the exact
linear residual flow, the induced gradient is identical to the dense two-state
reference, target forcing is aligned, and the second forward plus differencing
cost remains materially below dense recurrent backward work.

Controls exercise the declared risks: nonlinear transitions break the linear
dual difference identity, shifted targets create a false credit signal, and an
expensive second forward can erase the speed advantage.

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


SCHEMA = "r501_s51_tfds_gate.v1"
CONDITIONS = (
    "positive",
    "nonlinear_transition_fail",
    "target_shift_fail",
    "second_forward_cost_fail",
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
    state_dim: int
    input_dim: int
    dual_state_relative_error: float
    gradient_relative_error: float
    target_alignment_error: float
    descent_alignment_ratio: float
    second_forward_cost_ratio: float
    difference_cost_ratio: float
    audit_cost_ratio: float
    priced_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50151)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=12)
    parser.add_argument("--max-dual-state-error", type=float, default=1.0e-10)
    parser.add_argument("--max-gradient-error", type=float, default=1.0e-10)
    parser.add_argument("--max-target-alignment-error", type=float, default=1.0e-10)
    parser.add_argument("--min-descent-alignment-ratio", type=float, default=0.999999)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.75)
    parser.add_argument("--second-forward-cost-ratio", type=float, default=0.52)
    parser.add_argument("--difference-cost-ratio", type=float, default=0.05)
    parser.add_argument("--audit-cost-ratio", type=float, default=0.04)
    parser.add_argument("--high-second-forward-cost-ratio", type=float, default=0.82)
    return parser.parse_args()


def stable_matrix(rng: np.random.Generator, dim: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    eig = np.linspace(0.12, 0.68, dim, dtype=np.float64)
    return q @ np.diag(eig) @ q.T


def make_inputs(
    rng: np.random.Generator,
    length: int,
    input_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred = rng.normal(scale=0.20, size=(length, input_dim))
    residual = np.zeros_like(pred)
    centers = np.linspace(12, length - 13, 8, dtype=int)
    for center in centers:
        width = rng.integers(2, 5)
        direction = rng.normal(size=input_dim)
        direction /= max(float(np.linalg.norm(direction)), 1.0e-30)
        for offset in range(-width, width + 1):
            idx = int(np.clip(center + offset, 0, length - 1))
            residual[idx] += math.exp(-0.5 * (offset / max(width, 1)) ** 2) * direction
    residual *= 0.12
    target = pred + residual
    return pred.astype(np.float64), target.astype(np.float64)


def run_linear_forward(a: np.ndarray, b: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    states = np.zeros((inputs.shape[0] + 1, a.shape[0]), dtype=np.float64)
    for idx in range(inputs.shape[0]):
        states[idx + 1] = a @ states[idx] + b @ inputs[idx]
    return states[1:]


def run_nonlinear_forward(a: np.ndarray, b: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    states = np.zeros((inputs.shape[0] + 1, a.shape[0]), dtype=np.float64)
    for idx in range(inputs.shape[0]):
        states[idx + 1] = np.tanh(a @ states[idx] + b @ inputs[idx])
    return states[1:]


def dual_residual_forward(a: np.ndarray, b: np.ndarray, residual: np.ndarray) -> np.ndarray:
    states = np.zeros((residual.shape[0] + 1, a.shape[0]), dtype=np.float64)
    for idx in range(residual.shape[0]):
        states[idx + 1] = a @ states[idx] + b @ residual[idx]
    return states[1:]


def relative_error(candidate: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.linalg.norm((candidate - target).reshape(-1))
        / max(float(np.linalg.norm(target.reshape(-1))), 1.0e-30)
    )


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    a = stable_matrix(rng, args.state_dim)
    b = rng.normal(scale=0.18, size=(args.state_dim, args.input_dim))
    pred_inputs, target_inputs = make_inputs(rng, args.sequence_length, args.input_dim)
    forced_inputs = target_inputs.copy()
    residual_for_dual = target_inputs - pred_inputs
    second_forward_cost = args.second_forward_cost_ratio

    if condition == "target_shift_fail":
        forced_inputs = np.roll(target_inputs, shift=3, axis=0)
    elif condition == "second_forward_cost_fail":
        second_forward_cost = args.high_second_forward_cost_ratio

    if condition == "nonlinear_transition_fail":
        pred_states = run_nonlinear_forward(a, b, pred_inputs)
        forced_states = run_nonlinear_forward(a, b, forced_inputs)
    else:
        pred_states = run_linear_forward(a, b, pred_inputs)
        forced_states = run_linear_forward(a, b, forced_inputs)

    exact_dual = forced_states - pred_states
    candidate_dual = dual_residual_forward(a, b, residual_for_dual)
    dense_gradient = exact_dual.T @ pred_inputs
    candidate_gradient = candidate_dual.T @ pred_inputs
    dual_error = relative_error(candidate_dual, exact_dual)
    gradient_error = relative_error(candidate_gradient, dense_gradient)
    target_alignment_error = relative_error(forced_inputs - pred_inputs, residual_for_dual)
    numerator = float(np.sum(candidate_gradient * dense_gradient))
    denominator = max(float(np.sum(dense_gradient * dense_gradient)), 1.0e-30)
    descent_alignment = numerator / denominator
    priced_work = second_forward_cost + args.difference_cost_ratio + args.audit_cost_ratio
    projected_speedup = 1.0 / priced_work if priced_work > 0.0 else math.inf

    dual_pass = dual_error <= args.max_dual_state_error
    grad_pass = gradient_error <= args.max_gradient_error
    target_pass = target_alignment_error <= args.max_target_alignment_error
    descent_pass = descent_alignment >= args.min_descent_alignment_ratio
    price_pass = priced_work <= args.max_priced_work_ratio
    passed = bool(dual_pass and grad_pass and target_pass and descent_pass and price_pass)
    expected_pass = condition == "positive"
    identifying_correct = passed == expected_pass

    fail_reason = None
    if not passed:
        if condition == "nonlinear_transition_fail" and (not dual_pass or not grad_pass):
            fail_reason = "nonlinear_transition_breaks_dual_forward_identity"
        elif condition == "target_shift_fail" and not target_pass:
            fail_reason = "target_forcing_not_aligned_with_autonomous_tokens"
        elif condition == "second_forward_cost_fail" and not price_pass:
            fail_reason = "second_forward_recovers_dense_backward_cost"
        elif not dual_pass or not grad_pass:
            fail_reason = "nonlinear_transition_breaks_dual_forward_identity"
        elif not target_pass:
            fail_reason = "target_forcing_not_aligned_with_autonomous_tokens"
        elif not price_pass:
            fail_reason = "second_forward_recovers_dense_backward_cost"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=identifying_correct,
        fail_reason=fail_reason,
        sequence_length=args.sequence_length,
        state_dim=args.state_dim,
        input_dim=args.input_dim,
        dual_state_relative_error=dual_error,
        gradient_relative_error=gradient_error,
        target_alignment_error=target_alignment_error,
        descent_alignment_ratio=descent_alignment,
        second_forward_cost_ratio=float(second_forward_cost),
        difference_cost_ratio=float(args.difference_cost_ratio),
        audit_cost_ratio=float(args.audit_cost_ratio),
        priced_work_ratio=float(priced_work),
        projected_speedup=float(projected_speedup),
    )


def finite_trial(trial: TrialResult) -> bool:
    values = (
        trial.dual_state_relative_error,
        trial.gradient_relative_error,
        trial.target_alignment_error,
        trial.descent_alignment_ratio,
        trial.second_forward_cost_ratio,
        trial.difference_cost_ratio,
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
            "dual_state_relative_error_mean": float(
                np.mean([row.dual_state_relative_error for row in rows])
            ),
            "dual_state_relative_error_max": float(
                np.max([row.dual_state_relative_error for row in rows])
            ),
            "gradient_relative_error_mean": float(
                np.mean([row.gradient_relative_error for row in rows])
            ),
            "gradient_relative_error_max": float(
                np.max([row.gradient_relative_error for row in rows])
            ),
            "target_alignment_error_mean": float(
                np.mean([row.target_alignment_error for row in rows])
            ),
            "descent_alignment_ratio_mean": float(
                np.mean([row.descent_alignment_ratio for row in rows])
            ),
            "second_forward_cost_ratio_mean": float(
                np.mean([row.second_forward_cost_ratio for row in rows])
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
    if args.sequence_length < 4:
        raise ValueError("--sequence-length must be at least 4")
    if args.state_dim < 2 or args.input_dim < 1:
        raise ValueError("--state-dim must be >= 2 and --input-dim must be >= 1")

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
        "positive_condition_exact_dual_gradient": all(
            trial.dual_state_relative_error <= args.max_dual_state_error
            and trial.gradient_relative_error <= args.max_gradient_error
            for trial in trials
            if trial.condition == "positive"
        ),
        "nonlinear_control_all_fail_identity": all(
            (not trial.passed)
            and trial.fail_reason == "nonlinear_transition_breaks_dual_forward_identity"
            for trial in trials
            if trial.condition == "nonlinear_transition_fail"
        ),
        "target_shift_control_all_fail_alignment": all(
            (not trial.passed)
            and trial.fail_reason == "target_forcing_not_aligned_with_autonomous_tokens"
            for trial in trials
            if trial.condition == "target_shift_fail"
        ),
        "second_forward_cost_control_all_fail_price": all(
            (not trial.passed)
            and trial.fail_reason == "second_forward_recovers_dense_backward_cost"
            for trial in trials
            if trial.condition == "second_forward_cost_fail"
        ),
        "all_conditions_identified": all(trial.identifying_correct for trial in trials),
    }
    decision = (
        "PASS_R501_S51_TFDS_GATE"
        if all(required_passes.values())
        else "FAIL_R501_S51_TFDS_GATE"
    )
    disposition = (
        "synthetic_positive_control_identified_not_training_speed_claim"
        if decision.startswith("PASS_")
        else "tfds_gate_failed_or_non_identifying"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "CPU synthetic target-forced dual-state forward-flow gate. It "
            "checks a linear two-forward identity between target residuals, "
            "dual state differences, and dense two-state gradient references. "
            "It is not heldout CE progress, full BPTT replacement, or GPU "
            "training-speed evidence."
        ),
        "finding": {
            "round": "R501",
            "aspect": "Speed",
            "id": "R501-S51",
            "title": "Target-Forced Dual-State Forward Flow",
        },
        "thresholds": {
            "max_dual_state_error": args.max_dual_state_error,
            "max_gradient_error": args.max_gradient_error,
            "max_target_alignment_error": args.max_target_alignment_error,
            "min_descent_alignment_ratio": args.min_descent_alignment_ratio,
            "max_priced_work_ratio": args.max_priced_work_ratio,
        },
        "shape": {
            "seed": args.seed,
            "seed_count": args.seed_count,
            "sequence_length": args.sequence_length,
            "state_dim": args.state_dim,
            "input_dim": args.input_dim,
        },
        "required_passes": required_passes,
        "aggregate": aggregate(trials),
        "trials": [asdict(trial) for trial in trials],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    positive = result["aggregate"]["positive"]
    print(
        "[R501_S51_TFDS_GATE] "
        f"active=1 decision={decision} disposition={disposition} "
        f"positive_dual_state_error_mean={positive['dual_state_relative_error_mean']:.6e} "
        f"positive_gradient_error_mean={positive['gradient_relative_error_mean']:.6e} "
        f"positive_target_alignment_error_mean={positive['target_alignment_error_mean']:.6e} "
        f"positive_descent_alignment_ratio_mean={positive['descent_alignment_ratio_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
