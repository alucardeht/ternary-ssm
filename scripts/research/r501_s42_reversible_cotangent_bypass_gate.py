#!/usr/bin/env python3
"""R501 S42 reversible SSM cotangent bypass gate.

This CPU diagnostic implements the R501 Speed Math 9 finding:
"Reversible SSM Cotangent Bypass".

The gate checks whether a reversible state transition can reconstruct forward
states during the backward cotangent pass without storing the full activation
cache. The finding survives only when reconstruction is numerically exact enough,
cached and reconstructed gradients agree, the reversible parameterization does
not change the target loss on the positive fixture, and a memory-limited
batch-scaling price model clears the declared 0.50x time-to-target inequality.

Controls exercise the declared kills: a compute-bound regime where deleting the
activation cache does not pay for the reverse pass, a non-reversible target whose
best reversible/stable surrogate has a loss gap, and an ill-conditioned inverse
that amplifies final-state precision noise during reconstruction.

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


SCHEMA = "r501_s42_reversible_cotangent_bypass_gate.v1"
CONDITIONS = (
    "positive",
    "compute_bound_fail",
    "nonreversible_expressivity_fail",
    "ill_conditioned_inverse_fail",
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
    input_dim: int
    output_dim: int
    condition_number: float
    reconstruction_relative_error: float
    reconstruction_max_abs_error: float
    loss_gap_ratio: float
    grad_b_relative_error: float
    activation_memory_ratio: float
    baseline_step_cost: float
    candidate_step_cost: float
    projected_time_to_target_ratio: float
    projected_speedup: float
    price_components: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50142)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--segment-length", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=24)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=10)
    parser.add_argument("--batch-multiplier", type=float, default=2.0)
    parser.add_argument("--reverse-recompute-cost-ratio", type=float, default=0.25)
    parser.add_argument("--positive-cache-io-cost", type=float, default=1.55)
    parser.add_argument("--compute-bound-cache-io-cost", type=float, default=0.05)
    parser.add_argument("--activation-audit-cost", type=float, default=0.020)
    parser.add_argument("--final-state-noise", type=float, default=1.0e-12)
    parser.add_argument("--ill-conditioned-final-state-noise", type=float, default=1.0e-8)
    parser.add_argument("--max-reconstruction-relative-error", type=float, default=1.0e-7)
    parser.add_argument("--max-grad-b-relative-error", type=float, default=1.0e-7)
    parser.add_argument("--max-loss-gap-ratio", type=float, default=1.0e-8)
    parser.add_argument("--max-activation-memory-ratio", type=float, default=0.025)
    parser.add_argument("--max-time-to-target-ratio", type=float, default=0.50)
    return parser.parse_args()


def orthogonal_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def transition_pair(
    rng: np.random.Generator,
    state_dim: int,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if condition in {"positive", "compute_bound_fail"}:
        a = orthogonal_matrix(rng, state_dim)
        return a, a, a.T

    if condition == "nonreversible_expressivity_fail":
        left = orthogonal_matrix(rng, state_dim)
        right = orthogonal_matrix(rng, state_dim)
        target_singulars = np.linspace(0.18, 0.72, state_dim)
        target = left @ np.diag(target_singulars) @ right.T
        candidate = left @ right.T
        return target, candidate, np.linalg.inv(candidate)

    if condition == "ill_conditioned_inverse_fail":
        left = orthogonal_matrix(rng, state_dim)
        right = orthogonal_matrix(rng, state_dim)
        singulars = np.geomspace(0.94, 0.995, state_dim)
        a = left @ np.diag(singulars) @ right.T
        return a, a, np.linalg.inv(a)

    raise ValueError(f"unknown condition: {condition}")


def input_sequence(
    rng: np.random.Generator,
    segment_length: int,
    input_dim: int,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, segment_length, dtype=np.float64)
    columns = [
        np.ones_like(t),
        np.sin(2.0 * math.pi * t),
        np.cos(2.0 * math.pi * t),
        np.sin(4.0 * math.pi * t + 0.3),
        np.cos(6.0 * math.pi * t - 0.2),
        t,
        t * t,
        np.sign(np.sin(5.0 * math.pi * t)),
    ]
    x = np.stack(columns[:input_dim], axis=1)
    if input_dim > len(columns):
        extra = rng.normal(scale=0.25, size=(segment_length, input_dim - len(columns)))
        x = np.concatenate([x, extra], axis=1)
    return x.astype(np.float64)


def output_residual_sequence(
    rng: np.random.Generator,
    segment_length: int,
    output_dim: int,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, segment_length, dtype=np.float64)
    columns = [
        np.sin(3.0 * math.pi * t + 0.1),
        np.cos(5.0 * math.pi * t - 0.4),
        t - 0.5,
        (t - 0.5) * (t - 0.5),
    ]
    basis = np.stack(columns, axis=1)
    weights = rng.normal(scale=0.040, size=(basis.shape[1], output_dim))
    return (basis @ weights).astype(np.float64)


def forward_states(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    states = np.zeros((x.shape[0] + 1, a.shape[0]), dtype=np.float64)
    for t in range(x.shape[0]):
        states[t + 1] = a @ states[t] + b @ x[t]
    return states


def reverse_reconstruct(
    a_inv: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    final_state: np.ndarray,
) -> np.ndarray:
    states = np.zeros((x.shape[0] + 1, a_inv.shape[0]), dtype=np.float64)
    states[-1] = final_state
    for t in range(x.shape[0] - 1, -1, -1):
        states[t] = a_inv @ (states[t + 1] - b @ x[t])
    return states


def loss_and_grad_b(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    states: np.ndarray,
) -> tuple[float, np.ndarray]:
    segment_length = x.shape[0]
    pred = states[1:] @ c.T
    residual = pred - y
    loss = 0.5 * float(np.mean(np.sum(residual * residual, axis=1)))
    grad_h = (residual @ c) / float(segment_length)
    carry = np.zeros(a.shape[0], dtype=np.float64)
    grad_b = np.zeros_like(b)
    for t in range(segment_length - 1, -1, -1):
        carry = grad_h[t] + a.T @ carry
        grad_b += np.outer(carry, x[t])
    return loss, grad_b


def relative_norm(error: np.ndarray, exact: np.ndarray) -> float:
    return float(np.linalg.norm(error.ravel()) / max(np.linalg.norm(exact.ravel()), 1.0e-30))


def price_route(
    args: argparse.Namespace,
    condition: str,
    activation_memory_ratio: float,
) -> dict[str, float]:
    cache_io_cost = (
        args.compute_bound_cache_io_cost
        if condition == "compute_bound_fail"
        else args.positive_cache_io_cost
    )
    baseline_step_cost = 1.0 + cache_io_cost
    candidate_step_cost = (
        1.0
        + args.reverse_recompute_cost_ratio
        + args.activation_audit_cost
        + activation_memory_ratio
    )
    return {
        "baseline_step_cost": float(baseline_step_cost),
        "candidate_step_cost": float(candidate_step_cost),
        "batch_multiplier": float(args.batch_multiplier),
        "activation_memory_ratio": float(activation_memory_ratio),
        "reverse_recompute_cost_ratio": float(args.reverse_recompute_cost_ratio),
        "activation_audit_cost": float(args.activation_audit_cost),
    }


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    target_a, candidate_a, candidate_a_inv = transition_pair(rng, args.state_dim, condition)
    b = rng.normal(scale=0.060, size=(args.state_dim, args.input_dim))
    c = rng.normal(scale=0.080, size=(args.output_dim, args.state_dim))
    x = input_sequence(rng, args.segment_length, args.input_dim)

    target_states = forward_states(target_a, b, x)
    candidate_states = forward_states(candidate_a, b, x)
    y = target_states[1:] @ c.T + output_residual_sequence(
        rng,
        args.segment_length,
        args.output_dim,
    )
    target_loss, _ = loss_and_grad_b(target_a, b, c, x, y, target_states)
    candidate_loss, grad_b_cached = loss_and_grad_b(candidate_a, b, c, x, y, candidate_states)

    final_state_noise = (
        args.ill_conditioned_final_state_noise
        if condition == "ill_conditioned_inverse_fail"
        else args.final_state_noise
    )
    noisy_final = candidate_states[-1] + rng.normal(
        scale=final_state_noise,
        size=candidate_states[-1].shape,
    )
    reconstructed = reverse_reconstruct(candidate_a_inv, b, x, noisy_final)
    _, grad_b_reconstructed = loss_and_grad_b(candidate_a, b, c, x, y, reconstructed)

    reconstruction_error = reconstructed - candidate_states
    reconstruction_relative_error = relative_norm(reconstruction_error, candidate_states)
    reconstruction_max_abs_error = float(np.max(np.abs(reconstruction_error)))
    loss_gap_ratio = float(
        abs(candidate_loss - target_loss)
        / max(float(np.mean(np.sum(y * y, axis=1))), 1.0e-30)
    )
    grad_b_relative_error = relative_norm(grad_b_reconstructed - grad_b_cached, grad_b_cached)
    activation_memory_ratio = float(2.0 / float(args.segment_length + 1))
    condition_number = float(np.linalg.cond(candidate_a))
    price_components = price_route(args, condition, activation_memory_ratio)
    baseline_step_cost = price_components["baseline_step_cost"]
    candidate_step_cost = price_components["candidate_step_cost"]
    projected_time_ratio = float(
        (candidate_step_cost / baseline_step_cost) / max(args.batch_multiplier, 1.0e-30)
    )
    projected_speedup = 1.0 / projected_time_ratio if projected_time_ratio > 0.0 else math.inf

    reconstruction_pass = (
        reconstruction_relative_error <= args.max_reconstruction_relative_error
    )
    grad_pass = grad_b_relative_error <= args.max_grad_b_relative_error
    expressivity_pass = loss_gap_ratio <= args.max_loss_gap_ratio
    memory_pass = activation_memory_ratio <= args.max_activation_memory_ratio
    price_pass = projected_time_ratio <= args.max_time_to_target_ratio
    passed = bool(reconstruction_pass and grad_pass and expressivity_pass and memory_pass and price_pass)
    expected_pass = condition == "positive"
    identifying_correct = passed == expected_pass

    fail_reason = None
    if not passed:
        if not reconstruction_pass:
            fail_reason = "inverse_reconstruction_unstable"
        elif not grad_pass:
            fail_reason = "reconstructed_cotangent_gradient_mismatch"
        elif not expressivity_pass:
            fail_reason = "reversible_parameterization_expressivity_gap"
        elif not memory_pass:
            fail_reason = "activation_memory_not_reduced_enough"
        elif not price_pass:
            fail_reason = "batch_scaling_not_enough_without_activation_cache_bottleneck"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=identifying_correct,
        fail_reason=fail_reason,
        segment_length=args.segment_length,
        state_dim=args.state_dim,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        condition_number=condition_number,
        reconstruction_relative_error=reconstruction_relative_error,
        reconstruction_max_abs_error=reconstruction_max_abs_error,
        loss_gap_ratio=loss_gap_ratio,
        grad_b_relative_error=grad_b_relative_error,
        activation_memory_ratio=activation_memory_ratio,
        baseline_step_cost=baseline_step_cost,
        candidate_step_cost=candidate_step_cost,
        projected_time_to_target_ratio=projected_time_ratio,
        projected_speedup=float(projected_speedup),
        price_components=price_components,
    )


def finite_trial(trial: TrialResult) -> bool:
    values = (
        trial.condition_number,
        trial.reconstruction_relative_error,
        trial.reconstruction_max_abs_error,
        trial.loss_gap_ratio,
        trial.grad_b_relative_error,
        trial.activation_memory_ratio,
        trial.baseline_step_cost,
        trial.candidate_step_cost,
        trial.projected_time_to_target_ratio,
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
            "condition_number_mean": float(np.mean([row.condition_number for row in rows])),
            "condition_number_max": float(np.max([row.condition_number for row in rows])),
            "reconstruction_relative_error_mean": float(
                np.mean([row.reconstruction_relative_error for row in rows])
            ),
            "reconstruction_relative_error_max": float(
                np.max([row.reconstruction_relative_error for row in rows])
            ),
            "reconstruction_max_abs_error_max": float(
                np.max([row.reconstruction_max_abs_error for row in rows])
            ),
            "loss_gap_ratio_mean": float(np.mean([row.loss_gap_ratio for row in rows])),
            "loss_gap_ratio_max": float(np.max([row.loss_gap_ratio for row in rows])),
            "grad_b_relative_error_mean": float(
                np.mean([row.grad_b_relative_error for row in rows])
            ),
            "grad_b_relative_error_max": float(
                np.max([row.grad_b_relative_error for row in rows])
            ),
            "activation_memory_ratio_mean": float(
                np.mean([row.activation_memory_ratio for row in rows])
            ),
            "baseline_step_cost_mean": float(np.mean([row.baseline_step_cost for row in rows])),
            "candidate_step_cost_mean": float(
                np.mean([row.candidate_step_cost for row in rows])
            ),
            "projected_time_to_target_ratio_mean": float(
                np.mean([row.projected_time_to_target_ratio for row in rows])
            ),
            "projected_speedup_mean": float(
                np.mean([row.projected_speedup for row in rows])
            ),
        }
    return out


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.segment_length < 8:
        raise ValueError("--segment-length must be at least 8")
    if args.state_dim < 2 or args.input_dim < 1 or args.output_dim < 1:
        raise ValueError("dimensions must be positive")
    if args.batch_multiplier <= 1.0:
        raise ValueError("--batch-multiplier must be greater than 1")

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
        "positive_condition_reconstructs_exactly": all(
            trial.reconstruction_relative_error <= args.max_reconstruction_relative_error
            for trial in trials
            if trial.condition == "positive"
        ),
        "positive_condition_gradients_match": all(
            trial.grad_b_relative_error <= args.max_grad_b_relative_error
            for trial in trials
            if trial.condition == "positive"
        ),
        "compute_bound_control_all_fail_price": all(
            (not trial.passed)
            and trial.fail_reason == "batch_scaling_not_enough_without_activation_cache_bottleneck"
            for trial in trials
            if trial.condition == "compute_bound_fail"
        ),
        "nonreversible_control_all_fail_expressivity": all(
            (not trial.passed)
            and trial.fail_reason == "reversible_parameterization_expressivity_gap"
            for trial in trials
            if trial.condition == "nonreversible_expressivity_fail"
        ),
        "ill_conditioned_control_all_fail_reconstruction": all(
            (not trial.passed)
            and trial.fail_reason == "inverse_reconstruction_unstable"
            for trial in trials
            if trial.condition == "ill_conditioned_inverse_fail"
        ),
        "all_conditions_identified": all(trial.identifying_correct for trial in trials),
    }
    decision = (
        "PASS_R501_S42_REVERSIBLE_COTANGENT_BYPASS_GATE"
        if all(required_passes.values())
        else "FAIL_R501_S42_REVERSIBLE_COTANGENT_BYPASS_GATE"
    )
    disposition = (
        "synthetic_positive_control_identified_not_training_speed_claim"
        if decision.startswith("PASS_")
        else "reversible_cotangent_bypass_gate_failed_or_non_identifying"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "CPU synthetic reversible cotangent bypass gate. It checks exact "
            "state reconstruction, reconstructed-gradient agreement, reversible "
            "parameterization loss parity, activation-memory deletion, and a "
            "priced memory-limited 2x batch model. It is not heldout CE progress "
            "or GPU training-speed evidence."
        ),
        "finding": {
            "round": "R501",
            "aspect": "Speed",
            "id": "R501-S42",
            "title": "Reversible SSM Cotangent Bypass",
        },
        "thresholds": {
            "max_reconstruction_relative_error": args.max_reconstruction_relative_error,
            "max_grad_b_relative_error": args.max_grad_b_relative_error,
            "max_loss_gap_ratio": args.max_loss_gap_ratio,
            "max_activation_memory_ratio": args.max_activation_memory_ratio,
            "max_time_to_target_ratio": args.max_time_to_target_ratio,
            "required_projected_speedup": 1.0 / args.max_time_to_target_ratio,
        },
        "shape": {
            "seed": args.seed,
            "seed_count": args.seed_count,
            "segment_length": args.segment_length,
            "state_dim": args.state_dim,
            "input_dim": args.input_dim,
            "output_dim": args.output_dim,
            "batch_multiplier": args.batch_multiplier,
        },
        "required_passes": required_passes,
        "aggregate": aggregate(trials),
        "trials": [asdict(trial) for trial in trials],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    positive = result["aggregate"]["positive"]
    print(
        "[R501_S42_REVERSIBLE_COTANGENT_BYPASS_GATE] "
        f"active=1 decision={decision} disposition={disposition} "
        f"positive_reconstruction_error_mean={positive['reconstruction_relative_error_mean']:.6e} "
        f"positive_grad_b_error_mean={positive['grad_b_relative_error_mean']:.6e} "
        f"positive_loss_gap_mean={positive['loss_gap_ratio_mean']:.6e} "
        f"positive_activation_memory_ratio_mean={positive['activation_memory_ratio_mean']:.6f} "
        f"positive_time_to_target_ratio_mean={positive['projected_time_to_target_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
