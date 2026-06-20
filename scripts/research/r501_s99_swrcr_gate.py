#!/usr/bin/env python3
"""R501 S99 sub-warp recurrent checkpoint resampling gate.

This CPU diagnostic implements the sixth R501 Hardware Efficiency mathematical
finding: "Sub-Warp Recurrent Checkpoint Resampling (SWRCR)".

The gate constructs a ternary recurrent SSM trajectory, stores only k-spaced
checkpoints, and recomputes each local segment during the backward-style pass.
The positive condition must exactly reproduce the dense state trajectory while
charging only checkpoint residency and a priced sub-warp register window.

Controls exercise the declared kill criteria: corrupted checkpoints break
reconstruction, approximate recomputation creates drift, dense residency erases
the memory denominator, non-ternary recomputation is not hidden by memory-fetch
slack, and oversized intervals exceed the sub-warp register window.

This is a synthetic CPU recurrent-checkpoint resampling diagnostic, not heldout
CE progress, real VRAM telemetry, GPU training-speed evidence, or production
v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s99_swrcr_gate.v1"
CONDITIONS = (
    "positive",
    "corrupted_checkpoint_fail",
    "approximate_recompute_fail",
    "dense_residency_fail",
    "nonternary_compute_fail",
    "oversized_interval_fail",
)
FP16_BITS = 16.0


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    sequence_len: int
    state_dim: int
    input_dim: int
    checkpoint_interval: int
    checkpoint_count: int
    reconstruction_relative_error: float
    reconstruction_max_abs_error: float
    checkpoint_state_ratio: float
    hidden_latency_ratio: float
    register_window_fit_ratio: float
    dense_state_bits: float
    candidate_state_bits: float
    paid_state_bit_ratio: float
    projected_state_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50199)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sequence-len", type=int, default=512)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=8)
    parser.add_argument("--oversized-interval", type=int, default=64)
    parser.add_argument("--max-subwarp-interval", type=int, default=16)
    parser.add_argument("--ternary-cycle-per-value", type=float, default=1.0)
    parser.add_argument("--nonternary-cycle-per-value", type=float, default=12.0)
    parser.add_argument("--memory-fetch-cycle-per-value", type=float, default=8.0)
    parser.add_argument("--max-reconstruction-relative-error", type=float, default=1.0e-12)
    parser.add_argument("--max-reconstruction-abs-error", type=float, default=1.0e-12)
    parser.add_argument("--max-paid-state-bit-ratio", type=float, default=0.20)
    parser.add_argument("--min-state-efficiency-multiplier", type=float, default=4.0)
    parser.add_argument("--max-hidden-latency-ratio", type=float, default=1.0)
    parser.add_argument("--max-register-window-fit-ratio", type=float, default=1.0)
    return parser.parse_args()


def make_fixture(
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transition = rng.choice(np.array([-1.0, 0.0, 1.0], dtype=np.float64), size=args.state_dim)
    input_matrix = (
        rng.choice(np.array([-1.0, 0.0, 1.0], dtype=np.float64), size=(args.state_dim, args.input_dim))
        / math.sqrt(args.input_dim)
    )
    inputs = rng.normal(size=(args.sequence_len, args.input_dim)).astype(np.float64)
    initial = rng.normal(size=args.state_dim).astype(np.float64)
    return transition, input_matrix, inputs, initial


def forward_trajectory(
    transition: np.ndarray,
    input_matrix: np.ndarray,
    inputs: np.ndarray,
    initial: np.ndarray,
) -> np.ndarray:
    h = np.zeros((inputs.shape[0] + 1, transition.shape[0]), dtype=np.float64)
    h[0] = initial
    for t in range(1, inputs.shape[0] + 1):
        h[t] = transition * h[t - 1] + input_matrix @ inputs[t - 1]
    return h


def checkpoint_indices(sequence_len: int, interval: int) -> list[int]:
    indices = list(range(0, sequence_len + 1, interval))
    if indices[-1] != sequence_len:
        indices.append(sequence_len)
    return indices


def recompute_from_checkpoints(
    dense: np.ndarray,
    transition: np.ndarray,
    input_matrix: np.ndarray,
    inputs: np.ndarray,
    interval: int,
    condition: str,
) -> tuple[np.ndarray, int]:
    indices = checkpoint_indices(inputs.shape[0], interval)
    checkpoints = {idx: dense[idx].copy() for idx in indices}
    if condition == "corrupted_checkpoint_fail" and len(indices) > 2:
        checkpoints[indices[len(indices) // 2]] = checkpoints[indices[len(indices) // 2]] + 0.125

    if condition == "approximate_recompute_fail":
        recompute_transition = transition + 0.015
    else:
        recompute_transition = transition

    reconstructed = np.zeros_like(dense)
    recomputed_steps = 0
    for start, end in zip(indices[:-1], indices[1:]):
        reconstructed[start] = checkpoints[start]
        state = checkpoints[start].copy()
        for t in range(start + 1, end + 1):
            state = recompute_transition * state + input_matrix @ inputs[t - 1]
            reconstructed[t] = state
            recomputed_steps += 1
    return reconstructed, recomputed_steps


def relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference))
    return float(np.linalg.norm(candidate - reference) / max(denom, 1.0e-30))


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    transition, input_matrix, inputs, initial = make_fixture(rng, args)
    dense = forward_trajectory(transition, input_matrix, inputs, initial)

    interval = args.oversized_interval if condition == "oversized_interval_fail" else args.checkpoint_interval
    if condition == "dense_residency_fail":
        reconstructed = dense.copy()
        recomputed_steps = 0
    else:
        reconstructed, recomputed_steps = recompute_from_checkpoints(
            dense=dense,
            transition=transition,
            input_matrix=input_matrix,
            inputs=inputs,
            interval=interval,
            condition=condition,
        )

    rel_error = relative_error(reconstructed, dense)
    max_abs = float(np.max(np.abs(reconstructed - dense)))
    checkpoints = checkpoint_indices(args.sequence_len, interval)
    checkpoint_count = len(checkpoints)
    dense_state_count = args.sequence_len + 1
    if condition == "dense_residency_fail":
        stored_state_count = dense_state_count
    else:
        stored_state_count = checkpoint_count

    replaced_states = max(dense_state_count - stored_state_count, 1)
    cycle_per_value = (
        args.nonternary_cycle_per_value
        if condition == "nonternary_compute_fail"
        else args.ternary_cycle_per_value
    )
    recompute_cycles = recomputed_steps * args.state_dim * cycle_per_value
    memory_slack_cycles = replaced_states * args.state_dim * args.memory_fetch_cycle_per_value
    hidden_latency_ratio = float(recompute_cycles / max(memory_slack_cycles, 1.0e-30))
    register_fit = float(interval / max(args.max_subwarp_interval, 1))

    dense_bits = float(dense_state_count * args.state_dim * FP16_BITS)
    candidate_bits = float(stored_state_count * args.state_dim * FP16_BITS)
    paid_ratio = candidate_bits / dense_bits
    multiplier = 1.0 / paid_ratio

    failures: list[str] = []
    if rel_error > args.max_reconstruction_relative_error or max_abs > args.max_reconstruction_abs_error:
        failures.append("reconstruction_error")
    if paid_ratio > args.max_paid_state_bit_ratio or multiplier < args.min_state_efficiency_multiplier:
        failures.append("price_paid_state_bits")
    if hidden_latency_ratio > args.max_hidden_latency_ratio:
        failures.append("hidden_latency")
    if register_fit > args.max_register_window_fit_ratio:
        failures.append("register_window")

    fail_reason = "+".join(failures) if failures else None
    passed = fail_reason is None
    expected_pass = condition == "positive"
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=fail_reason,
        sequence_len=args.sequence_len,
        state_dim=args.state_dim,
        input_dim=args.input_dim,
        checkpoint_interval=interval,
        checkpoint_count=checkpoint_count,
        reconstruction_relative_error=rel_error,
        reconstruction_max_abs_error=max_abs,
        checkpoint_state_ratio=float(checkpoint_count / dense_state_count),
        hidden_latency_ratio=hidden_latency_ratio,
        register_window_fit_ratio=register_fit,
        dense_state_bits=dense_bits,
        candidate_state_bits=candidate_bits,
        paid_state_bit_ratio=paid_ratio,
        projected_state_efficiency_multiplier=multiplier,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "sequence_len",
        "state_dim",
        "input_dim",
        "checkpoint_interval",
        "checkpoint_count",
        "reconstruction_relative_error",
        "reconstruction_max_abs_error",
        "checkpoint_state_ratio",
        "hidden_latency_ratio",
        "register_window_fit_ratio",
        "dense_state_bits",
        "candidate_state_bits",
        "paid_state_bit_ratio",
        "projected_state_efficiency_multiplier",
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
    corrupted = [result for result in results if result.condition == "corrupted_checkpoint_fail"]
    approximate = [result for result in results if result.condition == "approximate_recompute_fail"]
    dense = [result for result in results if result.condition == "dense_residency_fail"]
    nonternary = [result for result in results if result.condition == "nonternary_compute_fail"]
    oversized = [result for result in results if result.condition == "oversized_interval_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_reconstruction": (
            aggregates["positive"]["reconstruction_relative_error_max"] <= 1.0e-12
            and aggregates["positive"]["reconstruction_max_abs_error_max"] <= 1.0e-12
        ),
        "positive_condition_priced": (
            aggregates["positive"]["paid_state_bit_ratio_mean"] <= 0.20
            and aggregates["positive"]["projected_state_efficiency_multiplier_mean"] >= 4.0
            and aggregates["positive"]["hidden_latency_ratio_mean"] <= 1.0
            and aggregates["positive"]["register_window_fit_ratio_mean"] <= 1.0
        ),
        "corrupted_checkpoint_control_all_fail_reconstruction": all(
            (not result.passed)
            and result.fail_reason is not None
            and "reconstruction_error" in result.fail_reason
            for result in corrupted
        ),
        "approximate_recompute_control_all_fail_reconstruction": all(
            (not result.passed)
            and result.fail_reason is not None
            and "reconstruction_error" in result.fail_reason
            for result in approximate
        ),
        "dense_residency_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "price_paid_state_bits" in result.fail_reason
            for result in dense
        ),
        "nonternary_compute_control_all_fail_latency": all(
            (not result.passed)
            and result.fail_reason is not None
            and "hidden_latency" in result.fail_reason
            for result in nonternary
        ),
        "oversized_interval_control_all_fail_register_window": all(
            (not result.passed)
            and result.fail_reason is not None
            and "register_window" in result.fail_reason
            for result in oversized
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
    decision = "PASS_R501_S99_SWRCR_GATE" if all(checks.values()) else "FAIL_R501_S99_SWRCR_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU sub-warp recurrent checkpoint resampling gate only; "
            "not heldout CE progress, not real VRAM telemetry, not GPU training "
            "speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S99_SWRCR_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_reconstruction_relative_error_max={positive['reconstruction_relative_error_max']:.6e} "
        f"positive_checkpoint_state_ratio_mean={positive['checkpoint_state_ratio_mean']:.6f} "
        f"positive_hidden_latency_ratio_mean={positive['hidden_latency_ratio_mean']:.6f} "
        f"positive_register_window_fit_ratio_mean={positive['register_window_fit_ratio_mean']:.6f} "
        f"positive_paid_state_bit_ratio_mean={positive['paid_state_bit_ratio_mean']:.6f} "
        f"positive_projected_state_efficiency_multiplier_mean={positive['projected_state_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
