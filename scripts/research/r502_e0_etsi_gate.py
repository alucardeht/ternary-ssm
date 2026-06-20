#!/usr/bin/env python3
"""R502 E0 exact ternary state inversion gate.

This CPU diagnostic implements the R502 Efficiency finding:
"Exact Ternary State Inversion (ETSI)".

The gate constructs a diagonal ternary SSM recurrence
h[t] = A * h[t-1] + B x[t], with A_i in {-1, 0, 1}. For non-zero A_i, the
previous state is recovered exactly by A_i * (h[t] - B_i x[t]). For A_i = 0,
the temporal chain is severed and the previous state is recovered from the
previous input branch B_i x[t-1].

Controls exercise the declared risks: recomputing Bx with FP drift breaks the
zero-residual certificate, mishandling the A_i=0 branch loses state, missing the
previous input makes zero-state reconstruction non-identifying, and storing too
much state erases the VRAM denominator.

This is a synthetic CPU exact-inversion and memory-ratio diagnostic, not heldout
CE progress, real corpus token efficiency, GPU training-speed evidence, or
production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r502_e0_etsi_gate.v1"
CONDITIONS = (
    "positive",
    "fp_drift_fail",
    "zero_branch_missing_fail",
    "previous_input_missing_fail",
    "state_cache_cost_fail",
)


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
    zero_fraction: float
    reconstruction_l2_max: float
    nonzero_reconstruction_l2_max: float
    zero_branch_reconstruction_l2_max: float
    objective_loss_delta_abs_max: float
    cached_state_vram_ratio: float
    projected_vram_saving_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50200)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sequence-len", type=int, default=96)
    parser.add_argument("--state-dim", type=int, default=48)
    parser.add_argument("--input-dim", type=int, default=7)
    parser.add_argument("--zero-fraction", type=float, default=0.25)
    parser.add_argument("--max-reconstruction-l2", type=float, default=1.0e-12)
    parser.add_argument("--max-objective-loss-delta", type=float, default=1.0e-12)
    parser.add_argument("--max-cached-state-vram-ratio", type=float, default=0.10)
    parser.add_argument("--expensive-cached-state-vram-ratio", type=float, default=0.16)
    return parser.parse_args()


def make_problem(args: argparse.Namespace, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zero_count = int(round(args.zero_fraction * args.state_dim))
    diag = np.empty(args.state_dim, dtype=np.float64)
    diag[:zero_count] = 0.0
    signs = rng.choice(np.array([-1.0, 1.0]), size=args.state_dim - zero_count)
    diag[zero_count:] = signs
    rng.shuffle(diag)

    input_matrix = rng.integers(-3, 4, size=(args.state_dim, args.input_dim)).astype(np.float64)
    inputs = rng.integers(-5, 6, size=(args.sequence_len, args.input_dim)).astype(np.float64)
    return diag, input_matrix, inputs


def forward_states(
    diag: np.ndarray,
    input_matrix: np.ndarray,
    inputs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    sequence_len = inputs.shape[0]
    state_dim = diag.shape[0]
    states = np.zeros((sequence_len, state_dim), dtype=np.float64)
    bx0 = input_matrix @ inputs[0]
    initial = rng.integers(-8, 9, size=state_dim).astype(np.float64)
    states[0] = np.where(diag == 0.0, bx0, initial)
    for t in range(1, sequence_len):
        states[t] = diag * states[t - 1] + input_matrix @ inputs[t]
    return states


def reconstruct_states(
    diag: np.ndarray,
    input_matrix: np.ndarray,
    inputs: np.ndarray,
    final_state: np.ndarray,
    rng: np.random.Generator,
    condition: str,
) -> np.ndarray:
    sequence_len = inputs.shape[0]
    state_dim = diag.shape[0]
    reconstructed = np.zeros((sequence_len, state_dim), dtype=np.float64)
    reconstructed[-1] = final_state
    nonzero = diag != 0.0
    zero = ~nonzero
    for t in reversed(range(1, sequence_len)):
        bx_t = input_matrix @ inputs[t]
        bx_prev = input_matrix @ inputs[t - 1]
        if condition == "fp_drift_fail":
            bx_t = bx_t + rng.normal(scale=2.0e-4, size=state_dim)
            bx_prev = bx_prev + rng.normal(scale=2.0e-4, size=state_dim)

        previous = np.zeros(state_dim, dtype=np.float64)
        previous[nonzero] = diag[nonzero] * (reconstructed[t, nonzero] - bx_t[nonzero])
        if condition == "zero_branch_missing_fail":
            previous[zero] = 0.0
        elif condition == "previous_input_missing_fail":
            previous[zero] = reconstructed[t, zero]
        else:
            previous[zero] = bx_prev[zero]
        reconstructed[t - 1] = previous
    return reconstructed


def objective_loss_delta_abs_max(
    states: np.ndarray,
    reconstructed: np.ndarray,
    rng: np.random.Generator,
) -> float:
    readout = rng.normal(scale=0.15, size=(states.shape[1], 11))
    target = states @ readout
    candidate = reconstructed @ readout
    per_step_delta = np.mean((candidate - target) * (candidate - target), axis=1)
    return float(np.max(np.abs(per_step_delta)))


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    diag, input_matrix, inputs = make_problem(args, rng)
    states = forward_states(diag, input_matrix, inputs, rng)
    reconstructed = reconstruct_states(
        diag,
        input_matrix,
        inputs,
        states[-1],
        rng,
        condition,
    )
    errors = reconstructed - states
    nonzero = diag != 0.0
    zero = ~nonzero
    per_step_l2 = np.linalg.norm(errors, axis=1)
    nonzero_l2 = np.linalg.norm(errors[:, nonzero], axis=1)
    zero_l2 = np.linalg.norm(errors[:, zero], axis=1)
    reconstruction_l2_max = float(np.max(per_step_l2))
    nonzero_l2_max = float(np.max(nonzero_l2))
    zero_l2_max = float(np.max(zero_l2))
    loss_delta = objective_loss_delta_abs_max(states, reconstructed, rng)

    baseline_state_values = args.sequence_len * args.state_dim
    etsi_state_values = args.state_dim
    vram_ratio = etsi_state_values / baseline_state_values
    if condition == "state_cache_cost_fail":
        vram_ratio = args.expensive_cached_state_vram_ratio
    saving_multiplier = 1.0 / vram_ratio
    zero_fraction = float(np.mean(zero))

    failed: list[str] = []
    if not finite(
        zero_fraction,
        reconstruction_l2_max,
        nonzero_l2_max,
        zero_l2_max,
        loss_delta,
        vram_ratio,
        saving_multiplier,
    ):
        failed.append("non_finite_metric")
    if reconstruction_l2_max > args.max_reconstruction_l2:
        failed.append("reconstruction")
    if nonzero_l2_max > args.max_reconstruction_l2:
        failed.append("nonzero_inverse")
    if zero_l2_max > args.max_reconstruction_l2:
        failed.append("zero_branch")
    if loss_delta > args.max_objective_loss_delta:
        failed.append("objective_delta")
    if vram_ratio > args.max_cached_state_vram_ratio:
        failed.append("vram_price")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed) if failed else None,
        sequence_len=args.sequence_len,
        state_dim=args.state_dim,
        input_dim=args.input_dim,
        zero_fraction=zero_fraction,
        reconstruction_l2_max=float(reconstruction_l2_max),
        nonzero_reconstruction_l2_max=float(nonzero_l2_max),
        zero_branch_reconstruction_l2_max=float(zero_l2_max),
        objective_loss_delta_abs_max=float(loss_delta),
        cached_state_vram_ratio=float(vram_ratio),
        projected_vram_saving_multiplier=float(saving_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "sequence_len",
        "state_dim",
        "input_dim",
        "zero_fraction",
        "reconstruction_l2_max",
        "nonzero_reconstruction_l2_max",
        "zero_branch_reconstruction_l2_max",
        "objective_loss_delta_abs_max",
        "cached_state_vram_ratio",
        "projected_vram_saving_multiplier",
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
    drift = [result for result in results if result.condition == "fp_drift_fail"]
    zero_missing = [result for result in results if result.condition == "zero_branch_missing_fail"]
    input_missing = [result for result in results if result.condition == "previous_input_missing_fail"]
    cost = [result for result in results if result.condition == "state_cache_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_reconstruction": (
            aggregates["positive"]["reconstruction_l2_max_max"] <= 1.0e-12
            and aggregates["positive"]["objective_loss_delta_abs_max_max"] <= 1.0e-12
        ),
        "positive_condition_zero_branch_exercised": (
            aggregates["positive"]["zero_fraction_mean"] > 0.0
            and aggregates["positive"]["zero_branch_reconstruction_l2_max_max"] <= 1.0e-12
        ),
        "positive_condition_vram_priced": (
            aggregates["positive"]["cached_state_vram_ratio_mean"] <= 0.10
        ),
        "fp_drift_control_all_fail_reconstruction": all(
            (not result.passed)
            and result.fail_reason is not None
            and ("reconstruction" in result.fail_reason or "objective_delta" in result.fail_reason)
            for result in drift
        ),
        "zero_branch_control_all_fail_zero_branch": all(
            (not result.passed)
            and result.fail_reason is not None
            and "zero_branch" in result.fail_reason
            for result in zero_missing
        ),
        "previous_input_control_all_fail_zero_branch": all(
            (not result.passed)
            and result.fail_reason is not None
            and "zero_branch" in result.fail_reason
            for result in input_missing
        ),
        "cost_control_all_fail_vram_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "vram_price" in result.fail_reason
            for result in cost
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
    decision = "PASS_R502_E0_ETSI_GATE" if all(checks.values()) else "FAIL_R502_E0_ETSI_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU exact ternary state inversion gate only; not heldout "
            "CE progress, not real corpus token efficiency, not GPU training "
            "speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_E0_ETSI_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_reconstruction_l2_max_max={positive['reconstruction_l2_max_max']:.12e} "
        f"positive_zero_branch_l2_max_max={positive['zero_branch_reconstruction_l2_max_max']:.12e} "
        f"positive_objective_loss_delta_abs_max_max={positive['objective_loss_delta_abs_max_max']:.12e} "
        f"positive_cached_state_vram_ratio_mean={positive['cached_state_vram_ratio_mean']:.6f} "
        f"positive_projected_vram_saving_multiplier_mean={positive['projected_vram_saving_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
