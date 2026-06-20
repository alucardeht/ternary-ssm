#!/usr/bin/env python3
"""R502 E2 bounded-drift integer state accumulation gate.

This CPU diagnostic implements the R502 Efficiency finding:
"Bounded-Drift Integer State Accumulation (BDISA)".

The gate projects recurrent input branches into fixed-point INT32 values before
state accumulation. Because A_i is ternary, h[t] = A * h[t - 1] + floor(Bx *
2^k) is pure integer addition, subtraction, or zero-branch replacement. A
positive case must reconstruct the full state trajectory bitwise from the final
integer state and fixed-point branches over at least one million scalar state
transitions.

Controls exercise the declared risks: FP32 accumulation drifts once integer
states exceed float32 unit precision, using round instead of floor changes the
certified branch, INT32 overflow invalidates the representation, and conversion
overhead can erase the priced denominator.

This is a synthetic CPU fixed-point recurrence and reconstruction diagnostic,
not heldout CE progress, real corpus token efficiency, GPU training-speed
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


SCHEMA = "r502_e2_bdisa_gate.v1"
INT32_MAX = np.iinfo(np.int32).max
CONDITIONS = (
    "positive",
    "fp32_accumulation_drift_fail",
    "wrong_rounding_fail",
    "overflow_risk_fail",
    "conversion_cost_fail",
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
    transition_cells: int
    scale_bits: int
    zero_fraction: float
    branch_mismatch_count: int
    bitwise_mismatch_count: int
    reconstruction_linf_max: int
    fp32_state_mismatch_count: int
    state_abs_max: int
    int32_overflow_margin: int
    bx_quantization_error_abs_max: float
    state_cache_vram_ratio: float
    priced_state_denominator_ratio: float
    projected_state_denominator_saving_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50202)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sequence-len", type=int, default=32768)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--zero-fraction", type=float, default=0.25)
    parser.add_argument("--scale-bits", type=int, default=16)
    parser.add_argument("--min-transition-cells", type=int, default=1_000_000)
    parser.add_argument("--max-bx-quantization-error", type=float, default=1.0 / 65536.0)
    parser.add_argument("--max-priced-state-denominator-ratio", type=float, default=0.20)
    parser.add_argument("--conversion-overhead-ratio", type=float, default=0.035)
    parser.add_argument("--expensive-conversion-overhead-ratio", type=float, default=0.28)
    parser.add_argument("--fp32-drift-branch-value", type=int, default=1025)
    parser.add_argument("--overflow-branch-value", type=int, default=100000)
    return parser.parse_args()


def scale(args: argparse.Namespace) -> int:
    return 1 << args.scale_bits


def make_diag(args: argparse.Namespace, rng: np.random.Generator, condition: str) -> np.ndarray:
    if condition in {"fp32_accumulation_drift_fail", "overflow_risk_fail"}:
        return np.ones(args.state_dim, dtype=np.int64)

    zero_count = int(round(args.zero_fraction * args.state_dim))
    diag = np.empty(args.state_dim, dtype=np.int64)
    diag[:zero_count] = 0
    diag[zero_count:] = rng.choice(np.array([-1, 1], dtype=np.int64), size=args.state_dim - zero_count)
    rng.shuffle(diag)
    return diag


def make_branch_real(
    args: argparse.Namespace,
    rng: np.random.Generator,
    condition: str,
) -> np.ndarray:
    fixed_scale = scale(args)
    if condition == "fp32_accumulation_drift_fail":
        branch = np.full(
            (args.sequence_len, args.state_dim),
            args.fp32_drift_branch_value,
            dtype=np.int64,
        )
        return branch.astype(np.float64) / fixed_scale
    if condition == "overflow_risk_fail":
        branch = np.full(
            (args.sequence_len, args.state_dim),
            args.overflow_branch_value,
            dtype=np.int64,
        )
        return branch.astype(np.float64) / fixed_scale

    return rng.uniform(
        low=-0.115,
        high=0.115,
        size=(args.sequence_len, args.state_dim),
    )


def floor_branch(branch_real: np.ndarray, fixed_scale: int) -> np.ndarray:
    return np.floor(branch_real * fixed_scale).astype(np.int64)


def rounded_branch(branch_real: np.ndarray, fixed_scale: int) -> np.ndarray:
    return np.rint(branch_real * fixed_scale).astype(np.int64)


def forward_integer_states(
    diag: np.ndarray,
    branches: np.ndarray,
    initial: np.ndarray,
) -> np.ndarray:
    sequence_len, state_dim = branches.shape
    states = np.zeros((sequence_len, state_dim), dtype=np.int64)
    states[0] = diag * initial + branches[0]
    for t in range(1, sequence_len):
        states[t] = diag * states[t - 1] + branches[t]
    return states


def forward_fp32_states(diag: np.ndarray, branches: np.ndarray, initial: np.ndarray) -> np.ndarray:
    sequence_len, state_dim = branches.shape
    states = np.zeros((sequence_len, state_dim), dtype=np.float32)
    diag32 = diag.astype(np.float32)
    branch32 = branches.astype(np.float32)
    states[0] = diag32 * initial.astype(np.float32) + branch32[0]
    for t in range(1, sequence_len):
        states[t] = diag32 * states[t - 1] + branch32[t]
    return np.rint(states).astype(np.int64)


def reconstruct_integer_states(
    diag: np.ndarray,
    branches: np.ndarray,
    final_state: np.ndarray,
) -> np.ndarray:
    sequence_len, state_dim = branches.shape
    reconstructed = np.zeros((sequence_len, state_dim), dtype=np.int64)
    reconstructed[-1] = final_state
    nonzero = diag != 0
    zero = ~nonzero
    for t in reversed(range(1, sequence_len)):
        previous = np.zeros(state_dim, dtype=np.int64)
        previous[nonzero] = diag[nonzero] * (reconstructed[t, nonzero] - branches[t, nonzero])
        previous[zero] = branches[t - 1, zero]
        reconstructed[t - 1] = previous
    return reconstructed


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    fixed_scale = scale(args)
    diag = make_diag(args, rng, condition)
    branch_real = make_branch_real(args, rng, condition)
    branch_reference = floor_branch(branch_real, fixed_scale)
    branch_forward = (
        rounded_branch(branch_real, fixed_scale)
        if condition == "wrong_rounding_fail"
        else branch_reference
    )
    initial = rng.integers(-17, 18, size=args.state_dim, dtype=np.int64)

    reference_states = forward_integer_states(diag, branch_reference, initial)
    if condition == "fp32_accumulation_drift_fail":
        candidate_states = forward_fp32_states(diag, branch_forward, initial)
    else:
        candidate_states = forward_integer_states(diag, branch_forward, initial)

    reconstructed = reconstruct_integer_states(diag, branch_reference, candidate_states[-1])
    errors = reconstructed - reference_states
    bitwise_mismatch_count = int(np.count_nonzero(errors))
    reconstruction_linf = int(np.max(np.abs(errors)))
    fp32_mismatch_count = (
        int(np.count_nonzero(candidate_states - reference_states))
        if condition == "fp32_accumulation_drift_fail"
        else 0
    )
    branch_mismatch_count = int(np.count_nonzero(branch_forward - branch_reference))
    state_abs_max = int(np.max(np.abs(reference_states)))
    int32_overflow_margin = int(INT32_MAX - state_abs_max)
    quantization_error = float(np.max(np.abs((branch_reference.astype(np.float64) / fixed_scale) - branch_real)))
    transition_cells = int(args.sequence_len * args.state_dim)
    zero_fraction = float(np.mean(diag == 0))
    state_cache_vram_ratio = float(args.state_dim / transition_cells)
    conversion_overhead = (
        args.expensive_conversion_overhead_ratio
        if condition == "conversion_cost_fail"
        else args.conversion_overhead_ratio
    )
    priced_ratio = state_cache_vram_ratio + conversion_overhead
    projected_multiplier = 1.0 / priced_ratio

    failed: list[str] = []
    if not finite(
        zero_fraction,
        quantization_error,
        state_cache_vram_ratio,
        priced_ratio,
        projected_multiplier,
    ):
        failed.append("non_finite_metric")
    if transition_cells < args.min_transition_cells:
        failed.append("transition_count")
    if branch_mismatch_count > 0:
        failed.append("branch_rounding")
    if bitwise_mismatch_count > 0 or reconstruction_linf != 0:
        failed.append("bitwise_reconstruction")
    if fp32_mismatch_count > 0:
        failed.append("fp32_drift")
    if int32_overflow_margin <= 0:
        failed.append("int32_overflow")
    if quantization_error > args.max_bx_quantization_error:
        failed.append("quantization_error")
    if priced_ratio > args.max_priced_state_denominator_ratio:
        failed.append("state_denominator_price")

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
        transition_cells=transition_cells,
        scale_bits=args.scale_bits,
        zero_fraction=zero_fraction,
        branch_mismatch_count=branch_mismatch_count,
        bitwise_mismatch_count=bitwise_mismatch_count,
        reconstruction_linf_max=reconstruction_linf,
        fp32_state_mismatch_count=fp32_mismatch_count,
        state_abs_max=state_abs_max,
        int32_overflow_margin=int32_overflow_margin,
        bx_quantization_error_abs_max=quantization_error,
        state_cache_vram_ratio=state_cache_vram_ratio,
        priced_state_denominator_ratio=priced_ratio,
        projected_state_denominator_saving_multiplier=projected_multiplier,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "sequence_len",
        "state_dim",
        "transition_cells",
        "scale_bits",
        "zero_fraction",
        "branch_mismatch_count",
        "bitwise_mismatch_count",
        "reconstruction_linf_max",
        "fp32_state_mismatch_count",
        "state_abs_max",
        "int32_overflow_margin",
        "bx_quantization_error_abs_max",
        "state_cache_vram_ratio",
        "priced_state_denominator_ratio",
        "projected_state_denominator_saving_multiplier",
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
    fp32 = [result for result in results if result.condition == "fp32_accumulation_drift_fail"]
    rounding = [result for result in results if result.condition == "wrong_rounding_fail"]
    overflow = [result for result in results if result.condition == "overflow_risk_fail"]
    cost = [result for result in results if result.condition == "conversion_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_million_transition_gate": (
            aggregates["positive"]["transition_cells_min"] >= 1_000_000
        ),
        "positive_condition_bitwise_exact_reconstruction": (
            aggregates["positive"]["bitwise_mismatch_count_max"] == 0.0
            and aggregates["positive"]["reconstruction_linf_max_max"] == 0.0
        ),
        "positive_condition_int32_bounded": (
            aggregates["positive"]["int32_overflow_margin_min"] > 0.0
        ),
        "positive_condition_floor_projection_quality": (
            aggregates["positive"]["bx_quantization_error_abs_max_max"] <= (1.0 / 65536.0)
        ),
        "positive_condition_priced_denominator": (
            aggregates["positive"]["priced_state_denominator_ratio_mean"] <= 0.20
        ),
        "fp32_control_all_fail_drift": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "fp32_drift" in result.fail_reason
                or "bitwise_reconstruction" in result.fail_reason
            )
            for result in fp32
        ),
        "wrong_rounding_control_all_fail_branch": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "branch_rounding" in result.fail_reason
                or "bitwise_reconstruction" in result.fail_reason
            )
            for result in rounding
        ),
        "overflow_control_all_fail_int32_bound": all(
            (not result.passed)
            and result.fail_reason is not None
            and "int32_overflow" in result.fail_reason
            for result in overflow
        ),
        "cost_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "state_denominator_price" in result.fail_reason
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
    decision = "PASS_R502_E2_BDISA_GATE" if all(checks.values()) else "FAIL_R502_E2_BDISA_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU bounded-drift integer state accumulation gate only; "
            "not heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_E2_BDISA_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_transition_cells_min={positive['transition_cells_min']:.0f} "
        f"positive_bitwise_mismatch_count_max={positive['bitwise_mismatch_count_max']:.0f} "
        f"positive_reconstruction_linf_max_max={positive['reconstruction_linf_max_max']:.0f} "
        f"positive_int32_overflow_margin_min={positive['int32_overflow_margin_min']:.0f} "
        f"positive_bx_quantization_error_abs_max_max={positive['bx_quantization_error_abs_max_max']:.12e} "
        f"positive_state_cache_vram_ratio_mean={positive['state_cache_vram_ratio_mean']:.12e} "
        f"positive_priced_state_denominator_ratio_mean={positive['priced_state_denominator_ratio_mean']:.6f} "
        f"positive_projected_state_denominator_saving_multiplier_mean={positive['projected_state_denominator_saving_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
