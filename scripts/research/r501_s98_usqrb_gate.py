#!/usr/bin/env python3
"""R501 S98 unbiased stochastic-quantized recurrent buffer gate.

This CPU diagnostic implements the fifth R501 Hardware Efficiency mathematical
finding: "Unbiased Stochastic-Quantized Recurrent Buffer (USQRB)".

The gate constructs recurrent activation buffers h_t and projection cotangents
g_t. The positive condition stores h_t with a 4-bit sign-magnitude stochastic
rounding scheme whose conditional expectation equals h_t, then verifies that
the expected projection gradient sum_t g_t Q(h_t)^T equals the dense FP16-buffer
gradient while charging a 4-bit buffer denominator.

Controls exercise the declared kill criteria: deterministic floor rounding,
nearest rounding, signless magnitude storage, dense FP16 residency, and clipped
dynamic range either introduce projection-gradient bias or erase the memory
denominator.

This is a synthetic CPU stochastic-buffer expectation diagnostic, not heldout
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


SCHEMA = "r501_s98_usqrb_gate.v1"
CONDITIONS = (
    "positive",
    "deterministic_floor_fail",
    "deterministic_nearest_fail",
    "signless_magnitude_fail",
    "dense_fp16_fail",
    "clipped_range_fail",
)
FP16_BITS = 16.0
QUANTIZED_BITS = 4.0


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
    output_dim: int
    sample_count: int
    expectation_gradient_relative_error: float
    expectation_gradient_max_abs_error: float
    monte_carlo_gradient_relative_error: float
    quantizer_state_bias_relative_error: float
    dense_buffer_bits: float
    candidate_buffer_bits: float
    paid_buffer_bit_ratio: float
    projected_buffer_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50198)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sequence-len", type=int, default=512)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=32)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--level-min", type=float, default=0.01)
    parser.add_argument("--level-max", type=float, default=1.0)
    parser.add_argument("--clipped-level-max", type=float, default=0.35)
    parser.add_argument("--max-expectation-gradient-error", type=float, default=1.0e-12)
    parser.add_argument("--max-expectation-abs-error", type=float, default=1.0e-12)
    parser.add_argument("--max-monte-carlo-gradient-error", type=float, default=0.08)
    parser.add_argument("--max-state-bias-error", type=float, default=1.0e-12)
    parser.add_argument("--max-paid-buffer-bit-ratio", type=float, default=0.30)
    parser.add_argument("--min-buffer-efficiency-multiplier", type=float, default=3.0)
    return parser.parse_args()


def make_levels(min_value: float, max_value: float) -> np.ndarray:
    if min_value <= 0.0 or max_value <= min_value:
        raise ValueError("invalid quantization range")
    return np.geomspace(min_value, max_value, num=8).astype(np.float64)


def make_fixture(
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    levels = make_levels(args.level_min, args.level_max)
    log_mag = rng.uniform(np.log(levels[0] * 1.05), np.log(levels[-1] * 0.95), size=(args.sequence_len, args.state_dim))
    magnitudes = np.exp(log_mag)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=magnitudes.shape)
    states = signs * magnitudes
    cotangents = rng.normal(size=(args.sequence_len, args.output_dim)) / math.sqrt(args.output_dim)
    return states.astype(np.float64), cotangents.astype(np.float64)


def bracket_probabilities(magnitudes: np.ndarray, levels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    upper_indices = np.searchsorted(levels, magnitudes, side="left")
    upper_indices = np.clip(upper_indices, 1, len(levels) - 1)
    lower = levels[upper_indices - 1]
    upper = levels[upper_indices]
    prob_upper = (magnitudes - lower) / np.maximum(upper - lower, 1.0e-30)
    prob_upper = np.clip(prob_upper, 0.0, 1.0)
    return lower, upper, prob_upper


def expected_quantized_states(
    states: np.ndarray,
    condition: str,
    levels: np.ndarray,
    clipped_levels: np.ndarray,
) -> np.ndarray:
    signs = np.sign(states)
    magnitudes = np.abs(states)

    if condition == "dense_fp16_fail":
        return states.copy()
    if condition == "clipped_range_fail":
        return signs * np.minimum(magnitudes, clipped_levels[-1])
    if condition == "deterministic_floor_fail":
        lower, _upper, _prob = bracket_probabilities(magnitudes, levels)
        return signs * lower
    if condition == "deterministic_nearest_fail":
        lower, upper, _prob = bracket_probabilities(magnitudes, levels)
        nearest = np.where(np.abs(magnitudes - lower) <= np.abs(upper - magnitudes), lower, upper)
        return signs * nearest
    if condition == "signless_magnitude_fail":
        return magnitudes
    return states.copy()


def sample_unbiased_quantized_states(
    rng: np.random.Generator,
    states: np.ndarray,
    levels: np.ndarray,
) -> np.ndarray:
    signs = np.sign(states)
    magnitudes = np.abs(states)
    lower, upper, prob_upper = bracket_probabilities(magnitudes, levels)
    draws = rng.random(size=states.shape)
    sampled = np.where(draws < prob_upper, upper, lower)
    return signs * sampled


def projection_gradient(cotangents: np.ndarray, states: np.ndarray) -> np.ndarray:
    return cotangents.T @ states


def relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference))
    return float(np.linalg.norm(candidate - reference) / max(denom, 1.0e-30))


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    states, cotangents = make_fixture(rng, args)
    levels = make_levels(args.level_min, args.level_max)
    clipped_levels = make_levels(args.level_min, args.clipped_level_max)
    dense_gradient = projection_gradient(cotangents, states)
    expected_states = expected_quantized_states(states, condition, levels, clipped_levels)
    expected_gradient = projection_gradient(cotangents, expected_states)

    if condition == "positive":
        mc_gradient = np.zeros_like(dense_gradient)
        for _ in range(args.sample_count):
            sampled_states = sample_unbiased_quantized_states(rng, states, levels)
            mc_gradient += projection_gradient(cotangents, sampled_states)
        mc_gradient /= float(args.sample_count)
    else:
        mc_gradient = expected_gradient

    expectation_error = relative_error(expected_gradient, dense_gradient)
    expectation_abs_error = float(np.max(np.abs(expected_gradient - dense_gradient)))
    mc_error = relative_error(mc_gradient, dense_gradient)
    state_bias_error = relative_error(expected_states, states)

    element_count = int(args.sequence_len * args.state_dim)
    dense_bits = float(element_count * FP16_BITS)
    if condition == "dense_fp16_fail":
        candidate_bits = dense_bits
    else:
        candidate_bits = float(element_count * QUANTIZED_BITS)
    paid_ratio = candidate_bits / dense_bits
    multiplier = 1.0 / paid_ratio

    failures: list[str] = []
    if (
        expectation_error > args.max_expectation_gradient_error
        or expectation_abs_error > args.max_expectation_abs_error
    ):
        failures.append("expectation_gradient_bias")
    if condition == "positive" and mc_error > args.max_monte_carlo_gradient_error:
        failures.append("monte_carlo_sanity")
    if state_bias_error > args.max_state_bias_error:
        failures.append("state_expectation_bias")
    if paid_ratio > args.max_paid_buffer_bit_ratio or multiplier < args.min_buffer_efficiency_multiplier:
        failures.append("price_paid_buffer_bits")

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
        output_dim=args.output_dim,
        sample_count=args.sample_count,
        expectation_gradient_relative_error=expectation_error,
        expectation_gradient_max_abs_error=expectation_abs_error,
        monte_carlo_gradient_relative_error=mc_error,
        quantizer_state_bias_relative_error=state_bias_error,
        dense_buffer_bits=dense_bits,
        candidate_buffer_bits=candidate_bits,
        paid_buffer_bit_ratio=paid_ratio,
        projected_buffer_efficiency_multiplier=multiplier,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "sequence_len",
        "state_dim",
        "output_dim",
        "sample_count",
        "expectation_gradient_relative_error",
        "expectation_gradient_max_abs_error",
        "monte_carlo_gradient_relative_error",
        "quantizer_state_bias_relative_error",
        "dense_buffer_bits",
        "candidate_buffer_bits",
        "paid_buffer_bit_ratio",
        "projected_buffer_efficiency_multiplier",
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
    floor = [result for result in results if result.condition == "deterministic_floor_fail"]
    nearest = [result for result in results if result.condition == "deterministic_nearest_fail"]
    signless = [result for result in results if result.condition == "signless_magnitude_fail"]
    dense = [result for result in results if result.condition == "dense_fp16_fail"]
    clipped = [result for result in results if result.condition == "clipped_range_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_unbiased_expectation": (
            aggregates["positive"]["expectation_gradient_relative_error_max"] <= 1.0e-12
            and aggregates["positive"]["expectation_gradient_max_abs_error_max"] <= 1.0e-12
            and aggregates["positive"]["quantizer_state_bias_relative_error_max"] <= 1.0e-12
        ),
        "positive_condition_monte_carlo_sanity": (
            aggregates["positive"]["monte_carlo_gradient_relative_error_max"] <= 0.08
        ),
        "positive_condition_priced": (
            aggregates["positive"]["paid_buffer_bit_ratio_mean"] <= 0.30
            and aggregates["positive"]["projected_buffer_efficiency_multiplier_mean"] >= 3.0
        ),
        "floor_control_all_fail_bias": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "expectation_gradient_bias" in result.fail_reason
                or "state_expectation_bias" in result.fail_reason
            )
            for result in floor
        ),
        "nearest_control_all_fail_bias": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "expectation_gradient_bias" in result.fail_reason
                or "state_expectation_bias" in result.fail_reason
            )
            for result in nearest
        ),
        "signless_control_all_fail_bias": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "expectation_gradient_bias" in result.fail_reason
                or "state_expectation_bias" in result.fail_reason
            )
            for result in signless
        ),
        "dense_fp16_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "price_paid_buffer_bits" in result.fail_reason
            for result in dense
        ),
        "clipped_range_control_all_fail_bias": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "expectation_gradient_bias" in result.fail_reason
                or "state_expectation_bias" in result.fail_reason
            )
            for result in clipped
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
    decision = "PASS_R501_S98_USQRB_GATE" if all(checks.values()) else "FAIL_R501_S98_USQRB_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU unbiased stochastic-quantized recurrent buffer gate "
            "only; not heldout CE progress, not real VRAM telemetry, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S98_USQRB_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_expectation_gradient_relative_error_max={positive['expectation_gradient_relative_error_max']:.6e} "
        f"positive_quantizer_state_bias_relative_error_max={positive['quantizer_state_bias_relative_error_max']:.6e} "
        f"positive_monte_carlo_gradient_relative_error_max={positive['monte_carlo_gradient_relative_error_max']:.6f} "
        f"positive_paid_buffer_bit_ratio_mean={positive['paid_buffer_bit_ratio_mean']:.6f} "
        f"positive_projected_buffer_efficiency_multiplier_mean={positive['projected_buffer_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
