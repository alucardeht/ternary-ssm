#!/usr/bin/env python3
"""R502 E6 state-event flip tracing gate.

This CPU diagnostic implements the R502 Efficiency finding:
"State-Event Flip Tracing (SEFT)".

The gate encodes a bimodal recurrent state by initial sign, per-channel
amplitude, and temporal sign-flip indices. A positive case must reconstruct the
state within a bounded residual while using sparse flip storage that is cheaper
than dense state storage.

Controls exercise the declared risks: frequent flips approach dense storage,
amplitude drift breaks sign*mu reconstruction, missing initial signs corrupt the
trajectory, and expensive timestamp metadata can erase the compression
denominator even when flips are sparse.

This is a synthetic CPU sign-flip compression diagnostic, not heldout CE
progress, real corpus token efficiency, GPU training-speed evidence, or
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


SCHEMA = "r502_e6_seft_gate.v1"
CONDITIONS = (
    "positive",
    "high_flip_frequency_fail",
    "amplitude_drift_fail",
    "missing_initial_sign_fail",
    "timestamp_cost_fail",
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
    flip_count: int
    flip_fraction: float
    sign_error_fraction: float
    reconstruction_relative_error_mean: float
    reconstruction_relative_error_max: float
    dense_state_bytes: int
    seft_bytes: int
    memory_ratio: float
    projected_memory_saving_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50206)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sequence-len", type=int, default=4096)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--flip-probability", type=float, default=0.025)
    parser.add_argument("--high-flip-probability", type=float, default=0.49)
    parser.add_argument("--noise-scale", type=float, default=0.018)
    parser.add_argument("--drift-scale", type=float, default=0.45)
    parser.add_argument("--bytes-per-dense-state", type=int, default=4)
    parser.add_argument("--bytes-per-channel-header", type=int, default=8)
    parser.add_argument("--bytes-per-flip", type=int, default=8)
    parser.add_argument("--expensive-bytes-per-flip", type=int, default=128)
    parser.add_argument("--max-reconstruction-relative-error", type=float, default=0.08)
    parser.add_argument("--max-sign-error-fraction", type=float, default=0.0)
    parser.add_argument("--max-memory-ratio", type=float, default=0.25)
    return parser.parse_args()


def make_signs(args: argparse.Namespace, rng: np.random.Generator, condition: str) -> np.ndarray:
    flip_probability = (
        args.high_flip_probability
        if condition == "high_flip_frequency_fail"
        else args.flip_probability
    )
    signs = np.empty((args.sequence_len, args.state_dim), dtype=np.int8)
    signs[0] = rng.choice(np.array([-1, 1], dtype=np.int8), size=args.state_dim)
    for t in range(1, args.sequence_len):
        flips = rng.random(args.state_dim) < flip_probability
        signs[t] = signs[t - 1]
        signs[t, flips] = -signs[t, flips]
    return signs


def make_states(
    args: argparse.Namespace,
    rng: np.random.Generator,
    signs: np.ndarray,
    condition: str,
) -> np.ndarray:
    mu = rng.uniform(0.85, 1.25, size=args.state_dim)
    amplitude = np.broadcast_to(mu, signs.shape).astype(np.float64).copy()
    if condition == "amplitude_drift_fail":
        drift = np.linspace(0.0, args.drift_scale, args.sequence_len, dtype=np.float64)[:, None]
        direction = rng.choice(np.array([-1.0, 1.0]), size=(1, args.state_dim))
        amplitude = np.maximum(0.05, amplitude + drift * direction)
    noise = rng.normal(scale=args.noise_scale, size=signs.shape)
    return signs.astype(np.float64) * amplitude + noise


def encode_flips(signs: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], int]:
    initial = signs[0].copy()
    flip_lists: list[np.ndarray] = []
    flip_count = 0
    for dim in range(signs.shape[1]):
        flips = np.flatnonzero(signs[1:, dim] != signs[:-1, dim]) + 1
        flip_lists.append(flips.astype(np.int64))
        flip_count += int(flips.shape[0])
    return initial, flip_lists, flip_count


def reconstruct_signs(
    sequence_len: int,
    initial: np.ndarray,
    flip_lists: list[np.ndarray],
    condition: str,
) -> np.ndarray:
    state_dim = initial.shape[0]
    signs = np.empty((sequence_len, state_dim), dtype=np.int8)
    if condition == "missing_initial_sign_fail":
        signs[0] = 1
    else:
        signs[0] = initial
    for dim, flips in enumerate(flip_lists):
        sign = int(signs[0, dim])
        flip_set = set(int(value) for value in flips)
        for t in range(1, sequence_len):
            if t in flip_set:
                sign = -sign
            signs[t, dim] = sign
    return signs


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    signs = make_signs(args, rng, condition)
    states = make_states(args, rng, signs, condition)
    initial, flip_lists, flip_count = encode_flips(signs)
    reconstructed_signs = reconstruct_signs(args.sequence_len, initial, flip_lists, condition)
    mu_est = np.mean(np.abs(states), axis=0)
    reconstructed = reconstructed_signs.astype(np.float64) * mu_est
    errors = reconstructed - states
    per_dim_scale = np.maximum(np.linalg.norm(states, axis=0), 1.0e-30)
    rel_by_dim = np.linalg.norm(errors, axis=0) / per_dim_scale
    rel_mean = float(np.mean(rel_by_dim))
    rel_max = float(np.max(rel_by_dim))
    sign_error_fraction = float(np.mean(reconstructed_signs != signs))
    possible_flips = (args.sequence_len - 1) * args.state_dim
    flip_fraction = float(flip_count / possible_flips)
    dense_bytes = int(args.sequence_len * args.state_dim * args.bytes_per_dense_state)
    bytes_per_flip = (
        args.expensive_bytes_per_flip
        if condition == "timestamp_cost_fail"
        else args.bytes_per_flip
    )
    seft_bytes = int(args.state_dim * args.bytes_per_channel_header + flip_count * bytes_per_flip)
    memory_ratio = float(seft_bytes / dense_bytes)
    projected_multiplier = 1.0 / memory_ratio

    failed: list[str] = []
    if not finite(flip_fraction, sign_error_fraction, rel_mean, rel_max, memory_ratio, projected_multiplier):
        failed.append("non_finite_metric")
    if sign_error_fraction > args.max_sign_error_fraction:
        failed.append("sign_reconstruction")
    if rel_max > args.max_reconstruction_relative_error:
        failed.append("amplitude_residual")
    if memory_ratio > args.max_memory_ratio:
        failed.append("memory_price")

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
        flip_count=flip_count,
        flip_fraction=flip_fraction,
        sign_error_fraction=sign_error_fraction,
        reconstruction_relative_error_mean=rel_mean,
        reconstruction_relative_error_max=rel_max,
        dense_state_bytes=dense_bytes,
        seft_bytes=seft_bytes,
        memory_ratio=memory_ratio,
        projected_memory_saving_multiplier=projected_multiplier,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "sequence_len",
        "state_dim",
        "flip_count",
        "flip_fraction",
        "sign_error_fraction",
        "reconstruction_relative_error_mean",
        "reconstruction_relative_error_max",
        "dense_state_bytes",
        "seft_bytes",
        "memory_ratio",
        "projected_memory_saving_multiplier",
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
    high_flip = [result for result in results if result.condition == "high_flip_frequency_fail"]
    drift = [result for result in results if result.condition == "amplitude_drift_fail"]
    initial = [result for result in results if result.condition == "missing_initial_sign_fail"]
    timestamp = [result for result in results if result.condition == "timestamp_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_reconstruction_bounded": (
            aggregates["positive"]["sign_error_fraction_max"] == 0.0
            and aggregates["positive"]["reconstruction_relative_error_max_max"] <= 0.08
        ),
        "positive_condition_memory_priced": (
            aggregates["positive"]["memory_ratio_mean"] <= 0.25
        ),
        "high_flip_control_all_fail_memory": all(
            (not result.passed)
            and result.fail_reason is not None
            and "memory_price" in result.fail_reason
            for result in high_flip
        ),
        "amplitude_drift_control_all_fail_residual": all(
            (not result.passed)
            and result.fail_reason is not None
            and "amplitude_residual" in result.fail_reason
            for result in drift
        ),
        "missing_initial_control_all_fail_sign": all(
            (not result.passed)
            and result.fail_reason is not None
            and "sign_reconstruction" in result.fail_reason
            for result in initial
        ),
        "timestamp_cost_control_all_fail_memory": all(
            (not result.passed)
            and result.fail_reason is not None
            and "memory_price" in result.fail_reason
            for result in timestamp
        ),
        "all_conditions_identified": all(result.identifying_correct for result in results),
    }


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.sequence_len < 2:
        raise ValueError("--sequence-len must be at least 2")
    if args.state_dim < 1:
        raise ValueError("--state-dim must be at least 1")

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
    decision = "PASS_R502_E6_SEFT_GATE" if all(checks.values()) else "FAIL_R502_E6_SEFT_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU state-event flip tracing gate only; not heldout CE "
            "progress, not real corpus token efficiency, not GPU training speed, "
            "and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_E6_SEFT_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_flip_fraction_mean={positive['flip_fraction_mean']:.6f} "
        f"positive_sign_error_fraction_max={positive['sign_error_fraction_max']:.12e} "
        f"positive_reconstruction_relative_error_max_max={positive['reconstruction_relative_error_max_max']:.12e} "
        f"positive_dense_state_bytes_mean={positive['dense_state_bytes_mean']:.0f} "
        f"positive_seft_bytes_mean={positive['seft_bytes_mean']:.0f} "
        f"positive_memory_ratio_mean={positive['memory_ratio_mean']:.6f} "
        f"positive_projected_memory_saving_multiplier_mean={positive['projected_memory_saving_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
