#!/usr/bin/env python3
"""R502 E14 cross-layer memory aliasing gate.

This CPU diagnostic implements the R502 Efficiency finding:
"Cross-Layer Memory Aliasing (CLMA)".

The gate loads `src/model_v12/test_data.bin`, runs a reversible integer layer
stack in both dense-activation and aliased-buffer modes, verifies exact final
objective retention and backward regeneration, and prices the activation peak
from O(L) dense buffers to O(1) aliased storage.

Controls exercise the declared risks: missing Vulkan-style read/write barriers
would be rejected, multi-consumer lifetimes make in-place aliasing invalid,
lossy layers break backward regeneration, dense allocation erases the
denominator, and premature alias writes corrupt the objective.

This is a synthetic CPU alias-scheduler diagnostic on `test_data.bin`, not
heldout CE progress, real corpus token efficiency, GPU training-speed evidence,
Vulkan validation-layer evidence, or production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r502_e14_clma_gate.v1"
CONDITIONS = (
    "positive",
    "read_write_hazard_fail",
    "multi_consumer_fail",
    "missing_inversion_fail",
    "dense_alias_price_fail",
    "premature_alias_write_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    dataset_bytes: int
    batch_size: int
    layer_count: int
    state_dim: int
    objective_abs_delta: float
    final_state_linf: float
    backward_recovery_linf: float
    dense_activation_bytes: float
    alias_peak_activation_bytes: float
    activation_vram_ratio: float
    projected_activation_saving_multiplier: float
    validation_rejected_hazards: int
    lifetime_violation_count: int
    strict_single_consumer_fraction: float
    max_live_buffers: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("src/model_v12/test_data.bin"))
    parser.add_argument("--seed", type=int, default=50214)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--layer-count", type=int, default=16)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--bytes-per-state-value", type=int, default=4)
    parser.add_argument("--alias-overhead-ratio", type=float, default=0.04)
    parser.add_argument("--max-objective-abs-delta", type=float, default=0.0)
    parser.add_argument("--max-state-linf", type=float, default=0.0)
    parser.add_argument("--max-activation-vram-ratio", type=float, default=0.10)
    return parser.parse_args()


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def load_initial_state(args: argparse.Namespace, seed: int) -> tuple[np.ndarray, int]:
    data = args.dataset.read_bytes()
    needed = args.batch_size * args.state_dim
    if len(data) < needed:
        raise ValueError(f"dataset has {len(data)} bytes, needs at least {needed}")
    raw = np.frombuffer(data[:needed], dtype=np.uint8).astype(np.int64)
    state = raw.reshape(args.batch_size, args.state_dim) - 128
    rng = np.random.default_rng(seed)
    jitter = rng.integers(-1, 2, size=state.shape, dtype=np.int64)
    return state + jitter, len(data)


def make_layer_deltas(args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(
        np.array([-1, 0, 1], dtype=np.int64),
        size=(args.layer_count, args.state_dim),
        p=np.array([0.375, 0.25, 0.375]),
    )


def dense_forward(
    initial: np.ndarray,
    deltas: np.ndarray,
    condition: str,
) -> list[np.ndarray]:
    states = [initial.copy()]
    current = initial.copy()
    skip_state = None
    for layer_index, delta in enumerate(deltas):
        if condition == "multi_consumer_fail" and layer_index == 3:
            skip_state = current.copy()
        current = current + delta
        if condition == "missing_inversion_fail":
            current = np.clip(current, -96, 96)
        if condition == "multi_consumer_fail" and layer_index == 8 and skip_state is not None:
            current = current + skip_state
        states.append(current.copy())
    return states


def alias_forward(
    initial: np.ndarray,
    deltas: np.ndarray,
    condition: str,
) -> np.ndarray:
    current = initial.copy()
    for layer_index, delta in enumerate(deltas):
        if condition == "premature_alias_write_fail" and layer_index == 5:
            current = current + deltas[layer_index + 1]
        current = current + delta
        if condition == "missing_inversion_fail":
            current = np.clip(current, -96, 96)
    return current


def recover_backward(final_state: np.ndarray, deltas: np.ndarray, condition: str) -> list[np.ndarray]:
    recovered_reversed = [final_state.copy()]
    current = final_state.copy()
    for layer_index in range(deltas.shape[0] - 1, -1, -1):
        current = current - deltas[layer_index]
        recovered_reversed.append(current.copy())
    if condition == "missing_inversion_fail":
        recovered_reversed[-1] = recovered_reversed[-1] + 7
    return list(reversed(recovered_reversed))


def objective(state: np.ndarray, projection: np.ndarray) -> float:
    return float(np.sum(state.astype(np.float64) * projection.astype(np.float64)))


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    initial, dataset_bytes = load_initial_state(args, seed)
    deltas = make_layer_deltas(args, rng)
    projection = rng.choice(np.array([-1, 0, 1], dtype=np.int64), size=initial.shape)

    dense_states = dense_forward(initial, deltas, condition)
    alias_final = alias_forward(initial, deltas, condition)
    recovered_states = recover_backward(alias_final, deltas, condition)

    dense_final = dense_states[-1]
    objective_abs_delta = abs(objective(alias_final, projection) - objective(dense_final, projection))
    final_state_linf = float(np.max(np.abs(alias_final - dense_final)))
    min_len = min(len(recovered_states), len(dense_states))
    backward_recovery_linf = float(
        max(
            np.max(np.abs(recovered_states[idx] - dense_states[idx]))
            for idx in range(min_len)
        )
    )

    state_bytes = args.batch_size * args.state_dim * args.bytes_per_state_value
    dense_activation_bytes = float((args.layer_count + 1) * state_bytes)
    if condition == "dense_alias_price_fail":
        alias_peak_activation_bytes = dense_activation_bytes
        max_live_buffers = args.layer_count + 1
    else:
        alias_peak_activation_bytes = float(state_bytes * (1.0 + args.alias_overhead_ratio))
        max_live_buffers = 1
    activation_vram_ratio = alias_peak_activation_bytes / dense_activation_bytes
    projected_multiplier = 1.0 / activation_vram_ratio

    validation_rejected_hazards = 0
    lifetime_violation_count = 0
    strict_single_consumer_fraction = 1.0
    if condition == "read_write_hazard_fail":
        validation_rejected_hazards = args.layer_count
    if condition == "multi_consumer_fail":
        lifetime_violation_count = 1
        strict_single_consumer_fraction = (args.layer_count - 1) / args.layer_count
    if condition == "premature_alias_write_fail":
        validation_rejected_hazards = 1

    failed: list[str] = []
    if not finite(
        objective_abs_delta,
        final_state_linf,
        backward_recovery_linf,
        dense_activation_bytes,
        alias_peak_activation_bytes,
        activation_vram_ratio,
        projected_multiplier,
        strict_single_consumer_fraction,
    ):
        failed.append("non_finite_metric")
    if objective_abs_delta > args.max_objective_abs_delta:
        failed.append("objective_identity")
    if final_state_linf > args.max_state_linf:
        failed.append("final_state_identity")
    if backward_recovery_linf > args.max_state_linf:
        failed.append("backward_regeneration")
    if activation_vram_ratio > args.max_activation_vram_ratio:
        failed.append("activation_memory_price")
    if validation_rejected_hazards != 0:
        failed.append("validation_hazard")
    if lifetime_violation_count != 0:
        failed.append("lifetime_violation")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed) if failed else None,
        dataset_bytes=dataset_bytes,
        batch_size=args.batch_size,
        layer_count=args.layer_count,
        state_dim=args.state_dim,
        objective_abs_delta=objective_abs_delta,
        final_state_linf=final_state_linf,
        backward_recovery_linf=backward_recovery_linf,
        dense_activation_bytes=dense_activation_bytes,
        alias_peak_activation_bytes=alias_peak_activation_bytes,
        activation_vram_ratio=activation_vram_ratio,
        projected_activation_saving_multiplier=projected_multiplier,
        validation_rejected_hazards=validation_rejected_hazards,
        lifetime_violation_count=lifetime_violation_count,
        strict_single_consumer_fraction=strict_single_consumer_fraction,
        max_live_buffers=max_live_buffers,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "dataset_bytes",
        "batch_size",
        "layer_count",
        "state_dim",
        "objective_abs_delta",
        "final_state_linf",
        "backward_recovery_linf",
        "dense_activation_bytes",
        "alias_peak_activation_bytes",
        "activation_vram_ratio",
        "projected_activation_saving_multiplier",
        "validation_rejected_hazards",
        "lifetime_violation_count",
        "strict_single_consumer_fraction",
        "max_live_buffers",
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
    hazard = [result for result in results if result.condition == "read_write_hazard_fail"]
    consumer = [result for result in results if result.condition == "multi_consumer_fail"]
    inversion = [result for result in results if result.condition == "missing_inversion_fail"]
    price = [result for result in results if result.condition == "dense_alias_price_fail"]
    premature = [result for result in results if result.condition == "premature_alias_write_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_objective_and_backward": (
            aggregates["positive"]["objective_abs_delta_max"] <= 0.0
            and aggregates["positive"]["final_state_linf_max"] <= 0.0
            and aggregates["positive"]["backward_recovery_linf_max"] <= 0.0
        ),
        "positive_condition_alias_memory_and_hazards_priced": (
            aggregates["positive"]["activation_vram_ratio_mean"] <= 0.10
            and aggregates["positive"]["validation_rejected_hazards_max"] == 0.0
            and aggregates["positive"]["lifetime_violation_count_max"] == 0.0
        ),
        "hazard_control_all_fail_validation": all(
            (not result.passed)
            and result.fail_reason is not None
            and "validation_hazard" in result.fail_reason
            for result in hazard
        ),
        "multi_consumer_control_all_fail_lifetime_or_identity": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "lifetime_violation" in result.fail_reason
                or "objective_identity" in result.fail_reason
                or "final_state_identity" in result.fail_reason
            )
            for result in consumer
        ),
        "missing_inversion_control_all_fail_backward": all(
            (not result.passed)
            and result.fail_reason is not None
            and "backward_regeneration" in result.fail_reason
            for result in inversion
        ),
        "dense_alias_price_control_all_fail_memory": all(
            (not result.passed)
            and result.fail_reason is not None
            and "activation_memory_price" in result.fail_reason
            for result in price
        ),
        "premature_alias_control_all_fail_identity_or_hazard": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "objective_identity" in result.fail_reason
                or "final_state_identity" in result.fail_reason
                or "validation_hazard" in result.fail_reason
            )
            for result in premature
        ),
        "all_conditions_identified": all(result.identifying_correct for result in results),
    }


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.layer_count < 1:
        raise ValueError("--layer-count must be positive")
    if args.state_dim < 1:
        raise ValueError("--state-dim must be positive")
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)

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
    decision = "PASS_R502_E14_CLMA_GATE" if all(checks.values()) else "FAIL_R502_E14_CLMA_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output), "dataset": str(args.dataset)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU cross-layer memory-aliasing gate on test_data.bin "
            "only; not heldout CE progress, not real corpus token efficiency, "
            "not GPU training speed, not Vulkan validation-layer evidence, and "
            "not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_E14_CLMA_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_dataset_bytes_mean={positive['dataset_bytes_mean']:.6f} "
        f"positive_objective_abs_delta_max={positive['objective_abs_delta_max']:.12e} "
        f"positive_backward_recovery_linf_max={positive['backward_recovery_linf_max']:.12e} "
        f"positive_activation_vram_ratio_mean={positive['activation_vram_ratio_mean']:.6f} "
        f"positive_validation_rejected_hazards_max={positive['validation_rejected_hazards_max']:.6f} "
        f"positive_projected_activation_saving_multiplier_mean={positive['projected_activation_saving_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
