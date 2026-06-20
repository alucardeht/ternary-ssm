#!/usr/bin/env python3
"""R502 E11 reversible ternary scaffolding gate.

This CPU diagnostic implements the R502 Efficiency finding:
"Reversible Ternary Scaffolding (RTS)".

The gate builds an integer-valued reversible coupling stack with ternary
transforms. The positive path updates one half of the state while keeping the
other half unchanged, so each layer can be reconstructed exactly from the final
state. It prices the advertised memory denominator against dense activation
caching and checks a synthetic target-progress proxy for the structural-coupling
kill criterion.

Controls exercise the declared risks: non-reversible coupling breaks exact
state recovery, lossy precision destroys reversibility, recomputation can cost
more than the backward path allows, dense anchoring erases the memory factor,
and structurally impaired coupling can fail the target-progress proxy.

This is a synthetic CPU reversible-coupling and target-progress diagnostic, not
heldout CE progress, real corpus token efficiency, GPU training-speed evidence,
or production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r502_e11_rts_gate.v1"
CONDITIONS = (
    "positive",
    "non_reversible_coupling_fail",
    "precision_rounding_fail",
    "recompute_cost_fail",
    "memory_anchor_fail",
    "target_progress_impair_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    batch_size: int
    layer_count: int
    state_dim: int
    anchored_state_count: int
    recovery_linf: float
    final_target_linf: float
    target_loss_before: float
    target_loss_after: float
    target_progress_ratio: float
    dense_activation_bytes: float
    rts_activation_bytes: float
    memory_ratio: float
    projected_memory_saving_multiplier: float
    recompute_cost_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50211)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--layer-count", type=int, default=16)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--bytes-per-state-value", type=int, default=4)
    parser.add_argument("--activation-overhead-ratio", type=float, default=0.04)
    parser.add_argument("--dense-backward-unit-cost", type=float, default=2.0)
    parser.add_argument("--forward-unit-cost", type=float, default=1.0)
    parser.add_argument("--recompute-unit-cost", type=float, default=0.98)
    parser.add_argument("--expensive-recompute-unit-cost", type=float, default=1.35)
    parser.add_argument("--max-recovery-linf", type=float, default=0.0)
    parser.add_argument("--max-final-target-linf", type=float, default=0.0)
    parser.add_argument("--min-target-progress-ratio", type=float, default=0.90)
    parser.add_argument("--max-memory-ratio", type=float, default=0.08)
    parser.add_argument("--max-recompute-cost-ratio", type=float, default=1.10)
    return parser.parse_args()


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def make_layer_params(
    rng: np.random.Generator,
    layer_count: int,
    half_dim: int,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    params: list[tuple[np.ndarray, np.ndarray, int]] = []
    for layer in range(layer_count):
        signs = rng.choice(np.array([-1, 0, 1], dtype=np.int64), size=half_dim)
        roll_signs = rng.choice(np.array([-1, 1], dtype=np.int64), size=half_dim)
        shift = int((layer % 7) + 1)
        params.append((signs, roll_signs, shift))
    return params


def ternary_transform(part: np.ndarray, params: tuple[np.ndarray, np.ndarray, int]) -> np.ndarray:
    signs, roll_signs, shift = params
    return part * signs + np.roll(part, shift=shift, axis=1) * roll_signs


def forward_layer(
    state: np.ndarray,
    params: tuple[np.ndarray, np.ndarray, int],
    layer_index: int,
    condition: str,
) -> np.ndarray:
    half = state.shape[1] // 2
    a = state[:, :half].copy()
    b = state[:, half:].copy()

    if condition == "non_reversible_coupling_fail":
        a_new = a + ternary_transform(b, params)
        b_new = b + ternary_transform(a, params)
    elif layer_index % 2 == 0:
        a_new = a
        b_new = b + ternary_transform(a, params)
    else:
        a_new = a + ternary_transform(b, params)
        b_new = b

    next_state = np.concatenate([a_new, b_new], axis=1)
    if condition == "precision_rounding_fail":
        next_state = np.clip(next_state, -64, 64)
    return next_state


def inverse_layer(
    state: np.ndarray,
    params: tuple[np.ndarray, np.ndarray, int],
    layer_index: int,
) -> np.ndarray:
    half = state.shape[1] // 2
    a = state[:, :half].copy()
    b = state[:, half:].copy()
    if layer_index % 2 == 0:
        a_prev = a
        b_prev = b - ternary_transform(a, params)
    else:
        a_prev = a - ternary_transform(b, params)
        b_prev = b
    return np.concatenate([a_prev, b_prev], axis=1)


def forward_stack(
    initial: np.ndarray,
    params: list[tuple[np.ndarray, np.ndarray, int]],
    condition: str,
) -> list[np.ndarray]:
    states = [initial.copy()]
    current = initial.copy()
    for layer_index, layer_params in enumerate(params):
        current = forward_layer(current, layer_params, layer_index, condition)
        states.append(current.copy())
    return states


def recover_stack(
    final_state: np.ndarray,
    params: list[tuple[np.ndarray, np.ndarray, int]],
) -> list[np.ndarray]:
    recovered_reversed = [final_state.copy()]
    current = final_state.copy()
    for layer_index in range(len(params) - 1, -1, -1):
        current = inverse_layer(current, params[layer_index], layer_index)
        recovered_reversed.append(current.copy())
    return list(reversed(recovered_reversed))


def mean_square(a: np.ndarray, b: np.ndarray) -> float:
    delta = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(delta * delta))


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    half_dim = args.state_dim // 2
    params = make_layer_params(rng, args.layer_count, half_dim)
    initial = rng.integers(-3, 4, size=(args.batch_size, args.state_dim), dtype=np.int64)

    target_states = forward_stack(initial, params, "positive")
    if condition == "target_progress_impair_fail":
        run_params = [
            (np.zeros_like(signs), np.zeros_like(roll_signs), shift)
            for signs, roll_signs, shift in params
        ]
    else:
        run_params = params
    run_states = forward_stack(initial, run_params, condition)
    recovered_states = recover_stack(run_states[-1], run_params)

    recovery_linf = float(
        max(
            np.max(np.abs(recovered.astype(np.int64) - original.astype(np.int64)))
            for recovered, original in zip(recovered_states, run_states)
        )
    )
    final_target_linf = float(np.max(np.abs(run_states[-1] - target_states[-1])))
    target_loss_before = mean_square(initial, target_states[-1])
    target_loss_after = mean_square(run_states[-1], target_states[-1])
    target_progress_ratio = 1.0 - (
        target_loss_after / max(target_loss_before, 1.0e-30)
    )

    dense_activation_bytes = float(
        (args.layer_count + 1)
        * args.batch_size
        * args.state_dim
        * args.bytes_per_state_value
    )
    anchored_state_count = (
        args.layer_count + 1 if condition == "memory_anchor_fail" else 1
    )
    rts_activation_bytes = float(
        anchored_state_count
        * args.batch_size
        * args.state_dim
        * args.bytes_per_state_value
        * (1.0 + args.activation_overhead_ratio)
    )
    memory_ratio = rts_activation_bytes / dense_activation_bytes
    projected_multiplier = 1.0 / memory_ratio
    recompute_unit = (
        args.expensive_recompute_unit_cost
        if condition == "recompute_cost_fail"
        else args.recompute_unit_cost
    )
    recompute_cost_ratio = (
        args.forward_unit_cost + recompute_unit
    ) / args.dense_backward_unit_cost

    failed: list[str] = []
    if not finite(
        recovery_linf,
        final_target_linf,
        target_loss_before,
        target_loss_after,
        target_progress_ratio,
        dense_activation_bytes,
        rts_activation_bytes,
        memory_ratio,
        projected_multiplier,
        recompute_cost_ratio,
    ):
        failed.append("non_finite_metric")
    if recovery_linf > args.max_recovery_linf:
        failed.append("state_recovery")
    if final_target_linf > args.max_final_target_linf:
        failed.append("final_target_identity")
    if target_progress_ratio < args.min_target_progress_ratio:
        failed.append("target_progress")
    if memory_ratio > args.max_memory_ratio:
        failed.append("memory_price")
    if recompute_cost_ratio > args.max_recompute_cost_ratio:
        failed.append("recompute_cost")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed) if failed else None,
        batch_size=args.batch_size,
        layer_count=args.layer_count,
        state_dim=args.state_dim,
        anchored_state_count=anchored_state_count,
        recovery_linf=recovery_linf,
        final_target_linf=final_target_linf,
        target_loss_before=target_loss_before,
        target_loss_after=target_loss_after,
        target_progress_ratio=target_progress_ratio,
        dense_activation_bytes=dense_activation_bytes,
        rts_activation_bytes=rts_activation_bytes,
        memory_ratio=memory_ratio,
        projected_memory_saving_multiplier=projected_multiplier,
        recompute_cost_ratio=recompute_cost_ratio,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "batch_size",
        "layer_count",
        "state_dim",
        "anchored_state_count",
        "recovery_linf",
        "final_target_linf",
        "target_loss_before",
        "target_loss_after",
        "target_progress_ratio",
        "dense_activation_bytes",
        "rts_activation_bytes",
        "memory_ratio",
        "projected_memory_saving_multiplier",
        "recompute_cost_ratio",
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
    non_reversible = [
        result for result in results if result.condition == "non_reversible_coupling_fail"
    ]
    precision = [result for result in results if result.condition == "precision_rounding_fail"]
    recompute = [result for result in results if result.condition == "recompute_cost_fail"]
    memory = [result for result in results if result.condition == "memory_anchor_fail"]
    target = [result for result in results if result.condition == "target_progress_impair_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_recovery": (
            aggregates["positive"]["recovery_linf_max"] <= 0.0
            and aggregates["positive"]["final_target_linf_max"] <= 0.0
        ),
        "positive_condition_target_progress_preserved": (
            aggregates["positive"]["target_progress_ratio_min"] >= 0.90
        ),
        "positive_condition_memory_and_cost_priced": (
            aggregates["positive"]["memory_ratio_mean"] <= 0.08
            and aggregates["positive"]["recompute_cost_ratio_mean"] <= 1.10
        ),
        "non_reversible_control_all_fail_recovery": all(
            (not result.passed)
            and result.fail_reason is not None
            and "state_recovery" in result.fail_reason
            for result in non_reversible
        ),
        "precision_control_all_fail_recovery": all(
            (not result.passed)
            and result.fail_reason is not None
            and "state_recovery" in result.fail_reason
            for result in precision
        ),
        "recompute_control_all_fail_cost": all(
            (not result.passed)
            and result.fail_reason is not None
            and "recompute_cost" in result.fail_reason
            for result in recompute
        ),
        "memory_control_all_fail_memory": all(
            (not result.passed)
            and result.fail_reason is not None
            and "memory_price" in result.fail_reason
            for result in memory
        ),
        "target_progress_control_all_fail_progress": all(
            (not result.passed)
            and result.fail_reason is not None
            and "target_progress" in result.fail_reason
            for result in target
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
    if args.state_dim < 2 or args.state_dim % 2 != 0:
        raise ValueError("--state-dim must be an even integer >= 2")

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
    decision = "PASS_R502_E11_RTS_GATE" if all(checks.values()) else "FAIL_R502_E11_RTS_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU reversible ternary scaffolding gate only; not "
            "heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_E11_RTS_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_layer_count_mean={positive['layer_count_mean']:.6f} "
        f"positive_recovery_linf_max={positive['recovery_linf_max']:.12e} "
        f"positive_final_target_linf_max={positive['final_target_linf_max']:.12e} "
        f"positive_target_progress_ratio_min={positive['target_progress_ratio_min']:.12e} "
        f"positive_memory_ratio_mean={positive['memory_ratio_mean']:.6f} "
        f"positive_recompute_cost_ratio_mean={positive['recompute_cost_ratio_mean']:.6f} "
        f"positive_projected_memory_saving_multiplier_mean={positive['projected_memory_saving_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
