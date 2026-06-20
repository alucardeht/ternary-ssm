#!/usr/bin/env python3
"""R501 S94 Sherman-Morrison spectral-partitioned recurrent inversion gate.

This CPU diagnostic implements the first R501 Hardware Efficiency mathematical
finding: "Sherman-Morrison Spectral-Partitioned Recurrent Inversion (SM-SPRI)".

The gate constructs a linear recurrent SSM trajectory with a diagonal transition
operator containing a well-conditioned invertible subspace and a singular
subspace. The positive condition reconstructs the full backward trajectory by
inverting only the stable coordinates and escrowing only the singular
coordinates, preserving the dense recurrent-gradient calculation while charging
only boundary plus singular-subspace residency.

Controls exercise the declared kill criteria: no singular escrow cannot
reconstruct singular coordinates, a wrong partition misses part of the singular
subspace, quantized escrow breaks exactness, dense residency erases the memory
denominator, and high repair overhead erases the denominator.

This is a synthetic CPU spectral-partitioned recurrent inversion diagnostic,
not heldout CE progress, real VRAM telemetry, GPU training-speed evidence, or
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


SCHEMA = "r501_s94_sm_spri_gate.v1"
CONDITIONS = (
    "positive",
    "no_escrow_fail",
    "wrong_partition_fail",
    "quantized_escrow_fail",
    "dense_residency_fail",
    "repair_cost_fail",
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
    stable_dim: int
    singular_dim: int
    singular_escrow_recall: float
    reconstruction_relative_error: float
    reconstruction_max_abs_error: float
    recurrent_gradient_relative_error: float
    stable_inverse_condition_proxy: float
    stored_state_ratio: float
    repair_overhead_ratio: float
    paid_vram_ratio: float
    projected_vram_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50194)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sequence-len", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--singular-dim", type=int, default=6)
    parser.add_argument("--stable-min", type=float, default=0.99)
    parser.add_argument("--stable-max", type=float, default=1.01)
    parser.add_argument("--repair-overhead", type=float, default=0.04)
    parser.add_argument("--expensive-repair-overhead", type=float, default=0.45)
    parser.add_argument("--max-reconstruction-error", type=float, default=1.0e-10)
    parser.add_argument("--max-gradient-error", type=float, default=1.0e-10)
    parser.add_argument("--min-escrow-recall", type=float, default=1.0)
    parser.add_argument("--max-paid-vram-ratio", type=float, default=0.35)
    parser.add_argument("--min-vram-efficiency-multiplier", type=float, default=2.5)
    return parser.parse_args()


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def make_system(
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if args.singular_dim >= args.state_dim:
        raise ValueError("singular_dim must be smaller than state_dim")
    singular_indices = rng.choice(args.state_dim, size=args.singular_dim, replace=False)
    singular_mask = np.zeros(args.state_dim, dtype=bool)
    singular_mask[singular_indices] = True
    transition = rng.uniform(args.stable_min, args.stable_max, size=args.state_dim)
    transition[singular_mask] = 0.0
    input_matrix = rng.normal(size=(args.state_dim, args.input_dim)) / math.sqrt(args.input_dim)
    inputs = rng.normal(size=(args.sequence_len, args.input_dim))
    return transition, input_matrix, inputs, singular_mask


def forward_trajectory(
    rng: np.random.Generator,
    transition: np.ndarray,
    input_matrix: np.ndarray,
    inputs: np.ndarray,
) -> np.ndarray:
    state_dim = transition.shape[0]
    h = np.zeros((inputs.shape[0] + 1, state_dim), dtype=np.float64)
    h[0] = rng.normal(size=state_dim)
    for t in range(1, inputs.shape[0] + 1):
        h[t] = transition * h[t - 1] + input_matrix @ inputs[t - 1]
    return h


def reconstruct_backward(
    h_final: np.ndarray,
    h_true: np.ndarray,
    transition: np.ndarray,
    input_matrix: np.ndarray,
    inputs: np.ndarray,
    singular_mask: np.ndarray,
    condition: str,
) -> tuple[np.ndarray, float, float]:
    sequence_len = inputs.shape[0]
    state_dim = transition.shape[0]
    reconstructed = np.zeros_like(h_true)
    reconstructed[-1] = h_final
    stable_mask = ~singular_mask
    escrow_mask = singular_mask.copy()
    stored_dim = int(np.count_nonzero(singular_mask))
    recall = 1.0

    if condition == "no_escrow_fail":
        escrow_mask[:] = False
        stored_dim = 0
        recall = 0.0
    elif condition == "wrong_partition_fail":
        singular_indices = np.flatnonzero(singular_mask)
        escrow_mask[:] = False
        escrow_mask[singular_indices[: max(1, len(singular_indices) // 2)]] = True
        stored_dim = int(np.count_nonzero(escrow_mask))
        recall = float(stored_dim / max(np.count_nonzero(singular_mask), 1))
    elif condition == "dense_residency_fail":
        escrow_mask[:] = True
        stored_dim = state_dim
        recall = 1.0

    for t in range(sequence_len, 0, -1):
        input_effect = input_matrix @ inputs[t - 1]
        previous = np.zeros(state_dim, dtype=np.float64)
        previous[stable_mask] = (
            (reconstructed[t, stable_mask] - input_effect[stable_mask])
            / transition[stable_mask]
        )
        previous[escrow_mask] = h_true[t - 1, escrow_mask]
        if condition == "quantized_escrow_fail":
            scale = np.maximum(np.mean(np.abs(h_true[:, escrow_mask]), axis=0), 1.0e-12)
            previous[escrow_mask] = np.sign(previous[escrow_mask]) * scale
        reconstructed[t - 1] = previous

    stored_ratio = float((sequence_len * stored_dim + state_dim) / (sequence_len * state_dim))
    return reconstructed, recall, stored_ratio


def recurrent_gradient(h: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
    return np.sum(cotangent * h[:-1], axis=0)


def first_failure(
    *,
    singular_escrow_recall: float,
    reconstruction_relative_error: float,
    recurrent_gradient_relative_error: float,
    paid_vram_ratio: float,
    projected_vram_efficiency_multiplier: float,
    args: argparse.Namespace,
) -> str | None:
    if not finite(
        singular_escrow_recall,
        reconstruction_relative_error,
        recurrent_gradient_relative_error,
        paid_vram_ratio,
        projected_vram_efficiency_multiplier,
    ):
        return "nonfinite_metric"
    if paid_vram_ratio > args.max_paid_vram_ratio:
        return "price_paid_vram"
    if projected_vram_efficiency_multiplier < args.min_vram_efficiency_multiplier:
        return "price_efficiency_multiplier"
    if singular_escrow_recall < args.min_escrow_recall:
        return "singular_escrow_recall"
    if reconstruction_relative_error > args.max_reconstruction_error:
        return "reconstruction_error"
    if recurrent_gradient_relative_error > args.max_gradient_error:
        return "recurrent_gradient_error"
    return None


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    transition, input_matrix, inputs, singular_mask = make_system(rng, args)
    h_true = forward_trajectory(rng, transition, input_matrix, inputs)
    reconstructed, recall, stored_ratio = reconstruct_backward(
        h_true[-1],
        h_true,
        transition,
        input_matrix,
        inputs,
        singular_mask,
        condition,
    )
    cotangent = rng.normal(size=(args.sequence_len, args.state_dim))
    true_grad = recurrent_gradient(h_true, cotangent)
    reconstructed_grad = recurrent_gradient(reconstructed, cotangent)

    reconstruction_error = float(
        np.linalg.norm(reconstructed - h_true) / max(np.linalg.norm(h_true), 1.0e-30)
    )
    max_abs_error = float(np.max(np.abs(reconstructed - h_true)))
    gradient_error = float(
        np.linalg.norm(reconstructed_grad - true_grad) / max(np.linalg.norm(true_grad), 1.0e-30)
    )
    stable = transition[~singular_mask]
    condition_proxy = float(np.max(np.abs(1.0 / stable)))
    repair_overhead = (
        args.expensive_repair_overhead
        if condition == "repair_cost_fail"
        else args.repair_overhead
    )
    paid_ratio = float(stored_ratio + repair_overhead)
    multiplier = float(1.0 / max(paid_ratio, 1.0e-30))

    fail_reason = first_failure(
        singular_escrow_recall=recall,
        reconstruction_relative_error=reconstruction_error,
        recurrent_gradient_relative_error=gradient_error,
        paid_vram_ratio=paid_ratio,
        projected_vram_efficiency_multiplier=multiplier,
        args=args,
    )
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
        stable_dim=int(args.state_dim - args.singular_dim),
        singular_dim=args.singular_dim,
        singular_escrow_recall=recall,
        reconstruction_relative_error=reconstruction_error,
        reconstruction_max_abs_error=max_abs_error,
        recurrent_gradient_relative_error=gradient_error,
        stable_inverse_condition_proxy=condition_proxy,
        stored_state_ratio=stored_ratio,
        repair_overhead_ratio=repair_overhead,
        paid_vram_ratio=paid_ratio,
        projected_vram_efficiency_multiplier=multiplier,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "sequence_len",
        "state_dim",
        "input_dim",
        "stable_dim",
        "singular_dim",
        "singular_escrow_recall",
        "reconstruction_relative_error",
        "reconstruction_max_abs_error",
        "recurrent_gradient_relative_error",
        "stable_inverse_condition_proxy",
        "stored_state_ratio",
        "repair_overhead_ratio",
        "paid_vram_ratio",
        "projected_vram_efficiency_multiplier",
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
    no_escrow = [result for result in results if result.condition == "no_escrow_fail"]
    wrong = [result for result in results if result.condition == "wrong_partition_fail"]
    quantized = [result for result in results if result.condition == "quantized_escrow_fail"]
    dense = [result for result in results if result.condition == "dense_residency_fail"]
    repair = [result for result in results if result.condition == "repair_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_reconstruction": (
            aggregates["positive"]["singular_escrow_recall_min"] >= 1.0
            and aggregates["positive"]["reconstruction_relative_error_max"] <= 1.0e-10
            and aggregates["positive"]["recurrent_gradient_relative_error_max"] <= 1.0e-10
        ),
        "positive_condition_priced": (
            aggregates["positive"]["paid_vram_ratio_mean"] <= 0.35
            and aggregates["positive"]["projected_vram_efficiency_multiplier_mean"] >= 2.5
        ),
        "no_escrow_control_all_fail_reconstruction_or_recall": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "singular_escrow_recall" in result.fail_reason
                or "reconstruction_error" in result.fail_reason
                or "recurrent_gradient_error" in result.fail_reason
            )
            for result in no_escrow
        ),
        "wrong_partition_control_all_fail_reconstruction_or_recall": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "singular_escrow_recall" in result.fail_reason
                or "reconstruction_error" in result.fail_reason
                or "recurrent_gradient_error" in result.fail_reason
            )
            for result in wrong
        ),
        "quantized_escrow_control_all_fail_exactness": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "reconstruction_error" in result.fail_reason
                or "recurrent_gradient_error" in result.fail_reason
            )
            for result in quantized
        ),
        "dense_residency_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "price" in result.fail_reason
            for result in dense
        ),
        "repair_cost_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "price" in result.fail_reason
            for result in repair
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
    decision = "PASS_R501_S94_SM_SPRI_GATE" if all(checks.values()) else "FAIL_R501_S94_SM_SPRI_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU spectral-partitioned recurrent inversion gate only; "
            "not heldout CE progress, not real VRAM telemetry, not GPU training "
            "speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S94_SM_SPRI_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_singular_escrow_recall_min={positive['singular_escrow_recall_min']:.6f} "
        f"positive_reconstruction_relative_error_max={positive['reconstruction_relative_error_max']:.6e} "
        f"positive_recurrent_gradient_relative_error_max={positive['recurrent_gradient_relative_error_max']:.6e} "
        f"positive_stored_state_ratio_mean={positive['stored_state_ratio_mean']:.6f} "
        f"positive_paid_vram_ratio_mean={positive['paid_vram_ratio_mean']:.6f} "
        f"positive_projected_vram_efficiency_multiplier_mean={positive['projected_vram_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
