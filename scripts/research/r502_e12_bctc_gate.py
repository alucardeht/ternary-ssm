#!/usr/bin/env python3
"""R502 E12 block-circulant ternary compression gate.

This CPU diagnostic implements the R502 Efficiency finding:
"Block-Circulant Ternary Compression (BCTC)".

The gate builds block-diagonal circulant ternary matrices and applies them both
as dense blocks and as exact integer cyclic convolutions. The positive path
checks bit-exact output identity, a target-progress proxy, ternary structure,
and the parameter denominator drop from N^2 to N per block.

Controls exercise the declared risks: a non-circulant teacher can destroy the
quality proxy, low-precision transform arithmetic can break exact integer
agreement, non-ternary kernels violate the substrate, dense storage erases the
parameter denominator, and over-compressed zero kernels destroy capacity.

This is a synthetic CPU block-circulant integer-convolution diagnostic, not
heldout CE progress, real corpus token efficiency, GPU training-speed evidence,
FNTT/Vulkan kernel evidence, or production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r502_e12_bctc_gate.v1"
CONDITIONS = (
    "positive",
    "non_circulant_teacher_fail",
    "float_transform_rounding_fail",
    "non_ternary_kernel_fail",
    "parameter_price_fail",
    "capacity_destroyed_fail",
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
    block_count: int
    block_size: int
    dense_parameter_count: int
    bctc_parameter_count: int
    parameter_ratio: float
    projected_parameter_saving_multiplier: float
    ternary_violation_count: int
    output_linf: float
    relative_output_l2: float
    target_loss_before: float
    target_loss_after: float
    target_progress_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50212)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-count", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--max-output-linf", type=float, default=0.0)
    parser.add_argument("--max-relative-output-l2", type=float, default=0.02)
    parser.add_argument("--min-target-progress-ratio", type=float, default=0.98)
    parser.add_argument("--max-parameter-ratio", type=float, default=0.05)
    return parser.parse_args()


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def make_ternary_kernels(
    rng: np.random.Generator,
    block_count: int,
    block_size: int,
) -> np.ndarray:
    return rng.choice(
        np.array([-1, 0, 1], dtype=np.int64),
        size=(block_count, block_size),
        p=np.array([0.375, 0.25, 0.375]),
    )


def circulant_dense(kernel: np.ndarray) -> np.ndarray:
    size = kernel.shape[0]
    dense = np.empty((size, size), dtype=np.int64)
    for row in range(size):
        for col in range(size):
            dense[row, col] = kernel[(row - col) % size]
    return dense


def cyclic_convolution(block: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    batch, size = block.shape
    output = np.zeros((batch, size), dtype=np.int64)
    for row in range(size):
        for col in range(size):
            output[:, row] += block[:, col] * kernel[(row - col) % size]
    return output


def apply_bctc(inputs: np.ndarray, kernels: np.ndarray, condition: str) -> np.ndarray:
    outputs: list[np.ndarray] = []
    block_size = kernels.shape[1]
    for block_index, kernel in enumerate(kernels):
        start = block_index * block_size
        end = start + block_size
        outputs.append(cyclic_convolution(inputs[:, start:end], kernel))
    output = np.concatenate(outputs, axis=1)
    if condition == "float_transform_rounding_fail":
        output = np.rint(output.astype(np.float64) * 0.97).astype(np.int64)
    return output


def apply_dense_circulant(inputs: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    outputs: list[np.ndarray] = []
    block_size = kernels.shape[1]
    for block_index, kernel in enumerate(kernels):
        start = block_index * block_size
        end = start + block_size
        dense = circulant_dense(kernel)
        outputs.append(inputs[:, start:end] @ dense.T)
    return np.concatenate(outputs, axis=1)


def apply_dense_blocks(inputs: np.ndarray, dense_blocks: list[np.ndarray]) -> np.ndarray:
    outputs: list[np.ndarray] = []
    block_size = dense_blocks[0].shape[0]
    for block_index, dense in enumerate(dense_blocks):
        start = block_index * block_size
        end = start + block_size
        outputs.append(inputs[:, start:end] @ dense.T)
    return np.concatenate(outputs, axis=1)


def mean_square(a: np.ndarray, b: np.ndarray) -> float:
    delta = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(delta * delta))


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    total_dim = args.block_count * args.block_size
    inputs = rng.integers(-5, 6, size=(args.batch_size, total_dim), dtype=np.int64)
    positive_kernels = make_ternary_kernels(rng, args.block_count, args.block_size)

    if condition == "non_ternary_kernel_fail":
        kernels = positive_kernels.copy()
        kernels[0, : max(1, args.block_size // 4)] = 2
        target = apply_bctc(inputs, kernels, "positive")
    elif condition == "capacity_destroyed_fail":
        kernels = np.zeros_like(positive_kernels)
        target = apply_dense_circulant(inputs, positive_kernels)
    else:
        kernels = positive_kernels
        if condition == "non_circulant_teacher_fail":
            dense_blocks = [
                rng.choice(
                    np.array([-1, 0, 1], dtype=np.int64),
                    size=(args.block_size, args.block_size),
                )
                for _ in range(args.block_count)
            ]
            target = apply_dense_blocks(inputs, dense_blocks)
        else:
            target = apply_dense_circulant(inputs, positive_kernels)

    candidate = apply_bctc(inputs, kernels, condition)
    output_delta = candidate.astype(np.int64) - target.astype(np.int64)
    output_linf = float(np.max(np.abs(output_delta)))
    relative_output_l2 = float(
        np.linalg.norm(output_delta.astype(np.float64))
        / max(np.linalg.norm(target.astype(np.float64)), 1.0e-30)
    )
    target_loss_before = mean_square(inputs, target)
    target_loss_after = mean_square(candidate, target)
    target_progress_ratio = 1.0 - (
        target_loss_after / max(target_loss_before, 1.0e-30)
    )

    dense_parameter_count = args.block_count * args.block_size * args.block_size
    if condition == "parameter_price_fail":
        bctc_parameter_count = dense_parameter_count
    else:
        bctc_parameter_count = args.block_count * args.block_size
    parameter_ratio = bctc_parameter_count / dense_parameter_count
    projected_multiplier = 1.0 / parameter_ratio
    ternary_violation_count = int(np.sum(~np.isin(kernels, np.array([-1, 0, 1]))))

    failed: list[str] = []
    if not finite(
        parameter_ratio,
        projected_multiplier,
        output_linf,
        relative_output_l2,
        target_loss_before,
        target_loss_after,
        target_progress_ratio,
    ):
        failed.append("non_finite_metric")
    if ternary_violation_count != 0:
        failed.append("ternary_structure")
    if output_linf > args.max_output_linf:
        failed.append("exact_integer_output")
    if relative_output_l2 > args.max_relative_output_l2:
        failed.append("quality_residual")
    if target_progress_ratio < args.min_target_progress_ratio:
        failed.append("target_progress")
    if parameter_ratio > args.max_parameter_ratio:
        failed.append("parameter_price")

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
        block_count=args.block_count,
        block_size=args.block_size,
        dense_parameter_count=dense_parameter_count,
        bctc_parameter_count=bctc_parameter_count,
        parameter_ratio=parameter_ratio,
        projected_parameter_saving_multiplier=projected_multiplier,
        ternary_violation_count=ternary_violation_count,
        output_linf=output_linf,
        relative_output_l2=relative_output_l2,
        target_loss_before=target_loss_before,
        target_loss_after=target_loss_after,
        target_progress_ratio=target_progress_ratio,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "batch_size",
        "block_count",
        "block_size",
        "dense_parameter_count",
        "bctc_parameter_count",
        "parameter_ratio",
        "projected_parameter_saving_multiplier",
        "ternary_violation_count",
        "output_linf",
        "relative_output_l2",
        "target_loss_before",
        "target_loss_after",
        "target_progress_ratio",
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
    non_circulant = [result for result in results if result.condition == "non_circulant_teacher_fail"]
    float_rounding = [result for result in results if result.condition == "float_transform_rounding_fail"]
    non_ternary = [result for result in results if result.condition == "non_ternary_kernel_fail"]
    price = [result for result in results if result.condition == "parameter_price_fail"]
    capacity = [result for result in results if result.condition == "capacity_destroyed_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_integer_output": (
            aggregates["positive"]["output_linf_max"] <= 0.0
            and aggregates["positive"]["relative_output_l2_max"] <= 0.0
        ),
        "positive_condition_quality_and_parameter_price": (
            aggregates["positive"]["target_progress_ratio_min"] >= 0.98
            and aggregates["positive"]["parameter_ratio_mean"] <= 0.05
            and aggregates["positive"]["ternary_violation_count_max"] == 0.0
        ),
        "non_circulant_control_all_fail_quality": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "quality_residual" in result.fail_reason
                or "target_progress" in result.fail_reason
            )
            for result in non_circulant
        ),
        "float_rounding_control_all_fail_exactness": all(
            (not result.passed)
            and result.fail_reason is not None
            and "exact_integer_output" in result.fail_reason
            for result in float_rounding
        ),
        "non_ternary_control_all_fail_structure": all(
            (not result.passed)
            and result.fail_reason is not None
            and "ternary_structure" in result.fail_reason
            for result in non_ternary
        ),
        "parameter_price_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "parameter_price" in result.fail_reason
            for result in price
        ),
        "capacity_control_all_fail_progress": all(
            (not result.passed)
            and result.fail_reason is not None
            and "target_progress" in result.fail_reason
            for result in capacity
        ),
        "all_conditions_identified": all(result.identifying_correct for result in results),
    }


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.block_count < 1:
        raise ValueError("--block-count must be positive")
    if args.block_size < 2:
        raise ValueError("--block-size must be at least 2")

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
    decision = "PASS_R502_E12_BCTC_GATE" if all(checks.values()) else "FAIL_R502_E12_BCTC_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU block-circulant integer-convolution gate only; not "
            "heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, not FNTT/Vulkan kernel evidence, and not production "
            "v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_E12_BCTC_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_block_size_mean={positive['block_size_mean']:.6f} "
        f"positive_output_linf_max={positive['output_linf_max']:.12e} "
        f"positive_relative_output_l2_max={positive['relative_output_l2_max']:.12e} "
        f"positive_target_progress_ratio_min={positive['target_progress_ratio_min']:.12e} "
        f"positive_parameter_ratio_mean={positive['parameter_ratio_mean']:.6f} "
        f"positive_projected_parameter_saving_multiplier_mean={positive['projected_parameter_saving_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
