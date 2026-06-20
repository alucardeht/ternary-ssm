#!/usr/bin/env python3
"""R502 E7 recurrent block-diagonal inversion LUT gate.

This CPU diagnostic implements the R502 Efficiency finding:
"Recurrent Block-Diagonal Inversion LUT (RBDI)".

The gate samples invertible 4x4 ternary recurrent blocks, stores their dense
inverse in a keyed lookup table, and verifies that LUT retrieval produces the
same inverse action as dense inversion on synthetic recurrent right-hand sides.

Controls exercise the declared risks: singular blocks do not have an inverse,
wrong LUT keys break structural equivalence, shared-memory bank conflicts can
erase the denominator, and an undersized cache budget cannot hold the projected
full LUT.

This is a synthetic CPU block-inversion and lookup-pricing diagnostic, not
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


SCHEMA = "r502_e7_rbdi_gate.v1"
CONDITIONS = (
    "positive",
    "singular_block_fail",
    "wrong_lut_key_fail",
    "bank_conflict_fail",
    "cache_capacity_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    block_size: int
    sampled_blocks: int
    invertible_fraction: float
    lut_entry_count: int
    output_l2_error_max: float
    inverse_residual_l2_max: float
    dense_inverse_units: float
    lut_read_units: float
    priced_cost_ratio: float
    projected_inverse_speedup: float
    estimated_full_lut_mb: float
    cache_budget_mb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50207)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--sampled-blocks", type=int, default=256)
    parser.add_argument("--rhs-count", type=int, default=8)
    parser.add_argument("--max-output-l2-error", type=float, default=1.0e-10)
    parser.add_argument("--max-inverse-residual-l2", type=float, default=1.0e-10)
    parser.add_argument("--max-priced-cost-ratio", type=float, default=0.35)
    parser.add_argument("--estimated-full-lut-mb", type=float, default=43.0)
    parser.add_argument("--cache-budget-mb", type=float, default=48.0)
    parser.add_argument("--small-cache-budget-mb", type=float, default=16.0)
    parser.add_argument("--lut-read-base-units", type=float, default=4.0)
    parser.add_argument("--bank-conflict-units", type=float, default=84.0)
    return parser.parse_args()


def block_key(block: np.ndarray) -> int:
    digits = (block.astype(np.int64).reshape(-1) + 1).tolist()
    key = 0
    scale = 1
    for digit in digits:
        key += int(digit) * scale
        scale *= 3
    return key


def random_ternary_block(rng: np.random.Generator, block_size: int) -> np.ndarray:
    return rng.choice(np.array([-1.0, 0.0, 1.0]), size=(block_size, block_size))


def singular_block(block_size: int) -> np.ndarray:
    block = np.eye(block_size, dtype=np.float64)
    block[1] = block[0]
    return block


def sample_blocks(args: argparse.Namespace, rng: np.random.Generator, condition: str) -> list[np.ndarray]:
    blocks: list[np.ndarray] = []
    if condition == "singular_block_fail":
        blocks.append(singular_block(args.block_size))
    attempts = 0
    while len(blocks) < args.sampled_blocks:
        attempts += 1
        if attempts > args.sampled_blocks * 200:
            raise RuntimeError("could not sample enough invertible ternary blocks")
        block = random_ternary_block(rng, args.block_size)
        if abs(float(np.linalg.det(block))) <= 1.0e-9:
            continue
        blocks.append(block)
    return blocks


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    blocks = sample_blocks(args, rng, condition)
    lut: dict[int, np.ndarray] = {}
    invertible_count = 0
    singular_seen = 0
    for block in blocks:
        key = block_key(block)
        try:
            inverse = np.linalg.inv(block)
        except np.linalg.LinAlgError:
            singular_seen += 1
            continue
        invertible_count += 1
        lut[key] = inverse

    output_errors: list[float] = []
    residuals: list[float] = []
    invertible_blocks = [block for block in blocks if block_key(block) in lut]
    keys = [block_key(block) for block in invertible_blocks]
    for block_index, block in enumerate(invertible_blocks):
        key = keys[block_index]
        inverse = lut[key]
        if condition == "wrong_lut_key_fail" and len(keys) > 1:
            inverse = lut[keys[(block_index + 1) % len(keys)]]
        rhs = rng.normal(scale=0.40, size=(args.rhs_count, args.block_size))
        dense_inverse = np.linalg.inv(block)
        dense_out = rhs @ dense_inverse.T
        lut_out = rhs @ inverse.T
        output_errors.append(float(np.max(np.linalg.norm(lut_out - dense_out, axis=1))))
        residuals.append(float(np.max(np.linalg.norm((block @ lut_out.T).T - rhs, axis=1))))

    output_error_max = float(max(output_errors) if output_errors else math.inf)
    residual_max = float(max(residuals) if residuals else math.inf)
    dense_units = float(args.sampled_blocks * (args.block_size ** 3))
    lut_units_per_block = (
        args.bank_conflict_units
        if condition == "bank_conflict_fail"
        else args.lut_read_base_units
    )
    lut_units = float(args.sampled_blocks * lut_units_per_block)
    cost_ratio = lut_units / dense_units
    projected_speedup = 1.0 / cost_ratio
    cache_budget = (
        args.small_cache_budget_mb
        if condition == "cache_capacity_fail"
        else args.cache_budget_mb
    )
    invertible_fraction = float(invertible_count / max(len(blocks), 1))

    failed: list[str] = []
    if not finite(
        invertible_fraction,
        output_error_max,
        residual_max,
        dense_units,
        lut_units,
        cost_ratio,
        projected_speedup,
        args.estimated_full_lut_mb,
        cache_budget,
    ):
        failed.append("non_finite_metric")
    if singular_seen > 0:
        failed.append("singular_block")
    if output_error_max > args.max_output_l2_error:
        failed.append("output_equivalence")
    if residual_max > args.max_inverse_residual_l2:
        failed.append("inverse_residual")
    if cost_ratio > args.max_priced_cost_ratio:
        failed.append("bank_conflict_price")
    if args.estimated_full_lut_mb > cache_budget:
        failed.append("cache_capacity")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed) if failed else None,
        block_size=args.block_size,
        sampled_blocks=args.sampled_blocks,
        invertible_fraction=invertible_fraction,
        lut_entry_count=len(lut),
        output_l2_error_max=output_error_max,
        inverse_residual_l2_max=residual_max,
        dense_inverse_units=dense_units,
        lut_read_units=lut_units,
        priced_cost_ratio=cost_ratio,
        projected_inverse_speedup=projected_speedup,
        estimated_full_lut_mb=args.estimated_full_lut_mb,
        cache_budget_mb=cache_budget,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "block_size",
        "sampled_blocks",
        "invertible_fraction",
        "lut_entry_count",
        "output_l2_error_max",
        "inverse_residual_l2_max",
        "dense_inverse_units",
        "lut_read_units",
        "priced_cost_ratio",
        "projected_inverse_speedup",
        "estimated_full_lut_mb",
        "cache_budget_mb",
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
    singular = [result for result in results if result.condition == "singular_block_fail"]
    wrong_key = [result for result in results if result.condition == "wrong_lut_key_fail"]
    bank = [result for result in results if result.condition == "bank_conflict_fail"]
    capacity = [result for result in results if result.condition == "cache_capacity_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_output": (
            aggregates["positive"]["output_l2_error_max_max"] <= 1.0e-10
            and aggregates["positive"]["inverse_residual_l2_max_max"] <= 1.0e-10
        ),
        "positive_condition_priced_lut_read": (
            aggregates["positive"]["priced_cost_ratio_mean"] <= 0.35
        ),
        "positive_condition_cache_capacity": (
            aggregates["positive"]["estimated_full_lut_mb_mean"]
            <= aggregates["positive"]["cache_budget_mb_mean"]
        ),
        "singular_control_all_fail": all(
            (not result.passed)
            and result.fail_reason is not None
            and "singular_block" in result.fail_reason
            for result in singular
        ),
        "wrong_key_control_all_fail_equivalence": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "output_equivalence" in result.fail_reason
                or "inverse_residual" in result.fail_reason
            )
            for result in wrong_key
        ),
        "bank_conflict_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "bank_conflict_price" in result.fail_reason
            for result in bank
        ),
        "cache_capacity_control_all_fail": all(
            (not result.passed)
            and result.fail_reason is not None
            and "cache_capacity" in result.fail_reason
            for result in capacity
        ),
        "all_conditions_identified": all(result.identifying_correct for result in results),
    }


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.block_size != 4:
        raise ValueError("this RBDI gate prices the R502 4x4 ternary block claim")
    if args.sampled_blocks < 2:
        raise ValueError("--sampled-blocks must be at least 2")

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
    decision = "PASS_R502_E7_RBDI_GATE" if all(checks.values()) else "FAIL_R502_E7_RBDI_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU recurrent block-diagonal inversion LUT gate only; "
            "not heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_E7_RBDI_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_sampled_blocks_mean={positive['sampled_blocks_mean']:.0f} "
        f"positive_output_l2_error_max_max={positive['output_l2_error_max_max']:.12e} "
        f"positive_inverse_residual_l2_max_max={positive['inverse_residual_l2_max_max']:.12e} "
        f"positive_priced_cost_ratio_mean={positive['priced_cost_ratio_mean']:.6f} "
        f"positive_projected_inverse_speedup_mean={positive['projected_inverse_speedup_mean']:.6f} "
        f"positive_estimated_full_lut_mb_mean={positive['estimated_full_lut_mb_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
