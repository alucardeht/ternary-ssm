#!/usr/bin/env python3
"""R502 E8 dynamic parameter precision escrow gate.

This CPU diagnostic implements the R502 Efficiency finding:
"Dynamic Parameter Precision Escrow (DPPE)".

The gate stores a synthetic parameter matrix as INT2 base weights plus FP8
momentum and activates FP32 repair only when a gradient proxy exceeds a trust
threshold. A positive case must keep the heldout residual fraction under 0.10,
keep active repair sparse, and preserve the optimizer-memory denominator.

Controls exercise the declared risks: unpaid quantization residuals break the
heldout residual budget, high-gradient coordinates need repair, low thresholds
cause escrow thrashing, and falling back to high-precision momentum erases the
memory denominator.

This is a synthetic CPU precision-escrow and memory-pricing diagnostic, not
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


SCHEMA = "r502_e8_dppe_gate.v1"
CONDITIONS = (
    "positive",
    "unpaid_residual_fail",
    "missing_repair_fail",
    "escrow_thrashing_fail",
    "momentum_precision_cost_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    param_count: int
    heldout_rows: int
    residual_fraction_heldout: float
    active_repair_fraction: float
    trigger_flip_fraction: float
    int2_weight_bits: float
    momentum_bits: float
    repair_bits_per_param: float
    optimizer_bits_per_param: float
    optimizer_memory_ratio: float
    projected_optimizer_memory_saving_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50208)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--heldout-rows", type=int, default=256)
    parser.add_argument("--trust-threshold", type=float, default=0.030)
    parser.add_argument("--max-residual-fraction-heldout", type=float, default=0.10)
    parser.add_argument("--max-active-repair-fraction", type=float, default=0.20)
    parser.add_argument("--max-trigger-flip-fraction", type=float, default=0.15)
    parser.add_argument("--max-optimizer-memory-ratio", type=float, default=0.22)
    parser.add_argument("--baseline-optimizer-bits-per-param", type=float, default=96.0)
    parser.add_argument("--int2-weight-bits", type=float, default=2.0)
    parser.add_argument("--fp8-momentum-bits", type=float, default=8.0)
    parser.add_argument("--fp32-momentum-bits", type=float, default=32.0)
    parser.add_argument("--fp32-repair-bits", type=float, default=32.0)
    return parser.parse_args()


def make_problem(args: argparse.Namespace, rng: np.random.Generator, condition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (args.input_dim, args.output_dim)
    base = rng.choice(np.array([-1.0, 0.0, 1.0]), size=shape) * 0.35
    active_mask = rng.random(shape) < 0.08
    repair = rng.normal(scale=0.006, size=shape)
    repair[active_mask] += rng.normal(scale=0.120, size=int(np.sum(active_mask)))
    gradient_proxy = np.abs(repair) + rng.normal(scale=0.002, size=shape)
    gradient_proxy = np.maximum(0.0, gradient_proxy)

    if condition == "unpaid_residual_fail":
        inactive = ~active_mask
        repair[inactive] += rng.normal(scale=0.085, size=int(np.sum(inactive)))
        gradient_proxy[inactive] = args.trust_threshold * 0.35
    elif condition == "escrow_thrashing_fail":
        repair += rng.normal(scale=0.035, size=shape)
        gradient_proxy = args.trust_threshold + rng.normal(scale=0.018, size=shape)
        gradient_proxy = np.maximum(0.0, gradient_proxy)

    true_weights = base + repair
    heldout = rng.normal(scale=0.50, size=(args.heldout_rows, args.input_dim))
    return base, repair, gradient_proxy, heldout, true_weights


def trigger_flip_fraction(
    args: argparse.Namespace,
    rng: np.random.Generator,
    gradient_proxy: np.ndarray,
    condition: str,
) -> float:
    previous = gradient_proxy > args.trust_threshold
    flips = 0
    total = 0
    for step in range(1, 12):
        if condition == "escrow_thrashing_fail":
            jitter = rng.normal(scale=0.026, size=gradient_proxy.shape)
        else:
            jitter = rng.normal(scale=0.004, size=gradient_proxy.shape)
        current = np.maximum(0.0, gradient_proxy + jitter) > args.trust_threshold
        flips += int(np.count_nonzero(current != previous))
        total += int(current.size)
        previous = current
    return float(flips / max(total, 1))


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    base, repair, gradient_proxy, heldout, true_weights = make_problem(args, rng, condition)
    active = gradient_proxy > args.trust_threshold
    applied_repair = repair * active
    if condition == "missing_repair_fail":
        applied_repair = np.zeros_like(repair)
    active_fraction = float(np.mean(active))
    candidate = base + applied_repair
    true_out = heldout @ true_weights
    candidate_out = heldout @ candidate
    residual_fraction = float(
        np.linalg.norm(candidate_out - true_out)
        / max(float(np.linalg.norm(true_out)), 1.0e-30)
    )
    flips = trigger_flip_fraction(args, rng, gradient_proxy, condition)
    momentum_bits = (
        args.fp32_momentum_bits
        if condition == "momentum_precision_cost_fail"
        else args.fp8_momentum_bits
    )
    repair_bits = active_fraction * args.fp32_repair_bits
    bits_per_param = args.int2_weight_bits + momentum_bits + repair_bits
    memory_ratio = bits_per_param / args.baseline_optimizer_bits_per_param
    projected_multiplier = 1.0 / memory_ratio

    failed: list[str] = []
    if not finite(residual_fraction, active_fraction, flips, bits_per_param, memory_ratio, projected_multiplier):
        failed.append("non_finite_metric")
    if residual_fraction > args.max_residual_fraction_heldout:
        failed.append("heldout_residual")
    if active_fraction > args.max_active_repair_fraction:
        failed.append("repair_density")
    if flips > args.max_trigger_flip_fraction:
        failed.append("escrow_thrashing")
    if memory_ratio > args.max_optimizer_memory_ratio:
        failed.append("optimizer_memory_price")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed) if failed else None,
        param_count=int(base.size),
        heldout_rows=args.heldout_rows,
        residual_fraction_heldout=residual_fraction,
        active_repair_fraction=active_fraction,
        trigger_flip_fraction=flips,
        int2_weight_bits=args.int2_weight_bits,
        momentum_bits=momentum_bits,
        repair_bits_per_param=repair_bits,
        optimizer_bits_per_param=bits_per_param,
        optimizer_memory_ratio=memory_ratio,
        projected_optimizer_memory_saving_multiplier=projected_multiplier,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "param_count",
        "heldout_rows",
        "residual_fraction_heldout",
        "active_repair_fraction",
        "trigger_flip_fraction",
        "int2_weight_bits",
        "momentum_bits",
        "repair_bits_per_param",
        "optimizer_bits_per_param",
        "optimizer_memory_ratio",
        "projected_optimizer_memory_saving_multiplier",
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
    unpaid = [result for result in results if result.condition == "unpaid_residual_fail"]
    missing = [result for result in results if result.condition == "missing_repair_fail"]
    thrash = [result for result in results if result.condition == "escrow_thrashing_fail"]
    momentum = [result for result in results if result.condition == "momentum_precision_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_residual_priced": (
            aggregates["positive"]["residual_fraction_heldout_max"] <= 0.10
        ),
        "positive_condition_sparse_and_stable_repair": (
            aggregates["positive"]["active_repair_fraction_mean"] <= 0.20
            and aggregates["positive"]["trigger_flip_fraction_mean"] <= 0.15
        ),
        "positive_condition_optimizer_memory_priced": (
            aggregates["positive"]["optimizer_memory_ratio_mean"] <= 0.22
        ),
        "unpaid_residual_control_all_fail": all(
            (not result.passed)
            and result.fail_reason is not None
            and "heldout_residual" in result.fail_reason
            for result in unpaid
        ),
        "missing_repair_control_all_fail": all(
            (not result.passed)
            and result.fail_reason is not None
            and "heldout_residual" in result.fail_reason
            for result in missing
        ),
        "thrashing_control_all_fail": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "escrow_thrashing" in result.fail_reason
                or "repair_density" in result.fail_reason
                or "optimizer_memory_price" in result.fail_reason
            )
            for result in thrash
        ),
        "momentum_cost_control_all_fail": all(
            (not result.passed)
            and result.fail_reason is not None
            and "optimizer_memory_price" in result.fail_reason
            for result in momentum
        ),
        "all_conditions_identified": all(result.identifying_correct for result in results),
    }


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.input_dim < 1 or args.output_dim < 1:
        raise ValueError("--input-dim and --output-dim must be positive")
    if args.heldout_rows < 1:
        raise ValueError("--heldout-rows must be positive")

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
    decision = "PASS_R502_E8_DPPE_GATE" if all(checks.values()) else "FAIL_R502_E8_DPPE_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU dynamic parameter precision escrow gate only; not "
            "heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_E8_DPPE_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_residual_fraction_heldout_max={positive['residual_fraction_heldout_max']:.12e} "
        f"positive_active_repair_fraction_mean={positive['active_repair_fraction_mean']:.6f} "
        f"positive_trigger_flip_fraction_mean={positive['trigger_flip_fraction_mean']:.6f} "
        f"positive_optimizer_bits_per_param_mean={positive['optimizer_bits_per_param_mean']:.6f} "
        f"positive_optimizer_memory_ratio_mean={positive['optimizer_memory_ratio_mean']:.6f} "
        f"positive_projected_optimizer_memory_saving_multiplier_mean={positive['projected_optimizer_memory_saving_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
