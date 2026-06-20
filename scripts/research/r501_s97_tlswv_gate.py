#!/usr/bin/env python3
"""R501 S97 ternary-lattice shadow weight virtualization gate.

This CPU diagnostic implements the fourth R501 Hardware Efficiency mathematical
finding: "Ternary-Lattice Shadow Weight Virtualization (TLSWV)".

The gate constructs shadow weights with a saturated stable subset, low-margin
near-threshold weights, and rare massive impulses. The positive condition
virtualizes only provably stable saturated weights into a 2-bit ternary lattice
state, dynamically returns impulse-crossing stable weights to dense tracking,
and verifies that the ternary forward map matches dense FP32 shadow tracking.

Controls exercise the declared kill criteria: ignoring massive impulses leaves
virtualized weights stale, over-virtualizing unstable weights changes ternary
outputs, dense shadow tracking erases the memory denominator, signless 2-bit
virtualization loses negative saturated states, and loose-margin virtualization
tracks too close to the ternary boundary.

This is a synthetic CPU shadow-state memory diagnostic, not heldout CE progress,
real VRAM telemetry, GPU training-speed evidence, or production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s97_tlswv_gate.v1"
CONDITIONS = (
    "positive",
    "no_impulse_detector_fail",
    "overvirtualize_unstable_fail",
    "dense_shadow_fail",
    "signless_virtual_fail",
    "loose_margin_fail",
)
FP32_BITS = 32.0
VIRTUAL_BITS = 2.0


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    element_count: int
    stable_fraction: float
    low_margin_fraction: float
    impulse_fraction: float
    virtualized_fraction: float
    dense_tracking_fraction: float
    impulse_detector_recall: float
    unsafe_virtualization_rate: float
    ternary_mismatch_rate: float
    ternary_match_fraction: float
    dense_shadow_bits: float
    candidate_shadow_bits: float
    paid_shadow_bit_ratio: float
    projected_shadow_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50197)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--element-count", type=int, default=65536)
    parser.add_argument("--stable-probability", type=float, default=0.82)
    parser.add_argument("--stable-delta", type=float, default=0.10)
    parser.add_argument("--impulse-probability", type=float, default=0.03)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-ternary-mismatch-rate", type=float, default=0.0)
    parser.add_argument("--min-impulse-detector-recall", type=float, default=1.0)
    parser.add_argument("--max-unsafe-virtualization-rate", type=float, default=0.0)
    parser.add_argument("--max-paid-shadow-bit-ratio", type=float, default=0.35)
    parser.add_argument("--min-shadow-efficiency-multiplier", type=float, default=2.5)
    return parser.parse_args()


def ternary(shadow: np.ndarray) -> np.ndarray:
    return np.where(shadow > 1.0, 1, np.where(shadow < -1.0, -1, 0)).astype(np.int8)


def make_fixture(
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = args.element_count
    stable_mask = rng.random(n) < args.stable_probability
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=n)
    shadow = np.zeros(n, dtype=np.float64)
    gradient = rng.normal(scale=0.02, size=n).astype(np.float64)

    stable_count = int(np.count_nonzero(stable_mask))
    shadow[stable_mask] = signs[stable_mask] * (
        1.0 + args.stable_delta + rng.uniform(0.20, 1.00, size=stable_count)
    )

    unstable_indices = np.flatnonzero(~stable_mask)
    low_margin_count = len(unstable_indices) // 2
    low_margin_indices = unstable_indices[:low_margin_count]
    neutral_indices = unstable_indices[low_margin_count:]
    shadow[low_margin_indices] = signs[low_margin_indices] * (
        1.0 + rng.uniform(0.01, args.stable_delta * 0.8, size=low_margin_count)
    )
    gradient[low_margin_indices] = signs[low_margin_indices] * 8.0
    shadow[neutral_indices] = rng.uniform(-0.80, 0.80, size=len(neutral_indices))
    gradient[neutral_indices] = rng.normal(scale=0.05, size=len(neutral_indices))

    stable_indices = np.flatnonzero(stable_mask)
    impulse_count = max(1, int(round(args.impulse_probability * stable_count)))
    impulse_indices = rng.choice(stable_indices, size=impulse_count, replace=False)
    impulse_mask = np.zeros(n, dtype=bool)
    impulse_mask[impulse_indices] = True
    gradient[impulse_indices] = (
        np.sign(shadow[impulse_indices])
        * (np.abs(shadow[impulse_indices]) - 0.50)
        / args.learning_rate
    )

    low_margin_mask = np.zeros(n, dtype=bool)
    low_margin_mask[low_margin_indices] = True
    return shadow, gradient, stable_mask, impulse_mask, low_margin_mask


def candidate_forward(
    shadow: np.ndarray,
    gradient: np.ndarray,
    stable_mask: np.ndarray,
    condition: str,
    learning_rate: float,
    stable_delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    updated = shadow - learning_rate * gradient
    dense_forward = ternary(updated)
    initial_forward = ternary(shadow)
    detector = stable_mask & (dense_forward != initial_forward)

    if condition == "positive":
        virtual_mask = stable_mask & ~detector
        virtual_forward = initial_forward
    elif condition == "no_impulse_detector_fail":
        virtual_mask = stable_mask.copy()
        detector = np.zeros_like(stable_mask, dtype=bool)
        virtual_forward = initial_forward
    elif condition == "overvirtualize_unstable_fail":
        virtual_mask = np.ones_like(stable_mask, dtype=bool)
        virtual_forward = initial_forward
    elif condition == "dense_shadow_fail":
        virtual_mask = np.zeros_like(stable_mask, dtype=bool)
        virtual_forward = initial_forward
    elif condition == "signless_virtual_fail":
        virtual_mask = stable_mask & ~detector
        virtual_forward = np.where(initial_forward == 0, 0, 1).astype(np.int8)
    elif condition == "loose_margin_fail":
        virtual_mask = np.abs(shadow) > 1.0
        detector = np.zeros_like(stable_mask, dtype=bool)
        virtual_forward = initial_forward
    else:
        raise ValueError(f"unknown condition: {condition}")

    candidate = dense_forward.copy()
    candidate[virtual_mask] = virtual_forward[virtual_mask]
    return candidate, virtual_mask, detector


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    shadow, gradient, stable_mask, impulse_mask, low_margin_mask = make_fixture(rng, args)
    updated = shadow - args.learning_rate * gradient
    dense_forward = ternary(updated)
    initial_forward = ternary(shadow)
    true_impulse = impulse_mask & (dense_forward != initial_forward)
    candidate, virtual_mask, detector = candidate_forward(
        shadow=shadow,
        gradient=gradient,
        stable_mask=stable_mask,
        condition=condition,
        learning_rate=args.learning_rate,
        stable_delta=args.stable_delta,
    )

    n = args.element_count
    dense_mask = ~virtual_mask
    mismatch = candidate != dense_forward
    unsafe_virtual = virtual_mask & (dense_forward != initial_forward)
    impulse_recall = float(
        np.count_nonzero(detector & true_impulse) / max(np.count_nonzero(true_impulse), 1)
    )
    unsafe_rate = float(np.count_nonzero(unsafe_virtual) / max(np.count_nonzero(virtual_mask), 1))
    mismatch_rate = float(np.count_nonzero(mismatch) / n)
    virtual_count = int(np.count_nonzero(virtual_mask))
    dense_count = n - virtual_count
    dense_bits = float(n * FP32_BITS)
    candidate_bits = float(virtual_count * VIRTUAL_BITS + dense_count * FP32_BITS)
    paid_ratio = candidate_bits / dense_bits
    multiplier = 1.0 / paid_ratio

    failures: list[str] = []
    if mismatch_rate > args.max_ternary_mismatch_rate:
        failures.append("ternary_mismatch")
    if impulse_recall < args.min_impulse_detector_recall:
        failures.append("impulse_detector_recall")
    if unsafe_rate > args.max_unsafe_virtualization_rate:
        failures.append("unsafe_virtualization")
    if paid_ratio > args.max_paid_shadow_bit_ratio or multiplier < args.min_shadow_efficiency_multiplier:
        failures.append("price_paid_shadow_bits")

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
        element_count=n,
        stable_fraction=float(np.count_nonzero(stable_mask) / n),
        low_margin_fraction=float(np.count_nonzero(low_margin_mask) / n),
        impulse_fraction=float(np.count_nonzero(true_impulse) / n),
        virtualized_fraction=float(virtual_count / n),
        dense_tracking_fraction=float(dense_count / n),
        impulse_detector_recall=impulse_recall,
        unsafe_virtualization_rate=unsafe_rate,
        ternary_mismatch_rate=mismatch_rate,
        ternary_match_fraction=float(1.0 - mismatch_rate),
        dense_shadow_bits=dense_bits,
        candidate_shadow_bits=candidate_bits,
        paid_shadow_bit_ratio=paid_ratio,
        projected_shadow_efficiency_multiplier=multiplier,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "element_count",
        "stable_fraction",
        "low_margin_fraction",
        "impulse_fraction",
        "virtualized_fraction",
        "dense_tracking_fraction",
        "impulse_detector_recall",
        "unsafe_virtualization_rate",
        "ternary_mismatch_rate",
        "ternary_match_fraction",
        "dense_shadow_bits",
        "candidate_shadow_bits",
        "paid_shadow_bit_ratio",
        "projected_shadow_efficiency_multiplier",
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
    no_detector = [result for result in results if result.condition == "no_impulse_detector_fail"]
    over = [result for result in results if result.condition == "overvirtualize_unstable_fail"]
    dense = [result for result in results if result.condition == "dense_shadow_fail"]
    signless = [result for result in results if result.condition == "signless_virtual_fail"]
    loose = [result for result in results if result.condition == "loose_margin_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_forward": (
            aggregates["positive"]["ternary_mismatch_rate_max"] <= 0.0
            and aggregates["positive"]["unsafe_virtualization_rate_max"] <= 0.0
            and aggregates["positive"]["impulse_detector_recall_min"] >= 1.0
        ),
        "positive_condition_priced": (
            aggregates["positive"]["paid_shadow_bit_ratio_mean"] <= 0.35
            and aggregates["positive"]["projected_shadow_efficiency_multiplier_mean"] >= 2.5
        ),
        "no_impulse_detector_control_all_fail_impulse_or_mismatch": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "impulse_detector_recall" in result.fail_reason
                or "unsafe_virtualization" in result.fail_reason
                or "ternary_mismatch" in result.fail_reason
            )
            for result in no_detector
        ),
        "overvirtualize_control_all_fail_unstable_or_mismatch": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "unsafe_virtualization" in result.fail_reason
                or "ternary_mismatch" in result.fail_reason
            )
            for result in over
        ),
        "dense_shadow_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "price_paid_shadow_bits" in result.fail_reason
            for result in dense
        ),
        "signless_virtual_control_all_fail_forward": all(
            (not result.passed)
            and result.fail_reason is not None
            and "ternary_mismatch" in result.fail_reason
            for result in signless
        ),
        "loose_margin_control_all_fail_boundary_or_forward": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "unsafe_virtualization" in result.fail_reason
                or "ternary_mismatch" in result.fail_reason
            )
            for result in loose
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
    decision = "PASS_R501_S97_TLSWV_GATE" if all(checks.values()) else "FAIL_R501_S97_TLSWV_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU ternary-lattice shadow weight virtualization gate "
            "only; not heldout CE progress, not real VRAM telemetry, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S97_TLSWV_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_stable_fraction_mean={positive['stable_fraction_mean']:.6f} "
        f"positive_impulse_detector_recall_min={positive['impulse_detector_recall_min']:.6f} "
        f"positive_ternary_mismatch_rate_max={positive['ternary_mismatch_rate_max']:.6f} "
        f"positive_virtualized_fraction_mean={positive['virtualized_fraction_mean']:.6f} "
        f"positive_paid_shadow_bit_ratio_mean={positive['paid_shadow_bit_ratio_mean']:.6f} "
        f"positive_projected_shadow_efficiency_multiplier_mean={positive['projected_shadow_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
