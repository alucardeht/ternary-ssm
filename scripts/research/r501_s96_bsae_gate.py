#!/usr/bin/env python3
"""R501 S96 boundary-sparse activation escrow gate.

This CPU diagnostic implements the third R501 Hardware Efficiency mathematical
finding: "Boundary-Sparse Activation Escrow (BSAE)".

The gate constructs a synthetic ternary-boundary fixture where only elements
with margin m_i - B_i rho_i <= 0 can induce a sign-flip gradient impulse. The
positive condition stores a one-bit boundary mask for every activation and
escrows full FP16 magnitudes only for boundary-active elements, then verifies
that the structural ternary gradient matches dense activation residency.

Controls exercise the declared kill criteria: no active escrow loses gradient
magnitudes, inverted boundaries store the wrong side, quantized active
magnitudes break exactness, dense activation residency erases the denominator,
and random masks miss active structural-gradient elements.

This is a synthetic CPU boundary-sparse activation memory diagnostic, not
heldout CE progress, real VRAM telemetry, GPU training-speed evidence, or
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


SCHEMA = "r501_s96_bsae_gate.v1"
CONDITIONS = (
    "positive",
    "no_escrow_fail",
    "inverted_boundary_fail",
    "quantized_magnitude_fail",
    "dense_activation_fail",
    "random_mask_fail",
)
FP16_BITS = 16.0
MASK_BITS = 1.0


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
    element_count: int
    active_fraction: float
    candidate_escrow_fraction: float
    active_recall: float
    inactive_rejection: float
    gradient_relative_error: float
    gradient_max_abs_error: float
    dense_activation_bits: float
    candidate_activation_bits: float
    paid_activation_bit_ratio: float
    projected_activation_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50196)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sequence-len", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--active-probability", type=float, default=0.12)
    parser.add_argument("--quantization-levels", type=int, default=16)
    parser.add_argument("--max-gradient-relative-error", type=float, default=1.0e-12)
    parser.add_argument("--max-gradient-abs-error", type=float, default=1.0e-12)
    parser.add_argument("--min-active-recall", type=float, default=1.0)
    parser.add_argument("--min-inactive-rejection", type=float, default=1.0)
    parser.add_argument("--max-paid-activation-bit-ratio", type=float, default=0.25)
    parser.add_argument("--min-activation-efficiency-multiplier", type=float, default=4.0)
    return parser.parse_args()


def make_fixture(
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (args.sequence_len, args.state_dim)
    activations = rng.normal(size=shape).astype(np.float64)
    cotangents = rng.normal(size=shape).astype(np.float64)
    boundary = rng.uniform(0.8, 1.2, size=shape)
    radius = rng.uniform(0.5, 1.5, size=shape)
    active = rng.random(size=shape) < args.active_probability
    offset = np.empty(shape, dtype=np.float64)
    offset[active] = -rng.uniform(0.01, 0.5, size=int(np.count_nonzero(active)))
    offset[~active] = rng.uniform(0.01, 0.5, size=int(np.count_nonzero(~active)))
    margin_proxy = boundary * radius + offset
    structural_margin = margin_proxy - boundary * radius
    structural_active = structural_margin <= 0.0
    return activations, cotangents, structural_active, boundary, radius


def dense_structural_gradient(
    activations: np.ndarray,
    cotangents: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    return np.where(active_mask, activations * cotangents, 0.0)


def quantize_active_values(values: np.ndarray, levels: int) -> np.ndarray:
    if levels < 2:
        raise ValueError("quantization_levels must be at least 2")
    scale = max(float(np.max(np.abs(values))), 1.0e-30)
    normalized = np.clip(values / scale, -1.0, 1.0)
    quantized = np.round((normalized + 1.0) * (levels - 1) / 2.0)
    restored = (quantized * 2.0 / (levels - 1) - 1.0) * scale
    return restored


def candidate_gradient(
    rng: np.random.Generator,
    activations: np.ndarray,
    cotangents: np.ndarray,
    active_mask: np.ndarray,
    condition: str,
    quantization_levels: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if condition == "inverted_boundary_fail":
        escrow_mask = ~active_mask
    elif condition == "random_mask_fail":
        flat_count = active_mask.size
        active_count = int(np.count_nonzero(active_mask))
        flat_mask = np.zeros(flat_count, dtype=bool)
        chosen = rng.choice(flat_count, size=active_count, replace=False)
        flat_mask[chosen] = True
        escrow_mask = flat_mask.reshape(active_mask.shape)
    else:
        escrow_mask = active_mask.copy()

    if condition == "no_escrow_fail":
        restored = np.zeros_like(activations)
    elif condition == "quantized_magnitude_fail":
        restored = np.zeros_like(activations)
        restored[escrow_mask] = quantize_active_values(
            activations[escrow_mask],
            levels=quantization_levels,
        )
    else:
        restored = np.zeros_like(activations)
        restored[escrow_mask] = activations[escrow_mask]

    dense_residency = condition == "dense_activation_fail"
    if dense_residency:
        restored = activations.copy()
        escrow_mask = np.ones_like(active_mask, dtype=bool)

    gradient = np.where(active_mask, restored * cotangents, 0.0)
    return gradient, escrow_mask, dense_residency


def relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference))
    return float(np.linalg.norm(candidate - reference) / max(denom, 1.0e-30))


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    activations, cotangents, active_mask, _boundary, _radius = make_fixture(rng, args)
    dense_gradient = dense_structural_gradient(activations, cotangents, active_mask)
    candidate, escrow_mask, dense_residency = candidate_gradient(
        rng=rng,
        activations=activations,
        cotangents=cotangents,
        active_mask=active_mask,
        condition=condition,
        quantization_levels=args.quantization_levels,
    )

    element_count = int(active_mask.size)
    active_count = int(np.count_nonzero(active_mask))
    inactive_count = element_count - active_count
    escrow_count = int(np.count_nonzero(escrow_mask))
    active_recall = float(
        np.count_nonzero(escrow_mask & active_mask) / max(active_count, 1)
    )
    inactive_rejection = float(
        np.count_nonzero((~escrow_mask) & (~active_mask)) / max(inactive_count, 1)
    )
    grad_error = relative_error(candidate, dense_gradient)
    max_abs_error = float(np.max(np.abs(candidate - dense_gradient)))
    dense_bits = float(element_count * FP16_BITS)
    if dense_residency:
        candidate_bits = float(element_count * (MASK_BITS + FP16_BITS))
    else:
        candidate_bits = float(element_count * MASK_BITS + escrow_count * FP16_BITS)
    paid_ratio = candidate_bits / dense_bits
    multiplier = 1.0 / paid_ratio

    failures: list[str] = []
    if grad_error > args.max_gradient_relative_error or max_abs_error > args.max_gradient_abs_error:
        failures.append("gradient_error")
    if active_recall < args.min_active_recall:
        failures.append("active_recall")
    if inactive_rejection < args.min_inactive_rejection:
        failures.append("inactive_rejection")
    if (
        paid_ratio > args.max_paid_activation_bit_ratio
        or multiplier < args.min_activation_efficiency_multiplier
    ):
        failures.append("price_paid_activation_bits")

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
        element_count=element_count,
        active_fraction=float(active_count / element_count),
        candidate_escrow_fraction=float(escrow_count / element_count),
        active_recall=active_recall,
        inactive_rejection=inactive_rejection,
        gradient_relative_error=grad_error,
        gradient_max_abs_error=max_abs_error,
        dense_activation_bits=dense_bits,
        candidate_activation_bits=candidate_bits,
        paid_activation_bit_ratio=paid_ratio,
        projected_activation_efficiency_multiplier=multiplier,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "sequence_len",
        "state_dim",
        "element_count",
        "active_fraction",
        "candidate_escrow_fraction",
        "active_recall",
        "inactive_rejection",
        "gradient_relative_error",
        "gradient_max_abs_error",
        "dense_activation_bits",
        "candidate_activation_bits",
        "paid_activation_bit_ratio",
        "projected_activation_efficiency_multiplier",
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
    inverted = [result for result in results if result.condition == "inverted_boundary_fail"]
    quantized = [result for result in results if result.condition == "quantized_magnitude_fail"]
    dense = [result for result in results if result.condition == "dense_activation_fail"]
    random_mask = [result for result in results if result.condition == "random_mask_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_gradient": (
            aggregates["positive"]["active_recall_min"] >= 1.0
            and aggregates["positive"]["inactive_rejection_min"] >= 1.0
            and aggregates["positive"]["gradient_relative_error_max"] <= 1.0e-12
            and aggregates["positive"]["gradient_max_abs_error_max"] <= 1.0e-12
        ),
        "positive_condition_priced": (
            aggregates["positive"]["paid_activation_bit_ratio_mean"] <= 0.25
            and aggregates["positive"]["projected_activation_efficiency_multiplier_mean"] >= 4.0
        ),
        "no_escrow_control_all_fail_gradient": all(
            (not result.passed)
            and result.fail_reason is not None
            and "gradient_error" in result.fail_reason
            for result in no_escrow
        ),
        "inverted_boundary_control_all_fail_boundary_or_gradient": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "active_recall" in result.fail_reason
                or "inactive_rejection" in result.fail_reason
                or "gradient_error" in result.fail_reason
            )
            for result in inverted
        ),
        "quantized_magnitude_control_all_fail_exactness": all(
            (not result.passed)
            and result.fail_reason is not None
            and "gradient_error" in result.fail_reason
            for result in quantized
        ),
        "dense_activation_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "price_paid_activation_bits" in result.fail_reason
            for result in dense
        ),
        "random_mask_control_all_fail_recall_or_gradient": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "active_recall" in result.fail_reason
                or "inactive_rejection" in result.fail_reason
                or "gradient_error" in result.fail_reason
            )
            for result in random_mask
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
    decision = "PASS_R501_S96_BSAE_GATE" if all(checks.values()) else "FAIL_R501_S96_BSAE_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU boundary-sparse activation escrow gate only; not "
            "heldout CE progress, not real VRAM telemetry, not GPU training "
            "speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S96_BSAE_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_active_fraction_mean={positive['active_fraction_mean']:.6f} "
        f"positive_active_recall_min={positive['active_recall_min']:.6f} "
        f"positive_inactive_rejection_min={positive['inactive_rejection_min']:.6f} "
        f"positive_gradient_relative_error_max={positive['gradient_relative_error_max']:.6e} "
        f"positive_paid_activation_bit_ratio_mean={positive['paid_activation_bit_ratio_mean']:.6f} "
        f"positive_projected_activation_efficiency_multiplier_mean={positive['projected_activation_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
