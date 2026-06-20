#!/usr/bin/env python3
"""R502-S23 Homomorphic Ternary Gradient Compression gate.

This synthetic CPU gate checks whether stochastic low-precision adjoint
compression preserves ternary flip decisions in expectation while reducing the
priced backward-memory denominator. Controls cover too few bits, biased
rounding, near-boundary flips, scale saturation, and quantization overhead.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


CONDITIONS = [
    "positive",
    "low_precision_variance_fail",
    "biased_rounding_fail",
    "near_boundary_fail",
    "scale_saturation_fail",
    "quant_cost_fail",
]


@dataclass
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str
    param_count: int
    quant_bits: int
    sample_count: int
    exact_flip_fraction: float
    flip_mismatch_fraction: float
    flip_probability_error_mean: float
    expected_sign_error_mean: float
    expected_gradient_relative_error: float
    quality_progress_ratio: float
    saturation_fraction: float
    memory_ratio: float
    quant_cost_ratio: float
    priced_work_ratio: float
    projected_memory_saving_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=50223)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--param-count", type=int, default=4096)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--baseline-bits", type=float, default=32.0)
    parser.add_argument("--quant-cost-ratio", type=float, default=0.04)
    parser.add_argument("--expensive-quant-cost-ratio", type=float, default=0.92)
    parser.add_argument("--max-flip-mismatch-fraction", type=float, default=0.01)
    parser.add_argument("--max-flip-probability-error", type=float, default=0.05)
    parser.add_argument("--max-expected-sign-error", type=float, default=0.10)
    parser.add_argument("--max-gradient-relative-error", type=float, default=0.04)
    parser.add_argument("--min-quality-progress-ratio", type=float, default=0.98)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.90)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def make_fixture(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gradient = rng.normal(scale=0.22, size=args.param_count)
    spike_mask = rng.random(args.param_count) < 0.08
    gradient[spike_mask] += rng.normal(scale=0.55, size=np.sum(spike_mask))
    exact_sign = rng.choice([-1.0, 1.0], size=args.param_count)
    if condition == "near_boundary_fail":
        margin = rng.uniform(0.0002, 0.004, size=args.param_count)
    else:
        margin = rng.uniform(0.08, 0.18, size=args.param_count)
    exact_post = exact_sign * margin
    latent_before = exact_post + args.learning_rate * gradient
    old_sign = np.where(latent_before >= 0.0, 1.0, -1.0)
    return gradient, latent_before, old_sign, exact_sign


def quantize_once(
    rng: np.random.Generator,
    gradient: np.ndarray,
    bits: int,
    condition: str,
) -> tuple[np.ndarray, float]:
    levels = max((2 ** (bits - 1)) - 1, 1)
    max_abs = max(float(np.max(np.abs(gradient))), 1e-12)
    if condition == "scale_saturation_fail":
        scale = max(float(np.quantile(np.abs(gradient), 0.45)) / levels, 1e-12)
    elif condition == "biased_rounding_fail":
        scale = max_abs / max(levels * 0.45, 1.0)
    else:
        scale = max_abs / levels * 1.01
    normalized = gradient / scale
    clipped = np.clip(normalized, -levels, levels)
    saturation = float(np.mean(np.abs(normalized) > levels))
    if condition == "biased_rounding_fail":
        rounded = np.floor(clipped)
    else:
        lower = np.floor(clipped)
        probability = clipped - lower
        rounded = lower + (rng.random(gradient.shape) < probability)
    return rounded * scale, saturation


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    gradient, latent_before, old_sign, exact_sign = make_fixture(rng, args, condition)
    exact_flip = exact_sign != old_sign
    bits = 8
    if condition == "low_precision_variance_fail":
        bits = 2
    if condition == "biased_rounding_fail":
        bits = 4
    quant_cost = (
        args.expensive_quant_cost_ratio
        if condition == "quant_cost_fail"
        else args.quant_cost_ratio
    )

    sum_quantized = np.zeros_like(gradient)
    flip_probability = np.zeros_like(gradient)
    sign_expectation = np.zeros_like(gradient)
    saturation_values: list[float] = []
    for _ in range(args.sample_count):
        quantized, saturation = quantize_once(rng, gradient, bits, condition)
        saturation_values.append(saturation)
        sum_quantized += quantized
        quantized_post = latent_before - args.learning_rate * quantized
        quantized_sign = np.where(quantized_post >= 0.0, 1.0, -1.0)
        flip_probability += quantized_sign != old_sign
        sign_expectation += quantized_sign

    mean_quantized = sum_quantized / args.sample_count
    flip_probability /= args.sample_count
    sign_expectation /= args.sample_count
    predicted_flip = flip_probability >= 0.5
    flip_mismatch = float(np.mean(predicted_flip != exact_flip))
    flip_probability_error = float(np.mean(np.abs(flip_probability - exact_flip.astype(float))))
    expected_sign_error = float(np.mean(np.abs(sign_expectation - exact_sign)))
    grad_rel_error = float(
        np.linalg.norm(mean_quantized - gradient) / max(np.linalg.norm(gradient), 1e-12)
    )
    progress = float(
        np.vdot(gradient, mean_quantized).real / max(np.vdot(gradient, gradient).real, 1e-12)
    )
    memory_ratio = bits / args.baseline_bits
    price = memory_ratio + quant_cost
    failed: list[str] = []
    if not np.isfinite(
        [
            flip_mismatch,
            flip_probability_error,
            expected_sign_error,
            grad_rel_error,
            progress,
            memory_ratio,
            quant_cost,
            price,
        ]
    ).all():
        failed.append("non_finite_metric")
    if flip_mismatch > args.max_flip_mismatch_fraction:
        failed.append("flip_mismatch")
    if flip_probability_error > args.max_flip_probability_error:
        failed.append("flip_probability_error")
    if expected_sign_error > args.max_expected_sign_error:
        failed.append("expected_sign_error")
    if grad_rel_error > args.max_gradient_relative_error:
        failed.append("gradient_error")
    if progress < args.min_quality_progress_ratio:
        failed.append("quality_progress")
    if price > args.max_priced_work_ratio:
        failed.append("priced_work")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed),
        param_count=args.param_count,
        quant_bits=bits,
        sample_count=args.sample_count,
        exact_flip_fraction=float(np.mean(exact_flip)),
        flip_mismatch_fraction=flip_mismatch,
        flip_probability_error_mean=flip_probability_error,
        expected_sign_error_mean=expected_sign_error,
        expected_gradient_relative_error=grad_rel_error,
        quality_progress_ratio=progress,
        saturation_fraction=float(np.mean(saturation_values)),
        memory_ratio=memory_ratio,
        quant_cost_ratio=quant_cost,
        priced_work_ratio=price,
        projected_memory_saving_multiplier=1.0 / max(memory_ratio, 1e-12),
    )


NUMERIC_FIELDS = [
    "param_count",
    "quant_bits",
    "sample_count",
    "exact_flip_fraction",
    "flip_mismatch_fraction",
    "flip_probability_error_mean",
    "expected_sign_error_mean",
    "expected_gradient_relative_error",
    "quality_progress_ratio",
    "saturation_fraction",
    "memory_ratio",
    "quant_cost_ratio",
    "priced_work_ratio",
    "projected_memory_saving_multiplier",
]


def aggregate(results: list[TrialResult]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition in CONDITIONS:
        subset = [result for result in results if result.condition == condition]
        summary: dict[str, Any] = {
            "count": len(subset),
            "passed_count": sum(result.passed for result in subset),
            "identifying_correct_count": sum(result.identifying_correct for result in subset),
            "fail_reasons": sorted({result.fail_reason for result in subset if result.fail_reason}),
        }
        for field in NUMERIC_FIELDS:
            values = [float(getattr(result, field)) for result in subset]
            summary[f"{field}_mean"] = float(np.mean(values))
            summary[f"{field}_max"] = float(np.max(values))
            summary[f"{field}_min"] = float(np.min(values))
        output[condition] = summary
    return output


def required_checks(results: list[TrialResult], aggregates: dict[str, Any]) -> dict[str, bool]:
    by_condition = {
        condition: [result for result in results if result.condition == condition]
        for condition in CONDITIONS
    }
    return {
        "all_metrics_finite": all(
            math.isfinite(float(value))
            for result in results
            for value in asdict(result).values()
            if isinstance(value, (int, float, np.floating))
        ),
        "all_conditions_identified": all(result.identifying_correct for result in results),
        "positive_condition_all_pass": all(result.passed for result in by_condition["positive"]),
        "positive_condition_flip_and_price": (
            aggregates["positive"]["flip_mismatch_fraction_max"] <= 0.01
            and aggregates["positive"]["flip_probability_error_mean_max"] <= 0.05
            and aggregates["positive"]["expected_gradient_relative_error_max"] <= 0.04
            and aggregates["positive"]["quality_progress_ratio_min"] >= 0.98
            and aggregates["positive"]["priced_work_ratio_max"] <= 0.90
        ),
        "low_precision_control_all_fail_variance": all(
            (not result.passed)
            and (
                "flip_probability_error" in result.fail_reason
                or "expected_sign_error" in result.fail_reason
                or "flip_mismatch" in result.fail_reason
            )
            for result in by_condition["low_precision_variance_fail"]
        ),
        "biased_rounding_control_all_fail_bias": all(
            (not result.passed)
            and (
                "gradient_error" in result.fail_reason
                or "quality_progress" in result.fail_reason
                or "flip_probability_error" in result.fail_reason
            )
            for result in by_condition["biased_rounding_fail"]
        ),
        "near_boundary_control_all_fail_flip": all(
            (not result.passed)
            and (
                "flip_probability_error" in result.fail_reason
                or "expected_sign_error" in result.fail_reason
                or "flip_mismatch" in result.fail_reason
            )
            for result in by_condition["near_boundary_fail"]
        ),
        "scale_saturation_control_all_fail_quality": all(
            (not result.passed)
            and result.saturation_fraction > 0.0
            and (
                "gradient_error" in result.fail_reason
                or "quality_progress" in result.fail_reason
                or "flip_probability_error" in result.fail_reason
            )
            for result in by_condition["scale_saturation_fail"]
        ),
        "quant_cost_control_all_fail_price": all(
            (not result.passed) and "priced_work" in result.fail_reason
            for result in by_condition["quant_cost_fail"]
        ),
    }


def main() -> int:
    args = parse_args()
    results = [
        run_trial(args, condition, args.seed + index)
        for index in range(args.seed_count)
        for condition in CONDITIONS
    ]
    aggregates = aggregate(results)
    checks = required_checks(results, aggregates)
    decision = "PASS_R502_S23_HTGC_GATE" if all(checks.values()) else "FAIL_R502_S23_HTGC_GATE"
    payload = {
        "round": "R502",
        "canonical_id": "R502-S23",
        "aspect": "Speed",
        "finding": "Homomorphic Ternary Gradient Compression",
        "decision": decision,
        "disposition": "synthetic_positive_control_identified_not_training_speed_claim",
        "args": vars(args) | {"output": str(args.output) if args.output else None},
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU homomorphic ternary gradient-compression gate only; "
            "not heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_S23_HTGC_GATE] "
        f"active=1 decision={decision} "
        f"disposition={payload['disposition']} "
        f"positive_flip_mismatch_fraction_max={positive['flip_mismatch_fraction_max']:.12f} "
        f"positive_flip_probability_error_mean_max={positive['flip_probability_error_mean_max']:.12f} "
        f"positive_quality_progress_ratio_min={positive['quality_progress_ratio_min']:.12f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.12f} "
        f"positive_memory_saving_multiplier_mean={positive['projected_memory_saving_multiplier_mean']:.12f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
