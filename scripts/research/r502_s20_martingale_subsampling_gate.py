#!/usr/bin/env python3
"""R502-S20 Martingale-Priced Backward Sub-Sampling gate.

This synthetic CPU gate checks whether variance-priced backward sub-sampling
with Horvitz-Thompson weights remains an unbiased martingale estimator while
reducing paid backward work. Controls cover missing inverse-probability weights,
uniform sampling under heavy-tailed gradient energy, too-low sampling mass,
proxy cost, and noisy proxy probabilities.
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
    "biased_weight_fail",
    "uniform_variance_fail",
    "low_sample_fail",
    "proxy_cost_fail",
    "noisy_proxy_fail",
]


@dataclass
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str
    token_count: int
    gradient_dim: int
    sample_count: int
    high_energy_fraction: float
    mean_sampling_fraction: float
    min_sampling_probability: float
    max_sampling_probability: float
    proxy_cost_ratio: float
    priced_work_ratio: float
    martingale_bias_fraction: float
    estimator_relative_rmse: float
    descent_ratio_p10: float
    descent_ratio_mean: float
    descent_per_paid_work_p10: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=50220)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--token-count", type=int, default=512)
    parser.add_argument("--gradient-dim", type=int, default=24)
    parser.add_argument("--sample-count", type=int, default=768)
    parser.add_argument("--target-sampling-fraction", type=float, default=0.30)
    parser.add_argument("--low-target-sampling-fraction", type=float, default=0.05)
    parser.add_argument("--min-probability", type=float, default=0.04)
    parser.add_argument("--max-probability", type=float, default=0.95)
    parser.add_argument("--proxy-cost-ratio", type=float, default=0.05)
    parser.add_argument("--expensive-proxy-cost-ratio", type=float, default=0.75)
    parser.add_argument("--high-energy-fraction", type=float, default=0.0625)
    parser.add_argument("--high-energy-scale", type=float, default=10.0)
    parser.add_argument("--low-energy-scale", type=float, default=0.20)
    parser.add_argument("--noise-scale", type=float, default=0.08)
    parser.add_argument("--max-martingale-bias-fraction", type=float, default=0.05)
    parser.add_argument("--max-estimator-relative-rmse", type=float, default=0.15)
    parser.add_argument("--min-descent-per-paid-work-p10", type=float, default=1.0)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.90)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def make_gradients(args: argparse.Namespace, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    direction = rng.normal(size=args.gradient_dim)
    direction = direction / np.linalg.norm(direction)
    token_count = args.token_count
    high_count = int(round(token_count * args.high_energy_fraction))
    high_mask = np.zeros(token_count, dtype=bool)
    high_mask[:high_count] = True
    rng.shuffle(high_mask)
    amplitudes = np.full(token_count, args.low_energy_scale, dtype=np.float64)
    amplitudes[high_mask] = args.high_energy_scale
    noise = rng.normal(scale=args.noise_scale, size=(token_count, args.gradient_dim))
    gradients = amplitudes[:, None] * direction[None, :] + noise
    return gradients, high_mask


def probabilities(
    args: argparse.Namespace,
    gradients: np.ndarray,
    rng: np.random.Generator,
    condition: str,
) -> np.ndarray:
    target = (
        args.low_target_sampling_fraction
        if condition == "low_sample_fail"
        else args.target_sampling_fraction
    )
    energy = np.sum(gradients * gradients, axis=1)
    if condition == "uniform_variance_fail":
        raw = np.ones_like(energy)
    elif condition == "noisy_proxy_fail":
        raw = np.sqrt(np.maximum(np.roll(energy, gradients.shape[0] // 3), 1e-12))
        raw = raw * rng.uniform(0.10, 2.0, size=raw.shape)
    else:
        raw = np.sqrt(np.maximum(energy, 1e-12))
    raw_mean = float(np.mean(raw))
    scaled = raw / raw_mean * target
    return np.clip(scaled, args.min_probability, args.max_probability)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    gradients, high_mask = make_gradients(args, rng)
    probs = probabilities(args, gradients, rng, condition)
    exact = np.sum(gradients, axis=0)
    exact_norm_sq = float(np.dot(exact, exact))

    masks = rng.random(size=(args.sample_count, args.token_count)) < probs[None, :]
    if condition == "biased_weight_fail":
        estimates = masks.astype(np.float64) @ gradients
    else:
        estimates = (masks.astype(np.float64) / probs[None, :]) @ gradients

    mean_estimate = np.mean(estimates, axis=0)
    errors = estimates - exact[None, :]
    martingale_bias = float(np.linalg.norm(mean_estimate - exact) / np.linalg.norm(exact))
    relative_rmse = float(np.sqrt(np.mean(np.sum(errors * errors, axis=1))) / np.linalg.norm(exact))
    if exact_norm_sq == 0.0:
        descent_ratios = np.zeros(args.sample_count, dtype=np.float64)
    else:
        descent_ratios = (estimates @ exact) / exact_norm_sq
    descent_ratio_p10 = float(np.percentile(descent_ratios, 10))
    descent_ratio_mean = float(np.mean(descent_ratios))
    proxy_cost = (
        args.expensive_proxy_cost_ratio
        if condition == "proxy_cost_fail"
        else args.proxy_cost_ratio
    )
    mean_sampling = float(np.mean(probs))
    price = mean_sampling + proxy_cost
    descent_per_paid = descent_ratio_p10 / price

    failed: list[str] = []
    if not np.isfinite(
        [
            martingale_bias,
            relative_rmse,
            descent_ratio_p10,
            descent_ratio_mean,
            descent_per_paid,
            price,
            mean_sampling,
        ]
    ).all():
        failed.append("non_finite_metric")
    if martingale_bias > args.max_martingale_bias_fraction:
        failed.append("martingale_bias")
    if relative_rmse > args.max_estimator_relative_rmse:
        failed.append("estimator_variance")
    if descent_per_paid < args.min_descent_per_paid_work_p10:
        failed.append("descent_per_paid_work")
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
        token_count=args.token_count,
        gradient_dim=args.gradient_dim,
        sample_count=args.sample_count,
        high_energy_fraction=float(np.mean(high_mask)),
        mean_sampling_fraction=mean_sampling,
        min_sampling_probability=float(np.min(probs)),
        max_sampling_probability=float(np.max(probs)),
        proxy_cost_ratio=proxy_cost,
        priced_work_ratio=price,
        martingale_bias_fraction=martingale_bias,
        estimator_relative_rmse=relative_rmse,
        descent_ratio_p10=descent_ratio_p10,
        descent_ratio_mean=descent_ratio_mean,
        descent_per_paid_work_p10=descent_per_paid,
    )


NUMERIC_FIELDS = [
    "token_count",
    "gradient_dim",
    "sample_count",
    "high_energy_fraction",
    "mean_sampling_fraction",
    "min_sampling_probability",
    "max_sampling_probability",
    "proxy_cost_ratio",
    "priced_work_ratio",
    "martingale_bias_fraction",
    "estimator_relative_rmse",
    "descent_ratio_p10",
    "descent_ratio_mean",
    "descent_per_paid_work_p10",
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
        "positive_condition_martingale_and_price": (
            aggregates["positive"]["martingale_bias_fraction_max"] <= 0.05
            and aggregates["positive"]["estimator_relative_rmse_max"] <= 0.35
            and aggregates["positive"]["descent_per_paid_work_p10_min"] >= 1.0
            and aggregates["positive"]["priced_work_ratio_max"] <= 0.90
        ),
        "biased_weight_control_all_fail_bias": all(
            (not result.passed)
            and (
                "martingale_bias" in result.fail_reason
                or "descent_per_paid_work" in result.fail_reason
            )
            for result in by_condition["biased_weight_fail"]
        ),
        "uniform_variance_control_all_fail_variance": all(
            (not result.passed)
            and (
                "estimator_variance" in result.fail_reason
                or "descent_per_paid_work" in result.fail_reason
            )
            for result in by_condition["uniform_variance_fail"]
        ),
        "low_sample_control_all_fail_variance": all(
            (not result.passed)
            and (
                "estimator_variance" in result.fail_reason
                or "descent_per_paid_work" in result.fail_reason
            )
            for result in by_condition["low_sample_fail"]
        ),
        "proxy_cost_control_all_fail_price": all(
            (not result.passed) and "priced_work" in result.fail_reason
            for result in by_condition["proxy_cost_fail"]
        ),
        "noisy_proxy_control_all_fail_variance": all(
            (not result.passed)
            and (
                "estimator_variance" in result.fail_reason
                or "descent_per_paid_work" in result.fail_reason
            )
            for result in by_condition["noisy_proxy_fail"]
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
    decision = "PASS_R502_S20_MARTINGALE_SUBSAMPLING_GATE" if all(checks.values()) else "FAIL_R502_S20_MARTINGALE_SUBSAMPLING_GATE"

    payload = {
        "round": "R502",
        "canonical_id": "R502-S20",
        "aspect": "Speed",
        "finding": "Martingale-Priced Backward Sub-Sampling",
        "decision": decision,
        "disposition": "synthetic_positive_control_identified_not_training_speed_claim",
        "args": vars(args) | {"output": str(args.output) if args.output else None},
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU martingale-priced backward sub-sampling gate only; "
            "not heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_S20_MARTINGALE_SUBSAMPLING_GATE] "
        f"active=1 decision={decision} "
        f"disposition={payload['disposition']} "
        f"positive_mean_sampling_fraction={positive['mean_sampling_fraction_mean']:.12f} "
        f"positive_martingale_bias_max={positive['martingale_bias_fraction_max']:.12e} "
        f"positive_estimator_rmse_max={positive['estimator_relative_rmse_max']:.12f} "
        f"positive_descent_per_paid_work_p10_min={positive['descent_per_paid_work_p10_min']:.12f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.12f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
