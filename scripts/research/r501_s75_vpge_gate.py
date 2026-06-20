#!/usr/bin/env python3
"""R501 S75 variance-priced gradient equalization gate.

This CPU diagnostic implements the R501 Quality Math 12 finding:
"Variance-Priced Gradient Equalization (VPGE)".

The gate constructs a synthetic batch-gradient estimator where per-token
gradient noise scales with token variance. A positive case must strictly
increase the signal-to-noise ratio (SNR) of the aggregated batch gradient while
preserving the CE gradient direction and reducing high-variance spike
contributions.

Controls exercise the declared risks: a bad variance estimate can amplify the
same spikes VPGE should suppress, variance weighting can bias the CE direction
when structural gradients correlate with variance, and a reduction pass that is
too expensive erases the priced denominator.

This is a synthetic CPU gradient-estimator diagnostic, not heldout CE progress,
real corpus token efficiency, GPU training-speed evidence, or production v12
integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s75_vpge_gate.v1"
CONDITIONS = (
    "positive",
    "noisy_variance_estimate_fail",
    "direction_bias_fail",
    "reduction_cost_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    token_count: int
    gradient_dim: int
    replicate_count: int
    baseline_snr: float
    vpge_snr: float
    snr_gain: float
    ce_gradient_cosine: float
    variance_estimate_relative_error: float
    high_variance_contribution_reduction: float
    priced_work_ratio: float
    projected_token_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50175)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--token-count", type=int, default=256)
    parser.add_argument("--gradient-dim", type=int, default=24)
    parser.add_argument("--replicate-count", type=int, default=512)
    parser.add_argument("--min-snr-gain", type=float, default=1.35)
    parser.add_argument("--min-ce-gradient-cosine", type=float, default=0.995)
    parser.add_argument("--max-variance-estimate-relative-error", type=float, default=0.10)
    parser.add_argument("--min-high-variance-contribution-reduction", type=float, default=0.45)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.88)
    parser.add_argument("--priced-work-ratio", type=float, default=0.81)
    parser.add_argument("--expensive-priced-work-ratio", type=float, default=0.93)
    return parser.parse_args()


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1.0e-30)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0e-30)
    return float(np.dot(a, b) / denom)


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def make_token_statistics(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_direction = normalize(rng.normal(size=args.gradient_dim))
    auxiliary_direction = rng.normal(size=args.gradient_dim)
    auxiliary_direction -= np.dot(auxiliary_direction, base_direction) * base_direction
    auxiliary_direction = normalize(auxiliary_direction)

    true_variance = np.ones(args.token_count, dtype=np.float64)
    high_count = args.token_count // 4
    mid_count = args.token_count // 4
    true_variance[:high_count] = 36.0
    true_variance[high_count : high_count + mid_count] = 9.0
    rng.shuffle(true_variance)

    true_gradients = np.repeat(base_direction[None, :], args.token_count, axis=0)
    true_gradients += rng.normal(scale=0.015, size=true_gradients.shape)

    if condition == "direction_bias_fail":
        low_variance = true_variance <= 1.0
        high_variance = true_variance >= 36.0
        true_gradients[low_variance] += 1.90 * auxiliary_direction
        true_gradients[high_variance] -= 0.45 * auxiliary_direction

    if condition == "noisy_variance_estimate_fail":
        estimated_variance = 36.0 / true_variance
        estimated_variance *= rng.lognormal(mean=0.0, sigma=0.18, size=args.token_count)
    else:
        estimated_variance = true_variance * rng.lognormal(
            mean=0.0,
            sigma=0.018,
            size=args.token_count,
        )

    return true_gradients, true_variance, estimated_variance


def aggregate_snr(
    rng: np.random.Generator,
    true_gradients: np.ndarray,
    true_variance: np.ndarray,
    weights: np.ndarray,
    replicate_count: int,
) -> tuple[float, float, float]:
    token_count, gradient_dim = true_gradients.shape
    noise = rng.normal(
        size=(replicate_count, token_count, gradient_dim),
        scale=np.sqrt(true_variance)[None, :, None],
    )
    observed = true_gradients[None, :, :] + noise

    baseline_expected = np.mean(true_gradients, axis=0)
    baseline_aggregates = np.mean(observed, axis=1)
    baseline_noise = baseline_aggregates - baseline_expected[None, :]
    baseline_noise_rms = math.sqrt(float(np.mean(np.sum(baseline_noise * baseline_noise, axis=1))))
    baseline_snr = float(np.linalg.norm(baseline_expected)) / max(baseline_noise_rms, 1.0e-30)

    weighted_true = weights[:, None] * true_gradients
    weighted_observed = weights[None, :, None] * observed
    vpge_expected = np.mean(weighted_true, axis=0)
    vpge_aggregates = np.mean(weighted_observed, axis=1)
    vpge_noise = vpge_aggregates - vpge_expected[None, :]
    vpge_noise_rms = math.sqrt(float(np.mean(np.sum(vpge_noise * vpge_noise, axis=1))))
    vpge_snr = float(np.linalg.norm(vpge_expected)) / max(vpge_noise_rms, 1.0e-30)

    ce_cosine = cosine(vpge_expected, baseline_expected)
    return baseline_snr, vpge_snr, ce_cosine


def contribution_reduction(
    true_gradients: np.ndarray,
    true_variance: np.ndarray,
    weights: np.ndarray,
) -> float:
    high_variance = true_variance >= 36.0
    baseline_contrib = np.linalg.norm(true_gradients[high_variance], axis=1) * np.sqrt(
        true_variance[high_variance]
    )
    vpge_contrib = baseline_contrib * weights[high_variance]
    baseline_mean = max(float(np.mean(baseline_contrib)), 1.0e-30)
    vpge_mean = float(np.mean(vpge_contrib))
    return float(1.0 - (vpge_mean / baseline_mean))


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    true_gradients, true_variance, estimated_variance = make_token_statistics(
        rng,
        args,
        condition,
    )
    weights = 1.0 / np.sqrt(np.maximum(estimated_variance, 1.0e-12))
    baseline_snr, vpge_snr, ce_cosine = aggregate_snr(
        rng,
        true_gradients,
        true_variance,
        weights,
        args.replicate_count,
    )
    snr_gain = vpge_snr / max(baseline_snr, 1.0e-30)
    variance_error = float(
        np.mean(np.abs(estimated_variance - true_variance) / np.maximum(true_variance, 1.0e-12))
    )
    high_variance_reduction = contribution_reduction(true_gradients, true_variance, weights)
    priced_work_ratio = (
        args.expensive_priced_work_ratio
        if condition == "reduction_cost_fail"
        else args.priced_work_ratio
    )
    projected_multiplier = 1.0 / priced_work_ratio

    failed: list[str] = []
    if not finite(
        baseline_snr,
        vpge_snr,
        snr_gain,
        ce_cosine,
        variance_error,
        high_variance_reduction,
        priced_work_ratio,
        projected_multiplier,
    ):
        failed.append("non_finite_metric")
    if snr_gain < args.min_snr_gain:
        failed.append("snr_gain")
    if ce_cosine < args.min_ce_gradient_cosine:
        failed.append("ce_direction")
    if variance_error > args.max_variance_estimate_relative_error:
        failed.append("variance_estimate")
    if high_variance_reduction < args.min_high_variance_contribution_reduction:
        failed.append("spike_reduction")
    if priced_work_ratio > args.max_priced_work_ratio:
        failed.append("price")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed) if failed else None,
        token_count=args.token_count,
        gradient_dim=args.gradient_dim,
        replicate_count=args.replicate_count,
        baseline_snr=float(baseline_snr),
        vpge_snr=float(vpge_snr),
        snr_gain=float(snr_gain),
        ce_gradient_cosine=float(ce_cosine),
        variance_estimate_relative_error=float(variance_error),
        high_variance_contribution_reduction=float(high_variance_reduction),
        priced_work_ratio=float(priced_work_ratio),
        projected_token_efficiency_multiplier=float(projected_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "token_count",
        "gradient_dim",
        "replicate_count",
        "baseline_snr",
        "vpge_snr",
        "snr_gain",
        "ce_gradient_cosine",
        "variance_estimate_relative_error",
        "high_variance_contribution_reduction",
        "priced_work_ratio",
        "projected_token_efficiency_multiplier",
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
    noisy = [result for result in results if result.condition == "noisy_variance_estimate_fail"]
    direction = [result for result in results if result.condition == "direction_bias_fail"]
    cost = [result for result in results if result.condition == "reduction_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_snr_improved": (
            aggregates["positive"]["snr_gain_mean"] >= 1.35
            and aggregates["positive"]["vpge_snr_mean"] > aggregates["positive"]["baseline_snr_mean"]
        ),
        "positive_condition_direction_preserved": (
            aggregates["positive"]["ce_gradient_cosine_min"] >= 0.995
        ),
        "positive_condition_spikes_reduced": (
            aggregates["positive"]["high_variance_contribution_reduction_min"] >= 0.45
        ),
        "positive_condition_priced": (
            aggregates["positive"]["priced_work_ratio_mean"] <= 0.88
        ),
        "noisy_variance_control_all_fail_estimate_or_snr": all(
            (not result.passed)
            and result.fail_reason is not None
            and ("variance_estimate" in result.fail_reason or "snr_gain" in result.fail_reason)
            for result in noisy
        ),
        "direction_control_all_fail_direction": all(
            (not result.passed)
            and result.fail_reason is not None
            and "ce_direction" in result.fail_reason
            for result in direction
        ),
        "cost_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "price" in result.fail_reason
            for result in cost
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
    decision = (
        "PASS_R501_S75_VPGE_GATE"
        if all(checks.values())
        else "FAIL_R501_S75_VPGE_GATE"
    )
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU variance-priced gradient equalization gate only; "
            "not heldout CE progress, not real corpus token efficiency, not "
            "GPU training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S75_VPGE_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_baseline_snr_mean={positive['baseline_snr_mean']:.6f} "
        f"positive_vpge_snr_mean={positive['vpge_snr_mean']:.6f} "
        f"positive_snr_gain_mean={positive['snr_gain_mean']:.6f} "
        f"positive_ce_gradient_cosine_min={positive['ce_gradient_cosine_min']:.6f} "
        f"positive_high_variance_contribution_reduction_min={positive['high_variance_contribution_reduction_min']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_token_efficiency_multiplier_mean={positive['projected_token_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
