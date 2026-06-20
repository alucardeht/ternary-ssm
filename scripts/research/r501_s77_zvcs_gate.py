#!/usr/bin/env python3
"""R501 S77 zero-variance credit subtraction gate.

This CPU diagnostic implements the R501 Quality Math 14 finding:
"Zero-Variance Credit Subtraction (ZVCS)".

The gate constructs active gradients with a true CE signal plus a large shared
noise component. A positive EMA control variate has near-zero expectation and
captures that shared noise, so subtracting it must reduce gradient variance
while preserving the true capability direction.

Controls exercise the declared risks: a drifted EMA becomes an adversarial
perturbation, an uncorrelated EMA increases variance instead of reducing it, and
an expensive shadow-model pass erases the priced denominator.

This is a synthetic CPU control-variate gradient diagnostic, not heldout CE
progress, real corpus token efficiency, GPU training-speed evidence, or
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


SCHEMA = "r501_s77_zvcs_gate.v1"
CONDITIONS = (
    "positive",
    "drifted_ema_fail",
    "uncorrelated_ema_fail",
    "shadow_cost_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    sample_count: int
    gradient_dim: int
    baseline_variance_trace: float
    zvcs_variance_trace: float
    variance_ratio: float
    baseline_snr: float
    zvcs_snr: float
    snr_gain: float
    gradient_cosine_to_ce: float
    mean_bias_norm: float
    ema_drift_norm: float
    priced_work_ratio: float
    projected_token_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50177)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sample-count", type=int, default=768)
    parser.add_argument("--gradient-dim", type=int, default=24)
    parser.add_argument("--shared-noise-scale", type=float, default=1.20)
    parser.add_argument("--active-idiosyncratic-scale", type=float, default=0.35)
    parser.add_argument("--ema-idiosyncratic-scale", type=float, default=0.25)
    parser.add_argument("--max-variance-ratio", type=float, default=0.35)
    parser.add_argument("--min-snr-gain", type=float, default=1.60)
    parser.add_argument("--min-gradient-cosine", type=float, default=0.995)
    parser.add_argument("--max-mean-bias-norm", type=float, default=0.10)
    parser.add_argument("--max-ema-drift-norm", type=float, default=0.12)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.80)
    parser.add_argument("--priced-work-ratio", type=float, default=0.74)
    parser.add_argument("--expensive-priced-work-ratio", type=float, default=0.86)
    return parser.parse_args()


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1.0e-30)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0e-30)
    return float(np.dot(a, b) / denom)


def variance_trace(samples: np.ndarray) -> float:
    centered = samples - np.mean(samples, axis=0, keepdims=True)
    return float(np.mean(np.sum(centered * centered, axis=1)))


def snr(samples: np.ndarray, signal: np.ndarray) -> float:
    mean = np.mean(samples, axis=0)
    noise = samples - mean[None, :]
    noise_rms = math.sqrt(float(np.mean(np.sum(noise * noise, axis=1))))
    return float(np.linalg.norm(signal)) / max(noise_rms, 1.0e-30)


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def make_gradients(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    signal = normalize(rng.normal(size=args.gradient_dim))
    orthogonal = rng.normal(size=args.gradient_dim)
    orthogonal -= np.dot(orthogonal, signal) * signal
    orthogonal = normalize(orthogonal)

    shared_noise = rng.normal(
        scale=args.shared_noise_scale,
        size=(args.sample_count, args.gradient_dim),
    )
    active = (
        signal[None, :]
        + shared_noise
        + rng.normal(
            scale=args.active_idiosyncratic_scale,
            size=(args.sample_count, args.gradient_dim),
        )
    )

    if condition == "uncorrelated_ema_fail":
        ema = rng.normal(
            scale=args.shared_noise_scale,
            size=(args.sample_count, args.gradient_dim),
        )
    else:
        ema = shared_noise + rng.normal(
            scale=args.ema_idiosyncratic_scale,
            size=(args.sample_count, args.gradient_dim),
        )

    ema_drift = np.zeros(args.gradient_dim, dtype=np.float64)
    if condition == "drifted_ema_fail":
        ema_drift = 0.62 * signal + 0.90 * orthogonal
        ema = ema + ema_drift[None, :]

    return signal, active, ema, ema_drift


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    signal, active, ema, ema_drift = make_gradients(rng, args, condition)
    zvcs = active - ema

    baseline_var = variance_trace(active)
    zvcs_var = variance_trace(zvcs)
    variance_ratio = zvcs_var / max(baseline_var, 1.0e-30)
    baseline_snr = snr(active, signal)
    zvcs_snr = snr(zvcs, signal)
    snr_gain = zvcs_snr / max(baseline_snr, 1.0e-30)
    gradient_cosine = cosine(np.mean(zvcs, axis=0), signal)
    mean_bias_norm = float(np.linalg.norm(np.mean(zvcs, axis=0) - signal))
    ema_drift_norm = float(np.linalg.norm(ema_drift))
    priced_work_ratio = (
        args.expensive_priced_work_ratio
        if condition == "shadow_cost_fail"
        else args.priced_work_ratio
    )
    projected_multiplier = 1.0 / priced_work_ratio

    failed: list[str] = []
    if not finite(
        baseline_var,
        zvcs_var,
        variance_ratio,
        baseline_snr,
        zvcs_snr,
        snr_gain,
        gradient_cosine,
        mean_bias_norm,
        ema_drift_norm,
        priced_work_ratio,
        projected_multiplier,
    ):
        failed.append("non_finite_metric")
    if variance_ratio > args.max_variance_ratio:
        failed.append("variance_reduction")
    if snr_gain < args.min_snr_gain:
        failed.append("snr_gain")
    if gradient_cosine < args.min_gradient_cosine:
        failed.append("ce_direction")
    if mean_bias_norm > args.max_mean_bias_norm:
        failed.append("mean_bias")
    if ema_drift_norm > args.max_ema_drift_norm:
        failed.append("ema_drift")
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
        sample_count=args.sample_count,
        gradient_dim=args.gradient_dim,
        baseline_variance_trace=float(baseline_var),
        zvcs_variance_trace=float(zvcs_var),
        variance_ratio=float(variance_ratio),
        baseline_snr=float(baseline_snr),
        zvcs_snr=float(zvcs_snr),
        snr_gain=float(snr_gain),
        gradient_cosine_to_ce=float(gradient_cosine),
        mean_bias_norm=float(mean_bias_norm),
        ema_drift_norm=float(ema_drift_norm),
        priced_work_ratio=float(priced_work_ratio),
        projected_token_efficiency_multiplier=float(projected_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "sample_count",
        "gradient_dim",
        "baseline_variance_trace",
        "zvcs_variance_trace",
        "variance_ratio",
        "baseline_snr",
        "zvcs_snr",
        "snr_gain",
        "gradient_cosine_to_ce",
        "mean_bias_norm",
        "ema_drift_norm",
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
    drifted = [result for result in results if result.condition == "drifted_ema_fail"]
    uncorrelated = [result for result in results if result.condition == "uncorrelated_ema_fail"]
    cost = [result for result in results if result.condition == "shadow_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_variance_reduced": (
            aggregates["positive"]["variance_ratio_mean"] <= 0.35
            and aggregates["positive"]["zvcs_variance_trace_mean"]
            < aggregates["positive"]["baseline_variance_trace_mean"]
        ),
        "positive_condition_direction_preserved": (
            aggregates["positive"]["gradient_cosine_to_ce_min"] >= 0.995
            and aggregates["positive"]["mean_bias_norm_max"] <= 0.10
        ),
        "positive_condition_ema_near_zero": (
            aggregates["positive"]["ema_drift_norm_max"] <= 0.12
        ),
        "positive_condition_priced": (
            aggregates["positive"]["priced_work_ratio_mean"] <= 0.80
        ),
        "drifted_ema_control_all_fail_direction": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "ce_direction" in result.fail_reason
                or "mean_bias" in result.fail_reason
                or "ema_drift" in result.fail_reason
            )
            for result in drifted
        ),
        "uncorrelated_ema_control_all_fail_variance": all(
            (not result.passed)
            and result.fail_reason is not None
            and ("variance_reduction" in result.fail_reason or "snr_gain" in result.fail_reason)
            for result in uncorrelated
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
        "PASS_R501_S77_ZVCS_GATE"
        if all(checks.values())
        else "FAIL_R501_S77_ZVCS_GATE"
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
            "Synthetic CPU zero-variance credit subtraction gate only; not "
            "heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S77_ZVCS_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_baseline_variance_trace_mean={positive['baseline_variance_trace_mean']:.6f} "
        f"positive_zvcs_variance_trace_mean={positive['zvcs_variance_trace_mean']:.6f} "
        f"positive_variance_ratio_mean={positive['variance_ratio_mean']:.6f} "
        f"positive_snr_gain_mean={positive['snr_gain_mean']:.6f} "
        f"positive_gradient_cosine_to_ce_min={positive['gradient_cosine_to_ce_min']:.6f} "
        f"positive_ema_drift_norm_max={positive['ema_drift_norm_max']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_token_efficiency_multiplier_mean={positive['projected_token_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
