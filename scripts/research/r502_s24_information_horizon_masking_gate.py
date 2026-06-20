#!/usr/bin/env python3
"""R502-S24 Information-Theoretic Horizon Masking gate.

This synthetic CPU gate checks whether a cheap forward proxy for future
information can mask low-information adjoint paths while preserving the dense
adjoint direction and reducing the priced backward horizon. Controls cover
expensive proxies, noisy proxies, long-context dependencies, aggressive
thresholds, and no-sparsity regimes.
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
    "expensive_proxy_fail",
    "proxy_noise_fail",
    "long_context_fail",
    "aggressive_threshold_fail",
    "no_sparsity_fail",
]


@dataclass
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str
    sequence_len: int
    state_dim: int
    retained_fraction: float
    mean_horizon_fraction: float
    dropped_energy_fraction: float
    proxy_false_negative_fraction: float
    relative_adjoint_error: float
    max_abs_adjoint_error: float
    descent_cosine: float
    proxy_cost_ratio: float
    priced_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=50224)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sequence-len", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=24)
    parser.add_argument("--threshold", type=float, default=0.055)
    parser.add_argument("--aggressive-threshold", type=float, default=0.28)
    parser.add_argument("--proxy-cost-ratio", type=float, default=0.04)
    parser.add_argument("--expensive-proxy-cost-ratio", type=float, default=0.88)
    parser.add_argument("--max-relative-adjoint-error", type=float, default=0.06)
    parser.add_argument("--min-descent-cosine", type=float, default=0.99)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.90)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def valid_lag_mask(sequence_len: int) -> np.ndarray:
    t = np.arange(sequence_len)[:, None]
    lag = np.arange(sequence_len)[None, :]
    return (t + lag) < sequence_len


def make_energy(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    sequence_len = args.sequence_len
    lag = np.arange(sequence_len)[None, :]
    tau = rng.uniform(8.0, 18.0, size=(sequence_len, 1))
    base = np.exp(-lag / tau)
    base *= rng.uniform(0.85, 1.15, size=(sequence_len, 1))
    mask = valid_lag_mask(sequence_len)
    base = np.where(mask, base, 0.0)
    proxy = base + rng.normal(scale=0.003, size=base.shape)
    threshold = args.threshold

    if condition == "proxy_noise_fail":
        proxy = rng.uniform(0.0, 0.11, size=base.shape)
        proxy = np.where(mask, proxy, 0.0)
    elif condition == "long_context_fail":
        long_band = ((lag >= 80) & (lag <= 112)).astype(np.float64)
        long_band = np.where(mask, long_band, 0.0)
        base = base + 0.42 * long_band
        proxy = np.where(long_band > 0.0, 0.01, proxy)
    elif condition == "aggressive_threshold_fail":
        threshold = args.aggressive_threshold
    elif condition == "no_sparsity_fail":
        base = np.where(mask, rng.uniform(0.42, 0.78, size=base.shape), 0.0)
        proxy = base + rng.normal(scale=0.002, size=base.shape)

    proxy = np.where(mask, np.clip(proxy, 0.0, None), 0.0)
    return base, proxy, threshold


def make_contributions(
    rng: np.random.Generator,
    energy: np.ndarray,
    state_dim: int,
) -> np.ndarray:
    directions = rng.normal(size=energy.shape + (state_dim,))
    directions /= np.maximum(np.linalg.norm(directions, axis=2, keepdims=True), 1e-12)
    return energy[:, :, None] * directions


def reduce_adjoint(contributions: np.ndarray, keep: np.ndarray) -> np.ndarray:
    return np.sum(contributions * keep[:, :, None], axis=1)


def horizon_fraction(keep: np.ndarray, valid: np.ndarray) -> float:
    retained = np.sum(keep, axis=1)
    available = np.sum(valid, axis=1)
    return float(np.mean(retained / np.maximum(available, 1)))


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    energy, proxy, threshold = make_energy(rng, args, condition)
    valid = valid_lag_mask(args.sequence_len)
    keep = (proxy >= threshold) & valid
    contributions = make_contributions(rng, energy, args.state_dim)
    full = reduce_adjoint(contributions, valid)
    candidate = reduce_adjoint(contributions, keep)
    delta = candidate - full
    full_norm = float(np.linalg.norm(full))
    candidate_norm = float(np.linalg.norm(candidate))
    rel_error = float(np.linalg.norm(delta) / max(full_norm, 1e-12))
    max_abs_error = float(np.max(np.abs(delta)))
    cosine = float(
        np.vdot(candidate.ravel(), full.ravel()).real
        / max(candidate_norm * full_norm, 1e-12)
    )
    retained_fraction = float(np.sum(keep) / max(np.sum(valid), 1))
    mean_horizon = horizon_fraction(keep, valid)
    dropped = valid & (~keep)
    dropped_energy = float(
        np.sum(contributions[dropped] ** 2) / max(np.sum(contributions[valid] ** 2), 1e-12)
    )
    true_relevant = (energy >= threshold) & valid
    false_negative = float(np.sum(true_relevant & (~keep)) / max(np.sum(true_relevant), 1))
    proxy_cost = (
        args.expensive_proxy_cost_ratio
        if condition == "expensive_proxy_fail"
        else args.proxy_cost_ratio
    )
    price = proxy_cost + retained_fraction
    speedup = 1.0 / max(price, 1e-12)

    failed: list[str] = []
    if not np.isfinite(
        [
            retained_fraction,
            mean_horizon,
            dropped_energy,
            false_negative,
            rel_error,
            max_abs_error,
            cosine,
            proxy_cost,
            price,
            speedup,
        ]
    ).all():
        failed.append("non_finite_metric")
    if rel_error > args.max_relative_adjoint_error:
        failed.append("adjoint_error")
    if cosine < args.min_descent_cosine:
        failed.append("descent_cosine")
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
        sequence_len=args.sequence_len,
        state_dim=args.state_dim,
        retained_fraction=retained_fraction,
        mean_horizon_fraction=mean_horizon,
        dropped_energy_fraction=dropped_energy,
        proxy_false_negative_fraction=false_negative,
        relative_adjoint_error=rel_error,
        max_abs_adjoint_error=max_abs_error,
        descent_cosine=cosine,
        proxy_cost_ratio=proxy_cost,
        priced_work_ratio=price,
        projected_speedup=speedup,
    )


NUMERIC_FIELDS = [
    "sequence_len",
    "state_dim",
    "retained_fraction",
    "mean_horizon_fraction",
    "dropped_energy_fraction",
    "proxy_false_negative_fraction",
    "relative_adjoint_error",
    "max_abs_adjoint_error",
    "descent_cosine",
    "proxy_cost_ratio",
    "priced_work_ratio",
    "projected_speedup",
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
        "positive_condition_adjoint_and_price": (
            aggregates["positive"]["relative_adjoint_error_max"] <= 0.06
            and aggregates["positive"]["descent_cosine_min"] >= 0.99
            and aggregates["positive"]["priced_work_ratio_max"] <= 0.90
        ),
        "expensive_proxy_control_all_fail_price": all(
            (not result.passed) and "priced_work" in result.fail_reason
            for result in by_condition["expensive_proxy_fail"]
        ),
        "proxy_noise_control_all_fail_error": all(
            (not result.passed)
            and result.proxy_false_negative_fraction > 0.0
            and (
                "adjoint_error" in result.fail_reason
                or "descent_cosine" in result.fail_reason
            )
            for result in by_condition["proxy_noise_fail"]
        ),
        "long_context_control_all_fail_error": all(
            (not result.passed)
            and result.proxy_false_negative_fraction > 0.0
            and (
                "adjoint_error" in result.fail_reason
                or "descent_cosine" in result.fail_reason
            )
            for result in by_condition["long_context_fail"]
        ),
        "aggressive_threshold_control_all_fail_error": all(
            (not result.passed)
            and (
                "adjoint_error" in result.fail_reason
                or "descent_cosine" in result.fail_reason
            )
            for result in by_condition["aggressive_threshold_fail"]
        ),
        "no_sparsity_control_all_fail_price": all(
            (not result.passed) and "priced_work" in result.fail_reason
            for result in by_condition["no_sparsity_fail"]
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
    decision = "PASS_R502_S24_ITHM_GATE" if all(checks.values()) else "FAIL_R502_S24_ITHM_GATE"
    payload = {
        "round": "R502",
        "canonical_id": "R502-S24",
        "aspect": "Speed",
        "finding": "Information-Theoretic Horizon Masking",
        "decision": decision,
        "disposition": "synthetic_positive_control_identified_not_training_speed_claim",
        "args": vars(args) | {"output": str(args.output) if args.output else None},
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU information-theoretic horizon masking gate only; "
            "not heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_S24_ITHM_GATE] "
        f"active=1 decision={decision} "
        f"disposition={payload['disposition']} "
        f"positive_retained_fraction_mean={positive['retained_fraction_mean']:.12f} "
        f"positive_relative_adjoint_error_max={positive['relative_adjoint_error_max']:.12f} "
        f"positive_descent_cosine_min={positive['descent_cosine_min']:.12f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.12f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.12f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
