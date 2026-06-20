#!/usr/bin/env python3
"""R501 S86 spectral normalization of ternary transitions gate.

This CPU diagnostic implements the R501 Quality Paradigm 8 finding:
"Spectral Normalization of Ternary Transitions (SNTT)".

The gate constructs scaled ternary-sign transition operators and certifies that
the current transition satisfies sigma_max(A) <= 1 - epsilon while retaining a
non-vanishing long-horizon signal. A positive case uses a signed-permutation
ternary pattern scaled just below one, so cotangent and state signals remain
measurable over the horizon without explosive amplification.

Controls exercise the declared kill criteria: unnormalized transitions can
explode, over-contractive transitions can pass an upper bound while destroying
long context, non-normal operators can have stable eigenvalues but unsafe
singular transients, certificates can be stale, or certification can cost too
much.

This is a synthetic CPU spectral-normalization diagnostic, not heldout CE
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


SCHEMA = "r501_s86_sntt_gate.v1"
CONDITIONS = (
    "positive",
    "unnormalized_explosion_fail",
    "overcontractive_fail",
    "nonnormal_transient_fail",
    "stale_certificate_fail",
    "certificate_cost_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    state_dim: int
    horizon: int
    epsilon: float
    spectral_norm: float
    min_singular_value: float
    spectral_bound_margin: float
    long_horizon_signal_ratio_mean: float
    long_horizon_signal_ratio_min: float
    long_horizon_signal_ratio_max: float
    transient_amplification_max: float
    ternary_pattern_violation_count: int
    certificate_overhead_ratio: float
    paid_work_ratio: float
    projected_long_context_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50186)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=0.002)
    parser.add_argument("--certificate-overhead", type=float, default=0.08)
    parser.add_argument("--expensive-certificate-overhead", type=float, default=0.75)
    parser.add_argument("--min-signal-ratio", type=float, default=0.65)
    parser.add_argument("--max-signal-ratio", type=float, default=1.05)
    parser.add_argument("--min-singular-value", type=float, default=0.99)
    parser.add_argument("--max-transient-amplification", type=float, default=1.05)
    parser.add_argument("--max-paid-work-ratio", type=float, default=0.35)
    parser.add_argument("--min-long-context-multiplier", type=float, default=2.0)
    return parser.parse_args()


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def signed_permutation(rng: np.random.Generator, dim: int, scale: float) -> np.ndarray:
    perm = rng.permutation(dim)
    signs = rng.choice(np.array([-1.0, 1.0]), size=dim)
    matrix = np.zeros((dim, dim), dtype=np.float64)
    matrix[np.arange(dim), perm] = signs
    return scale * matrix


def nonnormal_transition(dim: int) -> np.ndarray:
    matrix = 0.90 * np.eye(dim, dtype=np.float64)
    for idx in range(dim - 1):
        matrix[idx, idx + 1] = 0.55
    return matrix


def make_transition(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    target_scale = 1.0 - args.epsilon
    if condition == "unnormalized_explosion_fail":
        current = signed_permutation(rng, args.state_dim, 1.035)
        certified = current
    elif condition == "overcontractive_fail":
        current = signed_permutation(rng, args.state_dim, 0.94)
        certified = current
    elif condition == "nonnormal_transient_fail":
        current = nonnormal_transition(args.state_dim)
        certified = current
    elif condition == "stale_certificate_fail":
        certified = signed_permutation(rng, args.state_dim, target_scale)
        current = 1.035 * np.sign(certified)
    else:
        current = signed_permutation(rng, args.state_dim, target_scale)
        certified = current
    return current, certified


def signal_ratios(
    rng: np.random.Generator,
    matrix: np.ndarray,
    horizon: int,
    sample_count: int,
) -> tuple[float, float, float, float]:
    values = rng.normal(size=(sample_count, matrix.shape[0]))
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.0e-30)

    max_transient = 0.0
    current = values.copy()
    for _ in range(horizon):
        current = current @ matrix.T
        ratios = np.linalg.norm(current, axis=1)
        max_transient = max(max_transient, float(np.max(ratios)))

    final_ratios = np.linalg.norm(current, axis=1)
    return (
        float(np.mean(final_ratios)),
        float(np.min(final_ratios)),
        float(np.max(final_ratios)),
        max_transient,
    )


def ternary_pattern_violations(matrix: np.ndarray) -> int:
    abs_values = np.abs(matrix)
    nonzero = abs_values > 1.0e-12
    if not np.any(nonzero):
        return 0
    scale = float(np.median(abs_values[nonzero]))
    scaled = abs_values[nonzero] / max(scale, 1.0e-30)
    return int(np.sum(np.abs(scaled - 1.0) > 1.0e-9))


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    current, certified = make_transition(rng, args, condition)
    singular_values = np.linalg.svd(current, compute_uv=False)
    spectral_norm = float(np.max(singular_values))
    min_singular = float(np.min(singular_values))
    bound = 1.0 - args.epsilon
    margin = bound - spectral_norm
    ratio_mean, ratio_min, ratio_max, transient_max = signal_ratios(
        rng,
        current,
        args.horizon,
        args.sample_count,
    )
    violations = ternary_pattern_violations(current)

    certificate_overhead = (
        args.expensive_certificate_overhead
        if condition == "certificate_cost_fail"
        else args.certificate_overhead
    )
    paid_ratio = (1.0 / args.horizon) + certificate_overhead
    projected_multiplier = ratio_mean / max(paid_ratio, 1.0e-30)

    certified_singular = np.linalg.svd(certified, compute_uv=False)
    stale_gap = abs(float(np.max(certified_singular)) - spectral_norm)

    failed: list[str] = []
    if not finite(
        spectral_norm,
        min_singular,
        margin,
        ratio_mean,
        ratio_min,
        ratio_max,
        transient_max,
        certificate_overhead,
        paid_ratio,
        projected_multiplier,
        stale_gap,
    ):
        failed.append("non_finite_metric")
    if margin < -1.0e-9:
        failed.append("spectral_bound")
    if min_singular < args.min_singular_value:
        failed.append("singular_floor")
    if ratio_min < args.min_signal_ratio:
        failed.append("long_horizon_vanish")
    if ratio_max > args.max_signal_ratio:
        failed.append("long_horizon_explosion")
    if transient_max > args.max_transient_amplification:
        failed.append("transient_amplification")
    if violations != 0:
        failed.append("ternary_pattern")
    if stale_gap > 1.0e-6:
        failed.append("stale_certificate")
    if paid_ratio > args.max_paid_work_ratio:
        failed.append("price")
    if projected_multiplier < args.min_long_context_multiplier:
        failed.append("long_context_efficiency")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed) if failed else None,
        state_dim=args.state_dim,
        horizon=args.horizon,
        epsilon=args.epsilon,
        spectral_norm=float(spectral_norm),
        min_singular_value=float(min_singular),
        spectral_bound_margin=float(margin),
        long_horizon_signal_ratio_mean=float(ratio_mean),
        long_horizon_signal_ratio_min=float(ratio_min),
        long_horizon_signal_ratio_max=float(ratio_max),
        transient_amplification_max=float(transient_max),
        ternary_pattern_violation_count=violations,
        certificate_overhead_ratio=float(certificate_overhead),
        paid_work_ratio=float(paid_ratio),
        projected_long_context_multiplier=float(projected_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "state_dim",
        "horizon",
        "epsilon",
        "spectral_norm",
        "min_singular_value",
        "spectral_bound_margin",
        "long_horizon_signal_ratio_mean",
        "long_horizon_signal_ratio_min",
        "long_horizon_signal_ratio_max",
        "transient_amplification_max",
        "ternary_pattern_violation_count",
        "certificate_overhead_ratio",
        "paid_work_ratio",
        "projected_long_context_multiplier",
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
    explosion = [
        result for result in results if result.condition == "unnormalized_explosion_fail"
    ]
    overcontractive = [
        result for result in results if result.condition == "overcontractive_fail"
    ]
    nonnormal = [
        result for result in results if result.condition == "nonnormal_transient_fail"
    ]
    stale = [result for result in results if result.condition == "stale_certificate_fail"]
    cost = [result for result in results if result.condition == "certificate_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_satisfies_spectral_bound": (
            aggregates["positive"]["spectral_bound_margin_min"] >= -1.0e-9
            and aggregates["positive"]["spectral_norm_max"] <= 1.0
        ),
        "positive_condition_preserves_horizon_signal": (
            aggregates["positive"]["min_singular_value_min"] >= 0.99
            and aggregates["positive"]["long_horizon_signal_ratio_min_min"] >= 0.65
            and aggregates["positive"]["long_horizon_signal_ratio_max_max"] <= 1.05
        ),
        "positive_condition_priced": (
            aggregates["positive"]["paid_work_ratio_mean"] <= 0.35
            and aggregates["positive"]["projected_long_context_multiplier_mean"] >= 2.0
        ),
        "explosion_control_all_fail_bound_or_explosion": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "spectral_bound" in result.fail_reason
                or "long_horizon_explosion" in result.fail_reason
            )
            for result in explosion
        ),
        "overcontractive_control_all_fail_vanish": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "singular_floor" in result.fail_reason
                or "long_horizon_vanish" in result.fail_reason
            )
            for result in overcontractive
        ),
        "nonnormal_control_all_fail_transient_or_bound": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "transient_amplification" in result.fail_reason
                or "spectral_bound" in result.fail_reason
            )
            for result in nonnormal
        ),
        "stale_control_all_fail_stale_or_bound": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "stale_certificate" in result.fail_reason
                or "spectral_bound" in result.fail_reason
            )
            for result in stale
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
    decision = "PASS_R501_S86_SNTT_GATE" if all(checks.values()) else "FAIL_R501_S86_SNTT_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU spectral normalization of ternary transitions gate "
            "only; not heldout CE progress, not real corpus token efficiency, "
            "not GPU training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S86_SNTT_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_spectral_norm_max={positive['spectral_norm_max']:.6f} "
        f"positive_spectral_bound_margin_min={positive['spectral_bound_margin_min']:.6f} "
        f"positive_min_singular_value_min={positive['min_singular_value_min']:.6f} "
        f"positive_long_horizon_signal_ratio_min={positive['long_horizon_signal_ratio_min_min']:.6f} "
        f"positive_long_horizon_signal_ratio_mean={positive['long_horizon_signal_ratio_mean_mean']:.6f} "
        f"positive_transient_amplification_max={positive['transient_amplification_max_max']:.6f} "
        f"positive_paid_work_ratio_mean={positive['paid_work_ratio_mean']:.6f} "
        f"positive_projected_long_context_multiplier_mean={positive['projected_long_context_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
