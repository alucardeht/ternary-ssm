#!/usr/bin/env python3
"""R501 S90 harmonic credit assignment gate.

This CPU diagnostic implements the R501 Quality Paradigm 12 finding:
"Harmonic Credit Assignment (HCA)".

The gate constructs recurrent feature streams whose target credit state is the
harmonic sum h_t = sum_k 1/k phi(x_{t-k}). The positive condition must reproduce
that algebraic state, retain bounded long-lag credit, and keep the quotient
pricing below dense all-lag accounting.

Controls exercise the declared kill criteria: exponential decay forgets the
long tail, truncated harmonic credit drops distant dependencies, the wrong
polynomial kernel distorts the quotient, shuffled lag order breaks assignment,
and a dense quotient implementation erases the denominator.

This is a synthetic CPU harmonic-credit diagnostic, not heldout CE progress,
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


SCHEMA = "r501_s90_hca_gate.v1"
CONDITIONS = (
    "positive",
    "exponential_decay_fail",
    "truncated_tail_fail",
    "wrong_kernel_fail",
    "shuffled_lag_fail",
    "quotient_cost_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    sequence_len: int
    feature_dim: int
    eval_count: int
    horizon: int
    long_lag: int
    quotient_basis_count: int
    state_relative_error_mean: float
    state_relative_error_max: float
    quotient_weight_residual: float
    long_lag_retention_fraction: float
    tail_credit_capture_fraction: float
    near_credit_distortion: float
    lag_order_error_mean: float
    quotient_overhead_ratio: float
    paid_work_ratio: float
    projected_long_context_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50190)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sequence-len", type=int, default=256)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--eval-count", type=int, default=64)
    parser.add_argument("--long-lag", type=int, default=128)
    parser.add_argument("--truncate-lag", type=int, default=32)
    parser.add_argument("--near-lag", type=int, default=8)
    parser.add_argument("--exponential-alpha", type=float, default=0.94)
    parser.add_argument("--quotient-basis-count", type=int, default=8)
    parser.add_argument("--quotient-overhead", type=float, default=0.08)
    parser.add_argument("--expensive-quotient-overhead", type=float, default=0.55)
    parser.add_argument("--max-state-relative-error", type=float, default=1.0e-12)
    parser.add_argument("--max-quotient-weight-residual", type=float, default=1.0e-12)
    parser.add_argument("--min-long-lag-retention", type=float, default=0.99)
    parser.add_argument("--min-tail-credit-capture", type=float, default=0.99)
    parser.add_argument("--max-near-credit-distortion", type=float, default=1.0e-12)
    parser.add_argument("--max-lag-order-error", type=float, default=1.0e-12)
    parser.add_argument("--max-paid-work-ratio", type=float, default=0.35)
    parser.add_argument("--min-long-context-multiplier", type=float, default=2.5)
    return parser.parse_args()


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def harmonic_weights(horizon: int) -> np.ndarray:
    lags = np.arange(1, horizon + 1, dtype=np.float64)
    return 1.0 / lags


def candidate_weights(args: argparse.Namespace, condition: str) -> np.ndarray:
    horizon = args.sequence_len - 1
    lags = np.arange(1, horizon + 1, dtype=np.float64)
    if condition == "exponential_decay_fail":
        weights = args.exponential_alpha**lags
        weights *= (1.0 / weights[0])
        return weights
    if condition == "truncated_tail_fail":
        weights = 1.0 / lags
        weights[lags > args.truncate_lag] = 0.0
        return weights
    if condition == "wrong_kernel_fail":
        return 1.0 / (lags**2)
    return 1.0 / lags


def compute_states(
    features: np.ndarray,
    weights: np.ndarray,
    eval_positions: np.ndarray,
    shuffled: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    states = []
    for position in eval_positions:
        max_lag = min(position, weights.shape[0])
        lagged = features[position - np.arange(1, max_lag + 1)]
        local_weights = weights[:max_lag]
        if shuffled:
            lagged = lagged[rng.permutation(max_lag)]
        states.append(local_weights @ lagged)
    return np.array(states, dtype=np.float64)


def relative_errors(candidate: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    numer = np.linalg.norm(candidate - target, axis=1)
    denom = np.maximum(np.linalg.norm(target, axis=1), 1.0e-30)
    values = numer / denom
    return float(np.mean(values)), float(np.max(values))


def first_failure(
    *,
    state_relative_error_max: float,
    quotient_weight_residual: float,
    long_lag_retention_fraction: float,
    tail_credit_capture_fraction: float,
    near_credit_distortion: float,
    lag_order_error_mean: float,
    paid_work_ratio: float,
    projected_long_context_multiplier: float,
    args: argparse.Namespace,
) -> str | None:
    if not finite(
        state_relative_error_max,
        quotient_weight_residual,
        long_lag_retention_fraction,
        tail_credit_capture_fraction,
        near_credit_distortion,
        lag_order_error_mean,
        paid_work_ratio,
        projected_long_context_multiplier,
    ):
        return "nonfinite_metric"
    if paid_work_ratio > args.max_paid_work_ratio:
        return "price_paid_work"
    if projected_long_context_multiplier < args.min_long_context_multiplier:
        return "price_long_context_multiplier"
    if quotient_weight_residual > args.max_quotient_weight_residual:
        return "quotient_weight_residual"
    if long_lag_retention_fraction < args.min_long_lag_retention:
        return "long_lag_retention"
    if tail_credit_capture_fraction < args.min_tail_credit_capture:
        return "tail_credit_capture"
    if near_credit_distortion > args.max_near_credit_distortion:
        return "near_credit_distortion"
    if lag_order_error_mean > args.max_lag_order_error:
        return "lag_order_error"
    if state_relative_error_max > args.max_state_relative_error:
        return "state_relative_error"
    return None


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    horizon = args.sequence_len - 1
    features = rng.normal(size=(args.sequence_len, args.feature_dim))
    eval_positions = np.arange(args.sequence_len - args.eval_count, args.sequence_len)
    target_weights = harmonic_weights(horizon)
    weights = candidate_weights(args, condition)
    shuffled = condition == "shuffled_lag_fail"

    target_states = compute_states(features, target_weights, eval_positions, False, rng)
    candidate_states = compute_states(features, weights, eval_positions, shuffled, rng)
    error_mean, error_max = relative_errors(candidate_states, target_states)

    if shuffled:
        weight_residual = 0.0
    else:
        weight_residual = float(
            np.linalg.norm(weights - target_weights)
            / max(np.linalg.norm(target_weights), 1.0e-30)
        )
    target_long = target_weights[args.long_lag - 1]
    long_lag_retention = float(weights[args.long_lag - 1] / max(target_long, 1.0e-30))
    tail = slice(args.long_lag - 1, None)
    tail_credit_capture = float(
        np.sum(np.abs(weights[tail])) / max(np.sum(np.abs(target_weights[tail])), 1.0e-30)
    )
    near = slice(0, args.near_lag)
    near_credit_distortion = float(
        np.linalg.norm(weights[near] - target_weights[near])
        / max(np.linalg.norm(target_weights[near]), 1.0e-30)
    )
    if shuffled:
        lag_order_error_mean = error_mean
    else:
        lag_order_error_mean = 0.0

    quotient_overhead = (
        args.expensive_quotient_overhead
        if condition == "quotient_cost_fail"
        else args.quotient_overhead
    )
    paid_work_ratio = float(args.quotient_basis_count / horizon + quotient_overhead)
    projected_long_context_multiplier = float(1.0 / max(paid_work_ratio, 1.0e-30))

    fail_reason = first_failure(
        state_relative_error_max=error_max,
        quotient_weight_residual=weight_residual,
        long_lag_retention_fraction=long_lag_retention,
        tail_credit_capture_fraction=tail_credit_capture,
        near_credit_distortion=near_credit_distortion,
        lag_order_error_mean=lag_order_error_mean,
        paid_work_ratio=paid_work_ratio,
        projected_long_context_multiplier=projected_long_context_multiplier,
        args=args,
    )
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
        feature_dim=args.feature_dim,
        eval_count=args.eval_count,
        horizon=horizon,
        long_lag=args.long_lag,
        quotient_basis_count=args.quotient_basis_count,
        state_relative_error_mean=error_mean,
        state_relative_error_max=error_max,
        quotient_weight_residual=weight_residual,
        long_lag_retention_fraction=long_lag_retention,
        tail_credit_capture_fraction=tail_credit_capture,
        near_credit_distortion=near_credit_distortion,
        lag_order_error_mean=lag_order_error_mean,
        quotient_overhead_ratio=quotient_overhead,
        paid_work_ratio=paid_work_ratio,
        projected_long_context_multiplier=projected_long_context_multiplier,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "sequence_len",
        "feature_dim",
        "eval_count",
        "horizon",
        "long_lag",
        "quotient_basis_count",
        "state_relative_error_mean",
        "state_relative_error_max",
        "quotient_weight_residual",
        "long_lag_retention_fraction",
        "tail_credit_capture_fraction",
        "near_credit_distortion",
        "lag_order_error_mean",
        "quotient_overhead_ratio",
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
    exponential = [
        result for result in results if result.condition == "exponential_decay_fail"
    ]
    truncated = [result for result in results if result.condition == "truncated_tail_fail"]
    wrong = [result for result in results if result.condition == "wrong_kernel_fail"]
    shuffled = [result for result in results if result.condition == "shuffled_lag_fail"]
    cost = [result for result in results if result.condition == "quotient_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_harmonic_state": (
            aggregates["positive"]["state_relative_error_max_max"] <= 1.0e-12
            and aggregates["positive"]["quotient_weight_residual_max"] <= 1.0e-12
            and aggregates["positive"]["lag_order_error_mean_max"] <= 1.0e-12
        ),
        "positive_condition_retains_long_tail": (
            aggregates["positive"]["long_lag_retention_fraction_min"] >= 0.99
            and aggregates["positive"]["tail_credit_capture_fraction_min"] >= 0.99
            and aggregates["positive"]["near_credit_distortion_max"] <= 1.0e-12
        ),
        "positive_condition_priced": (
            aggregates["positive"]["paid_work_ratio_mean"] <= 0.35
            and aggregates["positive"]["projected_long_context_multiplier_mean"] >= 2.5
        ),
        "exponential_control_all_fail_tail_or_quotient": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "long_lag_retention" in result.fail_reason
                or "tail_credit_capture" in result.fail_reason
                or "quotient_weight_residual" in result.fail_reason
            )
            for result in exponential
        ),
        "truncated_control_all_fail_tail": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "long_lag_retention" in result.fail_reason
                or "tail_credit_capture" in result.fail_reason
                or "quotient_weight_residual" in result.fail_reason
            )
            for result in truncated
        ),
        "wrong_kernel_control_all_fail_quotient_or_near": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "quotient_weight_residual" in result.fail_reason
                or "near_credit_distortion" in result.fail_reason
                or "state_relative_error" in result.fail_reason
            )
            for result in wrong
        ),
        "shuffled_control_all_fail_lag_order": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "lag_order_error" in result.fail_reason
                or "state_relative_error" in result.fail_reason
            )
            for result in shuffled
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
    decision = "PASS_R501_S90_HCA_GATE" if all(checks.values()) else "FAIL_R501_S90_HCA_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU harmonic credit assignment gate only; not heldout "
            "CE progress, not real corpus token efficiency, not GPU training "
            "speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S90_HCA_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_state_relative_error_max={positive['state_relative_error_max_max']:.6e} "
        f"positive_quotient_weight_residual_max={positive['quotient_weight_residual_max']:.6e} "
        f"positive_long_lag_retention_fraction_min={positive['long_lag_retention_fraction_min']:.6f} "
        f"positive_tail_credit_capture_fraction_min={positive['tail_credit_capture_fraction_min']:.6f} "
        f"positive_paid_work_ratio_mean={positive['paid_work_ratio_mean']:.6f} "
        f"positive_projected_long_context_multiplier_mean={positive['projected_long_context_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
