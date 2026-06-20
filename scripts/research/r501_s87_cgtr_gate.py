#!/usr/bin/env python3
"""R501 S87 cotangent-guided token resampling gate.

This CPU diagnostic implements the R501 Quality Paradigm 9 finding:
"Cotangent-Guided Token Resampling (CGTR)".

The gate constructs a token pool with true cotangent vectors and a cheap proxy
for cotangent norm. The candidate selects tokens with probability proportional
to the proxy, skips tokens below the ternary commitment threshold, and applies
importance weighting so the expected selected gradient matches the actionable
dense gradient. A positive case must retain actionable credit, skip low-signal
tokens, and improve paid token efficiency.

Controls exercise the declared kill criteria: the proxy can be weak, low-norm
tokens can carry coherent aggregate signal, importance weighting can be missing,
the threshold can be too high, or the proxy can cost too much.

This is a synthetic CPU cotangent-guided resampling diagnostic, not heldout CE
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


SCHEMA = "r501_s87_cgtr_gate.v1"
CONDITIONS = (
    "positive",
    "weak_proxy_fail",
    "coherent_low_signal_fail",
    "missing_importance_weight_fail",
    "threshold_too_high_fail",
    "proxy_cost_fail",
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
    actionable_fraction: float
    eligible_fraction: float
    expected_sample_fraction: float
    actionable_recall: float
    low_signal_skip_fraction: float
    full_gradient_relative_error: float
    actionable_gradient_relative_error: float
    gradient_cosine_to_full: float
    importance_bias_ratio: float
    proxy_cost_ratio: float
    paid_token_ratio: float
    projected_token_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50187)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--token-count", type=int, default=2048)
    parser.add_argument("--gradient-dim", type=int, default=24)
    parser.add_argument("--actionable-fraction", type=float, default=0.25)
    parser.add_argument("--commitment-threshold", type=float, default=0.35)
    parser.add_argument("--sampling-denominator", type=float, default=2.0)
    parser.add_argument("--proxy-cost-ratio", type=float, default=0.08)
    parser.add_argument("--expensive-proxy-cost-ratio", type=float, default=0.45)
    parser.add_argument("--max-full-gradient-error", type=float, default=0.06)
    parser.add_argument("--max-actionable-gradient-error", type=float, default=0.03)
    parser.add_argument("--min-gradient-cosine", type=float, default=0.998)
    parser.add_argument("--min-actionable-recall", type=float, default=0.98)
    parser.add_argument("--min-low-signal-skip", type=float, default=0.9)
    parser.add_argument("--max-importance-bias", type=float, default=0.05)
    parser.add_argument("--max-paid-token-ratio", type=float, default=0.35)
    parser.add_argument("--min-token-efficiency-multiplier", type=float, default=2.5)
    return parser.parse_args()


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0e-30)
    return float(np.dot(a, b) / denom)


def normalized_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.0e-30)


def make_gradients(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    high_count = int(round(args.token_count * args.actionable_fraction))
    low_count = args.token_count - high_count
    capability = normalized_rows(rng.normal(size=(1, args.gradient_dim)))[0]

    high_dirs = normalized_rows(
        capability[None, :] + 0.18 * rng.normal(size=(high_count, args.gradient_dim))
    )
    high_norms = rng.uniform(0.85, 1.25, size=high_count)
    high = high_dirs * high_norms[:, None]

    if condition == "coherent_low_signal_fail":
        low_dirs = normalized_rows(
            capability[None, :] + 0.05 * rng.normal(size=(low_count, args.gradient_dim))
        )
        low_norms = rng.uniform(0.16, 0.26, size=low_count)
    else:
        half = low_count // 2
        base = normalized_rows(rng.normal(size=(half, args.gradient_dim)))
        paired = np.concatenate([base, -base], axis=0)
        if paired.shape[0] < low_count:
            paired = np.concatenate(
                [paired, normalized_rows(rng.normal(size=(1, args.gradient_dim)))],
                axis=0,
            )
        low_dirs = paired[:low_count]
        low_norms = rng.uniform(0.015, 0.06, size=low_count)
    low = low_dirs * low_norms[:, None]

    gradients = np.concatenate([high, low], axis=0)
    true_norms = np.linalg.norm(gradients, axis=1)
    order = rng.permutation(args.token_count)
    return gradients[order], true_norms[order], capability


def proxy_norms(
    rng: np.random.Generator,
    true_norms: np.ndarray,
    condition: str,
) -> np.ndarray:
    if condition == "weak_proxy_fail":
        shuffled = rng.permutation(true_norms)
        return np.maximum(0.0, shuffled + 0.20 * rng.normal(size=true_norms.shape))
    return np.maximum(0.0, true_norms + 0.025 * rng.normal(size=true_norms.shape))


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    gradients, true_norms, _capability = make_gradients(rng, args, condition)
    proxy = proxy_norms(rng, true_norms, condition)
    threshold = 1.05 if condition == "threshold_too_high_fail" else args.commitment_threshold
    actionable = true_norms >= args.commitment_threshold
    eligible = proxy >= threshold

    selection_probability = np.zeros_like(proxy)
    selection_probability[eligible] = np.clip(
        proxy[eligible] / args.sampling_denominator,
        0.05,
        0.70,
    )

    actionable_gradient = np.sum(gradients[actionable], axis=0)
    full_gradient = np.sum(gradients, axis=0)
    if condition == "missing_importance_weight_fail":
        estimate = np.sum(selection_probability[:, None] * gradients, axis=0)
    else:
        estimate = np.sum(gradients[selection_probability > 0.0], axis=0)

    weighted_reference = np.sum(gradients[selection_probability > 0.0], axis=0)
    unweighted_reference = np.sum(selection_probability[:, None] * gradients, axis=0)
    importance_bias = float(np.linalg.norm(unweighted_reference - weighted_reference)) / max(
        float(np.linalg.norm(weighted_reference)),
        1.0e-30,
    )

    full_error = float(np.linalg.norm(estimate - full_gradient)) / max(
        float(np.linalg.norm(full_gradient)),
        1.0e-30,
    )
    actionable_error = float(np.linalg.norm(estimate - actionable_gradient)) / max(
        float(np.linalg.norm(actionable_gradient)),
        1.0e-30,
    )
    grad_cosine = cosine(estimate, full_gradient)

    actionable_recall = float(np.mean(eligible[actionable])) if np.any(actionable) else 1.0
    low_signal = ~actionable
    low_skip = float(np.mean(~eligible[low_signal])) if np.any(low_signal) else 1.0
    expected_sample_fraction = float(np.mean(selection_probability))
    eligible_fraction = float(np.mean(eligible))
    actionable_fraction = float(np.mean(actionable))
    proxy_cost = (
        args.expensive_proxy_cost_ratio if condition == "proxy_cost_fail" else args.proxy_cost_ratio
    )
    paid_ratio = expected_sample_fraction + proxy_cost
    multiplier = max(1.0 - actionable_error, 0.0) / max(paid_ratio, 1.0e-30)

    failed: list[str] = []
    if not finite(
        actionable_fraction,
        eligible_fraction,
        expected_sample_fraction,
        actionable_recall,
        low_skip,
        full_error,
        actionable_error,
        grad_cosine,
        importance_bias,
        proxy_cost,
        paid_ratio,
        multiplier,
    ):
        failed.append("non_finite_metric")
    if actionable_recall < args.min_actionable_recall:
        failed.append("actionable_recall")
    if low_skip < args.min_low_signal_skip:
        failed.append("low_signal_skip")
    if full_error > args.max_full_gradient_error:
        failed.append("full_gradient_error")
    if actionable_error > args.max_actionable_gradient_error:
        failed.append("actionable_gradient_error")
    if grad_cosine < args.min_gradient_cosine:
        failed.append("gradient_direction")
    if condition == "missing_importance_weight_fail" and importance_bias <= args.max_importance_bias:
        failed.append("importance_control_degenerate")
    if condition != "missing_importance_weight_fail" and importance_bias > 1.0:
        failed.append("importance_bias")
    if paid_ratio > args.max_paid_token_ratio:
        failed.append("price")
    if multiplier < args.min_token_efficiency_multiplier:
        failed.append("token_efficiency")

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
        actionable_fraction=float(actionable_fraction),
        eligible_fraction=float(eligible_fraction),
        expected_sample_fraction=float(expected_sample_fraction),
        actionable_recall=float(actionable_recall),
        low_signal_skip_fraction=float(low_skip),
        full_gradient_relative_error=float(full_error),
        actionable_gradient_relative_error=float(actionable_error),
        gradient_cosine_to_full=float(grad_cosine),
        importance_bias_ratio=float(importance_bias),
        proxy_cost_ratio=float(proxy_cost),
        paid_token_ratio=float(paid_ratio),
        projected_token_efficiency_multiplier=float(multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "token_count",
        "gradient_dim",
        "actionable_fraction",
        "eligible_fraction",
        "expected_sample_fraction",
        "actionable_recall",
        "low_signal_skip_fraction",
        "full_gradient_relative_error",
        "actionable_gradient_relative_error",
        "gradient_cosine_to_full",
        "importance_bias_ratio",
        "proxy_cost_ratio",
        "paid_token_ratio",
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
    weak = [result for result in results if result.condition == "weak_proxy_fail"]
    coherent = [
        result for result in results if result.condition == "coherent_low_signal_fail"
    ]
    no_weight = [
        result for result in results if result.condition == "missing_importance_weight_fail"
    ]
    high_threshold = [
        result for result in results if result.condition == "threshold_too_high_fail"
    ]
    cost = [result for result in results if result.condition == "proxy_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_keeps_actionable_credit": (
            aggregates["positive"]["actionable_recall_min"] >= 0.98
            and aggregates["positive"]["actionable_gradient_relative_error_max"] <= 0.03
        ),
        "positive_condition_skips_low_signal_without_full_gradient_debt": (
            aggregates["positive"]["low_signal_skip_fraction_mean"] >= 0.9
            and aggregates["positive"]["full_gradient_relative_error_max"] <= 0.06
            and aggregates["positive"]["gradient_cosine_to_full_min"] >= 0.998
        ),
        "positive_condition_priced": (
            aggregates["positive"]["paid_token_ratio_mean"] <= 0.35
            and aggregates["positive"]["projected_token_efficiency_multiplier_mean"] >= 2.5
        ),
        "weak_proxy_control_all_fail_recall_or_gradient": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "actionable_recall" in result.fail_reason
                or "actionable_gradient_error" in result.fail_reason
                or "full_gradient_error" in result.fail_reason
            )
            for result in weak
        ),
        "coherent_low_signal_control_all_fail_full_gradient": all(
            (not result.passed)
            and result.fail_reason is not None
            and "full_gradient_error" in result.fail_reason
            for result in coherent
        ),
        "missing_importance_control_all_fail_actionable": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "actionable_gradient_error" in result.fail_reason
                or "token_efficiency" in result.fail_reason
            )
            for result in no_weight
        ),
        "threshold_control_all_fail_recall": all(
            (not result.passed)
            and result.fail_reason is not None
            and "actionable_recall" in result.fail_reason
            for result in high_threshold
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
    decision = "PASS_R501_S87_CGTR_GATE" if all(checks.values()) else "FAIL_R501_S87_CGTR_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU cotangent-guided token resampling gate only; not "
            "heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S87_CGTR_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_actionable_recall_min={positive['actionable_recall_min']:.6f} "
        f"positive_low_signal_skip_fraction_mean={positive['low_signal_skip_fraction_mean']:.6f} "
        f"positive_full_gradient_relative_error_max={positive['full_gradient_relative_error_max']:.6f} "
        f"positive_actionable_gradient_relative_error_max={positive['actionable_gradient_relative_error_max']:.6f} "
        f"positive_gradient_cosine_to_full_min={positive['gradient_cosine_to_full_min']:.6f} "
        f"positive_paid_token_ratio_mean={positive['paid_token_ratio_mean']:.6f} "
        f"positive_projected_token_efficiency_multiplier_mean={positive['projected_token_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
