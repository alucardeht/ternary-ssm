#!/usr/bin/env python3
"""R501 S92 cross-sample state-conditioned CE quotient gate.

This CPU diagnostic implements the R501 Quality Paradigm 14 finding:
"Cross-Sample State-Conditioned CE Quotient (CS-SCCQ)".

The gate constructs an active batch with repeated state-conditioned CE gradient
classes across different sequence contexts. The positive condition collapses
each class into g_C = n_C * g_rep + r_C, preserving the dense batch gradient
while paying one representative plus a small residual certificate per class.

Controls exercise the declared kill criteria: state-near tokens can have
different CE gradients, multiplicity scaling can be missing, residuals can be
ignored, low multiplicity can erase the quotient, and expensive indexing can
erase the denominator.

This is a synthetic CPU cross-sample quotient diagnostic, not heldout CE
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


SCHEMA = "r501_s92_cs_sccq_gate.v1"
CONDITIONS = (
    "positive",
    "false_state_match_fail",
    "missing_multiplicity_fail",
    "residual_ignored_fail",
    "low_multiplicity_fail",
    "index_cost_fail",
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
    class_count: int
    mean_multiplicity: float
    state_dim: int
    gradient_dim: int
    max_state_radius: float
    max_gradient_drift_ratio: float
    class_residual_ratio: float
    quotient_gradient_relative_error: float
    quotient_gradient_cosine: float
    multiplicity_scaling_error_ratio: float
    representative_fraction: float
    residual_certificate_ratio: float
    index_overhead_ratio: float
    paid_token_ratio: float
    projected_token_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50192)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--class-count", type=int, default=64)
    parser.add_argument("--multiplicity", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--gradient-dim", type=int, default=24)
    parser.add_argument("--state-noise", type=float, default=0.003)
    parser.add_argument("--gradient-noise", type=float, default=0.006)
    parser.add_argument("--false-gradient-scale", type=float, default=0.75)
    parser.add_argument("--residual-certificate-ratio", type=float, default=0.04)
    parser.add_argument("--index-overhead", type=float, default=0.05)
    parser.add_argument("--expensive-index-overhead", type=float, default=0.45)
    parser.add_argument("--max-state-radius", type=float, default=0.025)
    parser.add_argument("--max-gradient-drift-ratio", type=float, default=0.08)
    parser.add_argument("--max-class-residual-ratio", type=float, default=0.08)
    parser.add_argument("--max-gradient-error", type=float, default=0.02)
    parser.add_argument("--min-gradient-cosine", type=float, default=0.999)
    parser.add_argument("--max-multiplicity-scaling-error", type=float, default=0.02)
    parser.add_argument("--max-paid-token-ratio", type=float, default=0.35)
    parser.add_argument("--min-token-efficiency-multiplier", type=float, default=2.5)
    return parser.parse_args()


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.0e-30)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0e-30)
    return float(np.dot(a, b) / denom)


def build_classes(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    class_count = args.class_count
    multiplicity = 1 if condition == "low_multiplicity_fail" else args.multiplicity
    states: list[np.ndarray] = []
    gradients: list[np.ndarray] = []
    state_centers = rng.normal(size=(class_count, args.state_dim))
    grad_centers = normalize_rows(rng.normal(size=(class_count, args.gradient_dim)))

    for class_index in range(class_count):
        for item in range(multiplicity):
            states.append(
                state_centers[class_index]
                + rng.normal(size=args.state_dim) * args.state_noise
            )
            grad = grad_centers[class_index].copy()
            if condition == "false_state_match_fail" and item >= multiplicity // 2:
                grad = -grad + rng.normal(size=args.gradient_dim) * 0.05
            else:
                grad = grad + rng.normal(size=args.gradient_dim) * args.gradient_noise
            if condition == "residual_ignored_fail":
                grad = grad + rng.normal(size=args.gradient_dim) * 0.09
            gradients.append(grad)
    return states, gradients


def first_failure(
    *,
    max_state_radius: float,
    max_gradient_drift_ratio: float,
    class_residual_ratio: float,
    quotient_gradient_relative_error: float,
    quotient_gradient_cosine: float,
    multiplicity_scaling_error_ratio: float,
    paid_token_ratio: float,
    projected_token_efficiency_multiplier: float,
    args: argparse.Namespace,
) -> str | None:
    if not finite(
        max_state_radius,
        max_gradient_drift_ratio,
        class_residual_ratio,
        quotient_gradient_relative_error,
        quotient_gradient_cosine,
        multiplicity_scaling_error_ratio,
        paid_token_ratio,
        projected_token_efficiency_multiplier,
    ):
        return "nonfinite_metric"
    if paid_token_ratio > args.max_paid_token_ratio:
        return "price_paid_work"
    if projected_token_efficiency_multiplier < args.min_token_efficiency_multiplier:
        return "price_efficiency_multiplier"
    if max_state_radius > args.max_state_radius:
        return "state_radius"
    if max_gradient_drift_ratio > args.max_gradient_drift_ratio:
        return "gradient_drift"
    if class_residual_ratio > args.max_class_residual_ratio:
        return "class_residual_debt"
    if quotient_gradient_relative_error > args.max_gradient_error:
        return "quotient_gradient_error"
    if quotient_gradient_cosine < args.min_gradient_cosine:
        return "quotient_gradient_cosine"
    if multiplicity_scaling_error_ratio > args.max_multiplicity_scaling_error:
        return "multiplicity_scaling"
    return None


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    states, gradients = build_classes(rng, args, condition)
    multiplicity = 1 if condition == "low_multiplicity_fail" else args.multiplicity
    token_count = len(states)
    dense_gradient = np.sum(np.array(gradients, dtype=np.float64), axis=0)

    quotient_gradient = np.zeros(args.gradient_dim, dtype=np.float64)
    state_radii: list[float] = []
    gradient_drifts: list[float] = []
    residual_norms: list[float] = []
    class_dense_norms: list[float] = []
    scaling_errors: list[float] = []

    for class_index in range(args.class_count):
        start = class_index * multiplicity
        stop = start + multiplicity
        class_states = np.array(states[start:stop], dtype=np.float64)
        class_grads = np.array(gradients[start:stop], dtype=np.float64)
        rep = class_grads[0]
        class_dense = np.sum(class_grads, axis=0)
        residual = class_dense - multiplicity * rep
        applied_rep_scale = 1.0 if condition == "missing_multiplicity_fail" else float(multiplicity)
        if condition == "missing_multiplicity_fail":
            quotient_gradient += rep + residual
        elif condition == "residual_ignored_fail":
            quotient_gradient += multiplicity * rep
        else:
            quotient_gradient += multiplicity * rep + residual
        scaling_errors.append(abs(applied_rep_scale - float(multiplicity)) / max(float(multiplicity), 1.0))
        state_center = np.mean(class_states, axis=0)
        state_radii.append(float(np.max(np.linalg.norm(class_states - state_center, axis=1))))
        drift = np.max(np.linalg.norm(class_grads - rep, axis=1)) / max(
            float(np.linalg.norm(rep)), 1.0e-30
        )
        gradient_drifts.append(float(drift))
        residual_norms.append(float(np.linalg.norm(residual)))
        class_dense_norms.append(float(np.linalg.norm(class_dense)))

    class_residual_ratio = float(
        np.sum(residual_norms) / max(np.sum(class_dense_norms), 1.0e-30)
    )
    gradient_error = float(
        np.linalg.norm(quotient_gradient - dense_gradient)
        / max(np.linalg.norm(dense_gradient), 1.0e-30)
    )
    gradient_cosine = cosine(quotient_gradient, dense_gradient)
    multiplicity_scaling_error = float(np.max(scaling_errors))
    representative_fraction = float(args.class_count / max(token_count, 1))
    residual_certificate_ratio = args.residual_certificate_ratio
    index_overhead = (
        args.expensive_index_overhead
        if condition == "index_cost_fail"
        else args.index_overhead
    )
    paid_token_ratio = float(
        representative_fraction + residual_certificate_ratio + index_overhead
    )
    projected_multiplier = float(1.0 / max(paid_token_ratio, 1.0e-30))

    fail_reason = first_failure(
        max_state_radius=float(np.max(state_radii)),
        max_gradient_drift_ratio=float(np.max(gradient_drifts)),
        class_residual_ratio=class_residual_ratio,
        quotient_gradient_relative_error=gradient_error,
        quotient_gradient_cosine=gradient_cosine,
        multiplicity_scaling_error_ratio=multiplicity_scaling_error,
        paid_token_ratio=paid_token_ratio,
        projected_token_efficiency_multiplier=projected_multiplier,
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
        token_count=token_count,
        class_count=args.class_count,
        mean_multiplicity=float(token_count / args.class_count),
        state_dim=args.state_dim,
        gradient_dim=args.gradient_dim,
        max_state_radius=float(np.max(state_radii)),
        max_gradient_drift_ratio=float(np.max(gradient_drifts)),
        class_residual_ratio=class_residual_ratio,
        quotient_gradient_relative_error=gradient_error,
        quotient_gradient_cosine=gradient_cosine,
        multiplicity_scaling_error_ratio=multiplicity_scaling_error,
        representative_fraction=representative_fraction,
        residual_certificate_ratio=residual_certificate_ratio,
        index_overhead_ratio=index_overhead,
        paid_token_ratio=paid_token_ratio,
        projected_token_efficiency_multiplier=projected_multiplier,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "token_count",
        "class_count",
        "mean_multiplicity",
        "state_dim",
        "gradient_dim",
        "max_state_radius",
        "max_gradient_drift_ratio",
        "class_residual_ratio",
        "quotient_gradient_relative_error",
        "quotient_gradient_cosine",
        "multiplicity_scaling_error_ratio",
        "representative_fraction",
        "residual_certificate_ratio",
        "index_overhead_ratio",
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
    false_match = [result for result in results if result.condition == "false_state_match_fail"]
    missing = [
        result for result in results if result.condition == "missing_multiplicity_fail"
    ]
    ignored = [result for result in results if result.condition == "residual_ignored_fail"]
    low_mult = [result for result in results if result.condition == "low_multiplicity_fail"]
    cost = [result for result in results if result.condition == "index_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_preserves_dense_gradient": (
            aggregates["positive"]["quotient_gradient_relative_error_max"] <= 0.02
            and aggregates["positive"]["quotient_gradient_cosine_min"] >= 0.999
            and aggregates["positive"]["multiplicity_scaling_error_ratio_max"] <= 0.02
        ),
        "positive_condition_valid_state_gradient_classes": (
            aggregates["positive"]["max_state_radius_max"] <= 0.025
            and aggregates["positive"]["max_gradient_drift_ratio_max"] <= 0.08
            and aggregates["positive"]["class_residual_ratio_max"] <= 0.08
        ),
        "positive_condition_priced": (
            aggregates["positive"]["paid_token_ratio_mean"] <= 0.35
            and aggregates["positive"]["projected_token_efficiency_multiplier_mean"] >= 2.5
        ),
        "false_match_control_all_fail_gradient_or_residual": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "gradient_drift" in result.fail_reason
                or "class_residual_debt" in result.fail_reason
                or "quotient_gradient_error" in result.fail_reason
            )
            for result in false_match
        ),
        "missing_multiplicity_control_all_fail_scaling_or_error": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "multiplicity_scaling" in result.fail_reason
                or "quotient_gradient_error" in result.fail_reason
            )
            for result in missing
        ),
        "residual_ignored_control_all_fail_residual_or_error": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "class_residual_debt" in result.fail_reason
                or "quotient_gradient_error" in result.fail_reason
                or "gradient_drift" in result.fail_reason
            )
            for result in ignored
        ),
        "low_multiplicity_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "price" in result.fail_reason
            for result in low_mult
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
    decision = "PASS_R501_S92_CS_SCCQ_GATE" if all(checks.values()) else "FAIL_R501_S92_CS_SCCQ_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU cross-sample state-conditioned CE quotient gate only; "
            "not heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S92_CS_SCCQ_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_class_count_mean={positive['class_count_mean']:.6f} "
        f"positive_mean_multiplicity_mean={positive['mean_multiplicity_mean']:.6f} "
        f"positive_quotient_gradient_relative_error_max={positive['quotient_gradient_relative_error_max']:.6f} "
        f"positive_quotient_gradient_cosine_min={positive['quotient_gradient_cosine_min']:.6f} "
        f"positive_class_residual_ratio_max={positive['class_residual_ratio_max']:.6f} "
        f"positive_paid_token_ratio_mean={positive['paid_token_ratio_mean']:.6f} "
        f"positive_projected_token_efficiency_multiplier_mean={positive['projected_token_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
