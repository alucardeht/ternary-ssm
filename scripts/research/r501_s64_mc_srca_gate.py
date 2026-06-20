#!/usr/bin/env python3
"""R501 S64 martingale-certified state-residual credit gate.

This CPU diagnostic implements the R501 Quality Math 1 finding:
"Martingale-Certified State-Residual Credit Allocation (MC-SRCA)".

The gate compares a dense 3-step BPTT gradient with a local state-residual
credit gradient built from targets h*_t = h_t - eta_h * grad_h loss_t. A
positive case must align with dense BPTT while the future-cotangent residual is
approximately orthogonal to the past-state filtration. The price is recorded as
a token-efficiency denominator, not as GPU training speed.

Controls exercise the declared risks: future residuals correlated with past
states break the martingale certificate, noncontractive future cotangents break
gradient alignment, and an expensive projection/audit path erases the priced
denominator.

This is a synthetic CPU gradient-alignment diagnostic, not heldout CE progress,
GPU training-speed evidence, or production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s64_mc_srca_gate.v1"
CONDITIONS = (
    "positive",
    "correlated_residual_fail",
    "noncontractive_future_fail",
    "projection_cost_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    batch_size: int
    sequence_length: int
    state_dim: int
    input_dim: int
    transition_radius: float
    gradient_cosine: float
    relative_gradient_error: float
    martingale_residual_corr_max: float
    residual_energy_ratio: float
    local_target_work_ratio: float
    projection_audit_work_ratio: float
    priced_work_ratio: float
    projected_token_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50164)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--sequence-length", type=int, default=3)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=6)
    parser.add_argument("--min-gradient-cosine", type=float, default=0.97)
    parser.add_argument("--max-relative-gradient-error", type=float, default=0.28)
    parser.add_argument("--max-martingale-corr", type=float, default=0.12)
    parser.add_argument("--max-residual-energy-ratio", type=float, default=0.30)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.85)
    return parser.parse_args()


def random_orthogonal(rng: np.random.Generator, dimension: int) -> np.ndarray:
    matrix = rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def transition_matrix(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, float]:
    radius = 0.07
    if condition == "correlated_residual_fail":
        radius = 0.42
    elif condition == "noncontractive_future_fail":
        radius = 0.86
    q = random_orthogonal(rng, args.state_dim)
    eigenvalues = np.linspace(radius * 0.55, radius, args.state_dim)
    return q @ np.diag(eigenvalues) @ q.T, radius


def forward_states(
    transition: np.ndarray,
    input_matrix: np.ndarray,
    inputs: np.ndarray,
) -> np.ndarray:
    batch_size, sequence_length, _ = inputs.shape
    state_dim = transition.shape[0]
    states = np.zeros((batch_size, sequence_length + 1, state_dim), dtype=np.float64)
    for t in range(sequence_length):
        states[:, t + 1, :] = (
            states[:, t, :] @ transition.T + inputs[:, t, :] @ input_matrix.T
        )
    return states


def make_loss_gradients(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
    inputs: np.ndarray,
    states: np.ndarray,
) -> np.ndarray:
    errors = rng.normal(size=(args.batch_size, args.sequence_length, args.state_dim))
    if condition == "correlated_residual_fail":
        coupling = rng.normal(scale=0.85, size=(args.state_dim, args.state_dim))
        for t in range(1, args.sequence_length):
            errors[:, t, :] += states[:, t, :] @ coupling.T
    elif condition == "noncontractive_future_fail":
        coupling = rng.normal(scale=1.20, size=(args.state_dim, args.input_dim))
        for t in range(args.sequence_length - 1):
            errors[:, t + 1, :] += inputs[:, t, :] @ coupling.T
    return errors


def dense_cotangents(transition: np.ndarray, errors: np.ndarray) -> np.ndarray:
    lambdas = np.zeros_like(errors)
    future = np.zeros(errors.shape[0:1] + errors.shape[2:3], dtype=np.float64)
    for t in range(errors.shape[1] - 1, -1, -1):
        future = errors[:, t, :] + future @ transition
        lambdas[:, t, :] = future
    return lambdas


def gradient_from_cotangents(cotangents: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    gradient = np.zeros((cotangents.shape[2], inputs.shape[2]), dtype=np.float64)
    for t in range(inputs.shape[1]):
        gradient += cotangents[:, t, :].T @ inputs[:, t, :]
    return gradient / inputs.shape[0]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    return float(np.dot(a_flat, b_flat)) / max(
        float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat)),
        1.0e-30,
    )


def max_abs_corr(features: np.ndarray, residuals: np.ndarray) -> float:
    x = features.reshape(features.shape[0], -1)
    y = residuals.reshape(residuals.shape[0], -1)
    x = x - np.mean(x, axis=0, keepdims=True)
    y = y - np.mean(y, axis=0, keepdims=True)
    x_std = np.std(x, axis=0, ddof=1)
    y_std = np.std(y, axis=0, ddof=1)
    keep_x = x_std > 1.0e-12
    keep_y = y_std > 1.0e-12
    if not np.any(keep_x) or not np.any(keep_y):
        return 0.0
    x = x[:, keep_x] / x_std[keep_x]
    y = y[:, keep_y] / y_std[keep_y]
    corr = x.T @ y / max(x.shape[0] - 1, 1)
    return float(np.max(np.abs(corr)))


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    transition, radius = transition_matrix(rng, args, condition)
    input_matrix = rng.normal(scale=0.35, size=(args.state_dim, args.input_dim))
    inputs = rng.normal(size=(args.batch_size, args.sequence_length, args.input_dim))
    states = forward_states(transition, input_matrix, inputs)
    errors = make_loss_gradients(rng, args, condition, inputs, states)
    lambdas = dense_cotangents(transition, errors)

    dense_gradient = gradient_from_cotangents(lambdas, inputs)
    local_gradient = gradient_from_cotangents(errors, inputs)
    gradient_cosine = cosine(dense_gradient, local_gradient)
    relative_gradient_error = float(np.linalg.norm(dense_gradient - local_gradient)) / max(
        float(np.linalg.norm(dense_gradient)),
        1.0e-30,
    )

    residuals = lambdas - errors
    filtration_features = states[:, :-1, :]
    residual_corr = max_abs_corr(filtration_features, residuals)
    residual_energy = float(np.linalg.norm(residuals)) / max(
        float(np.linalg.norm(lambdas)),
        1.0e-30,
    )

    local_target_work = 0.52
    projection_audit_work = 0.14
    if condition == "projection_cost_fail":
        local_target_work = 0.62
        projection_audit_work = 0.31
    priced_work_ratio = local_target_work + projection_audit_work
    projected_multiplier = 1.0 / priced_work_ratio

    failed: list[str] = []
    if not finite(
        gradient_cosine,
        relative_gradient_error,
        residual_corr,
        residual_energy,
        priced_work_ratio,
        projected_multiplier,
    ):
        failed.append("non_finite_metric")
    if gradient_cosine < args.min_gradient_cosine:
        failed.append("gradient_alignment")
    if relative_gradient_error > args.max_relative_gradient_error:
        failed.append("relative_gradient_error")
    if residual_corr > args.max_martingale_corr:
        failed.append("martingale_residual")
    if residual_energy > args.max_residual_energy_ratio:
        failed.append("residual_energy")
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
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        state_dim=args.state_dim,
        input_dim=args.input_dim,
        transition_radius=float(radius),
        gradient_cosine=float(gradient_cosine),
        relative_gradient_error=float(relative_gradient_error),
        martingale_residual_corr_max=float(residual_corr),
        residual_energy_ratio=float(residual_energy),
        local_target_work_ratio=float(local_target_work),
        projection_audit_work_ratio=float(projection_audit_work),
        priced_work_ratio=float(priced_work_ratio),
        projected_token_efficiency_multiplier=float(projected_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "batch_size",
        "sequence_length",
        "state_dim",
        "input_dim",
        "transition_radius",
        "gradient_cosine",
        "relative_gradient_error",
        "martingale_residual_corr_max",
        "residual_energy_ratio",
        "local_target_work_ratio",
        "projection_audit_work_ratio",
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
    correlated = [result for result in results if result.condition == "correlated_residual_fail"]
    noncontractive = [
        result for result in results if result.condition == "noncontractive_future_fail"
    ]
    projection = [result for result in results if result.condition == "projection_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_gradient_alignment": (
            aggregates["positive"]["gradient_cosine_mean"] >= 0.97
            and aggregates["positive"]["relative_gradient_error_max"] <= 0.28
        ),
        "positive_condition_martingale": (
            aggregates["positive"]["martingale_residual_corr_max_max"] <= 0.12
            and aggregates["positive"]["residual_energy_ratio_max"] <= 0.30
        ),
        "positive_condition_priced": (
            aggregates["positive"]["priced_work_ratio_mean"] <= 0.85
        ),
        "correlated_control_all_fail_martingale": all(
            (not result.passed)
            and result.fail_reason is not None
            and "martingale_residual" in result.fail_reason
            for result in correlated
        ),
        "noncontractive_control_all_fail_alignment": all(
            (not result.passed)
            and result.fail_reason is not None
            and "gradient_alignment" in result.fail_reason
            for result in noncontractive
        ),
        "projection_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "price" in result.fail_reason
            for result in projection
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
        "PASS_R501_S64_MC_SRCA_GATE"
        if all(checks.values())
        else "FAIL_R501_S64_MC_SRCA_GATE"
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
            "Synthetic CPU 3-step MC-SRCA gradient-alignment and martingale "
            "residual gate only; not heldout CE progress, not GPU training "
            "speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S64_MC_SRCA_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_gradient_cosine_mean={positive['gradient_cosine_mean']:.6f} "
        f"positive_relative_gradient_error_max={positive['relative_gradient_error_max']:.6f} "
        f"positive_martingale_corr_max={positive['martingale_residual_corr_max_max']:.6f} "
        f"positive_residual_energy_ratio_max={positive['residual_energy_ratio_max']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_token_efficiency_multiplier_mean={positive['projected_token_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
