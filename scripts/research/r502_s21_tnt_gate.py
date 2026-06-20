#!/usr/bin/env python3
"""R502-S21 Ternary Operator Norm Truncation gate.

This synthetic CPU gate checks whether an analytic ternary operator-norm bound
can truncate BPTT horizons while keeping the omitted adjoint tail below the
ternary flip threshold. Controls cover near-unit spectral radius,
underestimated rho, non-normal transient growth, miscalibrated flip threshold,
and audit cost.
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
    "near_unit_radius_fail",
    "underestimated_rho_fail",
    "non_normal_fail",
    "epsilon_miscalibration_fail",
    "audit_cost_fail",
]


@dataclass
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str
    horizon_len: int
    state_dim: int
    true_spectral_radius: float
    declared_rho_bound: float
    operator_norm: float
    truncation_horizon: int
    horizon_ratio: float
    tail_error_norm: float
    tail_error_to_flip_ratio: float
    certified_bound_at_horizon: float
    bound_violation_ratio: float
    priced_work_ratio: float
    audit_cost_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=50221)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--horizon-len", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--flip-threshold", type=float, default=0.02)
    parser.add_argument("--miscalibrated-flip-threshold", type=float, default=0.50)
    parser.add_argument("--audit-cost-ratio", type=float, default=0.04)
    parser.add_argument("--expensive-audit-cost-ratio", type=float, default=0.95)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.90)
    parser.add_argument("--max-tail-error-to-flip-ratio", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def make_operator(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, float]:
    if condition == "near_unit_radius_fail":
        rho = 0.995
        signs = rng.choice([-1.0, 1.0], size=args.state_dim)
        return np.diag(signs * rng.uniform(0.985, rho, size=args.state_dim)), rho
    if condition == "underestimated_rho_fail":
        true_rho = 0.90
        declared = 0.65
        signs = rng.choice([-1.0, 1.0], size=args.state_dim)
        return np.diag(signs * rng.uniform(0.84, true_rho, size=args.state_dim)), declared
    if condition == "non_normal_fail":
        rho = 0.65
        matrix = np.zeros((args.state_dim, args.state_dim), dtype=np.float64)
        for i in range(0, args.state_dim, 2):
            matrix[i, i] = rho
            if i + 1 < args.state_dim:
                matrix[i + 1, i + 1] = rho
                matrix[i, i + 1] = 1.35
        return matrix, rho
    rho = 0.65
    signs = rng.choice([-1.0, 1.0], size=args.state_dim)
    return np.diag(signs * rng.uniform(0.45, rho, size=args.state_dim)), rho


def choose_horizon(max_grad_norm: float, rho_bound: float, epsilon: float, horizon_len: int) -> tuple[int, float]:
    denom = max(1.0 - rho_bound, 1e-9)
    for horizon in range(1, horizon_len + 1):
        bound = max_grad_norm * (rho_bound ** horizon) / denom
        if bound <= epsilon:
            return horizon, bound
    return horizon_len, max_grad_norm * (rho_bound ** horizon_len) / denom


def adjoint_sum(operator: np.ndarray, gradients: np.ndarray, horizon: int) -> np.ndarray:
    power = np.eye(operator.shape[0], dtype=np.float64)
    transpose = operator.T
    total = np.zeros(operator.shape[0], dtype=np.float64)
    for k in range(horizon):
        total += power @ gradients[k]
        power = transpose @ power
    return total


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    operator, declared_rho = make_operator(rng, args, condition)
    gradients = rng.normal(scale=0.03, size=(args.horizon_len, args.state_dim))
    gradients[0] += rng.normal(scale=0.10, size=args.state_dim)
    if condition == "epsilon_miscalibration_fail":
        gradients[6, 0] += 0.45
    max_grad_norm = float(np.max(np.linalg.norm(gradients, axis=1)))
    decision_epsilon = (
        args.miscalibrated_flip_threshold
        if condition == "epsilon_miscalibration_fail"
        else args.flip_threshold
    )
    trunc_horizon, certified_bound = choose_horizon(
        max_grad_norm,
        declared_rho,
        decision_epsilon,
        args.horizon_len,
    )
    exact = adjoint_sum(operator, gradients, args.horizon_len)
    truncated = adjoint_sum(operator, gradients, trunc_horizon)
    tail_error = float(np.linalg.norm(exact - truncated))
    tail_ratio = tail_error / args.flip_threshold
    bound_violation = tail_error / max(certified_bound, 1e-12)
    eigvals = np.linalg.eigvals(operator)
    spectral_radius = float(np.max(np.abs(eigvals)))
    operator_norm = float(np.linalg.norm(operator, ord=2))
    audit_cost = (
        args.expensive_audit_cost_ratio
        if condition == "audit_cost_fail"
        else args.audit_cost_ratio
    )
    horizon_ratio = trunc_horizon / args.horizon_len
    price = horizon_ratio + audit_cost

    failed: list[str] = []
    if not np.isfinite(
        [
            tail_error,
            tail_ratio,
            certified_bound,
            bound_violation,
            spectral_radius,
            operator_norm,
            price,
        ]
    ).all():
        failed.append("non_finite_metric")
    if tail_ratio > args.max_tail_error_to_flip_ratio:
        failed.append("tail_exceeds_flip_threshold")
    if bound_violation > 1.0 + 1e-9:
        failed.append("bound_violation")
    if trunc_horizon >= args.horizon_len:
        failed.append("no_truncation")
    if price > args.max_priced_work_ratio:
        failed.append("priced_work")
    if declared_rho + 1e-9 < spectral_radius:
        failed.append("rho_underestimate")
    if operator_norm > declared_rho + 1e-9:
        failed.append("operator_norm_exceeds_bound")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed),
        horizon_len=args.horizon_len,
        state_dim=args.state_dim,
        true_spectral_radius=spectral_radius,
        declared_rho_bound=declared_rho,
        operator_norm=operator_norm,
        truncation_horizon=trunc_horizon,
        horizon_ratio=horizon_ratio,
        tail_error_norm=tail_error,
        tail_error_to_flip_ratio=tail_ratio,
        certified_bound_at_horizon=certified_bound,
        bound_violation_ratio=bound_violation,
        priced_work_ratio=price,
        audit_cost_ratio=audit_cost,
    )


NUMERIC_FIELDS = [
    "horizon_len",
    "state_dim",
    "true_spectral_radius",
    "declared_rho_bound",
    "operator_norm",
    "truncation_horizon",
    "horizon_ratio",
    "tail_error_norm",
    "tail_error_to_flip_ratio",
    "certified_bound_at_horizon",
    "bound_violation_ratio",
    "priced_work_ratio",
    "audit_cost_ratio",
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
        "positive_condition_tail_and_price": (
            aggregates["positive"]["tail_error_to_flip_ratio_max"] <= 1.0
            and aggregates["positive"]["bound_violation_ratio_max"] <= 1.0
            and aggregates["positive"]["horizon_ratio_max"] < 1.0
            and aggregates["positive"]["priced_work_ratio_max"] <= 0.90
        ),
        "near_unit_control_all_fail_no_truncation": all(
            (not result.passed)
            and (
                "no_truncation" in result.fail_reason
                or "priced_work" in result.fail_reason
            )
            for result in by_condition["near_unit_radius_fail"]
        ),
        "underestimated_rho_control_all_fail_bound": all(
            (not result.passed)
            and (
                "rho_underestimate" in result.fail_reason
                or "bound_violation" in result.fail_reason
                or "tail_exceeds_flip_threshold" in result.fail_reason
            )
            for result in by_condition["underestimated_rho_fail"]
        ),
        "non_normal_control_all_fail_norm": all(
            (not result.passed)
            and (
                "operator_norm_exceeds_bound" in result.fail_reason
                or "bound_violation" in result.fail_reason
            )
            for result in by_condition["non_normal_fail"]
        ),
        "epsilon_control_all_fail_tail": all(
            (not result.passed)
            and "tail_exceeds_flip_threshold" in result.fail_reason
            for result in by_condition["epsilon_miscalibration_fail"]
        ),
        "audit_cost_control_all_fail_price": all(
            (not result.passed) and "priced_work" in result.fail_reason
            for result in by_condition["audit_cost_fail"]
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
    decision = "PASS_R502_S21_TNT_GATE" if all(checks.values()) else "FAIL_R502_S21_TNT_GATE"

    payload = {
        "round": "R502",
        "canonical_id": "R502-S21",
        "aspect": "Speed",
        "finding": "Ternary Operator Norm Truncation (TNT)",
        "decision": decision,
        "disposition": "synthetic_positive_control_identified_not_training_speed_claim",
        "args": vars(args) | {"output": str(args.output) if args.output else None},
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU ternary operator-norm truncation gate only; not "
            "heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_S21_TNT_GATE] "
        f"active=1 decision={decision} "
        f"disposition={payload['disposition']} "
        f"positive_truncation_horizon_mean={positive['truncation_horizon_mean']:.12f} "
        f"positive_horizon_ratio_mean={positive['horizon_ratio_mean']:.12f} "
        f"positive_tail_error_to_flip_max={positive['tail_error_to_flip_ratio_max']:.12e} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.12f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
