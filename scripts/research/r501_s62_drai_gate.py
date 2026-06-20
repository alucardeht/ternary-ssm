#!/usr/bin/env python3
"""R501 S62 dual-resolvent action integrator gate.

This CPU diagnostic implements the R501 Speed Paradigm 14 finding:
"Dual-Resolvent Action Integrator (DRAI)".

A dual-resolvent macro-action should integrate K local CE-descent micro-steps
into one verified transaction. The gate checks exact macro/micro equivalence on
curvature-bounded quadratic CE fixtures, verifies the descent lower bound

    D_lower(U) = -g(theta)^T U - 0.5 * beta * ||U||^2 - debt(U),

and prices the macro action as one dense transaction per K micro-steps plus a
single verifier cost.

Controls exercise the declared risks: an underestimated curvature bound breaks
the lower-bound certificate, a truncated macro-action creates unpriced debt, and
an expensive verifier erases the step-count reduction.

This is a synthetic CPU diagnostic, not heldout CE progress, GPU training-speed
evidence, or production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s62_drai_gate.v1"
CONDITIONS = (
    "positive",
    "curvature_bound_fail",
    "debt_underpriced_fail",
    "verifier_cost_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    dimension: int
    macro_k: int
    learning_rate: float
    beta_certificate_ratio: float
    verifier_work_ratio: float
    relative_macro_error: float
    actual_decrease: float
    lower_bound: float
    lower_bound_ratio: float
    bound_violation_ratio: float
    debt_ratio: float
    priced_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50162)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument("--macro-k", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument("--verifier-work-ratio", type=float, default=0.08)
    parser.add_argument("--max-relative-macro-error", type=float, default=1.0e-10)
    parser.add_argument("--max-bound-violation-ratio", type=float, default=1.0e-10)
    parser.add_argument("--max-debt-ratio", type=float, default=0.02)
    parser.add_argument("--min-lower-bound-ratio", type=float, default=0.30)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.50)
    return parser.parse_args()


def random_orthogonal(rng: np.random.Generator, dimension: int) -> np.ndarray:
    matrix = rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def make_problem(
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = random_orthogonal(rng, args.dimension)
    eigenvalues = np.linspace(0.18, 1.35, args.dimension)
    eigenvalues += rng.uniform(-0.015, 0.015, size=args.dimension)
    eigenvalues = np.clip(eigenvalues, 0.08, None)
    error = rng.normal(size=args.dimension)
    error /= max(float(np.linalg.norm(error)), 1.0e-30)
    error *= rng.uniform(1.6, 2.2)
    return q, eigenvalues, error


def apply_hessian(q: np.ndarray, eigenvalues: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return q @ (eigenvalues * (q.T @ vector))


def loss(q: np.ndarray, eigenvalues: np.ndarray, error: np.ndarray) -> float:
    return 0.5 * float(np.dot(error, apply_hessian(q, eigenvalues, error)))


def run_micro_steps(
    q: np.ndarray,
    eigenvalues: np.ndarray,
    error: np.ndarray,
    learning_rate: float,
    macro_k: int,
) -> np.ndarray:
    state = error.copy()
    for _ in range(macro_k):
        state = state - learning_rate * apply_hessian(q, eigenvalues, state)
    return state - error


def exact_resolvent_action(
    q: np.ndarray,
    eigenvalues: np.ndarray,
    error: np.ndarray,
    learning_rate: float,
    macro_k: int,
) -> np.ndarray:
    coordinates = q.T @ error
    factors = np.power(1.0 - learning_rate * eigenvalues, macro_k) - 1.0
    return q @ (factors * coordinates)


def truncated_resolvent_action(
    q: np.ndarray,
    eigenvalues: np.ndarray,
    error: np.ndarray,
    learning_rate: float,
    macro_k: int,
) -> np.ndarray:
    coordinates = q.T @ error
    factors = np.power(1.0 - learning_rate * eigenvalues, macro_k) - 1.0
    keep = max(1, eigenvalues.size // 4)
    mask = np.zeros_like(eigenvalues)
    mask[-keep:] = 1.0
    return q @ (factors * coordinates * mask)


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    q, eigenvalues, error = make_problem(rng, args)
    macro_k = args.macro_k
    verifier_work_ratio = args.verifier_work_ratio
    beta_certificate = float(np.max(eigenvalues))

    if condition == "verifier_cost_fail":
        macro_k = 3
        verifier_work_ratio = 0.27
    if condition == "curvature_bound_fail":
        beta_certificate = 0.0

    micro_action = run_micro_steps(q, eigenvalues, error, args.learning_rate, macro_k)
    if condition == "debt_underpriced_fail":
        macro_action = truncated_resolvent_action(
            q,
            eigenvalues,
            error,
            args.learning_rate,
            macro_k,
        )
    else:
        macro_action = exact_resolvent_action(
            q,
            eigenvalues,
            error,
            args.learning_rate,
            macro_k,
        )

    before = loss(q, eigenvalues, error)
    after_macro = loss(q, eigenvalues, error + macro_action)
    actual_decrease = before - after_macro
    initial_gradient = apply_hessian(q, eigenvalues, error)
    macro_norm = float(np.linalg.norm(macro_action))
    relative_macro_error = float(np.linalg.norm(macro_action - micro_action)) / max(
        float(np.linalg.norm(micro_action)),
        1.0e-30,
    )
    debt_ratio = relative_macro_error
    lower_bound = (
        -float(np.dot(initial_gradient, macro_action))
        - 0.5 * beta_certificate * macro_norm * macro_norm
        - debt_ratio * max(actual_decrease, 0.0)
    )
    lower_bound_ratio = lower_bound / max(abs(actual_decrease), 1.0e-30)
    bound_violation_ratio = max(lower_bound - actual_decrease, 0.0) / max(
        abs(actual_decrease),
        1.0e-30,
    )
    priced_work_ratio = (1.0 / macro_k) + verifier_work_ratio
    projected_speedup = 1.0 / priced_work_ratio
    beta_certificate_ratio = beta_certificate / max(float(np.max(eigenvalues)), 1.0e-30)

    failed: list[str] = []
    if not finite(
        relative_macro_error,
        actual_decrease,
        lower_bound,
        lower_bound_ratio,
        bound_violation_ratio,
        debt_ratio,
        priced_work_ratio,
        projected_speedup,
    ):
        failed.append("non_finite_metric")
    if relative_macro_error > args.max_relative_macro_error:
        failed.append("macro_exactness")
    if debt_ratio > args.max_debt_ratio:
        failed.append("debt")
    if bound_violation_ratio > args.max_bound_violation_ratio:
        failed.append("curvature_bound")
    if lower_bound_ratio < args.min_lower_bound_ratio:
        failed.append("lower_bound_strength")
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
        dimension=args.dimension,
        macro_k=macro_k,
        learning_rate=float(args.learning_rate),
        beta_certificate_ratio=float(beta_certificate_ratio),
        verifier_work_ratio=float(verifier_work_ratio),
        relative_macro_error=float(relative_macro_error),
        actual_decrease=float(actual_decrease),
        lower_bound=float(lower_bound),
        lower_bound_ratio=float(lower_bound_ratio),
        bound_violation_ratio=float(bound_violation_ratio),
        debt_ratio=float(debt_ratio),
        priced_work_ratio=float(priced_work_ratio),
        projected_speedup=float(projected_speedup),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "dimension",
        "macro_k",
        "learning_rate",
        "beta_certificate_ratio",
        "verifier_work_ratio",
        "relative_macro_error",
        "actual_decrease",
        "lower_bound",
        "lower_bound_ratio",
        "bound_violation_ratio",
        "debt_ratio",
        "priced_work_ratio",
        "projected_speedup",
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
    curvature = [result for result in results if result.condition == "curvature_bound_fail"]
    debt = [result for result in results if result.condition == "debt_underpriced_fail"]
    verifier = [result for result in results if result.condition == "verifier_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_and_bounded": (
            aggregates["positive"]["relative_macro_error_max"] <= 1.0e-10
            and aggregates["positive"]["debt_ratio_max"] <= 0.02
            and aggregates["positive"]["bound_violation_ratio_max"] <= 1.0e-10
            and aggregates["positive"]["lower_bound_ratio_mean"] >= 0.30
        ),
        "positive_condition_priced": (
            aggregates["positive"]["macro_k_mean"] >= 8.0
            and aggregates["positive"]["priced_work_ratio_mean"] <= 0.50
        ),
        "curvature_control_all_fail_bound": all(
            (not result.passed)
            and result.fail_reason is not None
            and "curvature_bound" in result.fail_reason
            for result in curvature
        ),
        "debt_control_all_fail_debt": all(
            (not result.passed)
            and result.fail_reason is not None
            and "macro_exactness" in result.fail_reason
            and "debt" in result.fail_reason
            for result in debt
        ),
        "verifier_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "price" in result.fail_reason
            for result in verifier
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
        "PASS_R501_S62_DRAI_GATE"
        if all(checks.values())
        else "FAIL_R501_S62_DRAI_GATE"
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
            "Synthetic CPU dual-resolvent macro-action gate only; not heldout "
            "CE progress, not GPU training speed, and not production v12 "
            "integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S62_DRAI_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_macro_k_mean={positive['macro_k_mean']:.6f} "
        f"positive_relative_macro_error_max={positive['relative_macro_error_max']:.6e} "
        f"positive_lower_bound_ratio_mean={positive['lower_bound_ratio_mean']:.6f} "
        f"positive_debt_ratio_max={positive['debt_ratio_max']:.6e} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
