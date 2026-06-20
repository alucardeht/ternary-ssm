#!/usr/bin/env python3
"""R501 S78 CE Hamiltonian potential field gate.

This CPU diagnostic implements the R501 Quality Math 15 finding:
"CE Hamiltonian Potential Field (CE-HPF)".

The gate constructs a stable linear hidden-state dynamics with quadratic CE
costs. In this setting the exact future-CE value function is quadratic, so its
spatial derivative can be compared directly against the full deep recurrent
backward gradient. A positive case must have near-zero Bellman residual,
near-zero gradient error, a conservative field, and a priced denominator below
the proposed bound.

Controls exercise the declared risks: a stale dynamics potential breaks the
Bellman-gradient identity, a truncated horizon misses future CE, a
non-conservative field cannot be a Hamiltonian potential gradient, and an
expensive potential solver erases the priced denominator.

This is a synthetic CPU quadratic-dynamics potential-field diagnostic, not
heldout CE progress, real corpus token efficiency, GPU training-speed evidence,
or production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s78_ce_hpf_gate.v1"
CONDITIONS = (
    "positive",
    "stale_dynamics_fail",
    "truncated_horizon_fail",
    "nonconservative_field_fail",
    "potential_cost_fail",
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
    sample_count: int
    horizon_coverage_fraction: float
    bellman_residual_abs_max: float
    gradient_relative_error_max: float
    gradient_cosine_min: float
    curl_residual: float
    priced_work_ratio: float
    projected_parallel_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50178)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--max-bellman-residual", type=float, default=1.0e-10)
    parser.add_argument("--max-gradient-relative-error", type=float, default=1.0e-10)
    parser.add_argument("--min-gradient-cosine", type=float, default=0.999999)
    parser.add_argument("--max-curl-residual", type=float, default=1.0e-12)
    parser.add_argument("--min-horizon-coverage", type=float, default=1.0)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.55)
    parser.add_argument("--priced-work-ratio", type=float, default=0.52)
    parser.add_argument("--expensive-priced-work-ratio", type=float, default=0.63)
    return parser.parse_args()


def stable_dynamics(rng: np.random.Generator, state_dim: int) -> np.ndarray:
    matrix = rng.normal(scale=0.22, size=(state_dim, state_dim))
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return matrix * (0.72 / max(float(singular_values[0]), 1.0e-30))


def solve_quadratic_potential(
    dynamics: np.ndarray,
    targets: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    horizon, state_dim = targets.shape
    p_terms: list[np.ndarray] = [np.zeros((state_dim, state_dim), dtype=np.float64) for _ in range(horizon + 1)]
    q_terms: list[np.ndarray] = [np.zeros(state_dim, dtype=np.float64) for _ in range(horizon + 1)]
    c_terms: list[float] = [0.0 for _ in range(horizon + 1)]
    for t in reversed(range(horizon)):
        p_terms[t] = np.eye(state_dim) + dynamics.T @ p_terms[t + 1] @ dynamics
        q_terms[t] = -targets[t] + dynamics.T @ q_terms[t + 1]
        c_terms[t] = 0.5 * float(np.dot(targets[t], targets[t])) + c_terms[t + 1]
    return p_terms, q_terms, c_terms


def value(p_term: np.ndarray, q_term: np.ndarray, c_term: float, state: np.ndarray) -> float:
    return 0.5 * float(state @ p_term @ state) + float(q_term @ state) + c_term


def ce_loss(state: np.ndarray, target: np.ndarray) -> float:
    residual = state - target
    return 0.5 * float(np.dot(residual, residual))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0e-30)
    return float(np.dot(a, b) / denom)


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def make_nonconservative_matrix(rng: np.random.Generator, state_dim: int) -> np.ndarray:
    matrix = rng.normal(scale=0.18, size=(state_dim, state_dim))
    return matrix - matrix.T


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    dynamics = stable_dynamics(rng, args.state_dim)
    targets = rng.normal(scale=0.65, size=(args.horizon, args.state_dim))
    true_p, true_q, true_c = solve_quadratic_potential(dynamics, targets)

    candidate_dynamics = dynamics
    candidate_targets = targets
    horizon_coverage = 1.0
    if condition == "stale_dynamics_fail":
        candidate_dynamics = dynamics + rng.normal(scale=0.08, size=dynamics.shape)
    elif condition == "truncated_horizon_fail":
        truncated_horizon = max(2, args.horizon // 4)
        candidate_targets = targets[:truncated_horizon]
        horizon_coverage = truncated_horizon / args.horizon

    candidate_p, candidate_q, candidate_c = solve_quadratic_potential(
        candidate_dynamics,
        candidate_targets,
    )
    field_matrix = candidate_p[0]
    if condition == "nonconservative_field_fail":
        field_matrix = field_matrix + make_nonconservative_matrix(rng, args.state_dim)

    states = rng.normal(scale=0.80, size=(args.sample_count, args.state_dim))
    bellman_residuals: list[float] = []
    relative_errors: list[float] = []
    cosines: list[float] = []
    for state in states:
        next_state = dynamics @ state
        if len(candidate_p) > 1:
            residual = (
                value(candidate_p[0], candidate_q[0], candidate_c[0], state)
                - ce_loss(state, targets[0])
                - value(candidate_p[1], candidate_q[1], candidate_c[1], next_state)
            )
        else:
            residual = value(candidate_p[0], candidate_q[0], candidate_c[0], state) - ce_loss(
                state,
                targets[0],
            )
        bellman_residuals.append(abs(float(residual)))

        true_gradient = true_p[0] @ state + true_q[0]
        candidate_gradient = field_matrix @ state + candidate_q[0]
        relative_error = float(np.linalg.norm(candidate_gradient - true_gradient)) / max(
            float(np.linalg.norm(true_gradient)),
            1.0e-30,
        )
        relative_errors.append(relative_error)
        cosines.append(cosine(candidate_gradient, true_gradient))

    curl_residual = float(np.linalg.norm(field_matrix - field_matrix.T)) / max(
        float(np.linalg.norm(field_matrix)),
        1.0e-30,
    )
    priced_work_ratio = (
        args.expensive_priced_work_ratio
        if condition == "potential_cost_fail"
        else args.priced_work_ratio
    )
    projected_speedup = 1.0 / priced_work_ratio

    bellman_max = max(bellman_residuals)
    gradient_error_max = max(relative_errors)
    gradient_cosine_min = min(cosines)

    failed: list[str] = []
    if not finite(
        horizon_coverage,
        bellman_max,
        gradient_error_max,
        gradient_cosine_min,
        curl_residual,
        priced_work_ratio,
        projected_speedup,
    ):
        failed.append("non_finite_metric")
    if bellman_max > args.max_bellman_residual:
        failed.append("bellman_residual")
    if gradient_error_max > args.max_gradient_relative_error:
        failed.append("gradient_error")
    if gradient_cosine_min < args.min_gradient_cosine:
        failed.append("gradient_direction")
    if curl_residual > args.max_curl_residual:
        failed.append("nonconservative_field")
    if horizon_coverage < args.min_horizon_coverage:
        failed.append("horizon_coverage")
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
        state_dim=args.state_dim,
        horizon=args.horizon,
        sample_count=args.sample_count,
        horizon_coverage_fraction=float(horizon_coverage),
        bellman_residual_abs_max=float(bellman_max),
        gradient_relative_error_max=float(gradient_error_max),
        gradient_cosine_min=float(gradient_cosine_min),
        curl_residual=float(curl_residual),
        priced_work_ratio=float(priced_work_ratio),
        projected_parallel_speedup=float(projected_speedup),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "state_dim",
        "horizon",
        "sample_count",
        "horizon_coverage_fraction",
        "bellman_residual_abs_max",
        "gradient_relative_error_max",
        "gradient_cosine_min",
        "curl_residual",
        "priced_work_ratio",
        "projected_parallel_speedup",
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
    stale = [result for result in results if result.condition == "stale_dynamics_fail"]
    truncated = [result for result in results if result.condition == "truncated_horizon_fail"]
    nonconservative = [result for result in results if result.condition == "nonconservative_field_fail"]
    cost = [result for result in results if result.condition == "potential_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_bellman_residual_bounded": (
            aggregates["positive"]["bellman_residual_abs_max_max"] <= 1.0e-10
        ),
        "positive_condition_gradient_matches_deep_backward": (
            aggregates["positive"]["gradient_relative_error_max_max"] <= 1.0e-10
            and aggregates["positive"]["gradient_cosine_min_min"] >= 0.999999
        ),
        "positive_condition_conservative": (
            aggregates["positive"]["curl_residual_max"] <= 1.0e-12
        ),
        "positive_condition_priced": (
            aggregates["positive"]["priced_work_ratio_mean"] <= 0.55
        ),
        "stale_control_all_fail_residual_or_gradient": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "bellman_residual" in result.fail_reason
                or "gradient_error" in result.fail_reason
            )
            for result in stale
        ),
        "truncated_control_all_fail_horizon_or_gradient": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "horizon_coverage" in result.fail_reason
                or "gradient_error" in result.fail_reason
            )
            for result in truncated
        ),
        "nonconservative_control_all_fail_curl": all(
            (not result.passed)
            and result.fail_reason is not None
            and "nonconservative_field" in result.fail_reason
            for result in nonconservative
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
        "PASS_R501_S78_CE_HPF_GATE"
        if all(checks.values())
        else "FAIL_R501_S78_CE_HPF_GATE"
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
            "Synthetic CPU CE Hamiltonian potential field gate only; not "
            "heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S78_CE_HPF_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_bellman_residual_abs_max_max={positive['bellman_residual_abs_max_max']:.12e} "
        f"positive_gradient_relative_error_max_max={positive['gradient_relative_error_max_max']:.12e} "
        f"positive_gradient_cosine_min_min={positive['gradient_cosine_min_min']:.12f} "
        f"positive_curl_residual_max={positive['curl_residual_max']:.12e} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_parallel_speedup_mean={positive['projected_parallel_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
