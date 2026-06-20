#!/usr/bin/env python3
"""R501 S59 cross-entropy potential-field adjoint gate.

This CPU diagnostic implements the R501 Speed Paradigm 11 finding:
"Cross-Entropy Potential Field Adjoint (CEPFA)".

For a linear state transition with quadratic CE surrogate losses, the exact
recurrent cotangent is the gradient of a time-indexed value potential:

    lambda_t = grad_h V_t(h_t)

The candidate evaluates that potential gradient in the forward path and audits
the Bellman residual sparsely instead of materializing dense step-by-step
backward cotangents. It survives only when the potential-gradient proxy matches
dense backward, full and sampled Bellman residuals are below the descent budget,
and the priced proxy plus audit route is cheaper than dense backward.

Controls exercise the declared risks: stale potential parameters break the
Bellman identity, nonlinear dynamics break the linear potential contract, and
an expensive potential head recovers dense cost.

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


SCHEMA = "r501_s59_cepfa_gate.v1"
CONDITIONS = (
    "positive",
    "stale_potential_fail",
    "nonlinear_dynamics_fail",
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
    steps: int
    state_dim: int
    proxy_relative_error: float
    bellman_residual_ratio: float
    sampled_bellman_residual_ratio: float
    descent_alignment_ratio: float
    potential_eval_work_ratio: float
    sparse_audit_work_ratio: float
    forward_overhead_ratio: float
    priced_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50159)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=10)
    parser.add_argument("--obs-dim", type=int, default=4)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--max-proxy-error", type=float, default=1.0e-10)
    parser.add_argument("--max-bellman-residual", type=float, default=1.0e-10)
    parser.add_argument("--max-sampled-bellman-residual", type=float, default=1.0e-10)
    parser.add_argument("--min-descent-alignment-ratio", type=float, default=0.999999)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.45)
    parser.add_argument("--potential-eval-work-ratio", type=float, default=0.27)
    parser.add_argument("--sparse-audit-work-ratio", type=float, default=0.07)
    parser.add_argument("--forward-overhead-ratio", type=float, default=0.04)
    return parser.parse_args()


def stable_transition(rng: np.random.Generator, dim: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    eig = rng.uniform(0.25, 0.70, size=dim) * rng.choice(np.array([-1.0, 1.0]), size=dim)
    return (q @ np.diag(eig) @ q.T).astype(np.float64)


def make_observer(rng: np.random.Generator, obs_dim: int, state_dim: int) -> np.ndarray:
    c = rng.normal(scale=0.35, size=(obs_dim, state_dim))
    q, _ = np.linalg.qr(c.T)
    return q[:, :obs_dim].T.astype(np.float64)


def forward_linear(a: np.ndarray, b: np.ndarray, h0: np.ndarray) -> np.ndarray:
    steps = b.shape[0]
    h = np.zeros((steps + 1, h0.size), dtype=np.float64)
    h[0] = h0
    for t in range(steps):
        h[t + 1] = a @ h[t] + b[t]
    return h


def forward_nonlinear(
    a: np.ndarray,
    b: np.ndarray,
    h0: np.ndarray,
    m: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    steps = b.shape[0]
    h = np.zeros((steps + 1, h0.size), dtype=np.float64)
    jacobians = np.zeros((steps, h0.size, h0.size), dtype=np.float64)
    h[0] = h0
    for t in range(steps):
        z = m @ h[t]
        nonlinear = np.tanh(z)
        h[t + 1] = a @ h[t] + b[t] + scale * nonlinear
        jacobians[t] = a + scale * (1.0 - nonlinear**2)[:, None] * m
    return h, jacobians


def loss_terms(
    c: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hessian = c.T @ c
    linear = -(targets @ c).astype(np.float64)
    return hessian.astype(np.float64), linear


def dense_backward(
    sources: np.ndarray,
    jacobians: np.ndarray,
) -> np.ndarray:
    steps = jacobians.shape[0]
    lambdas = np.zeros_like(sources, dtype=np.float64)
    lambdas[steps] = sources[steps]
    for t in range(steps - 1, -1, -1):
        lambdas[t] = sources[t] + jacobians[t].T @ lambdas[t + 1]
    return lambdas


def linear_value_potential(
    a: np.ndarray,
    b: np.ndarray,
    hessian: np.ndarray,
    linear_terms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    steps = b.shape[0]
    dim = a.shape[0]
    p_mat = np.zeros((steps + 1, dim, dim), dtype=np.float64)
    p_vec = np.zeros((steps + 1, dim), dtype=np.float64)
    p_mat[steps] = hessian
    p_vec[steps] = linear_terms[steps]
    for t in range(steps - 1, -1, -1):
        p_mat[t] = hessian + a.T @ p_mat[t + 1] @ a
        p_vec[t] = linear_terms[t] + a.T @ (p_mat[t + 1] @ b[t] + p_vec[t + 1])
    return p_mat, p_vec


def potential_gradient(p_mat: np.ndarray, p_vec: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.einsum("tij,tj->ti", p_mat, h) + p_vec


def bellman_residual(
    proxy: np.ndarray,
    sources: np.ndarray,
    jacobians: np.ndarray,
    samples: np.ndarray | None = None,
) -> float:
    steps = jacobians.shape[0]
    residuals = []
    indices = range(steps) if samples is None else samples
    for t in indices:
        residuals.append(proxy[t] - (sources[t] + jacobians[t].T @ proxy[t + 1]))
    residual = np.stack(residuals, axis=0)
    reference = proxy[:-1] if samples is None else proxy[samples]
    return float(np.linalg.norm(residual)) / max(float(np.linalg.norm(reference)), 1.0e-30)


def condition_costs(args: argparse.Namespace, condition: str) -> tuple[float, float, float]:
    if condition == "potential_cost_fail":
        return 0.39, 0.09, 0.05
    return (
        args.potential_eval_work_ratio,
        args.sparse_audit_work_ratio,
        args.forward_overhead_ratio,
    )


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    a = stable_transition(rng, args.state_dim)
    b = rng.normal(scale=0.12, size=(args.steps, args.state_dim)).astype(np.float64)
    h0 = rng.normal(scale=0.30, size=args.state_dim).astype(np.float64)
    c = make_observer(rng, args.obs_dim, args.state_dim)
    targets = rng.normal(scale=0.25, size=(args.steps + 1, args.obs_dim)).astype(np.float64)
    hessian, linear_terms = loss_terms(c, targets)

    if condition == "nonlinear_dynamics_fail":
        m = rng.normal(scale=0.20, size=(args.state_dim, args.state_dim))
        h, jacobians = forward_nonlinear(a, b, h0, m, scale=0.18)
        linear_h = forward_linear(a, b, h0)
        p_mat, p_vec = linear_value_potential(a, b, hessian, linear_terms)
        proxy = potential_gradient(p_mat, p_vec, linear_h)
    else:
        h = forward_linear(a, b, h0)
        jacobians = np.broadcast_to(a, (args.steps, args.state_dim, args.state_dim)).copy()
        p_mat, p_vec = linear_value_potential(a, b, hessian, linear_terms)
        if condition == "stale_potential_fail":
            p_mat = p_mat + rng.normal(scale=0.035, size=p_mat.shape)
            p_vec = p_vec + rng.normal(scale=0.060, size=p_vec.shape)
        proxy = potential_gradient(p_mat, p_vec, h)

    sources = h @ hessian.T + linear_terms
    dense_lambdas = dense_backward(sources, jacobians)
    sample_count = min(args.sample_count, args.steps)
    samples = np.sort(rng.choice(np.arange(args.steps), size=sample_count, replace=False))
    proxy_error = float(np.linalg.norm(proxy - dense_lambdas)) / max(
        float(np.linalg.norm(dense_lambdas)),
        1.0e-30,
    )
    full_residual = bellman_residual(proxy, sources, jacobians)
    sampled_residual = bellman_residual(proxy, sources, jacobians, samples=samples)
    descent_alignment = float(np.sum(proxy * dense_lambdas)) / max(
        float(np.sum(dense_lambdas * dense_lambdas)),
        1.0e-30,
    )
    potential_cost, audit_cost, overhead_cost = condition_costs(args, condition)
    priced_work_ratio = potential_cost + audit_cost + overhead_cost
    projected_speedup = 1.0 / priced_work_ratio if priced_work_ratio > 0.0 else math.inf

    failed: list[str] = []
    if not all(
        math.isfinite(value)
        for value in (
            proxy_error,
            full_residual,
            sampled_residual,
            descent_alignment,
            priced_work_ratio,
        )
    ):
        failed.append("non_finite_metric")
    if proxy_error > args.max_proxy_error:
        failed.append("proxy_error")
    if full_residual > args.max_bellman_residual:
        failed.append("bellman_residual")
    if sampled_residual > args.max_sampled_bellman_residual:
        failed.append("sampled_bellman_residual")
    if descent_alignment < args.min_descent_alignment_ratio:
        failed.append("descent_alignment")
    if priced_work_ratio > args.max_priced_work_ratio:
        failed.append("price")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=(passed == expected_pass),
        fail_reason=";".join(failed) if failed else None,
        steps=args.steps,
        state_dim=args.state_dim,
        proxy_relative_error=float(proxy_error),
        bellman_residual_ratio=float(full_residual),
        sampled_bellman_residual_ratio=float(sampled_residual),
        descent_alignment_ratio=float(descent_alignment),
        potential_eval_work_ratio=float(potential_cost),
        sparse_audit_work_ratio=float(audit_cost),
        forward_overhead_ratio=float(overhead_cost),
        priced_work_ratio=float(priced_work_ratio),
        projected_speedup=float(projected_speedup),
    )


def aggregate(results: list[TrialResult]) -> dict[str, Any]:
    by_condition: dict[str, list[TrialResult]] = {
        condition: [result for result in results if result.condition == condition]
        for condition in CONDITIONS
    }
    numeric_fields = (
        "proxy_relative_error",
        "bellman_residual_ratio",
        "sampled_bellman_residual_ratio",
        "descent_alignment_ratio",
        "potential_eval_work_ratio",
        "sparse_audit_work_ratio",
        "forward_overhead_ratio",
        "priced_work_ratio",
        "projected_speedup",
    )
    aggregates: dict[str, Any] = {}
    for condition, condition_results in by_condition.items():
        aggregates[condition] = {
            "count": len(condition_results),
            "passed_count": sum(result.passed for result in condition_results),
            "identifying_correct_count": sum(
                result.identifying_correct for result in condition_results
            ),
            "fail_reasons": sorted(
                {result.fail_reason for result in condition_results if result.fail_reason}
            ),
        }
        for field in numeric_fields:
            values = [float(getattr(result, field)) for result in condition_results]
            if not values:
                continue
            aggregates[condition][f"{field}_mean"] = float(np.mean(values))
            aggregates[condition][f"{field}_max"] = float(np.max(values))
            aggregates[condition][f"{field}_min"] = float(np.min(values))
    return aggregates


def all_finite(results: list[TrialResult]) -> bool:
    for result in results:
        for value in asdict(result).values():
            if isinstance(value, float) and not math.isfinite(value):
                return False
    return True


def required_checks(results: list[TrialResult], aggregates: dict[str, Any]) -> dict[str, bool]:
    positive = [result for result in results if result.condition == "positive"]
    stale = [result for result in results if result.condition == "stale_potential_fail"]
    nonlinear = [result for result in results if result.condition == "nonlinear_dynamics_fail"]
    cost = [result for result in results if result.condition == "potential_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_potential": (
            aggregates["positive"]["proxy_relative_error_max"] <= 1.0e-10
            and aggregates["positive"]["bellman_residual_ratio_max"] <= 1.0e-10
            and aggregates["positive"]["sampled_bellman_residual_ratio_max"] <= 1.0e-10
        ),
        "positive_condition_price": aggregates["positive"]["priced_work_ratio_mean"] <= 0.45,
        "stale_control_all_fail_bellman": all(
            (not result.passed)
            and result.fail_reason is not None
            and "proxy_error" in result.fail_reason
            and "bellman_residual" in result.fail_reason
            for result in stale
        ),
        "nonlinear_control_all_fail_contract": all(
            (not result.passed)
            and result.fail_reason is not None
            and "proxy_error" in result.fail_reason
            and "bellman_residual" in result.fail_reason
            for result in nonlinear
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
            results.append(run_trial(args, condition, args.seed + 1000 * offset))

    aggregates = aggregate(results)
    checks = required_checks(results, aggregates)
    decision = "PASS_R501_S59_CEPFA_GATE" if all(checks.values()) else "FAIL_R501_S59_CEPFA_GATE"
    payload = {
        "schema": SCHEMA,
        "decision": decision,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "conditions": CONDITIONS,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU cross-entropy potential-field adjoint gate only; not "
            "heldout CE progress, not GPU training speed, and not production "
            "v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S59_CEPFA_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_proxy_error_max={positive['proxy_relative_error_max']:.6e} "
        f"positive_bellman_residual_max={positive['bellman_residual_ratio_max']:.6e} "
        f"positive_sampled_residual_max={positive['sampled_bellman_residual_ratio_max']:.6e} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
