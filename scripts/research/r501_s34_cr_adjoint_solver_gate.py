#!/usr/bin/env python3
"""R501 S01 conjugate-residual adjoint solver gate.

This is a CPU diagnostic gate for the R501 speed finding
"Conjugate-Residual Adjoint Solver over Sequence Blocks".

The gate compares the exact recurrent adjoint

    lambda[t] = g[t] + A.T lambda[t + 1]

against a block-system solve of

    (I - Z A.T) Lambda = G

using a generalized conjugate residual iteration seeded from a prior segment.
It is not a measured GPU training-speed claim. It tests whether the proposed
mechanism has an identifying regime where:

1. the block-system residual is certified;
2. the solver matches the exact dense recurrent adjoint;
3. the iteration count remains below the dense fallback kill threshold; and
4. the priced parallel route is below 0.90x dense recurrent backward work.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s34_cr_adjoint_solver_gate.v1"
CONDITIONS = ("positive", "condition_fail", "pricing_fail", "random_control")


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    spectral_radius: float
    condition_proxy: float
    iterations: int
    dense_recurrent_matvecs: int
    iteration_fraction_of_dense: float
    relative_residual: float
    relative_solution_error: float
    priced_work_ratio: float
    projected_parallel_speedup: float
    dense_backward_seconds: float
    solver_seconds: float
    measured_cpu_time_ratio: float
    price_components: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50101)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--segment-length", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=24)
    parser.add_argument("--max-iters", type=int, default=80)
    parser.add_argument("--solver-eps", type=float, default=1.0e-5)
    parser.add_argument("--max-relative-residual", type=float, default=1.0e-5)
    parser.add_argument("--max-relative-solution-error", type=float, default=1.0e-4)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.90)
    parser.add_argument("--kill-iteration-fraction", type=float, default=0.50)
    parser.add_argument("--parallel-apply-ratio", type=float, default=0.075)
    parser.add_argument("--orthogonalization-ratio", type=float, default=0.010)
    parser.add_argument("--verification-ratio", type=float, default=0.045)
    parser.add_argument("--residual-settlement-ratio", type=float, default=0.025)
    parser.add_argument("--pricing-fail-extra-ratio", type=float, default=0.84)
    parser.add_argument("--timing-repeats", type=int, default=3)
    return parser.parse_args()


def orthogonal_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def transition_matrix(
    rng: np.random.Generator,
    state_dim: int,
    condition: str,
) -> tuple[np.ndarray, float, float]:
    if condition == "condition_fail":
        rho = 0.987
        eigs = np.linspace(0.955, rho, state_dim)
    elif condition == "random_control":
        rho = 0.88
        eigs = np.linspace(0.20, rho, state_dim)
    else:
        rho = 0.42
        eigs = np.linspace(0.04, rho, state_dim)

    q = orthogonal_matrix(rng, state_dim)
    a = q @ np.diag(eigs) @ q.T
    condition_proxy = float((1.0 + rho) / max(1.0e-12, 1.0 - rho))
    return a.astype(np.float64), float(rho), condition_proxy


def smooth_forcing(
    rng: np.random.Generator,
    segment_length: int,
    state_dim: int,
    high_frequency: bool,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, segment_length, dtype=np.float64)
    basis = [
        np.ones_like(t),
        t,
        t * t,
        np.sin(2.0 * math.pi * t),
        np.cos(2.0 * math.pi * t),
    ]
    if high_frequency:
        basis.extend(
            [
                np.sin(17.0 * math.pi * t),
                np.cos(19.0 * math.pi * t),
                np.sign(np.sin(23.0 * math.pi * t)),
            ]
        )
    x = np.stack(basis, axis=1)
    w = rng.normal(scale=0.35, size=(x.shape[1], state_dim))
    g = x @ w
    noise = 0.010 if not high_frequency else 0.085
    g += noise * rng.normal(size=g.shape)
    return g.astype(np.float64)


def build_forcing_pair(
    rng: np.random.Generator,
    condition: str,
    segment_length: int,
    state_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    high_frequency = condition in ("condition_fail", "random_control")
    prior = smooth_forcing(rng, segment_length, state_dim, high_frequency=False)
    if condition in ("positive", "pricing_fail"):
        current = prior + 0.006 * rng.normal(size=prior.shape)
    elif condition == "condition_fail":
        current = smooth_forcing(rng, segment_length, state_dim, high_frequency=True)
    elif condition == "random_control":
        current = rng.normal(scale=np.std(prior), size=prior.shape)
        rng.shuffle(current, axis=0)
    else:
        raise ValueError(f"unknown condition: {condition}")
    return prior, current.astype(np.float64)


def dense_recurrent_adjoint(g: np.ndarray, a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(g, dtype=np.float64)
    carry = np.zeros(g.shape[1], dtype=np.float64)
    for t in range(g.shape[0] - 1, -1, -1):
        carry = g[t] + carry @ a
        out[t] = carry
    return out


def block_system_apply(x: np.ndarray, a: np.ndarray) -> np.ndarray:
    y = np.array(x, copy=True)
    y[:-1] -= x[1:] @ a
    return y


def relative_norm(x: np.ndarray, denom: np.ndarray) -> float:
    numerator = float(np.linalg.norm(x.ravel()))
    denominator = float(np.linalg.norm(denom.ravel()))
    if denominator <= 0.0:
        return math.inf
    return numerator / denominator


def generalized_conjugate_residual(
    b: np.ndarray,
    a: np.ndarray,
    x0: np.ndarray,
    eps: float,
    max_iters: int,
) -> tuple[np.ndarray, int, float]:
    x = np.array(x0, copy=True)
    r = b - block_system_apply(x, a)
    b_norm = float(np.linalg.norm(b.ravel()))
    if b_norm <= 0.0:
        b_norm = 1.0
    rel_residual = float(np.linalg.norm(r.ravel()) / b_norm)
    if rel_residual <= eps:
        return x, 0, rel_residual

    p_basis: list[np.ndarray] = []
    ap_basis: list[np.ndarray] = []

    for iteration in range(1, max_iters + 1):
        p = np.array(r, copy=True)
        ap = block_system_apply(p, a)
        for old_p, old_ap in zip(p_basis, ap_basis):
            denom = float(np.vdot(old_ap.ravel(), old_ap.ravel()).real)
            if denom <= 1.0e-30:
                continue
            beta = float(np.vdot(ap.ravel(), old_ap.ravel()).real / denom)
            ap -= beta * old_ap
            p -= beta * old_p

        denom = float(np.vdot(ap.ravel(), ap.ravel()).real)
        if denom <= 1.0e-30:
            break
        alpha = float(np.vdot(r.ravel(), ap.ravel()).real / denom)
        x += alpha * p
        r -= alpha * ap
        rel_residual = float(np.linalg.norm(r.ravel()) / b_norm)
        p_basis.append(p)
        ap_basis.append(ap)
        if rel_residual <= eps:
            return x, iteration, rel_residual

    return x, max_iters, rel_residual


def median_seconds(func: Any, repeats: int) -> float:
    samples: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        func()
        samples.append(time.perf_counter() - start)
    return float(np.median(np.array(samples, dtype=np.float64)))


def price_route(
    args: argparse.Namespace,
    condition: str,
    iterations: int,
    segment_length: int,
    relative_solution_error: float,
) -> dict[str, float]:
    orthog_pairs = iterations * max(0, iterations - 1) / 2.0
    residual_scale = min(
        1.0,
        relative_solution_error / max(1.0e-30, args.max_relative_solution_error),
    )
    return {
        "parallel_cr_applies": float(iterations * args.parallel_apply_ratio),
        "krylov_orthogonalization": float(
            orthog_pairs * args.orthogonalization_ratio / max(1, segment_length)
        ),
        "block_residual_verification": float(args.verification_ratio),
        "residual_settlement": float(args.residual_settlement_ratio * residual_scale),
        "pricing_fail_extra": (
            float(args.pricing_fail_extra_ratio) if condition == "pricing_fail" else 0.0
        ),
    }


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    a, rho, condition_proxy = transition_matrix(rng, args.state_dim, condition)
    prior_g, current_g = build_forcing_pair(
        rng, condition, args.segment_length, args.state_dim
    )
    exact_prior = dense_recurrent_adjoint(prior_g, a)
    exact_current = dense_recurrent_adjoint(current_g, a)

    if condition in ("positive", "pricing_fail"):
        x0 = exact_prior + 0.001 * rng.normal(size=exact_prior.shape)
    elif condition == "condition_fail":
        x0 = np.zeros_like(exact_prior)
    else:
        x0 = rng.normal(scale=np.std(exact_prior), size=exact_prior.shape)

    solved, iterations, rel_residual = generalized_conjugate_residual(
        current_g, a, x0, args.solver_eps, args.max_iters
    )
    rel_solution_error = relative_norm(solved - exact_current, exact_current)
    price_components = price_route(
        args,
        condition,
        iterations,
        args.segment_length,
        rel_solution_error,
    )
    priced_work_ratio = float(sum(price_components.values()))
    projected_parallel_speedup = (
        1.0 / priced_work_ratio if priced_work_ratio > 0.0 else math.inf
    )

    dense_seconds = median_seconds(
        lambda: dense_recurrent_adjoint(current_g, a), args.timing_repeats
    )
    solver_seconds = median_seconds(
        lambda: generalized_conjugate_residual(
            current_g, a, x0, args.solver_eps, args.max_iters
        ),
        args.timing_repeats,
    )
    measured_cpu_time_ratio = solver_seconds / max(dense_seconds, 1.0e-30)

    dense_recurrent_matvecs = max(0, args.segment_length - 1)
    iteration_fraction = iterations / max(1, dense_recurrent_matvecs)
    residual_pass = rel_residual <= args.max_relative_residual
    solution_pass = rel_solution_error <= args.max_relative_solution_error
    iteration_pass = iteration_fraction < args.kill_iteration_fraction
    price_pass = priced_work_ratio <= args.max_priced_work_ratio
    passed = bool(residual_pass and solution_pass and iteration_pass and price_pass)
    expected_pass = condition == "positive"
    identifying_correct = passed == expected_pass

    fail_reason = None
    if not passed:
        if not residual_pass:
            fail_reason = "solver_residual_above_certificate"
        elif not solution_pass:
            fail_reason = "solution_error_above_exact_adjoint_tolerance"
        elif not iteration_pass:
            fail_reason = "iterations_reached_dense_fallback_kill_threshold"
        elif not price_pass:
            fail_reason = "priced_route_not_below_dense_backward_budget"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=identifying_correct,
        fail_reason=fail_reason,
        spectral_radius=rho,
        condition_proxy=condition_proxy,
        iterations=iterations,
        dense_recurrent_matvecs=dense_recurrent_matvecs,
        iteration_fraction_of_dense=iteration_fraction,
        relative_residual=float(rel_residual),
        relative_solution_error=float(rel_solution_error),
        priced_work_ratio=priced_work_ratio,
        projected_parallel_speedup=float(projected_parallel_speedup),
        dense_backward_seconds=dense_seconds,
        solver_seconds=solver_seconds,
        measured_cpu_time_ratio=float(measured_cpu_time_ratio),
        price_components=price_components,
    )


def finite_trial(trial: TrialResult) -> bool:
    values = (
        trial.spectral_radius,
        trial.condition_proxy,
        trial.iteration_fraction_of_dense,
        trial.relative_residual,
        trial.relative_solution_error,
        trial.priced_work_ratio,
        trial.projected_parallel_speedup,
        trial.dense_backward_seconds,
        trial.solver_seconds,
        trial.measured_cpu_time_ratio,
    )
    return all(math.isfinite(float(value)) for value in values)


def aggregate(trials: list[TrialResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[TrialResult]] = {}
    for trial in trials:
        grouped.setdefault(trial.condition, []).append(trial)

    out: dict[str, dict[str, float]] = {}
    for condition, rows in grouped.items():
        iterations = np.array([row.iterations for row in rows], dtype=np.float64)
        residuals = np.array([row.relative_residual for row in rows], dtype=np.float64)
        errors = np.array([row.relative_solution_error for row in rows], dtype=np.float64)
        prices = np.array([row.priced_work_ratio for row in rows], dtype=np.float64)
        speedups = np.array(
            [row.projected_parallel_speedup for row in rows], dtype=np.float64
        )
        time_ratios = np.array(
            [row.measured_cpu_time_ratio for row in rows], dtype=np.float64
        )
        out[condition] = {
            "count": float(len(rows)),
            "pass_count": float(sum(row.passed for row in rows)),
            "iterations_min": float(np.min(iterations)),
            "iterations_mean": float(np.mean(iterations)),
            "iterations_max": float(np.max(iterations)),
            "relative_residual_mean": float(np.mean(residuals)),
            "relative_solution_error_mean": float(np.mean(errors)),
            "priced_work_ratio_mean": float(np.mean(prices)),
            "projected_parallel_speedup_mean": float(np.mean(speedups)),
            "measured_cpu_time_ratio_mean": float(np.mean(time_ratios)),
        }
    return out


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.segment_length < 4:
        raise ValueError("--segment-length must be at least 4")
    if args.state_dim < 2:
        raise ValueError("--state-dim must be at least 2")

    trials: list[TrialResult] = []
    for seed_offset in range(args.seed_count):
        seed = args.seed + seed_offset
        for condition in CONDITIONS:
            trials.append(run_trial(args, condition, seed))

    required_passes = {
        "all_metrics_finite": all(finite_trial(trial) for trial in trials),
        "positive_condition_all_pass": all(
            trial.passed for trial in trials if trial.condition == "positive"
        ),
        "positive_condition_residual_certified": all(
            trial.relative_residual <= args.max_relative_residual
            for trial in trials
            if trial.condition == "positive"
        ),
        "positive_condition_exact_adjoint_match": all(
            trial.relative_solution_error <= args.max_relative_solution_error
            for trial in trials
            if trial.condition == "positive"
        ),
        "condition_fail_control_all_fail_iteration": all(
            (not trial.passed)
            and trial.fail_reason == "iterations_reached_dense_fallback_kill_threshold"
            for trial in trials
            if trial.condition == "condition_fail"
        ),
        "pricing_fail_control_all_fail_price": all(
            (not trial.passed)
            and trial.fail_reason == "priced_route_not_below_dense_backward_budget"
            for trial in trials
            if trial.condition == "pricing_fail"
        ),
        "random_control_all_fail": all(
            not trial.passed for trial in trials if trial.condition == "random_control"
        ),
        "all_conditions_identified": all(trial.identifying_correct for trial in trials),
    }
    decision = (
        "PASS_R501_S34_CR_ADJOINT_SOLVER_GATE"
        if all(required_passes.values())
        else "FAIL_R501_S34_CR_ADJOINT_SOLVER_GATE"
    )
    disposition = (
        "synthetic_positive_control_identified_not_training_speed_claim"
        if decision.startswith("PASS_")
        else "cr_adjoint_solver_gate_failed_or_non_identifying"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "CPU synthetic block-adjoint solver gate. It measures residual, "
            "exact-adjoint error, Krylov iterations, pricing, and controls. "
            "It is not a measured heldout CE progress or GPU training-speed claim."
        ),
        "finding": {
            "round": "R501",
            "aspect": "Speed",
            "id": "R501-S34",
            "title": "Conjugate-Residual Adjoint Solver over Sequence Blocks",
        },
        "thresholds": {
            "max_relative_residual": args.max_relative_residual,
            "max_relative_solution_error": args.max_relative_solution_error,
            "max_priced_work_ratio": args.max_priced_work_ratio,
            "kill_iteration_fraction": args.kill_iteration_fraction,
            "required_projected_parallel_speedup": 1.0 / args.max_priced_work_ratio,
        },
        "shape": {
            "seed": args.seed,
            "seed_count": args.seed_count,
            "segment_length": args.segment_length,
            "state_dim": args.state_dim,
            "max_iters": args.max_iters,
            "solver_eps": args.solver_eps,
        },
        "price_defaults": {
            "parallel_apply_ratio": args.parallel_apply_ratio,
            "orthogonalization_ratio": args.orthogonalization_ratio,
            "verification_ratio": args.verification_ratio,
            "residual_settlement_ratio": args.residual_settlement_ratio,
            "pricing_fail_extra_ratio": args.pricing_fail_extra_ratio,
        },
        "required_passes": required_passes,
        "aggregate": aggregate(trials),
        "trials": [asdict(trial) for trial in trials],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    positive = result["aggregate"]["positive"]
    print(
        "[R501_S34_CR_ADJOINT_SOLVER_GATE] "
        f"active=1 decision={decision} disposition={disposition} "
        f"positive_iterations_mean={positive['iterations_mean']:.6f} "
        f"positive_relative_residual_mean={positive['relative_residual_mean']:.6e} "
        f"positive_solution_error_mean={positive['relative_solution_error_mean']:.6e} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_parallel_speedup_mean']:.6f} "
        f"positive_cpu_time_ratio_mean={positive['measured_cpu_time_ratio_mean']:.6f} "
        f"seed_count={args.seed_count} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
