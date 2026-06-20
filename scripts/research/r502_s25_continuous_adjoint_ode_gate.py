#!/usr/bin/env python3
"""R502-S25 Continuous-Time Adjoint ODE Solver gate.

This synthetic CPU gate checks whether a continuous-time adjoint ODE solver can
use adaptive steps larger than one discrete token interval on smooth gradient
fields while preserving a fine-step adjoint reference and reducing priced work.
Controls cover oscillatory gradients, stiff dynamics, loose tolerances,
expensive solver steps, and discontinuous gradient fields.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


CONDITIONS = [
    "positive",
    "oscillatory_fail",
    "stiff_fail",
    "loose_tolerance_fail",
    "expensive_solver_fail",
    "discontinuity_fail",
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
    accepted_steps: int
    rejected_steps: int
    mean_step_size: float
    min_step_size: float
    max_step_size: float
    relative_adjoint_error: float
    max_abs_adjoint_error: float
    ode_residual_rms: float
    descent_cosine: float
    ode_step_cost_ratio: float
    priced_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=50225)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--horizon-len", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--reference-dt", type=float, default=0.25)
    parser.add_argument("--max-step", type=float, default=8.0)
    parser.add_argument("--min-step", type=float, default=0.125)
    parser.add_argument("--tolerance", type=float, default=2e-4)
    parser.add_argument("--loose-tolerance", type=float, default=0.08)
    parser.add_argument("--ode-step-cost-ratio", type=float, default=1.7)
    parser.add_argument("--expensive-ode-step-cost-ratio", type=float, default=8.0)
    parser.add_argument("--max-relative-adjoint-error", type=float, default=0.035)
    parser.add_argument("--max-ode-residual-rms", type=float, default=0.005)
    parser.add_argument("--min-descent-cosine", type=float, default=0.995)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.90)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def make_dynamics(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, Callable[[float], np.ndarray]]:
    if condition == "stiff_fail":
        decay = rng.uniform(0.22, 0.35, size=args.state_dim)
    else:
        decay = rng.uniform(0.002, 0.012, size=args.state_dim)
    amp1 = rng.normal(scale=0.08, size=args.state_dim)
    amp2 = rng.normal(scale=0.05, size=args.state_dim)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=args.state_dim)
    bias = rng.normal(scale=0.025, size=args.state_dim)

    def smooth_grad(t: float) -> np.ndarray:
        x = t / args.horizon_len
        return (
            bias
            + amp1 * np.sin((2.0 * np.pi * x) + phase)
            + amp2 * np.cos((4.0 * np.pi * x) + 0.5 * phase)
        )

    def oscillatory_grad(t: float) -> np.ndarray:
        return smooth_grad(t) + 0.14 * np.sin((2.0 * np.pi * t / 2.5) + phase)

    pulse_centers = rng.uniform(0.15 * args.horizon_len, 0.85 * args.horizon_len, size=8)
    pulse_amp = rng.normal(scale=1.8, size=(8, args.state_dim))

    def discontinuous_grad(t: float) -> np.ndarray:
        value = smooth_grad(t).copy()
        for center, amp in zip(pulse_centers, pulse_amp):
            if abs(t - center) <= 0.45:
                value += amp
        return value

    if condition == "oscillatory_fail" or condition == "loose_tolerance_fail":
        return decay, oscillatory_grad
    if condition == "discontinuity_fail":
        return decay, discontinuous_grad
    return decay, smooth_grad


def rhs(decay: np.ndarray, grad_fn: Callable[[float], np.ndarray], t: float, lam: np.ndarray) -> np.ndarray:
    return (-decay * lam) - grad_fn(t)


def rk4_step(
    decay: np.ndarray,
    grad_fn: Callable[[float], np.ndarray],
    t: float,
    lam: np.ndarray,
    h: float,
) -> np.ndarray:
    k1 = rhs(decay, grad_fn, t, lam)
    k2 = rhs(decay, grad_fn, t + 0.5 * h, lam + 0.5 * h * k1)
    k3 = rhs(decay, grad_fn, t + 0.5 * h, lam + 0.5 * h * k2)
    k4 = rhs(decay, grad_fn, t + h, lam + h * k3)
    return lam + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_fixed(
    decay: np.ndarray,
    grad_fn: Callable[[float], np.ndarray],
    horizon_len: int,
    dt: float,
) -> np.ndarray:
    t = float(horizon_len)
    lam = np.zeros(decay.shape[0], dtype=np.float64)
    while t > 0.0:
        h = -min(dt, t)
        lam = rk4_step(decay, grad_fn, t, lam, h)
        t += h
    return lam


def integrate_adaptive(
    decay: np.ndarray,
    grad_fn: Callable[[float], np.ndarray],
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, int, int, list[float], list[float]]:
    tolerance = args.loose_tolerance if condition == "loose_tolerance_fail" else args.tolerance
    max_step = args.max_step * 4.0 if condition == "loose_tolerance_fail" else args.max_step
    t = float(args.horizon_len)
    lam = np.zeros(args.state_dim, dtype=np.float64)
    h_abs = max_step
    accepted = 0
    rejected = 0
    accepted_steps: list[float] = []
    accepted_errors: list[float] = []
    while t > 0.0:
        h_abs = min(h_abs, t)
        h = -h_abs
        full = rk4_step(decay, grad_fn, t, lam, h)
        half = rk4_step(decay, grad_fn, t, lam, 0.5 * h)
        two_half = rk4_step(decay, grad_fn, t + 0.5 * h, half, 0.5 * h)
        err = float(np.linalg.norm(two_half - full) / max(np.linalg.norm(two_half), 1e-12))
        if err <= tolerance or h_abs <= args.min_step:
            lam = two_half
            t += h
            accepted += 1
            accepted_steps.append(h_abs)
            accepted_errors.append(err)
            if err < tolerance * 0.08:
                h_abs = min(max_step, h_abs * 1.7)
            elif err < tolerance * 0.45:
                h_abs = min(max_step, h_abs * 1.25)
        else:
            rejected += 1
            h_abs = max(args.min_step, h_abs * 0.5)
    return lam, accepted, rejected, accepted_steps, accepted_errors


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    decay, grad_fn = make_dynamics(rng, args, condition)
    reference = integrate_fixed(decay, grad_fn, args.horizon_len, args.reference_dt)
    candidate, accepted, rejected, step_sizes, accepted_errors = integrate_adaptive(
        decay,
        grad_fn,
        args,
        condition,
    )
    delta = candidate - reference
    reference_norm = float(np.linalg.norm(reference))
    candidate_norm = float(np.linalg.norm(candidate))
    rel_error = float(np.linalg.norm(delta) / max(reference_norm, 1e-12))
    max_abs = float(np.max(np.abs(delta)))
    cosine = float(np.vdot(candidate, reference).real / max(candidate_norm * reference_norm, 1e-12))
    residual = float(np.max(accepted_errors)) if accepted_errors else float("inf")
    step_cost = (
        args.expensive_ode_step_cost_ratio
        if condition == "expensive_solver_fail"
        else args.ode_step_cost_ratio
    )
    work = (accepted + rejected) * step_cost
    price = work / args.horizon_len
    speedup = args.horizon_len / max(work, 1e-12)
    failed: list[str] = []
    if not np.isfinite([rel_error, max_abs, cosine, residual, work, price, speedup]).all():
        failed.append("non_finite_metric")
    if rel_error > args.max_relative_adjoint_error:
        failed.append("adjoint_error")
    if residual > args.max_ode_residual_rms:
        failed.append("ode_residual")
    if cosine < args.min_descent_cosine:
        failed.append("descent_cosine")
    if price > args.max_priced_work_ratio:
        failed.append("priced_work")

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
        accepted_steps=accepted,
        rejected_steps=rejected,
        mean_step_size=float(np.mean(step_sizes)),
        min_step_size=float(np.min(step_sizes)),
        max_step_size=float(np.max(step_sizes)),
        relative_adjoint_error=rel_error,
        max_abs_adjoint_error=max_abs,
        ode_residual_rms=residual,
        descent_cosine=cosine,
        ode_step_cost_ratio=step_cost,
        priced_work_ratio=price,
        projected_speedup=speedup,
    )


NUMERIC_FIELDS = [
    "horizon_len",
    "state_dim",
    "accepted_steps",
    "rejected_steps",
    "mean_step_size",
    "min_step_size",
    "max_step_size",
    "relative_adjoint_error",
    "max_abs_adjoint_error",
    "ode_residual_rms",
    "descent_cosine",
    "ode_step_cost_ratio",
    "priced_work_ratio",
    "projected_speedup",
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
        "positive_condition_error_and_price": (
            aggregates["positive"]["relative_adjoint_error_max"] <= 0.035
            and aggregates["positive"]["ode_residual_rms_max"] <= 0.04
            and aggregates["positive"]["descent_cosine_min"] >= 0.995
            and aggregates["positive"]["priced_work_ratio_max"] <= 0.90
            and aggregates["positive"]["mean_step_size_mean"] > 1.0
        ),
        "oscillatory_control_all_fail_price_or_error": all(
            (not result.passed)
            and (
                "priced_work" in result.fail_reason
                or "adjoint_error" in result.fail_reason
                or "descent_cosine" in result.fail_reason
            )
            for result in by_condition["oscillatory_fail"]
        ),
        "stiff_control_all_fail_price_or_error": all(
            (not result.passed)
            and (
                "priced_work" in result.fail_reason
                or "adjoint_error" in result.fail_reason
                or "ode_residual" in result.fail_reason
            )
            for result in by_condition["stiff_fail"]
        ),
        "loose_tolerance_control_all_fail_error": all(
            (not result.passed)
            and (
                "adjoint_error" in result.fail_reason
                or "ode_residual" in result.fail_reason
                or "descent_cosine" in result.fail_reason
            )
            for result in by_condition["loose_tolerance_fail"]
        ),
        "expensive_solver_control_all_fail_price": all(
            (not result.passed) and "priced_work" in result.fail_reason
            for result in by_condition["expensive_solver_fail"]
        ),
        "discontinuity_control_all_fail_price_or_error": all(
            (not result.passed)
            and (
                "priced_work" in result.fail_reason
                or "adjoint_error" in result.fail_reason
                or "descent_cosine" in result.fail_reason
            )
            for result in by_condition["discontinuity_fail"]
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
    decision = "PASS_R502_S25_CTAO_GATE" if all(checks.values()) else "FAIL_R502_S25_CTAO_GATE"
    payload = {
        "round": "R502",
        "canonical_id": "R502-S25",
        "aspect": "Speed",
        "finding": "Continuous-Time Adjoint ODE Solver",
        "decision": decision,
        "disposition": "synthetic_positive_control_identified_not_training_speed_claim",
        "args": vars(args) | {"output": str(args.output) if args.output else None},
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU continuous-time adjoint ODE gate only; not heldout "
            "CE progress, not real corpus token efficiency, not GPU training "
            "speed, and not production v12 integration."
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_S25_CTAO_GATE] "
        f"active=1 decision={decision} "
        f"disposition={payload['disposition']} "
        f"positive_accepted_steps_mean={positive['accepted_steps_mean']:.12f} "
        f"positive_mean_step_size_mean={positive['mean_step_size_mean']:.12f} "
        f"positive_relative_adjoint_error_max={positive['relative_adjoint_error_max']:.12f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.12f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.12f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
