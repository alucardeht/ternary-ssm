#!/usr/bin/env python3
"""R501 S48 phase-area action integrator gate.

This CPU diagnostic implements the R501 Speed Math 15 finding:
"Phase-Area Action Integrator (Cotangent Potential Field)".

The gate models a local parameter-gradient field over a macro block. For a
conservative quadratic potential, it integrates the macro update analytically
and verifies a descent lower bound instead of materializing every micro-step
gradient. The dense baseline is an explicit micro-step Euler trajectory.

The finding survives only when the macro update matches the micro trajectory
within residual debt, the field curl is small, the certified lower bound remains
above target, and macro integration plus verification is cheaper than dense
micro-step credit.

Controls exercise the declared kills: high curl makes path integration accrue
residual debt, full verification recovers dense micro-step cost, and an
over-conservative curvature/debt bound invalidates the descent certificate.

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


SCHEMA = "r501_s48_phase_area_integrator_gate.v1"
CONDITIONS = (
    "positive",
    "curl_residual_fail",
    "verifier_cost_fail",
    "lower_bound_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    dim: int
    micro_steps: int
    verify_samples: int
    curl_ratio: float
    macro_update_relative_error: float
    residual_debt_ratio: float
    interaction_debt_ratio: float
    actual_potential_decrease: float
    lower_bound: float
    target: float
    lower_bound_ratio: float
    lower_bound_margin: float
    macro_integral_cost_ratio: float
    verify_cost_ratio: float
    priced_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50148)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--micro-steps", type=int, default=64)
    parser.add_argument("--macro-time", type=float, default=0.45)
    parser.add_argument("--verify-samples", type=int, default=8)
    parser.add_argument("--max-curl-ratio", type=float, default=0.05)
    parser.add_argument("--max-update-error", type=float, default=0.03)
    parser.add_argument("--max-residual-debt-ratio", type=float, default=0.05)
    parser.add_argument("--min-lower-bound-ratio", type=float, default=0.25)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.50)
    parser.add_argument("--macro-integral-cost-ratio", type=float, default=0.10)
    parser.add_argument("--verifier-unit-cost-ratio", type=float, default=0.20)
    parser.add_argument("--high-verifier-unit-cost-ratio", type=float, default=1.00)
    parser.add_argument("--curl-strength", type=float, default=0.90)
    parser.add_argument("--lower-bound-beta-multiplier", type=float, default=1.0)
    parser.add_argument("--fail-beta-multiplier", type=float, default=7.0)
    parser.add_argument("--target-decrease-fraction", type=float, default=0.25)
    parser.add_argument("--curl-debt-multiplier", type=float, default=1.50)
    return parser.parse_args()


def symmetric_positive_matrix(
    rng: np.random.Generator,
    dim: int,
) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    eig = np.linspace(0.18, 0.92, dim, dtype=np.float64)
    return q @ np.diag(eig) @ q.T


def skew_matrix(rng: np.random.Generator, dim: int) -> np.ndarray:
    raw = rng.normal(size=(dim, dim))
    skew = raw - raw.T
    norm = max(float(np.linalg.norm(skew, ord="fro")), 1.0e-30)
    return skew / norm


def matrix_exponential_symmetric(matrix: np.ndarray, time: float) -> np.ndarray:
    eig, basis = np.linalg.eigh(matrix)
    return basis @ np.diag(np.exp(-time * eig)) @ basis.T


def micro_integrate(theta: np.ndarray, field_matrix: np.ndarray, total_time: float, steps: int) -> np.ndarray:
    dt = total_time / float(steps)
    out = theta.copy()
    for _ in range(steps):
        out = out - dt * (field_matrix @ out)
    return out


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    q = symmetric_positive_matrix(rng, args.dim)
    field_matrix = q.copy()
    verify_samples = args.verify_samples
    verifier_unit_cost = args.verifier_unit_cost_ratio
    beta_multiplier = args.lower_bound_beta_multiplier

    if condition == "curl_residual_fail":
        field_matrix = q + args.curl_strength * skew_matrix(rng, args.dim)
    elif condition == "verifier_cost_fail":
        verify_samples = args.micro_steps
        verifier_unit_cost = args.high_verifier_unit_cost_ratio
    elif condition == "lower_bound_fail":
        beta_multiplier = args.fail_beta_multiplier

    theta0 = rng.normal(size=args.dim)
    theta0 /= max(float(np.linalg.norm(theta0)), 1.0e-30)
    theta_micro = micro_integrate(theta0, field_matrix, args.macro_time, args.micro_steps)
    theta_macro = matrix_exponential_symmetric(q, args.macro_time) @ theta0
    macro_update = theta_macro - theta0
    micro_update = theta_micro - theta0
    update_error = float(
        np.linalg.norm(macro_update - micro_update)
        / max(float(np.linalg.norm(micro_update)), 1.0e-30)
    )

    curl = field_matrix - field_matrix.T
    curl_ratio = float(
        np.linalg.norm(curl, ord="fro")
        / max(float(np.linalg.norm(field_matrix, ord="fro")), 1.0e-30)
    )
    potential0 = 0.5 * float(theta0 @ q @ theta0)
    potential1 = 0.5 * float(theta_macro @ q @ theta_macro)
    actual_decrease = max(potential0 - potential1, 1.0e-30)
    gradient0 = q @ theta0
    beta = float(np.max(np.linalg.eigvalsh(q)) * beta_multiplier)
    residual_debt_ratio = update_error
    interaction_debt_ratio = curl_ratio * args.curl_debt_multiplier
    residual_debt = residual_debt_ratio * actual_decrease
    interaction_debt = interaction_debt_ratio * actual_decrease
    lower_bound = (
        -float(gradient0 @ macro_update)
        - 0.5 * beta * float(macro_update @ macro_update)
        - interaction_debt
        - residual_debt
    )
    target = args.target_decrease_fraction * actual_decrease
    lower_bound_ratio = lower_bound / actual_decrease
    lower_bound_margin = lower_bound - target

    verify_cost_ratio = verifier_unit_cost * (float(verify_samples) / float(args.micro_steps))
    priced_work_ratio = args.macro_integral_cost_ratio + verify_cost_ratio
    projected_speedup = 1.0 / priced_work_ratio if priced_work_ratio > 0.0 else math.inf

    curl_pass = curl_ratio <= args.max_curl_ratio
    update_pass = update_error <= args.max_update_error
    debt_pass = residual_debt_ratio <= args.max_residual_debt_ratio
    lower_bound_pass = lower_bound_ratio >= args.min_lower_bound_ratio and lower_bound_margin >= 0.0
    price_pass = priced_work_ratio <= args.max_priced_work_ratio
    passed = bool(curl_pass and update_pass and debt_pass and lower_bound_pass and price_pass)
    expected_pass = condition == "positive"
    identifying_correct = passed == expected_pass

    fail_reason = None
    if not passed:
        if condition == "verifier_cost_fail" and not price_pass:
            fail_reason = "macro_verification_recovers_micro_step_cost"
        elif condition == "lower_bound_fail" and not lower_bound_pass:
            fail_reason = "descent_lower_bound_below_target"
        elif not curl_pass or not update_pass or not debt_pass:
            fail_reason = "nonconservative_curl_accumulates_residual_debt"
        elif not lower_bound_pass:
            fail_reason = "descent_lower_bound_below_target"
        elif not price_pass:
            fail_reason = "macro_verification_recovers_micro_step_cost"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=identifying_correct,
        fail_reason=fail_reason,
        dim=args.dim,
        micro_steps=args.micro_steps,
        verify_samples=verify_samples,
        curl_ratio=curl_ratio,
        macro_update_relative_error=update_error,
        residual_debt_ratio=residual_debt_ratio,
        interaction_debt_ratio=interaction_debt_ratio,
        actual_potential_decrease=actual_decrease,
        lower_bound=lower_bound,
        target=target,
        lower_bound_ratio=lower_bound_ratio,
        lower_bound_margin=lower_bound_margin,
        macro_integral_cost_ratio=float(args.macro_integral_cost_ratio),
        verify_cost_ratio=float(verify_cost_ratio),
        priced_work_ratio=float(priced_work_ratio),
        projected_speedup=float(projected_speedup),
    )


def finite_trial(trial: TrialResult) -> bool:
    values = (
        trial.curl_ratio,
        trial.macro_update_relative_error,
        trial.residual_debt_ratio,
        trial.interaction_debt_ratio,
        trial.actual_potential_decrease,
        trial.lower_bound,
        trial.target,
        trial.lower_bound_ratio,
        trial.lower_bound_margin,
        trial.macro_integral_cost_ratio,
        trial.verify_cost_ratio,
        trial.priced_work_ratio,
        trial.projected_speedup,
    )
    return all(math.isfinite(float(value)) for value in values)


def aggregate(trials: list[TrialResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[TrialResult]] = {}
    for trial in trials:
        grouped.setdefault(trial.condition, []).append(trial)
    out: dict[str, dict[str, float]] = {}
    for condition, rows in grouped.items():
        out[condition] = {
            "count": float(len(rows)),
            "pass_count": float(sum(row.passed for row in rows)),
            "curl_ratio_mean": float(np.mean([row.curl_ratio for row in rows])),
            "curl_ratio_max": float(np.max([row.curl_ratio for row in rows])),
            "macro_update_relative_error_mean": float(
                np.mean([row.macro_update_relative_error for row in rows])
            ),
            "macro_update_relative_error_max": float(
                np.max([row.macro_update_relative_error for row in rows])
            ),
            "residual_debt_ratio_mean": float(
                np.mean([row.residual_debt_ratio for row in rows])
            ),
            "interaction_debt_ratio_mean": float(
                np.mean([row.interaction_debt_ratio for row in rows])
            ),
            "actual_potential_decrease_mean": float(
                np.mean([row.actual_potential_decrease for row in rows])
            ),
            "lower_bound_mean": float(np.mean([row.lower_bound for row in rows])),
            "target_mean": float(np.mean([row.target for row in rows])),
            "lower_bound_ratio_mean": float(
                np.mean([row.lower_bound_ratio for row in rows])
            ),
            "lower_bound_margin_mean": float(
                np.mean([row.lower_bound_margin for row in rows])
            ),
            "verify_samples_mean": float(np.mean([row.verify_samples for row in rows])),
            "verify_cost_ratio_mean": float(
                np.mean([row.verify_cost_ratio for row in rows])
            ),
            "priced_work_ratio_mean": float(np.mean([row.priced_work_ratio for row in rows])),
            "projected_speedup_mean": float(
                np.mean([row.projected_speedup for row in rows])
            ),
        }
    return out


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.dim < 2:
        raise ValueError("--dim must be at least 2")
    if args.micro_steps < 2:
        raise ValueError("--micro-steps must be at least 2")
    if args.verify_samples < 1:
        raise ValueError("--verify-samples must be at least 1")

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
        "positive_condition_bound_and_price_hold": all(
            trial.lower_bound_ratio >= args.min_lower_bound_ratio
            and trial.priced_work_ratio <= args.max_priced_work_ratio
            for trial in trials
            if trial.condition == "positive"
        ),
        "curl_control_all_fail_residual_debt": all(
            (not trial.passed)
            and trial.fail_reason == "nonconservative_curl_accumulates_residual_debt"
            for trial in trials
            if trial.condition == "curl_residual_fail"
        ),
        "verifier_cost_control_all_fail_price": all(
            (not trial.passed)
            and trial.fail_reason == "macro_verification_recovers_micro_step_cost"
            for trial in trials
            if trial.condition == "verifier_cost_fail"
        ),
        "lower_bound_control_all_fail_certificate": all(
            (not trial.passed)
            and trial.fail_reason == "descent_lower_bound_below_target"
            for trial in trials
            if trial.condition == "lower_bound_fail"
        ),
        "all_conditions_identified": all(trial.identifying_correct for trial in trials),
    }
    decision = (
        "PASS_R501_S48_PHASE_AREA_INTEGRATOR_GATE"
        if all(required_passes.values())
        else "FAIL_R501_S48_PHASE_AREA_INTEGRATOR_GATE"
    )
    disposition = (
        "synthetic_positive_control_identified_not_training_speed_claim"
        if decision.startswith("PASS_")
        else "phase_area_integrator_gate_failed_or_non_identifying"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "CPU synthetic phase-area macro-action integrator gate. It checks "
            "conservative-field curl, macro-vs-micro update residual debt, "
            "descent lower-bound margin, and priced macro integration plus "
            "verification work. It is not heldout CE progress or GPU "
            "training-speed evidence."
        ),
        "finding": {
            "round": "R501",
            "aspect": "Speed",
            "id": "R501-S48",
            "title": "Phase-Area Action Integrator",
        },
        "thresholds": {
            "max_curl_ratio": args.max_curl_ratio,
            "max_update_error": args.max_update_error,
            "max_residual_debt_ratio": args.max_residual_debt_ratio,
            "min_lower_bound_ratio": args.min_lower_bound_ratio,
            "max_priced_work_ratio": args.max_priced_work_ratio,
            "target_decrease_fraction": args.target_decrease_fraction,
        },
        "shape": {
            "seed": args.seed,
            "seed_count": args.seed_count,
            "dim": args.dim,
            "micro_steps": args.micro_steps,
            "macro_time": args.macro_time,
            "verify_samples": args.verify_samples,
        },
        "required_passes": required_passes,
        "aggregate": aggregate(trials),
        "trials": [asdict(trial) for trial in trials],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    positive = result["aggregate"]["positive"]
    print(
        "[R501_S48_PHASE_AREA_INTEGRATOR_GATE] "
        f"active=1 decision={decision} disposition={disposition} "
        f"positive_curl_ratio_mean={positive['curl_ratio_mean']:.6f} "
        f"positive_macro_update_error_mean={positive['macro_update_relative_error_mean']:.6f} "
        f"positive_lower_bound_ratio_mean={positive['lower_bound_ratio_mean']:.6f} "
        f"positive_residual_debt_ratio_mean={positive['residual_debt_ratio_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
