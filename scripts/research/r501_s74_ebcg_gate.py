#!/usr/bin/env python3
"""R501 S74 energy-bounded capability growth gate.

This CPU diagnostic implements the R501 Quality Math 11 finding:
"Energy-Bounded Capability Growth (EBCG)".

The gate constructs a synthetic optimizer step with a global energy
E(theta) = L(theta) + lambda ||theta||^2. A positive case must accept clean CE
updates that decrease energy, reject noisy updates that a baseline optimizer
would accept and spike, and remain under the priced denominator.

Controls exercise the declared risks: stale energy estimates admit destructive
updates, strict monotonicity can reject a non-convex barrier-crossing update
with future payoff, and expensive scoring erases the failed-epoch denominator.

This is a synthetic CPU optimizer-energy diagnostic, not heldout CE progress,
real corpus token efficiency, GPU training-speed evidence, or production v12
integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s74_ebcg_gate.v1"
CONDITIONS = (
    "positive",
    "stale_energy_fail",
    "nonconvex_trap_fail",
    "scoring_cost_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    param_dim: int
    clean_accept_fraction: float
    noisy_reject_fraction: float
    baseline_noisy_energy_delta: float
    ebcg_noisy_energy_delta: float
    ebcg_clean_energy_delta: float
    accepted_loss_delta: float
    max_policy_energy_delta: float
    beneficial_rejection_fraction: float
    future_loss_delta_if_rejected: float
    priced_work_ratio: float
    projected_token_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50174)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--param-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.18)
    parser.add_argument("--energy-lambda", type=float, default=0.04)
    parser.add_argument("--epsilon-growth", type=float, default=0.01)
    parser.add_argument("--min-clean-accept", type=float, default=1.0)
    parser.add_argument("--min-noisy-reject", type=float, default=1.0)
    parser.add_argument("--max-policy-energy-delta", type=float, default=0.0)
    parser.add_argument("--max-beneficial-rejection", type=float, default=0.0)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.90)
    parser.add_argument("--priced-work-ratio", type=float, default=0.82)
    parser.add_argument("--expensive-priced-work-ratio", type=float, default=0.94)
    return parser.parse_args()


def energy(theta: np.ndarray, target: np.ndarray, lam: float) -> float:
    loss = 0.5 * float(np.dot(theta - target, theta - target))
    penalty = lam * float(np.dot(theta, theta))
    return loss + penalty


def loss(theta: np.ndarray, target: np.ndarray) -> float:
    return 0.5 * float(np.dot(theta - target, theta - target))


def gradient(theta: np.ndarray, target: np.ndarray, lam: float) -> np.ndarray:
    return (theta - target) + 2.0 * lam * theta


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    theta = rng.normal(scale=0.55, size=args.param_dim)
    target = rng.normal(scale=0.35, size=args.param_dim)
    before_energy = energy(theta, target, args.energy_lambda)
    before_loss = loss(theta, target)
    exact_grad = gradient(theta, target, args.energy_lambda)

    clean_update = -args.learning_rate * exact_grad
    clean_theta = theta + clean_update
    clean_energy_delta = energy(clean_theta, target, args.energy_lambda) - before_energy
    clean_loss_delta = loss(clean_theta, target) - before_loss

    uphill_grad = -0.85 * exact_grad + rng.normal(scale=0.03, size=args.param_dim)
    noisy_update = -args.learning_rate * uphill_grad
    noisy_theta = theta + noisy_update
    noisy_energy_delta = energy(noisy_theta, target, args.energy_lambda) - before_energy

    clean_accept = clean_energy_delta <= -args.epsilon_growth
    noisy_reject = noisy_energy_delta > -args.epsilon_growth
    beneficial_rejection = 0.0
    future_loss_delta = 0.0

    if condition == "stale_energy_fail":
        estimated_noisy_delta = noisy_energy_delta - abs(noisy_energy_delta) - args.epsilon_growth
        noisy_reject = not (estimated_noisy_delta <= -args.epsilon_growth)
    elif condition == "nonconvex_trap_fail":
        clean_accept = False
        beneficial_rejection = 1.0
        future_loss_delta = -0.35 * before_loss

    ebcg_clean_delta = clean_energy_delta if clean_accept else 0.0
    ebcg_noisy_delta = 0.0 if noisy_reject else noisy_energy_delta
    accepted_loss_delta = clean_loss_delta if clean_accept else 0.0
    max_policy_delta = max(ebcg_clean_delta, ebcg_noisy_delta)
    clean_accept_fraction = 1.0 if clean_accept else 0.0
    noisy_reject_fraction = 1.0 if noisy_reject else 0.0
    priced_work_ratio = (
        args.expensive_priced_work_ratio
        if condition == "scoring_cost_fail"
        else args.priced_work_ratio
    )
    projected_multiplier = 1.0 / priced_work_ratio

    failed: list[str] = []
    if not finite(
        clean_accept_fraction,
        noisy_reject_fraction,
        noisy_energy_delta,
        ebcg_noisy_delta,
        ebcg_clean_delta,
        accepted_loss_delta,
        max_policy_delta,
        beneficial_rejection,
        future_loss_delta,
        priced_work_ratio,
        projected_multiplier,
    ):
        failed.append("non_finite_metric")
    if clean_accept_fraction < args.min_clean_accept:
        failed.append("clean_acceptance")
    if noisy_reject_fraction < args.min_noisy_reject:
        failed.append("noisy_rejection")
    if max_policy_delta > args.max_policy_energy_delta:
        failed.append("energy_monotonicity")
    if beneficial_rejection > args.max_beneficial_rejection:
        failed.append("premature_convergence")
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
        param_dim=args.param_dim,
        clean_accept_fraction=float(clean_accept_fraction),
        noisy_reject_fraction=float(noisy_reject_fraction),
        baseline_noisy_energy_delta=float(noisy_energy_delta),
        ebcg_noisy_energy_delta=float(ebcg_noisy_delta),
        ebcg_clean_energy_delta=float(ebcg_clean_delta),
        accepted_loss_delta=float(accepted_loss_delta),
        max_policy_energy_delta=float(max_policy_delta),
        beneficial_rejection_fraction=float(beneficial_rejection),
        future_loss_delta_if_rejected=float(future_loss_delta),
        priced_work_ratio=float(priced_work_ratio),
        projected_token_efficiency_multiplier=float(projected_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "param_dim",
        "clean_accept_fraction",
        "noisy_reject_fraction",
        "baseline_noisy_energy_delta",
        "ebcg_noisy_energy_delta",
        "ebcg_clean_energy_delta",
        "accepted_loss_delta",
        "max_policy_energy_delta",
        "beneficial_rejection_fraction",
        "future_loss_delta_if_rejected",
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
    stale = [result for result in results if result.condition == "stale_energy_fail"]
    trap = [result for result in results if result.condition == "nonconvex_trap_fail"]
    cost = [result for result in results if result.condition == "scoring_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_accepts_clean": (
            aggregates["positive"]["clean_accept_fraction_mean"] == 1.0
            and aggregates["positive"]["ebcg_clean_energy_delta_max"] <= -0.01
        ),
        "positive_condition_rejects_noisy": (
            aggregates["positive"]["noisy_reject_fraction_mean"] == 1.0
            and aggregates["positive"]["baseline_noisy_energy_delta_min"] > 0.0
            and aggregates["positive"]["ebcg_noisy_energy_delta_max"] == 0.0
        ),
        "positive_condition_priced": (
            aggregates["positive"]["priced_work_ratio_mean"] <= 0.90
        ),
        "stale_control_all_fail_energy": all(
            (not result.passed)
            and result.fail_reason is not None
            and ("noisy_rejection" in result.fail_reason or "energy_monotonicity" in result.fail_reason)
            for result in stale
        ),
        "nonconvex_control_all_fail_premature": all(
            (not result.passed)
            and result.fail_reason is not None
            and "premature_convergence" in result.fail_reason
            for result in trap
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
        "PASS_R501_S74_EBCG_GATE"
        if all(checks.values())
        else "FAIL_R501_S74_EBCG_GATE"
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
            "Synthetic CPU energy-bounded capability-growth optimizer gate "
            "only; not heldout CE progress, not real corpus token efficiency, "
            "not GPU training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S74_EBCG_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_clean_accept_fraction_mean={positive['clean_accept_fraction_mean']:.6f} "
        f"positive_noisy_reject_fraction_mean={positive['noisy_reject_fraction_mean']:.6f} "
        f"positive_baseline_noisy_energy_delta_min={positive['baseline_noisy_energy_delta_min']:.6f} "
        f"positive_ebcg_clean_energy_delta_max={positive['ebcg_clean_energy_delta_max']:.6f} "
        f"positive_ebcg_noisy_energy_delta_max={positive['ebcg_noisy_energy_delta_max']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_token_efficiency_multiplier_mean={positive['projected_token_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
