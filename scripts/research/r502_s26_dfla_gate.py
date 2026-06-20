#!/usr/bin/env python3
"""R502-S26 Discrete Forward-Loss Annealing gate.

This synthetic CPU gate checks whether random ternary perturbations evaluated
by forward loss can produce monotonic descent in a low-active-dimensional regime
with cheaper priced work than a recurrent backward pass. Controls cover the
high-dimensional random-search kill case, noisy forward acceptance, forward
cost, too few samples, and near-optimal flat loss.
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
    "high_dimension_fail",
    "noisy_forward_fail",
    "forward_cost_fail",
    "low_sample_fail",
    "flat_loss_fail",
]

TERNARY_VALUES = np.array([-1, 0, 1], dtype=np.int8)


@dataclass
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str
    param_count: int
    active_dim: int
    anneal_steps: int
    samples_per_step: int
    proposal_size: int
    accepted_updates: int
    monotonic_violation_count: int
    initial_loss: float
    final_loss: float
    loss_drop_fraction: float
    total_forward_samples: int
    backward_cost_units: float
    forward_cost_units: float
    priced_work_ratio: float
    descent_per_paid_work: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=50226)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--param-count", type=int, default=128)
    parser.add_argument("--active-dim", type=int, default=16)
    parser.add_argument("--anneal-steps", type=int, default=12)
    parser.add_argument("--samples-per-step", type=int, default=32)
    parser.add_argument("--proposal-size", type=int, default=1)
    parser.add_argument("--backward-cost-multiplier", type=float, default=16.0)
    parser.add_argument("--forward-cost-per-sample", type=float, default=1.0)
    parser.add_argument("--expensive-forward-cost-per-sample", type=float, default=8.0)
    parser.add_argument("--min-loss-drop-fraction", type=float, default=0.25)
    parser.add_argument("--min-accepted-updates", type=int, default=4)
    parser.add_argument("--min-descent-per-paid-work", type=float, default=1.0)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.90)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def choose_other_value(rng: np.random.Generator, current: np.ndarray) -> np.ndarray:
    output = current.copy()
    for value in TERNARY_VALUES:
        mask = current == value
        alternatives = TERNARY_VALUES[TERNARY_VALUES != value]
        output[mask] = rng.choice(alternatives, size=int(np.sum(mask)))
    return output


def make_fixture(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if condition == "high_dimension_fail":
        param_count = 4096
        active_dim = 4096
    else:
        param_count = args.param_count
        active_dim = args.active_dim
    target = rng.choice(TERNARY_VALUES, size=param_count)
    weights = target.copy()
    active = rng.choice(param_count, size=active_dim, replace=False)
    mismatch_fraction = 0.04 if condition == "flat_loss_fail" else 0.55
    corrupt_count = max(1, int(round(active_dim * mismatch_fraction)))
    corrupt = rng.choice(active, size=corrupt_count, replace=False)
    weights[corrupt] = choose_other_value(rng, weights[corrupt])
    return weights, target, active


def true_loss(weights: np.ndarray, target: np.ndarray, active: np.ndarray) -> float:
    delta = weights[active].astype(np.float64) - target[active].astype(np.float64)
    return float(0.5 * np.vdot(delta, delta).real)


def propose(
    rng: np.random.Generator,
    weights: np.ndarray,
    active: np.ndarray,
    proposal_size: int,
) -> np.ndarray:
    candidate = weights.copy()
    chosen = rng.choice(active, size=min(proposal_size, len(active)), replace=False)
    candidate[chosen] = choose_other_value(rng, candidate[chosen])
    return candidate


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    weights, target, active = make_fixture(rng, args, condition)
    param_count = len(weights)
    active_dim = len(active)
    samples_per_step = 1 if condition == "low_sample_fail" else args.samples_per_step
    if condition == "high_dimension_fail":
        proposal_size = 64
    elif condition == "low_sample_fail":
        proposal_size = max(1, active_dim // 2)
    else:
        proposal_size = args.proposal_size
    forward_cost = (
        args.expensive_forward_cost_per_sample
        if condition == "forward_cost_fail"
        else args.forward_cost_per_sample
    )
    noise_scale = 1.2 if condition == "noisy_forward_fail" else 0.0
    current_true = true_loss(weights, target, active)
    initial_loss = current_true
    accepted = 0
    monotonic_violations = 0
    total_samples = 0
    for _ in range(args.anneal_steps):
        best_candidate = None
        best_observed = current_true + rng.normal(scale=noise_scale)
        for _sample in range(samples_per_step):
            total_samples += 1
            candidate = propose(rng, weights, active, proposal_size)
            candidate_true = true_loss(candidate, target, active)
            observed = candidate_true + rng.normal(scale=noise_scale)
            if observed < best_observed:
                best_observed = observed
                best_candidate = candidate
        if best_candidate is not None:
            new_true = true_loss(best_candidate, target, active)
            if new_true > current_true + 1e-12:
                monotonic_violations += 1
            weights = best_candidate
            current_true = new_true
            accepted += 1

    final_loss = current_true
    backward_cost = param_count * args.backward_cost_multiplier
    forward_units = total_samples * forward_cost
    price = forward_units / max(backward_cost, 1e-12)
    loss_drop_fraction = (initial_loss - final_loss) / max(initial_loss, 1e-12)
    descent_per_paid = loss_drop_fraction / max(price, 1e-12)

    failed: list[str] = []
    if not np.isfinite([initial_loss, final_loss, price, loss_drop_fraction, descent_per_paid]).all():
        failed.append("non_finite_metric")
    if monotonic_violations > 0:
        failed.append("monotonic_violation")
    if loss_drop_fraction < args.min_loss_drop_fraction:
        failed.append("insufficient_descent")
    if accepted < args.min_accepted_updates:
        failed.append("too_few_accepts")
    if descent_per_paid < args.min_descent_per_paid_work:
        failed.append("descent_per_paid_work")
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
        param_count=param_count,
        active_dim=active_dim,
        anneal_steps=args.anneal_steps,
        samples_per_step=samples_per_step,
        proposal_size=proposal_size,
        accepted_updates=accepted,
        monotonic_violation_count=monotonic_violations,
        initial_loss=initial_loss,
        final_loss=final_loss,
        loss_drop_fraction=loss_drop_fraction,
        total_forward_samples=total_samples,
        backward_cost_units=backward_cost,
        forward_cost_units=forward_units,
        priced_work_ratio=price,
        descent_per_paid_work=descent_per_paid,
    )


NUMERIC_FIELDS = [
    "param_count",
    "active_dim",
    "anneal_steps",
    "samples_per_step",
    "proposal_size",
    "accepted_updates",
    "monotonic_violation_count",
    "initial_loss",
    "final_loss",
    "loss_drop_fraction",
    "total_forward_samples",
    "backward_cost_units",
    "forward_cost_units",
    "priced_work_ratio",
    "descent_per_paid_work",
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
        "positive_condition_monotone_descent_and_price": (
            aggregates["positive"]["monotonic_violation_count_max"] == 0.0
            and aggregates["positive"]["loss_drop_fraction_min"] >= 0.25
            and aggregates["positive"]["accepted_updates_min"] >= 4.0
            and aggregates["positive"]["priced_work_ratio_max"] <= 0.90
            and aggregates["positive"]["descent_per_paid_work_min"] >= 1.0
        ),
        "high_dimension_control_all_fail_descent": all(
            (not result.passed)
            and (
                "insufficient_descent" in result.fail_reason
                or "too_few_accepts" in result.fail_reason
                or "descent_per_paid_work" in result.fail_reason
            )
            for result in by_condition["high_dimension_fail"]
        ),
        "noisy_forward_control_all_fail_monotonic": all(
            (not result.passed) and "monotonic_violation" in result.fail_reason
            for result in by_condition["noisy_forward_fail"]
        ),
        "forward_cost_control_all_fail_price": all(
            (not result.passed) and "priced_work" in result.fail_reason
            for result in by_condition["forward_cost_fail"]
        ),
        "low_sample_control_all_fail_descent": all(
            (not result.passed)
            and (
                "insufficient_descent" in result.fail_reason
                or "too_few_accepts" in result.fail_reason
                or "descent_per_paid_work" in result.fail_reason
            )
            for result in by_condition["low_sample_fail"]
        ),
        "flat_loss_control_all_fail_descent": all(
            (not result.passed)
            and (
                "insufficient_descent" in result.fail_reason
                or "too_few_accepts" in result.fail_reason
                or "descent_per_paid_work" in result.fail_reason
            )
            for result in by_condition["flat_loss_fail"]
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
    decision = "PASS_R502_S26_DFLA_GATE" if all(checks.values()) else "FAIL_R502_S26_DFLA_GATE"
    payload = {
        "round": "R502",
        "canonical_id": "R502-S26",
        "aspect": "Speed",
        "finding": "Discrete Forward-Loss Annealing (DFLA)",
        "decision": decision,
        "disposition": "synthetic_positive_control_identified_not_training_speed_claim",
        "args": vars(args) | {"output": str(args.output) if args.output else None},
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU discrete forward-loss annealing gate only; not "
            "heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_S26_DFLA_GATE] "
        f"active=1 decision={decision} "
        f"disposition={payload['disposition']} "
        f"positive_accepted_updates_min={positive['accepted_updates_min']:.0f} "
        f"positive_loss_drop_fraction_min={positive['loss_drop_fraction_min']:.12f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.12f} "
        f"positive_descent_per_paid_work_min={positive['descent_per_paid_work_min']:.12f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
