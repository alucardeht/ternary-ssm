#!/usr/bin/env python3
"""R501 S82 Fisher-weighted ternary commitment escrow gate.

This CPU diagnostic implements the R501 Quality Paradigm 4 finding:
"Fisher-Weighted Ternary Commitment Escrow (FWTCE)".

The gate constructs ternary weights with a diagonal Fisher estimate for prior
capabilities and a future update that proposes ternary flips. The candidate
blocks a proposed flip when Fisher_i * (delta w_i)^2 exceeds a repair budget.
A positive case must prevent high-Fisher capability damage while preserving
most low-risk future-update progress under a priced Fisher ledger.

Controls exercise the declared risks: Fisher can be uninformative, stale, too
strict, too permissive, or too expensive to maintain.

This is a synthetic CPU Fisher-escrow diagnostic, not heldout CE progress, real
corpus token efficiency, GPU training-speed evidence, or production v12
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


SCHEMA = "r501_s82_fwtce_gate.v1"
CONDITIONS = (
    "positive",
    "no_fisher_signal_fail",
    "overlock_fail",
    "underlock_fail",
    "stale_fisher_fail",
    "repair_cost_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    param_count: int
    proposed_flip_fraction: float
    locked_flip_fraction: float
    high_fisher_flip_block_fraction: float
    unsafe_high_fisher_leak_fraction: float
    allowed_low_fisher_flip_fraction: float
    baseline_prior_loss_delta: float
    escrow_prior_loss_delta: float
    prior_loss_reduction_fraction: float
    new_task_progress_retention: float
    ledger_work_units: float
    allowed_flip_work_units: float
    priced_work_ratio: float
    projected_capability_retention_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50182)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--param-count", type=int, default=512)
    parser.add_argument("--high-fisher-fraction", type=float, default=0.22)
    parser.add_argument("--epsilon-repair", type=float, default=2.0)
    parser.add_argument("--ledger-overhead", type=float, default=0.28)
    parser.add_argument("--expensive-ledger-overhead", type=float, default=0.63)
    parser.add_argument("--min-high-fisher-block", type=float, default=0.95)
    parser.add_argument("--max-unsafe-leak", type=float, default=0.05)
    parser.add_argument("--min-prior-loss-reduction", type=float, default=0.85)
    parser.add_argument("--min-new-task-retention", type=float, default=0.70)
    parser.add_argument("--min-allowed-low-fisher", type=float, default=0.70)
    parser.add_argument("--max-locked-flip-fraction", type=float, default=0.55)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.80)
    parser.add_argument("--min-capability-retention-multiplier", type=float, default=1.0)
    return parser.parse_args()


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def make_fixture(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    param_count = args.param_count
    weights = rng.choice(np.array([-1.0, 1.0]), size=param_count)
    high_count = int(round(param_count * args.high_fisher_fraction))
    high_idx = rng.choice(param_count, size=high_count, replace=False)
    high = np.zeros(param_count, dtype=bool)
    high[high_idx] = True

    true_fisher = np.full(param_count, 0.10, dtype=np.float64)
    true_fisher[high] = 3.00
    estimated_fisher = true_fisher.copy()

    if condition == "no_fisher_signal_fail":
        estimated_fisher = rng.uniform(0.05, 0.70, size=param_count)
    elif condition == "stale_fisher_fail":
        estimated_fisher = np.full(param_count, 0.10, dtype=np.float64)
        low_idx = np.flatnonzero(~high)
        stale_idx = rng.choice(low_idx, size=high_count, replace=False)
        estimated_fisher[stale_idx] = 3.00

    flip_probability = np.where(high, 0.86, 0.29)
    proposed = rng.random(param_count) < flip_probability
    delta = np.where(proposed, -2.0 * weights, 0.0)

    future_benefit = np.where(high, 0.16, 1.0) * proposed.astype(np.float64)
    future_benefit += proposed.astype(np.float64) * rng.uniform(0.0, 0.04, size=param_count)

    epsilon = args.epsilon_repair
    if condition == "overlock_fail":
        epsilon = 0.20
    elif condition == "underlock_fail":
        epsilon = 20.0

    return weights, true_fisher, estimated_fisher, delta, future_benefit, epsilon


def prior_loss_delta(true_fisher: np.ndarray, delta: np.ndarray) -> float:
    return 0.5 * float(np.sum(true_fisher * delta * delta))


def safe_fraction(numerator_mask: np.ndarray, denominator_mask: np.ndarray) -> float:
    denominator = int(np.count_nonzero(denominator_mask))
    if denominator == 0:
        return 1.0
    return float(np.count_nonzero(numerator_mask & denominator_mask)) / denominator


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    _weights, true_fisher, estimated_fisher, proposed_delta, future_benefit, epsilon = make_fixture(
        rng,
        args,
        condition,
    )
    proposed = proposed_delta != 0.0
    high_true = true_fisher >= 1.0
    low_true = ~high_true
    risk_score = estimated_fisher * proposed_delta * proposed_delta
    locked = proposed & (risk_score > epsilon)
    allowed = proposed & ~locked
    escrow_delta = np.where(allowed, proposed_delta, 0.0)

    baseline_prior = prior_loss_delta(true_fisher, proposed_delta)
    escrow_prior = prior_loss_delta(true_fisher, escrow_delta)
    prior_reduction = 1.0 - escrow_prior / max(baseline_prior, 1.0e-30)
    baseline_progress = float(np.sum(future_benefit[proposed]))
    escrow_progress = float(np.sum(future_benefit[allowed]))
    progress_retention = escrow_progress / max(baseline_progress, 1.0e-30)

    proposed_high = proposed & high_true
    proposed_low = proposed & low_true
    high_block = safe_fraction(locked, proposed_high)
    high_leak = safe_fraction(allowed, proposed_high)
    low_allowed = safe_fraction(allowed, proposed_low)
    proposed_fraction = float(np.count_nonzero(proposed)) / args.param_count
    locked_fraction = float(np.count_nonzero(locked)) / max(int(np.count_nonzero(proposed)), 1)
    allowed_flip_fraction = float(np.count_nonzero(allowed)) / args.param_count

    ledger_overhead = (
        args.expensive_ledger_overhead
        if condition == "repair_cost_fail"
        else args.ledger_overhead
    )
    ledger_work = ledger_overhead * args.param_count
    allowed_work = float(np.count_nonzero(allowed))
    priced_work_ratio = (ledger_work + allowed_work) / args.param_count
    capability_multiplier = (prior_reduction * max(progress_retention, 0.0)) / max(
        priced_work_ratio,
        1.0e-30,
    )

    failed: list[str] = []
    if not finite(
        proposed_fraction,
        locked_fraction,
        high_block,
        high_leak,
        low_allowed,
        baseline_prior,
        escrow_prior,
        prior_reduction,
        progress_retention,
        ledger_work,
        allowed_work,
        priced_work_ratio,
        capability_multiplier,
    ):
        failed.append("non_finite_metric")
    if high_block < args.min_high_fisher_block:
        failed.append("high_fisher_not_blocked")
    if high_leak > args.max_unsafe_leak:
        failed.append("unsafe_high_fisher_leak")
    if prior_reduction < args.min_prior_loss_reduction:
        failed.append("prior_capability_damage")
    if progress_retention < args.min_new_task_retention:
        failed.append("new_task_starved")
    if low_allowed < args.min_allowed_low_fisher:
        failed.append("low_fisher_overlocked")
    if locked_fraction > args.max_locked_flip_fraction:
        failed.append("overlocked")
    if priced_work_ratio > args.max_priced_work_ratio:
        failed.append("price")
    if capability_multiplier < args.min_capability_retention_multiplier:
        failed.append("capability_retention_multiplier")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed) if failed else None,
        param_count=args.param_count,
        proposed_flip_fraction=float(proposed_fraction),
        locked_flip_fraction=float(locked_fraction),
        high_fisher_flip_block_fraction=float(high_block),
        unsafe_high_fisher_leak_fraction=float(high_leak),
        allowed_low_fisher_flip_fraction=float(low_allowed),
        baseline_prior_loss_delta=float(baseline_prior),
        escrow_prior_loss_delta=float(escrow_prior),
        prior_loss_reduction_fraction=float(prior_reduction),
        new_task_progress_retention=float(progress_retention),
        ledger_work_units=float(ledger_work),
        allowed_flip_work_units=float(allowed_work),
        priced_work_ratio=float(priced_work_ratio),
        projected_capability_retention_multiplier=float(capability_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "param_count",
        "proposed_flip_fraction",
        "locked_flip_fraction",
        "high_fisher_flip_block_fraction",
        "unsafe_high_fisher_leak_fraction",
        "allowed_low_fisher_flip_fraction",
        "baseline_prior_loss_delta",
        "escrow_prior_loss_delta",
        "prior_loss_reduction_fraction",
        "new_task_progress_retention",
        "ledger_work_units",
        "allowed_flip_work_units",
        "priced_work_ratio",
        "projected_capability_retention_multiplier",
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
    no_signal = [result for result in results if result.condition == "no_fisher_signal_fail"]
    overlock = [result for result in results if result.condition == "overlock_fail"]
    underlock = [result for result in results if result.condition == "underlock_fail"]
    stale = [result for result in results if result.condition == "stale_fisher_fail"]
    cost = [result for result in results if result.condition == "repair_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_blocks_high_fisher": (
            aggregates["positive"]["high_fisher_flip_block_fraction_min"] >= 0.95
            and aggregates["positive"]["unsafe_high_fisher_leak_fraction_max"] <= 0.05
        ),
        "positive_condition_preserves_prior_capability": (
            aggregates["positive"]["prior_loss_reduction_fraction_min"] >= 0.85
        ),
        "positive_condition_preserves_future_progress": (
            aggregates["positive"]["new_task_progress_retention_min"] >= 0.70
            and aggregates["positive"]["allowed_low_fisher_flip_fraction_min"] >= 0.70
        ),
        "positive_condition_priced": (
            aggregates["positive"]["priced_work_ratio_mean"] <= 0.80
            and aggregates["positive"]["projected_capability_retention_multiplier_mean"] >= 1.0
        ),
        "no_signal_control_all_fail_leak_or_damage": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "high_fisher_not_blocked" in result.fail_reason
                or "unsafe_high_fisher_leak" in result.fail_reason
                or "prior_capability_damage" in result.fail_reason
            )
            for result in no_signal
        ),
        "overlock_control_all_fail_future_progress": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "new_task_starved" in result.fail_reason
                or "low_fisher_overlocked" in result.fail_reason
                or "overlocked" in result.fail_reason
            )
            for result in overlock
        ),
        "underlock_control_all_fail_leak": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "high_fisher_not_blocked" in result.fail_reason
                or "unsafe_high_fisher_leak" in result.fail_reason
                or "prior_capability_damage" in result.fail_reason
            )
            for result in underlock
        ),
        "stale_control_all_fail_leak_or_damage": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "high_fisher_not_blocked" in result.fail_reason
                or "unsafe_high_fisher_leak" in result.fail_reason
                or "prior_capability_damage" in result.fail_reason
            )
            for result in stale
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
        "PASS_R501_S82_FWTCE_GATE"
        if all(checks.values())
        else "FAIL_R501_S82_FWTCE_GATE"
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
            "Synthetic CPU Fisher-weighted ternary commitment escrow gate only; "
            "not heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S82_FWTCE_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_high_fisher_block_min={positive['high_fisher_flip_block_fraction_min']:.6f} "
        f"positive_unsafe_leak_max={positive['unsafe_high_fisher_leak_fraction_max']:.6f} "
        f"positive_prior_loss_reduction_min={positive['prior_loss_reduction_fraction_min']:.6f} "
        f"positive_new_task_retention_min={positive['new_task_progress_retention_min']:.6f} "
        f"positive_allowed_low_fisher_min={positive['allowed_low_fisher_flip_fraction_min']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_capability_retention_multiplier_mean={positive['projected_capability_retention_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
