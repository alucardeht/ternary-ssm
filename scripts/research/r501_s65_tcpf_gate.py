#!/usr/bin/env python3
"""R501 S65 ternary capability potential field gate.

This CPU diagnostic implements the R501 Quality Math 2 finding:
"Ternary Capability Potential Field (TCPF)".

The gate simulates a ternary optimizer chamber where each weight owns a
continuous potential accumulator. A weight may realize a ternary flip only when
the accumulated CE-gradient Hamiltonian crosses a certified energy barrier
Delta E. A positive case must keep candidate work exactly zero before the
barrier, realize all capability-bearing weights, avoid noisy false flips, and
stay inside the priced token-efficiency denominator.

Controls exercise the declared risks: a too-high barrier freezes capability
acquisition, removing the barrier turns high-variance noise into false work,
and a costly accumulator/audit path erases the denominator.

This is a synthetic CPU ternary-potential realization diagnostic, not heldout
CE progress, GPU training-speed evidence, or production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s65_tcpf_gate.v1"
CONDITIONS = (
    "positive",
    "high_barrier_freeze_fail",
    "no_barrier_noise_fail",
    "accumulator_cost_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    step_count: int
    weight_count: int
    stable_weight_count: int
    barrier: float
    candidate_work_fraction: float
    pre_threshold_candidate_fraction: float
    flip_recall: float
    false_flip_rate: float
    reversal_rate: float
    realized_loss_gap: float
    ce_drop_ratio: float
    accumulator_work_ratio: float
    priced_work_ratio: float
    projected_token_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50165)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--step-count", type=int, default=32)
    parser.add_argument("--weight-count", type=int, default=64)
    parser.add_argument("--stable-weight-count", type=int, default=12)
    parser.add_argument("--barrier", type=float, default=0.75)
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument("--shock-penalty", type=float, default=0.11)
    parser.add_argument("--max-pre-threshold-candidate-fraction", type=float, default=0.0)
    parser.add_argument("--min-flip-recall", type=float, default=0.95)
    parser.add_argument("--max-false-flip-rate", type=float, default=0.0)
    parser.add_argument("--max-reversal-rate", type=float, default=0.0)
    parser.add_argument("--max-realized-loss-gap", type=float, default=0.05)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.80)
    return parser.parse_args()


def make_evidence(
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    evidence = rng.normal(scale=0.006, size=(args.step_count, args.weight_count))
    stable = np.zeros(args.weight_count, dtype=bool)
    stable_indices = rng.choice(args.weight_count, size=args.stable_weight_count, replace=False)
    stable[stable_indices] = True
    signs = np.zeros(args.weight_count, dtype=np.float64)
    signs[stable] = rng.choice(np.array([-1.0, 1.0]), size=args.stable_weight_count)
    drift = 0.074
    evidence[:, stable] = (
        signs[stable][None, :] * drift
        + rng.normal(scale=0.004, size=(args.step_count, args.stable_weight_count))
    )
    return evidence, stable, signs


def simulate_tcpf(
    evidence: np.ndarray,
    stable: np.ndarray,
    signs: np.ndarray,
    args: argparse.Namespace,
    condition: str,
) -> dict[str, float]:
    barrier = args.barrier
    if condition == "high_barrier_freeze_fail":
        barrier = 4.0
    no_barrier = condition == "no_barrier_noise_fail"

    potential = np.zeros(args.weight_count, dtype=np.float64)
    realized = np.zeros(args.weight_count, dtype=np.float64)
    flipped = np.zeros(args.weight_count, dtype=bool)
    candidate_events = 0
    pre_threshold_events = 0
    false_flips = 0
    reversals = 0
    stable_flips = 0
    ce_drop = 0.0

    for step in range(args.step_count):
        potential[~flipped | no_barrier] += evidence[step, ~flipped | no_barrier]
        if no_barrier:
            candidates = np.abs(evidence[step]) > 0.010
        else:
            candidates = (~flipped) & (np.abs(potential) >= barrier)
        pre_threshold_events += int(np.count_nonzero(candidates & (np.abs(potential) < barrier)))
        for idx in np.flatnonzero(candidates):
            new_value = 1.0 if potential[idx] >= 0.0 else -1.0
            candidate_events += 1
            if realized[idx] != 0.0 and new_value != realized[idx]:
                reversals += 1
            realized[idx] = new_value
            if stable[idx] and new_value == signs[idx] and not flipped[idx]:
                stable_flips += 1
                ce_drop += args.alpha * barrier
            elif not stable[idx]:
                false_flips += 1
                ce_drop -= args.shock_penalty
            flipped[idx] = True

    expected_drop = args.alpha * args.barrier * args.stable_weight_count
    realized_loss_gap = max(expected_drop - ce_drop, 0.0) / max(expected_drop, 1.0e-30)
    ce_drop_ratio = ce_drop / max(expected_drop, 1.0e-30)
    candidate_work_fraction = candidate_events / (args.step_count * args.weight_count)
    pre_threshold_fraction = pre_threshold_events / (args.step_count * args.weight_count)
    false_flip_rate = false_flips / max(candidate_events, 1)
    reversal_rate = reversals / max(candidate_events, 1)
    flip_recall = stable_flips / args.stable_weight_count
    accumulator_work = 0.13
    audit_work = 0.07
    if condition == "accumulator_cost_fail":
        accumulator_work = 0.63
        audit_work = 0.24
    priced_work_ratio = accumulator_work + audit_work + 1.20 * candidate_work_fraction
    projected_multiplier = 1.0 / priced_work_ratio

    return {
        "barrier": float(barrier),
        "candidate_work_fraction": float(candidate_work_fraction),
        "pre_threshold_candidate_fraction": float(pre_threshold_fraction),
        "flip_recall": float(flip_recall),
        "false_flip_rate": float(false_flip_rate),
        "reversal_rate": float(reversal_rate),
        "realized_loss_gap": float(realized_loss_gap),
        "ce_drop_ratio": float(ce_drop_ratio),
        "accumulator_work_ratio": float(accumulator_work + audit_work),
        "priced_work_ratio": float(priced_work_ratio),
        "projected_token_efficiency_multiplier": float(projected_multiplier),
    }


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    evidence, stable, signs = make_evidence(rng, args)
    metrics = simulate_tcpf(evidence, stable, signs, args, condition)

    failed: list[str] = []
    if not finite(*metrics.values()):
        failed.append("non_finite_metric")
    if (
        metrics["pre_threshold_candidate_fraction"]
        > args.max_pre_threshold_candidate_fraction
    ):
        failed.append("pre_threshold_work")
    if metrics["flip_recall"] < args.min_flip_recall:
        failed.append("capability_frozen")
    if metrics["false_flip_rate"] > args.max_false_flip_rate:
        failed.append("false_flip")
    if metrics["reversal_rate"] > args.max_reversal_rate:
        failed.append("oscillation")
    if metrics["realized_loss_gap"] > args.max_realized_loss_gap:
        failed.append("loss_gap")
    if metrics["priced_work_ratio"] > args.max_priced_work_ratio:
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
        step_count=args.step_count,
        weight_count=args.weight_count,
        stable_weight_count=args.stable_weight_count,
        barrier=metrics["barrier"],
        candidate_work_fraction=metrics["candidate_work_fraction"],
        pre_threshold_candidate_fraction=metrics["pre_threshold_candidate_fraction"],
        flip_recall=metrics["flip_recall"],
        false_flip_rate=metrics["false_flip_rate"],
        reversal_rate=metrics["reversal_rate"],
        realized_loss_gap=metrics["realized_loss_gap"],
        ce_drop_ratio=metrics["ce_drop_ratio"],
        accumulator_work_ratio=metrics["accumulator_work_ratio"],
        priced_work_ratio=metrics["priced_work_ratio"],
        projected_token_efficiency_multiplier=metrics[
            "projected_token_efficiency_multiplier"
        ],
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "step_count",
        "weight_count",
        "stable_weight_count",
        "barrier",
        "candidate_work_fraction",
        "pre_threshold_candidate_fraction",
        "flip_recall",
        "false_flip_rate",
        "reversal_rate",
        "realized_loss_gap",
        "ce_drop_ratio",
        "accumulator_work_ratio",
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
    freeze = [result for result in results if result.condition == "high_barrier_freeze_fail"]
    noise = [result for result in results if result.condition == "no_barrier_noise_fail"]
    cost = [result for result in results if result.condition == "accumulator_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_barrier_exact": (
            aggregates["positive"]["pre_threshold_candidate_fraction_max"] == 0.0
            and aggregates["positive"]["flip_recall_min"] >= 0.95
            and aggregates["positive"]["false_flip_rate_max"] == 0.0
            and aggregates["positive"]["realized_loss_gap_max"] <= 0.05
        ),
        "positive_condition_priced": (
            aggregates["positive"]["priced_work_ratio_mean"] <= 0.80
        ),
        "freeze_control_all_fail_capability": all(
            (not result.passed)
            and result.fail_reason is not None
            and "capability_frozen" in result.fail_reason
            and "loss_gap" in result.fail_reason
            for result in freeze
        ),
        "noise_control_all_fail_false_work": all(
            (not result.passed)
            and result.fail_reason is not None
            and "pre_threshold_work" in result.fail_reason
            and "false_flip" in result.fail_reason
            for result in noise
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
        "PASS_R501_S65_TCPF_GATE"
        if all(checks.values())
        else "FAIL_R501_S65_TCPF_GATE"
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
            "Synthetic CPU ternary capability potential-field realization gate "
            "only; not heldout CE progress, not GPU training speed, and not "
            "production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S65_TCPF_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_candidate_work_fraction_mean={positive['candidate_work_fraction_mean']:.6f} "
        f"positive_pre_threshold_candidate_fraction_max={positive['pre_threshold_candidate_fraction_max']:.6e} "
        f"positive_flip_recall_mean={positive['flip_recall_mean']:.6f} "
        f"positive_realized_loss_gap_max={positive['realized_loss_gap_max']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_token_efficiency_multiplier_mean={positive['projected_token_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
