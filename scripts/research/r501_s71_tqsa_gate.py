#!/usr/bin/env python3
"""R501 S71 ternary quantization shock absorber gate.

This CPU diagnostic implements the R501 Quality Math 8 finding:
"Ternary Quantization Shock Absorber (TQSA)".

The gate constructs a synthetic ternary realization step where continuous
weights are projected to {-1, 0, 1}. Standard realization applies every
crossing, while TQSA delays crossings whose estimated state shock
||d h / d theta_i * Delta theta_i|| exceeds an epsilon budget. A positive case
must reduce the immediate CE proxy shock, respect the state-shock bound, still
realize enough safe weights, and remain under the priced denominator.

Controls exercise the declared risks: stale sensitivity estimates admit
destructive crossings, over-conservative estimates delay too much of the
realized artifact, and expensive scoring erases the repair-time denominator.

This is a synthetic CPU quantization-shock diagnostic, not heldout CE progress,
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


SCHEMA = "r501_s71_tqsa_gate.v1"
CONDITIONS = (
    "positive",
    "stale_sensitivity_fail",
    "overdelay_fail",
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
    sensitive_fraction: float
    standard_quantized_fraction: float
    tqsa_quantized_fraction: float
    delayed_fraction: float
    standard_loss_delta: float
    tqsa_loss_delta: float
    loss_reduction_fraction: float
    standard_state_shock_max: float
    tqsa_state_shock_max: float
    shock_bound_violation_fraction: float
    priced_work_ratio: float
    projected_token_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50171)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--param-dim", type=int, default=256)
    parser.add_argument("--sensitive-fraction", type=float, default=0.25)
    parser.add_argument("--epsilon-shock", type=float, default=0.08)
    parser.add_argument("--max-tqsa-loss-delta", type=float, default=0.004)
    parser.add_argument("--min-loss-reduction-fraction", type=float, default=0.75)
    parser.add_argument("--min-quantized-fraction", type=float, default=0.50)
    parser.add_argument("--max-shock-violation-fraction", type=float, default=0.0)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.90)
    parser.add_argument("--priced-work-ratio", type=float, default=0.74)
    parser.add_argument("--expensive-priced-work-ratio", type=float, default=0.93)
    return parser.parse_args()


def ternary_targets(rng: np.random.Generator, param_dim: int) -> np.ndarray:
    return rng.choice(np.array([-1.0, 0.0, 1.0], dtype=np.float64), size=param_dim)


def make_fixture(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sensitive_count = int(round(args.param_dim * args.sensitive_fraction))
    sensitive_mask = np.zeros(args.param_dim, dtype=bool)
    sensitive_mask[:sensitive_count] = True
    rng.shuffle(sensitive_mask)

    target = ternary_targets(rng, args.param_dim)
    residual = np.empty(args.param_dim, dtype=np.float64)
    residual[sensitive_mask] = rng.uniform(0.22, 0.38, size=int(np.count_nonzero(sensitive_mask)))
    residual[~sensitive_mask] = rng.uniform(0.18, 0.34, size=int(np.count_nonzero(~sensitive_mask)))
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=args.param_dim)
    theta = target - signs * residual

    true_sensitivity = np.empty(args.param_dim, dtype=np.float64)
    true_sensitivity[sensitive_mask] = rng.uniform(0.75, 1.20, size=int(np.count_nonzero(sensitive_mask)))
    true_sensitivity[~sensitive_mask] = rng.uniform(0.05, 0.17, size=int(np.count_nonzero(~sensitive_mask)))

    if condition == "stale_sensitivity_fail":
        estimated_sensitivity = true_sensitivity.copy()
        estimated_sensitivity[sensitive_mask] *= rng.uniform(0.04, 0.09, size=int(np.count_nonzero(sensitive_mask)))
    elif condition == "overdelay_fail":
        estimated_sensitivity = true_sensitivity * rng.uniform(4.5, 5.5, size=args.param_dim)
    else:
        estimated_sensitivity = true_sensitivity * rng.uniform(1.00, 1.03, size=args.param_dim)

    return theta, target, true_sensitivity, estimated_sensitivity


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def loss_delta(sensitivity: np.ndarray, delta: np.ndarray) -> float:
    shock = sensitivity * np.abs(delta)
    return float(np.mean(shock * shock))


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    theta, target, true_sensitivity, estimated_sensitivity = make_fixture(rng, args, condition)
    delta = target - theta
    standard_realize = np.ones(args.param_dim, dtype=bool)
    estimated_shock = estimated_sensitivity * np.abs(delta)
    tqsa_realize = estimated_shock < args.epsilon_shock
    standard_delta = np.where(standard_realize, delta, 0.0)
    tqsa_delta = np.where(tqsa_realize, delta, 0.0)

    true_shock_standard = true_sensitivity * np.abs(standard_delta)
    true_shock_tqsa = true_sensitivity * np.abs(tqsa_delta)
    standard_loss = loss_delta(true_sensitivity, standard_delta)
    tqsa_loss = loss_delta(true_sensitivity, tqsa_delta)
    loss_reduction = (standard_loss - tqsa_loss) / max(standard_loss, 1.0e-30)
    shock_violations = np.count_nonzero(true_shock_tqsa >= args.epsilon_shock)
    shock_violation_fraction = shock_violations / args.param_dim
    priced_work_ratio = (
        args.expensive_priced_work_ratio
        if condition == "scoring_cost_fail"
        else args.priced_work_ratio
    )
    projected_multiplier = 1.0 / priced_work_ratio

    failed: list[str] = []
    if not finite(
        standard_loss,
        tqsa_loss,
        loss_reduction,
        float(np.max(true_shock_standard)),
        float(np.max(true_shock_tqsa)),
        shock_violation_fraction,
        priced_work_ratio,
        projected_multiplier,
    ):
        failed.append("non_finite_metric")
    if tqsa_loss > args.max_tqsa_loss_delta:
        failed.append("loss_bound")
    if loss_reduction < args.min_loss_reduction_fraction:
        failed.append("loss_reduction")
    if float(np.mean(tqsa_realize)) < args.min_quantized_fraction:
        failed.append("realization_fraction")
    if shock_violation_fraction > args.max_shock_violation_fraction:
        failed.append("shock_bound")
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
        sensitive_fraction=float(np.count_nonzero(true_sensitivity > 0.5) / args.param_dim),
        standard_quantized_fraction=float(np.mean(standard_realize)),
        tqsa_quantized_fraction=float(np.mean(tqsa_realize)),
        delayed_fraction=float(1.0 - np.mean(tqsa_realize)),
        standard_loss_delta=float(standard_loss),
        tqsa_loss_delta=float(tqsa_loss),
        loss_reduction_fraction=float(loss_reduction),
        standard_state_shock_max=float(np.max(true_shock_standard)),
        tqsa_state_shock_max=float(np.max(true_shock_tqsa)),
        shock_bound_violation_fraction=float(shock_violation_fraction),
        priced_work_ratio=float(priced_work_ratio),
        projected_token_efficiency_multiplier=float(projected_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "param_dim",
        "sensitive_fraction",
        "standard_quantized_fraction",
        "tqsa_quantized_fraction",
        "delayed_fraction",
        "standard_loss_delta",
        "tqsa_loss_delta",
        "loss_reduction_fraction",
        "standard_state_shock_max",
        "tqsa_state_shock_max",
        "shock_bound_violation_fraction",
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
    stale = [result for result in results if result.condition == "stale_sensitivity_fail"]
    overdelay = [result for result in results if result.condition == "overdelay_fail"]
    cost = [result for result in results if result.condition == "scoring_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_reduces_shock": (
            aggregates["positive"]["loss_reduction_fraction_mean"] >= 0.75
            and aggregates["positive"]["tqsa_loss_delta_max"] <= 0.004
        ),
        "positive_condition_respects_state_bound": (
            aggregates["positive"]["shock_bound_violation_fraction_max"] == 0.0
            and aggregates["positive"]["tqsa_state_shock_max_max"] < 0.08
        ),
        "positive_condition_realizes_safe_weights": (
            aggregates["positive"]["tqsa_quantized_fraction_mean"] >= 0.50
        ),
        "positive_condition_priced": (
            aggregates["positive"]["priced_work_ratio_mean"] <= 0.90
        ),
        "stale_control_all_fail_bound": all(
            (not result.passed)
            and result.fail_reason is not None
            and ("shock_bound" in result.fail_reason or "loss_bound" in result.fail_reason)
            for result in stale
        ),
        "overdelay_control_all_fail_realization": all(
            (not result.passed)
            and result.fail_reason is not None
            and "realization_fraction" in result.fail_reason
            for result in overdelay
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
        "PASS_R501_S71_TQSA_GATE"
        if all(checks.values())
        else "FAIL_R501_S71_TQSA_GATE"
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
            "Synthetic CPU ternary quantization shock-absorber gate only; not "
            "heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S71_TQSA_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_tqsa_loss_delta_max={positive['tqsa_loss_delta_max']:.6f} "
        f"positive_standard_loss_delta_mean={positive['standard_loss_delta_mean']:.6f} "
        f"positive_loss_reduction_fraction_mean={positive['loss_reduction_fraction_mean']:.6f} "
        f"positive_tqsa_state_shock_max={positive['tqsa_state_shock_max_max']:.6f} "
        f"positive_tqsa_quantized_fraction_mean={positive['tqsa_quantized_fraction_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_token_efficiency_multiplier_mean={positive['projected_token_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
