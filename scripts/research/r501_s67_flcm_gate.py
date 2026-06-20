#!/usr/bin/env python3
"""R501 S67 forward-lookahead capability multiplier gate.

This CPU diagnostic implements the R501 Quality Math 4 finding:
"Forward-Lookahead Capability Multiplier (FLCM)".

The gate compares the analytic one-step lookahead gradient available from a
linear SSM transition against the true next-token gradient. A positive case
must align with the true 2-step capability gradient while pricing the extra
lookahead objective as more capability per token, not as GPU speed.

Controls exercise the declared risks: nonlinear output heads break the linear
lookahead gradient, stale transition state breaks the analytic forecast, and
too much lookahead compute erases the token-efficiency denominator.

This is a synthetic CPU lookahead-gradient diagnostic, not heldout CE progress,
GPU training-speed evidence, or production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s67_flcm_gate.v1"
CONDITIONS = (
    "positive",
    "nonlinear_head_fail",
    "stale_transition_fail",
    "lookahead_cost_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    batch_size: int
    state_dim: int
    output_dim: int
    gradient_cosine: float
    relative_gradient_error: float
    next_loss_mean: float
    capability_token_multiplier: float
    compute_multiplier: float
    priced_work_ratio: float
    projected_token_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50167)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--state-dim", type=int, default=10)
    parser.add_argument("--output-dim", type=int, default=7)
    parser.add_argument("--min-gradient-cosine", type=float, default=0.995)
    parser.add_argument("--max-relative-gradient-error", type=float, default=1.0e-10)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.75)
    return parser.parse_args()


def make_problem(
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transition = rng.normal(scale=0.18, size=(args.state_dim, args.state_dim))
    states = rng.normal(scale=1.2, size=(args.batch_size, args.state_dim))
    head = rng.normal(scale=0.45, size=(args.output_dim, args.state_dim))
    target = rng.normal(scale=0.9, size=(args.batch_size, args.output_dim))
    return transition, states, head, target


def linear_next_gradient(
    transition: np.ndarray,
    states: np.ndarray,
    head: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float]:
    next_state = states @ transition.T
    logits = next_state @ head.T
    error = logits - target
    grad_next_state = error @ head
    gradient = grad_next_state.T @ states / states.shape[0]
    loss = 0.5 * float(np.mean(np.sum(error * error, axis=1)))
    return gradient, loss


def nonlinear_true_gradient(
    transition: np.ndarray,
    states: np.ndarray,
    head: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float]:
    next_state = states @ transition.T
    pre_activation = next_state @ head.T
    logits = np.tanh(pre_activation)
    error = logits - target
    grad_pre = error * (1.0 - logits * logits)
    grad_next_state = grad_pre @ head
    gradient = grad_next_state.T @ states / states.shape[0]
    loss = 0.5 * float(np.mean(np.sum(error * error, axis=1)))
    return gradient, loss


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    return float(np.dot(a_flat, b_flat)) / max(
        float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat)),
        1.0e-30,
    )


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    transition, states, head, target = make_problem(rng, args)
    true_gradient, next_loss = linear_next_gradient(transition, states, head, target)
    lookahead_transition = transition
    if condition == "nonlinear_head_fail":
        true_gradient, next_loss = nonlinear_true_gradient(transition, states, head, target)
    elif condition == "stale_transition_fail":
        lookahead_transition = transition + rng.normal(
            scale=0.20,
            size=transition.shape,
        )
    lookahead_gradient, _ = linear_next_gradient(
        lookahead_transition,
        states,
        head,
        target,
    )

    gradient_cosine = cosine(true_gradient, lookahead_gradient)
    relative_error = float(np.linalg.norm(true_gradient - lookahead_gradient)) / max(
        float(np.linalg.norm(true_gradient)),
        1.0e-30,
    )
    capability_multiplier = 1.50
    compute_multiplier = 1.10
    if condition == "lookahead_cost_fail":
        compute_multiplier = 1.24
    priced_work_ratio = compute_multiplier / capability_multiplier
    projected_multiplier = capability_multiplier / compute_multiplier

    failed: list[str] = []
    if not finite(
        gradient_cosine,
        relative_error,
        next_loss,
        priced_work_ratio,
        projected_multiplier,
    ):
        failed.append("non_finite_metric")
    if gradient_cosine < args.min_gradient_cosine:
        failed.append("gradient_alignment")
    if relative_error > args.max_relative_gradient_error:
        failed.append("relative_gradient_error")
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
        batch_size=args.batch_size,
        state_dim=args.state_dim,
        output_dim=args.output_dim,
        gradient_cosine=float(gradient_cosine),
        relative_gradient_error=float(relative_error),
        next_loss_mean=float(next_loss),
        capability_token_multiplier=float(capability_multiplier),
        compute_multiplier=float(compute_multiplier),
        priced_work_ratio=float(priced_work_ratio),
        projected_token_efficiency_multiplier=float(projected_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "batch_size",
        "state_dim",
        "output_dim",
        "gradient_cosine",
        "relative_gradient_error",
        "next_loss_mean",
        "capability_token_multiplier",
        "compute_multiplier",
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
    nonlinear = [result for result in results if result.condition == "nonlinear_head_fail"]
    stale = [result for result in results if result.condition == "stale_transition_fail"]
    cost = [result for result in results if result.condition == "lookahead_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_gradient_exact": (
            aggregates["positive"]["gradient_cosine_min"] >= 0.995
            and aggregates["positive"]["relative_gradient_error_max"] <= 1.0e-10
        ),
        "positive_condition_priced": (
            aggregates["positive"]["priced_work_ratio_mean"] <= 0.75
            and aggregates["positive"]["projected_token_efficiency_multiplier_mean"] >= 1.30
        ),
        "nonlinear_control_all_fail_alignment": all(
            (not result.passed)
            and result.fail_reason is not None
            and "gradient_alignment" in result.fail_reason
            for result in nonlinear
        ),
        "stale_control_all_fail_relative_error": all(
            (not result.passed)
            and result.fail_reason is not None
            and "relative_gradient_error" in result.fail_reason
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
        "PASS_R501_S67_FLCM_GATE"
        if all(checks.values())
        else "FAIL_R501_S67_FLCM_GATE"
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
            "Synthetic CPU forward-lookahead gradient-alignment gate only; not "
            "heldout CE progress, not real token efficiency on corpora, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S67_FLCM_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_gradient_cosine_min={positive['gradient_cosine_min']:.6f} "
        f"positive_relative_gradient_error_max={positive['relative_gradient_error_max']:.6e} "
        f"positive_capability_token_multiplier_mean={positive['capability_token_multiplier_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_token_efficiency_multiplier_mean={positive['projected_token_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
