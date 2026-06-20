#!/usr/bin/env python3
"""R501 S45 local error-feedback forward eligibility gate.

This CPU diagnostic implements the R501 Speed Math 12 finding:
"Local Error-Feedback Forward Eligibility (Synthetic Adjoints)".

The gate simulates a forward synthetic-adjoint predictor whose local cotangent
error is periodically estimated and injected back into the forward momentum. The
finding survives only when the expected error decays over the segment, predicted
updates preserve martingale descent, phase-delay oscillations stay bounded, and
the priced forward predictor plus error injection route is below recurrent BPTT
work.

Controls exercise the declared kills: delayed overcorrection creates oscillation
and worsens step count, biased error does not converge to zero, and too-frequent
error injection recovers dense backward cost.

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


SCHEMA = "r501_s45_error_feedback_forward_gate.v1"
CONDITIONS = (
    "positive",
    "phase_delay_oscillation_fail",
    "biased_error_fail",
    "pricing_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    segment_length: int
    state_dim: int
    correction_interval: int
    correction_delay: int
    error_decay_ratio: float
    final_mean_error_norm: float
    martingale_descent_ratio: float
    phase_flip_rate: float
    projected_step_count_ratio: float
    priced_work_ratio: float
    projected_speedup: float
    price_components: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50145)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--segment-length", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--max-error-decay-ratio", type=float, default=0.35)
    parser.add_argument("--max-final-mean-error-norm", type=float, default=0.035)
    parser.add_argument("--min-martingale-descent-ratio", type=float, default=0.95)
    parser.add_argument("--max-phase-flip-rate", type=float, default=0.30)
    parser.add_argument("--max-step-count-ratio", type=float, default=1.05)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.90)
    parser.add_argument("--predictor-cost-ratio", type=float, default=0.14)
    parser.add_argument("--error-inject-cost-ratio", type=float, default=1.00)
    parser.add_argument("--audit-cost-ratio", type=float, default=0.025)
    return parser.parse_args()


def smooth_true_gradients(
    rng: np.random.Generator,
    segment_length: int,
    state_dim: int,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, segment_length, dtype=np.float64)
    columns = [
        np.ones_like(t),
        np.sin(2.0 * math.pi * t),
        np.cos(2.0 * math.pi * t),
        np.sin(4.0 * math.pi * t + 0.2),
        np.cos(6.0 * math.pi * t - 0.4),
        t - 0.5,
    ]
    basis = np.stack(columns, axis=1)
    weights = rng.normal(scale=0.20, size=(basis.shape[1], state_dim))
    grad = basis @ weights
    grad += 0.01 * rng.normal(size=grad.shape)
    return grad.astype(np.float64)


def condition_params(condition: str) -> dict[str, float | int]:
    if condition == "positive":
        return {
            "decay": 0.94,
            "gain": 0.18,
            "interval": 8,
            "delay": 2,
            "noise": 0.004,
            "bias": 0.0,
        }
    if condition == "phase_delay_oscillation_fail":
        return {
            "decay": 0.985,
            "gain": 1.25,
            "interval": 8,
            "delay": 10,
            "noise": 0.004,
            "bias": 0.0,
        }
    if condition == "biased_error_fail":
        return {
            "decay": 0.96,
            "gain": 0.18,
            "interval": 8,
            "delay": 2,
            "noise": 0.004,
            "bias": 0.010,
        }
    if condition == "pricing_fail":
        return {
            "decay": 0.94,
            "gain": 0.18,
            "interval": 1,
            "delay": 0,
            "noise": 0.004,
            "bias": 0.0,
        }
    raise ValueError(f"unknown condition: {condition}")


def simulate_error(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, int, int]:
    params = condition_params(condition)
    decay = float(params["decay"])
    gain = float(params["gain"])
    interval = int(params["interval"])
    delay = int(params["delay"])
    noise = float(params["noise"])
    bias_scale = float(params["bias"])
    bias = rng.normal(scale=bias_scale, size=args.state_dim)
    error = np.zeros((args.segment_length, args.state_dim), dtype=np.float64)
    error[0] = rng.normal(scale=0.16, size=args.state_dim)
    correction = np.zeros(args.state_dim, dtype=np.float64)
    for t in range(1, args.segment_length):
        if t % interval == 0 and t > delay:
            start = max(0, t - delay - interval)
            end = max(1, t - delay)
            estimate = np.mean(error[start:end], axis=0)
            correction = gain * estimate
        innovation = rng.normal(scale=noise, size=args.state_dim)
        error[t] = decay * error[t - 1] - correction + bias + innovation
    return error, interval, delay


def phase_flip_rate(error: np.ndarray) -> float:
    axis = error[0] / max(float(np.linalg.norm(error[0])), 1.0e-30)
    projection = error @ axis
    signs = np.sign(projection)
    signs[signs == 0.0] = 1.0
    flips = signs[1:] != signs[:-1]
    return float(np.mean(flips.astype(np.float64)))


def metrics_for_error(
    args: argparse.Namespace,
    true_grad: np.ndarray,
    error: np.ndarray,
) -> tuple[float, float, float, float, float]:
    q = max(1, error.shape[0] // 4)
    first = float(np.mean(np.linalg.norm(error[:q], axis=1)))
    last = float(np.mean(np.linalg.norm(error[-q:], axis=1)))
    decay_ratio = last / max(first, 1.0e-30)
    final_mean_norm = float(np.linalg.norm(np.mean(error[-q:], axis=0)))
    predicted = true_grad + error
    numerator = float(np.sum(predicted * true_grad))
    denominator = max(float(np.sum(true_grad * true_grad)), 1.0e-30)
    descent = numerator / denominator
    flips = phase_flip_rate(error)
    oscillation_penalty = max(0.0, flips - 0.20)
    error_penalty = max(0.0, decay_ratio - args.max_error_decay_ratio)
    step_ratio = (1.0 / max(descent, 1.0e-6)) * (1.0 + oscillation_penalty + error_penalty)
    return decay_ratio, final_mean_norm, descent, flips, float(step_ratio)


def price_route(args: argparse.Namespace, correction_interval: int) -> dict[str, float]:
    return {
        "forward_predictor": float(args.predictor_cost_ratio),
        "error_injection": float(args.error_inject_cost_ratio / float(correction_interval)),
        "martingale_audit": float(args.audit_cost_ratio),
    }


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    true_grad = smooth_true_gradients(rng, args.segment_length, args.state_dim)
    error, interval, delay = simulate_error(rng, args, condition)
    decay_ratio, final_mean_norm, descent, flips, step_ratio = metrics_for_error(
        args,
        true_grad,
        error,
    )
    price_components = price_route(args, interval)
    priced_work_ratio = float(sum(price_components.values()))
    projected_speedup = 1.0 / priced_work_ratio if priced_work_ratio > 0.0 else math.inf

    decay_pass = decay_ratio <= args.max_error_decay_ratio
    final_mean_pass = final_mean_norm <= args.max_final_mean_error_norm
    descent_pass = descent >= args.min_martingale_descent_ratio
    phase_pass = flips <= args.max_phase_flip_rate
    step_pass = step_ratio <= args.max_step_count_ratio
    price_pass = priced_work_ratio <= args.max_priced_work_ratio
    passed = bool(decay_pass and final_mean_pass and descent_pass and phase_pass and step_pass and price_pass)
    expected_pass = condition == "positive"
    identifying_correct = passed == expected_pass

    fail_reason = None
    if not passed:
        if condition == "pricing_fail" and not price_pass:
            fail_reason = "error_feedback_route_recovers_dense_backward_cost"
        elif condition == "biased_error_fail" and (not decay_pass or not final_mean_pass):
            fail_reason = "error_feedback_expectation_does_not_converge"
        elif not phase_pass or not step_pass or not descent_pass:
            fail_reason = "phase_delay_oscillation_breaks_martingale_descent"
        elif not decay_pass or not final_mean_pass:
            fail_reason = "error_feedback_expectation_does_not_converge"
        elif not price_pass:
            fail_reason = "error_feedback_route_recovers_dense_backward_cost"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=identifying_correct,
        fail_reason=fail_reason,
        segment_length=args.segment_length,
        state_dim=args.state_dim,
        correction_interval=interval,
        correction_delay=delay,
        error_decay_ratio=decay_ratio,
        final_mean_error_norm=final_mean_norm,
        martingale_descent_ratio=descent,
        phase_flip_rate=flips,
        projected_step_count_ratio=step_ratio,
        priced_work_ratio=priced_work_ratio,
        projected_speedup=float(projected_speedup),
        price_components=price_components,
    )


def finite_trial(trial: TrialResult) -> bool:
    values = (
        trial.error_decay_ratio,
        trial.final_mean_error_norm,
        trial.martingale_descent_ratio,
        trial.phase_flip_rate,
        trial.projected_step_count_ratio,
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
            "correction_interval_mean": float(
                np.mean([row.correction_interval for row in rows])
            ),
            "correction_delay_mean": float(np.mean([row.correction_delay for row in rows])),
            "error_decay_ratio_mean": float(
                np.mean([row.error_decay_ratio for row in rows])
            ),
            "error_decay_ratio_max": float(
                np.max([row.error_decay_ratio for row in rows])
            ),
            "final_mean_error_norm_mean": float(
                np.mean([row.final_mean_error_norm for row in rows])
            ),
            "martingale_descent_ratio_mean": float(
                np.mean([row.martingale_descent_ratio for row in rows])
            ),
            "phase_flip_rate_mean": float(np.mean([row.phase_flip_rate for row in rows])),
            "projected_step_count_ratio_mean": float(
                np.mean([row.projected_step_count_ratio for row in rows])
            ),
            "projected_step_count_ratio_max": float(
                np.max([row.projected_step_count_ratio for row in rows])
            ),
            "priced_work_ratio_mean": float(
                np.mean([row.priced_work_ratio for row in rows])
            ),
            "projected_speedup_mean": float(
                np.mean([row.projected_speedup for row in rows])
            ),
        }
    return out


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.segment_length < 16 or args.state_dim < 4:
        raise ValueError("--segment-length must be >= 16 and --state-dim must be >= 4")

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
        "positive_condition_error_decays": all(
            trial.error_decay_ratio <= args.max_error_decay_ratio
            for trial in trials
            if trial.condition == "positive"
        ),
        "phase_delay_control_all_fail_oscillation": all(
            (not trial.passed)
            and trial.fail_reason == "phase_delay_oscillation_breaks_martingale_descent"
            for trial in trials
            if trial.condition == "phase_delay_oscillation_fail"
        ),
        "biased_error_control_all_fail_expectation": all(
            (not trial.passed)
            and trial.fail_reason == "error_feedback_expectation_does_not_converge"
            for trial in trials
            if trial.condition == "biased_error_fail"
        ),
        "pricing_control_all_fail_price": all(
            (not trial.passed)
            and trial.fail_reason == "error_feedback_route_recovers_dense_backward_cost"
            for trial in trials
            if trial.condition == "pricing_fail"
        ),
        "all_conditions_identified": all(trial.identifying_correct for trial in trials),
    }
    decision = (
        "PASS_R501_S45_ERROR_FEEDBACK_FORWARD_GATE"
        if all(required_passes.values())
        else "FAIL_R501_S45_ERROR_FEEDBACK_FORWARD_GATE"
    )
    disposition = (
        "synthetic_positive_control_identified_not_training_speed_claim"
        if decision.startswith("PASS_")
        else "error_feedback_forward_gate_failed_or_non_identifying"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "CPU synthetic local error-feedback forward eligibility gate. It "
            "checks error expectation decay, martingale descent alignment, "
            "phase-delay oscillation, projected step count, and priced forward "
            "predictor plus error injection work. It is not heldout CE progress "
            "or GPU training-speed evidence."
        ),
        "finding": {
            "round": "R501",
            "aspect": "Speed",
            "id": "R501-S45",
            "title": "Local Error-Feedback Forward Eligibility",
        },
        "thresholds": {
            "max_error_decay_ratio": args.max_error_decay_ratio,
            "max_final_mean_error_norm": args.max_final_mean_error_norm,
            "min_martingale_descent_ratio": args.min_martingale_descent_ratio,
            "max_phase_flip_rate": args.max_phase_flip_rate,
            "max_step_count_ratio": args.max_step_count_ratio,
            "max_priced_work_ratio": args.max_priced_work_ratio,
        },
        "shape": {
            "seed": args.seed,
            "seed_count": args.seed_count,
            "segment_length": args.segment_length,
            "state_dim": args.state_dim,
        },
        "required_passes": required_passes,
        "aggregate": aggregate(trials),
        "trials": [asdict(trial) for trial in trials],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    positive = result["aggregate"]["positive"]
    print(
        "[R501_S45_ERROR_FEEDBACK_FORWARD_GATE] "
        f"active=1 decision={decision} disposition={disposition} "
        f"positive_error_decay_ratio_mean={positive['error_decay_ratio_mean']:.6f} "
        f"positive_martingale_descent_ratio_mean={positive['martingale_descent_ratio_mean']:.6f} "
        f"positive_phase_flip_rate_mean={positive['phase_flip_rate_mean']:.6f} "
        f"positive_step_count_ratio_mean={positive['projected_step_count_ratio_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
