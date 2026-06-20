#!/usr/bin/env python3
"""R501 S44 recurrent-cotangent predictor-corrector tube gate.

This CPU diagnostic implements the R501 Speed Math 11 finding:
"Recurrent-Cotangent Predictor-Corrector Tube".

The gate builds a lightweight low-rank predictor lambda_hat_t = P h_t and checks
the Bellman residual lambda_hat_t - A^T lambda_hat_{t+1} - grad_t through a
random subset projection. The finding survives only when the predictor is
accurate, the full Bellman residual is low, the sampled verifier agrees with the
full residual audit, and the priced predictor plus sampled verification route is
below recurrent dense backward work.

Controls exercise the declared kills: full Bellman checking recovers dense cost,
random subset projection can miss residual debt, and a poor predictor fails the
Bellman/error-bound invariant.

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


SCHEMA = "r501_s44_predictor_corrector_tube_gate.v1"
CONDITIONS = (
    "positive",
    "full_verify_cost_fail",
    "projection_aliasing_fail",
    "poor_predictor_fail",
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
    predictor_rank: int
    verifier_sample_count: int
    spectral_radius: float
    lambda_relative_error: float
    full_bellman_residual_ratio: float
    sampled_bellman_residual_ratio: float
    projection_miss_ratio: float
    invariant_bound_ratio: float
    priced_work_ratio: float
    projected_speedup: float
    price_components: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50144)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--segment-length", type=int, default=160)
    parser.add_argument("--state-dim", type=int, default=96)
    parser.add_argument("--hidden-feature-dim", type=int, default=48)
    parser.add_argument("--predictor-rank", type=int, default=6)
    parser.add_argument("--verifier-sample-count", type=int, default=8)
    parser.add_argument("--max-lambda-relative-error", type=float, default=2.0e-3)
    parser.add_argument("--max-full-bellman-residual-ratio", type=float, default=2.0e-3)
    parser.add_argument("--max-projection-miss-ratio", type=float, default=0.25)
    parser.add_argument("--max-invariant-bound-ratio", type=float, default=1.0)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.90)
    parser.add_argument("--predictor-cost-multiplier", type=float, default=2.0)
    parser.add_argument("--verifier-cost-multiplier", type=float, default=1.0)
    parser.add_argument("--audit-cost-ratio", type=float, default=0.020)
    return parser.parse_args()


def orthogonal_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def transition_matrix(rng: np.random.Generator, state_dim: int) -> np.ndarray:
    q = orthogonal_matrix(rng, state_dim)
    eigs = np.linspace(0.10, 0.62, state_dim)
    return (q @ np.diag(eigs) @ q.T).astype(np.float64)


def hidden_features(
    rng: np.random.Generator,
    segment_length: int,
    hidden_feature_dim: int,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, segment_length + 1, dtype=np.float64)
    columns = [
        np.ones_like(t),
        t,
        t * t,
        np.sin(2.0 * math.pi * t),
        np.cos(2.0 * math.pi * t),
        np.sin(4.0 * math.pi * t + 0.3),
        np.cos(6.0 * math.pi * t - 0.1),
    ]
    base = np.stack(columns, axis=1)
    mix = rng.normal(scale=0.20, size=(base.shape[1], hidden_feature_dim))
    return (base @ mix + 0.01 * rng.normal(size=(segment_length + 1, hidden_feature_dim))).astype(
        np.float64
    )


def low_rank_predictor(
    rng: np.random.Generator,
    hidden_feature_dim: int,
    state_dim: int,
    rank: int,
) -> np.ndarray:
    left = rng.normal(scale=0.09, size=(state_dim, rank))
    right = rng.normal(scale=0.09, size=(rank, hidden_feature_dim))
    return (left @ right).astype(np.float64)


def exact_cotangent(grad: np.ndarray, a: np.ndarray, terminal: np.ndarray) -> np.ndarray:
    lam = np.zeros((grad.shape[0] + 1, grad.shape[1]), dtype=np.float64)
    lam[-1] = terminal
    carry = terminal.copy()
    for t in range(grad.shape[0] - 1, -1, -1):
        carry = grad[t] + carry @ a
        lam[t] = carry
    return lam


def relative_norm(error: np.ndarray, exact: np.ndarray) -> float:
    return float(np.linalg.norm(error.ravel()) / max(np.linalg.norm(exact.ravel()), 1.0e-30))


def choose_verifier_indices(
    rng: np.random.Generator,
    state_dim: int,
    sample_count: int,
) -> np.ndarray:
    sample_count = min(sample_count, state_dim)
    return np.sort(rng.choice(state_dim, size=sample_count, replace=False))


def make_condition_data(
    args: argparse.Namespace,
    condition: str,
    rng: np.random.Generator,
    a: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    h = hidden_features(rng, args.segment_length, args.hidden_feature_dim)
    p = low_rank_predictor(
        rng,
        args.hidden_feature_dim,
        args.state_dim,
        args.predictor_rank,
    )
    base_hat = h @ p.T
    verifier_count = (
        args.state_dim if condition == "full_verify_cost_fail" else args.verifier_sample_count
    )
    verifier_indices = choose_verifier_indices(rng, args.state_dim, verifier_count)
    residual = np.zeros((args.segment_length, args.state_dim), dtype=np.float64)

    if condition == "projection_aliasing_fail":
        complement = np.setdiff1d(np.arange(args.state_dim), verifier_indices)
        chosen = complement[: max(1, min(len(complement), args.verifier_sample_count))]
        residual[:, chosen] = rng.normal(scale=0.035, size=(args.segment_length, len(chosen)))
    elif condition == "poor_predictor_fail":
        residual = rng.normal(scale=0.025, size=(args.segment_length, args.state_dim))

    grad = base_hat[:-1] - base_hat[1:] @ a - residual
    predicted = base_hat.copy()
    if condition == "poor_predictor_fail":
        predicted = base_hat + rng.normal(scale=0.055, size=base_hat.shape)
    return predicted, grad, base_hat[-1], verifier_indices, verifier_count


def price_route(args: argparse.Namespace, verifier_count: int) -> dict[str, float]:
    predictor = args.predictor_cost_multiplier * (args.predictor_rank / float(args.state_dim))
    verifier = args.verifier_cost_multiplier * (verifier_count / float(args.state_dim))
    return {
        "low_rank_predictor": float(predictor),
        "sampled_bellman_verify": float(verifier),
        "residual_audit": float(args.audit_cost_ratio),
    }


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    a = transition_matrix(rng, args.state_dim)
    rho = float(max(abs(np.linalg.eigvals(a))))
    predicted, grad, terminal, verifier_indices, verifier_count = make_condition_data(
        args,
        condition,
        rng,
        a,
    )
    exact = exact_cotangent(grad, a, terminal)
    bellman = predicted[:-1] - predicted[1:] @ a - grad
    lambda_error = predicted - exact
    lambda_relative_error = relative_norm(lambda_error, exact)
    full_residual_ratio = relative_norm(bellman, predicted[:-1])
    sampled_residual_ratio = relative_norm(
        bellman[:, verifier_indices],
        predicted[:-1, verifier_indices],
    )
    projection_miss_ratio = float(
        max(full_residual_ratio - sampled_residual_ratio, 0.0)
        / max(full_residual_ratio, 1.0e-30)
    )
    invariant_bound = full_residual_ratio / max(1.0 - rho, 1.0e-12)
    invariant_bound_ratio = float(
        lambda_relative_error / max(invariant_bound, args.max_lambda_relative_error)
    )
    price_components = price_route(args, verifier_count)
    priced_work_ratio = float(sum(price_components.values()))
    projected_speedup = 1.0 / priced_work_ratio if priced_work_ratio > 0.0 else math.inf

    lambda_pass = lambda_relative_error <= args.max_lambda_relative_error
    full_residual_pass = full_residual_ratio <= args.max_full_bellman_residual_ratio
    projection_pass = projection_miss_ratio <= args.max_projection_miss_ratio
    invariant_pass = invariant_bound_ratio <= args.max_invariant_bound_ratio
    price_pass = priced_work_ratio <= args.max_priced_work_ratio
    passed = bool(lambda_pass and full_residual_pass and projection_pass and invariant_pass and price_pass)
    expected_pass = condition == "positive"
    identifying_correct = passed == expected_pass

    fail_reason = None
    if not passed:
        if not projection_pass:
            fail_reason = "random_subset_projection_misses_residual_debt"
        elif not full_residual_pass:
            fail_reason = "bellman_residual_exceeds_budget"
        elif not lambda_pass:
            fail_reason = "predictor_cotangent_error_exceeds_budget"
        elif not invariant_pass:
            fail_reason = "bellman_error_bound_does_not_cover_lambda_error"
        elif not price_pass:
            fail_reason = "bellman_verifier_recovers_dense_backward_cost"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=identifying_correct,
        fail_reason=fail_reason,
        segment_length=args.segment_length,
        state_dim=args.state_dim,
        predictor_rank=args.predictor_rank,
        verifier_sample_count=verifier_count,
        spectral_radius=rho,
        lambda_relative_error=lambda_relative_error,
        full_bellman_residual_ratio=full_residual_ratio,
        sampled_bellman_residual_ratio=sampled_residual_ratio,
        projection_miss_ratio=projection_miss_ratio,
        invariant_bound_ratio=invariant_bound_ratio,
        priced_work_ratio=priced_work_ratio,
        projected_speedup=float(projected_speedup),
        price_components=price_components,
    )


def finite_trial(trial: TrialResult) -> bool:
    values = (
        trial.spectral_radius,
        trial.lambda_relative_error,
        trial.full_bellman_residual_ratio,
        trial.sampled_bellman_residual_ratio,
        trial.projection_miss_ratio,
        trial.invariant_bound_ratio,
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
            "verifier_sample_count_mean": float(
                np.mean([row.verifier_sample_count for row in rows])
            ),
            "spectral_radius_mean": float(np.mean([row.spectral_radius for row in rows])),
            "lambda_relative_error_mean": float(
                np.mean([row.lambda_relative_error for row in rows])
            ),
            "lambda_relative_error_max": float(
                np.max([row.lambda_relative_error for row in rows])
            ),
            "full_bellman_residual_ratio_mean": float(
                np.mean([row.full_bellman_residual_ratio for row in rows])
            ),
            "full_bellman_residual_ratio_max": float(
                np.max([row.full_bellman_residual_ratio for row in rows])
            ),
            "sampled_bellman_residual_ratio_mean": float(
                np.mean([row.sampled_bellman_residual_ratio for row in rows])
            ),
            "projection_miss_ratio_mean": float(
                np.mean([row.projection_miss_ratio for row in rows])
            ),
            "projection_miss_ratio_max": float(
                np.max([row.projection_miss_ratio for row in rows])
            ),
            "invariant_bound_ratio_mean": float(
                np.mean([row.invariant_bound_ratio for row in rows])
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
    if args.segment_length < 2:
        raise ValueError("--segment-length must be at least 2")
    if args.state_dim < 4:
        raise ValueError("--state-dim must be at least 4")
    if not (1 <= args.predictor_rank <= args.state_dim):
        raise ValueError("--predictor-rank must be in [1, state_dim]")
    if not (1 <= args.verifier_sample_count <= args.state_dim):
        raise ValueError("--verifier-sample-count must be in [1, state_dim]")

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
        "positive_condition_low_bellman_residual": all(
            trial.full_bellman_residual_ratio <= args.max_full_bellman_residual_ratio
            for trial in trials
            if trial.condition == "positive"
        ),
        "full_verify_control_all_fail_price": all(
            (not trial.passed)
            and trial.fail_reason == "bellman_verifier_recovers_dense_backward_cost"
            for trial in trials
            if trial.condition == "full_verify_cost_fail"
        ),
        "projection_aliasing_control_all_fail_miss": all(
            (not trial.passed)
            and trial.fail_reason == "random_subset_projection_misses_residual_debt"
            for trial in trials
            if trial.condition == "projection_aliasing_fail"
        ),
        "poor_predictor_control_all_fail_residual": all(
            (not trial.passed)
            and trial.fail_reason == "bellman_residual_exceeds_budget"
            for trial in trials
            if trial.condition == "poor_predictor_fail"
        ),
        "all_conditions_identified": all(trial.identifying_correct for trial in trials),
    }
    decision = (
        "PASS_R501_S44_PREDICTOR_CORRECTOR_TUBE_GATE"
        if all(required_passes.values())
        else "FAIL_R501_S44_PREDICTOR_CORRECTOR_TUBE_GATE"
    )
    disposition = (
        "synthetic_positive_control_identified_not_training_speed_claim"
        if decision.startswith("PASS_")
        else "predictor_corrector_tube_gate_failed_or_non_identifying"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "CPU synthetic recurrent-cotangent predictor-corrector gate. It "
            "checks predictor error, Bellman residual, random-subset verifier "
            "coverage, and priced predictor plus sampled verification work. It "
            "is not heldout CE progress or GPU training-speed evidence."
        ),
        "finding": {
            "round": "R501",
            "aspect": "Speed",
            "id": "R501-S44",
            "title": "Recurrent-Cotangent Predictor-Corrector Tube",
        },
        "thresholds": {
            "max_lambda_relative_error": args.max_lambda_relative_error,
            "max_full_bellman_residual_ratio": args.max_full_bellman_residual_ratio,
            "max_projection_miss_ratio": args.max_projection_miss_ratio,
            "max_invariant_bound_ratio": args.max_invariant_bound_ratio,
            "max_priced_work_ratio": args.max_priced_work_ratio,
            "required_projected_speedup": 1.0 / args.max_priced_work_ratio,
        },
        "shape": {
            "seed": args.seed,
            "seed_count": args.seed_count,
            "segment_length": args.segment_length,
            "state_dim": args.state_dim,
            "hidden_feature_dim": args.hidden_feature_dim,
            "predictor_rank": args.predictor_rank,
            "verifier_sample_count": args.verifier_sample_count,
        },
        "required_passes": required_passes,
        "aggregate": aggregate(trials),
        "trials": [asdict(trial) for trial in trials],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    positive = result["aggregate"]["positive"]
    print(
        "[R501_S44_PREDICTOR_CORRECTOR_TUBE_GATE] "
        f"active=1 decision={decision} disposition={disposition} "
        f"positive_lambda_error_mean={positive['lambda_relative_error_mean']:.6e} "
        f"positive_full_bellman_residual_mean={positive['full_bellman_residual_ratio_mean']:.6e} "
        f"positive_sampled_bellman_residual_mean={positive['sampled_bellman_residual_ratio_mean']:.6e} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
