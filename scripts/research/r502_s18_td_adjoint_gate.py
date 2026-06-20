#!/usr/bin/env python3
"""R502-S18 Temporal Difference Adjoint Estimation gate.

This synthetic CPU gate checks whether a small forward TD value model can
replace dense recurrent BPTT in a stable low-rank adjoint regime. It verifies
Bellman residual, online gradient direction, and paid work, then exercises
controls for underfit value rank, auxiliary cost, noisy TD targets, unstable
dynamics, and stale value coefficients.
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
    "underfit_value_fail",
    "aux_cost_fail",
    "variance_fail",
    "unstable_dynamics_fail",
    "stale_value_fail",
]


@dataclass
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str
    sequence_len: int
    state_dim: int
    latent_rank: int
    aux_rank: int
    spectral_radius: float
    bellman_residual_fraction_max: float
    heldout_bellman_residual_fraction_max: float
    gradient_relative_error: float
    descent_cosine: float
    descent_ratio: float
    descent_per_paid_work: float
    priced_work_ratio: float
    aux_param_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=50218)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sequence-len", type=int, default=192)
    parser.add_argument("--episode-len", type=int, default=48)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--latent-rank", type=int, default=4)
    parser.add_argument("--underfit-rank", type=int, default=1)
    parser.add_argument("--expensive-aux-rank", type=int, default=28)
    parser.add_argument("--aux-overhead-ratio", type=float, default=0.04)
    parser.add_argument("--variance-noise-scale", type=float, default=0.70)
    parser.add_argument("--max-bellman-residual-fraction", type=float, default=0.05)
    parser.add_argument("--max-gradient-relative-error", type=float, default=0.08)
    parser.add_argument("--min-descent-cosine", type=float, default=0.98)
    parser.add_argument("--min-descent-per-paid-work", type=float, default=1.0)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.50)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def orthonormal(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    raw = rng.normal(size=(rows, cols))
    q, _ = np.linalg.qr(raw)
    return q[:, :cols]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / denom)


def relative_norm(numerator: np.ndarray, denominator: np.ndarray) -> float:
    denom = float(np.linalg.norm(denominator))
    if denom == 0.0:
        return 0.0 if float(np.linalg.norm(numerator)) == 0.0 else math.inf
    return float(np.linalg.norm(numerator) / denom)


def priced_work_ratio(args: argparse.Namespace, aux_rank: int) -> float:
    dense_units = args.state_dim * args.state_dim
    aux_units = aux_rank * args.state_dim + aux_rank * aux_rank
    return float(aux_units / dense_units + args.aux_overhead_ratio)


def aux_param_ratio(args: argparse.Namespace, aux_rank: int) -> float:
    core_units = args.state_dim * args.state_dim
    aux_units = aux_rank * args.state_dim + aux_rank
    return float(aux_units / core_units)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    basis = orthonormal(rng, args.state_dim, args.latent_rank)

    if condition == "unstable_dynamics_fail":
        decays = rng.uniform(1.02, 1.08, size=args.latent_rank)
    else:
        decays = rng.uniform(0.55, 0.82, size=args.latent_rank)
    spectral_radius = float(np.max(np.abs(decays)))
    loss_coeff = rng.uniform(0.40, 1.0, size=args.latent_rank)

    safe_decays = np.minimum(decays, 0.95)
    exact_value_coeff = loss_coeff / (1.0 - safe_decays * safe_decays)
    aux_rank = args.latent_rank
    value_coeff = exact_value_coeff.copy()

    if condition == "underfit_value_fail":
        aux_rank = args.underfit_rank
        value_coeff[aux_rank:] = 0.0
    elif condition == "aux_cost_fail":
        aux_rank = args.expensive_aux_rank
    elif condition == "stale_value_fail":
        stale_decays = decays * 0.55
        value_coeff = loss_coeff / (1.0 - stale_decays * stale_decays)

    exact_grad_matrix = np.zeros((args.state_dim, args.state_dim), dtype=np.float64)
    td_grad_matrix = np.zeros_like(exact_grad_matrix)
    bellman_residuals: list[float] = []
    heldout_bellman_residuals: list[float] = []
    t_index = 0

    while t_index < args.sequence_len:
        remaining = args.sequence_len - t_index
        episode_len = min(args.episode_len, remaining)
        z_values = []
        z = rng.normal(size=args.latent_rank)
        for _ in range(episode_len):
            z_values.append(z.copy())
            z = decays * z

        exact_lambdas = []
        estimate_lambdas = []
        state_vectors = []
        grad_vectors = []
        for local_t, z_t in enumerate(z_values):
            h_t = basis @ z_t
            grad_h = basis @ (loss_coeff * z_t)
            exact_lambda = basis @ (exact_value_coeff * z_t)
            estimate_z = value_coeff * z_t
            estimate_lambda = basis @ estimate_z
            if condition == "variance_fail":
                noise = basis @ rng.normal(
                    scale=args.variance_noise_scale * max(1.0, float(np.linalg.norm(estimate_z))),
                    size=args.latent_rank,
                )
                estimate_lambda = estimate_lambda + noise

            exact_lambdas.append(exact_lambda)
            estimate_lambdas.append(estimate_lambda)
            state_vectors.append(h_t)
            grad_vectors.append(grad_h)

            exact_grad_matrix += np.outer(exact_lambda, h_t)
            td_grad_matrix += np.outer(estimate_lambda, h_t)

        for local_t in range(episode_len - 1):
            z_next = z_values[local_t + 1]
            next_estimate = estimate_lambdas[local_t + 1]
            bootstrap = grad_vectors[local_t] + basis @ (decays * (basis.T @ next_estimate))
            residual = relative_norm(estimate_lambdas[local_t] - bootstrap, exact_lambdas[local_t])
            bellman_residuals.append(residual)
            if t_index + local_t >= args.sequence_len // 2:
                heldout_bellman_residuals.append(residual)

        t_index += episode_len

    gradient_error = relative_norm(exact_grad_matrix - td_grad_matrix, exact_grad_matrix)
    direction = cosine(td_grad_matrix, exact_grad_matrix)
    denom = float(np.sum(exact_grad_matrix * exact_grad_matrix))
    descent_ratio = 0.0 if denom == 0.0 else float(np.sum(td_grad_matrix * exact_grad_matrix) / denom)
    price = priced_work_ratio(args, aux_rank)
    descent_per_paid = descent_ratio / price
    bellman_max = max(bellman_residuals)
    heldout_bellman_max = max(heldout_bellman_residuals) if heldout_bellman_residuals else bellman_max

    failed: list[str] = []
    if not np.isfinite(
        [
            bellman_max,
            heldout_bellman_max,
            gradient_error,
            direction,
            descent_ratio,
            descent_per_paid,
            price,
            spectral_radius,
        ]
    ).all():
        failed.append("non_finite_metric")
    if spectral_radius >= 1.0:
        failed.append("unstable_spectral_radius")
    if bellman_max > args.max_bellman_residual_fraction:
        failed.append("bellman_residual")
    if heldout_bellman_max > args.max_bellman_residual_fraction:
        failed.append("heldout_bellman_residual")
    if gradient_error > args.max_gradient_relative_error:
        failed.append("gradient_error")
    if direction < args.min_descent_cosine:
        failed.append("descent_alignment")
    if descent_per_paid < args.min_descent_per_paid_work:
        failed.append("online_descent_per_paid_work")
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
        sequence_len=args.sequence_len,
        state_dim=args.state_dim,
        latent_rank=args.latent_rank,
        aux_rank=aux_rank,
        spectral_radius=spectral_radius,
        bellman_residual_fraction_max=bellman_max,
        heldout_bellman_residual_fraction_max=heldout_bellman_max,
        gradient_relative_error=gradient_error,
        descent_cosine=direction,
        descent_ratio=descent_ratio,
        descent_per_paid_work=descent_per_paid,
        priced_work_ratio=price,
        aux_param_ratio=aux_param_ratio(args, aux_rank),
    )


NUMERIC_FIELDS = [
    "sequence_len",
    "state_dim",
    "latent_rank",
    "aux_rank",
    "spectral_radius",
    "bellman_residual_fraction_max",
    "heldout_bellman_residual_fraction_max",
    "gradient_relative_error",
    "descent_cosine",
    "descent_ratio",
    "descent_per_paid_work",
    "priced_work_ratio",
    "aux_param_ratio",
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
        "positive_condition_bellman_and_gradient": (
            aggregates["positive"]["bellman_residual_fraction_max_max"] <= 0.05
            and aggregates["positive"]["heldout_bellman_residual_fraction_max_max"] <= 0.05
            and aggregates["positive"]["gradient_relative_error_max"] <= 0.08
        ),
        "positive_condition_descent_and_price": (
            aggregates["positive"]["descent_cosine_min"] >= 0.98
            and aggregates["positive"]["descent_per_paid_work_min"] >= 1.0
            and aggregates["positive"]["priced_work_ratio_max"] <= 0.50
            and aggregates["positive"]["spectral_radius_max"] < 1.0
        ),
        "underfit_control_all_fail_residual": all(
            (not result.passed)
            and (
                "bellman_residual" in result.fail_reason
                or "gradient_error" in result.fail_reason
                or "descent_alignment" in result.fail_reason
            )
            for result in by_condition["underfit_value_fail"]
        ),
        "aux_cost_control_all_fail_price": all(
            (not result.passed) and "priced_work" in result.fail_reason
            for result in by_condition["aux_cost_fail"]
        ),
        "variance_control_all_fail_descent_or_residual": all(
            (not result.passed)
            and (
                "bellman_residual" in result.fail_reason
                or "gradient_error" in result.fail_reason
                or "online_descent_per_paid_work" in result.fail_reason
            )
            for result in by_condition["variance_fail"]
        ),
        "unstable_control_all_fail_spectral": all(
            (not result.passed) and "unstable_spectral_radius" in result.fail_reason
            for result in by_condition["unstable_dynamics_fail"]
        ),
        "stale_value_control_all_fail_bellman": all(
            (not result.passed)
            and (
                "bellman_residual" in result.fail_reason
                or "heldout_bellman_residual" in result.fail_reason
                or "gradient_error" in result.fail_reason
            )
            for result in by_condition["stale_value_fail"]
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
    decision = "PASS_R502_S18_TD_ADJOINT_GATE" if all(checks.values()) else "FAIL_R502_S18_TD_ADJOINT_GATE"

    payload = {
        "round": "R502",
        "canonical_id": "R502-S18",
        "aspect": "Speed",
        "finding": "Temporal Difference (TD) Adjoint Estimation",
        "decision": decision,
        "disposition": "synthetic_positive_control_identified_not_training_speed_claim",
        "args": vars(args) | {"output": str(args.output) if args.output else None},
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU TD-adjoint value gate only; not heldout CE progress, "
            "not real corpus token efficiency, not GPU training speed, and not "
            "production v12 integration."
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_S18_TD_ADJOINT_GATE] "
        f"active=1 decision={decision} "
        f"disposition={payload['disposition']} "
        f"positive_bellman_residual_max={positive['bellman_residual_fraction_max_max']:.12e} "
        f"positive_gradient_error_max={positive['gradient_relative_error_max']:.12e} "
        f"positive_descent_cosine_min={positive['descent_cosine_min']:.12e} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.12f} "
        f"positive_descent_per_paid_work_min={positive['descent_per_paid_work_min']:.12f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
