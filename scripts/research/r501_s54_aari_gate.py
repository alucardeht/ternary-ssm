#!/usr/bin/env python3
"""R501 S54 amortized algebraic recurrent inversion gate.

This CPU diagnostic implements the R501 Speed Paradigm 6 finding:
"Amortized Algebraic Recurrent Inversion (AARI)".

The dense recurrent backward pass solves the triangular linear system
lambda_t = g_t + A^T lambda_{t+1}. In the positive fixture, A has a fixed
orthogonal modal basis, so the whole backward trace can be recovered by a
single modal algebraic inverse rather than step-by-step dense transitions.

The candidate survives only when the algebraic solution matches the dense
recurrence, satisfies the block-system residual, remains well conditioned, and
keeps amortized block inversion work below the dense route.

Controls exercise the declared risks: time-varying transitions break the static
inverse, near-unit poles make the inverse ill-conditioned and too expensive,
and state-dependent Jacobians break the linear recurrence contract.

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


SCHEMA = "r501_s54_aari_gate.v1"
CONDITIONS = (
    "positive",
    "time_varying_transition_fail",
    "near_unit_condition_fail",
    "nonlinear_jacobian_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    steps: int
    state_dim: int
    relative_solution_error: float
    block_residual_ratio: float
    max_pole_radius: float
    inverse_condition_proxy: float
    modal_transform_ratio: float
    block_inverse_ratio: float
    condition_surcharge_ratio: float
    residual_audit_ratio: float
    priced_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50154)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--max-relative-solution-error", type=float, default=1.0e-10)
    parser.add_argument("--max-block-residual-ratio", type=float, default=1.0e-10)
    parser.add_argument("--max-inverse-condition-proxy", type=float, default=80.0)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.50)
    parser.add_argument("--modal-transform-ratio", type=float, default=0.080)
    parser.add_argument("--residual-audit-ratio", type=float, default=0.040)
    return parser.parse_args()


def orthogonal_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def smooth_gradients(
    rng: np.random.Generator,
    steps: int,
    state_dim: int,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    waves = np.stack(
        [
            np.ones_like(t),
            np.sin(2.0 * math.pi * t + rng.uniform(-0.3, 0.3)),
            np.cos(2.0 * math.pi * t + rng.uniform(-0.4, 0.4)),
            np.sin(4.0 * math.pi * t + rng.uniform(-0.5, 0.5)),
            np.cos(6.0 * math.pi * t + rng.uniform(-0.6, 0.6)),
        ],
        axis=1,
    )
    weights = rng.normal(scale=0.32, size=(waves.shape[1], state_dim))
    grad = waves @ weights
    grad += rng.normal(scale=0.006, size=(steps, state_dim))
    return grad.astype(np.float64)


def condition_rho(
    rng: np.random.Generator,
    state_dim: int,
    condition: str,
) -> np.ndarray:
    if condition == "near_unit_condition_fail":
        return rng.uniform(0.986, 0.997, size=state_dim).astype(np.float64)
    return rng.uniform(0.18, 0.72, size=state_dim).astype(np.float64)


def dense_static_backward(grad: np.ndarray, basis: np.ndarray, rho: np.ndarray) -> np.ndarray:
    a = basis @ np.diag(rho) @ basis.T
    out = np.zeros_like(grad)
    carry = np.zeros(grad.shape[1], dtype=np.float64)
    for idx in range(grad.shape[0] - 1, -1, -1):
        carry = grad[idx] + carry @ a
        out[idx] = carry
    return out


def dense_time_varying_backward(
    grad: np.ndarray,
    basis: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    steps = grad.shape[0]
    out = np.zeros_like(grad)
    carry = np.zeros(grad.shape[1], dtype=np.float64)
    phase = np.linspace(0.0, 2.0 * math.pi, steps, dtype=np.float64)
    for idx in range(steps - 1, -1, -1):
        scale = 1.0 + 0.22 * math.sin(float(phase[idx]))
        rho_t = np.clip(rho * scale, 0.02, 0.93)
        a_t = basis @ np.diag(rho_t) @ basis.T
        carry = grad[idx] + carry @ a_t
        out[idx] = carry
    return out


def dense_nonlinear_jacobian_backward(
    grad: np.ndarray,
    basis: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    grad_modal = grad @ basis
    out_modal = np.zeros_like(grad_modal)
    carry = np.zeros(grad.shape[1], dtype=np.float64)
    phase = np.linspace(0.0, 2.0 * math.pi, grad.shape[0], dtype=np.float64)
    channel_phase = np.linspace(0.0, math.pi, grad.shape[1], dtype=np.float64)
    for idx in range(grad.shape[0] - 1, -1, -1):
        deriv = 0.58 + 0.34 * np.sin(phase[idx] + channel_phase) ** 2
        carry = grad_modal[idx] + rho * deriv * carry
        out_modal[idx] = carry
    return out_modal @ basis.T


def algebraic_static_inverse(grad: np.ndarray, basis: np.ndarray, rho: np.ndarray) -> np.ndarray:
    grad_modal = grad @ basis
    steps, state_dim = grad_modal.shape
    solved = np.zeros_like(grad_modal)
    offsets = np.arange(steps, dtype=np.float64)
    for channel in range(state_dim):
        powers = rho[channel] ** offsets
        for idx in range(steps):
            length = steps - idx
            solved[idx, channel] = float(np.dot(powers[:length], grad_modal[idx:steps, channel]))
    return solved @ basis.T


def block_residual(candidate: np.ndarray, grad: np.ndarray, basis: np.ndarray, rho: np.ndarray) -> float:
    a = basis @ np.diag(rho) @ basis.T
    residual = np.zeros_like(candidate)
    residual[-1] = candidate[-1] - grad[-1]
    for idx in range(candidate.shape[0] - 1):
        residual[idx] = candidate[idx] - grad[idx] - candidate[idx + 1] @ a
    return float(
        np.linalg.norm(residual.reshape(-1))
        / max(float(np.linalg.norm(candidate.reshape(-1))), 1.0e-30)
    )


def relative_error(candidate: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.linalg.norm((candidate - target).reshape(-1))
        / max(float(np.linalg.norm(target.reshape(-1))), 1.0e-30)
    )


def condition_proxy(rho: np.ndarray, steps: int) -> float:
    spectral_gap = max(1.0 - float(np.max(np.abs(rho))), 1.0e-12)
    horizon_gain = 1.0 - float(np.max(np.abs(rho))) ** steps
    return float(horizon_gain / spectral_gap)


def price_route(
    args: argparse.Namespace,
    inverse_condition_proxy: float,
) -> tuple[float, float, float, float]:
    block_inverse_ratio = float(args.state_dim / args.steps)
    if inverse_condition_proxy <= args.max_inverse_condition_proxy:
        surcharge = 0.0
    else:
        surcharge = min(
            0.60,
            0.50 * (inverse_condition_proxy / args.max_inverse_condition_proxy - 1.0),
        )
    priced = (
        args.modal_transform_ratio
        + block_inverse_ratio
        + surcharge
        + args.residual_audit_ratio
    )
    return block_inverse_ratio, surcharge, priced, 1.0 / priced


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    basis = orthogonal_matrix(rng, args.state_dim)
    rho = condition_rho(rng, args.state_dim, condition)
    grad = smooth_gradients(rng, args.steps, args.state_dim)
    candidate = algebraic_static_inverse(grad, basis, rho)

    if condition == "time_varying_transition_fail":
        reference = dense_time_varying_backward(grad, basis, rho)
    elif condition == "nonlinear_jacobian_fail":
        reference = dense_nonlinear_jacobian_backward(grad, basis, rho)
    else:
        reference = dense_static_backward(grad, basis, rho)

    solution_error = relative_error(candidate, reference)
    residual_ratio = block_residual(candidate, grad, basis, rho)
    inv_condition = condition_proxy(rho, args.steps)
    block_inverse_ratio, surcharge, priced_work, projected_speedup = price_route(
        args,
        inv_condition,
    )

    solution_pass = solution_error <= args.max_relative_solution_error
    residual_pass = residual_ratio <= args.max_block_residual_ratio
    condition_pass = inv_condition <= args.max_inverse_condition_proxy
    price_pass = priced_work <= args.max_priced_work_ratio
    passed = bool(solution_pass and residual_pass and condition_pass and price_pass)
    expected_pass = condition == "positive"
    identifying_correct = passed == expected_pass

    fail_reason = None
    if not passed:
        if condition == "time_varying_transition_fail" and not solution_pass:
            fail_reason = "time_varying_transition_breaks_static_inverse"
        elif condition == "near_unit_condition_fail" and (not condition_pass or not price_pass):
            fail_reason = "ill_conditioned_inverse_recovers_dense_backward_cost"
        elif condition == "nonlinear_jacobian_fail" and not solution_pass:
            fail_reason = "state_dependent_jacobian_breaks_linear_inverse"
        elif not solution_pass:
            fail_reason = "time_varying_transition_breaks_static_inverse"
        elif not condition_pass or not price_pass:
            fail_reason = "ill_conditioned_inverse_recovers_dense_backward_cost"
        elif not residual_pass:
            fail_reason = "block_system_residual_too_large"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=identifying_correct,
        fail_reason=fail_reason,
        steps=args.steps,
        state_dim=args.state_dim,
        relative_solution_error=float(solution_error),
        block_residual_ratio=float(residual_ratio),
        max_pole_radius=float(np.max(np.abs(rho))),
        inverse_condition_proxy=float(inv_condition),
        modal_transform_ratio=float(args.modal_transform_ratio),
        block_inverse_ratio=float(block_inverse_ratio),
        condition_surcharge_ratio=float(surcharge),
        residual_audit_ratio=float(args.residual_audit_ratio),
        priced_work_ratio=float(priced_work),
        projected_speedup=float(projected_speedup),
    )


def finite_trial(trial: TrialResult) -> bool:
    values = (
        trial.relative_solution_error,
        trial.block_residual_ratio,
        trial.max_pole_radius,
        trial.inverse_condition_proxy,
        trial.modal_transform_ratio,
        trial.block_inverse_ratio,
        trial.condition_surcharge_ratio,
        trial.residual_audit_ratio,
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
            "relative_solution_error_mean": float(
                np.mean([row.relative_solution_error for row in rows])
            ),
            "relative_solution_error_max": float(
                np.max([row.relative_solution_error for row in rows])
            ),
            "block_residual_ratio_mean": float(
                np.mean([row.block_residual_ratio for row in rows])
            ),
            "block_residual_ratio_max": float(
                np.max([row.block_residual_ratio for row in rows])
            ),
            "max_pole_radius_mean": float(np.mean([row.max_pole_radius for row in rows])),
            "inverse_condition_proxy_mean": float(
                np.mean([row.inverse_condition_proxy for row in rows])
            ),
            "inverse_condition_proxy_max": float(
                np.max([row.inverse_condition_proxy for row in rows])
            ),
            "block_inverse_ratio_mean": float(
                np.mean([row.block_inverse_ratio for row in rows])
            ),
            "condition_surcharge_ratio_mean": float(
                np.mean([row.condition_surcharge_ratio for row in rows])
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
    if args.steps < 4:
        raise ValueError("--steps must be at least 4")
    if args.state_dim < 2:
        raise ValueError("--state-dim must be at least 2")

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
        "positive_condition_exact_solution_and_residual": all(
            trial.relative_solution_error <= args.max_relative_solution_error
            and trial.block_residual_ratio <= args.max_block_residual_ratio
            for trial in trials
            if trial.condition == "positive"
        ),
        "positive_condition_well_conditioned_and_priced": all(
            trial.inverse_condition_proxy <= args.max_inverse_condition_proxy
            and trial.priced_work_ratio <= args.max_priced_work_ratio
            for trial in trials
            if trial.condition == "positive"
        ),
        "time_varying_control_all_fail_static_inverse": all(
            (not trial.passed)
            and trial.fail_reason == "time_varying_transition_breaks_static_inverse"
            for trial in trials
            if trial.condition == "time_varying_transition_fail"
        ),
        "near_unit_control_all_fail_condition_price": all(
            (not trial.passed)
            and trial.fail_reason == "ill_conditioned_inverse_recovers_dense_backward_cost"
            for trial in trials
            if trial.condition == "near_unit_condition_fail"
        ),
        "nonlinear_control_all_fail_linear_inverse": all(
            (not trial.passed)
            and trial.fail_reason == "state_dependent_jacobian_breaks_linear_inverse"
            for trial in trials
            if trial.condition == "nonlinear_jacobian_fail"
        ),
        "all_conditions_identified": all(trial.identifying_correct for trial in trials),
    }
    decision = (
        "PASS_R501_S54_AARI_GATE"
        if all(required_passes.values())
        else "FAIL_R501_S54_AARI_GATE"
    )
    disposition = (
        "synthetic_positive_control_identified_not_training_speed_claim"
        if decision.startswith("PASS_")
        else "aari_gate_failed_or_non_identifying"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "CPU synthetic amortized algebraic recurrent inversion gate. It "
            "checks exact modal inversion of a fixed linear recurrent backward "
            "system, block residuals, conditioning, and amortized pricing. It "
            "is not heldout CE progress or GPU training-speed evidence."
        ),
        "finding": {
            "round": "R501",
            "aspect": "Speed",
            "id": "R501-S54",
            "title": "Amortized Algebraic Recurrent Inversion",
        },
        "thresholds": {
            "max_relative_solution_error": args.max_relative_solution_error,
            "max_block_residual_ratio": args.max_block_residual_ratio,
            "max_inverse_condition_proxy": args.max_inverse_condition_proxy,
            "max_priced_work_ratio": args.max_priced_work_ratio,
        },
        "shape": {
            "seed": args.seed,
            "seed_count": args.seed_count,
            "steps": args.steps,
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
        "[R501_S54_AARI_GATE] "
        f"active=1 decision={decision} disposition={disposition} "
        f"positive_solution_error_mean={positive['relative_solution_error_mean']:.6e} "
        f"positive_block_residual_mean={positive['block_residual_ratio_mean']:.6e} "
        f"positive_condition_proxy_mean={positive['inverse_condition_proxy_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
