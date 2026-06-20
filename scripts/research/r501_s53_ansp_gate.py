#!/usr/bin/env python3
"""R501 S53 autoregressive null-space projection gate.

This CPU diagnostic implements the R501 Speed Paradigm 5 finding:
"Autoregressive Null-Space Projection (ANSP)".

The CE gradient visible to the next-token logits lives in the row space of the
output head. The gate constructs dense hidden cotangents with both active
output-head components and null-space components, then drops the null component
before the recurrent backward transition. The candidate survives only when the
projected cotangent preserves the immediate logits, preserves one-step future
logits under a transition that respects the active/null decomposition, maintains
the active descent direction, and keeps projected backward work below the dense
route.

Controls exercise the declared risks: a high-rank output head recovers dense
backward cost, a leaky projector corrupts immediate logits, and transition
coupling can turn an immediate null component into a future active-logit error.

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


SCHEMA = "r501_s53_ansp_gate.v1"
CONDITIONS = (
    "positive",
    "full_rank_head_fail",
    "leaky_projector_fail",
    "future_coupling_fail",
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
    output_dim: int
    active_rank: int
    active_rank_fraction: float
    null_energy_fraction: float
    immediate_logit_residual_ratio: float
    future_logit_residual_ratio: float
    active_descent_alignment_ratio: float
    projected_backward_fraction: float
    projector_update_ratio: float
    residual_audit_ratio: float
    priced_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50153)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--active-rank", type=int, default=8)
    parser.add_argument("--full-rank", type=int, default=48)
    parser.add_argument("--max-immediate-logit-residual", type=float, default=1.0e-10)
    parser.add_argument("--max-future-logit-residual", type=float, default=1.0e-10)
    parser.add_argument("--min-descent-alignment-ratio", type=float, default=0.999999)
    parser.add_argument("--min-null-energy-fraction", type=float, default=0.50)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.70)
    parser.add_argument("--projector-update-ratio", type=float, default=0.035)
    parser.add_argument("--residual-audit-ratio", type=float, default=0.040)
    parser.add_argument("--leak-angle", type=float, default=0.18)
    parser.add_argument("--future-coupling-scale", type=float, default=0.35)
    return parser.parse_args()


def orthogonal_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def smooth_coefficients(
    rng: np.random.Generator,
    steps: int,
    rank: int,
    scale: float,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    waves = np.stack(
        [
            np.ones_like(t),
            np.sin(2.0 * math.pi * t + rng.uniform(-0.3, 0.3)),
            np.cos(2.0 * math.pi * t + rng.uniform(-0.3, 0.3)),
            np.sin(4.0 * math.pi * t + rng.uniform(-0.5, 0.5)),
            np.cos(6.0 * math.pi * t + rng.uniform(-0.7, 0.7)),
        ],
        axis=1,
    )
    weights = rng.normal(scale=scale, size=(waves.shape[1], rank))
    coeff = waves @ weights
    coeff += rng.normal(scale=0.015 * scale, size=(steps, rank))
    return coeff.astype(np.float64)


def make_output_head(
    rng: np.random.Generator,
    active_basis: np.ndarray,
    output_dim: int,
    active_rank: int,
) -> np.ndarray:
    left = rng.normal(size=(output_dim, active_rank))
    q, _ = np.linalg.qr(left)
    singular = np.linspace(1.2, 0.45, active_rank, dtype=np.float64)
    return q[:, :active_rank] @ np.diag(singular) @ active_basis.T


def make_transition(
    rng: np.random.Generator,
    basis: np.ndarray,
    active_rank: int,
    condition: str,
    coupling_scale: float,
) -> np.ndarray:
    state_dim = basis.shape[0]
    null_dim = state_dim - active_rank
    block = np.zeros((state_dim, state_dim), dtype=np.float64)
    active_diag = np.linspace(0.25, 0.70, active_rank, dtype=np.float64)
    null_diag = np.linspace(0.10, 0.55, null_dim, dtype=np.float64)
    block[:active_rank, :active_rank] = np.diag(active_diag)
    block[active_rank:, active_rank:] = np.diag(null_diag)

    if condition == "future_coupling_fail":
        coupling = rng.normal(scale=coupling_scale, size=(active_rank, null_dim))
        block[:active_rank, active_rank:] = coupling

    return basis @ block @ basis.T


def projector_basis(
    active_basis: np.ndarray,
    null_basis: np.ndarray,
    condition: str,
    leak_angle: float,
) -> np.ndarray:
    if condition != "leaky_projector_fail":
        return active_basis
    rank = active_basis.shape[1]
    leaked = (
        math.cos(leak_angle) * active_basis
        + math.sin(leak_angle) * null_basis[:, :rank]
    )
    q, _ = np.linalg.qr(leaked)
    return q[:, :rank]


def make_dense_cotangents(
    rng: np.random.Generator,
    steps: int,
    active_basis: np.ndarray,
    null_basis: np.ndarray,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active_rank = active_basis.shape[1]
    null_rank = null_basis.shape[1]
    active_coeff = smooth_coefficients(rng, steps, active_rank, scale=0.42)
    null_scale = 0.75
    if condition == "full_rank_head_fail":
        null_scale = 0.42
    null_coeff = smooth_coefficients(rng, steps, null_rank, scale=null_scale)
    active = active_coeff @ active_basis.T
    null = null_coeff @ null_basis.T
    dense = active + null
    return dense.astype(np.float64), active.astype(np.float64), null.astype(np.float64)


def relative_norm(numerator: np.ndarray, denominator: np.ndarray) -> float:
    return float(
        np.linalg.norm(numerator.reshape(-1))
        / max(float(np.linalg.norm(denominator.reshape(-1))), 1.0e-30)
    )


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    state_dim = args.state_dim
    if condition == "full_rank_head_fail":
        active_rank = args.full_rank
    else:
        active_rank = args.active_rank
    if not (0 < active_rank < state_dim):
        raise ValueError("active rank must be between 1 and state_dim - 1")
    if args.output_dim < active_rank:
        raise ValueError("output_dim must be at least active_rank")

    basis = orthogonal_matrix(rng, state_dim)
    active_basis = basis[:, :active_rank]
    null_basis = basis[:, active_rank:]
    output_head = make_output_head(rng, active_basis, args.output_dim, active_rank)
    dense, active_reference, null_reference = make_dense_cotangents(
        rng,
        args.steps,
        active_basis,
        null_basis,
        condition,
    )
    candidate_basis = projector_basis(
        active_basis,
        null_basis,
        condition,
        args.leak_angle,
    )
    projector = candidate_basis @ candidate_basis.T
    candidate = dense @ projector
    transition = make_transition(
        rng,
        basis,
        active_rank,
        condition,
        args.future_coupling_scale,
    )

    dropped = dense - candidate
    immediate_residual = dropped @ output_head.T
    immediate_reference = dense @ output_head.T
    future_residual = (dropped @ transition.T) @ output_head.T
    future_reference = (dense @ transition.T) @ output_head.T
    immediate_ratio = relative_norm(immediate_residual, immediate_reference)
    future_ratio = relative_norm(future_residual, future_reference)
    numerator = float(np.sum(candidate * active_reference))
    denominator = max(float(np.sum(active_reference * active_reference)), 1.0e-30)
    descent_alignment = numerator / denominator
    null_energy = float(np.sum(null_reference * null_reference))
    dense_energy = max(float(np.sum(dense * dense)), 1.0e-30)
    null_energy_fraction = null_energy / dense_energy
    projected_backward_fraction = active_rank / state_dim
    priced_work = (
        projected_backward_fraction
        + args.projector_update_ratio
        + args.residual_audit_ratio
    )
    projected_speedup = 1.0 / priced_work if priced_work > 0.0 else math.inf

    immediate_pass = immediate_ratio <= args.max_immediate_logit_residual
    future_pass = future_ratio <= args.max_future_logit_residual
    descent_pass = descent_alignment >= args.min_descent_alignment_ratio
    null_material_pass = null_energy_fraction >= args.min_null_energy_fraction
    price_pass = priced_work <= args.max_priced_work_ratio
    passed = bool(
        immediate_pass
        and future_pass
        and descent_pass
        and null_material_pass
        and price_pass
    )
    expected_pass = condition == "positive"
    identifying_correct = passed == expected_pass

    fail_reason = None
    if not passed:
        if condition == "full_rank_head_fail" and not price_pass:
            fail_reason = "high_rank_output_head_recovers_dense_backward_cost"
        elif condition == "leaky_projector_fail" and not immediate_pass:
            fail_reason = "leaky_projector_corrupts_immediate_logits"
        elif condition == "future_coupling_fail" and not future_pass:
            fail_reason = "null_component_couples_into_future_logits"
        elif not immediate_pass:
            fail_reason = "leaky_projector_corrupts_immediate_logits"
        elif not future_pass:
            fail_reason = "null_component_couples_into_future_logits"
        elif not price_pass:
            fail_reason = "high_rank_output_head_recovers_dense_backward_cost"
        elif not null_material_pass:
            fail_reason = "no_material_null_space_to_drop"
        elif not descent_pass:
            fail_reason = "projected_gradient_loses_active_descent"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=identifying_correct,
        fail_reason=fail_reason,
        steps=args.steps,
        state_dim=state_dim,
        output_dim=args.output_dim,
        active_rank=active_rank,
        active_rank_fraction=float(active_rank / state_dim),
        null_energy_fraction=float(null_energy_fraction),
        immediate_logit_residual_ratio=float(immediate_ratio),
        future_logit_residual_ratio=float(future_ratio),
        active_descent_alignment_ratio=float(descent_alignment),
        projected_backward_fraction=float(projected_backward_fraction),
        projector_update_ratio=float(args.projector_update_ratio),
        residual_audit_ratio=float(args.residual_audit_ratio),
        priced_work_ratio=float(priced_work),
        projected_speedup=float(projected_speedup),
    )


def finite_trial(trial: TrialResult) -> bool:
    values = (
        trial.active_rank_fraction,
        trial.null_energy_fraction,
        trial.immediate_logit_residual_ratio,
        trial.future_logit_residual_ratio,
        trial.active_descent_alignment_ratio,
        trial.projected_backward_fraction,
        trial.projector_update_ratio,
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
            "active_rank_mean": float(np.mean([row.active_rank for row in rows])),
            "active_rank_fraction_mean": float(
                np.mean([row.active_rank_fraction for row in rows])
            ),
            "null_energy_fraction_mean": float(
                np.mean([row.null_energy_fraction for row in rows])
            ),
            "immediate_logit_residual_ratio_mean": float(
                np.mean([row.immediate_logit_residual_ratio for row in rows])
            ),
            "immediate_logit_residual_ratio_max": float(
                np.max([row.immediate_logit_residual_ratio for row in rows])
            ),
            "future_logit_residual_ratio_mean": float(
                np.mean([row.future_logit_residual_ratio for row in rows])
            ),
            "future_logit_residual_ratio_max": float(
                np.max([row.future_logit_residual_ratio for row in rows])
            ),
            "active_descent_alignment_ratio_mean": float(
                np.mean([row.active_descent_alignment_ratio for row in rows])
            ),
            "projected_backward_fraction_mean": float(
                np.mean([row.projected_backward_fraction for row in rows])
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
    if args.steps < 2:
        raise ValueError("--steps must be at least 2")
    if args.state_dim < 4:
        raise ValueError("--state-dim must be at least 4")

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
        "positive_condition_exact_logit_residual": all(
            trial.immediate_logit_residual_ratio <= args.max_immediate_logit_residual
            and trial.future_logit_residual_ratio <= args.max_future_logit_residual
            for trial in trials
            if trial.condition == "positive"
        ),
        "positive_condition_sparse_and_priced": all(
            trial.priced_work_ratio <= args.max_priced_work_ratio
            and trial.projected_backward_fraction < 0.50
            for trial in trials
            if trial.condition == "positive"
        ),
        "full_rank_control_all_fail_price": all(
            (not trial.passed)
            and trial.fail_reason == "high_rank_output_head_recovers_dense_backward_cost"
            for trial in trials
            if trial.condition == "full_rank_head_fail"
        ),
        "leaky_projector_control_all_fail_logits": all(
            (not trial.passed)
            and trial.fail_reason == "leaky_projector_corrupts_immediate_logits"
            for trial in trials
            if trial.condition == "leaky_projector_fail"
        ),
        "future_coupling_control_all_fail_logits": all(
            (not trial.passed)
            and trial.fail_reason == "null_component_couples_into_future_logits"
            for trial in trials
            if trial.condition == "future_coupling_fail"
        ),
        "all_conditions_identified": all(trial.identifying_correct for trial in trials),
    }
    decision = (
        "PASS_R501_S53_ANSP_GATE"
        if all(required_passes.values())
        else "FAIL_R501_S53_ANSP_GATE"
    )
    disposition = (
        "synthetic_positive_control_identified_not_training_speed_claim"
        if decision.startswith("PASS_")
        else "ansp_gate_failed_or_non_identifying"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "CPU synthetic autoregressive null-space projection gate. It checks "
            "output-head row-space projection, immediate and one-step future "
            "logit residuals, active descent preservation, and projected "
            "backward pricing. It is not heldout CE progress or GPU "
            "training-speed evidence."
        ),
        "finding": {
            "round": "R501",
            "aspect": "Speed",
            "id": "R501-S53",
            "title": "Autoregressive Null-Space Projection",
        },
        "thresholds": {
            "max_immediate_logit_residual": args.max_immediate_logit_residual,
            "max_future_logit_residual": args.max_future_logit_residual,
            "min_descent_alignment_ratio": args.min_descent_alignment_ratio,
            "min_null_energy_fraction": args.min_null_energy_fraction,
            "max_priced_work_ratio": args.max_priced_work_ratio,
        },
        "shape": {
            "seed": args.seed,
            "seed_count": args.seed_count,
            "steps": args.steps,
            "state_dim": args.state_dim,
            "output_dim": args.output_dim,
            "active_rank": args.active_rank,
        },
        "required_passes": required_passes,
        "aggregate": aggregate(trials),
        "trials": [asdict(trial) for trial in trials],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    positive = result["aggregate"]["positive"]
    print(
        "[R501_S53_ANSP_GATE] "
        f"active=1 decision={decision} disposition={disposition} "
        f"positive_active_rank_fraction_mean={positive['active_rank_fraction_mean']:.6f} "
        f"positive_null_energy_fraction_mean={positive['null_energy_fraction_mean']:.6f} "
        f"positive_immediate_logit_residual_mean={positive['immediate_logit_residual_ratio_mean']:.6e} "
        f"positive_future_logit_residual_mean={positive['future_logit_residual_ratio_mean']:.6e} "
        f"positive_descent_alignment_mean={positive['active_descent_alignment_ratio_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
