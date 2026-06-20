#!/usr/bin/env python3
"""R501 S37 subspace-projected forward eligibility gate.

This CPU diagnostic implements the R501 Speed Math 4 finding:
"Subspace-Projected Forward Eligibility".

The gate models per-step transition-parameter gradients as dense eligibility
vectors grad_t in R^(d*d). It then projects them onto dyadic basis vectors
induced by the transition eigenspace. The finding survives only when a small
rank captures the gradient with a certified residual bound and the priced
projected trace route is below the dense recurrent backward budget.

Controls exercise the declared kill criteria: a flat Fisher/gradient spectrum,
a gradient subspace misaligned with the transition eigenspace, and a short
segment where the basis update cannot be amortized.

This is a synthetic CPU gate. It is not measured heldout CE progress, not GPU
training speed, and not a production v12 training-path promotion.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s37_subspace_forward_eligibility_gate.v1"
CONDITIONS = (
    "positive",
    "flat_spectrum_fail",
    "misaligned_krylov_fail",
    "short_segment_pricing_fail",
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
    selected_rank: int
    max_rank_budget: int
    mean_relative_residual: float
    max_relative_residual: float
    captured_energy_fraction: float
    priced_work_ratio: float
    projected_speedup: float
    projected_storage_ratio: float
    exact_trace_bytes: float
    projected_trace_bytes: float
    fisher_effective_rank: float
    price_components: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50137)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--segment-length", type=int, default=96)
    parser.add_argument("--state-dim", type=int, default=24)
    parser.add_argument("--short-segment-length", type=int, default=4)
    parser.add_argument("--max-relative-residual", type=float, default=2.0e-2)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.90)
    parser.add_argument("--max-projected-rank", type=int, default=8)
    parser.add_argument("--residual-audit-ratio", type=float, default=0.035)
    return parser.parse_args()


def orthogonal_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def smooth_coefficients(
    rng: np.random.Generator,
    segment_length: int,
    state_dim: int,
    active_modes: int,
    flat: bool,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, segment_length, dtype=np.float64)
    waves = np.stack(
        [
            np.ones_like(t),
            np.sin(2.0 * math.pi * t),
            np.cos(2.0 * math.pi * t),
            np.sin(4.0 * math.pi * t + 0.3),
            np.cos(6.0 * math.pi * t - 0.5),
        ],
        axis=1,
    )
    h_coeff = np.zeros((segment_length, state_dim), dtype=np.float64)
    c_coeff = np.zeros((segment_length, state_dim), dtype=np.float64)
    mode_count = state_dim if flat else active_modes
    for mode in range(mode_count):
        if flat:
            scale = 0.45
        else:
            scale = 0.80 / (mode + 1.0)
        h_w = rng.normal(scale=scale, size=waves.shape[1])
        c_w = rng.normal(scale=scale, size=waves.shape[1])
        h_coeff[:, mode] = waves @ h_w
        c_coeff[:, mode] = waves @ c_w
    return h_coeff, c_coeff


def dense_vectors_from_coefficients(coeff: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return coeff @ basis.T


def gradient_rows(c_dense: np.ndarray, h_dense: np.ndarray) -> np.ndarray:
    rows = [
        np.outer(c_dense[idx], h_dense[idx]).reshape(-1)
        for idx in range(c_dense.shape[0])
    ]
    return np.stack(rows, axis=0).astype(np.float64)


def dyad_basis(transition_basis: np.ndarray) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for left in range(transition_basis.shape[1]):
        for right in range(transition_basis.shape[1]):
            vectors.append(
                np.outer(transition_basis[:, left], transition_basis[:, right]).reshape(-1)
            )
    return np.stack(vectors, axis=0).astype(np.float64)


def relative_residuals(g: np.ndarray, basis_rows: np.ndarray, order: np.ndarray, rank: int) -> np.ndarray:
    if rank <= 0:
        recon = np.zeros_like(g)
    else:
        selected = basis_rows[order[:rank]]
        coeff = g @ selected.T
        recon = coeff @ selected
    residual = g - recon
    numerator = np.linalg.norm(residual, axis=1)
    denominator = np.maximum(np.linalg.norm(g, axis=1), 1.0e-30)
    return numerator / denominator


def energy_order(g: np.ndarray, basis_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coeff = g @ basis_rows.T
    energy = np.sum(coeff * coeff, axis=0)
    order = np.argsort(-energy)
    return order, energy


def effective_rank(energy: np.ndarray) -> float:
    positive = energy[energy > 0.0]
    if positive.size == 0:
        return 0.0
    probs = positive / float(np.sum(positive))
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1.0e-30))))
    return float(math.exp(entropy))


def select_rank(
    g: np.ndarray,
    basis_rows: np.ndarray,
    order: np.ndarray,
    max_relative_residual: float,
) -> tuple[int, float, float, float]:
    total_energy = float(np.sum(g * g))
    if total_energy <= 0.0:
        return 0, 0.0, 0.0, 1.0
    for rank in range(1, basis_rows.shape[0] + 1):
        residuals = relative_residuals(g, basis_rows, order, rank)
        if float(np.max(residuals)) <= max_relative_residual:
            captured = 1.0 - float(np.sum((residuals * np.linalg.norm(g, axis=1)) ** 2)) / total_energy
            return rank, float(np.mean(residuals)), float(np.max(residuals)), captured
    residuals = relative_residuals(g, basis_rows, order, basis_rows.shape[0])
    return (
        basis_rows.shape[0],
        float(np.mean(residuals)),
        float(np.max(residuals)),
        1.0,
    )


def price_route(
    selected_rank: int,
    segment_length: int,
    state_dim: int,
    residual_audit_ratio: float,
) -> dict[str, float]:
    return {
        "projected_trace_update": float(selected_rank / state_dim),
        "basis_update_amortized": float(selected_rank / segment_length),
        "residual_audit": float(residual_audit_ratio),
    }


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    segment_length = (
        args.short_segment_length
        if condition == "short_segment_pricing_fail"
        else args.segment_length
    )
    state_dim = args.state_dim
    transition_basis = orthogonal_matrix(rng, state_dim)
    active_modes = 2

    if condition == "misaligned_krylov_fail":
        data_basis = orthogonal_matrix(rng, state_dim)
        h_coeff, c_coeff = smooth_coefficients(
            rng, segment_length, state_dim, active_modes, flat=False
        )
        h_dense = dense_vectors_from_coefficients(h_coeff, data_basis)
        c_dense = dense_vectors_from_coefficients(c_coeff, data_basis)
    else:
        flat = condition == "flat_spectrum_fail"
        h_coeff, c_coeff = smooth_coefficients(
            rng, segment_length, state_dim, active_modes, flat=flat
        )
        h_dense = dense_vectors_from_coefficients(h_coeff, transition_basis)
        c_dense = dense_vectors_from_coefficients(c_coeff, transition_basis)

    g = gradient_rows(c_dense, h_dense)
    basis_rows = dyad_basis(transition_basis)
    order, energy = energy_order(g, basis_rows)
    selected_rank, mean_residual, max_residual, captured = select_rank(
        g, basis_rows, order, args.max_relative_residual
    )
    price_components = price_route(
        selected_rank,
        segment_length,
        state_dim,
        args.residual_audit_ratio,
    )
    priced_work_ratio = float(sum(price_components.values()))
    projected_speedup = 1.0 / priced_work_ratio if priced_work_ratio > 0.0 else math.inf
    exact_trace_bytes = float(segment_length * state_dim * state_dim * 4)
    projected_trace_bytes = float((segment_length * selected_rank + selected_rank * state_dim * state_dim) * 4)
    projected_storage_ratio = projected_trace_bytes / exact_trace_bytes

    rank_pass = selected_rank <= args.max_projected_rank
    residual_pass = max_residual <= args.max_relative_residual
    price_pass = priced_work_ratio <= args.max_priced_work_ratio
    passed = bool(rank_pass and residual_pass and price_pass)
    expected_pass = condition == "positive"
    identifying_correct = passed == expected_pass

    fail_reason = None
    if not passed:
        if not rank_pass:
            fail_reason = "projected_rank_exceeds_low_rank_budget"
        elif not residual_pass:
            fail_reason = "projection_residual_exceeds_certificate"
        elif not price_pass:
            fail_reason = "projected_trace_not_below_dense_backward_budget"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=identifying_correct,
        fail_reason=fail_reason,
        segment_length=segment_length,
        state_dim=state_dim,
        selected_rank=selected_rank,
        max_rank_budget=args.max_projected_rank,
        mean_relative_residual=mean_residual,
        max_relative_residual=max_residual,
        captured_energy_fraction=captured,
        priced_work_ratio=priced_work_ratio,
        projected_speedup=float(projected_speedup),
        projected_storage_ratio=projected_storage_ratio,
        exact_trace_bytes=exact_trace_bytes,
        projected_trace_bytes=projected_trace_bytes,
        fisher_effective_rank=effective_rank(energy),
        price_components=price_components,
    )


def finite_trial(trial: TrialResult) -> bool:
    values = (
        trial.mean_relative_residual,
        trial.max_relative_residual,
        trial.captured_energy_fraction,
        trial.priced_work_ratio,
        trial.projected_speedup,
        trial.projected_storage_ratio,
        trial.exact_trace_bytes,
        trial.projected_trace_bytes,
        trial.fisher_effective_rank,
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
            "selected_rank_mean": float(np.mean([row.selected_rank for row in rows])),
            "selected_rank_max": float(np.max([row.selected_rank for row in rows])),
            "mean_relative_residual_mean": float(
                np.mean([row.mean_relative_residual for row in rows])
            ),
            "max_relative_residual_max": float(
                np.max([row.max_relative_residual for row in rows])
            ),
            "captured_energy_fraction_mean": float(
                np.mean([row.captured_energy_fraction for row in rows])
            ),
            "priced_work_ratio_mean": float(
                np.mean([row.priced_work_ratio for row in rows])
            ),
            "projected_speedup_mean": float(
                np.mean([row.projected_speedup for row in rows])
            ),
            "projected_storage_ratio_mean": float(
                np.mean([row.projected_storage_ratio for row in rows])
            ),
            "fisher_effective_rank_mean": float(
                np.mean([row.fisher_effective_rank for row in rows])
            ),
        }
    return out


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.segment_length < 4:
        raise ValueError("--segment-length must be at least 4")
    if args.short_segment_length < 4:
        raise ValueError("--short-segment-length must be at least 4")
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
        "positive_condition_residual_certified": all(
            trial.max_relative_residual <= args.max_relative_residual
            for trial in trials
            if trial.condition == "positive"
        ),
        "positive_condition_rank_within_budget": all(
            trial.selected_rank <= args.max_projected_rank
            for trial in trials
            if trial.condition == "positive"
        ),
        "flat_spectrum_control_all_fail_rank": all(
            (not trial.passed)
            and trial.fail_reason == "projected_rank_exceeds_low_rank_budget"
            for trial in trials
            if trial.condition == "flat_spectrum_fail"
        ),
        "misaligned_control_all_fail_rank": all(
            (not trial.passed)
            and trial.fail_reason == "projected_rank_exceeds_low_rank_budget"
            for trial in trials
            if trial.condition == "misaligned_krylov_fail"
        ),
        "short_segment_control_all_fail_price": all(
            (not trial.passed)
            and trial.fail_reason == "projected_trace_not_below_dense_backward_budget"
            for trial in trials
            if trial.condition == "short_segment_pricing_fail"
        ),
        "all_conditions_identified": all(trial.identifying_correct for trial in trials),
    }
    decision = (
        "PASS_R501_S37_SUBSPACE_FORWARD_ELIGIBILITY_GATE"
        if all(required_passes.values())
        else "FAIL_R501_S37_SUBSPACE_FORWARD_ELIGIBILITY_GATE"
    )
    disposition = (
        "synthetic_positive_control_identified_not_training_speed_claim"
        if decision.startswith("PASS_")
        else "subspace_forward_eligibility_gate_failed_or_non_identifying"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "CPU synthetic subspace-projected forward eligibility gate. It "
            "checks low-rank dyadic eligibility projection residuals, effective "
            "Fisher rank, projected storage, and priced work. It is not heldout "
            "CE progress or GPU training-speed evidence."
        ),
        "finding": {
            "round": "R501",
            "aspect": "Speed",
            "id": "R501-S37",
            "title": "Subspace-Projected Forward Eligibility",
        },
        "thresholds": {
            "max_relative_residual": args.max_relative_residual,
            "max_priced_work_ratio": args.max_priced_work_ratio,
            "max_projected_rank": args.max_projected_rank,
            "required_projected_speedup": 1.0 / args.max_priced_work_ratio,
        },
        "shape": {
            "seed": args.seed,
            "seed_count": args.seed_count,
            "segment_length": args.segment_length,
            "short_segment_length": args.short_segment_length,
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
        "[R501_S37_SUBSPACE_FORWARD_ELIGIBILITY_GATE] "
        f"active=1 decision={decision} disposition={disposition} "
        f"positive_selected_rank_mean={positive['selected_rank_mean']:.6f} "
        f"positive_max_residual_max={positive['max_relative_residual_max']:.6e} "
        f"positive_captured_energy_mean={positive['captured_energy_fraction_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"positive_storage_ratio_mean={positive['projected_storage_ratio_mean']:.6f} "
        f"seed_count={args.seed_count} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
