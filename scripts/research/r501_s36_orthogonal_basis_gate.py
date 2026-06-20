#!/usr/bin/env python3
"""R501 S36 orthogonal-basis recurrent fast transform gate.

This CPU diagnostic implements the R501 Speed Math 3 finding:
"Orthogonal-Basis Recurrent Fast Transform (Frequency-Domain BPTT)".

For a static transition A = V Lambda V^-1, recurrent cotangent propagation can
be performed in the basis coordinates with per-step scalar multiplies:

    y[t] = g[t] V + y[t + 1] * diag(Lambda)
    lambda[t] = y[t] V^-1

The gate checks whether this basis route is an exact quotient of the dense
recurrent adjoint and whether the priced work model
O(T*d) + 2*O(d^2) is below the dense O(T*d^2) route. It also includes controls
for data-dependent bases, ill-conditioned non-orthogonal bases, and segments
too short for boundary transforms to amortize.

The result is a synthetic CPU gate. It is not a measured GPU training-speed
claim and does not promote a production v12 training path.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s36_orthogonal_basis_gate.v1"
CONDITIONS = (
    "positive",
    "moving_basis_fail",
    "nonorthogonal_fail",
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
    basis_condition_number: float
    max_transition_drift: float
    relative_adjoint_error: float
    max_abs_adjoint_error: float
    priced_work_ratio: float
    projected_speedup: float
    dense_backward_seconds: float
    basis_seconds: float
    measured_cpu_speedup: float
    price_components: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50136)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--segment-length", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--short-segment-length", type=int, default=16)
    parser.add_argument("--max-relative-adjoint-error", type=float, default=5.0e-6)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.12)
    parser.add_argument("--max-transition-drift", type=float, default=1.0e-10)
    parser.add_argument("--max-basis-condition-number", type=float, default=50.0)
    parser.add_argument("--boundary-transform-count", type=float, default=2.0)
    parser.add_argument("--audit-overhead-ratio", type=float, default=0.035)
    parser.add_argument("--timing-repeats", type=int, default=5)
    return parser.parse_args()


def orthogonal_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def smooth_basis_forcing(
    rng: np.random.Generator,
    segment_length: int,
    state_dim: int,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, segment_length, dtype=np.float64)
    columns = [
        np.ones_like(t),
        t,
        t * t,
        np.sin(2.0 * math.pi * t),
        np.cos(2.0 * math.pi * t),
        np.sin(4.0 * math.pi * t + 0.2),
        np.cos(6.0 * math.pi * t - 0.4),
    ]
    x = np.stack(columns, axis=1)
    weights = rng.normal(scale=0.18, size=(x.shape[1], state_dim))
    forcing = x @ weights
    forcing += 0.006 * rng.normal(size=forcing.shape)
    return forcing.astype(np.float64)


def dense_static_adjoint(g: np.ndarray, a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(g, dtype=np.float64)
    carry = np.zeros(g.shape[1], dtype=np.float64)
    for t in range(g.shape[0] - 1, -1, -1):
        carry = g[t] + carry @ a
        out[t] = carry
    return out


def dense_dynamic_adjoint(g: np.ndarray, transitions: list[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(g, dtype=np.float64)
    carry = np.zeros(g.shape[1], dtype=np.float64)
    for t in range(g.shape[0] - 1, -1, -1):
        carry = g[t] + carry @ transitions[t]
        out[t] = carry
    return out


def basis_static_adjoint(
    g: np.ndarray,
    basis: np.ndarray,
    inverse_basis: np.ndarray,
    eigenvalues: np.ndarray,
) -> np.ndarray:
    basis_dtype = np.float32
    g_basis = (g.astype(basis_dtype) @ basis.astype(basis_dtype)).astype(basis_dtype)
    eig = eigenvalues.astype(basis_dtype)
    out_basis = np.zeros_like(g_basis, dtype=basis_dtype)
    carry = np.zeros(g.shape[1], dtype=basis_dtype)
    for t in range(g.shape[0] - 1, -1, -1):
        carry = g_basis[t] + carry * eig
        out_basis[t] = carry
    out = out_basis @ inverse_basis.astype(basis_dtype)
    return out.astype(np.float64)


def median_seconds(func: Any, repeats: int) -> float:
    samples: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        func()
        samples.append(time.perf_counter() - start)
    return float(np.median(np.array(samples, dtype=np.float64)))


def relative_norm(x: np.ndarray, denom: np.ndarray) -> float:
    numerator = float(np.linalg.norm(x.ravel()))
    denominator = float(np.linalg.norm(denom.ravel()))
    if denominator <= 0.0:
        return math.inf
    return numerator / denominator


def ill_conditioned_basis(rng: np.random.Generator, n: int) -> np.ndarray:
    left = orthogonal_matrix(rng, n)
    right = orthogonal_matrix(rng, n)
    scales = np.geomspace(1.0, 1.0e-7, n)
    return left @ np.diag(scales) @ right.T


def transition_drift(transitions: list[np.ndarray], reference: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(reference)), 1.0e-30)
    return float(max(np.linalg.norm(a - reference) / denom for a in transitions))


def price_route(
    args: argparse.Namespace,
    condition: str,
    segment_length: int,
    state_dim: int,
    condition_number: float,
) -> dict[str, float]:
    dense_work = float(segment_length * state_dim * state_dim)
    scalar_recurrence = float(segment_length * state_dim) / dense_work
    boundary_transforms = (
        float(args.boundary_transform_count * state_dim * state_dim) / dense_work
    )
    dynamic_basis_refresh = 0.0
    if condition == "moving_basis_fail":
        dynamic_basis_refresh = float(2.0 * segment_length * state_dim * state_dim) / dense_work
    condition_settlement = 0.0
    if condition_number > args.max_basis_condition_number:
        condition_settlement = min(1.0, math.log10(condition_number) / 8.0) * 0.25
    return {
        "basis_scalar_recurrence": scalar_recurrence,
        "boundary_transforms": boundary_transforms,
        "audit_overhead": float(args.audit_overhead_ratio),
        "dynamic_basis_refresh": dynamic_basis_refresh,
        "condition_settlement": condition_settlement,
    }


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    segment_length = (
        args.short_segment_length
        if condition == "short_segment_pricing_fail"
        else args.segment_length
    )
    state_dim = args.state_dim
    eigenvalues = np.linspace(0.04, 0.72, state_dim, dtype=np.float64)

    if condition == "nonorthogonal_fail":
        basis = ill_conditioned_basis(rng, state_dim)
        inverse_basis = np.linalg.inv(basis)
        a0 = basis @ np.diag(eigenvalues) @ inverse_basis
    else:
        basis = orthogonal_matrix(rng, state_dim)
        inverse_basis = basis.T
        a0 = basis @ np.diag(eigenvalues) @ basis.T

    g_basis = smooth_basis_forcing(rng, segment_length, state_dim)
    g_dense = g_basis @ inverse_basis

    if condition == "moving_basis_fail":
        transitions: list[np.ndarray] = []
        for t in range(segment_length):
            angle = (t + 1) / float(segment_length)
            drift_basis = orthogonal_matrix(rng, state_dim)
            mixed_basis = (1.0 - 0.10 * angle) * basis + (0.10 * angle) * drift_basis
            q, _ = np.linalg.qr(mixed_basis)
            transitions.append(q @ np.diag(eigenvalues) @ q.T)
        dense = dense_dynamic_adjoint(g_dense, transitions)
        drift = transition_drift(transitions, a0)
    else:
        dense = dense_static_adjoint(g_dense, a0)
        drift = 0.0

    basis_out = basis_static_adjoint(g_dense, basis, inverse_basis, eigenvalues)
    error = basis_out - dense
    relative_error = relative_norm(error, dense)
    max_abs_error = float(np.max(np.abs(error)))
    condition_number = float(np.linalg.cond(basis))

    price_components = price_route(
        args=args,
        condition=condition,
        segment_length=segment_length,
        state_dim=state_dim,
        condition_number=condition_number,
    )
    priced_work_ratio = float(sum(price_components.values()))
    projected_speedup = 1.0 / priced_work_ratio if priced_work_ratio > 0.0 else math.inf

    dense_seconds = median_seconds(
        lambda: (
            dense_dynamic_adjoint(g_dense, transitions)
            if condition == "moving_basis_fail"
            else dense_static_adjoint(g_dense, a0)
        ),
        args.timing_repeats,
    )
    basis_seconds = median_seconds(
        lambda: basis_static_adjoint(g_dense, basis, inverse_basis, eigenvalues),
        args.timing_repeats,
    )
    measured_cpu_speedup = (
        dense_seconds / basis_seconds if basis_seconds > 0.0 else math.inf
    )

    static_pass = drift <= args.max_transition_drift
    condition_pass = condition_number <= args.max_basis_condition_number
    exact_pass = relative_error <= args.max_relative_adjoint_error
    price_pass = priced_work_ratio <= args.max_priced_work_ratio
    passed = bool(static_pass and condition_pass and exact_pass and price_pass)
    expected_pass = condition == "positive"
    identifying_correct = passed == expected_pass

    fail_reason = None
    if not passed:
        if not static_pass:
            fail_reason = "basis_changes_reintroduce_per_timestep_dense_work"
        elif not condition_pass:
            fail_reason = "basis_transform_condition_number_breaks_exact_quotient"
        elif not exact_pass:
            fail_reason = "basis_adjoint_not_exact_dense_quotient"
        elif not price_pass:
            fail_reason = "priced_route_not_below_dense_backward_budget"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=identifying_correct,
        fail_reason=fail_reason,
        segment_length=segment_length,
        state_dim=state_dim,
        basis_condition_number=condition_number,
        max_transition_drift=drift,
        relative_adjoint_error=float(relative_error),
        max_abs_adjoint_error=max_abs_error,
        priced_work_ratio=priced_work_ratio,
        projected_speedup=float(projected_speedup),
        dense_backward_seconds=dense_seconds,
        basis_seconds=basis_seconds,
        measured_cpu_speedup=float(measured_cpu_speedup),
        price_components=price_components,
    )


def finite_trial(trial: TrialResult) -> bool:
    values = (
        trial.basis_condition_number,
        trial.max_transition_drift,
        trial.relative_adjoint_error,
        trial.max_abs_adjoint_error,
        trial.priced_work_ratio,
        trial.projected_speedup,
        trial.dense_backward_seconds,
        trial.basis_seconds,
        trial.measured_cpu_speedup,
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
            "basis_condition_number_mean": float(
                np.mean([row.basis_condition_number for row in rows])
            ),
            "max_transition_drift_mean": float(
                np.mean([row.max_transition_drift for row in rows])
            ),
            "relative_adjoint_error_mean": float(
                np.mean([row.relative_adjoint_error for row in rows])
            ),
            "relative_adjoint_error_max": float(
                np.max([row.relative_adjoint_error for row in rows])
            ),
            "max_abs_adjoint_error_max": float(
                np.max([row.max_abs_adjoint_error for row in rows])
            ),
            "priced_work_ratio_mean": float(
                np.mean([row.priced_work_ratio for row in rows])
            ),
            "projected_speedup_mean": float(
                np.mean([row.projected_speedup for row in rows])
            ),
            "measured_cpu_speedup_mean": float(
                np.mean([row.measured_cpu_speedup for row in rows])
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
        "positive_condition_exact_dense_quotient": all(
            trial.relative_adjoint_error <= args.max_relative_adjoint_error
            for trial in trials
            if trial.condition == "positive"
        ),
        "positive_condition_priced_below_budget": all(
            trial.priced_work_ratio <= args.max_priced_work_ratio
            for trial in trials
            if trial.condition == "positive"
        ),
        "moving_basis_control_all_fail": all(
            (not trial.passed)
            and trial.fail_reason == "basis_changes_reintroduce_per_timestep_dense_work"
            for trial in trials
            if trial.condition == "moving_basis_fail"
        ),
        "nonorthogonal_control_all_fail": all(
            (not trial.passed)
            and trial.fail_reason
            == "basis_transform_condition_number_breaks_exact_quotient"
            for trial in trials
            if trial.condition == "nonorthogonal_fail"
        ),
        "short_segment_control_all_fail_price": all(
            (not trial.passed)
            and trial.fail_reason == "priced_route_not_below_dense_backward_budget"
            for trial in trials
            if trial.condition == "short_segment_pricing_fail"
        ),
        "all_conditions_identified": all(trial.identifying_correct for trial in trials),
    }
    decision = (
        "PASS_R501_S36_ORTHOGONAL_BASIS_GATE"
        if all(required_passes.values())
        else "FAIL_R501_S36_ORTHOGONAL_BASIS_GATE"
    )
    disposition = (
        "synthetic_positive_control_identified_not_training_speed_claim"
        if decision.startswith("PASS_")
        else "orthogonal_basis_gate_failed_or_non_identifying"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "CPU synthetic static-basis recurrent adjoint gate. It checks exact "
            "dense/basis quotient error, static-basis validity, transform "
            "conditioning, and priced O(T*d)+2O(d^2) work. It is not heldout CE "
            "progress or GPU training-speed evidence."
        ),
        "finding": {
            "round": "R501",
            "aspect": "Speed",
            "id": "R501-S36",
            "title": "Orthogonal-Basis Recurrent Fast Transform (Frequency-Domain BPTT)",
        },
        "thresholds": {
            "max_relative_adjoint_error": args.max_relative_adjoint_error,
            "max_priced_work_ratio": args.max_priced_work_ratio,
            "max_transition_drift": args.max_transition_drift,
            "max_basis_condition_number": args.max_basis_condition_number,
            "required_projected_speedup": 1.0 / args.max_priced_work_ratio,
        },
        "shape": {
            "seed": args.seed,
            "seed_count": args.seed_count,
            "segment_length": args.segment_length,
            "short_segment_length": args.short_segment_length,
            "state_dim": args.state_dim,
        },
        "price_defaults": {
            "boundary_transform_count": args.boundary_transform_count,
            "audit_overhead_ratio": args.audit_overhead_ratio,
        },
        "required_passes": required_passes,
        "aggregate": aggregate(trials),
        "trials": [asdict(trial) for trial in trials],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    positive = result["aggregate"]["positive"]
    print(
        "[R501_S36_ORTHOGONAL_BASIS_GATE] "
        f"active=1 decision={decision} disposition={disposition} "
        f"positive_relative_adjoint_error_mean={positive['relative_adjoint_error_mean']:.6e} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"positive_measured_cpu_speedup_mean={positive['measured_cpu_speedup_mean']:.6f} "
        f"positive_basis_condition_number_mean={positive['basis_condition_number_mean']:.6f} "
        f"seed_count={args.seed_count} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
