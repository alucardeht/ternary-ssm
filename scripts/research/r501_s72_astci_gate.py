#!/usr/bin/env python3
"""R501 S72 adjoint-state transition curvature integrator gate.

This CPU diagnostic implements the R501 Quality Math 9 finding:
"Adjoint-State Transition Curvature Integrator (ASTCI)".

The gate builds exact dense cotangent trajectories and a forward-owned
low-rank adjoint basis B_theta(q_t). A positive case must reconstruct dense
cotangents with bounded residual on heldout timesteps, preserve a proxy
objective, and stay under the memory-priced denominator.

Controls exercise the declared risks: the basis may miss a true cotangent
component, a long-context basis drift may make residuals blow up, and storing a
nearly dense basis can erase the intended memory denominator.

This is a synthetic CPU cotangent-basis reconstruction diagnostic, not heldout
CE progress, real corpus token efficiency, GPU training-speed evidence, or
production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s72_astci_gate.v1"
CONDITIONS = (
    "positive",
    "basis_miss_fail",
    "long_context_blowup_fail",
    "memory_cost_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    seq_len: int
    state_dim: int
    basis_rank: int
    residual_fraction_mean: float
    residual_fraction_max: float
    heldout_residual_fraction_mean: float
    heldout_residual_fraction_max: float
    objective_relative_error_mean: float
    objective_relative_error_max: float
    memory_ratio: float
    priced_work_ratio: float
    projected_token_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50172)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=24)
    parser.add_argument("--basis-rank", type=int, default=4)
    parser.add_argument("--max-heldout-residual", type=float, default=0.10)
    parser.add_argument("--max-objective-error", type=float, default=0.10)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.70)
    parser.add_argument("--basis-overhead", type=float, default=0.16)
    return parser.parse_args()


def orthonormal(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    matrix = rng.normal(size=(rows, cols))
    q, _r = np.linalg.qr(matrix)
    return q[:, :cols]


def orthogonal_component(
    rng: np.random.Generator,
    basis: np.ndarray,
    scale: float,
) -> np.ndarray:
    vector = rng.normal(size=basis.shape[0])
    vector = vector - basis @ (basis.T @ vector)
    vector /= max(float(np.linalg.norm(vector)), 1.0e-30)
    return scale * vector


def make_trajectory(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    rank = args.basis_rank
    if condition == "memory_cost_fail":
        rank = max(rank, int(math.ceil(0.75 * args.state_dim)))
    base_basis = orthonormal(rng, args.state_dim, rank)
    drift_basis = orthonormal(rng, args.state_dim, rank)
    coeffs = rng.normal(size=(args.seq_len, rank))
    exact = np.empty((args.seq_len, args.state_dim), dtype=np.float64)
    reconstructed = np.empty_like(exact)
    probes = rng.normal(size=(args.seq_len, args.state_dim))

    for t in range(args.seq_len):
        phase = t / max(args.seq_len - 1, 1)
        if condition == "long_context_blowup_fail":
            true_mix = min(0.55, 0.08 + 0.55 * phase * phase)
            true_basis = (1.0 - true_mix) * base_basis + true_mix * drift_basis
            true_basis, _r = np.linalg.qr(true_basis)
            used_basis = base_basis
        else:
            true_basis = base_basis
            used_basis = base_basis

        dense = true_basis @ coeffs[t]
        if condition == "basis_miss_fail":
            dense = dense + orthogonal_component(rng, true_basis, scale=0.45 * max(float(np.linalg.norm(dense)), 1.0e-30))
        else:
            dense = dense + orthogonal_component(rng, true_basis, scale=0.025 * max(float(np.linalg.norm(dense)), 1.0e-30))

        exact[t] = dense
        reconstructed[t] = used_basis @ coeffs[t]

    return exact, reconstructed, probes, rank


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    exact, reconstructed, probes, rank = make_trajectory(rng, args, condition)
    residual = exact - reconstructed
    exact_norm = np.linalg.norm(exact, axis=1)
    residual_norm = np.linalg.norm(residual, axis=1)
    residual_fraction = residual_norm / np.maximum(exact_norm, 1.0e-30)
    heldout_start = int(round(0.75 * args.seq_len))
    heldout = residual_fraction[heldout_start:]

    exact_objective = np.einsum("td,td->t", exact, probes)
    recon_objective = np.einsum("td,td->t", reconstructed, probes)
    probe_norm = np.linalg.norm(probes, axis=1)
    objective_denominator = np.maximum(exact_norm * probe_norm, 1.0e-30)
    objective_error = np.abs(exact_objective - recon_objective) / objective_denominator
    memory_ratio = rank / args.state_dim + args.basis_overhead
    priced_work_ratio = memory_ratio
    projected_multiplier = 1.0 / priced_work_ratio

    failed: list[str] = []
    if not finite(
        float(np.mean(residual_fraction)),
        float(np.max(residual_fraction)),
        float(np.mean(heldout)),
        float(np.max(heldout)),
        float(np.mean(objective_error)),
        float(np.max(objective_error)),
        memory_ratio,
        priced_work_ratio,
        projected_multiplier,
    ):
        failed.append("non_finite_metric")
    if float(np.max(heldout)) > args.max_heldout_residual:
        failed.append("heldout_residual")
    if float(np.max(objective_error)) > args.max_objective_error:
        failed.append("objective_equivalence")
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
        seq_len=args.seq_len,
        state_dim=args.state_dim,
        basis_rank=rank,
        residual_fraction_mean=float(np.mean(residual_fraction)),
        residual_fraction_max=float(np.max(residual_fraction)),
        heldout_residual_fraction_mean=float(np.mean(heldout)),
        heldout_residual_fraction_max=float(np.max(heldout)),
        objective_relative_error_mean=float(np.mean(objective_error)),
        objective_relative_error_max=float(np.max(objective_error)),
        memory_ratio=float(memory_ratio),
        priced_work_ratio=float(priced_work_ratio),
        projected_token_efficiency_multiplier=float(projected_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "seq_len",
        "state_dim",
        "basis_rank",
        "residual_fraction_mean",
        "residual_fraction_max",
        "heldout_residual_fraction_mean",
        "heldout_residual_fraction_max",
        "objective_relative_error_mean",
        "objective_relative_error_max",
        "memory_ratio",
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
    miss = [result for result in results if result.condition == "basis_miss_fail"]
    blowup = [result for result in results if result.condition == "long_context_blowup_fail"]
    cost = [result for result in results if result.condition == "memory_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_residual_bounded": (
            aggregates["positive"]["heldout_residual_fraction_max_max"] <= 0.10
            and aggregates["positive"]["residual_fraction_max_max"] <= 0.10
        ),
        "positive_condition_objective_equivalent": (
            aggregates["positive"]["objective_relative_error_max_max"] <= 0.10
        ),
        "positive_condition_priced": (
            aggregates["positive"]["priced_work_ratio_mean"] <= 0.70
        ),
        "basis_miss_control_all_fail_residual": all(
            (not result.passed)
            and result.fail_reason is not None
            and "heldout_residual" in result.fail_reason
            for result in miss
        ),
        "long_context_control_all_fail_residual": all(
            (not result.passed)
            and result.fail_reason is not None
            and "heldout_residual" in result.fail_reason
            for result in blowup
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
        "PASS_R501_S72_ASTCI_GATE"
        if all(checks.values())
        else "FAIL_R501_S72_ASTCI_GATE"
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
            "Synthetic CPU adjoint-state transition curvature integrator gate "
            "only; not heldout CE progress, not real corpus token efficiency, "
            "not GPU training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S72_ASTCI_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_residual_fraction_max={positive['residual_fraction_max_max']:.6f} "
        f"positive_heldout_residual_fraction_max={positive['heldout_residual_fraction_max_max']:.6f} "
        f"positive_objective_relative_error_max={positive['objective_relative_error_max_max']:.6f} "
        f"positive_basis_rank_mean={positive['basis_rank_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_token_efficiency_multiplier_mean={positive['projected_token_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
