#!/usr/bin/env python3
"""R501 S60 sparse orthogonal basis credit gate.

This CPU diagnostic implements the R501 Speed Paradigm 12 finding:
"Sparse Orthogonal Basis Credit (SOBC)".

For a dense rank-one credit update G = delta x^T, the candidate transforms both
factors into a certified orthogonal basis and updates only the sparse active
coefficient rectangle:

    G = U Sigma V^T

It survives only when the sparse-basis reconstruction is exactly equivalent to
the dense gradient, captures essentially all gradient energy, and the sparse
update plus basis/index overhead is cheaper than dense parameter updates.

Controls exercise the declared risks: dense excitation is not sparse in the
basis, the wrong basis destroys sparsity, and a too-large active rectangle
recovers dense cost even when exact.

This is a synthetic CPU diagnostic, not heldout CE progress, GPU training-speed
evidence, optimizer integration, or production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s60_sobc_gate.v1"
CONDITIONS = (
    "positive",
    "dense_excitation_fail",
    "wrong_basis_fail",
    "sparse_overhead_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    output_dim: int
    input_dim: int
    active_output_count: int
    active_input_count: int
    sparse_nnz_fraction: float
    reconstruction_error_ratio: float
    captured_energy_fraction: float
    missed_energy_fraction: float
    dense_equivalent_update_error: float
    sparse_update_work_ratio: float
    basis_overhead_ratio: float
    index_overhead_ratio: float
    priced_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50160)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--output-dim", type=int, default=32)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--active-output-count", type=int, default=4)
    parser.add_argument("--active-input-count", type=int, default=4)
    parser.add_argument("--max-reconstruction-error", type=float, default=1.0e-10)
    parser.add_argument("--min-captured-energy", type=float, default=0.999999)
    parser.add_argument("--max-sparse-nnz-fraction", type=float, default=0.10)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.75)
    parser.add_argument("--basis-overhead-ratio", type=float, default=0.08)
    parser.add_argument("--index-overhead-ratio", type=float, default=0.04)
    return parser.parse_args()


def random_orthogonal(rng: np.random.Generator, dim: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return q.astype(np.float64)


def active_counts(args: argparse.Namespace, condition: str) -> tuple[int, int]:
    if condition == "sparse_overhead_fail":
        return int(0.875 * args.output_dim), int(0.875 * args.input_dim)
    return args.active_output_count, args.active_input_count


def make_factor_basis_vector(
    rng: np.random.Generator,
    dim: int,
    active_count: int,
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    basis_vector = np.zeros(dim, dtype=np.float64)
    if condition == "dense_excitation_fail":
        basis_vector = rng.normal(scale=0.35, size=dim)
        support = np.argsort(np.abs(basis_vector))[-active_count:]
        return basis_vector, support
    support = np.sort(rng.choice(dim, size=active_count, replace=False))
    basis_vector[support] = rng.normal(scale=0.7, size=active_count)
    return basis_vector, support


def topk_mask(values: np.ndarray, k: int) -> np.ndarray:
    flat = np.abs(values).ravel()
    if k >= flat.size:
        return np.ones_like(values, dtype=bool)
    threshold_indices = np.argpartition(flat, -k)[-k:]
    mask = np.zeros(flat.size, dtype=bool)
    mask[threshold_indices] = True
    return mask.reshape(values.shape)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    active_out, active_in = active_counts(args, condition)
    u = random_orthogonal(rng, args.output_dim)
    v = random_orthogonal(rng, args.input_dim)
    delta_basis, _ = make_factor_basis_vector(rng, args.output_dim, active_out, condition)
    input_basis, _ = make_factor_basis_vector(rng, args.input_dim, active_in, condition)
    delta = u @ delta_basis
    x = v @ input_basis
    dense_gradient = np.outer(delta, x)

    candidate_u = u
    candidate_v = v
    if condition == "wrong_basis_fail":
        candidate_u = u @ random_orthogonal(rng, args.output_dim)
        candidate_v = v @ random_orthogonal(rng, args.input_dim)

    gradient_basis = candidate_u.T @ dense_gradient @ candidate_v
    target_nnz = active_out * active_in
    sparse_mask = topk_mask(gradient_basis, target_nnz)
    sparse_basis = np.where(sparse_mask, gradient_basis, 0.0)
    reconstructed = candidate_u @ sparse_basis @ candidate_v.T

    reconstruction_error = float(np.linalg.norm(reconstructed - dense_gradient)) / max(
        float(np.linalg.norm(dense_gradient)),
        1.0e-30,
    )
    total_energy = max(float(np.sum(gradient_basis * gradient_basis)), 1.0e-30)
    captured_energy = float(np.sum(sparse_basis * sparse_basis)) / total_energy
    sparse_nnz_fraction = float(np.count_nonzero(sparse_mask) / sparse_mask.size)
    dense_equivalent_update_error = float(
        np.linalg.norm((dense_gradient - reconstructed).ravel(), ord=np.inf)
    )
    sparse_update_work_ratio = sparse_nnz_fraction
    basis_overhead = args.basis_overhead_ratio
    index_overhead = args.index_overhead_ratio
    priced_work_ratio = sparse_update_work_ratio + basis_overhead + index_overhead
    projected_speedup = 1.0 / priced_work_ratio if priced_work_ratio > 0.0 else math.inf

    failed: list[str] = []
    if not all(
        math.isfinite(value)
        for value in (
            sparse_nnz_fraction,
            reconstruction_error,
            captured_energy,
            dense_equivalent_update_error,
            priced_work_ratio,
        )
    ):
        failed.append("non_finite_metric")
    if reconstruction_error > args.max_reconstruction_error:
        failed.append("reconstruction")
    if captured_energy < args.min_captured_energy:
        failed.append("captured_energy")
    if sparse_nnz_fraction > args.max_sparse_nnz_fraction:
        failed.append("sparse_fraction")
    if priced_work_ratio > args.max_priced_work_ratio:
        failed.append("price")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=(passed == expected_pass),
        fail_reason=";".join(failed) if failed else None,
        output_dim=args.output_dim,
        input_dim=args.input_dim,
        active_output_count=active_out,
        active_input_count=active_in,
        sparse_nnz_fraction=float(sparse_nnz_fraction),
        reconstruction_error_ratio=float(reconstruction_error),
        captured_energy_fraction=float(captured_energy),
        missed_energy_fraction=float(1.0 - captured_energy),
        dense_equivalent_update_error=float(dense_equivalent_update_error),
        sparse_update_work_ratio=float(sparse_update_work_ratio),
        basis_overhead_ratio=float(basis_overhead),
        index_overhead_ratio=float(index_overhead),
        priced_work_ratio=float(priced_work_ratio),
        projected_speedup=float(projected_speedup),
    )


def aggregate(results: list[TrialResult]) -> dict[str, Any]:
    by_condition: dict[str, list[TrialResult]] = {
        condition: [result for result in results if result.condition == condition]
        for condition in CONDITIONS
    }
    numeric_fields = (
        "active_output_count",
        "active_input_count",
        "sparse_nnz_fraction",
        "reconstruction_error_ratio",
        "captured_energy_fraction",
        "missed_energy_fraction",
        "dense_equivalent_update_error",
        "sparse_update_work_ratio",
        "basis_overhead_ratio",
        "index_overhead_ratio",
        "priced_work_ratio",
        "projected_speedup",
    )
    aggregates: dict[str, Any] = {}
    for condition, condition_results in by_condition.items():
        aggregates[condition] = {
            "count": len(condition_results),
            "passed_count": sum(result.passed for result in condition_results),
            "identifying_correct_count": sum(
                result.identifying_correct for result in condition_results
            ),
            "fail_reasons": sorted(
                {result.fail_reason for result in condition_results if result.fail_reason}
            ),
        }
        for field in numeric_fields:
            values = [float(getattr(result, field)) for result in condition_results]
            if not values:
                continue
            aggregates[condition][f"{field}_mean"] = float(np.mean(values))
            aggregates[condition][f"{field}_max"] = float(np.max(values))
            aggregates[condition][f"{field}_min"] = float(np.min(values))
    return aggregates


def all_finite(results: list[TrialResult]) -> bool:
    for result in results:
        for value in asdict(result).values():
            if isinstance(value, float) and not math.isfinite(value):
                return False
    return True


def required_checks(results: list[TrialResult], aggregates: dict[str, Any]) -> dict[str, bool]:
    positive = [result for result in results if result.condition == "positive"]
    dense = [result for result in results if result.condition == "dense_excitation_fail"]
    wrong = [result for result in results if result.condition == "wrong_basis_fail"]
    overhead = [result for result in results if result.condition == "sparse_overhead_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_sparse_basis": (
            aggregates["positive"]["reconstruction_error_ratio_max"] <= 1.0e-10
            and aggregates["positive"]["captured_energy_fraction_min"] >= 0.999999
        ),
        "positive_condition_sparse_and_priced": (
            aggregates["positive"]["sparse_nnz_fraction_mean"] <= 0.10
            and aggregates["positive"]["priced_work_ratio_mean"] <= 0.75
        ),
        "dense_excitation_control_all_fail_capture": all(
            (not result.passed)
            and result.fail_reason is not None
            and "reconstruction" in result.fail_reason
            and "captured_energy" in result.fail_reason
            for result in dense
        ),
        "wrong_basis_control_all_fail_capture": all(
            (not result.passed)
            and result.fail_reason is not None
            and "reconstruction" in result.fail_reason
            and "captured_energy" in result.fail_reason
            for result in wrong
        ),
        "sparse_overhead_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "sparse_fraction" in result.fail_reason
            and "price" in result.fail_reason
            for result in overhead
        ),
        "all_conditions_identified": all(result.identifying_correct for result in results),
    }


def main() -> int:
    args = parse_args()
    results: list[TrialResult] = []
    for condition in CONDITIONS:
        for offset in range(args.seed_count):
            results.append(run_trial(args, condition, args.seed + 1000 * offset))

    aggregates = aggregate(results)
    checks = required_checks(results, aggregates)
    decision = "PASS_R501_S60_SOBC_GATE" if all(checks.values()) else "FAIL_R501_S60_SOBC_GATE"
    payload = {
        "schema": SCHEMA,
        "decision": decision,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "conditions": CONDITIONS,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU sparse orthogonal basis credit gate only; not "
            "heldout CE progress, not GPU training speed, not optimizer "
            "integration, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S60_SOBC_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_sparse_nnz_fraction_mean={positive['sparse_nnz_fraction_mean']:.6f} "
        f"positive_reconstruction_error_max={positive['reconstruction_error_ratio_max']:.6e} "
        f"positive_captured_energy_min={positive['captured_energy_fraction_min']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
