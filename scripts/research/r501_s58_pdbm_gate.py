#!/usr/bin/env python3
"""R501 S58 phase-diagonalized backward masking gate.

This CPU diagnostic implements the R501 Speed Paradigm 10 finding:
"Phase-Diagonalized Backward Masking (PDBM)".

The candidate transforms a linear recurrent backward transition into an
orthogonal phase basis, keeps only block-diagonal phase flow, and drops
off-block cross-talk when the residual is below the ternary update noise
budget. It survives only when the masked backward cotangents preserve dense
ternary update decisions and the block-diagonal route is cheaper than dense
full-rank backward multiplication.

Controls exercise the declared risks: real off-block cross-talk is not
negligible, using the wrong phase basis destroys the block structure, and
over-wide blocks recover dense cost even when exactness holds.

This is a synthetic CPU diagnostic, not heldout CE progress, GPU training-speed
evidence, a Vulkan kernel benchmark, or production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s58_pdbm_gate.v1"
CONDITIONS = (
    "positive",
    "offdiag_crosstalk_fail",
    "wrong_basis_fail",
    "wide_block_price_fail",
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
    block_size: int
    block_count: int
    offdiag_energy_ratio: float
    residual_debt_ratio: float
    max_step_residual_ratio: float
    ternary_mismatch_fraction: float
    ternary_noise_budget_ratio: float
    descent_alignment_ratio: float
    block_work_ratio: float
    basis_transform_work_ratio: float
    verifier_work_ratio: float
    priced_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50158)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--max-offdiag-energy-ratio", type=float, default=0.020)
    parser.add_argument("--max-residual-debt-ratio", type=float, default=0.025)
    parser.add_argument("--max-ternary-mismatch-fraction", type=float, default=0.0)
    parser.add_argument("--max-ternary-budget-ratio", type=float, default=0.35)
    parser.add_argument("--min-descent-alignment-ratio", type=float, default=0.999)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.60)
    parser.add_argument("--ternary-margin", type=float, default=0.050)
    parser.add_argument("--basis-transform-work-ratio", type=float, default=0.13)
    parser.add_argument("--verifier-work-ratio", type=float, default=0.04)
    return parser.parse_args()


def condition_block_size(args: argparse.Namespace, condition: str) -> int:
    if condition == "wide_block_price_fail":
        return args.state_dim // 2
    return args.block_size


def random_orthogonal(rng: np.random.Generator, dim: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return q.astype(np.float64)


def block_mask(dim: int, block_size: int) -> np.ndarray:
    if dim % block_size != 0:
        raise ValueError("--state-dim must be divisible by block size")
    mask = np.zeros((dim, dim), dtype=bool)
    for start in range(0, dim, block_size):
        mask[start : start + block_size, start : start + block_size] = True
    return mask


def make_block_transition(
    rng: np.random.Generator,
    dim: int,
    block_size: int,
    condition: str,
) -> np.ndarray:
    basis_transition = np.zeros((dim, dim), dtype=np.float64)
    for start in range(0, dim, block_size):
        local = rng.normal(size=(block_size, block_size))
        q, _ = np.linalg.qr(local)
        eig = rng.uniform(0.25, 0.68, size=block_size)
        basis_transition[start : start + block_size, start : start + block_size] = (
            q @ np.diag(eig) @ q.T
        )

    offdiag_scale = 0.0007
    if condition == "offdiag_crosstalk_fail":
        offdiag_scale = 0.075
    offdiag = rng.normal(scale=offdiag_scale, size=(dim, dim))
    offdiag[block_mask(dim, block_size)] = 0.0
    transition = basis_transition + offdiag
    spectral_norm = float(np.linalg.norm(transition, ord=2))
    if spectral_norm > 0.88:
        transition *= 0.88 / spectral_norm
    return transition.astype(np.float64)


def make_sources(
    rng: np.random.Generator,
    steps: int,
    dim: int,
    margin: float,
) -> np.ndarray:
    source = rng.normal(scale=0.24, size=(steps + 1, dim))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(steps + 1, dim))
    source += signs * margin * 2.8
    return source.astype(np.float64)


def dense_backward(transition: np.ndarray, source: np.ndarray) -> np.ndarray:
    steps = source.shape[0] - 1
    lambdas = np.zeros_like(source, dtype=np.float64)
    lambdas[steps] = source[steps]
    for t in range(steps - 1, -1, -1):
        lambdas[t] = source[t] + transition.T @ lambdas[t + 1]
    return lambdas


def candidate_basis(
    rng: np.random.Generator,
    true_basis: np.ndarray,
    condition: str,
) -> np.ndarray:
    if condition != "wrong_basis_fail":
        return true_basis
    wrong_rotation = random_orthogonal(rng, true_basis.shape[0])
    return true_basis @ wrong_rotation


def masked_backward(
    transition_original: np.ndarray,
    source_original: np.ndarray,
    basis: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transition_basis = basis.T @ transition_original @ basis
    masked_transition = transition_basis * mask
    source_basis = source_original @ basis
    lambda_basis = dense_backward(masked_transition, source_basis)
    return lambda_basis @ basis.T, transition_basis


def ternary(values: np.ndarray, margin: float) -> np.ndarray:
    return np.where(values > margin, 1, np.where(values < -margin, -1, 0))


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    block_size = condition_block_size(args, condition)
    mask = block_mask(args.state_dim, block_size)
    true_basis = random_orthogonal(rng, args.state_dim)
    transition_basis = make_block_transition(rng, args.state_dim, block_size, condition)
    transition_original = true_basis @ transition_basis @ true_basis.T
    source_basis = make_sources(rng, args.steps, args.state_dim, args.ternary_margin)
    source_original = source_basis @ true_basis.T
    dense_lambdas = dense_backward(transition_original, source_original)
    q_candidate = candidate_basis(rng, true_basis, condition)
    masked_lambdas, transition_in_candidate_basis = masked_backward(
        transition_original,
        source_original,
        q_candidate,
        mask,
    )

    residual = masked_lambdas - dense_lambdas
    residual_debt = float(np.linalg.norm(residual)) / max(float(np.linalg.norm(dense_lambdas)), 1.0e-30)
    step_residual = np.linalg.norm(residual, axis=1) / np.maximum(
        np.linalg.norm(dense_lambdas, axis=1),
        1.0e-30,
    )
    offdiag_energy = float(np.linalg.norm(transition_in_candidate_basis * ~mask))
    total_energy = max(float(np.linalg.norm(transition_in_candidate_basis)), 1.0e-30)
    offdiag_energy_ratio = offdiag_energy / total_energy
    dense_ternary = ternary(dense_lambdas, args.ternary_margin)
    masked_ternary = ternary(masked_lambdas, args.ternary_margin)
    certified_margin = np.abs(dense_lambdas) >= (2.0 * args.ternary_margin)
    if np.any(certified_margin):
        ternary_mismatch_fraction = float(
            np.mean(dense_ternary[certified_margin] != masked_ternary[certified_margin])
        )
    else:
        ternary_mismatch_fraction = 0.0
    ternary_noise_budget = args.ternary_margin * math.sqrt(dense_lambdas.size)
    ternary_noise_budget_ratio = float(np.linalg.norm(residual)) / max(
        ternary_noise_budget,
        1.0e-30,
    )
    descent_alignment = float(np.sum(masked_lambdas * dense_lambdas)) / max(
        float(np.sum(dense_lambdas * dense_lambdas)),
        1.0e-30,
    )

    block_count = args.state_dim // block_size
    block_work_ratio = float(np.count_nonzero(mask) / mask.size)
    priced_work_ratio = (
        block_work_ratio + args.basis_transform_work_ratio + args.verifier_work_ratio
    )
    projected_speedup = 1.0 / priced_work_ratio if priced_work_ratio > 0.0 else math.inf

    failed: list[str] = []
    if not all(
        math.isfinite(value)
        for value in (
            offdiag_energy_ratio,
            residual_debt,
            ternary_mismatch_fraction,
            ternary_noise_budget_ratio,
            descent_alignment,
            priced_work_ratio,
        )
    ):
        failed.append("non_finite_metric")
    if offdiag_energy_ratio > args.max_offdiag_energy_ratio:
        failed.append("offdiag_crosstalk")
    if residual_debt > args.max_residual_debt_ratio:
        failed.append("residual_debt")
    if ternary_mismatch_fraction > args.max_ternary_mismatch_fraction:
        failed.append("ternary_mismatch")
    if ternary_noise_budget_ratio > args.max_ternary_budget_ratio:
        failed.append("ternary_budget")
    if descent_alignment < args.min_descent_alignment_ratio:
        failed.append("descent_alignment")
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
        steps=args.steps,
        state_dim=args.state_dim,
        block_size=block_size,
        block_count=block_count,
        offdiag_energy_ratio=float(offdiag_energy_ratio),
        residual_debt_ratio=float(residual_debt),
        max_step_residual_ratio=float(np.max(step_residual)),
        ternary_mismatch_fraction=float(ternary_mismatch_fraction),
        ternary_noise_budget_ratio=float(ternary_noise_budget_ratio),
        descent_alignment_ratio=float(descent_alignment),
        block_work_ratio=float(block_work_ratio),
        basis_transform_work_ratio=float(args.basis_transform_work_ratio),
        verifier_work_ratio=float(args.verifier_work_ratio),
        priced_work_ratio=float(priced_work_ratio),
        projected_speedup=float(projected_speedup),
    )


def aggregate(results: list[TrialResult]) -> dict[str, Any]:
    by_condition: dict[str, list[TrialResult]] = {
        condition: [result for result in results if result.condition == condition]
        for condition in CONDITIONS
    }
    numeric_fields = (
        "block_size",
        "block_count",
        "offdiag_energy_ratio",
        "residual_debt_ratio",
        "max_step_residual_ratio",
        "ternary_mismatch_fraction",
        "ternary_noise_budget_ratio",
        "descent_alignment_ratio",
        "block_work_ratio",
        "basis_transform_work_ratio",
        "verifier_work_ratio",
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
    offdiag = [result for result in results if result.condition == "offdiag_crosstalk_fail"]
    wrong_basis = [result for result in results if result.condition == "wrong_basis_fail"]
    wide_block = [result for result in results if result.condition == "wide_block_price_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_below_ternary_noise": (
            aggregates["positive"]["offdiag_energy_ratio_max"] <= 0.020
            and aggregates["positive"]["residual_debt_ratio_max"] <= 0.025
            and aggregates["positive"]["ternary_mismatch_fraction_max"] == 0.0
        ),
        "positive_condition_price": aggregates["positive"]["priced_work_ratio_mean"] <= 0.60,
        "offdiag_control_all_fail_crosstalk": all(
            (not result.passed)
            and result.fail_reason is not None
            and "offdiag_crosstalk" in result.fail_reason
            and "residual_debt" in result.fail_reason
            for result in offdiag
        ),
        "wrong_basis_control_all_fail_structure": all(
            (not result.passed)
            and result.fail_reason is not None
            and "offdiag_crosstalk" in result.fail_reason
            and "residual_debt" in result.fail_reason
            for result in wrong_basis
        ),
        "wide_block_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "price" in result.fail_reason
            for result in wide_block
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
    decision = "PASS_R501_S58_PDBM_GATE" if all(checks.values()) else "FAIL_R501_S58_PDBM_GATE"
    payload = {
        "schema": SCHEMA,
        "decision": decision,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "conditions": CONDITIONS,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU phase-diagonalized backward masking gate only; not "
            "heldout CE progress, not GPU training speed, not a Vulkan kernel "
            "benchmark, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S58_PDBM_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_block_work_ratio_mean={positive['block_work_ratio_mean']:.6f} "
        f"positive_offdiag_energy_ratio_max={positive['offdiag_energy_ratio_max']:.6e} "
        f"positive_residual_debt_ratio_max={positive['residual_debt_ratio_max']:.6e} "
        f"positive_ternary_mismatch_fraction_max={positive['ternary_mismatch_fraction_max']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
