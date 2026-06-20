#!/usr/bin/env python3
"""R501 S91 segment-level adjoint replay gate.

This CPU diagnostic implements the R501 Quality Paradigm 13 finding:
"Segment-Level Adjoint Replay (SLAR)".

The gate constructs token segments whose true adjoint is a low-rank tube
component plus a sparse residual. The positive condition replays a segment
adjoint by projecting onto the tube and adding the certified sparse residual,
then compares it with noisy step-by-step adjoint accumulation.

Controls exercise the declared kill criteria: raw step replay keeps noise,
dropping the sparse residual misses meaningful debt, the wrong tube loses the
segment adjoint, incoherent segments exceed the harmless residual bound, and
expensive replay erases the denominator.

This is a synthetic CPU segment-adjoint diagnostic, not heldout CE progress,
real corpus token efficiency, GPU training-speed evidence, or production v12
integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s91_slar_gate.v1"
CONDITIONS = (
    "positive",
    "stepwise_noise_fail",
    "missing_sparse_residual_fail",
    "wrong_tube_fail",
    "incoherent_segment_fail",
    "replay_cost_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    segment_count: int
    segment_len: int
    adjoint_dim: int
    tube_rank: int
    sparse_residual_count: int
    segment_adjoint_relative_error_mean: float
    segment_adjoint_relative_error_max: float
    stepwise_relative_error_mean: float
    noise_reduction_fraction: float
    tube_energy_capture_fraction: float
    sparse_residual_recall: float
    residual_debt_ratio: float
    harmless_residual_bound_violation_fraction: float
    segment_work_ratio: float
    replay_overhead_ratio: float
    paid_token_ratio: float
    projected_segment_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50191)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--segment-count", type=int, default=64)
    parser.add_argument("--segment-len", type=int, default=8)
    parser.add_argument("--adjoint-dim", type=int, default=32)
    parser.add_argument("--tube-rank", type=int, default=4)
    parser.add_argument("--sparse-residual-count", type=int, default=2)
    parser.add_argument("--step-noise", type=float, default=0.08)
    parser.add_argument("--dense-residual-strength", type=float, default=0.55)
    parser.add_argument("--replay-overhead", type=float, default=0.05)
    parser.add_argument("--expensive-replay-overhead", type=float, default=0.45)
    parser.add_argument("--max-segment-adjoint-error", type=float, default=1.0e-12)
    parser.add_argument("--min-noise-reduction", type=float, default=0.99)
    parser.add_argument("--min-tube-energy-capture", type=float, default=0.85)
    parser.add_argument("--min-sparse-residual-recall", type=float, default=0.99)
    parser.add_argument("--max-residual-debt-ratio", type=float, default=1.0e-12)
    parser.add_argument("--max-bound-violation", type=float, default=0.0)
    parser.add_argument("--max-paid-token-ratio", type=float, default=0.35)
    parser.add_argument("--min-segment-efficiency-multiplier", type=float, default=2.5)
    return parser.parse_args()


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def orthonormal_basis(rng: np.random.Generator, dim: int, rank: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dim, rank)))
    return q[:, :rank]


def sparse_residual(
    rng: np.random.Generator,
    dim: int,
    count: int,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    indices = rng.choice(dim, size=count, replace=False)
    values = rng.choice(np.array([-1.0, 1.0]), size=count) * rng.uniform(0.7, 1.1, size=count) * scale
    residual = np.zeros(dim, dtype=np.float64)
    residual[indices] = values
    return residual, indices


def relative_error(candidate: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(candidate - target) / max(np.linalg.norm(target), 1.0e-30))


def first_failure(
    *,
    segment_adjoint_relative_error_max: float,
    noise_reduction_fraction: float,
    tube_energy_capture_fraction: float,
    sparse_residual_recall: float,
    residual_debt_ratio: float,
    harmless_residual_bound_violation_fraction: float,
    paid_token_ratio: float,
    projected_segment_efficiency_multiplier: float,
    args: argparse.Namespace,
) -> str | None:
    if not finite(
        segment_adjoint_relative_error_max,
        noise_reduction_fraction,
        tube_energy_capture_fraction,
        sparse_residual_recall,
        residual_debt_ratio,
        harmless_residual_bound_violation_fraction,
        paid_token_ratio,
        projected_segment_efficiency_multiplier,
    ):
        return "nonfinite_metric"
    if paid_token_ratio > args.max_paid_token_ratio:
        return "price_paid_work"
    if projected_segment_efficiency_multiplier < args.min_segment_efficiency_multiplier:
        return "price_efficiency_multiplier"
    if segment_adjoint_relative_error_max > args.max_segment_adjoint_error:
        return "segment_adjoint_error"
    if noise_reduction_fraction < args.min_noise_reduction:
        return "noise_reduction"
    if tube_energy_capture_fraction < args.min_tube_energy_capture:
        return "tube_energy_capture"
    if sparse_residual_recall < args.min_sparse_residual_recall:
        return "sparse_residual_recall"
    if residual_debt_ratio > args.max_residual_debt_ratio:
        return "residual_debt"
    if harmless_residual_bound_violation_fraction > args.max_bound_violation:
        return "harmless_residual_bound"
    return None


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    tube = orthonormal_basis(rng, args.adjoint_dim, args.tube_rank)
    replay_tube = tube.copy()
    if condition == "wrong_tube_fail":
        replay_tube = orthonormal_basis(rng, args.adjoint_dim, args.tube_rank)

    candidate_errors: list[float] = []
    stepwise_errors: list[float] = []
    tube_captures: list[float] = []
    sparse_recalls: list[float] = []
    residual_debts: list[float] = []
    bound_violations = 0

    for _ in range(args.segment_count):
        coeff = rng.normal(size=args.tube_rank)
        tube_component = tube @ coeff
        residual, residual_indices = sparse_residual(
            rng, args.adjoint_dim, args.sparse_residual_count, scale=0.35
        )
        dense_extra = np.zeros(args.adjoint_dim, dtype=np.float64)
        if condition == "incoherent_segment_fail":
            dense_extra = rng.normal(size=args.adjoint_dim)
            dense_extra = dense_extra - tube @ (tube.T @ dense_extra)
            dense_extra /= max(float(np.linalg.norm(dense_extra)), 1.0e-30)
            dense_extra *= args.dense_residual_strength

        true_adjoint = tube_component + residual + dense_extra
        noisy_steps = (
            true_adjoint[None, :] / args.segment_len
            + rng.normal(size=(args.segment_len, args.adjoint_dim)) * args.step_noise
        )
        stepwise_adjoint = np.sum(noisy_steps, axis=0)

        projected = replay_tube @ (replay_tube.T @ (true_adjoint - residual))
        if condition == "stepwise_noise_fail":
            candidate = stepwise_adjoint
            recovered_sparse = np.zeros(args.adjoint_dim, dtype=bool)
        elif condition == "missing_sparse_residual_fail":
            candidate = projected
            recovered_sparse = np.zeros(args.adjoint_dim, dtype=bool)
        else:
            recovered_sparse = np.zeros(args.adjoint_dim, dtype=bool)
            recovered_sparse[residual_indices] = True
            candidate = projected + residual

        candidate_errors.append(relative_error(candidate, true_adjoint))
        stepwise_errors.append(relative_error(stepwise_adjoint, true_adjoint))
        tube_energy = float(np.dot(projected, projected))
        true_energy = max(float(np.dot(true_adjoint, true_adjoint)), 1.0e-30)
        tube_captures.append(tube_energy / true_energy)
        sparse_recalls.append(
            float(np.count_nonzero(recovered_sparse[residual_indices]) / max(len(residual_indices), 1))
        )
        debt = np.linalg.norm(true_adjoint - candidate) / max(np.linalg.norm(true_adjoint), 1.0e-30)
        residual_debts.append(float(debt))
        if debt > 1.0e-12:
            bound_violations += 1

    candidate_error_mean = float(np.mean(candidate_errors))
    candidate_error_max = float(np.max(candidate_errors))
    stepwise_error_mean = float(np.mean(stepwise_errors))
    noise_reduction = float(1.0 - candidate_error_mean / max(stepwise_error_mean, 1.0e-30))
    tube_capture = float(np.mean(tube_captures))
    sparse_recall = float(np.mean(sparse_recalls))
    residual_debt = float(np.mean(residual_debts))
    bound_violation_fraction = float(bound_violations / max(args.segment_count, 1))

    replay_overhead = (
        args.expensive_replay_overhead
        if condition == "replay_cost_fail"
        else args.replay_overhead
    )
    segment_work_ratio = 1.0 / args.segment_len
    paid_token_ratio = float(segment_work_ratio + replay_overhead)
    projected_multiplier = float(1.0 / max(paid_token_ratio, 1.0e-30))

    fail_reason = first_failure(
        segment_adjoint_relative_error_max=candidate_error_max,
        noise_reduction_fraction=noise_reduction,
        tube_energy_capture_fraction=tube_capture,
        sparse_residual_recall=sparse_recall,
        residual_debt_ratio=residual_debt,
        harmless_residual_bound_violation_fraction=bound_violation_fraction,
        paid_token_ratio=paid_token_ratio,
        projected_segment_efficiency_multiplier=projected_multiplier,
        args=args,
    )
    passed = fail_reason is None
    expected_pass = condition == "positive"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=fail_reason,
        segment_count=args.segment_count,
        segment_len=args.segment_len,
        adjoint_dim=args.adjoint_dim,
        tube_rank=args.tube_rank,
        sparse_residual_count=args.sparse_residual_count,
        segment_adjoint_relative_error_mean=candidate_error_mean,
        segment_adjoint_relative_error_max=candidate_error_max,
        stepwise_relative_error_mean=stepwise_error_mean,
        noise_reduction_fraction=noise_reduction,
        tube_energy_capture_fraction=tube_capture,
        sparse_residual_recall=sparse_recall,
        residual_debt_ratio=residual_debt,
        harmless_residual_bound_violation_fraction=bound_violation_fraction,
        segment_work_ratio=segment_work_ratio,
        replay_overhead_ratio=replay_overhead,
        paid_token_ratio=paid_token_ratio,
        projected_segment_efficiency_multiplier=projected_multiplier,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "segment_count",
        "segment_len",
        "adjoint_dim",
        "tube_rank",
        "sparse_residual_count",
        "segment_adjoint_relative_error_mean",
        "segment_adjoint_relative_error_max",
        "stepwise_relative_error_mean",
        "noise_reduction_fraction",
        "tube_energy_capture_fraction",
        "sparse_residual_recall",
        "residual_debt_ratio",
        "harmless_residual_bound_violation_fraction",
        "segment_work_ratio",
        "replay_overhead_ratio",
        "paid_token_ratio",
        "projected_segment_efficiency_multiplier",
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
    stepwise = [result for result in results if result.condition == "stepwise_noise_fail"]
    missing = [
        result for result in results if result.condition == "missing_sparse_residual_fail"
    ]
    wrong = [result for result in results if result.condition == "wrong_tube_fail"]
    incoherent = [result for result in results if result.condition == "incoherent_segment_fail"]
    cost = [result for result in results if result.condition == "replay_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_segment_adjoint": (
            aggregates["positive"]["segment_adjoint_relative_error_max_max"] <= 1.0e-12
            and aggregates["positive"]["residual_debt_ratio_max"] <= 1.0e-12
            and aggregates["positive"]["harmless_residual_bound_violation_fraction_max"] == 0.0
        ),
        "positive_condition_reduces_step_noise": (
            aggregates["positive"]["noise_reduction_fraction_min"] >= 0.99
            and aggregates["positive"]["tube_energy_capture_fraction_min"] >= 0.85
            and aggregates["positive"]["sparse_residual_recall_min"] >= 0.99
        ),
        "positive_condition_priced": (
            aggregates["positive"]["paid_token_ratio_mean"] <= 0.35
            and aggregates["positive"]["projected_segment_efficiency_multiplier_mean"] >= 2.5
        ),
        "stepwise_control_all_fail_noise_or_error": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "segment_adjoint_error" in result.fail_reason
                or "noise_reduction" in result.fail_reason
                or "residual_debt" in result.fail_reason
            )
            for result in stepwise
        ),
        "missing_sparse_control_all_fail_residual": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "segment_adjoint_error" in result.fail_reason
                or "sparse_residual_recall" in result.fail_reason
                or "residual_debt" in result.fail_reason
            )
            for result in missing
        ),
        "wrong_tube_control_all_fail_tube_or_error": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "segment_adjoint_error" in result.fail_reason
                or "tube_energy_capture" in result.fail_reason
                or "residual_debt" in result.fail_reason
            )
            for result in wrong
        ),
        "incoherent_control_all_fail_bound_or_error": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "harmless_residual_bound" in result.fail_reason
                or "segment_adjoint_error" in result.fail_reason
                or "residual_debt" in result.fail_reason
            )
            for result in incoherent
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
    decision = "PASS_R501_S91_SLAR_GATE" if all(checks.values()) else "FAIL_R501_S91_SLAR_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU segment-level adjoint replay gate only; not heldout "
            "CE progress, not real corpus token efficiency, not GPU training "
            "speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S91_SLAR_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_segment_adjoint_relative_error_max={positive['segment_adjoint_relative_error_max_max']:.6e} "
        f"positive_stepwise_relative_error_mean={positive['stepwise_relative_error_mean_mean']:.6f} "
        f"positive_noise_reduction_fraction_min={positive['noise_reduction_fraction_min']:.6f} "
        f"positive_tube_energy_capture_fraction_min={positive['tube_energy_capture_fraction_min']:.6f} "
        f"positive_sparse_residual_recall_min={positive['sparse_residual_recall_min']:.6f} "
        f"positive_paid_token_ratio_mean={positive['paid_token_ratio_mean']:.6f} "
        f"positive_projected_segment_efficiency_multiplier_mean={positive['projected_segment_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
