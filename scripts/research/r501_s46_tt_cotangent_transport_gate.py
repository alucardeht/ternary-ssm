#!/usr/bin/env python3
"""R501 S46 tensor-train cotangent transport gate.

This CPU diagnostic implements the R501 Speed Math 13 finding:
"Tensor-Train Compressed State Cotangent Transport".

The gate reshapes a cotangent vector into a small tensor, compresses it with a
Tensor-Train (TT) decomposition, applies a separable adjoint operator directly
to the TT cores, reconstructs the transported cotangent, and compares it with a
dense tensor-mode reference. The finding survives only when compression and
transport error stay below epsilon, the required TT rank stays far below the
sqrt(d) kill regime, and the priced TT route is at most half the dense
transition cost.

Controls exercise the declared kills: full-rank dense cotangents require rank
near sqrt(d), entangled operators are not represented by the separable TT-core
transport, and too-small compression rank violates the invariant even though it
looks cheap.

This is a synthetic CPU diagnostic, not heldout CE progress, GPU training-speed
evidence, or production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s46_tt_cotangent_transport_gate.v1"
CONDITIONS = (
    "positive",
    "rank_growth_fail",
    "entangled_operator_fail",
    "compression_budget_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    order: int
    core_dim: int
    dense_dim: int
    rank_cap: int
    tt_rank_used: int
    required_rank_for_epsilon: int
    sqrt_dense_dim: float
    rank_growth_ratio: float
    initial_compression_error: float
    transport_relative_error: float
    priced_work_ratio: float
    transport_work_ratio: float
    compression_work_ratio: float
    metadata_work_ratio: float
    projected_speedup: float
    dense_work: float
    tt_transport_work: float
    compression_work: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50146)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--core-dim", type=int, default=8)
    parser.add_argument("--positive-rank", type=int, default=3)
    parser.add_argument("--epsilon-compress", type=float, default=1.0e-9)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.50)
    parser.add_argument("--max-rank-growth-ratio", type=float, default=0.25)
    parser.add_argument("--metadata-work-ratio", type=float, default=0.06)
    parser.add_argument("--entangle-strength", type=float, default=0.22)
    return parser.parse_args()


def prod(values: tuple[int, ...]) -> int:
    return int(reduce(mul, values, 1))


def relative_error(candidate: np.ndarray, target: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(target.reshape(-1))), 1.0e-30)
    numerator = float(np.linalg.norm((candidate - target).reshape(-1)))
    return numerator / denominator


def tt_to_dense(cores: list[np.ndarray]) -> np.ndarray:
    result = cores[0][0]
    for core in cores[1:]:
        result = np.tensordot(result, core, axes=([-1], [0]))
    return np.squeeze(result, axis=-1)


def tt_max_rank(cores: list[np.ndarray]) -> int:
    ranks = [core.shape[0] for core in cores] + [cores[-1].shape[-1]]
    return int(max(ranks))


def tt_svd(tensor: np.ndarray, max_rank: int) -> list[np.ndarray]:
    dims = tuple(int(dim) for dim in tensor.shape)
    if max_rank < 1:
        raise ValueError("max_rank must be at least 1")
    cores: list[np.ndarray] = []
    unfolding = tensor.astype(np.float64, copy=True)
    prev_rank = 1
    for axis, dim in enumerate(dims[:-1]):
        unfolding = unfolding.reshape(prev_rank * dim, -1)
        u, s, vt = np.linalg.svd(unfolding, full_matrices=False)
        rank = min(max_rank, s.shape[0])
        core = u[:, :rank].reshape(prev_rank, dim, rank)
        cores.append(core)
        unfolding = (s[:rank, None] * vt[:rank, :])
        prev_rank = rank
    cores.append(unfolding.reshape(prev_rank, dims[-1], 1))
    return cores


def minimum_rank_for_error(
    tensor: np.ndarray,
    epsilon: float,
    max_rank: int,
) -> tuple[int, float]:
    last_error = math.inf
    for rank in range(1, max_rank + 1):
        cores = tt_svd(tensor, rank)
        error = relative_error(tt_to_dense(cores), tensor)
        last_error = error
        if error <= epsilon:
            return rank, error
    return max_rank, last_error


def make_low_rank_tensor(
    rng: np.random.Generator,
    dims: tuple[int, ...],
    rank: int,
) -> np.ndarray:
    cores: list[np.ndarray] = []
    for axis, dim in enumerate(dims):
        left = 1 if axis == 0 else rank
        right = 1 if axis == len(dims) - 1 else rank
        scale = 1.0 / math.sqrt(float(dim * max(left, right)))
        cores.append(rng.normal(scale=scale, size=(left, dim, right)))
    tensor = tt_to_dense(cores)
    norm = max(float(np.linalg.norm(tensor.reshape(-1))), 1.0e-30)
    return tensor / norm


def make_mode_matrices(
    rng: np.random.Generator,
    order: int,
    core_dim: int,
) -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    singular_values = np.linspace(0.72, 0.96, core_dim, dtype=np.float64)
    for _ in range(order):
        q, _ = np.linalg.qr(rng.normal(size=(core_dim, core_dim)))
        matrices.append(q @ np.diag(singular_values) @ q.T)
    return matrices


def mode_multiply(tensor: np.ndarray, matrix: np.ndarray, axis: int) -> np.ndarray:
    moved = np.moveaxis(tensor, axis, 0)
    multiplied = np.tensordot(matrix, moved, axes=(1, 0))
    return np.moveaxis(multiplied, 0, axis)


def apply_separable_dense(tensor: np.ndarray, matrices: list[np.ndarray]) -> np.ndarray:
    out = tensor
    for axis, matrix in enumerate(matrices):
        out = mode_multiply(out, matrix, axis)
    return out


def apply_separable_tt(cores: list[np.ndarray], matrices: list[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for core, matrix in zip(cores, matrices):
        out.append(np.einsum("ij,rjs->ris", matrix, core, optimize=True))
    return out


def apply_entangled_dense(
    tensor: np.ndarray,
    matrices: list[np.ndarray],
    strength: float,
) -> np.ndarray:
    base = apply_separable_dense(tensor, matrices)
    swapped = np.swapaxes(base, 0, 1)
    return base + strength * (swapped - base)


def compression_work(dims: tuple[int, ...], max_rank: int) -> float:
    work = 0.0
    prev_rank = 1
    for axis, dim in enumerate(dims[:-1]):
        rows = prev_rank * dim
        cols = prod(dims[axis + 1 :])
        rank = min(max_rank, rows, cols)
        work += float(rows * cols * rank)
        prev_rank = rank
    return work


def tt_transport_work(order: int, core_dim: int, rank: int) -> float:
    return float(order * (core_dim**2) * (rank**3))


def make_condition_tensor(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
    dims: tuple[int, ...],
) -> tuple[np.ndarray, int]:
    if condition in {"positive", "entangled_operator_fail"}:
        return make_low_rank_tensor(rng, dims, args.positive_rank), args.positive_rank
    if condition == "rank_growth_fail":
        tensor = rng.normal(size=dims)
        tensor /= max(float(np.linalg.norm(tensor.reshape(-1))), 1.0e-30)
        return tensor, int(round(math.sqrt(float(prod(dims)))))
    if condition == "compression_budget_fail":
        return make_low_rank_tensor(rng, dims, args.positive_rank), 1
    raise ValueError(f"unknown condition: {condition}")


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    dims = tuple([args.core_dim] * args.order)
    dense_dim = prod(dims)
    sqrt_dense_dim = math.sqrt(float(dense_dim))
    max_possible_rank = int(round(sqrt_dense_dim))
    tensor, rank_cap = make_condition_tensor(rng, args, condition, dims)
    matrices = make_mode_matrices(rng, args.order, args.core_dim)

    dense_target = apply_separable_dense(tensor, matrices)
    if condition == "entangled_operator_fail":
        dense_target = apply_entangled_dense(tensor, matrices, args.entangle_strength)

    input_cores = tt_svd(tensor, rank_cap)
    reconstructed_input = tt_to_dense(input_cores)
    initial_error = relative_error(reconstructed_input, tensor)
    candidate_cores = apply_separable_tt(input_cores, matrices)
    candidate_target = tt_to_dense(candidate_cores)
    transport_error = relative_error(candidate_target, dense_target)
    tt_rank = tt_max_rank(input_cores)
    required_rank, _ = minimum_rank_for_error(
        dense_target,
        args.epsilon_compress,
        max_possible_rank,
    )
    rank_growth_ratio = float(required_rank / sqrt_dense_dim)

    dense_work = float(dense_dim * dense_dim)
    transport_work = tt_transport_work(args.order, args.core_dim, max(tt_rank, required_rank))
    compress_work = compression_work(dims, max(tt_rank, required_rank))
    transport_ratio = transport_work / dense_work
    compression_ratio = compress_work / dense_work
    priced_ratio = transport_ratio + compression_ratio + args.metadata_work_ratio
    projected_speedup = 1.0 / priced_ratio if priced_ratio > 0.0 else math.inf

    compression_pass = initial_error <= args.epsilon_compress
    transport_pass = transport_error <= args.epsilon_compress
    rank_pass = rank_growth_ratio <= args.max_rank_growth_ratio
    price_pass = priced_ratio <= args.max_priced_work_ratio
    passed = bool(compression_pass and transport_pass and rank_pass and price_pass)
    expected_pass = condition == "positive"
    identifying_correct = passed == expected_pass

    fail_reason = None
    if not passed:
        if not compression_pass:
            fail_reason = "tt_compression_invariant_violated"
        elif not transport_pass:
            fail_reason = "adjoint_operator_not_represented_in_tt_core_space"
        elif not rank_pass or not price_pass:
            fail_reason = "required_tt_rank_recovers_dense_cost"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=identifying_correct,
        fail_reason=fail_reason,
        order=args.order,
        core_dim=args.core_dim,
        dense_dim=dense_dim,
        rank_cap=rank_cap,
        tt_rank_used=tt_rank,
        required_rank_for_epsilon=required_rank,
        sqrt_dense_dim=sqrt_dense_dim,
        rank_growth_ratio=rank_growth_ratio,
        initial_compression_error=initial_error,
        transport_relative_error=transport_error,
        priced_work_ratio=priced_ratio,
        transport_work_ratio=transport_ratio,
        compression_work_ratio=compression_ratio,
        metadata_work_ratio=float(args.metadata_work_ratio),
        projected_speedup=float(projected_speedup),
        dense_work=dense_work,
        tt_transport_work=transport_work,
        compression_work=compress_work,
    )


def finite_trial(trial: TrialResult) -> bool:
    values = (
        trial.rank_growth_ratio,
        trial.initial_compression_error,
        trial.transport_relative_error,
        trial.priced_work_ratio,
        trial.transport_work_ratio,
        trial.compression_work_ratio,
        trial.metadata_work_ratio,
        trial.projected_speedup,
        trial.dense_work,
        trial.tt_transport_work,
        trial.compression_work,
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
            "tt_rank_used_mean": float(np.mean([row.tt_rank_used for row in rows])),
            "tt_rank_used_max": float(np.max([row.tt_rank_used for row in rows])),
            "required_rank_for_epsilon_mean": float(
                np.mean([row.required_rank_for_epsilon for row in rows])
            ),
            "required_rank_for_epsilon_max": float(
                np.max([row.required_rank_for_epsilon for row in rows])
            ),
            "rank_growth_ratio_mean": float(
                np.mean([row.rank_growth_ratio for row in rows])
            ),
            "initial_compression_error_mean": float(
                np.mean([row.initial_compression_error for row in rows])
            ),
            "initial_compression_error_max": float(
                np.max([row.initial_compression_error for row in rows])
            ),
            "transport_relative_error_mean": float(
                np.mean([row.transport_relative_error for row in rows])
            ),
            "transport_relative_error_max": float(
                np.max([row.transport_relative_error for row in rows])
            ),
            "transport_work_ratio_mean": float(
                np.mean([row.transport_work_ratio for row in rows])
            ),
            "compression_work_ratio_mean": float(
                np.mean([row.compression_work_ratio for row in rows])
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
    if args.order < 3:
        raise ValueError("--order must be at least 3")
    if args.core_dim < 2:
        raise ValueError("--core-dim must be at least 2")
    if args.positive_rank < 1:
        raise ValueError("--positive-rank must be at least 1")

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
        "positive_condition_below_half_dense_price": all(
            trial.priced_work_ratio <= args.max_priced_work_ratio
            for trial in trials
            if trial.condition == "positive"
        ),
        "rank_growth_control_all_fail_price": all(
            (not trial.passed)
            and trial.fail_reason == "required_tt_rank_recovers_dense_cost"
            for trial in trials
            if trial.condition == "rank_growth_fail"
        ),
        "entangled_operator_control_all_fail_transport": all(
            (not trial.passed)
            and trial.fail_reason == "adjoint_operator_not_represented_in_tt_core_space"
            for trial in trials
            if trial.condition == "entangled_operator_fail"
        ),
        "compression_budget_control_all_fail_invariant": all(
            (not trial.passed)
            and trial.fail_reason == "tt_compression_invariant_violated"
            for trial in trials
            if trial.condition == "compression_budget_fail"
        ),
        "all_conditions_identified": all(trial.identifying_correct for trial in trials),
    }
    decision = (
        "PASS_R501_S46_TT_COTANGENT_TRANSPORT_GATE"
        if all(required_passes.values())
        else "FAIL_R501_S46_TT_COTANGENT_TRANSPORT_GATE"
    )
    disposition = (
        "synthetic_positive_control_identified_not_training_speed_claim"
        if decision.startswith("PASS_")
        else "tt_cotangent_transport_gate_failed_or_non_identifying"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "CPU synthetic Tensor-Train cotangent transport gate. It checks "
            "TT compression error, TT-core adjoint transport error, required "
            "rank growth, and priced work against dense transition cost. It "
            "is not heldout CE progress or GPU training-speed evidence."
        ),
        "finding": {
            "round": "R501",
            "aspect": "Speed",
            "id": "R501-S46",
            "title": "Tensor-Train Compressed State Cotangent Transport",
        },
        "thresholds": {
            "epsilon_compress": args.epsilon_compress,
            "max_priced_work_ratio": args.max_priced_work_ratio,
            "max_rank_growth_ratio": args.max_rank_growth_ratio,
            "metadata_work_ratio": args.metadata_work_ratio,
        },
        "shape": {
            "seed": args.seed,
            "seed_count": args.seed_count,
            "order": args.order,
            "core_dim": args.core_dim,
            "dense_dim": int(args.core_dim**args.order),
            "positive_rank": args.positive_rank,
        },
        "required_passes": required_passes,
        "aggregate": aggregate(trials),
        "trials": [asdict(trial) for trial in trials],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    positive = result["aggregate"]["positive"]
    print(
        "[R501_S46_TT_COTANGENT_TRANSPORT_GATE] "
        f"active=1 decision={decision} disposition={disposition} "
        f"positive_initial_compression_error_mean={positive['initial_compression_error_mean']:.6e} "
        f"positive_transport_relative_error_mean={positive['transport_relative_error_mean']:.6e} "
        f"positive_required_rank_mean={positive['required_rank_for_epsilon_mean']:.6f} "
        f"positive_rank_growth_ratio_mean={positive['rank_growth_ratio_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
