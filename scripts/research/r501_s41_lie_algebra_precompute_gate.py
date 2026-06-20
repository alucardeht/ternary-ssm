#!/usr/bin/env python3
"""R501 S41 Lie algebra generator precomputation gate.

This CPU diagnostic implements the R501 Speed Math 8 finding:
"State-Transition Lie Algebra Generator Precomputation".

The gate uses a finite token vocabulary whose state-transition generators have
closed-form sparse block exponentials. It compares dense per-token transition
application against a precomputed sparse/block representation. The finding
survives only when the precomputed representation is exact, exponentiation does
not densify the operator, and precompute cost amortizes over the sequence.

Controls exercise the declared kill criteria: exponentiation becoming dense,
precompute not amortizing over a large vocabulary, and an unsafe truncated
sparse representation that breaks exactness.

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


SCHEMA = "r501_s41_lie_algebra_precompute_gate.v1"
CONDITIONS = (
    "positive",
    "dense_exponential_fail",
    "large_vocab_precompute_fail",
    "truncated_structure_fail",
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
    vocab_size: int
    relative_solution_error: float
    max_abs_solution_error: float
    sparse_nnz_ratio: float
    precompute_amortized_ratio: float
    priced_work_ratio: float
    projected_speedup: float
    price_components: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50141)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--segment-length", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--large-vocab-size", type=int, default=512)
    parser.add_argument("--max-relative-solution-error", type=float, default=1.0e-10)
    parser.add_argument("--max-sparse-nnz-ratio", type=float, default=0.25)
    parser.add_argument("--max-precompute-amortized-ratio", type=float, default=0.25)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.50)
    parser.add_argument("--precompute-cost-factor", type=float, default=0.50)
    parser.add_argument("--audit-cost-ratio", type=float, default=0.020)
    return parser.parse_args()


def orthogonal_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def block_exponential_token(
    token: int,
    state_dim: int,
) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    dense = np.zeros((state_dim, state_dim), dtype=np.float64)
    blocks: list[tuple[int, np.ndarray]] = []
    for start in range(0, state_dim, 2):
        idx = start // 2
        angle = 0.015 * (token + 1) * (idx + 1)
        decay = 0.72 + 0.02 * ((token + idx) % 5)
        c = math.cos(angle)
        s = math.sin(angle)
        block = decay * np.array([[c, -s], [s, c]], dtype=np.float64)
        dense[start : start + 2, start : start + 2] = block
        blocks.append((start, block))
    return dense, blocks


def dense_token(rng: np.random.Generator, state_dim: int) -> np.ndarray:
    q = orthogonal_matrix(rng, state_dim)
    eigs = np.linspace(0.08, 0.78, state_dim)
    return (q @ np.diag(eigs) @ q.T).astype(np.float64)


def transition_tables(
    rng: np.random.Generator,
    condition: str,
    state_dim: int,
    vocab_size: int,
    large_vocab_size: int,
) -> tuple[list[np.ndarray], list[dict[str, Any]], list[int], int]:
    actual_vocab = large_vocab_size if condition == "large_vocab_precompute_fail" else vocab_size
    dense_tables: list[np.ndarray] = []
    reps: list[dict[str, Any]] = []
    nnz_counts: list[int] = []
    for token in range(actual_vocab):
        if condition == "dense_exponential_fail":
            dense = dense_token(rng, state_dim)
            dense_tables.append(dense)
            reps.append({"kind": "dense", "matrix": dense})
            nnz_counts.append(state_dim * state_dim)
        else:
            dense, blocks = block_exponential_token(token, state_dim)
            if condition == "truncated_structure_fail":
                off = rng.normal(scale=0.004, size=(state_dim, state_dim))
                for start in range(0, state_dim, 2):
                    off[start : start + 2, start : start + 2] = 0.0
                dense = dense + off
            dense_tables.append(dense)
            reps.append({"kind": "blocks", "blocks": blocks})
            nnz_counts.append(2 * state_dim)
    return dense_tables, reps, nnz_counts, actual_vocab


def apply_rep_transpose(carry: np.ndarray, rep: dict[str, Any], state_dim: int) -> np.ndarray:
    if rep["kind"] == "dense":
        return carry @ rep["matrix"]
    out = np.zeros(state_dim, dtype=np.float64)
    for start, block in rep["blocks"]:
        out[start : start + 2] = carry[start : start + 2] @ block
    return out


def dense_adjoint(
    forcing: np.ndarray,
    tokens: np.ndarray,
    dense_tables: list[np.ndarray],
) -> np.ndarray:
    out = np.zeros_like(forcing, dtype=np.float64)
    carry = np.zeros(forcing.shape[1], dtype=np.float64)
    for t in range(forcing.shape[0] - 1, -1, -1):
        carry = forcing[t] + carry @ dense_tables[int(tokens[t])]
        out[t] = carry
    return out


def sparse_adjoint(
    forcing: np.ndarray,
    tokens: np.ndarray,
    reps: list[dict[str, Any]],
) -> np.ndarray:
    out = np.zeros_like(forcing, dtype=np.float64)
    carry = np.zeros(forcing.shape[1], dtype=np.float64)
    state_dim = forcing.shape[1]
    for t in range(forcing.shape[0] - 1, -1, -1):
        carry = forcing[t] + apply_rep_transpose(carry, reps[int(tokens[t])], state_dim)
        out[t] = carry
    return out


def forcing_sequence(
    rng: np.random.Generator,
    segment_length: int,
    state_dim: int,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, segment_length, dtype=np.float64)
    x = np.stack(
        [
            np.ones_like(t),
            np.sin(2.0 * math.pi * t),
            np.cos(2.0 * math.pi * t),
            np.sin(4.0 * math.pi * t + 0.2),
        ],
        axis=1,
    )
    w = rng.normal(scale=0.050, size=(x.shape[1], state_dim))
    return (x @ w + 0.002 * rng.normal(size=(segment_length, state_dim))).astype(np.float64)


def relative_norm(error: np.ndarray, exact: np.ndarray) -> float:
    return float(np.linalg.norm(error.ravel()) / max(np.linalg.norm(exact.ravel()), 1.0e-30))


def price_route(
    args: argparse.Namespace,
    sparse_nnz_ratio: float,
    vocab_size: int,
) -> dict[str, float]:
    precompute = args.precompute_cost_factor * (vocab_size / float(args.segment_length))
    return {
        "sparse_algebra_apply": float(sparse_nnz_ratio),
        "precompute_amortized": float(precompute),
        "exactness_audit": float(args.audit_cost_ratio),
    }


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    dense_tables, reps, nnz_counts, vocab_size = transition_tables(
        rng, condition, args.state_dim, args.vocab_size, args.large_vocab_size
    )
    tokens = rng.integers(0, vocab_size, size=args.segment_length)
    forcing = forcing_sequence(rng, args.segment_length, args.state_dim)
    exact = dense_adjoint(forcing, tokens, dense_tables)
    approx = sparse_adjoint(forcing, tokens, reps)
    error = approx - exact
    relative_error = relative_norm(error, exact)
    max_abs_error = float(np.max(np.abs(error)))
    sparse_nnz_ratio = float(np.mean(nnz_counts) / float(args.state_dim * args.state_dim))
    price_components = price_route(args, sparse_nnz_ratio, vocab_size)
    precompute_ratio = float(price_components["precompute_amortized"])
    priced_work_ratio = float(sum(price_components.values()))
    projected_speedup = 1.0 / priced_work_ratio if priced_work_ratio > 0.0 else math.inf

    exact_pass = relative_error <= args.max_relative_solution_error
    sparsity_pass = sparse_nnz_ratio <= args.max_sparse_nnz_ratio
    precompute_pass = precompute_ratio <= args.max_precompute_amortized_ratio
    price_pass = priced_work_ratio <= args.max_priced_work_ratio
    passed = bool(exact_pass and sparsity_pass and precompute_pass and price_pass)
    expected_pass = condition == "positive"
    identifying_correct = passed == expected_pass

    fail_reason = None
    if not passed:
        if not exact_pass:
            fail_reason = "precomputed_sparse_representation_not_exact"
        elif not sparsity_pass:
            fail_reason = "generator_exponential_densifies_operator"
        elif not precompute_pass:
            fail_reason = "finite_vocab_precompute_not_amortized"
        elif not price_pass:
            fail_reason = "algebra_apply_plus_precompute_not_below_dense_budget"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=identifying_correct,
        fail_reason=fail_reason,
        segment_length=args.segment_length,
        state_dim=args.state_dim,
        vocab_size=vocab_size,
        relative_solution_error=relative_error,
        max_abs_solution_error=max_abs_error,
        sparse_nnz_ratio=sparse_nnz_ratio,
        precompute_amortized_ratio=precompute_ratio,
        priced_work_ratio=priced_work_ratio,
        projected_speedup=float(projected_speedup),
        price_components=price_components,
    )


def finite_trial(trial: TrialResult) -> bool:
    values = (
        trial.relative_solution_error,
        trial.max_abs_solution_error,
        trial.sparse_nnz_ratio,
        trial.precompute_amortized_ratio,
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
            "vocab_size_mean": float(np.mean([row.vocab_size for row in rows])),
            "relative_solution_error_mean": float(
                np.mean([row.relative_solution_error for row in rows])
            ),
            "relative_solution_error_max": float(
                np.max([row.relative_solution_error for row in rows])
            ),
            "max_abs_solution_error_max": float(
                np.max([row.max_abs_solution_error for row in rows])
            ),
            "sparse_nnz_ratio_mean": float(np.mean([row.sparse_nnz_ratio for row in rows])),
            "precompute_amortized_ratio_mean": float(
                np.mean([row.precompute_amortized_ratio for row in rows])
            ),
            "priced_work_ratio_mean": float(
                np.mean([row.priced_work_ratio for row in rows])
            ),
            "projected_speedup_mean": float(
                np.mean([row.projected_speedup for row in rows])
            ),
        }
    return out


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.state_dim < 2 or args.state_dim % 2 != 0:
        raise ValueError("--state-dim must be an even integer >= 2")
    if args.vocab_size < 1 or args.large_vocab_size < 1:
        raise ValueError("vocab sizes must be positive")

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
        "positive_condition_exact": all(
            trial.relative_solution_error <= args.max_relative_solution_error
            for trial in trials
            if trial.condition == "positive"
        ),
        "dense_exponential_control_all_fail_sparsity": all(
            (not trial.passed)
            and trial.fail_reason == "generator_exponential_densifies_operator"
            for trial in trials
            if trial.condition == "dense_exponential_fail"
        ),
        "large_vocab_control_all_fail_precompute": all(
            (not trial.passed)
            and trial.fail_reason == "finite_vocab_precompute_not_amortized"
            for trial in trials
            if trial.condition == "large_vocab_precompute_fail"
        ),
        "truncated_structure_control_all_fail_exactness": all(
            (not trial.passed)
            and trial.fail_reason == "precomputed_sparse_representation_not_exact"
            for trial in trials
            if trial.condition == "truncated_structure_fail"
        ),
        "all_conditions_identified": all(trial.identifying_correct for trial in trials),
    }
    decision = (
        "PASS_R501_S41_LIE_ALGEBRA_PRECOMPUTE_GATE"
        if all(required_passes.values())
        else "FAIL_R501_S41_LIE_ALGEBRA_PRECOMPUTE_GATE"
    )
    disposition = (
        "synthetic_positive_control_identified_not_training_speed_claim"
        if decision.startswith("PASS_")
        else "lie_algebra_precompute_gate_failed_or_non_identifying"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "CPU synthetic finite-token Lie-algebra precompute gate. It checks "
            "exact dense/sparse transition agreement, exponentiated sparsity, "
            "precompute amortization, and priced sparse apply work. It is not "
            "heldout CE progress or GPU training-speed evidence."
        ),
        "finding": {
            "round": "R501",
            "aspect": "Speed",
            "id": "R501-S41",
            "title": "State-Transition Lie Algebra Generator Precomputation",
        },
        "thresholds": {
            "max_relative_solution_error": args.max_relative_solution_error,
            "max_sparse_nnz_ratio": args.max_sparse_nnz_ratio,
            "max_precompute_amortized_ratio": args.max_precompute_amortized_ratio,
            "max_priced_work_ratio": args.max_priced_work_ratio,
            "required_projected_speedup": 1.0 / args.max_priced_work_ratio,
        },
        "shape": {
            "seed": args.seed,
            "seed_count": args.seed_count,
            "segment_length": args.segment_length,
            "state_dim": args.state_dim,
            "vocab_size": args.vocab_size,
            "large_vocab_size": args.large_vocab_size,
        },
        "required_passes": required_passes,
        "aggregate": aggregate(trials),
        "trials": [asdict(trial) for trial in trials],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    positive = result["aggregate"]["positive"]
    print(
        "[R501_S41_LIE_ALGEBRA_PRECOMPUTE_GATE] "
        f"active=1 decision={decision} disposition={disposition} "
        f"positive_relative_solution_error_mean={positive['relative_solution_error_mean']:.6e} "
        f"positive_sparse_nnz_ratio_mean={positive['sparse_nnz_ratio_mean']:.6f} "
        f"positive_precompute_ratio_mean={positive['precompute_amortized_ratio_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
