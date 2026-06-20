#!/usr/bin/env python3
"""R501 S100 exact-quotient padding elimination block gate.

This CPU diagnostic implements the seventh R501 Hardware Efficiency
mathematical finding: "Exact-Quotient Padding Elimination Block (EQPEB)".

The gate constructs padded variable-length recurrent sequences, packs only
active tokens, applies explicit sequence-boundary resets in the packed SSM
recurrence, and unpacks forward states plus input gradients back to the padded
layout. The positive condition must match the dense padded path exactly while
charging only active-token matrix residency.

Controls exercise the declared kill criteria: missing resets leak state across
sequences, wrong unpack maps gradients to the wrong padded positions, dropping
active tokens loses gradient mass, dense padding erases the denominator, and
including padded tokens erases the denominator even when exactness is preserved.

This is a synthetic CPU exact padding-elimination diagnostic, not heldout CE
progress, real VRAM telemetry, GPU training-speed evidence, or production v12
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


SCHEMA = "r501_s100_eqpeb_gate.v1"
CONDITIONS = (
    "positive",
    "no_reset_fail",
    "wrong_unpack_fail",
    "drop_active_fail",
    "dense_padding_fail",
    "include_padding_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    batch_size: int
    max_sequence_len: int
    input_dim: int
    state_dim: int
    active_token_count: int
    packed_token_count: int
    active_token_fraction: float
    active_token_recall: float
    forward_relative_error: float
    forward_max_abs_error: float
    gradient_relative_error: float
    gradient_max_abs_error: float
    paid_matrix_ratio: float
    projected_padding_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50200)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-sequence-len", type=int, default=64)
    parser.add_argument("--min-sequence-len", type=int, default=24)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--max-forward-relative-error", type=float, default=1.0e-12)
    parser.add_argument("--max-gradient-relative-error", type=float, default=1.0e-12)
    parser.add_argument("--max-abs-error", type=float, default=1.0e-12)
    parser.add_argument("--min-active-token-recall", type=float, default=1.0)
    parser.add_argument("--max-paid-matrix-ratio", type=float, default=0.80)
    parser.add_argument("--min-padding-efficiency-multiplier", type=float, default=1.25)
    return parser.parse_args()


def make_fixture(
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lengths = rng.integers(
        args.min_sequence_len,
        args.max_sequence_len + 1,
        size=args.batch_size,
        endpoint=False,
    )
    inputs = rng.normal(
        size=(args.batch_size, args.max_sequence_len, args.input_dim)
    ).astype(np.float64)
    cotangents = rng.normal(
        size=(args.batch_size, args.max_sequence_len, args.state_dim)
    ).astype(np.float64)
    transition = rng.uniform(-0.85, 0.85, size=args.state_dim).astype(np.float64)
    input_matrix = (
        rng.normal(size=(args.state_dim, args.input_dim)) / math.sqrt(args.input_dim)
    ).astype(np.float64)
    return lengths.astype(np.int64), inputs, cotangents, transition, input_matrix


def active_mask(lengths: np.ndarray, max_len: int) -> np.ndarray:
    positions = np.arange(max_len)[None, :]
    return positions < lengths[:, None]


def dense_forward_backward(
    lengths: np.ndarray,
    inputs: np.ndarray,
    cotangents: np.ndarray,
    transition: np.ndarray,
    input_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    batch, max_len, _input_dim = inputs.shape
    state_dim = transition.shape[0]
    states = np.zeros((batch, max_len, state_dim), dtype=np.float64)
    grad_x = np.zeros_like(inputs)

    for b in range(batch):
        state = np.zeros(state_dim, dtype=np.float64)
        for t in range(int(lengths[b])):
            state = transition * state + input_matrix @ inputs[b, t]
            states[b, t] = state
        adjoint = np.zeros(state_dim, dtype=np.float64)
        for t in range(int(lengths[b]) - 1, -1, -1):
            adjoint = cotangents[b, t] + transition * adjoint
            grad_x[b, t] = input_matrix.T @ adjoint
    return states, grad_x


def build_mapping(lengths: np.ndarray, max_len: int, condition: str) -> list[tuple[int, int]]:
    mapping: list[tuple[int, int]] = []
    for b, length in enumerate(lengths):
        active_length = int(length)
        if condition == "drop_active_fail":
            active_length = max(active_length - 1, 0)
        token_range = range(max_len) if condition == "include_padding_fail" else range(active_length)
        for t in token_range:
            mapping.append((b, t))
    return mapping


def packed_forward_backward(
    lengths: np.ndarray,
    inputs: np.ndarray,
    cotangents: np.ndarray,
    transition: np.ndarray,
    input_matrix: np.ndarray,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    batch, max_len, _input_dim = inputs.shape
    state_dim = transition.shape[0]
    mapping = build_mapping(lengths, max_len, condition)
    states = np.zeros((batch, max_len, state_dim), dtype=np.float64)
    grad_x = np.zeros_like(inputs)

    state = np.zeros(state_dim, dtype=np.float64)
    previous_b: int | None = None
    packed_states: list[np.ndarray] = []
    for b, t in mapping:
        if condition != "no_reset_fail" and b != previous_b:
            state = np.zeros(state_dim, dtype=np.float64)
        state = transition * state + input_matrix @ inputs[b, t]
        packed_states.append(state.copy())
        previous_b = b

    if condition == "wrong_unpack_fail":
        unpack_mapping = mapping[1:] + mapping[:1]
    else:
        unpack_mapping = mapping
    for (b, t), state_value in zip(unpack_mapping, packed_states):
        if t < int(lengths[b]):
            states[b, t] = state_value

    spans: list[tuple[int, int, int]] = []
    start = 0
    while start < len(mapping):
        b = mapping[start][0]
        end = start
        while end < len(mapping) and mapping[end][0] == b:
            end += 1
        spans.append((b, start, end))
        start = end

    for b, start_idx, end_idx in spans:
        adjoint = np.zeros(state_dim, dtype=np.float64)
        for idx in range(end_idx - 1, start_idx - 1, -1):
            original_b, t = mapping[idx]
            if original_b != b or t >= int(lengths[b]):
                continue
            adjoint = cotangents[b, t] + transition * adjoint
            target_b, target_t = (unpack_mapping[idx] if condition == "wrong_unpack_fail" else (b, t))
            if target_t < int(lengths[target_b]):
                grad_x[target_b, target_t] = input_matrix.T @ adjoint

    active_count = int(np.sum(lengths))
    represented_active = sum(1 for b, t in mapping if t < int(lengths[b]))
    recall = float(represented_active / max(active_count, 1))
    return states, grad_x, len(mapping), recall


def relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference))
    return float(np.linalg.norm(candidate - reference) / max(denom, 1.0e-30))


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    lengths, inputs, cotangents, transition, input_matrix = make_fixture(rng, args)
    dense_states, dense_grad_x = dense_forward_backward(
        lengths, inputs, cotangents, transition, input_matrix
    )
    if condition == "dense_padding_fail":
        candidate_states = dense_states.copy()
        candidate_grad_x = dense_grad_x.copy()
        packed_count = int(args.batch_size * args.max_sequence_len)
        recall = 1.0
    else:
        candidate_states, candidate_grad_x, packed_count, recall = packed_forward_backward(
            lengths, inputs, cotangents, transition, input_matrix, condition
        )

    mask = active_mask(lengths, args.max_sequence_len)[..., None]
    masked_candidate_states = np.where(mask, candidate_states, 0.0)
    masked_dense_states = np.where(mask, dense_states, 0.0)
    forward_diff = masked_candidate_states - masked_dense_states
    grad_mask = active_mask(lengths, args.max_sequence_len)[..., None]
    masked_candidate_grad_x = np.where(grad_mask, candidate_grad_x, 0.0)
    masked_dense_grad_x = np.where(grad_mask, dense_grad_x, 0.0)
    grad_diff = masked_candidate_grad_x - masked_dense_grad_x
    forward_error = relative_error(masked_candidate_states, masked_dense_states)
    gradient_error = relative_error(masked_candidate_grad_x, masked_dense_grad_x)
    forward_max_abs = float(np.max(np.abs(forward_diff)))
    gradient_max_abs = float(np.max(np.abs(grad_diff)))

    total_tokens = int(args.batch_size * args.max_sequence_len)
    active_count = int(np.sum(lengths))
    paid_ratio = float(packed_count / total_tokens)
    multiplier = 1.0 / paid_ratio

    failures: list[str] = []
    if forward_error > args.max_forward_relative_error or forward_max_abs > args.max_abs_error:
        failures.append("forward_error")
    if gradient_error > args.max_gradient_relative_error or gradient_max_abs > args.max_abs_error:
        failures.append("gradient_error")
    if recall < args.min_active_token_recall:
        failures.append("active_token_recall")
    if paid_ratio > args.max_paid_matrix_ratio or multiplier < args.min_padding_efficiency_multiplier:
        failures.append("price_paid_matrix")

    fail_reason = "+".join(failures) if failures else None
    passed = fail_reason is None
    expected_pass = condition == "positive"
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=fail_reason,
        batch_size=args.batch_size,
        max_sequence_len=args.max_sequence_len,
        input_dim=args.input_dim,
        state_dim=args.state_dim,
        active_token_count=active_count,
        packed_token_count=packed_count,
        active_token_fraction=float(active_count / total_tokens),
        active_token_recall=recall,
        forward_relative_error=forward_error,
        forward_max_abs_error=forward_max_abs,
        gradient_relative_error=gradient_error,
        gradient_max_abs_error=gradient_max_abs,
        paid_matrix_ratio=paid_ratio,
        projected_padding_efficiency_multiplier=multiplier,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "batch_size",
        "max_sequence_len",
        "input_dim",
        "state_dim",
        "active_token_count",
        "packed_token_count",
        "active_token_fraction",
        "active_token_recall",
        "forward_relative_error",
        "forward_max_abs_error",
        "gradient_relative_error",
        "gradient_max_abs_error",
        "paid_matrix_ratio",
        "projected_padding_efficiency_multiplier",
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
    no_reset = [result for result in results if result.condition == "no_reset_fail"]
    wrong = [result for result in results if result.condition == "wrong_unpack_fail"]
    drop = [result for result in results if result.condition == "drop_active_fail"]
    dense = [result for result in results if result.condition == "dense_padding_fail"]
    include = [result for result in results if result.condition == "include_padding_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_forward_and_gradient": (
            aggregates["positive"]["forward_relative_error_max"] <= 1.0e-12
            and aggregates["positive"]["gradient_relative_error_max"] <= 1.0e-12
            and aggregates["positive"]["active_token_recall_min"] >= 1.0
        ),
        "positive_condition_priced": (
            aggregates["positive"]["paid_matrix_ratio_mean"] <= 0.80
            and aggregates["positive"]["projected_padding_efficiency_multiplier_mean"] >= 1.25
        ),
        "no_reset_control_all_fail_forward_or_gradient": all(
            (not result.passed)
            and result.fail_reason is not None
            and ("forward_error" in result.fail_reason or "gradient_error" in result.fail_reason)
            for result in no_reset
        ),
        "wrong_unpack_control_all_fail_forward_or_gradient": all(
            (not result.passed)
            and result.fail_reason is not None
            and ("forward_error" in result.fail_reason or "gradient_error" in result.fail_reason)
            for result in wrong
        ),
        "drop_active_control_all_fail_recall_or_gradient": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "active_token_recall" in result.fail_reason
                or "gradient_error" in result.fail_reason
                or "forward_error" in result.fail_reason
            )
            for result in drop
        ),
        "dense_padding_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "price_paid_matrix" in result.fail_reason
            for result in dense
        ),
        "include_padding_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "price_paid_matrix" in result.fail_reason
            for result in include
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
    decision = "PASS_R501_S100_EQPEB_GATE" if all(checks.values()) else "FAIL_R501_S100_EQPEB_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU exact-quotient padding elimination gate only; not "
            "heldout CE progress, not real VRAM telemetry, not GPU training "
            "speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S100_EQPEB_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_active_token_fraction_mean={positive['active_token_fraction_mean']:.6f} "
        f"positive_active_token_recall_min={positive['active_token_recall_min']:.6f} "
        f"positive_forward_relative_error_max={positive['forward_relative_error_max']:.6e} "
        f"positive_gradient_relative_error_max={positive['gradient_relative_error_max']:.6e} "
        f"positive_paid_matrix_ratio_mean={positive['paid_matrix_ratio_mean']:.6f} "
        f"positive_projected_padding_efficiency_multiplier_mean={positive['projected_padding_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
