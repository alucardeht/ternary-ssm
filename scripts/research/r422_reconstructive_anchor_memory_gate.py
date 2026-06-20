#!/usr/bin/env python3
"""R422 reconstructive anchor memory gate.

Tests whether a compact state can preserve minimal latent anchors that
reconstruct delayed, future-relevant variables on demand. The gate separates a
true reconstructive latent factor from lossy summaries, surface scan, partial
anchors, and full flat memory.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_reconstructive_anchor_memory_gate.v1"


@dataclass
class ArmMetrics:
    arm: str
    accuracy: float
    mean_paid_work: float


@dataclass
class GateResult:
    schema: str
    decision: str
    pass_gate: bool
    mode: str
    variables: int
    rank: int
    anchor_dim: int
    query_terms: int
    queries: int
    anchor_accuracy: float
    partial_anchor_accuracy: float
    lossy_summary_accuracy: float
    surface_scan_accuracy: float
    full_memory_accuracy: float
    anchor_work_ratio: float
    full_memory_work_ratio: float
    anchor_gain_vs_best_admissible_control: float
    activation_log: str
    interpretation: str
    arms: list[ArmMetrics]
    elapsed_seconds: float


def dot_mod(a: list[int], b: list[int], modulus: int) -> int:
    return sum(x * y for x, y in zip(a, b)) % modulus


def make_coefficients(variables: int, rank: int, modulus: int, rng: random.Random) -> list[list[int]]:
    coeffs: list[list[int]] = []
    for _ in range(variables):
        row = [rng.randrange(modulus) for _ in range(rank)]
        if all(value == 0 for value in row):
            row[rng.randrange(rank)] = 1
        coeffs.append(row)
    return coeffs


def generate_episode(
    mode: str,
    rank: int,
    variables: int,
    coeffs: list[list[int]],
    modulus: int,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    anchors = [rng.randrange(modulus) for _ in range(rank)]
    if mode == "lowrank":
        values = [dot_mod(coeff, anchors, modulus) for coeff in coeffs]
    elif mode == "random":
        values = [rng.randrange(modulus) for _ in range(variables)]
    else:
        raise ValueError(f"unknown mode: {mode}")
    return anchors, values


def reconstruct_value(coeff: list[int], anchors: list[int], anchor_dim: int, modulus: int) -> int:
    return dot_mod(coeff[:anchor_dim], anchors[:anchor_dim], modulus)


def majority_value(values: list[int]) -> int:
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get)


def run_gate(args: argparse.Namespace) -> GateResult:
    start_time = time.perf_counter()
    rng = random.Random(args.seed)
    coeffs = make_coefficients(args.variables, args.rank, args.modulus, rng)
    partial_dim = max(1, min(args.anchor_dim - 1, args.rank // 2))

    anchor_correct = 0
    partial_correct = 0
    lossy_correct = 0
    surface_correct = 0
    full_correct = 0

    for _ in range(args.queries):
        anchors, values = generate_episode(args.mode, args.rank, args.variables, coeffs, args.modulus, rng)
        query_indices = [rng.randrange(args.variables) for _ in range(args.query_terms)]
        query_weights = [rng.randrange(1, args.modulus) for _ in range(args.query_terms)]
        truth = sum(weight * values[index] for weight, index in zip(query_weights, query_indices)) % args.modulus

        anchor_pred = 0
        partial_pred = 0
        for weight, index in zip(query_weights, query_indices):
            anchor_pred += weight * reconstruct_value(coeffs[index], anchors, args.anchor_dim, args.modulus)
            partial_pred += weight * reconstruct_value(coeffs[index], anchors, partial_dim, args.modulus)
        anchor_pred %= args.modulus
        partial_pred %= args.modulus

        summary_value = sum(values) % args.modulus
        lossy_pred = sum(query_weights) * summary_value % args.modulus
        surface_pred = majority_value(values)
        full_pred = truth

        anchor_correct += anchor_pred == truth
        partial_correct += partial_pred == truth
        lossy_correct += lossy_pred == truth
        surface_correct += surface_pred == truth
        full_correct += full_pred == truth

    anchor_accuracy = anchor_correct / args.queries
    partial_accuracy = partial_correct / args.queries
    lossy_accuracy = lossy_correct / args.queries
    surface_accuracy = surface_correct / args.queries
    full_accuracy = full_correct / args.queries

    anchor_work_ratio = args.anchor_dim / max(1, args.variables)
    partial_work_ratio = partial_dim / max(1, args.variables)
    summary_work_ratio = 1 / max(1, args.variables)
    full_work_ratio = 1.0
    best_admissible = max(partial_accuracy, lossy_accuracy, surface_accuracy)
    gain = anchor_accuracy / max(1e-9, best_admissible)

    pass_gate = (
        args.mode == "lowrank"
        and args.anchor_dim >= args.rank
        and anchor_accuracy >= args.min_accuracy
        and best_admissible <= args.max_control_accuracy
        and anchor_work_ratio <= args.max_work_ratio
    )
    if pass_gate:
        decision = "PASS_RECONSTRUCTIVE_ANCHOR_MEMORY_GATE"
        interpretation = (
            "Minimal latent anchors reconstruct delayed future-relevance queries "
            "without storing the full variable table."
        )
    elif args.mode != "lowrank":
        decision = "FAIL_NONRECONSTRUCTIVE_RANDOM_VARIABLE_CONTROL"
        interpretation = (
            "The anchor state cannot reconstruct variables that are not generated "
            "from a recoverable latent factor."
        )
    elif args.anchor_dim < args.rank:
        decision = "FAIL_INSUFFICIENT_ANCHOR_DIMENSION"
        interpretation = "The anchor state omitted latent factors needed for delayed reconstruction."
    elif best_admissible > args.max_control_accuracy:
        decision = "FAIL_LOSSY_OR_SURFACE_CONTROL_TOO_STRONG"
        interpretation = "A lossy, partial, or surface control is too strong to isolate anchor memory."
    else:
        decision = "FAIL_RECONSTRUCTIVE_ANCHOR_MEMORY_GATE"
        interpretation = "The reconstructive anchor did not clear accuracy or pricing thresholds."

    arms = [
        ArmMetrics("reconstructive_anchor", anchor_accuracy, anchor_work_ratio),
        ArmMetrics("partial_anchor", partial_accuracy, partial_work_ratio),
        ArmMetrics("lossy_summary", lossy_accuracy, summary_work_ratio),
        ArmMetrics("surface_scan_majority", surface_accuracy, summary_work_ratio),
        ArmMetrics("full_memory_oracle_pricing_control", full_accuracy, full_work_ratio),
    ]
    activation_log = (
        "[R422_RECONSTRUCTIVE_ANCHOR_MEMORY_GATE] "
        f"active=1 decision={decision} mode={args.mode} rank={args.rank} "
        f"anchor_dim={args.anchor_dim} no_gpu=1 no_training_integration=1"
    )
    return GateResult(
        schema=SCHEMA,
        decision=decision,
        pass_gate=pass_gate,
        mode=args.mode,
        variables=args.variables,
        rank=args.rank,
        anchor_dim=args.anchor_dim,
        query_terms=args.query_terms,
        queries=args.queries,
        anchor_accuracy=anchor_accuracy,
        partial_anchor_accuracy=partial_accuracy,
        lossy_summary_accuracy=lossy_accuracy,
        surface_scan_accuracy=surface_accuracy,
        full_memory_accuracy=full_accuracy,
        anchor_work_ratio=anchor_work_ratio,
        full_memory_work_ratio=full_work_ratio,
        anchor_gain_vs_best_admissible_control=gain,
        activation_log=activation_log,
        interpretation=interpretation,
        arms=arms,
        elapsed_seconds=time.perf_counter() - start_time,
    )


def write_result(result: GateResult, output: Path | None) -> None:
    payload = asdict(result)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result.activation_log)
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["lowrank", "random"], default="lowrank")
    parser.add_argument("--variables", type=int, default=256)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--anchor-dim", type=int, default=8)
    parser.add_argument("--query-terms", type=int, default=4)
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--modulus", type=int, default=251)
    parser.add_argument("--min-accuracy", type=float, default=0.95)
    parser.add_argument("--max-control-accuracy", type=float, default=0.20)
    parser.add_argument("--max-work-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42206)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    write_result(run_gate(args), args.output)


if __name__ == "__main__":
    main()
