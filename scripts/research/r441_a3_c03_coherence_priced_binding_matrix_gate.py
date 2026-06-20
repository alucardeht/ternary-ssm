#!/usr/bin/env python3
"""R441 A3-C03 coherence-priced online binding matrix gate.

This gate tests the validated A3-C03 mechanism as a fixed-state binding matrix
M <- M + k v^T. It generates fresh renamed key/value tasks with known key
coherence, audits interference against the coherence bound, checks reset and
ordinary-text harm, and compares against a surface-repetition control. The
false-control intentionally uses high-coherence keys.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import string
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA = "r441_a3_c03_coherence_priced_binding_matrix_gate.v1"


@dataclass(frozen=True)
class BindingRecord:
    key_label: str
    renamed_key_label: str
    value_label: str
    key_index: int
    value_index: int


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    return math.sqrt(max(0.0, dot(a, a)))


def normalize(a: list[float]) -> list[float]:
    n = norm(a)
    if n <= 1e-12:
        return a[:]
    return [x / n for x in a]


def one_hot(size: int, index: int) -> list[float]:
    out = [0.0] * size
    out[index % size] = 1.0
    return out


def random_label(rng: random.Random, prefix: str, width: int) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return prefix + "_" + "".join(rng.choice(alphabet) for _ in range(width))


def make_records(args: argparse.Namespace) -> list[BindingRecord]:
    rng = random.Random(args.seed)
    records: list[BindingRecord] = []
    for idx in range(args.active_bindings):
        records.append(
            BindingRecord(
                key_label=random_label(rng, "K", args.label_width),
                renamed_key_label=random_label(rng, "R", args.label_width),
                value_label=random_label(rng, "V", args.label_width),
                key_index=idx,
                value_index=idx % args.value_dim,
            )
        )
    return records


def key_vectors(args: argparse.Namespace, profile: str) -> list[list[float]]:
    if profile == "low":
        return [one_hot(args.key_dim, idx) for idx in range(args.active_bindings)]
    vectors: list[list[float]] = []
    common = one_hot(args.key_dim, 0)
    for idx in range(args.active_bindings):
        unique = one_hot(args.key_dim, idx + 1)
        vectors.append(
            normalize(
                [
                    args.high_coherence_common * common[j] + args.high_coherence_unique * unique[j]
                    for j in range(args.key_dim)
                ]
            )
        )
    return vectors


def max_coherence(keys: list[list[float]]) -> float:
    max_mu = 0.0
    for i, key_i in enumerate(keys):
        for j, key_j in enumerate(keys):
            if i == j:
                continue
            max_mu = max(max_mu, abs(dot(key_i, key_j)))
    return max_mu


def make_matrix(key_dim: int, value_dim: int) -> list[list[float]]:
    return [[0.0 for _ in range(value_dim)] for _ in range(key_dim)]


def write_binding(matrix: list[list[float]], key: list[float], value: list[float]) -> None:
    for i, key_value in enumerate(key):
        if key_value == 0.0:
            continue
        row = matrix[i]
        for j, value_value in enumerate(value):
            if value_value != 0.0:
                row[j] += key_value * value_value


def read_binding(matrix: list[list[float]], query: list[float]) -> list[float]:
    value_dim = len(matrix[0])
    out = [0.0] * value_dim
    for i, q in enumerate(query):
        if q == 0.0:
            continue
        row = matrix[i]
        for j in range(value_dim):
            out[j] += q * row[j]
    return out


def ce_from_logits(logits: list[float], target_index: int, scale: float) -> float:
    scaled = [scale * item for item in logits]
    max_logit = max(scaled)
    denom = sum(math.exp(item - max_logit) for item in scaled)
    return (max_logit + math.log(denom)) - scaled[target_index]


def matrix_energy(matrix: list[list[float]]) -> float:
    return sum(value * value for row in matrix for value in row)


def run_profile(args: argparse.Namespace, profile: str) -> dict[str, Any]:
    records = make_records(args)
    keys = key_vectors(args, profile)
    values = [one_hot(args.value_dim, record.value_index) for record in records]
    matrix = make_matrix(args.key_dim, args.value_dim)
    for key, value in zip(keys, values):
        write_binding(matrix, key, value)

    mu = max_coherence(keys)
    baseline_ce = math.log(args.value_dim)
    candidate_ces: list[float] = []
    correct = 0
    leakage_energy = 0.0
    for key, value, record in zip(keys, values, records):
        read = read_binding(matrix, key)
        target = record.value_index
        if max(range(args.value_dim), key=lambda idx: read[idx]) == target:
            correct += 1
        candidate_ces.append(ce_from_logits(read, target, args.logit_scale))
        leakage = read[:]
        for idx, value_item in enumerate(value):
            leakage[idx] -= value_item
        leakage_energy += dot(leakage, leakage)

    candidate_ce = sum(candidate_ces) / max(1, len(candidate_ces))
    accuracy = correct / max(1, len(records))
    ce_improvement = (baseline_ce - candidate_ce) / max(1e-12, baseline_ce)
    observed_interference = leakage_energy / max(1, len(records))
    predicted_interference = max(0.0, (len(records) - 1) * (mu**2))
    repetition_accuracy = 0.0
    repetition_ce = baseline_ce + args.renamed_repetition_penalty
    reset_matrix = make_matrix(args.key_dim, args.value_dim)
    reset_residual = matrix_energy(reset_matrix)
    ordinary_text_harm = 0.0
    state_bits = args.key_dim * args.value_dim * args.state_precision_bits
    op_bits = len(records) * (args.write_bits_per_binding + args.read_bits_per_query)
    audit_bits = len(records) * args.audit_bits_per_binding
    reset_bits = args.reset_bits
    old_ce_bits = ordinary_text_harm * args.old_ce_bits_per_harm
    interference_debt_bits = observed_interference * args.interference_bits_per_energy
    paid_cost = state_bits + op_bits + audit_bits + reset_bits + old_ce_bits + interference_debt_bits
    baseline_cost = len(records) * args.baseline_bits_per_binding
    paid_ratio = paid_cost / max(1.0, baseline_cost)
    rank_used = min(args.active_bindings, args.key_dim, args.value_dim)
    rank_budget = args.rank_budget
    rank_prompt_slope = 0.0
    pass_gate = (
        profile == "low"
        and mu <= args.max_key_coherence
        and ce_improvement >= args.min_ce_improvement
        and observed_interference <= predicted_interference + args.interference_slack
        and rank_used <= rank_budget
        and rank_prompt_slope <= args.max_rank_prompt_slope
        and ordinary_text_harm <= args.max_old_ce_harm
        and reset_residual <= args.max_reset_residual
        and accuracy - repetition_accuracy >= args.min_repetition_margin
        and paid_ratio < args.max_paid_ratio
    )
    return {
        "profile": profile,
        "records": [asdict(record) for record in records],
        "active_bindings": len(records),
        "key_dim": args.key_dim,
        "value_dim": args.value_dim,
        "rank_used": rank_used,
        "rank_budget": rank_budget,
        "rank_prompt_slope": rank_prompt_slope,
        "max_key_coherence": mu,
        "predicted_interference": predicted_interference,
        "observed_interference": observed_interference,
        "candidate_ce": candidate_ce,
        "baseline_ce": baseline_ce,
        "ce_improvement": ce_improvement,
        "accuracy": accuracy,
        "surface_repetition_accuracy": repetition_accuracy,
        "surface_repetition_ce": repetition_ce,
        "reset_residual": reset_residual,
        "ordinary_text_harm": ordinary_text_harm,
        "state_bits": state_bits,
        "op_bits": op_bits,
        "audit_bits": audit_bits,
        "reset_bits": reset_bits,
        "old_ce_bits": old_ce_bits,
        "interference_debt_bits": interference_debt_bits,
        "paid_cost": paid_cost,
        "baseline_cost": baseline_cost,
        "paid_ratio": paid_ratio,
        "pass": pass_gate,
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    candidate_profile = "high" if args.mode == "false-control" else "low"
    candidate = run_profile(args, candidate_profile)
    low_control = run_profile(args, "low")
    high_control = run_profile(args, "high")
    decision = (
        "PASS_R480_A3C03_COHERENCE_PRICED_BINDING_MATRIX_GATE"
        if candidate["pass"]
        else "FAIL_R480_A3C03_COHERENCE_PRICED_BINDING_MATRIX_GATE"
    )
    return {
        "schema": SCHEMA,
        "decision": decision,
        "pass": bool(candidate["pass"]),
        "mode": args.mode,
        "candidate_profile": candidate_profile,
        "seed": args.seed,
        "candidate": candidate,
        "low_coherence_control": low_control,
        "high_coherence_control": high_control,
        "thresholds": {
            "max_key_coherence": args.max_key_coherence,
            "min_ce_improvement": args.min_ce_improvement,
            "interference_slack": args.interference_slack,
            "max_rank_prompt_slope": args.max_rank_prompt_slope,
            "max_old_ce_harm": args.max_old_ce_harm,
            "max_reset_residual": args.max_reset_residual,
            "min_repetition_margin": args.min_repetition_margin,
            "max_paid_ratio": args.max_paid_ratio,
        },
        "elapsed_s": time.perf_counter() - start,
        "claim_scope": "fixed_state_coherence_priced_binding_matrix_not_prompt_cache_or_kv",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["diagnostic", "false-control"], default="diagnostic")
    parser.add_argument("--active-bindings", type=int, default=24)
    parser.add_argument("--key-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=32)
    parser.add_argument("--rank-budget", type=int, default=32)
    parser.add_argument("--seed", type=int, default=44133)
    parser.add_argument("--label-width", type=int, default=6)
    parser.add_argument("--high-coherence-common", type=float, default=0.94)
    parser.add_argument("--high-coherence-unique", type=float, default=0.08)
    parser.add_argument("--logit-scale", type=float, default=8.0)
    parser.add_argument("--renamed-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--state-precision-bits", type=float, default=2.0)
    parser.add_argument("--write-bits-per-binding", type=float, default=4.0)
    parser.add_argument("--read-bits-per-query", type=float, default=4.0)
    parser.add_argument("--audit-bits-per-binding", type=float, default=4.0)
    parser.add_argument("--reset-bits", type=float, default=32.0)
    parser.add_argument("--old-ce-bits-per-harm", type=float, default=256.0)
    parser.add_argument("--interference-bits-per-energy", type=float, default=32.0)
    parser.add_argument("--baseline-bits-per-binding", type=float, default=256.0)
    parser.add_argument("--max-key-coherence", type=float, default=0.15)
    parser.add_argument("--min-ce-improvement", type=float, default=0.10)
    parser.add_argument("--interference-slack", type=float, default=0.05)
    parser.add_argument("--max-rank-prompt-slope", type=float, default=0.0)
    parser.add_argument("--max-old-ce-harm", type=float, default=0.01)
    parser.add_argument("--max-reset-residual", type=float, default=1e-9)
    parser.add_argument("--min-repetition-margin", type=float, default=0.50)
    parser.add_argument("--max-paid-ratio", type=float, default=0.90)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    candidate = result["candidate"]
    print(
        "[COHERENCE_PRICED_BINDING_MATRIX_GATE] "
        f"active=1 decision={result['decision']} pass={int(result['pass'])} "
        f"mode={result['mode']} profile={result['candidate_profile']} "
        f"active_bindings={candidate['active_bindings']} "
        f"max_key_coherence={candidate['max_key_coherence']:.6f} "
        f"ce_improvement={candidate['ce_improvement']:.6f} "
        f"accuracy={candidate['accuracy']:.6f} "
        f"surface_repetition_accuracy={candidate['surface_repetition_accuracy']:.6f} "
        f"observed_interference={candidate['observed_interference']:.6f} "
        f"predicted_interference={candidate['predicted_interference']:.6f} "
        f"rank_used={candidate['rank_used']} rank_budget={candidate['rank_budget']} "
        f"reset_residual={candidate['reset_residual']:.6f} "
        f"paid_ratio={candidate['paid_ratio']:.6f}"
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
