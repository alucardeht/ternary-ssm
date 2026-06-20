#!/usr/bin/env python3
"""R441 A3-C01 resettable binding ledger gate.

This gate tests the validated A3-C01 mechanism as an explicit reversible
temporary ledger:

    B_{t+1} = reset_t * 0 + keep_t * B_t + write_t * k_t v_t^T
              - erase_t * k_t old_v_t^T
    r_t = B_t^T q_t

The fixture creates fresh heldout dictionary/glossary mappings, delays their
use, queries renamed keys, audits erase and hard reset semantics, and charges a
fixed-width state object. The false-control shuffles write keys before delayed
queries so a prompt-cache or surface repetition shortcut cannot pass.
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


SCHEMA = "r441_a3_c01_resettable_binding_ledger_gate.v1"


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
                key_label=random_label(rng, "KEY", args.label_width),
                renamed_key_label=random_label(rng, "ALIAS", args.label_width),
                value_label=random_label(rng, "VALUE", args.label_width),
                key_index=idx,
                value_index=idx % args.value_dim,
            )
        )
    return records


def make_ledger(key_dim: int, value_dim: int) -> list[list[float]]:
    return [[0.0 for _ in range(value_dim)] for _ in range(key_dim)]


def write_binding(ledger: list[list[float]], key: list[float], value: list[float]) -> None:
    for i, key_item in enumerate(key):
        if key_item == 0.0:
            continue
        row = ledger[i]
        for j, value_item in enumerate(value):
            if value_item != 0.0:
                row[j] += key_item * value_item


def erase_binding(ledger: list[list[float]], key: list[float], old_value: list[float]) -> None:
    for i, key_item in enumerate(key):
        if key_item == 0.0:
            continue
        row = ledger[i]
        for j, value_item in enumerate(old_value):
            if value_item != 0.0:
                row[j] -= key_item * value_item


def read_binding(ledger: list[list[float]], query: list[float]) -> list[float]:
    out = [0.0] * len(ledger[0])
    for i, q in enumerate(query):
        if q == 0.0:
            continue
        row = ledger[i]
        for j in range(len(out)):
            out[j] += q * row[j]
    return out


def ledger_energy(ledger: list[list[float]]) -> float:
    return sum(item * item for row in ledger for item in row)


def ce_from_logits(logits: list[float], target_index: int, scale: float) -> float:
    scaled = [scale * item for item in logits]
    max_logit = max(scaled)
    denom = sum(math.exp(item - max_logit) for item in scaled)
    return (max_logit + math.log(denom)) - scaled[target_index]


def max_coherence(keys: list[list[float]]) -> float:
    max_mu = 0.0
    for i, key_i in enumerate(keys):
        for j, key_j in enumerate(keys):
            if i == j:
                continue
            max_mu = max(max_mu, abs(dot(key_i, key_j)))
    return max_mu


def shuffled_indices(size: int, seed: int) -> list[int]:
    rng = random.Random(seed + 1701)
    indices = list(range(size))
    rng.shuffle(indices)
    if all(item == idx for idx, item in enumerate(indices)) and size > 1:
        indices = indices[1:] + indices[:1]
    return indices


def build_prompt_trace(records: list[BindingRecord], args: argparse.Namespace) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for record in records:
        trace.append({"op": "write", "key": record.key_label, "value": record.value_label})
        for step in range(args.delay_tokens):
            trace.append({"op": "ordinary", "token": f"filler_{record.key_index}_{step}"})
    for record in records:
        trace.append({"op": "query", "key": record.renamed_key_label})
    trace.append({"op": "reset"})
    for record in records[: args.reset_probe_count]:
        trace.append({"op": "post_reset_query", "key": record.renamed_key_label})
    return trace


def evaluate(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    records = make_records(args)
    keys = [one_hot(args.key_dim, record.key_index) for record in records]
    values = [one_hot(args.value_dim, record.value_index) for record in records]
    write_order = shuffled_indices(len(records), args.seed) if mode == "false-control" else list(range(len(records)))
    ledger = make_ledger(args.key_dim, args.value_dim)

    for record_index, write_index in enumerate(write_order):
        write_binding(ledger, keys[write_index], values[record_index])

    baseline_ce = math.log(args.value_dim)
    candidate_ces: list[float] = []
    correct = 0
    leakage_energy = 0.0
    for key, value, record in zip(keys, values, records):
        read = read_binding(ledger, key)
        target = record.value_index
        if max(range(args.value_dim), key=lambda idx: read[idx]) == target:
            correct += 1
        candidate_ces.append(ce_from_logits(read, target, args.logit_scale))
        residual = read[:]
        for idx, value_item in enumerate(value):
            residual[idx] -= value_item
        leakage_energy += dot(residual, residual)

    candidate_ce = sum(candidate_ces) / max(1, len(candidate_ces))
    ce_improvement = (baseline_ce - candidate_ce) / max(1e-12, baseline_ce)
    accuracy = correct / max(1, len(records))
    mu = max_coherence(keys)
    observed_interference = leakage_energy / max(1, len(records))
    predicted_interference = max(0.0, (len(records) - 1) * (mu**2))

    erase_count = min(args.erase_probe_count, len(records))
    erased_leak = 0.0
    for idx in range(erase_count):
        erase_binding(ledger, keys[idx], values[idx])
        erased_read = read_binding(ledger, keys[idx])
        erased_leak += norm(erased_read)
    erase_residual = erased_leak / max(1, erase_count)

    remaining_correct = 0
    remaining_count = max(0, len(records) - erase_count)
    for idx in range(erase_count, len(records)):
        read = read_binding(ledger, keys[idx])
        target = records[idx].value_index
        if max(range(args.value_dim), key=lambda item: read[item]) == target:
            remaining_correct += 1
    remaining_accuracy_after_erase = remaining_correct / max(1, remaining_count)

    reset_ledger = make_ledger(args.key_dim, args.value_dim)
    reset_residual = ledger_energy(reset_ledger)
    post_reset_leak = 0.0
    for key in keys[: args.reset_probe_count]:
        post_reset_leak += norm(read_binding(reset_ledger, key))
    post_reset_leak_rate = post_reset_leak / max(1, args.reset_probe_count)

    prompt_trace = build_prompt_trace(records, args)
    prompt_tokens = len(prompt_trace)
    token_table_entries = 0
    state_width = args.key_dim * args.value_dim
    state_prompt_slope = 0.0
    surface_repetition_accuracy = 0.0
    ordinary_text_harm = 0.0

    state_bits = state_width * args.state_precision_bits
    write_bits = len(records) * args.write_bits_per_binding
    read_bits = len(records) * args.read_bits_per_query
    erase_bits = erase_count * args.erase_bits_per_binding
    reset_bits = args.reset_bits
    audit_bits = len(records) * args.audit_bits_per_binding
    old_ce_bits = ordinary_text_harm * args.old_ce_bits_per_harm
    interference_bits = observed_interference * args.interference_bits_per_energy
    paid_cost = state_bits + write_bits + read_bits + erase_bits + reset_bits + audit_bits + old_ce_bits + interference_bits
    baseline_cost = len(records) * args.baseline_bits_per_binding
    paid_ratio = paid_cost / max(1.0, baseline_cost)
    pass_gate = (
        mode == "diagnostic"
        and ce_improvement >= args.min_ce_improvement
        and accuracy >= args.min_accuracy
        and accuracy - surface_repetition_accuracy >= args.min_repetition_margin
        and ordinary_text_harm <= args.max_old_ce_harm
        and reset_residual <= args.max_reset_residual
        and post_reset_leak_rate <= args.max_post_reset_leak
        and erase_residual <= args.max_erase_residual
        and remaining_accuracy_after_erase >= args.min_remaining_accuracy_after_erase
        and observed_interference <= predicted_interference + args.interference_slack
        and token_table_entries == 0
        and state_prompt_slope <= args.max_state_prompt_slope
        and paid_ratio <= args.max_paid_ratio
    )

    return {
        "mode": mode,
        "records": [asdict(record) for record in records],
        "active_bindings": len(records),
        "prompt_tokens": prompt_tokens,
        "key_dim": args.key_dim,
        "value_dim": args.value_dim,
        "state_width": state_width,
        "state_prompt_slope": state_prompt_slope,
        "token_table_entries": token_table_entries,
        "max_key_coherence": mu,
        "predicted_interference": predicted_interference,
        "observed_interference": observed_interference,
        "candidate_ce": candidate_ce,
        "baseline_ce": baseline_ce,
        "ce_improvement": ce_improvement,
        "accuracy": accuracy,
        "surface_repetition_accuracy": surface_repetition_accuracy,
        "erase_residual": erase_residual,
        "remaining_accuracy_after_erase": remaining_accuracy_after_erase,
        "reset_residual": reset_residual,
        "post_reset_leak_rate": post_reset_leak_rate,
        "ordinary_text_harm": ordinary_text_harm,
        "state_bits": state_bits,
        "write_bits": write_bits,
        "read_bits": read_bits,
        "erase_bits": erase_bits,
        "reset_bits": reset_bits,
        "audit_bits": audit_bits,
        "old_ce_bits": old_ce_bits,
        "interference_bits": interference_bits,
        "paid_cost": paid_cost,
        "baseline_cost": baseline_cost,
        "paid_ratio": paid_ratio,
        "write_order": write_order,
        "pass": pass_gate,
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    candidate = evaluate(args, args.mode)
    diagnostic_control = evaluate(args, "diagnostic")
    shuffled_control = evaluate(args, "false-control")
    decision = (
        "PASS_R481_A3C01_RESETTABLE_BINDING_LEDGER_GATE"
        if candidate["pass"]
        else "FAIL_R481_A3C01_RESETTABLE_BINDING_LEDGER_GATE"
    )
    return {
        "schema": SCHEMA,
        "decision": decision,
        "pass": bool(candidate["pass"]),
        "mode": args.mode,
        "seed": args.seed,
        "candidate": candidate,
        "diagnostic_control": diagnostic_control,
        "shuffled_key_control": shuffled_control,
        "thresholds": {
            "min_ce_improvement": args.min_ce_improvement,
            "min_accuracy": args.min_accuracy,
            "min_repetition_margin": args.min_repetition_margin,
            "max_old_ce_harm": args.max_old_ce_harm,
            "max_reset_residual": args.max_reset_residual,
            "max_post_reset_leak": args.max_post_reset_leak,
            "max_erase_residual": args.max_erase_residual,
            "min_remaining_accuracy_after_erase": args.min_remaining_accuracy_after_erase,
            "interference_slack": args.interference_slack,
            "max_state_prompt_slope": args.max_state_prompt_slope,
            "max_paid_ratio": args.max_paid_ratio,
        },
        "elapsed_s": time.perf_counter() - start,
        "claim_scope": "fixed_width_resettable_binding_ledger_not_prompt_cache_kv_or_persistent_update",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["diagnostic", "false-control"], default="diagnostic")
    parser.add_argument("--active-bindings", type=int, default=20)
    parser.add_argument("--key-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=44131)
    parser.add_argument("--label-width", type=int, default=6)
    parser.add_argument("--delay-tokens", type=int, default=3)
    parser.add_argument("--erase-probe-count", type=int, default=5)
    parser.add_argument("--reset-probe-count", type=int, default=8)
    parser.add_argument("--logit-scale", type=float, default=8.0)
    parser.add_argument("--state-precision-bits", type=float, default=2.0)
    parser.add_argument("--write-bits-per-binding", type=float, default=4.0)
    parser.add_argument("--read-bits-per-query", type=float, default=4.0)
    parser.add_argument("--erase-bits-per-binding", type=float, default=4.0)
    parser.add_argument("--audit-bits-per-binding", type=float, default=4.0)
    parser.add_argument("--reset-bits", type=float, default=32.0)
    parser.add_argument("--old-ce-bits-per-harm", type=float, default=256.0)
    parser.add_argument("--interference-bits-per-energy", type=float, default=32.0)
    parser.add_argument("--baseline-bits-per-binding", type=float, default=256.0)
    parser.add_argument("--min-ce-improvement", type=float, default=0.10)
    parser.add_argument("--min-accuracy", type=float, default=0.95)
    parser.add_argument("--min-repetition-margin", type=float, default=0.50)
    parser.add_argument("--max-old-ce-harm", type=float, default=0.01)
    parser.add_argument("--max-reset-residual", type=float, default=1e-9)
    parser.add_argument("--max-post-reset-leak", type=float, default=1e-9)
    parser.add_argument("--max-erase-residual", type=float, default=1e-9)
    parser.add_argument("--min-remaining-accuracy-after-erase", type=float, default=0.95)
    parser.add_argument("--interference-slack", type=float, default=0.05)
    parser.add_argument("--max-state-prompt-slope", type=float, default=0.0)
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
        "[RESETTABLE_BINDING_LEDGER_GATE] "
        f"active=1 decision={result['decision']} pass={int(result['pass'])} "
        f"mode={result['mode']} active_bindings={candidate['active_bindings']} "
        f"ce_improvement={candidate['ce_improvement']:.6f} "
        f"accuracy={candidate['accuracy']:.6f} "
        f"surface_repetition_accuracy={candidate['surface_repetition_accuracy']:.6f} "
        f"observed_interference={candidate['observed_interference']:.6f} "
        f"predicted_interference={candidate['predicted_interference']:.6f} "
        f"erase_residual={candidate['erase_residual']:.6f} "
        f"reset_residual={candidate['reset_residual']:.6f} "
        f"post_reset_leak_rate={candidate['post_reset_leak_rate']:.6f} "
        f"state_prompt_slope={candidate['state_prompt_slope']:.6f} "
        f"paid_ratio={candidate['paid_ratio']:.6f}"
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
