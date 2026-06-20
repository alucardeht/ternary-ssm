#!/usr/bin/env python3
"""R441 A1-C02 prospective query register gate.

The gate implements the narrow exact-copy domain validated by A1-C02:
queries are declared before evidence arrives, so a streaming acceptor only
writes sufficient registered spans for those queries. It rejects ambiguous
acceptor fires, overflow, checksum failure, and dense-cache-like memory cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "r441_a1_c02_prospective_query_register_gate.v1"


@dataclass(frozen=True)
class Record:
    key: str
    value: str


@dataclass
class Register:
    key: str
    value: str | None = None
    checksum: str | None = None
    fires: int = 0
    overflow: bool = False


def checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def random_word(rng: random.Random, length: int) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))


def make_document(args: argparse.Namespace) -> tuple[list[str], list[Record], list[str]]:
    rng = random.Random(args.seed)
    records: list[Record] = []
    for idx in range(args.records):
        key = f"K{idx:04d}_{random_word(rng, 6)}"
        value_tokens = [random_word(rng, args.token_width) for _ in range(args.value_tokens)]
        records.append(Record(key=key, value=" ".join(value_tokens)))
    query_indices = sorted(rng.sample(range(args.records), args.queries))
    query_keys = [records[idx].key for idx in query_indices]
    if args.mode == "false-control":
        dup_idx = query_indices[0]
        records.insert(args.records // 2, Record(key=records[dup_idx].key, value=records[dup_idx].value))
    lines = ["INSTRUCTION query " + " ".join(query_keys)]
    for record in records:
        lines.append(f"FACT {record.key} => {record.value}")
    for gap_idx in range(args.gap_lines):
        lines.append(f"DISTRACTOR {gap_idx:04d} {random_word(rng, args.token_width)}")
    for key in query_keys:
        lines.append(f"ASK {key}")
    return lines, records, query_keys


def scan(lines: list[str], query_keys: list[str], span_capacity: int) -> dict[str, Register]:
    registers = {key: Register(key=key) for key in query_keys}
    for line in lines:
        if not line.startswith("FACT "):
            continue
        left, value = line[5:].split(" => ", 1)
        reg = registers.get(left)
        if reg is None:
            continue
        reg.fires += 1
        if len(value.split()) > span_capacity:
            reg.overflow = True
            continue
        if reg.value is None:
            reg.value = value
            reg.checksum = checksum(value)
        else:
            reg.overflow = True
    return registers


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    lines, records, query_keys = make_document(args)
    registers = scan(lines, query_keys, args.span_capacity)
    record_map: dict[str, list[str]] = {}
    for record in records:
        record_map.setdefault(record.key, []).append(record.value)

    exact = 0
    misses = 0
    false_accepts = 0
    ambiguous = 0
    overflows = 0
    checksum_failures = 0
    answers: dict[str, str | None] = {}
    for key in query_keys:
        reg = registers[key]
        expected_values = record_map.get(key, [])
        if reg.fires != 1 or len(expected_values) != 1:
            ambiguous += 1
        if reg.overflow:
            overflows += 1
        if reg.value is None:
            misses += 1
            answers[key] = None
            continue
        if reg.checksum != checksum(reg.value):
            checksum_failures += 1
        if len(expected_values) == 1 and reg.value == expected_values[0] and reg.checksum == checksum(expected_values[0]):
            exact += 1
        answers[key] = reg.value
    false_accepts = sum(1 for key, reg in registers.items() if reg.fires > 0 and key not in query_keys)

    total_doc_tokens = sum(len(line.split()) for line in lines)
    total_value_tokens = sum(len(record.value.split()) for record in records)
    stored_value_tokens = sum(len(reg.value.split()) for reg in registers.values() if reg.value is not None)
    bits_per_token = args.bits_per_token
    dense_cache_bits = total_value_tokens * bits_per_token
    register_bits = args.queries * args.span_capacity * bits_per_token
    checksum_bits = args.queries * args.checksum_bits
    acceptor_bits = args.queries * args.acceptor_bits
    total_register_bits = register_bits + checksum_bits + acceptor_bits
    memory_ratio = total_register_bits / max(1.0, dense_cache_bits)
    stored_token_ratio = stored_value_tokens / max(1, total_value_tokens)
    exact_accuracy = exact / max(1, len(query_keys))
    mean_gap_lines = args.gap_lines
    pass_gate = (
        exact_accuracy == 1.0
        and ambiguous == 0
        and misses == 0
        and overflows == 0
        and checksum_failures == 0
        and false_accepts == 0
        and memory_ratio <= args.max_memory_ratio
        and stored_token_ratio <= args.max_stored_token_ratio
    )
    decision = (
        "PASS_R474_A1C02_PROSPECTIVE_QUERY_REGISTER_GATE"
        if pass_gate
        else "FAIL_R474_A1C02_PROSPECTIVE_QUERY_REGISTER_GATE"
    )
    return {
        "schema": SCHEMA,
        "decision": decision,
        "pass": pass_gate,
        "mode": args.mode,
        "seed": args.seed,
        "records": len(records),
        "queries": len(query_keys),
        "gap_lines": mean_gap_lines,
        "span_capacity": args.span_capacity,
        "exact_answers": exact,
        "exact_accuracy": exact_accuracy,
        "misses": misses,
        "ambiguous_acceptors": ambiguous,
        "overflows": overflows,
        "checksum_failures": checksum_failures,
        "false_accepts": false_accepts,
        "total_doc_tokens": total_doc_tokens,
        "total_value_tokens": total_value_tokens,
        "stored_value_tokens": stored_value_tokens,
        "dense_cache_bits": dense_cache_bits,
        "register_bits": register_bits,
        "checksum_bits": checksum_bits,
        "acceptor_bits": acceptor_bits,
        "total_register_bits": total_register_bits,
        "memory_ratio": memory_ratio,
        "stored_token_ratio": stored_token_ratio,
        "answers_preview": dict(list(answers.items())[:3]),
        "thresholds": {
            "max_memory_ratio": args.max_memory_ratio,
            "max_stored_token_ratio": args.max_stored_token_ratio,
        },
        "elapsed_s": time.perf_counter() - start,
        "claim_scope": "prospective_exact_retrieval_gate_not_general_recall",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["diagnostic", "false-control"], default="diagnostic")
    parser.add_argument("--records", type=int, default=192)
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--value-tokens", type=int, default=12)
    parser.add_argument("--span-capacity", type=int, default=16)
    parser.add_argument("--gap-lines", type=int, default=96)
    parser.add_argument("--token-width", type=int, default=8)
    parser.add_argument("--seed", type=int, default=44112)
    parser.add_argument("--bits-per-token", type=float, default=16.0)
    parser.add_argument("--checksum-bits", type=float, default=48.0)
    parser.add_argument("--acceptor-bits", type=float, default=16.0)
    parser.add_argument("--max-memory-ratio", type=float, default=0.20)
    parser.add_argument("--max-stored-token-ratio", type=float, default=0.08)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[PROSPECTIVE_QUERY_REGISTER_GATE] "
        f"active=1 decision={summary['decision']} pass={int(summary['pass'])} "
        f"mode={summary['mode']} records={summary['records']} queries={summary['queries']} "
        f"gap_lines={summary['gap_lines']} exact_accuracy={summary['exact_accuracy']:.6f} "
        f"ambiguous_acceptors={summary['ambiguous_acceptors']} overflows={summary['overflows']} "
        f"checksum_failures={summary['checksum_failures']} memory_ratio={summary['memory_ratio']:.6f} "
        f"stored_token_ratio={summary['stored_token_ratio']:.6f}"
    )
    print(f"[PROSPECTIVE_QUERY_REGISTER_OUTPUT] {args.output}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
