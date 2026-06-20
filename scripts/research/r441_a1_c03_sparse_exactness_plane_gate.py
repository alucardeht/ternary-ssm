#!/usr/bin/env python3
"""R441 A1-C03 sparse certified exactness-plane gate.

This is an oracle-frontier gate for one concrete sparse exactness plane:
a finite-field keyed slot ledger with rank and collision certificates. It does
not claim a trained promotion policy. It asks whether accepted sparse late-copy
obligations can be represented exactly, with zero false accepts, O(active)
memory, explicit overflow/rejection, and priced verifier/fallback cost below
the registered frontier threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import string
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA = "r441_a1_c03_sparse_exactness_plane_gate.v1"
PRIME = 65537


@dataclass(frozen=True)
class Fact:
    index: int
    key: str
    value_tokens: tuple[str, ...]
    promoted: bool


@dataclass(frozen=True)
class PlaneEntry:
    key: str
    slot: int
    key_tag: str
    checksum: str
    value_tokens: tuple[str, ...]


def digest(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def random_word(rng: random.Random, length: int) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def slot_for_key(key: str, slot_count: int) -> int:
    return stable_int("slot:" + key) % slot_count


def code_vector(key: str, dim: int) -> list[int]:
    seed = stable_int("code:" + key)
    rng = random.Random(seed)
    return [rng.randrange(1, PRIME) for _ in range(dim)]


def matrix_rank_mod(matrix: list[list[int]], prime: int) -> int:
    rows = [row[:] for row in matrix if any(v % prime for v in row)]
    if not rows:
        return 0
    rank = 0
    col = 0
    width = len(rows[0])
    while rank < len(rows) and col < width:
        pivot = None
        for r in range(rank, len(rows)):
            if rows[r][col] % prime:
                pivot = r
                break
        if pivot is None:
            col += 1
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inv = pow(rows[rank][col] % prime, -1, prime)
        rows[rank] = [(v * inv) % prime for v in rows[rank]]
        for r in range(len(rows)):
            if r == rank:
                continue
            factor = rows[r][col] % prime
            if factor:
                rows[r] = [(rows[r][c] - factor * rows[rank][c]) % prime for c in range(width)]
        rank += 1
        col += 1
    return rank


def make_facts(args: argparse.Namespace) -> tuple[list[Fact], list[int]]:
    rng = random.Random(args.seed)
    facts: list[Fact] = []
    selected: list[int] = []
    used_slots: set[int] = set()
    for idx in range(args.records):
        key = f"K{idx:04d}_{random_word(rng, args.key_width)}"
        value = tuple(random_word(rng, args.token_width) for _ in range(args.value_tokens))
        slot = slot_for_key(key, args.slot_count)
        promoted = False
        if len(selected) < args.promoted and slot not in used_slots:
            promoted = True
            selected.append(idx)
            used_slots.add(slot)
        facts.append(Fact(index=idx, key=key, value_tokens=value, promoted=promoted))
    if len(selected) < args.promoted:
        raise ValueError("could not select enough collision-free promoted facts")
    return facts, selected


def build_plane(facts: list[Fact], args: argparse.Namespace) -> tuple[dict[int, PlaneEntry], list[int]]:
    plane: dict[int, PlaneEntry] = {}
    collision_slots: list[int] = []
    for fact in facts:
        if not fact.promoted:
            continue
        slot = slot_for_key(fact.key, args.slot_count)
        if args.mode == "false-control" and fact.index == next(idx for idx, f in enumerate(facts) if f.promoted):
            slot = slot_for_key(next(f.key for f in facts if f.promoted and f.index != fact.index), args.slot_count)
        entry = PlaneEntry(
            key=fact.key,
            slot=slot,
            key_tag=digest(fact.key, args.key_tag_hex),
            checksum=digest(" ".join(fact.value_tokens), args.checksum_hex),
            value_tokens=fact.value_tokens,
        )
        if slot in plane:
            collision_slots.append(slot)
        plane[slot] = entry
    return plane, collision_slots


def make_queries(facts: list[Fact], args: argparse.Namespace) -> list[int]:
    rng = random.Random(args.seed + 17)
    promoted_indices = [fact.index for fact in facts if fact.promoted]
    unpromoted_indices = [fact.index for fact in facts if not fact.promoted]
    promoted_query_count = min(len(promoted_indices), max(1, int(args.queries * args.promoted_query_fraction)))
    rejected_query_count = args.queries - promoted_query_count
    queries = rng.sample(promoted_indices, promoted_query_count)
    queries.extend(rng.sample(unpromoted_indices, rejected_query_count))
    rng.shuffle(queries)
    return queries


def query_plane(fact: Fact, plane: dict[int, PlaneEntry], args: argparse.Namespace) -> tuple[str, bool]:
    slot = slot_for_key(fact.key, args.slot_count)
    entry = plane.get(slot)
    if entry is None:
        return "reject", False
    if entry.key_tag != digest(fact.key, args.key_tag_hex):
        return "reject", False
    if entry.checksum != digest(" ".join(entry.value_tokens), args.checksum_hex):
        return "reject", False
    exact = entry.value_tokens == fact.value_tokens
    return ("accept" if exact else "false_accept"), exact


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    facts, selected = make_facts(args)
    plane, collision_slots = build_plane(facts, args)
    queries = make_queries(facts, args)
    active_entries = [entry for entry in plane.values()]
    rank_matrix = [code_vector(entry.key, len(active_entries)) for entry in active_entries]
    rank = matrix_rank_mod(rank_matrix, PRIME)
    rank_pass = rank == len(active_entries)

    accepted_exact = 0
    false_accepts = 0
    rejected = 0
    queried_promoted = 0
    for query_idx in queries:
        fact = facts[query_idx]
        if fact.promoted:
            queried_promoted += 1
        status, exact = query_plane(fact, plane, args)
        if status == "accept" and exact:
            accepted_exact += 1
        elif status == "false_accept":
            false_accepts += 1
        else:
            rejected += 1

    current_failures = len(queries)
    accepted_exact_mass = accepted_exact / max(1, current_failures)
    false_accept_rate = false_accepts / max(1, len(queries))
    overflow_count = max(0, args.promoted - args.capacity)
    collision_failures = len(collision_slots)
    value_bits_per_fact = args.value_tokens * args.bits_per_token
    dense_context_bits = args.records * value_bits_per_fact
    slot_bits = math.ceil(math.log2(args.slot_count))
    entry_bits = slot_bits + args.key_tag_bits + value_bits_per_fact + args.checksum_bits
    plane_bits = len(active_entries) * entry_bits
    rank_certificate_bits = len(active_entries) * len(active_entries)
    verify_bits = len(queries) * args.verify_bits_per_query
    memory_bits = plane_bits + rank_certificate_bits
    memory_ratio = memory_bits / max(1.0, dense_context_bits)
    baseline_failure_bits = current_failures * value_bits_per_fact * args.failure_repair_multiplier
    rejected_repair_bits = rejected * value_bits_per_fact * args.failure_repair_multiplier
    candidate_total_bits = memory_bits + verify_bits + rejected_repair_bits
    total_cost_ratio = candidate_total_bits / max(1.0, baseline_failure_bits)
    queried_promoted_fraction = queried_promoted / max(1, len(queries))
    no_dense_storage = len(active_entries) <= args.capacity and len(active_entries) <= int(args.records * args.max_promotion_density)
    pass_gate = (
        accepted_exact_mass >= args.min_accepted_exact_mass
        and false_accept_rate == 0.0
        and overflow_count == 0
        and collision_failures == 0
        and rank_pass
        and no_dense_storage
        and memory_ratio <= args.max_memory_ratio
        and total_cost_ratio < args.max_total_cost_ratio
    )
    decision = (
        "PASS_R476_A1C03_SPARSE_EXACTNESS_PLANE_GATE"
        if pass_gate
        else "FAIL_R476_A1C03_SPARSE_EXACTNESS_PLANE_GATE"
    )
    return {
        "schema": SCHEMA,
        "decision": decision,
        "pass": pass_gate,
        "mode": args.mode,
        "seed": args.seed,
        "records": args.records,
        "queries": len(queries),
        "promoted": args.promoted,
        "capacity": args.capacity,
        "slot_count": args.slot_count,
        "active_entries": len(active_entries),
        "accepted_exact": accepted_exact,
        "accepted_exact_mass": accepted_exact_mass,
        "queried_promoted": queried_promoted,
        "queried_promoted_fraction": queried_promoted_fraction,
        "rejected_queries": rejected,
        "false_accepts": false_accepts,
        "false_accept_rate": false_accept_rate,
        "overflow_count": overflow_count,
        "collision_failures": collision_failures,
        "collision_slots": collision_slots[:8],
        "rank": rank,
        "rank_pass": rank_pass,
        "no_dense_storage": no_dense_storage,
        "dense_context_bits": dense_context_bits,
        "memory_bits": memory_bits,
        "plane_bits": plane_bits,
        "rank_certificate_bits": rank_certificate_bits,
        "verify_bits": verify_bits,
        "baseline_failure_bits": baseline_failure_bits,
        "rejected_repair_bits": rejected_repair_bits,
        "candidate_total_bits": candidate_total_bits,
        "memory_ratio": memory_ratio,
        "total_cost_ratio": total_cost_ratio,
        "thresholds": {
            "min_accepted_exact_mass": args.min_accepted_exact_mass,
            "max_memory_ratio": args.max_memory_ratio,
            "max_total_cost_ratio": args.max_total_cost_ratio,
            "max_promotion_density": args.max_promotion_density,
        },
        "selected_preview": selected[:8],
        "elapsed_s": time.perf_counter() - start,
        "claim_scope": "oracle_sparse_exactness_plane_frontier_not_trained_policy",
    }


def write_plane(path: Path, facts: list[Fact], plane: dict[int, PlaneEntry]) -> None:
    promoted = {fact.key for fact in facts if fact.promoted}
    with path.open("w", encoding="utf-8") as handle:
        for entry in plane.values():
            row = asdict(entry)
            row["value_tokens"] = list(entry.value_tokens)
            row["promoted"] = entry.key in promoted
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["diagnostic", "false-control"], default="diagnostic")
    parser.add_argument("--records", type=int, default=256)
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--promoted", type=int, default=12)
    parser.add_argument("--capacity", type=int, default=12)
    parser.add_argument("--slot-count", type=int, default=64)
    parser.add_argument("--value-tokens", type=int, default=8)
    parser.add_argument("--token-width", type=int, default=8)
    parser.add_argument("--key-width", type=int, default=6)
    parser.add_argument("--seed", type=int, default=44113)
    parser.add_argument("--bits-per-token", type=float, default=16.0)
    parser.add_argument("--key-tag-bits", type=float, default=40.0)
    parser.add_argument("--checksum-bits", type=float, default=48.0)
    parser.add_argument("--verify-bits-per-query", type=float, default=16.0)
    parser.add_argument("--failure-repair-multiplier", type=float, default=4.0)
    parser.add_argument("--promoted-query-fraction", type=float, default=0.30)
    parser.add_argument("--min-accepted-exact-mass", type=float, default=0.10)
    parser.add_argument("--max-memory-ratio", type=float, default=0.15)
    parser.add_argument("--max-total-cost-ratio", type=float, default=0.90)
    parser.add_argument("--max-promotion-density", type=float, default=0.08)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plane-jsonl", type=Path)
    args = parser.parse_args()
    args.key_tag_hex = max(1, int(args.key_tag_bits // 4))
    args.checksum_hex = max(1, int(args.checksum_bits // 4))
    return args


def main() -> int:
    args = parse_args()
    summary = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.plane_jsonl is not None:
        facts, _ = make_facts(args)
        plane, _ = build_plane(facts, args)
        args.plane_jsonl.parent.mkdir(parents=True, exist_ok=True)
        write_plane(args.plane_jsonl, facts, plane)
    print(
        "[SPARSE_EXACTNESS_PLANE_GATE] "
        f"active=1 decision={summary['decision']} pass={int(summary['pass'])} "
        f"mode={summary['mode']} records={summary['records']} queries={summary['queries']} "
        f"active_entries={summary['active_entries']} accepted_exact_mass={summary['accepted_exact_mass']:.6f} "
        f"false_accept_rate={summary['false_accept_rate']:.6f} collision_failures={summary['collision_failures']} "
        f"rank_pass={int(summary['rank_pass'])} memory_ratio={summary['memory_ratio']:.6f} "
        f"total_cost_ratio={summary['total_cost_ratio']:.6f}"
    )
    print(f"[SPARSE_EXACTNESS_PLANE_OUTPUT] {args.output}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
