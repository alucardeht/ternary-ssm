#!/usr/bin/env python3
"""Reversible state ledger identifying gate.

This is the bounded implementation gate for the R422 cluster
"Reversible State Ledger And Proof-Carrying Memory".

The probe builds held-out variable-binding streams where useful memory is a
latent state obligation, not retrieval of the original token sequence. It then
compares a proof-carrying reversible ledger against same-cell state, bit-budgeted
larger state, a larger hidden-state control, and a full dictionary/retrieval
upper bound.

This is not production training, not a GPU benchmark, and not a speedup claim.
It is a cheap falsifier for whether compact certified memory can beat ordinary
state capacity on long-dependency binding tasks without accepting false memory
certificates.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path


MASK64 = (1 << 64) - 1
SCHEMA = "r422_reversible_state_ledger_gate.v1"


def mix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def bucket_index(key_fp: int, seed: int, n_buckets: int) -> int:
    return mix64(key_fp ^ (seed * 0x9E3779B1)) % n_buckets


def answer_ce(predicted: int | None, target: int, value_count: int) -> float:
    if predicted is None:
        return math.log(value_count)
    if predicted == target:
        return -math.log(0.9)
    return -math.log(0.1 / max(1, value_count - 1))


class DirectState:
    def __init__(self, slots: int, *, blind: bool) -> None:
        self.slots = slots
        self.blind = blind
        self.cells: list[tuple[int, int] | int | None] = [None] * slots

    def add(self, key_fp: int, value: int) -> None:
        idx = bucket_index(key_fp, 1, self.slots)
        if self.blind:
            self.cells[idx] = value
        else:
            self.cells[idx] = (key_fp, value)

    def query(self, key_fp: int) -> int | None:
        item = self.cells[bucket_index(key_fp, 1, self.slots)]
        if item is None:
            return None
        if self.blind:
            return int(item)
        stored_key, value = item
        return value if stored_key == key_fp else None


class ReversibleLedger:
    """Signed parity ledger with local certificates.

    Each bucket stores count, key fingerprint xor, value xor, and checksum xor.
    The full key is never stored. Peeling reconstructs certified entries only
    when the bucket equations prove a singleton.
    """

    def __init__(self, buckets: int, hashes: int) -> None:
        self.buckets = buckets
        self.hashes = hashes
        self.rows: list[list[int]] = [[0, 0, 0, 0] for _ in range(buckets)]

    @staticmethod
    def checksum(key_fp: int, value: int) -> int:
        return mix64((key_fp << 16) ^ value) & 0xFFFFFFFF

    def indexes(self, key_fp: int) -> list[int]:
        return [bucket_index(key_fp, seed + 7, self.buckets) for seed in range(self.hashes)]

    def add(self, key_fp: int, value: int) -> None:
        checksum = self.checksum(key_fp, value)
        for idx in self.indexes(key_fp):
            row = self.rows[idx]
            row[0] += 1
            row[1] ^= key_fp
            row[2] ^= value
            row[3] ^= checksum

    def corrupt_random_bucket(self, rng: random.Random) -> None:
        idx = rng.randrange(self.buckets)
        field = rng.randrange(1, 4)
        self.rows[idx][field] ^= mix64(rng.randrange(1 << 30)) & 0xFFFF

    def peel(self) -> tuple[dict[int, int], int, int]:
        rows = [row[:] for row in self.rows]
        queue: deque[int] = deque()
        for idx, row in enumerate(rows):
            if self._is_singleton(row):
                queue.append(idx)

        decoded: dict[int, int] = {}
        peel_steps = 0
        while queue:
            idx = queue.popleft()
            row = rows[idx]
            if not self._is_singleton(row):
                continue
            count, key_fp, value, checksum = row
            if key_fp in decoded:
                continue
            decoded[key_fp] = value
            peel_steps += 1
            for peer in self.indexes(key_fp):
                peer_row = rows[peer]
                peer_row[0] -= count
                peer_row[1] ^= key_fp
                peer_row[2] ^= value
                peer_row[3] ^= checksum
                if self._is_singleton(peer_row):
                    queue.append(peer)

        remaining = sum(1 for row in rows if row != [0, 0, 0, 0])
        return decoded, remaining, peel_steps

    def _is_singleton(self, row: list[int]) -> bool:
        count, key_fp, value, checksum = row
        return abs(count) == 1 and checksum == self.checksum(key_fp, value)


@dataclass
class ArmStats:
    name: str
    ce: float
    accuracy: float
    abstain_rate: float
    false_accept_rate: float
    memory_bits: float
    memory_ratio_vs_ledger: float


@dataclass
class EpisodeStats:
    seed: int
    active_bindings: int
    ledger_decode_fraction: float
    ledger_remaining_buckets: int
    ledger_peel_steps: int
    corrupt_false_accept_rate: float
    arms: list[ArmStats]


def key_fingerprint(surface_key: int, key_fp_bits: int) -> int:
    mask = (1 << key_fp_bits) - 1
    return mix64(surface_key) & mask


def run_episode(
    *,
    seed: int,
    buckets: int,
    hashes: int,
    load: float,
    value_count: int,
    key_space: int,
    key_bits: int,
    key_fp_bits: int,
    value_bits: int,
    checksum_bits: int,
    queries_per_binding: int,
    false_queries_per_binding: int,
) -> EpisodeStats:
    rng = random.Random(seed)
    active_bindings = max(1, int(round(load * buckets)))
    surface_keys = rng.sample(range(key_space), active_bindings)
    values = {key: rng.randrange(value_count) for key in surface_keys}
    fingerprints = {key: key_fingerprint(key, key_fp_bits) for key in surface_keys}

    ledger = ReversibleLedger(buckets, hashes)
    blind = DirectState(buckets, blind=True)
    keyed_same = DirectState(buckets, blind=False)

    ledger_bucket_bits = 16 + key_fp_bits + value_bits + checksum_bits
    keyed_slot_bits = 1 + key_fp_bits + value_bits
    ledger_bits = buckets * ledger_bucket_bits
    keyed_bitmatched_slots = max(1, int(ledger_bits // keyed_slot_bits))
    keyed_bitmatched = DirectState(keyed_bitmatched_slots, blind=False)
    keyed_larger = DirectState(buckets * 4, blind=False)

    for key in surface_keys:
        value = values[key]
        fp = fingerprints[key]
        ledger.add(fp, value)
        blind.add(fp, value)
        keyed_same.add(fp, value)
        keyed_bitmatched.add(fp, value)
        keyed_larger.add(fp, value)

    decoded, remaining, peel_steps = ledger.peel()

    query_keys = [rng.choice(surface_keys) for _ in range(active_bindings * queries_per_binding)]
    absent_keys: list[int] = []
    while len(absent_keys) < active_bindings * false_queries_per_binding:
        candidate = rng.randrange(key_space, key_space * 2)
        absent_keys.append(candidate)

    arms = {
        "blind_same_cell_state": {
            "query": lambda fp: blind.query(fp),
            "memory_bits": buckets * (1 + value_bits),
        },
        "keyed_same_cell_state": {
            "query": lambda fp: keyed_same.query(fp),
            "memory_bits": buckets * keyed_slot_bits,
        },
        "keyed_bitmatched_larger_state": {
            "query": lambda fp: keyed_bitmatched.query(fp),
            "memory_bits": keyed_bitmatched_slots * keyed_slot_bits,
        },
        "keyed_4x_larger_state": {
            "query": lambda fp: keyed_larger.query(fp),
            "memory_bits": buckets * 4 * keyed_slot_bits,
        },
        "full_dictionary_retrieval": {
            "query": lambda fp: dictionary_query(fp, fingerprints, values),
            "memory_bits": active_bindings * (key_bits + value_bits) * 2,
        },
        "reversible_state_ledger": {
            "query": lambda fp: decoded.get(fp),
            "memory_bits": ledger_bits,
        },
    }

    arm_stats: list[ArmStats] = []
    for name, arm in arms.items():
        ce_sum = 0.0
        correct = 0
        abstain = 0
        false_accept = 0
        for key in query_keys:
            fp = fingerprints[key]
            target = values[key]
            predicted = arm["query"](fp)
            ce_sum += answer_ce(predicted, target, value_count)
            correct += int(predicted == target)
            abstain += int(predicted is None)
        for key in absent_keys:
            fp = key_fingerprint(key, key_fp_bits)
            false_accept += int(arm["query"](fp) is not None)

        n_queries = len(query_keys)
        n_false = len(absent_keys)
        memory_bits = float(arm["memory_bits"])
        arm_stats.append(
            ArmStats(
                name=name,
                ce=ce_sum / n_queries,
                accuracy=correct / n_queries,
                abstain_rate=abstain / n_queries,
                false_accept_rate=false_accept / n_false,
                memory_bits=memory_bits,
                memory_ratio_vs_ledger=memory_bits / float(ledger_bits),
            )
        )

    corrupt_false_accepts = 0
    corrupt_trials = 64
    for trial in range(corrupt_trials):
        corrupted = ReversibleLedger(buckets, hashes)
        corrupted.rows = [row[:] for row in ledger.rows]
        corrupted.corrupt_random_bucket(random.Random(seed + 1000 + trial))
        corrupted_decoded, _, _ = corrupted.peel()
        for key in absent_keys[: max(1, len(absent_keys) // 8)]:
            fp = key_fingerprint(key, key_fp_bits)
            corrupt_false_accepts += int(fp in corrupted_decoded)
    corrupt_den = corrupt_trials * max(1, len(absent_keys) // 8)

    return EpisodeStats(
        seed=seed,
        active_bindings=active_bindings,
        ledger_decode_fraction=len(decoded) / active_bindings,
        ledger_remaining_buckets=remaining,
        ledger_peel_steps=peel_steps,
        corrupt_false_accept_rate=corrupt_false_accepts / corrupt_den,
        arms=arm_stats,
    )


def dictionary_query(fp: int, fingerprints: dict[int, int], values: dict[int, int]) -> int | None:
    for surface_key, candidate_fp in fingerprints.items():
        if candidate_fp == fp:
            return values[surface_key]
    return None


def aggregate(episodes: list[EpisodeStats]) -> dict:
    by_arm: dict[str, list[ArmStats]] = {}
    for episode in episodes:
        for arm in episode.arms:
            by_arm.setdefault(arm.name, []).append(arm)

    arms = {}
    for name, rows in by_arm.items():
        arms[name] = {
            "ce": sum(row.ce for row in rows) / len(rows),
            "accuracy": sum(row.accuracy for row in rows) / len(rows),
            "abstain_rate": sum(row.abstain_rate for row in rows) / len(rows),
            "false_accept_rate": sum(row.false_accept_rate for row in rows) / len(rows),
            "memory_bits": sum(row.memory_bits for row in rows) / len(rows),
            "memory_ratio_vs_ledger": sum(row.memory_ratio_vs_ledger for row in rows) / len(rows),
        }

    ledger = arms["reversible_state_ledger"]
    keyed_same = arms["keyed_same_cell_state"]
    bitmatched = arms["keyed_bitmatched_larger_state"]
    larger = arms["keyed_4x_larger_state"]
    dictionary = arms["full_dictionary_retrieval"]

    decode_fraction = sum(ep.ledger_decode_fraction for ep in episodes) / len(episodes)
    remaining_buckets = sum(ep.ledger_remaining_buckets for ep in episodes) / len(episodes)
    corrupt_false = sum(ep.corrupt_false_accept_rate for ep in episodes) / len(episodes)

    ce_gain_vs_keyed_same = keyed_same["ce"] - ledger["ce"]
    ce_gain_vs_bitmatched = bitmatched["ce"] - ledger["ce"]
    ce_gain_vs_larger = larger["ce"] - ledger["ce"]
    dictionary_memory_ratio = dictionary["memory_bits"] / ledger["memory_bits"]

    pass_gate = (
        decode_fraction >= 0.98
        and ledger["false_accept_rate"] <= 0.005
        and corrupt_false <= 0.005
        and ce_gain_vs_keyed_same >= 0.20
        and ce_gain_vs_bitmatched >= 0.10
        and ce_gain_vs_larger >= 0.0
        and dictionary_memory_ratio >= 1.50
    )

    if pass_gate:
        decision = "PASS_REVERSIBLE_STATE_LEDGER_GATE"
    elif dictionary_memory_ratio < 1.50:
        decision = "FAIL_HIDDEN_RETRIEVAL_MEMORY_CONTROL"
    elif ce_gain_vs_bitmatched < 0.10:
        decision = "FAIL_BITMATCHED_STATE_CONTROL"
    elif ce_gain_vs_larger < 0.0:
        decision = "FAIL_LARGER_STATE_CONTROL"
    elif decode_fraction < 0.98:
        decision = "FAIL_UNSTABLE_REVERSIBLE_DECODE"
    elif ledger["false_accept_rate"] > 0.005 or corrupt_false > 0.005:
        decision = "FAIL_FALSE_CERTIFICATE_CONTROL"
    else:
        decision = "FAIL_WEAK_LEDGER_ADVANTAGE"

    return {
        "decision": decision,
        "pass_gate": pass_gate,
        "decode_fraction": decode_fraction,
        "remaining_buckets": remaining_buckets,
        "corrupt_false_accept_rate": corrupt_false,
        "ce_gain_vs_keyed_same": ce_gain_vs_keyed_same,
        "ce_gain_vs_bitmatched": ce_gain_vs_bitmatched,
        "ce_gain_vs_larger": ce_gain_vs_larger,
        "dictionary_memory_ratio_vs_ledger": dictionary_memory_ratio,
        "arms": arms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buckets", type=int, default=512)
    parser.add_argument("--hashes", type=int, default=3)
    parser.add_argument("--load", type=float, default=0.62)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42203)
    parser.add_argument("--value-count", type=int, default=128)
    parser.add_argument("--key-space", type=int, default=1_000_000_000)
    parser.add_argument("--key-bits", type=int, default=128)
    parser.add_argument("--key-fp-bits", type=int, default=32)
    parser.add_argument("--value-bits", type=int, default=8)
    parser.add_argument("--checksum-bits", type=int, default=32)
    parser.add_argument("--queries-per-binding", type=int, default=3)
    parser.add_argument("--false-queries-per-binding", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not (0.05 <= args.load <= 0.90):
        raise ValueError("--load must be in [0.05, 0.90]")
    if args.hashes < 2:
        raise ValueError("--hashes must be at least 2")
    if args.key_fp_bits < 16:
        raise ValueError("--key-fp-bits must be at least 16")
    if args.episodes < 3:
        raise ValueError("--episodes must be at least 3")

    start = time.perf_counter()
    episodes = [
        run_episode(
            seed=args.seed + idx,
            buckets=args.buckets,
            hashes=args.hashes,
            load=args.load,
            value_count=args.value_count,
            key_space=args.key_space,
            key_bits=args.key_bits,
            key_fp_bits=args.key_fp_bits,
            value_bits=args.value_bits,
            checksum_bits=args.checksum_bits,
            queries_per_binding=args.queries_per_binding,
            false_queries_per_binding=args.false_queries_per_binding,
        )
        for idx in range(args.episodes)
    ]
    result = aggregate(episodes)
    elapsed = time.perf_counter() - start

    params = vars(args).copy()
    params["output"] = str(args.output) if args.output else None
    params["elapsed_seconds"] = elapsed

    payload = {
        "schema": SCHEMA,
        "active": 1,
        "mode": "cpu_reversible_state_ledger_identifying_gate",
        "observer_only": 1,
        "no_training_integration": 1,
        "no_gpu": 1,
        "params": params,
        "result": result,
        "episodes": [asdict(ep) for ep in episodes],
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(
        "[R422_REVERSIBLE_STATE_LEDGER_GATE] "
        f"active=1 decision={result['decision']} "
        f"pass={int(result['pass_gate'])} "
        f"decode_fraction={result['decode_fraction']:.6f} "
        f"ce_gain_vs_keyed_same={result['ce_gain_vs_keyed_same']:.6f} "
        f"ce_gain_vs_bitmatched={result['ce_gain_vs_bitmatched']:.6f} "
        f"ce_gain_vs_larger={result['ce_gain_vs_larger']:.6f} "
        f"false_accept={result['arms']['reversible_state_ledger']['false_accept_rate']:.6f} "
        f"corrupt_false_accept={result['corrupt_false_accept_rate']:.6f} "
        f"dictionary_memory_ratio={result['dictionary_memory_ratio_vs_ledger']:.6f} "
        "no_training_integration=1 no_gpu=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
