#!/usr/bin/env python3
"""R421 Reversible Reference / Dictionary gate.

Builds a prefix-only dictionary from a train split, greedily expands exact
dictionary entries on held-out tokens, and prices residual tokens plus decoder
storage. A random same-size dictionary control checks whether coverage comes
from real repeated material rather than address leakage or generic compression.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


DATASET_MAGIC = 0x544F4B4E
DATASET_HEADER_BYTES = 16
DEFAULT_VOCAB = 32064
SCHEMA = "r421_reversible_reference_gate.v1"


@dataclass
class ReferenceResult:
    schema: str
    decision: str
    dataset: str
    n_tokens: int
    train_tokens: int
    heldout_tokens: int
    entry_len: int
    min_support: int
    entries: int
    heldout_coverage: float
    random_control_coverage: float
    coverage_advantage: float
    macro_events: int
    residual_tokens: int
    decoder_cost_tokens: float
    priced_work_ratio: float
    net_speedup_estimate: float
    activation_log: str
    elapsed_seconds: float


def read_v12_tokens(path: Path, max_tokens: int | None) -> np.ndarray:
    data = path.read_bytes()
    offset = 0
    n_tokens = len(data) // 2
    if len(data) >= DATASET_HEADER_BYTES:
        magic, version, header_tokens = struct.unpack_from("<IIQ", data, 0)
        if magic == DATASET_MAGIC:
            if version != 1:
                raise ValueError(f"unsupported dataset version: {version}")
            offset = DATASET_HEADER_BYTES
            n_tokens = int(header_tokens)
    raw = np.frombuffer(data, dtype="<u2", count=n_tokens, offset=offset).astype(np.int64)
    if max_tokens is not None:
        raw = raw[:max_tokens]
    return raw


def build_counts(tokens: np.ndarray, entry_len: int) -> Counter[tuple[int, ...]]:
    counts: Counter[tuple[int, ...]] = Counter()
    if tokens.size < entry_len:
        return counts
    for i in range(0, tokens.size - entry_len + 1):
        counts[tuple(int(x) for x in tokens[i : i + entry_len])] += 1
    return counts


def choose_dictionary(
    counts: Counter[tuple[int, ...]],
    entry_len: int,
    min_support: int,
    max_entries: int,
) -> set[tuple[int, ...]]:
    scored = []
    for entry, count in counts.items():
        if count >= min_support:
            saved = (entry_len - 1) * count - entry_len
            if saved > 0:
                scored.append((saved, count, entry))
    scored.sort(reverse=True)
    return {entry for _, _, entry in scored[:max_entries]}


def choose_random_dictionary(
    counts: Counter[tuple[int, ...]],
    size: int,
    seed: int,
) -> set[tuple[int, ...]]:
    rng = random.Random(seed)
    entries = list(counts.keys())
    rng.shuffle(entries)
    return set(entries[:size])


def cover(tokens: np.ndarray, dictionary: set[tuple[int, ...]], entry_len: int) -> tuple[int, int, int]:
    i = 0
    covered = 0
    macro_events = 0
    residual = 0
    n = tokens.size
    while i < n:
        if i + entry_len <= n and tuple(int(x) for x in tokens[i : i + entry_len]) in dictionary:
            covered += entry_len
            macro_events += 1
            i += entry_len
        else:
            residual += 1
            i += 1
    return covered, macro_events, residual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=200000)
    parser.add_argument("--entry-len", type=int, default=8)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--max-entries", type=int, default=4096)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--decoder-cost-scale", type=float, default=0.05)
    parser.add_argument("--min-coverage", type=float, default=0.10)
    parser.add_argument("--min-control-advantage", type=float, default=0.05)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42170)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    tokens = read_v12_tokens(args.dataset, args.max_tokens)
    train_end = max(args.entry_len, int(tokens.size * args.train_fraction))
    train = tokens[:train_end]
    heldout = tokens[train_end:]
    counts = build_counts(train, args.entry_len)
    dictionary = choose_dictionary(counts, args.entry_len, args.min_support, args.max_entries)
    random_dictionary = choose_random_dictionary(counts, len(dictionary), args.seed)

    covered, macro_events, residual = cover(heldout, dictionary, args.entry_len)
    random_covered, _, _ = cover(heldout, random_dictionary, args.entry_len)
    heldout_n = max(1, int(heldout.size))
    coverage = covered / heldout_n
    random_coverage = random_covered / heldout_n
    decoder_cost = len(dictionary) * args.entry_len * args.decoder_cost_scale
    priced_work = macro_events + residual + decoder_cost
    priced_ratio = priced_work / heldout_n
    speedup = 1.0 / max(priced_ratio, 1e-12)
    advantage = coverage - random_coverage
    passed = (
        coverage >= args.min_coverage
        and advantage >= args.min_control_advantage
        and priced_ratio <= args.max_priced_work_ratio
    )
    decision = "PASS_REVERSIBLE_REFERENCE_GATE" if passed else "FAIL_REVERSIBLE_REFERENCE_GATE"
    activation = (
        "[R421_REVERSIBLE_REFERENCE_GATE] "
        f"active=1 decision={decision} no_training_integration=1 no_gpu=1"
    )
    result = ReferenceResult(
        schema=SCHEMA,
        decision=decision,
        dataset=str(args.dataset),
        n_tokens=int(tokens.size),
        train_tokens=int(train.size),
        heldout_tokens=int(heldout.size),
        entry_len=args.entry_len,
        min_support=args.min_support,
        entries=len(dictionary),
        heldout_coverage=float(coverage),
        random_control_coverage=float(random_coverage),
        coverage_advantage=float(advantage),
        macro_events=macro_events,
        residual_tokens=residual,
        decoder_cost_tokens=float(decoder_cost),
        priced_work_ratio=float(priced_ratio),
        net_speedup_estimate=float(speedup),
        activation_log=activation,
        elapsed_seconds=time.perf_counter() - start,
    )
    print(activation)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
