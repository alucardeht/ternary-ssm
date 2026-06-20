#!/usr/bin/env python3
"""R421 Macro CE Quotient / Ledger Training Units gate.

This CPU-only gate builds predictive quotient atoms on a real token shard and
checks whether quotient classes reduce the training denominator while preserving
held-out original CE within a priced residual.

It is not a production training path. It tests the cheapest identifying claim:
contexts may share a quotient representative only when their held-out next-token
CE remains close to the finer context model and the quotient size plus residual
pricing is cheaper than the original context-event denominator.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


DATASET_MAGIC = 0x544F4B4E
DATASET_HEADER_BYTES = 16
DEFAULT_VOCAB = 32064
SCHEMA = "r421_macro_ce_quotient_gate.v1"


@dataclass
class QuotientResult:
    schema: str
    decision: str
    dataset: str
    n_tokens: int
    context_order: int
    train_events: int
    heldout_events: int
    unique_contexts: int
    quotient_classes: int
    quotient_ratio: float
    quotient_speedup_raw: float
    heldout_coverage: float
    context_ce: float
    quotient_ce: float
    random_class_ce: float
    ce_excess: float
    quotient_vs_random_ce_advantage: float
    residual_price: float
    priced_work_ratio: float
    net_speedup_estimate: float
    min_class_support: int
    signature_top_m: int
    entropy_bins: int
    activation_log: str
    elapsed_seconds: float


def read_v12_tokens(path: Path, max_tokens: int | None) -> tuple[np.ndarray, int]:
    data = path.read_bytes()
    offset = 0
    n_tokens = len(data) // 2
    vocab_size = DEFAULT_VOCAB
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
    if raw.size:
        vocab_size = max(DEFAULT_VOCAB, int(raw.max()) + 1)
    return raw, vocab_size


def context_tuple(tokens: np.ndarray, index: int, order: int) -> tuple[int, ...]:
    return tuple(int(x) for x in tokens[index - order : index])


def entropy(counter: Counter[int]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for count in counter.values():
        p = count / total
        h -= p * math.log(max(p, 1e-12))
    return h


def signature(counter: Counter[int], top_m: int, entropy_bins: int) -> tuple[tuple[int, ...], int]:
    total = max(1, sum(counter.values()))
    top = tuple(token for token, _ in counter.most_common(top_m))
    h_norm = entropy(counter) / max(math.log(max(total, 2)), 1e-12)
    bin_id = min(entropy_bins - 1, max(0, int(h_norm * entropy_bins)))
    return top, bin_id


def ce_from_counter(counter: Counter[int], token: int, vocab_size: int, alpha: float) -> float:
    total = sum(counter.values())
    p = (counter.get(token, 0) + alpha) / (total + alpha * vocab_size)
    return -math.log(max(p, 1e-12))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=100000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--context-order", type=int, default=2)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--min-class-support", type=int, default=4)
    parser.add_argument("--signature-top-m", type=int, default=2)
    parser.add_argument("--entropy-bins", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42140)
    parser.add_argument("--max-ce-excess", type=float, default=0.08)
    parser.add_argument("--min-random-control-advantage", type=float, default=0.02)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.90)
    parser.add_argument("--min-heldout-coverage", type=float, default=0.70)
    parser.add_argument("--residual-price-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    tokens, vocab_size = read_v12_tokens(args.dataset, None)
    if args.offset:
        tokens = tokens[args.offset :]
    if args.max_tokens:
        tokens = tokens[: args.max_tokens]
    if tokens.size <= args.context_order + 4:
        raise ValueError("not enough tokens for requested context order")

    train_end = max(args.context_order + 2, int(tokens.size * args.train_fraction))
    context_counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    for i in range(args.context_order, train_end):
        context_counts[context_tuple(tokens, i, args.context_order)][int(tokens[i])] += 1

    eligible_contexts = {
        ctx: counts
        for ctx, counts in context_counts.items()
        if sum(counts.values()) >= args.min_class_support
    }
    class_counts: dict[tuple[tuple[int, ...], int], Counter[int]] = defaultdict(Counter)
    context_to_class: dict[tuple[int, ...], tuple[tuple[int, ...], int]] = {}
    for ctx, counts in eligible_contexts.items():
        sig = signature(counts, args.signature_top_m, args.entropy_bins)
        context_to_class[ctx] = sig
        class_counts[sig].update(counts)

    rng = np.random.default_rng(args.seed)
    class_keys = list(class_counts.keys())
    random_class_counts: dict[int, Counter[int]] = defaultdict(Counter)
    context_to_random_class: dict[tuple[int, ...], int] = {}
    if class_keys:
        for ctx, counts in eligible_contexts.items():
            rid = int(rng.integers(0, len(class_keys)))
            context_to_random_class[ctx] = rid
            random_class_counts[rid].update(counts)

    context_ce_sum = 0.0
    quotient_ce_sum = 0.0
    random_ce_sum = 0.0
    covered = 0
    heldout = 0
    for i in range(max(args.context_order, train_end), tokens.size):
        ctx = context_tuple(tokens, i, args.context_order)
        token = int(tokens[i])
        heldout += 1
        counts = eligible_contexts.get(ctx)
        sig = context_to_class.get(ctx)
        rid = context_to_random_class.get(ctx)
        if counts is None or sig is None or rid is None:
            continue
        covered += 1
        context_ce_sum += ce_from_counter(counts, token, vocab_size, args.alpha)
        quotient_ce_sum += ce_from_counter(class_counts[sig], token, vocab_size, args.alpha)
        random_ce_sum += ce_from_counter(random_class_counts[rid], token, vocab_size, args.alpha)

    coverage = covered / heldout if heldout else 0.0
    if covered:
        context_ce = context_ce_sum / covered
        quotient_ce = quotient_ce_sum / covered
        random_ce = random_ce_sum / covered
    else:
        context_ce = float("inf")
        quotient_ce = float("inf")
        random_ce = float("inf")
    ce_excess = quotient_ce - context_ce if math.isfinite(quotient_ce) else float("inf")
    random_advantage = random_ce - quotient_ce if math.isfinite(random_ce) else float("-inf")
    quotient_ratio = len(class_counts) / max(1, len(eligible_contexts))
    residual_price = max(0.0, ce_excess) * args.residual_price_scale
    priced_work_ratio = quotient_ratio + residual_price + (1.0 - coverage)
    net_speedup = 1.0 / max(priced_work_ratio, 1e-12)
    pass_gate = (
        coverage >= args.min_heldout_coverage
        and ce_excess <= args.max_ce_excess
        and random_advantage >= args.min_random_control_advantage
        and priced_work_ratio <= args.max_priced_work_ratio
        and len(class_counts) < len(eligible_contexts)
    )
    decision = "PASS_MACRO_CE_QUOTIENT_GATE" if pass_gate else "FAIL_MACRO_CE_QUOTIENT_GATE"
    activation = (
        "[R421_MACRO_CE_QUOTIENT_GATE] "
        f"active=1 decision={decision} no_training_integration=1 no_gpu=1"
    )
    result = QuotientResult(
        schema=SCHEMA,
        decision=decision,
        dataset=str(args.dataset),
        n_tokens=int(tokens.size),
        context_order=args.context_order,
        train_events=max(0, train_end - args.context_order),
        heldout_events=heldout,
        unique_contexts=len(eligible_contexts),
        quotient_classes=len(class_counts),
        quotient_ratio=float(quotient_ratio),
        quotient_speedup_raw=float(1.0 / max(quotient_ratio, 1e-12)),
        heldout_coverage=float(coverage),
        context_ce=float(context_ce),
        quotient_ce=float(quotient_ce),
        random_class_ce=float(random_ce),
        ce_excess=float(ce_excess),
        quotient_vs_random_ce_advantage=float(random_advantage),
        residual_price=float(residual_price),
        priced_work_ratio=float(priced_work_ratio),
        net_speedup_estimate=float(net_speedup),
        min_class_support=args.min_class_support,
        signature_top_m=args.signature_top_m,
        entropy_bins=args.entropy_bins,
        activation_log=activation,
        elapsed_seconds=float(time.perf_counter() - start),
    )
    print(activation)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
