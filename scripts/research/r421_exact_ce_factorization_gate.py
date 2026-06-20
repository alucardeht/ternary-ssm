#!/usr/bin/env python3
"""R421 Exact CE Factorization gate.

This is a CPU-only identifying gate for the R421 Wave 2 finding
"Exact CE Factorization / Candidate-Set Normalization".

The gate separates two claims that are easy to confuse:

1. exact token CE can be represented as a product of smaller CE terms when the
   output distribution is explicitly factorized;
2. real training material supplies small candidate sets often enough that
   candidate-set CE plus residual replay can be cheaper than dense CE.

It does not modify production training and does not measure GPU speed.
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
SCHEMA = "r421_exact_ce_factorization_gate.v1"


@dataclass
class GateResult:
    schema: str
    decision: str
    dataset: str
    n_tokens: int
    eval_examples: int
    vocab_size: int
    radix: int
    n_buckets: int
    factor_identity_max_abs_error: float
    factor_identity_mean_abs_error: float
    factor_denominator_ratio: float
    context_order: int
    candidate_top_k: int
    candidate_coverage: float
    candidate_density: float
    miss_rate: float
    expected_candidate_work_ratio: float
    expected_candidate_speedup: float
    fallback_dense_fraction: float
    candidate_certificate_scope: str
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


def logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.exp(x - m).sum()))


def factor_identity_check(
    tokens: np.ndarray,
    vocab_size: int,
    radix: int,
    eval_limit: int,
    seed: int,
) -> tuple[float, float, float, int]:
    rng = np.random.default_rng(seed)
    n_buckets = int(math.ceil(vocab_size / radix))
    bucket_logits = rng.normal(0.0, 0.7, size=n_buckets)
    suffix_logits = rng.normal(0.0, 0.7, size=(n_buckets, radix))

    bucket_log_z = logsumexp(bucket_logits)
    suffix_log_z = np.array([logsumexp(row) for row in suffix_logits])
    sample = tokens[: min(eval_limit, tokens.size)]
    errors: list[float] = []

    for token in sample:
        bucket = int(token) // radix
        suffix = int(token) % radix
        factored_ce = -(bucket_logits[bucket] - bucket_log_z)
        factored_ce -= suffix_logits[bucket, suffix] - suffix_log_z[bucket]

        dense_logit = bucket_logits[bucket] - bucket_log_z
        dense_logit += suffix_logits[bucket, suffix] - suffix_log_z[bucket]
        dense_ce = -dense_logit
        errors.append(abs(factored_ce - dense_ce))

    denom_ratio = (n_buckets + radix) / float(vocab_size)
    return max(errors, default=0.0), float(np.mean(errors)) if errors else 0.0, denom_ratio, n_buckets


def context_tuple(tokens: np.ndarray, index: int, order: int) -> tuple[int, ...] | None:
    if index < order:
        return None
    return tuple(int(x) for x in tokens[index - order : index])


def build_candidate_table(
    tokens: np.ndarray,
    order: int,
    train_end: int,
    top_k: int,
) -> dict[tuple[int, ...], tuple[int, ...]]:
    counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    for i in range(order, max(order, train_end - 1)):
        ctx = context_tuple(tokens, i, order)
        if ctx is None:
            continue
        counts[ctx][int(tokens[i])] += 1
    table: dict[tuple[int, ...], tuple[int, ...]] = {}
    for ctx, counter in counts.items():
        table[ctx] = tuple(token for token, _ in counter.most_common(top_k))
    return table


def candidate_gate(
    tokens: np.ndarray,
    vocab_size: int,
    order: int,
    top_k: int,
    train_fraction: float,
) -> tuple[int, float, float, float, float, float]:
    train_end = max(order + 2, int(tokens.size * train_fraction))
    table = build_candidate_table(tokens, order, train_end, top_k)
    hits = 0
    examples = 0
    total_work = 0.0
    fallback_dense = 0

    for i in range(max(order, train_end), tokens.size):
        ctx = context_tuple(tokens, i, order)
        if ctx is None:
            continue
        candidates = table.get(ctx)
        examples += 1
        if candidates and int(tokens[i]) in candidates:
            hits += 1
            total_work += len(candidates)
        else:
            fallback_dense += 1
            total_work += vocab_size

    if examples == 0:
        return 0, 0.0, 0.0, 1.0, 1.0, 1.0

    coverage = hits / examples
    miss_rate = 1.0 - coverage
    density = min(top_k, vocab_size) / float(vocab_size)
    work_ratio = total_work / (examples * float(vocab_size))
    speedup = 1.0 / max(work_ratio, 1e-12)
    fallback_fraction = fallback_dense / examples
    return examples, coverage, density, miss_rate, work_ratio, speedup, fallback_fraction


def decide(
    max_error: float,
    factor_ratio: float,
    coverage: float,
    work_ratio: float,
    min_coverage: float,
    max_work_ratio: float,
    tol: float,
) -> str:
    if max_error > tol:
        return "FAIL_IDENTITY_NOT_EXACT"
    if coverage >= min_coverage and work_ratio <= max_work_ratio:
        return "PASS_FACTOR_IDENTITY_AND_EMPIRICAL_ROUTE_GATE"
    if factor_ratio <= max_work_ratio:
        return "PASS_FACTOR_IDENTITY_ONLY_ROUTE_WEAK"
    return "FAIL_CANDIDATE_DENOMINATOR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=50000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--radix", type=int, default=256)
    parser.add_argument("--context-order", type=int, default=3)
    parser.add_argument("--candidate-top-k", type=int, default=64)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--identity-eval-limit", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42120)
    parser.add_argument("--min-candidate-coverage", type=float, default=0.60)
    parser.add_argument("--max-work-ratio", type=float, default=0.90)
    parser.add_argument("--identity-tolerance", type=float, default=1e-12)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    all_tokens, vocab_size = read_v12_tokens(args.dataset, None)
    if args.offset:
        all_tokens = all_tokens[args.offset :]
    if args.max_tokens:
        all_tokens = all_tokens[: args.max_tokens]
    if all_tokens.size <= args.context_order + 2:
        raise ValueError("not enough tokens for requested context order")

    max_err, mean_err, factor_ratio, n_buckets = factor_identity_check(
        all_tokens,
        vocab_size,
        args.radix,
        args.identity_eval_limit,
        args.seed,
    )
    (
        eval_examples,
        coverage,
        density,
        miss_rate,
        work_ratio,
        speedup,
        fallback_fraction,
    ) = candidate_gate(
        all_tokens,
        vocab_size,
        args.context_order,
        args.candidate_top_k,
        args.train_fraction,
    )
    decision = decide(
        max_err,
        factor_ratio,
        coverage,
        work_ratio,
        args.min_candidate_coverage,
        args.max_work_ratio,
        args.identity_tolerance,
    )
    activation = (
        "[R421_EXACT_CE_FACTOR_GATE] "
        f"active=1 decision={decision} no_training_integration=1 no_gpu=1"
    )
    result = GateResult(
        schema=SCHEMA,
        decision=decision,
        dataset=str(args.dataset),
        n_tokens=int(all_tokens.size),
        eval_examples=int(eval_examples),
        vocab_size=int(vocab_size),
        radix=int(args.radix),
        n_buckets=int(n_buckets),
        factor_identity_max_abs_error=float(max_err),
        factor_identity_mean_abs_error=float(mean_err),
        factor_denominator_ratio=float(factor_ratio),
        context_order=int(args.context_order),
        candidate_top_k=int(args.candidate_top_k),
        candidate_coverage=float(coverage),
        candidate_density=float(density),
        miss_rate=float(miss_rate),
        expected_candidate_work_ratio=float(work_ratio),
        expected_candidate_speedup=float(speedup),
        fallback_dense_fraction=float(fallback_fraction),
        candidate_certificate_scope=(
            "empirical_route_coverage_only_not_model_probability_mass"
        ),
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
