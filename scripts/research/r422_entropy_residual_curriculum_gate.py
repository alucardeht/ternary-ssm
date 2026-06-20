#!/usr/bin/env python3
"""R422 target-preserving entropy-residual curriculum gate.

This gate is a real-shard material probe, not a GPU training-speed benchmark.
It tests the cheapest falsifiable claim behind the R422 TO-04 finding:
entropy-residual bands should concentrate examples that buy more held-out
original-CE improvement per paid target than uniform scheduling, while the
unselected examples remain an explicit deferred CE debt.
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
SCHEMA = "r422_entropy_residual_curriculum_gate.v1"


@dataclass
class PolicyResult:
    policy: str
    selected_events: int
    selected_fraction: float
    deferred_events: int
    debt_identity_error: int
    mean_selected_score: float
    mean_population_score: float
    heldout_ce: float
    ce_gain_vs_base: float
    ce_gain_vs_uniform: float
    ce_gain_vs_random: float
    ce_gain_vs_easy: float
    gain_per_paid_event: float
    equivalent_uniform_speedup: float
    pass_gate: bool


@dataclass
class GateResult:
    schema: str
    decision: str
    dataset: str
    n_tokens: int
    vocab_size: int
    context_order: int
    base_events: int
    pool_events: int
    heldout_events: int
    paid_fraction: float
    base_heldout_ce: float
    uniform_heldout_ce: float
    random_heldout_ce: float
    easy_heldout_ce: float
    best_policy: str
    best_equivalent_uniform_speedup: float
    best_ce_gain_vs_uniform: float
    best_debt_identity_error: int
    policies: list[PolicyResult]
    min_gain_vs_uniform: float
    min_equivalent_uniform_speedup: float
    activation_log: str
    interpretation: str
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


def backoff_tuple(tokens: np.ndarray, index: int, order: int) -> tuple[int, ...]:
    if order <= 1:
        return ()
    return tuple(int(x) for x in tokens[index - order + 1 : index])


def build_counts(
    tokens: np.ndarray,
    start: int,
    end: int,
    order: int,
) -> tuple[dict[tuple[int, ...], Counter[int]], dict[tuple[int, ...], Counter[int]], Counter[int]]:
    context_counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    backoff_counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    unigram: Counter[int] = Counter()
    for i in range(max(order, start), end):
        token = int(tokens[i])
        context_counts[context_tuple(tokens, i, order)][token] += 1
        backoff_counts[backoff_tuple(tokens, i, order)][token] += 1
        unigram[token] += 1
    return context_counts, backoff_counts, unigram


def clone_counts(counts: dict[tuple[int, ...], Counter[int]]) -> dict[tuple[int, ...], Counter[int]]:
    return {ctx: Counter(counter) for ctx, counter in counts.items()}


def entropy(counter: Counter[int]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for count in counter.values():
        p = count / total
        h -= p * math.log(max(p, 1e-12))
    return h


def ce_from_counter(counter: Counter[int] | None, token: int, vocab_size: int, alpha: float) -> float:
    if not counter:
        return math.log(vocab_size)
    total = sum(counter.values())
    p = (counter.get(token, 0) + alpha) / (total + alpha * vocab_size)
    return -math.log(max(p, 1e-12))


def heldout_ce(
    tokens: np.ndarray,
    start: int,
    end: int,
    order: int,
    counts: dict[tuple[int, ...], Counter[int]],
    unigram: Counter[int],
    vocab_size: int,
    alpha: float,
) -> float:
    total = 0.0
    n = 0
    for i in range(max(order, start), end):
        ctx = context_tuple(tokens, i, order)
        token = int(tokens[i])
        counter = counts.get(ctx)
        if counter is None:
            counter = unigram
        total += ce_from_counter(counter, token, vocab_size, alpha)
        n += 1
    return total / max(1, n)


def pool_events(tokens: np.ndarray, start: int, end: int, order: int) -> list[tuple[int, tuple[int, ...], tuple[int, ...], int]]:
    events = []
    for i in range(max(order, start), end):
        events.append((i, context_tuple(tokens, i, order), backoff_tuple(tokens, i, order), int(tokens[i])))
    return events


def add_events(
    counts: dict[tuple[int, ...], Counter[int]],
    unigram: Counter[int],
    events: list[tuple[int, tuple[int, ...], tuple[int, ...], int]],
    selected_indices: np.ndarray,
) -> tuple[dict[tuple[int, ...], Counter[int]], Counter[int]]:
    out_counts = clone_counts(counts)
    out_unigram = Counter(unigram)
    for idx in selected_indices:
        _, ctx, _, token = events[int(idx)]
        out_counts.setdefault(ctx, Counter())[token] += 1
        out_unigram[token] += 1
    return out_counts, out_unigram


def score_events(
    events: list[tuple[int, tuple[int, ...], tuple[int, ...], int]],
    context_counts: dict[tuple[int, ...], Counter[int]],
    backoff_counts: dict[tuple[int, ...], Counter[int]],
    unigram: Counter[int],
    pool_context_support: Counter[tuple[int, ...]],
    vocab_size: int,
    alpha: float,
) -> dict[str, np.ndarray]:
    scores: dict[str, list[float]] = {
        "target_surprisal": [],
        "entropy_residual": [],
        "support_priced_entropy_residual": [],
        "undertrained_context_residual": [],
    }
    for _, ctx, backoff, token in events:
        ctx_counter = context_counts.get(ctx)
        backoff_counter = backoff_counts.get(backoff)
        ctx_ce = ce_from_counter(ctx_counter, token, vocab_size, alpha)
        backoff_ce = ce_from_counter(backoff_counter or unigram, token, vocab_size, alpha)
        ctx_entropy = entropy(ctx_counter or Counter())
        backoff_entropy = entropy(backoff_counter or unigram)
        support = pool_context_support[ctx]
        entropy_residual = max(0.0, ctx_entropy - backoff_entropy)
        undertrained = max(0.0, ctx_ce - backoff_ce)
        support_price = math.log1p(support)
        scores["target_surprisal"].append(ctx_ce)
        scores["entropy_residual"].append(entropy_residual)
        scores["support_priced_entropy_residual"].append((ctx_ce + entropy_residual) * support_price)
        scores["undertrained_context_residual"].append(undertrained * support_price)
    return {name: np.asarray(values, dtype=np.float64) for name, values in scores.items()}


def select_top(scores: np.ndarray, n_select: int) -> np.ndarray:
    if n_select <= 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(scores, kind="stable")
    return order[-n_select:]


def select_bottom(scores: np.ndarray, n_select: int) -> np.ndarray:
    if n_select <= 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(scores, kind="stable")
    return order[:n_select]


def evaluate_policy(
    *,
    policy: str,
    selected: np.ndarray,
    events: list[tuple[int, tuple[int, ...], tuple[int, ...], int]],
    scores: np.ndarray,
    context_counts: dict[tuple[int, ...], Counter[int]],
    unigram: Counter[int],
    tokens: np.ndarray,
    heldout_start: int,
    heldout_end: int,
    order: int,
    vocab_size: int,
    alpha: float,
    base_ce: float,
    uniform_ce: float,
    random_ce: float,
    easy_ce: float,
    min_gain_vs_uniform: float,
    min_equivalent_uniform_speedup: float,
) -> PolicyResult:
    selected_counts, selected_unigram = add_events(context_counts, unigram, events, selected)
    ce = heldout_ce(tokens, heldout_start, heldout_end, order, selected_counts, selected_unigram, vocab_size, alpha)
    pool_n = len(events)
    paid = int(selected.size)
    deferred = pool_n - paid
    debt_identity_error = (paid + deferred) - pool_n
    gain_vs_base = base_ce - ce
    uniform_gain = base_ce - uniform_ce
    gain_per_paid = gain_vs_base / max(1, paid)
    speedup = gain_vs_base / max(uniform_gain, 1e-12) if gain_vs_base > 0.0 else 0.0
    selected_score = float(np.mean(scores[selected])) if paid else 0.0
    population_score = float(np.mean(scores)) if scores.size else 0.0
    gain_vs_uniform = uniform_ce - ce
    gain_vs_random = random_ce - ce
    gain_vs_easy = easy_ce - ce
    pass_gate = (
        debt_identity_error == 0
        and gain_vs_uniform >= min_gain_vs_uniform
        and gain_vs_random >= 0.0
        and gain_vs_easy >= min_gain_vs_uniform
        and speedup >= min_equivalent_uniform_speedup
        and selected_score > population_score
    )
    return PolicyResult(
        policy=policy,
        selected_events=paid,
        selected_fraction=paid / max(1, pool_n),
        deferred_events=deferred,
        debt_identity_error=debt_identity_error,
        mean_selected_score=selected_score,
        mean_population_score=population_score,
        heldout_ce=float(ce),
        ce_gain_vs_base=float(gain_vs_base),
        ce_gain_vs_uniform=float(gain_vs_uniform),
        ce_gain_vs_random=float(gain_vs_random),
        ce_gain_vs_easy=float(gain_vs_easy),
        gain_per_paid_event=float(gain_per_paid),
        equivalent_uniform_speedup=float(speedup),
        pass_gate=pass_gate,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=120000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--context-order", type=int, default=2)
    parser.add_argument("--base-fraction", type=float, default=0.35)
    parser.add_argument("--pool-fraction", type=float, default=0.35)
    parser.add_argument("--paid-fraction", type=float, default=0.35)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42204)
    parser.add_argument("--min-gain-vs-uniform", type=float, default=0.01)
    parser.add_argument("--min-equivalent-uniform-speedup", type=float, default=1.10)
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
    if tokens.size <= args.context_order + 16:
        raise ValueError("not enough tokens for requested context order")
    if not (0.05 <= args.base_fraction <= 0.80):
        raise ValueError("--base-fraction must be in [0.05, 0.80]")
    if not (0.05 <= args.pool_fraction <= 0.80):
        raise ValueError("--pool-fraction must be in [0.05, 0.80]")
    if not (0.05 <= args.paid_fraction <= 1.0):
        raise ValueError("--paid-fraction must be in [0.05, 1.0]")

    n = int(tokens.size)
    base_end = max(args.context_order + 2, int(n * args.base_fraction))
    pool_end = min(n - 2, base_end + max(2, int(n * args.pool_fraction)))
    heldout_start = pool_end
    heldout_end = n
    if heldout_end - heldout_start <= args.context_order + 2:
        raise ValueError("split leaves too little heldout data")

    context_counts, backoff_counts, unigram = build_counts(tokens, 0, base_end, args.context_order)
    events = pool_events(tokens, base_end, pool_end, args.context_order)
    pool_context_support = Counter(ctx for _, ctx, _, _ in events)
    n_select = max(1, min(len(events), int(round(len(events) * args.paid_fraction))))
    score_map = score_events(
        events,
        context_counts,
        backoff_counts,
        unigram,
        pool_context_support,
        vocab_size,
        args.alpha,
    )

    rng = np.random.default_rng(args.seed)
    uniform_selected = np.sort(rng.choice(len(events), size=n_select, replace=False))
    base_ce = heldout_ce(tokens, heldout_start, heldout_end, args.context_order, context_counts, unigram, vocab_size, args.alpha)
    uniform_counts, uniform_unigram = add_events(context_counts, unigram, events, uniform_selected)
    uniform_ce = heldout_ce(tokens, heldout_start, heldout_end, args.context_order, uniform_counts, uniform_unigram, vocab_size, args.alpha)

    reference_scores = score_map["support_priced_entropy_residual"]
    easy_selected = select_bottom(reference_scores, n_select)
    easy_counts, easy_unigram = add_events(context_counts, unigram, events, easy_selected)
    easy_ce = heldout_ce(tokens, heldout_start, heldout_end, args.context_order, easy_counts, easy_unigram, vocab_size, args.alpha)

    random_scores = rng.random(len(events), dtype=np.float64)
    random_selected = select_top(random_scores, n_select)
    random_counts, random_unigram = add_events(context_counts, unigram, events, random_selected)
    random_ce = heldout_ce(tokens, heldout_start, heldout_end, args.context_order, random_counts, random_unigram, vocab_size, args.alpha)

    policies: list[PolicyResult] = []
    for policy, scores in score_map.items():
        selected = select_top(scores, n_select)
        policies.append(
            evaluate_policy(
                policy=policy,
                selected=selected,
                events=events,
                scores=scores,
                context_counts=context_counts,
                unigram=unigram,
                tokens=tokens,
                heldout_start=heldout_start,
                heldout_end=heldout_end,
                order=args.context_order,
                vocab_size=vocab_size,
                alpha=args.alpha,
                base_ce=base_ce,
                uniform_ce=uniform_ce,
                random_ce=random_ce,
                easy_ce=easy_ce,
                min_gain_vs_uniform=args.min_gain_vs_uniform,
                min_equivalent_uniform_speedup=args.min_equivalent_uniform_speedup,
            )
        )

    best = max(policies, key=lambda p: (p.pass_gate, p.equivalent_uniform_speedup, p.ce_gain_vs_uniform))
    decision = "PASS_ENTROPY_RESIDUAL_CURRICULUM_GATE" if best.pass_gate else "FAIL_ENTROPY_RESIDUAL_CURRICULUM_GATE"
    activation = (
        "[R422_ENTROPY_RESIDUAL_CURRICULUM_GATE] "
        f"active=1 decision={decision} best_policy={best.policy} "
        "target_preserved_by_deferred_debt=1 no_gpu=1 no_training_integration=1"
    )
    interpretation = (
        "material_gate_only: this can support or falsify the entropy-band "
        "scheduler mechanism on real token shards, but cannot close a GPU "
        "training-speed claim without a v12 sampler/training run."
    )
    result = GateResult(
        schema=SCHEMA,
        decision=decision,
        dataset=str(args.dataset),
        n_tokens=n,
        vocab_size=vocab_size,
        context_order=args.context_order,
        base_events=max(0, base_end - args.context_order),
        pool_events=len(events),
        heldout_events=max(0, heldout_end - heldout_start),
        paid_fraction=args.paid_fraction,
        base_heldout_ce=float(base_ce),
        uniform_heldout_ce=float(uniform_ce),
        random_heldout_ce=float(random_ce),
        easy_heldout_ce=float(easy_ce),
        best_policy=best.policy,
        best_equivalent_uniform_speedup=float(best.equivalent_uniform_speedup),
        best_ce_gain_vs_uniform=float(best.ce_gain_vs_uniform),
        best_debt_identity_error=best.debt_identity_error,
        policies=policies,
        min_gain_vs_uniform=args.min_gain_vs_uniform,
        min_equivalent_uniform_speedup=args.min_equivalent_uniform_speedup,
        activation_log=activation,
        interpretation=interpretation,
        elapsed_seconds=float(time.perf_counter() - start),
    )
    print(activation)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
