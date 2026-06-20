#!/usr/bin/env python3
"""R441 A7-C02 predictive sufficient-state transition quotient gate.

This gate builds deterministic transcript-derived cells sigma_t and measures
whether their heldout next-token receipts preserve original target CE after
explicit residualizing of payload/hard tokens. It is a dataset-object gate, not
a training-speed result.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "r441_a7_c02_predictive_transition_quotient_gate.v1"


def make_stream(docs: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    regimes = ["REG_ALPHA", "REG_BETA", "REG_GAMMA", "REG_DELTA"]
    fields_by_regime = {
        "REG_ALPHA": ["ITEM_BALANCE", "ITEM_LEDGER", "ITEM_MARGIN"],
        "REG_BETA": ["ITEM_ROUTE", "ITEM_CAPACITY", "ITEM_WINDOW"],
        "REG_GAMMA": ["ITEM_AUTH", "ITEM_SCOPE", "ITEM_POLICY"],
        "REG_DELTA": ["ITEM_BATCH", "ITEM_QUEUE", "ITEM_RETRY"],
    }
    ops_by_regime = {
        "REG_ALPHA": ["OP_ADD", "OP_SET"],
        "REG_BETA": ["OP_SHIFT", "OP_HOLD"],
        "REG_GAMMA": ["OP_GRANT", "OP_REVOKE"],
        "REG_DELTA": ["OP_PUSH", "OP_POP"],
    }
    docs_out: list[list[str]] = []
    for doc_id in range(docs):
        tokens = ["DOC_OPEN"]
        for step in range(10 + doc_id % 5):
            regime = regimes[(doc_id + step) % len(regimes)]
            if rng.random() < 0.08:
                regime = rng.choice(regimes)
            field = fields_by_regime[regime][(doc_id + step * 2) % 3]
            op = ops_by_regime[regime][(doc_id * 3 + step) % 2]
            value = f"VAL_{(doc_id * 37 + step * 19) % 997:03d}"
            tokens.extend([regime, field, op, value, "SEP"])
        tokens.append("DOC_CLOSE")
        docs_out.append(tokens)
    return docs_out


def token_class(token: str) -> str:
    if token.startswith("REG_"):
        return "REG"
    if token.startswith("ITEM_"):
        return "ITEM"
    if token.startswith("OP_"):
        return "OP"
    if token.startswith("VAL_"):
        return "VAL"
    return token


def sigma_for(prefix: list[str], collapse_regime: bool) -> str:
    if not prefix:
        return "START"
    active_regime = "NO_REG"
    phase = "START"
    for token in reversed(prefix):
        if token.startswith("REG_"):
            active_regime = "REG_ANY" if collapse_regime else token
            break
    prev = prefix[-1]
    cls = token_class(prev)
    if cls == "DOC_OPEN" or prev == "DOC_OPEN":
        phase = "EXPECT_REG"
    elif cls == "REG":
        phase = "EXPECT_ITEM"
    elif cls == "ITEM":
        phase = "EXPECT_OP"
    elif cls == "OP":
        phase = "EXPECT_VALUE"
    elif cls == "VAL":
        phase = "EXPECT_SEP"
    elif prev == "SEP":
        phase = "EXPECT_REG_OR_CLOSE"
    else:
        phase = f"AFTER_{cls}"
    return f"{active_regime}|{phase}"


def prev_key(prefix: list[str]) -> str:
    return prefix[-1] if prefix else "START"


def collect_examples(streams: list[list[str]], collapse_regime: bool) -> list[tuple[str, str, str, str]]:
    examples: list[tuple[str, str, str, str]] = []
    for stream in streams:
        prefix: list[str] = []
        for token in stream:
            examples.append((sigma_for(prefix, collapse_regime), prev_key(prefix), token_class(token), token))
            prefix.append(token)
    return examples


def counts_by_key(examples: list[tuple[str, str, str, str]], key_index: int) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for example in examples:
        counts[example[key_index]][example[3]] += 1
    return dict(counts)


def nll_token(token: str, counts: Counter[str], vocab: set[str], alpha: float) -> float:
    denom = sum(counts.values()) + alpha * (len(vocab) + 1)
    if denom <= 0:
        return -math.log(1e-12)
    prob = ((counts.get(token, 0) + alpha) / denom) if token in vocab else (alpha / denom)
    return -math.log(max(prob, 1e-12))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    streams = make_stream(args.docs, args.seed)
    split = max(1, int(len(streams) * args.train_fraction))
    train_streams = streams[:split]
    heldout_streams = streams[split:]

    train_examples = collect_examples(train_streams, collapse_regime=args.mode == "false-control")
    heldout_examples = collect_examples(heldout_streams, collapse_regime=args.mode == "false-control")
    fine_train_examples = collect_examples(train_streams, collapse_regime=False)
    fine_heldout_examples = collect_examples(heldout_streams, collapse_regime=False)

    sigma_counts = counts_by_key(train_examples, 0)
    prev_counts = counts_by_key(train_examples, 1)
    fine_counts = counts_by_key(fine_train_examples, 0)
    payload_counts = Counter(ex[3] for ex in train_examples if ex[2] == "VAL")
    global_counts = Counter(ex[3] for ex in train_examples)
    vocab = set(global_counts)
    payload_vocab = set(payload_counts)

    sigma_nll = 0.0
    prev_nll = 0.0
    fine_nll = 0.0
    payload_nll = 0.0
    covered = 0
    residual = 0
    low_count = 0
    hard_misses = 0

    for idx, ex in enumerate(heldout_examples):
        sigma_key, prev, cls, token = ex
        fine_key = fine_heldout_examples[idx][0]
        sigma_count = sum(sigma_counts.get(sigma_key, Counter()).values())
        force_residual = cls == "VAL" or sigma_count < args.min_cell_count
        prev_nll += nll_token(token, prev_counts.get(prev, Counter()), vocab, args.alpha)
        if force_residual:
            residual += 1
            if cls == "VAL":
                payload_nll += nll_token(token, payload_counts, payload_vocab, args.alpha)
            else:
                payload_nll += nll_token(token, global_counts, vocab, args.alpha)
                low_count += 1
            continue
        covered += 1
        sigma_nll += nll_token(token, sigma_counts.get(sigma_key, Counter()), vocab, args.alpha)
        fine_nll += nll_token(token, fine_counts.get(fine_key, Counter()), vocab, args.alpha)
        if token not in sigma_counts.get(sigma_key, Counter()):
            hard_misses += 1

    total = max(1, len(heldout_examples))
    audit_nll = args.audit_cost * max(1, len(sigma_counts))
    quotient_bound_nll = sigma_nll + payload_nll + audit_nll
    prev_ce = prev_nll / total
    quotient_ce = quotient_bound_nll / total
    ce_gain = prev_ce / quotient_ce if quotient_ce > 0 else 0.0
    coverage = covered / total
    residual_mass = residual / total
    kl_proxy = max(0.0, (sigma_nll - fine_nll) / max(1, covered))
    hard_miss_rate = hard_misses / max(1, covered)
    pass_gate = (
        coverage >= args.min_coverage
        and residual_mass <= args.max_residual_mass
        and kl_proxy <= args.max_kl_proxy
        and ce_gain >= args.min_ce_gain
        and hard_miss_rate <= args.max_hard_miss_rate
    )
    decision = (
        "PASS_R473_A7C02_PREDICTIVE_TRANSITION_QUOTIENT_GATE"
        if pass_gate
        else "FAIL_R473_A7C02_PREDICTIVE_TRANSITION_QUOTIENT_GATE"
    )
    return {
        "schema": SCHEMA,
        "decision": decision,
        "pass": pass_gate,
        "mode": args.mode,
        "seed": args.seed,
        "docs": args.docs,
        "train_docs": len(train_streams),
        "heldout_docs": len(heldout_streams),
        "train_examples": len(train_examples),
        "heldout_examples": len(heldout_examples),
        "cells": len(sigma_counts),
        "covered_tokens": covered,
        "residual_tokens": residual,
        "low_count_residual_tokens": low_count,
        "coverage": coverage,
        "residual_mass": residual_mass,
        "sigma_nll": sigma_nll,
        "payload_nll": payload_nll,
        "audit_nll": audit_nll,
        "quotient_bound_nll": quotient_bound_nll,
        "prev_prefix_nll": prev_nll,
        "fine_sigma_nll": fine_nll,
        "prev_prefix_ce_proxy": prev_ce,
        "quotient_ce_bound_proxy": quotient_ce,
        "ce_gain": ce_gain,
        "kl_proxy": kl_proxy,
        "hard_miss_rate": hard_miss_rate,
        "thresholds": {
            "min_coverage": args.min_coverage,
            "max_residual_mass": args.max_residual_mass,
            "max_kl_proxy": args.max_kl_proxy,
            "min_ce_gain": args.min_ce_gain,
            "max_hard_miss_rate": args.max_hard_miss_rate,
        },
        "claim_scope": "predictive_transition_dataset_object_gate_not_training_speed",
    }


def write_receipts(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["diagnostic", "false-control"], default="diagnostic")
    parser.add_argument("--docs", type=int, default=96)
    parser.add_argument("--seed", type=int, default=44172)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--min-cell-count", type=int, default=8)
    parser.add_argument("--audit-cost", type=float, default=0.02)
    parser.add_argument("--min-coverage", type=float, default=0.45)
    parser.add_argument("--max-residual-mass", type=float, default=0.28)
    parser.add_argument("--max-kl-proxy", type=float, default=0.05)
    parser.add_argument("--min-ce-gain", type=float, default=1.03)
    parser.add_argument("--max-hard-miss-rate", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    summary = evaluate(args)
    summary["elapsed_s"] = time.perf_counter() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_receipts(args.output, summary)
    print(
        "[PREDICTIVE_TRANSITION_QUOTIENT_GATE] "
        f"active=1 decision={summary['decision']} pass={int(summary['pass'])} "
        f"mode={summary['mode']} docs={summary['docs']} cells={summary['cells']} "
        f"coverage={summary['coverage']:.6f} residual_mass={summary['residual_mass']:.6f} "
        f"kl_proxy={summary['kl_proxy']:.6f} hard_miss_rate={summary['hard_miss_rate']:.6f} "
        f"prev_prefix_ce_proxy={summary['prev_prefix_ce_proxy']:.6f} "
        f"quotient_ce_bound_proxy={summary['quotient_ce_bound_proxy']:.6f} "
        f"ce_gain={summary['ce_gain']:.6f}"
    )
    print(f"[PREDICTIVE_TRANSITION_QUOTIENT_OUTPUT] {args.output}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
