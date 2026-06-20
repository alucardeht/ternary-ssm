#!/usr/bin/env python3
"""Lossless macro-packet CE identifying gate.

This gate tests the R422 "Lossless Packet / Macro Original-CE Units" cluster on
real v12 token shards. It learns a reversible packet vocabulary only from a
training prefix, packetizes held-out tokens, decodes the packets back to the
original token stream, and prices packet count, decoder table, verification, and
preprocessing overhead.

This is not a tokenizer proposal by itself and not a speedup claim. It is a
cheap falsifier for whether a packet substrate can reduce paid original-token
work while preserving exact decoded original CE.
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


DATASET_MAGIC = 0x544F4B4E
DATASET_HEADER_BYTES = 16
SCHEMA = "r422_lossless_packet_ce_gate.v1"


@dataclass
class PacketGateResult:
    name: str
    decision: str
    pass_gate: bool
    token_count: int
    packet_count: int
    packet_vocab_size: int
    packet_token_mass: int
    reconstruction_exact: bool
    chain_rule_identity_error: float
    packet_count_ratio: float
    decoder_table_ratio: float
    verify_ratio: float
    preprocess_ratio: float
    priced_work_ratio: float
    average_packet_len: float
    macro_token_coverage: float
    packet_nll_per_original_token: float
    dense_unigram_nll: float


def read_v12_tokens(path: Path, max_tokens: int | None) -> list[int]:
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
    available = (len(data) - offset) // 2
    n = min(n_tokens, available)
    if max_tokens is not None:
        n = min(n, max_tokens)
    if n <= 16:
        raise ValueError("dataset does not contain enough tokens")
    return list(struct.unpack_from(f"<{n}H", data, offset))


def collect_spans(tokens: list[int], min_len: int, max_len: int) -> Counter[tuple[int, ...]]:
    counts: Counter[tuple[int, ...]] = Counter()
    n = len(tokens)
    for pos in range(n):
        limit = min(max_len, n - pos)
        for length in range(min_len, limit + 1):
            span = tuple(tokens[pos : pos + length])
            if len(set(span)) <= 1:
                continue
            counts[span] += 1
    return counts


def build_vocab(
    train_tokens: list[int],
    *,
    min_len: int,
    max_len: int,
    min_support: int,
    max_packets: int,
) -> dict[tuple[int, ...], int]:
    counts = collect_spans(train_tokens, min_len, max_len)
    scored = []
    for span, count in counts.items():
        if count < min_support:
            continue
        saved = (len(span) - 1) * (count - 1)
        if saved <= 0:
            continue
        scored.append((saved, len(span), count, span))
    scored.sort(reverse=True)
    return {span: idx for idx, (_, _, _, span) in enumerate(scored[:max_packets], start=1)}


def packetize(tokens: list[int], vocab: dict[tuple[int, ...], int], max_len: int) -> list[tuple[int, ...]]:
    packets: list[tuple[int, ...]] = []
    pos = 0
    n = len(tokens)
    while pos < n:
        chosen: tuple[int, ...] | None = None
        limit = min(max_len, n - pos)
        for length in range(limit, 1, -1):
            span = tuple(tokens[pos : pos + length])
            if span in vocab:
                chosen = span
                break
        if chosen is None:
            chosen = (tokens[pos],)
        packets.append(chosen)
        pos += len(chosen)
    return packets


def decode(packets: list[tuple[int, ...]]) -> list[int]:
    out: list[int] = []
    for packet in packets:
        out.extend(packet)
    return out


def empirical_nll(items: list[tuple[int, ...]] | list[int], train_items: list[tuple[int, ...]] | list[int]) -> float:
    counts = Counter(train_items)
    vocab = max(1, len(counts))
    denom = len(train_items) + vocab
    total = 0.0
    for item in items:
        total += -math.log((counts.get(item, 0) + 1) / denom)
    return total


def evaluate_packetizer(
    name: str,
    train_tokens: list[int],
    heldout_tokens: list[int],
    *,
    min_len: int,
    max_len: int,
    min_support: int,
    max_packets: int,
    decoder_token_cost: float,
    verify_cost: float,
    preprocess_cost: float,
    max_priced_ratio: float,
) -> PacketGateResult:
    vocab = build_vocab(
        train_tokens,
        min_len=min_len,
        max_len=max_len,
        min_support=min_support,
        max_packets=max_packets,
    )
    train_packets = packetize(train_tokens, vocab, max_len)
    heldout_packets = packetize(heldout_tokens, vocab, max_len)
    decoded = decode(heldout_packets)
    reconstruction_exact = decoded == heldout_tokens
    token_count = len(heldout_tokens)
    packet_count = len(heldout_packets)
    packet_token_mass = sum(len(packet) for packet in vocab)
    macro_token_mass = sum(len(packet) for packet in heldout_packets if len(packet) > 1)
    macro_token_coverage = macro_token_mass / token_count
    packet_count_ratio = packet_count / token_count
    decoder_table_ratio = decoder_token_cost * packet_token_mass / token_count
    verify_ratio = verify_cost * packet_count / token_count
    preprocess_ratio = preprocess_cost
    priced_work_ratio = packet_count_ratio + decoder_table_ratio + verify_ratio + preprocess_ratio
    average_packet_len = token_count / packet_count

    packet_nll = empirical_nll(heldout_packets, train_packets)
    dense_nll = empirical_nll(heldout_tokens, train_tokens)
    packet_nll_per_original_token = packet_nll / token_count
    dense_unigram_nll = dense_nll / token_count

    chain_rule_identity_error = 0.0 if reconstruction_exact else float("inf")
    pass_gate = (
        reconstruction_exact
        and chain_rule_identity_error == 0.0
        and priced_work_ratio <= max_priced_ratio
        and packet_count_ratio <= 0.80
        and macro_token_coverage >= 0.20
        and average_packet_len >= 1.20
    )
    if pass_gate:
        decision = "PASS_LOSSLESS_PACKET_CE_GATE"
    elif not reconstruction_exact:
        decision = "FAIL_RECONSTRUCTION"
    elif priced_work_ratio > max_priced_ratio:
        decision = "FAIL_PRICED_WORK_RATIO"
    elif macro_token_coverage < 0.20:
        decision = "FAIL_LOW_MACRO_COVERAGE"
    else:
        decision = "FAIL_PACKET_COUNT_REDUCTION"

    return PacketGateResult(
        name=name,
        decision=decision,
        pass_gate=pass_gate,
        token_count=token_count,
        packet_count=packet_count,
        packet_vocab_size=len(vocab),
        packet_token_mass=packet_token_mass,
        reconstruction_exact=reconstruction_exact,
        chain_rule_identity_error=chain_rule_identity_error,
        packet_count_ratio=packet_count_ratio,
        decoder_table_ratio=decoder_table_ratio,
        verify_ratio=verify_ratio,
        preprocess_ratio=preprocess_ratio,
        priced_work_ratio=priced_work_ratio,
        average_packet_len=average_packet_len,
        macro_token_coverage=macro_token_coverage,
        packet_nll_per_original_token=packet_nll_per_original_token,
        dense_unigram_nll=dense_unigram_nll,
    )


def shuffled_copy(tokens: list[int], seed: int) -> list[int]:
    rng = random.Random(seed)
    out = tokens[:]
    rng.shuffle(out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=300_000)
    parser.add_argument("--train-tokens", type=int, default=200_000)
    parser.add_argument("--heldout-tokens", type=int, default=80_000)
    parser.add_argument("--min-len", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=8)
    parser.add_argument("--min-support", type=int, default=4)
    parser.add_argument("--max-packets", type=int, default=4096)
    parser.add_argument("--decoder-token-cost", type=float, default=0.05)
    parser.add_argument("--verify-cost", type=float, default=0.02)
    parser.add_argument("--preprocess-cost", type=float, default=0.03)
    parser.add_argument("--max-priced-ratio", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42211)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.train_tokens + args.heldout_tokens > args.max_tokens:
        raise ValueError("train_tokens + heldout_tokens must be <= max_tokens")
    if args.min_len < 2 or args.max_len < args.min_len:
        raise ValueError("invalid packet length range")

    start = time.perf_counter()
    tokens = read_v12_tokens(args.dataset, args.max_tokens)
    if len(tokens) < args.train_tokens + args.heldout_tokens:
        raise ValueError(
            f"dataset has {len(tokens)} tokens, need {args.train_tokens + args.heldout_tokens}"
        )
    train = tokens[: args.train_tokens]
    heldout = tokens[args.train_tokens : args.train_tokens + args.heldout_tokens]

    main_result = evaluate_packetizer(
        "real_train_prefix",
        train,
        heldout,
        min_len=args.min_len,
        max_len=args.max_len,
        min_support=args.min_support,
        max_packets=args.max_packets,
        decoder_token_cost=args.decoder_token_cost,
        verify_cost=args.verify_cost,
        preprocess_cost=args.preprocess_cost,
        max_priced_ratio=args.max_priced_ratio,
    )
    shuffled_train = shuffled_copy(train, args.seed)
    shuffled_result = evaluate_packetizer(
        "shuffled_train_control",
        shuffled_train,
        heldout,
        min_len=args.min_len,
        max_len=args.max_len,
        min_support=args.min_support,
        max_packets=args.max_packets,
        decoder_token_cost=args.decoder_token_cost,
        verify_cost=args.verify_cost,
        preprocess_cost=args.preprocess_cost,
        max_priced_ratio=args.max_priced_ratio,
    )

    control_margin = shuffled_result.priced_work_ratio - main_result.priced_work_ratio
    final_pass = main_result.pass_gate and control_margin >= 0.10
    if final_pass:
        decision = "PASS_LOSSLESS_PACKET_CE_GATE"
    elif main_result.pass_gate:
        decision = "FAIL_SHUFFLED_PACKET_CONTROL"
    else:
        decision = main_result.decision

    elapsed = time.perf_counter() - start
    params = vars(args).copy()
    params["dataset"] = str(args.dataset)
    params["output"] = str(args.output) if args.output else None
    params["elapsed_seconds"] = elapsed
    payload = {
        "schema": SCHEMA,
        "active": 1,
        "mode": "cpu_lossless_packet_ce_identifying_gate",
        "no_training_integration": 1,
        "no_gpu": 1,
        "params": params,
        "decision": decision,
        "pass_gate": final_pass,
        "control_margin": control_margin,
        "results": {
            "real_train_prefix": asdict(main_result),
            "shuffled_train_control": asdict(shuffled_result),
        },
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(
        "[R422_LOSSLESS_PACKET_CE_GATE] "
        f"active=1 decision={decision} pass={int(final_pass)} "
        f"reconstruction_exact={int(main_result.reconstruction_exact)} "
        f"chain_rule_identity_error={main_result.chain_rule_identity_error:.6g} "
        f"packet_count_ratio={main_result.packet_count_ratio:.6f} "
        f"priced_work_ratio={main_result.priced_work_ratio:.6f} "
        f"macro_token_coverage={main_result.macro_token_coverage:.6f} "
        f"control_margin={control_margin:.6f} "
        f"packet_vocab_size={main_result.packet_vocab_size} "
        "no_training_integration=1 no_gpu=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
