#!/usr/bin/env python3
"""R441 A7-C01 reversible causal episode / transaction tape gate.

The gate tests whether deterministic document transactions can be represented
as a causal episode tape plus residual payload while preserving the original
surface exactly. It performs a heldout proxy CE audit: a flat text unigram model
is compared against an episode-event model plus payload residual model under
the same generated source split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA = "r441_a7_c01_reversible_episode_tape_gate.v1"
HEADER_RE = re.compile(r"^@transaction ledger:([A-Za-z0-9_-]+)\n")
LINE_RE = re.compile(r"^- (set|add|remove) ([a-z_][a-z0-9_]+) = ([A-Z0-9_-]+)\n", re.MULTILINE)
END_RE = re.compile(r"^@end\n?$", re.MULTILINE)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\w\s]", re.ASCII)


@dataclass(frozen=True)
class SourceDocument:
    doc_id: int
    family: str
    text: str


@dataclass(frozen=True)
class Episode:
    action: str
    field: str
    payload: str


@dataclass
class EpisodeTape:
    doc_id: int
    family: str
    parsed: bool
    ledger_id: str
    episodes: list[Episode]
    fallback_text: str
    original_hash: str
    reconstructed_hash: str
    roundtrip_ok: bool
    original_tokens: int
    event_tokens: list[str]
    payload_tokens: list[str]


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def make_transaction_doc(doc_id: int, rng: random.Random) -> SourceDocument:
    ledgers = ["north", "south", "inventory", "compliance", "capacity"]
    fields = [
        "customer_balance_delta",
        "warehouse_transfer_receipt",
        "regional_capacity_forecast",
        "shipment_authorization_state",
        "projected_liability_bucket",
        "inventory_reconciliation_window",
    ]
    actions = ["set", "add", "remove"]
    ledger_id = f"{rng.choice(ledgers)}-{doc_id % 11}"
    line_count = 6 + (doc_id % 4)
    lines = [f"@transaction ledger:{ledger_id}\n"]
    for row in range(line_count):
        action = actions[(doc_id + row) % len(actions)]
        field = fields[(doc_id * 3 + row * 5) % len(fields)]
        payload = f"V{(doc_id * 17 + row * 29) % 997:03d}"
        lines.append(f"- {action} {field} = {payload}\n")
    lines.append("@end\n")
    return SourceDocument(doc_id=doc_id, family="transaction_tape", text="".join(lines))


def make_raw_doc(doc_id: int, rng: random.Random) -> SourceDocument:
    phrases = [
        "This fallback paragraph is intentionally not a transaction tape.",
        "It must be charged as raw text and cannot help the episode gate.",
        "The parser should preserve it by falling back instead of guessing.",
    ]
    rng.shuffle(phrases)
    return SourceDocument(doc_id=doc_id, family="raw_fallback", text=" ".join(phrases) + "\n")


def make_corpus(docs: int, seed: int) -> list[SourceDocument]:
    rng = random.Random(seed)
    corpus: list[SourceDocument] = []
    for doc_id in range(docs):
        if doc_id % 6 == 5:
            corpus.append(make_raw_doc(doc_id, rng))
        else:
            corpus.append(make_transaction_doc(doc_id, rng))
    return corpus


def parse_document(doc: SourceDocument, audit_cost_tokens: float) -> EpisodeTape:
    del audit_cost_tokens
    header = HEADER_RE.match(doc.text)
    end = END_RE.search(doc.text)
    episodes = [Episode(action=m.group(1), field=m.group(2), payload=m.group(3)) for m in LINE_RE.finditer(doc.text)]
    parsed = header is not None and end is not None and len(episodes) > 0
    if parsed:
        ledger_id = header.group(1)
        rendered = render_tape(ledger_id, episodes)
        event_tokens = ["DOC_OPEN", f"LEDGER_{ledger_id}", *[tok for ep in episodes for tok in ("EP_OPEN", f"ACTION_{ep.action}", f"FIELD_{ep.field}", "EP_CLOSE")], "DOC_CLOSE"]
        payload_tokens = [ep.payload for ep in episodes]
        fallback_text = ""
    else:
        ledger_id = ""
        rendered = doc.text
        event_tokens = []
        payload_tokens = tokenize(doc.text)
        fallback_text = doc.text
    return EpisodeTape(
        doc_id=doc.doc_id,
        family=doc.family,
        parsed=parsed,
        ledger_id=ledger_id,
        episodes=episodes if parsed else [],
        fallback_text=fallback_text,
        original_hash=stable_hash(doc.text),
        reconstructed_hash=stable_hash(rendered),
        roundtrip_ok=rendered == doc.text,
        original_tokens=len(tokenize(doc.text)),
        event_tokens=event_tokens,
        payload_tokens=payload_tokens,
    )


def render_tape(ledger_id: str, episodes: list[Episode]) -> str:
    lines = [f"@transaction ledger:{ledger_id}\n"]
    for ep in episodes:
        lines.append(f"- {ep.action} {ep.field} = {ep.payload}\n")
    lines.append("@end\n")
    return "".join(lines)


def corrupt_first_payload(tapes: list[EpisodeTape]) -> None:
    for tape in tapes:
        if tape.episodes:
            episodes = list(tape.episodes)
            first = episodes[0]
            episodes[0] = Episode(action=first.action, field=first.field, payload=first.payload + "_BAD")
            rendered = render_tape(tape.ledger_id, episodes)
            tape.reconstructed_hash = stable_hash(rendered)
            tape.roundtrip_ok = rendered == ""
            return


def smoothed_nll(tokens: list[str], counts: Counter[str], vocab: set[str], alpha: float) -> float:
    if not tokens:
        return 0.0
    denom = sum(counts.values()) + alpha * (len(vocab) + 1)
    unk_prob = alpha / denom
    nll = 0.0
    for token in tokens:
        prob = (counts.get(token, 0) + alpha) / denom if token in vocab else unk_prob
        nll -= math.log(max(prob, 1e-12))
    return nll


def summarize(tapes: list[EpisodeTape], args: argparse.Namespace, elapsed_s: float) -> dict[str, Any]:
    split = max(1, int(len(tapes) * args.train_fraction))
    train = tapes[:split]
    heldout = tapes[split:]

    flat_train_tokens = [tok for tape in train for tok in tokenize(render_original(tape))]
    flat_heldout_tokens = [tok for tape in heldout for tok in tokenize(render_original(tape))]
    event_train_tokens = [tok for tape in train for tok in tape.event_tokens]
    event_heldout_tokens = [tok for tape in heldout for tok in tape.event_tokens]
    payload_train_tokens = [tok for tape in train for tok in tape.payload_tokens]
    payload_heldout_tokens = [tok for tape in heldout for tok in tape.payload_tokens]

    flat_counts = Counter(flat_train_tokens)
    event_counts = Counter(event_train_tokens)
    payload_counts = Counter(payload_train_tokens)
    flat_vocab = set(flat_counts)
    event_vocab = set(event_counts)
    payload_vocab = set(payload_counts)

    flat_nll = smoothed_nll(flat_heldout_tokens, flat_counts, flat_vocab, args.alpha)
    event_nll = smoothed_nll(event_heldout_tokens, event_counts, event_vocab, args.alpha)
    payload_nll = smoothed_nll(payload_heldout_tokens, payload_counts, payload_vocab, args.alpha)

    heldout_original_tokens = sum(tape.original_tokens for tape in heldout)
    parsed_original_tokens = sum(tape.original_tokens for tape in tapes if tape.parsed)
    total_original_tokens = sum(tape.original_tokens for tape in tapes)
    residual_tokens = sum(len(tape.payload_tokens) for tape in tapes if tape.parsed)
    event_tokens = sum(len(tape.event_tokens) for tape in tapes if tape.parsed)
    failed_parse_tokens = sum(tape.original_tokens for tape in tapes if not tape.parsed)
    audit_nll = args.audit_cost * max(1, sum(1 for tape in heldout if tape.parsed))
    episode_bound_nll = event_nll + payload_nll + audit_nll
    flat_ce = flat_nll / max(1, heldout_original_tokens)
    episode_ce_bound = episode_bound_nll / max(1, heldout_original_tokens)
    ce_gain = flat_ce / episode_ce_bound if episode_ce_bound > 0 else 0.0
    coverage = parsed_original_tokens / total_original_tokens if total_original_tokens > 0 else 0.0
    residual_density = residual_tokens / parsed_original_tokens if parsed_original_tokens > 0 else 1.0
    episode_token_ratio = event_tokens / parsed_original_tokens if parsed_original_tokens > 0 else 1.0
    sync_errors = [tape.doc_id for tape in tapes if not tape.roundtrip_ok]
    exact_roundtrip = not sync_errors
    pass_gate = (
        exact_roundtrip
        and coverage >= args.min_coverage
        and residual_density <= args.max_residual_density
        and ce_gain >= args.min_ce_gain
        and episode_token_ratio <= args.max_episode_token_ratio
    )
    decision = (
        "PASS_R472_A7C01_REVERSIBLE_EPISODE_TAPE_GATE"
        if pass_gate
        else "FAIL_R472_A7C01_REVERSIBLE_EPISODE_TAPE_GATE"
    )
    return {
        "schema": SCHEMA,
        "decision": decision,
        "pass": pass_gate,
        "mode": args.mode,
        "seed": args.seed,
        "docs": len(tapes),
        "train_docs": len(train),
        "heldout_docs": len(heldout),
        "parsed_docs": sum(1 for tape in tapes if tape.parsed),
        "fallback_docs": sum(1 for tape in tapes if not tape.parsed),
        "exact_roundtrip": exact_roundtrip,
        "sync_errors": len(sync_errors),
        "sync_error_doc_ids": sync_errors[:16],
        "total_original_tokens": total_original_tokens,
        "parsed_original_tokens": parsed_original_tokens,
        "failed_parse_tokens": failed_parse_tokens,
        "event_tokens": event_tokens,
        "residual_tokens": residual_tokens,
        "coverage": coverage,
        "residual_density": residual_density,
        "episode_token_ratio": episode_token_ratio,
        "heldout_original_tokens": heldout_original_tokens,
        "flat_heldout_nll": flat_nll,
        "event_heldout_nll": event_nll,
        "payload_heldout_nll": payload_nll,
        "audit_heldout_nll": audit_nll,
        "episode_bound_nll": episode_bound_nll,
        "flat_ce_proxy": flat_ce,
        "episode_ce_bound_proxy": episode_ce_bound,
        "ce_gain": ce_gain,
        "thresholds": {
            "min_coverage": args.min_coverage,
            "max_residual_density": args.max_residual_density,
            "min_ce_gain": args.min_ce_gain,
            "max_episode_token_ratio": args.max_episode_token_ratio,
        },
        "elapsed_s": elapsed_s,
        "claim_scope": "reversible_episode_dataset_object_gate_not_training_speed",
    }


def render_original(tape: EpisodeTape) -> str:
    if tape.parsed:
        return render_tape(tape.ledger_id, tape.episodes)
    return tape.fallback_text


def write_jsonl(path: Path, tapes: list[EpisodeTape]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for tape in tapes:
            row = asdict(tape)
            row["episodes"] = [asdict(ep) for ep in tape.episodes[:3]]
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["diagnostic", "false-control"], default="diagnostic")
    parser.add_argument("--docs", type=int, default=72)
    parser.add_argument("--seed", type=int, default=44171)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--audit-cost", type=float, default=0.05)
    parser.add_argument("--min-coverage", type=float, default=0.20)
    parser.add_argument("--max-residual-density", type=float, default=0.30)
    parser.add_argument("--min-ce-gain", type=float, default=1.03)
    parser.add_argument("--max-episode-token-ratio", type=float, default=1.20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tapes-jsonl", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    corpus = make_corpus(args.docs, args.seed)
    tapes = [parse_document(doc, args.audit_cost) for doc in corpus]
    if args.mode == "false-control":
        corrupt_first_payload(tapes)
    elapsed = time.perf_counter() - start
    summary = summarize(tapes, args, elapsed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.tapes_jsonl is not None:
        args.tapes_jsonl.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.tapes_jsonl, tapes)

    print(
        "[REVERSIBLE_EPISODE_TAPE_GATE] "
        f"active=1 decision={summary['decision']} pass={int(summary['pass'])} "
        f"mode={summary['mode']} docs={summary['docs']} parsed_docs={summary['parsed_docs']} "
        f"exact_roundtrip={int(summary['exact_roundtrip'])} sync_errors={summary['sync_errors']} "
        f"coverage={summary['coverage']:.6f} residual_density={summary['residual_density']:.6f} "
        f"episode_token_ratio={summary['episode_token_ratio']:.6f} "
        f"flat_ce_proxy={summary['flat_ce_proxy']:.6f} "
        f"episode_ce_bound_proxy={summary['episode_ce_bound_proxy']:.6f} "
        f"ce_gain={summary['ce_gain']:.6f}"
    )
    print(f"[REVERSIBLE_EPISODE_TAPE_OUTPUT] {args.output}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
