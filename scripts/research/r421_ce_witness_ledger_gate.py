#!/usr/bin/env python3
"""R421 CE Witness Ledger and resident proof-object gate.

This CPU-only support gate tests whether original-CE evidence can be carried as
a resident proof object instead of being reconstructed by a separate scoring or
verification pass.

The gate deliberately separates three ledger shapes:

1. CE-only witness rows: compact and exact for loss verification, but unable to
   certify the readout gradient.
2. Full-probability witness rows: exact for loss and gradient, but essentially
   as large as the materialized score surface.
3. Top-k plus residual witness rows: compact enough to be interesting, but only
   useful if the reconstructed gradient residual is small.

It does not modify production training and does not measure GPU speed.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


DATASET_MAGIC = 0x544F4B4E
DATASET_HEADER_BYTES = 16
DEFAULT_VOCAB = 32064
SCHEMA = "r421_ce_witness_ledger_gate.v1"


@dataclass
class LedgerRow:
    name: str
    ce_residual_ratio: float
    grad_residual_ratio: float
    byte_ratio: float
    decode_work_ratio: float
    stale_ce_residual_ratio: float
    priced_support_ratio: float
    estimated_support_speedup: float
    pass_full_gate: bool
    pass_ce_only_support: bool
    note: str


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
    available = (len(data) - offset) // 2
    n = min(n_tokens, available)
    if max_tokens is not None:
        n = min(n, max_tokens)
    if n <= 4:
        raise ValueError("dataset does not contain enough tokens")
    return np.frombuffer(data, dtype="<u2", count=n, offset=offset).astype(np.int64)


def choose_window(tokens: np.ndarray, seq_len: int, offset: int, vocab: int) -> np.ndarray:
    if tokens.shape[0] < seq_len + 1:
        raise ValueError("not enough tokens for requested sequence length")
    max_start = tokens.shape[0] - seq_len - 1
    start = min(max(offset, 0), max_start)
    window = tokens[start : start + seq_len + 1].copy()
    if window.max(initial=0) >= vocab:
        window %= vocab
    return window


def init_params(vocab: int, hidden: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    embed = rng.normal(0.0, 0.18, size=(vocab, hidden)).astype(np.float64)
    recurrent = rng.normal(0.0, 0.22 / math.sqrt(hidden), size=(hidden, hidden)).astype(np.float64)
    spectral = float(np.linalg.norm(recurrent, ord=2))
    if spectral > 0.92:
        recurrent *= 0.92 / spectral
    readout = rng.normal(0.0, 0.12 / math.sqrt(hidden), size=(vocab, hidden)).astype(np.float64)
    return {"embed": embed, "recurrent": recurrent, "readout": readout}


def states_for(window: np.ndarray, params: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    embed = params["embed"]
    recurrent = params["recurrent"]
    inputs = window[:-1]
    targets = window[1:]
    hidden = recurrent.shape[0]
    states = np.zeros((inputs.shape[0], hidden), dtype=np.float64)
    h = np.zeros(hidden, dtype=np.float64)
    for i, tok in enumerate(inputs):
        h = np.tanh(embed[tok] + recurrent @ h)
        states[i] = h
    return states, targets


def materialized_ce_probs_grad(
    states: np.ndarray,
    targets: np.ndarray,
    readout: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    logits = states @ readout.T
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z, dtype=np.float64)
    probs = exp / exp.sum(axis=1, keepdims=True)
    ce = float(-np.log(np.maximum(probs[np.arange(targets.shape[0]), targets], 1e-300)).mean())
    dlogits = probs.copy()
    dlogits[np.arange(targets.shape[0]), targets] -= 1.0
    dlogits /= float(targets.shape[0])
    grad = dlogits.T @ states
    return ce, probs, grad


def ce_from_probs(probs: np.ndarray, targets: np.ndarray) -> float:
    return float(-np.log(np.maximum(probs[np.arange(targets.shape[0]), targets], 1e-300)).mean())


def grad_from_probs(probs: np.ndarray, targets: np.ndarray, states: np.ndarray) -> np.ndarray:
    dlogits = probs.copy()
    dlogits[np.arange(targets.shape[0]), targets] -= 1.0
    dlogits /= float(targets.shape[0])
    return dlogits.T @ states


def topk_residual_probs(probs: np.ndarray, targets: np.ndarray, top_k: int) -> np.ndarray:
    rows, vocab = probs.shape
    approx = np.zeros_like(probs)
    k = min(max(top_k, 1), vocab)
    for row in range(rows):
        selected = np.argpartition(probs[row], -k)[-k:]
        target = int(targets[row])
        if target not in selected:
            selected[0] = target
        selected = np.unique(selected)
        selected_mass = float(probs[row, selected].sum())
        omitted = vocab - selected.shape[0]
        approx[row, selected] = probs[row, selected]
        if omitted > 0:
            approx[row, approx[row] == 0.0] = max(0.0, 1.0 - selected_mass) / omitted
    return approx


def perturb_readout(readout: np.ndarray, seed: int, scale: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    delta = rng.normal(0.0, scale / math.sqrt(readout.shape[1]), size=readout.shape)
    return readout + delta


def norm_ratio(delta: np.ndarray, baseline: np.ndarray) -> float:
    return float(np.linalg.norm(delta) / max(float(np.linalg.norm(baseline)), 1e-30))


def price_row(
    byte_ratio: float,
    decode_ratio: float,
    stale_ratio: float,
    memory_price: float,
    stale_repair_price: float,
    build_overhead: float,
) -> tuple[float, float]:
    priced = build_overhead + decode_ratio + memory_price * byte_ratio + stale_repair_price * stale_ratio
    speedup = 1.0 / max(priced, 1e-30)
    return float(priced), float(speedup)


def evaluate_window(
    tokens: np.ndarray,
    params: dict[str, np.ndarray],
    seq_len: int,
    offset: int,
    vocab: int,
    top_k_values: list[int],
    seed: int,
    logit_scale: float,
    stale_perturb_scale: float,
    memory_price: float,
    stale_repair_price: float,
    build_overhead: float,
    max_ce_residual: float,
    max_grad_residual: float,
    max_byte_ratio: float,
    max_priced_support_ratio: float,
    max_ce_only_byte_ratio: float,
) -> dict:
    window = choose_window(tokens, seq_len, offset, vocab)
    states, targets = states_for(window, params)
    readout = params["readout"] * logit_scale
    exact_ce, probs, exact_grad = materialized_ce_probs_grad(states, targets, readout)
    stale_readout = perturb_readout(readout, seed + offset + 1009, stale_perturb_scale)
    fresh_stale_ce, _, _ = materialized_ce_probs_grad(states, targets, stale_readout)

    n_rows = states.shape[0]
    materialized_bytes = float(n_rows * vocab * 4)
    compact_row_bytes = 32.0
    rows: list[LedgerRow] = []

    ce_only_bytes = n_rows * compact_row_bytes
    ce_only_byte_ratio = ce_only_bytes / materialized_bytes
    ce_only_stale = abs(exact_ce - fresh_stale_ce) / max(abs(fresh_stale_ce), 1e-30)
    ce_only_priced, ce_only_speedup = price_row(
        ce_only_byte_ratio,
        decode_ratio=0.02,
        stale_ratio=ce_only_stale,
        memory_price=memory_price,
        stale_repair_price=stale_repair_price,
        build_overhead=build_overhead,
    )
    rows.append(
        LedgerRow(
            name="ce_only_hash_ledger",
            ce_residual_ratio=0.0,
            grad_residual_ratio=1.0,
            byte_ratio=ce_only_byte_ratio,
            decode_work_ratio=0.02,
            stale_ce_residual_ratio=ce_only_stale,
            priced_support_ratio=ce_only_priced,
            estimated_support_speedup=ce_only_speedup,
            pass_full_gate=False,
            pass_ce_only_support=ce_only_byte_ratio <= max_ce_only_byte_ratio,
            note="Exact compact CE witness, but it cannot reconstruct the CE gradient.",
        )
    )

    full_prob_bytes = n_rows * vocab * 4.0
    full_prob_byte_ratio = full_prob_bytes / materialized_bytes
    full_grad = grad_from_probs(probs, targets, states)
    full_ce = ce_from_probs(probs, targets)
    full_priced, full_speedup = price_row(
        full_prob_byte_ratio,
        decode_ratio=0.95,
        stale_ratio=ce_only_stale,
        memory_price=memory_price,
        stale_repair_price=stale_repair_price,
        build_overhead=build_overhead,
    )
    full_ce_residual = abs(full_ce - exact_ce) / max(abs(exact_ce), 1e-30)
    full_grad_residual = norm_ratio(full_grad - exact_grad, exact_grad)
    rows.append(
        LedgerRow(
            name="full_probability_ledger",
            ce_residual_ratio=full_ce_residual,
            grad_residual_ratio=full_grad_residual,
            byte_ratio=full_prob_byte_ratio,
            decode_work_ratio=0.95,
            stale_ce_residual_ratio=ce_only_stale,
            priced_support_ratio=full_priced,
            estimated_support_speedup=full_speedup,
            pass_full_gate=(
                full_ce_residual <= max_ce_residual
                and full_grad_residual <= max_grad_residual
                and full_prob_byte_ratio <= max_byte_ratio
                and full_priced <= max_priced_support_ratio
            ),
            pass_ce_only_support=False,
            note="Exact CE and gradient witness, but it is as large as the materialized score surface.",
        )
    )

    for top_k in top_k_values:
        approx_probs = topk_residual_probs(probs, targets, top_k)
        approx_ce = ce_from_probs(approx_probs, targets)
        approx_grad = grad_from_probs(approx_probs, targets, states)
        ce_residual = abs(approx_ce - exact_ce) / max(abs(exact_ce), 1e-30)
        grad_residual = norm_ratio(approx_grad - exact_grad, exact_grad)
        entry_bytes = 2.0 + 4.0
        row_bytes = compact_row_bytes + float(top_k) * entry_bytes + 8.0
        byte_ratio = n_rows * row_bytes / materialized_bytes
        decode_ratio = min(1.0, max(0.03, top_k / float(vocab)))
        priced, speedup = price_row(
            byte_ratio,
            decode_ratio=decode_ratio,
            stale_ratio=ce_only_stale,
            memory_price=memory_price,
            stale_repair_price=stale_repair_price,
            build_overhead=build_overhead,
        )
        rows.append(
            LedgerRow(
                name=f"topk_residual_ledger_k{top_k}",
                ce_residual_ratio=ce_residual,
                grad_residual_ratio=grad_residual,
                byte_ratio=byte_ratio,
                decode_work_ratio=decode_ratio,
                stale_ce_residual_ratio=ce_only_stale,
                priced_support_ratio=priced,
                estimated_support_speedup=speedup,
                pass_full_gate=(
                    ce_residual <= max_ce_residual
                    and grad_residual <= max_grad_residual
                    and byte_ratio <= max_byte_ratio
                    and priced <= max_priced_support_ratio
                ),
                pass_ce_only_support=False,
                note="Compact gradient approximation; positive only if residual stays below the full certificate threshold.",
            )
        )

    return {
        "offset": int(offset),
        "logit_scale": float(logit_scale),
        "exact_ce": float(exact_ce),
        "fresh_after_stale_perturb_ce": float(fresh_stale_ce),
        "state_rows": int(n_rows),
        "materialized_score_bytes": int(materialized_bytes),
        "rows": [asdict(row) for row in rows],
    }


def parse_offsets(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("offset list cannot be empty")
    return values


def parse_topk(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("top-k list cannot be empty")
    return values


def summarize_rows(windows: list[dict], min_pass_fraction: float) -> dict:
    names = [row["name"] for row in windows[0]["rows"]]
    summaries = []
    for name in names:
        selected = [row for window in windows for row in window["rows"] if row["name"] == name]
        pass_full = sum(bool(row["pass_full_gate"]) for row in selected) / float(len(selected))
        pass_ce_only = sum(bool(row["pass_ce_only_support"]) for row in selected) / float(len(selected))
        summaries.append(
            {
                "name": name,
                "mean_ce_residual_ratio": float(np.mean([row["ce_residual_ratio"] for row in selected])),
                "mean_grad_residual_ratio": float(np.mean([row["grad_residual_ratio"] for row in selected])),
                "mean_byte_ratio": float(np.mean([row["byte_ratio"] for row in selected])),
                "mean_decode_work_ratio": float(np.mean([row["decode_work_ratio"] for row in selected])),
                "mean_stale_ce_residual_ratio": float(np.mean([row["stale_ce_residual_ratio"] for row in selected])),
                "mean_priced_support_ratio": float(np.mean([row["priced_support_ratio"] for row in selected])),
                "mean_estimated_support_speedup": float(np.mean([row["estimated_support_speedup"] for row in selected])),
                "pass_full_fraction": float(pass_full),
                "pass_ce_only_support_fraction": float(pass_ce_only),
                "passes_full_gate": bool(pass_full >= min_pass_fraction),
                "passes_ce_only_support": bool(pass_ce_only >= min_pass_fraction),
            }
        )
    return {
        "rows": summaries,
        "best_full_row": max(summaries, key=lambda row: row["pass_full_fraction"]),
        "best_support_row": min(summaries, key=lambda row: row["mean_priced_support_ratio"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("src/model_v12/test_data.bin"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-offsets", type=str, default="0,160,320")
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=421120)
    parser.add_argument("--top-k-values", type=str, default="16,64,256")
    parser.add_argument("--logit-scale", type=float, default=1.0)
    parser.add_argument("--stale-perturb-scale", type=float, default=0.006)
    parser.add_argument("--memory-price", type=float, default=0.12)
    parser.add_argument("--stale-repair-price", type=float, default=0.50)
    parser.add_argument("--build-overhead", type=float, default=0.02)
    parser.add_argument("--max-ce-residual", type=float, default=1e-8)
    parser.add_argument("--max-grad-residual", type=float, default=0.05)
    parser.add_argument("--max-byte-ratio", type=float, default=0.25)
    parser.add_argument("--max-priced-support-ratio", type=float, default=0.90)
    parser.add_argument("--max-ce-only-byte-ratio", type=float, default=0.02)
    parser.add_argument("--min-pass-fraction", type=float, default=0.67)
    return parser


def run_gate(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    tokens = read_v12_tokens(args.dataset, args.max_tokens)
    params = init_params(args.vocab, args.hidden, args.seed)
    top_k_values = parse_topk(args.top_k_values)
    offsets = parse_offsets(args.eval_offsets)
    windows = [
        evaluate_window(
            tokens=tokens,
            params=params,
            seq_len=args.seq_len,
            offset=offset,
            vocab=args.vocab,
            top_k_values=top_k_values,
            seed=args.seed,
            logit_scale=args.logit_scale,
            stale_perturb_scale=args.stale_perturb_scale,
            memory_price=args.memory_price,
            stale_repair_price=args.stale_repair_price,
            build_overhead=args.build_overhead,
            max_ce_residual=args.max_ce_residual,
            max_grad_residual=args.max_grad_residual,
            max_byte_ratio=args.max_byte_ratio,
            max_priced_support_ratio=args.max_priced_support_ratio,
            max_ce_only_byte_ratio=args.max_ce_only_byte_ratio,
        )
        for offset in offsets
    ]
    summary = summarize_rows(windows, args.min_pass_fraction)
    any_full_pass = any(row["passes_full_gate"] for row in summary["rows"])
    any_ce_only_support = any(row["passes_ce_only_support"] for row in summary["rows"])
    if any_full_pass:
        decision = "PASS_CE_WITNESS_LEDGER_FULL_GATE"
    elif any_ce_only_support:
        decision = "PASS_CE_ONLY_WITNESS_SUPPORT_GRADIENT_NOT_CERTIFIED"
    else:
        decision = "FAIL_RESIDENT_WITNESS_LEDGER_CURRENT_GATE"
    return {
        "schema": SCHEMA,
        "gate": "r421_ce_witness_ledger_resident_proof_objects",
        "decision": decision,
        "dataset": str(args.dataset),
        "seq_len": int(args.seq_len),
        "vocab": int(args.vocab),
        "hidden": int(args.hidden),
        "seed": int(args.seed),
        "eval_offsets": offsets,
        "top_k_values": top_k_values,
        "logit_scale": float(args.logit_scale),
        "summary": summary,
        "windows": windows,
        "thresholds": {
            "max_ce_residual": float(args.max_ce_residual),
            "max_grad_residual": float(args.max_grad_residual),
            "max_byte_ratio": float(args.max_byte_ratio),
            "max_priced_support_ratio": float(args.max_priced_support_ratio),
            "max_ce_only_byte_ratio": float(args.max_ce_only_byte_ratio),
            "min_pass_fraction": float(args.min_pass_fraction),
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "no_training_integration": True,
        "no_gpu": True,
        "interpretation": (
            "This support gate is positive only as a cheap resident original-CE "
            "proof object. CE-only rows can support score verification, but they "
            "do not certify training gradients. Full-gradient rows must be both "
            "small and cheaper than materialized verification."
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    payload = run_gate(args)
    best_support = payload["summary"]["best_support_row"]
    best_full = payload["summary"]["best_full_row"]
    print(
        "[R421_CE_WITNESS_LEDGER_GATE] "
        f"active=1 decision={payload['decision']} "
        f"seq_len={payload['seq_len']} hidden={payload['hidden']} "
        f"best_support={best_support['name']} "
        f"best_support_priced_ratio={best_support['mean_priced_support_ratio']:.6f} "
        f"best_full={best_full['name']} "
        f"best_full_pass_fraction={best_full['pass_full_fraction']:.6f} "
        "no_training_integration=1 no_gpu=1"
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
