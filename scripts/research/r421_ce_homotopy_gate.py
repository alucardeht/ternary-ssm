#!/usr/bin/env python3
"""R421 Objective Ladder / CE Homotopy With Settlement gate.

This CPU-only gate tests whether a cheaper stage objective can be used as an
algebraic bridge to original CE rather than as a proxy curriculum.

The tested bridge is a coarse marginal CE:

    L_fast = -log sum_{token in class(y)} p_theta(token | x)

followed by a priced original-CE settlement step. The gate compares the
two-stage update against a direct original-CE update on the same hidden states,
and rejects the bridge if it does not improve original CE progress per paid
work or if a random class partition performs similarly.

This is not a production trainer and does not modify the default path.
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
SCHEMA = "r421_ce_homotopy_gate.v1"


@dataclass
class WindowResult:
    offset: int
    base_ce: float
    direct_ce: float
    cheap_ce: float
    homotopy_ce: float
    random_homotopy_ce: float
    direct_progress: float
    cheap_progress: float
    homotopy_progress: float
    random_homotopy_progress: float
    homotopy_paid_ratio: float
    direct_progress_per_paid_work: float
    homotopy_progress_per_paid_work: float
    random_progress_per_paid_work: float
    progress_per_paid_ratio: float
    random_control_advantage: float
    settlement_residual_ratio: float
    pass_gate: bool
    decision: str


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


def full_ce_and_grad(states: np.ndarray, targets: np.ndarray, readout: np.ndarray) -> tuple[float, np.ndarray]:
    logits = states @ readout.T
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z, dtype=np.float64)
    probs = exp / exp.sum(axis=1, keepdims=True)
    ce = float(-np.log(np.maximum(probs[np.arange(targets.shape[0]), targets], 1e-300)).mean())
    dlogits = probs
    dlogits[np.arange(targets.shape[0]), targets] -= 1.0
    dlogits /= float(targets.shape[0])
    grad = dlogits.T @ states
    return ce, grad


def make_class_map(vocab: int, class_count: int, mode: str, seed: int) -> np.ndarray:
    if class_count <= 1 or class_count > vocab:
        raise ValueError("class_count must be between 2 and vocab")
    if mode == "radix":
        width = int(math.ceil(vocab / class_count))
        return np.minimum(np.arange(vocab, dtype=np.int64) // width, class_count - 1)
    if mode == "mod":
        return np.arange(vocab, dtype=np.int64) % class_count
    if mode == "random":
        rng = np.random.default_rng(seed)
        labels = np.arange(vocab, dtype=np.int64) % class_count
        rng.shuffle(labels)
        return labels
    raise ValueError(f"unsupported class map mode: {mode}")


def class_ce_and_grad(
    states: np.ndarray,
    targets: np.ndarray,
    readout: np.ndarray,
    class_map: np.ndarray,
    class_count: int,
) -> tuple[float, np.ndarray]:
    logits = states @ readout.T
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z, dtype=np.float64)
    probs = exp / exp.sum(axis=1, keepdims=True)
    target_classes = class_map[targets]
    class_probs = np.zeros((targets.shape[0], class_count), dtype=np.float64)
    np.add.at(class_probs, (np.arange(targets.shape[0])[:, None], class_map[None, :]), probs)
    target_mass = np.maximum(class_probs[np.arange(targets.shape[0]), target_classes], 1e-300)
    ce = float(-np.log(target_mass).mean())
    dlogits = probs.copy()
    for row, cls in enumerate(target_classes):
        mask = class_map == cls
        dlogits[row, mask] -= probs[row, mask] / target_mass[row]
    dlogits /= float(targets.shape[0])
    grad = dlogits.T @ states
    return ce, grad


def eval_ce(states: np.ndarray, targets: np.ndarray, readout: np.ndarray) -> float:
    ce, _ = full_ce_and_grad(states, targets, readout)
    return ce


def evaluate_window(
    tokens: np.ndarray,
    params: dict[str, np.ndarray],
    seq_len: int,
    offset: int,
    vocab: int,
    class_count: int,
    class_map: np.ndarray,
    random_class_map: np.ndarray,
    lr: float,
    settlement_fraction: float,
    bridge_overhead: float,
    max_settlement_residual: float,
    min_progress_per_paid_ratio: float,
    min_random_advantage: float,
    min_cheap_progress_ratio: float,
) -> WindowResult:
    window = choose_window(tokens, seq_len, offset, vocab)
    states, targets = states_for(window, params)
    readout = params["readout"]
    base_ce, full_grad = full_ce_and_grad(states, targets, readout)
    _, class_grad = class_ce_and_grad(states, targets, readout, class_map, class_count)
    _, random_class_grad = class_ce_and_grad(states, targets, readout, random_class_map, class_count)

    direct_readout = readout - lr * full_grad
    direct_ce = eval_ce(states, targets, direct_readout)

    cheap_readout = readout - lr * class_grad
    cheap_ce = eval_ce(states, targets, cheap_readout)
    _, settlement_grad = full_ce_and_grad(states, targets, cheap_readout)
    homotopy_readout = cheap_readout - lr * settlement_fraction * settlement_grad
    homotopy_ce = eval_ce(states, targets, homotopy_readout)

    random_readout = readout - lr * random_class_grad
    _, random_settlement_grad = full_ce_and_grad(states, targets, random_readout)
    random_homotopy_readout = random_readout - lr * settlement_fraction * random_settlement_grad
    random_homotopy_ce = eval_ce(states, targets, random_homotopy_readout)

    direct_progress = base_ce - direct_ce
    cheap_progress = base_ce - cheap_ce
    homotopy_progress = base_ce - homotopy_ce
    random_progress = base_ce - random_homotopy_ce
    class_work = class_count / float(vocab)
    paid_ratio = class_work + settlement_fraction + bridge_overhead
    direct_ppw = direct_progress
    homotopy_ppw = homotopy_progress / max(paid_ratio, 1e-30)
    random_ppw = random_progress / max(paid_ratio, 1e-30)
    ratio = homotopy_ppw / max(direct_ppw, 1e-30)
    random_advantage = homotopy_ppw - random_ppw
    settlement_residual_ratio = max(0.0, direct_progress - homotopy_progress) / max(direct_progress, 1e-30)
    cheap_ratio = cheap_progress / max(direct_progress, 1e-30)
    pass_gate = (
        ratio >= min_progress_per_paid_ratio
        and random_advantage >= min_random_advantage * max(abs(direct_ppw), 1e-30)
        and settlement_residual_ratio <= max_settlement_residual
        and cheap_ratio >= min_cheap_progress_ratio
        and homotopy_progress > 0.0
    )
    if pass_gate:
        decision = "PASS_CE_HOMOTOPY_BRIDGE_GATE"
    elif homotopy_progress <= 0.0 or cheap_progress < 0.0:
        decision = "FAIL_HOMOTOPY_HARMS_ORIGINAL_CE"
    elif cheap_ratio < min_cheap_progress_ratio:
        decision = "FAIL_FAST_STAGE_PROXY_WEAK"
    elif settlement_residual_ratio > max_settlement_residual:
        decision = "FAIL_SETTLEMENT_RESIDUAL_TOO_HIGH"
    elif random_advantage < min_random_advantage * max(abs(direct_ppw), 1e-30):
        decision = "FAIL_RANDOM_BRIDGE_CONTROL_NOT_BEATEN"
    else:
        decision = "FAIL_PROGRESS_PER_PAID_WORK_BELOW_BAR"
    return WindowResult(
        offset=int(offset),
        base_ce=float(base_ce),
        direct_ce=float(direct_ce),
        cheap_ce=float(cheap_ce),
        homotopy_ce=float(homotopy_ce),
        random_homotopy_ce=float(random_homotopy_ce),
        direct_progress=float(direct_progress),
        cheap_progress=float(cheap_progress),
        homotopy_progress=float(homotopy_progress),
        random_homotopy_progress=float(random_progress),
        homotopy_paid_ratio=float(paid_ratio),
        direct_progress_per_paid_work=float(direct_ppw),
        homotopy_progress_per_paid_work=float(homotopy_ppw),
        random_progress_per_paid_work=float(random_ppw),
        progress_per_paid_ratio=float(ratio),
        random_control_advantage=float(random_advantage),
        settlement_residual_ratio=float(settlement_residual_ratio),
        pass_gate=bool(pass_gate),
        decision=decision,
    )


def parse_offsets(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("offset list cannot be empty")
    return values


def run_gate(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    tokens = read_v12_tokens(args.dataset, args.max_tokens)
    params = init_params(args.vocab, args.hidden, args.seed)
    class_map = make_class_map(args.vocab, args.class_count, args.class_mode, args.seed)
    random_class_map = make_class_map(args.vocab, args.class_count, "random", args.seed + 17)
    offsets = parse_offsets(args.eval_offsets)
    windows = [
        evaluate_window(
            tokens=tokens,
            params=params,
            seq_len=args.seq_len,
            offset=offset,
            vocab=args.vocab,
            class_count=args.class_count,
            class_map=class_map,
            random_class_map=random_class_map,
            lr=args.lr,
            settlement_fraction=args.settlement_fraction,
            bridge_overhead=args.bridge_overhead,
            max_settlement_residual=args.max_settlement_residual,
            min_progress_per_paid_ratio=args.min_progress_per_paid_ratio,
            min_random_advantage=args.min_random_advantage,
            min_cheap_progress_ratio=args.min_cheap_progress_ratio,
        )
        for offset in offsets
    ]
    pass_fraction = sum(row.pass_gate for row in windows) / float(len(windows))
    mean_ratio = float(np.mean([row.progress_per_paid_ratio for row in windows]))
    mean_cheap = float(np.mean([row.cheap_progress / max(row.direct_progress, 1e-30) for row in windows]))
    mean_settlement = float(np.mean([row.settlement_residual_ratio for row in windows]))
    mean_random_advantage = float(np.mean([row.random_control_advantage for row in windows]))
    if pass_fraction >= args.min_pass_fraction:
        decision = "PASS_CE_HOMOTOPY_IDENTIFYING_GATE"
    elif mean_ratio > 1.0 and mean_cheap > 0.0:
        decision = "PARK_BRIDGE_SIGNAL_SETTLEMENT_OR_CONTROL_WEAK"
    else:
        decision = "FAIL_CE_HOMOTOPY_CURRENT_GATE"
    elapsed = time.perf_counter() - started
    return {
        "schema": SCHEMA,
        "gate": "r421_objective_ladder_ce_homotopy",
        "decision": decision,
        "dataset": str(args.dataset),
        "seq_len": int(args.seq_len),
        "vocab": int(args.vocab),
        "hidden": int(args.hidden),
        "seed": int(args.seed),
        "class_count": int(args.class_count),
        "class_mode": args.class_mode,
        "settlement_fraction": float(args.settlement_fraction),
        "bridge_overhead": float(args.bridge_overhead),
        "eval_offsets": offsets,
        "mean_progress_per_paid_ratio": mean_ratio,
        "mean_cheap_stage_progress_ratio": mean_cheap,
        "mean_settlement_residual_ratio": mean_settlement,
        "mean_random_control_advantage": mean_random_advantage,
        "pass_fraction": float(pass_fraction),
        "windows": [asdict(row) for row in windows],
        "thresholds": {
            "max_settlement_residual": float(args.max_settlement_residual),
            "min_progress_per_paid_ratio": float(args.min_progress_per_paid_ratio),
            "min_random_advantage": float(args.min_random_advantage),
            "min_cheap_progress_ratio": float(args.min_cheap_progress_ratio),
            "min_pass_fraction": float(args.min_pass_fraction),
        },
        "runtime_seconds": float(elapsed),
        "no_training_integration": True,
        "no_gpu": True,
        "interpretation": (
            "Positive only if the cheap coarse CE stage plus priced original-CE "
            "settlement beats direct original CE progress per paid work and beats "
            "a random class bridge. This is an identifying CPU gate, not a "
            "measured training speedup."
        ),
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
    parser.add_argument("--seed", type=int, default=421111)
    parser.add_argument("--class-count", type=int, default=256)
    parser.add_argument("--class-mode", choices=["radix", "mod"], default="radix")
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--settlement-fraction", type=float, default=0.25)
    parser.add_argument("--bridge-overhead", type=float, default=0.03)
    parser.add_argument("--max-settlement-residual", type=float, default=0.20)
    parser.add_argument("--min-progress-per-paid-ratio", type=float, default=1.10)
    parser.add_argument("--min-random-advantage", type=float, default=0.05)
    parser.add_argument("--min-cheap-progress-ratio", type=float, default=0.20)
    parser.add_argument("--min-pass-fraction", type=float, default=0.67)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_gate(args)
    print(
        "[R421_CE_HOMOTOPY_GATE] "
        f"active=1 decision={payload['decision']} "
        f"seq_len={payload['seq_len']} hidden={payload['hidden']} "
        f"class_count={payload['class_count']} settlement_fraction={payload['settlement_fraction']:.6f} "
        f"mean_progress_per_paid_ratio={payload['mean_progress_per_paid_ratio']:.6f} "
        f"mean_cheap_stage_progress_ratio={payload['mean_cheap_stage_progress_ratio']:.6f} "
        f"mean_settlement_residual_ratio={payload['mean_settlement_residual_ratio']:.6f} "
        f"pass_fraction={payload['pass_fraction']:.6f} "
        f"no_training_integration=1 no_gpu=1"
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
