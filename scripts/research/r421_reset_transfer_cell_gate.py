#!/usr/bin/env python3
"""R421 Reset And Transfer Cell Dynamics gate.

This CPU-only gate tests whether prefix-computable reset/transfer cells can
replace repeated recurrent transitions without changing original CE beyond a
priced residual.

The gate learns tiny transfer cells from calibration forward traces only:
for frequent tokens whose post-token states are low-variance, the cell stores a
mean state. On held-out windows, the candidate replaces the normal recurrent
state at those token positions with the stored state and blocks recurrent
credit through the reset point. The gate compares original CE, random-cell
controls, exact recurrent update progress, and recurrent chain work saved.

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
SCHEMA = "r421_reset_transfer_cell_gate.v1"


@dataclass
class CellStats:
    token: int
    support: int
    mean_variance: float
    state_norm: float


@dataclass
class WindowResult:
    offset: int
    transfer_fraction: float
    recurrent_chain_work_fraction: float
    base_ce: float
    transfer_ce: float
    random_ce: float
    ce_residual: float
    random_control_advantage: float
    base_update_ce: float
    transfer_update_ce: float
    base_progress: float
    transfer_progress: float
    progress_ratio: float
    priced_ratio: float
    whole_step_speedup_estimate: float
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


def stable_softmax_ce(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z, dtype=np.float64)
    probs = exp / exp.sum(axis=1, keepdims=True)
    ce = float(-np.log(np.maximum(probs[np.arange(targets.shape[0]), targets], 1e-300)).mean())
    dlogits = probs.astype(np.float64)
    dlogits[np.arange(targets.shape[0]), targets] -= 1.0
    dlogits /= float(targets.shape[0])
    return ce, dlogits


def init_params(vocab: int, hidden: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    embed = rng.normal(0.0, 0.18, size=(vocab, hidden)).astype(np.float64)
    recurrent = rng.normal(0.0, 0.22 / math.sqrt(hidden), size=(hidden, hidden)).astype(np.float64)
    spectral = float(np.linalg.norm(recurrent, ord=2))
    if spectral > 0.92:
        recurrent *= 0.92 / spectral
    readout = rng.normal(0.0, 0.12 / math.sqrt(hidden), size=(vocab, hidden)).astype(np.float64)
    return {"embed": embed, "recurrent": recurrent, "readout": readout}


def choose_window(tokens: np.ndarray, seq_len: int, offset: int, vocab: int) -> np.ndarray:
    if tokens.shape[0] < seq_len + 1:
        raise ValueError("not enough tokens for requested sequence length")
    max_start = tokens.shape[0] - seq_len - 1
    start = min(max(offset, 0), max_start)
    window = tokens[start : start + seq_len + 1].copy()
    if window.max(initial=0) >= vocab:
        window %= vocab
    return window


def forward(
    window: np.ndarray,
    params: dict[str, np.ndarray],
    cells: dict[int, np.ndarray] | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    embed = params["embed"]
    recurrent = params["recurrent"]
    readout = params["readout"]
    inputs = window[:-1]
    targets = window[1:]
    hidden = recurrent.shape[0]
    states = np.zeros((inputs.shape[0], hidden), dtype=np.float64)
    prev_states = np.zeros_like(states)
    reset_mask = np.zeros(inputs.shape[0], dtype=bool)
    h = np.zeros(hidden, dtype=np.float64)
    for i, tok in enumerate(inputs):
        prev_states[i] = h
        h = np.tanh(embed[tok] + recurrent @ h)
        if cells is not None and int(tok) in cells:
            h = cells[int(tok)].copy()
            reset_mask[i] = True
        states[i] = h
    logits = states @ readout.T
    ce, dlogits = stable_softmax_ce(logits, targets)
    direct_dh = dlogits @ readout
    return ce, states, prev_states, direct_dh, reset_mask


def recurrent_grad_and_work(
    states: np.ndarray,
    prev_states: np.ndarray,
    direct_dh: np.ndarray,
    recurrent: np.ndarray,
    reset_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    seq_len, hidden = states.shape
    grad = np.zeros((hidden, hidden), dtype=np.float64)
    paid_pairs = 0
    for loss_t in range(seq_len):
        dh = direct_dh[loss_t].copy()
        for j in range(loss_t, -1, -1):
            if reset_mask[j]:
                break
            da = dh * (1.0 - states[j] * states[j])
            grad += np.outer(da, prev_states[j])
            paid_pairs += 1
            dh = recurrent.T @ da
    full_pairs = seq_len * (seq_len + 1) / 2.0
    return grad, float(paid_pairs / max(full_pairs, 1.0))


def collect_cell_candidates(
    tokens: np.ndarray,
    params: dict[str, np.ndarray],
    seq_len: int,
    offsets: list[int],
    vocab: int,
    min_support: int,
    max_variance: float,
    max_cells: int,
) -> tuple[dict[int, np.ndarray], list[CellStats]]:
    by_token: dict[int, list[np.ndarray]] = {}
    for offset in offsets:
        window = choose_window(tokens, seq_len, offset, vocab)
        _, states, _, _, _ = forward(window, params, None)
        for tok, state in zip(window[:-1], states):
            by_token.setdefault(int(tok), []).append(state.copy())

    stats: list[tuple[float, int, np.ndarray, CellStats]] = []
    for tok, rows in by_token.items():
        if len(rows) < min_support:
            continue
        mat = np.stack(rows, axis=0)
        mean_state = mat.mean(axis=0)
        mean_var = float(np.mean(np.var(mat, axis=0)))
        if mean_var > max_variance:
            continue
        score = mean_var / max(float(len(rows)), 1.0)
        stats.append(
            (
                score,
                tok,
                mean_state,
                CellStats(
                    token=tok,
                    support=len(rows),
                    mean_variance=mean_var,
                    state_norm=float(np.linalg.norm(mean_state)),
                ),
            )
        )
    stats.sort(key=lambda item: (item[0], -item[3].support))
    selected = stats[:max_cells]
    return {tok: mean for _, tok, mean, _ in selected}, [item for _, _, _, item in selected]


def random_cells_from(
    cells: dict[int, np.ndarray],
    seed: int,
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    keys = list(cells.keys())
    values = [cells[k].copy() for k in keys]
    rng.shuffle(values)
    return {key: value for key, value in zip(keys, values)}


def evaluate_window(
    tokens: np.ndarray,
    params: dict[str, np.ndarray],
    cells: dict[int, np.ndarray],
    random_cells: dict[int, np.ndarray],
    seq_len: int,
    offset: int,
    vocab: int,
    lr: float,
    recurrent_backward_share: float,
    lookup_overhead: float,
    residual_price: float,
    max_ce_residual: float,
    min_transfer_fraction: float,
    min_progress_ratio: float,
    min_random_advantage: float,
    min_whole_step_speedup: float,
) -> WindowResult:
    window = choose_window(tokens, seq_len, offset, vocab)
    base_ce, base_states, base_prev, base_direct, base_mask = forward(window, params, None)
    transfer_ce, transfer_states, transfer_prev, transfer_direct, transfer_mask = forward(window, params, cells)
    random_ce, _, _, _, _ = forward(window, params, random_cells)

    base_grad, _ = recurrent_grad_and_work(
        base_states, base_prev, base_direct, params["recurrent"], base_mask
    )
    transfer_grad, work_fraction = recurrent_grad_and_work(
        transfer_states, transfer_prev, transfer_direct, params["recurrent"], transfer_mask
    )

    base_params = {k: v.copy() for k, v in params.items()}
    base_params["recurrent"] = params["recurrent"] - lr * base_grad
    base_update_ce, _, _, _, _ = forward(window, base_params, None)

    transfer_params = {k: v.copy() for k, v in params.items()}
    transfer_params["recurrent"] = params["recurrent"] - lr * transfer_grad
    transfer_update_ce, _, _, _, _ = forward(window, transfer_params, cells)

    base_progress = base_ce - base_update_ce
    transfer_progress = transfer_ce - transfer_update_ce
    progress_ratio = transfer_progress / max(base_progress, 1e-30)
    ce_residual = transfer_ce - base_ce
    transfer_fraction = float(np.mean(transfer_mask))
    random_advantage = random_ce - transfer_ce
    residual_cost = residual_price * max(0.0, ce_residual)
    recurrent_paid = work_fraction + lookup_overhead + residual_cost
    whole_step_ratio = 1.0 - recurrent_backward_share * max(
        0.0, 1.0 - recurrent_paid / max(progress_ratio, 1e-9)
    )
    speedup = 1.0 / max(whole_step_ratio, 1e-9)
    pass_gate = (
        transfer_fraction >= min_transfer_fraction
        and ce_residual <= max_ce_residual
        and progress_ratio >= min_progress_ratio
        and random_advantage >= min_random_advantage
        and speedup >= min_whole_step_speedup
        and transfer_progress > 0.0
    )
    if pass_gate:
        decision = "PASS_RESET_TRANSFER_CELL_GATE"
    elif transfer_fraction < min_transfer_fraction:
        decision = "FAIL_NO_MATERIAL_TRANSFER_COVERAGE"
    elif ce_residual > max_ce_residual:
        decision = "FAIL_ORIGINAL_CE_RESIDUAL_TOO_HIGH"
    elif random_advantage < min_random_advantage:
        decision = "FAIL_RANDOM_CELL_CONTROL_NOT_BEATEN"
    elif progress_ratio < min_progress_ratio or transfer_progress <= 0.0:
        decision = "FAIL_TRANSFER_DOES_NOT_PRESERVE_CE_PROGRESS"
    else:
        decision = "FAIL_WHOLE_STEP_SPEEDUP_BELOW_BAR"
    return WindowResult(
        offset=int(offset),
        transfer_fraction=transfer_fraction,
        recurrent_chain_work_fraction=float(work_fraction),
        base_ce=float(base_ce),
        transfer_ce=float(transfer_ce),
        random_ce=float(random_ce),
        ce_residual=float(ce_residual),
        random_control_advantage=float(random_advantage),
        base_update_ce=float(base_update_ce),
        transfer_update_ce=float(transfer_update_ce),
        base_progress=float(base_progress),
        transfer_progress=float(transfer_progress),
        progress_ratio=float(progress_ratio),
        priced_ratio=float(recurrent_paid),
        whole_step_speedup_estimate=float(speedup),
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
    calibration_offsets = parse_offsets(args.calibration_offsets)
    eval_offsets = parse_offsets(args.eval_offsets)
    cells, cell_stats = collect_cell_candidates(
        tokens=tokens,
        params=params,
        seq_len=args.seq_len,
        offsets=calibration_offsets,
        vocab=args.vocab,
        min_support=args.min_support,
        max_variance=args.max_cell_variance,
        max_cells=args.max_cells,
    )
    random_cells = random_cells_from(cells, args.seed + 17)
    windows = [
        evaluate_window(
            tokens=tokens,
            params=params,
            cells=cells,
            random_cells=random_cells,
            seq_len=args.seq_len,
            offset=offset,
            vocab=args.vocab,
            lr=args.lr,
            recurrent_backward_share=args.recurrent_backward_share,
            lookup_overhead=args.lookup_overhead,
            residual_price=args.residual_price,
            max_ce_residual=args.max_ce_residual,
            min_transfer_fraction=args.min_transfer_fraction,
            min_progress_ratio=args.min_progress_ratio,
            min_random_advantage=args.min_random_advantage,
            min_whole_step_speedup=args.min_whole_step_speedup,
        )
        for offset in eval_offsets
    ]
    pass_fraction = sum(row.pass_gate for row in windows) / float(len(windows))
    mean_transfer = float(np.mean([row.transfer_fraction for row in windows]))
    mean_ce_residual = float(np.mean([row.ce_residual for row in windows]))
    mean_progress = float(np.mean([row.progress_ratio for row in windows]))
    mean_work = float(np.mean([row.recurrent_chain_work_fraction for row in windows]))
    mean_speedup = float(np.mean([row.whole_step_speedup_estimate for row in windows]))
    if not cells:
        decision = "FAIL_NO_CERTIFIED_TRANSFER_CELLS"
    elif pass_fraction >= args.min_pass_fraction:
        decision = "PASS_RESET_TRANSFER_IDENTIFYING_GATE"
    elif mean_transfer >= args.min_transfer_fraction and mean_ce_residual <= args.max_ce_residual:
        decision = "PARK_TRANSFER_SIGNAL_WEAK_PROGRESS_OR_CONTROL"
    else:
        decision = "FAIL_RESET_TRANSFER_CURRENT_GATE"
    elapsed = time.perf_counter() - started
    return {
        "schema": SCHEMA,
        "gate": "r421_reset_transfer_cell_dynamics",
        "decision": decision,
        "dataset": str(args.dataset),
        "seq_len": int(args.seq_len),
        "vocab": int(args.vocab),
        "hidden": int(args.hidden),
        "seed": int(args.seed),
        "cell_count": len(cells),
        "cell_stats": [asdict(item) for item in cell_stats],
        "calibration_offsets": calibration_offsets,
        "eval_offsets": eval_offsets,
        "mean_transfer_fraction": mean_transfer,
        "mean_ce_residual": mean_ce_residual,
        "mean_progress_ratio": mean_progress,
        "mean_recurrent_chain_work_fraction": mean_work,
        "mean_whole_step_speedup_estimate": mean_speedup,
        "pass_fraction": float(pass_fraction),
        "windows": [asdict(row) for row in windows],
        "thresholds": {
            "min_support": int(args.min_support),
            "max_cell_variance": float(args.max_cell_variance),
            "max_cells": int(args.max_cells),
            "lookup_overhead": float(args.lookup_overhead),
            "residual_price": float(args.residual_price),
            "recurrent_backward_share": float(args.recurrent_backward_share),
            "max_ce_residual": float(args.max_ce_residual),
            "min_transfer_fraction": float(args.min_transfer_fraction),
            "min_progress_ratio": float(args.min_progress_ratio),
            "min_random_advantage": float(args.min_random_advantage),
            "min_whole_step_speedup": float(args.min_whole_step_speedup),
            "min_pass_fraction": float(args.min_pass_fraction),
        },
        "runtime_seconds": float(elapsed),
        "no_training_integration": True,
        "no_gpu": True,
        "interpretation": (
            "Positive only if prefix-computable transfer cells preserve original "
            "CE, beat random-cell controls, keep held-out recurrent update "
            "progress, and save enough recurrent chain work after lookup and "
            "residual costs. This is an identifying CPU gate, not a measured "
            "training speedup."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("src/model_v12/test_data.bin"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--calibration-offsets", type=str, default="0,160,320,480,640,800")
    parser.add_argument("--eval-offsets", type=str, default="1120,1280,1440")
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=421101)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--max-cell-variance", type=float, default=0.010)
    parser.add_argument("--max-cells", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--lookup-overhead", type=float, default=0.03)
    parser.add_argument("--residual-price", type=float, default=8.0)
    parser.add_argument("--recurrent-backward-share", type=float, default=0.35)
    parser.add_argument("--max-ce-residual", type=float, default=0.002)
    parser.add_argument("--min-transfer-fraction", type=float, default=0.08)
    parser.add_argument("--min-progress-ratio", type=float, default=0.85)
    parser.add_argument("--min-random-advantage", type=float, default=0.0001)
    parser.add_argument("--min-whole-step-speedup", type=float, default=1.10)
    parser.add_argument("--min-pass-fraction", type=float, default=0.67)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_gate(args)
    print(
        "[R421_RESET_TRANSFER_CELL_GATE] "
        f"active=1 decision={payload['decision']} "
        f"seq_len={payload['seq_len']} hidden={payload['hidden']} "
        f"cells={payload['cell_count']} "
        f"mean_transfer_fraction={payload['mean_transfer_fraction']:.6f} "
        f"mean_ce_residual={payload['mean_ce_residual']:.6f} "
        f"mean_progress_ratio={payload['mean_progress_ratio']:.6f} "
        f"mean_whole_step_speedup_estimate={payload['mean_whole_step_speedup_estimate']:.6f} "
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
