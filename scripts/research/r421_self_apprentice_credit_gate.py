#!/usr/bin/env python3
"""R421 Self-Apprentice State-Credit gate.

This CPU-only gate tests the R421 Wave 2 hypothesis:

    a lagged project-owned v12-like process predicts state-credit cotangents,
    while exact backward passes audit and settle only the residual.

The gate deliberately avoids any external teacher. It builds exact cotangent
records from calibration windows of the same tiny recurrent CE model, fits a
ridge "apprentice" from forward-only state features to exact preactivation
cotangents, and evaluates the apprentice on held-out real-token windows.

It is not a production trainer and does not modify the default path. A positive
result only justifies a deeper v12 trace/residency implementation.
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
SCHEMA = "r421_self_apprentice_credit_gate.v1"


@dataclass
class WindowResult:
    offset: int
    base_ce: float
    exact_update_ce: float
    apprentice_update_ce: float
    exact_progress: float
    apprentice_progress: float
    progress_ratio: float
    residual_fraction: float
    cosine_to_exact: float
    apprentice_recurrent_paid_ratio: float
    whole_step_priced_ratio_estimate: float
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


def forward(window: np.ndarray, params: dict[str, np.ndarray]) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    embed = params["embed"]
    recurrent = params["recurrent"]
    readout = params["readout"]
    inputs = window[:-1]
    targets = window[1:]
    hidden = recurrent.shape[0]
    states = np.zeros((inputs.shape[0], hidden), dtype=np.float64)
    prev_states = np.zeros_like(states)
    h = np.zeros(hidden, dtype=np.float64)
    for i, tok in enumerate(inputs):
        prev_states[i] = h
        h = np.tanh(embed[tok] + recurrent @ h)
        states[i] = h
    logits = states @ readout.T
    ce, dlogits = stable_softmax_ce(logits, targets)
    direct_dh = dlogits @ readout
    return ce, states, prev_states, direct_dh


def exact_state_credit(
    states: np.ndarray,
    direct_dh: np.ndarray,
    recurrent: np.ndarray,
) -> np.ndarray:
    seq_len, hidden = states.shape
    credit = np.zeros((seq_len, hidden), dtype=np.float64)
    for loss_t in range(seq_len):
        dh = direct_dh[loss_t].copy()
        for j in range(loss_t, -1, -1):
            da = dh * (1.0 - states[j] * states[j])
            credit[j] += da
            dh = recurrent.T @ da
    return credit


def recurrent_grad_from_credit(credit: np.ndarray, prev_states: np.ndarray) -> np.ndarray:
    return credit.T @ prev_states


def feature_matrix(states: np.ndarray, prev_states: np.ndarray, direct_dh: np.ndarray) -> np.ndarray:
    seq_len = states.shape[0]
    pos = np.linspace(-1.0, 1.0, num=seq_len, dtype=np.float64)[:, None]
    ones = np.ones((seq_len, 1), dtype=np.float64)
    local_delta = states - prev_states
    direct_preact = direct_dh * (1.0 - states * states)
    return np.concatenate(
        [
            ones,
            pos,
            states,
            prev_states,
            local_delta,
            states * prev_states,
            direct_dh,
            direct_preact,
        ],
        axis=1,
    )


def fit_ridge(features: np.ndarray, targets: np.ndarray, ridge: float) -> np.ndarray:
    xtx = features.T @ features
    reg = ridge * np.eye(xtx.shape[0], dtype=np.float64)
    xty = features.T @ targets
    return np.linalg.solve(xtx + reg, xty)


def train_apprentice(
    tokens: np.ndarray,
    params: dict[str, np.ndarray],
    seq_len: int,
    offsets: list[int],
    vocab: int,
    ridge: float,
) -> tuple[np.ndarray, dict[str, float]]:
    features: list[np.ndarray] = []
    credits: list[np.ndarray] = []
    norms: list[float] = []
    for offset in offsets:
        window = choose_window(tokens, seq_len, offset, vocab)
        _, states, prev_states, direct_dh = forward(window, params)
        exact = exact_state_credit(states, direct_dh, params["recurrent"])
        features.append(feature_matrix(states, prev_states, direct_dh))
        credits.append(exact)
        norms.append(float(np.linalg.norm(exact)))
    x = np.concatenate(features, axis=0)
    y = np.concatenate(credits, axis=0)
    weights = fit_ridge(x, y, ridge)
    pred = x @ weights
    residual = float(np.linalg.norm(y - pred) / max(np.linalg.norm(y), 1e-30))
    return weights, {
        "calibration_records": float(y.shape[0]),
        "calibration_credit_norm_mean": float(np.mean(norms)),
        "calibration_residual_fraction": residual,
    }


def evaluate_window(
    tokens: np.ndarray,
    params: dict[str, np.ndarray],
    weights: np.ndarray,
    seq_len: int,
    offset: int,
    vocab: int,
    lr: float,
    apprentice_cost: float,
    audit_rate: float,
    residual_price: float,
    recurrent_backward_share: float,
    max_residual: float,
    min_progress: float,
    max_recurrent_paid: float,
    min_whole_step_speedup: float,
) -> WindowResult:
    window = choose_window(tokens, seq_len, offset, vocab)
    base_ce, states, prev_states, direct_dh = forward(window, params)
    exact_credit = exact_state_credit(states, direct_dh, params["recurrent"])
    exact_grad = recurrent_grad_from_credit(exact_credit, prev_states)
    apprentice_credit = feature_matrix(states, prev_states, direct_dh) @ weights
    apprentice_grad = recurrent_grad_from_credit(apprentice_credit, prev_states)

    exact_norm = float(np.linalg.norm(exact_grad))
    apprentice_norm = float(np.linalg.norm(apprentice_grad))
    residual_fraction = float(np.linalg.norm(exact_grad - apprentice_grad) / max(exact_norm, 1e-30))
    cosine = float(np.sum(exact_grad * apprentice_grad) / max(exact_norm * apprentice_norm, 1e-30))

    exact_params = {k: v.copy() for k, v in params.items()}
    exact_params["recurrent"] = params["recurrent"] - lr * exact_grad
    exact_ce, _, _, _ = forward(window, exact_params)

    apprentice_params = {k: v.copy() for k, v in params.items()}
    apprentice_params["recurrent"] = params["recurrent"] - lr * apprentice_grad
    apprentice_ce, _, _, _ = forward(window, apprentice_params)

    exact_progress = base_ce - exact_ce
    apprentice_progress = base_ce - apprentice_ce
    progress_ratio = apprentice_progress / max(exact_progress, 1e-30)
    recurrent_paid = apprentice_cost + audit_rate + residual_price * max(0.0, residual_fraction)
    progress = max(progress_ratio, 1e-9)
    whole_step_ratio = 1.0 - recurrent_backward_share * max(0.0, 1.0 - recurrent_paid / progress)
    speedup = 1.0 / max(whole_step_ratio, 1e-9)
    pass_gate = (
        residual_fraction <= max_residual
        and progress_ratio >= min_progress
        and recurrent_paid <= max_recurrent_paid
        and speedup >= min_whole_step_speedup
        and apprentice_progress > 0.0
    )
    if pass_gate:
        decision = "PASS_SELF_APPRENTICE_STATE_CREDIT_GATE"
    elif apprentice_progress <= 0.0:
        decision = "FAIL_APPRENTICE_UPDATE_HARMS_CE"
    elif residual_fraction > max_residual:
        decision = "FAIL_APPRENTICE_RESIDUAL_DENSE"
    elif progress_ratio < min_progress:
        decision = "FAIL_APPRENTICE_PROGRESS_WEAK"
    elif recurrent_paid > max_recurrent_paid:
        decision = "FAIL_AUDIT_OR_SETTLEMENT_TOO_EXPENSIVE"
    else:
        decision = "FAIL_WHOLE_STEP_SPEEDUP_BELOW_BAR"
    return WindowResult(
        offset=int(offset),
        base_ce=float(base_ce),
        exact_update_ce=float(exact_ce),
        apprentice_update_ce=float(apprentice_ce),
        exact_progress=float(exact_progress),
        apprentice_progress=float(apprentice_progress),
        progress_ratio=float(progress_ratio),
        residual_fraction=float(residual_fraction),
        cosine_to_exact=float(cosine),
        apprentice_recurrent_paid_ratio=float(recurrent_paid),
        whole_step_priced_ratio_estimate=float(whole_step_ratio),
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
    train_offsets = parse_offsets(args.calibration_offsets)
    eval_offsets = parse_offsets(args.eval_offsets)
    weights, calib = train_apprentice(tokens, params, args.seq_len, train_offsets, args.vocab, args.ridge)
    windows = [
        evaluate_window(
            tokens=tokens,
            params=params,
            weights=weights,
            seq_len=args.seq_len,
            offset=offset,
            vocab=args.vocab,
            lr=args.lr,
            apprentice_cost=args.apprentice_cost,
            audit_rate=args.audit_rate,
            residual_price=args.residual_price,
            recurrent_backward_share=args.recurrent_backward_share,
            max_residual=args.max_residual,
            min_progress=args.min_progress,
            max_recurrent_paid=args.max_recurrent_paid,
            min_whole_step_speedup=args.min_whole_step_speedup,
        )
        for offset in eval_offsets
    ]
    pass_fraction = sum(row.pass_gate for row in windows) / float(len(windows))
    mean_speedup = float(np.mean([row.whole_step_speedup_estimate for row in windows]))
    mean_residual = float(np.mean([row.residual_fraction for row in windows]))
    mean_progress = float(np.mean([row.progress_ratio for row in windows]))
    if pass_fraction >= args.min_pass_fraction:
        decision = "PASS_SELF_APPRENTICE_IDENTIFYING_GATE"
    elif mean_progress > 0.0 and mean_residual < 1.0:
        decision = "PARK_APPRENTICE_SIGNAL_NEEDS_STRONGER_RESIDUAL_BOUND"
    else:
        decision = "FAIL_APPRENTICE_NON_IDENTIFYING_OR_HARMFUL"
    elapsed = time.perf_counter() - started
    return {
        "schema": SCHEMA,
        "gate": "r421_self_apprentice_state_credit",
        "decision": decision,
        "dataset": str(args.dataset),
        "seq_len": int(args.seq_len),
        "vocab": int(args.vocab),
        "hidden": int(args.hidden),
        "seed": int(args.seed),
        "lr": float(args.lr),
        "calibration_offsets": train_offsets,
        "eval_offsets": eval_offsets,
        "calibration": calib,
        "mean_eval_residual_fraction": mean_residual,
        "mean_eval_progress_ratio": mean_progress,
        "mean_whole_step_speedup_estimate": mean_speedup,
        "pass_fraction": float(pass_fraction),
        "windows": [asdict(row) for row in windows],
        "thresholds": {
            "apprentice_cost": float(args.apprentice_cost),
            "audit_rate": float(args.audit_rate),
            "residual_price": float(args.residual_price),
            "recurrent_backward_share": float(args.recurrent_backward_share),
            "max_residual": float(args.max_residual),
            "min_progress": float(args.min_progress),
            "max_recurrent_paid": float(args.max_recurrent_paid),
            "min_whole_step_speedup": float(args.min_whole_step_speedup),
            "min_pass_fraction": float(args.min_pass_fraction),
        },
        "runtime_seconds": float(elapsed),
        "no_training_integration": True,
        "no_gpu": True,
        "interpretation": (
            "Positive only if a project-owned apprentice trained from prior exact "
            "audits predicts held-out state cotangents well enough that residual "
            "settlement plus dense audit rate stays below the material speed budget. "
            "This is an identifying CPU gate, not a measured training speedup."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("src/model_v12/test_data.bin"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--calibration-offsets", type=str, default="0,160,320,480")
    parser.add_argument("--eval-offsets", type=str, default="640,800,960")
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42191)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--apprentice-cost", type=float, default=0.08)
    parser.add_argument("--audit-rate", type=float, default=0.15)
    parser.add_argument("--residual-price", type=float, default=0.30)
    parser.add_argument("--recurrent-backward-share", type=float, default=0.35)
    parser.add_argument("--max-residual", type=float, default=0.35)
    parser.add_argument("--min-progress", type=float, default=0.85)
    parser.add_argument("--max-recurrent-paid", type=float, default=0.45)
    parser.add_argument("--min-whole-step-speedup", type=float, default=1.10)
    parser.add_argument("--min-pass-fraction", type=float, default=0.67)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_gate(args)
    print(
        "[R421_SELF_APPRENTICE_CREDIT_GATE] "
        f"active=1 decision={payload['decision']} "
        f"seq_len={payload['seq_len']} hidden={payload['hidden']} "
        f"mean_residual={payload['mean_eval_residual_fraction']:.6f} "
        f"mean_progress_ratio={payload['mean_eval_progress_ratio']:.6f} "
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
