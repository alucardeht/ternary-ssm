#!/usr/bin/env python3
"""R421 Tail-Cotangent Debt gate.

This is a cheap CPU identifying gate for the R421 Wave 2 finding
"Tail-Cotangent Debt And Boundary Refresh".

The gate compares exact full recurrent cotangents against truncated recurrent
cotangents on a small next-token CE problem built from a real v12 uint16 token
shard. It is not a training benchmark and does not modify production training.
Its job is to test whether short horizons can preserve enough original-CE
descent to justify a deeper GPU/v12 implementation.
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
SCHEMA = "r421_tail_cotangent_debt_gate.v1"


@dataclass
class HorizonResult:
    horizon: int
    work_fraction: float
    captured_fraction: float
    residual_fraction: float
    cosine_to_full: float
    first_order_progress_ratio: float
    actual_progress_ratio: float
    full_update_ce: float
    approx_update_ce: float
    priced_ratio: float
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


def parse_horizons(raw: str, seq_len: int) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("horizons must be positive")
        values.append(min(value, seq_len))
    return sorted(set(values))


def stable_softmax_ce(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z, dtype=np.float64)
    probs = exp / exp.sum(axis=1, keepdims=True)
    target_probs = probs[np.arange(targets.shape[0]), targets]
    ce = float(-np.log(np.maximum(target_probs, 1e-300)).mean())
    dlogits = probs.astype(np.float64)
    dlogits[np.arange(targets.shape[0]), targets] -= 1.0
    dlogits /= float(targets.shape[0])
    return ce, dlogits


def init_params(vocab: int, hidden: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    embed = rng.normal(0.0, 0.18, size=(vocab, hidden)).astype(np.float64)
    recurrent = rng.normal(0.0, 0.22 / math.sqrt(hidden), size=(hidden, hidden)).astype(np.float64)
    # Keep the recurrent map stable enough that tail debt is not a numerical artifact.
    spectral = float(np.linalg.norm(recurrent, ord=2))
    if spectral > 0.92:
        recurrent *= 0.92 / spectral
    readout = rng.normal(0.0, 0.12 / math.sqrt(hidden), size=(vocab, hidden)).astype(np.float64)
    return {"embed": embed, "recurrent": recurrent, "readout": readout}


def forward(tokens: np.ndarray, params: dict[str, np.ndarray]) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    embed = params["embed"]
    recurrent = params["recurrent"]
    readout = params["readout"]
    inputs = tokens[:-1]
    targets = tokens[1:]
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


def recurrent_grad_for_horizon(
    states: np.ndarray,
    prev_states: np.ndarray,
    direct_dh: np.ndarray,
    recurrent: np.ndarray,
    horizon: int,
) -> np.ndarray:
    seq_len, hidden = states.shape
    grad = np.zeros((hidden, hidden), dtype=np.float64)
    for loss_t in range(seq_len):
        dh = direct_dh[loss_t].copy()
        stop = max(-1, loss_t - horizon)
        for j in range(loss_t, stop, -1):
            da = dh * (1.0 - states[j] * states[j])
            grad += np.outer(da, prev_states[j])
            dh = recurrent.T @ da
    return grad


def recurrent_work_fraction(seq_len: int, horizon: int) -> float:
    full_work = seq_len * (seq_len + 1) / 2.0
    truncated = sum(min(i + 1, horizon) for i in range(seq_len))
    return float(truncated / full_work)


def evaluate_horizon(
    tokens: np.ndarray,
    params: dict[str, np.ndarray],
    base_ce: float,
    states: np.ndarray,
    prev_states: np.ndarray,
    direct_dh: np.ndarray,
    full_grad: np.ndarray,
    full_update_ce: float,
    horizon: int,
    lr: float,
    max_residual: float,
    min_progress: float,
    max_priced_ratio: float,
    debt_price: float,
    recurrent_backward_share: float,
    min_whole_step_speedup: float,
) -> HorizonResult:
    recurrent = params["recurrent"]
    approx_grad = recurrent_grad_for_horizon(states, prev_states, direct_dh, recurrent, horizon)
    full_norm = float(np.linalg.norm(full_grad))
    approx_norm = float(np.linalg.norm(approx_grad))
    residual_norm = float(np.linalg.norm(full_grad - approx_grad))
    dot = float(np.sum(full_grad * approx_grad))
    denom = max(full_norm * approx_norm, 1e-30)
    cosine = dot / denom
    residual_fraction = residual_norm / max(full_norm, 1e-30)
    captured_fraction = max(0.0, 1.0 - residual_fraction)
    full_first_order = lr * full_norm * full_norm
    approx_first_order = lr * dot
    first_order_ratio = approx_first_order / max(full_first_order, 1e-30)

    params_approx = {k: v.copy() for k, v in params.items()}
    params_approx["recurrent"] = recurrent - lr * approx_grad
    approx_update_ce, _, _, _ = forward(tokens, params_approx)
    full_progress = base_ce - full_update_ce
    approx_progress = base_ce - approx_update_ce
    actual_progress_ratio = approx_progress / max(full_progress, 1e-30)

    work_fraction = recurrent_work_fraction(states.shape[0], horizon)
    debt_cost = debt_price * max(0.0, residual_fraction)
    progress = max(actual_progress_ratio, 1e-9)
    priced_ratio = (work_fraction + debt_cost) / progress
    whole_step_priced_ratio = 1.0 - recurrent_backward_share * max(0.0, 1.0 - priced_ratio)
    whole_step_speedup = 1.0 / max(whole_step_priced_ratio, 1e-9)
    pass_gate = (
        residual_fraction <= max_residual
        and actual_progress_ratio >= min_progress
        and priced_ratio <= max_priced_ratio
        and whole_step_speedup >= min_whole_step_speedup
        and approx_progress > 0.0
    )
    if pass_gate:
        decision = "PASS_TAIL_COTANGENT_DEBT_IDENTIFYING_GATE"
    elif approx_progress <= 0.0:
        decision = "FAIL_APPROX_UPDATE_HARMS_CE"
    elif residual_fraction > max_residual:
        decision = "FAIL_DENSE_TAIL_DEBT"
    elif actual_progress_ratio < min_progress:
        decision = "FAIL_INSUFFICIENT_ORIGINAL_CE_PROGRESS"
    elif whole_step_speedup < min_whole_step_speedup:
        decision = "FAIL_WHOLE_STEP_SPEEDUP_ESTIMATE_BELOW_BAR"
    else:
        decision = "FAIL_PRICED_RATIO"
    return HorizonResult(
        horizon=int(horizon),
        work_fraction=float(work_fraction),
        captured_fraction=float(captured_fraction),
        residual_fraction=float(residual_fraction),
        cosine_to_full=float(cosine),
        first_order_progress_ratio=float(first_order_ratio),
        actual_progress_ratio=float(actual_progress_ratio),
        full_update_ce=float(full_update_ce),
        approx_update_ce=float(approx_update_ce),
        priced_ratio=float(priced_ratio),
        whole_step_priced_ratio_estimate=float(whole_step_priced_ratio),
        whole_step_speedup_estimate=float(whole_step_speedup),
        pass_gate=bool(pass_gate),
        decision=decision,
    )


def choose_window(tokens: np.ndarray, seq_len: int, offset: int, vocab: int) -> np.ndarray:
    if tokens.shape[0] < seq_len + 1:
        raise ValueError("not enough tokens for requested sequence length")
    max_start = tokens.shape[0] - seq_len - 1
    start = min(max(offset, 0), max_start)
    window = tokens[start : start + seq_len + 1].copy()
    if window.max(initial=0) >= vocab:
        window %= vocab
    return window


def run_gate(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    tokens = read_v12_tokens(args.dataset, args.max_tokens)
    window = choose_window(tokens, args.seq_len, args.offset, args.vocab)
    params = init_params(args.vocab, args.hidden, args.seed)
    base_ce, states, prev_states, direct_dh = forward(window, params)
    full_grad = recurrent_grad_for_horizon(
        states, prev_states, direct_dh, params["recurrent"], args.seq_len
    )
    params_full = {k: v.copy() for k, v in params.items()}
    params_full["recurrent"] = params["recurrent"] - args.lr * full_grad
    full_update_ce, _, _, _ = forward(window, params_full)
    horizons = parse_horizons(args.horizons, args.seq_len)
    results = [
        evaluate_horizon(
            tokens=window,
            params=params,
            base_ce=base_ce,
            states=states,
            prev_states=prev_states,
            direct_dh=direct_dh,
            full_grad=full_grad,
            full_update_ce=full_update_ce,
            horizon=h,
            lr=args.lr,
            max_residual=args.max_residual,
            min_progress=args.min_progress,
            max_priced_ratio=args.max_priced_ratio,
            debt_price=args.debt_price,
            recurrent_backward_share=args.recurrent_backward_share,
            min_whole_step_speedup=args.min_whole_step_speedup,
        )
        for h in horizons
    ]
    best = min(results, key=lambda item: item.priced_ratio)
    pass_any = any(item.pass_gate for item in results)
    if pass_any:
        decision = "PASS_TAIL_DEBT_IDENTIFYING_GATE"
    elif any(item.actual_progress_ratio > 0.0 for item in results):
        decision = "FAIL_CURRENT_GATE_BUT_IDENTIFYING"
    else:
        decision = "FAIL_NON_IDENTIFYING_OR_HARMFUL"
    elapsed = time.perf_counter() - started
    return {
        "schema": SCHEMA,
        "gate": "r421_tail_cotangent_debt",
        "decision": decision,
        "dataset": str(args.dataset),
        "seq_len": int(args.seq_len),
        "offset": int(args.offset),
        "vocab": int(args.vocab),
        "hidden": int(args.hidden),
        "seed": int(args.seed),
        "lr": float(args.lr),
        "base_ce": float(base_ce),
        "full_update_ce": float(full_update_ce),
        "full_progress": float(base_ce - full_update_ce),
        "best_horizon": asdict(best),
        "horizons": [asdict(item) for item in results],
        "thresholds": {
            "max_residual": float(args.max_residual),
            "min_progress": float(args.min_progress),
            "max_priced_ratio": float(args.max_priced_ratio),
            "debt_price": float(args.debt_price),
            "recurrent_backward_share": float(args.recurrent_backward_share),
            "min_whole_step_speedup": float(args.min_whole_step_speedup),
        },
        "runtime_seconds": float(elapsed),
        "no_training_integration": True,
        "no_gpu": True,
        "interpretation": (
            "Positive only if a truncated recurrent horizon preserves original-CE "
            "progress after charging residual debt and a conservative whole-step "
            "share estimate. This is an identifying gate, not a measured speedup. "
            "A failure means the current cheap gate did not find sparse tail debt; "
            "it does not kill richer mature-checkpoint or v12-GPU cotangent witnesses."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("src/model_v12/test_data.bin"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42113)
    parser.add_argument("--horizons", type=str, default="4,8,16,32,64")
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--max-residual", type=float, default=0.45)
    parser.add_argument("--min-progress", type=float, default=0.85)
    parser.add_argument("--max-priced-ratio", type=float, default=0.90)
    parser.add_argument("--debt-price", type=float, default=0.20)
    parser.add_argument(
        "--recurrent-backward-share",
        type=float,
        default=0.35,
        help="Estimated share of paid step time controlled by recurrent cotangent propagation.",
    )
    parser.add_argument("--min-whole-step-speedup", type=float, default=1.10)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    payload = run_gate(args)
    print(
        "[R421_TAIL_COTANGENT_DEBT_GATE] "
        f"active=1 decision={payload['decision']} "
        f"seq_len={payload['seq_len']} hidden={payload['hidden']} "
        f"best_horizon={payload['best_horizon']['horizon']} "
        f"best_priced_ratio={payload['best_horizon']['priced_ratio']:.6f} "
        f"whole_step_speedup_estimate={payload['best_horizon']['whole_step_speedup_estimate']:.6f} "
        f"no_training_integration=1 no_gpu=1"
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
