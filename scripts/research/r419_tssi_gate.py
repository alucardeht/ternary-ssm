#!/usr/bin/env python3
"""R419 Ternary Sufficient-Statistic Import gate.

This is a CPU-only identifying gate for the R419 TSSI mechanism. It compiles
one-pass original-CE directional moments from a real v12 uint16 token shard into
ternary sign/zero readout commitments, then compares held-out initial CE and
gradient-sign alignment against density-matched random ternary readouts.

The residual-pilot mode adds a small full-softmax residual original-CE readout
pilot. A positive result is still a gate signal only; production GPU training
and full wall-clock pricing remain separate requirements.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DATASET_MAGIC = 0x544F4B4E
DATASET_HEADER_BYTES = 16
DEFAULT_VOCAB = 32064


@dataclass
class CandidateResult:
    feature_mode: str
    density: float
    scale: float
    realized_density: float
    calib_ce: float
    heldout_ce: float
    random_calib_ce_mean: float
    random_heldout_ce_mean: float
    random_heldout_ce_min: float
    sign_alignment: float
    sign_alignment_random_mean: float
    weighted_sign_alignment: float
    weighted_sign_alignment_random_mean: float
    score_abs_mean_selected: float
    score_abs_mean_all: float


@dataclass
class ResidualArmResult:
    arm: str
    init_ce: float
    final_ce: float
    best_ce: float
    best_step: int
    target_ce: float
    target_reached_step: int | None
    target_reached_train_seconds: float | None
    target_reached_total_seconds: float | None
    train_seconds: float
    total_seconds_including_build: float
    trace: list[dict[str, float | int]]


def parse_csv_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


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
    if n <= 2:
        raise ValueError("dataset does not contain enough tokens")
    arr = np.frombuffer(data, dtype="<u2", count=n, offset=offset).astype(np.int64)
    return arr


def split_tokens(tokens: np.ndarray, train_frac: float, calib_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = tokens.shape[0]
    train_end = max(2, int(n * train_frac))
    calib_end = max(train_end + 2, int(n * (train_frac + calib_frac)))
    calib_end = min(calib_end, n - 2)
    return tokens[:train_end], tokens[train_end:calib_end], tokens[calib_end:]


def hash64(x: np.ndarray | int) -> np.ndarray | int:
    # SplitMix64 finalizer, implemented on uint64 arrays for deterministic sparse features.
    z = np.asarray(x, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def make_feature_block(tokens: np.ndarray, dim: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Return sparse signed features X[t, k] and targets y[t]=tokens[t+1]."""
    contexts = tokens[:-1]
    targets = tokens[1:].astype(np.int64)
    n = contexts.shape[0]
    x = np.zeros((n, dim), dtype=np.float32)

    # A bias channel lets the compiler capture unconditional token skew.
    x[:, 0] = 1.0

    def add_hash(seed_values: np.ndarray, salt: int, weight: float = 1.0) -> None:
        h = hash64(seed_values.astype(np.uint64) + np.uint64(salt))
        idx = (h % np.uint64(dim)).astype(np.int64)
        sign = np.where(((h >> np.uint64(63)) & np.uint64(1)) == 0, weight, -weight).astype(np.float32)
        np.add.at(x, (np.arange(n), idx), sign)

    valid_modes = {"token", "bigram", "rolling", "ssm_state"}
    if mode not in valid_modes:
        raise ValueError(f"unknown feature mode {mode!r}; expected one of {sorted(valid_modes)}")

    add_hash(contexts, 0x101, 1.0)

    if mode in {"bigram", "rolling"}:
        prev = np.empty_like(contexts)
        prev[0] = 0
        prev[1:] = contexts[:-1]
        add_hash((prev << 16) ^ contexts, 0x202, 1.0)
        add_hash(prev, 0x303, 0.5)

    if mode == "rolling":
        state = np.zeros(n, dtype=np.uint64)
        acc = np.uint64(0xC0DEC0DEC0DEC0DE)
        for i, tok in enumerate(contexts.astype(np.uint64)):
            acc = hash64(acc ^ (tok + np.uint64(i * 1315423911)))
            state[i] = acc
        add_hash(state, 0x404, 0.75)
        add_hash(state ^ contexts.astype(np.uint64), 0x505, 0.75)

    if mode == "ssm_state":
        dims = np.arange(dim, dtype=np.uint64)
        decay_hash = hash64(dims + np.uint64(0x5150))
        decay = (0.86 + 0.12 * ((decay_hash % np.uint64(997)).astype(np.float32) / 996.0)).astype(np.float32)
        recurrent = np.zeros(dim, dtype=np.float32)
        token_hash = hash64(contexts.astype(np.uint64) + np.uint64(0x6160))
        aux_hash = hash64(contexts.astype(np.uint64) + np.uint64(0x7170))
        token_idx = (token_hash % np.uint64(dim)).astype(np.int64)
        aux_idx = (aux_hash % np.uint64(dim)).astype(np.int64)
        token_sign = np.where(((token_hash >> np.uint64(63)) & np.uint64(1)) == 0, 1.0, -1.0).astype(np.float32)
        aux_sign = np.where(((aux_hash >> np.uint64(63)) & np.uint64(1)) == 0, 1.0, -1.0).astype(np.float32)
        for i in range(n):
            recurrent *= decay
            recurrent[token_idx[i]] += token_sign[i]
            recurrent[aux_idx[i]] += 0.5 * aux_sign[i]
            np.clip(recurrent, -6.0, 6.0, out=recurrent)
            x[i] += recurrent

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.divide(x, np.maximum(norms, 1.0), out=x)
    return x, targets


def compile_scores(x: np.ndarray, y: np.ndarray, vocab: int) -> np.ndarray:
    # Negative gradient of full softmax CE at zero logits:
    # score[v] = sum_{t:y_t=v} x_t - (N / V) * mean_sum_x.
    total_x = x.sum(axis=0, dtype=np.float64).astype(np.float32)
    scores = np.zeros((vocab, x.shape[1]), dtype=np.float32)
    np.add.at(scores, y, x)
    scores -= total_x[None, :] / float(vocab)
    return scores


def realize_ternary(scores: np.ndarray, density: float) -> np.ndarray:
    abs_scores = np.abs(scores)
    if density <= 0:
        return np.zeros_like(scores, dtype=np.int8)
    if density >= 1:
        cutoff = 0.0
    else:
        cutoff = float(np.quantile(abs_scores, 1.0 - density))
    weights = np.zeros_like(scores, dtype=np.int8)
    selected = abs_scores > cutoff
    if not np.any(selected):
        max_idx = np.unravel_index(np.argmax(abs_scores), abs_scores.shape)
        selected[max_idx] = True
    weights[selected] = np.sign(scores[selected]).astype(np.int8)
    return weights


def density_matched_random(shape: tuple[int, int], nnz: int, rng: np.random.Generator) -> np.ndarray:
    total = shape[0] * shape[1]
    nnz = min(max(nnz, 0), total)
    flat = np.zeros(total, dtype=np.int8)
    if nnz > 0:
        idx = rng.choice(total, size=nnz, replace=False)
        flat[idx] = rng.choice(np.array([-1, 1], dtype=np.int8), size=nnz)
    return flat.reshape(shape)


def ce_for_weights(weights: np.ndarray, x: np.ndarray, y: np.ndarray, scale: float, batch: int = 512) -> float:
    w = weights.astype(np.float32, copy=False)
    total = 0.0
    n = x.shape[0]
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        logits = (x[start:stop] @ w.T) * scale
        row_max = logits.max(axis=1)
        shifted = logits - row_max[:, None]
        log_z = row_max + np.log(np.exp(shifted).sum(axis=1))
        total += float(np.sum(log_z - logits[np.arange(stop - start), y[start:stop]]))
    return total / max(n, 1)


def ce_for_logits_weights(weights: np.ndarray, x: np.ndarray, y: np.ndarray, batch: int = 512) -> float:
    total = 0.0
    n = x.shape[0]
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        logits = x[start:stop] @ weights.T
        row_max = logits.max(axis=1)
        shifted = logits - row_max[:, None]
        log_z = row_max + np.log(np.exp(shifted).sum(axis=1))
        total += float(np.sum(log_z - logits[np.arange(stop - start), y[start:stop]]))
    return total / max(n, 1)


def softmax(logits: np.ndarray) -> np.ndarray:
    row_max = logits.max(axis=1, keepdims=True)
    shifted = logits - row_max
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def sign_alignment(weights: np.ndarray, heldout_scores: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float, float]:
    mask = weights != 0
    if not np.any(mask):
        return 0.0, 0.0, 0.0, 0.0
    target_sign = np.sign(heldout_scores[mask])
    nonzero = target_sign != 0
    if not np.any(nonzero):
        return 0.0, 0.0, 0.0, 0.0
    selected_weights = weights[mask][nonzero]
    selected_target = target_sign[nonzero]
    selected_abs = np.abs(heldout_scores[mask][nonzero])
    align = float(np.mean(np.sign(selected_weights) == selected_target))
    weighted = float(np.sum((np.sign(selected_weights) == selected_target) * selected_abs) / np.sum(selected_abs))

    random_sign = rng.choice(np.array([-1, 1], dtype=np.int8), size=selected_weights.shape[0])
    random_align = float(np.mean(random_sign == selected_target))
    random_weighted = float(np.sum((random_sign == selected_target) * selected_abs) / np.sum(selected_abs))
    return align, random_align, weighted, random_weighted


def sign_shuffled(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros_like(weights)
    mask = weights != 0
    signs = weights[mask].copy()
    rng.shuffle(signs)
    out[mask] = signs
    return out


def train_residual_arm(
    arm: str,
    base_logits_weights: np.ndarray,
    train_x: np.ndarray,
    train_y: np.ndarray,
    heldout_x: np.ndarray,
    heldout_y: np.ndarray,
    target_ce: float,
    build_seconds: float,
    steps: int,
    batch_size: int,
    lr: float,
    eval_every: int,
    seed: int,
    weight_decay: float,
) -> ResidualArmResult:
    rng = np.random.default_rng(seed)
    residual = np.zeros_like(base_logits_weights, dtype=np.float32)
    init_ce = ce_for_logits_weights(base_logits_weights, heldout_x, heldout_y)
    best_ce = init_ce
    best_step = 0
    target_reached_step = 0 if init_ce <= target_ce else None
    target_reached_train_seconds = 0.0 if init_ce <= target_ce else None
    target_reached_total_seconds = build_seconds if init_ce <= target_ce else None
    trace: list[dict[str, float | int]] = [{"step": 0, "heldout_ce": init_ce, "train_seconds": 0.0}]
    train_start = time.perf_counter()

    n = train_x.shape[0]
    for step in range(1, steps + 1):
        idx = rng.integers(0, n, size=batch_size)
        xb = train_x[idx]
        yb = train_y[idx]
        weights = base_logits_weights + residual
        logits = xb @ weights.T
        probs = softmax(logits)
        probs[np.arange(batch_size), yb] -= 1.0
        grad = (probs.T @ xb) / float(batch_size)
        if weight_decay > 0.0:
            grad += weight_decay * residual
        residual -= lr * grad.astype(np.float32)

        if step % eval_every == 0 or step == steps:
            elapsed_train = time.perf_counter() - train_start
            ce = ce_for_logits_weights(base_logits_weights + residual, heldout_x, heldout_y)
            trace.append({"step": step, "heldout_ce": ce, "train_seconds": elapsed_train})
            if ce < best_ce:
                best_ce = ce
                best_step = step
            if target_reached_step is None and ce <= target_ce:
                target_reached_step = step
                target_reached_train_seconds = elapsed_train
                target_reached_total_seconds = build_seconds + elapsed_train

    train_seconds = time.perf_counter() - train_start
    final_ce = float(trace[-1]["heldout_ce"])
    total_seconds = train_seconds + build_seconds
    return ResidualArmResult(
        arm=arm,
        init_ce=float(init_ce),
        final_ce=final_ce,
        best_ce=float(best_ce),
        best_step=int(best_step),
        target_ce=float(target_ce),
        target_reached_step=target_reached_step,
        target_reached_train_seconds=target_reached_train_seconds,
        target_reached_total_seconds=target_reached_total_seconds,
        train_seconds=float(train_seconds),
        total_seconds_including_build=float(total_seconds),
        trace=trace,
    )


def run_residual_pilot(
    tokens: np.ndarray,
    vocab: int,
    dim: int,
    feature_mode: str,
    density: float,
    scale: float,
    train_frac: float,
    calib_frac: float,
    steps: int,
    batch_size: int,
    lr: float,
    eval_every: int,
    seed: int,
    weight_decay: float,
) -> dict:
    build_start = time.perf_counter()
    train_tokens, calib_tokens, heldout_tokens = split_tokens(tokens, train_frac, calib_frac)
    train_x, train_y = make_feature_block(train_tokens, dim, feature_mode)
    calib_x, calib_y = make_feature_block(calib_tokens, dim, feature_mode)
    heldout_x, heldout_y = make_feature_block(heldout_tokens, dim, feature_mode)
    scores = compile_scores(train_x, train_y, vocab)
    tssi_ternary = realize_ternary(scores, density)
    tssi_base = tssi_ternary.astype(np.float32) * float(scale)
    tssi_build_seconds = time.perf_counter() - build_start

    rng = np.random.default_rng(seed)
    nnz = int(np.count_nonzero(tssi_ternary))
    random_base = density_matched_random(tssi_ternary.shape, nnz, rng).astype(np.float32) * float(scale)
    shuffled_base = sign_shuffled(tssi_ternary, rng).astype(np.float32) * float(scale)
    zero_base = np.zeros_like(tssi_base, dtype=np.float32)

    # The main target asks whether controls can train their residual to the
    # imported model's starting point. This prices TSSI as an initial-gap removal
    # object before claiming any full training-speed result.
    target_ce = ce_for_logits_weights(tssi_base, heldout_x, heldout_y)
    calib_ce = ce_for_logits_weights(tssi_base, calib_x, calib_y)

    arms = [
        ("tssi", tssi_base, tssi_build_seconds),
        ("zero", zero_base, 0.0),
        ("random", random_base, 0.0),
        ("sign_shuffled", shuffled_base, 0.0),
    ]
    results = [
        train_residual_arm(
            arm=arm,
            base_logits_weights=base,
            train_x=train_x,
            train_y=train_y,
            heldout_x=heldout_x,
            heldout_y=heldout_y,
            target_ce=target_ce,
            build_seconds=build_seconds,
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            eval_every=eval_every,
            seed=seed + i * 1009,
            weight_decay=weight_decay,
        )
        for i, (arm, base, build_seconds) in enumerate(arms)
    ]

    zero = next(r for r in results if r.arm == "zero")
    random = next(r for r in results if r.arm == "random")
    shuffled = next(r for r in results if r.arm == "sign_shuffled")
    tssi = next(r for r in results if r.arm == "tssi")
    control_target_times = [
        r.target_reached_total_seconds
        for r in [zero, random, shuffled]
        if r.target_reached_total_seconds is not None
    ]
    best_control_target_time = min(control_target_times) if control_target_times else None
    target_speedup = (
        best_control_target_time / tssi.target_reached_total_seconds
        if best_control_target_time is not None and tssi.target_reached_total_seconds
        else None
    )
    same_budget_advantage = min(zero.final_ce, random.final_ce, shuffled.final_ce) - tssi.final_ce
    decision = "PASS_RESIDUAL_PILOT_INITIAL_GAP"
    if target_speedup is not None and target_speedup >= 1.10 and same_budget_advantage >= 0.0:
        decision = "PASS_RESIDUAL_PILOT_MATERIAL_HEADSTART"
    elif target_speedup is not None:
        decision = "WEAK_PASS_CONTROL_REACHES_TARGET"
    if same_budget_advantage < 0.0:
        decision = "FAIL_RESIDUAL_PILOT_CONTROL_OVERTAKES"

    return {
        "round": "R419",
        "gate": "tssi_residual_original_ce_pilot",
        "decision": decision,
        "tokens_total": int(tokens.shape[0]),
        "tokens_train": int(train_tokens.shape[0]),
        "tokens_calib": int(calib_tokens.shape[0]),
        "tokens_heldout": int(heldout_tokens.shape[0]),
        "vocab_size": int(vocab),
        "feature_dim": int(dim),
        "feature_mode": feature_mode,
        "density": float(density),
        "scale": float(scale),
        "realized_density": float(nnz / tssi_ternary.size),
        "tssi_build_seconds": float(tssi_build_seconds),
        "target_ce": float(target_ce),
        "tssi_calib_ce": float(calib_ce),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "eval_every": int(eval_every),
        "weight_decay": float(weight_decay),
        "arms": [asdict(r) for r in results],
        "best_control_target_total_seconds": best_control_target_time,
        "target_speedup_vs_best_control": target_speedup,
        "same_budget_ce_advantage": float(same_budget_advantage),
        "interpretation": (
            "CPU-only residual original-CE pilot. It tests whether a fixed TSSI "
            "base removes an initial CE gap that matched zero/random/sign-shuffled "
            "residual training cannot recover within the same short budget. It is "
            "not a production GPU speed measurement."
        ),
    }


def evaluate_mode(
    mode: str,
    train_tokens: np.ndarray,
    calib_tokens: np.ndarray,
    heldout_tokens: np.ndarray,
    vocab: int,
    dim: int,
    densities: Iterable[float],
    scales: Iterable[float],
    random_trials: int,
    seed: int,
) -> tuple[CandidateResult, list[CandidateResult]]:
    train_x, train_y = make_feature_block(train_tokens, dim, mode)
    calib_x, calib_y = make_feature_block(calib_tokens, dim, mode)
    heldout_x, heldout_y = make_feature_block(heldout_tokens, dim, mode)
    scores = compile_scores(train_x, train_y, vocab)
    heldout_scores = compile_scores(heldout_x, heldout_y, vocab)
    rng = np.random.default_rng(seed)

    results: list[CandidateResult] = []
    for density in densities:
        weights = realize_ternary(scores, density)
        nnz = int(np.count_nonzero(weights))
        realized_density = nnz / weights.size
        align, r_align, weighted, r_weighted = sign_alignment(weights, heldout_scores, rng)
        selected_abs = np.abs(scores[weights != 0])
        mean_selected = float(selected_abs.mean()) if selected_abs.size else 0.0
        mean_all = float(np.abs(scores).mean())
        random_weights = [density_matched_random(weights.shape, nnz, rng) for _ in range(random_trials)]
        random_alignments = [sign_alignment(rw, heldout_scores, rng) for rw in random_weights]

        for scale in scales:
            calib_ce = ce_for_weights(weights, calib_x, calib_y, scale)
            heldout_ce = ce_for_weights(weights, heldout_x, heldout_y, scale)
            random_calib = [ce_for_weights(rw, calib_x, calib_y, scale) for rw in random_weights]
            random_heldout = [ce_for_weights(rw, heldout_x, heldout_y, scale) for rw in random_weights]
            results.append(
                CandidateResult(
                    feature_mode=mode,
                    density=float(density),
                    scale=float(scale),
                    realized_density=float(realized_density),
                    calib_ce=float(calib_ce),
                    heldout_ce=float(heldout_ce),
                    random_calib_ce_mean=float(np.mean(random_calib)),
                    random_heldout_ce_mean=float(np.mean(random_heldout)),
                    random_heldout_ce_min=float(np.min(random_heldout)),
                    sign_alignment=float(align),
                    sign_alignment_random_mean=float(np.mean([x[1] for x in random_alignments] + [r_align])),
                    weighted_sign_alignment=float(weighted),
                    weighted_sign_alignment_random_mean=float(np.mean([x[3] for x in random_alignments] + [r_weighted])),
                    score_abs_mean_selected=mean_selected,
                    score_abs_mean_all=mean_all,
                )
            )
    best = min(results, key=lambda r: (r.calib_ce, r.heldout_ce))
    return best, results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=200_000)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--feature-modes", default="token,bigram,rolling")
    parser.add_argument("--densities", default="0.005,0.01,0.02,0.05")
    parser.add_argument("--scales", default="0.02,0.05,0.1,0.2,0.4")
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--calib-frac", type=float, default=0.20)
    parser.add_argument("--random-trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=419)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--residual-pilot", action="store_true")
    parser.add_argument("--pilot-feature-mode", default="bigram")
    parser.add_argument("--pilot-density", type=float, default=0.05)
    parser.add_argument("--pilot-scale", type=float, default=2.4)
    parser.add_argument("--pilot-steps", type=int, default=200)
    parser.add_argument("--pilot-batch-size", type=int, default=1024)
    parser.add_argument("--pilot-lr", type=float, default=0.2)
    parser.add_argument("--pilot-eval-every", type=int, default=20)
    parser.add_argument("--pilot-weight-decay", type=float, default=0.0)
    args = parser.parse_args()

    start = time.perf_counter()
    tokens = read_v12_tokens(args.dataset, args.max_tokens)
    if tokens.max(initial=0) >= args.vocab_size:
        raise ValueError(f"token id {int(tokens.max())} exceeds vocab-size {args.vocab_size}")
    if args.residual_pilot:
        payload = run_residual_pilot(
            tokens=tokens,
            vocab=args.vocab_size,
            dim=args.dim,
            feature_mode=args.pilot_feature_mode,
            density=args.pilot_density,
            scale=args.pilot_scale,
            train_frac=args.train_frac,
            calib_frac=args.calib_frac,
            steps=args.pilot_steps,
            batch_size=args.pilot_batch_size,
            lr=args.pilot_lr,
            eval_every=args.pilot_eval_every,
            seed=args.seed,
            weight_decay=args.pilot_weight_decay,
        )
        payload["dataset"] = str(args.dataset)
        payload["elapsed_seconds"] = time.perf_counter() - start
        text = json.dumps(payload, indent=2, sort_keys=True)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(text + "\n", encoding="utf-8")
        if args.summary_only:
            arms = {
                arm["arm"]: {
                    "init_ce": arm["init_ce"],
                    "final_ce": arm["final_ce"],
                    "best_ce": arm["best_ce"],
                    "target_reached_step": arm["target_reached_step"],
                    "target_reached_total_seconds": arm["target_reached_total_seconds"],
                    "total_seconds_including_build": arm["total_seconds_including_build"],
                }
                for arm in payload["arms"]
            }
            print(
                json.dumps(
                    {
                        "decision": payload["decision"],
                        "dataset": payload["dataset"],
                        "target_ce": payload["target_ce"],
                        "tssi_build_seconds": payload["tssi_build_seconds"],
                        "config": {
                            "feature_mode": payload["feature_mode"],
                            "density": payload["density"],
                            "scale": payload["scale"],
                            "steps": payload["steps"],
                            "batch_size": payload["batch_size"],
                            "lr": payload["lr"],
                        },
                        "arms": arms,
                        "target_speedup_vs_best_control": payload["target_speedup_vs_best_control"],
                        "same_budget_ce_advantage": payload["same_budget_ce_advantage"],
                        "elapsed_seconds": payload["elapsed_seconds"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(text)
        return 0

    train_tokens, calib_tokens, heldout_tokens = split_tokens(tokens, args.train_frac, args.calib_frac)

    modes = [m.strip() for m in args.feature_modes.split(",") if m.strip()]
    densities = parse_csv_floats(args.densities)
    scales = parse_csv_floats(args.scales)

    all_results: list[CandidateResult] = []
    best_by_mode = []
    for i, mode in enumerate(modes):
        best, results = evaluate_mode(
            mode,
            train_tokens,
            calib_tokens,
            heldout_tokens,
            args.vocab_size,
            args.dim,
            densities,
            scales,
            args.random_trials,
            args.seed + i * 1009,
        )
        best_by_mode.append(best)
        all_results.extend(results)

    best_overall = min(best_by_mode, key=lambda r: (r.calib_ce, r.heldout_ce))
    uniform_ce = math.log(args.vocab_size)
    margin_ce = uniform_ce - best_overall.heldout_ce
    random_margin_ce = best_overall.random_heldout_ce_mean - best_overall.heldout_ce
    sign_margin = best_overall.weighted_sign_alignment - best_overall.weighted_sign_alignment_random_mean
    decision = "PASS_PRE_GATE" if margin_ce > 0.01 and random_margin_ce > 0.01 and sign_margin > 0.03 else "FAIL_OR_WEAK_PRE_GATE"

    payload = {
        "round": "R419",
        "gate": "ternary_sufficient_statistic_import_pre_gate",
        "decision": decision,
        "dataset": str(args.dataset),
        "tokens_total": int(tokens.shape[0]),
        "tokens_train": int(train_tokens.shape[0]),
        "tokens_calib": int(calib_tokens.shape[0]),
        "tokens_heldout": int(heldout_tokens.shape[0]),
        "vocab_size": int(args.vocab_size),
        "feature_dim": int(args.dim),
        "uniform_ce": uniform_ce,
        "best": asdict(best_overall),
        "best_by_mode": [asdict(x) for x in best_by_mode],
        "all_results": [asdict(x) for x in all_results],
        "elapsed_seconds": time.perf_counter() - start,
        "interpretation": (
            "CPU-only identifying pre-gate. Positive result means the corpus moment import has "
            "non-random sign and held-out CE signal in this fixed feature map; it is not a "
            "training-speed measurement or production-path validation."
        ),
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if args.summary_only:
        print(
            json.dumps(
                {
                    "decision": payload["decision"],
                    "dataset": payload["dataset"],
                    "tokens": {
                        "total": payload["tokens_total"],
                        "train": payload["tokens_train"],
                        "calib": payload["tokens_calib"],
                        "heldout": payload["tokens_heldout"],
                    },
                    "uniform_ce": payload["uniform_ce"],
                    "best": payload["best"],
                    "elapsed_seconds": payload["elapsed_seconds"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
