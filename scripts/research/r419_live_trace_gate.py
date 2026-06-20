#!/usr/bin/env python3
"""R419 live SSM trace gate for TSSI.

This gate consumes the opt-in v12 R395 hidden-state trace, reconstructs the
sampled token windows from the dataset RNG, and tests whether TSSI sign/zero
commitments built from live SSM hidden states retain held-out CE and residual
original-CE head-start versus zero, density-matched random, and sign-shuffled
controls.

The script is intentionally research-only. It does not change the production
training path and a positive result here is still a live-trace pilot, not a
production GPU speedup.
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

from r419_tssi_gate import (
    ce_for_logits_weights,
    compile_scores,
    density_matched_random,
    parse_csv_floats,
    read_v12_tokens,
    realize_ternary,
    sign_alignment,
    sign_shuffled,
    train_residual_arm,
)


TRACE_MAGIC = b"R395SGT1"
TRACE_VERSION = 1
TRACE_HEADER_STRUCT = struct.Struct("<8sIIIII")
TRACE_RECORD_STRUCT = struct.Struct("<IIfff")
MASK64 = (1 << 64) - 1


@dataclass
class LiveTraceCandidate:
    layer_mode: str
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


def rotl64(x: int, k: int) -> int:
    return ((x << k) & MASK64) | (x >> (64 - k))


def splitmix64_step(seed: int) -> tuple[int, int]:
    seed = (seed + 0x9E3779B97F4A7C15) & MASK64
    z = seed
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return seed, (z ^ (z >> 31)) & MASK64


class DatasetRng:
    def __init__(self, seed: int) -> None:
        if seed == 0:
            seed = 1
        self.state: list[int] = []
        s = seed & MASK64
        for _ in range(4):
            s, value = splitmix64_step(s)
            self.state.append(value)

    def next(self) -> int:
        s = self.state
        result = (rotl64((s[1] * 5) & MASK64, 7) * 9) & MASK64
        t = (s[1] << 17) & MASK64
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = rotl64(s[3], 45)
        for i in range(4):
            s[i] &= MASK64
        return result


def read_trace(path: Path) -> tuple[dict[str, int], list[dict[str, float | int]], np.ndarray]:
    raw = path.read_bytes()
    if len(raw) < TRACE_HEADER_STRUCT.size:
        raise ValueError("trace file is too small for header")

    magic, version, n_layers, batch_size, seq_len, d_state = TRACE_HEADER_STRUCT.unpack_from(raw, 0)
    if magic != TRACE_MAGIC:
        raise ValueError(f"unexpected trace magic {magic!r}")
    if version != TRACE_VERSION:
        raise ValueError(f"unsupported trace version {version}")

    state_count = int(n_layers) * int(batch_size) * int(seq_len) * int(d_state)
    record_bytes = TRACE_RECORD_STRUCT.size + state_count * 4
    records: list[dict[str, float | int]] = []
    state_rows: list[np.ndarray] = []
    offset = TRACE_HEADER_STRUCT.size
    while offset < len(raw):
        if offset + record_bytes > len(raw):
            raise ValueError(
                f"trace has a truncated record at byte {offset}: "
                f"remaining={len(raw) - offset}, record={record_bytes}"
            )
        step, reserved, step_loss, runtime_ms, export_ms = TRACE_RECORD_STRUCT.unpack_from(raw, offset)
        offset += TRACE_RECORD_STRUCT.size
        arr = np.frombuffer(raw, dtype="<f4", count=state_count, offset=offset)
        state_rows.append(arr.reshape((n_layers, batch_size, seq_len, d_state)).copy())
        offset += state_count * 4

        if offset + TRACE_RECORD_STRUCT.size <= len(raw):
            trailing = TRACE_RECORD_STRUCT.unpack_from(raw, offset)
            if trailing[0] == step and trailing[1] == reserved:
                _, _, step_loss, runtime_ms, export_ms = trailing
                offset += TRACE_RECORD_STRUCT.size

        records.append(
            {
                "step": int(step),
                "reserved": int(reserved),
                "step_loss": float(step_loss),
                "runtime_ms": float(runtime_ms),
                "export_ms": float(export_ms),
            }
        )

    if not state_rows:
        raise ValueError("trace contains no records")
    states = np.stack(state_rows, axis=0).astype(np.float32, copy=False)
    header = {
        "n_layers": int(n_layers),
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "d_state": int(d_state),
        "records": int(states.shape[0]),
    }
    return header, records, states


def reconstruct_final_microstep_windows(
    tokens: np.ndarray,
    dataset_seed: int,
    records: Iterable[dict[str, float | int]],
    batch_size: int,
    seq_len: int,
    grad_accum_steps: int,
) -> np.ndarray:
    max_offset = int(tokens.shape[0] - seq_len)
    if max_offset <= 0:
        raise ValueError("dataset is too small for trace seq_len")
    rng = DatasetRng(dataset_seed)
    windows: list[np.ndarray] = []
    for _record in records:
        final_offsets: list[int] = []
        for acc in range(grad_accum_steps):
            offsets = [rng.next() % max_offset for _ in range(batch_size)]
            if acc == grad_accum_steps - 1:
                final_offsets = [int(o) for o in offsets]
        for offset in final_offsets:
            windows.append(tokens[offset : offset + seq_len])
    return np.stack(windows, axis=0).astype(np.int64)


def features_from_trace(states: np.ndarray, layer_mode: str, normalize_rows: bool) -> np.ndarray:
    records, layers, batch, seq_len, d_state = states.shape
    usable = states[:, :, :, :-1, :]
    if layer_mode == "last":
        x = usable[:, -1, :, :, :].reshape(records * batch * (seq_len - 1), d_state)
    elif layer_mode == "mean":
        x = usable.mean(axis=1).reshape(records * batch * (seq_len - 1), d_state)
    elif layer_mode == "all":
        x = np.transpose(usable, (0, 2, 3, 1, 4)).reshape(
            records * batch * (seq_len - 1),
            layers * d_state,
        )
    else:
        raise ValueError("layer_mode must be one of: last, mean, all")

    x = np.nan_to_num(x.astype(np.float32, copy=False), copy=False)
    if normalize_rows:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        np.divide(x, np.maximum(norms, 1.0), out=x)
    return x


def targets_from_windows(windows: np.ndarray) -> np.ndarray:
    return windows[:, 1:].reshape(-1).astype(np.int64)


def split_examples(
    x: np.ndarray,
    y: np.ndarray,
    train_frac: float,
    calib_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = x.shape[0]
    train_end = max(1, int(n * train_frac))
    calib_end = max(train_end + 1, int(n * (train_frac + calib_frac)))
    calib_end = min(calib_end, n - 1)
    if n < 8 or calib_end <= train_end or calib_end >= n:
        raise ValueError(f"not enough trace examples for split: {n}")
    return (
        x[:train_end],
        y[:train_end],
        x[train_end:calib_end],
        y[train_end:calib_end],
        x[calib_end:],
        y[calib_end:],
    )


def split_trace_examples(
    x: np.ndarray,
    y: np.ndarray,
    records: int,
    batch_size: int,
    seq_len: int,
    train_frac: float,
    calib_frac: float,
    split_mode: str,
    record_rotation: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int | str]]:
    if split_mode == "examples":
        train_x, train_y, calib_x, calib_y, heldout_x, heldout_y = split_examples(
            x,
            y,
            train_frac,
            calib_frac,
        )
        return (
            train_x,
            train_y,
            calib_x,
            calib_y,
            heldout_x,
            heldout_y,
            {
                "split_mode": "examples",
                "records": int(records),
                "train_records": -1,
                "calib_records": -1,
                "heldout_records": -1,
                "record_rotation": 0,
                "heldout_record_ids": "",
            },
        )
    if split_mode != "records":
        raise ValueError("split_mode must be one of: examples, records")

    if records < 3:
        raise ValueError(f"not enough trace records for record split: {records}")
    per_record = int(batch_size) * (int(seq_len) - 1)
    expected = int(records) * per_record
    if x.shape[0] != expected or y.shape[0] != expected:
        raise ValueError(
            "record split requires feature/target rows to match trace record layout: "
            f"x={x.shape[0]} y={y.shape[0]} expected={expected}"
        )

    train_end = max(1, int(records * train_frac))
    calib_end = max(train_end + 1, int(records * (train_frac + calib_frac)))
    calib_end = min(calib_end, records - 1)
    if calib_end <= train_end or calib_end >= records:
        raise ValueError(
            f"not enough trace records for split: records={records} "
            f"train_end={train_end} calib_end={calib_end}"
        )

    feature_dim = int(x.shape[1])
    xr = x.reshape(records, per_record, feature_dim)
    yr = y.reshape(records, per_record)
    record_ids = np.arange(records)
    rotation = int(record_rotation) % int(records)
    if rotation:
        xr = np.roll(xr, -rotation, axis=0)
        yr = np.roll(yr, -rotation, axis=0)
        record_ids = np.roll(record_ids, -rotation)

    def flatten_features(a: np.ndarray) -> np.ndarray:
        return a.reshape(-1, feature_dim)

    def flatten_targets(a: np.ndarray) -> np.ndarray:
        return a.reshape(-1)

    return (
        flatten_features(xr[:train_end]),
        flatten_targets(yr[:train_end]),
        flatten_features(xr[train_end:calib_end]),
        flatten_targets(yr[train_end:calib_end]),
        flatten_features(xr[calib_end:]),
        flatten_targets(yr[calib_end:]),
        {
            "split_mode": "records",
            "records": int(records),
            "train_records": int(train_end),
            "calib_records": int(calib_end - train_end),
            "heldout_records": int(records - calib_end),
            "record_rotation": int(rotation),
            "heldout_record_ids": ",".join(str(int(v)) for v in record_ids[calib_end:]),
        },
    )


def evaluate_live_trace(
    train_x: np.ndarray,
    train_y: np.ndarray,
    calib_x: np.ndarray,
    calib_y: np.ndarray,
    heldout_x: np.ndarray,
    heldout_y: np.ndarray,
    vocab: int,
    layer_mode: str,
    densities: list[float],
    scales: list[float],
    random_trials: int,
    seed: int,
) -> tuple[list[LiveTraceCandidate], LiveTraceCandidate, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    build_start = time.perf_counter()
    scores = compile_scores(train_x, train_y, vocab)
    heldout_scores = compile_scores(heldout_x, heldout_y, vocab)
    compile_seconds = time.perf_counter() - build_start

    candidates: list[LiveTraceCandidate] = []
    for density in densities:
        weights = realize_ternary(scores, density)
        nnz = int(np.count_nonzero(weights))
        realized_density = nnz / max(weights.size, 1)
        random_weights = [density_matched_random(weights.shape, nnz, rng) for _ in range(random_trials)]
        align, random_align, weighted, random_weighted = sign_alignment(weights, heldout_scores, rng)
        selected = np.abs(scores[weights != 0])
        for scale in scales:
            calib_ce = ce_for_logits_weights(weights.astype(np.float32) * float(scale), calib_x, calib_y)
            heldout_ce = ce_for_logits_weights(weights.astype(np.float32) * float(scale), heldout_x, heldout_y)
            random_calib = [
                ce_for_logits_weights(rw.astype(np.float32) * float(scale), calib_x, calib_y)
                for rw in random_weights
            ]
            random_heldout = [
                ce_for_logits_weights(rw.astype(np.float32) * float(scale), heldout_x, heldout_y)
                for rw in random_weights
            ]
            candidates.append(
                LiveTraceCandidate(
                    layer_mode=layer_mode,
                    density=float(density),
                    scale=float(scale),
                    realized_density=float(realized_density),
                    calib_ce=float(calib_ce),
                    heldout_ce=float(heldout_ce),
                    random_calib_ce_mean=float(np.mean(random_calib)),
                    random_heldout_ce_mean=float(np.mean(random_heldout)),
                    random_heldout_ce_min=float(np.min(random_heldout)),
                    sign_alignment=float(align),
                    sign_alignment_random_mean=float(random_align),
                    weighted_sign_alignment=float(weighted),
                    weighted_sign_alignment_random_mean=float(random_weighted),
                    score_abs_mean_selected=float(np.mean(selected)) if selected.size else 0.0,
                    score_abs_mean_all=float(np.mean(np.abs(scores))),
                )
            )

    best = min(candidates, key=lambda c: (c.calib_ce, c.heldout_ce))
    best_weights = realize_ternary(scores, best.density).astype(np.float32) * float(best.scale)
    return candidates, best, best_weights, float(compile_seconds)


def run_residual_pilot(
    train_x: np.ndarray,
    train_y: np.ndarray,
    heldout_x: np.ndarray,
    heldout_y: np.ndarray,
    tssi_base: np.ndarray,
    build_seconds: float,
    steps: int,
    batch_size: int,
    lr: float,
    eval_every: int,
    seed: int,
    weight_decay: float,
) -> dict:
    rng = np.random.default_rng(seed)
    nnz = int(np.count_nonzero(tssi_base))
    random_base = density_matched_random(tssi_base.shape, nnz, rng).astype(np.float32) * float(
        np.max(np.abs(tssi_base)) if nnz else 1.0
    )
    shuffled_base = sign_shuffled(np.sign(tssi_base).astype(np.int8), rng).astype(np.float32) * float(
        np.max(np.abs(tssi_base)) if nnz else 1.0
    )
    zero_base = np.zeros_like(tssi_base, dtype=np.float32)
    target_ce = ce_for_logits_weights(tssi_base, heldout_x, heldout_y)

    arms = [
        ("tssi", tssi_base, build_seconds),
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
            build_seconds=arm_build_seconds,
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            eval_every=eval_every,
            seed=seed + i * 1009,
            weight_decay=weight_decay,
        )
        for i, (arm, base, arm_build_seconds) in enumerate(arms)
    ]
    tssi = next(r for r in results if r.arm == "tssi")
    controls = [r for r in results if r.arm != "tssi"]
    control_target_times = [r.target_reached_total_seconds for r in controls if r.target_reached_total_seconds is not None]
    best_control_target_time = min(control_target_times) if control_target_times else None
    target_speedup = (
        best_control_target_time / tssi.target_reached_total_seconds
        if best_control_target_time is not None and tssi.target_reached_total_seconds
        else None
    )
    same_budget_advantage = min(r.final_ce for r in controls) - tssi.final_ce
    decision = "FAIL_RESIDUAL_PILOT"
    eps = 1.0e-9
    if target_speedup is not None and target_speedup >= 1.10 and same_budget_advantage > eps:
        decision = "PASS_RESIDUAL_PILOT_MATERIAL_HEADSTART"
    elif tssi.init_ce < min(r.init_ce for r in controls) - eps and same_budget_advantage > eps:
        decision = "PASS_RESIDUAL_PILOT_INITIAL_AND_FINAL_CE"
    elif tssi.init_ce < min(r.init_ce for r in controls) - eps:
        decision = "WEAK_INITIAL_CE_ONLY"

    return {
        "gate": "r419_live_trace_residual_pilot",
        "decision": decision,
        "target_ce": float(target_ce),
        "target_speedup_vs_best_control": float(target_speedup) if target_speedup is not None else None,
        "same_budget_ce_advantage": float(same_budget_advantage),
        "best_control_target_total_seconds": (
            float(best_control_target_time) if best_control_target_time is not None else None
        ),
        "arms": [asdict(r) for r in results],
    }


def summarize_decision(best: LiveTraceCandidate, examples: int, residual: dict | None) -> str:
    signal = (
        best.heldout_ce < best.random_heldout_ce_min
        and best.weighted_sign_alignment > best.weighted_sign_alignment_random_mean
    )
    if examples < 4096:
        return "SMOKE_ONLY_SIGNAL" if signal else "SMOKE_ONLY_NO_SIGNAL"
    if residual is not None:
        if residual["decision"] == "PASS_RESIDUAL_PILOT_MATERIAL_HEADSTART":
            return "PASS_LIVE_TRACE_RESIDUAL_HEADSTART"
        if signal:
            return "PARTIAL_SIGNAL_LIVE_TRACE_CE_ONLY"
        return "FAIL_LIVE_TRACE_GATE"
    return "PASS_LIVE_TRACE_PRE_GATE" if signal else "FAIL_LIVE_TRACE_PRE_GATE"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-seed", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--layer-mode", choices=["last", "mean", "all"], default="last")
    parser.add_argument("--split-mode", choices=["examples", "records"], default="examples")
    parser.add_argument("--record-rotation", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--calib-frac", type=float, default=0.20)
    parser.add_argument("--densities", default="0.005,0.01,0.02,0.05")
    parser.add_argument("--scales", default="0.8,1.6,2.4,3.2")
    parser.add_argument("--random-trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--no-normalize-rows", action="store_true")
    parser.add_argument("--residual-pilot", action="store_true")
    parser.add_argument("--pilot-steps", type=int, default=300)
    parser.add_argument("--pilot-batch-size", type=int, default=512)
    parser.add_argument("--pilot-lr", type=float, default=0.5)
    parser.add_argument("--pilot-eval-every", type=int, default=25)
    parser.add_argument("--pilot-weight-decay", type=float, default=0.0)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    trace_start = time.perf_counter()
    header, records, states = read_trace(args.trace)
    tokens = read_v12_tokens(args.dataset, None)
    windows = reconstruct_final_microstep_windows(
        tokens=tokens,
        dataset_seed=args.dataset_seed,
        records=records,
        batch_size=header["batch_size"],
        seq_len=header["seq_len"],
        grad_accum_steps=args.grad_accum_steps,
    )
    x = features_from_trace(states, args.layer_mode, normalize_rows=not args.no_normalize_rows)
    y = targets_from_windows(windows)
    parse_seconds = time.perf_counter() - trace_start
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"feature/target mismatch: {x.shape[0]} != {y.shape[0]}")
    if int(np.max(y)) >= args.vocab_size:
        raise ValueError(f"target token exceeds vocab-size {args.vocab_size}: max={int(np.max(y))}")

    train_x, train_y, calib_x, calib_y, heldout_x, heldout_y, split_meta = split_trace_examples(
        x,
        y,
        records=len(records),
        batch_size=header["batch_size"],
        seq_len=header["seq_len"],
        train_frac=args.train_frac,
        calib_frac=args.calib_frac,
        split_mode=args.split_mode,
        record_rotation=args.record_rotation,
    )
    densities = parse_csv_floats(args.densities)
    scales = parse_csv_floats(args.scales)
    candidates, best, best_weights, compile_seconds = evaluate_live_trace(
        train_x=train_x,
        train_y=train_y,
        calib_x=calib_x,
        calib_y=calib_y,
        heldout_x=heldout_x,
        heldout_y=heldout_y,
        vocab=args.vocab_size,
        layer_mode=args.layer_mode,
        densities=densities,
        scales=scales,
        random_trials=args.random_trials,
        seed=args.seed,
    )

    trace_runtime_seconds = float(sum(float(r["runtime_ms"]) for r in records) / 1000.0)
    trace_export_seconds = float(sum(float(r["export_ms"]) for r in records) / 1000.0)
    tssi_build_seconds = parse_seconds + compile_seconds + trace_runtime_seconds + trace_export_seconds
    residual = None
    if args.residual_pilot:
        residual = run_residual_pilot(
            train_x=train_x,
            train_y=train_y,
            heldout_x=heldout_x,
            heldout_y=heldout_y,
            tssi_base=best_weights,
            build_seconds=tssi_build_seconds,
            steps=args.pilot_steps,
            batch_size=args.pilot_batch_size,
            lr=args.pilot_lr,
            eval_every=args.pilot_eval_every,
            seed=args.seed,
            weight_decay=args.pilot_weight_decay,
        )

    decision = summarize_decision(best, int(x.shape[0]), residual)
    payload = {
        "gate": "r419_live_ssm_trace_tssi_gate",
        "decision": decision,
        "scope_warning": (
            "Live v12 hidden-state trace gate. This uses the production GPU path to capture SSM "
            "hidden states, but the import and residual pilot are CPU-side research evaluation. "
            "It is not a production training speedup."
        ),
        "trace": {
            "path": str(args.trace),
            **header,
            "runtime_seconds": trace_runtime_seconds,
            "export_seconds": trace_export_seconds,
            "step_losses": [float(r["step_loss"]) for r in records],
        },
        "dataset": {
            "path": str(args.dataset),
            "dataset_seed": int(args.dataset_seed),
            "tokens": int(tokens.shape[0]),
            "vocab_size": int(args.vocab_size),
            "grad_accum_steps": int(args.grad_accum_steps),
        },
        "features": {
            "layer_mode": args.layer_mode,
            "normalize_rows": not args.no_normalize_rows,
            **split_meta,
            "examples": int(x.shape[0]),
            "dim": int(x.shape[1]),
            "train_examples": int(train_x.shape[0]),
            "calib_examples": int(calib_x.shape[0]),
            "heldout_examples": int(heldout_x.shape[0]),
        },
        "pricing": {
            "parse_seconds": float(parse_seconds),
            "compile_seconds": float(compile_seconds),
            "trace_runtime_seconds": trace_runtime_seconds,
            "trace_export_seconds": trace_export_seconds,
            "tssi_build_seconds_charged": float(tssi_build_seconds),
        },
        "best": asdict(best),
        "candidates": [asdict(c) for c in candidates],
        "residual_pilot": residual,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if args.summary_only:
        best_payload = payload["best"]
        print(json.dumps({"decision": decision, "best": best_payload, "residual_pilot": residual}, indent=2))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
