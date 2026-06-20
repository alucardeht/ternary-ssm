#!/usr/bin/env python3
"""R420 online TSSI compiler gate.

This opt-in research gate tests the first R420 landing target: compile a TSSI
artifact while streaming the corpus, instead of exporting an offline hidden-state
trace and parsing it later. The script deliberately keeps the previous residual
original-CE control structure from R419, but replaces the build path with an
online accumulator that consumes feature rows once and stores only sufficient
statistics:

    score[y] += phi(context)
    total_phi += phi(context)
    adjusted_score = score - total_phi / vocab

The result is compared against the offline compiler for probe validity, then
zero, density-matched random, and sign-shuffled residual starts are run with the
online build cost charged to TSSI.

This is still a research gate. It does not claim a production GPU speedup unless
the live GPU path later exposes the same accumulator without hidden trace export.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from r419_tssi_gate import (  # noqa: E402
    DEFAULT_VOCAB,
    ce_for_logits_weights,
    compile_scores,
    density_matched_random,
    hash64,
    make_feature_block,
    read_v12_tokens,
    realize_ternary,
    sign_alignment,
    sign_shuffled,
    split_tokens,
    train_residual_arm,
)


UINT64_MASK = (1 << 64) - 1


@dataclass
class OnlineFeatureState:
    rolling_acc: int
    recurrent: np.ndarray | None
    decay: np.ndarray | None


@dataclass
class OnlineBuildResult:
    scores: np.ndarray
    raw_scores: np.ndarray
    total_features: np.ndarray
    examples: int
    accumulate_seconds: float
    realize_seconds: float
    realized_density: float
    nnz: int


@dataclass
class CompactAccumulatorResult:
    signed_state_sum: np.ndarray
    abs_state_sum: np.ndarray
    stats: np.ndarray
    selected_dims: np.ndarray
    selection_seconds: float


def hash64_scalar(x: int) -> int:
    return int(np.asarray(hash64(np.uint64(x & UINT64_MASK))).item()) & UINT64_MASK


def new_online_feature_state(dim: int, mode: str) -> OnlineFeatureState:
    decay = None
    recurrent = None
    if mode == "ssm_state":
        dims = np.arange(dim, dtype=np.uint64)
        decay_hash = hash64(dims + np.uint64(0x5150))
        decay = (0.86 + 0.12 * ((decay_hash % np.uint64(997)).astype(np.float32) / 996.0)).astype(np.float32)
        recurrent = np.zeros(dim, dtype=np.float32)
    return OnlineFeatureState(
        rolling_acc=0xC0DEC0DEC0DEC0DE,
        recurrent=recurrent,
        decay=decay,
    )


def add_hash_features(x: np.ndarray, rows: np.ndarray, seed_values: np.ndarray, dim: int, salt: int, weight: float) -> None:
    h = hash64(seed_values.astype(np.uint64) + np.uint64(salt))
    idx = (h % np.uint64(dim)).astype(np.int64)
    sign = np.where(((h >> np.uint64(63)) & np.uint64(1)) == 0, weight, -weight).astype(np.float32)
    np.add.at(x, (rows, idx), sign)


def make_online_feature_block(
    tokens: np.ndarray,
    row_start: int,
    row_stop: int,
    dim: int,
    mode: str,
    state: OnlineFeatureState,
) -> tuple[np.ndarray, np.ndarray]:
    if row_stop <= row_start:
        return np.zeros((0, dim), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    valid_modes = {"token", "bigram", "rolling", "ssm_state"}
    if mode not in valid_modes:
        raise ValueError(f"unknown feature mode {mode!r}; expected one of {sorted(valid_modes)}")

    contexts = tokens[row_start:row_stop].astype(np.int64)
    targets = tokens[row_start + 1 : row_stop + 1].astype(np.int64)
    n = contexts.shape[0]
    rows = np.arange(n)
    x = np.zeros((n, dim), dtype=np.float32)
    x[:, 0] = 1.0

    add_hash_features(x, rows, contexts, dim, 0x101, 1.0)

    if mode in {"bigram", "rolling"}:
        prev = np.empty_like(contexts)
        if row_start == 0:
            prev[0] = 0
            if n > 1:
                prev[1:] = contexts[:-1]
        else:
            prev[:] = tokens[row_start - 1 : row_stop - 1]
        add_hash_features(x, rows, (prev << 16) ^ contexts, dim, 0x202, 1.0)
        add_hash_features(x, rows, prev, dim, 0x303, 0.5)

    if mode == "rolling":
        state_values = np.zeros(n, dtype=np.uint64)
        acc = state.rolling_acc
        for i, tok in enumerate(contexts.astype(np.uint64)):
            global_i = row_start + i
            mixed = acc ^ ((int(tok) + global_i * 1315423911) & UINT64_MASK)
            acc = hash64_scalar(mixed)
            state_values[i] = np.uint64(acc)
        state.rolling_acc = acc
        add_hash_features(x, rows, state_values, dim, 0x404, 0.75)
        add_hash_features(x, rows, state_values ^ contexts.astype(np.uint64), dim, 0x505, 0.75)

    if mode == "ssm_state":
        if state.recurrent is None or state.decay is None:
            raise ValueError("ssm_state feature state was not initialized")
        token_hash = hash64(contexts.astype(np.uint64) + np.uint64(0x6160))
        aux_hash = hash64(contexts.astype(np.uint64) + np.uint64(0x7170))
        token_idx = (token_hash % np.uint64(dim)).astype(np.int64)
        aux_idx = (aux_hash % np.uint64(dim)).astype(np.int64)
        token_sign = np.where(((token_hash >> np.uint64(63)) & np.uint64(1)) == 0, 1.0, -1.0).astype(np.float32)
        aux_sign = np.where(((aux_hash >> np.uint64(63)) & np.uint64(1)) == 0, 1.0, -1.0).astype(np.float32)
        recurrent = state.recurrent
        for i in range(n):
            recurrent *= state.decay
            recurrent[token_idx[i]] += token_sign[i]
            recurrent[aux_idx[i]] += 0.5 * aux_sign[i]
            np.clip(recurrent, -6.0, 6.0, out=recurrent)
            x[i] += recurrent

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.divide(x, np.maximum(norms, 1.0), out=x)
    return x, targets


def compile_online_scores(
    train_tokens: np.ndarray,
    vocab: int,
    dim: int,
    feature_mode: str,
    density: float,
    chunk_rows: int,
) -> OnlineBuildResult:
    raw_scores = np.zeros((vocab, dim), dtype=np.float32)
    total_features = np.zeros(dim, dtype=np.float32)
    state = new_online_feature_state(dim, feature_mode)
    examples = max(0, int(train_tokens.shape[0]) - 1)

    accumulate_start = time.perf_counter()
    for row_start in range(0, examples, chunk_rows):
        row_stop = min(row_start + chunk_rows, examples)
        x, y = make_online_feature_block(train_tokens, row_start, row_stop, dim, feature_mode, state)
        total_features += x.sum(axis=0, dtype=np.float64).astype(np.float32)
        np.add.at(raw_scores, y, x)
    accumulate_seconds = time.perf_counter() - accumulate_start

    realize_start = time.perf_counter()
    scores = raw_scores - total_features[None, :] / float(vocab)
    ternary = realize_ternary(scores, density)
    realize_seconds = time.perf_counter() - realize_start
    nnz = int(np.count_nonzero(ternary))
    realized_density = float(nnz / ternary.size)

    return OnlineBuildResult(
        scores=scores,
        raw_scores=raw_scores,
        total_features=total_features,
        examples=examples,
        accumulate_seconds=float(accumulate_seconds),
        realize_seconds=float(realize_seconds),
        realized_density=realized_density,
        nnz=nnz,
    )


def compact_score_accumulator(
    scores: np.ndarray,
    examples: int,
    active_fraction: float,
) -> CompactAccumulatorResult:
    start = time.perf_counter()
    if not 0.0 < active_fraction <= 1.0:
        raise ValueError("--active-fraction must be in (0, 1]")
    signed_state_sum = scores.sum(axis=0, dtype=np.float64).astype(np.float32)
    abs_state_sum = np.abs(scores).sum(axis=0, dtype=np.float64).astype(np.float32)
    d_state = int(scores.shape[1])
    active_dims = max(1, min(d_state, int(math.ceil(d_state * active_fraction))))
    order = np.argsort(-abs_state_sum, kind="stable")
    selected_dims = np.sort(order[:active_dims].astype(np.int64))
    stats = np.array(
        [
            float(examples),
            float(np.sum(signed_state_sum, dtype=np.float64)),
            float(np.sum(abs_state_sum, dtype=np.float64)),
            float(np.sum(signed_state_sum.astype(np.float64) ** 2)),
        ],
        dtype=np.float32,
    )
    return CompactAccumulatorResult(
        signed_state_sum=signed_state_sum,
        abs_state_sum=abs_state_sum,
        stats=stats,
        selected_dims=selected_dims,
        selection_seconds=float(time.perf_counter() - start),
    )


def apply_dim_mask(weights: np.ndarray, selected_dims: np.ndarray) -> np.ndarray:
    masked = np.zeros_like(weights, dtype=np.float32)
    masked[:, selected_dims] = weights[:, selected_dims]
    return masked


def load_live_selector(path: Path, dim: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_dims = payload.get("selected_dims")
    if not isinstance(raw_dims, list) or not raw_dims:
        raise ValueError("--selector-json must contain a non-empty selected_dims list")
    selected_dims = np.asarray(raw_dims, dtype=np.int64)
    if selected_dims.ndim != 1:
        raise ValueError("selected_dims must be one-dimensional")
    if np.any(selected_dims < 0) or np.any(selected_dims >= dim):
        raise ValueError("selected_dims contains an index outside --dim")
    if np.unique(selected_dims).shape[0] != selected_dims.shape[0]:
        raise ValueError("selected_dims contains duplicates")
    payload["_selected_dims_np"] = np.sort(selected_dims)
    return payload


def density_matched_random_on_dims(
    shape: tuple[int, int],
    selected_dims: np.ndarray,
    nnz: int,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    local = density_matched_random((shape[0], int(selected_dims.shape[0])), nnz, rng).astype(np.float32)
    out[:, selected_dims] = local
    return out


def build_eval_blocks(
    train_tokens: np.ndarray,
    calib_tokens: np.ndarray,
    heldout_tokens: np.ndarray,
    dim: int,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    start = time.perf_counter()
    train_x, train_y = make_feature_block(train_tokens, dim, feature_mode)
    calib_x, calib_y = make_feature_block(calib_tokens, dim, feature_mode)
    heldout_x, heldout_y = make_feature_block(heldout_tokens, dim, feature_mode)
    return train_x, train_y, calib_x, calib_y, heldout_x, heldout_y, float(time.perf_counter() - start)


def validate_online_scores(
    online_scores: np.ndarray,
    train_x: np.ndarray,
    train_y: np.ndarray,
    vocab: int,
) -> dict:
    start = time.perf_counter()
    offline_scores = compile_scores(train_x, train_y, vocab)
    diff = online_scores - offline_scores
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    mean_abs = float(np.mean(np.abs(diff))) if diff.size else 0.0
    return {
        "offline_equivalence_max_abs": max_abs,
        "offline_equivalence_mean_abs": mean_abs,
        "verify_seconds": float(time.perf_counter() - start),
    }


def summarize_decision(payload: dict, equivalence_tol: float) -> str:
    max_abs = float(payload["validation"]["offline_equivalence_max_abs"])
    residual = payload.get("residual_pilot")
    if not math.isfinite(max_abs) or max_abs > equivalence_tol:
        return "INVALID_ONLINE_ACCUMULATOR_EQUIVALENCE"
    if residual is None:
        return "PASS_ONLINE_ACCUMULATOR_EQUIVALENCE_ONLY"
    if residual["decision"].startswith("PASS"):
        return "PASS_ONLINE_TSSI_RESIDUAL_GATE"
    if residual["decision"].startswith("WEAK"):
        return "WEAK_ONLINE_TSSI_RESIDUAL_GATE"
    return "FAIL_ONLINE_TSSI_RESIDUAL_GATE"


def run_gate(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    tokens = read_v12_tokens(args.dataset, args.max_tokens)
    train_tokens, calib_tokens, heldout_tokens = split_tokens(tokens, args.train_frac, args.calib_frac)

    print(
        "[ONLINE_TSSI_COMPILER] active=1 mode=in_process_stream "
        f"source=v12_token_shard feature_mode={args.feature_mode} "
        f"tokens={tokens.shape[0]} dim={args.dim} density={args.density} "
        "no_offline_trace_export=1 no_hidden_trace_file=1 no_training_default_change=1",
        flush=True,
    )

    online = compile_online_scores(
        train_tokens=train_tokens,
        vocab=args.vocab,
        dim=args.dim,
        feature_mode=args.feature_mode,
        density=args.density,
        chunk_rows=args.chunk_rows,
    )
    tssi_ternary = realize_ternary(online.scores, args.density)
    tssi_base = tssi_ternary.astype(np.float32) * float(args.scale)

    train_x, train_y, calib_x, calib_y, heldout_x, heldout_y, feature_seconds = build_eval_blocks(
        train_tokens, calib_tokens, heldout_tokens, args.dim, args.feature_mode
    )
    validation = validate_online_scores(online.scores, train_x, train_y, args.vocab)
    heldout_scores = compile_scores(heldout_x, heldout_y, args.vocab)
    align, rand_align, weighted_align, rand_weighted_align = sign_alignment(tssi_ternary, heldout_scores, rng)

    target_ce = ce_for_logits_weights(tssi_base, heldout_x, heldout_y)
    calib_ce = ce_for_logits_weights(tssi_base, calib_x, calib_y)

    residual = None
    if args.residual_pilot:
        compact = None
        selector_payload = None
        selected_dims = None
        selection_seconds = 0.0
        compact_stats = None
        if args.selector_json is not None:
            selector_payload = load_live_selector(args.selector_json, args.dim)
            selected_dims = selector_payload["_selected_dims_np"]
            compact_stats = selector_payload.get("stats")
            tssi_eval = apply_dim_mask(tssi_base, selected_dims)
            tssi_ternary_eval = apply_dim_mask(tssi_ternary.astype(np.float32), selected_dims)
        elif args.compact_accumulator_pricing:
            compact = compact_score_accumulator(online.scores, online.examples, args.active_fraction)
            selected_dims = compact.selected_dims
            selection_seconds = compact.selection_seconds
            compact_stats = {
                "count": float(compact.stats[0]),
                "signed_sum": float(compact.stats[1]),
                "abs_sum": float(compact.stats[2]),
                "signed_square_sum": float(compact.stats[3]),
                "abs_top_mean": float(np.mean(compact.abs_state_sum[compact.selected_dims])),
                "abs_all_mean": float(np.mean(compact.abs_state_sum)),
            }
            tssi_eval = apply_dim_mask(tssi_base, selected_dims)
            tssi_ternary_eval = apply_dim_mask(tssi_ternary.astype(np.float32), selected_dims)
        else:
            tssi_eval = tssi_base
            tssi_ternary_eval = tssi_ternary.astype(np.float32)
        nnz = int(np.count_nonzero(tssi_ternary))
        eval_nnz = int(np.count_nonzero(tssi_ternary_eval))
        if selected_dims is not None:
            random_base = density_matched_random_on_dims(
                tssi_ternary_eval.shape, selected_dims, eval_nnz, rng
            ) * float(args.scale)
        else:
            random_base = density_matched_random(tssi_ternary_eval.shape, eval_nnz, rng).astype(np.float32) * float(args.scale)
        shuffled_base = sign_shuffled(tssi_ternary_eval, rng).astype(np.float32) * float(args.scale)
        zero_base = np.zeros_like(tssi_eval, dtype=np.float32)
        build_seconds = (
            online.accumulate_seconds
            + online.realize_seconds
            + float(validation["verify_seconds"])
            + selection_seconds
        )
        arms = [
            ("tssi", tssi_eval, build_seconds),
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
                target_ce=ce_for_logits_weights(tssi_eval, heldout_x, heldout_y),
                build_seconds=cost,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                eval_every=args.eval_every,
                seed=args.seed + i * 1009,
                weight_decay=args.weight_decay,
            )
            for i, (arm, base, cost) in enumerate(arms)
        ]
        tssi = next(r for r in results if r.arm == "tssi")
        controls = [r for r in results if r.arm != "tssi"]
        control_target_times = [
            r.target_reached_total_seconds
            for r in controls
            if r.target_reached_total_seconds is not None
        ]
        best_control_target_time = min(control_target_times) if control_target_times else None
        target_speedup = (
            best_control_target_time / tssi.target_reached_total_seconds
            if best_control_target_time is not None and tssi.target_reached_total_seconds
            else None
        )
        same_budget_advantage = min(r.final_ce for r in controls) - tssi.final_ce
        residual_decision = "PASS_RESIDUAL_PILOT_INITIAL_GAP"
        if target_speedup is not None and target_speedup >= 1.10 and same_budget_advantage >= 0.0:
            residual_decision = "PASS_RESIDUAL_PILOT_MATERIAL_HEADSTART"
        elif target_speedup is not None:
            residual_decision = "WEAK_PASS_CONTROL_REACHES_TARGET"
        if same_budget_advantage < 0.0:
            residual_decision = "FAIL_RESIDUAL_PILOT_CONTROL_OVERTAKES"
        residual = {
            "gate": (
                "online_tssi_live_selector_residual_pilot"
                if args.selector_json is not None
                else "online_tssi_compact_accumulator_residual_pilot"
                if selected_dims is not None
                else "online_tssi_residual_original_ce_pilot"
            ),
            "decision": residual_decision,
            "target_ce": float(ce_for_logits_weights(tssi_eval, heldout_x, heldout_y)),
            "full_tssi_target_ce": float(target_ce),
            "compact_accumulator_pricing": bool(args.compact_accumulator_pricing or args.selector_json is not None),
            "selector_json": str(args.selector_json) if args.selector_json is not None else None,
            "selector_source": (
                selector_payload.get("source")
                if selector_payload is not None
                else "proxy_compact_score_accumulator"
                if compact is not None
                else None
            ),
            "active_fraction": (
                float(selector_payload.get("active_fraction", args.active_fraction))
                if selector_payload is not None
                else float(args.active_fraction)
                if selected_dims is not None
                else None
            ),
            "active_dims": int(selected_dims.shape[0]) if selected_dims is not None else None,
            "active_nnz": int(eval_nnz),
            "full_nnz": int(nnz),
            "selection_seconds": float(selection_seconds),
            "compact_accumulator_stats": compact_stats,
            "arms": [asdict(r) for r in results],
            "best_control_target_total_seconds": best_control_target_time,
            "target_speedup_vs_best_control": target_speedup,
            "same_budget_ce_advantage": float(same_budget_advantage),
        }

    payload = {
        "round": "R420",
        "gate": "online_tssi_compiler_gate",
        "dataset": str(args.dataset),
        "tokens_total": int(tokens.shape[0]),
        "tokens_train": int(train_tokens.shape[0]),
        "tokens_calib": int(calib_tokens.shape[0]),
        "tokens_heldout": int(heldout_tokens.shape[0]),
        "vocab_size": int(args.vocab),
        "feature_dim": int(args.dim),
        "feature_mode": args.feature_mode,
        "density": float(args.density),
        "scale": float(args.scale),
        "chunk_rows": int(args.chunk_rows),
        "online_build": {
            "examples": int(online.examples),
            "accumulate_seconds": float(online.accumulate_seconds),
            "realize_seconds": float(online.realize_seconds),
            "realized_density": float(online.realized_density),
            "nnz": int(online.nnz),
            "no_offline_trace_export": True,
            "no_hidden_trace_file": True,
        },
        "feature_materialization_seconds_for_residual_eval": feature_seconds,
        "validation": validation,
        "tssi_calib_ce": float(calib_ce),
        "tssi_heldout_ce": float(target_ce),
        "sign_alignment": float(align),
        "sign_alignment_random": float(rand_align),
        "weighted_sign_alignment": float(weighted_align),
        "weighted_sign_alignment_random": float(rand_weighted_align),
        "residual_pilot": residual,
        "interpretation": (
            "Opt-in online sufficient-statistic compiler gate. The TSSI build "
            "path streams feature rows into score moments and does not export "
            "or parse an offline hidden-state trace. Residual pilot costs charge "
            "accumulation, realization, and equivalence verification to TSSI. "
            "When compact_accumulator_pricing is enabled, the residual pilot "
            "uses either a compact proxy accumulator or a live Vulkan selector "
            "JSON as an active-dimension selector before comparing zero, random, "
            "and sign-shuffled controls. "
            "This validates the compiler accounting path, not a production GPU "
            "speedup claim by itself."
        ),
    }
    payload["decision"] = summarize_decision(payload, args.equivalence_tol)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-tokens", type=int, default=120_000)
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--feature-mode", choices=["token", "bigram", "rolling", "ssm_state"], default="bigram")
    parser.add_argument("--density", type=float, default=0.05)
    parser.add_argument("--scale", type=float, default=2.4)
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--calib-frac", type=float, default=0.20)
    parser.add_argument("--chunk-rows", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--equivalence-tol", type=float, default=1e-4)
    parser.add_argument("--residual-pilot", action="store_true")
    parser.add_argument("--compact-accumulator-pricing", action="store_true")
    parser.add_argument("--selector-json", type=Path)
    parser.add_argument("--active-fraction", type=float, default=0.50)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if args.max_tokens is not None and args.max_tokens <= 4:
        raise SystemExit("--max-tokens must be > 4")
    if args.dim < 2:
        raise SystemExit("--dim must be >= 2")
    if args.chunk_rows < 1:
        raise SystemExit("--chunk-rows must be >= 1")
    if args.train_frac <= 0.0 or args.calib_frac <= 0.0 or args.train_frac + args.calib_frac >= 1.0:
        raise SystemExit("--train-frac and --calib-frac must be positive and sum to less than 1")
    if args.selector_json is not None:
        args.compact_accumulator_pricing = True

    payload = run_gate(args)
    text = json.dumps(payload, indent=2)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
