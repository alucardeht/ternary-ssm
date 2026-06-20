#!/usr/bin/env python3
"""R420 round closeout gates.

This script closes the R420 follow-up questions that should not require a long
GPU run:

- whether the current live selector artifact contains enough information for a
  sign-bearing production realization;
- whether selector dimensions are stable enough across corpus shards;
- whether the selected active subspace retains enough proxy signal to justify
  further production work after the direct mask consumer failed;
- whether residual CE mass after TSSI is concentrated enough for unbiased
  residual active sampling;
- whether simple exact corpus atoms have enough support and low enough conflict
  to keep the atom path active in R420.

It is a research closeout gate, not a training benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from r419_tssi_gate import (  # noqa: E402
    DEFAULT_VOCAB,
    ce_for_logits_weights,
    compile_scores,
    make_feature_block,
    read_v12_tokens,
    realize_ternary,
    split_tokens,
)
from r420_online_tssi_compiler_gate import apply_dim_mask, load_live_selector  # noqa: E402


def finite_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def negative_log_likelihoods(weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    logits = x @ weights.T
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float64)
    denom = exp.sum(axis=1)
    target = exp[np.arange(y.shape[0]), y.astype(np.int64)]
    return (-np.log(np.maximum(target, 1e-300) / np.maximum(denom, 1e-300))).astype(np.float64)


def gate_sign_bearing(selector_payload: dict[str, Any]) -> dict[str, Any]:
    has_selected_dims = isinstance(selector_payload.get("selected_dims"), list)
    has_dim_signs = isinstance(selector_payload.get("signed_state_sum"), list)
    has_abs_dim_scores = isinstance(selector_payload.get("abs_state_sum"), list)
    has_vocab_state_commitments = any(
        key in selector_payload
        for key in (
            "ternary_commitments",
            "signed_vocab_state",
            "vocab_state_signs",
            "commitment_indices",
        )
    )
    decision = (
        "SIGN_BEARING_AVAILABLE"
        if has_vocab_state_commitments
        else "NO_SIGN_BEARING_PRODUCTION_ARTIFACT"
    )
    return {
        "decision": decision,
        "has_selected_dims": bool(has_selected_dims),
        "has_dim_signs": bool(has_dim_signs),
        "has_abs_dim_scores": bool(has_abs_dim_scores),
        "has_vocab_state_commitments": bool(has_vocab_state_commitments),
        "interpretation": (
            "The live selector can identify active dimensions and aggregate "
            "per-dimension signs, but it does not contain per-token/per-logit "
            "ternary commitments. A production sign-shuffled arm would be "
            "dishonest without a new sign-bearing artifact."
            if not has_vocab_state_commitments
            else "The selector contains an explicit sign-bearing commitment artifact."
        ),
    }


def gate_stability(
    train_tokens: np.ndarray,
    dim: int,
    feature_mode: str,
    vocab: int,
    selected_dims: np.ndarray,
    shards: int,
) -> dict[str, Any]:
    shard_signs: list[np.ndarray] = []
    shard_abs: list[np.ndarray] = []
    row_count = max(0, int(train_tokens.shape[0]) - 1)
    if row_count < shards:
        raise ValueError("not enough training rows for stability shards")
    edges = np.linspace(0, row_count, shards + 1, dtype=np.int64)
    for i in range(shards):
        start = int(edges[i])
        stop = int(edges[i + 1] + 1)
        shard_tokens = train_tokens[start:stop]
        x, y = make_feature_block(shard_tokens, dim, feature_mode)
        scores = compile_scores(x, y, vocab)
        signed = scores.sum(axis=0, dtype=np.float64)
        shard_signs.append(np.sign(signed).astype(np.int8))
        shard_abs.append(np.abs(scores).sum(axis=0, dtype=np.float64))

    sign_matrix = np.stack(shard_signs, axis=0)
    abs_matrix = np.stack(shard_abs, axis=0)
    nonzero = sign_matrix != 0
    agreement = []
    for j in range(dim):
        signs = sign_matrix[:, j]
        nz = signs[signs != 0]
        if nz.size == 0:
            agreement.append(0.0)
            continue
        positive = np.count_nonzero(nz > 0)
        negative = np.count_nonzero(nz < 0)
        agreement.append(max(positive, negative) / nz.size)
    agreement_arr = np.asarray(agreement, dtype=np.float64)
    selected = selected_dims.astype(np.int64)
    selected_abs = float(abs_matrix[:, selected].sum())
    total_abs = float(abs_matrix.sum())
    selected_abs_mass_fraction = selected_abs / total_abs if total_abs > 0.0 else 0.0
    selected_mean_agreement = float(np.mean(agreement_arr[selected]))
    all_mean_agreement = float(np.mean(agreement_arr))
    selected_nonzero_fraction = float(np.mean(nonzero[:, selected]))
    active_fraction = float(selected.shape[0] / dim)
    pass_gate = (
        selected_mean_agreement >= 0.75
        and selected_abs_mass_fraction >= active_fraction + 0.10
        and selected_nonzero_fraction >= 0.75
    )
    return {
        "decision": "PASS_STABILITY_SUPPORT" if pass_gate else "FAIL_STABILITY_SUPPORT",
        "shards": int(shards),
        "active_fraction": active_fraction,
        "selected_mean_sign_agreement": selected_mean_agreement,
        "all_mean_sign_agreement": all_mean_agreement,
        "selected_nonzero_fraction": selected_nonzero_fraction,
        "selected_abs_mass_fraction": selected_abs_mass_fraction,
        "thresholds": {
            "selected_mean_sign_agreement_min": 0.75,
            "selected_abs_mass_fraction_min": active_fraction + 0.10,
            "selected_nonzero_fraction_min": 0.75,
        },
    }


def gate_active_subspace(
    selector_payload: dict[str, Any],
    residual_payload: dict[str, Any] | None,
    production_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    abs_scores = np.asarray(selector_payload.get("abs_state_sum", []), dtype=np.float64)
    selected_dims = np.asarray(selector_payload.get("selected_dims", []), dtype=np.int64)
    score_mass_fraction = None
    if abs_scores.size and selected_dims.size:
        score_mass_fraction = float(abs_scores[selected_dims].sum() / max(abs_scores.sum(), 1e-300))

    residual = residual_payload.get("residual_pilot", {}) if residual_payload else {}
    proxy_advantage = residual.get("same_budget_ce_advantage")
    proxy_decision = residual.get("decision")

    production = production_payload or {}
    direct_eval = production.get("direct_selector_mask_eval_ce", {})
    tssi_eval = direct_eval.get("tssi")
    best_control = None
    if direct_eval:
        controls = [v for k, v in direct_eval.items() if k != "tssi" and v is not None]
        if controls:
            best_control = min(float(v) for v in controls)
    direct_margin = None
    if tssi_eval is not None and best_control is not None:
        direct_margin = best_control - float(tssi_eval)

    pass_gate = direct_margin is not None and direct_margin >= 0.0
    return {
        "decision": "PASS_ACTIVE_SUBSPACE_PRODUCTION" if pass_gate else "FAIL_ACTIVE_SUBSPACE_PRODUCTION",
        "selector_score_mass_fraction": finite_float(score_mass_fraction) if score_mass_fraction is not None else None,
        "proxy_same_budget_ce_advantage": finite_float(proxy_advantage) if proxy_advantage is not None else None,
        "proxy_decision": proxy_decision,
        "direct_eval_ce": direct_eval,
        "direct_best_control_minus_tssi": finite_float(direct_margin) if direct_margin is not None else None,
        "interpretation": (
            "The proxy/live-selector research gate remains positive, but the "
            "real mask-only production consumer did not beat controls. This "
            "closes the current active-subspace production path for R420."
            if not pass_gate
            else "The production active-subspace path beat controls."
        ),
    }


def gate_residual_sampling(
    train_tokens: np.ndarray,
    heldout_tokens: np.ndarray,
    dim: int,
    feature_mode: str,
    vocab: int,
    selected_dims: np.ndarray,
    density: float,
    scale: float,
) -> dict[str, Any]:
    train_x, train_y = make_feature_block(train_tokens, dim, feature_mode)
    heldout_x, heldout_y = make_feature_block(heldout_tokens, dim, feature_mode)
    scores = compile_scores(train_x, train_y, vocab)
    tssi = realize_ternary(scores, density).astype(np.float32) * float(scale)
    selected = apply_dim_mask(tssi, selected_dims)
    losses = negative_log_likelihoods(selected, heldout_x, heldout_y)
    total = float(losses.sum())
    if total <= 0.0:
        ess_fraction = 0.0
        top20_lift = 0.0
        top10_lift = 0.0
        max_weight_fraction = 1.0
    else:
        q = losses / total
        ess_fraction = float(1.0 / (losses.shape[0] * np.sum(q * q)))
        order = np.argsort(-losses, kind="stable")
        n20 = max(1, int(math.ceil(0.20 * losses.shape[0])))
        n10 = max(1, int(math.ceil(0.10 * losses.shape[0])))
        top20_lift = float(losses[order[:n20]].sum() / total / 0.20)
        top10_lift = float(losses[order[:n10]].sum() / total / 0.10)
        max_weight_fraction = float(np.max(q))
    pass_gate = ess_fraction >= 0.35 and top20_lift >= 1.50 and max_weight_fraction <= 0.05
    return {
        "decision": "PASS_RESIDUAL_ACTIVE_SAMPLING_GATE" if pass_gate else "FAIL_RESIDUAL_ACTIVE_SAMPLING_GATE",
        "heldout_examples": int(losses.shape[0]),
        "mean_residual_ce": float(np.mean(losses)),
        "std_residual_ce": float(np.std(losses)),
        "ess_fraction_loss_proportional": ess_fraction,
        "top20_residual_mass_lift": top20_lift,
        "top10_residual_mass_lift": top10_lift,
        "max_weight_fraction": max_weight_fraction,
        "thresholds": {
            "ess_fraction_min": 0.35,
            "top20_lift_min": 1.50,
            "max_weight_fraction_max": 0.05,
        },
    }


def gate_corpus_atoms(
    train_tokens: np.ndarray,
    heldout_tokens: np.ndarray,
    min_support: int,
    purity_threshold: float,
) -> dict[str, Any]:
    support: dict[int, dict[int, int]] = {}
    train = train_tokens.astype(np.int64)
    for a, b in zip(train[:-1], train[1:]):
        targets = support.setdefault(int(a), {})
        targets[int(b)] = targets.get(int(b), 0) + 1

    certified: dict[int, tuple[int, int, float]] = {}
    conflict_count = 0
    for context, targets in support.items():
        total = sum(targets.values())
        best_target, best_count = max(targets.items(), key=lambda item: item[1])
        purity = best_count / total if total else 0.0
        if total >= min_support and purity >= purity_threshold:
            certified[context] = (best_target, total, purity)
        elif total >= min_support:
            conflict_count += 1

    heldout = heldout_tokens.astype(np.int64)
    covered = 0
    correct = 0
    conflicts = 0
    for a, b in zip(heldout[:-1], heldout[1:]):
        atom = certified.get(int(a))
        if atom is None:
            continue
        covered += 1
        if int(b) == atom[0]:
            correct += 1
        else:
            conflicts += 1
    total_edges = max(1, int(heldout.shape[0] - 1))
    coverage = covered / total_edges
    heldout_purity = correct / covered if covered else 0.0
    pass_gate = coverage >= 0.10 and heldout_purity >= purity_threshold
    return {
        "decision": "PASS_CE_CERTIFIED_ATOM_COVERAGE" if pass_gate else "FAIL_CE_CERTIFIED_ATOM_COVERAGE",
        "atom_type": "exact_context_token_to_next_token",
        "min_support": int(min_support),
        "purity_threshold": float(purity_threshold),
        "train_certified_atoms": int(len(certified)),
        "train_conflicting_supported_contexts": int(conflict_count),
        "heldout_edges": int(total_edges),
        "heldout_covered_edges": int(covered),
        "heldout_coverage_fraction": float(coverage),
        "heldout_atom_purity": float(heldout_purity),
        "heldout_conflicts": int(conflicts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--selector-json", type=Path, required=True)
    parser.add_argument("--residual-json", type=Path)
    parser.add_argument("--production-json", type=Path)
    parser.add_argument("--max-tokens", type=int, default=30_000)
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--feature-mode", choices=["token", "bigram", "rolling", "ssm_state"], default="ssm_state")
    parser.add_argument("--density", type=float, default=0.05)
    parser.add_argument("--scale", type=float, default=2.4)
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--calib-frac", type=float, default=0.20)
    parser.add_argument("--stability-shards", type=int, default=6)
    parser.add_argument("--atom-min-support", type=int, default=4)
    parser.add_argument("--atom-purity", type=float, default=0.95)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    selector_payload = load_live_selector(args.selector_json, args.dim)
    selected_dims = selector_payload["_selected_dims_np"]
    residual_payload = load_json(args.residual_json)
    production_payload = load_json(args.production_json)

    tokens = read_v12_tokens(args.dataset, args.max_tokens)
    train_tokens, _calib_tokens, heldout_tokens = split_tokens(tokens, args.train_frac, args.calib_frac)

    payload = {
        "round": "R420",
        "gate": "round_closeout_gates",
        "dataset": str(args.dataset),
        "tokens_total": int(tokens.shape[0]),
        "tokens_train": int(train_tokens.shape[0]),
        "tokens_heldout": int(heldout_tokens.shape[0]),
        "feature_dim": int(args.dim),
        "feature_mode": args.feature_mode,
        "selector_json": str(args.selector_json),
        "residual_json": str(args.residual_json) if args.residual_json else None,
        "production_json": str(args.production_json) if args.production_json else None,
        "sign_bearing": gate_sign_bearing(selector_payload),
        "stability_thresholds": gate_stability(
            train_tokens,
            args.dim,
            args.feature_mode,
            args.vocab,
            selected_dims,
            args.stability_shards,
        ),
        "active_subspace": gate_active_subspace(selector_payload, residual_payload, production_payload),
        "residual_active_sampling": gate_residual_sampling(
            train_tokens,
            heldout_tokens,
            args.dim,
            args.feature_mode,
            args.vocab,
            selected_dims,
            args.density,
            args.scale,
        ),
        "ce_certified_corpus_atoms": gate_corpus_atoms(
            train_tokens,
            heldout_tokens,
            args.atom_min_support,
            args.atom_purity,
        ),
    }
    payload["decision"] = {
        "online_tssi_compiler": (
            "negative_for_current_production_speed_path"
            if payload["sign_bearing"]["decision"] == "NO_SIGN_BEARING_PRODUCTION_ARTIFACT"
            else "needs_sign_bearing_training_smoke"
        ),
        "active_subspace": payload["active_subspace"]["decision"],
        "stability_thresholds": payload["stability_thresholds"]["decision"],
        "residual_active_sampling": payload["residual_active_sampling"]["decision"],
        "ce_certified_corpus_atoms": payload["ce_certified_corpus_atoms"]["decision"],
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
