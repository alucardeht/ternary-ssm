#!/usr/bin/env python3
"""Audit Paper 7 direct factored gradient accumulation against dense G.

This script is a cheap algebraic falsifier for the Paper 7 claim boundary. It
does not benchmark the production trainer. Given per-position LM-head
cotangents ``delta`` and normalized hidden states ``h``, it materializes the
dense gradient

    G = delta.T @ h

for a controlled toy shape, then compares dense row/column marginals with the
streaming quantities used by direct factored gradient accumulation (DF-GA).

The first-order marginals should match to floating-point precision. The
second-moment marginals expose the cross-term residual that the paper must price
or validate under a martingale-like assumption.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUTPUT_ROOT = Path(".work/jobs/paper7-dfga")


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_mode_list(raw: str) -> list[str]:
    allowed = {"sampled_softmax", "gaussian_martingale", "correlated"}
    modes = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown mode(s): {', '.join(unknown)}")
    if not modes:
        raise argparse.ArgumentTypeError("expected at least one mode")
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7007)
    parser.add_argument("--modes", type=parse_mode_list,
                        default=parse_mode_list("sampled_softmax,correlated"))
    parser.add_argument("--tokens", type=parse_int_list, default=parse_int_list("64,128,256,512"))
    parser.add_argument("--vocab", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.10)
    parser.add_argument("--residual-budget", type=float, default=0.35)
    return parser.parse_args()


def normalize_hidden(h: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(h * h, axis=1, keepdims=True) + 1.0e-12)
    return h / rms


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def make_case(
    rng: np.random.Generator,
    mode: str,
    n_tokens: int,
    vocab: int,
    d_model: int,
    noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``delta`` [n_tokens, vocab] and ``h`` [n_tokens, d_model]."""
    if mode == "sampled_softmax":
        h = normalize_hidden(rng.normal(0.0, 1.0, size=(n_tokens, d_model)))
        logits = rng.normal(0.0, 1.0, size=(n_tokens, vocab))
        probs = softmax(logits)
        targets = rng.integers(0, vocab, size=n_tokens)
        delta = probs
        delta[np.arange(n_tokens), targets] -= 1.0
        return delta.astype(np.float64), h.astype(np.float64)

    if mode == "gaussian_martingale":
        h = normalize_hidden(rng.normal(0.0, 1.0, size=(n_tokens, d_model)))
        delta = rng.normal(0.0, 1.0, size=(n_tokens, vocab))
        delta -= np.mean(delta, axis=1, keepdims=True)
        return delta.astype(np.float64), h.astype(np.float64)

    if mode == "correlated":
        latent = rng.normal(0.0, 1.0, size=(n_tokens, 1))
        h_direction = rng.normal(0.0, 1.0, size=(1, d_model))
        d_direction = rng.normal(0.0, 1.0, size=(1, vocab))
        d_direction -= np.mean(d_direction)
        h = latent @ h_direction + noise * rng.normal(0.0, 1.0, size=(n_tokens, d_model))
        delta = latent @ d_direction + noise * rng.normal(0.0, 1.0, size=(n_tokens, vocab))
        delta -= np.mean(delta, axis=1, keepdims=True)
        return delta.astype(np.float64), normalize_hidden(h).astype(np.float64)

    raise ValueError(f"unhandled mode: {mode}")


def rel_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    denom = float(np.linalg.norm(expected))
    if denom <= 1.0e-30:
        return 0.0 if float(np.linalg.norm(actual)) <= 1.0e-30 else math.inf
    return float(np.linalg.norm(actual - expected) / denom)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1.0e-30:
        return math.nan
    return float(np.dot(a.ravel(), b.ravel()) / denom)


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def audit_case(delta: np.ndarray, h: np.ndarray) -> dict[str, float]:
    dense_g = delta.T @ h

    row_first_dense = np.sum(dense_g, axis=1)
    row_first_direct = np.sum(delta * np.sum(h, axis=1, keepdims=True), axis=0)
    col_first_dense = np.sum(dense_g, axis=0)
    col_first_direct = np.sum(np.sum(delta, axis=1, keepdims=True) * h, axis=0)

    row_sq_exact = np.sum(dense_g * dense_g, axis=1)
    col_sq_exact = np.sum(dense_g * dense_g, axis=0)
    row_sq_diag = (delta * delta).T @ np.sum(h * h, axis=1)
    col_sq_diag = (h * h).T @ np.sum(delta * delta, axis=1)

    return {
        "first_order_row_rel_l2": rel_l2(row_first_direct, row_first_dense),
        "first_order_col_rel_l2": rel_l2(col_first_direct, col_first_dense),
        "first_order_row_max_abs": float(np.max(np.abs(row_first_direct - row_first_dense))),
        "first_order_col_max_abs": float(np.max(np.abs(col_first_direct - col_first_dense))),
        "row_sq_diag_rel_l2": rel_l2(row_sq_diag, row_sq_exact),
        "col_sq_diag_rel_l2": rel_l2(col_sq_diag, col_sq_exact),
        "row_sq_diag_cosine": cosine(row_sq_diag, row_sq_exact),
        "col_sq_diag_cosine": cosine(col_sq_diag, col_sq_exact),
        "row_sq_diag_to_exact_sum": float(np.sum(row_sq_diag) / max(np.sum(row_sq_exact), 1.0e-30)),
        "col_sq_diag_to_exact_sum": float(np.sum(col_sq_diag) / max(np.sum(col_sq_exact), 1.0e-30)),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted(rows[0])
    return {key: summarize([row[key] for row in rows]) for key in keys}


def write_summary_markdown(
    path: Path,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    dense_bytes: int,
    factor_bytes: int,
) -> None:
    lines = [
        "# Paper 7 DF-GA Dense Audit",
        "",
        "This is a synthetic algebraic audit, not a production training-speed benchmark.",
        "",
        f"- seed: `{args.seed}`",
        f"- vocab: `{args.vocab}`",
        f"- d_model: `{args.d_model}`",
        f"- trials per cell: `{args.trials}`",
        f"- dense gradient bytes: `{dense_bytes}`",
        f"- direct factor bytes: `{factor_bytes}`",
        f"- dense/factor byte ratio: `{dense_bytes / max(factor_bytes, 1):.6f}`",
        "",
        "| mode | tokens | first row err max | first col err max | row sq residual mean | col sq residual mean | row cosine mean | col cosine mean | pass budget |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        a = record["aggregate"]
        pass_budget = (
            a["row_sq_diag_rel_l2"]["mean"] <= args.residual_budget
            and a["col_sq_diag_rel_l2"]["mean"] <= args.residual_budget
        )
        lines.append(
            "| {mode} | {tokens} | {r1:.3e} | {c1:.3e} | {r2:.6f} | {c2:.6f} | {rcos:.6f} | {ccos:.6f} | {pass_budget} |".format(
                mode=record["mode"],
                tokens=record["tokens"],
                r1=a["first_order_row_max_abs"]["max"],
                c1=a["first_order_col_max_abs"]["max"],
                r2=a["row_sq_diag_rel_l2"]["mean"],
                c2=a["col_sq_diag_rel_l2"]["mean"],
                rcos=a["row_sq_diag_cosine"]["mean"],
                ccos=a["col_sq_diag_cosine"]["mean"],
                pass_budget="yes" if pass_budget else "no",
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- First-order row/column marginals should be machine-precision exact.",
            "- Second-moment row/column residuals measure the dropped cross terms.",
            "- `sampled_softmax` is the martingale-like positive-control regime.",
            "- `correlated` is a negative control where cross terms are intentionally large.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.trials <= 0:
        raise SystemExit("--trials must be >= 1")
    if args.vocab <= 1 or args.d_model <= 1:
        raise SystemExit("--vocab and --d-model must be > 1")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / f"dense-audit-{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    records: list[dict[str, Any]] = []
    for mode in args.modes:
        for n_tokens in args.tokens:
            trial_rows: list[dict[str, float]] = []
            for _ in range(args.trials):
                delta, h = make_case(rng, mode, n_tokens, args.vocab, args.d_model, args.noise)
                trial_rows.append(audit_case(delta, h))
            records.append(
                {
                    "mode": mode,
                    "tokens": n_tokens,
                    "trials": trial_rows,
                    "aggregate": aggregate(trial_rows),
                }
            )

    dense_bytes = args.vocab * args.d_model * 4
    factor_bytes = (args.vocab + args.d_model) * 4
    payload = {
        "activation": "[PAPER7_DFGA_DENSE_AUDIT]",
        "timestamp": timestamp,
        "config": {
            "seed": args.seed,
            "modes": args.modes,
            "tokens": args.tokens,
            "vocab": args.vocab,
            "d_model": args.d_model,
            "trials": args.trials,
            "noise": args.noise,
            "residual_budget": args.residual_budget,
        },
        "memory_model": {
            "dense_gradient_bytes": dense_bytes,
            "direct_factor_bytes": factor_bytes,
            "dense_to_factor_byte_ratio": dense_bytes / max(factor_bytes, 1),
        },
        "records": records,
    }

    result_path = output_dir / "result.json"
    summary_path = output_dir / "summary.md"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_markdown(summary_path, records, args, dense_bytes, factor_bytes)

    print(
        "[PAPER7_DFGA_DENSE_AUDIT] "
        f"active=1 output_dir={output_dir} modes={','.join(args.modes)} "
        f"tokens={','.join(str(v) for v in args.tokens)} vocab={args.vocab} "
        f"d_model={args.d_model} trials={args.trials} "
        f"dense_to_factor_byte_ratio={dense_bytes / max(factor_bytes, 1):.6f}"
    )
    print(f"[PAPER7_DFGA_DENSE_AUDIT] result_json={result_path}")
    print(f"[PAPER7_DFGA_DENSE_AUDIT] summary_md={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
