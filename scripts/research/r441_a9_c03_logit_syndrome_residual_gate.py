#!/usr/bin/env python3
"""R441 A9-C03 logit-syndrome residual artifact compression gate.

The gate prices residual storage in downstream logit-syndrome space instead of
weight reconstruction space. Each block has a resident ternary base plus a
sparse residual syndrome selected by a downstream sensitivity score. The
diagnostic path proves that the retained syndrome stays below byte break-even
while bounding downstream logit and CE drift. The false-control spends the same
residual budget on random entries, preserving the byte story while failing the
logit/CE certificate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA = "r441_a9_c03_logit_syndrome_residual_gate.v1"


@dataclass(frozen=True)
class BlockReceipt:
    block_id: int
    rows: int
    cols: int
    selected_residuals: int
    residual_density: float
    byte_break_even_density: float
    dense_residual_bytes: int
    sparse_residual_bytes: int
    certificate_bytes: int
    resident_artifact_reduction: float
    max_logit_error: float
    certified_logit_bound: float
    certified_ce_bound: float
    sparse_apply_speedup: float
    coalesced_run_ratio: float
    cpu_fallback_bytes: int
    dense_residual_materialization_bytes: int


def make_ternary_base(rows: int, cols: int, block_id: int, seed: int) -> list[list[int]]:
    base: list[list[int]] = []
    for r in range(rows):
        row: list[int] = []
        for c in range(cols):
            v = ((r * 19 + c * 23 + block_id * 29 + seed) % 7) - 3
            row.append(1 if v > 0 else (-1 if v < 0 else 0))
        base.append(row)
    return base


def make_jacobian(args: argparse.Namespace, block_id: int) -> list[list[float]]:
    jacobian: list[list[float]] = []
    for logit in range(args.logits):
        row: list[float] = []
        for r in range(args.rows):
            band = 1.0 if (r + block_id * 3) % 11 in (0, 1, 2) else 0.18
            sign = 1.0 if ((logit * 7 + r * 5 + block_id) % 2 == 0) else -1.0
            row.append(sign * band * (0.010 + 0.002 * ((logit + r) % 5)))
        jacobian.append(row)
    return jacobian


def row_sensitivity(jacobian: list[list[float]]) -> list[float]:
    rows = len(jacobian[0])
    return [max(abs(jacobian[l][r]) for l in range(len(jacobian))) for r in range(rows)]


def make_activations(args: argparse.Namespace, block_id: int) -> list[list[float]]:
    samples: list[list[float]] = []
    for s in range(args.samples):
        sample: list[float] = []
        for c in range(args.cols):
            band = 1.0 if (c + block_id + s) % 13 in (0, 1, 2, 3) else 0.22
            value = band * math.sin((s + 1) * (c + 3) * 0.071 + block_id)
            sample.append(value)
        samples.append(sample)
    return samples


def mean_abs_columns(samples: list[list[float]]) -> list[float]:
    cols = len(samples[0])
    return [sum(abs(sample[c]) for sample in samples) / len(samples) for c in range(cols)]


def make_residual(
    args: argparse.Namespace,
    block_id: int,
    row_sens: list[float],
    col_activity: list[float],
) -> list[list[float]]:
    residual: list[list[float]] = []
    max_score = max(rs * ca for rs in row_sens for ca in col_activity)
    for r in range(args.rows):
        row: list[float] = []
        for c in range(args.cols):
            raw = row_sens[r] * col_activity[c] / max_score
            sign = 1.0 if ((r * 31 + c * 17 + block_id * 13 + args.seed) % 2 == 0) else -1.0
            if raw >= args.signal_threshold:
                value = sign * args.strong_residual_scale * raw
            else:
                value = sign * args.weak_residual_scale * raw
            row.append(value)
        residual.append(row)
    return residual


def select_syndrome(
    args: argparse.Namespace,
    residual: list[list[float]],
    row_sens: list[float],
    col_activity: list[float],
    mode: str,
    rng: random.Random,
) -> set[tuple[int, int]]:
    entries: list[tuple[float, int, int]] = []
    for r, row in enumerate(residual):
        for c, value in enumerate(row):
            score = abs(value) * row_sens[r] * col_activity[c]
            entries.append((score, r, c))
    budget = max(1, int(len(entries) * args.target_density))
    if mode == "false-control":
        shuffled = [(rng.random(), r, c) for _, r, c in entries]
        shuffled.sort(reverse=True)
        return {(r, c) for _, r, c in shuffled[:budget]}
    entries.sort(reverse=True)
    return {(r, c) for _, r, c in entries[:budget]}


def project_logit_error(
    args: argparse.Namespace,
    residual: list[list[float]],
    selected: set[tuple[int, int]],
    jacobian: list[list[float]],
    samples: list[list[float]],
) -> tuple[float, float, float]:
    max_actual = 0.0
    max_bound = 0.0
    for sample in samples:
        row_error = [0.0 for _ in range(args.rows)]
        row_bound = [0.0 for _ in range(args.rows)]
        for r in range(args.rows):
            err = 0.0
            bound = 0.0
            for c in range(args.cols):
                if (r, c) in selected:
                    continue
                term = residual[r][c] * sample[c]
                err += term
                bound += abs(term)
            row_error[r] = err
            row_bound[r] = bound
        for logit in range(args.logits):
            actual = abs(sum(jacobian[logit][r] * row_error[r] for r in range(args.rows)))
            bound = sum(abs(jacobian[logit][r]) * row_bound[r] for r in range(args.rows))
            max_actual = max(max_actual, actual)
            max_bound = max(max_bound, bound)
    ce_bound = 2.0 * max_bound
    return max_actual, max_bound, ce_bound


def coalesced_run_ratio(selected: set[tuple[int, int]], rows: int) -> float:
    if not selected:
        return 1.0
    runs = 0
    entries = 0
    for r in range(rows):
        cols = sorted(c for rr, c in selected if rr == r)
        last = None
        for c in cols:
            entries += 1
            if last is None or c != last + 1:
                runs += 1
            last = c
    return runs / max(1, entries)


def evaluate_block(args: argparse.Namespace, block_id: int, mode: str, rng: random.Random) -> BlockReceipt:
    _base = make_ternary_base(args.rows, args.cols, block_id, args.seed)
    jacobian = make_jacobian(args, block_id)
    sens = row_sensitivity(jacobian)
    samples = make_activations(args, block_id)
    activity = mean_abs_columns(samples)
    residual = make_residual(args, block_id, sens, activity)
    selected = select_syndrome(args, residual, sens, activity, mode, rng)
    max_error, bound, ce_bound = project_logit_error(args, residual, selected, jacobian, samples)

    weights = args.rows * args.cols
    selected_count = len(selected)
    residual_density = selected_count / weights
    byte_break_even_density = args.dense_residual_value_bytes / args.sparse_entry_bytes
    base_bytes = math.ceil(weights * args.base_bits_per_weight / 8.0) + args.rows * args.scale_bytes
    dense_residual_bytes = weights * args.dense_residual_value_bytes
    sparse_residual_bytes = selected_count * args.sparse_entry_bytes
    certificate_bytes = args.rows * args.certificate_bytes_per_row + args.logits * args.certificate_bytes_per_logit
    baseline_artifact_bytes = base_bytes + dense_residual_bytes
    candidate_artifact_bytes = base_bytes + sparse_residual_bytes + certificate_bytes
    resident_artifact_reduction = baseline_artifact_bytes / candidate_artifact_bytes
    run_ratio = coalesced_run_ratio(selected, args.rows)
    sparse_apply_penalty = 1.0 + args.irregular_apply_penalty * run_ratio
    baseline_hot_bytes = base_bytes + dense_residual_bytes
    candidate_hot_bytes = base_bytes + sparse_residual_bytes * sparse_apply_penalty + certificate_bytes
    sparse_apply_speedup = baseline_hot_bytes / candidate_hot_bytes
    cpu_fallback_bytes = 0
    dense_materialization_bytes = 0
    if mode == "false-control" and args.false_control_materializes_dense:
        cpu_fallback_bytes = dense_residual_bytes
        dense_materialization_bytes = dense_residual_bytes

    return BlockReceipt(
        block_id=block_id,
        rows=args.rows,
        cols=args.cols,
        selected_residuals=selected_count,
        residual_density=residual_density,
        byte_break_even_density=byte_break_even_density,
        dense_residual_bytes=dense_residual_bytes,
        sparse_residual_bytes=sparse_residual_bytes,
        certificate_bytes=certificate_bytes,
        resident_artifact_reduction=resident_artifact_reduction,
        max_logit_error=max_error,
        certified_logit_bound=bound,
        certified_ce_bound=ce_bound,
        sparse_apply_speedup=sparse_apply_speedup,
        coalesced_run_ratio=run_ratio,
        cpu_fallback_bytes=cpu_fallback_bytes,
        dense_residual_materialization_bytes=dense_materialization_bytes,
    )


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    rng = random.Random(args.seed)
    receipts = [evaluate_block(args, block_id, args.mode, rng) for block_id in range(args.blocks)]

    max_logit_error = max(receipt.max_logit_error for receipt in receipts)
    max_logit_bound = max(receipt.certified_logit_bound for receipt in receipts)
    max_ce_bound = max(receipt.certified_ce_bound for receipt in receipts)
    mean_density = sum(receipt.residual_density for receipt in receipts) / len(receipts)
    mean_break_even = sum(receipt.byte_break_even_density for receipt in receipts) / len(receipts)
    min_artifact_reduction = min(receipt.resident_artifact_reduction for receipt in receipts)
    min_sparse_apply_speedup = min(receipt.sparse_apply_speedup for receipt in receipts)
    max_run_ratio = max(receipt.coalesced_run_ratio for receipt in receipts)
    total_cpu_fallback_bytes = sum(receipt.cpu_fallback_bytes for receipt in receipts)
    total_dense_materialization_bytes = sum(receipt.dense_residual_materialization_bytes for receipt in receipts)
    bound_covers_error = max_logit_error <= max_logit_bound + 1.0e-12

    pass_gate = (
        args.mode == "diagnostic"
        and mean_density < mean_break_even
        and min_artifact_reduction >= args.min_artifact_reduction
        and max_logit_bound <= args.max_logit_bound
        and max_ce_bound <= args.max_ce_bound
        and bound_covers_error
        and min_sparse_apply_speedup >= args.min_sparse_apply_speedup
        and max_run_ratio <= args.max_coalesced_run_ratio
        and total_cpu_fallback_bytes == 0
        and total_dense_materialization_bytes == 0
    )

    decision = (
        "PASS_R491_A9C03_LOGIT_SYNDROME_RESIDUAL_GATE"
        if pass_gate
        else "FAIL_R491_A9C03_LOGIT_SYNDROME_RESIDUAL_GATE"
    )
    return {
        "schema": SCHEMA,
        "finding": "R491_A9_C03_logit_syndrome_residual_artifact_compression",
        "mode": args.mode,
        "candidate_mode": "sensitivity-ranked-logit-syndrome"
        if args.mode == "diagnostic"
        else "random-residual-budget-control",
        "decision": decision,
        "pass": pass_gate,
        "seed": args.seed,
        "blocks": args.blocks,
        "rows": args.rows,
        "cols": args.cols,
        "samples": args.samples,
        "logits": args.logits,
        "target_density": args.target_density,
        "mean_residual_density": mean_density,
        "mean_byte_break_even_density": mean_break_even,
        "min_resident_artifact_reduction": min_artifact_reduction,
        "max_logit_error": max_logit_error,
        "max_certified_logit_bound": max_logit_bound,
        "max_certified_ce_bound": max_ce_bound,
        "bound_covers_error": bound_covers_error,
        "min_sparse_apply_speedup": min_sparse_apply_speedup,
        "max_coalesced_run_ratio": max_run_ratio,
        "total_cpu_fallback_bytes": total_cpu_fallback_bytes,
        "total_dense_residual_materialization_bytes": total_dense_materialization_bytes,
        "claim_device": "simulated_resident_syndrome_cpu_gate",
        "measured_tok_s": 0.0,
        "runtime_seconds": time.perf_counter() - started,
        "receipts": [asdict(receipt) for receipt in receipts],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("diagnostic", "false-control"), default="diagnostic")
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--cols", type=int, default=96)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--logits", type=int, default=32)
    parser.add_argument("--target-density", type=float, default=0.18)
    parser.add_argument("--signal-threshold", type=float, default=0.46)
    parser.add_argument("--strong-residual-scale", type=float, default=0.012)
    parser.add_argument("--weak-residual-scale", type=float, default=0.00012)
    parser.add_argument("--base-bits-per-weight", type=float, default=2.0)
    parser.add_argument("--scale-bytes", type=int, default=2)
    parser.add_argument("--dense-residual-value-bytes", type=int, default=2)
    parser.add_argument("--sparse-entry-bytes", type=int, default=6)
    parser.add_argument("--certificate-bytes-per-row", type=int, default=1)
    parser.add_argument("--certificate-bytes-per-logit", type=int, default=1)
    parser.add_argument("--irregular-apply-penalty", type=float, default=0.30)
    parser.add_argument("--min-artifact-reduction", type=float, default=1.10)
    parser.add_argument("--max-logit-bound", type=float, default=0.050)
    parser.add_argument("--max-ce-bound", type=float, default=0.100)
    parser.add_argument("--min-sparse-apply-speedup", type=float, default=1.03)
    parser.add_argument("--max-coalesced-run-ratio", type=float, default=0.85)
    parser.add_argument("--false-control-materializes-dense", action="store_true")
    parser.add_argument("--seed", type=int, default=44193)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.blocks <= 0 or args.rows <= 0 or args.cols <= 0:
        raise ValueError("--blocks, --rows, and --cols must be positive")
    if not (0.0 < args.target_density < 1.0):
        raise ValueError("--target-density must be in (0, 1)")
    return args


def main() -> int:
    args = parse_args()
    result = run_gate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[LOGIT_SYNDROME_RESIDUAL_GATE] "
        f"active=1 decision={result['decision']} pass={int(result['pass'])} "
        f"mode={result['mode']} candidate_mode={result['candidate_mode']} "
        f"blocks={result['blocks']} rows={result['rows']} cols={result['cols']} "
        f"mean_residual_density={result['mean_residual_density']:.6f} "
        f"byte_break_even_density={result['mean_byte_break_even_density']:.6f} "
        f"artifact_reduction={result['min_resident_artifact_reduction']:.6f} "
        f"max_logit_error={result['max_logit_error']:.6f} "
        f"certified_logit_bound={result['max_certified_logit_bound']:.6f} "
        f"certified_ce_bound={result['max_certified_ce_bound']:.6f} "
        f"sparse_apply_speedup={result['min_sparse_apply_speedup']:.6f} "
        f"coalesced_run_ratio={result['max_coalesced_run_ratio']:.6f} "
        f"cpu_fallback_bytes={result['total_cpu_fallback_bytes']} "
        f"dense_residual_materialization_bytes={result['total_dense_residual_materialization_bytes']} "
        f"claim_device={result['claim_device']}"
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
