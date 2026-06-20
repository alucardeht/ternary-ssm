#!/usr/bin/env python3
"""R441 A9-C01 packed resident tile execution gate.

This is a narrow direct-execution gate for packed ternary tiles. It compares a
dense realized reference against a direct packed tile evaluator that consumes
resident sign/zero geometry plus row scales without reconstructing a full dense
matrix. The false-control reconstructs dense tiles in the hot path, so it can
match numerics while failing the A9 materialization and transient-denominator
requirements.
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


SCHEMA = "r441_a9_c01_packed_resident_tile_gate.v1"


@dataclass(frozen=True)
class TileReceipt:
    block_id: int
    rows: int
    cols: int
    tile_cols: int
    packed_payload_bytes: int
    descriptor_bytes: int
    scale_bytes: int
    scratch_bytes: int
    dense_materialization_bytes: int
    hot_payload_load_events: int


def make_matrix(args: argparse.Namespace, rng: random.Random) -> tuple[list[list[int]], list[float]]:
    matrix: list[list[int]] = []
    scales: list[float] = []
    for r in range(args.rows):
        scale = 0.5 + 0.125 * (r % 7)
        scales.append(scale)
        row: list[int] = []
        for c in range(args.cols):
            # Deterministic sparse-ish ternary pattern with enough nonzero mass
            # to make dense payload reads meaningful.
            v = ((r * 17 + c * 11 + rng.randint(0, 5)) % 5) - 2
            row.append(1 if v > 0 else (-1 if v < 0 else 0))
        matrix.append(row)
    return matrix, scales


def pack_matrix(matrix: list[list[int]], tile_cols: int) -> list[list[tuple[int, int]]]:
    packed: list[list[tuple[int, int]]] = []
    for row in matrix:
        packed_row: list[tuple[int, int]] = []
        for start in range(0, len(row), tile_cols):
            pos_bits = 0
            neg_bits = 0
            tile = row[start : start + tile_cols]
            for bit, value in enumerate(tile):
                if value > 0:
                    pos_bits |= 1 << bit
                elif value < 0:
                    neg_bits |= 1 << bit
            packed_row.append((pos_bits, neg_bits))
        packed.append(packed_row)
    return packed


def dense_reference(matrix: list[list[int]], scales: list[float], x: list[float]) -> list[float]:
    out: list[float] = []
    for row, scale in zip(matrix, scales):
        acc = 0.0
        for weight, value in zip(row, x):
            acc += scale * weight * value
        out.append(acc)
    return out


def packed_eval(
    packed: list[list[tuple[int, int]]],
    scales: list[float],
    x: list[float],
    tile_cols: int,
) -> list[float]:
    out: list[float] = []
    for packed_row, scale in zip(packed, scales):
        acc = 0.0
        for tile_idx, (pos_bits, neg_bits) in enumerate(packed_row):
            base = tile_idx * tile_cols
            bits = pos_bits
            while bits:
                lsb = bits & -bits
                bit = lsb.bit_length() - 1
                acc += x[base + bit]
                bits ^= lsb
            bits = neg_bits
            while bits:
                lsb = bits & -bits
                bit = lsb.bit_length() - 1
                acc -= x[base + bit]
                bits ^= lsb
        out.append(acc * scale)
    return out


def dense_reconstruct_control(
    packed: list[list[tuple[int, int]]],
    scales: list[float],
    x: list[float],
    rows: int,
    cols: int,
    tile_cols: int,
) -> list[float]:
    dense: list[list[int]] = []
    for packed_row in packed:
        row = [0] * cols
        for tile_idx, (pos_bits, neg_bits) in enumerate(packed_row):
            base = tile_idx * tile_cols
            for bit in range(tile_cols):
                if base + bit >= cols:
                    continue
                if (pos_bits >> bit) & 1:
                    row[base + bit] = 1
                elif (neg_bits >> bit) & 1:
                    row[base + bit] = -1
        dense.append(row)
    assert len(dense) == rows
    return dense_reference(dense, scales, x)


def max_abs_error(a: list[float], b: list[float]) -> float:
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def rms(a: list[float], b: list[float]) -> float:
    if not a:
        return 0.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def score_arm(args: argparse.Namespace, mode_name: str) -> dict[str, Any]:
    rng = random.Random(args.seed)
    blocks = []
    total_dense_bytes = 0
    total_packed_payload = 0
    total_descriptors = 0
    total_scales = 0
    total_scratch = 0
    total_dense_materialization = 0
    total_hot_loads = 0
    max_error = 0.0
    rel_rmse_acc = 0.0
    dense_hot_bytes = 0
    packed_hot_bytes = 0

    for block_id in range(args.blocks):
        matrix, scales = make_matrix(args, rng)
        packed = pack_matrix(matrix, args.tile_cols)
        x = [math.sin((block_id + 1) * (i + 3) * 0.017) for i in range(args.cols)]
        ref = dense_reference(matrix, scales, x)
        if mode_name == "packed-resident-tile":
            got = packed_eval(packed, scales, x, args.tile_cols)
            dense_materialization_bytes = 0
            scratch_bytes = args.rows * args.scratch_floats_per_row * 4
            hot_payload_load_events = 0
        else:
            got = dense_reconstruct_control(packed, scales, x, args.rows, args.cols, args.tile_cols)
            dense_materialization_bytes = args.rows * args.cols * args.dense_bytes_per_weight
            scratch_bytes = dense_materialization_bytes + args.rows * args.scratch_floats_per_row * 4
            hot_payload_load_events = args.rows

        tiles_per_row = math.ceil(args.cols / args.tile_cols)
        packed_payload_bytes = args.rows * tiles_per_row * 8
        descriptor_bytes = args.rows * tiles_per_row * args.descriptor_bytes_per_tile
        scale_bytes = args.rows * args.scale_bytes_per_row
        dense_bytes = args.rows * args.cols * args.dense_bytes_per_weight
        total_dense_bytes += dense_bytes
        total_packed_payload += packed_payload_bytes
        total_descriptors += descriptor_bytes
        total_scales += scale_bytes
        total_scratch += scratch_bytes
        total_dense_materialization += dense_materialization_bytes
        total_hot_loads += hot_payload_load_events
        err = max_abs_error(ref, got)
        max_error = max(max_error, err)
        denom = max(1e-12, rms(ref, [0.0] * len(ref)))
        rel_rmse_acc += rms(ref, got) / denom
        dense_hot_bytes += dense_bytes + args.rows * args.cols * args.input_read_bytes
        packed_hot_bytes += packed_payload_bytes + descriptor_bytes + scale_bytes + scratch_bytes
        if mode_name != "packed-resident-tile":
            packed_hot_bytes += dense_materialization_bytes

        blocks.append(
            TileReceipt(
                block_id=block_id,
                rows=args.rows,
                cols=args.cols,
                tile_cols=args.tile_cols,
                packed_payload_bytes=packed_payload_bytes,
                descriptor_bytes=descriptor_bytes,
                scale_bytes=scale_bytes,
                scratch_bytes=scratch_bytes,
                dense_materialization_bytes=dense_materialization_bytes,
                hot_payload_load_events=hot_payload_load_events,
            )
        )

    resident_packed_bytes = total_packed_payload + total_descriptors + total_scales
    resident_vram_ratio = resident_packed_bytes / max(1, total_dense_bytes)
    transient_dense_ratio = total_dense_materialization / max(1, resident_packed_bytes)
    scratch_ratio = total_scratch / max(1, resident_packed_bytes)
    projected_hot_speedup = dense_hot_bytes / max(1, packed_hot_bytes)
    rel_rmse = rel_rmse_acc / max(1, args.blocks)

    pass_gate = (
        mode_name == "packed-resident-tile"
        and max_error <= args.max_abs_error
        and rel_rmse <= args.max_rel_rmse
        and total_dense_materialization == 0
        and transient_dense_ratio <= args.max_transient_dense_ratio
        and scratch_ratio <= args.max_scratch_ratio
        and total_hot_loads == 0
        and resident_vram_ratio <= args.max_resident_vram_ratio
        and projected_hot_speedup >= args.min_projected_hot_speedup
    )

    return {
        "mode": mode_name,
        "blocks": args.blocks,
        "rows": args.rows,
        "cols": args.cols,
        "tile_cols": args.tile_cols,
        "max_abs_error": max_error,
        "relative_rmse": rel_rmse,
        "dense_reference_bytes": total_dense_bytes,
        "resident_packed_bytes": resident_packed_bytes,
        "packed_payload_bytes": total_packed_payload,
        "descriptor_bytes": total_descriptors,
        "scale_bytes": total_scales,
        "scratch_bytes": total_scratch,
        "dense_materialization_bytes": total_dense_materialization,
        "transient_dense_ratio": transient_dense_ratio,
        "scratch_ratio": scratch_ratio,
        "resident_vram_ratio": resident_vram_ratio,
        "hot_payload_load_events": total_hot_loads,
        "dense_hot_bytes": dense_hot_bytes,
        "packed_hot_bytes": packed_hot_bytes,
        "projected_hot_speedup": projected_hot_speedup,
        "receipt_sample": asdict(blocks[0]) if blocks else None,
        "claim_device": "simulated_resident_payload_cpu_direct_execution",
        "pass": pass_gate,
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    diagnostic = score_arm(args, "packed-resident-tile")
    dense_reconstruct = score_arm(args, "dense-reconstruct-control")
    candidate = dense_reconstruct if args.mode == "false-control" else diagnostic
    decision = (
        "PASS_R489_A9C01_PACKED_RESIDENT_TILE_GATE"
        if candidate["pass"]
        else "FAIL_R489_A9C01_PACKED_RESIDENT_TILE_GATE"
    )
    return {
        "schema": SCHEMA,
        "decision": decision,
        "pass": bool(candidate["pass"]),
        "mode": args.mode,
        "seed": args.seed,
        "candidate": candidate,
        "diagnostic_control": diagnostic,
        "dense_reconstruct_control": dense_reconstruct,
        "thresholds": {
            "max_abs_error": args.max_abs_error,
            "max_rel_rmse": args.max_rel_rmse,
            "max_transient_dense_ratio": args.max_transient_dense_ratio,
            "max_scratch_ratio": args.max_scratch_ratio,
            "max_resident_vram_ratio": args.max_resident_vram_ratio,
            "min_projected_hot_speedup": args.min_projected_hot_speedup,
        },
        "elapsed_s": time.perf_counter() - start,
        "claim_scope": "direct_packed_tile_execution_gate_not_gpu_production_or_measured_tok_s",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["diagnostic", "false-control"], default="diagnostic")
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--cols", type=int, default=128)
    parser.add_argument("--tile-cols", type=int, default=32)
    parser.add_argument("--seed", type=int, default=44191)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dense-bytes-per-weight", type=int, default=2)
    parser.add_argument("--input-read-bytes", type=int, default=2)
    parser.add_argument("--descriptor-bytes-per-tile", type=int, default=2)
    parser.add_argument("--scale-bytes-per-row", type=int, default=2)
    parser.add_argument("--scratch-floats-per-row", type=int, default=1)
    parser.add_argument("--max-abs-error", type=float, default=1e-6)
    parser.add_argument("--max-rel-rmse", type=float, default=1e-8)
    parser.add_argument("--max-transient-dense-ratio", type=float, default=0.10)
    parser.add_argument("--max-scratch-ratio", type=float, default=0.30)
    parser.add_argument("--max-resident-vram-ratio", type=float, default=0.35)
    parser.add_argument("--min-projected-hot-speedup", type=float, default=1.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    candidate = summary["candidate"]
    print(
        "[PACKED_RESIDENT_TILE_GATE] "
        f"active=1 decision={summary['decision']} pass={int(summary['pass'])} "
        f"mode={summary['mode']} candidate_mode={candidate['mode']} "
        f"blocks={candidate['blocks']} rows={candidate['rows']} cols={candidate['cols']} "
        f"max_abs_error={candidate['max_abs_error']:.9f} "
        f"relative_rmse={candidate['relative_rmse']:.9f} "
        f"dense_materialization_bytes={candidate['dense_materialization_bytes']} "
        f"transient_dense_ratio={candidate['transient_dense_ratio']:.6f} "
        f"scratch_ratio={candidate['scratch_ratio']:.6f} "
        f"resident_vram_ratio={candidate['resident_vram_ratio']:.6f} "
        f"hot_payload_load_events={candidate['hot_payload_load_events']} "
        f"projected_hot_speedup={candidate['projected_hot_speedup']:.6f} "
        f"claim_device={candidate['claim_device']}"
    )
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
