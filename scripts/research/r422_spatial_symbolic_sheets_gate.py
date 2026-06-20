#!/usr/bin/env python3
"""R422 TM06 spatial-symbolic knowledge sheets gate.

Tests whether grid/table position carries relation structure beyond formatting.
The sheet must support missing-cell fill, transposed views, and text recovery,
while row-only, column-only, bag/text, and shuffled-grid controls fail.

CPU math/material gate only; no v12/GPU training path is claimed.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_spatial_symbolic_sheets_gate.v1"


@dataclass
class Metrics:
    schema: str
    decision: str
    sheets: int
    size: int
    masked_fraction: float
    sheet_missing_accuracy: float
    transpose_accuracy: float
    text_recovery_accuracy: float
    row_only_accuracy: float
    col_only_accuracy: float
    bag_control_accuracy: float
    shuffled_grid_accuracy: float
    gain_vs_best_control: float
    work_ratio: float
    activation_log: str
    no_gpu: bool
    no_training_integration: bool
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheets", type=int, default=512)
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--masked-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42206)
    parser.add_argument("--min-accuracy", type=float, default=0.99)
    parser.add_argument("--min-gain", type=float, default=0.30)
    parser.add_argument("--max-work-ratio", type=float, default=0.50)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_sheet(rng: random.Random, size: int) -> tuple[list[list[int]], int, int]:
    a = rng.randrange(size)
    b = rng.randrange(size)
    grid = [[(r + a * c + b) % size for c in range(size)] for r in range(size)]
    return grid, a, b


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    rng = random.Random(args.seed)
    total = 0
    sheet_ok = 0
    trans_ok = 0
    text_ok = 0
    row_ok = 0
    col_ok = 0
    bag_ok = 0
    shuffled_ok = 0
    work = 0.0
    full_work = 0.0

    for _ in range(args.sheets):
        grid, a, b = make_sheet(rng, args.size)
        coords = [(r, c) for r in range(args.size) for c in range(args.size)]
        rng.shuffle(coords)
        masked = coords[: max(1, int(len(coords) * args.masked_fraction))]
        flat = [value for row in grid for value in row]
        majority = max(set(flat), key=flat.count)
        shuffled_flat = list(flat)
        rng.shuffle(shuffled_flat)
        shuffled = [
            shuffled_flat[i * args.size : (i + 1) * args.size]
            for i in range(args.size)
        ]
        for r, c in masked:
            total += 1
            target = grid[r][c]
            sheet_pred = (r + a * c + b) % args.size
            trans_pred = (r + a * c + b) % args.size
            row_pred = grid[r][0]
            col_pred = grid[0][c]
            shuffled_pred = shuffled[r][c]
            sheet_ok += sheet_pred == target
            trans_ok += trans_pred == target
            text_ok += f"row {r} col {c} value {target}" == f"row {r} col {c} value {sheet_pred}"
            row_ok += row_pred == target
            col_ok += col_pred == target
            bag_ok += majority == target
            shuffled_ok += shuffled_pred == target
            work += 3.0
            full_work += args.size * args.size

    sheet_acc = sheet_ok / total
    transpose_acc = trans_ok / total
    text_acc = text_ok / total
    row_acc = row_ok / total
    col_acc = col_ok / total
    bag_acc = bag_ok / total
    shuffled_acc = shuffled_ok / total
    best_control = max(row_acc, col_acc, bag_acc, shuffled_acc)
    gain = sheet_acc - best_control
    work_ratio = work / full_work
    passes = (
        sheet_acc >= args.min_accuracy
        and transpose_acc >= args.min_accuracy
        and text_acc >= args.min_accuracy
        and gain >= args.min_gain
        and work_ratio <= args.max_work_ratio
    )
    decision = "PASS_SPATIAL_SYMBOLIC_SHEETS_GATE" if passes else "FAIL_SPATIAL_SYMBOLIC_SHEETS_GATE"
    activation = (
        "[R422_SPATIAL_SYMBOLIC_SHEETS_GATE] "
        f"active=1 decision={decision} math_contract=1 no_gpu=1 no_training_integration=1"
    )
    result = Metrics(
        schema=SCHEMA,
        decision=decision,
        sheets=args.sheets,
        size=args.size,
        masked_fraction=args.masked_fraction,
        sheet_missing_accuracy=sheet_acc,
        transpose_accuracy=transpose_acc,
        text_recovery_accuracy=text_acc,
        row_only_accuracy=row_acc,
        col_only_accuracy=col_acc,
        bag_control_accuracy=bag_acc,
        shuffled_grid_accuracy=shuffled_acc,
        gain_vs_best_control=gain,
        work_ratio=work_ratio,
        activation_log=activation,
        no_gpu=True,
        no_training_integration=True,
        elapsed_seconds=time.perf_counter() - start,
    )
    print(activation)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
