#!/usr/bin/env python3
"""R421 Certificate-Class Batch Shape denominator gate.

This CPU-only support gate converts mechanism-specific gate JSON files into a
common original-CE denominator sheet. It is deliberately not a GPU benchmark and
does not tune batch size by occupancy. Its job is to prevent false speed claims
by pricing certificate class, retained CE progress, fallback debt, verification
overhead, and whole-step work share.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA = "r421_certificate_batch_shape_gate.v1"


@dataclass
class ShapeRow:
    name: str
    certificate_class: str
    source: str
    original_ce_progress_ratio: float
    mechanism_work_ratio: float
    whole_step_work_ratio: float
    verification_overhead: float
    fallback_or_debt_fraction: float
    priced_work_ratio: float
    certified_progress_per_paid_work: float
    net_speedup_estimate: float
    pass_gate: bool
    note: str


@dataclass
class SheetResult:
    schema: str
    decision: str
    rows: list[ShapeRow]
    best_row: str
    best_certified_progress_per_paid_work: float
    best_net_speedup_estimate: float
    min_material_speedup: float
    activation_log: str
    elapsed_seconds: float


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_from_tail(path: Path, data: dict[str, Any], overhead: float, min_speedup: float) -> ShapeRow:
    best = data.get("best_result", data.get("best_horizon"))
    if not isinstance(best, dict):
        raise KeyError("tail JSON must contain best_horizon or best_result")
    progress = float(best["actual_progress_ratio"])
    mechanism_ratio = float(best["whole_step_priced_ratio_estimate"])
    debt = float(best["residual_fraction"])
    priced = mechanism_ratio + overhead + debt
    score = progress / max(priced, 1e-12)
    speedup = score
    return ShapeRow(
        name="tail_cotangent_debt",
        certificate_class="recurrent_cotangent_debt",
        source=str(path),
        original_ce_progress_ratio=progress,
        mechanism_work_ratio=mechanism_ratio,
        whole_step_work_ratio=mechanism_ratio,
        verification_overhead=overhead,
        fallback_or_debt_fraction=debt,
        priced_work_ratio=priced,
        certified_progress_per_paid_work=score,
        net_speedup_estimate=speedup,
        pass_gate=speedup >= min_speedup and progress >= 0.99,
        note="Uses whole-step estimate already priced by recurrent backward share.",
    )


def row_from_exact_ce(
    path: Path,
    data: dict[str, Any],
    ce_head_share: float,
    overhead: float,
    min_speedup: float,
) -> ShapeRow:
    progress = 1.0 if float(data["factor_identity_max_abs_error"]) == 0.0 else 0.0
    mechanism_ratio = float(data["expected_candidate_work_ratio"])
    whole_step_ratio = (1.0 - ce_head_share) + ce_head_share * mechanism_ratio
    debt = float(data["fallback_dense_fraction"])
    priced = whole_step_ratio + overhead + debt * ce_head_share
    score = progress / max(priced, 1e-12)
    return ShapeRow(
        name="exact_ce_factor_route",
        certificate_class="factorized_ce_candidate_route",
        source=str(path),
        original_ce_progress_ratio=progress,
        mechanism_work_ratio=mechanism_ratio,
        whole_step_work_ratio=whole_step_ratio,
        verification_overhead=overhead,
        fallback_or_debt_fraction=debt,
        priced_work_ratio=priced,
        certified_progress_per_paid_work=score,
        net_speedup_estimate=score,
        pass_gate=score >= min_speedup and progress == 1.0,
        note="Candidate route coverage is empirical; exact model outside mass is not certified by this row.",
    )


def combo_row(
    tail: ShapeRow,
    exact: ShapeRow,
    overhead: float,
    min_speedup: float,
) -> ShapeRow:
    progress = min(tail.original_ce_progress_ratio, exact.original_ce_progress_ratio)
    whole_step_ratio = tail.whole_step_work_ratio * exact.whole_step_work_ratio
    debt = tail.fallback_or_debt_fraction + exact.fallback_or_debt_fraction
    priced = whole_step_ratio + overhead + debt * 0.05
    score = progress / max(priced, 1e-12)
    return ShapeRow(
        name="combined_tail_and_exact_ce",
        certificate_class="combined_recurrent_and_ce_denominator",
        source=f"{tail.source}; {exact.source}",
        original_ce_progress_ratio=progress,
        mechanism_work_ratio=tail.mechanism_work_ratio * exact.mechanism_work_ratio,
        whole_step_work_ratio=whole_step_ratio,
        verification_overhead=overhead,
        fallback_or_debt_fraction=debt,
        priced_work_ratio=priced,
        certified_progress_per_paid_work=score,
        net_speedup_estimate=score,
        pass_gate=score >= min_speedup and progress >= 0.99,
        note="Multiplicative support row; requires real-path interaction test before use.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tail-json", type=Path, required=True)
    parser.add_argument("--exact-json", type=Path, required=True)
    parser.add_argument("--ce-head-share", type=float, default=0.25)
    parser.add_argument("--tail-verify-overhead", type=float, default=0.03)
    parser.add_argument("--exact-verify-overhead", type=float, default=0.02)
    parser.add_argument("--combo-verify-overhead", type=float, default=0.06)
    parser.add_argument("--min-material-speedup", type=float, default=1.10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    tail_data = load_json(args.tail_json)
    exact_data = load_json(args.exact_json)
    tail = row_from_tail(args.tail_json, tail_data, args.tail_verify_overhead, args.min_material_speedup)
    exact = row_from_exact_ce(
        args.exact_json,
        exact_data,
        args.ce_head_share,
        args.exact_verify_overhead,
        args.min_material_speedup,
    )
    combined = combo_row(tail, exact, args.combo_verify_overhead, args.min_material_speedup)
    rows = [
        ShapeRow(
            name="dense_baseline",
            certificate_class="dense_original_ce",
            source="baseline",
            original_ce_progress_ratio=1.0,
            mechanism_work_ratio=1.0,
            whole_step_work_ratio=1.0,
            verification_overhead=0.0,
            fallback_or_debt_fraction=0.0,
            priced_work_ratio=1.0,
            certified_progress_per_paid_work=1.0,
            net_speedup_estimate=1.0,
            pass_gate=False,
            note="Reference denominator.",
        ),
        tail,
        exact,
        combined,
    ]
    best = max(rows, key=lambda row: row.certified_progress_per_paid_work)
    decision = "PASS_CERTIFICATE_CLASS_DENOMINATOR_SHEET" if any(row.pass_gate for row in rows[1:]) else "FAIL_NO_MATERIAL_CERTIFIED_SHAPE"
    activation = (
        "[R421_CERTIFICATE_BATCH_SHAPE_GATE] "
        f"active=1 decision={decision} no_training_integration=1 no_gpu=1"
    )
    result = SheetResult(
        schema=SCHEMA,
        decision=decision,
        rows=rows,
        best_row=best.name,
        best_certified_progress_per_paid_work=best.certified_progress_per_paid_work,
        best_net_speedup_estimate=best.net_speedup_estimate,
        min_material_speedup=args.min_material_speedup,
        activation_log=activation,
        elapsed_seconds=time.perf_counter() - start,
    )
    print(activation)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
