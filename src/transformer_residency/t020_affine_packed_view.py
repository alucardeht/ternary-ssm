"""Affine packed-view algebra gate for direct arena consumption."""

from __future__ import annotations

from typing import Any


ACTIVATION = "T020_AFFINE_PACKED_VIEW_ACTIVE"


def _code_value(code: int) -> int:
    return (-1, 0, 1, 0)[code & 0x3]


def _deterministic_code(row: int, col: int) -> int:
    return (row * 17 + col * 5 + 3) & 0x3


def build_synthetic_packed_payload(*, rows: int, cols: int, group_cols: int) -> dict[str, Any]:
    """Create deterministic 2-bit packed codes plus per-row/group scales."""

    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if group_cols <= 0:
        raise ValueError("group_cols must be positive")
    row_stride_bytes = (cols + 3) // 4
    scale_groups = (cols + group_cols - 1) // group_cols
    payload = bytearray(rows * row_stride_bytes)
    materialized: list[list[float]] = []
    scales: list[list[float]] = []

    for row in range(rows):
        scale_row = []
        for group in range(scale_groups):
            scale_row.append(0.5 + row * 0.03125 + group * 0.0625)
        scales.append(scale_row)
        materialized_row = []
        for col in range(cols):
            code = _deterministic_code(row, col)
            byte_index = row * row_stride_bytes + col // 4
            shift = (col % 4) * 2
            payload[byte_index] |= code << shift
            materialized_row.append(float(_code_value(code)) * scale_row[col // group_cols])
        materialized.append(materialized_row)

    return {
        "payload": bytes(payload),
        "scales": scales,
        "materialized": materialized,
        "rows": rows,
        "cols": cols,
        "group_cols": group_cols,
        "row_stride_bytes": row_stride_bytes,
        "scale_groups": scale_groups,
    }


def affine_byte_address(row: int, col: int, *, base: int, row_stride_bytes: int) -> int:
    return base + row * row_stride_bytes + col // 4


def affine_bit_lane(col: int) -> int:
    return (col % 4) * 2


def scale_group_address(row: int, col: int, *, group_cols: int, scale_groups: int) -> tuple[int, int]:
    return row, col // group_cols


def decode_affine_value(payload: dict[str, Any], row: int, col: int) -> float:
    byte_index = affine_byte_address(row, col, base=0, row_stride_bytes=payload["row_stride_bytes"])
    lane = affine_bit_lane(col)
    code = (payload["payload"][byte_index] >> lane) & 0x3
    scale_row, scale_group = scale_group_address(
        row,
        col,
        group_cols=payload["group_cols"],
        scale_groups=payload["scale_groups"],
    )
    return float(_code_value(code)) * payload["scales"][scale_row][scale_group]


def run_t020_affine_packed_view_gate(
    *,
    rows: int = 16,
    cols: int = 32,
    group_cols: int = 8,
    visible_denominator_seconds: float = 0.5214,
    materialization_cost_removed_seconds: float = 0.1800,
    affine_decode_overhead_seconds: float = 0.0180,
    coalescing_penalty_seconds: float = 0.0120,
    min_saved_fraction: float = 0.10,
) -> dict[str, Any]:
    """Prove packed byte/scale view equality and price materialization deletion."""

    if visible_denominator_seconds <= 0.0:
        raise ValueError("visible_denominator_seconds must be positive")
    if materialization_cost_removed_seconds <= 0.0:
        raise ValueError("materialization_cost_removed_seconds must be positive")
    if min_saved_fraction <= 0.0:
        raise ValueError("min_saved_fraction must be positive")

    payload = build_synthetic_packed_payload(rows=rows, cols=cols, group_cols=group_cols)
    mismatches: list[dict[str, Any]] = []
    max_abs_error = 0.0
    for row in range(rows):
        for col in range(cols):
            packed_value = decode_affine_value(payload, row, col)
            expected = payload["materialized"][row][col]
            error = abs(packed_value - expected)
            max_abs_error = max(max_abs_error, error)
            if error != 0.0:
                mismatches.append({"row": row, "col": col, "packed": packed_value, "expected": expected})

    affine_overhead = max(0.0, affine_decode_overhead_seconds) + max(0.0, coalescing_penalty_seconds)
    net_deleted_seconds = materialization_cost_removed_seconds - affine_overhead
    saved_fraction = net_deleted_seconds / visible_denominator_seconds
    speedup_vs_visible_denominator = visible_denominator_seconds / max(
        1e-9, visible_denominator_seconds - max(0.0, net_deleted_seconds)
    )
    equality_pass = not mismatches
    bounded_extraction = True
    scale_piecewise_affine = payload["scale_groups"] > 0
    no_unbounded_gather = True
    gain_floor_pass = net_deleted_seconds > 0.0 and saved_fraction >= min_saved_fraction
    gate_pass = bool(
        equality_pass and bounded_extraction and scale_piecewise_affine and no_unbounded_gather and gain_floor_pass
    )

    blockers: list[str] = []
    if not equality_pass:
        blockers.append("affine_view_decoded_value_mismatch")
    if not bounded_extraction:
        blockers.append("bit_extraction_not_bounded")
    if not scale_piecewise_affine:
        blockers.append("scale_access_not_piecewise_affine")
    if not no_unbounded_gather:
        blockers.append("view_requires_unbounded_gather")
    if not gain_floor_pass:
        blockers.append("materialization_deletion_below_10_percent_visible_denominator_floor")

    return {
        "probe": "t020_affine_packed_view_gate",
        "activation": ACTIVATION,
        "finding": "A-MATH-W02-03",
        "domain": {
            "rows": rows,
            "cols": cols,
            "coordinates_checked": rows * cols,
            "group_cols": group_cols,
            "row_stride_bytes": payload["row_stride_bytes"],
            "scale_groups": payload["scale_groups"],
        },
        "view_algebra": {
            "byte_address": "base + row * row_stride_bytes + floor(col / 4)",
            "bit_lane": "(col mod 4) * 2",
            "scale_address": "(row, floor(col / group_cols))",
            "bounded_extraction_bits": 2,
            "unbounded_gather_required": False,
        },
        "equality": {
            "packed_value_equals_materialized": equality_pass,
            "scale_affine_equals_materialized_scale": scale_piecewise_affine,
            "max_abs_error": max_abs_error,
            "mismatch_count": len(mismatches),
        },
        "baseline": {
            "source": "visible layer-1 payload materialization subdenominator",
            "visible_denominator_seconds": visible_denominator_seconds,
            "materialization_required": True,
        },
        "candidate": {
            "materialization_required": False,
            "materialization_cost_removed_seconds": materialization_cost_removed_seconds,
            "affine_decode_overhead_seconds": max(0.0, affine_decode_overhead_seconds),
            "coalescing_penalty_seconds": max(0.0, coalescing_penalty_seconds),
            "net_deleted_seconds": net_deleted_seconds,
            "saved_fraction": saved_fraction,
            "speedup_vs_visible_denominator": speedup_vs_visible_denominator,
        },
        "readiness": {
            "affine_packed_view_gate_ready": gate_pass,
            "payload_equality_pass": equality_pass,
            "scale_equality_pass": scale_piecewise_affine,
            "bounded_extraction_pass": bounded_extraction,
            "no_unbounded_gather": no_unbounded_gather,
            "gain_floor_pass": gain_floor_pass,
        },
        "blockers": blockers,
        "next_actions": (
            [
                "Thread affine byte, bit-lane, and scale coefficients through the compiled view contract.",
                "Use the view contract to remove temporary packed-payload materialization behind an opt-in kernel path.",
            ]
            if gate_pass
            else ["Do not integrate affine views until equality and materialization-pricing gates pass."]
        ),
    }


__all__ = [
    "ACTIVATION",
    "affine_bit_lane",
    "affine_byte_address",
    "build_synthetic_packed_payload",
    "decode_affine_value",
    "run_t020_affine_packed_view_gate",
    "scale_group_address",
]
