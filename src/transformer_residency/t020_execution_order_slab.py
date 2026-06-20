"""Execution-order slab arena gate for packed payload acquisition."""

from __future__ import annotations

from typing import Any


ACTIVATION = "T020_EXECUTION_ORDER_SLAB_ARENA_ACTIVE"


def layer1_execution_order_objects(*, lm_head_rows: int = 151936, lm_head_chunk_rows: int = 2048) -> list[dict[str, Any]]:
    """Return the T017 layer-1 packed payload acquisition order."""

    if lm_head_rows <= 0:
        raise ValueError("lm_head_rows must be positive")
    if lm_head_chunk_rows <= 0:
        raise ValueError("lm_head_chunk_rows must be positive")

    names: list[dict[str, Any]] = []
    prefix = "model.layers.0"
    for name in (
        f"{prefix}.self_attn.q_proj.weight",
        f"{prefix}.self_attn.k_proj.weight",
        f"{prefix}.self_attn.v_proj.weight",
        f"{prefix}.self_attn.o_proj.weight",
        f"{prefix}.mlp.gate_proj.weight",
        f"{prefix}.mlp.up_proj.weight",
        f"{prefix}.mlp.down_proj.weight",
    ):
        names.append({"name": name, "start": None, "stop": None, "consumer": "layer0"})
    for start in range(0, lm_head_rows, lm_head_chunk_rows):
        stop = min(lm_head_rows, start + lm_head_chunk_rows)
        names.append({"name": "lm_head.weight", "start": start, "stop": stop, "consumer": "lm_head"})
    return names


def run_t020_execution_order_slab_gate(
    *,
    baseline_objects: int = 82,
    baseline_layer1_acquisition_seconds: float = 0.5214,
    fixed_object_fraction: float = 0.70,
    offset_walk_nanoseconds: float = 50.0,
    min_acquisition_speedup: float = 2.0,
    lm_head_rows: int = 151936,
    lm_head_chunk_rows: int = 2048,
) -> dict[str, Any]:
    """Price a layer-1 execution-order slab against T017 object acquisition."""

    if baseline_objects <= 0:
        raise ValueError("baseline_objects must be positive")
    if baseline_layer1_acquisition_seconds <= 0.0:
        raise ValueError("baseline_layer1_acquisition_seconds must be positive")
    if not 0.0 <= fixed_object_fraction <= 1.0:
        raise ValueError("fixed_object_fraction must be between 0 and 1")
    if min_acquisition_speedup <= 1.0:
        raise ValueError("min_acquisition_speedup must be greater than 1")

    objects = layer1_execution_order_objects(lm_head_rows=lm_head_rows, lm_head_chunk_rows=lm_head_chunk_rows)
    fixed_seconds = baseline_layer1_acquisition_seconds * fixed_object_fraction
    sequential_bytes_seconds = baseline_layer1_acquisition_seconds - fixed_seconds
    per_object_fixed_seconds = fixed_seconds / baseline_objects
    offset_walk_seconds = len(objects) * max(0.0, offset_walk_nanoseconds) * 1.0e-9
    slab_acquisition_seconds = sequential_bytes_seconds + per_object_fixed_seconds + offset_walk_seconds
    acquisition_speedup = baseline_layer1_acquisition_seconds / slab_acquisition_seconds
    gate_pass = acquisition_speedup >= min_acquisition_speedup and len(objects) == baseline_objects

    return {
        "probe": "t020_execution_order_slab_gate",
        "activation": ACTIVATION,
        "finding": "A-MATH-W01-01",
        "baseline": {
            "source": "T017 layer-1 hot repeat acquisition",
            "objects": baseline_objects,
            "acquisition_seconds": baseline_layer1_acquisition_seconds,
            "fixed_object_fraction": fixed_object_fraction,
        },
        "candidate": {
            "objects": 1,
            "offset_entries": len(objects),
            "acquisition_seconds": slab_acquisition_seconds,
            "acquisition_speedup": acquisition_speedup,
            "sequential_bytes_seconds": sequential_bytes_seconds,
            "single_object_fixed_seconds": per_object_fixed_seconds,
            "offset_walk_seconds": offset_walk_seconds,
        },
        "execution_order": objects,
        "readiness": {
            "execution_order_slab_gate_ready": gate_pass,
            "same_payload_object_count": len(objects) == baseline_objects,
            "offset_table_order_objects": len(objects),
            "offset_table_not_block_scale": True,
            "acquisition_speedup_floor_pass": acquisition_speedup >= min_acquisition_speedup,
        },
        "blockers": [] if gate_pass else ["slab_acquisition_model_below_2x_or_object_count_mismatch"],
        "next_actions": (
            [
                "Build the layer slab file format with versioned receipt identity.",
                "Route T017 payload acquisition through the slab reader behind an opt-in flag.",
            ]
            if gate_pass
            else ["Do not build slab runtime until the acquisition-only gate clears 2x."]
        ),
    }


__all__ = ["ACTIVATION", "layer1_execution_order_objects", "run_t020_execution_order_slab_gate"]
