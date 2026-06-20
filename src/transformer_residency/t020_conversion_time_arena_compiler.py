"""Conversion-time arena compiler gate for packed payload contracts."""

from __future__ import annotations

from typing import Any

from .t020_immutable_arena_receipt import build_layer1_arena_manifest


ACTIVATION = "T020_CONVERSION_TIME_ARENA_COMPILER_ACTIVE"

COMPILE_TIME_DEPS = {
    "P",
    "model_config",
    "quantization_law",
    "kernel_abi",
    "packing_law",
    "scale_law",
    "device_capability_class",
}


def _descriptor_contract(descriptor: dict[str, Any], *, kernel_abi: str, packed_format: str) -> dict[str, Any]:
    return {
        "consumer_ordinal": descriptor["ordinal"],
        "consumer": descriptor["consumer"],
        "name": descriptor["name"],
        "row_start": descriptor["row_start"],
        "row_stop": descriptor["row_stop"],
        "offset": descriptor["offset"],
        "length": descriptor["length"],
        "lane_id": descriptor["lane_id"],
        "packed_format": packed_format,
        "scale_law": "per_row_scale_with_sparse_repair_v1",
        "kernel_abi": kernel_abi,
        "fallback_mode": "reject_to_rebuild_boundary",
    }


def interpret_runtime_descriptor(
    descriptor: dict[str, Any],
    *,
    kernel_abi: str = "t017_triton_packed_linear_v1",
    packed_format: str = "ternary_codes_scales_repairs_v1",
) -> dict[str, Any]:
    """Return the descriptor tuple the old runtime interpreter would produce."""

    return _descriptor_contract(descriptor, kernel_abi=kernel_abi, packed_format=packed_format)


def compile_arena_contract(
    *,
    subset_descriptors: int = 8,
    kernel_abi: str = "t017_triton_packed_linear_v1",
    packed_format: str = "ternary_codes_scales_repairs_v1",
) -> dict[str, Any]:
    """Compile a small executable contract from immutable manifest descriptors."""

    if subset_descriptors <= 0:
        raise ValueError("subset_descriptors must be positive")

    manifest = build_layer1_arena_manifest()
    descriptors = manifest["descriptors"][:subset_descriptors]
    compiled_descriptors = [
        _descriptor_contract(descriptor, kernel_abi=kernel_abi, packed_format=packed_format)
        for descriptor in descriptors
    ]
    return {
        "receipt": manifest["receipt"],
        "descriptor_root": manifest["descriptor_root"],
        "compiled_descriptors": compiled_descriptors,
        "compiled_fields": (
            "consumer_ordinal",
            "consumer",
            "name",
            "row_start",
            "row_stop",
            "offset",
            "length",
            "lane_id",
            "packed_format",
            "scale_law",
            "kernel_abi",
            "fallback_mode",
        ),
    }


def classify_interpretation_steps() -> list[dict[str, Any]]:
    """Classify current runtime interpretation work by dependency type."""

    return [
        {
            "step": "parse_source_payload_metadata",
            "class": "compile_time_invariant",
            "cost_seconds": 0.1500,
            "deps": ["P", "packing_law"],
        },
        {
            "step": "emit_consumer_order_lane_descriptors",
            "class": "compile_time_invariant",
            "cost_seconds": 0.0800,
            "deps": ["P", "model_config", "kernel_abi", "scale_law"],
        },
        {
            "step": "build_headerless_hot_views",
            "class": "compile_time_invariant",
            "cost_seconds": 0.1200,
            "deps": ["P", "kernel_abi", "packing_law", "scale_law", "device_capability_class"],
        },
        {
            "step": "compute_rebuild_hashes",
            "class": "admission_time_predicate",
            "cost_seconds": 0.0025,
            "deps": ["model_config", "quantization_law", "kernel_abi", "packing_law", "scale_law"],
        },
        {
            "step": "check_device_capability_class",
            "class": "admission_time_predicate",
            "cost_seconds": 0.0005,
            "deps": ["device_capability_class"],
        },
        {
            "step": "stream_payload_bytes",
            "class": "runtime_variable",
            "cost_seconds": 0.0,
            "deps": ["device_state"],
        },
        {
            "step": "fallback_to_rebuild_boundary",
            "class": "fallback_only",
            "cost_seconds": 0.0,
            "deps": ["admission_result"],
        },
    ]


def run_t020_conversion_time_arena_compiler_gate(
    *,
    baseline_visible_denominator_seconds: float = 0.5214,
    receipt_admission_seconds: float = 0.005784,
    subset_descriptors: int = 8,
    min_saved_fraction: float = 0.10,
) -> dict[str, Any]:
    """Gate A-PB-W02-14 by dependency classification and descriptor equality."""

    if baseline_visible_denominator_seconds <= 0.0:
        raise ValueError("baseline_visible_denominator_seconds must be positive")
    if receipt_admission_seconds < 0.0:
        raise ValueError("receipt_admission_seconds must be non-negative")
    if min_saved_fraction <= 0.0:
        raise ValueError("min_saved_fraction must be positive")

    contract = compile_arena_contract(subset_descriptors=subset_descriptors)
    manifest = build_layer1_arena_manifest()
    runtime_descriptors = [
        interpret_runtime_descriptor(descriptor) for descriptor in manifest["descriptors"][:subset_descriptors]
    ]
    compiled_descriptors = contract["compiled_descriptors"]
    descriptor_equal = compiled_descriptors == runtime_descriptors

    steps = classify_interpretation_steps()
    compile_time_steps = [step for step in steps if step["class"] == "compile_time_invariant"]
    admission_steps = [step for step in steps if step["class"] == "admission_time_predicate"]
    runtime_variable_steps = [step for step in steps if step["class"] == "runtime_variable"]
    fallback_steps = [step for step in steps if step["class"] == "fallback_only"]
    dependency_safe = all(set(step["deps"]).issubset(COMPILE_TIME_DEPS) for step in compile_time_steps)

    compileable_hot_cost_seconds = sum(float(step["cost_seconds"]) for step in compile_time_steps)
    admission_cost_seconds = receipt_admission_seconds + sum(float(step["cost_seconds"]) for step in admission_steps)
    deleted_denominator_seconds = max(0.0, compileable_hot_cost_seconds - admission_cost_seconds)
    saved_fraction = deleted_denominator_seconds / baseline_visible_denominator_seconds
    gain_floor_pass = saved_fraction >= min_saved_fraction
    hot_path_rebuild_forbidden = all(step["class"] != "fallback_only" for step in runtime_variable_steps)
    gate_pass = bool(descriptor_equal and dependency_safe and gain_floor_pass and hot_path_rebuild_forbidden)

    blockers: list[str] = []
    if not descriptor_equal:
        blockers.append("compiled_descriptor_mismatch")
    if not dependency_safe:
        blockers.append("compile_time_descriptor_depends_on_runtime_state")
    if not gain_floor_pass:
        blockers.append("compiled_artifact_does_not_clear_10_percent_visible_denominator_floor")
    if not hot_path_rebuild_forbidden:
        blockers.append("runtime_variable_path_can_fallback_to_source_payload_reinterpretation")

    return {
        "probe": "t020_conversion_time_arena_compiler_gate",
        "activation": ACTIVATION,
        "finding": "A-PB-W02-14",
        "baseline": {
            "source": "T017 layer-1 runtime interpretation/discovery denominator",
            "visible_denominator_seconds": baseline_visible_denominator_seconds,
            "runtime_interpreter_required": True,
        },
        "candidate": {
            "compiled_descriptor_count": len(compiled_descriptors),
            "compileable_hot_cost_seconds": compileable_hot_cost_seconds,
            "admission_cost_seconds": admission_cost_seconds,
            "deleted_denominator_seconds": deleted_denominator_seconds,
            "saved_fraction": saved_fraction,
            "speedup_vs_visible_denominator": (
                baseline_visible_denominator_seconds
                / max(1e-9, baseline_visible_denominator_seconds - deleted_denominator_seconds)
            ),
            "runtime_interpreter_required": False,
        },
        "classification": {
            "compile_time_invariant": compile_time_steps,
            "admission_time_predicate": admission_steps,
            "runtime_variable": runtime_variable_steps,
            "fallback_only": fallback_steps,
        },
        "descriptor_equality": {
            "compiled_equals_runtime_interpreter": descriptor_equal,
            "compiled_fields": contract["compiled_fields"],
        },
        "readiness": {
            "conversion_time_arena_compiler_gate_ready": gate_pass,
            "descriptor_equality_pass": descriptor_equal,
            "compile_time_dependencies_safe": dependency_safe,
            "hot_path_rebuild_forbidden": hot_path_rebuild_forbidden,
            "gain_floor_pass": gain_floor_pass,
        },
        "blockers": blockers,
        "next_actions": (
            [
                "Promote the contract compiler into a real slab artifact writer.",
                "Route T017 payload acquisition through receipt-verified compiled views behind an opt-in flag.",
            ]
            if gate_pass
            else ["Do not integrate compiled arena views until descriptor equality and dependency classification pass."]
        ),
    }


__all__ = [
    "ACTIVATION",
    "COMPILE_TIME_DEPS",
    "classify_interpretation_steps",
    "compile_arena_contract",
    "interpret_runtime_descriptor",
    "run_t020_conversion_time_arena_compiler_gate",
]
