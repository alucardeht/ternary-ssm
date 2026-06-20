"""T030 cumulative speed and GPU-resident efficiency path."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .t029_speed_gpu_efficiency import (
    BYTES_PER_GIB,
    R452_T028_PEAK_CUDA_GIB,
    R452_T028_RESIDENT_CUDA_GIB,
    R452_T028_TOKENS_PER_SECOND,
    _descriptor_memory_audit,
    _safe_float,
    _safe_int,
    run_t029_speed_gpu_efficiency,
)


ACTIVATION = "T030_SPEED_GPU_EFFICIENCY_ACTIVE"
TARGET_TOKENS_PER_SECOND = 40.0
DIRECT_5TRIT_SAVING_RATE = 0.103303327402366
MIN_STRUCTURAL_COVERAGE_FOR_1GIB = 1.0 / R452_T028_RESIDENT_CUDA_GIB


def _projection_table() -> dict[str, Any]:
    """Return the validator-adjusted T030 projection envelope."""

    return {
        "source": ".workspace/research/t030-speed-gpu-efficiency-extraordinary/top-10-projections.md",
        "baseline": {
            "label": "R452/T028 measured hot-loop baseline",
            "tokens_per_second": R452_T028_TOKENS_PER_SECOND,
            "resident_cuda_gib": R452_T028_RESIDENT_CUDA_GIB,
            "peak_cuda_gib": R452_T028_PEAK_CUDA_GIB,
        },
        "first_gate": {
            "ids": ["T030-GTV-01", "T030-GM-V01"],
            "tokens_per_second": [1.1, 1.7],
            "resident_cuda_gib": R452_T028_RESIDENT_CUDA_GIB * (1.0 - DIRECT_5TRIT_SAVING_RATE),
            "peak_cuda_gib": R452_T028_PEAK_CUDA_GIB - (R452_T028_RESIDENT_CUDA_GIB * DIRECT_5TRIT_SAVING_RATE),
            "requires_no_scratch_rebound": True,
        },
        "near_stack_if_gates_pass": {
            "tokens_per_second": [2.0, 6.0],
            "crosses_40_tok_s": False,
            "claim_boundary": "projection_only_until_real_measurement",
        },
        "structural_memory_gate": {
            "minimum_net_disjoint_coverage_for_1gib": MIN_STRUCTURAL_COVERAGE_FOR_1GIB,
            "coverage_source_required": "runtime_descriptor_table_or_direct_payload_scan",
        },
    }


def _descriptor_profile(receipt: dict[str, Any]) -> dict[str, Any]:
    descriptor_table = receipt.get("device_payload_descriptor_table") or {}
    total_elements = 0
    total_code_bytes = 0
    mlp_elements = 0
    attention_elements = 0
    head_elements = 0
    group5_descriptors = 0
    group4_equivalent_bytes = 0
    descriptor_names: list[str] = []

    for descriptor in descriptor_table.values():
        name = str(descriptor.get("name") or descriptor.get("key") or "")
        shape = descriptor.get("shape") or [0, 0]
        if len(shape) < 2:
            continue
        rows = _safe_int(shape[0])
        cols = _safe_int(shape[1])
        elements = max(0, rows * cols)
        trit = (descriptor.get("tensors") or {}).get("trit_codes") or {}
        code_bytes = _safe_int(trit.get("bytes"))
        total_elements += elements
        total_code_bytes += code_bytes
        group4_equivalent_bytes += rows * math.ceil(cols / 4)
        if int(descriptor.get("trit_group_size") or 0) == 5:
            group5_descriptors += 1
        if ".mlp." in name:
            mlp_elements += elements
        elif ".self_attn." in name:
            attention_elements += elements
        elif name == "lm_head.weight":
            head_elements += elements
        descriptor_names.append(name)

    descriptor_count = len(descriptor_table)
    quotient_scan_coverage = 0.0
    zero_skew_scan_coverage = 0.0
    residual_basis_scan_coverage = 0.0
    if total_elements > 0:
        quotient_scan_coverage = (mlp_elements + attention_elements) / total_elements
        zero_skew_scan_coverage = mlp_elements / total_elements
        residual_basis_scan_coverage = attention_elements / total_elements

    return {
        "source": "runtime_device_payload_descriptor_table",
        "descriptor_count": descriptor_count,
        "group5_descriptors": group5_descriptors,
        "all_descriptors_group5": descriptor_count > 0 and group5_descriptors == descriptor_count,
        "total_elements_described": total_elements,
        "total_code_bytes": total_code_bytes,
        "group4_equivalent_code_bytes": group4_equivalent_bytes,
        "mlp_element_fraction": mlp_elements / total_elements if total_elements > 0 else 0.0,
        "attention_element_fraction": attention_elements / total_elements if total_elements > 0 else 0.0,
        "head_element_fraction": head_elements / total_elements if total_elements > 0 else 0.0,
        "quotient_metadata_candidate_coverage": quotient_scan_coverage,
        "zero_skew_metadata_candidate_coverage": zero_skew_scan_coverage,
        "residual_basis_metadata_candidate_coverage": residual_basis_scan_coverage,
        "quotient_could_exceed_1gib_if_exact_scan_positive": quotient_scan_coverage >= MIN_STRUCTURAL_COVERAGE_FOR_1GIB,
        "zero_skew_could_exceed_1gib_if_exact_scan_positive": zero_skew_scan_coverage >= MIN_STRUCTURAL_COVERAGE_FOR_1GIB,
        "residual_basis_could_exceed_1gib_if_exact_scan_positive": residual_basis_scan_coverage
        >= MIN_STRUCTURAL_COVERAGE_FOR_1GIB,
        "descriptor_name_sample": descriptor_names[:12],
    }


def _t030_gate_audit(result: dict[str, Any], memory_audit: dict[str, Any]) -> dict[str, Any]:
    receipt = result.get("receipt") or {}
    readiness = result.get("readiness") or {}
    cache_stats = receipt.get("cache_stats") or {}
    descriptor_table = receipt.get("device_payload_descriptor_table") or {}
    descriptor_profile = _descriptor_profile(receipt)
    measured_tok_s = _safe_float(receipt.get("tokens_per_second"))
    comparable_to_r452 = _safe_int(receipt.get("requested_layers")) >= 64
    resident_gib = _safe_float(memory_audit.get("actual_device_resident_gib"))
    peak_gib = _safe_float(receipt.get("peak_cuda_allocated_gib"))
    hot_load_events = _safe_int(receipt.get("hot_payload_load_events"))
    hot_load_seconds = _safe_float(receipt.get("hot_payload_load_seconds"))
    device_resident_bytes = _safe_int(cache_stats.get("device_resident_bytes"))
    arena_misses = _safe_int(cache_stats.get("device_arena_misses"))
    arena_hits = _safe_int(cache_stats.get("device_arena_hits"))
    group5_active = bool(descriptor_profile["all_descriptors_group5"])

    return {
        "activation_codeword_direct_5trit_microdot": {
            "ids": ["T030-GTV-01", "T030-GM-V01"],
            "status": "implemented_hot_loop_gate" if group5_active else "blocked",
            "ready": bool(group5_active and readiness.get("t029_speed_gpu_efficiency_ready")),
            "trit_code_group_size": _safe_int(receipt.get("trit_code_group_size")),
            "code_gib_saved_vs_group4": memory_audit.get("code_gib_saved_vs_group4"),
            "resident_cuda_gib": resident_gib,
            "peak_cuda_gib": peak_gib,
            "speed_floor_preserved_vs_r452": (
                measured_tok_s >= R452_T028_TOKENS_PER_SECOND if comparable_to_r452 else None
            ),
            "comparable_to_r452": comparable_to_r452,
        },
        "same_input_projection_bundle": {
            "id": "T030-GTV-02",
            "status": "implemented_hot_loop_gate",
            "ready": bool(receipt.get("fused_mlp_gate_up_kernel")),
            "implemented_boundary": "mlp_gate_up_pair_kernel_with_qkv_triple_kernel_available_as_explicit_opt_in",
            "qkv_triple_kernel_active": bool(receipt.get("fused_qkv_compact_projection_kernel")),
            "remaining_boundary": (
                None
                if receipt.get("fused_qkv_compact_projection_kernel")
                else "qkv_triple_kernel_not_default_promoted_after_full64_speed_regression"
            ),
        },
        "resident_descriptor_tape": {
            "id": "T030-GTV-03",
            "status": "implemented_runtime_contract",
            "ready": bool(
                descriptor_table
                and receipt.get("device_payload_cache")
                and receipt.get("preload_device_payloads")
                and arena_hits > 0
                and arena_misses == 0
                and hot_load_events == 0
                and hot_load_seconds == 0.0
            ),
            "descriptor_count": len(descriptor_table),
            "device_arena_hits": arena_hits,
            "device_arena_misses": arena_misses,
            "hot_payload_load_events": hot_load_events,
            "hot_payload_load_seconds": hot_load_seconds,
        },
        "mlp_supernode": {
            "id": "T030-GTV-04",
            "status": "implemented_hot_loop_gate",
            "ready": bool(
                receipt.get("fused_mlp_gate_up_kernel")
                and receipt.get("packed_work_coalescing")
                and readiness.get("t028_native_compact_affine_contract_ready")
            ),
            "fused_gate_up": bool(receipt.get("fused_mlp_gate_up_kernel")),
            "residual_down_coalescing": bool(receipt.get("packed_work_coalescing")),
        },
        "attention_qkv_kv_supernode": {
            "id": "T030-GTV-05",
            "status": "implemented_hot_loop_gate",
            "ready": bool(receipt.get("native_attention_sdp")),
            "implemented_boundary": "causal_sdpa_attention_and_o_projection_residual_coalescing_with_qkv_triple_kernel_available_as_explicit_opt_in",
            "qkv_triple_kernel_active": bool(receipt.get("fused_qkv_compact_projection_kernel")),
            "remaining_boundary": (
                None
                if receipt.get("fused_qkv_compact_projection_kernel")
                else "qkv_triple_kernel_not_default_promoted_after_full64_speed_regression"
            ),
        },
        "consumer_transposed_layout": {
            "id": "T030-GTV-06",
            "status": "implemented_layout_contract",
            "ready": bool(
                _safe_int(receipt.get("packed_linear_block_m")) > 0
                and _safe_int(receipt.get("packed_linear_block_c")) > 0
                and _safe_int(receipt.get("packed_linear_block_r")) > 0
            ),
            "block_m": receipt.get("packed_linear_block_m"),
            "block_c": receipt.get("packed_linear_block_c"),
            "block_r": receipt.get("packed_linear_block_r"),
        },
        "static_instruction_families": {
            "id": "T030-GTV-07",
            "status": "implemented_descriptor_contract",
            "ready": bool(descriptor_table and descriptor_profile["total_code_bytes"] > 0),
            "descriptor_profile": descriptor_profile,
        },
        "exact_tile_quotient": {
            "id": "T030-GM-V02",
            "status": "implemented_metadata_gate",
            "ready": bool(descriptor_profile["quotient_could_exceed_1gib_if_exact_scan_positive"]),
            "minimum_net_coverage_for_1gib": MIN_STRUCTURAL_COVERAGE_FOR_1GIB,
            "candidate_coverage": descriptor_profile["quotient_metadata_candidate_coverage"],
            "claim_boundary": "coverage_only_not_exact_quotient_payload_saving",
        },
        "zero_skew_superblocks": {
            "id": "T030-GM-V03",
            "status": "implemented_metadata_gate",
            "ready": bool(descriptor_profile["zero_skew_could_exceed_1gib_if_exact_scan_positive"]),
            "minimum_net_coverage_for_1gib": MIN_STRUCTURAL_COVERAGE_FOR_1GIB,
            "candidate_coverage": descriptor_profile["zero_skew_metadata_candidate_coverage"],
            "claim_boundary": "coverage_only_not_exact_zero_skew_saving",
        },
        "cross_layer_residual_basis": {
            "id": "T030-GM-V04",
            "status": "implemented_metadata_gate",
            "ready": bool(descriptor_profile["residual_basis_could_exceed_1gib_if_exact_scan_positive"]),
            "minimum_net_coverage_for_1gib": MIN_STRUCTURAL_COVERAGE_FOR_1GIB,
            "candidate_coverage": descriptor_profile["residual_basis_metadata_candidate_coverage"],
            "claim_boundary": "coverage_only_not_cross_layer_basis_factorization",
        },
        "gpu_bound_no_offload_guard": {
            "ready": bool(
                receipt.get("device") == "cuda"
                and device_resident_bytes > 0
                and receipt.get("device_payload_cache")
                and receipt.get("preload_device_payloads")
                and not receipt.get("resident_payload_cache")
                and hot_load_events == 0
            ),
            "device": receipt.get("device"),
            "device_resident_bytes": device_resident_bytes,
            "resident_payload_cache": bool(receipt.get("resident_payload_cache")),
            "device_payload_cache": bool(receipt.get("device_payload_cache")),
        },
    }


def _speed_efficiency_comparison(receipt: dict[str, Any], memory_audit: dict[str, Any]) -> dict[str, Any]:
    measured_tok_s = _safe_float(receipt.get("tokens_per_second"))
    peak_cuda = _safe_float(receipt.get("peak_cuda_allocated_gib"))
    resident_cuda = _safe_float(memory_audit.get("actual_device_resident_gib"))
    comparable_to_r452 = _safe_int(receipt.get("requested_layers")) >= 64
    return {
        "baseline_source": "R452/T028",
        "comparison_scope": "full_all_layer" if comparable_to_r452 else "non_comparable_layer_limited_smoke",
        "baseline_tokens_per_second": R452_T028_TOKENS_PER_SECOND,
        "measured_tokens_per_second": measured_tok_s,
        "measured_speedup_vs_r452_t028": measured_tok_s / R452_T028_TOKENS_PER_SECOND
        if measured_tok_s > 0.0 and comparable_to_r452
        else None,
        "baseline_peak_cuda_allocated_gib": R452_T028_PEAK_CUDA_GIB,
        "measured_peak_cuda_allocated_gib": peak_cuda,
        "peak_cuda_delta_gib": peak_cuda - R452_T028_PEAK_CUDA_GIB if peak_cuda > 0.0 and comparable_to_r452 else None,
        "baseline_resident_cuda_gib": R452_T028_RESIDENT_CUDA_GIB,
        "measured_resident_cuda_gib": resident_cuda,
        "resident_cuda_delta_gib": resident_cuda - R452_T028_RESIDENT_CUDA_GIB if comparable_to_r452 else None,
        "measured_crosses_40_tok_s": measured_tok_s >= TARGET_TOKENS_PER_SECOND,
        "measured_clears_10_percent_materiality": comparable_to_r452
        and measured_tok_s >= R452_T028_TOKENS_PER_SECOND * 1.10,
        "production_tok_s_claim_allowed": comparable_to_r452
        and measured_tok_s >= R452_T028_TOKENS_PER_SECOND * 1.10,
    }


def run_t030_speed_gpu_efficiency(
    *,
    model_id: str = "Qwen/Qwen3-32B",
    local_model_dir: Path | None = None,
    allow_hf_config_fetch: bool = True,
    prompt_text: str = "Local compressed Qwen smoke:",
    max_new_tokens: int = 1,
    layer_limit: int | None = 1,
    device: str = "auto",
    target_mlp_bits: float = 2.45,
    lm_head_chunk_rows: int = 2048,
    device_payload_budget_gib: float | None = 10.5,
    packed_linear_block_m: int = 8,
    packed_linear_block_c: int = 128,
    packed_linear_block_r: int = 32,
    cache_dir: Path | None = None,
    refresh_cache: bool = False,
    fused_qkv_compact_projection_kernel: bool = False,
) -> dict[str, Any]:
    """Run the T030 cumulative stack over the real T029/T028 path."""

    result = run_t029_speed_gpu_efficiency(
        model_id=model_id,
        local_model_dir=local_model_dir,
        allow_hf_config_fetch=allow_hf_config_fetch,
        prompt_text=prompt_text,
        max_new_tokens=max_new_tokens,
        layer_limit=layer_limit,
        device=device,
        target_mlp_bits=target_mlp_bits,
        lm_head_chunk_rows=lm_head_chunk_rows,
        device_payload_budget_gib=device_payload_budget_gib,
        packed_linear_block_m=packed_linear_block_m,
        packed_linear_block_c=packed_linear_block_c,
        packed_linear_block_r=packed_linear_block_r,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
        fused_qkv_compact_projection_kernel=fused_qkv_compact_projection_kernel,
    )
    result["probe"] = "t030_speed_gpu_efficiency"
    result["activation"] = ACTIVATION

    receipt = result.get("receipt") or {}
    readiness = result.get("readiness", {})
    memory_audit = _descriptor_memory_audit(receipt)
    gate_audit = _t030_gate_audit(result, memory_audit)
    comparison = _speed_efficiency_comparison(receipt, memory_audit)

    all_core_ready = bool(
        gate_audit["activation_codeword_direct_5trit_microdot"]["ready"]
        and gate_audit["same_input_projection_bundle"]["ready"]
        and gate_audit["resident_descriptor_tape"]["ready"]
        and gate_audit["mlp_supernode"]["ready"]
        and gate_audit["attention_qkv_kv_supernode"]["ready"]
        and gate_audit["consumer_transposed_layout"]["ready"]
        and gate_audit["static_instruction_families"]["ready"]
        and gate_audit["gpu_bound_no_offload_guard"]["ready"]
    )
    structural_metadata_ready = bool(
        gate_audit["exact_tile_quotient"]["ready"]
        or gate_audit["zero_skew_superblocks"]["ready"]
        or gate_audit["cross_layer_residual_basis"]["ready"]
    )

    result["t030_projection"] = _projection_table()
    result["t030_gate_audit"] = gate_audit
    result["t030_memory_audit"] = memory_audit
    result["t030_speed_efficiency_comparison"] = comparison
    result["t030_stack"] = {
        "items_active": [
            "t029_exact_20_trit_equivalent_gpu_payload_memory_branch",
            "t028_native_generation_speed_stack",
            "t030_gtv_01_activation_codeword_direct_5trit_microdot",
            "t030_gtv_02_same_input_mlp_gate_up_projection_bundle",
            "t030_gtv_03_resident_descriptor_tape_persistent_token_contract",
            "t030_gtv_04_mlp_gate_up_down_supernode_contract",
            "t030_gtv_05_attention_sdpa_qkv_kv_supernode_boundary",
            "t030_gtv_06_consumer_transposed_layout_lane_width_selection",
            "t030_gtv_07_static_compact_affine_instruction_family_descriptor",
            "t030_gm_v02_exact_tile_quotient_metadata_gate",
            "t030_gm_v03_zero_skew_superblock_metadata_gate",
            "t030_gm_v04_cross_layer_residual_basis_metadata_gate",
        ],
        "items_hot_loop_integrated": [
            "direct_5trit_payload_consumption",
            *(
                ["qkv_single_compact_projection_kernel"]
                if receipt.get("fused_qkv_compact_projection_kernel")
                else []
            ),
            "mlp_gate_up_pair_kernel",
            "mlp_down_residual_coalescing",
            "native_attention_sdp",
            "device_payload_descriptor_table",
            "fail_closed_device_payload_arena",
            "gpu_argmax_head",
        ],
        "items_metadata_only": [
            "exact_tile_quotient_coverage_scan",
            "zero_skew_superblock_coverage_scan",
            "cross_layer_residual_basis_coverage_scan",
        ],
        "items_not_integrated_into_hot_loop": [
            *(
                []
                if receipt.get("fused_qkv_compact_projection_kernel")
                else ["qkv_single_compact_projection_kernel"]
            ),
            "real_all_layer_cuda_graph_capture",
            "structural_tile_quotient_direct_consumption_kernel",
            "zero_skew_superblock_direct_consumption_kernel",
            "cross_layer_residual_basis_factorized_runtime",
        ],
        "runtime_policy": "cumulative_t030_t029_t028_gpu_bound_no_offload_stack",
    }
    readiness["t030_speed_gpu_efficiency_ready"] = all_core_ready
    readiness["t030_structural_memory_metadata_ready"] = structural_metadata_ready
    readiness["t030_gpu_bound_no_offload_ready"] = gate_audit["gpu_bound_no_offload_guard"]["ready"]
    readiness["t030_production_tok_s_claim_allowed"] = comparison["production_tok_s_claim_allowed"]
    readiness["t030_measured_crosses_40_tok_s"] = comparison["measured_crosses_40_tok_s"]
    readiness["t030_full_all_layer_gpu_vram_win_ready"] = bool(
        comparison["comparison_scope"] == "full_all_layer"
        and comparison["resident_cuda_delta_gib"] is not None
        and comparison["resident_cuda_delta_gib"] <= -1.0
        and gate_audit["gpu_bound_no_offload_guard"]["ready"]
    )
    readiness["t030_all_top10_items_accounted"] = True
    if not all_core_ready:
        result.setdefault("blockers", []).append("t030_core_hot_loop_stack_not_ready")
    if not all_core_ready:
        result["next_actions"] = ["Fix T030 core hot-loop readiness before making any speed or VRAM claim."]
    elif comparison["comparison_scope"] != "full_all_layer":
        result["next_actions"] = [
            "Measure T030 against R452/T028 in the same clean window and keep it only if speed does not regress.",
            "Promote the structural memory items only after direct payload scans prove disjoint exact savings above 1 GiB.",
        ]
    elif comparison["production_tok_s_claim_allowed"]:
        result["next_actions"] = [
            "Repeat the full all-layer T030 speed-positive result before any default promotion.",
            "Keep structural memory items gated until direct payload scans prove disjoint exact savings above 1 GiB.",
        ]
    else:
        result["next_actions"] = [
            "Do not promote T030 as a speed path; keep it as a measured GPU-resident VRAM branch.",
            "Target the next speed implementation at real all-layer graph capture or compact-affine supernodes.",
        ]
    return result


__all__ = ["ACTIVATION", "run_t030_speed_gpu_efficiency"]
