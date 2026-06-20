"""T029 cumulative speed and GPU-memory efficiency path."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .t028_native_generation_speed import run_t028_native_generation_speed


ACTIVATION = "T029_SPEED_GPU_EFFICIENCY_ACTIVE"
R452_T028_TOKENS_PER_SECOND = 0.9280330765589875
R452_T028_PEAK_CUDA_GIB = 10.652063369750977
R452_T028_RESIDENT_CUDA_GIB = 10.458633184432983
BYTES_PER_GIB = 1024.0**3


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _descriptor_memory_audit(receipt: dict[str, Any]) -> dict[str, Any]:
    descriptor_table = receipt.get("device_payload_descriptor_table") or {}
    actual_device_bytes = _safe_int((receipt.get("cache_stats") or {}).get("device_resident_bytes"))
    group4_equivalent_code_bytes = 0
    actual_code_bytes = 0
    mlp_group4_equivalent_code_bytes = 0
    mlp_actual_code_bytes = 0
    descriptor_count = 0
    compact20_equivalent_descriptor_count = 0

    for descriptor in descriptor_table.values():
        descriptor_count += 1
        tensors = descriptor.get("tensors") or {}
        trit = tensors.get("trit_codes") or {}
        shape = descriptor.get("shape") or [0, 0]
        if len(shape) < 2:
            continue
        rows = _safe_int(shape[0])
        cols = _safe_int(shape[1])
        baseline_code_bytes = rows * math.ceil(cols / 4)
        candidate_code_bytes = _safe_int(trit.get("bytes"))
        group4_equivalent_code_bytes += baseline_code_bytes
        actual_code_bytes += candidate_code_bytes
        if int(descriptor.get("trit_group_size") or 0) in (5, 20):
            compact20_equivalent_descriptor_count += 1
        key_name = str(descriptor.get("name") or descriptor.get("key") or "")
        if ".mlp." in key_name:
            mlp_group4_equivalent_code_bytes += baseline_code_bytes
            mlp_actual_code_bytes += candidate_code_bytes

    code_saved_bytes = max(0, group4_equivalent_code_bytes - actual_code_bytes)
    mlp_code_saved_bytes = max(0, mlp_group4_equivalent_code_bytes - mlp_actual_code_bytes)
    group4_equivalent_device_bytes = actual_device_bytes + code_saved_bytes
    return {
        "source": "runtime_device_payload_descriptor_table",
        "descriptor_count": descriptor_count,
        "compact20_equivalent_descriptor_count": compact20_equivalent_descriptor_count,
        "actual_device_resident_bytes": actual_device_bytes,
        "actual_device_resident_gib": actual_device_bytes / BYTES_PER_GIB,
        "group4_equivalent_device_resident_bytes": group4_equivalent_device_bytes,
        "group4_equivalent_device_resident_gib": group4_equivalent_device_bytes / BYTES_PER_GIB,
        "code_bytes_group4_equivalent": group4_equivalent_code_bytes,
        "code_bytes_actual": actual_code_bytes,
        "code_bytes_saved_vs_group4": code_saved_bytes,
        "code_gib_saved_vs_group4": code_saved_bytes / BYTES_PER_GIB,
        "mlp_code_bytes_saved_vs_group4": mlp_code_saved_bytes,
        "mlp_code_gib_saved_vs_group4": mlp_code_saved_bytes / BYTES_PER_GIB,
        "r452_t028_resident_cuda_gib": R452_T028_RESIDENT_CUDA_GIB,
        "resident_gib_delta_vs_r452_t028": (actual_device_bytes / BYTES_PER_GIB) - R452_T028_RESIDENT_CUDA_GIB,
        "implemented_memory_findings": [
            "M-MATH-W01-01 exact 20-trit base-3 word recode, implemented as four exact 5-trit byte lanes",
            "M-MATH-W01-04 MLP-only exact 20-trit recode slice as a subset of the active full-code recode",
        ],
        "implemented_as_exclusive_alternatives": [
            "M-MATH-W01-02 entropy-tight ternary bitstream remains an incompatible codec alternative to the active aligned 20-trit words",
            "M-PB-W01-01 quaternary repair-state override remains an incompatible repair-index codec alternative",
            "M-MATH-W01-08 joint repair exception packet remains a repair entropy extension, not additive to this measurement",
        ],
    }


def run_t029_speed_gpu_efficiency(
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
    """Run T028 with the exact 20-trit-equivalent resident payload format enabled."""

    result = run_t028_native_generation_speed(
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
        trit_code_group_size=5,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
        fused_qkv_compact_projection_kernel=fused_qkv_compact_projection_kernel,
    )
    result["probe"] = "t029_speed_gpu_efficiency"
    result["activation"] = ACTIVATION

    receipt = result.get("receipt") or {}
    readiness = result.get("readiness", {})
    memory_audit = _descriptor_memory_audit(receipt)
    measured_tok_s = _safe_float(receipt.get("tokens_per_second"))
    peak_cuda = _safe_float(receipt.get("peak_cuda_allocated_gib"))
    process_working_set = receipt.get("process_working_set_gib")
    process_private = receipt.get("process_private_usage_gib")

    result["t029_memory_audit"] = memory_audit
    result["t029_speed_efficiency_comparison"] = {
        "baseline_source": "R452/T028",
        "baseline_tokens_per_second": R452_T028_TOKENS_PER_SECOND,
        "measured_tokens_per_second": measured_tok_s,
        "measured_speedup_vs_r452_t028": measured_tok_s / R452_T028_TOKENS_PER_SECOND
        if measured_tok_s > 0.0
        else 0.0,
        "baseline_peak_cuda_allocated_gib": R452_T028_PEAK_CUDA_GIB,
        "measured_peak_cuda_allocated_gib": peak_cuda,
        "peak_cuda_delta_gib": peak_cuda - R452_T028_PEAK_CUDA_GIB if peak_cuda > 0.0 else None,
        "baseline_resident_cuda_gib": R452_T028_RESIDENT_CUDA_GIB,
        "measured_resident_cuda_gib": memory_audit["actual_device_resident_gib"],
        "resident_cuda_delta_gib": memory_audit["actual_device_resident_gib"] - R452_T028_RESIDENT_CUDA_GIB,
        "process_working_set_gib": process_working_set,
        "process_private_usage_gib": process_private,
        "measured_crosses_40_tok_s": measured_tok_s >= 40.0,
        "production_tok_s_claim_allowed": measured_tok_s >= R452_T028_TOKENS_PER_SECOND * 1.10,
    }
    result["t029_stack"] = {
        "items_active": [
            "t028_native_generation_speed_stack",
            "m_math_w01_01_exact_20_trit_base3_word_recode",
            "m_math_w01_04_mlp_20_trit_slice_subsumed_by_full_recode",
            "s_math_w01_02_native_layer_supernode_contract_from_t028",
            "s_pb_w01_01_resident_whole_token_service_graph_contract_from_t028",
            "s_pb_w01_09_exact_multi_token_device_chunk_contract_from_t028",
            "s_math_w01_06_incremental_kv_contract_from_t028",
            "s_math_w01_13_streaming_mlp_down_projection_contract_from_t028",
            *(
                ["t030_compatible_fused_qkv_compact_projection_kernel"]
                if receipt.get("fused_qkv_compact_projection_kernel")
                else []
            ),
        ],
        "items_exclusive_not_coactivated": memory_audit["implemented_as_exclusive_alternatives"],
        "items_not_integrated_into_real_hot_loop": result.get("t028_stack", {}).get(
            "items_not_yet_integrated_into_hot_loop",
            [],
        ),
        "runtime_policy": "cumulative_t029_t028_plus_exact_20_trit_resident_payloads",
    }
    readiness["t029_speed_gpu_efficiency_ready"] = bool(
        readiness.get("smoke_pass")
        and memory_audit["compact20_equivalent_descriptor_count"] == memory_audit["descriptor_count"]
        and _safe_int(receipt.get("hot_payload_load_events")) == 0
    )
    readiness["t029_exact_20_trit_payloads_active"] = (
        memory_audit["compact20_equivalent_descriptor_count"] == memory_audit["descriptor_count"]
    )
    readiness["t029_code_bytes_saved_positive"] = memory_audit["code_bytes_saved_vs_group4"] > 0
    readiness["t029_code_gib_saved_vs_group4"] = memory_audit["code_gib_saved_vs_group4"]
    readiness["t029_measured_tokens_per_second"] = measured_tok_s
    if not readiness["t029_speed_gpu_efficiency_ready"]:
        result.setdefault("blockers", []).append("t029_speed_gpu_efficiency_not_ready")
    result["next_actions"] = [
        "If T029 is slower, keep the 20-trit resident format only as the memory branch and implement a low-division decode kernel before promotion.",
        "The incompatible repair/entropy codec findings need a separate exact codec branch before they can replace the active 20-trit format.",
    ]
    return result


__all__ = ["ACTIVATION", "run_t029_speed_gpu_efficiency"]
