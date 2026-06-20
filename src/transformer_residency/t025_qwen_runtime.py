"""T025 real GPU-resident compact Qwen runtime path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .t017_packed_linear_kernel import run_t017_kernel_qwen_inference
from .t024_exact_quotients import exact_quotient_summary


ACTIVATION = "T025_QWEN_RUNTIME_ACTIVE"


def run_t025_qwen_runtime(
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
    cache_dir: Path | None = None,
    refresh_cache: bool = False,
) -> dict[str, Any]:
    """Run the T025 fail-closed resident compact GPU path.

    T025 is the first real implementation slice from the T025 research round:
    compact trit payloads are moved to CUDA before the measured token loop,
    a device descriptor table is emitted, and the hot path fails closed on any
    payload miss instead of falling back to ``torch.load`` or CPU staging.
    """

    result = run_t017_kernel_qwen_inference(
        model_id=model_id,
        local_model_dir=local_model_dir,
        allow_hf_config_fetch=allow_hf_config_fetch,
        prompt_text=prompt_text,
        max_new_tokens=max_new_tokens,
        layer_limit=layer_limit,
        device=device,
        target_mlp_bits=target_mlp_bits,
        lm_head_chunk_rows=lm_head_chunk_rows,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
        resident_payload_cache=False,
        preload_payloads=False,
        device_payload_cache=True,
        device_payload_budget_gib=device_payload_budget_gib,
        preload_device_payloads=True,
        dense_tensor_cache=True,
        trit_packed_codes=True,
        device_credit_flow=True,
        packed_work_coalescing=True,
        inplace_mlp_intermediate=True,
        fail_closed_device_payloads=True,
        device_payload_descriptor_table=True,
    )
    result["probe"] = "t025_qwen_runtime"
    result["activation"] = ACTIVATION

    quotient_summary = None
    model_dir = result.get("model_dir")
    if model_dir:
        try:
            quotient_summary = exact_quotient_summary(model_dir=Path(model_dir), config=result.get("config") or {})
        except Exception as exc:  # pragma: no cover - diagnostic sidecar
            quotient_summary = {"error": f"{type(exc).__name__}:{exc}"}
    result["t025_exact_quotients"] = quotient_summary

    receipt = result.get("receipt") or {}
    readiness = result.get("readiness", {})
    cache_stats = receipt.get("cache_stats") or {}
    descriptor_table = receipt.get("device_payload_descriptor_table") or {}
    descriptor_count = len(descriptor_table)
    device_resident_bytes = int(cache_stats.get("device_resident_bytes", 0))
    hot_load_events = int(receipt.get("hot_payload_load_events", 0) or 0)
    hot_load_seconds = float(receipt.get("hot_payload_load_seconds", 0.0) or 0.0)
    arena_misses = int(cache_stats.get("device_arena_misses", 0))
    arena_hits = int(cache_stats.get("device_arena_hits", 0))
    ready = bool(
        readiness.get("kernel_qwen_inference_ready")
        and readiness.get("packed_work_coalescing_gate_ready")
        and readiness.get("inplace_mlp_intermediate")
        and receipt.get("device_payload_cache")
        and receipt.get("preload_device_payloads")
        and receipt.get("dense_tensor_cache")
        and receipt.get("trit_packed_codes")
        and receipt.get("fail_closed_device_payloads")
        and descriptor_count > 0
        and device_resident_bytes > 0
        and arena_hits > 0
        and arena_misses == 0
        and hot_load_events == 0
        and hot_load_seconds == 0.0
    )

    result["t025_stack"] = {
        "items_active": [
            "preloaded_cuda_trit_payload_arena",
            "device_payload_descriptor_table",
            "fail_closed_no_miss_hot_contract",
            "dense_tensor_cuda_cache",
            "packed_work_coalescing",
            "device_credit_flow_receipt",
            "radix3_trit_packed_codes",
            "inplace_mlp_gate_up_product",
        ],
        "items_forbidden": [
            "hot_torch_load_payload_miss",
            "cpu_payload_fallback",
            "ram_only_residency_claim",
        ],
        "runtime_policy": "fail_closed_gpu_resident_compact_payloads",
    }
    readiness["t025_qwen_runtime_ready"] = ready
    readiness["t025_fail_closed_no_miss_ready"] = bool(arena_hits > 0 and arena_misses == 0)
    readiness["t025_device_payload_descriptor_count"] = descriptor_count
    readiness["t025_device_payload_resident_bytes"] = device_resident_bytes
    readiness["t025_hot_payload_load_free"] = bool(
        receipt and hot_load_events == 0 and hot_load_seconds == 0.0
    )
    readiness["t025_real_gpu_payload_residency"] = bool(device_resident_bytes > 0 and descriptor_count > 0)
    result["next_actions"] = (
        ["Extend T025 from resident compact affine calls to fused layer decode."]
        if ready
        else ["Fix T025 residency or hot-miss blockers before comparing tok/s."]
    )
    if not ready:
        result.setdefault("blockers", []).append("t025_qwen_runtime_not_ready")
    return result


__all__ = ["ACTIVATION", "run_t025_qwen_runtime"]
