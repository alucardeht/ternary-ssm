"""T024 GPU-bound low-VRAM Qwen runtime path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .t017_packed_linear_kernel import run_t017_kernel_qwen_inference
from .t024_exact_quotients import exact_quotient_summary


ACTIVATION = "T024_QWEN_RUNTIME_ACTIVE"


def run_t024_qwen_runtime(
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
    cache_dir: Path | None = None,
    refresh_cache: bool = False,
) -> dict[str, Any]:
    """Run T024 as a GPU-bound successor to the T023 low-VRAM stack.

    T024 keeps the T023 constraint that packed payloads are not made resident in
    CUDA memory. Its first landed runtime improvement removes the extra MLP
    product tensor by multiplying the gate activation in place before the down
    projection, so the token loop keeps fewer live GPU intermediates without
    changing the math or shifting tensors into CPU/RAM residency.
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
        device_payload_cache=False,
        device_payload_budget_gib=None,
        preload_device_payloads=False,
        dense_tensor_cache=True,
        trit_packed_codes=True,
        device_credit_flow=True,
        packed_work_coalescing=True,
        inplace_mlp_intermediate=True,
    )
    result["probe"] = "t024_qwen_runtime"
    result["activation"] = ACTIVATION
    quotient_summary = None
    model_dir = result.get("model_dir")
    if model_dir:
        try:
            quotient_summary = exact_quotient_summary(model_dir=Path(model_dir), config=result.get("config") or {})
        except Exception as exc:  # pragma: no cover - diagnostic sidecar
            quotient_summary = {"error": f"{type(exc).__name__}:{exc}"}
    result["t024_exact_quotients"] = quotient_summary
    result["t024_stack"] = {
        "items_active": [
            "no_cuda_payload_residency",
            "dense_tensor_cuda_cache",
            "packed_work_coalescing",
            "device_credit_flow_receipt",
            "on_demand_register_decoded_packed_linear",
            "radix3_trit_packed_codes",
            "inplace_mlp_gate_up_product",
            "metadata_exact_quotient_checks",
            "process_ram_receipt",
        ],
        "items_intentionally_inactive": [
            "unbounded_device_payload_residency",
            "preloaded_cuda_payload_cache",
            "cpu_offload_scheduler",
        ],
        "runtime_policy": "gpu_bound_low_vram_token_loop_with_inplace_mlp_intermediate",
    }

    readiness = result.get("readiness", {})
    receipt = result.get("receipt") or {}
    cache_stats = receipt.get("cache_stats") or {}
    ready = bool(
        readiness.get("kernel_qwen_inference_ready")
        and readiness.get("packed_work_coalescing_gate_ready")
        and receipt.get("dense_tensor_cache")
        and not receipt.get("device_payload_cache")
        and not receipt.get("preload_device_payloads")
        and receipt.get("trit_packed_codes")
        and int(cache_stats.get("device_resident_bytes", 0)) == 0
        and receipt.get("device_credit_flow")
        and readiness.get("inplace_mlp_intermediate")
    )
    readiness["t024_qwen_runtime_ready"] = ready
    readiness["t024_gpu_bound_low_vram_ready"] = ready
    readiness["t024_no_cuda_payload_residency"] = int(cache_stats.get("device_resident_bytes", 0)) == 0
    readiness["t024_trit_packed_codes_ready"] = bool(receipt.get("trit_packed_codes"))
    readiness["t024_inplace_mlp_intermediate_ready"] = bool(readiness.get("inplace_mlp_intermediate"))
    readiness["t024_real_runtime_path"] = True
    result["next_actions"] = [] if ready else ["Fix T024 runtime readiness blockers before comparing against T023."]
    if not ready:
        result.setdefault("blockers", []).append("t024_qwen_runtime_not_ready")
    return result


__all__ = ["ACTIVATION", "run_t024_qwen_runtime"]
