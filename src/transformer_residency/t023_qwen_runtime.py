"""T023 low-VRAM Qwen runtime path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .t017_packed_linear_kernel import run_t017_kernel_qwen_inference


ACTIVATION = "T023_QWEN_RUNTIME_ACTIVE"


def run_t023_qwen_runtime(
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
    device_payload_budget_gib: float | None = 0.0,
    cache_dir: Path | None = None,
    refresh_cache: bool = False,
) -> dict[str, Any]:
    """Run the T023 low-VRAM stack on the real packed Qwen path.

    This path deliberately avoids CUDA payload residency by default instead of
    using the T022 9.5 GiB preload policy. It keeps the executed token loop real
    while avoiding the T022 failure mode where resident payloads consumed most
    of the card. The active runtime pieces are dense scalar/vector residency,
    packed-work coalescing, and register-decoded packed kernels.
    """

    use_device_payload_cache = bool(device_payload_budget_gib is not None and float(device_payload_budget_gib) > 0.0)

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
        device_payload_cache=use_device_payload_cache,
        device_payload_budget_gib=device_payload_budget_gib if use_device_payload_cache else None,
        preload_device_payloads=use_device_payload_cache,
        dense_tensor_cache=True,
        device_credit_flow=True,
        packed_work_coalescing=True,
    )
    result["probe"] = "t023_qwen_runtime"
    result["activation"] = ACTIVATION
    result["t023_stack"] = {
        "items_active": [
            "no_cuda_payload_residency" if not use_device_payload_cache else "bounded_cuda_payload_residency",
            "dense_tensor_cuda_cache",
            "packed_work_coalescing",
            "device_credit_flow_receipt",
            "on_demand_register_decoded_packed_linear",
        ],
        "items_intentionally_inactive": [
            "unbounded_device_payload_residency",
            "preloaded_cuda_payload_cache" if not use_device_payload_cache else "unbounded_cuda_payload_cache",
            "cpu_offload_scheduler",
        ],
        "runtime_policy": "real_token_loop_with_low_vram_residency",
    }

    readiness = result.get("readiness", {})
    receipt = result.get("receipt") or {}
    cache_stats = receipt.get("cache_stats") or {}
    device_resident_bytes = int(cache_stats.get("device_resident_bytes", 0))
    budget_bytes = None if device_payload_budget_gib is None else int(float(device_payload_budget_gib) * 1024**3)
    device_residency_ready = (
        device_resident_bytes == 0
        if not use_device_payload_cache
        else bool(device_resident_bytes > 0 and (budget_bytes is None or device_resident_bytes <= budget_bytes))
    )
    ready = bool(
        readiness.get("kernel_qwen_inference_ready")
        and readiness.get("packed_work_coalescing_gate_ready")
        and receipt.get("dense_tensor_cache")
        and bool(receipt.get("device_payload_cache")) == use_device_payload_cache
        and bool(receipt.get("preload_device_payloads")) == use_device_payload_cache
        and device_residency_ready
        and receipt.get("device_credit_flow")
    )
    readiness["t023_qwen_runtime_ready"] = ready
    readiness["t023_no_cuda_payload_residency"] = bool(not use_device_payload_cache and device_resident_bytes == 0)
    readiness["t023_bounded_cuda_payload_residency"] = bool(use_device_payload_cache and device_residency_ready)
    readiness["t023_device_payload_budget_gib"] = device_payload_budget_gib
    readiness["t023_dense_tensor_cache_ready"] = bool(receipt.get("dense_tensor_cache"))
    readiness["t023_device_credit_flow_receipt_present"] = bool(receipt.get("device_credit_flow"))
    readiness["t023_real_runtime_path"] = True
    result["next_actions"] = [] if ready else ["Fix T023 runtime readiness blockers before claiming a real low-VRAM runtime improvement."]
    if not ready:
        result.setdefault("blockers", []).append("t023_qwen_runtime_not_ready")
    return result


__all__ = ["ACTIVATION", "run_t023_qwen_runtime"]
