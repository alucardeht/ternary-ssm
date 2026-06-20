"""T022 cumulative resident Qwen runtime path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .t017_packed_linear_kernel import run_t017_kernel_qwen_inference
from .t021_cumulative_decoder_runtime_path import run_t021_cumulative_decoder_runtime_path_gate


ACTIVATION = "T022_QWEN_RUNTIME_ACTIVE"


def run_t022_qwen_runtime(
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
    device_payload_budget_gib: float | None = 9.5,
    cache_dir: Path | None = None,
    refresh_cache: bool = False,
) -> dict[str, Any]:
    """Run the cumulative T022 runtime stack on the real packed Qwen path.

    This path moves immutable packed payloads and static dense vectors into
    resident CUDA caches before timing. The measured interval is the decode
    transaction after conversion/preload, so the receipt still exposes preload
    cost separately.
    """

    gate = run_t021_cumulative_decoder_runtime_path_gate()
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
        device_credit_flow=True,
        packed_work_coalescing=True,
    )
    result["probe"] = "t022_qwen_runtime"
    result["activation"] = ACTIVATION
    result["t022_stack"] = {
        "t021_activation": gate["activation"],
        "ordered_runtime_gate_ready": bool(gate["readiness"]["cumulative_decoder_runtime_path_gate_ready"]),
        "items_active": [
            "device_first_payload_cache",
            "preloaded_cuda_payload_cache",
            "dense_tensor_cuda_cache",
            "device_credit_flow_receipt",
            "packed_work_coalescing",
        ],
    }

    readiness = result.get("readiness", {})
    receipt = result.get("receipt") or {}
    cache_stats = receipt.get("cache_stats") or {}
    device_stores = int(cache_stats.get("device_resident_stores", 0))
    device_hits = int(cache_stats.get("device_resident_hits", 0))
    device_failures = int(cache_stats.get("device_resident_failures", 0))
    ready = bool(
        readiness.get("kernel_qwen_inference_ready")
        and readiness.get("packed_work_coalescing_gate_ready")
        and gate["readiness"]["cumulative_decoder_runtime_path_gate_ready"]
        and receipt.get("device_credit_flow")
        and receipt.get("device_payload_cache")
        and receipt.get("preload_device_payloads")
        and receipt.get("dense_tensor_cache")
        and device_stores > 0
        and device_hits > 0
        and device_failures == 0
    )
    readiness["t022_qwen_runtime_ready"] = ready
    readiness["t022_cuda_payload_cache_ready"] = bool(device_stores > 0 and device_hits > 0 and device_failures == 0)
    readiness["t022_dense_tensor_cache_ready"] = bool(receipt.get("dense_tensor_cache"))
    readiness["t022_device_credit_flow_receipt_present"] = bool(receipt.get("device_credit_flow"))
    readiness["t022_ordered_runtime_gate_ready"] = bool(gate["readiness"]["cumulative_decoder_runtime_path_gate_ready"])
    result["next_actions"] = (
        ["Compare measured T022 tok/s against the stored T017 all-64 baseline and decide whether native kernels or graph capture is now the dominant denominator."]
        if ready
        else ["Fix T022 runtime readiness blockers before claiming a real cumulative tok/s improvement."]
    )
    if not ready:
        result.setdefault("blockers", []).append("t022_qwen_runtime_not_ready")
    return result


__all__ = ["ACTIVATION", "run_t022_qwen_runtime"]
