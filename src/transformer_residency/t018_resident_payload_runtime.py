"""T018 resident packed payload runtime for the T017 kernel path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .t017_packed_linear_kernel import KernelInferenceReceipt, run_t017_kernel_qwen_inference


ACTIVATION = "T018_RESIDENT_PAYLOAD_RUNTIME_ACTIVE"


def run_t018_resident_payload_qwen_inference(
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
    preload_payloads: bool = True,
    device_credit_flow: bool = False,
    packed_work_coalescing: bool = False,
) -> dict[str, Any]:
    """Run the T017 Qwen path with CPU-resident packed payloads.

    T018 separates packed payload residency from the measured decode path. With
    ``preload_payloads=True``, cache deserialization is measured as preload time
    and decode timing covers the resident payload arena plus packed kernels.
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
        resident_payload_cache=True,
        preload_payloads=preload_payloads,
        device_credit_flow=device_credit_flow,
        packed_work_coalescing=packed_work_coalescing,
    )
    result["probe"] = "t018_resident_payload_qwen_inference"
    result["activation"] = ACTIVATION
    readiness = result.get("readiness", {})
    readiness["resident_payload_qwen_inference_ready"] = bool(readiness.get("kernel_qwen_inference_ready"))
    receipt = result.get("receipt") or {}
    cache_stats = receipt.get("cache_stats") or {}
    resident_hits = int(cache_stats.get("resident_hits", 0))
    resident_stores = int(cache_stats.get("resident_stores", 0))
    result["next_actions"] = (
        ["Benchmark T018 against the latest T017 all-layer hot-cache baseline."]
        if readiness.get("resident_payload_qwen_inference_ready") and resident_hits == 0
        else ["If throughput remains below target, move to T019 pinned/GPU-resident payload staging."]
    )
    if resident_stores == 0 and readiness.get("resident_payload_qwen_inference_ready"):
        result["next_actions"] = ["Investigate why no packed payloads entered the resident arena."]
    return result


__all__ = ["ACTIVATION", "KernelInferenceReceipt", "run_t018_resident_payload_qwen_inference"]
