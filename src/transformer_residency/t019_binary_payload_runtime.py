"""T019 binary packed payload runtime for the T017 kernel path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .t017_packed_linear_kernel import run_t017_kernel_qwen_inference


ACTIVATION = "T019_BINARY_PAYLOAD_RUNTIME_ACTIVE"


def run_t019_binary_payload_qwen_inference(
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
    device_credit_flow: bool = False,
    packed_work_coalescing: bool = False,
) -> dict[str, Any]:
    """Run Qwen with raw binary/mmap packed payloads instead of torch.load."""

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
        binary_cache=True,
        device_credit_flow=device_credit_flow,
        packed_work_coalescing=packed_work_coalescing,
    )
    result["probe"] = "t019_binary_payload_qwen_inference"
    result["activation"] = ACTIVATION
    readiness = result.get("readiness", {})
    readiness["binary_payload_qwen_inference_ready"] = bool(readiness.get("kernel_qwen_inference_ready"))
    receipt = result.get("receipt") or {}
    cache_stats = receipt.get("cache_stats") or {}
    binary_loads = int(cache_stats.get("binary_payloads_loaded", 0))
    binary_writes = int(cache_stats.get("binary_payloads_written", 0))
    result["next_actions"] = (
        ["Re-run T019 after binary sidecars are populated to measure the hot binary path."]
        if readiness.get("binary_payload_qwen_inference_ready") and binary_writes > 0 and binary_loads == 0
        else ["If T019 remains below target, the next unresolved blocker is GPU/pinned staged payload transfer."]
    )
    return result


__all__ = ["ACTIVATION", "run_t019_binary_payload_qwen_inference"]
