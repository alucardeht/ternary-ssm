"""T021 cumulative decoder runtime path for Qwen inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .t017_packed_linear_kernel import run_t017_kernel_qwen_inference
from .t021_cumulative_decoder_runtime_path import run_t021_cumulative_decoder_runtime_path_gate


ACTIVATION = "T021_QWEN_RUNTIME_ACTIVE"


def run_t021_qwen_runtime(
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
    """Run T021 as a cumulative layer on the real packed Qwen runtime.

    Payload population is deliberately outside the measured decode interval.
    The measured tok/s is for the resident T021 runtime path; preload cost stays
    visible as `preload_seconds` in the receipt.
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
        resident_payload_cache=True,
        preload_payloads=True,
        device_credit_flow=True,
        packed_work_coalescing=True,
    )
    result["probe"] = "t021_qwen_runtime"
    result["activation"] = ACTIVATION
    result["t021_cumulative_gate"] = {
        "activation": gate["activation"],
        "readiness": gate["readiness"],
        "candidate": gate["candidate"],
        "blockers": gate["blockers"],
    }

    readiness = result.get("readiness", {})
    receipt = result.get("receipt") or {}
    cache_stats = receipt.get("cache_stats") or {}
    resident_stores = int(cache_stats.get("resident_stores", 0))
    t021_ready = bool(
        readiness.get("kernel_qwen_inference_ready")
        and gate["readiness"]["cumulative_decoder_runtime_path_gate_ready"]
        and readiness.get("device_credit_flow_gate_ready")
        and readiness.get("packed_work_coalescing_gate_ready")
        and receipt.get("resident_payload_cache")
        and resident_stores > 0
    )
    readiness["t021_qwen_runtime_ready"] = t021_ready
    readiness["t021_cumulative_gate_ready"] = bool(gate["readiness"]["cumulative_decoder_runtime_path_gate_ready"])
    readiness["t021_runtime_payload_reuse_ready"] = bool(receipt.get("resident_payload_cache") and resident_stores > 0)
    readiness["t021_runtime_path"] = True
    tokens = len(receipt.get("generated_token_ids") or [])
    elapsed = float(receipt.get("elapsed_seconds") or 0.0)
    cache_load_seconds = float(cache_stats.get("load_seconds") or 0.0)
    preload_seconds = float(receipt.get("preload_seconds") or 0.0)
    decode_without_population = max(0.0, elapsed - max(0.0, cache_load_seconds - preload_seconds))
    adjusted_tokens_per_second = tokens / decode_without_population if tokens > 0 and decode_without_population > 0.0 else 0.0
    readiness["t021_decode_seconds_excluding_payload_population"] = decode_without_population
    readiness["t021_tokens_per_second_excluding_payload_population"] = adjusted_tokens_per_second
    result["next_actions"] = (
        ["Compare --t021-qwen-runtime against --t017-kernel-qwen-inference on the same prompt and token count."]
        if t021_ready
        else ["Fix the T021 runtime readiness blockers before comparing against T017."]
    )
    if not t021_ready:
        result.setdefault("blockers", []).append("t021_qwen_runtime_not_ready")
    return result


__all__ = ["ACTIVATION", "run_t021_qwen_runtime"]
