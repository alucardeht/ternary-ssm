"""T027 cumulative generation-speed Qwen runtime path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .t017_packed_linear_kernel import run_t017_kernel_qwen_inference
from .t024_exact_quotients import exact_quotient_summary


ACTIVATION = "T027_GENERATION_SPEED_STACK_ACTIVE"
R445_TOKENS_PER_SECOND = 0.8650030409252335
R445_SECONDS_PER_TOKEN = 1.156065299990587


def _projection_table() -> dict[str, Any]:
    """Return the validated T027 projection envelope from the round synthesis."""

    return {
        "source": ".workspace/research/t027-generation-speed-extraordinary/synthesis.md",
        "baseline": {
            "label": "R445/T026 measured baseline",
            "seconds_per_token": R445_SECONDS_PER_TOKEN,
            "tokens_per_second": R445_TOKENS_PER_SECOND,
        },
        "validated_continuous_generation": {
            "conservative_tokens_per_second": [11.8, 22.2],
            "central_tokens_per_second": [16.7, 31.3],
            "optimistic_tokens_per_second": [23.8, 43.5],
            "robustly_crosses_40_tok_s": False,
        },
        "conditional_next_token_only": {
            "tokens_per_second": [33.3, 71.4],
            "counts_as_continuous_generation": False,
        },
    }


def _runtime_speed_comparison(receipt: dict[str, Any] | None) -> dict[str, Any]:
    measured_tok_s = float((receipt or {}).get("tokens_per_second") or 0.0)
    measured_s_per_token = 1.0 / measured_tok_s if measured_tok_s > 0.0 else None
    return {
        "baseline_source": "R445/T026",
        "baseline_tokens_per_second": R445_TOKENS_PER_SECOND,
        "baseline_seconds_per_token": R445_SECONDS_PER_TOKEN,
        "measured_tokens_per_second": measured_tok_s,
        "measured_seconds_per_token": measured_s_per_token,
        "measured_speedup_vs_r445": measured_tok_s / R445_TOKENS_PER_SECOND if measured_tok_s > 0.0 else 0.0,
        "measured_crosses_40_tok_s": measured_tok_s >= 40.0,
    }


def run_t027_qwen_runtime(
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
    packed_linear_block_m: int = 16,
    packed_linear_block_c: int = 128,
    packed_linear_block_r: int = 32,
    trit_code_group_size: int = 4,
    cache_dir: Path | None = None,
    refresh_cache: bool = False,
    preload_prompt_embeddings: bool = False,
    native_attention_sdp: bool = False,
    fused_qkv_compact_projection_kernel: bool = False,
) -> dict[str, Any]:
    """Run the T027 cumulative generation-speed stack on the real Qwen path.

    T027 keeps the T025/T026 GPU-resident, fail-closed compact payload contract
    and activates all existing exact runtime mechanisms that correspond to the
    validated T027 generation-speed round. The path is deliberately cumulative:
    it does not disable the T025/T026 residency and GPU-owned orchestration
    stack while enabling A01/A02/A03 mechanisms.
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
        defer_kernel_synchronization=True,
        gpu_argmax_head=True,
        packed_linear_block_m=packed_linear_block_m,
        packed_linear_block_c=packed_linear_block_c,
        packed_linear_block_r=packed_linear_block_r,
        trit_code_group_size=trit_code_group_size,
        fused_mlp_gate_up_kernel=True,
        fused_qkv_compact_projection_kernel=fused_qkv_compact_projection_kernel,
        preload_prompt_embeddings=preload_prompt_embeddings,
        native_attention_sdp=native_attention_sdp,
    )
    result["probe"] = "t027_qwen_runtime"
    result["activation"] = ACTIVATION

    quotient_summary = None
    model_dir = result.get("model_dir")
    if model_dir:
        try:
            quotient_summary = exact_quotient_summary(model_dir=Path(model_dir), config=result.get("config") or {})
        except Exception as exc:  # pragma: no cover - diagnostic sidecar
            quotient_summary = {"error": f"{type(exc).__name__}:{exc}"}
    result["t027_exact_quotients"] = quotient_summary

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
        and receipt.get("defer_kernel_synchronization")
        and receipt.get("gpu_argmax_head")
        and receipt.get("fused_mlp_gate_up_kernel")
        and (not fused_qkv_compact_projection_kernel or receipt.get("fused_qkv_compact_projection_kernel"))
        and descriptor_count > 0
        and device_resident_bytes > 0
        and arena_hits > 0
        and arena_misses == 0
        and hot_load_events == 0
        and hot_load_seconds == 0.0
    )

    result["t027_stack"] = {
        "items_active": [
            "t025_preloaded_cuda_trit_payload_arena",
            "t025_device_payload_descriptor_table",
            "t025_fail_closed_no_miss_hot_contract",
            "t026_deferred_cuda_projection_synchronization",
            "a01_exact_compact_affine_trit_codes",
            *(
                ["m_math_w01_01_exact_20_trit_base3_words"]
                if trit_code_group_size in (5, 20)
                else []
            ),
            "a01_static_descriptor_exact_quotients",
            "a01_mixed_exact_core_selector_contract",
            "a02_dense_tensor_cuda_cache",
            "a02_device_credit_flow_receipt",
            "a02_packed_work_coalescing",
            "a02_inplace_mlp_gate_up_product",
            "a02_fused_gate_up_pair_kernel",
            *(
                ["a02_fused_qkv_compact_projection_kernel"]
                if receipt.get("fused_qkv_compact_projection_kernel")
                else []
            ),
            "a03_gpu_greedy_lm_head_tournament",
            "a03_lazy_exact_greedy_selection_contract",
            *(
                ["a02_prompt_embedding_preloaded_device_rows"]
                if receipt.get("prompt_embeddings_preloaded")
                else []
            ),
            *(
                ["a01_attention_sdp_streaming_boundary"]
                if receipt.get("native_attention_sdp")
                else []
            ),
        ],
        "items_forbidden": [
            "per_projection_cuda_synchronize",
            "hot_torch_load_payload_miss",
            "cpu_payload_fallback",
            "ram_only_residency_claim",
            "approximate_logits",
            "quality_tradeoff",
        ],
        "runtime_policy": "cumulative_gpu_resident_generation_speed_stack",
    }
    result["t027_projection"] = _projection_table()
    result["t027_speed_comparison"] = _runtime_speed_comparison(receipt)

    readiness["t027_qwen_runtime_ready"] = ready
    readiness["t027_cumulative_over_t026"] = True
    readiness["t027_fail_closed_no_miss_ready"] = bool(arena_hits > 0 and arena_misses == 0)
    readiness["t027_device_payload_descriptor_count"] = descriptor_count
    readiness["t027_device_payload_resident_bytes"] = device_resident_bytes
    readiness["t027_hot_payload_load_free"] = bool(
        receipt and hot_load_events == 0 and hot_load_seconds == 0.0
    )
    readiness["t027_deferred_sync_ready"] = bool(receipt.get("defer_kernel_synchronization"))
    readiness["t027_gpu_greedy_tournament_ready"] = bool(receipt.get("gpu_argmax_head"))
    readiness["t027_fused_mlp_gate_up_ready"] = bool(receipt.get("fused_mlp_gate_up_kernel"))
    readiness["t027_fused_qkv_compact_projection_ready"] = bool(receipt.get("fused_qkv_compact_projection_kernel"))
    readiness["t027_prompt_embedding_preload_ready"] = bool(receipt.get("prompt_embeddings_preloaded"))
    readiness["t027_native_attention_sdp_ready"] = bool(receipt.get("native_attention_sdp"))
    readiness["t027_trit_code_group_size"] = int(receipt.get("trit_code_group_size") or trit_code_group_size)
    readiness["t027_real_gpu_payload_residency"] = bool(device_resident_bytes > 0 and descriptor_count > 0)
    readiness["t027_projection_crosses_40_only_optimistic_edge"] = True
    readiness["t027_projection_robustly_crosses_40_tok_s"] = False
    result["next_actions"] = (
        ["Replace the per-projection Triton runtime with a native compact layer/supernode kernel before expecting T027 projections to approach 40 tok/s."]
        if ready
        else ["Fix T027 cumulative runtime blockers before comparing tok/s."]
    )
    if not ready:
        result.setdefault("blockers", []).append("t027_qwen_runtime_not_ready")
    return result


__all__ = ["ACTIVATION", "run_t027_qwen_runtime"]
