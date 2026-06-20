"""T034 GPU-real runtime substrate gates."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from .footprint import BYTES_PER_GIB
from .t004_real_qwen_bridge import find_weight_dir, inventory_safetensors, resolve_qwen_config
from .t008_compressed_generation_smoke import _load_weight_map
from .t015_packed_qwen_inference import _read_rows, _select_device
from .t016_persistent_packed_cache import PackedCacheStats, _default_cache_dir, _payload_nbytes
from .t017_packed_linear_kernel import (
    _payload_from_cache_or_resident,
    _relative_rmse,
    _time_cuda,
    packed_linear_triton,
    run_t017_kernel_qwen_inference,
)


ACTIVATION = "T034_GPU_REAL_RUNTIME_GATES_ACTIVE"
R459_T033_ONE_LAYER_TOKENS_PER_SECOND = 2.884471992687007
R459_T033_ONE_LAYER_PEAK_CUDA_GIB = 0.5345873832702637
R459_T033_ONE_LAYER_RESIDENT_CUDA_GIB = 0.5114963054656982
TARGET_TOKENS_PER_SECOND = 40.0


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


def _tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().contiguous().cpu().float().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _run_structural_execution_view_gate(
    *,
    weight_dir: Path,
    weight_map: dict[str, str],
    cache_dir: Path,
    device: str,
    target_mlp_bits: float,
    trit_code_group_size: int,
    packed_linear_block_m: int,
    packed_linear_block_c: int,
    packed_linear_block_r: int,
    refresh_cache: bool,
    iterations: int,
) -> dict[str, Any]:
    import torch

    stats_trit = PackedCacheStats(cache_dir=str(cache_dir))
    stats_code = PackedCacheStats(cache_dir=str(cache_dir))
    projection_name = "model.layers.0.self_attn.q_proj.weight"
    x = _read_rows(weight_dir, weight_map, "model.embed_tokens.weight", [0], device=device)
    current_payload = _payload_from_cache_or_resident(
        model_dir=weight_dir,
        weight_map=weight_map,
        cache_dir=cache_dir,
        stats=stats_trit,
        name=projection_name,
        target_mlp_bits=target_mlp_bits,
        refresh_cache=refresh_cache,
        payload_cache=None,
        device_payload_cache={},
        device_payload_cache_device=device,
        trit_packed_codes=True,
        trit_code_group_size=trit_code_group_size,
    )
    execution_view_payload = _payload_from_cache_or_resident(
        model_dir=weight_dir,
        weight_map=weight_map,
        cache_dir=cache_dir,
        stats=stats_code,
        name=projection_name,
        target_mlp_bits=target_mlp_bits,
        refresh_cache=refresh_cache,
        payload_cache=None,
        device_payload_cache={},
        device_payload_cache_device=device,
        trit_packed_codes=False,
    )

    current = packed_linear_triton(
        x,
        current_payload,
        block_m=packed_linear_block_m,
        block_c=packed_linear_block_c,
        block_r=packed_linear_block_r,
    )
    execution_view = packed_linear_triton(
        x,
        execution_view_payload,
        block_m=packed_linear_block_m,
        block_c=packed_linear_block_c,
        block_r=packed_linear_block_r,
    )
    torch.cuda.synchronize()
    max_abs_error = float(torch.max(torch.abs(current.float() - execution_view.float())).detach().cpu().item())
    relative_rmse = _relative_rmse(execution_view, current)
    output_equivalent = max_abs_error <= 2.0e-3 and relative_rmse <= 2.0e-4

    current_seconds = _time_cuda(
        lambda: packed_linear_triton(
            x,
            current_payload,
            block_m=packed_linear_block_m,
            block_c=packed_linear_block_c,
            block_r=packed_linear_block_r,
        ),
        iterations=iterations,
    )
    execution_view_seconds = _time_cuda(
        lambda: packed_linear_triton(
            x,
            execution_view_payload,
            block_m=packed_linear_block_m,
            block_c=packed_linear_block_c,
            block_r=packed_linear_block_r,
        ),
        iterations=iterations,
    )

    current_bytes = _payload_nbytes(current_payload)
    execution_view_bytes = _payload_nbytes(execution_view_payload)
    speedup = current_seconds / execution_view_seconds if execution_view_seconds > 0.0 else 0.0
    return {
        "id": "T034-A-VAL01",
        "status": "implemented_real_gpu_gate",
        "projection": projection_name,
        "mechanism": "non_dense_structural_execution_view_against_current_compact_affine",
        "device": device,
        "output_equivalent": output_equivalent,
        "max_abs_error": max_abs_error,
        "relative_rmse": relative_rmse,
        "current_seconds_per_iter": current_seconds,
        "execution_view_seconds_per_iter": execution_view_seconds,
        "speedup_vs_current_compact_affine": speedup,
        "current_payload_bytes": current_bytes,
        "execution_view_payload_bytes": execution_view_bytes,
        "resident_growth_gib": (execution_view_bytes - current_bytes) / BYTES_PER_GIB,
        "current_output_sha256": _tensor_sha256(current),
        "execution_view_output_sha256": _tensor_sha256(execution_view),
        "ready_for_promotion": bool(output_equivalent and speedup >= 1.10 and execution_view_bytes <= current_bytes),
        "claim_boundary": "one_real_projection_gpu_gate_not_full_layer_or_full_model_speed_claim",
        "no_dense_shadow": True,
        "no_cpu_offload": True,
    }


def _run_segment_transaction_gate(*, device: str, iterations: int) -> dict[str, Any]:
    import torch

    width = 5120
    k = 2
    gen = torch.Generator(device="cpu").manual_seed(34034)
    deltas = torch.randn((k, width), generator=gen, dtype=torch.float32).to(device=device)
    serial_tokens = torch.tensor([89637, 220], device=device, dtype=torch.int64)
    packet_tokens = serial_tokens.clone()
    serial_state = torch.empty_like(deltas)
    pending_state = torch.empty_like(deltas)
    committed_state = torch.empty_like(deltas)
    status_word = torch.empty((1,), device=device, dtype=torch.int32)

    def serial_once() -> Any:
        serial_state[0].copy_(deltas[0])
        serial_state[1].copy_(deltas[1])
        return serial_state

    def transaction_once() -> Any:
        pending_state.copy_(deltas)
        tokens_match = torch.all(packet_tokens.eq(serial_tokens))
        status_word.fill_(1)
        status_word.mul_(tokens_match.to(dtype=torch.int32))
        committed_state.copy_(pending_state)
        return committed_state

    serial_once()
    transaction_once()
    torch.cuda.synchronize()
    state_equal = bool(torch.equal(serial_state, committed_state))
    tokens_equal = bool(torch.equal(serial_tokens, packet_tokens))
    status_value = int(status_word.detach().cpu().item())
    serial_digest = _tensor_sha256(serial_state)
    transaction_digest = _tensor_sha256(committed_state)

    serial_seconds = _time_cuda(serial_once, iterations=iterations)
    transaction_seconds = _time_cuda(transaction_once, iterations=iterations)
    speedup = serial_seconds / transaction_seconds if transaction_seconds > 0.0 else 0.0
    scratch_bytes = pending_state.numel() * pending_state.element_size()
    scratch_bytes += committed_state.numel() * committed_state.element_size()
    scratch_bytes += packet_tokens.numel() * packet_tokens.element_size()
    scratch_bytes += status_word.numel() * status_word.element_size()
    return {
        "id": "T034-B-V01-V02",
        "status": "implemented_real_gpu_gate",
        "mechanism": "exact_gpu_segment_transaction_with_pending_pages_and_device_status",
        "device": device,
        "segment_tokens": k,
        "state_equal_to_serial": state_equal,
        "tokens_equal_to_serial": tokens_equal,
        "device_status_word": status_value,
        "serial_state_sha256": serial_digest,
        "transaction_state_sha256": transaction_digest,
        "serial_seconds_per_iter": serial_seconds,
        "transaction_seconds_per_iter": transaction_seconds,
        "speedup_vs_serial_state_mutation": speedup,
        "scratch_gib": scratch_bytes / BYTES_PER_GIB,
        "host_acceptance_before_commit": False,
        "ready_for_promotion": bool(state_equal and tokens_equal and status_value == 1 and speedup >= 1.10),
        "claim_boundary": "real_gpu_state_transaction_gate_not_full_qwen_decode_state_machine",
        "no_cpu_offload": True,
    }


def _run_t034_runtime_path(
    *,
    model_id: str,
    local_model_dir: Path | None,
    allow_hf_config_fetch: bool,
    prompt_text: str,
    max_new_tokens: int,
    layer_limit: int | None,
    device: str,
    target_mlp_bits: float,
    lm_head_chunk_rows: int,
    device_payload_budget_gib: float | None,
    packed_linear_block_m: int,
    packed_linear_block_c: int,
    packed_linear_block_r: int,
    cache_dir: Path | None,
    refresh_cache: bool,
) -> dict[str, Any]:
    return run_t017_kernel_qwen_inference(
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
        trit_code_group_size=5,
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
        fused_mlp_gate_up_kernel=True,
        fused_qkv_compact_projection_kernel=False,
        preload_prompt_embeddings=True,
        native_attention_sdp=True,
        t034_device_argmax_embedding_splice=True,
    )


def run_t034_gpu_real_runtime_gates(
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
    gate_iterations: int = 10,
) -> dict[str, Any]:
    """Run the T034 cumulative GPU-real gates on the current packed Qwen path."""

    config = resolve_qwen_config(
        model_id=model_id,
        local_model_dir=local_model_dir,
        allow_hf_config_fetch=allow_hf_config_fetch,
    )
    weight_dir = find_weight_dir(config, local_model_dir)
    inventory = inventory_safetensors(weight_dir)
    blockers: list[str] = []
    structural_gate: dict[str, Any] | None = None
    transaction_gate: dict[str, Any] | None = None
    runtime_result: dict[str, Any] | None = None

    try:
        runtime_device = _select_device(device)
    except Exception as exc:
        runtime_device = "unavailable"
        blockers.append(f"device_selection_failed:{type(exc).__name__}:{exc}")

    if runtime_device != "cuda":
        blockers.append("t034_requires_cuda")
    if not config.matches_project_qwen3_32b_shape():
        blockers.append("config_shape_mismatch")
    if weight_dir is None or not inventory.local_weights_present:
        blockers.append("missing_local_qwen3_32b_safetensors")

    if not blockers and weight_dir is not None:
        weight_map = _load_weight_map(weight_dir)
        resolved_cache_dir = cache_dir or _default_cache_dir(weight_dir)
        structural_gate = _run_structural_execution_view_gate(
            weight_dir=weight_dir,
            weight_map=weight_map,
            cache_dir=resolved_cache_dir,
            device=runtime_device,
            target_mlp_bits=target_mlp_bits,
            trit_code_group_size=5,
            packed_linear_block_m=packed_linear_block_m,
            packed_linear_block_c=packed_linear_block_c,
            packed_linear_block_r=packed_linear_block_r,
            refresh_cache=refresh_cache,
            iterations=gate_iterations,
        )
        transaction_gate = _run_segment_transaction_gate(device=runtime_device, iterations=gate_iterations)
        runtime_result = _run_t034_runtime_path(
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
        )

    receipt = (runtime_result or {}).get("receipt") or {}
    readiness = (runtime_result or {}).get("readiness") or {}
    splice = (receipt.get("t034_device_argmax_embedding_splice") or {}).get("lm_head_certificate") or {}
    measured_tok_s = _safe_float(receipt.get("tokens_per_second"))
    peak_cuda = _safe_float(receipt.get("peak_cuda_allocated_gib"))
    resident_cuda = _safe_int((receipt.get("cache_stats") or {}).get("device_resident_bytes")) / BYTES_PER_GIB
    comparable_to_r459 = _safe_int(receipt.get("requested_layers")) == 1
    comparison = {
        "baseline_source": "R459/T033 one-layer smoke",
        "comparison_scope": "one_layer" if comparable_to_r459 else "non_comparable_layer_count",
        "baseline_tokens_per_second": R459_T033_ONE_LAYER_TOKENS_PER_SECOND,
        "measured_tokens_per_second": measured_tok_s,
        "measured_speedup_vs_r459_t033": (
            measured_tok_s / R459_T033_ONE_LAYER_TOKENS_PER_SECOND
            if measured_tok_s > 0.0 and comparable_to_r459
            else None
        ),
        "baseline_peak_cuda_gib": R459_T033_ONE_LAYER_PEAK_CUDA_GIB,
        "measured_peak_cuda_gib": peak_cuda,
        "peak_cuda_delta_gib": peak_cuda - R459_T033_ONE_LAYER_PEAK_CUDA_GIB if comparable_to_r459 else None,
        "baseline_resident_cuda_gib": R459_T033_ONE_LAYER_RESIDENT_CUDA_GIB,
        "measured_resident_cuda_gib": resident_cuda,
        "resident_cuda_delta_gib": resident_cuda - R459_T033_ONE_LAYER_RESIDENT_CUDA_GIB if comparable_to_r459 else None,
        "measured_crosses_40_tok_s": measured_tok_s >= TARGET_TOKENS_PER_SECOND,
        "production_tok_s_claim_allowed": bool(
            comparable_to_r459 and measured_tok_s >= R459_T033_ONE_LAYER_TOKENS_PER_SECOND * 1.10
        ),
    }

    structural_ready = bool(structural_gate and structural_gate["ready_for_promotion"])
    transaction_ready = bool(transaction_gate and transaction_gate["ready_for_promotion"])
    splice_ready = bool(splice.get("next_embedding_device_closure_proven") and not splice.get("host_token_id_dependency_before_next_embedding"))
    runtime_ready = bool(readiness.get("smoke_pass") and splice_ready)
    t034_ready = bool(runtime_ready and structural_ready and transaction_ready)
    next_actions = []
    if not structural_ready:
        next_actions.append("Replace the structural execution view with a faster direct-consumption kernel before promotion.")
    if not transaction_ready:
        next_actions.append("Embed the segment transaction into the real Qwen decode state machine before promotion.")
    if not splice_ready:
        next_actions.append("Fix device argmax-to-embedding closure before any whole-token graph capture attempt.")
    if runtime_ready and not comparison["production_tok_s_claim_allowed"]:
        next_actions.append("Do not claim speed; target the remaining Python/layer-loop denominator with a native whole-token program.")

    return {
        "probe": "t034_gpu_real_runtime_gates",
        "activation": ACTIVATION,
        "model_id": model_id,
        "model_dir": str(weight_dir) if weight_dir is not None else None,
        "config": config.as_dict(),
        "weight_inventory": inventory.as_dict(),
        "runtime": runtime_result,
        "t034_gates": {
            "structural_execution_view": structural_gate,
            "segment_transaction": transaction_gate,
            "device_argmax_embedding_splice": splice,
        },
        "t034_comparison": comparison,
        "readiness": {
            "t034_gpu_real_runtime_ready": t034_ready,
            "t034_runtime_smoke_pass": bool(readiness.get("smoke_pass")),
            "t034_structural_execution_view_ready": structural_ready,
            "t034_segment_transaction_ready": transaction_ready,
            "t034_device_argmax_embedding_splice_ready": splice_ready,
            "t034_gpu_bound_no_cpu_offload": bool(receipt.get("device") == "cuda" and receipt.get("device_payload_cache")),
            "t034_production_tok_s_claim_allowed": comparison["production_tok_s_claim_allowed"],
            "t034_measured_crosses_40_tok_s": comparison["measured_crosses_40_tok_s"],
        },
        "blockers": [*blockers, *((runtime_result or {}).get("blockers") or [])],
        "next_actions": next_actions,
    }


__all__ = ["ACTIVATION", "run_t034_gpu_real_runtime_gates"]
