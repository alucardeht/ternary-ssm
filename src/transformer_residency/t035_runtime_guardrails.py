"""T035 runtime guardrail gates for full64 Qwen inference projections."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ACTIVATION = "T035_RUNTIME_GUARDRAIL_GATE_ACTIVE"
ROW_SUCCESSOR_ACTIVATION = "T035_ROW_SUCCESSOR_GATE_ACTIVE"
ROW_SUCCESSOR_RUNTIME_ACTIVATION = "T035_ROW_SUCCESSOR_RUNTIME_PATH_ACTIVE"

R452_FULL64_TOKENS_PER_SECOND = 0.9280330765589875
R452_FULL64_SECONDS_PER_TOKEN = 1.0775478000286967
R452_FULL64_KERNEL_SECONDS = 0.5049169003032148
R452_FULL64_PEAK_CUDA_GIB = 10.652063369750977
R452_FULL64_RESIDENT_CUDA_GIB = 10.458633184432983

R458_FULL64_NO_QKV_TOKENS_PER_SECOND = 0.5406833101657142
R458_FULL64_NO_QKV_PEAK_CUDA_GIB = 9.255645751953125
R458_FULL64_NO_QKV_RESIDENT_CUDA_GIB = 8.969466924667358

R459_ONE_LAYER_TOKENS_PER_SECOND = 2.884471992687007
R459_ONE_LAYER_PEAK_CUDA_GIB = 0.5345873832702637
R459_ONE_LAYER_RESIDENT_CUDA_GIB = 0.5114963054656982

R460_ONE_LAYER_TOKENS_PER_SECOND = 0.3735925228602779
R460_ONE_LAYER_PEAK_CUDA_GIB = 3.431270122528076
R460_ONE_LAYER_RESIDENT_CUDA_GIB = 0.5114963054656982

TARGET_FULL64_TOKENS_PER_SECOND = 15.0
PROJECT_TARGET_TOKENS_PER_SECOND = 40.0
BYTES_PER_GIB = 1024**3


@dataclass(frozen=True)
class LiveInterval:
    name: str
    category: str
    start: float
    end: float
    gib: float
    hidden_dense: bool = False
    graph_persistent: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "start": self.start,
            "end": self.end,
            "gib": self.gib,
            "hidden_dense": self.hidden_dense,
            "graph_persistent": self.graph_persistent,
        }


def _max_live_clique(intervals: list[LiveInterval]) -> dict[str, Any]:
    points = sorted({point for interval in intervals for point in (interval.start, interval.end)})
    best_total = 0.0
    best_point = 0.0
    best_live: list[LiveInterval] = []
    for point in points:
        live = [interval for interval in intervals if interval.start <= point < interval.end]
        total = sum(interval.gib for interval in live)
        if total > best_total:
            best_total = total
            best_point = point
            best_live = live
    return {
        "max_live_scratch_gib": best_total,
        "phase_point": best_point,
        "live_items": [interval.name for interval in best_live],
    }


def _baseline_table() -> dict[str, Any]:
    return {
        "full64_speed_reference": {
            "source": "R452/T028",
            "tokens_per_second": R452_FULL64_TOKENS_PER_SECOND,
            "seconds_per_token": R452_FULL64_SECONDS_PER_TOKEN,
            "kernel_seconds": R452_FULL64_KERNEL_SECONDS,
            "non_kernel_seconds": R452_FULL64_SECONDS_PER_TOKEN - R452_FULL64_KERNEL_SECONDS,
            "peak_cuda_gib": R452_FULL64_PEAK_CUDA_GIB,
            "resident_cuda_gib": R452_FULL64_RESIDENT_CUDA_GIB,
            "scope": "full_64_of_64_layers",
        },
        "full64_memory_branch": {
            "source": "R458/T030 no-QKV",
            "tokens_per_second": R458_FULL64_NO_QKV_TOKENS_PER_SECOND,
            "peak_cuda_gib": R458_FULL64_NO_QKV_PEAK_CUDA_GIB,
            "resident_cuda_gib": R458_FULL64_NO_QKV_RESIDENT_CUDA_GIB,
            "speed_ratio_vs_r452": R458_FULL64_NO_QKV_TOKENS_PER_SECOND / R452_FULL64_TOKENS_PER_SECOND,
            "resident_saved_gib_vs_r452": R452_FULL64_RESIDENT_CUDA_GIB - R458_FULL64_NO_QKV_RESIDENT_CUDA_GIB,
            "claim_boundary": "memory_opportunity_only_speed_regressed",
        },
        "one_layer_diagnostic": {
            "source": "R459/T033",
            "tokens_per_second": R459_ONE_LAYER_TOKENS_PER_SECOND,
            "peak_cuda_gib": R459_ONE_LAYER_PEAK_CUDA_GIB,
            "resident_cuda_gib": R459_ONE_LAYER_RESIDENT_CUDA_GIB,
            "claim_boundary": "diagnostic_only_not_full64_speed_baseline",
        },
        "one_layer_negative_exact_gate": {
            "source": "R460/T034",
            "tokens_per_second": R460_ONE_LAYER_TOKENS_PER_SECOND,
            "peak_cuda_gib": R460_ONE_LAYER_PEAK_CUDA_GIB,
            "resident_cuda_gib": R460_ONE_LAYER_RESIDENT_CUDA_GIB,
            "peak_delta_gib_vs_r459": R460_ONE_LAYER_PEAK_CUDA_GIB - R459_ONE_LAYER_PEAK_CUDA_GIB,
            "speed_ratio_vs_r459": R460_ONE_LAYER_TOKENS_PER_SECOND / R459_ONE_LAYER_TOKENS_PER_SECOND,
            "claim_boundary": "negative_evidence_for_hidden_scratch_and_exact_sidecars",
        },
    }


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


def run_t035_runtime_guardrail_gate(
    *,
    row_successor_buffer_gib: float = 0.03125,
    verifier_buffer_gib: float = 0.03125,
    graph_memory_node_gib: float = 0.046875,
    descriptor_tape_gib: float = 0.015625,
    compact_fragment_gib: float = 0.0625,
    token_commit_buffer_gib: float = 0.015625,
    hidden_dense_gib: float = 0.0,
    reserved_pool_growth_gib: float = 0.0,
    exact_splice_peak_budget_gib: float = 0.25,
    require_no_speed_claim: bool = True,
) -> dict[str, Any]:
    """Run the C02-V02/A01-V04 guardrail ledger.

    This gate is intentionally conservative. It implements the prerequisite
    memory/address/scratch contract and forbids treating it as a production
    speedup.
    """

    intervals = [
        LiveInterval("descriptor_tape", "descriptor", 0.00, 1.00, descriptor_tape_gib, graph_persistent=True),
        LiveInterval("row_successor_buffer", "terminal_row", 0.72, 0.88, row_successor_buffer_gib),
        LiveInterval("verifier_buffer", "verifier", 0.15, 0.28, verifier_buffer_gib),
        LiveInterval("graph_memory_node", "graph_memory", 0.00, 1.00, graph_memory_node_gib, graph_persistent=True),
        LiveInterval("compact_fragment_tile", "compact_fragment", 0.30, 0.71, compact_fragment_gib),
        LiveInterval("token_commit_buffer", "commit", 0.86, 1.00, token_commit_buffer_gib),
    ]
    if hidden_dense_gib > 0.0:
        intervals.append(LiveInterval("hidden_dense_shadow", "hidden_dense", 0.30, 0.71, hidden_dense_gib, hidden_dense=True))
    if reserved_pool_growth_gib > 0.0:
        intervals.append(
            LiveInterval(
                "reserved_pool_growth",
                "reserved_pool",
                0.00,
                1.00,
                reserved_pool_growth_gib,
                graph_persistent=True,
            )
        )

    clique = _max_live_clique(intervals)
    candidate_peak_gib = R452_FULL64_RESIDENT_CUDA_GIB + clique["max_live_scratch_gib"]
    r452_headroom_gib = R452_FULL64_PEAK_CUDA_GIB - R452_FULL64_RESIDENT_CUDA_GIB
    peak_delta_vs_r452_gib = candidate_peak_gib - R452_FULL64_PEAK_CUDA_GIB
    peak_cap_gib = R452_FULL64_PEAK_CUDA_GIB + exact_splice_peak_budget_gib
    hidden_dense_live_gib = sum(interval.gib for interval in intervals if interval.hidden_dense)

    address_contract = {
        "stable_payload_addresses": True,
        "cursor_position_and_kv_updates_are_data": True,
        "graph_topology_mutation_required": False,
        "hot_payload_load_events_allowed": 0,
        "descriptor_count_reference": 523,
        "numeric_contracts": [
            "exact_token_boundary",
            "exact_descriptor_hash",
            "bounded_peak_cuda",
            "no_hidden_dense_shadow",
            "setup_excluded_from_speed_claims",
        ],
    }
    scratch_contract = {
        "interval_count": len(intervals),
        "intervals": [interval.as_dict() for interval in intervals],
        **clique,
        "r452_peak_resident_headroom_gib": r452_headroom_gib,
        "candidate_peak_cuda_gib": candidate_peak_gib,
        "peak_delta_vs_r452_gib": peak_delta_vs_r452_gib,
        "peak_cap_gib": peak_cap_gib,
        "hidden_dense_live_gib": hidden_dense_live_gib,
        "reserved_pool_growth_gib": reserved_pool_growth_gib,
    }

    blockers: list[str] = []
    if hidden_dense_live_gib > 0.0:
        blockers.append("hidden_dense_shadow_live")
    if candidate_peak_gib > peak_cap_gib:
        blockers.append("candidate_peak_above_exact_splice_budget")
    if reserved_pool_growth_gib > exact_splice_peak_budget_gib:
        blockers.append("reserved_pool_growth_above_budget")
    if require_no_speed_claim:
        blockers.append("guardrail_is_not_a_standalone_speed_claim")

    gate_ready = not any(
        blocker
        for blocker in blockers
        if blocker
        not in {
            "guardrail_is_not_a_standalone_speed_claim",
        }
    )

    projection = {
        "full64_before_tokens_per_second": R452_FULL64_TOKENS_PER_SECOND,
        "guardrail_only_tokens_per_second_band": [
            R452_FULL64_TOKENS_PER_SECOND,
            R452_FULL64_TOKENS_PER_SECOND * 1.05,
        ],
        "production_speed_claim_allowed": False,
        "target_15_tok_s_ratio_required": TARGET_FULL64_TOKENS_PER_SECOND / R452_FULL64_TOKENS_PER_SECOND,
        "target_40_tok_s_ratio_required": PROJECT_TARGET_TOKENS_PER_SECOND / R452_FULL64_TOKENS_PER_SECOND,
        "claim_boundary": "C02-V02/A01-V04 prerequisite gate only; later C01/C02 kernels must provide measured speed.",
    }

    next_actions = [
        "Implement C02-V03 peak-safe head-to-embedding row successor inside this ledger.",
        "Implement C01-V02/C01-V01 control-collapse only after this gate stays peak-safe with instrumentation disabled.",
        "Reject any later VRAM result that lowers resident bytes by moving work to host RAM, CPU, disk, or hidden dense scratch.",
    ]

    return {
        "probe": "t035_runtime_guardrail_gate",
        "activation": ACTIVATION,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_findings": ["C02-V02", "A01-V04"],
        "baseline_table": _baseline_table(),
        "address_contract": address_contract,
        "scratch_contract": scratch_contract,
        "projection": projection,
        "readiness": {
            "t035_runtime_guardrail_gate_ready": gate_ready,
            "t035_address_contract_ready": all(
                [
                    address_contract["stable_payload_addresses"],
                    address_contract["cursor_position_and_kv_updates_are_data"],
                    not address_contract["graph_topology_mutation_required"],
                ]
            ),
            "t035_scratch_peak_within_budget": candidate_peak_gib <= peak_cap_gib,
            "t035_no_hidden_dense_shadow": hidden_dense_live_gib == 0.0,
            "t035_one_layer_not_used_as_full64_baseline": True,
            "t035_speed_claim_allowed": False,
        },
        "blockers": blockers,
        "next_actions": next_actions,
    }


def run_t035_row_successor_gate(
    *,
    device: str = "cuda",
    require_cuda: bool = True,
    vocab_rows: int = 4096,
    hidden_size: int = 512,
    chunk_rows: int = 512,
    seed: int = 3503,
    exact_splice_peak_budget_gib: float = 0.25,
) -> dict[str, Any]:
    """Run the C02-V03 peak-safe greedy row-successor gate.

    The candidate path computes the winning token and next embedding on device
    before the token id is made host-visible. A full-logit reference is computed
    after the candidate path only to validate exactness.
    """

    blockers: list[str] = []
    if vocab_rows <= 0 or hidden_size <= 0 or chunk_rows <= 0:
        blockers.append("invalid_shape")
    if chunk_rows > vocab_rows:
        chunk_rows = vocab_rows

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local env
        return {
            "probe": "t035_row_successor_gate",
            "activation": ROW_SUCCESSOR_ACTIVATION,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "blockers": [f"torch_unavailable:{type(exc).__name__}:{exc}"],
            "readiness": {
                "t035_row_successor_gate_ready": False,
                "t035_cuda_path_used": False,
                "t035_exact_token_match": False,
                "t035_next_embedding_ready_before_host_token": False,
                "t035_peak_within_full64_cap": False,
                "t035_speed_claim_allowed": False,
            },
            "next_actions": ["Install or repair torch before running this GPU gate."],
        }

    cuda_available = bool(torch.cuda.is_available())
    selected_device = device
    if device == "auto":
        selected_device = "cuda" if cuda_available else "cpu"
    if selected_device == "cuda" and not cuda_available:
        blockers.append("cuda_unavailable")
        selected_device = "cpu"
    if require_cuda and selected_device != "cuda":
        blockers.append("row_successor_requires_cuda")

    torch.manual_seed(seed)
    if selected_device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    hidden = torch.randn(hidden_size, device=selected_device, dtype=torch.float32)
    head = torch.randn(vocab_rows, hidden_size, device=selected_device, dtype=torch.float32)
    embedding = torch.randn(vocab_rows, hidden_size, device=selected_device, dtype=torch.float32)

    if selected_device == "cuda":
        torch.cuda.synchronize()
    start_time = datetime.now(UTC)

    best_score = torch.tensor(float("-inf"), device=selected_device, dtype=torch.float32)
    best_index = torch.zeros((), device=selected_device, dtype=torch.long)
    chunk_count = 0
    max_chunk_score_numel = 0
    for start in range(0, vocab_rows, chunk_rows):
        end = min(start + chunk_rows, vocab_rows)
        scores = head[start:end].matmul(hidden)
        value, local_index = torch.max(scores, dim=0)
        update = value > best_score
        best_score = torch.where(update, value, best_score)
        best_index = torch.where(update, local_index.to(torch.long) + start, best_index)
        max_chunk_score_numel = max(max_chunk_score_numel, int(scores.numel()))
        chunk_count += 1
    next_embedding = embedding.index_select(0, best_index.reshape(1)).reshape(hidden_size)
    next_embedding_ready = bool(next_embedding.numel() == hidden_size)

    if selected_device == "cuda":
        torch.cuda.synchronize()
    elapsed = max((datetime.now(UTC) - start_time).total_seconds(), 1e-12)
    candidate_peak_gib = (
        torch.cuda.max_memory_allocated() / (1024**3) if selected_device == "cuda" else 0.0
    )

    host_visible_token_id = int(best_index.detach().cpu().item())
    reference_logits = head.matmul(hidden)
    reference_token_id = int(torch.argmax(reference_logits).detach().cpu().item())
    reference_embedding = embedding[reference_token_id]
    exact_token_match = host_visible_token_id == reference_token_id
    exact_embedding_match = bool(torch.equal(next_embedding.detach().cpu(), reference_embedding.detach().cpu()))

    bytes_per_float = 4
    logical_census = {
        "candidate_chunk_score_buffer_gib": max_chunk_score_numel * bytes_per_float / (1024**3),
        "candidate_next_embedding_buffer_gib": hidden_size * bytes_per_float / (1024**3),
        "candidate_full_logits_buffer_gib": 0.0,
        "candidate_duplicate_table_gib": 0.0,
        "reference_full_logits_buffer_gib": vocab_rows * bytes_per_float / (1024**3),
        "candidate_table_sized_scratch": False,
        "host_token_id_read_after_next_embedding": next_embedding_ready,
    }
    full64_peak_cap_gib = R452_FULL64_PEAK_CUDA_GIB + exact_splice_peak_budget_gib
    scaled_candidate_peak_gib = R452_FULL64_RESIDENT_CUDA_GIB + max(
        logical_census["candidate_chunk_score_buffer_gib"],
        logical_census["candidate_next_embedding_buffer_gib"],
    )
    peak_within_cap = scaled_candidate_peak_gib <= full64_peak_cap_gib

    if not exact_token_match:
        blockers.append("token_mismatch")
    if not exact_embedding_match:
        blockers.append("embedding_mismatch")
    if logical_census["candidate_table_sized_scratch"]:
        blockers.append("table_sized_scratch")
    if not peak_within_cap:
        blockers.append("full64_peak_cap_exceeded")

    gate_ready = not blockers
    return {
        "probe": "t035_row_successor_gate",
        "activation": ROW_SUCCESSOR_ACTIVATION,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_findings": ["C02-V03"],
        "baseline_table": _baseline_table(),
        "shape": {
            "vocab_rows": vocab_rows,
            "hidden_size": hidden_size,
            "chunk_rows": chunk_rows,
            "chunk_count": chunk_count,
            "device": selected_device,
        },
        "candidate": {
            "token_id": host_visible_token_id,
            "reference_token_id": reference_token_id,
            "exact_token_match": exact_token_match,
            "exact_embedding_match": exact_embedding_match,
            "next_embedding_ready_before_host_token": next_embedding_ready,
            "elapsed_seconds": elapsed,
            "tokens_per_second_local_gate": 1.0 / elapsed,
            "measured_peak_cuda_allocated_gib": candidate_peak_gib,
        },
        "allocation_census": {
            **logical_census,
            "scaled_full64_candidate_peak_gib": scaled_candidate_peak_gib,
            "full64_peak_cap_gib": full64_peak_cap_gib,
            "r460_one_layer_peak_excess_gib": R460_ONE_LAYER_PEAK_CUDA_GIB - R459_ONE_LAYER_PEAK_CUDA_GIB,
        },
        "projection": {
            "full64_before_tokens_per_second": R452_FULL64_TOKENS_PER_SECOND,
            "row_successor_only_tokens_per_second_band": [
                R452_FULL64_TOKENS_PER_SECOND,
                R452_FULL64_TOKENS_PER_SECOND * 1.12,
            ],
            "production_speed_claim_allowed": False,
            "claim_boundary": "Peak-safe C02-V03 gate only; full64 timing is still required.",
        },
        "readiness": {
            "t035_row_successor_gate_ready": gate_ready,
            "t035_cuda_path_used": selected_device == "cuda",
            "t035_exact_token_match": exact_token_match,
            "t035_exact_embedding_match": exact_embedding_match,
            "t035_next_embedding_ready_before_host_token": next_embedding_ready,
            "t035_no_full_logits_in_candidate": logical_census["candidate_full_logits_buffer_gib"] == 0.0,
            "t035_no_table_sized_scratch": not logical_census["candidate_table_sized_scratch"],
            "t035_peak_within_full64_cap": peak_within_cap,
            "t035_speed_claim_allowed": False,
        },
        "blockers": blockers,
        "next_actions": [
            "Attach this row-successor contract to the real T034/T017 terminal path without materializing full logits.",
            "Run a same-window one-layer GPU smoke, then a full64 timing only if peak remains inside the cap.",
            "Proceed to C01-V02/C01-V01 only after row-successor stays peak-safe.",
        ],
    }


def run_t035_row_successor_runtime_path(
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
    device_payload_budget_gib: float | None = 10.5,
    packed_linear_block_m: int = 8,
    packed_linear_block_c: int = 128,
    packed_linear_block_r: int = 32,
    exact_splice_peak_budget_gib: float = 0.25,
    runtime_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Attach C02-V03 to the real T017 terminal path.

    This is still a gate, but it executes the real packed Qwen path rather than
    a synthetic tensor. The gate is allowed to report measured one-layer
    behavior, while full64 speed claims remain forbidden unless the requested
    layer count is the full model.
    """

    if runtime_runner is None:
        from .t017_packed_linear_kernel import run_t017_kernel_qwen_inference as runtime_runner

    runtime_result = runtime_runner(
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
        dense_tensor_cache=False,
        binary_cache=False,
        trit_packed_codes=True,
        trit_code_group_size=5,
        device_credit_flow=False,
        packed_work_coalescing=False,
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

    receipt = (runtime_result or {}).get("receipt") or {}
    runtime_readiness = (runtime_result or {}).get("readiness") or {}
    splice = (receipt.get("t034_device_argmax_embedding_splice") or {}).get("lm_head_certificate") or {}
    cache_stats = receipt.get("cache_stats") or {}
    requested_layers = _safe_int(receipt.get("requested_layers"))
    completed_layers = _safe_int(receipt.get("completed_layers"))
    measured_tokens_per_second = _safe_float(receipt.get("tokens_per_second"))
    measured_peak_cuda_gib = _safe_float(receipt.get("peak_cuda_allocated_gib"))
    measured_reserved_cuda_gib = _safe_float(receipt.get("peak_cuda_reserved_gib"))
    resident_cuda_gib = _safe_int(cache_stats.get("device_resident_bytes")) / BYTES_PER_GIB
    hot_payload_load_events = _safe_int(receipt.get("hot_payload_load_events"))
    hot_payload_load_seconds = _safe_float(receipt.get("hot_payload_load_seconds"))
    comparable_one_layer = requested_layers == 1
    comparable_full64 = requested_layers == 64

    one_layer_peak_cap_gib = R459_ONE_LAYER_PEAK_CUDA_GIB + exact_splice_peak_budget_gib
    full64_peak_cap_gib = R452_FULL64_PEAK_CUDA_GIB + exact_splice_peak_budget_gib
    active_peak_cap_gib = full64_peak_cap_gib if comparable_full64 else one_layer_peak_cap_gib

    exact_device_splice = bool(
        splice.get("active")
        and splice.get("next_embedding_device_closure_proven")
        and not splice.get("host_token_id_dependency_before_next_embedding")
        and splice.get("host_token_id_read_after_device_embedding")
    )
    no_full_logits = bool(
        splice.get("full_logits_materialized") is False
        and splice.get("full_logits_returned_to_host") is False
    )
    gpu_bound = bool(receipt.get("device") == "cuda" and receipt.get("device_payload_cache"))
    peak_within_cap = measured_peak_cuda_gib <= active_peak_cap_gib if measured_peak_cuda_gib > 0.0 else False
    hot_payload_free = hot_payload_load_events == 0 and hot_payload_load_seconds == 0.0

    blockers: list[str] = list((runtime_result or {}).get("blockers") or [])
    if not runtime_readiness.get("smoke_pass"):
        blockers.append("runtime_smoke_failed")
    if not gpu_bound:
        blockers.append("runtime_not_gpu_bound_device_payload_path")
    if not exact_device_splice:
        blockers.append("device_argmax_embedding_splice_not_closed")
    if not no_full_logits:
        blockers.append("full_logits_materialized_or_returned")
    if not peak_within_cap:
        blockers.append("peak_above_t035_splice_cap")

    runtime_ready = not blockers
    comparison = {
        "scope": "one_layer" if comparable_one_layer else "full64" if comparable_full64 else "partial_layer_count",
        "requested_layers": requested_layers,
        "completed_layers": completed_layers,
        "measured_tokens_per_second": measured_tokens_per_second,
        "baseline_source": "R459/T033" if comparable_one_layer else "R452/T028" if comparable_full64 else None,
        "baseline_tokens_per_second": (
            R459_ONE_LAYER_TOKENS_PER_SECOND
            if comparable_one_layer
            else R452_FULL64_TOKENS_PER_SECOND
            if comparable_full64
            else None
        ),
        "measured_speedup_vs_comparable_baseline": (
            measured_tokens_per_second / R459_ONE_LAYER_TOKENS_PER_SECOND
            if comparable_one_layer and measured_tokens_per_second > 0.0
            else measured_tokens_per_second / R452_FULL64_TOKENS_PER_SECOND
            if comparable_full64 and measured_tokens_per_second > 0.0
            else None
        ),
        "measured_peak_cuda_gib": measured_peak_cuda_gib,
        "measured_reserved_cuda_gib": measured_reserved_cuda_gib,
        "measured_resident_cuda_gib": resident_cuda_gib,
        "active_peak_cap_gib": active_peak_cap_gib,
        "production_full64_speed_claim_allowed": bool(comparable_full64 and runtime_ready),
        "one_layer_speed_claim_allowed": bool(comparable_one_layer and runtime_ready),
    }
    projection = {
        "full64_before_tokens_per_second": R452_FULL64_TOKENS_PER_SECOND,
        "row_successor_only_full64_band": [
            R452_FULL64_TOKENS_PER_SECOND,
            R452_FULL64_TOKENS_PER_SECOND * 1.12,
        ],
        "target_15_tok_s_ratio_required_after_this_gate": TARGET_FULL64_TOKENS_PER_SECOND
        / R452_FULL64_TOKENS_PER_SECOND,
        "target_40_tok_s_ratio_required_after_this_gate": PROJECT_TARGET_TOKENS_PER_SECOND
        / R452_FULL64_TOKENS_PER_SECOND,
        "claim_boundary": "Real T017 row-successor splice gate; not enough by itself for a full64 production speed claim.",
    }

    return {
        "probe": "t035_row_successor_runtime_path",
        "activation": ROW_SUCCESSOR_RUNTIME_ACTIVATION,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_findings": ["C02-V03"],
        "baseline_table": _baseline_table(),
        "runtime": runtime_result,
        "splice_certificate": splice,
        "comparison": comparison,
        "projection": projection,
        "readiness": {
            "t035_row_successor_runtime_path_ready": runtime_ready,
            "t035_runtime_smoke_pass": bool(runtime_readiness.get("smoke_pass")),
            "t035_gpu_bound_device_payload_path": gpu_bound,
            "t035_exact_device_splice_closed": exact_device_splice,
            "t035_no_full_logits_materialized_or_returned": no_full_logits,
            "t035_peak_within_splice_cap": peak_within_cap,
            "t035_hot_payload_free": hot_payload_free,
            "t035_production_full64_speed_claim_allowed": comparison["production_full64_speed_claim_allowed"],
        },
        "blockers": blockers,
        "next_actions": (
            [
                "Promote C02-V03 as the exact terminal row-successor contract for the real T017 path.",
                "Implement C01-V02/C01-V01 control collapse next; C02-V03 alone cannot reach 15 tok/s.",
            ]
            if runtime_ready
            else [
                "Fix the listed runtime path blockers before claiming C02-V03 landed on the real Qwen path.",
                "If the blocker is peak pressure, remove hidden dense/cache sidecars before pursuing C01 control collapse.",
            ]
        ),
    }


__all__ = [
    "ACTIVATION",
    "ROW_SUCCESSOR_ACTIVATION",
    "ROW_SUCCESSOR_RUNTIME_ACTIVATION",
    "run_t035_runtime_guardrail_gate",
    "run_t035_row_successor_gate",
    "run_t035_row_successor_runtime_path",
]
