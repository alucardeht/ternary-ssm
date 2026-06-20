"""Packed work coalescing gate for residual-destination scheduling."""

from __future__ import annotations

from typing import Any


ACTIVATION = "T020_PACKED_WORK_COALESCING_ACTIVE"


def run_t020_packed_work_coalescing_gate(
    *,
    layers: int = 64,
    hidden_size: int = 5120,
    tokens: int = 1,
    activation_bytes: int = 4,
    min_legal_outputs_per_block: int = 2,
) -> dict[str, Any]:
    """Build the one-block residual-destination map for B-PB-W02-11."""

    if layers <= 0:
        raise ValueError("layers must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if activation_bytes <= 0:
        raise ValueError("activation_bytes must be positive")

    boundary_map = [
        {
            "producer": "self_attn.q_proj",
            "immediate_consumers": ["q_norm", "rope", "attention_scores"],
            "hard_barrier": "attention_softmax",
            "legal_residual_destination": False,
        },
        {
            "producer": "self_attn.k_proj",
            "immediate_consumers": ["k_norm", "rope", "attention_scores"],
            "hard_barrier": "attention_softmax",
            "legal_residual_destination": False,
        },
        {
            "producer": "self_attn.v_proj",
            "immediate_consumers": ["attention_context"],
            "hard_barrier": "attention_softmax",
            "legal_residual_destination": False,
        },
        {
            "producer": "self_attn.o_proj",
            "immediate_consumers": ["residual_add"],
            "hard_barrier": "post_attention_rmsnorm",
            "legal_residual_destination": True,
        },
        {
            "producer": "mlp.gate_proj",
            "immediate_consumers": ["silu", "gated_product"],
            "hard_barrier": "mlp_nonlinearity",
            "legal_residual_destination": False,
        },
        {
            "producer": "mlp.up_proj",
            "immediate_consumers": ["gated_product"],
            "hard_barrier": "mlp_nonlinearity",
            "legal_residual_destination": False,
        },
        {
            "producer": "mlp.down_proj",
            "immediate_consumers": ["residual_add"],
            "hard_barrier": "next_block_rmsnorm_or_final_norm",
            "legal_residual_destination": True,
        },
    ]
    legal = [row for row in boundary_map if row["legal_residual_destination"]]
    legal_outputs_per_block = len(legal)
    total_legal_outputs = legal_outputs_per_block * layers
    hidden_vector_bytes = hidden_size * tokens * activation_bytes
    deleted_branch_write_bytes = total_legal_outputs * hidden_vector_bytes
    deleted_branch_reread_bytes = total_legal_outputs * hidden_vector_bytes
    deleted_dense_bytes = deleted_branch_write_bytes + deleted_branch_reread_bytes
    gate_pass = legal_outputs_per_block >= min_legal_outputs_per_block

    return {
        "probe": "t020_packed_work_coalescing_gate",
        "activation": ACTIVATION,
        "finding": "B-PB-W02-11",
        "one_block_boundary_map": boundary_map,
        "legal_boundaries": [row["producer"] for row in legal],
        "traffic": {
            "layers": layers,
            "tokens": tokens,
            "hidden_size": hidden_size,
            "activation_bytes": activation_bytes,
            "hidden_vector_bytes": hidden_vector_bytes,
            "deleted_branch_write_bytes": deleted_branch_write_bytes,
            "deleted_branch_reread_bytes": deleted_branch_reread_bytes,
            "deleted_dense_bytes": deleted_dense_bytes,
            "deleted_dense_mib": deleted_dense_bytes / (1024.0 * 1024.0),
            "fused_residual_adds": total_legal_outputs,
        },
        "readiness": {
            "packed_work_coalescing_gate_ready": gate_pass,
            "legal_outputs_per_block": legal_outputs_per_block,
            "minimum_legal_outputs_per_block": min_legal_outputs_per_block,
            "hard_barriers_respected": True,
            "external_branch_tensor_observers": False,
        },
        "blockers": [] if gate_pass else ["fewer_than_two_projection_outputs_can_land_in_residual_accumulators"],
        "next_actions": (
            [
                "Enable opt-in residual-destination packed kernels for self_attn.o_proj and mlp.down_proj.",
                "Measure T017 with packed work coalescing after payload-load scheduling is no longer the dominant denominator.",
            ]
            if gate_pass
            else ["Do not implement residual-destination coalescing until the one-block map exposes at least two legal outputs."]
        ),
    }


__all__ = ["ACTIVATION", "run_t020_packed_work_coalescing_gate"]
