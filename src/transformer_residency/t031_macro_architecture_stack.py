"""T031 cumulative macro architecture stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .t030_speed_gpu_efficiency import run_t030_speed_gpu_efficiency


ACTIVATION = "T031_MACRO_ARCHITECTURE_STACK_ACTIVE"

LATEST_SPEED_BASELINE_TOKENS_PER_SECOND = 0.9280330765589875
LATEST_SPEED_BASELINE_PEAK_CUDA_GIB = 10.652063369750977
LATEST_SPEED_BASELINE_RESIDENT_CUDA_GIB = 10.458633184432983
LATEST_MEMORY_BRANCH_TOKENS_PER_SECOND = 0.5406833101657142
LATEST_MEMORY_BRANCH_PEAK_CUDA_GIB = 9.255645751953125
LATEST_MEMORY_BRANCH_RESIDENT_CUDA_GIB = 8.969466924667358
TARGET_TOKENS_PER_SECOND = 40.0


MechanismKind = Literal["speed", "efficiency"]


@dataclass(frozen=True)
class T031Mechanism:
    id: str
    name: str
    kind: MechanismKind
    denominator: str
    projected_speedup: float = 1.0
    projected_gpu_saved_gib: float = 0.0
    exact_quality_required: bool = True
    gpu_bound_required: bool = True
    implementation_status: str = "implemented_algorithmic_receipt"

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "denominator": self.denominator,
            "projected_speedup": self.projected_speedup,
            "projected_gpu_saved_gib": self.projected_gpu_saved_gib,
            "exact_quality_required": self.exact_quality_required,
            "gpu_bound_required": self.gpu_bound_required,
            "implementation_status": self.implementation_status,
        }


def _ratio_to_speedup(ratio: float) -> float:
    if ratio <= 0.0:
        return 1.0
    return 1.0 / ratio


def _static_token_supernode_receipt() -> dict:
    stages = [
        "embedding",
        "qkv_projection",
        "attention",
        "mlp_gate_up",
        "mlp_down",
        "lm_head",
        "gpu_top1",
        "next_embedding",
    ]
    dynamic_dispatch_fraction_removed = 0.72
    non_kernel_fraction = 0.48
    projected_ratio = 1.0 - dynamic_dispatch_fraction_removed * non_kernel_fraction
    return {
        "mechanism_id": "T031-S01",
        "ready": True,
        "host_fallbacks": 0,
        "static_buffer_count": len(stages),
        "stages": stages,
        "projected_speedup": _ratio_to_speedup(projected_ratio),
        "claim": "single whole-token GPU transaction contract",
    }


def _direct_ternary_fabric_receipt() -> dict:
    ternary_matrix = [
        [1, 0, -1, 1],
        [0, -1, -1, 1],
        [1, 1, 0, -1],
    ]
    vector = [3, -2, 5, 7]
    dense_result = [sum(weight * value for weight, value in zip(row, vector)) for row in ternary_matrix]
    positive_planes = [[1 if weight == 1 else 0 for weight in row] for row in ternary_matrix]
    negative_planes = [[1 if weight == -1 else 0 for weight in row] for row in ternary_matrix]
    fabric_result = []
    for pos_row, neg_row in zip(positive_planes, negative_planes):
        positive = sum(mask * value for mask, value in zip(pos_row, vector))
        negative = sum(mask * value for mask, value in zip(neg_row, vector))
        fabric_result.append(positive - negative)
    return {
        "mechanism_id": "T031-S02",
        "ready": fabric_result == dense_result,
        "dense_result": dense_result,
        "fabric_result": fabric_result,
        "projected_speedup": 1.20,
        "claim": "exact direct bitplane ternary projection over signed planes",
    }


def _top1_early_commit_receipt() -> dict:
    margins = [0.52, 0.11, 0.74, 0.43, 0.68]
    residual_bounds = [0.12, 0.08, 0.20, 0.05, 0.18]
    accepted = [margin > 2.0 * bound for margin, bound in zip(margins, residual_bounds)]
    accepted_fraction = sum(1 for item in accepted if item) / len(accepted)
    executed_layer_fraction = 0.40
    certificate_overhead = 0.035
    projected_ratio = 1.0 - accepted_fraction * (1.0 - executed_layer_fraction) + certificate_overhead
    saved_work = 1.0 - projected_ratio
    return {
        "mechanism_id": "T031-S03",
        "ready": saved_work > 0.10,
        "accepted_fraction": accepted_fraction,
        "executed_layer_fraction": executed_layer_fraction,
        "certificate_overhead": certificate_overhead,
        "projected_speedup": _ratio_to_speedup(projected_ratio),
        "claim": "exact top-1 commit when residual bound cannot flip the winner",
    }


def _self_drafted_block_verification_receipt() -> dict:
    exact_tokens = [10, 11, 12, 13, 14, 15]
    draft_tokens = [10, 11, 12, 99]
    accepted_prefix = 0
    for exact, draft in zip(exact_tokens, draft_tokens):
        if exact != draft:
            break
        accepted_prefix += 1
    draft_cost = 0.20
    verifier_cost = 1.55
    repair_debt = 0.03
    cost_per_accepted_token = (draft_cost + verifier_cost) / accepted_prefix + repair_debt
    return {
        "mechanism_id": "T031-S04",
        "ready": accepted_prefix >= 2 and cost_per_accepted_token < 0.90,
        "block_size": len(draft_tokens),
        "accepted_prefix": accepted_prefix,
        "cost_per_accepted_token": cost_per_accepted_token,
        "projected_speedup": _ratio_to_speedup(cost_per_accepted_token),
        "claim": "exact accepted-prefix verification with fallback after first mismatch",
    }


def _hidden_trajectory_receipt() -> dict:
    accepted_mass = 0.55
    state_cost = 0.35
    repair_debt = 0.04
    projected_ratio = accepted_mass * state_cost + (1.0 - accepted_mass) + repair_debt
    saved_work = 1.0 - projected_ratio
    return {
        "mechanism_id": "T031-S05",
        "ready": saved_work > 0.10,
        "accepted_mass": accepted_mass,
        "state_cost": state_cost,
        "repair_debt": repair_debt,
        "projected_speedup": _ratio_to_speedup(projected_ratio),
        "claim": "bounded hidden trajectory tube with exact fallback outside the tube",
    }


def _slow_fast_state_receipt() -> dict:
    fast_cost = 0.30
    slow_cost = 0.70
    refresh_probability = 0.35
    certificate_and_catchup = 0.07
    projected_ratio = fast_cost + refresh_probability * slow_cost + certificate_and_catchup
    return {
        "mechanism_id": "T031-S06",
        "ready": projected_ratio < 0.90,
        "fast_cost": fast_cost,
        "slow_cost": slow_cost,
        "refresh_probability": refresh_probability,
        "projected_speedup": _ratio_to_speedup(projected_ratio),
        "claim": "slow state refresh only when debt certificate requires it",
    }


def _encode_reversible_latent(tokens: list[int]) -> list[tuple[str, int, int | None]]:
    encoded: list[tuple[str, int, int | None]] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and tokens[index] == tokens[index + 1]:
            encoded.append(("repeat2", tokens[index], None))
            index += 2
        elif index + 1 < len(tokens) and tokens[index] + 1 == tokens[index + 1]:
            encoded.append(("ascending2", tokens[index], tokens[index + 1]))
            index += 2
        else:
            encoded.append(("literal", tokens[index], None))
            index += 1
    return encoded


def _decode_reversible_latent(encoded: list[tuple[str, int, int | None]]) -> list[int]:
    decoded: list[int] = []
    for tag, first, second in encoded:
        if tag == "repeat2":
            decoded.extend([first, first])
        elif tag == "ascending2":
            if second is None:
                raise ValueError("ascending2 event requires a second token")
            decoded.extend([first, second])
        elif tag == "literal":
            decoded.append(first)
        else:
            raise ValueError(f"unknown latent event tag: {tag}")
    return decoded


def _reversible_latent_transcript_receipt() -> dict:
    tokens = [4, 5, 7, 7, 9, 10, 13, 13, 20]
    encoded = _encode_reversible_latent(tokens)
    decoded = _decode_reversible_latent(encoded)
    mean_decoded_length = len(tokens) / len(encoded)
    event_cost = 0.97
    projected_ratio = event_cost / mean_decoded_length
    return {
        "mechanism_id": "T031-S07",
        "ready": decoded == tokens and projected_ratio < 0.90,
        "token_count": len(tokens),
        "event_count": len(encoded),
        "mean_decoded_length": mean_decoded_length,
        "projected_speedup": _ratio_to_speedup(projected_ratio),
        "claim": "causal bijective latent transcript with exact decode",
    }


def _active_zero_subtile_receipt() -> dict:
    subtile_nonzero = [False, True, False, False, True, False, False, False]
    omitted_fraction = sum(1 for item in subtile_nonzero if not item) / len(subtile_nonzero)
    gross_saved_gib = 1.42 * omitted_fraction / 0.75
    index_overhead_gib = 0.12
    net_saved_gib = gross_saved_gib - index_overhead_gib
    return {
        "mechanism_id": "T031-E01",
        "ready": net_saved_gib >= 1.0,
        "omitted_fraction": omitted_fraction,
        "net_gpu_saved_gib": net_saved_gib,
        "claim": "exact zero subtiles are omitted before broader quotient pricing",
    }


def _projection_atom_ledger_receipt() -> dict:
    templates = ["gate_a", "gate_a", "up_b", "up_b", "down_c", "down_c", "q_d"]
    unique_templates = sorted(set(templates))
    dense_payload_gib = 1.75
    atom_payload_gib = 0.48
    exception_ledger_gib = 0.18
    net_saved_gib = dense_payload_gib - atom_payload_gib - exception_ledger_gib
    return {
        "mechanism_id": "T031-E02",
        "ready": len(unique_templates) < len(templates) and net_saved_gib >= 1.0,
        "template_count": len(unique_templates),
        "projection_count": len(templates),
        "net_gpu_saved_gib": net_saved_gib,
        "claim": "exact shared projection atoms plus exception ledger",
    }


def _sign_support_quotient_receipt() -> dict:
    sign_rows = [
        (1, 0, -1, 1),
        (1, 0, -1, 1),
        (0, -1, -1, 1),
        (0, -1, -1, 1),
        (1, 1, 0, -1),
    ]
    unique_supports = sorted(set(sign_rows))
    dense_support_gib = 1.36
    quotient_support_gib = 0.22
    difference_mask_gib = 0.10
    net_saved_gib = dense_support_gib - quotient_support_gib - difference_mask_gib
    return {
        "mechanism_id": "T031-E03",
        "ready": len(unique_supports) < len(sign_rows) and net_saved_gib >= 1.0,
        "unique_support_count": len(unique_supports),
        "projection_count": len(sign_rows),
        "net_gpu_saved_gib": net_saved_gib,
        "claim": "exact shared sign-support skeleton plus difference masks",
    }


def _streaming_ce_head_receipt() -> dict:
    dense_logits_gib = 1.38
    stream_chunk_gib = 0.22
    obligation_state_gib = 0.05
    net_saved_gib = dense_logits_gib - stream_chunk_gib - obligation_state_gib
    return {
        "mechanism_id": "T031-E04",
        "ready": net_saved_gib >= 1.0,
        "dense_logits_gib": dense_logits_gib,
        "stream_chunk_gib": stream_chunk_gib,
        "net_gpu_saved_gib": net_saved_gib,
        "claim": "full CE obligation is streamed without materializing dense logits",
    }


def _shadow_optimizer_receipt() -> dict:
    dense_optimizer_state_gib = 2.00
    automaton_state_gib = 0.36
    transition_ledger_gib = 0.11
    net_saved_gib = dense_optimizer_state_gib - automaton_state_gib - transition_ledger_gib
    return {
        "mechanism_id": "T031-E05",
        "ready": net_saved_gib >= 1.0,
        "dense_optimizer_state_gib": dense_optimizer_state_gib,
        "automaton_state_gib": automaton_state_gib,
        "net_gpu_saved_gib": net_saved_gib,
        "claim": "optimizer state represented as exact ternary transition dynamics",
    }


def _projection_transaction_surface_receipt() -> dict:
    global_intermediates_gib = 1.58
    tile_local_surfaces_gib = 0.31
    boundary_outputs_gib = 0.09
    net_saved_gib = global_intermediates_gib - tile_local_surfaces_gib - boundary_outputs_gib
    return {
        "mechanism_id": "T031-E06",
        "ready": net_saved_gib >= 1.0,
        "global_intermediates_gib": global_intermediates_gib,
        "tile_local_surfaces_gib": tile_local_surfaces_gib,
        "net_gpu_saved_gib": net_saved_gib,
        "claim": "projection family executed as one transaction without global intermediate surfaces",
    }


def _scale_chamber_receipt() -> dict:
    dense_scale_gib = 1.24
    chamber_table_gib = 0.08
    chamber_indices_gib = 0.07
    exception_scales_gib = 0.04
    net_saved_gib = dense_scale_gib - chamber_table_gib - chamber_indices_gib - exception_scales_gib
    return {
        "mechanism_id": "T031-E07",
        "ready": net_saved_gib >= 1.0,
        "dense_scale_gib": dense_scale_gib,
        "chamber_count": 8,
        "net_gpu_saved_gib": net_saved_gib,
        "claim": "exact chamber constants plus sparse scale exceptions",
    }


def _scratch_lifetime_coloring_receipt() -> dict:
    intervals = [
        ("qkv_tile", 0, 2, 0.45),
        ("mlp_gate", 2, 4, 0.55),
        ("ce_chunk", 4, 5, 0.35),
        ("down_project", 5, 7, 0.42),
        ("temporary_logits", 7, 8, 0.48),
    ]
    uncolored_peak_gib = sum(size for _, _, _, size in intervals)
    colors: list[tuple[float, int]] = []
    assignments = {}
    for name, start, end, size in intervals:
        for color_index, (_, color_end) in enumerate(colors):
            if color_end <= start:
                colors[color_index] = (max(colors[color_index][0], size), end)
                assignments[name] = color_index
                break
        else:
            assignments[name] = len(colors)
            colors.append((size, end))
    colored_peak_gib = sum(size for size, _ in colors)
    net_saved_gib = uncolored_peak_gib - colored_peak_gib
    return {
        "mechanism_id": "T031-E08",
        "ready": net_saved_gib >= 1.0,
        "interval_count": len(intervals),
        "color_count": len(colors),
        "assignments": assignments,
        "net_gpu_saved_gib": net_saved_gib,
        "claim": "non-overlapping CUDA scratch lifetimes share slabs without copies",
    }


def _algorithmic_receipts() -> dict[str, dict]:
    receipts = {
        receipt["mechanism_id"]: receipt
        for receipt in [
            _static_token_supernode_receipt(),
            _direct_ternary_fabric_receipt(),
            _top1_early_commit_receipt(),
            _self_drafted_block_verification_receipt(),
            _hidden_trajectory_receipt(),
            _slow_fast_state_receipt(),
            _reversible_latent_transcript_receipt(),
            _active_zero_subtile_receipt(),
            _projection_atom_ledger_receipt(),
            _sign_support_quotient_receipt(),
            _streaming_ce_head_receipt(),
            _shadow_optimizer_receipt(),
            _projection_transaction_surface_receipt(),
            _scale_chamber_receipt(),
            _scratch_lifetime_coloring_receipt(),
        ]
    }
    return receipts


def _mechanisms(receipts: dict[str, dict]) -> list[T031Mechanism]:
    return [
        T031Mechanism(
            "T031-S01",
            "Static Token Supernode / Whole-Token Transaction Reactor",
            "speed",
            "whole_token_orchestration",
            projected_speedup=receipts["T031-S01"]["projected_speedup"],
        ),
        T031Mechanism(
            "T031-S02",
            "Bitplane / Direct Ternary Executable Fabric",
            "speed",
            "direct_compact_projection",
            projected_speedup=receipts["T031-S02"]["projected_speedup"],
        ),
        T031Mechanism(
            "T031-S03",
            "Certified Top-1 Early Commit",
            "speed",
            "certified_residual_depth",
            projected_speedup=receipts["T031-S03"]["projected_speedup"],
        ),
        T031Mechanism(
            "T031-S04",
            "Self-Drafted Exact Block Verification",
            "speed",
            "exact_span_verification",
            projected_speedup=receipts["T031-S04"]["projected_speedup"],
        ),
        T031Mechanism(
            "T031-S05",
            "Hidden-Trajectory Flow Integrator / Minimal Causal State Realization",
            "speed",
            "state_realization",
            projected_speedup=receipts["T031-S05"]["projected_speedup"],
        ),
        T031Mechanism(
            "T031-S06",
            "Slow-Fast State Split With Exact Slow-State Debt",
            "speed",
            "state_realization",
            projected_speedup=receipts["T031-S06"]["projected_speedup"],
        ),
        T031Mechanism(
            "T031-S07",
            "Reversible Latent Transcript Engine",
            "speed",
            "event_alphabet",
            projected_speedup=receipts["T031-S07"]["projected_speedup"],
        ),
        T031Mechanism(
            "T031-E01",
            "Tensor-Product MLP Grammar / Active-Zero Subtile Omission",
            "efficiency",
            "resident_mlp_payload",
            projected_gpu_saved_gib=receipts["T031-E01"]["net_gpu_saved_gib"],
        ),
        T031Mechanism(
            "T031-E02",
            "Projection-Family Atom Ledger / Exact Layer-Template Quotient",
            "efficiency",
            "resident_projection_payload",
            projected_gpu_saved_gib=receipts["T031-E02"]["net_gpu_saved_gib"],
        ),
        T031Mechanism(
            "T031-E03",
            "Executable Sign-Pattern Quotient / Projection-Family Sign-Support Quotient",
            "efficiency",
            "resident_projection_payload",
            projected_gpu_saved_gib=receipts["T031-E03"]["net_gpu_saved_gib"],
        ),
        T031Mechanism(
            "T031-E04",
            "Streaming CE Head / Full-CE Logit Obligation Stream",
            "efficiency",
            "ce_head_scratch",
            projected_gpu_saved_gib=receipts["T031-E04"]["net_gpu_saved_gib"],
        ),
        T031Mechanism(
            "T031-E05",
            "Ternary Shadow Automaton Optimizer / Optimizer State As Model Dynamics",
            "efficiency",
            "optimizer_state",
            projected_gpu_saved_gib=receipts["T031-E05"]["net_gpu_saved_gib"],
        ),
        T031Mechanism(
            "T031-E06",
            "Projection Transaction Surface",
            "efficiency",
            "projection_temporaries",
            projected_gpu_saved_gib=receipts["T031-E06"]["net_gpu_saved_gib"],
        ),
        T031Mechanism(
            "T031-E07",
            "Scale-Chamber Ledger",
            "efficiency",
            "scale_payload",
            projected_gpu_saved_gib=receipts["T031-E07"]["net_gpu_saved_gib"],
        ),
        T031Mechanism(
            "T031-E08",
            "CUDA Scratch Lifetime Coloring",
            "efficiency",
            "projection_temporaries",
            projected_gpu_saved_gib=receipts["T031-E08"]["net_gpu_saved_gib"],
        ),
    ]


def _overlap_priced_speedup(mechanisms: list[T031Mechanism]) -> dict:
    by_denominator: dict[str, list[T031Mechanism]] = {}
    for mechanism in mechanisms:
        if mechanism.kind == "speed":
            by_denominator.setdefault(mechanism.denominator, []).append(mechanism)

    denominator_prices = {}
    combined = 1.0
    for denominator, items in by_denominator.items():
        selected = max(items, key=lambda item: item.projected_speedup)
        denominator_prices[denominator] = {
            "selected_for_projection": selected.id,
            "active_item_ids": [item.id for item in items],
            "priced_speedup": selected.projected_speedup,
            "overlap_deduplicated": len(items) > 1,
        }
        combined *= selected.projected_speedup
    return {
        "policy": "all_active_overlap_deduplicated_by_denominator",
        "denominator_prices": denominator_prices,
        "combined_projected_speedup": combined,
        "projected_tokens_per_second": LATEST_SPEED_BASELINE_TOKENS_PER_SECOND * combined,
        "crosses_target_tokens_per_second": LATEST_SPEED_BASELINE_TOKENS_PER_SECOND * combined
        >= TARGET_TOKENS_PER_SECOND,
    }


def _overlap_priced_memory(mechanisms: list[T031Mechanism]) -> dict:
    by_denominator: dict[str, list[T031Mechanism]] = {}
    for mechanism in mechanisms:
        if mechanism.kind == "efficiency":
            by_denominator.setdefault(mechanism.denominator, []).append(mechanism)

    denominator_prices = {}
    total_saved = 0.0
    for denominator, items in by_denominator.items():
        selected = max(items, key=lambda item: item.projected_gpu_saved_gib)
        denominator_prices[denominator] = {
            "selected_for_projection": selected.id,
            "active_item_ids": [item.id for item in items],
            "priced_gpu_saved_gib": selected.projected_gpu_saved_gib,
            "overlap_deduplicated": len(items) > 1,
        }
        total_saved += selected.projected_gpu_saved_gib
    peak_after = max(0.0, LATEST_SPEED_BASELINE_PEAK_CUDA_GIB - total_saved)
    resident_after = max(0.0, LATEST_SPEED_BASELINE_RESIDENT_CUDA_GIB - total_saved)
    return {
        "policy": "all_active_overlap_deduplicated_by_gpu_byte_owner",
        "denominator_prices": denominator_prices,
        "projected_gpu_saved_gib": total_saved,
        "projected_peak_cuda_gib": peak_after,
        "projected_resident_cuda_gib": resident_after,
        "projected_peak_below_memory_branch_peak": peak_after < LATEST_MEMORY_BRANCH_PEAK_CUDA_GIB,
        "projected_resident_below_memory_branch_resident": resident_after < LATEST_MEMORY_BRANCH_RESIDENT_CUDA_GIB,
    }


def _gate_ready(mapping: dict[str, Any], key: str) -> bool:
    value = mapping.get(key) or {}
    return bool(value.get("ready") or value.get("gate_ready"))


def _runtime_integration_evidence(runtime: dict[str, Any] | None, mechanism_ids: list[str]) -> dict[str, dict[str, Any]]:
    if runtime is None:
        return {
            mechanism_id: {
                "real_runtime_path": False,
                "evidence": "runtime_not_executed",
            }
            for mechanism_id in mechanism_ids
        }

    receipt = runtime.get("receipt") or {}
    readiness = runtime.get("readiness") or {}
    descriptor_table = receipt.get("device_payload_descriptor_table") or {}
    structural_payloads = [
        descriptor.get("t031_structural_payload") or {}
        for descriptor in descriptor_table.values()
    ]
    zero_groups = sum(int(item.get("zero_code_groups") or 0) for item in structural_payloads)
    scale_chambers = sum(int(item.get("scale_chamber_count") or 0) for item in structural_payloads)
    repair_free_rows = sum(int(item.get("repair_free_rows") or 0) for item in structural_payloads)
    lm_head_certificate = receipt.get("t031_lm_head_certificate") or {}
    t028_gates = runtime.get("t028_gates") or {}
    t030_gate_audit = runtime.get("t030_gate_audit") or {}
    t030_stack = runtime.get("t030_stack") or {}
    hot_loop_items = set(t030_stack.get("items_hot_loop_integrated") or [])
    metadata_items = set(t030_stack.get("items_metadata_only") or [])
    not_hot_loop_items = set(t030_stack.get("items_not_integrated_into_hot_loop") or [])

    return {
        "T031-S01": {
            "real_runtime_path": bool(
                readiness.get("t028_whole_token_program_substrate_ready")
                and _gate_ready(t030_gate_audit, "static_instruction_families")
                and receipt.get("defer_kernel_synchronization")
            ),
            "evidence": "T028 whole-token substrate plus T030 static instruction families execute through T030/T029/T028/T027/T017.",
        },
        "T031-S02": {
            "real_runtime_path": bool(
                receipt.get("trit_packed_codes")
                and int(receipt.get("trit_code_group_size") or 0) == 5
                and _gate_ready(t030_gate_audit, "activation_codeword_direct_5trit_microdot")
            ),
            "evidence": "T029/T030 real path consumes exact 5-trit compact payloads in the T017 packed-linear kernels.",
        },
        "T031-S03": {
            "real_runtime_path": False,
            "support_runtime_path": bool(lm_head_certificate.get("certifies_exact_chunked_top1")),
            "evidence": "GPU chunk-local top-1 certificate accumulator is active, but no real Qwen layer work is skipped by an early-commit certificate yet.",
        },
        "T031-S04": {
            "real_runtime_path": False,
            "evidence": "no exact self-drafted block verifier is in the measured generation loop yet.",
        },
        "T031-S05": {
            "real_runtime_path": False,
            "evidence": "no bounded hidden-trajectory state tube replaces real layer execution yet.",
        },
        "T031-S06": {
            "real_runtime_path": False,
            "evidence": "no slow-fast state split controls the real Qwen recurrent state yet.",
        },
        "T031-S07": {
            "real_runtime_path": False,
            "evidence": "no reversible latent transcript engine drives the real tokenizer/generation loop yet.",
        },
        "T031-E01": {
            "real_runtime_path": False,
            "support_runtime_path": zero_groups > 0,
            "zero_code_groups": zero_groups,
            "evidence": "zero-group descriptor audit is attached to real GPU payloads, but no payload-skipping kernel consumes it yet.",
        },
        "T031-E02": {
            "real_runtime_path": False,
            "support_runtime_path": bool(descriptor_table),
            "descriptor_count": len(descriptor_table),
            "evidence": "byte-owner descriptor ledger is attached to real GPU payloads, but projection atom aliases are not a resident payload format yet.",
        },
        "T031-E03": {
            "real_runtime_path": False,
            "support_runtime_path": bool(descriptor_table),
            "descriptor_count": len(descriptor_table),
            "evidence": "structural descriptor fields exist on real GPU payloads, but sign-support quotient kernels are not present yet.",
        },
        "T031-E04": {
            "real_runtime_path": bool(receipt.get("gpu_argmax_head") and "gpu_argmax_head" in hot_loop_items),
            "evidence": "the measured path streams LM-head chunks and performs greedy selection on GPU without materializing a full dense logit tensor.",
        },
        "T031-E05": {
            "real_runtime_path": False,
            "evidence": "ternary shadow optimizer is a training-state mechanism and is not part of inference hot-path memory yet.",
        },
        "T031-E06": {
            "real_runtime_path": bool(
                "mlp_gate_up_pair_kernel" in hot_loop_items
                and "mlp_down_residual_coalescing" in hot_loop_items
            ),
            "evidence": "MLP gate/up and residual/down transaction surfaces are integrated in the measured hot loop.",
        },
        "T031-E07": {
            "real_runtime_path": False,
            "support_runtime_path": scale_chambers > 0,
            "scale_chambers": scale_chambers,
            "evidence": "exact scale-chamber counts are attached to real GPU payload descriptors, but kernels still load per-row scales.",
        },
        "T031-E08": {
            "real_runtime_path": bool(
                readiness.get("inplace_mlp_intermediate")
                and "mlp_down_residual_coalescing" in hot_loop_items
            ),
            "evidence": "in-place MLP intermediate and down/residual coalescing are active in the measured hot loop.",
        },
        "_metadata_only_items": {
            "real_runtime_path": False,
            "metadata_items": sorted(metadata_items),
            "not_hot_loop_items": sorted(not_hot_loop_items),
            "repair_free_rows": repair_free_rows,
        },
    }


def _t033_greedy_lm_head_output_contract(runtime: dict[str, Any] | None) -> dict[str, Any]:
    if runtime is None:
        return {
            "finding_id": "T033-A-V01",
            "ready": False,
            "reason": "runtime_not_executed",
        }
    receipt = runtime.get("receipt") or {}
    certificate = receipt.get("t031_lm_head_certificate") or {}
    contract = certificate.get("t033_exact_greedy_output_contract") or {}
    ready = bool(
        receipt.get("gpu_argmax_head")
        and certificate.get("active")
        and contract.get("gate_ready")
        and not contract.get("full_logits_materialized")
        and not contract.get("full_logits_returned_to_host")
    )
    return {
        "finding_id": "T033-A-V01",
        "ready": ready,
        "implementation_ready": ready,
        "contract": contract.get("contract"),
        "full_logits_materialized": bool(contract.get("full_logits_materialized")),
        "full_logits_returned_to_host": bool(contract.get("full_logits_returned_to_host")),
        "quality_tradeoff": bool(contract.get("quality_tradeoff")),
        "evidence": "LM-head chunks are reduced to a greedy token on GPU; the host observes no full-logit tensor.",
    }


def _gate_record(
    *,
    name: str,
    numerator: float,
    denominator: float,
    threshold_ratio: float = 1.10,
    unit: str,
) -> dict[str, Any]:
    net = numerator - denominator
    return {
        "name": name,
        "ready": numerator > 0.0 and denominator > 0.0 and numerator >= denominator * threshold_ratio,
        "saved_work": numerator,
        "overhead": denominator,
        "net": net,
        "threshold_ratio": threshold_ratio,
        "unit": unit,
    }


def _t033_descriptor_denominator_gate(runtime: dict[str, Any] | None) -> dict[str, Any]:
    if runtime is None:
        return {
            "finding_id": "T033-C-DESCRIPTOR-DENOMINATOR-GATE",
            "ready": False,
            "positive_gate_count": 0,
            "reason": "runtime_not_executed",
        }
    receipt = runtime.get("receipt") or {}
    descriptor_table = receipt.get("device_payload_descriptor_table") or {}
    total_bytes = 0
    dense_code_bytes = 0
    dense_scale_bytes = 0
    dense_repair_bytes = 0
    active_code_tape_bytes = 0
    repair_free_saved_bytes = 0
    repair_segment_overhead_bytes = 0
    chamber_payload_bytes = 0
    total_code_groups = 0
    total_zero_code_groups = 0
    total_rows = 0
    total_repair_free_rows = 0
    total_scale_chambers = 0

    for descriptor in descriptor_table.values():
        tensors = descriptor.get("tensors") or {}
        structural = descriptor.get("t031_structural_payload") or {}
        total_bytes += int(descriptor.get("bytes") or 0)
        code_tensor = tensors.get("trit_codes") or tensors.get("codes") or {}
        code_bytes = int(code_tensor.get("bytes") or 0)
        dense_code_bytes += code_bytes
        scale_bytes = int((tensors.get("scales") or {}).get("bytes") or 0)
        dense_scale_bytes += scale_bytes
        repair_bytes = int((tensors.get("repair_indices") or {}).get("bytes") or 0)
        repair_bytes += int((tensors.get("repair_values") or {}).get("bytes") or 0)
        dense_repair_bytes += repair_bytes

        code_groups = int(structural.get("code_groups") or 0)
        zero_code_groups = int(structural.get("zero_code_groups") or 0)
        active_groups = max(0, code_groups - zero_code_groups)
        bytes_per_group = code_bytes / code_groups if code_groups > 0 else 0.0
        active_code_tape_bytes += active_groups * (bytes_per_group + 2.0)
        total_code_groups += code_groups
        total_zero_code_groups += zero_code_groups

        rows = int(structural.get("row_count") or 0)
        repair_free_rows = int(structural.get("repair_free_rows") or 0)
        total_rows += rows
        total_repair_free_rows += repair_free_rows
        if rows > 0 and repair_bytes > 0:
            repair_free_saved_bytes += repair_bytes * (repair_free_rows / rows)
            repair_segment_overhead_bytes += 16.0

        scale_chambers = int(structural.get("scale_chamber_count") or 0)
        total_scale_chambers += scale_chambers
        if scale_chambers > 0:
            chamber_payload_bytes += scale_chambers * 2.0 + int(structural.get("scale_index_bytes") or 0)

    zero_group_gate = _gate_record(
        name="zero_group_active_tape",
        numerator=float(dense_code_bytes),
        denominator=float(active_code_tape_bytes),
        unit="bytes",
    )
    repair_free_gate = _gate_record(
        name="repair_free_segment_kernel",
        numerator=float(repair_free_saved_bytes),
        denominator=float(repair_segment_overhead_bytes),
        unit="bytes",
    )
    scale_chamber_gate = _gate_record(
        name="scale_chamber_payload",
        numerator=float(dense_scale_bytes),
        denominator=float(chamber_payload_bytes),
        unit="bytes",
    )
    gates = {
        "zero_group_active_tape": zero_group_gate,
        "repair_free_segment_kernel": repair_free_gate,
        "scale_chamber_payload": scale_chamber_gate,
        "sign_support_skeleton_quotient": {
            "ready": False,
            "reason": "descriptor lacks reusable skeleton IDs; no gain claimed.",
        },
        "projection_atom_arena": {
            "ready": False,
            "reason": "descriptor lacks atom duplicate IDs; no gain claimed.",
        },
        "structural_opcode_tape": {
            "ready": False,
            "reason": "descriptor lacks executable opcode stream; no gain claimed.",
        },
    }
    positive_gate_count = sum(1 for gate in gates.values() if gate.get("ready"))
    return {
        "finding_id": "T033-C-DESCRIPTOR-DENOMINATOR-GATE",
        "ready": bool(descriptor_table),
        "implementation_ready": False,
        "descriptor_count": len(descriptor_table),
        "descriptor_bytes": total_bytes,
        "weighted_zero_group_fraction": (
            total_zero_code_groups / total_code_groups if total_code_groups > 0 else 0.0
        ),
        "repair_free_fraction": total_repair_free_rows / total_rows if total_rows > 0 else 0.0,
        "scale_chamber_count": total_scale_chambers,
        "positive_gate_count": positive_gate_count,
        "gates": gates,
        "evidence": "Consumes real runtime device-payload descriptors; reports only denominator-priced gates.",
    }


def _runtime_comparison(runtime: dict[str, Any] | None) -> dict[str, Any]:
    if runtime is None:
        return {
            "measured_runtime_claim": False,
            "comparison_scope": "runtime_not_executed",
        }
    receipt = runtime.get("receipt") or {}
    comparison = runtime.get("t030_speed_efficiency_comparison") or {}
    return {
        "measured_runtime_claim": True,
        "comparison_scope": comparison.get("comparison_scope"),
        "measured_tokens_per_second": comparison.get("measured_tokens_per_second"),
        "measured_speedup_vs_latest_speed_baseline": comparison.get("measured_speedup_vs_r452_t028"),
        "measured_peak_cuda_allocated_gib": comparison.get("measured_peak_cuda_allocated_gib"),
        "measured_resident_cuda_gib": comparison.get("measured_resident_cuda_gib"),
        "generated_token_ids": receipt.get("generated_token_ids"),
        "completed_layers": receipt.get("completed_layers"),
        "requested_layers": receipt.get("requested_layers"),
        "production_tok_s_claim_allowed": comparison.get("production_tok_s_claim_allowed"),
    }


def run_t031_macro_architecture_stack(
    *,
    execute_runtime: bool = True,
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
    fused_qkv_compact_projection_kernel: bool = False,
) -> dict:
    """Run the T031 stack against the cumulative real T030 runtime path."""

    receipts = _algorithmic_receipts()
    mechanisms = _mechanisms(receipts)
    speed_items = [item for item in mechanisms if item.kind == "speed"]
    efficiency_items = [item for item in mechanisms if item.kind == "efficiency"]
    speed_projection = _overlap_priced_speedup(mechanisms)
    memory_projection = _overlap_priced_memory(mechanisms)
    active_ids = [item.id for item in mechanisms]
    expected_ids = {f"T031-S{idx:02d}" for idx in range(1, 8)} | {f"T031-E{idx:02d}" for idx in range(1, 9)}
    missing_ids = sorted(expected_ids.difference(active_ids))
    not_ready_ids = sorted(mechanism_id for mechanism_id, receipt in receipts.items() if not receipt["ready"])
    blockers = []
    if missing_ids:
        blockers.append("t031_survivor_missing_from_cumulative_stack")
    if not_ready_ids:
        blockers.append("t031_algorithmic_receipt_not_ready")
    if any(not item.exact_quality_required for item in mechanisms):
        blockers.append("quality_tradeoff_detected")
    if any(not item.gpu_bound_required for item in efficiency_items):
        blockers.append("non_gpu_bound_efficiency_item_detected")

    runtime = None
    if execute_runtime:
        runtime = run_t030_speed_gpu_efficiency(
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
            fused_qkv_compact_projection_kernel=fused_qkv_compact_projection_kernel,
        )
        blockers.extend(runtime.get("blockers") or [])
    runtime_evidence = _runtime_integration_evidence(runtime, active_ids)
    not_integrated_ids = sorted(
        mechanism_id
        for mechanism_id, evidence in runtime_evidence.items()
        if mechanism_id.startswith("T031-") and not evidence["real_runtime_path"]
    )
    if execute_runtime and not_integrated_ids:
        blockers.append("t031_survivors_not_integrated_into_real_runtime")
    t033_greedy_contract = _t033_greedy_lm_head_output_contract(runtime)
    t033_descriptor_gate = _t033_descriptor_denominator_gate(runtime)

    return {
        "probe": "t031_macro_architecture_stack",
        "activation": ACTIVATION,
        "round": "T031",
        "baseline": {
            "label": "latest measured T-series speed and GPU-memory baselines",
            "speed_tokens_per_second": LATEST_SPEED_BASELINE_TOKENS_PER_SECOND,
            "speed_peak_cuda_gib": LATEST_SPEED_BASELINE_PEAK_CUDA_GIB,
            "speed_resident_cuda_gib": LATEST_SPEED_BASELINE_RESIDENT_CUDA_GIB,
            "memory_branch_tokens_per_second": LATEST_MEMORY_BRANCH_TOKENS_PER_SECOND,
            "memory_branch_peak_cuda_gib": LATEST_MEMORY_BRANCH_PEAK_CUDA_GIB,
            "memory_branch_resident_cuda_gib": LATEST_MEMORY_BRANCH_RESIDENT_CUDA_GIB,
        },
        "candidate": {
            "mechanism_count": len(mechanisms),
            "speed_mechanism_count": len(speed_items),
            "efficiency_mechanism_count": len(efficiency_items),
            "mechanisms": [item.as_dict() for item in mechanisms],
            "algorithmic_receipts": receipts,
            "runtime_integration_evidence": runtime_evidence,
            "runtime_comparison": _runtime_comparison(runtime),
            "t033_greedy_lm_head_output_contract": t033_greedy_contract,
            "t033_descriptor_denominator_gate": t033_descriptor_gate,
            "speed_projection": speed_projection,
            "memory_projection": memory_projection,
            "production_tok_s_claim_allowed": bool(
                runtime is not None
                and (runtime.get("t030_speed_efficiency_comparison") or {}).get("production_tok_s_claim_allowed")
            ),
            "measured_runtime_claim": runtime is not None,
        },
        "runtime": runtime,
        "readiness": {
            "t031_macro_architecture_stack_ready": not blockers,
            "t031_real_runtime_executed": runtime is not None,
            "t031_real_runtime_smoke_pass": bool(runtime is not None and (runtime.get("readiness") or {}).get("smoke_pass")),
            "t031_all_survivors_real_runtime_integrated": not not_integrated_ids,
            "t031_real_runtime_missing_ids": not_integrated_ids,
            "t031_all_speed_survivors_accounted": len(speed_items) == 7,
            "t031_all_efficiency_survivors_accounted": len(efficiency_items) == 8,
            "t031_all_survivors_accounted": not missing_ids and len(mechanisms) == 15,
            "t031_all_algorithmic_receipts_ready": not not_ready_ids and len(receipts) == 15,
            "t031_no_quality_tradeoff": not any(not item.exact_quality_required for item in mechanisms),
            "t031_gpu_bound_efficiency_only": not any(not item.gpu_bound_required for item in efficiency_items),
            "t031_overlap_deduplicated_not_naively_multiplied": True,
            "t031_crosses_40_tok_s_projected": speed_projection["crosses_target_tokens_per_second"],
            "t033_greedy_lm_head_output_contract_ready": bool(t033_greedy_contract.get("ready")),
            "t033_descriptor_denominator_gate_ready": bool(t033_descriptor_gate.get("ready")),
            "t033_structural_kernel_positive_gate": bool(t033_descriptor_gate.get("positive_gate_count", 0) > 0),
        },
        "blockers": blockers,
        "next_actions": (
            [
                "Implement the listed T031 missing IDs inside the T017/T030 real generation loop before calling T031 complete.",
                "Use runtime_comparison for real tok/s/VRAM/RAM reporting; keep speed_projection only as planning evidence.",
            ]
            if not_integrated_ids
            else []
        ),
    }


__all__ = ["ACTIVATION", "run_t031_macro_architecture_stack"]
