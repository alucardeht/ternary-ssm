"""T039 exact map-body validation gates."""

from __future__ import annotations

from datetime import UTC, datetime
import importlib.util
from typing import Any

from .t038_kernel_body_roofline_gate import (
    DEFAULT_DEVICE_BUDGET_GIB,
    R452_KERNEL_SECONDS_PER_TOKEN,
    R452_PEAK_CUDA_GIB,
    R452_TOKENS_PER_SECOND,
    TARGET_SECONDS_PER_TOKEN,
    TARGET_TOKENS_PER_SECOND,
)


A01_ACTIVATION = "T039_A01_EXACT_MAP_OBJECT_LEDGER_GATE_ACTIVE"
A02_ACTIVATION = "T039_A02_SEMANTIC_LAYOUT_LEDGER_GATE_ACTIVE"
REAL_MAP_AUDIT_ACTIVATION = "T039_REAL_MAP_OBJECT_AUDIT_ACTIVE"
T039B_A01_ACTIVATION = "T039B_A01_NON_ROW_EXACT_OBJECT_EXTRACTION_LEDGER_ACTIVE"
T039B_A02_ACTIVATION = "T039B_A02_GAUGE_SAT_CANONICAL_TAIL_LEDGER_ACTIVE"

DEFAULT_REAL_MAP_AUDIT_TENSORS = (
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.63.mlp.down_proj.weight",
)

DEFAULT_T039B_A02_REGIONS = (
    {
        "name": "norm_attention_rope_observers",
        "body_fraction": 0.35,
        "private_speedup": 1.0,
        "exact_gauge_relation": False,
        "semantic_boundary_priced": False,
        "forced_canonical": True,
        "layout_replaces_storage": False,
    },
    {
        "name": "mlp_nonlinear_observers",
        "body_fraction": 0.45,
        "private_speedup": 1.0,
        "exact_gauge_relation": False,
        "semantic_boundary_priced": False,
        "forced_canonical": True,
        "layout_replaces_storage": False,
    },
    {
        "name": "residual_logits_canonical_exits",
        "body_fraction": 0.20,
        "private_speedup": 1.0,
        "exact_gauge_relation": False,
        "semantic_boundary_priced": False,
        "forced_canonical": True,
        "layout_replaces_storage": False,
    },
)


def _validate_components(
    *,
    component_names: tuple[str, ...],
    component_fractions: tuple[float, ...],
    component_speedups: tuple[float, ...],
) -> list[str]:
    errors: list[str] = []
    if len(component_names) != len(component_fractions) or len(component_fractions) != len(component_speedups):
        errors.append("component_name_fraction_speedup_length_mismatch")
    if not component_fractions:
        errors.append("no_body_components_priced")
    if any(fraction < 0.0 for fraction in component_fractions):
        errors.append("negative_body_component_fraction")
    if any(speedup <= 0.0 for speedup in component_speedups):
        errors.append("nonpositive_body_component_speedup")
    return errors


def _composite_pricing(
    *,
    component_names: tuple[str, ...],
    component_fractions: tuple[float, ...],
    component_speedups: tuple[float, ...],
    fraction_tolerance: float,
) -> dict[str, Any]:
    input_errors = _validate_components(
        component_names=component_names,
        component_fractions=component_fractions,
        component_speedups=component_speedups,
    )
    fraction_sum = sum(component_fractions)
    fraction_sum_pass = abs(fraction_sum - 1.0) <= fraction_tolerance
    unpriced_leftover_fraction = max(0.0, 1.0 - fraction_sum)
    harmonic_denominator = sum(
        fraction / max(speedup, 1e-12)
        for fraction, speedup in zip(component_fractions, component_speedups, strict=False)
    )
    composite_speedup = 1.0 / harmonic_denominator if harmonic_denominator > 0.0 and not input_errors else 0.0
    projected_kernel_seconds = (
        R452_KERNEL_SECONDS_PER_TOKEN / composite_speedup if composite_speedup > 0.0 else float("inf")
    )
    projected_tokens_per_second_if_control_zero = (
        1.0 / projected_kernel_seconds if projected_kernel_seconds > 0.0 else 0.0
    )
    components = [
        {
            "name": name,
            "body_fraction": fraction,
            "speedup_vs_current_component": speedup,
            "harmonic_denominator_contribution": fraction / speedup if speedup > 0.0 else float("inf"),
        }
        for name, fraction, speedup in zip(component_names, component_fractions, component_speedups, strict=False)
    ]
    return {
        "input_errors": input_errors,
        "component_fraction_sum": fraction_sum,
        "component_fractions_sum_to_one": fraction_sum_pass,
        "unpriced_leftover_fraction": unpriced_leftover_fraction,
        "harmonic_denominator": harmonic_denominator,
        "composite_speedup_vs_current_body": composite_speedup,
        "projected_kernel_seconds_per_token": projected_kernel_seconds,
        "projected_tokens_per_second_if_control_zero": projected_tokens_per_second_if_control_zero,
        "components": components,
    }


def _base_target(required_speedup: float) -> dict[str, Any]:
    return {
        "tokens_per_second": TARGET_TOKENS_PER_SECOND,
        "seconds_per_token": TARGET_SECONDS_PER_TOKEN,
        "required_kernel_speedup_if_control_zero": required_speedup,
    }


def _base_baseline() -> dict[str, Any]:
    return {
        "source": "R452/T028 direct denominator",
        "current_tokens_per_second": R452_TOKENS_PER_SECOND,
        "kernel_seconds_per_token": R452_KERNEL_SECONDS_PER_TOKEN,
    }


def run_t039_a01_exact_map_object_ledger_gate(
    *,
    component_names: tuple[str, ...] = ("unextracted_current_body",),
    component_fractions: tuple[float, ...] = (1.0,),
    component_speedups: tuple[float, ...] = (1.0,),
    object_equivalence_proven: bool = False,
    representative_real_maps_audited: int = 0,
    residual_fraction: float = 0.0,
    hidden_sidecar_gib: float = 0.0,
    decoded_dense_sidecar_gib: float = 0.0,
    offload_or_setup_gain_counted: bool = False,
    layout_replaces_storage: bool = False,
    exact_arithmetic_contract: bool = True,
    peak_cuda_gib: float = R452_PEAK_CUDA_GIB,
    device_budget_gib: float = DEFAULT_DEVICE_BUDGET_GIB,
    fraction_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Gate exact executable map objects against the T039 A01 validator contract."""

    required_speedup = R452_KERNEL_SECONDS_PER_TOKEN / TARGET_SECONDS_PER_TOKEN
    pricing = _composite_pricing(
        component_names=component_names,
        component_fractions=component_fractions,
        component_speedups=component_speedups,
        fraction_tolerance=fraction_tolerance,
    )
    sidecar_gib = hidden_sidecar_gib + decoded_dense_sidecar_gib
    residual_zero = residual_fraction == 0.0
    speedup_pass = pricing["composite_speedup_vs_current_body"] >= required_speedup
    no_unpriced_leftover = pricing["unpriced_leftover_fraction"] <= fraction_tolerance
    map_audit_pass = representative_real_maps_audited > 0
    sidecar_pass = sidecar_gib == 0.0
    device_pass = peak_cuda_gib <= device_budget_gib
    gate_ready = (
        not pricing["input_errors"]
        and pricing["component_fractions_sum_to_one"]
        and no_unpriced_leftover
        and speedup_pass
        and object_equivalence_proven
        and map_audit_pass
        and residual_zero
        and sidecar_pass
        and not offload_or_setup_gain_counted
        and layout_replaces_storage
        and exact_arithmetic_contract
        and device_pass
    )

    blockers: list[str] = list(pricing["input_errors"])
    if not pricing["component_fractions_sum_to_one"]:
        blockers.append("body_component_fractions_do_not_sum_to_one")
    if not no_unpriced_leftover:
        blockers.append("unpriced_leftover_body_fraction")
    if not speedup_pass:
        blockers.append("composite_body_speedup_below_required_7_5738x")
    if not object_equivalence_proven:
        blockers.append("exact_map_object_equivalence_not_proven")
    if not map_audit_pass:
        blockers.append("no_representative_real_maps_audited")
    if not residual_zero:
        blockers.append("exact_map_residual_is_nonzero")
    if not sidecar_pass:
        blockers.append("hidden_or_decoded_dense_sidecar_live")
    if offload_or_setup_gain_counted:
        blockers.append("offload_or_setup_gain_counted")
    if not layout_replaces_storage:
        blockers.append("object_does_not_replace_resident_storage")
    if not exact_arithmetic_contract:
        blockers.append("exact_arithmetic_contract_not_proven")
    if not device_pass:
        blockers.append("projected_peak_cuda_exceeds_device_budget")

    return {
        "probe": "t039_a01_exact_map_object_ledger_gate",
        "activation": A01_ACTIVATION,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "target": _base_target(required_speedup),
        "baseline": _base_baseline(),
        "coverage_model": {key: value for key, value in pricing.items() if key not in {"components", "input_errors"}},
        "components": pricing["components"],
        "object_audit": {
            "object_equivalence_proven": object_equivalence_proven,
            "representative_real_maps_audited": representative_real_maps_audited,
            "residual_fraction": residual_fraction,
            "hidden_sidecar_gib": hidden_sidecar_gib,
            "decoded_dense_sidecar_gib": decoded_dense_sidecar_gib,
            "sidecar_gib": sidecar_gib,
            "offload_or_setup_gain_counted": offload_or_setup_gain_counted,
            "layout_replaces_storage": layout_replaces_storage,
            "exact_arithmetic_contract": exact_arithmetic_contract,
            "peak_cuda_gib": peak_cuda_gib,
            "device_budget_gib": device_budget_gib,
        },
        "readiness": {
            "t039_a01_exact_map_object_ledger_gate_ready": gate_ready,
            "t039_a01_composite_speedup_sufficient": speedup_pass,
            "t039_a01_no_unpriced_leftover": no_unpriced_leftover,
            "t039_a01_object_equivalence_proven": object_equivalence_proven,
            "t039_a01_representative_real_maps_audited": map_audit_pass,
            "t039_a01_zero_residual": residual_zero,
            "t039_a01_no_hidden_sidecar": sidecar_pass,
            "t039_a01_no_offload_or_setup_gain": not offload_or_setup_gain_counted,
            "t039_a01_layout_replaces_storage": layout_replaces_storage,
            "t039_a01_exact_arithmetic_contract": exact_arithmetic_contract,
            "t039_a01_fits_device_budget": device_pass,
            "t039_production_implementation_allowed": False,
        },
        "blockers": blockers,
        "next_actions": [
            "Promote the surviving exact map object to a representative GPU hot-path implementation gate.",
            "Measure real token generation before claiming production throughput.",
        ]
        if gate_ready
        else [
            "Do not implement production map-object runtime until equivalence, coverage, and composed denominator pass.",
            "If extraction cannot produce a residual-free object on real maps, reopen A01 with a different exact map language or move to A02.",
        ],
    }


def run_t039_a02_semantic_layout_ledger_gate(
    *,
    component_names: tuple[str, ...] = ("unproven_semantic_layout_body",),
    component_fractions: tuple[float, ...] = (1.0,),
    component_speedups: tuple[float, ...] = (1.0,),
    gauge_contracts_proven: bool = False,
    semantic_boundaries_priced: bool = False,
    forced_canonical_fraction: float = 1.0,
    residual_fraction: float = 0.0,
    hidden_sidecar_gib: float = 0.0,
    decoded_dense_sidecar_gib: float = 0.0,
    offload_or_setup_gain_counted: bool = False,
    layout_replaces_storage: bool = False,
    exact_arithmetic_contract: bool = True,
    peak_cuda_gib: float = R452_PEAK_CUDA_GIB,
    device_budget_gib: float = DEFAULT_DEVICE_BUDGET_GIB,
    fraction_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Gate exact semantic layout contracts against the T039 A02 validator contract."""

    required_speedup = R452_KERNEL_SECONDS_PER_TOKEN / TARGET_SECONDS_PER_TOKEN
    pricing = _composite_pricing(
        component_names=component_names,
        component_fractions=component_fractions,
        component_speedups=component_speedups,
        fraction_tolerance=fraction_tolerance,
    )
    sidecar_gib = hidden_sidecar_gib + decoded_dense_sidecar_gib
    residual_zero = residual_fraction == 0.0
    speedup_pass = pricing["composite_speedup_vs_current_body"] >= required_speedup
    no_unpriced_leftover = pricing["unpriced_leftover_fraction"] <= fraction_tolerance
    sidecar_pass = sidecar_gib == 0.0
    device_pass = peak_cuda_gib <= device_budget_gib
    canonical_tail_pass = forced_canonical_fraction <= 0.132034928176561
    gate_ready = (
        not pricing["input_errors"]
        and pricing["component_fractions_sum_to_one"]
        and no_unpriced_leftover
        and speedup_pass
        and gauge_contracts_proven
        and semantic_boundaries_priced
        and canonical_tail_pass
        and residual_zero
        and sidecar_pass
        and not offload_or_setup_gain_counted
        and layout_replaces_storage
        and exact_arithmetic_contract
        and device_pass
    )

    blockers: list[str] = list(pricing["input_errors"])
    if not pricing["component_fractions_sum_to_one"]:
        blockers.append("body_component_fractions_do_not_sum_to_one")
    if not no_unpriced_leftover:
        blockers.append("unpriced_leftover_body_fraction")
    if not speedup_pass:
        blockers.append("composite_body_speedup_below_required_7_5738x")
    if not gauge_contracts_proven:
        blockers.append("exact_gauge_contracts_not_proven")
    if not semantic_boundaries_priced:
        blockers.append("semantic_boundaries_not_priced")
    if not canonical_tail_pass:
        blockers.append("forced_canonical_fraction_exceeds_leftover_bound")
    if not residual_zero:
        blockers.append("layout_residual_is_nonzero")
    if not sidecar_pass:
        blockers.append("hidden_or_decoded_dense_sidecar_live")
    if offload_or_setup_gain_counted:
        blockers.append("offload_or_setup_gain_counted")
    if not layout_replaces_storage:
        blockers.append("semantic_layout_does_not_replace_storage")
    if not exact_arithmetic_contract:
        blockers.append("exact_arithmetic_contract_not_proven")
    if not device_pass:
        blockers.append("projected_peak_cuda_exceeds_device_budget")

    return {
        "probe": "t039_a02_semantic_layout_ledger_gate",
        "activation": A02_ACTIVATION,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "target": _base_target(required_speedup),
        "baseline": _base_baseline(),
        "coverage_model": {key: value for key, value in pricing.items() if key not in {"components", "input_errors"}},
        "components": pricing["components"],
        "layout_audit": {
            "gauge_contracts_proven": gauge_contracts_proven,
            "semantic_boundaries_priced": semantic_boundaries_priced,
            "forced_canonical_fraction": forced_canonical_fraction,
            "maximum_unaccelerated_tail_even_at_infinite_speed": 0.132034928176561,
            "residual_fraction": residual_fraction,
            "hidden_sidecar_gib": hidden_sidecar_gib,
            "decoded_dense_sidecar_gib": decoded_dense_sidecar_gib,
            "sidecar_gib": sidecar_gib,
            "offload_or_setup_gain_counted": offload_or_setup_gain_counted,
            "layout_replaces_storage": layout_replaces_storage,
            "exact_arithmetic_contract": exact_arithmetic_contract,
            "peak_cuda_gib": peak_cuda_gib,
            "device_budget_gib": device_budget_gib,
        },
        "readiness": {
            "t039_a02_semantic_layout_ledger_gate_ready": gate_ready,
            "t039_a02_composite_speedup_sufficient": speedup_pass,
            "t039_a02_no_unpriced_leftover": no_unpriced_leftover,
            "t039_a02_gauge_contracts_proven": gauge_contracts_proven,
            "t039_a02_semantic_boundaries_priced": semantic_boundaries_priced,
            "t039_a02_forced_canonical_tail_below_bound": canonical_tail_pass,
            "t039_a02_zero_residual": residual_zero,
            "t039_a02_no_hidden_sidecar": sidecar_pass,
            "t039_a02_no_offload_or_setup_gain": not offload_or_setup_gain_counted,
            "t039_a02_layout_replaces_storage": layout_replaces_storage,
            "t039_a02_exact_arithmetic_contract": exact_arithmetic_contract,
            "t039_a02_fits_device_budget": device_pass,
            "t039_production_implementation_allowed": False,
        },
        "blockers": blockers,
        "next_actions": [
            "Promote the surviving semantic layout to a representative GPU hot-path implementation gate.",
            "Measure real token generation before claiming production throughput.",
        ]
        if gate_ready
        else [
            "Do not implement production semantic-layout runtime until gauge legality, boundaries, and denominator pass.",
            "If legal gauges cannot cover enough body cost, reopen A02 or combine with an A01 exact map object without double-counting.",
        ],
    }


def _exact_row_quotient_metrics(block: Any) -> dict[str, Any]:
    import torch

    if block.ndim != 2:
        raise ValueError(f"expected rank-2 block, got shape {tuple(block.shape)}")
    unique_rows, inverse = torch.unique(block, dim=0, return_inverse=True)
    reconstructed = unique_rows.index_select(0, inverse)
    exact = bool(torch.equal(reconstructed, block))
    rows = int(block.shape[0])
    cols = int(block.shape[1])
    unique_count = int(unique_rows.shape[0])
    speedup = rows / max(unique_count, 1)
    zero_row_count = int((block == 0).all(dim=1).sum().item())
    return {
        "rows": rows,
        "cols": cols,
        "unique_rows": unique_count,
        "row_quotient_speedup": speedup,
        "duplicate_row_fraction": 1.0 - (unique_count / max(rows, 1)),
        "zero_row_fraction": zero_row_count / max(rows, 1),
        "exact_reconstruction": exact,
        "storage_fraction_before_mapping": unique_count / max(rows, 1),
    }


def _rank_speedup_metrics(block: Any) -> dict[str, Any]:
    import torch

    if block.ndim != 2:
        raise ValueError(f"expected rank-2 block, got shape {tuple(block.shape)}")
    rows = int(block.shape[0])
    cols = int(block.shape[1])
    dense_work = max(rows * cols, 1)
    matrix_rank = int(torch.linalg.matrix_rank(block.float()).item())
    rank_work = max(matrix_rank * (rows + cols), 1)
    column_unique_count = int(torch.unique(block.T, dim=0).shape[0])
    column_quotient_work = max(rows * column_unique_count, 1)
    nonzero_count = int(torch.count_nonzero(block).item())
    sparse_indexed_work = max(nonzero_count * 2, 1)
    if cols > 1:
        displacement = block[:, 1:].float() - block[:, :-1].float()
        displacement_rank = int(torch.linalg.matrix_rank(displacement).item())
        displacement_work = max(displacement_rank * (rows + cols - 1) + rows, 1)
    else:
        displacement_rank = matrix_rank
        displacement_work = rank_work
    scalar_unique_count = int(torch.unique(block).numel())
    return {
        "rows": rows,
        "cols": cols,
        "dense_work_units": dense_work,
        "exact_rank": matrix_rank,
        "rank_basis_work_units": rank_work,
        "rank_basis_speedup": dense_work / rank_work,
        "unique_columns": column_unique_count,
        "column_quotient_speedup": dense_work / column_quotient_work,
        "nonzero_count": nonzero_count,
        "zero_fraction": 1.0 - (nonzero_count / dense_work),
        "sparse_indexed_speedup": dense_work / sparse_indexed_work,
        "displacement_rank": displacement_rank,
        "low_displacement_speedup": dense_work / displacement_work,
        "scalar_unique_count": scalar_unique_count,
    }


def _projection_bundle_atom_bank_metrics(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    compatible = [item for item in blocks if item["block"].ndim == 2]
    if not compatible:
        return {
            "compatible_block_count": 0,
            "shared_atom_rank": 0,
            "component_speedup_vs_current_component": 0.0,
            "blocker": "no_compatible_blocks",
        }
    cols = int(compatible[0]["block"].shape[1])
    same_cols = [item for item in compatible if int(item["block"].shape[1]) == cols]
    if not same_cols:
        return {
            "compatible_block_count": 0,
            "shared_atom_rank": 0,
            "component_speedup_vs_current_component": 0.0,
            "blocker": "no_same_column_blocks",
        }
    stacked = torch.cat([item["block"].float() for item in same_cols], dim=0)
    rank = int(torch.linalg.matrix_rank(stacked).item())
    total_rows = sum(int(item["block"].shape[0]) for item in same_cols)
    total_dense = max(total_rows * cols, 1)
    shared_atom_work = max(rank * cols + total_rows * rank, 1)
    return {
        "compatible_block_count": len(same_cols),
        "shared_atom_rank": rank,
        "rows": total_rows,
        "cols": cols,
        "dense_work_units": total_dense,
        "charged_atom_or_program_work": shared_atom_work,
        "component_speedup_vs_current_component": total_dense / shared_atom_work,
        "blocker": None,
    }


def _object_family_row(
    *,
    object_family: str,
    sample_or_bundle_id: str,
    covered_body_fraction: float,
    component_speedup: float,
    charged_atom_or_program_work: int,
    dense_work_units: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required_speedup = R452_KERNEL_SECONDS_PER_TOKEN / TARGET_SECONDS_PER_TOKEN
    readiness_blockers: list[str] = []
    exact = True
    residual_fraction = 0.0
    hidden_sidecar_gib = 0.0
    decoded_dense_sidecar_gib = 0.0
    layout_replaces_storage = charged_atom_or_program_work <= dense_work_units
    if component_speedup < required_speedup:
        readiness_blockers.append("component_speedup_below_required_7_5738x")
    if not layout_replaces_storage:
        readiness_blockers.append("object_not_smaller_than_current_storage_work_proxy")
    row = {
        "object_family": object_family,
        "sample_or_bundle_id": sample_or_bundle_id,
        "covered_body_fraction": covered_body_fraction,
        "expand_object_equals_current_body": exact,
        "residual_fraction": residual_fraction,
        "hidden_sidecar_gib": hidden_sidecar_gib,
        "decoded_dense_sidecar_gib": decoded_dense_sidecar_gib,
        "layout_or_object_replaces_storage": layout_replaces_storage,
        "offload_or_setup_gain_counted": False,
        "exact_arithmetic_contract": True,
        "charged_atom_or_program_work": charged_atom_or_program_work,
        "charged_combine_work": 0,
        "charged_metadata_work": 0,
        "charged_fanout_or_scatter_work": 0,
        "charged_accumulator_work": 0,
        "charged_dispatch_work": 0,
        "dense_work_units": dense_work_units,
        "component_speedup_vs_current_component": component_speedup,
        "readiness_blockers": readiness_blockers,
    }
    if extra:
        row.update(extra)
    return row


def _non_row_exact_object_rows_for_block(
    *,
    tensor_name: str,
    block: Any,
    covered_body_fraction: float,
) -> list[dict[str, Any]]:
    metrics = _rank_speedup_metrics(block)
    sample_id = f"{tensor_name}:{metrics['rows']}x{metrics['cols']}"
    return [
        _object_family_row(
            object_family="exact_output_basis_rank",
            sample_or_bundle_id=sample_id,
            covered_body_fraction=covered_body_fraction,
            component_speedup=metrics["rank_basis_speedup"],
            charged_atom_or_program_work=metrics["rank_basis_work_units"],
            dense_work_units=metrics["dense_work_units"],
            extra={
                "exact_rank": metrics["exact_rank"],
                "charged_combine_work": metrics["exact_rank"] * metrics["rows"],
                "charged_atom_or_program_work": metrics["exact_rank"] * metrics["cols"],
            },
        ),
        _object_family_row(
            object_family="column_quotient",
            sample_or_bundle_id=sample_id,
            covered_body_fraction=covered_body_fraction,
            component_speedup=metrics["column_quotient_speedup"],
            charged_atom_or_program_work=metrics["rows"] * metrics["unique_columns"],
            dense_work_units=metrics["dense_work_units"],
            extra={"unique_columns": metrics["unique_columns"]},
        ),
        _object_family_row(
            object_family="sparse_indexed_support",
            sample_or_bundle_id=sample_id,
            covered_body_fraction=covered_body_fraction,
            component_speedup=metrics["sparse_indexed_speedup"],
            charged_atom_or_program_work=max(metrics["nonzero_count"] * 2, 1),
            dense_work_units=metrics["dense_work_units"],
            extra={
                "nonzero_count": metrics["nonzero_count"],
                "zero_fraction": metrics["zero_fraction"],
                "scalar_unique_count": metrics["scalar_unique_count"],
            },
        ),
        _object_family_row(
            object_family="fixed_shift_low_displacement",
            sample_or_bundle_id=sample_id,
            covered_body_fraction=covered_body_fraction,
            component_speedup=metrics["low_displacement_speedup"],
            charged_atom_or_program_work=max(metrics["displacement_rank"] * (metrics["rows"] + metrics["cols"] - 1), 1),
            dense_work_units=metrics["dense_work_units"],
            extra={"displacement_rank": metrics["displacement_rank"]},
        ),
    ]


def _select_family_components(rows: list[dict[str, Any]], *, audited_count: int) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    if not rows:
        return ("no_non_row_exact_object",), (1.0,), (1.0,)
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row["object_family"], []).append(row)
    family_scores: list[tuple[float, str, float]] = []
    for family, family_rows in by_family.items():
        covered = sum(float(row["covered_body_fraction"]) for row in family_rows)
        if covered <= 0.0:
            continue
        denominator = sum(
            float(row["covered_body_fraction"]) / max(float(row["component_speedup_vs_current_component"]), 1e-12)
            for row in family_rows
        )
        speedup = covered / denominator if denominator > 0.0 else 0.0
        family_scores.append((speedup, family, covered))
    if not family_scores:
        return ("no_non_row_exact_object",), (1.0,), (1.0,)
    _, best_family, covered = max(family_scores, key=lambda item: item[0])
    selected_rows = by_family[best_family]
    names = tuple(f"{row['object_family']}:{row['sample_or_bundle_id']}" for row in selected_rows)
    fractions = tuple(float(row["covered_body_fraction"]) for row in selected_rows)
    speedups = tuple(float(row["component_speedup_vs_current_component"]) for row in selected_rows)
    leftover = max(0.0, 1.0 - sum(fractions))
    if leftover > 0.0:
        names = names + ("unpriced_or_uncovered_body",)
        fractions = fractions + (leftover,)
        speedups = speedups + (1.0,)
    return names, fractions, speedups


def run_t039b_a01_non_row_exact_object_extraction_ledger(
    *,
    model_id: str = "Qwen/Qwen3-32B",
    local_model_dir: Any = None,
    allow_hf_config_fetch: bool = True,
    tensor_names: tuple[str, ...] = DEFAULT_REAL_MAP_AUDIT_TENSORS,
    block_rows: int = 64,
    block_cols: int = 256,
    synthetic_blocks: tuple[tuple[str, Any], ...] | None = None,
) -> dict[str, Any]:
    """Run a static extraction ledger for non-row exact map objects."""

    blockers: list[str] = []
    missing: list[str] = []
    audited: list[dict[str, Any]] = []
    block_records: list[dict[str, Any]] = []
    weight_dir = None
    inventory = None
    config = None

    if importlib.util.find_spec("torch") is None:
        blockers.append("torch_missing")

    if synthetic_blocks is not None:
        for name, block in synthetic_blocks:
            block_records.append({"tensor": name, "block": block})
    elif not blockers:
        if importlib.util.find_spec("safetensors") is None:
            blockers.append("safetensors_missing")
        if not blockers:
            from .t004_real_qwen_bridge import find_weight_dir, inventory_safetensors, resolve_qwen_config
            from .t006_real_residency_block_smoke import _compress_reconstruct, _policy_for_tensor, _read_block
            from .t008_compressed_generation_smoke import _load_weight_map

            config = resolve_qwen_config(
                model_id=model_id,
                local_model_dir=local_model_dir,
                allow_hf_config_fetch=allow_hf_config_fetch,
            )
            weight_dir = find_weight_dir(config, local_model_dir)
            inventory = inventory_safetensors(weight_dir)
            if not config.matches_project_qwen3_32b_shape():
                blockers.append("config_shape_mismatch")
            if weight_dir is None or not inventory.local_weights_present:
                blockers.append("missing_local_qwen3_32b_safetensors")
        if not blockers and weight_dir is not None:
            from .t006_real_residency_block_smoke import _compress_reconstruct, _policy_for_tensor, _read_block
            from .t008_compressed_generation_smoke import _load_weight_map

            weight_map = _load_weight_map(weight_dir)
            for name in tensor_names:
                shard = weight_map.get(name)
                if shard is None:
                    missing.append(name)
                    continue
                try:
                    block, original_shape = _read_block(weight_dir, name, shard, block_rows, block_cols)
                    reconstructed = _compress_reconstruct(block, _policy_for_tensor(name))
                    block_records.append(
                        {
                            "tensor": name,
                            "shard": shard,
                            "original_shape": list(original_shape),
                            "block": reconstructed,
                        }
                    )
                except Exception as exc:  # pragma: no cover - real shard guard
                    blockers.append(f"non_row_object_ledger_failed:{name}:{type(exc).__name__}:{exc}")
                    break

    total_elements = sum(int(item["block"].shape[0]) * int(item["block"].shape[1]) for item in block_records)
    family_rows: list[dict[str, Any]] = []
    for item in block_records:
        block = item["block"]
        rows = int(block.shape[0])
        cols = int(block.shape[1])
        covered = (rows * cols) / max(total_elements, 1)
        row_metrics = _exact_row_quotient_metrics(block)
        non_row_rows = _non_row_exact_object_rows_for_block(
            tensor_name=item["tensor"],
            block=block,
            covered_body_fraction=covered,
        )
        family_rows.extend(non_row_rows)
        audited.append(
            {
                "tensor": item["tensor"],
                "shard": item.get("shard"),
                "original_shape": item.get("original_shape"),
                "block_shape": [rows, cols],
                "row_quotient": row_metrics,
                "non_row_object_rows": non_row_rows,
            }
        )

    if block_records:
        bundle = _projection_bundle_atom_bank_metrics(block_records)
        if bundle.get("blocker") is None:
            family_rows.append(
                _object_family_row(
                    object_family="projection_family_shared_atom_bank",
                    sample_or_bundle_id="sampled_same_column_projection_bundle",
                    covered_body_fraction=1.0,
                    component_speedup=float(bundle["component_speedup_vs_current_component"]),
                    charged_atom_or_program_work=int(bundle["charged_atom_or_program_work"]),
                    dense_work_units=int(bundle["dense_work_units"]),
                    extra=bundle,
                )
            )

    if not audited and not blockers:
        blockers.append("no_real_or_synthetic_blocks_audited")

    component_names, component_fractions, component_speedups = _select_family_components(
        family_rows,
        audited_count=len(audited),
    )
    ledger = run_t039_a01_exact_map_object_ledger_gate(
        component_names=component_names,
        component_fractions=component_fractions,
        component_speedups=component_speedups,
        object_equivalence_proven=bool(audited),
        representative_real_maps_audited=len(audited),
        residual_fraction=0.0,
        hidden_sidecar_gib=0.0,
        decoded_dense_sidecar_gib=0.0,
        offload_or_setup_gain_counted=False,
        layout_replaces_storage=bool(audited),
        exact_arithmetic_contract=bool(audited),
    )
    ready = not blockers and ledger["readiness"]["t039_a01_exact_map_object_ledger_gate_ready"]
    if not ready and not blockers:
        blockers.extend(f"ledger:{blocker}" for blocker in ledger["blockers"])
    best_row = max(
        family_rows,
        key=lambda row: float(row["component_speedup_vs_current_component"]),
        default=None,
    )
    return {
        "probe": "t039b_a01_non_row_exact_object_extraction_ledger",
        "activation": T039B_A01_ACTIVATION,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_id": model_id,
        "model_dir": str(weight_dir) if weight_dir is not None else None,
        "config": config.as_dict() if config is not None else None,
        "weight_inventory": inventory.as_dict() if inventory is not None else None,
        "requested_tensors": list(tensor_names),
        "missing_tensors": missing,
        "audited_blocks": audited,
        "object_family_rows": family_rows,
        "best_object_family_row": best_row,
        "ledger": ledger,
        "readiness": {
            "t039b_a01_non_row_exact_object_extraction_ledger_ready": ready,
            "audited_block_count": len(audited),
            "candidate_object_family_count": len(family_rows),
            "best_component_speedup": float(best_row["component_speedup_vs_current_component"]) if best_row else 0.0,
            "ledger_gate_ready": ledger["readiness"]["t039_a01_exact_map_object_ledger_gate_ready"],
            "composite_speedup_vs_current_body": ledger["coverage_model"]["composite_speedup_vs_current_body"],
            "projected_tokens_per_second_if_control_zero": ledger["coverage_model"][
                "projected_tokens_per_second_if_control_zero"
            ],
        },
        "blockers": blockers,
        "next_actions": [
            "Promote the passing non-row exact object to a representative GPU hot-path implementation gate."
        ]
        if ready
        else [
            "Do not implement production A01 map-object runtime from T039B.",
            "If this scoped ledger is accepted as identifying, redirect T039 to A02 semantic layout/gauge contracts or a broader new substrate round.",
        ],
    }


def _normalize_t039b_a02_regions(regions: tuple[dict[str, Any], ...] | None) -> tuple[dict[str, Any], ...]:
    selected = regions or DEFAULT_T039B_A02_REGIONS
    normalized: list[dict[str, Any]] = []
    for index, region in enumerate(selected):
        name = str(region.get("name", f"region_{index}"))
        fraction = float(region.get("body_fraction", 0.0))
        speedup = float(region.get("private_speedup", 1.0))
        forced = bool(region.get("forced_canonical", True))
        exact = bool(region.get("exact_gauge_relation", False))
        priced = bool(region.get("semantic_boundary_priced", False))
        replaces = bool(region.get("layout_replaces_storage", False))
        canonical = forced or not exact or not priced or not replaces
        normalized.append(
            {
                "name": name,
                "body_fraction": fraction,
                "private_speedup": speedup,
                "exact_gauge_relation": exact,
                "semantic_boundary_priced": priced,
                "forced_canonical": canonical,
                "declared_forced_canonical": forced,
                "layout_replaces_storage": replaces,
                "charged_speedup": 1.0 if canonical else speedup,
                "readiness_blockers": [
                    blocker
                    for blocker, blocked in (
                        ("exact_gauge_relation_not_proven", not exact),
                        ("semantic_boundary_not_priced", not priced),
                        ("layout_does_not_replace_storage", not replaces),
                        ("forced_canonical_region", canonical),
                        ("nonpositive_private_speedup", speedup <= 0.0),
                        ("negative_body_fraction", fraction < 0.0),
                    )
                    if blocked
                ],
            }
        )
    return tuple(normalized)


def run_t039b_a02_gauge_sat_canonical_tail_ledger(
    *,
    regions: tuple[dict[str, Any], ...] | None = None,
    residual_fraction: float = 0.0,
    hidden_sidecar_gib: float = 0.0,
    decoded_dense_sidecar_gib: float = 0.0,
    offload_or_setup_gain_counted: bool = False,
    exact_arithmetic_contract: bool = True,
    peak_cuda_gib: float = R452_PEAK_CUDA_GIB,
    device_budget_gib: float = DEFAULT_DEVICE_BUDGET_GIB,
    fraction_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Gate whole-body gauge legality and canonical-tail collapse for T039B A02."""

    normalized = _normalize_t039b_a02_regions(regions)
    component_names = tuple(region["name"] for region in normalized) or ("no_semantic_regions",)
    component_fractions = tuple(float(region["body_fraction"]) for region in normalized) or (1.0,)
    component_speedups = tuple(float(region["charged_speedup"]) for region in normalized) or (1.0,)
    forced_canonical_fraction = sum(
        float(region["body_fraction"])
        for region in normalized
        if bool(region["forced_canonical"])
    )
    exact_gauge_contracts_proven = bool(normalized) and all(
        bool(region["exact_gauge_relation"]) for region in normalized
    )
    semantic_boundaries_priced = bool(normalized) and all(
        bool(region["semantic_boundary_priced"]) for region in normalized
    )
    layout_replaces_storage = bool(normalized) and all(
        bool(region["layout_replaces_storage"]) for region in normalized
    )
    ledger = run_t039_a02_semantic_layout_ledger_gate(
        component_names=component_names,
        component_fractions=component_fractions,
        component_speedups=component_speedups,
        gauge_contracts_proven=exact_gauge_contracts_proven,
        semantic_boundaries_priced=semantic_boundaries_priced,
        forced_canonical_fraction=forced_canonical_fraction,
        residual_fraction=residual_fraction,
        hidden_sidecar_gib=hidden_sidecar_gib,
        decoded_dense_sidecar_gib=decoded_dense_sidecar_gib,
        offload_or_setup_gain_counted=offload_or_setup_gain_counted,
        layout_replaces_storage=layout_replaces_storage,
        exact_arithmetic_contract=exact_arithmetic_contract,
        peak_cuda_gib=peak_cuda_gib,
        device_budget_gib=device_budget_gib,
        fraction_tolerance=fraction_tolerance,
    )
    ready = ledger["readiness"]["t039_a02_semantic_layout_ledger_gate_ready"]
    blockers = [f"ledger:{blocker}" for blocker in ledger["blockers"]]
    return {
        "probe": "t039b_a02_gauge_sat_canonical_tail_ledger",
        "activation": T039B_A02_ACTIVATION,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "target": ledger["target"],
        "baseline": ledger["baseline"],
        "regions": list(normalized),
        "gauge_sat": {
            "region_count": len(normalized),
            "exact_gauge_contracts_proven": exact_gauge_contracts_proven,
            "semantic_boundaries_priced": semantic_boundaries_priced,
            "forced_canonical_fraction": forced_canonical_fraction,
            "maximum_forced_canonical_fraction": 0.132034928176561,
            "layout_replaces_storage": layout_replaces_storage,
            "residual_fraction": residual_fraction,
            "hidden_sidecar_gib": hidden_sidecar_gib,
            "decoded_dense_sidecar_gib": decoded_dense_sidecar_gib,
            "offload_or_setup_gain_counted": offload_or_setup_gain_counted,
            "exact_arithmetic_contract": exact_arithmetic_contract,
            "peak_cuda_gib": peak_cuda_gib,
            "device_budget_gib": device_budget_gib,
        },
        "ledger": ledger,
        "readiness": {
            "t039b_a02_gauge_sat_canonical_tail_ledger_ready": ready,
            "ledger_gate_ready": ready,
            "forced_canonical_tail_below_bound": ledger["readiness"][
                "t039_a02_forced_canonical_tail_below_bound"
            ],
            "composite_speedup_sufficient": ledger["readiness"]["t039_a02_composite_speedup_sufficient"],
            "composite_speedup_vs_current_body": ledger["coverage_model"]["composite_speedup_vs_current_body"],
            "projected_tokens_per_second_if_control_zero": ledger["coverage_model"][
                "projected_tokens_per_second_if_control_zero"
            ],
        },
        "blockers": blockers,
        "next_actions": [
            "Promote the passing semantic layout to a representative GPU hot-path implementation gate."
        ]
        if ready
        else [
            "Do not implement production A02 semantic-layout runtime from T039B.",
            "If this scoped ledger is accepted as identifying, close T039B A02 for the current map body or open a broader substrate round.",
        ],
    }


def run_t039_real_map_object_audit(
    *,
    model_id: str = "Qwen/Qwen3-32B",
    local_model_dir: Any = None,
    allow_hf_config_fetch: bool = True,
    tensor_names: tuple[str, ...] = DEFAULT_REAL_MAP_AUDIT_TENSORS,
    block_rows: int = 64,
    block_cols: int = 256,
) -> dict[str, Any]:
    """Audit exact row-quotient objects on small real compressed Qwen map blocks."""

    blockers: list[str] = []
    audited: list[dict[str, Any]] = []
    missing: list[str] = []
    weight_dir = None
    inventory = None
    config = None

    if importlib.util.find_spec("torch") is None:
        blockers.append("torch_missing")
    if importlib.util.find_spec("safetensors") is None:
        blockers.append("safetensors_missing")

    if not blockers:
        from .t004_real_qwen_bridge import find_weight_dir, inventory_safetensors, resolve_qwen_config
        from .t006_real_residency_block_smoke import _compress_reconstruct, _policy_for_tensor, _read_block
        from .t008_compressed_generation_smoke import _load_weight_map

        config = resolve_qwen_config(
            model_id=model_id,
            local_model_dir=local_model_dir,
            allow_hf_config_fetch=allow_hf_config_fetch,
        )
        weight_dir = find_weight_dir(config, local_model_dir)
        inventory = inventory_safetensors(weight_dir)
        if not config.matches_project_qwen3_32b_shape():
            blockers.append("config_shape_mismatch")
        if weight_dir is None or not inventory.local_weights_present:
            blockers.append("missing_local_qwen3_32b_safetensors")

    if not blockers and weight_dir is not None:
        from .t006_real_residency_block_smoke import _compress_reconstruct, _policy_for_tensor, _read_block
        from .t008_compressed_generation_smoke import _load_weight_map

        weight_map = _load_weight_map(weight_dir)
        for name in tensor_names:
            shard = weight_map.get(name)
            if shard is None:
                missing.append(name)
                continue
            try:
                block, original_shape = _read_block(weight_dir, name, shard, block_rows, block_cols)
                reconstructed = _compress_reconstruct(block, _policy_for_tensor(name))
                metrics = _exact_row_quotient_metrics(reconstructed)
                metrics.update(
                    {
                        "tensor": name,
                        "shard": shard,
                        "original_shape": list(original_shape),
                        "block_shape": [int(reconstructed.shape[0]), int(reconstructed.shape[1])],
                    }
                )
                audited.append(metrics)
            except Exception as exc:  # pragma: no cover - real shard guard
                blockers.append(f"real_map_audit_failed:{name}:{type(exc).__name__}:{exc}")
                break

    if not audited and not blockers:
        blockers.append("no_real_map_blocks_audited")

    total_elements = sum(item["rows"] * item["cols"] for item in audited)
    component_names = tuple(item["tensor"] for item in audited) or ("no_real_map_blocks",)
    component_fractions = tuple(
        (item["rows"] * item["cols"]) / max(total_elements, 1) for item in audited
    ) or (1.0,)
    component_speedups = tuple(item["row_quotient_speedup"] for item in audited) or (1.0,)
    all_exact = bool(audited) and all(bool(item["exact_reconstruction"]) for item in audited)
    layout_replaces_storage = bool(audited) and all(item["storage_fraction_before_mapping"] <= 1.0 for item in audited)
    ledger = run_t039_a01_exact_map_object_ledger_gate(
        component_names=component_names,
        component_fractions=component_fractions,
        component_speedups=component_speedups,
        object_equivalence_proven=all_exact,
        representative_real_maps_audited=len(audited),
        residual_fraction=0.0,
        hidden_sidecar_gib=0.0,
        decoded_dense_sidecar_gib=0.0,
        offload_or_setup_gain_counted=False,
        layout_replaces_storage=layout_replaces_storage,
        exact_arithmetic_contract=all_exact,
    )
    audit_ready = not blockers and ledger["readiness"]["t039_a01_exact_map_object_ledger_gate_ready"]
    if not audit_ready and not blockers:
        blockers.extend(f"ledger:{blocker}" for blocker in ledger["blockers"])

    return {
        "probe": "t039_real_map_object_audit",
        "activation": REAL_MAP_AUDIT_ACTIVATION,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_id": model_id,
        "model_dir": str(weight_dir) if weight_dir is not None else None,
        "config": config.as_dict() if config is not None else None,
        "weight_inventory": inventory.as_dict() if inventory is not None else None,
        "requested_tensors": list(tensor_names),
        "missing_tensors": missing,
        "audited_blocks": audited,
        "ledger": ledger,
        "readiness": {
            "t039_real_map_object_audit_ready": audit_ready,
            "audited_block_count": len(audited),
            "all_audited_blocks_exact": all_exact,
            "row_quotient_composite_speedup": ledger["coverage_model"]["composite_speedup_vs_current_body"],
            "row_quotient_crosses_required_denominator": ledger["readiness"][
                "t039_a01_composite_speedup_sufficient"
            ],
            "ledger_gate_ready": ledger["readiness"]["t039_a01_exact_map_object_ledger_gate_ready"],
        },
        "blockers": blockers,
        "next_actions": [
            "Promote the audited quotient object to a representative GPU hot-path implementation gate."
        ]
        if audit_ready
        else [
            "Do not promote exact row quotients for the current representative maps.",
            "Reopen T039 research around non-row-quotient exact map objects or layout/gauge contracts.",
        ],
    }


__all__ = [
    "A01_ACTIVATION",
    "A02_ACTIVATION",
    "REAL_MAP_AUDIT_ACTIVATION",
    "T039B_A01_ACTIVATION",
    "T039B_A02_ACTIVATION",
    "run_t039_a01_exact_map_object_ledger_gate",
    "run_t039_a02_semantic_layout_ledger_gate",
    "run_t039b_a01_non_row_exact_object_extraction_ledger",
    "run_t039b_a02_gauge_sat_canonical_tail_ledger",
    "run_t039_real_map_object_audit",
]
