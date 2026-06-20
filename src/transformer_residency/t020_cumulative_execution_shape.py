"""Cumulative T020 execution-shape gate."""

from __future__ import annotations

from math import prod

from .t020_acceptance_weighted_kv_materialization import run_t020_acceptance_weighted_kv_materialization_gate
from .t020_affine_packed_view import run_t020_affine_packed_view_gate
from .t020_basis_shared_kv_transform import run_t020_basis_shared_kv_transform_gate
from .t020_conversion_time_arena_compiler import run_t020_conversion_time_arena_compiler_gate
from .t020_device_credit_flow import run_t020_device_credit_flow_gate
from .t020_expected_accepted_span import run_t020_expected_accepted_span_gate
from .t020_execution_order_slab import run_t020_execution_order_slab_gate
from .t020_immutable_arena_receipt import run_t020_immutable_arena_receipt_gate
from .t020_packed_work_coalescing import run_t020_packed_work_coalescing_gate
from .t020_prefix_transaction_ledger import run_t020_prefix_transaction_ledger_gate
from .t020_recompute_on_reject import run_t020_recompute_on_reject_gate
from .t020_ring_exact_tail_receipts import run_t020_ring_exact_tail_receipts_gate


ACTIVATION = "T020_CUMULATIVE_EXECUTION_SHAPE_ACTIVE"


def _component_ready(result: dict, readiness_key: str) -> bool:
    return bool(result["readiness"].get(readiness_key)) and not result.get("blockers")


def _numeric_item(
    *,
    finding: str,
    label: str,
    scope: str,
    before: float,
    after: float,
    speedup: float,
    activation: str,
) -> dict:
    return {
        "finding": finding,
        "label": label,
        "scope": scope,
        "activation": activation,
        "before": before,
        "after": after,
        "speedup": speedup,
        "composes_in_stack": True,
    }


def run_t020_cumulative_execution_shape_gate(
    *,
    min_individual_speedup: float = 1.10,
    min_cumulative_speedup: float = 2.0,
) -> dict:
    """Run the implemented T020 findings as a cumulative declared stack."""

    if min_individual_speedup <= 1.0:
        raise ValueError("min_individual_speedup must be greater than 1")
    if min_cumulative_speedup <= 1.0:
        raise ValueError("min_cumulative_speedup must be greater than 1")

    device = run_t020_device_credit_flow_gate(min_speedup=min_individual_speedup)
    coalescing = run_t020_packed_work_coalescing_gate()
    slab = run_t020_execution_order_slab_gate()
    receipt = run_t020_immutable_arena_receipt_gate(min_speedup=min_individual_speedup)
    compiler = run_t020_conversion_time_arena_compiler_gate()
    affine = run_t020_affine_packed_view_gate()
    recompute = run_t020_recompute_on_reject_gate(min_speedup=min_individual_speedup)
    span = run_t020_expected_accepted_span_gate(min_speedup=min_individual_speedup)
    prefix = run_t020_prefix_transaction_ledger_gate(min_speedup=min_individual_speedup)
    materialization = run_t020_acceptance_weighted_kv_materialization_gate(min_speedup=min_individual_speedup)
    basis = run_t020_basis_shared_kv_transform_gate(min_speedup=min_individual_speedup)
    ring = run_t020_ring_exact_tail_receipts_gate(min_speedup=min_individual_speedup)

    component_results = {
        "device_credit_flow": device,
        "packed_work_coalescing": coalescing,
        "execution_order_slab": slab,
        "immutable_arena_receipt": receipt,
        "conversion_time_arena_compiler": compiler,
        "affine_packed_view": affine,
        "recompute_on_reject": recompute,
        "expected_accepted_span": span,
        "prefix_transaction_ledger": prefix,
        "acceptance_weighted_kv_materialization": materialization,
        "basis_shared_kv_transform": basis,
        "ring_exact_tail_receipts": ring,
    }
    readiness_keys = {
        "device_credit_flow": "device_credit_flow_gate_ready",
        "packed_work_coalescing": "packed_work_coalescing_gate_ready",
        "execution_order_slab": "execution_order_slab_gate_ready",
        "immutable_arena_receipt": "immutable_arena_receipt_gate_ready",
        "conversion_time_arena_compiler": "conversion_time_arena_compiler_gate_ready",
        "affine_packed_view": "affine_packed_view_gate_ready",
        "recompute_on_reject": "recompute_on_reject_gate_ready",
        "expected_accepted_span": "expected_accepted_span_gate_ready",
        "prefix_transaction_ledger": "prefix_transaction_ledger_gate_ready",
        "acceptance_weighted_kv_materialization": "acceptance_weighted_kv_materialization_gate_ready",
        "basis_shared_kv_transform": "basis_shared_kv_transform_gate_ready",
        "ring_exact_tail_receipts": "ring_exact_tail_receipts_gate_ready",
    }

    numeric_items = [
        _numeric_item(
            finding="B-MATH-W02-09",
            label="device_credit_flow",
            scope="hot_decode_total_seconds",
            before=device["baseline"]["total_seconds"],
            after=device["candidate"]["total_seconds"],
            speedup=device["candidate"]["speedup_vs_baseline"],
            activation=device["activation"],
        ),
        _numeric_item(
            finding="A-MATH-W01-01",
            label="execution_order_slab",
            scope="layer1_arena_acquisition",
            before=slab["baseline"]["acquisition_seconds"],
            after=slab["candidate"]["acquisition_seconds"],
            speedup=slab["candidate"]["acquisition_speedup"],
            activation=slab["activation"],
        ),
        _numeric_item(
            finding="A-MATH-W01-15",
            label="immutable_arena_receipt",
            scope="layer1_arena_identity",
            before=receipt["baseline"]["dynamic_discovery_seconds"],
            after=receipt["candidate"]["dynamic_discovery_seconds"],
            speedup=receipt["candidate"]["speedup_vs_dynamic_discovery"],
            activation=receipt["activation"],
        ),
        _numeric_item(
            finding="A-PB-W02-14",
            label="conversion_time_arena_compiler",
            scope="runtime_descriptor_interpretation",
            before=compiler["baseline"]["visible_denominator_seconds"],
            after=compiler["baseline"]["visible_denominator_seconds"]
            - compiler["candidate"]["deleted_denominator_seconds"],
            speedup=compiler["candidate"]["speedup_vs_visible_denominator"],
            activation=compiler["activation"],
        ),
        _numeric_item(
            finding="A-MATH-W02-03",
            label="affine_packed_view",
            scope="packed_payload_materialization",
            before=affine["baseline"]["visible_denominator_seconds"],
            after=affine["baseline"]["visible_denominator_seconds"] - affine["candidate"]["net_deleted_seconds"],
            speedup=affine["candidate"]["speedup_vs_visible_denominator"],
            activation=affine["activation"],
        ),
        _numeric_item(
            finding="C-MATH-W02-02",
            label="recompute_on_reject",
            scope="speculative_rejected_branch_kv",
            before=recompute["baseline"]["eager_cost"],
            after=recompute["candidate"]["policy_cost"],
            speedup=recompute["candidate"]["speedup_vs_eager"],
            activation=recompute["activation"],
        ),
        _numeric_item(
            finding="C-MATH-W01-02",
            label="expected_accepted_span",
            scope="accepted_token_denominator",
            before=span["baseline"]["cost_per_accepted_token"],
            after=span["candidate"]["cost_per_accepted_token"],
            speedup=span["candidate"]["speedup_vs_one_token"],
            activation=span["activation"],
        ),
        _numeric_item(
            finding="C-MATH-W01-01",
            label="prefix_transaction_ledger",
            scope="speculative_prefix_visibility",
            before=prefix["baseline"]["average_eager_mutation_cost"],
            after=prefix["candidate"]["average_transaction_ledger_cost"],
            speedup=prefix["candidate"]["speedup_vs_eager_mutation"],
            activation=prefix["activation"],
        ),
        _numeric_item(
            finding="C-MATH-W01-10",
            label="acceptance_weighted_kv_materialization",
            scope="accepted_span_kv_materialization",
            before=materialization["baseline"]["eager_materialization_cost"],
            after=materialization["candidate"]["delayed_materialization_cost"],
            speedup=materialization["candidate"]["speedup_vs_eager_materialization"],
            activation=materialization["activation"],
        ),
        _numeric_item(
            finding="C-MATH-W01-05",
            label="basis_shared_kv_transform",
            scope="representation_crossing",
            before=basis["baseline"]["representation_crossing_cost"],
            after=basis["candidate"]["basis_shared_cost"],
            speedup=basis["candidate"]["speedup_vs_representation_crossing"],
            activation=basis["activation"],
        ),
        _numeric_item(
            finding="C-MATH-W01-11",
            label="ring_exact_tail_receipts",
            scope="long_context_active_kv_policy",
            before=ring["baseline"]["baseline_active_cost"],
            after=ring["candidate"]["candidate_ring_tail_cost"],
            speedup=ring["candidate"]["speedup_vs_full_active_context"],
            activation=ring["activation"],
        ),
    ]

    structural_items = [
        {
            "finding": "B-PB-W02-11",
            "label": "packed_work_coalescing",
            "scope": "packed_kernel_residual_destinations",
            "activation": coalescing["activation"],
            "deleted_dense_mib": coalescing["traffic"]["deleted_dense_mib"],
            "fused_residual_adds": coalescing["traffic"]["fused_residual_adds"],
            "composes_in_stack": True,
        }
    ]

    cumulative_speedup = prod(item["speedup"] for item in numeric_items)
    baseline_tokens_per_second = device["baseline"]["tokens_per_second"]
    candidate_tokens_per_second = baseline_tokens_per_second * cumulative_speedup
    component_ready = {
        name: _component_ready(result, readiness_keys[name]) for name, result in component_results.items()
    }
    individual_gain_pass = all(item["speedup"] >= min_individual_speedup for item in numeric_items)
    cumulative_gain_pass = cumulative_speedup >= min_cumulative_speedup
    structural_gain_pass = coalescing["traffic"]["deleted_dense_mib"] > 0.0

    blockers: list[str] = []
    if not all(component_ready.values()):
        blockers.append("one_or_more_component_gates_not_ready")
    if not individual_gain_pass:
        blockers.append("one_or_more_numeric_gains_below_floor")
    if not cumulative_gain_pass:
        blockers.append("cumulative_speedup_below_floor")
    if not structural_gain_pass:
        blockers.append("packed_work_coalescing_deleted_no_dense_traffic")

    return {
        "probe": "t020_cumulative_execution_shape_gate",
        "activation": ACTIVATION,
        "finding": "T020-CUMULATIVE",
        "composition": {
            "policy": "multiply_scope_declared_speedups_and_keep_structural_deletions",
            "numeric_item_count": len(numeric_items),
            "structural_item_count": len(structural_items),
            "baseline_tokens_per_second": baseline_tokens_per_second,
            "candidate_tokens_per_second_proxy": candidate_tokens_per_second,
            "cumulative_speedup": cumulative_speedup,
            "min_individual_speedup": min_individual_speedup,
            "min_cumulative_speedup": min_cumulative_speedup,
        },
        "numeric_items": numeric_items,
        "structural_items": structural_items,
        "component_ready": component_ready,
        "readiness": {
            "cumulative_execution_shape_gate_ready": not blockers,
            "all_component_gates_ready": all(component_ready.values()),
            "all_numeric_gains_declared": individual_gain_pass,
            "coalescing_traffic_deleted": structural_gain_pass,
            "cumulative_speedup_floor_pass": cumulative_gain_pass,
            "declared_stack_is_cumulative": True,
        },
        "blockers": blockers,
        "next_actions": [
            "Wire the cumulative execution-shape contracts into one production decoder path instead of invoking the gates separately.",
            "Replace proxy composition with measured before/after tok/s once the cumulative decoder path is executable.",
        ],
    }


__all__ = ["ACTIVATION", "run_t020_cumulative_execution_shape_gate"]
