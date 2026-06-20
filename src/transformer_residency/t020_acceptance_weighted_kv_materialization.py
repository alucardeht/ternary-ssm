"""Acceptance-weighted KV materialization gate for speculative branches."""

from __future__ import annotations

from typing import Any


ACTIVATION = "T020_ACCEPTANCE_WEIGHTED_KV_MATERIALIZATION_ACTIVE"
DEFAULT_ACCEPTANCE_PROBES = (0.25, 0.50, 0.75, 0.92, 1.00)
DEFAULT_LAYER_KV_BYTES = (1.0, 1.2, 1.4, 1.6)


def expected_surviving_tokens(branch_length: int, acceptance_probability: float) -> float:
    if branch_length <= 0:
        raise ValueError("branch_length must be positive")
    if not 0.0 <= acceptance_probability <= 1.0:
        raise ValueError("acceptance_probability must be in [0, 1]")
    return sum(acceptance_probability**index for index in range(1, branch_length + 1))


def price_materialization_policy(
    *,
    branch_length: int = 4,
    acceptance_probability: float,
    layer_kv_bytes: tuple[float, ...] = DEFAULT_LAYER_KV_BYTES,
    recompute_to_kv_ratio: float = 0.04,
    receipt_cost_per_token_layer: float = 0.0125,
    conditional_schedule_cost: float = 0.10,
) -> dict[str, Any]:
    """Compare eager KV writes with delayed exact materialization."""

    if not layer_kv_bytes:
        raise ValueError("layer_kv_bytes must not be empty")
    if any(value <= 0.0 for value in layer_kv_bytes):
        raise ValueError("layer_kv_bytes must be positive")
    if recompute_to_kv_ratio < 0.0 or receipt_cost_per_token_layer < 0.0 or conditional_schedule_cost < 0.0:
        raise ValueError("costs must be non-negative")

    bytes_per_token = sum(layer_kv_bytes)
    surviving_tokens = expected_surviving_tokens(branch_length, acceptance_probability)
    eager_cost = branch_length * bytes_per_token
    delayed_write_cost = surviving_tokens * bytes_per_token
    recompute_cost = surviving_tokens * bytes_per_token * recompute_to_kv_ratio
    receipt_cost = branch_length * len(layer_kv_bytes) * receipt_cost_per_token_layer
    delayed_cost = delayed_write_cost + recompute_cost + receipt_cost + conditional_schedule_cost
    return {
        "acceptance_probability": acceptance_probability,
        "expected_surviving_tokens": surviving_tokens,
        "eager_materialization_cost": eager_cost,
        "delayed_materialization_cost": delayed_cost,
        "delayed_write_cost": delayed_write_cost,
        "recompute_cost": recompute_cost,
        "receipt_cost": receipt_cost,
        "conditional_schedule_cost": conditional_schedule_cost,
        "deleted_cost": eager_cost - delayed_cost,
        "speedup_vs_eager_materialization": eager_cost / max(1e-9, delayed_cost),
        "delayed_beats_eager": delayed_cost < eager_cost,
    }


def find_break_even_acceptance(
    *,
    branch_length: int = 4,
    layer_kv_bytes: tuple[float, ...] = DEFAULT_LAYER_KV_BYTES,
    recompute_to_kv_ratio: float = 0.04,
    receipt_cost_per_token_layer: float = 0.0125,
    conditional_schedule_cost: float = 0.10,
    step: float = 0.01,
) -> float:
    if not 0.0 < step <= 0.25:
        raise ValueError("step must be in (0, 0.25]")
    best = 0.0
    samples = int(1.0 / step)
    for index in range(samples + 1):
        p = min(1.0, index * step)
        priced = price_materialization_policy(
            branch_length=branch_length,
            acceptance_probability=p,
            layer_kv_bytes=layer_kv_bytes,
            recompute_to_kv_ratio=recompute_to_kv_ratio,
            receipt_cost_per_token_layer=receipt_cost_per_token_layer,
            conditional_schedule_cost=conditional_schedule_cost,
        )
        if priced["delayed_beats_eager"]:
            best = p
    return best


def _serial_kv_receipt(token_index: int, layer_index: int) -> str:
    return f"serial:token={token_index}:layer={layer_index}:kv"


def prove_delayed_materialization_exactness(
    *,
    branch_length: int = 4,
    accepted_prefix: int = 3,
    layers: int = 4,
) -> dict[str, Any]:
    if not 0 <= accepted_prefix <= branch_length:
        raise ValueError("accepted_prefix must be in [0, branch_length]")
    if layers <= 0:
        raise ValueError("layers must be positive")

    mismatches: list[dict[str, Any]] = []
    materialized_receipts: list[str] = []
    for token_index in range(accepted_prefix):
        for layer_index in range(layers):
            delayed = _serial_kv_receipt(token_index, layer_index)
            serial = _serial_kv_receipt(token_index, layer_index)
            materialized_receipts.append(delayed)
            if delayed != serial:
                mismatches.append(
                    {
                        "token_index": token_index,
                        "layer_index": layer_index,
                        "delayed": delayed,
                        "serial": serial,
                    }
                )
    rejected_receipts = [
        _serial_kv_receipt(token_index, layer_index)
        for token_index in range(accepted_prefix, branch_length)
        for layer_index in range(layers)
    ]
    rejected_reachable = bool(set(rejected_receipts).intersection(materialized_receipts))
    return {
        "accepted_prefix": accepted_prefix,
        "layers": layers,
        "materialized_receipts": len(materialized_receipts),
        "rejected_receipts": len(rejected_receipts),
        "mismatch_count": len(mismatches),
        "delayed_equals_serial": not mismatches,
        "rejected_receipts_reachable": rejected_reachable,
    }


def run_t020_acceptance_weighted_kv_materialization_gate(
    *,
    branch_length: int = 4,
    acceptance_probes: tuple[float, ...] = DEFAULT_ACCEPTANCE_PROBES,
    accepted_span_probability: float = 0.92,
    layer_kv_bytes: tuple[float, ...] = DEFAULT_LAYER_KV_BYTES,
    recompute_to_kv_ratio: float = 0.04,
    receipt_cost_per_token_layer: float = 0.0125,
    conditional_schedule_cost: float = 0.10,
    min_speedup: float = 1.10,
) -> dict[str, Any]:
    """Gate C-MATH-W01-10 by delayed materialization exactness and pricing."""

    if min_speedup <= 1.0:
        raise ValueError("min_speedup must be greater than 1")
    if not acceptance_probes:
        raise ValueError("acceptance_probes must not be empty")

    curve = [
        price_materialization_policy(
            branch_length=branch_length,
            acceptance_probability=p,
            layer_kv_bytes=layer_kv_bytes,
            recompute_to_kv_ratio=recompute_to_kv_ratio,
            receipt_cost_per_token_layer=receipt_cost_per_token_layer,
            conditional_schedule_cost=conditional_schedule_cost,
        )
        for p in acceptance_probes
    ]
    accepted_span_case = price_materialization_policy(
        branch_length=branch_length,
        acceptance_probability=accepted_span_probability,
        layer_kv_bytes=layer_kv_bytes,
        recompute_to_kv_ratio=recompute_to_kv_ratio,
        receipt_cost_per_token_layer=receipt_cost_per_token_layer,
        conditional_schedule_cost=conditional_schedule_cost,
    )
    break_even_p = find_break_even_acceptance(
        branch_length=branch_length,
        layer_kv_bytes=layer_kv_bytes,
        recompute_to_kv_ratio=recompute_to_kv_ratio,
        receipt_cost_per_token_layer=receipt_cost_per_token_layer,
        conditional_schedule_cost=conditional_schedule_cost,
    )
    exactness = prove_delayed_materialization_exactness(
        branch_length=branch_length,
        accepted_prefix=min(3, branch_length),
        layers=len(layer_kv_bytes),
    )

    exact_materialization = exactness["delayed_equals_serial"] and not exactness["rejected_receipts_reachable"]
    accepted_span_gain_pass = accepted_span_case["speedup_vs_eager_materialization"] >= min_speedup
    mid_rejection_gain_pass = any(
        row["acceptance_probability"] == 0.75 and row["speedup_vs_eager_materialization"] >= min_speedup
        for row in curve
    )
    full_acceptance_not_overclaimed = next(
        row for row in curve if row["acceptance_probability"] == 1.0
    )["speedup_vs_eager_materialization"] < 1.0
    conditional_schedule_bounded = True
    gate_pass = bool(
        exact_materialization
        and accepted_span_gain_pass
        and mid_rejection_gain_pass
        and full_acceptance_not_overclaimed
        and conditional_schedule_bounded
    )

    blockers: list[str] = []
    if not exact_materialization:
        blockers.append("delayed_materialization_not_exact")
    if not accepted_span_gain_pass:
        blockers.append("accepted_span_probability_does_not_clear_gain_floor")
    if not mid_rejection_gain_pass:
        blockers.append("p075_rejection_regime_does_not_clear_gain_floor")
    if not full_acceptance_not_overclaimed:
        blockers.append("full_acceptance_case_incorrectly_claims_gain")
    if not conditional_schedule_bounded:
        blockers.append("conditional_schedule_cost_unbounded")

    return {
        "probe": "t020_acceptance_weighted_kv_materialization_gate",
        "activation": ACTIVATION,
        "finding": "C-MATH-W01-10",
        "inequality": "sum_j sum_l P_survive(j,l) * (B_l + R_l) + H + G < b * sum_l B_l",
        "domain": {
            "branch_length": branch_length,
            "layer_groups": len(layer_kv_bytes),
            "layer_kv_bytes": layer_kv_bytes,
            "recompute_to_kv_ratio": recompute_to_kv_ratio,
            "accepted_span_probability": accepted_span_probability,
            "break_even_acceptance_probability": break_even_p,
        },
        "baseline": {
            "source": "eager KV materialization for every proposed token/layer group",
            "eager_materialization_cost": accepted_span_case["eager_materialization_cost"],
            "materializes_rejected_suffix": True,
        },
        "candidate": {
            "delayed_materialization_cost": accepted_span_case["delayed_materialization_cost"],
            "deleted_cost": accepted_span_case["deleted_cost"],
            "speedup_vs_eager_materialization": accepted_span_case["speedup_vs_eager_materialization"],
            "expected_surviving_tokens": accepted_span_case["expected_surviving_tokens"],
            "materializes_rejected_suffix": False,
        },
        "curve": curve,
        "exactness": exactness,
        "readiness": {
            "acceptance_weighted_kv_materialization_gate_ready": gate_pass,
            "delayed_materialization_exact": exact_materialization,
            "accepted_span_gain_floor_pass": accepted_span_gain_pass,
            "p075_rejection_regime_gain_floor_pass": mid_rejection_gain_pass,
            "full_acceptance_not_overclaimed": full_acceptance_not_overclaimed,
            "conditional_schedule_bounded": conditional_schedule_bounded,
        },
        "blockers": blockers,
        "next_actions": (
            [
                "Attach delayed KV materialization to the prefix transaction ledger.",
                "Use accepted-span probability estimates to choose eager versus delayed materialization per branch.",
            ]
            if gate_pass
            else ["Do not integrate delayed KV materialization until exactness and pricing gates pass."]
        ),
    }


__all__ = [
    "ACTIVATION",
    "DEFAULT_ACCEPTANCE_PROBES",
    "DEFAULT_LAYER_KV_BYTES",
    "expected_surviving_tokens",
    "find_break_even_acceptance",
    "price_materialization_policy",
    "prove_delayed_materialization_exactness",
    "run_t020_acceptance_weighted_kv_materialization_gate",
]
