"""Recompute-on-reject inequality gate for speculative KV branches."""

from __future__ import annotations

from typing import Any


ACTIVATION = "T020_RECOMPUTE_ON_REJECT_ACTIVE"


def branch_acceptance_probability(base_acceptance: float, depth: int) -> float:
    if not 0.0 < base_acceptance < 1.0:
        raise ValueError("base_acceptance must be between 0 and 1")
    if depth <= 0:
        raise ValueError("depth must be positive")
    return base_acceptance**depth


def price_recompute_policy(
    *,
    branch_depth: int = 4,
    branch_width: int = 4,
    base_acceptance: float = 0.60,
    kv_write_cost: float = 1.0,
    memory_pressure_cost: float = 0.25,
    depth_memory_pressure_step: float = 0.05,
    recompute_cost: float = 2.20,
    cleanup_cost: float = 0.20,
    receipt_cost: float = 0.05,
) -> dict[str, Any]:
    """Price store-versus-defer decisions across a synthetic branch tree."""

    if branch_depth <= 0 or branch_width <= 0:
        raise ValueError("branch_depth and branch_width must be positive")
    if kv_write_cost < 0.0 or memory_pressure_cost < 0.0 or recompute_cost < 0.0:
        raise ValueError("costs must be non-negative")

    rows: list[dict[str, Any]] = []
    baseline_cost = 0.0
    candidate_cost = 0.0
    stored_positions = 0
    deferred_positions = 0
    for depth in range(1, branch_depth + 1):
        p_accept = branch_acceptance_probability(base_acceptance, depth)
        write_cost = kv_write_cost
        memory_cost = memory_pressure_cost + depth_memory_pressure_step * depth
        cleanup_expected = (1.0 - p_accept) * cleanup_cost
        eager_cost = write_cost + memory_cost + cleanup_expected
        defer_cost = p_accept * recompute_cost + receipt_cost
        store = p_accept * recompute_cost > write_cost + memory_cost
        chosen_cost = eager_cost if store else defer_cost
        count = branch_width
        baseline_cost += count * eager_cost
        candidate_cost += count * chosen_cost
        if store:
            stored_positions += count
        else:
            deferred_positions += count
        rows.append(
            {
                "depth": depth,
                "positions": count,
                "acceptance_probability": p_accept,
                "write_cost": write_cost,
                "memory_pressure_cost": memory_cost,
                "cleanup_expected_cost": cleanup_expected,
                "recompute_if_accepted_cost": p_accept * recompute_cost,
                "receipt_cost": receipt_cost,
                "eager_cost": eager_cost,
                "defer_cost": defer_cost,
                "store_provisional_kv": store,
                "chosen_cost": chosen_cost,
                "gain_per_position": eager_cost - chosen_cost,
            }
        )

    total_positions = branch_depth * branch_width
    return {
        "rows": rows,
        "baseline_eager_cost": baseline_cost,
        "candidate_policy_cost": candidate_cost,
        "stored_positions": stored_positions,
        "deferred_positions": deferred_positions,
        "total_positions": total_positions,
        "stored_fraction": stored_positions / total_positions,
        "deleted_cost": baseline_cost - candidate_cost,
        "speedup_vs_eager": baseline_cost / max(1e-9, candidate_cost),
    }


def run_t020_recompute_on_reject_gate(
    *,
    branch_depth: int = 4,
    branch_width: int = 4,
    base_acceptance: float = 0.60,
    kv_write_cost: float = 1.0,
    memory_pressure_cost: float = 0.25,
    depth_memory_pressure_step: float = 0.05,
    recompute_cost: float = 2.20,
    cleanup_cost: float = 0.20,
    receipt_cost: float = 0.05,
    max_store_fraction: float = 0.80,
    min_speedup: float = 1.10,
) -> dict[str, Any]:
    """Gate C-MATH-W02-02 by the store/defer recompute inequality."""

    if not 0.0 < max_store_fraction <= 1.0:
        raise ValueError("max_store_fraction must be in (0, 1]")
    if min_speedup <= 1.0:
        raise ValueError("min_speedup must be greater than 1")

    policy = price_recompute_policy(
        branch_depth=branch_depth,
        branch_width=branch_width,
        base_acceptance=base_acceptance,
        kv_write_cost=kv_write_cost,
        memory_pressure_cost=memory_pressure_cost,
        depth_memory_pressure_step=depth_memory_pressure_step,
        recompute_cost=recompute_cost,
        cleanup_cost=cleanup_cost,
        receipt_cost=receipt_cost,
    )
    exact_recompute_receipt = True
    store_fraction_pass = policy["stored_fraction"] <= max_store_fraction
    positive_gain = policy["deleted_cost"] > 0.0
    speedup_pass = policy["speedup_vs_eager"] >= min_speedup
    gate_pass = bool(exact_recompute_receipt and store_fraction_pass and positive_gain and speedup_pass)

    blockers: list[str] = []
    if not exact_recompute_receipt:
        blockers.append("missing_deterministic_recompute_receipt")
    if not store_fraction_pass:
        blockers.append("optimal_policy_stores_more_than_80_percent_of_branch_positions")
    if not positive_gain:
        blockers.append("recompute_policy_does_not_delete_expected_rejected_branch_cost")
    if not speedup_pass:
        blockers.append("recompute_policy_below_10_percent_gain_floor")

    return {
        "probe": "t020_recompute_on_reject_gate",
        "activation": ACTIVATION,
        "finding": "C-MATH-W02-02",
        "inequality": "store provisional KV iff p_i * R_i > W_i + M_i",
        "baseline": {
            "source": "eager provisional KV writes plus rejected-branch cleanup",
            "eager_cost": policy["baseline_eager_cost"],
            "stores_all_positions": True,
        },
        "candidate": {
            "policy_cost": policy["candidate_policy_cost"],
            "deleted_cost": policy["deleted_cost"],
            "speedup_vs_eager": policy["speedup_vs_eager"],
            "stored_fraction": policy["stored_fraction"],
            "stored_positions": policy["stored_positions"],
            "deferred_positions": policy["deferred_positions"],
            "total_positions": policy["total_positions"],
            "requires_exact_recompute_receipt": exact_recompute_receipt,
        },
        "branch_tree": {
            "branch_depth": branch_depth,
            "branch_width": branch_width,
            "base_acceptance": base_acceptance,
            "rows": policy["rows"],
        },
        "readiness": {
            "recompute_on_reject_gate_ready": gate_pass,
            "store_fraction_below_80_percent": store_fraction_pass,
            "deleted_cost_positive": positive_gain,
            "speedup_floor_pass": speedup_pass,
            "exact_recompute_receipt_required": exact_recompute_receipt,
        },
        "blockers": blockers,
        "next_actions": (
            [
                "Represent deferred KV positions with deterministic recompute receipts.",
                "Attach the store/defer policy to the speculative decode branch-state ledger.",
            ]
            if gate_pass
            else ["Do not integrate recompute-on-reject until branch policy beats eager KV writes."]
        ),
    }


__all__ = [
    "ACTIVATION",
    "branch_acceptance_probability",
    "price_recompute_policy",
    "run_t020_recompute_on_reject_gate",
]
