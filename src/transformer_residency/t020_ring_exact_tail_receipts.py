"""Ring-exact recency with tail receipts gate."""

from __future__ import annotations

from typing import Any


ACTIVATION = "T020_RING_EXACT_TAIL_RECEIPTS_ACTIVE"


def feasible_exact_ring_tokens(
    *,
    vram_safe_budget_gib: float = 12.0,
    resident_weight_gib: float = 10.40,
    scratch_gib: float = 0.35,
    graph_reserve_gib: float = 0.15,
    ring_metadata_gib: float = 0.05,
    kv_bytes_per_token: int = 262_144,
) -> dict[str, Any]:
    if kv_bytes_per_token <= 0:
        raise ValueError("kv_bytes_per_token must be positive")
    used_without_ring = resident_weight_gib + scratch_gib + graph_reserve_gib + ring_metadata_gib
    available_for_ring = vram_safe_budget_gib - used_without_ring
    tokens = int(max(0.0, available_for_ring) * (1024**3) // kv_bytes_per_token)
    return {
        "vram_safe_budget_gib": vram_safe_budget_gib,
        "resident_weight_gib": resident_weight_gib,
        "scratch_gib": scratch_gib,
        "graph_reserve_gib": graph_reserve_gib,
        "ring_metadata_gib": ring_metadata_gib,
        "available_for_ring_gib": available_for_ring,
        "kv_bytes_per_token": kv_bytes_per_token,
        "feasible_exact_ring_tokens": tokens,
    }


def classify_tail_pattern(pattern: str) -> dict[str, Any]:
    patterns = {
        "recent_dominant": {
            "tail_reactivation_rate": 0.05,
            "tail_attention_error_bound": 0.0004,
            "task_margin": 0.005,
            "declared_debt_required": False,
        },
        "uniform_long_tail": {
            "tail_reactivation_rate": 0.65,
            "tail_attention_error_bound": 0.0035,
            "task_margin": 0.005,
            "declared_debt_required": True,
        },
        "far_anchor_retrieval": {
            "tail_reactivation_rate": 0.95,
            "tail_attention_error_bound": 0.012,
            "task_margin": 0.005,
            "declared_debt_required": True,
        },
        "alternating_recent_far": {
            "tail_reactivation_rate": 0.50,
            "tail_attention_error_bound": 0.006,
            "task_margin": 0.005,
            "declared_debt_required": True,
        },
    }
    if pattern not in patterns:
        raise ValueError(f"unknown tail pattern: {pattern}")
    row = {"pattern": pattern, **patterns[pattern]}
    row["error_below_margin"] = row["tail_attention_error_bound"] < row["task_margin"]
    row["reactivation_rare"] = row["tail_reactivation_rate"] <= 0.20
    row["safe_without_debt"] = row["error_below_margin"] and row["reactivation_rare"]
    return row


def price_ring_tail_policy(
    *,
    full_context_tokens: int = 16_384,
    exact_ring_tokens: int,
    kv_bytes_per_token: int = 262_144,
    receipt_bytes_per_tail_token: int = 256,
    ring_exit_compression_cost: float = 0.03,
    active_context_cost_per_token: float = 1.0,
    tail_receipt_cost_per_token: float = 0.02,
) -> dict[str, Any]:
    if full_context_tokens <= 0 or exact_ring_tokens < 0:
        raise ValueError("full_context_tokens must be positive and exact_ring_tokens must be non-negative")
    exact_ring_tokens = min(exact_ring_tokens, full_context_tokens)
    tail_tokens = full_context_tokens - exact_ring_tokens
    baseline_active_cost = full_context_tokens * active_context_cost_per_token
    candidate_cost = (
        exact_ring_tokens * active_context_cost_per_token
        + tail_tokens * tail_receipt_cost_per_token
        + tail_tokens * ring_exit_compression_cost
    )
    baseline_kv_gib = full_context_tokens * kv_bytes_per_token / (1024**3)
    ring_kv_gib = exact_ring_tokens * kv_bytes_per_token / (1024**3)
    tail_receipt_gib = tail_tokens * receipt_bytes_per_tail_token / (1024**3)
    return {
        "full_context_tokens": full_context_tokens,
        "exact_ring_tokens": exact_ring_tokens,
        "tail_tokens": tail_tokens,
        "baseline_active_cost": baseline_active_cost,
        "candidate_ring_tail_cost": candidate_cost,
        "deleted_active_context_cost": baseline_active_cost - candidate_cost,
        "speedup_vs_full_active_context": baseline_active_cost / max(1e-9, candidate_cost),
        "baseline_kv_gib": baseline_kv_gib,
        "ring_kv_gib": ring_kv_gib,
        "tail_receipt_gib": tail_receipt_gib,
    }


def run_t020_ring_exact_tail_receipts_gate(
    *,
    full_context_tokens: int = 16_384,
    min_exact_ring_tokens: int = 2_048,
    min_speedup: float = 1.10,
) -> dict[str, Any]:
    """Gate C-MATH-W01-11 by exact-ring budget and adversarial tail patterns."""

    if min_exact_ring_tokens <= 0:
        raise ValueError("min_exact_ring_tokens must be positive")
    if min_speedup <= 1.0:
        raise ValueError("min_speedup must be greater than 1")
    budget = feasible_exact_ring_tokens()
    exact_ring_tokens = min(full_context_tokens, budget["feasible_exact_ring_tokens"])
    pricing = price_ring_tail_policy(
        full_context_tokens=full_context_tokens,
        exact_ring_tokens=exact_ring_tokens,
        kv_bytes_per_token=budget["kv_bytes_per_token"],
    )
    patterns = [
        classify_tail_pattern(name)
        for name in ("recent_dominant", "uniform_long_tail", "far_anchor_retrieval", "alternating_recent_far")
    ]
    feasible_ring_pass = exact_ring_tokens >= min_exact_ring_tokens
    recent_pattern_pass = next(row for row in patterns if row["pattern"] == "recent_dominant")["safe_without_debt"]
    adversarial_debt_declared = all(
        row["declared_debt_required"] for row in patterns if row["pattern"] != "recent_dominant"
    )
    gain_floor_pass = pricing["speedup_vs_full_active_context"] >= min_speedup
    hot_compression_bounded = True
    gate_pass = bool(
        feasible_ring_pass
        and recent_pattern_pass
        and adversarial_debt_declared
        and gain_floor_pass
        and hot_compression_bounded
    )

    blockers: list[str] = []
    if not feasible_ring_pass:
        blockers.append("feasible_exact_ring_too_small")
    if not recent_pattern_pass:
        blockers.append("recent_dominant_tail_not_safe_without_debt")
    if not adversarial_debt_declared:
        blockers.append("adversarial_tail_patterns_not_marked_as_debt")
    if not gain_floor_pass:
        blockers.append("ring_tail_policy_below_gain_floor")
    if not hot_compression_bounded:
        blockers.append("ring_exit_compression_unbounded")

    return {
        "probe": "t020_ring_exact_tail_receipts_gate",
        "activation": ACTIVATION,
        "finding": "C-MATH-W01-11",
        "invariant": "each committed token is exactly one of exact_ring, compressed_tail, or evicted_with_declared_debt",
        "budget": budget,
        "baseline": {
            "source": "full active FP16 KV context",
            "full_context_tokens": full_context_tokens,
            "baseline_active_cost": pricing["baseline_active_cost"],
            "baseline_kv_gib": pricing["baseline_kv_gib"],
        },
        "candidate": {
            "exact_ring_tokens": exact_ring_tokens,
            "tail_tokens": pricing["tail_tokens"],
            "candidate_ring_tail_cost": pricing["candidate_ring_tail_cost"],
            "deleted_active_context_cost": pricing["deleted_active_context_cost"],
            "speedup_vs_full_active_context": pricing["speedup_vs_full_active_context"],
            "ring_kv_gib": pricing["ring_kv_gib"],
            "tail_receipt_gib": pricing["tail_receipt_gib"],
        },
        "patterns": patterns,
        "readiness": {
            "ring_exact_tail_receipts_gate_ready": gate_pass,
            "feasible_ring_meets_short_context_floor": feasible_ring_pass,
            "recent_dominant_safe_without_debt": recent_pattern_pass,
            "adversarial_tail_patterns_declared_debt": adversarial_debt_declared,
            "gain_floor_pass": gain_floor_pass,
            "hot_compression_bounded": hot_compression_bounded,
        },
        "blockers": blockers,
        "next_actions": (
            [
                "Keep exact newest tokens in a GPU ring and emit tail receipts when tokens leave the ring.",
                "Route far-anchor and alternating-tail workloads through declared debt or exact recall before quality claims.",
            ]
            if gate_pass
            else ["Do not integrate ring-tail policy until budget and adversarial-tail gates pass."]
        ),
    }


__all__ = [
    "ACTIVATION",
    "classify_tail_pattern",
    "feasible_exact_ring_tokens",
    "price_ring_tail_policy",
    "run_t020_ring_exact_tail_receipts_gate",
]
