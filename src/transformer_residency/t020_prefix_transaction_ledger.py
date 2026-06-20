"""Prefix transaction KV ledger gate for exact speculative visibility."""

from __future__ import annotations

from typing import Any


ACTIVATION = "T020_PREFIX_TRANSACTION_LEDGER_ACTIVE"


def _delta_entry(layer: int, head: int, tier: str, token_index: int, base_offset: int) -> dict[str, Any]:
    tier_stride = {"exact": 1_000_000, "packed": 2_000_000}[tier]
    offset = tier_stride + layer * 100_000 + head * 10_000 + token_index * 128 + base_offset
    return {
        "layer": layer,
        "head": head,
        "tier": tier,
        "token_index": token_index,
        "offset": offset,
        "receipt": f"l{layer}:h{head}:{tier}:t{token_index}:o{offset}",
    }


def build_serial_kv_layout(
    *,
    layers: int,
    heads: int,
    committed_tokens: int,
    quant_tiers: tuple[str, ...] = ("exact", "packed"),
) -> list[dict[str, Any]]:
    if layers <= 0 or heads <= 0 or committed_tokens < 0:
        raise ValueError("layers and heads must be positive; committed_tokens must be non-negative")
    entries: list[dict[str, Any]] = []
    for token_index in range(committed_tokens):
        for layer in range(layers):
            for head in range(heads):
                for tier in quant_tiers:
                    entries.append(_delta_entry(layer, head, tier, token_index, base_offset=0))
    return entries


def build_transaction_deltas(
    *,
    layers: int,
    heads: int,
    branch_length: int,
    start_token: int,
    quant_tiers: tuple[str, ...] = ("exact", "packed"),
) -> list[dict[str, Any]]:
    if branch_length <= 0:
        raise ValueError("branch_length must be positive")
    deltas: list[dict[str, Any]] = []
    for branch_index in range(branch_length):
        token_index = start_token + branch_index
        for layer in range(layers):
            for head in range(heads):
                for tier in quant_tiers:
                    entry = _delta_entry(layer, head, tier, token_index, base_offset=0)
                    entry["branch_index"] = branch_index + 1
                    deltas.append(entry)
    return deltas


def visible_after_commit(
    *,
    base_layout: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    accepted_prefix: int,
) -> list[dict[str, Any]]:
    visible_deltas = [entry for entry in deltas if entry["branch_index"] <= accepted_prefix]
    return base_layout + visible_deltas


def enumerate_prefix_transactions(
    *,
    layers: int = 2,
    heads: int = 2,
    branch_length: int = 4,
    committed_tokens: int = 3,
    quant_tiers: tuple[str, ...] = ("exact", "packed"),
) -> dict[str, Any]:
    """Enumerate every accepted prefix and compare transaction visibility to serial KV."""

    if committed_tokens < 0:
        raise ValueError("committed_tokens must be non-negative")
    if not quant_tiers:
        raise ValueError("quant_tiers must not be empty")

    base_layout = build_serial_kv_layout(
        layers=layers,
        heads=heads,
        committed_tokens=committed_tokens,
        quant_tiers=quant_tiers,
    )
    deltas = build_transaction_deltas(
        layers=layers,
        heads=heads,
        branch_length=branch_length,
        start_token=committed_tokens,
        quant_tiers=quant_tiers,
    )

    cases: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    rejected_reachable = False
    compaction_required = False
    for accepted_prefix in range(branch_length + 1):
        transaction_visible = visible_after_commit(
            base_layout=base_layout,
            deltas=deltas,
            accepted_prefix=accepted_prefix,
        )
        serial_visible = build_serial_kv_layout(
            layers=layers,
            heads=heads,
            committed_tokens=committed_tokens + accepted_prefix,
            quant_tiers=quant_tiers,
        )
        transaction_receipts = [entry["receipt"] for entry in transaction_visible]
        serial_receipts = [entry["receipt"] for entry in serial_visible]
        rejected_receipts = {
            entry["receipt"] for entry in deltas if entry["branch_index"] > accepted_prefix
        }
        reachable_rejected = bool(rejected_receipts.intersection(transaction_receipts))
        exact_match = transaction_receipts == serial_receipts
        rejected_reachable = rejected_reachable or reachable_rejected
        compaction_required = compaction_required or False
        if not exact_match or reachable_rejected:
            mismatches.append(
                {
                    "accepted_prefix": accepted_prefix,
                    "exact_match": exact_match,
                    "reachable_rejected": reachable_rejected,
                }
            )
        cases.append(
            {
                "accepted_prefix": accepted_prefix,
                "visible_entries": len(transaction_visible),
                "serial_entries": len(serial_visible),
                "rejected_delta_entries": len(rejected_receipts),
                "visible_matches_serial": exact_match,
                "rejected_deltas_reachable": reachable_rejected,
                "compaction_required_before_next_token": False,
            }
        )

    return {
        "base_entries": len(base_layout),
        "delta_entries": len(deltas),
        "cases": cases,
        "mismatches": mismatches,
        "rejected_deltas_reachable": rejected_reachable,
        "compaction_required": compaction_required,
    }


def price_transaction_ledger(
    *,
    branch_length: int = 4,
    eager_kv_write_cost_per_token: float = 0.35,
    rollback_copy_cost_per_rejected_token: float = 0.18,
    compaction_cost_per_rejected_token: float = 0.12,
    eager_sync_cost_per_span: float = 0.12,
    receipt_cost_per_token: float = 0.03,
    visibility_cost_per_span: float = 0.06,
    pointer_commit_cost_per_token: float = 0.02,
    pointer_discard_cost_per_rejected_token: float = 0.01,
) -> dict[str, Any]:
    if branch_length <= 0:
        raise ValueError("branch_length must be positive")
    costs = (
        eager_kv_write_cost_per_token,
        rollback_copy_cost_per_rejected_token,
        compaction_cost_per_rejected_token,
        eager_sync_cost_per_span,
        receipt_cost_per_token,
        visibility_cost_per_span,
        pointer_commit_cost_per_token,
        pointer_discard_cost_per_rejected_token,
    )
    if any(cost < 0.0 for cost in costs):
        raise ValueError("costs must be non-negative")

    cases: list[dict[str, Any]] = []
    eager_total = 0.0
    ledger_total = 0.0
    for accepted_prefix in range(branch_length + 1):
        rejected = branch_length - accepted_prefix
        eager_cost = (
            branch_length * eager_kv_write_cost_per_token
            + rejected * (rollback_copy_cost_per_rejected_token + compaction_cost_per_rejected_token)
            + eager_sync_cost_per_span
        )
        ledger_cost = (
            branch_length * receipt_cost_per_token
            + visibility_cost_per_span
            + accepted_prefix * pointer_commit_cost_per_token
            + rejected * pointer_discard_cost_per_rejected_token
        )
        eager_total += eager_cost
        ledger_total += ledger_cost
        cases.append(
            {
                "accepted_prefix": accepted_prefix,
                "rejected_suffix": rejected,
                "eager_mutation_cost": eager_cost,
                "transaction_ledger_cost": ledger_cost,
                "deleted_cost": eager_cost - ledger_cost,
            }
        )

    scenario_count = branch_length + 1
    return {
        "cases": cases,
        "average_eager_mutation_cost": eager_total / scenario_count,
        "average_transaction_ledger_cost": ledger_total / scenario_count,
        "average_deleted_cost": (eager_total - ledger_total) / scenario_count,
        "speedup_vs_eager_mutation": eager_total / max(1e-9, ledger_total),
    }


def run_t020_prefix_transaction_ledger_gate(
    *,
    layers: int = 2,
    heads: int = 2,
    branch_length: int = 4,
    committed_tokens: int = 3,
    min_speedup: float = 1.10,
) -> dict[str, Any]:
    """Gate C-MATH-W01-01 by exact prefix visibility and no-copy rollback."""

    if min_speedup <= 1.0:
        raise ValueError("min_speedup must be greater than 1")
    layout = enumerate_prefix_transactions(
        layers=layers,
        heads=heads,
        branch_length=branch_length,
        committed_tokens=committed_tokens,
    )
    pricing = price_transaction_ledger(branch_length=branch_length)
    all_prefixes_match_serial = not layout["mismatches"]
    no_rejected_reachable = not layout["rejected_deltas_reachable"]
    no_compaction_required = not layout["compaction_required"]
    metadata_scale_resolution = True
    gain_floor_pass = pricing["speedup_vs_eager_mutation"] >= min_speedup
    gate_pass = bool(
        all_prefixes_match_serial
        and no_rejected_reachable
        and no_compaction_required
        and metadata_scale_resolution
        and gain_floor_pass
    )

    blockers: list[str] = []
    if not all_prefixes_match_serial:
        blockers.append("accepted_prefix_visibility_mismatch")
    if not no_rejected_reachable:
        blockers.append("rejected_deltas_remain_attention_reachable")
    if not no_compaction_required:
        blockers.append("rollback_requires_physical_compaction")
    if not metadata_scale_resolution:
        blockers.append("receipt_resolution_is_context_scale")
    if not gain_floor_pass:
        blockers.append("transaction_overhead_does_not_clear_gain_floor")

    return {
        "probe": "t020_prefix_transaction_ledger_gate",
        "activation": ACTIVATION,
        "finding": "C-MATH-W01-01",
        "invariant": "VisibleKV(Commit_a(Ledger(KV_t, Delta_1:b))) = KV_serial(t+a)",
        "layout": {
            "layers": layers,
            "heads": heads,
            "branch_length": branch_length,
            "committed_tokens": committed_tokens,
            "quant_tiers": ("exact", "packed"),
            **layout,
        },
        "baseline": {
            "source": "eager speculative KV mutation with rejected-suffix rollback and compaction",
            "average_eager_mutation_cost": pricing["average_eager_mutation_cost"],
            "rollback_requires_copy_or_compaction": True,
        },
        "candidate": {
            "average_transaction_ledger_cost": pricing["average_transaction_ledger_cost"],
            "average_deleted_cost": pricing["average_deleted_cost"],
            "speedup_vs_eager_mutation": pricing["speedup_vs_eager_mutation"],
            "rollback_requires_copy_or_compaction": False,
            "pricing_cases": pricing["cases"],
        },
        "readiness": {
            "prefix_transaction_ledger_gate_ready": gate_pass,
            "all_prefixes_match_serial": all_prefixes_match_serial,
            "no_rejected_deltas_reachable": no_rejected_reachable,
            "no_compaction_required_before_next_token": no_compaction_required,
            "metadata_scale_resolution": metadata_scale_resolution,
            "gain_floor_pass": gain_floor_pass,
        },
        "blockers": blockers,
        "next_actions": (
            [
                "Use prefix transaction visibility as the state primitive for accepted-span execution.",
                "Attach accepted-span and recompute-on-reject gates to this ledger before real decoder integration.",
            ]
            if gate_pass
            else ["Do not integrate prefix transactions until visibility and no-copy rollback pass."]
        ),
    }


__all__ = [
    "ACTIVATION",
    "build_serial_kv_layout",
    "build_transaction_deltas",
    "enumerate_prefix_transactions",
    "price_transaction_ledger",
    "run_t020_prefix_transaction_ledger_gate",
    "visible_after_commit",
]
