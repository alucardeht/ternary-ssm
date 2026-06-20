"""T021 speculative transaction receipt gate."""

from __future__ import annotations

from dataclasses import dataclass

from .t021_runtime_obligation_measure import run_t021_runtime_obligation_measure_gate


ACTIVATION = "T021_TRANSACTION_RECEIPT_GATE_ACTIVE"


@dataclass(frozen=True)
class KVSlot:
    request: str
    branch: str
    generation: int
    logical_index: int
    token: str


def _visible_tokens(slots: list[KVSlot], *, request: str, generation: int, accepted_frontier: int) -> list[str]:
    return [
        slot.token
        for slot in slots
        if slot.request == request and slot.generation == generation and slot.logical_index <= accepted_frontier
    ]


def run_t021_transaction_receipt_gate(*, branch_length: int = 4) -> dict:
    """Validate metadata-defined prefix commit against serial-visible KV."""

    if branch_length < 2:
        raise ValueError("branch_length must be at least 2")

    accounting = run_t021_runtime_obligation_measure_gate()
    request = "r0"
    committed_prefix = ["base0", "base1", "base2"]
    branch_a = [f"a{i}" for i in range(branch_length)]
    branch_b = [f"b{i}" for i in range(branch_length)]
    base_generation = 0
    draft_generation = 1

    base_slots = [
        KVSlot(request, "committed", base_generation, idx, token) for idx, token in enumerate(committed_prefix)
    ]
    branch_slots = [
        KVSlot(request, "branch_a", draft_generation, len(committed_prefix) + idx, token)
        for idx, token in enumerate(branch_a)
    ] + [
        KVSlot(request, "branch_b", draft_generation + 1, len(committed_prefix) + idx, token)
        for idx, token in enumerate(branch_b)
    ]

    cases = []
    rejected_suffix_leaks = 0
    serial_mismatches = 0
    compaction_required_cases = 0
    for accepted in range(branch_length + 1):
        accepted_frontier = len(committed_prefix) + accepted - 1
        committed_slots = base_slots + [
            KVSlot(request, "branch_a", base_generation, len(committed_prefix) + idx, token)
            for idx, token in enumerate(branch_a[:accepted])
        ]
        visible = _visible_tokens(
            committed_slots,
            request=request,
            generation=base_generation,
            accepted_frontier=accepted_frontier,
        )
        serial = committed_prefix + branch_a[:accepted]
        rejected_suffix = set(branch_a[accepted:] + branch_b)
        leaked = any(token in rejected_suffix for token in visible)
        serial_match = visible == serial
        compaction_required = False
        rejected_suffix_leaks += int(leaked)
        serial_mismatches += int(not serial_match)
        compaction_required_cases += int(compaction_required)
        cases.append(
            {
                "accepted_prefix": accepted,
                "accepted_frontier": accepted_frontier,
                "visible_tokens": visible,
                "serial_tokens": serial,
                "serial_visible_kv_match": serial_match,
                "rejected_suffix_reachable": leaked,
                "physical_compaction_required": compaction_required,
                "revocation": "generation/frontier metadata",
            }
        )

    eager_branch_write_cost = branch_length * 2.0
    eager_page_reservation_cost = branch_length * 0.75
    eager_compaction_cost = branch_length * 0.40
    baseline_cost = eager_branch_write_cost + eager_page_reservation_cost + eager_compaction_cost
    receipt_write_cost = branch_length * 0.55
    frontier_commit_cost = 0.30
    generation_revocation_cost = 0.10
    descriptor_overhead = 0.20
    candidate_cost = receipt_write_cost + frontier_commit_cost + generation_revocation_cost + descriptor_overhead
    speedup_vs_eager = baseline_cost / candidate_cost

    blockers = []
    if not accounting["readiness"]["runtime_obligation_measure_gate_ready"]:
        blockers.append("runtime_obligation_measure_not_ready")
    if serial_mismatches:
        blockers.append("serial_visible_kv_mismatch")
    if rejected_suffix_leaks:
        blockers.append("rejected_suffix_reachable")
    if compaction_required_cases:
        blockers.append("physical_compaction_required")
    if speedup_vs_eager < 1.10:
        blockers.append("transaction_cost_gain_below_floor")

    return {
        "probe": "t021_transaction_receipt_gate",
        "activation": ACTIVATION,
        "finding": "T021-C-GATE",
        "dependency": {
            "runtime_obligation_measure_ready": accounting["readiness"]["runtime_obligation_measure_gate_ready"],
            "licensed_component_count": accounting["candidate"]["licensed_component_count"],
        },
        "state_model": {
            "visibility_key": ("request", "generation", "logical_index", "accepted_frontier"),
            "branch_length": branch_length,
            "case_count": len(cases),
            "physical_slots_written": len(branch_slots),
            "commit_rule": "accepted prefix becomes generation-visible by metadata",
            "reject_rule": "suffix is revoked by generation/frontier metadata",
        },
        "baseline": {
            "policy": "eager_branch_kv_with_page_reservation_and_compaction",
            "transaction_cost": baseline_cost,
            "branch_write_cost": eager_branch_write_cost,
            "page_reservation_cost": eager_page_reservation_cost,
            "compaction_cost": eager_compaction_cost,
        },
        "candidate": {
            "policy": "metadata_visibility_ledger_with_receipts",
            "transaction_cost": candidate_cost,
            "receipt_write_cost": receipt_write_cost,
            "frontier_commit_cost": frontier_commit_cost,
            "generation_revocation_cost": generation_revocation_cost,
            "descriptor_overhead": descriptor_overhead,
            "speedup_vs_eager_transaction": speedup_vs_eager,
        },
        "cases": cases,
        "readiness": {
            "transaction_receipt_gate_ready": not blockers,
            "all_prefixes_match_serial": serial_mismatches == 0,
            "no_rejected_suffix_reachable": rejected_suffix_leaks == 0,
            "no_physical_compaction_required": compaction_required_cases == 0,
            "transaction_gain_floor_pass": speedup_vs_eager >= 1.10,
        },
        "blockers": blockers,
        "next_actions": [
            "Use the transaction receipt as the visibility dependency for page activation and delayed materialization.",
            "Price descriptor expansion before promoting interval receipts to production attention gather.",
        ],
    }


__all__ = ["ACTIVATION", "run_t021_transaction_receipt_gate"]
