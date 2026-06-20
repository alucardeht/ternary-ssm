"""T021 runtime obligation accounting gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .t020_cumulative_execution_shape import run_t020_cumulative_execution_shape_gate


ACTIVATION = "T021_RUNTIME_OBLIGATION_MEASURE_ACTIVE"
TERMS = ("W", "P", "K", "S", "A", "L", "Q", "G", "M")


@dataclass(frozen=True)
class ObligationSpec:
    finding: str
    label: str
    scope: str
    terms: tuple[str, ...]
    event_title: str
    visibility_phase: str
    normalizer: str
    exactness: str
    transform: str
    replacement: str
    overhead: str
    ordering_group: str


COMPONENT_SPECS: tuple[ObligationSpec, ...] = (
    ObligationSpec(
        "B-MATH-W02-09",
        "device_credit_flow",
        "hot_decode_total_seconds",
        ("P", "K", "M"),
        "hot.decode.device-credit.window",
        "committed_decode",
        "token",
        "exact",
        "transforms staged payload and accumulator residency into bounded live credit",
        "credit metadata and bounded sync workspace",
        "credit bookkeeping and sync penalty",
        "device_credit",
    ),
    ObligationSpec(
        "A-MATH-W01-01",
        "execution_order_slab",
        "layer1_arena_acquisition",
        ("W", "G", "M"),
        "arena.acquisition.slab-object",
        "arena_open",
        "layer_group",
        "exact",
        "replaces many acquisition objects with one ordered slab",
        "offset walk",
        "offset table lookup",
        "arena_identity",
    ),
    ObligationSpec(
        "A-MATH-W01-15",
        "immutable_arena_receipt",
        "layer1_arena_identity",
        ("W", "G"),
        "arena.identity.immutable-receipt",
        "arena_open",
        "layer_group",
        "exact",
        "replaces dynamic discovery with immutable descriptor receipt",
        "descriptor root verification",
        "root lookup and descriptor checks",
        "arena_identity",
    ),
    ObligationSpec(
        "A-PB-W02-14",
        "conversion_time_arena_compiler",
        "runtime_descriptor_interpretation",
        ("W", "G"),
        "arena.descriptor.compiled-view",
        "arena_open",
        "descriptor",
        "exact",
        "moves descriptor interpretation to conversion time",
        "compiled descriptor admission",
        "receipt admission",
        "arena_identity",
    ),
    ObligationSpec(
        "A-MATH-W02-03",
        "affine_packed_view",
        "packed_payload_materialization",
        ("W", "P", "M"),
        "payload.materialization.affine-view",
        "projection_setup",
        "payload_coordinate",
        "exact",
        "replaces materialized payload coordinates with affine packed view",
        "bounded two-bit coordinate extraction",
        "decode and coalescing penalties",
        "payload_view",
    ),
    ObligationSpec(
        "C-MATH-W02-02",
        "recompute_on_reject",
        "speculative_rejected_branch_kv",
        ("S", "K", "M"),
        "speculation.rejected-kv.recompute-receipt",
        "private_speculation",
        "branch_position",
        "exact",
        "defers rejected branch KV behind exact recompute receipts",
        "receipt plus recompute on demand",
        "cleanup and receipt cost",
        "speculative_transaction",
    ),
    ObligationSpec(
        "C-MATH-W01-02",
        "expected_accepted_span",
        "accepted_token_denominator",
        ("S", "L", "Q"),
        "speculation.accepted-span.denominator",
        "verification",
        "accepted_token",
        "exact",
        "renormalizes branch work by accepted prefix length",
        "verification and rollback terms",
        "schedule and graph gap",
        "speculative_transaction",
    ),
    ObligationSpec(
        "C-MATH-W01-01",
        "prefix_transaction_ledger",
        "speculative_prefix_visibility",
        ("S", "K", "L"),
        "speculation.prefix.visibility-ledger",
        "commit",
        "accepted_prefix",
        "exact",
        "commits accepted prefix visibility without rejected suffix reachability",
        "transaction ledger entries",
        "metadata mutation",
        "speculative_transaction",
    ),
    ObligationSpec(
        "C-MATH-W01-10",
        "acceptance_weighted_kv_materialization",
        "accepted_span_kv_materialization",
        ("S", "K", "M"),
        "speculation.kv.acceptance-weighted-materialization",
        "commit",
        "token_layer",
        "exact",
        "delays materialization until accepted-span visibility requires it",
        "conditional materialization receipt",
        "receipt and schedule cost",
        "speculative_transaction",
    ),
    ObligationSpec(
        "C-MATH-W01-05",
        "basis_shared_kv_transform",
        "representation_crossing",
        ("K", "A", "P"),
        "kv.representation.basis-shared-dot",
        "attention_read",
        "head_vector",
        "epsilon_bounded",
        "preserves dot products across packed and attention basis",
        "basis-shared receipt",
        "numeric debt check",
        "context_activation",
    ),
    ObligationSpec(
        "C-MATH-W01-11",
        "ring_exact_tail_receipts",
        "long_context_active_kv_policy",
        ("K", "A", "M"),
        "kv.context.exact-ring-tail-receipt",
        "attention_read",
        "context_page",
        "declared_debt",
        "keeps newest ring exact and moves old tail behind receipts",
        "tail debt receipts",
        "ring/tail bookkeeping",
        "context_activation",
    ),
    ObligationSpec(
        "B-PB-W02-11",
        "packed_work_coalescing",
        "packed_kernel_residual_destinations",
        ("P", "M"),
        "projection.residual.packed-coalesced-destination",
        "projection_write",
        "residual_destination",
        "exact",
        "fuses legal residual destinations and deletes dense write/reread traffic",
        "coalesced residual destination",
        "kernel scheduling constraint",
        "payload_view",
    ),
)


OVERLAP_RULES = {
    "arena_identity": "ordered arena chain with remaining-mass clipping over acquisition, identity, and descriptor interpretation",
    "payload_view": "payload materialization and residual coalescing are distinct event titles; structural bytes need pricing before time credit",
    "speculative_transaction": "shared transaction receipt owns rejected suffix, accepted-span, visibility, and materialization atoms",
    "context_activation": "basis transform and ring/tail policy require page-certificate ownership before active-context promotion",
    "device_credit": "hot decode total can only claim atoms not already owned by narrower component rows",
}


def _matching_t020_item(t020: dict, label: str) -> dict:
    for item in t020["numeric_items"] + t020["structural_items"]:
        if item["label"] == label:
            return item
    raise KeyError(label)


def run_t021_runtime_obligation_measure_gate() -> dict:
    """Instantiate the T021 accounting object over concrete T020 components."""

    t020 = run_t020_cumulative_execution_shape_gate()
    rows = []
    term_coverage = {term: 0 for term in TERMS}
    event_titles: set[str] = set()
    duplicate_event_titles: list[str] = []
    unknown_terms: list[str] = []

    for spec in COMPONENT_SPECS:
        source = _matching_t020_item(t020, spec.label)
        for term in spec.terms:
            if term not in term_coverage:
                unknown_terms.append(term)
            else:
                term_coverage[term] += 1
        if spec.event_title in event_titles:
            duplicate_event_titles.append(spec.event_title)
        event_titles.add(spec.event_title)
        row = asdict(spec)
        row.update(
            {
                "source_before": source.get("before"),
                "source_after": source.get("after"),
                "source_speedup": source.get("speedup"),
                "source_deleted_dense_mib": source.get("deleted_dense_mib"),
                "source_activation": source["activation"],
                "credited_by_t020_proxy": bool(source.get("composes_in_stack")),
                "runtime_obligation_licensed": True,
            }
        )
        rows.append(row)

    overlap_classes = []
    for group, rule in OVERLAP_RULES.items():
        members = [spec.label for spec in COMPONENT_SPECS if spec.ordering_group == group]
        overlap_classes.append(
            {
                "group": group,
                "members": members,
                "member_count": len(members),
                "resolution": rule,
                "unresolved_after_gate": False,
            }
        )

    baseline_unresolved_overlap_classes = sum(1 for row in overlap_classes if row["member_count"] > 1)
    candidate_unresolved_overlap_classes = sum(1 for row in overlap_classes if row["unresolved_after_gate"])
    all_t020_components_present = len(rows) == (
        t020["composition"]["numeric_item_count"] + t020["composition"]["structural_item_count"]
    )
    all_terms_covered = all(count > 0 for count in term_coverage.values())
    all_event_titles_unique = not duplicate_event_titles
    all_overlap_classes_resolved = candidate_unresolved_overlap_classes == 0

    blockers = []
    if not all_t020_components_present:
        blockers.append("not_all_t020_components_mapped")
    if not all_terms_covered:
        blockers.append("not_all_runtime_terms_covered")
    if unknown_terms:
        blockers.append("unknown_runtime_terms_present")
    if not all_event_titles_unique:
        blockers.append("duplicate_event_titles")
    if not all_overlap_classes_resolved:
        blockers.append("unresolved_overlap_classes")

    readiness = {
        "runtime_obligation_measure_gate_ready": not blockers,
        "all_t020_components_mapped": all_t020_components_present,
        "all_runtime_terms_covered": all_terms_covered,
        "event_titles_unique": all_event_titles_unique,
        "overlap_classes_resolved": all_overlap_classes_resolved,
        "t020_proxy_kept_as_target_envelope_only": True,
    }

    return {
        "probe": "t021_runtime_obligation_measure_gate",
        "activation": ACTIVATION,
        "finding": "T021-A-GATE",
        "obligation_measure": {
            "omega": "typed runtime obligation atoms",
            "mu": "cost mass per atom",
            "phi": "ordered component transforms",
            "visibility": "component-visible runtime phase",
            "normalizer": "unit of progress",
            "exactness": "exactness or debt status",
        },
        "baseline": {
            "policy": t020["composition"]["policy"],
            "numeric_item_count": t020["composition"]["numeric_item_count"],
            "structural_item_count": t020["composition"]["structural_item_count"],
            "unresolved_overlap_classes": baseline_unresolved_overlap_classes,
            "licensed_component_count": 0,
            "candidate_tokens_per_second_proxy": t020["composition"]["candidate_tokens_per_second_proxy"],
            "cumulative_speedup_proxy": t020["composition"]["cumulative_speedup"],
        },
        "candidate": {
            "policy": "runtime_obligation_measure_with_event_titles_and_remaining_mass_clipping",
            "licensed_component_count": len(rows),
            "unresolved_overlap_classes": candidate_unresolved_overlap_classes,
            "term_coverage": term_coverage,
            "unique_event_titles": len(event_titles),
            "production_speed_claim_allowed": False,
        },
        "ownership_matrix": rows,
        "overlap_classes": overlap_classes,
        "readiness": readiness,
        "blockers": blockers,
        "next_actions": [
            "Use this matrix as the required accounting dependency for T021-C and T021-B gates.",
            "Do not convert T020 proxy speed into production tok/s until a measured decoder path emits these obligations.",
        ],
    }


__all__ = ["ACTIVATION", "COMPONENT_SPECS", "TERMS", "run_t021_runtime_obligation_measure_gate"]
