"""T021 page activation certificate probe."""

from __future__ import annotations

from math import exp, log

from .t021_runtime_obligation_measure import run_t021_runtime_obligation_measure_gate
from .t021_transaction_receipt_gate import run_t021_transaction_receipt_gate


ACTIVATION = "T021_PAGE_ACTIVATION_CERTIFICATE_ACTIVE"


def _case_bounds() -> list[dict]:
    return [
        {
            "case": "random_query",
            "hot_margin": 8.0,
            "tail_upper_bounds": [2.0, 1.5, 2.8, 1.2, 2.1, 1.0],
            "far_anchor_page": None,
        },
        {
            "case": "aligned_recent_query",
            "hot_margin": 9.0,
            "tail_upper_bounds": [3.2, 2.7, 2.1, 3.0, 1.9, 2.4],
            "far_anchor_page": None,
        },
        {
            "case": "adversarial_far_anchor_query",
            "hot_margin": 8.4,
            "tail_upper_bounds": [2.0, 7.2, 1.9, 2.1, 1.8, 2.3],
            "far_anchor_page": 1,
        },
        {
            "case": "diffuse_high_value_query",
            "hot_margin": 8.2,
            "tail_upper_bounds": [4.7, 4.5, 4.1, 3.7, 3.5, 3.3],
            "far_anchor_page": None,
        },
    ]


def run_t021_page_activation_certificate_gate(
    *,
    exact_ring_pages: int = 2,
    tail_pages: int = 6,
    epsilon_z: float = 0.50,
    value_bound: float = 1.0,
) -> dict:
    """Run the offline page-effect certificate over fixed synthetic pages."""

    if exact_ring_pages < 1:
        raise ValueError("exact_ring_pages must be positive")
    if tail_pages < 1:
        raise ValueError("tail_pages must be positive")
    if epsilon_z <= 0:
        raise ValueError("epsilon_z must be positive")

    accounting = run_t021_runtime_obligation_measure_gate()
    transaction = run_t021_transaction_receipt_gate()
    threshold_offset = log(2.0 * value_bound * tail_pages / epsilon_z)
    dense_tail_cost = tail_pages * 2.0
    certificate_cost = tail_pages * 0.08
    transfer_cost = 0.10
    padding_cost = 0.05
    receipt_update_cost = 0.05

    case_results = []
    total_tail_activations = 0
    far_anchor_skipped = False
    bound_failures = 0
    for case in _case_bounds():
        threshold = case["hot_margin"] - threshold_offset
        active_tail_pages = []
        skipped_tail_pages = []
        skipped_effect_bound = 0.0
        for page_index, upper_bound in enumerate(case["tail_upper_bounds"][:tail_pages]):
            active = upper_bound > threshold
            if active:
                active_tail_pages.append(page_index)
            else:
                skipped_tail_pages.append(page_index)
                skipped_effect_bound += 2.0 * value_bound * exp(upper_bound - case["hot_margin"])
        if case["far_anchor_page"] is not None and case["far_anchor_page"] in skipped_tail_pages:
            far_anchor_skipped = True
        if skipped_effect_bound > epsilon_z:
            bound_failures += 1
        total_tail_activations += len(active_tail_pages)
        active_fraction = (exact_ring_pages + len(active_tail_pages)) / (exact_ring_pages + tail_pages)
        case_results.append(
            {
                "case": case["case"],
                "hot_margin": case["hot_margin"],
                "certificate_threshold": threshold,
                "active_tail_pages": active_tail_pages,
                "skipped_tail_pages": skipped_tail_pages,
                "active_fraction_with_exact_ring": active_fraction,
                "skipped_effect_bound": skipped_effect_bound,
                "epsilon_z": epsilon_z,
                "bound_pass": skipped_effect_bound <= epsilon_z,
                "far_anchor_page": case["far_anchor_page"],
                "far_anchor_activated": case["far_anchor_page"] is None
                or case["far_anchor_page"] in active_tail_pages,
            }
        )

    case_count = len(case_results)
    average_tail_active_fraction = total_tail_activations / (case_count * tail_pages)
    candidate_tail_cost = (
        certificate_cost
        + average_tail_active_fraction * dense_tail_cost
        + transfer_cost
        + padding_cost
        + receipt_update_cost
    )
    denominator_speedup = dense_tail_cost / candidate_tail_cost
    certificate_cost_fraction = certificate_cost / dense_tail_cost

    blockers = []
    if not accounting["readiness"]["runtime_obligation_measure_gate_ready"]:
        blockers.append("runtime_obligation_measure_not_ready")
    if not transaction["readiness"]["transaction_receipt_gate_ready"]:
        blockers.append("transaction_receipt_gate_not_ready")
    if far_anchor_skipped:
        blockers.append("far_anchor_page_skipped")
    if bound_failures:
        blockers.append("page_certificate_bound_failure")
    if average_tail_active_fraction >= 0.80:
        blockers.append("activated_tail_fraction_too_dense")
    if certificate_cost_fraction >= 0.25:
        blockers.append("certificate_cost_approaches_dense_tail")
    if denominator_speedup < 1.10:
        blockers.append("page_certificate_denominator_gain_below_floor")

    return {
        "probe": "t021_page_activation_certificate_gate",
        "activation": ACTIVATION,
        "finding": "T021-B-GATE",
        "dependency": {
            "runtime_obligation_measure_ready": accounting["readiness"]["runtime_obligation_measure_gate_ready"],
            "transaction_receipt_gate_ready": transaction["readiness"]["transaction_receipt_gate_ready"],
        },
        "certificate": {
            "inequality": "U_p(q) <= m_H(q) - log(2 * V_max * N_C / epsilon_z)",
            "exact_ring_pages": exact_ring_pages,
            "tail_pages": tail_pages,
            "epsilon_z": epsilon_z,
            "value_bound": value_bound,
            "threshold_offset": threshold_offset,
        },
        "baseline": {
            "policy": "dense_tail_reconstruction_and_attention",
            "tail_cost": dense_tail_cost,
            "tail_pages_scored": tail_pages,
        },
        "candidate": {
            "policy": "exact_ring_plus_page_effect_certificate",
            "tail_cost": candidate_tail_cost,
            "certificate_cost": certificate_cost,
            "transfer_cost": transfer_cost,
            "padding_cost": padding_cost,
            "receipt_update_cost": receipt_update_cost,
            "average_tail_active_fraction": average_tail_active_fraction,
            "certificate_cost_fraction": certificate_cost_fraction,
            "denominator_speedup_vs_dense_tail": denominator_speedup,
        },
        "cases": case_results,
        "readiness": {
            "page_activation_certificate_gate_ready": not blockers,
            "no_far_anchor_skip": not far_anchor_skipped,
            "all_certificate_bounds_pass": bound_failures == 0,
            "activated_tail_fraction_below_dense_guard": average_tail_active_fraction < 0.80,
            "certificate_cost_below_dense_guard": certificate_cost_fraction < 0.25,
            "denominator_gain_floor_pass": denominator_speedup >= 1.10,
        },
        "blockers": blockers,
        "next_actions": [
            "Keep this as an offline denominator probe until a production decoder exposes page-level tail receipts.",
            "Do not report measured tok/s from this synthetic certificate.",
        ],
    }


__all__ = ["ACTIVATION", "run_t021_page_activation_certificate_gate"]
