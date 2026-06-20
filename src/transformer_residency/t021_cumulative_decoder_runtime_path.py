"""Cumulative T021 decoder runtime gate."""

from __future__ import annotations

from .t020_cumulative_execution_shape import run_t020_cumulative_execution_shape_gate
from .t021_page_activation_certificate import run_t021_page_activation_certificate_gate
from .t021_runtime_obligation_measure import run_t021_runtime_obligation_measure_gate
from .t021_transaction_receipt_gate import run_t021_transaction_receipt_gate


ACTIVATION = "T021_CUMULATIVE_DECODER_RUNTIME_PATH_ACTIVE"


def run_t021_cumulative_decoder_runtime_path_gate() -> dict:
    """Run the ordered T021 A/C/B gates as one cumulative decoder-path proof."""

    t020 = run_t020_cumulative_execution_shape_gate()
    accounting = run_t021_runtime_obligation_measure_gate()
    transaction = run_t021_transaction_receipt_gate()
    page_certificate = run_t021_page_activation_certificate_gate()

    independent_modeled_gains = {
        "transaction_receipt_speedup_vs_eager_transaction": transaction["candidate"][
            "speedup_vs_eager_transaction"
        ],
        "page_certificate_speedup_vs_dense_tail": page_certificate["candidate"][
            "denominator_speedup_vs_dense_tail"
        ],
    }
    non_overlapping_projection = (
        independent_modeled_gains["transaction_receipt_speedup_vs_eager_transaction"]
        * independent_modeled_gains["page_certificate_speedup_vs_dense_tail"]
    )

    blockers = []
    if not accounting["readiness"]["runtime_obligation_measure_gate_ready"]:
        blockers.append("runtime_obligation_measure_not_ready")
    if not transaction["readiness"]["transaction_receipt_gate_ready"]:
        blockers.append("transaction_receipt_gate_not_ready")
    if not page_certificate["readiness"]["page_activation_certificate_gate_ready"]:
        blockers.append("page_activation_certificate_gate_not_ready")

    return {
        "probe": "t021_cumulative_decoder_runtime_path_gate",
        "activation": ACTIVATION,
        "finding": "T021-CUMULATIVE",
        "baseline": {
            "policy": t020["composition"]["policy"],
            "candidate_tokens_per_second_proxy": t020["composition"]["candidate_tokens_per_second_proxy"],
            "cumulative_speedup_proxy": t020["composition"]["cumulative_speedup"],
            "runtime_obligation_components_licensed": 0,
            "transaction_receipt_gate_ready": False,
            "page_certificate_gate_ready": False,
        },
        "candidate": {
            "policy": "ordered_runtime_obligation_then_transaction_then_page_certificate",
            "runtime_obligation_components_licensed": accounting["candidate"]["licensed_component_count"],
            "unresolved_overlap_classes": accounting["candidate"]["unresolved_overlap_classes"],
            "transaction_receipt_gate_ready": transaction["readiness"]["transaction_receipt_gate_ready"],
            "page_certificate_gate_ready": page_certificate["readiness"]["page_activation_certificate_gate_ready"],
            "independent_modeled_gains": independent_modeled_gains,
            "non_overlapping_denominator_projection": non_overlapping_projection,
            "production_tok_s_claim_allowed": False,
        },
        "readiness": {
            "cumulative_decoder_runtime_path_gate_ready": not blockers,
            "runtime_obligation_measure_ready": accounting["readiness"]["runtime_obligation_measure_gate_ready"],
            "transaction_receipt_ready": transaction["readiness"]["transaction_receipt_gate_ready"],
            "page_activation_certificate_ready": page_certificate["readiness"][
                "page_activation_certificate_gate_ready"
            ],
            "t020_proxy_replaced_by_ordered_gates": True,
        },
        "subgates": {
            "runtime_obligation_measure": {
                "activation": accounting["activation"],
                "readiness": accounting["readiness"],
                "baseline": accounting["baseline"],
                "candidate": accounting["candidate"],
            },
            "transaction_receipt": {
                "activation": transaction["activation"],
                "readiness": transaction["readiness"],
                "baseline": transaction["baseline"],
                "candidate": transaction["candidate"],
            },
            "page_activation_certificate": {
                "activation": page_certificate["activation"],
                "readiness": page_certificate["readiness"],
                "baseline": page_certificate["baseline"],
                "candidate": page_certificate["candidate"],
            },
        },
        "blockers": blockers,
        "next_actions": [],
    }


__all__ = ["ACTIVATION", "run_t021_cumulative_decoder_runtime_path_gate"]
