"""Device credit-flow gate for packed payload execution scheduling."""

from __future__ import annotations

from typing import Any


ACTIVATION = "T020_DEVICE_CREDIT_FLOW_ACTIVE"


def _mib_to_gib(value: float) -> float:
    return value / 1024.0


def run_t020_device_credit_flow_gate(
    *,
    baseline_total_seconds: float = 35.6344,
    baseline_payload_load_seconds: float = 28.0466,
    baseline_kernel_seconds: float = 5.6611,
    baseline_tokens_per_second: float = 0.028063,
    device_budget_gib: float = 12.0,
    baseline_denominator_gib: float = 11.8424,
    staged_payload_window_mib: float = 48.0,
    accumulator_window_mib: float = 64.0,
    credit_state_mib: float = 8.0,
    sync_workspace_mib: float = 8.0,
    decode_seconds: float = 0.0,
    epilogue_seconds: float = 0.0,
    credit_overhead_fraction: float = 0.02,
    sync_penalty_seconds: float = 0.0,
    min_speedup: float = 1.10,
) -> dict[str, Any]:
    """Price a device-side credit network against the T017 hot-path denominator.

    The gate does not claim that payload transfer is fully solved. It tests the
    narrower invariant from B-MATH-W02-09: whether a bounded in-flight credit
    window can legally overlap packed payload movement with the existing kernel
    window under the 12 GiB residency denominator.
    """

    if baseline_total_seconds <= 0.0:
        raise ValueError("baseline_total_seconds must be positive")
    if baseline_tokens_per_second <= 0.0:
        raise ValueError("baseline_tokens_per_second must be positive")
    if min_speedup <= 1.0:
        raise ValueError("min_speedup must be greater than 1.0")

    load_seconds = max(0.0, baseline_payload_load_seconds)
    kernel_seconds = max(0.0, baseline_kernel_seconds)
    decode_seconds = max(0.0, decode_seconds)
    epilogue_seconds = max(0.0, epilogue_seconds)
    overlap_service_window_seconds = kernel_seconds + decode_seconds + epilogue_seconds
    gross_overlap_seconds = min(load_seconds, overlap_service_window_seconds)
    overhead_seconds = gross_overlap_seconds * max(0.0, credit_overhead_fraction)
    net_overlap_seconds = max(0.0, gross_overlap_seconds - overhead_seconds - max(0.0, sync_penalty_seconds))
    candidate_total_seconds = max(1e-9, baseline_total_seconds - net_overlap_seconds)
    speedup_vs_baseline = baseline_total_seconds / candidate_total_seconds
    candidate_tokens_per_second = baseline_tokens_per_second * speedup_vs_baseline
    saved_fraction = net_overlap_seconds / baseline_total_seconds

    baseline_slack_gib = device_budget_gib - baseline_denominator_gib
    live_credit_gib = _mib_to_gib(
        staged_payload_window_mib + accumulator_window_mib + credit_state_mib + sync_workspace_mib
    )
    memory_fits = baseline_slack_gib > 0.0 and live_credit_gib <= baseline_slack_gib
    overlap_positive = net_overlap_seconds > 0.0 and overlap_service_window_seconds > 0.0
    transfer_credit_utilization = (
        gross_overlap_seconds / overlap_service_window_seconds if overlap_service_window_seconds > 0.0 else float("inf")
    )
    compute_window_utilization = gross_overlap_seconds / load_seconds if load_seconds > 0.0 else 0.0
    queue_stable = transfer_credit_utilization <= 1.0 and compute_window_utilization <= 1.0
    gain_floor_pass = speedup_vs_baseline >= min_speedup
    gate_pass = bool(memory_fits and overlap_positive and queue_stable and gain_floor_pass)

    resource_table = [
        {
            "resource": "packed_payload_transfer",
            "demand_seconds": load_seconds,
            "overlap_service_seconds": overlap_service_window_seconds,
            "hidden_seconds": gross_overlap_seconds,
            "residual_seconds_after_overlap": max(0.0, load_seconds - gross_overlap_seconds),
            "credit_utilization": transfer_credit_utilization,
            "stable": transfer_credit_utilization <= 1.0,
        },
        {
            "resource": "packed_kernel_compute_window",
            "demand_seconds": overlap_service_window_seconds,
            "producer_seconds": load_seconds,
            "hidden_seconds": gross_overlap_seconds,
            "credit_utilization": compute_window_utilization,
            "stable": compute_window_utilization <= 1.0,
        },
        {
            "resource": "live_device_credit_memory",
            "demand_gib": live_credit_gib,
            "available_slack_gib": baseline_slack_gib,
            "credit_utilization": live_credit_gib / baseline_slack_gib if baseline_slack_gib > 0.0 else float("inf"),
            "stable": memory_fits,
        },
    ]

    blockers: list[str] = []
    if not memory_fits:
        blockers.append("device_credit_window_exceeds_12gib_slack")
    if not overlap_positive:
        blockers.append("no_positive_transfer_compute_overlap_window")
    if not queue_stable:
        blockers.append("resource_credit_queue_not_stable")
    if not gain_floor_pass:
        blockers.append("projected_speedup_below_10_percent_floor")

    result = {
        "probe": "t020_device_credit_flow_gate",
        "activation": ACTIVATION,
        "finding": "B-MATH-W02-09",
        "baseline": {
            "source": "T017 hot all-64 final plus T014-A denominator",
            "total_seconds": baseline_total_seconds,
            "payload_load_seconds": load_seconds,
            "kernel_seconds": kernel_seconds,
            "tokens_per_second": baseline_tokens_per_second,
            "denominator_total_gib": baseline_denominator_gib,
            "device_budget_gib": device_budget_gib,
            "slack_gib": baseline_slack_gib,
        },
        "candidate": {
            "net_overlap_seconds": net_overlap_seconds,
            "gross_overlap_seconds": gross_overlap_seconds,
            "credit_overhead_seconds": overhead_seconds,
            "sync_penalty_seconds": max(0.0, sync_penalty_seconds),
            "total_seconds": candidate_total_seconds,
            "tokens_per_second": candidate_tokens_per_second,
            "speedup_vs_baseline": speedup_vs_baseline,
            "saved_fraction": saved_fraction,
            "live_credit_gib": live_credit_gib,
        },
        "inequalities": {
            "resource_credit_stability": queue_stable,
            "memory_live_credit_leq_slack": memory_fits,
            "speedup_geq_minimum": gain_floor_pass,
            "min_speedup": min_speedup,
        },
        "resource_table": resource_table,
        "readiness": {
            "device_credit_flow_gate_ready": gate_pass,
            "partial_overlap_feasible": bool(overlap_positive and queue_stable),
            "memory_credit_fits_12gib": memory_fits,
            "gain_floor_pass": gain_floor_pass,
        },
        "blockers": blockers,
        "next_actions": (
            [
                "Promote device-credit scheduling into the packed Qwen runtime behind an opt-in scheduler flag.",
                "Use the credit table to size staged payload and accumulator windows before residual-branch coalescing.",
            ]
            if gate_pass
            else [
                "Retune the credit window or sync penalty and rerun this gate before production integration.",
            ]
        ),
    }
    return result


__all__ = ["ACTIVATION", "run_t020_device_credit_flow_gate"]
