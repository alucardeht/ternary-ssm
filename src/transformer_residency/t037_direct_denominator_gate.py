"""T037 direct exact resident-denominator gate."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


ACTIVATION = "T037_DIRECT_DENOMINATOR_GATE_ACTIVE"

R452_FULL64_TOKENS_PER_SECOND = 0.9280330765589875
R452_FULL64_SECONDS_PER_TOKEN = 1.0775478000286967
R452_KERNEL_SECONDS_PER_TOKEN = 0.5049169003032148
R452_NON_KERNEL_SECONDS_PER_TOKEN = 0.5726308997254819
TARGET_TOKENS_PER_SECOND = 15.0
TARGET_SECONDS_PER_TOKEN = 1.0 / TARGET_TOKENS_PER_SECOND
DEFAULT_PEAK_CUDA_GIB = 10.652063369750977
DEFAULT_DEVICE_BUDGET_GIB = 12.0


def _scenario(
    *,
    name: str,
    kernel_seconds: float,
    non_kernel_seconds: float,
    setup_seconds_per_token: float,
    peak_cuda_gib: float,
    device_budget_gib: float,
    exact_state_and_token: bool,
    hidden_sidecar_gib: float,
) -> dict[str, Any]:
    total = kernel_seconds + non_kernel_seconds + setup_seconds_per_token
    tok_s = 1.0 / total if total > 0.0 else 0.0
    target_pass = (
        total <= TARGET_SECONDS_PER_TOKEN
        and peak_cuda_gib <= device_budget_gib
        and exact_state_and_token
        and hidden_sidecar_gib == 0.0
    )
    return {
        "name": name,
        "kernel_seconds_per_token": kernel_seconds,
        "non_kernel_seconds_per_token": non_kernel_seconds,
        "setup_seconds_per_token": setup_seconds_per_token,
        "total_seconds_per_token": total,
        "tokens_per_second": tok_s,
        "peak_cuda_gib": peak_cuda_gib,
        "device_budget_gib": device_budget_gib,
        "hidden_sidecar_gib": hidden_sidecar_gib,
        "exact_state_and_token": exact_state_and_token,
        "target_pass": target_pass,
        "required_total_reduction_vs_r452": R452_FULL64_SECONDS_PER_TOKEN / TARGET_SECONDS_PER_TOKEN,
        "required_kernel_reduction_if_non_kernel_zero": R452_KERNEL_SECONDS_PER_TOKEN / TARGET_SECONDS_PER_TOKEN,
    }


def run_t037_direct_denominator_gate(
    *,
    measured_direct_seconds_per_token: float | None = None,
    measured_kernel_seconds_per_token: float | None = None,
    measured_non_kernel_seconds_per_token: float | None = None,
    peak_cuda_gib: float = DEFAULT_PEAK_CUDA_GIB,
    hidden_sidecar_gib: float = 0.0,
    device_budget_gib: float = DEFAULT_DEVICE_BUDGET_GIB,
    exact_state_and_token: bool = True,
    setup_seconds_per_token: float = 0.0,
) -> dict[str, Any]:
    """Fail-closed gate for direct exact resident full64 generation.

    This does not project a speedup. It prices the necessary inequality for an
    exact resident VM/conveyor replacement after the suffix-verifier frame failed.
    """

    if measured_direct_seconds_per_token is None:
        direct_kernel = (
            R452_KERNEL_SECONDS_PER_TOKEN
            if measured_kernel_seconds_per_token is None
            else float(measured_kernel_seconds_per_token)
        )
        direct_non_kernel = (
            R452_NON_KERNEL_SECONDS_PER_TOKEN
            if measured_non_kernel_seconds_per_token is None
            else float(measured_non_kernel_seconds_per_token)
        )
    else:
        measured_direct = float(measured_direct_seconds_per_token)
        if measured_kernel_seconds_per_token is None and measured_non_kernel_seconds_per_token is None:
            direct_kernel = measured_direct
            direct_non_kernel = 0.0
        else:
            direct_kernel = (
                0.0 if measured_kernel_seconds_per_token is None else float(measured_kernel_seconds_per_token)
            )
            direct_non_kernel = (
                measured_direct - direct_kernel
                if measured_non_kernel_seconds_per_token is None
                else float(measured_non_kernel_seconds_per_token)
            )

    current = _scenario(
        name="current_direct_exact_denominator",
        kernel_seconds=direct_kernel,
        non_kernel_seconds=direct_non_kernel,
        setup_seconds_per_token=setup_seconds_per_token,
        peak_cuda_gib=peak_cuda_gib,
        device_budget_gib=device_budget_gib,
        exact_state_and_token=exact_state_and_token,
        hidden_sidecar_gib=hidden_sidecar_gib,
    )
    zero_control = _scenario(
        name="control_only_removed",
        kernel_seconds=direct_kernel,
        non_kernel_seconds=0.0,
        setup_seconds_per_token=setup_seconds_per_token,
        peak_cuda_gib=peak_cuda_gib,
        device_budget_gib=device_budget_gib,
        exact_state_and_token=exact_state_and_token,
        hidden_sidecar_gib=hidden_sidecar_gib,
    )
    target_edge = _scenario(
        name="target_edge_exact_vm",
        kernel_seconds=TARGET_SECONDS_PER_TOKEN,
        non_kernel_seconds=0.0,
        setup_seconds_per_token=0.0,
        peak_cuda_gib=peak_cuda_gib,
        device_budget_gib=device_budget_gib,
        exact_state_and_token=exact_state_and_token,
        hidden_sidecar_gib=hidden_sidecar_gib,
    )
    scenarios = [current, zero_control, target_edge]

    gate_ready = current["target_pass"]
    blockers: list[str] = []
    if not exact_state_and_token:
        blockers.append("exact_state_or_token_not_proven")
    if hidden_sidecar_gib > 0.0:
        blockers.append("hidden_dense_or_host_sidecar_live")
    if peak_cuda_gib > device_budget_gib:
        blockers.append("projected_peak_cuda_exceeds_device_budget")
    if not current["target_pass"]:
        blockers.append("direct_denominator_does_not_reach_15_tok_s")
    if zero_control["total_seconds_per_token"] > TARGET_SECONDS_PER_TOKEN:
        blockers.append("control_removal_alone_cannot_reach_target")

    return {
        "probe": "t037_direct_denominator_gate",
        "activation": ACTIVATION,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "target": {
            "tokens_per_second": TARGET_TOKENS_PER_SECOND,
            "seconds_per_token": TARGET_SECONDS_PER_TOKEN,
        },
        "baseline": {
            "source": "R452/T028 full64 direct-token denominator",
            "tokens_per_second": R452_FULL64_TOKENS_PER_SECOND,
            "seconds_per_token": R452_FULL64_SECONDS_PER_TOKEN,
            "kernel_seconds_per_token": R452_KERNEL_SECONDS_PER_TOKEN,
            "non_kernel_seconds_per_token": R452_NON_KERNEL_SECONDS_PER_TOKEN,
        },
        "scenarios": scenarios,
        "readiness": {
            "t037_direct_denominator_gate_ready": gate_ready,
            "t037_direct_exact_vm_reaches_15_tok_s": current["target_pass"],
            "t037_control_removal_alone_reaches_15_tok_s": zero_control["target_pass"],
            "t037_exact_state_and_token_proven": exact_state_and_token,
            "t037_no_hidden_sidecar": hidden_sidecar_gib == 0.0,
            "t037_fits_device_budget": peak_cuda_gib <= device_budget_gib,
            "t037_production_implementation_allowed": gate_ready,
        },
        "best_current_row": current,
        "control_only_row": zero_control,
        "blockers": blockers,
        "next_actions": []
        if gate_ready
        else [
            "Do not promote a direct exact VM implementation until the measured denominator is <=0.0666667s/token.",
            "Start a new research round focused on replacing the kernel/body denominator, not boundary receipts.",
        ],
    }


__all__ = ["ACTIVATION", "run_t037_direct_denominator_gate"]
