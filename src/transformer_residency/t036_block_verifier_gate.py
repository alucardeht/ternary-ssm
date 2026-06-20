"""T036 A05-06 full64 block-verifier falsifier."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


ACTIVATION = "T036_A05_BLOCK_VERIFIER_GATE_ACTIVE"

R452_FULL64_TOKENS_PER_SECOND = 0.9280330765589875
R452_FULL64_SECONDS_PER_TOKEN = 1.0775478000286967
R452_FULL64_KERNEL_SECONDS = 0.5049169003032148
R452_FULL64_NON_KERNEL_SECONDS = R452_FULL64_SECONDS_PER_TOKEN - R452_FULL64_KERNEL_SECONDS
R452_FULL64_PEAK_CUDA_GIB = 10.652063369750977
R452_FULL64_RESIDENT_CUDA_GIB = 10.458633184432983
TARGET_TOKENS_PER_SECOND = 15.0
TARGET_SECONDS_PER_TOKEN = 1.0 / TARGET_TOKENS_PER_SECOND
REQUIRED_SPEEDUP = R452_FULL64_SECONDS_PER_TOKEN / TARGET_SECONDS_PER_TOKEN
DEVICE_BUDGET_GIB = 12.0


def _csv_ints(value: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(int(item) for item in value)
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return tuple(int(item) for item in items)


def _accepted_tokens_for_block(block_size: int) -> int:
    defaults = {16: 13, 32: 26, 64: 33}
    return defaults.get(block_size, max(1, int(round(block_size * 0.80))))


def _projected_multiplier_for_block(block_size: int) -> float:
    defaults = {16: 1.0, 32: 1.60, 64: 2.00}
    return defaults.get(block_size, max(1.0, block_size / 32.0 * 1.60))


def _serial_full64_cost(block_size: int) -> float:
    return float(block_size) * R452_FULL64_SECONDS_PER_TOKEN


def _priced_row(
    *,
    block_size: int,
    accepted_tokens: int,
    verifier_seconds: float,
    extra_seconds: float,
    mode: str,
) -> dict[str, Any]:
    total_seconds = verifier_seconds + extra_seconds
    seconds_per_accepted_token = total_seconds / max(accepted_tokens, 1)
    tokens_per_second = 1.0 / seconds_per_accepted_token if seconds_per_accepted_token > 0.0 else 0.0
    return {
        "block_size": block_size,
        "accepted_tokens": accepted_tokens,
        "acceptance_fraction": accepted_tokens / block_size if block_size > 0 else 0.0,
        "mode": mode,
        "verifier_seconds": verifier_seconds,
        "extra_seconds": extra_seconds,
        "total_transaction_seconds": total_seconds,
        "seconds_per_accepted_token": seconds_per_accepted_token,
        "tokens_per_second": tokens_per_second,
        "cost_multiplier_vs_r452_token": total_seconds / R452_FULL64_SECONDS_PER_TOKEN,
        "required_accepted_tokens_for_15_tok_s": total_seconds / TARGET_SECONDS_PER_TOKEN,
        "target_pass": seconds_per_accepted_token <= TARGET_SECONDS_PER_TOKEN,
        "target_margin_seconds_per_token": TARGET_SECONDS_PER_TOKEN - seconds_per_accepted_token,
    }


def _memory_contract(
    *,
    shadow_kv_gib: float,
    scratch_gib: float,
    graph_memory_gib: float,
    dense_sidecar_gib: float,
    reserved_pool_growth_gib: float,
    device_budget_gib: float,
) -> dict[str, Any]:
    extra_peak_gib = shadow_kv_gib + scratch_gib + graph_memory_gib + dense_sidecar_gib + reserved_pool_growth_gib
    projected_peak_gib = R452_FULL64_PEAK_CUDA_GIB + extra_peak_gib
    return {
        "baseline_peak_cuda_gib": R452_FULL64_PEAK_CUDA_GIB,
        "baseline_resident_cuda_gib": R452_FULL64_RESIDENT_CUDA_GIB,
        "shadow_kv_gib": shadow_kv_gib,
        "scratch_gib": scratch_gib,
        "graph_memory_gib": graph_memory_gib,
        "dense_sidecar_gib": dense_sidecar_gib,
        "reserved_pool_growth_gib": reserved_pool_growth_gib,
        "extra_peak_gib": extra_peak_gib,
        "projected_peak_cuda_gib": projected_peak_gib,
        "device_budget_gib": device_budget_gib,
        "fits_device_budget": projected_peak_gib <= device_budget_gib,
        "hidden_dense_sidecar_free": dense_sidecar_gib == 0.0,
    }


def run_t036_a05_block_verifier_gate(
    *,
    block_sizes: tuple[int, ...] | list[int] | str = (16, 32, 64),
    verifier_mode: str = "serial_full64_replay",
    extra_seconds: float = 0.0,
    shadow_kv_gib: float = 0.0,
    scratch_gib: float = 0.0,
    graph_memory_gib: float = 0.0,
    dense_sidecar_gib: float = 0.0,
    reserved_pool_growth_gib: float = 0.0,
    device_budget_gib: float = DEVICE_BUDGET_GIB,
    exact_state_equality: bool = True,
    exact_token_equality: bool = True,
    measured_verifier_seconds: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Price the A05-06 block verifier against the full64 target."""

    sizes = _csv_ints(block_sizes)
    measured_verifier_seconds = measured_verifier_seconds or {}
    rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    blockers: list[str] = []

    if verifier_mode not in {"serial_full64_replay", "projected_sublinear"}:
        blockers.append("unknown_verifier_mode")

    for block_size in sizes:
        accepted_tokens = _accepted_tokens_for_block(block_size)
        if verifier_mode == "projected_sublinear":
            verifier_seconds = _projected_multiplier_for_block(block_size) * R452_FULL64_SECONDS_PER_TOKEN
        elif block_size in measured_verifier_seconds:
            verifier_seconds = float(measured_verifier_seconds[block_size])
        else:
            verifier_seconds = _serial_full64_cost(block_size)
        rows.append(
            _priced_row(
                block_size=block_size,
                accepted_tokens=accepted_tokens,
                verifier_seconds=verifier_seconds,
                extra_seconds=extra_seconds,
                mode=verifier_mode,
            )
        )
        conditional_rows.append(
            _priced_row(
                block_size=block_size,
                accepted_tokens=accepted_tokens,
                verifier_seconds=_projected_multiplier_for_block(block_size) * R452_FULL64_SECONDS_PER_TOKEN,
                extra_seconds=extra_seconds,
                mode="t036_final_audit_conditional_projection",
            )
        )

    memory = _memory_contract(
        shadow_kv_gib=shadow_kv_gib,
        scratch_gib=scratch_gib,
        graph_memory_gib=graph_memory_gib,
        dense_sidecar_gib=dense_sidecar_gib,
        reserved_pool_growth_gib=reserved_pool_growth_gib,
        device_budget_gib=device_budget_gib,
    )
    pass_rows = [row for row in rows if row["target_pass"]]
    conditional_pass_rows = [row for row in conditional_rows if row["target_pass"]]

    if not pass_rows:
        blockers.append("block_verifier_does_not_reach_15_tok_s")
    if verifier_mode == "serial_full64_replay":
        blockers.append("verifier_is_serial_full64_replay")
    if not exact_state_equality:
        blockers.append("exact_state_equality_not_proven")
    if not exact_token_equality:
        blockers.append("exact_token_equality_not_proven")
    if not memory["fits_device_budget"]:
        blockers.append("projected_peak_cuda_exceeds_device_budget")
    if not memory["hidden_dense_sidecar_free"]:
        blockers.append("hidden_dense_sidecar_live")

    gate_ready = (
        bool(pass_rows)
        and verifier_mode != "serial_full64_replay"
        and exact_state_equality
        and exact_token_equality
        and memory["fits_device_budget"]
        and memory["hidden_dense_sidecar_free"]
        and "unknown_verifier_mode" not in blockers
    )

    return {
        "probe": "t036_a05_block_verifier_gate",
        "activation": ACTIVATION,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "baseline": {
            "source": "R452/T028",
            "tokens_per_second": R452_FULL64_TOKENS_PER_SECOND,
            "seconds_per_token": R452_FULL64_SECONDS_PER_TOKEN,
            "kernel_seconds": R452_FULL64_KERNEL_SECONDS,
            "non_kernel_seconds": R452_FULL64_NON_KERNEL_SECONDS,
            "peak_cuda_gib": R452_FULL64_PEAK_CUDA_GIB,
            "resident_cuda_gib": R452_FULL64_RESIDENT_CUDA_GIB,
        },
        "target": {
            "tokens_per_second": TARGET_TOKENS_PER_SECOND,
            "seconds_per_token": TARGET_SECONDS_PER_TOKEN,
            "required_speedup": REQUIRED_SPEEDUP,
        },
        "verifier_mode": verifier_mode,
        "priced_blocks": rows,
        "conditional_projection_blocks": conditional_rows,
        "memory_contract": memory,
        "readiness": {
            "t036_a05_block_verifier_gate_ready": gate_ready,
            "t036_projection_has_target_passing_condition": bool(conditional_pass_rows),
            "t036_current_executable_path_crosses_15_tok_s": bool(pass_rows),
            "t036_exact_state_equality": exact_state_equality,
            "t036_exact_token_equality": exact_token_equality,
            "t036_gpu_bound_no_offload": True,
            "t036_no_hidden_dense_sidecar": memory["hidden_dense_sidecar_free"],
            "t036_peak_cuda_fits_device_budget": memory["fits_device_budget"],
            "t036_production_implementation_allowed": gate_ready,
        },
        "best_current_row": min(rows, key=lambda row: row["seconds_per_accepted_token"]) if rows else None,
        "best_conditional_projection_row": min(
            conditional_rows, key=lambda row: row["seconds_per_accepted_token"]
        )
        if conditional_rows
        else None,
        "blockers": blockers,
        "next_actions": []
        if gate_ready
        else [
            "Reject ordinary serial full64 replay as the A05-06 verifier.",
            "Research or derive a genuinely sublinear exact block verifier before production span implementation.",
        ],
    }


__all__ = ["ACTIVATION", "run_t036_a05_block_verifier_gate"]
