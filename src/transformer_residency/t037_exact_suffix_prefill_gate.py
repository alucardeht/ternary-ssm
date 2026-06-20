"""T037 exact suffix-prefill verifier gate."""

from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter
from typing import Any


ACTIVATION = "T037_EXACT_SUFFIX_PREFILL_GATE_ACTIVE"

R452_FULL64_TOKENS_PER_SECOND = 0.9280330765589875
R452_FULL64_SECONDS_PER_TOKEN = 1.0775478000286967
TARGET_TOKENS_PER_SECOND = 15.0
TARGET_SECONDS_PER_TOKEN = 1.0 / TARGET_TOKENS_PER_SECOND
DEVICE_BUDGET_GIB = 12.0
R452_FULL64_PEAK_CUDA_GIB = 10.652063369750977


def _parse_blocks(value: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(int(item) for item in value)
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def _accepted_tokens_for_block(block_size: int) -> int:
    defaults = {16: 13, 32: 26, 64: 33}
    return defaults.get(block_size, max(1, int(round(block_size * 0.80))))


def _next_token(sequence: tuple[int, ...], *, vocab_size: int) -> int:
    position = len(sequence)
    weighted = sum((idx + 1) * token for idx, token in enumerate(sequence))
    return int((weighted + 17 * position + 31) % vocab_size)


def _serial_targets(prefix: tuple[int, ...], proposal: tuple[int, ...], *, vocab_size: int) -> tuple[int, ...]:
    seq = list(prefix)
    targets: list[int] = []
    for _ in proposal:
        token = _next_token(tuple(seq), vocab_size=vocab_size)
        targets.append(token)
        seq.append(token)
    return tuple(targets)


def _teacher_forced_shifted_targets(
    prefix: tuple[int, ...],
    proposal: tuple[int, ...],
    *,
    boundary_token: int,
    vocab_size: int,
) -> tuple[int, ...]:
    targets = [boundary_token]
    for index in range(1, len(proposal)):
        targets.append(_next_token(prefix + proposal[:index], vocab_size=vocab_size))
    return tuple(targets)


def _state_checksum(sequence: tuple[int, ...], *, vocab_size: int) -> int:
    checksum = 0
    for index, token in enumerate(sequence):
        checksum = (checksum * 1315423911 + (index + 1) * 2654435761 + token) % (vocab_size * 1000003)
    return checksum


def _verify_mismatch_slicing(
    *,
    prefix: tuple[int, ...],
    proposal: tuple[int, ...],
    vocab_size: int,
) -> dict[str, Any]:
    boundary = _next_token(prefix, vocab_size=vocab_size)
    shifted = _teacher_forced_shifted_targets(prefix, proposal, boundary_token=boundary, vocab_size=vocab_size)
    matches = [int(lhs == rhs) for lhs, rhs in zip(shifted, proposal, strict=True)]
    accepted = 0
    for match in matches:
        if not match:
            break
        accepted += 1
    committed = prefix + proposal[:accepted]
    serial_after_commit = prefix + _serial_targets(prefix, proposal[:accepted], vocab_size=vocab_size)
    return {
        "accepted": accepted,
        "matches": matches,
        "shifted_targets": list(shifted),
        "committed_equals_serial": committed == serial_after_commit,
        "committed_state_checksum": _state_checksum(committed, vocab_size=vocab_size),
        "serial_state_checksum": _state_checksum(serial_after_commit, vocab_size=vocab_size),
        "boundary_token": _next_token(committed, vocab_size=vocab_size),
        "serial_boundary_token": _next_token(serial_after_commit, vocab_size=vocab_size),
    }


def _toy_exactness_census(
    *,
    block_size: int,
    prefix_tokens: int,
    vocab_size: int,
) -> dict[str, Any]:
    prefix = tuple((index * 7 + 3) % vocab_size for index in range(prefix_tokens))
    exact_proposal = _serial_targets(prefix, tuple(0 for _ in range(block_size)), vocab_size=vocab_size)
    rows: list[dict[str, Any]] = []
    start = perf_counter()
    clean = _verify_mismatch_slicing(prefix=prefix, proposal=exact_proposal, vocab_size=vocab_size)
    rows.append({"forced_mismatch_index": None, **clean})
    for mismatch_index in range(block_size):
        proposal = list(exact_proposal)
        proposal[mismatch_index] = (proposal[mismatch_index] + 1) % vocab_size
        rows.append(
            {
                "forced_mismatch_index": mismatch_index,
                **_verify_mismatch_slicing(prefix=prefix, proposal=tuple(proposal), vocab_size=vocab_size),
            }
        )
    elapsed = perf_counter() - start
    all_exact = all(
        row["committed_equals_serial"]
        and row["committed_state_checksum"] == row["serial_state_checksum"]
        and row["boundary_token"] == row["serial_boundary_token"]
        for row in rows
    )
    mismatch_ok = all(
        row["accepted"] == (block_size if row["forced_mismatch_index"] is None else row["forced_mismatch_index"])
        for row in rows
    )
    return {
        "block_size": block_size,
        "prefix_tokens": prefix_tokens,
        "vocab_size": vocab_size,
        "toy_elapsed_seconds": elapsed,
        "all_committed_states_match_serial": all_exact,
        "all_first_mismatch_indices_match": mismatch_ok,
        "case_count": len(rows),
        "sample_rows": rows[: min(6, len(rows))],
    }


def _pricing_row(
    *,
    block_size: int,
    accepted_tokens: int,
    suffix_prefill_seconds: float | None,
    extra_seconds: float,
    timing_source: str,
) -> dict[str, Any]:
    target_block_budget = accepted_tokens * TARGET_SECONDS_PER_TOKEN
    serial_accepted_seconds = accepted_tokens * R452_FULL64_SECONDS_PER_TOKEN
    if suffix_prefill_seconds is None:
        seconds_per_accepted = None
        tok_s = None
        target_pass = False
        total_seconds = None
    else:
        total_seconds = suffix_prefill_seconds + extra_seconds
        seconds_per_accepted = total_seconds / max(accepted_tokens, 1)
        tok_s = 1.0 / seconds_per_accepted if seconds_per_accepted > 0 else 0.0
        target_pass = seconds_per_accepted <= TARGET_SECONDS_PER_TOKEN
    return {
        "block_size": block_size,
        "accepted_tokens": accepted_tokens,
        "timing_source": timing_source,
        "suffix_prefill_seconds": suffix_prefill_seconds,
        "extra_seconds": extra_seconds,
        "total_seconds": total_seconds,
        "target_block_budget_seconds": target_block_budget,
        "serial_accepted_seconds": serial_accepted_seconds,
        "required_speedup_vs_serial_accepted": serial_accepted_seconds / target_block_budget,
        "seconds_per_accepted_token": seconds_per_accepted,
        "tokens_per_second": tok_s,
        "target_pass": target_pass,
    }


def run_t037_exact_suffix_prefill_gate(
    *,
    block_sizes: tuple[int, ...] | list[int] | str = (16, 32, 64),
    prefix_tokens: int = 8,
    vocab_size: int = 257,
    full64_suffix_prefill_seconds: float | None = None,
    extra_seconds: float = 0.0,
    shadow_kv_gib: float = 0.0,
    scratch_gib: float = 0.0,
    graph_memory_gib: float = 0.0,
    dense_sidecar_gib: float = 0.0,
    device_budget_gib: float = DEVICE_BUDGET_GIB,
    timing_source: str = "missing_full64_measurement",
) -> dict[str, Any]:
    """Validate exact suffix-prefill semantics and price the missing full64 gate."""

    sizes = _parse_blocks(block_sizes)
    exactness = [
        _toy_exactness_census(block_size=block_size, prefix_tokens=prefix_tokens, vocab_size=vocab_size)
        for block_size in sizes
    ]
    all_exact = all(
        item["all_committed_states_match_serial"] and item["all_first_mismatch_indices_match"] for item in exactness
    )
    pricing = [
        _pricing_row(
            block_size=block_size,
            accepted_tokens=_accepted_tokens_for_block(block_size),
            suffix_prefill_seconds=full64_suffix_prefill_seconds,
            extra_seconds=extra_seconds,
            timing_source=timing_source,
        )
        for block_size in sizes
    ]
    pass_rows = [row for row in pricing if row["target_pass"]]
    extra_peak = shadow_kv_gib + scratch_gib + graph_memory_gib + dense_sidecar_gib
    projected_peak = R452_FULL64_PEAK_CUDA_GIB + extra_peak
    memory_ok = projected_peak <= device_budget_gib and dense_sidecar_gib == 0.0

    blockers: list[str] = []
    if not all_exact:
        blockers.append("suffix_prefill_exactness_failed")
    if full64_suffix_prefill_seconds is None:
        blockers.append("full64_suffix_prefill_timing_missing")
    if not pass_rows:
        blockers.append("suffix_prefill_pricing_does_not_reach_15_tok_s")
    if dense_sidecar_gib > 0.0:
        blockers.append("hidden_dense_sidecar_live")
    if projected_peak > device_budget_gib:
        blockers.append("projected_peak_cuda_exceeds_device_budget")

    gate_ready = all_exact and bool(pass_rows) and memory_ok and timing_source == "full64_measured"
    return {
        "probe": "t037_exact_suffix_prefill_gate",
        "activation": ACTIVATION,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "baseline": {
            "source": "R452/T028",
            "tokens_per_second": R452_FULL64_TOKENS_PER_SECOND,
            "seconds_per_token": R452_FULL64_SECONDS_PER_TOKEN,
        },
        "target": {
            "tokens_per_second": TARGET_TOKENS_PER_SECOND,
            "seconds_per_token": TARGET_SECONDS_PER_TOKEN,
        },
        "exactness_census": exactness,
        "pricing": pricing,
        "memory_contract": {
            "baseline_peak_cuda_gib": R452_FULL64_PEAK_CUDA_GIB,
            "shadow_kv_gib": shadow_kv_gib,
            "scratch_gib": scratch_gib,
            "graph_memory_gib": graph_memory_gib,
            "dense_sidecar_gib": dense_sidecar_gib,
            "projected_peak_cuda_gib": projected_peak,
            "device_budget_gib": device_budget_gib,
            "fits_device_budget": projected_peak <= device_budget_gib,
            "hidden_dense_sidecar_free": dense_sidecar_gib == 0.0,
        },
        "readiness": {
            "t037_exact_suffix_prefill_gate_ready": gate_ready,
            "t037_suffix_prefill_exactness_ready": all_exact,
            "t037_full64_timing_measured": timing_source == "full64_measured",
            "t037_pricing_crosses_15_tok_s": bool(pass_rows),
            "t037_gpu_bound_no_offload": True,
            "t037_no_hidden_dense_sidecar": dense_sidecar_gib == 0.0,
            "t037_production_implementation_allowed": gate_ready,
        },
        "best_pricing_row": min(
            pricing,
            key=lambda row: row["seconds_per_accepted_token"]
            if row["seconds_per_accepted_token"] is not None
            else float("inf"),
        )
        if pricing
        else None,
        "blockers": blockers,
        "next_actions": []
        if gate_ready
        else [
            "Implement or measure a real full64 suffix-prefill verifier path before production promotion.",
            "If full64 suffix prefill cannot beat 2.2s for B=64 accepted=33, abandon the A05-06 verifier frame.",
        ],
    }


__all__ = ["ACTIVATION", "run_t037_exact_suffix_prefill_gate"]
