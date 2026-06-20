#!/usr/bin/env python3
"""R502 E3 phase-area execution segmentation gate.

This CPU diagnostic implements the R502 Efficiency finding:
"Phase-Area Execution Segmentation (PAES)".

The gate uses the real src/model_v12/test_data.bin length to price a synthetic
recurrent trajectory. The dense baseline materializes all layer/time/state
cells. PAES streams the same recurrence phase by phase, keeps only phase
boundaries plus a small ring window, and must produce the identical synthetic
objective checksum and final state while keeping peak memory strictly below the
1/S segmentation bound.

Controls exercise the declared risks: missing phase boundaries changes the
trajectory, retaining a full segment buffer violates the strict memory bound,
segment synchronization above 10% kills the denominator, and release debt from
previous phases can erase the peak-memory gain.

This is a synthetic CPU segmentation and peak-memory diagnostic, not heldout CE
progress, real corpus token efficiency, GPU training-speed evidence, or
production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r502_e3_paes_gate.v1"
CONDITIONS = (
    "positive",
    "missing_boundary_fail",
    "full_segment_buffer_fail",
    "sync_overhead_fail",
    "release_debt_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    dataset_bytes: int
    sequence_len: int
    layers: int
    state_dim: int
    segments: int
    segment_len_max: int
    active_window: int
    objective_checksum_abs_delta: int
    final_state_linf_max: int
    baseline_vram_bytes: int
    paes_peak_vram_bytes: int
    peak_vram_ratio: float
    segment_bound_ratio: float
    strict_peak_margin: float
    release_debt_cells: int
    sync_overhead_ratio: float
    effective_time_ratio: float
    projected_peak_vram_saving_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("src/model_v12/test_data.bin"))
    parser.add_argument("--seed", type=int, default=50203)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--active-window", type=int, default=128)
    parser.add_argument("--bytes-per-state", type=int, default=4)
    parser.add_argument("--max-sync-overhead-ratio", type=float, default=0.10)
    parser.add_argument("--sync-overhead-ratio", type=float, default=0.04)
    parser.add_argument("--expensive-sync-overhead-ratio", type=float, default=0.125)
    return parser.parse_args()


def load_tokens(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if not data:
        raise ValueError(f"dataset is empty: {path}")
    return np.frombuffer(data, dtype=np.uint8).astype(np.int64)


def make_problem(args: argparse.Namespace, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diag = rng.choice(
        np.array([-1, 0, 1], dtype=np.int64),
        size=(args.layers, args.state_dim),
        p=np.array([0.38, 0.24, 0.38], dtype=np.float64),
    )
    layer_offsets = rng.integers(-5, 6, size=(args.layers, 1), dtype=np.int64)
    dim_offsets = rng.integers(-7, 8, size=(1, args.state_dim), dtype=np.int64)
    readout = rng.integers(-2, 3, size=(args.layers, args.state_dim), dtype=np.int64)
    branch_offsets = layer_offsets + dim_offsets
    return diag, branch_offsets, readout


def token_branch(token: int, branch_offsets: np.ndarray) -> np.ndarray:
    return ((token + branch_offsets) % 17) - 8


def dense_execute(
    tokens: np.ndarray,
    diag: np.ndarray,
    branch_offsets: np.ndarray,
    readout: np.ndarray,
) -> tuple[np.ndarray, int]:
    sequence_len = int(tokens.shape[0])
    layers, state_dim = diag.shape
    states = np.zeros((sequence_len, layers, state_dim), dtype=np.int64)
    state = np.zeros((layers, state_dim), dtype=np.int64)
    checksum = 0
    for t, token in enumerate(tokens):
        state = diag * state + token_branch(int(token), branch_offsets)
        states[t] = state
        checksum += int(np.sum(state * readout))
    return states, checksum


def segmented_execute(
    tokens: np.ndarray,
    diag: np.ndarray,
    branch_offsets: np.ndarray,
    readout: np.ndarray,
    segments: int,
    condition: str,
) -> tuple[np.ndarray, int, int]:
    sequence_len = int(tokens.shape[0])
    layers, state_dim = diag.shape
    boundaries = np.zeros((segments + 1, layers, state_dim), dtype=np.int64)
    checksum = 0
    segment_len_max = 0
    for segment_idx in range(segments):
        start = (segment_idx * sequence_len) // segments
        end = ((segment_idx + 1) * sequence_len) // segments
        segment_len_max = max(segment_len_max, end - start)
        if condition == "missing_boundary_fail" and segment_idx > 0:
            state = np.zeros((layers, state_dim), dtype=np.int64)
        else:
            state = boundaries[segment_idx].copy()
        for token in tokens[start:end]:
            state = diag * state + token_branch(int(token), branch_offsets)
            checksum += int(np.sum(state * readout))
        boundaries[segment_idx + 1] = state
    return boundaries[-1], checksum, segment_len_max


def memory_price(
    args: argparse.Namespace,
    sequence_len: int,
    segment_len_max: int,
    condition: str,
) -> tuple[int, int, int, float, float, float, float]:
    layer_state_cells = args.layers * args.state_dim
    baseline_cells = sequence_len * layer_state_cells
    boundary_cells = (args.segments + 1) * layer_state_cells
    active_cells = args.active_window * layer_state_cells
    release_debt_cells = 0
    if condition == "full_segment_buffer_fail":
        active_cells = segment_len_max * layer_state_cells
    elif condition == "release_debt_fail":
        active_cells = segment_len_max * layer_state_cells
        release_debt_cells = segment_len_max * layer_state_cells

    peak_cells = boundary_cells + active_cells + release_debt_cells
    baseline_bytes = baseline_cells * args.bytes_per_state
    peak_bytes = peak_cells * args.bytes_per_state
    peak_ratio = peak_bytes / baseline_bytes
    segment_bound_ratio = 1.0 / float(args.segments)
    strict_margin = segment_bound_ratio - peak_ratio
    projected_multiplier = 1.0 / peak_ratio
    return (
        baseline_bytes,
        peak_bytes,
        release_debt_cells,
        peak_ratio,
        segment_bound_ratio,
        strict_margin,
        projected_multiplier,
    )


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def run_trial(args: argparse.Namespace, tokens: np.ndarray, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    diag, branch_offsets, readout = make_problem(args, rng)
    dense_states, dense_checksum = dense_execute(tokens, diag, branch_offsets, readout)
    segmented_final, segmented_checksum, segment_len_max = segmented_execute(
        tokens,
        diag,
        branch_offsets,
        readout,
        args.segments,
        condition,
    )
    dense_final = dense_states[-1]
    final_delta = segmented_final - dense_final
    final_linf = int(np.max(np.abs(final_delta)))
    objective_delta = int(abs(segmented_checksum - dense_checksum))
    (
        baseline_bytes,
        peak_bytes,
        release_debt_cells,
        peak_ratio,
        segment_bound_ratio,
        strict_margin,
        projected_multiplier,
    ) = memory_price(args, int(tokens.shape[0]), segment_len_max, condition)
    sync_overhead = (
        args.expensive_sync_overhead_ratio
        if condition == "sync_overhead_fail"
        else args.sync_overhead_ratio
    )
    effective_time_ratio = 1.0 + sync_overhead

    failed: list[str] = []
    if not finite(peak_ratio, segment_bound_ratio, strict_margin, sync_overhead, effective_time_ratio, projected_multiplier):
        failed.append("non_finite_metric")
    if objective_delta != 0 or final_linf != 0:
        failed.append("execution_identity")
    if not strict_margin > 0.0:
        failed.append("peak_vram_bound")
    if sync_overhead > args.max_sync_overhead_ratio:
        failed.append("sync_overhead")
    if release_debt_cells > 0:
        failed.append("release_debt")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed) if failed else None,
        dataset_bytes=int(tokens.shape[0]),
        sequence_len=int(tokens.shape[0]),
        layers=args.layers,
        state_dim=args.state_dim,
        segments=args.segments,
        segment_len_max=segment_len_max,
        active_window=args.active_window,
        objective_checksum_abs_delta=objective_delta,
        final_state_linf_max=final_linf,
        baseline_vram_bytes=baseline_bytes,
        paes_peak_vram_bytes=peak_bytes,
        peak_vram_ratio=float(peak_ratio),
        segment_bound_ratio=float(segment_bound_ratio),
        strict_peak_margin=float(strict_margin),
        release_debt_cells=release_debt_cells,
        sync_overhead_ratio=float(sync_overhead),
        effective_time_ratio=float(effective_time_ratio),
        projected_peak_vram_saving_multiplier=float(projected_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "dataset_bytes",
        "sequence_len",
        "layers",
        "state_dim",
        "segments",
        "segment_len_max",
        "active_window",
        "objective_checksum_abs_delta",
        "final_state_linf_max",
        "baseline_vram_bytes",
        "paes_peak_vram_bytes",
        "peak_vram_ratio",
        "segment_bound_ratio",
        "strict_peak_margin",
        "release_debt_cells",
        "sync_overhead_ratio",
        "effective_time_ratio",
        "projected_peak_vram_saving_multiplier",
    )
    aggregate: dict[str, Any] = {
        "count": len(results),
        "passed_count": sum(result.passed for result in results),
        "identifying_correct_count": sum(result.identifying_correct for result in results),
        "fail_reasons": sorted({result.fail_reason for result in results if result.fail_reason}),
    }
    for field in fields:
        values = np.array([float(getattr(result, field)) for result in results], dtype=np.float64)
        aggregate[f"{field}_mean"] = float(np.mean(values))
        aggregate[f"{field}_min"] = float(np.min(values))
        aggregate[f"{field}_max"] = float(np.max(values))
    return aggregate


def all_finite(results: list[TrialResult]) -> bool:
    for result in results:
        for value in asdict(result).values():
            if isinstance(value, float) and not math.isfinite(value):
                return False
    return True


def required_checks(results: list[TrialResult], aggregates: dict[str, Any]) -> dict[str, bool]:
    positive = [result for result in results if result.condition == "positive"]
    missing = [result for result in results if result.condition == "missing_boundary_fail"]
    full = [result for result in results if result.condition == "full_segment_buffer_fail"]
    sync = [result for result in results if result.condition == "sync_overhead_fail"]
    release = [result for result in results if result.condition == "release_debt_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_execution_identity": (
            aggregates["positive"]["objective_checksum_abs_delta_max"] == 0.0
            and aggregates["positive"]["final_state_linf_max_max"] == 0.0
        ),
        "positive_condition_peak_vram_below_segment_bound": (
            aggregates["positive"]["peak_vram_ratio_max"]
            < aggregates["positive"]["segment_bound_ratio_min"]
        ),
        "positive_condition_sync_overhead_priced": (
            aggregates["positive"]["sync_overhead_ratio_max"] <= 0.10
        ),
        "missing_boundary_control_all_fail_identity": all(
            (not result.passed)
            and result.fail_reason is not None
            and "execution_identity" in result.fail_reason
            for result in missing
        ),
        "full_segment_control_all_fail_peak_bound": all(
            (not result.passed)
            and result.fail_reason is not None
            and "peak_vram_bound" in result.fail_reason
            for result in full
        ),
        "sync_control_all_fail_overhead": all(
            (not result.passed)
            and result.fail_reason is not None
            and "sync_overhead" in result.fail_reason
            for result in sync
        ),
        "release_debt_control_all_fail_peak_or_debt": all(
            (not result.passed)
            and result.fail_reason is not None
            and ("peak_vram_bound" in result.fail_reason or "release_debt" in result.fail_reason)
            for result in release
        ),
        "all_conditions_identified": all(result.identifying_correct for result in results),
    }


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.layers < 1:
        raise ValueError("--layers must be at least 1")
    if args.state_dim < 1:
        raise ValueError("--state-dim must be at least 1")
    if args.segments < 2:
        raise ValueError("--segments must be at least 2")
    if args.active_window < 1:
        raise ValueError("--active-window must be at least 1")

    tokens = load_tokens(args.dataset)
    results: list[TrialResult] = []
    for condition in CONDITIONS:
        for offset in range(args.seed_count):
            results.append(run_trial(args, tokens, condition, args.seed + offset))

    aggregates = {
        condition: aggregate_condition(
            [result for result in results if result.condition == condition]
        )
        for condition in CONDITIONS
    }
    checks = required_checks(results, aggregates)
    decision = "PASS_R502_E3_PAES_GATE" if all(checks.values()) else "FAIL_R502_E3_PAES_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output), "dataset": str(args.dataset)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU phase-area execution segmentation gate only; not "
            "heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_E3_PAES_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_dataset_bytes_mean={positive['dataset_bytes_mean']:.0f} "
        f"positive_objective_checksum_abs_delta_max={positive['objective_checksum_abs_delta_max']:.0f} "
        f"positive_final_state_linf_max_max={positive['final_state_linf_max_max']:.0f} "
        f"positive_baseline_vram_bytes_mean={positive['baseline_vram_bytes_mean']:.0f} "
        f"positive_paes_peak_vram_bytes_mean={positive['paes_peak_vram_bytes_mean']:.0f} "
        f"positive_peak_vram_ratio_mean={positive['peak_vram_ratio_mean']:.8f} "
        f"positive_segment_bound_ratio_mean={positive['segment_bound_ratio_mean']:.8f} "
        f"positive_sync_overhead_ratio_mean={positive['sync_overhead_ratio_mean']:.6f} "
        f"positive_projected_peak_vram_saving_multiplier_mean={positive['projected_peak_vram_saving_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
