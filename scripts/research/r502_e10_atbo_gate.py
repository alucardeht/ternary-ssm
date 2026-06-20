#!/usr/bin/env python3
"""R502 E10 asynchronous token-boundary offloading gate.

This CPU diagnostic implements the R502 Efficiency finding:
"Asynchronous Token-Boundary Offloading (ATBO)".

The gate builds a synthetic recurrent forward path, offloads exact token
segments to a host-resident stream at interval K, and verifies that the replayed
states preserve the objective while VRAM is bounded by the active segment. The
transfer model prices the stated inequality T_compute(K) >= T_transfer(K) and
the declared throughput-drop gate.

Controls exercise the declared risks: PCIe latency can outpace computation,
lost token-boundary segments break exact replay, tiny K can lose to fixed async
overhead, and insufficient host RAM can move the bottleneck instead of reducing
it.

This is a synthetic CPU offload-scheduler and objective-identity diagnostic,
not heldout CE progress, real corpus token efficiency, GPU training-speed
evidence, Vulkan DMA evidence, or production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r502_e10_atbo_gate.v1"
CONDITIONS = (
    "positive",
    "pcie_latency_fail",
    "missing_boundary_fail",
    "small_k_overhead_fail",
    "host_ram_pressure_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    sequence_len: int
    state_dim: int
    boundary_interval: int
    transfer_count: int
    objective_abs_delta: float
    replay_objective_abs_delta: float
    final_state_linf: float
    replay_state_linf: float
    compute_time_us_total: float
    transfer_time_us_max: float
    transfer_hidden_margin_min: float
    async_overhead_fraction: float
    throughput_drop_fraction: float
    dense_vram_bytes: float
    atbo_vram_bytes: float
    vram_ratio: float
    host_bytes: float
    host_budget_ratio: float
    projected_vram_saving_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50210)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sequence-len", type=int, default=4096)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--boundary-interval", type=int, default=128)
    parser.add_argument("--small-boundary-interval", type=int, default=8)
    parser.add_argument("--bytes-per-state-value", type=int, default=4)
    parser.add_argument("--compute-us-per-token", type=float, default=1.2)
    parser.add_argument("--pcie-gb-per-second", type=float, default=24.0)
    parser.add_argument("--pcie-latency-us", type=float, default=8.0)
    parser.add_argument("--bad-pcie-gb-per-second", type=float, default=2.0)
    parser.add_argument("--bad-pcie-latency-us", type=float, default=600.0)
    parser.add_argument("--async-overhead-us-per-transfer", type=float, default=0.8)
    parser.add_argument("--staging-buffer-multiplier", type=float, default=1.15)
    parser.add_argument("--host-ram-budget-multiplier", type=float, default=2.0)
    parser.add_argument("--bad-host-ram-budget-multiplier", type=float, default=0.5)
    parser.add_argument("--max-objective-abs-delta", type=float, default=1.0e-12)
    parser.add_argument("--max-state-linf", type=float, default=1.0e-12)
    parser.add_argument("--max-throughput-drop-fraction", type=float, default=0.05)
    parser.add_argument("--max-vram-ratio", type=float, default=0.20)
    parser.add_argument("--max-host-budget-ratio", type=float, default=1.0)
    return parser.parse_args()


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def build_dense_path(
    sequence_len: int,
    state_dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray]:
    weights = rng.normal(loc=0.0, scale=0.035, size=(state_dim,))
    inputs = rng.normal(loc=0.0, scale=0.07, size=(sequence_len, state_dim))
    h = rng.normal(loc=0.0, scale=0.02, size=(state_dim,))
    states = np.empty((sequence_len, state_dim), dtype=np.float64)
    objective = 0.0
    for idx in range(sequence_len):
        h = np.tanh(0.91 * h + inputs[idx])
        states[idx] = h
        objective += float(np.dot(h, weights))
    return states, objective / float(sequence_len), weights


def stream_atbo(
    dense_states: np.ndarray,
    weights: np.ndarray,
    boundary_interval: int,
    condition: str,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    sequence_len, state_dim = dense_states.shape
    host_segments: list[np.ndarray] = []
    segment: list[np.ndarray] = []
    objective = 0.0
    missing_segment_index = 1
    segment_index = 0

    for idx in range(sequence_len):
        state = dense_states[idx].copy()
        segment.append(state)
        objective += float(np.dot(state, weights))
        at_boundary = len(segment) == boundary_interval or idx == sequence_len - 1
        if at_boundary:
            segment_array = np.stack(segment, axis=0)
            if condition == "missing_boundary_fail" and segment_index == missing_segment_index:
                segment_array = np.zeros_like(segment_array)
            host_segments.append(segment_array)
            segment = []
            segment_index += 1

    replay_states = np.concatenate(host_segments, axis=0)
    replay_objective = float(np.sum(replay_states @ weights) / float(sequence_len))
    return replay_states, objective / float(sequence_len), replay_objective, segment_index


def transfer_metrics(args: argparse.Namespace, boundary_interval: int, condition: str) -> tuple[float, float, float, float]:
    pcie_gb_per_second = (
        args.bad_pcie_gb_per_second
        if condition == "pcie_latency_fail"
        else args.pcie_gb_per_second
    )
    pcie_latency_us = (
        args.bad_pcie_latency_us
        if condition == "pcie_latency_fail"
        else args.pcie_latency_us
    )
    bytes_per_us = pcie_gb_per_second * 1000.0
    segment_bytes = boundary_interval * args.state_dim * args.bytes_per_state_value
    compute_us = boundary_interval * args.compute_us_per_token
    transfer_us = pcie_latency_us + (segment_bytes / bytes_per_us)
    hidden_margin = (compute_us - transfer_us) / max(compute_us, 1.0e-30)
    overrun_us = max(0.0, transfer_us - compute_us)
    return compute_us, transfer_us, hidden_margin, overrun_us


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    if condition == "small_k_overhead_fail":
        boundary_interval = args.small_boundary_interval
    else:
        boundary_interval = args.boundary_interval

    rng = np.random.default_rng(seed)
    dense_states, dense_objective, weights = build_dense_path(
        args.sequence_len,
        args.state_dim,
        rng,
    )
    replay_states, atbo_objective, replay_objective, transfer_count = stream_atbo(
        dense_states,
        weights,
        boundary_interval,
        condition,
    )
    objective_abs_delta = abs(atbo_objective - dense_objective)
    replay_objective_abs_delta = abs(replay_objective - dense_objective)
    final_state_linf = float(np.max(np.abs(replay_states[-1] - dense_states[-1])))
    replay_state_linf = float(np.max(np.abs(replay_states - dense_states)))

    compute_us_per_segment, transfer_us, hidden_margin, overrun_us = transfer_metrics(
        args,
        boundary_interval,
        condition,
    )
    compute_time_us_total = args.sequence_len * args.compute_us_per_token
    async_overhead_fraction = (
        transfer_count * args.async_overhead_us_per_transfer / compute_time_us_total
    )
    throughput_drop_fraction = (
        transfer_count * overrun_us / compute_time_us_total
    ) + async_overhead_fraction

    dense_vram_bytes = float(
        args.sequence_len * args.state_dim * args.bytes_per_state_value
    )
    active_segment_bytes = float(
        boundary_interval * args.state_dim * args.bytes_per_state_value
    )
    atbo_vram_bytes = active_segment_bytes * (1.0 + args.staging_buffer_multiplier)
    vram_ratio = atbo_vram_bytes / dense_vram_bytes
    host_bytes = dense_vram_bytes
    budget_multiplier = (
        args.bad_host_ram_budget_multiplier
        if condition == "host_ram_pressure_fail"
        else args.host_ram_budget_multiplier
    )
    host_budget_ratio = host_bytes / max(host_bytes * budget_multiplier, 1.0e-30)
    projected_multiplier = 1.0 / vram_ratio

    failed: list[str] = []
    if not finite(
        objective_abs_delta,
        replay_objective_abs_delta,
        final_state_linf,
        replay_state_linf,
        compute_time_us_total,
        transfer_us,
        hidden_margin,
        async_overhead_fraction,
        throughput_drop_fraction,
        dense_vram_bytes,
        atbo_vram_bytes,
        vram_ratio,
        host_bytes,
        host_budget_ratio,
        projected_multiplier,
        compute_us_per_segment,
    ):
        failed.append("non_finite_metric")
    if objective_abs_delta > args.max_objective_abs_delta:
        failed.append("forward_objective_identity")
    if replay_objective_abs_delta > args.max_objective_abs_delta:
        failed.append("replay_identity")
    if final_state_linf > args.max_state_linf or replay_state_linf > args.max_state_linf:
        failed.append("state_identity")
    if hidden_margin < 0.0:
        failed.append("transfer_hidden")
    if throughput_drop_fraction > args.max_throughput_drop_fraction:
        failed.append("throughput_drop")
    if vram_ratio > args.max_vram_ratio:
        failed.append("vram_price")
    if host_budget_ratio > args.max_host_budget_ratio:
        failed.append("host_budget")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed) if failed else None,
        sequence_len=args.sequence_len,
        state_dim=args.state_dim,
        boundary_interval=boundary_interval,
        transfer_count=transfer_count,
        objective_abs_delta=objective_abs_delta,
        replay_objective_abs_delta=replay_objective_abs_delta,
        final_state_linf=final_state_linf,
        replay_state_linf=replay_state_linf,
        compute_time_us_total=compute_time_us_total,
        transfer_time_us_max=transfer_us,
        transfer_hidden_margin_min=hidden_margin,
        async_overhead_fraction=async_overhead_fraction,
        throughput_drop_fraction=throughput_drop_fraction,
        dense_vram_bytes=dense_vram_bytes,
        atbo_vram_bytes=atbo_vram_bytes,
        vram_ratio=vram_ratio,
        host_bytes=host_bytes,
        host_budget_ratio=host_budget_ratio,
        projected_vram_saving_multiplier=projected_multiplier,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "sequence_len",
        "state_dim",
        "boundary_interval",
        "transfer_count",
        "objective_abs_delta",
        "replay_objective_abs_delta",
        "final_state_linf",
        "replay_state_linf",
        "compute_time_us_total",
        "transfer_time_us_max",
        "transfer_hidden_margin_min",
        "async_overhead_fraction",
        "throughput_drop_fraction",
        "dense_vram_bytes",
        "atbo_vram_bytes",
        "vram_ratio",
        "host_bytes",
        "host_budget_ratio",
        "projected_vram_saving_multiplier",
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
    pcie = [result for result in results if result.condition == "pcie_latency_fail"]
    missing = [result for result in results if result.condition == "missing_boundary_fail"]
    small_k = [result for result in results if result.condition == "small_k_overhead_fail"]
    host = [result for result in results if result.condition == "host_ram_pressure_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_objective_and_replay": (
            aggregates["positive"]["objective_abs_delta_max"] <= 1.0e-12
            and aggregates["positive"]["replay_objective_abs_delta_max"] <= 1.0e-12
            and aggregates["positive"]["replay_state_linf_max"] <= 1.0e-12
        ),
        "positive_condition_transfer_hidden_and_throughput_priced": (
            aggregates["positive"]["transfer_hidden_margin_min_min"] >= 0.0
            and aggregates["positive"]["throughput_drop_fraction_max"] <= 0.05
        ),
        "positive_condition_vram_and_host_priced": (
            aggregates["positive"]["vram_ratio_mean"] <= 0.20
            and aggregates["positive"]["host_budget_ratio_mean"] <= 1.0
        ),
        "pcie_latency_control_all_fail_transfer_or_throughput": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "transfer_hidden" in result.fail_reason
                or "throughput_drop" in result.fail_reason
            )
            for result in pcie
        ),
        "missing_boundary_control_all_fail_replay": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "replay_identity" in result.fail_reason
                or "state_identity" in result.fail_reason
            )
            for result in missing
        ),
        "small_k_control_all_fail_throughput": all(
            (not result.passed)
            and result.fail_reason is not None
            and "throughput_drop" in result.fail_reason
            for result in small_k
        ),
        "host_pressure_control_all_fail_host_budget": all(
            (not result.passed)
            and result.fail_reason is not None
            and "host_budget" in result.fail_reason
            for result in host
        ),
        "all_conditions_identified": all(result.identifying_correct for result in results),
    }


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.sequence_len < 2:
        raise ValueError("--sequence-len must be at least 2")
    if args.state_dim < 1:
        raise ValueError("--state-dim must be positive")
    if args.boundary_interval < 1 or args.small_boundary_interval < 1:
        raise ValueError("boundary intervals must be positive")
    if args.sequence_len % args.boundary_interval != 0:
        raise ValueError("--sequence-len must be divisible by --boundary-interval")
    if args.sequence_len % args.small_boundary_interval != 0:
        raise ValueError("--sequence-len must be divisible by --small-boundary-interval")

    results: list[TrialResult] = []
    for condition in CONDITIONS:
        for offset in range(args.seed_count):
            results.append(run_trial(args, condition, args.seed + offset))

    aggregates = {
        condition: aggregate_condition(
            [result for result in results if result.condition == condition]
        )
        for condition in CONDITIONS
    }
    checks = required_checks(results, aggregates)
    decision = "PASS_R502_E10_ATBO_GATE" if all(checks.values()) else "FAIL_R502_E10_ATBO_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU token-boundary offload-scheduler gate only; not "
            "heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, not Vulkan DMA evidence, and not production v12 "
            "integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_E10_ATBO_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_boundary_interval_mean={positive['boundary_interval_mean']:.6f} "
        f"positive_objective_abs_delta_max={positive['objective_abs_delta_max']:.12e} "
        f"positive_replay_state_linf_max={positive['replay_state_linf_max']:.12e} "
        f"positive_transfer_hidden_margin_min={positive['transfer_hidden_margin_min_min']:.12e} "
        f"positive_throughput_drop_fraction_max={positive['throughput_drop_fraction_max']:.12e} "
        f"positive_vram_ratio_mean={positive['vram_ratio_mean']:.6f} "
        f"positive_host_budget_ratio_mean={positive['host_budget_ratio_mean']:.6f} "
        f"positive_projected_vram_saving_multiplier_mean={positive['projected_vram_saving_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
