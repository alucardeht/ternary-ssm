#!/usr/bin/env python3
"""R426 theoretical performance gates.

This script implements the first executable gates from R426:

1. function-preserving staged width / structural dormant birth;
2. reachability-null inactive cone.

The gates are CPU-only proof-object checks. They do not modify production
training, do not use GPU, and do not claim measured full-training speedup.
Their job is to decide whether the R426 math can move from research synthesis
to a real implementation path without hiding dense work.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


DATASET_MAGIC = 0x544F4B4E
DATASET_HEADER_BYTES = 16
SCHEMA = "r426_theoretical_performance_gates.v1"


@dataclass
class GateResult:
    gate: str
    decision: str
    status: str
    primary_metric: float
    priced_ratio: float
    projected_speedup_if_promoted: float
    activation_log: str
    metrics: dict[str, float | int | str]


@dataclass
class SuiteResult:
    schema: str
    decision: str
    dataset: str | None
    tokens_used: int
    no_gpu: int
    no_training_integration: int
    activation_log: str
    gates: list[GateResult]
    elapsed_seconds: float


def read_v12_tokens(path: Path | None, max_tokens: int, seed: int) -> np.ndarray:
    if path is None:
        rng = np.random.default_rng(seed)
        base = np.arange(max_tokens, dtype=np.int64) % 997
        noise = rng.integers(0, 127, size=max_tokens, dtype=np.int64)
        return ((base * 37 + noise) % 4096).astype(np.int64)

    data = path.read_bytes()
    offset = 0
    n_tokens = len(data) // 2
    if len(data) >= DATASET_HEADER_BYTES:
        magic, version, header_tokens = struct.unpack_from("<IIQ", data, 0)
        if magic == DATASET_MAGIC:
            if version != 1:
                raise ValueError(f"unsupported dataset version: {version}")
            offset = DATASET_HEADER_BYTES
            n_tokens = int(header_tokens)
    available = (len(data) - offset) // 2
    count = min(n_tokens, available, max_tokens)
    if count <= 32:
        raise ValueError("dataset does not contain enough tokens for R426 gates")
    return np.frombuffer(data, dtype="<u2", count=count, offset=offset).astype(np.int64)


def token_features(tokens: np.ndarray, rows: int, width: int) -> np.ndarray:
    idx = np.arange(rows, dtype=np.float64)
    toks = tokens[:rows].astype(np.float64)
    cols = []
    for j in range(width):
        phase = (j + 1) * 0.017
        cols.append(np.sin((toks + 3.0 * j) * phase) + np.cos((idx + toks % 31.0) * phase * 0.7))
    return np.stack(cols, axis=1)


def parse_phase_schedule(text: str) -> list[tuple[float, float]]:
    phases: list[tuple[float, float]] = []
    for raw in text.split(","):
        part = raw.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"phase '{part}' must use width_fraction:duration_fraction")
        width_text, duration_text = part.split(":", 1)
        width = float(width_text)
        duration = float(duration_text)
        if not (0.0 < width <= 1.0):
            raise ValueError(f"invalid width fraction {width}")
        if not (0.0 < duration <= 1.0):
            raise ValueError(f"invalid duration fraction {duration}")
        phases.append((width, duration))
    total = sum(duration for _, duration in phases)
    if not phases or not math.isfinite(total) or total <= 0.0:
        raise ValueError("phase schedule is empty")
    return [(width, duration / total) for width, duration in phases]


def gate_result(
    gate: str,
    pass_gate: bool,
    status: str,
    primary_metric: float,
    priced_ratio: float,
    metrics: dict[str, float | int | str],
) -> GateResult:
    if not np.isfinite(primary_metric) or not np.isfinite(priced_ratio):
        decision = "INVALID_PROBE_NONFINITE"
    else:
        decision = f"PASS_{gate.upper()}" if pass_gate else f"FAIL_{gate.upper()}"
    speedup = 1.0 / max(priced_ratio, 1e-12) if np.isfinite(priced_ratio) else 0.0
    activation = (
        f"[R426_THEORETICAL_PERFORMANCE_GATE] active=1 gate={gate} "
        f"decision={decision} no_gpu=1 no_training_integration=1"
    )
    return GateResult(
        gate=gate,
        decision=decision,
        status=status,
        primary_metric=float(primary_metric),
        priced_ratio=float(priced_ratio),
        projected_speedup_if_promoted=float(speedup),
        activation_log=activation,
        metrics=metrics,
    )


def staged_width_structural_birth_gate(
    rng: np.random.Generator,
    tokens: np.ndarray,
    phase_schedule: list[tuple[float, float]],
    recurrent_share: float,
    birth_overhead: float,
    release_overhead: float,
    anchor_overhead: float,
    repair_overhead: float,
    catchup_overhead: float,
    tolerance: float,
) -> GateResult:
    old_dim = 12
    new_dim = 5
    inp_dim = 7
    out_dim = 6
    steps = min(192, tokens.size - 1)

    a_old = np.diag(rng.uniform(0.35, 0.82, size=old_dim))
    # Add small lower-triangular mixing while preserving stability.
    mix = np.tril(rng.normal(0.0, 0.025, size=(old_dim, old_dim)), k=-1)
    a_old = a_old + mix
    b_old = rng.normal(0.0, 0.30, size=(old_dim, inp_dim))
    c_old = rng.normal(0.0, 0.25, size=(out_dim, old_dim))

    born_dim = old_dim + new_dim
    a_born = np.zeros((born_dim, born_dim))
    a_born[:old_dim, :old_dim] = a_old
    a_born[old_dim:, old_dim:] = np.diag(rng.uniform(0.20, 0.65, size=new_dim))
    b_born = np.zeros((born_dim, inp_dim))
    b_born[:old_dim, :] = b_old
    c_born = np.zeros((out_dim, born_dim))
    c_born[:, :old_dim] = c_old

    xs = token_features(tokens, steps, inp_dim)
    h_old = np.zeros(old_dim)
    h_born = np.zeros(born_dim)
    max_state_error = 0.0
    max_logit_error = 0.0
    for x in xs:
        h_old = a_old @ h_old + b_old @ x
        h_born = a_born @ h_born + b_born @ x
        old_logits = c_old @ h_old
        born_logits = c_born @ h_born
        max_state_error = max(max_state_error, float(np.max(np.abs(h_old - h_born[:old_dim]))))
        max_logit_error = max(max_logit_error, float(np.max(np.abs(old_logits - born_logits))))

    recurrent_share = min(max(recurrent_share, 0.0), 1.0)
    phase_area = 0.0
    min_width = 1.0
    full_width_seen = 0
    for width, duration in phase_schedule:
        min_width = min(min_width, width)
        full_width_seen = max(full_width_seen, int(abs(width - 1.0) < 1e-12))
        area = recurrent_share * width * width + (1.0 - recurrent_share) * width
        phase_area += duration * area
    overhead = birth_overhead + release_overhead + anchor_overhead + repair_overhead + catchup_overhead
    priced_ratio = phase_area + overhead
    saved_fraction = 1.0 - priced_ratio
    pass_gate = (
        max_state_error <= tolerance
        and max_logit_error <= tolerance
        and priced_ratio <= 0.90
        and min_width < 1.0
        and full_width_seen == 1
    )
    return gate_result(
        "staged_width_structural_birth_gate",
        pass_gate,
        "validated_for_gate",
        saved_fraction,
        priced_ratio,
        {
            "old_dim": old_dim,
            "new_dim": new_dim,
            "steps": steps,
            "max_state_error": max_state_error,
            "max_logit_error": max_logit_error,
            "phase_area_ratio": phase_area,
            "birth_overhead": birth_overhead,
            "release_overhead": release_overhead,
            "anchor_overhead": anchor_overhead,
            "repair_overhead": repair_overhead,
            "catchup_overhead": catchup_overhead,
            "total_overhead": overhead,
            "saved_fraction": saved_fraction,
            "min_width_fraction": min_width,
            "full_width_phase_present": full_width_seen,
            "phase_schedule": ",".join(f"{w:.6g}:{d:.6g}" for w, d in phase_schedule),
        },
    )


def reachable_blocks_from_outputs(block_edges: np.ndarray, output_blocks: set[int]) -> set[int]:
    reachable = set(output_blocks)
    changed = True
    while changed:
        changed = False
        for src in range(block_edges.shape[1]):
            for dst in range(block_edges.shape[0]):
                if dst in reachable and block_edges[dst, src] and src not in reachable:
                    reachable.add(src)
                    changed = True
    return reachable


def reachability_null_inactive_cone_gate(
    rng: np.random.Generator,
    tokens: np.ndarray,
    blocks: int,
    block_dim: int,
    active_blocks: int,
    tolerance: float,
    min_omitted_fraction: float,
) -> GateResult:
    active_blocks = max(1, min(active_blocks, blocks - 1))
    dim = blocks * block_dim
    inp_dim = 5
    out_dim = 4
    steps = min(96, tokens.size - 1)
    output_blocks = set(range(active_blocks))

    a = np.zeros((dim, dim))
    block_edges = np.zeros((blocks, blocks), dtype=bool)
    for dst in range(blocks):
        for src in range(blocks):
            dst_reachable = dst < active_blocks
            src_reachable = src < active_blocks
            if dst_reachable and not src_reachable:
                continue
            if dst_reachable and src_reachable and src <= dst:
                scale = 0.035
            elif (not dst_reachable) and src_reachable:
                scale = 0.025
            elif (not dst_reachable) and (not src_reachable) and src <= dst:
                scale = 0.020
            else:
                continue
            r0 = dst * block_dim
            c0 = src * block_dim
            block = rng.normal(0.0, scale, size=(block_dim, block_dim))
            if dst == src:
                block += np.eye(block_dim) * rng.uniform(0.25, 0.70)
            a[r0 : r0 + block_dim, c0 : c0 + block_dim] = block
            block_edges[dst, src] = bool(np.max(np.abs(block)) > 1e-12)

    b = rng.normal(0.0, 0.18, size=(dim, inp_dim))
    c = np.zeros((out_dim, dim))
    for block_id in output_blocks:
        c[:, block_id * block_dim : (block_id + 1) * block_dim] = rng.normal(0.0, 0.22, size=(out_dim, block_dim))

    xs = token_features(tokens[1:], steps, inp_dim)
    target = token_features(tokens[2:], steps, out_dim) * 0.25

    states = [np.zeros(dim)]
    ys = []
    h = np.zeros(dim)
    for x in xs:
        h = a @ h + b @ x
        states.append(h.copy())
        ys.append(c @ h)
    y = np.stack(ys, axis=0)
    dy = (y - target) / float(steps)

    grad_a = np.zeros_like(a)
    grad_b = np.zeros_like(b)
    grad_c = np.zeros_like(c)
    dh_next = np.zeros(dim)
    for t in range(steps - 1, -1, -1):
        h_t = states[t + 1]
        h_prev = states[t]
        grad_c += np.outer(dy[t], h_t)
        dh = c.T @ dy[t] + dh_next
        grad_a += np.outer(dh, h_prev)
        grad_b += np.outer(dh, xs[t])
        dh_next = a.T @ dh

    reachable = reachable_blocks_from_outputs(block_edges, output_blocks)
    inactive = sorted(set(range(blocks)) - reachable)
    omitted_mask_a = np.zeros_like(a, dtype=bool)
    omitted_mask_b = np.zeros_like(b, dtype=bool)
    for block_id in inactive:
        r0 = block_id * block_dim
        omitted_mask_a[r0 : r0 + block_dim, :] = True
        omitted_mask_b[r0 : r0 + block_dim, :] = True

    max_omitted_grad = 0.0
    if inactive:
        max_omitted_grad = max(
            float(np.max(np.abs(grad_a[omitted_mask_a]))),
            float(np.max(np.abs(grad_b[omitted_mask_b]))),
        )
    total_trainable = float(a.size + b.size)
    omitted_trainable = float(np.count_nonzero(omitted_mask_a) + np.count_nonzero(omitted_mask_b))
    omitted_fraction = omitted_trainable / max(total_trainable, 1.0)
    certificate_overhead = 0.025 + 0.0025 * blocks
    priced_ratio = 1.0 - omitted_fraction + certificate_overhead
    pass_gate = (
        bool(inactive)
        and max_omitted_grad <= tolerance
        and omitted_fraction >= min_omitted_fraction
        and priced_ratio <= 0.90
    )
    return gate_result(
        "reachability_null_inactive_cone_gate",
        pass_gate,
        "validated_for_gate",
        omitted_fraction,
        priced_ratio,
        {
            "blocks": blocks,
            "block_dim": block_dim,
            "active_blocks": active_blocks,
            "inactive_blocks": len(inactive),
            "reachable_blocks": len(reachable),
            "steps": steps,
            "max_omitted_grad": max_omitted_grad,
            "omitted_fraction": omitted_fraction,
            "certificate_overhead": certificate_overhead,
            "structurally_absent_active_from_inactive_edges": active_blocks * (blocks - active_blocks),
            "min_omitted_fraction": min_omitted_fraction,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--max-tokens", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42600)
    parser.add_argument("--phase-schedule", default="0.50:0.55,0.75:0.30,1.00:0.15")
    parser.add_argument("--recurrent-share", type=float, default=0.65)
    parser.add_argument("--birth-overhead", type=float, default=0.015)
    parser.add_argument("--release-overhead", type=float, default=0.035)
    parser.add_argument("--anchor-overhead", type=float, default=0.025)
    parser.add_argument("--repair-overhead", type=float, default=0.020)
    parser.add_argument("--catchup-overhead", type=float, default=0.060)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--block-dim", type=int, default=4)
    parser.add_argument("--active-blocks", type=int, default=5)
    parser.add_argument("--min-omitted-fraction", type=float, default=0.10)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    tokens = read_v12_tokens(args.dataset, args.max_tokens, args.seed)
    rng = np.random.default_rng(args.seed)
    phase_schedule = parse_phase_schedule(args.phase_schedule)

    gates = [
        staged_width_structural_birth_gate(
            rng,
            tokens,
            phase_schedule,
            args.recurrent_share,
            args.birth_overhead,
            args.release_overhead,
            args.anchor_overhead,
            args.repair_overhead,
            args.catchup_overhead,
            args.tolerance,
        ),
        reachability_null_inactive_cone_gate(
            rng,
            tokens,
            args.blocks,
            args.block_dim,
            args.active_blocks,
            args.tolerance,
            args.min_omitted_fraction,
        ),
    ]
    pass_count = sum(1 for gate in gates if gate.decision.startswith("PASS_"))
    decision = "PASS_R426_THEORETICAL_PERFORMANCE_GATES" if pass_count == len(gates) else "PARTIAL_R426_THEORETICAL_PERFORMANCE_GATES"
    activation = (
        f"[R426_THEORETICAL_PERFORMANCE_GATES] active=1 decision={decision} "
        f"pass_count={pass_count} fail_count={len(gates) - pass_count} no_gpu=1 no_training_integration=1"
    )
    result = SuiteResult(
        schema=SCHEMA,
        decision=decision,
        dataset=str(args.dataset) if args.dataset else None,
        tokens_used=int(tokens.size),
        no_gpu=1,
        no_training_integration=1,
        activation_log=activation,
        gates=gates,
        elapsed_seconds=time.perf_counter() - start,
    )
    print(activation)
    for gate in gates:
        print(gate.activation_log)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
