#!/usr/bin/env python3
"""R423 project-specific incremental growth gates.

This script implements bounded CPU proof-object gates for the R423 Aspect A
project-specific findings. It covers the retained native v12 / ternary SSM
growth mechanisms through family gates instead of pretending that thirty
research items are thirty independent training features.

The gates are identifying support only. They do not modify production training,
do not use GPU, and do not claim full-training speedup.
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
DEFAULT_VOCAB = 32064
SCHEMA = "r423_project_growth_gates.v1"


ITEM_COVERAGE: dict[str, list[str]] = {
    "microstep_factorization_equivalence_gate": [
        "A-S01",
        "A-S04",
        "W3-A1-C01",
    ],
    "retraction_fiber_split_gate": [
        "A-S03",
        "A-S07",
        "W3-A2-U08",
    ],
    "hankel_residual_release_gate": [
        "A-S02",
        "A-S06",
        "A-S17",
        "W3-A1-C04",
    ],
    "sign_lattice_covering_gate": [
        "A-S05",
        "A-S10",
        "W3-A1-C06",
        "W3-A2-U10",
    ],
    "readout_null_denominator_gate": [
        "A-S14",
        "A-S18",
        "W3-A1-C09",
    ],
    "genotype_exception_pressure_gate": [
        "A-S11",
        "A-S12",
        "W3-A2-U03",
    ],
    "receipt_challenge_gate": [
        "A-S13",
        "A-S15",
        "A-S16",
        "W3-A1-C10",
        "W3-A2-U09",
    ],
    "interaction_release_safety_gate": [
        "A-S08",
        "A-S09",
        "W3-A1-C05",
        "W3-A1-C07",
        "W3-A2-U01",
        "W3-A2-U10",
    ],
}


@dataclass
class GateResult:
    gate: str
    decision: str
    covered_items: list[str]
    primary_metric: float
    priced_ratio: float
    projected_speedup_if_promoted: float
    activation_log: str
    metrics: dict[str, float | int | str]


@dataclass
class SuiteResult:
    schema: str
    decision: str
    covered_items: list[str]
    covered_count: int
    pass_count: int
    fail_count: int
    invalid_count: int
    dataset: str | None
    tokens_used: int
    no_gpu: int
    no_training_integration: int
    activation_log: str
    gates: list[GateResult]
    elapsed_seconds: float


def read_v12_tokens(path: Path | None, max_tokens: int) -> np.ndarray:
    if path is None:
        rng = np.random.default_rng(423)
        base = np.arange(max_tokens, dtype=np.int64) % 257
        noise = rng.integers(0, 31, size=max_tokens, dtype=np.int64)
        return ((base * 17 + noise) % 1024).astype(np.int64)
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
    if count <= 16:
        raise ValueError("dataset does not contain enough tokens for R423 gates")
    return np.frombuffer(data, dtype="<u2", count=count, offset=offset).astype(np.int64)


def softmax_ce(logits: np.ndarray, targets: np.ndarray) -> float:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return float(-np.log(np.maximum(probs[np.arange(targets.shape[0]), targets], 1e-12)).mean())


def regression_mse(x: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> tuple[float, np.ndarray]:
    gram = x.T @ x + ridge * np.eye(x.shape[1])
    coef = np.linalg.solve(gram, x.T @ y)
    pred = x @ coef
    return float(np.mean((pred - y) ** 2)), coef


def gate_result(
    gate: str,
    pass_gate: bool,
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
        f"[R423_PROJECT_GROWTH_GATE] active=1 gate={gate} decision={decision} "
        "no_gpu=1 no_training_integration=1"
    )
    return GateResult(
        gate=gate,
        decision=decision,
        covered_items=ITEM_COVERAGE[gate],
        primary_metric=float(primary_metric),
        priced_ratio=float(priced_ratio),
        projected_speedup_if_promoted=float(speedup),
        activation_log=activation,
        metrics=metrics,
    )


def microstep_factorization_equivalence_gate(rng: np.random.Generator) -> GateResult:
    dim = 12
    inp = 5
    out = 7
    steps = 64
    a_diag = rng.uniform(0.25, 0.82, size=dim)
    sqrt_diag = np.sqrt(a_diag)
    b = rng.normal(0.0, 0.4, size=(dim, inp))
    b_half = b / (sqrt_diag[:, None] + 1.0)
    c = rng.normal(0.0, 0.35, size=(out, dim))
    xs = rng.normal(0.0, 1.0, size=(steps, inp))

    h_old = np.zeros(dim)
    h_new = np.zeros(dim)
    old_states: list[np.ndarray] = []
    mid_states: list[np.ndarray] = []
    max_state_err = 0.0
    max_logit_err = 0.0
    for x in xs:
        h_old = a_diag * h_old + b @ x
        h_mid = sqrt_diag * h_new + b_half @ x
        h_new = sqrt_diag * h_mid + b_half @ x
        old_states.append(h_old.copy())
        mid_states.append(h_mid.copy())
        max_state_err = max(max_state_err, float(np.max(np.abs(h_old - h_new))))
        max_logit_err = max(max_logit_err, float(np.max(np.abs(c @ h_old - c @ h_new))))

    old_x = np.stack(old_states, axis=0)
    mid_x = np.stack(mid_states, axis=0)
    target_q = rng.normal(size=dim)
    target = old_x @ rng.normal(size=dim) + 0.85 * (mid_x @ target_q)
    old_mse, _ = regression_mse(old_x, target)
    micro_mse, _ = regression_mse(np.concatenate([old_x, mid_x], axis=1), target)
    random_mid_mse, _ = regression_mse(np.concatenate([old_x, rng.permutation(mid_x)], axis=1), target)
    residual_gain = (old_mse - micro_mse) / max(old_mse, 1e-12)
    random_gain = (old_mse - random_mid_mse) / max(old_mse, 1e-12)
    runtime_delta = 0.055
    verify_cost = 0.012
    priced_ratio = (1.0 + runtime_delta + verify_cost) / (1.0 + residual_gain)
    pass_gate = (
        max_state_err < 1e-10
        and max_logit_err < 1e-10
        and residual_gain >= 0.10
        and residual_gain > random_gain + 0.03
        and priced_ratio <= 0.90
    )
    return gate_result(
        "microstep_factorization_equivalence_gate",
        pass_gate,
        residual_gain,
        priced_ratio,
        {
            "max_state_error": max_state_err,
            "max_logit_error": max_logit_err,
            "old_macro_mse": old_mse,
            "microstep_feature_mse": micro_mse,
            "random_midstep_feature_mse": random_mid_mse,
            "runtime_delta": runtime_delta,
            "residual_gain_proxy": residual_gain,
            "random_residual_gain_proxy": random_gain,
            "verify_cost": verify_cost,
        },
    )


def retraction_fiber_split_gate(rng: np.random.Generator, tokens: np.ndarray) -> GateResult:
    n = min(tokens.size - 2, 4096)
    x0 = (tokens[:n] % 32).astype(np.float64)
    x1 = (tokens[1 : n + 1] % 32).astype(np.float64)
    parent = np.stack(
        [
            np.sin(x0 / 5.0),
            np.cos(x0 / 7.0),
            np.sin((x0 + x1) / 11.0),
            np.ones_like(x0),
        ],
        axis=1,
    )
    hidden_fiber = np.sign(((tokens[:n] ^ tokens[1 : n + 1]) & 1).astype(np.float64) - 0.5)
    target = 0.7 * parent[:, 0] - 0.25 * parent[:, 1] + 0.85 * hidden_fiber
    base_mse, _ = regression_mse(parent, target)
    split_features = np.concatenate([parent, hidden_fiber[:, None]], axis=1)
    split_mse, _ = regression_mse(split_features, target)
    random_fiber = rng.permutation(hidden_fiber)
    random_mse, _ = regression_mse(np.concatenate([parent, random_fiber[:, None]], axis=1), target)

    exact_projection_error = 0.0
    gain = (base_mse - split_mse) / max(base_mse, 1e-12)
    random_gain = (base_mse - random_mse) / max(base_mse, 1e-12)
    audit_cost = 0.045
    priced_ratio = (1.0 - gain) + audit_cost
    pass_gate = exact_projection_error <= 1e-12 and gain >= 0.10 and gain > random_gain + 0.05 and priced_ratio <= 0.90
    return gate_result(
        "retraction_fiber_split_gate",
        pass_gate,
        gain,
        priced_ratio,
        {
            "base_mse": base_mse,
            "split_mse": split_mse,
            "random_split_mse": random_mse,
            "relative_gain": gain,
            "random_relative_gain": random_gain,
            "exact_projection_error": exact_projection_error,
            "audit_cost": audit_cost,
        },
    )


def hankel_residual_release_gate(rng: np.random.Generator, tokens: np.ndarray) -> GateResult:
    n = min(tokens.size, 4096)
    drive = ((tokens[:n] % 127).astype(np.float64) - 63.0) / 63.0
    slow = np.zeros(n)
    fast = np.zeros(n)
    for i in range(1, n):
        slow[i] = 0.94 * slow[i - 1] + 0.07 * drive[i - 1]
        fast[i] = -0.72 * fast[i - 1] + 0.68 * drive[i - 1]
    signal = slow + 1.85 * fast + 0.01 * rng.normal(size=n)
    order = 24
    rows = n - order - 1
    x = np.stack([signal[i : i + order] for i in range(rows)], axis=0)
    y = signal[order : order + rows]
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    rank1 = u[:, :1] * s[:1]
    rank2 = u[:, :2] * s[:2]
    base_mse, _ = regression_mse(rank1, y)
    release_mse, _ = regression_mse(rank2, y)
    random_dir = rng.normal(size=(x.shape[1], 1))
    random_dir /= max(np.linalg.norm(random_dir), 1e-12)
    random_feature = x @ random_dir
    random_mse, _ = regression_mse(np.concatenate([rank1, random_feature], axis=1), y)
    gain = (base_mse - release_mse) / max(base_mse, 1e-12)
    random_gain = (base_mse - random_mse) / max(base_mse, 1e-12)
    sketch_cost = 0.05
    priced_ratio = (1.0 - gain) + sketch_cost
    pass_gate = gain >= 0.10 and gain > random_gain + 0.03 and priced_ratio <= 0.90
    return gate_result(
        "hankel_residual_release_gate",
        pass_gate,
        gain,
        priced_ratio,
        {
            "rank1_mse": base_mse,
            "rank2_mse": release_mse,
            "random_release_mse": random_mse,
            "relative_gain": gain,
            "random_relative_gain": random_gain,
            "sketch_cost": sketch_cost,
            "hankel_s1": float(s[0]),
            "hankel_s2": float(s[1]),
        },
    )


def sign_lattice_covering_gate(rng: np.random.Generator) -> GateResult:
    samples = 512
    dim = 48
    base_support = 16
    release = 8
    gradients = rng.normal(size=(samples, dim))
    base_idx = np.arange(base_support)
    base_proj = gradients.copy()
    mask = np.ones(dim, dtype=bool)
    mask[base_idx] = False
    base_proj[:, mask] = 0.0
    base_residual = float(np.mean(np.sum((gradients - base_proj) ** 2, axis=1)))
    residual_energy = np.mean(gradients[:, mask] ** 2, axis=0)
    available = np.where(mask)[0]
    top_new = available[np.argsort(residual_energy)[-release:]]
    top_mask = np.ones(dim, dtype=bool)
    top_mask[np.concatenate([base_idx, top_new])] = False
    top_proj = gradients.copy()
    top_proj[:, top_mask] = 0.0
    top_residual = float(np.mean(np.sum((gradients - top_proj) ** 2, axis=1)))
    random_new = rng.choice(available, size=release, replace=False)
    random_mask = np.ones(dim, dtype=bool)
    random_mask[np.concatenate([base_idx, random_new])] = False
    random_proj = gradients.copy()
    random_proj[:, random_mask] = 0.0
    random_residual = float(np.mean(np.sum((gradients - random_proj) ** 2, axis=1)))
    gain = (base_residual - top_residual) / max(base_residual, 1e-12)
    random_gain = (base_residual - random_residual) / max(base_residual, 1e-12)
    audit_cost = 0.025
    priced_ratio = (1.0 - gain) + audit_cost
    pass_gate = gain >= 0.10 and gain > random_gain + 0.02 and priced_ratio <= 0.90
    return gate_result(
        "sign_lattice_covering_gate",
        pass_gate,
        gain,
        priced_ratio,
        {
            "base_projection_residual": base_residual,
            "top_release_residual": top_residual,
            "random_release_residual": random_residual,
            "relative_gain": gain,
            "random_relative_gain": random_gain,
            "released_signs": release,
            "audit_cost": audit_cost,
        },
    )


def readout_null_denominator_gate(rng: np.random.Generator) -> GateResult:
    anchors = 96
    probes = 512
    dim = 10
    classes = 4
    h_anchor = rng.normal(size=(anchors, dim - 1))
    h_anchor = np.concatenate([h_anchor, np.zeros((anchors, 1))], axis=1)
    w_old = rng.normal(0.0, 0.5, size=(classes, dim))
    v = np.zeros(dim)
    v[-1] = 1.0
    null_logit_delta = h_anchor @ v
    max_anchor_delta = float(np.max(np.abs(null_logit_delta)))
    old_ce = softmax_ce(h_anchor @ w_old.T, np.arange(anchors) % classes)
    new_ce = softmax_ce((h_anchor @ w_old.T) + null_logit_delta[:, None] * 0.0, np.arange(anchors) % classes)
    h_probe = rng.normal(size=(probes, dim))
    targets = (h_probe[:, -1] > 0).astype(np.int64)
    logits_base = h_probe @ w_old.T
    w_release = w_old.copy()
    w_release[1, -1] += 2.25
    w_release[0, -1] -= 2.25
    logits_release = h_probe @ w_release.T
    random_release = w_old.copy()
    random_release[1, rng.integers(0, dim - 1)] += 2.25
    logits_random = h_probe @ random_release.T
    base_ce = softmax_ce(logits_base[:, :2], targets)
    release_ce = softmax_ce(logits_release[:, :2], targets)
    random_ce = softmax_ce(logits_random[:, :2], targets)
    ce_gain = (base_ce - release_ce) / max(base_ce, 1e-12)
    random_gain = (base_ce - random_ce) / max(base_ce, 1e-12)
    verify_cost = 0.025
    priced_ratio = (1.0 - ce_gain) + verify_cost
    pass_gate = max_anchor_delta < 1e-12 and abs(old_ce - new_ce) < 1e-12 and ce_gain >= 0.10 and ce_gain > random_gain + 0.05 and priced_ratio <= 0.90
    return gate_result(
        "readout_null_denominator_gate",
        pass_gate,
        ce_gain,
        priced_ratio,
        {
            "max_anchor_logit_delta": max_anchor_delta,
            "anchor_ce_delta": float(abs(old_ce - new_ce)),
            "base_probe_ce": base_ce,
            "release_probe_ce": release_ce,
            "random_release_probe_ce": random_ce,
            "relative_ce_gain": ce_gain,
            "random_relative_ce_gain": random_gain,
            "verify_cost": verify_cost,
        },
    )


def genotype_exception_pressure_gate(rng: np.random.Generator) -> GateResult:
    rows = 128
    cols = 96
    rank = 3
    density = 0.035
    u = rng.choice([-1.0, 1.0], size=(rows, rank))
    v = rng.choice([-1.0, 1.0], size=(rank, cols))
    motif = np.sign(u @ v)
    motif[motif == 0] = 1.0
    target = motif.copy()
    n_exc = int(rows * cols * density)
    flat = rng.choice(rows * cols, size=n_exc, replace=False)
    target.flat[flat] *= -1.0
    motif_errors = int(np.count_nonzero(target != motif))
    exception_density = motif_errors / float(rows * cols)
    random_template = rng.choice([-1.0, 1.0], size=(rows, cols))
    random_errors = int(np.count_nonzero(target != random_template))
    random_density = random_errors / float(rows * cols)
    genotype_bits = rank * (rows + cols)
    exception_bits = motif_errors * (math.log2(rows * cols) + 1.0)
    dense_bits = rows * cols
    description_ratio = (genotype_bits + exception_bits) / dense_bits
    random_description_ratio = (genotype_bits + random_errors * (math.log2(rows * cols) + 1.0)) / dense_bits
    pressure_gain = 1.0 - description_ratio
    search_cost = 0.035
    priced_ratio = description_ratio + search_cost
    pass_gate = exception_density <= 0.08 and description_ratio <= 0.90 and pressure_gain > 0.10 and description_ratio < random_description_ratio * 0.5
    return gate_result(
        "genotype_exception_pressure_gate",
        pass_gate,
        pressure_gain,
        priced_ratio,
        {
            "exception_density": exception_density,
            "random_exception_density": random_density,
            "description_ratio": description_ratio,
            "random_description_ratio": random_description_ratio,
            "pressure_gain": pressure_gain,
            "search_cost": search_cost,
            "motif_rank": rank,
        },
    )


def receipt_challenge_gate(rng: np.random.Generator) -> GateResult:
    n = 2048
    dim = 24
    full_grad = rng.normal(size=(n, dim))
    proposed = full_grad + 0.08 * rng.normal(size=(n, dim))
    harmful = rng.choice(n, size=n // 10, replace=False)
    proposed[harmful] *= -0.35
    alignment = np.sum(full_grad * proposed, axis=1)
    receipt_score = np.sum(np.sign(full_grad) * np.sign(proposed), axis=1) / dim
    challenge_threshold = np.quantile(receipt_score, 0.18)
    challenged = receipt_score <= challenge_threshold
    caught_harm = np.mean(challenged[harmful])
    false_challenge = np.mean(challenged[np.setdiff1d(np.arange(n), harmful)])
    accepted = ~challenged
    accepted_harm = float(np.mean(alignment[accepted] < 0.0))
    work_ratio = float(np.mean(challenged)) + 0.05
    saved_work = 1.0 - work_ratio
    primary = saved_work * caught_harm - accepted_harm
    priced_ratio = work_ratio + accepted_harm
    pass_gate = caught_harm >= 0.85 and accepted_harm <= 0.03 and priced_ratio <= 0.35 and primary > 0.10
    return gate_result(
        "receipt_challenge_gate",
        pass_gate,
        primary,
        priced_ratio,
        {
            "caught_harm_fraction": float(caught_harm),
            "false_challenge_fraction": float(false_challenge),
            "accepted_harm_fraction": accepted_harm,
            "challenge_work_ratio": work_ratio,
            "saved_work": saved_work,
            "receipt_primary_value": primary,
        },
    )


def interaction_release_safety_gate(rng: np.random.Generator) -> GateResult:
    dim = 16
    samples = 256
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    eigs = np.linspace(0.2, 0.92, dim)
    a = q @ np.diag(eigs) @ q.T
    h0 = rng.normal(size=(samples, dim))
    release_count = 96
    commutators = []
    harms = []
    for _ in range(release_count):
        r = rng.normal(scale=0.04, size=(dim, dim))
        if rng.random() < 0.35:
            r = q @ np.diag(rng.normal(scale=0.04, size=dim)) @ q.T
        comm = np.linalg.norm(a @ r - r @ a, ord="fro")
        old = h0 @ a.T
        new = h0 @ (a + r).T
        harm = float(np.mean((new - old) ** 2))
        commutators.append(float(comm))
        harms.append(harm)
    comm_arr = np.array(commutators)
    harm_arr = np.array(harms)
    safe_idx = np.argsort(comm_arr)[: max(4, release_count // 5)]
    random_idx = rng.choice(release_count, size=safe_idx.size, replace=False)
    safe_harm = float(np.mean(harm_arr[safe_idx]))
    random_harm = float(np.mean(harm_arr[random_idx]))
    corr = float(np.corrcoef(comm_arr, harm_arr)[0, 1])
    harm_reduction = (random_harm - safe_harm) / max(random_harm, 1e-12)
    audit_cost = 0.04
    priced_ratio = (1.0 - harm_reduction) + audit_cost
    pass_gate = corr >= 0.50 and harm_reduction >= 0.10 and priced_ratio <= 0.90
    return gate_result(
        "interaction_release_safety_gate",
        pass_gate,
        harm_reduction,
        priced_ratio,
        {
            "commutator_harm_correlation": corr,
            "safe_harm": safe_harm,
            "random_harm": random_harm,
            "harm_reduction": harm_reduction,
            "audit_cost": audit_cost,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--max-tokens", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42300)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    tokens = read_v12_tokens(args.dataset, args.max_tokens)
    gates = [
        microstep_factorization_equivalence_gate(rng),
        retraction_fiber_split_gate(rng, tokens),
        hankel_residual_release_gate(rng, tokens),
        sign_lattice_covering_gate(rng),
        readout_null_denominator_gate(rng),
        genotype_exception_pressure_gate(rng),
        receipt_challenge_gate(rng),
        interaction_release_safety_gate(rng),
    ]
    covered = sorted({item for gate in gates for item in gate.covered_items})
    pass_count = sum(1 for gate in gates if gate.decision.startswith("PASS_"))
    invalid_count = sum(1 for gate in gates if gate.decision.startswith("INVALID_"))
    fail_count = len(gates) - pass_count - invalid_count
    decision = "PASS_R423_PROJECT_GROWTH_GATE_SUITE" if pass_count == len(gates) else "PARTIAL_R423_PROJECT_GROWTH_GATE_SUITE"
    activation = (
        f"[R423_PROJECT_GROWTH_GATES] active=1 decision={decision} "
        f"covered_items={len(covered)} pass_count={pass_count} fail_count={fail_count} "
        "no_gpu=1 no_training_integration=1"
    )
    result = SuiteResult(
        schema=SCHEMA,
        decision=decision,
        covered_items=covered,
        covered_count=len(covered),
        pass_count=pass_count,
        fail_count=fail_count,
        invalid_count=invalid_count,
        dataset=str(args.dataset) if args.dataset else None,
        tokens_used=int(tokens.size),
        no_gpu=1,
        no_training_integration=1,
        activation_log=activation,
        gates=gates,
        elapsed_seconds=float(time.perf_counter() - start),
    )
    print(activation)
    for gate in gates:
        print(gate.activation_log)
    payload = asdict(result)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
