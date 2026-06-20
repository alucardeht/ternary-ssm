#!/usr/bin/env python3
"""Original-CE credit debt identifying gate.

This probe is the first bounded implementation gate for the R422 cluster
"Original-CE Credit Debt And Certified Local Updates".

It builds a tiny recurrent next-token CE problem from a real v12 uint16 token
shard, materializes the exact recurrent CE gradient, then asks whether a small
certificate-selected active surface can preserve original-CE update progress
after charging debt, certificate, verifier, and random-control costs.

This is not production training, not a GPU benchmark, and not a final speedup
claim. It is a cheap falsifier for the claim that omitted recurrent credit can
be represented as priced debt instead of always paying dense full backward.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


DATASET_MAGIC = 0x544F4B4E
DATASET_HEADER_BYTES = 16
DEFAULT_VOCAB = 32064
SCHEMA = "original_ce_credit_debt_gate.v1"


@dataclass
class ArmResult:
    arm: str
    selected_fraction: float
    work_fraction: float
    residual_fraction: float
    cosine_to_full: float
    first_order_progress_ratio: float
    actual_progress_ratio: float
    update_ce: float
    priced_ratio: float
    whole_step_priced_ratio_estimate: float
    whole_step_speedup_estimate: float
    beats_random_progress: bool
    pass_gate: bool
    decision: str


def read_v12_tokens(path: Path, max_tokens: int | None) -> np.ndarray:
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
    n = min(n_tokens, available)
    if max_tokens is not None:
        n = min(n, max_tokens)
    if n <= 4:
        raise ValueError("dataset does not contain enough tokens")
    return np.frombuffer(data, dtype="<u2", count=n, offset=offset).astype(np.int64)


def stable_softmax_ce(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z, dtype=np.float64)
    probs = exp / exp.sum(axis=1, keepdims=True)
    target_probs = probs[np.arange(targets.shape[0]), targets]
    ce = float(-np.log(np.maximum(target_probs, 1e-300)).mean())
    dlogits = probs.astype(np.float64)
    dlogits[np.arange(targets.shape[0]), targets] -= 1.0
    dlogits /= float(targets.shape[0])
    return ce, dlogits


def init_params(vocab: int, hidden: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    embed = rng.normal(0.0, 0.18, size=(vocab, hidden)).astype(np.float64)
    recurrent = rng.normal(0.0, 0.22 / math.sqrt(hidden), size=(hidden, hidden)).astype(np.float64)
    spectral = float(np.linalg.norm(recurrent, ord=2))
    if spectral > 0.92:
        recurrent *= 0.92 / spectral
    readout = rng.normal(0.0, 0.12 / math.sqrt(hidden), size=(vocab, hidden)).astype(np.float64)
    return {"embed": embed, "recurrent": recurrent, "readout": readout}


def choose_window(tokens: np.ndarray, seq_len: int, offset: int, vocab: int) -> np.ndarray:
    if tokens.shape[0] < seq_len + 1:
        raise ValueError("not enough tokens for requested sequence length")
    max_start = tokens.shape[0] - seq_len - 1
    start = min(max(offset, 0), max_start)
    window = tokens[start : start + seq_len + 1].copy()
    if window.max(initial=0) >= vocab:
        window %= vocab
    return window


def forward(tokens: np.ndarray, params: dict[str, np.ndarray]) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    embed = params["embed"]
    recurrent = params["recurrent"]
    readout = params["readout"]
    inputs = tokens[:-1]
    targets = tokens[1:]
    hidden = recurrent.shape[0]
    states = np.zeros((inputs.shape[0], hidden), dtype=np.float64)
    prev_states = np.zeros_like(states)
    h = np.zeros(hidden, dtype=np.float64)
    for i, tok in enumerate(inputs):
        prev_states[i] = h
        h = np.tanh(embed[tok] + recurrent @ h)
        states[i] = h
    logits = states @ readout.T
    ce, dlogits = stable_softmax_ce(logits, targets)
    direct_dh = dlogits @ readout
    return ce, states, prev_states, direct_dh


def gradient_contributions(
    states: np.ndarray,
    prev_states: np.ndarray,
    direct_dh: np.ndarray,
    recurrent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    seq_len, hidden = states.shape
    full_by_step = np.zeros((seq_len, hidden, hidden), dtype=np.float64)
    for loss_t in range(seq_len):
        dh = direct_dh[loss_t].copy()
        for j in range(loss_t, -1, -1):
            da = dh * (1.0 - states[j] * states[j])
            full_by_step[j] += np.outer(da, prev_states[j])
            dh = recurrent.T @ da
    local_by_step = np.zeros_like(full_by_step)
    for j in range(seq_len):
        da = direct_dh[j] * (1.0 - states[j] * states[j])
        local_by_step[j] = np.outer(da, prev_states[j])
    return full_by_step, local_by_step


def certificate_scores(
    states: np.ndarray,
    prev_states: np.ndarray,
    direct_dh: np.ndarray,
) -> np.ndarray:
    state_delta = np.linalg.norm(states - prev_states, axis=1)
    local_pressure = np.linalg.norm(direct_dh, axis=1)
    state_energy = np.linalg.norm(prev_states, axis=1)
    return local_pressure * (state_delta + 1e-9) * (state_energy + 1e-9)


def top_mask(scores: np.ndarray, fraction: float) -> np.ndarray:
    n = scores.shape[0]
    k = max(1, min(n, int(math.ceil(fraction * n))))
    idx = np.argpartition(scores, -k)[-k:]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def random_masks(n: int, fraction: float, count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    k = max(1, min(n, int(math.ceil(fraction * n))))
    masks: list[np.ndarray] = []
    for _ in range(count):
        idx = rng.choice(n, size=k, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        masks.append(mask)
    return masks


def update_ce(tokens: np.ndarray, params: dict[str, np.ndarray], grad: np.ndarray, lr: float) -> float:
    next_params = {k: v.copy() for k, v in params.items()}
    next_params["recurrent"] = params["recurrent"] - lr * grad
    ce, _, _, _ = forward(tokens, next_params)
    return ce


def evaluate_arm(
    *,
    arm: str,
    tokens: np.ndarray,
    params: dict[str, np.ndarray],
    base_ce: float,
    full_update_ce: float,
    full_grad: np.ndarray,
    approx_grad: np.ndarray,
    selected_fraction: float,
    work_fraction: float,
    lr: float,
    debt_price: float,
    certificate_price: float,
    verifier_price: float,
    recurrent_backward_share: float,
    max_residual: float,
    min_progress: float,
    max_priced_ratio: float,
    min_whole_step_speedup: float,
    random_progress_ratio: float,
) -> ArmResult:
    full_norm = float(np.linalg.norm(full_grad))
    approx_norm = float(np.linalg.norm(approx_grad))
    residual = float(np.linalg.norm(full_grad - approx_grad) / max(full_norm, 1e-30))
    dot = float(np.sum(full_grad * approx_grad))
    cosine = dot / max(full_norm * approx_norm, 1e-30)
    full_first_order = lr * full_norm * full_norm
    approx_first_order = lr * dot
    first_order_ratio = approx_first_order / max(full_first_order, 1e-30)
    candidate_ce = update_ce(tokens, params, approx_grad, lr)
    full_progress = base_ce - full_update_ce
    candidate_progress = base_ce - candidate_ce
    actual_progress_ratio = candidate_progress / max(full_progress, 1e-30)
    charged = work_fraction + certificate_price * selected_fraction + verifier_price + debt_price * residual
    priced_ratio = charged / max(actual_progress_ratio, 1e-9)
    whole_step_priced_ratio = 1.0 - recurrent_backward_share * max(0.0, 1.0 - priced_ratio)
    whole_step_speedup = 1.0 / max(whole_step_priced_ratio, 1e-9)
    beats_random = actual_progress_ratio > random_progress_ratio + 1e-6
    pass_gate = (
        residual <= max_residual
        and actual_progress_ratio >= min_progress
        and priced_ratio <= max_priced_ratio
        and whole_step_speedup >= min_whole_step_speedup
        and candidate_progress > 0.0
        and beats_random
    )
    if pass_gate:
        decision = "PASS_ORIGINAL_CE_CREDIT_DEBT_GATE"
    elif candidate_progress <= 0.0:
        decision = "FAIL_APPROX_UPDATE_HARMS_CE"
    elif not beats_random:
        decision = "FAIL_CERTIFICATE_DOES_NOT_BEAT_RANDOM_SURFACE"
    elif residual > max_residual:
        decision = "FAIL_DENSE_DEBT"
    elif actual_progress_ratio < min_progress:
        decision = "FAIL_INSUFFICIENT_ORIGINAL_CE_PROGRESS"
    elif whole_step_speedup < min_whole_step_speedup:
        decision = "FAIL_WHOLE_STEP_SPEEDUP_ESTIMATE_BELOW_BAR"
    else:
        decision = "FAIL_PRICED_RATIO"
    return ArmResult(
        arm=arm,
        selected_fraction=float(selected_fraction),
        work_fraction=float(work_fraction),
        residual_fraction=float(residual),
        cosine_to_full=float(cosine),
        first_order_progress_ratio=float(first_order_ratio),
        actual_progress_ratio=float(actual_progress_ratio),
        update_ce=float(candidate_ce),
        priced_ratio=float(priced_ratio),
        whole_step_priced_ratio_estimate=float(whole_step_priced_ratio),
        whole_step_speedup_estimate=float(whole_step_speedup),
        beats_random_progress=bool(beats_random),
        pass_gate=bool(pass_gate),
        decision=decision,
    )


def parse_fractions(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        value = float(part.strip())
        if value <= 0.0 or value > 1.0:
            raise ValueError("surface fractions must be in (0, 1]")
        values.append(value)
    return sorted(set(values))


def run_gate(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    tokens = read_v12_tokens(args.dataset, args.max_tokens)
    window = choose_window(tokens, args.seq_len, args.offset, args.vocab)
    params = init_params(args.vocab, args.hidden, args.seed)
    base_ce, states, prev_states, direct_dh = forward(window, params)
    full_by_step, local_by_step = gradient_contributions(
        states, prev_states, direct_dh, params["recurrent"]
    )
    full_grad = full_by_step.sum(axis=0)
    local_grad = local_by_step.sum(axis=0)
    full_update_ce = update_ce(window, params, full_grad, args.lr)
    full_progress = base_ce - full_update_ce
    scores = certificate_scores(states, prev_states, direct_dh)
    oracle_scores = np.linalg.norm(full_by_step.reshape(full_by_step.shape[0], -1), axis=1)

    arms: list[ArmResult] = []
    for fraction in parse_fractions(args.surface_fractions):
        random_progress_values = []
        for mask in random_masks(window.shape[0] - 1, fraction, args.random_controls, args.seed + 17):
            random_grad = full_by_step[mask].sum(axis=0)
            random_ce = update_ce(window, params, random_grad, args.lr)
            random_progress_values.append((base_ce - random_ce) / max(full_progress, 1e-30))
        random_progress = float(np.mean(random_progress_values))

        cert_mask = top_mask(scores, fraction)
        oracle_mask = top_mask(oracle_scores, fraction)
        arms.append(
            evaluate_arm(
                arm=f"certificate_surface_full_contrib_{fraction:.3f}",
                tokens=window,
                params=params,
                base_ce=base_ce,
                full_update_ce=full_update_ce,
                full_grad=full_grad,
                approx_grad=full_by_step[cert_mask].sum(axis=0),
                selected_fraction=fraction,
                work_fraction=fraction,
                lr=args.lr,
                debt_price=args.debt_price,
                certificate_price=args.certificate_price,
                verifier_price=args.verifier_price,
                recurrent_backward_share=args.recurrent_backward_share,
                max_residual=args.max_residual,
                min_progress=args.min_progress,
                max_priced_ratio=args.max_priced_ratio,
                min_whole_step_speedup=args.min_whole_step_speedup,
                random_progress_ratio=random_progress,
            )
        )
        arms.append(
            evaluate_arm(
                arm=f"certificate_surface_local_contrib_{fraction:.3f}",
                tokens=window,
                params=params,
                base_ce=base_ce,
                full_update_ce=full_update_ce,
                full_grad=full_grad,
                approx_grad=local_by_step[cert_mask].sum(axis=0),
                selected_fraction=fraction,
                work_fraction=max(args.local_update_work_floor, fraction * args.local_update_work_multiplier),
                lr=args.lr,
                debt_price=args.debt_price,
                certificate_price=args.certificate_price,
                verifier_price=args.verifier_price,
                recurrent_backward_share=args.recurrent_backward_share,
                max_residual=args.max_residual,
                min_progress=args.min_progress,
                max_priced_ratio=args.max_priced_ratio,
                min_whole_step_speedup=args.min_whole_step_speedup,
                random_progress_ratio=random_progress,
            )
        )
        arms.append(
            evaluate_arm(
                arm=f"oracle_surface_full_contrib_{fraction:.3f}",
                tokens=window,
                params=params,
                base_ce=base_ce,
                full_update_ce=full_update_ce,
                full_grad=full_grad,
                approx_grad=full_by_step[oracle_mask].sum(axis=0),
                selected_fraction=fraction,
                work_fraction=fraction,
                lr=args.lr,
                debt_price=args.debt_price,
                certificate_price=args.certificate_price,
                verifier_price=args.verifier_price,
                recurrent_backward_share=args.recurrent_backward_share,
                max_residual=args.max_residual,
                min_progress=args.min_progress,
                max_priced_ratio=args.max_priced_ratio,
                min_whole_step_speedup=args.min_whole_step_speedup,
                random_progress_ratio=random_progress,
            )
        )

    full_local = evaluate_arm(
        arm="all_steps_local_direct",
        tokens=window,
        params=params,
        base_ce=base_ce,
        full_update_ce=full_update_ce,
        full_grad=full_grad,
        approx_grad=local_grad,
        selected_fraction=1.0,
        work_fraction=args.local_update_work_multiplier,
        lr=args.lr,
        debt_price=args.debt_price,
        certificate_price=args.certificate_price,
        verifier_price=args.verifier_price,
        recurrent_backward_share=args.recurrent_backward_share,
        max_residual=args.max_residual,
        min_progress=args.min_progress,
        max_priced_ratio=args.max_priced_ratio,
        min_whole_step_speedup=args.min_whole_step_speedup,
        random_progress_ratio=-1.0,
    )
    arms.append(full_local)
    best = min(arms, key=lambda item: item.priced_ratio)
    pass_certificate = any(
        item.pass_gate and item.arm.startswith("certificate_surface_full_contrib")
        for item in arms
    )
    pass_local_surface = any(
        item.pass_gate and "local_contrib" in item.arm
        for item in arms
    )
    pass_local_all_steps = full_local.pass_gate
    pass_oracle = any(item.pass_gate and item.arm.startswith("oracle_surface") for item in arms)
    if full_progress <= 0.0:
        decision = "INVALID_PROBE_FULL_UPDATE_HAS_NO_CE_PROGRESS"
    elif pass_local_surface:
        decision = "PASS_CERTIFIED_LOCAL_SURFACE_UPDATE_GATE"
    elif pass_local_all_steps:
        decision = "PASS_LOCAL_DIRECT_CREDIT_GATE_SPARSE_SURFACE_NOT_PASSED"
    elif pass_certificate:
        decision = "PASS_CREDIT_DEBT_SURFACE_GATE_LOCAL_UPDATE_NOT_YET"
    elif pass_oracle:
        decision = "ORACLE_ONLY_SURFACE_POTENTIAL_CERTIFICATE_MISALIGNED"
    elif any(item.actual_progress_ratio > 0.0 for item in arms):
        decision = "FAIL_CURRENT_GATE_BUT_IDENTIFYING"
    else:
        decision = "INVALID_PROBE_ALL_ARMS_HARMFUL"
    elapsed = time.perf_counter() - started
    return {
        "schema": SCHEMA,
        "gate": "original_ce_credit_debt",
        "decision": decision,
        "dataset": str(args.dataset),
        "seq_len": int(args.seq_len),
        "offset": int(args.offset),
        "vocab": int(args.vocab),
        "hidden": int(args.hidden),
        "seed": int(args.seed),
        "lr": float(args.lr),
        "base_ce": float(base_ce),
        "full_update_ce": float(full_update_ce),
        "full_progress": float(full_progress),
        "best_arm": asdict(best),
        "arms": [asdict(item) for item in arms],
        "thresholds": {
            "max_residual": float(args.max_residual),
            "min_progress": float(args.min_progress),
            "max_priced_ratio": float(args.max_priced_ratio),
            "debt_price": float(args.debt_price),
            "certificate_price": float(args.certificate_price),
            "verifier_price": float(args.verifier_price),
            "recurrent_backward_share": float(args.recurrent_backward_share),
            "min_whole_step_speedup": float(args.min_whole_step_speedup),
        },
        "probe_validity": {
            "uses_real_token_shard": True,
            "full_update_has_positive_progress": bool(full_progress > 0.0),
            "has_random_surface_control": True,
            "oracle_arm_is_upper_bound_not_implementation": True,
            "certificate_arm_uses_local_scores_only": True,
        },
        "runtime_seconds": float(elapsed),
        "no_training_integration": True,
        "no_gpu": True,
        "interpretation": (
            "A positive certificate arm supports the credit-debt active-surface "
            "mechanism. A positive local arm additionally supports certified "
            "local updates. Oracle-only means the gradient mass is sparse but the "
            "current local certificate is not good enough. This probe cannot kill "
            "warmup-dependent or mature-checkpoint variants."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("src/model_v12/test_data.bin"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42201)
    parser.add_argument("--surface-fractions", type=str, default="0.05,0.10,0.20,0.35")
    parser.add_argument("--random-controls", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--max-residual", type=float, default=0.45)
    parser.add_argument("--min-progress", type=float, default=0.85)
    parser.add_argument("--max-priced-ratio", type=float, default=0.90)
    parser.add_argument("--debt-price", type=float, default=0.20)
    parser.add_argument("--certificate-price", type=float, default=0.03)
    parser.add_argument("--verifier-price", type=float, default=0.02)
    parser.add_argument("--local-update-work-floor", type=float, default=0.08)
    parser.add_argument("--local-update-work-multiplier", type=float, default=0.35)
    parser.add_argument("--recurrent-backward-share", type=float, default=0.35)
    parser.add_argument("--min-whole-step-speedup", type=float, default=1.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_gate(args)
    print(
        "[ORIGINAL_CE_CREDIT_DEBT_GATE] "
        f"active=1 decision={payload['decision']} "
        f"seq_len={payload['seq_len']} hidden={payload['hidden']} "
        f"best_arm={payload['best_arm']['arm']} "
        f"best_priced_ratio={payload['best_arm']['priced_ratio']:.6f} "
        f"whole_step_speedup_estimate={payload['best_arm']['whole_step_speedup_estimate']:.6f} "
        f"no_training_integration=1 no_gpu=1"
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
