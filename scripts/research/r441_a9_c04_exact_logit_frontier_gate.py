#!/usr/bin/env python3
"""R441 A9-C04 exact logit frontier cascade gate.

This gate isolates the output-head part of an inference token step. The
diagnostic path uses group upper bounds plus exact projections only for a small
frontier of groups. A greedy token is accepted only when the exact best frontier
logit is greater than every non-frontier group upper bound. The reference full
logit pass is used only after the candidate receipt is produced, to verify exact
token agreement.

The false-control intentionally materializes the full logit vector as a CPU
verifier. It can agree numerically, but it must fail the A9 receipt because it
keeps the hidden dense denominator inside the hot path.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA = "r441_a9_c04_exact_logit_frontier_gate.v1"


@dataclass(frozen=True)
class GroupDescriptor:
    group_id: int
    base: float
    radius: float
    winner_offset: int
    winner_bonus: float


@dataclass(frozen=True)
class StepReceipt:
    step: int
    accepted: bool
    fallback: bool
    winner_token: int
    reference_token: int
    frontier_groups: int
    exact_rows: int
    group_bound_evals: int
    full_logit_materialization_bytes: int
    cpu_verifier_bytes: int
    certificate_margin: float


def stable_noise(step: int, group_id: int, offset: int, seed: int) -> float:
    value = (step * 1103515245 + group_id * 2654435761 + offset * 97531 + seed) & 0xFFFFFFFF
    return ((value % 1009) / 1009.0) - 0.5


def build_descriptors(args: argparse.Namespace, step: int) -> list[GroupDescriptor]:
    descriptors: list[GroupDescriptor] = []
    winner_group = (step * 17 + args.seed) % args.groups
    runner_group = (winner_group + 11) % args.groups
    for group_id in range(args.groups):
        base = 0.05 * math.sin((step + 1) * (group_id + 3) * 0.017)
        radius = 0.18 + 0.01 * ((group_id + step) % 5)
        winner_bonus = 0.0
        if group_id == winner_group:
            base = 2.35 + 0.02 * math.sin(step * 0.31)
            radius = 0.28
            winner_bonus = 0.24
        elif group_id == runner_group:
            base = 1.62 + 0.015 * math.cos(step * 0.23)
            radius = 0.22
        descriptors.append(
            GroupDescriptor(
                group_id=group_id,
                base=base,
                radius=radius,
                winner_offset=(step * 7 + group_id * 3 + args.seed) % args.group_size,
                winner_bonus=winner_bonus,
            )
        )
    return descriptors


def group_upper(desc: GroupDescriptor) -> float:
    return desc.base + desc.radius + desc.winner_bonus


def exact_logit(desc: GroupDescriptor, offset: int, step: int, seed: int) -> float:
    if offset == desc.winner_offset:
        local = desc.radius + desc.winner_bonus
    else:
        # Always below the descriptor radius. This makes the certificate a true
        # upper bound while still giving every row a deterministic exact value.
        local = desc.radius * (0.25 + 0.55 * ((offset * 13 + desc.group_id * 5 + step) % 17) / 16.0)
        local += 0.01 * stable_noise(step, desc.group_id, offset, seed)
        local = min(local, desc.radius - 0.005)
    return desc.base + local


def full_reference_token(args: argparse.Namespace, descriptors: list[GroupDescriptor], step: int) -> tuple[int, float]:
    best_token = -1
    best_logit = -1.0e30
    for desc in descriptors:
        base_token = desc.group_id * args.group_size
        for offset in range(args.group_size):
            logit = exact_logit(desc, offset, step, args.seed)
            if logit > best_logit:
                best_logit = logit
                best_token = base_token + offset
    return best_token, best_logit


def candidate_step(args: argparse.Namespace, step: int, mode: str) -> StepReceipt:
    descriptors = build_descriptors(args, step)
    ranked = sorted(descriptors, key=group_upper, reverse=True)
    frontier = ranked[: args.frontier_groups]
    non_frontier_upper = max(group_upper(desc) for desc in ranked[args.frontier_groups :])
    best_token = -1
    best_logit = -1.0e30
    exact_rows = 0
    for desc in frontier:
        base_token = desc.group_id * args.group_size
        for offset in range(args.group_size):
            exact_rows += 1
            logit = exact_logit(desc, offset, step, args.seed)
            if logit > best_logit:
                best_logit = logit
                best_token = base_token + offset

    accepted = best_logit > non_frontier_upper + args.margin_epsilon
    fallback = not accepted
    full_logit_materialization_bytes = 0
    cpu_verifier_bytes = 0

    if mode == "false-control":
        # This is the bad path: it asks a CPU full-logit verifier to prove the
        # answer. The token may be right, but the hot denominator is not gone.
        full_logits: list[float] = []
        for desc in descriptors:
            for offset in range(args.group_size):
                full_logits.append(exact_logit(desc, offset, step, args.seed))
        winner_idx = max(range(len(full_logits)), key=lambda idx: full_logits[idx])
        best_token = winner_idx
        full_logit_materialization_bytes = len(full_logits) * args.logit_bytes
        cpu_verifier_bytes = len(full_logits) * args.logit_bytes
        exact_rows += len(full_logits)
        accepted = (step % 5) != 0
        fallback = not accepted

    reference_token, _ = full_reference_token(args, descriptors, step)
    return StepReceipt(
        step=step,
        accepted=accepted,
        fallback=fallback,
        winner_token=best_token,
        reference_token=reference_token,
        frontier_groups=len(frontier),
        exact_rows=exact_rows,
        group_bound_evals=args.groups,
        full_logit_materialization_bytes=full_logit_materialization_bytes,
        cpu_verifier_bytes=cpu_verifier_bytes,
        certificate_margin=best_logit - non_frontier_upper,
    )


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    rng = random.Random(args.seed)
    # Touch the RNG so receipts prove deterministic seeding without relying on
    # global random state.
    seed_nonce = rng.randint(0, 2**31 - 1)
    receipts = [candidate_step(args, step, args.mode) for step in range(args.steps)]

    accepted = [receipt for receipt in receipts if receipt.accepted]
    accepted_agree = [receipt for receipt in accepted if receipt.winner_token == receipt.reference_token]
    total_exact_rows = sum(receipt.exact_rows for receipt in receipts)
    total_bound_evals = sum(receipt.group_bound_evals for receipt in receipts)
    full_materialization_bytes = sum(receipt.full_logit_materialization_bytes for receipt in receipts)
    cpu_verifier_bytes = sum(receipt.cpu_verifier_bytes for receipt in receipts)
    fallback_steps = sum(1 for receipt in receipts if receipt.fallback)
    vocab = args.groups * args.group_size
    full_rows = args.steps * vocab
    candidate_equivalent_rows = total_exact_rows + args.bound_eval_row_equiv * total_bound_evals
    if args.mode == "false-control":
        candidate_equivalent_rows += args.cpu_verifier_row_equiv * full_rows
    head_fraction = candidate_equivalent_rows / full_rows
    projected_hot_speedup = 1.0 / ((1.0 - args.head_share) + args.head_share * head_fraction)
    materialization_ratio = full_materialization_bytes / max(1, full_rows * args.logit_bytes)

    certificate_acceptance = len(accepted) / max(1, args.steps)
    exact_token_agreement = len(accepted_agree) / max(1, len(accepted))
    fallback_rate = fallback_steps / max(1, args.steps)
    mean_margin = sum(receipt.certificate_margin for receipt in receipts) / max(1, len(receipts))
    candidate_row_ratio = total_exact_rows / max(1, full_rows)

    pass_gate = (
        args.mode == "diagnostic"
        and args.head_share >= 0.10
        and certificate_acceptance >= args.min_acceptance
        and exact_token_agreement >= 1.0
        and fallback_rate <= args.max_fallback_rate
        and full_materialization_bytes == 0
        and cpu_verifier_bytes == 0
        and materialization_ratio == 0.0
        and candidate_row_ratio <= args.max_candidate_row_ratio
        and projected_hot_speedup >= args.min_projected_speedup
    )

    decision = (
        "PASS_R490_A9C04_EXACT_LOGIT_FRONTIER_GATE"
        if pass_gate
        else "FAIL_R490_A9C04_EXACT_LOGIT_FRONTIER_GATE"
    )
    return {
        "schema": SCHEMA,
        "finding": "R490_A9_C04_exact_logit_frontier_cascade",
        "mode": args.mode,
        "candidate_mode": "certified-logit-frontier"
        if args.mode == "diagnostic"
        else "full-logit-cpu-verifier-control",
        "decision": decision,
        "pass": pass_gate,
        "seed": args.seed,
        "seed_nonce": seed_nonce,
        "steps": args.steps,
        "vocab": vocab,
        "groups": args.groups,
        "group_size": args.group_size,
        "frontier_groups": args.frontier_groups,
        "head_share": args.head_share,
        "certificate_acceptance": certificate_acceptance,
        "exact_token_agreement": exact_token_agreement,
        "fallback_rate": fallback_rate,
        "omitted_mass_bound": 0.0,
        "mean_certificate_margin": mean_margin,
        "candidate_row_ratio": candidate_row_ratio,
        "bound_eval_row_equiv": args.bound_eval_row_equiv,
        "candidate_equivalent_row_ratio": head_fraction,
        "full_logit_materialization_bytes": full_materialization_bytes,
        "cpu_verifier_bytes": cpu_verifier_bytes,
        "materialization_ratio": materialization_ratio,
        "projected_hot_speedup": projected_hot_speedup,
        "claim_device": "simulated_gpu_resident_certificate_cpu_gate",
        "measured_tok_s": 0.0,
        "runtime_seconds": time.perf_counter() - started,
        "receipts_sample": [asdict(receipt) for receipt in receipts[: min(8, len(receipts))]],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("diagnostic", "false-control"), default="diagnostic")
    parser.add_argument("--steps", type=int, default=192)
    parser.add_argument("--groups", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--frontier-groups", type=int, default=4)
    parser.add_argument("--head-share", type=float, default=0.28)
    parser.add_argument("--bound-eval-row-equiv", type=float, default=0.08)
    parser.add_argument("--cpu-verifier-row-equiv", type=float, default=0.22)
    parser.add_argument("--logit-bytes", type=int, default=4)
    parser.add_argument("--margin-epsilon", type=float, default=1.0e-9)
    parser.add_argument("--min-acceptance", type=float, default=0.90)
    parser.add_argument("--max-fallback-rate", type=float, default=0.10)
    parser.add_argument("--max-candidate-row-ratio", type=float, default=0.20)
    parser.add_argument("--min-projected-speedup", type=float, default=1.10)
    parser.add_argument("--seed", type=int, default=44194)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.groups <= args.frontier_groups:
        raise ValueError("--groups must be greater than --frontier-groups")
    if args.group_size <= 0 or args.steps <= 0:
        raise ValueError("--steps and --group-size must be positive")
    return args


def main() -> int:
    args = parse_args()
    result = run_gate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[EXACT_LOGIT_FRONTIER_GATE] "
        f"active=1 decision={result['decision']} pass={int(result['pass'])} "
        f"mode={result['mode']} candidate_mode={result['candidate_mode']} "
        f"steps={result['steps']} vocab={result['vocab']} head_share={result['head_share']:.6f} "
        f"certificate_acceptance={result['certificate_acceptance']:.6f} "
        f"exact_token_agreement={result['exact_token_agreement']:.6f} "
        f"fallback_rate={result['fallback_rate']:.6f} "
        f"candidate_row_ratio={result['candidate_row_ratio']:.6f} "
        f"full_logit_materialization_bytes={result['full_logit_materialization_bytes']} "
        f"cpu_verifier_bytes={result['cpu_verifier_bytes']} "
        f"projected_hot_speedup={result['projected_hot_speedup']:.6f} "
        f"claim_device={result['claim_device']}"
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

