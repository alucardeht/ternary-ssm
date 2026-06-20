#!/usr/bin/env python3
"""R422 latent transcript marginal CE gate."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_latent_transcript_marginal_ce_gate.v1"


@dataclass
class ArmMetrics:
    arm: str
    mean_ce: float
    mean_gap_to_exact: float


@dataclass
class GateResult:
    schema: str
    decision: str
    pass_gate: bool
    mode: str
    examples: int
    steps: int
    total_transcripts: int
    mean_valid_transcripts: float
    canonical_trace_ce: float
    exact_marginal_ce: float
    marginal_gain_nats: float
    topk_bound_ce: float
    topk_gap_nats: float
    weak_checker_ce: float
    weak_false_mass: float
    enumeration_work_ratio: float
    activation_log: str
    interpretation: str
    arms: list[ArmMetrics]
    elapsed_seconds: float


def logsumexp(values: list[float]) -> float:
    if not values:
        return float("-inf")
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def enumerate_transcripts(steps: int) -> list[tuple[int, ...]]:
    return list(itertools.product((-1, 1), repeat=steps))


def transcript_answer(transcript: tuple[int, ...]) -> int:
    return sum(transcript)


def transcript_parity(transcript: tuple[int, ...]) -> int:
    return transcript_answer(transcript) & 1


def path_logprob(transcript: tuple[int, ...], step_logits: list[float]) -> float:
    total = 0.0
    for move, logit in zip(transcript, step_logits):
        p_plus = 1.0 / (1.0 + math.exp(-logit))
        p = p_plus if move == 1 else 1.0 - p_plus
        total += math.log(max(p, 1e-12))
    return total


def run_gate(args: argparse.Namespace) -> GateResult:
    start_time = time.perf_counter()
    rng = random.Random(args.seed)
    transcripts = enumerate_transcripts(args.steps)
    total_transcripts = len(transcripts)

    canonical_ces: list[float] = []
    exact_ces: list[float] = []
    topk_ces: list[float] = []
    weak_ces: list[float] = []
    weak_false_masses: list[float] = []
    valid_counts: list[int] = []

    if args.mode == "too_large" and total_transcripts <= args.max_transcripts:
        raise ValueError("too_large mode requires steps large enough to exceed max transcripts")

    for _ in range(args.examples):
        step_logits = [rng.gauss(0.0, args.logit_scale) for _ in range(args.steps)]
        hidden = tuple(1 if rng.random() < (1.0 / (1.0 + math.exp(-logit))) else -1 for logit in step_logits)
        target_answer = transcript_answer(hidden)
        target_parity = target_answer & 1
        logps = [path_logprob(transcript, step_logits) for transcript in transcripts]
        canonical_ce = -path_logprob(hidden, step_logits)

        valid_logs = [
            logp for transcript, logp in zip(transcripts, logps) if transcript_answer(transcript) == target_answer
        ]
        parity_logs = [
            logp for transcript, logp in zip(transcripts, logps) if transcript_parity(transcript) == target_parity
        ]
        valid_counts.append(len(valid_logs))
        exact_log = logsumexp(valid_logs)
        topk_log = logsumexp(sorted(valid_logs, reverse=True)[: args.topk])
        weak_log = logsumexp(parity_logs)
        weak_false = max(0.0, 1.0 - math.exp(exact_log - weak_log))

        canonical_ces.append(canonical_ce)
        exact_ces.append(-exact_log)
        topk_ces.append(-topk_log)
        weak_ces.append(-weak_log)
        weak_false_masses.append(weak_false)

    canonical_trace_ce = sum(canonical_ces) / args.examples
    exact_marginal_ce = sum(exact_ces) / args.examples
    topk_bound_ce = sum(topk_ces) / args.examples
    weak_checker_ce = sum(weak_ces) / args.examples
    marginal_gain = canonical_trace_ce - exact_marginal_ce
    topk_gap = topk_bound_ce - exact_marginal_ce
    weak_false_mass = sum(weak_false_masses) / args.examples
    mean_valid = sum(valid_counts) / args.examples
    enumeration_work_ratio = total_transcripts / max(1, args.max_transcripts)

    if args.mode == "exact":
        pass_gate = (
            total_transcripts <= args.max_transcripts
            and marginal_gain >= args.min_gain_nats
            and topk_gap >= args.min_topk_gap_nats
            and weak_false_mass >= args.min_weak_false_mass
        )
    else:
        pass_gate = False

    if pass_gate:
        decision = "PASS_LATENT_TRANSCRIPT_MARGINAL_CE_GATE"
        interpretation = (
            "Exact enumeration gives a tight latent-transcript marginal CE and "
            "shows why loose top-k or weak-checker bounds are not admissible."
        )
    elif args.mode == "topk":
        decision = "FAIL_LOOSE_VARIATIONAL_TOPK_BOUND"
        interpretation = "The top-k transcript bound remains too loose to replace exact marginalization."
    elif args.mode == "weak":
        decision = "FAIL_WEAK_LATENT_CHECKER_CONTROL"
        interpretation = "The weak checker lowers CE by accepting large invalid transcript mass."
    elif args.mode == "too_large":
        decision = "FAIL_ENUMERATION_COST_CONTROL"
        interpretation = "Exact marginalization is no longer cheap enough for the tiny-domain contract."
    else:
        decision = "FAIL_LATENT_TRANSCRIPT_MARGINAL_CE_GATE"
        interpretation = "The exact marginal gate did not clear gain, bound, or checker controls."

    arms = [
        ArmMetrics("canonical_observed_trace", canonical_trace_ce, canonical_trace_ce - exact_marginal_ce),
        ArmMetrics("exact_marginal", exact_marginal_ce, 0.0),
        ArmMetrics("topk_bound", topk_bound_ce, topk_gap),
        ArmMetrics("weak_checker_marginal", weak_checker_ce, weak_checker_ce - exact_marginal_ce),
    ]
    activation_log = (
        "[R422_LATENT_TRANSCRIPT_MARGINAL_CE_GATE] "
        f"active=1 decision={decision} mode={args.mode} steps={args.steps} "
        f"transcripts={total_transcripts} no_gpu=1 no_training_integration=1"
    )
    return GateResult(
        schema=SCHEMA,
        decision=decision,
        pass_gate=pass_gate,
        mode=args.mode,
        examples=args.examples,
        steps=args.steps,
        total_transcripts=total_transcripts,
        mean_valid_transcripts=mean_valid,
        canonical_trace_ce=canonical_trace_ce,
        exact_marginal_ce=exact_marginal_ce,
        marginal_gain_nats=marginal_gain,
        topk_bound_ce=topk_bound_ce,
        topk_gap_nats=topk_gap,
        weak_checker_ce=weak_checker_ce,
        weak_false_mass=weak_false_mass,
        enumeration_work_ratio=enumeration_work_ratio,
        activation_log=activation_log,
        interpretation=interpretation,
        arms=arms,
        elapsed_seconds=time.perf_counter() - start_time,
    )


def write_result(result: GateResult, output: Path | None) -> None:
    payload = asdict(result)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result.activation_log)
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["exact", "topk", "weak", "too_large"], default="exact")
    parser.add_argument("--examples", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--max-transcripts", type=int, default=512)
    parser.add_argument("--logit-scale", type=float, default=1.1)
    parser.add_argument("--min-gain-nats", type=float, default=1.0)
    parser.add_argument("--min-topk-gap-nats", type=float, default=0.50)
    parser.add_argument("--min-weak-false-mass", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42205)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    write_result(run_gate(args), args.output)


if __name__ == "__main__":
    main()
