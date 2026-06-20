#!/usr/bin/env python3
"""R422 ternary cellular automaton language-core gate.

This gate tests the smallest structural claim for R422-MS-04: a ternary local
CA must retain non-local dependencies with lower active compute than a sequence
model. A radius-1 local CA has a finite propagation cone; if it only succeeds
with sequence-length-proportional rounds, this realization fails the gate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_ternary_ca_language_gate.v1"


@dataclass
class CaseResult:
    length: int
    logarithmic_steps: int
    linear_steps: int
    log_answer_correct: bool
    linear_answer_correct: bool
    log_active_compute_ratio: float
    linear_active_compute_ratio: float


@dataclass
class GateResult:
    schema: str
    decision: str
    pass_gate: bool
    cases: int
    log_accuracy: float
    linear_accuracy: float
    mean_log_active_compute_ratio: float
    mean_linear_active_compute_ratio: float
    activation_log: str
    interpretation: str
    case_results: list[CaseResult]
    elapsed_seconds: float


def make_sequence(length: int, rng: random.Random, background: str) -> list[int]:
    seq = [0] * length
    seq[0] = rng.choice([-1, 1])
    if background == "random":
        for i in range(1, length - 1):
            seq[i] = rng.choice([-1, 0, 1])
    seq[-1] = 0
    return seq


def ca_step(state: list[int]) -> list[int]:
    """Propagate left-neighbor nonzero signal one cell right per round."""
    out = list(state)
    for i in range(1, len(state)):
        if state[i] == 0 and state[i - 1] != 0:
            out[i] = state[i - 1]
    return out


def run_ca(seq: list[int], steps: int) -> tuple[list[int], int]:
    state = list(seq)
    active_compute = 0
    for _ in range(steps):
        active_compute += len(state)
        state = ca_step(state)
    return state, active_compute


def evaluate_case(length: int, rng: random.Random, background: str) -> CaseResult:
    seq = make_sequence(length, rng, background)
    target = seq[0]
    log_steps = max(1, math.ceil(math.log2(length)))
    linear_steps = length - 1
    log_state, log_compute = run_ca(seq, log_steps)
    linear_state, linear_compute = run_ca(seq, linear_steps)
    baseline_sequence_compute = length
    return CaseResult(
        length=length,
        logarithmic_steps=log_steps,
        linear_steps=linear_steps,
        log_answer_correct=log_state[-1] == target,
        linear_answer_correct=linear_state[-1] == target,
        log_active_compute_ratio=log_compute / max(1, baseline_sequence_compute),
        linear_active_compute_ratio=linear_compute / max(1, baseline_sequence_compute),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=96)
    parser.add_argument("--min-length", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--background", choices=["zero", "random"], default="zero")
    parser.add_argument("--seed", type=int, default=42204)
    parser.add_argument("--min-log-accuracy", type=float, default=0.95)
    parser.add_argument("--max-active-compute-ratio", type=float, default=0.90)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    rng = random.Random(args.seed)
    results = [
        evaluate_case(rng.randrange(args.min_length, args.max_length + 1), rng, args.background)
        for _ in range(args.cases)
    ]
    n = max(1, len(results))
    log_acc = sum(r.log_answer_correct for r in results) / n
    linear_acc = sum(r.linear_answer_correct for r in results) / n
    mean_log_ratio = sum(r.log_active_compute_ratio for r in results) / n
    mean_linear_ratio = sum(r.linear_active_compute_ratio for r in results) / n
    pass_gate = (
        log_acc >= args.min_log_accuracy
        and mean_log_ratio <= args.max_active_compute_ratio
    )
    if pass_gate:
        decision = "PASS_TERNARY_CA_LANGUAGE_GATE"
    elif log_acc < args.min_log_accuracy and linear_acc >= args.min_log_accuracy:
        decision = "FAIL_FINITE_CONE_REQUIRES_LINEAR_STEPS"
    elif mean_log_ratio > args.max_active_compute_ratio:
        decision = "FAIL_ACTIVE_COMPUTE_PRICING"
    else:
        decision = "FAIL_CA_DEPENDENCY_RETENTION"
    activation = (
        "[R422_TERNARY_CA_LANGUAGE_GATE] "
        f"active=1 decision={decision} pass={int(pass_gate)} "
        f"radius=1 background={args.background} finite_cone=1 "
        "no_gpu=1 no_training_integration=1"
    )
    payload = GateResult(
        schema=SCHEMA,
        decision=decision,
        pass_gate=pass_gate,
        cases=len(results),
        log_accuracy=log_acc,
        linear_accuracy=linear_acc,
        mean_log_active_compute_ratio=mean_log_ratio,
        mean_linear_active_compute_ratio=mean_linear_ratio,
        activation_log=activation,
        interpretation=(
            "structural_gate_only: tests radius-1 local CA dependency transport; "
            "zero background isolates propagation; random background is an adversarial "
            "collision control; future CA variants need nonlocal shortcuts, hierarchy, "
            "or active compression."
        ),
        case_results=results,
        elapsed_seconds=time.perf_counter() - start,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(activation)
    print(
        json.dumps(
            {
                "decision": decision,
                "pass_gate": pass_gate,
                "log_accuracy": log_acc,
                "linear_accuracy": linear_acc,
                "mean_log_active_compute_ratio": mean_log_ratio,
                "mean_linear_active_compute_ratio": mean_linear_ratio,
                "background": args.background,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
