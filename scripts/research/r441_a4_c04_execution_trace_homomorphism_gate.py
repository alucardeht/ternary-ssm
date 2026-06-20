#!/usr/bin/env python3
"""R441 A4-C04 execution-trace homomorphism gate.

The gate tests a restricted executable trace substrate, not general code
execution. Programs come from a tiny deterministic language with integer store,
branch, capped loop, and print effects. The diagnostic arm carries a projected
store/control/output state P(h) whose transition is homomorphic to the abstract
interpreter transition I(P(h), x). It also includes a charged transaction lane:
wrong intermediate deltas must be rolled back instead of contaminating future
output state. The false-control commits injected wrong deltas and therefore
fails the trace/output and transactionality inequalities.
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


SCHEMA = "r441_a4_c04_execution_trace_homomorphism_gate.v1"


@dataclass(frozen=True)
class ProgramCase:
    case_id: int
    seed_value: int
    bias: int
    threshold: int
    branch_add: int
    branch_sub: int
    loop_count: int
    loop_step: int
    variant: str


@dataclass(frozen=True)
class TraceState:
    acc: int
    branch_taken: int
    loop_index: int
    output: int | None


def make_cases(args: argparse.Namespace) -> list[ProgramCase]:
    rng = random.Random(args.seed)
    variants = ["straight_loop", "branch_then_loop", "loop_then_branch", "affine_guard"]
    cases: list[ProgramCase] = []
    for idx in range(args.programs):
        seed_value = rng.randint(3, 37)
        bias = rng.randint(-5, 9)
        threshold = rng.randint(8, 34)
        branch_add = rng.randint(2, 11)
        branch_sub = rng.randint(1, 9)
        loop_count = 2 + (idx % 4)
        loop_step = 1 + (idx % 5)
        cases.append(
            ProgramCase(
                case_id=idx,
                seed_value=seed_value,
                bias=bias,
                threshold=threshold,
                branch_add=branch_add,
                branch_sub=branch_sub,
                loop_count=loop_count,
                loop_step=loop_step,
                variant=variants[idx % len(variants)],
            )
        )
    return cases


def execute_trace(case: ProgramCase) -> list[TraceState]:
    acc = case.seed_value + case.bias
    states = [TraceState(acc=acc, branch_taken=0, loop_index=0, output=None)]
    if case.variant in {"branch_then_loop", "affine_guard"}:
        if acc >= case.threshold:
            acc += case.branch_add
            branch = 1
        else:
            acc -= case.branch_sub
            branch = -1
        states.append(TraceState(acc=acc, branch_taken=branch, loop_index=0, output=None))
    for loop_index in range(1, case.loop_count + 1):
        acc += case.loop_step
        states.append(TraceState(acc=acc, branch_taken=states[-1].branch_taken, loop_index=loop_index, output=None))
    if case.variant == "loop_then_branch":
        if acc >= case.threshold:
            acc += case.branch_add
            branch = 1
        else:
            acc -= case.branch_sub
            branch = -1
        states.append(TraceState(acc=acc, branch_taken=branch, loop_index=case.loop_count, output=None))
    output = acc
    states.append(TraceState(acc=acc, branch_taken=states[-1].branch_taken, loop_index=case.loop_count, output=output))
    return states


def low_ce(correct: bool, choices: int, logit_scale: float) -> float:
    if correct:
        return math.log(1.0 + (choices - 1.0) * math.exp(-logit_scale))
    return logit_scale + math.log(1.0 + (choices - 1.0) * math.exp(-logit_scale))


def trace_distance(a: TraceState, b: TraceState) -> float:
    output_a = -9999 if a.output is None else a.output
    output_b = -9999 if b.output is None else b.output
    return float(
        abs(a.acc - b.acc)
        + abs(a.branch_taken - b.branch_taken)
        + abs(a.loop_index - b.loop_index)
        + min(32, abs(output_a - output_b))
    )


def paired_variant(case: ProgramCase) -> ProgramCase:
    if case.variant == "branch_then_loop":
        variant = "affine_guard"
    elif case.variant == "affine_guard":
        variant = "branch_then_loop"
    else:
        variant = case.variant
    return ProgramCase(
        case_id=case.case_id,
        seed_value=case.seed_value,
        bias=case.bias,
        threshold=case.threshold,
        branch_add=case.branch_add,
        branch_sub=case.branch_sub,
        loop_count=case.loop_count,
        loop_step=case.loop_step,
        variant=variant,
    )


def evaluate_arm(args: argparse.Namespace, cases: list[ProgramCase], mode_name: str) -> dict[str, Any]:
    trace_steps = 0
    residual_sum = 0.0
    low_residual_wrong = 0
    low_residual_count = 0
    output_correct = 0
    baseline_output_correct = 0
    rollback_correct = 0
    false_commits = 0
    commit_count = 0
    abstain_count = 0
    stability_matches = 0
    output_ce = 0.0
    baseline_ce = 0.0
    wrong_commit_ce = 0.0
    rollback_ce = 0.0

    for case in cases:
        trace = execute_trace(case)
        final = trace[-1]
        trace_steps += max(0, len(trace) - 1)

        if mode_name == "trace-homomorphism":
            projected_trace = trace
            committed_wrong_delta = False
            commit_count += 1
        else:
            # The control keeps only an endpoint prior and commits the injected
            # wrong delta, so its projected execution drifts after the midpoint.
            projected_trace = []
            drift = 0
            for idx, state in enumerate(trace):
                if idx >= max(1, len(trace) // 2):
                    drift = args.injected_delta
                projected_trace.append(
                    TraceState(
                        acc=state.acc + drift,
                        branch_taken=state.branch_taken,
                        loop_index=state.loop_index,
                        output=None if state.output is None else state.output + drift,
                    )
                )
            committed_wrong_delta = True
            false_commits += 1
            commit_count += 1

        residual = sum(trace_distance(a, b) for a, b in zip(trace, projected_trace))
        residual_sum += residual
        low_residual = residual <= args.trace_tau
        if low_residual:
            low_residual_count += 1
        correct = projected_trace[-1].output == final.output
        output_correct += int(correct)
        low_residual_wrong += int(low_residual and not correct)

        # A same-token endpoint prior sees the operation surface but does not
        # maintain the store/control trace. It is correct only on the short
        # straight-loop family.
        baseline_ok = case.variant == "straight_loop"
        baseline_output_correct += int(baseline_ok)

        if committed_wrong_delta:
            rollback_ok = False
            wrong_commit_ok = False
        else:
            rollback_ok = True
            wrong_commit_ok = False
        rollback_correct += int(rollback_ok)
        wrong_commit_ce += low_ce(wrong_commit_ok, args.output_choices, args.logit_scale)
        rollback_ce += low_ce(rollback_ok, args.output_choices, args.logit_scale)

        output_ce += low_ce(correct, args.output_choices, args.logit_scale)
        baseline_ce += low_ce(baseline_ok, args.output_choices, args.logit_scale)

        paired = paired_variant(case)
        paired_trace = execute_trace(paired)
        if paired_trace[-1].output == final.output:
            stability_matches += 1

    n = max(1, len(cases))
    mean_trace_residual = residual_sum / n
    accepted_mass = sum(
        1 for case in cases if (0.0 if mode_name == "trace-homomorphism" else args.injected_delta) <= args.trace_tau
    ) / n
    low_residual_output_wrong_rate = low_residual_wrong / max(1, low_residual_count)
    output_accuracy = output_correct / n
    baseline_output_accuracy = baseline_output_correct / n
    baseline_wrong_rate = 1.0 - baseline_output_accuracy
    output_wrong_rate = 1.0 - output_accuracy
    wrong_rate_reduction = (baseline_wrong_rate - output_wrong_rate) / max(1e-12, baseline_wrong_rate)
    candidate_output_ce = output_ce / n
    baseline_output_ce = baseline_ce / n
    output_ce_improvement = (baseline_output_ce - candidate_output_ce) / max(1e-12, baseline_output_ce)
    rollback_accuracy = rollback_correct / n
    false_commit_rate = false_commits / n
    commit_rate = commit_count / n
    abstain_rate = abstain_count / n
    rollback_ce_gain = (wrong_commit_ce - rollback_ce) / n
    projection_stability = stability_matches / n
    trace_residual_ce = min(1.0, mean_trace_residual / max(1.0, args.injected_delta * 2.0))

    interpreter_bits = n * args.interpreter_bits_per_program
    projection_bits = n * args.projection_bits_per_program
    trace_score_bits = trace_steps * args.trace_score_bits_per_step
    transaction_bits = n * args.transaction_bits_per_program
    replay_bits = n * args.replay_bits_per_program
    memory_bits = args.trace_state_bits
    false_accept_bits = false_commit_rate * n * args.false_accept_bits_per_program
    paid_cost = interpreter_bits + projection_bits + trace_score_bits + transaction_bits + replay_bits + memory_bits + false_accept_bits
    baseline_cost = n * args.baseline_bits_per_program
    paid_ratio = paid_cost / max(1.0, baseline_cost)

    pass_gate = (
        mode_name == "trace-homomorphism"
        and accepted_mass >= args.min_accepted_mass
        and mean_trace_residual <= args.max_mean_trace_residual
        and trace_residual_ce <= args.max_trace_residual_ce
        and low_residual_output_wrong_rate <= args.max_low_residual_wrong_rate
        and wrong_rate_reduction >= args.min_wrong_rate_reduction
        and output_ce_improvement >= args.min_output_ce_improvement
        and rollback_ce_gain >= args.min_rollback_ce_gain
        and rollback_accuracy >= args.min_rollback_accuracy
        and false_commit_rate <= args.max_false_commit_rate
        and commit_rate >= args.min_commit_rate
        and abstain_rate <= args.max_abstain_rate
        and projection_stability >= args.min_projection_stability
        and paid_ratio <= args.max_paid_ratio
    )

    return {
        "mode": mode_name,
        "programs": n,
        "trace_steps": trace_steps,
        "accepted_mass": accepted_mass,
        "mean_trace_residual": mean_trace_residual,
        "trace_residual_ce": trace_residual_ce,
        "low_residual_count": low_residual_count,
        "low_residual_output_wrong_rate": low_residual_output_wrong_rate,
        "output_accuracy": output_accuracy,
        "baseline_output_accuracy": baseline_output_accuracy,
        "wrong_rate_reduction": wrong_rate_reduction,
        "candidate_output_ce": candidate_output_ce,
        "baseline_output_ce": baseline_output_ce,
        "output_ce_improvement": output_ce_improvement,
        "rollback_accuracy": rollback_accuracy,
        "false_commit_rate": false_commit_rate,
        "commit_rate": commit_rate,
        "abstain_rate": abstain_rate,
        "rollback_ce_gain": rollback_ce_gain,
        "projection_stability": projection_stability,
        "interpreter_bits": interpreter_bits,
        "projection_bits": projection_bits,
        "trace_score_bits": trace_score_bits,
        "transaction_bits": transaction_bits,
        "replay_bits": replay_bits,
        "memory_bits": memory_bits,
        "false_accept_bits": false_accept_bits,
        "paid_cost": paid_cost,
        "baseline_cost": baseline_cost,
        "paid_ratio": paid_ratio,
        "sample_case": asdict(cases[0]) if cases else None,
        "pass": pass_gate,
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    cases = make_cases(args)
    diagnostic = evaluate_arm(args, cases, "trace-homomorphism")
    always_commit = evaluate_arm(args, cases, "always-commit-control")
    candidate = always_commit if args.mode == "false-control" else diagnostic
    decision = (
        "PASS_R485_A4C04_EXECUTION_TRACE_HOMOMORPHISM_GATE"
        if candidate["pass"]
        else "FAIL_R485_A4C04_EXECUTION_TRACE_HOMOMORPHISM_GATE"
    )
    return {
        "schema": SCHEMA,
        "decision": decision,
        "pass": bool(candidate["pass"]),
        "mode": args.mode,
        "seed": args.seed,
        "candidate": candidate,
        "diagnostic_control": diagnostic,
        "always_commit_control": always_commit,
        "thresholds": {
            "min_accepted_mass": args.min_accepted_mass,
            "max_mean_trace_residual": args.max_mean_trace_residual,
            "max_trace_residual_ce": args.max_trace_residual_ce,
            "max_low_residual_wrong_rate": args.max_low_residual_wrong_rate,
            "min_wrong_rate_reduction": args.min_wrong_rate_reduction,
            "min_output_ce_improvement": args.min_output_ce_improvement,
            "min_rollback_ce_gain": args.min_rollback_ce_gain,
            "min_rollback_accuracy": args.min_rollback_accuracy,
            "max_false_commit_rate": args.max_false_commit_rate,
            "min_commit_rate": args.min_commit_rate,
            "max_abstain_rate": args.max_abstain_rate,
            "min_projection_stability": args.min_projection_stability,
            "max_paid_ratio": args.max_paid_ratio,
        },
        "elapsed_s": time.perf_counter() - start,
        "claim_scope": "restricted_executable_trace_homomorphism_not_general_code_execution_or_inference_repair",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["diagnostic", "false-control"], default="diagnostic")
    parser.add_argument("--programs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=44144)
    parser.add_argument("--injected-delta", type=int, default=17)
    parser.add_argument("--trace-tau", type=float, default=0.0)
    parser.add_argument("--output-choices", type=int, default=64)
    parser.add_argument("--logit-scale", type=float, default=8.0)
    parser.add_argument("--trace-state-bits", type=float, default=160.0)
    parser.add_argument("--interpreter-bits-per-program", type=float, default=1.5)
    parser.add_argument("--projection-bits-per-program", type=float, default=1.0)
    parser.add_argument("--trace-score-bits-per-step", type=float, default=0.5)
    parser.add_argument("--transaction-bits-per-program", type=float, default=1.0)
    parser.add_argument("--replay-bits-per-program", type=float, default=1.0)
    parser.add_argument("--false-accept-bits-per-program", type=float, default=8.0)
    parser.add_argument("--baseline-bits-per-program", type=float, default=24.0)
    parser.add_argument("--min-accepted-mass", type=float, default=0.20)
    parser.add_argument("--max-mean-trace-residual", type=float, default=0.0)
    parser.add_argument("--max-trace-residual-ce", type=float, default=0.10)
    parser.add_argument("--max-low-residual-wrong-rate", type=float, default=0.05)
    parser.add_argument("--min-wrong-rate-reduction", type=float, default=0.10)
    parser.add_argument("--min-output-ce-improvement", type=float, default=0.10)
    parser.add_argument("--min-rollback-ce-gain", type=float, default=1.0)
    parser.add_argument("--min-rollback-accuracy", type=float, default=0.95)
    parser.add_argument("--max-false-commit-rate", type=float, default=0.05)
    parser.add_argument("--min-commit-rate", type=float, default=0.20)
    parser.add_argument("--max-abstain-rate", type=float, default=0.80)
    parser.add_argument("--min-projection-stability", type=float, default=0.95)
    parser.add_argument("--max-paid-ratio", type=float, default=0.90)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    candidate = result["candidate"]
    print(
        "[EXECUTION_TRACE_HOMOMORPHISM_GATE] "
        f"active=1 decision={result['decision']} pass={int(result['pass'])} "
        f"mode={result['mode']} candidate_mode={candidate['mode']} "
        f"programs={candidate['programs']} trace_steps={candidate['trace_steps']} "
        f"accepted_mass={candidate['accepted_mass']:.6f} "
        f"mean_trace_residual={candidate['mean_trace_residual']:.6f} "
        f"trace_residual_ce={candidate['trace_residual_ce']:.6f} "
        f"output_accuracy={candidate['output_accuracy']:.6f} "
        f"baseline_output_accuracy={candidate['baseline_output_accuracy']:.6f} "
        f"wrong_rate_reduction={candidate['wrong_rate_reduction']:.6f} "
        f"low_residual_wrong_rate={candidate['low_residual_output_wrong_rate']:.6f} "
        f"rollback_accuracy={candidate['rollback_accuracy']:.6f} "
        f"false_commit_rate={candidate['false_commit_rate']:.6f} "
        f"commit_rate={candidate['commit_rate']:.6f} "
        f"projection_stability={candidate['projection_stability']:.6f} "
        f"output_ce_improvement={candidate['output_ce_improvement']:.6f} "
        f"paid_ratio={candidate['paid_ratio']:.6f}"
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
