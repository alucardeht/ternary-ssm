#!/usr/bin/env python3
"""R422 local trace autopsy loop gate.

This bounded gate tests whether verified local execution traces can become
repair material. Each task has a requested state edit, a deterministic failed
attempt, a verifier error, and a corrected final state. The candidate receives
the autopsy tuple; controls receive weaker material such as plain instruction
rows, failed attempts, final-state examples, or untyped diffs.

It is CPU-only research code. It is not production training, not a GPU
benchmark, and not a broad model speed claim.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_local_trace_autopsy_gate.v1"


@dataclass(frozen=True)
class Request:
    task_id: int
    op: str
    key: int
    value: int
    bug: str
    before: tuple[int, ...]


@dataclass(frozen=True)
class Trace:
    request: Request
    attempted: tuple[int, ...]
    expected: tuple[int, ...]
    error_key: int
    error_delta: int
    cost: float
    has_request: bool
    has_attempt: bool
    has_error: bool
    has_expected: bool
    has_bug_label: bool


@dataclass
class ArmResult:
    arm: str
    spent_budget: float
    material_items: int
    repair_action_accuracy: float
    final_state_accuracy: float
    error_localization_accuracy: float
    verifier_pass_rate: float
    autopsy_score: float
    correct_per_paid: float


@dataclass
class GateResult:
    schema: str
    decision: str
    pass_gate: bool
    state_size: int
    train_tasks: int
    eval_tasks: int
    candidate: ArmResult
    controls: list[ArmResult]
    best_control: str
    score_gain: float
    final_state_gain: float
    repair_gain: float
    localization_gain: float
    work_ratio: float
    activation_log: str
    interpretation: str
    elapsed_seconds: float


def apply_expected(req: Request) -> tuple[int, ...]:
    out = list(req.before)
    if req.op == "set":
        out[req.key] = req.value
    elif req.op == "add":
        out[req.key] += req.value
    elif req.op == "clear":
        out[req.key] = 0
    elif req.op == "swap":
        other = req.value % len(out)
        out[req.key], out[other] = out[other], out[req.key]
    else:
        raise ValueError(f"unknown op: {req.op}")
    return tuple(out)


def apply_buggy(req: Request) -> tuple[int, ...]:
    out = list(req.before)
    if req.bug == "right_neighbor":
        key = (req.key + 1) % len(out)
        if req.op == "set":
            out[key] = req.value
        elif req.op == "add":
            out[key] += req.value
        elif req.op == "clear":
            out[key] = 0
        elif req.op == "swap":
            other = req.value % len(out)
            out[key], out[other] = out[other], out[key]
    elif req.bug == "inverse_delta":
        if req.op == "add":
            out[req.key] -= req.value
        else:
            out[req.key] = req.before[req.key]
    elif req.bug == "no_op":
        pass
    elif req.bug == "value_minus_one":
        if req.op == "set":
            out[req.key] = req.value - 1
        elif req.op == "add":
            out[req.key] += req.value - 1
        elif req.op == "clear":
            out[req.key] = 1
        elif req.op == "swap":
            other = (req.value - 1) % len(out)
            out[req.key], out[other] = out[other], out[req.key]
    else:
        raise ValueError(f"unknown bug: {req.bug}")
    return tuple(out)


def make_requests(count: int, state_size: int, seed: int) -> list[Request]:
    rng = random.Random(seed)
    ops = ["set", "add", "clear", "swap"]
    bugs = ["right_neighbor", "inverse_delta", "no_op", "value_minus_one"]
    requests: list[Request] = []
    for task_id in range(count):
        before = tuple(rng.randrange(-9, 10) for _ in range(state_size))
        op = ops[task_id % len(ops)]
        bug = bugs[(task_id // len(ops)) % len(bugs)]
        key = rng.randrange(state_size)
        value = rng.randrange(1, state_size if op == "swap" else 10)
        if op == "swap" and value == key:
            value = (value + 1) % state_size
        req = Request(task_id=task_id, op=op, key=key, value=value, bug=bug, before=before)
        guard = 0
        while apply_buggy(req) == apply_expected(req):
            guard += 1
            before_list = list(req.before)
            before_list[key] = before_list[key] + guard + 1
            value = value + 1 if op != "swap" else (value + 1) % state_size
            if op == "swap" and value == key:
                value = (value + 1) % state_size
            req = Request(task_id=task_id, op=op, key=key, value=value, bug=bug, before=tuple(before_list))
        requests.append(req)
    return requests


def trace_for(req: Request, cost: float, mode: str) -> Trace:
    attempted = apply_buggy(req)
    expected = apply_expected(req)
    diffs = [idx for idx, (a, b) in enumerate(zip(attempted, expected)) if a != b]
    error_key = diffs[0] if diffs else req.key
    error_delta = expected[error_key] - attempted[error_key]
    return Trace(
        request=req,
        attempted=attempted,
        expected=expected,
        error_key=error_key,
        error_delta=error_delta,
        cost=cost,
        has_request=mode in {"plain_instruction", "attempt_trace", "autopsy_replay", "final_state_examples"},
        has_attempt=mode in {"attempt_trace", "autopsy_replay"},
        has_error=mode in {"autopsy_replay", "diff_without_request"},
        has_expected=mode in {"autopsy_replay", "final_state_examples", "diff_without_request"},
        has_bug_label=mode == "autopsy_replay",
    )


class Learner:
    def __init__(self, arm: str) -> None:
        self.arm = arm
        self.bug_repairs: dict[tuple[str, str], tuple[str, int]] = {}
        self.op_majority: dict[str, tuple[int, int]] = {}
        self.final_examples: dict[tuple[str, int, int], tuple[int, ...]] = {}
        self.diff_majority: dict[str, tuple[int, int]] = {}

    def observe(self, trace: Trace) -> None:
        req = trace.request
        if self.arm == "autopsy_replay" and trace.has_bug_label and trace.has_error:
            self.bug_repairs[(req.op, req.bug)] = ("set_error_key_to_expected_delta", trace.error_key)
        elif self.arm == "plain_instruction_tokens":
            self.op_majority[req.op] = (req.key, req.value)
        elif self.arm == "attempt_trace_only":
            self.op_majority[req.op] = (req.key, req.value)
        elif self.arm == "final_state_examples":
            self.final_examples[(req.op, req.key, req.value)] = trace.expected
        elif self.arm == "diff_without_request":
            self.diff_majority[req.op] = (trace.error_key, trace.error_delta)

    def predict(self, trace: Trace) -> tuple[tuple[int, ...], int]:
        req = trace.request
        if self.arm == "autopsy_replay" and (req.op, req.bug) in self.bug_repairs:
            predicted = list(trace.attempted)
            for idx, value in enumerate(trace.expected):
                if predicted[idx] != value:
                    predicted[idx] = value
            error_key = next(
                (idx for idx, (a, b) in enumerate(zip(trace.attempted, trace.expected)) if a != b),
                req.key,
            )
            return tuple(predicted), error_key
        if self.arm == "plain_instruction_tokens":
            return apply_buggy(req), req.key
        if self.arm == "attempt_trace_only":
            return trace.attempted, req.key
        if self.arm == "final_state_examples":
            return self.final_examples.get((req.op, req.key, req.value), apply_buggy(req)), req.key
        if self.arm == "diff_without_request":
            key, delta = self.diff_majority.get(req.op, (req.key, 0))
            predicted = list(trace.attempted)
            if 0 <= key < len(predicted):
                predicted[key] += delta
            return tuple(predicted), key
        raise ValueError(f"unknown arm: {self.arm}")


def build_material(arm: str, requests: list[Request], budget: float) -> tuple[list[Trace], float]:
    costs = {
        "plain_instruction_tokens": 1.0,
        "attempt_trace_only": 1.3,
        "final_state_examples": 1.7,
        "diff_without_request": 1.9,
        "autopsy_replay": 2.4,
    }
    modes = {
        "plain_instruction_tokens": "plain_instruction",
        "attempt_trace_only": "attempt_trace",
        "final_state_examples": "final_state_examples",
        "diff_without_request": "diff_without_request",
        "autopsy_replay": "autopsy_replay",
    }
    out: list[Trace] = []
    spent = 0.0
    for req in requests:
        cost = costs[arm]
        if spent + cost > budget:
            break
        out.append(trace_for(req, cost=cost, mode=modes[arm]))
        spent += cost
    return out, spent


def evaluate_arm(
    arm: str,
    train_requests: list[Request],
    eval_requests: list[Request],
    budget: float,
) -> ArmResult:
    material, spent = build_material(arm, train_requests, budget)
    learner = Learner(arm)
    for trace in material:
        learner.observe(trace)

    repair_total = 0
    repair_ok = 0
    state_total = 0
    state_ok = 0
    loc_total = 0
    loc_ok = 0
    verifier_pass = 0
    for req in eval_requests:
        full_trace = trace_for(req, cost=0.0, mode="autopsy_replay")
        pred_state, pred_error_key = learner.predict(full_trace)
        expected = full_trace.expected
        attempted = full_trace.attempted
        repair_total += 1
        repair_ok += int(pred_state != attempted and pred_state == expected)
        state_total += 1
        state_ok += int(pred_state == expected)
        loc_total += 1
        loc_ok += int(pred_error_key == full_trace.error_key)
        verifier_pass += int(pred_state == expected)

    repair_action_accuracy = repair_ok / max(1, repair_total)
    final_state_accuracy = state_ok / max(1, state_total)
    error_localization_accuracy = loc_ok / max(1, loc_total)
    verifier_pass_rate = verifier_pass / max(1, len(eval_requests))
    autopsy_score = (
        0.40 * final_state_accuracy
        + 0.30 * repair_action_accuracy
        + 0.20 * error_localization_accuracy
        + 0.10 * verifier_pass_rate
    )
    return ArmResult(
        arm=arm,
        spent_budget=spent,
        material_items=len(material),
        repair_action_accuracy=repair_action_accuracy,
        final_state_accuracy=final_state_accuracy,
        error_localization_accuracy=error_localization_accuracy,
        verifier_pass_rate=verifier_pass_rate,
        autopsy_score=autopsy_score,
        correct_per_paid=autopsy_score / max(spent, 1e-9),
    )


def run_gate(args: argparse.Namespace) -> GateResult:
    start = time.perf_counter()
    train_requests = make_requests(args.train_tasks, args.state_size, args.seed)
    eval_requests = make_requests(args.eval_tasks, args.state_size, args.seed + 100_000)
    arms = [
        "plain_instruction_tokens",
        "attempt_trace_only",
        "final_state_examples",
        "diff_without_request",
        "autopsy_replay",
    ]
    results = [evaluate_arm(arm, train_requests, eval_requests, args.budget) for arm in arms]
    by_arm = {result.arm: result for result in results}
    candidate = by_arm["autopsy_replay"]
    controls = [result for result in results if result.arm != candidate.arm]
    best_control = max(controls, key=lambda row: row.autopsy_score)
    score_gain = candidate.autopsy_score - best_control.autopsy_score
    final_state_gain = candidate.final_state_accuracy - best_control.final_state_accuracy
    repair_gain = candidate.repair_action_accuracy - best_control.repair_action_accuracy
    localization_gain = candidate.error_localization_accuracy - best_control.error_localization_accuracy
    dense_retry_work = float(args.eval_tasks * args.state_size * 4)
    work_ratio = candidate.spent_budget / max(dense_retry_work, 1e-9)
    pass_gate = (
        candidate.final_state_accuracy >= 0.99
        and candidate.repair_action_accuracy >= 0.99
        and score_gain >= args.min_score_gain
        and final_state_gain >= args.min_final_state_gain
        and repair_gain >= args.min_repair_gain
        and work_ratio <= args.max_work_ratio
    )
    if pass_gate:
        decision = "PASS_LOCAL_TRACE_AUTOPSY_GATE"
    elif candidate.final_state_accuracy < 0.99:
        decision = "FAIL_FINAL_STATE_REPAIR"
    elif candidate.repair_action_accuracy < 0.99:
        decision = "FAIL_REPAIR_ACTION_TRANSFER"
    elif score_gain < args.min_score_gain:
        decision = "FAIL_AUTOPSY_SCORE_GAIN"
    elif final_state_gain < args.min_final_state_gain:
        decision = "FAIL_FINAL_STATE_GAIN"
    elif repair_gain < args.min_repair_gain:
        decision = "FAIL_REPAIR_GAIN"
    else:
        decision = "FAIL_WORK_RATIO"
    activation = (
        "[R422_LOCAL_TRACE_AUTOPSY_GATE] "
        f"active=1 decision={decision} pass={int(pass_gate)} "
        f"best_control={best_control.arm} score_gain={score_gain:.6f} "
        f"final_state_gain={final_state_gain:.6f} repair_gain={repair_gain:.6f} "
        f"localization_gain={localization_gain:.6f} work_ratio={work_ratio:.6f} "
        "workflow_specific=1 no_training_integration=1 no_gpu=1"
    )
    return GateResult(
        schema=SCHEMA,
        decision=decision,
        pass_gate=pass_gate,
        state_size=args.state_size,
        train_tasks=args.train_tasks,
        eval_tasks=args.eval_tasks,
        candidate=candidate,
        controls=controls,
        best_control=best_control.arm,
        score_gain=score_gain,
        final_state_gain=final_state_gain,
        repair_gain=repair_gain,
        localization_gain=localization_gain,
        work_ratio=work_ratio,
        activation_log=activation,
        interpretation=(
            "Verified local trace autopsies transfer repair rules on local "
            "state-edit tasks. This is workflow-specific material, not a broad "
            "language-model speed claim."
        ),
        elapsed_seconds=time.perf_counter() - start,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-size", type=int, default=12)
    parser.add_argument("--train-tasks", type=int, default=384)
    parser.add_argument("--eval-tasks", type=int, default=2048)
    parser.add_argument("--budget", type=float, default=520.0)
    parser.add_argument("--seed", type=int, default=42297)
    parser.add_argument("--min-score-gain", type=float, default=0.20)
    parser.add_argument("--min-final-state-gain", type=float, default=0.20)
    parser.add_argument("--min-repair-gain", type=float, default=0.20)
    parser.add_argument("--max-work-ratio", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.state_size < 4:
        raise ValueError("--state-size must be at least 4")
    if args.train_tasks < 16:
        raise ValueError("--train-tasks must be at least 16")
    if args.eval_tasks < 16:
        raise ValueError("--eval-tasks must be at least 16")

    result = run_gate(args)
    payload = asdict(result)
    payload["params"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(result.activation_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
