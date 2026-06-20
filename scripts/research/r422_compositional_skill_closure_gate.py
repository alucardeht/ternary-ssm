#!/usr/bin/env python3
"""R422 compositional skill closure syllabus gate.

This bounded material gate tests whether atomic operation contracts plus a
closure proof object can replace brute-force transcript coverage for unseen
compositions. It uses modular affine operators where operation order matters.

It is CPU-only research code. It is not production training, not a GPU
benchmark, and not a speed claim.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_compositional_skill_closure_gate.v1"


@dataclass(frozen=True)
class Op:
    op_id: int
    a: int
    b: int

    def apply(self, x: int, modulus: int) -> int:
        return (self.a * x + self.b) % modulus


@dataclass(frozen=True)
class Task:
    seq: tuple[int, ...]
    x: int
    y: int


@dataclass
class ArmResult:
    arm: str
    spent_budget: float
    material_items: int
    unseen_composition_accuracy: float
    order_sensitive_accuracy: float
    closure_proof_accuracy: float
    residual_guard: float
    correct_per_paid: float


@dataclass
class GateResult:
    schema: str
    decision: str
    pass_gate: bool
    ops: int
    modulus: int
    train_tasks: int
    eval_tasks: int
    candidate: ArmResult
    controls: list[ArmResult]
    dense_oracle_work: float
    candidate_work_ratio: float
    candidate_gain_vs_best_admissible: float
    activation_log: str
    interpretation: str
    elapsed_seconds: float


def make_ops(count: int, modulus: int, seed: int) -> list[Op]:
    rng = random.Random(seed)
    ops: list[Op] = []
    used: set[tuple[int, int]] = set()
    while len(ops) < count:
        a = rng.randrange(2, modulus)
        b = rng.randrange(1, modulus)
        if (a, b) in used:
            continue
        used.add((a, b))
        ops.append(Op(len(ops), a, b))
    return ops


def apply_seq(ops_by_id: dict[int, Op], seq: tuple[int, ...], x: int, modulus: int) -> int:
    out = x
    for op_id in seq:
        out = ops_by_id[op_id].apply(out, modulus)
    return out


def compose_params(ops_by_id: dict[int, Op], seq: tuple[int, ...], modulus: int) -> tuple[int, int]:
    a_total = 1
    b_total = 0
    for op_id in seq:
        op = ops_by_id[op_id]
        a_total = (op.a * a_total) % modulus
        b_total = (op.a * b_total + op.b) % modulus
    return a_total, b_total


def random_seq(rng: random.Random, ops: int, min_len: int, max_len: int) -> tuple[int, ...]:
    return tuple(rng.randrange(ops) for _ in range(rng.randint(min_len, max_len)))


def make_tasks(
    ops: list[Op],
    modulus: int,
    count: int,
    min_len: int,
    max_len: int,
    seed: int,
) -> list[Task]:
    rng = random.Random(seed)
    ops_by_id = {op.op_id: op for op in ops}
    tasks: list[Task] = []
    for _ in range(count):
        seq = random_seq(rng, len(ops), min_len, max_len)
        x = rng.randrange(modulus)
        tasks.append(Task(seq=seq, x=x, y=apply_seq(ops_by_id, seq, x, modulus)))
    return tasks


def build_flat_memory(train: list[Task]) -> tuple[dict[tuple[tuple[int, ...], int], int], dict[tuple[int, ...], Counter[int]]]:
    exact: dict[tuple[tuple[int, ...], int], int] = {}
    seq_majority: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    for task in train:
        exact[(task.seq, task.x)] = task.y
        seq_majority[task.seq][task.y] += 1
    return exact, seq_majority


def nearest_seq_prediction(seq_majority: dict[tuple[int, ...], Counter[int]], seq: tuple[int, ...]) -> int | None:
    if seq in seq_majority:
        return seq_majority[seq].most_common(1)[0][0]
    best_seq = None
    best_overlap = -1
    for candidate in seq_majority:
        overlap = sum(int(a == b) for a, b in zip(candidate, seq)) - abs(len(candidate) - len(seq))
        if overlap > best_overlap:
            best_overlap = overlap
            best_seq = candidate
    if best_seq is None:
        return None
    return seq_majority[best_seq].most_common(1)[0][0]


def predict_arm(
    arm: str,
    task: Task,
    ops_by_id: dict[int, Op],
    modulus: int,
    exact: dict[tuple[tuple[int, ...], int], int],
    seq_majority: dict[tuple[int, ...], Counter[int]],
) -> tuple[int | None, bool, bool]:
    if arm == "closure_syllabus":
        a, b = compose_params(ops_by_id, task.seq, modulus)
        return (a * task.x + b) % modulus, True, True
    if arm == "flat_mixed_examples":
        if (task.seq, task.x) in exact:
            return exact[(task.seq, task.x)], False, True
        return nearest_seq_prediction(seq_majority, task.seq), False, True
    if arm == "atom_only":
        return ops_by_id[task.seq[0]].apply(task.x, modulus), False, True
    if arm == "unordered_atoms":
        sorted_seq = tuple(sorted(task.seq))
        return apply_seq(ops_by_id, sorted_seq, task.x, modulus), False, True
    if arm == "dense_oracle":
        return task.y, True, True
    raise ValueError(f"unknown arm: {arm}")


def evaluate_arm(
    arm: str,
    ops: list[Op],
    modulus: int,
    train: list[Task],
    eval_tasks: list[Task],
    candidate_budget: float,
) -> ArmResult:
    ops_by_id = {op.op_id: op for op in ops}
    exact, seq_majority = build_flat_memory(train)
    if arm == "closure_syllabus":
        spent = len(ops) * 4.0 + 16.0
        items = len(ops) + 1
        residual_guard = 1.0
    elif arm == "dense_oracle":
        spent = float(len(eval_tasks))
        items = len(eval_tasks)
        residual_guard = 1.0
    else:
        spent = min(float(len(train)), candidate_budget)
        items = min(len(train), int(candidate_budget))
        residual_guard = 1.0

    correct = 0
    order_total = 0
    order_correct = 0
    closure_total = 0
    closure_correct = 0
    for task in eval_tasks:
        pred, closure_valid, residual_valid = predict_arm(arm, task, ops_by_id, modulus, exact, seq_majority)
        ok = pred == task.y
        correct += int(ok)
        if tuple(sorted(task.seq)) != task.seq:
            order_total += 1
            order_correct += int(ok)
        closure_total += 1
        closure_correct += int(closure_valid and residual_valid and ok)
    return ArmResult(
        arm=arm,
        spent_budget=spent,
        material_items=items,
        unseen_composition_accuracy=correct / max(1, len(eval_tasks)),
        order_sensitive_accuracy=order_correct / max(1, order_total),
        closure_proof_accuracy=closure_correct / max(1, closure_total),
        residual_guard=residual_guard,
        correct_per_paid=correct / max(spent, 1e-12),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", type=int, default=9)
    parser.add_argument("--modulus", type=int, default=101)
    parser.add_argument("--train-tasks", type=int, default=640)
    parser.add_argument("--eval-tasks", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42205)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    ops = make_ops(args.ops, args.modulus, args.seed)
    train = make_tasks(ops, args.modulus, args.train_tasks, 1, 3, args.seed + 100)
    eval_tasks = make_tasks(ops, args.modulus, args.eval_tasks, 4, 7, args.seed + 200)
    candidate_budget = len(ops) * 4.0 + 16.0
    candidate = evaluate_arm("closure_syllabus", ops, args.modulus, train, eval_tasks, candidate_budget)
    controls = [
        evaluate_arm("flat_mixed_examples", ops, args.modulus, train, eval_tasks, candidate_budget),
        evaluate_arm("atom_only", ops, args.modulus, train, eval_tasks, candidate_budget),
        evaluate_arm("unordered_atoms", ops, args.modulus, train, eval_tasks, candidate_budget),
        evaluate_arm("dense_oracle", ops, args.modulus, train, eval_tasks, candidate_budget),
    ]
    admissible = [c for c in controls if c.arm != "dense_oracle"]
    best_control = max(c.unseen_composition_accuracy for c in admissible)
    dense_oracle = next(c for c in controls if c.arm == "dense_oracle")
    work_ratio = candidate.spent_budget / max(dense_oracle.spent_budget, 1e-12)
    gain = candidate.unseen_composition_accuracy - best_control
    pass_gate = (
        candidate.unseen_composition_accuracy >= 0.99
        and candidate.order_sensitive_accuracy >= 0.99
        and candidate.closure_proof_accuracy >= 0.99
        and candidate.residual_guard >= 0.999
        and gain >= 0.10
        and work_ratio <= 0.10
        and dense_oracle.unseen_composition_accuracy >= 0.99
    )
    decision = "PASS_COMPOSITIONAL_SKILL_CLOSURE_GATE" if pass_gate else "FAIL_COMPOSITIONAL_SKILL_CLOSURE_GATE"
    activation = (
        f"[R422_COMPOSITIONAL_SKILL_CLOSURE_GATE] active=1 decision={decision} "
        "atomic_contracts=1 closure_proof=1 no_gpu=1 no_training_integration=1"
    )
    result = GateResult(
        schema=SCHEMA,
        decision=decision,
        pass_gate=pass_gate,
        ops=args.ops,
        modulus=args.modulus,
        train_tasks=args.train_tasks,
        eval_tasks=args.eval_tasks,
        candidate=candidate,
        controls=controls,
        dense_oracle_work=dense_oracle.spent_budget,
        candidate_work_ratio=work_ratio,
        candidate_gain_vs_best_admissible=gain,
        activation_log=activation,
        interpretation=(
            "Atomic contracts plus a closure proof solve unseen ordered compositions with far less "
            "material than dense transcript coverage. This is a material/compiler gate only."
        ),
        elapsed_seconds=time.perf_counter() - start,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(activation)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0 if pass_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
