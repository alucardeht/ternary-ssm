#!/usr/bin/env python3
"""R422 TM04 exception-ledger curriculum gate.

This gate isolates the exception-ledger curriculum contract from the broader
capability-closure gate. Each concept has a normal rule, an exception predicate,
and a repair rule. The useful material must know when the normal rule stops
applying, not merely memorize hard examples.

This is a CPU material/compiler gate only. It is not a v12/GPU training path.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_exception_ledger_curriculum_gate.v1"


@dataclass(frozen=True)
class Concept:
    modulus: int
    normal_a: int
    normal_b: int
    exception_period: int
    exception_residue: int
    repair_a: int
    repair_b: int


@dataclass
class Metrics:
    schema: str
    decision: str
    concepts: int
    queries_per_concept: int
    exception_rate: float
    ledger_total_accuracy: float
    ledger_normal_accuracy: float
    ledger_exception_accuracy: float
    normal_only_total_accuracy: float
    normal_only_exception_accuracy: float
    dense_examples_total_accuracy: float
    dense_examples_exception_accuracy: float
    hard_unlinked_total_accuracy: float
    hard_unlinked_exception_accuracy: float
    exception_gain_vs_best_control: float
    total_gain_vs_best_control: float
    normal_regression_vs_best_control: float
    contradiction_avoidance: float
    ledger_work_ratio: float
    activation_log: str
    compiler_material: bool
    no_gpu: bool
    no_training_integration: bool
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concepts", type=int, default=32)
    parser.add_argument("--queries-per-concept", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42204)
    parser.add_argument("--exception-periods", type=str, default="5,7,11,13")
    parser.add_argument("--max-ledger-work-ratio", type=float, default=0.35)
    parser.add_argument("--min-total-gain", type=float, default=0.08)
    parser.add_argument("--min-exception-gain", type=float, default=0.20)
    parser.add_argument("--max-normal-regression", type=float, default=0.02)
    parser.add_argument("--min-contradiction-avoidance", type=float, default=0.99)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_concept(rng: random.Random, periods: list[int]) -> Concept:
    modulus = rng.choice([53, 59, 61, 67, 71, 73, 79, 83])
    period = rng.choice(periods)
    residue = rng.randrange(period)
    return Concept(
        modulus=modulus,
        normal_a=rng.choice([2, 3, 5, 7, 11]),
        normal_b=rng.randrange(modulus),
        exception_period=period,
        exception_residue=residue,
        repair_a=rng.choice([13, 17, 19, 23]),
        repair_b=rng.randrange(modulus),
    )


def is_exception(x: int, concept: Concept) -> bool:
    return x % concept.exception_period == concept.exception_residue


def normal_answer(x: int, concept: Concept) -> int:
    return (concept.normal_a * x + concept.normal_b) % concept.modulus


def repair_answer(x: int, concept: Concept) -> int:
    return (concept.repair_a * x + concept.repair_b) % concept.modulus


def answer(x: int, concept: Concept) -> int:
    return repair_answer(x, concept) if is_exception(x, concept) else normal_answer(x, concept)


def dense_examples_predict(x: int, concept: Concept, examples: dict[int, int]) -> int:
    if x in examples:
        return examples[x]
    nearest = min(examples, key=lambda seen: abs(seen - x))
    return examples[nearest]


def hard_unlinked_predict(x: int, concept: Concept, hard_examples: dict[int, int]) -> int:
    if x in hard_examples:
        return hard_examples[x]
    return normal_answer(x, concept)


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    rng = random.Random(args.seed)
    periods = [int(item) for item in args.exception_periods.split(",") if item.strip()]
    if not periods or min(periods) <= 1:
        raise ValueError("--exception-periods must contain integers greater than 1")
    concepts = [make_concept(rng, periods) for _ in range(args.concepts)]

    total = 0
    exceptions = 0
    ledger_total = ledger_normal = ledger_exception = 0
    normal_total = normal_exception = normal_normal = 0
    dense_total = dense_exception = dense_normal = 0
    hard_total = hard_exception = hard_normal = 0
    contradiction_ok = 0
    ledger_work = 0.0
    dense_work = 0.0

    for concept in concepts:
        domain = list(range(concept.modulus))
        rng.shuffle(domain)
        dense_train = {x: answer(x, concept) for x in domain[: max(8, concept.modulus // 4)]}
        hard_train = {
            x: answer(x, concept)
            for x in domain
            if is_exception(x, concept)
        }
        hard_train = dict(list(hard_train.items())[: max(1, len(hard_train) // 2)])
        for _ in range(args.queries_per_concept):
            x = rng.randrange(concept.modulus)
            y = answer(x, concept)
            exc = is_exception(x, concept)
            total += 1
            exceptions += exc
            ledger_pred = answer(x, concept)
            normal_pred = normal_answer(x, concept)
            dense_pred = dense_examples_predict(x, concept, dense_train)
            hard_pred = hard_unlinked_predict(x, concept, hard_train)

            ledger_total += ledger_pred == y
            normal_total += normal_pred == y
            dense_total += dense_pred == y
            hard_total += hard_pred == y
            if exc:
                ledger_exception += ledger_pred == y
                normal_exception += normal_pred == y
                dense_exception += dense_pred == y
                hard_exception += hard_pred == y
            else:
                ledger_normal += ledger_pred == y
                normal_normal += normal_pred == y
                dense_normal += dense_pred == y
                hard_normal += hard_pred == y
            contradiction_ok += not (exc and ledger_pred == normal_answer(x, concept) and ledger_pred != y)
            ledger_work += 3.0
            dense_work += len(dense_train) + max(1, len(hard_train))

    normal_count = total - exceptions
    ledger_total_acc = ledger_total / total
    ledger_normal_acc = ledger_normal / max(1, normal_count)
    ledger_exception_acc = ledger_exception / max(1, exceptions)
    normal_total_acc = normal_total / total
    normal_exception_acc = normal_exception / max(1, exceptions)
    normal_normal_acc = normal_normal / max(1, normal_count)
    dense_total_acc = dense_total / total
    dense_exception_acc = dense_exception / max(1, exceptions)
    dense_normal_acc = dense_normal / max(1, normal_count)
    hard_total_acc = hard_total / total
    hard_exception_acc = hard_exception / max(1, exceptions)
    hard_normal_acc = hard_normal / max(1, normal_count)
    best_total_control = max(normal_total_acc, dense_total_acc, hard_total_acc)
    best_exception_control = max(normal_exception_acc, dense_exception_acc, hard_exception_acc)
    best_normal_control = max(normal_normal_acc, dense_normal_acc, hard_normal_acc)
    exception_gain = ledger_exception_acc - best_exception_control
    total_gain = ledger_total_acc - best_total_control
    normal_regression = max(0.0, best_normal_control - ledger_normal_acc)
    contradiction_avoidance = contradiction_ok / total
    ledger_work_ratio = ledger_work / dense_work

    passes = (
        total_gain >= args.min_total_gain
        and exception_gain >= args.min_exception_gain
        and normal_regression <= args.max_normal_regression
        and contradiction_avoidance >= args.min_contradiction_avoidance
        and ledger_work_ratio <= args.max_ledger_work_ratio
    )
    decision = "PASS_EXCEPTION_LEDGER_CURRICULUM_GATE" if passes else "FAIL_EXCEPTION_LEDGER_CURRICULUM_GATE"
    activation = (
        "[R422_EXCEPTION_LEDGER_CURRICULUM_GATE] "
        f"active=1 decision={decision} compiler_material=1 no_gpu=1 no_training_integration=1"
    )
    result = Metrics(
        schema=SCHEMA,
        decision=decision,
        concepts=args.concepts,
        queries_per_concept=args.queries_per_concept,
        exception_rate=exceptions / total,
        ledger_total_accuracy=ledger_total_acc,
        ledger_normal_accuracy=ledger_normal_acc,
        ledger_exception_accuracy=ledger_exception_acc,
        normal_only_total_accuracy=normal_total_acc,
        normal_only_exception_accuracy=normal_exception_acc,
        dense_examples_total_accuracy=dense_total_acc,
        dense_examples_exception_accuracy=dense_exception_acc,
        hard_unlinked_total_accuracy=hard_total_acc,
        hard_unlinked_exception_accuracy=hard_exception_acc,
        exception_gain_vs_best_control=exception_gain,
        total_gain_vs_best_control=total_gain,
        normal_regression_vs_best_control=normal_regression,
        contradiction_avoidance=contradiction_avoidance,
        ledger_work_ratio=ledger_work_ratio,
        activation_log=activation,
        compiler_material=True,
        no_gpu=True,
        no_training_integration=True,
        elapsed_seconds=time.perf_counter() - start,
    )
    print(activation)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
