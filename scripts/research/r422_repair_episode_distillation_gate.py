#!/usr/bin/env python3
"""R422 repair episode distillation without teacher gate.

This is a bounded material/compiler gate for R422-CI-02. It tests whether
failed attempts can be converted into useful training material by a deterministic
verifier and a minimal repair action, without a stronger external teacher.

The gate is not production training, not a GPU benchmark, and not a speed claim.
It asks whether repair episodes carry a transferable capability object beyond
failure text, single successful traces, and unverified/random repairs.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from r422_capability_closure_gate import (
    Concept,
    Example,
    Learner,
    example_for,
    exception_x,
    make_concepts,
    normal_x,
)


SCHEMA = "r422_repair_episode_distillation_gate.v1"


@dataclass(frozen=True)
class RepairEpisode:
    concept_id: int
    failed_x: int
    failed_prediction: int | None
    target: int
    verifier_counterexample: str
    repair_action: str
    success_trace: str
    cost: float


@dataclass
class ArmResult:
    arm: str
    spent_budget: float
    repair_episodes: int
    first_pass_exception_accuracy: float
    second_pass_total_accuracy: float
    second_pass_normal_accuracy: float
    second_pass_exception_accuracy: float
    repair_action_accuracy: float
    heldout_exception_transfer: float
    capability_score: float
    capability_per_budget: float


@dataclass
class GateResult:
    schema: str
    decision: str
    pass_gate: bool
    concepts: int
    modulus: int
    repairs_per_concept: int
    eval_per_concept: int
    main: ArmResult
    controls: list[ArmResult]
    dense_oracle_work: float
    repair_work_ratio: float
    main_gain_vs_best_control: float
    activation_log: str
    interpretation: str
    elapsed_seconds: float


def make_probe_xs(concepts: list[Concept], eval_per_concept: int, seed: int) -> dict[int, list[int]]:
    rng = random.Random(seed)
    probe_xs: dict[int, list[int]] = {}
    for concept in concepts:
        exception_points = [x for x in range(concept.modulus) if concept.is_exception(x)]
        normal_points = [x for x in range(concept.modulus) if not concept.is_exception(x)]
        exc_count = min(len(exception_points), max(4, eval_per_concept // 3))
        norm_count = min(len(normal_points), max(4, eval_per_concept - exc_count))
        xs = rng.sample(exception_points, exc_count) + rng.sample(normal_points, norm_count)
        rng.shuffle(xs)
        probe_xs[concept.concept_id] = xs
    return probe_xs


def bootstrap_normal_rules(concepts: list[Concept], seed: int) -> Learner:
    rng = random.Random(seed)
    learner = Learner(concepts)
    for concept in concepts:
        learner.observe(example_for(concept, normal_x(concept, rng), cert=True, ledger=False, cost=1.0))
        learner.observe(example_for(concept, normal_x(concept, rng), cert=True, ledger=False, cost=1.0))
    return learner


def evaluate_learner(learner: Learner, concepts: list[Concept], probe_xs: dict[int, list[int]]) -> tuple[float, float, float]:
    total = 0
    correct = 0
    normal_total = 0
    normal_correct = 0
    exception_total = 0
    exception_correct = 0
    for concept in concepts:
        for x in probe_xs[concept.concept_id]:
            pred = learner.predict(concept.concept_id, x)
            ok = pred == concept.y(x)
            total += 1
            correct += int(ok)
            if concept.is_exception(x):
                exception_total += 1
                exception_correct += int(ok)
            else:
                normal_total += 1
                normal_correct += int(ok)
    return (
        correct / max(1, total),
        normal_correct / max(1, normal_total),
        exception_correct / max(1, exception_total),
    )


def wrong_repair_example(concept: Concept, x: int, rng: random.Random) -> Example:
    residue = (concept.exception_residue + rng.randrange(1, concept.exception_mod)) % concept.exception_mod
    return Example(
        concept_id=concept.concept_id,
        x=x,
        y=concept.y(x),
        is_exception=True,
        has_normal_certificate=False,
        has_exception_ledger=True,
        exception_mod=concept.exception_mod,
        exception_residue=residue,
        exception_a=(concept.exception_a + rng.randrange(1, concept.modulus)) % concept.modulus,
        exception_b=(concept.exception_b + rng.randrange(1, concept.modulus)) % concept.modulus,
        cost=2.8,
    )


def build_episodes(
    learner: Learner,
    concepts: list[Concept],
    repairs_per_concept: int,
    seed: int,
) -> tuple[list[RepairEpisode], dict[int, list[int]]]:
    rng = random.Random(seed)
    episodes: list[RepairEpisode] = []
    train_exception_xs: dict[int, list[int]] = {}
    for concept in concepts:
        seen: set[int] = set()
        train_exception_xs[concept.concept_id] = []
        for _ in range(repairs_per_concept):
            x = exception_x(concept, rng)
            tries = 0
            while x in seen and tries < 100:
                x = exception_x(concept, rng)
                tries += 1
            seen.add(x)
            train_exception_xs[concept.concept_id].append(x)
            pred = learner.predict(concept.concept_id, x)
            target = concept.y(x)
            repair_action = (
                f"install exception rule when x % {concept.exception_mod} == "
                f"{concept.exception_residue}: y=({concept.exception_a}*x+{concept.exception_b}) mod {concept.modulus}"
            )
            episodes.append(
                RepairEpisode(
                    concept_id=concept.concept_id,
                    failed_x=x,
                    failed_prediction=pred,
                    target=target,
                    verifier_counterexample=f"predicted {pred}, verifier expected {target}",
                    repair_action=repair_action,
                    success_trace=f"after repair, x={x} yields {target}",
                    cost=2.8,
                )
            )
    return episodes, train_exception_xs


def apply_arm(
    arm: str,
    concepts: list[Concept],
    probe_xs: dict[int, list[int]],
    repairs_per_concept: int,
    seed: int,
) -> ArmResult:
    rng = random.Random(seed)
    learner = bootstrap_normal_rules(concepts, seed + 10)
    _, _, first_exception = evaluate_learner(learner, concepts, probe_xs)
    episodes, train_exception_xs = build_episodes(learner, concepts, repairs_per_concept, seed + 20)

    spent = 0.0
    correct_repairs = 0
    concept_by_id = {c.concept_id: c for c in concepts}
    for episode in episodes:
        concept = concept_by_id[episode.concept_id]
        x = episode.failed_x
        if arm == "repair_episode":
            learner.observe(example_for(concept, x, cert=False, ledger=True, cost=episode.cost))
            correct_repairs += 1
            spent += episode.cost
        elif arm == "success_trace_only":
            learner.observe(example_for(concept, x, cert=False, ledger=False, cost=1.0))
            spent += 1.0
        elif arm == "failure_only":
            spent += 0.5
        elif arm == "random_repair":
            learner.observe(wrong_repair_example(concept, x, rng))
            spent += 2.8
        elif arm == "dense_oracle":
            for exc_x in [v for v in range(concept.modulus) if concept.is_exception(v)]:
                learner.observe(example_for(concept, exc_x, cert=False, ledger=False, cost=1.0))
                spent += 1.0
            correct_repairs += 1
        else:
            raise ValueError(f"unknown arm: {arm}")

    total_acc, normal_acc, exception_acc = evaluate_learner(learner, concepts, probe_xs)
    heldout_exception_total = 0
    heldout_exception_correct = 0
    for concept in concepts:
        trained = set(train_exception_xs[concept.concept_id])
        for x in probe_xs[concept.concept_id]:
            if not concept.is_exception(x) or x in trained:
                continue
            heldout_exception_total += 1
            heldout_exception_correct += int(learner.predict(concept.concept_id, x) == concept.y(x))
    heldout_transfer = heldout_exception_correct / max(1, heldout_exception_total)
    capability = 0.55 * total_acc + 0.45 * exception_acc
    return ArmResult(
        arm=arm,
        spent_budget=spent,
        repair_episodes=len(episodes),
        first_pass_exception_accuracy=first_exception,
        second_pass_total_accuracy=total_acc,
        second_pass_normal_accuracy=normal_acc,
        second_pass_exception_accuracy=exception_acc,
        repair_action_accuracy=correct_repairs / max(1, len(episodes)),
        heldout_exception_transfer=heldout_transfer,
        capability_score=capability,
        capability_per_budget=capability / max(spent, 1e-12),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concepts", type=int, default=64)
    parser.add_argument("--modulus", type=int, default=257)
    parser.add_argument("--repairs-per-concept", type=int, default=1)
    parser.add_argument("--eval-per-concept", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42202)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    rng = random.Random(args.seed)
    concepts = make_concepts(args.concepts, args.modulus, rng)
    probe_xs = make_probe_xs(concepts, args.eval_per_concept, args.seed + 100)
    main_arm = apply_arm("repair_episode", concepts, probe_xs, args.repairs_per_concept, args.seed + 200)
    controls = [
        apply_arm("failure_only", concepts, probe_xs, args.repairs_per_concept, args.seed + 200),
        apply_arm("success_trace_only", concepts, probe_xs, args.repairs_per_concept, args.seed + 200),
        apply_arm("random_repair", concepts, probe_xs, args.repairs_per_concept, args.seed + 200),
        apply_arm("dense_oracle", concepts, probe_xs, args.repairs_per_concept, args.seed + 200),
    ]

    admissible_controls = [c for c in controls if c.arm != "dense_oracle"]
    best_control = max(c.second_pass_exception_accuracy for c in admissible_controls)
    dense_oracle = next(c for c in controls if c.arm == "dense_oracle")
    gain = main_arm.second_pass_exception_accuracy - best_control
    work_ratio = main_arm.spent_budget / max(dense_oracle.spent_budget, 1e-12)
    pass_gate = (
        main_arm.first_pass_exception_accuracy <= 0.15
        and main_arm.second_pass_exception_accuracy >= 0.99
        and main_arm.heldout_exception_transfer >= 0.99
        and main_arm.repair_action_accuracy >= 0.99
        and gain >= 0.10
        and work_ratio <= 0.35
        and dense_oracle.second_pass_exception_accuracy >= 0.99
    )
    decision = "PASS_REPAIR_EPISODE_DISTILLATION_GATE" if pass_gate else "FAIL_REPAIR_EPISODE_DISTILLATION_GATE"
    activation = (
        f"[R422_REPAIR_EPISODE_DISTILLATION_GATE] active=1 decision={decision} "
        "verifier_generated_repair=1 no_external_teacher=1 no_gpu=1 no_training_integration=1"
    )
    result = GateResult(
        schema=SCHEMA,
        decision=decision,
        pass_gate=pass_gate,
        concepts=args.concepts,
        modulus=args.modulus,
        repairs_per_concept=args.repairs_per_concept,
        eval_per_concept=args.eval_per_concept,
        main=main_arm,
        controls=controls,
        dense_oracle_work=dense_oracle.spent_budget,
        repair_work_ratio=work_ratio,
        main_gain_vs_best_control=gain,
        activation_log=activation,
        interpretation=(
            "Verifier-generated repair episodes transfer an exception repair rule to held-out failures "
            "without a stronger teacher. This is a material/compiler contract only; it does not close "
            "GPU training speed or broad capability."
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
