#!/usr/bin/env python3
"""R422 concept-addressed non-retrieval memory gate."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_concept_addressed_memory_gate.v1"


@dataclass(frozen=True)
class Concept:
    concept_id: int
    group: int
    attrs: tuple[int, ...]
    a: int
    b: int

    def apply(self, x: int, modulus: int) -> int:
        return (self.a * x + self.b) % modulus


@dataclass(frozen=True)
class Alias:
    concept_id: int
    tokens: tuple[str, ...]
    attrs: tuple[int, ...]
    seen: bool


@dataclass
class ArmMetrics:
    arm: str
    accuracy: float
    mean_paid_work: float


@dataclass
class GateResult:
    schema: str
    decision: str
    pass_gate: bool
    concepts: int
    heldout_queries: int
    concept_memory_accuracy: float
    nearest_neighbor_accuracy: float
    soft_prompt_accuracy: float
    topic_bucket_accuracy: float
    concept_memory_work_ratio: float
    activation_log: str
    interpretation: str
    arms: list[ArmMetrics]
    elapsed_seconds: float


def random_attrs(width: int, rng: random.Random) -> tuple[int, ...]:
    return tuple(rng.randrange(2) for _ in range(width))


def make_alias(concept: Concept, alias_id: int, seen: bool, rng: random.Random, attr_noise: float) -> Alias:
    attrs = list(concept.attrs)
    for i in range(len(attrs)):
        if rng.random() < attr_noise:
            attrs[i] = 1 - attrs[i]
    prefix = "seen" if seen else "held"
    noise = rng.randrange(1_000_000)
    tokens = (
        f"{prefix}_{alias_id}_{noise}",
        f"group_{concept.group}",
        f"shape_{sum(attrs) % 5}",
    )
    return Alias(concept.concept_id, tokens, tuple(attrs), seen)


def token_overlap(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    return len(set(a) & set(b))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concepts", type=int, default=128)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--attr-width", type=int, default=10)
    parser.add_argument("--modulus", type=int, default=17)
    parser.add_argument("--seen-aliases", type=int, default=2)
    parser.add_argument("--heldout-aliases", type=int, default=3)
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--attr-noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42204)
    parser.add_argument("--min-concept-accuracy", type=float, default=0.95)
    parser.add_argument("--max-control-accuracy", type=float, default=0.55)
    parser.add_argument("--max-work-ratio", type=float, default=0.75)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    rng = random.Random(args.seed)

    concepts: list[Concept] = []
    used_attrs: set[tuple[int, ...]] = set()
    for cid in range(args.concepts):
        attrs = random_attrs(args.attr_width, rng)
        while attrs in used_attrs:
            attrs = random_attrs(args.attr_width, rng)
        used_attrs.add(attrs)
        concepts.append(
            Concept(
                concept_id=cid,
                group=cid % args.groups,
                attrs=attrs,
                a=rng.randrange(1, args.modulus),
                b=rng.randrange(args.modulus),
            )
        )
    concept_by_id = {concept.concept_id: concept for concept in concepts}
    concept_by_attrs = {concept.attrs: concept for concept in concepts}

    seen_aliases: list[Alias] = []
    heldout_aliases: list[Alias] = []
    for concept in concepts:
        for i in range(args.seen_aliases):
            seen_aliases.append(make_alias(concept, i, True, rng, 0.0))
        for i in range(args.heldout_aliases):
            heldout_aliases.append(make_alias(concept, i, False, rng, args.attr_noise))

    alias_to_concept = {alias.tokens: alias.concept_id for alias in seen_aliases}
    group_rule: dict[int, tuple[int, int]] = {}
    for group in range(args.groups):
        rules = Counter((concept.a, concept.b) for concept in concepts if concept.group == group)
        group_rule[group] = rules.most_common(1)[0][0]

    correct = defaultdict(int)
    work = defaultdict(int)
    for _ in range(args.queries):
        alias = rng.choice(heldout_aliases)
        concept = concept_by_id[alias.concept_id]
        x = rng.randrange(args.modulus)
        gold = concept.apply(x, args.modulus)

        addressed = concept_by_attrs.get(alias.attrs)
        pred = addressed.apply(x, args.modulus) if addressed else -1
        correct["concept_memory"] += int(pred == gold)
        work["concept_memory"] += args.attr_width

        nearest = max(seen_aliases, key=lambda item: token_overlap(item.tokens, alias.tokens))
        near_concept = concept_by_id[nearest.concept_id]
        correct["nearest_neighbor"] += int(near_concept.apply(x, args.modulus) == gold)
        work["nearest_neighbor"] += len(seen_aliases) * 3

        soft_id = alias_to_concept.get(alias.tokens)
        if soft_id is None:
            pred_soft = -1
        else:
            pred_soft = concept_by_id[soft_id].apply(x, args.modulus)
        correct["soft_prompt"] += int(pred_soft == gold)
        work["soft_prompt"] += 1

        a, b = group_rule[concept.group]
        correct["topic_bucket"] += int(((a * x + b) % args.modulus) == gold)
        work["topic_bucket"] += 1

    arms: list[ArmMetrics] = []
    for arm in ("concept_memory", "nearest_neighbor", "soft_prompt", "topic_bucket"):
        arms.append(
            ArmMetrics(
                arm=arm,
                accuracy=correct[arm] / args.queries,
                mean_paid_work=work[arm] / args.queries,
            )
        )

    concept_arm = next(arm for arm in arms if arm.arm == "concept_memory")
    controls = [arm for arm in arms if arm.arm != "concept_memory"]
    best_control = max(arm.accuracy for arm in controls)
    retrieval_work = next(arm.mean_paid_work for arm in arms if arm.arm == "nearest_neighbor")
    work_ratio = concept_arm.mean_paid_work / max(1e-9, retrieval_work)
    pass_gate = (
        concept_arm.accuracy >= args.min_concept_accuracy
        and best_control <= args.max_control_accuracy
        and work_ratio <= args.max_work_ratio
    )
    if pass_gate:
        decision = "PASS_CONCEPT_ADDRESSED_MEMORY_GATE"
    elif concept_arm.accuracy < args.min_concept_accuracy:
        decision = "FAIL_CONCEPT_ADDRESS_TRANSFER"
    elif best_control > args.max_control_accuracy:
        decision = "FAIL_RETRIEVAL_OR_TOPIC_CONTROL"
    else:
        decision = "FAIL_WORK_PRICING"

    activation = (
        "[R422_CONCEPT_ADDRESSED_MEMORY_GATE] "
        f"active=1 decision={decision} pass={int(pass_gate)} "
        f"attr_noise={args.attr_noise:.6f} no_gpu=1 no_training_integration=1"
    )
    payload = GateResult(
        schema=SCHEMA,
        decision=decision,
        pass_gate=pass_gate,
        concepts=args.concepts,
        heldout_queries=args.queries,
        concept_memory_accuracy=concept_arm.accuracy,
        nearest_neighbor_accuracy=next(arm.accuracy for arm in arms if arm.arm == "nearest_neighbor"),
        soft_prompt_accuracy=next(arm.accuracy for arm in arms if arm.arm == "soft_prompt"),
        topic_bucket_accuracy=next(arm.accuracy for arm in arms if arm.arm == "topic_bucket"),
        concept_memory_work_ratio=work_ratio,
        activation_log=activation,
        interpretation=(
            "controlled_material_gate_only: tests concept-addressed state activation "
            "against nearest-neighbor and soft-prompt controls; not a production v12 path."
        ),
        arms=arms,
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
                "concept_memory_accuracy": payload.concept_memory_accuracy,
                "nearest_neighbor_accuracy": payload.nearest_neighbor_accuracy,
                "soft_prompt_accuracy": payload.soft_prompt_accuracy,
                "topic_bucket_accuracy": payload.topic_bucket_accuracy,
                "concept_memory_work_ratio": payload.concept_memory_work_ratio,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
