#!/usr/bin/env python3
"""R422 minimal causal intervention pair gate.

This bounded material gate tests whether proof-carrying intervention pairs can
identify the one causal variable in a confounded training stream. The fixture is
constructed so ordinary examples and non-minimal counterfactuals leave a stable
distractor perfectly correlated with the true causal variable during training.

It is CPU-only research code. It is not production training, not a GPU
benchmark, and not a speed claim.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_minimal_causal_intervention_gate.v1"


@dataclass(frozen=True)
class Concept:
    concept_id: int
    modulus: int
    causal_idx: int
    spurious_idx: int
    spurious_offset: int
    a: int
    b: int

    def y(self, xs: tuple[int, ...]) -> int:
        return (self.a * xs[self.causal_idx] + self.b) % self.modulus

    def make_confounded(self, causal_x: int, rng: random.Random, dims: int) -> tuple[int, ...]:
        xs = [rng.randrange(self.modulus) for _ in range(dims)]
        xs[self.causal_idx] = causal_x
        for idx in range(dims):
            if idx == self.causal_idx:
                continue
            xs[idx] = (causal_x + self.spurious_offset + 3 * idx) % self.modulus
        return tuple(xs)

    def break_spurious(self, causal_x: int, rng: random.Random, dims: int) -> tuple[int, ...]:
        xs = [rng.randrange(self.modulus) for _ in range(dims)]
        xs[self.causal_idx] = causal_x
        for idx in range(dims):
            if idx == self.causal_idx:
                continue
            confounded_value = (causal_x + self.spurious_offset + 3 * idx) % self.modulus
            while xs[idx] == confounded_value:
                xs[idx] = rng.randrange(self.modulus)
        return tuple(xs)

    def alter_distractors(self, xs: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
        out = list(xs)
        for idx in range(len(out)):
            if idx == self.causal_idx:
                continue
            old = out[idx]
            while out[idx] == old:
                out[idx] = rng.randrange(self.modulus)
        return tuple(out)

    def alter_one_distractor(self, xs: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
        out = list(xs)
        old = out[self.spurious_idx]
        while out[self.spurious_idx] == old:
            out[self.spurious_idx] = rng.randrange(self.modulus)
        return tuple(out)


@dataclass(frozen=True)
class Example:
    concept_id: int
    xs: tuple[int, ...]
    y: int
    cost: float


@dataclass(frozen=True)
class InterventionPair:
    before: Example
    after: Example
    causal_idx: int | None
    delta_x: int
    delta_y: int
    distractors_stable: bool
    certificate: bool
    cost: float


@dataclass
class ArmResult:
    arm: str
    spent_budget: float
    material_items: int
    causal_index_accuracy: float
    heldout_counterfactual_accuracy: float
    distractor_invariance_accuracy: float
    base_ce_guard_accuracy: float
    causal_score: float
    correct_per_paid: float


@dataclass
class GateResult:
    schema: str
    decision: str
    pass_gate: bool
    dims: int
    concepts: int
    modulus: int
    candidate: ArmResult
    controls: list[ArmResult]
    best_control: str
    score_gain: float
    counterfactual_gain: float
    invariance_gain: float
    base_regression: float
    correct_per_paid_ratio: float
    activation_log: str
    interpretation: str
    elapsed_seconds: float


class Learner:
    def __init__(self, concepts: list[Concept], dims: int) -> None:
        self.concepts = {c.concept_id: c for c in concepts}
        self.dims = dims
        self.examples: dict[int, list[Example]] = {c.concept_id: [] for c in concepts}
        self.certified_rules: dict[int, tuple[int, int, int]] = {}
        self.inferred_rules: dict[int, tuple[int, int, int]] = {}
        self.memory: dict[tuple[int, tuple[int, ...]], int] = {}

    def observe_example(self, ex: Example) -> None:
        self.examples[ex.concept_id].append(ex)
        self.memory[(ex.concept_id, ex.xs)] = ex.y

    def observe_pair(self, pair: InterventionPair) -> None:
        self.observe_example(pair.before)
        self.observe_example(pair.after)
        if not pair.certificate or pair.causal_idx is None:
            return
        concept = self.concepts[pair.before.concept_id]
        inv = mod_inverse(pair.delta_x, concept.modulus)
        if inv is None:
            return
        a = (pair.delta_y * inv) % concept.modulus
        b = (pair.before.y - a * pair.before.xs[pair.causal_idx]) % concept.modulus
        self.certified_rules[pair.before.concept_id] = (pair.causal_idx, a, b)

    def finalize(self) -> None:
        for concept_id, rows in self.examples.items():
            if concept_id in self.certified_rules:
                continue
            rule = infer_feature_rule(rows, self.dims, self.concepts[concept_id].modulus)
            if rule is not None:
                self.inferred_rules[concept_id] = rule

    def rule_for(self, concept_id: int) -> tuple[int, int, int] | None:
        if concept_id in self.certified_rules:
            return self.certified_rules[concept_id]
        return self.inferred_rules.get(concept_id)

    def predict(self, concept_id: int, xs: tuple[int, ...]) -> int | None:
        if (concept_id, xs) in self.memory:
            return self.memory[(concept_id, xs)]
        rule = self.rule_for(concept_id)
        if rule is None:
            return None
        idx, a, b = rule
        return (a * xs[idx] + b) % self.concepts[concept_id].modulus


def mod_inverse(x: int, modulus: int) -> int | None:
    try:
        return pow(x % modulus, -1, modulus)
    except ValueError:
        return None


def infer_feature_rule(rows: list[Example], dims: int, modulus: int) -> tuple[int, int, int] | None:
    if len(rows) < 2:
        return None
    ranked: list[tuple[int, int, int, int]] = []
    for idx in range(dims):
        scores: Counter[tuple[int, int]] = Counter()
        limit = min(len(rows), 28)
        for i in range(limit):
            for j in range(i + 1, limit):
                dx = (rows[j].xs[idx] - rows[i].xs[idx]) % modulus
                inv = mod_inverse(dx, modulus)
                if inv is None:
                    continue
                a = ((rows[j].y - rows[i].y) * inv) % modulus
                b = (rows[i].y - a * rows[i].xs[idx]) % modulus
                score = sum(1 for row in rows if (a * row.xs[idx] + b) % modulus == row.y)
                scores[(a, b)] = max(scores[(a, b)], score)
        if not scores:
            continue
        (a, b), score = scores.most_common(1)[0]
        ranked.append((score, -idx, a, b))
    if not ranked:
        return None
    score, neg_idx, a, b = max(ranked)
    if score < max(3, int(0.75 * len(rows))):
        return None
    return -neg_idx, a, b


def make_concepts(count: int, dims: int, modulus: int, seed: int) -> list[Concept]:
    rng = random.Random(seed)
    concepts: list[Concept] = []
    for concept_id in range(count):
        causal_idx = 1 + (concept_id % (dims - 1))
        spurious_idx = 0
        offset = rng.randrange(1, modulus)
        a = rng.randrange(1, modulus)
        b = rng.randrange(modulus)
        concepts.append(
            Concept(
                concept_id=concept_id,
                modulus=modulus,
                causal_idx=causal_idx,
                spurious_idx=spurious_idx,
                spurious_offset=offset,
                a=a,
                b=b,
            )
        )
    return concepts


def example_for(concept: Concept, xs: tuple[int, ...], cost: float) -> Example:
    return Example(concept_id=concept.concept_id, xs=xs, y=concept.y(xs), cost=cost)


def certified_pair(concept: Concept, rng: random.Random, dims: int) -> InterventionPair:
    x0 = rng.randrange(concept.modulus)
    x1 = rng.randrange(concept.modulus)
    while x1 == x0:
        x1 = rng.randrange(concept.modulus)
    before_xs = concept.make_confounded(x0, rng, dims)
    after = list(before_xs)
    after[concept.causal_idx] = x1
    after_xs = tuple(after)
    before = example_for(concept, before_xs, cost=1.0)
    after_ex = example_for(concept, after_xs, cost=1.0)
    return InterventionPair(
        before=before,
        after=after_ex,
        causal_idx=concept.causal_idx,
        delta_x=(x1 - x0) % concept.modulus,
        delta_y=(after_ex.y - before.y) % concept.modulus,
        distractors_stable=True,
        certificate=True,
        cost=2.7,
    )


def nonminimal_pair(concept: Concept, rng: random.Random, dims: int, cost: float) -> InterventionPair:
    x0 = rng.randrange(concept.modulus)
    x1 = rng.randrange(concept.modulus)
    while x1 == x0:
        x1 = rng.randrange(concept.modulus)
    before_xs = concept.make_confounded(x0, rng, dims)
    after_xs = concept.make_confounded(x1, rng, dims)
    before = example_for(concept, before_xs, cost=1.0)
    after = example_for(concept, after_xs, cost=1.0)
    changed = [i for i, (a, b) in enumerate(zip(before_xs, after_xs)) if a != b]
    return InterventionPair(
        before=before,
        after=after,
        causal_idx=None,
        delta_x=(x1 - x0) % concept.modulus,
        delta_y=(after.y - before.y) % concept.modulus,
        distractors_stable=changed == [concept.causal_idx],
        certificate=False,
        cost=cost,
    )


def build_material(arm: str, concepts: list[Concept], dims: int, budget: float, seed: int) -> list[Example | InterventionPair]:
    rng = random.Random(seed)
    items: list[Example | InterventionPair] = []
    spent = 0.0
    cursor = 0
    while spent < budget:
        concept = concepts[cursor % len(concepts)]
        cursor += 1
        if arm == "confounded_examples":
            x = rng.randrange(concept.modulus)
            item: Example | InterventionPair = example_for(
                concept, concept.make_confounded(x, rng, dims), cost=1.0
            )
            item_cost = item.cost
        elif arm == "frontier_confounded_examples":
            x = rng.randrange(concept.modulus)
            item = example_for(concept, concept.make_confounded(x, rng, dims), cost=1.15)
            item_cost = item.cost
        elif arm == "ordinary_counterfactual_pairs":
            item = nonminimal_pair(concept, rng, dims, cost=2.2)
            item_cost = item.cost
        elif arm == "distractor_flip_pairs":
            base_x = rng.randrange(concept.modulus)
            before_xs = concept.make_confounded(base_x, rng, dims)
            after_xs = concept.alter_one_distractor(before_xs, rng)
            before = example_for(concept, before_xs, cost=1.0)
            after = example_for(concept, after_xs, cost=1.0)
            item = InterventionPair(
                before=before,
                after=after,
                causal_idx=None,
                delta_x=0,
                delta_y=0,
                distractors_stable=False,
                certificate=False,
                cost=2.2,
            )
            item_cost = item.cost
        elif arm == "minimal_certified_interventions":
            item = certified_pair(concept, rng, dims)
            item_cost = item.cost
        else:
            raise ValueError(f"unknown arm: {arm}")
        if spent + item_cost > budget:
            break
        items.append(item)
        spent += item_cost
    return items


def observe_material(learner: Learner, material: list[Example | InterventionPair]) -> float:
    spent = 0.0
    for item in material:
        if isinstance(item, Example):
            learner.observe_example(item)
            spent += item.cost
        else:
            learner.observe_pair(item)
            spent += item.cost
    learner.finalize()
    return spent


def evaluate_arm(
    arm: str,
    concepts: list[Concept],
    dims: int,
    budget: float,
    seed: int,
    eval_per_concept: int,
) -> ArmResult:
    material = build_material(arm, concepts, dims, budget, seed)
    learner = Learner(concepts, dims)
    spent = observe_material(learner, material)

    causal_seen = 0
    causal_ok = 0
    cf_total = 0
    cf_ok = 0
    inv_total = 0
    inv_ok = 0
    base_total = 0
    base_ok = 0
    rng = random.Random(seed + 50_000)
    for concept in concepts:
        rule = learner.rule_for(concept.concept_id)
        if rule is not None:
            causal_seen += 1
            causal_ok += int(rule[0] == concept.causal_idx)
        for _ in range(eval_per_concept):
            x = rng.randrange(concept.modulus)
            base = concept.break_spurious(x, rng, dims)
            pred = learner.predict(concept.concept_id, base)
            target = concept.y(base)
            cf_total += 1
            cf_ok += int(pred == target)

            shifted = concept.alter_distractors(base, rng)
            pred_shifted = learner.predict(concept.concept_id, shifted)
            inv_total += 1
            inv_ok += int(pred_shifted == target)

            confounded = concept.make_confounded(x, rng, dims)
            pred_base = learner.predict(concept.concept_id, confounded)
            base_total += 1
            base_ok += int(pred_base == concept.y(confounded))

    causal_index_accuracy = causal_ok / max(1, causal_seen)
    heldout_counterfactual_accuracy = cf_ok / max(1, cf_total)
    distractor_invariance_accuracy = inv_ok / max(1, inv_total)
    base_ce_guard_accuracy = base_ok / max(1, base_total)
    causal_score = (
        0.45 * heldout_counterfactual_accuracy
        + 0.35 * distractor_invariance_accuracy
        + 0.15 * base_ce_guard_accuracy
        + 0.05 * causal_index_accuracy
    )
    return ArmResult(
        arm=arm,
        spent_budget=spent,
        material_items=len(material),
        causal_index_accuracy=causal_index_accuracy,
        heldout_counterfactual_accuracy=heldout_counterfactual_accuracy,
        distractor_invariance_accuracy=distractor_invariance_accuracy,
        base_ce_guard_accuracy=base_ce_guard_accuracy,
        causal_score=causal_score,
        correct_per_paid=causal_score / max(spent, 1e-9),
    )


def run_gate(args: argparse.Namespace) -> GateResult:
    start = time.perf_counter()
    concepts = make_concepts(args.concepts, args.dims, args.modulus, args.seed)
    arms = [
        "confounded_examples",
        "frontier_confounded_examples",
        "ordinary_counterfactual_pairs",
        "distractor_flip_pairs",
        "minimal_certified_interventions",
    ]
    results = [
        evaluate_arm(arm, concepts, args.dims, args.budget, args.seed + idx * 1000, args.eval_per_concept)
        for idx, arm in enumerate(arms)
    ]
    by_arm = {result.arm: result for result in results}
    candidate = by_arm["minimal_certified_interventions"]
    controls = [result for result in results if result.arm != candidate.arm]
    best_control = max(controls, key=lambda row: row.causal_score)
    score_gain = candidate.causal_score - best_control.causal_score
    counterfactual_gain = candidate.heldout_counterfactual_accuracy - best_control.heldout_counterfactual_accuracy
    invariance_gain = candidate.distractor_invariance_accuracy - best_control.distractor_invariance_accuracy
    base_regression = best_control.base_ce_guard_accuracy - candidate.base_ce_guard_accuracy
    correct_per_paid_ratio = candidate.correct_per_paid / max(best_control.correct_per_paid, 1e-12)
    pass_gate = (
        candidate.causal_index_accuracy >= 0.99
        and score_gain >= args.min_score_gain
        and counterfactual_gain >= args.min_counterfactual_gain
        and invariance_gain >= args.min_invariance_gain
        and base_regression <= args.max_base_regression
        and correct_per_paid_ratio >= args.min_paid_ratio
    )
    if pass_gate:
        decision = "PASS_MINIMAL_CAUSAL_INTERVENTION_GATE"
    elif candidate.causal_index_accuracy < 0.99:
        decision = "FAIL_CAUSAL_CERTIFICATE_RECOVERY"
    elif score_gain < args.min_score_gain:
        decision = "FAIL_CAUSAL_SCORE_GAIN"
    elif counterfactual_gain < args.min_counterfactual_gain:
        decision = "FAIL_COUNTERFACTUAL_GENERALIZATION_GAIN"
    elif invariance_gain < args.min_invariance_gain:
        decision = "FAIL_DISTRACTOR_INVARIANCE_GAIN"
    elif base_regression > args.max_base_regression:
        decision = "FAIL_BASE_GUARD_REGRESSION"
    else:
        decision = "FAIL_PAID_CAUSAL_RATIO"
    activation = (
        "[R422_MINIMAL_CAUSAL_INTERVENTION_GATE] "
        f"active=1 decision={decision} pass={int(pass_gate)} "
        f"best_control={best_control.arm} score_gain={score_gain:.6f} "
        f"counterfactual_gain={counterfactual_gain:.6f} "
        f"invariance_gain={invariance_gain:.6f} "
        f"base_regression={base_regression:.6f} "
        f"correct_per_paid_ratio={correct_per_paid_ratio:.6f} "
        "no_training_integration=1 no_gpu=1"
    )
    return GateResult(
        schema=SCHEMA,
        decision=decision,
        pass_gate=pass_gate,
        dims=args.dims,
        concepts=args.concepts,
        modulus=args.modulus,
        candidate=candidate,
        controls=controls,
        best_control=best_control.arm,
        score_gain=score_gain,
        counterfactual_gain=counterfactual_gain,
        invariance_gain=invariance_gain,
        base_regression=base_regression,
        correct_per_paid_ratio=correct_per_paid_ratio,
        activation_log=activation,
        interpretation=(
            "Minimal certified intervention pairs identify the causal coordinate "
            "under a confounded training stream; this validates material/compiler "
            "structure only, not GPU training speed."
        ),
        elapsed_seconds=time.perf_counter() - start,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, default=6)
    parser.add_argument("--concepts", type=int, default=48)
    parser.add_argument("--modulus", type=int, default=101)
    parser.add_argument("--budget", type=float, default=520.0)
    parser.add_argument("--eval-per-concept", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42286)
    parser.add_argument("--min-score-gain", type=float, default=0.20)
    parser.add_argument("--min-counterfactual-gain", type=float, default=0.20)
    parser.add_argument("--min-invariance-gain", type=float, default=0.20)
    parser.add_argument("--max-base-regression", type=float, default=0.02)
    parser.add_argument("--min-paid-ratio", type=float, default=1.10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.dims < 3:
        raise ValueError("--dims must be at least 3")
    if args.concepts < 4:
        raise ValueError("--concepts must be at least 4")
    if args.modulus < 17:
        raise ValueError("--modulus must be at least 17")

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
