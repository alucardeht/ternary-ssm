#!/usr/bin/env python3
"""Capability closure and minimal causal intervention gate.

This is the bounded implementation gate for the R422 cluster
"Capability Closure And Minimal Causal Intervention Material".

The probe uses a small deterministic verifier workbench: each concept has a
normal affine rule plus a certified exception predicate and exception rule. It
compares equal paid-budget training material streams:

- random labeled examples;
- frontier-selected labeled examples;
- ordinary counterfactual pairs;
- closure material without the exception ledger;
- closure material with minimal causal intervention and exception ledger.

This is not production training and not a broad data claim. It asks whether a
capability-acquisition certificate can beat ordinary curriculum/counterfactual
material on a verifiable held-out skill family without an external teacher.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_capability_closure_gate.v1"


@dataclass(frozen=True)
class Concept:
    concept_id: int
    modulus: int
    a: int
    b: int
    exception_mod: int
    exception_residue: int
    exception_a: int
    exception_b: int

    def is_exception(self, x: int) -> bool:
        return x % self.exception_mod == self.exception_residue

    def y(self, x: int) -> int:
        if self.is_exception(x):
            return (self.exception_a * x + self.exception_b) % self.modulus
        return (self.a * x + self.b) % self.modulus


@dataclass
class Example:
    concept_id: int
    x: int
    y: int
    is_exception: bool
    has_normal_certificate: bool = False
    has_exception_ledger: bool = False
    exception_mod: int | None = None
    exception_residue: int | None = None
    exception_a: int | None = None
    exception_b: int | None = None
    cost: float = 1.0


@dataclass
class ArmResult:
    arm: str
    spent_budget: float
    examples_seen: int
    concepts_touched: int
    total_accuracy: float
    normal_accuracy: float
    exception_accuracy: float
    exception_share: float
    capability_score: float
    capability_per_budget: float


class Learner:
    def __init__(self, concepts: list[Concept]) -> None:
        self.concepts = {c.concept_id: c for c in concepts}
        self.examples: dict[int, list[Example]] = {c.concept_id: [] for c in concepts}
        self.normal_rules: dict[int, tuple[int, int]] = {}
        self.exception_rules: dict[int, tuple[int, int, int, int]] = {}
        self.memory: dict[tuple[int, int], int] = {}

    def observe(self, ex: Example) -> None:
        self.examples[ex.concept_id].append(ex)
        self.memory[(ex.concept_id, ex.x)] = ex.y
        if ex.has_normal_certificate and not ex.is_exception:
            rows = [e for e in self.examples[ex.concept_id] if e.has_normal_certificate and not e.is_exception]
            rule = fit_affine(rows, self.concepts[ex.concept_id].modulus)
            if rule is not None:
                self.normal_rules[ex.concept_id] = rule
        if ex.has_exception_ledger:
            assert ex.exception_mod is not None
            assert ex.exception_residue is not None
            assert ex.exception_a is not None
            assert ex.exception_b is not None
            self.exception_rules[ex.concept_id] = (
                ex.exception_mod,
                ex.exception_residue,
                ex.exception_a,
                ex.exception_b,
            )
        elif ex.concept_id not in self.normal_rules:
            rule = infer_majority_affine(self.examples[ex.concept_id], self.concepts[ex.concept_id].modulus)
            if rule is not None:
                self.normal_rules[ex.concept_id] = rule

    def predict(self, concept_id: int, x: int) -> int | None:
        if concept_id in self.exception_rules:
            mod, residue, a_exc, b_exc = self.exception_rules[concept_id]
            concept = self.concepts[concept_id]
            if x % mod == residue:
                return (a_exc * x + b_exc) % concept.modulus
        if (concept_id, x) in self.memory:
            return self.memory[(concept_id, x)]
        if concept_id in self.normal_rules:
            a, b = self.normal_rules[concept_id]
            return (a * x + b) % self.concepts[concept_id].modulus
        return None


def mod_inverse(x: int, modulus: int) -> int | None:
    try:
        return pow(x % modulus, -1, modulus)
    except ValueError:
        return None


def fit_affine(rows: list[Example], modulus: int) -> tuple[int, int] | None:
    if len(rows) < 2:
        return None
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            dx = (rows[j].x - rows[i].x) % modulus
            inv = mod_inverse(dx, modulus)
            if inv is None:
                continue
            a = ((rows[j].y - rows[i].y) * inv) % modulus
            b = (rows[i].y - a * rows[i].x) % modulus
            return a, b
    return None


def infer_majority_affine(rows: list[Example], modulus: int) -> tuple[int, int] | None:
    if len(rows) < 4:
        return None
    candidates: dict[tuple[int, int], int] = {}
    limit = min(len(rows), 18)
    for i in range(limit):
        for j in range(i + 1, limit):
            dx = (rows[j].x - rows[i].x) % modulus
            inv = mod_inverse(dx, modulus)
            if inv is None:
                continue
            a = ((rows[j].y - rows[i].y) * inv) % modulus
            b = (rows[i].y - a * rows[i].x) % modulus
            score = sum(1 for row in rows if (a * row.x + b) % modulus == row.y)
            candidates[(a, b)] = max(candidates.get((a, b), 0), score)
    if not candidates:
        return None
    best, score = max(candidates.items(), key=lambda item: item[1])
    return best if score >= max(3, len(rows) // 2) else None


def make_concepts(count: int, modulus: int, rng: random.Random) -> list[Concept]:
    exception_mods = [5, 7, 11, 13]
    concepts: list[Concept] = []
    for cid in range(count):
        a = rng.randrange(1, modulus)
        b = rng.randrange(modulus)
        exc_mod = rng.choice(exception_mods)
        exc_residue = rng.randrange(exc_mod)
        exc_a = rng.randrange(1, modulus)
        exc_b = rng.randrange(modulus)
        concepts.append(
            Concept(
                concept_id=cid,
                modulus=modulus,
                a=a,
                b=b,
                exception_mod=exc_mod,
                exception_residue=exc_residue,
                exception_a=exc_a,
                exception_b=exc_b,
            )
        )
    return concepts


def example_for(concept: Concept, x: int, *, cert: bool, ledger: bool, cost: float) -> Example:
    return Example(
        concept_id=concept.concept_id,
        x=x,
        y=concept.y(x),
        is_exception=concept.is_exception(x),
        has_normal_certificate=cert and not concept.is_exception(x),
        has_exception_ledger=ledger and concept.is_exception(x),
        exception_mod=concept.exception_mod if ledger and concept.is_exception(x) else None,
        exception_residue=concept.exception_residue if ledger and concept.is_exception(x) else None,
        exception_a=concept.exception_a if ledger and concept.is_exception(x) else None,
        exception_b=concept.exception_b if ledger and concept.is_exception(x) else None,
        cost=cost,
    )


def normal_x(concept: Concept, rng: random.Random) -> int:
    while True:
        x = rng.randrange(concept.modulus)
        if not concept.is_exception(x):
            return x


def exception_x(concept: Concept, rng: random.Random) -> int:
    candidates = [x for x in range(concept.modulus) if concept.is_exception(x)]
    return rng.choice(candidates)


def build_stream(arm: str, concepts: list[Concept], budget: float, rng: random.Random) -> list[Example]:
    stream: list[Example] = []
    spent = 0.0
    cursor = 0
    while spent < budget:
        concept = concepts[cursor % len(concepts)]
        cursor += 1
        if arm == "random_examples":
            x = rng.randrange(concept.modulus)
            ex = example_for(concept, x, cert=False, ledger=False, cost=1.0)
            batch = [ex]
        elif arm == "frontier_sampling":
            use_exception = rng.random() < 0.35
            x = exception_x(concept, rng) if use_exception else normal_x(concept, rng)
            batch = [example_for(concept, x, cert=False, ledger=False, cost=1.15)]
        elif arm == "ordinary_counterfactual_pairs":
            x0 = normal_x(concept, rng)
            x1 = (x0 + 1) % concept.modulus
            batch = [
                example_for(concept, x0, cert=False, ledger=False, cost=1.1),
                example_for(concept, x1, cert=False, ledger=False, cost=1.1),
            ]
        elif arm == "closure_without_exception_ledger":
            x0 = normal_x(concept, rng)
            x1 = normal_x(concept, rng)
            x2 = exception_x(concept, rng)
            batch = [
                example_for(concept, x0, cert=True, ledger=False, cost=1.3),
                example_for(concept, x1, cert=True, ledger=False, cost=1.3),
                example_for(concept, x2, cert=False, ledger=False, cost=1.1),
            ]
        elif arm == "closure_with_intervention_ledger":
            x0 = normal_x(concept, rng)
            x1 = normal_x(concept, rng)
            e0 = exception_x(concept, rng)
            e1 = exception_x(concept, rng)
            batch = [
                example_for(concept, x0, cert=True, ledger=False, cost=1.3),
                example_for(concept, x1, cert=True, ledger=False, cost=1.3),
                example_for(concept, e0, cert=False, ledger=True, cost=2.4),
                example_for(concept, e1, cert=False, ledger=True, cost=2.4),
            ]
        else:
            raise ValueError(f"unknown arm: {arm}")
        if spent + sum(ex.cost for ex in batch) > budget:
            break
        stream.extend(batch)
        spent += sum(ex.cost for ex in batch)
    return stream


def evaluate(arm: str, concepts: list[Concept], budget: float, seed: int, eval_per_concept: int) -> ArmResult:
    rng = random.Random(seed)
    stream = build_stream(arm, concepts, budget, rng)
    learner = Learner(concepts)
    spent = 0.0
    for ex in stream:
        learner.observe(ex)
        spent += ex.cost

    total = 0
    correct = 0
    normal_total = 0
    normal_correct = 0
    exception_total = 0
    exception_correct = 0
    eval_rng = random.Random(seed + 10_000)
    for concept in concepts:
        xs = eval_rng.sample(range(concept.modulus), min(eval_per_concept, concept.modulus))
        for x in xs:
            target = concept.y(x)
            pred = learner.predict(concept.concept_id, x)
            ok = pred == target
            total += 1
            correct += int(ok)
            if concept.is_exception(x):
                exception_total += 1
                exception_correct += int(ok)
            else:
                normal_total += 1
                normal_correct += int(ok)

    total_accuracy = correct / total
    normal_accuracy = normal_correct / max(1, normal_total)
    exception_accuracy = exception_correct / max(1, exception_total)
    exception_share = exception_total / total
    capability_score = 0.65 * total_accuracy + 0.35 * exception_accuracy
    return ArmResult(
        arm=arm,
        spent_budget=spent,
        examples_seen=len(stream),
        concepts_touched=len({ex.concept_id for ex in stream}),
        total_accuracy=total_accuracy,
        normal_accuracy=normal_accuracy,
        exception_accuracy=exception_accuracy,
        exception_share=exception_share,
        capability_score=capability_score,
        capability_per_budget=capability_score / max(spent, 1e-9),
    )


def run_trial(seed: int, args: argparse.Namespace) -> dict:
    concepts = make_concepts(args.concepts, args.modulus, random.Random(seed))
    arms = [
        "random_examples",
        "frontier_sampling",
        "ordinary_counterfactual_pairs",
        "closure_without_exception_ledger",
        "closure_with_intervention_ledger",
    ]
    results = [
        evaluate(arm, concepts, args.budget, seed + idx * 1000, args.eval_per_concept)
        for idx, arm in enumerate(arms)
    ]
    return {result.arm: asdict(result) for result in results}


def aggregate(trials: list[dict]) -> dict:
    arms = sorted(trials[0].keys())
    out: dict[str, dict[str, float]] = {}
    for arm in arms:
        rows = [trial[arm] for trial in trials]
        metrics = rows[0].keys()
        out[arm] = {
            metric: sum(float(row[metric]) for row in rows) / len(rows)
            for metric in metrics
            if metric != "arm"
        }
    closure = out["closure_with_intervention_ledger"]
    controls = {arm: out[arm] for arm in arms if arm != "closure_with_intervention_ledger"}
    best_control_name, best_control = max(
        controls.items(), key=lambda item: item[1]["capability_score"]
    )
    score_gain = closure["capability_score"] - best_control["capability_score"]
    total_gain = closure["total_accuracy"] - best_control["total_accuracy"]
    exception_gain = closure["exception_accuracy"] - best_control["exception_accuracy"]
    normal_regression = best_control["normal_accuracy"] - closure["normal_accuracy"]
    capability_per_budget_ratio = (
        closure["capability_per_budget"] / max(best_control["capability_per_budget"], 1e-12)
    )
    pass_gate = (
        score_gain >= 0.10
        and total_gain >= 0.08
        and exception_gain >= 0.20
        and normal_regression <= 0.02
        and capability_per_budget_ratio >= 1.10
    )
    if pass_gate:
        decision = "PASS_CAPABILITY_CLOSURE_GATE"
    elif exception_gain < 0.20:
        decision = "FAIL_EXCEPTION_TRANSFER_GAIN"
    elif score_gain < 0.10 or total_gain < 0.08:
        decision = "FAIL_CAPABILITY_GAIN"
    elif normal_regression > 0.02:
        decision = "FAIL_BASE_CE_NORMAL_REGRESSION"
    else:
        decision = "FAIL_PAID_CAPABILITY_RATIO"
    return {
        "decision": decision,
        "pass_gate": pass_gate,
        "best_control": best_control_name,
        "score_gain": score_gain,
        "total_gain": total_gain,
        "exception_gain": exception_gain,
        "normal_regression": normal_regression,
        "capability_per_budget_ratio": capability_per_budget_ratio,
        "arms": out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concepts", type=int, default=48)
    parser.add_argument("--modulus", type=int, default=97)
    parser.add_argument("--budget", type=float, default=1200.0)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--eval-per-concept", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42273)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.modulus < 17:
        raise ValueError("--modulus must be at least 17")
    if args.concepts < 4:
        raise ValueError("--concepts must be at least 4")

    trials = [run_trial(args.seed + idx, args) for idx in range(args.trials)]
    result = aggregate(trials)
    params = vars(args).copy()
    params["output"] = str(args.output) if args.output else None
    payload = {
        "schema": SCHEMA,
        "active": 1,
        "mode": "cpu_verifiable_capability_closure_gate",
        "no_training_integration": 1,
        "no_gpu": 1,
        "params": params,
        "result": result,
        "trials": trials,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(
        "[R422_CAPABILITY_CLOSURE_GATE] "
        f"active=1 decision={result['decision']} pass={int(result['pass_gate'])} "
        f"best_control={result['best_control']} "
        f"score_gain={result['score_gain']:.6f} "
        f"total_gain={result['total_gain']:.6f} "
        f"exception_gain={result['exception_gain']:.6f} "
        f"normal_regression={result['normal_regression']:.6f} "
        f"capability_per_budget_ratio={result['capability_per_budget_ratio']:.6f} "
        "no_training_integration=1 no_gpu=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
