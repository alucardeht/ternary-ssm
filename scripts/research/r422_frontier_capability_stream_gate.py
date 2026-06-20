#!/usr/bin/env python3
"""R422 frontier-only capability stream gate.

This is a verifier-backed training-material gate for R422-CI-01. It tests
whether a live frontier-band stream improves capability per paid example versus
uniform, easy, and hard streams while preserving a residual capability guard.

The gate is not a GPU benchmark and does not change production training.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from r422_capability_closure_gate import (
    Concept,
    Example,
    Learner,
    exception_x,
    example_for,
    make_concepts,
    normal_x,
)


SCHEMA = "r422_frontier_capability_stream_gate.v1"


@dataclass
class Candidate:
    examples: list[Example]
    cost: float
    kind: str
    concept_id: int
    before_score: float
    after_score: float
    predicted_gain: float
    gain_per_cost: float
    band: str


@dataclass
class ArmResult:
    arm: str
    spent_budget: float
    examples_seen: int
    candidate_batches: int
    frontier_batches: int
    easy_batches: int
    hard_batches: int
    positive_gain_batches: int
    mean_predicted_gain: float
    mean_gain_per_cost: float
    total_accuracy: float
    normal_accuracy: float
    exception_accuracy: float
    capability_score: float
    capability_per_budget: float
    residual_guard_pass: bool


def clone_learner(learner: Learner) -> Learner:
    out = Learner(list(learner.concepts.values()))
    out.examples = {cid: list(rows) for cid, rows in learner.examples.items()}
    out.normal_rules = dict(learner.normal_rules)
    out.exception_rules = dict(learner.exception_rules)
    out.memory = dict(learner.memory)
    return out


def concept_score(learner: Learner, concept: Concept, xs: list[int]) -> tuple[float, float, float]:
    total = 0
    correct = 0
    normal_total = 0
    normal_correct = 0
    exception_total = 0
    exception_correct = 0
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
    total_acc = correct / max(1, total)
    normal_acc = normal_correct / max(1, normal_total)
    exception_acc = exception_correct / max(1, exception_total)
    capability = 0.65 * total_acc + 0.35 * exception_acc
    return total_acc, normal_acc, capability


def global_score(learner: Learner, concepts: list[Concept], probe_xs: dict[int, list[int]]) -> tuple[float, float, float, float]:
    total = 0
    correct = 0
    normal_total = 0
    normal_correct = 0
    exception_total = 0
    exception_correct = 0
    for concept in concepts:
        for x in probe_xs[concept.concept_id]:
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
    total_acc = correct / max(1, total)
    normal_acc = normal_correct / max(1, normal_total)
    exception_acc = exception_correct / max(1, exception_total)
    capability = 0.65 * total_acc + 0.35 * exception_acc
    return total_acc, normal_acc, exception_acc, capability


def make_candidate_examples(kind: str, concept: Concept, rng: random.Random) -> list[Example]:
    if kind == "normal_certificate_pair":
        return [
            example_for(concept, normal_x(concept, rng), cert=True, ledger=False, cost=1.3),
            example_for(concept, normal_x(concept, rng), cert=True, ledger=False, cost=1.3),
        ]
    if kind == "exception_ledger":
        return [
            example_for(concept, exception_x(concept, rng), cert=False, ledger=True, cost=2.4)
        ]
    if kind == "plain_example":
        return [
            example_for(concept, rng.randrange(concept.modulus), cert=False, ledger=False, cost=1.0)
        ]
    if kind == "counterfactual_pair":
        x0 = normal_x(concept, rng)
        x1 = (x0 + 1) % concept.modulus
        return [
            example_for(concept, x0, cert=False, ledger=False, cost=1.1),
            example_for(concept, x1, cert=False, ledger=False, cost=1.1),
        ]
    raise ValueError(f"unknown candidate kind: {kind}")


def classify_band(before_score: float, predicted_gain: float, low: float, high: float) -> str:
    if predicted_gain <= 0.0:
        return "stale_or_no_gain"
    if before_score >= high:
        return "easy_saturated"
    if before_score <= low:
        return "hard_unformed"
    return "frontier"


def build_candidates(
    learner: Learner,
    concepts: list[Concept],
    probe_xs: dict[int, list[int]],
    rng: random.Random,
    count: int,
    low: float,
    high: float,
) -> list[Candidate]:
    kinds = [
        "normal_certificate_pair",
        "exception_ledger",
        "plain_example",
        "counterfactual_pair",
    ]
    candidates: list[Candidate] = []
    for _ in range(count):
        concept = rng.choice(concepts)
        kind = rng.choice(kinds)
        examples = make_candidate_examples(kind, concept, rng)
        before = concept_score(learner, concept, probe_xs[concept.concept_id])[2]
        after_learner = clone_learner(learner)
        for ex in examples:
            after_learner.observe(ex)
        after = concept_score(after_learner, concept, probe_xs[concept.concept_id])[2]
        cost = sum(ex.cost for ex in examples)
        gain = after - before
        band = classify_band(before, gain, low, high)
        candidates.append(
            Candidate(
                examples=examples,
                cost=cost,
                kind=kind,
                concept_id=concept.concept_id,
                before_score=before,
                after_score=after,
                predicted_gain=gain,
                gain_per_cost=gain / max(cost, 1e-12),
                band=band,
            )
        )
    return candidates


def choose_candidate(arm: str, candidates: list[Candidate], rng: random.Random) -> Candidate:
    if arm == "uniform_stream":
        return rng.choice(candidates)
    if arm == "easy_stream":
        easy = [c for c in candidates if c.band == "easy_saturated"]
        if easy:
            return max(easy, key=lambda c: (c.before_score, c.gain_per_cost))
        return max(candidates, key=lambda c: (c.before_score, -abs(c.predicted_gain)))
    if arm == "hard_stream":
        hard = [c for c in candidates if c.band == "hard_unformed"]
        if hard:
            return max(hard, key=lambda c: (c.predicted_gain, c.gain_per_cost))
        return min(candidates, key=lambda c: (c.before_score, -c.predicted_gain))
    if arm == "frontier_only_stream":
        frontier = [c for c in candidates if c.band == "frontier"]
        if frontier:
            return max(frontier, key=lambda c: (c.gain_per_cost, c.predicted_gain))
        positive = [c for c in candidates if c.predicted_gain > 0.0]
        if positive:
            return max(positive, key=lambda c: (c.gain_per_cost, c.predicted_gain))
        return max(candidates, key=lambda c: c.gain_per_cost)
    raise ValueError(f"unknown arm: {arm}")


def evaluate_arm(
    arm: str,
    concepts: list[Concept],
    budget: float,
    seed: int,
    candidate_count: int,
    frontier_low: float,
    frontier_high: float,
    probe_xs: dict[int, list[int]],
    eval_xs: dict[int, list[int]],
    uniform_normal_accuracy: float | None = None,
) -> ArmResult:
    rng = random.Random(seed)
    learner = Learner(concepts)
    spent = 0.0
    chosen: list[Candidate] = []
    examples_seen = 0
    while spent < budget:
        candidates = build_candidates(
            learner,
            concepts,
            probe_xs,
            rng,
            candidate_count,
            frontier_low,
            frontier_high,
        )
        cand = choose_candidate(arm, candidates, rng)
        if spent + cand.cost > budget:
            break
        for ex in cand.examples:
            learner.observe(ex)
        spent += cand.cost
        examples_seen += len(cand.examples)
        chosen.append(cand)

    total_acc, normal_acc, exception_acc, capability = global_score(learner, concepts, eval_xs)
    frontier_batches = sum(1 for c in chosen if c.band == "frontier")
    easy_batches = sum(1 for c in chosen if c.band == "easy_saturated")
    hard_batches = sum(1 for c in chosen if c.band == "hard_unformed")
    positive_batches = sum(1 for c in chosen if c.predicted_gain > 0.0)
    mean_gain = sum(c.predicted_gain for c in chosen) / max(1, len(chosen))
    mean_gain_per_cost = sum(c.gain_per_cost for c in chosen) / max(1, len(chosen))
    guard_pass = True
    if uniform_normal_accuracy is not None:
        guard_pass = normal_acc + 0.02 >= uniform_normal_accuracy
    return ArmResult(
        arm=arm,
        spent_budget=spent,
        examples_seen=examples_seen,
        candidate_batches=len(chosen),
        frontier_batches=frontier_batches,
        easy_batches=easy_batches,
        hard_batches=hard_batches,
        positive_gain_batches=positive_batches,
        mean_predicted_gain=mean_gain,
        mean_gain_per_cost=mean_gain_per_cost,
        total_accuracy=total_acc,
        normal_accuracy=normal_acc,
        exception_accuracy=exception_acc,
        capability_score=capability,
        capability_per_budget=capability / max(spent, 1e-12),
        residual_guard_pass=guard_pass,
    )


def make_probe_sets(concepts: list[Concept], seed: int, probe_per_concept: int, eval_per_concept: int) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    rng = random.Random(seed)
    probe_xs: dict[int, list[int]] = {}
    eval_xs: dict[int, list[int]] = {}
    for concept in concepts:
        xs = list(range(concept.modulus))
        rng.shuffle(xs)
        probe_xs[concept.concept_id] = xs[: min(probe_per_concept, len(xs))]
        eval_xs[concept.concept_id] = xs[-min(eval_per_concept, len(xs)) :]
    return probe_xs, eval_xs


def run_trial(seed: int, args: argparse.Namespace) -> dict[str, dict]:
    concepts = make_concepts(args.concepts, args.modulus, random.Random(seed))
    probe_xs, eval_xs = make_probe_sets(concepts, seed + 50_000, args.probe_per_concept, args.eval_per_concept)
    uniform = evaluate_arm(
        "uniform_stream",
        concepts,
        args.budget,
        seed + 1000,
        args.candidate_count,
        args.frontier_low,
        args.frontier_high,
        probe_xs,
        eval_xs,
    )
    arms = [uniform]
    for idx, arm in enumerate(["easy_stream", "hard_stream", "frontier_only_stream"], start=2):
        arms.append(
            evaluate_arm(
                arm,
                concepts,
                args.budget,
                seed + idx * 1000,
                args.candidate_count,
                args.frontier_low,
                args.frontier_high,
                probe_xs,
                eval_xs,
                uniform_normal_accuracy=uniform.normal_accuracy,
            )
        )
    return {arm.arm: asdict(arm) for arm in arms}


def aggregate(trials: list[dict[str, dict]], args: argparse.Namespace) -> dict:
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
    frontier = out["frontier_only_stream"]
    controls = {arm: out[arm] for arm in arms if arm != "frontier_only_stream"}
    best_control_name, best_control = max(
        controls.items(), key=lambda item: item[1]["capability_score"]
    )
    uniform = out["uniform_stream"]
    score_gain = frontier["capability_score"] - best_control["capability_score"]
    uniform_score_gain = frontier["capability_score"] - uniform["capability_score"]
    total_gain = frontier["total_accuracy"] - best_control["total_accuracy"]
    normal_regression = uniform["normal_accuracy"] - frontier["normal_accuracy"]
    frontier_fraction = frontier["frontier_batches"] / max(1.0, frontier["candidate_batches"])
    capability_per_budget_ratio = frontier["capability_per_budget"] / max(
        best_control["capability_per_budget"], 1e-12
    )
    pass_gate = (
        score_gain >= args.min_score_gain
        and uniform_score_gain >= args.min_uniform_score_gain
        and total_gain >= args.min_total_gain
        and normal_regression <= args.max_normal_regression
        and capability_per_budget_ratio >= args.min_capability_per_budget_ratio
        and frontier_fraction >= args.min_frontier_fraction
    )
    if pass_gate:
        decision = "PASS_FRONTIER_CAPABILITY_STREAM_GATE"
    elif frontier_fraction < args.min_frontier_fraction:
        decision = "FAIL_LIVE_FRONTIER_BAND_DENSITY"
    elif normal_regression > args.max_normal_regression:
        decision = "FAIL_RESIDUAL_CAPABILITY_GUARD"
    elif score_gain < args.min_score_gain or uniform_score_gain < args.min_uniform_score_gain:
        decision = "FAIL_FRONTIER_CAPABILITY_GAIN"
    elif total_gain < args.min_total_gain:
        decision = "FAIL_TOTAL_CAPABILITY_GAIN"
    else:
        decision = "FAIL_PAID_CAPABILITY_RATIO"
    return {
        "decision": decision,
        "pass_gate": pass_gate,
        "best_control": best_control_name,
        "score_gain": score_gain,
        "uniform_score_gain": uniform_score_gain,
        "total_gain": total_gain,
        "normal_regression": normal_regression,
        "frontier_fraction": frontier_fraction,
        "capability_per_budget_ratio": capability_per_budget_ratio,
        "arms": out,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concepts", type=int, default=48)
    parser.add_argument("--modulus", type=int, default=97)
    parser.add_argument("--budget", type=float, default=480.0)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--candidate-count", type=int, default=24)
    parser.add_argument("--probe-per-concept", type=int, default=24)
    parser.add_argument("--eval-per-concept", type=int, default=48)
    parser.add_argument("--frontier-low", type=float, default=0.20)
    parser.add_argument("--frontier-high", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42291)
    parser.add_argument("--min-score-gain", type=float, default=0.10)
    parser.add_argument("--min-uniform-score-gain", type=float, default=0.08)
    parser.add_argument("--min-total-gain", type=float, default=0.08)
    parser.add_argument("--max-normal-regression", type=float, default=0.02)
    parser.add_argument("--min-capability-per-budget-ratio", type=float, default=1.10)
    parser.add_argument("--min-frontier-fraction", type=float, default=0.45)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    if args.concepts < 4:
        raise ValueError("--concepts must be at least 4")
    if args.modulus < 17:
        raise ValueError("--modulus must be at least 17")
    if not (0.0 <= args.frontier_low < args.frontier_high <= 1.0):
        raise ValueError("frontier band must satisfy 0 <= low < high <= 1")

    trials = [run_trial(args.seed + idx, args) for idx in range(args.trials)]
    result = aggregate(trials, args)
    params = vars(args).copy()
    params["output"] = str(args.output) if args.output else None
    payload = {
        "schema": SCHEMA,
        "active": 1,
        "mode": "cpu_verifiable_frontier_capability_stream_gate",
        "no_training_integration": 1,
        "no_gpu": 1,
        "params": params,
        "result": result,
        "trials": trials,
        "elapsed_seconds": time.perf_counter() - start,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "[R422_FRONTIER_CAPABILITY_STREAM_GATE] "
        f"active=1 decision={result['decision']} pass={int(result['pass_gate'])} "
        f"best_control={result['best_control']} "
        f"score_gain={result['score_gain']:.6f} "
        f"uniform_score_gain={result['uniform_score_gain']:.6f} "
        f"total_gain={result['total_gain']:.6f} "
        f"normal_regression={result['normal_regression']:.6f} "
        f"frontier_fraction={result['frontier_fraction']:.6f} "
        f"capability_per_budget_ratio={result['capability_per_budget_ratio']:.6f} "
        "no_training_integration=1 no_gpu=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
