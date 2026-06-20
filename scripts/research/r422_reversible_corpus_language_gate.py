#!/usr/bin/env python3
"""R422 reversible corpus language gate.

This gate tests whether controlled corpus records can become reversible objects:
surface text, canonical fact graph, reconstruction plan, and residual map. It
is a material/compiler gate, not a production training path.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_reversible_corpus_language_gate.v1"


TEMPLATES = [
    "{s} carefully records that {o} is the {r}.",
    "In the ledger, {s} marks {o} as {r}.",
    "The document states: {s} has {r} {o}.",
    "For audit, {s} links {r} to {o}.",
]


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    object: str


@dataclass
class RenderedFact:
    fact: Fact
    template_id: int


@dataclass
class ReversibleObject:
    canonical_graph: list[Fact]
    render_plan: list[tuple[int, int]]
    residual_lines: list[str]


@dataclass
class CaseResult:
    facts: int
    surface_tokens: int
    object_tokens: int
    object_token_ratio: float
    reconstruction_exact: bool
    graph_answer_correct: bool
    shuffled_graph_correct: bool
    residual_ratio: float
    query_work_ratio: float


@dataclass
class GateResult:
    schema: str
    decision: str
    pass_gate: bool
    cases: int
    exact_reconstruction_fraction: float
    graph_answer_accuracy: float
    shuffled_graph_accuracy: float
    mean_object_token_ratio: float
    mean_query_work_ratio: float
    max_residual_ratio: float
    activation_log: str
    interpretation: str
    case_results: list[CaseResult]
    elapsed_seconds: float


def render_fact(rendered: RenderedFact) -> str:
    fact = rendered.fact
    return TEMPLATES[rendered.template_id].format(s=fact.subject, r=fact.relation, o=fact.object)


def tokenize(text: str) -> list[str]:
    return text.replace(".", " .").replace(":", " :").split()


def make_case(facts: int, rng: random.Random) -> list[RenderedFact]:
    subjects = [f"entity_{i}" for i in range(facts)]
    relations = ["owner", "color", "room", "tool", "status", "target"]
    objects = [f"value_{rng.randrange(1000, 9999)}" for _ in range(facts)]
    rendered: list[RenderedFact] = []
    for idx in range(facts):
        fact = Fact(
            subject=subjects[idx],
            relation=relations[(idx * 3 + rng.randrange(len(relations))) % len(relations)],
            object=objects[idx],
        )
        rendered.append(RenderedFact(fact=fact, template_id=rng.randrange(len(TEMPLATES))))
    rng.shuffle(rendered)
    return rendered


def surface_text(rendered: list[RenderedFact]) -> str:
    return "\n".join(render_fact(item) for item in rendered) + "\n"


def encode(rendered: list[RenderedFact]) -> ReversibleObject:
    canonical = sorted((item.fact for item in rendered), key=lambda f: (f.subject, f.relation, f.object))
    index = {fact: idx for idx, fact in enumerate(canonical)}
    plan = [(index[item.fact], item.template_id) for item in rendered]
    return ReversibleObject(canonical_graph=canonical, render_plan=plan, residual_lines=[])


def decode(obj: ReversibleObject) -> str:
    rows: list[str] = []
    for fact_idx, template_id in obj.render_plan:
        rows.append(render_fact(RenderedFact(fact=obj.canonical_graph[fact_idx], template_id=template_id)))
    rows.extend(obj.residual_lines)
    return "\n".join(rows) + "\n"


def object_token_count(obj: ReversibleObject) -> int:
    graph_tokens = 3 * len(obj.canonical_graph)
    plan_tokens = 2 * len(obj.render_plan)
    residual_tokens = sum(len(tokenize(line)) for line in obj.residual_lines)
    header_tokens = 3
    return graph_tokens + plan_tokens + residual_tokens + header_tokens


def query_answer(obj: ReversibleObject, fact: Fact) -> str | None:
    for row in obj.canonical_graph:
        if row.subject == fact.subject and row.relation == fact.relation:
            return row.object
    return None


def shuffled_control(obj: ReversibleObject, fact: Fact, rng: random.Random) -> str | None:
    objects = [row.object for row in obj.canonical_graph]
    rng.shuffle(objects)
    shuffled = [
        Fact(subject=row.subject, relation=row.relation, object=objects[idx])
        for idx, row in enumerate(obj.canonical_graph)
    ]
    return query_answer(ReversibleObject(shuffled, obj.render_plan, []), fact)


def evaluate_case(rendered: list[RenderedFact], rng: random.Random) -> CaseResult:
    text = surface_text(rendered)
    obj = encode(rendered)
    decoded = decode(obj)
    surface_tokens = len(tokenize(text))
    obj_tokens = object_token_count(obj)
    query_item = rng.choice(rendered).fact
    answer = query_answer(obj, query_item)
    shuffled_answer = shuffled_control(obj, query_item, rng)
    graph_work = len(obj.canonical_graph)
    plain_work = surface_tokens
    return CaseResult(
        facts=len(rendered),
        surface_tokens=surface_tokens,
        object_tokens=obj_tokens,
        object_token_ratio=obj_tokens / max(1, surface_tokens),
        reconstruction_exact=decoded == text,
        graph_answer_correct=answer == query_item.object,
        shuffled_graph_correct=shuffled_answer == query_item.object,
        residual_ratio=sum(len(tokenize(line)) for line in obj.residual_lines) / max(1, surface_tokens),
        query_work_ratio=graph_work / max(1, plain_work),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=80)
    parser.add_argument("--min-facts", type=int, default=12)
    parser.add_argument("--max-facts", type=int, default=72)
    parser.add_argument("--seed", type=int, default=42201)
    parser.add_argument("--max-object-token-ratio", type=float, default=0.75)
    parser.add_argument("--max-query-work-ratio", type=float, default=0.30)
    parser.add_argument("--max-residual-ratio", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    rng = random.Random(args.seed)
    results = [
        evaluate_case(make_case(rng.randrange(args.min_facts, args.max_facts + 1), rng), rng)
        for _ in range(args.cases)
    ]
    n = max(1, len(results))
    exact = sum(r.reconstruction_exact for r in results) / n
    graph_acc = sum(r.graph_answer_correct for r in results) / n
    shuffled_acc = sum(r.shuffled_graph_correct for r in results) / n
    mean_object_ratio = sum(r.object_token_ratio for r in results) / n
    mean_query_ratio = sum(r.query_work_ratio for r in results) / n
    max_residual = max((r.residual_ratio for r in results), default=1.0)
    pass_gate = (
        exact == 1.0
        and graph_acc == 1.0
        and shuffled_acc <= 0.05
        and mean_object_ratio <= args.max_object_token_ratio
        and mean_query_ratio <= args.max_query_work_ratio
        and max_residual <= args.max_residual_ratio
    )
    if pass_gate:
        decision = "PASS_REVERSIBLE_CORPUS_LANGUAGE_GATE"
    elif exact < 1.0 or max_residual > args.max_residual_ratio:
        decision = "FAIL_SURFACE_CANONICAL_SURFACE_LOOP"
    elif graph_acc < 1.0:
        decision = "FAIL_CANONICAL_GRAPH_QUERY"
    elif shuffled_acc > 0.05:
        decision = "FAIL_SHUFFLED_GRAPH_CONTROL"
    elif mean_object_ratio > args.max_object_token_ratio:
        decision = "FAIL_OBJECT_TOKEN_PRICING"
    else:
        decision = "FAIL_QUERY_WORK_PRICING"

    activation = (
        "[R422_REVERSIBLE_CORPUS_LANGUAGE_GATE] "
        f"active=1 decision={decision} pass={int(pass_gate)} "
        "object=surface_canonical_surface residual_map=explicit no_gpu=1 no_training_integration=1"
    )
    payload = GateResult(
        schema=SCHEMA,
        decision=decision,
        pass_gate=pass_gate,
        cases=len(results),
        exact_reconstruction_fraction=exact,
        graph_answer_accuracy=graph_acc,
        shuffled_graph_accuracy=shuffled_acc,
        mean_object_token_ratio=mean_object_ratio,
        mean_query_work_ratio=mean_query_ratio,
        max_residual_ratio=max_residual,
        activation_log=activation,
        interpretation=(
            "material_compiler_gate_only: validates controlled reversible corpus objects, "
            "not broad corpus parsing or GPU training speed."
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
                "exact_reconstruction_fraction": exact,
                "graph_answer_accuracy": graph_acc,
                "shuffled_graph_accuracy": shuffled_acc,
                "mean_object_token_ratio": mean_object_ratio,
                "mean_query_work_ratio": mean_query_ratio,
                "max_residual_ratio": max_residual,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
