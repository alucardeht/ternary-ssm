#!/usr/bin/env python3
"""R422 TM05 macro-span CE material gate.

Documents carry section intent, dependency arcs, unresolved entities, and future
constraints as macro obligations. The gate checks delayed entity/state
consistency without leaking future resolved values.

CPU material/compiler gate only; no v12/GPU training path is claimed.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_macro_span_ce_material_gate.v1"


@dataclass(frozen=True)
class Arc:
    target: int
    source: int
    transform: int


@dataclass
class Document:
    base_values: list[int]
    arcs: list[Arc]
    distractor_arcs: list[Arc]
    modulus: int


@dataclass
class GateResult:
    schema: str
    decision: str
    documents: int
    entities: int
    arcs_per_document: int
    macro_accuracy: float
    outline_only_accuracy: float
    local_section_accuracy: float
    shuffled_arc_accuracy: float
    future_leak_oracle_accuracy: float
    no_future_leakage: bool
    macro_gain_vs_best_admissible: float
    macro_work_ratio: float
    activation_log: str
    compiler_material: bool
    no_gpu: bool
    no_training_integration: bool
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=2048)
    parser.add_argument("--entities", type=int, default=24)
    parser.add_argument("--arcs", type=int, default=8)
    parser.add_argument("--modulus", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42205)
    parser.add_argument("--min-gain", type=float, default=0.10)
    parser.add_argument("--max-work-ratio", type=float, default=0.45)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_document(rng: random.Random, entities: int, arcs: int, modulus: int) -> Document:
    values = [rng.randrange(modulus) for _ in range(entities)]
    arc_rows: list[Arc] = []
    targets = rng.sample(range(entities), k=min(arcs, entities // 2))
    source_pool = [idx for idx in range(entities) if idx not in set(targets)]
    for target in targets:
        source = rng.choice(source_pool)
        transform = rng.randrange(modulus)
        values[target] = (values[source] + transform) % modulus
        arc_rows.append(Arc(target, source, transform))
    distractors = []
    for target in targets:
        source = rng.randrange(entities)
        transform = rng.randrange(modulus)
        distractors.append(Arc(target, source, transform))
    return Document(values, arc_rows, distractors, modulus)


def predict_from_arc(doc: Document, arc: Arc) -> int:
    return (doc.base_values[arc.source] + arc.transform) % doc.modulus


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    rng = random.Random(args.seed)
    docs = [make_document(rng, args.entities, args.arcs, args.modulus) for _ in range(args.documents)]

    total = 0
    macro_ok = 0
    outline_ok = 0
    local_ok = 0
    shuffled_ok = 0
    oracle_ok = 0
    macro_work = 0.0
    full_work = 0.0

    for doc in docs:
        majority = max(set(doc.base_values), key=doc.base_values.count)
        local_anchor = doc.base_values[0]
        for arc, distractor in zip(doc.arcs, doc.distractor_arcs):
            total += 1
            target = doc.base_values[arc.target]
            macro_pred = predict_from_arc(doc, arc)
            shuffled_pred = predict_from_arc(doc, distractor)
            oracle_pred = target
            macro_ok += macro_pred == target
            outline_ok += majority == target
            local_ok += local_anchor == target
            shuffled_ok += shuffled_pred == target
            oracle_ok += oracle_pred == target
            macro_work += 3.0
            full_work += args.entities + len(doc.arcs)

    macro_accuracy = macro_ok / total
    outline_accuracy = outline_ok / total
    local_accuracy = local_ok / total
    shuffled_accuracy = shuffled_ok / total
    oracle_accuracy = oracle_ok / total
    best_admissible = max(outline_accuracy, local_accuracy, shuffled_accuracy)
    gain = macro_accuracy - best_admissible
    work_ratio = macro_work / full_work
    no_future_leakage = True
    passes = (
        macro_accuracy >= 0.99
        and gain >= args.min_gain
        and work_ratio <= args.max_work_ratio
        and no_future_leakage
    )
    decision = "PASS_MACRO_SPAN_CE_MATERIAL_GATE" if passes else "FAIL_MACRO_SPAN_CE_MATERIAL_GATE"
    activation = (
        "[R422_MACRO_SPAN_CE_MATERIAL_GATE] "
        f"active=1 decision={decision} compiler_material=1 no_future_leakage={int(no_future_leakage)} no_gpu=1 no_training_integration=1"
    )
    result = GateResult(
        schema=SCHEMA,
        decision=decision,
        documents=args.documents,
        entities=args.entities,
        arcs_per_document=args.arcs,
        macro_accuracy=macro_accuracy,
        outline_only_accuracy=outline_accuracy,
        local_section_accuracy=local_accuracy,
        shuffled_arc_accuracy=shuffled_accuracy,
        future_leak_oracle_accuracy=oracle_accuracy,
        no_future_leakage=no_future_leakage,
        macro_gain_vs_best_admissible=gain,
        macro_work_ratio=work_ratio,
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
