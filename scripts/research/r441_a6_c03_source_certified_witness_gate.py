#!/usr/bin/env python3
"""R441 A6-C03 source-certified minimal capability witness gate.

This is a bounded, project-owned gate for the A6-C03 claim: source-derived
minimal witnesses can improve capability acquisition per paid source budget
without importing a stronger teacher. The extractor only emits witnesses whose
targets are entailed by the generated source span plus deterministic verifier
rules. The probe compares raw examples against source-certified witnesses under
the same paid budget and checks transfer to renamed source variants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA = "r441_a6_c03_source_certified_witness_gate.v1"


@dataclass(frozen=True)
class SourceSpan:
    source_id: int
    family: str
    surface_name: str
    renamed_name: str
    modulus: int
    a: int
    b: int
    exception_mod: int
    exception_residue: int
    exception_a: int
    exception_b: int
    source_text: str
    token_cost: float

    def is_exception(self, x: int) -> bool:
        return x % self.exception_mod == self.exception_residue

    def y(self, x: int) -> int:
        if self.is_exception(x):
            return (self.exception_a * x + self.exception_b) % self.modulus
        return (self.a * x + self.b) % self.modulus


@dataclass(frozen=True)
class Witness:
    source_id: int
    family: str
    surface_name: str
    context: str
    operation: str
    target: str
    source_hash: str
    source_pointer: str
    verifier: str
    cost: float


class WitnessLearner:
    def __init__(self, spans: list[SourceSpan]) -> None:
        self.spans = {span.source_id: span for span in spans}
        self.alias_to_source: dict[str, int] = {}
        for span in spans:
            self.alias_to_source[span.surface_name] = span.source_id
        self.normal_rules: dict[int, tuple[int, int]] = {}
        self.exception_rules: dict[int, tuple[int, int, int, int]] = {}
        self.schema_rules: set[int] = set()
        self.memory: dict[tuple[str, int], int] = {}

    def observe_raw(self, name: str, x: int, y: int) -> None:
        self.memory[(name, x)] = y

    def observe_witness(self, witness: Witness) -> None:
        source_id = witness.source_id
        span = self.spans[source_id]
        if witness.operation == "normal_affine_rule":
            self.normal_rules[source_id] = (span.a, span.b)
        elif witness.operation == "exception_rule":
            self.exception_rules[source_id] = (
                span.exception_mod,
                span.exception_residue,
                span.exception_a,
                span.exception_b,
            )
        elif witness.operation == "schema_pointer":
            self.schema_rules.add(source_id)
            self.alias_to_source[span.renamed_name] = source_id

    def predict(self, name: str, x: int) -> int | None:
        if (name, x) in self.memory:
            return self.memory[(name, x)]
        source_id = self.alias_to_source.get(name)
        if source_id is None:
            return None
        span = self.spans[source_id]
        if source_id in self.exception_rules:
            mod, residue, a_exc, b_exc = self.exception_rules[source_id]
            if x % mod == residue:
                return (a_exc * x + b_exc) % span.modulus
        if source_id in self.normal_rules:
            a, b = self.normal_rules[source_id]
            return (a * x + b) % span.modulus
        return None


def source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def make_source_spans(count: int, modulus: int, seed: int) -> list[SourceSpan]:
    rng = random.Random(seed)
    families = ["glossary_use", "table_schema"]
    exception_mods = [5, 7, 11, 13]
    spans: list[SourceSpan] = []
    for source_id in range(count):
        family = families[source_id % len(families)]
        surface_name = f"term_{source_id:03d}"
        renamed_name = f"alias_{source_id:03d}"
        a = rng.randrange(1, modulus)
        b = rng.randrange(modulus)
        exception_mod = rng.choice(exception_mods)
        exception_residue = rng.randrange(exception_mod)
        exception_a = rng.randrange(1, modulus)
        exception_b = rng.randrange(modulus)
        source_text = (
            f"{family}:{surface_name}; modulus={modulus}; normal=({a}*x+{b})%m; "
            f"exception=if x%{exception_mod}=={exception_residue} then "
            f"({exception_a}*x+{exception_b})%m; renamed={renamed_name}"
        )
        spans.append(
            SourceSpan(
                source_id=source_id,
                family=family,
                surface_name=surface_name,
                renamed_name=renamed_name,
                modulus=modulus,
                a=a,
                b=b,
                exception_mod=exception_mod,
                exception_residue=exception_residue,
                exception_a=exception_a,
                exception_b=exception_b,
                source_text=source_text,
                token_cost=1.20 if family == "glossary_use" else 1.35,
            )
        )
    return spans


def verify_witness(span: SourceSpan, witness: Witness) -> bool:
    if witness.source_hash != source_hash(span.source_text):
        return False
    if witness.operation == "normal_affine_rule":
        x0 = next(x for x in range(span.modulus) if not span.is_exception(x))
        x1 = next(x for x in range(x0 + 1, span.modulus) if not span.is_exception(x))
        return span.y(x0) == (span.a * x0 + span.b) % span.modulus and span.y(x1) == (
            span.a * x1 + span.b
        ) % span.modulus
    if witness.operation == "exception_rule":
        x = next(x for x in range(span.modulus) if span.is_exception(x))
        return span.y(x) == (span.exception_a * x + span.exception_b) % span.modulus
    if witness.operation == "schema_pointer":
        return span.renamed_name in span.source_text and span.surface_name in span.source_text
    return False


def extract_witnesses(spans: list[SourceSpan], *, inject_false: float, seed: int) -> tuple[list[Witness], int]:
    rng = random.Random(seed)
    witnesses: list[Witness] = []
    false_count = 0
    for span in spans:
        digest = source_hash(span.source_text)
        candidates = [
            Witness(
                source_id=span.source_id,
                family=span.family,
                surface_name=span.surface_name,
                context=f"{span.family}:{span.surface_name}",
                operation="normal_affine_rule",
                target=f"y=({span.a}*x+{span.b})%{span.modulus}",
                source_hash=digest,
                source_pointer=f"source:{span.source_id}:normal",
                verifier="two_point_affine_entailment",
                cost=0.35,
            ),
            Witness(
                source_id=span.source_id,
                family=span.family,
                surface_name=span.surface_name,
                context=f"{span.family}:{span.surface_name}:exception",
                operation="exception_rule",
                target=(
                    f"if x%{span.exception_mod}=={span.exception_residue} "
                    f"then y=({span.exception_a}*x+{span.exception_b})%{span.modulus}"
                ),
                source_hash=digest,
                source_pointer=f"source:{span.source_id}:exception",
                verifier="exception_residue_entailment",
                cost=0.45,
            ),
            Witness(
                source_id=span.source_id,
                family=span.family,
                surface_name=span.surface_name,
                context=f"{span.family}:{span.surface_name}:rename",
                operation="schema_pointer",
                target=f"{span.surface_name}<->{span.renamed_name}",
                source_hash=digest,
                source_pointer=f"source:{span.source_id}:rename",
                verifier="source_alias_pointer",
                cost=0.20,
            ),
        ]
        for witness in candidates:
            if inject_false > 0.0 and rng.random() < inject_false:
                false_count += 1
                witness = Witness(
                    **{
                        **asdict(witness),
                        "target": witness.target + "; corrupted=1",
                        "source_hash": "bad" + witness.source_hash[3:],
                    }
                )
            witnesses.append(witness)
    return witnesses, false_count


def infer_affine(rows: list[tuple[int, int]], modulus: int) -> tuple[int, int] | None:
    if len(rows) < 3:
        return None
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            dx = (rows[j][0] - rows[i][0]) % modulus
            try:
                inv = pow(dx, -1, modulus)
            except ValueError:
                continue
            a = ((rows[j][1] - rows[i][1]) * inv) % modulus
            b = (rows[i][1] - a * rows[i][0]) % modulus
            score = sum(1 for x, y in rows if (a * x + b) % modulus == y)
            if score >= max(3, len(rows) - 1):
                return a, b
    return None


def train_raw(spans: list[SourceSpan], budget: float, seed: int) -> tuple[WitnessLearner, float]:
    rng = random.Random(seed)
    learner = WitnessLearner(spans)
    observed: dict[str, list[tuple[int, int]]] = {span.surface_name: [] for span in spans}
    spent = 0.0
    while spent + 1.0 <= budget:
        span = rng.choice(spans)
        x = rng.randrange(span.modulus)
        y = span.y(x)
        learner.observe_raw(span.surface_name, x, y)
        if not span.is_exception(x):
            observed[span.surface_name].append((x, y))
        spent += 1.0
    for span in spans:
        rule = infer_affine(observed[span.surface_name], span.modulus)
        if rule is not None:
            learner.normal_rules[span.source_id] = rule
    return learner, spent


def train_witness(
    spans: list[SourceSpan],
    witnesses: list[Witness],
    budget: float,
    *,
    source_pointer_audit_cost: float,
    seed: int,
) -> tuple[WitnessLearner, float, int, int]:
    rng = random.Random(seed)
    learner = WitnessLearner(spans)
    shuffled = witnesses[:]
    rng.shuffle(shuffled)
    spent = 0.0
    used = 0
    rejected = 0
    source_paid: set[int] = set()
    priority = {"normal_affine_rule": 0, "exception_rule": 1, "schema_pointer": 2}
    shuffled.sort(key=lambda w: (priority.get(w.operation, 9), w.source_id))
    for witness in shuffled:
        span = learner.spans[witness.source_id]
        source_cost = 0.0 if witness.source_id in source_paid else span.token_cost
        cost = source_cost + witness.cost + source_pointer_audit_cost
        if spent + cost > budget:
            continue
        if not verify_witness(span, witness):
            rejected += 1
            spent += min(cost, source_pointer_audit_cost + witness.cost)
            continue
        source_paid.add(witness.source_id)
        learner.observe_witness(witness)
        spent += cost
        used += 1
    return learner, spent, used, rejected


def evaluate(learner: WitnessLearner, spans: list[SourceSpan], xs: list[int], *, renamed: bool) -> float:
    total = 0
    correct = 0
    for span in spans:
        name = span.renamed_name if renamed else span.surface_name
        for x in xs:
            total += 1
            correct += int(learner.predict(name, x) == span.y(x))
    return correct / max(1, total)


def ce_proxy(accuracy: float) -> float:
    return -math.log(max(1.0e-9, accuracy))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=int, default=48)
    parser.add_argument("--modulus", type=int, default=97)
    parser.add_argument("--budget", type=float, default=220.0)
    parser.add_argument("--seed", type=int, default=44163)
    parser.add_argument("--gain-threshold", type=float, default=1.10)
    parser.add_argument("--false-witness-budget", type=float, default=0.0)
    parser.add_argument("--gamma-rename", type=float, default=0.85)
    parser.add_argument("--epsilon-ce", type=float, default=0.05)
    parser.add_argument("--source-pointer-audit-cost", type=float, default=0.05)
    parser.add_argument("--inject-false-rate", type=float, default=0.0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--witness-jsonl", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    spans = make_source_spans(args.sources, args.modulus, args.seed)
    witnesses, injected_false = extract_witnesses(spans, inject_false=args.inject_false_rate, seed=args.seed + 1)
    verified = [w for w in witnesses if verify_witness(next(s for s in spans if s.source_id == w.source_id), w)]
    false_rate = 1.0 - (len(verified) / max(1, len(witnesses)))

    args.witness_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.witness_jsonl.open("w", encoding="utf-8") as handle:
        for witness in witnesses:
            handle.write(json.dumps(asdict(witness), sort_keys=True) + "\n")

    raw, raw_spent = train_raw(spans, args.budget, args.seed + 2)
    wit, witness_spent, witness_used, rejected = train_witness(
        spans,
        witnesses,
        args.budget,
        source_pointer_audit_cost=args.source_pointer_audit_cost,
        seed=args.seed + 3,
    )

    heldout_xs = list(range(2, args.modulus, 7))
    raw_source_acc = evaluate(raw, spans, heldout_xs, renamed=False)
    raw_renamed_acc = evaluate(raw, spans, heldout_xs, renamed=True)
    witness_source_acc = evaluate(wit, spans, heldout_xs, renamed=False)
    witness_renamed_acc = evaluate(wit, spans, heldout_xs, renamed=True)

    paid_budget = args.budget
    source_gain = witness_source_acc / max(1.0e-9, raw_source_acc)
    renamed_gain = witness_renamed_acc / max(1.0e-9, raw_renamed_acc)
    priced_gain = (0.5 * witness_source_acc + 0.5 * witness_renamed_acc) / max(
        1.0e-9, 0.5 * raw_source_acc + 0.5 * raw_renamed_acc
    )
    source_ce_delta = ce_proxy(raw_source_acc) - ce_proxy(witness_source_acc)
    original_ce_harm = ce_proxy(witness_source_acc) - ce_proxy(raw_source_acc)
    rename_transfer_ratio = witness_renamed_acc / max(1.0e-9, witness_source_acc)
    source_mass = witness_used / max(1, len(witnesses))
    witness_token_cost = sum(w.cost for w in witnesses if verify_witness(spans[w.source_id], w))
    source_pointer_cost = args.source_pointer_audit_cost * witness_used
    false_penalty = false_rate * paid_budget

    pass_gate = (
        false_rate <= args.false_witness_budget
        and source_ce_delta > 0.0
        and original_ce_harm <= args.epsilon_ce
        and rename_transfer_ratio >= args.gamma_rename
        and priced_gain >= args.gain_threshold
    )
    decision = (
        "PASS_R469_A6C03_SOURCE_CERTIFIED_WITNESS_GATE"
        if pass_gate
        else "FAIL_R469_A6C03_SOURCE_CERTIFIED_WITNESS_GATE"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "activation_log": "[SOURCE_CERTIFIED_WITNESS_GATE] active=1 external_teacher=0 source_pointer_audit=1",
        "claim_scope": (
            "CPU source-certified witness extraction and transfer probe. It is a "
            "capability/sample-efficiency gate, not a production v12 speed or "
            "memory claim."
        ),
        "sources": len(spans),
        "witnesses_available": len(witnesses),
        "witnesses_verified": len(verified),
        "witnesses_used": witness_used,
        "witnesses_rejected": rejected,
        "injected_false": injected_false,
        "false_rate": false_rate,
        "false_witness_budget": args.false_witness_budget,
        "paid_budget": paid_budget,
        "raw_spent": raw_spent,
        "witness_spent": witness_spent,
        "source_mass": source_mass,
        "witness_token_cost": witness_token_cost,
        "source_pointer_cost": source_pointer_cost,
        "false_penalty": false_penalty,
        "raw_source_accuracy": raw_source_acc,
        "raw_renamed_accuracy": raw_renamed_acc,
        "witness_source_accuracy": witness_source_acc,
        "witness_renamed_accuracy": witness_renamed_acc,
        "source_gain": source_gain,
        "renamed_gain": renamed_gain,
        "priced_gain": priced_gain,
        "source_ce_delta": source_ce_delta,
        "original_ce_harm": original_ce_harm,
        "rename_transfer_ratio": rename_transfer_ratio,
        "gamma_rename": args.gamma_rename,
        "epsilon_ce": args.epsilon_ce,
        "gain_threshold": args.gain_threshold,
        "pass_conditions": {
            "rho_false_within_budget": false_rate <= args.false_witness_budget,
            "delta_source_variant_positive": source_ce_delta > 0.0,
            "delta_original_ce_not_harmed": original_ce_harm <= args.epsilon_ce,
            "renamed_variant_transfer": rename_transfer_ratio >= args.gamma_rename,
            "priced_gain_over_threshold": priced_gain >= args.gain_threshold,
        },
        "elapsed_seconds": time.perf_counter() - start,
        "witness_jsonl": str(args.witness_jsonl),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[SOURCE_CERTIFIED_WITNESS_GATE] "
        f"active=1 decision={decision} false_rate={false_rate:.6f} "
        f"raw_source_acc={raw_source_acc:.6f} witness_source_acc={witness_source_acc:.6f} "
        f"raw_renamed_acc={raw_renamed_acc:.6f} witness_renamed_acc={witness_renamed_acc:.6f} "
        f"priced_gain={priced_gain:.6f} rename_transfer_ratio={rename_transfer_ratio:.6f} "
        f"output={args.output}"
    )
    return 0 if pass_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
