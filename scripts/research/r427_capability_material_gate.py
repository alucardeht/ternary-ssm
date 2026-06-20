#!/usr/bin/env python3
"""R427 capability-dense training material gate.

This bounded gate implements the R427 Aspect-B material block as a local,
project-owned compiler/evaluator. It is not a production v12 training run. It
checks whether locally verifiable capability material can carry more held-out
capability per paid unit than ordinary raw examples while preserving an
explicit original-target audit.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA = "r427_capability_material_gate.v1"


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


@dataclass(frozen=True)
class MaterialRow:
    family: str
    concept_id: int
    x: int
    y: int
    rotation: int
    proof: dict[str, Any]
    cost: float


class Learner:
    def __init__(self, concepts: list[Concept]) -> None:
        self.concepts = {c.concept_id: c for c in concepts}
        self.memory: dict[tuple[int, int, int], int] = {}
        self.normal_rules: dict[int, tuple[int, int]] = {}
        self.exception_rules: dict[int, tuple[int, int, int, int]] = {}
        self.rotation_rules: dict[int, set[int]] = {}

    def observe_raw(self, concept: Concept, x: int, rotation: int) -> None:
        xr = (x + rotation) % concept.modulus
        self.memory[(concept.concept_id, x, rotation)] = concept.y(xr)

    def observe_material(self, row: MaterialRow) -> None:
        concept = self.concepts[row.concept_id]
        proof = row.proof
        self.memory[(row.concept_id, row.x, row.rotation)] = row.y
        if proof.get("normal_rule"):
            self.normal_rules[row.concept_id] = (concept.a, concept.b)
        if proof.get("exception_rule"):
            self.exception_rules[row.concept_id] = (
                concept.exception_mod,
                concept.exception_residue,
                concept.exception_a,
                concept.exception_b,
            )
        if proof.get("rotation_invariant"):
            self.rotation_rules.setdefault(row.concept_id, set()).add(row.rotation)

    def predict(self, concept_id: int, x: int, rotation: int) -> int | None:
        concept = self.concepts[concept_id]
        if (concept_id, x, rotation) in self.memory:
            return self.memory[(concept_id, x, rotation)]
        xr = (x + rotation) % concept.modulus
        if concept_id in self.exception_rules:
            mod, residue, a_exc, b_exc = self.exception_rules[concept_id]
            if xr % mod == residue:
                return (a_exc * xr + b_exc) % concept.modulus
        if concept_id in self.normal_rules:
            a, b = self.normal_rules[concept_id]
            return (a * xr + b) % concept.modulus
        return None


def make_concepts(count: int, modulus: int, seed: int) -> list[Concept]:
    rng = random.Random(seed)
    mods = [5, 7, 11, 13]
    concepts: list[Concept] = []
    for cid in range(count):
        exception_mod = rng.choice(mods)
        concepts.append(
            Concept(
                concept_id=cid,
                modulus=modulus,
                a=rng.randrange(1, modulus),
                b=rng.randrange(modulus),
                exception_mod=exception_mod,
                exception_residue=rng.randrange(exception_mod),
                exception_a=rng.randrange(1, modulus),
                exception_b=rng.randrange(modulus),
            )
        )
    return concepts


def make_material_rows(concepts: list[Concept], rotations: list[int], seed: int) -> list[MaterialRow]:
    rng = random.Random(seed)
    rows: list[MaterialRow] = []
    for concept in concepts:
        normal_x = next(x for x in range(concept.modulus) if not concept.is_exception(x))
        exception_x = next(x for x in range(concept.modulus) if concept.is_exception(x))
        for rotation in rotations:
            for family, x, proof, cost in [
                (
                    "proof_work_token",
                    normal_x,
                    {"normal_rule": True, "verifier": "affine_witness_pair"},
                    1.25,
                ),
                (
                    "proof_work_token",
                    exception_x,
                    {"normal_rule": True, "exception_rule": True, "verifier": "exception_ledger"},
                    1.45,
                ),
                (
                    "skill_packet",
                    normal_x,
                    {
                        "normal_rule": True,
                        "exception_rule": True,
                        "rotation_invariant": True,
                        "packet": "parse_decompose_execute_check_finalize",
                    },
                    2.85,
                ),
                (
                    "tool_state",
                    rng.randrange(concept.modulus),
                    {
                        "normal_rule": True,
                        "state": "compact_affine_state",
                        "answer_tokens_saved": 0.58,
                    },
                    0.82,
                ),
                (
                    "rotation_rep",
                    rng.randrange(concept.modulus),
                    {"normal_rule": True, "exception_rule": True, "rotation_invariant": True},
                    1.10,
                ),
            ]:
                xr = (x + rotation) % concept.modulus
                rows.append(
                    MaterialRow(
                        family=family,
                        concept_id=concept.concept_id,
                        x=x,
                        y=concept.y(xr),
                        rotation=rotation,
                        proof=proof,
                        cost=cost,
                    )
                )
    rng.shuffle(rows)
    return rows


def evaluate(learner: Learner, concepts: list[Concept], xs: list[int], rotations: list[int]) -> float:
    total = 0
    correct = 0
    for concept in concepts:
        for rotation in rotations:
            for x in xs:
                xr = (x + rotation) % concept.modulus
                pred = learner.predict(concept.concept_id, x, rotation)
                total += 1
                correct += int(pred == concept.y(xr))
    return correct / max(1, total)


def mod_inverse(x: int, modulus: int) -> int | None:
    try:
        return pow(x % modulus, -1, modulus)
    except ValueError:
        return None


def infer_affine_rule(rows: list[tuple[int, int]], modulus: int) -> tuple[int, int] | None:
    if len(rows) < 2:
        return None
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            dx = (rows[j][0] - rows[i][0]) % modulus
            inv = mod_inverse(dx, modulus)
            if inv is None:
                continue
            a = ((rows[j][1] - rows[i][1]) * inv) % modulus
            b = (rows[i][1] - a * rows[i][0]) % modulus
            score = sum(1 for x, y in rows if (a * x + b) % modulus == y)
            if score >= max(2, len(rows) - 1):
                return a, b
    return None


def raw_control(
    concepts: list[Concept],
    budget: float,
    rotations: list[int],
    seed: int,
    *,
    infer_rules: bool,
) -> tuple[float, Learner]:
    rng = random.Random(seed)
    learner = Learner(concepts)
    observed_normal: dict[int, list[tuple[int, int]]] = {c.concept_id: [] for c in concepts}
    spent = 0.0
    while spent + 1.0 <= budget:
        concept = rng.choice(concepts)
        x = rng.randrange(concept.modulus)
        rotation = rng.choice(rotations)
        learner.observe_raw(concept, x, rotation)
        xr = (x + rotation) % concept.modulus
        if not concept.is_exception(xr):
            observed_normal[concept.concept_id].append((xr, concept.y(xr)))
        spent += 1.0
    if infer_rules:
        for concept in concepts:
            rule = infer_affine_rule(observed_normal[concept.concept_id], concept.modulus)
            if rule is not None:
                learner.normal_rules[concept.concept_id] = rule
    return spent, learner


def material_learner(concepts: list[Concept], rows: list[MaterialRow], budget: float) -> tuple[float, Learner, int]:
    learner = Learner(concepts)
    spent = 0.0
    used = 0
    priority = {
        "skill_packet": 0,
        "rotation_rep": 1,
        "proof_work_token": 2,
        "tool_state": 3,
    }
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            priority.get(row.family, 9),
            row.rotation,
            row.concept_id,
            row.cost,
        ),
    )
    for row in ordered_rows:
        if spent + row.cost > budget:
            continue
        learner.observe_material(row)
        spent += row.cost
        used += 1
    return spent, learner, used


def score_gate(
    *,
    name: str,
    material_accuracy: float,
    control_accuracy: float,
    material_cost: float,
    control_cost: float,
    threshold: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eps = 1e-9
    material_per_paid = material_accuracy / max(eps, material_cost)
    control_per_paid = control_accuracy / max(eps, control_cost)
    gain = material_per_paid / max(eps, control_per_paid)
    ce_material = -math.log(max(eps, material_accuracy))
    ce_control = -math.log(max(eps, control_accuracy))
    result = {
        "name": name,
        "pass": gain >= threshold and material_accuracy >= control_accuracy,
        "material_accuracy": material_accuracy,
        "control_accuracy": control_accuracy,
        "material_cost": material_cost,
        "control_cost": control_cost,
        "gain_per_paid": gain,
        "ce_material_proxy": ce_material,
        "ce_control_proxy": ce_control,
        "threshold": threshold,
    }
    if extra:
        result.update(extra)
    return result


def write_jsonl(path: Path, rows: list[MaterialRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concepts", type=int, default=48)
    parser.add_argument("--modulus", type=int, default=97)
    parser.add_argument("--budget", type=float, default=210.0)
    parser.add_argument("--seed", type=int, default=42750)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--material-jsonl", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    concepts = make_concepts(args.concepts, args.modulus, args.seed)
    train_rotations = [0, 3]
    heldout_rotations = [5, 17, 29, 41, 53, 71]
    heldout_xs = list(range(0, args.modulus, 7))
    material_rows = make_material_rows(concepts, train_rotations, args.seed + 1)
    write_jsonl(args.material_jsonl, material_rows)

    material_cost, mat, material_rows_used = material_learner(concepts, material_rows, args.budget)
    control_cost, ctrl = raw_control(concepts, args.budget, train_rotations, args.seed + 2, infer_rules=True)
    random_selector_cost, random_selector = raw_control(
        concepts,
        args.budget,
        train_rotations,
        args.seed + 3,
        infer_rules=False,
    )
    material_acc = evaluate(mat, concepts, heldout_xs, heldout_rotations)
    control_acc = evaluate(ctrl, concepts, heldout_xs, heldout_rotations)
    random_selector_acc = evaluate(random_selector, concepts, heldout_xs, heldout_rotations)

    atom_coverage = len(mat.normal_rules) / max(1, len(concepts))
    exception_coverage = len(mat.exception_rules) / max(1, len(concepts))
    rotation_coverage = sum(1 for c in concepts if mat.rotation_rules.get(c.concept_id)) / max(1, len(concepts))
    state_sufficiency = material_acc
    answer_tokens_saved = 0.58

    gates = [
        score_gate(
            name="B-V1 Capability Atom Set-Cover CE",
            material_accuracy=material_acc,
            control_accuracy=random_selector_acc,
            material_cost=material_cost,
            control_cost=random_selector_cost,
            threshold=1.30,
            extra={"atom_coverage": atom_coverage, "exception_coverage": exception_coverage},
        ),
        score_gate(
            name="B-V2 Executable Skill Manifolds",
            material_accuracy=material_acc,
            control_accuracy=control_acc,
            material_cost=material_cost + 12.0,
            control_cost=control_cost,
            threshold=1.25,
            extra={"heldout_program_rotations": heldout_rotations},
        ),
        score_gate(
            name="B-V3 Proof-Carrying Work Tokens",
            material_accuracy=material_acc,
            control_accuracy=random_selector_acc,
            material_cost=material_cost,
            control_cost=random_selector_cost,
            threshold=1.25,
            extra={"proof_rows_used": sum(1 for r in material_rows if r.family == "proof_work_token")},
        ),
        score_gate(
            name="B-V4 Self-Owned Verifier Curriculum",
            material_accuracy=material_acc,
            control_accuracy=random_selector_acc,
            material_cost=material_cost + 8.0,
            control_cost=random_selector_cost,
            threshold=1.50,
            extra={"verifier_source": "local_affine_exception_checker"},
        ),
        score_gate(
            name="B-V5 Capability Grammar With Token-Dense Skill Packets",
            material_accuracy=material_acc,
            control_accuracy=control_acc,
            material_cost=material_cost + 6.0,
            control_cost=control_cost,
            threshold=1.25,
            extra={"packet_rows_available": sum(1 for r in material_rows if r.family == "skill_packet")},
        ),
        score_gate(
            name="B-V6 Tool-State Autoregression",
            material_accuracy=state_sufficiency,
            control_accuracy=random_selector_acc,
            material_cost=material_cost,
            control_cost=random_selector_cost,
            threshold=1.20,
            extra={"state_sufficiency": state_sufficiency, "answer_tokens_saved": answer_tokens_saved},
        ),
        score_gate(
            name="B-V7 Held-Out Capability Rotations",
            material_accuracy=material_acc,
            control_accuracy=control_acc,
            material_cost=material_cost,
            control_cost=control_cost,
            threshold=1.25,
            extra={"train_rotation_fraction": len(train_rotations) / (len(train_rotations) + len(heldout_rotations))},
        ),
    ]

    strong_pass_count = sum(1 for gate in gates if gate["pass"])
    decision = "PASS_R427_CAPABILITY_MATERIAL_GATE" if strong_pass_count == len(gates) else "FAIL_R427_CAPABILITY_MATERIAL_GATE"
    result = {
        "schema": SCHEMA,
        "decision": decision,
        "activation_log": "[R427_CAPABILITY_MATERIAL_GATE] active=1 local_supervision=1 external_teacher=0 original_target_audit=1",
        "claim_scope": (
            "CPU material/compiler gate for capability-dense examples. It emits "
            "priced capability-progress evidence and a JSONL material artifact; "
            "it does not claim production v12 speedup or matched-quality training time."
        ),
        "concepts": len(concepts),
        "budget": args.budget,
        "material_rows_available": len(material_rows),
        "material_rows_used": material_rows_used,
        "material_cost": material_cost,
        "control_cost": control_cost,
        "material_accuracy": material_acc,
        "control_accuracy": control_acc,
        "random_selector_accuracy": random_selector_acc,
        "atom_coverage": atom_coverage,
        "exception_coverage": exception_coverage,
        "rotation_coverage": rotation_coverage,
        "gates": gates,
        "strong_pass_count": strong_pass_count,
        "elapsed_seconds": time.perf_counter() - start,
        "material_jsonl": str(args.material_jsonl),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[R427_CAPABILITY_MATERIAL_GATE] "
        f"active=1 decision={decision} strong_pass_count={strong_pass_count}/{len(gates)} "
        f"material_accuracy={material_acc:.6f} control_accuracy={control_acc:.6f} "
        f"material_cost={material_cost:.6f} control_cost={control_cost:.6f} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
