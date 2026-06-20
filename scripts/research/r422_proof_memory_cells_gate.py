#!/usr/bin/env python3
"""R422 proof-carrying memory cells gate.

Tests whether memory cells gated by local certificates reduce false recall on
conflicting probes compared with confidence-only and retrieval-like controls.
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


SCHEMA = "r422_proof_memory_cells_gate.v1"


@dataclass(frozen=True)
class Cell:
    key: int
    value: int
    proof: str
    confidence: float


@dataclass(frozen=True)
class Probe:
    key: int
    claimed_value: int
    proof: str
    gold_mode: str
    gold_value: int


@dataclass
class ArmMetrics:
    arm: str
    mode_score: float
    valid_answer_accuracy: float
    invalid_defer_accuracy: float
    false_recall_rate: float
    mean_ce: float
    activation_rate: float


@dataclass
class GateResult:
    schema: str
    decision: str
    pass_gate: bool
    probes: int
    valid_fraction: float
    proof_mode_score: float
    proof_false_recall_rate: float
    proof_mean_ce: float
    best_control_mean_ce: float
    ce_gain_vs_best_control: float
    activation_log: str
    interpretation: str
    arms: list[ArmMetrics]
    elapsed_seconds: float


def make_proof(key: int, value: int, salt: str, chars: int) -> str:
    return hashlib.sha256(f"{salt}:{key}:{value}".encode("ascii")).hexdigest()[:chars]


def make_world(args: argparse.Namespace, rng: random.Random) -> tuple[dict[int, Cell], list[Probe]]:
    salt = f"r422-sm03-{args.seed}"
    cells: dict[int, Cell] = {}
    for key in range(args.cells):
        value = rng.randrange(args.values)
        cells[key] = Cell(
            key=key,
            value=value,
            proof=make_proof(key, value, salt, args.proof_chars),
            confidence=rng.uniform(0.65, 0.95),
        )

    probes: list[Probe] = []
    for _ in range(args.probes):
        key = rng.randrange(args.cells)
        cell = cells[key]
        p = rng.random()
        if p < args.valid_fraction:
            probes.append(Probe(key, cell.value, cell.proof, "answer", cell.value))
        elif p < args.valid_fraction + args.conflict_fraction:
            wrong = rng.choice([v for v in range(args.values) if v != cell.value])
            if rng.random() < args.adversarial_confidence_fraction:
                cells[key] = Cell(key, cell.value, cell.proof, rng.uniform(0.90, 0.99))
            probes.append(Probe(key, wrong, make_proof(key, wrong, salt, args.proof_chars), "defer", cell.value))
        else:
            wrong = rng.choice([v for v in range(args.values) if v != cell.value])
            probes.append(Probe(key, wrong, "", "defer", cell.value))
    return cells, probes


def ce(prob: float) -> float:
    return -math.log(max(1e-9, min(1.0, prob)))


def score_arm(arm: str, cells: dict[int, Cell], probes: list[Probe]) -> ArmMetrics:
    valid = [probe for probe in probes if probe.gold_mode == "answer"]
    invalid = [probe for probe in probes if probe.gold_mode == "defer"]
    valid_correct = 0
    invalid_defer = 0
    false_recall = 0
    active = 0
    ce_sum = 0.0
    for probe in probes:
        cell = cells[probe.key]
        if arm == "proof_memory":
            if probe.proof == cell.proof and probe.claimed_value == cell.value:
                mode, value, prob = "answer", cell.value, 0.96
            else:
                mode, value, prob = "defer", None, 0.95
        elif arm == "confidence_only":
            if cell.confidence >= 0.72:
                mode, value, prob = "answer", probe.claimed_value, cell.confidence
            else:
                mode, value, prob = "defer", None, 1.0 - cell.confidence
        elif arm == "retrieval_like":
            mode, value, prob = "answer", cell.value, 0.80
        elif arm == "no_memory":
            mode, value, prob = "defer", None, 0.80
        else:
            raise ValueError(f"unknown arm: {arm}")

        active += int(mode == "answer")
        if probe.gold_mode == "answer":
            ok = mode == "answer" and value == probe.gold_value
            valid_correct += int(ok)
            ce_sum += ce(prob if ok else 1.0 - prob)
        else:
            ok = mode == "defer"
            invalid_defer += int(ok)
            false_recall += int(not ok)
            ce_sum += ce(prob if ok else 1.0 - prob)

    valid_acc = valid_correct / max(1, len(valid))
    defer_acc = invalid_defer / max(1, len(invalid))
    false_rate = false_recall / max(1, len(invalid))
    mode_score = 0.65 * valid_acc + 0.35 * defer_acc - false_rate
    return ArmMetrics(
        arm=arm,
        mode_score=mode_score,
        valid_answer_accuracy=valid_acc,
        invalid_defer_accuracy=defer_acc,
        false_recall_rate=false_rate,
        mean_ce=ce_sum / max(1, len(probes)),
        activation_rate=active / max(1, len(probes)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=512)
    parser.add_argument("--values", type=int, default=16)
    parser.add_argument("--probes", type=int, default=4096)
    parser.add_argument("--valid-fraction", type=float, default=0.55)
    parser.add_argument("--conflict-fraction", type=float, default=0.30)
    parser.add_argument("--adversarial-confidence-fraction", type=float, default=0.50)
    parser.add_argument("--proof-chars", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42203)
    parser.add_argument("--min-mode-score", type=float, default=0.95)
    parser.add_argument("--max-false-recall", type=float, default=0.02)
    parser.add_argument("--min-ce-gain", type=float, default=0.25)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    rng = random.Random(args.seed)
    cells, probes = make_world(args, rng)
    arms = [
        score_arm(name, cells, probes)
        for name in ("proof_memory", "confidence_only", "retrieval_like", "no_memory")
    ]
    proof = next(arm for arm in arms if arm.arm == "proof_memory")
    best_control_ce = min(arm.mean_ce for arm in arms if arm.arm != "proof_memory")
    ce_gain = best_control_ce - proof.mean_ce
    valid_fraction = sum(1 for probe in probes if probe.gold_mode == "answer") / max(1, len(probes))
    pass_gate = (
        proof.mode_score >= args.min_mode_score
        and proof.false_recall_rate <= args.max_false_recall
        and ce_gain >= args.min_ce_gain
    )
    if pass_gate:
        decision = "PASS_PROOF_MEMORY_CELLS_GATE"
    elif proof.false_recall_rate > args.max_false_recall:
        decision = "FAIL_FALSE_RECALL"
    elif proof.mode_score < args.min_mode_score:
        decision = "FAIL_MODE_SCORE"
    else:
        decision = "FAIL_CE_GAIN"

    activation = (
        "[R422_PROOF_MEMORY_CELLS_GATE] "
        f"active=1 decision={decision} pass={int(pass_gate)} "
        "local_certificate=1 no_gpu=1 no_training_integration=1"
    )
    payload = GateResult(
        schema=SCHEMA,
        decision=decision,
        pass_gate=pass_gate,
        probes=len(probes),
        valid_fraction=valid_fraction,
        proof_mode_score=proof.mode_score,
        proof_false_recall_rate=proof.false_recall_rate,
        proof_mean_ce=proof.mean_ce,
        best_control_mean_ce=best_control_ce,
        ce_gain_vs_best_control=ce_gain,
        activation_log=activation,
        interpretation=(
            "controlled_conflict_gate_only: tests local certificate activation; "
            "not a production v12 path or speed claim."
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
                "proof_mode_score": proof.mode_score,
                "proof_false_recall_rate": proof.false_recall_rate,
                "proof_mean_ce": proof.mean_ce,
                "best_control_mean_ce": best_control_ce,
                "ce_gain_vs_best_control": ce_gain,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
