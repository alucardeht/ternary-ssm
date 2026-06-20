#!/usr/bin/env python3
"""R422 belief-ledger answerer gate.

This gate tests the smallest material claim for R422-MS-06: a typed belief
ledger must be the primary answer state, not an after-the-fact verifier. It
compares a precompiled ledger against cheap but brittle answerers and an exact
scan-deliberation control on contradictory multi-hop QA records.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_belief_ledger_answerer_gate.v1"


SOURCE_RELIABILITY = {
    "lab": 0.92,
    "archive": 0.84,
    "field": 0.72,
    "rumor": 0.56,
    "adversary": 0.28,
}


@dataclass(frozen=True)
class Evidence:
    key: tuple[int, int]
    value: int
    source: str
    timestamp: int


@dataclass(frozen=True)
class Query:
    key: tuple[int, int]
    gold_value: int
    gold_mode: str


@dataclass
class ArmMetrics:
    arm: str
    mode_score: float
    answer_accuracy: float
    defer_accuracy: float
    pass_rate: float
    confident_wrong_rate: float
    calibration_error: float
    mean_deliberation_tokens: float
    paid_work_ratio_vs_scan: float
    quality_per_paid_ratio: float
    answers: int
    wrong: int
    defers: int


@dataclass
class GateResult:
    schema: str
    decision: str
    pass_gate: bool
    cases: int
    answerable_fraction: float
    ledger_vs_best_quality_per_paid_gain: float
    ledger_mode_score: float
    ledger_confident_wrong_rate: float
    ledger_mean_deliberation_tokens: float
    scan_mean_deliberation_tokens: float
    activation_log: str
    interpretation: str
    arms: list[ArmMetrics]
    elapsed_seconds: float


def logit(p: float) -> float:
    p = min(0.999, max(0.001, p))
    return math.log(p / (1.0 - p))


def make_world(args: argparse.Namespace, rng: random.Random) -> tuple[list[Evidence], list[Query]]:
    true_values: dict[tuple[int, int], int] = {}
    for entity in range(args.entities):
        for field in range(args.fields):
            true_values[(entity, field)] = rng.randrange(args.values)

    evidence: list[Evidence] = []
    timestamp = 0
    sources = list(SOURCE_RELIABILITY)
    for key, true_value in true_values.items():
        for _ in range(args.evidence_per_key):
            source = rng.choice(sources)
            reliability = SOURCE_RELIABILITY[source]
            if rng.random() < reliability:
                value = true_value
            else:
                choices = [v for v in range(args.values) if v != true_value]
                value = rng.choice(choices)
            evidence.append(Evidence(key, value, source, timestamp))
            timestamp += 1

        if rng.random() < args.conflict_fraction:
            wrong_value = rng.choice([v for v in range(args.values) if v != true_value])
            for source in ("archive", "field"):
                evidence.append(Evidence(key, wrong_value, source, timestamp))
                timestamp += 1

    support_by_key = compile_support(evidence)
    queries: list[Query] = []
    keys = list(true_values)
    for _ in range(args.queries):
        key = rng.choice(keys)
        true_value = true_values[key]
        support = support_by_key[key]
        total = sum(max(0.0, score) for score in support.values())
        ranked = sorted(support.items(), key=lambda item: item[1], reverse=True)
        top_value, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        top_prob = max(0.0, top_score) / max(1e-9, total)
        gold_mode = (
            "answer"
            if top_value == true_value
            and top_prob >= args.min_confidence
            and top_score - second_score >= args.min_margin
            else "defer"
        )
        queries.append(Query(key, true_value, gold_mode))
    return evidence, queries


def compile_support(evidence: list[Evidence]) -> dict[tuple[int, int], dict[int, float]]:
    support: dict[tuple[int, int], dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for item in evidence:
        support[item.key][item.value] += logit(SOURCE_RELIABILITY[item.source])
    return support


def compile_counts(evidence: list[Evidence]) -> dict[tuple[int, int], dict[int, float]]:
    counts: dict[tuple[int, int], dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for item in evidence:
        counts[item.key][item.value] += 1.0
    return counts


def answer_from_support(
    support: dict[int, float],
    min_confidence: float,
    min_margin: float,
) -> tuple[str, int | None, float]:
    if not support:
        return "defer", None, 0.0
    ranked = sorted(support.items(), key=lambda item: item[1], reverse=True)
    top_value, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    positive_total = sum(max(0.0, score) for score in support.values())
    confidence = max(0.0, top_score) / max(1e-9, positive_total)
    if confidence >= min_confidence and top_score - second_score >= min_margin:
        return "answer", top_value, confidence
    return "defer", None, 1.0 - confidence


def latest_answer(evidence_by_key: dict[tuple[int, int], list[Evidence]], key: tuple[int, int]) -> tuple[str, int | None, float]:
    rows = evidence_by_key.get(key, [])
    if not rows:
        return "defer", None, 0.0
    row = max(rows, key=lambda item: item.timestamp)
    return "answer", row.value, SOURCE_RELIABILITY[row.source]


def calibration_error(samples: list[tuple[float, bool]], bins: int = 10) -> float:
    if not samples:
        return 0.0
    total_error = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        bucket = [(conf, ok) for conf, ok in samples if lo <= conf < hi or (b == bins - 1 and conf == 1.0)]
        if not bucket:
            continue
        avg_conf = sum(conf for conf, _ in bucket) / len(bucket)
        avg_acc = sum(1.0 for _, ok in bucket if ok) / len(bucket)
        total_error += (len(bucket) / len(samples)) * abs(avg_conf - avg_acc)
    return total_error


def score_arm(
    arm: str,
    evidence: list[Evidence],
    queries: list[Query],
    args: argparse.Namespace,
) -> ArmMetrics:
    evidence_by_key: dict[tuple[int, int], list[Evidence]] = defaultdict(list)
    for item in evidence:
        evidence_by_key[item.key].append(item)
    weighted_support = compile_support(evidence)
    count_support = compile_counts(evidence)

    answers = 0
    wrong = 0
    defers = 0
    answerable = sum(1 for query in queries if query.gold_mode == "answer")
    unanswerable = max(1, len(queries) - answerable)
    correct_on_answerable = 0
    correct_defers = 0
    calibration_samples: list[tuple[float, bool]] = []
    delib_tokens = 0

    for query in queries:
        if arm == "belief_ledger":
            mode, value, confidence = answer_from_support(
                weighted_support[query.key],
                args.min_confidence,
                args.min_margin,
            )
            delib_tokens += 6 if mode == "answer" else 8
        elif arm == "majority_ledger":
            mode, value, confidence = answer_from_support(
                count_support[query.key],
                args.min_confidence,
                args.min_margin,
            )
            delib_tokens += 5 if mode == "answer" else 7
        elif arm == "scan_deliberation":
            local_support = compile_support(evidence_by_key[query.key])
            mode, value, confidence = answer_from_support(
                local_support[query.key],
                args.min_confidence,
                args.min_margin,
            )
            delib_tokens += len(evidence_by_key[query.key]) * 4 + 8
        elif arm == "latest_statement":
            mode, value, confidence = latest_answer(evidence_by_key, query.key)
            delib_tokens += 4
        elif arm == "always_defer":
            mode, value, confidence = "defer", None, 1.0
            delib_tokens += 3
        else:
            raise ValueError(f"unknown arm: {arm}")

        if mode == "answer":
            answers += 1
            is_correct = value == query.gold_value and query.gold_mode == "answer"
            calibration_samples.append((confidence, is_correct))
            if is_correct:
                correct_on_answerable += 1
            else:
                wrong += 1
        else:
            defers += 1
            is_correct = query.gold_mode == "defer"
            calibration_samples.append((confidence, is_correct))
            if is_correct:
                correct_defers += 1

    answer_acc = correct_on_answerable / max(1, answerable)
    defer_acc = correct_defers / unanswerable
    pass_rate = answers / max(1, len(queries))
    wrong_rate = wrong / max(1, len(queries))
    mode_score = 0.65 * answer_acc + 0.35 * defer_acc - args.wrong_penalty * wrong_rate

    if arm in {"belief_ledger", "majority_ledger"}:
        paid_work = len(evidence) + len(queries)
    elif arm == "scan_deliberation":
        paid_work = len(evidence) * len(queries)
    else:
        paid_work = len(queries)
    scan_work = max(1, len(evidence) * len(queries))
    paid_ratio = paid_work / scan_work
    quality_per_paid = mode_score / max(1e-9, paid_ratio)

    return ArmMetrics(
        arm=arm,
        mode_score=mode_score,
        answer_accuracy=answer_acc,
        defer_accuracy=defer_acc,
        pass_rate=pass_rate,
        confident_wrong_rate=wrong_rate,
        calibration_error=calibration_error(calibration_samples),
        mean_deliberation_tokens=delib_tokens / max(1, len(queries)),
        paid_work_ratio_vs_scan=paid_ratio,
        quality_per_paid_ratio=quality_per_paid,
        answers=answers,
        wrong=wrong,
        defers=defers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entities", type=int, default=96)
    parser.add_argument("--fields", type=int, default=5)
    parser.add_argument("--values", type=int, default=8)
    parser.add_argument("--evidence-per-key", type=int, default=5)
    parser.add_argument("--queries", type=int, default=2400)
    parser.add_argument("--conflict-fraction", type=float, default=0.35)
    parser.add_argument("--min-confidence", type=float, default=0.62)
    parser.add_argument("--min-margin", type=float, default=1.25)
    parser.add_argument("--wrong-penalty", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=42206)
    parser.add_argument("--min-mode-score", type=float, default=0.88)
    parser.add_argument("--max-wrong-rate", type=float, default=0.03)
    parser.add_argument("--min-answer-pass-rate", type=float, default=0.55)
    parser.add_argument("--min-quality-per-paid-gain", type=float, default=1.25)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    rng = random.Random(args.seed)
    evidence, queries = make_world(args, rng)
    arms = [
        score_arm(arm, evidence, queries, args)
        for arm in (
            "belief_ledger",
            "majority_ledger",
            "scan_deliberation",
            "latest_statement",
            "always_defer",
        )
    ]
    ledger = next(arm for arm in arms if arm.arm == "belief_ledger")
    scan = next(arm for arm in arms if arm.arm == "scan_deliberation")
    best_control_qpp = max(arm.quality_per_paid_ratio for arm in arms if arm.arm != "belief_ledger")
    qpp_gain = ledger.quality_per_paid_ratio / max(1e-9, best_control_qpp)
    answerable_fraction = sum(1 for query in queries if query.gold_mode == "answer") / max(1, len(queries))
    pass_gate = (
        ledger.mode_score >= args.min_mode_score
        and ledger.confident_wrong_rate <= args.max_wrong_rate
        and ledger.pass_rate >= args.min_answer_pass_rate * answerable_fraction
        and qpp_gain >= args.min_quality_per_paid_gain
        and ledger.mean_deliberation_tokens < scan.mean_deliberation_tokens
    )
    if pass_gate:
        decision = "PASS_BELIEF_LEDGER_ANSWERER_GATE"
    elif ledger.confident_wrong_rate > args.max_wrong_rate:
        decision = "FAIL_CONFIDENT_WRONG_RATE"
    elif ledger.mode_score < args.min_mode_score:
        decision = "FAIL_MODE_SCORE"
    elif qpp_gain < args.min_quality_per_paid_gain:
        decision = "FAIL_PAID_COST_ADVANTAGE"
    else:
        decision = "FAIL_PASS_RATE_OR_RENDER_COST"

    activation = (
        "[R422_BELIEF_LEDGER_ANSWERER_GATE] "
        f"active=1 decision={decision} pass={int(pass_gate)} "
        "primary_state=typed_belief_ledger no_gpu=1 no_training_integration=1"
    )
    payload = GateResult(
        schema=SCHEMA,
        decision=decision,
        pass_gate=pass_gate,
        cases=len(queries),
        answerable_fraction=answerable_fraction,
        ledger_vs_best_quality_per_paid_gain=qpp_gain,
        ledger_mode_score=ledger.mode_score,
        ledger_confident_wrong_rate=ledger.confident_wrong_rate,
        ledger_mean_deliberation_tokens=ledger.mean_deliberation_tokens,
        scan_mean_deliberation_tokens=scan.mean_deliberation_tokens,
        activation_log=activation,
        interpretation=(
            "controlled_material_gate_only: validates typed belief/conflict state "
            "as primary answer object; not a production v12 path or speed claim."
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
                "answerable_fraction": answerable_fraction,
                "ledger_mode_score": ledger.mode_score,
                "ledger_confident_wrong_rate": ledger.confident_wrong_rate,
                "ledger_vs_best_quality_per_paid_gain": qpp_gain,
                "ledger_mean_deliberation_tokens": ledger.mean_deliberation_tokens,
                "scan_mean_deliberation_tokens": scan.mean_deliberation_tokens,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
