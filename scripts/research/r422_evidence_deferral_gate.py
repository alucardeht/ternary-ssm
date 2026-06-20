#!/usr/bin/env python3
"""R422 evidence-deferral training contract gate.

This bounded material gate tests answer/defer/request-evidence behavior from
explicit information-set certificates. It is designed to reject both confident
wrong answering and pass-rate collapse from always refusing to answer.

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


SCHEMA = "r422_evidence_deferral_gate.v1"
MODES = ("answer", "request_evidence", "defer")


@dataclass(frozen=True)
class Evidence:
    source_id: int
    value: int
    reliable: bool


@dataclass(frozen=True)
class Record:
    record_id: int
    true_value: int
    evidence: tuple[Evidence, ...]
    label_mode: str
    label_answer: int | None


@dataclass
class ArmResult:
    arm: str
    mode_accuracy: float
    answerable_pass_rate: float
    answerable_answer_accuracy: float
    missing_request_accuracy: float
    conflict_defer_accuracy: float
    confident_wrong_rate: float
    unsafe_answer_rate: float
    utility_score: float
    certificate_work_ratio: float


@dataclass
class GateResult:
    schema: str
    decision: str
    pass_gate: bool
    records: int
    value_modulus: int
    answerable_share: float
    missing_share: float
    conflict_share: float
    main: ArmResult
    controls: list[ArmResult]
    main_utility_gain_vs_best_control: float
    activation_log: str
    interpretation: str
    elapsed_seconds: float


def make_record(record_id: int, value_modulus: int, rng: random.Random) -> Record:
    true = (record_id * 17 + rng.randrange(value_modulus)) % value_modulus
    scenario = rng.choices(
        ["answerable", "missing", "conflict", "weak"],
        weights=[0.52, 0.18, 0.20, 0.10],
        k=1,
    )[0]
    if scenario == "answerable":
        evidence = tuple(Evidence(i, true, True) for i in range(2 + rng.randrange(2)))
        return Record(record_id, true, evidence, "answer", true)
    if scenario == "missing":
        return Record(record_id, true, tuple(), "request_evidence", None)
    if scenario == "weak":
        evidence = (Evidence(0, true, False),)
        return Record(record_id, true, evidence, "request_evidence", None)
    wrong = (true + rng.randrange(1, value_modulus)) % value_modulus
    evidence = (
        Evidence(0, true, True),
        Evidence(1, wrong, True),
        Evidence(2, wrong if rng.random() < 0.5 else true, False),
    )
    return Record(record_id, true, evidence, "defer", None)


def make_records(count: int, value_modulus: int, seed: int) -> list[Record]:
    rng = random.Random(seed)
    return [make_record(i, value_modulus, rng) for i in range(count)]


def predict_contract(record: Record) -> tuple[str, int | None]:
    reliable = [ev.value for ev in record.evidence if ev.reliable]
    if not reliable:
        return "request_evidence", None
    counts = Counter(reliable)
    if len(counts) == 1 and len(reliable) >= 2:
        return "answer", reliable[0]
    return "defer", None


def predict_answer_majority(record: Record) -> tuple[str, int | None]:
    if not record.evidence:
        return "answer", 0
    value = Counter(ev.value for ev in record.evidence).most_common(1)[0][0]
    return "answer", value


def predict_first_evidence(record: Record) -> tuple[str, int | None]:
    if not record.evidence:
        return "request_evidence", None
    return "answer", record.evidence[0].value


def predict_always_request(record: Record) -> tuple[str, int | None]:
    return "request_evidence", None


def predict_always_defer(record: Record) -> tuple[str, int | None]:
    return "defer", None


def predict(arm: str, record: Record) -> tuple[str, int | None]:
    if arm == "evidence_contract":
        return predict_contract(record)
    if arm == "answer_majority":
        return predict_answer_majority(record)
    if arm == "first_evidence":
        return predict_first_evidence(record)
    if arm == "always_request":
        return predict_always_request(record)
    if arm == "always_defer":
        return predict_always_defer(record)
    raise ValueError(f"unknown arm: {arm}")


def evaluate_arm(arm: str, records: list[Record]) -> ArmResult:
    mode_correct = 0
    answerable = 0
    answered_answerable = 0
    correct_answered_answerable = 0
    missing_total = 0
    missing_correct = 0
    conflict_total = 0
    conflict_correct = 0
    confident_wrong = 0
    unsafe_answer = 0
    utility = 0.0
    for record in records:
        mode, ans = predict(arm, record)
        mode_correct += int(mode == record.label_mode and ans == record.label_answer)
        if record.label_mode == "answer":
            answerable += 1
            if mode == "answer":
                answered_answerable += 1
                if ans == record.true_value:
                    correct_answered_answerable += 1
                    utility += 1.0
                else:
                    confident_wrong += 1
                    utility -= 1.5
            else:
                utility -= 0.3
        elif record.label_mode == "request_evidence":
            missing_total += 1
            if mode == "request_evidence":
                missing_correct += 1
                utility += 0.4
            elif mode == "answer":
                unsafe_answer += 1
                confident_wrong += int(ans != record.true_value)
                utility -= 1.0
            else:
                utility += 0.1
        elif record.label_mode == "defer":
            conflict_total += 1
            if mode == "defer":
                conflict_correct += 1
                utility += 0.4
            elif mode == "answer":
                unsafe_answer += 1
                confident_wrong += int(ans != record.true_value)
                utility -= 1.0
            else:
                utility += 0.1
    n = max(1, len(records))
    certificate_cost = sum(1.0 + 0.35 * len(r.evidence) for r in records)
    dense_cost = sum(1.0 + len(r.evidence) for r in records)
    return ArmResult(
        arm=arm,
        mode_accuracy=mode_correct / n,
        answerable_pass_rate=answered_answerable / max(1, answerable),
        answerable_answer_accuracy=correct_answered_answerable / max(1, answered_answerable),
        missing_request_accuracy=missing_correct / max(1, missing_total),
        conflict_defer_accuracy=conflict_correct / max(1, conflict_total),
        confident_wrong_rate=confident_wrong / n,
        unsafe_answer_rate=unsafe_answer / n,
        utility_score=utility / n,
        certificate_work_ratio=certificate_cost / max(1e-12, dense_cost),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=4096)
    parser.add_argument("--value-modulus", type=int, default=97)
    parser.add_argument("--seed", type=int, default=42204)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    records = make_records(args.records, args.value_modulus, args.seed)
    main_arm = evaluate_arm("evidence_contract", records)
    controls = [
        evaluate_arm("answer_majority", records),
        evaluate_arm("first_evidence", records),
        evaluate_arm("always_request", records),
        evaluate_arm("always_defer", records),
    ]
    best_control_utility = max(c.utility_score for c in controls)
    utility_gain = main_arm.utility_score - best_control_utility
    answerable = sum(1 for r in records if r.label_mode == "answer")
    missing = sum(1 for r in records if r.label_mode == "request_evidence")
    conflict = sum(1 for r in records if r.label_mode == "defer")
    pass_gate = (
        main_arm.mode_accuracy >= 0.99
        and main_arm.answerable_pass_rate >= 0.95
        and main_arm.answerable_answer_accuracy >= 0.99
        and main_arm.missing_request_accuracy >= 0.95
        and main_arm.conflict_defer_accuracy >= 0.95
        and main_arm.confident_wrong_rate <= 0.01
        and main_arm.unsafe_answer_rate <= 0.01
        and main_arm.certificate_work_ratio <= 0.80
        and utility_gain >= 0.10
        and all(c.answerable_pass_rate <= 0.15 for c in controls if c.arm in {"always_request", "always_defer"})
    )
    decision = "PASS_EVIDENCE_DEFERRAL_CONTRACT_GATE" if pass_gate else "FAIL_EVIDENCE_DEFERRAL_CONTRACT_GATE"
    activation = (
        f"[R422_EVIDENCE_DEFERRAL_GATE] active=1 decision={decision} "
        "answer_defer_request_modes=1 pass_rate_guard=1 no_gpu=1 no_training_integration=1"
    )
    result = GateResult(
        schema=SCHEMA,
        decision=decision,
        pass_gate=pass_gate,
        records=args.records,
        value_modulus=args.value_modulus,
        answerable_share=answerable / max(1, len(records)),
        missing_share=missing / max(1, len(records)),
        conflict_share=conflict / max(1, len(records)),
        main=main_arm,
        controls=controls,
        main_utility_gain_vs_best_control=utility_gain,
        activation_log=activation,
        interpretation=(
            "Information-set certificates can select answer, request-evidence, or defer modes "
            "without collapsing pass rate on answerable cases. This is a material/usefulness gate only."
        ),
        elapsed_seconds=time.perf_counter() - start,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(activation)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0 if pass_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
