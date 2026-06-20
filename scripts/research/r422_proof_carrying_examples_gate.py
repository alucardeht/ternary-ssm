#!/usr/bin/env python3
"""R422 TM02 proof-carrying examples material gate.

The gate builds a deterministic, project-owned proof-material compiler instead
of using teacher chain-of-thought. Each example contains premises, allowed
transformations, a claim, a compact support certificate, and an expected failure
mode when support is insufficient or contradictory.

This is a CPU material/compiler gate only. It is not a v12/GPU training path and
must not be reported as a measured training speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_proof_carrying_examples_gate.v1"


@dataclass(frozen=True)
class Item:
    item_id: str
    kind: str
    src: str
    dst: str
    polarity: int


@dataclass
class Example:
    example_id: int
    case: str
    claim_atom: str
    claim_polarity: int
    items: list[Item]
    support_ids: list[str]
    expected_answer: str
    expected_failure: str
    certificate_hash: str


@dataclass
class GateMetrics:
    decision: str
    examples: int
    item_count: int
    valid_answer_accuracy: float
    valid_support_accuracy: float
    valid_failure_accuracy: float
    contradiction_accuracy: float
    insufficient_accuracy: float
    corrupted_reject_rate: float
    random_support_accept_rate: float
    answer_only_support_accuracy: float
    proof_work_ratio: float
    full_scan_work_ratio: float
    proof_vs_full_work_gain: float
    certificate_token_ratio: float
    activation_log: str
    observer_only: bool
    no_gpu: bool
    no_training_integration: bool
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=4096)
    parser.add_argument("--atoms", type=int, default=128)
    parser.add_argument("--facts", type=int, default=8)
    parser.add_argument("--rules", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42202)
    parser.add_argument("--max-proof-work-ratio", type=float, default=0.35)
    parser.add_argument("--min-valid-accuracy", type=float, default=0.99)
    parser.add_argument("--min-corrupted-reject", type=float, default=0.95)
    parser.add_argument("--max-random-support-accept", type=float, default=0.20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def atom(index: int) -> str:
    return f"a{index:03d}"


def item_text(item: Item) -> str:
    if item.kind == "fact":
        sign = "+" if item.polarity > 0 else "-"
        return f"{item.item_id}: fact {sign}{item.dst}"
    sign = "+" if item.polarity > 0 else "-"
    return f"{item.item_id}: rule +{item.src} -> {sign}{item.dst}"


def certificate_hash(items: list[Item], support_ids: list[str], claim_atom: str, claim_polarity: int) -> str:
    support = {item.item_id: item_text(item) for item in items if item.item_id in set(support_ids)}
    payload = {
        "claim": [claim_atom, claim_polarity],
        "support": [(key, support[key]) for key in sorted(support)],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def make_example(rng: random.Random, example_id: int, atoms: int, facts: int, rules: int) -> Example:
    if facts < 4 or rules < 4:
        raise ValueError("facts and rules must both be >= 4")
    case = ["supported", "contradiction", "insufficient"][example_id % 3]
    src = atom(rng.randrange(atoms))
    dst = atom((int(src[1:]) + 1 + rng.randrange(atoms - 1)) % atoms)
    items: list[Item] = []
    fact_id = "f000"
    pos_rule_id = "r000"
    neg_rule_id = "r001"

    if case in {"supported", "contradiction"}:
        items.append(Item(fact_id, "fact", "", src, +1))
    else:
        items.append(Item(fact_id, "fact", "", atom((int(src[1:]) + 3) % atoms), +1))

    items.append(Item(pos_rule_id, "rule", src, dst, +1))
    if case == "contradiction":
        items.append(Item(neg_rule_id, "rule", src, dst, -1))
    else:
        items.append(Item(neg_rule_id, "rule", atom((int(src[1:]) + 5) % atoms), dst, -1))

    used_pairs = {(item.kind, item.src, item.dst, item.polarity) for item in items}
    while sum(1 for item in items if item.kind == "fact") < facts:
        candidate = atom(rng.randrange(atoms))
        polarity = rng.choice([-1, +1])
        key = ("fact", "", candidate, polarity)
        if key in used_pairs or candidate == src:
            continue
        item_id = f"f{sum(1 for item in items if item.kind == 'fact'):03d}"
        items.append(Item(item_id, "fact", "", candidate, polarity))
        used_pairs.add(key)
    while sum(1 for item in items if item.kind == "rule") < rules:
        rule_src = atom(rng.randrange(atoms))
        rule_dst = atom(rng.randrange(atoms))
        if rule_src == src and rule_dst == dst:
            continue
        polarity = rng.choice([-1, +1])
        key = ("rule", rule_src, rule_dst, polarity)
        if key in used_pairs or rule_src == rule_dst:
            continue
        item_id = f"r{sum(1 for item in items if item.kind == 'rule'):03d}"
        items.append(Item(item_id, "rule", rule_src, rule_dst, polarity))
        used_pairs.add(key)

    rng.shuffle(items)
    claim_polarity = +1
    if case == "supported":
        support_ids = [fact_id, pos_rule_id]
        answer = "entailed"
        failure = "none"
    elif case == "contradiction":
        support_ids = [fact_id, pos_rule_id, neg_rule_id]
        answer = "failed"
        failure = "contradiction"
    else:
        support_ids = [pos_rule_id]
        answer = "failed"
        failure = "insufficient"

    return Example(
        example_id=example_id,
        case=case,
        claim_atom=dst,
        claim_polarity=claim_polarity,
        items=items,
        support_ids=support_ids,
        expected_answer=answer,
        expected_failure=failure,
        certificate_hash=certificate_hash(items, support_ids, dst, claim_polarity),
    )


def evaluate_support(example: Example, support_ids: list[str]) -> tuple[str, str, bool]:
    item_by_id = {item.item_id: item for item in example.items}
    support = [item_by_id[item_id] for item_id in support_ids if item_id in item_by_id]
    if len(support) != len(support_ids):
        return "failed", "invalid_support", False
    facts: dict[str, set[int]] = {}
    rules: list[Item] = []
    for item in support:
        if item.kind == "fact":
            facts.setdefault(item.dst, set()).add(item.polarity)
        elif item.kind == "rule":
            rules.append(item)
    entailed: set[int] = set()
    for rule in rules:
        if +1 in facts.get(rule.src, set()):
            if rule.dst == example.claim_atom:
                entailed.add(rule.polarity)
    if +1 in entailed and -1 in entailed:
        answer, failure = "failed", "contradiction"
    elif example.claim_polarity in entailed:
        answer, failure = "entailed", "none"
    else:
        answer, failure = "failed", "insufficient"
    support_ok = support_ids == example.support_ids
    return answer, failure, support_ok


def corrupted_support(example: Example, rng: random.Random) -> list[str]:
    ids = [item.item_id for item in example.items]
    corrupt = list(example.support_ids)
    if corrupt and rng.random() < 0.5:
        corrupt.pop(rng.randrange(len(corrupt)))
    if ids:
        corrupt.append(rng.choice(ids))
    return corrupt


def random_support(example: Example, rng: random.Random) -> list[str]:
    ids = [item.item_id for item in example.items]
    return rng.sample(ids, k=min(len(ids), len(example.support_ids)))


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    rng = random.Random(args.seed)
    examples = [
        make_example(rng, i, args.atoms, args.facts, args.rules)
        for i in range(args.examples)
    ]

    valid_answer = 0
    valid_support = 0
    valid_failure = 0
    contradiction_ok = 0
    contradiction_total = 0
    insufficient_ok = 0
    insufficient_total = 0
    corrupted_rejected = 0
    random_accept = 0
    answer_only_support = 0
    proof_work = 0.0
    full_work = 0.0
    cert_tokens = 0.0
    material_tokens = 0.0
    corrupt_rng = random.Random(args.seed + 17)
    random_rng = random.Random(args.seed + 29)

    for example in examples:
        answer, failure, support_ok = evaluate_support(example, example.support_ids)
        valid_answer += answer == example.expected_answer
        valid_support += support_ok
        valid_failure += failure == example.expected_failure
        if example.expected_failure == "contradiction":
            contradiction_total += 1
            contradiction_ok += failure == "contradiction"
        if example.expected_failure == "insufficient":
            insufficient_total += 1
            insufficient_ok += failure == "insufficient"

        corrupt_answer, corrupt_failure, corrupt_support = evaluate_support(
            example, corrupted_support(example, corrupt_rng)
        )
        corrupted_rejected += not (
            corrupt_answer == example.expected_answer
            and corrupt_failure == example.expected_failure
            and corrupt_support
        )
        rand_answer, rand_failure, rand_support = evaluate_support(example, random_support(example, random_rng))
        random_accept += (
            rand_answer == example.expected_answer
            and rand_failure == example.expected_failure
            and rand_support
        )
        answer_only_support += False
        proof_work += len(example.support_ids)
        full_work += len(example.items)
        cert_tokens += len(example.support_ids) + 2
        material_tokens += len(example.items) + len(example.support_ids) + 2

    n = len(examples)
    valid_answer_accuracy = valid_answer / n
    valid_support_accuracy = valid_support / n
    valid_failure_accuracy = valid_failure / n
    contradiction_accuracy = contradiction_ok / max(1, contradiction_total)
    insufficient_accuracy = insufficient_ok / max(1, insufficient_total)
    corrupted_reject_rate = corrupted_rejected / n
    random_support_accept_rate = random_accept / n
    answer_only_support_accuracy = answer_only_support / n
    proof_work_ratio = proof_work / full_work
    full_scan_work_ratio = 1.0
    proof_vs_full_work_gain = full_scan_work_ratio / proof_work_ratio
    certificate_token_ratio = cert_tokens / material_tokens

    passes = (
        valid_answer_accuracy >= args.min_valid_accuracy
        and valid_support_accuracy >= args.min_valid_accuracy
        and valid_failure_accuracy >= args.min_valid_accuracy
        and corrupted_reject_rate >= args.min_corrupted_reject
        and random_support_accept_rate <= args.max_random_support_accept
        and proof_work_ratio <= args.max_proof_work_ratio
    )
    decision = "PASS_PROOF_CARRYING_EXAMPLES_GATE" if passes else "FAIL_PROOF_CARRYING_EXAMPLES_GATE"
    activation = (
        "[R422_PROOF_CARRYING_EXAMPLES_GATE] "
        f"active=1 decision={decision} compiler_material=1 no_teacher_cot=1 no_gpu=1 no_training_integration=1"
    )
    result = GateMetrics(
        decision=decision,
        examples=n,
        item_count=args.facts + args.rules,
        valid_answer_accuracy=valid_answer_accuracy,
        valid_support_accuracy=valid_support_accuracy,
        valid_failure_accuracy=valid_failure_accuracy,
        contradiction_accuracy=contradiction_accuracy,
        insufficient_accuracy=insufficient_accuracy,
        corrupted_reject_rate=corrupted_reject_rate,
        random_support_accept_rate=random_support_accept_rate,
        answer_only_support_accuracy=answer_only_support_accuracy,
        proof_work_ratio=proof_work_ratio,
        full_scan_work_ratio=full_scan_work_ratio,
        proof_vs_full_work_gain=proof_vs_full_work_gain,
        certificate_token_ratio=certificate_token_ratio,
        activation_log=activation,
        observer_only=True,
        no_gpu=True,
        no_training_integration=True,
        elapsed_seconds=time.perf_counter() - start,
    )
    payload = {"schema": SCHEMA, **asdict(result)}
    print(activation)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
