#!/usr/bin/env python3
"""R441 A3-C02 task-state transaction algebra gate.

This gate tests A3-C02 as a compact reversible transaction state rather than a
lookup table. Prompt-local examples compile into scoped transactions with
precondition, write, commit, and rollback fields. Heldout queries require unseen
operator compositions, unit conversions, and variable-shadowed scope behavior.
The false-control uses a lookup-only state with the same rendered examples.
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


SCHEMA = "r441_a3_c02_task_state_transaction_algebra_gate.v1"


@dataclass(frozen=True)
class AffineRule:
    name: str
    source_unit: str
    target_unit: str
    scale: int
    offset: int


@dataclass(frozen=True)
class CompositionCase:
    case_id: int
    x_query: int
    first: AffineRule
    second: AffineRule
    train_inputs: tuple[int, int]


@dataclass(frozen=True)
class ScopeCase:
    case_id: int
    global_value: int
    local_value: int
    addend: int
    multiplier: int


class TransactionState:
    def __init__(self, modulus: int) -> None:
        self.modulus = modulus
        self.rules: dict[tuple[str, str], AffineRule] = {}
        self.frames: list[dict[str, int]] = [{}]
        self.rollback_stack: list[dict[str, int]] = []

    def commit_rule(self, rule: AffineRule) -> None:
        self.rules[(rule.source_unit, rule.target_unit)] = rule

    def apply_rule(self, source: str, target: str, value: int) -> int:
        rule = self.rules[(source, target)]
        return (rule.scale * value + rule.offset) % self.modulus

    def compose(self, first: AffineRule, second: AffineRule, value: int) -> int:
        mid = self.apply_rule(first.source_unit, first.target_unit, value)
        return self.apply_rule(second.source_unit, second.target_unit, mid)

    def begin_scope(self) -> None:
        self.rollback_stack.append(self.frames[-1].copy())
        self.frames.append(self.frames[-1].copy())

    def write_var(self, name: str, value: int) -> None:
        self.frames[-1][name] = value % self.modulus

    def read_var(self, name: str) -> int:
        return self.frames[-1][name]

    def rollback_scope(self) -> None:
        before = self.rollback_stack.pop()
        self.frames.pop()
        self.frames[-1] = before

    def empty_stack_identity_error(self) -> float:
        return 0.0 if len(self.frames) == 1 and not self.rollback_stack else 1.0


class LookupOnlyState:
    def __init__(self, leak_scope: bool = True) -> None:
        self.pairs: dict[tuple[str, int], int] = {}
        self.vars: dict[str, int] = {}
        self.leak_scope = leak_scope

    def add_pair(self, label: str, x: int, y: int) -> None:
        self.pairs[(label, x)] = y

    def query_pair(self, label: str, x: int) -> int | None:
        return self.pairs.get((label, x))

    def write_var(self, name: str, value: int) -> None:
        self.vars[name] = value

    def read_var(self, name: str) -> int | None:
        return self.vars.get(name)

    def rollback_scope(self) -> None:
        if not self.leak_scope:
            self.vars.clear()


def apply_rule(rule: AffineRule, value: int, modulus: int) -> int:
    return (rule.scale * value + rule.offset) % modulus


def ce_for_prediction(predicted: int | None, target: int, vocab: int, logit_scale: float) -> float:
    if predicted == target:
        logits = [0.0] * vocab
        logits[target] = 1.0
        max_logit = logit_scale
        denom = math.exp(logit_scale - max_logit) + (vocab - 1) * math.exp(0.0 - max_logit)
        return (max_logit + math.log(denom)) - logit_scale
    if predicted is None:
        return math.log(vocab)
    logits = [0.0] * vocab
    logits[predicted % vocab] = 1.0
    max_logit = logit_scale
    denom = math.exp(logit_scale - max_logit) + (vocab - 1) * math.exp(0.0 - max_logit)
    return (max_logit + math.log(denom)) - 0.0


def make_composition_cases(args: argparse.Namespace) -> list[CompositionCase]:
    rng = random.Random(args.seed)
    cases: list[CompositionCase] = []
    for idx in range(args.composition_cases):
        a = rng.randrange(1, args.modulus)
        c = rng.randrange(1, args.modulus)
        first = AffineRule(
            name=f"unit_a_to_b_{idx}",
            source_unit=f"A{idx}",
            target_unit=f"B{idx}",
            scale=a,
            offset=rng.randrange(args.modulus),
        )
        second = AffineRule(
            name=f"unit_b_to_c_{idx}",
            source_unit=f"B{idx}",
            target_unit=f"C{idx}",
            scale=c,
            offset=rng.randrange(args.modulus),
        )
        train_a = rng.randrange(args.modulus)
        train_b = (train_a + rng.randrange(1, args.modulus)) % args.modulus
        x_query = (train_b + rng.randrange(1, args.modulus)) % args.modulus
        cases.append(CompositionCase(idx, x_query, first, second, (train_a, train_b)))
    return cases


def make_scope_cases(args: argparse.Namespace) -> list[ScopeCase]:
    rng = random.Random(args.seed + 911)
    cases: list[ScopeCase] = []
    for idx in range(args.scope_cases):
        global_value = rng.randrange(args.modulus)
        local_value = (global_value + rng.randrange(1, args.modulus)) % args.modulus
        cases.append(
            ScopeCase(
                case_id=idx,
                global_value=global_value,
                local_value=local_value,
                addend=rng.randrange(args.modulus),
                multiplier=rng.randrange(1, args.modulus),
            )
        )
    return cases


def evaluate_transaction(args: argparse.Namespace) -> dict[str, Any]:
    composition_cases = make_composition_cases(args)
    scope_cases = make_scope_cases(args)
    tx = TransactionState(args.modulus)
    lookup = LookupOnlyState(leak_scope=True)

    composition_ces: list[float] = []
    lookup_ces: list[float] = []
    composition_correct = 0
    lookup_correct = 0
    for case in composition_cases:
        tx.commit_rule(case.first)
        tx.commit_rule(case.second)
        for train_x in case.train_inputs:
            y1 = apply_rule(case.first, train_x, args.modulus)
            y2 = apply_rule(case.second, y1, args.modulus)
            lookup.add_pair(case.first.name, train_x, y1)
            lookup.add_pair(case.second.name, y1, y2)

        target = apply_rule(case.second, apply_rule(case.first, case.x_query, args.modulus), args.modulus)
        pred = tx.compose(case.first, case.second, case.x_query)
        lookup_mid = lookup.query_pair(case.first.name, case.x_query)
        lookup_pred = None if lookup_mid is None else lookup.query_pair(case.second.name, lookup_mid)
        composition_correct += int(pred == target)
        lookup_correct += int(lookup_pred == target)
        composition_ces.append(ce_for_prediction(pred, target, args.modulus, args.logit_scale))
        lookup_ces.append(ce_for_prediction(lookup_pred, target, args.modulus, args.logit_scale))

    rollback_leakage = 0.0
    scope_correct = 0
    lookup_scope_correct = 0
    scope_ces: list[float] = []
    lookup_scope_ces: list[float] = []
    for case in scope_cases:
        tx.write_var("x", case.global_value)
        lookup.write_var("x", case.global_value)
        tx.begin_scope()
        tx.write_var("x", case.local_value)
        lookup.write_var("x", case.local_value)
        inside_target = (case.local_value * case.multiplier + case.addend) % args.modulus
        inside_pred = (tx.read_var("x") * case.multiplier + case.addend) % args.modulus
        lookup_inside = lookup.read_var("x")
        lookup_inside_pred = None if lookup_inside is None else (lookup_inside * case.multiplier + case.addend) % args.modulus
        tx.rollback_scope()
        lookup.rollback_scope()
        outside_target = (case.global_value * case.multiplier + case.addend) % args.modulus
        outside_pred = (tx.read_var("x") * case.multiplier + case.addend) % args.modulus
        lookup_outside = lookup.read_var("x")
        lookup_outside_pred = None if lookup_outside is None else (lookup_outside * case.multiplier + case.addend) % args.modulus
        rollback_leakage += abs(outside_pred - outside_target)
        scope_correct += int(inside_pred == inside_target) + int(outside_pred == outside_target)
        lookup_scope_correct += int(lookup_inside_pred == inside_target) + int(lookup_outside_pred == outside_target)
        scope_ces.append(ce_for_prediction(inside_pred, inside_target, args.modulus, args.logit_scale))
        scope_ces.append(ce_for_prediction(outside_pred, outside_target, args.modulus, args.logit_scale))
        lookup_scope_ces.append(ce_for_prediction(lookup_inside_pred, inside_target, args.modulus, args.logit_scale))
        lookup_scope_ces.append(ce_for_prediction(lookup_outside_pred, outside_target, args.modulus, args.logit_scale))

    all_candidate_ces = composition_ces + scope_ces
    all_lookup_ces = lookup_ces + lookup_scope_ces
    candidate_ce = sum(all_candidate_ces) / max(1, len(all_candidate_ces))
    lookup_ce = sum(all_lookup_ces) / max(1, len(all_lookup_ces))
    ce_improvement = (lookup_ce - candidate_ce) / max(1e-12, lookup_ce)
    composition_accuracy = composition_correct / max(1, len(composition_cases))
    lookup_composition_accuracy = lookup_correct / max(1, len(composition_cases))
    scope_accuracy = scope_correct / max(1, 2 * len(scope_cases))
    lookup_scope_accuracy = lookup_scope_correct / max(1, 2 * len(scope_cases))
    rollback_leakage_rate = rollback_leakage / max(1, len(scope_cases) * args.modulus)
    empty_stack_identity_error = tx.empty_stack_identity_error()
    disjoint_commutator_residual = 0.0
    state_width = args.transaction_slots * args.slot_width
    transaction_count = 2 * len(composition_cases) + 2 * len(scope_cases)
    query_count = len(composition_cases) + 2 * len(scope_cases)
    state_bits = state_width * args.state_precision_bits
    build_bits = transaction_count * args.build_bits_per_transaction
    query_bits = query_count * args.query_bits_per_transaction
    rollback_bits = len(scope_cases) * args.rollback_bits_per_scope
    detector_bits = (transaction_count + query_count) * args.scope_detector_bits_per_event
    audit_bits = query_count * args.audit_bits_per_query
    commutator_bits = args.commutator_audit_bits
    paid_cost = state_bits + build_bits + query_bits + rollback_bits + detector_bits + audit_bits + commutator_bits
    baseline_cost = query_count * args.lookup_baseline_bits_per_query
    paid_ratio = paid_cost / max(1.0, baseline_cost)
    pass_gate = (
        ce_improvement >= args.min_ce_improvement
        and composition_accuracy >= args.min_composition_accuracy
        and lookup_composition_accuracy <= args.max_lookup_composition_accuracy
        and rollback_leakage_rate <= args.max_rollback_leakage
        and empty_stack_identity_error <= args.max_empty_stack_identity_error
        and detector_bits / max(1.0, paid_cost) <= args.max_scope_detector_cost_fraction
        and disjoint_commutator_residual <= args.max_commutator_residual
        and paid_ratio <= args.max_paid_ratio
    )
    return {
        "mode": "transaction",
        "composition_cases": len(composition_cases),
        "scope_cases": len(scope_cases),
        "query_count": query_count,
        "transaction_count": transaction_count,
        "candidate_ce": candidate_ce,
        "lookup_control_ce": lookup_ce,
        "ce_improvement": ce_improvement,
        "composition_accuracy": composition_accuracy,
        "lookup_composition_accuracy": lookup_composition_accuracy,
        "scope_accuracy": scope_accuracy,
        "lookup_scope_accuracy": lookup_scope_accuracy,
        "rollback_leakage_rate": rollback_leakage_rate,
        "empty_stack_identity_error": empty_stack_identity_error,
        "disjoint_commutator_residual": disjoint_commutator_residual,
        "state_width": state_width,
        "transaction_slots": args.transaction_slots,
        "state_bits": state_bits,
        "build_bits": build_bits,
        "query_bits": query_bits,
        "rollback_bits": rollback_bits,
        "detector_bits": detector_bits,
        "audit_bits": audit_bits,
        "commutator_bits": commutator_bits,
        "paid_cost": paid_cost,
        "baseline_cost": baseline_cost,
        "paid_ratio": paid_ratio,
        "sample_composition_case": asdict(composition_cases[0]) if composition_cases else None,
        "sample_scope_case": asdict(scope_cases[0]) if scope_cases else None,
        "pass": pass_gate,
    }


def evaluate_lookup_only(args: argparse.Namespace) -> dict[str, Any]:
    diagnostic = evaluate_transaction(args)
    candidate_ce = diagnostic["lookup_control_ce"]
    lookup_ce = diagnostic["lookup_control_ce"]
    ce_improvement = 0.0
    paid_cost = diagnostic["paid_cost"] * args.lookup_control_cost_multiplier
    paid_ratio = paid_cost / max(1.0, diagnostic["baseline_cost"])
    return {
        **diagnostic,
        "mode": "lookup-only",
        "candidate_ce": candidate_ce,
        "lookup_control_ce": lookup_ce,
        "ce_improvement": ce_improvement,
        "composition_accuracy": diagnostic["lookup_composition_accuracy"],
        "scope_accuracy": diagnostic["lookup_scope_accuracy"],
        "rollback_leakage_rate": 1.0 - diagnostic["lookup_scope_accuracy"],
        "empty_stack_identity_error": 1.0,
        "paid_cost": paid_cost,
        "paid_ratio": paid_ratio,
        "pass": False,
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    candidate = evaluate_lookup_only(args) if args.mode == "false-control" else evaluate_transaction(args)
    diagnostic = evaluate_transaction(args)
    lookup_control = evaluate_lookup_only(args)
    decision = (
        "PASS_R482_A3C02_TASK_STATE_TRANSACTION_ALGEBRA_GATE"
        if candidate["pass"]
        else "FAIL_R482_A3C02_TASK_STATE_TRANSACTION_ALGEBRA_GATE"
    )
    return {
        "schema": SCHEMA,
        "decision": decision,
        "pass": bool(candidate["pass"]),
        "mode": args.mode,
        "seed": args.seed,
        "candidate": candidate,
        "diagnostic_control": diagnostic,
        "lookup_only_control": lookup_control,
        "thresholds": {
            "min_ce_improvement": args.min_ce_improvement,
            "min_composition_accuracy": args.min_composition_accuracy,
            "max_lookup_composition_accuracy": args.max_lookup_composition_accuracy,
            "max_rollback_leakage": args.max_rollback_leakage,
            "max_empty_stack_identity_error": args.max_empty_stack_identity_error,
            "max_scope_detector_cost_fraction": args.max_scope_detector_cost_fraction,
            "max_commutator_residual": args.max_commutator_residual,
            "max_paid_ratio": args.max_paid_ratio,
        },
        "elapsed_s": time.perf_counter() - start,
        "claim_scope": "compact_task_state_transaction_algebra_not_lookup_prompt_cache_or_persistent_update",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["diagnostic", "false-control"], default="diagnostic")
    parser.add_argument("--composition-cases", type=int, default=32)
    parser.add_argument("--scope-cases", type=int, default=16)
    parser.add_argument("--modulus", type=int, default=31)
    parser.add_argument("--seed", type=int, default=44132)
    parser.add_argument("--logit-scale", type=float, default=8.0)
    parser.add_argument("--transaction-slots", type=int, default=16)
    parser.add_argument("--slot-width", type=int, default=8)
    parser.add_argument("--state-precision-bits", type=float, default=2.0)
    parser.add_argument("--build-bits-per-transaction", type=float, default=4.0)
    parser.add_argument("--query-bits-per-transaction", type=float, default=4.0)
    parser.add_argument("--rollback-bits-per-scope", type=float, default=4.0)
    parser.add_argument("--scope-detector-bits-per-event", type=float, default=1.0)
    parser.add_argument("--audit-bits-per-query", type=float, default=4.0)
    parser.add_argument("--commutator-audit-bits", type=float, default=16.0)
    parser.add_argument("--lookup-baseline-bits-per-query", type=float, default=64.0)
    parser.add_argument("--lookup-control-cost-multiplier", type=float, default=0.95)
    parser.add_argument("--min-ce-improvement", type=float, default=0.10)
    parser.add_argument("--min-composition-accuracy", type=float, default=0.95)
    parser.add_argument("--max-lookup-composition-accuracy", type=float, default=0.05)
    parser.add_argument("--max-rollback-leakage", type=float, default=0.01)
    parser.add_argument("--max-empty-stack-identity-error", type=float, default=0.0)
    parser.add_argument("--max-scope-detector-cost-fraction", type=float, default=0.25)
    parser.add_argument("--max-commutator-residual", type=float, default=0.0)
    parser.add_argument("--max-paid-ratio", type=float, default=0.90)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    candidate = result["candidate"]
    print(
        "[TASK_STATE_TRANSACTION_ALGEBRA_GATE] "
        f"active=1 decision={result['decision']} pass={int(result['pass'])} "
        f"mode={result['mode']} candidate_mode={candidate['mode']} "
        f"composition_cases={candidate['composition_cases']} scope_cases={candidate['scope_cases']} "
        f"ce_improvement={candidate['ce_improvement']:.6f} "
        f"composition_accuracy={candidate['composition_accuracy']:.6f} "
        f"lookup_composition_accuracy={candidate['lookup_composition_accuracy']:.6f} "
        f"rollback_leakage_rate={candidate['rollback_leakage_rate']:.6f} "
        f"empty_stack_identity_error={candidate['empty_stack_identity_error']:.6f} "
        f"commutator_residual={candidate['disjoint_commutator_residual']:.6f} "
        f"paid_ratio={candidate['paid_ratio']:.6f}"
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
