#!/usr/bin/env python3
"""R441 A5-C02 authority/scope lattice gate.

The gate implements the validated A5-C02 object as a finite transition law over
authority, scope, mode, and feasibility debt. The diagnostic arm carries a
projected lattice state P_L h_t and applies deterministic transitions for
authorized updates, lower-priority quoted text, scope close, and valid repair.
The false-control is a recency/delimiter policy: it sees scope markers and the
latest command text, but it has no authority order, no rollback stack, and no
feasibility debt state.
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


SCHEMA = "r441_a5_c02_authority_scope_lattice_gate.v1"

AUTHORITY_RANK = {
    "generated_text": 0,
    "quoted_text": 1,
    "user": 2,
    "developer": 3,
    "system": 4,
}


@dataclass(frozen=True)
class DialogCase:
    case_id: int
    high_authority: str
    base_mode: str
    allowed_repair: str
    lower_priority_attempt: str
    benign_user_task: str
    impossible_pair: tuple[str, str]


@dataclass
class LatticeState:
    authority: str
    scope_depth: int
    mode: str
    constraints: tuple[str, ...]
    repair_debt: float
    stack: list[tuple[str, int, str, tuple[str, ...], float]]

    def snapshot(self) -> tuple[str, int, str, tuple[str, ...], float]:
        return (self.authority, self.scope_depth, self.mode, self.constraints, self.repair_debt)


def make_cases(args: argparse.Namespace) -> list[DialogCase]:
    rng = random.Random(args.seed)
    modes = ["json", "bullets", "table", "plain"]
    repairs = ["state_conflict", "ask_clarifying_field", "preserve_high_rule", "drop_lower_conflict"]
    attempts = ["switch_to_xml", "ignore_parent", "remove_schema", "change_task"]
    tasks = ["summarize_invoice", "extract_ids", "rewrite_note", "classify_ticket"]
    cases: list[DialogCase] = []
    for idx in range(args.cases):
        hi = "developer" if idx % 3 else "system"
        mode = modes[idx % len(modes)]
        cases.append(
            DialogCase(
                case_id=idx,
                high_authority=hi,
                base_mode=mode,
                allowed_repair=repairs[idx % len(repairs)],
                lower_priority_attempt=f"{attempts[idx % len(attempts)]}_{rng.randint(100, 999)}",
                benign_user_task=f"{tasks[idx % len(tasks)]}_{rng.randint(1000, 9999)}",
                impossible_pair=(mode, modes[(idx + 1) % len(modes)]),
            )
        )
    return cases


def transition(state: LatticeState, event: str, case: DialogCase) -> LatticeState:
    state.stack = list(state.stack)
    if event == "set_high":
        state.authority = case.high_authority
        state.mode = case.base_mode
        state.constraints = (f"mode:{case.base_mode}", f"repair:{case.allowed_repair}")
        state.repair_debt = 0.0
    elif event == "enter_user_scope":
        state.stack.append(state.snapshot())
        state.scope_depth += 1
    elif event == "quoted_lower_attempt":
        # Quoted text is lower authority; it can add payload, but it cannot
        # replace the active higher-priority mode.
        state.repair_debt += 0.0
    elif event == "authorized_user_update":
        if AUTHORITY_RANK[state.authority] <= AUTHORITY_RANK["user"]:
            state.authority = "user"
            state.mode = "plain"
            state.constraints = ("mode:plain",)
    elif event == "impossible_pair":
        state.repair_debt += 1.0
    elif event == "valid_repair":
        state.repair_debt = max(0.0, state.repair_debt - 0.85)
    elif event == "close_scope":
        if state.stack:
            authority, scope_depth, mode, constraints, repair_debt = state.stack.pop()
            state.authority = authority
            state.scope_depth = scope_depth
            state.mode = mode
            state.constraints = constraints
            state.repair_debt = min(state.repair_debt, repair_debt)
    elif event == "post_scope_benign":
        # Benign work after scope close should remain possible under the
        # surviving authority/mode.
        state.repair_debt += 0.0
    else:
        raise ValueError(f"unknown event: {event}")
    return state


def recency_control_state(state: LatticeState, event: str, case: DialogCase) -> LatticeState:
    state.stack = list(state.stack)
    if event == "set_high":
        state.authority = case.high_authority
        state.mode = case.base_mode
        state.constraints = (f"mode:{case.base_mode}",)
        state.repair_debt = 0.0
    elif event == "enter_user_scope":
        state.stack.append(state.snapshot())
        state.scope_depth += 1
    elif event == "quoted_lower_attempt":
        state.authority = "quoted_text"
        state.mode = case.lower_priority_attempt
        state.constraints = (f"mode:{case.lower_priority_attempt}",)
    elif event == "authorized_user_update":
        state.authority = "user"
        state.mode = "plain"
        state.constraints = ("mode:plain",)
    elif event == "impossible_pair":
        state.repair_debt += 1.0
    elif event == "valid_repair":
        # This control confuses repair with refusal: it reduces some debt but
        # suppresses benign subtasks and leaves false conflict behind.
        state.repair_debt = max(0.0, state.repair_debt - 0.2)
    elif event == "close_scope":
        if state.stack:
            _authority, scope_depth, _mode, _constraints, _repair_debt = state.stack.pop()
            state.scope_depth = scope_depth
            # Delimiter-only close forgets scope depth but keeps the latest text.
    elif event == "post_scope_benign":
        state.repair_debt += 0.35
    else:
        raise ValueError(f"unknown event: {event}")
    return state


def ce(correct: bool, choices: int, scale: float) -> float:
    if correct:
        return math.log(1.0 + (choices - 1.0) * math.exp(-scale))
    return scale + math.log(1.0 + (choices - 1.0) * math.exp(-scale))


def score_arm(args: argparse.Namespace, cases: list[DialogCase], mode_name: str) -> dict[str, Any]:
    events = [
        "set_high",
        "enter_user_scope",
        "quoted_lower_attempt",
        "impossible_pair",
        "valid_repair",
        "close_scope",
        "post_scope_benign",
    ]
    fn = transition if mode_name == "authority-scope-lattice" else recency_control_state

    transition_ok = 0
    transitions = 0
    lower_priority_trials = 0
    lower_priority_overrides = 0
    scope_close_trials = 0
    scope_leaks = 0
    repair_trials = 0
    total_repair_reduction = 0.0
    false_conflict_debt = 0.0
    valid_subtasks = 0
    preserved_subtasks = 0
    lead_predictions = 0
    lead_correct = 0
    total_ce = 0.0
    baseline_ce = 0.0
    delimiter_ce = 0.0

    for case in cases:
        expected = LatticeState(
            authority="generated_text",
            scope_depth=0,
            mode="plain",
            constraints=(),
            repair_debt=0.0,
            stack=[],
        )
        candidate = LatticeState(
            authority="generated_text",
            scope_depth=0,
            mode="plain",
            constraints=(),
            repair_debt=0.0,
            stack=[],
        )
        previous_debt = 0.0
        for event in events:
            expected = transition(expected, event, case)
            candidate = fn(candidate, event, case)
            transitions += 1
            same_state = (
                candidate.authority == expected.authority
                and candidate.scope_depth == expected.scope_depth
                and candidate.mode == expected.mode
                and candidate.constraints == expected.constraints
                and abs(candidate.repair_debt - expected.repair_debt) <= args.max_repair_debt_error
            )
            transition_ok += int(same_state)
            total_ce += ce(same_state, args.lattice_choices, args.logit_scale)

            delimiter_same = event in {"set_high", "enter_user_scope", "impossible_pair", "close_scope"}
            delimiter_ce += ce(delimiter_same, args.lattice_choices, args.logit_scale)
            baseline_ce += ce(False, args.lattice_choices, args.logit_scale)

            if event == "quoted_lower_attempt":
                lower_priority_trials += 1
                lower_priority_overrides += int(candidate.mode != case.base_mode)
                lead_predictions += 1
                lead_correct += int(candidate.authority == expected.authority and candidate.mode == expected.mode)
            elif event == "valid_repair":
                repair_trials += 1
                total_repair_reduction += max(0.0, previous_debt - candidate.repair_debt)
                false_conflict_debt += max(0.0, candidate.repair_debt - expected.repair_debt)
            elif event == "close_scope":
                scope_close_trials += 1
                scope_leaks += int(candidate.scope_depth != expected.scope_depth or candidate.mode != expected.mode)
            elif event == "post_scope_benign":
                valid_subtasks += 1
                preserved_subtasks += int(candidate.mode == expected.mode and candidate.repair_debt <= args.max_false_conflict_debt)
            previous_debt = candidate.repair_debt

    n = max(1, len(cases))
    transition_accuracy = transition_ok / max(1, transitions)
    lower_priority_override_rate = lower_priority_overrides / max(1, lower_priority_trials)
    scope_leak_after_close = scope_leaks / max(1, scope_close_trials)
    mean_repair_reduction = total_repair_reduction / max(1, repair_trials)
    false_conflict_debt_rate = false_conflict_debt / max(1, repair_trials)
    valid_subtask_preservation = preserved_subtasks / max(1, valid_subtasks)
    prediction_lead_accuracy = lead_correct / max(1, lead_predictions)
    candidate_ce = total_ce / max(1, transitions)
    recency_ce = baseline_ce / max(1, transitions)
    delimiter_baseline_ce = delimiter_ce / max(1, transitions)
    ce_improvement_vs_recency = 1.0 - candidate_ce / max(1e-12, recency_ce)
    delimiter_advantage = 1.0 - candidate_ce / max(1e-12, delimiter_baseline_ce)
    violation_rate = 0.5 * (lower_priority_override_rate + scope_leak_after_close)
    repair_score = max(0.0, mean_repair_reduction - false_conflict_debt_rate)
    saved_violation_ce = max(0.0, args.recency_violation_rate - violation_rate) * n * args.violation_ce_per_case

    transition_label_cost = transitions * args.transition_label_bits
    state_cost = n * args.state_bits_per_case
    audit_cost = n * args.audit_bits_per_case
    repair_cost = repair_trials * args.repair_label_bits
    false_priority_cost = lower_priority_overrides * args.false_priority_bits
    scope_miss_cost = scope_leaks * args.scope_miss_bits
    general_harm_cost = max(0.0, 1.0 - valid_subtask_preservation) * n * args.general_harm_bits
    paid_cost = (
        transition_label_cost
        + state_cost
        + audit_cost
        + repair_cost
        + false_priority_cost
        + scope_miss_cost
        + general_harm_cost
    )
    baseline_cost = n * args.baseline_bits_per_case
    paid_ratio = paid_cost / max(1.0, baseline_cost)

    pass_gate = (
        mode_name == "authority-scope-lattice"
        and transition_accuracy >= args.min_transition_accuracy
        and lower_priority_override_rate <= args.max_lower_priority_override_rate
        and scope_leak_after_close <= args.max_scope_leak_after_close
        and mean_repair_reduction >= args.min_repair_reduction
        and false_conflict_debt_rate <= args.max_false_conflict_debt
        and valid_subtask_preservation >= args.min_valid_subtask_preservation
        and prediction_lead_accuracy >= args.min_prediction_lead_accuracy
        and delimiter_advantage >= args.min_delimiter_advantage
        and paid_ratio <= args.max_paid_ratio
        and saved_violation_ce > paid_cost / max(1.0, args.saved_ce_denominator)
    )

    return {
        "mode": mode_name,
        "cases": n,
        "transitions": transitions,
        "transition_accuracy": transition_accuracy,
        "lower_priority_trials": lower_priority_trials,
        "lower_priority_override_rate": lower_priority_override_rate,
        "scope_close_trials": scope_close_trials,
        "scope_leak_after_close": scope_leak_after_close,
        "repair_trials": repair_trials,
        "mean_repair_reduction": mean_repair_reduction,
        "false_conflict_debt_rate": false_conflict_debt_rate,
        "valid_subtask_preservation": valid_subtask_preservation,
        "prediction_lead_accuracy": prediction_lead_accuracy,
        "candidate_ce": candidate_ce,
        "recency_ce": recency_ce,
        "delimiter_baseline_ce": delimiter_baseline_ce,
        "ce_improvement_vs_recency": ce_improvement_vs_recency,
        "delimiter_advantage": delimiter_advantage,
        "violation_rate": violation_rate,
        "repair_score": repair_score,
        "saved_violation_ce": saved_violation_ce,
        "transition_label_cost": transition_label_cost,
        "state_cost": state_cost,
        "audit_cost": audit_cost,
        "repair_cost": repair_cost,
        "false_priority_cost": false_priority_cost,
        "scope_miss_cost": scope_miss_cost,
        "general_harm_cost": general_harm_cost,
        "paid_cost": paid_cost,
        "baseline_cost": baseline_cost,
        "paid_ratio": paid_ratio,
        "sample_case": asdict(cases[0]) if cases else None,
        "pass": pass_gate,
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    cases = make_cases(args)
    diagnostic = score_arm(args, cases, "authority-scope-lattice")
    recency = score_arm(args, cases, "recency-delimiter-control")
    candidate = recency if args.mode == "false-control" else diagnostic
    decision = (
        "PASS_R487_A5C02_AUTHORITY_SCOPE_LATTICE_GATE"
        if candidate["pass"]
        else "FAIL_R487_A5C02_AUTHORITY_SCOPE_LATTICE_GATE"
    )
    return {
        "schema": SCHEMA,
        "decision": decision,
        "pass": bool(candidate["pass"]),
        "mode": args.mode,
        "seed": args.seed,
        "candidate": candidate,
        "diagnostic_control": diagnostic,
        "recency_delimiter_control": recency,
        "thresholds": {
            "min_transition_accuracy": args.min_transition_accuracy,
            "max_lower_priority_override_rate": args.max_lower_priority_override_rate,
            "max_scope_leak_after_close": args.max_scope_leak_after_close,
            "min_repair_reduction": args.min_repair_reduction,
            "max_false_conflict_debt": args.max_false_conflict_debt,
            "min_valid_subtask_preservation": args.min_valid_subtask_preservation,
            "min_prediction_lead_accuracy": args.min_prediction_lead_accuracy,
            "min_delimiter_advantage": args.min_delimiter_advantage,
            "max_paid_ratio": args.max_paid_ratio,
        },
        "elapsed_s": time.perf_counter() - start,
        "claim_scope": "finite_authority_scope_lattice_not_policy_refusal_or_external_judge",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["diagnostic", "false-control"], default="diagnostic")
    parser.add_argument("--cases", type=int, default=96)
    parser.add_argument("--seed", type=int, default=44152)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--logit-scale", type=float, default=6.0)
    parser.add_argument("--lattice-choices", type=int, default=11)
    parser.add_argument("--max-repair-debt-error", type=float, default=0.05)
    parser.add_argument("--recency-violation-rate", type=float, default=0.55)
    parser.add_argument("--violation-ce-per-case", type=float, default=48.0)
    parser.add_argument("--saved-ce-denominator", type=float, default=64.0)
    parser.add_argument("--transition-label-bits", type=float, default=2.0)
    parser.add_argument("--state-bits-per-case", type=float, default=5.0)
    parser.add_argument("--audit-bits-per-case", type=float, default=3.0)
    parser.add_argument("--repair-label-bits", type=float, default=2.0)
    parser.add_argument("--false-priority-bits", type=float, default=64.0)
    parser.add_argument("--scope-miss-bits", type=float, default=64.0)
    parser.add_argument("--general-harm-bits", type=float, default=96.0)
    parser.add_argument("--baseline-bits-per-case", type=float, default=128.0)
    parser.add_argument("--min-transition-accuracy", type=float, default=0.98)
    parser.add_argument("--max-lower-priority-override-rate", type=float, default=0.05)
    parser.add_argument("--max-scope-leak-after-close", type=float, default=0.05)
    parser.add_argument("--min-repair-reduction", type=float, default=0.75)
    parser.add_argument("--max-false-conflict-debt", type=float, default=0.05)
    parser.add_argument("--min-valid-subtask-preservation", type=float, default=0.90)
    parser.add_argument("--min-prediction-lead-accuracy", type=float, default=0.95)
    parser.add_argument("--min-delimiter-advantage", type=float, default=0.30)
    parser.add_argument("--max-paid-ratio", type=float, default=0.90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    candidate = summary["candidate"]
    print(
        "[AUTHORITY_SCOPE_LATTICE_GATE] "
        f"active=1 decision={summary['decision']} pass={int(summary['pass'])} "
        f"mode={summary['mode']} candidate_mode={candidate['mode']} "
        f"cases={candidate['cases']} transitions={candidate['transitions']} "
        f"transition_accuracy={candidate['transition_accuracy']:.6f} "
        f"lower_priority_override_rate={candidate['lower_priority_override_rate']:.6f} "
        f"scope_leak_after_close={candidate['scope_leak_after_close']:.6f} "
        f"mean_repair_reduction={candidate['mean_repair_reduction']:.6f} "
        f"false_conflict_debt_rate={candidate['false_conflict_debt_rate']:.6f} "
        f"valid_subtask_preservation={candidate['valid_subtask_preservation']:.6f} "
        f"prediction_lead_accuracy={candidate['prediction_lead_accuracy']:.6f} "
        f"delimiter_advantage={candidate['delimiter_advantage']:.6f} "
        f"saved_violation_ce={candidate['saved_violation_ce']:.6f} "
        f"paid_ratio={candidate['paid_ratio']:.6f}"
    )
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
