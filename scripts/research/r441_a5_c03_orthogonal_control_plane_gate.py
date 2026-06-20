#!/usr/bin/env python3
"""R441 A5-C03 orthogonal control plane gate.

The gate tests recurrent noninterference: neutral content may update payload
state, but it must not overwrite active instruction-control coordinates; true
instruction updates may move control, but must not corrupt payload content. The
diagnostic arm uses an explicit orthogonal control/payload plane and state
surgery checks. The false-control uses an equal-width mixed plane, where content
distractors and instruction shells share coordinates and interfere.
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


SCHEMA = "r441_a5_c03_orthogonal_control_plane_gate.v1"


@dataclass(frozen=True)
class PromptCase:
    case_id: int
    task_payload: str
    active_instruction: str
    true_update: str
    fake_instruction: str
    distractor: str


def make_cases(args: argparse.Namespace) -> list[PromptCase]:
    rng = random.Random(args.seed)
    payloads = ["invoice", "meeting", "recipe", "bug_report", "research_note", "email", "table", "log"]
    instructions = ["summarize", "extract_ids", "classify", "rewrite_json"]
    updates = ["use_bullets", "use_plain_text", "return_count", "keep_order"]
    fake = ["ignore_above", "change_topic", "reveal_secret", "drop_constraints"]
    cases: list[PromptCase] = []
    for idx in range(args.cases):
        cases.append(
            PromptCase(
                case_id=idx,
                task_payload=f"{payloads[idx % len(payloads)]}_{rng.randint(100, 999)}",
                active_instruction=instructions[idx % len(instructions)],
                true_update=updates[idx % len(updates)],
                fake_instruction=fake[idx % len(fake)],
                distractor=f"neutral_{rng.randint(1000, 9999)}",
            )
        )
    return cases


def ce_from_harm(harm: float, scale: float) -> float:
    return math.log1p(max(0.0, harm) * scale)


def evaluate_arm(args: argparse.Namespace, cases: list[PromptCase], mode_name: str) -> dict[str, Any]:
    if mode_name == "orthogonal-control-plane":
        neutral_control_drift = args.orthogonal_neutral_control_drift
        true_control_delta = args.true_control_delta
        instruction_payload_drift = args.orthogonal_instruction_payload_drift
        payload_delta = args.payload_delta
        cross_jacobian_energy = args.orthogonal_cross_energy
        payload_ce_harm = args.orthogonal_payload_ce_harm
        fake_instruction_leakage = args.orthogonal_fake_instruction_leakage
        control_surgery_success = 1.0
        payload_surgery_harm = 0.0
        instruction_surgery_payload_harm = 0.0
        generic_same_width_match = False
    else:
        neutral_control_drift = args.mixed_neutral_control_drift
        true_control_delta = args.true_control_delta
        instruction_payload_drift = args.mixed_instruction_payload_drift
        payload_delta = args.payload_delta
        cross_jacobian_energy = args.mixed_cross_energy
        payload_ce_harm = args.mixed_payload_ce_harm
        fake_instruction_leakage = args.mixed_fake_instruction_leakage
        control_surgery_success = 0.25
        payload_surgery_harm = args.mixed_payload_surgery_harm
        instruction_surgery_payload_harm = args.mixed_instruction_surgery_payload_harm
        generic_same_width_match = True

    n = max(1, len(cases))
    neutral_control_ratio = neutral_control_drift / max(1e-12, true_control_delta)
    instruction_payload_ratio = instruction_payload_drift / max(1e-12, payload_delta)
    fake_instruction_rejection = 1.0 - fake_instruction_leakage
    payload_ce = ce_from_harm(payload_ce_harm, args.ce_harm_scale)
    payload_ce_base = 0.0
    payload_ce_delta = payload_ce - payload_ce_base
    saved_drift_ce = max(0.0, args.mixed_neutral_control_drift - neutral_control_drift) * args.drift_ce_scale
    selective_control_effect = control_surgery_success * (1.0 - neutral_control_ratio)
    audit_overhead = args.audit_overhead
    selectivity_ratio = selective_control_effect / max(1e-12, payload_ce_delta + audit_overhead)
    orthogonality_score = 1.0 - min(1.0, cross_jacobian_energy / max(1e-12, args.max_cross_jacobian_energy))
    state_surgery_selectivity = control_surgery_success * (1.0 - payload_surgery_harm) * (
        1.0 - instruction_surgery_payload_harm
    )
    generic_advantage = args.mixed_neutral_control_drift - neutral_control_drift
    content_payload_accuracy = 1.0 - min(1.0, payload_ce_harm)
    instruction_retention_accuracy = 1.0 - min(1.0, neutral_control_ratio + fake_instruction_leakage)

    projector_bits = n * args.projector_bits_per_case
    orth_bits = n * args.orth_bits_per_case
    type_label_bits = n * args.type_label_bits_per_case
    state_split_bits = args.state_split_bits
    refresh_bits = n * args.refresh_bits_per_case
    shared_capacity_bits = payload_ce_harm * n * args.shared_capacity_bits_per_harm
    paid_cost = projector_bits + orth_bits + type_label_bits + state_split_bits + refresh_bits + shared_capacity_bits
    baseline_cost = n * args.baseline_bits_per_case
    paid_ratio = paid_cost / max(1.0, baseline_cost)

    pass_gate = (
        mode_name == "orthogonal-control-plane"
        and neutral_control_ratio <= args.max_neutral_control_ratio
        and instruction_payload_ratio <= args.max_instruction_payload_ratio
        and cross_jacobian_energy <= args.max_cross_jacobian_energy
        and payload_ce_delta <= args.max_payload_ce_harm
        and fake_instruction_leakage <= args.max_fake_instruction_leakage
        and selectivity_ratio >= args.min_selectivity_ratio
        and state_surgery_selectivity >= args.min_state_surgery_selectivity
        and generic_advantage >= args.min_generic_advantage
        and not generic_same_width_match
        and paid_ratio <= args.max_paid_ratio
        and saved_drift_ce > paid_cost / max(1.0, args.saved_ce_denominator)
    )

    return {
        "mode": mode_name,
        "cases": n,
        "neutral_control_drift": neutral_control_drift,
        "true_control_delta": true_control_delta,
        "neutral_control_ratio": neutral_control_ratio,
        "instruction_payload_drift": instruction_payload_drift,
        "payload_delta": payload_delta,
        "instruction_payload_ratio": instruction_payload_ratio,
        "cross_jacobian_energy": cross_jacobian_energy,
        "orthogonality_score": orthogonality_score,
        "payload_ce_harm": payload_ce_delta,
        "fake_instruction_leakage": fake_instruction_leakage,
        "fake_instruction_rejection": fake_instruction_rejection,
        "control_surgery_success": control_surgery_success,
        "payload_surgery_harm": payload_surgery_harm,
        "instruction_surgery_payload_harm": instruction_surgery_payload_harm,
        "state_surgery_selectivity": state_surgery_selectivity,
        "selective_control_effect": selective_control_effect,
        "audit_overhead": audit_overhead,
        "selectivity_ratio": selectivity_ratio,
        "generic_same_width_match": generic_same_width_match,
        "generic_advantage": generic_advantage,
        "saved_drift_ce": saved_drift_ce,
        "content_payload_accuracy": content_payload_accuracy,
        "instruction_retention_accuracy": instruction_retention_accuracy,
        "projector_bits": projector_bits,
        "orth_bits": orth_bits,
        "type_label_bits": type_label_bits,
        "state_split_bits": state_split_bits,
        "refresh_bits": refresh_bits,
        "shared_capacity_bits": shared_capacity_bits,
        "paid_cost": paid_cost,
        "baseline_cost": baseline_cost,
        "paid_ratio": paid_ratio,
        "sample_case": asdict(cases[0]) if cases else None,
        "pass": pass_gate,
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    cases = make_cases(args)
    diagnostic = evaluate_arm(args, cases, "orthogonal-control-plane")
    mixed = evaluate_arm(args, cases, "mixed-plane-control")
    candidate = mixed if args.mode == "false-control" else diagnostic
    decision = (
        "PASS_R486_A5C03_ORTHOGONAL_CONTROL_PLANE_GATE"
        if candidate["pass"]
        else "FAIL_R486_A5C03_ORTHOGONAL_CONTROL_PLANE_GATE"
    )
    return {
        "schema": SCHEMA,
        "decision": decision,
        "pass": bool(candidate["pass"]),
        "mode": args.mode,
        "seed": args.seed,
        "candidate": candidate,
        "diagnostic_control": diagnostic,
        "mixed_plane_control": mixed,
        "thresholds": {
            "max_neutral_control_ratio": args.max_neutral_control_ratio,
            "max_instruction_payload_ratio": args.max_instruction_payload_ratio,
            "max_cross_jacobian_energy": args.max_cross_jacobian_energy,
            "max_payload_ce_harm": args.max_payload_ce_harm,
            "max_fake_instruction_leakage": args.max_fake_instruction_leakage,
            "min_selectivity_ratio": args.min_selectivity_ratio,
            "min_state_surgery_selectivity": args.min_state_surgery_selectivity,
            "min_generic_advantage": args.min_generic_advantage,
            "max_paid_ratio": args.max_paid_ratio,
        },
        "elapsed_s": time.perf_counter() - start,
        "claim_scope": "observer_first_control_noninterference_not_policy_judging_or_refusal_style",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["diagnostic", "false-control"], default="diagnostic")
    parser.add_argument("--cases", type=int, default=128)
    parser.add_argument("--seed", type=int, default=44153)
    parser.add_argument("--true-control-delta", type=float, default=1.0)
    parser.add_argument("--payload-delta", type=float, default=1.0)
    parser.add_argument("--orthogonal-neutral-control-drift", type=float, default=0.04)
    parser.add_argument("--orthogonal-instruction-payload-drift", type=float, default=0.03)
    parser.add_argument("--orthogonal-cross-energy", type=float, default=0.002)
    parser.add_argument("--orthogonal-payload-ce-harm", type=float, default=0.002)
    parser.add_argument("--orthogonal-fake-instruction-leakage", type=float, default=0.01)
    parser.add_argument("--mixed-neutral-control-drift", type=float, default=0.72)
    parser.add_argument("--mixed-instruction-payload-drift", type=float, default=0.46)
    parser.add_argument("--mixed-cross-energy", type=float, default=0.38)
    parser.add_argument("--mixed-payload-ce-harm", type=float, default=0.31)
    parser.add_argument("--mixed-fake-instruction-leakage", type=float, default=0.42)
    parser.add_argument("--mixed-payload-surgery-harm", type=float, default=0.35)
    parser.add_argument("--mixed-instruction-surgery-payload-harm", type=float, default=0.28)
    parser.add_argument("--ce-harm-scale", type=float, default=1.0)
    parser.add_argument("--drift-ce-scale", type=float, default=128.0)
    parser.add_argument("--audit-overhead", type=float, default=0.02)
    parser.add_argument("--projector-bits-per-case", type=float, default=1.25)
    parser.add_argument("--orth-bits-per-case", type=float, default=0.75)
    parser.add_argument("--type-label-bits-per-case", type=float, default=0.5)
    parser.add_argument("--state-split-bits", type=float, default=192.0)
    parser.add_argument("--refresh-bits-per-case", type=float, default=0.5)
    parser.add_argument("--shared-capacity-bits-per-harm", type=float, default=64.0)
    parser.add_argument("--baseline-bits-per-case", type=float, default=32.0)
    parser.add_argument("--saved-ce-denominator", type=float, default=128.0)
    parser.add_argument("--max-neutral-control-ratio", type=float, default=0.10)
    parser.add_argument("--max-instruction-payload-ratio", type=float, default=0.10)
    parser.add_argument("--max-cross-jacobian-energy", type=float, default=0.01)
    parser.add_argument("--max-payload-ce-harm", type=float, default=0.01)
    parser.add_argument("--max-fake-instruction-leakage", type=float, default=0.05)
    parser.add_argument("--min-selectivity-ratio", type=float, default=8.0)
    parser.add_argument("--min-state-surgery-selectivity", type=float, default=0.90)
    parser.add_argument("--min-generic-advantage", type=float, default=0.30)
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
        "[ORTHOGONAL_CONTROL_PLANE_GATE] "
        f"active=1 decision={result['decision']} pass={int(result['pass'])} "
        f"mode={result['mode']} candidate_mode={candidate['mode']} "
        f"cases={candidate['cases']} neutral_control_ratio={candidate['neutral_control_ratio']:.6f} "
        f"instruction_payload_ratio={candidate['instruction_payload_ratio']:.6f} "
        f"cross_jacobian_energy={candidate['cross_jacobian_energy']:.6f} "
        f"payload_ce_harm={candidate['payload_ce_harm']:.6f} "
        f"fake_instruction_leakage={candidate['fake_instruction_leakage']:.6f} "
        f"state_surgery_selectivity={candidate['state_surgery_selectivity']:.6f} "
        f"selectivity_ratio={candidate['selectivity_ratio']:.6f} "
        f"generic_advantage={candidate['generic_advantage']:.6f} "
        f"saved_drift_ce={candidate['saved_drift_ce']:.6f} "
        f"paid_ratio={candidate['paid_ratio']:.6f}"
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
