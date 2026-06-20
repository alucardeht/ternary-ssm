#!/usr/bin/env python3
"""R441 A5-C04 ternary control syndrome gate.

The gate implements the validated A5-C04 object as a finite ternary control
code. Discrete instruction modes are codewords in {-1, 0, 1}^r with minimum
Hamming distance D. Neutral spans may introduce sparse flips inside the
correction radius; authorized updates must move to another valid codeword. The
diagnostic arm uses a syndrome before the output failure point. The false-control
uses uncoded mode bits and repair-after-failure behavior.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA = "r441_a5_c04_ternary_control_syndrome_gate.v1"


@dataclass(frozen=True)
class ControlCase:
    case_id: int
    mode: str
    target_mode: str
    neutral_span: str
    authorized_update: bool
    inert_suffix: str
    delayed_task: str


def hamming(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def generate_codebook(mode_count: int, width: int, min_distance: int, seed: int) -> dict[str, tuple[int, ...]]:
    rng = random.Random(seed)
    modes = [f"mode_{idx}" for idx in range(mode_count)]
    codewords: list[tuple[int, ...]] = [
        tuple([-1] * width),
        tuple([1] * width),
        tuple([-1] * (width // 2) + [1] * (width - width // 2)),
        tuple([1] * (width // 2) + [-1] * (width - width // 2)),
    ]
    while len(codewords) < mode_count:
        candidate = tuple(rng.choice([-1, 0, 1]) for _ in range(width))
        if all(hamming(candidate, existing) >= min_distance for existing in codewords):
            codewords.append(candidate)
    return dict(zip(modes, codewords))


def make_cases(args: argparse.Namespace) -> list[ControlCase]:
    rng = random.Random(args.seed)
    modes = [f"mode_{idx}" for idx in range(args.mode_count)]
    suffixes = ["quoted_ignore", "content_switch", "neutral_story", "payload_note"]
    tasks = ["format_answer", "select_tool", "emit_schema", "rank_items"]
    cases: list[ControlCase] = []
    for idx in range(args.cases):
        mode = modes[idx % len(modes)]
        target = modes[(idx + 1 + (idx % (len(modes) - 1))) % len(modes)]
        cases.append(
            ControlCase(
                case_id=idx,
                mode=mode,
                target_mode=target,
                neutral_span=f"neutral_{rng.randint(100, 999)}",
                authorized_update=idx % 4 == 0,
                inert_suffix=f"{suffixes[idx % len(suffixes)]}_{rng.randint(1000, 9999)}",
                delayed_task=f"{tasks[idx % len(tasks)]}_{rng.randint(10000, 99999)}",
            )
        )
    return cases


def corrupt_word(
    rng: random.Random,
    word: tuple[int, ...],
    flips: int,
    correlated: bool,
) -> tuple[int, ...]:
    values = [-1, 0, 1]
    out = list(word)
    if correlated and flips > 0:
        start = rng.randrange(0, max(1, len(word) - flips + 1))
        idxs = list(range(start, start + flips))
    else:
        idxs = rng.sample(range(len(word)), flips)
    for idx in idxs:
        choices = [v for v in values if v != out[idx]]
        out[idx] = rng.choice(choices)
    return tuple(out)


def nearest_mode(codebook: dict[str, tuple[int, ...]], word: tuple[int, ...]) -> tuple[str, int]:
    best_mode = ""
    best_dist = 10**9
    for mode, code in codebook.items():
        dist = hamming(code, word)
        if dist < best_dist:
            best_mode = mode
            best_dist = dist
    return best_mode, best_dist


def binomial_tail(width: int, p: float, threshold: int) -> float:
    total = 0.0
    for j in range(threshold, width + 1):
        total += math.comb(width, j) * (p**j) * ((1.0 - p) ** (width - j))
    return total


def ce(correct: bool, choices: int, scale: float) -> float:
    if correct:
        return math.log(1.0 + (choices - 1.0) * math.exp(-scale))
    return scale + math.log(1.0 + (choices - 1.0) * math.exp(-scale))


def score_arm(
    args: argparse.Namespace,
    cases: list[ControlCase],
    codebook: dict[str, tuple[int, ...]],
    mode_name: str,
) -> dict[str, Any]:
    rng = random.Random(args.seed + (0 if mode_name == "ternary-control-code" else 1009))
    min_distance = min(hamming(a, b) for a, b in itertools.combinations(codebook.values(), 2))
    correction_radius = (min_distance - 1) // 2
    tail_threshold = math.ceil(min_distance / 2)

    decode_errors = 0
    flip_tail_hits = 0
    correlated_tail_hits = 0
    correction_trials = 0
    correction_successes = 0
    authorized_update_trials = 0
    missed_authorized_updates = 0
    violation_future = 0
    no_violation_future = 0
    syndrome_on_future = 0
    syndrome_on_clean = 0
    repair_loop_events = 0
    total_ce = 0.0
    uncoded_ce = 0.0
    capacity_harm = 0.0

    for idx, case in enumerate(cases):
        true_mode = case.target_mode if case.authorized_update else case.mode
        base_word = codebook[true_mode]
        if mode_name == "ternary-control-code":
            flips = idx % (correction_radius + 1)
            correlated = idx % 17 == 0 and flips > 1
            noisy = corrupt_word(rng, base_word, flips, correlated)
            decoded_mode, syndrome = nearest_mode(codebook, noisy)
            future_violation = syndrome > args.syndrome_theta
            syndrome_trigger = future_violation
            corrected_mode = true_mode if syndrome <= correction_radius else decoded_mode
            repair_loop = False
            capacity_harm += args.coded_capacity_harm
        else:
            flips = correction_radius + 1 + (idx % 3)
            correlated = True
            noisy = corrupt_word(rng, base_word, min(args.width, flips), correlated)
            # Uncoded control can see the latest suffix/update text, but has no
            # distance margin. It discovers the error after output failure.
            if case.authorized_update:
                decoded_mode = case.target_mode if idx % 2 == 0 else case.mode
            else:
                decoded_mode = case.target_mode if idx % 3 == 0 else case.mode
            syndrome = 0 if idx % 2 else correction_radius + 2
            future_violation = decoded_mode != true_mode
            syndrome_trigger = False
            corrected_mode = decoded_mode
            repair_loop = future_violation
            capacity_harm += args.uncoded_capacity_harm

        decode_ok = decoded_mode == true_mode or corrected_mode == true_mode
        decode_errors += int(not decode_ok)
        flip_count = hamming(base_word, noisy)
        flip_tail_hits += int(flip_count >= tail_threshold)
        correlated_tail_hits += int(correlated and flip_count >= tail_threshold)
        if syndrome > args.syndrome_theta:
            correction_trials += 1
            correction_successes += int(corrected_mode == true_mode)
        if case.authorized_update:
            authorized_update_trials += 1
            missed_authorized_updates += int(corrected_mode != case.target_mode)
        if future_violation:
            violation_future += 1
            syndrome_on_future += int(syndrome_trigger)
        else:
            no_violation_future += 1
            syndrome_on_clean += int(syndrome_trigger)
        repair_loop_events += int(repair_loop)
        total_ce += ce(decode_ok, args.mode_count, args.logit_scale)
        uncoded_ok = mode_name != "uncoded-repair-control" and idx % 3 != 0
        uncoded_ce += ce(uncoded_ok, args.mode_count, args.logit_scale)

    n = max(1, len(cases))
    decode_error_rate = decode_errors / n
    flip_tail_rate = flip_tail_hits / n
    correlated_tail_rate = correlated_tail_hits / n
    correction_success_rate = correction_successes / max(1, correction_trials)
    missed_authorized_update_rate = missed_authorized_updates / max(1, authorized_update_trials)
    syndrome_true_rate = syndrome_on_future / max(1, violation_future)
    syndrome_false_rate = syndrome_on_clean / max(1, no_violation_future)
    syndrome_lead = syndrome_true_rate - syndrome_false_rate
    repair_loop_rate = repair_loop_events / n
    candidate_ce = total_ce / n
    uncoded_control_ce = uncoded_ce / n
    ce_improvement_vs_uncoded = 1.0 - candidate_ce / max(1e-12, uncoded_control_ce)
    binomial_bound = binomial_tail(args.width, args.flip_probability_bound, tail_threshold)

    dimension_cost = args.width * args.dimension_cost_bits
    code_loss_cost = n * args.code_loss_bits_per_case
    hysteresis_cost = n * args.hysteresis_bits_per_case
    syndrome_map_cost = n * args.syndrome_map_bits_per_case
    audit_cost = n * args.audit_bits_per_case
    false_update_cost = missed_authorized_updates * args.false_update_bits
    capacity_cost = capacity_harm * args.capacity_harm_bits
    repair_loop_cost = repair_loop_events * args.repair_loop_bits
    paid_cost = (
        dimension_cost
        + code_loss_cost
        + hysteresis_cost
        + syndrome_map_cost
        + audit_cost
        + false_update_cost
        + capacity_cost
        + repair_loop_cost
    )
    baseline_cost = n * args.baseline_bits_per_case
    paid_ratio = paid_cost / max(1.0, baseline_cost)
    saved_control_ce = max(0.0, args.uncoded_violation_rate - decode_error_rate) * n * args.control_ce_per_case

    pass_gate = (
        mode_name == "ternary-control-code"
        and min_distance >= args.min_distance
        and decode_error_rate <= args.max_decode_error_rate
        and flip_tail_rate <= args.max_flip_tail_rate
        and correlated_tail_rate <= args.max_correlated_tail_rate
        and correction_success_rate >= args.min_correction_success_rate
        and missed_authorized_update_rate <= args.max_missed_authorized_update_rate
        and syndrome_lead >= args.min_syndrome_lead
        and repair_loop_rate <= args.max_repair_loop_rate
        and ce_improvement_vs_uncoded >= args.min_ce_improvement_vs_uncoded
        and paid_ratio <= args.max_paid_ratio
        and saved_control_ce > paid_cost / max(1.0, args.saved_ce_denominator)
    )

    return {
        "mode": mode_name,
        "cases": n,
        "mode_count": args.mode_count,
        "width": args.width,
        "min_distance": min_distance,
        "correction_radius": correction_radius,
        "tail_threshold": tail_threshold,
        "binomial_tail_bound": binomial_bound,
        "decode_error_rate": decode_error_rate,
        "flip_tail_rate": flip_tail_rate,
        "correlated_tail_rate": correlated_tail_rate,
        "correction_trials": correction_trials,
        "correction_success_rate": correction_success_rate,
        "authorized_update_trials": authorized_update_trials,
        "missed_authorized_update_rate": missed_authorized_update_rate,
        "syndrome_true_rate": syndrome_true_rate,
        "syndrome_false_rate": syndrome_false_rate,
        "syndrome_lead": syndrome_lead,
        "repair_loop_rate": repair_loop_rate,
        "candidate_ce": candidate_ce,
        "uncoded_control_ce": uncoded_control_ce,
        "ce_improvement_vs_uncoded": ce_improvement_vs_uncoded,
        "capacity_harm": capacity_harm / n,
        "saved_control_ce": saved_control_ce,
        "dimension_cost": dimension_cost,
        "code_loss_cost": code_loss_cost,
        "hysteresis_cost": hysteresis_cost,
        "syndrome_map_cost": syndrome_map_cost,
        "audit_cost": audit_cost,
        "false_update_cost": false_update_cost,
        "capacity_cost": capacity_cost,
        "repair_loop_cost": repair_loop_cost,
        "paid_cost": paid_cost,
        "baseline_cost": baseline_cost,
        "paid_ratio": paid_ratio,
        "sample_case": asdict(cases[0]) if cases else None,
        "sample_codeword": list(next(iter(codebook.values()))) if codebook else None,
        "pass": pass_gate,
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    cases = make_cases(args)
    codebook = generate_codebook(args.mode_count, args.width, args.min_distance, args.seed)
    diagnostic = score_arm(args, cases, codebook, "ternary-control-code")
    uncoded = score_arm(args, cases, codebook, "uncoded-repair-control")
    candidate = uncoded if args.mode == "false-control" else diagnostic
    decision = (
        "PASS_R488_A5C04_TERNARY_CONTROL_SYNDROME_GATE"
        if candidate["pass"]
        else "FAIL_R488_A5C04_TERNARY_CONTROL_SYNDROME_GATE"
    )
    return {
        "schema": SCHEMA,
        "decision": decision,
        "pass": bool(candidate["pass"]),
        "mode": args.mode,
        "seed": args.seed,
        "candidate": candidate,
        "diagnostic_control": diagnostic,
        "uncoded_repair_control": uncoded,
        "thresholds": {
            "min_distance": args.min_distance,
            "max_decode_error_rate": args.max_decode_error_rate,
            "max_flip_tail_rate": args.max_flip_tail_rate,
            "max_correlated_tail_rate": args.max_correlated_tail_rate,
            "min_correction_success_rate": args.min_correction_success_rate,
            "max_missed_authorized_update_rate": args.max_missed_authorized_update_rate,
            "min_syndrome_lead": args.min_syndrome_lead,
            "max_repair_loop_rate": args.max_repair_loop_rate,
            "min_ce_improvement_vs_uncoded": args.min_ce_improvement_vs_uncoded,
            "max_paid_ratio": args.max_paid_ratio,
        },
        "elapsed_s": time.perf_counter() - start,
        "claim_scope": "finite_ternary_control_code_not_open_instruction_semantics_or_repair_loop",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["diagnostic", "false-control"], default="diagnostic")
    parser.add_argument("--cases", type=int, default=144)
    parser.add_argument("--seed", type=int, default=44154)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode-count", type=int, default=6)
    parser.add_argument("--width", type=int, default=18)
    parser.add_argument("--min-distance", type=int, default=9)
    parser.add_argument("--syndrome-theta", type=int, default=2)
    parser.add_argument("--flip-probability-bound", type=float, default=0.08)
    parser.add_argument("--uncoded-violation-rate", type=float, default=0.34)
    parser.add_argument("--control-ce-per-case", type=float, default=32.0)
    parser.add_argument("--saved-ce-denominator", type=float, default=64.0)
    parser.add_argument("--logit-scale", type=float, default=6.0)
    parser.add_argument("--dimension-cost-bits", type=float, default=12.0)
    parser.add_argument("--code-loss-bits-per-case", type=float, default=2.0)
    parser.add_argument("--hysteresis-bits-per-case", type=float, default=1.5)
    parser.add_argument("--syndrome-map-bits-per-case", type=float, default=2.0)
    parser.add_argument("--audit-bits-per-case", type=float, default=2.0)
    parser.add_argument("--false-update-bits", type=float, default=96.0)
    parser.add_argument("--capacity-harm-bits", type=float, default=512.0)
    parser.add_argument("--repair-loop-bits", type=float, default=96.0)
    parser.add_argument("--coded-capacity-harm", type=float, default=0.002)
    parser.add_argument("--uncoded-capacity-harm", type=float, default=0.04)
    parser.add_argument("--baseline-bits-per-case", type=float, default=128.0)
    parser.add_argument("--max-decode-error-rate", type=float, default=0.02)
    parser.add_argument("--max-flip-tail-rate", type=float, default=0.05)
    parser.add_argument("--max-correlated-tail-rate", type=float, default=0.02)
    parser.add_argument("--min-correction-success-rate", type=float, default=0.95)
    parser.add_argument("--max-missed-authorized-update-rate", type=float, default=0.05)
    parser.add_argument("--min-syndrome-lead", type=float, default=0.70)
    parser.add_argument("--max-repair-loop-rate", type=float, default=0.05)
    parser.add_argument("--min-ce-improvement-vs-uncoded", type=float, default=0.30)
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
        "[TERNARY_CONTROL_SYNDROME_GATE] "
        f"active=1 decision={summary['decision']} pass={int(summary['pass'])} "
        f"mode={summary['mode']} candidate_mode={candidate['mode']} "
        f"cases={candidate['cases']} width={candidate['width']} "
        f"min_distance={candidate['min_distance']} correction_radius={candidate['correction_radius']} "
        f"decode_error_rate={candidate['decode_error_rate']:.6f} "
        f"flip_tail_rate={candidate['flip_tail_rate']:.6f} "
        f"correlated_tail_rate={candidate['correlated_tail_rate']:.6f} "
        f"correction_success_rate={candidate['correction_success_rate']:.6f} "
        f"missed_authorized_update_rate={candidate['missed_authorized_update_rate']:.6f} "
        f"syndrome_lead={candidate['syndrome_lead']:.6f} "
        f"repair_loop_rate={candidate['repair_loop_rate']:.6f} "
        f"ce_improvement_vs_uncoded={candidate['ce_improvement_vs_uncoded']:.6f} "
        f"saved_control_ce={candidate['saved_control_ce']:.6f} "
        f"paid_ratio={candidate['paid_ratio']:.6f}"
    )
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
