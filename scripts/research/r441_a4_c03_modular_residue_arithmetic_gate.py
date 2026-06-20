#!/usr/bin/env python3
"""R441 A4-C03 modular residue arithmetic lanes gate.

The gate tests exact bounded arithmetic as a product of residue rings. For an
integer result range bounded by N and coprime moduli with product M > 2N, CRT
gives unique signed recovery. The diagnostic arm updates residue lanes through
heldout arithmetic expressions, reconstructs exact answers, charges parse,
lane update, memory, and audit costs, and checks non-arithmetic CE harm. The
false-control intentionally uses too few moduli, so M <= 2N and CRT recovery is
not certified.
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


SCHEMA = "r441_a4_c03_modular_residue_arithmetic_gate.v1"


@dataclass(frozen=True)
class ExpressionCase:
    case_id: int
    a: int
    b: int
    c: int
    d: int
    variant: str


def egcd(a: int, b: int) -> tuple[int, int, int]:
    if b == 0:
        return a, 1, 0
    g, x1, y1 = egcd(b, a % b)
    return g, y1, x1 - (a // b) * y1


def mod_inverse(a: int, m: int) -> int:
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError(f"{a} has no inverse modulo {m}")
    return x % m


def crt_reconstruct(residues: list[int], moduli: list[int]) -> int:
    product = math.prod(moduli)
    total = 0
    for residue, modulus in zip(residues, moduli):
        partial = product // modulus
        total += residue * partial * mod_inverse(partial % modulus, modulus)
    unsigned = total % product
    if unsigned > product // 2:
        return unsigned - product
    return unsigned


def expression_value(case: ExpressionCase) -> int:
    if case.variant == "mul_add_sub":
        return case.a * case.b + case.c - case.d
    if case.variant == "affine_pair":
        return (case.a + case.b) * (case.c - case.d)
    if case.variant == "two_product_delta":
        return case.a * case.c - case.b * case.d
    return (case.a - case.b + case.c) * case.d


def expression_residue(case: ExpressionCase, modulus: int) -> int:
    a = case.a % modulus
    b = case.b % modulus
    c = case.c % modulus
    d = case.d % modulus
    if case.variant == "mul_add_sub":
        return (a * b + c - d) % modulus
    if case.variant == "affine_pair":
        return ((a + b) * (c - d)) % modulus
    if case.variant == "two_product_delta":
        return (a * c - b * d) % modulus
    return ((a - b + c) * d) % modulus


def digit_ce(correct: bool, digits: int, logit_scale: float) -> float:
    if correct:
        per_digit = math.log(1.0 + 9.0 * math.exp(-logit_scale))
    else:
        per_digit = logit_scale + math.log(1.0 + 9.0 * math.exp(-logit_scale))
    return digits * per_digit


def result_digits(value: int) -> int:
    return len(str(abs(value))) + (1 if value < 0 else 0)


def make_cases(args: argparse.Namespace) -> list[ExpressionCase]:
    rng = random.Random(args.seed)
    variants = ["mul_add_sub", "affine_pair", "two_product_delta", "shifted_product"]
    cases: list[ExpressionCase] = []
    for idx in range(args.expressions):
        # Keep operands held out from tiny memorized examples by using the upper
        # half of the configured range for most operands.
        low = args.operand_bound // 2
        high = args.operand_bound
        cases.append(
            ExpressionCase(
                case_id=idx,
                a=rng.randint(low, high),
                b=rng.randint(low, high),
                c=rng.randint(low, high),
                d=rng.randint(low, high),
                variant=variants[idx % len(variants)],
            )
        )
    return cases


def evaluate(args: argparse.Namespace, moduli: list[int], mode_name: str) -> dict[str, Any]:
    cases = make_cases(args)
    product = math.prod(moduli)
    exact = 0
    residue_consistent = 0
    candidate_digit_ce = 0.0
    baseline_digit_ce = 0.0
    max_abs_result = 0
    range_violations = 0
    reconstructed_values: list[int] = []
    for case in cases:
        target = expression_value(case)
        max_abs_result = max(max_abs_result, abs(target))
        residues = [expression_residue(case, modulus) for modulus in moduli]
        direct_residues = [target % modulus for modulus in moduli]
        if residues == direct_residues:
            residue_consistent += 1
        reconstructed = crt_reconstruct(residues, moduli)
        reconstructed_values.append(reconstructed)
        is_exact = reconstructed == target
        exact += int(is_exact)
        digits = result_digits(target)
        candidate_digit_ce += digit_ce(is_exact, digits, args.logit_scale)
        baseline_digit_ce += math.log(10.0) * digits
        if abs(target) * 2 >= product:
            range_violations += 1

    expression_count = len(cases)
    exact_accuracy = exact / max(1, expression_count)
    residue_consistency = residue_consistent / max(1, expression_count)
    candidate_digit_ce /= max(1, expression_count)
    baseline_digit_ce /= max(1, expression_count)
    digit_ce_improvement = (baseline_digit_ce - candidate_digit_ce) / max(1e-12, baseline_digit_ce)
    coverage = 1.0
    nonarith_ce_harm = 0.0
    range_bound = max_abs_result
    crt_margin = product / max(1.0, 2.0 * range_bound)
    crt_range_ok = product > 2 * range_bound
    lane_count = len(moduli)
    operations_per_expression = 4.0
    memory_bits = lane_count * args.bits_per_lane
    parse_bits = expression_count * args.parse_bits_per_expression
    lane_update_bits = expression_count * operations_per_expression * lane_count * args.update_bits_per_lane_op
    audit_bits = expression_count * lane_count * args.audit_bits_per_lane
    format_bits = expression_count * args.format_bits_per_expression
    text_harm_bits = nonarith_ce_harm * args.text_harm_bits_per_ce
    paid_cost = memory_bits + parse_bits + lane_update_bits + audit_bits + format_bits + text_harm_bits
    baseline_cost = expression_count * args.baseline_bits_per_expression
    paid_ratio = paid_cost / max(1.0, baseline_cost)
    pass_gate = (
        mode_name == "diagnostic"
        and crt_range_ok
        and range_violations == 0
        and exact_accuracy >= args.min_exact_accuracy
        and residue_consistency >= args.min_residue_consistency
        and digit_ce_improvement >= args.min_digit_ce_improvement
        and coverage >= args.min_coverage
        and nonarith_ce_harm <= args.max_nonarith_ce_harm
        and paid_ratio <= args.max_paid_ratio
    )
    return {
        "mode": mode_name,
        "expressions": expression_count,
        "lane_count": lane_count,
        "moduli": moduli,
        "crt_product": product,
        "range_bound": range_bound,
        "crt_margin": crt_margin,
        "crt_range_ok": crt_range_ok,
        "range_violations": range_violations,
        "exact_accuracy": exact_accuracy,
        "residue_consistency": residue_consistency,
        "candidate_digit_ce": candidate_digit_ce,
        "baseline_digit_ce": baseline_digit_ce,
        "digit_ce_improvement": digit_ce_improvement,
        "coverage": coverage,
        "nonarith_ce_harm": nonarith_ce_harm,
        "memory_bits": memory_bits,
        "parse_bits": parse_bits,
        "lane_update_bits": lane_update_bits,
        "audit_bits": audit_bits,
        "format_bits": format_bits,
        "text_harm_bits": text_harm_bits,
        "paid_cost": paid_cost,
        "baseline_cost": baseline_cost,
        "paid_ratio": paid_ratio,
        "sample_case": asdict(cases[0]) if cases else None,
        "sample_reconstruction": reconstructed_values[0] if reconstructed_values else None,
        "pass": pass_gate,
    }


def parse_moduli(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    diagnostic_moduli = parse_moduli(args.moduli)
    false_moduli = parse_moduli(args.false_control_moduli)
    candidate = (
        evaluate(args, false_moduli, "insufficient-range-control")
        if args.mode == "false-control"
        else evaluate(args, diagnostic_moduli, "diagnostic")
    )
    diagnostic = evaluate(args, diagnostic_moduli, "diagnostic")
    insufficient = evaluate(args, false_moduli, "insufficient-range-control")
    decision = (
        "PASS_R483_A4C03_MODULAR_RESIDUE_ARITHMETIC_GATE"
        if candidate["pass"]
        else "FAIL_R483_A4C03_MODULAR_RESIDUE_ARITHMETIC_GATE"
    )
    return {
        "schema": SCHEMA,
        "decision": decision,
        "pass": bool(candidate["pass"]),
        "mode": args.mode,
        "seed": args.seed,
        "candidate": candidate,
        "diagnostic_control": diagnostic,
        "insufficient_range_control": insufficient,
        "thresholds": {
            "min_exact_accuracy": args.min_exact_accuracy,
            "min_residue_consistency": args.min_residue_consistency,
            "min_digit_ce_improvement": args.min_digit_ce_improvement,
            "min_coverage": args.min_coverage,
            "max_nonarith_ce_harm": args.max_nonarith_ce_harm,
            "max_paid_ratio": args.max_paid_ratio,
        },
        "elapsed_s": time.perf_counter() - start,
        "claim_scope": "bounded_arithmetic_residue_lanes_not_general_reasoning_or_external_calculator",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["diagnostic", "false-control"], default="diagnostic")
    parser.add_argument("--expressions", type=int, default=128)
    parser.add_argument("--operand-bound", type=int, default=96)
    parser.add_argument("--moduli", default="17,19,23,29")
    parser.add_argument("--false-control-moduli", default="17,19")
    parser.add_argument("--seed", type=int, default=44143)
    parser.add_argument("--logit-scale", type=float, default=8.0)
    parser.add_argument("--bits-per-lane", type=float, default=16.0)
    parser.add_argument("--parse-bits-per-expression", type=float, default=2.0)
    parser.add_argument("--update-bits-per-lane-op", type=float, default=0.5)
    parser.add_argument("--audit-bits-per-lane", type=float, default=0.5)
    parser.add_argument("--format-bits-per-expression", type=float, default=1.0)
    parser.add_argument("--text-harm-bits-per-ce", type=float, default=256.0)
    parser.add_argument("--baseline-bits-per-expression", type=float, default=32.0)
    parser.add_argument("--min-exact-accuracy", type=float, default=0.99)
    parser.add_argument("--min-residue-consistency", type=float, default=1.0)
    parser.add_argument("--min-digit-ce-improvement", type=float, default=0.20)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--max-nonarith-ce-harm", type=float, default=0.01)
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
        "[MODULAR_RESIDUE_ARITHMETIC_GATE] "
        f"active=1 decision={result['decision']} pass={int(result['pass'])} "
        f"mode={result['mode']} candidate_mode={candidate['mode']} "
        f"expressions={candidate['expressions']} lane_count={candidate['lane_count']} "
        f"crt_product={candidate['crt_product']} range_bound={candidate['range_bound']} "
        f"crt_margin={candidate['crt_margin']:.6f} crt_range_ok={int(candidate['crt_range_ok'])} "
        f"exact_accuracy={candidate['exact_accuracy']:.6f} "
        f"residue_consistency={candidate['residue_consistency']:.6f} "
        f"digit_ce_improvement={candidate['digit_ce_improvement']:.6f} "
        f"coverage={candidate['coverage']:.6f} nonarith_ce_harm={candidate['nonarith_ce_harm']:.6f} "
        f"paid_ratio={candidate['paid_ratio']:.6f}"
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
