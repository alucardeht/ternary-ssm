#!/usr/bin/env python3
"""R502 E4 adjoint-state algebraic sparsity gate.

This CPU diagnostic implements the R502 Efficiency finding:
"Adjoint-State Algebraic Sparsity (ASAS)".

The gate compares dense recurrent cotangent propagation,
lambda[t] = A * lambda[t + 1] + local[t], against a ternary route/sign-flip
implementation. For A_i in {-1, 0, 1}, the recurrent multiply is exactly a
copy, negation, or zero route; no FP32 multiplier is required.

Controls exercise the declared risks: non-ternary recurrent coefficients break
the algebraic route, zero dimensions must not carry cotangents, local adjoint
terms must still be added, and instruction issue cost can erase the denominator
if route/sign operations are not hidden by the compiler.

This is a synthetic CPU cotangent-equivalence and issue-slot diagnostic, not
heldout CE progress, real corpus token efficiency, GPU training-speed evidence,
or production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r502_e4_asas_gate.v1"
CONDITIONS = (
    "positive",
    "non_ternary_fail",
    "zero_route_fail",
    "local_term_missing_fail",
    "issue_slot_cost_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    sequence_len: int
    state_dim: int
    dense_fma_issue_slots: int
    route_issue_slots: float
    issue_slot_ratio: float
    projected_recurrent_multiplier: float
    ternary_fraction: float
    zero_fraction: float
    copy_fraction: float
    sign_flip_fraction: float
    cotangent_l2_error_max: float
    parameter_gradient_l2_error: float
    mismatch_fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50204)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sequence-len", type=int, default=4096)
    parser.add_argument("--state-dim", type=int, default=256)
    parser.add_argument("--zero-fraction", type=float, default=0.25)
    parser.add_argument("--max-cotangent-l2-error", type=float, default=1.0e-12)
    parser.add_argument("--max-parameter-gradient-l2-error", type=float, default=1.0e-12)
    parser.add_argument("--max-mismatch-fraction", type=float, default=0.0)
    parser.add_argument("--max-issue-slot-ratio", type=float, default=0.55)
    parser.add_argument("--copy-issue-weight", type=float, default=0.12)
    parser.add_argument("--sign-flip-issue-weight", type=float, default=0.18)
    parser.add_argument("--zero-route-issue-weight", type=float, default=0.03)
    parser.add_argument("--route-overhead-ratio", type=float, default=0.08)
    parser.add_argument("--expensive-route-overhead-ratio", type=float, default=0.86)
    return parser.parse_args()


def make_diag(args: argparse.Namespace, rng: np.random.Generator, condition: str) -> np.ndarray:
    zero_count = int(round(args.zero_fraction * args.state_dim))
    diag = np.empty(args.state_dim, dtype=np.float64)
    diag[:zero_count] = 0.0
    diag[zero_count:] = rng.choice(np.array([-1.0, 1.0]), size=args.state_dim - zero_count)
    rng.shuffle(diag)
    if condition == "non_ternary_fail":
        nonzero = np.flatnonzero(diag != 0.0)
        diag[nonzero[: max(1, len(nonzero) // 5)]] = 0.5
    return diag


def dense_backward(diag: np.ndarray, local_terms: np.ndarray) -> np.ndarray:
    sequence_len, state_dim = local_terms.shape
    lambdas = np.zeros((sequence_len + 1, state_dim), dtype=np.float64)
    for t in reversed(range(sequence_len)):
        lambdas[t] = diag * lambdas[t + 1] + local_terms[t]
    return lambdas


def route_backward(
    diag: np.ndarray,
    local_terms: np.ndarray,
    condition: str,
) -> np.ndarray:
    sequence_len, state_dim = local_terms.shape
    lambdas = np.zeros((sequence_len + 1, state_dim), dtype=np.float64)
    positive = diag > 0.0
    negative = diag < 0.0
    zero = diag == 0.0
    for t in reversed(range(sequence_len)):
        routed = np.zeros(state_dim, dtype=np.float64)
        routed[positive] = lambdas[t + 1, positive]
        routed[negative] = -lambdas[t + 1, negative]
        if condition == "zero_route_fail":
            routed[zero] = lambdas[t + 1, zero]
        if condition == "local_term_missing_fail":
            lambdas[t] = routed
        else:
            lambdas[t] = routed + local_terms[t]
    return lambdas


def parameter_gradient(lambdas: np.ndarray, states: np.ndarray) -> np.ndarray:
    return np.sum(lambdas[1:] * states[:-1], axis=0)


def issue_slot_price(args: argparse.Namespace, diag: np.ndarray, condition: str) -> tuple[int, float, float, float]:
    dense_slots = int(args.sequence_len * args.state_dim)
    zero_fraction = float(np.mean(diag == 0.0))
    copy_fraction = float(np.mean(diag > 0.0))
    sign_fraction = float(np.mean(diag < 0.0))
    overhead = (
        args.expensive_route_overhead_ratio
        if condition == "issue_slot_cost_fail"
        else args.route_overhead_ratio
    )
    ratio = (
        copy_fraction * args.copy_issue_weight
        + sign_fraction * args.sign_flip_issue_weight
        + zero_fraction * args.zero_route_issue_weight
        + overhead
    )
    route_slots = ratio * dense_slots
    multiplier = 1.0 / ratio
    return dense_slots, route_slots, ratio, multiplier


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    diag = make_diag(args, rng, condition)
    local_terms = rng.normal(scale=0.35, size=(args.sequence_len, args.state_dim))
    states = rng.normal(scale=0.45, size=(args.sequence_len + 1, args.state_dim))
    dense = dense_backward(diag, local_terms)
    routed = route_backward(diag, local_terms, condition)
    errors = routed - dense
    cotangent_l2_error_max = float(np.max(np.linalg.norm(errors, axis=1)))
    mismatch_fraction = float(np.mean(np.abs(errors) > 1.0e-12))
    grad_error = float(np.linalg.norm(parameter_gradient(routed, states) - parameter_gradient(dense, states)))
    dense_slots, route_slots, issue_ratio, multiplier = issue_slot_price(args, diag, condition)
    ternary_fraction = float(np.mean(np.isin(diag, np.array([-1.0, 0.0, 1.0]))))
    zero_fraction = float(np.mean(diag == 0.0))
    copy_fraction = float(np.mean(diag > 0.0))
    sign_fraction = float(np.mean(diag < 0.0))

    failed: list[str] = []
    if not finite(
        issue_ratio,
        multiplier,
        ternary_fraction,
        zero_fraction,
        copy_fraction,
        sign_fraction,
        cotangent_l2_error_max,
        grad_error,
        mismatch_fraction,
    ):
        failed.append("non_finite_metric")
    if ternary_fraction < 1.0:
        failed.append("non_ternary_recurrence")
    if cotangent_l2_error_max > args.max_cotangent_l2_error:
        failed.append("cotangent_error")
    if grad_error > args.max_parameter_gradient_l2_error:
        failed.append("parameter_gradient")
    if mismatch_fraction > args.max_mismatch_fraction:
        failed.append("mismatch_fraction")
    if issue_ratio > args.max_issue_slot_ratio:
        failed.append("issue_slot_price")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed) if failed else None,
        sequence_len=args.sequence_len,
        state_dim=args.state_dim,
        dense_fma_issue_slots=dense_slots,
        route_issue_slots=float(route_slots),
        issue_slot_ratio=float(issue_ratio),
        projected_recurrent_multiplier=float(multiplier),
        ternary_fraction=ternary_fraction,
        zero_fraction=zero_fraction,
        copy_fraction=copy_fraction,
        sign_flip_fraction=sign_fraction,
        cotangent_l2_error_max=cotangent_l2_error_max,
        parameter_gradient_l2_error=grad_error,
        mismatch_fraction=mismatch_fraction,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "sequence_len",
        "state_dim",
        "dense_fma_issue_slots",
        "route_issue_slots",
        "issue_slot_ratio",
        "projected_recurrent_multiplier",
        "ternary_fraction",
        "zero_fraction",
        "copy_fraction",
        "sign_flip_fraction",
        "cotangent_l2_error_max",
        "parameter_gradient_l2_error",
        "mismatch_fraction",
    )
    aggregate: dict[str, Any] = {
        "count": len(results),
        "passed_count": sum(result.passed for result in results),
        "identifying_correct_count": sum(result.identifying_correct for result in results),
        "fail_reasons": sorted({result.fail_reason for result in results if result.fail_reason}),
    }
    for field in fields:
        values = np.array([float(getattr(result, field)) for result in results], dtype=np.float64)
        aggregate[f"{field}_mean"] = float(np.mean(values))
        aggregate[f"{field}_min"] = float(np.min(values))
        aggregate[f"{field}_max"] = float(np.max(values))
    return aggregate


def all_finite(results: list[TrialResult]) -> bool:
    for result in results:
        for value in asdict(result).values():
            if isinstance(value, float) and not math.isfinite(value):
                return False
    return True


def required_checks(results: list[TrialResult], aggregates: dict[str, Any]) -> dict[str, bool]:
    positive = [result for result in results if result.condition == "positive"]
    non_ternary = [result for result in results if result.condition == "non_ternary_fail"]
    zero_route = [result for result in results if result.condition == "zero_route_fail"]
    local_missing = [result for result in results if result.condition == "local_term_missing_fail"]
    issue = [result for result in results if result.condition == "issue_slot_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_cotangent_and_gradient": (
            aggregates["positive"]["cotangent_l2_error_max_max"] <= 1.0e-12
            and aggregates["positive"]["parameter_gradient_l2_error_max"] <= 1.0e-12
            and aggregates["positive"]["mismatch_fraction_max"] == 0.0
        ),
        "positive_condition_ternary_route_exercised": (
            aggregates["positive"]["zero_fraction_mean"] > 0.0
            and aggregates["positive"]["copy_fraction_mean"] > 0.0
            and aggregates["positive"]["sign_flip_fraction_mean"] > 0.0
        ),
        "positive_condition_issue_slots_priced": (
            aggregates["positive"]["issue_slot_ratio_mean"] <= 0.55
        ),
        "non_ternary_control_all_fail": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "non_ternary_recurrence" in result.fail_reason
                or "cotangent_error" in result.fail_reason
                or "parameter_gradient" in result.fail_reason
            )
            for result in non_ternary
        ),
        "zero_route_control_all_fail": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "cotangent_error" in result.fail_reason
                or "parameter_gradient" in result.fail_reason
                or "mismatch_fraction" in result.fail_reason
            )
            for result in zero_route
        ),
        "local_term_control_all_fail": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "cotangent_error" in result.fail_reason
                or "parameter_gradient" in result.fail_reason
                or "mismatch_fraction" in result.fail_reason
            )
            for result in local_missing
        ),
        "issue_cost_control_all_fail": all(
            (not result.passed)
            and result.fail_reason is not None
            and "issue_slot_price" in result.fail_reason
            for result in issue
        ),
        "all_conditions_identified": all(result.identifying_correct for result in results),
    }


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.sequence_len < 2:
        raise ValueError("--sequence-len must be at least 2")
    if args.state_dim < 2:
        raise ValueError("--state-dim must be at least 2")

    results: list[TrialResult] = []
    for condition in CONDITIONS:
        for offset in range(args.seed_count):
            results.append(run_trial(args, condition, args.seed + offset))

    aggregates = {
        condition: aggregate_condition(
            [result for result in results if result.condition == condition]
        )
        for condition in CONDITIONS
    }
    checks = required_checks(results, aggregates)
    decision = "PASS_R502_E4_ASAS_GATE" if all(checks.values()) else "FAIL_R502_E4_ASAS_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU adjoint-state algebraic sparsity gate only; not "
            "heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_E4_ASAS_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_dense_fma_issue_slots_mean={positive['dense_fma_issue_slots_mean']:.0f} "
        f"positive_cotangent_l2_error_max_max={positive['cotangent_l2_error_max_max']:.12e} "
        f"positive_parameter_gradient_l2_error_max={positive['parameter_gradient_l2_error_max']:.12e} "
        f"positive_mismatch_fraction_max={positive['mismatch_fraction_max']:.12e} "
        f"positive_zero_fraction_mean={positive['zero_fraction_mean']:.6f} "
        f"positive_issue_slot_ratio_mean={positive['issue_slot_ratio_mean']:.6f} "
        f"positive_projected_recurrent_multiplier_mean={positive['projected_recurrent_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
