#!/usr/bin/env python3
"""R501 S56 quotient-state transition skipping gate.

This CPU diagnostic implements the R501 Speed Paradigm 8 finding:
"Quotient-State Transition Skipping (QSTS)".

For semantically null or padding spans, the token-conditioned SSM transition is
constant and the skipped interior has no objective source. The candidate
replaces dense per-token transition/backward work with an exact macro transition

    h_{t+k} = A^k h_t + sum_{j=0}^{k-1} A^j b
    lambda_t = (A^k)^T lambda_{t+k}

and only survives when boundary state, boundary cotangent, and objective values
match the dense path while the priced macro route is cheaper than dense
transition-by-transition backward evaluation.

Controls exercise the declared risks: false quotient grouping over variable
tokens breaks exact macro equivalence, hidden interior CE sources break the
no-source quotient contract, and fragmented null spans recover dense cost.

This is a synthetic CPU diagnostic, not heldout CE progress, GPU training-speed
evidence, or production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s56_qsts_gate.v1"
CONDITIONS = (
    "positive",
    "variable_token_fail",
    "interior_source_fail",
    "fragmented_price_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    steps: int
    state_dim: int
    run_length: int
    null_run_count: int
    skipped_transition_fraction: float
    macro_boundary_state_error: float
    macro_boundary_lambda_error: float
    objective_relative_error: float
    transition_class_mismatch_fraction: float
    interior_source_debt_ratio: float
    dense_transition_work_fraction: float
    macro_exponentiation_work_fraction: float
    audit_cost_ratio: float
    priced_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50156)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--state-dim", type=int, default=14)
    parser.add_argument("--observable-dim", type=int, default=5)
    parser.add_argument("--run-length", type=int, default=8)
    parser.add_argument("--max-state-error", type=float, default=1.0e-10)
    parser.add_argument("--max-lambda-error", type=float, default=1.0e-10)
    parser.add_argument("--max-objective-error", type=float, default=1.0e-10)
    parser.add_argument("--max-transition-mismatch", type=float, default=0.0)
    parser.add_argument("--max-interior-source-debt-ratio", type=float, default=1.0e-10)
    parser.add_argument("--min-skipped-transition-fraction", type=float, default=0.60)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.85)
    parser.add_argument("--exponentiation-cost-per-run", type=float, default=1.15)
    parser.add_argument("--audit-cost-ratio", type=float, default=0.045)
    return parser.parse_args()


def stable_matrix(rng: np.random.Generator, dim: int, radius: float) -> np.ndarray:
    raw = rng.normal(size=(dim, dim))
    q, _ = np.linalg.qr(raw)
    eig = rng.uniform(0.35, radius, size=dim)
    signs = rng.choice(np.array([-1.0, 1.0]), size=dim)
    return (q @ np.diag(eig * signs) @ q.T).astype(np.float64)


def geometric_offset(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    acc = np.zeros_like(b, dtype=np.float64)
    power = np.eye(a.shape[0], dtype=np.float64)
    for _ in range(k):
        acc += power @ b
        power = a @ power
    return acc


def build_null_mask(args: argparse.Namespace, condition: str) -> np.ndarray:
    mask = np.zeros(args.steps, dtype=bool)
    if condition == "fragmented_price_fail":
        # Short, frequent spans still satisfy quotient exactness but fail price.
        for start in range(1, args.steps, 4):
            mask[start : min(start + 1, args.steps)] = True
        return mask

    cycle = args.run_length + 2
    for start in range(1, args.steps, cycle):
        end = min(start + args.run_length, args.steps)
        mask[start:end] = True
    return mask


def null_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    t = 0
    while t < mask.size:
        if not mask[t]:
            t += 1
            continue
        start = t
        while t < mask.size and mask[t]:
            t += 1
        runs.append((start, t))
    return runs


def make_fixture(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[int, int]],
]:
    if args.observable_dim >= args.state_dim:
        raise ValueError("--observable-dim must be smaller than --state-dim")

    null_mask = build_null_mask(args, condition)
    runs = null_runs(null_mask)

    a_null = stable_matrix(rng, args.state_dim, radius=0.72)
    b_null = rng.normal(scale=0.05, size=args.state_dim).astype(np.float64)

    a_seq = np.zeros((args.steps, args.state_dim, args.state_dim), dtype=np.float64)
    b_seq = np.zeros((args.steps, args.state_dim), dtype=np.float64)
    for t in range(args.steps):
        if null_mask[t]:
            a_seq[t] = a_null
            b_seq[t] = b_null
        else:
            a_seq[t] = stable_matrix(rng, args.state_dim, radius=0.68)
            b_seq[t] = rng.normal(scale=0.14, size=args.state_dim)

    if condition == "variable_token_fail":
        # The detector has incorrectly grouped a token-varying span as a single
        # null quotient. Dense uses the true per-token maps; the candidate macro
        # uses the first map in each grouped span and should fail exactness.
        for run_index, (start, end) in enumerate(runs):
            if run_index % 2 == 0:
                for t in range(start + 1, end):
                    a_seq[t] = a_null + rng.normal(scale=0.018, size=a_null.shape)
                    b_seq[t] = b_null + rng.normal(scale=0.060, size=b_null.shape)

    c = np.zeros((args.observable_dim, args.state_dim), dtype=np.float64)
    c[:, : args.observable_dim] = np.eye(args.observable_dim, dtype=np.float64)
    targets = rng.normal(scale=0.25, size=(args.steps + 1, args.observable_dim))
    h0 = rng.normal(scale=0.35, size=args.state_dim).astype(np.float64)
    return a_seq, b_seq, c, targets, h0, null_mask, runs


def dense_forward(a_seq: np.ndarray, b_seq: np.ndarray, h0: np.ndarray) -> np.ndarray:
    steps = a_seq.shape[0]
    h = np.zeros((steps + 1, h0.size), dtype=np.float64)
    h[0] = h0
    for t in range(steps):
        h[t + 1] = a_seq[t] @ h[t] + b_seq[t]
    return h


def source_and_objective(
    h: np.ndarray,
    c: np.ndarray,
    targets: np.ndarray,
    null_mask: np.ndarray,
    condition: str,
) -> tuple[np.ndarray, float, np.ndarray]:
    steps = null_mask.size
    source = np.zeros_like(h, dtype=np.float64)
    source_mask = np.zeros(steps + 1, dtype=bool)
    for t in range(1, steps + 1):
        transition_is_null = null_mask[t - 1]
        has_source = not transition_is_null
        if condition == "interior_source_fail" and transition_is_null:
            has_source = True
        if not has_source:
            continue
        residual = c @ h[t] - targets[t]
        source[t] = c.T @ residual
        source_mask[t] = True
    losses = 0.5 * np.sum((h @ c.T - targets) ** 2, axis=1)
    objective = float(np.sum(losses[source_mask]))
    return source, objective, source_mask


def dense_backward(a_seq: np.ndarray, source: np.ndarray) -> np.ndarray:
    steps = a_seq.shape[0]
    lambdas = np.zeros_like(source, dtype=np.float64)
    lambdas[steps] = source[steps]
    for t in range(steps - 1, -1, -1):
        lambdas[t] = source[t] + a_seq[t].T @ lambdas[t + 1]
    return lambdas


def macro_forward(
    a_seq: np.ndarray,
    b_seq: np.ndarray,
    h0: np.ndarray,
    null_mask: np.ndarray,
    runs: list[tuple[int, int]],
) -> np.ndarray:
    steps = a_seq.shape[0]
    run_by_start = {start: end for start, end in runs}
    h = np.zeros((steps + 1, h0.size), dtype=np.float64)
    h[0] = h0
    t = 0
    while t < steps:
        if null_mask[t]:
            end = run_by_start[t]
            k = end - t
            a0 = a_seq[t]
            b0 = b_seq[t]
            h[end] = np.linalg.matrix_power(a0, k) @ h[t] + geometric_offset(a0, b0, k)
            t = end
        else:
            h[t + 1] = a_seq[t] @ h[t] + b_seq[t]
            t += 1
    return h


def macro_backward(
    a_seq: np.ndarray,
    source: np.ndarray,
    null_mask: np.ndarray,
    runs: list[tuple[int, int]],
) -> np.ndarray:
    steps = a_seq.shape[0]
    run_by_end = {end: start for start, end in runs}
    lambdas = np.zeros_like(source, dtype=np.float64)
    lambdas[steps] = source[steps]
    t = steps - 1
    while t >= 0:
        if null_mask[t]:
            end = t + 1
            while end not in run_by_end:
                end += 1
            start = run_by_end[end]
            k = end - start
            a0 = a_seq[start]
            lambdas[start] = source[start] + np.linalg.matrix_power(a0, k).T @ lambdas[end]
            t = start - 1
        else:
            lambdas[t] = source[t] + a_seq[t].T @ lambdas[t + 1]
            t -= 1
    return lambdas


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    numerator = float(np.linalg.norm(a - b))
    denominator = max(float(np.linalg.norm(a)), 1.0e-30)
    return numerator / denominator


def transition_mismatch_fraction(
    a_seq: np.ndarray,
    b_seq: np.ndarray,
    runs: list[tuple[int, int]],
) -> float:
    checked = 0
    mismatched = 0
    for start, end in runs:
        a0 = a_seq[start]
        b0 = b_seq[start]
        for t in range(start, end):
            checked += 1
            if not (
                np.allclose(a_seq[t], a0, rtol=0.0, atol=1.0e-12)
                and np.allclose(b_seq[t], b0, rtol=0.0, atol=1.0e-12)
            ):
                mismatched += 1
    return float(mismatched / max(checked, 1))


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    a_seq, b_seq, c, targets, h0, null_mask, runs = make_fixture(rng, args, condition)
    h_dense = dense_forward(a_seq, b_seq, h0)
    dense_source, dense_objective, dense_source_mask = source_and_objective(
        h_dense,
        c,
        targets,
        null_mask,
        condition,
    )
    dense_lambdas = dense_backward(a_seq, dense_source)

    h_macro = macro_forward(a_seq, b_seq, h0, null_mask, runs)
    candidate_condition = "positive" if condition != "interior_source_fail" else condition
    candidate_source, candidate_objective, _ = source_and_objective(
        h_macro,
        c,
        targets,
        null_mask,
        candidate_condition,
    )
    lambda_macro = macro_backward(a_seq, candidate_source, null_mask, runs)

    boundary_state_errors = []
    boundary_lambda_errors = []
    for start, end in runs:
        boundary_state_errors.append(relative_error(h_dense[end], h_macro[end]))
        boundary_lambda_errors.append(relative_error(dense_lambdas[start], lambda_macro[start]))

    objective_relative_error = abs(dense_objective - candidate_objective) / max(
        abs(dense_objective),
        1.0e-30,
    )
    transition_mismatch = transition_mismatch_fraction(a_seq, b_seq, runs)
    skipped_transitions = int(np.count_nonzero(null_mask))
    interior_source_mask = np.zeros(args.steps + 1, dtype=bool)
    for start, end in runs:
        interior_source_mask[start + 1 : end + 1] = True
    interior_source_norm = float(np.linalg.norm(dense_source[interior_source_mask]))
    total_source_norm = float(np.linalg.norm(dense_source[dense_source_mask]))
    interior_source_debt_ratio = interior_source_norm / max(total_source_norm, 1.0e-30)

    dense_transition_work_fraction = (args.steps - skipped_transitions) / args.steps
    macro_exponentiation_work_fraction = (
        len(runs) * args.exponentiation_cost_per_run / args.steps
    )
    priced_work_ratio = (
        dense_transition_work_fraction
        + macro_exponentiation_work_fraction
        + args.audit_cost_ratio
    )
    projected_speedup = 1.0 / priced_work_ratio if priced_work_ratio > 0.0 else math.inf

    macro_boundary_state_error = max(boundary_state_errors) if boundary_state_errors else 0.0
    macro_boundary_lambda_error = max(boundary_lambda_errors) if boundary_lambda_errors else 0.0
    skipped_transition_fraction = skipped_transitions / args.steps

    failed: list[str] = []
    if not math.isfinite(macro_boundary_state_error) or not math.isfinite(macro_boundary_lambda_error):
        failed.append("non_finite_boundary_error")
    if macro_boundary_state_error > args.max_state_error:
        failed.append("state_equivalence")
    if macro_boundary_lambda_error > args.max_lambda_error:
        failed.append("lambda_equivalence")
    if objective_relative_error > args.max_objective_error:
        failed.append("objective_equivalence")
    if transition_mismatch > args.max_transition_mismatch:
        failed.append("transition_class_certification")
    if interior_source_debt_ratio > args.max_interior_source_debt_ratio:
        failed.append("interior_source_debt")
    if skipped_transition_fraction < args.min_skipped_transition_fraction:
        failed.append("skip_fraction")
    if priced_work_ratio > args.max_priced_work_ratio:
        failed.append("price")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=(passed == expected_pass),
        fail_reason=";".join(failed) if failed else None,
        steps=args.steps,
        state_dim=args.state_dim,
        run_length=args.run_length,
        null_run_count=len(runs),
        skipped_transition_fraction=float(skipped_transition_fraction),
        macro_boundary_state_error=float(macro_boundary_state_error),
        macro_boundary_lambda_error=float(macro_boundary_lambda_error),
        objective_relative_error=float(objective_relative_error),
        transition_class_mismatch_fraction=float(transition_mismatch),
        interior_source_debt_ratio=float(interior_source_debt_ratio),
        dense_transition_work_fraction=float(dense_transition_work_fraction),
        macro_exponentiation_work_fraction=float(macro_exponentiation_work_fraction),
        audit_cost_ratio=float(args.audit_cost_ratio),
        priced_work_ratio=float(priced_work_ratio),
        projected_speedup=float(projected_speedup),
    )


def aggregate(results: list[TrialResult]) -> dict[str, Any]:
    by_condition: dict[str, list[TrialResult]] = {
        condition: [result for result in results if result.condition == condition]
        for condition in CONDITIONS
    }
    aggregates: dict[str, Any] = {}
    numeric_fields = (
        "null_run_count",
        "skipped_transition_fraction",
        "macro_boundary_state_error",
        "macro_boundary_lambda_error",
        "objective_relative_error",
        "transition_class_mismatch_fraction",
        "interior_source_debt_ratio",
        "dense_transition_work_fraction",
        "macro_exponentiation_work_fraction",
        "audit_cost_ratio",
        "priced_work_ratio",
        "projected_speedup",
    )
    for condition, condition_results in by_condition.items():
        aggregates[condition] = {
            "count": len(condition_results),
            "passed_count": sum(result.passed for result in condition_results),
            "identifying_correct_count": sum(
                result.identifying_correct for result in condition_results
            ),
            "fail_reasons": sorted(
                {result.fail_reason for result in condition_results if result.fail_reason}
            ),
        }
        for field in numeric_fields:
            values = [float(getattr(result, field)) for result in condition_results]
            if not values:
                continue
            aggregates[condition][f"{field}_mean"] = float(np.mean(values))
            aggregates[condition][f"{field}_max"] = float(np.max(values))
            aggregates[condition][f"{field}_min"] = float(np.min(values))
    return aggregates


def all_finite(results: list[TrialResult]) -> bool:
    for result in results:
        for key, value in asdict(result).items():
            if isinstance(value, float) and not math.isfinite(value):
                return False
    return True


def required_checks(results: list[TrialResult], aggregates: dict[str, Any]) -> dict[str, bool]:
    positive = [result for result in results if result.condition == "positive"]
    variable = [result for result in results if result.condition == "variable_token_fail"]
    interior = [result for result in results if result.condition == "interior_source_fail"]
    fragmented = [result for result in results if result.condition == "fragmented_price_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_macro": (
            aggregates["positive"]["macro_boundary_state_error_max"] <= 1.0e-10
            and aggregates["positive"]["macro_boundary_lambda_error_max"] <= 1.0e-10
            and aggregates["positive"]["objective_relative_error_max"] <= 1.0e-10
        ),
        "positive_condition_skip_and_price": (
            aggregates["positive"]["skipped_transition_fraction_mean"] >= 0.60
            and aggregates["positive"]["priced_work_ratio_mean"] <= 0.85
        ),
        "variable_control_all_fail_equivalence": all(
            (not result.passed)
            and result.fail_reason is not None
            and "transition_class_certification" in result.fail_reason
            and "state_equivalence" in result.fail_reason
            for result in variable
        ),
        "interior_control_all_fail_source_debt": all(
            (not result.passed)
            and result.fail_reason is not None
            and "interior_source_debt" in result.fail_reason
            and "lambda_equivalence" in result.fail_reason
            for result in interior
        ),
        "fragmented_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "skip_fraction" in result.fail_reason
            and "price" in result.fail_reason
            for result in fragmented
        ),
        "all_conditions_identified": all(result.identifying_correct for result in results),
    }


def main() -> int:
    args = parse_args()
    results: list[TrialResult] = []
    for condition in CONDITIONS:
        for offset in range(args.seed_count):
            results.append(run_trial(args, condition, args.seed + 1000 * offset))

    aggregates = aggregate(results)
    checks = required_checks(results, aggregates)
    decision = "PASS_R501_S56_QSTS_GATE" if all(checks.values()) else "FAIL_R501_S56_QSTS_GATE"
    payload = {
        "schema": SCHEMA,
        "decision": decision,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "conditions": CONDITIONS,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU quotient-state transition skipping gate only; not "
            "heldout CE progress, not GPU training speed, and not production "
            "v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S56_QSTS_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_skipped_transition_fraction_mean={positive['skipped_transition_fraction_mean']:.6f} "
        f"positive_state_error_max={positive['macro_boundary_state_error_max']:.6e} "
        f"positive_lambda_error_max={positive['macro_boundary_lambda_error_max']:.6e} "
        f"positive_objective_error_max={positive['objective_relative_error_max']:.6e} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
