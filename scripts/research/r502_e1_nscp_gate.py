#!/usr/bin/env python3
"""R502 E1 null-state cotangent pruning gate.

This CPU diagnostic implements the R502 Efficiency finding:
"Null-State Cotangent Pruning (NSCP)".

The gate constructs diagonal ternary recurrent cotangent propagation. A positive
case must match the dense backward recurrence exactly while skipping recurrent
cotangent reads for dimensions where A_i = 0. Those dimensions satisfy
lambda[t, i] = local_grad[t, i] exactly because the temporal chain is severed.

Controls exercise the declared risks: an incorrect zero mask drops live
cotangent chains, a stale sparsity map applies the wrong branch after A changes,
a recurrence with no zero states has no priced saving, and thread-divergence
overhead can erase the compute denominator.

This is a synthetic CPU cotangent-equivalence and compute-ratio diagnostic, not
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


SCHEMA = "r502_e1_nscp_gate.v1"
CONDITIONS = (
    "positive",
    "wrong_zero_mask_fail",
    "stale_sparsity_map_fail",
    "no_zero_state_fail",
    "divergence_cost_fail",
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
    zero_fraction: float
    dense_read_fraction: float
    pruned_read_fraction: float
    residual_fraction: float
    cotangent_l2_error_max: float
    parameter_gradient_l2_error: float
    temporary_vram_ratio: float
    priced_compute_ratio: float
    projected_compute_saving_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50201)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sequence-len", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--zero-fraction", type=float, default=0.375)
    parser.add_argument("--max-residual-fraction", type=float, default=0.0)
    parser.add_argument("--max-cotangent-l2-error", type=float, default=1.0e-12)
    parser.add_argument("--max-parameter-gradient-l2-error", type=float, default=1.0e-12)
    parser.add_argument("--min-zero-fraction", type=float, default=0.10)
    parser.add_argument("--max-priced-compute-ratio", type=float, default=0.80)
    parser.add_argument("--branch-overhead", type=float, default=0.055)
    parser.add_argument("--expensive-branch-overhead", type=float, default=0.34)
    return parser.parse_args()


def make_diag(args: argparse.Namespace, rng: np.random.Generator, condition: str) -> np.ndarray:
    zero_fraction = 0.0 if condition == "no_zero_state_fail" else args.zero_fraction
    zero_count = int(round(zero_fraction * args.state_dim))
    diag = np.empty(args.state_dim, dtype=np.float64)
    diag[:zero_count] = 0.0
    diag[zero_count:] = rng.choice(np.array([-1.0, 1.0]), size=args.state_dim - zero_count)
    rng.shuffle(diag)
    return diag


def dense_cotangents(diag: np.ndarray, local_grads: np.ndarray) -> np.ndarray:
    sequence_len, state_dim = local_grads.shape
    lambdas = np.zeros((sequence_len + 1, state_dim), dtype=np.float64)
    for t in reversed(range(sequence_len)):
        lambdas[t] = diag * lambdas[t + 1] + local_grads[t]
    return lambdas


def pruned_cotangents(
    diag: np.ndarray,
    local_grads: np.ndarray,
    zero_mask: np.ndarray,
) -> np.ndarray:
    sequence_len, state_dim = local_grads.shape
    lambdas = np.zeros((sequence_len + 1, state_dim), dtype=np.float64)
    nonzero_mask = ~zero_mask
    for t in reversed(range(sequence_len)):
        lambdas[t, zero_mask] = local_grads[t, zero_mask]
        lambdas[t, nonzero_mask] = (
            diag[nonzero_mask] * lambdas[t + 1, nonzero_mask]
            + local_grads[t, nonzero_mask]
        )
    return lambdas


def parameter_gradient(lambdas: np.ndarray, states: np.ndarray) -> np.ndarray:
    return np.sum(lambdas[1:] * states[:-1], axis=0)


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    diag = make_diag(args, rng, condition)
    true_zero_mask = diag == 0.0
    zero_mask = true_zero_mask.copy()

    if condition == "wrong_zero_mask_fail":
        nonzero_indices = np.flatnonzero(~true_zero_mask)
        zero_mask[nonzero_indices[: max(1, len(nonzero_indices) // 8)]] = True
    elif condition == "stale_sparsity_map_fail":
        stale_diag = diag.copy()
        nonzero_indices = np.flatnonzero(~true_zero_mask)
        stale_diag[nonzero_indices[: max(1, len(nonzero_indices) // 6)]] = 0.0
        zero_mask = stale_diag == 0.0

    local_grads = rng.normal(scale=0.45, size=(args.sequence_len, args.state_dim))
    states = rng.normal(scale=0.60, size=(args.sequence_len + 1, args.state_dim))
    dense = dense_cotangents(diag, local_grads)
    pruned = pruned_cotangents(diag, local_grads, zero_mask)
    errors = pruned - dense
    cotangent_l2_error_max = float(np.max(np.linalg.norm(errors, axis=1)))
    residual_fraction = float(np.mean(np.abs(errors) > 1.0e-12))
    grad_error = float(np.linalg.norm(parameter_gradient(pruned, states) - parameter_gradient(dense, states)))

    dense_read_fraction = 1.0
    pruned_read_fraction = float(np.mean(~zero_mask))
    zero_fraction = float(np.mean(true_zero_mask))
    temporary_vram_ratio = pruned_read_fraction
    overhead = (
        args.expensive_branch_overhead
        if condition == "divergence_cost_fail"
        else args.branch_overhead
    )
    priced_compute_ratio = pruned_read_fraction + overhead
    projected_multiplier = 1.0 / priced_compute_ratio

    failed: list[str] = []
    if not finite(
        zero_fraction,
        dense_read_fraction,
        pruned_read_fraction,
        residual_fraction,
        cotangent_l2_error_max,
        grad_error,
        temporary_vram_ratio,
        priced_compute_ratio,
        projected_multiplier,
    ):
        failed.append("non_finite_metric")
    if residual_fraction > args.max_residual_fraction:
        failed.append("residual_fraction")
    if cotangent_l2_error_max > args.max_cotangent_l2_error:
        failed.append("cotangent_error")
    if grad_error > args.max_parameter_gradient_l2_error:
        failed.append("parameter_gradient")
    if zero_fraction < args.min_zero_fraction:
        failed.append("no_zero_savings")
    if priced_compute_ratio > args.max_priced_compute_ratio:
        failed.append("compute_price")

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
        zero_fraction=zero_fraction,
        dense_read_fraction=dense_read_fraction,
        pruned_read_fraction=pruned_read_fraction,
        residual_fraction=residual_fraction,
        cotangent_l2_error_max=float(cotangent_l2_error_max),
        parameter_gradient_l2_error=float(grad_error),
        temporary_vram_ratio=float(temporary_vram_ratio),
        priced_compute_ratio=float(priced_compute_ratio),
        projected_compute_saving_multiplier=float(projected_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "sequence_len",
        "state_dim",
        "zero_fraction",
        "dense_read_fraction",
        "pruned_read_fraction",
        "residual_fraction",
        "cotangent_l2_error_max",
        "parameter_gradient_l2_error",
        "temporary_vram_ratio",
        "priced_compute_ratio",
        "projected_compute_saving_multiplier",
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
    wrong = [result for result in results if result.condition == "wrong_zero_mask_fail"]
    stale = [result for result in results if result.condition == "stale_sparsity_map_fail"]
    no_zero = [result for result in results if result.condition == "no_zero_state_fail"]
    cost = [result for result in results if result.condition == "divergence_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_gradient_equivalence": (
            aggregates["positive"]["residual_fraction_max"] == 0.0
            and aggregates["positive"]["cotangent_l2_error_max_max"] <= 1.0e-12
            and aggregates["positive"]["parameter_gradient_l2_error_max"] <= 1.0e-12
        ),
        "positive_condition_prunes_zero_states": (
            aggregates["positive"]["zero_fraction_mean"] >= 0.10
            and aggregates["positive"]["pruned_read_fraction_mean"]
            < aggregates["positive"]["dense_read_fraction_mean"]
        ),
        "positive_condition_priced": (
            aggregates["positive"]["priced_compute_ratio_mean"] <= 0.80
        ),
        "wrong_mask_control_all_fail_gradient": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "residual_fraction" in result.fail_reason
                or "cotangent_error" in result.fail_reason
                or "parameter_gradient" in result.fail_reason
            )
            for result in wrong
        ),
        "stale_map_control_all_fail_gradient": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "residual_fraction" in result.fail_reason
                or "cotangent_error" in result.fail_reason
                or "parameter_gradient" in result.fail_reason
            )
            for result in stale
        ),
        "no_zero_control_all_fail_savings": all(
            (not result.passed)
            and result.fail_reason is not None
            and ("no_zero_savings" in result.fail_reason or "compute_price" in result.fail_reason)
            for result in no_zero
        ),
        "cost_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "compute_price" in result.fail_reason
            for result in cost
        ),
        "all_conditions_identified": all(result.identifying_correct for result in results),
    }


def main() -> int:
    args = parse_args()
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
    decision = "PASS_R502_E1_NSCP_GATE" if all(checks.values()) else "FAIL_R502_E1_NSCP_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU null-state cotangent pruning gate only; not heldout "
            "CE progress, not real corpus token efficiency, not GPU training "
            "speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_E1_NSCP_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_zero_fraction_mean={positive['zero_fraction_mean']:.6f} "
        f"positive_residual_fraction_max={positive['residual_fraction_max']:.12e} "
        f"positive_cotangent_l2_error_max_max={positive['cotangent_l2_error_max_max']:.12e} "
        f"positive_parameter_gradient_l2_error_max={positive['parameter_gradient_l2_error_max']:.12e} "
        f"positive_pruned_read_fraction_mean={positive['pruned_read_fraction_mean']:.6f} "
        f"positive_priced_compute_ratio_mean={positive['priced_compute_ratio_mean']:.6f} "
        f"positive_projected_compute_saving_multiplier_mean={positive['projected_compute_saving_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
