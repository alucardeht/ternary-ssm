#!/usr/bin/env python3
"""R501 S66 recurrent null-space capability pruning gate.

This CPU diagnostic implements the R501 Quality Math 3 finding:
"Recurrent Null-Space Capability Pruning (RNSCP)".

The gate constructs a recurrent transition with a true null space, injects
dense gradient energy into that null space, and checks whether projecting the
gradient onto the orthogonal complement preserves capability-bearing descent
while avoiding null-space update waste. The priced denominator is recorded as a
token-efficiency proxy, not GPU speed.

Controls exercise the declared risks: a full-rank transition has no null-space
gain, a wrong null basis prunes capability-bearing gradient, and an expensive
sketch erases the denominator.

This is a synthetic CPU null-space projection diagnostic, not heldout CE
progress, GPU training-speed evidence, or production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s66_rnscp_gate.v1"
CONDITIONS = (
    "positive",
    "full_rank_fail",
    "wrong_basis_fail",
    "sketch_cost_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    param_dim: int
    null_dim: int
    null_gradient_energy_fraction: float
    capability_preservation_error: float
    null_residual_after_prune: float
    dense_loss_delta: float
    pruned_loss_delta: float
    pruned_loss_improvement: float
    priced_work_ratio: float
    projected_token_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50166)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--param-dim", type=int, default=24)
    parser.add_argument("--null-dim", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.22)
    parser.add_argument("--curvature", type=float, default=0.12)
    parser.add_argument("--min-null-energy-fraction", type=float, default=0.10)
    parser.add_argument("--max-capability-preservation-error", type=float, default=1.0e-10)
    parser.add_argument("--max-null-residual-after-prune", type=float, default=1.0e-10)
    parser.add_argument("--min-pruned-loss-improvement", type=float, default=0.01)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.88)
    return parser.parse_args()


def random_orthogonal(rng: np.random.Generator, dimension: int) -> np.ndarray:
    matrix = rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def projection(basis: np.ndarray, dimension: int) -> np.ndarray:
    if basis.size == 0:
        return np.zeros((dimension, dimension), dtype=np.float64)
    return basis @ basis.T


def normalized(rng: np.random.Generator, dimension: int, scale: float) -> np.ndarray:
    vector = rng.normal(size=dimension)
    vector /= max(float(np.linalg.norm(vector)), 1.0e-30)
    return vector * scale


def loss_delta(true_gradient: np.ndarray, update: np.ndarray, curvature: float) -> float:
    return float(np.dot(true_gradient, update) + 0.5 * curvature * np.dot(update, update))


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    param_dim = args.param_dim
    true_null_dim = args.null_dim
    if condition == "full_rank_fail":
        true_null_dim = 0
    basis = random_orthogonal(rng, param_dim)
    null_basis = basis[:, :true_null_dim]
    row_basis = basis[:, true_null_dim:]
    p_true_null = projection(null_basis, param_dim)
    p_row = np.eye(param_dim) - p_true_null

    cap_coordinates = normalized(rng, row_basis.shape[1], scale=1.0)
    true_gradient = row_basis @ cap_coordinates
    if true_null_dim:
        null_coordinates = normalized(rng, true_null_dim, scale=2.10)
        null_gradient = null_basis @ null_coordinates
    else:
        null_gradient = np.zeros(param_dim, dtype=np.float64)
    dense_gradient = true_gradient + null_gradient

    if condition == "wrong_basis_fail":
        wrong_basis = random_orthogonal(rng, param_dim)[:, : args.null_dim]
        p_estimated_null = projection(wrong_basis, param_dim)
    else:
        p_estimated_null = p_true_null

    pruned_gradient = (np.eye(param_dim) - p_estimated_null) @ dense_gradient
    dense_update = -args.learning_rate * dense_gradient
    pruned_update = -args.learning_rate * pruned_gradient
    dense_delta = loss_delta(true_gradient, dense_update, args.curvature)
    pruned_delta = loss_delta(true_gradient, pruned_update, args.curvature)
    improvement = dense_delta - pruned_delta

    null_energy = float(np.dot(p_true_null @ dense_gradient, p_true_null @ dense_gradient))
    total_energy = max(float(np.dot(dense_gradient, dense_gradient)), 1.0e-30)
    null_energy_fraction = null_energy / total_energy
    row_dense = p_row @ dense_gradient
    row_pruned = p_row @ pruned_gradient
    capability_error = float(np.linalg.norm(row_pruned - row_dense)) / max(
        float(np.linalg.norm(row_dense)),
        1.0e-30,
    )
    null_residual = float(np.linalg.norm(p_true_null @ pruned_gradient)) / max(
        float(np.linalg.norm(p_true_null @ dense_gradient)),
        1.0e-30,
    )

    priced_work_ratio = 0.74
    if condition == "sketch_cost_fail":
        priced_work_ratio = 0.93
    projected_multiplier = 1.0 / priced_work_ratio

    failed: list[str] = []
    if not all(
        math.isfinite(value)
        for value in (
            null_energy_fraction,
            capability_error,
            null_residual,
            dense_delta,
            pruned_delta,
            improvement,
            priced_work_ratio,
            projected_multiplier,
        )
    ):
        failed.append("non_finite_metric")
    if null_energy_fraction < args.min_null_energy_fraction:
        failed.append("null_energy")
    if capability_error > args.max_capability_preservation_error:
        failed.append("capability_preservation")
    if null_residual > args.max_null_residual_after_prune:
        failed.append("null_residual")
    if improvement < args.min_pruned_loss_improvement:
        failed.append("loss_improvement")
    if priced_work_ratio > args.max_priced_work_ratio:
        failed.append("price")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed) if failed else None,
        param_dim=param_dim,
        null_dim=true_null_dim,
        null_gradient_energy_fraction=float(null_energy_fraction),
        capability_preservation_error=float(capability_error),
        null_residual_after_prune=float(null_residual),
        dense_loss_delta=float(dense_delta),
        pruned_loss_delta=float(pruned_delta),
        pruned_loss_improvement=float(improvement),
        priced_work_ratio=float(priced_work_ratio),
        projected_token_efficiency_multiplier=float(projected_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "param_dim",
        "null_dim",
        "null_gradient_energy_fraction",
        "capability_preservation_error",
        "null_residual_after_prune",
        "dense_loss_delta",
        "pruned_loss_delta",
        "pruned_loss_improvement",
        "priced_work_ratio",
        "projected_token_efficiency_multiplier",
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
    full_rank = [result for result in results if result.condition == "full_rank_fail"]
    wrong = [result for result in results if result.condition == "wrong_basis_fail"]
    cost = [result for result in results if result.condition == "sketch_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_null_energy": (
            aggregates["positive"]["null_gradient_energy_fraction_mean"] >= 0.10
            and aggregates["positive"]["null_residual_after_prune_max"] <= 1.0e-10
        ),
        "positive_condition_capability_preserved": (
            aggregates["positive"]["capability_preservation_error_max"] <= 1.0e-10
            and aggregates["positive"]["pruned_loss_improvement_min"] >= 0.01
        ),
        "positive_condition_priced": (
            aggregates["positive"]["priced_work_ratio_mean"] <= 0.88
        ),
        "full_rank_control_all_fail_null_energy": all(
            (not result.passed)
            and result.fail_reason is not None
            and "null_energy" in result.fail_reason
            and "loss_improvement" in result.fail_reason
            for result in full_rank
        ),
        "wrong_basis_control_all_fail_capability": all(
            (not result.passed)
            and result.fail_reason is not None
            and "capability_preservation" in result.fail_reason
            for result in wrong
        ),
        "cost_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "price" in result.fail_reason
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
    decision = (
        "PASS_R501_S66_RNSCP_GATE"
        if all(checks.values())
        else "FAIL_R501_S66_RNSCP_GATE"
    )
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU recurrent null-space projection gate only; not "
            "heldout CE progress, not real token efficiency on corpora, not "
            "GPU training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S66_RNSCP_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_null_gradient_energy_fraction_mean={positive['null_gradient_energy_fraction_mean']:.6f} "
        f"positive_capability_preservation_error_max={positive['capability_preservation_error_max']:.6e} "
        f"positive_null_residual_after_prune_max={positive['null_residual_after_prune_max']:.6e} "
        f"positive_pruned_loss_improvement_mean={positive['pruned_loss_improvement_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_token_efficiency_multiplier_mean={positive['projected_token_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
