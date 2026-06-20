#!/usr/bin/env python3
"""R501 S76 orthogonal-credit parameter realization gate.

This CPU diagnostic implements the R501 Quality Math 13 finding:
"Orthogonal-Credit Parameter Realization (OCPR)".

The gate constructs a parameter vector with an old routing-sensitive token and
a new content update. A positive case must project the new-token update into
the null space of the routing basis, yielding zero old-token CE degradation
while still improving the new-token realization loss.

Controls exercise the declared risks: an unprojected update causes forgetting,
a stale routing basis leaves interference in the true routing subspace, a
near-full routing subspace over-constrains learning, and an expensive projection
erases the priced denominator.

This is a synthetic CPU orthogonal-credit diagnostic, not heldout CE progress,
real corpus token efficiency, GPU training-speed evidence, or production v12
integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s76_ocpr_gate.v1"
CONDITIONS = (
    "positive",
    "no_projection_fail",
    "stale_routing_basis_fail",
    "overconstrained_space_fail",
    "projection_cost_fail",
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
    routing_rank: int
    nullspace_rank: int
    old_ce_degradation: float
    baseline_old_ce_degradation: float
    content_loss_delta: float
    baseline_content_loss_delta: float
    orthogonality_residual: float
    update_norm: float
    priced_work_ratio: float
    projected_token_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50176)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--param-dim", type=int, default=32)
    parser.add_argument("--routing-rank", type=int, default=8)
    parser.add_argument("--overconstrained-routing-rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.70)
    parser.add_argument("--content-scale", type=float, default=1.0)
    parser.add_argument("--routing-interference-scale", type=float, default=0.65)
    parser.add_argument("--max-old-ce-degradation", type=float, default=1.0e-12)
    parser.add_argument("--min-baseline-old-ce-degradation", type=float, default=0.05)
    parser.add_argument("--min-content-loss-reduction", type=float, default=0.35)
    parser.add_argument("--max-orthogonality-residual", type=float, default=1.0e-10)
    parser.add_argument("--min-nullspace-rank", type=int, default=4)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.82)
    parser.add_argument("--priced-work-ratio", type=float, default=0.76)
    parser.add_argument("--expensive-priced-work-ratio", type=float, default=0.88)
    return parser.parse_args()


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1.0e-30)


def orthonormal_rows(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    matrix = rng.normal(size=(cols, rows))
    q, _r = np.linalg.qr(matrix)
    return q[:, :rows].T


def row_orthonormalize(matrix: np.ndarray) -> np.ndarray:
    q, _r = np.linalg.qr(matrix.T)
    return q[:, : matrix.shape[0]].T


def nullspace(row_basis: np.ndarray, tol: float = 1.0e-10) -> np.ndarray:
    _u, singular_values, vh = np.linalg.svd(row_basis, full_matrices=True)
    rank = int(np.sum(singular_values > tol))
    return vh[rank:]


def project_null(update: np.ndarray, row_basis: np.ndarray) -> np.ndarray:
    return update - row_basis.T @ (row_basis @ update)


def old_ce_loss(weights: np.ndarray, routing_basis: np.ndarray, old_logits: np.ndarray) -> float:
    residual = routing_basis @ weights - old_logits
    return 0.5 * float(np.dot(residual, residual))


def content_loss(weights: np.ndarray, target: np.ndarray) -> float:
    residual = weights - target
    return 0.5 * float(np.dot(residual, residual))


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def make_problem(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    routing_rank = (
        args.overconstrained_routing_rank
        if condition == "overconstrained_space_fail"
        else args.routing_rank
    )
    routing_basis = orthonormal_rows(rng, routing_rank, args.param_dim)
    null_basis = nullspace(routing_basis)

    weights = rng.normal(scale=0.20, size=args.param_dim)
    old_logits = routing_basis @ weights

    if null_basis.size:
        coeff = rng.normal(size=null_basis.shape[0])
        content_delta = args.content_scale * normalize(coeff @ null_basis)
    else:
        content_delta = np.zeros(args.param_dim, dtype=np.float64)

    route_coeff = rng.normal(size=routing_rank)
    routing_delta = args.routing_interference_scale * normalize(route_coeff @ routing_basis)
    target = weights + content_delta + routing_delta

    projection_basis = routing_basis
    if condition == "stale_routing_basis_fail":
        if null_basis.size:
            stale = routing_basis.copy()
            stale[0] = normalize(0.82 * routing_basis[0] + 0.58 * null_basis[0])
            projection_basis = row_orthonormalize(stale)

    return weights, routing_basis, projection_basis, old_logits, target


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    weights, routing_basis, projection_basis, old_logits, target = make_problem(
        rng,
        args,
        condition,
    )
    routing_rank = routing_basis.shape[0]
    null_rank = nullspace(routing_basis).shape[0]

    before_old = old_ce_loss(weights, routing_basis, old_logits)
    before_content = content_loss(weights, target)
    raw_update = -args.learning_rate * (weights - target)
    baseline_weights = weights + raw_update
    baseline_old_delta = old_ce_loss(baseline_weights, routing_basis, old_logits) - before_old
    baseline_content_delta = content_loss(baseline_weights, target) - before_content

    if condition == "no_projection_fail":
        update = raw_update
    else:
        update = project_null(raw_update, projection_basis)

    candidate_weights = weights + update
    old_delta = old_ce_loss(candidate_weights, routing_basis, old_logits) - before_old
    content_delta = content_loss(candidate_weights, target) - before_content
    update_norm = float(np.linalg.norm(update))
    orthogonality_residual = float(np.linalg.norm(routing_basis @ update)) / max(update_norm, 1.0e-30)
    priced_work_ratio = (
        args.expensive_priced_work_ratio
        if condition == "projection_cost_fail"
        else args.priced_work_ratio
    )
    projected_multiplier = 1.0 / priced_work_ratio

    failed: list[str] = []
    if not finite(
        old_delta,
        baseline_old_delta,
        content_delta,
        baseline_content_delta,
        orthogonality_residual,
        update_norm,
        priced_work_ratio,
        projected_multiplier,
    ):
        failed.append("non_finite_metric")
    if old_delta > args.max_old_ce_degradation:
        failed.append("old_token_degradation")
    if baseline_old_delta < args.min_baseline_old_ce_degradation:
        failed.append("non_identifying_baseline")
    if -content_delta < args.min_content_loss_reduction:
        failed.append("new_learning")
    if orthogonality_residual > args.max_orthogonality_residual:
        failed.append("orthogonality")
    if null_rank < args.min_nullspace_rank:
        failed.append("nullspace_capacity")
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
        param_dim=args.param_dim,
        routing_rank=int(routing_rank),
        nullspace_rank=int(null_rank),
        old_ce_degradation=float(old_delta),
        baseline_old_ce_degradation=float(baseline_old_delta),
        content_loss_delta=float(content_delta),
        baseline_content_loss_delta=float(baseline_content_delta),
        orthogonality_residual=float(orthogonality_residual),
        update_norm=float(update_norm),
        priced_work_ratio=float(priced_work_ratio),
        projected_token_efficiency_multiplier=float(projected_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "param_dim",
        "routing_rank",
        "nullspace_rank",
        "old_ce_degradation",
        "baseline_old_ce_degradation",
        "content_loss_delta",
        "baseline_content_loss_delta",
        "orthogonality_residual",
        "update_norm",
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
    no_projection = [result for result in results if result.condition == "no_projection_fail"]
    stale = [result for result in results if result.condition == "stale_routing_basis_fail"]
    overconstrained = [result for result in results if result.condition == "overconstrained_space_fail"]
    cost = [result for result in results if result.condition == "projection_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_zero_old_degradation": (
            aggregates["positive"]["old_ce_degradation_max"] <= 1.0e-12
            and aggregates["positive"]["baseline_old_ce_degradation_min"] >= 0.05
        ),
        "positive_condition_learns_new_content": (
            aggregates["positive"]["content_loss_delta_max"] <= -0.35
        ),
        "positive_condition_orthogonal": (
            aggregates["positive"]["orthogonality_residual_max"] <= 1.0e-10
        ),
        "positive_condition_priced": (
            aggregates["positive"]["priced_work_ratio_mean"] <= 0.82
        ),
        "no_projection_control_all_fail_interference": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "old_token_degradation" in result.fail_reason
                or "orthogonality" in result.fail_reason
            )
            for result in no_projection
        ),
        "stale_basis_control_all_fail_interference": all(
            (not result.passed)
            and result.fail_reason is not None
            and "old_token_degradation" in result.fail_reason
            for result in stale
        ),
        "overconstrained_control_all_fail_learning": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "new_learning" in result.fail_reason
                or "nullspace_capacity" in result.fail_reason
            )
            for result in overconstrained
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
        "PASS_R501_S76_OCPR_GATE"
        if all(checks.values())
        else "FAIL_R501_S76_OCPR_GATE"
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
            "Synthetic CPU orthogonal-credit parameter realization gate only; "
            "not heldout CE progress, not real corpus token efficiency, not "
            "GPU training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S76_OCPR_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_old_ce_degradation_max={positive['old_ce_degradation_max']:.12f} "
        f"positive_baseline_old_ce_degradation_min={positive['baseline_old_ce_degradation_min']:.6f} "
        f"positive_content_loss_delta_mean={positive['content_loss_delta_mean']:.6f} "
        f"positive_orthogonality_residual_max={positive['orthogonality_residual_max']:.12f} "
        f"positive_nullspace_rank_mean={positive['nullspace_rank_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_token_efficiency_multiplier_mean={positive['projected_token_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
