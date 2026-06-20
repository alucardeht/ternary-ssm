#!/usr/bin/env python3
"""R501 S81 exact equivalence quotient for contextual synonyms gate.

This CPU diagnostic implements the R501 Quality Paradigm 3 finding:
"Exact Equivalence Quotient for Contextual Synonyms (EEQ-CS)".

The gate constructs repeated contextual-synonym tokens whose SSM hidden states
fall inside an epsilon ball and whose CE cotangents are nearly equivalent. The
candidate pays one backward representative per quotient class and scales that
representative by class multiplicity. A positive case must preserve the dense
cotangent sum while reducing the paid class count.

Controls exercise the declared risks: hidden-state closeness can be a false
proxy for cotangent equivalence, an overly wide epsilon can merge different
states, low multiplicity can erase the denominator, multiplicity scaling can be
forgotten, and a costly quotient index can recover dense cost.

This is a synthetic CPU quotient-cotangent diagnostic, not heldout CE progress,
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


SCHEMA = "r501_s81_eeq_cs_gate.v1"
CONDITIONS = (
    "positive",
    "false_equivalence_fail",
    "epsilon_too_wide_fail",
    "low_multiplicity_fail",
    "bad_multiplicity_scaling_fail",
    "index_cost_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    token_count: int
    state_dim: int
    cotangent_dim: int
    quotient_class_count: int
    mean_multiplicity: float
    max_hidden_radius: float
    max_cotangent_drift_ratio: float
    quotient_gradient_relative_error: float
    quotient_gradient_cosine: float
    multiplicity_scaling_error_ratio: float
    dense_work_units: float
    quotient_work_units: float
    index_work_units: float
    priced_work_ratio: float
    projected_token_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50181)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--token-count", type=int, default=384)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--cotangent-dim", type=int, default=16)
    parser.add_argument("--class-count", type=int, default=96)
    parser.add_argument("--epsilon-h", type=float, default=0.025)
    parser.add_argument("--max-hidden-radius", type=float, default=0.025)
    parser.add_argument("--max-cotangent-drift", type=float, default=0.08)
    parser.add_argument("--max-gradient-relative-error", type=float, default=0.035)
    parser.add_argument("--min-gradient-cosine", type=float, default=0.999)
    parser.add_argument("--min-mean-multiplicity", type=float, default=2.0)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.45)
    parser.add_argument("--min-token-efficiency-multiplier", type=float, default=2.0)
    parser.add_argument("--index-overhead", type=float, default=0.06)
    parser.add_argument("--expensive-index-overhead", type=float, default=0.31)
    return parser.parse_args()


def normalized_rows(rng: np.random.Generator, count: int, dim: int, scale: float) -> np.ndarray:
    rows = rng.normal(size=(count, dim))
    norms = np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1.0e-30)
    return scale * rows / norms


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0e-30)
    return float(np.dot(a, b) / denom)


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def labels_for_condition(args: argparse.Namespace, condition: str) -> tuple[np.ndarray, int]:
    if condition == "low_multiplicity_fail":
        class_count = int(round(args.token_count * 0.84))
        labels = np.arange(args.token_count) % class_count
        return labels, class_count
    labels = np.arange(args.token_count) % args.class_count
    return labels, args.class_count


def make_fixture(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels, class_count = labels_for_condition(args, condition)
    map_matrix = rng.normal(scale=0.32, size=(args.cotangent_dim, args.state_dim))

    if condition == "epsilon_too_wide_fail":
        pair_count = max(1, class_count // 2)
        base_centers = normalized_rows(rng, pair_count, args.state_dim, scale=1.0)
        offsets = normalized_rows(rng, pair_count, args.state_dim, scale=0.060)
        centers = np.empty((class_count, args.state_dim), dtype=np.float64)
        centers[0::2] = base_centers[: centers[0::2].shape[0]]
        centers[1::2] = base_centers[: centers[1::2].shape[0]] + offsets[: centers[1::2].shape[0]]
    else:
        centers = normalized_rows(rng, class_count, args.state_dim, scale=1.0)

    states = centers[labels] + rng.normal(scale=0.0025, size=(args.token_count, args.state_dim))
    cotangent_centers = centers @ map_matrix.T
    cotangents = cotangent_centers[labels] + rng.normal(
        scale=0.0025,
        size=(args.token_count, args.cotangent_dim),
    )

    if condition == "false_equivalence_fail":
        offsets = normalized_rows(rng, args.token_count, args.cotangent_dim, scale=0.55)
        cotangents = cotangent_centers[labels] + offsets
    elif condition == "epsilon_too_wide_fail":
        drift = normalized_rows(rng, class_count, args.cotangent_dim, scale=0.42)
        cotangents = cotangent_centers[labels] + drift[labels]

    if condition == "epsilon_too_wide_fail":
        group_ids = labels // 2
    else:
        group_ids = labels
    return states, cotangents, group_ids


def group_metrics(
    states: np.ndarray,
    cotangents: np.ndarray,
    group_ids: np.ndarray,
    scale_multiplicity: bool,
) -> tuple[float, float, np.ndarray]:
    dense = np.sum(cotangents, axis=0)
    quotient = np.zeros_like(dense)
    max_hidden_radius = 0.0
    max_cotangent_drift = 0.0

    for group in np.unique(group_ids):
        idx = np.flatnonzero(group_ids == group)
        representative = idx[0]
        count = idx.size
        state_center = np.mean(states[idx], axis=0)
        hidden_radius = float(np.max(np.linalg.norm(states[idx] - state_center, axis=1)))
        max_hidden_radius = max(max_hidden_radius, hidden_radius)

        rep_cotangent = cotangents[representative]
        cotangent_norm = max(float(np.linalg.norm(rep_cotangent)), 1.0e-30)
        drift = float(np.max(np.linalg.norm(cotangents[idx] - rep_cotangent, axis=1))) / cotangent_norm
        max_cotangent_drift = max(max_cotangent_drift, drift)

        quotient += rep_cotangent * (count if scale_multiplicity else 1.0)

    return max_hidden_radius, max_cotangent_drift, quotient


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    states, cotangents, group_ids = make_fixture(rng, args, condition)
    scale_multiplicity = condition != "bad_multiplicity_scaling_fail"
    hidden_radius, cotangent_drift, quotient = group_metrics(
        states,
        cotangents,
        group_ids,
        scale_multiplicity,
    )

    dense = np.sum(cotangents, axis=0)
    dense_norm = max(float(np.linalg.norm(dense)), 1.0e-30)
    relative_error = float(np.linalg.norm(quotient - dense)) / dense_norm
    scaling_error = relative_error
    gradient_cosine = cosine(quotient, dense)
    class_count = int(np.unique(group_ids).size)
    mean_multiplicity = args.token_count / class_count

    dense_work = float(args.token_count)
    quotient_work = float(class_count)
    overhead = args.expensive_index_overhead if condition == "index_cost_fail" else args.index_overhead
    index_work = overhead * dense_work
    priced_work_ratio = (quotient_work + index_work) / dense_work
    retained_signal = max(0.0, 1.0 - relative_error)
    projected_multiplier = retained_signal / max(priced_work_ratio, 1.0e-30)

    failed: list[str] = []
    if not finite(
        mean_multiplicity,
        hidden_radius,
        cotangent_drift,
        relative_error,
        gradient_cosine,
        scaling_error,
        dense_work,
        quotient_work,
        index_work,
        priced_work_ratio,
        projected_multiplier,
    ):
        failed.append("non_finite_metric")
    if mean_multiplicity < args.min_mean_multiplicity:
        failed.append("low_multiplicity")
    if hidden_radius > args.max_hidden_radius:
        failed.append("hidden_radius")
    if cotangent_drift > args.max_cotangent_drift:
        failed.append("cotangent_drift")
    if relative_error > args.max_gradient_relative_error:
        failed.append("quotient_gradient_error")
    if gradient_cosine < args.min_gradient_cosine:
        failed.append("quotient_gradient_direction")
    if scaling_error > args.max_gradient_relative_error:
        failed.append("multiplicity_scaling")
    if priced_work_ratio > args.max_priced_work_ratio:
        failed.append("price")
    if projected_multiplier < args.min_token_efficiency_multiplier:
        failed.append("token_efficiency")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed) if failed else None,
        token_count=args.token_count,
        state_dim=args.state_dim,
        cotangent_dim=args.cotangent_dim,
        quotient_class_count=class_count,
        mean_multiplicity=float(mean_multiplicity),
        max_hidden_radius=float(hidden_radius),
        max_cotangent_drift_ratio=float(cotangent_drift),
        quotient_gradient_relative_error=float(relative_error),
        quotient_gradient_cosine=float(gradient_cosine),
        multiplicity_scaling_error_ratio=float(scaling_error),
        dense_work_units=float(dense_work),
        quotient_work_units=float(quotient_work),
        index_work_units=float(index_work),
        priced_work_ratio=float(priced_work_ratio),
        projected_token_efficiency_multiplier=float(projected_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "token_count",
        "state_dim",
        "cotangent_dim",
        "quotient_class_count",
        "mean_multiplicity",
        "max_hidden_radius",
        "max_cotangent_drift_ratio",
        "quotient_gradient_relative_error",
        "quotient_gradient_cosine",
        "multiplicity_scaling_error_ratio",
        "dense_work_units",
        "quotient_work_units",
        "index_work_units",
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
    false_eq = [result for result in results if result.condition == "false_equivalence_fail"]
    wide = [result for result in results if result.condition == "epsilon_too_wide_fail"]
    low = [result for result in results if result.condition == "low_multiplicity_fail"]
    bad_scale = [result for result in results if result.condition == "bad_multiplicity_scaling_fail"]
    cost = [result for result in results if result.condition == "index_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_equivalence_tight": (
            aggregates["positive"]["max_hidden_radius_max"] <= 0.025
            and aggregates["positive"]["max_cotangent_drift_ratio_max"] <= 0.08
        ),
        "positive_condition_gradient_preserved": (
            aggregates["positive"]["quotient_gradient_relative_error_max"] <= 0.035
            and aggregates["positive"]["quotient_gradient_cosine_min"] >= 0.999
        ),
        "positive_condition_scaled_by_multiplicity": (
            aggregates["positive"]["multiplicity_scaling_error_ratio_max"] <= 0.035
        ),
        "positive_condition_priced": (
            aggregates["positive"]["mean_multiplicity_mean"] >= 2.0
            and aggregates["positive"]["priced_work_ratio_mean"] <= 0.45
            and aggregates["positive"]["projected_token_efficiency_multiplier_mean"] >= 2.0
        ),
        "false_equivalence_control_all_fail_cotangent_or_gradient": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "cotangent_drift" in result.fail_reason
                or "quotient_gradient_error" in result.fail_reason
            )
            for result in false_eq
        ),
        "wide_epsilon_control_all_fail_radius_or_gradient": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "hidden_radius" in result.fail_reason
                or "cotangent_drift" in result.fail_reason
                or "quotient_gradient_error" in result.fail_reason
            )
            for result in wide
        ),
        "low_multiplicity_control_all_fail_price_or_multiplicity": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "low_multiplicity" in result.fail_reason
                or "price" in result.fail_reason
            )
            for result in low
        ),
        "bad_scaling_control_all_fail_scaling": all(
            (not result.passed)
            and result.fail_reason is not None
            and "multiplicity_scaling" in result.fail_reason
            for result in bad_scale
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
        "PASS_R501_S81_EEQ_CS_GATE"
        if all(checks.values())
        else "FAIL_R501_S81_EEQ_CS_GATE"
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
            "Synthetic CPU exact-equivalence contextual-synonym quotient gate "
            "only; not heldout CE progress, not real corpus token efficiency, "
            "not GPU training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S81_EEQ_CS_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_quotient_class_count_mean={positive['quotient_class_count_mean']:.6f} "
        f"positive_mean_multiplicity_mean={positive['mean_multiplicity_mean']:.6f} "
        f"positive_max_hidden_radius_max={positive['max_hidden_radius_max']:.6f} "
        f"positive_max_cotangent_drift_ratio_max={positive['max_cotangent_drift_ratio_max']:.6f} "
        f"positive_quotient_gradient_relative_error_max={positive['quotient_gradient_relative_error_max']:.6f} "
        f"positive_quotient_gradient_cosine_min={positive['quotient_gradient_cosine_min']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_token_efficiency_multiplier_mean={positive['projected_token_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
