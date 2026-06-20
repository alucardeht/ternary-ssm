#!/usr/bin/env python3
"""R501 S89 certified rare-token escrow gate.

This CPU diagnostic implements the R501 Quality Paradigm 11 finding:
"Certified Rare-Token Escrow (CRTE)".

The gate constructs batches containing high-CE rare-token gradients that are
too diluted or locally contradictory to cross a ternary commitment threshold
under ordinary batch averaging. The candidate escrows only certified high-CE
rare tokens when their gradient norm times batch alignment is below threshold,
then releases them when a later lattice geometry is compatible.

Controls exercise the declared kill criteria: no escrow loses the rare credit,
premature release still cannot realize the capability, stale geometry releases
into the wrong lattice, low-CE false escrow poisons the queue, and expensive
escrow erases the denominator.

This is a synthetic CPU rare-token escrow diagnostic, not heldout CE progress,
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


SCHEMA = "r501_s89_crte_gate.v1"
CONDITIONS = (
    "positive",
    "no_escrow_fail",
    "premature_release_fail",
    "stale_geometry_fail",
    "low_ce_false_escrow_fail",
    "escrow_cost_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    dim: int
    batch_count: int
    batch_size: int
    rare_token_count: int
    common_token_count: int
    rare_token_recall: float
    false_escrow_fraction: float
    release_precision: float
    stale_release_harm_fraction: float
    baseline_unique_capability_fraction: float
    escrow_unique_capability_fraction: float
    unique_capability_gain: float
    majority_retention_fraction: float
    escrow_queue_fraction: float
    escrow_overhead_ratio: float
    paid_work_ratio: float
    projected_unique_capability_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50189)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--batch-count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rare-per-batch", type=int, default=1)
    parser.add_argument("--high-ce-threshold", type=float, default=1.5)
    parser.add_argument("--commitment-threshold", type=float, default=0.25)
    parser.add_argument("--conflict-threshold", type=float, default=0.25)
    parser.add_argument("--escrow-overhead", type=float, default=0.08)
    parser.add_argument("--expensive-escrow-overhead", type=float, default=0.45)
    parser.add_argument("--min-rare-recall", type=float, default=0.99)
    parser.add_argument("--max-false-escrow-fraction", type=float, default=0.02)
    parser.add_argument("--min-release-precision", type=float, default=0.99)
    parser.add_argument("--max-stale-release-harm", type=float, default=0.02)
    parser.add_argument("--max-baseline-capability-fraction", type=float, default=0.35)
    parser.add_argument("--min-escrow-capability-fraction", type=float, default=0.99)
    parser.add_argument("--min-majority-retention", type=float, default=0.99)
    parser.add_argument("--max-paid-work-ratio", type=float, default=0.35)
    parser.add_argument("--min-unique-capability-multiplier", type=float, default=3.0)
    return parser.parse_args()


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.0e-30)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0e-30)
    return float(np.dot(a, b) / denom)


def orthogonal_component(vector: np.ndarray, against: np.ndarray) -> np.ndarray:
    denom = max(float(np.dot(against, against)), 1.0e-30)
    return vector - against * (float(np.dot(vector, against)) / denom)


def first_failure(
    *,
    rare_token_recall: float,
    false_escrow_fraction: float,
    release_precision: float,
    stale_release_harm_fraction: float,
    baseline_unique_capability_fraction: float,
    escrow_unique_capability_fraction: float,
    majority_retention_fraction: float,
    paid_work_ratio: float,
    projected_unique_capability_multiplier: float,
    args: argparse.Namespace,
) -> str | None:
    if not finite(
        rare_token_recall,
        false_escrow_fraction,
        release_precision,
        stale_release_harm_fraction,
        baseline_unique_capability_fraction,
        escrow_unique_capability_fraction,
        majority_retention_fraction,
        paid_work_ratio,
        projected_unique_capability_multiplier,
    ):
        return "nonfinite_metric"
    if false_escrow_fraction > args.max_false_escrow_fraction:
        return "false_escrow_low_ce"
    if paid_work_ratio > args.max_paid_work_ratio:
        return "price_paid_work"
    if rare_token_recall < args.min_rare_recall:
        return "rare_recall"
    if release_precision < args.min_release_precision:
        return "release_precision"
    if stale_release_harm_fraction > args.max_stale_release_harm:
        return "stale_release_harm"
    if baseline_unique_capability_fraction > args.max_baseline_capability_fraction:
        return "baseline_not_washed_out"
    if escrow_unique_capability_fraction < args.min_escrow_capability_fraction:
        return "rare_capability_acquisition"
    if majority_retention_fraction < args.min_majority_retention:
        return "majority_retention"
    if projected_unique_capability_multiplier < args.min_unique_capability_multiplier:
        return "unique_capability_multiplier"
    return None


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    if args.rare_per_batch >= args.batch_size:
        raise ValueError("rare_per_batch must be smaller than batch_size")

    rare_gradients: list[np.ndarray] = []
    common_gradients: list[np.ndarray] = []
    rare_baseline_credit: list[float] = []
    escrowed_rare = 0
    escrowed_false = 0
    released_effective = 0.0
    released_total = 0
    stale_harm = 0.0

    disable_ce_gate = condition == "low_ce_false_escrow_fail"
    no_escrow = condition == "no_escrow_fail"
    premature_release = condition == "premature_release_fail"
    stale_geometry = condition == "stale_geometry_fail"

    for _ in range(args.batch_count):
        rare_dirs = normalize_rows(rng.normal(size=(args.rare_per_batch, args.dim)))
        common_count = args.batch_size - args.rare_per_batch
        batch_common: list[np.ndarray] = []
        for _common in range(common_count):
            raw = rng.normal(size=args.dim)
            for rare_dir in rare_dirs:
                raw = orthogonal_component(raw, rare_dir)
            raw /= max(float(np.linalg.norm(raw)), 1.0e-30)
            batch_common.append(0.55 * raw)

        batch_common_array = np.array(batch_common, dtype=np.float64)
        batch_rare_array = rare_dirs.copy()
        full_batch = np.concatenate([batch_rare_array, batch_common_array], axis=0)
        full_batch_mean = np.mean(full_batch, axis=0)
        common_mean = np.mean(batch_common_array, axis=0)
        common_gradients.extend(batch_common_array)

        for rare_dir in batch_rare_array:
            rare_gradients.append(rare_dir)
            baseline_projection = max(0.0, float(np.dot(full_batch_mean, rare_dir)))
            rare_baseline_credit.append(
                min(1.0, baseline_projection / args.commitment_threshold)
            )
            conflict_score = np.linalg.norm(rare_dir) * cosine(rare_dir, common_mean)
            high_ce = True
            should_escrow = (
                high_ce
                and conflict_score < args.conflict_threshold
                and not no_escrow
            )
            if should_escrow:
                escrowed_rare += 1
                if premature_release:
                    compatibility = 0.0
                elif stale_geometry:
                    compatibility = 0.25
                else:
                    compatibility = 1.0
                released_total += 1
                released_effective += compatibility
                stale_harm += max(0.0, 1.0 - compatibility)

        if disable_ce_gate:
            for common in batch_common_array:
                conflict_score = np.linalg.norm(common) * cosine(common, full_batch_mean)
                if conflict_score < args.conflict_threshold:
                    escrowed_false += 1

    rare_token_count = len(rare_gradients)
    common_token_count = len(common_gradients)
    total_token_count = rare_token_count + common_token_count
    escrowed_total = escrowed_rare + escrowed_false

    rare_token_recall = float(escrowed_rare / max(rare_token_count, 1))
    false_escrow_fraction = float(escrowed_false / max(escrowed_total, 1))
    release_precision = float(released_effective / max(released_total, 1))
    stale_release_harm_fraction = float(stale_harm / max(released_total, 1))
    baseline_unique_capability_fraction = float(np.mean(rare_baseline_credit))
    escrow_unique_capability_fraction = float(released_effective / max(rare_token_count, 1))
    unique_capability_gain = float(
        escrow_unique_capability_fraction
        / max(baseline_unique_capability_fraction, 1.0 / max(rare_token_count, 1))
    )

    common_oracle = np.sum(np.array(common_gradients, dtype=np.float64), axis=0)
    if disable_ce_gate and escrowed_false > 0:
        retained_common = common_oracle * (1.0 - false_escrow_fraction)
    else:
        retained_common = common_oracle
    majority_retention_fraction = float(
        max(0.0, np.dot(retained_common, common_oracle))
        / max(float(np.dot(common_oracle, common_oracle)), 1.0e-30)
    )

    escrow_overhead = (
        args.expensive_escrow_overhead
        if condition == "escrow_cost_fail"
        else args.escrow_overhead
    )
    escrow_queue_fraction = float(escrowed_total / max(total_token_count, 1))
    paid_work_ratio = float(escrow_queue_fraction + escrow_overhead)
    projected_unique_capability_multiplier = float(1.0 / max(paid_work_ratio, 1.0e-30))

    fail_reason = first_failure(
        rare_token_recall=rare_token_recall,
        false_escrow_fraction=false_escrow_fraction,
        release_precision=release_precision,
        stale_release_harm_fraction=stale_release_harm_fraction,
        baseline_unique_capability_fraction=baseline_unique_capability_fraction,
        escrow_unique_capability_fraction=escrow_unique_capability_fraction,
        majority_retention_fraction=majority_retention_fraction,
        paid_work_ratio=paid_work_ratio,
        projected_unique_capability_multiplier=projected_unique_capability_multiplier,
        args=args,
    )
    passed = fail_reason is None
    expected_pass = condition == "positive"

    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=fail_reason,
        dim=args.dim,
        batch_count=args.batch_count,
        batch_size=args.batch_size,
        rare_token_count=rare_token_count,
        common_token_count=common_token_count,
        rare_token_recall=rare_token_recall,
        false_escrow_fraction=false_escrow_fraction,
        release_precision=release_precision,
        stale_release_harm_fraction=stale_release_harm_fraction,
        baseline_unique_capability_fraction=baseline_unique_capability_fraction,
        escrow_unique_capability_fraction=escrow_unique_capability_fraction,
        unique_capability_gain=unique_capability_gain,
        majority_retention_fraction=majority_retention_fraction,
        escrow_queue_fraction=escrow_queue_fraction,
        escrow_overhead_ratio=escrow_overhead,
        paid_work_ratio=paid_work_ratio,
        projected_unique_capability_multiplier=projected_unique_capability_multiplier,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "dim",
        "batch_count",
        "batch_size",
        "rare_token_count",
        "common_token_count",
        "rare_token_recall",
        "false_escrow_fraction",
        "release_precision",
        "stale_release_harm_fraction",
        "baseline_unique_capability_fraction",
        "escrow_unique_capability_fraction",
        "unique_capability_gain",
        "majority_retention_fraction",
        "escrow_queue_fraction",
        "escrow_overhead_ratio",
        "paid_work_ratio",
        "projected_unique_capability_multiplier",
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
    no_escrow = [result for result in results if result.condition == "no_escrow_fail"]
    premature = [result for result in results if result.condition == "premature_release_fail"]
    stale = [result for result in results if result.condition == "stale_geometry_fail"]
    low_ce = [result for result in results if result.condition == "low_ce_false_escrow_fail"]
    cost = [result for result in results if result.condition == "escrow_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_certifies_and_captures_rare_tokens": (
            aggregates["positive"]["rare_token_recall_min"] >= 0.99
            and aggregates["positive"]["escrow_unique_capability_fraction_min"] >= 0.99
            and aggregates["positive"]["baseline_unique_capability_fraction_max"] <= 0.35
        ),
        "positive_condition_releases_only_compatible_geometry": (
            aggregates["positive"]["release_precision_min"] >= 0.99
            and aggregates["positive"]["stale_release_harm_fraction_max"] <= 0.02
            and aggregates["positive"]["false_escrow_fraction_max"] <= 0.02
        ),
        "positive_condition_preserves_majority_and_price": (
            aggregates["positive"]["majority_retention_fraction_min"] >= 0.99
            and aggregates["positive"]["paid_work_ratio_mean"] <= 0.35
            and aggregates["positive"]["projected_unique_capability_multiplier_mean"] >= 3.0
        ),
        "no_escrow_control_all_fail_recall_or_capability": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "rare_recall" in result.fail_reason
                or "rare_capability_acquisition" in result.fail_reason
            )
            for result in no_escrow
        ),
        "premature_release_control_all_fail_release_or_capability": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "release_precision" in result.fail_reason
                or "rare_capability_acquisition" in result.fail_reason
                or "stale_release_harm" in result.fail_reason
            )
            for result in premature
        ),
        "stale_geometry_control_all_fail_release_or_harm": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "release_precision" in result.fail_reason
                or "stale_release_harm" in result.fail_reason
                or "rare_capability_acquisition" in result.fail_reason
            )
            for result in stale
        ),
        "low_ce_control_all_fail_false_escrow": all(
            (not result.passed)
            and result.fail_reason is not None
            and "false_escrow" in result.fail_reason
            for result in low_ce
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
    decision = "PASS_R501_S89_CRTE_GATE" if all(checks.values()) else "FAIL_R501_S89_CRTE_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU certified rare-token escrow gate only; not heldout "
            "CE progress, not real corpus token efficiency, not GPU training "
            "speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S89_CRTE_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_rare_token_recall_min={positive['rare_token_recall_min']:.6f} "
        f"positive_baseline_unique_capability_fraction_mean={positive['baseline_unique_capability_fraction_mean']:.6f} "
        f"positive_escrow_unique_capability_fraction_mean={positive['escrow_unique_capability_fraction_mean']:.6f} "
        f"positive_release_precision_min={positive['release_precision_min']:.6f} "
        f"positive_majority_retention_fraction_min={positive['majority_retention_fraction_min']:.6f} "
        f"positive_paid_work_ratio_mean={positive['paid_work_ratio_mean']:.6f} "
        f"positive_projected_unique_capability_multiplier_mean={positive['projected_unique_capability_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
