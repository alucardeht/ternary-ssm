#!/usr/bin/env python3
"""R501 S61 lookahead-certified CE anchor pruning gate.

This CPU diagnostic implements the R501 Speed Paradigm 13 finding:
"Lookahead-Certified CE Anchor Pruning (LCCAP)".

A cheap low-resolution lookahead marks segments whose current trajectory is
already inside a target CE safety zone. The candidate prunes dense backward work
on those anchors only when true CE slack and omitted-gradient debt certify that
zeroing the segment gradient is harmless.

Controls exercise the declared risks: a false low-resolution anchor hides high
true CE, low CE can still hide large gradient debt, and too few anchors recover
dense backward cost.

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


SCHEMA = "r501_s61_lccap_gate.v1"
CONDITIONS = (
    "positive",
    "false_anchor_fail",
    "hidden_gradient_fail",
    "low_prune_price_fail",
)


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    segment_count: int
    segment_length: int
    pruned_segment_fraction: float
    true_anchor_precision: float
    max_pruned_loss_excess: float
    omitted_gradient_debt_ratio: float
    omitted_descent_harm_ratio: float
    dense_backward_fraction: float
    lookahead_work_ratio: float
    certificate_work_ratio: float
    priced_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50161)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--segment-count", type=int, default=32)
    parser.add_argument("--segment-length", type=int, default=8)
    parser.add_argument("--gradient-dim", type=int, default=12)
    parser.add_argument("--target-loss", type=float, default=0.18)
    parser.add_argument("--epsilon-safe", type=float, default=0.025)
    parser.add_argument("--max-loss-excess", type=float, default=0.0)
    parser.add_argument("--max-gradient-debt-ratio", type=float, default=0.035)
    parser.add_argument("--max-descent-harm-ratio", type=float, default=0.035)
    parser.add_argument("--min-pruned-segment-fraction", type=float, default=0.55)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.65)
    parser.add_argument("--lookahead-work-ratio", type=float, default=0.12)
    parser.add_argument("--certificate-work-ratio", type=float, default=0.045)
    return parser.parse_args()


def anchor_count(args: argparse.Namespace, condition: str) -> int:
    if condition == "low_prune_price_fail":
        return int(round(0.20 * args.segment_count))
    return int(round(0.72 * args.segment_count))


def make_fixture(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchors = np.zeros(args.segment_count, dtype=bool)
    anchors[rng.choice(args.segment_count, size=anchor_count(args, condition), replace=False)] = True
    true_loss = rng.uniform(0.24, 0.42, size=args.segment_count)
    lookahead_loss = true_loss + rng.normal(scale=0.012, size=args.segment_count)
    gradients = rng.normal(scale=0.22, size=(args.segment_count, args.segment_length, args.gradient_dim))

    for idx, is_anchor in enumerate(anchors):
        if not is_anchor:
            continue
        true_loss[idx] = rng.uniform(args.target_loss - 0.030, args.target_loss + 0.010)
        lookahead_loss[idx] = true_loss[idx] - rng.uniform(0.006, 0.020)
        gradients[idx] = rng.normal(
            scale=0.004,
            size=(args.segment_length, args.gradient_dim),
        )

    if condition == "false_anchor_fail":
        bad = np.flatnonzero(anchors)[: max(1, args.segment_count // 5)]
        true_loss[bad] = rng.uniform(args.target_loss + 0.16, args.target_loss + 0.30, size=bad.size)
        lookahead_loss[bad] = rng.uniform(args.target_loss - 0.020, args.target_loss + 0.010, size=bad.size)
        gradients[bad] = rng.normal(scale=0.28, size=(bad.size, args.segment_length, args.gradient_dim))
    elif condition == "hidden_gradient_fail":
        bad = np.flatnonzero(anchors)[: max(1, args.segment_count // 4)]
        true_loss[bad] = rng.uniform(args.target_loss - 0.020, args.target_loss + 0.010, size=bad.size)
        lookahead_loss[bad] = true_loss[bad] - rng.uniform(0.004, 0.018, size=bad.size)
        gradients[bad] = rng.normal(scale=0.30, size=(bad.size, args.segment_length, args.gradient_dim))
    return true_loss, lookahead_loss, gradients


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    true_loss, lookahead_loss, gradients = make_fixture(rng, args, condition)
    threshold = args.target_loss + args.epsilon_safe
    pruned = lookahead_loss <= threshold
    true_anchor = true_loss <= threshold
    pruned_count = int(np.count_nonzero(pruned))
    pruned_fraction = pruned_count / args.segment_count
    true_anchor_precision = float(np.mean(true_anchor[pruned])) if pruned_count else 1.0
    loss_excess = np.maximum(true_loss[pruned] - threshold, 0.0)
    max_loss_excess = float(np.max(loss_excess)) if loss_excess.size else 0.0
    segment_gradient_norm = np.linalg.norm(gradients.reshape(args.segment_count, -1), axis=1)
    omitted_norm = float(np.linalg.norm(segment_gradient_norm[pruned]))
    total_norm = max(float(np.linalg.norm(segment_gradient_norm)), 1.0e-30)
    omitted_debt = omitted_norm / total_norm
    descent_harm = float(np.sum(segment_gradient_norm[pruned] ** 2)) / max(
        float(np.sum(segment_gradient_norm**2)),
        1.0e-30,
    )
    dense_backward_fraction = 1.0 - pruned_fraction
    priced_work_ratio = (
        dense_backward_fraction + args.lookahead_work_ratio + args.certificate_work_ratio
    )
    projected_speedup = 1.0 / priced_work_ratio if priced_work_ratio > 0.0 else math.inf

    failed: list[str] = []
    if not all(
        math.isfinite(value)
        for value in (
            pruned_fraction,
            true_anchor_precision,
            max_loss_excess,
            omitted_debt,
            descent_harm,
            priced_work_ratio,
        )
    ):
        failed.append("non_finite_metric")
    if pruned_fraction < args.min_pruned_segment_fraction:
        failed.append("prune_fraction")
    if max_loss_excess > args.max_loss_excess:
        failed.append("loss_certificate")
    if omitted_debt > args.max_gradient_debt_ratio:
        failed.append("gradient_debt")
    if descent_harm > args.max_descent_harm_ratio:
        failed.append("descent_harm")
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
        segment_count=args.segment_count,
        segment_length=args.segment_length,
        pruned_segment_fraction=float(pruned_fraction),
        true_anchor_precision=float(true_anchor_precision),
        max_pruned_loss_excess=float(max_loss_excess),
        omitted_gradient_debt_ratio=float(omitted_debt),
        omitted_descent_harm_ratio=float(descent_harm),
        dense_backward_fraction=float(dense_backward_fraction),
        lookahead_work_ratio=float(args.lookahead_work_ratio),
        certificate_work_ratio=float(args.certificate_work_ratio),
        priced_work_ratio=float(priced_work_ratio),
        projected_speedup=float(projected_speedup),
    )


def aggregate(results: list[TrialResult]) -> dict[str, Any]:
    by_condition: dict[str, list[TrialResult]] = {
        condition: [result for result in results if result.condition == condition]
        for condition in CONDITIONS
    }
    numeric_fields = (
        "pruned_segment_fraction",
        "true_anchor_precision",
        "max_pruned_loss_excess",
        "omitted_gradient_debt_ratio",
        "omitted_descent_harm_ratio",
        "dense_backward_fraction",
        "lookahead_work_ratio",
        "certificate_work_ratio",
        "priced_work_ratio",
        "projected_speedup",
    )
    aggregates: dict[str, Any] = {}
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
        for value in asdict(result).values():
            if isinstance(value, float) and not math.isfinite(value):
                return False
    return True


def required_checks(results: list[TrialResult], aggregates: dict[str, Any]) -> dict[str, bool]:
    positive = [result for result in results if result.condition == "positive"]
    false_anchor = [result for result in results if result.condition == "false_anchor_fail"]
    hidden = [result for result in results if result.condition == "hidden_gradient_fail"]
    low_prune = [result for result in results if result.condition == "low_prune_price_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_certified_harmless": (
            aggregates["positive"]["max_pruned_loss_excess_max"] <= 0.0
            and aggregates["positive"]["omitted_gradient_debt_ratio_max"] <= 0.035
            and aggregates["positive"]["omitted_descent_harm_ratio_max"] <= 0.035
        ),
        "positive_condition_prune_and_price": (
            aggregates["positive"]["pruned_segment_fraction_mean"] >= 0.55
            and aggregates["positive"]["priced_work_ratio_mean"] <= 0.65
        ),
        "false_anchor_control_all_fail_loss_and_debt": all(
            (not result.passed)
            and result.fail_reason is not None
            and "loss_certificate" in result.fail_reason
            and "gradient_debt" in result.fail_reason
            for result in false_anchor
        ),
        "hidden_gradient_control_all_fail_debt": all(
            (not result.passed)
            and result.fail_reason is not None
            and "gradient_debt" in result.fail_reason
            and "descent_harm" in result.fail_reason
            for result in hidden
        ),
        "low_prune_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "prune_fraction" in result.fail_reason
            and "price" in result.fail_reason
            for result in low_prune
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
    decision = "PASS_R501_S61_LCCAP_GATE" if all(checks.values()) else "FAIL_R501_S61_LCCAP_GATE"
    payload = {
        "schema": SCHEMA,
        "decision": decision,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "conditions": CONDITIONS,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU lookahead-certified CE anchor pruning gate only; not "
            "heldout CE progress, not GPU training speed, and not production "
            "v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S61_LCCAP_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_pruned_segment_fraction_mean={positive['pruned_segment_fraction_mean']:.6f} "
        f"positive_max_loss_excess_max={positive['max_pruned_loss_excess_max']:.6e} "
        f"positive_omitted_debt_max={positive['omitted_gradient_debt_ratio_max']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
