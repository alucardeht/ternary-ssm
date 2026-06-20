#!/usr/bin/env python3
"""R501 S83 lossless back-translation of residual error gate.

This CPU diagnostic implements the R501 Quality Paradigm 5 finding:
"Lossless Back-Translation of Residual Error (LBTRE)".

The gate constructs layered token traces with a CE residual caused by a specific
rank-collapse layer. The candidate detects the unique rank drop and routes the
original CE cotangent only to that bottleneck layer. A positive case must match
the oracle bottleneck gradient while bypassing layers that preserved rank.

Controls exercise the declared risks: rank drops can be ambiguous, unrelated to
the true residual, absent, distributed across every layer, or too expensive to
detect.

This is a synthetic CPU layer-routing diagnostic, not heldout CE progress, real
corpus token efficiency, GPU training-speed evidence, or production v12
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


SCHEMA = "r501_s83_lbtre_gate.v1"
CONDITIONS = (
    "positive",
    "ambiguous_rank_fail",
    "wrong_rank_proxy_fail",
    "no_collapse_fail",
    "dense_bottleneck_fail",
    "detector_cost_fail",
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
    layer_count: int
    gradient_dim: int
    routing_accuracy: float
    ambiguous_drop_fraction: float
    no_signal_fraction: float
    residual_capture_fraction: float
    oracle_gradient_relative_error: float
    oracle_gradient_cosine: float
    non_bottleneck_update_energy_fraction: float
    updated_layer_fraction: float
    detector_work_units: float
    routed_work_units: float
    priced_work_ratio: float
    projected_token_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50183)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--token-count", type=int, default=192)
    parser.add_argument("--layer-count", type=int, default=6)
    parser.add_argument("--gradient-dim", type=int, default=20)
    parser.add_argument("--base-rank", type=int, default=18)
    parser.add_argument("--rank-drop", type=int, default=5)
    parser.add_argument("--detector-overhead", type=float, default=0.08)
    parser.add_argument("--expensive-detector-overhead", type=float, default=0.38)
    parser.add_argument("--min-routing-accuracy", type=float, default=0.95)
    parser.add_argument("--max-ambiguous-fraction", type=float, default=0.05)
    parser.add_argument("--max-no-signal-fraction", type=float, default=0.02)
    parser.add_argument("--min-residual-capture", type=float, default=0.98)
    parser.add_argument("--max-relative-error", type=float, default=0.05)
    parser.add_argument("--min-gradient-cosine", type=float, default=0.999)
    parser.add_argument("--max-non-bottleneck-energy", type=float, default=0.02)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.45)
    parser.add_argument("--min-token-efficiency-multiplier", type=float, default=2.0)
    return parser.parse_args()


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0e-30)
    return float(np.dot(a, b) / denom)


def make_trace(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    token_count = args.token_count
    layer_count = args.layer_count
    ranks = np.full((token_count, layer_count + 1), args.base_rank, dtype=np.int64)
    oracle = np.zeros((token_count, layer_count, args.gradient_dim), dtype=np.float64)
    true_layers = rng.integers(0, layer_count, size=token_count)
    gradients = rng.normal(size=(token_count, args.gradient_dim))
    gradients /= np.maximum(np.linalg.norm(gradients, axis=1, keepdims=True), 1.0e-30)

    if condition == "no_collapse_fail":
        for token, layer in enumerate(true_layers):
            oracle[token, layer] = gradients[token]
        return ranks, oracle, true_layers

    for token, layer in enumerate(true_layers):
        if condition == "wrong_rank_proxy_fail":
            drop_layer = (layer + 2) % layer_count
            ranks[token, drop_layer + 1 :] -= args.rank_drop
            oracle[token, layer] = gradients[token]
        elif condition == "ambiguous_rank_fail":
            second = (layer + 1) % layer_count
            first, last = sorted((layer, second))
            ranks[token, first + 1 : last + 1] -= args.rank_drop
            ranks[token, last + 1 :] -= 2 * args.rank_drop
            oracle[token, layer] = 0.5 * gradients[token]
            oracle[token, second] = 0.5 * gradients[token]
        elif condition == "dense_bottleneck_fail":
            for each_layer in range(layer_count):
                ranks[token, each_layer + 1 :] -= args.rank_drop
                oracle[token, each_layer] = gradients[token] / layer_count
        else:
            ranks[token, layer + 1 :] -= args.rank_drop
            oracle[token, layer] = gradients[token]
    return ranks, oracle, true_layers


def detect_layer(rank_row: np.ndarray) -> tuple[int, int]:
    drops = rank_row[:-1] - rank_row[1:]
    positive = np.flatnonzero(drops > 0)
    if positive.size == 0:
        return -1, 0
    return int(positive[int(np.argmax(drops[positive]))]), int(positive.size)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    ranks, oracle, true_layers = make_trace(rng, args, condition)
    candidate = np.zeros_like(oracle)
    detected_layers: list[int] = []
    ambiguous = 0
    no_signal = 0
    updated = 0

    for token in range(args.token_count):
        detected, positive_count = detect_layer(ranks[token])
        detected_layers.append(detected)
        if positive_count != 1:
            ambiguous += int(positive_count > 1)
            no_signal += int(positive_count == 0)
        if detected >= 0:
            candidate[token, detected] = np.sum(oracle[token], axis=0)
            updated += 1

    detected_array = np.array(detected_layers, dtype=np.int64)
    routing_accuracy = float(np.mean(detected_array == true_layers))
    ambiguous_fraction = ambiguous / args.token_count
    no_signal_fraction = no_signal / args.token_count

    oracle_flat = oracle.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    oracle_norm = max(float(np.linalg.norm(oracle_flat)), 1.0e-30)
    relative_error = float(np.linalg.norm(candidate_flat - oracle_flat)) / oracle_norm
    gradient_cosine = cosine(candidate_flat, oracle_flat)
    residual_capture = float(np.dot(candidate_flat, oracle_flat)) / max(
        float(np.dot(oracle_flat, oracle_flat)),
        1.0e-30,
    )

    oracle_support = np.linalg.norm(oracle, axis=2) > 1.0e-12
    candidate_energy = np.sum(candidate * candidate, axis=2)
    total_candidate_energy = max(float(np.sum(candidate_energy)), 1.0e-30)
    non_bottleneck_energy = float(np.sum(candidate_energy[~oracle_support])) / total_candidate_energy

    updated_layer_fraction = updated / (args.token_count * args.layer_count)
    detector_overhead = (
        args.expensive_detector_overhead
        if condition == "detector_cost_fail"
        else args.detector_overhead
    )
    dense_work = float(args.token_count * args.layer_count)
    detector_work = detector_overhead * dense_work
    routed_work = float(updated)
    priced_work_ratio = (detector_work + routed_work) / dense_work
    projected_multiplier = max(residual_capture, 0.0) / max(priced_work_ratio, 1.0e-30)

    failed: list[str] = []
    if not finite(
        routing_accuracy,
        ambiguous_fraction,
        no_signal_fraction,
        residual_capture,
        relative_error,
        gradient_cosine,
        non_bottleneck_energy,
        updated_layer_fraction,
        detector_work,
        routed_work,
        priced_work_ratio,
        projected_multiplier,
    ):
        failed.append("non_finite_metric")
    if routing_accuracy < args.min_routing_accuracy:
        failed.append("routing_accuracy")
    if ambiguous_fraction > args.max_ambiguous_fraction:
        failed.append("ambiguous_rank_drop")
    if no_signal_fraction > args.max_no_signal_fraction:
        failed.append("no_rank_collapse_signal")
    if residual_capture < args.min_residual_capture:
        failed.append("residual_capture")
    if relative_error > args.max_relative_error:
        failed.append("oracle_gradient_error")
    if gradient_cosine < args.min_gradient_cosine:
        failed.append("oracle_gradient_direction")
    if non_bottleneck_energy > args.max_non_bottleneck_energy:
        failed.append("non_bottleneck_energy")
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
        layer_count=args.layer_count,
        gradient_dim=args.gradient_dim,
        routing_accuracy=float(routing_accuracy),
        ambiguous_drop_fraction=float(ambiguous_fraction),
        no_signal_fraction=float(no_signal_fraction),
        residual_capture_fraction=float(residual_capture),
        oracle_gradient_relative_error=float(relative_error),
        oracle_gradient_cosine=float(gradient_cosine),
        non_bottleneck_update_energy_fraction=float(non_bottleneck_energy),
        updated_layer_fraction=float(updated_layer_fraction),
        detector_work_units=float(detector_work),
        routed_work_units=float(routed_work),
        priced_work_ratio=float(priced_work_ratio),
        projected_token_efficiency_multiplier=float(projected_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "token_count",
        "layer_count",
        "gradient_dim",
        "routing_accuracy",
        "ambiguous_drop_fraction",
        "no_signal_fraction",
        "residual_capture_fraction",
        "oracle_gradient_relative_error",
        "oracle_gradient_cosine",
        "non_bottleneck_update_energy_fraction",
        "updated_layer_fraction",
        "detector_work_units",
        "routed_work_units",
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
    ambiguous = [result for result in results if result.condition == "ambiguous_rank_fail"]
    wrong = [result for result in results if result.condition == "wrong_rank_proxy_fail"]
    no_collapse = [result for result in results if result.condition == "no_collapse_fail"]
    dense = [result for result in results if result.condition == "dense_bottleneck_fail"]
    cost = [result for result in results if result.condition == "detector_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_routes_exact_bottleneck": (
            aggregates["positive"]["routing_accuracy_min"] >= 0.95
            and aggregates["positive"]["ambiguous_drop_fraction_max"] <= 0.05
            and aggregates["positive"]["no_signal_fraction_max"] <= 0.02
        ),
        "positive_condition_matches_oracle_gradient": (
            aggregates["positive"]["oracle_gradient_relative_error_max"] <= 0.05
            and aggregates["positive"]["oracle_gradient_cosine_min"] >= 0.999
            and aggregates["positive"]["residual_capture_fraction_min"] >= 0.98
        ),
        "positive_condition_bypasses_non_bottlenecks": (
            aggregates["positive"]["non_bottleneck_update_energy_fraction_max"] <= 0.02
        ),
        "positive_condition_priced": (
            aggregates["positive"]["priced_work_ratio_mean"] <= 0.45
            and aggregates["positive"]["projected_token_efficiency_multiplier_mean"] >= 2.0
        ),
        "ambiguous_control_all_fail_ambiguity_or_gradient": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "ambiguous_rank_drop" in result.fail_reason
                or "oracle_gradient_error" in result.fail_reason
            )
            for result in ambiguous
        ),
        "wrong_proxy_control_all_fail_routing": all(
            (not result.passed)
            and result.fail_reason is not None
            and "routing_accuracy" in result.fail_reason
            for result in wrong
        ),
        "no_collapse_control_all_fail_no_signal": all(
            (not result.passed)
            and result.fail_reason is not None
            and "no_rank_collapse_signal" in result.fail_reason
            for result in no_collapse
        ),
        "dense_control_all_fail_ambiguity_or_energy": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "ambiguous_rank_drop" in result.fail_reason
                or "oracle_gradient_error" in result.fail_reason
                or "non_bottleneck_energy" in result.fail_reason
            )
            for result in dense
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
        "PASS_R501_S83_LBTRE_GATE"
        if all(checks.values())
        else "FAIL_R501_S83_LBTRE_GATE"
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
            "Synthetic CPU lossless back-translation of residual error gate "
            "only; not heldout CE progress, not real corpus token efficiency, "
            "not GPU training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S83_LBTRE_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_routing_accuracy_min={positive['routing_accuracy_min']:.6f} "
        f"positive_residual_capture_min={positive['residual_capture_fraction_min']:.6f} "
        f"positive_oracle_gradient_relative_error_max={positive['oracle_gradient_relative_error_max']:.6f} "
        f"positive_non_bottleneck_energy_max={positive['non_bottleneck_update_energy_fraction_max']:.6f} "
        f"positive_updated_layer_fraction_mean={positive['updated_layer_fraction_mean']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_token_efficiency_multiplier_mean={positive['projected_token_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
