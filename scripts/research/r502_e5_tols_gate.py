#!/usr/bin/env python3
"""R502 E5 target-only logit sparsification gate.

This CPU diagnostic implements the R502 Efficiency finding:
"Target-Only Logit Sparsification (TOLS)".

The gate compares exact cross-entropy logit gradients with a target-exact,
stratified sampled partition estimator. The target logit is always evaluated
exactly, while non-target logits are sampled by vocabulary buckets and weighted
back into an amortized partition estimate.

Controls exercise the declared risks: high-variance partitions require near
dense samples, omitting the target term breaks CE, biased partition weights break
the estimator, and using a near-full sample erases the logit-memory/time
denominator.

This is a synthetic CPU sampled-logit gradient and denominator diagnostic, not
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


SCHEMA = "r502_e5_tols_gate.v1"
CONDITIONS = (
    "positive",
    "high_variance_partition_fail",
    "missing_target_fail",
    "biased_partition_fail",
    "sample_size_cost_fail",
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
    vocab_size: int
    sample_count: int
    sample_fraction: float
    gradient_l2_error_mean: float
    gradient_l2_error_max: float
    partition_relative_error_mean: float
    partition_relative_error_max: float
    target_gradient_abs_error_max: float
    logit_vram_ratio: float
    tols_time_ratio: float
    projected_logit_vram_saving_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50205)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--token-count", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--bucket-count", type=int, default=16)
    parser.add_argument("--max-gradient-l2-error", type=float, default=0.09)
    parser.add_argument("--max-partition-relative-error", type=float, default=0.08)
    parser.add_argument("--max-target-gradient-abs-error", type=float, default=0.05)
    parser.add_argument("--max-tols-time-ratio", type=float, default=0.90)
    parser.add_argument("--max-logit-vram-ratio", type=float, default=0.90)
    parser.add_argument("--estimator-overhead-ratio", type=float, default=0.08)
    parser.add_argument("--large-sample-fraction", type=float, default=0.92)
    return parser.parse_args()


def stratified_sample(
    rng: np.random.Generator,
    vocab_size: int,
    bucket_count: int,
    sample_count: int,
    target: int,
) -> np.ndarray:
    per_bucket = max(1, sample_count // bucket_count)
    sampled: list[np.ndarray] = []
    used: set[int] = set()
    for bucket in range(bucket_count):
        candidates = np.arange(bucket, vocab_size, bucket_count, dtype=np.int64)
        candidates = candidates[candidates != target]
        take = min(per_bucket, candidates.shape[0])
        chosen = rng.choice(candidates, size=take, replace=False)
        sampled.append(chosen)
        used.update(int(value) for value in chosen)
    remaining = sample_count - sum(part.shape[0] for part in sampled)
    if remaining > 0:
        pool = np.array([idx for idx in range(vocab_size) if idx != target and idx not in used], dtype=np.int64)
        sampled.append(rng.choice(pool, size=remaining, replace=False))
    out = np.concatenate(sampled)
    rng.shuffle(out)
    return out[:sample_count]


def make_logits(
    rng: np.random.Generator,
    vocab_size: int,
    bucket_count: int,
    sample: np.ndarray,
    target: int,
    condition: str,
) -> np.ndarray:
    buckets = np.arange(vocab_size) % bucket_count
    bucket_offsets = rng.normal(scale=0.045, size=bucket_count)
    logits = bucket_offsets[buckets] + rng.normal(scale=0.025, size=vocab_size)
    logits[target] = rng.normal(scale=0.035)
    if condition == "high_variance_partition_fail":
        sampled = set(int(value) for value in sample)
        unsampled = np.array(
            [idx for idx in range(vocab_size) if idx != target and idx not in sampled],
            dtype=np.int64,
        )
        spikes = rng.choice(unsampled, size=min(12, unsampled.shape[0]), replace=False)
        logits[spikes] = 5.0 + rng.normal(scale=0.05, size=spikes.shape[0])
    return logits


def exact_gradient(logits: np.ndarray, target: int) -> tuple[np.ndarray, float, np.ndarray]:
    shifted = logits - float(np.max(logits))
    exp_values = np.exp(shifted)
    partition = float(np.sum(exp_values))
    probs = exp_values / partition
    gradient = probs.copy()
    gradient[target] -= 1.0
    return gradient, partition, exp_values


def estimated_gradient(
    exp_values: np.ndarray,
    target: int,
    sample: np.ndarray,
    bucket_count: int,
    condition: str,
) -> tuple[np.ndarray, float]:
    vocab_size = exp_values.shape[0]
    gradient = np.zeros(vocab_size, dtype=np.float64)
    target_exp = float(exp_values[target])
    partition_estimate = target_exp
    for bucket in range(bucket_count):
        bucket_indices = np.arange(bucket, vocab_size, bucket_count, dtype=np.int64)
        bucket_indices = bucket_indices[bucket_indices != target]
        bucket_sample = sample[sample % bucket_count == bucket]
        if bucket_sample.shape[0] == 0:
            continue
        if condition == "biased_partition_fail":
            weight = 1.0
        else:
            weight = float(bucket_indices.shape[0]) / float(bucket_sample.shape[0])
        partition_estimate += weight * float(np.sum(exp_values[bucket_sample]))

    if condition == "missing_target_fail":
        gradient[target] = 0.0
    else:
        gradient[target] = target_exp / partition_estimate - 1.0

    for bucket in range(bucket_count):
        bucket_indices = np.arange(bucket, vocab_size, bucket_count, dtype=np.int64)
        bucket_indices = bucket_indices[bucket_indices != target]
        bucket_sample = sample[sample % bucket_count == bucket]
        if bucket_sample.shape[0] == 0:
            continue
        if condition == "biased_partition_fail":
            weight = 1.0
        else:
            weight = float(bucket_indices.shape[0]) / float(bucket_sample.shape[0])
        gradient[bucket_sample] = weight * exp_values[bucket_sample] / partition_estimate
    return gradient, float(partition_estimate)


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    sample_count = args.sample_count
    overhead = args.estimator_overhead_ratio
    if condition == "sample_size_cost_fail":
        sample_count = int(round(args.large_sample_fraction * args.vocab_size))
        overhead = 0.04

    gradient_errors: list[float] = []
    partition_errors: list[float] = []
    target_errors: list[float] = []
    for _ in range(args.token_count):
        target = int(rng.integers(0, args.vocab_size))
        sample = stratified_sample(
            rng,
            args.vocab_size,
            args.bucket_count,
            sample_count,
            target,
        )
        logits = make_logits(rng, args.vocab_size, args.bucket_count, sample, target, condition)
        exact, partition, exp_values = exact_gradient(logits, target)
        estimated, partition_estimate = estimated_gradient(
            exp_values,
            target,
            sample,
            args.bucket_count,
            condition,
        )
        delta = estimated - exact
        gradient_errors.append(float(np.linalg.norm(delta)))
        partition_errors.append(float(abs(partition_estimate - partition) / max(partition, 1.0e-30)))
        target_errors.append(float(abs(estimated[target] - exact[target])))

    sample_fraction = float(sample_count / args.vocab_size)
    logit_vram_ratio = sample_fraction
    tols_time_ratio = sample_fraction + overhead
    projected_multiplier = 1.0 / logit_vram_ratio
    grad_mean = float(np.mean(gradient_errors))
    grad_max = float(np.max(gradient_errors))
    part_mean = float(np.mean(partition_errors))
    part_max = float(np.max(partition_errors))
    target_max = float(np.max(target_errors))

    failed: list[str] = []
    if not finite(
        sample_fraction,
        grad_mean,
        grad_max,
        part_mean,
        part_max,
        target_max,
        logit_vram_ratio,
        tols_time_ratio,
        projected_multiplier,
    ):
        failed.append("non_finite_metric")
    if grad_max > args.max_gradient_l2_error:
        failed.append("gradient_residual")
    if part_max > args.max_partition_relative_error:
        failed.append("partition_variance")
    if target_max > args.max_target_gradient_abs_error:
        failed.append("target_gradient")
    if logit_vram_ratio > args.max_logit_vram_ratio:
        failed.append("logit_vram_price")
    if tols_time_ratio > args.max_tols_time_ratio:
        failed.append("time_price")

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
        vocab_size=args.vocab_size,
        sample_count=sample_count,
        sample_fraction=sample_fraction,
        gradient_l2_error_mean=grad_mean,
        gradient_l2_error_max=grad_max,
        partition_relative_error_mean=part_mean,
        partition_relative_error_max=part_max,
        target_gradient_abs_error_max=target_max,
        logit_vram_ratio=logit_vram_ratio,
        tols_time_ratio=tols_time_ratio,
        projected_logit_vram_saving_multiplier=projected_multiplier,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "token_count",
        "vocab_size",
        "sample_count",
        "sample_fraction",
        "gradient_l2_error_mean",
        "gradient_l2_error_max",
        "partition_relative_error_mean",
        "partition_relative_error_max",
        "target_gradient_abs_error_max",
        "logit_vram_ratio",
        "tols_time_ratio",
        "projected_logit_vram_saving_multiplier",
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
    high_variance = [result for result in results if result.condition == "high_variance_partition_fail"]
    missing_target = [result for result in results if result.condition == "missing_target_fail"]
    biased = [result for result in results if result.condition == "biased_partition_fail"]
    cost = [result for result in results if result.condition == "sample_size_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_gradient_residual_bounded": (
            aggregates["positive"]["gradient_l2_error_max_max"] <= 0.09
            and aggregates["positive"]["partition_relative_error_max_max"] <= 0.08
            and aggregates["positive"]["target_gradient_abs_error_max_max"] <= 0.05
        ),
        "positive_condition_denominator_priced": (
            aggregates["positive"]["logit_vram_ratio_mean"] <= 0.90
            and aggregates["positive"]["tols_time_ratio_mean"] <= 0.90
        ),
        "high_variance_control_all_fail_partition_or_gradient": all(
            (not result.passed)
            and result.fail_reason is not None
            and ("partition_variance" in result.fail_reason or "gradient_residual" in result.fail_reason)
            for result in high_variance
        ),
        "missing_target_control_all_fail_target": all(
            (not result.passed)
            and result.fail_reason is not None
            and "target_gradient" in result.fail_reason
            for result in missing_target
        ),
        "biased_partition_control_all_fail_partition": all(
            (not result.passed)
            and result.fail_reason is not None
            and "partition_variance" in result.fail_reason
            for result in biased
        ),
        "sample_size_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and ("logit_vram_price" in result.fail_reason or "time_price" in result.fail_reason)
            for result in cost
        ),
        "all_conditions_identified": all(result.identifying_correct for result in results),
    }


def main() -> int:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("--seed-count must be at least 1")
    if args.token_count < 1:
        raise ValueError("--token-count must be at least 1")
    if args.vocab_size < 16:
        raise ValueError("--vocab-size must be at least 16")
    if not 1 <= args.sample_count < args.vocab_size:
        raise ValueError("--sample-count must be in [1, vocab-size)")
    if args.bucket_count < 1:
        raise ValueError("--bucket-count must be at least 1")

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
    decision = "PASS_R502_E5_TOLS_GATE" if all(checks.values()) else "FAIL_R502_E5_TOLS_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU target-only logit sparsification gate only; not "
            "heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_E5_TOLS_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_gradient_l2_error_max_max={positive['gradient_l2_error_max_max']:.12e} "
        f"positive_partition_relative_error_max_max={positive['partition_relative_error_max_max']:.12e} "
        f"positive_target_gradient_abs_error_max_max={positive['target_gradient_abs_error_max_max']:.12e} "
        f"positive_sample_fraction_mean={positive['sample_fraction_mean']:.6f} "
        f"positive_logit_vram_ratio_mean={positive['logit_vram_ratio_mean']:.6f} "
        f"positive_tols_time_ratio_mean={positive['tols_time_ratio_mean']:.6f} "
        f"positive_projected_logit_vram_saving_multiplier_mean={positive['projected_logit_vram_saving_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
