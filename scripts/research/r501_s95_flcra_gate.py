#!/usr/bin/env python3
"""R501 S95 fused logit-cotangent register accumulation gate.

This CPU diagnostic implements the second R501 Hardware Efficiency mathematical
finding: "Fused Logit-Cotangent Register Accumulation (FLCRA)".

The gate compares the standard dense CE backward path, which materializes a
full token-by-vocabulary logit/probability tensor, with a row-streamed path that
computes the same local hidden-state cotangent one token row at a time. The
positive condition must match the dense CE cotangent while charging only a
single row register window instead of global T x V logit residency.

Controls exercise the declared kill criteria: raw logits skip softmax,
missing-target cotangents omit the one-hot subtraction, sampled-vocabulary
softmax changes the CE distribution, materialized logits erase the VRAM
denominator, and extra recomputation erases the zero-overhead constraint.

This is a synthetic CPU logit-cotangent memory diagnostic, not heldout CE
progress, real VRAM telemetry, GPU training-speed evidence, or production v12
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


SCHEMA = "r501_s95_flcra_gate.v1"
CONDITIONS = (
    "positive",
    "raw_logits_fail",
    "missing_target_fail",
    "sampled_vocab_fail",
    "materialized_logits_fail",
    "extra_compute_fail",
)
FP16_BYTES = 2


@dataclass(frozen=True)
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str | None
    sequence_len: int
    vocab_dim: int
    hidden_dim: int
    sampled_vocab_size: int
    dense_logit_elements: int
    candidate_register_elements: int
    dense_logit_bytes_fp16: int
    projected_dense_logit_mib_4096x32768_fp16: float
    projected_saved_mib_4096x32768_fp16: float
    cotangent_relative_error: float
    cotangent_max_abs_error: float
    cotangent_cosine: float
    probability_normalization_error: float
    target_probability_error: float
    compute_work_ratio: float
    paid_logit_vram_ratio: float
    projected_logit_vram_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50195)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--sequence-len", type=int, default=128)
    parser.add_argument("--vocab-dim", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--sampled-vocab-size", type=int, default=256)
    parser.add_argument("--projection-sequence-len", type=int, default=4096)
    parser.add_argument("--projection-vocab-dim", type=int, default=32768)
    parser.add_argument("--max-cotangent-relative-error", type=float, default=1.0e-12)
    parser.add_argument("--max-cotangent-abs-error", type=float, default=1.0e-12)
    parser.add_argument("--min-cotangent-cosine", type=float, default=0.999999999999)
    parser.add_argument("--max-probability-normalization-error", type=float, default=1.0e-12)
    parser.add_argument("--max-target-probability-error", type=float, default=1.0e-12)
    parser.add_argument("--max-paid-logit-vram-ratio", type=float, default=0.02)
    parser.add_argument("--max-compute-work-ratio", type=float, default=1.05)
    parser.add_argument("--min-logit-vram-efficiency-multiplier", type=float, default=32.0)
    return parser.parse_args()


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference))
    return float(np.linalg.norm(candidate - reference) / max(denom, 1.0e-30))


def cosine(candidate: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(candidate) * np.linalg.norm(reference))
    if denom <= 1.0e-30:
        return 1.0 if np.allclose(candidate, reference) else 0.0
    return float(np.sum(candidate * reference) / denom)


def make_fixture(
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = rng.normal(size=(args.sequence_len, args.hidden_dim)) / math.sqrt(args.hidden_dim)
    classifier = rng.normal(size=(args.vocab_dim, args.hidden_dim)) / math.sqrt(args.hidden_dim)
    targets = rng.integers(0, args.vocab_dim, size=args.sequence_len, endpoint=False)
    return states.astype(np.float64), classifier.astype(np.float64), targets.astype(np.int64)


def dense_ce_cotangent(
    states: np.ndarray,
    classifier: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    logits = states @ classifier.T
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=1, keepdims=True)
    weights = probs.copy()
    weights[np.arange(targets.shape[0]), targets] -= 1.0
    return weights @ classifier, probs[np.arange(targets.shape[0]), targets]


def streamed_ce_cotangent(
    states: np.ndarray,
    classifier: np.ndarray,
    targets: np.ndarray,
    condition: str,
    sampled_vocab_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    sequence_len, hidden_dim = states.shape
    vocab_dim = classifier.shape[0]
    cotangent = np.zeros((sequence_len, hidden_dim), dtype=np.float64)
    target_probs = np.zeros(sequence_len, dtype=np.float64)
    max_norm_error = 0.0

    for row_idx in range(sequence_len):
        logits = classifier @ states[row_idx]
        weights = np.zeros(vocab_dim, dtype=np.float64)

        if condition == "raw_logits_fail":
            scale = max(float(np.max(np.abs(logits))), 1.0e-30)
            pseudo_probs = logits / scale
            weights[:] = pseudo_probs
            target_probs[row_idx] = pseudo_probs[targets[row_idx]]
            max_norm_error = max(max_norm_error, abs(float(np.sum(pseudo_probs)) - 1.0))
        elif condition == "sampled_vocab_fail":
            target = int(targets[row_idx])
            sample_budget = min(sampled_vocab_size, vocab_dim)
            if sample_budget < 2:
                raise ValueError("sampled_vocab_size must be at least 2")
            candidates = np.delete(np.arange(vocab_dim), target)
            sampled = rng.choice(candidates, size=sample_budget - 1, replace=False)
            subset = np.concatenate(([target], sampled))
            subset_probs = stable_softmax(logits[subset])
            weights[subset] = subset_probs
            target_probs[row_idx] = subset_probs[0]
            max_norm_error = max(max_norm_error, abs(float(np.sum(subset_probs)) - 1.0))
        else:
            probs = stable_softmax(logits)
            weights[:] = probs
            target_probs[row_idx] = probs[targets[row_idx]]
            max_norm_error = max(max_norm_error, abs(float(np.sum(probs)) - 1.0))

        if condition != "missing_target_fail":
            weights[targets[row_idx]] -= 1.0
        cotangent[row_idx] = weights @ classifier

    return cotangent, target_probs, max_norm_error


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    states, classifier, targets = make_fixture(rng, args)
    dense_cotangent, dense_target_probs = dense_ce_cotangent(states, classifier, targets)
    candidate_cotangent, candidate_target_probs, norm_error = streamed_ce_cotangent(
        states=states,
        classifier=classifier,
        targets=targets,
        condition=condition,
        sampled_vocab_size=args.sampled_vocab_size,
        rng=rng,
    )

    dense_elements = int(args.sequence_len * args.vocab_dim)
    if condition == "materialized_logits_fail":
        register_elements = dense_elements
    elif condition == "sampled_vocab_fail":
        register_elements = int(args.sampled_vocab_size)
    else:
        register_elements = int(args.vocab_dim)

    if condition == "extra_compute_fail":
        compute_work_ratio = 2.0
    else:
        compute_work_ratio = 1.0

    paid_ratio = float(register_elements / dense_elements)
    multiplier = float(1.0 / paid_ratio) if paid_ratio > 0.0 else math.inf
    dense_bytes = dense_elements * FP16_BYTES
    projected_dense_bytes = (
        args.projection_sequence_len * args.projection_vocab_dim * FP16_BYTES
    )
    projected_register_bytes = args.projection_vocab_dim * FP16_BYTES
    projected_saved_mib = float((projected_dense_bytes - projected_register_bytes) / 1048576.0)

    cot_error = relative_error(candidate_cotangent, dense_cotangent)
    max_abs = float(np.max(np.abs(candidate_cotangent - dense_cotangent)))
    cot_cosine = cosine(candidate_cotangent, dense_cotangent)
    target_prob_error = float(np.max(np.abs(candidate_target_probs - dense_target_probs)))

    failures: list[str] = []
    if cot_error > args.max_cotangent_relative_error or max_abs > args.max_cotangent_abs_error:
        failures.append("cotangent_error")
    if cot_cosine < args.min_cotangent_cosine:
        failures.append("cotangent_cosine")
    if norm_error > args.max_probability_normalization_error:
        failures.append("probability_normalization")
    if target_prob_error > args.max_target_probability_error:
        failures.append("target_probability")
    if paid_ratio > args.max_paid_logit_vram_ratio or multiplier < args.min_logit_vram_efficiency_multiplier:
        failures.append("price_paid_logit_vram")
    if compute_work_ratio > args.max_compute_work_ratio:
        failures.append("compute_overhead")

    fail_reason = "+".join(failures) if failures else None
    passed = fail_reason is None
    expected_pass = condition == "positive"
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=fail_reason,
        sequence_len=args.sequence_len,
        vocab_dim=args.vocab_dim,
        hidden_dim=args.hidden_dim,
        sampled_vocab_size=args.sampled_vocab_size,
        dense_logit_elements=dense_elements,
        candidate_register_elements=register_elements,
        dense_logit_bytes_fp16=dense_bytes,
        projected_dense_logit_mib_4096x32768_fp16=float(projected_dense_bytes / 1048576.0),
        projected_saved_mib_4096x32768_fp16=projected_saved_mib,
        cotangent_relative_error=cot_error,
        cotangent_max_abs_error=max_abs,
        cotangent_cosine=cot_cosine,
        probability_normalization_error=norm_error,
        target_probability_error=target_prob_error,
        compute_work_ratio=compute_work_ratio,
        paid_logit_vram_ratio=paid_ratio,
        projected_logit_vram_efficiency_multiplier=multiplier,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "sequence_len",
        "vocab_dim",
        "hidden_dim",
        "sampled_vocab_size",
        "dense_logit_elements",
        "candidate_register_elements",
        "dense_logit_bytes_fp16",
        "projected_dense_logit_mib_4096x32768_fp16",
        "projected_saved_mib_4096x32768_fp16",
        "cotangent_relative_error",
        "cotangent_max_abs_error",
        "cotangent_cosine",
        "probability_normalization_error",
        "target_probability_error",
        "compute_work_ratio",
        "paid_logit_vram_ratio",
        "projected_logit_vram_efficiency_multiplier",
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
    raw = [result for result in results if result.condition == "raw_logits_fail"]
    missing = [result for result in results if result.condition == "missing_target_fail"]
    sampled = [result for result in results if result.condition == "sampled_vocab_fail"]
    materialized = [result for result in results if result.condition == "materialized_logits_fail"]
    extra = [result for result in results if result.condition == "extra_compute_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_cotangent": (
            aggregates["positive"]["cotangent_relative_error_max"] <= 1.0e-12
            and aggregates["positive"]["cotangent_max_abs_error_max"] <= 1.0e-12
            and aggregates["positive"]["cotangent_cosine_min"] >= 0.999999999999
            and aggregates["positive"]["probability_normalization_error_max"] <= 1.0e-12
            and aggregates["positive"]["target_probability_error_max"] <= 1.0e-12
        ),
        "positive_condition_priced": (
            aggregates["positive"]["paid_logit_vram_ratio_mean"] <= 0.02
            and aggregates["positive"]["projected_logit_vram_efficiency_multiplier_mean"] >= 32.0
            and aggregates["positive"]["compute_work_ratio_mean"] <= 1.05
        ),
        "raw_logits_control_all_fail_softmax_or_exactness": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "probability_normalization" in result.fail_reason
                or "cotangent_error" in result.fail_reason
                or "target_probability" in result.fail_reason
            )
            for result in raw
        ),
        "missing_target_control_all_fail_exactness": all(
            (not result.passed)
            and result.fail_reason is not None
            and "cotangent_error" in result.fail_reason
            for result in missing
        ),
        "sampled_vocab_control_all_fail_distribution_or_exactness": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "target_probability" in result.fail_reason
                or "cotangent_error" in result.fail_reason
            )
            for result in sampled
        ),
        "materialized_logits_control_all_fail_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "price_paid_logit_vram" in result.fail_reason
            for result in materialized
        ),
        "extra_compute_control_all_fail_compute_price": all(
            (not result.passed)
            and result.fail_reason is not None
            and "compute_overhead" in result.fail_reason
            for result in extra
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
    decision = "PASS_R501_S95_FLCRA_GATE" if all(checks.values()) else "FAIL_R501_S95_FLCRA_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU fused logit-cotangent register accumulation gate only; "
            "not heldout CE progress, not real VRAM telemetry, not GPU training "
            "speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S95_FLCRA_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_cotangent_relative_error_max={positive['cotangent_relative_error_max']:.6e} "
        f"positive_cotangent_max_abs_error_max={positive['cotangent_max_abs_error_max']:.6e} "
        f"positive_cotangent_cosine_min={positive['cotangent_cosine_min']:.12f} "
        f"positive_paid_logit_vram_ratio_mean={positive['paid_logit_vram_ratio_mean']:.6f} "
        f"positive_projected_logit_vram_efficiency_multiplier_mean={positive['projected_logit_vram_efficiency_multiplier_mean']:.6f} "
        f"positive_projected_saved_mib_4096x32768_fp16_mean={positive['projected_saved_mib_4096x32768_fp16_mean']:.6f} "
        f"positive_compute_work_ratio_mean={positive['compute_work_ratio_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
