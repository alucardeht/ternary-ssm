#!/usr/bin/env python3
"""R501 S70 information-theoretic credit assignment gate.

This CPU diagnostic implements the R501 Quality Math 7 finding:
"Information-Theoretic Credit Assignment (ITCA)".

The gate treats token losses as a Shannon-surprise proxy and uses that proxy to
sample cotangent rows where gradient norm is expected to be large. The positive
condition must preserve the full CE gradient expectation with an unbiased
importance estimator while reducing estimator variance versus uniform sampled
CE under a priced denominator.

Controls exercise the declared risks: raw surprise weighting biases the
expected gradient, flat surprise carries no variance-reduction signal, spurious
high-surprise rows can collapse credit into the wrong direction, and an
expensive information ledger erases the denominator.

This is a synthetic CPU estimator-variance diagnostic, not heldout CE progress,
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


SCHEMA = "r501_s70_itca_gate.v1"
CONDITIONS = (
    "positive",
    "biased_weights_fail",
    "flat_surprise_fail",
    "collapse_fail",
    "ledger_cost_fail",
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
    gradient_dim: int
    sample_count: int
    sampled_token_fraction: float
    ce_variance_trace: float
    itca_variance_trace: float
    variance_reduction_ratio: float
    estimator_bias_norm: float
    gradient_cosine_to_ce: float
    surprise_weighted_cosine_to_ce: float
    high_surprise_token_fraction: float
    high_surprise_gradient_energy_fraction: float
    priced_work_ratio: float
    projected_token_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50170)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--token-count", type=int, default=512)
    parser.add_argument("--gradient-dim", type=int, default=12)
    parser.add_argument("--sample-count", type=int, default=160)
    parser.add_argument("--mc-samples", type=int, default=512)
    parser.add_argument("--max-bias-norm", type=float, default=0.04)
    parser.add_argument("--min-gradient-cosine", type=float, default=0.985)
    parser.add_argument("--max-variance-reduction-ratio", type=float, default=0.80)
    parser.add_argument("--min-surprise-weighted-cosine", type=float, default=0.99)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.80)
    parser.add_argument("--ledger-overhead", type=float, default=0.34)
    parser.add_argument("--expensive-ledger-overhead", type=float, default=0.55)
    return parser.parse_args()


def normalized(rng: np.random.Generator, size: int) -> np.ndarray:
    vector = rng.normal(size=size)
    vector /= max(float(np.linalg.norm(vector)), 1.0e-30)
    return vector


def orthogonalized(
    rng: np.random.Generator,
    reference: np.ndarray,
    negative_mix: float = 0.0,
) -> np.ndarray:
    vector = normalized(rng, reference.size)
    vector = vector - float(np.dot(vector, reference)) * reference
    vector /= max(float(np.linalg.norm(vector)), 1.0e-30)
    vector = (1.0 - negative_mix) * vector - negative_mix * reference
    vector /= max(float(np.linalg.norm(vector)), 1.0e-30)
    return vector


def make_fixture(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    target = normalized(rng, args.gradient_dim)
    distractor = orthogonalized(rng, target, negative_mix=0.35)

    if condition == "flat_surprise_fail":
        losses = 1.0 + rng.normal(scale=0.003, size=args.token_count)
        amplitudes = rng.gamma(shape=2.0, scale=0.6, size=args.token_count)
        gradients = amplitudes[:, None] * target[None, :]
        gradients += rng.normal(scale=0.18, size=(args.token_count, args.gradient_dim))
        return losses, gradients

    raw_surprise = rng.gamma(shape=1.35, scale=1.0, size=args.token_count)
    losses = 0.20 + raw_surprise
    gradients = losses[:, None] * target[None, :]
    gradients += rng.normal(scale=0.035, size=(args.token_count, args.gradient_dim))

    if condition == "collapse_fail":
        cutoff = float(np.quantile(losses, 0.78))
        high = losses >= cutoff
        gradients[high] = 1.35 * losses[high, None] * distractor[None, :]
        gradients[~high] = 0.45 * losses[~high, None] * target[None, :]
        gradients += rng.normal(scale=0.025, size=(args.token_count, args.gradient_dim))

    return losses, gradients


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0e-30)
    return float(np.dot(a, b) / denom)


def sample_estimators(
    rng: np.random.Generator,
    losses: np.ndarray,
    gradients: np.ndarray,
    args: argparse.Namespace,
    raw_surprise_weighted: bool,
) -> tuple[np.ndarray, np.ndarray]:
    token_count = gradients.shape[0]
    uniform_estimators = np.empty((args.mc_samples, args.gradient_dim), dtype=np.float64)
    itca_estimators = np.empty((args.mc_samples, args.gradient_dim), dtype=np.float64)
    uniform_prob = np.full(token_count, 1.0 / token_count, dtype=np.float64)
    surprise_prob = losses / max(float(np.sum(losses)), 1.0e-30)
    relative_surprise = losses / max(float(np.mean(losses)), 1.0e-30)

    for draw in range(args.mc_samples):
        ce_idx = rng.choice(token_count, size=args.sample_count, replace=True, p=uniform_prob)
        uniform_estimators[draw] = np.mean(gradients[ce_idx], axis=0)

        if raw_surprise_weighted:
            itca_idx = rng.choice(token_count, size=args.sample_count, replace=True, p=uniform_prob)
            itca_estimators[draw] = np.mean(
                relative_surprise[itca_idx, None] * gradients[itca_idx],
                axis=0,
            )
        else:
            itca_idx = rng.choice(token_count, size=args.sample_count, replace=True, p=surprise_prob)
            itca_estimators[draw] = np.mean(
                gradients[itca_idx] / (token_count * surprise_prob[itca_idx, None]),
                axis=0,
            )

    return uniform_estimators, itca_estimators


def variance_trace(samples: np.ndarray) -> float:
    return float(np.sum(np.var(samples, axis=0, ddof=1)))


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    losses, gradients = make_fixture(rng, args, condition)
    raw_mode = condition == "biased_weights_fail"
    ce_samples, itca_samples = sample_estimators(rng, losses, gradients, args, raw_mode)

    full_ce = np.mean(gradients, axis=0)
    itca_mean = np.mean(itca_samples, axis=0)
    ce_variance = variance_trace(ce_samples)
    itca_variance = variance_trace(itca_samples)
    variance_ratio = itca_variance / max(ce_variance, 1.0e-30)
    bias_norm = float(np.linalg.norm(itca_mean - full_ce)) / max(float(np.linalg.norm(full_ce)), 1.0e-30)
    gradient_cosine = cosine(itca_mean, full_ce)
    relative_surprise = losses / max(float(np.mean(losses)), 1.0e-30)
    surprise_weighted_gradient = np.mean(relative_surprise[:, None] * gradients, axis=0)
    surprise_cosine = cosine(surprise_weighted_gradient, full_ce)
    high_cutoff = float(np.quantile(losses, 0.75))
    high = losses >= high_cutoff
    total_energy = max(float(np.sum(np.einsum("ij,ij->i", gradients, gradients))), 1.0e-30)
    high_energy = float(np.sum(np.einsum("ij,ij->i", gradients[high], gradients[high])))
    high_energy_fraction = high_energy / total_energy
    high_fraction = float(np.count_nonzero(high)) / gradients.shape[0]
    overhead = args.expensive_ledger_overhead if condition == "ledger_cost_fail" else args.ledger_overhead
    sampled_fraction = args.sample_count / args.token_count
    priced_work_ratio = sampled_fraction + overhead
    projected_multiplier = 1.0 / priced_work_ratio

    failed: list[str] = []
    if not finite(
        ce_variance,
        itca_variance,
        variance_ratio,
        bias_norm,
        gradient_cosine,
        surprise_cosine,
        high_fraction,
        high_energy_fraction,
        priced_work_ratio,
        projected_multiplier,
    ):
        failed.append("non_finite_metric")
    if bias_norm > args.max_bias_norm:
        failed.append("estimator_bias")
    if gradient_cosine < args.min_gradient_cosine:
        failed.append("gradient_cosine")
    if variance_ratio > args.max_variance_reduction_ratio:
        failed.append("variance_reduction")
    if surprise_cosine < args.min_surprise_weighted_cosine:
        failed.append("surprise_collapse")
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
        token_count=args.token_count,
        gradient_dim=args.gradient_dim,
        sample_count=args.sample_count,
        sampled_token_fraction=float(sampled_fraction),
        ce_variance_trace=float(ce_variance),
        itca_variance_trace=float(itca_variance),
        variance_reduction_ratio=float(variance_ratio),
        estimator_bias_norm=float(bias_norm),
        gradient_cosine_to_ce=float(gradient_cosine),
        surprise_weighted_cosine_to_ce=float(surprise_cosine),
        high_surprise_token_fraction=float(high_fraction),
        high_surprise_gradient_energy_fraction=float(high_energy_fraction),
        priced_work_ratio=float(priced_work_ratio),
        projected_token_efficiency_multiplier=float(projected_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "token_count",
        "gradient_dim",
        "sample_count",
        "sampled_token_fraction",
        "ce_variance_trace",
        "itca_variance_trace",
        "variance_reduction_ratio",
        "estimator_bias_norm",
        "gradient_cosine_to_ce",
        "surprise_weighted_cosine_to_ce",
        "high_surprise_token_fraction",
        "high_surprise_gradient_energy_fraction",
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
    biased = [result for result in results if result.condition == "biased_weights_fail"]
    flat = [result for result in results if result.condition == "flat_surprise_fail"]
    collapse = [result for result in results if result.condition == "collapse_fail"]
    cost = [result for result in results if result.condition == "ledger_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_unbiased": (
            aggregates["positive"]["estimator_bias_norm_max"] <= 0.04
            and aggregates["positive"]["gradient_cosine_to_ce_min"] >= 0.985
        ),
        "positive_condition_variance_reduced": (
            aggregates["positive"]["variance_reduction_ratio_mean"] <= 0.80
        ),
        "positive_condition_surprise_aligned": (
            aggregates["positive"]["surprise_weighted_cosine_to_ce_min"] >= 0.99
        ),
        "positive_condition_priced": (
            aggregates["positive"]["priced_work_ratio_mean"] <= 0.80
        ),
        "biased_control_all_fail_bias": all(
            (not result.passed)
            and result.fail_reason is not None
            and "estimator_bias" in result.fail_reason
            for result in biased
        ),
        "flat_control_all_fail_variance": all(
            (not result.passed)
            and result.fail_reason is not None
            and "variance_reduction" in result.fail_reason
            for result in flat
        ),
        "collapse_control_all_fail_alignment": all(
            (not result.passed)
            and result.fail_reason is not None
            and "surprise_collapse" in result.fail_reason
            for result in collapse
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
        "PASS_R501_S70_ITCA_GATE"
        if all(checks.values())
        else "FAIL_R501_S70_ITCA_GATE"
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
            "Synthetic CPU information-theoretic credit-assignment estimator "
            "gate only; not heldout CE progress, not real corpus token "
            "efficiency, not GPU training speed, and not production v12 "
            "integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S70_ITCA_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_bias_norm_max={positive['estimator_bias_norm_max']:.6f} "
        f"positive_gradient_cosine_min={positive['gradient_cosine_to_ce_min']:.6f} "
        f"positive_variance_reduction_ratio_mean={positive['variance_reduction_ratio_mean']:.6f} "
        f"positive_surprise_weighted_cosine_min={positive['surprise_weighted_cosine_to_ce_min']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_token_efficiency_multiplier_mean={positive['projected_token_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
