#!/usr/bin/env python3
"""R501 S79 information-theoretic active sensing gate.

This CPU diagnostic implements the R501 Quality Paradigm 1 finding:
"Information-Theoretic Active Sensing (Surprise-Proportional Masking)".

The gate constructs token-level CE witnesses with an expected predictable CE
baseline. The candidate keeps only tokens whose realized CE exceeds that
expectation by a margin. A positive case must retain the actionable CE excess,
the dense recurrent gradient, and the state-update contribution while skipping
enough predictable tokens to improve priced token efficiency.

Controls exercise the declared risks: CE can be a weak proxy for useful credit,
an over-strict threshold can drop useful surprise, a corpus can lack sparse
surprise, predictable tokens can still carry state-transition debt, and an
expensive selector can erase the denominator.

This is a synthetic CPU active-sensing mask diagnostic, not heldout CE progress,
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


SCHEMA = "r501_s79_information_active_sensing_gate.v1"
CONDITIONS = (
    "positive",
    "weak_ce_proxy_fail",
    "threshold_too_high_fail",
    "no_surprise_sparsity_fail",
    "state_coupling_fail",
    "selector_cost_fail",
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
    state_dim: int
    retained_fraction: float
    actionable_ce_retained_fraction: float
    gradient_energy_retained_fraction: float
    gradient_residual_ratio: float
    gradient_cosine: float
    state_residual_ratio: float
    lattice_noise_avoided_fraction: float
    dense_work_units: float
    masked_work_units: float
    selector_work_units: float
    priced_work_ratio: float
    projected_token_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50179)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--token-count", type=int, default=512)
    parser.add_argument("--gradient-dim", type=int, default=24)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--surprise-margin", type=float, default=0.35)
    parser.add_argument("--max-retained-fraction", type=float, default=0.40)
    parser.add_argument("--min-actionable-ce-retained", type=float, default=0.90)
    parser.add_argument("--min-gradient-energy-retained", type=float, default=0.92)
    parser.add_argument("--max-gradient-residual-ratio", type=float, default=0.06)
    parser.add_argument("--min-gradient-cosine", type=float, default=0.995)
    parser.add_argument("--max-state-residual-ratio", type=float, default=0.06)
    parser.add_argument("--min-lattice-noise-avoided", type=float, default=0.55)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.45)
    parser.add_argument("--min-token-efficiency-multiplier", type=float, default=2.0)
    parser.add_argument("--selector-overhead", type=float, default=0.06)
    parser.add_argument("--expensive-selector-overhead", type=float, default=0.27)
    return parser.parse_args()


def normalized(rng: np.random.Generator, size: int) -> np.ndarray:
    vector = rng.normal(size=size)
    vector /= max(float(np.linalg.norm(vector)), 1.0e-30)
    return vector


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0e-30)
    return float(np.dot(a, b) / denom)


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def make_fixture(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    token_count = args.token_count
    target_grad = normalized(rng, args.gradient_dim)
    target_state = normalized(rng, args.state_dim)
    expected_ce = 0.55 + rng.normal(scale=0.025, size=token_count)
    ce_excess = np.clip(rng.normal(loc=0.04, scale=0.018, size=token_count), 0.0, None)

    high_fraction = 0.24
    if condition == "no_surprise_sparsity_fail":
        high_fraction = 0.82
    high_count = int(round(token_count * high_fraction))
    high_idx = rng.choice(token_count, size=high_count, replace=False)
    high = np.zeros(token_count, dtype=bool)
    high[high_idx] = True

    if condition == "threshold_too_high_fail":
        ce_excess[high] = 0.82 + rng.gamma(shape=1.15, scale=0.20, size=high_count)
    else:
        ce_excess[high] = 0.92 + rng.gamma(shape=1.35, scale=0.24, size=high_count)

    actual_ce = expected_ce + ce_excess
    gradient_amplitude = 0.018 + 0.008 * rng.random(token_count)
    state_amplitude = 0.014 + 0.006 * rng.random(token_count)
    gradient_amplitude[high] = 0.95 + 0.75 * ce_excess[high]
    state_amplitude[high] = 0.80 + 0.55 * ce_excess[high]

    if condition == "weak_ce_proxy_fail":
        low_idx = np.flatnonzero(~high)
        important_count = max(1, int(round(0.18 * token_count)))
        important_idx = rng.choice(low_idx, size=important_count, replace=False)
        gradient_amplitude[high] = 0.08 + 0.03 * rng.random(high_count)
        state_amplitude[high] = 0.08 + 0.03 * rng.random(high_count)
        gradient_amplitude[important_idx] = 1.30 + 0.50 * rng.random(important_count)
        state_amplitude[important_idx] = 1.00 + 0.45 * rng.random(important_count)

    if condition == "state_coupling_fail":
        low_idx = np.flatnonzero(~high)
        coupled_count = max(1, int(round(0.28 * token_count)))
        coupled_idx = rng.choice(low_idx, size=coupled_count, replace=False)
        state_amplitude[coupled_idx] = 0.92 + 0.40 * rng.random(coupled_count)

    gradients = gradient_amplitude[:, None] * target_grad[None, :]
    gradients += rng.normal(scale=0.010, size=(token_count, args.gradient_dim))
    state_deltas = state_amplitude[:, None] * target_state[None, :]
    state_deltas += rng.normal(scale=0.008, size=(token_count, args.state_dim))
    return expected_ce, actual_ce, gradients, state_deltas


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    expected_ce, actual_ce, gradients, state_deltas = make_fixture(rng, args, condition)

    margin = 1.45 if condition == "threshold_too_high_fail" else args.surprise_margin
    mask = (actual_ce - expected_ce) > margin
    retained_count = int(np.count_nonzero(mask))
    retained_fraction = retained_count / args.token_count

    actionable_ce = np.clip(actual_ce - expected_ce, 0.0, None)
    actionable_total = max(float(np.sum(actionable_ce)), 1.0e-30)
    actionable_retained = float(np.sum(actionable_ce[mask])) / actionable_total

    dense_gradient = np.sum(gradients, axis=0)
    masked_gradient = np.sum(gradients[mask], axis=0) if retained_count else np.zeros(args.gradient_dim)
    dense_gradient_norm = max(float(np.linalg.norm(dense_gradient)), 1.0e-30)
    gradient_residual = float(np.linalg.norm(dense_gradient - masked_gradient)) / dense_gradient_norm
    gradient_direction = cosine(masked_gradient, dense_gradient)
    gradient_energy = np.einsum("ij,ij->i", gradients, gradients)
    gradient_energy_retained = float(np.sum(gradient_energy[mask])) / max(
        float(np.sum(gradient_energy)),
        1.0e-30,
    )

    dense_state = np.sum(state_deltas, axis=0)
    masked_state = np.sum(state_deltas[mask], axis=0) if retained_count else np.zeros(args.state_dim)
    state_residual = float(np.linalg.norm(dense_state - masked_state)) / max(
        float(np.linalg.norm(dense_state)),
        1.0e-30,
    )

    dense_work = float(args.token_count)
    masked_work = float(retained_count)
    selector_overhead = (
        args.expensive_selector_overhead
        if condition == "selector_cost_fail"
        else args.selector_overhead
    )
    selector_work = float(args.token_count) * selector_overhead
    priced_work_ratio = (masked_work + selector_work) / dense_work
    projected_multiplier = actionable_retained / max(priced_work_ratio, 1.0e-30)
    lattice_noise_avoided = 1.0 - retained_fraction

    failed: list[str] = []
    if not finite(
        retained_fraction,
        actionable_retained,
        gradient_energy_retained,
        gradient_residual,
        gradient_direction,
        state_residual,
        lattice_noise_avoided,
        dense_work,
        masked_work,
        selector_work,
        priced_work_ratio,
        projected_multiplier,
    ):
        failed.append("non_finite_metric")
    if retained_fraction > args.max_retained_fraction:
        failed.append("retention_not_sparse")
    if actionable_retained < args.min_actionable_ce_retained:
        failed.append("actionable_ce_debt")
    if gradient_energy_retained < args.min_gradient_energy_retained:
        failed.append("gradient_energy_debt")
    if gradient_residual > args.max_gradient_residual_ratio:
        failed.append("gradient_residual")
    if gradient_direction < args.min_gradient_cosine:
        failed.append("gradient_direction")
    if state_residual > args.max_state_residual_ratio:
        failed.append("state_update_debt")
    if lattice_noise_avoided < args.min_lattice_noise_avoided:
        failed.append("lattice_noise_not_avoided")
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
        gradient_dim=args.gradient_dim,
        state_dim=args.state_dim,
        retained_fraction=float(retained_fraction),
        actionable_ce_retained_fraction=float(actionable_retained),
        gradient_energy_retained_fraction=float(gradient_energy_retained),
        gradient_residual_ratio=float(gradient_residual),
        gradient_cosine=float(gradient_direction),
        state_residual_ratio=float(state_residual),
        lattice_noise_avoided_fraction=float(lattice_noise_avoided),
        dense_work_units=float(dense_work),
        masked_work_units=float(masked_work),
        selector_work_units=float(selector_work),
        priced_work_ratio=float(priced_work_ratio),
        projected_token_efficiency_multiplier=float(projected_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "token_count",
        "gradient_dim",
        "state_dim",
        "retained_fraction",
        "actionable_ce_retained_fraction",
        "gradient_energy_retained_fraction",
        "gradient_residual_ratio",
        "gradient_cosine",
        "state_residual_ratio",
        "lattice_noise_avoided_fraction",
        "dense_work_units",
        "masked_work_units",
        "selector_work_units",
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
    weak_proxy = [result for result in results if result.condition == "weak_ce_proxy_fail"]
    high_threshold = [result for result in results if result.condition == "threshold_too_high_fail"]
    no_sparsity = [result for result in results if result.condition == "no_surprise_sparsity_fail"]
    state_coupling = [result for result in results if result.condition == "state_coupling_fail"]
    cost = [result for result in results if result.condition == "selector_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_sparse": aggregates["positive"]["retained_fraction_mean"] <= 0.40,
        "positive_condition_preserves_actionable_ce": (
            aggregates["positive"]["actionable_ce_retained_fraction_min"] >= 0.90
        ),
        "positive_condition_preserves_gradient": (
            aggregates["positive"]["gradient_energy_retained_fraction_min"] >= 0.92
            and aggregates["positive"]["gradient_residual_ratio_max"] <= 0.06
            and aggregates["positive"]["gradient_cosine_min"] >= 0.995
        ),
        "positive_condition_preserves_state": (
            aggregates["positive"]["state_residual_ratio_max"] <= 0.06
        ),
        "positive_condition_avoids_lattice_noise": (
            aggregates["positive"]["lattice_noise_avoided_fraction_min"] >= 0.55
        ),
        "positive_condition_priced": (
            aggregates["positive"]["priced_work_ratio_mean"] <= 0.45
            and aggregates["positive"]["projected_token_efficiency_multiplier_mean"] >= 2.0
        ),
        "weak_proxy_control_all_fail_credit_debt": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "gradient_residual" in result.fail_reason
                or "gradient_energy_debt" in result.fail_reason
                or "state_update_debt" in result.fail_reason
            )
            for result in weak_proxy
        ),
        "high_threshold_control_all_fail_ce_or_gradient_debt": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "actionable_ce_debt" in result.fail_reason
                or "gradient_residual" in result.fail_reason
                or "state_update_debt" in result.fail_reason
            )
            for result in high_threshold
        ),
        "no_sparsity_control_all_fail_price_or_noise": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "retention_not_sparse" in result.fail_reason
                or "lattice_noise_not_avoided" in result.fail_reason
                or "price" in result.fail_reason
            )
            for result in no_sparsity
        ),
        "state_coupling_control_all_fail_state_debt": all(
            (not result.passed)
            and result.fail_reason is not None
            and "state_update_debt" in result.fail_reason
            for result in state_coupling
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
        "PASS_R501_S79_IAS_GATE"
        if all(checks.values())
        else "FAIL_R501_S79_IAS_GATE"
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
            "Synthetic CPU information-theoretic active-sensing mask gate only; "
            "not heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S79_IAS_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_retained_fraction_mean={positive['retained_fraction_mean']:.6f} "
        f"positive_actionable_ce_retained_fraction_min={positive['actionable_ce_retained_fraction_min']:.6f} "
        f"positive_gradient_residual_ratio_max={positive['gradient_residual_ratio_max']:.6f} "
        f"positive_gradient_cosine_min={positive['gradient_cosine_min']:.6f} "
        f"positive_state_residual_ratio_max={positive['state_residual_ratio_max']:.6f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.6f} "
        f"positive_projected_token_efficiency_multiplier_mean={positive['projected_token_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
