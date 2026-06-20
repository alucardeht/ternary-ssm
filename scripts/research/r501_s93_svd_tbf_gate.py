#!/usr/bin/env python3
"""R501 S93 singular-value directed ternary bit-flip gate.

This CPU diagnostic implements the R501 Quality Paradigm 15 finding:
"Singular-Value Directed Ternary Bit-Flips (SVD-TBF)".

The gate constructs token-gradient matrices with a zero-mean but high-variance
capability direction. Raw element-wise batch-gradient signs are intentionally
non-identifying because the signal cancels in the mean; the positive condition
must recover the dominant right singular vector, ternary-project it onto a
small flip support, and preserve the true capability direction while paying
only sparse flip plus SVD pricing.

Controls exercise the declared kill criteria: raw element-wise flips miss the
zero-mean capability, random directions miss it, a low singular gap makes the
top direction non-identifying, continuous SVD vectors violate the ternary
realization boundary, and dense/expensive SVD pricing erases the denominator.

This is a synthetic CPU SVD-directed ternary bit-flip diagnostic, not heldout CE
progress, real corpus token efficiency, GPU training-speed evidence, or
production v12 integration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "r501_s93_svd_tbf_gate.v1"
CONDITIONS = (
    "positive",
    "raw_elementwise_fail",
    "random_direction_fail",
    "low_singular_gap_fail",
    "nonternary_direction_fail",
    "svd_cost_fail",
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
    support_size: int
    top_singular_energy_fraction: float
    singular_gap_ratio: float
    top_vector_alignment: float
    ternary_flip_alignment: float
    capability_capture_fraction: float
    raw_flip_capture_fraction: float
    candidate_vs_raw_capture_gain: float
    sign_mismatch_fraction: float
    ternary_violation_count: int
    selected_support_fraction: float
    svd_overhead_ratio: float
    paid_work_ratio: float
    projected_flip_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50193)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--token-count", type=int, default=512)
    parser.add_argument("--gradient-dim", type=int, default=64)
    parser.add_argument("--support-size", type=int, default=12)
    parser.add_argument("--dense-support-size", type=int, default=56)
    parser.add_argument("--signal-strength", type=float, default=4.0)
    parser.add_argument("--secondary-signal-strength", type=float, default=3.85)
    parser.add_argument("--noise-scale", type=float, default=0.30)
    parser.add_argument("--svd-overhead", type=float, default=0.08)
    parser.add_argument("--expensive-svd-overhead", type=float, default=0.45)
    parser.add_argument("--min-top-energy", type=float, default=0.55)
    parser.add_argument("--min-singular-gap", type=float, default=1.35)
    parser.add_argument("--min-top-vector-alignment", type=float, default=0.92)
    parser.add_argument("--min-ternary-flip-alignment", type=float, default=0.88)
    parser.add_argument("--min-capability-capture", type=float, default=0.78)
    parser.add_argument("--min-capture-gain", type=float, default=3.0)
    parser.add_argument("--max-sign-mismatch", type=float, default=0.08)
    parser.add_argument("--max-ternary-violations", type=int, default=0)
    parser.add_argument("--max-paid-work-ratio", type=float, default=0.35)
    parser.add_argument("--min-efficiency-multiplier", type=float, default=2.5)
    return parser.parse_args()


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1.0e-30)


def make_sparse_ternary_direction(
    rng: np.random.Generator,
    dim: int,
    support_size: int,
    forbidden: set[int] | None = None,
) -> np.ndarray:
    forbidden = forbidden or set()
    candidates = np.array([index for index in range(dim) if index not in forbidden])
    if support_size > len(candidates):
        raise ValueError("support_size exceeds available coordinates")
    support = rng.choice(candidates, size=support_size, replace=False)
    direction = np.zeros(dim, dtype=np.float64)
    direction[support] = rng.choice(np.array([-1.0, 1.0]), size=support_size)
    return normalize(direction)


def balanced_coefficients(
    rng: np.random.Generator,
    token_count: int,
    strength: float,
) -> np.ndarray:
    half = token_count // 2
    signs = np.concatenate([np.ones(half), -np.ones(token_count - half)])
    rng.shuffle(signs)
    return signs * strength


def build_gradient_matrix(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    true_direction = make_sparse_ternary_direction(
        rng, args.gradient_dim, args.support_size
    )
    primary = balanced_coefficients(rng, args.token_count, args.signal_strength)
    gradients = primary[:, None] * true_direction[None, :]

    if condition == "low_singular_gap_fail":
        primary_support = set(np.flatnonzero(true_direction))
        secondary_direction = make_sparse_ternary_direction(
            rng, args.gradient_dim, args.support_size, forbidden=primary_support
        )
        secondary = balanced_coefficients(
            rng, args.token_count, args.secondary_signal_strength
        )
        gradients = gradients + secondary[:, None] * secondary_direction[None, :]

    gradients = gradients + rng.normal(
        size=gradients.shape,
        scale=args.noise_scale,
    )
    return gradients, true_direction


def orient(vector: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return vector if float(np.dot(vector, reference)) >= 0.0 else -vector


def ternary_project(vector: np.ndarray, support_size: int) -> np.ndarray:
    active = np.argsort(np.abs(vector))[-support_size:]
    projected = np.zeros_like(vector)
    signs = np.sign(vector[active])
    signs[signs == 0.0] = 1.0
    projected[active] = signs
    return normalize(projected)


def raw_elementwise_update(gradients: np.ndarray, support_size: int) -> np.ndarray:
    mean_gradient = np.mean(gradients, axis=0)
    if np.allclose(mean_gradient, 0.0):
        mean_gradient = np.sum(gradients, axis=0)
    return ternary_project(mean_gradient, support_size)


def random_ternary_update(
    rng: np.random.Generator,
    dim: int,
    support_size: int,
) -> np.ndarray:
    support = rng.choice(dim, size=support_size, replace=False)
    update = np.zeros(dim, dtype=np.float64)
    update[support] = rng.choice(np.array([-1.0, 1.0]), size=support_size)
    return normalize(update)


def ternary_violation_count(update: np.ndarray) -> int:
    rounded = np.isclose(update, -1.0) | np.isclose(update, 0.0) | np.isclose(update, 1.0)
    normalized_ternary = np.isclose(np.abs(update), 1.0 / math.sqrt(np.count_nonzero(update)))
    allowed = (np.isclose(update, 0.0)) | normalized_ternary
    return int(np.size(update) - np.count_nonzero(rounded | allowed))


def sign_mismatch_fraction(update: np.ndarray, true_direction: np.ndarray) -> float:
    true_support = np.flatnonzero(true_direction)
    update_sign = np.sign(update[true_support])
    true_sign = np.sign(true_direction[true_support])
    mismatch_a = np.mean(update_sign != true_sign)
    mismatch_b = np.mean(update_sign != -true_sign)
    return float(min(mismatch_a, mismatch_b))


def first_failure(
    *,
    top_singular_energy_fraction: float,
    singular_gap_ratio: float,
    top_vector_alignment: float,
    ternary_flip_alignment: float,
    capability_capture_fraction: float,
    candidate_vs_raw_capture_gain: float,
    sign_mismatch_fraction_value: float,
    ternary_violation_count_value: int,
    paid_work_ratio: float,
    projected_flip_efficiency_multiplier: float,
    args: argparse.Namespace,
) -> str | None:
    if not finite(
        top_singular_energy_fraction,
        singular_gap_ratio,
        top_vector_alignment,
        ternary_flip_alignment,
        capability_capture_fraction,
        candidate_vs_raw_capture_gain,
        sign_mismatch_fraction_value,
        paid_work_ratio,
        projected_flip_efficiency_multiplier,
    ):
        return "nonfinite_metric"
    if ternary_violation_count_value > args.max_ternary_violations:
        return "nonternary_update"
    if paid_work_ratio > args.max_paid_work_ratio:
        return "price_paid_work"
    if projected_flip_efficiency_multiplier < args.min_efficiency_multiplier:
        return "price_efficiency_multiplier"
    if top_singular_energy_fraction < args.min_top_energy:
        return "singular_energy"
    if singular_gap_ratio < args.min_singular_gap:
        return "singular_gap"
    if top_vector_alignment < args.min_top_vector_alignment:
        return "top_vector_alignment"
    if ternary_flip_alignment < args.min_ternary_flip_alignment:
        return "ternary_flip_alignment"
    if capability_capture_fraction < args.min_capability_capture:
        return "capability_capture"
    if candidate_vs_raw_capture_gain < args.min_capture_gain:
        return "raw_flip_gain"
    if sign_mismatch_fraction_value > args.max_sign_mismatch:
        return "sign_mismatch"
    return None


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    gradients, true_direction = build_gradient_matrix(rng, args, condition)
    _, singular_values, vt = np.linalg.svd(gradients, full_matrices=False)
    top_vector = orient(vt[0], true_direction)
    support_size = (
        args.dense_support_size if condition == "svd_cost_fail" else args.support_size
    )

    raw_update = raw_elementwise_update(gradients, args.support_size)
    raw_update = orient(raw_update, true_direction)
    raw_capture = float(np.dot(raw_update, true_direction) ** 2)

    if condition == "raw_elementwise_fail":
        update = raw_update
    elif condition == "random_direction_fail":
        update = orient(
            random_ternary_update(rng, args.gradient_dim, args.support_size),
            true_direction,
        )
    elif condition == "nonternary_direction_fail":
        update = normalize(top_vector)
    else:
        update = ternary_project(top_vector, support_size)
        update = orient(update, true_direction)

    top_energy = float((singular_values[0] ** 2) / max(np.sum(singular_values**2), 1.0e-30))
    singular_gap = float(singular_values[0] / max(singular_values[1], 1.0e-30))
    top_alignment = float(abs(np.dot(top_vector, true_direction)))
    flip_alignment = float(abs(np.dot(update, true_direction)))
    capture = float(flip_alignment**2)
    capture_gain = float(capture / max(raw_capture, 1.0e-12))
    mismatch = sign_mismatch_fraction(update, true_direction)
    violations = ternary_violation_count(update)
    support_fraction = float(np.count_nonzero(update) / args.gradient_dim)
    svd_overhead = (
        args.expensive_svd_overhead if condition == "svd_cost_fail" else args.svd_overhead
    )
    paid_ratio = float(support_fraction + svd_overhead)
    multiplier = float(1.0 / max(paid_ratio, 1.0e-30))

    fail_reason = first_failure(
        top_singular_energy_fraction=top_energy,
        singular_gap_ratio=singular_gap,
        top_vector_alignment=top_alignment,
        ternary_flip_alignment=flip_alignment,
        capability_capture_fraction=capture,
        candidate_vs_raw_capture_gain=capture_gain,
        sign_mismatch_fraction_value=mismatch,
        ternary_violation_count_value=violations,
        paid_work_ratio=paid_ratio,
        projected_flip_efficiency_multiplier=multiplier,
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
        token_count=args.token_count,
        gradient_dim=args.gradient_dim,
        support_size=support_size,
        top_singular_energy_fraction=top_energy,
        singular_gap_ratio=singular_gap,
        top_vector_alignment=top_alignment,
        ternary_flip_alignment=flip_alignment,
        capability_capture_fraction=capture,
        raw_flip_capture_fraction=raw_capture,
        candidate_vs_raw_capture_gain=capture_gain,
        sign_mismatch_fraction=mismatch,
        ternary_violation_count=violations,
        selected_support_fraction=support_fraction,
        svd_overhead_ratio=svd_overhead,
        paid_work_ratio=paid_ratio,
        projected_flip_efficiency_multiplier=multiplier,
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "token_count",
        "gradient_dim",
        "support_size",
        "top_singular_energy_fraction",
        "singular_gap_ratio",
        "top_vector_alignment",
        "ternary_flip_alignment",
        "capability_capture_fraction",
        "raw_flip_capture_fraction",
        "candidate_vs_raw_capture_gain",
        "sign_mismatch_fraction",
        "ternary_violation_count",
        "selected_support_fraction",
        "svd_overhead_ratio",
        "paid_work_ratio",
        "projected_flip_efficiency_multiplier",
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
    raw = [result for result in results if result.condition == "raw_elementwise_fail"]
    random = [result for result in results if result.condition == "random_direction_fail"]
    low_gap = [result for result in results if result.condition == "low_singular_gap_fail"]
    nonternary = [result for result in results if result.condition == "nonternary_direction_fail"]
    cost = [result for result in results if result.condition == "svd_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_recovers_svd_direction": (
            aggregates["positive"]["top_vector_alignment_min"] >= 0.92
            and aggregates["positive"]["top_singular_energy_fraction_min"] >= 0.55
            and aggregates["positive"]["singular_gap_ratio_min"] >= 1.35
        ),
        "positive_condition_ternary_capture": (
            aggregates["positive"]["ternary_flip_alignment_min"] >= 0.88
            and aggregates["positive"]["capability_capture_fraction_min"] >= 0.78
            and aggregates["positive"]["sign_mismatch_fraction_max"] <= 0.08
            and aggregates["positive"]["ternary_violation_count_max"] == 0.0
        ),
        "positive_condition_filters_raw_flips": (
            aggregates["positive"]["candidate_vs_raw_capture_gain_min"] >= 3.0
        ),
        "positive_condition_priced": (
            aggregates["positive"]["paid_work_ratio_mean"] <= 0.35
            and aggregates["positive"]["projected_flip_efficiency_multiplier_mean"] >= 2.5
        ),
        "raw_control_all_fail_capture_or_gain": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "capability_capture" in result.fail_reason
                or "raw_flip_gain" in result.fail_reason
                or "ternary_flip_alignment" in result.fail_reason
                or "sign_mismatch" in result.fail_reason
            )
            for result in raw
        ),
        "random_control_all_fail_alignment_or_capture": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "top_vector_alignment" in result.fail_reason
                or "ternary_flip_alignment" in result.fail_reason
                or "capability_capture" in result.fail_reason
                or "sign_mismatch" in result.fail_reason
            )
            for result in random
        ),
        "low_gap_control_all_fail_singular_gap_or_energy": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "singular_gap" in result.fail_reason
                or "singular_energy" in result.fail_reason
            )
            for result in low_gap
        ),
        "nonternary_control_all_fail_ternary_boundary": all(
            (not result.passed)
            and result.fail_reason is not None
            and "nonternary_update" in result.fail_reason
            for result in nonternary
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
    decision = "PASS_R501_S93_SVD_TBF_GATE" if all(checks.values()) else "FAIL_R501_S93_SVD_TBF_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU singular-value directed ternary bit-flip gate only; "
            "not heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S93_SVD_TBF_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_top_singular_energy_fraction_min={positive['top_singular_energy_fraction_min']:.6f} "
        f"positive_singular_gap_ratio_min={positive['singular_gap_ratio_min']:.6f} "
        f"positive_top_vector_alignment_min={positive['top_vector_alignment_min']:.6f} "
        f"positive_ternary_flip_alignment_min={positive['ternary_flip_alignment_min']:.6f} "
        f"positive_capability_capture_fraction_min={positive['capability_capture_fraction_min']:.6f} "
        f"positive_candidate_vs_raw_capture_gain_min={positive['candidate_vs_raw_capture_gain_min']:.6f} "
        f"positive_paid_work_ratio_mean={positive['paid_work_ratio_mean']:.6f} "
        f"positive_projected_flip_efficiency_multiplier_mean={positive['projected_flip_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
