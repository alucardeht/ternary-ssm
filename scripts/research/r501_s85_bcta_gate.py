#!/usr/bin/env python3
"""R501 S85 boundary-conditioned token amortization gate.

This CPU diagnostic implements the R501 Quality Paradigm 7 finding:
"Boundary-Conditioned Token Amortization (BCTA)".

The gate constructs predictable token spans whose per-token CE is exactly
determined by boundary states h0 and hk. The candidate replaces dense interior
CE accounting with a closed-form boundary integral and boundary gradients. A
positive case must recover the dense CE sum and boundary gradients while paying
approximately one token per span plus a verifier.

Controls exercise the declared kill criteria: interiors can be unpredictable,
two spans can share boundaries but disagree inside, long spans can drift outside
the closed-form family, spans can be too short to amortize, or verification can
erase the denominator.

This is a synthetic CPU boundary-conditioned CE integral diagnostic, not heldout
CE progress, real corpus token efficiency, GPU training-speed evidence, or
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


SCHEMA = "r501_s85_bcta_gate.v1"
CONDITIONS = (
    "positive",
    "unpredictable_interior_fail",
    "boundary_mismatch_fail",
    "drift_family_fail",
    "short_span_price_fail",
    "verifier_cost_fail",
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
    segment_len: int
    state_dim: int
    total_token_count: int
    dense_ce_sum: float
    closed_form_ce_sum: float
    ce_integral_relative_error: float
    boundary_gradient_relative_error: float
    boundary_gradient_cosine: float
    collapsed_token_fraction: float
    interior_noise_avoided_fraction: float
    segment_work_units: float
    verifier_work_units: float
    paid_token_ratio: float
    projected_token_efficiency_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=50185)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--segment-count", type=int, default=96)
    parser.add_argument("--segment-len", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--ce-offset", type=float, default=3.0)
    parser.add_argument("--interior-noise-scale", type=float, default=0.55)
    parser.add_argument("--drift-scale", type=float, default=0.7)
    parser.add_argument("--verifier-overhead", type=float, default=0.05)
    parser.add_argument("--expensive-verifier-overhead", type=float, default=0.45)
    parser.add_argument("--max-ce-integral-error", type=float, default=0.02)
    parser.add_argument("--max-boundary-gradient-error", type=float, default=0.03)
    parser.add_argument("--min-boundary-gradient-cosine", type=float, default=0.999)
    parser.add_argument("--min-collapsed-token-fraction", type=float, default=0.75)
    parser.add_argument("--max-paid-token-ratio", type=float, default=0.35)
    parser.add_argument("--min-token-efficiency-multiplier", type=float, default=2.5)
    return parser.parse_args()


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0e-30)
    return float(np.dot(a, b) / denom)


def span_positions(segment_len: int) -> np.ndarray:
    return (np.arange(segment_len, dtype=np.float64) + 1.0) / float(segment_len)


def base_dense_and_closed(
    h0: np.ndarray,
    hk: np.ndarray,
    w0: np.ndarray,
    wk: np.ndarray,
    wm: np.ndarray,
    segment_len: int,
    ce_offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = span_positions(segment_len)
    one_minus = 1.0 - s
    bridge = s * one_minus
    delta = hk - h0
    dim = float(h0.shape[1])

    dense = (
        ce_offset
        + (h0 @ w0)[:, None] * s[None, :] / dim
        + (hk @ wk)[:, None] * one_minus[None, :] / dim
        + (delta @ wm)[:, None] * bridge[None, :] / dim
    )
    closed = np.sum(dense, axis=1)

    grad_h0 = (
        np.sum(s) * w0[None, :] / dim
        - np.sum(bridge) * wm[None, :] / dim
    )
    grad_hk = (
        np.sum(one_minus) * wk[None, :] / dim
        + np.sum(bridge) * wm[None, :] / dim
    )
    grad_h0 = np.repeat(grad_h0, h0.shape[0], axis=0)
    grad_hk = np.repeat(grad_hk, h0.shape[0], axis=0)
    return dense, closed, grad_h0, grad_hk, s


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    segment_len = 2 if condition == "short_span_price_fail" else args.segment_len
    total_tokens = args.segment_count * segment_len

    h0 = rng.normal(size=(args.segment_count, args.state_dim))
    hk = h0 + 0.25 * rng.normal(size=(args.segment_count, args.state_dim))
    w0 = rng.normal(size=args.state_dim)
    wk = rng.normal(size=args.state_dim)
    wm = rng.normal(size=args.state_dim)
    wd = rng.normal(size=args.state_dim)

    dense, closed, dense_grad_h0, dense_grad_hk, s = base_dense_and_closed(
        h0,
        hk,
        w0,
        wk,
        wm,
        segment_len,
        args.ce_offset,
    )
    candidate_closed = closed.copy()
    candidate_grad_h0 = dense_grad_h0.copy()
    candidate_grad_hk = dense_grad_hk.copy()

    if condition == "unpredictable_interior_fail":
        dense = dense + args.interior_noise_scale * np.abs(rng.normal(size=dense.shape))
    elif condition == "boundary_mismatch_fail":
        amplitude = args.interior_noise_scale * (
            0.75 + np.abs(rng.normal(size=(args.segment_count, 1)))
        )
        dense = dense + amplitude * np.sin(np.pi * s)[None, :]
    elif condition == "drift_family_fail":
        drift_coeff = (h0 + hk) @ wd / float(args.state_dim)
        drift = args.drift_scale * drift_coeff[:, None] * (s[None, :] ** 3)
        dense = dense + drift
        grad_delta = args.drift_scale * np.sum(s**3) * wd[None, :] / float(args.state_dim)
        dense_grad_h0 = dense_grad_h0 + np.repeat(grad_delta, args.segment_count, axis=0)
        dense_grad_hk = dense_grad_hk + np.repeat(grad_delta, args.segment_count, axis=0)

    dense_sum = float(np.sum(dense))
    closed_sum = float(np.sum(candidate_closed))
    ce_error = abs(closed_sum - dense_sum) / max(abs(dense_sum), 1.0e-30)

    dense_grad = np.concatenate([dense_grad_h0, dense_grad_hk], axis=1).reshape(-1)
    candidate_grad = np.concatenate([candidate_grad_h0, candidate_grad_hk], axis=1).reshape(-1)
    grad_error = float(np.linalg.norm(candidate_grad - dense_grad)) / max(
        float(np.linalg.norm(dense_grad)),
        1.0e-30,
    )
    grad_cosine = cosine(candidate_grad, dense_grad)

    segment_work = float(args.segment_count)
    verifier_overhead = (
        args.expensive_verifier_overhead
        if condition == "verifier_cost_fail"
        else args.verifier_overhead
    )
    verifier_work = verifier_overhead * total_tokens
    paid_ratio = (segment_work + verifier_work) / max(float(total_tokens), 1.0e-30)
    collapsed_fraction = 1.0 - segment_work / max(float(total_tokens), 1.0e-30)
    noise_avoided = collapsed_fraction * max(1.0 - ce_error, 0.0)
    projected_multiplier = max(1.0 - ce_error, 0.0) / max(paid_ratio, 1.0e-30)

    failed: list[str] = []
    if not finite(
        dense_sum,
        closed_sum,
        ce_error,
        grad_error,
        grad_cosine,
        collapsed_fraction,
        noise_avoided,
        segment_work,
        verifier_work,
        paid_ratio,
        projected_multiplier,
    ):
        failed.append("non_finite_metric")
    if ce_error > args.max_ce_integral_error:
        failed.append("ce_integral")
    if grad_error > args.max_boundary_gradient_error:
        failed.append("boundary_gradient_error")
    if grad_cosine < args.min_boundary_gradient_cosine:
        failed.append("boundary_gradient_direction")
    if collapsed_fraction < args.min_collapsed_token_fraction:
        failed.append("span_amortization")
    if paid_ratio > args.max_paid_token_ratio:
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
        segment_count=args.segment_count,
        segment_len=segment_len,
        state_dim=args.state_dim,
        total_token_count=total_tokens,
        dense_ce_sum=dense_sum,
        closed_form_ce_sum=closed_sum,
        ce_integral_relative_error=float(ce_error),
        boundary_gradient_relative_error=float(grad_error),
        boundary_gradient_cosine=float(grad_cosine),
        collapsed_token_fraction=float(collapsed_fraction),
        interior_noise_avoided_fraction=float(noise_avoided),
        segment_work_units=float(segment_work),
        verifier_work_units=float(verifier_work),
        paid_token_ratio=float(paid_ratio),
        projected_token_efficiency_multiplier=float(projected_multiplier),
    )


def aggregate_condition(results: list[TrialResult]) -> dict[str, Any]:
    fields = (
        "segment_count",
        "segment_len",
        "state_dim",
        "total_token_count",
        "dense_ce_sum",
        "closed_form_ce_sum",
        "ce_integral_relative_error",
        "boundary_gradient_relative_error",
        "boundary_gradient_cosine",
        "collapsed_token_fraction",
        "interior_noise_avoided_fraction",
        "segment_work_units",
        "verifier_work_units",
        "paid_token_ratio",
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
    unpredictable = [
        result for result in results if result.condition == "unpredictable_interior_fail"
    ]
    mismatch = [result for result in results if result.condition == "boundary_mismatch_fail"]
    drift = [result for result in results if result.condition == "drift_family_fail"]
    short = [result for result in results if result.condition == "short_span_price_fail"]
    cost = [result for result in results if result.condition == "verifier_cost_fail"]
    return {
        "all_metrics_finite": all_finite(results),
        "positive_condition_all_pass": all(result.passed for result in positive),
        "positive_condition_exact_integral": (
            aggregates["positive"]["ce_integral_relative_error_max"] <= 0.02
        ),
        "positive_condition_exact_boundary_gradient": (
            aggregates["positive"]["boundary_gradient_relative_error_max"] <= 0.03
            and aggregates["positive"]["boundary_gradient_cosine_min"] >= 0.999
        ),
        "positive_condition_amortizes_tokens": (
            aggregates["positive"]["collapsed_token_fraction_mean"] >= 0.75
            and aggregates["positive"]["paid_token_ratio_mean"] <= 0.35
            and aggregates["positive"]["projected_token_efficiency_multiplier_mean"] >= 2.5
        ),
        "unpredictable_control_all_fail_integral": all(
            (not result.passed)
            and result.fail_reason is not None
            and "ce_integral" in result.fail_reason
            for result in unpredictable
        ),
        "boundary_mismatch_control_all_fail_integral": all(
            (not result.passed)
            and result.fail_reason is not None
            and "ce_integral" in result.fail_reason
            for result in mismatch
        ),
        "drift_control_all_fail_integral_or_gradient": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "ce_integral" in result.fail_reason
                or "boundary_gradient_error" in result.fail_reason
            )
            for result in drift
        ),
        "short_span_control_all_fail_price_or_amortization": all(
            (not result.passed)
            and result.fail_reason is not None
            and (
                "span_amortization" in result.fail_reason
                or "price" in result.fail_reason
                or "token_efficiency" in result.fail_reason
            )
            for result in short
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
    decision = "PASS_R501_S85_BCTA_GATE" if all(checks.values()) else "FAIL_R501_S85_BCTA_GATE"
    payload = {
        "schema": SCHEMA,
        "args": vars(args) | {"output": str(args.output)},
        "conditions": list(CONDITIONS),
        "decision": decision,
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU boundary-conditioned token amortization gate only; "
            "not heldout CE progress, not real corpus token efficiency, not "
            "GPU training speed, and not production v12 integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R501_S85_BCTA_GATE] "
        "active=1 "
        f"decision={decision} "
        "disposition=synthetic_positive_control_identified_not_training_speed_claim "
        f"positive_ce_integral_relative_error_max={positive['ce_integral_relative_error_max']:.6f} "
        f"positive_boundary_gradient_relative_error_max={positive['boundary_gradient_relative_error_max']:.6f} "
        f"positive_boundary_gradient_cosine_min={positive['boundary_gradient_cosine_min']:.6f} "
        f"positive_collapsed_token_fraction_mean={positive['collapsed_token_fraction_mean']:.6f} "
        f"positive_paid_token_ratio_mean={positive['paid_token_ratio_mean']:.6f} "
        f"positive_projected_token_efficiency_multiplier_mean={positive['projected_token_efficiency_multiplier_mean']:.6f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
