#!/usr/bin/env python3
"""R502-S19 Segment-Boundary Adjoint Sheaf Matching gate.

This synthetic CPU gate checks whether segmented local adjoints plus sparse
boundary matching can reconstruct the global recurrent adjoint while preserving
a parallel wall-clock denominator. It includes controls for dense boundary
flux, missing matches, wrong flux estimates, expensive boundary reconciliation,
and long-tail spectral risk.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


CONDITIONS = [
    "positive",
    "dense_flux_fail",
    "no_match_fail",
    "wrong_flux_fail",
    "boundary_cost_fail",
    "spectral_tail_fail",
]


@dataclass
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str
    sequence_len: int
    state_dim: int
    segment_count: int
    segment_len: int
    boundary_count: int
    matched_boundary_fraction: float
    high_flux_fraction: float
    relative_adjoint_error: float
    boundary_error_fraction: float
    descent_cosine: float
    priced_work_ratio: float
    local_parallel_ratio: float
    boundary_match_cost_ratio: float
    spectral_radius: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=50219)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--segment-count", type=int, default=16)
    parser.add_argument("--segment-len", type=int, default=16)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--low-segment-scale", type=float, default=0.01)
    parser.add_argument("--high-segment-scale", type=float, default=1.0)
    parser.add_argument("--high-segment-count", type=int, default=4)
    parser.add_argument("--flux-threshold", type=float, default=0.35)
    parser.add_argument("--boundary-match-cost-ratio", type=float, default=0.95)
    parser.add_argument("--expensive-boundary-match-cost-ratio", type=float, default=4.0)
    parser.add_argument("--schedule-overhead-ratio", type=float, default=0.04)
    parser.add_argument("--max-relative-adjoint-error", type=float, default=0.05)
    parser.add_argument("--max-boundary-error-fraction", type=float, default=0.05)
    parser.add_argument("--max-matched-boundary-fraction", type=float, default=0.80)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.90)
    parser.add_argument("--max-spectral-radius", type=float, default=0.98)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def make_decays(rng: np.random.Generator, args: argparse.Namespace, condition: str) -> np.ndarray:
    if condition == "spectral_tail_fail":
        return rng.uniform(0.985, 0.997, size=args.state_dim)
    return rng.uniform(0.60, 0.86, size=args.state_dim)


def make_segment_scales(
    rng: np.random.Generator,
    args: argparse.Namespace,
    condition: str,
) -> np.ndarray:
    if condition == "dense_flux_fail":
        return np.full(args.segment_count, args.high_segment_scale, dtype=np.float64)
    scales = np.full(args.segment_count, args.low_segment_scale, dtype=np.float64)
    high_count = min(args.high_segment_count, args.segment_count)
    high_indices = rng.choice(args.segment_count, size=high_count, replace=False)
    scales[high_indices] = args.high_segment_scale
    return scales


def full_backward(gradients: np.ndarray, decays: np.ndarray) -> np.ndarray:
    lambdas = np.zeros_like(gradients)
    carry = np.zeros(gradients.shape[1], dtype=np.float64)
    for t in range(gradients.shape[0] - 1, -1, -1):
        carry = gradients[t] + decays * carry
        lambdas[t] = carry
    return lambdas


def local_segment_backward(gradients: np.ndarray, decays: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    local = np.zeros_like(gradients)
    for segment in range(args.segment_count):
        start = segment * args.segment_len
        end = start + args.segment_len
        carry = np.zeros(args.state_dim, dtype=np.float64)
        for t in range(end - 1, start - 1, -1):
            carry = gradients[t] + decays * carry
            local[t] = carry
    return local


def apply_boundary_matches(
    local: np.ndarray,
    exact: np.ndarray,
    decays: np.ndarray,
    matched: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    reconstructed = local.copy()
    for boundary_index, should_match in enumerate(matched):
        if not should_match:
            continue
        segment = boundary_index
        start = segment * args.segment_len
        end = start + args.segment_len
        next_start = end
        carry = decays * exact[next_start]
        for t in range(end - 1, start - 1, -1):
            reconstructed[t] += carry
            carry = decays * carry
    return reconstructed


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / denom)


def relative_norm(numerator: np.ndarray, denominator: np.ndarray) -> float:
    denom = float(np.linalg.norm(denominator))
    if denom == 0.0:
        return 0.0 if float(np.linalg.norm(numerator)) == 0.0 else math.inf
    return float(np.linalg.norm(numerator) / denom)


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    sequence_len = args.segment_count * args.segment_len
    decays = make_decays(rng, args, condition)
    scales = make_segment_scales(rng, args, condition)
    gradients = np.zeros((sequence_len, args.state_dim), dtype=np.float64)
    for segment, scale in enumerate(scales):
        start = segment * args.segment_len
        end = start + args.segment_len
        gradients[start:end] = rng.normal(scale=scale, size=(args.segment_len, args.state_dim))

    exact = full_backward(gradients, decays)
    local = local_segment_backward(gradients, decays, args)

    fluxes = []
    for boundary in range(args.segment_count - 1):
        next_start = (boundary + 1) * args.segment_len
        fluxes.append(float(np.linalg.norm(decays * exact[next_start])))
    fluxes_np = np.asarray(fluxes, dtype=np.float64)
    threshold = math.inf if condition == "no_match_fail" else args.flux_threshold
    matched = fluxes_np >= threshold
    if condition == "wrong_flux_fail":
        matched = matched & ((np.arange(matched.size) % 2) == 0)

    reconstructed = apply_boundary_matches(local, exact, decays, matched, args)
    error = exact - reconstructed
    relative_error = relative_norm(error, exact)
    boundary_error = 0.0
    for boundary in range(args.segment_count - 1):
        end = (boundary + 1) * args.segment_len - 1
        boundary_error += float(np.dot(error[end], error[end]))
    exact_energy = float(np.sum(exact * exact))
    boundary_error_fraction = 0.0 if exact_energy == 0.0 else boundary_error / exact_energy

    dense_grad = exact.T @ gradients
    sheaf_grad = reconstructed.T @ gradients
    direction = cosine(sheaf_grad, dense_grad)

    matched_fraction = float(np.mean(matched)) if matched.size else 0.0
    high_flux_fraction = float(np.mean(fluxes_np >= args.flux_threshold)) if fluxes_np.size else 0.0
    local_parallel_ratio = 1.0 / args.segment_count
    boundary_cost = (
        args.expensive_boundary_match_cost_ratio
        if condition == "boundary_cost_fail"
        else args.boundary_match_cost_ratio
    )
    price = local_parallel_ratio + matched_fraction * boundary_cost + args.schedule_overhead_ratio
    spectral_radius = float(np.max(np.abs(decays)))

    failed: list[str] = []
    if not np.isfinite(
        [
            relative_error,
            boundary_error_fraction,
            direction,
            matched_fraction,
            high_flux_fraction,
            price,
            spectral_radius,
        ]
    ).all():
        failed.append("non_finite_metric")
    if relative_error > args.max_relative_adjoint_error:
        failed.append("adjoint_reconstruction")
    if boundary_error_fraction > args.max_boundary_error_fraction:
        failed.append("boundary_error")
    if direction < 0.98:
        failed.append("descent_alignment")
    if high_flux_fraction > args.max_matched_boundary_fraction:
        failed.append("dense_boundary_flux")
    if price > args.max_priced_work_ratio:
        failed.append("priced_work")
    if spectral_radius > args.max_spectral_radius:
        failed.append("spectral_tail_risk")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed),
        sequence_len=sequence_len,
        state_dim=args.state_dim,
        segment_count=args.segment_count,
        segment_len=args.segment_len,
        boundary_count=args.segment_count - 1,
        matched_boundary_fraction=matched_fraction,
        high_flux_fraction=high_flux_fraction,
        relative_adjoint_error=relative_error,
        boundary_error_fraction=boundary_error_fraction,
        descent_cosine=direction,
        priced_work_ratio=price,
        local_parallel_ratio=local_parallel_ratio,
        boundary_match_cost_ratio=boundary_cost,
        spectral_radius=spectral_radius,
    )


NUMERIC_FIELDS = [
    "sequence_len",
    "state_dim",
    "segment_count",
    "segment_len",
    "boundary_count",
    "matched_boundary_fraction",
    "high_flux_fraction",
    "relative_adjoint_error",
    "boundary_error_fraction",
    "descent_cosine",
    "priced_work_ratio",
    "local_parallel_ratio",
    "boundary_match_cost_ratio",
    "spectral_radius",
]


def aggregate(results: list[TrialResult]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition in CONDITIONS:
        subset = [result for result in results if result.condition == condition]
        summary: dict[str, Any] = {
            "count": len(subset),
            "passed_count": sum(result.passed for result in subset),
            "identifying_correct_count": sum(result.identifying_correct for result in subset),
            "fail_reasons": sorted({result.fail_reason for result in subset if result.fail_reason}),
        }
        for field in NUMERIC_FIELDS:
            values = [float(getattr(result, field)) for result in subset]
            summary[f"{field}_mean"] = float(np.mean(values))
            summary[f"{field}_max"] = float(np.max(values))
            summary[f"{field}_min"] = float(np.min(values))
        output[condition] = summary
    return output


def required_checks(results: list[TrialResult], aggregates: dict[str, Any]) -> dict[str, bool]:
    by_condition = {
        condition: [result for result in results if result.condition == condition]
        for condition in CONDITIONS
    }
    return {
        "all_metrics_finite": all(
            math.isfinite(float(value))
            for result in results
            for value in asdict(result).values()
            if isinstance(value, (int, float, np.floating))
        ),
        "all_conditions_identified": all(result.identifying_correct for result in results),
        "positive_condition_all_pass": all(result.passed for result in by_condition["positive"]),
        "positive_condition_reconstruction_and_price": (
            aggregates["positive"]["relative_adjoint_error_max"] <= 0.05
            and aggregates["positive"]["boundary_error_fraction_max"] <= 0.05
            and aggregates["positive"]["descent_cosine_min"] >= 0.98
            and aggregates["positive"]["priced_work_ratio_max"] <= 0.90
            and aggregates["positive"]["high_flux_fraction_max"] <= 0.80
        ),
        "dense_flux_control_all_fail_flux_or_price": all(
            (not result.passed)
            and (
                "dense_boundary_flux" in result.fail_reason
                or "priced_work" in result.fail_reason
            )
            for result in by_condition["dense_flux_fail"]
        ),
        "no_match_control_all_fail_reconstruction": all(
            (not result.passed)
            and (
                "adjoint_reconstruction" in result.fail_reason
                or "boundary_error" in result.fail_reason
            )
            for result in by_condition["no_match_fail"]
        ),
        "wrong_flux_control_all_fail_reconstruction": all(
            (not result.passed)
            and (
                "adjoint_reconstruction" in result.fail_reason
                or "boundary_error" in result.fail_reason
                or "descent_alignment" in result.fail_reason
            )
            for result in by_condition["wrong_flux_fail"]
        ),
        "boundary_cost_control_all_fail_price": all(
            (not result.passed) and "priced_work" in result.fail_reason
            for result in by_condition["boundary_cost_fail"]
        ),
        "spectral_tail_control_all_fail_tail": all(
            (not result.passed)
            and (
                "spectral_tail_risk" in result.fail_reason
                or "adjoint_reconstruction" in result.fail_reason
            )
            for result in by_condition["spectral_tail_fail"]
        ),
    }


def main() -> int:
    args = parse_args()
    results = [
        run_trial(args, condition, args.seed + index)
        for index in range(args.seed_count)
        for condition in CONDITIONS
    ]
    aggregates = aggregate(results)
    checks = required_checks(results, aggregates)
    decision = "PASS_R502_S19_SEGMENT_SHEAF_GATE" if all(checks.values()) else "FAIL_R502_S19_SEGMENT_SHEAF_GATE"

    payload = {
        "round": "R502",
        "canonical_id": "R502-S19",
        "aspect": "Speed",
        "finding": "Segment-Boundary Adjoint Sheaf Matching",
        "decision": decision,
        "disposition": "synthetic_positive_control_identified_not_training_speed_claim",
        "args": vars(args) | {"output": str(args.output) if args.output else None},
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU segment-boundary adjoint sheaf gate only; not heldout "
            "CE progress, not real corpus token efficiency, not GPU training "
            "speed, and not production v12 integration."
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_S19_SEGMENT_SHEAF_GATE] "
        f"active=1 decision={decision} "
        f"disposition={payload['disposition']} "
        f"positive_matched_boundary_fraction_mean={positive['matched_boundary_fraction_mean']:.12f} "
        f"positive_relative_adjoint_error_max={positive['relative_adjoint_error_max']:.12e} "
        f"positive_boundary_error_fraction_max={positive['boundary_error_fraction_max']:.12e} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.12f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
