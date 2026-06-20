#!/usr/bin/env python3
"""R502-S22 Fourier-Domain Adjoint Convolution gate.

This synthetic CPU gate checks whether a time-invariant linear SSM segment can
compute adjoints as FFT-based causal convolutions while preserving the dense
sequential adjoint and beating a priced dense-matmul denominator. Controls cover
short segments, time-varying transitions, non-diagonal transitions, circular
aliasing, and FFT constant-factor failure.
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
    "small_segment_fail",
    "time_varying_fail",
    "non_diagonal_fail",
    "circular_alias_fail",
    "fft_cost_fail",
]


@dataclass
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str
    segment_len: int
    state_dim: int
    fft_len: int
    transition_radius: float
    offdiag_fraction: float
    transition_variation_linf: float
    relative_adjoint_error: float
    max_abs_adjoint_error: float
    descent_cosine: float
    dense_work_units: float
    fdac_work_units: float
    priced_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=50222)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--segment-len", type=int, default=512)
    parser.add_argument("--small-segment-len", type=int, default=16)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--transition-radius", type=float, default=0.74)
    parser.add_argument("--max-relative-adjoint-error", type=float, default=1e-10)
    parser.add_argument("--min-descent-cosine", type=float, default=0.999999999)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.90)
    parser.add_argument("--fft-cost-constant", type=float, default=0.35)
    parser.add_argument("--expensive-fft-cost-constant", type=float, default=3.0)
    parser.add_argument("--fft-launch-overhead-units", type=float, default=24000.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def make_gradients(rng: np.random.Generator, segment_len: int, state_dim: int) -> np.ndarray:
    gradients = rng.normal(scale=0.05, size=(segment_len, state_dim))
    gradients += 0.03 * np.sin(np.linspace(0.0, 8.0 * np.pi, segment_len))[:, None]
    gradients[::37] += rng.normal(scale=0.12, size=(len(gradients[::37]), state_dim))
    return gradients.astype(np.float64)


def exact_diagonal_adjoint(gradients: np.ndarray, diagonal: np.ndarray) -> np.ndarray:
    output = np.zeros_like(gradients)
    carry = np.zeros(gradients.shape[1], dtype=np.float64)
    for t in range(gradients.shape[0] - 1, -1, -1):
        carry = gradients[t] + diagonal * carry
        output[t] = carry
    return output


def exact_time_varying_adjoint(gradients: np.ndarray, diagonals: np.ndarray) -> np.ndarray:
    output = np.zeros_like(gradients)
    carry = np.zeros(gradients.shape[1], dtype=np.float64)
    for t in range(gradients.shape[0] - 1, -1, -1):
        carry = gradients[t] + diagonals[t] * carry
        output[t] = carry
    return output


def exact_dense_adjoint(gradients: np.ndarray, transition: np.ndarray) -> np.ndarray:
    output = np.zeros_like(gradients)
    carry = np.zeros(gradients.shape[1], dtype=np.float64)
    transpose = transition.T
    for t in range(gradients.shape[0] - 1, -1, -1):
        carry = gradients[t] + transpose @ carry
        output[t] = carry
    return output


def fdac_diagonal_adjoint(
    gradients: np.ndarray,
    diagonal: np.ndarray,
    *,
    linear_padding: bool,
) -> tuple[np.ndarray, int]:
    segment_len, state_dim = gradients.shape
    fft_len = next_power_of_two((2 * segment_len) - 1) if linear_padding else segment_len
    reversed_gradients = gradients[::-1]
    output_reversed = np.zeros_like(gradients)
    for dim in range(state_dim):
        kernel = diagonal[dim] ** np.arange(segment_len, dtype=np.float64)
        grad_fft = np.fft.rfft(reversed_gradients[:, dim], n=fft_len)
        kernel_fft = np.fft.rfft(kernel, n=fft_len)
        convolved = np.fft.irfft(grad_fft * kernel_fft, n=fft_len)
        output_reversed[:, dim] = convolved[:segment_len]
    return output_reversed[::-1], fft_len


def priced_work(
    segment_len: int,
    state_dim: int,
    fft_len: int,
    fft_cost_constant: float,
    launch_overhead_units: float,
) -> tuple[float, float, float, float]:
    dense = float(segment_len * state_dim * state_dim)
    fdac = float(
        fft_cost_constant * state_dim * fft_len * math.log2(max(fft_len, 2))
        + launch_overhead_units
    )
    ratio = fdac / dense
    speedup = dense / max(fdac, 1e-12)
    return dense, fdac, ratio, speedup


def offdiag_fraction(transition: np.ndarray) -> float:
    offdiag = transition.copy()
    np.fill_diagonal(offdiag, 0.0)
    return float(np.linalg.norm(offdiag) / max(np.linalg.norm(transition), 1e-12))


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    segment_len = args.small_segment_len if condition == "small_segment_fail" else args.segment_len
    gradients = make_gradients(rng, segment_len, args.state_dim)
    diagonal = rng.uniform(0.35, args.transition_radius, size=args.state_dim)
    diagonal *= rng.choice([-1.0, 1.0], size=args.state_dim)
    transition_radius = float(np.max(np.abs(diagonal)))
    transition_variation = 0.0
    offdiag = 0.0
    linear_padding = condition != "circular_alias_fail"
    fft_constant = (
        args.expensive_fft_cost_constant
        if condition == "fft_cost_fail"
        else args.fft_cost_constant
    )

    if condition == "time_varying_fail":
        time = np.linspace(0.0, 2.0 * np.pi, segment_len)
        modulation = 0.24 * np.sin(time)[:, None]
        diagonals = np.clip(diagonal[None, :] + modulation, -0.96, 0.96)
        exact = exact_time_varying_adjoint(gradients, diagonals)
        candidate, fft_len = fdac_diagonal_adjoint(gradients, diagonal, linear_padding=True)
        transition_variation = float(np.max(np.abs(diagonals - diagonal[None, :])))
        transition_radius = float(np.max(np.abs(diagonals)))
    elif condition == "non_diagonal_fail":
        base = np.diag(diagonal)
        mix = rng.normal(scale=0.035, size=(args.state_dim, args.state_dim))
        np.fill_diagonal(mix, 0.0)
        transition = base + mix
        spectral = max(float(np.max(np.abs(np.linalg.eigvals(transition)))), 1e-12)
        transition *= min(0.92 / spectral, 1.0)
        exact = exact_dense_adjoint(gradients, transition)
        candidate, fft_len = fdac_diagonal_adjoint(
            gradients,
            np.diag(transition),
            linear_padding=True,
        )
        transition_radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
        offdiag = offdiag_fraction(transition)
    elif condition == "circular_alias_fail":
        alias_diagonal = rng.uniform(0.985, 0.995, size=args.state_dim)
        alias_diagonal *= rng.choice([-1.0, 1.0], size=args.state_dim)
        exact = exact_diagonal_adjoint(gradients, alias_diagonal)
        candidate, fft_len = fdac_diagonal_adjoint(
            gradients,
            alias_diagonal,
            linear_padding=False,
        )
        transition_radius = float(np.max(np.abs(alias_diagonal)))
    else:
        exact = exact_diagonal_adjoint(gradients, diagonal)
        candidate, fft_len = fdac_diagonal_adjoint(gradients, diagonal, linear_padding=True)

    delta = candidate - exact
    exact_norm = float(np.linalg.norm(exact))
    candidate_norm = float(np.linalg.norm(candidate))
    rel_error = float(np.linalg.norm(delta) / max(exact_norm, 1e-12))
    max_abs_error = float(np.max(np.abs(delta)))
    cosine = float(
        np.vdot(candidate.ravel(), exact.ravel()).real
        / max(candidate_norm * exact_norm, 1e-12)
    )
    dense_units, fdac_units, price, speedup = priced_work(
        segment_len,
        args.state_dim,
        fft_len,
        fft_constant,
        args.fft_launch_overhead_units,
    )

    failed: list[str] = []
    if not np.isfinite(
        [
            rel_error,
            max_abs_error,
            cosine,
            dense_units,
            fdac_units,
            price,
            speedup,
            transition_radius,
            offdiag,
            transition_variation,
        ]
    ).all():
        failed.append("non_finite_metric")
    if rel_error > args.max_relative_adjoint_error:
        failed.append("adjoint_error")
    if cosine < args.min_descent_cosine:
        failed.append("descent_cosine")
    if price > args.max_priced_work_ratio:
        failed.append("priced_work")

    expected_pass = condition == "positive"
    passed = not failed
    return TrialResult(
        condition=condition,
        seed=seed,
        expected_pass=expected_pass,
        passed=passed,
        identifying_correct=passed == expected_pass,
        fail_reason=";".join(failed),
        segment_len=segment_len,
        state_dim=args.state_dim,
        fft_len=fft_len,
        transition_radius=transition_radius,
        offdiag_fraction=offdiag,
        transition_variation_linf=transition_variation,
        relative_adjoint_error=rel_error,
        max_abs_adjoint_error=max_abs_error,
        descent_cosine=cosine,
        dense_work_units=dense_units,
        fdac_work_units=fdac_units,
        priced_work_ratio=price,
        projected_speedup=speedup,
    )


NUMERIC_FIELDS = [
    "segment_len",
    "state_dim",
    "fft_len",
    "transition_radius",
    "offdiag_fraction",
    "transition_variation_linf",
    "relative_adjoint_error",
    "max_abs_adjoint_error",
    "descent_cosine",
    "dense_work_units",
    "fdac_work_units",
    "priced_work_ratio",
    "projected_speedup",
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
        "positive_condition_exact_and_priced": (
            aggregates["positive"]["relative_adjoint_error_max"] <= 1e-10
            and aggregates["positive"]["descent_cosine_min"] >= 0.999999999
            and aggregates["positive"]["priced_work_ratio_max"] <= 0.90
        ),
        "small_segment_control_all_fail_price": all(
            (not result.passed) and "priced_work" in result.fail_reason
            for result in by_condition["small_segment_fail"]
        ),
        "time_varying_control_all_fail_error": all(
            (not result.passed)
            and result.transition_variation_linf > 0.0
            and (
                "adjoint_error" in result.fail_reason
                or "descent_cosine" in result.fail_reason
            )
            for result in by_condition["time_varying_fail"]
        ),
        "non_diagonal_control_all_fail_error": all(
            (not result.passed)
            and result.offdiag_fraction > 0.0
            and (
                "adjoint_error" in result.fail_reason
                or "descent_cosine" in result.fail_reason
            )
            for result in by_condition["non_diagonal_fail"]
        ),
        "circular_alias_control_all_fail_error": all(
            (not result.passed)
            and (
                "adjoint_error" in result.fail_reason
                or "descent_cosine" in result.fail_reason
            )
            for result in by_condition["circular_alias_fail"]
        ),
        "fft_cost_control_all_fail_price": all(
            (not result.passed) and "priced_work" in result.fail_reason
            for result in by_condition["fft_cost_fail"]
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
    decision = "PASS_R502_S22_FDAC_GATE" if all(checks.values()) else "FAIL_R502_S22_FDAC_GATE"
    payload = {
        "round": "R502",
        "canonical_id": "R502-S22",
        "aspect": "Speed",
        "finding": "Fourier-Domain Adjoint Convolution (FDAC)",
        "decision": decision,
        "disposition": "synthetic_positive_control_identified_not_training_speed_claim",
        "args": vars(args) | {"output": str(args.output) if args.output else None},
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU Fourier-domain adjoint convolution gate only; not "
            "heldout CE progress, not real corpus token efficiency, not GPU "
            "training speed, and not production v12 integration."
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_S22_FDAC_GATE] "
        f"active=1 decision={decision} "
        f"disposition={payload['disposition']} "
        f"positive_segment_len_mean={positive['segment_len_mean']:.0f} "
        f"positive_relative_adjoint_error_max={positive['relative_adjoint_error_max']:.12e} "
        f"positive_descent_cosine_min={positive['descent_cosine_min']:.12f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.12f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.12f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
