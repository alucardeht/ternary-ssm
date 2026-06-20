#!/usr/bin/env python3
"""R502-S29 Stochastic Trace Jacobian Estimation gate.

This synthetic CPU gate checks whether a small number of Rademacher Hutchinson
probes can estimate Jacobian trace cheaply enough to replace a dense JVP in a
near-isotropic regime. Controls cover high estimator variance, too few probes,
biased probes, expensive probe counts, and anisotropic JVPs where the trace is
accurate but the downstream direction is wrong.
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
    "high_variance_fail",
    "low_sample_fail",
    "biased_probe_fail",
    "expensive_samples_fail",
    "anisotropic_jvp_fail",
]


@dataclass
class TrialResult:
    condition: str
    seed: int
    expected_pass: bool
    passed: bool
    identifying_correct: bool
    fail_reason: str
    matrix_count: int
    state_dim: int
    probe_count: int
    trace_relative_error_mean: float
    trace_relative_error_max: float
    trace_relative_rmse: float
    jvp_relative_error_mean: float
    jvp_relative_error_max: float
    jvp_cosine_min: float
    exact_work_units: float
    hutchinson_work_units: float
    priced_work_ratio: float
    projected_speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=50229)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--matrix-count", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=96)
    parser.add_argument("--probe-count", type=int, default=8)
    parser.add_argument("--expensive-probe-count", type=int, default=96)
    parser.add_argument("--max-trace-relative-error", type=float, default=0.05)
    parser.add_argument("--max-jvp-relative-error", type=float, default=0.08)
    parser.add_argument("--min-jvp-cosine", type=float, default=0.995)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.50)
    parser.add_argument("--probe-overhead-ratio", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def condition_config(args: argparse.Namespace, condition: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "probe_count": args.probe_count,
        "offdiag_scale": 0.03,
        "diag_mode": "near_identity",
        "probe_mode": "rademacher",
    }
    if condition == "high_variance_fail":
        cfg["offdiag_scale"] = 3.00
    elif condition == "low_sample_fail":
        cfg["offdiag_scale"] = 1.50
        cfg["probe_count"] = 1
    elif condition == "biased_probe_fail":
        cfg["probe_mode"] = "biased_bernoulli"
    elif condition == "expensive_samples_fail":
        cfg["probe_count"] = args.expensive_probe_count
    elif condition == "anisotropic_jvp_fail":
        cfg["diag_mode"] = "anisotropic"
        cfg["offdiag_scale"] = 0.02
    return cfg


def make_matrix(
    rng: np.random.Generator,
    state_dim: int,
    offdiag_scale: float,
    diag_mode: str,
) -> np.ndarray:
    if diag_mode == "anisotropic":
        diagonal = np.concatenate(
            [
                np.full(state_dim // 2, 0.20, dtype=np.float64),
                np.full(state_dim - state_dim // 2, 1.80, dtype=np.float64),
            ]
        )
        rng.shuffle(diagonal)
    else:
        diagonal = 1.0 + 0.005 * rng.normal(size=state_dim)

    raw = rng.normal(size=(state_dim, state_dim))
    offdiag = (raw + raw.T) * (0.5 * offdiag_scale / math.sqrt(state_dim))
    np.fill_diagonal(offdiag, 0.0)
    return np.diag(diagonal) + offdiag


def make_probe(
    rng: np.random.Generator,
    probe_count: int,
    state_dim: int,
    probe_mode: str,
) -> np.ndarray:
    if probe_mode == "biased_bernoulli":
        return rng.integers(0, 2, size=(probe_count, state_dim), endpoint=False).astype(np.float64)
    return rng.choice(np.array([-1.0, 1.0]), size=(probe_count, state_dim))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    numerator = float(np.vdot(left, right).real)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 1.0 if np.linalg.norm(left - right) == 0.0 else 0.0
    return numerator / denominator


def run_trial(args: argparse.Namespace, condition: str, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    cfg = condition_config(args, condition)
    probe_count = int(cfg["probe_count"])
    trace_errors: list[float] = []
    trace_sq_errors: list[float] = []
    jvp_errors: list[float] = []
    jvp_cosines: list[float] = []

    for _ in range(args.matrix_count):
        matrix = make_matrix(
            rng,
            args.state_dim,
            float(cfg["offdiag_scale"]),
            str(cfg["diag_mode"]),
        )
        exact_trace = float(np.trace(matrix))
        probes = make_probe(rng, probe_count, args.state_dim, str(cfg["probe_mode"]))
        values = np.einsum("bi,ij,bj->b", probes, matrix, probes)
        estimate = float(np.mean(values))
        trace_error = abs(estimate - exact_trace) / max(abs(exact_trace), 1e-12)
        trace_errors.append(trace_error)
        trace_sq_errors.append(trace_error * trace_error)

        vector = rng.normal(size=args.state_dim)
        exact_jvp = matrix @ vector
        trace_proxy_jvp = (estimate / args.state_dim) * vector
        jvp_errors.append(
            float(np.linalg.norm(trace_proxy_jvp - exact_jvp) / max(np.linalg.norm(exact_jvp), 1e-12))
        )
        jvp_cosines.append(cosine_similarity(trace_proxy_jvp, exact_jvp))

    exact_work = float(args.matrix_count * args.state_dim * args.state_dim)
    hutch_work = float(args.matrix_count * probe_count * args.state_dim)
    price = hutch_work / max(exact_work, 1e-12) + args.probe_overhead_ratio
    speedup = 1.0 / max(price, 1e-12)

    trace_mean = float(np.mean(trace_errors))
    trace_max = float(np.max(trace_errors))
    trace_rmse = float(math.sqrt(np.mean(trace_sq_errors)))
    jvp_mean = float(np.mean(jvp_errors))
    jvp_max = float(np.max(jvp_errors))
    jvp_cosine_min = float(np.min(jvp_cosines))

    failed: list[str] = []
    if not np.isfinite([trace_mean, trace_max, trace_rmse, jvp_mean, jvp_max, jvp_cosine_min, price, speedup]).all():
        failed.append("non_finite_metric")
    if trace_max > args.max_trace_relative_error:
        failed.append("trace_relative_error")
    if jvp_mean > args.max_jvp_relative_error:
        failed.append("jvp_relative_error")
    if jvp_cosine_min < args.min_jvp_cosine:
        failed.append("jvp_cosine")
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
        matrix_count=args.matrix_count,
        state_dim=args.state_dim,
        probe_count=probe_count,
        trace_relative_error_mean=trace_mean,
        trace_relative_error_max=trace_max,
        trace_relative_rmse=trace_rmse,
        jvp_relative_error_mean=jvp_mean,
        jvp_relative_error_max=jvp_max,
        jvp_cosine_min=jvp_cosine_min,
        exact_work_units=exact_work,
        hutchinson_work_units=hutch_work,
        priced_work_ratio=price,
        projected_speedup=speedup,
    )


NUMERIC_FIELDS = [
    "matrix_count",
    "state_dim",
    "probe_count",
    "trace_relative_error_mean",
    "trace_relative_error_max",
    "trace_relative_rmse",
    "jvp_relative_error_mean",
    "jvp_relative_error_max",
    "jvp_cosine_min",
    "exact_work_units",
    "hutchinson_work_units",
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


def required_checks(results: list[TrialResult], aggregates: dict[str, Any], args: argparse.Namespace) -> dict[str, bool]:
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
        "positive_condition_trace_jvp_and_price": (
            aggregates["positive"]["trace_relative_error_max_max"] <= args.max_trace_relative_error
            and aggregates["positive"]["jvp_relative_error_mean_max"] <= args.max_jvp_relative_error
            and aggregates["positive"]["jvp_cosine_min_min"] >= args.min_jvp_cosine
            and aggregates["positive"]["priced_work_ratio_max"] <= args.max_priced_work_ratio
        ),
        "high_variance_control_all_fail_trace": all(
            (not result.passed) and "trace_relative_error" in result.fail_reason
            for result in by_condition["high_variance_fail"]
        ),
        "low_sample_control_all_fail_trace": all(
            (not result.passed) and "trace_relative_error" in result.fail_reason
            for result in by_condition["low_sample_fail"]
        ),
        "biased_probe_control_all_fail_trace": all(
            (not result.passed) and "trace_relative_error" in result.fail_reason
            for result in by_condition["biased_probe_fail"]
        ),
        "expensive_samples_control_all_fail_price": all(
            (not result.passed) and "priced_work" in result.fail_reason
            for result in by_condition["expensive_samples_fail"]
        ),
        "anisotropic_jvp_control_all_fail_jvp_despite_trace": all(
            (not result.passed)
            and "trace_relative_error" not in result.fail_reason
            and (
                "jvp_relative_error" in result.fail_reason
                or "jvp_cosine" in result.fail_reason
            )
            for result in by_condition["anisotropic_jvp_fail"]
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
    checks = required_checks(results, aggregates, args)
    decision = "PASS_R502_S29_STJE_GATE" if all(checks.values()) else "FAIL_R502_S29_STJE_GATE"
    payload = {
        "round": "R502",
        "canonical_id": "R502-S29",
        "aspect": "Speed",
        "finding": "Stochastic Trace Jacobian Estimation",
        "decision": decision,
        "disposition": "synthetic_positive_control_identified_not_training_speed_claim",
        "args": vars(args) | {"output": str(args.output) if args.output else None},
        "required_checks": checks,
        "aggregates": aggregates,
        "trials": [asdict(result) for result in results],
        "claim_boundary": (
            "Synthetic CPU stochastic trace Jacobian estimation gate only; "
            "not heldout CE progress, not real corpus token efficiency, not "
            "GPU training speed, and not production v12 integration."
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    positive = aggregates["positive"]
    print(
        "[R502_S29_STJE_GATE] "
        f"active=1 decision={decision} "
        f"disposition={payload['disposition']} "
        f"positive_trace_relative_error_max_max={positive['trace_relative_error_max_max']:.12f} "
        f"positive_jvp_relative_error_mean_max={positive['jvp_relative_error_mean_max']:.12f} "
        f"positive_jvp_cosine_min_min={positive['jvp_cosine_min_min']:.12f} "
        f"positive_priced_work_ratio_mean={positive['priced_work_ratio_mean']:.12f} "
        f"positive_projected_speedup_mean={positive['projected_speedup_mean']:.12f} "
        f"seed_count={args.seed_count} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
