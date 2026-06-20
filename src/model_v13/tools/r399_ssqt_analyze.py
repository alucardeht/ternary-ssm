#!/usr/bin/env python3
"""R399 SSQT diagnostic gate analyzer."""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


EPS = 1.0e-12
DEFAULT_TRACE = os.path.join(".work", "jobs", "r399-ssqt-gate-01", "artifacts", "ssqt_trace.jsonl")
DEFAULT_METRICS = os.path.join(".work", "jobs", "r399-ssqt-gate-01", "artifacts", "ssqt_metrics.json")
PASS_REDUNDANCY = 2.2
PASS_EXPLAINED_VARIANCE = 0.50
FRONTIER_TOLERANCE = 0.02


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def parse_trace(path: str) -> Tuple[Optional[dict], List[dict], List[str]]:
    meta = None
    records: List[dict] = []
    malformed: List[str] = []

    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    malformed.append(f"json_decode_line_{line_no}:{exc.msg}")
                    continue
                kind = payload.get("kind")
                if kind == "meta":
                    meta = payload
                elif kind == "record":
                    records.append(payload)
                else:
                    malformed.append(f"unknown_kind_line_{line_no}")
    except FileNotFoundError:
        return None, [], ["trace_not_found"]

    return meta, records, malformed


def compute_metrics(meta: Optional[dict], records: List[dict], malformed: List[str]) -> dict:
    invalid_reasons = list(malformed)
    grouped: Dict[str, List[dict]] = defaultdict(list)
    split_pairs = 0
    split_disagreements = 0
    comparable_frontier = 0
    frontier_disagreements = 0
    raw_gradients: List[float] = []
    active_frontier_count = 0

    if not meta or not meta.get("activation_line_emitted"):
        invalid_reasons.append("no_activation_line")

    for record in records:
        try:
            quotient_key = str(record["quotient_key"])
            raw_gradient = float(record["raw_gradient"])
            quotient_gradient = float(record["quotient_gradient"])
            raw_frontier_pressure = float(record["raw_frontier_pressure"])
            quotient_frontier_pressure = float(record["quotient_frontier_pressure"])
            split_key = int(record["split_key"])
            active_frontier = int(record["active_frontier"])
        except (KeyError, TypeError, ValueError):
            invalid_reasons.append("malformed_trace_output")
            continue

        values = (
            raw_gradient,
            quotient_gradient,
            raw_frontier_pressure,
            quotient_frontier_pressure,
        )
        if not all(math.isfinite(value) for value in values):
            invalid_reasons.append("nonfinite_trace_values")
            continue

        grouped[quotient_key].append(
            {
                "raw_gradient": raw_gradient,
                "quotient_gradient": quotient_gradient,
                "raw_frontier_pressure": raw_frontier_pressure,
                "quotient_frontier_pressure": quotient_frontier_pressure,
                "split_key": split_key,
                "active_frontier": active_frontier,
            }
        )
        raw_gradients.append(raw_gradient)
        if active_frontier:
            active_frontier_count += 1
            comparable_frontier += 1
            if sign(raw_frontier_pressure) != sign(quotient_frontier_pressure):
                frontier_disagreements += 1

    n_records = sum(len(items) for items in grouped.values())
    n_groups = len(grouped)

    if not records:
        invalid_reasons.append("no_records")
    if n_records == 0 or n_groups == 0:
        invalid_reasons.append("no_data")
    if active_frontier_count == 0:
        invalid_reasons.append("no_active_frontier_cells")
    if raw_gradients and max(abs(value) for value in raw_gradients) <= EPS:
        invalid_reasons.append("all_zero_gradient_sketches")

    group_sizes = [len(items) for items in grouped.values()]
    duplicate_group_exists = any(size > 1 for size in group_sizes)
    if grouped and not duplicate_group_exists:
        invalid_reasons.append("no_duplicate_quotient_groups")

    redundancy = None
    if n_records > 0 and n_groups > 0:
        redundancy = n_records / float(n_groups)

    total_mean = None
    total_variance = None
    explained_variance = None
    if raw_gradients:
        total_mean = sum(raw_gradients) / len(raw_gradients)
        total_variance = sum((value - total_mean) ** 2 for value in raw_gradients) / len(raw_gradients)
        if total_variance <= EPS:
            invalid_reasons.append("zero_gradient_variance_denominator")
        else:
            explained_numer = 0.0
            for items in grouped.values():
                mean = sum(item["raw_gradient"] for item in items) / len(items)
                explained_numer += len(items) * (mean - total_mean) ** 2
            explained_variance = explained_numer / (len(raw_gradients) * total_variance)

    for items in grouped.values():
        split_values: Dict[int, List[float]] = defaultdict(list)
        for item in items:
            split_values[item["split_key"]].append(item["raw_frontier_pressure"])
        if len(split_values) < 2:
            continue
        left_key, right_key = sorted(split_values)[:2]
        left_mean = sum(split_values[left_key]) / len(split_values[left_key])
        right_mean = sum(split_values[right_key]) / len(split_values[right_key])
        split_pairs += 1
        if sign(left_mean) != sign(right_mean):
            split_disagreements += 1

    if split_pairs == 0 and n_records > 0:
        invalid_reasons.append("no_split_noise_pairs")
        baseline_noise = None
    else:
        baseline_noise = (split_disagreements / float(split_pairs)) if split_pairs else None

    if comparable_frontier == 0 and n_records > 0:
        invalid_reasons.append("no_comparable_frontier_records")
        frontier_disagreement = None
    else:
        frontier_disagreement = (
            frontier_disagreements / float(comparable_frontier) if comparable_frontier else None
        )

    invalid_reasons = sorted(set(invalid_reasons))
    threshold = (baseline_noise + FRONTIER_TOLERANCE) if baseline_noise is not None else None

    status = "INVALID"
    if not invalid_reasons:
        if (
            redundancy is not None
            and explained_variance is not None
            and frontier_disagreement is not None
            and baseline_noise is not None
            and redundancy >= PASS_REDUNDANCY
            and explained_variance >= PASS_EXPLAINED_VARIANCE
            and frontier_disagreement <= threshold
        ):
            status = "PASS"
        else:
            status = "FAIL"

    return {
        "schema": "r399.ssqt.metrics.v1",
        "status": status,
        "invalid_reasons": invalid_reasons,
        "N": n_records,
        "M": n_groups,
        "redundancy": redundancy,
        "explained_variance": explained_variance,
        "frontier_disagreement": frontier_disagreement,
        "baseline_noise": baseline_noise,
        "frontier_threshold": threshold,
        "synthetic_fixture": bool(meta.get("synthetic_fixture")) if meta else False,
        "trace_source": meta.get("trace_source") if meta else None,
        "no_training_integration": meta.get("no_training_integration") if meta else None,
        "blocker": meta.get("blocker") if meta else None,
        "scenario": meta.get("scenario") if meta else None,
        "steps": meta.get("steps") if meta else None,
        "batch_size": meta.get("batch_size") if meta else None,
        "seq_len": meta.get("seq_len") if meta else None,
        "trace_sample": meta.get("trace_sample") if meta else None,
        "frontier_cells": meta.get("frontier_cells") if meta else None,
    }


def fmt_metric(value: Optional[float]) -> str:
    if value is None:
        return "nan"
    return f"{value:.6f}"


def emit_result_line(metrics: dict) -> None:
    print(
        "[R399 SSQT GATE RESULT] "
        f"status={metrics['status']} "
        f"N={metrics['N']} "
        f"M={metrics['M']} "
        f"redundancy={fmt_metric(metrics['redundancy'])} "
        f"explained_variance={fmt_metric(metrics['explained_variance'])} "
        f"frontier_disagreement={fmt_metric(metrics['frontier_disagreement'])} "
        f"baseline_noise={fmt_metric(metrics['baseline_noise'])}"
    )


def fixture_records(name: str) -> Tuple[dict, List[dict]]:
    meta = {
        "kind": "meta",
        "schema": "r399.ssqt.trace.v1",
        "activation_line_emitted": True,
        "synthetic_fixture": True,
        "blocker": "self_test_fixture",
        "scenario": name,
        "steps": 1,
        "batch_size": 2,
        "seq_len": 4,
        "trace_sample": 4,
        "frontier_cells": 1,
    }
    records: List[dict] = []
    if name == "invalid":
        return meta, records
    if name == "default_smoke":
        group_count = 10
        total_records = 32
        for index in range(total_records):
            group_index = index % group_count
            duplicate_ordinal = index // group_count
            split_key = duplicate_ordinal & 1
            base = sign_preserving_base(group_index)
            raw_noise = small_noise(index, split_key)
            records.append(
                {
                    "kind": "record",
                    "schema": "r399.ssqt.trace.v1",
                    "exposure_id": index,
                    "quotient_key": f"z-{group_index:03d}",
                    "raw_gradient": base + raw_noise,
                    "quotient_gradient": base + 0.12 * raw_noise,
                    "raw_frontier_pressure": base + 0.20 * raw_noise,
                    "quotient_frontier_pressure": base,
                    "split_key": split_key,
                    "active_frontier": 1,
                }
            )
        return meta, records

    template: Iterable[Tuple[str, float, float, float, float, int]] = []
    if name == "pass":
        template = [
            ("z0", 0.82, 0.80, 0.79, 0.80, 0),
            ("z0", 0.79, 0.80, 0.77, 0.80, 1),
            ("z0", 0.83, 0.80, 0.80, 0.80, 0),
            ("z1", -0.71, -0.70, -0.69, -0.70, 1),
            ("z1", -0.69, -0.70, -0.68, -0.70, 0),
            ("z1", -0.74, -0.70, -0.73, -0.70, 1),
        ]
    elif name == "fail":
        template = [
            ("z0", 0.85, 0.80, 0.82, -0.80, 0),
            ("z0", -0.20, 0.80, -0.19, -0.80, 1),
            ("z1", 0.81, -0.70, 0.82, 0.70, 0),
            ("z1", -0.77, -0.70, -0.76, 0.70, 1),
        ]
    else:
        raise ValueError(f"unknown self-test fixture: {name}")

    for index, (quotient_key, raw_gradient, quotient_gradient, raw_frontier_pressure, quotient_frontier_pressure, split_key) in enumerate(template):
        records.append(
            {
                "kind": "record",
                "schema": "r399.ssqt.trace.v1",
                "exposure_id": index,
                "quotient_key": quotient_key,
                "raw_gradient": raw_gradient,
                "quotient_gradient": quotient_gradient,
                "raw_frontier_pressure": raw_frontier_pressure,
                "quotient_frontier_pressure": quotient_frontier_pressure,
                "split_key": split_key,
                "active_frontier": 1,
            }
        )
    return meta, records


def sign_preserving_base(group_index: int) -> float:
    base = ((group_index % 7) - 3) * 0.45 + (((group_index // 7) % 5) - 2) * 0.08
    if abs(base) < 0.12:
        base += 0.28 if (group_index % 2 == 0) else -0.28
    return base


def small_noise(index: int, split: int) -> float:
    return 0.035 * math.sin(0.37 * (index + 1)) + 0.018 * math.cos(0.19 * (index + 3 * split + 1))


def run_self_test() -> int:
    expectations = {"pass": "PASS", "fail": "FAIL", "invalid": "INVALID"}
    for name, expected in expectations.items():
        meta, records = fixture_records(name)
        metrics = compute_metrics(meta, records, [])
        if metrics["status"] != expected:
            print(
                f"self-test failed fixture={name} expected={expected} got={metrics['status']}",
                file=sys.stderr,
            )
            return 1
    meta, records = fixture_records("default_smoke")
    metrics = compute_metrics(meta, records, [])
    if metrics["status"] == "INVALID":
        print("self-test failed fixture=default_smoke expected=PASS|FAIL got=INVALID", file=sys.stderr)
        return 1
    if metrics["baseline_noise"] is None or not math.isfinite(metrics["baseline_noise"]):
        print("self-test failed fixture=default_smoke baseline_noise is not finite", file=sys.stderr)
        return 1
    if "no_split_noise_pairs" in metrics["invalid_reasons"]:
        print("self-test failed fixture=default_smoke retained no_split_noise_pairs", file=sys.stderr)
        return 1
    print("[R399 SSQT GATE SELFTEST] status=PASS fixtures=4")
    return 0


def main(argv: List[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        return run_self_test()
    if len(argv) > 3:
        print("usage: r399_ssqt_analyze.py [TRACE_PATH] [METRICS_PATH] | --self-test", file=sys.stderr)
        return 2

    trace_path = argv[1] if len(argv) >= 2 else DEFAULT_TRACE
    metrics_path = argv[2] if len(argv) == 3 else DEFAULT_METRICS

    meta, records, malformed = parse_trace(trace_path)
    metrics = compute_metrics(meta, records, malformed)
    ensure_parent_dir(metrics_path)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    emit_result_line(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
