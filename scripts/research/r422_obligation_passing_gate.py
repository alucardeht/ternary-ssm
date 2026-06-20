#!/usr/bin/env python3
"""R422 CA07 obligation-passing observer gate.

This gate reuses existing exact v12 CPU forward state-erasure traces as an
observer-only source. It asks a narrower question than generic obligation
scheduling:

Can a compact obligation handle select state slices whose originating CE
liability is materially concentrated, while beating local auxiliary-loss and
raw-magnitude controls?

It is not a GPU training integration and must not be used as a measured speedup.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable


SCHEMA = "r422_obligation_passing_gate.v1"


@dataclass
class ScoreMetrics:
    score_name: str
    selected_fraction: float
    selected_count: int
    precision: float
    positive_share: float
    negative_share: float
    signed_share: float
    positive_per_work: float
    signed_per_work: float


@dataclass
class RunResult:
    input_path: str
    contexts: int
    records: int
    positive_prevalence: float
    positive_value_total: float
    meta: dict
    metrics: list[ScoreMetrics]


@dataclass
class GateResult:
    schema: str
    decision: str
    activation_log: str
    inputs: list[str]
    selected_fractions: list[float]
    narrow_fraction: float
    material_fraction: float
    min_narrow_signed_per_work: float
    min_narrow_precision: float
    min_material_positive_share: float
    min_material_signed_per_work: float
    max_aux_material_signed_per_work: float
    max_raw_material_signed_per_work: float
    min_obligation_vs_aux_signed_margin: float
    min_obligation_vs_raw_signed_margin: float
    observer_only: bool
    no_gpu: bool
    no_training_integration: bool
    elapsed_seconds: float
    runs: list[RunResult]


Record = dict[str, float | int | str]
Scorer = Callable[[Record], float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--fraction", type=float, action="append", default=[0.025, 0.05])
    parser.add_argument("--narrow-fraction", type=float, default=0.025)
    parser.add_argument("--material-fraction", type=float, default=0.05)
    parser.add_argument("--min-narrow-signed-per-work", type=float, default=1.15)
    parser.add_argument("--min-narrow-precision", type=float, default=0.65)
    parser.add_argument("--min-material-positive-share", type=float, default=0.10)
    parser.add_argument("--min-material-signed-per-work", type=float, default=1.15)
    parser.add_argument("--min-control-margin", type=float, default=0.10)
    parser.add_argument("--random-trials", type=int, default=64)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_trace(path: Path) -> tuple[dict, list[Record]]:
    meta: dict = {}
    records: list[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.get("kind")
            if kind == "meta":
                meta = row
            elif kind == "record":
                records.append(row)
            elif kind in {"pair", "summary"}:
                continue
            else:
                raise ValueError(f"{path}:{line_no}: unsupported kind={kind!r}")
    if not records:
        raise ValueError(f"{path}: no record rows found")
    return meta, records


def safe_log_norm(value: object) -> float:
    return math.log1p(1_000_000.0 * max(0.0, float(value)))


def obligation_score(row: Record) -> float:
    # Recent write pressure is the candidate obligation carrier. Raw block norm,
    # local CE, and the prior predictor are downweighted so this cannot pass by
    # becoming ordinary magnitude ranking or auxiliary-loss ranking.
    return (
        2.0 * safe_log_norm(row["recent_write_norm"])
        - 0.5 * safe_log_norm(row["block_norm"])
        - float(row["predictor_score"])
        - float(row["local_ce"]) / 100.0
    )


def random_score_factory(seed: int) -> Scorer:
    rng = random.Random(seed)
    cache: dict[tuple[int, int, int, int], float] = {}

    def score(row: Record) -> float:
        key = (
            int(row["context"]),
            int(row["layer"]),
            int(row["group"]),
            int(row["start"]),
        )
        if key not in cache:
            cache[key] = rng.random()
        return cache[key]

    return score


def make_scorers(random_seed: int) -> dict[str, Scorer]:
    return {
        "obligation_write_minus_aux": obligation_score,
        "aux_local_ce": lambda row: float(row["local_ce"]),
        "prior_predictor": lambda row: float(row["predictor_score"]),
        "raw_block_norm": lambda row: float(row["block_norm"]),
        "recent_write_only": lambda row: float(row["recent_write_norm"]),
        "random": random_score_factory(random_seed),
    }


def compute_metrics(
    records: list[Record],
    positive_total: float,
    score_name: str,
    scorer: Scorer,
    fraction: float,
) -> ScoreMetrics:
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"selected fraction must be in (0,1], got {fraction}")
    k = max(1, int(len(records) * fraction))
    selected_fraction = k / len(records)
    selected = sorted(records, key=scorer, reverse=True)[:k]
    positives = [max(0.0, float(row["delta_ce"])) for row in selected]
    negatives = [-min(0.0, float(row["delta_ce"])) for row in selected]
    positive_share = sum(positives) / positive_total if positive_total > 0.0 else 0.0
    negative_share = sum(negatives) / positive_total if positive_total > 0.0 else 0.0
    signed_share = positive_share - negative_share
    precision = sum(1 for row in selected if float(row["delta_ce"]) > 0.0) / k
    return ScoreMetrics(
        score_name=score_name,
        selected_fraction=selected_fraction,
        selected_count=k,
        precision=precision,
        positive_share=positive_share,
        negative_share=negative_share,
        signed_share=signed_share,
        positive_per_work=positive_share / selected_fraction,
        signed_per_work=signed_share / selected_fraction,
    )


def find_metric(run: RunResult, score_name: str, fraction: float) -> ScoreMetrics:
    candidates = [
        metric
        for metric in run.metrics
        if metric.score_name == score_name
        and abs(metric.selected_fraction - fraction) <= max(0.005, 1.0 / max(1, run.records))
    ]
    if not candidates:
        raise KeyError(f"missing metrics for score={score_name} fraction={fraction}")
    return candidates[0]


def min_metric(runs: Iterable[RunResult], score_name: str, fraction: float, field: str) -> float:
    return min(float(getattr(find_metric(run, score_name, fraction), field)) for run in runs)


def max_metric(runs: Iterable[RunResult], score_name: str, fraction: float, field: str) -> float:
    return max(float(getattr(find_metric(run, score_name, fraction), field)) for run in runs)


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    fractions = sorted(set(args.fraction + [args.narrow_fraction, args.material_fraction]))
    runs: list[RunResult] = []
    for input_index, path in enumerate(args.input):
        meta, records = load_trace(path)
        positive_total = sum(max(0.0, float(row["delta_ce"])) for row in records)
        if positive_total <= 0.0:
            raise ValueError(f"{path}: no positive originating CE liability")
        scorers = make_scorers(422070 + input_index)
        metrics: list[ScoreMetrics] = []
        for fraction in fractions:
            for score_name, scorer in scorers.items():
                if score_name == "random":
                    random_metrics = [
                        compute_metrics(
                            records,
                            positive_total,
                            score_name,
                            random_score_factory(422070 + input_index * 1000 + trial),
                            fraction,
                        )
                        for trial in range(args.random_trials)
                    ]
                    metrics.append(
                        ScoreMetrics(
                            score_name=score_name,
                            selected_fraction=random_metrics[0].selected_fraction,
                            selected_count=random_metrics[0].selected_count,
                            precision=statistics.mean(m.precision for m in random_metrics),
                            positive_share=statistics.mean(m.positive_share for m in random_metrics),
                            negative_share=statistics.mean(m.negative_share for m in random_metrics),
                            signed_share=statistics.mean(m.signed_share for m in random_metrics),
                            positive_per_work=statistics.mean(m.positive_per_work for m in random_metrics),
                            signed_per_work=statistics.mean(m.signed_per_work for m in random_metrics),
                        )
                    )
                else:
                    metrics.append(compute_metrics(records, positive_total, score_name, scorer, fraction))
        runs.append(
            RunResult(
                input_path=str(path),
                contexts=int(meta.get("contexts", 0)),
                records=len(records),
                positive_prevalence=sum(1 for row in records if float(row["delta_ce"]) > 0.0) / len(records),
                positive_value_total=positive_total,
                meta=meta,
                metrics=metrics,
            )
        )

    score = "obligation_write_minus_aux"
    narrow_fraction = args.narrow_fraction
    material_fraction = args.material_fraction
    min_narrow_signed = min_metric(runs, score, narrow_fraction, "signed_per_work")
    min_narrow_precision = min_metric(runs, score, narrow_fraction, "precision")
    min_material_positive = min_metric(runs, score, material_fraction, "positive_share")
    min_material_signed = min_metric(runs, score, material_fraction, "signed_per_work")
    max_aux_material_signed = max_metric(runs, "aux_local_ce", material_fraction, "signed_per_work")
    max_raw_material_signed = max_metric(runs, "raw_block_norm", material_fraction, "signed_per_work")
    min_aux_margin = min(
        find_metric(run, score, material_fraction).signed_per_work
        - find_metric(run, "aux_local_ce", material_fraction).signed_per_work
        for run in runs
    )
    min_raw_margin = min(
        find_metric(run, score, material_fraction).signed_per_work
        - find_metric(run, "raw_block_norm", material_fraction).signed_per_work
        for run in runs
    )

    narrow_pass = (
        min_narrow_signed >= args.min_narrow_signed_per_work
        and min_narrow_precision >= args.min_narrow_precision
    )
    material_positive_pass = (
        min_material_positive >= args.min_material_positive_share
        and min_aux_margin >= args.min_control_margin
    )
    material_signed_pass = (
        min_material_signed >= args.min_material_signed_per_work
        and min_raw_margin >= args.min_control_margin
    )
    if narrow_pass and material_positive_pass and material_signed_pass:
        decision = "PASS_OBLIGATION_PASSING_OBSERVER_GATE"
    elif narrow_pass and material_positive_pass:
        decision = "PARTIAL_POSITIVE_OBLIGATION_HANDLE_FALSE_HARM_PRICING_OPEN"
    elif narrow_pass:
        decision = "PARK_NARROW_OBLIGATION_DENSITY_SIGNAL"
    else:
        decision = "FAIL_NO_ORIGINATING_CE_OBLIGATION_SIGNAL"

    activation = (
        "[R422_OBLIGATION_PASSING_GATE] "
        f"active=1 decision={decision} observer_only=1 no_gpu=1 no_training_integration=1"
    )
    result = GateResult(
        schema=SCHEMA,
        decision=decision,
        activation_log=activation,
        inputs=[str(path) for path in args.input],
        selected_fractions=fractions,
        narrow_fraction=narrow_fraction,
        material_fraction=material_fraction,
        min_narrow_signed_per_work=min_narrow_signed,
        min_narrow_precision=min_narrow_precision,
        min_material_positive_share=min_material_positive,
        min_material_signed_per_work=min_material_signed,
        max_aux_material_signed_per_work=max_aux_material_signed,
        max_raw_material_signed_per_work=max_raw_material_signed,
        min_obligation_vs_aux_signed_margin=min_aux_margin,
        min_obligation_vs_raw_signed_margin=min_raw_margin,
        observer_only=True,
        no_gpu=True,
        no_training_integration=True,
        elapsed_seconds=time.perf_counter() - start,
        runs=runs,
    )
    print(activation)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
