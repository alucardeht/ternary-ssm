#!/usr/bin/env python3
"""R421 Obligation-Ledger Scheduling gate.

This lightweight gate reuses existing exact v12 CPU state-erasure decision JSONs
from R405 Q2 and maps them into the R421 obligation-ledger question:

Can training be routed by compact unresolved CE obligations/state handles rather
than corpus order, while keeping held-out CE debt bounded?

The answer requires both real state value and a cheap selector. Concentrated
state value alone is preserved as support, not promoted.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean


SCHEMA = "r421_obligation_ledger_gate.v1"


@dataclass
class LedgerAggregate:
    schema: str
    decision: str
    inputs: list[str]
    runs: int
    min_positive_prevalence: float
    mean_positive_prevalence: float
    min_top_quartile_value_share: float
    mean_top_quartile_value_share: float
    max_auc: float
    max_precision_ratio: float
    max_priced_projected_progress: float
    max_mean_interaction_harm: float
    has_state_value_signal: bool
    has_cheap_selector: bool
    clears_priced_progress: bool
    activation_log: str
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--min-positive-prevalence", type=float, default=0.10)
    parser.add_argument("--min-top-quartile-share", type=float, default=0.50)
    parser.add_argument("--min-auc", type=float, default=0.65)
    parser.add_argument("--min-precision-ratio", type=float, default=2.0)
    parser.add_argument("--min-priced-progress", type=float, default=1.15)
    parser.add_argument("--max-interaction-harm", type=float, default=0.30)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    rows = [load(path) for path in args.input]
    if not rows:
        raise ValueError("at least one input decision JSON is required")

    positive_prevalence = [float(row["positive_prevalence"]) for row in rows]
    top_share = [float(row["top_quartile_value_share"]) for row in rows]
    auc = [float(row["auc"]) for row in rows]
    precision = [float(row["precision_topk_ratio"]) for row in rows]
    priced = [float(row["priced_projected_progress"]) for row in rows]
    harm = [float(row["mean_interaction_harm"]) for row in rows]

    has_state_value = (
        min(positive_prevalence) >= args.min_positive_prevalence
        and min(top_share) >= args.min_top_quartile_share
        and max(harm) <= args.max_interaction_harm
    )
    has_selector = max(auc) >= args.min_auc or max(precision) >= args.min_precision_ratio
    clears_priced = max(priced) >= args.min_priced_progress
    if has_state_value and has_selector and clears_priced:
        decision = "PASS_OBLIGATION_LEDGER_GATE"
    elif has_state_value:
        decision = "PARK_STATE_VALUE_SIGNAL_WEAK_SELECTOR"
    else:
        decision = "FAIL_NO_COMPACT_OBLIGATION_SIGNAL"

    activation = (
        "[R421_OBLIGATION_LEDGER_GATE] "
        f"active=1 decision={decision} no_training_integration=1 no_gpu=1"
    )
    result = LedgerAggregate(
        schema=SCHEMA,
        decision=decision,
        inputs=[str(path) for path in args.input],
        runs=len(rows),
        min_positive_prevalence=min(positive_prevalence),
        mean_positive_prevalence=mean(positive_prevalence),
        min_top_quartile_value_share=min(top_share),
        mean_top_quartile_value_share=mean(top_share),
        max_auc=max(auc),
        max_precision_ratio=max(precision),
        max_priced_projected_progress=max(priced),
        max_mean_interaction_harm=max(harm),
        has_state_value_signal=has_state_value,
        has_cheap_selector=has_selector,
        clears_priced_progress=clears_priced,
        activation_log=activation,
        elapsed_seconds=time.perf_counter() - start,
    )
    print(activation)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
