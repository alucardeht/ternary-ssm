#!/usr/bin/env python3
"""R421 Cotangent/State Freshness adapter gate.

This gate maps the existing tail-cotangent debt JSONs into the broader
freshness/residency claim. It checks whether short-horizon replay preserves CE
progress with small residual debt after adding memory and freshness-check costs.

It is an adapter, not a separate production implementation.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r421_cotangent_freshness_gate.v1"


@dataclass
class FreshnessResult:
    schema: str
    decision: str
    source: str
    actual_progress_ratio: float
    residual_fraction: float
    whole_step_priced_ratio_estimate: float
    freshness_check_overhead: float
    memory_overhead: float
    priced_ratio: float
    net_speedup_estimate: float
    merge_recommendation: str
    activation_log: str
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tail-json", type=Path, required=True)
    parser.add_argument("--freshness-check-overhead", type=float, default=0.02)
    parser.add_argument("--memory-overhead", type=float, default=0.03)
    parser.add_argument("--min-progress-ratio", type=float, default=0.99)
    parser.add_argument("--max-residual-fraction", type=float, default=0.01)
    parser.add_argument("--min-speedup", type=float, default=1.10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    data = json.loads(args.tail_json.read_text(encoding="utf-8"))
    best = data.get("best_result", data.get("best_horizon"))
    if not isinstance(best, dict):
        raise KeyError("tail JSON must contain best_horizon or best_result")

    progress = float(best["actual_progress_ratio"])
    residual = float(best["residual_fraction"])
    base_priced = float(best["whole_step_priced_ratio_estimate"])
    priced = base_priced + args.freshness_check_overhead + args.memory_overhead + residual
    speedup = progress / max(priced, 1e-12)
    passed = (
        progress >= args.min_progress_ratio
        and residual <= args.max_residual_fraction
        and speedup >= args.min_speedup
    )
    decision = "PASS_COTANGENT_FRESHNESS_ADAPTER_GATE" if passed else "FAIL_COTANGENT_FRESHNESS_ADAPTER_GATE"
    activation = (
        "[R421_COTANGENT_FRESHNESS_GATE] "
        f"active=1 decision={decision} no_training_integration=1 no_gpu=1"
    )
    result = FreshnessResult(
        schema=SCHEMA,
        decision=decision,
        source=str(args.tail_json),
        actual_progress_ratio=progress,
        residual_fraction=residual,
        whole_step_priced_ratio_estimate=base_priced,
        freshness_check_overhead=args.freshness_check_overhead,
        memory_overhead=args.memory_overhead,
        priced_ratio=priced,
        net_speedup_estimate=speedup,
        merge_recommendation="merge_with_tail_cotangent_debt_until_real_v12_trace_distinguishes_it",
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
