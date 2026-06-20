#!/usr/bin/env python3
"""Analyze the R436 amortized CE martingale verifier gate.

The gate consumes a real v12 training log with the existing
VARIANCE_PRICED_CREDIT_GATE active. It does not change training behavior. It
tests whether the variance-credit estimator can be re-priced under the R436
martingale verifier contract instead of dense per-step score/verify costs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


KV_RE = re.compile(r"([A-Za-z0-9_/-]+)=([^\s]+)")
TRAIN_RE = re.compile(
    r"\[TRAIN\] step=(?P<step>\d+) loss=(?P<loss>[0-9.eE+-]+).*?"
    r"tok/s=(?P<toks>[0-9.eE+-]+) vram=(?P<vram>[0-9.eE+-]+)MB"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overhead-ceiling", type=float, default=0.58)
    parser.add_argument("--absolute-overhead-ceiling", type=float, default=0.631168)
    parser.add_argument("--max-debt-ratio", type=float, default=0.10)
    parser.add_argument("--max-priced-ratio", type=float, default=0.90)
    parser.add_argument("--min-progress-proxy", type=float, default=0.50)
    parser.add_argument("--max-dense-score-verify", type=float, default=0.25)
    return parser.parse_args()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def coerce_value(raw_value: str) -> Any:
    try:
        if any(ch in raw_value for ch in ".eE"):
            value = float(raw_value)
            return value if math.isfinite(value) else raw_value
        return int(raw_value)
    except ValueError:
        return raw_value


def parse_kv_line(line: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {"line": line}
    for key, raw_value in KV_RE.findall(line):
        parsed[key] = coerce_value(raw_value)
    return parsed


def latest_variance_line(text: str) -> dict[str, Any] | None:
    lines = [
        line
        for line in text.splitlines()
        if "[VARIANCE_PRICED_CREDIT_GATE]" in line
        and " step=" in line
        and "descent_per_paid_work=" in line
    ]
    if not lines:
        return None
    return parse_kv_line(lines[-1])


def parse_train_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in TRAIN_RE.finditer(text):
        rows.append(
            {
                "step": int(match.group("step")),
                "loss": float(match.group("loss")),
                "tok_s": float(match.group("toks")),
                "vram_mb": float(match.group("vram")),
            }
        )
    return rows


def finite_number(row: dict[str, Any] | None, key: str) -> bool:
    if row is None:
        return False
    value = row.get(key)
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def metric(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return default
    return float(value)


def main() -> int:
    args = parse_args()
    text = read_text(args.log)
    train_rows = parse_train_rows(text)
    variance = latest_variance_line(text)

    required_keys = [
        "sampled_fraction",
        "alignment",
        "variance_proxy",
        "score_fraction",
        "verify_fraction",
        "overhead",
        "descent_per_paid_work",
    ]
    variance_finite = variance is not None and all(
        finite_number(variance, key) for key in required_keys
    )

    if variance is None:
        variance = {}

    sampled_fraction = metric(variance, "sampled_fraction")
    score_fraction = metric(variance, "score_fraction")
    verify_fraction = metric(variance, "verify_fraction")
    overhead = metric(variance, "overhead")
    alignment = metric(variance, "alignment")
    variance_proxy = metric(variance, "variance_proxy")

    # R436 uses original CE as authority. This proxy treats alignment as the
    # accepted progress carrier and charges both estimator variance and
    # misalignment as residual CE debt.
    certified_progress_proxy = max(0.0, alignment - variance_proxy)
    residual_debt = max(0.0, 1.0 - alignment) + max(0.0, variance_proxy)
    debt_ratio = (
        residual_debt / certified_progress_proxy
        if certified_progress_proxy > 0.0
        else math.inf
    )
    amortized_overhead = score_fraction + verify_fraction + overhead
    priced_ratio = sampled_fraction + amortized_overhead + residual_debt
    projected_speedup_if_promoted = (
        1.0 / priced_ratio if priced_ratio > 0.0 else math.inf
    )

    required_passes = {
        "log_exists": args.log.exists() and args.log.stat().st_size > 0,
        "train_completed": "[TRAIN] Complete" in text,
        "no_bad_failure": not any(
            token in text.lower() for token in ("fatal", "[error]", "nan", "inf")
        ),
        "variance_requested": "[VARIANCE_PRICED_CREDIT_GATE] requested=1" in text,
        "variance_active": "[VARIANCE_PRICED_CREDIT_GATE] active=1" in text,
        "variance_finite": variance_finite,
        "variance_identifying": variance.get("identifying") == 1,
        "dense_verify_replaced": (score_fraction + verify_fraction)
        <= args.max_dense_score_verify,
        "overhead_under_validated_ceiling": amortized_overhead <= args.overhead_ceiling,
        "overhead_under_absolute_ceiling": amortized_overhead
        <= args.absolute_overhead_ceiling,
        "residual_debt_under_budget": debt_ratio <= args.max_debt_ratio,
        "progress_proxy_material": certified_progress_proxy >= args.min_progress_proxy,
        "priced_ratio_under_budget": priced_ratio <= args.max_priced_ratio,
    }

    decision = (
        "PASS_R436_AMORTIZED_CE_MARTINGALE_VERIFIER_GATE"
        if all(required_passes.values())
        else "FAIL_R436_AMORTIZED_CE_MARTINGALE_VERIFIER_GATE"
    )
    disposition = (
        "martingale_verifier_pricing_gate_passed_not_speed_claim"
        if decision.startswith("PASS_")
        else "martingale_verifier_pricing_gate_failed_or_non_identifying"
    )

    result: dict[str, Any] = {
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "Real GPU observer gate over variance-priced credit. This validates "
            "whether martingale-style amortized verification can price the "
            "existing estimator signal. It is not a production speedup claim."
        ),
        "manifest": str(args.manifest),
        "log": str(args.log),
        "required_passes": required_passes,
        "thresholds": {
            "overhead_ceiling": args.overhead_ceiling,
            "absolute_overhead_ceiling": args.absolute_overhead_ceiling,
            "max_debt_ratio": args.max_debt_ratio,
            "max_priced_ratio": args.max_priced_ratio,
            "min_progress_proxy": args.min_progress_proxy,
            "max_dense_score_verify": args.max_dense_score_verify,
        },
        "martingale_metrics": {
            "sampled_fraction": sampled_fraction,
            "score_fraction": score_fraction,
            "verify_fraction": verify_fraction,
            "overhead": overhead,
            "amortized_overhead": amortized_overhead,
            "alignment": alignment,
            "variance_proxy": variance_proxy,
            "certified_progress_proxy": certified_progress_proxy,
            "residual_debt": residual_debt,
            "debt_ratio": debt_ratio,
            "priced_ratio": priced_ratio,
            "projected_speedup_if_promoted": projected_speedup_if_promoted,
        },
        "metrics": {
            "variance_priced": variance,
            "first_train_row": train_rows[0] if train_rows else None,
            "last_train_row": train_rows[-1] if train_rows else None,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[R436_MARTINGALE_VERIFIER_GATE] "
        f"decision={decision} disposition={disposition} "
        f"priced_ratio={priced_ratio:.6f} debt_ratio={debt_ratio:.6f} "
        f"projected_speedup_if_promoted={projected_speedup_if_promoted:.6f} "
        f"output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
