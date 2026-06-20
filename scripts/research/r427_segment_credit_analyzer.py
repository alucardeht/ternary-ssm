#!/usr/bin/env python3
"""Analyze the R427 segment-credit certificate smoke.

This analyzer checks that the real GPU training path exposes finite credit
certificate objects for R427 C/D findings:

- boundary cotangent tube credit;
- variance-priced credit allocation;
- original-CE credit/debt surface;
- checkpoint receipt challenge.

It reports activation and pricing quality. It does not claim a production
training speedup, because this block is an observer/certificate stack.
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


def latest_line(text: str, marker: str, *, result_line: bool = False) -> dict[str, Any] | None:
    lines = [line for line in text.splitlines() if marker in line]
    if result_line:
        lines = [line for line in lines if " step=" in line]
    if not lines:
        return None
    return parse_kv_line(lines[-1])


def latest_receipt_line(text: str) -> dict[str, Any] | None:
    # R364 crashmap output can interleave inside the long receipt marker on
    # Windows console streams. Anchor on the stable suffix and the result keys.
    candidates = []
    for line in text.splitlines():
        if (
            "RECEIPT_CHALLENGE_GATE] step=" in line
            and "caught_harm_fraction=" in line
            and "priced_ratio=" in line
        ):
            candidates.append(line)
    if not candidates:
        return None
    return parse_kv_line(candidates[-1])


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


def finite_numeric_values(row: dict[str, Any] | None, keys: list[str]) -> bool:
    if row is None:
        return False
    for key in keys:
        value = row.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return False
    return True


def any_pass(row: dict[str, Any] | None, keys: list[str]) -> bool:
    if row is None:
        return False
    return any(row.get(key) == 1 for key in keys)


def main() -> int:
    args = parse_args()
    text = read_text(args.log)
    train_rows = parse_train_rows(text)

    boundary = latest_line(text, "[BOUNDARY_COTANGENT_CREDIT_GATE]", result_line=True)
    variance = latest_line(text, "[VARIANCE_PRICED_CREDIT_GATE]", result_line=True)
    original = latest_line(text, "[ORIGINAL_CE_CREDIT_DEBT_OBSERVER]", result_line=True)
    receipt = latest_receipt_line(text)

    required_passes = {
        "log_exists": args.log.exists() and args.log.stat().st_size > 0,
        "train_completed": "[TRAIN] Complete" in text,
        "no_bad_failure": not any(token in text.lower() for token in ("fatal", "[error]", "nan", "inf")),
        "boundary_requested": "[BOUNDARY_COTANGENT_CREDIT_GATE] requested=1" in text,
        "boundary_active": "[BOUNDARY_COTANGENT_CREDIT_GATE] active=1" in text,
        "boundary_finite": finite_numeric_values(
            boundary,
            ["residual_fraction", "priced_ratio", "descent_per_paid_work"],
        ),
        "variance_requested": "[VARIANCE_PRICED_CREDIT_GATE] requested=1" in text,
        "variance_active": "[VARIANCE_PRICED_CREDIT_GATE] active=1" in text,
        "variance_finite": finite_numeric_values(
            variance,
            ["sampled_fraction", "variance_proxy", "descent_per_paid_work"],
        ),
        "original_ce_requested": "[ORIGINAL_CE_CREDIT_DEBT_OBSERVER] requested=1" in text,
        "original_ce_active": "[ORIGINAL_CE_CREDIT_DEBT_OBSERVER] active=1" in text,
        "original_ce_finite": finite_numeric_values(
            original,
            ["local_progress_ratio", "local_residual_fraction", "surface_progress_ratio"],
        ),
        "receipt_requested": "[R423_NATIVE_GROWTH_RECEIPT_CHALLENGE_GATE] requested=1" in text,
        "receipt_finite": finite_numeric_values(
            receipt,
            ["caught_harm_fraction", "priced_ratio", "mean_full_risk", "mean_receipt_risk"],
        ),
    }

    material_signals = {
        "boundary_pass": boundary.get("pass") == 1 if boundary else False,
        "variance_pass": variance.get("pass") == 1 if variance else False,
        "original_ce_any_pass": any_pass(original, ["local_pass", "surface_pass"]),
        "receipt_pass": receipt.get("pass") == 1 if receipt else False,
    }
    positive_signal_count = sum(1 for passed in material_signals.values() if passed)

    decision = (
        "PASS_R427_SEGMENT_CREDIT_CERTIFICATE_ACTIVATION"
        if all(required_passes.values())
        else "FAIL_R427_SEGMENT_CREDIT_CERTIFICATE_ACTIVATION"
    )
    disposition = (
        "observer_activation_with_material_positive_signal"
        if decision.startswith("PASS_") and positive_signal_count > 0
        else "observer_activation_only"
        if decision.startswith("PASS_")
        else "activation_failed"
    )

    result: dict[str, Any] = {
        "decision": decision,
        "disposition": disposition,
        "claim_scope": (
            "Real GPU observer/certificate activation for segment credit. "
            "Finite metrics prove the path is wired; pass bits identify whether "
            "the current tiny smoke exhibits material priced credit. This is not "
            "a production speedup claim."
        ),
        "manifest": str(args.manifest),
        "log": str(args.log),
        "required_passes": required_passes,
        "material_signals": material_signals,
        "positive_signal_count": positive_signal_count,
        "metrics": {
            "boundary_cotangent": boundary,
            "variance_priced": variance,
            "original_ce_credit_debt": original,
            "receipt_challenge": receipt,
            "first_train_row": train_rows[0] if train_rows else None,
            "last_train_row": train_rows[-1] if train_rows else None,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[R427_SEGMENT_CREDIT_ANALYZER] "
        f"decision={decision} disposition={disposition} output={args.output}"
    )
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
