#!/usr/bin/env python3
"""Analyze the R427 no-reset growth ladder smoke logs.

The analyzer is intentionally conservative. It does not claim a production
speedup. It checks whether the implemented training preset actually exercised a
small-to-large checkpoint lift and then continued training with the live R423
growth mechanisms enabled.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


TRAIN_RE = re.compile(
    r"\[TRAIN\] step=(?P<step>\d+) loss=(?P<loss>[0-9.eE+-]+).*?"
    r"tok/s=(?P<toks>[0-9.eE+-]+) vram=(?P<vram>[0-9.eE+-]+)MB"
)
RESUME_RE = re.compile(r"\[TRAIN\] Resumed from step (?P<step>\d+)")
KV_RE = re.compile(r"([A-Za-z0-9_/-]+)=([^\s]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--staged-log", required=True, type=Path)
    parser.add_argument("--template-log", required=True, type=Path)
    parser.add_argument("--lift-log", required=True, type=Path)
    parser.add_argument("--continuation-log", required=True, type=Path)
    parser.add_argument("--baseline-log", type=Path)
    return parser.parse_args()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def has_bad_failure(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("fatal", "[error]", "nan", "inf"))


def parse_train_log(path: Path) -> dict[str, Any]:
    text = read_text(path)
    train_rows: list[dict[str, float | int]] = []
    for match in TRAIN_RE.finditer(text):
        train_rows.append(
            {
                "step": int(match.group("step")),
                "loss": float(match.group("loss")),
                "tok_s": float(match.group("toks")),
                "vram_mb": float(match.group("vram")),
            }
        )
    resume_match = RESUME_RE.search(text)
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "complete": "[TRAIN] Complete" in text,
        "bad_failure": has_bad_failure(text),
        "resumed_step": int(resume_match.group("step")) if resume_match else None,
        "train_rows": train_rows,
        "first_train_row": train_rows[0] if train_rows else None,
        "last_train_row": train_rows[-1] if train_rows else None,
    }


def latest_kv_line(text: str, marker: str) -> dict[str, Any] | None:
    lines = [line for line in text.splitlines() if marker in line]
    if not lines:
        return None
    line = lines[-1]
    parsed: dict[str, Any] = {"line": line}
    for key, raw_value in KV_RE.findall(line):
        value: Any = raw_value
        try:
            if any(ch in raw_value for ch in ".eE"):
                value = float(raw_value)
                if not math.isfinite(value):
                    value = raw_value
            else:
                value = int(raw_value)
        except ValueError:
            value = raw_value
        parsed[key] = value
    return parsed


def main() -> int:
    args = parse_args()
    continuation_text = read_text(args.continuation_log)
    lift_text = read_text(args.lift_log)

    staged = parse_train_log(args.staged_log)
    template = parse_train_log(args.template_log)
    continuation = parse_train_log(args.continuation_log)
    baseline = parse_train_log(args.baseline_log) if args.baseline_log else None

    activation = {
        "checkpoint_lift_ok": "[R426_CHECKPOINT_LIFT]" in lift_text and "status=ok" in lift_text,
        "readout_requested": "[R423_NATIVE_GROWTH_READOUT_QUOTIENT] requested=1" in continuation_text,
        "readout_active": "[R423_NATIVE_GROWTH_READOUT_QUOTIENT] active=1" in continuation_text,
        "readout_result": latest_kv_line(
            continuation_text,
            "[R423_NATIVE_GROWTH_READOUT_QUOTIENT_RESULT]",
        ),
        "interaction_requested": "[R423_NATIVE_GROWTH_INTERACTION_SAFE_UPDATE] requested=1" in continuation_text,
        "interaction_update": latest_kv_line(
            continuation_text,
            "[R423_NATIVE_GROWTH_INTERACTION_SAFE_UPDATE] step=",
        ),
        "genotype_requested": "[R423_NATIVE_GROWTH_GENOTYPE_COMPACT_UPDATE] requested=1" in continuation_text,
        "genotype_summary": latest_kv_line(
            continuation_text,
            "[R423_NATIVE_GROWTH_GENOTYPE_COMPACT_UPDATE_SUMMARY]",
        ),
    }

    required_passes = {
        "staged_completed": bool(staged["complete"] and not staged["bad_failure"]),
        "template_completed": bool(template["complete"] and not template["bad_failure"]),
        "lift_completed": bool(activation["checkpoint_lift_ok"]),
        "continuation_resumed": continuation["resumed_step"] is not None,
        "continuation_completed": bool(continuation["complete"] and not continuation["bad_failure"]),
        "readout_live": bool(activation["readout_requested"] and activation["readout_active"] and activation["readout_result"]),
        "interaction_live": bool(activation["interaction_requested"] and activation["interaction_update"]),
        "genotype_live": bool(activation["genotype_requested"] and activation["genotype_summary"]),
    }

    if baseline is not None:
        required_passes["baseline_completed"] = bool(baseline["complete"] and not baseline["bad_failure"])

    decision = (
        "PASS_R427_UNIFIED_GROWTH_LADDER_ACTIVATION"
        if all(required_passes.values())
        else "FAIL_R427_UNIFIED_GROWTH_LADDER_ACTIVATION"
    )

    result: dict[str, Any] = {
        "decision": decision,
        "claim_scope": (
            "Activation and lineage smoke only. This proves the no-reset "
            "small-to-large checkpoint lift and live growth mechanisms are "
            "wired together; it does not claim production speedup."
        ),
        "manifest": str(args.manifest),
        "required_passes": required_passes,
        "activation": activation,
        "logs": {
            "staged": staged,
            "template": template,
            "continuation": continuation,
            "baseline": baseline,
            "lift_log": str(args.lift_log),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[R427_GROWTH_LADDER_ANALYZER] decision={decision} output={args.output}")
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
