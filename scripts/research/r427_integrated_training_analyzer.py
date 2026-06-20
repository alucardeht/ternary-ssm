#!/usr/bin/env python3
"""Analyze the integrated R427 trainable preset.

This checks the whole executable composition:

1. capability material is generated and passes its gate;
2. local artifact shape is generated and passes its gate;
3. both are materialized into a v12 uint16 dataset;
4. a smaller checkpoint is lifted into the full model;
5. full continuation resumes and trains on the materialized dataset while the
   growth stack and segment-credit observers are live.
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
    parser.add_argument("--material-gate", required=True, type=Path)
    parser.add_argument("--artifact-gate", required=True, type=Path)
    parser.add_argument("--materialized-summary", required=True, type=Path)
    parser.add_argument("--staged-log", required=True, type=Path)
    parser.add_argument("--template-log", required=True, type=Path)
    parser.add_argument("--lift-log", required=True, type=Path)
    parser.add_argument("--continuation-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}


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


def latest_line(text: str, marker: str, *, step_required: bool = False) -> dict[str, Any] | None:
    lines = [line for line in text.splitlines() if marker in line]
    if step_required:
        lines = [line for line in lines if " step=" in line]
    if not lines:
        return None
    return parse_kv_line(lines[-1])


def latest_receipt_line(text: str) -> dict[str, Any] | None:
    lines = [
        line
        for line in text.splitlines()
        if "RECEIPT_CHALLENGE_GATE] step=" in line
        and "caught_harm_fraction=" in line
        and "priced_ratio=" in line
    ]
    if not lines:
        return None
    return parse_kv_line(lines[-1])


def has_bad_failure(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("fatal", "[error]", "nan", "inf"))


def parse_train_log(path: Path) -> dict[str, Any]:
    text = read_text(path)
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
    resume_match = RESUME_RE.search(text)
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "complete": "[TRAIN] Complete" in text,
        "bad_failure": has_bad_failure(text),
        "resumed_step": int(resume_match.group("step")) if resume_match else None,
        "first_train_row": rows[0] if rows else None,
        "last_train_row": rows[-1] if rows else None,
    }


def finite_numeric_values(row: dict[str, Any] | None, keys: list[str]) -> bool:
    if row is None:
        return False
    for key in keys:
        value = row.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return False
    return True


def infer_continuation_steps_from_manifest(manifest: dict[str, Any]) -> int | None:
    arms = manifest.get("arms")
    if not isinstance(arms, list):
        return None
    for arm in arms:
        if not isinstance(arm, dict) or arm.get("name") != "r427-integrated-lifted-continuation":
            continue
        command = arm.get("command")
        if not isinstance(command, str):
            return None
        match = re.search(r'\.\\train\.exe\s+"[^"]+"\s+(?P<steps>\d+)\b', command)
        return int(match.group("steps")) if match else None
    return None


def main() -> int:
    args = parse_args()
    manifest = read_json(args.manifest)
    material_gate = read_json(args.material_gate)
    artifact_gate = read_json(args.artifact_gate)
    materialized = read_json(args.materialized_summary)
    lift_text = read_text(args.lift_log)
    continuation_text = read_text(args.continuation_log)

    staged = parse_train_log(args.staged_log)
    template = parse_train_log(args.template_log)
    continuation = parse_train_log(args.continuation_log)

    readout = latest_line(continuation_text, "[R423_NATIVE_GROWTH_READOUT_QUOTIENT_RESULT]")
    interaction = latest_line(continuation_text, "[R423_NATIVE_GROWTH_INTERACTION_SAFE_UPDATE] step=")
    genotype = latest_line(continuation_text, "[R423_NATIVE_GROWTH_GENOTYPE_COMPACT_UPDATE_SUMMARY]")
    boundary = latest_line(continuation_text, "[BOUNDARY_COTANGENT_CREDIT_GATE]", step_required=True)
    variance = latest_line(continuation_text, "[VARIANCE_PRICED_CREDIT_GATE]", step_required=True)
    original = latest_line(continuation_text, "[ORIGINAL_CE_CREDIT_DEBT_OBSERVER]", step_required=True)
    r429_decision = latest_line(continuation_text, "[R429_DECISION_EQUIVALENCE_WITNESS]", step_required=True)
    receipt = latest_receipt_line(continuation_text)

    steps = manifest.get("steps") if isinstance(manifest.get("steps"), dict) else {}
    continuation_steps = steps.get("continuation")
    try:
        continuation_steps = int(continuation_steps)
    except (TypeError, ValueError):
        continuation_steps = infer_continuation_steps_from_manifest(manifest)
    zero_continuation_diagnostic = continuation_steps == 0

    required_passes = {
        "material_gate_pass": material_gate.get("decision") == "PASS_R427_CAPABILITY_MATERIAL_GATE",
        "artifact_gate_pass": artifact_gate.get("decision") == "PASS_R427_LOCAL_ARTIFACT_SHAPE_GATE",
        "materialized_dataset_ok": materialized.get("status") == "ok"
        and int(materialized.get("token_count", 0)) >= 8192
        and int(materialized.get("byte_size", 0)) >= 16384,
        "staged_completed": bool(staged["complete"] and not staged["bad_failure"]),
        "template_completed": bool(template["complete"] and not template["bad_failure"]),
        "lift_completed": "[R426_CHECKPOINT_LIFT]" in lift_text and "status=ok" in lift_text,
        "continuation_resumed": continuation["resumed_step"] is not None,
        "continuation_completed": bool(continuation["complete"] and not continuation["bad_failure"]),
        "growth_readout_live": (
            "[R423_NATIVE_GROWTH_READOUT_QUOTIENT] active=1" in continuation_text
            if zero_continuation_diagnostic
            else "[R423_NATIVE_GROWTH_READOUT_QUOTIENT] active=1" in continuation_text and readout is not None
        ),
        "growth_interaction_live": "[R423_NATIVE_GROWTH_INTERACTION_SAFE_UPDATE] requested=1" in continuation_text
        and (zero_continuation_diagnostic or interaction is not None),
        "growth_genotype_live": "[R423_NATIVE_GROWTH_GENOTYPE_COMPACT_UPDATE] requested=1" in continuation_text
        and (zero_continuation_diagnostic or genotype is not None),
        "boundary_credit_live": "[BOUNDARY_COTANGENT_CREDIT_GATE] active=1" in continuation_text
        and (
            zero_continuation_diagnostic
            or finite_numeric_values(boundary, ["residual_fraction", "priced_ratio", "descent_per_paid_work"])
        ),
        "variance_credit_live": "[VARIANCE_PRICED_CREDIT_GATE] active=1" in continuation_text
        and (
            zero_continuation_diagnostic
            or finite_numeric_values(variance, ["sampled_fraction", "variance_proxy", "descent_per_paid_work"])
        ),
        "original_ce_credit_live": "[ORIGINAL_CE_CREDIT_DEBT_OBSERVER] active=1" in continuation_text
        and (
            zero_continuation_diagnostic
            or finite_numeric_values(original, ["local_progress_ratio", "local_residual_fraction", "surface_progress_ratio"])
        ),
        "receipt_credit_live": "[R423_NATIVE_GROWTH_RECEIPT_CHALLENGE_GATE] requested=1" in continuation_text
        and (
            zero_continuation_diagnostic
            or finite_numeric_values(receipt, ["caught_harm_fraction", "priced_ratio", "mean_full_risk", "mean_receipt_risk"])
        ),
        "r429_decision_equivalence_live": (
            "[R429_DECISION_EQUIVALENCE_WITNESS] requested=1" not in continuation_text
            or (
                zero_continuation_diagnostic
                or (
                    r429_decision is not None
                    and int(r429_decision.get("mismatch_updates", 1)) == 0
                    and int(r429_decision.get("identifying", 0)) == 1
                )
            )
        ),
    }

    material_signals = {
        "readout_net_speedup_estimate": readout.get("net_speedup_estimate") if readout else None,
        "interaction_pass": interaction.get("pass") if interaction else None,
        "genotype_pass": genotype.get("pass") if genotype else None,
        "boundary_pass": boundary.get("pass") if boundary else None,
        "variance_pass": variance.get("pass") if variance else None,
        "original_ce_local_pass": original.get("local_pass") if original else None,
        "original_ce_surface_pass": original.get("surface_pass") if original else None,
        "receipt_pass": receipt.get("pass") if receipt else None,
        "r429_decision_equivalence_pass": r429_decision.get("pass") if r429_decision else None,
        "r429_decision_equivalence_mismatch_updates": r429_decision.get("mismatch_updates") if r429_decision else None,
        "r429_decision_equivalence_certified_fraction": r429_decision.get("certified_fraction") if r429_decision else None,
        "r429_decision_equivalence_avoided_work_fraction": r429_decision.get("avoided_work_fraction") if r429_decision else None,
    }

    if all(required_passes.values()):
        decision = (
            "PASS_R427_INTEGRATED_STAGED_DIAGNOSTIC"
            if zero_continuation_diagnostic
            else "PASS_R427_INTEGRATED_TRAINING_PRESET"
        )
    else:
        decision = "FAIL_R427_INTEGRATED_TRAINING_PRESET"
    result = {
        "decision": decision,
        "claim_scope": (
            "Integrated executable preset readiness. This proves R427 material, "
            "artifact descriptor, no-reset growth, and credit observers can run "
            "together on the v12 training path. It does not claim final 7B "
            "quality, final speedup, or overnight convergence."
        ),
        "manifest": str(args.manifest),
        "steps": steps,
        "zero_continuation_diagnostic": zero_continuation_diagnostic,
        "required_passes": required_passes,
        "material_signals": material_signals,
        "material_gate": {
            "decision": material_gate.get("decision"),
            "material_accuracy": material_gate.get("material_accuracy"),
            "control_accuracy": material_gate.get("control_accuracy"),
            "strong_pass_count": material_gate.get("strong_pass_count"),
        },
        "artifact_gate": {
            "decision": artifact_gate.get("decision"),
            "pass_count": artifact_gate.get("pass_count"),
            "gate_count": artifact_gate.get("gate_count"),
        },
        "materialized_dataset": materialized,
        "logs": {
            "staged": staged,
            "template": template,
            "continuation": continuation,
            "lift_log": str(args.lift_log),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[R427_INTEGRATED_TRAINING_ANALYZER] decision={decision} output={args.output}")
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
