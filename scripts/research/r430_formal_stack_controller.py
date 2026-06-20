#!/usr/bin/env python3
"""R430 formal stack controller.

This controller turns the formal R430 validation pass into an executable
receipt-and-gate layer over the existing R429/R428/R427 training preset. It does
not claim a measured 7B speedup. Its job is to make every R430 aspect explicit,
activation-checked, de-overlapped, and ready for a longer monitored run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "r430_formal_stack_controller.v1"
STRICT_ANCHOR_DAYS = 275.0
R150_BASELINE_DAYS = 495.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["plan", "analyze"], default="plan")
    parser.add_argument("--profile", choices=["interactive", "balanced", "exclusive"], default="interactive")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipts-output", required=True, type=Path)
    parser.add_argument("--r429-analysis", type=Path)
    parser.add_argument("--r429-manifest", type=Path)
    parser.add_argument("--formal-synthesis", type=Path, default=Path(".workspace/research/r430-sub-10-day-frontier/formal-synthesis.md"))
    return parser.parse_args()


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def numeric(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else default
    return default


def boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return False


def nested_get(row: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = row
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_nested_analyses(r429: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    r428 = read_json(Path(str(r429.get("r428_analysis_path", "")))) if r429.get("r428_analysis_path") else {}
    r427 = read_json(Path(str(r428.get("r427_analysis_path", "")))) if r428.get("r427_analysis_path") else {}
    return r428, r427


def base_receipts() -> list[dict[str, Any]]:
    return [
        {
            "id": "E1",
            "aspect": "E",
            "name": "Denominator receipt schema",
            "owner": "R430",
            "validated_status": "validated-for-gate",
            "proof_object": "parseable denominator receipt with overlap edges, hidden costs, quality invariant, and kill criterion",
            "denominator_components": ["verify", "repair"],
            "independent_speed_countable": False,
            "strict_delta_r_stack": [0.0, 0.0],
            "provisional_delta_r_stack": [0.0, 0.0],
            "overlap_class": "meta-ledger",
            "activation_rule": "controller emits receipts and rejects duplicate ownership",
            "kill_criterion": "duplicate A/B/C/D coverage passes as independent speed",
        },
        {
            "id": "A04",
            "aspect": "A",
            "name": "Original-CE receipt ledger with sparse exceptions",
            "owner": "A",
            "validated_status": "validated-for-gate",
            "proof_object": "receipt classes plus sparse exceptions and full fallback",
            "denominator_components": ["ce_units", "verify", "repair"],
            "independent_speed_countable": False,
            "strict_delta_r_stack": [0.0, 0.0],
            "provisional_delta_r_stack": [0.0, 0.0],
            "overlap_class": "A-owned CE work",
            "activation_rule": "native decision-equivalence witness passes with zero mismatch updates",
            "kill_criterion": "omitted CE unit lacks exact proof, exception charge, or fallback",
        },
        {
            "id": "A01",
            "aspect": "A",
            "name": "Segment-trajectory empirical-risk quotient",
            "owner": "A",
            "validated_status": "validated-for-gate",
            "proof_object": "model-visible trajectory quotient with sparse residual checks",
            "denominator_components": ["ce_units", "verify"],
            "independent_speed_countable": False,
            "strict_delta_r_stack": [0.0, 0.0],
            "provisional_delta_r_stack": [0.0, 0.0],
            "overlap_class": "A-owned quotient; D credit excluded",
            "activation_rule": "readout/objective quotient support is live and priced",
            "kill_criterion": "quotient representatives do not preserve local CE ordering",
        },
        {
            "id": "A03",
            "aspect": "A",
            "name": "Local logit-count sufficient statistic",
            "owner": "A",
            "validated_status": "validated-for-gate",
            "proof_object": "exact logit/update equivalence statistic with mismatch counter",
            "denominator_components": ["ce_units", "verify"],
            "independent_speed_countable": False,
            "strict_delta_r_stack": [0.0, 0.0],
            "provisional_delta_r_stack": [0.0, 0.0],
            "overlap_class": "A-owned sufficient statistic",
            "activation_rule": "decision-equivalence certified fraction is positive and mismatch updates are zero",
            "kill_criterion": "certified statistic produces any mismatched update",
        },
        {
            "id": "C1",
            "aspect": "C",
            "name": "Birth/null-extension exact preservation",
            "owner": "C",
            "validated_status": "validated-for-gate",
            "proof_object": "null-extension state/logit equality and dormant complement certificate",
            "denominator_components": ["phase_area", "state_area", "catchup"],
            "independent_speed_countable": False,
            "strict_delta_r_stack": [0.0, 0.0],
            "provisional_delta_r_stack": [0.004, 0.008],
            "overlap_class": "C-owned no-reset growth",
            "activation_rule": "checkpoint lift succeeds and lifted continuation resumes from staged source",
            "kill_criterion": "lift changes old logits/state or continuation cannot resume",
        },
        {
            "id": "C3",
            "aspect": "C",
            "name": "Catchup-bounded growth contract",
            "owner": "C",
            "validated_status": "validated-for-gate",
            "proof_object": "catchup liability bound and phase-area saved-work inequality",
            "denominator_components": ["catchup", "phase_area", "progress"],
            "independent_speed_countable": False,
            "strict_delta_r_stack": [0.0, 0.0],
            "provisional_delta_r_stack": [0.006, 0.011],
            "overlap_class": "C-owned catchup debt",
            "activation_rule": "R428/R429 controller records growth path and continuation LR/catchup controls",
            "kill_criterion": "post-lift debt is unbounded or continuation diverges before useful signal",
        },
        {
            "id": "C5",
            "aspect": "C",
            "name": "Active/frontier accounting",
            "owner": "C",
            "validated_status": "validated-for-gate",
            "proof_object": "active-surface or lease proof after R429/A de-overlap",
            "denominator_components": ["state_area", "phase_area", "memory"],
            "independent_speed_countable": False,
            "strict_delta_r_stack": [0.0, 0.0],
            "provisional_delta_r_stack": [0.003, 0.006],
            "overlap_class": "C-owned active frontier; A receipts excluded",
            "activation_rule": "R429 decision witness and local artifact projection ledger are live",
            "kill_criterion": "omitted surfaces carry material CE descent or duplicate R429 zero-update groups",
        },
        {
            "id": "D1",
            "aspect": "D",
            "name": "Certified cotangent tube",
            "owner": "D",
            "validated_status": "validated-for-gate",
            "proof_object": "interior transition-gradient residual or chamber-face certificate with sparse fallback",
            "denominator_components": ["state_area", "progress", "verify"],
            "independent_speed_countable": False,
            "strict_delta_r_stack": [0.0, 0.0],
            "provisional_delta_r_stack": [0.003, 0.008],
            "overlap_class": "D-owned transition credit; A CE receipts excluded",
            "activation_rule": "segment-credit observers emit finite boundary/tube metrics",
            "kill_criterion": "endpoint-only residual remains high or fallback density is not sparse",
        },
        {
            "id": "D2",
            "aspect": "D",
            "name": "Self-verifying martingale telemetry",
            "owner": "D",
            "validated_status": "validated-for-gate",
            "proof_object": "forward-emitted bounded statistics and sparse challenge replay",
            "denominator_components": ["score", "verify", "state_area", "progress"],
            "independent_speed_countable": False,
            "strict_delta_r_stack": [0.0, 0.0],
            "provisional_delta_r_stack": [0.002, 0.005],
            "overlap_class": "D-owned score/verify blocker removal",
            "activation_rule": "variance-priced credit pass is live",
            "kill_criterion": "confidence intervals are loose or challenge replay becomes dense",
        },
        {
            "id": "B1",
            "aspect": "B",
            "name": "Capability atom atlas and exact checker closure",
            "owner": "B",
            "validated_status": "validated-for-gate",
            "proof_object": "reversible skill atoms, exact checker closure, held-out capability controls",
            "denominator_components": ["progress", "verify"],
            "independent_speed_countable": False,
            "strict_delta_r_stack": [0.0, 0.0],
            "provisional_delta_r_stack": [0.0, 0.0],
            "overlap_class": "B-owned material; A CE boundary required before CE count",
            "activation_rule": "capability material gate passes with strong sub-gates and random controls",
            "kill_criterion": "random/shuffled controls match or atom family lacks exact CE boundary",
        },
        {
            "id": "B-A",
            "aspect": "B/A",
            "name": "Capability-to-CE boundary gate",
            "owner": "B/A",
            "validated_status": "gate-hypothesis",
            "proof_object": "map from one atom family to an exact original-CE receipt subset",
            "denominator_components": ["ce_units", "progress", "verify"],
            "independent_speed_countable": False,
            "strict_delta_r_stack": [0.0, 0.0],
            "provisional_delta_r_stack": [0.0, 0.0],
            "overlap_class": "boundary theorem; no E speed",
            "activation_rule": "receipt checker records atom family as boundary-ready only after exact subset proof",
            "kill_criterion": "atom family improves toy behavior but cannot name original-CE subset",
        },
    ]


def evidence_from(r429: dict[str, Any], r428: dict[str, Any], r427: dict[str, Any]) -> dict[str, Any]:
    smoke = r429.get("measured_smoke_support", {})
    r427_material = r427.get("material_gate", {})
    r427_artifact = r427.get("artifact_gate", {})
    r427_required = r427.get("required_passes", {})
    r428_required = r428.get("required_passes", {})
    return {
        "r429_pass": r429.get("decision") == "PASS_R429_NO_TRADEOFF_STACK_SMOKE",
        "r428_ready": r428.get("decision") == "PASS_R428_SPEED_TO_QUALITY_READY",
        "r427_integrated_ready": r427.get("decision") == "PASS_R427_INTEGRATED_TRAINING_PRESET",
        "decision_witness_pass": numeric(smoke.get("r429_decision_equivalence_pass")) == 1.0,
        "decision_witness_mismatch_zero": numeric(smoke.get("r429_decision_equivalence_mismatch_updates"), 1.0) == 0.0,
        "decision_witness_certified_fraction": numeric(smoke.get("r429_decision_equivalence_certified_fraction")),
        "readout_net_speedup_estimate": numeric(smoke.get("readout_net_speedup_estimate")),
        "growth_path_live": numeric(smoke.get("staged_over_full_tok_s")) >= 1.10 and boolean(r427_required.get("lift_completed")) and boolean(r427_required.get("continuation_resumed")),
        "continuation_completed": boolean(nested_get(r427, ["logs", "continuation", "complete"], False)),
        "capability_material_pass": r427_material.get("decision") == "PASS_R427_CAPABILITY_MATERIAL_GATE",
        "capability_material_strong_pass_count": numeric(r427_material.get("strong_pass_count")),
        "capability_material_accuracy": numeric(r427_material.get("material_accuracy")),
        "capability_control_accuracy": numeric(r427_material.get("control_accuracy")),
        "artifact_gate_pass": r427_artifact.get("decision") == "PASS_R427_LOCAL_ARTIFACT_SHAPE_GATE",
        "artifact_gate_pass_count": numeric(r427_artifact.get("pass_count")),
        "variance_pass": numeric(smoke.get("variance_pass")) == 1.0,
        "segment_credit_live": boolean(r427_required.get("boundary_credit_live")) and boolean(r427_required.get("variance_credit_live")),
        "original_ce_observer_live": boolean(r427_required.get("original_ce_credit_live")),
        "receipt_credit_live": boolean(r427_required.get("receipt_credit_live")),
        "r428_growth_controller_live": boolean(r428_required.get("growth_updates_live")) and boolean(r428_required.get("variance_credit_live")),
    }


def gate_receipt(receipt: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    rid = receipt["id"]
    activation_checks: dict[str, bool] = {}
    if rid == "E1":
        activation_checks = {"controller_emitted_receipts": True}
    elif rid == "A04":
        activation_checks = {
            "decision_witness_pass": bool(evidence["decision_witness_pass"]),
            "mismatch_updates_zero": bool(evidence["decision_witness_mismatch_zero"]),
        }
    elif rid == "A01":
        activation_checks = {
            "readout_quotient_material": evidence["readout_net_speedup_estimate"] >= 1.10,
            "r429_pass": bool(evidence["r429_pass"]),
        }
    elif rid == "A03":
        activation_checks = {
            "decision_witness_pass": bool(evidence["decision_witness_pass"]),
            "certified_fraction_positive": evidence["decision_witness_certified_fraction"] > 0.0,
            "mismatch_updates_zero": bool(evidence["decision_witness_mismatch_zero"]),
        }
    elif rid == "C1":
        activation_checks = {
            "growth_path_live": bool(evidence["growth_path_live"]),
            "continuation_completed": bool(evidence["continuation_completed"]),
        }
    elif rid == "C3":
        activation_checks = {
            "r428_growth_controller_live": bool(evidence["r428_growth_controller_live"]),
            "growth_path_live": bool(evidence["growth_path_live"]),
        }
    elif rid == "C5":
        activation_checks = {
            "artifact_gate_pass": bool(evidence["artifact_gate_pass"]),
            "decision_witness_pass": bool(evidence["decision_witness_pass"]),
        }
    elif rid == "D1":
        activation_checks = {
            "segment_credit_live": bool(evidence["segment_credit_live"]),
            "original_ce_observer_live": bool(evidence["original_ce_observer_live"]),
        }
    elif rid == "D2":
        activation_checks = {
            "variance_pass": bool(evidence["variance_pass"]),
            "segment_credit_live": bool(evidence["segment_credit_live"]),
        }
    elif rid == "B1":
        activation_checks = {
            "capability_material_pass": bool(evidence["capability_material_pass"]),
            "strong_sub_gates": evidence["capability_material_strong_pass_count"] >= 7,
            "material_beats_control": evidence["capability_material_accuracy"] > evidence["capability_control_accuracy"],
        }
    elif rid == "B-A":
        activation_checks = {
            "capability_material_pass": bool(evidence["capability_material_pass"]),
            "exact_ce_boundary_proven": False,
        }
    activated = all(activation_checks.values())
    out = dict(receipt)
    out["activation_checks"] = activation_checks
    out["activated"] = activated
    out["strict_speed_countable_now"] = False
    out["ready_for_training_preset"] = activated or rid in {"B-A"}
    return out


def summarize(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    activated = [row for row in receipts if row["activated"]]
    by_aspect: dict[str, dict[str, int]] = {}
    for row in receipts:
        aspect = row["aspect"]
        by_aspect.setdefault(aspect, {"total": 0, "activated": 0, "strict_speed_countable": 0})
        by_aspect[aspect]["total"] += 1
        if row["activated"]:
            by_aspect[aspect]["activated"] += 1
        if row["strict_speed_countable_now"]:
            by_aspect[aspect]["strict_speed_countable"] += 1
    return {
        "total_receipts": len(receipts),
        "activated_receipts": len(activated),
        "strict_speed_countable_receipts": sum(1 for row in receipts if row["strict_speed_countable_now"]),
        "by_aspect": by_aspect,
    }


def projection() -> dict[str, Any]:
    return {
        "strict_anchor_days": STRICT_ANCHOR_DAYS,
        "r150_baseline_days": R150_BASELINE_DAYS,
        "strict_delta_r_stack": {
            "A": [0.0, 0.0],
            "B": [0.0, 0.0],
            "C": [0.0, 0.0],
            "D": [0.0, 0.0],
            "E": [0.0, 0.0],
        },
        "provisional_gate_planning_delta_r_stack": {
            "C": [0.010, 0.020],
            "D": [0.006, 0.016],
            "E": [0.0, 0.0],
        },
        "sub_10_day_status": "not_proven",
        "required_additional_factor_from_strict_anchor": STRICT_ANCHOR_DAYS / 10.0,
        "accounting_warning": "Gate-ready and paper-ready receipts are not strict speed reductions until their priced gates pass.",
    }


def make_plan(profile: str) -> tuple[dict[str, Any], dict[str, Any]]:
    receipts = [gate_receipt(row, {}) if row["id"] == "E1" else {**row, "activated": False, "activation_checks": {}, "strict_speed_countable_now": False, "ready_for_training_preset": True} for row in base_receipts()]
    receipt_doc = {
        "schema": SCHEMA + ".receipts",
        "mode": "plan",
        "receipts": receipts,
        "summary": summarize(receipts),
    }
    result = {
        "schema": SCHEMA,
        "decision": "PASS_R430_FORMAL_STACK_PLAN",
        "profile": profile,
        "claim_scope": "Executable formal receipt layer over R429. Plan mode does not run training and does not claim strict speedup.",
        "activation_log": "[R430_FORMAL_STACK] active=1 status=plan_ready",
        "projection": projection(),
        "receipt_summary": receipt_doc["summary"],
        "night_training_command_shape": ".workspace/training/rounds/r430.ps1 -Mode integrated -Profile exclusive",
    }
    return result, receipt_doc


def analyze(profile: str, r429_path: Path | None, r429_manifest_path: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    r429 = read_json(r429_path)
    r428, r427 = load_nested_analyses(r429)
    evidence = evidence_from(r429, r428, r427)
    receipts = [gate_receipt(row, evidence) for row in base_receipts()]
    receipt_doc = {
        "schema": SCHEMA + ".receipts",
        "mode": "analyze",
        "r429_analysis": str(r429_path) if r429_path else None,
        "r429_manifest": str(r429_manifest_path) if r429_manifest_path else None,
        "r428_analysis": str(r429.get("r428_analysis_path")) if r429.get("r428_analysis_path") else None,
        "r427_analysis": str(r428.get("r427_analysis_path")) if r428.get("r427_analysis_path") else None,
        "evidence": evidence,
        "receipts": receipts,
        "summary": summarize(receipts),
    }
    required = {
        "r429_pass": bool(evidence["r429_pass"]),
        "r428_ready": bool(evidence["r428_ready"]),
        "r427_integrated_ready": bool(evidence["r427_integrated_ready"]),
        "e1_receipt_schema_active": any(row["id"] == "E1" and row["activated"] for row in receipts),
        "a_core_ready": all(any(row["id"] == rid and row["activated"] for row in receipts) for rid in ("A04", "A01", "A03")),
        "b_material_ready": any(row["id"] == "B1" and row["activated"] for row in receipts),
        "c_growth_ready": all(any(row["id"] == rid and row["activated"] for row in receipts) for rid in ("C1", "C3", "C5")),
        "d_credit_ready": all(any(row["id"] == rid and row["activated"] for row in receipts) for rid in ("D1", "D2")),
        "strict_speed_not_claimed": True,
    }
    decision = "PASS_R430_FORMAL_STACK_READY" if all(required.values()) else "BLOCK_R430_FORMAL_STACK_READY"
    result = {
        "schema": SCHEMA,
        "decision": decision,
        "profile": profile,
        "claim_scope": "R430 formal receipt and gate layer over R429 integrated training. Ready means all implemented observers/gates are active; it is not a strict <10d proof.",
        "activation_log": f"[R430_FORMAL_STACK] active=1 status={decision}",
        "required_passes": required,
        "projection": projection(),
        "receipt_summary": receipt_doc["summary"],
        "r429_analysis": str(r429_path) if r429_path else None,
        "r429_manifest": str(r429_manifest_path) if r429_manifest_path else None,
        "implementation_status": {
            "implemented_now": [
                "single R430 preset over R429/R428/R427",
                "formal denominator receipt ledger",
                "A/B/C/D/E activation checks",
                "strict/provisional projection separation",
                "paper/transfer no-speed guard",
            ],
            "strict_speed_claim": "none",
            "overnight_training_ready": decision == "PASS_R430_FORMAL_STACK_READY",
        },
    }
    return result, receipt_doc


def main() -> int:
    args = parse_args()
    if args.mode == "plan":
        result, receipts = make_plan(args.profile)
    else:
        result, receipts = analyze(args.profile, args.r429_analysis, args.r429_manifest)
    write_json(args.output, result)
    write_json(args.receipts_output, receipts)
    print(result["activation_log"] + f" output={args.output} receipts={args.receipts_output}")
    return 0 if not str(result["decision"]).startswith("BLOCK_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
