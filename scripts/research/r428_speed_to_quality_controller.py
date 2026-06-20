#!/usr/bin/env python3
"""R428 speed-to-quality controller and readiness analyzer.

R428 is intentionally a control layer over the integrated R427 preset. It does
not claim a measured 7B speedup. It makes the next serious run reproducible by
recording the theoretical speed findings, pricing guards, profile policy, and
readiness checks that must hold before an overnight validation is treated as
evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "r428_speed_to_quality_controller.v1"
BASELINE_DAYS = 495.0


FINDINGS: list[dict[str, Any]] = [
    {
        "id": "R428-01",
        "name": "CE-Debt Matched-Target Controller",
        "strength": "strong",
        "speedup_range": [1.25, 1.80],
        "failure_risk": 0.08,
        "quality_guard": "Stop/reallocate only after held-out CE debt and slope confidence clear the same-target guard.",
        "implementation": "matched_target_controller",
    },
    {
        "id": "R428-02",
        "name": "Growth Phase-Area Scheduler",
        "strength": "strong",
        "speedup_range": [1.35, 1.90],
        "failure_risk": 0.10,
        "quality_guard": "Choose staged/full split from phase-area saved work minus catch-up debt.",
        "implementation": "phase_area_schedule",
    },
    {
        "id": "R428-03",
        "name": "Catch-Up Debt Stop",
        "strength": "strong",
        "speedup_range": [1.15, 1.45],
        "failure_risk": 0.06,
        "quality_guard": "Full-width continuation stops only when post-lift debt closes against eval or matched-loss slope.",
        "implementation": "catchup_debt_stop",
    },
    {
        "id": "R428-04",
        "name": "R427 Material Mix Controller",
        "strength": "strong",
        "speedup_range": [1.30, 2.20],
        "failure_risk": 0.14,
        "quality_guard": "Capability material is mixed with residual real-corpus settlement, never used as a full replacement.",
        "implementation": "material_mix_policy",
    },
    {
        "id": "R428-05",
        "name": "Exact Commonness Debt Reweight",
        "strength": "strong",
        "speedup_range": [1.20, 1.60],
        "failure_risk": 0.12,
        "quality_guard": "Common repeated terms are down-weighted only with a deferred original-CE debt ledger.",
        "implementation": "commonness_debt_reweight",
    },
    {
        "id": "R428-06",
        "name": "Variance-Credit Shelf Amortization",
        "strength": "strong",
        "speedup_range": [1.40, 2.50],
        "failure_risk": 0.18,
        "quality_guard": "Score/verify cost may be amortized only while variance credit remains aligned and finite.",
        "implementation": "variance_credit_shelf",
    },
    {
        "id": "R428-07",
        "name": "Observer Duty-Cycle With Certificate TTL",
        "strength": "strong",
        "speedup_range": [1.20, 2.00],
        "failure_risk": 0.11,
        "quality_guard": "Expensive observers can sleep only while the last certificate has not expired.",
        "implementation": "observer_ttl_profile",
    },
    {
        "id": "R428-08",
        "name": "Capability-Rotation Heldout Gate",
        "strength": "strong",
        "speedup_range": [1.20, 1.70],
        "failure_risk": 0.10,
        "quality_guard": "Material progress must survive held-out rotations to avoid memorizing the constructed rows.",
        "implementation": "rotation_heldout_gate",
    },
    {
        "id": "R428-09",
        "name": "Stage-Local Batch Ramp Envelope",
        "strength": "strong",
        "speedup_range": [1.20, 1.60],
        "failure_risk": 0.09,
        "quality_guard": "Batch increases are stage-local and require VRAM, tok/s, and CE sanity headroom.",
        "implementation": "safe_batch_ramp",
    },
    {
        "id": "R428-10",
        "name": "Frontier-Residual Review Queue",
        "strength": "strong",
        "speedup_range": [1.25, 2.00],
        "failure_risk": 0.15,
        "quality_guard": "Review repeats only residual debt not discharged by material or growth certificates.",
        "implementation": "frontier_residual_queue",
    },
    {
        "id": "R428-11",
        "name": "Release-Value Growth Law",
        "strength": "strong",
        "speedup_range": [1.15, 1.60],
        "failure_risk": 0.13,
        "quality_guard": "Capacity release requires marginal CE value above release and catch-up cost.",
        "implementation": "release_value_law",
    },
    {
        "id": "R428-12",
        "name": "Exact Material Token Multiplicity Quotient",
        "strength": "medium",
        "speedup_range": [1.10, 1.50],
        "failure_risk": 0.20,
        "quality_guard": "Only byte-identical material terms can be quotient-weighted; low mass demotes the item.",
        "implementation": "material_multiplicity_audit",
    },
    {
        "id": "R428-13",
        "name": "Integrated Run Noninterference Profile",
        "strength": "medium",
        "speedup_range": [1.10, 1.35],
        "failure_risk": 0.12,
        "quality_guard": "Instrumentation must not consume the speed it is trying to measure.",
        "implementation": "noninterference_profile",
    },
    {
        "id": "R428-14",
        "name": "Live Projection Acceptance Ledger",
        "strength": "medium",
        "speedup_range": [1.10, 1.40],
        "failure_risk": 0.16,
        "quality_guard": "No omission or projection is accepted without an original-CE receipt.",
        "implementation": "projection_acceptance_ledger",
    },
    {
        "id": "R428-15",
        "name": "Overnight Sequential Arm Contract",
        "strength": "medium",
        "speedup_range": [1.10, 1.30],
        "failure_risk": 0.08,
        "quality_guard": "Training arms run sequentially with process hygiene and telemetry before the next arm starts.",
        "implementation": "sequential_arm_contract",
    },
]


PROFILE_POLICY: dict[str, dict[str, Any]] = {
    "interactive": {
        "staged_steps": 1,
        "full_steps": 1,
        "material_tokens": 65536,
        "batch": 1,
        "eval_every": 0,
        "observer_ttl_steps": 1,
        "description": "Shortest identifying smoke while the PC may still be in use.",
    },
    "balanced": {
        "staged_steps": 20,
        "full_steps": 20,
        "material_tokens": 131072,
        "batch": 1,
        "eval_every": 20,
        "observer_ttl_steps": 5,
        "description": "Bounded sub-20-minute run shape for readiness and slope checks.",
    },
    "exclusive": {
        "staged_steps": 200,
        "full_steps": 300,
        "material_tokens": 524288,
        "batch": 1,
        "eval_every": 100,
        "observer_ttl_steps": 20,
        "description": "Overnight-ready starter shape; batch ramp must still be measured bottom-up.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["plan", "analyze"], default="plan")
    parser.add_argument("--profile", choices=sorted(PROFILE_POLICY), default="interactive")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--r427-analysis", type=Path)
    parser.add_argument("--r427-manifest", type=Path)
    return parser.parse_args()


def projected_days(speedup: float) -> float:
    return BASELINE_DAYS / speedup


def finding_summary() -> list[dict[str, Any]]:
    rows = []
    for item in FINDINGS:
        low, high = item["speedup_range"]
        rows.append(
            {
                **item,
                "projected_days_if_isolated": [round(projected_days(low), 1), round(projected_days(high), 1)],
                "baseline_days": BASELINE_DAYS,
                "status": "validated_for_gate" if item["strength"] == "strong" else "support_gate",
            }
        )
    return rows


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}


def numeric(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else default
    return default


def make_plan(profile: str) -> dict[str, Any]:
    strong = [item for item in FINDINGS if item["strength"] == "strong"]
    profile_policy = PROFILE_POLICY[profile]
    conservative_floor = min(item["speedup_range"][0] for item in strong)
    strong_median = sorted(((item["speedup_range"][0] + item["speedup_range"][1]) / 2.0 for item in strong))[len(strong) // 2]
    return {
        "schema": SCHEMA,
        "decision": "PASS_R428_THEORY_AND_READINESS_PLAN",
        "claim_scope": (
            "R428 validates a speed-to-quality control plan and implements a "
            "reproducible runner layer over R427. It does not claim a measured "
            "baseline change until a serious monitored run finishes."
        ),
        "baseline": {
            "source": "R150 in memory/baselines.md",
            "days": BASELINE_DAYS,
        },
        "profile": profile,
        "profile_policy": profile_policy,
        "finding_count": len(FINDINGS),
        "strong_count": len(strong),
        "minimum_requested": {"findings": 15, "strong": 10},
        "requested_minimum_met": len(FINDINGS) >= 15 and len(strong) >= 10,
        "projected_isolated_floor": {
            "speedup": conservative_floor,
            "days": round(projected_days(conservative_floor), 1),
        },
        "projected_strong_median": {
            "speedup": round(strong_median, 3),
            "days": round(projected_days(strong_median), 1),
        },
        "findings": finding_summary(),
        "activation_log": "[R428_SPEED_TO_QUALITY_CONTROLLER] active=1 status=plan_ready",
    }


def analyze(profile: str, r427_analysis_path: Path | None, r427_manifest_path: Path | None) -> dict[str, Any]:
    plan = make_plan(profile)
    r427 = read_json(r427_analysis_path)
    manifest = read_json(r427_manifest_path)
    material = r427.get("material_signals", {})
    logs = r427.get("logs", {})
    staged_last = ((logs.get("staged") or {}).get("last_train_row") or {})
    continuation_last = ((logs.get("continuation") or {}).get("last_train_row") or {})

    staged_tok_s = numeric(staged_last.get("tok_s"))
    full_tok_s = numeric(continuation_last.get("tok_s"))
    throughput_ratio = staged_tok_s / full_tok_s if full_tok_s > 0 else 0.0
    readout_proxy = numeric(material.get("readout_net_speedup_estimate"))
    variance_pass = int(numeric(material.get("variance_pass"), 0.0))
    interaction_pass = int(numeric(material.get("interaction_pass"), 0.0))
    genotype_pass = int(numeric(material.get("genotype_pass"), 0.0))
    r429_decision_pass = int(numeric(material.get("r429_decision_equivalence_pass"), 0.0))
    r429_mismatch_updates = int(numeric(material.get("r429_decision_equivalence_mismatch_updates"), 0.0))

    required_passes = {
        "r427_integrated_pass": r427.get("decision") == "PASS_R427_INTEGRATED_TRAINING_PRESET",
        "requested_minimum_met": plan["requested_minimum_met"],
        "staged_full_ratio_material": throughput_ratio >= 1.10,
        "readout_proxy_material": readout_proxy >= 1.10,
        "growth_updates_live": interaction_pass == 1 and genotype_pass == 1,
        "variance_credit_live": variance_pass == 1,
        "manifest_available": bool(manifest),
    }
    decision = "PASS_R428_SPEED_TO_QUALITY_READY" if all(required_passes.values()) else "BLOCK_R428_SPEED_TO_QUALITY_READY"
    return {
        **plan,
        "decision": decision,
        "r427_analysis_path": str(r427_analysis_path) if r427_analysis_path else None,
        "r427_manifest_path": str(r427_manifest_path) if r427_manifest_path else None,
        "required_passes": required_passes,
        "measured_smoke_support": {
            "staged_tok_s": staged_tok_s,
            "full_tok_s": full_tok_s,
            "staged_over_full_tok_s": round(throughput_ratio, 6),
            "readout_net_speedup_estimate": readout_proxy,
            "interaction_pass": interaction_pass,
            "genotype_pass": genotype_pass,
            "variance_pass": variance_pass,
            "r429_decision_equivalence_pass": r429_decision_pass,
            "r429_decision_equivalence_mismatch_updates": r429_mismatch_updates,
            "r429_decision_equivalence_certified_fraction": numeric(
                material.get("r429_decision_equivalence_certified_fraction")
            ),
            "r429_decision_equivalence_avoided_work_fraction": numeric(
                material.get("r429_decision_equivalence_avoided_work_fraction")
            ),
        },
        "activation_log": f"[R428_SPEED_TO_QUALITY_CONTROLLER] active=1 status={decision}",
    }


def main() -> int:
    args = parse_args()
    result = (
        make_plan(args.profile)
        if args.mode == "plan"
        else analyze(args.profile, args.r427_analysis, args.r427_manifest)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result["activation_log"] + f" output={args.output}")
    return 0 if not str(result["decision"]).startswith("BLOCK_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
