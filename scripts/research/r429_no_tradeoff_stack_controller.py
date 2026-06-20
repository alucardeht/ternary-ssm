#!/usr/bin/env python3
"""R429 no-tradeoff stack controller.

This controller turns the R429 five-aspect research synthesis into an executable
control layer over the existing R428/R427 integrated preset. It does not claim a
measured 7B speedup. It records which no-tradeoff mechanisms are currently
native-applied, which are represented by existing executable support, and what
projection is defensible from a tiny smoke.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "r429_no_tradeoff_stack_controller.v1"
BASELINE_DAYS = 495.0


ASPECTS: list[dict[str, Any]] = [
    {
        "id": "A",
        "name": "Objective Work Removal And CE Witnesses",
        "implementation_status": "planned_native_witness",
        "native_applied": False,
        "no_tradeoff_contract": "exact ternary update equivalence or full fallback",
        "current_support": [
            "macro CE quotient support from R427/R428",
            "decision-equivalence witness TODO",
        ],
        "conservative_speedup_range": [1.20, 1.50],
        "strong_speedup_range": [1.70, 2.50],
    },
    {
        "id": "B",
        "name": "Dynamics, Optimizer, And Stability-To-Speed",
        "implementation_status": "native_executable_stability_layer",
        "native_applied": True,
        "no_tradeoff_contract": "safe update damping and continuation LR controls preserve the same training target",
        "current_support": [
            "interaction-safe update",
            "variance-priced credit",
            "continuation LR override",
        ],
        "conservative_speedup_range": [1.20, 1.50],
        "strong_speedup_range": [1.60, 2.40],
    },
    {
        "id": "C",
        "name": "Progressive Growth, Width/Depth/State Scheduling, And Catch-Up",
        "implementation_status": "native_executable_growth_path",
        "native_applied": True,
        "no_tradeoff_contract": "smaller checkpoint lifts into full-width continuation without reset",
        "current_support": [
            "staged smaller v12 source",
            "checkpoint lift",
            "full-width continuation",
        ],
        "conservative_speedup_range": [1.20, 1.50],
        "strong_speedup_range": [1.70, 2.60],
    },
    {
        "id": "D",
        "name": "Training Material, Corpus Compilers, And Capability-Dense Streams",
        "implementation_status": "material_compiler_native_to_preset",
        "native_applied": True,
        "no_tradeoff_contract": "capability material must pass held-out material/control checks and keep residual settlement",
        "current_support": [
            "R427 capability material gate",
            "materialized uint16 training bin",
        ],
        "conservative_speedup_range": [1.15, 1.40],
        "strong_speedup_range": [1.60, 2.40],
    },
    {
        "id": "E",
        "name": "Memory Realization And Local Inference Artifact",
        "implementation_status": "artifact_shape_gate_native_to_preset",
        "native_applied": True,
        "no_tradeoff_contract": "representation-equivalent artifact, exact reconstruction, certified eviction, or fallback",
        "current_support": [
            "R427 local artifact shape gate",
            "projection ledger",
            "quotient head shape",
        ],
        "conservative_speedup_range": [1.20, 1.50],
        "strong_speedup_range": [1.80, 3.00],
    },
]


def aspects_with_native_a(native_a: bool) -> list[dict[str, Any]]:
    aspects = []
    for item in ASPECTS:
        row = dict(item)
        if row["id"] == "A" and native_a:
            row["implementation_status"] = "native_decision_equivalence_witness_applied"
            row["native_applied"] = True
            row["current_support"] = [
                "native --r429-decision-equivalence-witness",
                "exact zero-gradient ternary no-update certificates",
                "full fallback for every uncertified group",
                "native mismatch counter",
            ]
        aspects.append(row)
    return aspects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["plan", "analyze"], default="plan")
    parser.add_argument("--profile", choices=["interactive", "balanced", "exclusive"], default="interactive")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--r428-analysis", type=Path)
    parser.add_argument("--r428-manifest", type=Path)
    return parser.parse_args()


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


def days_for(speedup: float) -> float:
    return BASELINE_DAYS / speedup


def overlap_discounted_projection(native_a: bool) -> dict[str, Any]:
    """Return deliberately conservative stack projections.

    Aspect ranges overlap heavily, so this is not a product of aspect speedups.
    Native A decision-equivalence is the largest missing applied piece; without
    it, the stack projection is capped lower.
    """

    if native_a:
        conservative = [1.80, 2.60]
        strong = [2.80, 4.50]
    else:
        conservative = [1.35, 1.80]
        strong = [2.00, 3.00]
    return {
        "baseline_days": BASELINE_DAYS,
        "conservative_speedup": conservative,
        "conservative_days": [round(days_for(conservative[1]), 1), round(days_for(conservative[0]), 1)],
        "strong_speedup": strong,
        "strong_days": [round(days_for(strong[1]), 1), round(days_for(strong[0]), 1)],
        "independence_warning": "Aspect ranges overlap; these are capped stack projections, not multiplied factors.",
    }


def make_plan(profile: str) -> dict[str, Any]:
    native_a = bool(next(item for item in ASPECTS if item["id"] == "A")["native_applied"])
    return {
        "schema": SCHEMA,
        "decision": "PASS_R429_NO_TRADEOFF_STACK_PLAN",
        "profile": profile,
        "claim_scope": (
            "Executable no-tradeoff control stack over R428/R427. Analyze mode "
            "requires the native R429 decision-equivalence witness before the "
            "five-aspect stack can pass. No measured 7B speedup is claimed from "
            "plan mode."
        ),
        "no_tradeoff_definition": (
            "No lossy compression. Accepted mechanisms must preserve the relevant "
            "output, state, update decision, or reconstruct exactly; otherwise they "
            "must fall back to the full path."
        ),
        "aspects": aspects_with_native_a(native_a),
        "projection": overlap_discounted_projection(native_a),
        "activation_log": "[R429_NO_TRADEOFF_STACK] active=1 status=plan_ready",
    }


def analyze(profile: str, r428_analysis_path: Path | None, r428_manifest_path: Path | None) -> dict[str, Any]:
    plan = make_plan(profile)
    r428 = read_json(r428_analysis_path)
    manifest = read_json(r428_manifest_path)
    smoke = r428.get("measured_smoke_support", {})

    required = {
        "r428_ready": r428.get("decision") == "PASS_R428_SPEED_TO_QUALITY_READY",
        "manifest_available": bool(manifest),
        "no_tradeoff_contract_recorded": True,
        "aspect_a_native_decision_witness_live": int(numeric(smoke.get("r429_decision_equivalence_pass"))) == 1
        and int(numeric(smoke.get("r429_decision_equivalence_mismatch_updates"), 1.0)) == 0,
        "growth_path_live": numeric(smoke.get("staged_over_full_tok_s")) >= 1.10,
        "readout_or_objective_support_live": numeric(smoke.get("readout_net_speedup_estimate")) >= 1.10,
        "stability_updates_live": int(numeric(smoke.get("interaction_pass"))) == 1,
        "genotype_growth_live": int(numeric(smoke.get("genotype_pass"))) == 1,
        "credit_support_live": int(numeric(smoke.get("variance_pass"))) == 1,
    }
    decision = "PASS_R429_NO_TRADEOFF_STACK_SMOKE" if all(required.values()) else "BLOCK_R429_NO_TRADEOFF_STACK_SMOKE"

    native_a = bool(required["aspect_a_native_decision_witness_live"])
    return {
        **plan,
        "decision": decision,
        "aspects": aspects_with_native_a(native_a),
        "r428_analysis_path": str(r428_analysis_path) if r428_analysis_path else None,
        "r428_manifest_path": str(r428_manifest_path) if r428_manifest_path else None,
        "required_passes": required,
        "measured_smoke_support": smoke,
        "projection": overlap_discounted_projection(native_a),
        "native_gap": {},
        "native_completion": {
            "aspect_a_decision_equivalence_witness": "native_applied"
            if native_a
            else "missing_or_failed_native_counter",
            "meaning": "All five R429 aspects are active in the integrated preset when the required passes are true.",
        },
        "activation_log": f"[R429_NO_TRADEOFF_STACK] active=1 status={decision}",
    }


def main() -> int:
    args = parse_args()
    result = (
        make_plan(args.profile)
        if args.mode == "plan"
        else analyze(args.profile, args.r428_analysis, args.r428_manifest)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result["activation_log"] + f" output={args.output}")
    return 0 if not str(result["decision"]).startswith("BLOCK_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
