#!/usr/bin/env python3
"""Minimal evidence deferral self-test probe for R390 Slice 4."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


CONTROL_LABELS = ("answer", "retrieve", "ask_for_source", "defer")
ARMS = (
    "baseline_repaired_text",
    "baseline_post_evidence",
    "ablation_resolution_without_control",
    "candidate_control_episode",
)
ARM_COST_UNITS = {
    "baseline_repaired_text": 1.00,
    "baseline_post_evidence": 1.25,
    "ablation_resolution_without_control": 1.40,
    "candidate_control_episode": 1.50,
}
CONTROL_UTILITY_FLOOR = 1.10


@dataclass(frozen=True)
class FixtureRow:
    row_id: str
    prompt: str
    evidence: str
    control_label: str
    expected_resolution: str
    baseline_repaired_text: str
    baseline_repaired_text_resolution: str
    baseline_post_evidence: str
    baseline_post_evidence_resolution: str
    ablation_resolution_without_control: str
    ablation_resolution_without_control_resolution: str
    candidate_control_episode: str
    candidate_control_episode_resolution: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the built-in deterministic four-row fixture.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--seed", type=int, default=390, help="Deterministic seed.")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Slice 4 only supports --self-test.")
    return args


def build_self_test_fixture() -> list[FixtureRow]:
    return [
        FixtureRow(
            row_id="row-answer",
            prompt="What is 2 + 2?",
            evidence="No external evidence needed.",
            control_label="answer",
            expected_resolution="4",
            baseline_repaired_text="answer",
            baseline_repaired_text_resolution="4",
            baseline_post_evidence="retrieve",
            baseline_post_evidence_resolution="4",
            ablation_resolution_without_control="resolution_only",
            ablation_resolution_without_control_resolution="4",
            candidate_control_episode="control=answer",
            candidate_control_episode_resolution="4",
        ),
        FixtureRow(
            row_id="row-retrieve",
            prompt="What does the policy manual say about expense approvals?",
            evidence="The policy manual exists but is not quoted here.",
            control_label="retrieve",
            expected_resolution="Approvals require the policy manual excerpt before answering.",
            baseline_repaired_text="answer",
            baseline_repaired_text_resolution="Approvals are manager-only under all circumstances.",
            baseline_post_evidence="ask_for_source",
            baseline_post_evidence_resolution="Approvals require the policy manual excerpt before answering.",
            ablation_resolution_without_control="resolution_only",
            ablation_resolution_without_control_resolution="Approvals require the policy manual excerpt before answering.",
            candidate_control_episode="control=retrieve",
            candidate_control_episode_resolution="Approvals require the policy manual excerpt before answering.",
        ),
        FixtureRow(
            row_id="row-ask-for-source",
            prompt="Summarize the claim in the report I mentioned earlier.",
            evidence="No report excerpt is present in the conversation.",
            control_label="ask_for_source",
            expected_resolution="Please provide the report excerpt before I summarize the claim.",
            baseline_repaired_text="defer",
            baseline_repaired_text_resolution="I cannot summarize the claim from a missing report excerpt.",
            baseline_post_evidence="answer",
            baseline_post_evidence_resolution="The report says profits doubled year over year.",
            ablation_resolution_without_control="resolution_only",
            ablation_resolution_without_control_resolution="Please provide the report excerpt before I summarize the claim.",
            candidate_control_episode="control=ask_for_source",
            candidate_control_episode_resolution="Please provide the report excerpt before I summarize the claim.",
        ),
        FixtureRow(
            row_id="row-defer",
            prompt="Decide whether the contract is legally enforceable in Brazil.",
            evidence="The contract text and governing-law clause are unavailable.",
            control_label="defer",
            expected_resolution="I cannot determine enforceability without the contract text and governing-law clause.",
            baseline_repaired_text="retrieve",
            baseline_repaired_text_resolution="I cannot determine enforceability without the contract text and governing-law clause.",
            baseline_post_evidence="answer",
            baseline_post_evidence_resolution="The contract is definitely enforceable in Brazil.",
            ablation_resolution_without_control="resolution_only",
            ablation_resolution_without_control_resolution="I cannot determine enforceability without the contract text and governing-law clause.",
            candidate_control_episode="control=defer",
            candidate_control_episode_resolution="I cannot determine enforceability without the contract text and governing-law clause.",
        ),
    ]


def decode_control(text: str) -> str | None:
    normalized = text.strip().lower()
    for label in CONTROL_LABELS:
        if normalized == label or f"control={label}" in normalized:
            return label
    return None


def control_accuracy(rows: list[FixtureRow], arm_name: str) -> float:
    correct = 0
    total = 0
    for row in rows:
        emitted = decode_control(getattr(row, arm_name))
        total += 1
        if emitted == row.control_label:
            correct += 1
    if total == 0:
        return math.nan
    return correct / total


def normalize_resolution(text: str) -> str:
    return text.strip().lower()


def resolution_quality(rows: list[FixtureRow], arm_name: str) -> float:
    correct = 0
    total = 0
    for row in rows:
        emitted = normalize_resolution(getattr(row, arm_name))
        expected = normalize_resolution(row.expected_resolution)
        total += 1
        if emitted == expected:
            correct += 1
    if total == 0:
        return math.nan
    return correct / total


def evaluate(rows: list[FixtureRow]) -> dict[str, object]:
    arm_metrics = {}
    for arm_name in ARMS:
        accuracy = control_accuracy(rows, arm_name)
        resolution_arm_name = f"{arm_name}_resolution"
        resolution = resolution_quality(rows, resolution_arm_name)
        episode_cost_units = ARM_COST_UNITS[arm_name]
        utility = (
            accuracy / episode_cost_units
            if math.isfinite(accuracy) and math.isfinite(episode_cost_units) and episode_cost_units > 0.0
            else math.nan
        )
        arm_metrics[arm_name] = {
            "control_accuracy": accuracy,
            "resolution_quality": resolution,
            "episode_cost_units": episode_cost_units,
            "cost_normalized_control_utility": utility,
        }

    comparator_arms = [arm_name for arm_name in ARMS if arm_name != "candidate_control_episode"]
    finite_comparator_utilities = [
        (
            arm_name,
            arm_metrics[arm_name]["cost_normalized_control_utility"],
        )
        for arm_name in comparator_arms
        if math.isfinite(arm_metrics[arm_name]["cost_normalized_control_utility"])
    ]
    strongest_comparator_arm = (
        max(finite_comparator_utilities, key=lambda item: item[1])[0]
        if finite_comparator_utilities
        else None
    )
    strongest_comparator_metrics = (
        arm_metrics[strongest_comparator_arm]
        if strongest_comparator_arm is not None
        else None
    )
    best_comparator_control_accuracy = (
        strongest_comparator_metrics["control_accuracy"]
        if strongest_comparator_metrics is not None
        else math.nan
    )
    best_comparator_cost_normalized_control_utility = (
        strongest_comparator_metrics["cost_normalized_control_utility"]
        if strongest_comparator_metrics is not None
        else math.nan
    )
    best_comparator_resolution_quality = (
        strongest_comparator_metrics["resolution_quality"]
        if strongest_comparator_metrics is not None
        else math.nan
    )

    baseline_scores = [
        arm_metrics["baseline_repaired_text"]["control_accuracy"],
        arm_metrics["baseline_post_evidence"]["control_accuracy"],
        arm_metrics["ablation_resolution_without_control"]["control_accuracy"],
    ]
    finite_baseline_scores = [score for score in baseline_scores if math.isfinite(score)]
    best_baseline = max(finite_baseline_scores) if finite_baseline_scores else math.nan
    candidate = arm_metrics["candidate_control_episode"]["control_accuracy"]
    delta = (
        candidate - best_comparator_control_accuracy
        if math.isfinite(candidate) and math.isfinite(best_comparator_control_accuracy)
        else math.nan
    )

    baseline_utilities = [
        arm_metrics["baseline_repaired_text"]["cost_normalized_control_utility"],
        arm_metrics["baseline_post_evidence"]["cost_normalized_control_utility"],
        arm_metrics["ablation_resolution_without_control"]["cost_normalized_control_utility"],
    ]
    finite_baseline_utilities = [utility for utility in baseline_utilities if math.isfinite(utility)]
    best_baseline_utility = max(finite_baseline_utilities) if finite_baseline_utilities else math.nan
    candidate_utility = arm_metrics["candidate_control_episode"]["cost_normalized_control_utility"]
    utility_ratio = (
        candidate_utility / best_comparator_cost_normalized_control_utility
        if math.isfinite(candidate_utility)
        and math.isfinite(best_comparator_cost_normalized_control_utility)
        and best_comparator_cost_normalized_control_utility > 0.0
        else math.nan
    )
    utility_delta = (
        candidate_utility - best_comparator_cost_normalized_control_utility
        if math.isfinite(candidate_utility)
        and math.isfinite(best_comparator_cost_normalized_control_utility)
        else math.nan
    )

    baseline_resolution_scores = [
        arm_metrics["baseline_repaired_text"]["resolution_quality"],
        arm_metrics["baseline_post_evidence"]["resolution_quality"],
        arm_metrics["ablation_resolution_without_control"]["resolution_quality"],
    ]
    finite_baseline_resolution_scores = [
        score for score in baseline_resolution_scores if math.isfinite(score)
    ]
    best_baseline_resolution_quality = (
        max(finite_baseline_resolution_scores)
        if finite_baseline_resolution_scores
        else math.nan
    )
    candidate_resolution_quality = arm_metrics["candidate_control_episode"]["resolution_quality"]
    resolution_quality_delta = (
        candidate_resolution_quality - best_comparator_resolution_quality
        if math.isfinite(candidate_resolution_quality)
        and math.isfinite(best_comparator_resolution_quality)
        else math.nan
    )
    resolution_quality_guard_passed = (
        resolution_quality_delta >= 0.0
        if math.isfinite(resolution_quality_delta)
        else False
    )
    return {
        "arm_metrics": arm_metrics,
        "best_baseline_control_accuracy": best_baseline,
        "strongest_comparator_arm": strongest_comparator_arm,
        "best_comparator_control_accuracy": best_comparator_control_accuracy,
        "candidate_control_accuracy": candidate,
        "control_accuracy_delta": delta,
        "best_baseline_cost_normalized_control_utility": best_baseline_utility,
        "best_comparator_cost_normalized_control_utility": best_comparator_cost_normalized_control_utility,
        "candidate_cost_normalized_control_utility": candidate_utility,
        "cost_normalized_control_utility_ratio": utility_ratio,
        "cost_normalized_control_utility_delta": utility_delta,
        "control_utility_floor": CONTROL_UTILITY_FLOOR,
        "control_utility_threshold_passed": utility_ratio >= CONTROL_UTILITY_FLOOR if math.isfinite(utility_ratio) else False,
        "best_baseline_resolution_quality": best_baseline_resolution_quality,
        "best_comparator_resolution_quality": best_comparator_resolution_quality,
        "candidate_resolution_quality": candidate_resolution_quality,
        "resolution_quality_delta": resolution_quality_delta,
        "resolution_quality_guard_passed": resolution_quality_guard_passed,
    }


def compute_validity(rows: list[FixtureRow], metrics: dict[str, object]) -> dict[str, object]:
    counts = Counter(row.control_label for row in rows)
    all_labels_have_support = all(counts.get(label, 0) > 0 for label in CONTROL_LABELS)
    ablation_present = "ablation_resolution_without_control" in metrics["arm_metrics"]
    candidate_finite = math.isfinite(metrics["candidate_control_accuracy"])
    baseline_finite = any(
        math.isfinite(metrics["arm_metrics"][baseline]["control_accuracy"])
        for baseline in (
            "baseline_repaired_text",
            "baseline_post_evidence",
            "ablation_resolution_without_control",
        )
    )
    delta_finite = math.isfinite(metrics["control_accuracy_delta"])
    arm_costs_positive_and_finite = all(
        math.isfinite(metrics["arm_metrics"][arm_name]["episode_cost_units"])
        and metrics["arm_metrics"][arm_name]["episode_cost_units"] > 0.0
        for arm_name in ARMS
    )
    ablation_cost_positive_and_finite = (
        math.isfinite(metrics["arm_metrics"]["ablation_resolution_without_control"]["episode_cost_units"])
        and metrics["arm_metrics"]["ablation_resolution_without_control"]["episode_cost_units"] > 0.0
    )
    candidate_utility_finite = math.isfinite(metrics["candidate_cost_normalized_control_utility"])
    best_baseline_utility = metrics["best_baseline_cost_normalized_control_utility"]
    best_baseline_utility_finite_and_positive = math.isfinite(best_baseline_utility) and best_baseline_utility > 0.0
    best_comparator_utility = metrics["best_comparator_cost_normalized_control_utility"]
    best_comparator_utility_finite_and_positive = (
        math.isfinite(best_comparator_utility) and best_comparator_utility > 0.0
    )
    utility_ratio_finite = math.isfinite(metrics["cost_normalized_control_utility_ratio"])
    utility_delta_finite = math.isfinite(metrics["cost_normalized_control_utility_delta"])
    candidate_resolution_quality_finite = math.isfinite(metrics["candidate_resolution_quality"])
    baseline_resolution_quality_finite = any(
        math.isfinite(metrics["arm_metrics"][baseline]["resolution_quality"])
        for baseline in (
            "baseline_repaired_text",
            "baseline_post_evidence",
            "ablation_resolution_without_control",
        )
    )
    best_baseline_resolution_quality_finite = math.isfinite(metrics["best_baseline_resolution_quality"])
    best_comparator_resolution_quality_finite = math.isfinite(metrics["best_comparator_resolution_quality"])
    resolution_quality_delta_finite = math.isfinite(metrics["resolution_quality_delta"])
    ablation_resolution_quality_finite = math.isfinite(
        metrics["arm_metrics"]["ablation_resolution_without_control"]["resolution_quality"]
    )
    ablation_hides_control_label = all(
        decode_control(row.ablation_resolution_without_control) != row.control_label
        and row.control_label not in row.ablation_resolution_without_control.strip().lower()
        for row in rows
    )
    strongest_comparator_present = isinstance(metrics["strongest_comparator_arm"], str)
    resolution_quality_guard_passed = metrics["resolution_quality_guard_passed"] is True
    control_utility_threshold_passed = metrics["control_utility_threshold_passed"] is True
    status = "valid" if all([
        all_labels_have_support,
        ablation_present,
        candidate_finite,
        baseline_finite,
        delta_finite,
        arm_costs_positive_and_finite,
        ablation_cost_positive_and_finite,
        candidate_utility_finite,
        best_baseline_utility_finite_and_positive,
        best_comparator_utility_finite_and_positive,
        utility_ratio_finite,
        utility_delta_finite,
        candidate_resolution_quality_finite,
        baseline_resolution_quality_finite,
        best_baseline_resolution_quality_finite,
        best_comparator_resolution_quality_finite,
        resolution_quality_delta_finite,
        ablation_resolution_quality_finite,
        ablation_hides_control_label,
        strongest_comparator_present,
        control_utility_threshold_passed,
        resolution_quality_guard_passed,
    ]) else "blocked_invalid_probe"
    return {
        "status": status,
        "all_four_control_labels_have_support": all_labels_have_support,
        "ablation_resolution_without_control_present": ablation_present,
        "candidate_metric_finite": candidate_finite,
        "at_least_one_baseline_metric_finite": baseline_finite,
        "control_accuracy_delta_finite": delta_finite,
        "arm_costs_positive_and_finite": arm_costs_positive_and_finite,
        "ablation_cost_positive_and_finite": ablation_cost_positive_and_finite,
        "candidate_cost_normalized_control_utility_finite": candidate_utility_finite,
        "best_baseline_cost_normalized_control_utility_finite_and_positive": best_baseline_utility_finite_and_positive,
        "best_comparator_cost_normalized_control_utility_finite_and_positive": best_comparator_utility_finite_and_positive,
        "cost_normalized_control_utility_ratio_finite": utility_ratio_finite,
        "cost_normalized_control_utility_delta_finite": utility_delta_finite,
        "candidate_resolution_quality_finite": candidate_resolution_quality_finite,
        "at_least_one_baseline_resolution_quality_finite": baseline_resolution_quality_finite,
        "best_baseline_resolution_quality_finite": best_baseline_resolution_quality_finite,
        "best_comparator_resolution_quality_finite": best_comparator_resolution_quality_finite,
        "resolution_quality_delta_finite": resolution_quality_delta_finite,
        "ablation_resolution_quality_finite": ablation_resolution_quality_finite,
        "ablation_hides_control_label": ablation_hides_control_label,
        "strongest_comparator_present": strongest_comparator_present,
        "control_utility_threshold_passed": control_utility_threshold_passed,
        "resolution_quality_guard_passed": resolution_quality_guard_passed,
        "fixture_label_counts": {label: counts.get(label, 0) for label in CONTROL_LABELS},
    }


def compute_decision(validity: dict[str, object]) -> str:
    return "slice4_ablation_guard_valid" if validity["status"] == "valid" else "blocked_invalid_probe"


def write_manifest(
    out_dir: Path,
    seed: int,
    rows: list[FixtureRow],
    metrics: dict[str, object],
    validity: dict[str, object],
    decision: str,
) -> Path:
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "probe": "r390_evidence_deferral",
        "mode": "self_test",
        "seed": seed,
        "fixture_row_count": len(rows),
        "fixture_rows": [asdict(row) for row in rows],
        "arms": list(ARMS),
        "arm_cost_units": ARM_COST_UNITS,
        "metrics": metrics,
        "validity": validity,
        "decision": decision,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    rows = build_self_test_fixture()
    metrics = evaluate(rows)
    validity = compute_validity(rows, metrics)
    decision = compute_decision(validity)

    args.out.mkdir(parents=True, exist_ok=True)
    write_manifest(args.out, args.seed, rows, metrics, validity, decision)

    print(
        "R390_EVIDENCE_DEFERRAL_ACTIVATION"
        f" mode=self_test seed={args.seed} rows={len(rows)} arms={','.join(ARMS)}"
    )
    print(
        "R390_EVIDENCE_DEFERRAL_VALIDITY"
        f" status={validity['status']}"
        f" all_four_control_labels_have_support={str(validity['all_four_control_labels_have_support']).lower()}"
        f" candidate_metric_finite={str(validity['candidate_metric_finite']).lower()}"
        f" at_least_one_baseline_metric_finite={str(validity['at_least_one_baseline_metric_finite']).lower()}"
        f" ablation_resolution_without_control_present={str(validity['ablation_resolution_without_control_present']).lower()}"
        f" control_accuracy_delta_finite={str(validity['control_accuracy_delta_finite']).lower()}"
        f" arm_costs_positive_and_finite={str(validity['arm_costs_positive_and_finite']).lower()}"
        f" ablation_cost_positive_and_finite={str(validity['ablation_cost_positive_and_finite']).lower()}"
        f" candidate_cost_normalized_control_utility_finite={str(validity['candidate_cost_normalized_control_utility_finite']).lower()}"
        f" best_baseline_cost_normalized_control_utility_finite_and_positive={str(validity['best_baseline_cost_normalized_control_utility_finite_and_positive']).lower()}"
        f" best_comparator_cost_normalized_control_utility_finite_and_positive={str(validity['best_comparator_cost_normalized_control_utility_finite_and_positive']).lower()}"
        f" cost_normalized_control_utility_ratio_finite={str(validity['cost_normalized_control_utility_ratio_finite']).lower()}"
        f" cost_normalized_control_utility_delta_finite={str(validity['cost_normalized_control_utility_delta_finite']).lower()}"
        f" candidate_resolution_quality_finite={str(validity['candidate_resolution_quality_finite']).lower()}"
        f" at_least_one_baseline_resolution_quality_finite={str(validity['at_least_one_baseline_resolution_quality_finite']).lower()}"
        f" best_baseline_resolution_quality_finite={str(validity['best_baseline_resolution_quality_finite']).lower()}"
        f" best_comparator_resolution_quality_finite={str(validity['best_comparator_resolution_quality_finite']).lower()}"
        f" resolution_quality_delta_finite={str(validity['resolution_quality_delta_finite']).lower()}"
        f" ablation_resolution_quality_finite={str(validity['ablation_resolution_quality_finite']).lower()}"
        f" ablation_hides_control_label={str(validity['ablation_hides_control_label']).lower()}"
        f" strongest_comparator_present={str(validity['strongest_comparator_present']).lower()}"
        f" control_utility_threshold_passed={str(validity['control_utility_threshold_passed']).lower()}"
        f" resolution_quality_guard_passed={str(validity['resolution_quality_guard_passed']).lower()}"
    )
    print(
        "R390_EVIDENCE_DEFERRAL_DECISION"
        f" decision={decision}"
        f" ablation_resolution_without_control_present={str(validity['ablation_resolution_without_control_present']).lower()}"
        f" strongest_comparator_arm={metrics['strongest_comparator_arm']}"
        f" control_accuracy_delta={metrics['control_accuracy_delta']:.6f}"
        f" cost_normalized_control_utility_ratio={metrics['cost_normalized_control_utility_ratio']:.6f}"
        f" resolution_quality_delta={metrics['resolution_quality_delta']:.6f}"
        f" control_utility_floor={metrics['control_utility_floor']:.2f}"
        f" control_utility_threshold_passed={str(metrics['control_utility_threshold_passed']).lower()}"
        f" resolution_quality_guard_passed={str(metrics['resolution_quality_guard_passed']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
