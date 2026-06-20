#!/usr/bin/env python3
"""Offline Capability-Deficit Graph Tutoring diagnostic probe for R390."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REQUIRED_FIELDS = (
    "episode_id",
    "capabilities",
    "cost",
    "estimate_pass",
    "heldout_retest_pass",
    "post_teach_retest_pass",
    "loss_score",
    "uncertainty_score",
)

CAUSAL_FIELDS = ("target_after_prereq_pass", "target_direct_pass")
ACTION_CAPABILITIES = {"answer", "retrieve", "ask", "defer"}
EPSILON = 1e-12


@dataclass
class CapabilitySummary:
    name: str
    m_hat: float
    deficit: float
    uncertainty: float
    prerequisite_closed_deficit: float
    estimator_count: int
    heldout_count: int
    retest_pass_rate: Optional[float]
    post_teach_pass_rate: Optional[float]
    improvement_rate: float
    source_count: int
    direct_prereqs: List[str]
    all_prereqs: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=Path, help="JSONL offline teaching observations.")
    mode.add_argument(
        "--self-test",
        action="store_true",
        help="Use the built-in deterministic synthetic fixture.",
    )
    mode.add_argument(
        "--corrected-self-test",
        action="store_true",
        help="Use the deterministic synthetic fixture with a finite positive comparator.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--lambda-prereq",
        type=float,
        default=0.5,
        help="Weight applied to prerequisite deficits when closing a capability deficit.",
    )
    parser.add_argument(
        "--retest-overhead",
        type=float,
        default=0.10,
        help="Fractional retest overhead for the adjusted routing advantage.",
    )
    parser.add_argument("--seed", type=int, default=390, help="Deterministic seed.")
    return parser.parse_args()


def _coerce_binary(value: object, field: str, episode_id: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    raise ValueError(f"{episode_id}: field {field!r} must be binary 0/1.")


def load_jsonl_records(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object.")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no JSONL records found.")
    return records


def build_self_test_fixture(seed: int) -> List[dict]:
    rng = random.Random(seed)
    layout = [
        ("synthesis", ["evidence_selection", "contradiction_resolution"], 0.96, 0.73, 0, 0, 0, 1, 0),
        ("synthesis", ["evidence_selection", "contradiction_resolution"], 0.94, 0.71, 0, 0, 0, 1, 0),
        ("synthesis", ["evidence_selection", "contradiction_resolution"], 0.92, 0.69, 0, 0, 0, 1, 0),
        ("evidence_selection", ["source_identification"], 0.91, 0.66, 0, 0, 0, 1, 0),
        ("evidence_selection", ["source_identification"], 0.89, 0.63, 0, 0, 0, 1, 0),
        ("contradiction_resolution", ["source_identification"], 0.88, 0.62, 0, 0, 0, 1, 0),
        ("source_identification", [], 0.68, 0.58, 0, 0, 1, None, None),
        ("source_identification", [], 0.66, 0.55, 0, 0, 1, None, None),
        ("source_identification", [], 0.64, 0.53, 0, 1, 1, None, None),
        ("source_identification", [], 0.61, 0.50, 1, 1, 1, None, None),
        ("citation_grounding", ["source_identification"], 0.59, 0.45, 1, 1, 1, 1, 1),
        ("citation_grounding", ["source_identification"], 0.57, 0.42, 1, 1, 1, 1, 1),
    ]
    records: List[dict] = []
    for index, (
        capability,
        prerequisites,
        loss_score,
        uncertainty_score,
        estimate_pass,
        heldout_retest_pass,
        post_teach_retest_pass,
        after_prereq,
        direct_pass,
    ) in enumerate(layout, start=1):
        jitter = (rng.random() - 0.5) * 0.02
        record = {
            "episode_id": f"selftest-{index:03d}",
            "baseline_rank": index,
            "loss_score": round(loss_score + jitter, 6),
            "uncertainty_score": round(uncertainty_score - jitter, 6),
            "capabilities": [capability],
            "prerequisites": prerequisites,
            "cost": 1.0,
            "estimate_pass": estimate_pass,
            "heldout_retest_pass": heldout_retest_pass,
            "post_teach_retest_pass": post_teach_retest_pass,
        }
        if after_prereq is not None:
            record["target_after_prereq_pass"] = after_prereq
        if direct_pass is not None:
            record["target_direct_pass"] = direct_pass
        records.append(record)
    return records


def build_corrected_self_test_fixture(seed: int) -> List[dict]:
    rng = random.Random(seed)
    layout = [
        ("synthesis", ["evidence_selection", "contradiction_resolution"], 0.96, 0.73, 0, 0, 0, 1, 0),
        ("synthesis", ["evidence_selection", "contradiction_resolution"], 0.94, 0.71, 0, 0, 0, 1, 0),
        ("evidence_selection", ["source_identification"], 0.91, 0.66, 0, 0, 0, 1, 0),
        ("contradiction_resolution", ["source_identification"], 0.89, 0.63, 0, 0, 0, 1, 0),
        ("source_identification", [], 0.68, 0.58, 0, 0, 1, None, None),
        ("synthesis", ["evidence_selection", "contradiction_resolution"], 0.92, 0.69, 0, 0, 0, 1, 0),
        ("evidence_selection", ["source_identification"], 0.88, 0.62, 0, 0, 0, 1, 0),
        ("source_identification", [], 0.66, 0.55, 0, 0, 1, None, None),
        ("source_identification", [], 0.64, 0.53, 0, 1, 1, None, None),
        ("source_identification", [], 0.61, 0.50, 1, 1, 1, None, None),
        ("citation_grounding", ["source_identification"], 0.59, 0.45, 1, 1, 1, 1, 1),
        ("citation_grounding", ["source_identification"], 0.57, 0.42, 1, 1, 1, 1, 1),
    ]
    records: List[dict] = []
    for index, (
        capability,
        prerequisites,
        loss_score,
        uncertainty_score,
        estimate_pass,
        heldout_retest_pass,
        post_teach_retest_pass,
        after_prereq,
        direct_pass,
    ) in enumerate(layout, start=1):
        jitter = (rng.random() - 0.5) * 0.02
        record = {
            "episode_id": f"corrected-selftest-{index:03d}",
            "baseline_rank": index,
            "loss_score": round(loss_score + jitter, 6),
            "uncertainty_score": round(uncertainty_score - jitter, 6),
            "capabilities": [capability],
            "prerequisites": prerequisites,
            "cost": 1.0,
            "estimate_pass": estimate_pass,
            "heldout_retest_pass": heldout_retest_pass,
            "post_teach_retest_pass": post_teach_retest_pass,
        }
        if after_prereq is not None:
            record["target_after_prereq_pass"] = after_prereq
        if direct_pass is not None:
            record["target_direct_pass"] = direct_pass
        records.append(record)
    return records


def prepare_records(raw_records: Sequence[dict]) -> List[dict]:
    prepared: List[dict] = []
    for input_order, raw in enumerate(raw_records, start=1):
        for field in REQUIRED_FIELDS:
            if field not in raw:
                raise ValueError(f"Missing required field {field!r} in record {raw!r}.")
        episode_id = str(raw["episode_id"])
        capabilities = raw["capabilities"]
        if not isinstance(capabilities, list) or not capabilities:
            raise ValueError(f"{episode_id}: capabilities must be a non-empty list.")
        normalized_capabilities = [str(item) for item in capabilities]
        prerequisites = [str(item) for item in raw.get("prerequisites", [])]
        cost = float(raw["cost"])
        if cost <= 0.0:
            raise ValueError(f"{episode_id}: cost must be positive.")
        baseline_rank = int(raw.get("baseline_rank", input_order))
        record = {
            "episode_id": episode_id,
            "baseline_rank": baseline_rank,
            "input_order": input_order,
            "loss_score": float(raw["loss_score"]),
            "uncertainty_score": float(raw["uncertainty_score"]),
            "capabilities": normalized_capabilities,
            "prerequisites": prerequisites,
            "cost": cost,
            "estimate_pass": _coerce_binary(raw["estimate_pass"], "estimate_pass", episode_id),
            "heldout_retest_pass": _coerce_binary(
                raw["heldout_retest_pass"], "heldout_retest_pass", episode_id
            ),
            "post_teach_retest_pass": _coerce_binary(
                raw["post_teach_retest_pass"], "post_teach_retest_pass", episode_id
            ),
        }
        for field in CAUSAL_FIELDS:
            if field in raw and raw[field] is not None:
                record[field] = _coerce_binary(raw[field], field, episode_id)
        prepared.append(record)
    return prepared


def average(values: Iterable[float]) -> Optional[float]:
    materialized = list(values)
    if not materialized:
        return None
    return sum(materialized) / len(materialized)


def build_prereq_map(records: Sequence[dict]) -> Dict[str, List[str]]:
    prereq_map: Dict[str, List[str]] = {}
    for record in records:
        direct_prereqs = list(dict.fromkeys(record["prerequisites"]))
        for capability in record["capabilities"]:
            existing = prereq_map.get(capability)
            if existing is None:
                prereq_map[capability] = direct_prereqs
            else:
                merged = list(dict.fromkeys(existing + direct_prereqs))
                prereq_map[capability] = merged
        for prereq in direct_prereqs:
            prereq_map.setdefault(prereq, [])
    return prereq_map


def closure_for_capability(
    capability: str, prereq_map: Dict[str, List[str]], visiting: Optional[set] = None
) -> List[str]:
    visiting = set() if visiting is None else set(visiting)
    if capability in visiting:
        return []
    visiting.add(capability)
    closed: List[str] = []
    for prereq in prereq_map.get(capability, []):
        if prereq not in closed:
            closed.append(prereq)
        for nested in closure_for_capability(prereq, prereq_map, visiting):
            if nested not in closed:
                closed.append(nested)
    return closed


def summarize_capabilities(
    records: Sequence[dict], lambda_prereq: float
) -> Tuple[Dict[str, CapabilitySummary], Dict[str, List[str]], List[dict]]:
    prereq_map = build_prereq_map(records)
    stats = defaultdict(
        lambda: {
            "estimate_pass": [],
            "heldout_retest_pass": [],
            "post_teach_retest_pass": [],
            "uncertainty_score": [],
            "source_count": 0,
        }
    )
    edges: Dict[Tuple[str, str], Dict[str, object]] = {}
    for record in records:
        for capability in record["capabilities"]:
            bucket = stats[capability]
            bucket["estimate_pass"].append(record["estimate_pass"])
            bucket["heldout_retest_pass"].append(record["heldout_retest_pass"])
            bucket["post_teach_retest_pass"].append(record["post_teach_retest_pass"])
            bucket["uncertainty_score"].append(record["uncertainty_score"])
            bucket["source_count"] += 1
            for prereq in prereq_map.get(capability, []):
                key = (prereq, capability)
                edge = edges.setdefault(
                    key,
                    {
                        "from": prereq,
                        "to": capability,
                        "count": 0,
                        "causal_after_sum": 0,
                        "causal_after_count": 0,
                    },
                )
                edge["count"] = int(edge["count"]) + 1
                if "target_after_prereq_pass" in record:
                    edge["causal_after_sum"] = int(edge["causal_after_sum"]) + int(
                        record["target_after_prereq_pass"]
                    )
                    edge["causal_after_count"] = int(edge["causal_after_count"]) + 1
    summaries: Dict[str, CapabilitySummary] = {}
    closure_map = {
        capability: closure_for_capability(capability, prereq_map) for capability in prereq_map
    }
    raw_deficits: Dict[str, float] = {}
    for capability, bucket in stats.items():
        m_hat = average(bucket["estimate_pass"]) or 0.0
        raw_deficits[capability] = max(0.0, 1.0 - m_hat)
    for capability in sorted(prereq_map):
        bucket = stats[capability]
        m_hat = average(bucket["estimate_pass"]) or 0.0
        heldout_pass_rate = average(bucket["heldout_retest_pass"])
        post_teach_pass_rate = average(bucket["post_teach_retest_pass"])
        uncertainty = average(bucket["uncertainty_score"]) or 0.0
        deficit = max(0.0, 1.0 - m_hat)
        prereq_closed_deficit = deficit + lambda_prereq * sum(
            raw_deficits.get(prereq, 0.0) for prereq in closure_map.get(capability, [])
        )
        improvement_rate = max(0.0, (post_teach_pass_rate or 0.0) - (heldout_pass_rate or 0.0))
        summaries[capability] = CapabilitySummary(
            name=capability,
            m_hat=m_hat,
            deficit=deficit,
            uncertainty=uncertainty,
            prerequisite_closed_deficit=prereq_closed_deficit,
            estimator_count=len(bucket["estimate_pass"]),
            heldout_count=len(bucket["heldout_retest_pass"]),
            retest_pass_rate=heldout_pass_rate,
            post_teach_pass_rate=post_teach_pass_rate,
            improvement_rate=improvement_rate,
            source_count=int(bucket["source_count"]),
            direct_prereqs=prereq_map.get(capability, []),
            all_prereqs=closure_map.get(capability, []),
        )
    edge_rows: List[dict] = []
    for edge_key in sorted(edges):
        edge = edges[edge_key]
        count = int(edge["count"])
        causal_count = int(edge["causal_after_count"])
        mean_after = (
            float(edge["causal_after_sum"]) / causal_count if causal_count else None
        )
        edge_rows.append(
            {
                "from": edge["from"],
                "to": edge["to"],
                "weight": count / max(1, summaries[edge["to"]].source_count),
                "source_count": count,
                "mean_target_after_prereq_pass": mean_after,
            }
        )
    return summaries, prereq_map, edge_rows


def ordinal_ranks(
    records: Sequence[dict], key_name: str, reverse: bool
) -> Dict[str, int]:
    ordered = sorted(
        records,
        key=lambda row: (
            -float(row[key_name]) if reverse else float(row[key_name]),
            row["baseline_rank"],
            row["episode_id"],
        ),
    )
    return {record["episode_id"]: index for index, record in enumerate(ordered, start=1)}


def choose_capability(record: dict, summaries: Dict[str, CapabilitySummary]) -> str:
    return max(
        record["capabilities"],
        key=lambda capability: (
            summaries[capability].prerequisite_closed_deficit,
            summaries[capability].deficit,
            -summaries[capability].m_hat,
            capability,
        ),
    )


def compute_episode_rows(
    records: Sequence[dict], summaries: Dict[str, CapabilitySummary]
) -> Tuple[List[dict], float]:
    loss_ranks = ordinal_ranks(records, "loss_score", reverse=True)
    uncertainty_ranks = ordinal_ranks(records, "uncertainty_score", reverse=True)
    episode_rows: List[dict] = []
    for record in records:
        selected_capability = choose_capability(record, summaries)
        summary = summaries[selected_capability]
        blocked = any(
            summaries[prereq].deficit >= 0.5 for prereq in summary.direct_prereqs
        )
        mastered = summary.m_hat >= 0.5
        irrelevant = record["heldout_retest_pass"] == 1
        expected_reduction = max(
            0.0, float(record["post_teach_retest_pass"] - record["heldout_retest_pass"])
        )
        gain = (summary.prerequisite_closed_deficit * expected_reduction) / record["cost"]
        episode_rows.append(
            {
                "episode_id": record["episode_id"],
                "baseline_rank": record["baseline_rank"],
                "loss_score": record["loss_score"],
                "uncertainty_score": record["uncertainty_score"],
                "selected_capability": selected_capability,
                "prerequisite_closed_score": summary.prerequisite_closed_deficit,
                "expected_deficit_reduction_per_cost": gain,
                "loss_rank": loss_ranks[record["episode_id"]],
                "uncertainty_rank": uncertainty_ranks[record["episode_id"]],
                "blocked": blocked,
                "mastered": mastered,
                "irrelevant": irrelevant,
                "capabilities": list(record["capabilities"]),
                "direct_prerequisites": list(summary.direct_prereqs),
                "cost": record["cost"],
            }
        )
    ordered = sorted(
        episode_rows,
        key=lambda row: (
            -row["expected_deficit_reduction_per_cost"],
            -row["prerequisite_closed_score"],
            row["baseline_rank"],
            row["episode_id"],
        ),
    )
    cdgt_ranks = {
        row["episode_id"]: index for index, row in enumerate(ordered, start=1)
    }
    for row in episode_rows:
        row["cdgt_rank"] = cdgt_ranks[row["episode_id"]]
    selection_budget = float(max(1, math.ceil(len(episode_rows) * 0.5)))
    return episode_rows, selection_budget


def mean_gain(rows: Sequence[dict]) -> float:
    if not rows:
        return 0.0
    return sum(row["expected_deficit_reduction_per_cost"] for row in rows) / len(rows)


def compute_auc(scores_and_labels: Sequence[Tuple[float, int]]) -> Optional[float]:
    if not scores_and_labels:
        return None
    positives = sum(label for _, label in scores_and_labels)
    negatives = len(scores_and_labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(scores_and_labels, key=lambda pair: pair[0])
    ranks: List[float] = [0.0] * len(ordered)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for slot in range(index, end):
            ranks[slot] = average_rank
        index = end
    positive_rank_sum = sum(rank for rank, (_, label) in zip(ranks, ordered) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def spearman(rank_a: Dict[str, int], rank_b: Dict[str, int]) -> Optional[float]:
    keys = sorted(set(rank_a) & set(rank_b))
    if len(keys) < 2:
        return None
    n = len(keys)
    squared_delta = sum((rank_a[key] - rank_b[key]) ** 2 for key in keys)
    return 1.0 - (6.0 * squared_delta) / (n * (n * n - 1.0))


def safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if abs(denominator) <= EPSILON:
        return None
    return numerator / denominator


def classify_probe_validity(
    mean_g_cdgt: float,
    comparator_means: Dict[str, float],
    *,
    synthetic_only: bool,
) -> Dict[str, object]:
    strongest_name, strongest_value = max(
        comparator_means.items(),
        key=lambda item: (item[1], item[0]),
    )
    reasons: List[str] = []
    positive_comparators = {
        name: value for name, value in comparator_means.items() if value > EPSILON
    }
    if positive_comparators:
        denominator_name, denominator_value = max(
            positive_comparators.items(),
            key=lambda item: (item[1], item[0]),
        )
        status = "valid"
        reasons.append("finite positive comparator available")
    else:
        denominator_name = strongest_name
        denominator_value = strongest_value
        if mean_g_cdgt <= EPSILON:
            status = "invalid_all_arms_zero"
            reasons.append("CDGT and all comparator arms produced zero mean gain")
        else:
            status = "invalid_no_denominator"
            reasons.append("all comparator arms produced zero mean gain")
            reasons.append("routing advantage requires a finite positive comparator")
    return {
        "status": status,
        "reason": reasons[0],
        "reasons": reasons,
        "denominator_arm_name": denominator_name,
        "denominator_arm_value": denominator_value,
        "strongest_comparator_name": strongest_name,
        "strongest_comparator_mean_g": strongest_value,
        "synthetic_only": synthetic_only,
    }


def compute_lift(records: Sequence[dict]) -> Tuple[Optional[float], Optional[float], int, bool]:
    after_values: List[int] = []
    direct_values: List[int] = []
    saw_any_causal_field = False
    for record in records:
        has_after = "target_after_prereq_pass" in record
        has_direct = "target_direct_pass" in record
        saw_any_causal_field = saw_any_causal_field or has_after or has_direct
        if has_after and has_direct:
            after_values.append(int(record["target_after_prereq_pass"]))
            direct_values.append(int(record["target_direct_pass"]))
    if not after_values or not direct_values:
        return None, None, 0, saw_any_causal_field
    after_rate = sum(after_values) / len(after_values)
    direct_rate = sum(direct_values) / len(direct_values)
    absolute = after_rate - direct_rate
    relative = None if direct_rate <= EPSILON else absolute / direct_rate
    return absolute, relative, len(after_values), saw_any_causal_field


def collapse_audit(
    summaries: Dict[str, CapabilitySummary],
    edge_count: int,
    rank_corr_cdgt_loss: Optional[float],
    rank_corr_cdgt_uncertainty: Optional[float],
    residual_gain: float,
    heldout_count: int,
    causal_pair_count: int,
) -> Dict[str, dict]:
    capability_names = {name.lower() for name in summaries}
    source_count_values = [summary.source_count for summary in summaries.values()]
    generic_curriculum = heldout_count == 0 or causal_pair_count == 0
    f1 = (
        (rank_corr_cdgt_loss is not None and rank_corr_cdgt_loss >= 0.85)
        and residual_gain <= 0.0
    )
    f2 = max(source_count_values, default=0) <= 1
    f3 = edge_count == 0
    f4 = bool(capability_names) and all(
        "source" in name or "conflict" in name for name in capability_names
    )
    evidence_demand = bool(capability_names) and capability_names.issubset(ACTION_CAPABILITIES)
    intervention_outcome = bool(capability_names) and all(
        any(token in name for token in ("intervention", "outcome", "episode", "action"))
        for name in capability_names
    )
    pb_g9 = (
        (
            rank_corr_cdgt_loss is not None
            and rank_corr_cdgt_loss >= 0.85
        )
        or (
            rank_corr_cdgt_uncertainty is not None
            and rank_corr_cdgt_uncertainty >= 0.85
        )
    ) and residual_gain <= 0.0
    return {
        "generic_curriculum": {
            "active": generic_curriculum,
            "reason": (
                "No live held-out/retest split or no prerequisite causal pairs were available."
                if generic_curriculum
                else "Held-out retest and prerequisite evidence are both present."
            ),
        },
        "f1_post_failure_repair": {
            "active": f1,
            "reason": (
                "CDGT rank closely tracks loss-only repair pressure with no residual retest gain."
                if f1
                else "CDGT rank is not reducible to loss-only repair pressure."
            ),
        },
        "f2_retrieval_key_collapse": {
            "active": f2,
            "reason": (
                "Each capability appears only once, so graph nodes behave like retrieval keys."
                if f2
                else "Capabilities have repeated evidence and are not unique per episode."
            ),
        },
        "f3_branch_bundle_collapse": {
            "active": f3,
            "reason": (
                "No prerequisite edges were found, so the graph cannot choose capability order."
                if f3
                else "Prerequisite edges are present and affect routing."
            ),
        },
        "f4_source_conflict_only": {
            "active": f4,
            "reason": (
                "Capability names are only source/conflict families."
                if f4
                else "Capability families extend beyond source/conflict arbitration."
            ),
        },
        "evidence_demand_collapse": {
            "active": evidence_demand,
            "reason": (
                "Capability nodes are only answer/retrieve/ask/defer actions."
                if evidence_demand
                else "Capability nodes are not limited to answer/retrieve/ask/defer actions."
            ),
        },
        "intervention_outcome_collapse": {
            "active": intervention_outcome,
            "reason": (
                "Capability names only encode action or outcome episode types."
                if intervention_outcome
                else "Capability names are not just action/outcome episode labels."
            ),
        },
        "pb_g9_loss_entropy_collapse": {
            "active": pb_g9,
            "reason": (
                "CDGT rank is explained by loss or uncertainty with no residual retest gain."
                if pb_g9
                else "CDGT retains signal beyond loss/uncertainty ranking."
            ),
        },
    }


def determine_decision(
    metrics: Dict[str, Optional[float]],
    *,
    probe_validity: Dict[str, object],
    causal_pair_count: int,
    saw_any_causal_field: bool,
    collapse_report: Dict[str, dict],
) -> Tuple[str, str, List[str], List[str]]:
    if probe_validity["status"] != "valid":
        invalid_reasons = [str(reason) for reason in probe_validity["reasons"]]
        return "invalid", str(probe_validity["reason"]), [], invalid_reasons

    kill_reasons: List[str] = []
    weak_reasons: List[str] = []
    routing_advantage_adjusted = metrics["routing_advantage_overhead_adjusted"]
    routing_advantage_raw = metrics["routing_advantage_raw"]
    deficit_auc = metrics["deficit_auc"]
    blocked_prereq_mass = metrics["blocked_prereq_mass"]
    baseline_waste_total = metrics["baseline_waste_total"]
    prereq_lift_absolute = metrics["prerequisite_lift_absolute"]
    prereq_lift_relative = metrics["prerequisite_lift_relative"]
    rank_corr_cdgt_loss = metrics["rank_corr_cdgt_loss"]
    rank_corr_cdgt_uncertainty = metrics["rank_corr_cdgt_uncertainty"]
    residual_gain = metrics["residual_gain"]

    if routing_advantage_adjusted is None or routing_advantage_adjusted < 1.2:
        kill_reasons.append("adjusted routing advantage stayed below the 1.2 kill floor")
    if (
        routing_advantage_raw is not None
        and routing_advantage_adjusted is not None
        and routing_advantage_adjusted >= routing_advantage_raw - EPSILON
    ):
        pass
    elif (
        routing_advantage_raw is not None
        and routing_advantage_adjusted is not None
        and routing_advantage_adjusted < routing_advantage_raw
        and routing_advantage_adjusted < 1.0
    ):
        kill_reasons.append("retest overhead erased the raw routing advantage")
    if deficit_auc is not None and deficit_auc < 0.60:
        kill_reasons.append("deficit AUC stayed below the 0.60 kill floor")
    if blocked_prereq_mass is not None and blocked_prereq_mass < 0.10:
        kill_reasons.append("blocked prerequisite mass stayed below 0.10")
    if saw_any_causal_field and causal_pair_count > 0:
        if prereq_lift_absolute is None:
            kill_reasons.append("causal prerequisite lift could not be computed from provided fields")
        elif prereq_lift_absolute < 0.0:
            kill_reasons.append("causal prerequisite lift was negative")
    if (
        rank_corr_cdgt_loss is not None
        and rank_corr_cdgt_loss >= 0.85
        and residual_gain <= 0.0
    ):
        kill_reasons.append("CDGT rank collapsed into loss rank with no residual retest gain")
    if (
        rank_corr_cdgt_uncertainty is not None
        and rank_corr_cdgt_uncertainty >= 0.85
        and residual_gain <= 0.0
    ):
        kill_reasons.append("CDGT rank collapsed into uncertainty rank with no residual retest gain")
    if collapse_report["pb_g9_loss_entropy_collapse"]["active"]:
        kill_reasons.append("PB-G9 collapse guard triggered")

    if deficit_auc is None:
        weak_reasons.append("deficit AUC is null because held-out labels are degenerate")
    elif 0.60 <= deficit_auc < 0.70:
        weak_reasons.append("deficit AUC stayed in the weak band")
    if routing_advantage_adjusted is not None and 1.2 <= routing_advantage_adjusted < 1.5:
        weak_reasons.append("adjusted routing advantage stayed in the weak band")
    if prereq_lift_absolute is not None and 0.05 <= prereq_lift_absolute < 0.15:
        weak_reasons.append("prerequisite lift stayed in the weak band")
    if causal_pair_count == 0:
        weak_reasons.append("missing paired causal fields prevented a real prerequisite lift estimate")

    pass_conditions = [
        routing_advantage_adjusted is not None and routing_advantage_adjusted >= 1.5,
        deficit_auc is not None and deficit_auc >= 0.70,
        baseline_waste_total is not None and baseline_waste_total >= 0.33,
        blocked_prereq_mass is not None and blocked_prereq_mass > 0.20,
        (
            prereq_lift_absolute is not None and prereq_lift_absolute > 0.15
        )
        or (
            prereq_lift_relative is not None and prereq_lift_relative > 0.25
        ),
        (
            (rank_corr_cdgt_loss is not None and rank_corr_cdgt_loss < 0.85)
            and (rank_corr_cdgt_uncertainty is not None and rank_corr_cdgt_uncertainty < 0.85)
        )
        or residual_gain > 0.0,
    ]

    if kill_reasons:
        return "kill", kill_reasons[0], kill_reasons, weak_reasons
    if all(pass_conditions):
        return "pass", "all offline gate thresholds cleared", kill_reasons, weak_reasons
    if not weak_reasons:
        weak_reasons.append("offline signal was insufficient for a pass but did not trigger kill")
    return "weak", weak_reasons[0], kill_reasons, weak_reasons


def json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {value!r}")


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def main() -> int:
    args = parse_args()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    fixture_mode: Optional[str] = None
    if args.self_test:
        fixture_mode = "self_test"
        raw_records = build_self_test_fixture(args.seed)
    elif args.corrected_self_test:
        fixture_mode = "corrected_self_test"
        raw_records = build_corrected_self_test_fixture(args.seed)
    else:
        raw_records = load_jsonl_records(args.input)
    records = prepare_records(raw_records)
    summaries, prereq_map, edge_rows = summarize_capabilities(records, args.lambda_prereq)
    episode_rows, selection_budget = compute_episode_rows(records, summaries)

    ordered_by_cdgt = sorted(
        episode_rows,
        key=lambda row: (row["cdgt_rank"], row["baseline_rank"], row["episode_id"]),
    )
    ordered_by_baseline = sorted(
        episode_rows,
        key=lambda row: (row["baseline_rank"], row["episode_id"]),
    )
    ordered_by_loss = sorted(
        episode_rows,
        key=lambda row: (row["loss_rank"], row["baseline_rank"], row["episode_id"]),
    )
    ordered_by_uncertainty = sorted(
        episode_rows,
        key=lambda row: (row["uncertainty_rank"], row["baseline_rank"], row["episode_id"]),
    )
    budget_count = int(selection_budget)
    top_cdgt = ordered_by_cdgt[:budget_count]
    top_baseline = ordered_by_baseline[:budget_count]
    top_loss = ordered_by_loss[:budget_count]
    top_uncertainty = ordered_by_uncertainty[:budget_count]
    mean_g_cdgt = mean_gain(top_cdgt)
    mean_g_baseline = mean_gain(top_baseline)
    mean_g_loss = mean_gain(top_loss)
    mean_g_uncertainty = mean_gain(top_uncertainty)
    comparator_means = {
        "static": mean_g_baseline,
        "loss": mean_g_loss,
        "uncertainty": mean_g_uncertainty,
    }
    probe_validity = classify_probe_validity(
        mean_g_cdgt,
        comparator_means,
        synthetic_only=fixture_mode is not None,
    )
    routing_advantage_raw = (
        safe_ratio(mean_g_cdgt, float(probe_validity["denominator_arm_value"]))
        if probe_validity["status"] == "valid"
        else None
    )
    routing_advantage_overhead_adjusted = (
        None
        if routing_advantage_raw is None
        else routing_advantage_raw / (1.0 + args.retest_overhead)
    )

    auc_rows = []
    for record in records:
        selected_capability = choose_capability(record, summaries)
        score = summaries[selected_capability].prerequisite_closed_deficit
        label = 1 - int(record["heldout_retest_pass"])
        auc_rows.append((score, label))
    deficit_auc = compute_auc(auc_rows)

    total_cost = sum(record["cost"] for record in records)
    baseline_waste_cost = 0.0
    blocked_prereq_cost = 0.0
    for row in ordered_by_baseline:
        if row["mastered"] or row["blocked"] or row["irrelevant"]:
            baseline_waste_cost += row["cost"]
        if row["blocked"] and row["direct_prerequisites"]:
            blocked_prereq_cost += row["cost"]
    baseline_waste_total = baseline_waste_cost / total_cost if total_cost else 0.0
    blocked_prereq_mass = blocked_prereq_cost / total_cost if total_cost else 0.0

    prereq_lift_absolute, prereq_lift_relative, causal_pair_count, saw_any_causal_field = compute_lift(records)

    cdgt_rank_map = {row["episode_id"]: row["cdgt_rank"] for row in episode_rows}
    loss_rank_map = {row["episode_id"]: row["loss_rank"] for row in episode_rows}
    uncertainty_rank_map = {
        row["episode_id"]: row["uncertainty_rank"] for row in episode_rows
    }
    rank_corr_cdgt_loss = spearman(cdgt_rank_map, loss_rank_map)
    rank_corr_cdgt_uncertainty = spearman(cdgt_rank_map, uncertainty_rank_map)
    residual_gain = mean_g_cdgt - mean_g_baseline
    heldout_count = len(records)

    collapse_report = collapse_audit(
        summaries,
        len(edge_rows),
        rank_corr_cdgt_loss,
        rank_corr_cdgt_uncertainty,
        residual_gain,
        heldout_count,
        causal_pair_count,
    )

    metrics: Dict[str, Optional[float]] = {
        "routing_advantage_raw": routing_advantage_raw,
        "routing_advantage_overhead_adjusted": routing_advantage_overhead_adjusted,
        "deficit_auc": deficit_auc,
        "baseline_waste_total": baseline_waste_total,
        "blocked_prereq_mass": blocked_prereq_mass,
        "prerequisite_lift_absolute": prereq_lift_absolute,
        "prerequisite_lift_relative": prereq_lift_relative,
        "rank_corr_cdgt_loss": rank_corr_cdgt_loss,
        "rank_corr_cdgt_uncertainty": rank_corr_cdgt_uncertainty,
        "episode_count": float(len(records)),
        "capability_count": float(len(summaries)),
        "edge_count": float(len(edge_rows)),
        "heldout_count": float(heldout_count),
        "mean_g_cdgt": mean_g_cdgt,
        "mean_g_baseline": mean_g_baseline,
        "mean_g_static": mean_g_baseline,
        "mean_g_loss": mean_g_loss,
        "mean_g_uncertainty": mean_g_uncertainty,
        "strongest_comparator_mean_g": float(probe_validity["strongest_comparator_mean_g"]),
        "residual_gain": residual_gain,
    }

    decision, decision_reason, kill_reasons, weak_reasons = determine_decision(
        metrics,
        probe_validity=probe_validity,
        causal_pair_count=causal_pair_count,
        saw_any_causal_field=saw_any_causal_field,
        collapse_report=collapse_report,
    )

    active_collapse_flags = sorted(
        key for key, value in collapse_report.items() if value["active"]
    )
    episode_rows_for_output: List[dict] = []
    for row in sorted(
        episode_rows,
        key=lambda item: (item["cdgt_rank"], item["baseline_rank"], item["episode_id"]),
    ):
        row_copy = dict(row)
        row_copy["collapse_audit_flags"] = active_collapse_flags
        episode_rows_for_output.append(row_copy)

    graph_payload = {
        "nodes": [
            {
                "capability": summary.name,
                "target_threshold": 0.5,
                "source_count": summary.source_count,
                "estimate_pass_rate": summary.m_hat,
                "heldout_retest_pass_rate": summary.retest_pass_rate,
                "post_teach_retest_pass_rate": summary.post_teach_pass_rate,
                "direct_prerequisites": summary.direct_prereqs,
                "prerequisite_closure": summary.all_prereqs,
            }
            for summary in sorted(summaries.values(), key=lambda item: item.name)
        ],
        "edges": edge_rows,
        "target_thresholds": {summary.name: 0.5 for summary in summaries.values()},
        "source_counts": {summary.name: summary.source_count for summary in summaries.values()},
    }
    deficits_rows = [
        {
            "capability": summary.name,
            "m_hat": summary.m_hat,
            "deficit": summary.deficit,
            "uncertainty": summary.uncertainty,
            "prerequisite_closed_deficit": summary.prerequisite_closed_deficit,
            "estimator_count": summary.estimator_count,
            "heldout_count": summary.heldout_count,
            "retest_pass_rate": summary.retest_pass_rate,
        }
        for summary in sorted(summaries.values(), key=lambda item: item.name)
    ]

    manifest = {
        "run": {
            "input": fixture_mode.replace("_", "-") if fixture_mode is not None else str(args.input),
            "out": str(out_dir),
            "seed": args.seed,
            "lambda_prereq": args.lambda_prereq,
            "retest_overhead": args.retest_overhead,
            "self_test": fixture_mode is not None,
            "fixture_mode": fixture_mode,
            "corrected_self_test": bool(args.corrected_self_test),
            "selection_budget": budget_count,
        },
        "counts": {
            "episode_count": len(records),
            "capability_count": len(summaries),
            "edge_count": len(edge_rows),
            "heldout_count": heldout_count,
            "causal_pair_count": causal_pair_count,
        },
        "metrics": metrics,
        "probe_validity": probe_validity,
        "decision": decision,
        "reason": decision_reason,
        "kill_reasons": kill_reasons,
        "weak_reasons": weak_reasons,
        "collapse_audit": collapse_report,
        "active_collapse_flags": active_collapse_flags,
        "notes": [
            "Deficits are estimated from estimate_pass only; held-out retest labels are reserved for validation metrics.",
            "Expected episode reduction uses post_teach_retest_pass - heldout_retest_pass and is clipped at zero.",
            "Zero-denominator routing comparisons are invalid probe states rather than valid kills.",
        ],
    }

    write_json(out_dir / "capability_graph.json", graph_payload)
    write_jsonl(out_dir / "deficits.jsonl", deficits_rows)
    write_jsonl(out_dir / "teaching_episodes.jsonl", episode_rows_for_output)
    write_json(out_dir / "manifest.json", manifest)

    input_label = fixture_mode.replace("_", "-") if fixture_mode is not None else str(args.input)
    print(
        "R390_CDGT_ACTIVATION"
        f" mode={fixture_mode if fixture_mode is not None else 'input'}"
        f" seed={args.seed}"
        f" input={input_label}"
        f" out={out_dir}"
        f" lambda_prereq={args.lambda_prereq:.6g}"
        f" retest_overhead={args.retest_overhead:.6g}"
    )
    print(
        "R390_CDGT_METRICS"
        f" probe_validity_status={probe_validity['status']}"
        f" strongest_comparator_name={probe_validity['strongest_comparator_name']}"
        f" strongest_comparator_mean_g={probe_validity['strongest_comparator_mean_g']:.6g}"
        f" denominator_arm_name={probe_validity['denominator_arm_name']}"
        f" denominator_arm_value={probe_validity['denominator_arm_value']:.6g}"
        f" routing_advantage_raw={routing_advantage_raw if routing_advantage_raw is not None else 'null'}"
        f" routing_advantage_overhead_adjusted={routing_advantage_overhead_adjusted if routing_advantage_overhead_adjusted is not None else 'null'}"
        f" deficit_auc={deficit_auc if deficit_auc is not None else 'null'}"
        f" baseline_waste_total={baseline_waste_total:.6g}"
        f" blocked_prereq_mass={blocked_prereq_mass:.6g}"
        f" prerequisite_lift_absolute={prereq_lift_absolute if prereq_lift_absolute is not None else 'null'}"
        f" prerequisite_lift_relative={prereq_lift_relative if prereq_lift_relative is not None else 'null'}"
        f" rank_corr_cdgt_loss={rank_corr_cdgt_loss if rank_corr_cdgt_loss is not None else 'null'}"
        f" rank_corr_cdgt_uncertainty={rank_corr_cdgt_uncertainty if rank_corr_cdgt_uncertainty is not None else 'null'}"
        f" mean_g_cdgt={mean_g_cdgt:.6g}"
        f" mean_g_static={mean_g_baseline:.6g}"
        f" mean_g_loss={mean_g_loss:.6g}"
        f" mean_g_uncertainty={mean_g_uncertainty:.6g}"
        f" episode_count={len(records)}"
        f" capability_count={len(summaries)}"
        f" edge_count={len(edge_rows)}"
        f" heldout_count={heldout_count}"
    )
    print(f"R390_CDGT_DECISION {decision} reason={decision_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
