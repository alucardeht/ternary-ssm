#!/usr/bin/env python3
"""Offline Intervention-Outcome Teaching Episode diagnostic probe for R390."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


CONDITIONS = ("R_plain", "R_matched", "IOTE", "IOTE_no_outcome")
REQUIRED_INPUT_FIELDS = (
    "episode_id",
    "split",
    "family",
    "variant",
    "prompt",
    "state",
    "bad_action",
    "bad_outcome",
    "corrected_action",
    "corrected_outcome",
    "failure_cause",
    "verifier_success",
)
PASS_THRESHOLDS = {
    "a_iote": 1.6,
    "a_iote_matched": 1.35,
    "exec_gain_abs": 0.10,
    "exec_error_reduction": 0.25,
    "delta_align": 0.15,
    "delta_cause": 0.20,
}
WEAK_THRESHOLDS = {
    "a_iote_low": 1.2,
    "a_iote_matched_low": 1.1,
    "exec_gain_abs_low": 0.03,
    "delta_align_low": 0.05,
}
TOKEN_RE = re.compile(r"[a-z0-9_]+")
ACTION_RE = re.compile(r"^(?P<operator>[a-z0-9_]+)\((?P<argument>.*)\)$")
EPSILON = 1e-12
OUTCOME_QUERY_FIELD = "desired_outcome"
STATE_DYNAMIC_KEYS = {"item", "lane", "priority", OUTCOME_QUERY_FIELD}


@dataclass(frozen=True)
class BaseCase:
    episode_id: str
    split: str
    family: str
    variant: str
    prompt: str
    state: Dict[str, str]
    bad_action: str
    bad_outcome: str
    corrected_action: str
    corrected_outcome: str
    failure_cause: str
    correction_delta: str
    verifier_id: str
    verifier_success: bool
    plain_final_answer: str
    matched_note: str
    query_outcome: Optional[str]


@dataclass
class ConditionEpisode:
    episode_id: str
    split: str
    condition: str
    family: str
    variant: str
    prompt: str
    item_key: str
    state_text: str
    query_text: str
    memory_text: str
    corrected_action: str
    corrected_operator: str
    corrected_argument: str
    corrected_outcome: str
    failure_cause: str
    bad_action: str
    bad_operator: str
    bad_argument: str
    bad_outcome: str
    verifier_id: str
    correction_delta: str
    query_outcome: Optional[str]
    token_cost: float
    overhead_cost: float
    total_cost: float
    visible_fields: Dict[str, bool]
    visible_label_values: Dict[str, Optional[str]]
    visible_label_sources: Dict[str, Optional[str]]
    feature_fields: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true", help="Use the built-in deterministic fixture.")
    mode.add_argument("--input", type=Path, help="JSONL input with explicit IOTE fields.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--seed", type=int, default=390, help="Deterministic seed for fixture generation.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit over base cases after loading.")
    parser.add_argument(
        "--executor-overhead",
        type=float,
        default=12.0,
        help="Cost added per executor or verifier event.",
    )
    parser.add_argument(
        "--token-cost-mode",
        choices=("whitespace", "chars"),
        default="whitespace",
        help="Text cost accounting mode.",
    )
    return parser.parse_args()


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator <= EPSILON:
        return None
    return numerator / denominator


def relative_error_reduction(baseline_success: float, candidate_success: float) -> Optional[float]:
    baseline_error = 1.0 - baseline_success
    candidate_error = 1.0 - candidate_success
    if baseline_error <= EPSILON:
        return None
    return max(0.0, (baseline_error - candidate_error) / baseline_error)


def rounded(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value, 6)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def serialize_state(state: Dict[str, str]) -> str:
    parts = [f"{key}={state[key]}" for key in sorted(state)]
    return " ".join(parts)


def text_cost(text: str, mode: str) -> float:
    if mode == "chars":
        return float(len(text))
    return float(len(text.split()))


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
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
                raise ValueError(f"{path}:{line_number}: each row must be a JSON object.")
            rows.append(record)
    if not rows:
        raise ValueError(f"{path}: no JSONL rows found.")
    return rows


def parse_action(action: str) -> Tuple[str, str]:
    match = ACTION_RE.match(action.strip())
    if not match:
        return action.strip(), ""
    return match.group("operator"), match.group("argument")


def build_action(operator: str, argument: str) -> str:
    return f"{operator}({argument})" if argument else operator


def operator_surface(operator: str) -> List[str]:
    return tokenize(operator.replace("_", " "))


def normalize_query_outcome(record: dict, state: Dict[str, str], corrected_outcome: str) -> Optional[str]:
    if OUTCOME_QUERY_FIELD in record:
        return str(record[OUTCOME_QUERY_FIELD])
    if OUTCOME_QUERY_FIELD in state:
        return str(state[OUTCOME_QUERY_FIELD])
    return None


def build_cases_from_input(rows: Sequence[dict]) -> List[BaseCase]:
    cases: List[BaseCase] = []
    for record in rows:
        missing = [field for field in REQUIRED_INPUT_FIELDS if field not in record]
        if missing:
            raise ValueError(
                "Input row is not an explicit IOTE episode; missing required fields: " + ", ".join(missing)
            )
        state = record["state"]
        if not isinstance(state, dict) or not state:
            raise ValueError(f"{record.get('episode_id', '<unknown>')}: state must be a non-empty object.")
        normalized_state = {str(key): str(value) for key, value in state.items()}
        corrected_outcome = str(record["corrected_outcome"])
        cases.append(
            BaseCase(
                episode_id=str(record["episode_id"]),
                split=str(record["split"]),
                family=str(record["family"]),
                variant=str(record["variant"]),
                prompt=str(record["prompt"]),
                state=normalized_state,
                bad_action=str(record["bad_action"]),
                bad_outcome=str(record["bad_outcome"]),
                corrected_action=str(record["corrected_action"]),
                corrected_outcome=corrected_outcome,
                failure_cause=str(record["failure_cause"]),
                correction_delta=str(record.get("correction_delta", "repair")),
                verifier_id=str(record.get("verifier_id", "external_verifier")),
                verifier_success=bool(record["verifier_success"]),
                plain_final_answer=str(record.get("plain_final_answer", corrected_outcome)),
                matched_note=str(record.get("matched_note", corrected_outcome)),
                query_outcome=normalize_query_outcome(record, normalized_state, corrected_outcome),
            )
        )
    train_count = sum(1 for case in cases if case.split == "train")
    heldout_count = sum(1 for case in cases if case.split == "heldout")
    if train_count == 0 or heldout_count == 0:
        raise ValueError("Input rows must include both train and heldout splits.")
    return cases


def family_specs() -> List[dict]:
    return [
        {
            "family": "retire_ticket",
            "prompt": "Retire ticket {item} from backlog.",
            "item_prefix": "rt",
            "state_base": {"surface": "retire", "module": "tickets"},
            "bad_action": "delete_ticket({item})",
            "variants": [
                {
                    "name": "history",
                    "state": {
                        "guard": "history",
                        "policy": "audit",
                        "transition": "archive",
                        OUTCOME_QUERY_FIELD: "ok_ticket_archived_with_history",
                    },
                    "corrected_action": "archive_ticket({item})",
                    "corrected_outcome": "ok_ticket_archived_with_history",
                    "bad_outcome": "error_history_delete_blocked",
                    "failure_cause": "history_guard",
                    "plain_final_answer": "Archive ticket {item} so the history stays intact.",
                    "matched_note": "Use archive ticket on {item} to preserve the audit history.",
                    "correction_delta": "delete_to_archive",
                },
                {
                    "name": "lease",
                    "state": {
                        "guard": "lease",
                        "policy": "release",
                        "transition": "release_delete",
                        OUTCOME_QUERY_FIELD: "ok_lease_released_ticket_deleted",
                    },
                    "corrected_action": "release_then_delete_ticket({item})",
                    "corrected_outcome": "ok_lease_released_ticket_deleted",
                    "bad_outcome": "error_lease_release_required",
                    "failure_cause": "lease_guard",
                    "plain_final_answer": "Release then delete ticket {item} once the lease is clear.",
                    "matched_note": "Use release then delete ticket on {item} after clearing the active lease.",
                    "correction_delta": "delete_to_release_delete",
                },
            ],
        },
        {
            "family": "set_cell",
            "prompt": "Set cell {item} to the requested value.",
            "item_prefix": "c",
            "state_base": {"surface": "set_value", "module": "sheet"},
            "bad_action": "write_cell({item})",
            "variants": [
                {
                    "name": "formula",
                    "state": {
                        "guard": "formula",
                        "policy": "override",
                        "transition": "override_formula",
                        OUTCOME_QUERY_FIELD: "ok_formula_overridden",
                    },
                    "corrected_action": "override_formula({item})",
                    "corrected_outcome": "ok_formula_overridden",
                    "bad_outcome": "error_formula_write_blocked",
                    "failure_cause": "formula_guard",
                    "plain_final_answer": "Override formula for cell {item} instead of writing directly.",
                    "matched_note": "Use override formula on {item} because direct writes are blocked.",
                    "correction_delta": "write_to_override",
                },
                {
                    "name": "mirror",
                    "state": {
                        "guard": "mirror",
                        "policy": "sync_source",
                        "transition": "update_source",
                        OUTCOME_QUERY_FIELD: "ok_source_updated_and_synced",
                    },
                    "corrected_action": "update_source_then_sync({item})",
                    "corrected_outcome": "ok_source_updated_and_synced",
                    "bad_outcome": "error_mirror_source_required",
                    "failure_cause": "mirror_guard",
                    "plain_final_answer": "Update source then sync cell {item} through the mirror pipeline.",
                    "matched_note": "Use update source then sync on {item} so the mirror stays authoritative.",
                    "correction_delta": "write_to_sync_source",
                },
            ],
        },
        {
            "family": "rename_queue",
            "prompt": "Rename queue {item} to its target label.",
            "item_prefix": "q",
            "state_base": {"surface": "rename", "module": "queues"},
            "bad_action": "rename_queue({item})",
            "variants": [
                {
                    "name": "alias",
                    "state": {
                        "guard": "alias",
                        "policy": "swap",
                        "transition": "swap_alias",
                        OUTCOME_QUERY_FIELD: "ok_alias_swapped",
                    },
                    "corrected_action": "swap_alias({item})",
                    "corrected_outcome": "ok_alias_swapped",
                    "bad_outcome": "error_alias_reserved_rename_blocked",
                    "failure_cause": "alias_guard",
                    "plain_final_answer": "Swap alias for queue {item} rather than renaming in place.",
                    "matched_note": "Use swap alias on {item} because the alias slot is reserved.",
                    "correction_delta": "rename_to_swap_alias",
                },
                {
                    "name": "retire",
                    "state": {
                        "guard": "retire",
                        "policy": "link_old",
                        "transition": "link_retire",
                        OUTCOME_QUERY_FIELD: "ok_queue_linked_and_retired",
                    },
                    "corrected_action": "link_and_retire_queue({item})",
                    "corrected_outcome": "ok_queue_linked_and_retired",
                    "bad_outcome": "error_retire_link_required",
                    "failure_cause": "retire_guard",
                    "plain_final_answer": "Link and retire queue {item} so the old route keeps forwarding.",
                    "matched_note": "Use link and retire queue on {item} instead of renaming it directly.",
                    "correction_delta": "rename_to_link_retire",
                },
            ],
        },
        {
            "family": "move_file",
            "prompt": "Move file {item} into the target storage.",
            "item_prefix": "f",
            "state_base": {"surface": "move", "module": "storage"},
            "bad_action": "move_file({item})",
            "variants": [
                {
                    "name": "snapshot",
                    "state": {
                        "guard": "snapshot",
                        "policy": "copy_mark",
                        "transition": "copy_then_mark",
                        OUTCOME_QUERY_FIELD: "ok_file_copied_and_marked_stale",
                    },
                    "corrected_action": "copy_then_mark_stale({item})",
                    "corrected_outcome": "ok_file_copied_and_marked_stale",
                    "bad_outcome": "error_snapshot_move_blocked",
                    "failure_cause": "snapshot_guard",
                    "plain_final_answer": "Copy then mark stale for file {item} to keep snapshots valid.",
                    "matched_note": "Use copy then mark stale on {item} because snapshots forbid a direct move.",
                    "correction_delta": "move_to_copy_mark",
                },
                {
                    "name": "quorum",
                    "state": {
                        "guard": "quorum",
                        "policy": "clone_vote",
                        "transition": "clone_quorum",
                        OUTCOME_QUERY_FIELD: "ok_file_cloned_to_quorum",
                    },
                    "corrected_action": "clone_to_quorum({item})",
                    "corrected_outcome": "ok_file_cloned_to_quorum",
                    "bad_outcome": "error_quorum_clone_required",
                    "failure_cause": "quorum_guard",
                    "plain_final_answer": "Clone to quorum for file {item} instead of moving the only copy.",
                    "matched_note": "Use clone to quorum on {item} because quorum storage needs replicated placement.",
                    "correction_delta": "move_to_clone_quorum",
                },
            ],
        },
        {
            "family": "route_bundle",
            "prompt": "Route bundle {item} through the requested outcome lane.",
            "item_prefix": "b",
            "state_base": {
                "surface": "route_bundle",
                "module": "router",
                "guard": "shadow",
                "policy": "target_lane",
                "transition": "shadow_lane",
            },
            "bad_action": "route_bundle({item})",
            "variants": [
                {
                    "name": "archive_shadow",
                    "state": {
                        OUTCOME_QUERY_FIELD: "ok_bundle_archived_shadow",
                    },
                    "corrected_action": "shadow_archive_bundle({item})",
                    "corrected_outcome": "ok_bundle_archived_shadow",
                    "bad_outcome": "error_shadow_lane_archive_required",
                    "failure_cause": "shadow_target_guard",
                    "plain_final_answer": "Shadow archive bundle {item} so it lands in the archived lane.",
                    "matched_note": "Use shadow archive bundle on {item} when the requested lane is archived shadow.",
                    "correction_delta": "route_to_shadow_archive",
                },
                {
                    "name": "escalate_shadow",
                    "state": {
                        OUTCOME_QUERY_FIELD: "ok_bundle_escalated_shadow",
                    },
                    "corrected_action": "shadow_escalate_bundle({item})",
                    "corrected_outcome": "ok_bundle_escalated_shadow",
                    "bad_outcome": "error_shadow_lane_escalate_required",
                    "failure_cause": "shadow_target_guard",
                    "plain_final_answer": "Shadow escalate bundle {item} so it lands in the escalated lane.",
                    "matched_note": "Use shadow escalate bundle on {item} when the requested lane is escalated shadow.",
                    "correction_delta": "route_to_shadow_escalate",
                },
            ],
        },
    ]


def build_self_test_fixture(seed: int) -> List[BaseCase]:
    del seed
    cases: List[BaseCase] = []
    for family_spec in family_specs():
        for variant_spec in family_spec["variants"]:
            for offset in range(4):
                item = f"{family_spec['item_prefix']}{offset + 1}_{variant_spec['name']}"
                split = "heldout" if offset == 3 else "train"
                state = dict(family_spec["state_base"])
                state.update(variant_spec["state"])
                state.update(
                    {
                        "item": item,
                        "lane": ("ops", "fin", "qa", "night")[offset],
                        "priority": ("p1", "p2", "p3", "p4")[offset],
                    }
                )
                cases.append(
                    BaseCase(
                        episode_id=f"{family_spec['family']}-{variant_spec['name']}-{offset + 1}",
                        split=split,
                        family=family_spec["family"],
                        variant=variant_spec["name"],
                        prompt=family_spec["prompt"].format(item=item),
                        state=state,
                        bad_action=family_spec["bad_action"].format(item=item),
                        bad_outcome=variant_spec["bad_outcome"],
                        corrected_action=variant_spec["corrected_action"].format(item=item),
                        corrected_outcome=variant_spec["corrected_outcome"],
                        failure_cause=variant_spec["failure_cause"],
                        correction_delta=variant_spec["correction_delta"],
                        verifier_id=f"verify_{family_spec['family']}",
                        verifier_success=True,
                        plain_final_answer=variant_spec["plain_final_answer"].format(item=item),
                        matched_note=variant_spec["matched_note"].format(item=item),
                        query_outcome=state.get(OUTCOME_QUERY_FIELD),
                    )
                )
    return cases


def visible_fields_for_condition(condition: str) -> Dict[str, bool]:
    visible = {
        "prompt": True,
        "state": True,
        "plain_final_answer": False,
        "matched_note": False,
        "bad_action": False,
        "bad_outcome": False,
        "failure_cause": False,
        "corrected_action": False,
        "corrected_outcome": False,
        "correction_delta": False,
        "verifier_id": False,
    }
    if condition == "R_plain":
        visible["plain_final_answer"] = True
    elif condition == "R_matched":
        visible["plain_final_answer"] = True
        visible["matched_note"] = True
    elif condition == "IOTE":
        visible.update(
            {
                "bad_action": True,
                "bad_outcome": True,
                "failure_cause": True,
                "corrected_action": True,
                "corrected_outcome": True,
                "correction_delta": True,
                "verifier_id": True,
            }
        )
    elif condition == "IOTE_no_outcome":
        visible.update(
            {
                "bad_action": True,
                "failure_cause": True,
                "corrected_action": True,
                "correction_delta": True,
                "verifier_id": True,
            }
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return visible


def render_condition_memory(case: BaseCase, condition: str) -> Tuple[str, float, Dict[str, bool], List[str]]:
    visible_fields = visible_fields_for_condition(condition)
    state_text = serialize_state(case.state)
    lines = [f"prompt={case.prompt}", f"state={state_text}"]
    feature_fields = ["prompt", "state"]
    if visible_fields["plain_final_answer"]:
        lines.append(f"correct_final_answer={case.plain_final_answer}")
        feature_fields.append("plain_final_answer")
    if visible_fields["matched_note"]:
        lines.append(f"repair_note={case.matched_note}")
        feature_fields.append("matched_note")
    if visible_fields["bad_action"]:
        lines.append(f"bad_action={case.bad_action}")
        feature_fields.append("bad_action")
    if visible_fields["bad_outcome"]:
        lines.append(f"bad_outcome={case.bad_outcome}")
        feature_fields.append("bad_outcome")
    if visible_fields["failure_cause"]:
        lines.append(f"failure_cause={case.failure_cause}")
        feature_fields.append("failure_cause")
    if visible_fields["corrected_action"]:
        lines.append(f"corrected_action={case.corrected_action}")
        feature_fields.append("corrected_action")
    if visible_fields["corrected_outcome"]:
        lines.append(f"corrected_outcome={case.corrected_outcome}")
        feature_fields.append("corrected_outcome")
    if visible_fields["correction_delta"]:
        lines.append(f"correction_delta={case.correction_delta}")
        feature_fields.append("correction_delta")
    if visible_fields["verifier_id"]:
        lines.append(f"verifier_id={case.verifier_id}")
        feature_fields.append("verifier_id")
    if condition == "R_plain":
        overhead_events = 0.0
    elif condition == "R_matched":
        overhead_events = 0.0
    elif condition == "IOTE":
        overhead_events = 3.0
    else:
        overhead_events = 1.0
    return "\n".join(lines) + "\n", overhead_events, visible_fields, feature_fields


def infer_visible_operator_label(case: BaseCase, visible_fields: Dict[str, bool]) -> Tuple[Optional[str], Optional[str]]:
    corrected_operator, _ = parse_action(case.corrected_action)
    if visible_fields["corrected_action"]:
        return corrected_operator, "corrected_action"
    prose_parts: List[str] = []
    if visible_fields["plain_final_answer"]:
        prose_parts.append(case.plain_final_answer)
    if visible_fields["matched_note"]:
        prose_parts.append(case.matched_note)
    prose_tokens = set(tokenize(" ".join(prose_parts)))
    required_tokens = operator_surface(corrected_operator)
    if required_tokens and all(token in prose_tokens for token in required_tokens):
        return corrected_operator, "prose_operator_hint"
    return None, None


def infer_visible_outcome_label(case: BaseCase, visible_fields: Dict[str, bool]) -> Tuple[Optional[str], Optional[str]]:
    if visible_fields["corrected_outcome"]:
        return case.corrected_outcome, "corrected_outcome"
    return None, None


def infer_visible_cause_label(case: BaseCase, visible_fields: Dict[str, bool]) -> Tuple[Optional[str], Optional[str]]:
    if visible_fields["failure_cause"]:
        return case.failure_cause, "failure_cause"
    return None, None


def build_query_text(case: BaseCase) -> str:
    state_text = serialize_state(case.state)
    parts = [f"prompt={case.prompt}", f"state={state_text}"]
    if case.query_outcome:
        parts.append(f"{OUTCOME_QUERY_FIELD}={case.query_outcome}")
    return " ".join(parts)


def build_condition_episodes(
    cases: Sequence[BaseCase], token_cost_mode: str, executor_overhead: float
) -> List[ConditionEpisode]:
    episodes: List[ConditionEpisode] = []
    for case in cases:
        corrected_operator, corrected_argument = parse_action(case.corrected_action)
        bad_operator, bad_argument = parse_action(case.bad_action)
        item_key = case.state.get("item", corrected_argument)
        query_text = build_query_text(case)
        state_text = serialize_state(case.state)
        for condition in CONDITIONS:
            memory_text, overhead_events, visible_fields, feature_fields = render_condition_memory(case, condition)
            operator_label, operator_source = infer_visible_operator_label(case, visible_fields)
            outcome_label, outcome_source = infer_visible_outcome_label(case, visible_fields)
            cause_label, cause_source = infer_visible_cause_label(case, visible_fields)
            token_cost = text_cost(memory_text, token_cost_mode)
            overhead_cost = overhead_events * executor_overhead
            episodes.append(
                ConditionEpisode(
                    episode_id=case.episode_id,
                    split=case.split,
                    condition=condition,
                    family=case.family,
                    variant=case.variant,
                    prompt=case.prompt,
                    item_key=item_key,
                    state_text=state_text,
                    query_text=query_text,
                    memory_text=memory_text,
                    corrected_action=case.corrected_action,
                    corrected_operator=corrected_operator,
                    corrected_argument=corrected_argument,
                    corrected_outcome=case.corrected_outcome,
                    failure_cause=case.failure_cause,
                    bad_action=case.bad_action,
                    bad_operator=bad_operator,
                    bad_argument=bad_argument,
                    bad_outcome=case.bad_outcome,
                    verifier_id=case.verifier_id,
                    correction_delta=case.correction_delta,
                    query_outcome=case.query_outcome,
                    token_cost=token_cost,
                    overhead_cost=overhead_cost,
                    total_cost=token_cost + overhead_cost,
                    visible_fields=visible_fields,
                    visible_label_values={
                        "operator": operator_label,
                        "outcome": outcome_label,
                        "cause": cause_label,
                    },
                    visible_label_sources={
                        "operator": operator_source,
                        "outcome": outcome_source,
                        "cause": cause_source,
                    },
                    feature_fields=feature_fields,
                )
            )
    return episodes


class VisibleArmModel:
    def __init__(self) -> None:
        self.operator_weights: Dict[str, Counter[str]] = defaultdict(Counter)
        self.outcome_weights: Dict[str, Counter[str]] = defaultdict(Counter)
        self.cause_weights: Dict[str, Counter[str]] = defaultdict(Counter)
        self.operator_prior: Counter[str] = Counter()
        self.outcome_prior: Counter[str] = Counter()
        self.cause_prior: Counter[str] = Counter()
        self.training_counts = {"operator": 0, "outcome": 0, "cause": 0}

    def add(self, episode: ConditionEpisode) -> None:
        tokens = tokenize(episode.memory_text)
        operator_label = episode.visible_label_values["operator"]
        outcome_label = episode.visible_label_values["outcome"]
        cause_label = episode.visible_label_values["cause"]
        if operator_label is not None:
            for token in tokens:
                self.operator_weights[token][operator_label] += 1
            self.operator_prior[operator_label] += 1
            self.training_counts["operator"] += 1
        if outcome_label is not None:
            for token in tokens:
                self.outcome_weights[token][outcome_label] += 1
            self.outcome_prior[outcome_label] += 1
            self.training_counts["outcome"] += 1
        if cause_label is not None:
            for token in tokens:
                self.cause_weights[token][cause_label] += 1
            self.cause_prior[cause_label] += 1
            self.training_counts["cause"] += 1

    def _predict_label(
        self, weights: Dict[str, Counter[str]], prior: Counter[str], query: str, fallback: str
    ) -> str:
        scores: Counter[str] = Counter()
        for label, weight in prior.items():
            scores[label] += weight * 0.01
        for token in tokenize(query):
            for label, weight in weights.get(token, {}).items():
                scores[label] += weight
        if not scores:
            return fallback
        best_label, _ = max(scores.items(), key=lambda item: (item[1], item[0]))
        return best_label

    def predict(self, episode: ConditionEpisode) -> Tuple[str, str, str]:
        operator = self._predict_label(
            self.operator_weights,
            self.operator_prior,
            episode.query_text,
            episode.bad_operator,
        )
        outcome = self._predict_label(
            self.outcome_weights,
            self.outcome_prior,
            episode.query_text,
            episode.bad_outcome,
        )
        cause = self._predict_label(
            self.cause_weights,
            self.cause_prior,
            episode.query_text,
            "unknown_cause",
        )
        return operator, outcome, cause


def execute_action(case: BaseCase, action: str) -> str:
    corrected_operator, corrected_argument = parse_action(case.corrected_action)
    bad_operator, bad_argument = parse_action(case.bad_action)
    predicted_operator, predicted_argument = parse_action(action)
    if predicted_operator == corrected_operator and predicted_argument == corrected_argument:
        return case.corrected_outcome
    if predicted_operator == bad_operator and predicted_argument == bad_argument:
        return case.bad_outcome
    if predicted_operator == corrected_operator:
        return f"error_wrong_parameters_{case.family}"
    return f"error_wrong_family_{predicted_operator or 'unknown'}"


def verify(case: BaseCase, action: str, outcome: str) -> bool:
    return bool(case.verifier_success and action == case.corrected_action and outcome == case.corrected_outcome)


def bind_argument_from_state(case: BaseCase) -> str:
    argument = case.state.get("item")
    if argument:
        return argument
    _, corrected_argument = parse_action(case.corrected_action)
    return corrected_argument


def evaluate_before(cases: Sequence[BaseCase]) -> Tuple[float, List[dict]]:
    rows: List[dict] = []
    success_count = 0
    for case in cases:
        bad_operator, bad_argument = parse_action(case.bad_action)
        corrected_operator, corrected_argument = parse_action(case.corrected_action)
        executed_outcome = execute_action(case, case.bad_action)
        success = verify(case, case.bad_action, executed_outcome)
        success_count += int(success)
        rows.append(
            {
                "condition": "before",
                "episode_id": case.episode_id,
                "family": case.family,
                "variant": case.variant,
                "prompt": case.prompt,
                "query_outcome": case.query_outcome,
                "predicted_operator": bad_operator,
                "predicted_argument": bad_argument,
                "predicted_action": case.bad_action,
                "predicted_outcome": case.bad_outcome,
                "predicted_cause": case.failure_cause,
                "executed_outcome": executed_outcome,
                "success": success,
                "expected_operator": corrected_operator,
                "expected_argument": corrected_argument,
                "expected_action": case.corrected_action,
                "expected_outcome": case.corrected_outcome,
                "expected_cause": case.failure_cause,
                "bad_operator": bad_operator,
                "bad_argument": bad_argument,
                "bad_action": case.bad_action,
                "operator_correct": False,
                "binding_correct": bad_argument == corrected_argument,
                "action_exact_correct": False,
                "switch_correct": False,
                "outcome_sensitive": False,
            }
        )
    return success_count / max(1, len(cases)), rows


def build_models(episodes: Sequence[ConditionEpisode]) -> Dict[str, VisibleArmModel]:
    models = {condition: VisibleArmModel() for condition in CONDITIONS}
    for episode in episodes:
        if episode.split != "train":
            continue
        models[episode.condition].add(episode)
    return models


def contrast_signature(case: BaseCase) -> Tuple[Tuple[str, str], ...]:
    normalized = {key: value for key, value in case.state.items() if key not in STATE_DYNAMIC_KEYS}
    return tuple(sorted(normalized.items()))


def detect_outcome_sensitive_ids(cases: Sequence[BaseCase]) -> List[str]:
    grouped: Dict[Tuple[str, str, str, Tuple[Tuple[str, str], ...]], List[BaseCase]] = defaultdict(list)
    for case in cases:
        bad_operator, _ = parse_action(case.bad_action)
        corrected_operator, _ = parse_action(case.corrected_action)
        if case.query_outcome is None:
            continue
        grouped[(case.family, bad_operator, case.failure_cause, contrast_signature(case))].append(case)
    selected: List[str] = []
    for group_cases in grouped.values():
        corrected_operators = {parse_action(case.corrected_action)[0] for case in group_cases}
        outcomes = {case.query_outcome for case in group_cases}
        if len(corrected_operators) >= 2 and len(outcomes) >= 2:
            for case in group_cases:
                selected.append(case.episode_id)
    return sorted(set(selected))


def evaluate_conditions(
    cases: Sequence[BaseCase], episodes: Sequence[ConditionEpisode]
) -> Tuple[Dict[str, Dict[str, float]], List[dict], Dict[str, object]]:
    heldout_lookup = {
        (episode.condition, episode.episode_id): episode
        for episode in episodes
        if episode.split == "heldout"
    }
    models = build_models(episodes)
    metrics: Dict[str, Dict[str, float]] = {}
    rows: List[dict] = []
    heldout_cases = [case for case in cases if case.split == "heldout"]
    outcome_sensitive_ids = set(detect_outcome_sensitive_ids(heldout_cases))
    for condition in CONDITIONS:
        operator_hits = 0
        binding_hits = 0
        action_hits = 0
        outcome_hits = 0
        cause_hits = 0
        exec_hits = 0
        switch_hits = 0
        for case in heldout_cases:
            episode = heldout_lookup[(condition, case.episode_id)]
            predicted_operator, predicted_outcome, predicted_cause = models[condition].predict(episode)
            predicted_argument = bind_argument_from_state(case)
            predicted_action = build_action(predicted_operator, predicted_argument)
            executed_outcome = execute_action(case, predicted_action)
            success = verify(case, predicted_action, executed_outcome)
            operator_correct = predicted_operator == episode.corrected_operator
            binding_correct = predicted_argument == episode.corrected_argument
            action_exact = operator_correct and binding_correct
            outcome_correct = predicted_outcome == case.corrected_outcome
            cause_correct = predicted_cause == case.failure_cause
            switch_correct = action_exact and predicted_action != case.bad_action
            operator_hits += int(operator_correct)
            binding_hits += int(binding_correct)
            action_hits += int(action_exact)
            outcome_hits += int(outcome_correct)
            cause_hits += int(cause_correct)
            exec_hits += int(success)
            switch_hits += int(switch_correct)
            rows.append(
                {
                    "condition": condition,
                    "episode_id": case.episode_id,
                    "family": case.family,
                    "variant": case.variant,
                    "prompt": case.prompt,
                    "query_outcome": case.query_outcome,
                    "predicted_operator": predicted_operator,
                    "predicted_argument": predicted_argument,
                    "predicted_action": predicted_action,
                    "predicted_outcome": predicted_outcome,
                    "predicted_cause": predicted_cause,
                    "executed_outcome": executed_outcome,
                    "success": success,
                    "expected_operator": episode.corrected_operator,
                    "expected_argument": episode.corrected_argument,
                    "expected_action": case.corrected_action,
                    "expected_outcome": case.corrected_outcome,
                    "expected_cause": case.failure_cause,
                    "bad_operator": episode.bad_operator,
                    "bad_argument": episode.bad_argument,
                    "bad_action": case.bad_action,
                    "operator_correct": operator_correct,
                    "binding_correct": binding_correct,
                    "action_exact_correct": action_exact,
                    "switch_correct": switch_correct,
                    "outcome_sensitive": case.episode_id in outcome_sensitive_ids,
                }
            )
        count = max(1, len(heldout_cases))
        metrics[condition] = {
            "executable_success": exec_hits / count,
            "operator_accuracy": operator_hits / count,
            "binding_accuracy": binding_hits / count,
            "action_exact_accuracy": action_hits / count,
            "outcome_accuracy": outcome_hits / count,
            "switch_accuracy": switch_hits / count,
            "cause_accuracy": cause_hits / count,
        }

    sensitivity_rows = [row for row in rows if row["outcome_sensitive"]]
    per_condition_rows: Dict[str, List[dict]] = defaultdict(list)
    for row in sensitivity_rows:
        per_condition_rows[row["condition"]].append(row)
    sensitivity_summary = {
        "pair_case_count": len(outcome_sensitive_ids),
        "scored_case_count": len(sensitivity_rows),
        "conditions": {},
    }
    for condition in CONDITIONS:
        subset = per_condition_rows.get(condition, [])
        count = max(1, len(subset))
        sensitivity_summary["conditions"][condition] = {
            "operator_accuracy": sum(int(row["operator_correct"]) for row in subset) / count,
            "action_exact_accuracy": sum(int(row["action_exact_correct"]) for row in subset) / count,
            "executable_success": sum(int(row["success"]) for row in subset) / count,
        }
    iote_rows = {row["episode_id"]: row for row in per_condition_rows.get("IOTE", [])}
    no_outcome_rows = {row["episode_id"]: row for row in per_condition_rows.get("IOTE_no_outcome", [])}
    shared_ids = sorted(set(iote_rows) & set(no_outcome_rows))
    prediction_change_rate = 0.0
    if shared_ids:
        prediction_change_rate = sum(
            int(iote_rows[episode_id]["predicted_operator"] != no_outcome_rows[episode_id]["predicted_operator"])
            for episode_id in shared_ids
        ) / len(shared_ids)
    sensitivity_summary["prediction_change_rate_iote_vs_no_outcome"] = prediction_change_rate
    sensitivity_summary["shared_case_count_iote_vs_no_outcome"] = len(shared_ids)
    return metrics, rows, sensitivity_summary


def aggregate_costs(episodes: Sequence[ConditionEpisode]) -> Dict[str, float]:
    totals = {condition: 0.0 for condition in CONDITIONS}
    for episode in episodes:
        if episode.split == "train":
            totals[episode.condition] += episode.total_cost
    return totals


def monotone_cause_relation(condition_metrics: Dict[str, Dict[str, float]]) -> bool:
    pairs = [
        ("R_matched", "IOTE_no_outcome"),
        ("IOTE_no_outcome", "IOTE"),
    ]
    for left, right in pairs:
        if condition_metrics[right]["cause_accuracy"] + EPSILON < condition_metrics[left]["cause_accuracy"]:
            return False
        if condition_metrics[right]["executable_success"] + EPSILON < condition_metrics[left]["executable_success"]:
            return False
    return True


def build_visible_label_audit(episodes: Sequence[ConditionEpisode]) -> Tuple[Dict[str, object], List[str]]:
    per_condition: Dict[str, dict] = {}
    violations: List[str] = []
    for condition in CONDITIONS:
        train_episodes = [episode for episode in episodes if episode.condition == condition and episode.split == "train"]
        visible_fields = visible_fields_for_condition(condition)
        feature_fields = sorted({field for episode in train_episodes for field in episode.feature_fields})
        label_sources: Dict[str, Counter[str]] = {
            "operator": Counter(),
            "outcome": Counter(),
            "cause": Counter(),
        }
        sample_labels: Dict[str, List[str]] = {
            "operator": [],
            "outcome": [],
            "cause": [],
        }
        for episode in train_episodes:
            for label_name in ("operator", "outcome", "cause"):
                value = episode.visible_label_values[label_name]
                source = episode.visible_label_sources[label_name]
                if value is None:
                    continue
                if source is None:
                    violations.append(
                        f"{condition}:{episode.episode_id}:{label_name}: visible label has no source declaration."
                    )
                    continue
                label_sources[label_name][source] += 1
                if value not in sample_labels[label_name]:
                    sample_labels[label_name].append(value)
            if episode.visible_label_values["operator"] is not None and not (
                episode.visible_fields["corrected_action"]
                or episode.visible_label_sources["operator"] == "prose_operator_hint"
            ):
                violations.append(
                    f"{condition}:{episode.episode_id}: operator label is not visible in this arm."
                )
            if episode.visible_label_values["outcome"] is not None and not episode.visible_fields["corrected_outcome"]:
                violations.append(f"{condition}:{episode.episode_id}: outcome label leaked into a hidden arm.")
            if episode.visible_label_values["cause"] is not None and not episode.visible_fields["failure_cause"]:
                violations.append(f"{condition}:{episode.episode_id}: cause label leaked into a hidden arm.")
        per_condition[condition] = {
            "visible_fields": visible_fields,
            "feature_fields": feature_fields,
            "eligible_training_labels": {
                label_name: sorted(value for value in sample_labels[label_name])
                for label_name in sample_labels
            },
            "label_sources": {label_name: dict(counter) for label_name, counter in label_sources.items()},
            "train_episode_count": len(train_episodes),
        }
    audit = {
        "conditions": per_condition,
        "hidden_label_violations": violations,
    }
    return audit, violations


def compute_collapse_audit(
    cases: Sequence[BaseCase],
    condition_metrics: Dict[str, Dict[str, float]],
    same_schedule: bool,
) -> Dict[str, object]:
    heldout_families = {case.family for case in cases if case.split == "heldout"}
    heldout_variants = {(case.family, case.variant) for case in cases if case.split == "heldout"}
    iote_exec = condition_metrics["IOTE"]["executable_success"]
    matched_exec = condition_metrics["R_matched"]["executable_success"]
    no_outcome_exec = condition_metrics["IOTE_no_outcome"]["executable_success"]
    flags = {
        "generic_tool_use_collapse": {
            "active": False,
            "reason": "Rows contain explicit bad_action, bad_outcome, corrected_action, corrected_outcome, and verifier fields.",
        },
        "f1_repair_collapse": {
            "active": iote_exec <= matched_exec + EPSILON,
            "reason": "Full IOTE must beat repaired text on executable success under the same heldout split.",
        },
        "f2_memory_collapse": {
            "active": len(heldout_families) < 4 or len(heldout_variants) < 8,
            "reason": "Heldout split must require state-dependent switching across multiple families and variants.",
        },
        "f3_branch_repair_collapse": {
            "active": False,
            "reason": "The fixture executes bad and corrected actions instead of scoring textual alternatives only.",
        },
        "f4_source_conflict_collapse": {
            "active": False,
            "reason": "The fixture covers stateful operational actions, not source-choice labels.",
        },
        "evidence_demand_collapse": {
            "active": False,
            "reason": "Actions are concrete executor operations, not ask/defer/evidence-demand responses.",
        },
        "cdgt_schedule_collapse": {
            "active": not same_schedule,
            "reason": "All conditions must share the same train/heldout schedule and differ only in teaching content.",
        },
    }
    extra = {
        "iote_no_outcome_matches_full_iote": abs(iote_exec - no_outcome_exec) <= 0.02,
        "full_iote_beats_matched_exec": iote_exec > matched_exec,
        "full_iote_beats_no_outcome_exec": iote_exec > no_outcome_exec,
        "same_schedule": same_schedule,
    }
    active = sorted(name for name, payload in flags.items() if payload["active"])
    return {"flags": flags, "active_flags": active, "extra_checks": extra}


def derive_finding_decision(
    metrics: Dict[str, Optional[float]],
    condition_metrics: Dict[str, Dict[str, float]],
    collapse_audit: Dict[str, object],
) -> Tuple[str, str, List[str], List[str]]:
    weak_reasons: List[str] = []
    kill_reasons: List[str] = []
    pass_reasons: List[str] = []

    active_flags = collapse_audit["active_flags"]
    iote_exec = condition_metrics["IOTE"]["executable_success"]
    matched_exec = condition_metrics["R_matched"]["executable_success"]
    no_outcome_exec = condition_metrics["IOTE_no_outcome"]["executable_success"]
    exec_gain = iote_exec - matched_exec
    err_reduction = relative_error_reduction(matched_exec, iote_exec)
    cause_monotone = bool(metrics["cause_monotone_relation"])

    a_iote = metrics["A_IOTE"]
    a_iote_matched = metrics["A_IOTE_matched"]

    pass_ratio_plain = a_iote is not None and a_iote >= PASS_THRESHOLDS["a_iote"]
    pass_ratio_matched = a_iote_matched is not None and a_iote_matched >= PASS_THRESHOLDS["a_iote_matched"]
    pass_exec_gain = exec_gain >= PASS_THRESHOLDS["exec_gain_abs"] or (
        err_reduction is not None and err_reduction >= PASS_THRESHOLDS["exec_error_reduction"]
    )
    pass_delta_align = metrics["Delta_align"] is not None and metrics["Delta_align"] >= PASS_THRESHOLDS["delta_align"]
    pass_delta_cause = metrics["Delta_cause"] is not None and (
        metrics["Delta_cause"] >= PASS_THRESHOLDS["delta_cause"] or cause_monotone
    )
    pass_delta_exec = metrics["Delta_exec"] is not None and metrics["Delta_exec"] > 0.0

    if a_iote is None and (metrics["G_R"] is None or metrics["G_R"] <= 0.0):
        kill_reasons.append("A_IOTE is null because repaired-text plain gain is non-positive after overhead.")
    if a_iote_matched is None and (metrics["G_M"] is None or metrics["G_M"] <= 0.0) and not pass_exec_gain:
        kill_reasons.append("A_IOTE_matched is null and absolute executable gain does not clear pass floors.")
    if a_iote is not None and a_iote < WEAK_THRESHOLDS["a_iote_low"]:
        kill_reasons.append("A_IOTE is below the kill floor after measured overhead.")
    if a_iote_matched is not None and a_iote_matched <= WEAK_THRESHOLDS["a_iote_matched_low"]:
        kill_reasons.append("A_IOTE_matched is at or below the kill floor against matched repaired text.")
    if iote_exec <= matched_exec + EPSILON:
        kill_reasons.append("IOTE does not improve heldout executable success over R_matched.")
    if abs(iote_exec - no_outcome_exec) <= 0.02:
        kill_reasons.append("IOTE_no_outcome matches full IOTE, so executed consequences are not load-bearing.")
    if active_flags:
        kill_reasons.append("Collapse audit has active flags: " + ", ".join(active_flags))

    if not kill_reasons:
        if a_iote is None or not pass_ratio_plain:
            weak_reasons.append("A_IOTE does not clear the pass ratio against R_plain.")
        if a_iote_matched is None or not pass_ratio_matched:
            weak_reasons.append("A_IOTE_matched does not clear the pass ratio against R_matched.")
        if not pass_exec_gain:
            weak_reasons.append("Heldout executable success gain over R_matched is below the pass floor.")
        if not pass_delta_exec:
            weak_reasons.append("Delta_exec is not positive.")
        if not pass_delta_align:
            weak_reasons.append("Delta_align is below the pass floor.")
        if not pass_delta_cause:
            weak_reasons.append("Delta_cause is below the pass floor and lacks a monotone relation.")
        if iote_exec <= no_outcome_exec + EPSILON:
            weak_reasons.append("IOTE does not beat IOTE_no_outcome strongly enough.")

    if not kill_reasons and not weak_reasons:
        pass_reasons.extend(
            [
                "IOTE clears both cost-normalized ratio gates.",
                "Heldout executable success beats matched repaired text by the required margin.",
                "Structured executed consequences beat the no-outcome ablation.",
                "Operator, action, and cause metrics clear the mechanistic pass floors.",
            ]
        )
        return "pass", "; ".join(pass_reasons), weak_reasons, kill_reasons

    if kill_reasons:
        return "kill", "; ".join(kill_reasons), weak_reasons, kill_reasons
    return "weak", "; ".join(weak_reasons), weak_reasons, kill_reasons


def build_validity_gates(
    episodes: Sequence[ConditionEpisode],
    condition_metrics: Dict[str, Dict[str, float]],
    metrics: Dict[str, Optional[float]],
    visible_label_audit: Dict[str, object],
    outcome_sensitivity: Dict[str, object],
) -> Tuple[Dict[str, object], str, List[str]]:
    hidden_violations = list(visible_label_audit["hidden_label_violations"])
    binding_values = [condition_metrics[condition]["binding_accuracy"] for condition in CONDITIONS]
    nondegenerate_primary = any(
        condition_metrics[condition]["executable_success"] > 0.0
        or condition_metrics[condition]["operator_accuracy"] > 0.0
        or condition_metrics[condition]["action_exact_accuracy"] > 0.0
        for condition in CONDITIONS
    )
    gates = {
        "activation_valid": {
            "valid": True,
            "reason": "The probe emits an explicit activation line and writes manifest artifacts for every run.",
        },
        "label_audit_valid": {
            "valid": not hidden_violations,
            "reason": "No hidden action/outcome/cause labels leaked into arms that cannot see them."
            if not hidden_violations
            else "Hidden-label leakage was detected in the visibility audit.",
        },
        "binding_valid": {
            "valid": all(value >= 1.0 - EPSILON for value in binding_values),
            "reason": "The arm-neutral binder reconstructs the held-out argument directly from visible state."
            if all(value >= 1.0 - EPSILON for value in binding_values)
            else "The arm-neutral binder failed to recover held-out arguments for one or more conditions.",
        },
        "outcome_identifying_valid": {
            "valid": bool(outcome_sensitivity["pair_case_count"] > 0 and outcome_sensitivity["shared_case_count_iote_vs_no_outcome"] > 0),
            "reason": "Outcome-sensitive contrast cases were present and scored across full IOTE and the no-outcome ablation."
            if outcome_sensitivity["pair_case_count"] > 0 and outcome_sensitivity["shared_case_count_iote_vs_no_outcome"] > 0
            else "No scored outcome-sensitive contrast exists for comparing IOTE with the no-outcome ablation.",
        },
        "nondegenerate_valid": {
            "valid": nondegenerate_primary,
            "reason": "Primary operator/action/executable metrics are not all zero across every condition."
            if nondegenerate_primary
            else "Primary operator/action/executable metrics collapsed to all zero across every condition.",
        },
    }
    invalid_reasons = [payload["reason"] for payload in gates.values() if not payload["valid"]]
    status = "valid" if not invalid_reasons else "invalid_probe"
    if status == "valid" and metrics["A_IOTE"] is None and metrics["A_IOTE_matched"] is None:
        invalid_reasons.append("Both ratio gates are null; the corrected probe still lacks a cost-normalized comparison.")
        gates["nondegenerate_valid"] = {
            "valid": False,
            "reason": "Both ratio gates are null; the corrected probe still lacks a cost-normalized comparison.",
        }
        status = "invalid_probe"
    return gates, status, invalid_reasons


def main() -> int:
    args = parse_args()
    if args.executor_overhead < 0.0:
        raise ValueError("--executor-overhead must be non-negative.")
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.self_test:
        cases = build_self_test_fixture(args.seed)
        input_label = "self-test"
        mode_label = "self_test"
    else:
        cases = build_cases_from_input(load_jsonl(args.input))
        input_label = str(args.input)
        mode_label = "input"

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive when provided.")
        cases = list(cases[: args.limit])

    train_count = sum(1 for case in cases if case.split == "train")
    heldout_count = sum(1 for case in cases if case.split == "heldout")
    if train_count == 0 or heldout_count == 0:
        raise ValueError("The selected cases must include both train and heldout splits.")

    episodes = build_condition_episodes(cases, args.token_cost_mode, args.executor_overhead)
    condition_metrics, heldout_rows, outcome_sensitivity = evaluate_conditions(cases, episodes)
    executable_success_before, before_rows = evaluate_before([case for case in cases if case.split == "heldout"])
    heldout_predictions = before_rows + heldout_rows
    costs = aggregate_costs(episodes)
    visible_label_audit, hidden_violations = build_visible_label_audit(episodes)

    g_i = (condition_metrics["IOTE"]["executable_success"] - executable_success_before) / max(EPSILON, costs["IOTE"])
    g_r = (condition_metrics["R_plain"]["executable_success"] - executable_success_before) / max(EPSILON, costs["R_plain"])
    g_m = (condition_metrics["R_matched"]["executable_success"] - executable_success_before) / max(
        EPSILON, costs["R_matched"]
    )
    a_iote = safe_ratio(g_i, g_r) if g_r > 0.0 else None
    a_iote_matched = safe_ratio(g_i, g_m) if g_m > 0.0 else None
    delta_exec = g_i - g_m
    delta_align = (
        condition_metrics["IOTE"]["action_exact_accuracy"] - condition_metrics["R_matched"]["action_exact_accuracy"]
    )
    delta_cause = condition_metrics["IOTE"]["cause_accuracy"] - condition_metrics["R_matched"]["cause_accuracy"]
    same_schedule = True
    cause_monotone = monotone_cause_relation(condition_metrics)
    collapse_audit = compute_collapse_audit(cases, condition_metrics, same_schedule)

    metrics: Dict[str, Optional[float]] = {
        "cost_iote": costs["IOTE"],
        "cost_r_plain": costs["R_plain"],
        "cost_r_matched": costs["R_matched"],
        "cost_iote_over_r_plain": safe_ratio(costs["IOTE"], costs["R_plain"]),
        "cost_iote_over_r_matched": safe_ratio(costs["IOTE"], costs["R_matched"]),
        "executable_success_before": executable_success_before,
        "executable_success_iote": condition_metrics["IOTE"]["executable_success"],
        "executable_success_r_plain": condition_metrics["R_plain"]["executable_success"],
        "executable_success_r_matched": condition_metrics["R_matched"]["executable_success"],
        "operator_accuracy_iote": condition_metrics["IOTE"]["operator_accuracy"],
        "operator_accuracy_r_plain": condition_metrics["R_plain"]["operator_accuracy"],
        "operator_accuracy_r_matched": condition_metrics["R_matched"]["operator_accuracy"],
        "binding_accuracy_iote": condition_metrics["IOTE"]["binding_accuracy"],
        "binding_accuracy_r_matched": condition_metrics["R_matched"]["binding_accuracy"],
        "action_exact_accuracy_iote": condition_metrics["IOTE"]["action_exact_accuracy"],
        "action_exact_accuracy_r_matched": condition_metrics["R_matched"]["action_exact_accuracy"],
        "outcome_accuracy_iote": condition_metrics["IOTE"]["outcome_accuracy"],
        "outcome_accuracy_r_matched": condition_metrics["R_matched"]["outcome_accuracy"],
        "switch_accuracy_iote": condition_metrics["IOTE"]["switch_accuracy"],
        "switch_accuracy_r_matched": condition_metrics["R_matched"]["switch_accuracy"],
        "cause_accuracy_iote": condition_metrics["IOTE"]["cause_accuracy"],
        "cause_accuracy_r_matched": condition_metrics["R_matched"]["cause_accuracy"],
        "g_iote_no_outcome": (
            (condition_metrics["IOTE_no_outcome"]["executable_success"] - executable_success_before)
            / max(EPSILON, costs["IOTE_no_outcome"])
        ),
        "G_I": g_i,
        "G_R": g_r,
        "G_M": g_m,
        "A_IOTE": a_iote,
        "A_IOTE_matched": a_iote_matched,
        "Delta_exec": delta_exec,
        "Delta_align": delta_align,
        "Delta_cause": delta_cause,
        "cause_monotone_relation": 1.0 if cause_monotone else 0.0,
        "matched_relative_error_reduction": relative_error_reduction(
            condition_metrics["R_matched"]["executable_success"],
            condition_metrics["IOTE"]["executable_success"],
        ),
        "iote_minus_no_outcome_exec": (
            condition_metrics["IOTE"]["executable_success"] - condition_metrics["IOTE_no_outcome"]["executable_success"]
        ),
        "outcome_prediction_change_rate": outcome_sensitivity["prediction_change_rate_iote_vs_no_outcome"],
    }

    validity_gates, validity_status, invalidity_reasons = build_validity_gates(
        episodes,
        condition_metrics,
        metrics,
        visible_label_audit,
        outcome_sensitivity,
    )

    if validity_status == "valid":
        finding_decision, finding_reason, weak_reasons, kill_reasons = derive_finding_decision(
            metrics,
            condition_metrics,
            collapse_audit,
        )
        decision_eligible = True
    else:
        finding_decision = "blocked_invalid_probe"
        finding_reason = "; ".join(invalidity_reasons)
        weak_reasons = []
        kill_reasons = []
        decision_eligible = False

    activation_line = (
        "R390_IOTE_ACTIVATION"
        f" mode={mode_label}"
        f" seed={args.seed}"
        f" out={out_dir}"
        f" input={input_label}"
        f" token_cost_mode={args.token_cost_mode}"
        f" executor_overhead={args.executor_overhead:.6g}"
    )

    manifest = {
        "run": {
            "mode": mode_label,
            "input": input_label,
            "out": str(out_dir),
            "seed": args.seed,
            "limit": args.limit,
            "executor_overhead": args.executor_overhead,
            "token_cost_mode": args.token_cost_mode,
            "self_test": bool(args.self_test),
        },
        "counts": {
            "base_case_count": len(cases),
            "train_count": train_count,
            "heldout_count": heldout_count,
            "condition_count": len(CONDITIONS),
            "episode_count": len(episodes),
        },
        "thresholds": {
            "pass": PASS_THRESHOLDS,
            "weak": WEAK_THRESHOLDS,
        },
        "metrics": {key: rounded(value) for key, value in metrics.items()},
        "condition_metrics": {
            condition: {metric: rounded(value) for metric, value in values.items()}
            for condition, values in condition_metrics.items()
        },
        "visible_label_audit": visible_label_audit,
        "outcome_sensitivity": outcome_sensitivity,
        "validity_gates": validity_gates,
        "validity_status": validity_status,
        "invalidity_reasons": invalidity_reasons,
        "finding_decision_eligible": decision_eligible,
        "finding_decision": finding_decision,
        "finding_reason": finding_reason,
        "weak_reasons": weak_reasons,
        "kill_reasons": kill_reasons,
        "collapse_audit": collapse_audit,
        "activation_line": activation_line,
        "notes": [
            "Each arm trains only on labels that are visible in its rendered episode text or explicitly recoverable from visible prose.",
            "Primary executable evaluation predicts the operator and binds the held-out argument with an arm-neutral state binder.",
            "Invalid probes must not be interpreted as finding kills or passes.",
        ],
    }

    metrics_json = {
        "validity_status": validity_status,
        "finding_decision_eligible": decision_eligible,
        "finding_decision": finding_decision,
        "finding_reason": finding_reason,
        "metrics": manifest["metrics"],
        "condition_metrics": manifest["condition_metrics"],
    }

    episode_rows = [
        {
            "episode_id": episode.episode_id,
            "split": episode.split,
            "condition": episode.condition,
            "family": episode.family,
            "variant": episode.variant,
            "prompt": episode.prompt,
            "item_key": episode.item_key,
            "query_outcome": episode.query_outcome,
            "state_text": episode.state_text,
            "query_text": episode.query_text,
            "memory_text": episode.memory_text,
            "corrected_action": episode.corrected_action,
            "corrected_operator": episode.corrected_operator,
            "corrected_argument": episode.corrected_argument,
            "corrected_outcome": episode.corrected_outcome,
            "failure_cause": episode.failure_cause,
            "bad_action": episode.bad_action,
            "bad_operator": episode.bad_operator,
            "bad_argument": episode.bad_argument,
            "bad_outcome": episode.bad_outcome,
            "verifier_id": episode.verifier_id,
            "correction_delta": episode.correction_delta,
            "token_cost": rounded(episode.token_cost),
            "overhead_cost": rounded(episode.overhead_cost),
            "total_cost": rounded(episode.total_cost),
            "visible_fields": episode.visible_fields,
            "visible_label_values": episode.visible_label_values,
            "visible_label_sources": episode.visible_label_sources,
            "feature_fields": episode.feature_fields,
        }
        for episode in episodes
    ]

    readme_text = "\n".join(
        [
            "# R390 IOTE Probe",
            "",
            "This directory contains one offline diagnostic run for Intervention-Outcome Teaching Episodes.",
            "",
            f"- Validity status: `{validity_status}`",
            f"- Finding decision eligible: `{str(decision_eligible).lower()}`",
            f"- Finding decision: `{finding_decision}`",
            f"- Reason: {finding_reason}",
            "",
            "Interpretation:",
            "- `validity_status=valid` is required before any finding-level pass/weak/kill conclusion is meaningful.",
            "- `blocked_invalid_probe` means the probe needs correction or broader evidence before R390 can be closed or advanced.",
            "",
            "Primary files:",
            "- `manifest.json`: full inputs, metrics, visible-label audit, validity gates, and decision eligibility.",
            "- `episodes.jsonl`: generated or loaded condition rows with rendered visibility and label sources.",
            "- `heldout_predictions.jsonl`: per-heldout predictions, bound arguments, and executed outcomes.",
            "- `visible_label_audit.json`: arm-by-arm visibility and hidden-label leakage checks.",
            "- `metrics.json`: compact metric summary.",
        ]
    )

    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "metrics.json", metrics_json)
    write_json(out_dir / "visible_label_audit.json", visible_label_audit)
    write_json(out_dir / "outcome_sensitivity.json", outcome_sensitivity)
    write_jsonl(out_dir / "episodes.jsonl", episode_rows)
    write_jsonl(out_dir / "heldout_predictions.jsonl", heldout_predictions)
    (out_dir / "README.md").write_text(readme_text + "\n", encoding="utf-8")

    print(activation_line)
    print(
        "R390_IOTE_VALIDITY"
        f" status={validity_status}"
        f" label_audit_valid={str(validity_gates['label_audit_valid']['valid']).lower()}"
        f" binding_valid={str(validity_gates['binding_valid']['valid']).lower()}"
        f" outcome_identifying_valid={str(validity_gates['outcome_identifying_valid']['valid']).lower()}"
        f" nondegenerate_valid={str(validity_gates['nondegenerate_valid']['valid']).lower()}"
    )
    print(
        "R390_IOTE_METRICS"
        f" A_IOTE={rounded(a_iote) if a_iote is not None else 'null'}"
        f" A_IOTE_matched={rounded(a_iote_matched) if a_iote_matched is not None else 'null'}"
        f" Delta_exec={rounded(delta_exec)}"
        f" Delta_align={rounded(delta_align)}"
        f" Delta_cause={rounded(delta_cause)}"
    )
    print(f"R390_IOTE_DECISION {finding_decision} reason={finding_reason}")
    if hidden_violations:
        for violation in hidden_violations:
            print(f"R390_IOTE_AUDIT_VIOLATION {violation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
