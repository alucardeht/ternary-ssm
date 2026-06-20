#!/usr/bin/env python3
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


SCHEMA = "r400.common_observability.v1"
RECORD_REQUIRED_FIELDS = (
    "kind",
    "schema",
    "run_id",
    "step",
    "microbatch",
    "layer",
    "seq_index",
    "horizon_k",
    "matrix_family",
    "objective_kind",
    "coord_index",
    "alignment_key",
    "object_kind",
    "producer_stage",
    "consumer_stage",
    "source_name",
    "source_valid",
    "shape",
    "sample_count",
    "l2",
    "nonzero_count",
    "finite_count",
    "zero_reason",
)
META_REQUIRED_FIELDS = (
    "kind",
    "schema",
    "mode",
    "source",
    "no_training_behavior_change",
    "activation_line_emitted",
    "max_steps",
    "max_records",
    "sample",
    "horizon_k",
    "batch_size",
)
SUMMARY_REQUIRED_FIELDS = (
    "kind",
    "schema",
    "records",
    "aligned_groups",
    "live_state_groups",
    "live_transition_groups",
    "live_boundary_groups",
    "live_objective_grad_groups",
    "decision",
    "reason",
)
OBJECTIVE_KINDS = {
    "credit_dA_ps",
    "lm_transition_grad_sketch",
    "bootstrap_transition_grad_sketch",
}
BOUNDARY_KINDS = {
    "boundary_shadow_pre",
    "boundary_shadow_post",
    "ternary_sample_mask",
    "ternary_q_mask",
}
EXPLICIT_ABSENCE_REASONS = {
    "source_missing",
    "not_produced",
    "objective_missing",
    "not_aligned",
}


def fail(reason: str) -> int:
    print(f"decision=FAIL_INFRASTRUCTURE reason={reason}")
    return 1


def is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"malformed_json_line_{line_no}: {exc.msg}") from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"non_object_line_{line_no}")
                rows.append(payload)
    except OSError as exc:
        raise ValueError(f"read_failed: {exc}") from exc
    return rows


def validate_required_fields(payload: dict, required_fields: tuple[str, ...], label: str) -> None:
    for field in required_fields:
        if field not in payload:
            raise ValueError(f"{label}_missing_{field}")


def validate_meta(meta: dict) -> None:
    validate_required_fields(meta, META_REQUIRED_FIELDS, "meta")
    if meta["kind"] != "meta":
        raise ValueError("meta_kind_invalid")
    if meta["schema"] != SCHEMA:
        raise ValueError("meta_schema_mismatch")
    if meta["mode"] != "diagnostic-only":
        raise ValueError("meta_mode_invalid")
    if meta["source"] != "real-gpu-training-observability":
        raise ValueError("meta_source_invalid")
    if meta["no_training_behavior_change"] is not True:
        raise ValueError("meta_behavior_change_detected")
    if meta["activation_line_emitted"] is not True:
        raise ValueError("meta_activation_missing")
    for key in ("max_steps", "max_records", "sample", "horizon_k", "batch_size"):
        if not isinstance(meta[key], int) or meta[key] <= 0:
            raise ValueError(f"meta_invalid_{key}")


def validate_record(record: dict) -> None:
    validate_required_fields(record, RECORD_REQUIRED_FIELDS, "record")
    if record["kind"] != "record":
        raise ValueError("record_kind_invalid")
    if record["schema"] != SCHEMA:
        raise ValueError("record_schema_mismatch")
    if not isinstance(record["run_id"], str) or not record["run_id"]:
        raise ValueError("record_invalid_run_id")
    for key in ("step", "microbatch", "layer", "seq_index", "horizon_k", "coord_index", "sample_count"):
        if not isinstance(record[key], int):
            raise ValueError(f"record_invalid_{key}")
    for key in (
        "matrix_family",
        "objective_kind",
        "alignment_key",
        "object_kind",
        "producer_stage",
        "consumer_stage",
        "source_name",
        "shape",
        "zero_reason",
    ):
        if not isinstance(record[key], str) or not record[key]:
            raise ValueError(f"record_invalid_{key}")
    if not isinstance(record["source_valid"], bool):
        raise ValueError("record_invalid_source_valid")
    if not is_finite_number(record["l2"]) or float(record["l2"]) < 0.0:
        raise ValueError("record_invalid_l2")
    for key in ("nonzero_count", "finite_count"):
        if not isinstance(record[key], int) or record[key] < 0:
            raise ValueError(f"record_invalid_{key}")
    if record["nonzero_count"] > record["finite_count"]:
        raise ValueError("record_nonzero_exceeds_finite")


def validate_summary(summary: dict, records: list[dict], meta: dict) -> None:
    validate_required_fields(summary, SUMMARY_REQUIRED_FIELDS, "summary")
    if summary["kind"] != "summary":
        raise ValueError("summary_kind_invalid")
    if summary["schema"] != SCHEMA:
        raise ValueError("summary_schema_mismatch")
    if not isinstance(summary["decision"], str) or not isinstance(summary["reason"], str):
        raise ValueError("summary_invalid_decision_reason")
    for key in SUMMARY_REQUIRED_FIELDS[2:8]:
        if not isinstance(summary[key], int) or summary[key] < 0:
            raise ValueError(f"summary_invalid_{key}")
    if summary["records"] != len(records):
        raise ValueError("summary_records_mismatch")
    if summary["records"] > meta["max_records"]:
        raise ValueError("record_cap_violation")


def record_is_live(record: dict) -> bool:
    return record["source_valid"] and record["l2"] > 0.0 and record["nonzero_count"] > 0


def record_is_explicit_absence(record: dict) -> bool:
    return (not record["source_valid"]) and record["zero_reason"] in EXPLICIT_ABSENCE_REASONS


def classify(path: Path) -> int:
    try:
        rows = load_jsonl(path)
    except ValueError as exc:
        return fail(str(exc))

    meta_rows = [row for row in rows if row.get("kind") == "meta"]
    summary_rows = [row for row in rows if row.get("kind") == "summary"]
    record_rows = [row for row in rows if row.get("kind") == "record"]

    if len(meta_rows) != 1:
        return fail("meta_count_invalid")
    if len(summary_rows) != 1:
        return fail("summary_count_invalid")

    try:
        validate_meta(meta_rows[0])
        for record in record_rows:
            validate_record(record)
        validate_summary(summary_rows[0], record_rows, meta_rows[0])
    except ValueError as exc:
        return fail(str(exc))

    if not record_rows:
        print("decision=NON_IDENTIFYING_REAL_PATH reason=no_records")
        return 0

    groups = defaultdict(list)
    for record in record_rows:
        groups[record["alignment_key"]].append(record)

    live_state_group = False
    live_transition_group = False
    objective_evidence_group = False
    boundary_evidence_group = False

    for group_records in groups.values():
        object_map = {record["object_kind"]: record for record in group_records}

        state_record = object_map.get("state_current")
        transition_record = object_map.get("transition_A")
        if state_record and record_is_live(state_record):
            live_state_group = True
        if transition_record and record_is_live(transition_record):
            live_transition_group = True

        for object_kind in OBJECTIVE_KINDS:
            record = object_map.get(object_kind)
            if not record:
                continue
            if record_is_live(record) or record_is_explicit_absence(record):
                objective_evidence_group = True
                break

        boundary_absence = any(
            record["object_kind"] in BOUNDARY_KINDS and record_is_explicit_absence(record)
            for record in group_records
        )
        if boundary_absence:
            boundary_evidence_group = True
        else:
            boundary_pre = object_map.get("boundary_shadow_pre")
            boundary_post = object_map.get("boundary_shadow_post")
            boundary_sample = object_map.get("ternary_sample_mask")
            boundary_q = object_map.get("ternary_q_mask")
            if (
                boundary_pre
                and boundary_post
                and record_is_live(boundary_pre)
                and record_is_live(boundary_post)
                and (
                    (boundary_sample and boundary_sample["source_valid"])
                    or (boundary_q and boundary_q["source_valid"])
                )
            ):
                boundary_evidence_group = True

    summary = summary_rows[0]
    if summary["aligned_groups"] != len(groups):
        return fail("summary_aligned_groups_mismatch")

    if live_state_group and live_transition_group and objective_evidence_group and boundary_evidence_group:
        print("decision=PASS_IDENTIFYING_INFRASTRUCTURE reason=aligned_real_path_observability")
        return 0
    if not live_state_group or not live_transition_group:
        print("decision=NON_IDENTIFYING_REAL_PATH reason=missing_live_state_or_transition")
        return 0
    if not objective_evidence_group:
        print("decision=NON_IDENTIFYING_REAL_PATH reason=missing_credit_or_objective_grad")
        return 0
    if not boundary_evidence_group:
        print("decision=NON_IDENTIFYING_REAL_PATH reason=missing_boundary_alignment")
        return 0

    return fail("unclassified_groups")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python src/model_v12/tools/r400_observability_analyze.py <trace.jsonl>", file=sys.stderr)
        return 1
    path = Path(argv[1])
    if not path.is_file():
        return fail("trace_path_missing")
    return classify(path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
