#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path


SCHEMA = "r400.sofsb.observability_alignment.v1"
SOURCE = "real-ssm-gpu-training"
REQUIRED_INT_POSITIVE = ("seq_len", "batch_size", "d_state", "horizon_k", "sample_count")
REQUIRED_INT_NONNEG = ("step", "layer", "microbatch", "lm_grad_nonzero", "boot_grad_nonzero", "nan_or_inf_count")
REQUIRED_FLOAT_NONNEG = (
    "h_t_l2",
    "h_future_l2",
    "future_residual_l2",
    "lm_loss",
    "bootstrap_loss",
    "lm_grad_l2",
    "boot_grad_l2",
    "lm_grad_variance_proxy",
    "boot_grad_variance_proxy",
)


def print_invalid(reason: str) -> int:
    print(f"classification=invalid_probe reason={reason}")
    return 1


def print_blocked(reason: str) -> int:
    print(f"classification=blocked_by_non_identifying_real_path reason={reason}")
    return 2


def print_failed(reason: str) -> int:
    print(f"classification=failed_alignment_variance_gate reason={reason}")
    return 3


def print_pass(reason: str) -> int:
    print(f"classification=observability_alignment_pass reason={reason}")
    return 0


def is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def load_jsonl(path: Path):
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


def validate_record(record: dict) -> None:
    if record.get("schema") != SCHEMA:
        raise ValueError("schema_mismatch")
    if record.get("source") != SOURCE:
        raise ValueError("unsupported_source")
    if record.get("transition_family") != "A":
        raise ValueError("unsupported_transition_family")
    if record.get("state_source") != "layer.fwd_h_cache":
        raise ValueError("unsupported_state_source")
    if record.get("future_target_source") != "layer.fwd_h_cache.stop_gradient":
        raise ValueError("unsupported_future_target_source")
    if record.get("objective_separated") is not True:
        raise ValueError("objective_not_separated")

    for key in REQUIRED_INT_POSITIVE:
        value = record.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"invalid_{key}")
    for key in REQUIRED_INT_NONNEG:
        value = record.get(key)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid_{key}")
    for key in REQUIRED_FLOAT_NONNEG:
        value = record.get(key)
        if not is_finite_number(value):
            raise ValueError(f"non_finite_{key}")
        if float(value) < 0.0:
            raise ValueError(f"negative_{key}")

    for key in ("grad_dot", "grad_cosine"):
        value = record.get(key)
        if not is_finite_number(value):
            raise ValueError(f"non_finite_{key}")
    if abs(float(record["grad_cosine"])) > 1.000001:
        raise ValueError("invalid_grad_cosine")


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[mid]
    return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])


def summarize(records: list[dict]) -> dict:
    comparable = []
    summary = {
        "records": len(records),
        "comparable_records": 0,
        "state_nonzero_records": 0,
        "future_nonzero_records": 0,
        "lm_grad_nonzero_records": 0,
        "boot_grad_nonzero_records": 0,
        "median_cosine": 0.0,
        "variance_lm": 0.0,
        "variance_boot": 0.0,
        "variance_ratio": 0.0,
    }
    seen_keys = set()

    for record in records:
        key = (
            record["step"],
            record["layer"],
            record["microbatch"],
            record["horizon_k"],
            record["transition_family"],
        )
        if key in seen_keys:
            raise ValueError("duplicate_comparable_key")
        seen_keys.add(key)

        if record["h_t_l2"] > 0.0:
            summary["state_nonzero_records"] += 1
        if record["h_future_l2"] > 0.0:
            summary["future_nonzero_records"] += 1
        if record["lm_grad_l2"] > 0.0 and record["lm_grad_nonzero"] > 0:
            summary["lm_grad_nonzero_records"] += 1
        if record["boot_grad_l2"] > 0.0 and record["boot_grad_nonzero"] > 0:
            summary["boot_grad_nonzero_records"] += 1

        if record["nan_or_inf_count"] > 0:
            continue
        if record["h_t_l2"] <= 0.0:
            continue
        if record["h_future_l2"] <= 0.0:
            continue
        if record["lm_grad_l2"] <= 0.0 or record["lm_grad_nonzero"] <= 0:
            continue
        if record["boot_grad_l2"] <= 0.0 or record["boot_grad_nonzero"] <= 0:
            continue
        comparable.append(record)

    summary["comparable_records"] = len(comparable)
    if comparable:
        summary["median_cosine"] = median([float(r["grad_cosine"]) for r in comparable])
        summary["variance_lm"] = sum(float(r["lm_grad_variance_proxy"]) for r in comparable) / len(comparable)
        summary["variance_boot"] = sum(float(r["boot_grad_variance_proxy"]) for r in comparable) / len(comparable)
        if summary["variance_lm"] > 0.0:
            summary["variance_ratio"] = summary["variance_boot"] / summary["variance_lm"]
    return summary


def validate_summary(last_summary: dict | None, computed: dict) -> None:
    if last_summary is None:
        raise ValueError("summary_missing")
    if last_summary.get("schema") != SCHEMA:
        raise ValueError("summary_schema_mismatch")
    for key in (
        "records",
        "comparable_records",
        "state_nonzero_records",
        "future_nonzero_records",
        "lm_grad_nonzero_records",
        "boot_grad_nonzero_records",
    ):
        if int(last_summary.get(key, -1)) != int(computed[key]):
            raise ValueError(f"summary_contradicts_{key}")
    for key in ("median_cosine", "variance_lm", "variance_boot", "variance_ratio"):
        value = last_summary.get(key)
        if not is_finite_number(value):
            raise ValueError(f"summary_non_finite_{key}")
        if key != "median_cosine" and float(value) < 0.0:
            raise ValueError(f"summary_negative_{key}")
        if abs(float(value) - float(computed[key])) > 1.0e-6:
            raise ValueError(f"summary_contradicts_{key}")


def classify(path: Path) -> int:
    try:
        rows = load_jsonl(path)
    except ValueError as exc:
        return print_invalid(str(exc))

    meta = None
    records = []
    last_summary = None
    for row in rows:
        kind = row.get("kind")
        if kind == "meta":
            meta = row
        elif kind == "record":
            records.append(row)
        elif kind == "summary":
            last_summary = row
        else:
            return print_invalid("unknown_kind")

    if meta is None:
        return print_invalid("meta_missing")
    if meta.get("schema") != SCHEMA:
        return print_invalid("meta_schema_mismatch")
    if meta.get("source") != SOURCE:
        return print_invalid("meta_source_mismatch")
    if meta.get("no_training_behavior_change") != 1:
        return print_invalid("meta_behavior_mismatch")
    if meta.get("full_equal_wall_clock") is not False:
        return print_invalid("meta_equal_wall_clock_mismatch")
    if not records:
        return print_blocked("no_records")

    try:
        for record in records:
            validate_record(record)
        computed = summarize(records)
        validate_summary(last_summary, computed)
    except ValueError as exc:
        reason = str(exc)
        invalid_prefixes = (
            "malformed_",
            "non_object_",
            "schema_",
            "unsupported_",
            "invalid_",
            "negative_",
            "duplicate_",
            "summary_",
        )
        if reason.startswith(invalid_prefixes):
            return print_invalid(reason)
        return print_blocked(reason)

    if computed["comparable_records"] < 8:
        return print_blocked("comparable_records_below_minimum")
    if computed["state_nonzero_records"] != computed["records"]:
        return print_blocked("state_observability_missing")
    if computed["future_nonzero_records"] != computed["records"]:
        return print_blocked("future_target_observability_missing")
    if computed["lm_grad_nonzero_records"] != computed["records"]:
        return print_blocked("lm_transition_grad_missing")
    if computed["boot_grad_nonzero_records"] != computed["records"]:
        return print_blocked("bootstrap_transition_grad_missing")

    if computed["median_cosine"] >= 0.20 and computed["variance_ratio"] <= 0.50:
        return print_pass("observability_alignment_pass")

    return print_failed(
        "median_cosine_below_threshold"
        if computed["median_cosine"] < 0.20
        else "variance_ratio_above_threshold"
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python src/model_v12/tools/r400_sofsb_analyze.py <trace.jsonl>", file=sys.stderr)
        return 1
    path = Path(argv[1])
    if not path.is_file():
        return print_invalid("trace_path_missing")
    return classify(path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
