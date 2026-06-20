#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path


SCHEMA = "r400.cchc.horizon_observability.v1"
SOURCE = "real-ssm-gpu-backward"
TAIL_THRESHOLD = 0.05
ALIGNMENT_THRESHOLD = 0.99
REQUIRED_ARRAY_KEYS = (
    "state_energy_by_offset",
    "transition_energy_by_offset",
    "adjoint_energy_by_offset",
    "credit_energy_by_offset",
)
REQUIRED_SCALAR_KEYS = (
    "state_l2",
    "transition_l2",
    "adjoint_l2",
    "credit_l2",
)
REQUIRED_NONZERO_KEYS = (
    "state_nonzero",
    "transition_nonzero",
    "adjoint_nonzero",
    "credit_nonzero",
)


def fail_invalid(reason: str) -> int:
    print(f"classification=invalid_probe reason={reason}")
    return 1


def fail_blocked(reason: str) -> int:
    print(f"classification=blocked_by_non_identifying_real_path reason={reason}")
    return 2


def is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def load_jsonl(path: Path):
    records = []
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
                records.append(payload)
    except OSError as exc:
        raise ValueError(f"read_failed: {exc}") from exc
    return records


def validate_record(record):
    if record.get("schema") != SCHEMA:
        raise ValueError("schema_mismatch")
    if record.get("source") != SOURCE:
        raise ValueError("unsupported_source")
    if "state_source_valid" in record and not isinstance(record.get("state_source_valid"), bool):
        raise ValueError("invalid_state_source_valid")
    if "state_source" in record and not isinstance(record.get("state_source"), str):
        raise ValueError("invalid_state_source")

    seq_len = record.get("seq_len")
    batch_size = record.get("batch_size")
    d_state = record.get("d_state")
    if not isinstance(seq_len, int) or seq_len <= 0:
        raise ValueError("invalid_seq_len")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("invalid_batch_size")
    if not isinstance(d_state, int) or d_state <= 0:
        raise ValueError("invalid_d_state")

    for key in REQUIRED_SCALAR_KEYS:
        value = record.get(key)
        if not is_finite_number(value):
            raise ValueError(f"non_finite_{key}")
        if float(value) < 0.0:
            raise ValueError(f"negative_{key}")

    for key in REQUIRED_NONZERO_KEYS:
        value = record.get(key)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid_{key}")

    for key in REQUIRED_ARRAY_KEYS:
        values = record.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"missing_{key}")
        if len(values) != seq_len:
            raise ValueError(f"inconsistent_{key}_length")
        for value in values:
            if not is_finite_number(value):
                raise ValueError(f"non_finite_{key}")
            if float(value) < 0.0:
                raise ValueError(f"negative_{key}")

    tail_mass_by_k = record.get("tail_mass_by_k")
    if not isinstance(tail_mass_by_k, dict):
        raise ValueError("missing_tail_mass_by_k")
    for key, value in tail_mass_by_k.items():
        try:
            k_value = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid_tail_mass_key") from exc
        if k_value < 0 or k_value >= seq_len:
            raise ValueError("invalid_tail_mass_horizon")
        if not is_finite_number(value):
            raise ValueError("non_finite_tail_mass")
        if float(value) < 0.0:
            raise ValueError("negative_tail_mass")

    alignment = record.get("alignment_cosine_dAps_vs_dh_hprev")
    if alignment is not None and not is_finite_number(alignment):
        raise ValueError("non_finite_alignment")


def classify(path: Path) -> int:
    try:
        rows = load_jsonl(path)
    except ValueError as exc:
        return fail_invalid(str(exc))

    meta = None
    records = []
    for row in rows:
        kind = row.get("kind")
        if kind == "meta":
            meta = row
        elif kind == "record":
            records.append(row)

    if meta is None:
        return fail_blocked("meta_missing")
    if meta.get("schema") != SCHEMA:
        return fail_invalid("meta_schema_mismatch")
    if meta.get("source") != SOURCE:
        return fail_invalid("meta_source_mismatch")

    if not records:
        return fail_blocked("no_real_gpu_ssm_backward_records")

    try:
        for record in records:
            validate_record(record)
    except ValueError as exc:
        reason = str(exc)
        invalid_prefixes = ("schema_", "unsupported_source", "invalid_", "inconsistent_", "negative_")
        if reason.startswith(invalid_prefixes):
            return fail_invalid(reason)
        return fail_blocked(reason)

    non_identifying_reasons = []
    compressible_horizon_found = False

    for record in records:
        if record.get("state_source_valid") is False:
            return fail_blocked("state_source_unavailable")
        state_source = record.get("state_source")
        if state_source is not None and state_source != "layer.fwd_h_cache":
            return fail_invalid("unsupported_state_source")

        for key in REQUIRED_SCALAR_KEYS:
            if float(record[key]) <= 0.0:
                non_identifying_reasons.append(f"zero_{key}")
        for key in REQUIRED_NONZERO_KEYS:
            if record[key] <= 0:
                non_identifying_reasons.append(f"zero_{key}")

        alignment = record.get("alignment_cosine_dAps_vs_dh_hprev")
        if alignment is not None and float(alignment) < ALIGNMENT_THRESHOLD:
            non_identifying_reasons.append("alignment_below_threshold")

        if record.get("nan_or_inf_count", 0) > 0:
            non_identifying_reasons.append("nan_or_inf_present")

        for value in record["tail_mass_by_k"].values():
            if float(value) <= TAIL_THRESHOLD:
                compressible_horizon_found = True
                break

    if non_identifying_reasons:
        return fail_blocked(non_identifying_reasons[0])

    if compressible_horizon_found:
        print("classification=observability_pass reason=aligned_real_path_horizon_observability")
        return 0

    print("classification=observability_identified_no_compressible_horizon_yet reason=no_tail_mass_below_threshold")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python src/model_v12/tools/r400_cchc_analyze.py <trace.jsonl>", file=sys.stderr)
        return 1

    path = Path(argv[1])
    if not path.is_file():
        return fail_invalid("trace_path_missing")
    return classify(path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
