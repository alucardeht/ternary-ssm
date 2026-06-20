#!/usr/bin/env python3
"""Offline reducer scaffold for R395 semigroup-closure trace reports.

Current binary trace contract:
- magic: ``R395SGT1``
- version: ``1``
- header layout: ``<8sIIIII``
- record layout: ``<IIfff``
- payload shape per record: ``[n_layers, batch_size, seq_len, d_state]`` float32
  in C order
- report schema: ``r395.sg.report.v1``
"""

import json
import math
import os
import struct
import sys
from array import array


TRACE_MAGIC = b"R395SGT1"
TRACE_HEADER = struct.Struct("<8sIIIII")
TRACE_RECORD = struct.Struct("<IIfff")
EPS = 1.0e-12
FLOAT32_SIZE = 4


def read_trace(path):
    with open(path, "rb") as handle:
        header_blob = handle.read(TRACE_HEADER.size)
        if len(header_blob) != TRACE_HEADER.size:
            raise ValueError("trace header is truncated")
        magic, version, n_layers, batch_size, seq_len, d_state = TRACE_HEADER.unpack(header_blob)
        if magic != TRACE_MAGIC:
            raise ValueError("unexpected trace magic")
        if version != 1:
            raise ValueError(f"unsupported trace version: {version}")
        if any(value == 0 for value in (n_layers, batch_size, seq_len, d_state)):
            raise ValueError("trace dimensions must be non-zero")

        state_count = n_layers * batch_size * seq_len * d_state
        payload_size = state_count * FLOAT32_SIZE
        record_size = TRACE_RECORD.size + payload_size
        trailing_blob = handle.read()
        if len(trailing_blob) % record_size != 0:
            raise ValueError("trace payload is truncated")

        records = []
        offset = 0
        while offset < len(trailing_blob):
            record_blob = trailing_blob[offset : offset + TRACE_RECORD.size]
            step, _reserved, step_loss, runtime_ms, export_ms = TRACE_RECORD.unpack(record_blob)
            payload_blob_start = offset + TRACE_RECORD.size
            payload_blob_end = payload_blob_start + payload_size
            payload_blob = trailing_blob[payload_blob_start:payload_blob_end]
            if len(payload_blob) != payload_size:
                raise ValueError("trace payload is truncated")
            payload = array("f")
            payload.frombytes(payload_blob)
            records.append(
                {
                    "step": step,
                    "step_loss": float(step_loss),
                    "runtime_ms": float(runtime_ms),
                    "export_ms": float(export_ms),
                    "states": payload,
                }
            )
            offset += record_size

    return {
        "n_layers": n_layers,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_state": d_state,
        "records": records,
    }


def state_offset(meta, layer, batch, t, i):
    ds = meta["d_state"]
    seq_len = meta["seq_len"]
    batch_size = meta["batch_size"]
    return (((layer * batch_size) + batch) * seq_len + t) * ds + i


def fit_alpha(payload, meta, layer, batch, horizon, prefix_len):
    ds = meta["d_state"]
    numer = [0.0] * ds
    denom = [0.0] * ds
    limit = prefix_len - horizon
    for t in range(limit):
        base_x = state_offset(meta, layer, batch, t, 0)
        base_y = state_offset(meta, layer, batch, t + horizon, 0)
        for i in range(ds):
            x = payload[base_x + i]
            y = payload[base_y + i]
            numer[i] += x * y
            denom[i] += x * x
    return [numer[i] / denom[i] if denom[i] > EPS else 0.0 for i in range(ds)]


def fit_alpha_shuffle(payload, meta, layer, batch, horizon, prefix_len):
    ds = meta["d_state"]
    numer = [0.0] * ds
    denom = [0.0] * ds
    limit = prefix_len - horizon
    for t in range(limit):
        shuffled_t = ((t * 7) + 3) % limit
        base_x = state_offset(meta, layer, batch, t, 0)
        base_y = state_offset(meta, layer, batch, shuffled_t + horizon, 0)
        for i in range(ds):
            x = payload[base_x + i]
            y = payload[base_y + i]
            numer[i] += x * y
            denom[i] += x * x
    return [numer[i] / denom[i] if denom[i] > EPS else 0.0 for i in range(ds)]


def fit_alpha_time_shift(payload, meta, layer, batch, horizon, prefix_len):
    ds = meta["d_state"]
    numer = [0.0] * ds
    denom = [0.0] * ds
    limit = prefix_len - horizon - 1
    for t in range(limit):
        base_x = state_offset(meta, layer, batch, t, 0)
        base_y = state_offset(meta, layer, batch, t + horizon + 1, 0)
        for i in range(ds):
            x = payload[base_x + i]
            y = payload[base_y + i]
            numer[i] += x * y
            denom[i] += x * x
    return [numer[i] / denom[i] if denom[i] > EPS else 0.0 for i in range(ds)]


def eval_alpha(payload, meta, layer, batch, horizon, alpha, eval_start):
    ds = meta["d_state"]
    seq_len = meta["seq_len"]
    numer = 0.0
    denom = 0.0
    samples = 0
    for t in range(eval_start, seq_len - horizon):
        base_x = state_offset(meta, layer, batch, t, 0)
        base_y = state_offset(meta, layer, batch, t + horizon, 0)
        for i in range(ds):
            x = payload[base_x + i]
            target = payload[base_y + i]
            pred = alpha[i] * x
            diff = target - pred
            numer += diff * diff
            denom += target * target
            samples += 1
    if samples == 0 or denom <= EPS:
        return None
    return numer / denom


def composed_alpha(alpha_a, alpha_b):
    return [a * b for a, b in zip(alpha_a, alpha_b)]


def closure_materiality(alpha_direct, alpha_comp):
    numer = 0.0
    denom = 0.0
    for direct_value, comp_value in zip(alpha_direct, alpha_comp):
        numer += abs(direct_value - comp_value)
        denom += abs(direct_value)
    return numer / max(denom, EPS)


def analyze_record(meta, record):
    seq_len = meta["seq_len"]
    payload = record["states"]
    prefix_len = (seq_len * 2) // 3
    eval_start = prefix_len
    max_horizon = min(4, prefix_len // 3, (seq_len - eval_start - 1))
    invalid_reasons = []

    if max_horizon < 2:
        invalid_reasons.append("insufficient_horizons")
        return None, invalid_reasons

    pair_errors = []
    pair_materiality = []
    pair_lifts = []

    for layer in range(meta["n_layers"]):
        for batch in range(meta["batch_size"]):
            direct = {}
            shuffle = {}
            shifted = {}
            for horizon in range(1, max_horizon + 1):
                direct[horizon] = fit_alpha(payload, meta, layer, batch, horizon, prefix_len)
                shuffle[horizon] = fit_alpha_shuffle(payload, meta, layer, batch, horizon, prefix_len)
                shifted[horizon] = fit_alpha_time_shift(payload, meta, layer, batch, horizon, prefix_len)

            for a, b in ((1, 1), (1, 2), (2, 1), (2, 2)):
                horizon = a + b
                if horizon > max_horizon:
                    continue
                direct_err = eval_alpha(payload, meta, layer, batch, horizon, direct[horizon], eval_start)
                composed_err = eval_alpha(
                    payload,
                    meta,
                    layer,
                    batch,
                    horizon,
                    composed_alpha(direct[a], direct[b]),
                    eval_start,
                )
                shuffle_err = eval_alpha(
                    payload,
                    meta,
                    layer,
                    batch,
                    horizon,
                    composed_alpha(direct[a], shuffle[b]),
                    eval_start,
                )
                shift_err = eval_alpha(payload, meta, layer, batch, horizon, shifted[horizon], eval_start)

                if None in (direct_err, composed_err, shuffle_err, shift_err):
                    continue
                if not all(math.isfinite(value) for value in (direct_err, composed_err, shuffle_err, shift_err)):
                    continue

                materiality = closure_materiality(direct[horizon], composed_alpha(direct[a], direct[b]))
                control_err = min(shuffle_err, shift_err)
                lift = (control_err - composed_err) / max(control_err, EPS)

                pair_errors.append((direct_err, composed_err, shuffle_err, shift_err))
                pair_materiality.append(materiality)
                pair_lifts.append(lift)

    if not pair_errors:
        invalid_reasons.append("insufficient_valid_pairs")
        return None, invalid_reasons

    d_direct = sum(item[0] for item in pair_errors) / len(pair_errors)
    d_composed = sum(item[1] for item in pair_errors) / len(pair_errors)
    d_shuffle = sum(item[2] for item in pair_errors) / len(pair_errors)
    d_time_shift = sum(item[3] for item in pair_errors) / len(pair_errors)
    materiality = sum(pair_materiality) / len(pair_materiality)
    future_state_error_lift = sum(pair_lifts) / len(pair_lifts)
    gamma = record["export_ms"] / max(record["runtime_ms"], EPS)
    eta_p_proxy = max(0.0, future_state_error_lift * min(1.0, materiality))
    s_net = (1.0 + eta_p_proxy) / (1.0 + gamma)

    if not all(math.isfinite(value) for value in (d_direct, d_composed, d_shuffle, d_time_shift, materiality, future_state_error_lift, gamma, eta_p_proxy, s_net)):
        invalid_reasons.append("nonfinite_metrics")
    if materiality <= 1.0e-4:
        invalid_reasons.append("all_zero_or_vacuous_defect")

    if invalid_reasons:
        return None, invalid_reasons

    control_survives = d_composed < d_shuffle and d_composed < d_time_shift
    threshold = 0.10 + 1.10 * gamma

    if control_survives and s_net >= 1.10 and eta_p_proxy >= threshold:
        verdict = "PASS"
    else:
        verdict = "KILL"

    return {
        "mode": "trace",
        "validity_status": "valid",
        "invalid_reasons": [],
        "D_direct": d_direct,
        "D_composed": d_composed,
        "D_shuffle": d_shuffle,
        "D_time_shift": d_time_shift,
        "closure_materiality": materiality,
        "future_state_error_lift": future_state_error_lift,
        "loss_progress_proxy": future_state_error_lift,
        "cost_fraction_gamma": gamma,
        "eta_p_proxy": eta_p_proxy,
        "S_net": s_net,
        "verdict": verdict,
        "step": record["step"],
        "step_loss": record["step_loss"],
        "runtime_step_ms": record["runtime_ms"],
        "export_overhead_ms": record["export_ms"],
    }, invalid_reasons


def build_invalid_report(meta, reasons):
    return {
        "schema": "r395.sg.report.v1",
        "mode": "trace",
        "validity_status": "invalid_probe",
        "invalid_reasons": reasons,
        "D_direct": None,
        "D_composed": None,
        "D_shuffle": None,
        "D_time_shift": None,
        "closure_materiality": None,
        "future_state_error_lift": None,
        "loss_progress_proxy": None,
        "cost_fraction_gamma": None,
        "eta_p_proxy": None,
        "S_net": None,
        "verdict": "INVALID",
        "trace_shape": {
            "n_layers": meta["n_layers"],
            "batch_size": meta["batch_size"],
            "seq_len": meta["seq_len"],
            "d_state": meta["d_state"],
            "records": len(meta["records"]),
        },
    }


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main(argv):
    if len(argv) not in (2, 3):
        print("usage: r395_semigroup_closure_probe.py TRACE_PATH [REPORT_PATH]", file=sys.stderr)
        return 2

    trace_path = argv[1]
    report_path = argv[2] if len(argv) == 3 else os.path.join(
        ".work",
        "jobs",
        "r395-semigroup-closure-01",
        "artifacts",
        "r395.sg.report.json",
    )

    meta = read_trace(trace_path)
    if not meta["records"]:
        report = build_invalid_report(meta, ["empty_trace"])
    else:
        report, reasons = analyze_record(meta, meta["records"][-1])
        if report is None:
            report = build_invalid_report(meta, reasons)
        else:
            report["schema"] = "r395.sg.report.v1"
            report["trace_shape"] = {
                "n_layers": meta["n_layers"],
                "batch_size": meta["batch_size"],
                "seq_len": meta["seq_len"],
                "d_state": meta["d_state"],
                "records": len(meta["records"]),
            }

    ensure_parent_dir(report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"[R395_REPORT] validity_status={report['validity_status']} "
        f"verdict={report['verdict']} report={report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
