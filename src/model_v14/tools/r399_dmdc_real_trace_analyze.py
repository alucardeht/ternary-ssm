import argparse
import json
import math
from pathlib import Path


EPS = 1.0e-12
BIAS_TOL = 2.0
PREFIX_TOL = 3.0
ANALYZER_VERSION = "r399.dmdc.real_trace.analyze.v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def classify(metrics):
    if not metrics["activation_line_verified"]:
        return "INVALID", "missing_activation_line"
    if metrics["synthetic_fixture"]:
        return "INVALID", "synthetic_fixture_true"
    if not metrics["gpu_owned_path"]:
        return "INVALID", "gpu_owned_path_false"
    if metrics["batch_size"] > 4:
        return "INVALID", "batch_size_gt_4"
    if metrics["steps_observed"] < 2:
        return "INVALID", "steps_lt_2"
    if metrics["useful_records"] < 1024:
        return "INVALID", "insufficient_useful_records"
    if metrics["family_useful"].get("W_in", 0) == 0:
        return "INVALID", "missing_W_in_useful_records"
    if metrics["family_useful"].get("W_out", 0) == 0:
        return "INVALID", "missing_W_out_useful_records"
    if not (metrics["random_noise_energy"] > EPS):
        return "INVALID", "zero_random_noise_denominator"

    finite_keys = ("f_q", "rho", "h", "lhs", "bias_abs", "max_prefix_discrepancy")
    if any(not math.isfinite(metrics[key]) for key in finite_keys):
        return "INVALID", "non_finite_metric"
    if not (0.0 <= metrics["f_q"] <= 1.0) or metrics["rho"] < 0.0 or metrics["h"] <= 0.0:
        return "INVALID", "metric_out_of_range"
    if metrics["bias_abs"] > BIAS_TOL or metrics["max_prefix_discrepancy"] > PREFIX_TOL:
        return "FAIL", "bias_detected"
    if metrics["lhs"] <= 0.50:
        return "PASS", "passed_gate"

    gate_base = 1.0 - (metrics["f_q"] * (1.0 - metrics["rho"]))
    required_fq = float("inf")
    allowed_rho = float("inf")
    allowed_h = float("inf")
    if (1.0 - metrics["rho"]) > EPS:
        required_fq = (1.0 - (0.50 / (1.0 + metrics["h"]))) / (1.0 - metrics["rho"])
    if metrics["f_q"] > EPS:
        allowed_rho = 1.0 - ((1.0 - (0.50 / (1.0 + metrics["h"]))) / metrics["f_q"])
    if gate_base > EPS:
        allowed_h = (0.50 / gate_base) - 1.0

    base_gate_passes = math.isfinite(gate_base) and gate_base <= 0.50
    if base_gate_passes and math.isfinite(allowed_h) and allowed_h >= 0.0 and metrics["h"] > allowed_h:
        return "FAIL", "overhead_too_high"
    if metrics["rho"] >= 1.0 or (math.isfinite(allowed_rho) and allowed_rho >= 0.0 and metrics["rho"] > allowed_rho):
        return "FAIL", "rho_too_high"
    if (
        not base_gate_passes
        and math.isfinite(required_fq)
        and 0.0 <= required_fq <= 1.0
        and metrics["f_q"] < required_fq
    ):
        return "FAIL", "insufficient_f_q"
    return "FAIL", "lhs_gt_0_50"


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    meta = None
    trace_summary = None
    steps = set()
    families_observed = set()
    family_useful = {"W_in": 0, "W_out": 0}
    records = 0
    useful_records = 0
    random_noise_energy = 0.0
    det_noise_energy = 0.0
    useful_signal_energy = 0.0
    useful_weight_sum = 0.0
    det_noise_weighted_sum = 0.0
    prefix_cumulative = 0.0
    max_prefix_discrepancy = 0.0

    with input_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            kind = item.get("kind")
            if kind == "meta":
                meta = item
            elif kind == "record":
                records += 1
                family = item.get("family", "unknown")
                families_observed.add(family)
                steps.add(int(item.get("step", 0)))
                eligible = bool(item.get("eligible", False))
                if eligible:
                    useful_records += 1
                    if family in family_useful:
                        family_useful[family] += 1
                    useful_weight = safe_float(item.get("useful_weight"))
                    det_noise = safe_float(item.get("det_noise"))
                    random_noise_energy += safe_float(item.get("random_noise_energy"))
                    det_noise_energy += safe_float(item.get("det_noise_energy"))
                    useful_signal_energy += safe_float(item.get("useful_signal_energy"))
                    useful_weight_sum += useful_weight
                    det_noise_weighted_sum += useful_weight * det_noise
                    prefix_cumulative += useful_weight * det_noise
                    if random_noise_energy > EPS:
                        prefix_norm = abs(prefix_cumulative) / math.sqrt(random_noise_energy + EPS)
                        max_prefix_discrepancy = max(max_prefix_discrepancy, prefix_norm)
            elif kind == "summary":
                trace_summary = item

    batch_size = int(meta.get("batch_size", 0)) if meta else 0
    seq_len = int(meta.get("seq_len", 0)) if meta else 0
    gpu_owned_path = bool(meta.get("gpu_owned_path", False)) if meta else False
    synthetic_fixture = bool(meta.get("synthetic_fixture", True)) if meta else True
    activation_line_verified = bool(meta.get("activation_line_emitted", False)) if meta else False

    if useful_weight_sum > EPS and random_noise_energy > EPS:
        bias_scale = math.sqrt(random_noise_energy / useful_weight_sum)
        bias_abs = abs(det_noise_weighted_sum / useful_weight_sum) / bias_scale if bias_scale > EPS else float("nan")
    else:
        bias_abs = float("nan")

    denom = random_noise_energy + useful_signal_energy
    f_q = random_noise_energy / denom if denom > EPS else float("nan")
    rho = det_noise_energy / random_noise_energy if random_noise_energy > EPS else float("nan")

    if trace_summary is not None:
        h_work_unit_proxy = safe_float(trace_summary.get("h_work_unit_proxy"))
        h_wall_proxy = safe_float(trace_summary.get("h_wall_proxy"))
    else:
        h_work_unit_proxy = 1.0 if records > 0 else float("nan")
        h_wall_proxy = 0.0

    if math.isfinite(h_work_unit_proxy):
        h = max(h_work_unit_proxy, h_wall_proxy if math.isfinite(h_wall_proxy) else 0.0)
    else:
        h = float("nan")
    lhs = (1.0 - f_q * (1.0 - rho)) * (1.0 + h) if all(math.isfinite(x) for x in (f_q, rho, h)) else float("nan")

    metrics = {
        "analyzer_version": ANALYZER_VERSION,
        "input_path": str(input_path),
        "activation_line_verified": activation_line_verified,
        "schema": meta.get("schema") if meta else None,
        "mode": meta.get("mode") if meta else None,
        "source": meta.get("source") if meta else None,
        "scope": meta.get("scope") if meta else None,
        "synthetic_fixture": synthetic_fixture,
        "gpu_owned_path": gpu_owned_path,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "max_steps": int(meta.get("max_steps", 0)) if meta else 0,
        "sample_per_family": int(meta.get("sample_per_family", 0)) if meta else 0,
        "steps_observed": len(steps),
        "records": records,
        "useful_records": useful_records,
        "families_observed": sorted(families_observed),
        "family_useful": family_useful,
        "random_noise_energy": random_noise_energy,
        "det_noise_energy": det_noise_energy,
        "useful_signal_energy": useful_signal_energy,
        "f_q": f_q,
        "rho": rho,
        "h": h,
        "h_work_unit_proxy": h_work_unit_proxy,
        "h_wall_proxy": h_wall_proxy,
        "lhs": lhs,
        "bias_abs": bias_abs,
        "max_prefix_discrepancy": max_prefix_discrepancy,
    }

    decision, reason = classify(metrics)
    metrics["decision"] = decision
    metrics["reason"] = reason

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    lhs_text = "nan" if not math.isfinite(metrics["lhs"]) else f"{metrics['lhs']:.9g}"
    fq_text = "nan" if not math.isfinite(metrics["f_q"]) else f"{metrics['f_q']:.9g}"
    rho_text = "nan" if not math.isfinite(metrics["rho"]) else f"{metrics['rho']:.9g}"
    h_text = "nan" if not math.isfinite(metrics["h"]) else f"{metrics['h']:.9g}"
    print(
        f"[R399_DMDC_TRACE_ANALYZE] decision={decision} lhs={lhs_text} "
        f"f_q={fq_text} rho={rho_text} h={h_text} reason={reason}"
    )


if __name__ == "__main__":
    main()
