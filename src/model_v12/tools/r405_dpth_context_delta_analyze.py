import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


ANALYZER_VERSION = "r405_dpth_context_delta_analyze.v1"
EPS = 1.0e-12
TAU_RATIO_THRESHOLD = 0.45
ORACLE_OVER_CONTROL_THRESHOLD = 1.25


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-contexts-per-candidate", type=int, default=32)
    return parser.parse_args()


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def lambda_proxy(values):
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) * (x - mean) for x in values) / len(values)
    if mean <= 0.0:
        return 0.0
    return mean * (mean * mean / (mean * mean + var + EPS))


def same_sign_persistence(values):
    signs = [1 if x > 0.0 else -1 if x < 0.0 else 0 for x in values]
    pairs = [(a, b) for a, b in zip(signs, signs[1:]) if a != 0 and b != 0]
    if not pairs:
        return float("nan")
    return sum(1 for a, b in pairs if a == b) / len(pairs)


def candidate_key(item):
    return (
        item.get("family"),
        int(item.get("layer", -1)),
        int(item.get("flat_index", -1)),
        int(item.get("alternative_value", 0)),
    )


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    meta = None
    duplicate_meta_rows = 0
    grouped = defaultdict(list)
    with input_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            kind = item.get("kind")
            if kind == "meta":
                if meta is None:
                    meta = item
                else:
                    duplicate_meta_rows += 1
            elif kind == "context_delta":
                delta = safe_float(item.get("delta_ce"))
                baseline_ce = safe_float(item.get("baseline_ce"))
                if math.isfinite(delta) and math.isfinite(baseline_ce):
                    grouped[candidate_key(item)].append(
                        {
                            "delta": delta,
                            "baseline_ce": baseline_ce,
                            "context": int(item.get("context", -1)),
                        }
                    )

    candidate_metrics = []
    for key, rows in grouped.items():
        if len(rows) < args.min_contexts_per_candidate:
            continue
        rows = sorted(rows, key=lambda row: row["context"])
        deltas = [row["delta"] for row in rows]
        positives = [x for x in deltas if x > 0.0]
        if not positives:
            continue
        baseline_lambda = lambda_proxy(deltas)
        oracle_lambda = lambda_proxy(positives)
        tau_ratio = baseline_lambda / oracle_lambda if oracle_lambda > EPS else float("inf")
        baseline_persistence = same_sign_persistence(deltas)
        oracle_persistence = same_sign_persistence(positives)

        positive_count = len(positives)
        loss_control_rows = sorted(rows, key=lambda row: row["baseline_ce"], reverse=True)[:positive_count]
        loss_control_lambda = lambda_proxy([row["delta"] for row in loss_control_rows])
        oracle_over_loss = (
            oracle_lambda / loss_control_lambda
            if loss_control_lambda > EPS
            else float("inf")
        )
        value = sum(positives)
        candidate_metrics.append(
            {
                "family": key[0],
                "layer": key[1],
                "flat_index": key[2],
                "alternative_value": key[3],
                "contexts": len(rows),
                "positive_fraction": positive_count / len(rows),
                "positive_value": value,
                "baseline_lambda": baseline_lambda,
                "oracle_packet_lambda": oracle_lambda,
                "tau_packet_over_tau_baseline": tau_ratio,
                "baseline_sign_persistence": baseline_persistence,
                "oracle_sign_persistence": oracle_persistence,
                "loss_control_lambda": loss_control_lambda,
                "oracle_over_loss_control": oracle_over_loss,
            }
        )

    total_value = sum(max(0.0, item["positive_value"]) for item in candidate_metrics)
    if total_value > 0.0:
        weighted_tau = sum(
            item["tau_packet_over_tau_baseline"] * max(0.0, item["positive_value"])
            for item in candidate_metrics
            if math.isfinite(item["tau_packet_over_tau_baseline"])
        ) / total_value
        weighted_oracle_over_loss = sum(
            min(item["oracle_over_loss_control"], 10.0) * max(0.0, item["positive_value"])
            for item in candidate_metrics
            if math.isfinite(item["oracle_over_loss_control"])
        ) / total_value
    else:
        weighted_tau = float("nan")
        weighted_oracle_over_loss = float("nan")

    pass_tau = math.isfinite(weighted_tau) and weighted_tau <= TAU_RATIO_THRESHOLD
    pass_control = (
        math.isfinite(weighted_oracle_over_loss)
        and weighted_oracle_over_loss >= ORACLE_OVER_CONTROL_THRESHOLD
    )

    if meta is None:
        decision = "INVALID_PROBE"
        reason = "missing_meta"
    elif duplicate_meta_rows:
        decision = "INVALID_PROBE"
        reason = "duplicate_meta"
    elif not bool(meta.get("per_context_deltas")):
        decision = "INVALID_PROBE"
        reason = "missing_per_context_deltas"
    elif not candidate_metrics:
        decision = "INVALID_PROBE"
        reason = "no_candidate_denominator"
    elif pass_tau and pass_control:
        decision = "ORACLE_BOUND_PASSES_NEEDS_PACKET_ARM"
        reason = "oracle_context_packet_bound_clears_pretraining_gate"
    else:
        decision = "FAILED_PRETRAINING_GATE"
        reason = "oracle_context_packet_bound_below_thresholds"

    metrics = {
        "analyzer_version": ANALYZER_VERSION,
        "input_path": str(input_path),
        "meta_present": meta is not None,
        "duplicate_meta_rows": duplicate_meta_rows,
        "contexts": int(meta.get("contexts", 0)) if meta else 0,
        "candidate_count": len(candidate_metrics),
        "total_positive_value": total_value,
        "weighted_tau_packet_over_tau_baseline": weighted_tau,
        "weighted_oracle_over_loss_control": weighted_oracle_over_loss,
        "tau_ratio_threshold": TAU_RATIO_THRESHOLD,
        "oracle_over_control_threshold": ORACLE_OVER_CONTROL_THRESHOLD,
        "decision": decision,
        "reason": reason,
        "top_candidates": sorted(
            candidate_metrics,
            key=lambda item: item["positive_value"],
            reverse=True,
        )[:20],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    def fmt(value):
        return "nan" if not math.isfinite(value) else f"{value:.9g}"

    print(
        f"[R405_DPTH_CONTEXT_DELTA_ANALYZE] decision={decision} "
        f"candidates={len(candidate_metrics)} "
        f"weighted_tau={fmt(weighted_tau)} "
        f"oracle_over_loss={fmt(weighted_oracle_over_loss)} "
        f"reason={reason}"
    )


if __name__ == "__main__":
    main()
