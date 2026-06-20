import argparse
import json
import math
from pathlib import Path


ANALYZER_VERSION = "r405_tcfvl_analyze.v1"
EPS = 1.0e-12
DELTA_EPS = 1.0e-6
PREVALENCE_THRESHOLD = 0.05
PRECISION_RATIO_THRESHOLD = 2.0
AUC_THRESHOLD = 0.65
INTERACTION_HARM_THRESHOLD = 0.25
PRICED_RATIO_THRESHOLD = 1.15


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


def score_record(record):
    family = record.get("family")
    family_bias = {
        "W_in": 0.03,
        "W_out": 0.02,
        "lm_head": 0.01,
    }.get(family, 0.0)
    current_value = safe_float(record.get("current_value"))
    alternative_value = safe_float(record.get("alternative_value"))
    abs_shadow = safe_float(record.get("abs_shadow"))
    layer = int(record.get("layer", -1))
    return abs_shadow + 0.10 * abs(alternative_value - current_value) + family_bias + (0.001 * layer if layer >= 0 else 0.0)


def auc_from_scores(records):
    positives = [r for r in records if r["label"]]
    negatives = [r for r in records if not r["label"]]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    total = 0.0
    for pos in positives:
        for neg in negatives:
            total += 1.0
            if pos["score"] > neg["score"]:
                wins += 1.0
            elif pos["score"] == neg["score"]:
                wins += 0.5
    return wins / total if total > 0.0 else float("nan")


def classify(metrics):
    if not metrics["meta_present"]:
        return "INVALID_PROBE", "missing_meta_row"
    if metrics["duplicate_meta_rows"] > 0:
        return "INVALID_PROBE", "duplicate_meta_row"
    if not metrics["activation_flag"]:
        return "INVALID_PROBE", "missing_activation_flag"
    if metrics["batch_size"] > 4:
        return "INVALID_PROBE", "batch_size_gt_4"
    if metrics["ce_mode"] != "full":
        return "INVALID_PROBE", "non_full_ce_mode"
    if metrics["record_count"] == 0:
        return "INVALID_PROBE", "no_records"
    if metrics["restore_failures"] > 0:
        return "INVALID_PROBE", "restore_failure_observed"
    if metrics["supported_family_count"] == 0:
        return "BLOCKED_BY_NON_IDENTIFYING_REAL_PATH", "no_supported_families"
    if metrics["non_finite_records"] > 0:
        return "INVALID_PROBE", "non_finite_record"
    if metrics["all_zero_deltas"]:
        return "INVALID_PROBE", "all_zero_deltas"
    if metrics["positive_count"] == 0 or metrics["negative_count"] == 0:
        return "INVALID_PROBE", "missing_positive_or_negative_denominator"
    if not math.isfinite(metrics["interaction_harm"]):
        return "INVALID_PROBE", "missing_interaction_row"
    if not math.isfinite(metrics["priced_ratio"]):
        return "INVALID_PROBE", "no_priced_baseline_denominator"

    passes_prevalence = metrics["useful_delta_prevalence"] >= PREVALENCE_THRESHOLD
    passes_ranking = (
        metrics["precision_top10_ratio"] >= PRECISION_RATIO_THRESHOLD
        or metrics["auc"] >= AUC_THRESHOLD
    )
    passes_interaction = metrics["interaction_harm"] < INTERACTION_HARM_THRESHOLD
    passes_priced = metrics["priced_ratio"] > PRICED_RATIO_THRESHOLD

    if passes_prevalence and passes_ranking and passes_interaction and passes_priced:
        return "PASS_IDENTIFYING_GATE", "passed_all_thresholds"

    near_prevalence = metrics["useful_delta_prevalence"] >= 0.8 * PREVALENCE_THRESHOLD
    near_ranking = (
        metrics["precision_top10_ratio"] >= 0.8 * PRECISION_RATIO_THRESHOLD
        or metrics["auc"] >= 0.9 * AUC_THRESHOLD
    )
    near_interaction = metrics["interaction_harm"] < 0.30
    near_priced = metrics["priced_ratio"] > 0.9 * PRICED_RATIO_THRESHOLD
    if metrics["positive_count"] > 0 and (near_prevalence or near_ranking or near_interaction or near_priced):
        return "WEAK_SIGNAL", "finite_but_below_thresholds"
    return "FAILED_IDENTIFYING_GATE", "finite_identifying_failure"


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    meta = None
    duplicate_meta_rows = 0
    summary = None
    records = []
    interactions = {}

    with input_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            kind = item.get("kind")
            if kind == "meta":
                if meta is not None:
                    duplicate_meta_rows += 1
                    continue
                meta = item
            elif kind == "record":
                delta_ce = safe_float(item.get("delta_ce"))
                flipped_ce = safe_float(item.get("flipped_ce"))
                baseline_ce = safe_float(item.get("baseline_ce"))
                score = score_record(item)
                label = math.isfinite(delta_ce) and delta_ce > DELTA_EPS
                records.append(
                    {
                        "raw": item,
                        "delta_ce": delta_ce,
                        "flipped_ce": flipped_ce,
                        "baseline_ce": baseline_ce,
                        "score": score,
                        "label": label,
                        "finite": all(math.isfinite(x) for x in (delta_ce, flipped_ce, baseline_ce, score)),
                    }
                )
            elif kind == "interaction":
                interactions[item.get("selection")] = item
            elif kind == "summary":
                summary = item

    valid_records = [r for r in records if r["finite"]]
    sorted_records = sorted(valid_records, key=lambda r: (r["score"], r["delta_ce"]), reverse=True)
    top10 = sorted_records[:10]
    positive_count = sum(1 for r in valid_records if r["label"])
    negative_count = sum(1 for r in valid_records if not r["label"])
    positive_rate = (positive_count / len(valid_records)) if valid_records else float("nan")
    precision_top10 = (
        sum(1 for r in top10 if r["label"]) / len(top10)
        if top10
        else float("nan")
    )
    precision_ratio = (
        precision_top10 / positive_rate
        if positive_rate and math.isfinite(positive_rate) and positive_rate > EPS and math.isfinite(precision_top10)
        else float("nan")
    )
    auc = auc_from_scores(valid_records)

    score_interaction = interactions.get("score_topk", {})
    random_interaction = interactions.get("random_baseline", {})
    interaction_harm = safe_float(score_interaction.get("interaction_harm"))
    score_joint = safe_float(score_interaction.get("joint_delta"))
    random_joint = safe_float(random_interaction.get("joint_delta"))
    priced_ratio = score_joint / random_joint if math.isfinite(score_joint) and math.isfinite(random_joint) and abs(random_joint) > EPS else float("nan")

    metrics = {
        "analyzer_version": ANALYZER_VERSION,
        "input_path": str(input_path),
        "meta_present": meta is not None,
        "activation_flag": bool(meta.get("activation_flag")) if meta else False,
        "ce_mode": meta.get("ce_mode") if meta else None,
        "batch_size": int(meta.get("batch_size", 0)) if meta else 0,
        "contexts": int(meta.get("contexts", 0)) if meta else 0,
        "duplicate_meta_rows": duplicate_meta_rows,
        "record_count": len(records),
        "valid_record_count": len(valid_records),
        "non_finite_records": sum(1 for r in records if not r["finite"]),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "all_zero_deltas": len(valid_records) > 0 and all(abs(r["delta_ce"]) <= DELTA_EPS for r in valid_records),
        "useful_delta_prevalence": (positive_count / len(valid_records)) if valid_records else float("nan"),
        "random_positive_rate": positive_rate,
        "precision_top10": precision_top10,
        "precision_top10_ratio": precision_ratio,
        "auc": auc,
        "interaction_harm": interaction_harm,
        "priced_ratio": priced_ratio,
        "restore_failures": int(summary.get("restore_failures", 0)) if summary else 0,
        "supported_family_count": len(summary.get("supported_families", [])) if summary else 0,
        "supported_families": summary.get("supported_families", []) if summary else [],
        "forward_eval_count": int(summary.get("forward_eval_count", 0)) if summary else 0,
        "baseline_ce": safe_float(summary.get("baseline_ce")) if summary else float("nan"),
    }

    decision, reason = classify(metrics)
    metrics["decision"] = decision
    metrics["reason"] = reason

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    prevalence_text = "nan" if not math.isfinite(metrics["useful_delta_prevalence"]) else f"{metrics['useful_delta_prevalence']:.9g}"
    precision_text = "nan" if not math.isfinite(metrics["precision_top10_ratio"]) else f"{metrics['precision_top10_ratio']:.9g}"
    auc_text = "nan" if not math.isfinite(metrics["auc"]) else f"{metrics['auc']:.9g}"
    interaction_text = "nan" if not math.isfinite(metrics["interaction_harm"]) else f"{metrics['interaction_harm']:.9g}"
    priced_text = "nan" if not math.isfinite(metrics["priced_ratio"]) else f"{metrics['priced_ratio']:.9g}"
    print(
        f"[R405_TCFVL_ANALYZE] decision={decision} prevalence={prevalence_text} "
        f"precision_top10={precision_text} auc={auc_text} interaction_harm={interaction_text} "
        f"priced_ratio={priced_text} reason={reason}"
    )


if __name__ == "__main__":
    main()
