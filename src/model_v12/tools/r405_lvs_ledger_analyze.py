import argparse
import json
import math
from pathlib import Path


ANALYZER_VERSION = "r405_lvs_ledger_analyze.v1"
DELTA_EPS = 1.0e-6
PREVALENCE_THRESHOLD = 0.10
CONCENTRATION_THRESHOLD = 0.50
PRECISION_RATIO_THRESHOLD = 2.0
AUC_THRESHOLD = 0.65
INTERACTION_HARM_THRESHOLD = 0.30
PRICED_PROGRESS_THRESHOLD = 1.15


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--ledger-reuse",
        type=float,
        default=128.0,
        help="Amortized reuse count for pricing sampled erasure labels.",
    )
    return parser.parse_args()


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def auc_from_scored_records(records, score_key):
    positives = [r for r in records if r["label"]]
    negatives = [r for r in records if not r["label"]]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    total = 0.0
    for pos in positives:
        for neg in negatives:
            total += 1.0
            if pos[score_key] > neg[score_key]:
                wins += 1.0
            elif pos[score_key] == neg[score_key]:
                wins += 0.5
    return wins / total if total else float("nan")


def precision_ratio_from_scores(records, score_key, positive_prevalence):
    sorted_by_score = sorted(records, key=lambda r: r[score_key], reverse=True)
    top_k = max(1, min(10, len(sorted_by_score)))
    precision_topk = (
        sum(1 for r in sorted_by_score[:top_k] if r["label"]) / top_k
        if top_k
        else float("nan")
    )
    precision_ratio = (
        precision_topk / positive_prevalence
        if math.isfinite(positive_prevalence) and positive_prevalence > 0.0
        else float("nan")
    )
    return precision_topk, precision_ratio


def assign_leave_one_out_scores(records):
    global_sum = sum(max(0.0, r["delta"]) for r in records)
    global_count = len(records)
    layer_stats = {}
    block_stats = {}
    for record in records:
        raw = record["raw"]
        layer_key = int(raw.get("layer", -1))
        block_key = (layer_key, int(raw.get("group", -1)))
        value = max(0.0, record["delta"])
        layer_stats[layer_key] = (
            layer_stats.get(layer_key, (0.0, 0))[0] + value,
            layer_stats.get(layer_key, (0.0, 0))[1] + 1,
        )
        block_stats[block_key] = (
            block_stats.get(block_key, (0.0, 0))[0] + value,
            block_stats.get(block_key, (0.0, 0))[1] + 1,
        )
    fallback = global_sum / global_count if global_count else 0.0
    for record in records:
        raw = record["raw"]
        layer_key = int(raw.get("layer", -1))
        block_key = (layer_key, int(raw.get("group", -1)))
        value = max(0.0, record["delta"])
        layer_sum, layer_count = layer_stats[layer_key]
        block_sum, block_count = block_stats[block_key]
        record["layer_loo_score"] = (
            (layer_sum - value) / (layer_count - 1)
            if layer_count > 1
            else fallback
        )
        record["block_loo_score"] = (
            (block_sum - value) / (block_count - 1)
            if block_count > 1
            else record["layer_loo_score"]
        )


def classify(metrics):
    if not metrics["meta_present"]:
        return "INVALID_PROBE", "missing_meta"
    if metrics["duplicate_meta_rows"] > 0:
        return "INVALID_PROBE", "duplicate_meta"
    if not metrics["activation_flag"]:
        return "INVALID_PROBE", "missing_activation"
    if metrics["source"] != "v12-cpu-forward-state-erasure":
        return "INVALID_PROBE", "wrong_source"
    if metrics["ce_mode"] != "full":
        return "INVALID_PROBE", "non_full_ce"
    if metrics["record_count"] == 0:
        return "INVALID_PROBE", "no_records"
    if metrics["non_finite_records"] > 0:
        return "INVALID_PROBE", "non_finite_record"
    if metrics["smoke_scale"]:
        return "SMOKE_ONLY", "context_count_below_formal_gate"
    if metrics["positive_count"] == 0:
        return "FAILED_IDENTIFYING_GATE", "no_positive_erasure_value"
    if metrics["pair_count"] == 0 or not math.isfinite(metrics["mean_interaction_harm"]):
        return "INVALID_PROBE", "missing_pair_interaction"

    passes_prevalence = metrics["positive_prevalence"] >= PREVALENCE_THRESHOLD
    passes_concentration = metrics["top_quartile_value_share"] >= CONCENTRATION_THRESHOLD
    passes_predictor = (
        metrics["precision_topk_ratio"] >= PRECISION_RATIO_THRESHOLD
        or metrics["auc"] >= AUC_THRESHOLD
    )
    passes_interaction = metrics["mean_interaction_harm"] <= INTERACTION_HARM_THRESHOLD
    passes_priced = metrics["priced_projected_progress"] >= PRICED_PROGRESS_THRESHOLD

    if passes_prevalence and passes_concentration and passes_predictor and passes_interaction and passes_priced:
        return "PASS_OBSERVATIONAL_GATE", "passed_sparse_predictable_priced_thresholds"

    near_prevalence = metrics["positive_prevalence"] >= 0.8 * PREVALENCE_THRESHOLD
    near_concentration = metrics["top_quartile_value_share"] >= 0.8 * CONCENTRATION_THRESHOLD
    near_predictor = (
        metrics["precision_topk_ratio"] >= 0.8 * PRECISION_RATIO_THRESHOLD
        or metrics["auc"] >= 0.9 * AUC_THRESHOLD
    )
    near_interaction = metrics["mean_interaction_harm"] <= 1.15 * INTERACTION_HARM_THRESHOLD
    near_priced = metrics["priced_projected_progress"] >= 0.9 * PRICED_PROGRESS_THRESHOLD
    if near_prevalence and near_concentration and near_predictor and near_interaction and near_priced:
        return "WEAK_SIGNAL_NEEDS_PILOT", "near_thresholds_without_closing_priced_gain"
    return "FAILED_IDENTIFYING_GATE", "below_sparse_predictable_priced_thresholds"


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    meta = None
    duplicate_meta_rows = 0
    summary = None
    records = []
    pairs = []

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
                else:
                    meta = item
            elif kind == "record":
                delta = safe_float(item.get("delta_ce"))
                score = safe_float(item.get("predictor_score"))
                finite = math.isfinite(delta) and math.isfinite(score)
                records.append(
                    {
                        "raw": item,
                        "delta": delta,
                        "local_feature_score": score,
                        "label": finite and delta > DELTA_EPS,
                        "finite": finite,
                    }
                )
            elif kind == "pair":
                harm = safe_float(item.get("interaction_harm"))
                pairs.append({"raw": item, "harm": harm, "finite": math.isfinite(harm)})
            elif kind == "summary":
                summary = item

    valid_records = [r for r in records if r["finite"]]
    assign_leave_one_out_scores(valid_records)
    positives = [r for r in valid_records if r["label"]]
    positive_value_total = sum(max(0.0, r["delta"]) for r in valid_records)
    positive_prevalence = len(positives) / len(valid_records) if valid_records else float("nan")

    sorted_by_value = sorted(valid_records, key=lambda r: max(0.0, r["delta"]), reverse=True)
    top_quartile_n = max(1, math.ceil(len(valid_records) * 0.25)) if valid_records else 0
    top_quartile_value = sum(max(0.0, r["delta"]) for r in sorted_by_value[:top_quartile_n])
    top_quartile_share = (
        top_quartile_value / positive_value_total
        if positive_value_total > 0.0
        else float("nan")
    )

    score_variants = {}
    for score_key in ("local_feature_score", "layer_loo_score", "block_loo_score"):
        precision_topk, precision_ratio = precision_ratio_from_scores(
            valid_records, score_key, positive_prevalence
        )
        score_variants[score_key] = {
            "precision_topk": precision_topk,
            "precision_topk_ratio": precision_ratio,
            "auc": auc_from_scored_records(valid_records, score_key),
        }
    best_score_key = max(
        score_variants,
        key=lambda key: max(
            score_variants[key]["precision_topk_ratio"]
            if math.isfinite(score_variants[key]["precision_topk_ratio"])
            else -1.0,
            score_variants[key]["auc"]
            if math.isfinite(score_variants[key]["auc"])
            else -1.0,
        ),
    )
    precision_topk = score_variants[best_score_key]["precision_topk"]
    precision_ratio = score_variants[best_score_key]["precision_topk_ratio"]
    auc = score_variants[best_score_key]["auc"]

    finite_pairs = [p for p in pairs if p["finite"]]
    mean_harm = (
        sum(p["harm"] for p in finite_pairs) / len(finite_pairs)
        if finite_pairs
        else float("nan")
    )

    predictor_strength = 0.0
    if math.isfinite(precision_ratio) and precision_ratio > 1.0:
        predictor_strength = max(predictor_strength, min(1.0, (precision_ratio - 1.0) / 2.0))
    if math.isfinite(auc) and auc > 0.5:
        predictor_strength = max(predictor_strength, min(1.0, (auc - 0.5) / 0.25))
    concentration = top_quartile_share if math.isfinite(top_quartile_share) else 0.0
    prevalence = positive_prevalence if math.isfinite(positive_prevalence) else 0.0
    gross_signal = min(1.0, prevalence * concentration * predictor_strength * 4.0)
    overhead = (
        float(meta.get("blocks_per_context", 0)) / args.ledger_reuse
        if meta and args.ledger_reuse > 0.0
        else float("inf")
    )
    priced_projected_progress = (1.0 + 0.30 * gross_signal) / (1.0 + overhead)

    metrics = {
        "analyzer_version": ANALYZER_VERSION,
        "input_path": str(input_path),
        "meta_present": meta is not None,
        "duplicate_meta_rows": duplicate_meta_rows,
        "activation_flag": bool(meta.get("activation_flag")) if meta else False,
        "source": meta.get("source") if meta else None,
        "ce_mode": meta.get("ce_mode") if meta else None,
        "contexts": int(meta.get("contexts", 0)) if meta else 0,
        "horizon": int(meta.get("horizon", 0)) if meta else 0,
        "groups": int(meta.get("groups", 0)) if meta else 0,
        "blocks_per_context": int(meta.get("blocks_per_context", 0)) if meta else 0,
        "smoke_scale": bool(meta.get("smoke_scale")) if meta else False,
        "record_count": len(records),
        "valid_record_count": len(valid_records),
        "non_finite_records": sum(1 for r in records if not r["finite"]),
        "positive_count": len(positives),
        "positive_prevalence": positive_prevalence,
        "positive_value_total": positive_value_total,
        "top_quartile_value_share": top_quartile_share,
        "precision_topk": precision_topk,
        "precision_topk_ratio": precision_ratio,
        "auc": auc,
        "predictor_source": best_score_key,
        "score_variants": score_variants,
        "pair_count": len(pairs),
        "finite_pair_count": len(finite_pairs),
        "mean_interaction_harm": mean_harm,
        "ledger_reuse": args.ledger_reuse,
        "pricing_overhead": overhead,
        "gross_signal": gross_signal,
        "priced_projected_progress": priced_projected_progress,
        "forward_eval_count": int(summary.get("forward_eval_count", 0)) if summary else 0,
        "baseline_future_loss_mean": safe_float(summary.get("baseline_future_loss_mean")) if summary else float("nan"),
    }
    decision, reason = classify(metrics)
    metrics["decision"] = decision
    metrics["reason"] = reason

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    def fmt(value):
        return "nan" if not math.isfinite(value) else f"{value:.9g}"

    print(
        f"[R405_LVS_LEDGER_ANALYZE] decision={decision} "
        f"prevalence={fmt(metrics['positive_prevalence'])} "
        f"concentration={fmt(metrics['top_quartile_value_share'])} "
        f"precision_ratio={fmt(metrics['precision_topk_ratio'])} "
        f"auc={fmt(metrics['auc'])} "
        f"interaction_harm={fmt(metrics['mean_interaction_harm'])} "
        f"priced_progress={fmt(metrics['priced_projected_progress'])} "
        f"reason={reason}"
    )


if __name__ == "__main__":
    main()
