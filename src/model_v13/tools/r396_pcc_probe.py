from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path


HORIZON = 64
SEED = 396
ROWS_TOTAL = 768
TRAIN_FRACTION = 0.70
RIDGE_SCALE = 1e-3
SHUFFLE_BLOCKS = 16
SHUFFLE_BLOCK_SIZE = HORIZON
SELF_TEST_REASON = "synthetic_self_test_validates_probe_machinery_only"
SYNTHETIC_ONLY_REASON = "synthetic_only_self_test_not_finding_evidence"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _is_finite_number(value: float) -> bool:
    return math.isfinite(value)


def _build_synthetic_trace() -> list[dict[str, float | int | bool]]:
    rows: list[dict[str, float | int | bool]] = []
    losses = [1.25]

    for step in range(ROWS_TOTAL):
        phase = 2.0 * math.pi * step / 48.0
        x_phase_sin = math.sin(phase)
        x_phase_cos = math.cos(phase)
        x_trend = ((step % 17) - 8.0) / 8.0
        x_mix = 0.45 * x_phase_sin - 0.35 * x_phase_cos + 0.20 * x_trend
        increment = (
            0.010
            + 0.020 * x_phase_sin
            - 0.014 * x_phase_cos
            + 0.012 * x_trend
            + 0.018 * x_mix
            + 0.003 * math.sin(2.0 * math.pi * step / 7.0)
        )
        next_loss = losses[-1] + increment
        row = {
            "step": step,
            "loss": losses[-1],
            "pcc_x_phase_sin": x_phase_sin,
            "pcc_x_phase_cos": x_phase_cos,
            "pcc_x_trend": x_trend,
            "pcc_x_mix": x_mix,
            "pcc_x_phase_sin_available_at_step": step,
            "pcc_x_phase_cos_available_at_step": step,
            "pcc_x_trend_available_at_step": step,
            "pcc_x_mix_available_at_step": step,
            "baseline_step_wall_ms": 2.50,
            "feature_extract_ms": 0.06,
            "compensator_predict_ms": 0.03,
            "compensator_update_ms": 0.02,
            "no_second_suffix_pass": True,
            "synthetic_only": True,
        }
        rows.append(row)
        losses.append(next_loss)

    return rows


def _build_eligible_rows(
    rows: list[dict[str, float | int | bool]],
) -> tuple[list[list[float]], list[float], bool]:
    feature_names = [
        "pcc_x_phase_sin",
        "pcc_x_phase_cos",
        "pcc_x_trend",
        "pcc_x_mix",
    ]
    features: list[list[float]] = []
    labels: list[float] = []
    provenance_ok = True

    for step in range(len(rows) - HORIZON):
        row = rows[step]
        feature_vector: list[float] = []
        for feature_name in feature_names:
            value = float(row[feature_name])
            available_at_step = int(row[f"{feature_name}_available_at_step"])
            provenance_ok = provenance_ok and available_at_step <= step
            feature_vector.append(value)
        future_losses = [float(rows[offset]["loss"]) for offset in range(step + 1, step + HORIZON + 1)]
        label = _mean(future_losses) - float(row["loss"])
        features.append(feature_vector)
        labels.append(label)

    return features, labels, provenance_ok


def _center_matrix(
    rows: list[list[float]],
) -> tuple[list[list[float]], list[float]]:
    columns = len(rows[0])
    means = [_mean([row[index] for row in rows]) for index in range(columns)]
    centered = [[row[index] - means[index] for index in range(columns)] for row in rows]
    return centered, means


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [matrix[index][:] + [vector[index]] for index in range(size)]

    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda idx: abs(augmented[idx][pivot_index]))
        pivot_value = augmented[pivot_row][pivot_index]
        if abs(pivot_value) < 1e-12:
            raise ValueError("singular_ridge_system")
        if pivot_row != pivot_index:
            augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]

        pivot_value = augmented[pivot_index][pivot_index]
        for column_index in range(pivot_index, size + 1):
            augmented[pivot_index][column_index] /= pivot_value

        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            if factor == 0.0:
                continue
            for column_index in range(pivot_index, size + 1):
                augmented[row_index][column_index] -= factor * augmented[pivot_index][column_index]

    return [augmented[index][-1] for index in range(size)]


def _fit_ridge(
    train_features: list[list[float]],
    train_labels: list[float],
) -> tuple[list[float], float, float]:
    centered_features, feature_means = _center_matrix(train_features)
    label_mean = _mean(train_labels)
    centered_labels = [label - label_mean for label in train_labels]
    dimensions = len(train_features[0])

    xtx = [[0.0 for _ in range(dimensions)] for _ in range(dimensions)]
    xty = [0.0 for _ in range(dimensions)]
    for feature_row, centered_label in zip(centered_features, centered_labels):
        for row_index in range(dimensions):
            xty[row_index] += feature_row[row_index] * centered_label
            for column_index in range(dimensions):
                xtx[row_index][column_index] += feature_row[row_index] * feature_row[column_index]

    trace_xtx = sum(xtx[index][index] for index in range(dimensions))
    ridge_lambda = RIDGE_SCALE * trace_xtx / dimensions
    for index in range(dimensions):
        xtx[index][index] += ridge_lambda

    weights = _solve_linear_system(xtx, xty)
    intercept = label_mean - _dot(weights, feature_means)
    return weights, intercept, ridge_lambda


def _predict(features: list[list[float]], weights: list[float], intercept: float) -> list[float]:
    return [_dot(row, weights) + intercept for row in features]


def _variance_against_mean(values: list[float], baseline_mean: float) -> float:
    return _mean([(value - baseline_mean) ** 2 for value in values])


def _r2_from_predictions(labels: list[float], predictions: list[float], baseline_mean: float) -> tuple[float, float, float]:
    baseline_var = _variance_against_mean(labels, baseline_mean)
    residual_var = _mean([(label - prediction) ** 2 for label, prediction in zip(labels, predictions)])
    if baseline_var <= 0.0:
        raise ValueError("zero_baseline_variance")
    return baseline_var, residual_var, 1.0 - (residual_var / baseline_var)


def _block_shuffle_labels(labels: list[float], seed: int) -> list[float]:
    blocks = [labels[index:index + SHUFFLE_BLOCK_SIZE] for index in range(0, len(labels), SHUFFLE_BLOCK_SIZE)]
    order = list(range(len(blocks)))
    random.Random(seed).shuffle(order)
    shuffled: list[float] = []
    for block_index in order:
        shuffled.extend(blocks[block_index])
    return shuffled


def _compute_shuffled_r2(
    train_features: list[list[float]],
    train_labels: list[float],
    test_features: list[list[float]],
    test_labels: list[float],
    baseline_mean: float,
) -> float:
    shuffled_scores: list[float] = []
    for shuffle_index in range(SHUFFLE_BLOCKS):
        shuffled_labels = _block_shuffle_labels(train_labels, SEED + shuffle_index)
        weights, intercept, _ = _fit_ridge(train_features, shuffled_labels)
        predictions = _predict(test_features, weights, intercept)
        _, _, shuffled_r2 = _r2_from_predictions(test_labels, predictions, baseline_mean)
        shuffled_scores.append(shuffled_r2)
    return _mean(shuffled_scores)


def _json_ready_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        name: float(value) if _is_finite_number(value) else 0.0
        for name, value in metrics.items()
    }


def _run_self_test(report_path: Path) -> int:
    rows = _build_synthetic_trace()
    features, labels, provenance_ok = _build_eligible_rows(rows)
    rows_eligible = len(labels)
    minimum_eligible_rows = max(512, 8 * HORIZON)
    validity_reasons = [SYNTHETIC_ONLY_REASON]

    if rows_eligible < minimum_eligible_rows:
        validity_reasons.append("insufficient_rows")

    train_rows = int(rows_eligible * TRAIN_FRACTION)
    train_features = features[:train_rows]
    test_features = features[train_rows:]
    train_labels = labels[:train_rows]
    test_labels = labels[train_rows:]

    metrics = {
        "baseline_var": float("nan"),
        "residual_var": float("nan"),
        "r2_oos": float("nan"),
        "r2_shuffled": float("nan"),
        "delta_c_est": float("nan"),
        "delta_p_est": float("nan"),
        "delta_p_required": float("nan"),
        "s_net_proxy": float("nan"),
    }
    leakage_guard = "fail"
    decision = "invalid"
    decision_reason = "probe_validation_failed"
    exit_code = 1

    try:
        if not provenance_ok:
            validity_reasons.append("provenance_failed")

        baseline_step_wall_ms = float(rows[0]["baseline_step_wall_ms"])
        feature_extract_ms = float(rows[0]["feature_extract_ms"])
        compensator_predict_ms = float(rows[0]["compensator_predict_ms"])
        compensator_update_ms = float(rows[0]["compensator_update_ms"])
        online_overhead_ms = feature_extract_ms + compensator_predict_ms + compensator_update_ms
        delta_c_est = online_overhead_ms / baseline_step_wall_ms

        weights, intercept, _ = _fit_ridge(train_features, train_labels)
        predictions = _predict(test_features, weights, intercept)
        train_label_mean = _mean(train_labels)
        baseline_var, residual_var, r2_oos = _r2_from_predictions(test_labels, predictions, train_label_mean)
        r2_shuffled = _compute_shuffled_r2(
            train_features=train_features,
            train_labels=train_labels,
            test_features=test_features,
            test_labels=test_labels,
            baseline_mean=train_label_mean,
        )
        delta_p_est = max(0.0, r2_oos)
        delta_p_required = delta_c_est / max(1e-12, 1.0 - delta_c_est)
        s_net_proxy = delta_p_est - delta_p_required

        metrics = {
            "baseline_var": baseline_var,
            "residual_var": residual_var,
            "r2_oos": r2_oos,
            "r2_shuffled": r2_shuffled,
            "delta_c_est": delta_c_est,
            "delta_p_est": delta_p_est,
            "delta_p_required": delta_p_required,
            "s_net_proxy": s_net_proxy,
        }

        if not all(_is_finite_number(value) for value in metrics.values()):
            validity_reasons.append("non_finite_metrics")
        if baseline_var <= 0.0:
            validity_reasons.append("zero_baseline_variance")
        leakage_guard = "pass" if provenance_ok and r2_shuffled < 0.03 else "fail"
        if leakage_guard != "pass":
            validity_reasons.append("shuffle_guard_failed")
        if not bool(rows[0]["no_second_suffix_pass"]):
            validity_reasons.append("second_suffix_pass_detected")
        if not bool(rows[0]["synthetic_only"]):
            validity_reasons.append("synthetic_only_flag_missing")

        hard_failures = [reason for reason in validity_reasons if reason != SYNTHETIC_ONLY_REASON]
        if not hard_failures:
            decision = "self_test_pass_not_evidence"
            decision_reason = SELF_TEST_REASON
            exit_code = 0
        else:
            decision_reason = ",".join(hard_failures)
    except ValueError as error:
        validity_reasons.append(str(error))
        decision_reason = str(error)

    report = {
        "schema": "r396.pcc.selftest.report.v1",
        "mode": "self_test",
        "synthetic_only": True,
        "seed": SEED,
        "horizon": HORIZON,
        "rows_total": len(rows),
        "rows_eligible": rows_eligible,
        "validity_status": "invalid",
        "validity_reasons": validity_reasons,
        "metrics": _json_ready_metrics(metrics),
        "leakage_guard": leakage_guard,
        "decision": decision,
        "decision_reason": decision_reason,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "R396_PCC_ACTIVATION "
        f"mode=self_test source=synthetic seed={SEED} steps={len(rows)} batch_size=4 synthetic_only=true"
    )
    print(
        "R396_PCC_VALIDITY "
        f"status=invalid reasons={','.join(validity_reasons)}"
    )
    print(
        "R396_PCC_METRICS "
        f"rows={rows_eligible} horizon={HORIZON} "
        f"r2_oos={_format_float(metrics['r2_oos'])} "
        f"delta_c_est={_format_float(metrics['delta_c_est'])} "
        f"delta_p_est={_format_float(metrics['delta_p_est'])} "
        f"delta_p_required={_format_float(metrics['delta_p_required'])} "
        f"s_net_proxy={_format_float(metrics['s_net_proxy'])}"
    )
    print(
        "R396_PCC_COMPARATORS "
        f"baseline_var={_format_float(metrics['baseline_var'])} "
        f"residual_var={_format_float(metrics['residual_var'])} "
        f"shuffled_r2={_format_float(metrics['r2_shuffled'])} "
        f"leakage_guard={leakage_guard}"
    )
    print(f"R396_PCC_DECISION {decision} reason={decision_reason}")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R396 PCC synthetic self-test probe")
    parser.add_argument("--self-test", action="store_true", help="run the deterministic synthetic self-test")
    parser.add_argument("--report", required=True, help="path to write the JSON report")
    parser.add_argument("--trace", help="unsupported in Slice 1", default=None)
    args = parser.parse_args(argv)

    if args.trace is not None:
        print("unsupported: --trace is not available in R396 PCC Slice 1", file=sys.stderr)
        return 2
    if not args.self_test:
        print("unsupported: only --self-test is available in R396 PCC Slice 1", file=sys.stderr)
        return 2
    return _run_self_test(Path(args.report))


if __name__ == "__main__":
    raise SystemExit(main())
