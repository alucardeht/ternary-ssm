#!/usr/bin/env python3
"""Deterministic R397 charge-defect probe self-test."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


TRACE_SCHEMA = "r397.cd.trace.v1"
REPORT_SCHEMA = "r397.cd.report.v1"
EPS = 1e-12
LAGS = (1, 2, 4)
SELF_TEST_SEED = 397
TIE_TOLERANCE = 1e-6
RIDGE_GRID = (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)


@dataclass
class LayerSeries:
    layer: int
    steps: np.ndarray
    x: np.ndarray
    loss_before: np.ndarray
    loss_after: np.ndarray
    tokens: np.ndarray
    step_wall_ms: np.ndarray
    summary_extract_ms: np.ndarray
    probe_update_ms: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true", help="Run the deterministic synthetic self-test.")
    mode.add_argument("--trace", type=Path, help="Path to a trace JSON object with schema r397.cd.trace.v1.")
    parser.add_argument("--report", type=Path, required=True, help="Output JSON report path.")
    return parser.parse_args()


def load_trace(path: Path) -> MutableMapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("trace payload must be a JSON object")
    return payload


def build_self_test_trace(seed: int = SELF_TEST_SEED) -> MutableMapping[str, object]:
    rng = np.random.default_rng(seed)
    total_steps = 192
    summary_dim = 24
    defect_dim = 4
    t = np.arange(total_steps, dtype=np.float64)

    charge = np.stack(
        [
            np.sin(2.0 * math.pi * t / 48.0) + 0.15 * np.sin(2.0 * math.pi * t / 96.0),
            np.cos(2.0 * math.pi * t / 60.0 + 0.35) + 0.0025 * t,
        ],
        axis=1,
    )
    defect = np.stack(
        [
            0.45 * np.sin(2.0 * math.pi * t / 5.0 + 0.4),
            0.35 * np.cos(2.0 * math.pi * t / 7.0 + 0.2),
            0.30 * np.sin(2.0 * math.pi * t / 9.0 + 0.9),
            0.25 * np.cos(2.0 * math.pi * t / 11.0 + 0.5),
        ],
        axis=1,
    )
    defect += 0.04 * rng.standard_normal((total_steps, defect_dim))
    latent = np.concatenate([charge, defect], axis=1)
    mix = rng.standard_normal((latent.shape[1], summary_dim))
    mix /= np.linalg.norm(mix, axis=1, keepdims=True)

    q_energy = np.sum(charge ** 2, axis=1)
    d_energy = np.sum(defect ** 2, axis=1)
    interaction = charge[:, 0] * (0.55 + 0.35 * d_energy) - 0.18 * charge[:, 1] * defect[:, 1]
    progress = 0.030 + 0.028 * np.tanh(interaction) + 0.011 * q_energy - 0.0085 * d_energy
    progress += 0.0015 * np.sin(2.0 * math.pi * t / 13.0)
    progress = np.clip(progress, 0.0025, 0.115)

    steps: List[Dict[str, object]] = []
    states: List[Dict[str, object]] = []
    base_loss = 1.85 - 0.0017 * t + 0.028 * np.sin(2.0 * math.pi * t / 37.0)
    for layer in range(2):
        layer_shift = 0.045 * layer
        layer_mix = mix + 0.015 * layer * rng.standard_normal(mix.shape)
        x = latent @ layer_mix + 0.03 * rng.standard_normal((total_steps, summary_dim))
        x += layer_shift * np.outer(np.sin(2.0 * math.pi * t / 24.0), np.linspace(-0.7, 0.7, summary_dim))
        if layer == 0:
            layer_progress = progress
        else:
            layer_progress = np.clip(progress * 0.94 + 0.002 * np.cos(2.0 * math.pi * t / 17.0), 0.002, 0.11)
        loss_before = base_loss + 0.03 * layer + 0.009 * np.cos(2.0 * math.pi * t / 29.0 + 0.2 * layer)
        loss_after = loss_before * (1.0 - layer_progress)
        tokens = 384 + ((7 * t.astype(np.int64) + 17 * layer) % 113)
        step_wall_ms = 6.8 + 0.018 * t + 0.20 * np.sin(2.0 * math.pi * t / 21.0 + 0.1 * layer)
        summary_extract_ms = 0.34 + 0.03 * np.abs(np.sin(2.0 * math.pi * t / 19.0 + 0.15 * layer))
        probe_update_ms = 0.17 + 0.02 * np.abs(np.cos(2.0 * math.pi * t / 23.0 + 0.25 * layer))
        if layer == 0:
            for idx in range(total_steps):
                steps.append(
                    {
                        "step": int(idx),
                        "loss_before": float(loss_before[idx]),
                        "loss_after": float(loss_after[idx]),
                        "tokens": int(tokens[idx]),
                        "step_wall_ms": float(step_wall_ms[idx]),
                        "summary_extract_ms": float(summary_extract_ms[idx]),
                        "probe_update_ms": float(probe_update_ms[idx]),
                    }
                )
        for idx in range(total_steps):
            states.append({"step": int(idx), "layer": layer, "x": [float(value) for value in x[idx]]})

    return {
        "schema": TRACE_SCHEMA,
        "meta": {
            "run_id": "r397-self-test",
            "seed": seed,
            "model_id": "synthetic-self-test",
            "summary_kind": "layer_mean_or_compact_hidden_summary",
            "summary_dim": summary_dim,
            "batch_size": 4,
            "fit_fraction": 0.60,
            "projection_source": "prefix_only",
            "phase_surrogate": "deterministic_circular_shift",
        },
        "steps": steps,
        "states": states,
    }


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_trace(trace: Mapping[str, object]) -> Tuple[List[LayerSeries], List[str], Mapping[str, object]]:
    global_invalid_reasons: List[str] = []
    if trace.get("schema") != TRACE_SCHEMA:
        global_invalid_reasons.append("missing required field or wrong schema")
        return [], global_invalid_reasons, {}
    meta = trace.get("meta")
    steps = trace.get("steps")
    states = trace.get("states")
    if not isinstance(meta, dict) or not isinstance(steps, list) or not isinstance(states, list):
        global_invalid_reasons.append("missing required field")
        return [], global_invalid_reasons, {}

    required_meta = (
        "run_id",
        "seed",
        "model_id",
        "summary_kind",
        "summary_dim",
        "batch_size",
        "fit_fraction",
        "projection_source",
    )
    for field in required_meta:
        if field not in meta:
            global_invalid_reasons.append("missing required field")
            return [], global_invalid_reasons, meta

    summary_dim = int(meta["summary_dim"])
    fit_fraction = float(meta["fit_fraction"])
    if summary_dim <= 0 or not (0.0 < fit_fraction < 1.0):
        global_invalid_reasons.append("missing required field")
        return [], global_invalid_reasons, meta

    step_records: List[Dict[str, float]] = []
    seen_steps = set()
    last_step = None
    for entry in steps:
        if not isinstance(entry, dict):
            global_invalid_reasons.append("missing required field")
            return [], global_invalid_reasons, meta
        required_step_fields = (
            "step",
            "loss_before",
            "loss_after",
            "tokens",
            "step_wall_ms",
            "summary_extract_ms",
            "probe_update_ms",
        )
        if any(field not in entry for field in required_step_fields):
            global_invalid_reasons.append("missing required field")
            return [], global_invalid_reasons, meta
        step_value = entry["step"]
        if not isinstance(step_value, int):
            global_invalid_reasons.append("missing required field")
            return [], global_invalid_reasons, meta
        if last_step is not None and step_value <= last_step:
            global_invalid_reasons.append("steps are not strictly increasing")
            return [], global_invalid_reasons, meta
        last_step = step_value
        if step_value in seen_steps:
            global_invalid_reasons.append("steps are not strictly increasing")
            return [], global_invalid_reasons, meta
        seen_steps.add(step_value)
        numeric_fields = {field: entry[field] for field in required_step_fields if field != "step"}
        if not all(_is_finite_number(value) for value in numeric_fields.values()):
            global_invalid_reasons.append("non-finite numeric value")
            return [], global_invalid_reasons, meta
        step_records.append(
            {
                "step": float(step_value),
                "loss_before": float(entry["loss_before"]),
                "loss_after": float(entry["loss_after"]),
                "tokens": float(entry["tokens"]),
                "step_wall_ms": float(entry["step_wall_ms"]),
                "summary_extract_ms": float(entry["summary_extract_ms"]),
                "probe_update_ms": float(entry["probe_update_ms"]),
            }
        )

    step_lookup = {int(record["step"]): record for record in step_records}
    layer_to_states: Dict[int, List[Tuple[int, Sequence[float]]]] = {}
    for entry in states:
        if not isinstance(entry, dict) or any(field not in entry for field in ("step", "layer", "x")):
            global_invalid_reasons.append("missing required field")
            return [], global_invalid_reasons, meta
        step_value = entry["step"]
        layer = entry["layer"]
        vector = entry["x"]
        if not isinstance(step_value, int) or not isinstance(layer, int) or not isinstance(vector, list):
            global_invalid_reasons.append("missing required field")
            return [], global_invalid_reasons, meta
        if step_value not in step_lookup:
            global_invalid_reasons.append("states reference missing steps")
            return [], global_invalid_reasons, meta
        if len(vector) != summary_dim:
            global_invalid_reasons.append("summary_dim mismatch")
            return [], global_invalid_reasons, meta
        if not all(_is_finite_number(value) for value in vector):
            global_invalid_reasons.append("non-finite numeric value")
            return [], global_invalid_reasons, meta
        layer_to_states.setdefault(layer, []).append((step_value, [float(value) for value in vector]))

    layers: List[LayerSeries] = []
    for layer, raw_states in sorted(layer_to_states.items()):
        raw_states.sort(key=lambda item: item[0])
        ordered_steps = np.array([step for step, _ in raw_states], dtype=np.int64)
        x = np.array([vector for _, vector in raw_states], dtype=np.float64)
        step_payload = [step_lookup[int(step)] for step in ordered_steps]
        layers.append(
            LayerSeries(
                layer=layer,
                steps=ordered_steps,
                x=x,
                loss_before=np.array([item["loss_before"] for item in step_payload], dtype=np.float64),
                loss_after=np.array([item["loss_after"] for item in step_payload], dtype=np.float64),
                tokens=np.array([item["tokens"] for item in step_payload], dtype=np.float64),
                step_wall_ms=np.array([item["step_wall_ms"] for item in step_payload], dtype=np.float64),
                summary_extract_ms=np.array([item["summary_extract_ms"] for item in step_payload], dtype=np.float64),
                probe_update_ms=np.array([item["probe_update_ms"] for item in step_payload], dtype=np.float64),
            )
        )
    return layers, global_invalid_reasons, meta


def lag_covariance(x: np.ndarray, lag: int) -> np.ndarray:
    if x.shape[0] <= lag:
        return np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    left = x[:-lag]
    right = x[lag:]
    return (left.T @ right) / float(left.shape[0])


def material_share(x: np.ndarray, projection: np.ndarray, lags: Sequence[int], eps: float) -> Tuple[float, float]:
    p = projection @ projection.T
    numerator = 0.0
    denominator = 0.0
    for lag in lags:
        cov = lag_covariance(x, lag)
        denominator += float(np.sum(cov * cov))
        projected = p @ cov @ p
        numerator += float(np.sum(projected * projected))
    if denominator <= eps:
        return 0.0, denominator
    return numerator / denominator, denominator


def choose_charge_projection(x_fit: np.ndarray, eps: float) -> Tuple[np.ndarray, int, int, float]:
    n_fit, dim = x_fit.shape
    k_max = max(1, min(32, dim // 8, int(math.sqrt(n_fit))))
    aggregate = np.zeros((dim, dim), dtype=np.float64)
    for lag in LAGS:
        cov = lag_covariance(x_fit, lag)
        aggregate += 0.5 * (cov + cov.T)
    eigenvalues, eigenvectors = np.linalg.eigh(aggregate)
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order]
    best_k = 1
    best_score = -math.inf
    best_projection = basis[:, :1]
    for k in range(1, k_max + 1):
        candidate = basis[:, :k]
        score, _ = material_share(x_fit, candidate, LAGS, eps)
        if score > best_score + 1e-12:
            best_k = k
            best_score = score
            best_projection = candidate
    return best_projection, best_k, k_max, best_score


def weighted_mse(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    total_weight = float(np.sum(weights))
    if total_weight <= EPS:
        return math.inf
    residual = y_true - y_pred
    return float(np.sum(weights * residual * residual) / total_weight)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total_weight = float(np.sum(weights))
    if total_weight <= EPS:
        return 0.0
    return float(np.sum(weights * values) / total_weight)


def weighted_corr(feature: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    feature_centered = feature - weighted_mean(feature, weights)
    target_centered = target - weighted_mean(target, weights)
    numerator = float(np.sum(weights * feature_centered * target_centered))
    feature_den = float(np.sum(weights * feature_centered * feature_centered))
    target_den = float(np.sum(weights * target_centered * target_centered))
    denominator = math.sqrt(max(feature_den * target_den, 0.0))
    if denominator <= EPS:
        return 0.0
    return numerator / denominator


def fit_ridge_with_intercept(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_eval: np.ndarray,
    ridge: float,
) -> np.ndarray:
    feature_mean = np.mean(x_train, axis=0)
    feature_scale = np.std(x_train, axis=0)
    feature_scale = np.where(feature_scale <= EPS, 1.0, feature_scale)
    x_train_scaled = (x_train - feature_mean) / feature_scale
    x_eval_scaled = (x_eval - feature_mean) / feature_scale
    design_train = np.column_stack([np.ones(x_train_scaled.shape[0], dtype=np.float64), x_train_scaled])
    design_eval = np.column_stack([np.ones(x_eval_scaled.shape[0], dtype=np.float64), x_eval_scaled])
    sqrt_w = np.sqrt(w_train)[:, None]
    lhs = design_train * sqrt_w
    rhs = y_train * sqrt_w[:, 0]
    penalty = np.eye(design_train.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    coeffs = np.linalg.solve(lhs.T @ lhs + penalty, lhs.T @ rhs)
    return design_eval @ coeffs


def tune_and_score_arm(
    x_prefix: np.ndarray,
    y_prefix: np.ndarray,
    w_prefix: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    w_valid: np.ndarray,
) -> Tuple[float, float]:
    prefix_count = x_prefix.shape[0]
    tune_start = max(1, int(math.floor(prefix_count * 0.8)))
    tune_start = min(tune_start, prefix_count - 1)
    x_train = x_prefix[:tune_start]
    y_train = y_prefix[:tune_start]
    w_train = w_prefix[:tune_start]
    x_tune = x_prefix[tune_start:]
    y_tune = y_prefix[tune_start:]
    w_tune = w_prefix[tune_start:]
    best_ridge = RIDGE_GRID[0]
    best_tune_error = math.inf
    for ridge in RIDGE_GRID:
        tune_pred = fit_ridge_with_intercept(x_train, y_train, w_train, x_tune, ridge)
        tune_error = weighted_mse(y_tune, tune_pred, w_tune)
        if tune_error < best_tune_error - 1e-15:
            best_tune_error = tune_error
            best_ridge = ridge
    valid_pred = fit_ridge_with_intercept(x_prefix, y_prefix, w_prefix, x_valid, best_ridge)
    return weighted_mse(y_valid, valid_pred, w_valid), float(best_ridge)


def build_raw_budget_features(
    x_all: np.ndarray,
    dx_all: np.ndarray,
    feature_count: int,
    prefix_indices: np.ndarray,
    y_prefix: np.ndarray,
    w_prefix: np.ndarray,
) -> np.ndarray:
    raw = np.concatenate([x_all, dx_all], axis=1)
    if feature_count >= raw.shape[1]:
        return raw
    correlations = []
    for index in range(raw.shape[1]):
        corr = abs(weighted_corr(raw[prefix_indices, index], y_prefix, w_prefix))
        correlations.append((corr, index))
    correlations.sort(key=lambda item: (-item[0], item[1]))
    chosen = [index for _, index in correlations[:feature_count]]
    return raw[:, chosen]


def deterministic_shuffle(matrix: np.ndarray, layer: int) -> np.ndarray:
    rng = np.random.default_rng(SELF_TEST_SEED + 1000 + layer)
    permutation = rng.permutation(matrix.shape[0])
    return matrix[permutation]


def deterministic_phase_surrogate(matrix: np.ndarray, layer: int) -> np.ndarray:
    shifted = np.empty_like(matrix)
    length = matrix.shape[0]
    for dim in range(matrix.shape[1]):
        shift = ((layer + 1) * (dim + 3) * 7) % max(length, 1)
        if shift == 0:
            shift = (dim + 1) % max(length, 1)
        shifted[:, dim] = np.roll(matrix[:, dim], shift)
    return shifted


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    total = sum(weights)
    threshold = 0.5 * total
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= threshold:
            return float(value)
    return float(ordered[-1][0])


def evaluate_layer(series: LayerSeries, fit_fraction: float, summary_dim: int, eps: float) -> Dict[str, object]:
    invalid_reasons: List[str] = []
    if series.x.shape[1] != summary_dim:
        invalid_reasons.append("summary_dim mismatch")
    if series.steps.shape[0] < 128:
        invalid_reasons.append("fewer than 128 ordered steps for layer")
    if invalid_reasons:
        return {
            "layer": series.layer,
            "valid": False,
            "invalid_reasons": invalid_reasons,
            "selected_k": 0,
            "k_max": 0,
            "metrics": {},
            "arm_errors": {},
            "arm_scores": {},
            "inequality_margins": {},
            "weight": 0.0,
            "layer_pass": False,
        }

    n_steps = series.steps.shape[0]
    fit_count = int(math.floor(n_steps * fit_fraction))
    fit_count = min(max(fit_count, 1), n_steps - 1)
    x_fit_raw = series.x[:fit_count]
    fit_mean = np.mean(x_fit_raw, axis=0)
    x_all = series.x - fit_mean
    x_fit = x_all[:fit_count]
    x_valid = x_all[fit_count:]
    if np.max(np.std(x_all, axis=0)) <= eps:
        invalid_reasons.append("all-zero or constant x_t")
    projection, selected_k, k_max, prefix_materiality = choose_charge_projection(x_fit, eps)
    q_all = x_all @ projection
    reconstructed = q_all @ projection.T
    d_all = x_all - reconstructed
    y_all = (series.loss_before - series.loss_after) / np.maximum(np.abs(series.loss_before), eps)
    if float(np.var(y_all)) <= eps:
        invalid_reasons.append("Var(y_t) <= eps")

    delta_q_all = np.zeros_like(q_all)
    delta_q_all[1:] = q_all[1:] - q_all[:-1]
    delta_d_all = np.zeros_like(d_all)
    delta_d_all[1:] = d_all[1:] - d_all[:-1]
    r_all = np.sum(delta_d_all * delta_d_all, axis=1, keepdims=True)
    defect_energy_all = np.sum(d_all * d_all, axis=1, keepdims=True)
    candidate_all = np.concatenate([q_all, delta_q_all, defect_energy_all, r_all, q_all * r_all], axis=1)

    prefix_indices = np.arange(1, fit_count, dtype=np.int64)
    valid_indices = np.arange(fit_count, n_steps, dtype=np.int64)
    if prefix_indices.size < 16 or valid_indices.size < 16:
        invalid_reasons.append("fewer than 128 ordered steps for layer")

    y_prefix = y_all[prefix_indices]
    y_valid = y_all[valid_indices]
    w_prefix = series.tokens[prefix_indices]
    w_valid = series.tokens[valid_indices]
    e_const = weighted_mse(y_valid, np.full_like(y_valid, weighted_mean(y_prefix, w_prefix)), w_valid)
    if e_const <= eps:
        invalid_reasons.append("E_const <= eps")

    m_value, material_denominator = material_share(x_valid, projection, LAGS, eps)
    shuffled_valid = deterministic_shuffle(x_valid, series.layer)
    m_shuffle, _ = material_share(shuffled_valid, projection, LAGS, eps)
    if material_denominator <= eps:
        invalid_reasons.append("sum_lag ||C_lag||_F^2 <= eps")

    raw_budget_all = build_raw_budget_features(
        x_all=x_all,
        dx_all=np.vstack([np.zeros((1, x_all.shape[1]), dtype=np.float64), x_all[1:] - x_all[:-1]]),
        feature_count=candidate_all.shape[1],
        prefix_indices=prefix_indices,
        y_prefix=y_prefix,
        w_prefix=w_prefix,
    )
    shuffle_q_all = q_all.copy()
    shuffle_q_all[valid_indices] = deterministic_shuffle(q_all[valid_indices], series.layer)
    shuffle_delta_q_all = np.zeros_like(shuffle_q_all)
    shuffle_delta_q_all[1:] = shuffle_q_all[1:] - shuffle_q_all[:-1]
    shuffle_all = np.concatenate(
        [shuffle_q_all, shuffle_delta_q_all, defect_energy_all, r_all, shuffle_q_all * r_all],
        axis=1,
    )
    phase_q_all = q_all.copy()
    phase_q_all[valid_indices] = deterministic_phase_surrogate(q_all[valid_indices], series.layer)
    phase_delta_q_all = np.zeros_like(phase_q_all)
    phase_delta_q_all[1:] = phase_q_all[1:] - phase_q_all[:-1]
    phase_all = np.concatenate(
        [phase_q_all, phase_delta_q_all, defect_energy_all, r_all, phase_q_all * r_all],
        axis=1,
    )
    defect_only_all = np.concatenate([defect_energy_all, r_all], axis=1)

    arm_sources = {
        "candidate_cd": candidate_all,
        "raw_budget": raw_budget_all,
        "shuffle_charge": shuffle_all,
        "phase_charge": phase_all,
        "defect_only": defect_only_all,
    }
    arm_errors: Dict[str, float] = {}
    arm_scores: Dict[str, float] = {}
    arm_ridges: Dict[str, float] = {}
    for name, features in arm_sources.items():
        arm_error, ridge = tune_and_score_arm(
            x_prefix=features[prefix_indices],
            y_prefix=y_prefix,
            w_prefix=w_prefix,
            x_valid=features[valid_indices],
            y_valid=y_valid,
            w_valid=w_valid,
        )
        arm_errors[name] = arm_error
        arm_scores[name] = 1.0 - arm_error / max(e_const, eps)
        arm_ridges[name] = ridge

    baseline_competitors = (
        arm_errors["raw_budget"],
        arm_errors["shuffle_charge"],
        arm_errors["phase_charge"],
        arm_errors["defect_only"],
    )
    b_value = min(baseline_competitors)
    if b_value <= eps:
        invalid_reasons.append("B <= eps")
    if max(arm_errors.values()) - min(arm_errors.values()) <= TIE_TOLERANCE * max(e_const, eps):
        invalid_reasons.append("all arm errors tie within tolerance")

    q_valid = q_all[valid_indices]
    d_valid = d_all[valid_indices]
    delta_q_valid = delta_q_all[valid_indices]
    delta_d_valid = delta_d_all[valid_indices]
    q_var = float(np.sum(np.var(q_valid, axis=0)))
    dq_var = float(np.sum(np.var(delta_q_valid, axis=0)))
    d_var = float(np.sum(np.var(d_valid, axis=0)))
    dd_var = float(np.sum(np.var(delta_d_valid, axis=0)))
    if q_var <= eps or d_var <= eps or dq_var <= eps or dd_var <= eps:
        invalid_reasons.append("all-zero or constant q_t or d_t where denominator is required")
    t_q = q_var / max(dq_var, eps)
    t_d = d_var / max(dd_var, eps)
    timescale_ratio = t_q / max(t_d, eps)

    active_wall = series.step_wall_ms[valid_indices] - series.summary_extract_ms[valid_indices] - series.probe_update_ms[valid_indices]
    active_total = float(np.sum(active_wall))
    overhead_total = float(np.sum(series.summary_extract_ms[valid_indices] + series.probe_update_ms[valid_indices]))
    if active_total <= eps:
        invalid_reasons.append("delta_c_proxy cannot be computed when structural and credit inequalities otherwise pass")
        delta_c_proxy = math.inf
    else:
        delta_c_proxy = overhead_total / max(active_total, eps)

    delta_p_proxy = (b_value - arm_errors["candidate_cd"]) / max(b_value, eps) if b_value > eps else -math.inf
    metrics = {
        "prefix_materiality": prefix_materiality,
        "M": m_value,
        "M_time_shuffle": m_shuffle,
        "T_q": t_q,
        "T_d": t_d,
        "timescale_ratio": timescale_ratio,
        "delta_c_proxy": delta_c_proxy,
        "delta_p_proxy": delta_p_proxy,
        "E_const": e_const,
        "B": b_value,
        "phase_surrogate": "deterministic_circular_shift",
    }
    margins = {
        "materiality_floor": m_value - 0.25,
        "shuffle_gap": (m_value - m_shuffle) - 0.10,
        "shuffle_ratio": (m_value / max(m_shuffle, eps)) - 1.50,
        "timescale_ratio": timescale_ratio - 3.0,
        "score_floor": arm_scores["candidate_cd"] - 0.05,
        "delta_p_net": delta_p_proxy - (0.10 + 1.10 * delta_c_proxy),
        "beats_raw_budget": arm_errors["raw_budget"] - arm_errors["candidate_cd"],
        "beats_shuffle_charge": arm_errors["shuffle_charge"] - arm_errors["candidate_cd"],
        "beats_phase_charge": arm_errors["phase_charge"] - arm_errors["candidate_cd"],
        "beats_defect_only": arm_errors["defect_only"] - arm_errors["candidate_cd"],
    }
    layer_pass = (
        not invalid_reasons
        and margins["materiality_floor"] >= 0.0
        and margins["shuffle_gap"] >= 0.0
        and margins["shuffle_ratio"] >= 0.0
        and margins["timescale_ratio"] >= 0.0
        and margins["score_floor"] >= 0.0
        and margins["delta_p_net"] >= 0.0
        and margins["beats_raw_budget"] > 0.0
        and margins["beats_shuffle_charge"] > 0.0
        and margins["beats_phase_charge"] > 0.0
        and margins["beats_defect_only"] > 0.0
    )
    return {
        "layer": series.layer,
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "selected_k": selected_k,
        "k_max": k_max,
        "metrics": metrics,
        "arm_errors": arm_errors,
        "arm_scores": arm_scores,
        "arm_ridges": arm_ridges,
        "inequality_margins": margins,
        "weight": float(np.sum(w_valid)),
        "layer_pass": layer_pass,
    }


def build_report(mode: str, trace: Mapping[str, object]) -> Dict[str, object]:
    layers, global_invalid_reasons, meta = validate_trace(trace)
    report_layers = []
    valid_layers = []
    for layer in layers:
        evaluated = evaluate_layer(layer, float(meta["fit_fraction"]), int(meta["summary_dim"]), EPS)
        report_layers.append(
            {
                "layer": evaluated["layer"],
                "valid": evaluated["valid"],
                "invalid_reasons": evaluated["invalid_reasons"],
                "selected_k": evaluated["selected_k"],
                "k_max": evaluated["k_max"],
                "metrics": evaluated["metrics"],
                "arm_errors": evaluated["arm_errors"],
                "arm_scores": evaluated["arm_scores"],
                "inequality_margins": evaluated["inequality_margins"],
            }
        )
        if evaluated["valid"]:
            valid_layers.append(evaluated)

    invalid_reasons = list(global_invalid_reasons)
    if not valid_layers:
        if not invalid_reasons:
            invalid_reasons.append("no valid layer after schema and metric validation")
        verdict = "INVALID"
        aggregate = {
            "valid_layer_count": 0,
            "passing_layer_count": 0,
            "weighted_median": {},
            "inequality_margins": {},
            "registered_failures": [],
        }
    else:
        weights = [float(layer["weight"]) for layer in valid_layers]
        metric_names = ("M", "M_time_shuffle", "T_q", "T_d", "timescale_ratio", "delta_c_proxy", "delta_p_proxy")
        margin_names = (
            "materiality_floor",
            "shuffle_gap",
            "shuffle_ratio",
            "timescale_ratio",
            "score_floor",
            "delta_p_net",
            "beats_raw_budget",
            "beats_shuffle_charge",
            "beats_phase_charge",
            "beats_defect_only",
        )
        weighted_metrics = {
            name: weighted_median([float(layer["metrics"][name]) for layer in valid_layers], weights) for name in metric_names
        }
        weighted_margins = {
            name: weighted_median([float(layer["inequality_margins"][name]) for layer in valid_layers], weights)
            for name in margin_names
        }
        passing_layer_count = sum(1 for layer in valid_layers if bool(layer["layer_pass"]))
        registered_failures = [name for name, value in weighted_margins.items() if value < 0.0]
        enough_layers_pass = passing_layer_count / len(valid_layers) >= 0.5
        if invalid_reasons:
            verdict = "INVALID"
        elif enough_layers_pass and not registered_failures:
            verdict = "PASS"
        else:
            verdict = "KILL"
        aggregate = {
            "valid_layer_count": len(valid_layers),
            "passing_layer_count": passing_layer_count,
            "weighted_median": weighted_metrics,
            "inequality_margins": weighted_margins,
            "registered_failures": registered_failures,
        }

    return {
        "schema": REPORT_SCHEMA,
        "mode": mode,
        "verdict": verdict,
        "invalid_reasons": invalid_reasons,
        "eps": EPS,
        "meta": {
            "trace_schema": TRACE_SCHEMA,
            "fit_fraction": meta.get("fit_fraction"),
            "summary_dim": meta.get("summary_dim"),
            "seed": meta.get("seed"),
            "projection_source": meta.get("projection_source"),
            "phase_surrogate": meta.get("phase_surrogate", "deterministic_circular_shift"),
        },
        "aggregate": aggregate,
        "layers": report_layers,
    }


def emit_stdout(report: Mapping[str, object], trace: Mapping[str, object]) -> None:
    state_items = trace.get("states", []) if isinstance(trace, dict) else []
    layer_count = len({int(item["layer"]) for item in state_items if isinstance(item, dict) and "layer" in item})
    step_items = trace.get("steps", []) if isinstance(trace, dict) else []
    trace_meta = trace.get("meta", {}) if isinstance(trace, dict) else {}
    activation = {
        "mode": report["mode"],
        "trace_schema": report["meta"]["trace_schema"],
        "seed": report["meta"].get("seed", trace_meta.get("seed")),
        "summary_dim": report["meta"].get("summary_dim", trace_meta.get("summary_dim")),
        "step_count": len(step_items),
        "layer_count": layer_count,
    }
    metrics = {
        "verdict": report["verdict"],
        "valid_layer_count": report["aggregate"]["valid_layer_count"],
        "passing_layer_count": report["aggregate"]["passing_layer_count"],
        "weighted_median": report["aggregate"]["weighted_median"],
    }
    decision = {
        "verdict": report["verdict"],
        "invalid_reasons": report["invalid_reasons"],
        "inequality_margins": report["aggregate"]["inequality_margins"],
        "registered_failures": report["aggregate"].get("registered_failures", []),
    }
    print("ACTIVATION " + json.dumps(activation, sort_keys=True))
    print("METRICS " + json.dumps(metrics, sort_keys=True))
    print("DECISION " + json.dumps(decision, sort_keys=True))


def main() -> int:
    args = parse_args()
    if args.self_test:
        trace = build_self_test_trace()
        mode = "self_test"
    else:
        trace = load_trace(args.trace)
        mode = "trace"
    report = build_report(mode=mode, trace=trace)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    emit_stdout(report, trace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
