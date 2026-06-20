from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


PROBE_LABEL = "r397.roq.diagonal_rescale.v1"
SELF_TEST_REASON = "synthetic_self_test_validates_diagonal_rescale_contract_only"
SYNTHETIC_ONLY_REASON = "diagnostic_only_not_production_evidence"
EPS_BASE = 1e-12
EPS_ENERGY = 1e-12
EPS_PROGRESS = 1e-12
TAU_ENERGY = 1e-9
TAU_TANGENT = 0.02
ETA_H_PREFERRED = 0.98
ETA_H_WEAK = 0.95
KAPPA_MAX = 1e8
MAX_UNOBSERVABLE_FRACTION = 1e-9


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm_sq(values: list[float]) -> float:
    return _dot(values, values)


def _fro_sq(matrix: list[list[float]]) -> float:
    return sum(_norm_sq(row) for row in matrix)


def _zeros_like_matrix(rows: int, cols: int) -> list[list[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def _clone_matrix(matrix: list[list[float]]) -> list[list[float]]:
    return [row[:] for row in matrix]


def _column(matrix: list[list[float]], column_index: int) -> list[float]:
    return [row[column_index] for row in matrix]


def _matrix_vector(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [_dot(row, vector) for row in matrix]


def _vector_add(left: list[float], right: list[float]) -> list[float]:
    return [a + b for a, b in zip(left, right)]


def _vector_sub(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def _matrix_add(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [[a + b for a, b in zip(left_row, right_row)] for left_row, right_row in zip(left, right)]


def _matrix_sub(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [[a - b for a, b in zip(left_row, right_row)] for left_row, right_row in zip(left, right)]


def _scale_vector(values: list[float], scalar: float) -> list[float]:
    return [scalar * value for value in values]


def _scale_matrix(matrix: list[list[float]], scalar: float) -> list[list[float]]:
    return [[scalar * value for value in row] for row in matrix]


def _all_finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _all_finite_matrix(matrix: list[list[float]]) -> bool:
    return all(_all_finite(row) for row in matrix)


def _count_active_update(update: dict[str, Any], eps: float) -> int:
    count = 0
    count += sum(1 for value in update["A"] if abs(value) > eps)
    count += sum(1 for value in update["B"] if abs(value) > eps)
    count += sum(
        1
        for row in update["C"]
        for value in row
        if abs(value) > eps
    )
    if update.get("h") is not None:
        count += sum(1 for value in update["h"] if abs(value) > eps)
    return count


def _build_weights(scope: dict[str, Any]) -> dict[str, float]:
    dimension = len(scope["A"])
    eps = EPS_BASE
    weights = {
        "A": 1.0 / max(eps, _norm_sq(scope["A"]) / dimension),
        "B": 1.0 / max(eps, _norm_sq(scope["B"]) / dimension),
        "C": 1.0 / max(eps, _fro_sq(scope["C"]) / dimension),
        "h": 0.0,
    }
    if scope.get("h") is not None:
        weights["h"] = 1.0 / max(eps, _norm_sq(scope["h"]) / dimension)
    return weights


def _update_energy(update: dict[str, Any], weights: dict[str, float]) -> float:
    total = 0.0
    total += weights["A"] * _norm_sq(update["A"])
    total += weights["B"] * _norm_sq(update["B"])
    total += weights["C"] * _fro_sq(update["C"])
    if update.get("h") is not None:
        total += weights["h"] * _norm_sq(update["h"])
    return total


def _gauge_denominator(
    scope: dict[str, Any],
    weights: dict[str, float],
    axis_index: int,
) -> float:
    column_values = _column(scope["C"], axis_index)
    denominator = weights["B"] * (scope["B"][axis_index] ** 2)
    denominator += weights["C"] * _norm_sq(column_values)
    if scope.get("h") is not None:
        denominator += weights["h"] * (scope["h"][axis_index] ** 2)
    return denominator


def _gauge_update(scope: dict[str, Any], q_values: list[float]) -> dict[str, Any]:
    dimension = len(scope["A"])
    delta_a = [0.0 for _ in range(dimension)]
    delta_b = [q_values[index] * scope["B"][index] for index in range(dimension)]
    delta_c = _zeros_like_matrix(dimension, dimension)
    for axis_index in range(dimension):
        for row_index in range(dimension):
            delta_c[row_index][axis_index] = -q_values[axis_index] * scope["C"][row_index][axis_index]
    delta_h = None
    if scope.get("h") is not None:
        delta_h = [q_values[index] * scope["h"][index] for index in range(dimension)]
    return {
        "A": delta_a,
        "B": delta_b,
        "C": delta_c,
        "h": delta_h,
    }


def _zero_update(scope: dict[str, Any]) -> dict[str, Any]:
    dimension = len(scope["A"])
    return {
        "A": [0.0 for _ in range(dimension)],
        "B": [0.0 for _ in range(dimension)],
        "C": _zeros_like_matrix(dimension, dimension),
        "h": None if scope.get("h") is None else [0.0 for _ in range(dimension)],
    }


def _sum_updates(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    delta_h = None
    if left.get("h") is not None or right.get("h") is not None:
        left_h = left.get("h") or [0.0 for _ in range(len(left["A"]))]
        right_h = right.get("h") or [0.0 for _ in range(len(right["A"]))]
        delta_h = _vector_add(left_h, right_h)
    return {
        "A": _vector_add(left["A"], right["A"]),
        "B": _vector_add(left["B"], right["B"]),
        "C": _matrix_add(left["C"], right["C"]),
        "h": delta_h,
    }


def _subtract_updates(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    delta_h = None
    if left.get("h") is not None or right.get("h") is not None:
        left_h = left.get("h") or [0.0 for _ in range(len(left["A"]))]
        right_h = right.get("h") or [0.0 for _ in range(len(right["A"]))]
        delta_h = _vector_sub(left_h, right_h)
    return {
        "A": _vector_sub(left["A"], right["A"]),
        "B": _vector_sub(left["B"], right["B"]),
        "C": _matrix_sub(left["C"], right["C"]),
        "h": delta_h,
    }


def _observable_gauge_energy(
    scope: dict[str, Any],
    update: dict[str, Any],
    weights: dict[str, float],
    active_axes: set[int],
) -> float:
    total = 0.0
    for axis_index in range(len(scope["A"])):
        if axis_index in active_axes:
            continue
        total += weights["B"] * (update["B"][axis_index] ** 2)
        total += weights["C"] * _norm_sq(_column(update["C"], axis_index))
        if update.get("h") is not None:
            total += weights["h"] * (update["h"][axis_index] ** 2)
    return total


def _cb(scope: dict[str, Any]) -> list[float]:
    return _matrix_vector(scope["C"], scope["B"])


def _delta_cb(scope: dict[str, Any], update: dict[str, Any]) -> list[float]:
    delta_from_b = _matrix_vector(scope["C"], update["B"])
    delta_from_c = _matrix_vector(update["C"], scope["B"])
    return _vector_add(delta_from_b, delta_from_c)


def _progress_proxy_factory(scope: dict[str, Any]) -> tuple[list[float], Any]:
    current_cb = _cb(scope)
    target_cb = _vector_add(current_cb, [0.6, -0.3, 0.4, -0.2])
    residual = _vector_sub(target_cb, current_cb)

    def _proxy(update: dict[str, Any]) -> float:
        return _dot(residual, _delta_cb(scope, update))

    return target_cb, _proxy


def _shuffle_tangent_q(q_values: list[float]) -> list[float]:
    if len(q_values) <= 1:
        return q_values[:]
    return q_values[1:] + q_values[:1]


def _projection_report(
    scope: dict[str, Any],
    update: dict[str, Any],
    *,
    requested_scope: str,
    require_progress: bool,
    progress_proxy: Any | None,
) -> dict[str, Any]:
    invalid_codes: list[str] = []
    if requested_scope != "diagonal_rescale":
        invalid_codes.append("INVALID_FULL_SIMILARITY_NOT_COMPATIBLE")

    if not _all_finite(scope["A"]) or not _all_finite(scope["B"]) or not _all_finite_matrix(scope["C"]):
        invalid_codes.append("INVALID_NONFINITE_PROJECTION")
    if not _all_finite(update["A"]) or not _all_finite(update["B"]) or not _all_finite_matrix(update["C"]):
        invalid_codes.append("INVALID_NONFINITE_PROJECTION")
    if scope.get("h") is not None and not _all_finite(scope["h"]):
        invalid_codes.append("INVALID_NONFINITE_PROJECTION")
    if update.get("h") is not None and not _all_finite(update["h"]):
        invalid_codes.append("INVALID_NONFINITE_PROJECTION")

    weights = _build_weights(scope)
    active_update_count = _count_active_update(update, EPS_BASE)
    if active_update_count == 0:
        invalid_codes.append("INVALID_NO_ACTIVE_UPDATE")

    denominators = [_gauge_denominator(scope, weights, axis_index) for axis_index in range(len(scope["A"]))]
    active_axes = {axis_index for axis_index, value in enumerate(denominators) if value >= EPS_BASE}
    if not active_axes:
        invalid_codes.append("INVALID_NO_OBSERVABLE_GAUGE")

    q_values = [0.0 for _ in scope["A"]]
    for axis_index in active_axes:
        numerator = weights["B"] * scope["B"][axis_index] * update["B"][axis_index]
        numerator -= weights["C"] * _dot(_column(scope["C"], axis_index), _column(update["C"], axis_index))
        if scope.get("h") is not None and update.get("h") is not None:
            numerator += weights["h"] * scope["h"][axis_index] * update["h"][axis_index]
        q_values[axis_index] = numerator / denominators[axis_index]

    tangent_update = _gauge_update(scope, q_values)
    horizontal_update = _subtract_updates(update, tangent_update)
    total_energy = _update_energy(update, weights)
    tangent_energy = _update_energy(tangent_update, weights)
    horizontal_energy = _update_energy(horizontal_update, weights)
    section_condition = math.inf
    if active_axes:
        active_denominators = [denominators[index] for index in sorted(active_axes)]
        section_condition = max(active_denominators) / min(active_denominators)
        if not math.isfinite(section_condition) or section_condition > KAPPA_MAX:
            invalid_codes.append("INVALID_UNSTABLE_SECTION")

    excluded_energy = _observable_gauge_energy(scope, update, weights, active_axes)
    if total_energy <= EPS_ENERGY:
        invalid_codes.append("INVALID_ZERO_DENOMINATOR")
    elif excluded_energy / total_energy > MAX_UNOBSERVABLE_FRACTION:
        invalid_codes.append("INVALID_UNSTABLE_SECTION")

    energy_gap = abs(total_energy - tangent_energy - horizontal_energy)
    rho_orbit = 0.0
    if total_energy > EPS_ENERGY:
        rho_orbit = tangent_energy / total_energy

    p_original = None
    p_horizontal = None
    p_tangent = None
    p_shuffle = None
    eta_h = None
    if progress_proxy is not None:
        p_original = float(progress_proxy(update))
        p_horizontal = float(progress_proxy(horizontal_update))
        p_tangent = float(progress_proxy(tangent_update))
        shuffled_q = _shuffle_tangent_q(q_values)
        p_shuffle = float(progress_proxy(_gauge_update(scope, shuffled_q)))
        progress_values = [p_original, p_horizontal, p_tangent, p_shuffle]
        if not _all_finite(progress_values):
            invalid_codes.append("INVALID_PROGRESS_UNOBSERVABLE")
        elif abs(p_original) <= EPS_PROGRESS:
            if require_progress:
                invalid_codes.append("INVALID_PROGRESS_UNOBSERVABLE")
        else:
            eta_h = p_horizontal / p_original
    elif require_progress:
        invalid_codes.append("INVALID_PROGRESS_UNOBSERVABLE")

    if not math.isfinite(total_energy) or not math.isfinite(tangent_energy) or not math.isfinite(horizontal_energy):
        invalid_codes.append("INVALID_NONFINITE_PROJECTION")
    if not math.isfinite(rho_orbit) or rho_orbit < -EPS_BASE or rho_orbit > 1.0 + EPS_BASE:
        invalid_codes.append("INVALID_NONFINITE_PROJECTION")
    if total_energy > EPS_ENERGY and energy_gap > TAU_ENERGY * total_energy:
        invalid_codes.append("INVALID_NONFINITE_PROJECTION")

    delta_c_estimate = 0.0
    scale_h = None
    g_net = None
    if eta_h is not None:
        residual = max(EPS_BASE, 1.0 - min(max(rho_orbit, 0.0), 1.0 - EPS_BASE))
        scale_h = 1.0 / math.sqrt(residual)
        g_net = eta_h * scale_h / (1.0 + delta_c_estimate)

    verdict = "INVALID" if invalid_codes else "PASS"
    return {
        "verdict": verdict,
        "invalid_codes": sorted(set(invalid_codes)),
        "weights": weights,
        "q_values": q_values,
        "E_total": total_energy,
        "E_tan": tangent_energy,
        "E_hor": horizontal_energy,
        "rho_orbit": rho_orbit,
        "active_update_count": active_update_count,
        "active_gauge_axis_count": len(active_axes),
        "section_condition": section_condition,
        "p_original": p_original,
        "p_horizontal": p_horizontal,
        "p_tangent": p_tangent,
        "p_shuffle": p_shuffle,
        "eta_h": eta_h,
        "delta_c_estimate": delta_c_estimate,
        "G_net": g_net,
        "energy_gap": energy_gap,
        "tangent_update": tangent_update,
        "horizontal_update": horizontal_update,
    }


def _base_scope() -> dict[str, Any]:
    return {
        "A": [0.5, 0.8, 1.1, 1.4],
        "B": [1.0, -2.0, 1.5, 0.75],
        "C": [
            [1.2, -0.4, 0.3, 0.8],
            [0.5, 1.1, -0.7, 0.2],
            [-0.6, 0.9, 1.4, -1.0],
            [0.3, -0.2, 0.6, 1.3],
        ],
        "h": None,
    }


def _is_close(left: float, right: float, tolerance: float) -> bool:
    return abs(left - right) <= tolerance


def _horizontal_fixture(scope: dict[str, Any], axis_index: int, delta_b_value: float) -> dict[str, Any]:
    update = _zero_update(scope)
    update["B"][axis_index] = delta_b_value
    weights = _build_weights(scope)
    column_values = _column(scope["C"], axis_index)
    column_norm_sq = _norm_sq(column_values)
    scale = weights["B"] * scope["B"][axis_index] * delta_b_value
    scale /= weights["C"] * column_norm_sq
    for row_index in range(len(scope["A"])):
        update["C"][row_index][axis_index] = scale * scope["C"][row_index][axis_index]
    return update


def _test_pure_gauge() -> dict[str, Any]:
    scope = _base_scope()
    q_values = [0.2, -0.1, 0.15, -0.05]
    report = _projection_report(
        scope,
        _gauge_update(scope, q_values),
        requested_scope="diagonal_rescale",
        require_progress=False,
        progress_proxy=None,
    )
    horizontal_ratio = math.sqrt(report["E_hor"] / report["E_total"]) if report["E_total"] > EPS_ENERGY else math.inf
    passed = report["verdict"] == "PASS"
    passed = passed and _is_close(report["rho_orbit"], 1.0, 1e-9)
    passed = passed and horizontal_ratio <= 1e-8
    return {
        "name": "pure_diagonal_gauge_update",
        "verdict": "PASS" if passed else "INVALID",
        "expected": {
            "rho_orbit": 1.0,
            "horizontal_residual_ratio_max": 1e-8,
        },
        "observed": {
            "rho_orbit": report["rho_orbit"],
            "horizontal_residual_ratio": horizontal_ratio,
            "E_total": report["E_total"],
            "E_tan": report["E_tan"],
            "E_hor": report["E_hor"],
            "active_update_count": report["active_update_count"],
            "active_gauge_axis_count": report["active_gauge_axis_count"],
            "section_condition": report["section_condition"],
            "invalid_codes": report["invalid_codes"],
        },
        "invalid_codes": report["invalid_codes"],
    }


def _test_pure_horizontal_a() -> dict[str, Any]:
    scope = _base_scope()
    update = _zero_update(scope)
    update["A"] = [0.4, -0.3, 0.2, -0.1]
    report = _projection_report(
        scope,
        update,
        requested_scope="diagonal_rescale",
        require_progress=False,
        progress_proxy=None,
    )
    a_match = max(abs(a - b) for a, b in zip(update["A"], report["horizontal_update"]["A"]))
    b_match = max(abs(value) for value in report["horizontal_update"]["B"])
    c_match = max(abs(value) for row in report["horizontal_update"]["C"] for value in row)
    passed = report["verdict"] == "PASS"
    passed = passed and _is_close(report["rho_orbit"], 0.0, 1e-12)
    passed = passed and a_match <= 1e-12 and b_match <= 1e-12 and c_match <= 1e-12
    return {
        "name": "pure_horizontal_a_update",
        "verdict": "PASS" if passed else "INVALID",
        "expected": {
            "rho_orbit": 0.0,
            "horizontal_matches_original_max": 1e-12,
        },
        "observed": {
            "rho_orbit": report["rho_orbit"],
            "a_match_max_abs": a_match,
            "b_horizontal_max_abs": b_match,
            "c_horizontal_max_abs": c_match,
            "E_total": report["E_total"],
            "E_tan": report["E_tan"],
            "E_hor": report["E_hor"],
            "invalid_codes": report["invalid_codes"],
        },
        "invalid_codes": report["invalid_codes"],
    }


def _test_mixed_known_update() -> dict[str, Any]:
    scope = _base_scope()
    q_values = [0.1, -0.05, 0.08, 0.12]
    gauge_update = _gauge_update(scope, q_values)
    horizontal_update = _zero_update(scope)
    horizontal_update["A"] = [0.3, 0.0, -0.2, 0.1]
    update = _sum_updates(gauge_update, horizontal_update)
    report = _projection_report(
        scope,
        update,
        requested_scope="diagonal_rescale",
        require_progress=False,
        progress_proxy=None,
    )
    weights = _build_weights(scope)
    expected_tangent_energy = _update_energy(gauge_update, weights)
    expected_horizontal_energy = _update_energy(horizontal_update, weights)
    expected_rho = expected_tangent_energy / (expected_tangent_energy + expected_horizontal_energy)
    passed = report["verdict"] == "PASS"
    passed = passed and _is_close(report["rho_orbit"], expected_rho, 1e-9)
    return {
        "name": "mixed_known_update",
        "verdict": "PASS" if passed else "INVALID",
        "expected": {
            "rho_orbit": expected_rho,
            "construction": "u=L(q)+r_with_A_only_horizontal_component",
        },
        "observed": {
            "rho_orbit": report["rho_orbit"],
            "expected_rho_orbit": expected_rho,
            "E_total": report["E_total"],
            "E_tan": report["E_tan"],
            "E_hor": report["E_hor"],
            "invalid_codes": report["invalid_codes"],
        },
        "invalid_codes": report["invalid_codes"],
    }


def _test_degenerate_axis() -> dict[str, Any]:
    scope = _base_scope()
    degenerate_scope = {
        "A": scope["A"][:],
        "B": scope["B"][:],
        "C": _clone_matrix(scope["C"]),
        "h": None,
    }
    degenerate_scope["B"][0] = 0.0
    for row_index in range(len(degenerate_scope["C"])):
        degenerate_scope["C"][row_index][0] = 0.0

    update_valid = _gauge_update(degenerate_scope, [0.0, 0.2, 0.0, 0.0])
    report_valid = _projection_report(
        degenerate_scope,
        update_valid,
        requested_scope="diagonal_rescale",
        require_progress=False,
        progress_proxy=None,
    )

    zero_scope = {
        "A": scope["A"][:],
        "B": [0.0, 0.0, 0.0, 0.0],
        "C": _zeros_like_matrix(4, 4),
        "h": None,
    }
    update_invalid = _zero_update(zero_scope)
    update_invalid["B"][1] = 1.0
    report_invalid = _projection_report(
        zero_scope,
        update_invalid,
        requested_scope="diagonal_rescale",
        require_progress=False,
        progress_proxy=None,
    )

    passed = report_valid["verdict"] == "PASS"
    passed = passed and report_valid["active_gauge_axis_count"] == 3
    passed = passed and report_invalid["verdict"] == "INVALID"
    passed = passed and "INVALID_NO_OBSERVABLE_GAUGE" in report_invalid["invalid_codes"]
    return {
        "name": "degenerate_axis_handling",
        "verdict": "PASS" if passed else "INVALID",
        "expected": {
            "excluded_axis_count": 1,
            "all_zero_gauge_invalid_code": "INVALID_NO_OBSERVABLE_GAUGE",
        },
        "observed": {
            "active_gauge_axis_count_after_exclusion": report_valid["active_gauge_axis_count"],
            "excluded_case_invalid_codes": report_valid["invalid_codes"],
            "all_zero_gauge_invalid_codes": report_invalid["invalid_codes"],
        },
        "invalid_codes": sorted(set(report_valid["invalid_codes"] + report_invalid["invalid_codes"])),
    }


def _test_forbidden_full_similarity() -> dict[str, Any]:
    scope = _base_scope()
    update = _gauge_update(scope, [0.05, -0.03, 0.02, 0.01])
    report = _projection_report(
        scope,
        update,
        requested_scope="full_similarity",
        require_progress=False,
        progress_proxy=None,
    )
    passed = report["verdict"] == "INVALID"
    passed = passed and report["invalid_codes"] == ["INVALID_FULL_SIMILARITY_NOT_COMPATIBLE"]
    return {
        "name": "forbidden_full_similarity_request",
        "verdict": "PASS" if passed else "INVALID",
        "expected": {
            "invalid_code": "INVALID_FULL_SIMILARITY_NOT_COMPATIBLE",
        },
        "observed": {
            "invalid_codes": report["invalid_codes"],
            "rho_orbit": report["rho_orbit"],
        },
        "invalid_codes": report["invalid_codes"],
    }


def _test_progress_safety() -> dict[str, Any]:
    scope = _base_scope()
    _, proxy = _progress_proxy_factory(scope)
    q_values = [0.12, -0.08, 0.06, 0.04]
    gauge_update = _gauge_update(scope, q_values)
    horizontal_update = _horizontal_fixture(scope, axis_index=2, delta_b_value=0.4)
    update = _sum_updates(gauge_update, horizontal_update)
    report = _projection_report(
        scope,
        update,
        requested_scope="diagonal_rescale",
        require_progress=True,
        progress_proxy=proxy,
    )
    tangent_ratio = abs(report["p_tangent"]) / abs(report["p_original"]) if report["p_original"] else math.inf
    passed = report["verdict"] == "PASS"
    passed = passed and tangent_ratio <= TAU_TANGENT
    passed = passed and report["eta_h"] is not None and abs(report["eta_h"] - 1.0) <= 1e-9
    return {
        "name": "progress_safety_sanity",
        "verdict": "PASS" if passed else "INVALID",
        "expected": {
            "p_tangent_ratio_max": TAU_TANGENT,
            "eta_h": 1.0,
            "horizontal_progress_preservation_min": ETA_H_PREFERRED,
        },
        "observed": {
            "p_original": report["p_original"],
            "p_horizontal": report["p_horizontal"],
            "p_tangent": report["p_tangent"],
            "p_shuffle": report["p_shuffle"],
            "eta_h": report["eta_h"],
            "rho_orbit": report["rho_orbit"],
            "G_net": report["G_net"],
            "invalid_codes": report["invalid_codes"],
        },
        "invalid_codes": report["invalid_codes"],
    }


def _run_self_tests() -> dict[str, Any]:
    self_tests = [
        _test_pure_gauge(),
        _test_pure_horizontal_a(),
        _test_mixed_known_update(),
        _test_degenerate_axis(),
        _test_forbidden_full_similarity(),
        _test_progress_safety(),
    ]
    verdict = "PASS" if all(test["verdict"] == "PASS" for test in self_tests) else "INVALID"
    invalid_codes = sorted(
        {
            code
            for test in self_tests
            for code in test["invalid_codes"]
            if test["verdict"] != "PASS"
        }
    )
    return {
        "probe_label": PROBE_LABEL,
        "mode": "self-test",
        "verdict": verdict,
        "self_tests": self_tests,
        "summary": {
            "diagnostic_only": True,
            "synthetic_only": True,
            "reason": SELF_TEST_REASON,
            "finding_status": SYNTHETIC_ONLY_REASON,
            "passed_tests": sum(1 for test in self_tests if test["verdict"] == "PASS"),
            "total_tests": len(self_tests),
        },
        "thresholds": {
            "eps_base": EPS_BASE,
            "eps_energy": EPS_ENERGY,
            "eps_progress": EPS_PROGRESS,
            "tau_energy": TAU_ENERGY,
            "tau_tangent": TAU_TANGENT,
            "eta_h_preferred": ETA_H_PREFERRED,
            "eta_h_weak": ETA_H_WEAK,
            "kappa_max": KAPPA_MAX,
            "max_unobservable_fraction": MAX_UNOBSERVABLE_FRACTION,
        },
        "metric": {
            "name": "weighted_euclidean_default_contract_metric",
            "scope": "A_vector_B_vector_C_matrix",
            "formula": {
                "w_A": "1 / max(eps, ||A||_2^2 / n)",
                "w_B": "1 / max(eps, ||B||_2^2 / n)",
                "w_C": "1 / max(eps, ||C||_F^2 / n)",
            },
            "notes": "Synthetic deterministic fixtures only. Delta A is always horizontal under this restricted contract.",
        },
        "invalid_codes": invalid_codes,
    }


def _run_trace_mode(trace_path: Path) -> dict[str, Any]:
    return {
        "probe_label": PROBE_LABEL,
        "mode": "trace",
        "verdict": "INVALID",
        "self_tests": [],
        "summary": {
            "diagnostic_only": True,
            "synthetic_only": False,
            "reason": "trace_mode_is_read_only_and_not_part_of_slice_1_validation",
            "trace_path": str(trace_path),
        },
        "thresholds": {
            "eps_base": EPS_BASE,
            "eps_energy": EPS_ENERGY,
            "eps_progress": EPS_PROGRESS,
            "tau_energy": TAU_ENERGY,
            "tau_tangent": TAU_TANGENT,
            "eta_h_preferred": ETA_H_PREFERRED,
            "eta_h_weak": ETA_H_WEAK,
            "kappa_max": KAPPA_MAX,
            "max_unobservable_fraction": MAX_UNOBSERVABLE_FRACTION,
        },
        "metric": {
            "name": "weighted_euclidean_default_contract_metric",
            "scope": "A_vector_B_vector_C_matrix",
            "notes": "Trace mode is intentionally not used for Slice 1 validation.",
        },
        "invalid_codes": ["INVALID_MISSING_REALIZATION_FIELDS"],
    }


def _write_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R397 diagonal-rescale ROQ probe")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--self-test", action="store_true", help="Run deterministic synthetic self-test")
    mode_group.add_argument("--trace", type=Path, help="Optional read-only trace path")
    parser.add_argument("--report", type=Path, required=True, help="Write JSON report to this path")
    args = parser.parse_args(argv)

    mode = "self-test" if args.self_test else "trace"
    print(f"R397_ROQ_PROBE_ACTIVE label={PROBE_LABEL} mode={mode}")

    if args.self_test:
        payload = _run_self_tests()
    else:
        payload = _run_trace_mode(args.trace)

    _write_report(args.report, payload)
    print(f"R397_ROQ_PROBE_VERDICT verdict={payload['verdict']} report={args.report}")
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
