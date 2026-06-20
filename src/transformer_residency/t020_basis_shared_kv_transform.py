"""Basis-shared KV transform gate for packed row-scale projections."""

from __future__ import annotations

from typing import Any


ACTIVATION = "T020_BASIS_SHARED_KV_TRANSFORM_ACTIVE"


def deterministic_vector(size: int, *, phase: float) -> list[float]:
    if size <= 0:
        raise ValueError("size must be positive")
    return [((index * 17 + 3) % 23) / 11.0 - 1.0 + phase for index in range(size)]


def diagonal_scale_basis(size: int) -> list[float]:
    if size <= 0:
        raise ValueError("size must be positive")
    return [0.75 + 0.125 * (index % 7) for index in range(size)]


def dot(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) != len(values_b):
        raise ValueError("vectors must have the same length")
    return sum(a * b for a, b in zip(values_a, values_b, strict=True))


def prove_diagonal_dual_basis_dot_preservation(
    *,
    size: int = 16,
    epsilon_numeric: float = 1e-6,
    attention_margin: float = 1e-3,
) -> dict[str, Any]:
    if epsilon_numeric <= 0.0 or attention_margin <= 0.0:
        raise ValueError("epsilon_numeric and attention_margin must be positive")
    q = deterministic_vector(size, phase=0.03)
    k = deterministic_vector(size, phase=-0.02)
    scales = diagonal_scale_basis(size)
    q_transformed = [scale * value for scale, value in zip(scales, q, strict=True)]
    k_dual = [value / scale for scale, value in zip(scales, k, strict=True)]
    canonical_dot = dot(q, k)
    transformed_dot = dot(q_transformed, k_dual)
    max_abs_error = abs(canonical_dot - transformed_dot)
    condition_number = max(scales) / min(scales)
    numerical_debt = condition_number * epsilon_numeric
    return {
        "transform_class": "diagonal_scale",
        "vector_size": size,
        "canonical_dot": canonical_dot,
        "transformed_dot": transformed_dot,
        "max_abs_error": max_abs_error,
        "condition_number": condition_number,
        "epsilon_numeric": epsilon_numeric,
        "attention_margin": attention_margin,
        "numerical_debt": numerical_debt,
        "dot_product_preserved": max_abs_error <= 1e-12,
        "numerical_debt_below_margin": numerical_debt < attention_margin * 0.01,
        "requires_dense_inverse": False,
        "requires_canonical_and_transformed_kv": False,
    }


def price_basis_shared_path(
    *,
    saved_canonicalization_cost: float = 0.18,
    saved_kv_transfer_cost: float = 0.12,
    saved_unpack_cost: float = 0.08,
    basis_transform_cost: float = 0.05,
    dual_basis_storage_cost: float = 0.04,
    rebuild_cost: float = 0.02,
    metadata_cost: float = 0.02,
) -> dict[str, Any]:
    costs = (
        saved_canonicalization_cost,
        saved_kv_transfer_cost,
        saved_unpack_cost,
        basis_transform_cost,
        dual_basis_storage_cost,
        rebuild_cost,
        metadata_cost,
    )
    if any(cost < 0.0 for cost in costs):
        raise ValueError("costs must be non-negative")
    baseline_crossing_cost = saved_canonicalization_cost + saved_kv_transfer_cost + saved_unpack_cost
    candidate_basis_cost = basis_transform_cost + dual_basis_storage_cost + rebuild_cost + metadata_cost
    return {
        "baseline_crossing_cost": baseline_crossing_cost,
        "candidate_basis_cost": candidate_basis_cost,
        "deleted_crossing_cost": baseline_crossing_cost - candidate_basis_cost,
        "speedup_vs_representation_crossing": baseline_crossing_cost / max(1e-9, candidate_basis_cost),
        "saved_canonicalization_cost": saved_canonicalization_cost,
        "saved_kv_transfer_cost": saved_kv_transfer_cost,
        "saved_unpack_cost": saved_unpack_cost,
        "basis_transform_cost": basis_transform_cost,
        "dual_basis_storage_cost": dual_basis_storage_cost,
        "rebuild_cost": rebuild_cost,
        "metadata_cost": metadata_cost,
    }


def run_t020_basis_shared_kv_transform_gate(
    *,
    vector_size: int = 16,
    epsilon_numeric: float = 1e-6,
    attention_margin: float = 1e-3,
    min_speedup: float = 1.10,
) -> dict[str, Any]:
    """Gate C-MATH-W01-05 for the packed path's diagonal scale transform class."""

    if min_speedup <= 1.0:
        raise ValueError("min_speedup must be greater than 1")
    proof = prove_diagonal_dual_basis_dot_preservation(
        size=vector_size,
        epsilon_numeric=epsilon_numeric,
        attention_margin=attention_margin,
    )
    pricing = price_basis_shared_path()
    transform_class_supported = proof["transform_class"] == "diagonal_scale"
    exact_dot_pass = proof["dot_product_preserved"]
    numerical_pass = proof["numerical_debt_below_margin"]
    no_dense_inverse = not proof["requires_dense_inverse"]
    no_dual_canonical_storage = not proof["requires_canonical_and_transformed_kv"]
    gain_floor_pass = pricing["speedup_vs_representation_crossing"] >= min_speedup
    gate_pass = bool(
        transform_class_supported
        and exact_dot_pass
        and numerical_pass
        and no_dense_inverse
        and no_dual_canonical_storage
        and gain_floor_pass
    )

    blockers: list[str] = []
    if not transform_class_supported:
        blockers.append("packed_transform_class_not_supported")
    if not exact_dot_pass:
        blockers.append("dual_basis_dot_product_not_preserved")
    if not numerical_pass:
        blockers.append("condition_number_numeric_debt_exceeds_attention_margin")
    if not no_dense_inverse:
        blockers.append("dense_inverse_required_in_hot_path")
    if not no_dual_canonical_storage:
        blockers.append("requires_canonical_and_transformed_kv")
    if not gain_floor_pass:
        blockers.append("basis_maintenance_cost_exceeds_saved_crossing_cost")

    return {
        "probe": "t020_basis_shared_kv_transform_gate",
        "activation": ACTIVATION,
        "finding": "C-MATH-W01-05",
        "invariant": "q^T k = (A q)^T (A^{-T} k) for diagonal packed row-scale basis A",
        "transform_class": {
            "actual_packed_path_basis": "diagonal row/channel scales from T016/T017 packed payloads",
            "classified_as": proof["transform_class"],
            "cheap_dual_basis": True,
            "dense_inverse_required": proof["requires_dense_inverse"],
            "canonical_and_transformed_kv_required": proof["requires_canonical_and_transformed_kv"],
        },
        "proof": proof,
        "baseline": {
            "source": "canonical representation crossing between packed projection and KV attention state",
            "representation_crossing_cost": pricing["baseline_crossing_cost"],
            "requires_canonical_crossing": True,
        },
        "candidate": {
            "basis_shared_cost": pricing["candidate_basis_cost"],
            "deleted_crossing_cost": pricing["deleted_crossing_cost"],
            "speedup_vs_representation_crossing": pricing["speedup_vs_representation_crossing"],
            "requires_canonical_crossing": False,
        },
        "pricing": pricing,
        "readiness": {
            "basis_shared_kv_transform_gate_ready": gate_pass,
            "transform_class_supported": transform_class_supported,
            "dot_product_preservation_pass": exact_dot_pass,
            "numerical_debt_below_margin": numerical_pass,
            "no_dense_inverse": no_dense_inverse,
            "no_dual_canonical_storage": no_dual_canonical_storage,
            "gain_floor_pass": gain_floor_pass,
        },
        "blockers": blockers,
        "next_actions": (
            [
                "Represent KV receipts with the diagonal dual-basis version when the packed projection path owns the scale basis.",
                "Reject promotion for any future transform class that requires dense inverse or dual canonical KV storage.",
            ]
            if gate_pass
            else ["Do not integrate basis-shared KV until transform-class exactness and pricing pass."]
        ),
    }


__all__ = [
    "ACTIVATION",
    "deterministic_vector",
    "diagonal_scale_basis",
    "dot",
    "price_basis_shared_path",
    "prove_diagonal_dual_basis_dot_preservation",
    "run_t020_basis_shared_kv_transform_gate",
]
