"""Attention and KV certificates for transformer residency probes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class AttentionSimplexReport:
    total_variation: float
    value_norm_bound: float
    output_error_bound: float
    max_allowed_output_error: float
    anchor_mass: float
    passes: bool

    def as_dict(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class KVTrustHorizonReport:
    anchor_mass: float
    tail_norm_bound: float
    leakage_bound: float
    update_frequency: float
    max_update_frequency: float
    output_error_bound: float
    max_allowed_output_error: float
    passes: bool

    def as_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def _normalize_probabilities(x: np.ndarray) -> np.ndarray:
    probs = np.asarray(x, dtype=np.float64)
    total = probs.sum()
    if total <= 0:
        raise ValueError("probabilities must have positive mass")
    return probs / total


def attention_simplex_drift(
    dense_attention: np.ndarray,
    resident_attention: np.ndarray,
    *,
    value_norm_bound: float,
    max_allowed_output_error: float,
    anchor_indices: tuple[int, ...] = (),
) -> AttentionSimplexReport:
    dense = _normalize_probabilities(dense_attention)
    resident = _normalize_probabilities(resident_attention)
    if dense.shape != resident.shape:
        raise ValueError("attention distributions must have the same shape")
    tv = 0.5 * float(np.sum(np.abs(dense - resident)))
    output_bound = tv * value_norm_bound
    anchor_mass = float(dense[list(anchor_indices)].sum()) if anchor_indices else float(dense.max())
    return AttentionSimplexReport(
        total_variation=tv,
        value_norm_bound=value_norm_bound,
        output_error_bound=output_bound,
        max_allowed_output_error=max_allowed_output_error,
        anchor_mass=anchor_mass,
        passes=output_bound <= max_allowed_output_error,
    )


def kv_trust_horizon(
    *,
    anchor_mass: float,
    tail_norm_bound: float,
    leakage_bound: float,
    update_frequency: float,
    max_update_frequency: float,
    max_allowed_output_error: float,
) -> KVTrustHorizonReport:
    if not 0.0 <= anchor_mass <= 1.0:
        raise ValueError("anchor_mass must be in [0, 1]")
    output_error_bound = tail_norm_bound * leakage_bound * (1.0 - anchor_mass)
    return KVTrustHorizonReport(
        anchor_mass=anchor_mass,
        tail_norm_bound=tail_norm_bound,
        leakage_bound=leakage_bound,
        update_frequency=update_frequency,
        max_update_frequency=max_update_frequency,
        output_error_bound=output_error_bound,
        max_allowed_output_error=max_allowed_output_error,
        passes=output_error_bound <= max_allowed_output_error
        and update_frequency <= max_update_frequency,
    )
