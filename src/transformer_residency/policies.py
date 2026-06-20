"""Policy and repair certificates for transformer residency probes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SupportEnvelopeReport:
    distance: float
    predicted_residual: float
    budget: float
    passes: bool

    def as_dict(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptiveMode:
    name: str
    cost: float
    predicted_error: float
    metadata_bytes: int


@dataclass(frozen=True)
class AdaptiveSelectorReport:
    selected_mode: str
    selected_cost: float
    selected_error: float
    selected_metadata_bytes: int
    budget: float
    passes: bool

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return asdict(self)


@dataclass(frozen=True)
class PatchAtomReport:
    before_error: float
    after_error: float
    bytes_cost: int
    max_bytes_cost: int
    support_matched: bool
    passes: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class ResidualCorrelationReport:
    correlation: float
    min_safe_correlation: float
    before_budget: float
    after_budget: float
    passes: bool

    def as_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def support_exit_certificate(
    x: np.ndarray,
    center: np.ndarray,
    shape_scale: np.ndarray,
    *,
    residual_slope: float,
    budget: float,
) -> SupportEnvelopeReport:
    if x.shape != center.shape or x.shape != shape_scale.shape:
        raise ValueError("x, center, and shape_scale must have the same shape")
    scaled = (x - center) * shape_scale
    distance = float(np.linalg.norm(scaled))
    predicted = residual_slope * distance
    return SupportEnvelopeReport(
        distance=distance,
        predicted_residual=predicted,
        budget=budget,
        passes=predicted <= budget,
    )


def select_adaptive_mode(modes: list[AdaptiveMode] | tuple[AdaptiveMode, ...], *, budget: float) -> AdaptiveSelectorReport:
    feasible = [mode for mode in modes if mode.predicted_error <= budget]
    if not feasible:
        cheapest = min(modes, key=lambda mode: mode.cost)
        return AdaptiveSelectorReport(
            selected_mode=cheapest.name,
            selected_cost=cheapest.cost,
            selected_error=cheapest.predicted_error,
            selected_metadata_bytes=cheapest.metadata_bytes,
            budget=budget,
            passes=False,
        )
    selected = min(feasible, key=lambda mode: mode.cost)
    return AdaptiveSelectorReport(
        selected_mode=selected.name,
        selected_cost=selected.cost,
        selected_error=selected.predicted_error,
        selected_metadata_bytes=selected.metadata_bytes,
        budget=budget,
        passes=True,
    )


def patch_atom_certificate(
    *,
    before_error: float,
    after_error: float,
    bytes_cost: int,
    max_bytes_cost: int,
    support_matched: bool,
) -> PatchAtomReport:
    return PatchAtomReport(
        before_error=before_error,
        after_error=after_error,
        bytes_cost=bytes_cost,
        max_bytes_cost=max_bytes_cost,
        support_matched=support_matched,
        passes=support_matched and after_error < before_error and bytes_cost <= max_bytes_cost,
    )


def residual_correlation_alarm(
    residual_a: np.ndarray,
    residual_b: np.ndarray,
    *,
    min_safe_correlation: float,
    budget: float,
) -> ResidualCorrelationReport:
    a = np.asarray(residual_a, dtype=np.float64).ravel()
    b = np.asarray(residual_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("residual vectors must have the same shape")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    correlation = float(np.dot(a, b) / denom) if denom > 0 else 0.0
    before_budget = float(np.linalg.norm(a) + np.linalg.norm(b))
    after_budget = float(np.linalg.norm(a + b))
    return ResidualCorrelationReport(
        correlation=correlation,
        min_safe_correlation=min_safe_correlation,
        before_budget=before_budget,
        after_budget=after_budget,
        passes=correlation <= min_safe_correlation and after_budget <= budget,
    )
