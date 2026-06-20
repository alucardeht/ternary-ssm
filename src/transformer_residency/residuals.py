"""Residual-collapse probes for down-visible transformer residency."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualCollapseReport:
    """Rank and energy summary for hidden residuals before and after down projection."""

    hidden_energy: float
    visible_energy: float
    energy_ratio: float
    hidden_effective_rank: float
    visible_effective_rank: float
    rank_ratio: float
    passes_energy_gate: bool
    passes_rank_gate: bool

    def as_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def _covariance_effective_rank(x: np.ndarray, eps: float = 1e-12) -> float:
    centered = x - x.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    total = float(eigvals.sum())
    if total <= eps:
        return 0.0
    probs = eigvals / total
    entropy = -float(np.sum(probs * np.log(probs + eps)))
    return float(np.exp(entropy))


def residual_collapse_report(
    hidden_residual: np.ndarray,
    down_projection: np.ndarray,
    *,
    max_visible_energy_ratio: float = 0.35,
    max_visible_rank_ratio: float = 0.50,
) -> ResidualCollapseReport:
    """Measure whether hidden residual becomes smaller after a down projection.

    `hidden_residual` has shape `[samples, intermediate]`.
    `down_projection` has shape `[intermediate, hidden]`.
    """

    if hidden_residual.ndim != 2:
        raise ValueError("hidden_residual must be a 2D array")
    if down_projection.ndim != 2:
        raise ValueError("down_projection must be a 2D array")
    if hidden_residual.shape[1] != down_projection.shape[0]:
        raise ValueError("hidden_residual width must match down_projection rows")

    visible_residual = hidden_residual @ down_projection
    hidden_energy = float(np.mean(np.sum(hidden_residual * hidden_residual, axis=1)))
    visible_energy = float(np.mean(np.sum(visible_residual * visible_residual, axis=1)))
    energy_ratio = visible_energy / hidden_energy if hidden_energy > 0.0 else float("inf")

    hidden_rank = _covariance_effective_rank(hidden_residual)
    visible_rank = _covariance_effective_rank(visible_residual)
    rank_ratio = visible_rank / hidden_rank if hidden_rank > 0.0 else float("inf")

    return ResidualCollapseReport(
        hidden_energy=hidden_energy,
        visible_energy=visible_energy,
        energy_ratio=energy_ratio,
        hidden_effective_rank=hidden_rank,
        visible_effective_rank=visible_rank,
        rank_ratio=rank_ratio,
        passes_energy_gate=energy_ratio <= max_visible_energy_ratio,
        passes_rank_gate=rank_ratio <= max_visible_rank_ratio,
    )
