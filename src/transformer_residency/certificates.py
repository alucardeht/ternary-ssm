"""Local quality certificates for resident transformer approximations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MarginCertificate:
    accepted: bool
    winner: int
    runner_up: int
    margin: float
    error_radius: float
    safety_ratio: float

    def as_dict(self) -> dict[str, bool | float | int]:
        return asdict(self)


@dataclass(frozen=True)
class GateCliffCertificate:
    accepted_fraction: float
    rejected_fraction: float
    min_slack: float
    mean_slack: float
    accepted: bool

    def as_dict(self) -> dict[str, bool | float]:
        return asdict(self)


def margin_certificate(
    logits: np.ndarray,
    error_radius: float,
    *,
    safety_factor: float = 1.0,
) -> MarginCertificate:
    """Certify that bounded logit error cannot flip the top-1 decision."""

    if logits.ndim != 1:
        raise ValueError("logits must be a 1D array")
    if logits.shape[0] < 2:
        raise ValueError("at least two logits are required")
    if error_radius < 0:
        raise ValueError("error_radius must be non-negative")

    top_two = np.argpartition(logits, -2)[-2:]
    ordered = top_two[np.argsort(logits[top_two])[::-1]]
    winner = int(ordered[0])
    runner_up = int(ordered[1])
    margin = float(logits[winner] - logits[runner_up])
    required = float(2.0 * error_radius * safety_factor)

    return MarginCertificate(
        accepted=margin > required,
        winner=winner,
        runner_up=runner_up,
        margin=margin,
        error_radius=float(error_radius),
        safety_ratio=margin / required if required > 0 else float("inf"),
    )


def gate_cliff_certificate(
    gate_preactivation: np.ndarray,
    residual_bound: float | np.ndarray,
    *,
    margin: float = 0.0,
    min_accepted_fraction: float = 0.95,
) -> GateCliffCertificate:
    """Certify that MLP gate perturbation cannot cross the gate cliff."""

    if residual_bound is None:
        raise ValueError("residual_bound is required")
    if np.any(np.asarray(residual_bound) < 0):
        raise ValueError("residual_bound must be non-negative")
    if margin < 0:
        raise ValueError("margin must be non-negative")

    slack = np.abs(gate_preactivation) - np.asarray(residual_bound) - margin
    accepted_mask = slack > 0.0
    accepted_fraction = float(np.mean(accepted_mask)) if slack.size else 0.0

    return GateCliffCertificate(
        accepted_fraction=accepted_fraction,
        rejected_fraction=1.0 - accepted_fraction,
        min_slack=float(np.min(slack)) if slack.size else 0.0,
        mean_slack=float(np.mean(slack)) if slack.size else 0.0,
        accepted=accepted_fraction >= min_accepted_fraction,
    )
