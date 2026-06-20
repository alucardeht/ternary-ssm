"""Causal intervention receipts and denominator accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class InterventionProof:
    component: str
    intervention: str
    delta: float
    overlap_class: str
    accepted: bool

    def as_dict(self) -> dict[str, bool | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class DenominatorReceipt:
    work_removed: float = 0.0
    progress_gain: float = 0.0
    build_cost: float = 0.0
    score_cost: float = 0.0
    verification_cost: float = 0.0
    memory_cost: float = 0.0
    repair_cost: float = 0.0
    quality_risk: float = 0.0
    overlap_class: str = "O0"

    @property
    def net(self) -> float:
        costs = (
            self.build_cost
            + self.score_cost
            + self.verification_cost
            + self.memory_cost
            + self.repair_cost
            + self.quality_risk
        )
        return float(self.work_removed + self.progress_gain - costs)

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


def no_bypass_delta(
    live_score: float,
    ablated_score: float,
    *,
    component: str,
    intervention: str,
    overlap_class: str,
    min_delta: float,
) -> InterventionProof:
    """Build the minimal causal proof object for state-use validation."""

    delta = float(live_score - ablated_score)
    return InterventionProof(
        component=component,
        intervention=intervention,
        delta=delta,
        overlap_class=overlap_class,
        accepted=delta > min_delta,
    )

