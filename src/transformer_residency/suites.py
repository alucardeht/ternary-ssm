"""Synthetic gate suites with positive and negative controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .attention import attention_simplex_drift, kv_trust_horizon
from .certificates import gate_cliff_certificate, margin_certificate
from .fragments import FragmentClause, compile_fragment_schedule
from .policies import (
    AdaptiveMode,
    patch_atom_certificate,
    residual_correlation_alarm,
    select_adaptive_mode,
    support_exit_certificate,
)
from .receipts import TileReceipt
from .residuals import residual_collapse_report


@dataclass(frozen=True)
class GateCheck:
    name: str
    expected_pass: bool
    observed_pass: bool
    metrics: dict[str, float | int | bool | str]

    @property
    def matched_expectation(self) -> bool:
        return self.expected_pass == self.observed_pass

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["matched_expectation"] = self.matched_expectation
        return data


@dataclass(frozen=True)
class GateSuiteReport:
    checks: tuple[GateCheck, ...]

    @property
    def matched(self) -> bool:
        return all(check.matched_expectation for check in self.checks)

    @property
    def unexpected(self) -> tuple[GateCheck, ...]:
        return tuple(check for check in self.checks if not check.matched_expectation)

    def as_dict(self) -> dict[str, object]:
        return {
            "matched": self.matched,
            "total_checks": len(self.checks),
            "unexpected_count": len(self.unexpected),
            "checks": [check.as_dict() for check in self.checks],
        }


def _orthonormal(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(rows, cols)))
    return q[:, :cols]


def _down_invisible_residual_case(
    rng: np.random.Generator,
    samples: int = 96,
    intermediate: int = 64,
    hidden: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    down_basis = _orthonormal(rng, intermediate, hidden)
    down_projection = down_basis / np.sqrt(hidden)
    noise = rng.normal(size=(samples, intermediate))
    invisible = noise - noise @ down_basis @ down_basis.T
    visible = 0.02 * rng.normal(size=(samples, hidden)) @ down_basis.T
    return invisible + visible, down_projection


def _down_visible_residual_case(
    rng: np.random.Generator,
    samples: int = 96,
    intermediate: int = 64,
    hidden: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    down_basis = _orthonormal(rng, intermediate, hidden)
    down_projection = down_basis / np.sqrt(hidden)
    visible = rng.normal(size=(samples, hidden)) @ down_basis.T
    return visible, down_projection


def _residual_check(
    name: str,
    expected_pass: bool,
    hidden_residual: np.ndarray,
    down_projection: np.ndarray,
) -> GateCheck:
    report = residual_collapse_report(hidden_residual, down_projection)
    observed = report.passes_energy_gate and report.passes_rank_gate
    return GateCheck(
        name=name,
        expected_pass=expected_pass,
        observed_pass=observed,
        metrics=report.as_dict(),
    )


def _receipt_check(name: str, expected_pass: bool, receipt: TileReceipt) -> GateCheck:
    return GateCheck(
        name=name,
        expected_pass=expected_pass,
        observed_pass=receipt.passes(0.10),
        metrics=receipt.as_dict(),
    )


def _margin_check(name: str, expected_pass: bool, logits: np.ndarray, error_radius: float) -> GateCheck:
    cert = margin_certificate(logits, error_radius=error_radius)
    return GateCheck(
        name=name,
        expected_pass=expected_pass,
        observed_pass=cert.accepted,
        metrics=cert.as_dict(),
    )


def _gate_cliff_check(
    name: str,
    expected_pass: bool,
    gate_values: np.ndarray,
    residual_bound: float,
) -> GateCheck:
    cert = gate_cliff_certificate(
        gate_values,
        residual_bound=residual_bound,
        margin=0.05,
        min_accepted_fraction=0.90,
    )
    return GateCheck(
        name=name,
        expected_pass=expected_pass,
        observed_pass=cert.accepted,
        metrics=cert.as_dict(),
    )


def _attention_check(
    name: str,
    expected_pass: bool,
    dense: np.ndarray,
    resident: np.ndarray,
) -> GateCheck:
    report = attention_simplex_drift(
        dense,
        resident,
        value_norm_bound=1.5,
        max_allowed_output_error=0.15,
        anchor_indices=(0,),
    )
    return GateCheck(name, expected_pass, report.passes, report.as_dict())


def _support_check(name: str, expected_pass: bool, x: np.ndarray) -> GateCheck:
    report = support_exit_certificate(
        x,
        np.zeros_like(x),
        np.ones_like(x),
        residual_slope=0.05,
        budget=0.25,
    )
    return GateCheck(name, expected_pass, report.passes, report.as_dict())


def _selector_check(name: str, expected_pass: bool, modes: tuple[AdaptiveMode, ...]) -> GateCheck:
    report = select_adaptive_mode(modes, budget=0.10)
    return GateCheck(name, expected_pass, report.passes, report.as_dict())


def _patch_check(
    name: str,
    expected_pass: bool,
    *,
    before: float,
    after: float,
    bytes_cost: int,
    support_matched: bool,
) -> GateCheck:
    report = patch_atom_certificate(
        before_error=before,
        after_error=after,
        bytes_cost=bytes_cost,
        max_bytes_cost=256,
        support_matched=support_matched,
    )
    return GateCheck(name, expected_pass, report.passes, report.as_dict())


def _correlation_check(name: str, expected_pass: bool, a: np.ndarray, b: np.ndarray) -> GateCheck:
    report = residual_correlation_alarm(
        a,
        b,
        min_safe_correlation=-0.20,
        budget=2.0,
    )
    return GateCheck(name, expected_pass, report.passes, report.as_dict())


def _kv_check(
    name: str,
    expected_pass: bool,
    *,
    anchor_mass: float,
    update_frequency: float,
) -> GateCheck:
    report = kv_trust_horizon(
        anchor_mass=anchor_mass,
        tail_norm_bound=1.0,
        leakage_bound=0.30,
        update_frequency=update_frequency,
        max_update_frequency=0.20,
        max_allowed_output_error=0.08,
    )
    return GateCheck(name, expected_pass, report.passes, report.as_dict())


def _fragment_check(name: str, expected_pass: bool, clauses: tuple[FragmentClause, ...]) -> GateCheck:
    report = compile_fragment_schedule(clauses)
    return GateCheck(name, expected_pass, report.passes, report.as_dict())


def run_default_gate_suite(seed: int = 1001) -> GateSuiteReport:
    """Run pass/fail synthetic controls for the initial residency gates."""

    rng = np.random.default_rng(seed)
    invisible_residual, invisible_down = _down_invisible_residual_case(rng)
    visible_residual, visible_down = _down_visible_residual_case(rng)

    checks = [
        _residual_check(
            "residual_collapse_positive_control",
            True,
            invisible_residual,
            invisible_down,
        ),
        _residual_check(
            "residual_collapse_negative_control",
            False,
            visible_residual,
            visible_down,
        ),
        _receipt_check(
            "tile_receipt_positive_control",
            True,
            TileReceipt(
                dense_elements=128 * 128,
                payload_bits_per_element=2.5,
                metadata_bytes=256,
                scale_bytes=128,
                correction_bytes=512,
                scratch_bytes=128,
            ),
        ),
        _receipt_check(
            "tile_receipt_negative_control",
            False,
            TileReceipt(
                dense_elements=128 * 128,
                payload_bits_per_element=2.5,
                metadata_bytes=4096,
                scale_bytes=2048,
                correction_bytes=16384,
                scratch_bytes=4096,
            ),
        ),
        _margin_check(
            "logit_margin_positive_control",
            True,
            np.array([1.0, 0.55, 0.1, -0.2], dtype=np.float64),
            0.08,
        ),
        _margin_check(
            "logit_margin_negative_control",
            False,
            np.array([1.0, 0.93, 0.1, -0.2], dtype=np.float64),
            0.08,
        ),
        _gate_cliff_check(
            "mlp_gate_cliff_positive_control",
            True,
            np.full((16, 16), 0.70, dtype=np.float64),
            0.20,
        ),
        _gate_cliff_check(
            "mlp_gate_cliff_negative_control",
            False,
            np.array([[-0.01, 0.02, 0.30, -0.04]], dtype=np.float64),
            0.20,
        ),
        _attention_check(
            "attention_simplex_positive_control",
            True,
            np.array([0.82, 0.10, 0.05, 0.03]),
            np.array([0.78, 0.12, 0.06, 0.04]),
        ),
        _attention_check(
            "attention_simplex_negative_control",
            False,
            np.array([0.50, 0.49, 0.01, 0.00]),
            np.array([0.05, 0.90, 0.05, 0.00]),
        ),
        _support_check("support_exit_positive_control", True, np.array([1.0, 1.0])),
        _support_check("support_exit_negative_control", False, np.array([5.0, 5.0])),
        _selector_check(
            "adaptive_selector_positive_control",
            True,
            (
                AdaptiveMode("cheap", cost=1.0, predicted_error=0.06, metadata_bytes=64),
                AdaptiveMode("repair", cost=2.0, predicted_error=0.02, metadata_bytes=128),
            ),
        ),
        _selector_check(
            "adaptive_selector_negative_control",
            False,
            (
                AdaptiveMode("cheap", cost=1.0, predicted_error=0.30, metadata_bytes=64),
                AdaptiveMode("repair", cost=2.0, predicted_error=0.20, metadata_bytes=128),
            ),
        ),
        _patch_check(
            "patch_atom_positive_control",
            True,
            before=0.30,
            after=0.10,
            bytes_cost=128,
            support_matched=True,
        ),
        _patch_check(
            "patch_atom_negative_control",
            False,
            before=0.30,
            after=0.32,
            bytes_cost=128,
            support_matched=True,
        ),
        _correlation_check(
            "residual_correlation_positive_control",
            True,
            np.array([1.0, 0.0]),
            np.array([-0.8, 0.0]),
        ),
        _correlation_check(
            "residual_correlation_negative_control",
            False,
            np.array([1.0, 0.0]),
            np.array([0.8, 0.0]),
        ),
        _kv_check(
            "kv_trust_horizon_positive_control",
            True,
            anchor_mass=0.85,
            update_frequency=0.10,
        ),
        _kv_check(
            "kv_trust_horizon_negative_control",
            False,
            anchor_mass=0.40,
            update_frequency=0.35,
        ),
        _fragment_check(
            "fragment_clause_positive_control",
            True,
            (
                FragmentClause("base_mma", 4096, 256, 0, 128, 24, 64),
                FragmentClause("residual_companion", 2048, 128, 512, 128, 40, 64),
            ),
        ),
        _fragment_check(
            "fragment_clause_negative_control",
            False,
            (
                FragmentClause("base_mma", 4096, 256, 0, 128, 24, 64),
                FragmentClause("dense_reconstruct", 1024, 256, 4096, 512, 96, 64, requires_dense_reconstruction=True),
            ),
        ),
    ]
    return GateSuiteReport(tuple(checks))
