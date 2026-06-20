"""T003 RAM-efficiency gates for the transformer residency line."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .footprint import BYTES_PER_GIB, TransformerShape, qwen3_32b_shape
from .kv_margin import KVMarginScenario, run_t002_kv_margin_sweep


@dataclass(frozen=True)
class RepairByteLedger:
    """First-class byte accounting for weight repair and metadata costs."""

    gross_saved_gib: float
    repair_atom_gib: float
    metadata_gib: float
    scratch_gib: float
    min_net_saved_gib: float = 0.25
    max_repair_fraction_of_gross_saved: float = 0.50

    @property
    def charged_gib(self) -> float:
        return self.repair_atom_gib + self.metadata_gib + self.scratch_gib

    @property
    def net_saved_gib(self) -> float:
        return self.gross_saved_gib - self.charged_gib

    @property
    def repair_fraction_of_gross_saved(self) -> float:
        if self.gross_saved_gib <= 0.0:
            return float("inf")
        return self.charged_gib / self.gross_saved_gib

    @property
    def passes(self) -> bool:
        return (
            self.net_saved_gib >= self.min_net_saved_gib
            and self.repair_fraction_of_gross_saved <= self.max_repair_fraction_of_gross_saved
        )

    def as_dict(self) -> dict[str, float | bool]:
        data = asdict(self)
        data.update(
            {
                "charged_gib": self.charged_gib,
                "net_saved_gib": self.net_saved_gib,
                "repair_fraction_of_gross_saved": self.repair_fraction_of_gross_saved,
                "passes": self.passes,
            }
        )
        return data


@dataclass(frozen=True)
class KiviPlusKVPolicy:
    """KIVI-style old-cache quantization with an exact newest-token window."""

    context_tokens: int
    exact_recent_tokens: int
    old_key_bits: float
    old_value_bits: float
    metadata_overhead_gib: float
    key_logit_error: float
    value_output_error: float
    topk_margin: float
    safety_margin: float = 0.08
    max_kv_gib: float = 0.90

    @property
    def old_tokens(self) -> int:
        return self.context_tokens - self.exact_recent_tokens

    @property
    def residual_error_bound(self) -> float:
        return self.key_logit_error + self.value_output_error

    @property
    def residual_budget(self) -> float:
        return self.topk_margin - self.safety_margin

    @property
    def residual_passes(self) -> bool:
        return self.old_tokens >= 0 and self.residual_error_bound <= self.residual_budget

    def kv_gib(self, shape: TransformerShape | None = None) -> float:
        if self.context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        if not 0 <= self.exact_recent_tokens <= self.context_tokens:
            raise ValueError("exact_recent_tokens must be in [0, context_tokens]")
        if self.old_key_bits <= 0.0 or self.old_value_bits <= 0.0:
            raise ValueError("old key/value bits must be positive")

        current_shape = shape or qwen3_32b_shape()
        fp16_bytes_per_token = current_shape.layers * 2 * current_shape.kv_heads * current_shape.head_dim * 2
        key_or_value_fp16_bytes = fp16_bytes_per_token / 2.0
        exact_bytes = self.exact_recent_tokens * fp16_bytes_per_token
        old_bytes = self.old_tokens * key_or_value_fp16_bytes * (
            (self.old_key_bits / 16.0) + (self.old_value_bits / 16.0)
        )
        return (exact_bytes + old_bytes) / BYTES_PER_GIB + self.metadata_overhead_gib

    def as_dict(self, shape: TransformerShape | None = None) -> dict[str, float | int | bool]:
        kv_gib = self.kv_gib(shape)
        data = asdict(self)
        data.update(
            {
                "old_tokens": self.old_tokens,
                "kv_gib": kv_gib,
                "residual_error_bound": self.residual_error_bound,
                "residual_budget": self.residual_budget,
                "residual_passes": self.residual_passes,
                "fits_max_kv_gib": kv_gib <= self.max_kv_gib,
                "passes": self.residual_passes and kv_gib <= self.max_kv_gib,
            }
        )
        return data


@dataclass(frozen=True)
class ResidencyReceipt:
    """Stack-level residency receipt with a deterministic audit hash."""

    weight_runtime_gib: float
    kv_gib: float
    repair_charged_gib: float
    page_table_gib: float
    scratch_gib: float
    safety_gib: float
    denominator_gib: float = 12.0

    @property
    def total_without_safety_gib(self) -> float:
        return self.weight_runtime_gib + self.kv_gib + self.repair_charged_gib + self.page_table_gib + self.scratch_gib

    @property
    def total_with_safety_gib(self) -> float:
        return self.total_without_safety_gib + self.safety_gib

    @property
    def slack_without_safety_gib(self) -> float:
        return self.denominator_gib - self.total_without_safety_gib

    @property
    def slack_with_safety_gib(self) -> float:
        return self.denominator_gib - self.total_with_safety_gib

    @property
    def fits_with_safety(self) -> bool:
        return self.total_with_safety_gib <= self.denominator_gib

    @property
    def audit_hash(self) -> str:
        payload = {
            "denominator_gib": round(self.denominator_gib, 6),
            "kv_gib": round(self.kv_gib, 6),
            "page_table_gib": round(self.page_table_gib, 6),
            "repair_charged_gib": round(self.repair_charged_gib, 6),
            "safety_gib": round(self.safety_gib, 6),
            "scratch_gib": round(self.scratch_gib, 6),
            "weight_runtime_gib": round(self.weight_runtime_gib, 6),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def as_dict(self) -> dict[str, float | bool | str]:
        data = asdict(self)
        data.update(
            {
                "total_without_safety_gib": self.total_without_safety_gib,
                "total_with_safety_gib": self.total_with_safety_gib,
                "slack_without_safety_gib": self.slack_without_safety_gib,
                "slack_with_safety_gib": self.slack_with_safety_gib,
                "fits_with_safety": self.fits_with_safety,
                "audit_hash": self.audit_hash,
            }
        )
        return data


@dataclass(frozen=True)
class ByteTokSPredictor:
    """Memory-bandwidth tok/s proxy from emitted byte receipts."""

    bandwidth_gib_per_second: float
    weight_runtime_gib: float
    kv_read_gib_per_token: float
    streams_per_token: float = 1.0
    verifier_multiplier: float = 1.0

    @property
    def gib_per_token(self) -> float:
        return self.weight_runtime_gib * self.streams_per_token + self.kv_read_gib_per_token

    @property
    def tokens_per_second(self) -> float:
        if self.bandwidth_gib_per_second <= 0.0:
            raise ValueError("bandwidth_gib_per_second must be positive")
        if self.gib_per_token <= 0.0:
            raise ValueError("gib_per_token must be positive")
        if self.verifier_multiplier <= 0.0:
            raise ValueError("verifier_multiplier must be positive")
        return (self.bandwidth_gib_per_second / self.gib_per_token) * self.verifier_multiplier

    def as_dict(self) -> dict[str, float]:
        data = asdict(self)
        data.update(
            {
                "gib_per_token": self.gib_per_token,
                "tokens_per_second": self.tokens_per_second,
            }
        )
        return data


def _scenario_by_name(result: dict[str, object], name: str) -> dict[str, object]:
    scenarios = result["scenarios"]
    if not isinstance(scenarios, list):
        raise TypeError("scenarios must be a list")
    for scenario in scenarios:
        if isinstance(scenario, dict) and scenario.get("name") == name:
            return scenario
    raise KeyError(name)


def _kv_read_gib_per_token(policy: KiviPlusKVPolicy, shape: TransformerShape) -> float:
    return policy.kv_gib(shape) / policy.context_tokens


def run_t003_receipt_kv_gate(
    *,
    bandwidth_gib_per_second: float = 514.568,
    shape: TransformerShape | None = None,
) -> dict[str, object]:
    """Run the first T003 receipt, repair-byte, KV precision, and tok/s gate."""

    current_shape = shape or qwen3_32b_shape()
    t002 = run_t002_kv_margin_sweep(bandwidth_gib_per_second=bandwidth_gib_per_second, shape=current_shape)
    baseline = _scenario_by_name(t002, "strong_weight_tighten_tiered_int4_kv_16k")

    repair_ledger = RepairByteLedger(
        gross_saved_gib=0.74,
        repair_atom_gib=0.17,
        metadata_gib=0.07,
        scratch_gib=0.03,
    )
    if not repair_ledger.passes:
        raise AssertionError("default T003 repair ledger should pass")

    kv_policy = KiviPlusKVPolicy(
        context_tokens=16_384,
        exact_recent_tokens=1_024,
        old_key_bits=2.0,
        old_value_bits=3.0,
        metadata_overhead_gib=0.03,
        key_logit_error=0.07,
        value_output_error=0.05,
        topk_margin=0.24,
        max_kv_gib=0.90,
    )
    kv_data = kv_policy.as_dict(current_shape)
    if not bool(kv_data["passes"]):
        raise AssertionError("default T003 KV policy should pass")

    baseline_weight_gib = float(baseline["weight_gib"])
    t003_weight_runtime_gib = baseline_weight_gib - repair_ledger.net_saved_gib
    receipt = ResidencyReceipt(
        weight_runtime_gib=t003_weight_runtime_gib,
        kv_gib=float(kv_data["kv_gib"]),
        repair_charged_gib=0.0,
        page_table_gib=0.04,
        scratch_gib=0.05,
        safety_gib=0.25,
    )

    baseline_predictor = ByteTokSPredictor(
        bandwidth_gib_per_second=bandwidth_gib_per_second,
        weight_runtime_gib=float(baseline["weight_gib"]),
        kv_read_gib_per_token=float(baseline["kv_gib"]) / int(baseline["context_tokens"]),
    )
    t003_predictor = ByteTokSPredictor(
        bandwidth_gib_per_second=bandwidth_gib_per_second,
        weight_runtime_gib=t003_weight_runtime_gib,
        kv_read_gib_per_token=_kv_read_gib_per_token(kv_policy, current_shape),
        verifier_multiplier=1.18,
    )

    baseline_total = float(baseline["total_gib"])
    baseline_slack = float(baseline["slack_gib"])
    baseline_tok_s = baseline_predictor.tokens_per_second
    t003_tok_s = t003_predictor.tokens_per_second
    validations = {
        "activation": "T003_RECEIPT_KV_GATE_ACTIVE",
        "repair_ledger_passes": repair_ledger.passes,
        "kv_policy_passes": bool(kv_data["passes"]),
        "receipt_hash_present": len(receipt.audit_hash) == 16,
        "receipt_fits_with_safety": receipt.fits_with_safety,
        "slack_improves_by_at_least_0_50_gib": receipt.slack_without_safety_gib - baseline_slack >= 0.50,
        "kv_gib_below_t002_int4_kv": float(kv_data["kv_gib"]) < float(baseline["kv_gib"]),
        "tok_s_proxy_improves_by_at_least_10pct": t003_tok_s / baseline_tok_s >= 1.10,
    }

    t003_scenario = KVMarginScenario(
        name="t003_receipt_kivi_exact_recent_16k",
        context_tokens=int(baseline["context_tokens"]),
        weight_gib=t003_weight_runtime_gib,
        kv_gib=float(kv_data["kv_gib"]),
        total_gib=receipt.total_without_safety_gib,
        slack_gib=receipt.slack_without_safety_gib,
        synthetic_tokens_per_second=t003_tok_s,
        speedup_vs_baseline=t003_tok_s / baseline_tok_s,
        vram_saved_vs_baseline_gib=baseline_total - receipt.total_without_safety_gib,
        notes="A4-20/A1-02/A2-01/A6-20 receipt-and-KV-precision gate.",
    )

    return {
        "probe": "t003_receipt_kv_gate",
        "activation": "T003_RECEIPT_KV_GATE_ACTIVE",
        "shape": current_shape.as_dict(),
        "baseline": {
            "name": baseline["name"],
            "context_tokens": baseline["context_tokens"],
            "total_gib": baseline_total,
            "slack_gib": baseline_slack,
            "kv_gib": baseline["kv_gib"],
            "tok_s_proxy": baseline_tok_s,
        },
        "repair_byte_ledger": repair_ledger.as_dict(),
        "kv_policy": kv_data,
        "residency_receipt": receipt.as_dict(),
        "tok_s_predictor": {
            "baseline": baseline_predictor.as_dict(),
            "t003": t003_predictor.as_dict(),
            "speedup_vs_baseline": t003_tok_s / baseline_tok_s,
        },
        "scenario": t003_scenario.as_dict(),
        "validations": validations,
        "overall": {
            "pass": all(bool(value) for value in validations.values() if isinstance(value, bool)),
            "baseline_total_gib": baseline_total,
            "t003_total_gib": receipt.total_without_safety_gib,
            "baseline_slack_gib": baseline_slack,
            "t003_slack_gib": receipt.slack_without_safety_gib,
            "t003_slack_with_safety_gib": receipt.slack_with_safety_gib,
            "vram_saved_vs_baseline_gib": baseline_total - receipt.total_without_safety_gib,
            "baseline_tok_s_proxy": baseline_tok_s,
            "t003_tok_s_proxy": t003_tok_s,
            "tok_s_speedup_vs_baseline": t003_tok_s / baseline_tok_s,
        },
    }
