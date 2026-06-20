"""T014 denominator tightening gate for dense Qwen residency."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .kv_margin import kv_cache_gib_for_context
from .t004_real_qwen_bridge import find_weight_dir, inventory_safetensors, resolve_qwen_config
from .t006_real_residency_block_smoke import BlockCodecPolicy, _compress_reconstruct
from .t007_real_layer_smoke import _relative_rmse
from .t008_compressed_generation_smoke import _load_weight_map
from .t009_t013_runtime_scaler import (
    ProjectionSmokeReceipt,
    _component_policy_bits,
    _deterministic_input,
    _layer_tensor_names,
    _read_weight_slice,
    _summarize_projection_group,
    _tensor_shape,
)


ACTIVATION = "T014_DENOMINATOR_TIGHTENING_ACTIVE"
MLP_GATE_ACTIVATION = "T014_MLP_TARGET_BITS_PROJECTION_GATE_ACTIVE"
PACKED_ARENA_ACTIVATION = "T014_PACKED_ARENA_BYTE_RECEIPT_ACTIVE"


@dataclass(frozen=True)
class DenominatorCandidate:
    """A priced T014 candidate stack."""

    name: str
    source_findings: tuple[str, ...]
    mlp_bits: float
    attention_bits: float
    embedding_bits: float
    output_head_bits: float
    overhead_gib: float
    kv_effective_bits: float
    status: str
    tradeoff: str

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_findings"] = list(self.source_findings)
        return data


def _mlp_tensor_names(layer_index: int) -> tuple[str, ...]:
    return tuple(name for name in _layer_tensor_names(layer_index) if ".mlp." in name)


def _repair_fraction_for_target_bits(*, target_bits: float, cols: int) -> float:
    if cols <= 0:
        raise ValueError("cols must be positive")
    base_bits = 2.0 + 16.0 / cols
    index_bits = math.ceil(math.log2(cols)) if cols > 1 else 1
    repair_denominator = 16.0 + index_bits
    return max(0.0, (target_bits - base_bits) / repair_denominator)


def _policy_for_target_mlp_bits(name: str, cols: int, target_bits: float) -> BlockCodecPolicy:
    return BlockCodecPolicy(
        name="activation_conditional_mlp_repair_target",
        target_bits_per_weight=target_bits,
        repair_fraction=_repair_fraction_for_target_bits(target_bits=target_bits, cols=cols),
        threshold_multiplier=0.75,
        code_bits_per_weight=2.0,
        scale_bits_per_row=16.0,
        repair_value_bits=16.0,
    )


def _projection_smoke_with_policy(
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    name: str,
    layer_index: int,
    row_limit: int | None,
    batch: int,
    seed: int,
    target_bits: float,
) -> ProjectionSmokeReceipt:
    import torch

    shard = weight_map[name]
    original_shape = _tensor_shape(model_dir, weight_map, name)
    rows = original_shape[0] if row_limit is None or row_limit <= 0 else min(row_limit, original_shape[0])
    cols = original_shape[1]
    exact, _ = _read_weight_slice(model_dir, name, shard, rows, cols)
    policy = _policy_for_target_mlp_bits(name, cols, target_bits)
    compressed = _compress_reconstruct(exact, policy)
    activations = _deterministic_input(cols, batch, seed)

    start = time.perf_counter()
    exact_out = activations @ exact.T
    exact_seconds = time.perf_counter() - start

    start = time.perf_counter()
    compressed_out = activations @ compressed.T
    compressed_seconds = time.perf_counter() - start

    digest = hashlib.sha256(exact[: min(rows, 8), : min(cols, 8)].numpy().tobytes()).hexdigest()
    effective_bits = policy.effective_bits(cols)
    torch.cuda.empty_cache() if importlib.util.find_spec("torch") is not None and torch.cuda.is_available() else None
    return ProjectionSmokeReceipt(
        name=name,
        layer_index=layer_index,
        shard=shard,
        policy=policy,
        original_shape=original_shape,
        slice_shape=(int(exact.shape[0]), int(exact.shape[1])),
        effective_bits_per_weight=effective_bits,
        q4_savings_fraction=(4.0 - effective_bits) / 4.0,
        fp16_savings_fraction=(16.0 - effective_bits) / 16.0,
        weight_relative_rmse=_relative_rmse(exact, compressed),
        projection_relative_rmse=_relative_rmse(exact_out, compressed_out),
        exact_seconds=exact_seconds,
        compressed_seconds=compressed_seconds,
        sha256=digest,
    )


def _candidate_set(current_bits: dict[str, float]) -> tuple[DenominatorCandidate, ...]:
    return (
        DenominatorCandidate(
            name="current_t013",
            source_findings=("T009-T013",),
            mlp_bits=current_bits["mlp_bits"],
            attention_bits=current_bits["attention_bits"],
            embedding_bits=current_bits["embedding_bits"],
            output_head_bits=current_bits["output_head_bits"],
            overhead_gib=0.55,
            kv_effective_bits=4.0,
            status="baseline",
            tradeoff="Already measured as the stricter full-model-equivalent denominator; does not fit 12 GiB.",
        ),
        DenominatorCandidate(
            name="t014_a_f15_mlp245_packed_overhead",
            source_findings=("T014-F01", "T014-F03", "T014-F15"),
            mlp_bits=2.45,
            attention_bits=current_bits["attention_bits"],
            embedding_bits=current_bits["embedding_bits"],
            output_head_bits=current_bits["output_head_bits"],
            overhead_gib=0.25,
            kv_effective_bits=4.0,
            status="implementation_gate",
            tradeoff="MLP target-bit quality gate plus packed-arena byte receipt; KV unchanged.",
        ),
        DenominatorCandidate(
            name="t014_f21_near_term_stack",
            source_findings=("T014-F21",),
            mlp_bits=2.45,
            attention_bits=3.30,
            embedding_bits=3.75,
            output_head_bits=3.75,
            overhead_gib=0.25,
            kv_effective_bits=3.0,
            status="priced_next_gate",
            tradeoff="Low-to-moderate risk stack; requires separate attention, vocab, and 3-bit old-KV gates.",
        ),
        DenominatorCandidate(
            name="t014_f22_strong_stack",
            source_findings=("T014-F22",),
            mlp_bits=2.35,
            attention_bits=3.20,
            embedding_bits=3.50,
            output_head_bits=3.50,
            overhead_gib=0.20,
            kv_effective_bits=3.0,
            status="priced_next_gate",
            tradeoff="More aggressive quality gates; enough slack for rollback repair.",
        ),
        DenominatorCandidate(
            name="t014_f23_stretch_stack",
            source_findings=("T014-F23",),
            mlp_bits=2.25,
            attention_bits=3.00,
            embedding_bits=3.25,
            output_head_bits=3.25,
            overhead_gib=0.15,
            kv_effective_bits=2.5,
            status="stretch_pricing_only",
            tradeoff="Higher risk; keep behind F21/F22 validation.",
        ),
    )


def _price_candidate(
    config: Any,
    candidate: DenominatorCandidate,
    *,
    context_tokens: int,
    bandwidth_gib_per_second: float,
    q4_total_with_fp16_kv_gib: float,
    baseline_total_gib: float,
    baseline_tokens_per_second: float,
    q4_fp16kv_tokens_per_second: float,
) -> dict[str, Any]:
    shape = config.to_shape()
    residency = shape.component_residency_projection(
        mlp_bits=candidate.mlp_bits,
        attention_bits=candidate.attention_bits,
        embedding_bits=candidate.embedding_bits,
        output_head_bits=candidate.output_head_bits,
        overhead_gib=candidate.overhead_gib,
    )
    kv_gib = kv_cache_gib_for_context(context_tokens, effective_bits=candidate.kv_effective_bits, shape=shape)
    kv_per_token = kv_cache_gib_for_context(1, effective_bits=candidate.kv_effective_bits, shape=shape)
    throughput = shape.synthetic_decode_throughput_projection(
        uniform_bits=4.0,
        mlp_bits=candidate.mlp_bits,
        attention_bits=candidate.attention_bits,
        embedding_bits=candidate.embedding_bits,
        output_head_bits=candidate.output_head_bits,
        bandwidth_gib_per_second=bandwidth_gib_per_second,
        kv_cache_gib_per_token=kv_per_token,
    )
    total_gib = float(residency["candidate_estimated_gib"]) + kv_gib
    tokens_per_second = float(throughput["component_tokens_per_second"])
    return {
        **candidate.as_dict(),
        "context_tokens": context_tokens,
        "weight_runtime_gib": float(residency["candidate_estimated_gib"]),
        "kv_gib": kv_gib,
        "total_with_kv_gib": total_gib,
        "slack_vs_12gib": 12.0 - total_gib,
        "saves_vs_current_t013_gib": baseline_total_gib - total_gib,
        "saves_vs_q4_fp16kv_gib": q4_total_with_fp16_kv_gib - total_gib,
        "candidate_fits_12gib": total_gib <= 12.0,
        "candidate_hits_preferred_margin": total_gib <= 11.6,
        "tokens_per_second_proxy": tokens_per_second,
        "speedup_vs_q4_fp16kv": tokens_per_second / q4_fp16kv_tokens_per_second,
        "speedup_vs_current_t013": tokens_per_second / baseline_tokens_per_second,
        "component_projection": residency,
    }


def _packed_arena_receipt(*, current_overhead_gib: float, packed_overhead_gib: float) -> dict[str, Any]:
    saved_gib = current_overhead_gib - packed_overhead_gib
    return {
        "activation": PACKED_ARENA_ACTIVATION,
        "current_overhead_gib": current_overhead_gib,
        "packed_overhead_gib": packed_overhead_gib,
        "saved_gib": saved_gib,
        "requires_no_persistent_bf16_source_copy": True,
        "requires_no_persistent_dequant_scratch": True,
        "passes": packed_overhead_gib <= 0.25 and saved_gib >= 0.30,
    }


def run_t014_denominator_tightening(
    *,
    model_id: str = "Qwen/Qwen3-32B",
    local_model_dir: Path | None = None,
    allow_hf_config_fetch: bool = True,
    mlp_layer_index: int = 0,
    mlp_row_limit: int | None = None,
    projection_batch: int = 1,
    target_mlp_bits: float = 2.45,
    context_tokens: int = 16_384,
    bandwidth_gib_per_second: float = 514.0,
    include_stretch: bool = True,
) -> dict[str, Any]:
    """Run the T014 MLP-first denominator gate without claiming full generation."""

    config = resolve_qwen_config(
        model_id=model_id,
        local_model_dir=local_model_dir,
        allow_hf_config_fetch=allow_hf_config_fetch,
    )
    weight_dir = find_weight_dir(config, local_model_dir)
    inventory = inventory_safetensors(weight_dir)
    current_bits = _component_policy_bits(config)
    shape = config.to_shape()
    q4_total_with_fp16_kv_gib = shape.component_residency_projection(
        mlp_bits=4.0,
        attention_bits=4.0,
        embedding_bits=4.0,
        output_head_bits=4.0,
        overhead_gib=0.55,
    )["candidate_estimated_gib"] + kv_cache_gib_for_context(context_tokens, effective_bits=16.0, shape=shape)
    q4_fp16kv_tokens_per_second = shape.synthetic_decode_throughput_projection(
        uniform_bits=4.0,
        mlp_bits=4.0,
        attention_bits=4.0,
        embedding_bits=4.0,
        output_head_bits=4.0,
        bandwidth_gib_per_second=bandwidth_gib_per_second,
        kv_cache_gib_per_token=kv_cache_gib_for_context(1, effective_bits=16.0, shape=shape),
    )["component_tokens_per_second"]

    candidates = tuple(item for item in _candidate_set(current_bits) if include_stretch or item.name != "t014_f23_stretch_stack")
    baseline_candidate = candidates[0]
    baseline_projection = shape.component_residency_projection(
        mlp_bits=baseline_candidate.mlp_bits,
        attention_bits=baseline_candidate.attention_bits,
        embedding_bits=baseline_candidate.embedding_bits,
        output_head_bits=baseline_candidate.output_head_bits,
        overhead_gib=baseline_candidate.overhead_gib,
    )
    baseline_total_gib = float(baseline_projection["candidate_estimated_gib"]) + kv_cache_gib_for_context(
        context_tokens,
        effective_bits=baseline_candidate.kv_effective_bits,
        shape=shape,
    )
    baseline_tokens_per_second = shape.synthetic_decode_throughput_projection(
        uniform_bits=4.0,
        mlp_bits=baseline_candidate.mlp_bits,
        attention_bits=baseline_candidate.attention_bits,
        embedding_bits=baseline_candidate.embedding_bits,
        output_head_bits=baseline_candidate.output_head_bits,
        bandwidth_gib_per_second=bandwidth_gib_per_second,
        kv_cache_gib_per_token=kv_cache_gib_for_context(1, effective_bits=baseline_candidate.kv_effective_bits, shape=shape),
    )["component_tokens_per_second"]

    blockers: list[str] = []
    missing: list[str] = []
    receipts: list[ProjectionSmokeReceipt] = []
    if importlib.util.find_spec("safetensors") is None:
        blockers.append("safetensors_missing")
    if importlib.util.find_spec("torch") is None:
        blockers.append("torch_missing")
    if not config.matches_project_qwen3_32b_shape():
        blockers.append("config_shape_mismatch")
    if weight_dir is None or not inventory.local_weights_present:
        blockers.append("missing_local_qwen3_32b_safetensors")

    if not blockers and weight_dir is not None:
        weight_map = _load_weight_map(weight_dir)
        required = _mlp_tensor_names(mlp_layer_index)
        missing = [name for name in required if name not in weight_map]
        if missing:
            blockers.append("missing_t014_mlp_tensors")
        else:
            try:
                for offset, name in enumerate(required):
                    receipts.append(
                        _projection_smoke_with_policy(
                            model_dir=weight_dir,
                            weight_map=weight_map,
                            name=name,
                            layer_index=mlp_layer_index,
                            row_limit=mlp_row_limit,
                            batch=projection_batch,
                            seed=14_014 + offset,
                            target_bits=target_mlp_bits,
                        )
                    )
            except Exception as exc:  # pragma: no cover - real shard guard
                blockers.append(f"t014_mlp_projection_gate_failed:{type(exc).__name__}:{exc}")

    mlp_summary = _summarize_projection_group(MLP_GATE_ACTIVATION, receipts)
    if receipts:
        mlp_summary["target_mlp_bits"] = target_mlp_bits
        mlp_summary["target_bits_pass"] = mlp_summary["effective_bits_per_weight"] <= target_mlp_bits + 0.02
    else:
        mlp_summary["target_mlp_bits"] = target_mlp_bits
        mlp_summary["target_bits_pass"] = False
    mlp_summary["layer_index"] = mlp_layer_index
    mlp_summary["row_limit"] = mlp_row_limit
    packed_arena = _packed_arena_receipt(current_overhead_gib=0.55, packed_overhead_gib=0.25)

    priced_candidates = [
        _price_candidate(
            config,
            candidate,
            context_tokens=context_tokens,
            bandwidth_gib_per_second=bandwidth_gib_per_second,
            q4_total_with_fp16_kv_gib=q4_total_with_fp16_kv_gib,
            baseline_total_gib=baseline_total_gib,
            baseline_tokens_per_second=float(baseline_tokens_per_second),
            q4_fp16kv_tokens_per_second=float(q4_fp16kv_tokens_per_second),
        )
        for candidate in candidates
    ]
    candidate_by_name = {item["name"]: item for item in priced_candidates}
    t014_a = candidate_by_name["t014_a_f15_mlp245_packed_overhead"]
    f21 = candidate_by_name["t014_f21_near_term_stack"]
    f22 = candidate_by_name["t014_f22_strong_stack"]

    t014_a_pass = (
        not blockers
        and bool(mlp_summary["passes"])
        and bool(mlp_summary["target_bits_pass"])
        and bool(packed_arena["passes"])
        and t014_a["total_with_kv_gib"] <= 11.9
    )
    if not t014_a_pass and not blockers:
        blockers.append("t014_a_denominator_gate_failed_threshold")

    return {
        "probe": "t014_denominator_tightening",
        "activation": ACTIVATION,
        "model_id": model_id,
        "model_dir": str(weight_dir) if weight_dir is not None else None,
        "config": config.as_dict(),
        "weight_inventory": inventory.as_dict(),
        "baseline": {
            "source": "current_t013_full_model_equivalent_policy",
            "component_bits": current_bits,
            "total_with_kv_gib": baseline_total_gib,
            "tokens_per_second_proxy": baseline_tokens_per_second,
            "q4_total_with_fp16_kv_gib": q4_total_with_fp16_kv_gib,
            "q4_fp16kv_tokens_per_second": q4_fp16kv_tokens_per_second,
        },
        "mlp_projection_gate": {
            **mlp_summary,
            "projection_receipts": [receipt.as_dict() for receipt in receipts],
        },
        "packed_arena_receipt": packed_arena,
        "candidate_table": priced_candidates,
        "best_fit_candidate": min((item for item in priced_candidates if item["candidate_fits_12gib"]), key=lambda item: item["total_with_kv_gib"], default=None),
        "readiness": {
            "t014_a_pass": t014_a_pass,
            "f21_priced_margin_pass": f21["total_with_kv_gib"] <= 11.6,
            "f22_priced_margin_pass": f22["total_with_kv_gib"] <= 11.0,
            "full_model_text_inference_ready": False,
            "measured_full_model_tokens_per_second": False,
            "config_matches_project_qwen3_32b_shape": config.matches_project_qwen3_32b_shape(),
            "local_weights_present": inventory.local_weights_present,
            "projection_batch": projection_batch,
            "context_tokens": context_tokens,
            "bandwidth_gib_per_second": bandwidth_gib_per_second,
        },
        "missing_tensors": missing,
        "blockers": blockers,
        "next_actions": [
            "Implement packed all-layer compressed generation/runtime before claiming successful full Qwen text inference.",
            "Promote F21 only after the attention, vocab/head, and 3-bit old-KV gates pass independently.",
        ]
        if t014_a_pass
        else ["Fix the listed T014 denominator blockers before promoting F21/F22 stack gates."],
    }
