"""T009-T013 full-width runtime scaling probes for real Qwen weights."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .footprint import BYTES_PER_GIB
from .kv_margin import kv_cache_gib_for_context, qwen3_32b_kv_bytes_per_token
from .t004_real_qwen_bridge import find_weight_dir, inventory_safetensors, resolve_qwen_config
from .t006_real_residency_block_smoke import BlockCodecPolicy, _compress_reconstruct, _policy_for_tensor
from .t007_real_layer_smoke import _relative_rmse
from .t008_compressed_generation_smoke import _load_weight_map


ACTIVATION = "T009_T013_QWEN_RUNTIME_SCALER_ACTIVE"
T009_ACTIVATION = "T009_FULL_WIDTH_LAYER_PROJECTION_SMOKE_ACTIVE"
T010_ACTIVATION = "T010_MULTILAYER_WIDTH_SCALE_SMOKE_ACTIVE"
T011_ACTIVATION = "T011_TOKENIZER_PROMPT_SMOKE_ACTIVE"
T012_ACTIVATION = "T012_KV_CACHE_LEDGER_ACTIVE"
T013_ACTIVATION = "T013_FULL_MODEL_EQUIVALENT_PRICING_ACTIVE"


@dataclass(frozen=True)
class ProjectionSmokeReceipt:
    name: str
    layer_index: int
    shard: str
    policy: BlockCodecPolicy
    original_shape: tuple[int, int]
    slice_shape: tuple[int, int]
    effective_bits_per_weight: float
    q4_savings_fraction: float
    fp16_savings_fraction: float
    weight_relative_rmse: float
    projection_relative_rmse: float
    exact_seconds: float
    compressed_seconds: float
    sha256: str

    @property
    def element_count(self) -> int:
        return int(self.slice_shape[0]) * int(self.slice_shape[1])

    @property
    def passes(self) -> bool:
        return (
            math.isfinite(self.effective_bits_per_weight)
            and self.weight_relative_rmse <= 0.90
            and self.projection_relative_rmse <= 1.25
            and self.compressed_seconds > 0.0
        )

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["policy"] = self.policy.as_dict(self.slice_shape[1])
        data["original_shape"] = list(self.original_shape)
        data["slice_shape"] = list(self.slice_shape)
        data["passes"] = self.passes
        return data


def _read_weight_slice(model_dir: Path, tensor_name: str, shard_name: str, rows: int, cols: int) -> tuple[Any, tuple[int, int]]:
    from safetensors import safe_open

    with safe_open(model_dir / shard_name, framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(tensor_name)
        shape = tuple(int(dim) for dim in tensor_slice.get_shape())
        if len(shape) != 2:
            raise ValueError(f"{tensor_name} must be rank 2, got shape {shape}")
        block = tensor_slice[: min(rows, shape[0]), : min(cols, shape[1])].detach().cpu().float()
    return block, (shape[0], shape[1])


def _tensor_shape(model_dir: Path, weight_map: dict[str, str], name: str) -> tuple[int, int]:
    from safetensors import safe_open

    shard = weight_map[name]
    with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
        shape = tuple(int(dim) for dim in handle.get_slice(name).get_shape())
    if len(shape) != 2:
        raise ValueError(f"{name} must be rank 2, got shape {shape}")
    return shape[0], shape[1]


def _layer_tensor_names(layer_index: int) -> tuple[str, ...]:
    prefix = f"model.layers.{layer_index}"
    return (
        f"{prefix}.self_attn.q_proj.weight",
        f"{prefix}.self_attn.k_proj.weight",
        f"{prefix}.self_attn.v_proj.weight",
        f"{prefix}.self_attn.o_proj.weight",
        f"{prefix}.mlp.gate_proj.weight",
        f"{prefix}.mlp.up_proj.weight",
        f"{prefix}.mlp.down_proj.weight",
    )


def _deterministic_input(width: int, batch: int, seed: int) -> Any:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(batch, width, generator=generator) / math.sqrt(max(1, width))


def _projection_smoke(
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    name: str,
    layer_index: int,
    row_limit: int | None,
    batch: int,
    seed: int,
) -> ProjectionSmokeReceipt:
    import torch

    shard = weight_map[name]
    original_shape = _tensor_shape(model_dir, weight_map, name)
    rows = original_shape[0] if row_limit is None or row_limit <= 0 else min(row_limit, original_shape[0])
    cols = original_shape[1]
    exact, _ = _read_weight_slice(model_dir, name, shard, rows, cols)
    policy = _policy_for_tensor(name)
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


def _weighted_average(items: Sequence[ProjectionSmokeReceipt], attr: str) -> float:
    total = sum(item.element_count for item in items)
    if total <= 0:
        return 0.0
    return sum(float(getattr(item, attr)) * item.element_count for item in items) / total


def _summarize_projection_group(activation: str, receipts: Sequence[ProjectionSmokeReceipt]) -> dict[str, Any]:
    pass_count = sum(1 for receipt in receipts if receipt.passes)
    total_elements = sum(receipt.element_count for receipt in receipts)
    exact_seconds = sum(receipt.exact_seconds for receipt in receipts)
    compressed_seconds = sum(receipt.compressed_seconds for receipt in receipts)
    return {
        "activation": activation,
        "projection_count": len(receipts),
        "passing_projection_count": pass_count,
        "total_elements": total_elements,
        "effective_bits_per_weight": _weighted_average(receipts, "effective_bits_per_weight"),
        "q4_savings_fraction": _weighted_average(receipts, "q4_savings_fraction"),
        "fp16_savings_fraction": _weighted_average(receipts, "fp16_savings_fraction"),
        "avg_weight_relative_rmse": _weighted_average(receipts, "weight_relative_rmse"),
        "avg_projection_relative_rmse": _weighted_average(receipts, "projection_relative_rmse"),
        "exact_seconds": exact_seconds,
        "compressed_seconds": compressed_seconds,
        "compressed_projection_throughput_elements_per_second": total_elements / compressed_seconds if compressed_seconds > 0 else 0.0,
        "passes": len(receipts) > 0 and pass_count == len(receipts),
    }


def _tokenizer_prompt_smoke(model_dir: Path, prompt_text: str, *, require_tokenizer: bool) -> dict[str, Any]:
    blockers: list[str] = []
    token_ids: list[int] = []
    decoded_prefix = ""
    source = "transformers_tokenizer"
    if importlib.util.find_spec("transformers") is None:
        blockers.append("transformers_missing")
    else:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
            token_ids = [int(token) for token in tokenizer.encode(prompt_text, add_special_tokens=False)]
            decoded_prefix = tokenizer.decode(token_ids[: min(16, len(token_ids))])
        except Exception as exc:  # pragma: no cover - local tokenizer guard
            blockers.append(f"tokenizer_load_failed:{type(exc).__name__}:{exc}")
    if not token_ids and not require_tokenizer:
        source = "fallback_prompt_bytes"
        blockers = []
        token_ids = [int(byte) for byte in prompt_text.encode("utf-8")]
        decoded_prefix = prompt_text
    return {
        "activation": T011_ACTIVATION,
        "source": source,
        "prompt_text": prompt_text,
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "decoded_prefix": decoded_prefix,
        "passes": len(token_ids) > 0 and not blockers and (source == "transformers_tokenizer" or not require_tokenizer),
        "blockers": blockers,
    }


def _kv_cache_ledger(config: Any, *, context_tokens: Sequence[int], kv_effective_bits: float, exact_recent_tokens: int) -> dict[str, Any]:
    shape = config.to_shape()
    fp16_bytes_per_token = qwen3_32b_kv_bytes_per_token(shape)
    contexts: dict[str, Any] = {}
    for context in context_tokens:
        fp16_gib = kv_cache_gib_for_context(int(context), effective_bits=16.0, shape=shape)
        compressed_gib = kv_cache_gib_for_context(int(context), effective_bits=kv_effective_bits, shape=shape)
        exact_recent_gib = kv_cache_gib_for_context(min(int(context), exact_recent_tokens), effective_bits=16.0, shape=shape)
        old_tokens = max(0, int(context) - exact_recent_tokens)
        old_compressed_gib = kv_cache_gib_for_context(old_tokens, effective_bits=kv_effective_bits, shape=shape) if old_tokens else 0.0
        tiered_gib = exact_recent_gib + old_compressed_gib
        contexts[str(context)] = {
            "fp16_gib": fp16_gib,
            "uniform_compressed_gib": compressed_gib,
            "tiered_exact_recent_gib": tiered_gib,
            "tiered_savings_vs_fp16_gib": fp16_gib - tiered_gib,
            "tiered_savings_vs_fp16_fraction": (fp16_gib - tiered_gib) / fp16_gib if fp16_gib > 0 else 0.0,
        }
    return {
        "activation": T012_ACTIVATION,
        "fp16_bytes_per_token": fp16_bytes_per_token,
        "fp16_kib_per_token": fp16_bytes_per_token / 1024.0,
        "kv_effective_bits": kv_effective_bits,
        "exact_recent_tokens": exact_recent_tokens,
        "contexts": contexts,
        "passes": all(item["tiered_exact_recent_gib"] < item["fp16_gib"] for item in contexts.values()),
    }


def _component_policy_bits(config: Any) -> dict[str, float]:
    shape = config.to_shape()
    mlp_cols = (shape.hidden_size, shape.hidden_size, shape.intermediate_size)
    attn_cols = (shape.hidden_size, shape.hidden_size, shape.hidden_size, shape.query_width)
    mlp_policy = _policy_for_tensor("model.layers.0.mlp.gate_proj.weight")
    attn_policy = _policy_for_tensor("model.layers.0.self_attn.q_proj.weight")
    embed_policy = _policy_for_tensor("model.embed_tokens.weight")
    mlp_bits = sum(mlp_policy.effective_bits(cols) for cols in mlp_cols) / len(mlp_cols)
    attn_bits = sum(attn_policy.effective_bits(cols) for cols in attn_cols) / len(attn_cols)
    embed_bits = embed_policy.effective_bits(shape.hidden_size)
    return {
        "mlp_bits": mlp_bits,
        "attention_bits": attn_bits,
        "embedding_bits": embed_bits,
        "output_head_bits": embed_bits,
    }


def _full_model_equivalent_pricing(
    config: Any,
    *,
    bandwidth_gib_per_second: float,
    runtime_overhead_gib: float,
    kv_effective_bits: float,
    context_tokens: int,
    measured_full_width_summary: dict[str, Any],
) -> dict[str, Any]:
    shape = config.to_shape()
    bits = _component_policy_bits(config)
    residency = shape.component_residency_projection(
        mlp_bits=bits["mlp_bits"],
        attention_bits=bits["attention_bits"],
        embedding_bits=bits["embedding_bits"],
        output_head_bits=bits["output_head_bits"],
        overhead_gib=runtime_overhead_gib,
    )
    q4_residency = shape.component_residency_projection(
        mlp_bits=4.0,
        attention_bits=4.0,
        embedding_bits=4.0,
        output_head_bits=4.0,
        overhead_gib=runtime_overhead_gib,
    )
    kv_gib = kv_cache_gib_for_context(context_tokens, effective_bits=kv_effective_bits, shape=shape)
    throughput = shape.synthetic_decode_throughput_projection(
        uniform_bits=3.0,
        mlp_bits=bits["mlp_bits"],
        attention_bits=bits["attention_bits"],
        embedding_bits=bits["embedding_bits"],
        output_head_bits=bits["output_head_bits"],
        bandwidth_gib_per_second=bandwidth_gib_per_second,
        kv_cache_gib_per_token=kv_cache_gib_for_context(1, effective_bits=kv_effective_bits, shape=shape),
    )
    q4_throughput = shape.synthetic_decode_throughput_projection(
        uniform_bits=4.0,
        mlp_bits=4.0,
        attention_bits=4.0,
        embedding_bits=4.0,
        output_head_bits=4.0,
        bandwidth_gib_per_second=bandwidth_gib_per_second,
        kv_cache_gib_per_token=kv_cache_gib_for_context(1, effective_bits=16.0, shape=shape),
    )
    candidate_total_gib = float(residency["candidate_estimated_gib"]) + kv_gib
    q4_total_gib = float(q4_residency["candidate_estimated_gib"]) + kv_cache_gib_for_context(context_tokens, effective_bits=16.0, shape=shape)
    candidate_fits = candidate_total_gib <= 12.0
    pricing_blockers = [] if candidate_fits else ["candidate_exceeds_12gib_envelope"]
    return {
        "activation": T013_ACTIVATION,
        "component_bits": bits,
        "runtime_overhead_gib": runtime_overhead_gib,
        "context_tokens": context_tokens,
        "kv_effective_bits": kv_effective_bits,
        "candidate_weight_gib": residency["candidate_estimated_gib"],
        "candidate_total_with_kv_gib": candidate_total_gib,
        "q4_total_with_fp16_kv_gib": q4_total_gib,
        "candidate_saves_vs_q4_total_gib": q4_total_gib - candidate_total_gib,
        "candidate_fits_12gib": candidate_fits,
        "full_model_equivalent_tokens_per_second": throughput["component_tokens_per_second"],
        "q4_fp16kv_tokens_per_second": q4_throughput["component_tokens_per_second"],
        "speedup_vs_q4_fp16kv": throughput["component_tokens_per_second"] / q4_throughput["component_tokens_per_second"],
        "measured_full_width_projection_summary": measured_full_width_summary,
        "pricing_blockers": pricing_blockers,
        "passes": (q4_total_gib - candidate_total_gib) > 0.0
        and throughput["component_tokens_per_second"] > q4_throughput["component_tokens_per_second"],
    }


def run_t009_t013_runtime_scaler(
    *,
    model_id: str = "Qwen/Qwen3-32B",
    local_model_dir: Path | None = None,
    allow_hf_config_fetch: bool = True,
    full_layer_index: int = 0,
    scale_layer_indices: Sequence[int] = (0, 1, 2, 3),
    scale_row_limit: int = 512,
    projection_batch: int = 1,
    prompt_text: str = "Local compressed Qwen smoke:",
    require_tokenizer: bool = True,
    kv_context_tokens: Sequence[int] = (4096, 8192, 16384),
    kv_effective_bits: float = 4.0,
    exact_recent_tokens: int = 1024,
    bandwidth_gib_per_second: float = 514.0,
    runtime_overhead_gib: float = 0.55,
) -> dict[str, Any]:
    """Run the T009-T013 staged scaler without mutating the default model path."""

    config = resolve_qwen_config(
        model_id=model_id,
        local_model_dir=local_model_dir,
        allow_hf_config_fetch=allow_hf_config_fetch,
    )
    weight_dir = find_weight_dir(config, local_model_dir)
    inventory = inventory_safetensors(weight_dir)
    blockers: list[str] = []
    missing: list[str] = []
    t009_receipts: list[ProjectionSmokeReceipt] = []
    t010_receipts: list[ProjectionSmokeReceipt] = []
    tokenizer = {"activation": T011_ACTIVATION, "passes": False, "blockers": ["not_run"]}
    kv_ledger: dict[str, Any] = {"activation": T012_ACTIVATION, "passes": False}
    pricing: dict[str, Any] = {"activation": T013_ACTIVATION, "passes": False}

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
        full_names = _layer_tensor_names(full_layer_index)
        scale_names = tuple(name for layer_index in scale_layer_indices for name in _layer_tensor_names(int(layer_index)))
        required = tuple(dict.fromkeys(full_names + scale_names + ("model.embed_tokens.weight", "lm_head.weight")))
        missing = [name for name in required if name not in weight_map]
        if missing:
            blockers.append("missing_runtime_scaler_tensors")
        else:
            try:
                for offset, name in enumerate(full_names):
                    t009_receipts.append(
                        _projection_smoke(
                            model_dir=weight_dir,
                            weight_map=weight_map,
                            name=name,
                            layer_index=full_layer_index,
                            row_limit=None,
                            batch=projection_batch,
                            seed=9009 + offset,
                        )
                    )
                for layer_offset, layer_index in enumerate(scale_layer_indices):
                    for tensor_offset, name in enumerate(_layer_tensor_names(int(layer_index))):
                        t010_receipts.append(
                            _projection_smoke(
                                model_dir=weight_dir,
                                weight_map=weight_map,
                                name=name,
                                layer_index=int(layer_index),
                                row_limit=scale_row_limit,
                                batch=projection_batch,
                                seed=10_010 + layer_offset * 17 + tensor_offset,
                            )
                        )
                tokenizer = _tokenizer_prompt_smoke(weight_dir, prompt_text, require_tokenizer=require_tokenizer)
                kv_ledger = _kv_cache_ledger(
                    config,
                    context_tokens=kv_context_tokens,
                    kv_effective_bits=kv_effective_bits,
                    exact_recent_tokens=exact_recent_tokens,
                )
                t009_summary = _summarize_projection_group(T009_ACTIVATION, t009_receipts)
                pricing = _full_model_equivalent_pricing(
                    config,
                    bandwidth_gib_per_second=bandwidth_gib_per_second,
                    runtime_overhead_gib=runtime_overhead_gib,
                    kv_effective_bits=kv_effective_bits,
                    context_tokens=max(int(item) for item in kv_context_tokens),
                    measured_full_width_summary=t009_summary,
                )
            except Exception as exc:  # pragma: no cover - real shard guard
                blockers.append(f"runtime_scaler_failed:{type(exc).__name__}:{exc}")

    t009_summary = _summarize_projection_group(T009_ACTIVATION, t009_receipts)
    t010_summary = _summarize_projection_group(T010_ACTIVATION, t010_receipts)
    t010_summary["layer_indices"] = [int(index) for index in scale_layer_indices]
    t010_summary["row_limit"] = scale_row_limit
    smoke_pass = (
        t009_summary["passes"]
        and t010_summary["passes"]
        and bool(tokenizer.get("passes"))
        and bool(kv_ledger.get("passes"))
        and bool(pricing.get("passes"))
        and not blockers
    )
    if not smoke_pass and not blockers:
        blockers.append("t009_t013_runtime_scaler_failed_threshold")

    pricing_blockers = list(pricing.get("pricing_blockers", []))
    if smoke_pass and pricing_blockers:
        next_actions = [
            "Tighten the full-model-equivalent denominator below 12 GiB before claiming local full Qwen residency.",
            "Replace projection smoke with fused GPU kernels and an all-layer tokenizer/KV generation path before claiming measured full Qwen tok/s.",
        ]
    elif smoke_pass:
        next_actions = [
            "Replace the projection smoke with fused GPU kernels and an all-layer tokenizer/KV generation path before claiming measured full Qwen tok/s."
        ]
    else:
        next_actions = ["Fix the listed scaler blockers before using T009-T013 as the next runtime bridge."]

    return {
        "probe": "t009_t013_runtime_scaler",
        "activation": ACTIVATION,
        "model_id": model_id,
        "model_dir": str(weight_dir) if weight_dir is not None else None,
        "config": config.as_dict(),
        "weight_inventory": inventory.as_dict(),
        "t009_full_width_layer": {
            **t009_summary,
            "layer_index": full_layer_index,
            "projection_receipts": [receipt.as_dict() for receipt in t009_receipts],
        },
        "t010_multilayer_scale": {
            **t010_summary,
            "projection_receipts": [receipt.as_dict() for receipt in t010_receipts],
        },
        "t011_tokenizer_prompt": tokenizer,
        "t012_kv_cache_ledger": kv_ledger,
        "t013_full_model_equivalent_pricing": pricing,
        "missing_tensors": missing,
        "readiness": {
            "smoke_pass": smoke_pass,
            "full_layer_index": full_layer_index,
            "scale_layer_indices": [int(index) for index in scale_layer_indices],
            "scale_row_limit": scale_row_limit,
            "projection_batch": projection_batch,
            "require_tokenizer": require_tokenizer,
            "config_matches_project_qwen3_32b_shape": config.matches_project_qwen3_32b_shape(),
            "local_weights_present": inventory.local_weights_present,
        },
        "blockers": blockers,
        "pricing_blockers": pricing_blockers,
        "next_actions": next_actions,
    }
