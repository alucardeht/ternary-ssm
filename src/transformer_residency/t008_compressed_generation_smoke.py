"""T008 sliced multi-layer compressed generation smoke on real Qwen weights."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .t004_real_qwen_bridge import find_weight_dir, inventory_safetensors, resolve_qwen_config
from .t006_real_residency_block_smoke import BlockCodecPolicy, _compress_reconstruct, _policy_for_tensor
from .t007_real_layer_smoke import LayerTensorReceipt, _relative_rmse


ACTIVATION = "T008_COMPRESSED_GENERATION_SMOKE_ACTIVE"


@dataclass(frozen=True)
class GenerationSmokeReceipt:
    layer_indices: tuple[int, ...]
    hidden_cols: int
    attention_width: int
    mlp_width: int
    vocab_rows: int
    prompt_tokens: tuple[int, ...]
    generated_tokens: tuple[int, ...]
    exact_top1_tokens: tuple[int, ...]
    top1_agreement: float
    avg_logits_relative_rmse: float
    max_logits_relative_rmse: float
    effective_bits_per_weight: float
    q4_savings_fraction: float
    compressed_seconds: float
    compressed_tokens_per_second: float
    generation_sha256: str

    @property
    def passes(self) -> bool:
        return (
            len(self.generated_tokens) > 0
            and self.effective_bits_per_weight <= 3.25
            and self.q4_savings_fraction >= 0.15
            and self.avg_logits_relative_rmse <= 1.50
            and self.max_logits_relative_rmse <= 3.00
            and self.compressed_tokens_per_second > 0.0
        )

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["layer_indices"] = list(self.layer_indices)
        data["prompt_tokens"] = list(self.prompt_tokens)
        data["generated_tokens"] = list(self.generated_tokens)
        data["exact_top1_tokens"] = list(self.exact_top1_tokens)
        data["passes"] = self.passes
        return data


def _load_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return {str(name): str(shard) for name, shard in payload.get("weight_map", {}).items()}


def _read_weight_slice(model_dir: Path, tensor_name: str, shard_name: str, rows: int, cols: int) -> tuple[Any, tuple[int, ...]]:
    from safetensors import safe_open

    with safe_open(model_dir / shard_name, framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(tensor_name)
        shape = tuple(int(dim) for dim in tensor_slice.get_shape())
        if len(shape) != 2:
            raise ValueError(f"{tensor_name} must be rank 2, got shape {shape}")
        block = tensor_slice[: min(rows, shape[0]), : min(cols, shape[1])].detach().cpu().float()
    return block, shape


def _load_project_tensor(
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    name: str,
    rows: int,
    cols: int,
) -> tuple[Any, Any, LayerTensorReceipt]:
    shard = weight_map.get(name)
    if shard is None:
        raise KeyError(name)
    exact, original_shape = _read_weight_slice(model_dir, name, shard, rows, cols)
    policy = _policy_for_tensor(name)
    compressed = _compress_reconstruct(exact, policy)
    effective_bits = policy.effective_bits(int(exact.shape[1]))
    receipt = LayerTensorReceipt(
        name=name,
        shard=shard,
        policy=policy,
        original_shape=original_shape,
        slice_shape=(int(exact.shape[0]), int(exact.shape[1])),
        effective_bits_per_weight=effective_bits,
        q4_savings_fraction=(4.0 - effective_bits) / 4.0,
        weight_relative_rmse=_relative_rmse(exact, compressed),
    )
    return exact, compressed, receipt


def _weighted_average(items: Sequence[LayerTensorReceipt], attr: str) -> float:
    total = sum(item.element_count for item in items)
    if total <= 0:
        return 0.0
    return sum(float(getattr(item, attr)) * item.element_count for item in items) / total


def _silu(value: Any) -> Any:
    import torch

    return value * torch.sigmoid(value)


def _layer_forward(x: Any, weights: dict[str, Any]) -> Any:
    import torch

    q = x @ weights["q"].T
    k = x @ weights["k"].T
    v = x @ weights["v"].T
    attention_scores = (q @ k.T) / math.sqrt(max(1, q.shape[1]))
    attention = torch.softmax(attention_scores, dim=-1) @ v
    attention_out = attention @ weights["o"].T

    gate = _silu(x @ weights["gate"].T)
    up = x @ weights["up"].T
    mlp_out = (gate * up) @ weights["down"].T
    return x + attention_out + mlp_out


def _load_layer_pair(
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    layer_index: int,
    hidden_cols: int,
    attention_width: int,
    mlp_width: int,
) -> tuple[dict[str, Any], dict[str, Any], list[LayerTensorReceipt]]:
    prefix = f"model.layers.{layer_index}"
    specs = {
        "q": (f"{prefix}.self_attn.q_proj.weight", attention_width, hidden_cols),
        "k": (f"{prefix}.self_attn.k_proj.weight", attention_width, hidden_cols),
        "v": (f"{prefix}.self_attn.v_proj.weight", attention_width, hidden_cols),
        "o": (f"{prefix}.self_attn.o_proj.weight", hidden_cols, attention_width),
        "gate": (f"{prefix}.mlp.gate_proj.weight", mlp_width, hidden_cols),
        "up": (f"{prefix}.mlp.up_proj.weight", mlp_width, hidden_cols),
        "down": (f"{prefix}.mlp.down_proj.weight", hidden_cols, mlp_width),
    }
    exact: dict[str, Any] = {}
    compressed: dict[str, Any] = {}
    receipts: list[LayerTensorReceipt] = []
    for key, (name, rows, cols) in specs.items():
        exact_block, compressed_block, receipt = _load_project_tensor(
            model_dir=model_dir,
            weight_map=weight_map,
            name=name,
            rows=rows,
            cols=cols,
        )
        exact[key] = exact_block
        compressed[key] = compressed_block
        receipts.append(receipt)
    return exact, compressed, receipts


def _decode_once(tokens: Sequence[int], embed: Any, layers: Sequence[dict[str, Any]], lm_head: Any) -> Any:
    import torch

    ids = torch.tensor([int(token) % int(embed.shape[0]) for token in tokens], dtype=torch.long)
    x = embed.index_select(0, ids)
    for layer in layers:
        x = _layer_forward(x, layer)
    return x[-1] @ lm_head.T


def _run_micro_generation(
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    layer_indices: Sequence[int],
    hidden_cols: int,
    attention_width: int,
    mlp_width: int,
    vocab_rows: int,
    prompt_tokens: Sequence[int],
    max_new_tokens: int,
) -> tuple[GenerationSmokeReceipt, list[LayerTensorReceipt]]:
    import torch

    exact_embed, compressed_embed, embed_receipt = _load_project_tensor(
        model_dir=model_dir,
        weight_map=weight_map,
        name="model.embed_tokens.weight",
        rows=vocab_rows,
        cols=hidden_cols,
    )
    exact_lm, compressed_lm, lm_receipt = _load_project_tensor(
        model_dir=model_dir,
        weight_map=weight_map,
        name="lm_head.weight",
        rows=vocab_rows,
        cols=hidden_cols,
    )

    exact_layers: list[dict[str, Any]] = []
    compressed_layers: list[dict[str, Any]] = []
    receipts: list[LayerTensorReceipt] = [embed_receipt, lm_receipt]
    for layer_index in layer_indices:
        exact_layer, compressed_layer, layer_receipts = _load_layer_pair(
            model_dir=model_dir,
            weight_map=weight_map,
            layer_index=int(layer_index),
            hidden_cols=hidden_cols,
            attention_width=attention_width,
            mlp_width=mlp_width,
        )
        exact_layers.append(exact_layer)
        compressed_layers.append(compressed_layer)
        receipts.extend(layer_receipts)

    generated: list[int] = []
    exact_top1: list[int] = []
    logits_errors: list[float] = []
    tokens = [int(token) % vocab_rows for token in prompt_tokens]

    started = time.perf_counter()
    for _ in range(max_new_tokens):
        exact_logits = _decode_once(tokens, exact_embed, exact_layers, exact_lm)
        compressed_logits = _decode_once(tokens, compressed_embed, compressed_layers, compressed_lm)
        logits_errors.append(_relative_rmse(exact_logits, compressed_logits))
        exact_next = int(torch.argmax(exact_logits).item())
        compressed_next = int(torch.argmax(compressed_logits).item())
        exact_top1.append(exact_next)
        generated.append(compressed_next)
        tokens.append(compressed_next)
    elapsed = time.perf_counter() - started

    agreement = sum(1 for exact, compressed in zip(exact_top1, generated) if exact == compressed) / len(generated)
    digest = hashlib.sha256(json.dumps({"prompt": list(prompt_tokens), "generated": generated}).encode("utf-8")).hexdigest()
    receipt = GenerationSmokeReceipt(
        layer_indices=tuple(int(index) for index in layer_indices),
        hidden_cols=hidden_cols,
        attention_width=attention_width,
        mlp_width=mlp_width,
        vocab_rows=vocab_rows,
        prompt_tokens=tuple(int(token) for token in prompt_tokens),
        generated_tokens=tuple(generated),
        exact_top1_tokens=tuple(exact_top1),
        top1_agreement=agreement,
        avg_logits_relative_rmse=sum(logits_errors) / len(logits_errors),
        max_logits_relative_rmse=max(logits_errors),
        effective_bits_per_weight=_weighted_average(receipts, "effective_bits_per_weight"),
        q4_savings_fraction=_weighted_average(receipts, "q4_savings_fraction"),
        compressed_seconds=elapsed,
        compressed_tokens_per_second=len(generated) / elapsed if elapsed > 0 else 0.0,
        generation_sha256=digest,
    )
    return receipt, receipts


def run_t008_compressed_generation_smoke(
    *,
    model_id: str = "Qwen/Qwen3-32B",
    local_model_dir: Path | None = None,
    allow_hf_config_fetch: bool = True,
    layer_indices: Sequence[int] = (0, 1),
    hidden_cols: int = 256,
    attention_width: int = 128,
    mlp_width: int = 384,
    vocab_rows: int = 512,
    prompt_tokens: Sequence[int] = (1, 2, 3, 4),
    max_new_tokens: int = 4,
) -> dict[str, Any]:
    """Run a real-weight sliced multi-layer compressed generation loop."""

    config = resolve_qwen_config(
        model_id=model_id,
        local_model_dir=local_model_dir,
        allow_hf_config_fetch=allow_hf_config_fetch,
    )
    weight_dir = find_weight_dir(config, local_model_dir)
    inventory = inventory_safetensors(weight_dir)
    blockers: list[str] = []
    missing: list[str] = []
    generation_receipt: GenerationSmokeReceipt | None = None
    tensor_receipts: list[LayerTensorReceipt] = []

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
        required = ["model.embed_tokens.weight", "lm_head.weight"]
        for layer_index in layer_indices:
            prefix = f"model.layers.{int(layer_index)}"
            required.extend(
                [
                    f"{prefix}.self_attn.q_proj.weight",
                    f"{prefix}.self_attn.k_proj.weight",
                    f"{prefix}.self_attn.v_proj.weight",
                    f"{prefix}.self_attn.o_proj.weight",
                    f"{prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.up_proj.weight",
                    f"{prefix}.mlp.down_proj.weight",
                ]
            )
        missing = [name for name in required if name not in weight_map]
        if missing:
            blockers.append("missing_generation_tensors")
        else:
            try:
                generation_receipt, tensor_receipts = _run_micro_generation(
                    model_dir=weight_dir,
                    weight_map=weight_map,
                    layer_indices=layer_indices,
                    hidden_cols=hidden_cols,
                    attention_width=attention_width,
                    mlp_width=mlp_width,
                    vocab_rows=vocab_rows,
                    prompt_tokens=prompt_tokens,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as exc:  # pragma: no cover - real shard guard
                blockers.append(f"generation_smoke_failed:{type(exc).__name__}:{exc}")

    smoke_pass = generation_receipt.passes if generation_receipt is not None and not blockers else False
    if not smoke_pass and not blockers:
        blockers.append("compressed_generation_smoke_failed_threshold")

    return {
        "probe": "t008_compressed_generation_smoke",
        "activation": ACTIVATION,
        "model_id": model_id,
        "model_dir": str(weight_dir) if weight_dir is not None else None,
        "config": config.as_dict(),
        "weight_inventory": inventory.as_dict(),
        "generation_receipt": generation_receipt.as_dict() if generation_receipt is not None else None,
        "tensor_receipts": [item.as_dict() for item in tensor_receipts],
        "missing_tensors": missing,
        "readiness": {
            "smoke_pass": smoke_pass,
            "layer_indices": [int(index) for index in layer_indices],
            "hidden_cols": hidden_cols,
            "attention_width": attention_width,
            "mlp_width": mlp_width,
            "vocab_rows": vocab_rows,
            "max_new_tokens": max_new_tokens,
            "config_matches_project_qwen3_32b_shape": config.matches_project_qwen3_32b_shape(),
            "local_weights_present": inventory.local_weights_present,
        },
        "blockers": blockers,
        "next_actions": [
            "Scale the compressed generation smoke to more layers and replace the sliced surrogate path with a full hidden-width loader."
        ]
        if smoke_pass
        else ["Tighten logits agreement or policy thresholds before scaling the compressed generation path."],
    }
