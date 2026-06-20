"""T007 sliced real-layer smoke using the project compressed block codec."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .t004_real_qwen_bridge import find_weight_dir, inventory_safetensors, resolve_qwen_config
from .t006_real_residency_block_smoke import BlockCodecPolicy, _compress_reconstruct, _policy_for_tensor


ACTIVATION = "T007_REAL_LAYER_RESIDENCY_SMOKE_ACTIVE"


@dataclass(frozen=True)
class LayerTensorReceipt:
    name: str
    shard: str
    policy: BlockCodecPolicy
    original_shape: tuple[int, ...]
    slice_shape: tuple[int, int]
    effective_bits_per_weight: float
    q4_savings_fraction: float
    weight_relative_rmse: float

    @property
    def element_count(self) -> int:
        return self.slice_shape[0] * self.slice_shape[1]

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["policy"] = self.policy.as_dict(self.slice_shape[1])
        data["original_shape"] = list(self.original_shape)
        data["slice_shape"] = list(self.slice_shape)
        data["element_count"] = self.element_count
        return data


@dataclass(frozen=True)
class LayerSmokeReceipt:
    layer_index: int
    hidden_cols: int
    attention_width: int
    mlp_width: int
    sequence_tokens: int
    effective_bits_per_weight: float
    q4_savings_fraction: float
    attention_relative_rmse: float
    mlp_relative_rmse: float
    layer_relative_rmse: float
    activation_sha256: str

    @property
    def passes(self) -> bool:
        return (
            self.effective_bits_per_weight <= 3.15
            and self.attention_relative_rmse <= 0.95
            and self.mlp_relative_rmse <= 0.85
            and self.layer_relative_rmse <= 0.08
        )

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
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
            raise ValueError(f"{tensor_name} must be a rank-2 weight tensor, got shape {shape}")
        block = tensor_slice[: min(rows, shape[0]), : min(cols, shape[1])].detach().cpu().float()
    return block, shape


def _relative_rmse(reference: Any, candidate: Any) -> float:
    import torch

    error = candidate - reference
    denom = torch.sqrt(torch.mean(reference.float() ** 2)).clamp_min(1e-8)
    return float(torch.sqrt(torch.mean(error.float() ** 2)) / denom)


def _silu(value: Any) -> Any:
    import torch

    return value * torch.sigmoid(value)


def _load_layer_tensor(
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
    block, original_shape = _read_weight_slice(model_dir, name, shard, rows, cols)
    policy = _policy_for_tensor(name)
    reconstructed = _compress_reconstruct(block, policy)
    effective_bits = policy.effective_bits(int(block.shape[1]))
    receipt = LayerTensorReceipt(
        name=name,
        shard=shard,
        policy=policy,
        original_shape=original_shape,
        slice_shape=(int(block.shape[0]), int(block.shape[1])),
        effective_bits_per_weight=effective_bits,
        q4_savings_fraction=(4.0 - effective_bits) / 4.0,
        weight_relative_rmse=_relative_rmse(block, reconstructed),
    )
    return block, reconstructed, receipt


def _weighted_average(items: list[LayerTensorReceipt], attr: str) -> float | None:
    total = sum(item.element_count for item in items)
    if total <= 0:
        return None
    return sum(float(getattr(item, attr)) * item.element_count for item in items) / total


def _run_sliced_layer(
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    layer_index: int,
    hidden_cols: int,
    attention_width: int,
    mlp_width: int,
    sequence_tokens: int,
    seed: int,
) -> tuple[LayerSmokeReceipt, list[LayerTensorReceipt]]:
    import torch

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
        exact_block, compressed_block, receipt = _load_layer_tensor(
            model_dir=model_dir,
            weight_map=weight_map,
            name=name,
            rows=rows,
            cols=cols,
        )
        exact[key] = exact_block
        compressed[key] = compressed_block
        receipts.append(receipt)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn(sequence_tokens, hidden_cols, generator=generator)

    def layer_path(weights: dict[str, Any]) -> tuple[Any, Any, Any]:
        q = x @ weights["q"].T
        k = x @ weights["k"].T
        v = x @ weights["v"].T
        attention_scores = (q @ k.T) / math.sqrt(max(1, q.shape[1]))
        attention = torch.softmax(attention_scores, dim=-1) @ v
        attention_out = attention @ weights["o"].T

        gate = _silu(x @ weights["gate"].T)
        up = x @ weights["up"].T
        mlp_hidden = gate * up
        mlp_out = mlp_hidden @ weights["down"].T
        return attention_out, mlp_out, x + attention_out + mlp_out

    attention_exact, mlp_exact, layer_exact = layer_path(exact)
    attention_compressed, mlp_compressed, layer_compressed = layer_path(compressed)
    digest = hashlib.sha256(x.numpy().tobytes() + layer_compressed.detach().numpy().tobytes()).hexdigest()

    receipt = LayerSmokeReceipt(
        layer_index=layer_index,
        hidden_cols=hidden_cols,
        attention_width=attention_width,
        mlp_width=mlp_width,
        sequence_tokens=sequence_tokens,
        effective_bits_per_weight=float(_weighted_average(receipts, "effective_bits_per_weight") or 0.0),
        q4_savings_fraction=float(_weighted_average(receipts, "q4_savings_fraction") or 0.0),
        attention_relative_rmse=_relative_rmse(attention_exact, attention_compressed),
        mlp_relative_rmse=_relative_rmse(mlp_exact, mlp_compressed),
        layer_relative_rmse=_relative_rmse(layer_exact, layer_compressed),
        activation_sha256=digest,
    )
    return receipt, receipts


def run_t007_real_layer_residency_smoke(
    *,
    model_id: str = "Qwen/Qwen3-32B",
    local_model_dir: Path | None = None,
    allow_hf_config_fetch: bool = True,
    layer_index: int = 0,
    hidden_cols: int = 256,
    attention_width: int = 128,
    mlp_width: int = 384,
    sequence_tokens: int = 8,
    seed: int = 7007,
) -> dict[str, Any]:
    """Run a sliced real Qwen layer through exact and compressed local paths."""

    config = resolve_qwen_config(
        model_id=model_id,
        local_model_dir=local_model_dir,
        allow_hf_config_fetch=allow_hf_config_fetch,
    )
    weight_dir = find_weight_dir(config, local_model_dir)
    inventory = inventory_safetensors(weight_dir)
    blockers: list[str] = []
    missing: list[str] = []
    layer_receipt: LayerSmokeReceipt | None = None
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
        required = [
            f"model.layers.{layer_index}.self_attn.q_proj.weight",
            f"model.layers.{layer_index}.self_attn.k_proj.weight",
            f"model.layers.{layer_index}.self_attn.v_proj.weight",
            f"model.layers.{layer_index}.self_attn.o_proj.weight",
            f"model.layers.{layer_index}.mlp.gate_proj.weight",
            f"model.layers.{layer_index}.mlp.up_proj.weight",
            f"model.layers.{layer_index}.mlp.down_proj.weight",
        ]
        missing = [name for name in required if name not in weight_map]
        if missing:
            blockers.append("missing_layer_tensors")
        else:
            try:
                layer_receipt, tensor_receipts = _run_sliced_layer(
                    model_dir=weight_dir,
                    weight_map=weight_map,
                    layer_index=layer_index,
                    hidden_cols=hidden_cols,
                    attention_width=attention_width,
                    mlp_width=mlp_width,
                    sequence_tokens=sequence_tokens,
                    seed=seed,
                )
            except Exception as exc:  # pragma: no cover - real shard guard
                blockers.append(f"layer_smoke_failed:{type(exc).__name__}:{exc}")

    smoke_pass = layer_receipt.passes if layer_receipt is not None and not blockers else False
    if not smoke_pass and not blockers:
        blockers.append("real_layer_residency_smoke_failed_threshold")

    return {
        "probe": "t007_real_layer_residency_smoke",
        "activation": ACTIVATION,
        "model_id": model_id,
        "model_dir": str(weight_dir) if weight_dir is not None else None,
        "config": config.as_dict(),
        "weight_inventory": inventory.as_dict(),
        "layer_receipt": layer_receipt.as_dict() if layer_receipt is not None else None,
        "tensor_receipts": [item.as_dict() for item in tensor_receipts],
        "missing_tensors": missing,
        "readiness": {
            "smoke_pass": smoke_pass,
            "layer_index": layer_index,
            "hidden_cols": hidden_cols,
            "attention_width": attention_width,
            "mlp_width": mlp_width,
            "sequence_tokens": sequence_tokens,
            "config_matches_project_qwen3_32b_shape": config.matches_project_qwen3_32b_shape(),
            "local_weights_present": inventory.local_weights_present,
        },
        "blockers": blockers,
        "next_actions": [
            "Promote the sliced layer smoke into a multi-layer compressed loader and generation loop."
        ]
        if smoke_pass
        else ["Tune the layer codec policy or add captured residual repair before building the full compressed loader."],
    }
