"""T006 project-owned compressed block smoke on real Qwen weights."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .t004_real_qwen_bridge import find_weight_dir, inventory_safetensors, resolve_qwen_config


ACTIVATION = "T006_REAL_RESIDENCY_BLOCK_SMOKE_ACTIVE"

DEFAULT_BLOCK_TENSORS = (
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.63.self_attn.q_proj.weight",
    "model.layers.63.mlp.down_proj.weight",
)


@dataclass(frozen=True)
class BlockCodecPolicy:
    name: str
    target_bits_per_weight: float
    repair_fraction: float
    threshold_multiplier: float = 0.75
    code_bits_per_weight: float = 2.0
    scale_bits_per_row: float = 16.0
    repair_value_bits: float = 16.0

    def effective_bits(self, cols: int) -> float:
        if cols <= 0:
            raise ValueError("cols must be positive")
        index_bits = math.ceil(math.log2(cols)) if cols > 1 else 1
        return (
            self.code_bits_per_weight
            + self.scale_bits_per_row / cols
            + self.repair_fraction * (self.repair_value_bits + index_bits)
        )

    def as_dict(self, cols: int) -> dict[str, Any]:
        data = asdict(self)
        data["effective_bits_per_weight"] = self.effective_bits(cols)
        return data


@dataclass(frozen=True)
class BlockSmokeResult:
    tensor: str
    shard: str
    policy: BlockCodecPolicy
    original_shape: tuple[int, ...]
    block_shape: tuple[int, int]
    effective_bits_per_weight: float
    q4_savings_fraction: float
    fp16_savings_fraction: float
    weight_relative_rmse: float
    matmul_relative_rmse: float
    sample_sha256: str

    @property
    def passes(self) -> bool:
        return (
            self.effective_bits_per_weight <= self.policy.target_bits_per_weight + 0.15
            and self.matmul_relative_rmse <= 0.45
            and self.weight_relative_rmse <= 0.70
        )

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["policy"] = self.policy.as_dict(self.block_shape[1])
        data["original_shape"] = list(self.original_shape)
        data["block_shape"] = list(self.block_shape)
        data["passes"] = self.passes
        return data


def _load_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return {str(name): str(shard) for name, shard in payload.get("weight_map", {}).items()}


def _policy_for_tensor(name: str) -> BlockCodecPolicy:
    if ".mlp." in name:
        return BlockCodecPolicy(name="mlp_ternary_sparse_repair", target_bits_per_weight=2.80, repair_fraction=0.03)
    if "embed_tokens" in name or "lm_head" in name:
        return BlockCodecPolicy(name="embedding_q4_boundary", target_bits_per_weight=4.0, repair_fraction=0.08)
    return BlockCodecPolicy(name="attention_ternary_sparse_repair", target_bits_per_weight=3.50, repair_fraction=0.06)


def _read_block(model_dir: Path, tensor_name: str, shard_name: str, rows: int, cols: int) -> tuple[Any, tuple[int, ...]]:
    from safetensors import safe_open

    with safe_open(model_dir / shard_name, framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(tensor_name)
        shape = tuple(int(dim) for dim in tensor_slice.get_shape())
        if len(shape) != 2:
            raise ValueError(f"{tensor_name} must be a rank-2 weight tensor, got shape {shape}")
        block = tensor_slice[: min(rows, shape[0]), : min(cols, shape[1])].detach().cpu().float()
    return block, shape


def _compress_reconstruct(block: Any, policy: BlockCodecPolicy) -> Any:
    import torch

    scales = block.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
    threshold = scales * policy.threshold_multiplier
    base = torch.where(block.abs() >= threshold, block.sign() * scales, torch.zeros_like(block))

    residual = block - base
    cols = int(block.shape[1])
    k = max(1, int(math.ceil(cols * policy.repair_fraction)))
    values, indices = torch.topk(residual.abs(), k=k, dim=1)
    signed_values = residual.gather(1, indices)
    repair = torch.zeros_like(block)
    repair.scatter_(1, indices, signed_values)
    return base + repair


def _relative_rmse(reference: Any, candidate: Any) -> float:
    import torch

    error = candidate - reference
    denom = torch.sqrt(torch.mean(reference.float() ** 2)).clamp_min(1e-8)
    return float(torch.sqrt(torch.mean(error.float() ** 2)) / denom)


def _block_smoke(model_dir: Path, tensor_name: str, shard_name: str, rows: int, cols: int, batch: int, seed: int) -> BlockSmokeResult:
    import torch

    block, original_shape = _read_block(model_dir, tensor_name, shard_name, rows, cols)
    policy = _policy_for_tensor(tensor_name)
    reconstructed = _compress_reconstruct(block, policy)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    activations = torch.randn(batch, block.shape[1], generator=generator)
    reference_out = activations @ block.T
    candidate_out = activations @ reconstructed.T

    digest = hashlib.sha256(block.numpy().tobytes() + reconstructed.numpy().tobytes()).hexdigest()
    effective_bits = policy.effective_bits(int(block.shape[1]))
    return BlockSmokeResult(
        tensor=tensor_name,
        shard=shard_name,
        policy=policy,
        original_shape=original_shape,
        block_shape=(int(block.shape[0]), int(block.shape[1])),
        effective_bits_per_weight=effective_bits,
        q4_savings_fraction=(4.0 - effective_bits) / 4.0,
        fp16_savings_fraction=(16.0 - effective_bits) / 16.0,
        weight_relative_rmse=_relative_rmse(block, reconstructed),
        matmul_relative_rmse=_relative_rmse(reference_out, candidate_out),
        sample_sha256=digest,
    )


def run_t006_real_residency_block_smoke(
    *,
    model_id: str = "Qwen/Qwen3-32B",
    local_model_dir: Path | None = None,
    allow_hf_config_fetch: bool = True,
    tensor_names: Sequence[str] | None = None,
    block_rows: int = 64,
    block_cols: int = 256,
    batch: int = 16,
    seed: int = 6006,
    min_required_blocks: int = 4,
) -> dict[str, Any]:
    """Exercise the project compressed-residency codec on real Qwen blocks."""

    config = resolve_qwen_config(
        model_id=model_id,
        local_model_dir=local_model_dir,
        allow_hf_config_fetch=allow_hf_config_fetch,
    )
    weight_dir = find_weight_dir(config, local_model_dir)
    inventory = inventory_safetensors(weight_dir)
    requested = tuple(tensor_names or DEFAULT_BLOCK_TENSORS)
    blockers: list[str] = []
    missing: list[str] = []
    blocks: list[BlockSmokeResult] = []

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
        for name in requested:
            shard = weight_map.get(name)
            if shard is None:
                missing.append(name)
                continue
            try:
                blocks.append(_block_smoke(weight_dir, name, shard, block_rows, block_cols, batch, seed + len(blocks)))
            except Exception as exc:  # pragma: no cover - real shard guard
                blockers.append(f"block_smoke_failed:{name}:{type(exc).__name__}:{exc}")
                break

    pass_count = sum(1 for item in blocks if item.passes)
    smoke_pass = pass_count >= min_required_blocks and len(blocks) >= min_required_blocks and not blockers
    if not smoke_pass and not blockers:
        blockers.append("insufficient_passing_real_residency_blocks")

    avg_effective_bits = sum(item.effective_bits_per_weight for item in blocks) / len(blocks) if blocks else None
    avg_q4_savings = sum(item.q4_savings_fraction for item in blocks) / len(blocks) if blocks else None
    avg_matmul_error = sum(item.matmul_relative_rmse for item in blocks) / len(blocks) if blocks else None

    return {
        "probe": "t006_real_residency_block_smoke",
        "activation": ACTIVATION,
        "model_id": model_id,
        "model_dir": str(weight_dir) if weight_dir is not None else None,
        "config": config.as_dict(),
        "weight_inventory": inventory.as_dict(),
        "requested_tensors": list(requested),
        "blocks": [item.as_dict() for item in blocks],
        "missing_tensors": missing,
        "readiness": {
            "smoke_pass": smoke_pass,
            "sampled_blocks": len(blocks),
            "passing_blocks": pass_count,
            "min_required_blocks": min_required_blocks,
            "block_rows": block_rows,
            "block_cols": block_cols,
            "batch": batch,
            "avg_effective_bits_per_weight": avg_effective_bits,
            "avg_q4_savings_fraction": avg_q4_savings,
            "avg_matmul_relative_rmse": avg_matmul_error,
            "config_matches_project_qwen3_32b_shape": config.matches_project_qwen3_32b_shape(),
            "local_weights_present": inventory.local_weights_present,
        },
        "blockers": blockers,
        "next_actions": [
            "Promote this block codec into a layer-level captured-residual/KV smoke before claiming full generation."
        ]
        if smoke_pass
        else ["Fix the listed blockers before treating the compressed residency codec as active on real Qwen weights."],
    }
