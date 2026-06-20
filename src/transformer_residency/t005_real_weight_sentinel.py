"""T005 sentinel that samples real Qwen safetensors without full-model load."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .t004_real_qwen_bridge import (
    find_weight_dir,
    inventory_safetensors,
    resolve_qwen_config,
)


ACTIVATION = "T005_REAL_WEIGHT_SENTINEL_ACTIVE"

DEFAULT_SENTINEL_TENSORS = (
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.63.self_attn.q_proj.weight",
    "model.layers.63.mlp.down_proj.weight",
    "lm_head.weight",
)


@dataclass(frozen=True)
class TensorSentinel:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    sample_shape: tuple[int, ...]
    finite_ratio: float
    nonzero_ratio: float
    abs_mean: float
    std: float
    sample_sha256: str

    @property
    def passes(self) -> bool:
        return self.finite_ratio >= 0.999 and self.nonzero_ratio > 0.0 and self.abs_mean > 0.0

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        data["sample_shape"] = list(self.sample_shape)
        data["passes"] = self.passes
        return data


def _load_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return {str(name): str(shard) for name, shard in payload.get("weight_map", {}).items()}


def _slice_prefix(tensor_slice: Any, sample_rows: int, sample_cols: int) -> Any:
    shape = tuple(int(dim) for dim in tensor_slice.get_shape())
    if len(shape) == 0:
        return tensor_slice[:]
    if len(shape) == 1:
        return tensor_slice[: min(sample_cols, shape[0])]
    slices = [slice(0, min(sample_rows, shape[0])), slice(0, min(sample_cols, shape[1]))]
    slices.extend(slice(0, 1) for _ in shape[2:])
    return tensor_slice[tuple(slices)]


def _sample_tensor(model_dir: Path, tensor_name: str, shard_name: str, sample_rows: int, sample_cols: int) -> TensorSentinel:
    from safetensors import safe_open

    shard_path = model_dir / shard_name
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(tensor_name)
        shape = tuple(int(dim) for dim in tensor_slice.get_shape())
        dtype = str(handle.get_tensor(tensor_name).dtype) if len(shape) == 0 else str(_slice_prefix(tensor_slice, 1, 1).dtype)
        sample = _slice_prefix(tensor_slice, sample_rows, sample_cols).detach().cpu().float()

    flat = sample.reshape(-1)
    finite = flat.isfinite()
    finite_ratio = float(finite.float().mean().item()) if flat.numel() else 0.0
    nonzero_ratio = float((flat != 0).float().mean().item()) if flat.numel() else 0.0
    abs_mean = float(flat.abs().mean().item()) if flat.numel() else 0.0
    std = float(flat.std(unbiased=False).item()) if flat.numel() else 0.0
    digest = hashlib.sha256(sample.numpy().tobytes()).hexdigest()
    return TensorSentinel(
        name=tensor_name,
        shard=shard_name,
        dtype=dtype,
        shape=shape,
        sample_shape=tuple(int(dim) for dim in sample.shape),
        finite_ratio=finite_ratio,
        nonzero_ratio=nonzero_ratio,
        abs_mean=abs_mean,
        std=std,
        sample_sha256=digest,
    )


def run_t005_real_weight_sentinel(
    *,
    model_id: str = "Qwen/Qwen3-32B",
    local_model_dir: Path | None = None,
    allow_hf_config_fetch: bool = True,
    tensor_names: Sequence[str] | None = None,
    sample_rows: int = 8,
    sample_cols: int = 8,
    min_required_tensors: int = 4,
) -> dict[str, Any]:
    """Read tiny real tensor prefixes and validate they are usable sentinel data."""

    config = resolve_qwen_config(
        model_id=model_id,
        local_model_dir=local_model_dir,
        allow_hf_config_fetch=allow_hf_config_fetch,
    )
    weight_dir = find_weight_dir(config, local_model_dir)
    inventory = inventory_safetensors(weight_dir)
    requested = tuple(tensor_names or DEFAULT_SENTINEL_TENSORS)
    blockers: list[str] = []
    sentinels: list[TensorSentinel] = []
    missing: list[str] = []

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
                sentinels.append(_sample_tensor(weight_dir, name, shard, sample_rows, sample_cols))
            except Exception as exc:  # pragma: no cover - real shard guard
                blockers.append(f"sample_failed:{name}:{type(exc).__name__}:{exc}")
                break

    pass_count = sum(1 for item in sentinels if item.passes)
    sentinel_pass = pass_count >= min_required_tensors and len(sentinels) >= min_required_tensors and not blockers
    if not sentinel_pass and not blockers:
        blockers.append("insufficient_passing_real_tensor_sentinels")

    return {
        "probe": "t005_real_weight_sentinel",
        "activation": ACTIVATION,
        "model_id": model_id,
        "model_dir": str(weight_dir) if weight_dir is not None else None,
        "config": config.as_dict(),
        "weight_inventory": inventory.as_dict(),
        "requested_tensors": list(requested),
        "sentinels": [item.as_dict() for item in sentinels],
        "missing_tensors": missing,
        "readiness": {
            "sentinel_pass": sentinel_pass,
            "sampled_tensors": len(sentinels),
            "passing_tensors": pass_count,
            "min_required_tensors": min_required_tensors,
            "sample_rows": sample_rows,
            "sample_cols": sample_cols,
            "config_matches_project_qwen3_32b_shape": config.matches_project_qwen3_32b_shape(),
            "local_weights_present": inventory.local_weights_present,
        },
        "blockers": blockers,
        "next_actions": [
            "Use these tensor-level sentinels as the real-weight precondition for captured residual/KV residency gates."
        ]
        if sentinel_pass
        else ["Fix the listed blockers before treating real Qwen weights as project-runtime inputs."],
    }
