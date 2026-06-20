"""T004 bridge from synthetic receipts to real Qwen runtime readiness."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import struct
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .footprint import BYTES_PER_GIB, TransformerShape, qwen3_32b_shape
from .t003_ram_efficiency import run_t003_receipt_kv_gate


ACTIVATION = "T004_REAL_QWEN_BRIDGE_ACTIVE"


@dataclass(frozen=True)
class QwenConfigSnapshot:
    """Compact Qwen config snapshot used to compare real metadata to T001-T003."""

    model_id: str
    source: str
    source_path: str
    model_type: str
    architectures: tuple[str, ...]
    layers: int
    hidden_size: int
    intermediate_size: int
    attention_heads: int
    kv_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    config_sha256: str

    @property
    def output_head_included(self) -> bool:
        return not self.tie_word_embeddings

    def to_shape(self) -> TransformerShape:
        return TransformerShape(
            name=f"{self.model_id} config snapshot",
            layers=self.layers,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            attention_heads=self.attention_heads,
            kv_heads=self.kv_heads,
            configured_head_dim=self.head_dim,
            vocab_size=self.vocab_size,
            include_output_head=self.output_head_included,
        )

    def matches_project_qwen3_32b_shape(self) -> bool:
        project_shape = qwen3_32b_shape()
        current_shape = self.to_shape()
        return (
            self.model_type == "qwen3"
            and current_shape.layers == project_shape.layers
            and current_shape.hidden_size == project_shape.hidden_size
            and current_shape.intermediate_size == project_shape.intermediate_size
            and current_shape.attention_heads == project_shape.attention_heads
            and current_shape.kv_heads == project_shape.kv_heads
            and current_shape.head_dim == project_shape.head_dim
            and current_shape.vocab_size == project_shape.vocab_size
            and self.output_head_included
        )

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["architectures"] = list(self.architectures)
        data["output_head_included"] = self.output_head_included
        data["matches_project_qwen3_32b_shape"] = self.matches_project_qwen3_32b_shape()
        data["estimated_weight_elements"] = self.to_shape().estimated_weight_elements
        data["estimated_weight_billions"] = self.to_shape().estimated_weight_elements / 1e9
        return data


@dataclass(frozen=True)
class SafetensorsInventory:
    """Header-only inventory for local safetensors shards."""

    model_dir: str | None
    shard_count: int
    total_file_gib: float
    tensor_count: int
    tensor_payload_gib: float
    largest_tensor_name: str | None
    largest_tensor_gib: float
    has_weight_index: bool
    sample_shards: tuple[str, ...]
    error: str | None = None

    @property
    def local_weights_present(self) -> bool:
        return self.shard_count > 0 and self.tensor_count > 0 and self.error is None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sample_shards"] = list(self.sample_shards)
        data["local_weights_present"] = self.local_weights_present
        return data


def _package_presence() -> dict[str, bool]:
    packages = ["torch", "transformers", "safetensors", "huggingface_hub", "accelerate", "bitsandbytes"]
    return {name: importlib.util.find_spec(name) is not None for name in packages}


def _torch_cuda_snapshot() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"torch_available": False, "cuda_available": False}
    try:
        import torch

        data: dict[str, Any] = {
            "torch_available": True,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            data.update(
                {
                    "device_name": props.name,
                    "total_memory_gib": props.total_memory / BYTES_PER_GIB,
                    "allocated_gib": torch.cuda.memory_allocated(0) / BYTES_PER_GIB,
                    "reserved_gib": torch.cuda.memory_reserved(0) / BYTES_PER_GIB,
                }
            )
        return data
    except Exception as exc:  # pragma: no cover - environment-dependent guard
        return {"torch_available": True, "cuda_available": False, "error": repr(exc)}


def _nvidia_smi_snapshot() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False}
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - environment-dependent guard
        return {"available": False, "error": repr(exc)}
    first = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    parts = [part.strip() for part in first.split(",")]
    if len(parts) != 4:
        return {"available": True, "raw": first}
    return {
        "available": True,
        "device_name": parts[0],
        "memory_total_mib": int(float(parts[1])),
        "memory_free_mib": int(float(parts[2])),
        "driver_version": parts[3],
        "memory_total_gib": int(float(parts[1])) / 1024.0,
        "memory_free_gib": int(float(parts[2])) / 1024.0,
    }


def _disk_free_gib(path: Path) -> float | None:
    try:
        target = path if path.exists() else path.parent
        return shutil.disk_usage(target).free / BYTES_PER_GIB
    except Exception:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_config_snapshot(path: Path, *, model_id: str, source: str) -> QwenConfigSnapshot:
    payload = json.loads(path.read_text(encoding="utf-8"))
    head_dim = int(payload.get("head_dim") or (int(payload["hidden_size"]) // int(payload["num_attention_heads"])))
    return QwenConfigSnapshot(
        model_id=model_id,
        source=source,
        source_path=str(path),
        model_type=str(payload.get("model_type", "")),
        architectures=tuple(str(item) for item in payload.get("architectures", [])),
        layers=int(payload["num_hidden_layers"]),
        hidden_size=int(payload["hidden_size"]),
        intermediate_size=int(payload["intermediate_size"]),
        attention_heads=int(payload["num_attention_heads"]),
        kv_heads=int(payload["num_key_value_heads"]),
        head_dim=head_dim,
        vocab_size=int(payload["vocab_size"]),
        max_position_embeddings=int(payload.get("max_position_embeddings", 0)),
        tie_word_embeddings=bool(payload.get("tie_word_embeddings", False)),
        config_sha256=_sha256_file(path),
    )


def _hf_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for value in (os.environ.get("HF_HOME"), os.environ.get("HUGGINGFACE_HUB_CACHE")):
        if value:
            path = Path(value)
            roots.append(path / "hub" if path.name != "hub" else path)
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return list(dict.fromkeys(roots))


def _candidate_model_dirs(model_id: str, local_model_dir: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if local_model_dir is not None:
        candidates.append(local_model_dir)

    cache_name = "models--" + model_id.replace("/", "--")
    for root in _hf_cache_roots():
        repo_root = root / cache_name
        snapshots = repo_root / "snapshots"
        if snapshots.exists():
            candidates.extend(sorted((p for p in snapshots.iterdir() if p.is_dir()), reverse=True))
        if repo_root.exists():
            candidates.append(repo_root)
    return list(dict.fromkeys(candidates))


def resolve_qwen_config(
    *,
    model_id: str = "Qwen/Qwen3-32B",
    local_model_dir: Path | None = None,
    allow_hf_config_fetch: bool = True,
) -> QwenConfigSnapshot:
    for candidate in _candidate_model_dirs(model_id, local_model_dir):
        config_path = candidate / "config.json"
        if config_path.exists():
            source = "local_model_dir" if local_model_dir is not None and candidate == local_model_dir else "hf_cache"
            return _read_config_snapshot(config_path, model_id=model_id, source=source)

    if allow_hf_config_fetch and importlib.util.find_spec("huggingface_hub") is not None:
        from huggingface_hub import hf_hub_download

        config_path = Path(hf_hub_download(model_id, "config.json"))
        return _read_config_snapshot(config_path, model_id=model_id, source="hf_hub_config_only")

    shape = qwen3_32b_shape()
    return QwenConfigSnapshot(
        model_id=model_id,
        source="project_static_shape",
        source_path="src.transformer_residency.footprint.qwen3_32b_shape",
        model_type="qwen3",
        architectures=("Qwen3ForCausalLM",),
        layers=shape.layers,
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        attention_heads=shape.attention_heads,
        kv_heads=shape.kv_heads,
        head_dim=shape.head_dim,
        vocab_size=shape.vocab_size,
        max_position_embeddings=0,
        tie_word_embeddings=False,
        config_sha256="project-static",
    )


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path} is too small to be a safetensors file")
        header_len = struct.unpack("<Q", prefix)[0]
        header = handle.read(header_len)
    payload = json.loads(header.decode("utf-8"))
    tensors = {name: value for name, value in payload.items() if name != "__metadata__"}
    return tensors


def inventory_safetensors(model_dir: Path | None) -> SafetensorsInventory:
    if model_dir is None:
        return SafetensorsInventory(
            model_dir=None,
            shard_count=0,
            total_file_gib=0.0,
            tensor_count=0,
            tensor_payload_gib=0.0,
            largest_tensor_name=None,
            largest_tensor_gib=0.0,
            has_weight_index=False,
            sample_shards=(),
            error="no_model_dir",
        )
    if not model_dir.exists():
        return SafetensorsInventory(
            model_dir=str(model_dir),
            shard_count=0,
            total_file_gib=0.0,
            tensor_count=0,
            tensor_payload_gib=0.0,
            largest_tensor_name=None,
            largest_tensor_gib=0.0,
            has_weight_index=False,
            sample_shards=(),
            error="model_dir_missing",
        )

    shards = sorted(model_dir.glob("*.safetensors"))
    total_file_bytes = sum(path.stat().st_size for path in shards)
    tensor_count = 0
    tensor_payload_bytes = 0
    largest_tensor_name: str | None = None
    largest_tensor_bytes = 0
    try:
        for shard in shards:
            for name, item in _read_safetensors_header(shard).items():
                offsets = item.get("data_offsets", [0, 0])
                size = int(offsets[1]) - int(offsets[0])
                tensor_count += 1
                tensor_payload_bytes += size
                if size > largest_tensor_bytes:
                    largest_tensor_bytes = size
                    largest_tensor_name = name
    except Exception as exc:
        return SafetensorsInventory(
            model_dir=str(model_dir),
            shard_count=len(shards),
            total_file_gib=total_file_bytes / BYTES_PER_GIB,
            tensor_count=tensor_count,
            tensor_payload_gib=tensor_payload_bytes / BYTES_PER_GIB,
            largest_tensor_name=largest_tensor_name,
            largest_tensor_gib=largest_tensor_bytes / BYTES_PER_GIB,
            has_weight_index=(model_dir / "model.safetensors.index.json").exists(),
            sample_shards=tuple(path.name for path in shards[:5]),
            error=repr(exc),
        )

    return SafetensorsInventory(
        model_dir=str(model_dir),
        shard_count=len(shards),
        total_file_gib=total_file_bytes / BYTES_PER_GIB,
        tensor_count=tensor_count,
        tensor_payload_gib=tensor_payload_bytes / BYTES_PER_GIB,
        largest_tensor_name=largest_tensor_name,
        largest_tensor_gib=largest_tensor_bytes / BYTES_PER_GIB,
        has_weight_index=(model_dir / "model.safetensors.index.json").exists(),
        sample_shards=tuple(path.name for path in shards[:5]),
    )


def find_weight_dir(config: QwenConfigSnapshot, local_model_dir: Path | None) -> Path | None:
    if local_model_dir is not None:
        return local_model_dir
    config_path = Path(config.source_path)
    if config_path.name == "config.json" and list(config_path.parent.glob("*.safetensors")):
        return config_path.parent
    for candidate in _candidate_model_dirs(config.model_id, local_model_dir):
        if list(candidate.glob("*.safetensors")):
            return candidate
    if config_path.name == "config.json" and config_path.parent.exists():
        return config_path.parent
    return local_model_dir


def run_t004_real_qwen_bridge(
    *,
    model_id: str = "Qwen/Qwen3-32B",
    local_model_dir: Path | None = None,
    allow_hf_config_fetch: bool = True,
    denominator_gib: float = 12.0,
) -> dict[str, Any]:
    """Run the real Qwen bridge preflight without loading full model weights."""

    config = resolve_qwen_config(
        model_id=model_id,
        local_model_dir=local_model_dir,
        allow_hf_config_fetch=allow_hf_config_fetch,
    )
    weight_dir = find_weight_dir(config, local_model_dir)
    inventory = inventory_safetensors(weight_dir)
    t003 = run_t003_receipt_kv_gate()
    packages = _package_presence()
    torch_cuda = _torch_cuda_snapshot()
    smi = _nvidia_smi_snapshot()
    disk_path = local_model_dir or (Path(config.source_path).parent if Path(config.source_path).exists() else Path.cwd())

    config_matches = config.matches_project_qwen3_32b_shape()
    local_weights_present = inventory.local_weights_present
    can_inspect_real_weights = local_weights_present and inventory.tensor_count > 0
    can_attempt_real_runtime = (
        bool(torch_cuda.get("cuda_available"))
        and bool(packages["transformers"])
        and bool(packages["safetensors"])
        and local_weights_present
        and config_matches
    )
    generation_ready = False

    blockers: list[str] = []
    if not config_matches:
        blockers.append("config_shape_mismatch")
    if not local_weights_present:
        blockers.append("missing_local_qwen3_32b_safetensors")
    if not bool(torch_cuda.get("cuda_available")):
        blockers.append("cuda_not_available")
    if not packages["transformers"]:
        blockers.append("transformers_missing")
    if local_weights_present and not generation_ready:
        blockers.append("full_compressed_generation_runtime_not_implemented")

    next_actions: list[str] = []
    if "missing_local_qwen3_32b_safetensors" in blockers:
        next_actions.append(
            "Download or point --local-model-dir at Qwen/Qwen3-32B safetensors before captured residual or generation gates."
        )
    if can_inspect_real_weights:
        next_actions.append("Run the captured residual/KV sentinel gate against local safetensors headers and selected tensors.")
    if "full_compressed_generation_runtime_not_implemented" in blockers:
        next_actions.append("Implement the opt-in residency runtime before claiming real generation tok/s.")

    return {
        "probe": "t004_real_qwen_bridge",
        "activation": ACTIVATION,
        "model_id": model_id,
        "config": config.as_dict(),
        "environment": {
            "packages": packages,
            "torch_cuda": torch_cuda,
            "nvidia_smi": smi,
            "disk_free_gib_near_model": _disk_free_gib(disk_path),
        },
        "weight_inventory": inventory.as_dict(),
        "t003_reference": {
            "source": "run_t003_receipt_kv_gate",
            "total_without_safety_gib": t003["residency_receipt"]["total_without_safety_gib"],
            "total_with_safety_gib": t003["residency_receipt"]["total_with_safety_gib"],
            "slack_with_safety_gib": t003["residency_receipt"]["slack_with_safety_gib"],
            "tok_s_proxy": t003["tok_s_predictor"]["t003"]["tokens_per_second"],
        },
        "readiness": {
            "bridge_pass": True,
            "config_matches_project_qwen3_32b_shape": config_matches,
            "local_weights_present": local_weights_present,
            "can_inspect_real_weights": can_inspect_real_weights,
            "can_attempt_real_runtime": can_attempt_real_runtime,
            "generation_ready": generation_ready,
            "denominator_gib": denominator_gib,
        },
        "blockers": blockers,
        "next_actions": next_actions,
    }
