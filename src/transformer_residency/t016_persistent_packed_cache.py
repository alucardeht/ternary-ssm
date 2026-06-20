"""T016 persistent compressed cache before fused kernels."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open

from .footprint import BYTES_PER_GIB
from .t004_real_qwen_bridge import find_weight_dir, inventory_safetensors, resolve_qwen_config
from .t006_real_residency_block_smoke import BlockCodecPolicy, _policy_for_tensor
from .t008_compressed_generation_smoke import _load_weight_map
from .t014_denominator_tightening import _policy_for_target_mlp_bits
from .t015_packed_qwen_inference import (
    _FallbackTokenizer,
    _apply_rope,
    _cuda_memory,
    _read_rows,
    _read_tensor,
    _rms_norm,
    _select_device,
    _silu,
    _t014_a_denominator,
    _tokenizer,
)


ACTIVATION = "T016_PERSISTENT_PACKED_CACHE_ACTIVE"


@dataclass
class PackedCacheStats:
    cache_dir: str
    tensors_built: int = 0
    tensors_loaded: int = 0
    resident_hits: int = 0
    resident_stores: int = 0
    resident_bytes: int = 0
    device_resident_hits: int = 0
    device_resident_stores: int = 0
    device_resident_skips: int = 0
    device_resident_failures: int = 0
    device_resident_bytes: int = 0
    device_arena_hits: int = 0
    device_arena_misses: int = 0
    binary_payloads_loaded: int = 0
    binary_payloads_written: int = 0
    binary_cache_bytes_read: int = 0
    binary_cache_bytes_written: int = 0
    trit_payloads: int = 0
    trit_payloads_loaded: int = 0
    trit_payloads_written: int = 0
    trit_cache_bytes_read: int = 0
    trit_cache_bytes_written: int = 0
    trit_code_bytes: int = 0
    unpacked_code_bytes_avoided: int = 0
    cache_bytes_read: int = 0
    cache_bytes_written: int = 0
    pack_seconds: float = 0.0
    load_seconds: float = 0.0
    binary_load_seconds: float = 0.0
    binary_write_seconds: float = 0.0
    reconstruct_seconds: float = 0.0

    @property
    def cache_hit_count(self) -> int:
        return self.tensors_loaded

    @property
    def cache_build_count(self) -> int:
        return self.tensors_built

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cache_hit_count"] = self.cache_hit_count
        data["cache_build_count"] = self.cache_build_count
        data["resident_gib"] = self.resident_bytes / BYTES_PER_GIB
        data["device_resident_gib"] = self.device_resident_bytes / BYTES_PER_GIB
        data["binary_cache_gib_read"] = self.binary_cache_bytes_read / BYTES_PER_GIB
        data["binary_cache_gib_written"] = self.binary_cache_bytes_written / BYTES_PER_GIB
        data["trit_cache_gib_read"] = self.trit_cache_bytes_read / BYTES_PER_GIB
        data["trit_cache_gib_written"] = self.trit_cache_bytes_written / BYTES_PER_GIB
        data["trit_code_gib"] = self.trit_code_bytes / BYTES_PER_GIB
        data["unpacked_code_gib_avoided"] = self.unpacked_code_bytes_avoided / BYTES_PER_GIB
        data["cache_gib_read"] = self.cache_bytes_read / BYTES_PER_GIB
        data["cache_gib_written"] = self.cache_bytes_written / BYTES_PER_GIB
        return data


@dataclass(frozen=True)
class PersistentCacheInferenceReceipt:
    prompt_text: str
    tokenizer_source: str
    prompt_token_count: int
    generated_token_ids: tuple[int, ...]
    generated_text: str
    requested_layers: int
    completed_layers: int
    device: str
    elapsed_seconds: float
    tokens_per_second: float
    peak_cuda_allocated_gib: float | None
    peak_cuda_reserved_gib: float | None
    denominator_total_gib: float
    denominator_slack_gib: float
    cache_stats: dict[str, Any]
    output_sha256: str

    @property
    def passes(self) -> bool:
        return (
            self.completed_layers == self.requested_layers
            and self.requested_layers > 0
            and len(self.generated_token_ids) > 0
            and bool(self.generated_text)
            and self.denominator_total_gib <= 12.0
            and self.elapsed_seconds > 0.0
        )

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["generated_token_ids"] = list(self.generated_token_ids)
        data["passes"] = self.passes
        return data


def _safe_cache_name(name: str, *, start: int | None = None, stop: int | None = None, policy_name: str = "") -> str:
    payload = json.dumps({"name": name, "start": start, "stop": stop, "policy": policy_name}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24] + ".pt"


def _payload_cache_key(
    model_dir: Path,
    weight_map: dict[str, str],
    name: str,
    *,
    target_mlp_bits: float,
    start: int | None = None,
    stop: int | None = None,
) -> str:
    cols = int(_tensor_shape(model_dir, weight_map, name)[1])
    policy = _policy_for_name(name, cols, target_mlp_bits)
    return _safe_cache_name(name, start=start, stop=stop, policy_name=policy.name)


def _payload_nbytes(payload: dict[str, Any]) -> int:
    total = 0
    for value in payload.values():
        if hasattr(value, "numel") and hasattr(value, "element_size"):
            total += int(value.numel()) * int(value.element_size())
    return total


def _pack_trit_codes(codes: Any, *, group_size: int = 4) -> Any:
    """Pack ternary {-1, 0, 1} codes into exact base-3 groups."""

    import torch

    if group_size not in (4, 5, 20):
        raise ValueError("only group_size=4, group_size=5, or group_size=20 is supported")
    code_cpu = codes.to(device="cpu", dtype=torch.int16).contiguous()
    rows, cols = [int(dim) for dim in code_cpu.shape]
    groups = math.ceil(cols / group_size)
    padded = torch.ones((rows, groups * group_size), dtype=torch.int16)
    padded[:, :cols] = code_cpu + 1
    packed_dtype = torch.uint8 if group_size <= 5 else torch.uint32
    packed = torch.zeros((rows, groups), dtype=packed_dtype)
    powers = torch.tensor([3**index for index in range(group_size)], dtype=torch.int64)
    view = padded.reshape(rows, groups, group_size)
    packed[:, :] = torch.sum(view.to(torch.int64) * powers.view(1, 1, group_size), dim=2).to(packed_dtype)
    return packed


def _payload_with_trit_codes(
    payload: dict[str, Any],
    *,
    drop_unpacked: bool = True,
    group_size: int = 4,
) -> dict[str, Any]:
    """Return a payload view that carries radix-3 packed ternary codes."""

    if (
        "trit_codes" in payload
        and int(payload.get("trit_group_size", 4)) == group_size
        and (not drop_unpacked or "codes" not in payload)
    ):
        return payload
    if "codes" not in payload:
        raise KeyError("codes")
    packed = dict(payload)
    packed["trit_codes"] = _pack_trit_codes(payload["codes"], group_size=group_size)
    packed["trit_group_size"] = group_size
    packed["unpacked_code_bytes"] = int(payload["codes"].numel()) * int(payload["codes"].element_size())
    packed["trit_code_bytes"] = int(packed["trit_codes"].numel()) * int(packed["trit_codes"].element_size())
    if drop_unpacked:
        del packed["codes"]
    return packed


def _dtype_name(tensor: Any) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> Any:
    import torch

    return getattr(torch, name)


def _binary_payload_base(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / "binary-payloads" / cache_key.removesuffix(".pt")


def _binary_payload_meta_path(cache_dir: Path, cache_key: str) -> Path:
    return _binary_payload_base(cache_dir, cache_key).with_suffix(".json")


def _write_binary_payload(cache_dir: Path, cache_key: str, payload: dict[str, Any], stats: PackedCacheStats) -> None:
    started = time.perf_counter()
    base = _binary_payload_base(cache_dir, cache_key)
    base.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, Any] = {}
    bytes_written = 0
    for name in ("codes", "scales", "repair_indices", "repair_values"):
        tensor = payload[name].contiguous().cpu()
        suffix = _dtype_name(tensor)
        path = base.with_suffix(f".{name}.{suffix}.bin")
        tensor.numpy().tofile(path)
        bytes_written += path.stat().st_size
        tensors[name] = {
            "file": path.name,
            "dtype": suffix,
            "shape": [int(dim) for dim in tensor.shape],
            "bytes": path.stat().st_size,
        }
    meta = {
        "shape": [int(dim) for dim in payload["shape"]],
        "policy": payload["policy"],
        "tensors": tensors,
    }
    meta_path = _binary_payload_meta_path(cache_dir, cache_key)
    meta_path.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
    bytes_written += meta_path.stat().st_size
    stats.binary_payloads_written += 1
    stats.binary_cache_bytes_written += bytes_written
    stats.binary_write_seconds += time.perf_counter() - started


def _read_binary_payload(cache_dir: Path, cache_key: str, stats: PackedCacheStats) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    meta_path = _binary_payload_meta_path(cache_dir, cache_key)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "shape": tuple(int(dim) for dim in meta["shape"]),
        "policy": meta["policy"],
    }
    bytes_read = meta_path.stat().st_size
    for name, spec in meta["tensors"].items():
        shape = tuple(int(dim) for dim in spec["shape"])
        size = math.prod(shape)
        path = meta_path.parent / spec["file"]
        payload[name] = torch.from_file(str(path), shared=False, size=size, dtype=_dtype_from_name(spec["dtype"])).reshape(shape)
        bytes_read += int(spec["bytes"])
    stats.binary_payloads_loaded += 1
    stats.binary_cache_bytes_read += bytes_read
    stats.binary_load_seconds += time.perf_counter() - started
    return payload


def _read_weight(
    model_dir: Path,
    weight_map: dict[str, str],
    name: str,
    *,
    start: int | None = None,
    stop: int | None = None,
) -> Any:
    shard = weight_map[name]
    with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
        if start is None:
            return handle.get_tensor(name).detach().float()
        tensor_slice = handle.get_slice(name)
        return tensor_slice[start:stop, :].detach().float()


def _tensor_shape(model_dir: Path, weight_map: dict[str, str], name: str) -> tuple[int, ...]:
    shard = weight_map[name]
    with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
        return tuple(int(dim) for dim in handle.get_slice(name).get_shape())


def _pack_tensor(exact: Any, policy: BlockCodecPolicy) -> dict[str, Any]:
    import torch

    cols = int(exact.shape[1])
    scales = exact.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
    threshold = scales * policy.threshold_multiplier
    codes = torch.where(exact.abs() >= threshold, exact.sign(), torch.zeros_like(exact)).to(torch.int8)
    base = codes.float() * scales
    residual = exact - base
    k = max(1, int(math.ceil(cols * policy.repair_fraction)))
    _, indices = torch.topk(residual.abs(), k=k, dim=1)
    values = residual.gather(1, indices)
    index_dtype = torch.int16 if cols <= 32767 else torch.int32
    return {
        "shape": tuple(int(dim) for dim in exact.shape),
        "policy": policy.as_dict(cols),
        "codes": codes.cpu(),
        "scales": scales.squeeze(1).half().cpu(),
        "repair_indices": indices.to(index_dtype).cpu(),
        "repair_values": values.half().cpu(),
    }


def _reconstruct_packed(payload: dict[str, Any], *, device: str) -> Any:
    import torch

    start = time.perf_counter()
    codes = payload["codes"].to(device=device, dtype=torch.float32)
    scales = payload["scales"].to(device=device, dtype=torch.float32).unsqueeze(1)
    dense = codes * scales
    indices = payload["repair_indices"].to(device=device, dtype=torch.long)
    values = payload["repair_values"].to(device=device, dtype=torch.float32)
    dense.scatter_add_(1, indices, values)
    return dense, time.perf_counter() - start


def _policy_for_name(name: str, cols: int, target_mlp_bits: float) -> BlockCodecPolicy:
    if ".mlp." in name:
        return _policy_for_target_mlp_bits(name, cols, target_mlp_bits)
    if name == "lm_head.weight":
        return BlockCodecPolicy(name="lm_head_persistent_q4_boundary", target_bits_per_weight=4.0, repair_fraction=0.08)
    return _policy_for_tensor(name)


def _cached_weight(
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    cache_dir: Path,
    stats: PackedCacheStats,
    name: str,
    device: str,
    target_mlp_bits: float,
    start: int | None = None,
    stop: int | None = None,
    refresh_cache: bool = False,
) -> Any:
    payload = _cached_payload(
        model_dir=model_dir,
        weight_map=weight_map,
        cache_dir=cache_dir,
        stats=stats,
        name=name,
        target_mlp_bits=target_mlp_bits,
        start=start,
        stop=stop,
        refresh_cache=refresh_cache,
    )
    dense, reconstruct_seconds = _reconstruct_packed(payload, device=device)
    stats.reconstruct_seconds += reconstruct_seconds
    return dense


def _cached_payload(
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    cache_dir: Path,
    stats: PackedCacheStats,
    name: str,
    target_mlp_bits: float,
    start: int | None = None,
    stop: int | None = None,
    refresh_cache: bool = False,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cols = int(_tensor_shape(model_dir, weight_map, name)[1])
    policy = _policy_for_name(name, cols, target_mlp_bits)
    cache_path = cache_dir / _safe_cache_name(name, start=start, stop=stop, policy_name=policy.name)
    if cache_path.exists() and not refresh_cache:
        started = time.perf_counter()
        payload = __import__("torch").load(cache_path, map_location="cpu")
        stats.load_seconds += time.perf_counter() - started
        stats.tensors_loaded += 1
        stats.cache_bytes_read += cache_path.stat().st_size
    else:
        exact = _read_weight(model_dir, weight_map, name, start=start, stop=stop)
        started = time.perf_counter()
        payload = _pack_tensor(exact, policy)
        stats.pack_seconds += time.perf_counter() - started
        stats.tensors_built += 1
        __import__("torch").save(payload, cache_path)
        stats.cache_bytes_written += cache_path.stat().st_size
        del exact
    return payload


def _cached_binary_payload(
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    cache_dir: Path,
    stats: PackedCacheStats,
    name: str,
    target_mlp_bits: float,
    start: int | None = None,
    stop: int | None = None,
    refresh_cache: bool = False,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _payload_cache_key(
        model_dir,
        weight_map,
        name,
        target_mlp_bits=target_mlp_bits,
        start=start,
        stop=stop,
    )
    if _binary_payload_meta_path(cache_dir, cache_key).exists() and not refresh_cache:
        return _read_binary_payload(cache_dir, cache_key, stats)

    payload = _cached_payload(
        model_dir=model_dir,
        weight_map=weight_map,
        cache_dir=cache_dir,
        stats=stats,
        name=name,
        target_mlp_bits=target_mlp_bits,
        start=start,
        stop=stop,
        refresh_cache=refresh_cache,
    )
    _write_binary_payload(cache_dir, cache_key, payload, stats)
    return payload


def _attention_forward_cached(
    x: Any,
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    cache_dir: Path,
    stats: PackedCacheStats,
    layer_index: int,
    input_norm: Any,
    q_norm: Any,
    k_norm: Any,
    device: str,
    rms_eps: float,
    rope_theta: float,
    target_mlp_bits: float,
    refresh_cache: bool,
) -> Any:
    import torch

    prefix = f"model.layers.{layer_index}.self_attn"
    h = _rms_norm(x, input_norm, rms_eps)
    q_weight = _cached_weight(
        model_dir=model_dir,
        weight_map=weight_map,
        cache_dir=cache_dir,
        stats=stats,
        name=f"{prefix}.q_proj.weight",
        device=device,
        target_mlp_bits=target_mlp_bits,
        refresh_cache=refresh_cache,
    )
    q = h @ q_weight.T
    del q_weight
    k_weight = _cached_weight(
        model_dir=model_dir,
        weight_map=weight_map,
        cache_dir=cache_dir,
        stats=stats,
        name=f"{prefix}.k_proj.weight",
        device=device,
        target_mlp_bits=target_mlp_bits,
        refresh_cache=refresh_cache,
    )
    k = h @ k_weight.T
    del k_weight
    v_weight = _cached_weight(
        model_dir=model_dir,
        weight_map=weight_map,
        cache_dir=cache_dir,
        stats=stats,
        name=f"{prefix}.v_proj.weight",
        device=device,
        target_mlp_bits=target_mlp_bits,
        refresh_cache=refresh_cache,
    )
    v = h @ v_weight.T
    del v_weight

    head_dim = int(q_norm.shape[0])
    q_heads = int(q.shape[-1] // head_dim)
    kv_heads = int(k.shape[-1] // head_dim)
    seq_len = int(q.shape[0])
    q = q.reshape(seq_len, q_heads, head_dim)
    k = k.reshape(seq_len, kv_heads, head_dim)
    v = v.reshape(seq_len, kv_heads, head_dim)
    q = _rms_norm(q, q_norm, rms_eps)
    k = _rms_norm(k, k_norm, rms_eps)
    positions = torch.arange(seq_len, device=x.device)
    q = _apply_rope(q, theta=rope_theta, positions=positions)
    k = _apply_rope(k, theta=rope_theta, positions=positions)
    repeat = max(1, q_heads // kv_heads)
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)

    scores = torch.einsum("shd,thd->hst", q, k) / math.sqrt(head_dim)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    context = torch.einsum("hst,thd->shd", attn, v).reshape(seq_len, q_heads * head_dim)

    o_weight = _cached_weight(
        model_dir=model_dir,
        weight_map=weight_map,
        cache_dir=cache_dir,
        stats=stats,
        name=f"{prefix}.o_proj.weight",
        device=device,
        target_mlp_bits=target_mlp_bits,
        refresh_cache=refresh_cache,
    )
    out = context @ o_weight.T
    del o_weight
    return x + out


def _mlp_forward_cached(
    x: Any,
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    cache_dir: Path,
    stats: PackedCacheStats,
    layer_index: int,
    post_norm: Any,
    device: str,
    rms_eps: float,
    target_mlp_bits: float,
    refresh_cache: bool,
) -> Any:
    prefix = f"model.layers.{layer_index}.mlp"
    h = _rms_norm(x, post_norm, rms_eps)
    gate_weight = _cached_weight(
        model_dir=model_dir,
        weight_map=weight_map,
        cache_dir=cache_dir,
        stats=stats,
        name=f"{prefix}.gate_proj.weight",
        device=device,
        target_mlp_bits=target_mlp_bits,
        refresh_cache=refresh_cache,
    )
    gate = _silu(h @ gate_weight.T)
    del gate_weight
    up_weight = _cached_weight(
        model_dir=model_dir,
        weight_map=weight_map,
        cache_dir=cache_dir,
        stats=stats,
        name=f"{prefix}.up_proj.weight",
        device=device,
        target_mlp_bits=target_mlp_bits,
        refresh_cache=refresh_cache,
    )
    up = h @ up_weight.T
    del up_weight
    hidden = gate * up
    del gate, up
    down_weight = _cached_weight(
        model_dir=model_dir,
        weight_map=weight_map,
        cache_dir=cache_dir,
        stats=stats,
        name=f"{prefix}.down_proj.weight",
        device=device,
        target_mlp_bits=target_mlp_bits,
        refresh_cache=refresh_cache,
    )
    out = hidden @ down_weight.T
    del down_weight, hidden
    return x + out


def _lm_head_argmax_cached(
    x_last: Any,
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    cache_dir: Path,
    stats: PackedCacheStats,
    device: str,
    chunk_rows: int,
    target_mlp_bits: float,
    refresh_cache: bool,
) -> int:
    import torch

    best_value: float | None = None
    best_index = 0
    with safe_open(model_dir / weight_map["lm_head.weight"], framework="pt", device="cpu") as handle:
        rows = int(handle.get_slice("lm_head.weight").get_shape()[0])
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        weight = _cached_weight(
            model_dir=model_dir,
            weight_map=weight_map,
            cache_dir=cache_dir,
            stats=stats,
            name="lm_head.weight",
            device=device,
            target_mlp_bits=target_mlp_bits,
            start=start,
            stop=stop,
            refresh_cache=refresh_cache,
        )
        logits = x_last @ weight.T
        value, index = torch.max(logits, dim=-1)
        if best_value is None or float(value.item()) > best_value:
            best_value = float(value.item())
            best_index = start + int(index.item())
        del weight, logits
    return best_index


def _default_cache_dir(model_dir: Path) -> Path:
    digest = hashlib.sha256(str(model_dir).encode("utf-8")).hexdigest()[:12]
    return Path(".work") / "cache" / "t016-persistent-packed-qwen" / digest


def run_t016_persistent_packed_cache(
    *,
    model_id: str = "Qwen/Qwen3-32B",
    local_model_dir: Path | None = None,
    allow_hf_config_fetch: bool = True,
    prompt_text: str = "Local compressed Qwen smoke:",
    max_new_tokens: int = 1,
    layer_limit: int | None = None,
    device: str = "auto",
    target_mlp_bits: float = 2.45,
    lm_head_chunk_rows: int = 2048,
    cache_dir: Path | None = None,
    refresh_cache: bool = False,
) -> dict[str, Any]:
    """Run token generation through a persistent compressed cache."""

    config = resolve_qwen_config(
        model_id=model_id,
        local_model_dir=local_model_dir,
        allow_hf_config_fetch=allow_hf_config_fetch,
    )
    weight_dir = find_weight_dir(config, local_model_dir)
    inventory = inventory_safetensors(weight_dir)
    blockers: list[str] = []
    receipt: PersistentCacheInferenceReceipt | None = None
    missing: list[str] = []

    if importlib.util.find_spec("torch") is None:
        blockers.append("torch_missing")
    if importlib.util.find_spec("safetensors") is None:
        blockers.append("safetensors_missing")
    if importlib.util.find_spec("transformers") is None:
        blockers.append("transformers_missing")
    if not config.matches_project_qwen3_32b_shape():
        blockers.append("config_shape_mismatch")
    if weight_dir is None or not inventory.local_weights_present:
        blockers.append("missing_local_qwen3_32b_safetensors")

    if not blockers and weight_dir is not None:
        try:
            import torch

            runtime_device = _select_device(device)
            if runtime_device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            resolved_cache_dir = cache_dir or _default_cache_dir(weight_dir)
            stats = PackedCacheStats(cache_dir=str(resolved_cache_dir))
            weight_map = _load_weight_map(weight_dir)
            total_layers = int(config.layers)
            requested_layers = total_layers if layer_limit is None or layer_limit <= 0 else min(int(layer_limit), total_layers)
            required = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
            for layer_index in range(requested_layers):
                prefix = f"model.layers.{layer_index}"
                required.extend(
                    [
                        f"{prefix}.input_layernorm.weight",
                        f"{prefix}.post_attention_layernorm.weight",
                        f"{prefix}.self_attn.q_norm.weight",
                        f"{prefix}.self_attn.k_norm.weight",
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
                blockers.append("missing_t016_cache_tensors")
            else:
                try:
                    tokenizer = _tokenizer(weight_dir)
                    tokenizer_source = "transformers_tokenizer"
                except Exception:
                    tokenizer = _FallbackTokenizer()
                    tokenizer_source = "fallback_byte_tokenizer"
                token_ids = [int(token) for token in tokenizer.encode(prompt_text, add_special_tokens=False)]
                if not token_ids:
                    token_ids = [int(getattr(tokenizer, "bos_token_id", 0) or 0)]
                generated: list[int] = []
                started = time.perf_counter()
                denominator_total, denominator_slack = _t014_a_denominator(config)
                completed_layers = 0
                for _ in range(max_new_tokens):
                    rows = [int(token) for token in token_ids + generated]
                    x = _read_rows(weight_dir, weight_map, "model.embed_tokens.weight", rows, device=runtime_device)
                    completed_layers = 0
                    for layer_index in range(requested_layers):
                        prefix = f"model.layers.{layer_index}"
                        input_norm = _read_tensor(weight_dir, weight_map, f"{prefix}.input_layernorm.weight", device=runtime_device)
                        post_norm = _read_tensor(weight_dir, weight_map, f"{prefix}.post_attention_layernorm.weight", device=runtime_device)
                        q_norm = _read_tensor(weight_dir, weight_map, f"{prefix}.self_attn.q_norm.weight", device=runtime_device)
                        k_norm = _read_tensor(weight_dir, weight_map, f"{prefix}.self_attn.k_norm.weight", device=runtime_device)
                        x = _attention_forward_cached(
                            x,
                            model_dir=weight_dir,
                            weight_map=weight_map,
                            cache_dir=resolved_cache_dir,
                            stats=stats,
                            layer_index=layer_index,
                            input_norm=input_norm,
                            q_norm=q_norm,
                            k_norm=k_norm,
                            device=runtime_device,
                            rms_eps=float(config.as_dict().get("rms_norm_eps", 1e-6)),
                            rope_theta=1_000_000.0,
                            target_mlp_bits=target_mlp_bits,
                            refresh_cache=refresh_cache,
                        )
                        x = _mlp_forward_cached(
                            x,
                            model_dir=weight_dir,
                            weight_map=weight_map,
                            cache_dir=resolved_cache_dir,
                            stats=stats,
                            layer_index=layer_index,
                            post_norm=post_norm,
                            device=runtime_device,
                            rms_eps=float(config.as_dict().get("rms_norm_eps", 1e-6)),
                            target_mlp_bits=target_mlp_bits,
                            refresh_cache=refresh_cache,
                        )
                        completed_layers += 1
                        del input_norm, post_norm, q_norm, k_norm
                    final_norm = _read_tensor(weight_dir, weight_map, "model.norm.weight", device=runtime_device)
                    x = _rms_norm(x, final_norm, 1e-6)
                    next_token = _lm_head_argmax_cached(
                        x[-1],
                        model_dir=weight_dir,
                        weight_map=weight_map,
                        cache_dir=resolved_cache_dir,
                        stats=stats,
                        device=runtime_device,
                        chunk_rows=lm_head_chunk_rows,
                        target_mlp_bits=target_mlp_bits,
                        refresh_cache=refresh_cache,
                    )
                    generated.append(next_token)
                    del x, final_norm
                elapsed = time.perf_counter() - started
                peak_allocated, peak_reserved = _cuda_memory(runtime_device)
                generated_text = tokenizer.decode(generated, skip_special_tokens=False)
                digest = hashlib.sha256(json.dumps({"prompt": prompt_text, "generated": generated}).encode("utf-8")).hexdigest()
                receipt = PersistentCacheInferenceReceipt(
                    prompt_text=prompt_text,
                    tokenizer_source=tokenizer_source,
                    prompt_token_count=len(token_ids),
                    generated_token_ids=tuple(generated),
                    generated_text=generated_text,
                    requested_layers=requested_layers,
                    completed_layers=completed_layers,
                    device=runtime_device,
                    elapsed_seconds=elapsed,
                    tokens_per_second=len(generated) / elapsed if elapsed > 0 else 0.0,
                    peak_cuda_allocated_gib=peak_allocated,
                    peak_cuda_reserved_gib=peak_reserved,
                    denominator_total_gib=denominator_total,
                    denominator_slack_gib=denominator_slack,
                    cache_stats=stats.as_dict(),
                    output_sha256=digest,
                )
        except Exception as exc:  # pragma: no cover - real runtime guard
            blockers.append(f"t016_persistent_cache_failed:{type(exc).__name__}:{exc}")

    smoke_pass = receipt.passes if receipt is not None and not blockers else False
    if not smoke_pass and not blockers:
        blockers.append("t016_persistent_cache_failed_threshold")

    return {
        "probe": "t016_persistent_packed_cache",
        "activation": ACTIVATION,
        "model_id": model_id,
        "model_dir": str(weight_dir) if weight_dir is not None else None,
        "config": config.as_dict(),
        "weight_inventory": inventory.as_dict(),
        "receipt": receipt.as_dict() if receipt is not None else None,
        "missing_tensors": missing,
        "readiness": {
            "smoke_pass": smoke_pass,
            "full_model_text_inference_ready": bool(receipt is not None and receipt.requested_layers == config.layers and smoke_pass),
            "measured_tokens_per_second": receipt.tokens_per_second if receipt is not None else None,
            "config_matches_project_qwen3_32b_shape": config.matches_project_qwen3_32b_shape(),
            "local_weights_present": inventory.local_weights_present,
            "requested_device": device,
            "layer_limit": layer_limit,
            "max_new_tokens": max_new_tokens,
            "refresh_cache": refresh_cache,
        },
        "blockers": blockers,
        "next_actions": [
            "Move from cache-backed dense materialization to fused dequant/matmul kernels after this pre-kernel cache boundary."
        ]
        if smoke_pass
        else ["Fix the listed T016 cache blockers before starting fused kernel work."],
    }
