"""T015 streamed compressed Qwen inference smoke."""

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
from .kv_margin import kv_cache_gib_for_context
from .t004_real_qwen_bridge import find_weight_dir, inventory_safetensors, resolve_qwen_config
from .t006_real_residency_block_smoke import BlockCodecPolicy, _compress_reconstruct, _policy_for_tensor
from .t008_compressed_generation_smoke import _load_weight_map
from .t009_t013_runtime_scaler import _component_policy_bits
from .t014_denominator_tightening import _policy_for_target_mlp_bits


ACTIVATION = "T015_PACKED_QWEN_STREAMED_INFERENCE_ACTIVE"


@dataclass(frozen=True)
class StreamedInferenceReceipt:
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


def _read_tensor(model_dir: Path, weight_map: dict[str, str], name: str, *, device: str = "cpu") -> Any:
    shard = weight_map[name]
    with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(name).detach().float()
    if device != "cpu":
        tensor = tensor.to(device)
    return tensor


def _read_rows(model_dir: Path, weight_map: dict[str, str], name: str, rows: list[int], *, device: str) -> Any:
    shard = weight_map[name]
    with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
        full = handle.get_tensor(name)
        tensor = full[rows].detach().float()
    if device != "cpu":
        tensor = tensor.to(device)
    return tensor


def _compressed_weight(
    model_dir: Path,
    weight_map: dict[str, str],
    name: str,
    *,
    device: str,
    target_mlp_bits: float,
) -> tuple[Any, float]:
    exact = _read_tensor(model_dir, weight_map, name, device="cpu")
    if ".mlp." in name:
        policy = _policy_for_target_mlp_bits(name, int(exact.shape[1]), target_mlp_bits)
    else:
        policy = _policy_for_tensor(name)
    compressed = _compress_reconstruct(exact, policy)
    effective_bits = policy.effective_bits(int(exact.shape[1]))
    if device != "cpu":
        compressed = compressed.to(device)
    return compressed, effective_bits


def _rms_norm(x: Any, weight: Any, eps: float) -> Any:
    import torch

    return x * torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps) * weight


def _silu(x: Any) -> Any:
    import torch

    return x * torch.sigmoid(x)


def _apply_rope(x: Any, *, theta: float, positions: Any) -> Any:
    import torch

    head_dim = int(x.shape[-1])
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=x.device, dtype=x.float().dtype) / head_dim))
    freqs = torch.outer(positions.to(x.device, dtype=x.float().dtype), inv_freq)
    cos = freqs.cos().unsqueeze(1)
    sin = freqs.sin().unsqueeze(1)
    even = x[..., 0::2].float()
    odd = x[..., 1::2].float()
    out = torch.empty_like(x.float())
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = odd * cos + even * sin
    return out.to(x.dtype)


def _attention_forward(
    x: Any,
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    layer_index: int,
    input_norm: Any,
    q_norm: Any,
    k_norm: Any,
    device: str,
    rms_eps: float,
    rope_theta: float,
    target_mlp_bits: float,
) -> tuple[Any, list[float]]:
    import torch

    prefix = f"model.layers.{layer_index}.self_attn"
    h = _rms_norm(x, input_norm, rms_eps)
    q_weight, q_bits = _compressed_weight(model_dir, weight_map, f"{prefix}.q_proj.weight", device=device, target_mlp_bits=target_mlp_bits)
    q = h @ q_weight.T
    del q_weight
    k_weight, k_bits = _compressed_weight(model_dir, weight_map, f"{prefix}.k_proj.weight", device=device, target_mlp_bits=target_mlp_bits)
    k = h @ k_weight.T
    del k_weight
    v_weight, v_bits = _compressed_weight(model_dir, weight_map, f"{prefix}.v_proj.weight", device=device, target_mlp_bits=target_mlp_bits)
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

    o_weight, o_bits = _compressed_weight(model_dir, weight_map, f"{prefix}.o_proj.weight", device=device, target_mlp_bits=target_mlp_bits)
    out = context @ o_weight.T
    del o_weight
    return x + out, [q_bits, k_bits, v_bits, o_bits]


def _mlp_forward(
    x: Any,
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    layer_index: int,
    post_norm: Any,
    device: str,
    rms_eps: float,
    target_mlp_bits: float,
) -> tuple[Any, list[float]]:
    prefix = f"model.layers.{layer_index}.mlp"
    h = _rms_norm(x, post_norm, rms_eps)
    gate_weight, gate_bits = _compressed_weight(model_dir, weight_map, f"{prefix}.gate_proj.weight", device=device, target_mlp_bits=target_mlp_bits)
    gate = _silu(h @ gate_weight.T)
    del gate_weight
    up_weight, up_bits = _compressed_weight(model_dir, weight_map, f"{prefix}.up_proj.weight", device=device, target_mlp_bits=target_mlp_bits)
    up = h @ up_weight.T
    del up_weight
    hidden = gate * up
    del gate, up
    down_weight, down_bits = _compressed_weight(model_dir, weight_map, f"{prefix}.down_proj.weight", device=device, target_mlp_bits=target_mlp_bits)
    out = hidden @ down_weight.T
    del down_weight, hidden
    return x + out, [gate_bits, up_bits, down_bits]


def _lm_head_argmax_chunked(
    x_last: Any,
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    device: str,
    chunk_rows: int,
    target_mlp_bits: float,
) -> int:
    import torch

    shard = weight_map["lm_head.weight"]
    best_value: Any | None = None
    best_index = 0
    with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice("lm_head.weight")
        rows, _ = [int(dim) for dim in tensor_slice.get_shape()]
        for start in range(0, rows, chunk_rows):
            stop = min(rows, start + chunk_rows)
            exact = tensor_slice[start:stop, :].detach().float()
            policy = BlockCodecPolicy(
                name="lm_head_streamed_q4_boundary",
                target_bits_per_weight=4.0,
                repair_fraction=0.08,
            )
            compressed = _compress_reconstruct(exact, policy)
            if device != "cpu":
                compressed = compressed.to(device)
            logits = x_last @ compressed.T
            value, index = torch.max(logits, dim=-1)
            if best_value is None or bool(value.item() > best_value):
                best_value = float(value.item())
                best_index = start + int(index.item())
            del exact, compressed, logits
    return best_index


def _tokenizer(model_dir: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)


class _FallbackTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [int(byte) for byte in text.encode("utf-8")]

    def decode(self, tokens: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return " ".join(f"<token:{int(token)}>" for token in tokens)


def _select_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if importlib.util.find_spec("torch") is None:
        return "cpu"
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _cuda_memory(device: str) -> tuple[float | None, float | None]:
    if device != "cuda":
        return None, None
    import torch

    return torch.cuda.max_memory_allocated() / BYTES_PER_GIB, torch.cuda.max_memory_reserved() / BYTES_PER_GIB


def _t014_a_denominator(config: Any) -> tuple[float, float]:
    shape = config.to_shape()
    current_bits = _component_policy_bits(config)
    residency = shape.component_residency_projection(
        mlp_bits=2.45,
        attention_bits=current_bits["attention_bits"],
        embedding_bits=current_bits["embedding_bits"],
        output_head_bits=current_bits["output_head_bits"],
        overhead_gib=0.25,
    )
    total = float(residency["candidate_estimated_gib"]) + kv_cache_gib_for_context(16_384, effective_bits=4.0, shape=shape)
    return total, 12.0 - total


def run_t015_packed_qwen_inference(
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
) -> dict[str, Any]:
    """Run a streamed compressed Qwen token-generation smoke."""

    config = resolve_qwen_config(
        model_id=model_id,
        local_model_dir=local_model_dir,
        allow_hf_config_fetch=allow_hf_config_fetch,
    )
    weight_dir = find_weight_dir(config, local_model_dir)
    inventory = inventory_safetensors(weight_dir)
    blockers: list[str] = []
    receipt: StreamedInferenceReceipt | None = None
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
                blockers.append("missing_t015_inference_tensors")
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
                        x, _ = _attention_forward(
                            x,
                            model_dir=weight_dir,
                            weight_map=weight_map,
                            layer_index=layer_index,
                            input_norm=input_norm,
                            q_norm=q_norm,
                            k_norm=k_norm,
                            device=runtime_device,
                            rms_eps=float(config.as_dict().get("rms_norm_eps", 1e-6)),
                            rope_theta=1_000_000.0,
                            target_mlp_bits=target_mlp_bits,
                        )
                        x, _ = _mlp_forward(
                            x,
                            model_dir=weight_dir,
                            weight_map=weight_map,
                            layer_index=layer_index,
                            post_norm=post_norm,
                            device=runtime_device,
                            rms_eps=float(config.as_dict().get("rms_norm_eps", 1e-6)),
                            target_mlp_bits=target_mlp_bits,
                        )
                        completed_layers += 1
                        del input_norm, post_norm, q_norm, k_norm
                    final_norm = _read_tensor(weight_dir, weight_map, "model.norm.weight", device=runtime_device)
                    x = _rms_norm(x, final_norm, 1e-6)
                    next_token = _lm_head_argmax_chunked(
                        x[-1],
                        model_dir=weight_dir,
                        weight_map=weight_map,
                        device=runtime_device,
                        chunk_rows=lm_head_chunk_rows,
                        target_mlp_bits=target_mlp_bits,
                    )
                    generated.append(next_token)
                    del x, final_norm
                elapsed = time.perf_counter() - started
                peak_allocated, peak_reserved = _cuda_memory(runtime_device)
                generated_text = tokenizer.decode(generated, skip_special_tokens=False)
                digest = hashlib.sha256(json.dumps({"prompt": prompt_text, "generated": generated}).encode("utf-8")).hexdigest()
                receipt = StreamedInferenceReceipt(
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
                    output_sha256=digest,
                )
        except Exception as exc:  # pragma: no cover - real runtime guard
            blockers.append(f"t015_streamed_inference_failed:{type(exc).__name__}:{exc}")

    smoke_pass = receipt.passes if receipt is not None and not blockers else False
    if not smoke_pass and not blockers:
        blockers.append("t015_streamed_inference_failed_threshold")

    return {
        "probe": "t015_packed_qwen_streamed_inference",
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
        },
        "blockers": blockers,
        "next_actions": [
            "Replace streamed on-the-fly reconstruction with a persistent packed GPU arena before claiming optimized tok/s."
        ]
        if smoke_pass
        else ["Fix the listed T015 inference blockers before claiming text inference through the project codec."],
    }
