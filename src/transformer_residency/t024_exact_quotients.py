"""Exact quotient checks for T024 GPU-bound runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from safetensors import safe_open

from .t008_compressed_generation_smoke import _load_weight_map


def exact_quotient_summary(*, model_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Return exact, no-quality-tradeoff quotient opportunities for the model."""

    weight_map = _load_weight_map(model_dir)
    embed_shape = _tensor_shape(model_dir, weight_map, "model.embed_tokens.weight")
    lm_head_shape = _tensor_shape(model_dir, weight_map, "lm_head.weight")
    configured_vocab = int(config.get("vocab_size", 0) or 0)
    tie_word_embeddings = bool(config.get("tie_word_embeddings", False))
    invalid_lm_head_rows = max(0, int(lm_head_shape[0]) - configured_vocab)
    alias_active = bool(tie_word_embeddings and embed_shape == lm_head_shape)
    return {
        "lm_head_embedding_alias_active": alias_active,
        "tie_word_embeddings": tie_word_embeddings,
        "embedding_shape": list(embed_shape),
        "lm_head_shape": list(lm_head_shape),
        "invalid_lm_head_rows": invalid_lm_head_rows,
        "invalid_lm_head_row_quotient_active": invalid_lm_head_rows > 0,
        "zero_block_quotient_active": False,
        "duplicate_code_block_quotient_active": False,
        "claim_boundary": "metadata-exact quotients only; no tensor equality hash or duplicate-code indirection is applied unless metadata proves exact applicability",
    }


def _tensor_shape(model_dir: Path, weight_map: dict[str, str], name: str) -> tuple[int, ...]:
    shard = weight_map[name]
    with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
        return tuple(int(dim) for dim in handle.get_slice(name).get_shape())


__all__ = ["exact_quotient_summary"]
