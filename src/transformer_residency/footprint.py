"""Coarse transformer footprint estimates for residency gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass


BYTES_PER_GIB = 1024**3


@dataclass(frozen=True)
class TransformerShape:
    name: str
    layers: int
    hidden_size: int
    intermediate_size: int
    attention_heads: int
    kv_heads: int
    configured_head_dim: int | None = None
    vocab_size: int = 151_936
    include_embeddings: bool = True
    include_output_head: bool = True

    @property
    def head_dim(self) -> int:
        if self.configured_head_dim is not None:
            return self.configured_head_dim
        if self.hidden_size % self.attention_heads != 0:
            raise ValueError("hidden_size must be divisible by attention_heads")
        return self.hidden_size // self.attention_heads

    @property
    def query_width(self) -> int:
        return self.attention_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def attention_elements_per_layer(self) -> int:
        q_proj = self.hidden_size * self.query_width
        k_proj = self.hidden_size * self.kv_width
        v_proj = self.hidden_size * self.kv_width
        o_proj = self.query_width * self.hidden_size
        return q_proj + k_proj + v_proj + o_proj

    @property
    def mlp_elements_per_layer(self) -> int:
        gate = self.hidden_size * self.intermediate_size
        up = self.hidden_size * self.intermediate_size
        down = self.intermediate_size * self.hidden_size
        return gate + up + down

    @property
    def embedding_elements(self) -> int:
        if not self.include_embeddings:
            return 0
        return self.vocab_size * self.hidden_size

    @property
    def output_head_elements(self) -> int:
        if not self.include_output_head:
            return 0
        return self.vocab_size * self.hidden_size

    @property
    def estimated_weight_elements(self) -> int:
        return (
            self.layers * (self.attention_elements_per_layer + self.mlp_elements_per_layer)
            + self.embedding_elements
            + self.output_head_elements
        )

    @property
    def estimated_mlp_fraction(self) -> float:
        layer_total = self.attention_elements_per_layer + self.mlp_elements_per_layer
        if layer_total <= 0:
            return 0.0
        return self.mlp_elements_per_layer / layer_total

    def gib_at_bits(self, bits_per_weight: float, overhead_gib: float = 0.0) -> float:
        payload_bytes = self.estimated_weight_elements * bits_per_weight / 8.0
        return payload_bytes / BYTES_PER_GIB + overhead_gib

    def residency_projection(
        self,
        *,
        candidate_bits: float,
        overhead_gib: float = 1.0,
        q4_overhead_gib: float = 1.0,
    ) -> dict[str, float | int | str]:
        fp16_gib = self.gib_at_bits(16.0, overhead_gib=0.0)
        q4_gib = self.gib_at_bits(4.0, overhead_gib=q4_overhead_gib)
        candidate_gib = self.gib_at_bits(candidate_bits, overhead_gib=overhead_gib)
        return {
            "name": self.name,
            "estimated_weight_elements": self.estimated_weight_elements,
            "estimated_weight_billions": self.estimated_weight_elements / 1e9,
            "estimated_mlp_fraction": self.estimated_mlp_fraction,
            "fp16_weights_gib": fp16_gib,
            "q4_estimated_gib": q4_gib,
            "candidate_bits_per_weight": candidate_bits,
            "candidate_estimated_gib": candidate_gib,
            "candidate_saves_gib_vs_q4": q4_gib - candidate_gib,
            "candidate_fits_12gib": candidate_gib <= 12.0,
        }

    def component_residency_projection(
        self,
        *,
        mlp_bits: float,
        attention_bits: float,
        embedding_bits: float,
        output_head_bits: float,
        overhead_gib: float = 1.0,
        q4_overhead_gib: float = 1.0,
    ) -> dict[str, float | int | str | bool]:
        mlp_elements = self.layers * self.mlp_elements_per_layer
        attention_elements = self.layers * self.attention_elements_per_layer
        embedding_elements = self.embedding_elements
        output_head_elements = self.output_head_elements
        payload_bits = (
            mlp_elements * mlp_bits
            + attention_elements * attention_bits
            + embedding_elements * embedding_bits
            + output_head_elements * output_head_bits
        )
        total_elements = self.estimated_weight_elements
        effective_bits = payload_bits / total_elements if total_elements else 0.0
        candidate_gib = payload_bits / 8.0 / BYTES_PER_GIB + overhead_gib
        q4_gib = self.gib_at_bits(4.0, overhead_gib=q4_overhead_gib)
        return {
            "name": self.name,
            "estimated_weight_elements": total_elements,
            "estimated_weight_billions": total_elements / 1e9,
            "mlp_elements": mlp_elements,
            "attention_elements": attention_elements,
            "embedding_elements": embedding_elements,
            "output_head_elements": output_head_elements,
            "mlp_bits": mlp_bits,
            "attention_bits": attention_bits,
            "embedding_bits": embedding_bits,
            "output_head_bits": output_head_bits,
            "effective_bits_per_weight": effective_bits,
            "candidate_estimated_gib": candidate_gib,
            "q4_estimated_gib": q4_gib,
            "candidate_saves_gib_vs_q4": q4_gib - candidate_gib,
            "candidate_fits_12gib": candidate_gib <= 12.0,
        }

    def synthetic_decode_throughput_projection(
        self,
        *,
        uniform_bits: float,
        mlp_bits: float,
        attention_bits: float,
        embedding_bits: float,
        output_head_bits: float,
        bandwidth_gib_per_second: float,
        weight_streams_per_token: float = 1.0,
        kv_cache_gib_per_token: float = 0.0,
    ) -> dict[str, float | int | str]:
        if bandwidth_gib_per_second <= 0.0:
            raise ValueError("bandwidth_gib_per_second must be positive")
        if weight_streams_per_token <= 0.0:
            raise ValueError("weight_streams_per_token must be positive")
        if kv_cache_gib_per_token < 0.0:
            raise ValueError("kv_cache_gib_per_token must be non-negative")

        q4_weight_payload_gib = self.gib_at_bits(4.0, overhead_gib=0.0)
        uniform_weight_payload_gib = self.gib_at_bits(uniform_bits, overhead_gib=0.0)

        mlp_elements = self.layers * self.mlp_elements_per_layer
        attention_elements = self.layers * self.attention_elements_per_layer
        component_payload_bits = (
            mlp_elements * mlp_bits
            + attention_elements * attention_bits
            + self.embedding_elements * embedding_bits
            + self.output_head_elements * output_head_bits
        )
        component_weight_payload_gib = component_payload_bits / 8.0 / BYTES_PER_GIB

        def tokens_per_second(weight_payload_gib: float) -> float:
            gib_per_token = weight_payload_gib * weight_streams_per_token + kv_cache_gib_per_token
            return bandwidth_gib_per_second / gib_per_token

        q4_tokens_per_second = tokens_per_second(q4_weight_payload_gib)
        uniform_tokens_per_second = tokens_per_second(uniform_weight_payload_gib)
        component_tokens_per_second = tokens_per_second(component_weight_payload_gib)

        return {
            "name": self.name,
            "model": "synthetic_memory_bound_decode_weight_stream",
            "bandwidth_gib_per_second": bandwidth_gib_per_second,
            "weight_streams_per_token": weight_streams_per_token,
            "kv_cache_gib_per_token": kv_cache_gib_per_token,
            "q4_weight_payload_gib": q4_weight_payload_gib,
            "uniform_weight_payload_gib": uniform_weight_payload_gib,
            "component_weight_payload_gib": component_weight_payload_gib,
            "q4_tokens_per_second": q4_tokens_per_second,
            "uniform_tokens_per_second": uniform_tokens_per_second,
            "component_tokens_per_second": component_tokens_per_second,
            "uniform_speedup_vs_q4": uniform_tokens_per_second / q4_tokens_per_second,
            "component_speedup_vs_q4": component_tokens_per_second / q4_tokens_per_second,
            "component_speedup_vs_uniform": component_tokens_per_second / uniform_tokens_per_second,
        }

    def as_dict(self) -> dict[str, int | str | bool]:
        return asdict(self)


def qwen3_32b_shape() -> TransformerShape:
    """Return the dense Qwen3-32B shape used by the T001 research artifacts."""

    return TransformerShape(
        name="Qwen3-32B dense approximation",
        layers=64,
        hidden_size=5120,
        intermediate_size=25600,
        attention_heads=64,
        kv_heads=8,
        configured_head_dim=128,
    )
