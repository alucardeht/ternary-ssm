from dataclasses import dataclass, field
from typing import List
import math

@dataclass
class ModelConfig:
    d_model: int = 256
    n_layer: int = 8
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int = 16
    vocab_size: int = 8192
    seq_len: int = 512

    n_heads: int = 4
    window_size: int = 128
    swa_layer_indices: List[int] = field(default_factory=lambda: [4])

    use_routing: bool = False
    routing_capacity: float = 0.5

    use_ternary: bool = False
    ternary_group_size: int = 128
    ste_type: str = 'standard'  # 'standard', 'biste', 'hestia', 'quest'

    use_ssd: bool = False
    ssd_n_heads: int = 16

    use_rwkv7: bool = False
    rwkv7_head_size: int = 64
    rwkv7_n_heads: int = None
    rwkv7_lora_rank: int = 32
    rwkv7_chunk_size: int = 64

    use_amp: bool = False
    use_bf16: bool = False
    use_gradient_checkpointing: bool = False
    checkpoint_every_n: int = 1
    use_sdpa: bool = True

    use_torch_compile: bool = False
    use_cpu_offload: bool = False

    use_distillation: bool = False
    teacher_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    distill_alpha: float = 0.5
    distill_temperature: float = 2.0
    use_tsld: bool = False

    tied_embeddings: bool = True
    dropout: float = 0.0

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    def __post_init__(self):
        if self.use_rwkv7 and self.use_ssd:
            raise ValueError("use_rwkv7 and use_ssd are mutually exclusive")
        if self.rwkv7_n_heads is None and self.use_rwkv7:
            self.rwkv7_n_heads = self.d_model // self.rwkv7_head_size

    def is_swa_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.swa_layer_indices


def config_1m() -> ModelConfig:
    return ModelConfig(
        d_model=256, n_layer=8, d_state=16, d_conv=4, expand=2,
        dt_rank=16, vocab_size=8192, seq_len=128,
        n_heads=4, window_size=128, swa_layer_indices=[4],
    )


def config_10m() -> ModelConfig:
    return ModelConfig(
        d_model=512, n_layer=16, d_state=16, d_conv=4, expand=2,
        dt_rank=32, vocab_size=16384, seq_len=1024,
        n_heads=8, window_size=128, swa_layer_indices=[5, 11],
    )


def config_100m() -> ModelConfig:
    return ModelConfig(
        d_model=768, n_layer=24, d_state=16, d_conv=4, expand=2,
        dt_rank=48, vocab_size=32000, seq_len=2048,
        n_heads=12, window_size=128, swa_layer_indices=[6, 12, 18],
    )


def config_scale_10m() -> ModelConfig:
    """~11.6M params for scale validation. seq_len=128 for SSD compatibility."""
    return ModelConfig(
        d_model=384, n_layer=8, d_state=16, d_conv=4, expand=2,
        dt_rank=24, vocab_size=8192, seq_len=128,
        n_heads=6, window_size=128, swa_layer_indices=[4],
    )


def config_scale_50m() -> ModelConfig:
    """~56.9M params for scale validation. seq_len=128 for SSD compatibility."""
    return ModelConfig(
        d_model=640, n_layer=18, d_state=16, d_conv=4, expand=2,
        dt_rank=40, vocab_size=8192, seq_len=128,
        n_heads=10, window_size=128, swa_layer_indices=[6, 12],
        use_torch_compile=True,
    )


def config_scale_100m() -> ModelConfig:
    """~112.5M params for scale validation. seq_len=128 for SSD compatibility."""
    return ModelConfig(
        d_model=768, n_layer=24, d_state=16, d_conv=4, expand=2,
        dt_rank=48, vocab_size=16384, seq_len=128,
        n_heads=12, window_size=128, swa_layer_indices=[6, 12, 18],
        use_torch_compile=True,
    )


def config_wikitext_6m() -> ModelConfig:
    """~6M params for WikiText-103. vocab=32000, seq_len=256."""
    return ModelConfig(
        d_model=256, n_layer=8, d_state=16, d_conv=4, expand=2,
        dt_rank=16, vocab_size=32000, seq_len=256,
        n_heads=4, window_size=128, swa_layer_indices=[4],
        use_ssd=True, ssd_n_heads=16,
    )


def config_wikitext_50m() -> ModelConfig:
    """~50M params for WikiText-103. vocab=32000, seq_len=256."""
    return ModelConfig(
        d_model=640, n_layer=18, d_state=16, d_conv=4, expand=2,
        dt_rank=40, vocab_size=32000, seq_len=256,
        n_heads=10, window_size=128, swa_layer_indices=[6, 12],
        use_ssd=True, ssd_n_heads=40,
        use_torch_compile=True,
        dropout=0.1,
    )


def config_wikitext_100m() -> ModelConfig:
    """~100M params for WikiText-103. vocab=32000, seq_len=256."""
    return ModelConfig(
        d_model=768, n_layer=24, d_state=16, d_conv=4, expand=2,
        dt_rank=48, vocab_size=32000, seq_len=256,
        n_heads=12, window_size=128, swa_layer_indices=[6, 12, 18],
        use_ssd=True, ssd_n_heads=48,
        use_torch_compile=True,
    )


def config_wikitext_bench() -> ModelConfig:
    """~6M params for fast A/B benchmarking. Same as wikitext_6m."""
    return ModelConfig(
        d_model=256, n_layer=8, d_state=16, d_conv=4, expand=2,
        dt_rank=16, vocab_size=32000, seq_len=256,
        n_heads=4, window_size=128, swa_layer_indices=[4],
    )
