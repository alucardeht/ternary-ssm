import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from .config import ModelConfig
from .components.rmsnorm import RMSNorm
from .components.ssm import SelectiveSSM, SelectiveSSMSSD
from .components.attention import SlidingWindowAttention
from .components.router import MoDRouter
from .components.ternary import TernaryLinear
from .components.rwkv7 import RWKV7Block


class ModelBlock(nn.Module):
    """Single model block — either SSM or SWA, with optional routing."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_swa = config.is_swa_layer(layer_idx)
        self.is_rwkv7 = False

        # Pre-norm
        self.norm = RMSNorm(config.d_model)

        # Core block
        if self.is_swa:
            self.block = SlidingWindowAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                window_size=config.window_size,
                use_ternary=config.use_ternary,
                ternary_group_size=config.ternary_group_size,
                max_seq_len=config.seq_len,
                dropout=config.dropout,
                use_sdpa=config.use_sdpa,
            )
        else:
            if config.use_rwkv7:
                self.is_rwkv7 = True
                self.block = RWKV7Block(
                    d_model=config.d_model,
                    head_size=config.rwkv7_head_size,
                    n_heads=config.rwkv7_n_heads,
                    lora_rank=config.rwkv7_lora_rank,
                    use_ternary=config.use_ternary,
                    ternary_group_size=config.ternary_group_size,
                    chunk_size=config.rwkv7_chunk_size,
                )
            elif config.use_ssd:
                self.block = SelectiveSSMSSD(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                    dt_rank=config.dt_rank,
                    n_heads=config.ssd_n_heads,
                    use_ternary=config.use_ternary,
                    ternary_group_size=config.ternary_group_size,
                    dropout=config.dropout,
                )
            else:
                self.block = SelectiveSSM(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                    dt_rank=config.dt_rank,
                    use_ternary=config.use_ternary,
                    ternary_group_size=config.ternary_group_size,
                    dropout=config.dropout,
                )

        # Router (optional, only for SWA layers)
        self.router = None
        if config.use_routing and self.is_swa:
            self.router = MoDRouter(config.d_model, capacity=config.routing_capacity)

    def forward(self, x: torch.Tensor, v_first: torch.Tensor = None):
        residual = x
        x_norm = self.norm(x)

        if self.router is not None:
            mask, weights = self.router(x_norm)

            if self.is_swa:
                block_out = self.block(x) - x
            elif self.is_rwkv7:
                block_out, v_first = self.block(x_norm, v_first=v_first)
            else:
                block_out = self.block(x_norm)

            mask_ste = mask - weights.detach() + weights
            out = residual + mask_ste * block_out
            if self.is_rwkv7:
                return out, v_first
            return out
        else:
            if self.is_swa:
                return self.block(x)
            elif self.is_rwkv7:
                block_out, v_first = self.block(x_norm, v_first=v_first)
                out = residual + block_out
                return out, v_first
            else:
                return residual + self.block(x_norm)


class ModelV2(nn.Module):
    """Model v2: Hybrid SSM-Attention with ternary weights and dynamic routing.

    Architecture:
        Embedding -> [N layers of SSM/SWA blocks] -> RMSNorm -> Output Head

    Properties:
        - ~87.5% SSM layers, ~12.5% SWA layers
        - Ternary weights for projections (when enabled)
        - MoD routing (when enabled)
        - Pre-norm architecture throughout
        - Tied embeddings
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embedding (FP16 always)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

        # Layers
        self.layers = nn.ModuleList([
            ModelBlock(config, i) for i in range(config.n_layer)
        ])

        # Final norm
        self.final_norm = RMSNorm(config.d_model)

        # Output scaling (for μP coordinate check)
        self.output_multiplier = 1.0

        # Output head (tied with embedding)
        if config.tied_embeddings:
            self.lm_head = None  # Will use embedding.weight
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token indices
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        x = self.embedding(input_ids)  # (B, L, d_model)

        v_first = None
        for i, layer in enumerate(self.layers):
            if self.config.use_gradient_checkpointing and self.training and i % self.config.checkpoint_every_n == 0 and i > 0:
                if getattr(layer, 'is_rwkv7', False):
                    x, v_first = grad_checkpoint(layer, x, v_first, use_reentrant=False)
                else:
                    x = grad_checkpoint(layer, x, use_reentrant=False)
            else:
                result = layer(x, v_first) if getattr(layer, 'is_rwkv7', False) else layer(x)
                if isinstance(result, tuple):
                    x, v_first = result
                else:
                    x = result

        x = self.final_norm(x)

        # Output projection
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.embedding.weight)

        logits = logits * self.output_multiplier

        return logits

    def count_parameters(self) -> dict:
        """Count parameters by category."""
        total = 0
        ternary = 0
        fp16 = 0

        for name, param in self.named_parameters():
            n = param.numel()
            total += n

            # Check if this is a ternary layer
            if self.config.use_ternary and any(k in name for k in ['in_proj', 'out_proj', 'x_proj', 'q_proj', 'k_proj', 'v_proj', 'mlp_up', 'mlp_gate', 'mlp_down']):
                ternary += n
            else:
                fp16 += n

        return {
            'total': total,
            'ternary': ternary,
            'fp16': fp16,
            'ternary_pct': ternary / total * 100 if total > 0 else 0,
            'memory_mb_fp16': total * 2 / 1024**2,
            'memory_mb_mixed': (ternary * 0.2 + fp16 * 2) / 1024**2,
        }

    def get_param_groups(self, base_lr: float) -> list:
        """Create parameter groups with differential learning rates.

        Returns list of dicts for optimizer:
          - Router: 5x LR
          - A matrix: 0.1x LR
          - Embeddings: 0.5x LR
          - Everything else: 1x LR
        """
        router_params = []
        a_params = []
        embed_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'router' in name:
                router_params.append(param)
            elif 'A_log' in name:
                a_params.append(param)
            elif 'embedding' in name or 'lm_head' in name:
                embed_params.append(param)
            else:
                other_params.append(param)

        return [
            {'params': other_params, 'lr': base_lr},
            {'params': router_params, 'lr': base_lr * 1.0, 'name': 'router'},
            {'params': a_params, 'lr': base_lr * 0.1, 'name': 'a_matrix'},
            {'params': embed_params, 'lr': base_lr * 0.5, 'weight_decay': 0.0, 'name': 'embedding'},
        ]

    def set_quantization_enabled(self, enabled: bool):
        """Enable or disable ternary quantization across all TernaryLinear layers."""
        for module in self.modules():
            if isinstance(module, TernaryLinear):
                module.quantize_enabled = enabled

    def set_quantization_layer_wise(self, enabled: bool, keep_fp16_layers: list = None):
        """Enable quantization with optional layer-wise exceptions.

        Args:
            enabled: Enable ternary quantization
            keep_fp16_layers: List of layer indices to keep in FP16 (e.g., [0, n_layer-1])
        """
        if keep_fp16_layers is None:
            keep_fp16_layers = []
        for i, layer in enumerate(self.layers):
            for module in layer.modules():
                if isinstance(module, TernaryLinear):
                    if i in keep_fp16_layers:
                        module.quantize_enabled = False
                    else:
                        module.quantize_enabled = enabled

    def set_routing_enabled(self, enabled: bool):
        """Enable or disable token routing across all MoDRouter layers."""
        for module in self.modules():
            if isinstance(module, MoDRouter):
                module.routing_enabled = enabled

    def set_ste_type(self, ste_type: str):
        """Set gradient estimator type across all TernaryLinear layers."""
        for module in self.modules():
            if isinstance(module, TernaryLinear):
                module.ste_type = ste_type

    def set_training_progress(self, step: int, total_steps: int):
        """Update training step for HESTIA temperature scheduling.

        Pre-computes tau for all HESTIA layers to avoid redundant computation
        on each forward pass.
        """
        progress = step / max(1, total_steps)
        tau = 2.0 * 0.5 * (1.0 + math.cos(math.pi * progress))
        tau = max(tau, 0.01)

        for module in self.modules():
            if isinstance(module, TernaryLinear):
                module.training_step = step
                module.total_training_steps = total_steps
                if module.ste_type == 'hestia':
                    module._cached_tau = tau

    def set_gradient_checkpointing(self, enabled: bool, every_n: int = 1):
        """Enable or disable gradient checkpointing."""
        self.config.use_gradient_checkpointing = enabled
        self.config.checkpoint_every_n = every_n

    def convert_ternary_to_bf16(self):
        """Convert ternary layer shadow weights to BF16 for memory savings.

        Only converts TernaryLinear weights. Embedding, norms, and other
        FP32 parameters remain in FP32 for precision.
        """
        converted = 0
        for module in self.modules():
            if isinstance(module, TernaryLinear):
                module.weight.data = module.weight.data.to(torch.bfloat16)
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(torch.bfloat16)
                converted += 1
        return converted

    def get_muon_param_groups(self, base_lr: float, muon_lr_mult: float = 10.0) -> list:
        """Create parameter groups for Muon + AdamW hybrid optimizer.

        Muon: All 2D hidden weight matrices (SSM projections, SWA Q/K/V/O, FFN)
        AdamW: Embeddings, output head, 1D params (norms, biases), A_log

        Args:
            base_lr: Base learning rate for AdamW group
            muon_lr_mult: Multiplier for Muon LR (default 10x, typical for Muon)

        Returns:
            List of param group dicts with 'use_muon' flag
        """
        muon_params = []
        adamw_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if 'embedding' in name or 'lm_head' in name:
                adamw_params.append(param)
            elif 'A_log' in name:
                adamw_params.append(param)
            elif param.ndim < 2:
                adamw_params.append(param)
            else:
                muon_params.append(param)

        return [
            {
                'params': muon_params,
                'use_muon': True,
                'lr': base_lr * muon_lr_mult,
                'weight_decay': 0.01,
            },
            {
                'params': adamw_params,
                'use_muon': False,
                'lr': base_lr,
                'betas': (0.9, 0.95),
                'weight_decay': 0.1,
            },
        ]

    def set_gradual_threshold(self, enabled: bool, start: float = 0.5, end: float = 0.7):
        """Enable gradual threshold softening across all TernaryLinear layers."""
        for module in self.modules():
            if isinstance(module, TernaryLinear):
                module.gradual_threshold = enabled
                module.threshold_start = start
                module.threshold_end = end

    def set_pressure_blend(self, enabled: bool, duration: float = 0.2):
        """Enable pressure blending (gradual FP->quantized) across all TernaryLinear layers."""
        for module in self.modules():
            if isinstance(module, TernaryLinear):
                module.pressure_blend = enabled
                module.pressure_duration = duration

    def reinit_ternary_standard(self):
        """Re-init TernaryLinear weights with standard Xavier (for progressive training FP32 warmup)."""
        for module in self.modules():
            if isinstance(module, TernaryLinear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)

    @torch.no_grad()
    def generate(self, prompt_ids, max_tokens=128, temperature=1.0, top_k=50, top_p=0.9):
        from .inference import generate as _generate
        return _generate(self, prompt_ids, max_tokens, temperature, top_k, top_p)
