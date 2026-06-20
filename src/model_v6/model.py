import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from .config import ModelConfig
from .components.rmsnorm import RMSNorm, BitShiftNorm
from .components.ssm import SelectiveSSM, SelectiveSSMSSD, EigenmodeMixedSSMSSD
from .components.attention import SlidingWindowAttention
from .components.router import MoDRouter
from .components.ternary import TernaryLinear, DEFAULT_THRESHOLD_COEFF
from .components.rwkv7 import RWKV7Block
from .components.moe import TernaryMoEFFN


class ReversibleBlock(nn.Module):
    """Reversible residual block - inputs reconstructed from outputs.

    A2-F4: REVERSIBLE-LAYERS-REVNET
    Uses RevNet-style coupling:
        y1 = x1 + F(x2)
        y2 = x2 + G(y1)

    Inverse (reconstruction during backward):
        x2 = y2 - G(y1)
        x1 = y1 - F(x2)

    This allows reconstruction of activations during backward without storing them.
    Reduces activation memory from O(L) to O(1) for L layers.

    Args:
        f_block: Function block F (e.g., SSM layer)
        g_block: Function block G (e.g., FFN layer)
    """

    def __init__(self, f_block: nn.Module, g_block: nn.Module):
        super().__init__()
        self.f_block = f_block
        self.g_block = g_block

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple:
        """Forward pass through reversible block.

        Args:
            x1: First part of input
            x2: Second part of input

        Returns:
            Tuple (y1, y2) where y1 and y2 are the output parts.
        """
        y1 = x1 + self.f_block(x2)
        y2 = x2 + self.g_block(y1)
        return y1, y2

    def inverse(self, y1: torch.Tensor, y2: torch.Tensor) -> tuple:
        """Inverse pass to reconstruct inputs from outputs.

        Args:
            y1: First part of output
            y2: Second part of output

        Returns:
            Tuple (x1, x2) of reconstructed inputs.
        """
        x2 = y2 - self.g_block(y1)
        x1 = y1 - self.f_block(x2)
        return x1, x2


def _get_norm_class(config):
    if getattr(config, 'use_bitshift_norm', False):
        return BitShiftNorm
    return RMSNorm


class ModelBlock(nn.Module):
    """Single model block - either SSM or SWA, with optional routing."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_swa = config.is_swa_layer(layer_idx)
        self.is_rwkv7 = False

        # Pre-norm
        Norm = _get_norm_class(config)
        self.norm = Norm(config.d_model)

        # Core block
        if self.is_swa:
            Norm = _get_norm_class(config)
            self.block = SlidingWindowAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                window_size=config.window_size,
                use_ternary=config.use_ternary,
                ternary_group_size=config.ternary_group_size,
                max_seq_len=config.seq_len,
                dropout=config.dropout,
                use_sdpa=config.use_sdpa,
                use_head_gating=getattr(config, 'use_head_gating', False),
                n_sink_tokens=config.n_sink_tokens if getattr(config, 'use_sink_tokens', False) else 0,
                use_phase_space=config.use_phase_space_attn,
                d_state=config.d_state,
                activation_fn=config.activation_fn,
                norm_class=Norm,
            )
            if hasattr(config, 'use_moa') and config.use_moa and config.moa_window_sizes is not None:
                heads = config.n_heads
                if len(config.moa_window_sizes) == heads:
                    self.block.per_head_windows = list(config.moa_window_sizes)
                else:
                    self.block.per_head_windows = [
                        config.moa_window_sizes[i % len(config.moa_window_sizes)]
                        for i in range(heads)
                    ]
                print(f"[MoA] Layer {layer_idx}: per-head window sizes = {self.block.per_head_windows}")
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
                if getattr(config, 'use_eigenmode_mixed', False):
                    self.block = EigenmodeMixedSSMSSD(
                        d_model=config.d_model,
                        d_state=config.d_state,
                        d_conv=config.d_conv,
                        expand=config.expand,
                        dt_rank=config.dt_rank,
                        n_heads=config.ssd_n_heads,
                        slow_fraction=config.eigenmode_slow_fraction,
                        use_ternary=config.use_ternary,
                        ternary_group_size=config.ternary_group_size,
                        dropout=config.dropout,
                        use_ssm_checkpoint=getattr(config, 'use_ssm_checkpoint', False),
                    )
                else:
                    is_lti = (getattr(config, 'use_ctri', False)
                              and not config.is_swa_layer(layer_idx)
                              and layer_idx not in getattr(config, 'ctri_selective_layers', [0]))
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
                        use_ssm_checkpoint=config.use_ssm_checkpoint,
                        config=config,
                        lti_mode=is_lti,
                    )
                    if config.use_ssd and hasattr(config, 'use_slow_heads') and config.use_slow_heads:
                        new_a_log = SelectiveSSMSSD._init_a_log(
                            config.ssd_n_heads,
                            use_slow_heads=True,
                            slow_fraction=config.slow_head_fraction,
                            medium_fraction=config.medium_head_fraction,
                        )
                        self.block.A_log.data.copy_(new_a_log)
                        print(f"[Q5-SlowHeads] Layer {layer_idx}: A_log initialized with slow/medium/fast bands")
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
                    use_ssm_checkpoint=config.use_ssm_checkpoint,
                )

        # Router (optional, only for SWA layers)
        self.router = None
        if config.use_routing and self.is_swa:
            self.router = MoDRouter(config.d_model, capacity=config.routing_capacity)

        # MoE FFN (optional, adds expert-routed SwiGLU FFN)
        self.moe_ffn = None
        self.moe_norm = None
        moe_indices = getattr(config, 'moe_layer_indices', None) or []
        if config.use_moe and layer_idx in moe_indices:
            d_ff = config.moe_d_ff if config.moe_d_ff > 0 else config.d_model
            Norm = _get_norm_class(config)
            self.moe_ffn = TernaryMoEFFN(
                d_model=config.d_model,
                d_ff=d_ff,
                num_experts=config.moe_num_experts,
                top_k=config.moe_top_k,
                use_ternary=config.use_ternary,
                ternary_group_size=config.ternary_group_size,
                load_balance_coeff=getattr(config, 'moe_load_balance_coeff', 0.01),
                activation_fn=config.activation_fn,
            )
            self.moe_norm = Norm(config.d_model)

    def _apply_moe(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MoE FFN sub-layer if configured."""
        if self.moe_ffn is not None:
            return x + self.moe_ffn(self.moe_norm(x))
        return x

    def forward(self, x: torch.Tensor, v_first: torch.Tensor = None, ssm_bc: dict = None, return_bc: bool = False, boundary_mask: torch.Tensor = None):
        residual = x
        x_norm = self.norm(x)

        if self.router is not None:
            mask, weights = self.router(x_norm)

            if self.is_swa:
                block_out = self.block(x, ssm_bc=ssm_bc) - x
            elif self.is_rwkv7:
                block_out, v_first = self.block(x_norm, v_first=v_first)
            else:
                block_out = self.block(x_norm, boundary_mask=boundary_mask)

            mask_ste = mask - weights.detach() + weights
            out = residual + mask_ste * block_out
            if self.is_rwkv7:
                return self._apply_moe(out), v_first
            return self._apply_moe(out)
        else:
            if self.is_swa:
                return self._apply_moe(self.block(x, ssm_bc=ssm_bc))
            elif self.is_rwkv7:
                block_out, v_first = self.block(x_norm, v_first=v_first)
                out = residual + block_out
                return self._apply_moe(out), v_first
            else:
                if return_bc:
                    ssm_result = self.block(x_norm, return_bc=True, boundary_mask=boundary_mask)
                    if isinstance(ssm_result, tuple):
                        ssm_out, bc_dict = ssm_result
                        return self._apply_moe(residual + ssm_out), bc_dict
                return self._apply_moe(residual + self.block(x_norm, boundary_mask=boundary_mask))


class ModelV2(nn.Module):
    """Model v2: Hybrid SSM-Attention with ternary weights and dynamic routing.

    Architecture:
        Embedding -> [N layers of SSM/SWA blocks] -> RMSNorm -> Output Head

    Properties:
        - ~87.5% SSM layers, ~12.5% SWA layers
        - Ternary weights for projections (when enabled)
        - MoD routing (when enabled)
        - Pre-norm architecture throughout
        - Tied embeddings (auto-untied when ternary_lm_head=True)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embedding (FP16 always)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

        # TADA defaults (set before branch so they always exist)
        self.tada_enabled = False
        self.tada_adaptive_k = False
        self.tada_alphas = None
        self.tada_rank_monitor_interval = 0
        self._tada_step_counter = 0

        # Layers
        if config.use_recursive_ssm:
            # RecursiveSSM: shared SSM block + independent SWA layers
            n_swa = len(config.swa_layer_indices)
            self.n_recursive_iterations = (
                config.recursive_ssm_iterations
                if config.recursive_ssm_iterations > 0
                else (config.n_layer - n_swa)
            )

            # Shared SSM block (one instance, called N times)
            self.shared_ssm_block = SelectiveSSMSSD(
                d_model=config.d_model, d_state=config.d_state,
                d_conv=config.d_conv, expand=config.expand,
                dt_rank=config.dt_rank, n_heads=config.ssd_n_heads,
                use_ternary=config.use_ternary,
                ternary_group_size=config.ternary_group_size,
                dropout=config.dropout,
                use_ssm_checkpoint=config.use_ssm_checkpoint,
                config=config,
            )
            self.shared_ssm_norm = RMSNorm(config.d_model)

            # Iteration embeddings -- learned per-iteration offset
            self.iteration_embeddings = nn.Parameter(
                torch.zeros(self.n_recursive_iterations, config.d_model)
            )
            nn.init.normal_(self.iteration_embeddings, std=0.02)

            # TADA: Per-iteration residual injection scalars
            self.tada_enabled = getattr(config, 'use_tada', False)
            if self.tada_enabled and getattr(config, 'tada_residual_injection', True):
                self.tada_alphas = nn.Parameter(
                    torch.ones(self.n_recursive_iterations) * 0.1
                )
                print(f"[TADA] Residual injection enabled: {self.n_recursive_iterations} learned alpha scalars")
            else:
                self.tada_alphas = None

            # TADA: Adaptive K components
            self.tada_adaptive_k = getattr(config, 'tada_adaptive_k', False) and self.tada_enabled
            if self.tada_adaptive_k:
                self.tada_k_max = (
                    config.tada_adaptive_k_max
                    if getattr(config, 'tada_adaptive_k_max', 0) > 0
                    else self.n_recursive_iterations
                )
                self.tada_difficulty_proj = nn.Linear(config.d_model, 1, bias=False)
                nn.init.normal_(self.tada_difficulty_proj.weight, std=0.01)
                print(f"[TADA] Adaptive K enabled: max_K={self.tada_k_max}")

            # TADA: Rank monitoring
            self.tada_rank_monitor_interval = getattr(config, 'tada_rank_monitor_interval', 0)
            self._tada_step_counter = 0

            # Per-Depth LR Scaling (R4-A3-F7)
            self._per_depth_lr_enabled = getattr(config, 'use_per_depth_lr', False)
            if self._per_depth_lr_enabled:
                min_scale = config.per_depth_lr_min
                early_end = config.per_depth_lr_early_end
                late_start = config.per_depth_lr_late_start
                if late_start == 0:
                    late_start = int(self.n_recursive_iterations * 0.66)
                scales = torch.ones(self.n_recursive_iterations)
                for i in range(self.n_recursive_iterations):
                    if i <= early_end:
                        scales[i] = 1.0
                    elif i >= late_start:
                        scales[i] = min_scale
                    else:
                        t = (i - early_end) / max(1, late_start - early_end)
                        scales[i] = 1.0 - t * (1.0 - min_scale)
                self.register_buffer('_depth_lr_scales', scales, persistent=False)
                print(f"[PerDepthLR] Enabled: early({early_end})=1.0, late({late_start}+)={min_scale:.2f}")

            # RecursiveSSM Iteration Load-Balancing (R4-A3-F6)
            self._recursive_lb_enabled = getattr(config, 'use_recursive_load_balance', False)
            self._recursive_lb_coeff = getattr(config, 'recursive_load_balance_coeff', 0.01)
            self._iteration_contrib_norms = []

            # Intermediate state collection for RCL/RAFU/GCP
            self._collect_recursive_states = getattr(config, 'collect_recursive_states', False)
            self._last_intermediate_states = []
            self._delta_hooks = []

            # UCB-Bandit Iteration Selection (R4-A3-P2)
            self._ucb_enabled = getattr(config, 'use_ucb_bandit', False)
            if self._ucb_enabled:
                self._ucb_c = getattr(config, 'ucb_exploration_c', 1.0)
                self._ucb_min_iters = min(
                    getattr(config, 'ucb_min_iterations', 4),
                    self.n_recursive_iterations,
                )
                self.register_buffer('_ucb_value_sum', torch.zeros(self.n_recursive_iterations), persistent=False)
                self.register_buffer('_ucb_count', torch.ones(self.n_recursive_iterations), persistent=False)
                self.register_buffer('_ucb_total_pulls', torch.tensor(float(self.n_recursive_iterations)), persistent=False)
                self._ucb_log_step = 0
                print(f"[UCB-Bandit] Enabled: min_iters={self._ucb_min_iters}, c={self._ucb_c}")

            # SWA layers remain independent
            self.layers = nn.ModuleList()
            for idx in config.swa_layer_indices:
                self.layers.append(ModelBlock(config, idx))

            # Build schedule: sequence of (type, index) pairs
            self._recursive_schedule = []
            ssm_iter = 0
            swa_mod_idx = 0
            for layer_idx in range(config.n_layer):
                if config.is_swa_layer(layer_idx):
                    self._recursive_schedule.append(('swa', swa_mod_idx))
                    swa_mod_idx += 1
                else:
                    self._recursive_schedule.append(('ssm', ssm_iter))
                    ssm_iter += 1

            print(f"[RecursiveSSM] {self.n_recursive_iterations} SSM iterations sharing 1 block, "
                  f"{n_swa} SWA layer(s) independent, "
                  f"iter_embed shape={self.iteration_embeddings.shape}")
        else:
            # Standard: independent layers
            self.layers = nn.ModuleList([
                ModelBlock(config, i) for i in range(config.n_layer)
            ])
            self._per_depth_lr_enabled = False
            self._ucb_enabled = False
            self._collect_recursive_states = False
            self._last_intermediate_states = []
            self._recursive_lb_enabled = False
            self._recursive_lb_coeff = 0.0
            self._iteration_contrib_norms = []

        # Final norm
        Norm = _get_norm_class(config)
        self.final_norm = Norm(config.d_model)

        # Output scaling (for muP coordinate check)
        self.output_multiplier = 1.0

        # Output head
        # When ternary_lm_head is requested, MUST untie embeddings because:
        # - Input embeddings stay FP16 (lookup table, no matmul)
        # - LM head needs TernaryLinear (matmul, ternary for fast inference)
        effective_tied = config.tied_embeddings
        if config.ternary_lm_head and config.use_ternary and config.tied_embeddings:
            effective_tied = False
            import warnings
            warnings.warn(
                "ternary_lm_head=True requires untied embeddings. "
                "Overriding tied_embeddings=True -> False for this model instance."
            )

        if effective_tied:
            self.lm_head = None  # Will use embedding.weight
        else:
            if config.ternary_lm_head and config.use_ternary:
                self.lm_head = TernaryLinear(config.d_model, config.vocab_size, bias=False, config=config)
            else:
                self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Multi-Token Prediction heads (F7 + M6)
        if config.mtp_num_heads > 0:
            self.mtp_heads = nn.ModuleList([
                nn.Linear(config.d_model, config.vocab_size, bias=False)
                for _ in range(config.mtp_num_heads)
            ])
        else:
            self.mtp_heads = None

        moe_layers = [i for i, layer in enumerate(self.layers) if layer.moe_ffn is not None]
        if moe_layers:
            print(f"[ModelV2] MoE enabled: {len(moe_layers)} layers with experts: {moe_layers}")
            for idx in moe_layers:
                moe = self.layers[idx].moe_ffn
                print(f"  Layer {idx}: {moe.num_experts} experts, top_k={moe.top_k}, d_ff={moe.d_ff}")

        if getattr(config, 'use_ctri', False):
            ssm_layers = [i for i in range(config.n_layer) if not config.is_swa_layer(i)]
            sel_layers = [i for i in ssm_layers if i in config.ctri_selective_layers]
            lti_layers = [i for i in ssm_layers if i not in config.ctri_selective_layers]
            print(f"[CTRI] Selective SSM layers: {sel_layers}")
            print(f"[CTRI] LTI (FFT conv training) layers: {lti_layers}")
            print(f"[CTRI] SWA layers: {list(config.swa_layer_indices)}")

        if getattr(config, 'activation_fn', 'silu') != 'silu' or getattr(config, 'use_bitshift_norm', False):
            parts = []
            if getattr(config, 'activation_fn', 'silu') == 'relu':
                parts.append('ReLU (replacing SiLU)')
            if getattr(config, 'use_bitshift_norm', False):
                parts.append('BitShiftNorm (replacing RMSNorm)')
            print(f"[FPIT] Enabled: {', '.join(parts)}")

        if getattr(config, 'ternary_aware_checkpointing', False):
            n_ssm = sum(1 for l in self.layers if not l.is_swa)
            n_swa = sum(1 for l in self.layers if l.is_swa)
            print(f"[TERNARY-CKPT] {n_ssm} SSM layers will RECOMPUTE | {n_swa} SWA layer(s) will STORE activations")

        # INSTANT: Low-rank activation compression for SWA layers (R2-A2-F6)
        if getattr(config, 'use_instant', False):
            instant_rank = getattr(config, 'instant_rank', 32)
            instant_gap = getattr(config, 'instant_update_gap', 50)
            instant_warmup = getattr(config, 'instant_warmup', 100)
            n_instant = 0
            for layer in self.layers:
                if layer.is_swa and hasattr(layer.block, 'enable_instant'):
                    layer.block.enable_instant(
                        rank=instant_rank,
                        update_gap=instant_gap,
                        warmup=instant_warmup
                    )
                    n_instant += 1
            if n_instant > 0:
                print(f"[INSTANT] {n_instant} SWA layer(s) configured with rank={instant_rank}")

        # FTEC: Factored Ternary Exponential Capacity
        if getattr(config, 'use_ftec', False):
            from .components.factored_ternary import replace_with_factored
            replaced = replace_with_factored(
                self,
                target_layers=config.ftec_target_layers,
                n_factors=config.ftec_n_factors,
                sparsity=config.ftec_sparsity,
                pattern=config.ftec_pattern,
                skip_connection=config.ftec_skip_connection,
                config=config
            )
            print(f"[FTEC] Replaced {replaced} TernaryLinear layers with "
                  f"FactoredTernaryLinear (K={config.ftec_n_factors}, c={config.ftec_sparsity}, "
                  f"pattern={config.ftec_pattern})")

        # Report ternary recomputation memory savings
        if config.use_ternary:
            ternary_params = 0
            ternary_layers = 0
            pwst_layers = 0
            for module in self.modules():
                if isinstance(module, TernaryLinear):
                    ternary_params += module.weight.numel()
                    ternary_layers += 1
                    if module.use_pwst:
                        pwst_layers += 1

            if ternary_params > 0:
                saved_mb = (ternary_params * 1) / (1024**2)
                if os.environ.get('VERBOSE_INIT'):
                    print(f"[RECOMPUTE] Ternary recompute active: saving {saved_mb:.1f}MB "
                          f"({ternary_layers} layers, {ternary_params/1e6:.1f}M params)")

            if pwst_layers > 0:
                if os.environ.get('VERBOSE_INIT'):
                    print(f"[PWST] Active on {pwst_layers} layers ({ternary_params/1e6:.1f}M params)")

    def _monitor_tada_rank(self):
        """Log effective rank of shared SSM block's ternary matrices."""
        if not self.tada_enabled or self.tada_rank_monitor_interval == 0:
            return
        if torch.compiler.is_compiling():
            return

        self._tada_step_counter += 1
        if self._tada_step_counter % self.tada_rank_monitor_interval != 0:
            return

        with torch.no_grad():
            for name, module in self.shared_ssm_block.named_modules():
                if hasattr(module, 'quantize_enabled') and hasattr(module, 'weight'):
                    w = module.weight.data
                    threshold = module.threshold_coeff * w.abs().mean()
                    w_ternary = torch.zeros_like(w)
                    w_ternary[w > threshold] = 1.0
                    w_ternary[w < -threshold] = -1.0

                    if w_ternary.shape[0] == w_ternary.shape[1]:
                        s = torch.linalg.svdvals(w_ternary.float())
                        eff_rank = (s > 0.01).sum().item()
                        max_rank = min(w_ternary.shape)
                        if not torch.compiler.is_compiling():
                            print(f"[TADA-Rank] {name}: eff_rank={eff_rank}/{max_rank} "
                                  f"({100*eff_rank/max_rank:.1f}%)")

    def _ucb_select_iterations(self) -> set:
        """Select iterations using UCB algorithm based on value estimation."""
        n = self.n_recursive_iterations
        mean_values = self._ucb_value_sum / self._ucb_count
        total = self._ucb_total_pulls.item()
        ucb_scores = mean_values + self._ucb_c * torch.sqrt(
            torch.log(torch.tensor(total + 1.0, device=mean_values.device)) / self._ucb_count
        )
        selected = {0, n - 1}
        k_remaining = max(self._ucb_min_iters - len(selected), 0)
        if k_remaining > 0:
            remaining_indices = [i for i in range(n) if i not in selected]
            remaining_scores = ucb_scores[remaining_indices]
            _, topk_idx = remaining_scores.topk(min(k_remaining, len(remaining_indices)))
            for idx in topk_idx:
                selected.add(remaining_indices[idx.item()])
        return selected

    def _ucb_update_stats(self, iter_idx: int, delta_norm: float):
        """Update UCB statistics for an iteration."""
        if self._ucb_enabled:
            self._ucb_value_sum[iter_idx] += delta_norm
            self._ucb_count[iter_idx] += 1
            self._ucb_total_pulls += 1

    def get_intermediate_states(self) -> list:
        """Get per-iteration hidden states from last forward pass (R4-A3-M3/M6)."""
        return self._last_intermediate_states

    def forward(self, input_ids: torch.Tensor, return_hidden_at: int = None, boundary_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token indices
            boundary_mask: (batch, seq_len) bool tensor for document boundaries (packing)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        if self._collect_recursive_states:
            self._last_intermediate_states = []

        x = self.embedding(input_ids)  # (B, L, d_model)

        v_first = None
        hidden_at_k = None
        ssm_bc = None
        _bc_layers = set()
        if self.config.use_phase_space_attn:
            for idx in range(len(self.layers)):
                if (not self.layers[idx].is_swa
                    and not getattr(self.layers[idx], 'is_rwkv7', False)
                    and idx + 1 < len(self.layers)
                    and self.layers[idx + 1].is_swa):
                    _bc_layers.add(idx)

        if self.config.use_recursive_ssm:
            # RecursiveSSM forward with optional adaptive depth early exit
            ad_enabled = (
                getattr(self.config, 'adaptive_depth_exit', False)
                and not self.training
            )
            if ad_enabled:
                ad_epsilon = self.config.adaptive_depth_epsilon
                ad_min_iters = self.config.adaptive_depth_min_iters
                converged = torch.zeros(
                    x.shape[0], x.shape[1], dtype=torch.bool, device=x.device
                )
                exit_iterations = torch.full(
                    (x.shape[0], x.shape[1]),
                    fill_value=-1, dtype=torch.long, device=x.device,
                )
                ssm_iter_count = 0

            # TADA: Store residual for per-iteration injection
            x_residual = None
            if self.tada_enabled:
                x_residual = x.clone()

            # TADA: Compute per-token difficulty for adaptive K
            token_k = None
            if self.tada_adaptive_k and self.training:
                with torch.no_grad():
                    difficulty = torch.sigmoid(self.tada_difficulty_proj(x)).squeeze(-1)
                    min_k = getattr(self.config, 'adaptive_depth_min_iters', 2)
                    token_k = (min_k + difficulty * (self.tada_k_max - min_k)).long()

            _anderson_enabled = getattr(self.config, 'use_anderson_acceleration', False) and not self.training
            _anderson_x_prev = None
            _anderson_beta = getattr(self.config, 'anderson_beta', 0.3)

            _geom_enabled = getattr(self.config, 'use_geometric_extrapolation', False) and not self.training
            _geom_prev_delta_norm = None
            _geom_extrapolated = False
            _geom_min_iters = getattr(self.config, 'geom_extrap_min_iters', 4)
            _geom_ratio_threshold = getattr(self.config, 'geom_extrap_ratio_threshold', 0.8)

            ucb_selected = None
            if self._ucb_enabled and self.training:
                ucb_selected = self._ucb_select_iterations()
                self._ucb_log_step += 1
                if self._ucb_log_step % 100 == 0:
                    mean_vals = (self._ucb_value_sum / self._ucb_count).tolist()
                    if not torch.compiler.is_compiling():
                        print(f"[UCB-Bandit] Step {self._ucb_log_step}: selected={sorted(ucb_selected)}")

            for step_type, step_idx in self._recursive_schedule:
                if step_type == 'ssm':
                    if ucb_selected is not None and step_idx not in ucb_selected:
                        continue

                    x_norm = self.shared_ssm_norm(x)
                    x_input = x_norm + self.iteration_embeddings[step_idx]

                    # A2-M7: Ternary-Aware Activation Checkpointing
                    # RecursiveSSM shared SSM block is always ternary (add/sub), cheap to recompute
                    should_cp = False
                    if self.training:
                        if self.config.ternary_aware_checkpointing:
                            should_cp = True
                        elif self.config.use_gradient_checkpointing and step_idx % self.config.checkpoint_every_n == 0:
                            should_cp = True

                    if should_cp:
                        delta = torch.utils.checkpoint.checkpoint(
                            self.shared_ssm_block, x_input, use_reentrant=True
                        )
                        if self._ucb_enabled:
                            self._ucb_update_stats(step_idx, delta.detach().norm().item())
                        if self._recursive_lb_enabled and self.training:
                            self._iteration_contrib_norms.append(delta.detach().norm())
                        if self._per_depth_lr_enabled:
                            delta = delta * self._depth_lr_scales[step_idx]
                        # TADA: Apply residual injection after checkpoint
                        if self.tada_alphas is not None:
                            if self.tada_adaptive_k and self.training:
                                active = (step_idx < token_k).unsqueeze(-1).float()
                                x_before_update = x.clone() if _anderson_enabled else None
                                x = x + (delta + self.tada_alphas[step_idx] * x_residual) * active
                            else:
                                x_before_update = x.clone() if _anderson_enabled else None
                                x = x + delta + self.tada_alphas[step_idx] * x_residual
                        else:
                            x_before_update = x.clone() if _anderson_enabled else None
                            x = x + delta
                            if _anderson_enabled and _anderson_x_prev is not None:
                                x = x + _anderson_beta * (x_before_update - _anderson_x_prev)
                            if _anderson_enabled:
                                _anderson_x_prev = x_before_update
                        if self._collect_recursive_states and self.training:
                            self._last_intermediate_states.append(x.detach())
                    else:
                        delta = self.shared_ssm_block(x_input)
                        if self._ucb_enabled:
                            self._ucb_update_stats(step_idx, delta.detach().norm().item())
                        if self._recursive_lb_enabled and self.training:
                            self._iteration_contrib_norms.append(delta.detach().norm())
                        if self._per_depth_lr_enabled:
                            delta = delta * self._depth_lr_scales[step_idx]

                        # TADA: Apply residual injection
                        if self.tada_alphas is not None:
                            if self.tada_adaptive_k and self.training:
                                active = (step_idx < token_k).unsqueeze(-1).float()
                                x_before_update = x.clone() if _anderson_enabled else None
                                x = x + (delta + self.tada_alphas[step_idx] * x_residual) * active
                            else:
                                x_before_update = x.clone() if _anderson_enabled else None
                                x = x + delta + self.tada_alphas[step_idx] * x_residual
                        elif ad_enabled:
                            ssm_iter_count += 1
                            x_before_update = x.clone() if _anderson_enabled else None
                            active_mask = (~converged).unsqueeze(-1).float()
                            x = x + delta * active_mask
                            if _anderson_enabled and _anderson_x_prev is not None:
                                x = x + _anderson_beta * (x_before_update - _anderson_x_prev)
                            if _anderson_enabled:
                                _anderson_x_prev = x_before_update

                            if ssm_iter_count >= ad_min_iters:
                                delta_norm = delta.detach().norm(dim=-1)
                                newly_converged = (~converged) & (delta_norm < ad_epsilon)
                                exit_iterations[newly_converged] = ssm_iter_count
                                converged = converged | newly_converged

                                if converged.all():
                                    current_idx = self._recursive_schedule.index(
                                        ('ssm', step_idx)
                                    )
                                    remaining = self._recursive_schedule[current_idx + 1:]
                                    for rem_type, rem_idx in remaining:
                                        if rem_type == 'swa':
                                            swa_layer = self.layers[rem_idx]
                                            x = swa_layer(x)
                                    total_tokens = converged.numel()
                                    early_exited = (exit_iterations >= 0).sum().item()
                                    avg_exit = exit_iterations[
                                        exit_iterations >= 0
                                    ].float().mean().item() if early_exited > 0 else float(ssm_iter_count)
                                    if not torch.compiler.is_compiling():
                                        print(
                                            f"[AdaptiveDepth] ALL converged at SSM iter "
                                            f"{ssm_iter_count}/{self.n_recursive_iterations} | "
                                            f"avg_exit_iter={avg_exit:.1f} | "
                                            f"early_exited={early_exited}/{total_tokens} "
                                            f"({100*early_exited/total_tokens:.1f}%)"
                                        )
                                    break
                        else:
                            x_before_update = x.clone() if _anderson_enabled else None
                            x = x + delta
                            if _anderson_enabled and _anderson_x_prev is not None:
                                x = x + _anderson_beta * (x_before_update - _anderson_x_prev)
                            if _anderson_enabled:
                                _anderson_x_prev = x_before_update
                            if _geom_enabled and not _geom_extrapolated:
                                delta_norm = delta.norm().item()
                                if _geom_prev_delta_norm is not None and step_idx >= _geom_min_iters:
                                    ratio = delta_norm / (_geom_prev_delta_norm + 1e-12)
                                    if 0 < ratio < _geom_ratio_threshold:
                                        correction = delta * (ratio / (1 - ratio))
                                        x = x + correction
                                        _geom_extrapolated = True
                                _geom_prev_delta_norm = delta_norm

                            if _geom_enabled and _geom_extrapolated:
                                continue
                        if self._collect_recursive_states and self.training:
                            self._last_intermediate_states.append(x.detach())

                elif step_type == 'swa':
                    swa_layer = self.layers[step_idx]
                    x = swa_layer(x)
            else:
                # for-else: runs if loop completed without break
                if ad_enabled:
                    total_tokens = converged.numel()
                    early_exited = (exit_iterations >= 0).sum().item()
                    avg_exit = exit_iterations[
                        exit_iterations >= 0
                    ].float().mean().item() if early_exited > 0 else float(self.n_recursive_iterations)
                    if not torch.compiler.is_compiling():
                        print(
                            f"[AdaptiveDepth] Completed all {self.n_recursive_iterations} "
                            f"SSM iters | early_exited={early_exited}/{total_tokens} "
                            f"({100*early_exited/total_tokens:.1f}%) | "
                            f"avg_exit_iter={avg_exit:.1f}"
                        )
        else:
            for i, layer in enumerate(self.layers):
                # A2-M7: Ternary-Aware Activation Checkpointing
                # F1: Selective checkpointing policy
                # SSM layers (ternary add/sub) are ~100x cheaper to recompute than SWA (attention).
                # Always recompute SSM, never recompute SWA.
                should_checkpoint = False
                if self.training:
                    granularity = getattr(self.config, 'activation_checkpoint_granularity', 'none')

                    if granularity == 'full':
                        # Checkpoint every layer
                        should_checkpoint = True
                    elif granularity == 'selective':
                        # F1: Only checkpoint SSM/non-attention layers (expensive ops)
                        should_checkpoint = not layer.is_swa
                    elif self.config.ternary_aware_checkpointing and not layer.is_swa:
                        # M7: always recompute SSM layers (cheap add/sub)
                        should_checkpoint = True
                    elif self.config.use_gradient_checkpointing and i % self.config.checkpoint_every_n == 0 and not layer.is_swa:
                        # Legacy: generic checkpointing (every Nth non-SWA layer)
                        should_checkpoint = True

                if should_checkpoint:
                    if getattr(layer, 'is_rwkv7', False):
                        x, v_first = grad_checkpoint(layer, x, v_first, use_reentrant=True)
                    else:
                        def _layer_forward(x_in):
                            return layer(x_in, boundary_mask=boundary_mask)
                        x = grad_checkpoint(_layer_forward, x, use_reentrant=True)
                else:
                    need_bc = (i in _bc_layers)

                    if getattr(layer, 'is_rwkv7', False):
                        x, v_first = layer(x, v_first)
                    elif need_bc:
                        x, ssm_bc = layer(x, v_first=None, ssm_bc=ssm_bc, return_bc=True, boundary_mask=boundary_mask)
                    else:
                        x = layer(x, v_first=None, ssm_bc=None, return_bc=False, boundary_mask=boundary_mask)
                        if layer.is_swa:
                            ssm_bc = None

                if return_hidden_at is not None and i == return_hidden_at:
                    hidden_at_k = x

        x = self.final_norm(x)

        # Output projection
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.embedding.weight)

        logits = logits * self.output_multiplier

        # Multi-Token Prediction (F7 + M6) - training only
        mtp_logits = None
        if self.mtp_heads is not None and self.training:
            mtp_logits = [head(x) for head in self.mtp_heads]

        if return_hidden_at is not None:
            if mtp_logits is not None:
                return (logits, hidden_at_k, mtp_logits)
            return logits, hidden_at_k
        if mtp_logits is not None:
            return logits, mtp_logits
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
            is_ternary_named = any(k in name for k in ['in_proj', 'out_proj', 'x_proj', 'q_proj', 'k_proj', 'v_proj', 'mlp_up', 'mlp_gate', 'mlp_down'])
            is_ternary_lm_head = 'lm_head' in name and self.lm_head is not None and isinstance(self.lm_head, TernaryLinear)

            if self.config.use_ternary and (is_ternary_named or is_ternary_lm_head):
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

    def get_aga_param_split(self, base_lr: float):
        """Split parameters into ternary weights (for AGA) and everything else (for AdamW).

        Returns:
            (ternary_weight_params, non_ternary_param_groups)
        """
        ternary_weight_ids = set()
        for module in self.modules():
            if hasattr(module, 'quantize_enabled') and hasattr(module, 'group_size'):
                ternary_weight_ids.add(id(module.weight))

        ternary_weights = []
        router_params = []
        a_params = []
        embed_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in ternary_weight_ids:
                ternary_weights.append(param)
            elif 'router' in name:
                router_params.append(param)
            elif 'A_log' in name or 'A_shadow' in name:
                a_params.append(param)
            elif 'embedding' in name or 'lm_head' in name:
                embed_params.append(param)
            else:
                other_params.append(param)

        non_ternary_groups = []
        if other_params:
            non_ternary_groups.append({'params': other_params, 'lr': base_lr})
        if router_params:
            non_ternary_groups.append({'params': router_params, 'lr': base_lr * 1.0, 'name': 'router'})
        if a_params:
            non_ternary_groups.append({'params': a_params, 'lr': base_lr * 0.1, 'name': 'a_matrix'})
        if embed_params:
            non_ternary_groups.append({'params': embed_params, 'lr': base_lr * 0.5, 'weight_decay': 0.0, 'name': 'embedding'})

        return ternary_weights, non_ternary_groups

    def set_quantization_enabled(self, enabled: bool):
        """Enable or disable ternary quantization across all TernaryLinear layers."""
        for module in self.modules():
            if hasattr(module, 'quantize_enabled'):
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
                if hasattr(module, 'quantize_enabled'):
                    if i in keep_fp16_layers:
                        module.quantize_enabled = False
                    else:
                        module.quantize_enabled = enabled

    def set_routing_enabled(self, enabled: bool):
        """Enable or disable token routing across all MoDRouter layers."""
        for module in self.modules():
            if hasattr(module, 'routing_enabled'):
                module.routing_enabled = enabled

    def set_ste_type(self, ste_type: str):
        """Set gradient estimator type across all TernaryLinear layers."""
        for module in self.modules():
            if hasattr(module, 'ste_type'):
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
            if hasattr(module, 'training_step'):
                module.training_step = step
                module.total_training_steps = total_steps
                if hasattr(module, 'ste_type') and module.ste_type == 'hestia':
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
            if hasattr(module, 'quantize_enabled'):
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

    def set_gradual_threshold(self, enabled: bool, start: float = 0.5, end: float = DEFAULT_THRESHOLD_COEFF):
        """Enable gradual threshold softening across all TernaryLinear layers."""
        for module in self.modules():
            if hasattr(module, 'threshold_coeff'):
                module.gradual_threshold = enabled
                module.threshold_start = start
                module.threshold_end = end

    def set_pressure_blend(self, enabled: bool, duration: float = 0.2):
        """Enable pressure blending (gradual FP->quantized) across all TernaryLinear layers."""
        for module in self.modules():
            if hasattr(module, 'pressure_blend'):
                module.pressure_blend = enabled
                module.pressure_duration = duration

    def compute_contractive_reg(self) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        target = self.config.contractive_target
        for module in self.modules():
            if hasattr(module, 'A_log') and not getattr(module, 'use_tscs', False):
                A_mag = torch.exp(module.A_log)
                decay_at_unit_dt = torch.exp(-A_mag)
                violation = torch.clamp(decay_at_unit_dt - target, min=0.0)
                penalty = penalty + (violation ** 2).sum()
        return penalty

    def compute_frar_loss(self) -> torch.Tensor:
        """
        Fast Residual Alignment Regularization (F12).

        Minimizes L1 gap between FP32 shadow weights and ternary quantization.
        Encourages shadow weights to align with their quantized values,
        reducing oscillation near thresholds during training.

        Returns:
            Scalar alignment loss (mean absolute difference across ternary layers)
        """
        total_alignment = 0.0
        num_ternary_layers = 0

        for module in self.modules():
            if hasattr(module, 'ste_type'):
                w_shadow = module.weight

                with torch.no_grad():
                    group_size = getattr(module, 'group_size', 128)
                    threshold_coeff = getattr(module, 'threshold_coeff', DEFAULT_THRESHOLD_COEFF)

                    weight_2d = w_shadow.view(-1, group_size) if w_shadow.numel() >= group_size else w_shadow.view(1, -1)
                    abs_normalized = weight_2d.abs()
                    threshold = threshold_coeff * abs_normalized.mean(dim=-1, keepdim=True)

                    w_ternary = torch.where(
                        weight_2d > threshold,
                        torch.ones_like(weight_2d),
                        torch.where(
                            weight_2d < -threshold,
                            -torch.ones_like(weight_2d),
                            torch.zeros_like(weight_2d)
                        )
                    ).view(w_shadow.shape)

                alignment = (w_shadow - w_ternary).abs().mean()
                total_alignment = total_alignment + alignment
                num_ternary_layers += 1

        if num_ternary_layers == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        return total_alignment / num_ternary_layers

    def compute_recursive_load_balance_loss(self) -> torch.Tensor:
        """R4-A3-F6: Penalize uneven contribution across RecursiveSSM iterations.
        L_aux = coeff * sum_i (f_i - 1/E)^2 where f_i = norm_i / sum(norms).
        """
        if not getattr(self, '_recursive_lb_enabled', False) or len(self._iteration_contrib_norms) == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        norms = torch.stack(self._iteration_contrib_norms)
        total = norms.sum()
        if total < 1e-8:
            self._iteration_contrib_norms = []
            return torch.tensor(0.0, device=norms.device)
        fractions = norms / total
        E = len(norms)
        target = 1.0 / E
        loss = self._recursive_lb_coeff * ((fractions - target) ** 2).sum()
        self._iteration_contrib_norms = []
        return loss

    def reinit_ternary_standard(self):
        """Re-init TernaryLinear weights with standard Xavier (for progressive training FP32 warmup)."""
        for module in self.modules():
            if hasattr(module, 'ste_type'):
                nn.init.xavier_uniform_(module.weight, gain=1.0)

    @torch.no_grad()
    def generate(self, prompt_ids, max_tokens=128, temperature=1.0, top_k=50, top_p=0.9,
                 use_triton=False):
        from .inference import generate as _generate, prepare_for_inference
        if use_triton:
            device = next(self.parameters()).device
            prepare_for_inference(self, device=device, backend='triton', verbose=False)
        return _generate(self, prompt_ids, max_tokens, temperature, top_k, top_p)
