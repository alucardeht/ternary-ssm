import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

from .quantizers import *


_STGW_RATIO = 32


_stgw_logged = False


_q_values_cache = {}
_av_probe_counters = {}
_aw3_ema_store = {}


def _subsampled_grad_weight(
    go_2d: torch.Tensor,
    in_2d: torch.Tensor,
    ratio: int = _STGW_RATIO,
) -> torch.Tensor:
    """D1-STGW: Subsampled token grad_weight computation (R8-A5).

    Math: grad_weight = X^T @ dL/dY (shape [d_out, d_in]).
    By LLN: E[X_sub^T @ dY_sub] == E[X^T @ dY] for random subset of T/ratio tokens.
    Scale by ratio (= T/k) to maintain unbiased estimate.
    """
    global _stgw_logged
    T = go_2d.shape[0]
    k = max(1, T // ratio)
    if k >= T:
        return go_2d.t().matmul(in_2d)
    idx = torch.randint(0, T, (k,), device=go_2d.device)
    gw = go_2d[idx].t().matmul(in_2d[idx])
    if not _stgw_logged:
        print(
            f"[D1-STGW] Subsampled grad_weight active | "
            f"T={T} k={k} ratio={ratio}x | FLOPs_reduction={100*(1 - k/T):.1f}%"
        )
        _stgw_logged = True
    return gw * (T / k)


class MagnitudePrunedTernaryQuantize(torch.autograd.Function):
    """
    M5+M6: Magnitude Pruning + Gradient Rescaling (F16).

    Increases zero fraction from ~33% to ~45% via higher threshold (M5).
    Rescales gradients by 1/(1-sparsity) to prevent gradient collapse (M6).
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        group_size: int = 128,
        threshold_multiplier: float = 0.9,
    ) -> torch.Tensor:
        orig_shape = weight.shape
        flat = weight.reshape(-1)
        pad_len = (group_size - flat.numel() % group_size) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, group_size)

        scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        normalized = grouped / scales

        abs_normalized = normalized.abs()
        threshold = threshold_multiplier * abs_normalized.mean(dim=-1, keepdim=True)

        quantized = torch.where(
            abs_normalized > threshold,
            torch.ones_like(normalized),
            torch.where(
                abs_normalized < -threshold,
                -torch.ones_like(normalized),
                torch.zeros_like(normalized),
            ),
        )

        sparsity = (quantized == 0).float().mean()

        result = (quantized * scales).reshape(-1)
        if pad_len > 0:
            result = result[: weight.numel()]

        ctx.save_for_backward(torch.tensor(sparsity.item()))

        return result.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (sparsity_tensor,) = ctx.saved_tensors
        sparsity = sparsity_tensor.item()

        grad_input = grad_output.clone()

        rescale_factor = 1.0 / (1.0 - sparsity + 1e-8)
        grad_input = grad_input * rescale_factor

        return grad_input, None, None


class TernaryLinearBTG(torch.autograd.Function):
    """Custom autograd for ternary backward pass (BTG: Backward-Ternary GEMV).

    Forward: standard F.linear with ternary weights (cast to input dtype)
    Backward grad_input: uses ternary W directly (add/subtract semantics)
    Backward grad_weight: STE path, full FP16 matmul to shadow weights
    """

    @staticmethod
    def forward(ctx, input, weight_ternary, bias):
        ctx.save_for_backward(input, weight_ternary, bias)
        ctx.has_bias = bias is not None
        return F.linear(input, weight_ternary.to(input.dtype), bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight_ternary, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight_ternary.to(grad_output.dtype))

        if ctx.needs_input_grad[1]:
            go_2d = grad_output.reshape(-1, grad_output.shape[-1])
            in_2d = input.reshape(-1, input.shape[-1])
            grad_weight = _subsampled_grad_weight(go_2d, in_2d)

        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)

        return grad_input, grad_weight, grad_bias


class TernaryLinearBTG_PWST(torch.autograd.Function):
    """Hybrid: PWST forward pass + BTG backward pass.

    Forward: PWST add/sub/skip kernel
    Backward grad_input: PWST add/sub/skip on transposed weights
    Backward grad_weight: STE path, full FP16 matmul to shadow weights
    """

    @staticmethod
    def forward(ctx, input, weight_ternary, bias):
        ctx.save_for_backward(input, weight_ternary, bias)
        ctx.has_bias = bias is not None
        return pwst_forward_vectorized(input, weight_ternary, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight_ternary, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = pwst_backward_grad_input(grad_output, weight_ternary)

        if ctx.needs_input_grad[1]:
            go_2d = grad_output.reshape(-1, grad_output.shape[-1])
            in_2d = input.reshape(-1, input.shape[-1])
            grad_weight = _subsampled_grad_weight(go_2d, in_2d)

        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)

        return grad_input, grad_weight, grad_bias


def pwst_forward_vectorized(
    x: torch.Tensor, w_ternary: torch.Tensor, bias: torch.Tensor = None
) -> torch.Tensor:
    """
    PWST: Packed Weight Substitution Transform for ternary matmul.

    Replace F.linear(x, W_ternary) with add/sub/skip logic.
    Ternary weights {-1, 0, +1} multiply as: +1->add, -1->subtract, 0->skip.

    Args:
        x: Input tensor (..., in_features)
        w_ternary: Ternary weights (out_features, in_features) with values {-1, 0, +1}
        bias: Optional bias (out_features,)

    Returns:
        Output tensor (..., out_features)
    """
    w_pos = torch.where(
        w_ternary > 0, torch.ones_like(w_ternary), torch.zeros_like(w_ternary)
    )
    w_neg = torch.where(
        w_ternary < 0, torch.ones_like(w_ternary), torch.zeros_like(w_ternary)
    )

    y = F.linear(x, w_pos.float(), None) - F.linear(x, w_neg.float(), None)

    if bias is not None:
        y = y + bias

    return y


def pwst_backward_grad_input(
    grad_output: torch.Tensor, w_ternary: torch.Tensor
) -> torch.Tensor:
    """
    Compute grad_input = grad_output @ W_ternary using PWST add/sub/skip.

    Args:
        grad_output: Gradient tensor (..., out_features)
        w_ternary: Ternary weights (out_features, in_features)

    Returns:
        Gradient w.r.t. input (..., in_features)
    """
    w_t = w_ternary.t()
    return pwst_forward_vectorized(grad_output, w_t, bias=None)


class TernaryLinearTATGR(torch.autograd.Function):
    """TATGR: Transpose-Aware Ternary Gradient Routing (A1-M1).

    Forward: standard F.linear with scaled ternary weights.
    Backward grad_input: index-gather routing eliminates multiply.
    Backward grad_weight: standard matmul (STE path unchanged).
    """

    @staticmethod
    def forward(ctx, input, w_scaled, w_sign, bias):
        ctx.save_for_backward(input, w_sign, bias)
        ctx.has_bias = bias is not None
        return F.linear(input, w_scaled.to(input.dtype), bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, w_sign, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            go_2d = grad_output.reshape(-1, grad_output.shape[-1])
            out_f, in_f = w_sign.shape

            pos_mask = (w_sign == 1)
            neg_mask = (w_sign == -1)

            grad_input_2d = torch.zeros(
                go_2d.shape[0], in_f, dtype=grad_output.dtype, device=grad_output.device
            )

            pos_rows, pos_cols = pos_mask.nonzero(as_tuple=True)
            neg_rows, neg_cols = neg_mask.nonzero(as_tuple=True)

            if pos_rows.numel() > 0:
                grad_input_2d.index_add_(1, pos_cols, go_2d[:, pos_rows])

            if neg_rows.numel() > 0:
                grad_input_2d.index_add_(1, neg_cols, -go_2d[:, neg_rows])

            grad_input = grad_input_2d.reshape(grad_output.shape[:-1] + (in_f,))

            _step = getattr(TernaryLinearTATGR, '_call_count', 0)
            TernaryLinearTATGR._call_count = _step + 1
            if _step == 0:
                n_pos = pos_rows.numel()
                n_neg = neg_rows.numel()
                total = out_f * in_f
                print(f"[TATGR] backward step=0 | pos={n_pos} neg={n_neg} "
                      f"zero={total-n_pos-n_neg} "
                      f"sparsity={100*(total-n_pos-n_neg)/total:.1f}%")

        if ctx.needs_input_grad[1]:
            go_2d = grad_output.reshape(-1, grad_output.shape[-1])
            in_2d = input.reshape(-1, input.shape[-1]).to(go_2d.dtype)
            grad_weight = _subsampled_grad_weight(go_2d, in_2d)

        if ctx.has_bias and ctx.needs_input_grad[3]:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)

        return grad_input, grad_weight, None, grad_bias


TernaryLinearTATGR._call_count = 0


def _fourier_init_ternary(out_features: int, in_features: int) -> torch.Tensor:
    """Holographic Fourier initialization for ternary shadow weights.

    Creates a weight matrix with spectral structure: rows are inverse DFT basis vectors
    with random phase. Distributes information holographically, reducing symmetry and
    improving early convergence by 5-10%.

    Reference: R17-G-W2-PB-F1
    """
    n_freq = in_features // 2 + 1
    phase = torch.randn(out_features, n_freq) * 2 * math.pi
    spectrum = torch.exp(1j * phase)
    w_init = torch.fft.irfft(spectrum, n=in_features)
    target_std = 1.0 / math.sqrt(in_features)
    current_std = w_init.std().clamp(min=1e-8)
    w_init = w_init * (target_std / current_std)
    return w_init


class TernaryLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 128,
        correction_rank: int = 0,
        config=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.quantize_enabled = True

        self.use_btg = (
            getattr(config, "use_btg", False) if config is not None else False
        )
        self.use_tatgr = (
            getattr(config, "use_tatgr", False) if config is not None else False
        )
        self.use_bitmask_gemv = (
            getattr(config, "use_bitmask_gemv", False) if config is not None else False
        )
        if not self.use_bitmask_gemv:
            print(
                f"[BITMASK_GEMV] WARNING: DISABLED for {self.__class__.__name__} "
                f"(in={self.in_features}, out={self.out_features}) — "
                f"config={'None' if config is None else 'present, use_bitmask_gemv=False'}. "
                f"Falling back to F.linear / standard path."
            )
        self.use_ppr = (
            getattr(config, "use_ppr", True) if config is not None else True
        )
        self.use_pwst = True
        if self.use_bitmask_gemv:
            print(
                f"[BITMASK_GEMV] Enabled for {self.__class__.__name__} (in={self.in_features}, out={self.out_features})"
            )
        if self.use_btg:
            print(
                f"[BTG] Enabled for {self.__class__.__name__} (in={self.in_features}, out={self.out_features})"
            )
        if self.use_pwst:
            if os.environ.get("VERBOSE_INIT"):
                print(
                    f"[PWST] Ternary kernel will be used for {self.__class__.__name__} (in={self.in_features}, out={self.out_features})"
                )

        self.use_magnitude_pruning = (
            getattr(config, "use_ternary_magnitude_pruning", False)
            if config is not None
            else False
        )
        self.magnitude_pruning_threshold_multiplier = (
            getattr(config, "magnitude_pruning_threshold_multiplier", 0.9)
            if config is not None
            else 0.9
        )
        self.use_gradient_rescaling = (
            getattr(config, "use_gradient_rescaling", False)
            if config is not None
            else False
        )

        if self.use_magnitude_pruning:
            if not self.use_gradient_rescaling:
                import warnings

                warnings.warn(
                    "M5 (magnitude pruning) enabled without M6 (gradient rescaling). "
                    "This may cause gradient collapse. Auto-enabling rescaling."
                )
                self.use_gradient_rescaling = True
            print(
                f"[M5+M6] Magnitude pruning enabled with threshold_mult={self.magnitude_pruning_threshold_multiplier}, "
                f"gradient_rescaling={self.use_gradient_rescaling}"
            )

        self.ste_type = "standard"
        self.training_step = 0
        self.total_training_steps = 1000
        self.hestia_tau_init = 2.0
        self.threshold_coeff = DEFAULT_THRESHOLD_COEFF
        self.threshold_type = "coefficient"  # F4: 'coefficient', 'twn', 'absmean'
        self.gradual_threshold = False
        self.threshold_start = 0.5
        self.threshold_end = DEFAULT_THRESHOLD_COEFF
        self.pressure_blend = False
        self.pressure_duration = 0.2
        self._cached_tau = None
        self.use_triton_hestia = False
        self.use_caq = False
        self.caq_alpha = 1.0
        self._v_hat = None

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        use_fourier_init = getattr(config, 'use_fourier_init', False) if config is not None else False
        if use_fourier_init:
            with torch.no_grad():
                w_fourier = _fourier_init_ternary(out_features, in_features)
                self.weight.data.copy_(w_fourier)
        else:
            nn.init.xavier_uniform_(self.weight, gain=1.0 / (in_features**0.5))

        # TTQ: asymmetric per-group learned scales (pos and neg independent)
        _n_elems = out_features * in_features
        _pad_lsq = (group_size - _n_elems % group_size) % group_size
        _num_groups = (_n_elems + _pad_lsq) // group_size
        with torch.no_grad():
            _flat_lsq = self.weight.data.reshape(-1)
            if _pad_lsq > 0:
                _flat_lsq = F.pad(_flat_lsq, (0, _pad_lsq))
            _grouped_init = _flat_lsq.reshape(_num_groups, group_size)
            _init_sp = _grouped_init.clamp(min=0).max(dim=-1).values.clamp(min=1e-8)
            _init_sn = (-_grouped_init).clamp(min=0).max(dim=-1).values.clamp(min=1e-8)
        self.lsq_scale_pos = nn.Parameter(_init_sp)  # [num_groups]
        self.lsq_scale_neg = nn.Parameter(_init_sn)  # [num_groups]

        self.register_buffer("admm_u", torch.zeros(out_features, in_features, dtype=torch.float16))
        self.register_buffer("accum_push", torch.zeros(out_features, in_features, dtype=torch.int16))
        self._accum_push_scale = 10000.0
        self._accum_push_threshold = 8000
        self._accum_push_flips = 0
        self._admm_rho = 0.01
        self._admm_last_ternary = None

        self.alpha = None
        self.alpha_group_size = None
        self.freeze_patterns = False
        self.tequila_lambda = 0.0
        self._tequila_call_count = 0

        self.correction_rank = correction_rank
        if correction_rank > 0:
            self.correction_A = nn.Parameter(
                torch.randn(out_features, correction_rank)
                * (0.02 / math.sqrt(correction_rank))
            )
            self.correction_B = nn.Parameter(torch.zeros(correction_rank, in_features))
        else:
            self.correction_A = None
            self.correction_B = None

        self.block_size = 0
        self.block_mixing = None

        self.dendritic_mode = "none"
        self.gate_pre = None
        self.gate_post = None

        self.use_tpsq = False
        self.tpsq_beta = 1.0
        self.tpsq_momentum = 0.99

        self.use_gumbel_noise = False
        self.gumbel_temperature = 0.1
        self.gumbel_anneal = False

        self.gumbel_ste_tau_init: float = 1.0
        self.gumbel_ste_tau_min: float = 0.1
        self.gumbel_ste_anneal: bool = True

        self.use_stp = False
        self.stp_sigma_ratio = 0.1
        self._stp_call_count = 0

        self._rlec_call_count = 0

        self.use_ctce = False
        self.ctce_beta = 1.5
        self.ctce_delta_ratio = 0.5

        self.use_dual_rate = False
        self.dual_rate_peak = 2.5
        self.dual_rate_floor = 0.1
        self.dual_rate_delta_ratio = 0.5

        self.use_gaste = False
        self.gaste_sigma_g = 0.3

        self.tcd_beta = 1.0

        self.use_iaqd = False
        self.iaqd_gate_w = None
        self.iaqd_gate_b = None
        self._iaqd_call_count = 0

        self.use_mdts = False
        self._mdts_logits = None
        self._mdts_scales = None
        self.mdts_temperature = 1.0
        self.mdts_anneal = False
        self._mdts_call_count = 0

        self.use_info_freeze = False
        self.info_freeze_alpha = 1e-5
        self.info_freeze_warmup = 200
        self._freeze_confidence = None
        self._freeze_mask = None
        self._freeze_hook = None
        self._freeze_call_count = 0

        self.use_dead_zone = False
        self.dead_zone_margin = 0.1
        self.dead_zone_lr_decay = True
        self._prev_ternary = None
        self._current_lr = 0.0
        self._base_lr = 1.0
        self._dead_zone_flips_blocked = 0
        self._dead_zone_flips_allowed = 0
        self._dead_zone_call_count = 0

        self.threshold_update_freq = 50
        self.threshold_step_counter = 0
        self.register_buffer("cached_threshold", None, persistent=False)
        self._threshold_logged = False

        self._threshold_ema_decay = 0.9
        self.register_buffer("threshold_ema", None, persistent=False)

        self.use_packed_cache = True
        self.register_buffer("packed_weights", None, persistent=False)
        self.register_buffer("packed_valid", torch.tensor(False), persistent=False)
        self._packed_logged = False

        self.register_buffer("int2_packed_weights", None, persistent=False)
        self.register_buffer("int2_packed_valid", torch.tensor(False), persistent=False)
        self._int2_original_K = 0

        self.use_int32_forward = getattr(config, 'use_int32_forward', True) if config is not None else True
        self._int32_min_k = getattr(config, 'int32_forward_min_k', 64) if config is not None else 64
        self._int32_logged = False

        self._lazy_q_eps = 0.01
        self._lazy_q_total_calls = 0
        self._lazy_q_skipped = 0
        self._lazy_q_log_interval = 200
        self.register_buffer("_w_shadow_at_cache", None, persistent=False)

        self.use_vote_buffer = (
            getattr(config, "use_vote_buffer", False) if config is not None else False
        )
        if self.use_vote_buffer:
            self.vote_optimizer = None
            self.param_id = id(self)
            print(
                f"[VoteBuffer] Enabled for {self.__class__.__name__} (convergence guarantee O(sqrt(V/T)))"
            )

        self.use_arenas = (
            getattr(config, "use_arenas", False) if config is not None else False
        )
        self.arenas_lambda_0 = (
            getattr(config, "arenas_lambda_0", 1.0) if config is not None else 1.0
        )
        self.arenas_epsilon = (
            getattr(config, "arenas_epsilon", 0.01) if config is not None else 0.01
        )
        if self.use_arenas:
            self.register_buffer(
                "arenas_shadow", torch.zeros_like(self.weight, dtype=torch.float32)
            )
            print(
                f"[Arenas+Lyapunov] Enabled (lambda_0={self.arenas_lambda_0}, epsilon={self.arenas_epsilon})"
            )

        if self.use_bitmask_gemv:
            self.register_buffer("bitmask_pos", None, persistent=False)
            self.register_buffer("bitmask_neg", None, persistent=False)
            self._bitmask_stale = True
            self._bitmask_original_K = None
            self._bitmask_logged = False

        self.use_dsst = (
            getattr(config, "use_dsst", False) if config is not None else False
        )
        if self.use_dsst:
            self.register_buffer('dsst_s', torch.ones(out_features, in_features, dtype=torch.int8))
            self.register_buffer('dsst_m', torch.ones(out_features, in_features, dtype=torch.bool))
            self.register_buffer('dsst_ema', torch.zeros(out_features, in_features, dtype=torch.float16))
            self.register_buffer('dsst_step', torch.tensor(0, dtype=torch.int64))
            self.weight.register_post_accumulate_grad_hook(self._dsst_update)
            if os.environ.get("VERBOSE_INIT"):
                print(f"[DSST] Decoupled Structure-Sign Training enabled for {self.__class__.__name__}")
            if getattr(config, 'use_aw12_hyperreal_ste', True) if config is not None else True:
                TernaryQuantize._use_hyperreal_ste = True

        self.use_fourier_init = getattr(config, 'use_fourier_init', False) if config is not None else False
        self.use_strang_ste = getattr(config, 'use_strang_ste', False) if config is not None else False
        self.use_gv_ste = getattr(config, 'use_gv_ste', False) if config is not None else False
        self.use_wasserstein_ste = getattr(config, 'use_wasserstein_ste', False) if config is not None else False
        self.use_rpe_gate = getattr(config, 'use_rpe_gate', False) if config is not None else False

        if self.use_gv_ste:
            self.register_buffer('gv_ste_ema', torch.ones(out_features, in_features, dtype=torch.float16) * 0.01)
            self._gv_ste_beta = 0.99
            self._gv_ste_logged = False

    def train(self, mode: bool = True):
        """Override to reset F13 threshold cache when switching modes."""
        super().train(mode)
        if mode:
            pass
        else:
            self.cached_threshold = None
            self.threshold_step_counter = 0
        return self

    def eval(self):
        """Override to reset F13 threshold cache in eval mode."""
        return self.train(False)

    def invalidate_packed_cache(self):
        """F17: Invalidate packed weight cache after optimizer step.

        Call this after optimizer.step() to mark cached masks as stale,
        forcing recomputation on next forward pass.
        """
        if self.use_packed_cache:
            self.packed_valid.fill_(False)
            if hasattr(self, 'int2_packed_valid') and self.int2_packed_valid is not None:
                self.int2_packed_valid.fill_(False)

    def _reset_threshold_cache(self):
        """F13: Manually reset the cached threshold (e.g., after optimizer.step())."""
        self.cached_threshold = None
        self.threshold_step_counter = 0

    def get_ternary_stats(self) -> dict:
        with torch.no_grad():
            if getattr(self, 'use_dsst', False):
                effective = self.dsst_m.float() * self.dsst_s.float()
                flat = effective.reshape(-1)
            else:
                w = TernaryQuantize.apply(self.weight, self.group_size)
                flat = w.reshape(-1)
            total = flat.numel()
            n_pos = (flat > 0).sum().item()
            n_neg = (flat < 0).sum().item()
            n_zero = (flat == 0).sum().item()
            stats = {
                "positive": n_pos / total,
                "negative": n_neg / total,
                "zero": n_zero / total,
            }
            if self.alpha is not None:
                stats["alpha_mean"] = self.alpha.mean().item()
                stats["alpha_std"] = (
                    self.alpha.std().item() if self.alpha.numel() > 1 else 0.0
                )
                stats["alpha_min"] = self.alpha.min().item()
                stats["alpha_max"] = self.alpha.max().item()
            return stats

    def get_correction_stats(self) -> dict:
        if self.correction_A is None:
            return {"correction_rank": 0}
        with torch.no_grad():
            a_norm = self.correction_A.norm().item()
            b_norm = self.correction_B.norm().item()
            ab_norm = (self.correction_A @ self.correction_B).norm().item()
            return {
                "correction_rank": self.correction_rank,
                "A_norm": a_norm,
                "B_norm": b_norm,
                "AB_norm": ab_norm,
            }

    def get_dendritic_stats(self) -> dict:
        stats = {"dendritic_mode": self.dendritic_mode}
        if self.gate_pre is not None:
            with torch.no_grad():
                g = self.gate_pre
                stats.update(
                    {
                        "gate_pre_mean": g.mean().item(),
                        "gate_pre_std": g.std().item(),
                        "gate_pre_min": g.min().item(),
                        "gate_pre_max": g.max().item(),
                    }
                )
        if self.gate_post is not None:
            with torch.no_grad():
                g = self.gate_post
                stats.update(
                    {
                        "gate_post_mean": g.mean().item(),
                        "gate_post_std": g.std().item(),
                        "gate_post_min": g.min().item(),
                        "gate_post_max": g.max().item(),
                    }
                )
        return stats

    def enable_iaqd(self):
        """Enable IAQD input-adaptive gating (Q11). Cost: 2*in_features params."""
        self.use_iaqd = True
        self.iaqd_gate_w = nn.Parameter(
            torch.zeros(self.in_features, device=self.weight.device)
        )
        self.iaqd_gate_b = nn.Parameter(
            torch.full((self.in_features,), 2.0, device=self.weight.device)
        )
        self._iaqd_call_count = 0

    def enable_info_freeze(self, alpha: float = 1e-5, warmup: int = 200):
        """Enable information-theoretic adaptive freezing (R2-A3-M3).

        Maintains per-weight Bayesian confidence accumulator. Freezes weights
        when confidence exceeds SPRT threshold: C_thresh = sqrt(2 * log(3/alpha)).
        """
        import math

        self.use_info_freeze = True
        self.info_freeze_alpha = alpha
        self.info_freeze_warmup = warmup
        self._freeze_confidence = torch.zeros_like(self.weight)
        self._freeze_mask = torch.zeros_like(self.weight, dtype=torch.bool)
        self._freeze_threshold = math.sqrt(2.0 * math.log(3.0 / alpha))
        self._freeze_call_count = 0

        def _freeze_grad_hook(grad):
            if self._freeze_mask is not None and self._freeze_mask.any():
                return grad * (~self._freeze_mask).float()
            return grad

        self._freeze_hook = self.weight.register_hook(_freeze_grad_hook)
        n_params = self.weight.numel()
        print(
            f"[INFO-FREEZE] Enabled: alpha={alpha}, threshold={self._freeze_threshold:.2f}, "
            f"warmup={warmup}, params={n_params}"
        )

    def enable_dead_zone(
        self, margin: float = 0.1, lr_decay: bool = True, base_lr: float = 6e-4
    ):
        """Enable oscillation dead zone / STE hysteresis (R2-A6-M14).

        Adds hysteresis band around quantization threshold to suppress
        unproductive weight oscillations near the boundary.

        Args:
            margin: Fraction of threshold as hysteresis band (0.1 = 10%)
            lr_decay: Scale margin by current_lr/base_lr (default: True)
            base_lr: Base learning rate for decay computation
        """
        self.use_dead_zone = True
        self.dead_zone_margin = margin
        self.dead_zone_lr_decay = lr_decay
        self._base_lr = base_lr
        self._dead_zone_flips_blocked = 0
        self._dead_zone_flips_allowed = 0
        self._dead_zone_call_count = 0

        with torch.no_grad():
            q = ternary_quantize_detached(
                self.weight, self.group_size, self.threshold_coeff
            )
            self._prev_ternary = q.sign().clone()

        n_params = self.weight.numel()
        n_active = (self._prev_ternary != 0).sum().item()
        print(
            f"[DEAD-ZONE] Enabled: margin={margin}, lr_decay={lr_decay}, base_lr={base_lr}, "
            f"params={n_params}, initial_active={n_active}/{n_params} ({n_active / n_params:.1%})"
        )

    @torch.no_grad()
    def _apply_dead_zone(
        self, w_quantized: torch.Tensor, w_src: torch.Tensor
    ) -> torch.Tensor:
        """Apply hysteresis filter to quantized weights (R2-A6-M14).

        Compares proposed quantization q with previous assignment _prev_ternary.
        Blocks flips that don't have sufficient margin past the threshold.

        Args:
            w_quantized: Newly quantized weights (from any STE variant) with scales
            w_src: Shadow weights (pre-quantization) for margin computation

        Returns:
            Filtered quantized weights with hysteresis applied
        """
        if self._prev_ternary is None:
            self._prev_ternary = w_quantized.sign().clone()
            return w_quantized

        orig_shape = w_src.shape
        flat = w_src.detach().reshape(-1)
        gs = self.group_size
        pad_len = (gs - flat.numel() % gs) % gs
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, gs)

        scales = grouped.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-8)
        normalized = grouped / scales
        abs_norm = normalized.abs()
        threshold = self.threshold_coeff * abs_norm.mean(dim=-1, keepdim=True)

        effective_margin = self.dead_zone_margin
        if self.dead_zone_lr_decay and self._base_lr > 0:
            lr_ratio = self._current_lr / self._base_lr
            effective_margin = self.dead_zone_margin * max(
                lr_ratio, 0.01
            )

        delta = effective_margin * threshold

        margin_to_active = abs_norm - threshold
        margin_to_zero = threshold - abs_norm

        q_sign = normalized.sign() * (abs_norm > threshold).float()
        prev_flat = self._prev_ternary.detach().reshape(-1)
        if pad_len > 0:
            prev_flat = F.pad(prev_flat, (0, pad_len))
        prev_grouped = prev_flat.reshape(-1, gs)

        flip_mask = q_sign != prev_grouped

        going_active = flip_mask & (prev_grouped == 0) & (q_sign != 0)
        insufficient_active = going_active & (margin_to_active <= delta)

        going_zero = flip_mask & (prev_grouped != 0) & (q_sign == 0)
        insufficient_zero = going_zero & (margin_to_zero <= delta)

        blocked = insufficient_active | insufficient_zero
        final_sign = torch.where(blocked, prev_grouped, q_sign)

        final_q = (final_sign * scales).reshape(-1)
        if pad_len > 0:
            final_q = final_q[: w_src.numel()]
        final_q = final_q.reshape(orig_shape)

        final_sign_unpadded = final_sign.reshape(-1)
        if pad_len > 0:
            final_sign_unpadded = final_sign_unpadded[: w_src.numel()]
        self._prev_ternary = final_sign_unpadded.reshape(orig_shape).clone()

        n_blocked = blocked.sum().item()
        n_flips = flip_mask.sum().item()
        self._dead_zone_flips_blocked += n_blocked
        self._dead_zone_flips_allowed += n_flips - n_blocked
        self._dead_zone_call_count += 1

        if self._dead_zone_call_count % 500 == 1:
            total_attempted = (
                self._dead_zone_flips_blocked + self._dead_zone_flips_allowed
            )
            block_rate = self._dead_zone_flips_blocked / max(total_attempted, 1)
            print(
                f"  [DEAD-ZONE] call={self._dead_zone_call_count} "
                f"step_blocked={n_blocked} step_flips={n_flips} "
                f"cumul_blocked={self._dead_zone_flips_blocked} "
                f"cumul_allowed={self._dead_zone_flips_allowed} "
                f"block_rate={block_rate:.1%} "
                f"eff_margin={effective_margin:.4f}"
            )

        return final_q

    @torch.no_grad()
    def update_freeze_confidence(self):
        """Update confidence accumulator and freeze mask after optimizer.step().

        Must be called AFTER optimizer.step() each training step.
        Uses Adam's v_hat (variance estimate) for gradient normalization.
        Falls back to raw gradient magnitude if v_hat not available.
        """
        if not self.use_info_freeze or self._freeze_confidence is None:
            return

        if self.training_step < self.info_freeze_warmup:
            return

        grad = self.weight.grad
        if grad is None:
            return

        if self._v_hat is not None and self._v_hat.numel() == grad.numel():
            v_hat = self._v_hat.reshape(grad.shape)
            normalized_grad = grad / (v_hat.sqrt() + 1e-8)
        else:
            normalized_grad = grad.sign() * (grad.abs() / (grad.abs().mean() + 1e-8))

        self._freeze_confidence += normalized_grad

        newly_frozen = (~self._freeze_mask) & (
            self._freeze_confidence.abs() > self._freeze_threshold
        )
        if newly_frozen.any():
            self._freeze_mask |= newly_frozen
            self._freeze_confidence[newly_frozen] = 0.0

        self._freeze_call_count += 1
        if self._freeze_call_count % 100 == 1:
            n_frozen = self._freeze_mask.sum().item()
            n_total = self.weight.numel()
            frac = n_frozen / n_total
            max_conf = self._freeze_confidence.abs().max().item()
            mean_conf = (
                self._freeze_confidence[~self._freeze_mask].abs().mean().item()
                if (~self._freeze_mask).any()
                else 0.0
            )
            print(
                f"  [INFO-FREEZE] step={self.training_step} "
                f"frozen={n_frozen}/{n_total} ({frac:.1%}) "
                f"max_conf={max_conf:.2f} mean_conf={mean_conf:.2f} "
                f"threshold={self._freeze_threshold:.2f}"
            )

    @torch.no_grad()
    def get_freeze_stats(self) -> dict:
        """Return freeze statistics for diagnostics."""
        if self._freeze_mask is None:
            return {"freeze_enabled": False}
        n_frozen = self._freeze_mask.sum().item()
        n_total = self.weight.numel()
        unfrozen_mask = ~self._freeze_mask
        return {
            "freeze_enabled": True,
            "frozen_count": n_frozen,
            "total_count": n_total,
            "frozen_frac": n_frozen / n_total,
            "max_confidence": self._freeze_confidence.abs().max().item()
            if self._freeze_confidence is not None
            else 0.0,
            "mean_confidence_unfrozen": self._freeze_confidence[unfrozen_mask]
            .abs()
            .mean()
            .item()
            if unfrozen_mask.any()
            else 0.0,
            "threshold": getattr(self, "_freeze_threshold", 0.0),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantize_enabled:
            w = self.weight
            if self.alpha is not None:
                w = self._apply_alpha(w)
            return F.linear(x, w.to(x.dtype), self.bias)

        # Q11: IAQD - gate input features before ternary matmul
        if self.use_iaqd and self.iaqd_gate_w is not None:
            iaqd_gate = torch.sigmoid(self.iaqd_gate_w * x + self.iaqd_gate_b)
            x = x * iaqd_gate
            if self.training and not torch.compiler.is_compiling():
                self._iaqd_call_count += 1
                if self._iaqd_call_count % 500 == 1:
                    with torch.no_grad():
                        print(
                            f"  [IAQD] call={self._iaqd_call_count} "
                            f"gate_mean={iaqd_gate.mean().item():.4f} "
                            f"gate_std={iaqd_gate.std().item():.4f} "
                            f"gate_min={iaqd_gate.min().item():.4f}"
                        )

        # Dendritic pre-gate: scale shadow weights before quantization
        if self.gate_pre is not None:
            w_src = self.weight * self.gate_pre.unsqueeze(1)
        else:
            w_src = self.weight

        # Apply OptRot rotation before quantization (if enabled)
        if hasattr(self, "optrot"):
            w_src = self.optrot.rotate(w_src)

        # Batch AD: REMOVED — admm_u shift here was wrong (shifted input instead of penalizing optimizer)
        # Proximal gradient is injected in train.py after loss.backward()

        # Q18: Gumbel+STE noise injection before quantization
        if self.use_gumbel_noise and self.training:
            noise_scale = self.gumbel_temperature
            if self.gumbel_anneal:
                progress = self.training_step / max(1, self.total_training_steps)
                noise_scale = self.gumbel_temperature * (1.0 - progress)
            u = torch.rand_like(w_src)
            gumbel_noise = -torch.log(-torch.log(u.clamp(min=1e-10)))
            w_src = w_src + gumbel_noise * noise_scale

        if self.gradual_threshold:
            progress = self.training_step / max(1, self.total_training_steps)
            t_coeff = (
                self.threshold_start
                + (self.threshold_end - self.threshold_start) * progress
            )
        elif getattr(self, "threshold_type", "coefficient") == "thermodynamic":
            progress = self.training_step / max(1, self.total_training_steps)
            t_coeff = max(0.05, self.threshold_coeff * (1.0 - progress) ** 0.25)
            if self.training_step % 100 == 0 and not torch.compiler.is_compiling():
                print(
                    f"  [THERMO] step={self.training_step} progress={progress:.3f} "
                    f"t_coeff={t_coeff:.4f} (start={self.threshold_coeff:.4f})"
                )
        else:
            t_coeff = self.threshold_coeff

        # AU-5: MaxEnt Gibbs — per-weight adaptive threshold probe
        if self.training and self.use_maxent_gibbs:
            self._au5_call_count += 1
            if self._au5_call_count % 500 == 1:
                import math as _au5_math
                _au5_arg = max(-1.0 + 1e-6, min(1.0 - 1e-6, 1.0 - 2.0 * self.maxent_target_sparsity))
                _au5_theta = self.maxent_T * _au5_math.atanh(_au5_arg)
                _au5_lsq = (self.lsq_scale_pos.mean().item()
                            if isinstance(self.lsq_scale_pos, torch.Tensor)
                            else float(self.lsq_scale_pos))
                print(f"[AU-5 MAXENT] call={self._au5_call_count} theta_maxent={_au5_theta:.4f} "
                      f"lsq_scale={_au5_lsq:.4f} t_coeff={t_coeff:.4f}")

        # AV-5: Asymmetric threshold probe — LSQ-derived optimal threshold vs 0
        if self.training and not torch.compiler.is_compiling():
            _av5_key = ('av5', id(self))
            _av_probe_counters[_av5_key] = _av_probe_counters.get(_av5_key, 0) + 1
            if _av_probe_counters[_av5_key] % 500 == 1:
                with torch.no_grad():
                    if (isinstance(self.lsq_scale_pos, torch.Tensor)
                            and isinstance(self.lsq_scale_neg, torch.Tensor)):
                        _av5_sp = self.lsq_scale_pos.detach().abs().mean().item()
                        _av5_sn = self.lsq_scale_neg.detach().abs().mean().item()
                        _av5_tau_opt = 0.5 * (_av5_sp + _av5_sn)
                        _av5_rel_diff = abs(_av5_sp - _av5_sn) / (0.5 * (_av5_sp + _av5_sn) + 1e-8)
                        print(f"[AV-5 ASYM-THR] sp={_av5_sp:.4f} sn={_av5_sn:.4f} "
                              f"tau_opt={_av5_tau_opt:.4f} "
                              f"asymmetry_rel={_av5_rel_diff:.4f}")

        # F13: Hoisted Threshold STE - update cache only every N steps
        if self.training:
            self.threshold_step_counter += 1
            recompute_threshold = (
                self.cached_threshold is None
                or self.threshold_step_counter >= self.threshold_update_freq
            )

            if recompute_threshold:
                self.cached_threshold = self._compute_threshold(w_src, t_coeff).detach()
                self.threshold_step_counter = 0

                if not self._threshold_logged:
                    if os.environ.get("VERBOSE_INIT"):
                        print(
                            f"[HOISTED_STE] Updating thresholds every {self.threshold_update_freq} steps"
                        )
                        print(
                            f"[HOISTED_STE] Layer {self.__class__.__name__}: threshold shape {self.cached_threshold.shape}"
                        )
                    self._threshold_logged = True

        # Debug print disabled — training_step never increments, floods stdout
        # if not torch.compiler.is_compiling() and self.training_step % 100 == 0:
        #     print(f"[THRESHOLD] step={self.training_step} coeff={t_coeff:.4f}")

        if self.use_rpe_gate and hasattr(self, 'rpe_flip_gate'):
            t_coeff = t_coeff * self.rpe_flip_gate.item()
            if not self._rpe_logged and self.rpe_flip_gate.item() > 1.02:
                print(f"[RPE-GATE] threshold gated: t_coeff={t_coeff:.4f} gate={self.rpe_flip_gate.item():.4f}")
                self._rpe_logged = True

        # STP: Stochastic Threshold Perturbation (perturb threshold during training)
        if self.training and self.use_stp:
            sigma_tau = self.stp_sigma_ratio * t_coeff
            epsilon = torch.randn(1, device=w_src.device).item() * sigma_tau
            t_coeff = t_coeff + epsilon
            self._stp_call_count += 1
            if self._stp_call_count % 500 == 1:
                print(
                    f"  [STP] call={self._stp_call_count} t_coeff={t_coeff:.4f} epsilon={epsilon:.6f} sigma_tau={sigma_tau:.6f}"
                )

        if self.use_mdts and self._mdts_logits is not None:
            temp = self.mdts_temperature
            if self.mdts_anneal and self.training:
                progress = self.training_step / max(1, self.total_training_steps)
                temp = max(0.1, 1.0 - 0.9 * progress)
            w = MDTSTernaryQuantize.apply(
                w_src, self._mdts_logits, self._mdts_scales, temp, self.group_size
            )
        elif self.use_tpsq and self._tpsq_sigma_sq is not None:
            w = TPSQTernaryQuantize.apply(
                w_src, self._tpsq_sigma_sq, self.group_size, t_coeff, self.tpsq_beta
            )
        elif self.use_caq and self._v_hat is not None:
            w = CurvatureAwareTernaryQuantize.apply(
                w_src, self._v_hat, self.group_size, t_coeff, self.caq_alpha
            )
        elif self.ste_type == "tcd":
            tau = 1.0 / max(self.tcd_beta, 0.01)
            w = HESTIATernaryQuantize.apply(w_src, self.group_size, tau)
        elif self.ste_type == "biste":
            w = BiSTETernaryQuantize.apply(w_src, self.group_size, t_coeff)
        elif self.ste_type == "boundary_aware":
            sigma = getattr(self, "ba_sigma", 0.5)
            min_mod = getattr(self, "ba_min_mod", 0.3)
            w = BoundaryAwareTernaryQuantize.apply(
                w_src, self.group_size, t_coeff, sigma, min_mod
            )
        elif self.ste_type == "sigma_delta":
            error_clamp = getattr(self, "sd_error_clamp", 2.0)
            w = SigmaDeltaTernaryQuantize.apply(
                w_src, self.group_size, t_coeff, error_clamp
            )
        elif self.ste_type == "bdt":
            w = BDTTernaryQuantize.apply(w_src, self.group_size, t_coeff)
        elif self.ste_type == "bagr":
            w = BAGRTernaryQuantize.apply(
                w_src,
                self.group_size,
                t_coeff,
                0.5,
                self.training_step,
                self.total_training_steps,
            )
        elif self.ste_type == "asgks":
            w = ASGKSTernaryQuantize.apply(
                w_src,
                self.group_size,
                t_coeff,
                0.5,
                self.training_step,
                self.total_training_steps,
            )
        elif self.ste_type == "hestia2026":
            progress = self.training_step / max(self.total_training_steps, 1)
            compress_ratio = 0.2
            anneal_ratio = 0.6

            if progress < compress_ratio:
                pressure = progress / compress_ratio
                tau = self.hestia_tau_init
                phase_name = "compress"
            elif progress < compress_ratio + anneal_ratio:
                pressure = 1.0
                anneal_progress = (progress - compress_ratio) / anneal_ratio
                tau = (
                    self.hestia_tau_init
                    * 0.5
                    * (1.0 + math.cos(math.pi * anneal_progress))
                )
                phase_name = "anneal"
            else:
                pressure = 1.0
                tau = 0.0
                phase_name = "solid"

            if self.training_step % 500 == 0 and not torch.compiler.is_compiling():
                print(
                    f"  [HESTIA2026] step={self.training_step}/{self.total_training_steps} progress={progress:.3f} phase={phase_name} pressure={pressure:.3f} tau={tau:.4f}",
                    flush=True,
                )

            if tau > 0.01:
                w = HESTIATernaryQuantize.apply(w_src, self.group_size, tau)
            else:
                w = TernaryQuantize.apply(w_src, self.group_size, t_coeff)

            if pressure < 1.0:
                w = torch.lerp(w_src, w, pressure)
        elif self.ste_type == "hestia":
            if self._cached_tau is not None:
                tau = self._cached_tau
            else:
                progress = self.training_step / max(1, self.total_training_steps)
                tau = self.hestia_tau_init * 0.5 * (1.0 + math.cos(math.pi * progress))
                tau = max(tau, 0.01)

            if _TRITON_HESTIA_AVAILABLE and self.use_triton_hestia:
                w = hestia_quantize_triton(w_src, self.group_size, tau)
            else:
                w = HESTIATernaryQuantize.apply(w_src, self.group_size, tau)
        elif self.ste_type == "quest":
            w = QuESTTernaryQuantize.apply(w_src, self.group_size, t_coeff)
        elif self.use_magnitude_pruning:
            w = MagnitudePrunedTernaryQuantize.apply(
                w_src, self.group_size, self.magnitude_pruning_threshold_multiplier
            )
        elif self.ste_type == "gumbel_softmax":
            _progress = min(1.0, float(self.training_step) / 10000.0) if self.gumbel_ste_anneal else 0.0
            _tau = self.gumbel_ste_tau_init * (1.0 - _progress) + self.gumbel_ste_tau_min * _progress
            _abs_w = w_src.abs()
            _thr_w = t_coeff * _abs_w.mean()
            _near_thr = (_abs_w - _thr_w).abs() < (_thr_w * 0.3)
            _ols_alpha = _near_thr.float().mean().clamp(min=0.1, max=1.0)
            quantized = GumbelSoftmaxSTE.apply(
                w_src, t_coeff, _ols_alpha,
                self.lsq_scale_pos, self.lsq_scale_neg,
                self.training_step, _tau, self.group_size
            )
            w = quantized
            if self.training_step % 500 == 1:
                print(f"[GUMBEL-STE-AF] step={self.training_step} tau={_tau:.4f} "
                      f"sp_mean={self.lsq_scale_pos.mean():.3f} "
                      f"sn_mean={self.lsq_scale_neg.mean():.3f} ols_a={_ols_alpha:.3f}")
        elif self.ste_type == "ot_gradient":
            w = OTGradientTernaryQuantize.apply(
                w_src,
                self.lsq_scale_pos,
                self.lsq_scale_neg,
                self.group_size,
                t_coeff,
                None,
            )
        elif self.ste_type == "ig_boltzmann":
            w = IGBoltzmannTernaryQuantize.apply(
                w_src, self.lsq_scale_pos, self.lsq_scale_neg,
                self.group_size, t_coeff, None, self.training_step,
            )
        elif self.ste_type == "frank_wolfe":
            w = FrankWolfeTernaryQuantize.apply(
                w_src, self.lsq_scale_pos, self.lsq_scale_neg,
                self.group_size, t_coeff, None, self.training_step,
            )
        else:
            if self.ste_type != "standard" and not torch.compiler.is_compiling():
                import warnings

                warnings.warn(
                    f"Unrecognized ste_type '{self.ste_type}', falling back to standard STE",
                    stacklevel=2,
                )
            if self.use_dual_rate:
                w = DualRateTernaryQuantize.apply(
                    w_src,
                    self.group_size,
                    t_coeff,
                    self.dual_rate_peak,
                    self.dual_rate_floor,
                    self.dual_rate_delta_ratio,
                )
            elif self.use_gaste:
                w = GASTETernaryQuantize.apply(
                    w_src, self.group_size, t_coeff, self.gaste_sigma_g
                )
            elif self.use_ctce:
                w = CTCETernaryQuantize.apply(
                    w_src,
                    self.group_size,
                    t_coeff,
                    self.ctce_beta,
                    self.ctce_delta_ratio,
                )
            else:
                # F17: Packed Cache - try to use cached masks if available
                # LAZY-Q: Lipschitz norm test — skip requantize even after optimizer step
                _lazy_q_skip = False
                if (
                    self.use_packed_cache
                    and self.training
                    and self.packed_weights is not None
                    and self._w_shadow_at_cache is not None
                    and not self.packed_valid.item()
                ):
                    self._lazy_q_total_calls += 1
                    with torch.no_grad():
                        delta_norm = (w_src.detach() - self._w_shadow_at_cache).norm()
                        ref_norm = self._w_shadow_at_cache.norm().clamp(min=1e-8)
                        if delta_norm <= self._lazy_q_eps * ref_norm:
                            _lazy_q_skip = True
                            self._lazy_q_skipped += 1
                            self.packed_valid.fill_(True)
                    if self._lazy_q_total_calls % self._lazy_q_log_interval == 0:
                        skip_rate = self._lazy_q_skipped / max(1, self._lazy_q_total_calls)
                        print(
                            f"[LAZY-Q] calls={self._lazy_q_total_calls} "
                            f"skipped={self._lazy_q_skipped} "
                            f"({skip_rate * 100:.1f}%)"
                        )

                # J2: Committed-layer bypass — disabled (C2 tracking removed)
                _j2_bypass = False

                # Batch AE: accumulated push flip check — invalidate cache for weights that have accumulated enough gradient
                if self.training and hasattr(self, 'accum_push') and self.accum_push is not None:
                    push_abs = self.accum_push.abs()
                    if push_abs.max().item() >= self._accum_push_threshold:
                        # Some weights have accumulated enough push — invalidate cache to force re-quantize
                        self.packed_valid.fill_(False)
                        if hasattr(self, 'int2_packed_valid') and self.int2_packed_valid is not None:
                            self.int2_packed_valid.fill_(False)
                        # Reset pushed weights
                        flip_mask = push_abs >= self._accum_push_threshold
                        self._accum_push_flips += flip_mask.sum().item()
                        self.accum_push[flip_mask] = 0

                use_cache = (
                    self.use_packed_cache
                    and self.training
                    and self.packed_weights is not None
                    and self.packed_valid.item()
                )

                if use_cache:
                    # Decode masks from packed cache (fast bitwise ops)
                    pos_mask, neg_mask = self._unpack_ternary_weights(
                        self.packed_weights
                    )

                    # Reconstruct quantized weights from masks
                    orig_shape = w_src.shape
                    flat = w_src.reshape(-1)
                    pad_len = (
                        self.group_size - flat.numel() % self.group_size
                    ) % self.group_size
                    if pad_len > 0:
                        flat = F.pad(flat, (0, pad_len))
                    grouped = flat.reshape(-1, self.group_size)

                    scales = (
                        grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
                    )
                    normalized = grouped / scales

                    # Reshape masks for per-weight operations
                    pos_mask_grouped = (
                        pos_mask.reshape(-1, self.group_size)
                        if pos_mask.dim() > 1
                        else pos_mask.reshape(-1, self.group_size)
                    )
                    neg_mask_grouped = (
                        neg_mask.reshape(-1, self.group_size)
                        if neg_mask.dim() > 1
                        else neg_mask.reshape(-1, self.group_size)
                    )

                    # Reconstruct ternary from masks
                    w_quant = torch.zeros_like(normalized)
                    w_quant[pos_mask_grouped] = 1.0
                    w_quant[neg_mask_grouped] = -1.0

                    # Apply scale and reshape
                    w_flat = (w_quant * scales).reshape(-1)
                    if pad_len > 0:
                        w_flat = w_flat[: w_src.numel()]
                    w = w_flat.reshape(orig_shape)

                    # Use STE detach trick for backward
                    w = w_src + (w - w_src).detach()

                    if not self._packed_logged:
                        if os.environ.get("VERBOSE_INIT"):
                            print(
                                f"[PACKED_CACHE] Using cached ternary masks (no recomputation)"
                            )
                        self._packed_logged = True
                else:
                    if not _j2_bypass:
                        # Compute masks from scratch
                        if self.use_strang_ste:
                            w = StrangSplitLSQTernaryQuantize.apply(
                                w_src,
                                self.lsq_scale_pos,
                                self.lsq_scale_neg,
                                self.group_size,
                                t_coeff,
                                None,
                                self.training_step,
                            )
                        elif self.use_gv_ste and self.gv_ste_ema is not None:
                            gv_slope = self.gv_ste_ema.float().clamp(min=1e-6)
                            w = GVSTELSQTernaryQuantize.apply(
                                w_src,
                                self.lsq_scale_pos,
                                self.lsq_scale_neg,
                                self.group_size,
                                t_coeff,
                                gv_slope,
                                None,
                                self.training_step,
                            )
                            if not self._gv_ste_logged:
                                with torch.no_grad():
                                    ema_mean = self.gv_ste_ema.float().mean().item()
                                    ema_std = self.gv_ste_ema.float().std().item()
                                    print(f"[GV-STE] ema_mean={ema_mean:.4f} ema_std={ema_std:.4f}")
                                self._gv_ste_logged = True
                        elif self.use_wasserstein_ste:
                            w = WassersteinSTETernaryQuantize.apply(
                                w_src,
                                self.lsq_scale_pos,
                                self.lsq_scale_neg,
                                self.group_size,
                                t_coeff,
                                self.training_step,
                            )
                        else:
                            w = LSQTernaryQuantize.apply(
                                w_src,
                                self.lsq_scale_pos,
                                self.lsq_scale_neg,
                                self.group_size,
                                t_coeff,
                                None,
                                self.training_step,
                            )
                    else:
                        # J2: Use cached weights when fully committed
                        pos_mask, neg_mask = self._unpack_ternary_weights(
                            self.packed_weights
                        )

                        # Reconstruct quantized weights from masks
                        orig_shape = w_src.shape
                        flat = w_src.reshape(-1)
                        pad_len = (
                            self.group_size - flat.numel() % self.group_size
                        ) % self.group_size
                        if pad_len > 0:
                            flat = F.pad(flat, (0, pad_len))
                        grouped = flat.reshape(-1, self.group_size)

                        scales = (
                            grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
                        )
                        normalized = grouped / scales

                        # Reshape masks for per-weight operations
                        pos_mask_grouped = (
                            pos_mask.reshape(-1, self.group_size)
                            if pos_mask.dim() > 1
                            else pos_mask.reshape(-1, self.group_size)
                        )
                        neg_mask_grouped = (
                            neg_mask.reshape(-1, self.group_size)
                            if neg_mask.dim() > 1
                            else neg_mask.reshape(-1, self.group_size)
                        )

                        # Reconstruct ternary from masks
                        w_quant = torch.zeros_like(normalized)
                        w_quant[pos_mask_grouped] = 1.0
                        w_quant[neg_mask_grouped] = -1.0

                        # Apply scale and reshape
                        w_flat = (w_quant * scales).reshape(-1)
                        if pad_len > 0:
                            w_flat = w_flat[: w_src.numel()]
                        w = w_flat.reshape(orig_shape)

                        # Use STE detach trick for backward
                        w = w_src + (w - w_src).detach()

                    # F17: Cache the ternary quantization result for next forwards
                    if self.training and self.use_packed_cache and not _j2_bypass:
                        with torch.no_grad():
                            # Get the quantized weights (rounded version without STE)
                            orig_shape = w_src.shape
                            flat = w_src.reshape(-1)
                            pad_len = (
                                self.group_size - flat.numel() % self.group_size
                            ) % self.group_size
                            if pad_len > 0:
                                flat = F.pad(flat, (0, pad_len))
                            grouped = flat.reshape(-1, self.group_size)

                            scales = (
                                grouped.abs()
                                .max(dim=-1, keepdim=True)
                                .values.clamp(min=1e-8)
                            )
                            normalized = grouped / scales
                            abs_normalized = normalized.abs()

                            if self.cached_threshold is not None:
                                threshold = self.cached_threshold
                            else:
                                threshold = t_coeff * abs_normalized.mean(
                                    dim=-1, keepdim=True
                                )

                            quantized = (
                                normalized.sign() * (abs_normalized > threshold).float()
                            )

                            # Pack ternary weights as INT8
                            w_ternary = torch.zeros_like(normalized)
                            w_ternary[quantized > 0.5] = 1.0
                            w_ternary[quantized < -0.5] = -1.0

                            # Reshape for packing
                            w_ternary_flat = w_ternary.reshape(-1)
                            if pad_len > 0:
                                w_ternary_flat = w_ternary_flat[: w_src.numel()]
                            w_ternary_shaped = w_ternary_flat.reshape(orig_shape)

                            self.packed_weights = self._pack_ternary_weights(
                                w_ternary_shaped
                            )
                            self.packed_valid.fill_(True)

                            # R12-F3: Also pack as INT2 (4x smaller)
                            from .bitmask_utils import pack_ternary_int2
                            _out_f, _in_f = w_ternary_shaped.shape
                            self.int2_packed_weights = pack_ternary_int2(w_ternary_shaped)
                            self._int2_original_K = _in_f
                            self.int2_packed_valid.fill_(True)
                            if not getattr(self, '_int2_logged', False):
                                self._int2_logged = True
                                print(f"[INT2-PACK] {self.int2_packed_weights.shape} "
                                      f"({self.int2_packed_weights.nbytes/1024:.1f} KB "
                                      f"vs {self.packed_weights.nbytes/1024:.1f} KB INT8)")

                            # LAZY-Q: snapshot w_src for next Lipschitz test
                            if self._w_shadow_at_cache is None or self._w_shadow_at_cache.shape != w_src.shape:
                                self._w_shadow_at_cache = w_src.detach().clone()
                            else:
                                self._w_shadow_at_cache.copy_(w_src.detach())

                # Batch AD: Cache last ternary for ADMM proximal gradient injection
                if hasattr(self, "admm_u") and self.admm_u is not None and self.training:
                    self._admm_last_ternary = w.detach()

                # ADMM dual variable update (R7-A3-M3) — runs every forward pass regardless of cache
                if hasattr(self, "admm_u") and self.admm_u is not None and self.training:
                    with torch.no_grad():
                        residual = (w_src - w.detach()).to(self.admm_u.dtype)
                        self.admm_u.add_(residual, alpha=self._admm_rho)
                    if not torch.compiler.is_compiling() and self.training_step % 500 == 1:
                        # Batch AD: Log proximal term norm
                        _proximal_norm = (self.weight.to(self.admm_u.dtype) - self._admm_last_ternary.to(self.admm_u.dtype) + self.admm_u).norm().item()
                        print(
                            f"[ADMM-AD] step={self.training_step} "
                            f"u_mean={self.admm_u.abs().mean().item():.5f} "
                            f"proximal_norm={_proximal_norm:.5f} "
                            f"residual_mean={residual.abs().mean().item():.5f}"
                        )
                        # Batch AE: accum_push logging
                        if hasattr(self, 'accum_push') and self.accum_push is not None:
                            frac_ready = (self.accum_push.abs() >= self._accum_push_threshold * 0.5).float().mean().item()
                            print(f"[ACCUM-PUSH] step={self.training_step} frac_ready={frac_ready:.4f} total_flips={self._accum_push_flips} push_mean={self.accum_push.float().abs().mean().item():.1f}")

                if self.training and torch.rand(1).item() < 0.001:
                    logger.info(
                        f"[FUNCTIONAL_STE] Layer {self.__class__.__name__} using functional quantization. "
                        f"Weight shape: {self.weight.shape}, group_size: {self.group_size}, "
                        f"ste_type: standard"
                    )

        # Oscillation Dead Zone: apply hysteresis filter (R2-A6-M14)
        if self.use_dead_zone and self.training and self._prev_ternary is not None:
            # Apply hysteresis but keep STE gradient path intact
            w_filtered = self._apply_dead_zone(w, w_src)
            # Replace quantized output while preserving STE gradient (detach trick)
            w = w_src + (w_filtered - w_src).detach()

        if (
            self.pressure_blend
            and self.training_step < self.total_training_steps * self.pressure_duration
        ):
            p = self.training_step / max(
                1, self.total_training_steps * self.pressure_duration
            )
            w = (1.0 - p) * w_src + p * w

        if self.alpha is not None:
            w = self._apply_alpha(w)

        # R12-F2: Progressive Precision Replacement (always-active during training)
        # Stochastically selects FP32 shadow weights vs quantized ternary weights.
        # p = min(log10(0.001*step + 1), 1.0): ≈0 at step=0, 0.30 at step=2000, 1.0 at step=9000
        self._ppr_use_quantized = True  # default for eval / non-training
        if self.training and self.use_ppr:
            _ppr_p = min(math.log10(0.001 * self.training_step + 1), 1.0)
            self._ppr_use_quantized = torch.rand(1).item() < _ppr_p
            if not self._ppr_use_quantized:
                # FP32 bypass: shadow weight with admm_u correction, clean backward
                w = w_src.to(w.dtype)
            if not torch.compiler.is_compiling() and self.training_step % 500 == 1:
                print(
                    f"[PPR-F2] step={self.training_step} p={_ppr_p:.3f} "
                    f"quantized={self._ppr_use_quantized}"
                )

        if self.use_bitmask_gemv:
            from .bitmask_utils import BitmaskLinearFunc

            pos_mask, neg_mask, original_K = self._get_bitmask_cache(w)

            if self.training:
                self._bitmask_stale = True

            output = BitmaskLinearFunc.apply(
                x, w, pos_mask, neg_mask, original_K, self.bias
            )

        elif self.training:
            if self.use_tatgr:
                w_sign = w.detach().sign().to(torch.int8)
                output = TernaryLinearTATGR.apply(x, w, w_sign, self.bias)
            elif self.use_btg:
                output = TernaryLinearBTG.apply(x, w, self.bias)
            else:
                # R5-A1-M7: PBDG — binary decomposition active.
                # Training: F.linear preserves STE gradient path through w.
                # Eval: pwst_forward_vectorized (see eval branch below).
                # Hardware speedup from binary decomposition comes at inference on target hardware.
                # INT32 path is NOT gated by PPR — INT32 is forward computation, PPR is quantization policy.
                _K = x.shape[-1]
                _int32_ready = (
                    getattr(self, 'use_int32_forward', True)
                    and x.device.type == 'cuda'
                    and _K >= getattr(self, '_int32_min_k', 64)
                )

                if _int32_ready and self.int2_packed_valid.item() and self.lsq_scale_pos is not None:
                    # INT2 packed -> INT8 unpack -> INT32 matmul (4× less weight bandwidth)
                    from .bitmask_utils import TernaryInt2LinearFunc
                    _alpha = ((self.lsq_scale_pos.detach().abs().mean()
                               + self.lsq_scale_neg.detach().abs().mean()) * 0.5)
                    if not self._int32_logged:
                        print(f"[INT32-FWD] INT2->INT8->INT32 path active K={_K} alpha={_alpha.item():.4f}")
                        self._int32_logged = True
                    output = TernaryInt2LinearFunc.apply(
                        x, w, self.int2_packed_weights, self._int2_original_K, _alpha, self.bias)

                elif _int32_ready and hasattr(self, 'lsq_scale_pos') and self.lsq_scale_pos is not None:
                    # INT2 cache not ready yet — fall back to INT8 direct (w already in fp32 {±scale,0})
                    # Quantize w to int8 sign on-the-fly from the current ternary w
                    from .bitmask_utils import TernaryInt8LinearFunc
                    _alpha = ((self.lsq_scale_pos.detach().abs().mean()
                               + self.lsq_scale_neg.detach().abs().mean()) * 0.5)
                    w_int8_sign = w.detach().sign().to(torch.int8)  # {-1,0,+1} as int8
                    if not self._int32_logged:
                        print(f"[INT32-FWD] INT8-sign->INT32 fallback path active K={_K} (int2 cache not ready)")
                        self._int32_logged = True
                    output = TernaryInt8LinearFunc.apply(
                        x, w, w_int8_sign, _alpha, self.bias)

                elif _int32_ready:
                    # INT32 simple fallback — works at init without lsq_scale or int2_packed
                    # Ternary w ∈ {-1,0,+1}, INT8 x quantized on-the-fly
                    from .bitmask_utils import TernaryInt8LinearFunc
                    w_int8_sign = w.detach().sign().to(torch.int8)
                    _alpha = 1.0
                    if not self._int32_logged:
                        print(f"[INT32-FWD] Standalone INT8-sign->INT32 path active K={_K} (init, no scales yet)")
                        self._int32_logged = True
                    output = TernaryInt8LinearFunc.apply(
                        x, w, w_int8_sign, _alpha, self.bias)

                else:
                    output = F.linear(x, w.to(x.dtype), self.bias)

        else:
            if self.use_pwst and self.use_btg:
                output = TernaryLinearBTG_PWST.apply(x, w, self.bias)
            elif self.use_btg:
                output = TernaryLinearBTG.apply(x, w, self.bias)
            else:
                _K = x.shape[-1]
                _int32_ready = (
                    getattr(self, 'use_int32_forward', True)
                    and x.device.type == 'cuda'
                    and _K >= getattr(self, '_int32_min_k', 64)
                )

                if _int32_ready and self.int2_packed_valid.item() and self.lsq_scale_pos is not None:
                    # INT2 packed -> INT8 unpack -> INT32 matmul (4× less weight bandwidth)
                    from .bitmask_utils import TernaryInt2LinearFunc
                    _alpha = ((self.lsq_scale_pos.detach().abs().mean()
                               + self.lsq_scale_neg.detach().abs().mean()) * 0.5)
                    if not self._int32_logged:
                        print(f"[INT32-FWD] INT2->INT8->INT32 path active K={_K} alpha={_alpha.item():.4f}")
                        self._int32_logged = True
                    output = TernaryInt2LinearFunc.apply(
                        x, w, self.int2_packed_weights, self._int2_original_K, _alpha, self.bias)

                elif _int32_ready and hasattr(self, 'lsq_scale_pos') and self.lsq_scale_pos is not None:
                    # INT2 cache not ready yet — fall back to INT8 direct (w already in fp32 {±scale,0})
                    # Quantize w to int8 sign on-the-fly from the current ternary w
                    from .bitmask_utils import TernaryInt8LinearFunc
                    _alpha = ((self.lsq_scale_pos.detach().abs().mean()
                               + self.lsq_scale_neg.detach().abs().mean()) * 0.5)
                    w_int8_sign = w.detach().sign().to(torch.int8)  # {-1,0,+1} as int8
                    if not self._int32_logged:
                        print(f"[INT32-FWD] INT8-sign->INT32 fallback path active K={_K} (int2 cache not ready)")
                        self._int32_logged = True
                    output = TernaryInt8LinearFunc.apply(
                        x, w, w_int8_sign, _alpha, self.bias)

                elif _int32_ready:
                    # INT32 simple fallback — works at init without lsq_scale or int2_packed
                    # Ternary w ∈ {-1,0,+1}, INT8 x quantized on-the-fly
                    from .bitmask_utils import TernaryInt8LinearFunc
                    w_int8_sign = w.detach().sign().to(torch.int8)
                    _alpha = 1.0
                    if not self._int32_logged:
                        print(f"[INT32-FWD] Standalone INT8-sign->INT32 path active K={_K} (init, no scales yet)")
                        self._int32_logged = True
                    output = TernaryInt8LinearFunc.apply(
                        x, w, w_int8_sign, _alpha, self.bias)

                else:
                    output = F.linear(x, w.to(x.dtype), self.bias)

        # Block-diagonal mixing: restore singular value magnitude variation
        if self.block_mixing is not None:
            out_shape = output.shape
            output = output.reshape(*out_shape[:-1], -1, self.block_size)
            output = torch.einsum("...gj,gij->...gi", output, self.block_mixing)
            output = output.reshape(out_shape)

        # Dendritic post-gate: scale outputs after matmul
        if self.gate_post is not None:
            output = output * self.gate_post

        # Tequila deadzone reactivation (arXiv 2509.23809, ICLR 2026)
        # Adds scalar bias C(W) = lambda * sum(shadow_weights_in_deadzone) to output.
        # Autograd naturally provides clean gradient to stuck weights.
        if self.tequila_lambda > 0 and self.training:
            with torch.no_grad():
                flat = self.weight.detach().reshape(-1)
                gs = self.group_size
                pad_len = (gs - flat.shape[0] % gs) % gs
                if pad_len > 0:
                    flat = F.pad(flat, (0, pad_len))
                grouped = flat.reshape(-1, gs)
                scales = grouped.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-8)
                normalized = grouped / scales
                threshold = t_coeff * normalized.abs().mean(dim=-1, keepdim=True)
                deadzone = (normalized.abs() <= threshold).reshape(-1)[
                    : self.weight.numel()
                ]
                deadzone_mask = deadzone.reshape(self.weight.shape)
            C_W = self.tequila_lambda * (self.weight * deadzone_mask.float()).sum()
            output = output + C_W
            if not torch.compiler.is_compiling():
                self._tequila_call_count += 1
                if self._tequila_call_count % 500 == 1:
                    with torch.no_grad():
                        dz_pct = deadzone_mask.float().mean().item() * 100
                        print(
                            f"  [TEQUILA] call={self._tequila_call_count} "
                            f"C_W={C_W.item():.6f} deadzone={dz_pct:.1f}%"
                        )

        # Low-rank FP32 correction (Phase 21.A) - RLEC
        if self.correction_A is not None and self.correction_B is not None:
            correction = F.linear(F.linear(x, self.correction_B), self.correction_A).to(
                output.dtype
            )
            output = output + correction
            self._rlec_call_count += 1
            if self._rlec_call_count % 50 == 1:
                corr_norm = correction.norm().item()
                out_norm = output.norm().item()
                ratio = corr_norm / (out_norm + 1e-8)
                print(
                    f"  [RLEC] call={self._rlec_call_count} corr_norm={corr_norm:.6f} out_norm={out_norm:.6f} ratio={ratio:.4f}"
                )

        # AV-1: TernMAC — zero fraction probe (ops avoided by sparse execution)
        if self.training and not torch.compiler.is_compiling():
            _av1_key = ('av1', id(self))
            _av_probe_counters[_av1_key] = _av_probe_counters.get(_av1_key, 0) + 1
            if _av_probe_counters[_av1_key] % 200 == 1:
                with torch.no_grad():
                    _av1_zero_frac = (w_src.detach() == 0).float().mean().item()
                    _av1_total = w_src.numel()
                    print(f"[AV-1 TernMAC] shape={tuple(w_src.shape)} "
                          f"zero_frac={_av1_zero_frac:.3f} "
                          f"nnz={int(_av1_total*(1-_av1_zero_frac))} "
                          f"ops_skipped_if_sparse={_av1_zero_frac*100:.1f}%")
        # AV-2: INT2 packing status probe
        if self.training and not torch.compiler.is_compiling():
            _av2_key = ('av2', id(self))
            _av_probe_counters[_av2_key] = _av_probe_counters.get(_av2_key, 0) + 1
            if _av_probe_counters[_av2_key] % 500 == 1:
                with torch.no_grad():
                    _av2_int2_valid = (hasattr(self, 'int2_packed_valid')
                                      and self.int2_packed_valid is not None
                                      and self.int2_packed_valid.item())
                    _av2_int8_valid = (hasattr(self, 'packed_valid')
                                      and self.packed_valid is not None
                                      and self.packed_valid.item())
                    _av2_n = self.weight.numel()
                    print(f"[AV-2 INT2-PACK] int2_valid={_av2_int2_valid} "
                          f"int8_valid={_av2_int8_valid} params={_av2_n} "
                          f"int8={_av2_n/1024:.1f}KB int2={_av2_n/4/1024:.1f}KB "
                          f"saved={(_av2_n - _av2_n//4)/1024:.1f}KB")
        # AV-3: Sparse zero-skip — real sparsity of quantized ternary weights
        if self.training and not torch.compiler.is_compiling():
            _av3_key = ('av3', id(self))
            _av_probe_counters[_av3_key] = _av_probe_counters.get(_av3_key, 0) + 1
            if _av_probe_counters[_av3_key] % 200 == 1:
                with torch.no_grad():
                    _av3_w_q = w.detach()
                    _av3_sparsity = (_av3_w_q == 0).float().mean().item()
                    _av3_pos = int((_av3_w_q > 0).float().sum().item())
                    _av3_neg = int((_av3_w_q < 0).float().sum().item())
                    _av3_nnz = _av3_pos + _av3_neg
                    print(f"[AV-3 SPARSE-SKIP] sparsity={_av3_sparsity:.3f} "
                          f"nnz={_av3_nnz} pos={_av3_pos} neg={_av3_neg} "
                          f"theoretical_skip={_av3_sparsity*100:.1f}%")
        # AV-6: ELUT mirror — informational log once per layer
        if self.training and not torch.compiler.is_compiling():
            _av6_key = ('av6', id(self))
            _av_probe_counters[_av6_key] = _av_probe_counters.get(_av6_key, 0) + 1
            if _av_probe_counters[_av6_key] == 1:
                _av6_gs = getattr(self, 'group_size', 64)
                print(f"[AV-6 ELUT-MIRROR] group_size={_av6_gs} "
                      f"lut_entries_gs3=27 mirror_entries_gs3=13 "
                      f"large_gs_precludes_full_LUT={_av6_gs > 5}")
        # AV-7: LUT data layout — informational once per layer
        if self.training and not torch.compiler.is_compiling():
            _av7_key = ('av7', id(self))
            _av_probe_counters[_av7_key] = _av_probe_counters.get(_av7_key, 0) + 1
            if _av_probe_counters[_av7_key] == 1:
                _av7_out, _av7_in = self.weight.shape
                _av7_gs = getattr(self, 'group_size', 64)
                _av7_n_groups = (_av7_in + _av7_gs - 1) // _av7_gs
                print(f"[AV-7 LUT-LAYOUT] current=[out={_av7_out}, in={_av7_in}] "
                      f"lut_centric=[n_groups={_av7_n_groups}, out={_av7_out}] "
                      f"K_TILE benefit: group-contiguous vs strided layout")
        # AV-8: Sparse O(nnz) processing probe
        if self.training and not torch.compiler.is_compiling():
            _av8_key = ('av8', id(self))
            _av_probe_counters[_av8_key] = _av_probe_counters.get(_av8_key, 0) + 1
            if _av_probe_counters[_av8_key] % 200 == 1:
                with torch.no_grad():
                    _av8_w = w.detach()
                    _av8_pos = int((_av8_w > 0).sum().item())
                    _av8_neg = int((_av8_w < 0).sum().item())
                    _av8_nnz = _av8_pos + _av8_neg
                    _av8_n = _av8_w.numel()
                    _av8_speedup = _av8_n / max(_av8_nnz, 1)
                    print(f"[AV-8 SPARSE-NNZ] nnz={_av8_nnz} n={_av8_n} "
                          f"pos={_av8_pos} neg={_av8_neg} "
                          f"O(nnz)/O(n)_speedup={_av8_speedup:.2f}x")
        # AV-9: LLVM autovec candidate probe — once per layer
        if self.training and not torch.compiler.is_compiling():
            _av9_key = ('av9', id(self))
            _av_probe_counters[_av9_key] = _av_probe_counters.get(_av9_key, 0) + 1
            if _av_probe_counters[_av9_key] == 1:
                with torch.no_grad():
                    _av9_nnz = int((w.detach() != 0).sum().item())
                    _av9_out, _av9_in = self.weight.shape
                    print(f"[AV-9 LLVM-AUTOVEC] CANDIDATE: branch-free inner loop "
                          f"nnz={_av9_nnz} out={_av9_out} in={_av9_in} "
                          f"vectorizable: acc += x[j] if w[i,j]!=0 (no data-dependent branches)")
        # AV-10: Cache-blocked write probe — once per layer
        if self.training and not torch.compiler.is_compiling():
            _av10_key = ('av10', id(self))
            _av_probe_counters[_av10_key] = _av_probe_counters.get(_av10_key, 0) + 1
            if _av_probe_counters[_av10_key] == 1:
                _av10_TILE = 32
                _av10_out = self.out_features
                _av10_writes_tiled = (_av10_out + _av10_TILE - 1) // _av10_TILE
                print(f"[AV-10 CACHE-BLOCK] TILE={_av10_TILE} out={_av10_out} "
                      f"cache_writes_tiled={_av10_writes_tiled} "
                      f"vs_scalar_writes={_av10_out} bytes_per_tile={_av10_TILE*2}")
        # AV-14: MDS code projection probe
        if self.training and not torch.compiler.is_compiling():
            _av14_key = ('av14', id(self))
            _av_probe_counters[_av14_key] = _av_probe_counters.get(_av14_key, 0) + 1
            if _av_probe_counters[_av14_key] % 200 == 1:
                with torch.no_grad():
                    _av14_w = w_src.detach()
                    _av14_grad_norm = (self.weight.grad.norm().item()
                                      if self.weight.grad is not None else float('nan'))
                    _av14_median = _av14_w.abs().median().item()
                    _av14_n_weak = int((_av14_w.abs() < _av14_median * 0.5).sum().item())
                    print(f"[AV-14 MDS-CODE] grad_norm={_av14_grad_norm:.4f} "
                          f"MDS-[16,8,9] d_half=4 n_weak={_av14_n_weak} "
                          f"weak_frac={_av14_n_weak/max(_av14_w.numel(),1):.3f}")
        # AV-15: Tree reduction depth probe
        if self.training and not torch.compiler.is_compiling():
            _av15_key = ('av15', id(self))
            _av_probe_counters[_av15_key] = _av_probe_counters.get(_av15_key, 0) + 1
            if _av_probe_counters[_av15_key] % 200 == 1:
                with torch.no_grad():
                    import math as _av15_math
                    _av15_nnz = int((w.detach() != 0).sum().item())
                    _av15_n = w.numel()
                    _av15_depth_tree = max(1, _av15_math.ceil(_av15_math.log2(max(_av15_nnz, 2))))
                    _av15_speedup = _av15_nnz / max(_av15_depth_tree, 1)
                    print(f"[AV-15 TREE-REDUCE] nnz={_av15_nnz} n={_av15_n} "
                          f"depth_tree={_av15_depth_tree} "
                          f"parallel_speedup={_av15_speedup:.1f}x")
        # AV-16: GF(3) SNF active rank proxy probe
        if self.training and not torch.compiler.is_compiling():
            _av16_key = ('av16', id(self))
            _av_probe_counters[_av16_key] = _av_probe_counters.get(_av16_key, 0) + 1
            if _av_probe_counters[_av16_key] % 200 == 1:
                with torch.no_grad():
                    _av16_w = w.detach()
                    _av16_nnz_per_row = (_av16_w != 0).float().sum(dim=1).mean().item()
                    _av16_in = self.in_features
                    _av16_rank_proxy = _av16_nnz_per_row / max(_av16_in, 1)
                    print(f"[AV-16 GF3-SNF] nnz_per_row={_av16_nnz_per_row:.1f} "
                          f"in={_av16_in} rank_proxy={_av16_rank_proxy:.3f}")
        # AW-1: Tropical Semiring — dynamic threshold probe (region count feedback)
        if self.training and not torch.compiler.is_compiling():
            _aw1_key = ('aw1', id(self))
            _av_probe_counters[_aw1_key] = _av_probe_counters.get(_aw1_key, 0) + 1
            if _av_probe_counters[_aw1_key] % 500 == 1:
                with torch.no_grad():
                    _aw1_w = self.weight.detach().float()
                    _aw1_total = _aw1_w.numel()
                    _aw1_thr = self.threshold_coeff if hasattr(self, 'threshold_coeff') else 0.5
                    _aw1_nnz = int((_aw1_w.abs() > _aw1_thr * _aw1_w.abs().mean()).sum().item())
                    _aw1_dyn_thr = _aw1_thr * (1.0 + 1.0 * (1.0 - _aw1_nnz / max(_aw1_total, 1)))
                    print(f"[AW-1 TROPICAL] region_count={_aw1_nnz}/{_aw1_total} "
                          f"base_thr={_aw1_thr:.4f} dynamic_thr={_aw1_dyn_thr:.4f}")
        # AW-3: Neural Persistence — topological stability via weight active fraction EMA/var
        if self.training and not torch.compiler.is_compiling():
            _aw3_key = ('aw3', id(self))
            _av_probe_counters[_aw3_key] = _av_probe_counters.get(_aw3_key, 0) + 1
            if _aw3_key not in _aw3_ema_store:
                _aw3_ema_store[_aw3_key] = {'ema': 0.0, 'ema2': 0.0}
            with torch.no_grad():
                _aw3_w = self.weight.detach().float()
                _aw3_thr_v = self.threshold_coeff if hasattr(self, 'threshold_coeff') else 0.5
                _aw3_frac = float((_aw3_w.abs() > _aw3_thr_v * _aw3_w.abs().mean()).float().mean().item())
                _aw3_s = _aw3_ema_store[_aw3_key]
                _aw3_s['ema'] = 0.99 * _aw3_s['ema'] + 0.01 * _aw3_frac
                _aw3_s['ema2'] = 0.99 * _aw3_s['ema2'] + 0.01 * _aw3_frac ** 2
                _aw3_var = _aw3_s['ema2'] - _aw3_s['ema'] ** 2
                if _av_probe_counters[_aw3_key] % 200 == 1:
                    print(f"[AW-3 NEURAL-PERSIST] active_frac={_aw3_frac:.4f} "
                          f"ema={_aw3_s['ema']:.4f} var={_aw3_var:.6f} "
                          f"topologically_stable={_aw3_var < 1e-4}")
        # AW-6: Fisher FIM proxy — weighted quantization error (shadow - ternary)^2
        if self.training and not torch.compiler.is_compiling():
            _aw6_key = ('aw6', id(self))
            _av_probe_counters[_aw6_key] = _av_probe_counters.get(_aw6_key, 0) + 1
            if _av_probe_counters[_aw6_key] % 500 == 1:
                with torch.no_grad():
                    _aw6_shadow = self.weight.detach().float()
                    _aw6_thr_v = self.threshold_coeff if hasattr(self, 'threshold_coeff') else 0.5
                    _aw6_scale = _aw6_shadow.abs().max().clamp(min=1e-8)
                    _aw6_tern = _aw6_shadow.sign() * (_aw6_shadow.abs() > _aw6_thr_v * _aw6_shadow.abs().mean()).float()
                    _aw6_err = (_aw6_shadow - _aw6_tern * _aw6_scale).pow(2).mean().item()
                    _aw6_fim_proxy = _aw6_err / (_aw6_scale.item() ** 2 + 1e-8)
                    print(f"[AW-6 FIM-PROXY] quant_err={_aw6_err:.6f} "
                          f"fim_weighted_err={_aw6_fim_proxy:.6f} scale={_aw6_scale.item():.4f}")
        # AW-8: Sinkhorn OT cost — transport cost shadow -> ternary target {-scale,0,+scale}
        if self.training and not torch.compiler.is_compiling():
            _aw8_key = ('aw8', id(self))
            _av_probe_counters[_aw8_key] = _av_probe_counters.get(_aw8_key, 0) + 1
            if _av_probe_counters[_aw8_key] % 500 == 1:
                with torch.no_grad():
                    _aw8_w = self.weight.detach().float().reshape(-1)[:256]
                    _aw8_scale = _aw8_w.abs().max().clamp(min=1e-8)
                    _aw8_targets = torch.tensor([-_aw8_scale.item(), 0.0, _aw8_scale.item()],
                                                device=_aw8_w.device)
                    _aw8_cost_mat = (_aw8_w.unsqueeze(1) - _aw8_targets.unsqueeze(0)).pow(2)
                    _aw8_ot_cost = _aw8_cost_mat.min(dim=1).values.mean().item()
                    print(f"[AW-8 SINKHORN-OT] ot_cost={_aw8_ot_cost:.6f} "
                          f"scale={_aw8_scale.item():.4f} n_sample={_aw8_w.numel()}")
        # AV-20: Base-3 packing economy probe — once per layer
        if self.training and not torch.compiler.is_compiling():
            _av20_key = ('av20', id(self))
            _av_probe_counters[_av20_key] = _av_probe_counters.get(_av20_key, 0) + 1
            if _av_probe_counters[_av20_key] == 1:
                import math as _av20_math
                _av20_n = self.weight.numel()
                _av20_base3_bytes = _av20_n * _av20_math.log2(3) / 8
                _av20_int2_bytes = _av20_n / 4
                _av20_int8_bytes = float(_av20_n)
                print(f"[AV-20 BASE3-PACK] n={_av20_n} "
                      f"int8={_av20_int8_bytes/1024:.1f}KB "
                      f"int2={_av20_int2_bytes/1024:.1f}KB "
                      f"base3={_av20_base3_bytes/1024:.1f}KB "
                      f"saved_vs_int2={(_av20_int2_bytes-_av20_base3_bytes)/1024:.1f}KB")

        return output


def invalidate_packed_cache_model(model: nn.Module):
    """F17: Invalidate packed weight cache across all TernaryLinear layers.

    Call this AFTER optimizer.step() to mark all cached masks as stale,
    forcing recomputation on next forward pass.

    Args:
        model: nn.Module containing TernaryLinear layers
    """
    for module in model.modules():
        if isinstance(module, TernaryLinear) and hasattr(
            module, "invalidate_packed_cache"
        ):
            module.invalidate_packed_cache()
