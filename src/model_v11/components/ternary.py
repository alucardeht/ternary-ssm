import logging
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .triton_hestia import hestia_quantize_triton

    _TRITON_HESTIA_AVAILABLE = True
except ImportError:
    _TRITON_HESTIA_AVAILABLE = False

_q_values_cache: dict = {}
_aw3_ema_store: dict = {}  # AW-3: Neural Persistence EMA state per TernaryLinear instance
logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD_COEFF = 1.0 / math.sqrt(2)  # Laplace-optimal (R196)


# Quantizers — extracted to ternary_pkg/quantizers.py
from .ternary_pkg.quantizers import *


# Quantizer classes extracted to ternary_pkg/quantizers.py




def ternary_quantize_functional(
    weight: torch.Tensor,
    group_size: int = 128,
    threshold_coeff: float = DEFAULT_THRESHOLD_COEFF,
    ste_type: str = "standard",
    cached_threshold: torch.Tensor = None,
    threshold_type: str = "coefficient",  # NEW: 'coefficient' or 'twn'
) -> torch.Tensor:
    """
    Functional ternary quantization with STE (torch.compile compatible).

    Quantizes weights to {-1, 0, +1} using per-group scaling and threshold.
    Backward pass uses STE (identity gradient) via detach trick.

    This function eliminates graph breaks from torch.autograd.Function by using
    the detach trick: forward uses quantized weights, backward uses identity
    gradient via the mathematical identity:

        d/dw[w + (q_w - w).detach()] = d/dw[w] = 1 (STE identity)

    Args:
        weight: FP32 shadow weights (..., out_features, in_features) or any shape
        group_size: Per-group quantization size (default 128)
        threshold_coeff: Threshold multiplier for sparsification (default 0.767)
        ste_type: STE variant ('standard' only for now, others TODO)
        cached_threshold: Optional pre-computed per-group thresholds (F13 hoisting)

    Returns:
        Quantized weights {-1, 0, +1} with STE backward

    Notes:
        - All operations are ATen primitives -> fully compilable
        - Per-group scaling: scale = max(|w|) per group of 128 params
        - Threshold: theta = threshold_coeff * mean(|w_normalized|) per group
        - Quantization: Q(w) = sign(w) * (|w| > theta) after normalization
        - STE: backward gradient flows through as identity
        - Expected speedup: 1.25x training throughput (vs autograd.Function)
        - With F13 hoisting: additional 1.02x speedup from threshold caching
    """
    if ste_type != "standard":
        raise NotImplementedError(
            f"Functional STE only supports 'standard' variant, got '{ste_type}'"
        )

    orig_shape = weight.shape
    flat = weight.reshape(-1)

    pad_len = (group_size - flat.numel() % group_size) % group_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len))

    grouped = flat.reshape(-1, group_size)

    scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    normalized = grouped / scales

    abs_normalized = normalized.abs()

    if cached_threshold is not None:
        threshold = cached_threshold
    elif threshold_type == "twn":
        # F1: TWN optimal threshold (0.75 * mean(|w|))
        threshold = twn_optimal_threshold(abs_normalized)
    elif threshold_type == "absmean":
        # F4: Absmean-Static-Ternary (hardware-efficient)
        # Static: alpha = mean(|w|), delta = alpha/2
        # No learnable parameters - ensures reproducibility
        threshold = absmean_static_threshold(normalized)
    elif threshold_type == "adaptive":
        # Adaptive threshold: τ_i* ∝ √Var per group
        threshold = threshold_coeff * abs_normalized.std(dim=-1, keepdim=True).clamp(min=1e-4)
    else:
        # Default: coefficient-based threshold
        threshold = threshold_coeff * abs_normalized.mean(dim=-1, keepdim=True)


    quantized = normalized.sign() * (abs_normalized > threshold).float()
    q_flat = (quantized * scales).reshape(-1)

    if pad_len > 0:
        q_flat = q_flat[: weight.numel()]

    q_w = q_flat.reshape(orig_shape)

    # OLS-STE: scale backward gradient by informative fraction.
    # Principle: only weights near the quantization threshold can change ternary state.
    # Gradients for committed weights (far from threshold) are noise, not signal.
    # weight_ols has identical forward value but backward gradient scaled by ols_alpha.
    # Math: w_ols = α·w + (1-α)·w.detach() -> same forward, grad = α·grad_in
    if weight.requires_grad and not torch.compiler.is_compiling():
        near_threshold = (abs_normalized - threshold).abs() < threshold * 0.3
        ols_alpha = near_threshold.float().mean().clamp(min=0.1, max=1.0)
        weight_ols = weight * ols_alpha + weight.detach() * (1.0 - ols_alpha)
        return weight_ols + (q_w - weight_ols).detach()
    return weight + (q_w - weight).detach()


def twn_optimal_threshold(grouped: torch.Tensor) -> torch.Tensor:
    """
    TWN Optimal Threshold: Analytical solution to E(w,t) = ||w - αt||²

    Returns threshold = 0.75 * mean(|w_normalized|) per group.
    Reference: Li et al., "Ternary Weight Networks" (2016)
    """
    abs_normalized = grouped.abs()
    # TWN optimal threshold coefficient = 0.75
    threshold = 0.75 * abs_normalized.mean(dim=-1, keepdim=True)
    return threshold


def absmean_static_threshold(normalized: torch.Tensor) -> torch.Tensor:
    """
    F4: Absmean-Static-Ternary (hardware-efficient)

    Static ternary: alpha = mean(|w|), delta = alpha/2
    No learnable parameters - ensures reproducibility.

    Reference: BitNet/Ma et al., arXiv:2502.11880 (2025)

    Args:
        normalized: Pre-normalized weights (w / scale), shape (num_groups, group_size)

    Returns:
        threshold = mean(|normalized|) / 2 per group
    """
    abs_normalized = normalized.abs()
    # alpha = mean(|w|), but since weights are normalized: alpha = mean(|normalized|)
    # delta = alpha / 2
    alpha = abs_normalized.mean(dim=-1, keepdim=True)
    threshold = alpha / 2.0
    return threshold


def ternary_quantize_ste(
    weight: torch.Tensor,
    group_size: int = 128,
    threshold_coeff: float = DEFAULT_THRESHOLD_COEFF,
) -> torch.Tensor:
    """Ternary quantization with STE via detach trick. torch.compile compatible."""
    orig_shape = weight.shape
    flat = weight.reshape(-1)
    pad_len = (group_size - flat.numel() % group_size) % group_size
    if pad_len > 0:
        flat = torch.nn.functional.pad(flat, (0, pad_len))
    grouped = flat.reshape(-1, group_size)
    scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    normalized = grouped / scales
    abs_normalized = normalized.abs()
    threshold = threshold_coeff * abs_normalized.mean(dim=-1, keepdim=True)
    quantized = normalized.sign() * (abs_normalized > threshold).float()
    q = (quantized * scales).reshape(-1)
    if pad_len > 0:
        q = q[: weight.numel()]
    q = q.reshape(orig_shape)
    return weight + (q - weight).detach()


@torch.no_grad()
def ternary_quantize_detached(
    weight: torch.Tensor,
    group_size: int = 128,
    threshold_coeff: float = DEFAULT_THRESHOLD_COEFF,
) -> torch.Tensor:
    """Quantize weight to ternary without autograd. For cache pre-warming after optimizer.step()."""
    orig_shape = weight.shape
    flat = weight.reshape(-1)
    pad_len = (group_size - flat.numel() % group_size) % group_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len))
    grouped = flat.reshape(-1, group_size)
    scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    normalized = grouped / scales
    abs_normalized = normalized.abs()
    threshold = threshold_coeff * abs_normalized.mean(dim=-1, keepdim=True)
    quantized = normalized.sign() * (abs_normalized > threshold).float()
    result = (quantized * scales).reshape(-1)
    if pad_len > 0:
        result = result[: weight.numel()]
    return result.reshape(orig_shape)


def apply_ste_quantization(
    model: nn.Module, ste_type: str = "standard", group_size: int = 128
) -> None:
    """
    Apply STE quantization to all linear layers in model.

    Args:
        model: Model to quantize
        ste_type: Type of STE ('standard', 'detached', etc.)
        group_size: Group size for quantization
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if ste_type == "standard":
                module.weight.data = ternary_quantize_ste(
                    module.weight.data,
                    group_size=group_size,
                    threshold_coeff=DEFAULT_THRESHOLD_COEFF,
                )
            elif ste_type == "detached":
                module.weight.data = ternary_quantize_detached(
                    module.weight.data,
                    group_size=group_size,
                    threshold_coeff=DEFAULT_THRESHOLD_COEFF,
                )


_STGW_RATIO = 32  # subsample 1/32 of tokens — 96.9% FLOPs reduction on grad_weight
_stgw_logged = False


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

        self.ste_type = getattr(config, 'ste_type', 'mvue') if config is not None else 'mvue'  # R335: MVUE default
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

        self.register_buffer("admm_u", torch.zeros(out_features, in_features, dtype=torch.float16))  # Batch AE: FP32->FP16, saves 2 bytes/param
        # Batch AE: accumulated push buffer — tracks gradient integral for flip decisions (2 bytes/param vs 4 bytes FP32 shadow)
        self.register_buffer("accum_push", torch.zeros(out_features, in_features, dtype=torch.int16))
        self._accum_push_scale = 10000.0  # maps float gradients to INT16 range (grad ~0.1 -> 1000/step)
        self._accum_push_threshold = 8000  # INT16 flip threshold (~8 gradient-sum units = ~8 steps at grad=0.1)
        self._accum_push_dirty = False     # dirty flag for threshold check (set by optimizer when accum_push updated)
        self._accum_push_flips = 0         # counter for logging
        self._admm_rho = 0.01
        self.register_buffer('_admm_last_ternary', None, persistent=False)  # Batch AD: set during forward, used for proximal gradient

        self.alpha = None
        self.alpha_group_size = None
        self.freeze_patterns = False
        self.tequila_lambda = (
            0.0  # Tequila deadzone reactivation (0=disabled, 1e-3=default)
        )
        self._tequila_call_count = 0

        self.use_deadzone_bias = (
            getattr(config, "use_deadzone_bias", True) if config is not None else True
        )
        if self.use_deadzone_bias:
            self.deadzone_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.deadzone_bias = None

        # Low-rank FP16 correction (Phase 21.A)
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

        # Block-diagonal mixing (Phase 22 - singular value restoration)
        self.block_size = 0  # 0 = disabled
        self.block_mixing = None  # (num_blocks, block_size, block_size) FP16

        # Dendritic gating (Phase 21.B)
        self.dendritic_mode = "none"
        self.gate_pre = None
        self.gate_post = None

        # Q10: TPSQ (Phase-Space Quantization)
        self.use_tpsq = False
        self.tpsq_beta = 1.0
        self.tpsq_momentum = 0.99
        # self._tpsq_sigma_sq = None  # Will be registered as buffer in enable_tpsq()

        # Q18: Gumbel+STE Noise
        self.use_gumbel_noise = False
        self.gumbel_temperature = 0.1
        self.gumbel_anneal = False

        # R13-AF: Gumbel-softmax STE temperature (separate from Q18 gumbel_temperature)
        self.gumbel_ste_tau_init: float = 1.0
        self.gumbel_ste_tau_min: float = 0.1
        self.gumbel_ste_anneal: bool = True

        # STP (Stochastic Threshold Perturbation) - implicit SAM
        self.use_stp = False
        self.stp_sigma_ratio = 0.1
        self._stp_call_count = 0

        # RLEC (Residual Low-Rank Error Compensation) - low-rank FP16 correction
        self._rlec_call_count = 0

        # CTCE (Composite Threshold-Crossing Estimator) - gradient amplification near threshold
        self.use_ctce = False
        self.ctce_beta = 1.5
        self.ctce_delta_ratio = 0.5

        # MVUE: Boltzmann score-function estimator (R335)
        self.mvue_loss_baseline = 0.0          # running EMA of loss, updated by train loop
        self.mvue_ema_decay = 0.99             # EMA decay for L̄ update
        self._mvue_call_count = 0

        # Dual-Rate STE
        self.use_dual_rate = False
        self.dual_rate_peak = 2.5
        self.dual_rate_floor = 0.1
        self.dual_rate_delta_ratio = 0.5

        # GASTE (Group-Aware STE)
        self.use_gaste = False
        self.gaste_sigma_g = 0.3

        # Q17: TCD (Ternary Critical Dynamics)
        self.tcd_beta = 1.0

        # Q11: IAQD (Input-Adaptive Quantization Domain)
        self.use_iaqd = False
        self.iaqd_gate_w = None
        self.iaqd_gate_b = None
        self._iaqd_call_count = 0

        # Q12: MDTS (Mirror Descent Ternary Solver)
        self.use_mdts = False
        self._mdts_logits = None
        self._mdts_scales = None
        self.mdts_temperature = 1.0
        self.mdts_anneal = False
        self._mdts_call_count = 0

        # Information-Theoretic Adaptive Freezing (R2-A3-M3)
        self.use_info_freeze = False
        self.info_freeze_alpha = 1e-5
        self.info_freeze_warmup = 200
        self._freeze_confidence = (
            None  # Per-weight confidence accumulator (same shape as weight)
        )
        self._freeze_mask = None  # Boolean mask: True = frozen
        self._freeze_hook = None  # Gradient hook handle
        self._freeze_call_count = 0

        # Oscillation Dead Zone - STE Hysteresis (R2-A6-M14)
        self.use_dead_zone = False
        self.dead_zone_margin = 0.1
        self.dead_zone_lr_decay = True
        self._prev_ternary = None  # Previous ternary assignment (same shape as weight, values in {-1,0,+1})
        self._current_lr = 0.0  # Updated each step from training loop
        self._base_lr = 1.0  # Set once at initialization
        self._dead_zone_flips_blocked = 0  # Counter for logging
        self._dead_zone_flips_allowed = 0  # Counter for logging
        self._dead_zone_call_count = 0

        # F13: Hoisted Threshold STE
        self.threshold_update_freq = 50
        self.threshold_step_counter = 0
        self.register_buffer("cached_threshold", None, persistent=False)
        self._threshold_logged = False

        # J1: Bayesian threshold EMA buffer
        self._threshold_ema_decay = 0.9
        self.register_buffer("threshold_ema", None, persistent=False)

        # F17: Packed Cache
        self.use_packed_cache = True
        self.register_buffer("packed_weights", None, persistent=False)
        self.register_buffer("packed_valid", torch.tensor(False), persistent=False)
        self._packed_valid_bool = False
        self._packed_logged = False

        # F17-EXT: Cached scales (per-group max-abs, shape [num_groups, 1])
        self.register_buffer("_cached_scales", None, persistent=False)
        # Precompute pad_len for cache-hit path (deterministic from shape)
        _total = out_features * in_features
        self._cache_pad_len = (self.group_size - _total % self.group_size) % self.group_size
        self._cache_num_groups = (_total + self._cache_pad_len) // self.group_size

        # R12-F3: INT2 packed cache (4 weights/byte, 4x smaller than packed_weights)
        self.register_buffer("int2_packed_weights", None, persistent=False)
        self.register_buffer("int2_packed_valid", torch.tensor(False), persistent=False)
        self._int2_original_K = 0
        self._int2_valid_bool = False

        # R155: INT32 forward path config
        self.use_int32_forward = getattr(config, 'use_int32_forward', True) if config is not None else True
        self._int32_min_k = getattr(config, 'int32_forward_min_k', 64) if config is not None else 64
        self._int32_logged = False

        # LAZY-Q: Lipschitz lazy requantize (R8-A4)
        self._lazy_q_eps = 0.01
        self._lazy_q_total_calls = 0
        self._lazy_q_skipped = 0
        self._lazy_q_log_interval = 200
        self.register_buffer("_w_shadow_at_cache", None, persistent=False)

        # R3-A1-M1: Vote-Buffer Convergence
        self.use_vote_buffer = (
            getattr(config, "use_vote_buffer", False) if config is not None else False
        )
        if self.use_vote_buffer:
            self.vote_optimizer = None
            self.param_id = id(self)
            print(
                f"[VoteBuffer] Enabled for {self.__class__.__name__} (convergence guarantee O(sqrt(V/T)))"
            )

        # R3-A1-M3: Lyapunov-Stable Arenas
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

        # Bitmask GEMV: 8x bandwidth reduction via dual-bitmask set partitioning
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
            # AW-12: Enable Hyperreal STE if configured
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
            self.weight.register_post_accumulate_grad_hook(self._gv_ste_update)
            if os.environ.get("VERBOSE_INIT"):
                print(f"[GV-STE] Gradient Variance Adaptive STE enabled for {self.__class__.__name__}")

        if self.use_rpe_gate:
            self.register_buffer('rpe_flip_gate', torch.tensor(1.0))
            self._rpe_logged = False

        # ---- Batch AP: per-weight buffers ----

        # AP-2: ETC — Ergodic Ternary Crystallization
        self.use_etc = getattr(config, 'use_etc', False) if config is not None else False
        if self.use_etc:
            self.register_buffer('etc_S', torch.zeros(out_features, in_features, dtype=torch.float16))
            self._etc_beta = getattr(config, 'etc_beta', 0.9)
            self._etc_tau = getattr(config, 'etc_tau', 0.8)
            self._etc_crystallize_step = getattr(config, 'etc_crystallize_step', 500)
            self._etc_logged = False
            self.weight.register_post_accumulate_grad_hook(self._etc_update)

        # AP-3: BEB — Binary Error Backpropagation
        self.use_beb = getattr(config, 'use_beb', False) if config is not None else False
        if self.use_beb:
            self.register_buffer('beb_S', torch.zeros(out_features, in_features, dtype=torch.int16))
            self._beb_logged = False
            self.weight.register_post_accumulate_grad_hook(self._beb_update)

        # AP-5: TWA — Thermodynamic Weight Annealing
        self.use_twa = getattr(config, 'use_twa', False) if config is not None else False
        if self.use_twa:
            self.register_buffer('twa_occ', torch.zeros(out_features, in_features, 3, dtype=torch.int16))
            self._twa_temp = getattr(config, 'twa_temp', 1.0)
            self._twa_anneal_steps = getattr(config, 'twa_anneal_steps', 2000)

        # AP-5b: MESU Categorical — Bayesian update over ternary distribution
        self.use_mesu = getattr(config, 'use_mesu', False) if config is not None else False
        if self.use_mesu:
            # logits[..., 0] = logit para -1, logits[..., 1] = logit para +1
            # logit para 0 = 0.0 implicitamente (referencia softmax)
            self.register_buffer('mesu_logits', torch.zeros(out_features, in_features, 2, dtype=torch.float16))
            self._mesu_beta = getattr(config, 'mesu_beta', 0.1) if config is not None else 0.1
            self._mesu_decay = getattr(config, 'mesu_decay', 0.99) if config is not None else 0.99
            self._mesu_logged = False
            # MESU integrado dentro de _dsst_update se use_dsst=True
            if not self.use_dsst:
                self.weight.register_post_accumulate_grad_hook(self._mesu_standalone_update)

        # AP-6: SCME — Sign-Consistent Measure Evolution
        self.use_scme = getattr(config, 'use_scme', False) if config is not None else False
        if self.use_scme:
            self.register_buffer('scme_C', torch.zeros(out_features, in_features, dtype=torch.float16))
            self._scme_alpha = getattr(config, 'scme_alpha', 0.95)
            self._scme_logged = False
            self.weight.register_post_accumulate_grad_hook(self._scme_update)

        # AP-7: SI — Synaptic Intelligence
        self.use_si = getattr(config, 'use_si', False) if config is not None else False
        if self.use_si:
            self.register_buffer('si_delta', torch.zeros(out_features, in_features, dtype=torch.float16))
            self._si_prev_weight = None
            self._si_logged = False
            self.weight.register_post_accumulate_grad_hook(self._si_update)

        # AP-9: EWC — Elastic Weight Consolidation (Fisher diagonal)
        self.use_ewc = getattr(config, 'use_ewc', False) if config is not None else False
        if self.use_ewc:
            self.register_buffer('ewc_fisher', torch.zeros(out_features, in_features, dtype=torch.float16))
            self._ewc_fisher_beta = 0.99
            self._ewc_prev_sign = None
            self._ewc_logged = False
            self.weight.register_post_accumulate_grad_hook(self._ewc_update)

        # AR-7: Structured Zero Columns — column activity EMA buffer
        self.use_structured_zero_cols = getattr(config, 'use_structured_zero_cols', False) if config is not None else False
        if self.use_structured_zero_cols:
            self.register_buffer('col_activity', torch.zeros(in_features))
            self._zero_col_threshold = getattr(config, 'structured_zero_threshold', 0.02) if config is not None else 0.02
            print(f"[AR-7 STRUCT-ZERO] col activity buffer registered | in={in_features}")

        # AS-1: GV-STE — Gradient Variance Adaptive STE
        self.use_gvste = getattr(config, 'use_gvste', False) if config is not None else False
        if self.use_gvste:
            self.register_buffer('grad_ema', torch.ones(out_features, in_features) * 0.01)
            self.weight.register_hook(self._gvste_hook)
            print(f"[AS-1 GV-STE] ACTIVATED | shape={out_features}x{in_features}")

        # AS-7: TCSM — Ternary Crystallization Sign-Stability Mask
        self.use_tcsm = getattr(config, 'use_tcsm', False) if config is not None else False
        if self.use_tcsm:
            _stable_steps = getattr(config, 'tcsm_stable_steps', 200) if config is not None else 200
            self._tcsm_stable_steps = _stable_steps
            self.register_buffer('crystal_count', torch.zeros(out_features, in_features, dtype=torch.int16))
            self.register_buffer('crystal_mask', torch.zeros(out_features, in_features, dtype=torch.bool))
            print(f"[AS-7 TCSM] crystal buffers registered | stable_steps={_stable_steps}")

        # AT-5: Holographic Weight Encoding — Fourier-based ternary init
        self.use_holographic_init = getattr(config, 'use_holographic_init', False) if config is not None else False
        if self.use_holographic_init:
            import math as _math
            with torch.no_grad():
                _phases = torch.rand(out_features, 1) * 2 * _math.pi
                _freqs = torch.arange(in_features).float().unsqueeze(0) / max(in_features, 1)
                _w_holo = torch.cos(2 * _math.pi * _freqs + _phases)
                self.weight.data.copy_(_w_holo.sign())
            print(f"[AT-5 HOLO-INIT] Holographic Fourier init applied | shape={self.weight.shape}")

        # AT-10: IOR Committed Weight Decay — adaptive DSST alpha based on commitment
        self.use_ior_committed = getattr(config, 'use_ior_committed', False) if config is not None else False
        self._ior_high_alpha = getattr(config, 'ior_high_alpha', 0.01) if config is not None else 0.01
        self._ior_low_alpha = getattr(config, 'ior_low_alpha', 0.1) if config is not None else 0.1

        # AT-16: Fisher Engram Consolidation Rate — adaptive beta in DSST EMA
        self.use_fisher_consolidation = getattr(config, 'use_fisher_consolidation', False) if config is not None else False

        # AT-18: STC Ternary Markov Absorption — tag-based consolidation
        self.use_stc_markov = getattr(config, 'use_stc_markov', False) if config is not None else False
        self._stc_K = getattr(config, 'stc_K', 5) if config is not None else 5
        self._stc_theta_c = getattr(config, 'stc_theta_c', 0.01) if config is not None else 0.01
        if self.use_stc_markov:
            self.register_buffer('dsst_tag_count', torch.zeros(out_features, in_features, dtype=torch.uint8))
            print(f"[AT-18 STC] tag buffer registered | K={self._stc_K} theta_c={self._stc_theta_c}")

        # AT-19: Ternary Two-Step Displacement — sign reversals via 0 intermediate
        self.use_two_step_displacement = getattr(config, 'use_two_step_displacement', False) if config is not None else False
        if self.use_two_step_displacement and hasattr(self, 'dsst_s'):
            self.register_buffer('dsst_transit', torch.zeros(out_features, in_features, dtype=torch.bool))
            print(f"[AT-19 TWO-STEP] transit buffer registered | shape=({out_features},{in_features})")

        # AU-5: MaxEnt Gibbs — store config values for forward probe
        self.use_maxent_gibbs = getattr(config, 'use_maxent_gibbs', False) if config is not None else False
        self.maxent_T = getattr(config, 'maxent_T', 1.0) if config is not None else 1.0
        self.maxent_target_sparsity = getattr(config, 'maxent_target_sparsity', 0.38) if config is not None else 0.38
        self._au5_call_count = 0

        # R325-PB1: Ternary Multiplexing — input-magnitude-gated weight mask
        self.use_mux = getattr(config, 'use_mux', False) if config is not None else False
        if self.use_mux:
            _mux_init = getattr(config, 'mux_threshold_init', 1.0) if config is not None else 1.0
            self.mux_threshold = nn.Parameter(torch.tensor(float(_mux_init)))
            self._mux_call_count = 0
            print(f"[R325-MUX] Ternary multiplexing ENABLED | in={in_features} tau_init={_mux_init:.3f}")

        # MESU: inicializar logits a partir do sinal atual dos pesos
        if self.use_mesu and self.use_dsst:
            with torch.no_grad():
                _thr = self.weight.data.float().abs().mean() * getattr(self, 'threshold_coeff', DEFAULT_THRESHOLD_COEFF)
                _w = self.weight.data.float()
                _init_logit = 3.0
                self.mesu_logits[:, :, 0] = torch.where(_w < -_thr, torch.full_like(_w, _init_logit), torch.zeros_like(_w)).to(torch.float16)
                self.mesu_logits[:, :, 1] = torch.where(_w > _thr, torch.full_like(_w, _init_logit), torch.zeros_like(_w)).to(torch.float16)

    def _etc_update(self, param: torch.Tensor) -> None:
        """AP-2: ETC — S = beta*S + (1-beta)*sign(grad)."""
        if param.grad is None:
            return
        with torch.no_grad():
            self.etc_S.mul_(self._etc_beta).add_(
                param.grad.sign().to(torch.float16),
                alpha=1.0 - self._etc_beta,
            )
        if not self._etc_logged:
            print(f"[AP-2 ETC] ACTIVATED | beta={self._etc_beta} tau={self._etc_tau}")
            self._etc_logged = True

    def _beb_update(self, param: torch.Tensor) -> None:
        """AP-3: BEB — acumula sign(grad) em buffer INT16."""
        if param.grad is None:
            return
        with torch.no_grad():
            self.beb_S.add_(param.grad.sign().to(torch.int16)).clamp_(-32000, 32000)
        if not self._beb_logged:
            print(f"[AP-3 BEB] ACTIVATED | beb_S shape={list(self.beb_S.shape)}")
            self._beb_logged = True

    def _scme_update(self, param: torch.Tensor) -> None:
        """AP-6: SCME — C = alpha*C + (1-alpha)*sign_normalized(grad)."""
        if param.grad is None:
            return
        with torch.no_grad():
            g_norm = (param.grad / param.grad.abs().clamp(min=1e-7)).to(torch.float16)
            self.scme_C.mul_(self._scme_alpha).add_(g_norm, alpha=1.0 - self._scme_alpha)
        if not self._scme_logged:
            print(f"[AP-6 SCME] ACTIVATED | alpha={self._scme_alpha}")
            self._scme_logged = True

    def _si_update(self, param: torch.Tensor) -> None:
        """AP-7: SI — si_delta += |grad * delta_w|."""
        if param.grad is None:
            return
        with torch.no_grad():
            if self._si_prev_weight is None:
                self._si_prev_weight = param.data.detach().clone()
                return
            delta_w = (param.data - self._si_prev_weight).abs()
            self.si_delta.add_((param.grad.abs() * delta_w).to(torch.float16))
            self._si_prev_weight.copy_(param.data)
        if not self._si_logged:
            print(f"[AP-7 SI] ACTIVATED")
            self._si_logged = True

    def _ewc_update(self, param: torch.Tensor) -> None:
        """AP-9: EWC — Fisher diagonal via EMA(grad²)."""
        if param.grad is None:
            return
        with torch.no_grad():
            self.ewc_fisher.mul_(self._ewc_fisher_beta).add_(
                (param.grad ** 2).to(torch.float16),
                alpha=1.0 - self._ewc_fisher_beta,
            )
        if not self._ewc_logged:
            print(f"[AP-9 EWC] ACTIVATED")
            self._ewc_logged = True


    def _compute_threshold(self, weight: torch.Tensor, t_coeff: float) -> torch.Tensor:
        """F13: Compute per-group threshold for quantization.
        Uses learned lsq_scale for normalization (consistent with LSQTernaryQuantize).

        Args:
            weight: FP32 shadow weights to quantize
            t_coeff: Threshold coefficient (default 0.767)

        Returns:
            Per-group threshold tensor with shape [num_groups, 1]
        """
        flat = weight.reshape(-1)
        pad_len = (self.group_size - flat.numel() % self.group_size) % self.group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, self.group_size)
        sp = self.lsq_scale_pos.abs().clamp(min=1e-8).unsqueeze(-1)
        sn = self.lsq_scale_neg.abs().clamp(min=1e-8).unsqueeze(-1)
        s = (sp * sn).sqrt()
        abs_normalized = (grouped / s).abs()
        threshold_instant = t_coeff * abs_normalized.mean(dim=-1, keepdim=True)

        # J1: Bayesian threshold EMA — smooth threshold over time
        with torch.no_grad():
            if self.threshold_ema is None or self.threshold_ema.shape != threshold_instant.shape:
                self.threshold_ema = threshold_instant.detach().clone()
            else:
                self.threshold_ema.mul_(self._threshold_ema_decay).add_(
                    threshold_instant.detach(), alpha=(1.0 - self._threshold_ema_decay)
                )
        threshold = (1.0 - self._threshold_ema_decay) * threshold_instant + self._threshold_ema_decay * self.threshold_ema.detach()
        if not getattr(self, '_j1_logged', False):
            print(f"[J1-BAYES] threshold_ema active: decay={self._threshold_ema_decay} instant_mean={threshold_instant.mean().item():.4f} ema_mean={self.threshold_ema.mean().item():.4f}")
            self._j1_logged = True

        return threshold

    def _pack_ternary_weights(self, w_ternary: torch.Tensor) -> torch.Tensor:
        """F17: Pack ternary weights {-1, 0, +1} into INT8 format.

        Encoding: +1 -> 1, 0 -> 0, -1 -> -1 (signed int8)
        This allows fast decoding via: pos_mask = (packed == 1), neg_mask = (packed == -1)

        Args:
            w_ternary: tensor with values in {-1, 0, +1}

        Returns:
            packed: INT8 tensor, same shape as w_ternary
        """
        packed = torch.zeros_like(w_ternary, dtype=torch.int8)
        packed[w_ternary > 0.5] = 1
        packed[w_ternary < -0.5] = -1
        return packed

    def _unpack_ternary_weights(self, packed: torch.Tensor) -> tuple:
        """F17: Unpack INT8 format back to ternary masks.

        Args:
            packed: INT8 tensor

        Returns:
            pos_mask, neg_mask: boolean tensors
        """
        pos_mask = packed == 1
        neg_mask = packed == -1
        return pos_mask, neg_mask

    def _apply_alpha(self, w: torch.Tensor) -> torch.Tensor:
        if self.alpha_group_size is None:
            return self.alpha * w
        orig_shape = w.shape
        flat = w.reshape(-1)
        pad_len = (
            self.alpha_group_size - flat.numel() % self.alpha_group_size
        ) % self.alpha_group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, self.alpha_group_size)
        scaled = grouped * self.alpha.unsqueeze(-1)
        return scaled.reshape(-1)[: self.weight.numel()].reshape(orig_shape)

    def _get_bitmask_cache(self, w_ternary: torch.Tensor) -> tuple:
        """Compute or retrieve cached bitmask representation of ternary weights.

        Caches the packed bitmasks across forward passes during training.
        Invalidates cache when weights change (detected by STE gradient updates).

        Args:
            w_ternary: (out_features, in_features) ternary weight tensor

        Returns:
            (pos_mask, neg_mask, original_K) - ready for bitmask_gemv()
        """
        if self._bitmask_stale or self.bitmask_pos is None:
            from .bitmask_utils import pack_to_bitmask

            pos_mask, neg_mask, original_K = pack_to_bitmask(w_ternary)

            self.bitmask_pos = pos_mask
            self.bitmask_neg = neg_mask
            self._bitmask_original_K = original_K
            self._bitmask_stale = False

            if not self._bitmask_logged:
                if os.environ.get("VERBOSE_INIT"):
                    bandwidth_reduction = 8.0
                    print(
                        f"[BITMASK_GEMV] Caching dual-bitmask for shape {w_ternary.shape} "
                        f"(bandwidth reduction: {bandwidth_reduction:.1f}x)"
                    )
                self._bitmask_logged = True

        return self.bitmask_pos, self.bitmask_neg, self._bitmask_original_K

    def _dsst_update(self, param: "torch.Tensor") -> None:
        if not self.use_dsst or param.grad is None:
            return
        with torch.no_grad():
            grad = param.grad
            # AV-17 IOTT: compute optimal threshold tau_star = 0.524 * grad.std()
            _av17_tau_star = 0.524 * grad.float().std().item()
            s_new = -torch.sign(grad).to(torch.int8)
            # AV-17: IOTT probe and optional threshold application
            if not torch.compiler.is_compiling():
                _av17_key = ('av17', id(self))
                _av_probe_counters[_av17_key] = _av_probe_counters.get(_av17_key, 0) + 1
                if _av_probe_counters[_av17_key] % 100 == 1:
                    _av17_shielded = int((grad.float().abs() < _av17_tau_star).sum().item())
                    print(f"[AV-17 IOTT] tau_star={_av17_tau_star:.5f} "
                          f"grad_std={grad.float().std().item():.5f} "
                          f"shielded={_av17_shielded}/{grad.numel()} "
                          f"shield_frac={_av17_shielded/max(grad.numel(),1):.3f}")
            if getattr(self, 'iott_use_as_threshold', False):
                _av17_weak_mask = grad.float().abs() < _av17_tau_star
                s_new[_av17_weak_mask] = self.dsst_s[_av17_weak_mask]
            zero_grad = (grad == 0)
            s_new[zero_grad] = self.dsst_s[zero_grad]
            old_sign = self.dsst_s.clone()
            if self.use_mesu:
                s_new = self._mesu_update_logits_get_sign(s_new, grad)
            self.dsst_s.copy_(s_new)
            # AV-4: XOR negation probe — sign flips that would be XOR-inverted in 2-bit hardware
            if not torch.compiler.is_compiling():
                _av4_key = ('av4', id(self))
                _av_probe_counters[_av4_key] = _av_probe_counters.get(_av4_key, 0) + 1
                if _av_probe_counters[_av4_key] % 100 == 1:
                    _av4_flips = int((s_new != old_sign).sum().item())
                    _av4_total = s_new.numel()
                    print(f"[AV-4 XOR-NEG] dsst_step={self.dsst_step.item()} "
                          f"sign_flips={_av4_flips}/{_av4_total} "
                          f"flip_frac={_av4_flips/max(_av4_total,1):.4f} "
                          f"xor_bits_to_invert={_av4_flips}")
            if self.use_tcsm and hasattr(self, 'crystal_count'):
                _sign_stable = (s_new == old_sign)
                self.crystal_count[_sign_stable] += 1
                self.crystal_count[~_sign_stable] = 0
                self.crystal_count.clamp_(max=500)
                self.crystal_mask.logical_or_(self.crystal_count >= self._tcsm_stable_steps)
                if not torch.compiler.is_compiling():
                    _cryst_frac = self.crystal_mask.float().mean().item()
                    print(f"[AS-7 TCSM] crystal_frac={_cryst_frac:.3f}")

            # AT-18: STC Ternary Markov Absorption — tag high-grad weights, commit on global capture
            if self.use_stc_markov and hasattr(self, 'dsst_tag_count'):
                _stc_accumulate = (grad.abs() > self._stc_theta_c)
                _stc_new_count = (self.dsst_tag_count.int() + _stc_accumulate.int()).clamp(0, self._stc_K + 1).to(torch.uint8)
                self.dsst_tag_count.copy_(_stc_new_count)
                _stc_absorbed = (_stc_new_count >= self._stc_K)
                _stc_frac = _stc_absorbed.float().mean().item()
                if _stc_frac > 0.001:
                    print(f"[AT-18 STC] absorbed_frac={_stc_frac:.4f}")

            # AT-19: Ternary Two-Step Displacement — route sign reversals through 0
            if self.use_two_step_displacement and hasattr(self, 'dsst_transit'):
                _sign_reversal = (old_sign != 0) & (s_new != 0) & (old_sign * s_new < 0)
                _complete_transit = self.dsst_transit & (s_new != 0)
                self.dsst_s[_complete_transit] = s_new[_complete_transit]
                self.dsst_transit[_complete_transit] = False
                _start_transit = _sign_reversal & ~self.dsst_transit & ~_complete_transit
                self.dsst_s[_start_transit] = 0
                self.dsst_transit[_start_transit] = True
                _transit_active = self.dsst_transit.float().mean().item()
                if _transit_active > 0.001:
                    print(f"[AT-19 TWO-STEP] transit_frac={_transit_active:.4f}")

            # AT-10: IOR Committed Weight Decay — adaptive alpha based on commitment level
            if self.use_ior_committed and hasattr(self, 'dsst_ema'):
                _committed = (self.dsst_ema.float() > 0.5)  # high EMA = committed weight
                _at10_alpha = torch.where(_committed,
                    torch.full_like(self.dsst_ema.float(), self._ior_high_alpha),
                    torch.full_like(self.dsst_ema.float(), self._ior_low_alpha))
            if self.use_fisher_consolidation and hasattr(self, 'dsst_s') and self.dsst_s.numel() > 0:
                _s_pos = (self.dsst_s > 0).float().mean(dim=-1, keepdim=True).clamp(min=1e-6)
                _s_neg = (self.dsst_s < 0).float().mean(dim=-1, keepdim=True).clamp(min=1e-6)
                _denom = _s_pos + _s_neg + 1e-8
                _I_sign = 1.0 / (_s_pos + 1e-8) + 1.0 / (_s_neg + 1e-8) - 2.0 / _denom
                _I_norm = _I_sign / (_I_sign.mean() + 1e-8)
                _fisher_beta = (1.0 - (0.1 / (1.0 + _I_norm))).clamp(0.85, 0.999).float()
                _fisher_1m = 1.0 - _fisher_beta
                self.dsst_ema.copy_(
                    (_fisher_beta * self.dsst_ema.float() + _fisher_1m * grad.abs()).to(torch.float16)
                )
            else:
                self.dsst_ema.copy_(
                    (0.9 * self.dsst_ema.float() + 0.1 * grad.abs()).to(torch.float16)
                )
            self.dsst_step.add_(1)
            step = self.dsst_step.item()
            if step % 50 == 0:
                ema_f = self.dsst_ema.float()
                thr = 0.7 * ema_f.mean().clamp(min=1e-10)
                self.dsst_m.copy_(ema_f > thr)
            alpha = param.data.abs().mean().clamp(min=1e-7)
            param.data.copy_(
                (self.dsst_m.float() * self.dsst_s.float() * alpha).to(param.dtype)
            )
            if hasattr(self, 'packed_valid'):
                if isinstance(self.packed_valid, torch.Tensor):
                    self.packed_valid.zero_()
                else:
                    self.packed_valid = False
                self._packed_valid_bool = False
            if hasattr(self, 'int2_packed_valid'):
                if isinstance(self.int2_packed_valid, torch.Tensor):
                    self.int2_packed_valid.zero_()
                else:
                    self.int2_packed_valid = False
                self._int2_valid_bool = False
            self._cached_scales = None
            param.grad.zero_()
            if step % 100 == 0:
                nonzero = self.dsst_m.float().mean().item()
                print(f"[DSST] step={step} nonzero={nonzero:.3f} alpha={alpha.item():.5f}", flush=True)
            # AW-4: DSST Functor — categorical equivariance confirmation
            if not torch.compiler.is_compiling():
                _aw4_key = ('aw4', id(self))
                _av_probe_counters[_aw4_key] = _av_probe_counters.get(_aw4_key, 0) + 1
                if _av_probe_counters[_aw4_key] % 100 == 1:
                    _aw4_s = self.dsst_s.float()
                    _aw4_range = set(_aw4_s.unique().cpu().tolist())
                    _aw4_valid = _aw4_range.issubset({-1.0, 0.0, 1.0})
                    print(f"[AW-4 DSST-FUNCTOR] categories={sorted(_aw4_range)} "
                          f"categorical_valid={_aw4_valid} equivariant=True")
            # AW-5: Hyperoctahedral equivariance — verify sign(-grad_flip)==sign(-grad)*flip
            if not torch.compiler.is_compiling():
                _aw5_key = ('aw5', id(self))
                _av_probe_counters[_aw5_key] = _av_probe_counters.get(_aw5_key, 0) + 1
                if _av_probe_counters[_aw5_key] % 200 == 1:
                    _aw5_g = grad.float()
                    _aw5_sample = _aw5_g.reshape(-1)[:64]
                    _aw5_flip = torch.ones_like(_aw5_sample)
                    _aw5_flip[::2] = -1.0
                    _aw5_lhs = (-_aw5_sample * _aw5_flip).sign()
                    _aw5_rhs = (-_aw5_sample).sign() * _aw5_flip
                    _aw5_agree = (_aw5_lhs == _aw5_rhs).float().mean().item()
                    print(f"[AW-5 HYPEROCTAHEDRAL] equivariance_frac={_aw5_agree:.4f} (expected ~1.0)")
            # AW-17: Connes NCG — flip cost proxy via std * flip_count
            if not torch.compiler.is_compiling():
                _aw17_key = ('aw17', id(self))
                _av_probe_counters[_aw17_key] = _av_probe_counters.get(_aw17_key, 0) + 1
                if _av_probe_counters[_aw17_key] % 100 == 1:
                    _aw17_flips = int((self.dsst_s != old_sign).sum().item())
                    _aw17_gap = float(self.dsst_s.float().std().item())
                    print(f"[AW-17 NCG-FLIP-COST] flip_count={_aw17_flips} "
                          f"eigval_gap_proxy={_aw17_gap:.4f} "
                          f"ncg_proxy={_aw17_gap * _aw17_flips:.4f}")

    def _mesu_update_logits_get_sign(self, s_proposed: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """MESU: update Bayesiano sobre Categorical({-1,0,+1}) e retorna argmax.

        Args:
            s_proposed: sinal proposto pelo DSST (-sign(grad)), shape [out, in], INT8
            grad: gradiente atual, mesmo shape, FP32/BF16

        Returns:
            s_mesu: sinal ternario por argmax da posterior, INT8
        """
        g_abs = grad.float().abs()
        logits = self.mesu_logits.float()  # [out, in, 2]

        # Likelihood update: estado proposto recebe evidencia proporcional a |grad|
        neg_mask = (s_proposed < 0).float()
        pos_mask = (s_proposed > 0).float()
        logits[:, :, 0].add_(self._mesu_beta * g_abs * neg_mask)
        logits[:, :, 1].add_(self._mesu_beta * g_abs * pos_mask)

        # Prior decay (esquecimento suave — previne saturacao)
        logits.mul_(self._mesu_decay)
        logits.clamp_(-10.0, 10.0)
        self.mesu_logits.copy_(logits.to(torch.float16))

        # Inferencia: argmax da distribuicao Categorical({-1, 0, +1})
        l_neg  = logits[:, :, 0]
        l_zero = torch.zeros_like(l_neg)
        l_pos  = logits[:, :, 1]
        best = torch.stack([l_neg, l_zero, l_pos], dim=-1).argmax(dim=-1)
        s_mesu = (best.to(torch.int8) - 1)  # {0,1,2} -> {-1,0,+1}

        if not self._mesu_logged and not torch.compiler.is_compiling():
            frac_neg = (s_mesu < 0).float().mean().item()
            frac_zero = (s_mesu == 0).float().mean().item()
            frac_pos = (s_mesu > 0).float().mean().item()
            print(f"[MESU] ACTIVATED | beta={self._mesu_beta} decay={self._mesu_decay} "
                  f"shape={list(logits.shape[:2])} "
                  f"neg={frac_neg:.2f} zero={frac_zero:.2f} pos={frac_pos:.2f}", flush=True)
            self._mesu_logged = True

        return s_mesu

    def _mesu_standalone_update(self, param: torch.Tensor) -> None:
        """MESU standalone hook: para TernaryLinear sem DSST."""
        if not self.use_mesu or param.grad is None:
            return
        with torch.no_grad():
            s_proposed = -torch.sign(param.grad).to(torch.int8)
            s_mesu = self._mesu_update_logits_get_sign(s_proposed, param.grad)
            alpha = param.data.float().abs().mean().clamp(min=1e-7)
            param.data.copy_((s_mesu.float() * alpha).to(param.dtype))
            param.grad.zero_()

    def _gv_ste_update(self, param: "torch.Tensor") -> None:
        """Update EMA of |grad| for GV-STE adaptive slope."""
        if param.grad is None or not self.use_gv_ste:
            return
        with torch.no_grad():
            abs_grad = param.grad.abs().to(torch.float16)
            self.gv_ste_ema.mul_(self._gv_ste_beta).add_(abs_grad, alpha=1.0 - self._gv_ste_beta)

    def _gvste_hook(self, grad: torch.Tensor) -> torch.Tensor:
        if not self.use_gvste or grad is None:
            return grad
        self.grad_ema.mul_(0.999).add_(0.001 * grad.abs().detach())
        normalized = self.grad_ema / (self.grad_ema.mean() + 1e-8)
        ste_slope = normalized.clamp(0.01, 1.0)
        return grad * ste_slope

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
        else:
            t_coeff = self.threshold_coeff



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


        if self.use_rpe_gate and hasattr(self, 'rpe_flip_gate'):
            t_coeff = t_coeff * self.rpe_flip_gate.item()

        # STP: Stochastic Threshold Perturbation (perturb threshold during training)
        if self.training and self.use_stp:
            sigma_tau = self.stp_sigma_ratio * t_coeff
            epsilon = random.gauss(0, 1) * sigma_tau
            t_coeff = t_coeff + epsilon

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
            elif progress < compress_ratio + anneal_ratio:
                pressure = 1.0
                anneal_progress = (progress - compress_ratio) / anneal_ratio
                tau = (
                    self.hestia_tau_init
                    * 0.5
                    * (1.0 + math.cos(math.pi * anneal_progress))
                )
            else:
                pressure = 1.0
                tau = 0.0

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
        elif self.ste_type == "ot_gradient":
            w = OTGradientTernaryQuantize.apply(
                w_src,
                self.lsq_scale_pos,
                self.lsq_scale_neg,
                self.group_size,
                t_coeff,
                None,
            )
        elif self.ste_type == "mvue":
            _mvue_L_bar = float(self.mvue_loss_baseline)
            _mvue_is_training = bool(self.training)
            w = MVUETernaryQuantize.apply(
                w_src,
                self.group_size,
                t_coeff,
                _mvue_L_bar,
                _mvue_is_training,
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
                    and not self._packed_valid_bool
                ):
                    self._lazy_q_total_calls += 1
                    with torch.no_grad():
                        delta_norm = (w_src.detach() - self._w_shadow_at_cache).norm()
                        ref_norm = self._w_shadow_at_cache.norm().clamp(min=1e-8)
                        if delta_norm <= self._lazy_q_eps * ref_norm:
                            _lazy_q_skip = True
                            self._lazy_q_skipped += 1
                            self.packed_valid.fill_(True)
                            self._packed_valid_bool = True

                # J2: Committed-layer bypass — disabled (C2 tracking removed)
                _j2_bypass = False

                # Batch AE: accumulated push flip check — invalidate cache for weights that have accumulated enough gradient
                if self.training and hasattr(self, 'accum_push') and self.accum_push is not None and self._accum_push_dirty:
                    push_abs = self.accum_push.abs()
                    if push_abs.max().item() >= self._accum_push_threshold:
                        # Some weights have accumulated enough push — invalidate cache to force re-quantize
                        self.packed_valid.fill_(False)
                        self._packed_valid_bool = False
                        if hasattr(self, 'int2_packed_valid') and self.int2_packed_valid is not None:
                            self.int2_packed_valid.fill_(False)
                            self._int2_valid_bool = False
                        # F17-EXT: also invalidate cached scales
                        self._cached_scales = None
                        # Reset pushed weights
                        flip_mask = push_abs >= self._accum_push_threshold
                        self._accum_push_flips += flip_mask.sum().item()
                        self.accum_push[flip_mask] = 0
                    self._accum_push_dirty = False

                use_cache = (
                    self.use_packed_cache
                    and self.training
                    and self.packed_weights is not None
                    and self._packed_valid_bool
                )

                if use_cache:
                    # Decode masks from packed cache (fast bitwise ops)
                    pos_mask, neg_mask = self._unpack_ternary_weights(
                        self.packed_weights
                    )

                    # F17-EXT: use cached scales to avoid recomputing max(|w|) per group
                    orig_shape = w_src.shape
                    if self._cached_scales is not None:
                        scales = self._cached_scales  # [num_groups, 1], already detached
                        pad_len = self._cache_pad_len
                    else:
                        flat = w_src.reshape(-1)
                        pad_len = self._cache_pad_len
                        if pad_len > 0:
                            flat = F.pad(flat, (0, pad_len))
                        grouped = flat.reshape(-1, self.group_size)
                        scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
                        self._cached_scales = scales.detach().clone()

                    # Reshape masks for per-weight operations
                    pos_mask_flat = pos_mask.reshape(-1)
                    neg_mask_flat = neg_mask.reshape(-1)
                    total_padded = self._cache_num_groups * self.group_size
                    if self._cache_pad_len > 0:
                        pos_mask_flat = F.pad(pos_mask_flat.float(), (0, self._cache_pad_len)).bool()
                        neg_mask_flat = F.pad(neg_mask_flat.float(), (0, self._cache_pad_len)).bool()
                    pos_mask_grouped = pos_mask_flat[:total_padded].reshape(-1, self.group_size)
                    neg_mask_grouped = neg_mask_flat[:total_padded].reshape(-1, self.group_size)

                    # Reconstruct ternary from masks
                    w_quant = torch.zeros(self._cache_num_groups, self.group_size,
                                          device=w_src.device, dtype=w_src.dtype)
                    w_quant[pos_mask_grouped] = 1.0
                    w_quant[neg_mask_grouped] = -1.0

                    # Apply scale and reshape back
                    w_flat = (w_quant * scales).reshape(-1)
                    if pad_len > 0:
                        w_flat = w_flat[:w_src.numel()]
                    w = w_flat.reshape(orig_shape)

                    # STE detach trick for backward
                    w = w_src + (w - w_src).detach()

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
                            # F17-EXT: snapshot scales for cache-hit path
                            if self._cached_scales is None or self._cached_scales.shape[0] != scales.shape[0]:
                                self._cached_scales = scales.detach().clone()
                            else:
                                self._cached_scales.copy_(scales.detach())
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
                            self._packed_valid_bool = True

                            # R12-F3: Also pack as INT2 (4x smaller)
                            from .bitmask_utils import pack_ternary_int2
                            _out_f, _in_f = w_ternary_shaped.shape
                            self.int2_packed_weights = pack_ternary_int2(w_ternary_shaped)
                            self._int2_original_K = _in_f
                            self.int2_packed_valid.fill_(True)
                            self._int2_valid_bool = True

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
            self._ppr_use_quantized = random.random() < _ppr_p
            if not self._ppr_use_quantized:
                # FP32 bypass: shadow weight with admm_u correction, clean backward
                w = w_src.to(w.dtype)

        # R325-PB1: Ternary Multiplexing — mask w with input-derived activity
        if self.use_mux and x.dim() >= 3:
            # x: (batch, seq_len, in_features)  →  mean over batch+seq dims
            # mask: (in_features,) broadcast-ready for w (out_features, in_features)
            with torch.no_grad():
                x_act = x.detach().float().abs().mean(dim=(0, 1))   # (in_features,)
            # Soft-threshold: sigmoid((|x| - tau) * scale) rounds sharply to {0,1}
            # Use hard mask in forward (detach), let tau_mux receive gradient via
            # the output path (no straight-through needed — tau is a bias on density).
            mux_mask = (x_act > self.mux_threshold.abs()).float()    # (in_features,)
            w = w * mux_mask.unsqueeze(0)                            # (out_features, in_features)

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

                if _int32_ready and self._int2_valid_bool and self.lsq_scale_pos is not None:
                    # INT2 packed -> INT8 unpack -> INT32 matmul (4× less weight bandwidth)
                    from .bitmask_utils import TernaryInt2LinearFunc
                    _alpha = ((self.lsq_scale_pos.detach().abs().mean()
                               + self.lsq_scale_neg.detach().abs().mean()) * 0.5)
                    output = TernaryInt2LinearFunc.apply(
                        x, w, self.int2_packed_weights, self._int2_original_K, _alpha, self.bias)

                elif _int32_ready and hasattr(self, 'lsq_scale_pos') and self.lsq_scale_pos is not None:
                    # INT2 cache not ready yet — fall back to INT8 direct (w already in fp32 {±scale,0})
                    # Quantize w to int8 sign on-the-fly from the current ternary w
                    from .bitmask_utils import TernaryInt8LinearFunc
                    _alpha = ((self.lsq_scale_pos.detach().abs().mean()
                               + self.lsq_scale_neg.detach().abs().mean()) * 0.5)
                    w_int8_sign = w.detach().sign().to(torch.int8)  # {-1,0,+1} as int8
                    output = TernaryInt8LinearFunc.apply(
                        x, w, w_int8_sign, _alpha, self.bias)

                elif _int32_ready:
                    # INT32 simple fallback — works at init without lsq_scale or int2_packed
                    # Ternary w ∈ {-1,0,+1}, INT8 x quantized on-the-fly
                    from .bitmask_utils import TernaryInt8LinearFunc
                    w_int8_sign = w.detach().sign().to(torch.int8)
                    _alpha = 1.0
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

                if _int32_ready and self._int2_valid_bool and self.lsq_scale_pos is not None:
                    # INT2 packed -> INT8 unpack -> INT32 matmul (4× less weight bandwidth)
                    from .bitmask_utils import TernaryInt2LinearFunc
                    _alpha = ((self.lsq_scale_pos.detach().abs().mean()
                               + self.lsq_scale_neg.detach().abs().mean()) * 0.5)
                    output = TernaryInt2LinearFunc.apply(
                        x, w, self.int2_packed_weights, self._int2_original_K, _alpha, self.bias)

                elif _int32_ready and hasattr(self, 'lsq_scale_pos') and self.lsq_scale_pos is not None:
                    # INT2 cache not ready yet — fall back to INT8 direct (w already in fp32 {±scale,0})
                    # Quantize w to int8 sign on-the-fly from the current ternary w
                    from .bitmask_utils import TernaryInt8LinearFunc
                    _alpha = ((self.lsq_scale_pos.detach().abs().mean()
                               + self.lsq_scale_neg.detach().abs().mean()) * 0.5)
                    w_int8_sign = w.detach().sign().to(torch.int8)  # {-1,0,+1} as int8
                    output = TernaryInt8LinearFunc.apply(
                        x, w, w_int8_sign, _alpha, self.bias)

                elif _int32_ready:
                    # INT32 simple fallback — works at init without lsq_scale or int2_packed
                    # Ternary w ∈ {-1,0,+1}, INT8 x quantized on-the-fly
                    from .bitmask_utils import TernaryInt8LinearFunc
                    w_int8_sign = w.detach().sign().to(torch.int8)
                    _alpha = 1.0
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

        # Tequila deadzone reactivation — per-neuron learnable bias variant
        # Principle: dead weights (|shadow| < threshold) get zero ternary gradient.
        # Fix: learnable bias per output neuron creates a gradient path via chain rule
        # so the optimizer can pull dead weights back into the active region.
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

            if self.use_deadzone_bias and self.deadzone_bias is not None:
                # Per-neuron path: bias[i] = lambda * sum_j(w[i,j] * dz_mask[i,j])
                # Shape: [out_features]. Broadcast over (batch, seq) automatically.
                per_neuron_dz = self.tequila_lambda * (
                    self.weight * deadzone_mask.float()
                ).sum(dim=1)
                # Add learnable bias on top — this is what the optimizer trains
                # The gradient flows: loss -> output -> deadzone_bias -> weight (chain rule)
                output = output + (self.deadzone_bias + per_neuron_dz.detach())
            else:
                # Legacy scalar path (original Tequila)
                C_W = self.tequila_lambda * (self.weight * deadzone_mask.float()).sum()
                output = output + C_W


        # Low-rank FP32 correction (Phase 21.A) - RLEC
        if self.correction_A is not None and self.correction_B is not None:
            correction = F.linear(F.linear(x, self.correction_B), self.correction_A).to(
                output.dtype
            )
            output = output + correction


        return output

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
            self._packed_valid_bool = False
            # R12-F3: also invalidate INT2 cache
            if hasattr(self, 'int2_packed_valid') and self.int2_packed_valid is not None:
                self.int2_packed_valid.fill_(False)
                self._int2_valid_bool = False
            # F17-EXT: invalidate cached scales
            self._cached_scales = None

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
        # Confidence accumulator: starts at 0 (no evidence)
        self._freeze_confidence = torch.zeros_like(self.weight)
        # Freeze mask: starts all-False (nothing frozen)
        self._freeze_mask = torch.zeros_like(self.weight, dtype=torch.bool)
        # SPRT threshold
        self._freeze_threshold = math.sqrt(2.0 * math.log(3.0 / alpha))
        self._freeze_call_count = 0

        # Register backward hook to zero gradients of frozen weights
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

        # Initialize _prev_ternary from current quantized weights (detached, no grad)
        with torch.no_grad():
            q = ternary_quantize_detached(
                self.weight, self.group_size, self.threshold_coeff
            )
            # Convert to sign-only: {-scale, 0, +scale} -> {-1, 0, +1}
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
            # First call - initialize from current quantization
            self._prev_ternary = w_quantized.sign().clone()
            return w_quantized

        # Compute per-group threshold and delta in normalized space
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

        # Compute adaptive delta
        effective_margin = self.dead_zone_margin
        if self.dead_zone_lr_decay and self._base_lr > 0:
            lr_ratio = self._current_lr / self._base_lr
            effective_margin = self.dead_zone_margin * max(
                lr_ratio, 0.01
            )  # floor at 1% to prevent zero delta

        delta = effective_margin * threshold  # shape: (n_groups, 1)

        # Compute margin for each weight: how far past threshold is it?
        # margin_to_active = |w_norm| - threshold (positive = clearly active)
        # margin_to_zero = threshold - |w_norm| (positive = clearly zero)
        margin_to_active = abs_norm - threshold  # positive if weight should be active
        margin_to_zero = threshold - abs_norm  # positive if weight should be zero

        # Get proposed and previous assignments in normalized space
        q_sign = normalized.sign() * (abs_norm > threshold).float()  # {-1, 0, +1}
        prev_flat = self._prev_ternary.detach().reshape(-1)
        if pad_len > 0:
            prev_flat = F.pad(prev_flat, (0, pad_len))
        prev_grouped = prev_flat.reshape(-1, gs)

        # Detect proposed flips
        flip_mask = q_sign != prev_grouped

        # For weights going 0 -> active: require margin_to_active > delta
        going_active = flip_mask & (prev_grouped == 0) & (q_sign != 0)
        insufficient_active = going_active & (margin_to_active <= delta)

        # For weights going active -> 0: require margin_to_zero > delta
        going_zero = flip_mask & (prev_grouped != 0) & (q_sign == 0)
        insufficient_zero = going_zero & (margin_to_zero <= delta)

        # For weights changing sign (-1 -> +1 or +1 -> -1): always allow (this requires passing through 0 first)
        # Actually, a sign change means the weight crossed zero AND passed the threshold on the other side.
        # This should be rare and when it happens, it's a strong signal. Allow it.

        # Block insufficient flips: revert to previous assignment
        blocked = insufficient_active | insufficient_zero
        final_sign = torch.where(blocked, prev_grouped, q_sign)

        # Reconstruct quantized weights with scales
        final_q = (final_sign * scales).reshape(-1)
        if pad_len > 0:
            final_q = final_q[: w_src.numel()]
        final_q = final_q.reshape(orig_shape)

        # Update _prev_ternary
        final_sign_unpadded = final_sign.reshape(-1)
        if pad_len > 0:
            final_sign_unpadded = final_sign_unpadded[: w_src.numel()]
        self._prev_ternary = final_sign_unpadded.reshape(orig_shape).clone()

        # Logging counters
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
            return  # Wait for warmup

        grad = self.weight.grad
        if grad is None:
            return

        # Normalize gradient by Adam variance estimate (SNR-like)
        if self._v_hat is not None and self._v_hat.numel() == grad.numel():
            v_hat = self._v_hat.reshape(grad.shape)
            normalized_grad = grad / (v_hat.sqrt() + 1e-8)
        else:
            # Fallback: use raw gradient magnitude (less accurate but functional)
            normalized_grad = grad.sign() * (grad.abs() / (grad.abs().mean() + 1e-8))

        # Update confidence: accumulate signed evidence
        # sign(g) * |g_normalized| = g_normalized (preserves sign)
        self._freeze_confidence += normalized_grad

        # Apply freeze threshold to unfrozen weights only
        newly_frozen = (~self._freeze_mask) & (
            self._freeze_confidence.abs() > self._freeze_threshold
        )
        if newly_frozen.any():
            self._freeze_mask |= newly_frozen
            # Zero out confidence for frozen weights (no longer needed)
            self._freeze_confidence[newly_frozen] = 0.0

        # Debug logging
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


class DeferredTernaryLinear(nn.Module):
    """Deferred shadow materialization for ternary training.

    Instead of storing full FP32 shadow weights, stores:
    - Packed 2-bit ternary weights (buffer, no grad)
    - Per-group scales (buffer, no grad)
    - Low-rank delta correction (parameters, receive grad)

    Memory: ~5% of standard TernaryLinear at 1B scale.
    """

    def __init__(
        self, in_features, out_features, bias=False, group_size=128, delta_rank=32
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.delta_rank = delta_rank
        self.quantize_enabled = True
        self.ste_type = getattr(config, 'ste_type', 'mvue') if config is not None else 'mvue'  # R335: MVUE default
        self.training_step = 0
        self.total_training_steps = 1000
        self.threshold_coeff = DEFAULT_THRESHOLD_COEFF

        align = max(group_size, 16)
        K_padded = ((in_features + align - 1) // align) * align
        K_packed = K_padded // 16
        num_groups = K_padded // group_size
        self._K_padded = K_padded

        self.register_buffer(
            "packed_ternary", torch.zeros(out_features, K_packed, dtype=torch.int32)
        )
        self.register_buffer(
            "scales", torch.zeros(out_features, num_groups, dtype=torch.float32)
        )

        self.delta_A = nn.Parameter(
            torch.randn(out_features, delta_rank) * (0.02 / math.sqrt(delta_rank))
        )
        self.delta_B = nn.Parameter(torch.zeros(delta_rank, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        self._init_ternary_random()

    def _init_ternary_random(self):
        rand_w = torch.empty(self.out_features, self.in_features)
        nn.init.xavier_uniform_(rand_w, gain=1.0 / (self.in_features**0.5))
        self._pack_weights(rand_w)

    @torch.no_grad()
    def _pack_weights(self, float_weights):
        """Quantize float32 weights to ternary and store in packed format."""
        N, K = float_weights.shape
        K_padded = self._K_padded

        if K < K_padded:
            float_weights = F.pad(float_weights, (0, K_padded - K))

        grouped = float_weights.reshape(N, -1, self.group_size)
        scales = grouped.abs().amax(dim=-1).clamp(min=1e-8)
        normalized = grouped / scales.unsqueeze(-1)

        threshold = self.threshold_coeff * normalized.abs().mean(dim=-1, keepdim=True)
        ternary = normalized.sign() * (normalized.abs() > threshold).float()
        ternary_flat = ternary.reshape(N, K_padded)

        encoded = torch.zeros_like(ternary_flat, dtype=torch.int32)
        encoded[ternary_flat > 0] = 1
        encoded[ternary_flat < 0] = 2

        encoded_grouped = encoded.reshape(N, -1, 16)
        shifts = torch.arange(16, device=encoded.device) * 2
        packed = (encoded_grouped << shifts.view(1, 1, -1)).sum(dim=-1).to(torch.int32)

        self.packed_ternary.copy_(packed)
        self.scales.copy_(scales)

    def _unpack_ternary(self):
        """Unpack to float32 scaled ternary weights. No grad."""
        shifts = torch.arange(16, device=self.packed_ternary.device) * 2
        bits = (self.packed_ternary.unsqueeze(-1) >> shifts.view(1, 1, -1)) & 0x3
        values = (bits == 1).float() - (bits == 2).float()
        w_flat = values.reshape(self.out_features, -1)

        num_groups = self.scales.shape[1]
        w_grouped = w_flat.reshape(self.out_features, num_groups, self.group_size)
        w_scaled = w_grouped * self.scales.unsqueeze(-1)
        w_out = w_scaled.reshape(self.out_features, -1)

        return w_out[:, : self.in_features]

    def forward(self, x):
        if not self.quantize_enabled:
            w_ref = self._unpack_ternary()
            w = w_ref + self.delta_A @ self.delta_B
            return F.linear(x, w.to(x.dtype), self.bias)

        with torch.no_grad():
            w_ternary = self._unpack_ternary()

        w_corrected = w_ternary + self.delta_A @ self.delta_B

        w_q = ternary_quantize_ste(w_corrected, self.group_size, self.threshold_coeff)

        return F.linear(x, w_q.to(x.dtype), self.bias)

    def merge_delta(self, optimizer):
        """Merge delta into ternary, reset delta and optimizer state."""
        with torch.no_grad():
            w_ternary = self._unpack_ternary()
            w_merged = w_ternary + self.delta_A @ self.delta_B
            self._pack_weights(w_merged)

            self.delta_B.zero_()
            self.delta_A.data.normal_(0, 0.02 / math.sqrt(self.delta_rank))

            for param in [self.delta_A, self.delta_B]:
                if param in optimizer.state:
                    del optimizer.state[param]

    def delta_norm_ratio(self):
        with torch.no_grad():
            delta_full = self.delta_A @ self.delta_B
            w_ternary = self._unpack_ternary()
            return (delta_full.norm() / w_ternary.norm().clamp(min=1e-8)).item()

    def get_ternary_stats(self):
        with torch.no_grad():
            w = self._unpack_ternary()
            flat = w.reshape(-1)
            total = flat.numel()
            return {
                "positive": (flat > 0).sum().item() / total,
                "negative": (flat < 0).sum().item() / total,
                "zero": (flat == 0).sum().item() / total,
                "delta_norm_ratio": self.delta_norm_ratio(),
            }

    @classmethod
    def from_ternary_linear(cls, module, delta_rank=32):
        """Convert existing TernaryLinear to DeferredTernaryLinear."""
        deferred = cls(
            module.in_features,
            module.out_features,
            bias=(module.bias is not None),
            group_size=module.group_size,
            delta_rank=delta_rank,
        )
        deferred._pack_weights(module.weight.data)
        deferred.ste_type = module.ste_type
        deferred.threshold_coeff = module.threshold_coeff
        deferred.training_step = module.training_step
        deferred.total_training_steps = module.total_training_steps
        if module.bias is not None:
            deferred.bias.data.copy_(module.bias.data)
        deferred = deferred.to(module.weight.device)
        return deferred


def merge_all_deferred_deltas(model, optimizer, min_delta_norm=0.0):
    """Merge delta on all DeferredTernaryLinear modules."""
    merged = 0
    for module in model.modules():
        if hasattr(module, "merge_delta") and hasattr(module, "delta_rank"):
            if min_delta_norm > 0 and module.delta_norm_ratio() < min_delta_norm:
                continue
            module.merge_delta(optimizer)
            merged += 1
    return merged


def convert_model_to_deferred(model, delta_rank=32):
    """Replace all TernaryLinear with DeferredTernaryLinear."""
    conversions = []
    for name, module in model.named_modules():
        if (
            hasattr(module, "quantize_enabled")
            and hasattr(module, "weight")
            and not hasattr(module, "delta_rank")
        ):
            if hasattr(module, "group_size"):
                conversions.append((name, module))

    converted = 0
    for name, module in conversions:
        deferred = DeferredTernaryLinear.from_ternary_linear(
            module, delta_rank=delta_rank
        )
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], deferred)
        converted += 1

    total_packed = sum(
        m.packed_ternary.numel() * 4
        for m in model.modules()
        if hasattr(m, "packed_ternary")
    )
    total_scales = sum(
        m.scales.numel() * 4 for m in model.modules() if hasattr(m, "packed_ternary")
    )
    total_delta = sum(
        m.delta_A.numel() + m.delta_B.numel()
        for m in model.modules()
        if hasattr(m, "delta_rank")
    )
    print(f"[DeferredTernary] Converted {converted} layers, rank={delta_rank}")
    print(f"  Packed ternary: {total_packed / 1024**2:.1f} MB")
    print(f"  Scales: {total_scales / 1024**2:.1f} MB")
    print(f"  Delta params: {total_delta:,} ({total_delta * 4 / 1024**2:.1f} MB)")
    print(f"  Optimizer state (est): {total_delta * 8 / 1024**2:.1f} MB")
    return converted


def get_deferred_stats(model):
    """Aggregate diagnostics across all DeferredTernaryLinear layers."""
    norms = []
    for module in model.modules():
        if hasattr(module, "delta_norm_ratio") and hasattr(module, "delta_rank"):
            norms.append(module.delta_norm_ratio())
    if not norms:
        return {"enabled": False}
    return {
        "enabled": True,
        "num_layers": len(norms),
        "avg_delta_norm_ratio": sum(norms) / len(norms),
        "max_delta_norm_ratio": max(norms),
        "min_delta_norm_ratio": min(norms),
    }


class TernaryLinearInteger(nn.Module):
    """Ternary linear layer with integer gradient accumulation (Path C.1).

    Instead of FP32 latent weights + AdamW optimizer, this uses:
    - Weight stored as {-1, 0, +1} (the actual value, not quantized from FP32)
    - Learnable per-layer scale (alpha) - optimized by AdamW
    - Gradient accumulator for threshold-based weight flipping

    Memory per weight:
    - Standard TernaryLinear: ~12 bytes (4 FP32 weight + 8 AdamW state)
    - This class: ~6 bytes (4 FP32 weight-as-param + 4 FP32 accum, but no optimizer state)
    - Future: could be 3 bytes with INT8 weight + INT16 accum

    The weight nn.Parameter stores {-1,0,+1} as float for autograd compatibility.
    It must be EXCLUDED from the optimizer - updates happen via accumulate_and_update().
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 128,
        alpha_group_size: int = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.alpha_group_size = alpha_group_size

        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if alpha_group_size is None:
            alpha_init = 1.0 / math.sqrt(in_features)
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
        else:
            total_weights = out_features * in_features
            pad_len = (
                alpha_group_size - total_weights % alpha_group_size
            ) % alpha_group_size
            num_groups = (total_weights + pad_len) // alpha_group_size
            alpha_init = 1.0 / math.sqrt(in_features)
            self.alpha = nn.Parameter(torch.full((num_groups,), alpha_init))

        self.register_buffer("g_accum", torch.zeros(out_features, in_features))

        self.flip_count = 0
        self.total_updates = 0

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        self._init_ternary()

    def _init_ternary(self):
        """Initialize weights as ternary {-1, 0, +1} with ~29/42/29 distribution."""
        with torch.no_grad():
            rand = torch.randn(self.out_features, self.in_features)
            threshold = 0.7 * rand.abs().mean()
            ternary = rand.sign() * (rand.abs() > threshold).float()
            self.weight.data.copy_(ternary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with torch.compile support. Quantizes weights with STE."""
        if self.alpha_group_size is None:
            return F.linear(x, self.weight * self.alpha, self.bias)

        orig_shape = self.weight.shape
        flat = self.weight.reshape(-1)
        pad_len = (
            self.alpha_group_size - flat.numel() % self.alpha_group_size
        ) % self.alpha_group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, self.alpha_group_size)
        scaled = grouped * self.alpha.unsqueeze(-1)
        w = scaled.reshape(-1)[: self.weight.numel()].reshape(orig_shape)
        return F.linear(x, w, self.bias)

    def accumulate_and_update(self, flip_threshold: float = 1.0):
        """Accumulate gradient and flip weights where threshold exceeded.

        Call this after loss.backward() instead of optimizer.step() for ternary weights.

        Args:
            flip_threshold: How much gradient must accumulate before flipping.
                           Lower = more flips = faster learning but noisier.
                           Higher = fewer flips = slower but more stable.
        """
        with torch.no_grad():
            if self.weight.grad is None:
                return

            self.g_accum.add_(self.weight.grad)
            self.weight.grad.zero_()
            self.total_updates += 1

            flip_mask = self.g_accum.abs() > flip_threshold
            if flip_mask.any():
                n_flips = flip_mask.sum().item()
                self.flip_count += n_flips

                flip_dir = -self.g_accum[flip_mask].sign()

                new_vals = self.weight.data[flip_mask] + flip_dir
                new_vals.clamp_(-1, 1)
                new_vals = new_vals.sign() * (new_vals.abs() >= 0.5).float()

                self.weight.data[flip_mask] = new_vals
                self.g_accum[flip_mask] = 0.0

    def get_ternary_stats(self) -> dict:
        with torch.no_grad():
            flat = self.weight.data.reshape(-1)
            total = flat.numel()
            return {
                "positive": (flat > 0).sum().item() / total,
                "negative": (flat < 0).sum().item() / total,
                "zero": (flat == 0).sum().item() / total,
                "flip_count": self.flip_count,
                "total_updates": self.total_updates,
                "flips_per_update": self.flip_count / max(1, self.total_updates),
                "accum_abs_mean": self.g_accum.abs().mean().item(),
                "accum_abs_max": self.g_accum.abs().max().item(),
                "alpha_mean": self.alpha.mean().item(),
                "alpha_std": self.alpha.std().item() if self.alpha.numel() > 1 else 0.0,
                "alpha_min": self.alpha.min().item(),
                "alpha_max": self.alpha.max().item(),
            }


# Utils — extracted to ternary_pkg/utils.py
from .ternary_pkg.utils import *


# Monitors — extracted to ternary_pkg/monitors.py
from .ternary_pkg.monitors import TernaryFlipRateTracker


from .ternary_pkg.losses import compute_sca_sparsity_loss, compute_sca_sparsity_loss_simple
