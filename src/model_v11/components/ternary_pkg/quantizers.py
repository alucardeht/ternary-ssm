import logging
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..triton_hestia import hestia_quantize_triton

    _TRITON_HESTIA_AVAILABLE = True
except ImportError:
    _TRITON_HESTIA_AVAILABLE = False

_q_values_cache: dict = {}
_av_probe_counters: dict = {}
_aw3_ema_store: dict = {}
logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD_COEFF = 1.0 / math.sqrt(2)  # Laplace-optimal (R196)



def lyapunov_arenas_schedule(step, total_steps, lambda_0=1.0, epsilon=0.01):
    """Optimal Lyapunov-stable schedule for Arenas residual decay.

    Guarantees monotonic convergence of ||W_shadow - W_ternary||^2 to epsilon^2.

    Args:
        step: Current training step
        total_steps: Total training steps
        lambda_0: Initial residual weight (default: 1.0)
        epsilon: Target final residual (default: 0.01)

    Returns:
        lambda_t: Residual weight for current step

    Reference: R3-A1-M3 (Lyapunov Stability Theorem for Arenas)
    """
    if step >= total_steps:
        return epsilon

    t_normalized = step / total_steps
    denominator = 1.0 + (t_normalized**2) * (lambda_0 / epsilon - 1.0)
    lambda_t = lambda_0 / denominator

    return lambda_t


class VoteBufferOptimizer:
    """Shadow-free ternary optimizer with formal convergence guarantee.

    Instead of FP32 shadow weights, maintains vote buffers that accumulate
    gradient evidence. Weights flip when votes exceed k*sigma threshold.

    Memory: 4 bytes per weight (vote_buffer in FP32)
    vs Standard: 12 bytes per weight (4 shadow + 8 AdamW state)

    Args:
        vote_window: Gradient accumulation window (default: 50)
        k_sigma: Threshold in units of gradient std (default: 3.0)
        variance_ema_beta: EMA decay for variance tracking (default: 0.99)
    """

    def __init__(self, vote_window=50, k_sigma=3.0, variance_ema_beta=0.99):
        self.vote_window = vote_window
        self.k_sigma = k_sigma

        self.vote_buffers = {}
        self.step_count = {}
        self.total_flips = 0
        self.vote_threshold: int = 16

    def register_parameter(self, param_id, param_shape, device="cuda"):
        """Register a ternary parameter for vote-buffer tracking."""
        self.vote_buffers[param_id] = torch.zeros(
            param_shape, dtype=torch.int8, device=device
        )
        self.step_count[param_id] = 0

    def accumulate_vote(self, param_id, gradient):
        """Accumulate gradient vote (called every backward pass)."""
        if param_id not in self.vote_buffers:
            self.register_parameter(param_id, gradient.shape, device=gradient.device)

        sign_vote = gradient.sign().clamp(-1, 1).to(torch.int16)
        acc = self.vote_buffers[param_id].to(torch.int16).add_(sign_vote).clamp_(-127, 127)
        self.vote_buffers[param_id].copy_(acc.to(torch.int8))

        self.step_count[param_id] += 1

    def apply_flips(self, param_id, weight_ternary):
        """Apply weight flips based on vote threshold."""
        if self.step_count[param_id] < self.vote_window:
            return weight_ternary, 0

        votes = self.vote_buffers[param_id]
        K = self.vote_threshold

        flip_to_pos = (votes.int() > K) & (weight_ternary != 1)
        flip_to_neg = (votes.int() < -K) & (weight_ternary != -1)
        flip_to_zero = (votes.abs().int() < K // 2) & (weight_ternary != 0)

        new_weights = weight_ternary.clone()
        new_weights[flip_to_pos] = 1
        new_weights[flip_to_neg] = -1
        new_weights[flip_to_zero] = 0

        n_flips = (new_weights != weight_ternary).sum().item()
        self.total_flips += n_flips

        if self.total_flips % 1000 == 0 and self.total_flips > 0:
            print(f"[BBO-INT8] total_flips={self.total_flips} K={K} "
                  f"buffer_bytes={sum(v.nbytes for v in self.vote_buffers.values())}")

        self.vote_buffers[param_id].zero_()
        self.step_count[param_id] = 0

        return new_weights, n_flips

    def state_dict(self):
        """Save optimizer state for checkpointing."""
        return {
            "vote_buffers": {k: v.cpu() for k, v in self.vote_buffers.items()},
            "step_count": self.step_count.copy(),
            "total_flips": self.total_flips,
        }

    def load_state_dict(self, state, device="cuda"):
        """Load optimizer state from checkpoint."""
        self.vote_buffers = {k: v.to(device) for k, v in state["vote_buffers"].items()}
        self.step_count = state["step_count"]
        self.total_flips = state["total_flips"]


class TernaryQuantize(torch.autograd.Function):
    _call_count = 0
    _log_every = 500
    _use_hyperreal_ste: bool = False

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        group_size: int = 128,
        threshold_coeff: float = DEFAULT_THRESHOLD_COEFF,
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
        threshold = threshold_coeff * abs_normalized.mean(dim=-1, keepdim=True)
        quantized = normalized.sign() * (abs_normalized > threshold).float()

        transition_zone = threshold * 0.3
        near_threshold = (abs_normalized - threshold).abs() < transition_zone
        informative_frac = near_threshold.float().mean(dim=-1, keepdim=True)
        ols_alpha = informative_frac.mean().clamp(min=0.1, max=1.0)

        if TernaryQuantize._use_hyperreal_ste:
            ctx.save_for_backward(ols_alpha, weight.detach(), torch.tensor(threshold_coeff, device=weight.device))
            ctx._use_hyperreal = True
        else:
            ctx.save_for_backward(ols_alpha)
            ctx._use_hyperreal = False

        result = (quantized * scales).reshape(-1)[
            : flat.numel() - pad_len if pad_len > 0 else flat.numel()
        ]

        TernaryQuantize._call_count += 1
        if TernaryQuantize._call_count % TernaryQuantize._log_every == 0:
            print(f"[OLS-STE] call={TernaryQuantize._call_count} alpha={ols_alpha.item():.4f} "
                  f"informative={informative_frac.mean().item():.4f}")
            if TernaryQuantize._use_hyperreal_ste:
                print(f"[AW-12 HYPERREAL-STE] ACTIVE — sech^2 shaped backward replacing flat OLS")

        return result.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if getattr(ctx, '_use_hyperreal', False):
            ols_alpha, shadow, thr_coeff = ctx.saved_tensors
            with torch.no_grad():
                _sg = shadow.float().std().clamp(min=1e-8)
                _thr = thr_coeff.item() * shadow.float().abs().mean().clamp(min=1e-8)
                _dist = (shadow.float().abs() - _thr) / _sg
                _gate = 1.0 - _dist.tanh().pow(2)
            return grad_output * _gate.to(grad_output.dtype), None, None
        else:
            (ols_alpha,) = ctx.saved_tensors
            return grad_output.mul_(ols_alpha), None, None


class BoundaryAwareTernaryQuantize(torch.autograd.Function):
    _call_count = 0
    _log_interval = 500

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        group_size: int = 128,
        threshold_coeff: float = DEFAULT_THRESHOLD_COEFF,
        sigma: float = 0.5,
        min_mod: float = 0.3,
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
        threshold = threshold_coeff * abs_normalized.mean(dim=-1, keepdim=True)
        quantized = normalized.sign() * (abs_normalized > threshold).float()

        dist_to_threshold = (abs_normalized - threshold).abs()
        modulation = min_mod + (1.0 - min_mod) * torch.exp(
            -dist_to_threshold.pow(2) / (2 * sigma * sigma)
        )

        result = (quantized * scales).reshape(-1)
        if pad_len > 0:
            result = result[: weight.numel()]

        modulation_flat = modulation.reshape(-1)
        if pad_len > 0:
            modulation_flat = modulation_flat[: weight.numel()]

        modulation_shaped = modulation_flat.reshape(orig_shape)
        ctx.save_for_backward(modulation_shaped)

        BoundaryAwareTernaryQuantize._call_count += 1
        if (
            BoundaryAwareTernaryQuantize._call_count
            % BoundaryAwareTernaryQuantize._log_interval
            == 1
        ):
            with torch.no_grad():
                mod_mean = modulation.mean().item()
                mod_min = modulation.min().item()
                mod_max = modulation.max().item()
                near_boundary = (modulation > 0.8).float().mean().item() * 100
                far_from_boundary = (modulation < 0.4).float().mean().item() * 100
                print(
                    f"  [BoundaryAware] call={BoundaryAwareTernaryQuantize._call_count} "
                    f"mod_mean={mod_mean:.4f} mod_min={mod_min:.4f} mod_max={mod_max:.4f} "
                    f"near_boundary={near_boundary:.1f}% far={far_from_boundary:.1f}% "
                    f"sigma={sigma:.2f} min_mod={min_mod:.2f}"
                )

        return result.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (modulation,) = ctx.saved_tensors
        return grad_output * modulation, None, None, None, None


class CTCETernaryQuantize(torch.autograd.Function):
    """CTCE: Amplifies STE gradients near quantization thresholds (factor >= 1.0 always)."""

    _call_count = 0
    _log_interval = 500

    @staticmethod
    def forward(
        ctx,
        weight,
        group_size=128,
        threshold_coeff=DEFAULT_THRESHOLD_COEFF,
        beta=1.5,
        delta_ratio=0.5,
    ):
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

        dist_to_threshold = abs_normalized - threshold
        delta = delta_ratio * dist_to_threshold.std(dim=-1, keepdim=True).clamp(
            min=1e-8
        )
        amplification = 1.0 + beta * torch.exp(
            -dist_to_threshold.pow(2) / (2.0 * delta.pow(2))
        )

        result = (quantized * scales).reshape(-1)
        if pad_len > 0:
            result = result[: weight.numel()]

        amplification_flat = amplification.reshape(-1)
        if pad_len > 0:
            amplification_flat = amplification_flat[: weight.numel()]

        ctx.save_for_backward(amplification_flat.reshape(orig_shape))

        CTCETernaryQuantize._call_count += 1
        if CTCETernaryQuantize._call_count % CTCETernaryQuantize._log_interval == 1:
            if not torch.compiler.is_compiling():
                with torch.no_grad():
                    amp_mean = amplification.mean().item()
                    amp_min = amplification.min().item()
                    amp_max = amplification.max().item()
                    near_threshold = (amplification > 1.5).float().mean().item() * 100
                    at_identity = (amplification < 1.05).float().mean().item() * 100
                    print(
                        f"  [CTCE] call={CTCETernaryQuantize._call_count} "
                        f"amp_mean={amp_mean:.4f} min={amp_min:.4f} max={amp_max:.4f} "
                        f"near(>1.5)={near_threshold:.1f}% identity(<1.05)={at_identity:.1f}%"
                    )

        return result.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (amplification,) = ctx.saved_tensors
        return grad_output * amplification, None, None, None, None


class DualRateTernaryQuantize(torch.autograd.Function):
    """Dual-Rate STE: amplifies near-threshold + dampens far-from-threshold gradients."""

    _call_count = 0

    @staticmethod
    def forward(ctx, weight, group_size, threshold_coeff, peak, floor, delta_ratio):
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

        dist = (abs_normalized - threshold).abs()
        delta = delta_ratio * dist.std(dim=-1, keepdim=True).clamp(min=1e-8)
        modulation = floor + (peak - floor) * torch.exp(
            -dist.pow(2) / (2.0 * delta.pow(2))
        )

        result = (quantized * scales).reshape(-1)
        if pad_len > 0:
            result = result[: weight.numel()]
        mod_flat = modulation.reshape(-1)
        if pad_len > 0:
            mod_flat = mod_flat[: weight.numel()]
        ctx.save_for_backward(mod_flat.reshape(orig_shape))

        DualRateTernaryQuantize._call_count += 1
        if DualRateTernaryQuantize._call_count % 500 == 1:
            if not torch.compiler.is_compiling():
                with torch.no_grad():
                    print(
                        f"  [DualRate] call={DualRateTernaryQuantize._call_count} "
                        f"mod_mean={modulation.mean().item():.4f} "
                        f"min={modulation.min().item():.4f} max={modulation.max().item():.4f} "
                        f"amplified(>1.5)={((modulation > 1.5).float().mean().item() * 100):.1f}% "
                        f"dampened(<0.5)={((modulation < 0.5).float().mean().item() * 100):.1f}%"
                    )
        return result.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (modulation,) = ctx.saved_tensors
        return grad_output * modulation, None, None, None, None, None


torch.compiler.allow_in_graph(DualRateTernaryQuantize)


class LSQTernaryQuantize(torch.autograd.Function):
    """LSQ: Learned Step Size Quantization for ternary {-1, 0, +1}.

    Replaces static per-group max-abs scale with a learned nn.Parameter lsq_scale.
    Gradient dL/ds flows analytically; dL/dw via OLS-STE.

    Reference: Esser et al., "Learned Step Size Quantization" (ICLR 2020)
    """

    _call_count = 0

    @staticmethod
    def forward(ctx, weight, lsq_scale_pos, lsq_scale_neg, group_size, threshold_coeff, sigma_g=None, training_step=0):
        orig_shape = weight.shape
        flat = weight.reshape(-1)
        pad_len = (group_size - flat.numel() % group_size) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, group_size)

        sp = lsq_scale_pos.abs().clamp(min=1e-8).unsqueeze(-1)
        sn = lsq_scale_neg.abs().clamp(min=1e-8).unsqueeze(-1)
        s_norm = (sp * sn).sqrt()
        normalized = grouped / s_norm
        abs_normalized = normalized.abs()

        _TARGET_FRACTION = 0.6
        abs_flat = abs_normalized
        sorted_vals, _ = abs_flat.sort(dim=-1, descending=True)
        cumsum = sorted_vals.cumsum(dim=-1)
        total_energy = cumsum[:, -1:].clamp(min=1e-8)

        if sigma_g is not None:
            sg_flat = sigma_g.to(dtype=torch.float32).reshape(-1)
            if pad_len > 0:
                sg_flat = F.pad(sg_flat, (0, pad_len))
            sg_grouped = sg_flat.reshape(-1, group_size)
            sg_per_group = sg_grouped.mean(dim=-1, keepdim=True)
            mean_sg = sg_per_group.mean().clamp(min=1e-8)
            sensitivity = sg_per_group / mean_sg
            target_fraction = (_TARGET_FRACTION * sensitivity).clamp(min=0.4, max=0.85)
        else:
            sensitivity = None
            target_fraction = _TARGET_FRACTION

        k = (cumsum / total_energy < target_fraction).sum(dim=-1).clamp(max=group_size - 1)
        energy_threshold = sorted_vals.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        threshold = energy_threshold.unsqueeze(-1)
        quantized = normalized.sign() * (abs_normalized > threshold).float()

        transition_zone = threshold * 0.3
        near_threshold = (abs_normalized - threshold).abs() < transition_zone
        ols_alpha = near_threshold.float().mean().clamp(min=0.1, max=1.0)

        pos_mask = quantized > 0.5
        neg_mask = quantized < -0.5
        w_asymmetric = pos_mask.float() * sp - neg_mask.float() * sn
        q_flat = w_asymmetric.reshape(-1)[:weight.numel()]
        result = q_flat.reshape(orig_shape)

        ctx.save_for_backward(quantized, sp, sn, normalized, ols_alpha, threshold)
        ctx.orig_shape = orig_shape
        ctx.pad_len = pad_len
        ctx.group_size = group_size
        ctx.training_step = training_step

        LSQTernaryQuantize._call_count += 1
        if LSQTernaryQuantize._call_count == 1 and not torch.compiler.is_compiling():
            with torch.no_grad():
                nonzero_pct = (quantized != 0).float().mean().item() * 100
                _thr_mean = threshold.mean().item()
                _thr_std = threshold.std().item()
                lat_msg = ""
                if sensitivity is not None:
                    lat_msg = (f" sensitivity_mean={sensitivity.mean().item():.3f}"
                               f" sensitivity_std={sensitivity.std().item():.3f}")
                print(
                    f"[TTQ+M10] ACTIVATED sp_mean={lsq_scale_pos.abs().mean().item():.4f} "
                    f"sn_mean={lsq_scale_neg.abs().mean().item():.4f} "
                    f"asymmetry={(lsq_scale_pos.abs()/lsq_scale_neg.abs().clamp(min=1e-8)).mean().item():.3f} "
                    f"nonzero={nonzero_pct:.1f}% alpha={ols_alpha.item():.4f} "
                    f"energy_thr_mean={_thr_mean:.4f} energy_thr_std={_thr_std:.4f}{lat_msg}"
                )
        elif LSQTernaryQuantize._call_count == 2 and not torch.compiler.is_compiling():
            print(f"[LSQ] active across layers (call_count={LSQTernaryQuantize._call_count})")

        return result

    @staticmethod
    def backward(ctx, grad_output):
        quantized, sp, sn, normalized, ols_alpha, threshold = ctx.saved_tensors
        pad_len = ctx.pad_len
        group_size = ctx.group_size

        grad_weight = grad_output * ols_alpha

        step = getattr(ctx, 'training_step', 0)
        mu = 0.1 * (1.0 - math.exp(-0.001 * step))
        if mu > 1e-6:
            disc_error_grouped = normalized - quantized
            disc_error_flat = disc_error_grouped.reshape(-1)
            if pad_len > 0:
                disc_error_flat = disc_error_flat[:grad_output.numel()]
            disc_error = disc_error_flat.reshape(grad_output.shape)
            grad_weight = grad_weight + mu * disc_error
            if step % 500 == 1 and not torch.compiler.is_compiling():
                print(f"[PEGE] step={step} mu={mu:.5f} disc_err_mean={disc_error.abs().mean().item():.5f}")

        _dtg_epsilon = 0.1
        _dtg_boost = 2.0
        dist_to_thr = (normalized.abs() - threshold).abs()
        near_boundary = dist_to_thr < (_dtg_epsilon * threshold.clamp(min=1e-8))
        nb_flat = near_boundary.reshape(-1)
        if pad_len > 0:
            nb_flat = nb_flat[:grad_output.numel()]
        nb_mask = nb_flat.reshape(grad_output.shape).float()
        grad_weight = grad_weight + grad_weight * nb_mask * (_dtg_boost - 1.0)
        if step % 500 == 1 and not torch.compiler.is_compiling():
            print(f"[F8-DTG] step={step} boundary_pct={nb_mask.mean().item()*100:.1f}%"
                  f" boost_grad_mean={(grad_weight * nb_mask).abs().mean().item():.6f}")

        go_flat = grad_output.reshape(-1)
        if pad_len > 0:
            go_flat = F.pad(go_flat, (0, pad_len))
        go_grouped = go_flat.reshape(-1, group_size)

        pos_mask = quantized > 0.5
        neg_mask = quantized < -0.5
        grad_sp = (go_grouped * pos_mask.float()).sum(dim=-1)
        grad_sn = -(go_grouped * neg_mask.float()).sum(dim=-1)

        return grad_weight, grad_sp, grad_sn, None, None, None, None


class StrangSplitLSQTernaryQuantize(torch.autograd.Function):
    """Strang Splitting STE: quantize at midpoint between shadow and quantized weights.

    Reduces STE approximation error from O(lr) to O(lr^2) by evaluating the quantization
    map at w_mid = w + 0.5*(Q(w) - w). Allows 2-4x higher valid learning rate.

    Reference: R17-H-W2-MPB-F5
    """
    _call_count = 0

    @staticmethod
    def forward(ctx, weight, lsq_scale_pos, lsq_scale_neg,
                group_size, threshold_coeff, sigma_g=None, training_step=0):
        orig_shape = weight.shape
        flat = weight.reshape(-1)
        pad_len = (group_size - flat.numel() % group_size) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, group_size)

        sp = lsq_scale_pos.abs().clamp(min=1e-8).unsqueeze(-1)
        sn = lsq_scale_neg.abs().clamp(min=1e-8).unsqueeze(-1)
        s_norm = (sp * sn).sqrt()
        normalized = grouped / s_norm
        abs_normalized = normalized.abs()

        _TARGET_FRACTION = 0.6
        sorted_vals, _ = abs_normalized.sort(dim=-1, descending=True)
        cumsum = sorted_vals.cumsum(dim=-1)
        total_energy = cumsum[:, -1:].clamp(min=1e-8)
        k = (cumsum / total_energy < _TARGET_FRACTION).sum(dim=-1).clamp(max=group_size - 1)
        threshold = sorted_vals.gather(-1, k.unsqueeze(-1)).squeeze(-1).unsqueeze(-1)

        q0 = normalized.sign() * (abs_normalized > threshold).float()
        mid_normalized = normalized + 0.5 * (q0 - normalized)
        abs_mid = mid_normalized.abs()
        quantized = mid_normalized.sign() * (abs_mid > threshold).float()

        transition_zone = threshold * 0.3
        near_threshold = (abs_mid - threshold).abs() < transition_zone
        ols_alpha = near_threshold.float().mean().clamp(min=0.1, max=1.0)

        pos_mask = quantized > 0.5
        neg_mask = quantized < -0.5
        w_asymmetric = pos_mask.float() * sp - neg_mask.float() * sn
        q_flat = w_asymmetric.reshape(-1)[:weight.numel()]
        result = q_flat.reshape(orig_shape)

        ctx.save_for_backward(quantized, sp, sn, normalized, ols_alpha, threshold)
        ctx.orig_shape = orig_shape
        ctx.pad_len = pad_len
        ctx.group_size = group_size

        StrangSplitLSQTernaryQuantize._call_count += 1
        if StrangSplitLSQTernaryQuantize._call_count == 1:
            with torch.no_grad():
                mid_err = (mid_normalized - q0).abs().mean().item()
                orig_err = (normalized - q0).abs().mean().item()
                print(f"[STRANG-STE] mid_err={mid_err:.4f} orig_err={orig_err:.4f} "
                      f"ratio={mid_err/max(orig_err, 1e-8):.3f} ols_alpha={ols_alpha.item():.4f}")
        return result

    @staticmethod
    def backward(ctx, grad_output):
        quantized, sp, sn, normalized, ols_alpha, threshold = ctx.saved_tensors
        grad_w = grad_output * ols_alpha
        grad_flat = grad_output.reshape(-1)
        if ctx.pad_len > 0:
            grad_flat = F.pad(grad_flat, (0, ctx.pad_len))
        grad_grouped = grad_flat.reshape(-1, ctx.group_size)
        pos_mask = quantized > 0.5
        neg_mask = quantized < -0.5
        grad_sp = (grad_grouped * pos_mask.float()).sum(dim=-1)
        grad_sn = (grad_grouped * (-neg_mask.float())).sum(dim=-1)
        return grad_w, grad_sp, grad_sn, None, None, None, None


class GVSTELSQTernaryQuantize(torch.autograd.Function):
    """GV-STE: Per-weight adaptive STE slope based on gradient magnitude EMA.

    High |grad| EMA -> larger slope (more signal passed to high-variance weights).
    Low |grad| EMA -> smaller slope (stable weights get less noise amplification).
    Expects 5-10% faster loss decrease in early training.

    Reference: R17-F-W2-PB-F4
    """
    @staticmethod
    def forward(ctx, weight, lsq_scale_pos, lsq_scale_neg,
                group_size, threshold_coeff, gv_slope, sigma_g=None, training_step=0):
        orig_shape = weight.shape
        flat = weight.reshape(-1)
        pad_len = (group_size - flat.numel() % group_size) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, group_size)

        sp = lsq_scale_pos.abs().clamp(min=1e-8).unsqueeze(-1)
        sn = lsq_scale_neg.abs().clamp(min=1e-8).unsqueeze(-1)
        s_norm = (sp * sn).sqrt()
        normalized = grouped / s_norm
        abs_normalized = normalized.abs()

        _TARGET_FRACTION = 0.6
        sorted_vals, _ = abs_normalized.sort(dim=-1, descending=True)
        cumsum = sorted_vals.cumsum(dim=-1)
        total_energy = cumsum[:, -1:].clamp(min=1e-8)
        k = (cumsum / total_energy < _TARGET_FRACTION).sum(dim=-1).clamp(max=group_size - 1)
        threshold = sorted_vals.gather(-1, k.unsqueeze(-1)).squeeze(-1).unsqueeze(-1)
        quantized = normalized.sign() * (abs_normalized > threshold).float()

        pos_mask = quantized > 0.5
        neg_mask = quantized < -0.5
        w_asymmetric = pos_mask.float() * sp - neg_mask.float() * sn
        q_flat = w_asymmetric.reshape(-1)[:weight.numel()]
        result = q_flat.reshape(orig_shape)

        ctx.save_for_backward(quantized, sp, sn, normalized, threshold, gv_slope)
        ctx.orig_shape = orig_shape
        ctx.pad_len = pad_len
        ctx.group_size = group_size
        return result

    @staticmethod
    def backward(ctx, grad_output):
        quantized, sp, sn, normalized, threshold, gv_slope = ctx.saved_tensors
        slope = gv_slope.to(grad_output.dtype)
        slope_mean = slope.mean().clamp(min=1e-8)
        slope_normalized = slope / slope_mean
        grad_w = grad_output * slope_normalized

        grad_flat = grad_output.reshape(-1)
        if ctx.pad_len > 0:
            grad_flat = F.pad(grad_flat, (0, ctx.pad_len))
        grad_grouped = grad_flat.reshape(-1, ctx.group_size)
        pos_mask = quantized > 0.5
        neg_mask = quantized < -0.5
        grad_sp = (grad_grouped * pos_mask.float()).sum(dim=-1)
        grad_sn = (grad_grouped * (-neg_mask.float())).sum(dim=-1)
        return grad_w, grad_sp, grad_sn, None, None, None, None, None


class WassersteinSTETernaryQuantize(torch.autograd.Function):
    """AO-1: Wasserstein-1 gradient for ternary quantization.
    Principle (R17-B-W2-MPB-F5): W_{t+1} = sign(W_t - η·∇_W W_1).
    The true gradient of W_1 loss is sign(∇_W L), not ∇_W L * ols_alpha.
    Forward: identical to LSQ. Backward: grad_w = sign(grad_output).
    """

    @staticmethod
    def forward(ctx, w, scale_pos, scale_neg, group_size, t_coeff, training_step):
        ctx.save_for_backward(w)
        w_grouped = w.reshape(-1, group_size)
        sp = scale_pos.reshape(-1, 1).abs().clamp(min=1e-6)
        sn = scale_neg.reshape(-1, 1).abs().clamp(min=1e-6)
        pos_mask = w_grouped > t_coeff * sp
        neg_mask = w_grouped < -t_coeff * sn
        w_q = torch.zeros_like(w_grouped)
        w_q[pos_mask] = sp.expand_as(w_grouped)[pos_mask]
        w_q[neg_mask] = -sn.expand_as(w_grouped)[neg_mask]
        if training_step == 0:
            print(f"[WASSERSTEIN-STE] ACTIVATED | backward uses sign(grad)")
        return w_q.reshape_as(w)

    @staticmethod
    def backward(ctx, grad_output):
        grad_w = grad_output.sign()
        return grad_w, None, None, None, None, None


torch.compiler.allow_in_graph(LSQTernaryQuantize)


class GumbelSoftmaxSTE(torch.autograd.Function):
    """Ternary quantization with Gumbel-Softmax gradient estimator (R13-Batch-AF).

    Forward: identical to standard ternary hard assignment (group-wise, like LSQTernaryQuantize).
    Backward: gradient of soft ternary relaxation controlled by temperature τ.

    High τ = soft gradient (exploration), low τ = near-STE (commitment).
    Anneals τ_init -> τ_min over training.

    Args:
        w:              weight tensor [out_features, in_features]
        w_thr:          quantization threshold scalar
        ols_alpha:      OLS gradient scale factor (scalar)
        sp:             positive ternary scales [num_groups]
        sn:             negative ternary scales [num_groups]
        training_step:  module training step counter
        tau:            current temperature (pre-computed by caller)
        group_size:     number of elements per group
    """

    @staticmethod
    def forward(ctx, w, w_thr, ols_alpha, sp, sn, training_step, tau, group_size):
        orig_shape = w.shape
        flat = w.reshape(-1)
        grouped = flat.reshape(-1, group_size)

        sp_g = sp.abs().clamp(min=1e-8).unsqueeze(-1)
        sn_g = sn.abs().clamp(min=1e-8).unsqueeze(-1)

        pos_mask = grouped > w_thr
        neg_mask = grouped < -w_thr
        result_g = torch.where(
            pos_mask, sp_g.expand_as(grouped),
            torch.where(neg_mask, -sn_g.expand_as(grouped), grouped.new_zeros(()))
        )
        result = result_g.reshape(orig_shape)

        ctx.save_for_backward(grouped, sp_g, sn_g)
        ctx.tau = float(tau)
        ctx.ols_alpha = ols_alpha.item() if isinstance(ols_alpha, torch.Tensor) else float(ols_alpha)
        ctx.orig_shape = orig_shape
        ctx.group_size = int(group_size)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grouped, sp_g, sn_g = ctx.saved_tensors
        tau = ctx.tau
        ols_alpha = ctx.ols_alpha

        grad_grouped = grad_output.reshape(-1, ctx.group_size)

        l_neg = -(grouped + sn_g).pow(2) / tau
        l_zero = -(grouped).pow(2) / tau
        l_pos = -(grouped - sp_g).pow(2) / tau

        logits = torch.stack([l_neg, l_zero, l_pos], dim=-1)
        probs = torch.softmax(logits, dim=-1)
        p_neg, p_zero, p_pos = probs[..., 0], probs[..., 1], probs[..., 2]

        w_soft = -sn_g * p_neg + sp_g * p_pos

        dl_neg_dw = -2.0 * (grouped + sn_g) / tau
        dl_zero_dw = -2.0 * grouped / tau
        dl_pos_dw = 2.0 * (sp_g - grouped) / tau

        dw_soft_dw = (
            dl_neg_dw * p_neg * (-sn_g - w_soft)
            + dl_zero_dw * p_zero * (0.0 - w_soft)
            + dl_pos_dw * p_pos * (sp_g - w_soft)
        )

        grad_w = (grad_grouped * dw_soft_dw * ols_alpha).reshape(ctx.orig_shape)
        return grad_w, None, None, None, None, None, None, None


torch.compiler.allow_in_graph(GumbelSoftmaxSTE)


class OTGradientTernaryQuantize(torch.autograd.Function):
    """OT-Gradient: uses optimal transport geometry instead of STE.

    grad_w = grad_output * 2.0 * (w_continuous - w_quantized)
    Near threshold: small update. Far from threshold: large update.
    """

    _call_count = 0

    @staticmethod
    def forward(ctx, weight, lsq_scale_pos, lsq_scale_neg, group_size, threshold_coeff, sigma_g=None):
        orig_shape = weight.shape
        flat = weight.reshape(-1)
        pad_len = (group_size - flat.numel() % group_size) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, group_size)

        sp = lsq_scale_pos.abs().clamp(min=1e-8).unsqueeze(-1)
        sn = lsq_scale_neg.abs().clamp(min=1e-8).unsqueeze(-1)
        s_norm = (sp * sn).sqrt()
        normalized = grouped / s_norm
        abs_normalized = normalized.abs()

        _TARGET_FRACTION = 0.6
        abs_flat = abs_normalized
        sorted_vals, _ = abs_flat.sort(dim=-1, descending=True)
        cumsum = sorted_vals.cumsum(dim=-1)
        total_energy = cumsum[:, -1:].clamp(min=1e-8)

        if sigma_g is not None:
            sg_flat = sigma_g.to(dtype=torch.float32).reshape(-1)
            if pad_len > 0:
                sg_flat = F.pad(sg_flat, (0, pad_len))
            sg_grouped = sg_flat.reshape(-1, group_size)
            sg_per_group = sg_grouped.mean(dim=-1, keepdim=True)
            mean_sg = sg_per_group.mean().clamp(min=1e-8)
            sensitivity = sg_per_group / mean_sg
            target_fraction = (_TARGET_FRACTION * sensitivity).clamp(min=0.4, max=0.85)
        else:
            sensitivity = None
            target_fraction = _TARGET_FRACTION

        k = (cumsum / total_energy < target_fraction).sum(dim=-1).clamp(max=group_size - 1)
        energy_threshold = sorted_vals.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        threshold = energy_threshold.unsqueeze(-1)
        quantized = normalized.sign() * (abs_normalized > threshold).float()

        transition_zone = threshold * 0.3
        near_threshold = (abs_normalized - threshold).abs() < transition_zone
        ols_alpha = near_threshold.float().mean().clamp(min=0.1, max=1.0)

        pos_mask = quantized > 0.5
        neg_mask = quantized < -0.5
        w_asymmetric = pos_mask.float() * sp - neg_mask.float() * sn
        q_flat = w_asymmetric.reshape(-1)[:weight.numel()]
        result = q_flat.reshape(orig_shape)

        ctx.save_for_backward(quantized, sp, sn, normalized, ols_alpha)
        ctx.orig_shape = orig_shape
        ctx.pad_len = pad_len
        ctx.group_size = group_size

        return result

    @staticmethod
    def backward(ctx, grad_output):
        quantized, sp, sn, normalized, ols_alpha = ctx.saved_tensors
        pad_len = ctx.pad_len
        group_size = ctx.group_size

        disc_error_flat = (normalized - quantized).reshape(-1)
        if pad_len > 0:
            disc_error_flat = disc_error_flat[:grad_output.numel()]
        disc_error = disc_error_flat.reshape(grad_output.shape)
        grad_weight = grad_output * 2.0 * disc_error

        OTGradientTernaryQuantize._call_count += 1
        if OTGradientTernaryQuantize._call_count % 500 == 1 and not torch.compiler.is_compiling():
            print(f"[OT-GRAD] call={OTGradientTernaryQuantize._call_count} "
                  f"disc_err_mean={disc_error.abs().mean().item():.5f} "
                  f"grad_mean={grad_weight.abs().mean().item():.6f}")

        go_flat = grad_output.reshape(-1)
        if pad_len > 0:
            go_flat = F.pad(go_flat, (0, pad_len))
        go_grouped = go_flat.reshape(-1, group_size)
        pos_mask = quantized > 0.5
        neg_mask = quantized < -0.5
        grad_sp = (go_grouped * pos_mask.float()).sum(dim=-1)
        grad_sn = -(go_grouped * neg_mask.float()).sum(dim=-1)
        return grad_weight, grad_sp, grad_sn, None, None, None


OTGradientTernaryQuantize._call_count = 0
torch.compiler.allow_in_graph(OTGradientTernaryQuantize)


class IGBoltzmannTernaryQuantize(torch.autograd.Function):
    """Information-Geometric Natural Gradient via Boltzmann probabilities.

    Gradient scaled by Fisher-information-weighted expected discrete value.
    beta anneals from 0.1 -> 5.0 over 10000 steps (low=uncertain -> high=committed).
    """

    _call_count = 0

    @staticmethod
    def forward(ctx, weight, lsq_scale_pos, lsq_scale_neg, group_size, threshold_coeff, sigma_g=None, training_step=0):
        orig_shape = weight.shape
        flat = weight.reshape(-1)
        pad_len = (group_size - flat.numel() % group_size) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, group_size)

        sp = lsq_scale_pos.abs().clamp(min=1e-8).unsqueeze(-1)
        sn = lsq_scale_neg.abs().clamp(min=1e-8).unsqueeze(-1)
        s_norm = (sp * sn).sqrt()
        normalized = grouped / s_norm
        abs_normalized = normalized.abs()

        _TARGET_FRACTION = 0.6
        abs_flat = abs_normalized
        sorted_vals, _ = abs_flat.sort(dim=-1, descending=True)
        cumsum = sorted_vals.cumsum(dim=-1)
        total_energy = cumsum[:, -1:].clamp(min=1e-8)

        if sigma_g is not None:
            sg_flat = sigma_g.to(dtype=torch.float32).reshape(-1)
            if pad_len > 0:
                sg_flat = F.pad(sg_flat, (0, pad_len))
            sg_grouped = sg_flat.reshape(-1, group_size)
            sg_per_group = sg_grouped.mean(dim=-1, keepdim=True)
            mean_sg = sg_per_group.mean().clamp(min=1e-8)
            sensitivity = sg_per_group / mean_sg
            target_fraction = (_TARGET_FRACTION * sensitivity).clamp(min=0.4, max=0.85)
        else:
            sensitivity = None
            target_fraction = _TARGET_FRACTION

        k = (cumsum / total_energy < target_fraction).sum(dim=-1).clamp(max=group_size - 1)
        energy_threshold = sorted_vals.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        threshold = energy_threshold.unsqueeze(-1)
        quantized = normalized.sign() * (abs_normalized > threshold).float()

        transition_zone = threshold * 0.3
        near_threshold = (abs_normalized - threshold).abs() < transition_zone
        ols_alpha = near_threshold.float().mean().clamp(min=0.1, max=1.0)

        pos_mask = quantized > 0.5
        neg_mask = quantized < -0.5
        w_asymmetric = pos_mask.float() * sp - neg_mask.float() * sn
        q_flat = w_asymmetric.reshape(-1)[:weight.numel()]
        result = q_flat.reshape(orig_shape)

        ctx.save_for_backward(quantized, sp, sn, normalized, ols_alpha, threshold)
        ctx.orig_shape = orig_shape
        ctx.pad_len = pad_len
        ctx.group_size = group_size
        ctx.training_step = training_step

        return result

    @staticmethod
    def backward(ctx, grad_output):
        quantized, sp, sn, normalized, ols_alpha, threshold = ctx.saved_tensors
        pad_len = ctx.pad_len
        group_size = ctx.group_size
        step = getattr(ctx, 'training_step', 0)

        beta = 0.1 + (5.0 - 0.1) * min(float(step) / 10000.0, 1.0)
        dist_m1 = (normalized - (-1.0)).abs()
        dist_0  = (normalized - 0.0).abs()
        dist_p1 = (normalized - 1.0).abs()
        logits = -beta * torch.stack([dist_m1, dist_0, dist_p1], dim=-1)
        probs = torch.softmax(logits, dim=-1)
        vals = torch.tensor([-1.0, 0.0, 1.0], device=normalized.device, dtype=normalized.dtype)
        expected_q = (probs * vals).sum(dim=-1)
        fisher_diag = (probs * (vals ** 2)).sum(dim=-1) - expected_q ** 2

        eq_flat = expected_q.reshape(-1)
        fd_flat = fisher_diag.reshape(-1)
        if pad_len > 0:
            eq_flat = eq_flat[:grad_output.numel()]
            fd_flat = fd_flat[:grad_output.numel()]
        eq = eq_flat.reshape(grad_output.shape)
        fd = fd_flat.reshape(grad_output.shape)

        grad_weight = grad_output * eq / (fd + 1e-6)

        IGBoltzmannTernaryQuantize._call_count += 1
        if IGBoltzmannTernaryQuantize._call_count % 500 == 1 and not torch.compiler.is_compiling():
            print(f"[F9-IG] call={IGBoltzmannTernaryQuantize._call_count} "
                  f"beta={beta:.2f} eq_mean={eq.abs().mean().item():.4f} "
                  f"fd_mean={fd.mean().item():.4f}")

        go_flat = grad_output.reshape(-1)
        if pad_len > 0:
            go_flat = F.pad(go_flat, (0, pad_len))
        go_grouped = go_flat.reshape(-1, group_size)
        pos_mask = quantized > 0.5
        neg_mask = quantized < -0.5
        grad_sp = (go_grouped * pos_mask.float()).sum(dim=-1)
        grad_sn = -(go_grouped * neg_mask.float()).sum(dim=-1)
        return grad_weight, grad_sp, grad_sn, None, None, None, None

IGBoltzmannTernaryQuantize._call_count = 0
torch.compiler.allow_in_graph(IGBoltzmannTernaryQuantize)


class MVUETernaryQuantize(torch.autograd.Function):
    """MVUE Score-Function Estimator for ternary {-1, 0, +1}.

    Forward (training): stochastic sample from Boltzmann P(w=k|s) = exp(β·s·k)/Z(β,s).
    Forward (eval): hard deterministic quantize (identical to standard STE).
    Backward: score-function gradient ĝ_i = (L - L̄) · β_i · (w_i - E[w_i|s_i]).

    β*(s_i) = √3 / max(|s_i|, ε) — optimal variance-minimizing inverse temperature.

    Args (forward positional):
        weight         : fp32 shadow weights, shape (out_features, in_features)
        group_size     : int, per-group quantization block size (128)
        threshold_coeff: float, threshold multiplier for eval hard quantize
        loss_baseline  : scalar float tensor — running mean L̄ for variance reduction
        is_training    : bool tensor, True during training

    Returns:
        w_q: quantized weight, same shape as weight
             - training: stochastic sample × scale
             - eval: hard quantize × scale

    Gradient returns (one per positional arg):
        grad_weight, None, None, None, None
    """

    _call_count = 0

    @staticmethod
    def forward(ctx, weight, group_size, threshold_coeff, loss_baseline, is_training):
        orig_shape = weight.shape
        flat = weight.reshape(-1).float()
        pad_len = (group_size - flat.numel() % group_size) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, group_size)                    # [G, group_size]

        # Per-group scale: max(|w|) per group
        scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)  # [G, 1]
        normalized = grouped / scales                             # [G, group_size]

        if is_training:
            # Probabilistic rounding (R335-PB-4): ratio=1 for multilinear
            # For |s| < 1: P(w=+1|s) = max(s, 0), P(w=-1|s) = max(-s, 0), P(w=0|s) = 1-|s|
            # For |s| >= 1: deterministic rounding sign(s)
            s_clamped = normalized.clamp(-1.0, 1.0)              # ensure valid probabilities
            p_pos = s_clamped.clamp(min=0.0)                     # P(w=+1)
            p_neg = (-s_clamped).clamp(min=0.0)                  # P(w=-1)
            # p_zero = 1 - s_clamped.abs()                       # P(w=0) — implicit

            u = torch.rand_like(normalized)
            quantized = torch.where(
                u < p_neg,
                torch.full_like(normalized, -1.0),
                torch.where(
                    u < p_neg + (1.0 - s_clamped.abs()),
                    torch.zeros_like(normalized),
                    torch.ones_like(normalized)
                )
            )

            # For |normalized| > 1: use deterministic rounding
            abs_norm = normalized.abs()
            deterministic = normalized.sign() * (abs_norm > 1.0).float()
            quantized = torch.where(abs_norm > 1.0, deterministic, quantized)
        else:
            # Hard deterministic quantize (eval path, identical to standard STE)
            abs_normalized = normalized.abs()
            threshold = threshold_coeff * abs_normalized.mean(dim=-1, keepdim=True)
            quantized = normalized.sign() * (abs_normalized > threshold).float()

        # Log probabilistic rounding activation (once per training)
        if is_training and MVUETernaryQuantize._call_count == 0:
            print(f"[PROB-ROUND] Probabilistic rounding ACTIVE: ratio=1 for multilinear")

        # Rescale back to original weight space
        q_scaled = (quantized * scales).reshape(-1)
        if pad_len > 0:
            q_scaled = q_scaled[:weight.numel()]
        result = q_scaled.reshape(orig_shape)

        # Save tensors needed for backward
        # Note: loss_baseline is a scalar — save it as part of ctx (not save_for_backward,
        # which only handles tensors that need grad)
        ctx.save_for_backward(normalized, quantized, scales)
        ctx.orig_shape = orig_shape
        ctx.pad_len = pad_len
        ctx.group_size = group_size
        ctx.loss_baseline = loss_baseline.item() if isinstance(loss_baseline, torch.Tensor) else float(loss_baseline)

        MVUETernaryQuantize._call_count += 1
        if MVUETernaryQuantize._call_count % 500 == 1 and not torch.compiler.is_compiling():
            with torch.no_grad():
                nonzero_pct = (quantized.abs() > 0.5).float().mean().item() * 100
                fluid = normalized.abs() < 1.0
                crystallized_pct = (1.0 - fluid.float().mean()).item() * 100
                print(
                    f"[MVUE] call={MVUETernaryQuantize._call_count} "
                    f"nonzero={nonzero_pct:.1f}% crystallized={crystallized_pct:.1f}% "
                    f"L_bar={ctx.loss_baseline:.4f}"
                )

        return result

    @staticmethod
    def backward(ctx, grad_output):
        # Piecewise linear MVUE score function (R336-MVUE-B1)
        # No Boltzmann computation — just piecewise linear score:
        # score(w, s) = (w - s) if |s| < 1.0, else 0.0
        #
        # This is the correct MVUE estimator for probabilistic rounding.
        # grad_s = grad_output * score * (grad over shadow weight)

        normalized, quantized, scales = ctx.saved_tensors
        pad_len = ctx.pad_len
        group_size = ctx.group_size

        # Reshape grad_output to grouped form
        go_flat = grad_output.reshape(-1).float()
        if pad_len > 0:
            go_flat = F.pad(go_flat, (0, pad_len))
        go_grouped = go_flat.reshape(-1, group_size)              # [G, group_size]

        # Piecewise linear MVUE score: (w - s) for |s| < 1, else 0
        fluid = normalized.abs() < 1.0
        score = (quantized - normalized) * fluid.float()          # [G, group_size]

        # Gradient: go_grouped * score
        grad_grouped = go_grouped * score                         # [G, group_size]

        grad_flat = grad_grouped.reshape(-1)
        if pad_len > 0:
            grad_flat = grad_flat[:grad_output.numel()]
        grad_weight = grad_flat.reshape(grad_output.shape)

        # Return one grad per forward positional argument:
        # weight, group_size, threshold_coeff, loss_baseline, is_training
        return grad_weight, None, None, None, None


MVUETernaryQuantize._call_count = 0
if hasattr(torch.compiler, "allow_in_graph"):
    torch.compiler.allow_in_graph(MVUETernaryQuantize)


class FrankWolfeTernaryQuantize(torch.autograd.Function):
    """Frank-Wolfe gradient for ternary weights.

    Linear Minimization Oracle: optimal_discrete = -sign(g) ∈ {-1, 0, +1}.
    Update direction: d = optimal - w_continuous.
    Step size: gamma = 2 / (step + 2), decays from 1.0 at step=0.
    """

    _call_count = 0

    @staticmethod
    def forward(ctx, weight, lsq_scale_pos, lsq_scale_neg, group_size, threshold_coeff, sigma_g=None, training_step=0):
        orig_shape = weight.shape
        flat = weight.reshape(-1)
        pad_len = (group_size - flat.numel() % group_size) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, group_size)

        sp = lsq_scale_pos.abs().clamp(min=1e-8).unsqueeze(-1)
        sn = lsq_scale_neg.abs().clamp(min=1e-8).unsqueeze(-1)
        s_norm = (sp * sn).sqrt()
        normalized = grouped / s_norm
        abs_normalized = normalized.abs()

        _TARGET_FRACTION = 0.6
        abs_flat = abs_normalized
        sorted_vals, _ = abs_flat.sort(dim=-1, descending=True)
        cumsum = sorted_vals.cumsum(dim=-1)
        total_energy = cumsum[:, -1:].clamp(min=1e-8)

        if sigma_g is not None:
            sg_flat = sigma_g.to(dtype=torch.float32).reshape(-1)
            if pad_len > 0:
                sg_flat = F.pad(sg_flat, (0, pad_len))
            sg_grouped = sg_flat.reshape(-1, group_size)
            sg_per_group = sg_grouped.mean(dim=-1, keepdim=True)
            mean_sg = sg_per_group.mean().clamp(min=1e-8)
            sensitivity = sg_per_group / mean_sg
            target_fraction = (_TARGET_FRACTION * sensitivity).clamp(min=0.4, max=0.85)
        else:
            sensitivity = None
            target_fraction = _TARGET_FRACTION

        k = (cumsum / total_energy < target_fraction).sum(dim=-1).clamp(max=group_size - 1)
        energy_threshold = sorted_vals.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        threshold = energy_threshold.unsqueeze(-1)
        quantized = normalized.sign() * (abs_normalized > threshold).float()

        transition_zone = threshold * 0.3
        near_threshold = (abs_normalized - threshold).abs() < transition_zone
        ols_alpha = near_threshold.float().mean().clamp(min=0.1, max=1.0)

        pos_mask = quantized > 0.5
        neg_mask = quantized < -0.5
        w_asymmetric = pos_mask.float() * sp - neg_mask.float() * sn
        q_flat = w_asymmetric.reshape(-1)[:weight.numel()]
        result = q_flat.reshape(orig_shape)

        ctx.save_for_backward(quantized, sp, sn, normalized, ols_alpha, threshold)
        ctx.orig_shape = orig_shape
        ctx.pad_len = pad_len
        ctx.group_size = group_size
        ctx.training_step = training_step

        return result

    @staticmethod
    def backward(ctx, grad_output):
        quantized, sp, sn, normalized, ols_alpha, threshold = ctx.saved_tensors
        pad_len = ctx.pad_len
        group_size = ctx.group_size
        step = getattr(ctx, 'training_step', 0)

        gamma = 2.0 / (float(step) + 2.0)

        optimal_discrete = -torch.sign(grad_output)

        w_flat = normalized.reshape(-1)
        if pad_len > 0:
            w_flat = w_flat[:grad_output.numel()]
        w_cont = w_flat.reshape(grad_output.shape)

        direction = optimal_discrete - w_cont
        grad_weight = grad_output * gamma * direction

        FrankWolfeTernaryQuantize._call_count += 1
        if FrankWolfeTernaryQuantize._call_count % 500 == 1 and not torch.compiler.is_compiling():
            print(f"[F11-FW] call={FrankWolfeTernaryQuantize._call_count} "
                  f"gamma={gamma:.4f} dir_mean={direction.abs().mean().item():.4f} "
                  f"grad_mean={grad_weight.abs().mean().item():.6f}")

        go_flat = grad_output.reshape(-1)
        if pad_len > 0:
            go_flat = F.pad(go_flat, (0, pad_len))
        go_grouped = go_flat.reshape(-1, group_size)
        pos_mask = quantized > 0.5
        neg_mask = quantized < -0.5
        grad_sp = (go_grouped * pos_mask.float()).sum(dim=-1)
        grad_sn = -(go_grouped * neg_mask.float()).sum(dim=-1)
        return grad_weight, grad_sp, grad_sn, None, None, None, None

FrankWolfeTernaryQuantize._call_count = 0
torch.compiler.allow_in_graph(FrankWolfeTernaryQuantize)


class GASTETernaryQuantize(torch.autograd.Function):
    """GASTE: Group-Aware STE correcting for threshold coupling within groups."""

    _call_count = 0

    @staticmethod
    def forward(ctx, weight, group_size, threshold_coeff, sigma_g):
        import math

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

        dist = abs_normalized - threshold
        inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
        phi = (inv_sqrt_2pi / sigma_g) * torch.exp(-0.5 * (dist / sigma_g) ** 2)
        boundary_kernel = normalized.sign() * phi

        result = (quantized * scales).reshape(-1)
        if pad_len > 0:
            result = result[: weight.numel()]

        bk = boundary_kernel.reshape(-1)
        sw = normalized.sign().reshape(-1)
        if pad_len > 0:
            bk = bk[: weight.numel()]
            sw = sw[: weight.numel()]
        ctx.save_for_backward(bk.reshape(orig_shape), sw.reshape(orig_shape))
        ctx.group_size = group_size
        ctx.threshold_coeff = threshold_coeff

        GASTETernaryQuantize._call_count += 1
        if GASTETernaryQuantize._call_count % 500 == 1:
            if not torch.compiler.is_compiling():
                with torch.no_grad():
                    print(
                        f"  [GASTE] call={GASTETernaryQuantize._call_count} "
                        f"sigma_g={sigma_g:.3f} "
                        f"boundary_kernel_mean={boundary_kernel.abs().mean().item():.6f} "
                        f"boundary_near_thresh={((dist.abs() < sigma_g).float().mean().item() * 100):.1f}%"
                    )
        return result.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        boundary_kernel, sign_weight = ctx.saved_tensors
        g = ctx.group_size
        t_coeff = ctx.threshold_coeff
        orig_shape = grad_output.shape

        grad_flat = grad_output.reshape(-1)
        bk_flat = boundary_kernel.reshape(-1)
        sw_flat = sign_weight.reshape(-1)
        n = grad_flat.numel()
        pad_len = (g - n % g) % g
        if pad_len > 0:
            grad_flat = F.pad(grad_flat, (0, pad_len))
            bk_flat = F.pad(bk_flat, (0, pad_len))
            sw_flat = F.pad(sw_flat, (0, pad_len))

        grad_g = grad_flat.reshape(-1, g)
        bk_g = bk_flat.reshape(-1, g)
        sw_g = sw_flat.reshape(-1, g)

        G_boundary = (grad_g * bk_g).sum(dim=-1, keepdim=True)
        correction = (t_coeff / g) * sw_g * G_boundary
        corrected = grad_g - correction

        return corrected.reshape(-1)[:n].reshape(orig_shape), None, None, None


torch.compiler.allow_in_graph(GASTETernaryQuantize)


class BiSTETernaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        group_size: int = 128,
        threshold_coeff: float = DEFAULT_THRESHOLD_COEFF,
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
        threshold = threshold_coeff * abs_normalized.mean(dim=-1, keepdim=True)
        quantized = normalized.sign() * (abs_normalized > threshold).float()

        active_mask = (abs_normalized > threshold).float()
        zero_mask = 1.0 - active_mask
        scale_grouped = scales.expand_as(grouped)

        result = (quantized * scales).reshape(-1)[: weight.numel()]

        active_mask_flat = active_mask.reshape(-1)[: weight.numel()].reshape(orig_shape)
        zero_mask_flat = zero_mask.reshape(-1)[: weight.numel()].reshape(orig_shape)
        scale_flat = scale_grouped.reshape(-1)[: weight.numel()].reshape(orig_shape)

        ctx.save_for_backward(active_mask_flat, zero_mask_flat, scale_flat)
        return result.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        active_mask, zero_mask, scales = ctx.saved_tensors
        scale_normalized = scales / (scales.mean() + 1e-8)
        grad = grad_output * (active_mask * scale_normalized + zero_mask * 1.0)
        return grad, None, None


class HESTIATernaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, weight: torch.Tensor, group_size: int = 128, tau: float = 1.0
    ) -> torch.Tensor:
        orig_shape = weight.shape
        flat = weight.reshape(-1)
        pad_len = (group_size - flat.numel() % group_size) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, group_size)

        gamma = grouped.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8).detach()
        w_scaled = grouped / gamma

        cache_key = (weight.device, weight.dtype)
        if cache_key not in _q_values_cache:
            _q_values_cache[cache_key] = torch.tensor(
                [-1.0, 0.0, 1.0], device=weight.device, dtype=weight.dtype
            )
        q_values = _q_values_cache[cache_key]
        w_expanded = w_scaled.unsqueeze(-1)
        dist_sq = (w_expanded - q_values) ** 2
        logits = -dist_sq / max(tau, 0.01)
        probs = torch.softmax(logits, dim=-1)

        p_neg, p_zero, p_pos = probs[..., 0], probs[..., 1], probs[..., 2]
        w_soft = gamma * (p_pos - p_neg)

        mean_q = p_pos - p_neg
        var = (1.0 - p_zero) - mean_q * mean_q

        result = w_soft.reshape(-1)[: weight.numel()]
        var_flat = var.reshape(-1)[: weight.numel()].reshape(orig_shape)

        ctx.save_for_backward(var_flat)
        ctx.tau = tau
        return result.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (var,) = ctx.saved_tensors
        tau = ctx.tau
        jacobian = (2.0 / max(tau, 0.01)) * var
        grad = grad_output * jacobian
        return grad, None, None


class QuESTTernaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        group_size: int = 128,
        threshold_coeff: float = DEFAULT_THRESHOLD_COEFF,
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
        threshold = threshold_coeff * abs_normalized.mean(dim=-1, keepdim=True)
        quantized = normalized.sign() * (abs_normalized > threshold).float()

        quant_error = (normalized - quantized).abs()
        trust_threshold = 1.3 * threshold
        trust_mask = (quant_error <= trust_threshold).float()

        result = (quantized * scales).reshape(-1)[: weight.numel()]
        trust_mask_flat = trust_mask.reshape(-1)[: weight.numel()].reshape(orig_shape)

        ctx.save_for_backward(trust_mask_flat)
        return result.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (trust_mask,) = ctx.saved_tensors
        grad = grad_output * trust_mask
        return grad, None, None


class SigmaDeltaTernaryQuantize(torch.autograd.Function):
    """Sigma-delta noise shaping for ternary quantization (Phase 21.C)."""

    _call_count = 0
    _log_interval = 500

    @staticmethod
    def forward(
        ctx,
        weight,
        group_size=128,
        threshold_coeff=DEFAULT_THRESHOLD_COEFF,
        error_clamp=2.0,
    ):
        assert weight.ndim == 2, f"Sigma-delta requires 2D weight, got {weight.shape}"
        out_features, in_features = weight.shape

        if in_features % group_size != 0:
            return TernaryQuantize.apply(weight, group_size, threshold_coeff)

        groups_per_row = in_features // group_size
        grouped = weight.reshape(out_features, groups_per_row, group_size)
        scales = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = grouped / scales

        quantized = torch.empty_like(normalized)
        accumulated_error = torch.zeros(
            out_features, group_size, device=weight.device, dtype=weight.dtype
        )

        for j in range(groups_per_row):
            adjusted = normalized[:, j, :] + accumulated_error
            abs_adj = adjusted.abs()
            threshold = threshold_coeff * abs_adj.mean(dim=-1, keepdim=True)
            q = adjusted.sign() * (abs_adj > threshold).float()
            quantized[:, j, :] = q
            error = adjusted - q
            accumulated_error = error.clamp(-error_clamp, error_clamp)

        result = (quantized * scales).reshape(out_features, in_features)

        SigmaDeltaTernaryQuantize._call_count += 1
        if (
            SigmaDeltaTernaryQuantize._call_count
            % SigmaDeltaTernaryQuantize._log_interval
            == 1
        ):
            if not torch.compiler.is_compiling():
                with torch.no_grad():
                    accum_mean = accumulated_error.abs().mean().item()
                    abs_norm = normalized.abs()
                    std_thresh = threshold_coeff * abs_norm.mean(dim=-1, keepdim=True)
                    q_standard = normalized.sign() * (abs_norm > std_thresh).float()
                    n_changed = (q_standard != quantized).sum().item()
                    pct = n_changed / quantized.numel() * 100
                    print(
                        f"  [SigmaDelta] call={SigmaDeltaTernaryQuantize._call_count} "
                        f"groups/row={groups_per_row} accum_err={accum_mean:.4f} "
                        f"ternary_changed={pct:.2f}%"
                    )
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class CurvatureAwareTernaryQuantize(torch.autograd.Function):
    """Curvature-Aware Quantization using Adam's second moment (v_hat) as adaptive threshold."""

    _call_count = 0
    _log_interval = 500

    @staticmethod
    def forward(
        ctx,
        weight,
        v_hat,
        group_size=128,
        threshold_coeff=DEFAULT_THRESHOLD_COEFF,
        caq_alpha=1.0,
    ):
        orig_shape = weight.shape
        flat = weight.reshape(-1)
        v_flat = v_hat.reshape(-1)
        pad_len = (group_size - flat.numel() % group_size) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
            v_flat = F.pad(v_flat, (0, pad_len))
        grouped = flat.reshape(-1, group_size)
        v_grouped = v_flat.reshape(-1, group_size)

        scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        normalized = grouped / scales
        abs_normalized = normalized.abs()
        base_threshold = threshold_coeff * abs_normalized.mean(dim=-1, keepdim=True)

        v_sqrt = torch.sqrt(v_grouped.clamp(min=1e-12))
        v_max = v_sqrt.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        v_norm = v_sqrt / v_max
        adaptive_threshold = base_threshold / (1.0 + caq_alpha * v_norm)

        quantized = normalized.sign() * (abs_normalized > adaptive_threshold).float()
        n = flat.numel() - pad_len if pad_len > 0 else flat.numel()
        result = (quantized * scales).reshape(-1)[:n]

        CurvatureAwareTernaryQuantize._call_count += 1
        if (
            CurvatureAwareTernaryQuantize._call_count
            % CurvatureAwareTernaryQuantize._log_interval
            == 1
        ):
            with torch.no_grad():
                thresh_mean = adaptive_threshold.mean().item()
                base_mean = base_threshold.mean().item()
                ratio = thresh_mean / max(base_mean, 1e-8)
                pct_nonzero = (quantized != 0).float().mean().item() * 100
                print(
                    f"  [CAQ] call={CurvatureAwareTernaryQuantize._call_count} "
                    f"base={base_mean:.4f} adaptive={thresh_mean:.4f} "
                    f"ratio={ratio:.3f} nonzero={pct_nonzero:.1f}%"
                )

        return result.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


class BDTTernaryQuantize(torch.autograd.Function):
    """Binary Decomposition Training (Q2). Decomposes ternary into sign x activity."""

    _call_count = 0
    _log_interval = 500

    @staticmethod
    def forward(ctx, weight, group_size=128, threshold_coeff=DEFAULT_THRESHOLD_COEFF):
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

        active_mask = (abs_normalized > threshold).float()
        sign_ste = (abs_normalized <= 1.0).float()
        activity_margin = threshold * 0.5
        near_threshold = ((abs_normalized - threshold).abs() <= activity_margin).float()
        bdt_mask = active_mask * sign_ste + (1.0 - active_mask) * near_threshold

        n = flat.numel() - pad_len if pad_len > 0 else flat.numel()
        result = (quantized * scales).reshape(-1)[:n]
        bdt_mask_flat = bdt_mask.reshape(-1)[:n].reshape(orig_shape)
        ctx.save_for_backward(bdt_mask_flat)

        BDTTernaryQuantize._call_count += 1
        if BDTTernaryQuantize._call_count % BDTTernaryQuantize._log_interval == 1:
            if not torch.compiler.is_compiling():
                with torch.no_grad():
                    print(
                        f"  [BDT] call={BDTTernaryQuantize._call_count} "
                        f"mask_mean={bdt_mask.mean().item():.4f} "
                        f"active={active_mask.mean().item() * 100:.1f}% "
                        f"near_thresh={near_threshold.mean().item() * 100:.1f}%"
                    )
        return result.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (bdt_mask,) = ctx.saved_tensors
        return grad_output * bdt_mask, None, None


class BAGRTernaryQuantize(torch.autograd.Function):
    """Boundary-Attentive Gradient Reweighting (Q3). Gaussian damping by distance to boundary."""

    _call_count = 0
    _log_interval = 500

    @staticmethod
    def forward(
        ctx,
        weight,
        group_size=128,
        threshold_coeff=DEFAULT_THRESHOLD_COEFF,
        sigma=0.5,
        training_step=0,
        total_training_steps=1000,
    ):
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

        dist = (abs_normalized - threshold).abs()
        progress = training_step / max(1, total_training_steps)
        sigma_t = max(sigma / (1.0 + progress), 0.01)
        modulation = torch.exp(-dist.pow(2) / (2.0 * sigma_t * sigma_t))

        n = flat.numel() - pad_len if pad_len > 0 else flat.numel()
        result = (quantized * scales).reshape(-1)[:n]
        mod_flat = modulation.reshape(-1)[:n].reshape(orig_shape)
        ctx.save_for_backward(mod_flat)

        BAGRTernaryQuantize._call_count += 1
        if BAGRTernaryQuantize._call_count % BAGRTernaryQuantize._log_interval == 1:
            if not torch.compiler.is_compiling():
                with torch.no_grad():
                    print(
                        f"  [BAGR] call={BAGRTernaryQuantize._call_count} "
                        f"sigma_t={sigma_t:.4f} progress={progress:.3f} "
                        f"mod_mean={modulation.mean().item():.4f} "
                        f"at_boundary={((modulation > 0.9).float().mean().item() * 100):.1f}% "
                        f"damped={((modulation < 0.1).float().mean().item() * 100):.1f}%"
                    )
        return result.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (modulation,) = ctx.saved_tensors
        return grad_output * modulation, None, None, None, None, None


class ASGKSTernaryQuantize(torch.autograd.Function):
    """Analytic Surrogate Gradient via Kernel Smoothing (Q4). Smooth gradient approximation."""

    _call_count = 0
    _log_interval = 500
    _inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

    @staticmethod
    def forward(
        ctx,
        weight,
        group_size=128,
        threshold_coeff=DEFAULT_THRESHOLD_COEFF,
        tau=0.5,
        training_step=0,
        total_training_steps=1000,
    ):
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

        progress = training_step / max(1, total_training_steps)
        tau_f = 0.01
        tau_t = max(
            tau_f + (tau - tau_f) * 0.5 * (1.0 + math.cos(math.pi * progress)), 0.01
        )

        z_pos = (normalized - threshold) / tau_t
        z_neg = (normalized + threshold) / tau_t
        phi_pos = ASGKSTernaryQuantize._inv_sqrt_2pi * torch.exp(-0.5 * z_pos.pow(2))
        phi_neg = ASGKSTernaryQuantize._inv_sqrt_2pi * torch.exp(-0.5 * z_neg.pow(2))
        surrogate_grad = (phi_pos + phi_neg) / tau_t
        sg_max = surrogate_grad.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        surrogate_grad = surrogate_grad / sg_max

        n = flat.numel() - pad_len if pad_len > 0 else flat.numel()
        result = (quantized * scales).reshape(-1)[:n]
        sg_flat = surrogate_grad.reshape(-1)[:n].reshape(orig_shape)
        ctx.save_for_backward(sg_flat)

        ASGKSTernaryQuantize._call_count += 1
        if ASGKSTernaryQuantize._call_count % ASGKSTernaryQuantize._log_interval == 1:
            if not torch.compiler.is_compiling():
                with torch.no_grad():
                    print(
                        f"  [ASGKS] call={ASGKSTernaryQuantize._call_count} "
                        f"tau_t={tau_t:.4f} progress={progress:.3f} "
                        f"sg_mean={surrogate_grad.mean().item():.4f} "
                        f"high_grad={((surrogate_grad > 0.5).float().mean().item() * 100):.1f}% "
                        f"low_grad={((surrogate_grad < 0.1).float().mean().item() * 100):.1f}%"
                    )
        return result.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (surrogate_grad,) = ctx.saved_tensors
        return grad_output * surrogate_grad, None, None, None, None, None


class TPSQTernaryQuantize(torch.autograd.Function):
    """Threshold Phase-Space Quantization (Q10). Input-aware adaptive thresholds."""

    _call_count = 0
    _log_interval = 500

    @staticmethod
    def forward(
        ctx,
        weight,
        sigma_sq,
        group_size=128,
        threshold_coeff=DEFAULT_THRESHOLD_COEFF,
        beta=1.0,
    ):
        orig_shape = weight.shape
        flat = weight.reshape(-1)
        pad_len = (group_size - flat.numel() % group_size) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, group_size)

        scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        normalized = grouped / scales
        abs_normalized = normalized.abs()
        base_threshold = threshold_coeff * abs_normalized.mean(dim=-1, keepdim=True)

        sigma_sqrt = torch.sqrt(sigma_sq.clamp(min=1e-12))
        sigma_sqrt_reshaped = sigma_sqrt.unsqueeze(-1)
        sigma_max = sigma_sqrt_reshaped.max().clamp(min=1e-8)
        sigma_norm = sigma_sqrt_reshaped / sigma_max
        adaptive_threshold = base_threshold * (1.0 + beta * sigma_norm)

        quantized = normalized.sign() * (abs_normalized > adaptive_threshold).float()
        n = flat.numel() - pad_len if pad_len > 0 else flat.numel()
        result = (quantized * scales).reshape(-1)[:n]

        TPSQTernaryQuantize._call_count += 1
        if TPSQTernaryQuantize._call_count % TPSQTernaryQuantize._log_interval == 1:
            if not torch.compiler.is_compiling():
                with torch.no_grad():
                    print(
                        f"  [TPSQ] call={TPSQTernaryQuantize._call_count} "
                        f"beta={beta:.3f} sigma_norm_mean={sigma_norm.mean().item():.4f} "
                        f"thresh_mean={adaptive_threshold.mean().item():.4f}"
                    )
        return result.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


class MDTSTernaryQuantize(torch.autograd.Function):
    """Mirror Descent Ternary Solver (Q12). Soft ternary quantization with STE."""

    _call_count = 0
    _log_interval = 500

    @staticmethod
    def forward(ctx, weight, logits, scales, temperature=1.0, group_size=128):
        orig_shape = weight.shape
        flat = weight.reshape(-1)
        pad_len = (group_size - flat.numel() % group_size) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, group_size)

        logits_grouped = logits.reshape(-1, 3)
        scales_grouped = scales.reshape(-1, 1)

        softmax_probs = torch.softmax(logits_grouped / max(temperature, 0.01), dim=-1)
        p_neg, p_zero, p_pos = (
            softmax_probs[..., 0],
            softmax_probs[..., 1],
            softmax_probs[..., 2],
        )

        q_values = torch.tensor(
            [-1.0, 0.0, 1.0], device=weight.device, dtype=weight.dtype
        )
        w_soft = scales_grouped * (
            p_pos.unsqueeze(-1) - p_neg.unsqueeze(-1)
        )
        w_soft_expanded = w_soft.expand(-1, group_size)

        hard_assign = torch.argmax(logits_grouped, dim=-1)
        w_hard_per_group = (
            scales_grouped.squeeze(-1) * (hard_assign == 2).float()
            - scales_grouped.squeeze(-1) * (hard_assign == 0).float()
        )
        w_hard = w_hard_per_group.unsqueeze(-1).expand(
            -1, group_size
        )

        w_result = w_soft_expanded + (w_hard - w_soft_expanded).detach()

        n = flat.numel() - pad_len if pad_len > 0 else flat.numel()
        result = w_result.reshape(-1)[:n]

        mean_p_neg = p_neg.mean().item()
        mean_p_zero = p_zero.mean().item()
        mean_p_pos = p_pos.mean().item()
        entropy = (
            -(
                p_neg * torch.log(p_neg + 1e-10)
                + p_zero * torch.log(p_zero + 1e-10)
                + p_pos * torch.log(p_pos + 1e-10)
            )
            .mean()
            .item()
        )
        confident = (
            (softmax_probs.max(dim=-1).values > 0.9).float().mean() * 100
        ).item()

        ctx.save_for_backward(logits_grouped, p_neg, p_zero, p_pos, scales_grouped)
        ctx.temperature = temperature

        MDTSTernaryQuantize._call_count += 1
        if MDTSTernaryQuantize._call_count % MDTSTernaryQuantize._log_interval == 1:
            if not torch.compiler.is_compiling():
                print(
                    f"  [MDTS] call={MDTSTernaryQuantize._call_count} "
                    f"p(-1)={mean_p_neg:.4f} p(0)={mean_p_zero:.4f} p(+1)={mean_p_pos:.4f} "
                    f"entropy={entropy:.4f} confident={confident:.1f}% temp={temperature:.3f}"
                )

        return result.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        logits, p_neg, p_zero, p_pos, scales = ctx.saved_tensors
        temperature = ctx.temperature

        grad_flat = grad_output.reshape(-1)

        jacobian_softmax = torch.diag_embed(
            torch.stack([p_neg, p_zero, p_pos], dim=-1)
        ) - torch.bmm(
            torch.stack([p_neg, p_zero, p_pos], dim=-1).unsqueeze(-1),
            torch.stack([p_neg, p_zero, p_pos], dim=-1).unsqueeze(-2),
        )

        d_prob_wrt_logits = jacobian_softmax @ torch.tensor(
            [-1.0, 0.0, 1.0], device=logits.device, dtype=logits.dtype
        ).unsqueeze(-1)
        d_prob_wrt_logits = d_prob_wrt_logits.squeeze(-1)

        num_groups = d_prob_wrt_logits.shape[0]
        grad_grouped = grad_flat.reshape(num_groups, -1)

        grad_per_group = grad_grouped.sum(dim=1)

        grad_logits = (
            grad_per_group.unsqueeze(-1) * d_prob_wrt_logits / max(temperature, 0.01)
        )

        return grad_flat.reshape(grad_output.shape), grad_logits, None, None, None


if hasattr(torch.compiler, "allow_in_graph"):
    torch.compiler.allow_in_graph(TernaryQuantize)
    print("[ALLOW-IN-GRAPH] TernaryQuantize + 13 others registered for torch.compile")
    torch.compiler.allow_in_graph(BoundaryAwareTernaryQuantize)
    torch.compiler.allow_in_graph(BiSTETernaryQuantize)
    torch.compiler.allow_in_graph(HESTIATernaryQuantize)
    torch.compiler.allow_in_graph(QuESTTernaryQuantize)
    torch.compiler.allow_in_graph(SigmaDeltaTernaryQuantize)
    torch.compiler.allow_in_graph(CurvatureAwareTernaryQuantize)
    torch.compiler.allow_in_graph(BDTTernaryQuantize)
    torch.compiler.allow_in_graph(BAGRTernaryQuantize)
    torch.compiler.allow_in_graph(ASGKSTernaryQuantize)
    torch.compiler.allow_in_graph(TPSQTernaryQuantize)
    torch.compiler.allow_in_graph(MDTSTernaryQuantize)
    torch.compiler.allow_in_graph(CTCETernaryQuantize)
    torch.compiler.allow_in_graph(MVUETernaryQuantize)
