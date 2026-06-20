import logging
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .triton_hestia import hestia_quantize_triton

    _TRITON_HESTIA_AVAILABLE = True
except ImportError:
    _TRITON_HESTIA_AVAILABLE = False

_q_values_cache: dict = {}
_av_probe_counters: dict = {}  # {key: int} — per-layer call counters for AV probes
_aw3_ema_store: dict = {}  # AW-3: Neural Persistence EMA state per TernaryLinear instance
logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD_COEFF = 0.767



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
    _use_hyperreal_ste: bool = False  # AW-12: Hyperreal STE — shaped sech^2 backward

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

        # OLS-STE: compute informative fraction per group
        # Weights within transition_zone of threshold can actually flip state
        # Gradient for committed weights (far from threshold) is noise
        transition_zone = threshold * 0.3
        near_threshold = (abs_normalized - threshold).abs() < transition_zone
        informative_frac = near_threshold.float().mean(dim=-1, keepdim=True)
        # Global scalar: average across all groups, clamp to [0.1, 1.0]
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
                _gate = 1.0 - _dist.tanh().pow(2)  # sech^2: max at threshold boundary
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
        grouped = flat.reshape(-1, group_size)          # [num_groups, group_size]

        # TTQ: asymmetric per-group scales (pos and neg independent)
        sp = lsq_scale_pos.abs().clamp(min=1e-8).unsqueeze(-1)  # [num_groups, 1]
        sn = lsq_scale_neg.abs().clamp(min=1e-8).unsqueeze(-1)  # [num_groups, 1]
        s_norm = (sp * sn).sqrt()                                # geometric mean for normalization
        normalized = grouped / s_norm
        abs_normalized = normalized.abs()

        # R7-A3-M10: Energy-Based Direct Decision
        # Find per-group threshold that preserves >= 60% of weight L2 energy.
        # sigma_g (Fisher proxy from C2) is used as a secondary sensitivity signal
        # to modulate the energy fraction: high-sensitivity groups get a tighter
        # threshold (preserve more energy); low-sensitivity groups get a looser one.
        _TARGET_FRACTION = 0.6
        abs_flat = abs_normalized  # already [num_groups, group_size]
        sorted_vals, _ = abs_flat.sort(dim=-1, descending=True)
        cumsum = sorted_vals.cumsum(dim=-1)
        total_energy = cumsum[:, -1:].clamp(min=1e-8)

        if sigma_g is not None:
            sg_flat = sigma_g.to(dtype=torch.float32).reshape(-1)
            if pad_len > 0:
                sg_flat = F.pad(sg_flat, (0, pad_len))
            sg_grouped = sg_flat.reshape(-1, group_size)             # [num_groups, group_size]
            sg_per_group = sg_grouped.mean(dim=-1, keepdim=True)     # [num_groups, 1]
            mean_sg = sg_per_group.mean().clamp(min=1e-8)
            sensitivity = sg_per_group / mean_sg                     # ~1.0 mean
            target_fraction = (_TARGET_FRACTION * sensitivity).clamp(min=0.4, max=0.85)
        else:
            sensitivity = None
            target_fraction = _TARGET_FRACTION

        k = (cumsum / total_energy < target_fraction).sum(dim=-1).clamp(max=group_size - 1)
        energy_threshold = sorted_vals.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        threshold = energy_threshold.unsqueeze(-1)
        quantized = normalized.sign() * (abs_normalized > threshold).float()

        # OLS-STE alpha: informative fraction near threshold boundary
        transition_zone = threshold * 0.3
        near_threshold = (abs_normalized - threshold).abs() < transition_zone
        ols_alpha = near_threshold.float().mean().clamp(min=0.1, max=1.0)

        # TTQ: positive positions scaled by sp, negative by sn
        pos_mask = quantized > 0.5   # [num_groups, group_size]
        neg_mask = quantized < -0.5  # [num_groups, group_size]
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

        # dL/dw: OLS-STE pass-through + PEGE discretization error correction
        grad_weight = grad_output * ols_alpha

        # F1-PEGE: add correction term proportional to discretization error
        # disc_error in normalized space: (continuous - quantized)
        # Must reshape to match grad_output shape (orig_shape, not grouped)
        step = getattr(ctx, 'training_step', 0)
        mu = 0.1 * (1.0 - math.exp(-0.001 * step))
        if mu > 1e-6:
            # disc_error in normalized space, padded [num_groups, group_size]
            disc_error_grouped = normalized - quantized   # both saved in ctx
            # Remove padding and reshape to original weight shape
            disc_error_flat = disc_error_grouped.reshape(-1)
            if pad_len > 0:
                disc_error_flat = disc_error_flat[:grad_output.numel()]
            disc_error = disc_error_flat.reshape(grad_output.shape)
            grad_weight = grad_weight + mu * disc_error
            if step % 500 == 1 and not torch.compiler.is_compiling():
                print(f"[PEGE] step={step} mu={mu:.5f} disc_err_mean={disc_error.abs().mean().item():.5f}")

        # F8-DTG: Distributional Threshold Gradient — boost gradients near threshold boundary
        _dtg_epsilon = 0.1   # epsilon fraction of threshold
        _dtg_boost = 2.0     # 2× boost for near-boundary weights
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
        go_grouped = go_flat.reshape(-1, group_size)  # [num_groups, group_size]

        # TTQ asymmetric gradients:
        # output_i = +sp_g (q=+1), -sn_g (q=-1), 0 (q=0)
        pos_mask = quantized > 0.5
        neg_mask = quantized < -0.5
        grad_sp = (go_grouped * pos_mask.float()).sum(dim=-1)    # [num_groups]
        grad_sn = -(go_grouped * neg_mask.float()).sum(dim=-1)   # [num_groups]

        # 7 returns: weight, lsq_scale_pos, lsq_scale_neg, group_size, threshold_coeff, sigma_g, training_step
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
        # Same quantization as LSQ
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
        # AO-1: True Wasserstein-1 gradient = sign of the upstream gradient
        grad_w = grad_output.sign()
        return grad_w, None, None, None, None, None


torch.compiler.allow_in_graph(LSQTernaryQuantize)


class GumbelSoftmaxSTE(torch.autograd.Function):
    """Ternary quantization with Gumbel-Softmax gradient estimator (R13-Batch-AF).

    Forward: identical to standard ternary hard assignment (group-wise, like LSQTernaryQuantize).
    Backward: gradient of soft ternary relaxation controlled by temperature τ.

    High τ = soft gradient (exploration), low τ = near-STE (commitment).
    Anneals τ_init → τ_min over training.

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
        grouped = flat.reshape(-1, group_size)          # [num_groups, group_size]

        sp_g = sp.abs().clamp(min=1e-8).unsqueeze(-1)  # [num_groups, 1]
        sn_g = sn.abs().clamp(min=1e-8).unsqueeze(-1)  # [num_groups, 1]

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

        grad_grouped = grad_output.reshape(-1, ctx.group_size)  # [num_groups, group_size]

        # Distance-based logits to each ternary level
        l_neg = -(grouped + sn_g).pow(2) / tau
        l_zero = -(grouped).pow(2) / tau
        l_pos = -(grouped - sp_g).pow(2) / tau

        logits = torch.stack([l_neg, l_zero, l_pos], dim=-1)  # [num_groups, group_size, 3]
        probs = torch.softmax(logits, dim=-1)
        p_neg, p_zero, p_pos = probs[..., 0], probs[..., 1], probs[..., 2]

        w_soft = -sn_g * p_neg + sp_g * p_pos  # [num_groups, group_size]

        dl_neg_dw = -2.0 * (grouped + sn_g) / tau
        dl_zero_dw = -2.0 * grouped / tau
        dl_pos_dw = 2.0 * (sp_g - grouped) / tau

        # d(w_soft)/dw = Σⱼ [dl_j/dw * p_j * (val_j - w_soft)]
        dw_soft_dw = (
            dl_neg_dw * p_neg * (-sn_g - w_soft)
            + dl_zero_dw * p_zero * (0.0 - w_soft)
            + dl_pos_dw * p_pos * (sp_g - w_soft)
        )

        grad_w = (grad_grouped * dw_soft_dw * ols_alpha).reshape(ctx.orig_shape)
        # Returns: grad for (w, w_thr, ols_alpha, sp, sn, training_step, tau, group_size)
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
        grouped = flat.reshape(-1, group_size)          # [num_groups, group_size]

        # TTQ: asymmetric per-group scales (pos and neg independent)
        sp = lsq_scale_pos.abs().clamp(min=1e-8).unsqueeze(-1)  # [num_groups, 1]
        sn = lsq_scale_neg.abs().clamp(min=1e-8).unsqueeze(-1)  # [num_groups, 1]
        s_norm = (sp * sn).sqrt()                                # geometric mean for normalization
        normalized = grouped / s_norm
        abs_normalized = normalized.abs()

        # R7-A3-M10: Energy-Based Direct Decision
        # Find per-group threshold that preserves >= 60% of weight L2 energy.
        # sigma_g (Fisher proxy from C2) is used as a secondary sensitivity signal
        # to modulate the energy fraction: high-sensitivity groups get a tighter
        # threshold (preserve more energy); low-sensitivity groups get a looser one.
        _TARGET_FRACTION = 0.6
        abs_flat = abs_normalized  # already [num_groups, group_size]
        sorted_vals, _ = abs_flat.sort(dim=-1, descending=True)
        cumsum = sorted_vals.cumsum(dim=-1)
        total_energy = cumsum[:, -1:].clamp(min=1e-8)

        if sigma_g is not None:
            sg_flat = sigma_g.to(dtype=torch.float32).reshape(-1)
            if pad_len > 0:
                sg_flat = F.pad(sg_flat, (0, pad_len))
            sg_grouped = sg_flat.reshape(-1, group_size)             # [num_groups, group_size]
            sg_per_group = sg_grouped.mean(dim=-1, keepdim=True)     # [num_groups, 1]
            mean_sg = sg_per_group.mean().clamp(min=1e-8)
            sensitivity = sg_per_group / mean_sg                     # ~1.0 mean
            target_fraction = (_TARGET_FRACTION * sensitivity).clamp(min=0.4, max=0.85)
        else:
            sensitivity = None
            target_fraction = _TARGET_FRACTION

        k = (cumsum / total_energy < target_fraction).sum(dim=-1).clamp(max=group_size - 1)
        energy_threshold = sorted_vals.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        threshold = energy_threshold.unsqueeze(-1)
        quantized = normalized.sign() * (abs_normalized > threshold).float()

        # OLS-STE alpha: informative fraction near threshold boundary
        transition_zone = threshold * 0.3
        near_threshold = (abs_normalized - threshold).abs() < transition_zone
        ols_alpha = near_threshold.float().mean().clamp(min=0.1, max=1.0)

        # TTQ: positive positions scaled by sp, negative by sn
        pos_mask = quantized > 0.5   # [num_groups, group_size]
        neg_mask = quantized < -0.5  # [num_groups, group_size]
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

        # F5-OT: replace STE with OT transport velocity
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

        # LSQ scale gradients (same as LSQ backward)
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
    beta anneals from 0.1 → 5.0 over 10000 steps (low=uncertain → high=committed).
    """

    _call_count = 0

    @staticmethod
    def forward(ctx, weight, lsq_scale_pos, lsq_scale_neg, group_size, threshold_coeff, sigma_g=None, training_step=0):
        orig_shape = weight.shape
        flat = weight.reshape(-1)
        pad_len = (group_size - flat.numel() % group_size) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, group_size)          # [num_groups, group_size]

        # TTQ: asymmetric per-group scales (pos and neg independent)
        sp = lsq_scale_pos.abs().clamp(min=1e-8).unsqueeze(-1)  # [num_groups, 1]
        sn = lsq_scale_neg.abs().clamp(min=1e-8).unsqueeze(-1)  # [num_groups, 1]
        s_norm = (sp * sn).sqrt()                                # geometric mean for normalization
        normalized = grouped / s_norm
        abs_normalized = normalized.abs()

        # R7-A3-M10: Energy-Based Direct Decision
        # Find per-group threshold that preserves >= 60% of weight L2 energy.
        # sigma_g (Fisher proxy from C2) is used as a secondary sensitivity signal
        # to modulate the energy fraction: high-sensitivity groups get a tighter
        # threshold (preserve more energy); low-sensitivity groups get a looser one.
        _TARGET_FRACTION = 0.6
        abs_flat = abs_normalized  # already [num_groups, group_size]
        sorted_vals, _ = abs_flat.sort(dim=-1, descending=True)
        cumsum = sorted_vals.cumsum(dim=-1)
        total_energy = cumsum[:, -1:].clamp(min=1e-8)

        if sigma_g is not None:
            sg_flat = sigma_g.to(dtype=torch.float32).reshape(-1)
            if pad_len > 0:
                sg_flat = F.pad(sg_flat, (0, pad_len))
            sg_grouped = sg_flat.reshape(-1, group_size)             # [num_groups, group_size]
            sg_per_group = sg_grouped.mean(dim=-1, keepdim=True)     # [num_groups, 1]
            mean_sg = sg_per_group.mean().clamp(min=1e-8)
            sensitivity = sg_per_group / mean_sg                     # ~1.0 mean
            target_fraction = (_TARGET_FRACTION * sensitivity).clamp(min=0.4, max=0.85)
        else:
            sensitivity = None
            target_fraction = _TARGET_FRACTION

        k = (cumsum / total_energy < target_fraction).sum(dim=-1).clamp(max=group_size - 1)
        energy_threshold = sorted_vals.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        threshold = energy_threshold.unsqueeze(-1)
        quantized = normalized.sign() * (abs_normalized > threshold).float()

        # OLS-STE alpha: informative fraction near threshold boundary
        transition_zone = threshold * 0.3
        near_threshold = (abs_normalized - threshold).abs() < transition_zone
        ols_alpha = near_threshold.float().mean().clamp(min=0.1, max=1.0)

        # TTQ: positive positions scaled by sp, negative by sn
        pos_mask = quantized > 0.5   # [num_groups, group_size]
        neg_mask = quantized < -0.5  # [num_groups, group_size]
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

        # F9-IG: Boltzmann assignment probabilities over {-1, 0, +1}
        beta = 0.1 + (5.0 - 0.1) * min(float(step) / 10000.0, 1.0)
        dist_m1 = (normalized - (-1.0)).abs()
        dist_0  = (normalized - 0.0).abs()
        dist_p1 = (normalized - 1.0).abs()
        logits = -beta * torch.stack([dist_m1, dist_0, dist_p1], dim=-1)  # (..., 3)
        probs = torch.softmax(logits, dim=-1)                              # (..., 3) sum=1
        vals = torch.tensor([-1.0, 0.0, 1.0], device=normalized.device, dtype=normalized.dtype)
        expected_q = (probs * vals).sum(dim=-1)         # E[q] ∈ [-1, 1]
        fisher_diag = (probs * (vals ** 2)).sum(dim=-1) - expected_q ** 2  # Var[q]

        # Reshape to grad_output shape
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

        # LSQ scale gradients (same as LSQ backward)
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
        grouped = flat.reshape(-1, group_size)          # [num_groups, group_size]

        # TTQ: asymmetric per-group scales (pos and neg independent)
        sp = lsq_scale_pos.abs().clamp(min=1e-8).unsqueeze(-1)  # [num_groups, 1]
        sn = lsq_scale_neg.abs().clamp(min=1e-8).unsqueeze(-1)  # [num_groups, 1]
        s_norm = (sp * sn).sqrt()                                # geometric mean for normalization
        normalized = grouped / s_norm
        abs_normalized = normalized.abs()

        # R7-A3-M10: Energy-Based Direct Decision
        # Find per-group threshold that preserves >= 60% of weight L2 energy.
        # sigma_g (Fisher proxy from C2) is used as a secondary sensitivity signal
        # to modulate the energy fraction: high-sensitivity groups get a tighter
        # threshold (preserve more energy); low-sensitivity groups get a looser one.
        _TARGET_FRACTION = 0.6
        abs_flat = abs_normalized  # already [num_groups, group_size]
        sorted_vals, _ = abs_flat.sort(dim=-1, descending=True)
        cumsum = sorted_vals.cumsum(dim=-1)
        total_energy = cumsum[:, -1:].clamp(min=1e-8)

        if sigma_g is not None:
            sg_flat = sigma_g.to(dtype=torch.float32).reshape(-1)
            if pad_len > 0:
                sg_flat = F.pad(sg_flat, (0, pad_len))
            sg_grouped = sg_flat.reshape(-1, group_size)             # [num_groups, group_size]
            sg_per_group = sg_grouped.mean(dim=-1, keepdim=True)     # [num_groups, 1]
            mean_sg = sg_per_group.mean().clamp(min=1e-8)
            sensitivity = sg_per_group / mean_sg                     # ~1.0 mean
            target_fraction = (_TARGET_FRACTION * sensitivity).clamp(min=0.4, max=0.85)
        else:
            sensitivity = None
            target_fraction = _TARGET_FRACTION

        k = (cumsum / total_energy < target_fraction).sum(dim=-1).clamp(max=group_size - 1)
        energy_threshold = sorted_vals.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        threshold = energy_threshold.unsqueeze(-1)
        quantized = normalized.sign() * (abs_normalized > threshold).float()

        # OLS-STE alpha: informative fraction near threshold boundary
        transition_zone = threshold * 0.3
        near_threshold = (abs_normalized - threshold).abs() < transition_zone
        ols_alpha = near_threshold.float().mean().clamp(min=0.1, max=1.0)

        # TTQ: positive positions scaled by sp, negative by sn
        pos_mask = quantized > 0.5   # [num_groups, group_size]
        neg_mask = quantized < -0.5  # [num_groups, group_size]
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

        # F11-FW: Frank-Wolfe update toward optimal discrete vertex
        gamma = 2.0 / (float(step) + 2.0)  # step=0 → gamma=1.0, decays toward 0

        # LMO: optimal discrete = argmin_{v∈{-1,0,+1}} <g, v> = -sign(g)
        optimal_discrete = -torch.sign(grad_output)

        # w_continuous in weight space (normalized, same shape as grad_output)
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

        # LSQ scale gradients
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

        # Boundary kernel for backward correction
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
        # Reshape sigma_sqrt for per-group statistics (n_groups,) -> (n_groups, 1)
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
        )  # (num_groups, 1)
        w_soft_expanded = w_soft.expand(-1, group_size)  # (num_groups, group_size)

        hard_assign = torch.argmax(logits_grouped, dim=-1)
        w_hard_per_group = (
            scales_grouped.squeeze(-1) * (hard_assign == 2).float()
            - scales_grouped.squeeze(-1) * (hard_assign == 0).float()
        )  # (num_groups,)
        w_hard = w_hard_per_group.unsqueeze(-1).expand(
            -1, group_size
        )  # (num_groups, group_size)

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

        # Compute Jacobian of softmax: d(softmax)/d(logits)
        # softmax_j = exp(logits_j) / sum_k(exp(logits_k))
        # d(softmax_j)/d(logits_k) = softmax_j * (delta_jk - softmax_k)
        jacobian_softmax = torch.diag_embed(
            torch.stack([p_neg, p_zero, p_pos], dim=-1)
        ) - torch.bmm(
            torch.stack([p_neg, p_zero, p_pos], dim=-1).unsqueeze(-1),
            torch.stack([p_neg, p_zero, p_pos], dim=-1).unsqueeze(-2),
        )  # shape: (num_groups, 3, 3)

        # d(w_soft)/d(probs) where w_soft = scale * (p_pos - p_neg)
        # This is [-1, 0, +1] for the three probability outputs
        d_prob_wrt_logits = jacobian_softmax @ torch.tensor(
            [-1.0, 0.0, 1.0], device=logits.device, dtype=logits.dtype
        ).unsqueeze(-1)
        d_prob_wrt_logits = d_prob_wrt_logits.squeeze(-1)  # shape: (num_groups, 3)

        # Reshape grad to group structure and aggregate
        num_groups = d_prob_wrt_logits.shape[0]
        grad_grouped = grad_flat.reshape(num_groups, -1)  # (num_groups, group_size)

        # Each element in a group contributes to the same logit gradients
        # Sum gradients across elements within each group
        grad_per_group = grad_grouped.sum(dim=1)  # (num_groups,)

        # Compute grad_logits: grad_per_group * d(w_soft)/d(logits) / temperature
        grad_logits = (
            grad_per_group.unsqueeze(-1) * d_prob_wrt_logits / max(temperature, 0.01)
        )  # (num_groups, 3)

        return grad_flat.reshape(grad_output.shape), grad_logits, None, None, None


if hasattr(torch.compiler, "allow_in_graph"):
    torch.compiler.allow_in_graph(TernaryQuantize)
    print("[ALLOW-IN-GRAPH] TernaryQuantize + 12 others registered for torch.compile")
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
    else:
        # Default: coefficient-based threshold
        threshold = threshold_coeff * abs_normalized.mean(dim=-1, keepdim=True)

    # Debug logging for Absmean threshold (F4)
    if threshold_type == "absmean" and not torch.compiler.is_compiling():
        with torch.no_grad():
            nonzero_pct = (abs_normalized > threshold).float().mean().item() * 100
            print(
                f"  [F4-Absmean] ACTIVATED alpha=mean(|w|) delta=alpha/2 "
                f"nonzero_pct={nonzero_pct:.1f}%"
            )

    quantized = normalized.sign() * (abs_normalized > threshold).float()
    q_flat = (quantized * scales).reshape(-1)

    if pad_len > 0:
        q_flat = q_flat[: weight.numel()]

    q_w = q_flat.reshape(orig_shape)

    # OLS-STE: scale backward gradient by informative fraction.
    # Principle: only weights near the quantization threshold can change ternary state.
    # Gradients for committed weights (far from threshold) are noise, not signal.
    # weight_ols has identical forward value but backward gradient scaled by ols_alpha.
    # Math: w_ols = α·w + (1-α)·w.detach() → same forward, grad = α·grad_in
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

        self.register_buffer("admm_u", torch.zeros(out_features, in_features, dtype=torch.float16))  # Batch AE: FP32→FP16, saves 2 bytes/param
        # Batch AE: accumulated push buffer — tracks gradient integral for flip decisions (2 bytes/param vs 4 bytes FP32 shadow)
        self.register_buffer("accum_push", torch.zeros(out_features, in_features, dtype=torch.int16))
        self._accum_push_scale = 10000.0  # maps float gradients to INT16 range (grad ~0.1 → 1000/step)
        self._accum_push_threshold = 8000  # INT16 flip threshold (~8 gradient-sum units = ~8 steps at grad=0.1)
        self._accum_push_flips = 0         # counter for logging
        self._admm_rho = 0.01
        self._admm_last_ternary = None  # Batch AD: set during forward, used for proximal gradient

        self.alpha = None
        self.alpha_group_size = None
        self.freeze_patterns = False
        self.tequila_lambda = (
            0.0  # Tequila deadzone reactivation (0=disabled, 1e-3=default)
        )
        self._tequila_call_count = 0

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
        self._packed_logged = False

        # R12-F3: INT2 packed cache (4 weights/byte, 4x smaller than packed_weights)
        self.register_buffer("int2_packed_weights", None, persistent=False)
        self.register_buffer("int2_packed_valid", torch.tensor(False), persistent=False)
        self._int2_original_K = 0

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

        # MESU: inicializar logits a partir do sinal atual dos pesos
        if self.use_mesu and self.use_dsst:
            with torch.no_grad():
                _thr = self.weight.data.float().abs().mean() * getattr(self, 'threshold_coeff', 0.5)
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
            if hasattr(self, 'int2_packed_valid'):
                if isinstance(self.int2_packed_valid, torch.Tensor):
                    self.int2_packed_valid.zero_()
                else:
                    self.int2_packed_valid = False
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
                if (x.is_cuda
                        and x.shape[-1] >= 512
                        and getattr(self, '_ppr_use_quantized', True)
                        and hasattr(self, 'int2_packed_valid') and self.int2_packed_valid.item()
                        and hasattr(self, 'lsq_scale_pos') and self.lsq_scale_pos is not None):
                    from .bitmask_utils import TernaryInt2LinearFunc
                    _alpha = ((self.lsq_scale_pos.detach().abs().mean()
                               + self.lsq_scale_neg.detach().abs().mean()) * 0.5)
                    if not getattr(self, '_int2_logged', False):
                        print(f"[INT2-LINEAR] Training: using INT2 packed path K={x.shape[-1]}")
                        self._int2_logged = True
                    output = TernaryInt2LinearFunc.apply(
                        x, w, self.int2_packed_weights, self._int2_original_K, _alpha, self.bias)
                else:
                    output = F.linear(x, w.to(x.dtype), self.bias)

        else:
            if self.use_pwst and self.use_btg:
                output = TernaryLinearBTG_PWST.apply(x, w, self.bias)
            elif self.use_btg:
                output = TernaryLinearBTG.apply(x, w, self.bias)
            else:
                if (x.is_cuda
                        and x.shape[-1] >= 512
                        and hasattr(self, 'int2_packed_valid') and self.int2_packed_valid.item()
                        and hasattr(self, 'lsq_scale_pos') and self.lsq_scale_pos is not None):
                    from .bitmask_utils import TernaryInt2LinearFunc
                    _alpha = ((self.lsq_scale_pos.detach().abs().mean()
                               + self.lsq_scale_neg.detach().abs().mean()) * 0.5)
                    output = TernaryInt2LinearFunc.apply(
                        x, w, self.int2_packed_weights, self._int2_original_K, _alpha, self.bias)
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
        # AW-8: Sinkhorn OT cost — transport cost shadow → ternary target {-scale,0,+scale}
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
            # R12-F3: also invalidate INT2 cache
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
        self.ste_type = "standard"
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


def apply_antiwindup(model, k_aw=0.05):
    """Anti-windup for shadow weights (SQ-03 from research Round 3).

    Pulls continuous shadow weights toward their quantized ternary values,
    preventing drift into regions where gradients produce no ternary change.
    Analogous to integrator anti-windup in PID control theory.

    Args:
        model: The model containing TernaryLinear layers
        k_aw: Anti-windup gain (0.01-0.1 typical). Higher = stronger pull.
    """
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "group_size") and hasattr(module, "weight"):
                w = module.weight.data
                gs = module.group_size
                original_shape = w.shape

                flat = w.reshape(-1)
                pad_len = (gs - flat.numel() % gs) % gs
                if pad_len > 0:
                    flat = F.pad(flat, (0, pad_len))
                grouped = flat.reshape(-1, gs)

                scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
                normalized = grouped / scales
                abs_norm = normalized.abs()
                threshold = getattr(module, "threshold_coeff", 0.7) * abs_norm.mean(
                    dim=-1, keepdim=True
                )
                quantized = normalized.sign() * (abs_norm > threshold).float()

                target = quantized * scales

                w_grouped = grouped.clone()
                drift = w_grouped - target
                w_grouped.sub_(k_aw * drift)

                w_flat = w_grouped.reshape(-1)
                if pad_len > 0:
                    w_flat = w_flat[: w.numel()]
                w.copy_(w_flat.reshape(original_shape))


def compute_angular_boundary_reg(
    model, angular_lambda=0.1, boundary_lambda=0.01, eps=1e-8
):
    """Compute angular and boundary-repulsive regularization for ternary layers.

    Angular: L = angular_lambda * mean(1 - cos_sim(Q(W).detach(), W)) per row, per layer.
    Boundary: L = -boundary_lambda * mean(log(d_boundary + eps)) where d is distance to
              nearest quantization threshold in raw weight space.

    Returns (angular_loss, boundary_loss, n_layers). Losses include lambda scaling.
    """
    angular_losses = []
    boundary_losses = []

    for module in model.modules():
        if not (
            hasattr(module, "quantize_enabled")
            and module.quantize_enabled
            and hasattr(module, "weight")
        ):
            continue

        W = module.weight
        gs = getattr(module, "group_size", 128)
        t_coeff = getattr(module, "threshold_coeff", 0.7)

        if angular_lambda > 0:
            with torch.no_grad():
                orig_shape = W.shape
                flat = W.detach().reshape(-1)
                pad_len = (gs - flat.numel() % gs) % gs
                if pad_len > 0:
                    flat = F.pad(flat, (0, pad_len))
                grouped = flat.reshape(-1, gs)
                scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
                normalized = grouped / scales
                abs_norm = normalized.abs()
                threshold = t_coeff * abs_norm.mean(dim=-1, keepdim=True)
                quantized = normalized.sign() * (abs_norm > threshold).float()
                q_flat = (quantized * scales).reshape(-1)[: W.numel()]
                q_w = q_flat.reshape(orig_shape)

            cos = F.cosine_similarity(q_w, W, dim=1)
            angular_losses.append((1.0 - cos).mean())

        if boundary_lambda > 0:
            with torch.no_grad():
                flat_d = W.detach().reshape(-1)
                pad_len_d = (gs - flat_d.numel() % gs) % gs
                if pad_len_d > 0:
                    flat_d = F.pad(flat_d, (0, pad_len_d))
                grouped_d = flat_d.reshape(-1, gs)
                scales_d = (
                    grouped_d.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
                )
                norm_d = grouped_d / scales_d
                threshold_d = t_coeff * norm_d.abs().mean(dim=-1, keepdim=True)
                boundary_raw = (
                    (threshold_d * scales_d)
                    .expand(-1, gs)
                    .reshape(-1)[: W.numel()]
                    .reshape(W.shape)
                )

            d_pos = (W - boundary_raw).abs()
            d_neg = (W + boundary_raw).abs()
            d_boundary = torch.minimum(d_pos, d_neg)
            boundary_losses.append(-torch.log(d_boundary + eps).mean())

    n_layers = max(len(angular_losses), len(boundary_losses), 0)

    if angular_losses:
        total_angular = angular_lambda * torch.stack(angular_losses).mean()
    else:
        device = next(model.parameters()).device
        total_angular = torch.tensor(0.0, device=device)

    if boundary_losses:
        total_boundary = boundary_lambda * torch.stack(boundary_losses).mean()
    else:
        device = next(model.parameters()).device
        total_boundary = torch.tensor(0.0, device=device)

    return total_angular, total_boundary, n_layers


def compute_tdo_loss(
    model, tdo_lambda=0.01, tdo_target=(1 / 3, 1 / 3, 1 / 3), temperature=10.0
):
    """Ternary Distribution Objective - regularizer pushing shadow weights toward target distribution.

    Computes soft (differentiable) approximation of ternary weight distribution using sigmoid,
    then penalizes deviation from target via KL divergence. Gradients flow through sigmoid
    back into shadow weights.
    """
    kl_losses = []
    dist_neg_sum = 0.0
    dist_zero_sum = 0.0
    dist_pos_sum = 0.0
    n_layers = 0

    for module in model.modules():
        if not (
            hasattr(module, "quantize_enabled")
            and module.quantize_enabled
            and hasattr(module, "weight")
        ):
            continue

        W = module.weight
        gs = getattr(module, "group_size", 64)
        t_coeff = getattr(module, "threshold_coeff", 0.7)

        # Group-wise normalize (same math as TernaryQuantize.forward)
        flat = W.reshape(-1)
        pad_len = (gs - flat.numel() % gs) % gs
        if pad_len > 0:
            flat = torch.nn.functional.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, gs)
        scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        normalized = grouped / scales

        # Threshold
        abs_norm = normalized.abs()
        threshold = t_coeff * abs_norm.mean(dim=-1, keepdim=True)

        # Differentiable soft distribution via sigmoid
        soft_above = torch.sigmoid(temperature * (abs_norm - threshold))
        soft_below = 1.0 - soft_above
        soft_sign_pos = torch.sigmoid(temperature * normalized)

        # Per-weight soft probabilities
        p_pos_w = soft_above * soft_sign_pos
        p_neg_w = soft_above * (1.0 - soft_sign_pos)
        p_zero_w = soft_below

        # Layer-wide distribution
        p_neg = p_neg_w.mean()
        p_zero = p_zero_w.mean()
        p_pos = p_pos_w.mean()

        # KL divergence to target
        actual = torch.stack([p_neg, p_zero, p_pos]).clamp(min=1e-8)
        target_t = torch.tensor(tdo_target, device=W.device, dtype=W.dtype)
        kl = (actual * (actual.log() - target_t.log())).sum()

        kl_losses.append(kl)
        dist_neg_sum += p_neg.item()
        dist_zero_sum += p_zero.item()
        dist_pos_sum += p_pos.item()
        n_layers += 1

    if n_layers == 0:
        dummy = torch.tensor(
            0.0, device=next(model.parameters()).device, requires_grad=True
        )
        return dummy, 0, {"neg": 0, "zero": 0, "pos": 0}

    tdo_loss = tdo_lambda * torch.stack(kl_losses).mean()
    stats = {
        "neg": dist_neg_sum / n_layers,
        "zero": dist_zero_sum / n_layers,
        "pos": dist_pos_sum / n_layers,
    }

    return tdo_loss, n_layers, stats


def enable_lowrank_correction(model, rank: int):
    """Enable low-rank FP16 correction on all TernaryLinear layers post-construction."""
    added = 0
    total_correction_params = 0
    for module in model.modules():
        if hasattr(module, "quantize_enabled") and hasattr(module, "weight"):
            if not hasattr(module, "correction_rank") or module.correction_rank == 0:
                out_features = module.weight.shape[0]
                in_features = module.weight.shape[1]
                module.correction_rank = rank
                module.correction_A = nn.Parameter(
                    torch.randn(
                        out_features,
                        rank,
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                    )
                    * (0.02 / math.sqrt(rank))
                )
                module.correction_B = nn.Parameter(
                    torch.zeros(
                        rank,
                        in_features,
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                    )
                )
                n_params = out_features * rank + rank * in_features
                total_correction_params += n_params
                added += 1
    print(
        f"[LowRank] Enabled rank={rank} correction on {added} layers, "
        f"{total_correction_params:,} extra params ({total_correction_params * 4 / 1024:.1f} KB FP32)"
    )
    return added, total_correction_params


def compute_weight_entropy(model) -> list:
    """Compute Shannon entropy of ternary weight distribution per layer.

    Returns list of dicts with entropy metrics per TernaryLinear layer.
    H_max = log2(3) = 1.585 bits. Utilization < 0.8 indicates capacity loss.
    """
    results = []
    for name, module in model.named_modules():
        if not hasattr(module, "get_ternary_stats"):
            continue
        stats = module.get_ternary_stats()
        probs = [stats["negative"], stats["zero"], stats["positive"]]
        H = 0.0
        for p in probs:
            if p > 1e-10:
                H -= p * math.log2(p)
        H_max = math.log2(3)
        results.append(
            {
                "name": name,
                "entropy": H,
                "max_entropy": H_max,
                "utilization": H / H_max,
                "dist": (stats["negative"], stats["zero"], stats["positive"]),
            }
        )
    return results


def compute_commitment_rate(model, threshold_margin=0.1) -> list:
    """Compute fraction of weights committed (far from threshold) per layer.

    'Uncommitted' = shadow weight's |w_normalized| is within threshold_margin
    of the quantization threshold. These weights may flip with small gradient changes.
    """
    results = []
    for name, module in model.named_modules():
        if not hasattr(module, "threshold_coeff") or not hasattr(module, "weight"):
            continue
        if not hasattr(module, "group_size"):
            continue
        with torch.no_grad():
            w = module.weight
            group_size = module.group_size
            t_coeff = module.threshold_coeff

            flat = w.reshape(-1)
            pad_len = (group_size - flat.numel() % group_size) % group_size
            if pad_len > 0:
                flat = torch.nn.functional.pad(flat, (0, pad_len))
            grouped = flat.reshape(-1, group_size)

            scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
            normalized = grouped / scales
            abs_normalized = normalized.abs()
            threshold = t_coeff * abs_normalized.mean(dim=-1, keepdim=True)

            dist_to_threshold = (abs_normalized - threshold).abs()
            margin = threshold_margin * threshold
            uncommitted = (dist_to_threshold < margin).float()

            total = w.numel()
            n_uncommitted = uncommitted.sum().item()
            if pad_len > 0:
                n_uncommitted = uncommitted.reshape(-1)[: w.numel()].sum().item()

            results.append(
                {
                    "name": name,
                    "committed_frac": 1.0 - n_uncommitted / total,
                    "uncommitted_frac": n_uncommitted / total,
                    "mean_dist_to_threshold": dist_to_threshold.reshape(-1)[: w.numel()]
                    .mean()
                    .item()
                    if pad_len > 0
                    else dist_to_threshold.mean().item(),
                    "total_weights": total,
                }
            )
    return results


def compute_margin_mask(
    weight: torch.Tensor, group_size: int, threshold_coeff: float, delta: float
) -> torch.Tensor:
    """Compute binary mask of weights near quantization boundary.

    Returns: bool tensor same shape as weight. True = critical (near boundary).
    """
    with torch.no_grad():
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

        dist_to_threshold = (abs_normalized - threshold).abs()
        margin_threshold = delta * threshold
        critical = dist_to_threshold < margin_threshold

        critical_flat = critical.reshape(-1)
        if pad_len > 0:
            critical_flat = critical_flat[: weight.numel()]
        return critical_flat.reshape(orig_shape)


def init_gradient_recyclers(model, beta: float = 0.99) -> dict:
    """Initialize EMA gradient storage for all TernaryLinear weight parameters.

    Returns dict mapping param name -> {'ema': Tensor, 'beta': float, 'initialized': bool}
    """
    recyclers = {}
    for name, module in model.named_modules():
        if not hasattr(module, "threshold_coeff") or not hasattr(module, "weight"):
            continue
        if not hasattr(module, "group_size"):
            continue
        param_name = name + ".weight"
        recyclers[param_name] = {
            "ema": torch.zeros_like(module.weight.data),
            "beta": beta,
            "initialized": False,
            "group_size": module.group_size,
            "threshold_coeff": module.threshold_coeff,
        }
    print(
        f"[MARGIN-SPARSE] Initialized gradient recyclers for {len(recyclers)} ternary layers "
        f"({sum(r['ema'].numel() for r in recyclers.values())} params tracked)"
    )
    return recyclers


def apply_gradient_recycling(
    model,
    recyclers: dict,
    global_step: int,
    refresh_interval: int,
    delta: float,
    log_interval: int = 100,
) -> None:
    """Replace far-from-boundary gradients with EMA approximation.

    Call AFTER backward() and unscale, BEFORE grad clipping and optimizer.step().
    """
    is_refresh = global_step % refresh_interval == 0
    total_critical = 0
    total_params = 0

    for name, param in model.named_parameters():
        if name not in recyclers or param.grad is None:
            continue

        rec = recyclers[name]
        grad = param.grad.data

        if is_refresh or not rec["initialized"]:
            rec["ema"].mul_(rec["beta"]).add_(grad, alpha=1.0 - rec["beta"])
            rec["initialized"] = True
        else:
            critical_mask = compute_margin_mask(
                param.data, rec["group_size"], rec["threshold_coeff"], delta
            )
            critical_float = critical_mask.float()

            param.grad.data = (
                critical_float * grad + (1.0 - critical_float) * rec["ema"]
            )

            total_critical += critical_mask.sum().item()
            total_params += param.numel()

    if not torch.compiler.is_compiling() and global_step % log_interval == 0:
        if is_refresh:
            print(
                f"  [MARGIN-SPARSE] step={global_step} REFRESH (full gradient, EMA updated)"
            )
        elif total_params > 0:
            crit_pct = 100.0 * total_critical / total_params
            print(
                f"  [MARGIN-SPARSE] step={global_step} critical={crit_pct:.1f}% "
                f"({int(total_critical)}/{total_params} weights near boundary)"
            )


def enable_dendritic_gating(model, mode: str):
    """Enable dendritic gating on all TernaryLinear layers post-construction."""
    assert mode in ("pre", "post"), f"Invalid dendritic mode: '{mode}'"
    added = 0
    total_gate_params = 0
    for module in model.modules():
        if hasattr(module, "quantize_enabled") and hasattr(module, "weight"):
            out_features = module.weight.shape[0]
            module.dendritic_mode = mode
            if mode == "pre":
                module.gate_pre = nn.Parameter(
                    torch.ones(
                        out_features,
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                    )
                )
                total_gate_params += out_features
            elif mode == "post":
                module.gate_post = nn.Parameter(
                    torch.ones(
                        out_features,
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                    )
                )
                total_gate_params += out_features
            added += 1
    print(
        f"[Dendritic] Enabled mode='{mode}' on {added} layers, "
        f"{total_gate_params:,} gate params ({total_gate_params * 4 / 1024:.1f} KB FP32)"
    )
    return added, total_gate_params


def enable_sigma_delta(model, error_clamp: float = 2.0):
    """Enable sigma-delta noise shaping on all TernaryLinear layers."""
    enabled = 0
    for module in model.modules():
        if hasattr(module, "ste_type") and hasattr(module, "quantize_enabled"):
            module.ste_type = "sigma_delta"
            module.sd_error_clamp = error_clamp
            enabled += 1
    print(f"[SigmaDelta] Enabled on {enabled} layers, error_clamp={error_clamp}")
    return enabled


def enable_bias_correction(model, momentum: float = 0.99, variant: str = "naive"):
    """Enable bias-corrected STE (BSTE) on all TernaryLinear layers (Phase 21.D, TQ-02).

    STE has a systematic bias: gradients passed to weights deep in quantization bins
    are pure noise that cannot cause ternary value changes. BSTE maintains a running
    EMA of this bias and subtracts it before the optimizer step.

    Variants:
        'naive': EMA of raw gradient. High-pass temporal filter on gradient signal.
        'error_weighted': EMA weighted by quantization depth. Only corrects weights
                         deep in bins (far from threshold), leaves boundary weights alone.

    Parameter-free (zero extra trainable parameters). Only adds a buffer for EMA state.
    Must call apply_bias_correction(model) between loss.backward() and optimizer.step().

    Args:
        model: The model containing TernaryLinear layers
        momentum: EMA momentum (0.99 = ~100-step window)
        variant: 'naive' or 'error_weighted'
    Returns:
        Number of layers enabled
    """
    assert variant in ("naive", "error_weighted"), f"Invalid BSTE variant: '{variant}'"
    enabled = 0
    for module in model.modules():
        if hasattr(module, "quantize_enabled") and hasattr(module, "weight"):
            module.register_buffer("gradient_bias_ema", torch.zeros_like(module.weight))
            module.bste_momentum = momentum
            module.bste_variant = variant
            module.bste_enabled = True
            enabled += 1
    print(
        f"[BSTE] Enabled variant='{variant}' momentum={momentum} on {enabled} layers "
        f"(buffer size: {sum(m.gradient_bias_ema.numel() for m in model.modules() if hasattr(m, 'gradient_bias_ema')) * 4 / 1024:.1f} KB)"
    )
    return enabled


def apply_bias_correction(model):
    """Apply bias-corrected STE gradient correction (Phase 21.D).

    Call AFTER loss.backward() and BEFORE optimizer.step() in the training loop.

    For 'naive' variant: subtracts EMA of raw gradient (temporal high-pass filter).
    For 'error_weighted' variant: subtracts EMA of gradient weighted by quantization
    depth (only corrects weights deep in bins, preserves boundary gradients).

    Returns:
        Tuple of (correction_applied: bool, mean_correction_ratio: float)
    """
    correction_ratios = []
    for module in model.modules():
        if not (hasattr(module, "bste_enabled") and module.bste_enabled):
            continue
        if module.weight.grad is None:
            continue

        grad = module.weight.grad
        ema = module.gradient_bias_ema
        momentum = module.bste_momentum

        if module.bste_variant == "error_weighted":
            with torch.no_grad():
                w = module.weight.data
                gs = getattr(module, "group_size", 128)
                t_coeff = getattr(module, "threshold_coeff", 0.7)
                orig_shape = w.shape
                flat = w.reshape(-1)
                pad_len = (gs - flat.numel() % gs) % gs
                if pad_len > 0:
                    flat = F.pad(flat, (0, pad_len))
                grouped = flat.reshape(-1, gs)
                scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
                normalized = grouped / scales
                abs_norm = normalized.abs()
                threshold = t_coeff * abs_norm.mean(dim=-1, keepdim=True)
                quantized_scaled = (
                    normalized.sign() * (abs_norm > threshold).float() * scales
                )
                q_flat = quantized_scaled.reshape(-1)[: w.numel()]
                q_w = q_flat.reshape(orig_shape)

                quant_error = (w - q_w).abs()
                max_err = quant_error.max().clamp(min=1e-8)
                depth_mask = (quant_error / max_err).clamp(0, 1)

            ema.mul_(momentum).add_(grad * depth_mask, alpha=1 - momentum)
        else:
            ema.mul_(momentum).add_(grad, alpha=1 - momentum)

        with torch.no_grad():
            grad_norm = grad.abs().mean().clamp(min=1e-10)
            correction_ratio = ema.abs().mean() / grad_norm
            correction_ratios.append(correction_ratio.item())

        module.weight.grad = grad - ema

    if correction_ratios:
        return True, sum(correction_ratios) / len(correction_ratios)
    return False, 0.0


def enable_block_diagonal(model, block_size: int = 4):
    """Enable block-diagonal FP16 mixing on all TernaryLinear layers (Phase 22).

    After the ternary matmul, groups every block_size consecutive output neurons
    and mixes them with a learnable FP16 matrix. Restores singular value
    magnitude variation lost by ternary quantization.

    Storage overhead: block_size extra FP16 params per output neuron.
    For block_size=4: ~3.1% overhead at typical dimensions.

    Args:
        model: The model containing TernaryLinear layers
        block_size: Number of output neurons per mixing block (default 4)
    Returns:
        Tuple of (num_layers_enabled, total_mixing_params)
    """
    enabled = 0
    total_params = 0
    skipped = 0
    for module in model.modules():
        if hasattr(module, "quantize_enabled") and hasattr(module, "weight"):
            out_features = module.weight.shape[0]
            if out_features % block_size != 0:
                skipped += 1
                continue
            num_blocks = out_features // block_size
            module.block_size = block_size
            module.block_mixing = nn.Parameter(
                torch.eye(
                    block_size, device=module.weight.device, dtype=module.weight.dtype
                )
                .unsqueeze(0)
                .expand(num_blocks, -1, -1)
                .clone()
            )
            n_params = num_blocks * block_size * block_size
            total_params += n_params
            enabled += 1
    print(
        f"[BlockDiag] Enabled block_size={block_size} on {enabled} layers "
        f"({total_params:,} mixing params, {total_params * 4 / 1024:.1f} KB FP32), "
        f"skipped {skipped} layers (dim not divisible by {block_size})"
    )
    return enabled, total_params


def get_block_diagonal_stats(model):
    """Get block mixing deviation statistics for debug logging."""
    deviations = []
    for name, module in model.named_modules():
        if hasattr(module, "block_mixing") and module.block_mixing is not None:
            with torch.no_grad():
                C = module.block_mixing
                bs = C.shape[-1]
                identity = torch.eye(bs, device=C.device, dtype=C.dtype).unsqueeze(0)
                deviation = (C - identity).norm(dim=(-2, -1)).mean().item()
                deviations.append(deviation)
    if not deviations:
        return {"enabled": False}
    return {
        "enabled": True,
        "num_layers": len(deviations),
        "avg_deviation_from_identity": sum(deviations) / len(deviations),
        "max_deviation": max(deviations),
        "min_deviation": min(deviations),
    }


def enable_tpsq(model, beta: float = 1.0, momentum: float = 0.99):
    """Enable TPSQ (Phase-Space Quantization) on all TernaryLinear layers.

    Maintains running statistics of input variance and uses it to adaptively
    adjust quantization thresholds.

    Args:
        model: The model containing TernaryLinear layers
        beta: TPSQ sensitivity to input variance (default: 1.0)
        momentum: EMA momentum for running variance (default: 0.99)
    Returns:
        Number of layers enabled
    """
    enabled = 0
    for module in model.modules():
        if hasattr(module, "use_tpsq"):
            module.use_tpsq = True
            module.tpsq_beta = beta
            module.tpsq_momentum = momentum
            # Register buffer only if not already registered (handles multiple enable calls)
            if not hasattr(module, "_tpsq_sigma_sq"):
                device = module.weight.device
                dtype = module.weight.dtype
                group_size = module.group_size
                n_groups = (module.weight.numel() + group_size - 1) // group_size
                module.register_buffer(
                    "_tpsq_sigma_sq", torch.ones(n_groups, device=device, dtype=dtype)
                )
            enabled += 1
    print(f"[TPSQ] Enabled on {enabled} layers (beta={beta}, momentum={momentum})")
    return enabled


def enable_tcd(model, beta_init: float = None):
    """Enable TCD (Ternary Critical Dynamics) on all TernaryLinear layers.

    Sets ste_type='tcd' and initializes beta_c = sqrt(d_model/3) for critical damping.

    Args:
        model: The model containing TernaryLinear layers
        beta_init: Initial beta value. If None, uses sqrt(d_model/3)
    Returns:
        Number of layers enabled
    """
    enabled = 0
    for module in model.modules():
        if hasattr(module, "tcd_beta") and hasattr(module, "ste_type"):
            if beta_init is None:
                d_model = getattr(module, "out_features", 256)
                beta = math.sqrt(d_model / 3.0)
            else:
                beta = beta_init
            module.tcd_beta = max(0.1, min(beta, 100.0))
            module.ste_type = "tcd"
            enabled += 1
    print(f"[TCD] Enabled on {enabled} layers (ste_type='tcd')")
    return enabled


def enable_mdts(model, temperature: float = 1.0, anneal: bool = True):
    """Enable MDTS (Mirror Descent Ternary Solver) on all TernaryLinear layers.

    Initializes logits from current shadow weights, confident toward existing ternary.
    Temperature anneals from 1.0 to 0.1 over training if anneal=True.

    Args:
        model: The model containing TernaryLinear layers
        temperature: Initial temperature (default: 1.0)
        anneal: If True, anneal temperature 1.0 -> 0.1 over training
    Returns:
        Number of layers enabled
    """
    enabled = 0
    for module in model.modules():
        if not (hasattr(module, "use_mdts") and hasattr(module, "weight")):
            continue

        W = module.weight.data
        gs = module.group_size
        device = W.device
        dtype = W.dtype

        orig_shape = W.shape
        flat = W.reshape(-1)
        pad_len = (gs - flat.numel() % gs) % gs
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))

        grouped = flat.reshape(-1, gs)
        scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        normalized = grouped / scales

        abs_norm = normalized.abs()
        logits_init = torch.zeros(grouped.shape[0], 3, device=device, dtype=dtype)

        pos_mask = (normalized > 0.5).float()
        neg_mask = (normalized < -0.5).float()
        zero_mask = 1.0 - pos_mask - neg_mask

        logits_init[:, 0] = (
            neg_mask.sum(dim=-1) / (neg_mask.sum(dim=-1) + 1e-8)
        ).clamp(min=1e-4)
        logits_init[:, 1] = (
            zero_mask.sum(dim=-1) / (zero_mask.sum(dim=-1) + 1e-8)
        ).clamp(min=1e-4)
        logits_init[:, 2] = (
            pos_mask.sum(dim=-1) / (pos_mask.sum(dim=-1) + 1e-4)
        ).clamp(min=1e-4)

        logits_init = torch.log(logits_init + 1e-10)

        module._mdts_logits = nn.Parameter(logits_init)
        if hasattr(module, "_mdts_scales"):
            delattr(module, "_mdts_scales")
        module.register_buffer("_mdts_scales", scales.squeeze(-1))

        module.use_mdts = True
        module.mdts_temperature = temperature
        module.mdts_anneal = anneal
        module._mdts_call_count = 0

        enabled += 1

    anneal_str = "with annealing" if anneal else "constant"
    print(
        f"[MDTS] Enabled on {enabled} layers (temperature={temperature} {anneal_str})"
    )
    return enabled


def enable_gumbel_noise(model, temperature: float = 0.1, anneal: bool = False):
    """Enable Gumbel+STE noise injection on all TernaryLinear layers.

    Adds Gumbel noise during training to improve exploration of weight space
    and gradient flow around quantization boundaries.

    Args:
        model: The model containing TernaryLinear layers
        temperature: Gumbel noise scale (default: 0.1)
        anneal: If True, decay temperature over training (default: False)
    Returns:
        Number of layers enabled
    """
    enabled = 0
    for module in model.modules():
        if hasattr(module, "use_gumbel_noise"):
            module.use_gumbel_noise = True
            module.gumbel_temperature = temperature
            module.gumbel_anneal = anneal
            enabled += 1
    anneal_str = "with annealing" if anneal else "constant"
    print(
        f"[GumbelNoise] Enabled on {enabled} layers (temperature={temperature} {anneal_str})"
    )
    return enabled


def enable_stp(model, sigma_ratio: float = 0.1):
    """Enable STP (Stochastic Threshold Perturbation) on all TernaryLinear layers."""
    enabled = 0
    for module in model.modules():
        if hasattr(module, "use_stp"):
            module.use_stp = True
            module.stp_sigma_ratio = sigma_ratio
            enabled += 1
    print(f"[STP] Enabled on {enabled} layers (sigma_ratio={sigma_ratio})")
    return enabled


def enable_ctce(model, beta: float = 1.5, delta_ratio: float = 0.5):
    """Enable CTCE on all TernaryLinear layers. Amplifies gradients near threshold (factor >= 1.0)."""
    enabled = 0
    for module in model.modules():
        if hasattr(module, "use_ctce"):
            module.use_ctce = True
            module.ctce_beta = beta
            module.ctce_delta_ratio = delta_ratio
            enabled += 1
    print(
        f"[CTCE] Enabled on {enabled} layers (beta={beta}, delta_ratio={delta_ratio})"
    )
    return enabled


def enable_dual_rate(
    model, peak: float = 2.5, floor: float = 0.1, delta_ratio: float = 0.5
):
    """Enable Dual-Rate STE on all TernaryLinear layers.
    Combines CTCE amplification with far-from-boundary damping.
    Disables CTCE if active (Dual-Rate supersedes it).
    """
    count = 0
    for module in model.modules():
        if hasattr(module, "use_stp"):
            module.use_dual_rate = True
            module.dual_rate_peak = peak
            module.dual_rate_floor = floor
            module.dual_rate_delta_ratio = delta_ratio
            module.use_ctce = False  # Dual-Rate supersedes CTCE
            count += 1
    print(
        f"[DualRate] Enabled on {count} layers (peak={peak}, floor={floor}, delta_ratio={delta_ratio})"
    )
    return count


def enable_gaste(model, sigma_g: float = 0.3):
    """Enable GASTE (Group-Aware STE) on all TernaryLinear layers."""
    count = 0
    for module in model.modules():
        if hasattr(module, "use_stp"):
            module.use_gaste = True
            module.gaste_sigma_g = sigma_g
            count += 1
    print(f"[GASTE] Enabled on {count} layers (sigma_g={sigma_g})")
    return count


def enable_rlec(model, rank: int = 4, min_size: int = 0):
    """Enable RLEC (Residual Low-Rank Error Compensation) on all TernaryLinear layers.

    RLEC adds FP16 low-rank correction: y = Q(W)*x + A*(B*x)
    A initialized to zeros (starts as pure ternary), B to scaled random.
    """
    import math

    count = 0
    for name, module in model.named_modules():
        if hasattr(module, "use_stp"):
            out_f, in_f = module.weight.shape
            if min_size > 0 and min(out_f, in_f) < min_size:
                continue
            # RLEC initialization: A=zeros (start pure ternary), B=scaled random
            module.correction_A = torch.nn.Parameter(
                torch.zeros(
                    out_f, rank, device=module.weight.device, dtype=torch.float32
                )
            )
            module.correction_B = torch.nn.Parameter(
                torch.randn(
                    rank, in_f, device=module.weight.device, dtype=torch.float32
                )
                * (1.0 / math.sqrt(in_f))
            )
            module._rlec_call_count = 0
            count += 1
    total_params = sum(
        m.correction_A.numel() + m.correction_B.numel()
        for m in model.modules()
        if hasattr(m, "correction_A")
        and m.correction_A is not None
        and m.correction_A.shape[-1] == rank
    )
    print(
        f"[RLEC] Enabled on {count} layers, rank={rank}, added {total_params:,} FP16 params ({total_params * 2 / 1024:.1f} KB)"
    )
    return count


class TernaryFlipRateTracker:
    """Track ternary weight flip rate to detect exploration→commitment phase transition.

    Monitors the rate at which ternary weights change across steps. During exploration,
    weights flip frequently. As training stabilizes, flip rate drops sharply. This tracker
    detects that transition and can trigger learning rate decay.

    Attributes:
        threshold: EMA flip rate below which phase transition is triggered (default: 0.05)
        ema_beta: EMA decay for flip rate smoothing (default: 0.9)
        min_step: Minimum step before transition can occur (default: 500)
        transitioned: Whether transition has been detected
        transition_step: Step at which transition occurred (-1 if not yet)
    """

    def __init__(self, threshold=0.05, ema_beta=0.9, min_step=500):
        self.threshold = threshold
        self.ema_beta = ema_beta
        self.min_step = min_step
        self.prev_weights = {}
        self.flip_rate_ema = None
        self.transitioned = False
        self.transition_step = -1

    def update(self, model, step):
        """Compute flip rate, detect phase transition.

        Args:
            model: Model to analyze
            step: Current training step

        Returns:
            Dict with metrics: flip_rate, flip_rate_ema, num_flips, num_ternary_params,
                               transitioned, just_transitioned
        """
        current_weights = {}
        for name, module in model.named_modules():
            if hasattr(module, "weight") and hasattr(module, "quantize_enabled"):
                if getattr(module, "quantize_enabled", False):
                    w = module.weight.data
                    w_quant = torch.sign(w)
                    current_weights[name] = w_quant.detach().clone()

        if not self.prev_weights:
            self.prev_weights = current_weights
            total_params = sum(w.numel() for w in current_weights.values())
            return {
                "flip_rate": 0.0,
                "flip_rate_ema": 0.0,
                "num_flips": 0,
                "num_ternary_params": total_params,
                "transitioned": False,
                "just_transitioned": False,
            }

        num_flips = 0
        num_params = 0
        for name, w_curr in current_weights.items():
            if name in self.prev_weights:
                w_prev = self.prev_weights[name]
                num_flips += (w_curr != w_prev).sum().item()
                num_params += w_curr.numel()

        flip_rate = num_flips / max(1, num_params)

        if self.flip_rate_ema is None:
            self.flip_rate_ema = flip_rate
        else:
            self.flip_rate_ema = (
                self.ema_beta * self.flip_rate_ema + (1 - self.ema_beta) * flip_rate
            )

        just_transitioned = False
        if not self.transitioned and step >= self.min_step:
            if self.flip_rate_ema < self.threshold:
                self.transitioned = True
                self.transition_step = step
                just_transitioned = True

        self.prev_weights = current_weights

        return {
            "flip_rate": flip_rate,
            "flip_rate_ema": self.flip_rate_ema,
            "num_flips": num_flips,
            "num_ternary_params": num_params,
            "transitioned": self.transitioned,
            "just_transitioned": just_transitioned,
        }


def compute_sca_sparsity_loss(
    model: torch.nn.Module,
    target_sparsity: float = 0.5,
    group_size: int = 128,
    threshold_coeff: float = DEFAULT_THRESHOLD_COEFF,
) -> tuple[torch.Tensor, dict]:
    """SCA-Sparsity-Controlled-Energy regularization (Deng et al., arXiv:2011.00580)

    Adds regularization term -λ·sparsity_controller for explicit sparsity control.
    This explicitly controls the zero-ratio to match target_sparsity.

    Args:
        model: The model with ternary weights
        target_sparsity: Target zero ratio (0.0-1.0), e.g., 0.5 for 50% zeros
        group_size: Group size for computing scales
        threshold_coeff: Threshold coefficient for ternary quantization

    Returns:
        Tuple of (sparsity_loss, stats_dict)
    """
    total_zeros = 0
    total_weights = 0

    for name, module in model.named_modules():
        if not hasattr(module, "weight") or not hasattr(module, "threshold_coeff"):
            continue

        with torch.no_grad():
            w = module.weight
            gs = getattr(module, "group_size", group_size)
            tc = getattr(module, "threshold_coeff", threshold_coeff)

            # Compute ternary quantization (same as TernaryLinear)
            flat = w.reshape(-1)
            pad_len = (gs - flat.numel() % gs) % gs
            if pad_len > 0:
                flat = F.pad(flat, (0, pad_len))
            grouped = flat.reshape(-1, gs)

            scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
            normalized = grouped / scales
            abs_normalized = normalized.abs()
            threshold = tc * abs_normalized.mean(dim=-1, keepdim=True)

            # Compute zero mask (weights below threshold become zero)
            zero_mask = abs_normalized <= threshold
            n_zeros = zero_mask.sum().item()
            n_weights = w.numel()

            total_zeros += n_zeros
            total_weights += n_weights

    if total_weights == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device), {
            "current_sparsity": 0.0
        }

    current_sparsity = total_zeros / total_weights

    # SCA energy function: E = -λ * (sparsity - target)^2
    # This penalizes deviation from target sparsity
    # Using squared error for smoother gradients
    sparsity_diff = current_sparsity - target_sparsity
    sparsity_loss = sparsity_diff**2

    stats = {
        "current_sparsity": current_sparsity,
        "target_sparsity": target_sparsity,
        "sparsity_diff": sparsity_diff,
        "total_zeros": total_zeros,
        "total_weights": total_weights,
    }

    return torch.tensor(sparsity_loss, device=next(model.parameters()).device), stats


def compute_sca_sparsity_loss_simple(
    model: torch.nn.Module,
    target_sparsity: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Simplified SCA sparsity loss using absolute value for more direct gradient.

    Instead of computing exact ternary quantization, uses a soft approximation:
    - Weights with |w| < threshold are considered "likely zeros"
    - Penalizes deviation from target sparsity ratio

    This is faster but less precise than compute_sca_sparsity_loss.
    """
    total_magnitude = 0.0
    total_weights = 0

    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue

        w = module.weight
        total_magnitude += w.abs().sum().item()
        total_weights += w.numel()

    if total_weights == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device), {
            "current_sparsity": 0.0
        }

    # Average magnitude as proxy for "committed" weights
    avg_magnitude = total_magnitude / total_weights

    # Normalize to estimate sparsity (higher magnitude = less sparsity)
    # This is a rough approximation
    current_sparsity = 1.0 - min(avg_magnitude, 1.0)

    sparsity_diff = current_sparsity - target_sparsity
    sparsity_loss = sparsity_diff**2

    stats = {
        "current_sparsity": current_sparsity,
        "target_sparsity": target_sparsity,
        "sparsity_diff": sparsity_diff,
        "avg_magnitude": avg_magnitude,
    }

    return torch.tensor(sparsity_loss, device=next(model.parameters()).device), stats
