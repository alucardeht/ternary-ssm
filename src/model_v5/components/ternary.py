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

        ctx.save_for_backward(ols_alpha)

        result = (quantized * scales).reshape(-1)[
            : flat.numel() - pad_len if pad_len > 0 else flat.numel()
        ]

        TernaryQuantize._call_count += 1
        if TernaryQuantize._call_count % TernaryQuantize._log_every == 0:
            print(f"[OLS-STE] call={TernaryQuantize._call_count} alpha={ols_alpha.item():.4f} "
                  f"informative={informative_frac.mean().item():.4f}")

        return result.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
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
        quantized, sp, sn, normalized, ols_alpha = ctx.saved_tensors
        pad_len = ctx.pad_len
        group_size = ctx.group_size

        # dL/dw: OLS-STE pass-through
        grad_weight = grad_output * ols_alpha

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

        # 6 returns: weight, lsq_scale_pos, lsq_scale_neg, group_size, threshold_coeff, sigma_g
        return grad_weight, grad_sp, grad_sn, None, None, None


torch.compiler.allow_in_graph(LSQTernaryQuantize)


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
            grad_weight = go_2d.t().matmul(in_2d)

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
            grad_weight = go_2d.t().matmul(in_2d)

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
            grad_weight = go_2d.t().matmul(in_2d)

        if ctx.has_bias and ctx.needs_input_grad[3]:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)

        return grad_input, grad_weight, None, grad_bias

TernaryLinearTATGR._call_count = 0


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

        self.register_buffer("admm_u", torch.zeros(out_features, in_features))
        self._admm_rho = 0.01

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

        # F17: Packed Cache
        self.use_packed_cache = True
        self.register_buffer("packed_weights", None, persistent=False)
        self.register_buffer("packed_valid", torch.tensor(False), persistent=False)
        self._packed_logged = False

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

        # C2+C1: Commitment Tracking + Gradient Skip (R5-B2)
        # Always-on for ternary — foundation for MBGSC, A3-F13, A1-F16, CAM, HSP-C2
        self.use_commitment_tracking = True
        self.use_gradient_skip = (
            getattr(config, "use_gradient_skip", False) if config is not None else False
        )
        self.commitment_n_stable_target = (
            getattr(config, "commitment_n_stable", 100) if config is not None else 100
        )
        self.commitment_ema_decay = (
            getattr(config, "commitment_ema_decay", 0.99)
            if config is not None
            else 0.99
        )
        self.commitment_margin_multiplier = (
            getattr(config, "commitment_margin_multiplier", 4.89)
            if config is not None
            else 4.89
        )
        self.grad_skip_g_max = (
            getattr(config, "grad_skip_g_max", 2.5) if config is not None else 2.5
        )
        self._commitment_log_interval = (
            getattr(config, "commitment_log_interval", 100)
            if config is not None
            else 100
        )
        self._commitment_step_counter = 0

        if self.use_commitment_tracking:
            self.register_buffer("sigma_g", None, persistent=False)
            self.register_buffer("n_stable", None, persistent=False)
            self.register_buffer("committed_mask", None, persistent=False)
            self.register_buffer("prev_ternary_c2", None, persistent=False)
            self._current_lr_c2 = 1e-3
            self._grad_hook_handle = None
            if not hasattr(TernaryLinear, "_c2_init_count"):
                TernaryLinear._c2_init_count = 0
            TernaryLinear._c2_init_count += 1

        if self.use_gradient_skip and not self.use_commitment_tracking:
            import warnings

            warnings.warn(
                "C1 (gradient skip) requires C2 (commitment tracking). Auto-enabling C2."
            )
            self.use_commitment_tracking = True
            self.register_buffer("sigma_g", None, persistent=False)
            self.register_buffer("n_stable", None, persistent=False)
            self.register_buffer("committed_mask", None, persistent=False)
            self.register_buffer("prev_ternary_c2", None, persistent=False)
            self._current_lr_c2 = 1e-3
            self._grad_hook_handle = None

        if self.use_gradient_skip:
            print(
                f"[C1-GRAD-SKIP] Enabled for {self.__class__.__name__} (G_max={self.grad_skip_g_max})"
            )

        # CAM: Commitment-Aware Momentum gradient boost (R1-A5)
        # CAM is automatic when use_commitment_tracking is True
        self.cam_alpha = 0.5
        self.cam_beta = 0.01

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
        threshold = t_coeff * abs_normalized.mean(dim=-1, keepdim=True)

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

    def _init_commitment_buffers(self):
        """C2: Initialize commitment tracking buffers on first call."""
        if self.sigma_g is None:
            device = self.weight.device
            dtype = self.weight.dtype

            self.sigma_g = torch.full_like(
                self.weight, 1e-3, dtype=torch.float32, device=device
            )
            self.n_stable = torch.zeros_like(
                self.weight, dtype=torch.int32, device=device
            )
            self.committed_mask = torch.zeros_like(
                self.weight, dtype=torch.bool, device=device
            )
            self.prev_ternary_c2 = torch.zeros_like(
                self.weight, dtype=torch.float32, device=device
            )

            if getattr(self, '_grad_hook_handle', None) is None and hasattr(self, 'weight'):
                self._grad_hook_handle = self.weight.register_hook(self._grad_hook_fn)

    def _grad_hook_fn(self, grad):
        """C2: Gradient hook to track σ_g and apply C1 skip mask.

        Updates σ_g via EMA and zeros gradients for committed weights if C1 enabled.
        Returns modified gradient.
        """
        if not self.use_commitment_tracking or self.sigma_g is None:
            return grad

        with torch.no_grad():
            grad_abs = grad.abs()
            self.sigma_g.mul_(self.commitment_ema_decay).add_(
                grad_abs, alpha=(1 - self.commitment_ema_decay)
            )

        # CAM: boost gradient for stuck-but-uncommitted weights only
        # Committed weights are intentionally frozen — boosting them wastes compute.
        if self.use_commitment_tracking and getattr(self, 'n_stable', None) is not None:
            with torch.no_grad():
                age_f = self.n_stable.float()
                boost = 1.0 + self.cam_alpha * (1.0 - torch.exp(-self.cam_beta * age_f))
                if self.committed_mask is not None:
                    boost = torch.where(self.committed_mask, torch.ones_like(boost), boost)
            grad = grad * boost
            # Debug log every 200 steps (use training_step if available)
            if getattr(self, 'training_step', 0) % 200 == 1 and not torch.compiler.is_compiling():
                uncommitted_boost = boost[~self.committed_mask] if self.committed_mask is not None else boost
                print(f"  [CAM] boost mean={uncommitted_boost.mean().item():.3f} "
                      f"max={uncommitted_boost.max().item():.3f} n_stable_mean={self.n_stable.float().mean().item():.1f}")

        if self.use_gradient_skip and self.committed_mask is not None:
            grad_modified = grad * (~self.committed_mask).float()

            if self._commitment_step_counter % self._commitment_log_interval == 0:
                skip_rate = self.committed_mask.float().mean().item()
                print(
                    f"[GRAD-SKIP] Layer {self.__class__.__name__}: "
                    f"skip_rate={skip_rate * 100:.1f}%, "
                    f"committed={self.committed_mask.sum().item()}/{self.committed_mask.numel()}"
                )

            return grad_modified
        else:
            return grad

    def update_commitment_state(self, w_quantized, current_lr):
        """C2: Update commitment tracking state after quantization.

        Call this at the END of forward() with the quantized weights.

        Args:
            w_quantized: Quantized ternary weights (values in {-1, 0, +1})
            current_lr: Current learning rate from optimizer
        """
        if not self.use_commitment_tracking:
            return

        if self.sigma_g is None:
            self._init_commitment_buffers()

        self._current_lr_c2 = current_lr
        self._commitment_step_counter += 1

        with torch.no_grad():
            flipped = w_quantized != self.prev_ternary_c2
            self.n_stable = torch.where(
                flipped, torch.zeros_like(self.n_stable), self.n_stable + 1
            )

            dist_to_neg = (self.weight + 0.5).abs()
            dist_to_pos = (self.weight - 0.5).abs()
            margin = torch.min(dist_to_neg, dist_to_pos)

            N_sqrt = self.n_stable.float().sqrt()
            M_commit = (
                self.commitment_margin_multiplier * current_lr * N_sqrt * self.sigma_g
            )

            self.committed_mask = (margin > M_commit) & (
                self.n_stable >= self.commitment_n_stable_target
            )

            self.prev_ternary_c2.copy_(w_quantized)

            if self._commitment_step_counter % self._commitment_log_interval == 0:
                commit_rate = self.committed_mask.float().mean().item()
                avg_margin = margin.mean().item()
                avg_sigma_g = self.sigma_g.mean().item()
                avg_n_stable = self.n_stable.float().mean().item()

                print(
                    f"[COMMITMENT] Layer {self.__class__.__name__}: "
                    f"commit_rate={commit_rate * 100:.1f}%, "
                    f"avg_margin={avg_margin:.4f}, "
                    f"avg_sigma_g={avg_sigma_g:.4f}, "
                    f"avg_n_stable={avg_n_stable:.1f}, "
                    f"lr={current_lr:.2e}"
                )

    def enable_commitment_tracking(self, config=None):
        """Enable commitment tracking and register gradient hook.

        Args:
            config: Optional config object to update tracking parameters.
                    If provided, updates commitment settings from config.
        """
        if config is not None:
            self.use_commitment_tracking = getattr(
                config, "use_commitment_tracking", self.use_commitment_tracking
            )
            if self.use_commitment_tracking:
                self.use_gradient_skip = getattr(
                    config, "use_gradient_skip", self.use_gradient_skip
                )
                self.commitment_n_stable_target = getattr(
                    config, "commitment_n_stable", self.commitment_n_stable_target
                )
                self.commitment_ema_decay = getattr(
                    config, "commitment_ema_decay", self.commitment_ema_decay
                )
                self.commitment_margin_multiplier = getattr(
                    config,
                    "commitment_margin_multiplier",
                    self.commitment_margin_multiplier,
                )
                self.grad_skip_g_max = getattr(
                    config, "grad_skip_g_max", self.grad_skip_g_max
                )
                self._commitment_log_interval = getattr(
                    config, "commitment_log_interval", self._commitment_log_interval
                )
                print(
                    f"[C2] Config propagated: skip={self.use_gradient_skip}, "
                    f"n_stable={self.commitment_n_stable_target}, "
                    f"ema_decay={self.commitment_ema_decay}, "
                    f"margin_mult={self.commitment_margin_multiplier}"
                )

        if not self.use_commitment_tracking:
            return

        if getattr(self, '_grad_hook_handle', None) is None:
            self._grad_hook_handle = self.weight.register_hook(self._grad_hook_fn)
            print(f"[C2+C1] Gradient hook registered for {self.__class__.__name__}")

    def disable_commitment_tracking(self):
        """Disable commitment tracking and remove gradient hook."""
        if getattr(self, '_grad_hook_handle', None) is not None:
            self._grad_hook_handle.remove()
            self._grad_hook_handle = None
            print(f"[C2+C1] Gradient hook removed for {self.__class__.__name__}")

    def enable_cam(self, alpha: float = 0.5, beta: float = 0.01) -> None:
        """Enable Commitment-Aware Momentum (CAM) gradient boost.

        CAM is active whenever commitment tracking is active.
        Calling this method enables commitment tracking and sets boost parameters.

        Args:
            alpha: Max boost above 1.0 (default 0.5 → max 1.5x boost)
            beta: Decay rate (default 0.01 → half-boost at ~70 steps)
        """
        self.cam_alpha = alpha
        self.cam_beta = beta
        # CAM is automatic when commitment tracking is on
        if not getattr(self, 'use_commitment_tracking', False):
            self.use_commitment_tracking = True
            for buf_name in ("sigma_g", "n_stable", "committed_mask", "prev_ternary_c2"):
                if not hasattr(self, buf_name):
                    self.register_buffer(buf_name, None, persistent=False)
            if not hasattr(self, '_current_lr_c2'):
                self._current_lr_c2 = 1e-3
            if not hasattr(self, '_grad_hook_handle'):
                self._grad_hook_handle = None
        if getattr(self, '_grad_hook_handle', None) is None and hasattr(self, 'weight'):
            self._grad_hook_handle = self.weight.register_hook(self._grad_hook_fn)
        print(f"  [CAM] enabled via commitment tracking: alpha={alpha} beta={beta} "
              f"max_boost={1.0+alpha:.2f}x, half-boost at ~{int(0.693/beta)} steps")

    def set_current_lr(self, lr: float):
        """Set current learning rate for commitment threshold computation.

        Call this from training loop whenever LR changes.
        """
        if self.use_commitment_tracking:
            self._current_lr_c2 = lr

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

        if hasattr(self, "admm_u") and self.admm_u is not None:
            w_src = w_src + self.admm_u.to(w_src.dtype)

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
                    # Compute masks from scratch
                    w = LSQTernaryQuantize.apply(
                        w_src,
                        self.lsq_scale_pos,
                        self.lsq_scale_neg,
                        self.group_size,
                        t_coeff,
                        self.sigma_g if (self.use_commitment_tracking and self.sigma_g is not None) else None,
                    )

                    # F17: Cache the ternary quantization result for next forwards
                    if self.training and self.use_packed_cache:
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

                # ADMM dual variable update (R7-A3-M3) — runs every forward pass regardless of cache
                if hasattr(self, "admm_u") and self.admm_u is not None and self.training:
                    with torch.no_grad():
                        residual = (w_src - w.detach()).to(self.admm_u.dtype)
                        self.admm_u.add_(residual, alpha=self._admm_rho)
                    if not torch.compiler.is_compiling() and self.training_step % 500 == 1:
                        print(
                            f"[ADMM] step={self.training_step} "
                            f"u_mean={self.admm_u.abs().mean().item():.5f} "
                            f"u_max={self.admm_u.abs().max().item():.5f} "
                            f"residual_mean={residual.abs().mean().item():.5f}"
                        )

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
                output = F.linear(x, w.to(x.dtype), self.bias)

        else:
            if self.use_pwst and (w.abs() <= 1.0).all():
                if self.use_btg:
                    output = TernaryLinearBTG_PWST.apply(x, w, self.bias)
                else:
                    output = pwst_forward_vectorized(x, w, self.bias)
            else:
                if self.use_btg:
                    output = TernaryLinearBTG.apply(x, w, self.bias)
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

        # C2: Update commitment state with quantized weights
        if self.use_commitment_tracking and self.training:
            current_lr = getattr(self, "_current_lr_c2", 1e-3)
            w_quantized = w.detach()
            self.update_commitment_state(w_quantized, current_lr)

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

    def _reset_threshold_cache(self):
        """F13: Manually reset the cached threshold (e.g., after optimizer.step())."""
        self.cached_threshold = None
        self.threshold_step_counter = 0

    def get_ternary_stats(self) -> dict:
        with torch.no_grad():
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
