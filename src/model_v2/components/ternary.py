import math
import torch
import torch.nn as nn
import torch.nn.functional as F


_q_values_cache: dict = {}


class TernaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, group_size: int = 128, threshold_coeff: float = 0.7) -> torch.Tensor:
        orig_shape = weight.shape
        flat = weight.reshape(-1)
        pad_len = (group_size - flat.numel() % group_size) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, group_size)

        scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        normalized = grouped / scales

        threshold = threshold_coeff * normalized.abs().mean(dim=-1, keepdim=True)
        quantized = normalized.sign() * (normalized.abs() > threshold).float()

        result = (quantized * scales).reshape(-1)[:flat.numel() - pad_len if pad_len > 0 else flat.numel()]

        ctx.save_for_backward(weight)

        return result.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None, None


class BiSTETernaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, group_size: int = 128, threshold_coeff: float = 0.7) -> torch.Tensor:
        orig_shape = weight.shape
        flat = weight.reshape(-1)
        pad_len = (group_size - flat.numel() % group_size) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, group_size)

        scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        normalized = grouped / scales

        threshold = threshold_coeff * normalized.abs().mean(dim=-1, keepdim=True)
        quantized = normalized.sign() * (normalized.abs() > threshold).float()

        active_mask = (normalized.abs() > threshold).float()
        zero_mask = 1.0 - active_mask
        scale_grouped = scales.expand_as(grouped)

        result = (quantized * scales).reshape(-1)[:weight.numel()]

        active_mask_flat = active_mask.reshape(-1)[:weight.numel()].reshape(orig_shape)
        zero_mask_flat = zero_mask.reshape(-1)[:weight.numel()].reshape(orig_shape)
        scale_flat = scale_grouped.reshape(-1)[:weight.numel()].reshape(orig_shape)

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
    def forward(ctx, weight: torch.Tensor, group_size: int = 128, tau: float = 1.0) -> torch.Tensor:
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
            _q_values_cache[cache_key] = torch.tensor([-1.0, 0.0, 1.0], device=weight.device, dtype=weight.dtype)
        q_values = _q_values_cache[cache_key]
        w_expanded = w_scaled.unsqueeze(-1)
        dist_sq = (w_expanded - q_values) ** 2
        logits = -dist_sq / max(tau, 0.01)
        probs = torch.softmax(logits, dim=-1)

        p_neg, p_zero, p_pos = probs[..., 0], probs[..., 1], probs[..., 2]
        w_soft = gamma * (p_pos - p_neg)

        mean_q = p_pos - p_neg
        var = (1.0 - p_zero) - mean_q * mean_q

        result = w_soft.reshape(-1)[:weight.numel()]
        var_flat = var.reshape(-1)[:weight.numel()].reshape(orig_shape)

        ctx.save_for_backward(var_flat)
        ctx.tau = tau
        return result.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        var, = ctx.saved_tensors
        tau = ctx.tau
        jacobian = (2.0 / max(tau, 0.01)) * var
        grad = grad_output * jacobian
        return grad, None, None


class QuESTTernaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, group_size: int = 128, threshold_coeff: float = 0.7) -> torch.Tensor:
        orig_shape = weight.shape
        flat = weight.reshape(-1)
        pad_len = (group_size - flat.numel() % group_size) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, group_size)

        scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        normalized = grouped / scales

        threshold = threshold_coeff * normalized.abs().mean(dim=-1, keepdim=True)
        quantized = normalized.sign() * (normalized.abs() > threshold).float()

        quant_error = (normalized - quantized).abs()
        trust_threshold = 1.3 * threshold
        trust_mask = (quant_error <= trust_threshold).float()

        result = (quantized * scales).reshape(-1)[:weight.numel()]
        trust_mask_flat = trust_mask.reshape(-1)[:weight.numel()].reshape(orig_shape)

        ctx.save_for_backward(trust_mask_flat)
        return result.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        trust_mask, = ctx.saved_tensors
        grad = grad_output * trust_mask
        return grad, None, None


class TernaryLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 group_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.quantize_enabled = True

        self.ste_type = 'standard'
        self.training_step = 0
        self.total_training_steps = 1000
        self.hestia_tau_init = 2.0
        self.threshold_coeff = 0.7
        self.gradual_threshold = False
        self.threshold_start = 0.5
        self.threshold_end = 0.7
        self.pressure_blend = False
        self.pressure_duration = 0.2
        self._cached_tau = None
        self.use_triton_hestia = False

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        nn.init.xavier_uniform_(self.weight, gain=1.0 / (in_features ** 0.5))

        self.alpha = None
        self.alpha_group_size = None
        self.freeze_patterns = False

    def _apply_alpha(self, w: torch.Tensor) -> torch.Tensor:
        if self.alpha_group_size is None:
            return self.alpha * w
        orig_shape = w.shape
        flat = w.reshape(-1)
        pad_len = (self.alpha_group_size - flat.numel() % self.alpha_group_size) % self.alpha_group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, self.alpha_group_size)
        scaled = grouped * self.alpha.unsqueeze(-1)
        return scaled.reshape(-1)[:self.weight.numel()].reshape(orig_shape)

    @torch._dynamo.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantize_enabled:
            w = self.weight
            if self.alpha is not None:
                w = self._apply_alpha(w)
            return F.linear(x, w.to(x.dtype), self.bias)

        if self.gradual_threshold:
            progress = self.training_step / max(1, self.total_training_steps)
            t_coeff = self.threshold_start + (self.threshold_end - self.threshold_start) * progress
        else:
            t_coeff = self.threshold_coeff

        if self.ste_type == 'biste':
            w = BiSTETernaryQuantize.apply(self.weight, self.group_size, t_coeff)
        elif self.ste_type == 'hestia':
            if self._cached_tau is not None:
                tau = self._cached_tau
            else:
                progress = self.training_step / max(1, self.total_training_steps)
                tau = self.hestia_tau_init * 0.5 * (1.0 + math.cos(math.pi * progress))
                tau = max(tau, 0.01)

            if self.use_triton_hestia:
                from src.model_v2.components.triton_hestia import hestia_quantize_triton
                w = hestia_quantize_triton(self.weight, self.group_size, tau)
            else:
                w = HESTIATernaryQuantize.apply(self.weight, self.group_size, tau)
        elif self.ste_type == 'quest':
            w = QuESTTernaryQuantize.apply(self.weight, self.group_size, t_coeff)
        else:
            w = TernaryQuantize.apply(self.weight, self.group_size, t_coeff)

        if self.pressure_blend and self.training_step < self.total_training_steps * self.pressure_duration:
            p = self.training_step / max(1, self.total_training_steps * self.pressure_duration)
            w = (1.0 - p) * self.weight + p * w

        if self.alpha is not None:
            w = self._apply_alpha(w)

        return F.linear(x, w.to(x.dtype), self.bias)

    def get_ternary_stats(self) -> dict:
        with torch.no_grad():
            w = TernaryQuantize.apply(self.weight, self.group_size)
            flat = w.reshape(-1)
            total = flat.numel()
            n_pos = (flat > 0).sum().item()
            n_neg = (flat < 0).sum().item()
            n_zero = (flat == 0).sum().item()
            stats = {
                'positive': n_pos / total,
                'negative': n_neg / total,
                'zero': n_zero / total,
            }
            if self.alpha is not None:
                stats['alpha_mean'] = self.alpha.mean().item()
                stats['alpha_std'] = self.alpha.std().item() if self.alpha.numel() > 1 else 0.0
                stats['alpha_min'] = self.alpha.min().item()
                stats['alpha_max'] = self.alpha.max().item()
            return stats


class TernaryLinearInteger(nn.Module):
    """Ternary linear layer with integer gradient accumulation (Path C.1).

    Instead of FP32 latent weights + AdamW optimizer, this uses:
    - Weight stored as {-1, 0, +1} (the actual value, not quantized from FP32)
    - Learnable per-layer scale (alpha) — optimized by AdamW
    - Gradient accumulator for threshold-based weight flipping

    Memory per weight:
    - Standard TernaryLinear: ~12 bytes (4 FP32 weight + 8 AdamW state)
    - This class: ~6 bytes (4 FP32 weight-as-param + 4 FP32 accum, but no optimizer state)
    - Future: could be 3 bytes with INT8 weight + INT16 accum

    The weight nn.Parameter stores {-1,0,+1} as float for autograd compatibility.
    It must be EXCLUDED from the optimizer — updates happen via accumulate_and_update().
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 group_size: int = 128, alpha_group_size: int = None):
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
            pad_len = (alpha_group_size - total_weights % alpha_group_size) % alpha_group_size
            num_groups = (total_weights + pad_len) // alpha_group_size
            alpha_init = 1.0 / math.sqrt(in_features)
            self.alpha = nn.Parameter(torch.full((num_groups,), alpha_init))

        self.register_buffer('g_accum', torch.zeros(out_features, in_features))

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
        if self.alpha_group_size is None:
            return F.linear(x, self.weight * self.alpha, self.bias)

        orig_shape = self.weight.shape
        flat = self.weight.reshape(-1)
        pad_len = (self.alpha_group_size - flat.numel() % self.alpha_group_size) % self.alpha_group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, self.alpha_group_size)
        scaled = grouped * self.alpha.unsqueeze(-1)
        w = scaled.reshape(-1)[:self.weight.numel()].reshape(orig_shape)
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
                'positive': (flat > 0).sum().item() / total,
                'negative': (flat < 0).sum().item() / total,
                'zero': (flat == 0).sum().item() / total,
                'flip_count': self.flip_count,
                'total_updates': self.total_updates,
                'flips_per_update': self.flip_count / max(1, self.total_updates),
                'accum_abs_mean': self.g_accum.abs().mean().item(),
                'accum_abs_max': self.g_accum.abs().max().item(),
                'alpha_mean': self.alpha.mean().item(),
                'alpha_std': self.alpha.std().item() if self.alpha.numel() > 1 else 0.0,
                'alpha_min': self.alpha.min().item(),
                'alpha_max': self.alpha.max().item(),
            }
