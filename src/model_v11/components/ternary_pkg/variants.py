import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .quantizers import *


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
        group_size = getattr(module, "group_size", 128)

        w_flat = W.data.reshape(-1)
        pad_len = (group_size - w_flat.numel() % group_size) % group_size
        if pad_len > 0:
            w_flat = F.pad(w_flat, (0, pad_len))
        grouped = w_flat.reshape(-1, group_size)

        scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        normalized = grouped / scales

        abs_norm = normalized.abs()
        threshold = getattr(module, "threshold_coeff", 0.7) * abs_norm.mean(dim=-1, keepdim=True)

        q_sign = normalized.sign() * (abs_norm > threshold).float()
        q_scaled = q_sign * scales

        w_for_ang = grouped
        q_for_ang = q_scaled

        cos_sim = (
            (w_for_ang * q_for_ang).sum(dim=-1) /
            (w_for_ang.norm(dim=-1) * q_for_ang.norm(dim=-1).clamp(min=1e-8) + 1e-8)
        )
        angular_loss_val = (1 - cos_sim).mean() * angular_lambda
        angular_losses.append(angular_loss_val)

        distance_to_threshold = (abs_norm - threshold).clamp(min=1e-8)
        boundary_loss_val = -torch.log(distance_to_threshold + eps).mean() * boundary_lambda
        boundary_losses.append(boundary_loss_val)

    if not angular_losses:
        return torch.tensor(0.0), torch.tensor(0.0), 0

    total_angular = sum(angular_losses)
    total_boundary = sum(boundary_losses)
    return total_angular, total_boundary, len(angular_losses)
