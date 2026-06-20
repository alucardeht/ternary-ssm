import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PackedTernaryBitmap:
    w_val: torch.Tensor
    w_sign: torch.Tensor
    scales: torch.Tensor
    bias: Optional[torch.Tensor]
    in_features: int
    out_features: int
    K_padded: int
    group_size: int


def pack_ternary_bitmap(module) -> PackedTernaryBitmap:
    weight = module.weight.data
    N, K = weight.shape
    device = weight.device
    group_size = module.group_size

    pad_to = max(group_size, 16)
    K_padded = ((K + pad_to - 1) // pad_to) * pad_to
    if K_padded > K:
        weight = F.pad(weight, (0, K_padded - K))

    grouped = weight.reshape(-1, group_size)
    scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    normalized = grouped / scales
    threshold = 0.7 * normalized.abs().mean(dim=-1, keepdim=True)
    ternary = normalized.sign() * (normalized.abs() > threshold).float()

    num_groups = K_padded // group_size
    scales = scales.squeeze(-1).reshape(N, num_groups)

    ternary_flat = ternary.reshape(N, K_padded)

    K_packed = K_padded // 32
    ternary_reshaped = ternary_flat.reshape(N, K_packed, 32)

    nonzero = (ternary_reshaped != 0).int()
    negative = (ternary_reshaped < 0).int()

    bit_positions = (1 << torch.arange(32, device=device, dtype=torch.int32))
    w_val = (nonzero * bit_positions).sum(dim=-1).int()
    w_sign = (negative * bit_positions).sum(dim=-1).int()

    bias_tensor = module.bias.data.clone() if module.bias is not None else None

    if hasattr(module, 'alpha') and module.alpha is not None:
        if module.alpha_group_size is None:
            scales = scales * module.alpha.data.item()
        else:
            alpha_flat = module.alpha.data
            num_alpha_groups = (K_padded + module.alpha_group_size - 1) // module.alpha_group_size
            if alpha_flat.numel() == num_groups:
                scales = scales * alpha_flat.unsqueeze(0)

    return PackedTernaryBitmap(
        w_val=w_val,
        w_sign=w_sign,
        scales=scales,
        bias=bias_tensor,
        in_features=K,
        out_features=N,
        K_padded=K_padded,
        group_size=group_size,
    )


@triton.jit
def _ternary_gemv_v1_kernel(
    x_ptr,
    w_val_ptr,
    w_sign_ptr,
    scales_ptr,
    bias_ptr,
    y_ptr,
    N: tl.int32,
    K_packed: tl.int32,
    num_groups: tl.int32,
    stride_w: tl.int32,
    stride_s: tl.int32,
    HAS_BIAS: tl.constexpr,
    WORDS_PER_GROUP: tl.constexpr,
):
    n_idx = tl.program_id(0)
    if n_idx >= N:
        return

    acc = tl.zeros((), dtype=tl.float32)

    for g in range(num_groups):
        group_sum = tl.zeros((), dtype=tl.float32)

        for w in range(WORDS_PER_GROUP):
            word_idx = g * WORDS_PER_GROUP + w

            val_word = tl.load(w_val_ptr + n_idx * stride_w + word_idx)
            sign_word = tl.load(w_sign_ptr + n_idx * stride_w + word_idx)

            bit_offsets = tl.arange(0, 32)
            val_bits = (val_word >> bit_offsets) & 1
            sign_bits = (sign_word >> bit_offsets) & 1

            k_base = word_idx * 32
            x_vals = tl.load(x_ptr + k_base + bit_offsets).to(tl.float32)

            w_ternary = (val_bits - 2 * (val_bits & sign_bits)).to(tl.float32)
            contribution = x_vals * w_ternary
            group_sum += tl.sum(contribution)

        scale = tl.load(scales_ptr + n_idx * stride_s + g)
        acc += group_sum * scale

    if HAS_BIAS:
        acc += tl.load(bias_ptr + n_idx)

    tl.store(y_ptr + n_idx, acc.to(tl.float16))


def ternary_gemv_v1(x: torch.Tensor, packed: PackedTernaryBitmap) -> torch.Tensor:
    """V1 kernel wrapper — one program per output row. Kept for benchmarking."""
    K_padded = packed.K_padded
    N = packed.out_features
    group_size = packed.group_size
    num_groups = K_padded // group_size

    x_flat = x.reshape(-1)
    if x_flat.shape[0] < K_padded:
        x_flat = F.pad(x_flat, (0, K_padded - x_flat.shape[0]))

    x_device = x_flat.to(torch.float16)
    y = torch.zeros(N, dtype=torch.float16, device=packed.w_val.device)

    K_packed = K_padded // 32
    words_per_group = group_size // 32
    has_bias = packed.bias is not None

    _ternary_gemv_v1_kernel[(N,)](
        x_device,
        packed.w_val, packed.w_sign, packed.scales,
        packed.bias if has_bias else torch.zeros(N, device=packed.w_val.device),
        y, N, K_packed, num_groups,
        packed.w_val.stride(0), packed.scales.stride(0),
        HAS_BIAS=has_bias,
        WORDS_PER_GROUP=words_per_group,
    )
    return y


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 4}, num_warps=1),
        triton.Config({'BLOCK_N': 8}, num_warps=1),
        triton.Config({'BLOCK_N': 8}, num_warps=2),
        triton.Config({'BLOCK_N': 16}, num_warps=2),
        triton.Config({'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_N': 32}, num_warps=2),
        triton.Config({'BLOCK_N': 32}, num_warps=4),
    ],
    key=['N', 'K_packed', 'num_groups'],
)
@triton.jit
def _ternary_gemv_v2_kernel(
    x_ptr,
    w_val_ptr,
    w_sign_ptr,
    scales_ptr,
    bias_ptr,
    y_ptr,
    N: tl.int32,
    K_packed: tl.int32,
    num_groups: tl.int32,
    stride_w: tl.int32,
    stride_s: tl.int32,
    HAS_BIAS: tl.constexpr,
    WORDS_PER_GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    bit_offsets = tl.arange(0, 32)

    for g in range(num_groups):
        group_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for w in range(WORDS_PER_GROUP):
            word_idx = g * WORDS_PER_GROUP + w
            k_base = word_idx * 32

            x_vals = tl.load(x_ptr + k_base + bit_offsets).to(tl.float32)

            w_addrs = n_offsets * stride_w + word_idx
            val_words = tl.load(w_val_ptr + w_addrs, mask=mask_n, other=0)
            sign_words = tl.load(w_sign_ptr + w_addrs, mask=mask_n, other=0)

            val_bits = (val_words[:, None] >> bit_offsets[None, :]) & 1
            sign_bits = (sign_words[:, None] >> bit_offsets[None, :]) & 1

            w_ternary = (val_bits - 2 * (val_bits & sign_bits)).to(tl.float32)

            contribution = x_vals[None, :] * w_ternary
            group_sum += tl.sum(contribution, axis=1)

        scales = tl.load(scales_ptr + n_offsets * stride_s + g, mask=mask_n, other=0.0)
        acc += group_sum * scales

    if HAS_BIAS:
        bias = tl.load(bias_ptr + n_offsets, mask=mask_n, other=0.0)
        acc += bias

    tl.store(y_ptr + n_offsets, acc.to(tl.float16), mask=mask_n)


def ternary_gemv(x: torch.Tensor, packed: PackedTernaryBitmap) -> torch.Tensor:
    """Default kernel for inference. Uses V1 (single-row) with optimized arithmetic."""
    K_padded = packed.K_padded
    N = packed.out_features
    group_size = packed.group_size
    num_groups = K_padded // group_size

    x_flat = x.reshape(-1)
    if x_flat.shape[0] < K_padded:
        x_flat = F.pad(x_flat, (0, K_padded - x_flat.shape[0]))

    x_device = x_flat.to(torch.float16)
    y = torch.zeros(N, dtype=torch.float16, device=packed.w_val.device)

    K_packed = K_padded // 32
    words_per_group = group_size // 32
    has_bias = packed.bias is not None

    _ternary_gemv_v1_kernel[(N,)](
        x_device,
        packed.w_val, packed.w_sign, packed.scales,
        packed.bias if has_bias else torch.zeros(N, device=packed.w_val.device),
        y, N, K_packed, num_groups,
        packed.w_val.stride(0), packed.scales.stride(0),
        HAS_BIAS=has_bias,
        WORDS_PER_GROUP=words_per_group,
    )
    return y


def ternary_gemv_v2(x: torch.Tensor, packed: PackedTernaryBitmap) -> torch.Tensor:
    """V2 kernel — multi-row with autotune. Better for larger matrices (100M+ models)."""
    K_padded = packed.K_padded
    N = packed.out_features
    group_size = packed.group_size
    num_groups = K_padded // group_size

    x_flat = x.reshape(-1)
    if x_flat.shape[0] < K_padded:
        x_flat = F.pad(x_flat, (0, K_padded - x_flat.shape[0]))

    x_device = x_flat.to(torch.float16)
    y = torch.zeros(N, dtype=torch.float16, device=packed.w_val.device)

    K_packed = K_padded // 32
    words_per_group = group_size // 32
    has_bias = packed.bias is not None

    grid = lambda META: (triton.cdiv(N, META['BLOCK_N']),)

    _ternary_gemv_v2_kernel[grid](
        x_device,
        packed.w_val, packed.w_sign, packed.scales,
        packed.bias if has_bias else torch.zeros(N, device=packed.w_val.device),
        y, N, K_packed, num_groups,
        packed.w_val.stride(0), packed.scales.stride(0),
        HAS_BIAS=has_bias,
        WORDS_PER_GROUP=words_per_group,
    )
    return y


class TernaryTritonLinear(nn.Module):
    def __init__(self, packed: PackedTernaryBitmap):
        super().__init__()
        self.register_buffer('w_val', packed.w_val)
        self.register_buffer('w_sign', packed.w_sign)
        self.register_buffer('scales', packed.scales)
        if packed.bias is not None:
            self.register_buffer('bias_param', packed.bias)
        else:
            self.bias_param = None
        self.in_features = packed.in_features
        self.out_features = packed.out_features
        self.K_padded = packed.K_padded
        self.group_size = packed.group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        K = x.shape[-1]
        x_flat = x.reshape(-1, K)
        B = x_flat.shape[0]

        if B == 1:
            packed = PackedTernaryBitmap(
                self.w_val,
                self.w_sign,
                self.scales,
                self.bias_param,
                self.in_features,
                self.out_features,
                self.K_padded,
                self.group_size,
            )
            y = ternary_gemv(x_flat[0], packed)
            return y.reshape(*orig_shape, self.out_features)

        return self._fallback_forward(x_flat)

    def _fallback_forward(self, x: torch.Tensor) -> torch.Tensor:
        bit_offsets = torch.arange(32, device=self.w_val.device, dtype=torch.int32)
        val_expanded = (
            (self.w_val.unsqueeze(-1) >> bit_offsets.view(1, 1, 32)) & 1
        ).float()
        sign_expanded = (
            (self.w_sign.unsqueeze(-1) >> bit_offsets.view(1, 1, 32)) & 1
        ).float()

        ternary = val_expanded * (1.0 - 2.0 * sign_expanded)
        ternary = ternary.reshape(self.out_features, self.K_padded)

        num_groups = self.K_padded // self.group_size
        scales_expanded = self.scales.unsqueeze(-1).expand(-1, -1, self.group_size)
        scales_flat = scales_expanded.reshape(self.out_features, self.K_padded)
        w = ternary * scales_flat
        w = w[:, : self.in_features]

        return F.linear(x, w.to(x.dtype), self.bias_param)


def convert_model_to_triton(model, device):
    from ...components.ternary import TernaryLinear

    n_converted = 0
    n_skipped = 0

    for name, module in list(model.named_modules()):
        if isinstance(module, TernaryLinear):
            packed = pack_ternary_bitmap(module)
            replacement = TernaryTritonLinear(packed)

            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], replacement)
            n_converted += 1

    return model, n_converted, n_skipped


def verify_triton_against_baseline(module, device) -> Tuple[float, float]:
    from ...components.ternary import TernaryLinear

    if not isinstance(module, TernaryLinear):
        return {
            "error": "module is not TernaryLinear",
            "max_abs_error": float('nan'),
            "mean_abs_error": float('nan'),
        }

    torch.manual_seed(42)
    test_input = torch.randn(1, module.in_features, device=device, dtype=torch.float32)

    with torch.no_grad():
        baseline_output = module(test_input)

        packed = pack_ternary_bitmap(module)
        triton_module = TernaryTritonLinear(packed)
        triton_module = triton_module.to(device)

        triton_output = triton_module(test_input)

    diff = (baseline_output - triton_output.float()).abs()
    max_error = diff.max().item()
    mean_error = diff.mean().item()

    return {
        "max_abs_error": max_error,
        "mean_abs_error": mean_error,
        "relative_error": max_error / (baseline_output.abs().max().item() + 1e-8),
        "passed": max_error < 1e-2,
        "baseline_shape": baseline_output.shape,
        "triton_shape": triton_output.shape,
    }
