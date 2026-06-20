"""Fused SSM layer kernels (K1-K4) - Phase 23 mega-kernel architecture.

K1: RMSNorm + in_proj ternary GEMV (implemented)
K2: Conv1d state update + depthwise conv + SiLU + x_proj ternary GEMV (implemented)
K3: dt_proj + softplus + SSM recurrence (implemented)
K4: SiLU gating + out_proj ternary GEMV + residual (implemented)

This mega-kernel architecture processes a single SSM step in 5 Triton kernel launches:
  K1 (1 launch): RMSNorm + in_proj ternary GEMV -> buf_xz
  K2 (2 launches): conv1d update + SiLU -> buf_xconv, x_proj GEMV -> buf_xdbc
  K3 (1 launch): dt_proj + softplus + SSM recurrence -> buf_y
  K4 (1 launch): SiLU gating + out_proj ternary GEMV + residual -> buf_output

Down from ~25-30 PyTorch kernel launches in the unfused path.
"""

import torch
import triton
import triton.language as tl
from dataclasses import dataclass
from typing import Optional


def _next_power_of_2(n: int) -> int:
    """Return smallest power of 2 >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


@dataclass
class FusedSSMLayerBuffers:
    """Pre-allocated CUDA Graph-compatible buffers for SSM layer computation."""

    buf_xz: torch.Tensor
    buf_xdbc: torch.Tensor
    buf_xconv: torch.Tensor
    buf_y: torch.Tensor
    buf_output: torch.Tensor
    device: torch.device

    @staticmethod
    def create(
        d_model: int,
        d_inner: int,
        d_state: int,
        dt_rank: int,
        device: torch.device,
    ) -> "FusedSSMLayerBuffers":
        """Allocate all buffers for a single SSM step.

        Args:
            d_model: Output dimension (640 in Phase 23)
            d_inner: Inner dimension (1280 in Phase 23)
            d_state: SSM state dimension (16 in Phase 23)
            dt_rank: dt_proj rank (40 in Phase 23)
            device: torch.device (cuda or cpu)

        Returns:
            FusedSSMLayerBuffers with all tensors allocated.
        """
        buf_xz = torch.empty(
            2 * d_inner,
            dtype=torch.float16,
            device=device,
        )

        buf_xdbc = torch.empty(
            dt_rank + 2 * d_state,
            dtype=torch.float32,
            device=device,
        )

        buf_xconv = torch.empty(
            d_inner,
            dtype=torch.float16,
            device=device,
        )

        buf_y = torch.empty(
            d_inner,
            dtype=torch.float32,
            device=device,
        )

        buf_output = torch.empty(
            d_model,
            dtype=torch.float16,
            device=device,
        )

        return FusedSSMLayerBuffers(
            buf_xz=buf_xz,
            buf_xdbc=buf_xdbc,
            buf_xconv=buf_xconv,
            buf_y=buf_y,
            buf_output=buf_output,
            device=device,
        )


_first_call_k1 = True
_first_call_k2 = True
_first_call_k3 = True
_first_call_k4 = True


@triton.jit
def _fused_dt_ssm_kernel(
    x_dbc_ptr,
    x_conv_ptr,
    h_ptr,
    y_ptr,
    dt_proj_weight_ptr,
    dt_proj_bias_ptr,
    A_log_ptr,
    D_ptr,
    n_heads: tl.int32,
    d_state: tl.int32,
    HEAD_DIM: tl.constexpr,
    D_STATE: tl.constexpr,
    DT_RANK: tl.constexpr,
    DT_RANK_PAD: tl.constexpr,
):
    """K3 kernel: dt_proj + softplus + SSM recurrence.

    Grid: (n_heads,) - one program per head.

    Inputs:
        x_dbc_ptr: buf_xdbc (dt_rank + 2*d_state values as fp32)
        x_conv_ptr: buf_xconv (d_inner values as fp16, per-head sliced below)
        h_ptr: SSM state (1, n_heads, d_state, head_dim) fp32
        y_ptr: buf_y output (d_inner values as fp32, per-head sliced below)
        dt_proj_weight_ptr: (d_inner, dt_rank) fp32
        dt_proj_bias_ptr: (d_inner,) fp32
        A_log_ptr: (n_heads,) fp32
        D_ptr: (d_inner,) fp32

    Computes:
        1. Dense GEMV: dt_vals[0:32] = dt_proj(dt_input)
        2. Softplus: dt_vals = log(1 + exp(dt_vals))
        3. Aggregate: dt_head = mean(dt_vals[0:32])
        4. SSM recurrence with state update and output accumulation
        5. D skip connection
    """
    pid = tl.program_id(0)

    dt_input_offs = tl.arange(0, DT_RANK_PAD)
    dt_mask = dt_input_offs < DT_RANK
    dt_input = tl.load(x_dbc_ptr + dt_input_offs, mask=dt_mask, other=0.0).to(tl.float32)

    y_acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    dim_offs = tl.arange(0, HEAD_DIM)

    for d in tl.static_range(HEAD_DIM):
        dt_proj_idx = pid * HEAD_DIM + d
        dt_proj_weight_row = tl.load(dt_proj_weight_ptr + dt_proj_idx * DT_RANK + dt_input_offs, mask=dt_mask, other=0.0).to(tl.float32)
        dt_proj_bias_val = tl.load(dt_proj_bias_ptr + dt_proj_idx)

        dt_val = tl.sum(dt_proj_weight_row * dt_input) + dt_proj_bias_val
        dt_val = tl.log(1.0 + tl.exp(dt_val))

        if d == 0:
            dt_head_acc = dt_val
        else:
            dt_head_acc += dt_val

    dt_head = dt_head_acc / HEAD_DIM

    x_conv_base = pid * HEAD_DIM
    x_head = tl.load(x_conv_ptr + x_conv_base + dim_offs).to(tl.float32)

    A_log_val = tl.load(A_log_ptr + pid)
    A_val = -tl.exp(A_log_val)
    dA = tl.exp(dt_head * A_val)

    for s in tl.static_range(D_STATE):
        h_base = pid * d_state * HEAD_DIM + s * HEAD_DIM
        h_vals = tl.load(h_ptr + h_base + dim_offs).to(tl.float32)

        B_s = tl.load(x_dbc_ptr + DT_RANK + s).to(tl.float32)
        C_s = tl.load(x_dbc_ptr + DT_RANK + D_STATE + s).to(tl.float32)

        h_new = dA * h_vals + dt_head * B_s * x_head

        tl.store(h_ptr + h_base + dim_offs, h_new)

        y_acc += C_s * h_new

    D_base = pid * HEAD_DIM
    D_vals = tl.load(D_ptr + D_base + dim_offs).to(tl.float32)
    y_acc += D_vals * x_head

    y_base = pid * HEAD_DIM
    tl.store(y_ptr + y_base + dim_offs, y_acc)


def fused_dt_ssm_step(
    x_dbc: torch.Tensor,
    x_conv: torch.Tensor,
    h: torch.Tensor,
    dt_proj_weight: torch.Tensor,
    dt_proj_bias: torch.Tensor,
    A_log: torch.Tensor,
    D: torch.Tensor,
    y_buf: torch.Tensor,
) -> torch.Tensor:
    """Python wrapper for K3 (dt_proj + softplus + SSM recurrence).

    Args:
        x_dbc: (dt_rank + 2*d_state,) fp32 - [dt_input (0:40), B (40:56), C (56:72)]
        x_conv: (d_inner,) fp16 - conv1d output, per-head sliced in kernel
        h: (n_heads, d_state, head_dim) fp32 - SSM state, mutated in-place
        dt_proj_weight: (d_inner, dt_rank) fp32
        dt_proj_bias: (d_inner,) fp32
        A_log: (n_heads,) fp32
        D: (d_inner,) fp32
        y_buf: (d_inner,) fp32 - pre-allocated output buffer

    Returns:
        y_buf: (d_inner,) fp32 - SSM output
    """
    global _first_call_k3
    if _first_call_k3:
        print("[Triton SSM] K3 kernel (dt_proj + softplus + recurrence) active")
        _first_call_k3 = False

    x_dbc = x_dbc.contiguous().float()
    x_conv = x_conv.contiguous().half()
    h = h.contiguous().float()
    dt_proj_weight = dt_proj_weight.contiguous().float()
    dt_proj_bias = dt_proj_bias.contiguous().float()
    A_log = A_log.contiguous().float()
    D = D.contiguous().float()

    n_heads = int(h.shape[0])
    d_state = int(h.shape[1])
    head_dim = int(h.shape[2])
    d_inner = n_heads * head_dim
    dt_rank = int(dt_proj_weight.shape[1])

    grid = (n_heads,)
    dt_rank_pad = _next_power_of_2(dt_rank)

    _fused_dt_ssm_kernel[grid](
        x_dbc,
        x_conv,
        h,
        y_buf,
        dt_proj_weight,
        dt_proj_bias,
        A_log,
        D,
        n_heads,
        d_state,
        HEAD_DIM=head_dim,
        D_STATE=d_state,
        DT_RANK=dt_rank,
        DT_RANK_PAD=dt_rank_pad,
    )

    return y_buf


@triton.jit
def _fused_rmsnorm_inproj_kernel(
    x_ptr,
    norm_weight_ptr,
    w_val_ptr,
    w_sign_ptr,
    scales_ptr,
    y_ptr,
    N: tl.int32,
    D_MODEL: tl.constexpr,
    D_MODEL_PAD: tl.constexpr,
    num_groups: tl.int32,
    stride_w: tl.int32,
    stride_s: tl.int32,
    EPS: tl.constexpr,
    WORDS_PER_GROUP: tl.constexpr,
):
    """K1 kernel: fused RMSNorm + in_proj ternary GEMV.

    Grid: (N,) - one program per output element.

    Pass 1: Compute inv_rms = rsqrt(sum(x^2) / D_MODEL + EPS)
    Pass 2: GEMV with on-the-fly normalization: x_normed = x * inv_rms * weight
    """
    n_idx = tl.program_id(0)
    if n_idx >= N:
        return

    # ---- Pass 1: compute inv_rms scalar ----
    x_offs = tl.arange(0, D_MODEL_PAD)
    x_mask = x_offs < D_MODEL
    x_raw = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0).to(tl.float32)
    sq_sum = tl.sum(x_raw * x_raw)
    inv_rms = tl.rsqrt(sq_sum / D_MODEL + EPS)

    # ---- Pass 2: ternary GEMV with on-the-fly normalization ----
    acc = tl.zeros((), dtype=tl.float32)
    bit_offsets = tl.arange(0, 32)

    for g in range(num_groups):
        group_sum = tl.zeros((), dtype=tl.float32)

        for w in range(WORDS_PER_GROUP):
            word_idx = g * WORDS_PER_GROUP + w
            k_base = word_idx * 32

            # Load raw x chunk and norm weight, normalize on-the-fly
            x_chunk = tl.load(x_ptr + k_base + bit_offsets).to(tl.float32)
            w_norm = tl.load(norm_weight_ptr + k_base + bit_offsets).to(tl.float32)
            x_normed = x_chunk * inv_rms * w_norm

            # Unpack ternary weights for this word
            val_word = tl.load(w_val_ptr + n_idx * stride_w + word_idx)
            sign_word = tl.load(w_sign_ptr + n_idx * stride_w + word_idx)

            val_bits = (val_word >> bit_offsets) & 1
            sign_bits = (sign_word >> bit_offsets) & 1
            w_ternary = (val_bits - 2 * (val_bits & sign_bits)).to(tl.float32)

            group_sum += tl.sum(x_normed * w_ternary)

        scale = tl.load(scales_ptr + n_idx * stride_s + g)
        acc += group_sum * scale

    tl.store(y_ptr + n_idx, acc.to(tl.float16))


def fused_rmsnorm_inproj_step(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    w_val: torch.Tensor,
    w_sign: torch.Tensor,
    scales: torch.Tensor,
    y_buf: torch.Tensor,
    d_model: int,
    group_size: int,
) -> torch.Tensor:
    """K1 wrapper: fused RMSNorm + in_proj ternary GEMV.

    Args:
        x: (d_model,) fp16 - single token embedding
        norm_weight: (d_model,) fp16/fp32 - RMSNorm learnable weight
        w_val: (N, K_packed) int32 - packed ternary val bitmap
        w_sign: (N, K_packed) int32 - packed ternary sign bitmap
        scales: (N, num_groups) float - per-group quantization scales
        y_buf: (N,) fp16 - pre-allocated output buffer (N = 2*d_inner)
        d_model: int - input dimension
        group_size: int - ternary quantization group size

    Returns:
        y_buf: (N,) fp16 - first d_inner elements are x_inner, last d_inner are z
    """
    global _first_call_k1
    if _first_call_k1:
        print("[Triton SSM] K1 kernel (RMSNorm + in_proj ternary GEMV) active")
        _first_call_k1 = False

    x = x.contiguous().half()
    norm_weight = norm_weight.contiguous()
    w_val = w_val.contiguous()
    w_sign = w_sign.contiguous()
    scales = scales.contiguous()

    N = int(w_val.shape[0])
    K_padded = int(w_val.shape[1]) * 32
    num_groups = K_padded // group_size
    words_per_group = group_size // 32
    d_model_pad = _next_power_of_2(d_model)

    grid = (N,)

    _fused_rmsnorm_inproj_kernel[grid](
        x,
        norm_weight,
        w_val,
        w_sign,
        scales,
        y_buf,
        N,
        D_MODEL=d_model,
        D_MODEL_PAD=d_model_pad,
        num_groups=num_groups,
        stride_w=int(w_val.stride(0)),
        stride_s=int(scales.stride(0)),
        EPS=1e-6,
        WORDS_PER_GROUP=words_per_group,
    )

    return y_buf


@triton.jit
def _fused_gate_outproj_residual_kernel(
    y_ptr,
    z_ptr,
    x_t_ptr,
    w_val_ptr,
    w_sign_ptr,
    scales_ptr,
    output_ptr,
    N: tl.int32,
    num_groups: tl.int32,
    stride_w: tl.int32,
    stride_s: tl.int32,
    WORDS_PER_GROUP: tl.constexpr,
):
    """K4 kernel: SiLU gating + out_proj ternary GEMV + residual add.

    Grid: (d_model,) - one program per output element.

    For each output element n:
      1. Iterate groups/words of d_inner dimension
      2. For each 32-element word: load y(fp32), load z(fp16->fp32),
         compute silu(z)*y on-the-fly, multiply by unpacked ternary weight
      3. Accumulate group_sum, multiply by per-group scale
      4. Add residual x_t[n]
      5. Store as fp16
    """
    n_idx = tl.program_id(0)
    if n_idx >= N:
        return

    acc = tl.zeros((), dtype=tl.float32)
    bit_offsets = tl.arange(0, 32)

    for g in range(num_groups):
        group_sum = tl.zeros((), dtype=tl.float32)

        for w in range(WORDS_PER_GROUP):
            word_idx = g * WORDS_PER_GROUP + w
            k_base = word_idx * 32

            y_vals = tl.load(y_ptr + k_base + bit_offsets).to(tl.float32)
            z_vals = tl.load(z_ptr + k_base + bit_offsets).to(tl.float32)
            silu_z = z_vals * (1.0 / (1.0 + tl.exp(-z_vals)))
            gated = silu_z * y_vals

            val_word = tl.load(w_val_ptr + n_idx * stride_w + word_idx)
            sign_word = tl.load(w_sign_ptr + n_idx * stride_w + word_idx)
            val_bits = (val_word >> bit_offsets) & 1
            sign_bits = (sign_word >> bit_offsets) & 1
            w_ternary = (val_bits - 2 * (val_bits & sign_bits)).to(tl.float32)

            group_sum += tl.sum(gated * w_ternary)

        scale = tl.load(scales_ptr + n_idx * stride_s + g)
        acc += group_sum * scale

    residual = tl.load(x_t_ptr + n_idx).to(tl.float32)
    result = residual + acc

    tl.store(output_ptr + n_idx, result.to(tl.float16))


def fused_gate_outproj_residual_step(
    y: torch.Tensor,
    z: torch.Tensor,
    x_t: torch.Tensor,
    w_val: torch.Tensor,
    w_sign: torch.Tensor,
    scales: torch.Tensor,
    output_buf: torch.Tensor,
    d_model: int,
    group_size: int,
) -> torch.Tensor:
    """K4 wrapper: fused SiLU gating + out_proj ternary GEMV + residual add.

    Args:
        y: (d_inner,) fp32 - K3 SSM output (buf_y)
        z: (d_inner,) fp16 - K1 gate branch (second half of buf_xz)
        x_t: (d_model,) fp16 - residual input
        w_val: (d_model, K_packed) int32 - out_proj packed ternary val bitmap
        w_sign: (d_model, K_packed) int32 - out_proj packed ternary sign bitmap
        scales: (d_model, num_groups) float - out_proj per-group scales
        output_buf: (d_model,) fp16 - pre-allocated output buffer
        d_model: int - output dimension (640)
        group_size: int - ternary quantization group size (128)

    Returns:
        output_buf: (d_model,) fp16 - residual + out_proj(silu(z) * y)
    """
    global _first_call_k4
    if _first_call_k4:
        print("[Triton SSM] K4 kernel (gate + out_proj + residual) active")
        _first_call_k4 = False

    y = y.contiguous().float()
    z = z.contiguous().half()
    x_t = x_t.contiguous().half()
    w_val = w_val.contiguous()
    w_sign = w_sign.contiguous()
    scales = scales.contiguous()

    N = d_model
    K_padded = int(w_val.shape[1]) * 32
    num_groups = K_padded // group_size
    words_per_group = group_size // 32

    grid = (N,)

    _fused_gate_outproj_residual_kernel[grid](
        y,
        z,
        x_t,
        w_val,
        w_sign,
        scales,
        output_buf,
        N,
        num_groups,
        int(w_val.stride(0)),
        int(scales.stride(0)),
        WORDS_PER_GROUP=words_per_group,
    )

    return output_buf


@triton.jit
def _fused_conv_silu_kernel(
    x_inner_ptr,
    conv_state_ptr,
    conv_weight_ptr,
    conv_bias_ptr,
    x_conv_ptr,
    d_inner: tl.int32,
    D_CONV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """K2a kernel: conv1d state update + depthwise convolution + SiLU.

    Grid: (cdiv(d_inner, BLOCK_D),) - one program per block of channels.
    """
    pid = tl.program_id(0)
    c_start = pid * BLOCK_D
    c_offs = c_start + tl.arange(0, BLOCK_D)
    c_mask = c_offs < d_inner

    x_new = tl.load(x_inner_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    s0 = tl.load(conv_state_ptr + c_offs * D_CONV + 0, mask=c_mask, other=0.0).to(tl.float32)
    s1 = tl.load(conv_state_ptr + c_offs * D_CONV + 1, mask=c_mask, other=0.0).to(tl.float32)
    s2 = tl.load(conv_state_ptr + c_offs * D_CONV + 2, mask=c_mask, other=0.0).to(tl.float32)
    s3 = tl.load(conv_state_ptr + c_offs * D_CONV + 3, mask=c_mask, other=0.0).to(tl.float32)

    new_s0 = s1
    new_s1 = s2
    new_s2 = s3
    new_s3 = x_new

    tl.store(conv_state_ptr + c_offs * D_CONV + 0, new_s0.to(tl.float16), mask=c_mask)
    tl.store(conv_state_ptr + c_offs * D_CONV + 1, new_s1.to(tl.float16), mask=c_mask)
    tl.store(conv_state_ptr + c_offs * D_CONV + 2, new_s2.to(tl.float16), mask=c_mask)
    tl.store(conv_state_ptr + c_offs * D_CONV + 3, new_s3.to(tl.float16), mask=c_mask)

    w0 = tl.load(conv_weight_ptr + c_offs * D_CONV + 0, mask=c_mask, other=0.0).to(tl.float32)
    w1 = tl.load(conv_weight_ptr + c_offs * D_CONV + 1, mask=c_mask, other=0.0).to(tl.float32)
    w2 = tl.load(conv_weight_ptr + c_offs * D_CONV + 2, mask=c_mask, other=0.0).to(tl.float32)
    w3 = tl.load(conv_weight_ptr + c_offs * D_CONV + 3, mask=c_mask, other=0.0).to(tl.float32)

    conv_out = new_s0 * w0 + new_s1 * w1 + new_s2 * w2 + new_s3 * w3
    bias = tl.load(conv_bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    conv_out = conv_out + bias

    sigmoid_val = 1.0 / (1.0 + tl.exp(-conv_out))
    silu_out = conv_out * sigmoid_val

    tl.store(x_conv_ptr + c_offs, silu_out.to(tl.float16), mask=c_mask)


@triton.jit
def _fused_xproj_gemv_kernel(
    x_conv_ptr,
    w_val_ptr,
    w_sign_ptr,
    scales_ptr,
    x_dbc_ptr,
    N_OUT: tl.int32,
    num_groups: tl.int32,
    stride_w: tl.int32,
    stride_s: tl.int32,
    WORDS_PER_GROUP: tl.constexpr,
):
    """K2b kernel: x_proj ternary GEMV.

    Grid: (N_OUT,) = (72,) - one program per output element.
    Output stored as fp32 (K3 reads buf_xdbc as fp32).
    """
    n_idx = tl.program_id(0)
    if n_idx >= N_OUT:
        return

    acc = tl.zeros((), dtype=tl.float32)
    bit_offsets = tl.arange(0, 32)

    for g in range(num_groups):
        group_sum = tl.zeros((), dtype=tl.float32)

        for w in range(WORDS_PER_GROUP):
            word_idx = g * WORDS_PER_GROUP + w
            k_base = word_idx * 32

            x_vals = tl.load(x_conv_ptr + k_base + bit_offsets).to(tl.float32)

            val_word = tl.load(w_val_ptr + n_idx * stride_w + word_idx)
            sign_word = tl.load(w_sign_ptr + n_idx * stride_w + word_idx)
            val_bits = (val_word >> bit_offsets) & 1
            sign_bits = (sign_word >> bit_offsets) & 1
            w_ternary = (val_bits - 2 * (val_bits & sign_bits)).to(tl.float32)

            group_sum += tl.sum(x_vals * w_ternary)

        scale = tl.load(scales_ptr + n_idx * stride_s + g)
        acc += group_sum * scale

    tl.store(x_dbc_ptr + n_idx, acc)


def fused_conv_silu_xproj_step(
    x_inner: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    xproj_w_val: torch.Tensor,
    xproj_w_sign: torch.Tensor,
    xproj_scales: torch.Tensor,
    x_conv_buf: torch.Tensor,
    x_dbc_buf: torch.Tensor,
    d_inner: int,
    d_conv: int,
    group_size: int,
) -> tuple:
    """K2 wrapper: fused conv1d + SiLU + x_proj ternary GEMV.

    Two kernel launches:
      K2a: conv_state update + depthwise conv + SiLU -> x_conv_buf
      K2b: x_proj ternary GEMV -> x_dbc_buf

    Args:
        x_inner: (d_inner,) fp16 - first half of K1 output
        conv_state: (d_inner, d_conv) fp16 - rolling buffer, MUTATED IN-PLACE
        conv_weight: (d_inner, d_conv) fp32 - conv1d.weight.squeeze(1)
        conv_bias: (d_inner,) fp32 - conv1d.bias
        xproj_w_val: (N_OUT, K_packed) int32
        xproj_w_sign: (N_OUT, K_packed) int32
        xproj_scales: (N_OUT, num_groups) float
        x_conv_buf: (d_inner,) fp16 - output buffer
        x_dbc_buf: (N_OUT,) fp32 - output buffer
        d_inner: int
        d_conv: int
        group_size: int

    Returns:
        (x_conv_buf, x_dbc_buf)
    """
    global _first_call_k2
    if _first_call_k2:
        print("[Triton SSM] K2 kernel (conv1d + SiLU + x_proj GEMV) active")
        _first_call_k2 = False

    x_inner = x_inner.contiguous().half()
    conv_state = conv_state.contiguous()
    conv_weight = conv_weight.contiguous()
    conv_bias = conv_bias.contiguous()

    BLOCK_D = 128
    grid_k2a = (triton.cdiv(d_inner, BLOCK_D),)

    _fused_conv_silu_kernel[grid_k2a](
        x_inner,
        conv_state,
        conv_weight,
        conv_bias,
        x_conv_buf,
        d_inner,
        D_CONV=d_conv,
        BLOCK_D=BLOCK_D,
    )

    N_OUT = int(xproj_w_val.shape[0])
    K_padded = int(xproj_w_val.shape[1]) * 32
    num_groups = K_padded // group_size
    words_per_group = group_size // 32

    grid_k2b = (N_OUT,)

    _fused_xproj_gemv_kernel[grid_k2b](
        x_conv_buf,
        xproj_w_val,
        xproj_w_sign,
        xproj_scales,
        x_dbc_buf,
        N_OUT,
        num_groups,
        int(xproj_w_val.stride(0)),
        int(xproj_scales.stride(0)),
        WORDS_PER_GROUP=words_per_group,
    )

    return x_conv_buf, x_dbc_buf
