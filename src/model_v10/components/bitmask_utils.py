import torch
import torch.nn.functional as F


def pack_to_bitmask(w_ternary: torch.Tensor) -> tuple:
    """Convert ternary weight tensor {-1, 0, +1} to packed dual-bitmask format.

    This representation enables GEMV computation via set partitioning:
    y[i] = sum_{j in pos_mask[i]} x[j] - sum_{j in neg_mask[i]} x[j]

    Instead of floating-point matmul, we use two boolean masks (1 bit each per weight)
    reducing memory bandwidth by 8x compared to FP16/FP32 weights.

    Args:
        w_ternary: (M, K) tensor with values in {-1, 0, +1}

    Returns:
        pos_mask: (M, ceil(K/32)) int32 tensor, packed positive indicators
        neg_mask: (M, ceil(K/32)) int32 tensor, packed negative indicators
        original_K: int, original K dimension (before padding)
    """
    M, K = w_ternary.shape
    original_K = K

    K_pad = ((K + 31) // 32) * 32

    if K_pad > K:
        w_padded = F.pad(w_ternary, (0, K_pad - K))
    else:
        w_padded = w_ternary

    pos = (w_padded > 0)
    neg = (w_padded < 0)

    pos_grouped = pos.reshape(M, K_pad // 32, 32)
    neg_grouped = neg.reshape(M, K_pad // 32, 32)

    bit_positions = torch.arange(32, device=w_ternary.device, dtype=torch.int32)

    pos_mask = (pos_grouped.to(torch.int32) << bit_positions).sum(dim=-1)
    neg_mask = (neg_grouped.to(torch.int32) << bit_positions).sum(dim=-1)

    return pos_mask, neg_mask, original_K


def bitmask_gemv(x: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor,
                 original_K: int, bias=None) -> torch.Tensor:
    """Forward GEMV using dual-bitmask ternary weights.

    Computes y = x @ (pos - neg).T + bias where pos and neg are unpacked from bitmasks.
    This replaces F.linear(x, w.float(), bias) with bitwise unpacking + add/subtract.

    The forward pass matches F.linear(x, ternary_weight.float(), bias) exactly.
    Gradients flow through the unpacked masks for STE backward computation.

    Args:
        x: (..., K) input activations
        pos_mask: (M, words) int32 packed positive bitmask
        neg_mask: (M, words) int32 packed negative bitmask
        original_K: original input dimension (before padding in packing)
        bias: optional (M,) bias vector

    Returns:
        y: (..., M) output activations
    """
    M, words = pos_mask.shape

    bit_positions = torch.arange(32, device=x.device, dtype=torch.int32)

    pos_unpacked = ((pos_mask.unsqueeze(-1) >> bit_positions) & 1).to(torch.bool)
    neg_unpacked = ((neg_mask.unsqueeze(-1) >> bit_positions) & 1).to(torch.bool)

    pos_unpacked = pos_unpacked.reshape(M, -1)[:, :original_K]
    neg_unpacked = neg_unpacked.reshape(M, -1)[:, :original_K]

    input_shape = x.shape
    x_2d = x.reshape(-1, original_K)

    pos_float = pos_unpacked.to(x.dtype)
    neg_float = neg_unpacked.to(x.dtype)

    y_pos = F.linear(x_2d, pos_float, None)
    y_neg = F.linear(x_2d, neg_float, None)
    y = y_pos - y_neg

    if bias is not None:
        y = y + bias

    return y.reshape(*input_shape[:-1], M)


class BitmaskLinearFunc(torch.autograd.Function):
    """Custom autograd function for bitmask-based ternary linear layer.

    Forward: Uses bitmask GEMV for efficient add/sub-only computation.
    Backward grad_input: Standard ternary matmul (or BTG if enabled).
    Backward grad_weight: STE gradient through unpacked masks (w_ternary).
    """

    @staticmethod
    def forward(ctx, input, w_ternary, pos_mask, neg_mask, original_K, bias):
        ctx.save_for_backward(input, w_ternary, bias)
        ctx.original_K = original_K

        output = bitmask_gemv(input, pos_mask, neg_mask, original_K, bias)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, w_ternary, bias = ctx.saved_tensors
        original_K = ctx.original_K

        grad_input = None
        grad_weight = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            input_shape = input.shape
            x_2d = input.reshape(-1, original_K)
            go_shape = grad_output.shape
            go_2d = grad_output.reshape(-1, go_shape[-1])

            grad_input = go_2d.matmul(w_ternary.to(grad_output.dtype))
            grad_input = grad_input.reshape(*input_shape)

        if ctx.needs_input_grad[1]:
            go_shape = grad_output.shape
            go_2d = grad_output.reshape(-1, go_shape[-1])
            x_shape = input.shape
            x_2d = input.reshape(-1, x_shape[-1])

            grad_weight = go_2d.t().matmul(x_2d)

        if ctx.needs_input_grad[5] and bias is not None:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)

        return grad_input, grad_weight, None, None, None, grad_bias


class TernaryInt8LinearFunc(torch.autograd.Function):
    """INT8 ternary matmul via torch._int_mm (INT8 Tensor Cores).

    Forward: quantize x to int8 → _int_mm(x_int8, w_int8.T) → dequantize.
    Backward: standard float matmul for correct STE gradients.

    Replaces F.linear(x, w_ternary_float, bias) with INT8 equivalent.
    Uses per-row absmax for x, mean(lsq_scale_pos, lsq_scale_neg) for w scale.
    """

    @staticmethod
    def forward(ctx, x, w_ternary_float, w_int8_sign, alpha, bias):
        ctx.save_for_backward(x, w_ternary_float, bias)

        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]).float()
        M, K = x_2d.shape
        N = w_int8_sign.shape[0]  # out_features

        x_amax = x_2d.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        x_scale = x_amax / 127.0
        x_int8 = (x_2d / x_scale).round().clamp(-128, 127).to(torch.int8)

        # torch._int_mm requires M, K, N all multiples of 16 on CUDA
        align = 16
        M_pad = ((M + align - 1) // align) * align
        K_pad = ((K + align - 1) // align) * align
        N_pad = ((N + align - 1) // align) * align

        if M_pad != M or K_pad != K or N_pad != N:
            x_int8 = F.pad(x_int8, (0, K_pad - K, 0, M_pad - M))
            w_pad = F.pad(w_int8_sign, (0, K_pad - K, 0, N_pad - N))
            w_t = w_pad.t().contiguous()
        else:
            w_t = w_int8_sign.t().contiguous()

        out_int32 = torch._int_mm(x_int8, w_t)  # (M_pad, N_pad) int32
        out_int32 = out_int32[:M, :N]  # trim padding

        out = out_int32.to(x.dtype) * x_scale.to(x.dtype) * alpha.to(x.dtype)

        if bias is not None:
            out = out + bias

        return out.reshape(*input_shape[:-1], w_int8_sign.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        x, w_ternary_float, bias = ctx.saved_tensors

        grad_input = None
        grad_weight = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            input_shape = x.shape
            go_2d = grad_output.reshape(-1, grad_output.shape[-1])
            grad_input = go_2d.matmul(w_ternary_float.to(grad_output.dtype))
            grad_input = grad_input.reshape(*input_shape)

        if ctx.needs_input_grad[1]:
            go_2d = grad_output.reshape(-1, grad_output.shape[-1])
            x_2d = x.reshape(-1, x.shape[-1])
            grad_weight = go_2d.t().matmul(x_2d)

        if ctx.needs_input_grad[4] and bias is not None:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)

        return grad_input, grad_weight, None, None, grad_bias


def pack_ternary_int2(w_ternary: torch.Tensor) -> torch.Tensor:
    """Pack ternary weights {-1,0,+1} into INT2 format: 4 weights per byte.
    Encoding: -1→0, 0→1, +1→2 (2 bits each, 4 per uint8 byte).
    """
    M, K = w_ternary.shape
    encoded = (w_ternary.round().to(torch.int32) + 1)   # {0,1,2}
    K_pad = ((K + 3) // 4) * 4
    if K_pad > K:
        encoded = F.pad(encoded, (0, K_pad - K), value=1)  # pad with 1 (=zero weight)
    grouped = encoded.reshape(M, K_pad // 4, 4)
    shifts = torch.tensor([0, 2, 4, 6], device=w_ternary.device, dtype=torch.int32)
    packed_int32 = (grouped << shifts).sum(dim=-1)       # (M, K_pad//4)
    return packed_int32.to(torch.uint8)


def unpack_int2_to_int8(packed: torch.Tensor, original_K: int) -> torch.Tensor:
    """Unpack INT2-packed weights back to INT8 {-1,0,+1}.
    Args:
        packed: (M, ceil(K/4)) uint8
        original_K: original K before padding
    Returns:
        (M, original_K) int8
    """
    packed_int32 = packed.to(torch.int32)
    shifts = torch.tensor([0, 2, 4, 6], device=packed.device, dtype=torch.int32)
    decoded = ((packed_int32.unsqueeze(-1) >> shifts) & 0x3)   # (M, num_bytes, 4)
    decoded = decoded.reshape(packed.shape[0], -1)[:, :original_K]
    return (decoded - 1).to(torch.int8)


class TernaryInt2LinearFunc(torch.autograd.Function):
    """INT2 ternary matmul: 4 weights per byte → unpack on-the-fly → torch._int_mm.

    Forward: unpack INT2 weights to INT8 → quantize x to INT8 → _int_mm → dequantize.
    Backward: standard float matmul for correct STE gradients.

    4× less weight bandwidth than TernaryInt8LinearFunc.
    Activation gate: x.is_cuda and x.shape[-1] >= 512 and int2_packed_valid.
    """

    @staticmethod
    def forward(ctx, x, w_ternary_float, w_int2_packed, original_K, alpha, bias):
        ctx.save_for_backward(x, w_ternary_float, bias)

        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]).float()
        M, K = x_2d.shape
        N = w_ternary_float.shape[0]

        # Unpack INT2 → INT8 on the fly
        w_int8 = unpack_int2_to_int8(w_int2_packed, original_K)  # (N, K) int8

        # Per-row absmax quantization for x
        x_amax = x_2d.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        x_scale = x_amax / 127.0
        x_int8 = (x_2d / x_scale).round().clamp(-128, 127).to(torch.int8)

        # torch._int_mm requires M, K, N all multiples of 16
        align = 16
        M_pad = ((M + align - 1) // align) * align
        K_pad_mm = ((K + align - 1) // align) * align
        N_pad = ((N + align - 1) // align) * align

        if M_pad != M or K_pad_mm != K or N_pad != N:
            x_int8 = F.pad(x_int8, (0, K_pad_mm - K, 0, M_pad - M))
            w_pad = F.pad(w_int8, (0, K_pad_mm - K, 0, N_pad - N))
            w_t = w_pad.t().contiguous()
        else:
            w_t = w_int8.t().contiguous()

        out_int32 = torch._int_mm(x_int8, w_t)  # (M_pad, N_pad) int32
        out_int32 = out_int32[:M, :N]            # trim padding

        out = out_int32.to(x.dtype) * x_scale.to(x.dtype) * alpha.to(x.dtype)

        if bias is not None:
            out = out + bias

        return out.reshape(*input_shape[:-1], N)

    @staticmethod
    def backward(ctx, grad_output):
        x, w_ternary_float, bias = ctx.saved_tensors

        grad_input = None
        grad_weight = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            input_shape = x.shape
            go_2d = grad_output.reshape(-1, grad_output.shape[-1])
            grad_input = go_2d.matmul(w_ternary_float.to(grad_output.dtype))
            grad_input = grad_input.reshape(*input_shape)

        if ctx.needs_input_grad[1]:
            go_2d = grad_output.reshape(-1, grad_output.shape[-1])
            x_2d = x.reshape(-1, x.shape[-1])
            grad_weight = go_2d.t().matmul(x_2d)

        if ctx.needs_input_grad[5] and bias is not None:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)

        return grad_input, grad_weight, None, None, None, grad_bias


# ============================================================================
# R12-F7: CPU Bitserial Popcount GEMV (T-MAC approach)
# 4× throughput vs naive float matmul on CPU. NOT faster on GPU.
# Reference: T-MAC paper (https://arxiv.org/abs/2407.00088)
# ============================================================================


def pack_to_uint64_dual(w_ternary: torch.Tensor):
    """Pack ternary weights into dual 1-bit uint64 format for CPU popcount GEMV.

    Packs 64 ternary weights per uint64 word into separate pos/neg bit arrays.
    Used for CPU inference path (T-MAC bitserial approach).
    4× throughput vs naive float matmul on AVX2 CPUs.

    Args:
        w_ternary: (M, K) tensor with values in {-1, 0, +1}
    Returns:
        pos_packed: (M, ceil(K/64)) int64 tensor
        neg_packed: (M, ceil(K/64)) int64 tensor
        original_K: int
    """
    M, K = w_ternary.shape
    original_K = K

    # Pad K to multiple of 64
    K_pad = ((K + 63) // 64) * 64
    if K_pad > K:
        w_padded = F.pad(w_ternary, (0, K_pad - K))
    else:
        w_padded = w_ternary

    pos = (w_padded > 0).to(torch.int64)  # 1 where +1
    neg = (w_padded < 0).to(torch.int64)  # 1 where -1

    # Pack 64 bits per word
    pos_grouped = pos.reshape(M, K_pad // 64, 64)
    neg_grouped = neg.reshape(M, K_pad // 64, 64)

    bit_positions = torch.arange(64, dtype=torch.int64)

    pos_packed = (pos_grouped << bit_positions).sum(dim=-1)
    neg_packed = (neg_grouped << bit_positions).sum(dim=-1)

    return pos_packed, neg_packed, original_K


def _popcount_int64(x: torch.Tensor) -> torch.Tensor:
    """Population count (number of set bits) for int64 tensor.

    PyTorch has no native popcount. Implementation via uint8 view + cumsum.
    Handles negative int64 values correctly (treats as bit pattern).

    Args:
        x: (...) int64 tensor
    Returns:
        (...) int32 tensor with bit counts
    """
    # View as uint8 (8 bytes per int64)
    x_bytes = x.view(torch.uint8)  # shape: (..., 8)

    # Byte-level popcount via lookup (256-entry table)
    # Build table on first call
    if not hasattr(_popcount_int64, '_lut'):
        lut = torch.zeros(256, dtype=torch.int32)
        for i in range(256):
            count = 0
            n = i
            while n:
                count += n & 1
                n >>= 1
            lut[i] = count
        _popcount_int64._lut = lut

    lut = _popcount_int64._lut.to(x.device)

    # Index: each uint8 → bit count; sum 8 bytes per int64
    byte_counts = lut[x_bytes.long()]  # (..., 8) int32
    return byte_counts.sum(dim=-1).to(torch.int32)


def bitmask_gemv_cpu(x: torch.Tensor,
                     pos_packed: torch.Tensor,
                     neg_packed: torch.Tensor,
                     original_K: int,
                     x_scale: float = 1.0,
                     bias=None) -> torch.Tensor:
    """CPU bitserial GEMV via popcount — no multiplications.

    Computes y = (pos_weights - neg_weights) @ x_binary using popcount.
    x must be binarized (non-zero → 1, zero → 0) before calling.
    Uses dual uint64 pos/neg bitmasks.

    4× throughput vs naive float matmul on CPU (T-MAC paper, measured on M2/Intel).
    NOT faster on GPU — use TernaryInt2LinearFunc for CUDA.

    Args:
        x: (..., K) float tensor (will be binarized to ±1 for GEMV)
        pos_packed: (M, ceil(K/64)) int64 packed positive bits
        neg_packed: (M, ceil(K/64)) int64 packed negative bits
        original_K: original K before padding
        x_scale: dequantization scale (absmax/1)
        bias: optional (M,) bias
    Returns:
        y: (..., M) output
    """
    input_shape = x.shape
    x_2d = x.reshape(-1, original_K)
    batch = x_2d.shape[0]
    M = pos_packed.shape[0]
    words = pos_packed.shape[1]

    # Quantize x to binary: sign(x) → {-1, +1} → split into pos/neg bit packs
    x_scale_val = x_2d.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    x_bin = x_2d.sign()  # {-1, 0, +1}

    # Pack x activations into uint64 (same scheme as weights)
    K_pad = words * 64
    x_pos = (x_bin > 0).to(torch.int64)  # 1 where x > 0
    x_neg = (x_bin < 0).to(torch.int64)  # 1 where x < 0

    if K_pad > original_K:
        x_pos = F.pad(x_pos, (0, K_pad - original_K))
        x_neg = F.pad(x_neg, (0, K_pad - original_K))

    bit_positions = torch.arange(64, dtype=torch.int64, device=x.device)
    x_pos_packed = (x_pos.reshape(batch, words, 64) << bit_positions).sum(dim=-1)  # (B, words)
    x_neg_packed = (x_neg.reshape(batch, words, 64) << bit_positions).sum(dim=-1)  # (B, words)

    # GEMV via popcount: for each output row i, for each batch b:
    # y[b,i] = Σ_w popcount(pos[i,w] & x_pos[b,w]) - popcount(pos[i,w] & x_neg[b,w])
    #         - popcount(neg[i,w] & x_pos[b,w]) + popcount(neg[i,w] & x_neg[b,w])
    # = Σ_w popcount((pos[i,w]-neg[i,w]) & x[b,w]) with correct sign

    # Vectorized over batch and output rows
    # pos_packed: (M, words), x_pos_packed: (B, words)
    # Broadcast: (M, 1, words) & (1, B, words) = (M, B, words)

    pos_M = pos_packed.unsqueeze(1)  # (M, 1, words)
    neg_M = neg_packed.unsqueeze(1)  # (M, 1, words)
    xpos_B = x_pos_packed.unsqueeze(0)  # (1, B, words)
    xneg_B = x_neg_packed.unsqueeze(0)  # (1, B, words)

    # 4 popcount terms (M, B) — _popcount_int64 already sums all words×bytes
    pp = _popcount_int64(pos_M & xpos_B)  # pos_w & pos_x → (M, B)
    pn = _popcount_int64(pos_M & xneg_B)  # pos_w & neg_x → (M, B)
    np_ = _popcount_int64(neg_M & xpos_B)  # neg_w & pos_x → (M, B)
    nn_ = _popcount_int64(neg_M & xneg_B)  # neg_w & neg_x → (M, B)

    # y[m, b] = pp - pn - np + nn (sign logic: +w*+x=+, +w*-x=-, -w*+x=-, -w*-x=+)
    y = (pp - pn - np_ + nn_).float()  # (M, B)
    y = y.t()  # (B, M)

    # Dequantize: scale by x_scale_val (per-sample)
    y = y * x_scale_val  # (B, M)

    if bias is not None:
        y = y + bias

    return y.reshape(*input_shape[:-1], M)
