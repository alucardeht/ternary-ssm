"""Fused SSM recurrence kernel in Triton.

Replaces 9-14 PyTorch kernel launches for SSM state update with a single
Triton kernel. Grid: (n_heads, batch) - one program per (head, batch) pair.

Operations fused:
- dA computation: exp(dt * A)
- State update: h = dA * h + dt * B * x  (for each state dimension)
- Output readout: y = sum(C * h)
- D skip connection: y += D * x
"""
import torch
import triton
import triton.language as tl

_first_call = True


@triton.jit
def _fused_ssm_recurrence_kernel(
    # Tensor pointers
    h_ptr,       # (batch, n_heads, d_state, head_dim) FP32 - read+write
    y_ptr,       # (batch, d_inner) FP32 - write only
    dt_ptr,      # (batch, n_heads) FP32 - read only
    A_ptr,       # (n_heads,) FP32 - read only (already negated)
    B_ptr,       # (batch, d_state) FP32 - read only
    C_ptr,       # (batch, d_state) FP32 - read only
    x_ptr,       # (batch, n_heads, head_dim) FP32 - read only
    D_ptr,       # (d_inner,) FP32 - read only
    # Scalar dimensions
    n_heads: tl.int32,
    d_state: tl.int32,
    # Strides for h (4D tensor)
    stride_h_batch: tl.int32,
    stride_h_head: tl.int32,
    stride_h_state: tl.int32,
    # Strides for x (3D tensor)
    stride_x_batch: tl.int32,
    stride_x_head: tl.int32,
    # Stride for y (2D tensor)
    stride_y_batch: tl.int32,
    # Constexpr for Triton unrolling
    HEAD_DIM: tl.constexpr,
    D_STATE: tl.constexpr,
):
    head_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    # Load per-head scalars
    dt_val = tl.load(dt_ptr + batch_idx * n_heads + head_idx)
    A_val = tl.load(A_ptr + head_idx)
    dA = tl.exp(dt_val * A_val)

    # Load x vector for this head: x[batch_idx, head_idx, 0:head_dim]
    dim_offs = tl.arange(0, HEAD_DIM)
    x_base = batch_idx * stride_x_batch + head_idx * stride_x_head
    x_vals = tl.load(x_ptr + x_base + dim_offs)

    # Accumulate output
    y_acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # Loop over state dimensions (unrolled by Triton)
    for s in tl.static_range(D_STATE):
        B_val = tl.load(B_ptr + batch_idx * d_state + s)
        C_val = tl.load(C_ptr + batch_idx * d_state + s)

        # h[batch_idx, head_idx, s, 0:head_dim]
        h_base = batch_idx * stride_h_batch + head_idx * stride_h_head + s * stride_h_state
        h_vals = tl.load(h_ptr + h_base + dim_offs)

        # State update: h_new = dA * h + dt * B * x
        h_new = dA * h_vals + dt_val * B_val * x_vals

        # Write back in-place
        tl.store(h_ptr + h_base + dim_offs, h_new)

        # Accumulate: y += C * h_new
        y_acc += C_val * h_new

    # D skip connection: D[head_idx*head_dim : (head_idx+1)*head_dim] * x
    D_base = head_idx * HEAD_DIM
    D_vals = tl.load(D_ptr + D_base + dim_offs)
    y_acc += D_vals * x_vals

    # Store output: y[batch_idx, head_idx*head_dim : (head_idx+1)*head_dim]
    y_base = batch_idx * stride_y_batch + head_idx * HEAD_DIM
    tl.store(y_ptr + y_base + dim_offs, y_acc)


def fused_ssm_recurrence_step(
    h: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    x: torch.Tensor,
    D: torch.Tensor,
    y_buf: torch.Tensor = None,
) -> torch.Tensor:
    """Python wrapper for the fused SSM recurrence Triton kernel.

    Args:
        h: (batch, n_heads, d_state, head_dim) FP32 - mutated in-place
        dt: (batch, n_heads) FP32
        A: (n_heads,) FP32 (already negative: -exp(A_log))
        B: (batch, d_state) FP32
        C: (batch, d_state) FP32
        x: (batch, n_heads, head_dim) FP32
        D: (d_inner,) FP32
        y_buf: optional pre-allocated output buffer for CUDA Graph compat

    Returns:
        y: (batch, d_inner) FP32
    """
    global _first_call
    if _first_call:
        print("[Triton SSM] Fused recurrence kernel active")
        _first_call = False

    h = h.contiguous()
    dt = dt.contiguous().float()
    A = A.contiguous().float()
    B = B.contiguous().float()
    C = C.contiguous().float()
    x = x.contiguous().float()
    D = D.contiguous().float()

    batch = int(h.shape[0])
    n_heads = int(h.shape[1])
    d_state_val = int(h.shape[2])
    head_dim = int(h.shape[3])
    d_inner = n_heads * head_dim

    if y_buf is None:
        y_buf = torch.empty(batch, d_inner, dtype=torch.float32, device=h.device)

    # Grid: one program per (head, batch) - same pattern as ternary_gemv_v1
    grid = (n_heads, batch)

    # Launch - all strides as individual ints, matching GEMV pattern exactly
    _fused_ssm_recurrence_kernel[grid](
        h, y_buf, dt, A, B, C, x, D,
        n_heads,
        d_state_val,
        int(h.stride(0)),
        int(h.stride(1)),
        int(h.stride(2)),
        int(x.stride(0)),
        int(x.stride(1)),
        int(y_buf.stride(0)),
        HEAD_DIM=head_dim,
        D_STATE=d_state_val,
    )

    return y_buf
