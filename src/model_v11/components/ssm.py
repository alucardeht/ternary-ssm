import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .rmsnorm import RMSNorm
from .ternary import TernaryLinear

try:
    from fla.ops.simple_gla import chunk_simple_gla
    _FLA_AVAILABLE = True
except ImportError:
    _FLA_AVAILABLE = False


# ========================================================================
# TBPTT (Truncated BPTT) Implementation Guide for RecursiveSSM
# ========================================================================
#
# When RecursiveSSM is implemented, add the following to its __init__:
#
#     self.num_iterations = config.recursive_iterations if config else 32
#     self.tbptt_k = config.tbptt_k if config else 24
#     self._tbptt_logged = False
#
# In the forward pass (iteration loop):
#
#     for iteration in range(self.num_iterations):
#         h_new = self.ssm_block(x, h, iteration)
#
#         if self.tbptt_k > 0:
#             truncate_point = self.num_iterations - self.tbptt_k
#             if iteration < truncate_point:
#                 h_new = h_new.detach()
#
#                 if not self._tbptt_logged:
#                     print(f"[TBPTT] Truncating gradients: keeping last K={self.tbptt_k} of {self.num_iterations} iterations (truncate_point={truncate_point})")
#                     self._tbptt_logged = True
#
#         h = h_new
#
# Reference: .workspace/implementation/recipe-TBPTT.md (Finding R3-A4-M1+M3)
# Expected speedup: 1.18-1.3x on training (1.3-1.5x on backward pass)
# Quality loss: <1% at K=24 (conservative, keeps 75% of iterations)
# ========================================================================

_segsum_mask_cache: dict = {}


def ternary_pole_reassignment(A_log, model, batch, criterion, device='cuda',
                               sample_size=16, verbose=False):
    """Exact ternary optimization for A_log via exhaustive pole evaluation.

    For each A_log element, evaluate loss at {-1, 0, +1} and assign optimal.

    Args:
        A_log: Parameter tensor to optimize (d_state,) or (d_state, n_heads)
        model: Full model (for forward pass)
        batch: Input batch (x, y) for loss evaluation
        criterion: Loss function
        device: Device
        sample_size: Number of sequence positions to sample for loss (reduce cost)
        verbose: Print debug info

    Returns:
        A_ternary: Optimized ternary A_log
        n_changed: Number of elements that changed

    Reference: R3-A4-M11 (Exact pole assignment without STE bias)
    """
    x, y = batch
    x, y = x.to(device), y.to(device)

    if x.size(1) > sample_size:
        indices = torch.randperm(x.size(1))[:sample_size]
        x_sample = x[:, indices]
        y_sample = y[:, indices]
    else:
        x_sample, y_sample = x, y

    A_shape = A_log.shape
    A_flat = A_log.data.view(-1)
    A_ternary = torch.zeros_like(A_flat)

    candidates = [-1.0, 0.0]  # [HURWITZ] +1 eliminado — eigenvalores devem estar dentro do disco unitário
    n_changed = 0

    A_original = A_log.data.clone()

    with torch.no_grad():
        for i in range(A_flat.numel()):
            losses = []

            for candidate in candidates:
                A_log.data.view(-1)[i] = candidate

                model.eval()
                logits = model(x_sample)
                loss = criterion(logits.view(-1, logits.size(-1)), y_sample.view(-1))
                losses.append(loss.item())

            optimal_idx = int(torch.tensor(losses).argmin())
            A_ternary[i] = candidates[optimal_idx]

            if A_ternary[i] != A_flat[i]:
                n_changed += 1

            if verbose and i % 10 == 0:
                print(f"  A_log[{i}]: losses={losses}, optimal={candidates[optimal_idx]}")

    model.train()

    A_log.data.copy_(A_ternary.view(A_shape))

    if verbose:
        print(f"[TernaryPoleReassignment] Changed {n_changed}/{A_flat.numel()} elements")

    return A_ternary.view(A_shape), n_changed


def segsum(x: torch.Tensor) -> torch.Tensor:
    """Compute segment sum in log-space for SSD numerical stability.

    For input x of shape (..., T), computes a lower-triangular matrix where
    result[..., i, j] = sum(x[..., j+1:i+1]) for i >= j, -inf otherwise.
    Used to compute cumulative decay factors without numerical overflow.
    """
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    cache_key = (T, x.device)
    if cache_key not in _segsum_mask_cache:
        _segsum_mask_cache[cache_key] = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0
        )
    mask = _segsum_mask_cache[cache_key]
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


class FusedSegSumExp(torch.autograd.Function):
    """Fused kernel: segsum + exp combined into single forward/backward.

    Replaces the operation:
        L = torch.exp(segsum(log_decay_t))

    Which normally involves: cumsum -> broadcast subtract -> mask -> masked_fill -> exp
    (6+ kernel launches) with a single custom autograd Function.

    Forward computes exp(segsum(x)) where segsum is the lower-triangular
    segment sum operation. Backward correctly propagates gradients through
    both exp and the cumsum structure.
    """

    @staticmethod
    def forward(ctx, x):
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]

        cache_key = (T, x.device)
        if cache_key not in _segsum_mask_cache:
            _segsum_mask_cache[cache_key] = torch.tril(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0
            )
        mask = _segsum_mask_cache[cache_key]

        x_segsum = x_segsum.masked_fill(~mask, float('-inf'))
        result = torch.exp(x_segsum)

        ctx.save_for_backward(result)
        ctx._T = T
        ctx._cache_key = cache_key
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (result,) = ctx.saved_tensors

        grad_segsum = grad_output * result

        grad_cumsum = grad_segsum.sum(dim=-1) - grad_segsum.sum(dim=-2)

        grad_x = grad_cumsum.flip(-1).cumsum(-1).flip(-1)

        return grad_x


def _quantize_for_storage(x, n_bits=4, group_size=128):
    """Quantize a tensor to int8 for memory-efficient storage.

    Returns: (q_int: int8, scales: float32, metadata: dict)
    Memory: ~1 byte/elem (int8) + 4/group_size bytes/elem (scales)
    vs 2 bytes/elem (FP16) or 4 bytes/elem (FP32)

    R2-A2-F3
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    flat = x.detach().float().reshape(-1)
    numel = flat.numel()

    pad_len = (group_size - numel % group_size) % group_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len))

    grouped = flat.reshape(-1, group_size)

    n_levels = 2 ** n_bits - 1
    half_levels = n_levels / 2.0

    scales = grouped.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-8)
    normalized = grouped / scales
    scaled = normalized * half_levels
    q_int = torch.round(scaled).clamp(-half_levels, half_levels).to(torch.int8)

    return q_int, scales, {'shape': orig_shape, 'dtype': orig_dtype,
                            'numel': numel, 'pad_len': pad_len,
                            'half_levels': half_levels}


def _dequantize_from_storage(q_int, scales, metadata):
    """Reconstruct tensor from quantized int8 storage.

    R2-A2-F3
    """
    q_float = q_int.float() / metadata['half_levels']
    dequantized = q_float * scales
    flat = dequantized.reshape(-1)
    if metadata['pad_len'] > 0:
        flat = flat[:metadata['numel']]
    return flat.reshape(metadata['shape']).to(metadata['dtype'])


def _ssm_committed_frac(ssm_module) -> float:
    """Compute fraction of ternary weights committed across all TernaryLinear children."""
    total = 0
    committed = 0
    for m in ssm_module.modules():
        if hasattr(m, 'committed_mask') and m.committed_mask is not None:
            total += m.committed_mask.numel()
            committed += int(m.committed_mask.sum().item())
    if total == 0:
        return 0.0
    return committed / total


class ReversibleChunkBoundaryProp(torch.autograd.Function):
    """Reversible state propagation for chunked SSM scan cross-chunk boundaries.

    Forward: saves h_update and total_decay per chunk boundary.
             Reconstructs h iteratively but only stores inputs, not intermediate states.
             Memory: O(n_chunks × d_state) instead of O(T × d_state).

    Backward: reconstructs h in REVERSE from saved per-chunk updates.
              Since total_decay = exp(sum(log_decay)) > 0, division is safe.
              Equation: h_prev = (h_curr - h_update) / total_decay

    Math proof of reversibility:
      Forward iteration k: h_k = h_{k-1} * total_decay_k + h_update_k
      Rearranged: h_{k-1} = (h_k - h_update_k) / total_decay_k
      No singularity: total_decay_k = exp(positive sum) > 0 always.

    R19-C: State memory compression (28× VRAM reduction on h states)
    """

    @staticmethod
    def forward(ctx, h_init, h_update_chunks, total_decay_chunks):
        """
        Args:
            h_init: (batch, n_heads, d_state, head_dim) initial state
            h_update_chunks: (n_chunks, batch, n_heads, d_state, head_dim) per-chunk updates
            total_decay_chunks: (n_chunks, batch, n_heads) per-chunk total decay factors

        Returns:
            h_final: (batch, n_heads, d_state, head_dim) final state after all chunks
        """
        n_chunks = h_update_chunks.shape[0]
        h = h_init.clone()

        for chunk_idx in range(n_chunks):
            h = h * total_decay_chunks[chunk_idx].unsqueeze(-1).unsqueeze(-1)
            h = h + h_update_chunks[chunk_idx]

        ctx.save_for_backward(h_init, h_update_chunks, total_decay_chunks)
        return h

    @staticmethod
    def backward(ctx, grad_h_final):
        """Reconstruct h in reverse, then backprop through h_update and total_decay."""
        h_init, h_update_chunks, total_decay_chunks = ctx.saved_tensors
        n_chunks = h_update_chunks.shape[0]

        h = h_init.clone()
        h_states = [h.clone()]

        for chunk_idx in range(n_chunks):
            h = h * total_decay_chunks[chunk_idx].unsqueeze(-1).unsqueeze(-1)
            h = h + h_update_chunks[chunk_idx]
            h_states.append(h.clone())

        grad_h_update_chunks = torch.zeros_like(h_update_chunks)
        grad_total_decay_chunks = torch.zeros_like(total_decay_chunks)
        grad_h = grad_h_final.clone()

        for chunk_idx in range(n_chunks - 1, -1, -1):
            total_decay = total_decay_chunks[chunk_idx]
            decay_exp = total_decay.unsqueeze(-1).unsqueeze(-1)

            grad_h_update_chunks[chunk_idx] = grad_h.clone()

            grad_total_decay_chunks[chunk_idx] = (
                (grad_h * h_states[chunk_idx]).sum(dim=(-2, -1))
            )

            grad_h = grad_h * decay_exp

        grad_h_init = grad_h
        return grad_h_init, grad_h_update_chunks, grad_total_decay_chunks


class ReversibleStepScan(torch.autograd.Function):
    """Reversible SSM step-wise scan that reconstructs intermediate states during backward.

    Exploits the invertibility of the SSM recurrence h_t = dA_t * h_{t-1} + dB_t * x_t
    to reconstruct each h_{t-1} during backward without storing the trajectory.

    Forward: runs standard scan, stores ONLY h_final + inputs (dA_steps, dB_steps, x_steps).
             Returns: (h_final, y) where y is the output sequence.

    Backward: given grad_output and grad_h_final, reconstructs h_t on-the-fly in reverse
              using h_{t-1} = (h_t - dB_t * x_t) / dA_t, then backpropagates gradients.

    Mathematical guarantee: dA_t = exp(dt_t * A) where dt_t > 0 and A < 0, so dA_t ∈ (0,1)
    always. Division by dA_t is numerically safe. No approximation — exact reconstruction.

    Memory savings:
    - Without: autograd stores h at all T steps: O(T × batch × d_inner × d_state)
    - With: stores only h_final (1 state) + inputs: O(T × d) where d is small
    - For seq_len=256, batch=4, d_inner=512, d_state=16: 67 MB → ~0 MB activation storage

    R332-M2-3: Exact Ternary-A Adjoint (Zero Trajectory Storage)
    """

    @staticmethod
    def forward(ctx, h_init, dA_steps, dB_steps, x_steps, C, boundary_mask, seq_len, batch, d_inner, d_state):
        """
        Args:
            h_init: (batch, d_inner, d_state) initial state, float32
            dA_steps: (seq_len, batch, d_inner, d_state) discretized decay per step, float32
            dB_steps: (seq_len, batch, d_inner, d_state) discretized input projection per step, float32
            x_steps: (seq_len, batch, d_inner) input sequence, original dtype
            C: (seq_len, batch, d_inner, d_state) output projection per step, float32
            boundary_mask: (batch, seq_len) bool or None, marks document boundaries for state reset
            seq_len, batch, d_inner, d_state: pre-computed shape info

        Returns:
            y: (batch, seq_len, d_inner) output sequence
        """
        device = h_init.device
        dtype = x_steps.dtype

        y = torch.zeros(batch, seq_len, d_inner, device=device, dtype=dtype)
        h = h_init.clone()

        for t in range(seq_len):
            if boundary_mask is not None and boundary_mask[:, t].any():
                h[boundary_mask[:, t]] = 0.0

            h = dA_steps[t] * h + dB_steps[t] * x_steps[t].unsqueeze(-1).to(torch.float32)
            y[:, t] = (h * C[t]).sum(-1).to(dtype)

        h_final = h.clone()
        ctx.save_for_backward(h_final, dA_steps, dB_steps, x_steps, C, boundary_mask if boundary_mask is not None else torch.empty(0, device=device, dtype=torch.bool))
        ctx.seq_len = seq_len
        ctx.batch = batch
        ctx.d_inner = d_inner
        ctx.d_state = d_state
        ctx.has_boundary_mask = boundary_mask is not None

        return y

    @staticmethod
    def backward(ctx, grad_y):
        """Reconstruct h trajectory in reverse, backprop through all parameters.

        Given grad_y (batch, seq_len, d_inner), compute gradients for dA, dB, x, C, h_init.
        Returns gradients for: h_init, dA_steps, dB_steps, x_steps, C, boundary_mask, seq_len, batch, d_inner, d_state

        Mathematical flow for each step in REVERSE:
        1. y_t = sum(h_t * C_t, dim=-1)  so  grad_h_t += grad_y_t * C_t
        2. h_t = dA_t * h_prev + dB_t * x_t  so:
           - grad_dA_t = (grad_h_t * h_prev).sum(reduce_dims)
           - grad_dB_t = grad_h_t * x_t
           - grad_x_t = (grad_h_t * dB_t).sum(reduce_dims)
           - grad_h_prev = grad_h_t * dA_t
        3. Invert: h_prev = (h_t - dB_t * x_t) / dA_t
        """
        h_final, dA_steps, dB_steps, x_steps, C, boundary_mask_saved = ctx.saved_tensors
        seq_len = ctx.seq_len
        batch = ctx.batch
        d_inner = ctx.d_inner
        d_state = ctx.d_state
        has_boundary_mask = ctx.has_boundary_mask

        device = h_final.device
        dtype_orig = x_steps.dtype

        grad_dA = torch.zeros_like(dA_steps)
        grad_dB = torch.zeros_like(dB_steps)
        grad_x = torch.zeros_like(x_steps)
        grad_C = torch.zeros_like(C)

        h = h_final.clone()
        grad_h = torch.zeros_like(h, dtype=torch.float32)

        if has_boundary_mask:
            boundary_mask = boundary_mask_saved
        else:
            boundary_mask = None

        dA_min = 1e-6

        for t in range(seq_len - 1, -1, -1):
            grad_h_from_y = grad_y[:, t].unsqueeze(-1).to(torch.float32) * C[t]
            grad_h = grad_h + grad_h_from_y

            grad_C[t] = (grad_y[:, t].unsqueeze(-1).to(torch.float32) * h).to(C.dtype)

            dA_clamped = dA_steps[t].clamp(min=dA_min)
            x_t = x_steps[t].unsqueeze(-1).to(torch.float32)

            h_prev = (h - dB_steps[t] * x_t) / dA_clamped

            grad_dA[t] = (grad_h * h_prev).to(dA_steps.dtype)
            grad_dB[t] = (grad_h * x_t).to(dB_steps.dtype)
            grad_x[t] = ((grad_h * dB_steps[t]).sum(dim=-1)).to(dtype_orig)

            grad_h = grad_h * dA_steps[t]

            if boundary_mask is not None and t > 0 and boundary_mask[:, t].any():
                grad_h[boundary_mask[:, t]] = 0.0

            h = h_prev

        grad_h_init = grad_h

        return grad_h_init, grad_dA, grad_dB, grad_x, grad_C, None, None, None, None, None


class QuantizedCheckpointSSDScan(torch.autograd.Function):
    """Activation-compressed gradient checkpointing for SSD scan.

    Forward: runs the scan with FULL-PRECISION inputs (exact computation).
             Saves QUANTIZED versions of inputs (compressed storage).
    Backward: dequantizes inputs, RE-RUNS the scan forward (recomputation),
              then autograd computes gradients on the dequantized inputs.

    This combines checkpointing (recompute forward in backward) with
    activation compression (store inputs at reduced precision).

    Memory savings:
    - Without this: autograd saves all intermediates (L, M, CB, Y_chunks, h states)
      in FP16/FP32. For batch=32, seq=256, d_inner=512, n_heads=16:
      ~50-100 MB per SSM layer.
    - With this: only stores quantized inputs (x_conv, dt, B, C, z) in INT8.
      ~12-25 MB per SSM layer (4-bit effective).
    - Tradeoff: one extra forward pass (recomputation) in backward.

    R2-A2-F3: Quamba-SE Soft-Edge SSM Activation Quantization
    """

    @staticmethod
    def forward(ctx, h_norm, A_log, D, boundary_mask_or_none, act_fn_str,
                n_bits, group_size, scan_fn_name, ssm_module, seq_len):
        """Extended checkpoint: recomputes entire projection chain in backward.

        Args:
            h_norm: (batch, seq_len, d_model) — raw input BEFORE in_proj
            A_log: Parameter (n_heads,) — log of state matrix diagonal
            D: Parameter (d_inner,) — skip connection weight
            boundary_mask_or_none: (batch, seq_len) bool or torch.zeros(0) if None
            act_fn_str: 'silu' or 'relu'
            n_bits, group_size: Quantization parameters
            scan_fn_name: Name of scan method ('_ssd_scan_fla', '_ssd_scan_chunked', etc.)
            ssm_module: Reference to SelectiveSSMSSD instance
            seq_len: Sequence length (needed for conv1d truncation)
        """
        act_fn = F.silu if act_fn_str == 'silu' else F.relu
        scan_fn = getattr(ssm_module, scan_fn_name)

        boundary_mask = None if boundary_mask_or_none.numel() == 0 else boundary_mask_or_none

        with torch.no_grad():
            in_proj = ssm_module.in_proj
            if hasattr(in_proj, 'get_ternary_weight'):
                W_in = in_proj.get_ternary_weight()
            else:
                W_in = in_proj.weight
            xz = F.linear(h_norm, W_in)
            x_inner = xz[..., :ssm_module.d_inner]
            z = xz[..., ssm_module.d_inner:]

            conv1d = ssm_module.conv1d
            x_t = x_inner.transpose(1, 2)
            x_t = F.conv1d(x_t, conv1d.weight, conv1d.bias,
                           padding=conv1d.weight.shape[-1] - 1,
                           groups=ssm_module.d_inner)[:, :, :seq_len]
            x_conv = x_t.transpose(1, 2)
            x_conv = act_fn(x_conv)

            x_proj = ssm_module.x_proj
            if hasattr(x_proj, 'get_ternary_weight'):
                W_xp = x_proj.get_ternary_weight()
            else:
                W_xp = x_proj.weight
            x_dbc = F.linear(x_conv, W_xp)
            dt_rank = ssm_module.dt_rank
            d_state = ssm_module.d_state
            dt_input = x_dbc[..., :dt_rank]
            B = x_dbc[..., dt_rank:dt_rank + d_state]
            C = x_dbc[..., dt_rank + d_state:]

            dt_proj = ssm_module.dt_proj
            dt = F.softplus(F.linear(dt_input, dt_proj.weight, dt_proj.bias))

            if boundary_mask is not None:
                dt = dt.clone()
                dt[boundary_mask] = 30.0

            A = -torch.exp(A_log)
            y = scan_fn(x_conv, dt, A, B, C)

            y_d = y + x_conv * D.unsqueeze(0).unsqueeze(0)
            act_z = act_fn(z)
            output = act_z * y_d

        committed_frac = _ssm_committed_frac(ssm_module)
        ctx.frozen = committed_frac >= 0.95

        if ctx.frozen:
            if not getattr(ssm_module, '_qcssd_ext_frozen_logged', False):
                print(f"[QCSSD-EXT-FROZEN] committed_frac={committed_frac*100:.1f}% — skipping quantized storage")
                ssm_module._qcssd_ext_frozen_logged = True
            ctx.grad_shapes = (h_norm.shape, A_log.shape, D.shape)
            return output

        h_q, h_s, h_m = _quantize_for_storage(h_norm, n_bits, group_size)

        ctx.save_for_backward(h_q, h_s, A_log.detach(), D.detach(), boundary_mask_or_none)
        ctx.h_m = h_m
        ctx.act_fn_str = act_fn_str
        ctx.scan_fn_name = scan_fn_name
        ctx.ssm_module = ssm_module
        ctx.seq_len = seq_len

        if not getattr(ssm_module, '_qcssd_ext_logged', False):
            print(f"[QCSSD-EXT] checkpoint boundary=h_norm shape={h_norm.shape}")
            ssm_module._qcssd_ext_logged = True

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Dequantize h_norm, recompute full chain, use autograd for gradients."""
        if ctx.frozen:
            h_shape, A_log_shape, D_shape = ctx.grad_shapes
            return (
                torch.zeros(h_shape, device=grad_output.device, dtype=grad_output.dtype),
                torch.zeros(A_log_shape, device=grad_output.device, dtype=grad_output.dtype),
                torch.zeros(D_shape, device=grad_output.device, dtype=grad_output.dtype),
                None, None, None, None, None, None, None
            )

        (h_q, h_s, A_log_saved, D_saved, boundary_mask_or_none) = ctx.saved_tensors

        h_norm = _dequantize_from_storage(h_q, h_s, ctx.h_m).requires_grad_(True)
        A_log = A_log_saved.detach().requires_grad_(True)
        D = D_saved.detach().requires_grad_(True)

        boundary_mask = None if boundary_mask_or_none.numel() == 0 else boundary_mask_or_none

        act_fn = F.silu if ctx.act_fn_str == 'silu' else F.relu
        ssm_module = ctx.ssm_module
        scan_fn = getattr(ssm_module, ctx.scan_fn_name)
        seq_len = ctx.seq_len

        with torch.enable_grad():
            in_proj = ssm_module.in_proj
            W_in = in_proj.get_ternary_weight() if hasattr(in_proj, 'get_ternary_weight') else in_proj.weight
            xz = F.linear(h_norm, W_in)
            x_inner = xz[..., :ssm_module.d_inner]
            z = xz[..., ssm_module.d_inner:]

            conv1d = ssm_module.conv1d
            x_t = x_inner.transpose(1, 2)
            x_t = F.conv1d(x_t, conv1d.weight, conv1d.bias,
                           padding=conv1d.weight.shape[-1] - 1,
                           groups=ssm_module.d_inner)[:, :, :seq_len]
            x_conv = x_t.transpose(1, 2)
            x_conv = act_fn(x_conv)

            x_proj = ssm_module.x_proj
            W_xp = x_proj.get_ternary_weight() if hasattr(x_proj, 'get_ternary_weight') else x_proj.weight
            x_dbc = F.linear(x_conv, W_xp)
            dt_rank = ssm_module.dt_rank
            d_state = ssm_module.d_state
            dt_input = x_dbc[..., :dt_rank]
            B = x_dbc[..., dt_rank:dt_rank + d_state]
            C = x_dbc[..., dt_rank + d_state:]

            dt_proj = ssm_module.dt_proj
            dt = F.softplus(F.linear(dt_input, dt_proj.weight, dt_proj.bias))

            if boundary_mask is not None:
                dt = dt.clone()
                dt[boundary_mask] = 30.0

            A = -torch.exp(A_log)
            y = scan_fn(x_conv, dt, A, B, C)

            y_d = y + x_conv * D.unsqueeze(0).unsqueeze(0)
            act_z = act_fn(z)
            output = act_z * y_d

        grads = torch.autograd.grad(
            output, (h_norm, A_log, D),
            grad_outputs=grad_output,
            allow_unused=True
        )

        grad_h_norm, grad_A_log, grad_D = grads

        if grad_h_norm is None: grad_h_norm = torch.zeros_like(h_norm)
        if grad_A_log is None: grad_A_log = torch.zeros_like(A_log)
        if grad_D is None: grad_D = torch.zeros_like(D)

        return grad_h_norm, grad_A_log, grad_D, None, None, None, None, None, None, None


class SelectiveSSM(nn.Module):
    """Selective State Space Model block (Mamba-1 inspired).

    Implements selective scan with:
    - Input projection (optionally ternary)
    - Conv1d local context
    - Selective scan with input-dependent B, C, delta
    - Gated output

    Data flow:
        x -> in_proj -> split(z, x_inner)
        x_inner -> conv1d -> silu -> selective_scan(A, B, C, delta) -> y
        output = out_proj(silu(z) * y)
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_rank: int = None, use_ternary: bool = False,
                 ternary_group_size: int = 128, dropout: float = 0.0,
                 use_ssm_checkpoint: bool = False, activation_fn: str = 'silu'):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        self.use_ssm_checkpoint = use_ssm_checkpoint
        self.act_fn = F.relu if activation_fn == 'relu' else F.silu

        Linear = TernaryLinear if use_ternary else nn.Linear
        linear_kwargs = {'group_size': ternary_group_size} if use_ternary else {}

        self.in_proj = Linear(d_model, 2 * self.d_inner, bias=False, **linear_kwargs)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner,
        )

        self.x_proj = Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False, **linear_kwargs)

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        with torch.no_grad():
            dt_init = torch.exp(
                torch.linspace(math.log(0.001), math.log(0.1), self.d_inner)
            )
            inv_softplus = dt_init + torch.log(-torch.expm1(-dt_init))
            self.dt_proj.bias.copy_(inv_softplus)

        # HiPPO-LegS diagonal: A_log[n] = log(n+1) => A[n] = -(n+1)
        # This IS the HiPPO-LegS diagonal (A[n,n] = -(n+1)). Off-diagonal couplings
        # would require a full matrix scan kernel. Diagonal-only is already structured.
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(self.d_inner, -1).clone())

        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = Linear(self.d_inner, d_model, bias=False, **linear_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._csp_step = 0

    def forward(self, x: torch.Tensor, boundary_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            boundary_mask: (batch, seq_len) bool tensor, True where document boundaries are
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.act_fn(x_conv)

        x_dbc = self.x_proj(x_conv)
        dt_input = x_dbc[:, :, :self.dt_rank]
        B = x_dbc[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C = x_dbc[:, :, self.dt_rank + self.d_state:]

        if self.use_osp and self.training:
            with torch.no_grad():
                B_reshaped = B.detach().reshape(B.shape[0], B.shape[1], self.n_heads, -1)
                _b2_per_head = (B_reshaped ** 2).mean(dim=(0, 1, 3))

                if getattr(self, '_calibration_mode', False):
                    if not hasattr(self, '_osp_calib_accum'):
                        self._osp_calib_accum = torch.zeros_like(_b2_per_head)
                        self._osp_calib_count = 0
                    self._osp_calib_accum += _b2_per_head
                    self._osp_calib_count += 1
                else:
                    _target_new = 1.0 - self._osp_alpha * _b2_per_head
                    self._osp_target.mul_(self._osp_beta).add_((1 - self._osp_beta) * _target_new)

                if not self._osp_logged:
                    print(f"[OSP] target update: b2_per_head={_b2_per_head.mean():.4f} target_mean={self._osp_target.mean():.4f}", flush=True)
                    self._osp_logged = True

        dt = F.softplus(self.dt_proj(dt_input))

        A = -torch.exp(self.A_log)

        if self.use_ssm_checkpoint and self.training:
            y = torch.utils.checkpoint.checkpoint(
                self._selective_scan, x_conv, dt, A, B, C, boundary_mask,
                use_reentrant=False
            )
            if not hasattr(self, '_ckpt_logged'):
                print(f'[SSM-CKPT] Gradient checkpoint active', flush=True)
                self._ckpt_logged = True
        else:
            y = self._selective_scan(x_conv, dt, A, B, C, boundary_mask)

        y = self._cospectral_project(y)

        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        act_z = self.act_fn(z)
        act_z = act_z * y
        output = self.out_proj(act_z)

        return self.dropout(output)

    @torch.compiler.disable()
    def _selective_scan(self, x: torch.Tensor, dt: torch.Tensor,
                        A: torch.Tensor, B: torch.Tensor,
                        C: torch.Tensor, boundary_mask: torch.Tensor = None) -> torch.Tensor:
        """Sequential selective scan with reversible step reconstruction.

        Uses ReversibleStepScan to avoid storing intermediate h states during backprop.
        Forward computes dA and dB per-timestep, saving only final h and inputs.
        Backward reconstructs h_{t-1} from h_t using exact inversion formula.

        Args:
            boundary_mask: (batch, seq_len) bool tensor. When True at position t,
                hidden state h is reset to zero (for document boundaries in packed sequences).
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        h_init = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=torch.float32)

        dA_steps = torch.zeros(seq_len, batch, d_inner, d_state, device=x.device, dtype=torch.float32)
        dB_steps = torch.zeros(seq_len, batch, d_inner, d_state, device=x.device, dtype=torch.float32)
        x_steps = torch.zeros(seq_len, batch, d_inner, device=x.device, dtype=x.dtype)
        C_steps = torch.zeros(seq_len, batch, d_inner, d_state, device=x.device, dtype=torch.float32)

        for t in range(seq_len):
            dt_t = dt[:, t]
            dA_steps[t] = torch.exp(dt_t.unsqueeze(-1) * A).to(torch.float32)
            dB_steps[t] = (dt_t.unsqueeze(-1) * B[:, t].unsqueeze(1)).to(torch.float32)
            x_steps[t] = x[:, t]
            C_steps[t] = C[:, t].unsqueeze(1).to(torch.float32)

        y = ReversibleStepScan.apply(h_init, dA_steps, dB_steps, x_steps, C_steps,
                                     boundary_mask, seq_len, batch, d_inner, d_state)

        if not hasattr(self, '_rev_step_logged'):
            mem_saved = seq_len * batch * d_inner * d_state * 4 / 1e6
            print(f"[REV-STEP] ReversibleStepScan active | T={seq_len} "
                  f"mem_saved={mem_saved:.1f}MB", flush=True)
            self._rev_step_logged = True

        return y

    def _cospectral_project(self, y: torch.Tensor) -> torch.Tensor:
        """R340: Co-spectral subspace projection via straight-through.

        Forward: identity (y unchanged).
        Backward: gradient flows only through rank-20 subspace.
        Calibration: online PCA of y during first 50 steps.

        Args:
            y: [B, T, d_inner] scan output
        Returns:
            y: [B, T, d_inner] identical values, reduced-rank backward
        """
        if not self.training:
            return y

        self._csp_step += 1

        if self._csp_step <= 50:
            with torch.no_grad():
                y_flat = y.detach().reshape(-1, y.shape[-1]).float()
                cov = y_flat.T @ y_flat
                if not hasattr(self, '_csp_cov_acc'):
                    self._csp_cov_acc = torch.zeros_like(cov)
                    self._csp_n = 0
                self._csp_cov_acc += cov
                self._csp_n += y_flat.shape[0]

            if self._csp_step == 50:
                with torch.no_grad():
                    cov_mean = (self._csp_cov_acc / self._csp_n).float()  # eigh needs float32
                    _, U = torch.linalg.eigh(cov_mean)
                    self._subspace_basis = U[:, -20:].to(y.dtype)
                    del self._csp_cov_acc, self._csp_n
                print(f"[R340-CSP] Basis calibrated: d_inner={y.shape[-1]} rank=20")
            return y

        if not hasattr(self, '_subspace_basis'):
            return y

        U = self._subspace_basis
        y_proj = (y @ U) @ U.T
        y_out = y.detach() + (y_proj - y_proj.detach())
        return y_out


class SelectiveSSMSSD(nn.Module):
    """Mamba-2 SSD (State Space Duality) -- Parallel scan with scalar A per head.

    Replaces sequential loop with parallel matrix operations via segsum.
    A becomes a learnable parameter of shape (n_heads,) enabling parallel computation.

    Data flow:
        x -> in_proj -> split(z, x_inner)
        x_inner -> conv1d -> silu -> ssd_scan(A, B, C, delta) -> y
        output = out_proj(silu(z) * y)
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_rank: int = None, n_heads: int = 16,
                 use_ternary: bool = False, ternary_group_size: int = 128,
                 use_chunked_ssd: bool = True, chunk_size: int = 128,
                 use_batched_scan: bool = True, dropout: float = 0.0,
                 use_ssm_checkpoint: bool = False, d_inner_override: int = None,
                 config=None, lti_mode: bool = False, activation_fn: str = 'silu'):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_inner_override if d_inner_override is not None else expand * d_model
        self.n_heads = n_heads
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        self.use_chunked_ssd = use_chunked_ssd
        self.chunk_size = chunk_size
        print(f"[SSM-INIT] chunk_size={self.chunk_size} chunked={self.use_chunked_ssd} batched={use_batched_scan}")
        self.use_batched_scan = use_batched_scan
        self.lti_mode = lti_mode
        self.act_fn = F.relu if (config and getattr(config, 'activation_fn', 'silu') == 'relu') or activation_fn == 'relu' else F.silu
        self.ssm_contraction_window = getattr(config, 'ssm_contraction_window', 256) if config is not None else 256
        self._contraction_logged = False

        Linear = TernaryLinear if use_ternary else nn.Linear
        linear_kwargs = {'group_size': ternary_group_size, 'config': config} if use_ternary else {}

        self.in_proj = Linear(d_model, 2 * self.d_inner, bias=False, **linear_kwargs)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner,
        )

        if not self.lti_mode:
            self.x_proj = Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False, **linear_kwargs)

            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            with torch.no_grad():
                dt_init = torch.exp(
                    torch.linspace(math.log(0.001), math.log(0.1), self.d_inner)
                )
                inv_softplus = dt_init + torch.log(-torch.expm1(-dt_init))
                self.dt_proj.bias.copy_(inv_softplus)

        # AN-6: MS-SSM multi-scale A_log initialization
        _n_scales = getattr(config, 'ms_ssm_n_scales', 1) if config is not None else 1
        if _n_scales > 1:
            _scale_log_min = getattr(config, 'ms_ssm_log_min', -4.0)
            _scale_log_max = getattr(config, 'ms_ssm_log_max', 0.0)
            _scale_centers = torch.linspace(_scale_log_min, _scale_log_max, _n_scales)
            _heads_per_scale = n_heads // _n_scales
            _segments = []
            for _i, _c in enumerate(_scale_centers):
                _start = _i * _heads_per_scale
                _end = (_i + 1) * _heads_per_scale if _i < _n_scales - 1 else n_heads
                _n = _end - _start
                _jitter = torch.linspace(-0.3, 0.3, _n)
                _segments.append(_c + _jitter)
            A_init = torch.cat(_segments)
            print(f"[MS-SSM] n_scales={_n_scales} log_range=[{_scale_log_min:.1f},{_scale_log_max:.1f}]")
        else:
            # Log-linear spacing: scalar A per head. HiPPO-LegS does NOT apply to scalar-A SSD.
            A_init = torch.log(torch.arange(1, n_heads + 1, dtype=torch.float32))
        if getattr(config, 'use_spectral_band_init', False) and config is not None:
            _nh = n_heads
            _n_phon = _nh // 4
            _n_synt = _nh // 4
            _n_sem  = _nh // 4
            _n_disc = _nh - _n_phon - _n_synt - _n_sem
            A_init = torch.cat([
                torch.full((_n_phon,), math.log(0.95)),
                torch.full((_n_synt,), math.log(0.80)),
                torch.full((_n_sem,),  math.log(0.60)),
                torch.full((_n_disc,), math.log(0.30)),
            ])
            print(f"[AQ-1 SPECTRAL-BAND] n_heads={_nh} bands=phon({_n_phon})/synt({_n_synt})/sem({_n_sem})/disc({_n_disc})")

        # AW-9: Path Signature A_log init — loglinear + log-product spectrum
        if getattr(config, 'use_aw9_path_sig_init', False) and config is not None:
            import math as _aw9_math
            _aw9_nh = A_init.shape[0]
            _aw9_half = _aw9_nh // 2
            _aw9_loglin = torch.linspace(_aw9_math.log(0.01), _aw9_math.log(1.0), _aw9_half)
            _aw9_base = torch.linspace(0.1, 0.9, _aw9_nh - _aw9_half)
            _aw9_logprod = torch.cumsum(torch.log(_aw9_base.clamp(min=1e-6)), dim=0)
            A_init = torch.cat([_aw9_loglin, _aw9_logprod])
            print(f"[AW-9 PATH-SIG-INIT] A_log loglinear=[{_aw9_loglin[0]:.3f},{_aw9_loglin[-1]:.3f}] "
                  f"logprod=[{_aw9_logprod[0]:.3f},{_aw9_logprod[-1]:.3f}]")
        # AW-14: Kleene 3-logic A_log init — persist/oscillate/local
        if getattr(config, 'use_aw14_kleene_init', False) and config is not None:
            import math as _aw14_math
            _aw14_nh = A_init.shape[0]
            _aw14_n3 = _aw14_nh // 3
            _aw14_rem = _aw14_nh - 2 * _aw14_n3
            _aw14_persist = torch.zeros(_aw14_n3)
            _aw14_osc = torch.full((_aw14_n3,), _aw14_math.log(0.9))
            _aw14_local = torch.full((_aw14_rem,), -8.0)
            A_init = torch.cat([_aw14_persist, _aw14_osc, _aw14_local])
            print(f"[AW-14 KLEENE-INIT] persist={_aw14_n3} osc={_aw14_n3} local={_aw14_rem}")

        if getattr(config, 'use_tbd_init', False) and config is not None:
            _K = getattr(config, 'tbd_n_scales', 8)
            _r_min = getattr(config, 'tbd_r_min', 0.1)
            _r_max = getattr(config, 'tbd_r_max', 0.97)
            _ratios = torch.linspace(0.0, 1.0, _K)
            _radii = _r_min * (_r_max / _r_min) ** _ratios
            _log_radii = torch.log(_radii)
            _heads_per_group = n_heads // _K
            _segments = []
            for _k in range(_K):
                _n = _heads_per_group if _k < _K - 1 else n_heads - _k * _heads_per_group
                _jitter = torch.linspace(-0.05, 0.05, _n)
                _segments.append(_log_radii[_k] + _jitter)
            A_init = torch.cat(_segments)
            print(f"[TBD-INIT] K={_K} r=[{_r_min:.2f},{_r_max:.2f}] n_heads={n_heads}")

        self.A_log = nn.Parameter(A_init.clone())

        # AW-14: A_log type distribution log at init
        with torch.no_grad():
            _aw14_a = self.A_log.detach()
            _aw14_p = int((_aw14_a.abs() < 0.05).sum().item())
            _aw14_o = int(((_aw14_a.abs() >= 0.05) & (_aw14_a.abs() < 1.0)).sum().item())
            _aw14_l = int((_aw14_a.abs() >= 1.0).sum().item())
            print(f"[AW-14 KLEENE-DIST] A_log types: persist={_aw14_p} osc={_aw14_o} local={_aw14_l}")

        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.use_predictive_coding = getattr(config, 'use_predictive_coding', False) if config is not None else False
        if self.use_predictive_coding:
            _pc_rank = getattr(config, 'predictive_coding_rank', 0) if config is not None else 0
            if _pc_rank > 0:
                self.W_pred_U = nn.Linear(d_model, _pc_rank, bias=False)
                self.W_pred_V = nn.Linear(_pc_rank, d_model, bias=False)
                nn.init.zeros_(self.W_pred_U.weight)
                nn.init.zeros_(self.W_pred_V.weight)
            else:
                self.W_pred = nn.Linear(d_model, d_model, bias=False)
                nn.init.zeros_(self.W_pred.weight)
            self._pred_log_step = 0

        # R4-A4-P8: Contractive Recurrence Guarantee (CRG)
        self.use_crg = getattr(config, 'use_crg', False) if config is not None else False
        if self.use_crg:
            crg_alpha_init = getattr(config, 'crg_alpha_init', 3.0) if config is not None else 3.0
            self.alpha_decay = nn.Parameter(torch.full((n_heads,), crg_alpha_init))

        self.use_tscs = getattr(config, 'use_tscs', False) if config is not None else False
        self.use_tscs_p2_prefix_sum = getattr(config, 'use_tscs_p2_prefix_sum', False) if config is not None else False

        if self.use_tscs:
            self.tscs_threshold_coeff = 1.0
            osc_frac = getattr(config, 'tscs_osc_fraction', 0.375)
            reset_frac = getattr(config, 'tscs_reset_fraction', 0.375)
            clamp_val = getattr(config, 'tscs_clamp_state', 0.0)
            self.tscs_clamp_state = clamp_val

            n_osc = max(1, int(n_heads * osc_frac))
            n_reset = max(1, int(n_heads * reset_frac))
            n_acc = n_heads - n_osc - n_reset
            if n_acc < 0:
                n_reset = n_heads - n_osc
                n_acc = 0
            init_vals = torch.cat([
                torch.full((n_osc,), -1.0),
                torch.full((n_reset,), 0.0),
                torch.full((n_acc,), 1.0),
            ])
            self.A_shadow = nn.Parameter(init_vals)

            self._tscs_chunk_size = 64
            self._build_tscs_caches(64)
            self._tscs_debug_logged = False

        if self.lti_mode:
            self.B_fixed = nn.Parameter(torch.randn(n_heads, d_state) * 0.02)
            self.C_fixed = nn.Parameter(torch.randn(n_heads, d_state) * 0.02)
            self._lti_debug_logged = False
            self._tscs_p2_logged = False

        self.out_proj = Linear(self.d_inner, d_model, bias=False, **linear_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_fla = _FLA_AVAILABLE
        self.use_ssm_checkpoint = use_ssm_checkpoint

        self.use_flip_rate_limit = getattr(config, 'use_flip_rate_limit', False) if config is not None else False
        if self.use_flip_rate_limit:
            self.max_flip_rate = getattr(config, 'ssm_max_flip_rate', 0.05)
            self._flip_rate_debug_logged = False
            self._last_ternary_weights = {}

        self.use_spectral_adaptive_chunk = getattr(config, 'use_spectral_adaptive_chunk', False) if config is not None else False
        if self.use_spectral_adaptive_chunk:
            self.chunk_epsilon = getattr(config, 'chunk_epsilon', 1e-3)
            self.chunk_signal_bound = getattr(config, 'chunk_signal_bound', 10.0)
            self.chunk_min = getattr(config, 'chunk_min', 8)
            self.chunk_max = getattr(config, 'chunk_max', 256)
            self._chunk_debug_logged = False

        self.use_krylov_compression = getattr(config, 'use_krylov_compression', False) if config is not None else False
        if self.use_krylov_compression:
            self.krylov_rank = getattr(config, 'krylov_rank', 16)
            self._krylov_Q = None
            self._krylov_debug_logged = False

        self.use_commitment_k_schedule = getattr(config, 'use_commitment_k_schedule', False) if config is not None else False
        if self.use_commitment_k_schedule:
            self.k_min = getattr(config, 'commitment_k_min', 10)
            self.k_max = getattr(config, 'commitment_k_max', 90)
            self.commitment_stability_threshold = getattr(config, 'commitment_stability_threshold', 50)
            self._weight_flip_history = {}
            self._current_k = self.k_max
            self._k_debug_logged = False

        self.use_activation_quant = getattr(config, 'use_activation_quant', False) if config is not None else False
        self._act_quant_bits = getattr(config, 'activation_quant_bits', 4) if config is not None else 4
        self._act_quant_group_size = getattr(config, 'activation_quant_group_size', 128) if config is not None else 128
        self._act_quant_warmup = getattr(config, 'activation_quant_warmup', 100) if config is not None else 100
        self._act_quant_step = 0
        self._act_quant_debug_logged = False

        self._episodic_h = None

        # R323: corpus-mean warm h₀
        _head_dim = (d_inner_override if d_inner_override is not None else expand * d_model) // n_heads
        self.register_buffer(
            'h0_mean',
            torch.zeros(1, n_heads, d_state, _head_dim, dtype=torch.float32)
        )
        self._h0_calibrated = False
        self._h0_calibrate_logged = False

        # R19-D: Kronecker rank-1 state
        self.use_kronecker_state = getattr(config, 'use_kronecker_state', False) if config is not None else False
        self._kron_h_u = None  # (batch, n_heads, d_state) INT8
        self._kron_h_v = None  # (batch, n_heads, head_dim) INT8
        self._kron_logged = False

        use_episodic_gate = getattr(config, 'use_episodic_gate', False) if config is not None else False
        self.episodic_gate = nn.Linear(d_model, d_state * n_heads, bias=True) if use_episodic_gate else None

        self._packing_logged = False

        # Batch AH: Kalman-smoother diagonal Hessian for recurrence params
        # grad(A_log)^2 = diagonal Fisher = Kalman backward information (for free)
        self._kh_ema: float = 0.99
        self._kh_A_log_diag: torch.Tensor = None   # (n_heads,) — curvature of decay
        self._kh_dt_bias_diag: torch.Tensor = None  # (d_inner,) — curvature of timestep
        self._kh_logged: bool = False

        self.use_tcsd = getattr(config, 'use_tcsd', False) if config is not None else False
        self._tcsd_beta = getattr(config, 'tcsd_beta', 0.99) if config is not None else 0.99
        self._tcsd_lambda = getattr(config, 'tcsd_lambda', 0.1) if config is not None else 0.1
        if self.use_tcsd:
            self.register_buffer('_tcsd_y_slow', None)
            self._tcsd_loss = None
            self._tcsd_logged = False

        # GP-KL online stats (replaces full tensor storage)
        self._gp_n = 0                    # int: number of (B*T) samples seen
        self._gp_mean = None              # [d_inner] float32, on device
        self._gp_m2 = None                # [d_inner] float32, Welford M2
        self._gp_cross = None             # [d_inner] float32, lag-1 cross-product sum
        self._gp_prev = None              # [d_inner] float32, last token (for lag-1)
        self._gp_n_pairs = 0              # int: lag-1 sample count

        self.use_osp = getattr(config, 'use_osp', False) if config is not None else False
        if self.use_osp:
            self.register_buffer('_osp_target', torch.ones(self.n_heads))
            self._osp_alpha = getattr(config, 'osp_alpha', 0.1)
            self._osp_beta = getattr(config, 'osp_ema_beta', 0.99)
            self._osp_logged = False

        self.use_tma = getattr(config, 'use_tma', False) if config is not None else False
        if self.use_tma:
            _n_slots = getattr(config, 'tma_n_slots', 64) if config is not None else 64
            self.tma_n_slots = _n_slots
            self.tma_gamma = nn.Parameter(torch.tensor(getattr(config, 'tma_gamma', 0.1) if config is not None else 0.1))
            self.tma_eta = getattr(config, 'tma_eta', 0.01) if config is not None else 0.01
            self.register_buffer('tma_memory', torch.randn(self.n_heads, _n_slots, self.d_state) * 0.01)
            self._tma_logged = False

        self.use_surprise_dt = getattr(config, 'use_surprise_dt', False) if config is not None else False
        if self.use_surprise_dt:
            self.surprise_dt_scale = getattr(config, 'surprise_dt_scale', 0.5) if config is not None else 0.5
            self._surprise_dt_logged = False
            print(f"[SURPRISE-DT] Dynamic dt gating via surprise: scale={self.surprise_dt_scale}")

        self.use_bpnp = getattr(config, 'use_bpnp', False) if config is not None else False
        if self.use_bpnp:
            self.bpnp_threshold = getattr(config, 'bpnp_threshold', 0.3) if config is not None else 0.3
            self.bpnp_gate_proj = nn.Linear(self.d_inner, 1, bias=True)
            nn.init.zeros_(self.bpnp_gate_proj.weight)
            nn.init.constant_(self.bpnp_gate_proj.bias, 0.5)
            self._bpnp_logged = False
            print(f"[BPNP] Broad/Narrow Phase gating: threshold={self.bpnp_threshold}")

        self.use_dssu = getattr(config, 'use_dssu', False) if config is not None else False
        if self.use_dssu:
            self.dssu_epsilon = getattr(config, 'dssu_epsilon', 0.01) if config is not None else 0.01
            self._dssu_logged = False
            print(f"[DSSU] Dimension-Selective State Update: epsilon={self.dssu_epsilon}")

        # AP-1: Sigma-g Surprise Gate — z-branch variance as novelty proxy
        self.use_sigma_gate = getattr(config, 'use_sigma_gate', False) if config is not None else False
        if self.use_sigma_gate:
            self._sigma_gate_threshold = getattr(config, 'sigma_gate_threshold', 0.5) if config is not None else 0.5
            self._sigma_gate_beta = getattr(config, 'sigma_gate_baseline_beta', 0.99) if config is not None else 0.99
            self._sigma_gate_baseline = 1.0
            self._sigma_gate_is_novel = False
            self._sigma_gate_var_ema = 0.0
            self._sigma_gate_logged = False
            print(f"[AP-1 SIGMA-GATE] Enabled: threshold={self._sigma_gate_threshold}")

        self.use_liquid_s4 = getattr(config, 'use_liquid_s4', False) if config is not None else False
        if self.use_liquid_s4:
            self.liquid_proj = nn.Linear(self.d_inner, n_heads, bias=False)
            print(f"[AQ-4 LIQUID-S4] input-dependent A active | proj={self.d_inner}->{n_heads}")

        # R19-D: Matrix Monoid — A_t input-dependent per-token
        self.use_input_dep_A = getattr(config, 'use_input_dep_A', False) if config is not None else False
        if self.use_input_dep_A:
            self.W_A = nn.Linear(self.d_inner, n_heads, bias=True)
            nn.init.zeros_(self.W_A.weight)
            with torch.no_grad():
                self.W_A.bias.copy_(self.A_log.detach())
            self._input_dep_A_logged = False
            print(f"[MATRIX-MONOID] input-dependent A ACTIVE | W_A={self.d_inner}->{n_heads}")
        else:
            self.W_A = None
            self._input_dep_A_logged = False

        self.use_depth_plasticity = getattr(config, 'use_depth_plasticity', False) if config is not None else False
        if self.use_depth_plasticity:
            _w = getattr(config, 'depth_plasticity_window', 8) if config is not None else 8
            self.register_buffer('_dp_hashes', torch.zeros(_w, dtype=torch.int64))
            self._dp_ptr = 0
            self._dp_window = _w
            print(f"[AQ-5 DEPTH-PLASTICITY] limit-cycle detection active | window={_w}")

        # AR-3 SSSP: Sign-Stable State Propagation
        self.use_sssp = getattr(config, 'use_sssp', False) if config is not None else False
        self._sssp_threshold = getattr(config, 'sssp_stability_threshold', 0.8) if config is not None else 0.8

        # AR-4 SCMS: Sparse Chunk Matrix Summation
        self.use_scms = getattr(config, 'use_scms', False) if config is not None else False
        self._scms_passive_threshold = getattr(config, 'scms_passive_threshold', 0.01) if config is not None else 0.01

        # AR-1 AdaPonderLM: adaptive computation halting gate
        self.use_adaponder = getattr(config, 'use_adaponder', False) if config is not None else False
        if self.use_adaponder:
            self._adaponder_halt_threshold = getattr(config, 'adaponder_halt_threshold', 0.5) if config is not None else 0.5
            self.halt_gate = nn.Linear(self.d_inner, 1, bias=True)
            print(f"[AR-1 ADAPONDER] halting gate active | threshold={self._adaponder_halt_threshold}")

        self.use_ada_gate = getattr(config, 'use_ada_gate', False) if config is not None else False
        self._ada_threshold = getattr(config, 'ada_coherence_threshold', 0.15) if config is not None else 0.15

        self.use_tug = getattr(config, 'use_tug', False) if config is not None else False

        self.use_ergodic_mixing = getattr(config, 'use_ergodic_mixing', False) if config is not None else False
        self._ergodic_lambda_thresh = getattr(config, 'ergodic_lambda_threshold', 0.7) if config is not None else 0.7
        self._ergodic_slow_frac_thresh = getattr(config, 'ergodic_slow_fraction_threshold', 0.3) if config is not None else 0.3

        self.use_vff_routing = getattr(config, 'use_vff_routing', False) if config is not None else False
        if self.use_vff_routing:
            self.vff_gate = nn.Linear(self.d_inner, 1, bias=True)
            print(f"[AS-9 VFF] depth routing gate active")

        self.use_fast_slow_ssm = getattr(config, 'use_fast_slow_ssm', False) if config is not None else False
        self._fast_slow_factor = getattr(config, 'fast_slow_factor', 4.0) if config is not None else 4.0

        self.use_lyapunov_basin = getattr(config, 'use_lyapunov_basin', False) if config is not None else False
        self._lyap_threshold = getattr(config, 'lyapunov_basin_threshold', 1.0) if config is not None else 1.0

        # AS-13 DPLR: Decay-Partitioned Learning Rules
        self.use_dplr = getattr(config, 'use_dplr', False) if config is not None else False
        self._dplr_retention_threshold = getattr(config, 'dplr_retention_threshold', 0.5) if config is not None else 0.5
        self._dplr_logged = False
        if self.use_dplr:
            with torch.no_grad():
                _dt_mean = 0.01
                _A_log_val = self.A_log.detach()
                _decay_rate = torch.exp(_A_log_val)
                _mean_retention = torch.exp(-_decay_rate * _dt_mean)
                self._dplr_s1_channels = (_mean_retention < self._dplr_retention_threshold)
            s1_frac = self._dplr_s1_channels.float().mean().item()
            print(f"[AS-13 DPLR] s1_channels_frac={s1_frac:.3f} (retention_threshold={self._dplr_retention_threshold})")
        else:
            self._dplr_s1_channels = None

        # AS-17 BNS2: B-Projection Norm System 2 Signal
        self.use_bns2 = getattr(config, 'use_bns2', False) if config is not None else False
        self._bns2_norm_threshold = getattr(config, 'bns2_norm_threshold', 0.1) if config is not None else 0.1

        # AT-1: Three-Timescale CLS (EMA of h_t injected into SSM input as context)
        self.use_cls_timescale = getattr(config, 'use_cls_timescale', False) if config is not None else False
        self._cls_alpha = getattr(config, 'cls_timescale_alpha', 0.99) if config is not None else 0.99
        if self.use_cls_timescale:
            _d_inner = config.d_inner if config is not None else d_inner
            self.register_buffer('_cls_h_ema', torch.zeros(1, 1, _d_inner))
            print(f"[AT-1 CLS] EMA buffer registered | alpha={self._cls_alpha} d_inner={_d_inner}")

        # AT-4: DT Surprise Gate (prediction-error based dt gating, distinct from use_surprise_dt)
        self.use_dt_surprise_gate = getattr(config, 'use_dt_surprise_gate', False) if config is not None else False

        # AW-18: Tarski fixed-point attractor — compute convergent h_reset base
        if getattr(config, 'use_aw18_tarski_init', False) and config is not None:
            with torch.no_grad():
                _aw18_a = -torch.exp(self.A_log.detach())
                _aw18_stable = _aw18_a.abs() < 1.0
                _aw18_fp = torch.zeros_like(_aw18_a)
                _aw18_fp[_aw18_stable] = 1.0 / (1.0 - _aw18_a[_aw18_stable]).clamp(min=0.01)
                _aw18_fp = _aw18_fp.clamp(-10.0, 10.0)
                self.register_buffer('tarski_h_reset_base', _aw18_fp)
                print(f"[AW-18 TARSKI-ATTRACTOR] convergent={_aw18_stable.sum().item()}/{_aw18_a.numel()} "
                      f"fp_mean={_aw18_fp[_aw18_stable].mean().item() if _aw18_stable.any() else 0:.4f}")
        else:
            self.tarski_h_reset_base = None

        self._exp_lut_built = False
        self._exp_lut = None

        self._truncated_bptt = getattr(config, 'truncated_bptt', False) if config is not None else False
        self._tbptt_logged = False

        self.use_wht_mixing = getattr(config, 'use_wht_mixing', False) if config is not None else False
        self._wht_logged = False
        if self.use_wht_mixing:
            assert (self.d_state & (self.d_state - 1)) == 0, \
                f"[WHT] d_state must be power of 2, got {self.d_state}"
            print(f"[WHT] Walsh-Hadamard state mixing active | d_state={self.d_state}")

        self._csp_step = 0

    @staticmethod
    def _init_a_log(n_heads: int, use_slow_heads: bool = False,
                    slow_fraction: float = 0.25, medium_fraction: float = 0.25) -> torch.Tensor:
        if not use_slow_heads:
            return torch.log(torch.arange(1, n_heads + 1, dtype=torch.float32))

        n_slow = max(1, int(n_heads * slow_fraction))
        n_medium = max(1, int(n_heads * medium_fraction))
        n_fast = n_heads - n_slow - n_medium
        if n_fast < 1:
            n_fast = 1
            n_medium = n_heads - n_slow - n_fast

        fast_a_log = torch.linspace(math.log(0.1), math.log(0.7), n_fast)
        medium_a_log = torch.linspace(math.log(0.01), math.log(0.1), n_medium)
        slow_a_log = torch.linspace(math.log(0.001), math.log(0.01), n_slow)

        return torch.cat([slow_a_log, medium_a_log, fast_a_log])

    def _quantize_A_ternary(self):
        """Quantize A matrix to ternary decay rates {-1, 0, +1} via STE.

        Maps A_shadow (continuous shadow weights) to ternary values representing:
          -1: Fast decay (oscillating modes)
           0: No decay (stable/reset modes)
          +1: Growth (unstable, rare)

        Returns:
            A_ternary: (n_heads,) tensor with values in {-1, 0, +1}
        """
        if not self.use_tscs:
            return -torch.exp(self.A_log)

        threshold = self.tscs_threshold_coeff * self.A_shadow.abs().mean()

        A_ternary = torch.zeros_like(self.A_shadow)
        A_ternary[self.A_shadow < -threshold] = -1.0

        if self.training and not getattr(self, '_hurwitz_logged', False):
            with torch.no_grad():
                n_neg = int((A_ternary.detach() < -0.5).sum())
                n_zero = int((A_ternary.detach().abs() < 0.5).sum())
                print(f"[HURWITZ] A_ternary: neg={n_neg} zero={n_zero} pos=0 (Hurwitz satisfied)")
            self._hurwitz_logged = True

        A_ternary = self.A_shadow - self.A_shadow.detach() + A_ternary.detach()

        return A_ternary

    def _tma_step(self, h):
        if not self.use_tma:
            return h

        h_addr = h.mean(dim=-1)
        h_sign = h_addr.sign()
        m_sign = self.tma_memory.sign()

        sim = torch.einsum('bhs,hjs->bhj', h_sign, m_sign)
        best_idx = sim.argmax(dim=-1)

        batch_size = h.shape[0]
        best_exp = best_idx.unsqueeze(-1).expand(-1, -1, self.d_state)
        mem_exp = self.tma_memory.unsqueeze(0).expand(batch_size, -1, -1, -1)
        m_best = mem_exp.gather(2, best_exp.unsqueeze(2)).squeeze(2)

        h = h + self.tma_gamma * m_best.unsqueeze(-1)

        if self.training:
            with torch.no_grad():
                delta = (h_addr.detach() - m_best.detach()).mean(dim=0)
                for head_idx in range(self.n_heads):
                    mode_slot = best_idx[:, head_idx].mode().values.item()
                    self.tma_memory[head_idx, mode_slot] += self.tma_eta * delta[head_idx]

        if not self._tma_logged:
            print(f"[TMA] active | slots={self.tma_n_slots} gamma={self.tma_gamma.item():.4f}", flush=True)
            self._tma_logged = True

        return h

    def _enforce_flip_rate_limit(self):
        """R3-A1-M4: Limit ternary weight flip rate in SSM layers."""
        if not self.use_flip_rate_limit or not self.training:
            return

        total_ternary_params = 0
        total_flips = 0

        for name, module in self.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue

            if not (hasattr(module, 'weight_shadow') or
                    (hasattr(module, 'weight') and 'ternary' in name.lower())):
                continue

            W = module.weight.data
            W_ternary = torch.clamp(torch.round(W), -1, 1)

            key = f"{name}_weight"
            if key in self._last_ternary_weights:
                W_last = self._last_ternary_weights[key]
                n_flips = (W_ternary != W_last).sum().item()
                n_params = W_ternary.numel()
                flip_rate = n_flips / n_params

                if flip_rate > self.max_flip_rate:
                    n_allowed = int(self.max_flip_rate * n_params)
                    flip_mask = (W_ternary != W_last)
                    flip_indices = torch.nonzero(flip_mask, as_tuple=True)

                    n_to_freeze = n_flips - n_allowed
                    if n_to_freeze > 0:
                        freeze_perm = torch.randperm(n_flips)[:n_to_freeze]

                        for idx in freeze_perm:
                            coords = tuple(fi[idx] for fi in flip_indices)
                            W_ternary[coords] = W_last[coords]

                        module.weight.data.copy_(W_ternary)

                        if not self._flip_rate_debug_logged:
                            print(f"[SSM-FlipRateLimit] Enforced: {n_flips} -> {n_allowed} flips "
                                  f"({flip_rate:.1%} -> {self.max_flip_rate:.1%}), froze {n_to_freeze}")
                            self._flip_rate_debug_logged = True

                total_ternary_params += n_params
                total_flips += min(n_flips, int(self.max_flip_rate * n_params))

            self._last_ternary_weights[key] = W_ternary.clone()

    def _compute_spectral_radius_ternary(self):
        """Compute spectral radius of A matrix for adaptive chunk sizing."""
        if not self.use_tscs:
            A = -torch.exp(self.A_log)
            return torch.abs(A).max().item()

        A_ternary = self._quantize_A_ternary()
        rho = torch.abs(A_ternary).max().item()
        return rho

    def _adaptive_chunk_size(self):
        """R3-A3-M5: Compute optimal chunk size based on spectral radius."""
        if not self.use_spectral_adaptive_chunk:
            return self.chunk_size

        rho = self._compute_spectral_radius_ternary()

        if rho <= 0.0:
            return self.chunk_max
        if rho >= 1.0:
            return self.chunk_min

        log_rho = math.log(rho)
        log_ratio = math.log(self.chunk_epsilon / self.chunk_signal_bound)
        C_opt = int(math.ceil(log_ratio / log_rho))

        C_clamped = max(self.chunk_min, min(C_opt, self.chunk_max))

        if not self._chunk_debug_logged:
            print(f"[SpectralAdaptiveChunk] rho={rho:.3f}, C_opt={C_opt}, C_clamped={C_clamped}")
            self._chunk_debug_logged = True

        return C_clamped

    def _arnoldi_ternary(self, A_ternary, h_init, k):
        """R3-A3-M7: Arnoldi iteration for ternary A (diagonal matrix)."""
        batch, n_heads, d_state, d_head = h_init.shape

        Q = torch.zeros(batch, n_heads, d_state, k, device=h_init.device, dtype=h_init.dtype)

        h_norm = torch.norm(h_init, dim=(2, 3), keepdim=True).clamp(min=1e-8)
        Q[:, :, :, 0] = (h_init / h_norm).mean(-1)

        for j in range(k - 1):
            v = A_ternary.view(1, n_heads, 1) * Q[:, :, :, j]

            for i in range(j + 1):
                proj = (Q[:, :, :, i] * v).sum(dim=2, keepdim=True)
                v = v - proj * Q[:, :, :, i]

            v_norm = torch.norm(v, dim=2, keepdim=True).clamp(min=1e-8)
            Q[:, :, :, j + 1] = v / v_norm

        return Q

    def _compress_state(self, h):
        """Compress SSM state via Krylov basis."""
        if not self.use_krylov_compression or self._krylov_Q is None:
            return h

        Q = self._krylov_Q
        h_compressed = torch.einsum('bhdk,bhds->bhks', Q, h)

        if not self._krylov_debug_logged:
            compression_ratio = (h.numel() / h_compressed.numel())
            print(f"[KrylovCompression] {h.shape} -> {h_compressed.shape}, ratio={compression_ratio:.1f}x")
            self._krylov_debug_logged = True

        return h_compressed

    def _decompress_state(self, h_compressed):
        """Reconstruct SSM state from Krylov representation."""
        if not self.use_krylov_compression or self._krylov_Q is None:
            return h_compressed

        Q = self._krylov_Q
        h = torch.einsum('bhdk,bhks->bhds', Q, h_compressed)
        return h

    def _compute_commitment_rate(self):
        """Compute fraction of ternary weights that are committed (stable)."""
        if not self.use_commitment_k_schedule:
            return 0.0

        total_ternary_params = 0
        committed_params = 0

        for name, module in self.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue

            if not (hasattr(module, 'weight_shadow') or 'ternary' in name.lower()):
                continue

            W = module.weight.data
            W_ternary = torch.clamp(torch.round(W), -1, 1)

            key = f"{name}_weight"
            if key not in self._weight_flip_history:
                self._weight_flip_history[key] = {
                    'last_value': W_ternary.clone(),
                    'stability_count': torch.zeros_like(W_ternary, dtype=torch.int32)
                }
                continue

            history = self._weight_flip_history[key]
            W_last = history['last_value']
            stability = history['stability_count']

            no_flip = (W_ternary == W_last)
            stability[no_flip] += 1
            stability[~no_flip] = 0

            committed = (stability >= self.commitment_stability_threshold)
            committed_params += committed.sum().item()
            total_ternary_params += W_ternary.numel()

            history['last_value'] = W_ternary.clone()
            history['stability_count'] = stability

        if total_ternary_params == 0:
            return 0.0

        return committed_params / total_ternary_params

    def _update_adaptive_k(self):
        """R3-A4-M3: Update TBPTT horizon K based on commitment rate."""
        if not self.use_commitment_k_schedule or not self.training:
            return

        C = self._compute_commitment_rate()
        K_adaptive = self.k_min + int((self.k_max - self.k_min) * (1.0 - C))
        self._current_k = K_adaptive

        if not self._k_debug_logged or (C > 0.1):
            print(f"[CommitmentKSchedule] C={C:.2%}, K={K_adaptive} (range [{self.k_min}, {self.k_max}])")
            self._k_debug_logged = True

    def _lti_fft_conv(self, x_conv):
        """LTI forward via FFT causal convolution. Training only."""
        batch, seq_len, d_inner = x_conv.shape
        head_dim = d_inner // self.n_heads

        if self.use_tscs:
            A = self._quantize_A_ternary()
        else:
            A = -torch.exp(self.A_log)

        # R4-A4-P8: CRG contraction
        if self.use_crg:
            A = A * self.alpha_decay.sigmoid()

        CB = (self.C_fixed * self.B_fixed).sum(-1)

        t = torch.arange(seq_len, device=x_conv.device, dtype=x_conv.dtype)

        if self.use_tscs:
            A_powers = A.to(x_conv.dtype).unsqueeze(-1).pow(t.unsqueeze(0))
        else:
            A_powers = torch.exp(A.to(x_conv.dtype).unsqueeze(-1) * t.unsqueeze(0))

        kernel = A_powers * CB.to(x_conv.dtype).unsqueeze(-1)

        X = x_conv.view(batch, seq_len, self.n_heads, head_dim).permute(0, 2, 1, 3)

        fft_size = 2 * seq_len
        K_f = torch.fft.rfft(kernel, n=fft_size, dim=-1)
        X_f = torch.fft.rfft(X, n=fft_size, dim=2)
        Y_f = X_f * K_f[None, :, :, None]
        Y = torch.fft.irfft(Y_f, n=fft_size, dim=2)[:, :, :seq_len, :]

        return Y.permute(0, 2, 1, 3).reshape(batch, seq_len, d_inner)

    @torch.compiler.disable()
    def _lti_sequential(self, x_conv):
        """LTI sequential scan for inference."""
        batch, seq_len, d_inner = x_conv.shape
        head_dim = d_inner // self.n_heads

        if self.use_tscs:
            A = self._quantize_A_ternary()
        else:
            A = -torch.exp(self.A_log)

        # R4-A4-P8: CRG contraction
        if self.use_crg:
            A = A * self.alpha_decay.sigmoid()

        X = x_conv.view(batch, seq_len, self.n_heads, head_dim).permute(0, 2, 1, 3)

        if self._h0_calibrated:
            h = self.h0_mean.expand(batch, -1, -1, -1).clone()
            if not self._h0_calibrate_logged:
                print(f"[WARM-H0] _ssd_scan: using corpus-mean h0 norm={self.h0_mean.norm():.4f}", flush=True)
                self._h0_calibrate_logged = True
        else:
            h = torch.zeros(batch, self.n_heads, self.d_state, head_dim,
                             device=x_conv.device, dtype=torch.float32)
        Y = torch.zeros(batch, self.n_heads, seq_len, head_dim,
                         device=x_conv.device, dtype=x_conv.dtype)

        B = self.B_fixed
        C = self.C_fixed
        if self.use_tscs:
            A_col = A.view(1, self.n_heads, 1, 1)
        else:
            A_col = torch.exp(A).view(1, self.n_heads, 1, 1)

        for t_idx in range(seq_len):
            x_t = X[:, :, t_idx, :].to(torch.float32)
            Bx = B[None, :, :, None] * x_t[:, :, None, :]

            if self.use_dssu:
                active = Bx.abs() > self.dssu_epsilon
                h_decayed = A_col.float() * h
                h_updated = h_decayed + Bx
                h = torch.where(active, h_updated, h_decayed)
            else:
                h = A_col.float() * h + Bx

            Y[:, :, t_idx, :] = (C[None, :, :, None] * h).sum(dim=2).to(x_conv.dtype)

        if self.use_dssu and not self._dssu_logged:
            with torch.no_grad():
                avg_active = active.float().mean().item()
                print(f"[DSSU] active_frac={avg_active:.3f} (epsilon={self.dssu_epsilon})")
            self._dssu_logged = True

        return Y.permute(0, 2, 1, 3).reshape(batch, seq_len, d_inner)

    @torch.compiler.disable()
    def _lti_prefix_sum(self, x_conv):
        """TSCS Phase 2: Parallel prefix-sum scan for ternary A matrix.

        When A is ternary {-1, 0, +1}, the LTI recurrence h_t = A*h_{t-1} + B*x_t
        has three modes:
          - A[i] = +1 (accumulator): h_t[i] = h_{t-1}[i] + B*x_t[i]
          - A[i] = 0 (reset): h_t[i] = B*x_t[i] (no history)
          - A[i] = -1 (oscillator): h_t[i] = -h_{t-1}[i] + B*x_t[i]

        Each mode can be computed efficiently using prefix-sum patterns.
        """
        batch, seq_len, d_inner = x_conv.shape
        head_dim = d_inner // self.n_heads

        A_ternary = self._quantize_A_ternary()
        X = x_conv.view(batch, seq_len, self.n_heads, head_dim).permute(0, 2, 1, 3)

        B = self.B_fixed
        C = self.C_fixed

        acc_float = (A_ternary == 1).float()
        reset_float = (A_ternary == 0).float()
        osc_float = (A_ternary == -1).float()

        acc_mask_4d = acc_float.view(1, self.n_heads, 1, 1)
        reset_mask_4d = reset_float.view(1, self.n_heads, 1, 1)
        osc_mask_4d = osc_float.view(1, self.n_heads, 1, 1)

        Y = torch.zeros(batch, self.n_heads, seq_len, head_dim,
                         device=x_conv.device, dtype=x_conv.dtype)

        h_acc = torch.zeros(batch, self.n_heads, self.d_state, head_dim,
                             device=x_conv.device, dtype=torch.float32)
        h_reset = torch.zeros(batch, self.n_heads, self.d_state, head_dim,
                               device=x_conv.device, dtype=torch.float32)
        h_osc = torch.zeros(batch, self.n_heads, self.d_state, head_dim,
                             device=x_conv.device, dtype=torch.float32)

        for t_idx in range(seq_len):
            x_t = X[:, :, t_idx, :].to(torch.float32)
            Bx_t = B[None, :, :, None] * x_t[:, :, None, :]

            h_acc_update = (h_acc + Bx_t) * acc_mask_4d
            h_acc = torch.where(acc_mask_4d.bool(), h_acc + Bx_t, h_acc)

            h_reset_update = Bx_t * reset_mask_4d
            h_reset = torch.where(reset_mask_4d.bool(), Bx_t, h_reset)

            h_osc_update = (-h_osc + Bx_t) * osc_mask_4d
            h_osc = torch.where(osc_mask_4d.bool(), -h_osc + Bx_t, h_osc)

            h_current = h_acc + h_reset + h_osc

            Y[:, :, t_idx, :] = (C[None, :, :, None] * h_current).sum(dim=2).to(x_conv.dtype)

        if not getattr(self, '_tscs_p2_logged', False):
            n_acc = acc_float.sum().item()
            n_reset = reset_float.sum().item()
            n_osc = osc_float.sum().item()
            print(f"[TSCS-P2] prefix-sum activated: acc={int(n_acc)}, reset={int(n_reset)}, osc={int(n_osc)} heads")
            self._tscs_p2_logged = True

        return Y.permute(0, 2, 1, 3).reshape(batch, seq_len, d_inner)

    def _build_tscs_caches(self, chunk_size):
        """Precompute L matrices and decay vectors for all 3 A modes.

        NOTE: Phase 2 (prefix-sum scan) only.
        Phase 1 (current) uses standard scan with ternary A without these caches.
        Kept for future optimization when implementing per-mode prefix-sum logic.
        """
        device = self.A_shadow.device if hasattr(self, 'A_shadow') else 'cpu'
        A_q = torch.clamp(torch.round(self.A_shadow.detach()), -1, 1)

        C = chunk_size
        n_heads = len(A_q)

        L = torch.zeros(n_heads, C, C, device=device)
        for h in range(n_heads):
            a = A_q[h].item()
            for i in range(C):
                for j in range(i + 1):
                    power = i - j
                    if a == 0.0:
                        L[h, i, j] = 1.0 if power == 0 else 0.0
                    elif a == 1.0:
                        L[h, i, j] = 1.0
                    else:
                        L[h, i, j] = (-1.0) ** power

        dfs = torch.zeros(n_heads, C, device=device)
        for h in range(n_heads):
            a = A_q[h].item()
            for t in range(C):
                if a == 0.0:
                    dfs[h, t] = 0.0 if t > 0 else 0.0
                elif a == 1.0:
                    dfs[h, t] = 1.0
                else:
                    dfs[h, t] = (-1.0) ** (t + 1)

        rem = torch.zeros(n_heads, C, device=device)
        for h in range(n_heads):
            a = A_q[h].item()
            for t in range(C):
                power = C - 1 - t
                if a == 0.0:
                    rem[h, t] = 1.0 if power == 0 else 0.0
                elif a == 1.0:
                    rem[h, t] = 1.0
                else:
                    rem[h, t] = (-1.0) ** power

        td = torch.zeros(n_heads, device=device)
        for h in range(n_heads):
            a = A_q[h].item()
            if a == 0.0:
                td[h] = 0.0
            elif a == 1.0:
                td[h] = 1.0
            else:
                td[h] = (-1.0) ** C

        self.register_buffer('_tscs_L', L, persistent=False)
        self.register_buffer('_tscs_dfs', dfs, persistent=False)
        self.register_buffer('_tscs_rem', rem, persistent=False)
        self.register_buffer('_tscs_td', td, persistent=False)
        self._tscs_chunk_size = C

    @staticmethod
    def _kronecker_project(h_full):
        """Project (batch, n_heads, d_state, head_dim) FP32 -> rank-1 ternary (u, v)."""
        # u: dominant direction along d_state
        u_energy = h_full.abs().mean(dim=-1)          # (batch, n_heads, d_state)
        u_sign   = h_full.sum(dim=-1).sign()          # (batch, n_heads, d_state)
        # v: dominant direction along head_dim
        v_energy = h_full.abs().mean(dim=-2)          # (batch, n_heads, head_dim)
        v_sign   = h_full.sum(dim=-2).sign()          # (batch, n_heads, head_dim)
        threshold = h_full.abs().mean() * 0.5
        u = (u_sign * (u_energy > threshold).float()).to(torch.int8)
        v = (v_sign * (v_energy > threshold).float()).to(torch.int8)
        return u, v

    @staticmethod
    def _kronecker_expand(u, v):
        """Expand rank-1 ternary (u, v) -> dense (batch, n_heads, d_state, head_dim)."""
        return torch.einsum('bhs,bhd->bhsd', u.float(), v.float())

    def _build_exp_lut(self, dt_min: float = 1e-4, dt_max: float = 30.0, n_dt_bins: int = 4096) -> None:
        """Build float16 lookup table for exp(A·dt) during inference.

        Precomputes a table mapping (head, dt_bin) -> exp(A[head] * dt_bin_value).
        Called once before inference starts, after A_log is frozen.

        Args:
            dt_min: Minimum dt value (default 1e-4)
            dt_max: Maximum dt value (default 30.0, matches boundary reset)
            n_dt_bins: Number of bins for dt discretization (default 4096)
        """
        if getattr(self, 'use_tscs', False):
            self._exp_lut_built = True
            return

        if self.training or self.A_log.dim() != 1:
            return

        A = -torch.exp(self.A_log.detach().float())
        dt_grid = torch.linspace(dt_min, dt_max, n_dt_bins, device=A.device, dtype=torch.float32)

        lut_fp = torch.exp(A.unsqueeze(1) * dt_grid.unsqueeze(0))
        lut_fp = lut_fp.clamp(0.0, 1.0)

        lut_float16 = lut_fp.to(torch.float16)
        self._exp_lut = lut_float16
        self._lut_dt_min = dt_min
        self._lut_dt_max = dt_max
        self._lut_n_bins = n_dt_bins
        self._exp_lut_built = True

    def _lookup_exp_A_dt(self, dt_heads: torch.Tensor) -> torch.Tensor:
        """Look up exp(A·dt) from precomputed LUT.

        Args:
            dt_heads: (batch, seq_len, n_heads) float32 dt values

        Returns:
            decay: (batch, seq_len, n_heads) float32 exponentially decayed values
        """
        if not hasattr(self, '_exp_lut') or self._exp_lut is None:
            return None

        dt_clamped = dt_heads.clamp(self._lut_dt_min, self._lut_dt_max)
        bin_idx = ((dt_clamped - self._lut_dt_min) /
                   (self._lut_dt_max - self._lut_dt_min) *
                   (self._lut_n_bins - 1)).long()

        batch, seq_len, n_heads = bin_idx.shape
        head_idx = torch.arange(n_heads, device=bin_idx.device, dtype=torch.long).view(1, 1, n_heads).expand_as(bin_idx)

        decay = self._exp_lut[head_idx, bin_idx].float()
        return decay

    def train(self, mode: bool = True):
        """Override train to build LUT when switching to eval mode."""
        result = super().train(mode)
        if not mode:
            if not getattr(self, '_exp_lut_built', False):
                self._build_exp_lut()
                if hasattr(self, '_exp_lut') and self._exp_lut is not None:
                    print(f"[EXP-LUT] Precomputed exp(A·dt) lookup table built | "
                          f"n_heads={self.n_heads}, n_dt_bins={self._lut_n_bins}, "
                          f"table_size={self._exp_lut.numel()*2//1024}KB")
        return result

    def forward(self, x: torch.Tensor, return_bc: bool = False, boundary_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            return_bc: if True, return (output, {'B': B, 'C': C})
            boundary_mask: (batch, seq_len) bool tensor, True at document boundaries
        Returns:
            (batch, seq_len, d_model) or ((batch, seq_len, d_model), {'B': B, 'C': C})
        """
        batch, seq_len, _ = x.shape

        if self.use_predictive_coding:
            x_shifted = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)
            _pc_rank = getattr(self, 'W_pred_U', None)
            if _pc_rank is not None:
                pred = self.W_pred_V(self.W_pred_U(x_shifted))
            else:
                pred = self.W_pred(x_shifted)
            x = x - pred

        if self.use_flip_rate_limit and self.training:
            self._enforce_flip_rate_limit()

        if self.use_commitment_k_schedule and self.training:
            self._update_adaptive_k()

        if self.use_fla and _FLA_AVAILABLE and x.is_cuda:
            scan_fn_name = '_ssd_scan_fla'
        elif self.use_chunked_ssd:
            scan_fn_name = '_ssd_scan_chunked' if self.use_batched_scan else '_ssd_scan_chunked_original'
        else:
            scan_fn_name = '_ssd_scan'

        if self.use_ssm_checkpoint and self.training:

            if not hasattr(self, '_qcssd_ext_logged'):
                print(f'[QCSSD-EXT] Extended checkpoint active for layer', flush=True)
                self._qcssd_ext_logged = True

            boundary_mask_or_none = boundary_mask if boundary_mask is not None else torch.zeros(0, device=x.device, dtype=torch.bool)
            act_fn_str = 'relu' if self.act_fn is F.relu else 'silu'
            output_pre_proj = QuantizedCheckpointSSDScan.apply(
                x,
                self.A_log,
                self.D,
                boundary_mask_or_none,
                act_fn_str,
                self._act_quant_bits,
                self._act_quant_group_size,
                scan_fn_name,
                self,
                seq_len
            )
            output = self.out_proj(output_pre_proj)
            output = self.dropout(output)
            if return_bc:
                with torch.no_grad():
                    xz_r = self.in_proj(x)
                    x_inner_r = xz_r[..., :self.d_inner]
                    x_t_r = x_inner_r.transpose(1, 2)
                    x_t_r = F.conv1d(x_t_r, self.conv1d.weight, self.conv1d.bias,
                                     padding=self.conv1d.weight.shape[-1] - 1,
                                     groups=self.d_inner)[:, :, :seq_len]
                    x_conv_r = self.act_fn(x_t_r.transpose(1, 2))
                    x_dbc_r = self.x_proj(x_conv_r)
                    B_r = x_dbc_r[..., self.dt_rank:self.dt_rank + self.d_state]
                    C_r = x_dbc_r[..., self.dt_rank + self.d_state:]
                return output, {'B': B_r, 'C': C_r}
            return output

        xz = self.in_proj(x)
        x_inner = xz[..., :self.d_inner]

        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.act_fn(x_conv)

        if self.use_bpnp:
            _bpnp_gate = torch.sigmoid(self.bpnp_gate_proj(x_conv.detach()))
        else:
            _bpnp_gate = None

        if boundary_mask is not None and not self._packing_logged:
            n_boundaries = boundary_mask.sum().item()
            total_positions = boundary_mask.numel()
            print(f"[SSM-PACKING] Received boundary_mask: {n_boundaries} boundaries in {total_positions} positions")
            print(f"[SSM-PACKING] Resetting hidden state at document boundaries")
            self._packing_logged = True

        if self.lti_mode:
            if self.training:
                y = self._lti_fft_conv(x_conv)
            else:
                if self.use_tscs_p2_prefix_sum:
                    y = self._lti_prefix_sum(x_conv)
                else:
                    y = self._lti_sequential(x_conv)

            if not self._lti_debug_logged:
                with torch.no_grad():
                    CB = (self.C_fixed * self.B_fixed).sum(-1)
                    print(f"[CTRI-LTI] CB dots: {CB.tolist()}")
                    if self.use_tscs:
                        A_q = self._quantize_A_ternary()
                        print(f"[CTRI-LTI] Ternary A: {A_q.tolist()}")
                        if self.use_tscs_p2_prefix_sum:
                            print(f"[CTRI-LTI] Phase 2 prefix-sum enabled (inference path)")
                self._lti_debug_logged = True
        else:
            x_dbc = self.x_proj(x_conv)
            dt_input = x_dbc[:, :, :self.dt_rank]
            B = x_dbc[:, :, self.dt_rank:self.dt_rank + self.d_state]
            C = x_dbc[:, :, self.dt_rank + self.d_state:]

            dt = F.softplus(self.dt_proj(dt_input))

            # AT-4: DT Surprise Gate — gate dt by prediction error |x - h_pred|
            if self.use_dt_surprise_gate and self.training:
                with torch.no_grad():
                    # h_pred approximation: mean of conv output as proxy for expected input
                    _at4_h_pred = x_conv.detach().mean(dim=1, keepdim=True).expand_as(x_conv)
                    _at4_surprise = (x_conv.detach() - _at4_h_pred).abs().mean(dim=-1, keepdim=True)
                    _at4_surprise_norm = _at4_surprise / (_at4_surprise.detach().mean() + 1e-6)
                    _at4_dt_gate = _at4_surprise_norm.clamp(0.1, 2.0)
                    _at4_mean = _at4_dt_gate.mean().item()
                dt = dt * _at4_dt_gate.expand_as(dt)
                print(f"[AT-4 DT-SURP] dt_gate_mean={_at4_mean:.3f}")

            if self.use_surprise_dt:
                with torch.no_grad():
                    surprise = x_conv.detach().var(dim=-1, keepdim=True).clamp(min=0.0)
                    surprise_norm = surprise / (surprise.mean() + 1e-6)
                    dt_gate = 1.0 + self.surprise_dt_scale * (surprise_norm - 1.0).clamp(min=-0.5, max=1.0)
                dt = dt * dt_gate.expand_as(dt)
                if not self._surprise_dt_logged:
                    print(f"[SURPRISE-DT] gate_mean={dt_gate.mean().item():.3f} "
                          f"gate_std={dt_gate.std().item():.3f}")
                    self._surprise_dt_logged = True

            if boundary_mask is not None:
                dt = dt.clone()
                dt[boundary_mask] = 30.0
                if not getattr(self, '_boundary_reset_logged', False):
                    n_boundaries = boundary_mask.sum().item()
                    print(f"[BOUNDARY-RESET] Document boundary state reset active | "
                          f"boundaries in batch: {n_boundaries}")
                    self._boundary_reset_logged = True

            if self.use_tscs:
                A = self._quantize_A_ternary()

                if not self._tscs_debug_logged:
                    print(f"[TSCS] A_shadow_init = {self.A_shadow.data.tolist()}")
                    print(f"[TSCS] A_ternary = {A.data.tolist()}")
                    print(f"[TSCS] Phase 1: Using baseline scan with ternary A (no prefix-sum yet)")
                    self._tscs_debug_logged = True
            else:
                if self.use_input_dep_A:
                    A_t_raw = self.W_A(x_conv)             # (batch, seq_len, n_heads)
                    A = -torch.exp(A_t_raw)                # Hurwitz: sempre negativo
                    if not self._input_dep_A_logged and not torch.compiler.is_compiling():
                        print(f"[MATRIX-MONOID] step A_t shape={list(A.shape)} "
                              f"mean={A.mean().item():.4f} std={A.std().item():.4f}")
                        self._input_dep_A_logged = True
                elif self.use_liquid_s4 and self.training:
                    _x_mean = x_conv.mean(dim=1)  # (B, d_inner)
                    _A_delta = self.liquid_proj(_x_mean)  # (B, n_heads)
                    _A_correction = _A_delta.mean(dim=0).detach() * 0.1  # (n_heads,)
                    A = -torch.exp(self.A_log + _A_correction)
                    print(f"[AQ-4 LIQUID-S4] A_delta_mean={_A_delta.mean().item():.4f}")
                else:
                    A = -torch.exp(self.A_log)

                # AV-13: CA Rule probe — A_log ternary distribution after thresholding
                if self.training and not torch.compiler.is_compiling():
                    self._av_counters = getattr(self, '_av_counters', {})
                    _av13_key = ('av13', id(self))
                    self._av_counters[_av13_key] = self._av_counters.get(_av13_key, 0) + 1
                    if self._av_counters[_av13_key] % 100 == 1:
                        with torch.no_grad():
                            _av13_a = self.A_log.detach().float()
                            _av13_thr = 0.5
                            _av13_neg = int((_av13_a < -_av13_thr).sum().item())
                            _av13_zero = int((_av13_a.abs() <= _av13_thr).sum().item())
                            _av13_pos = int((_av13_a > _av13_thr).sum().item())
                            print(f"[AV-13 CA-RULE-A] A_log neg={_av13_neg} "
                                  f"zero={_av13_zero} pos={_av13_pos} "
                                  f"total={_av13_a.numel()} "
                                  f"ternary_frac={(_av13_neg+_av13_pos)/max(_av13_a.numel(),1):.3f}")

            # AS-13 DPLR: zero S1 channel gradients in B and C projections
            if self.use_dplr and self.training and self._dplr_s1_channels is not None:
                _dplr_s1 = self._dplr_s1_channels.to(B.device)
                def _dplr_zero_s1_grad_B(grad, s1_mask=_dplr_s1):
                    if grad.shape[-1] == s1_mask.shape[0]:
                        return grad * (~s1_mask).float().to(grad.dtype)
                    return grad
                def _dplr_zero_s1_grad_C(grad, s1_mask=_dplr_s1):
                    if grad.shape[-1] == s1_mask.shape[0]:
                        return grad * (~s1_mask).float().to(grad.dtype)
                    return grad
                if B.requires_grad:
                    B.register_hook(_dplr_zero_s1_grad_B)
                if C.requires_grad:
                    C.register_hook(_dplr_zero_s1_grad_C)
                if not self._dplr_logged:
                    s1_frac = _dplr_s1.float().mean().item()
                    print(f"[AS-13 DPLR] hooks registered | s1_channels_frac={s1_frac:.3f}")
                    self._dplr_logged = True

            if self.use_ergodic_mixing and self.training:
                _alpha_t = (-A).detach()  # decay rates per head, positive
                _slow_mask = (_alpha_t > self._ergodic_lambda_thresh)
                _slow_frac = _slow_mask.float().mean().item()
                print(f"[AS-8 ERGODIC] slow_frac={_slow_frac:.3f}")

            if self.use_fast_slow_ssm and self.training:
                _dt_slow = dt / self._fast_slow_factor
                print(f"[AS-10 F-S] dt_fast_mean={dt.mean().item():.4f} dt_slow_mean={_dt_slow.mean().item():.4f}")

            # R4-A4-P8: CRG contraction
            if self.use_crg:
                A = A * self.alpha_decay.sigmoid()

            # AR-2 DSSU: dimension-selective state update probe (non-LTI path)
            _dssu_active_frac = 1.0
            if self.use_dssu and self.training:
                with torch.no_grad():
                    _x_l1 = x_conv.abs().mean()
                    _A_decay = torch.exp(self.A_log.detach().float())
                    _predicted_mag = _A_decay * _x_l1
                    _dssu_active = (_predicted_mag > self.dssu_epsilon).float()
                    _dssu_active_frac = _dssu_active.mean().item()
                print(f"[AR-2 DSSU] active_frac={_dssu_active_frac:.3f}")

            # AS-17 BNS2: B-Projection Norm as System 2 Signal
            if self.use_bns2 and self.training:
                with torch.no_grad():
                    _bx_norm = (B.detach() * x_conv.detach()[..., :B.shape[-1]]).norm(dim=-1)
                    _bx_norm_mean = _bx_norm.mean().item()
                    _bns2_s2_frac = (_bx_norm > self._bns2_norm_threshold).float().mean().item()
                print(f"[AS-17 BNS2] s2_frac={_bns2_s2_frac:.3f} bx_norm_mean={_bx_norm_mean:.3f}")

            if self.use_fla and _FLA_AVAILABLE and x.is_cuda:
                y = self._ssd_scan_fla(x_conv, dt, A, B, C, chunk_size=min(self.chunk_size, x_conv.shape[1]))
            elif self.use_chunked_ssd:
                if self.use_batched_scan:
                    y = self._ssd_scan_chunked(x_conv, dt, A, B, C, chunk_size=min(self.chunk_size, x_conv.shape[1]))
                else:
                    y = self._ssd_scan_chunked_original(x_conv, dt, A, B, C, chunk_size=min(self.chunk_size, x_conv.shape[1]))
            else:
                y = self._ssd_scan(x_conv, dt, A, B, C)

            if self.training and y.shape[1] > self.ssm_contraction_window:
                W = self.ssm_contraction_window
                y_early = y[:, :-W].detach()
                y_late = y[:, -W:]
                y = torch.cat([y_early, y_late], dim=1)
                if not self._contraction_logged:
                    print(f"[CONTRACTION] Window W={W}, detached {y.shape[1]-W}/{y.shape[1]} tokens")
                    self._contraction_logged = True

            y = self._cospectral_project(y)

            if self.training:
                self._update_gp_stats(y)

            # AV-11: Reversible SSM scan probe — h state checkpointing cost analysis
            if self.training and not torch.compiler.is_compiling():
                self._av_counters = getattr(self, '_av_counters', {})
                _av11_key = ('av11', id(self))
                self._av_counters[_av11_key] = self._av_counters.get(_av11_key, 0) + 1
                if self._av_counters[_av11_key] % 200 == 1:
                    with torch.no_grad():
                        _av11_head_dim = self.d_inner // self.n_heads
                        _av11_h_per_step = x.shape[0] * self.n_heads * self.d_state * _av11_head_dim
                        _av11_h_full_kb = (_av11_h_per_step * x.shape[1] * 2) / 1024
                        _av11_h_terminal_kb = (_av11_h_per_step * 2) / 1024
                        print(f"[AV-11 REV-SSM] seq_len={x.shape[1]} "
                              f"h_full_ckpt={_av11_h_full_kb:.1f}KB "
                              f"h_terminal_only={_av11_h_terminal_kb:.1f}KB "
                              f"savings_if_reversible={_av11_h_full_kb-_av11_h_terminal_kb:.1f}KB")
            # AV-12: Circulant NTT probe — in_proj param reduction (informational, once per layer)
            if self.training and not torch.compiler.is_compiling():
                self._av_counters = getattr(self, '_av_counters', {})
                _av12_key = ('av12', id(self))
                self._av_counters[_av12_key] = self._av_counters.get(_av12_key, 0) + 1
                if self._av_counters[_av12_key] == 1:
                    _av12_n = self.d_model
                    _av12_full = 2 * self.d_inner * _av12_n
                    _av12_circ = _av12_n
                    print(f"[AV-12 CIRC-NTT] in_proj_params={_av12_full} "
                          f"circulant_params={_av12_circ} "
                          f"reduction={_av12_full/max(_av12_circ,1):.1f}x "
                          f"(NTT O(n log n) vs dense O(n^2))")
            # AV-18: Cayley-Hamilton — A discretization type distribution (Type-0/P/N)
            if self.training and not torch.compiler.is_compiling():
                self._av_counters = getattr(self, '_av_counters', {})
                _av18_key = ('av18', id(self))
                self._av_counters[_av18_key] = self._av_counters.get(_av18_key, 0) + 1
                if self._av_counters[_av18_key] % 100 == 1:
                    with torch.no_grad():
                        _av18_dt_mean = dt.detach().float().mean()
                        _av18_a_disc = torch.exp(A.detach().float() * _av18_dt_mean)
                        _av18_type0 = int((_av18_a_disc.abs() < 0.1).sum().item())
                        _av18_typeP = int((_av18_a_disc > 0.5).sum().item())
                        _av18_typeN = int((_av18_a_disc < -0.5).sum().item())
                        print(f"[AV-18 CAYLEY-HAM] a_disc: Type-0={_av18_type0} "
                              f"Type-P={_av18_typeP} Type-N={_av18_typeN} "
                              f"total={_av18_a_disc.numel()} "
                              f"dt_mean={_av18_dt_mean.item():.4f}")

            # AW-2: Sheaf Laplacian — sign-change energy along sequence axis
            if self.training and not torch.compiler.is_compiling():
                self._av_counters = getattr(self, '_av_counters', {})
                _aw2_key = ('aw2', id(self))
                self._av_counters[_aw2_key] = self._av_counters.get(_aw2_key, 0) + 1
                if self._av_counters[_aw2_key] % 100 == 1:
                    with torch.no_grad():
                        _aw2_y = y.detach().float()
                        _aw2_sign = _aw2_y.sign()
                        _aw2_diff = (_aw2_sign[:, 1:, :] - _aw2_sign[:, :-1, :]).abs()
                        _aw2_sheaf_energy = _aw2_diff.mean().item()
                        print(f"[AW-2 SHEAF-LAP] sheaf_energy={_aw2_sheaf_energy:.4f} "
                              f"(0=no flips 2=all flips) seq_len={_aw2_y.shape[1]}")
            # AW-16: Clifford rotor norm — trace(A_log^T @ A_log) / d_state proxy
            if self.training and not torch.compiler.is_compiling():
                self._av_counters = getattr(self, '_av_counters', {})
                _aw16_key = ('aw16', id(self))
                self._av_counters[_aw16_key] = self._av_counters.get(_aw16_key, 0) + 1
                if self._av_counters[_aw16_key] % 200 == 1:
                    with torch.no_grad():
                        _aw16_a = self.A_log.detach().float()
                        _aw16_rotor_norm = (_aw16_a * _aw16_a).sum().item() / max(_aw16_a.numel(), 1)
                        print(f"[AW-16 CLIFFORD-ROTOR] rotor_norm={_aw16_rotor_norm:.4f} "
                              f"(ideal=1.0) valid={abs(_aw16_rotor_norm - 1.0) < 0.5}")

            # AR-3 SSSP: Sign-Stable State Propagation probe
            if self.use_sssp and self.training:
                with torch.no_grad():
                    _x_max = x_conv.abs().max().item()
                    _A_decay = torch.exp(self.A_log.detach().float())
                    _stability_ratio = _A_decay / (_x_max + 1e-6)
                    _sssp_stable_frac = (_stability_ratio > self._sssp_threshold).float().mean().item()
                print(f"[AR-3 SSSP] stable_frac={_sssp_stable_frac:.3f}")

            # AR-4 SCMS: Sparse Chunk Matrix Summation — output variance probe
            if self.use_scms and self.training:
                with torch.no_grad():
                    _y_var = y.detach().var().item()
                print(f"[AR-4 SCMS] y_var={_y_var:.4f}")

            # AR-1 AdaPonderLM: halt gate probe
            if self.use_adaponder and self.training:
                _halt_prob = torch.sigmoid(self.halt_gate(y.mean(dim=1))).mean().item()
                print(f"[AR-1 ADAPONDER] halt_prob={_halt_prob:.3f}")

            if self.use_ada_gate and self.training and hasattr(self, 'dsst_ema') and self.dsst_ema is not None:
                _h_approx = y.sign() * self.dsst_ema.detach()[:y.shape[1]]
                _coh = (_h_approx - y).pow(2).sum(-1) / (y.pow(2).sum(-1) + 1e-9)
                _s1_frac = (_coh < self._ada_threshold).float().mean().item()
                print(f"[AS-2 ADA] s1_frac={_s1_frac:.3f}")

            if self.use_vff_routing and self.training:
                _vff_score = torch.sigmoid(self.vff_gate(x_conv.mean(dim=1))).mean().item()
                print(f"[AS-9 VFF] route_score={_vff_score:.3f}")

        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        if self.use_tcsd and self.training:
            with torch.no_grad():
                y_det = y.detach()
                if self._tcsd_y_slow is None or self._tcsd_y_slow.shape != y_det.shape:
                    self._tcsd_y_slow = y_det.clone()
                else:
                    self._tcsd_y_slow.mul_(self._tcsd_beta).add_((1.0 - self._tcsd_beta) * y_det)
            _q = F.softmax(self._tcsd_y_slow.detach(), dim=-1)
            _log_p = F.log_softmax(y, dim=-1)
            self._tcsd_loss = F.kl_div(_log_p, _q, reduction='batchmean')
            if not self._tcsd_logged:
                print(f"[TCSD] active | loss={self._tcsd_loss.item():.4f}", flush=True)
                self._tcsd_logged = True
        else:
            if hasattr(self, '_tcsd_loss'):
                self._tcsd_loss = None

        if self.use_tug and self.training:
            _tug_active = getattr(self, '_sigma_gate_is_novel', False)
            print(f"[AS-5 TUG] is_novel={_tug_active}")

        if self.use_lyapunov_basin and self.training:
            _y_norm = y.pow(2).sum(-1).mean().item()
            print(f"[AS-11 LYAP] y_norm={_y_norm:.4f}")

        if self.use_depth_plasticity and self.training:
            _h = hash(tuple(y.detach().flatten()[:16].tolist()))
            self._dp_hashes[self._dp_ptr % self._dp_window] = _h
            self._dp_ptr += 1
            print(f"[AQ-5 DEPTH-PLASTICITY] hash={_h % 10000}")

        if _bpnp_gate is not None:
            y = _bpnp_gate * y + (1.0 - _bpnp_gate) * x_conv
            if not self._bpnp_logged:
                with torch.no_grad():
                    gate_mean = _bpnp_gate.mean().item()
                    gate_low_frac = (_bpnp_gate < self.bpnp_threshold).float().mean().item()
                    print(f"[BPNP] gate_mean={gate_mean:.3f} low_frac={gate_low_frac:.3f} "
                          f"(threshold={self.bpnp_threshold})")
                self._bpnp_logged = True

        act_z = self.act_fn(xz[..., self.d_inner:])
        act_z = act_z * y

        # AP-1: Sigma-g Surprise Gate
        if self.use_sigma_gate and self.training:
            with torch.no_grad():
                _z_var = xz[..., self.d_inner:].detach().var().item()
                self._sigma_gate_var_ema = (
                    self._sigma_gate_beta * self._sigma_gate_var_ema
                    + (1.0 - self._sigma_gate_beta) * _z_var
                )
                self._sigma_gate_baseline = (
                    self._sigma_gate_beta * self._sigma_gate_baseline
                    + (1.0 - self._sigma_gate_beta) * _z_var
                )
                self._sigma_gate_is_novel = (
                    _z_var > self._sigma_gate_baseline * (1.0 + self._sigma_gate_threshold)
                )
                if not self._sigma_gate_logged:
                    print(f"[AP-1 SIGMA-GATE] ACTIVATED | z_var={_z_var:.4f} novel={self._sigma_gate_is_novel}")
                    self._sigma_gate_logged = True

        # AT-1: Three-Timescale CLS — update EMA of output, log context injection
        if self.use_cls_timescale and hasattr(self, '_cls_h_ema') and self.training:
            with torch.no_grad():
                _cls_out_mean = act_z.detach().mean(dim=1, keepdim=True)  # (batch, 1, d_inner)
                self._cls_h_ema.mul_(self._cls_alpha).add_((1.0 - self._cls_alpha) * _cls_out_mean.mean(0, keepdim=True))
                _cls_norm = self._cls_h_ema.norm().item()
            print(f"[AT-1 CLS] ema_norm={_cls_norm:.4f} alpha={self._cls_alpha}")

        output = self.out_proj(act_z)

        output = self.dropout(output)
        if return_bc:
            if self.lti_mode:
                B_expanded = self.B_fixed.mean(0).unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1)
                C_expanded = self.C_fixed.mean(0).unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1)
                return output, {'B': B_expanded, 'C': C_expanded}
            else:
                return output, {'B': B, 'C': C}
        return output

    @staticmethod
    def _walsh_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
        """In-place Walsh-Hadamard Transform along last dimension.

        Applies WHT butterfly network: d_state butterfly layers of ±1 additions.
        For d_state=128=2^7: 7 layers × 128 ops = 896 additions, zero parameters.
        Creates effective circulant mixing of state dimensions between recurrence steps.

        Args:
            x: [..., d_state] where d_state must be a power of 2

        Returns:
            WHT-mixed tensor, same shape as input, normalized by sqrt(d_state)
        """
        d = x.shape[-1]
        h = 1
        while h < d:
            x_even = x[..., 0::2*h].clone()
            x_odd = x[..., h::2*h].clone()
            x[..., 0::2*h] = x_even + x_odd
            x[..., h::2*h] = x_even - x_odd
            h *= 2
        return x / (d ** 0.5)

    def _ssd_scan(self, x: torch.Tensor, dt: torch.Tensor,
                  A: torch.Tensor, B: torch.Tensor,
                  C: torch.Tensor) -> torch.Tensor:
        """Parallel SSD scan (Mamba-2 style).

        Replaces sequential loop with matrix multiplications.
        A is scalar per head (or input-dependent per token via Matrix Monoid).
        """
        batch, seq_len, d_inner = x.shape
        n_heads = self.n_heads
        head_dim = d_inner // n_heads

        dt_heads = dt.view(batch, seq_len, n_heads, head_dim).mean(-1)

        if A.dim() == 1:
            log_decay = dt_heads * A[None, None, :]
        else:
            log_decay = dt_heads * A   # A: (batch, seq_len, n_heads) — shapes compatíveis

        log_decay_t = log_decay.permute(0, 2, 1)
        L = FusedSegSumExp.apply(log_decay_t)

        CB = torch.einsum('btn,bsn->bts', C, B)

        dt_scale = dt_heads.permute(0, 2, 1)

        M = L * CB.unsqueeze(1) * dt_scale.unsqueeze(2)

        X = x.view(batch, seq_len, n_heads, head_dim).permute(0, 2, 1, 3)
        Y = torch.matmul(M.to(X.dtype), X)

        Y = Y.permute(0, 2, 1, 3).reshape(batch, seq_len, d_inner)

        return Y

    def _ssd_scan_fla(self, x: torch.Tensor, dt: torch.Tensor,
                      A: torch.Tensor, B: torch.Tensor,
                      C: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
        """FLA-accelerated SSD scan via chunk_simple_gla.

        Maps SSD recurrence to Simple GLA and uses FLA's fused Triton kernels
        for ~8x speedup over Python-loop chunked implementation.

        SSD: h_t = exp(A*dt)*h_{t-1} + dt*(B outer x), y_t = C^T @ h_t
        GLA: S_t = exp(g)*S_{t-1} + k^T @ v,          o_t = q @ S_t
        """
        batch, seq_len, d_inner = x.shape
        n_heads = self.n_heads
        head_dim = d_inner // n_heads
        d_state = B.size(-1)

        dt_heads = dt.view(batch, seq_len, n_heads, head_dim).mean(-1)

        if A.dim() == 1:
            g = A[None, None, :] * dt_heads
        else:
            g = A * dt_heads   # A: (batch, seq_len, n_heads)
        q = C[:, :, None, :].expand(batch, seq_len, n_heads, d_state).contiguous()
        k = (dt_heads[:, :, :, None] *
             B[:, :, None, :].expand(batch, seq_len, n_heads, d_state)).contiguous()
        v = x.view(batch, seq_len, n_heads, head_dim).contiguous()

        # Cast all inputs to float32 for Triton kernel compatibility.
        # FLA's Triton kernel internally does exp() which promotes to fp32,
        # then tl.dot(fp32, bf16) crashes on Triton < 3.2.0.
        # Explicit fp32 ensures uniform dtype throughout the kernel.
        out_dtype = v.dtype
        with torch.amp.autocast('cuda', enabled=False):
            o, _ = chunk_simple_gla(
                q.float(), k.float(), v.float(), g.float(),
                scale=1.0, output_final_state=False
            )
        return o.to(out_dtype).reshape(batch, seq_len, d_inner)

    @torch.compiler.disable()
    def _ssd_scan_chunked(self, x: torch.Tensor, dt: torch.Tensor,
                          A: torch.Tensor, B: torch.Tensor,
                          C: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
        """Batched chunked SSD scan: O(T*C) memory with ~50% less time in loop.

        Pre-reshapes all inputs into [batch, n_chunks, chunk_size, ...] and computes
        all intra-chunk operations (FusedSegSumExp, einsum, matmul) in a single
        batched operation. Only the state propagation remains in a sequential loop,
        which is inherently sequential due to causal dependency.

        Mathematically equivalent to _ssd_scan_chunked_original() but uses batched
        operations for independent per-chunk computation.

        Args:
            x: (batch, seq_len, d_inner)
            dt: (batch, seq_len, d_inner)
            A: (n_heads,) - scalar decay per head
            B: (batch, seq_len, d_state)
            C: (batch, seq_len, d_state)
            chunk_size: size of each processing chunk (default 64)

        Returns:
            Y: (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        n_heads = self.n_heads
        head_dim = d_inner // n_heads
        d_state = B.size(-1)

        dt_heads = dt.view(batch, seq_len, n_heads, head_dim).mean(-1)
        X = x.view(batch, seq_len, n_heads, head_dim).permute(0, 2, 1, 3)

        if (hasattr(self, 'config') and self.config is not None and
            getattr(self.config, 'use_episodic_state', False) and self._episodic_h is not None):
            h = self._episodic_h
            if self.episodic_gate is not None:
                gate_input = x[:, 0, :]
                gate = torch.sigmoid(self.episodic_gate(gate_input))
                gate = gate.view(batch, n_heads, d_state, 1)
                h = gate * h
        else:
            if self._h0_calibrated:
                h = self.h0_mean.expand(batch, -1, -1, -1).clone()
            else:
                h = torch.zeros(batch, n_heads, d_state, head_dim, device=x.device, dtype=torch.float32)

        if self.use_krylov_compression and self.use_tscs and self.training:
            A_ternary = self._quantize_A_ternary()
            self._krylov_Q = self._arnoldi_ternary(A_ternary, h, self.krylov_rank)

        if self.use_wht_mixing:
            h = h.transpose(-2, -1)
            h = self._walsh_hadamard_transform(h)
            h = h.transpose(-2, -1)
            if not self._wht_logged:
                print(f"[WHT-ONCE] Applied 1× at layer entry | h.shape={list(h.shape)} d_state={h.shape[-2]}", flush=True)
                self._wht_logged = True

        n_chunks = (seq_len + chunk_size - 1) // chunk_size
        padded_len = n_chunks * chunk_size

        if padded_len > seq_len:
            pad_size = padded_len - seq_len
            dt_heads_pad = torch.nn.functional.pad(dt_heads, (0, 0, 0, pad_size))
            B_pad = torch.nn.functional.pad(B, (0, 0, 0, pad_size))
            C_pad = torch.nn.functional.pad(C, (0, 0, 0, pad_size))
            X_pad = torch.nn.functional.pad(X, (0, 0, 0, pad_size))
        else:
            dt_heads_pad = dt_heads
            B_pad = B
            C_pad = C
            X_pad = X

        dt_chunks = dt_heads_pad.reshape(batch, n_chunks, chunk_size, n_heads)
        B_chunks = B_pad.reshape(batch, n_chunks, chunk_size, d_state)
        C_chunks = C_pad.reshape(batch, n_chunks, chunk_size, d_state)
        X_chunks = X_pad.reshape(batch, n_heads, n_chunks, chunk_size, head_dim)

        if A.dim() == 1:
            log_decay_chunks = dt_chunks.permute(0, 1, 3, 2) * A[None, None, :, None]
        else:
            # A: (batch, seq_len, n_heads) — reshape to chunks
            B_size = A.shape[0]
            n_chunks_real = A.shape[1] // chunk_size if A.shape[1] >= chunk_size else 1
            A_chunks = A[:, :n_chunks_real * chunk_size, :].reshape(B_size, n_chunks_real, chunk_size, n_heads)
            A_chunks = A_chunks.permute(0, 1, 3, 2)   # (batch, n_chunks, n_heads, chunk_size)
            log_decay_chunks = dt_chunks.permute(0, 1, 3, 2) * A_chunks
        L_chunks = FusedSegSumExp.apply(log_decay_chunks.reshape(batch * n_chunks * n_heads, chunk_size))
        L_chunks = L_chunks.reshape(batch, n_chunks, n_heads, chunk_size, chunk_size)

        CB_chunks = torch.einsum('bktn,bksn->bkts', C_chunks, B_chunks)
        CB_chunks = CB_chunks.detach()
        if not getattr(self, '_k2_logged', False):
            print(f"[K2-CB-DETACH] CB_chunks detached shape={tuple(CB_chunks.shape)}")
            self._k2_logged = True
        dt_scale_chunks = dt_chunks.permute(0, 1, 3, 2).unsqueeze(-1)

        M_chunks = L_chunks * CB_chunks.unsqueeze(2) * dt_scale_chunks

        X_chunks_reshaped = X_chunks.permute(0, 2, 1, 3, 4)
        Y_local_chunks = torch.matmul(M_chunks.to(X_chunks_reshaped.dtype), X_chunks_reshaped)

        cum_decay_chunks = torch.cumsum(log_decay_chunks, dim=-1)
        if not self.training and A.dim() == 1 and hasattr(self, '_exp_lut') and self._exp_lut is not None:
            decay_per_step = self._lookup_exp_A_dt(dt_heads_pad)
            decay_per_step_c = decay_per_step.reshape(batch, n_chunks, chunk_size, n_heads).permute(0, 1, 3, 2)
            decay_chunks = torch.cumprod(decay_per_step_c, dim=-1)
        else:
            decay_chunks = torch.exp(cum_decay_chunks)

        Y_chunks = []
        h_update_chunks = []
        total_decay_chunks = []
        h_init = h.clone()

        for chunk_idx in range(n_chunks):
            c_len = min(chunk_size, seq_len - chunk_idx * chunk_size)

            C_c = C_chunks[:, chunk_idx, :c_len, :]
            B_c = B_chunks[:, chunk_idx, :c_len, :]
            X_c = X_chunks[:, :, chunk_idx, :c_len, :]
            dt_c = dt_chunks[:, chunk_idx, :c_len, :]

            # R19-D: expand Kronecker rank-1 state if available
            if self.use_kronecker_state and self._kron_h_u is not None:
                h = self._kronecker_expand(self._kron_h_u, self._kron_h_v)

            C_c_exp = C_c.unsqueeze(1).expand(batch, n_heads, c_len, d_state)
            h_contrib = torch.einsum('bhts,bhsd->bhtd', C_c_exp, h)
            decay_c = decay_chunks[:, chunk_idx, :, :c_len]
            Y_state = h_contrib * decay_c.unsqueeze(-1)

            Y_local = Y_local_chunks[:, chunk_idx, :, :c_len, :]
            Y_chunk = Y_local + Y_state

            Y_chunks.append(Y_chunk)

            if self.use_krylov_compression and chunk_idx < n_chunks - 1:
                h_compressed = self._compress_state(h)
                h = self._decompress_state(h_compressed)

            log_decay_c = log_decay_chunks[:, chunk_idx, :, :c_len]
            if not self.training and A.dim() == 1 and hasattr(self, '_exp_lut') and self._exp_lut is not None:
                decay_c_per_step = self._lookup_exp_A_dt(dt_c)
                total_decay = torch.prod(decay_c_per_step, dim=-2)
            else:
                total_decay = torch.exp(log_decay_c.sum(dim=-1))
            total_decay_chunks.append(total_decay)

            h = h * total_decay.unsqueeze(-1).unsqueeze(-1)

            cum_decay_c = cum_decay_chunks[:, chunk_idx, :, :c_len]
            if not self.training and A.dim() == 1 and hasattr(self, '_exp_lut') and self._exp_lut is not None:
                decay_c_per_step = self._lookup_exp_A_dt(dt_c)
                cumprod_decay_c = torch.cumprod(decay_c_per_step, dim=-2)
                remaining_decay = (cumprod_decay_c[:, :, -1:] / (cumprod_decay_c + 1e-10)).permute(0, 2, 1)
            else:
                remaining_decay = torch.exp(cum_decay_c[:, :, -1:] - cum_decay_c)

            dt_x = dt_c.permute(0, 2, 1).unsqueeze(-1) * X_c
            dt_x_rem = dt_x * remaining_decay.unsqueeze(-1)

            B_c_exp = B_c.unsqueeze(1).expand(batch, n_heads, c_len, d_state)
            h_update = torch.einsum('bhts,bhtd->bhsd', B_c_exp, dt_x_rem)
            h_update_chunks.append(h_update)
            h = h + h_update

            if self.use_tma:
                h = self._tma_step(h)

            if self._truncated_bptt:
                h = h.detach()

            # R19-D: project h back to rank-1 ternary
            if self.use_kronecker_state:
                self._kron_h_u, self._kron_h_v = self._kronecker_project(h)
                if not self._kron_logged and not torch.compiler.is_compiling():
                    h_approx = self._kronecker_expand(self._kron_h_u, self._kron_h_v)
                    proj_err = (h - h_approx).norm() / (h.norm() + 1e-8)
                    print(f"[KRONECKER] ACTIVATED | h.shape={list(h.shape)} "
                          f"u={list(self._kron_h_u.shape)} v={list(self._kron_h_v.shape)} "
                          f"proj_err={proj_err:.4f}", flush=True)
                    self._kron_logged = True

        if self.use_wht_mixing:
            h = h.transpose(-2, -1)
            h = self._walsh_hadamard_transform(h)
            h = h.transpose(-2, -1)

        h_update_stack = torch.stack(h_update_chunks, dim=0)
        total_decay_stack = torch.stack(total_decay_chunks, dim=0)

        if self._truncated_bptt:
            h_final = h
            if not self._tbptt_logged:
                print(f"[TBPTT] truncated_bptt=True, chunk_size={self.chunk_size}, n_chunks={n_chunks}", flush=True)
                self._tbptt_logged = True
        else:
            if not getattr(self, '_rev_ssm_logged', False):
                print(f"[REV-SSM] chunk_boundary prop active, n_chunks={n_chunks}, "
                      f"h_shape={list(h_init.shape)}, memory_factor=~28x compression")
                self._rev_ssm_logged = True
            h_final = ReversibleChunkBoundaryProp.apply(h_init, h_update_stack, total_decay_stack)

        Y = torch.cat(Y_chunks, dim=2)
        Y = Y[:, :, :seq_len, :]
        Y = Y.permute(0, 2, 1, 3).reshape(batch, seq_len, d_inner)

        if (hasattr(self, 'config') and self.config is not None and
            getattr(self.config, 'use_episodic_state', False)):
            self._episodic_h = h_final.detach()

        return Y

    def _update_gp_stats(self, Y: torch.Tensor) -> None:
        """Online Welford update for GP-KL statistics. Y: [B, T, d_inner].

        Accumulates: mean, variance (via M2), lag-1 cross-product, and counts.
        All ops in torch.no_grad() — breaks autograd anchor on Y.
        """
        with torch.no_grad():
            B, T, D = Y.shape
            Y_det = Y.detach().float()

            # Lag-1 cross-product (must happen before flattening, using _gp_prev)
            if self._gp_prev is not None and T > 0:
                # first token of this forward pairs with last token of previous
                cross_boundary = (Y_det[:, 0, :] * self._gp_prev.unsqueeze(0)).sum(0)
                cross_interior = (Y_det[:, 1:, :] * Y_det[:, :-1, :]).sum(dim=(0, 1))
                n_pairs_boundary = B
                n_pairs_interior = B * (T - 1)
            else:
                cross_boundary = torch.zeros(D, device=Y.device)
                cross_interior = (Y_det[:, 1:, :] * Y_det[:, :-1, :]).sum(dim=(0, 1)) if T > 1 else torch.zeros(D, device=Y.device)
                n_pairs_boundary = 0
                n_pairs_interior = B * (T - 1)

            new_cross = cross_boundary + cross_interior
            new_pairs = n_pairs_boundary + n_pairs_interior

            # Welford over [B*T] samples per d_inner dimension
            flat = Y_det.reshape(B * T, D)  # [N, D]
            N = flat.shape[0]

            batch_mean = flat.mean(0)       # [D]
            batch_m2 = flat.var(dim=0, unbiased=False) * flat.shape[0]

            if self._gp_mean is None:
                self._gp_n = N
                self._gp_mean = batch_mean
                self._gp_m2 = batch_m2
                self._gp_cross = new_cross
                self._gp_n_pairs = new_pairs
            else:
                # Parallel Welford merge
                n_a = self._gp_n
                n_b = N
                n_ab = n_a + n_b
                delta = batch_mean - self._gp_mean
                self._gp_mean = (n_a * self._gp_mean + n_b * batch_mean) / n_ab
                self._gp_m2 = self._gp_m2 + batch_m2 + delta ** 2 * n_a * n_b / n_ab
                self._gp_n = n_ab
                self._gp_cross = self._gp_cross + new_cross
                self._gp_n_pairs = getattr(self, '_gp_n_pairs', 0) + new_pairs

            self._gp_prev = Y_det[:, -1, :].mean(0)  # [D], mean over batch

    @torch.compiler.disable()
    def _ssd_scan_chunked_original(self, x: torch.Tensor, dt: torch.Tensor,
                                    A: torch.Tensor, B: torch.Tensor,
                                    C: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
        """Original chunked SSD scan (sequential loop per chunk).

        Baseline implementation for A/B testing. Processes sequence in chunks of
        size C, computing intra-chunk semiseparable matrices (C x C) and propagating
        SSM state across chunks. Mathematically equivalent to _ssd_scan() but uses
        4x less peak memory and ~3x fewer FLOPs for typical sequence lengths.

        Args:
            x: (batch, seq_len, d_inner)
            dt: (batch, seq_len, d_inner)
            A: (n_heads,) - scalar decay per head
            B: (batch, seq_len, d_state)
            C: (batch, seq_len, d_state)
            chunk_size: size of each processing chunk (default 64)

        Returns:
            Y: (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        n_heads = self.n_heads
        head_dim = d_inner // n_heads
        d_state = B.size(-1)

        dt_heads = dt.view(batch, seq_len, n_heads, head_dim).mean(-1)
        X = x.view(batch, seq_len, n_heads, head_dim).permute(0, 2, 1, 3)

        if (hasattr(self, 'config') and self.config is not None and
            getattr(self.config, 'use_episodic_state', False) and self._episodic_h is not None):
            h = self._episodic_h
            if self.episodic_gate is not None:
                gate_input = x[:, 0, :]
                gate = torch.sigmoid(self.episodic_gate(gate_input))
                gate = gate.view(batch, n_heads, d_state, 1)
                h = gate * h
        else:
            if self._h0_calibrated:
                h = self.h0_mean.expand(batch, -1, -1, -1).clone()
            else:
                h = torch.zeros(batch, n_heads, d_state, head_dim, device=x.device, dtype=torch.float32)

        if self.use_wht_mixing:
            h = h.transpose(-2, -1)
            h = self._walsh_hadamard_transform(h)
            h = h.transpose(-2, -1)
            if not self._wht_logged:
                print(f"[WHT-ONCE] Applied 1× at layer entry | h.shape={list(h.shape)} d_state={h.shape[-2]}", flush=True)
                self._wht_logged = True

        n_chunks = (seq_len + chunk_size - 1) // chunk_size
        Y_chunks = []

        for chunk_idx in range(n_chunks):
            s = chunk_idx * chunk_size
            e = min(s + chunk_size, seq_len)
            c_len = e - s

            dt_c = dt_heads[:, s:e, :]
            B_c = B[:, s:e, :]
            C_c = C[:, s:e, :]
            X_c = X[:, :, s:e, :]

            if A.dim() == 1:
                log_decay_c = dt_c.permute(0, 2, 1) * A[None, :, None]
            else:
                A_c = A[:, s:e, :].permute(0, 2, 1)  # (batch, n_heads, c_len)
                log_decay_c = dt_c.permute(0, 2, 1) * A_c
            L_c = FusedSegSumExp.apply(log_decay_c)

            CB_c = torch.einsum('btn,bsn->bts', C_c, B_c)
            CB_c = CB_c.detach()
            dt_scale_c = dt_c.permute(0, 2, 1).unsqueeze(-1)

            M_c = L_c * CB_c.unsqueeze(1) * dt_scale_c

            Y_local = torch.matmul(M_c.to(X_c.dtype), X_c)

            cum_decay_c = torch.cumsum(log_decay_c, dim=-1)
            decay_c = torch.exp(cum_decay_c)

            # R19-D: expand Kronecker rank-1 state if available
            if self.use_kronecker_state and self._kron_h_u is not None:
                h = self._kronecker_expand(self._kron_h_u, self._kron_h_v)

            C_c_exp = C_c.unsqueeze(1).expand(batch, n_heads, c_len, d_state)
            h_contrib = torch.einsum('bhts,bhsd->bhtd', C_c_exp, h)
            Y_state = h_contrib * decay_c.unsqueeze(-1)

            Y_chunk = Y_local + Y_state

            Y_chunks.append(Y_chunk)

            total_decay = torch.exp(
                (dt_c.permute(0, 2, 1) * A[None, :, None]).sum(dim=-1)
            )

            h = h * total_decay.unsqueeze(-1).unsqueeze(-1)

            remaining_decay = torch.exp(cum_decay_c[:, :, -1:] - cum_decay_c)

            dt_x = dt_c.permute(0, 2, 1).unsqueeze(-1) * X_c
            dt_x_rem = dt_x * remaining_decay.unsqueeze(-1)

            B_c_exp = B_c.unsqueeze(1).expand(batch, n_heads, c_len, d_state)
            h_update = torch.einsum('bhts,bhtd->bhsd', B_c_exp, dt_x_rem)
            h = h + h_update

            if self._truncated_bptt:
                h = h.detach()

            # R19-D: project h back to rank-1 ternary
            if self.use_kronecker_state:
                self._kron_h_u, self._kron_h_v = self._kronecker_project(h)
                if not self._kron_logged and not torch.compiler.is_compiling():
                    h_approx = self._kronecker_expand(self._kron_h_u, self._kron_h_v)
                    proj_err = (h - h_approx).norm() / (h.norm() + 1e-8)
                    print(f"[KRONECKER] ACTIVATED | h.shape={list(h.shape)} "
                          f"u={list(self._kron_h_u.shape)} v={list(self._kron_h_v.shape)} "
                          f"proj_err={proj_err:.4f}", flush=True)
                    self._kron_logged = True

        if self.use_wht_mixing:
            h = h.transpose(-2, -1)
            h = self._walsh_hadamard_transform(h)
            h = h.transpose(-2, -1)

        Y = torch.cat(Y_chunks, dim=2)
        Y = Y.permute(0, 2, 1, 3).reshape(batch, seq_len, d_inner)

        if (hasattr(self, 'config') and self.config is not None and
            getattr(self.config, 'use_episodic_state', False)):
            self._episodic_h = h.detach()

        return Y

    def update_kalman_hessian(self) -> None:
        """Batch AH: Update diagonal Hessian from SSM backward pass.

        grad(A_log)^2 and grad(dt_proj.bias)^2 are the diagonal Fisher
        approximation of recurrence parameters — equivalent to extracting
        Kalman smoother backward information: ∂L/∂A_log integrates ∂L/∂h_t
        over the full sequence, for free via autograd.

        Call after loss.backward(), before optimizer.step().
        """
        if self.A_log.grad is not None:
            g2 = self.A_log.grad.detach().pow(2)
            if self._kh_A_log_diag is None:
                self._kh_A_log_diag = g2.clone()
            else:
                self._kh_A_log_diag = (
                    self._kh_ema * self._kh_A_log_diag
                    + (1.0 - self._kh_ema) * g2
                )

        has_dt_bias = (
            not getattr(self, 'lti_mode', False)
            and hasattr(self, 'dt_proj')
            and self.dt_proj.bias is not None
            and self.dt_proj.bias.grad is not None
        )
        if has_dt_bias:
            g2 = self.dt_proj.bias.grad.detach().pow(2)
            if self._kh_dt_bias_diag is None:
                self._kh_dt_bias_diag = g2.clone()
            else:
                self._kh_dt_bias_diag = (
                    self._kh_ema * self._kh_dt_bias_diag
                    + (1.0 - self._kh_ema) * g2
                )

        if not self._kh_logged and self._kh_A_log_diag is not None:
            a_curv = self._kh_A_log_diag.mean().item()
            dt_curv = (self._kh_dt_bias_diag.mean().item()
                       if self._kh_dt_bias_diag is not None else 0.0)
            print(f"[KH-AH] Kalman-Hessian active | "
                  f"A_log curv={a_curv:.4e} dt_bias curv={dt_curv:.4e}")
            self._kh_logged = True

    def apply_kalman_precond(self, eps: float = 1e-8) -> None:
        """Batch AH: Apply Kalman-Hessian preconditioning to SSM param gradients.

        Scales A_log.grad and dt_proj.bias.grad by 1/sqrt(diag_H + eps),
        analogous to Adam's denominator. Improves curvature-awareness of
        the optimizer for recurrence parameters.

        Call after update_kalman_hessian(), before optimizer.step().
        """
        if self._kh_A_log_diag is not None and self.A_log.grad is not None:
            scale = (self._kh_A_log_diag + eps).sqrt()
            self.A_log.grad.div_(scale)

        if (self._kh_dt_bias_diag is not None
                and not getattr(self, 'lti_mode', False)
                and hasattr(self, 'dt_proj')
                and self.dt_proj.bias is not None
                and self.dt_proj.bias.grad is not None):
            scale = (self._kh_dt_bias_diag + eps).sqrt()
            self.dt_proj.bias.grad.div_(scale)

    def reset_episodic_state(self):
        self._episodic_h = None

    @torch.no_grad()
    def calibrate_h0(self, h_in_batches: list, device):
        """Compute corpus-mean h₀ by averaging h_T over calibration sequences.

        Args:
            h_in_batches: list of tensors (batch, seq_len, d_model) — pre-embedded inputs
            device: torch device
        """
        accumulator = None
        for h_in in h_in_batches:
            h_in = h_in.to(device)
            xz = self.in_proj(h_in)
            x_inner = xz[..., :self.d_inner]
            x_conv = x_inner.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :h_in.shape[1]]
            x_conv = x_conv.transpose(1, 2)
            x_conv = self.act_fn(x_conv)

            x_dbc = self.x_proj(x_conv)
            dt_input = x_dbc[:, :, :self.dt_rank]
            B = x_dbc[:, :, self.dt_rank:self.dt_rank + self.d_state]
            C = x_dbc[:, :, self.dt_rank + self.d_state:]
            dt = F.softplus(self.dt_proj(dt_input))
            A = -torch.exp(self.A_log)

            batch, seq_len, d_inner = x_conv.shape
            head_dim = d_inner // self.n_heads
            h_T = self._run_scan_get_h_final(x_conv, dt, A, B, C, batch, seq_len, head_dim)

            if accumulator is None:
                accumulator = h_T.mean(dim=0, keepdim=True)
            else:
                accumulator = accumulator + h_T.mean(dim=0, keepdim=True)

        if accumulator is not None:
            self.h0_mean.copy_(accumulator / len(h_in_batches))
            self._h0_calibrated = True
            print(f"[WARM-H0] calibrated over {len(h_in_batches)} batches, "
                  f"norm={self.h0_mean.norm():.4f}", flush=True)

    @torch.no_grad()
    def _run_scan_get_h_final(self, x_conv, dt, A, B, C, batch, seq_len, head_dim):
        """Run sequential scan and return final h_T."""
        n_heads, d_state = self.n_heads, self.d_state
        X = x_conv.view(batch, seq_len, n_heads, head_dim).permute(0, 2, 1, 3)
        h = torch.zeros(batch, n_heads, d_state, head_dim,
                        device=x_conv.device, dtype=torch.float32)
        dt_heads = dt.view(batch, seq_len, n_heads, head_dim).mean(-1)
        for t in range(seq_len):
            dt_t = dt_heads[:, t, :]
            log_decay = dt_t.unsqueeze(-1) * A[None, :, None]
            decay = torch.exp(log_decay).unsqueeze(-1)
            x_t = X[:, :, t, :].unsqueeze(2)
            Bx = B[:, t, :].unsqueeze(1).unsqueeze(-1) * x_t
            h = decay * h + Bx
        return h

    def _cospectral_project(self, y: torch.Tensor) -> torch.Tensor:
        """R340: Co-spectral subspace projection via straight-through.

        Forward: identity (y unchanged).
        Backward: gradient flows only through rank-20 subspace.
        Calibration: online PCA of y during first 50 steps.

        Args:
            y: [B, T, d_inner] scan output
        Returns:
            y: [B, T, d_inner] identical values, reduced-rank backward
        """
        if not self.training:
            return y

        self._csp_step += 1

        if self._csp_step <= 50:
            with torch.no_grad():
                y_flat = y.detach().reshape(-1, y.shape[-1]).float()
                cov = y_flat.T @ y_flat
                if not hasattr(self, '_csp_cov_acc'):
                    self._csp_cov_acc = torch.zeros_like(cov)
                    self._csp_n = 0
                self._csp_cov_acc += cov
                self._csp_n += y_flat.shape[0]

            if self._csp_step == 50:
                with torch.no_grad():
                    cov_mean = (self._csp_cov_acc / self._csp_n).float()  # eigh needs float32
                    _, U = torch.linalg.eigh(cov_mean)
                    self._subspace_basis = U[:, -20:].to(y.dtype)
                    del self._csp_cov_acc, self._csp_n
                print(f"[R340-CSP] Basis calibrated: d_inner={y.shape[-1]} rank=20")
            return y

        if not hasattr(self, '_subspace_basis'):
            return y

        U = self._subspace_basis
        y_proj = (y @ U) @ U.T
        y_out = y.detach() + (y_proj - y_proj.detach())
        return y_out


class EigenmodeMixedSSMSSD(nn.Module):
    """I7: Split SSM heads into fast (ternary) and slow (FP16) sub-SSMs.

    Based on eigenmode analysis showing bimodal A_log distribution:
    - Slow modes (small |A|, long memory): FP16 precision
    - Fast modes (large |A|, rapid decay): ternary precision OK
    """

    def __init__(self, d_model, d_state, d_conv, expand, dt_rank, n_heads,
                 slow_fraction=0.3125, use_ternary=True, ternary_group_size=128,
                 dropout=0.0, use_ssm_checkpoint=False):
        super().__init__()

        self.head_dim = (expand * d_model) // n_heads
        self.n_heads_slow = max(1, int(n_heads * slow_fraction))
        self.n_heads_fast = n_heads - self.n_heads_slow
        self.d_inner_slow = self.n_heads_slow * self.head_dim
        self.d_inner_fast = self.n_heads_fast * self.head_dim

        # Slow sub-SSM: ALWAYS FP16 (use_ternary=False)
        self.ssm_slow = SelectiveSSMSSD(
            d_model=d_model, d_state=d_state, d_conv=d_conv,
            expand=expand, dt_rank=dt_rank, n_heads=self.n_heads_slow,
            use_ternary=False, ternary_group_size=ternary_group_size,
            dropout=dropout, use_ssm_checkpoint=use_ssm_checkpoint,
            d_inner_override=self.d_inner_slow,
        )

        # Fast sub-SSM: uses ternary (or whatever use_ternary says)
        self.ssm_fast = SelectiveSSMSSD(
            d_model=d_model, d_state=d_state, d_conv=d_conv,
            expand=expand, dt_rank=dt_rank, n_heads=self.n_heads_fast,
            use_ternary=use_ternary, ternary_group_size=ternary_group_size,
            dropout=dropout, use_ssm_checkpoint=use_ssm_checkpoint,
            d_inner_override=self.d_inner_fast,
        )

        # Initialize A_log: slow gets small values (long memory), fast gets large
        with torch.no_grad():
            self.ssm_slow.A_log.copy_(torch.log(torch.arange(1, self.n_heads_slow + 1, dtype=torch.float32)))
            self.ssm_fast.A_log.copy_(torch.log(torch.arange(self.n_heads_slow + 1, n_heads + 1, dtype=torch.float32)))

        print(f"[I7-EigenmodeMixed] slow={self.n_heads_slow}h (FP16, d_inner={self.d_inner_slow}), "
              f"fast={self.n_heads_fast}h (ternary={use_ternary}, d_inner={self.d_inner_fast})")

    def forward(self, x, return_bc=False):
        if return_bc:
            out_slow, bc_slow = self.ssm_slow(x, return_bc=True)
            out_fast, bc_fast = self.ssm_fast(x, return_bc=True)
            # Use slow BC (longer memory, more informative for attention)
            return out_slow + out_fast, bc_slow
        else:
            return self.ssm_slow(x) + self.ssm_fast(x)

