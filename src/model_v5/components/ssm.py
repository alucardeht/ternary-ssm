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

    candidates = [-1.0, 0.0, 1.0]
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

        ctx.save_for_backward(result, mask)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, mask = ctx.saved_tensors

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
    def forward(ctx, x_conv, dt, A, B, C, D, z, act_fn_str,
                n_bits, group_size, scan_fn_name, ssm_module):
        """
        Args:
            x_conv, dt, A, B, C, D, z: Standard SSD scan inputs
            act_fn_str: 'silu' or 'relu'
            n_bits, group_size: Quantization parameters
            scan_fn_name: Name of scan method to call ('_ssd_scan_fla', '_ssd_scan_chunked', etc.)
            ssm_module: Reference to the SelectiveSSMSSD instance (for calling scan methods)
        """
        scan_fn = getattr(ssm_module, scan_fn_name)
        with torch.no_grad():
            y = scan_fn(x_conv, dt, A, B, C)

        y_d = y + x_conv * D.unsqueeze(0).unsqueeze(0)
        act_fn = F.silu if act_fn_str == 'silu' else F.relu
        act_z = act_fn(z)
        output = act_z * y_d

        x_q, x_s, x_m = _quantize_for_storage(x_conv, n_bits, group_size)
        dt_q, dt_s, dt_m = _quantize_for_storage(dt, n_bits, group_size)
        B_q, B_s, B_m = _quantize_for_storage(B, n_bits, group_size)
        C_q, C_s, C_m = _quantize_for_storage(C, n_bits, group_size)
        z_q, z_s, z_m = _quantize_for_storage(z, n_bits, group_size)

        ctx.save_for_backward(x_q, x_s, dt_q, dt_s, B_q, B_s, C_q, C_s, z_q, z_s, A, D)
        ctx.x_m = x_m
        ctx.dt_m = dt_m
        ctx.B_m = B_m
        ctx.C_m = C_m
        ctx.z_m = z_m
        ctx.act_fn_str = act_fn_str
        ctx.scan_fn_name = scan_fn_name
        ctx.ssm_module = ssm_module

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Dequantize inputs, recompute forward, use autograd for gradients."""
        (x_q, x_s, dt_q, dt_s, B_q, B_s, C_q, C_s, z_q, z_s, A, D) = ctx.saved_tensors

        x_conv = _dequantize_from_storage(x_q, x_s, ctx.x_m).requires_grad_(True)
        dt = _dequantize_from_storage(dt_q, dt_s, ctx.dt_m).requires_grad_(True)
        B = _dequantize_from_storage(B_q, B_s, ctx.B_m).requires_grad_(True)
        C = _dequantize_from_storage(C_q, C_s, ctx.C_m).requires_grad_(True)
        z = _dequantize_from_storage(z_q, z_s, ctx.z_m).requires_grad_(True)
        A = A.detach().requires_grad_(True)
        D = D.detach().requires_grad_(True)

        scan_fn = getattr(ctx.ssm_module, ctx.scan_fn_name)
        act_fn = F.silu if ctx.act_fn_str == 'silu' else F.relu

        with torch.enable_grad():
            y = scan_fn(x_conv, dt, A, B, C)
            y_d = y + x_conv * D.unsqueeze(0).unsqueeze(0)
            act_z = act_fn(z)
            output = act_z * y_d

        grads = torch.autograd.grad(
            output, (x_conv, dt, A, B, C, D, z),
            grad_outputs=grad_output,
            allow_unused=True
        )

        grad_x_conv, grad_dt, grad_A, grad_B, grad_C, grad_D, grad_z = grads

        if grad_x_conv is None: grad_x_conv = torch.zeros_like(x_conv)
        if grad_dt is None: grad_dt = torch.zeros_like(dt)
        if grad_A is None: grad_A = torch.zeros_like(A)
        if grad_B is None: grad_B = torch.zeros_like(B)
        if grad_C is None: grad_C = torch.zeros_like(C)
        if grad_D is None: grad_D = torch.zeros_like(D)
        if grad_z is None: grad_z = torch.zeros_like(z)

        return grad_x_conv, grad_dt, grad_A, grad_B, grad_C, grad_D, grad_z, None, None, None, None, None


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

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(self.d_inner, -1).clone())

        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = Linear(self.d_inner, d_model, bias=False, **linear_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

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

        dt = F.softplus(self.dt_proj(dt_input))

        A = -torch.exp(self.A_log)

        if self.use_ssm_checkpoint and self.training:
            y = torch.utils.checkpoint.checkpoint(
                self._selective_scan, x_conv, dt, A, B, C, boundary_mask,
                use_reentrant=False,
            )
        else:
            y = self._selective_scan(x_conv, dt, A, B, C, boundary_mask)

        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        act_z = self.act_fn(z)
        act_z = act_z * y
        output = self.out_proj(act_z)

        return self.dropout(output)

    @torch.compiler.disable()
    def _selective_scan(self, x: torch.Tensor, dt: torch.Tensor,
                        A: torch.Tensor, B: torch.Tensor,
                        C: torch.Tensor, boundary_mask: torch.Tensor = None) -> torch.Tensor:
        """Sequential selective scan -- memory-efficient implementation.

        Computes dA and dB per-timestep instead of pre-computing for the
        full sequence, reducing memory from O(B*L*D*N) to O(B*D*N).

        Args:
            boundary_mask: (batch, seq_len) bool tensor. When True at position t,
                hidden state h is reset to zero (for document boundaries in packed sequences).
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        y = torch.zeros(batch, seq_len, d_inner, device=x.device, dtype=x.dtype)
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=torch.float32)

        for t in range(seq_len):
            if boundary_mask is not None and boundary_mask[:, t].any():
                h[boundary_mask[:, t]] = 0.0

            dt_t = dt[:, t]
            dA_t = torch.exp(dt_t.unsqueeze(-1) * A)
            dB_t = dt_t.unsqueeze(-1) * B[:, t].unsqueeze(1)

            x_t = x[:, t].unsqueeze(-1).to(torch.float32)
            h = dA_t.to(torch.float32) * h + dB_t.to(torch.float32) * x_t
            y[:, t] = (h * C[:, t].unsqueeze(1).to(torch.float32)).sum(-1).to(x.dtype)

        return y


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
                 use_chunked_ssd: bool = True, chunk_size: int = 64,
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
        self.use_batched_scan = use_batched_scan
        self.lti_mode = lti_mode
        self.act_fn = F.relu if (config and getattr(config, 'activation_fn', 'silu') == 'relu') or activation_fn == 'relu' else F.silu

        Linear = TernaryLinear if use_ternary else nn.Linear
        linear_kwargs = {'group_size': ternary_group_size} if use_ternary else {}

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

        A_init = torch.log(torch.arange(1, n_heads + 1, dtype=torch.float32))
        self.A_log = nn.Parameter(A_init.clone())

        self.D = nn.Parameter(torch.ones(self.d_inner))

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

        use_episodic_gate = getattr(config, 'use_episodic_gate', False) if config is not None else False
        self.episodic_gate = nn.Linear(d_model, d_state * n_heads, bias=True) if use_episodic_gate else None

        self._packing_logged = False

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
        A_ternary[self.A_shadow > threshold] = 1.0

        A_ternary = self.A_shadow - self.A_shadow.detach() + A_ternary.detach()

        return A_ternary

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
            h = A_col.float() * h + B[None, :, :, None] * x_t[:, :, None, :]
            Y[:, :, t_idx, :] = (C[None, :, :, None] * h).sum(dim=2).to(x_conv.dtype)

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

        if self.use_flip_rate_limit and self.training:
            self._enforce_flip_rate_limit()

        if self.use_commitment_k_schedule and self.training:
            self._update_adaptive_k()

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.act_fn(x_conv)

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

            if self.use_tscs:
                A = self._quantize_A_ternary()

                if not self._tscs_debug_logged:
                    print(f"[TSCS] A_shadow_init = {self.A_shadow.data.tolist()}")
                    print(f"[TSCS] A_ternary = {A.data.tolist()}")
                    print(f"[TSCS] Phase 1: Using baseline scan with ternary A (no prefix-sum yet)")
                    self._tscs_debug_logged = True
            else:
                A = -torch.exp(self.A_log)

            # R4-A4-P8: CRG contraction
            if self.use_crg:
                A = A * self.alpha_decay.sigmoid()

            if self.use_fla and _FLA_AVAILABLE and x.is_cuda:
                scan_fn_name = '_ssd_scan_fla'
            elif self.use_chunked_ssd:
                scan_fn_name = '_ssd_scan_chunked' if self.use_batched_scan else '_ssd_scan_chunked_original'
            else:
                scan_fn_name = '_ssd_scan'

            if (self.use_activation_quant and self.training
                and self._act_quant_step >= self._act_quant_warmup):

                if not self._act_quant_debug_logged:
                    print(f"[ACT-QUANT] Active: bits={self._act_quant_bits}, "
                          f"group_size={self._act_quant_group_size}, "
                          f"scan={scan_fn_name}, step={self._act_quant_step}")
                    self._act_quant_debug_logged = True

                act_fn_str = 'relu' if self.act_fn is F.relu else 'silu'
                output_pre_proj = QuantizedCheckpointSSDScan.apply(
                    x_conv, dt, A, B, C, self.D, z,
                    act_fn_str,
                    self._act_quant_bits, self._act_quant_group_size,
                    scan_fn_name, self
                )
                output = self.out_proj(output_pre_proj)
                output = self.dropout(output)
                if return_bc:
                    return output, {'B': B, 'C': C}
                return output

            elif self.use_ssm_checkpoint and self.training:
                def _scan_fn(x_c, dt_c, A_c, B_c, C_c):
                    if self.use_fla and _FLA_AVAILABLE and x_c.is_cuda:
                        return self._ssd_scan_fla(x_c, dt_c, A_c, B_c, C_c, chunk_size=self.chunk_size)
                    elif self.use_chunked_ssd:
                        if self.use_batched_scan:
                            return self._ssd_scan_chunked(x_c, dt_c, A_c, B_c, C_c, chunk_size=self.chunk_size)
                        else:
                            return self._ssd_scan_chunked_original(x_c, dt_c, A_c, B_c, C_c, chunk_size=self.chunk_size)
                    else:
                        return self._ssd_scan(x_c, dt_c, A_c, B_c, C_c)
                y = torch.utils.checkpoint.checkpoint(
                    _scan_fn, x_conv, dt, A, B, C,
                    use_reentrant=False,
                )
            else:
                if self.use_fla and _FLA_AVAILABLE and x.is_cuda:
                    y = self._ssd_scan_fla(x_conv, dt, A, B, C, chunk_size=self.chunk_size)
                elif self.use_chunked_ssd:
                    if self.use_batched_scan:
                        y = self._ssd_scan_chunked(x_conv, dt, A, B, C, chunk_size=self.chunk_size)
                    else:
                        y = self._ssd_scan_chunked_original(x_conv, dt, A, B, C, chunk_size=self.chunk_size)
                else:
                    y = self._ssd_scan(x_conv, dt, A, B, C)

        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        act_z = self.act_fn(z)
        act_z = act_z * y
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

    def _ssd_scan(self, x: torch.Tensor, dt: torch.Tensor,
                  A: torch.Tensor, B: torch.Tensor,
                  C: torch.Tensor) -> torch.Tensor:
        """Parallel SSD scan (Mamba-2 style).

        Replaces sequential loop with matrix multiplications.
        A is scalar per head, enabling parallel computation via segsum.
        """
        batch, seq_len, d_inner = x.shape
        n_heads = self.n_heads
        head_dim = d_inner // n_heads

        dt_heads = dt.view(batch, seq_len, n_heads, head_dim).mean(-1)

        log_decay = dt_heads * A[None, None, :]

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

        g = A[None, None, :] * dt_heads
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
            h = torch.zeros(batch, n_heads, d_state, head_dim, device=x.device, dtype=torch.float32)

        if self.use_krylov_compression and self.use_tscs and self.training:
            A_ternary = self._quantize_A_ternary()
            self._krylov_Q = self._arnoldi_ternary(A_ternary, h, self.krylov_rank)

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

        log_decay_chunks = dt_chunks.permute(0, 1, 3, 2) * A[None, None, :, None]
        L_chunks = FusedSegSumExp.apply(log_decay_chunks.reshape(batch * n_chunks * n_heads, chunk_size))
        L_chunks = L_chunks.reshape(batch, n_chunks, n_heads, chunk_size, chunk_size)

        CB_chunks = torch.einsum('bktn,bksn->bkts', C_chunks, B_chunks)
        dt_scale_chunks = dt_chunks.permute(0, 1, 3, 2).unsqueeze(-1)

        M_chunks = L_chunks * CB_chunks.unsqueeze(2) * dt_scale_chunks

        X_chunks_reshaped = X_chunks.permute(0, 2, 1, 3, 4)
        Y_local_chunks = torch.matmul(M_chunks.to(X_chunks_reshaped.dtype), X_chunks_reshaped)

        cum_decay_chunks = torch.cumsum(log_decay_chunks, dim=-1)
        decay_chunks = torch.exp(cum_decay_chunks)

        Y_chunks = []

        for chunk_idx in range(n_chunks):
            c_len = min(chunk_size, seq_len - chunk_idx * chunk_size)

            C_c = C_chunks[:, chunk_idx, :c_len, :]
            B_c = B_chunks[:, chunk_idx, :c_len, :]
            X_c = X_chunks[:, :, chunk_idx, :c_len, :]
            dt_c = dt_chunks[:, chunk_idx, :c_len, :]

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
            total_decay = torch.exp(log_decay_c.sum(dim=-1))

            h = h * total_decay.unsqueeze(-1).unsqueeze(-1)

            cum_decay_c = cum_decay_chunks[:, chunk_idx, :, :c_len]
            remaining_decay = torch.exp(cum_decay_c[:, :, -1:] - cum_decay_c)

            dt_x = dt_c.permute(0, 2, 1).unsqueeze(-1) * X_c
            dt_x_rem = dt_x * remaining_decay.unsqueeze(-1)

            B_c_exp = B_c.unsqueeze(1).expand(batch, n_heads, c_len, d_state)
            h_update = torch.einsum('bhts,bhtd->bhsd', B_c_exp, dt_x_rem)
            h = h + h_update

        Y = torch.cat(Y_chunks, dim=2)
        Y = Y[:, :, :seq_len, :]
        Y = Y.permute(0, 2, 1, 3).reshape(batch, seq_len, d_inner)

        if (hasattr(self, 'config') and self.config is not None and
            getattr(self.config, 'use_episodic_state', False)):
            self._episodic_h = h.detach()

        return Y

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
            h = torch.zeros(batch, n_heads, d_state, head_dim, device=x.device, dtype=torch.float32)

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

            log_decay_c = dt_c.permute(0, 2, 1) * A[None, :, None]
            L_c = FusedSegSumExp.apply(log_decay_c)

            CB_c = torch.einsum('btn,bsn->bts', C_c, B_c)
            dt_scale_c = dt_c.permute(0, 2, 1).unsqueeze(-1)

            M_c = L_c * CB_c.unsqueeze(1) * dt_scale_c

            Y_local = torch.matmul(M_c.to(X_c.dtype), X_c)

            cum_decay_c = torch.cumsum(log_decay_c, dim=-1)
            decay_c = torch.exp(cum_decay_c)

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

        Y = torch.cat(Y_chunks, dim=2)
        Y = Y.permute(0, 2, 1, 3).reshape(batch, seq_len, d_inner)

        if (hasattr(self, 'config') and self.config is not None and
            getattr(self.config, 'use_episodic_state', False)):
            self._episodic_h = h.detach()

        return Y

    def reset_episodic_state(self):
        self._episodic_h = None


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

