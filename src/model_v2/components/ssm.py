import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .rmsnorm import RMSNorm
from .ternary import TernaryLinear


_segsum_mask_cache: dict = {}


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

    Which normally involves: cumsum → broadcast subtract → mask → masked_fill → exp
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
                 ternary_group_size: int = 128, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_dbc = self.x_proj(x_conv)
        dt_input = x_dbc[:, :, :self.dt_rank]
        B = x_dbc[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C = x_dbc[:, :, self.dt_rank + self.d_state:]

        dt = F.softplus(self.dt_proj(dt_input))

        A = -torch.exp(self.A_log)

        y = self._selective_scan(x_conv, dt, A, B, C)

        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        silu_z = F.silu(z)
        silu_z.mul_(y)
        output = self.out_proj(silu_z)

        return self.dropout(output)

    def _selective_scan(self, x: torch.Tensor, dt: torch.Tensor,
                        A: torch.Tensor, B: torch.Tensor,
                        C: torch.Tensor) -> torch.Tensor:
        """Sequential selective scan -- memory-efficient implementation.

        Computes dA and dB per-timestep instead of pre-computing for the
        full sequence, reducing memory from O(B*L*D*N) to O(B*D*N).
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        y = torch.zeros(batch, seq_len, d_inner, device=x.device, dtype=x.dtype)
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=torch.float32)

        for t in range(seq_len):
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
                 use_batched_scan: bool = True, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.n_heads = n_heads
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        self.use_chunked_ssd = use_chunked_ssd
        self.chunk_size = chunk_size
        self.use_batched_scan = use_batched_scan

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

        A_init = torch.log(torch.arange(1, n_heads + 1, dtype=torch.float32))
        self.A_log = nn.Parameter(A_init.clone())

        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = Linear(self.d_inner, d_model, bias=False, **linear_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_dbc = self.x_proj(x_conv)
        dt_input = x_dbc[:, :, :self.dt_rank]
        B = x_dbc[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C = x_dbc[:, :, self.dt_rank + self.d_state:]

        dt = F.softplus(self.dt_proj(dt_input))

        A = -torch.exp(self.A_log)

        if self.use_chunked_ssd:
            if self.use_batched_scan:
                y = self._ssd_scan_chunked(x_conv, dt, A, B, C, chunk_size=self.chunk_size)
            else:
                y = self._ssd_scan_chunked_original(x_conv, dt, A, B, C, chunk_size=self.chunk_size)
        else:
            y = self._ssd_scan(x_conv, dt, A, B, C)

        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        silu_z = F.silu(z)
        silu_z.mul_(y)
        output = self.out_proj(silu_z)

        return self.dropout(output)

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

        h = torch.zeros(batch, n_heads, d_state, head_dim, device=x.device, dtype=torch.float32)

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

        return Y

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

        return Y

