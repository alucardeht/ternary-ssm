import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .rmsnorm import RMSNorm
from .ternary import TernaryLinear


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor, seq_len: int):
        return self.cos_cached[:, :, :seq_len], self.sin_cached[:, :, :seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class SlidingWindowAttention(nn.Module):
    """Sliding Window Attention block with RoPE.

    For the ~12.5% of layers that handle retrieval.
    Window size = 128 tokens (industry standard).
    """
    def __init__(self, d_model: int, n_heads: int, window_size: int = 128,
                 use_ternary: bool = False, ternary_group_size: int = 128,
                 max_seq_len: int = 2048, dropout: float = 0.0, use_sdpa: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.use_sdpa = use_sdpa

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        Linear = TernaryLinear if use_ternary else nn.Linear
        linear_kwargs = {'group_size': ternary_group_size} if use_ternary else {}

        self.q_proj = Linear(d_model, d_model, bias=False, **linear_kwargs)
        self.k_proj = Linear(d_model, d_model, bias=False, **linear_kwargs)
        self.v_proj = Linear(d_model, d_model, bias=False, **linear_kwargs)
        self.out_proj = Linear(d_model, d_model, bias=False, **linear_kwargs)

        mlp_hidden = int(d_model * 2.67)
        self.mlp_up = Linear(d_model, mlp_hidden, bias=False, **linear_kwargs)
        self.mlp_gate = Linear(d_model, mlp_hidden, bias=False, **linear_kwargs)
        self.mlp_down = Linear(mlp_hidden, d_model, bias=False, **linear_kwargs)

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)

        self.attn_norm = RMSNorm(d_model)
        self.mlp_norm = RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._cached_mask = None
        self._cached_mask_float = None
        self._cached_mask_len = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        residual = x
        x_norm = self.attn_norm(x)
        attn_out = self._attention(x_norm)
        x = residual + self.dropout(attn_out)

        residual = x
        x_norm = self.mlp_norm(x)
        gate_out = F.silu(self.mlp_gate(x_norm))
        gate_out.mul_(self.mlp_up(x_norm))
        mlp_out = self.mlp_down(gate_out)
        x = residual + self.dropout(mlp_out)

        return x

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(q, L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self._cached_mask is None or self._cached_mask_len != L or self._cached_mask.device != x.device:
            row_idx = torch.arange(L, device=x.device)
            col_idx = torch.arange(L, device=x.device)
            causal = col_idx.unsqueeze(0) <= row_idx.unsqueeze(1)
            window = col_idx.unsqueeze(0) >= (row_idx.unsqueeze(1) - self.window_size + 1)
            self._cached_mask = causal & window
            self._cached_mask_float = torch.where(self._cached_mask.unsqueeze(0).unsqueeze(0), 0.0, float('-inf'))
            self._cached_mask_len = L
        mask = self._cached_mask

        if self.use_sdpa:
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=self._cached_mask_float,
                dropout_p=self.dropout.p if isinstance(self.dropout, nn.Dropout) and self.training else 0.0,
                is_causal=False,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_out = torch.matmul(attn_weights, v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(attn_out)
