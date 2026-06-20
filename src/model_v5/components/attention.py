import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .rmsnorm import RMSNorm
from .ternary import TernaryLinear
from .instant import InstantProjector


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
                 max_seq_len: int = 2048, dropout: float = 0.0, use_sdpa: bool = False,
                 use_head_gating: bool = False, n_sink_tokens: int = 0,
                 use_phase_space: bool = False, d_state: int = 16,
                 activation_fn: str = 'silu', norm_class=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.use_sdpa = use_sdpa
        self.store_attn_weights = False
        self._last_attn_weights = None
        self.per_head_windows = None
        self.act_fn = F.relu if activation_fn == 'relu' else F.silu

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

        Norm = norm_class if norm_class is not None else RMSNorm
        self.attn_norm = Norm(d_model)
        self.mlp_norm = Norm(d_model)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.use_head_gating = use_head_gating
        if use_head_gating:
            GateLinear = TernaryLinear if use_ternary else nn.Linear
            gate_kwargs = {'group_size': ternary_group_size} if use_ternary else {}
            self.head_gate_proj = GateLinear(d_model, n_heads, bias=False, **gate_kwargs)
            self.head_gate_bias = nn.Parameter(torch.full((n_heads,), 2.0))
            if not use_ternary:
                nn.init.zeros_(self.head_gate_proj.weight)
            print(f"[HeadGating] Enabled: {d_model}->{n_heads}, ternary={use_ternary}, init_bias=+2.0")

        self.n_sink_tokens = n_sink_tokens
        if n_sink_tokens > 0:
            print(f"[SinkTokens] Enabled: first {n_sink_tokens} position(s) always visible")

        self.use_phase_space = use_phase_space
        if use_phase_space:
            assert d_state >= 32, f"Phase-space attention needs d_state >= 32, got {d_state}"
            self.q_proj_ps = nn.Linear(d_state, d_model, bias=False)
            self.k_proj_ps = nn.Linear(d_state, d_model, bias=False)
            print(f"[PhaseSpaceAttn] d_state={d_state} -> d_model={d_model}")

        self._cached_mask = None
        self._cached_mask_float = None
        self._cached_mask_len = 0

        # INSTANT low-rank activation compression (R2-A2-F6)
        self.use_instant = False
        self._instant_projectors = {}
        self._instant_step = 0
        self._instant_warmup = 100
        self._instant_debug_logged = False

    def enable_instant(self, rank: int = 32, update_gap: int = 50, warmup: int = 100):
        """Enable INSTANT low-rank activation compression for this SWA layer.

        Uses saved_tensors_hooks to transparently compress all activations
        saved during the forward pass. No changes to layer computation needed.

        R2-A2-F6
        """
        self.use_instant = True
        self._instant_rank = rank
        self._instant_update_gap = update_gap
        self._instant_warmup = warmup
        self._instant_projectors = {}  # Lazily populated per-dimension
        self._instant_debug_logged = False

        print(f"[INSTANT] Enabled on SWA layer: rank={rank}, update_gap={update_gap}, warmup={warmup}")

    def _make_instant_hooks(self):
        """Create pack/unpack hooks for INSTANT activation compression."""
        projectors = self._instant_projectors

        def pack(tensor):
            if not isinstance(tensor, torch.Tensor):
                return tensor
            if tensor.ndim < 2 or tensor.numel() < 512:
                return tensor  # Skip small tensors (bias, scalars)
            d_in = tensor.shape[-1]
            key = f"d{d_in}"
            if key not in projectors:
                rank = min(self._instant_rank, d_in // 2)
                if rank < 4:
                    return tensor  # Too small to compress
                projectors[key] = InstantProjector(rank=rank, update_gap=self._instant_update_gap)
            proj = projectors[key]
            if proj.proj_matrix is None:
                # First call - initialize projection
                proj.project(tensor)
            x_proj = (tensor @ proj.proj_matrix.T).contiguous()
            return (x_proj, key)

        def unpack(packed):
            if isinstance(packed, tuple) and len(packed) == 2:
                x_proj, key = packed
                if isinstance(key, str) and key in projectors:
                    proj = projectors[key]
                    return (x_proj @ proj.proj_matrix).contiguous()
            return packed

        return pack, unpack

    def forward(self, x: torch.Tensor, ssm_bc: dict = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            ssm_bc: dict with 'B' and 'C' tensors from SSM state-space (optional, for phase-space attention)
        Returns:
            (batch, seq_len, d_model)
        """
        _use_instant = (self.use_instant and self.training
                        and self._instant_step >= self._instant_warmup)

        if _use_instant:
            if not self._instant_debug_logged:
                print(f"  [INSTANT] Active at step {self._instant_step}: "
                      f"compressing SWA activations via saved_tensors_hooks")
                self._instant_debug_logged = True

            pack_fn, unpack_fn = self._make_instant_hooks()
            with torch.autograd.graph.saved_tensors_hooks(pack_fn, unpack_fn):
                return self._forward_impl(x, ssm_bc)
        else:
            return self._forward_impl(x, ssm_bc)

    def _forward_impl(self, x: torch.Tensor, ssm_bc: dict = None) -> torch.Tensor:
        """Actual forward computation (extracted for INSTANT hook wrapping)."""
        residual = x
        x_norm = self.attn_norm(x)
        attn_out = self._attention(x_norm, ssm_bc=ssm_bc)
        x = residual + self.dropout(attn_out)

        residual = x
        x_norm = self.mlp_norm(x)
        gate_out = self.act_fn(self.mlp_gate(x_norm))
        gate_out = gate_out * self.mlp_up(x_norm)
        mlp_out = self.mlp_down(gate_out)
        x = residual + self.dropout(mlp_out)

        return x

    def _attention(self, x: torch.Tensor, ssm_bc: dict = None) -> torch.Tensor:
        B, L, _ = x.shape

        if self.use_phase_space and ssm_bc is not None:
            q = self.q_proj_ps(ssm_bc['C']).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj_ps(ssm_bc['B']).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        else:
            q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(q, L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.per_head_windows is not None:
            row_idx = torch.arange(L, device=x.device)
            col_idx = torch.arange(L, device=x.device)
            causal = col_idx.unsqueeze(0) <= row_idx.unsqueeze(1)

            windows_t = torch.tensor(self.per_head_windows, device=x.device).view(-1, 1, 1)
            window = col_idx.unsqueeze(0) >= (row_idx.unsqueeze(1) - windows_t + 1)
            mask = causal.unsqueeze(0) & window

            if self.n_sink_tokens > 0:
                sink = col_idx.unsqueeze(0) < self.n_sink_tokens
                mask = mask | (causal.unsqueeze(0) & sink.unsqueeze(0))

            mask_float = torch.where(mask, 0.0, float('-inf'))

            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            if mask_float.dim() == 3:
                scores = scores.masked_fill(~mask.unsqueeze(0), float('-inf'))
            else:
                scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            if self.store_attn_weights:
                self._last_attn_weights = attn_weights.detach()
            attn_weights = self.dropout(attn_weights)
            attn_out = torch.matmul(attn_weights, v)

        else:
            if self._cached_mask is None or self._cached_mask_len != L or self._cached_mask.device != x.device:
                row_idx = torch.arange(L, device=x.device)
                col_idx = torch.arange(L, device=x.device)
                causal = col_idx.unsqueeze(0) <= row_idx.unsqueeze(1)
                window = col_idx.unsqueeze(0) >= (row_idx.unsqueeze(1) - self.window_size + 1)
                mask = causal & window

                if self.n_sink_tokens > 0:
                    sink = col_idx.unsqueeze(0) < self.n_sink_tokens
                    mask = mask | (causal & sink)

                self._cached_mask = mask
                self._cached_mask_float = torch.where(self._cached_mask.unsqueeze(0).unsqueeze(0), 0.0, float('-inf'))
                self._cached_mask_len = L
            mask = self._cached_mask

            if self.use_sdpa and not self.store_attn_weights:
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
                if self.store_attn_weights:
                    self._last_attn_weights = attn_weights.detach()
                attn_weights = self.dropout(attn_weights)
                attn_out = torch.matmul(attn_weights, v)

        if self.use_head_gating:
            gate = torch.sigmoid(self.head_gate_proj(x) + self.head_gate_bias.to(x.dtype))
            gate = gate.transpose(1, 2).unsqueeze(-1)
            attn_out = attn_out * gate

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(attn_out)
