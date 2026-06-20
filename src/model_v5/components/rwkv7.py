"""RWKV-7 Goose time-mixing block.
Drop-in replacement for SelectiveSSMSSD with element-wise gating recurrence.
Reference: https://arxiv.org/abs/2503.14456
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ternary import TernaryLinear


class RWKV7Block(nn.Module):
    """RWKV-7 time-mixing block with data-dependent gating.

    State update (per head, per timestep):
        S_t = S_{t-1} * w_t + (S_{t-1} @ (kk*a)) @ (-kk)^T + v^T @ k^T
        y_t = S_t @ r_t

    Where w_t is data-dependent decay, a is learning rate gate,
    kk is L2-normalized key, r is receptance (query).
    """

    def __init__(
        self,
        d_model: int,
        head_size: int = 64,
        n_heads: int = None,
        lora_rank: int = 32,
        use_ternary: bool = False,
        ternary_group_size: int = 128,
        chunk_size: int = 64,
    ):
        super().__init__()
        assert d_model % head_size == 0, f"d_model ({d_model}) must be divisible by head_size ({head_size})"

        self.d_model = d_model
        self.head_size = head_size
        self.n_heads = n_heads or d_model // head_size
        self.lora_rank = lora_rank
        self.chunk_size = chunk_size
        self.is_rwkv7 = True

        Linear = TernaryLinear if use_ternary else nn.Linear
        linear_kwargs = dict(group_size=ternary_group_size) if use_ternary else {}

        # Main projections
        self.W_r = Linear(d_model, d_model, **linear_kwargs)
        self.W_k = Linear(d_model, d_model, **linear_kwargs)
        self.W_v = Linear(d_model, d_model, **linear_kwargs)
        self.W_o = Linear(d_model, d_model, **linear_kwargs)

        # Token shift interpolation (learnable per-channel mixing of current vs previous token)
        self.x_r = nn.Parameter(torch.zeros(d_model))
        self.x_k = nn.Parameter(torch.zeros(d_model))
        self.x_v = nn.Parameter(torch.zeros(d_model))
        self.x_w = nn.Parameter(torch.zeros(d_model))
        self.x_a = nn.Parameter(torch.zeros(d_model))
        self.x_g = nn.Parameter(torch.zeros(d_model))

        # Decay gate (w) via LoRA: d_model -> lora_rank -> d_model
        self.w0 = nn.Parameter(torch.zeros(d_model))  # base decay
        self.w1 = nn.Parameter(torch.zeros(d_model, lora_rank))
        self.w2 = nn.Parameter(torch.zeros(lora_rank, d_model))

        # Learning rate gate (a) via LoRA
        self.a0 = nn.Parameter(torch.zeros(d_model))
        self.a1 = nn.Parameter(torch.zeros(d_model, lora_rank))
        self.a2 = nn.Parameter(torch.zeros(lora_rank, d_model))

        # Output gate (g) via LoRA (larger rank for expressiveness)
        gate_rank = min(lora_rank * 4, d_model)
        self.g1 = nn.Parameter(torch.zeros(d_model, gate_rank))
        self.g2 = nn.Parameter(torch.zeros(gate_rank, d_model))

        # Key scaling and modulation
        self.k_k = nn.Parameter(torch.ones(d_model) * 0.7)
        self.k_a = nn.Parameter(torch.ones(d_model))

        # Receptance-key scaling for linear attention term
        self.r_k = nn.Parameter(torch.zeros(self.n_heads, self.head_size))

        # V-first residual (cross-layer value blending)
        self.v0 = nn.Parameter(torch.zeros(d_model))
        self.v1 = nn.Parameter(torch.zeros(d_model, lora_rank))
        self.v2 = nn.Parameter(torch.zeros(lora_rank, d_model))

        # GroupNorm (per head)
        self.group_norm = nn.GroupNorm(self.n_heads, d_model, eps=64e-5)

        self.use_chunked = True

        self._init_weights()

    def _init_weights(self):
        # Initialize LoRA matrices with small random values
        for name in ['w1', 'w2', 'a1', 'a2', 'g1', 'g2', 'v1', 'v2']:
            param = getattr(self, name)
            nn.init.normal_(param, std=0.01)

        # Decay base: start with moderate decay (log scale)
        nn.init.uniform_(self.w0, -1.0, 0.0)

        # Main projection weights
        for proj in [self.W_r, self.W_k, self.W_v, self.W_o]:
            if hasattr(proj, 'weight') and not isinstance(proj, TernaryLinear):
                nn.init.xavier_uniform_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)

    @torch.compiler.disable()
    def forward_chunked(
        self, x: torch.Tensor, v_first: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        H = self.n_heads
        N = self.head_size

        x_prev = F.pad(x, (0, 0, 1, -1))
        xx = x_prev - x

        xr = x + xx * self.x_r
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xw = x + xx * self.x_w
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.W_r(xr).view(B, T, H, N)
        k = self.W_k(xk).view(B, T, H, N)
        v = self.W_v(xv).view(B, T, H, N)

        if v_first is None:
            v_first = v
        else:
            alpha = torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2).view(B, T, H, N)
            v = v + (v_first - v) * alpha

        w = -F.softplus(-(self.w0 + (torch.tanh(xw @ self.w1) @ self.w2))) - 0.5
        w = torch.exp(w).view(B, T, H, N)

        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2).view(B, T, H, N)

        g = (torch.sigmoid(xg @ self.g1) @ self.g2).view(B, T, H, N)

        kk = k * self.k_k.view(1, 1, H, N)
        kk = F.normalize(kk, dim=-1, p=2.0)
        k = k * (1 + (a - 1) * self.k_a.view(1, 1, H, N))

        r_shaped = r.view(B, T, H, N, 1)
        k_shaped = k.view(B, T, H, 1, N)
        v_shaped = v.view(B, T, H, N, 1)
        w_shaped = w.view(B, T, H, 1, N)

        aa_shaped = (kk * a).view(B, T, H, N, 1)
        bb_shaped = (-kk).view(B, T, H, 1, N)

        aa_bb = aa_shaped @ bb_shaped

        vk = v_shaped @ k_shaped

        state = torch.zeros(B, H, N, N, device=x.device, dtype=torch.float32)
        out = torch.zeros(B, T, H, N, device=x.device, dtype=x.dtype)

        chunk_size = self.chunk_size
        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            chunk_len = chunk_end - chunk_start

            w_chunk = w_shaped[:, chunk_start:chunk_end]
            aa_bb_chunk = aa_bb[:, chunk_start:chunk_end]
            vk_chunk = vk[:, chunk_start:chunk_end]
            r_chunk = r_shaped[:, chunk_start:chunk_end]

            for t in range(chunk_len):
                state = state * w_chunk[:, t] + state @ aa_bb_chunk[:, t] + vk_chunk[:, t]
                out[:, chunk_start + t] = (state @ r_chunk[:, t]).squeeze(-1)

        out_flat = out.view(B, T, C)
        out_flat = self.group_norm(out_flat.transpose(1, 2)).transpose(1, 2)

        r_flat = r.view(B, T, C)
        k_flat = k.view(B, T, C)
        v_flat = v.view(B, T, C)
        r_k = self.r_k.view(1, 1, H, N).expand_as(r).reshape(B, T, C)
        la_term = (r_flat * k_flat * r_k).view(B, T, H, N).sum(dim=-1, keepdim=True)
        la_term = la_term.expand_as(v).reshape(B, T, C) * v_flat
        out_flat = out_flat + la_term

        output = self.W_o(out_flat * g.view(B, T, C))

        return output, v_first

    @torch.compiler.disable()
    def forward(
        self, x: torch.Tensor, v_first: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_chunked:
            return self.forward_chunked(x, v_first)

        B, T, C = x.shape
        H = self.n_heads
        N = self.head_size

        x_prev = F.pad(x, (0, 0, 1, -1))
        xx = x_prev - x

        xr = x + xx * self.x_r
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xw = x + xx * self.x_w
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.W_r(xr).view(B, T, H, N)
        k = self.W_k(xk).view(B, T, H, N)
        v = self.W_v(xv).view(B, T, H, N)

        if v_first is None:
            v_first = v
        else:
            alpha = torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2).view(B, T, H, N)
            v = v + (v_first - v) * alpha

        w = -F.softplus(-(self.w0 + (torch.tanh(xw @ self.w1) @ self.w2))) - 0.5
        w = torch.exp(w).view(B, T, H, N)

        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2).view(B, T, H, N)

        g = (torch.sigmoid(xg @ self.g1) @ self.g2).view(B, T, H, N)

        kk = k * self.k_k.view(1, 1, H, N)
        kk = F.normalize(kk, dim=-1, p=2.0)
        k = k * (1 + (a - 1) * self.k_a.view(1, 1, H, N))

        state = torch.zeros(B, H, N, N, device=x.device, dtype=torch.float32)
        out = torch.zeros(B, T, H, N, device=x.device, dtype=x.dtype)

        for t in range(T):
            rr = r[:, t].view(B, H, N, 1)
            kk_t = k[:, t].view(B, H, 1, N)
            vv = v[:, t].view(B, H, N, 1)
            ww = w[:, t].view(B, H, 1, N)

            aa = (kk[:, t] * a[:, t]).view(B, H, N, 1)
            bb = (-kk[:, t]).view(B, H, 1, N)

            state = state * ww + state @ aa @ bb + vv @ kk_t

            out[:, t] = (state @ rr).squeeze(-1)

        out_flat = out.view(B, T, C)
        out_flat = self.group_norm(out_flat.transpose(1, 2)).transpose(1, 2)

        r_flat = r.view(B, T, C)
        k_flat = k.view(B, T, C)
        v_flat = v.view(B, T, C)
        r_k = self.r_k.view(1, 1, H, N).expand_as(r).reshape(B, T, C)
        la_term = (r_flat * k_flat * r_k).view(B, T, H, N).sum(dim=-1, keepdim=True)
        la_term = la_term.expand_as(v).reshape(B, T, C) * v_flat
        out_flat = out_flat + la_term

        output = self.W_o(out_flat * g.view(B, T, C))

        return output, v_first
