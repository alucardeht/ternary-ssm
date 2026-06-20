"""Mixture of Ternary Experts (MoTE) FFN layer.

Each expert is a small SwiGLU FFN with ternary weights.
Router is FP16 for routing precision.
Designed for from-scratch training (no pre-trained shared expert).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ternary import TernaryLinear


class TernaryFFNExpert(nn.Module):
    """Single SwiGLU FFN expert with optional ternary weights."""

    def __init__(self, d_model: int, d_ff: int, use_ternary: bool = True,
                 ternary_group_size: int = 128, activation_fn: str = 'silu'):
        super().__init__()
        Linear = TernaryLinear if use_ternary else nn.Linear
        linear_kwargs = {'group_size': ternary_group_size} if use_ternary else {}

        self.mlp_gate = Linear(d_model, d_ff, bias=False, **linear_kwargs)
        self.mlp_up = Linear(d_model, d_ff, bias=False, **linear_kwargs)
        self.mlp_down = Linear(d_ff, d_model, bias=False, **linear_kwargs)
        self.act_fn = F.relu if activation_fn == 'relu' else F.silu

        # SwiGLU-correct init: override TernaryLinear default gain=1/sqrt(n)
        # which is too aggressive for 3-matmul SwiGLU (compounds to ~0.0001x)
        if use_ternary:
            nn.init.xavier_uniform_(self.mlp_gate.weight, gain=1.0)
            nn.init.xavier_uniform_(self.mlp_up.weight, gain=1.0)
            nn.init.xavier_uniform_(self.mlp_down.weight, gain=1.0 / math.sqrt(d_ff))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_down(self.act_fn(self.mlp_gate(x)) * self.mlp_up(x))


class TernaryMoEFFN(nn.Module):
    """Mixture of Ternary Experts FFN layer.

    Routes each token to top_k experts via learned softmax router.
    All experts are ternary SwiGLU FFNs. Router stays FP16.
    """

    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4,
                 top_k: int = 1, use_ternary: bool = True,
                 ternary_group_size: int = 128,
                 load_balance_coeff: float = 0.01, activation_fn: str = 'silu'):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_coeff = load_balance_coeff
        self._last_aux_loss = 0.0

        # Router: always FP16 for precision
        self.router = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)

        # Experts: ternary SwiGLU FFNs
        self.experts = nn.ModuleList([
            TernaryFFNExpert(d_model, d_ff, use_ternary, ternary_group_size, activation_fn)
            for _ in range(num_experts)
        ])

        # Learnable output scale for MoE magnitude control
        self.output_scale = nn.Parameter(torch.ones(1))

        # Routing stats for monitoring
        self.register_buffer('_expert_counts',
                            torch.zeros(num_experts), persistent=False)
        self.register_buffer('_total_tokens',
                            torch.tensor(0.0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)
        N = x_flat.shape[0]

        # Route
        logits = self.router(x_flat)
        probs = F.softmax(logits, dim=-1)

        # Load balancing auxiliary loss (Switch Transformer style)
        if self.training and self.load_balance_coeff > 0:
            tokens_per_expert = F.one_hot(probs.argmax(dim=-1), self.num_experts).float().mean(dim=0)
            router_prob_per_expert = probs.mean(dim=0)
            self._last_aux_loss = self.load_balance_coeff * self.num_experts * (tokens_per_expert * router_prob_per_expert).sum()
        else:
            self._last_aux_loss = 0.0

        if self.top_k == 1:
            top_prob, top_idx = probs.max(dim=-1)

            output = torch.zeros_like(x_flat)
            for i, expert in enumerate(self.experts):
                mask = (top_idx == i)
                if not mask.any():
                    continue
                expert_input = x_flat[mask]
                expert_output = expert(expert_input)
                output[mask] = expert_output * top_prob[mask].unsqueeze(-1)

            if not torch.compiler.is_compiling():
                with torch.no_grad():
                    for i in range(self.num_experts):
                        self._expert_counts[i] += (top_idx == i).sum().float()
                    self._total_tokens += N
        else:
            top_k_probs, top_k_idx = probs.topk(self.top_k, dim=-1)
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

            output = torch.zeros_like(x_flat)
            for i, expert in enumerate(self.experts):
                expert_in_topk = (top_k_idx == i)
                token_mask = expert_in_topk.any(dim=-1)
                if not token_mask.any():
                    continue
                expert_input = x_flat[token_mask]
                expert_output = expert(expert_input)
                slot_mask = expert_in_topk[token_mask]
                expert_prob = (top_k_probs[token_mask] * slot_mask.float()).sum(dim=-1)
                output[token_mask] += expert_output * expert_prob.unsqueeze(-1)

            if not torch.compiler.is_compiling():
                with torch.no_grad():
                    for i in range(self.num_experts):
                        self._expert_counts[i] += (top_k_idx == i).any(dim=-1).sum().float()
                    self._total_tokens += N

        return (output * self.output_scale).reshape(B, T, D)

    def get_stats(self) -> dict:
        total = self._total_tokens.item()
        if total == 0:
            return {'total_tokens': 0}
        counts = self._expert_counts.cpu().numpy()
        fractions = counts / total
        return {
            'total_tokens': int(total),
            'expert_fractions': fractions.tolist(),
            'balance_ratio': float(fractions.min() / max(fractions.max(), 1e-8)),
        }

    def reset_stats(self):
        self._expert_counts.zero_()
        self._total_tokens.zero_()
