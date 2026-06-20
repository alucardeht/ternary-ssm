"""Information-Bottleneck Layerwise Loss (R4-A3-M7).

Per-iteration auxiliary losses:
- Compression: ||h_l - h_{l-1}||^2 / ||h_{l-1}||^2
- Relevance: -CE(probe(h_l), y)
- L_aux = sum lambda_l * [-beta*Relevance + gamma*max(0, eps-Compression)]
"""
import torch
import torch.nn.functional as F


class IBLLProbe(torch.nn.Module):
    """Lightweight probe for intermediate supervision."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = torch.nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


def compute_ibll_loss(
    intermediate_states: list,
    targets: torch.Tensor,
    probe: torch.nn.Module,
    beta: float = 1.0,
    gamma: float = 0.1,
    eps: float = 0.01,
    sample_every: int = 8,
) -> torch.Tensor:
    """Compute IB layerwise loss from intermediate states.

    Args:
        intermediate_states: list of (batch, seq, d_model) per iteration
        targets: (batch, seq) target ids
        probe: projection to vocab for relevance computation
        beta: relevance weight
        gamma: compression encouragement weight
        eps: minimum compression threshold
        sample_every: compute at every N-th iteration

    Returns:
        Scalar IBLL loss.
    """
    if len(intermediate_states) < 2:
        return torch.tensor(0.0, device=intermediate_states[0].device if intermediate_states else 'cpu')

    total = torch.tensor(0.0, device=intermediate_states[0].device)
    n_terms = 0

    for i in range(sample_every, len(intermediate_states), sample_every):
        h_l = intermediate_states[i]
        h_prev = intermediate_states[max(0, i - sample_every)]

        # Compression: how much did h change?
        compress = (h_l - h_prev).pow(2).sum(dim=-1).mean()
        h_prev_norm = h_prev.pow(2).sum(dim=-1).mean() + 1e-8
        compress_ratio = compress / h_prev_norm

        # Relevance: can we predict targets from this layer?
        logits_l = probe(h_l)
        relevance = F.cross_entropy(
            logits_l.view(-1, logits_l.shape[-1]),
            targets.view(-1),
            ignore_index=-100,
        )

        # IB objective: maximize relevance, encourage minimum compression
        ib_loss = -beta * (-relevance) + gamma * F.relu(torch.tensor(eps) - compress_ratio)
        total = total + ib_loss
        n_terms += 1

    return total / max(n_terms, 1)
