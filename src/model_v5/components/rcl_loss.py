"""Recursive Consensus Loss (R4-A3-M3).

Weights per-iteration losses by convergence: converged iterations dominate gradient.
c_l = 1 - ||h_l - h_{l-1}||^2 / (||h_l||^2 + eps)
L_RCL = (1/Z) * sum(c_l^beta * L_l)
"""
import torch
import torch.nn.functional as F


def compute_rcl_loss(
    intermediate_states: list,
    targets: torch.Tensor,
    output_proj: torch.nn.Linear,
    beta: float = 1.0,
    sample_every: int = 4,
) -> torch.Tensor:
    """Compute RCL from intermediate RecursiveSSM states.

    Args:
        intermediate_states: list of (batch, seq, d_model) tensors, one per iteration
        targets: (batch, seq) target token ids
        output_proj: projection to vocab (shared with main output head)
        beta: convergence weighting exponent (0=uniform, 2=strong convergence bias)
        sample_every: compute loss every N iterations (efficiency)

    Returns:
        Scalar RCL loss.
    """
    if len(intermediate_states) < 2:
        return torch.tensor(0.0, device=intermediate_states[0].device if intermediate_states else 'cpu')

    total_loss = torch.tensor(0.0, device=intermediate_states[0].device)
    total_weight = 0.0

    for i in range(sample_every, len(intermediate_states), sample_every):
        h_l = intermediate_states[i]
        h_prev = intermediate_states[i - sample_every]

        # Convergence metric
        diff_norm = (h_l - h_prev).pow(2).sum(dim=-1).mean()
        h_norm = h_l.pow(2).sum(dim=-1).mean() + 1e-8
        c_l = 1.0 - diff_norm / h_norm
        c_l = c_l.clamp(min=0.0, max=1.0)

        # Per-iteration loss via shared output projection
        logits_l = output_proj(h_l)
        loss_l = F.cross_entropy(
            logits_l.view(-1, logits_l.shape[-1]),
            targets.view(-1),
            ignore_index=-100,
        )

        weight = c_l.pow(beta).item()
        total_loss = total_loss + weight * loss_l
        total_weight += weight

    if total_weight > 0:
        total_loss = total_loss / total_weight

    return total_loss
