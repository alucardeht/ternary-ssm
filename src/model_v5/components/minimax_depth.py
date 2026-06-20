"""Minimax Depth Fairness (R4-A3-P6).

Minimize MAXIMUM per-depth loss (Rawlsian fairness).
Hard: g = grad(L_{argmax}), soft: g = sum softmax(L/tau) * grad(L_l)
"""
import torch
import torch.nn.functional as F


def compute_minimax_depth_loss(
    intermediate_states: list,
    targets: torch.Tensor,
    output_proj: torch.nn.Module,
    tau: float = 1.0,
    sample_every: int = 4,
    hard: bool = False,
) -> torch.Tensor:
    """Compute minimax depth fairness loss.

    Args:
        intermediate_states: per-iteration hidden states
        targets: target ids
        output_proj: vocab projection
        tau: softmax temperature (lower = harder minimax)
        sample_every: compute at every N-th depth
        hard: if True, only backprop through worst depth

    Returns:
        Scalar loss (worst-case or softmax-weighted).
    """
    if len(intermediate_states) < 2:
        return torch.tensor(0.0, device=intermediate_states[0].device if intermediate_states else 'cpu')

    per_depth_losses = []

    for i in range(sample_every, len(intermediate_states), sample_every):
        h = intermediate_states[i]
        logits = output_proj(h)
        loss_d = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            targets.view(-1),
            ignore_index=-100,
        )
        per_depth_losses.append(loss_d)

    if not per_depth_losses:
        return torch.tensor(0.0, device=intermediate_states[0].device)

    losses = torch.stack(per_depth_losses)

    if hard:
        # Hard minimax: only worst depth
        return losses.max()
    else:
        # Soft minimax: softmax-weighted
        weights = F.softmax(losses.detach() / tau, dim=0)
        return (weights * losses).sum()
