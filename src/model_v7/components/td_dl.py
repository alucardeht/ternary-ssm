"""Temporal Difference Depth Learning (R4-A3-P8).

32 recursive applications as MDP. TD learning for per-depth credit assignment.
V_l = -CE(proj(h_l), y), delta_l = V_{l+1} - V_l
"""
import torch
import torch.nn.functional as F


def compute_td_depth_loss(
    intermediate_states: list,
    targets: torch.Tensor,
    output_proj: torch.nn.Module,
    td_lambda: float = 0.8,
    sample_every: int = 2,
) -> torch.Tensor:
    """Compute TD(lambda) loss for depth credit assignment.

    Value at depth l = negative CE loss at that depth.
    TD error = V_{l+1} - V_l (how much did the next iteration help?)

    Args:
        intermediate_states: per-iteration hidden states
        targets: target ids
        output_proj: projection to vocab
        td_lambda: eligibility trace decay (0=TD(0), 1=Monte Carlo)
        sample_every: compute at every N-th iteration

    Returns:
        TD-weighted loss that assigns credit per depth.
    """
    if len(intermediate_states) < 2:
        return torch.tensor(0.0, device=intermediate_states[0].device if intermediate_states else 'cpu')

    # Compute per-depth values
    values = []
    sampled_indices = list(range(0, len(intermediate_states), sample_every))

    for i in sampled_indices:
        h = intermediate_states[i]
        logits = output_proj(h)
        ce = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            targets.view(-1),
            ignore_index=-100,
        )
        values.append(-ce)  # Value = negative loss (higher = better)

    if len(values) < 2:
        return torch.tensor(0.0, device=intermediate_states[0].device)

    # TD errors
    td_errors = []
    for i in range(len(values) - 1):
        delta = values[i + 1] - values[i]  # How much did next depth improve?
        td_errors.append(delta)

    # TD(lambda) returns: weighted sum of n-step returns
    # G_t^lambda = (1-lambda) * sum_{n=1}^{inf} lambda^{n-1} * G_t^{(n)}
    # Simplified: weight TD errors by lambda^k eligibility traces
    total_loss = torch.tensor(0.0, device=intermediate_states[0].device)
    eligibility = 1.0

    for i, delta in enumerate(td_errors):
        # We want to maximize improvement (positive TD errors)
        # Loss = -delta (penalize lack of improvement)
        total_loss = total_loss - eligibility * delta
        eligibility *= td_lambda

    # Normalize by number of terms
    total_loss = total_loss / max(len(td_errors), 1)

    return total_loss
