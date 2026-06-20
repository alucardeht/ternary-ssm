"""Recurrence-Aware Flip Utility (R4-A3-M6).

Backward recurrence: psi_k = delta_k + A^T * psi_{k+1}
Utility: U = Psi^T @ H
FlipBenefit(i,j; cur->tgt) = -(tgt - cur) * U_ij
"""
import torch


def compute_rafu_utility(
    forward_states: list,
    backward_signals: list,
    A_matrix: torch.Tensor = None,
) -> torch.Tensor:
    """Compute per-weight flip utility from forward/backward recurrence.

    Simplified version: utility per shared weight = correlation between
    forward activations and backward error signals across all iterations.

    Args:
        forward_states: list of L tensors (batch, seq, d_model) from forward
        backward_signals: list of L gradient tensors from backward hooks
        A_matrix: optional state transition matrix for recurrence (unused in simplified)

    Returns:
        Utility matrix (d_model, d_model) - higher values = more impactful positions.
    """
    if len(forward_states) == 0 or len(backward_signals) == 0:
        return None

    L = min(len(forward_states), len(backward_signals))
    device = forward_states[0].device

    # Simplified: accumulate H^T @ delta across iterations
    d = forward_states[0].shape[-1]
    utility = torch.zeros(d, d, device=device)

    for k in range(L):
        h_k = forward_states[k].detach()  # (batch, seq, d)
        delta_k = backward_signals[k].detach()  # (batch, seq, d)

        # Average over batch and seq: correlation per (i,j) pair
        h_flat = h_k.reshape(-1, d)  # (B*S, d)
        d_flat = delta_k.reshape(-1, d)  # (B*S, d)

        utility += (h_flat.T @ d_flat) / h_flat.shape[0]

    return utility / L


def apply_rafu_flip_filter(
    grad: torch.Tensor,
    utility: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Filter gradient based on RAFU utility - suppress low-utility flips.

    Args:
        grad: parameter gradient (out_dim, in_dim)
        utility: utility matrix (d_model, d_model) or compatible shape
        threshold: suppress gradients where |utility| < threshold * max(|utility|)

    Returns:
        Filtered gradient.
    """
    if utility is None or grad.shape != utility.shape:
        return grad

    u_abs = utility.abs()
    u_max = u_abs.max().clamp(min=1e-8)
    mask = (u_abs / u_max) > threshold

    return grad * mask.float()
