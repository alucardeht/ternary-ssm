"""Dynamic Loss Weighting (R4-A3-F8).

Auto-balances multi-component losses by magnitude.
Formula: w_i(t) = (L_mean / L_i(t))^gamma
"""
import torch
import math


def apply_dynamic_loss_weighting(
    losses: dict,
    gamma: float = 1.0,
) -> tuple:
    """Reweight losses so no single component dominates.

    Args:
        losses: dict mapping name -> scalar loss tensor
        gamma: exponent controlling balance strength (0=uniform, 1=full balance)

    Returns:
        (total_loss, info_dict with per-component weights)
    """
    active = {k: v for k, v in losses.items()
              if v is not None and (isinstance(v, torch.Tensor) and v.item() > 0)}

    if len(active) <= 1:
        total = sum(v for v in active.values())
        if isinstance(total, (int, float)):
            total = torch.tensor(total, requires_grad=True)
        return total, {f'dlw_{k}_w': 1.0 for k in active}

    # Geometric mean of all loss magnitudes
    log_sum = sum(math.log(max(v.item(), 1e-8)) for v in active.values())
    geom_mean = math.exp(log_sum / len(active))

    total = torch.tensor(0.0, device=next(iter(active.values())).device)
    info = {}
    for name, loss_val in active.items():
        w = (geom_mean / max(loss_val.item(), 1e-8)) ** gamma
        total = total + w * loss_val
        info[f'dlw_{name}_w'] = w

    return total, info
