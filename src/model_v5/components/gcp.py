"""Gradient Consensus Polarization (R4-A3-M5).

For shared weights used across L recursive applications:
- Consensus: c(j) = mean of per-application gradients
- Conflict: sigma^2(j) = variance of per-application gradients
- Agreement: r(j) = c^2 / (c^2 + sigma^2)
- Polarized: g_GCP = L * c * r^alpha

Amplifies consensus gradients across recursive applications and suppresses ambiguous ones.
"""
import torch


def apply_gcp(
    grad: torch.Tensor,
    per_app_grads: list,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Apply Gradient Consensus Polarization to shared weight gradients.

    Args:
        grad: current accumulated gradient (L applications summed)
        per_app_grads: list of L gradient tensors, one per application
        alpha: polarization strength (0=no effect, 1=standard, 2=aggressive)

    Returns:
        Polarized gradient tensor (same shape as grad).
    """
    if len(per_app_grads) < 2:
        return grad

    L = len(per_app_grads)
    stacked = torch.stack(per_app_grads)  # (L, *shape)

    # Consensus = mean
    c = stacked.mean(dim=0)  # (*shape)

    # Variance
    sigma2 = stacked.var(dim=0, unbiased=False)  # (*shape)

    # Agreement ratio
    c2 = c * c
    r = c2 / (c2 + sigma2 + 1e-8)  # (*shape), in [0, 1]

    # Polarized gradient
    g_gcp = L * c * (r ** alpha)

    return g_gcp


def apply_gcp_simple(
    grad: torch.Tensor,
    n_applications: int = 32,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Simplified GCP when per-application gradients aren't available.

    Uses a heuristic: if grad magnitude is large relative to weight norm,
    it's likely consensual; if small, likely conflicting.
    This is an approximation that doesn't require per-app grad tracking.

    Args:
        grad: accumulated gradient from all applications
        n_applications: number of recursive applications
        alpha: polarization strength

    Returns:
        Filtered gradient.
    """
    # Per-element estimated consensus = grad / L
    c = grad / n_applications

    # Heuristic variance estimate: assume uniform distribution of signs
    # If consensus is strong (|c| >> 0), variance is low
    # If consensus is weak (|c| ~ 0), variance is high
    c_abs = c.abs()
    c_max = c_abs.max().clamp(min=1e-8)

    # Agreement ratio heuristic: |c| / max(|c|)
    r = c_abs / c_max

    # Polarize
    g_gcp = n_applications * c * (r ** alpha)

    return g_gcp
