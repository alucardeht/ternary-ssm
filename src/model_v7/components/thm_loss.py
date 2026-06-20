"""Ternary Hamming Margin Loss (R4-A3-M1).

Forces ternary weight rows to be maximally distinct via Hamming distance margins.
Uses element-wise comparison (equivalent to POPCOUNT on packed representations).
"""
import torch
import torch.nn.functional as F


def compute_thm_loss(
    model: torch.nn.Module,
    margin: float = 0.3,
    max_pairs: int = 100,
) -> torch.Tensor:
    """Compute Hamming margin regularization over ternary layers.

    For each TernaryLinear, compute pairwise Hamming distances between
    weight rows and penalize pairs closer than margin * d.

    Args:
        model: model with TernaryLinear modules
        margin: target minimum Hamming distance ratio (0-1)
        max_pairs: max random pairs to check per layer (for efficiency)

    Returns:
        Scalar regularization loss.
    """
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n_layers = 0

    for module in model.modules():
        if not hasattr(module, 'quantize_enabled'):
            continue
        if not hasattr(module, 'weight'):
            continue

        # Get ternary weights from shadow
        w = module.weight.data
        threshold = getattr(module, 'threshold_coeff', 0.5) * w.abs().mean()
        w_ternary = torch.sign(w) * (w.abs() > threshold).float()

        out_dim, in_dim = w_ternary.shape[:2] if w_ternary.dim() >= 2 else (w_ternary.shape[0], 1)
        if out_dim < 2 or in_dim < 2:
            continue

        w_flat = w_ternary.view(out_dim, -1)
        d = w_flat.shape[1]
        target_dist = margin * d

        # Random pairs for efficiency
        n_pairs = min(max_pairs, out_dim * (out_dim - 1) // 2)
        idx_i = torch.randint(0, out_dim, (n_pairs,), device=w.device)
        idx_j = torch.randint(0, out_dim, (n_pairs,), device=w.device)
        # Ensure different indices
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]

        if len(idx_i) == 0:
            continue

        # Hamming distance = count of differing positions
        hamming = (w_flat[idx_i] != w_flat[idx_j]).float().sum(dim=-1)

        # Hinge loss: penalize pairs closer than target
        violations = F.relu(target_dist - hamming)
        if violations.sum() > 0:
            total_loss = total_loss + violations.mean()
            n_layers += 1

    return total_loss / max(n_layers, 1)


def compute_thm_loss_detached(model, margin=0.3, max_pairs=100):
    """Same as compute_thm_loss but returns detached scalar for monitoring."""
    with torch.no_grad():
        return compute_thm_loss(model, margin, max_pairs).item()
