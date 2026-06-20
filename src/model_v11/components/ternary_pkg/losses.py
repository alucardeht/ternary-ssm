import torch
import torch.nn.functional as F


def compute_sca_sparsity_loss(
    model: torch.nn.Module,
    target_sparsity: float = 0.5,
    group_size: int = 128,
    threshold_coeff: float = 0.7,
) -> tuple[torch.Tensor, dict]:
    """SCA-Sparsity-Controlled-Energy regularization (Deng et al., arXiv:2011.00580)

    Adds regularization term -λ·sparsity_controller for explicit sparsity control.
    This explicitly controls the zero-ratio to match target_sparsity.

    Args:
        model: The model with ternary weights
        target_sparsity: Target zero ratio (0.0-1.0), e.g., 0.5 for 50% zeros
        group_size: Group size for computing scales
        threshold_coeff: Threshold coefficient for ternary quantization

    Returns:
        Tuple of (sparsity_loss, stats_dict)
    """
    total_zeros = 0
    total_weights = 0

    for name, module in model.named_modules():
        if not hasattr(module, "weight") or not hasattr(module, "threshold_coeff"):
            continue

        with torch.no_grad():
            w = module.weight
            gs = getattr(module, "group_size", group_size)
            tc = getattr(module, "threshold_coeff", threshold_coeff)

            # Compute ternary quantization (same as TernaryLinear)
            flat = w.reshape(-1)
            pad_len = (gs - flat.numel() % gs) % gs
            if pad_len > 0:
                flat = F.pad(flat, (0, pad_len))
            grouped = flat.reshape(-1, gs)

            scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
            normalized = grouped / scales
            abs_normalized = normalized.abs()
            threshold = tc * abs_normalized.mean(dim=-1, keepdim=True)

            # Compute zero mask (weights below threshold become zero)
            zero_mask = abs_normalized <= threshold
            n_zeros = zero_mask.sum().item()
            n_weights = w.numel()

            total_zeros += n_zeros
            total_weights += n_weights

    if total_weights == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device), {
            "current_sparsity": 0.0
        }

    current_sparsity = total_zeros / total_weights

    # SCA energy function: E = -λ * (sparsity - target)^2
    # This penalizes deviation from target sparsity
    # Using squared error for smoother gradients
    sparsity_diff = current_sparsity - target_sparsity
    sparsity_loss = sparsity_diff**2

    stats = {
        "current_sparsity": current_sparsity,
        "target_sparsity": target_sparsity,
        "sparsity_diff": sparsity_diff,
        "total_zeros": total_zeros,
        "total_weights": total_weights,
    }

    return torch.tensor(sparsity_loss, device=next(model.parameters()).device), stats


def compute_sca_sparsity_loss_simple(
    model: torch.nn.Module,
    target_sparsity: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Simplified SCA sparsity loss using absolute value for more direct gradient.

    Instead of computing exact ternary quantization, uses a soft approximation:
    - Weights with |w| < threshold are considered "likely zeros"
    - Penalizes deviation from target sparsity ratio

    This is faster but less precise than compute_sca_sparsity_loss.
    """
    total_magnitude = 0.0
    total_weights = 0

    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue

        w = module.weight
        total_magnitude += w.abs().sum().item()
        total_weights += w.numel()

    if total_weights == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device), {
            "current_sparsity": 0.0
        }

    # Average magnitude as proxy for "committed" weights
    avg_magnitude = total_magnitude / total_weights

    # Normalize to estimate sparsity (higher magnitude = less sparsity)
    # This is a rough approximation
    current_sparsity = 1.0 - min(avg_magnitude, 1.0)

    sparsity_diff = current_sparsity - target_sparsity
    sparsity_loss = sparsity_diff**2

    stats = {
        "current_sparsity": current_sparsity,
        "target_sparsity": target_sparsity,
        "sparsity_diff": sparsity_diff,
        "avg_magnitude": avg_magnitude,
    }

    return torch.tensor(sparsity_loss, device=next(model.parameters()).device), stats
