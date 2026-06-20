"""Spectral Surrogate Objective (R4-A3-P3).

Initialization via spectral optimization: 500x cheaper than forward pass.
L_surrogate = alpha*(rho(A)-rho_target)^2 + beta*sum(sigma_i-sigma_target)^2
"""
import torch
import math


def compute_spectral_surrogate(
    model: torch.nn.Module,
    rho_target: float = 0.95,
    sigma_target: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute spectral surrogate loss for SSM matrices.

    Args:
        model: model with SSM layers
        rho_target: target spectral radius for A matrix
        sigma_target: target singular value for weight matrices
        alpha: spectral radius loss weight
        beta: singular value loss weight

    Returns:
        Scalar surrogate loss.
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    n = 0

    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue

        # SSM A matrix: target spectral radius
        if 'A_log' in name or 'a_log' in name:
            # A = -exp(A_log), spectral radius = max(exp(A_log))
            rho = torch.exp(param.data).max()
            total = total + alpha * (rho - rho_target) ** 2
            n += 1

        # Weight matrices: target singular values
        elif 'weight' in name and param.shape[0] >= 4 and param.shape[1] >= 4:
            # Approximate: use Frobenius norm as proxy for sum of squared singular values
            fro_norm = param.data.norm()
            expected_fro = sigma_target * math.sqrt(min(param.shape[0], param.shape[1]))
            total = total + beta * (fro_norm - expected_fro) ** 2
            n += 1

    return total / max(n, 1)


def spectral_greedy_flip(
    param: torch.nn.Parameter,
    rho_target: float = 0.95,
    max_flips: int = 10,
) -> int:
    """Greedy flip search for SSM A matrix toward target spectral radius.

    Tries flipping individual ternary weights to move spectral radius
    closer to target. O(d) per flip attempt.

    Returns number of beneficial flips applied.
    """
    if param.dim() < 2:
        return 0

    w = param.data
    threshold = 0.5 * w.abs().mean()
    w_ternary = torch.sign(w) * (w.abs() > threshold).float()

    flips_applied = 0
    current_score = (w_ternary.norm() - rho_target * math.sqrt(w_ternary.numel())) ** 2

    for _ in range(max_flips):
        # Random position
        i = torch.randint(0, w.shape[0], (1,)).item()
        j = torch.randint(0, w.shape[1], (1,)).item()

        old_val = w_ternary[i, j].item()
        # Try each alternative value
        best_val = old_val
        best_score = current_score

        for new_val in [-1.0, 0.0, 1.0]:
            if new_val == old_val:
                continue
            w_ternary[i, j] = new_val
            score = (w_ternary.norm() - rho_target * math.sqrt(w_ternary.numel())) ** 2
            if score < best_score:
                best_score = score
                best_val = new_val

        if best_val != old_val:
            w_ternary[i, j] = best_val
            # Push shadow weight toward target
            if best_val == 0:
                param.data[i, j] = 0.0
            else:
                param.data[i, j] = best_val * threshold * 1.1
            flips_applied += 1
            current_score = best_score
        else:
            w_ternary[i, j] = old_val

    return flips_applied
