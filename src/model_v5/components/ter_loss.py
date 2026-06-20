"""Ternary Entropy Rate Loss (R4-A3-M4).

Penalizes output distributions more peaked than ternary channel capacity allows.
C_upper = (d/2) * p_active * log(1 + ||h||^2 / (d * sigma^2_noise))
H_floor = max(0, log(V) - C_upper)
L_TER = mu * ReLU(H_floor - H(p))
"""
import torch
import torch.nn.functional as F
import math


def compute_ter_loss(
    logits: torch.Tensor,
    model: torch.nn.Module,
    vocab_size: int,
    mu: float = 0.1,
    noise_var: float = 0.01,
) -> torch.Tensor:
    """Compute ternary entropy rate regularizer.

    Args:
        logits: (batch*seq, vocab) raw logits
        model: model to estimate p_active (fraction of non-zero ternary weights)
        vocab_size: vocabulary size V
        mu: penalty weight
        noise_var: estimated noise variance in activations

    Returns:
        Scalar penalty loss.
    """
    # Estimate p_active from ternary layers
    total_params = 0
    active_params = 0
    for module in model.modules():
        if hasattr(module, 'quantize_enabled') and hasattr(module, 'weight'):
            w = module.weight.data
            threshold = getattr(module, 'threshold_coeff', 0.5) * w.abs().mean()
            n = w.numel()
            n_active = (w.abs() > threshold).sum().item()
            total_params += n
            active_params += n_active

    p_active = active_params / max(total_params, 1)

    # Channel capacity upper bound
    h_norm_sq = logits.detach().pow(2).mean()  # ||h||^2 proxy
    d = logits.shape[-1]
    C_upper = (d / 2.0) * p_active * math.log(1 + h_norm_sq.item() / (d * noise_var + 1e-8))

    # Entropy floor
    log_V = math.log(vocab_size)
    H_floor = max(0.0, log_V - C_upper)

    # Actual output entropy
    probs = F.softmax(logits.detach(), dim=-1)
    H_actual = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1).mean().item()

    # Penalty: ReLU(H_floor - H_actual)
    penalty = max(0.0, H_floor - H_actual)

    return torch.tensor(penalty * mu, device=logits.device, requires_grad=False)
