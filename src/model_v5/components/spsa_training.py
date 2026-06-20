"""SPSA (Simultaneous Perturbation Stochastic Approximation) training for ternary weights.

Zero backward pass needed - random perturbations estimate all flip utilities from forward passes.
Trades 2x forward passes for zero backward pass, enabling memory-efficient ternary training.
"""

import torch
import torch.nn as nn


def spsa_ternary_update(model, loss_fn, x, targets, device, perturbation_scale: float = 0.1, lr: float = 0.01):
    """SPSA update for ternary weights - no backward pass needed.

    Algorithm:
    1. Sample random perturbation delta ∈ {-1, +1} for each weight
    2. Forward pass with W + c*delta -> loss_plus
    3. Forward pass with W - c*delta -> loss_minus
    4. Gradient estimate: g = (loss_plus - loss_minus) / (2 * c * delta)
    5. Update: W = quantize(W - lr * g)

    Trades 2x forward passes for zero backward pass.
    Memory: O(1) - no activation storage needed.

    Args:
        model: The model
        loss_fn: Loss function
        x: Input tensor
        targets: Target tensor
        device: Device
        perturbation_scale: c - perturbation magnitude
        lr: Learning rate for SPSA updates

    Returns:
        avg_loss: Average loss of the two forward passes
    """
    original_weights = {}
    for name, param in model.named_parameters():
        original_weights[name] = param.data.clone()

    deltas = {}
    for name, param in model.named_parameters():
        deltas[name] = torch.randint(0, 2, param.shape, device=device).float() * 2 - 1

    for name, param in model.named_parameters():
        param.data.add_(deltas[name], alpha=perturbation_scale)

    with torch.no_grad():
        logits_plus = model(x)
        if isinstance(logits_plus, tuple):
            logits_plus = logits_plus[0]
        loss_plus = loss_fn(logits_plus.reshape(-1, logits_plus.size(-1)), targets.reshape(-1))

    for name, param in model.named_parameters():
        param.data.add_(deltas[name], alpha=-2 * perturbation_scale)

    with torch.no_grad():
        logits_minus = model(x)
        if isinstance(logits_minus, tuple):
            logits_minus = logits_minus[0]
        loss_minus = loss_fn(logits_minus.reshape(-1, logits_minus.size(-1)), targets.reshape(-1))

    for name, param in model.named_parameters():
        param.data.copy_(original_weights[name])

    grad_estimate_scale = (loss_plus.item() - loss_minus.item()) / (2 * perturbation_scale)

    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad.zero_()
        gradient = grad_estimate_scale / (deltas[name] + 1e-8)
        param.grad = gradient

    return (loss_plus.item() + loss_minus.item()) / 2
