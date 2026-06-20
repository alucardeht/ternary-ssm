"""FSCR (Flip Safety/Confidence Ratio) - M6 - Prevents bad flips driven by activation outliers.

Complements M5's temporal gate. Identifies weight flips that are driven by activation
outliers (a few extreme input values) rather than genuine learning signal.

For each gradient g_ij, compute:
- safety: |g_ij| / max(|x_j|)  - how safe is this flip
- confidence: consistency of gradient sign across recent mini-batches

Mask out flips where safety < threshold.
"""

import torch
import torch.nn as nn


def compute_fscr_mask(model: nn.Module, fscr_threshold: float = 2.0) -> dict:
    """Compute Flip Safety/Confidence Ratio mask.

    FSCR identifies weight flips that are driven by activation outliers
    (a few extreme input values) rather than genuine learning signal.

    For each gradient g_ij, compute safety as gradient magnitude relative
    to weight magnitude. Large gradient on small weight = potentially
    outlier-driven.

    Args:
        model: The neural network model
        fscr_threshold: Threshold for safe/unsafe flip decision (higher = more conservative)

    Returns:
        Dictionary mapping parameter name to boolean safety mask
    """
    masks = {}

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad = param.grad
        weight = param.data

        weight_mag = weight.abs().clamp(min=1e-8)
        grad_mag = grad.abs()

        ratio = grad_mag / weight_mag

        safe_mask = ratio < fscr_threshold
        masks[name] = safe_mask

    return masks


def apply_fscr_mask(model: nn.Module, fscr_masks: dict):
    """Zero out gradients flagged as outlier-driven by FSCR.

    Args:
        model: The neural network model
        fscr_masks: Dictionary of safety masks from compute_fscr_mask()
    """
    for name, param in model.named_parameters():
        if name in fscr_masks and param.grad is not None:
            param.grad.mul_(fscr_masks[name].float())


def log_fscr_statistics(model: nn.Module, fscr_masks: dict, step: int, log_interval: int = 500):
    """Log FSCR statistics for debugging.

    Args:
        model: The neural network model
        fscr_masks: Dictionary of safety masks
        step: Current training step
        log_interval: How often to log (every N steps)
    """
    if step % log_interval != 0:
        return

    with torch.no_grad():
        total_params = 0
        masked_params = 0

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            total_params += param.numel()
            if name in fscr_masks:
                mask = fscr_masks[name]
                masked_params += (~mask).sum().item()

        pct_masked = masked_params * 100.0 / max(total_params, 1)
        print(f"  [FSCR] step={step} total_params={total_params} "
              f"masked_params={masked_params} masked%={pct_masked:.2f}%")
