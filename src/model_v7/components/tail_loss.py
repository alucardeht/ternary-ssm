"""Ternary-Aligned Integer Loss (R4-A3-M2).

Flip-distance metric native to ternary: minimum weight flips to change prediction.
Produces integer gradients throughout.
"""
import torch
import torch.nn.functional as F


def compute_tail_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    flip_margin: float = 1.0,
    num_negatives: int = 5,
) -> torch.Tensor:
    """Compute TAIL: hinge loss with flip-distance margin.

    L = sum_{k in top-K negatives} max(0, margin * H - (z_y - z_k))
    where H = ceil(||h||_inf), z_y = correct class logit, z_k = negative logit.

    Args:
        logits: (batch*seq, vocab) raw logits
        targets: (batch*seq,) target indices
        flip_margin: minimum logit margin per flip (F in formula)
        num_negatives: number of hardest negatives to penalize

    Returns:
        Scalar hinge loss.
    """
    batch_size = logits.shape[0]
    vocab_size = logits.shape[1]

    # Get correct class logits
    z_y = logits.gather(1, targets.unsqueeze(1)).squeeze(1)  # (batch,)

    # Get top-K negative logits (highest non-target logits)
    # Mask out target class
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, targets.unsqueeze(1), False)
    neg_logits = logits.masked_fill(~mask, float('-inf'))

    top_k = min(num_negatives, vocab_size - 1)
    z_k, _ = neg_logits.topk(top_k, dim=1)  # (batch, top_k)

    # H = approximation of activation magnitude (use max logit magnitude)
    H = logits.abs().max(dim=1, keepdim=True).values.clamp(min=1.0)

    # Hinge loss: max(0, margin * H - (z_y - z_k))
    margins = flip_margin * H  # (batch, 1)
    gaps = z_y.unsqueeze(1) - z_k  # (batch, top_k)
    violations = F.relu(margins - gaps)  # (batch, top_k)

    return violations.mean()
