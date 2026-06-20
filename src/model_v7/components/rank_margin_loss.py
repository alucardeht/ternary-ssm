"""Rank-based margin loss for language modeling.

Integer-compatible alternative to softmax+cross-entropy.
Replace softmax + cross-entropy (requires float exp/log) with rank-margin loss.
With ternary weights and piecewise-linear activations, enables fully integer training.
"""

import torch
import torch.nn as nn


class RankMarginLoss(nn.Module):
    """Rank-based margin loss for language modeling.

    Instead of softmax + cross-entropy (requires float exp/log):
    - Score correct token
    - Score top-K negative tokens
    - Loss = sum max(0, margin + neg_score - pos_score)

    With ternary weights + integer activations, scores are integers.
    Comparison and subtraction are integer operations.

    This enables fully integer training pipeline.
    """

    def __init__(self, margin: float = 1.0, num_negatives: int = 64):
        super().__init__()
        self.margin = margin
        self.num_negatives = num_negatives

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch*seq, vocab_size]
            targets: [batch*seq] - target token indices

        Returns:
            loss: scalar
        """
        batch_size = logits.size(0)
        vocab_size = logits.size(1)

        pos_scores = logits[torch.arange(batch_size, device=logits.device), targets]

        neg_indices = torch.randint(0, vocab_size, (batch_size, self.num_negatives), device=logits.device)
        neg_scores = logits.gather(1, neg_indices)

        margins = self.margin + neg_scores - pos_scores.unsqueeze(1)
        loss = torch.clamp(margins, min=0).mean()

        return loss
