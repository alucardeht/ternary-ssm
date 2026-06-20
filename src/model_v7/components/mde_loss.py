"""Minimax Depth Equilibrium (R4-A3-P1).

min_W max_{p in Delta_N} sum p_d * L_d(W)
Adversary update: p_d <- p_d * exp(eta * L_d) / Z
"""
import torch
import torch.nn.functional as F


class MinimaxDepthEquilibrium:
    """Adversarial depth weighting for RecursiveSSM."""

    def __init__(self, n_depths: int, eta: float = 0.1, device: torch.device = None):
        self.n_depths = n_depths
        self.eta = eta
        # Adversary weights (uniform initialization)
        self.log_weights = torch.zeros(n_depths, device=device)

    def compute_loss(
        self,
        intermediate_states: list,
        targets: torch.Tensor,
        output_proj: torch.nn.Module,
        sample_every: int = 4,
    ) -> torch.Tensor:
        """Compute minimax loss with adversarial depth weighting.

        Args:
            intermediate_states: per-iteration hidden states
            targets: target token ids
            output_proj: projection to vocab
            sample_every: compute loss every N depths

        Returns:
            Adversary-weighted loss.
        """
        if len(intermediate_states) < 2:
            return torch.tensor(0.0, device=self.log_weights.device)

        per_depth_losses = []
        depth_indices = []

        for i in range(sample_every, len(intermediate_states), sample_every):
            h = intermediate_states[i]
            logits = output_proj(h)
            loss_d = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                targets.view(-1),
                ignore_index=-100,
            )
            per_depth_losses.append(loss_d)
            depth_indices.append(i)

        if not per_depth_losses:
            return torch.tensor(0.0, device=self.log_weights.device)

        losses = torch.stack(per_depth_losses)

        # Adversary weights (softmax of log_weights)
        n = len(depth_indices)
        weights = F.softmax(self.log_weights[:n], dim=0)

        # Weighted loss
        total = (weights * losses).sum()

        # Update adversary (multiplicative weights)
        with torch.no_grad():
            self.log_weights[:n] += self.eta * losses.detach()
            # Normalize to prevent drift
            self.log_weights -= self.log_weights.mean()

        return total
