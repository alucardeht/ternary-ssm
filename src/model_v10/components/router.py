import torch
import torch.nn as nn


class MoDRouter(nn.Module):
    """Mixture of Depths (MoD) Router - Top-k Expert-Choice.

    Based on Raposo et al. 2024. Each layer selects top-k tokens by router score.
    Deterministic selection, no Gumbel noise. Gradients flow through router weights.
    """

    def __init__(self, d_model: int, capacity: float = 0.5):
        super().__init__()
        self.capacity = capacity
        self.routing_enabled = True
        self.proj = nn.Linear(d_model, 1, bias=True)

        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.constant_(self.proj.bias, 0.0)

        self.register_buffer('_last_process_ratio', torch.tensor(0.0))
        self.register_buffer('_last_mean_weight', torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> tuple:
        """Returns (mask, weights) for top-k token selection.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            mask: (batch, seq_len, 1) binary mask, 1.0 for selected tokens
            weights: (batch, seq_len, 1) sigmoid router weights (for gradient flow)
        """
        batch, seq_len, _ = x.shape

        if not self.routing_enabled:
            ones = torch.ones(batch, seq_len, device=x.device, dtype=x.dtype)
            with torch.no_grad():
                self._last_process_ratio = torch.tensor(1.0, device=x.device)
                self._last_mean_weight = torch.tensor(1.0, device=x.device)
            return ones.unsqueeze(-1), ones.unsqueeze(-1)

        k = max(1, int(self.capacity * seq_len))

        logits = self.proj(x).squeeze(-1)
        weights = torch.sigmoid(logits)

        _, top_indices = torch.topk(weights, k, dim=1)

        mask = torch.zeros_like(weights)
        mask.scatter_(1, top_indices, 1.0)

        with torch.no_grad():
            self._last_process_ratio = mask.mean()
            self._last_mean_weight = weights.mean()

        return mask.unsqueeze(-1), weights.unsqueeze(-1)

    def get_stats(self) -> dict:
        """Return cached routing stats (no input needed)."""
        return {
            'process_ratio': self._last_process_ratio.item(),
            'mean_weight': self._last_mean_weight.item(),
        }
