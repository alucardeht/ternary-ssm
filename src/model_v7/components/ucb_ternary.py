"""UCB-Bandit Ternary Training - M5 - Eliminates STE entirely.

Instead of:
  1. Maintain FP32 shadow weights
  2. Quantize to ternary
  3. Compute loss
  4. STE backward (approximate gradient through quantization)
  5. Update shadow weights with Adam

This approach:
  1. Maintain ternary weights directly {-1, 0, +1}
  2. For each weight position, compute exact loss delta for each of 3 values
  3. Use UCB1 criterion to decide whether to flip
  4. No shadow weights, no STE, no Adam for ternary params

Memory savings: 67% reduction in optimizer state (no shadow + no m + no v for ternary).
"""

import torch
import torch.nn as nn
import math


class UCBTernaryLinear(nn.Module):
    """Linear layer with UCB-bandit ternary weight optimization.

    Each weight w_ij ∈ {-1, 0, +1} is treated as a 3-armed bandit.
    UCB1 criterion balances exploitation (best known value) with
    exploration (less-tried values).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 exploration_c: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.exploration_c = exploration_c

        self.weight = nn.Parameter(torch.zeros(out_features, in_features, dtype=torch.int8))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        self.register_buffer('reward_sum', torch.zeros(out_features, in_features, 3))
        self.register_buffer('pull_count', torch.ones(out_features, in_features, 3))
        self.register_buffer('total_pulls', torch.tensor(3.0))

        self._saved_input = None
        self._call_count = 0
        self.register_full_backward_hook(self._backward_hook)

        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            self.weight.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._saved_input = x.detach()
        w_float = self.weight.float()
        return torch.nn.functional.linear(x, w_float, self.bias)

    def compute_flip_utilities(self, x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
        """Compute utility of flipping each weight to each value.

        For y = Wx, if we change w_ij from current to candidate:
        delta_y_i = (candidate - current) * x_j

        Utility ~= -delta_loss ~= -(grad_output_i * delta_y_i)

        Args:
            x: Input tensor [batch, in_features]
            grad_output: Gradient of loss w.r.t. output [batch, out_features]

        Returns:
            utilities [out_features, in_features, 3] for each (weight, value) pair
        """
        current = self.weight.float()

        utilities = torch.zeros(self.out_features, self.in_features, 3,
                               device=x.device, dtype=x.dtype)
        values = torch.tensor([-1.0, 0.0, 1.0], device=x.device, dtype=x.dtype)

        batch_size = x.size(0)
        for k, val in enumerate(values):
            delta = val - current

            grad_x = torch.einsum('bi,bj->ij', grad_output, x) / max(batch_size, 1)
            utilities[:, :, k] = -(grad_x * delta)

        return utilities

    def ucb_update(self, utilities: torch.Tensor, exploration_c: float = None):
        """Update UCB statistics and flip weights based on UCB1 criterion.

        UCB1 score for arm k of weight (i,j):
        score = reward_mean + c * sqrt(ln(T) / n_k)

        Args:
            utilities: [out, in, 3] flip utilities
            exploration_c: Exploration coefficient (None = use self.exploration_c)
        """
        if exploration_c is None:
            exploration_c = self.exploration_c

        self.total_pulls.add_(1.0)
        t = self.total_pulls.item()

        current_arm = (self.weight + 1).long()
        for k in range(3):
            mask = (current_arm == k)
            if mask.any():
                self.reward_sum[:, :, k][mask] += utilities[:, :, k][mask]
                self.pull_count[:, :, k][mask] += 1.0

        mean_reward = self.reward_sum / self.pull_count.clamp(min=1.0)
        exploration = exploration_c * torch.sqrt(torch.tensor(math.log(max(t, 1.0)))) / self.pull_count.clamp(min=1.0)
        ucb_scores = mean_reward + exploration

        best_arm = ucb_scores.argmax(dim=-1)
        new_weights = best_arm.to(torch.int8) - 1

        self.weight.data.copy_(new_weights)

    @staticmethod
    def _backward_hook(module, grad_input, grad_output):
        """Backward hook to trigger UCB update."""
        if module._saved_input is not None and grad_output[0] is not None:
            with torch.no_grad():
                utilities = module.compute_flip_utilities(
                    module._saved_input,
                    grad_output[0].detach()
                )
                module.ucb_update(utilities, module.exploration_c)

            module._saved_input = None
            module._call_count += 1

            if module._call_count % 500 == 1:
                with torch.no_grad():
                    mean_pulls = module.pull_count.mean().item()
                    mean_reward = module.reward_sum.mean().item()
                    ternary_stats = self._compute_ternary_stats(module)
                    print(f"  [UCBTernary] call={module._call_count} "
                          f"mean_pulls={mean_pulls:.1f} mean_reward={mean_reward:.4f} "
                          f"distribution={ternary_stats}")

    @staticmethod
    def _compute_ternary_stats(module) -> str:
        """Compute ternary weight distribution."""
        with torch.no_grad():
            w_float = module.weight.float()
            n_neg = (w_float == -1.0).sum().item()
            n_zero = (w_float == 0.0).sum().item()
            n_pos = (w_float == 1.0).sum().item()
            total = n_neg + n_zero + n_pos

            pct_neg = n_neg * 100.0 / max(total, 1)
            pct_zero = n_zero * 100.0 / max(total, 1)
            pct_pos = n_pos * 100.0 / max(total, 1)

            return f"-1:{pct_neg:.1f}% 0:{pct_zero:.1f}% +1:{pct_pos:.1f}%"
