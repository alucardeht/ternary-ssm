"""Kendall's Uncertainty-Weighted Multi-Task Loss (R4-A3-F5).

Formula: L = sum_i (1/(2*exp(s_i))) * L_i + sum_i s_i
where s_i = log(sigma_i^2) are learnable log-variance parameters.

Reference: Kendall et al., Multi-Task Learning Using Uncertainty to
Weigh Losses for Scene Understanding, CVPR 2018.
"""
import torch
import torch.nn as nn


class UncertaintyWeightedLoss(nn.Module):
    """Automatically balances multiple loss terms using learned uncertainty.

    Each task gets a learnable log-variance s_i = log(sigma_i^2).
    The loss becomes: L = sum_i (1/(2*exp(s_i))) * L_i + sum_i s_i

    The regularizer sum_i s_i prevents all s_i -> infinity (which would
    make all weights -> 0).
    """

    def __init__(self, n_tasks: int = 4, initial_log_var: float = 0.0):
        super().__init__()
        self.log_vars = nn.Parameter(torch.full((n_tasks,), initial_log_var))
        self.n_tasks = n_tasks

    def forward(self, losses: dict) -> tuple:
        """Combine multiple losses with learned uncertainty weights.

        Args:
            losses: dict mapping task_name -> scalar loss tensor.
                    Only non-None losses are weighted.

        Returns:
            (total_loss, info_dict) where info_dict has per-task weights.
        """
        total = torch.tensor(0.0, device=self.log_vars.device)
        info = {}

        task_idx = 0
        for name, loss_val in losses.items():
            if loss_val is None:
                task_idx += 1
                continue
            if isinstance(loss_val, (int, float)) and loss_val == 0:
                task_idx += 1
                continue

            if task_idx < self.n_tasks:
                s_i = self.log_vars[task_idx]
                precision = 0.5 * torch.exp(-s_i)
                weighted = precision * loss_val + 0.5 * s_i
                total = total + weighted
                info[f'uw_{name}_weight'] = precision.item()
                info[f'uw_{name}_logvar'] = s_i.item()
            else:
                total = total + loss_val
            task_idx += 1

        return total, info
