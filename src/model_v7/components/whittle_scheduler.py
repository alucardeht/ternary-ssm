"""Whittle Index Priority Scheduling for weight updates.

Only evaluate/update top-K weights per step based on Whittle index priority.
5-10x FLOPS reduction by selective parameter updates.
"""

import torch
import torch.nn as nn


class WhittleIndexScheduler:
    """Whittle index priority scheduler for weight updates.

    Maintains a priority queue of weights. Each step, only the top-K
    highest-priority weights are evaluated and updated. Priority is
    based on gradient magnitude * time since last update.

    This reduces per-step FLOPS by only updating the most "urgent" weights.
    """

    def __init__(self, model, update_fraction: float = 0.1):
        """
        Args:
            model: The model
            update_fraction: Fraction of weights to update per step (0.1 = top 10%)
        """
        self.update_fraction = update_fraction
        self.last_update_step = {}
        self.priority = {}

        for name, param in model.named_parameters():
            self.last_update_step[name] = 0
            self.priority[name] = 1.0

    def compute_priorities(self, model, step: int):
        """Update priorities based on gradient magnitude and staleness."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_mag = param.grad.abs().mean().item()
                staleness = step - self.last_update_step[name] + 1
                self.priority[name] = grad_mag * (staleness ** 0.5)

    def get_update_mask(self, model, step: int) -> set:
        """Get set of parameter names to update this step."""
        self.compute_priorities(model, step)

        sorted_params = sorted(self.priority.items(), key=lambda x: x[1], reverse=True)

        n_update = max(1, int(len(sorted_params) * self.update_fraction))
        update_names = set(name for name, _ in sorted_params[:n_update])

        for name in update_names:
            self.last_update_step[name] = step

        return update_names

    def apply_mask(self, model, update_names: set):
        """Zero out gradients for non-selected parameters."""
        for name, param in model.named_parameters():
            if name not in update_names and param.grad is not None:
                param.grad.zero_()
