from typing import Dict, Any
import torch
from torch.optim import Optimizer


class TSignSGD(Optimizer):
    """Ternary Sign SGD with gradient magnitude filtering.

    Only the top K% of gradient magnitudes trigger weight updates. This creates
    sparse updates that align well with ternary weight patterns and STE-based
    gradient accumulation.

    Key properties:
    - Minimal memory: only threshold per parameter (no momentum or v like Adam)
    - Sparse updates: ~K% of weights change per step
    - Works well with STE: sign updates accumulate in shadow weights
    - Useful for knowledge injection (ATM weights, alpha updates)

    Args:
        params: Iterable of parameters or dicts defining parameter groups
        lr: Learning rate (default: 1e-3)
        filter_ratio: Fraction of gradients to keep (default: 0.95, keep top 5%)
        threshold_update_interval: Update threshold magnitude every N steps (default: 10)
        weight_decay: L2 weight decay (default: 0.0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        filter_ratio: float = 0.95,
        threshold_update_interval: int = 10,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= filter_ratio <= 1.0:
            raise ValueError(f"Invalid filter_ratio: {filter_ratio}")
        if threshold_update_interval < 1:
            raise ValueError(f"Invalid threshold_update_interval: {threshold_update_interval}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            filter_ratio=filter_ratio,
            threshold_update_interval=threshold_update_interval,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        """Perform a single optimizer step.

        Updates parameters using sign of filtered gradients. Filters based on
        magnitude, keeping only top K% of gradients.

        Args:
            closure: Optional closure that reevaluates the model and returns loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            filter_ratio = group['filter_ratio']
            interval = group['threshold_update_interval']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['threshold'] = torch.tensor(0.0, device=p.device, dtype=torch.float32)

                state['step'] += 1

                if state['step'] % interval == 1 or interval == 1:
                    abs_grad = grad.abs()
                    state['threshold'] = torch.quantile(
                        abs_grad.float().flatten(), filter_ratio
                    )

                mask = (grad.abs() >= state['threshold']).float()

                if wd > 0:
                    p.data.mul_(1 - lr * wd)

                p.data.add_(mask * grad.sign(), alpha=-lr)

        return loss
