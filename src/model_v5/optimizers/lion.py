"""Lion optimizer — Sign Gradient Descent with Momentum.

Reference: Chen et al. (2023) "Symbolic Discovery of Optimization Algorithms"
arXiv:2302.06675

Formula per step:
    c_t   = beta1 * m_{t-1} + (1 - beta1) * g_t
    p_t   = p_{t-1} - lr * (sign(c_t) + wd * p_{t-1})
    m_t   = beta2 * m_{t-1} + (1 - beta2) * g_t

Memory: 1 state tensor per param (exp_avg only).
No exp_avg_sq. ~33% less optimizer memory than AdamW at the same param count.

LR note: Lion applies sign(update), so step magnitude is always exactly lr.
Use lr ~10x smaller than AdamW (e.g., 6e-5 where AdamW uses 6e-4).
"""

import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros(p.shape, dtype=torch.float16, device=p.device)

                m = state["exp_avg"].float()

                # Decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Compute update direction: sign(beta1*m + (1-beta1)*g)
                update = beta1 * m + (1.0 - beta1) * grad

                # C-Lion: only update where momentum and gradient agree
                mask = (update * grad > 0).float()
                ratio = mask.mean()
                if not hasattr(self, '_clion_logged'):
                    print(f"[C-LION] Cautious mask active | agreement={ratio.item():.1%} of params updated")
                    self._clion_logged = True
                mask = mask / (ratio + 1e-8)
                update = update * mask

                p.add_(update.sign_(), alpha=-lr)

                # Update momentum: m = beta2*m + (1-beta2)*g
                m.mul_(beta2).add_(grad, alpha=1.0 - beta2)

                # Store momentum back as FP16
                state["exp_avg"].copy_(m.half())

                if not hasattr(self, '_qlion_logged'):
                    print(f"[Q-LION] FP16 momentum active | shape={p.shape} | "
                          f"saved={p.numel() * 2 / 1024:.0f}KB vs FP32")
                    self._qlion_logged = True

        return loss
