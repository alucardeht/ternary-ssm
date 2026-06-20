"""
Low-Rank Momentum AdamW optimizer (F20).

Factorizes momentum state m = U @ V^T (rank-r) instead of storing full dense m.
Reduces optimizer memory from 8 bytes/param to ~2-4 bytes/param for large layers.

Based on: MoFaSGD (arXiv 2025) — "Low-Rank Momentum Factorization for Large-Scale Training"
"""

import torch
from torch.optim import Optimizer
import math


class LowRankMomentumAdamW(Optimizer):
    """
    AdamW optimizer with low-rank momentum factorization.

    Stores m = U @ V^T (rank-r) instead of full m matrix.
    For square layers (d×d), reduces memory from d² to 2dr.

    Args:
        params: Model parameters
        lr: Learning rate
        betas: (beta1, beta2) for Adam
        eps: Epsilon for numerical stability
        weight_decay: Weight decay coefficient
        rank: Low-rank factorization dimension (default 64)
        svd_interval: Recompute SVD every N steps (default 500)
        min_dim: Only factorize params with min(shape) >= min_dim (default 128)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, rank=64, svd_interval=500, min_dim=128):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        rank=rank, svd_interval=svd_interval, min_dim=min_dim)
        super().__init__(params, defaults)

        self.step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            rank = group['rank']
            svd_interval = group['svd_interval']
            min_dim = group['min_dim']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0

                    state['v'] = torch.zeros_like(p)

                    param_shape = p.shape

                    if len(param_shape) == 2 and min(param_shape) >= min_dim:
                        d_in, d_out = param_shape
                        r = min(rank, min(d_in, d_out))

                        state['use_low_rank'] = True
                        state['U'] = torch.zeros(d_in, r, dtype=p.dtype, device=p.device)
                        state['V'] = torch.zeros(d_out, r, dtype=p.dtype, device=p.device)
                        state['rank'] = r

                        print(f"[LowRankMomentum] Factorizing {param_shape} with rank={r} "
                              f"(memory: {d_in*d_out*4/(1024**2):.2f}MB -> "
                              f"{(d_in+d_out)*r*4/(1024**2):.2f}MB, "
                              f"{(d_in+d_out)*r/(d_in*d_out)*100:.1f}% of full)")
                    else:
                        state['use_low_rank'] = False
                        state['m'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']

                v = state['v']
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if state['use_low_rank']:
                    U = state['U']
                    V = state['V']
                    r = state['rank']

                    m_full = U @ V.T

                    m_full.mul_(beta1).add_(grad, alpha=1 - beta1)

                    if step % svd_interval == 0 or step == 1:
                        try:
                            U_svd, S, Vt_svd = torch.linalg.svd(m_full, full_matrices=False)

                            U[:, :] = U_svd[:, :r] @ torch.diag(S[:r].sqrt())
                            V[:, :] = Vt_svd[:r, :].T @ torch.diag(S[:r].sqrt())

                            if step % (svd_interval * 10) == 0:
                                m_reconstructed = U @ V.T
                                error = (m_full - m_reconstructed).norm() / (m_full.norm() + 1e-8)
                                print(f"[LowRankMomentum SVD] step={step}, "
                                      f"reconstruction_error={error.item():.6f} (should be < 0.02)")
                        except RuntimeError as e:
                            print(f"[LowRankMomentum] SVD failed at step {step}: {e}. "
                                  f"Falling back to full-rank for this parameter.")
                            state['use_low_rank'] = False
                            state['m'] = m_full
                            del state['U']
                            del state['V']
                            m = state['m']
                    else:
                        pass

                    m = m_full
                else:
                    m = state['m']
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1

                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                p.addcdiv_(m, denom, value=-step_size)

        return loss


def create_low_rank_momentum_optimizer(model, config, base_lr):
    """
    Factory function to create LowRankMomentumAdamW optimizer.

    Args:
        model: ModelV2 instance
        config: ModelConfig with low_rank_momentum_* flags
        base_lr: Base learning rate

    Returns:
        LowRankMomentumAdamW optimizer instance
    """
    return LowRankMomentumAdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        rank=config.low_rank_momentum_rank,
        svd_interval=config.low_rank_svd_interval,
        min_dim=config.low_rank_min_dim,
    )
