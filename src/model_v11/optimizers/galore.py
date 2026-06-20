import torch
from torch.optim import Optimizer
import math


class GaLoreProjector:
    """SVD-based gradient projector for gradient low-rank optimization.

    Projects gradients to/from a low-rank subspace determined by SVD.
    Supports both left and right projections depending on gradient shape.

    Args:
        rank: Target rank for low-rank projection
        update_proj_gap: Recompute SVD projection every N steps (default: 200)
        scale: Scaling factor for projected gradients (default: 1.0)
    """

    def __init__(self, rank, update_proj_gap=200, scale=1.0):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self._project_right = None

    def project(self, full_grad, step):
        """Project gradient to low-rank subspace.

        Args:
            full_grad: Full gradient tensor (2D)
            step: Current optimizer step for SVD update scheduling

        Returns:
            Low-rank projected gradient
        """
        if self.ortho_matrix is None or step % self.update_proj_gap == 0:
            self._update_projection(full_grad)

        if self._project_right:
            return full_grad @ self.ortho_matrix.T
        else:
            return self.ortho_matrix.T @ full_grad

    def project_back(self, low_grad):
        """Project gradient back to full-rank space.

        Args:
            low_grad: Low-rank gradient tensor

        Returns:
            Full-rank gradient with scaling applied
        """
        if self._project_right:
            return (low_grad @ self.ortho_matrix) * self.scale
        else:
            return (self.ortho_matrix @ low_grad) * self.scale

    def _update_projection(self, full_grad):
        """Recompute SVD-based projection matrix.

        For tall matrices (m >= n): right projection using V^T (right singular vectors).
        For wide matrices (m < n): left projection using U (left singular vectors).

        Args:
            full_grad: Full gradient tensor for SVD computation
        """
        m, n = full_grad.shape
        effective_rank = min(self.rank, min(m, n))

        grad_f32 = full_grad.float()

        if m >= n:
            _, _, Vh = torch.svd(grad_f32)
            self.ortho_matrix = Vh[:effective_rank, :].to(full_grad.dtype)
            self._project_right = True
        else:
            U, _, _ = torch.svd(grad_f32)
            self.ortho_matrix = U[:, :effective_rank].to(full_grad.dtype)
            self._project_right = False


class GaLoreAdamW(Optimizer):
    """GaLore (Gradient Low-Rank Projection) optimizer with AdamW.

    Reduces optimizer memory by projecting gradients to low-rank subspace.
    Standard AdamW update applied in low-rank space, then projected back to full rank.

    For parameters without rank specified or non-2D gradients, falls back to standard AdamW.

    Args:
        params: Iterable of parameters or parameter groups
        lr: Learning rate (default: 1e-3)
        betas: Adam momentum coefficients (default: (0.9, 0.999))
        eps: Numerical stability constant (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._first_step_logged = False

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimizer step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            wd = group['weight_decay']
            rank = group.get('rank')
            update_proj_gap = group.get('update_proj_gap', 200)
            galore_scale = group.get('galore_scale', 1.0)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('GaLoreAdamW does not support sparse gradients')

                state = self.state[p]

                if rank is None or grad.ndim != 2:
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = None
                        state['exp_avg_sq'] = None

                    state['step'] += 1
                    step = state['step']

                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    if state['exp_avg'] is None:
                        exp_avg = torch.zeros_like(p, dtype=torch.float32)
                        exp_avg_sq = torch.zeros_like(p, dtype=torch.float32)
                    else:
                        exp_avg = state['exp_avg']
                        exp_avg_sq = state['exp_avg_sq']

                    exp_avg.mul_(beta1).add_(grad.float(), alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad.float(), grad.float(), value=1 - beta2)

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    step_size = lr / bias_correction1

                    if wd != 0:
                        p.data.add_(p.data.float(), alpha=-lr * wd)

                    p.data.add_(exp_avg / denom, alpha=-step_size)

                    state['exp_avg'] = exp_avg
                    state['exp_avg_sq'] = exp_avg_sq

                else:
                    if len(state) == 0:
                        state['step'] = 0
                        state['projector'] = GaLoreProjector(rank, update_proj_gap, galore_scale)
                        state['exp_avg_low'] = None
                        state['exp_avg_sq_low'] = None

                    state['step'] += 1
                    step = state['step']

                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    projector = state['projector']
                    grad_low = projector.project(grad.float(), step)

                    if state['exp_avg_low'] is None:
                        exp_avg_low = torch.zeros_like(grad_low)
                        exp_avg_sq_low = torch.zeros_like(grad_low)
                    else:
                        exp_avg_low = state['exp_avg_low']
                        exp_avg_sq_low = state['exp_avg_sq_low']

                    exp_avg_low.mul_(beta1).add_(grad_low, alpha=1 - beta1)
                    exp_avg_sq_low.mul_(beta2).addcmul_(grad_low, grad_low, value=1 - beta2)

                    denom = (exp_avg_sq_low.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    step_size = lr / bias_correction1

                    update_full = projector.project_back(exp_avg_low / denom)

                    if wd != 0:
                        p.data.add_(p.data.float(), alpha=-lr * wd)

                    p.data.add_(update_full.to(p.dtype), alpha=-step_size)

                    state['exp_avg_low'] = exp_avg_low
                    state['exp_avg_sq_low'] = exp_avg_sq_low

        if not self._first_step_logged:
            galore_params = []
            for group in self.param_groups:
                rank = group.get('rank')
                if rank is not None:
                    galore_params.extend([
                        (p.shape, rank) for p in group['params']
                        if p.grad is not None and p.grad.ndim == 2
                    ])
            if galore_params:
                print(f"GaLoreAdamW: {len(galore_params)} parameters with low-rank projection")
                for shape, rank in galore_params[:3]:
                    print(f"  Shape {shape} -> rank {rank}")
            self._first_step_logged = True

        return loss
