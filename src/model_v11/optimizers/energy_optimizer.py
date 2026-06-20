"""Energy-Based Ternary Optimizer — stateless 0-byte/param decision optimizer.

Uses energy function E(v) = -g·v + lambda|v| to directly select ternary values {-1, 0, +1}
without maintaining optimizer state. Shadow weights updated toward optimal ternary.

Memory: 0 bytes/param (stateless — no accumulators, no momentum, no variance)

Mathematical principle:
    E(v) = -g·v + lambda|v|
    For v ∈ {-1, 0, +1}:
      E(-1) = g + lambda
      E(0) = 0
      E(+1) = -g + lambda
    w* = argmin(E(v))
    p.data -= lr * (p.data - w*)  # Move shadow toward optimal ternary

Simplified variant (paper version):
    If |g| > lambda: w* = sign(g)
    Else: w* = 0
"""

import torch
from torch.optim import Optimizer
import math


class EnergyTernaryOptimizer(Optimizer):
    """Stateless energy-based optimizer for ternary weight training.

    Energy function approach: each gradient determines optimal ternary value
    based on balance between gradient alignment and sparsity regularization.

    Args:
        params: Parameter groups. Each param group dict should have:
            - 'params': list of parameters
            - 'is_ternary': bool, whether these are ternary params (default: True)
            - 'lr': learning rate for this group (default: 1e-3)
            - 'lambda_sparse': sparsity coefficient lambda in energy function (default: 0.01)
        lr: Default learning rate (default: 1e-3)
        lambda_sparse: Default sparsity coefficient (default: 0.01)
        collect_stats: Enable activation tracking for diagnostics (default: True)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        lambda_sparse: float = 0.01,
        hysteresis_factor: float = 0.1,
        collect_stats: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if lambda_sparse < 0.0:
            raise ValueError(f"Invalid lambda_sparse: {lambda_sparse}")
        if hysteresis_factor < 0.0:
            raise ValueError(f"Invalid hysteresis_factor: {hysteresis_factor}")

        defaults = dict(lr=lr, lambda_sparse=lambda_sparse, hysteresis_factor=hysteresis_factor)
        super().__init__(params, defaults)

        self.total_steps = 0
        self.total_flips = 0
        self.total_stable = 0
        self.total_transitions = 0
        self.collect_stats = collect_stats

    @torch.no_grad()
    def step(self, closure=None):
        """Perform energy-based optimization step.

        For each ternary parameter:
        1. Compute energy E(v) for v ∈ {-1, 0, +1}
        2. Select v* = argmin E(v)
        3. Update shadow weight: p -= lr * (p - v*)
        4. Track flips for diagnostics

        Args:
            closure: Optional closure for loss computation (unused)

        Returns:
            Loss value if closure provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.total_steps += 1
        step_count = self.total_steps

        for group in self.param_groups:
            lr = group['lr']
            is_ternary = group.get('is_ternary', True)
            lambda_fallback = group.get('lambda_sparse', 0.01)

            if not is_ternary:
                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                if torch.isinf(p.grad).any() or torch.isnan(p.grad).any():
                    continue

                grad = p.grad.data

                lambda_sparse = torch.quantile(grad.abs().flatten(), 0.3).item()
                lambda_sparse = max(1e-6, min(lambda_sparse, 1.0))

                E_neg = grad + lambda_sparse
                E_zero = torch.zeros_like(grad)
                E_pos = -grad + lambda_sparse

                energies = torch.stack([E_neg, E_zero, E_pos], dim=-1)
                min_indices = energies.argmin(dim=-1)

                optimal_ternary = (min_indices - 1).float()

                current_ternary = torch.sign(p.data)

                energy_current = torch.where(
                    current_ternary == -1, E_neg,
                    torch.where(current_ternary == 0, E_zero, E_pos)
                )

                energy_optimal = energies.min(dim=-1)[0]

                hysteresis_factor = group.get('hysteresis_factor', 0.1)
                hysteresis_threshold = hysteresis_factor * lambda_sparse
                energy_improvement = energy_current - energy_optimal
                should_transition = energy_improvement > hysteresis_threshold

                new_ternary = torch.where(should_transition, optimal_ternary, current_ternary)

                if self.collect_stats:
                    n_transitions = should_transition.sum().item()
                    n_stable = (~should_transition).sum().item()
                    self.total_transitions += n_transitions
                    self.total_stable += n_stable
                    flips = (current_ternary != optimal_ternary).sum().item()
                    self.total_flips += flips

                # Direct ternary assignment — hysteresis prevents oscillation
                p.data.copy_(new_ternary)

        if step_count % 50 == 0:
            stats = self._compute_step_stats()
            neg_pct = stats['neg_pct']
            zero_pct = stats['zero_pct']
            pos_pct = stats['pos_pct']
            energy_mean = stats['energy_mean']
            lambda_mean = stats['lambda_mean']

            total_weights = self.total_stable + self.total_transitions
            stable_rate = 100.0 * self.total_stable / max(total_weights, 1)
            transition_rate = 100.0 * self.total_transitions / max(total_weights, 1)

            print(f"[ENERGY] step={step_count}, lambda_mean={lambda_mean:.4f}, "
                  f"dist=({neg_pct:.1f}/{zero_pct:.1f}/{pos_pct:.1f}), "
                  f"energy_mean={energy_mean:.4f}, "
                  f"stable={stable_rate:.1f}%, transitions={transition_rate:.1f}%")

        return loss

    @torch.no_grad()
    def _compute_step_stats(self):
        """Compute activation statistics for monitoring."""
        neg_count = 0
        zero_count = 0
        pos_count = 0
        total_params = 0
        total_energy = 0.0
        total_lambda = 0.0
        n_params = 0

        for group in self.param_groups:
            is_ternary = group.get('is_ternary', True)
            if not is_ternary:
                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                total_params += grad.numel()

                lambda_sparse = torch.quantile(grad.abs().flatten(), 0.3).item()
                lambda_sparse = max(1e-6, min(lambda_sparse, 1.0))
                total_lambda += lambda_sparse
                n_params += 1

                neg_count += (p.data == -1).sum().item()
                zero_count += (p.data == 0).sum().item()
                pos_count += (p.data == 1).sum().item()

                E_neg = grad + lambda_sparse
                E_zero = torch.zeros_like(grad)
                E_pos = -grad + lambda_sparse

                energies = torch.stack([E_neg, E_zero, E_pos], dim=-1)
                energy_min = energies.min(dim=-1)[0]
                total_energy += energy_min.sum().item()

        neg_pct = 100.0 * neg_count / max(total_params, 1)
        zero_pct = 100.0 * zero_count / max(total_params, 1)
        pos_pct = 100.0 * pos_count / max(total_params, 1)
        energy_mean = total_energy / max(total_params, 1)
        lambda_mean = total_lambda / max(n_params, 1)

        return {
            'neg_pct': neg_pct,
            'zero_pct': zero_pct,
            'pos_pct': pos_pct,
            'energy_mean': energy_mean,
            'lambda_mean': lambda_mean,
        }

    @torch.no_grad()
    def get_stats(self):
        """Return optimizer statistics for monitoring.

        Returns:
            dict with keys:
            - 'total_steps': Total optimizer steps executed
            - 'total_flips': Total ternary state transitions that would have occurred
            - 'flips_per_step': Average potential flips per step
            - 'total_stable': Total weights kept in current state due to hysteresis
            - 'total_transitions': Total weights actually transitioned (honored hysteresis)
            - 'stable_rate': Percentage of decisions kept due to hysteresis
            - 'transition_rate': Percentage of decisions that transitioned
        """
        flips_per_step = self.total_flips / max(1, self.total_steps)
        total_weights = self.total_stable + self.total_transitions
        stable_rate = 100.0 * self.total_stable / max(total_weights, 1)
        transition_rate = 100.0 * self.total_transitions / max(total_weights, 1)

        return {
            'total_steps': self.total_steps,
            'total_flips': self.total_flips,
            'flips_per_step': flips_per_step,
            'total_stable': self.total_stable,
            'total_transitions': self.total_transitions,
            'stable_rate': stable_rate,
            'transition_rate': transition_rate,
        }
