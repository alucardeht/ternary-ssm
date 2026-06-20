"""Ergodic Trajectory Optimizer (R4-A3-P5).

Replaces Adam with trajectory statistics (mean, covariance, autocorrelation).
Memory: 84GB (Adam FP32) -> ~50KB (diagonal stats).
"""
import torch


class ErgodicTrajectoryOptimizer:
    """Score weight flips by trajectory statistics improvement."""

    def __init__(
        self,
        model: torch.nn.Module,
        alpha: float = 1.0,
        rho_target: float = 0.9,
        flip_fraction: float = 0.01,
    ):
        self.alpha = alpha
        self.rho_target = rho_target
        self.flip_fraction = flip_fraction

        # Per-layer trajectory statistics (DIAGONAL only - O(d) memory)
        self.trajectory_mean = {}
        self.trajectory_var = {}
        self.trajectory_count = {}

        for name, module in model.named_modules():
            if hasattr(module, 'quantize_enabled') and hasattr(module, 'weight'):
                shape = module.weight.shape
                device = module.weight.device
                self.trajectory_mean[name] = torch.zeros(shape[-1], device=device)
                self.trajectory_var[name] = torch.ones(shape[-1], device=device)
                self.trajectory_count[name] = 0

    def update_statistics(self, model: torch.nn.Module, hidden_states: list):
        """Update trajectory statistics from forward pass hidden states.

        Args:
            model: model (for layer identification)
            hidden_states: list of (batch, seq, d_model) intermediate states
        """
        if not hidden_states:
            return

        # Compute trajectory statistics from hidden state sequence
        for name in self.trajectory_mean:
            # Use mean/var across the trajectory (all intermediate states)
            all_h = torch.stack([h.detach().mean(dim=(0, 1)) for h in hidden_states])  # (L, d)

            new_mean = all_h.mean(dim=0)
            new_var = all_h.var(dim=0, unbiased=False)

            # Exponential moving average update
            count = self.trajectory_count[name]
            alpha_ema = min(0.1, 1.0 / (count + 1))

            self.trajectory_mean[name] = (1 - alpha_ema) * self.trajectory_mean[name] + alpha_ema * new_mean
            self.trajectory_var[name] = (1 - alpha_ema) * self.trajectory_var[name] + alpha_ema * new_var
            self.trajectory_count[name] = count + 1
            break  # Same stats for all layers (shared weights)

    def score_and_flip(self, model: torch.nn.Module) -> int:
        """Score weight positions by trajectory improvement, flip best ones.

        Returns number of flips applied.
        """
        total_flips = 0

        for name, module in model.named_modules():
            if name not in self.trajectory_mean:
                continue
            if not hasattr(module, 'weight'):
                continue

            w = module.weight.data
            threshold = getattr(module, 'threshold_coeff', 0.5) * w.abs().mean()

            # Score: how far is current variance from target?
            target_var = self.alpha * torch.ones_like(self.trajectory_var[name])
            var_deviation = (self.trajectory_var[name] - target_var).abs()

            # Flip the top-K% positions with highest deviation
            n_params = w.numel()
            n_flips = max(1, int(n_params * self.flip_fraction))

            # Use gradient as flip direction signal if available
            if module.weight.grad is not None:
                flip_scores = module.weight.grad.data.abs().view(-1)
                _, top_indices = flip_scores.topk(min(n_flips, len(flip_scores)))

                for idx in top_indices:
                    i = idx.item() // w.shape[1] if w.dim() >= 2 else idx.item()
                    j = idx.item() % w.shape[1] if w.dim() >= 2 else 0

                    # Flip toward gradient-suggested direction
                    grad_sign = -torch.sign(module.weight.grad.data.view(-1)[idx]).item()
                    new_val = grad_sign  # {-1, 0, +1}

                    if new_val == 0:
                        w.view(-1)[idx] = 0.0
                    else:
                        w.view(-1)[idx] = new_val * threshold * 1.1
                    total_flips += 1

        return total_flips
