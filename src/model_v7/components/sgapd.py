"""Spin-Glass Annealing with Phase Detection (R4-A3-P4).

Ternary network as Blume-Capel spin-1 system.
Periodic MC sweeps alongside gradient descent.
Acceptance: min(1, exp(-delta_loss / temperature))
"""
import torch
import math


class SpinGlassAnnealer:
    """Monte Carlo annealer for ternary weights."""

    def __init__(
        self,
        initial_temp: float = 1.0,
        min_temp: float = 0.01,
        decay: float = 0.995,
        sweeps_per_step: int = 10,
    ):
        self.temperature = initial_temp
        self.min_temp = min_temp
        self.decay = decay
        self.sweeps_per_step = sweeps_per_step
        # Order parameters for phase detection
        self.magnetization_history = []
        self.ea_order_history = []

    def mc_sweep(
        self,
        param: torch.nn.Parameter,
        loss_fn,
        input_ids: torch.Tensor,
        model: torch.nn.Module,
    ) -> int:
        """One MC sweep: propose flips, accept/reject via Metropolis.

        Simplified: use random perturbation + loss comparison.

        Args:
            param: weight parameter to sweep
            loss_fn: callable returning scalar loss
            input_ids: input for loss computation
            model: full model

        Returns:
            Number of accepted flips.
        """
        accepted = 0
        w = param.data
        threshold = 0.5 * w.abs().mean()

        for _ in range(self.sweeps_per_step):
            # Random position
            i = torch.randint(0, w.shape[0], (1,)).item() if w.dim() >= 1 else 0
            j = torch.randint(0, w.shape[1], (1,)).item() if w.dim() >= 2 else 0

            old_val = w[i, j].item()

            # Propose random ternary value
            candidates = [-1.0, 0.0, 1.0]
            new_val = candidates[torch.randint(0, 3, (1,)).item()]
            if new_val == old_val:
                continue

            # Apply flip to shadow weight
            if new_val == 0:
                new_shadow = 0.0
            else:
                new_shadow = new_val * threshold * 1.1

            old_shadow = w[i, j].item()
            w[i, j] = new_shadow

            # We can't easily compute loss delta without a forward pass,
            # so use gradient as proxy: delta_loss ~= grad * (new - old)
            if param.grad is not None:
                delta_loss = param.grad[i, j].item() * (new_shadow - old_shadow)
            else:
                delta_loss = 0.0

            # Metropolis acceptance
            if delta_loss < 0:
                accepted += 1  # Always accept improvements
            elif self.temperature > 1e-8:
                accept_prob = math.exp(-delta_loss / self.temperature)
                if torch.rand(1).item() < accept_prob:
                    accepted += 1
                else:
                    w[i, j] = old_shadow  # Reject
            else:
                w[i, j] = old_shadow  # Reject at T=0

        return accepted

    def update_temperature(self):
        """Decay temperature."""
        self.temperature = max(self.min_temp, self.temperature * self.decay)

    def compute_order_params(self, model: torch.nn.Module) -> dict:
        """Compute order parameters for phase detection.

        Returns:
            dict with magnetization m and Edwards-Anderson order q.
        """
        total_m = 0.0
        total_q = 0.0
        n = 0

        for module in model.modules():
            if hasattr(module, 'quantize_enabled') and hasattr(module, 'weight'):
                w = module.weight.data
                threshold = getattr(module, 'threshold_coeff', 0.5) * w.abs().mean()
                w_ternary = torch.sign(w) * (w.abs() > threshold).float()

                total_m += w_ternary.mean().item()
                total_q += (w_ternary ** 2).mean().item()  # = 1 - fraction_zeros
                n += 1

        m = total_m / max(n, 1)
        q = total_q / max(n, 1)

        self.magnetization_history.append(m)
        self.ea_order_history.append(q)

        return {'magnetization': m, 'edwards_anderson': q, 'temperature': self.temperature}
