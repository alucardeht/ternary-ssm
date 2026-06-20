"""Bayesian Flip Accumulator (R4-A3-M8).

Replaces STE shadow weights with Bayesian posterior over {-1,0,+1}.
Flips weight when posterior confidence exceeds threshold.

Evidence: e_v = sum_l -g_l * (v - q_current)
Update: log pi_v^{t+1} = rho * log pi_v^t + e_v/tau - log Z
Flip: when max(pi) > theta AND argmax != current
"""
import torch


class BayesianFlipAccumulator:
    """Per-weight Bayesian posterior tracker for ternary values."""

    def __init__(
        self,
        param_shape: tuple,
        device: torch.device,
        rho: float = 0.95,
        tau: float = 1.0,
        theta: float = 0.9,
    ):
        self.rho = rho
        self.tau = tau
        self.theta = theta
        # Log-posterior for {-1, 0, +1}: shape (3, *param_shape)
        self.log_pi = torch.zeros(3, *param_shape, device=device)
        # Values: -1, 0, +1
        self.values = torch.tensor([-1.0, 0.0, 1.0], device=device)

    def update(self, grad: torch.Tensor, current_ternary: torch.Tensor):
        """Update posterior based on gradient evidence.

        Args:
            grad: gradient for this parameter (same shape as param)
            current_ternary: current ternary weight values {-1,0,+1}
        """
        # Evidence for each value: e_v = -grad * (v - current)
        for v_idx, v in enumerate(self.values):
            e_v = -grad * (v - current_ternary)
            self.log_pi[v_idx] = self.rho * self.log_pi[v_idx] + e_v / self.tau

        # Normalize (log-space)
        log_Z = torch.logsumexp(self.log_pi, dim=0)
        self.log_pi = self.log_pi - log_Z.unsqueeze(0)

    def get_flips(self, current_ternary: torch.Tensor) -> tuple:
        """Identify weights that should flip based on posterior confidence.

        Returns:
            (flip_mask, new_values) where flip_mask is bool, new_values is {-1,0,+1}
        """
        pi = torch.softmax(self.log_pi, dim=0)  # (3, *shape)
        max_pi, best_idx = pi.max(dim=0)  # (*shape)

        # Map index to value
        new_values = self.values[best_idx]

        # Flip only where: confidence > theta AND best != current
        flip_mask = (max_pi > self.theta) & (new_values != current_ternary)

        return flip_mask, new_values

    def apply_flips(self, param: torch.nn.Parameter, threshold_coeff: float = 0.5):
        """Apply accumulated flip decisions to parameter.

        Args:
            param: the FP32 shadow weight parameter
            threshold_coeff: ternary quantization threshold coefficient
        """
        with torch.no_grad():
            w = param.data
            threshold = threshold_coeff * w.abs().mean()
            current_ternary = torch.sign(w) * (w.abs() > threshold).float()

            flip_mask, new_values = self.get_flips(current_ternary)
            n_flips = flip_mask.sum().item()

            if n_flips > 0:
                # Push shadow weights toward the target ternary value
                # For value +1: set shadow to +threshold * 1.1
                # For value -1: set shadow to -threshold * 1.1
                # For value 0: set shadow to 0
                target_shadow = new_values * threshold * 1.1
                param.data[flip_mask] = target_shadow[flip_mask]

                # Reset posterior for flipped weights
                self.log_pi[:, flip_mask] = 0.0

            return n_flips
