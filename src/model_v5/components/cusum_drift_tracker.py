"""
CUSUM Drift Detection & SNR-Optimal Update Frequency (R4-A6-M5, R4-A6-M6)

Implements two complementary statistical methods for tracking committed weights:
- M5 (CUSUM): Replace Adam optimizer with low-cost cumulative sum test for drift detection
- M6 (SNR Skip): Dynamically compute optimal update skip frequency based on signal-to-noise ratio

CUSUM uses ~2 FLOPs/step vs Adam's ~20 FLOPs/step for deeply committed weights.
SNR skip reduces update frequency for semi-committed weights: K_skip = ceil(1/SNR^2).
"""

import math
from typing import Dict, List, Tuple, Optional
import torch


class CUSUMDriftTracker:
    """
    Detects when gradient distributions shift (drift) via CUSUM (Cumulative Sum Control Chart).

    For committed weights, replaces Adam with cheap drift detection. Once P_flip rises above
    threshold (weight might flip back to active state), transitions back to full Adam.

    Algorithm: S_k = max(0, S_{k-1} + (x_k - mu) - delta)
    - x_k = gradient norm of block k
    - mu = historical mean (running average)
    - delta = sensitivity parameter (0.5 typical)
    - Alarm when S_k > threshold
    """

    def __init__(
        self,
        cusum_threshold: float = 5.0,
        cusum_sensitivity: float = 0.5,
        drift_detection_alpha: float = 0.01,
    ):
        """
        Args:
            cusum_threshold: CUSUM alarm threshold (sigma units above baseline)
            cusum_sensitivity: Sensitivity δ in CUSUM formula (0.5 typical for balanced detection)
            drift_detection_alpha: EMA decay for running mean/variance (0.01 = slow update)
        """
        self.threshold = cusum_threshold
        self.sensitivity = cusum_sensitivity
        self.alpha = drift_detection_alpha

        self.cusum_states: Dict[str, torch.Tensor] = {}
        self.gradient_means: Dict[str, torch.Tensor] = {}
        self.gradient_vars: Dict[str, torch.Tensor] = {}
        self.drift_flags: Dict[str, torch.Tensor] = {}
        self.block_counts: Dict[str, int] = {}

    def register_blocks(
        self,
        param_name: str,
        num_blocks: int,
        block_size: int = 128,
    ) -> None:
        """Register a parameter for CUSUM tracking."""
        device = torch.device("cpu")
        self.cusum_states[param_name] = torch.zeros(num_blocks, device=device)
        self.gradient_means[param_name] = torch.ones(num_blocks, device=device) * 1e-6
        self.gradient_vars[param_name] = torch.ones(num_blocks, device=device) * 1e-6
        self.drift_flags[param_name] = torch.zeros(num_blocks, dtype=torch.bool, device=device)
        self.block_counts[param_name] = num_blocks

    def update(
        self,
        param_name: str,
        grad_block: torch.Tensor,
        block_idx: int,
    ) -> None:
        """
        Update CUSUM statistic for a single block's gradient.

        Args:
            param_name: Parameter identifier
            grad_block: Gradient tensor for this block (reshaped to 1D or already scalar norm)
            block_idx: Which block index (0 to num_blocks-1)
        """
        if param_name not in self.cusum_states:
            return

        grad_norm = torch.norm(grad_block).item()
        grad_norm = max(grad_norm, 1e-8)

        mu = self.gradient_means[param_name][block_idx].item()
        sigma = math.sqrt(self.gradient_vars[param_name][block_idx].item())

        x_normalized = (grad_norm - mu) / (sigma + 1e-8)

        s_prev = self.cusum_states[param_name][block_idx].item()
        s_new = max(0.0, s_prev + x_normalized - self.sensitivity)
        self.cusum_states[param_name][block_idx] = torch.tensor(s_new, device=grad_block.device)

        self.gradient_means[param_name][block_idx] = (
            (1 - self.alpha) * mu + self.alpha * grad_norm
        )
        self.gradient_vars[param_name][block_idx] = (
            (1 - self.alpha) * self.gradient_vars[param_name][block_idx] +
            self.alpha * (grad_norm - mu) ** 2
        )

        if s_new > self.threshold:
            self.drift_flags[param_name][block_idx] = True

    def detect_drift(self, param_name: str) -> List[int]:
        """
        Return indices of blocks where drift was detected (CUSUM exceeded threshold).

        Returns:
            List of block indices with active drift flags
        """
        if param_name not in self.drift_flags:
            return []

        drifted = torch.where(self.drift_flags[param_name])[0].tolist()
        return drifted

    def reset_block(self, param_name: str, block_idx: int) -> None:
        """Reset CUSUM counter after block is reclassified (back to Adam)."""
        if param_name in self.cusum_states:
            self.cusum_states[param_name][block_idx] = 0.0
            self.drift_flags[param_name][block_idx] = False

    def reset_all(self) -> None:
        """Reset all CUSUM states."""
        for param_name in self.cusum_states:
            self.cusum_states[param_name].zero_()
            self.drift_flags[param_name].zero_()

    def get_drift_report(self) -> Dict[str, dict]:
        """
        Comprehensive drift report for all tracked parameters.

        Returns:
            Dict mapping param_name -> {num_drifted, cusum_mean, cusum_max, flags}
        """
        report = {}
        for param_name in self.cusum_states:
            drifted = torch.sum(self.drift_flags[param_name]).item()
            cusum_tensor = self.cusum_states[param_name]
            report[param_name] = {
                "num_drifted_blocks": int(drifted),
                "total_blocks": self.block_counts[param_name],
                "drift_rate": drifted / max(1, self.block_counts[param_name]),
                "cusum_mean": cusum_tensor.mean().item(),
                "cusum_max": cusum_tensor.max().item(),
            }
        return report


class SNRSkipScheduler:
    """
    Computes optimal update skip interval K_skip based on Signal-to-Noise Ratio.

    For semi-committed weights (not deeply enough committed for CUSUM):
    K_skip = ceil(1 / SNR^2) where SNR = |m_hat| / sqrt(v_hat + eps)

    Reduces update frequency: accumulate gradients in FP16 buffer for K_skip steps,
    then perform one Adam step using the mean accumulated gradient.
    Recomputes K_skip every N steps since SNR evolves during training.
    """

    def __init__(
        self,
        snr_skip_k_max: int = 256,
        snr_skip_recompute_interval: int = 100,
        min_skip: int = 1,
    ):
        """
        Args:
            snr_skip_k_max: Maximum skip interval (capped at this value)
            snr_skip_recompute_interval: Recompute K_skip every N training steps
            min_skip: Minimum skip interval (1 = no skipping if SNR too high)
        """
        self.k_max = snr_skip_k_max
        self.recompute_interval = snr_skip_recompute_interval
        self.min_skip = min_skip

        self.k_skip_schedule: Dict[str, int] = {}
        self.grad_accumulator: Dict[str, torch.Tensor] = {}
        self.step_counters: Dict[str, int] = {}
        self.last_recompute_step: Dict[str, int] = {}

    def register_parameter(self, param_name: str, param_shape: torch.Size) -> None:
        """Register a parameter for SNR-skip tracking."""
        self.k_skip_schedule[param_name] = 1
        self.grad_accumulator[param_name] = torch.zeros(
            param_shape,
            dtype=torch.float16,
            device=torch.device("cpu"),
        )
        self.step_counters[param_name] = 0
        self.last_recompute_step[param_name] = 0

    def should_update(self, param_name: str, global_step: int) -> bool:
        """
        Check if we should perform an Adam update step for this parameter.

        Returns True every K_skip steps. Returns False otherwise (accumulate only).
        """
        if param_name not in self.step_counters:
            return True

        k_skip = self.k_skip_schedule[param_name]
        counter = self.step_counters[param_name]

        return (counter % k_skip) == 0

    def accumulate_gradient(self, param_name: str, grad: torch.Tensor) -> None:
        """Accumulate gradient in FP16 buffer (bias toward lower precision)."""
        if param_name in self.grad_accumulator:
            self.grad_accumulator[param_name] += grad.to(torch.float16)
        self.step_counters[param_name] += 1

    def get_accumulated_grad(self, param_name: str) -> Optional[torch.Tensor]:
        """
        Get the accumulated gradient and reset accumulator.
        Called when should_update() returns True.

        Returns mean of accumulated gradients (scaled down by K_skip).
        """
        if param_name not in self.grad_accumulator:
            return None

        k_skip = self.k_skip_schedule[param_name]
        accum = self.grad_accumulator[param_name]
        mean_grad = accum / max(1, k_skip)

        self.grad_accumulator[param_name].zero_()
        return mean_grad

    def recompute_k_skip(
        self,
        param_name: str,
        adam_m: torch.Tensor,
        adam_v: torch.Tensor,
        global_step: int,
    ) -> int:
        """
        Recompute K_skip = ceil(1/SNR^2) based on current Adam moments.

        SNR = |m_hat| / sqrt(v_hat) measures signal confidence.
        High SNR -> weights are stable -> can skip updates.
        Low SNR -> weights still learning -> must update often.

        Args:
            adam_m: Adam first moment (biased toward gradient direction)
            adam_v: Adam second moment (gradient variance)
            global_step: Current training step (for interval checking)

        Returns:
            Updated K_skip value
        """
        if param_name not in self.last_recompute_step:
            self.last_recompute_step[param_name] = 0

        if global_step - self.last_recompute_step[param_name] < self.recompute_interval:
            return self.k_skip_schedule[param_name]

        m_norm = torch.abs(adam_m).mean().item()
        v_norm = torch.sqrt(adam_v.mean()).item()

        snr = m_norm / (v_norm + 1e-8)
        k_skip_float = 1.0 / (snr ** 2 + 1e-8)
        k_skip = max(
            self.min_skip,
            min(self.k_max, math.ceil(k_skip_float))
        )

        self.k_skip_schedule[param_name] = k_skip
        self.last_recompute_step[param_name] = global_step

        return k_skip

    def get_skip_schedule(self) -> Dict[str, int]:
        """Return current K_skip values for all tracked parameters."""
        return dict(self.k_skip_schedule)
