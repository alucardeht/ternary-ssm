"""
Sparse Update Scheduling (R4-A6-F1-B, R4-A6-F2-B)

Implements frequency-based and gradient-sensitivity-based sparse update strategies:
- F1-B: Partial Parameter Update (SUPAR) - Only update committed blocks every N steps
- F2-B: Gradient Sensitivity Tracking - Freeze weights with consistently low gradient variance

Reduces optimizer compute by selectively skipping updates for stable/committed weights.
"""

from typing import Dict, List, Optional, Set
import torch


class SparseUpdateScheduler:
    """
    Manages per-block update frequency based on commitment status.

    Tiers:
    - ACTIVE: Every step (normal training)
    - TRANSITIONING: Every 2 steps
    - COMMITTED: Every 8 steps (deep commitment, weight stable)
    - FROZEN: Never (permanent commitment)

    Zero-cost: leverages existing commitment masks from CommitmentDetector.
    """

    def __init__(
        self,
        sparse_update_committed_interval: int = 8,
        sparse_update_transitioning_interval: int = 2,
    ):
        """
        Args:
            sparse_update_committed_interval: Update interval for COMMITTED blocks
            sparse_update_transitioning_interval: Update interval for TRANSITIONING blocks
        """
        self.committed_interval = sparse_update_committed_interval
        self.transitioning_interval = sparse_update_transitioning_interval

        self.block_last_update: Dict[str, Dict[int, int]] = {}
        self.block_tiers: Dict[str, Dict[int, str]] = {}

    def register_parameter(
        self,
        param_name: str,
        num_blocks: int,
    ) -> None:
        """Register a parameter for sparse update tracking."""
        self.block_last_update[param_name] = {}
        self.block_tiers[param_name] = {}

        for block_idx in range(num_blocks):
            self.block_last_update[param_name][block_idx] = 0
            self.block_tiers[param_name][block_idx] = "ACTIVE"

    def set_block_tier(
        self,
        param_name: str,
        block_idx: int,
        tier: str,
    ) -> None:
        """
        Update tier classification for a block.
        Tier must be one of: ACTIVE, TRANSITIONING, COMMITTED, FROZEN
        """
        if param_name in self.block_tiers:
            self.block_tiers[param_name][block_idx] = tier

    def should_update_block(
        self,
        param_name: str,
        block_idx: int,
        step: int,
    ) -> bool:
        """
        Determine if a block should receive an optimizer update this step.

        Returns True if:
        - Tier is ACTIVE (always update)
        - Tier is TRANSITIONING and (step - last_update) >= transitioning_interval
        - Tier is COMMITTED and (step - last_update) >= committed_interval
        - Tier is FROZEN (never update)
        """
        if param_name not in self.block_tiers:
            return True

        tier = self.block_tiers[param_name].get(block_idx, "ACTIVE")

        if tier == "ACTIVE":
            return True
        elif tier == "FROZEN":
            return False
        elif tier == "TRANSITIONING":
            last_update = self.block_last_update[param_name].get(block_idx, 0)
            return (step - last_update) >= self.transitioning_interval
        elif tier == "COMMITTED":
            last_update = self.block_last_update[param_name].get(block_idx, 0)
            return (step - last_update) >= self.committed_interval
        else:
            return True

    def mark_block_updated(
        self,
        param_name: str,
        block_idx: int,
        step: int,
    ) -> None:
        """Record that a block was updated at this step."""
        if param_name in self.block_last_update:
            self.block_last_update[param_name][block_idx] = step

    def get_update_schedule(self, param_name: str) -> Dict[int, str]:
        """
        Return tier classification for all blocks of a parameter.

        Returns:
            Dict mapping block_idx -> tier (ACTIVE, TRANSITIONING, COMMITTED, FROZEN)
        """
        return dict(self.block_tiers.get(param_name, {}))

    def create_update_mask(
        self,
        param_name: str,
        num_blocks: int,
        step: int,
    ) -> torch.Tensor:
        """
        Create a boolean mask indicating which blocks should update this step.

        Returns:
            1D tensor of shape [num_blocks] with True = should update, False = skip
        """
        mask = torch.zeros(num_blocks, dtype=torch.bool)
        for block_idx in range(num_blocks):
            if self.should_update_block(param_name, block_idx, step):
                mask[block_idx] = True
        return mask


class GradientSensitivityTracker:
    """
    Tracks gradient variance per parameter as a proxy for freezing without commitment detection.

    Uses EMA to maintain running gradient variance. If variance drops below threshold
    for K consecutive steps, marks parameter as frozen. Unfreezes on large gradient spikes.

    Independent of CommitmentDetector - pure gradient-based signal.
    Works even when Adam state is unavailable (e.g., CUSUM-tracked params).
    """

    def __init__(
        self,
        grad_sensitivity_threshold: float = 1e-6,
        grad_sensitivity_patience: int = 50,
        ema_decay: float = 0.99,
    ):
        """
        Args:
            grad_sensitivity_threshold: Variance threshold below which to freeze
            grad_sensitivity_patience: Consecutive steps below threshold to trigger freeze
            ema_decay: EMA decay for gradient variance tracking (0.99 = slow update)
        """
        self.threshold = grad_sensitivity_threshold
        self.patience = grad_sensitivity_patience
        self.ema_decay = ema_decay

        self.grad_variance: Dict[str, torch.Tensor] = {}
        self.freeze_counters: Dict[str, int] = {}
        self.frozen_flags: Dict[str, bool] = {}
        self.param_names: Set[str] = set()

    def register_parameter(self, param_name: str, param_shape: torch.Size) -> None:
        """Register a parameter for sensitivity tracking."""
        self.param_names.add(param_name)
        self.grad_variance[param_name] = torch.ones(1, device=torch.device("cpu")) * 1e-8
        self.freeze_counters[param_name] = 0
        self.frozen_flags[param_name] = False

    def update(self, param_name: str, grad: torch.Tensor) -> None:
        """
        Update variance estimate with current gradient.

        Uses EMA: var = decay * var + (1 - decay) * grad^2
        """
        if param_name not in self.param_names:
            self.register_parameter(param_name, grad.shape)

        grad_squared = (grad ** 2).mean()

        var_old = self.grad_variance[param_name]
        var_new = (
            self.ema_decay * var_old +
            (1.0 - self.ema_decay) * grad_squared
        )
        self.grad_variance[param_name] = var_new

        if var_new < self.threshold:
            self.freeze_counters[param_name] += 1
            if self.freeze_counters[param_name] >= self.patience:
                self.frozen_flags[param_name] = True
        else:
            self.freeze_counters[param_name] = 0
            if var_new > self.threshold * 10:
                self.frozen_flags[param_name] = False

    def is_frozen(self, param_name: str) -> bool:
        """Check if parameter is currently frozen (low sensitivity)."""
        return self.frozen_flags.get(param_name, False)

    def get_frozen_params(self) -> List[str]:
        """Return list of currently frozen parameters."""
        return [name for name, frozen in self.frozen_flags.items() if frozen]

    def get_sensitivity_report(self) -> Dict[str, dict]:
        """
        Comprehensive report of gradient sensitivity for all tracked parameters.

        Returns:
            Dict mapping param_name -> {variance, frozen, freeze_counter}
        """
        report = {}
        for param_name in self.param_names:
            report[param_name] = {
                "variance": self.grad_variance[param_name].item(),
                "threshold": self.threshold,
                "frozen": self.frozen_flags[param_name],
                "freeze_counter": self.freeze_counters[param_name],
                "patience": self.patience,
            }
        return report

    def reset_parameter(self, param_name: str) -> None:
        """Reset tracking for a specific parameter (on unfreeze/recommit)."""
        if param_name in self.param_names:
            self.grad_variance[param_name] = torch.ones(1) * 1e-8
            self.freeze_counters[param_name] = 0
            self.frozen_flags[param_name] = False
