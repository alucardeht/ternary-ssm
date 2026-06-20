"""
C3: Progressive Layer Freezing based on Commitment Tracking

Freezes layers when ternary weight commitment rate exceeds threshold.
Frozen layers skip gradient computation (forward and backward).
"""

import torch
import torch.nn as nn
from typing import Set, Dict, List


class ProgressiveFreezer:
    """Freeze layers based on ternary weight commitment rate.

    When a layer's commitment rate exceeds freeze_threshold, the layer
    is frozen (requires_grad_(False)) to skip gradient computation.

    Args:
        model: The model to apply freezing to
        freeze_threshold: Commitment rate threshold (0-1, default 0.99)
        check_interval: Steps between freeze checks (default 100)
        min_step: Minimum step before freezing starts (default 500)
    """

    def __init__(
        self,
        model: nn.Module,
        freeze_threshold: float = 0.99,
        check_interval: int = 100,
        min_step: int = 500,
    ):
        self.model = model
        self.threshold = freeze_threshold
        self.interval = check_interval
        self.min_step = min_step
        self.frozen_layers: Set[int] = set()
        self.freeze_history: List[tuple] = []

        self.n_layers = len(model.layers) if hasattr(model, 'layers') else 0

        if self.n_layers > 0:
            print(f"[C3-FREEZE] Initialized: threshold={freeze_threshold:.2%}, "
                  f"check_interval={check_interval}, min_step={min_step}")

    def check_and_freeze(self, step: int) -> dict:
        """Check and freeze layers if commitment threshold is met.

        Args:
            step: Current training step

        Returns:
            dict with 'newly_frozen', 'total_frozen', 'total_layers'
        """
        if step < self.min_step:
            return {
                'newly_frozen': [],
                'total_frozen': len(self.frozen_layers),
                'total_layers': self.n_layers
            }

        if step % self.interval != 0:
            return {
                'newly_frozen': [],
                'total_frozen': len(self.frozen_layers),
                'total_layers': self.n_layers
            }

        newly_frozen = []

        for i, layer in enumerate(self.model.layers):
            if i in self.frozen_layers:
                continue

            commit_rate = self._get_layer_commitment_rate(layer)

            if commit_rate > self.threshold:
                layer.requires_grad_(False)
                self.frozen_layers.add(i)
                newly_frozen.append(i)
                self.freeze_history.append((step, i, commit_rate))

                print(f"[C3-FREEZE] Layer {i} FROZEN at step {step} "
                      f"(commit_rate={commit_rate:.4f}, {len(self.frozen_layers)}/{self.n_layers} frozen)")

        if step % (self.interval * 10) == 0 and len(self.frozen_layers) > 0:
            pct = 100 * len(self.frozen_layers) / self.n_layers
            print(f"[C3-FREEZE] Progress: {len(self.frozen_layers)}/{self.n_layers} layers frozen ({pct:.1f}%)")

        return {
            'newly_frozen': newly_frozen,
            'total_frozen': len(self.frozen_layers),
            'total_layers': self.n_layers
        }

    def _get_layer_commitment_rate(self, layer: nn.Module) -> float:
        """Compute commitment rate for a layer.

        Commitment rate = fraction of ternary weights that are committed
        (margin > threshold AND stable for N steps).

        Args:
            layer: Layer module to check

        Returns:
            Commitment rate in [0, 1], or 0.0 if no commitment tracking
        """
        committed_count = 0
        total_count = 0

        for module in layer.modules():
            if not hasattr(module, 'use_commitment_tracking'):
                continue
            if not module.use_commitment_tracking:
                continue
            if not hasattr(module, 'committed_mask') or module.committed_mask is None:
                continue

            committed_count += module.committed_mask.sum().item()
            total_count += module.committed_mask.numel()

        return committed_count / total_count if total_count > 0 else 0.0

    def unfreeze_all(self):
        """Unfreeze all layers (for debugging/testing)."""
        for layer in self.model.layers:
            layer.requires_grad_(True)
        self.frozen_layers.clear()
        print(f"[C3-FREEZE] All layers unfrozen")

    def get_stats(self) -> Dict:
        """Get current freezing statistics."""
        return {
            'frozen_layers': sorted(list(self.frozen_layers)),
            'n_frozen': len(self.frozen_layers),
            'n_total': self.n_layers,
            'pct_frozen': 100 * len(self.frozen_layers) / self.n_layers if self.n_layers > 0 else 0.0,
            'freeze_history': self.freeze_history,
        }

    def get_frozen_param_count(self) -> tuple:
        """Count how many parameters are frozen.

        Returns:
            (frozen_count, total_count) tuple
        """
        frozen = sum(1 for p in self.model.parameters() if not p.requires_grad)
        total = sum(1 for p in self.model.parameters())
        return frozen, total
