"""Ternary Delta Chronicle (R4-A4-M7).

Records every ternary weight flip: (index, old_value, new_value, step).
At 0.1% flip rate: 2-4 KB/step. 200-400 MB for 100K steps.
Enables training archaeology and surgical rollback.
"""
import torch
import torch.nn as nn
from typing import List, Optional
import struct
import io


class DeltaChronicle:
    """Record every ternary weight flip during training."""

    def __init__(self, max_history_mb: float = 500.0):
        self.max_history_mb = max_history_mb
        self._prev_weights = {}  # name -> int8 tensor
        self._history = []       # list of (step, flips_dict)
        self._total_bytes = 0
        self._total_flips = 0
        self._initialized = False

    def initialize(self, model: nn.Module):
        """Snapshot initial ternary weights."""
        self._prev_weights.clear()
        for name, p in model.named_parameters():
            if self._is_ternary(p):
                self._prev_weights[name] = p.data.sign().to(torch.int8).clone()
        self._initialized = True

    def record_step(self, model: nn.Module, step: int) -> dict:
        """Record all weight flips that happened this step.

        Returns dict with flip statistics.
        """
        if not self._initialized:
            self.initialize(model)
            return {'flips': 0, 'flip_rate': 0.0}

        step_flips = {}
        total_flips = 0
        total_params = 0

        for name, p in model.named_parameters():
            if name not in self._prev_weights:
                continue

            current = p.data.sign().to(torch.int8)
            prev = self._prev_weights[name]
            diff = current - prev
            changed = (diff != 0)

            if changed.any():
                positions = changed.nonzero(as_tuple=False)
                old_vals = prev[changed]
                new_vals = current[changed]

                step_flips[name] = {
                    'positions': positions.to(torch.int32),
                    'old': old_vals,
                    'new': new_vals,
                }
                n_flips = len(positions)
                total_flips += n_flips
                self._total_bytes += n_flips * 6

            total_params += p.numel()
            self._prev_weights[name] = current.clone()

        if step_flips:
            self._history.append((step, step_flips))
            self._total_flips += total_flips

        while self._total_bytes > self.max_history_mb * 1024 * 1024 and self._history:
            _, old_flips = self._history.pop(0)
            for v in old_flips.values():
                self._total_bytes -= len(v['positions']) * 6

        flip_rate = total_flips / max(total_params, 1)
        return {
            'flips': total_flips,
            'flip_rate': flip_rate,
            'total_history_flips': self._total_flips,
            'history_mb': self._total_bytes / (1024 * 1024),
        }

    def get_flips_in_range(self, start_step: int, end_step: int) -> list:
        """Get all flips between two steps."""
        return [
            (step, flips) for step, flips in self._history
            if start_step <= step <= end_step
        ]

    def get_param_flip_history(self, param_name: str) -> list:
        """Get flip history for a specific parameter."""
        return [
            (step, flips[param_name])
            for step, flips in self._history
            if param_name in flips
        ]

    def get_stats(self) -> dict:
        return {
            'total_flips': self._total_flips,
            'history_entries': len(self._history),
            'history_mb': self._total_bytes / (1024 * 1024),
            'max_history_mb': self.max_history_mb,
            'tracked_params': len(self._prev_weights),
        }

    @staticmethod
    def _is_ternary(param: torch.Tensor) -> bool:
        if not param.is_floating_point() or param.numel() < 64:
            return False
        unique = param.unique()
        return len(unique) <= 3 and all(v.item() in (-1.0, 0.0, 1.0) for v in unique)
