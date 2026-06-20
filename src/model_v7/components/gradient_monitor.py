"""Gradient Health Metrics Monitor (R4-A4-F10).

Tracks gradient norms, update norms, and detects anomalies
using running statistics with a configurable window.
"""
import torch
import torch.nn as nn
from collections import deque


class GradientMonitor:
    """Monitor gradient and update health during training."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._grad_norms = deque(maxlen=window_size)
        self._update_norms = deque(maxlen=window_size)
        self._per_module_grad_norms = {}  # name -> deque
        self._prev_params = {}  # name -> tensor (for update norms)
        self._anomaly_count = 0

    def log_grad_norms(self, model: nn.Module, step: int):
        """Log global and per-module gradient norms after backward."""
        total_norm_sq = 0.0
        for name, p in model.named_parameters():
            if p.grad is not None:
                pnorm = p.grad.data.float().norm().item()
                total_norm_sq += pnorm ** 2
                if name not in self._per_module_grad_norms:
                    self._per_module_grad_norms[name] = deque(maxlen=self.window_size)
                self._per_module_grad_norms[name].append(pnorm)
        global_norm = total_norm_sq ** 0.5
        self._grad_norms.append(global_norm)

        # Check for anomaly
        if self._detect_anomaly(self._grad_norms, global_norm):
            self._anomaly_count += 1
            print(f"[GRAD-MONITOR] WARNING step {step}: grad_norm={global_norm:.4f} is anomalous "
                  f"(mean={self._running_mean(self._grad_norms):.4f}, "
                  f"std={self._running_std(self._grad_norms):.4f})")

        # Check for NaN/Inf
        if not torch.isfinite(torch.tensor(global_norm)):
            print(f"[GRAD-MONITOR] CRITICAL step {step}: grad_norm is NaN/Inf!")
            self._anomaly_count += 1

        return global_norm

    def snapshot_params(self, model: nn.Module):
        """Snapshot current params for update norm computation."""
        self._prev_params = {
            name: p.data.clone() for name, p in model.named_parameters()
            if p.requires_grad
        }

    def log_update_norms(self, model: nn.Module, step: int):
        """Log parameter update norms after optimizer.step()."""
        if not self._prev_params:
            return 0.0

        total_update_sq = 0.0
        for name, p in model.named_parameters():
            if name in self._prev_params:
                delta = (p.data - self._prev_params[name]).float().norm().item()
                total_update_sq += delta ** 2
        update_norm = total_update_sq ** 0.5
        self._update_norms.append(update_norm)

        # Detect grad-without-update or update-without-grad anomaly
        if len(self._grad_norms) > 0 and len(self._update_norms) > 10:
            last_grad = self._grad_norms[-1] if self._grad_norms else 0
            if last_grad > 0 and update_norm < 1e-10:
                print(f"[GRAD-MONITOR] WARNING step {step}: gradient present but update near-zero "
                      f"(grad={last_grad:.4f}, update={update_norm:.2e})")
            if update_norm > 0 and self._detect_anomaly(self._update_norms, update_norm):
                print(f"[GRAD-MONITOR] WARNING step {step}: update_norm={update_norm:.4f} is anomalous")

        # Clear snapshot to save memory
        self._prev_params.clear()
        return update_norm

    def _detect_anomaly(self, history: deque, value: float, z_threshold: float = 3.0) -> bool:
        """Detect if value is anomalous based on running stats."""
        if len(history) < 10:
            return False
        mean = self._running_mean(history)
        std = self._running_std(history)
        if std < 1e-8:
            return False
        return abs(value - mean) > z_threshold * std

    @staticmethod
    def _running_mean(history: deque) -> float:
        if not history:
            return 0.0
        return sum(history) / len(history)

    @staticmethod
    def _running_std(history: deque) -> float:
        if len(history) < 2:
            return 0.0
        mean = sum(history) / len(history)
        var = sum((x - mean) ** 2 for x in history) / (len(history) - 1)
        return var ** 0.5

    def get_health_report(self) -> dict:
        """Return current gradient health summary."""
        return {
            'anomaly_count': self._anomaly_count,
            'grad_norm_mean': self._running_mean(self._grad_norms),
            'grad_norm_std': self._running_std(self._grad_norms),
            'update_norm_mean': self._running_mean(self._update_norms),
            'update_norm_std': self._running_std(self._update_norms),
            'window_size': self.window_size,
            'samples': len(self._grad_norms),
        }
