"""TRANSOM-TEE Anomaly Detection with Auto-Rollback (R4-A4-F6).

Multi-metric monitoring (loss, grad norm, update norm) + statistical
anomaly detection. Auto-rollback on divergence/NaN using checkpoint
ring buffer.
"""
import torch
import torch.nn as nn
from collections import deque
from typing import Optional


class AnomalyDetector:
    """Detect training anomalies and trigger auto-rollback."""

    def __init__(self, window_size: int = 100, z_threshold: float = 3.0,
                 ring_buffer_size: int = 3, nan_patience: int = 0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.ring_buffer_size = ring_buffer_size
        self.nan_patience = nan_patience

        self._metrics = {}
        self._checkpoint_ring = deque(maxlen=ring_buffer_size)
        self._nan_count = 0
        self._anomaly_count = 0
        self._rollback_count = 0

    def update_metric(self, name: str, value: float, step: int) -> dict:
        """Update a metric and check for anomaly.

        Returns dict with 'is_anomaly', 'is_nan', 'z_score', 'should_rollback'.
        """
        result = {
            'is_anomaly': False,
            'is_nan': False,
            'z_score': 0.0,
            'should_rollback': False,
        }

        if not torch.isfinite(torch.tensor(value)):
            self._nan_count += 1
            result['is_nan'] = True
            result['is_anomaly'] = True
            if self._nan_count > self.nan_patience:
                result['should_rollback'] = True
                print(f"[ANOMALY] CRITICAL step {step}: {name}=NaN/Inf "
                      f"(count={self._nan_count}). ROLLBACK triggered.")
            return result

        self._nan_count = 0

        if name not in self._metrics:
            self._metrics[name] = deque(maxlen=self.window_size)
        self._metrics[name].append(value)

        history = self._metrics[name]
        if len(history) >= 20:
            mean = sum(history) / len(history)
            var = sum((x - mean) ** 2 for x in history) / (len(history) - 1)
            std = var ** 0.5
            if std > 1e-8:
                z = (value - mean) / std
                result['z_score'] = z
                if abs(z) > self.z_threshold:
                    result['is_anomaly'] = True
                    self._anomaly_count += 1
                    print(f"[ANOMALY] step {step}: {name}={value:.4f} "
                          f"z={z:.1f} (mean={mean:.4f}, std={std:.4f})")
                    if abs(z) > self.z_threshold * 2:
                        result['should_rollback'] = True
                        print(f"[ANOMALY] EXTREME anomaly: z={z:.1f} > {self.z_threshold * 2:.1f}. "
                              f"ROLLBACK triggered.")

        return result

    def save_checkpoint_to_ring(self, state_dict: dict, step: int):
        """Save checkpoint to ring buffer for rollback."""
        import io
        buf = io.BytesIO()
        torch.save(state_dict, buf)
        self._checkpoint_ring.append({
            'buffer': buf,
            'step': step,
        })

    def rollback(self) -> Optional[dict]:
        """Rollback to most recent good checkpoint in ring buffer."""
        if not self._checkpoint_ring:
            print("[ANOMALY] No checkpoints in ring buffer for rollback!")
            return None

        ckpt = self._checkpoint_ring[-1]
        ckpt['buffer'].seek(0)
        state_dict = torch.load(ckpt['buffer'], map_location='cpu', weights_only=False)
        self._rollback_count += 1
        print(f"[ANOMALY] Rolled back to step {ckpt['step']} "
              f"(rollback #{self._rollback_count})")
        return state_dict

    def get_stats(self) -> dict:
        return {
            'anomaly_count': self._anomaly_count,
            'rollback_count': self._rollback_count,
            'ring_buffer_size': len(self._checkpoint_ring),
            'nan_count': self._nan_count,
        }
