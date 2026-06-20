import torch
import logging

logger = logging.getLogger(__name__)


class CUSUMDriftDetector:
    """Detect weight drift via cumulative sum control chart.

    Monitors committed weights. If gradient accumulates beyond expected noise,
    triggers uncommitting (revert to Adam).
    """

    def __init__(self, threshold_factor=3.0, alarm_level=10.0):
        self.threshold_factor = threshold_factor
        self.alarm_level = alarm_level
        self.cusum_state = {}
        self.alarm_count = 0

    def update(self, param, grad, optimizer_state):
        """Update CUSUM for parameter. Return True if drift detected."""
        param_id = id(param)

        if param_id not in self.cusum_state:
            self.cusum_state[param_id] = 0.0

        if optimizer_state is None or 'exp_avg_sq' not in optimizer_state:
            return False

        v = optimizer_state['exp_avg_sq']
        if v is None or v.numel() == 0:
            return False

        noise_std = torch.sqrt(v.mean()).item()
        if noise_std < 1e-8:
            return False

        threshold = self.threshold_factor * noise_std
        grad_norm = grad.abs().mean().item()

        S = max(0.0, self.cusum_state[param_id] + grad_norm - threshold)
        self.cusum_state[param_id] = S

        if S > self.alarm_level:
            self.alarm_count += 1
            self.cusum_state[param_id] = 0.0
            return True

        return False

    def reset(self, param):
        param_id = id(param)
        if param_id in self.cusum_state:
            del self.cusum_state[param_id]

    def get_stats(self):
        active = [s for s in self.cusum_state.values() if s > 0]
        return {
            'active_count': len(active),
            'max_s': max(active) if active else 0.0,
            'alarms': self.alarm_count
        }
