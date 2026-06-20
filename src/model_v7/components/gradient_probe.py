import torch
import logging

logger = logging.getLogger(__name__)


class OutputGradientProbe:
    """Monitor layer health via output gradient statistics.

    Tracks dL/dy variance (free from backward pass). Low variance indicates
    dying layer, high variance indicates instability.
    """

    def __init__(self, ema_decay=0.99, var_alarm_low=1e-6, var_alarm_high=1e3):
        self.ema_decay = ema_decay
        self.var_alarm_low = var_alarm_low
        self.var_alarm_high = var_alarm_high

        self.ema_mean = None
        self.ema_var = None
        self.alarms = []

    def update(self, grad_output):
        """Called from backward hook with dL/dy tensor."""
        if grad_output is None or grad_output.numel() == 0:
            return

        with torch.no_grad():
            mean = grad_output.mean().item()
            var = grad_output.var().item()

            if self.ema_mean is None:
                self.ema_mean = mean
                self.ema_var = max(var, 1e-8)
            else:
                self.ema_mean = (self.ema_decay * self.ema_mean +
                                (1 - self.ema_decay) * mean)
                self.ema_var = (self.ema_decay * self.ema_var +
                               (1 - self.ema_decay) * max(var, 1e-8))

            if self.ema_var < self.var_alarm_low:
                alarm = f"Low variance {self.ema_var:.2e} - layer dying"
                self.alarms.append(alarm)
                logger.warning(f"[Probe] {alarm}")
            elif self.ema_var > self.var_alarm_high:
                alarm = f"High variance {self.ema_var:.2e} - instability"
                self.alarms.append(alarm)
                logger.warning(f"[Probe] {alarm}")

    def get_stats(self):
        return {
            'mean': self.ema_mean or 0.0,
            'var': self.ema_var or 0.0,
            'alarms': len(self.alarms)
        }


def register_output_probe(module, name):
    """Register backward hook to monitor output gradients."""
    probe = OutputGradientProbe()

    def hook(module, grad_input, grad_output):
        if grad_output and grad_output[0] is not None:
            probe.update(grad_output[0])

    module.register_full_backward_hook(hook)
    module.output_probe = probe
    return probe
