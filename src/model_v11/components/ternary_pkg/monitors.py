import torch


class TernaryFlipRateTracker:
    """Track ternary weight flip rate to detect exploration->commitment phase transition.

    Monitors the rate at which ternary weights change across steps. During exploration,
    weights flip frequently. As training stabilizes, flip rate drops sharply. This tracker
    detects that transition and can trigger learning rate decay.

    Attributes:
        threshold: EMA flip rate below which phase transition is triggered (default: 0.05)
        ema_beta: EMA decay for flip rate smoothing (default: 0.9)
        min_step: Minimum step before transition can occur (default: 500)
        transitioned: Whether transition has been detected
        transition_step: Step at which transition occurred (-1 if not yet)
    """

    def __init__(self, threshold=0.05, ema_beta=0.9, min_step=500):
        self.threshold = threshold
        self.ema_beta = ema_beta
        self.min_step = min_step
        self.prev_weights = {}
        self.flip_rate_ema = None
        self.transitioned = False
        self.transition_step = -1

    def update(self, model, step):
        """Compute flip rate, detect phase transition.

        Args:
            model: Model to analyze
            step: Current training step

        Returns:
            Dict with metrics: flip_rate, flip_rate_ema, num_flips, num_ternary_params,
                               transitioned, just_transitioned
        """
        current_weights = {}
        for name, module in model.named_modules():
            if hasattr(module, "weight") and hasattr(module, "quantize_enabled"):
                if getattr(module, "quantize_enabled", False):
                    w = module.weight.data
                    w_quant = torch.sign(w)
                    current_weights[name] = w_quant.detach().clone()

        if not self.prev_weights:
            self.prev_weights = current_weights
            total_params = sum(w.numel() for w in current_weights.values())
            return {
                "flip_rate": 0.0,
                "flip_rate_ema": 0.0,
                "num_flips": 0,
                "num_ternary_params": total_params,
                "transitioned": False,
                "just_transitioned": False,
            }

        num_flips = 0
        num_params = 0
        for name, w_curr in current_weights.items():
            if name in self.prev_weights:
                w_prev = self.prev_weights[name]
                num_flips += (w_curr != w_prev).sum().item()
                num_params += w_curr.numel()

        flip_rate = num_flips / max(1, num_params)

        if self.flip_rate_ema is None:
            self.flip_rate_ema = flip_rate
        else:
            self.flip_rate_ema = (
                self.ema_beta * self.flip_rate_ema + (1 - self.ema_beta) * flip_rate
            )

        just_transitioned = False
        if not self.transitioned and step >= self.min_step:
            if self.flip_rate_ema < self.threshold:
                self.transitioned = True
                self.transition_step = step
                just_transitioned = True

        self.prev_weights = current_weights

        return {
            "flip_rate": flip_rate,
            "flip_rate_ema": self.flip_rate_ema,
            "num_flips": num_flips,
            "num_ternary_params": num_params,
            "transitioned": self.transitioned,
            "just_transitioned": just_transitioned,
        }
