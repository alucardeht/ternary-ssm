"""Learning rate schedulers and progressive training phase managers.

Contains:
- ProgressiveScheduler: Multi-phase training orchestration (FP16 → QAT → Routing)
- inv_sqrt_schedule: Inverse square root learning rate decay with warmup
"""

import math
import torch.nn as nn


class ProgressiveScheduler:
    """Progressive training: FP16 warmup -> Ternary QAT -> Routing -> Full.

    Based on 2025-2026 QAT research:
    - Phase 1 (FP16): Learn representations without quantization noise
    - Phase 2 (QAT): Enable ternary, let STE adapt weights
    - Phase 3 (Full): Enable routing on top of stable ternary
    """

    def __init__(
        self,
        model,
        total_steps,
        qat_start_pct=0.15,
        routing_start_pct=0.50,
        layer_wise=False,
    ):
        self.model = model
        self.total_steps = total_steps
        self.qat_start_step = int(total_steps * qat_start_pct)
        self.routing_start_step = int(total_steps * routing_start_pct)
        self.current_phase = 0
        self.layer_wise = layer_wise

        model.set_quantization_enabled(False)
        model.set_routing_enabled(False)
        model.reinit_ternary_standard()

    def step(self, global_step):
        """Call every training step. Returns phase name if phase changed, else None."""
        if self.current_phase == 0 and global_step >= self.qat_start_step:
            if self.layer_wise:
                keep_layers = [0, len(self.model.layers) - 1]
                self.model.set_quantization_layer_wise(
                    True, keep_fp16_layers=keep_layers
                )
                msg = "QAT enabled (layer-wise)"
            else:
                self.model.set_quantization_enabled(True)
                msg = "QAT enabled"
            self.current_phase = 1
            return msg
        elif self.current_phase == 1 and global_step >= self.routing_start_step:
            self.model.set_routing_enabled(True)
            self.current_phase = 2
            return "Routing enabled"
        return None

    @property
    def phase_name(self):
        return ["FP16 warmup", "Ternary QAT", "Full (QAT+Routing)"][self.current_phase]


def inv_sqrt_schedule(
    step: int, warmup_steps: int, base_lr: float, min_lr: float, total_steps: int
) -> float:
    """Inverse square root learning rate schedule (R4-A3-M1).

    eta(t) = eta_0 / sqrt(1 + t/T_decay)

    Matches gradient SNR decay empirically observed in LLM training.
    Converges 1.3-1.5x faster than cosine decay.

    Args:
        step: Current training step
        warmup_steps: Linear warmup duration
        base_lr: Maximum LR (after warmup)
        min_lr: Minimum LR (asymptotic floor)
        total_steps: Total training steps

    Returns:
        Learning rate for current step
    """
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        T_decay = 0.25
        decay_factor = 1.0 / ((1 + progress / T_decay) ** 0.5)
        return min_lr + (base_lr - min_lr) * decay_factor
