"""
AutoFreeze: Layer-level gradient-norm convergence freezing.

Principle: A layer whose per-layer gradient norm has converged (relative change
< threshold over a window) is no longer learning meaningfully. Freeze it.

Complementary to SPRT (weight-level) and C3 ProgressiveFreezer (commitment-rate).
A layer where SPRT has frozen >90% of its weights is a prime AutoFreeze candidate.

Always active (no flags) when use_ternary=True. Grace period enforces step >= 500
before any freezing.
"""

import torch
import torch.nn as nn
from collections import deque
from typing import Dict, Set, List, Optional


class AutoFreeze:
    """Freeze layers when per-layer gradient norm has converged.

    Algorithm per layer i:
        1. Every check_interval steps, compute grad_norm_i for all params in layer i.
        2. Append to a window of size window_size.
        3. When len(window) == window_size:
               rel_change = std(window) / (mean(window) + eps)
           If rel_change < convergence_threshold AND step >= grace_period -> freeze.
        4. SPRT synergy: if SPRT has frozen >= sprt_synergy_threshold fraction of a
           layer's weights, halve the threshold (easier to freeze when SPRT agrees).
    """

    def __init__(
        self,
        model: nn.Module,
        check_interval: int = 50,
        window_size: int = 10,
        convergence_threshold: float = 0.05,
        grace_period: int = 500,
        sprt_synergy_threshold: float = 0.90,
    ):
        self.model = model
        self.check_interval = check_interval
        self.window_size = window_size
        self.convergence_threshold = convergence_threshold
        self.grace_period = grace_period
        self.sprt_synergy_threshold = sprt_synergy_threshold

        self.frozen_layers: Set[int] = set()
        self._grad_norm_windows: Dict[int, deque] = {}
        self._freeze_history: List[tuple] = []

        self.n_layers = len(model.layers) if hasattr(model, "layers") else 0
        for i in range(self.n_layers):
            self._grad_norm_windows[i] = deque(maxlen=window_size)

        print(
            f"[AutoFreeze] Initialized: {self.n_layers} layers | "
            f"check_interval={check_interval} | window={window_size} | "
            f"threshold={convergence_threshold:.2%} | grace={grace_period}"
        )

    def _get_layer_grad_norm(self, layer: nn.Module) -> Optional[float]:
        total_sq = 0.0
        has_grad = False
        for param in layer.parameters():
            if param.grad is not None:
                total_sq += param.grad.detach().float().norm().item() ** 2
                has_grad = True
        return total_sq ** 0.5 if has_grad else None

    def _get_sprt_frozen_fraction(self, layer: nn.Module) -> float:
        frozen, total = 0, 0
        for module in layer.modules():
            if (
                hasattr(module, "use_info_freeze")
                and module.use_info_freeze
                and hasattr(module, "_freeze_mask")
                and module._freeze_mask is not None
            ):
                frozen += module._freeze_mask.sum().item()
                total += module._freeze_mask.numel()
        return frozen / total if total > 0 else 0.0

    def step(self, global_step: int) -> dict:
        """Call AFTER loss.backward() and BEFORE optimizer.zero_grad().
        Gradients must be live."""
        if global_step < self.grace_period or global_step % self.check_interval != 0:
            return {"newly_frozen": [], "total_frozen": len(self.frozen_layers)}

        newly_frozen = []

        for i, layer in enumerate(self.model.layers):
            if i in self.frozen_layers:
                continue

            gnorm = self._get_layer_grad_norm(layer)
            if gnorm is None:
                continue

            self._grad_norm_windows[i].append(gnorm)
            window = self._grad_norm_windows[i]
            if len(window) < self.window_size:
                continue

            norms = list(window)
            mean_norm = sum(norms) / len(norms)
            if mean_norm < 1e-10:
                continue

            std_norm = (sum((v - mean_norm) ** 2 for v in norms) / len(norms)) ** 0.5
            rel_change = std_norm / (mean_norm + 1e-10)

            eff_threshold = self.convergence_threshold
            sprt_frac = self._get_sprt_frozen_fraction(layer)
            if sprt_frac >= self.sprt_synergy_threshold:
                eff_threshold *= 2.0

            if rel_change < eff_threshold:
                layer.requires_grad_(False)
                self.frozen_layers.add(i)
                newly_frozen.append(i)
                self._freeze_history.append((global_step, i, rel_change, sprt_frac))
                print(
                    f"[AutoFreeze] Layer {i} FROZEN at step {global_step} | "
                    f"rel_change={rel_change:.4f} < eff_threshold={eff_threshold:.4f} | "
                    f"mean_gnorm={mean_norm:.4e} | sprt_frac={sprt_frac:.2%} | "
                    f"frozen={len(self.frozen_layers)}/{self.n_layers}"
                )

        if newly_frozen:
            pct = 100 * len(self.frozen_layers) / max(1, self.n_layers)
            print(
                f"[AutoFreeze] Total frozen: {len(self.frozen_layers)}/{self.n_layers} ({pct:.1f}%)"
            )

        return {"newly_frozen": newly_frozen, "total_frozen": len(self.frozen_layers)}

    def get_stats(self) -> dict:
        return {
            "frozen_layers": sorted(self.frozen_layers),
            "n_frozen": len(self.frozen_layers),
            "n_total": self.n_layers,
            "pct_frozen": 100 * len(self.frozen_layers) / max(1, self.n_layers),
        }
