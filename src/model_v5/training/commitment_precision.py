import torch
import math


class CommitmentDrivenPrecision:
    """P2: Reduce gradient precision based on weight commitment."""

    def __init__(
        self,
        eviction_threshold: float = 0.9,
        freeze_threshold: float = 0.99,
        min_bits: int = 4,
        max_bits: int = 8,
        log_interval: int = 100,
    ):
        self.eviction_threshold = eviction_threshold
        self.freeze_threshold = freeze_threshold
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.log_interval = log_interval

        self.step_count = 0
        self.stats = {}

    def apply(self, model, step: int):
        """Apply commitment-driven precision to all TernaryLinear gradients."""
        self.step_count = step
        self.stats = {}

        for name, module in model.named_modules():
            if not hasattr(module, 'committed_mask'):
                continue
            if module.committed_mask is None:
                continue

            weight = module.weight
            if weight.grad is None:
                continue

            committed_mask = module.committed_mask.float()

            self._apply_precision_reduction(weight.grad, committed_mask, name)

        if step % self.log_interval == 0 and self.stats:
            self._log_stats()

    def _apply_precision_reduction(self, grad, committed_mask, layer_name):
        """Apply precision reduction based on commitment levels."""
        C = committed_mask

        frozen_count = 0
        quantized_count = 0
        full_count = 0

        freeze_mask = C > self.freeze_threshold
        if freeze_mask.any():
            grad[freeze_mask] = 0.0
            frozen_count = freeze_mask.sum().item()

        quantize_mask = (C > 0.5) & (C <= self.freeze_threshold)
        if quantize_mask.any():
            avg_commitment = C[quantize_mask].mean().item()
            precision_factor = math.sqrt(1.0 - avg_commitment)
            n_levels = int(2 ** (self.min_bits + (self.max_bits - self.min_bits) * precision_factor))

            grad_region = grad[quantize_mask]
            grad_min = grad_region.min()
            grad_max = grad_region.max()

            if grad_max > grad_min:
                grad_norm = (grad_region - grad_min) / (grad_max - grad_min + 1e-8)
                grad_quant = (grad_norm * n_levels).round() / n_levels
                grad[quantize_mask] = grad_quant * (grad_max - grad_min) + grad_min

            quantized_count = quantize_mask.sum().item()

        full_count = grad.numel() - frozen_count - quantized_count

        self.stats[layer_name] = {
            'frozen': frozen_count,
            'quantized': quantized_count,
            'full': full_count,
            'total': grad.numel(),
        }

    def _log_stats(self):
        """Log precision reduction statistics."""
        total_frozen = 0
        total_quantized = 0
        total_full = 0
        total_all = 0

        for layer_name, counts in self.stats.items():
            total_frozen += counts['frozen']
            total_quantized += counts['quantized']
            total_full += counts['full']
            total_all += counts['total']

        if total_all > 0:
            frozen_pct = 100.0 * total_frozen / total_all
            quantized_pct = 100.0 * total_quantized / total_all
            full_pct = 100.0 * total_full / total_all
            print(f"[P2-CDP] step={self.step_count}, frozen={frozen_pct:.1f}%, "
                  f"quantized={quantized_pct:.1f}%, full_precision={full_pct:.1f}%")
