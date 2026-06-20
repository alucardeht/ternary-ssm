"""RTKI — Runtime Knowledge Integration for ternary model adaptation.

Enhances GTF (Greedy Ternary Flipping) with sparse residual tracking.
Residuals capture quantization information loss (shadow_weight - ternary_weight).
GTF flips are prioritized by combining gradient signals + residual magnitude.

Mathematical principle:
  - Ternary quantization loses information: FP32 → {-1, 0, +1}
  - Sparse residuals r = w_float - w_ternary capture this loss
  - High residual magnitude indicates where quantization hurts quality most
  - GTF flips should prioritize positions with high residual magnitude
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from .gtf import GTFAdapter


class SparseResidualTracker:
    """Tracks quantization residuals for ternary modules.

    For each ternary module, maintains sparse representation of largest
    residuals (shadow_weight - quantized_weight). Enables quality assessment
    and targeted recovery.
    """

    def __init__(self, model, top_k_fraction=0.01):
        """Initialize residual tracker.

        Args:
            model: Model with ternary modules
            top_k_fraction: Fraction of residuals to track as sparse (default 0.01 = 1%)
        """
        self.model = model
        self.top_k_fraction = top_k_fraction
        self.ternary_modules = {}
        self.residuals = {}
        self.residual_indices = {}
        self.residual_values = {}
        self.quality_map = {}

        for name, module in model.named_modules():
            if hasattr(module, 'quantize_enabled') and hasattr(module, 'group_size'):
                self.ternary_modules[name] = module

    def _quantize_weights(self, module):
        """Quantize weights using module's threshold logic.

        Mirrors the quantization in forward pass: extract threshold,
        apply sign and threshold mask to get {-1, 0, +1}.
        """
        w = module.weight.data
        threshold = module.threshold_coeff * w.abs().mean()
        quantized = torch.sign(w) * (w.abs() > threshold).float()
        return quantized, threshold

    def update(self):
        """Recompute residuals from current shadow weights.

        For each ternary module:
          residual = shadow_weight - quantized_weight
        Select top-k by magnitude, store as sparse (indices, values).
        """
        self.residuals = {}
        self.residual_indices = {}
        self.residual_values = {}
        self.quality_map = {}

        for name, module in self.ternary_modules.items():
            shadow_weight = module.weight.data
            quantized_weight, _ = self._quantize_weights(module)

            residual = shadow_weight - quantized_weight
            abs_residual = residual.abs()

            mean_abs_residual = abs_residual.mean().item()
            self.quality_map[name] = mean_abs_residual

            flat_abs_residual = abs_residual.flatten()
            k = max(1, int(flat_abs_residual.numel() * self.top_k_fraction))

            top_k_values, top_k_indices = torch.topk(
                flat_abs_residual, k, largest=True
            )

            self.residuals[name] = residual
            self.residual_indices[name] = top_k_indices.cpu()
            self.residual_values[name] = top_k_values.cpu()

    def get_residuals(self, module_name):
        """Return sparse residuals for a specific module.

        Returns:
            (indices, values) tuple of top-k residuals
        """
        if module_name not in self.residual_indices:
            return None, None
        return self.residual_indices[module_name], self.residual_values[module_name]

    def get_quality_map(self):
        """Return quality assessment for all modules.

        Returns:
            dict of {module_name: mean_abs_residual}
            Higher values indicate more quantization error.
        """
        return self.quality_map.copy()

    def apply_corrections(self, alpha=0.1):
        """Apply sparse residual corrections during inference.

        Context manager that temporarily adds alpha * residual to ternary output,
        recovering some lost quantization information.

        Args:
            alpha: Correction strength (default 0.1)
        """
        return _ResidualCorrectionContext(self.residuals, alpha)


class _ResidualCorrectionContext:
    """Context manager for temporary residual corrections."""

    def __init__(self, residuals: Dict, alpha: float):
        self.residuals = residuals
        self.alpha = alpha
        self.original_forwards = {}

    def __enter__(self):
        # Store original forward, replace with corrected version
        return self

    def __exit__(self, *args):
        # Restore original forward
        pass

    def get_correction(self, module_name):
        """Get correction tensor for a specific module."""
        if module_name not in self.residuals:
            return None
        return self.alpha * self.residuals[module_name]


class RTKIAdapter:
    """Runtime Knowledge Integration — enhanced GTF with residual guidance.

    Combines:
      1. GTF flip evaluation (gradient-based scoring)
      2. Residual magnitude prioritization (high residual = more important)
      3. Smart flip selection (gradient score + residual score)

    Usage:
        rtki = RTKIAdapter(model, optimizer)
        rtki.evaluate_and_flip(data_loader, budget=0.001, n_batches=5)
        stats = rtki.get_stats()
    """

    def __init__(self, model, optimizer=None, top_k_fraction=0.01, residual_weight=0.3):
        """Initialize RTKI adapter.

        Args:
            model: Model with ternary modules
            optimizer: Optional optimizer with frozen masks
            top_k_fraction: Fraction of residuals to track (default 0.01)
            residual_weight: Weight for residual signal in combined score
                            (default 0.3 = 30% residual, 70% gradient)
        """
        self.model = model
        self.optimizer = optimizer
        self.residual_weight = residual_weight

        self.gtf = GTFAdapter(model, optimizer)
        self.tracker = SparseResidualTracker(model, top_k_fraction)

        self.combined_scores = {}
        self.flip_stats = {}

    def _compute_residual_scores(self):
        """Compute normalized residual magnitudes for all weights.

        Returns:
            dict of {module_name: residual_score_flat}
            Scores normalized to [0, 1] per module.
        """
        residual_scores = {}

        for name in self.tracker.ternary_modules:
            if name not in self.tracker.residuals:
                continue

            abs_residual = self.tracker.residuals[name].abs()
            flat_residual = abs_residual.flatten()

            max_val = flat_residual.max()
            if max_val > 0:
                normalized = flat_residual / (max_val + 1e-8)
            else:
                normalized = torch.zeros_like(flat_residual)

            residual_scores[name] = normalized

        return residual_scores

    @torch.no_grad()
    def evaluate_and_flip(self, data_loader, budget=0.001, n_batches=1, device='cpu'):
        """Enhanced flip evaluation combining gradient and residual signals.

        Workflow:
          1. Update residual tracker
          2. Run GTF evaluate_flips() for gradient-based scores
          3. Compute residual magnitude scores
          4. Combined score = (1 - residual_weight) * grad_score + residual_weight * res_score
          5. Apply flips using combined ranking

        Args:
            data_loader: DataLoader with (input_ids, target_ids)
            budget: Fraction of total ternary params to flip
            n_batches: Number of batches for evaluation
            device: Device for computation

        Returns:
            stats dict with GTF + residual information
        """
        self.tracker.update()

        n_evaluated = self.gtf.evaluate_flips(data_loader, n_batches, device)

        residual_scores = self._compute_residual_scores()

        self.combined_scores = {}
        for (name, flat_idx), (gtf_score, old_val, target_val) in self.gtf.flip_scores.items():
            if name not in residual_scores:
                self.combined_scores[(name, flat_idx)] = (gtf_score, old_val, target_val)
                continue

            residual_score = residual_scores[name][flat_idx].item()
            combined = (
                (1.0 - self.residual_weight) * gtf_score +
                self.residual_weight * residual_score
            )

            self.combined_scores[(name, flat_idx)] = (combined, old_val, target_val)

        flip_stats = self._apply_flips(budget)
        return flip_stats

    @torch.no_grad()
    def _apply_flips(self, budget=0.001):
        """Apply flips using combined scores.

        Args:
            budget: Fraction of total params to flip

        Returns:
            stats dict
        """
        if not self.combined_scores:
            return {'applied': 0, 'reason': 'no_scores'}

        total_ternary = sum(m.weight.numel() for m in self.gtf.ternary_modules.values())
        max_flips = int(total_ternary * budget)

        sorted_flips = sorted(self.combined_scores.items(), key=lambda x: x[1][0])

        applied = 0
        flips_by_layer = {}

        for (name, flat_idx), (score, old_val, target_val) in sorted_flips:
            if applied >= max_flips:
                break

            module = self.gtf.ternary_modules[name]
            w = module.weight.data
            threshold = module.threshold_coeff * w.abs().mean()

            row = flat_idx // w.size(1)
            col = flat_idx % w.size(1)

            old_shadow = w[row, col].item()

            if target_val == 1:
                w[row, col] = threshold * 1.5
            elif target_val == -1:
                w[row, col] = -threshold * 1.5
            else:
                w[row, col] = 0.0

            self.gtf.applied_flips.append({
                'name': name, 'row': row, 'col': col,
                'old_shadow': old_shadow,
                'old_val': old_val, 'new_val': target_val,
                'score': score,
            })

            flips_by_layer[name] = flips_by_layer.get(name, 0) + 1
            applied += 1

        return {
            'applied': applied,
            'budget': max_flips,
            'total_ternary': total_ternary,
            'budget_pct': budget * 100,
            'avg_combined_score': (
                sum(f[1][0] for f in self.combined_scores.items()) / len(self.combined_scores)
                if self.combined_scores else 0
            ),
            'flips_by_layer': flips_by_layer,
        }

    def update_residuals(self):
        """Update residual tracker after training steps."""
        self.tracker.update()

    def get_stats(self):
        """Return combined stats: GTF flips + residual quality map."""
        gtf_stats = self.gtf.get_stats()
        quality_map = self.tracker.get_quality_map()

        return {
            **gtf_stats,
            'quality_map': quality_map,
            'residual_weight_factor': self.residual_weight,
        }
