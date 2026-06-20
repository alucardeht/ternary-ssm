"""
Metabolic Probe for Commitment Detection (R3-A6-P3)

Lightweight probe that estimates weight commitment WITHOUT requiring full shadow weight
and Adam state access. Uses output gradient statistics as proxy for layer activity.

Key insight: If output gradient becomes stable/low-variance, the layer's forward activation
is "committed"-the function it computes has converged. This is a cheaper alternative to
full Adam-state-based commitment detection.

Attached as backward hook to monitor ||dL/dy||_2 per layer.
"""

import math
import torch
from typing import Dict, Optional, Tuple
from collections import defaultdict


class MetabolicProbe:
    """
    Lightweight commitment estimator based on output gradient statistics.

    For each monitored layer, tracks:
    - EMA of ||dL/dy||_2 (output gradient norm)
    - EMA of variance in gradient norm over recent steps
    - Coefficient of variation: sqrt(var) / ema

    If CoV < threshold for K steps, layer is "metabolically dormant" -> committed.
    If gradient spikes suddenly, uncommit immediately (adaptive).

    Zero-cost: only two scalar EMAs per layer per step.
    """

    def __init__(
        self,
        alpha: float = 0.99,
        cv_threshold: float = 0.5,
    ):
        """
        Args:
            alpha: EMA decay for gradient norm (higher = slower adaptation)
            cv_threshold: Coefficient of variation threshold for dormancy trigger
        """
        self.alpha = alpha
        self.cv_threshold = cv_threshold

        self.layer_stats = defaultdict(
            lambda: {
                'ema': None,
                'ema_sq': None,
                'count': 0,
                'grad_history': [],
            }
        )

        self.committed_layers = set()
        self.step_count = 0

    @staticmethod
    def compute_output_gradient_norm(grad_output: torch.Tensor) -> float:
        """
        Compute ||dL/dy||_2 from gradient tensor(s).

        Handles both single tensor and tuple of gradients (for multi-output layers).

        Args:
            grad_output: Gradient w.r.t. layer output (tensor or tuple)

        Returns:
            Scalar norm value.
        """
        if isinstance(grad_output, tuple):
            grad_list = [g for g in grad_output if g is not None]
            if len(grad_list) == 0:
                return 0.0
            norm = sum(g.norm().item() ** 2 for g in grad_list) ** 0.5
        else:
            if grad_output is None:
                return 0.0
            norm = grad_output.norm().item()

        return norm

    def update(
        self,
        param_name: str,
        grad_output: torch.Tensor,
    ) -> None:
        """
        Update gradient statistics for a layer based on backward pass.

        Args:
            param_name: Name of the layer (e.g., "ssm_blocks.0")
            grad_output: Output gradient tensor(s) from backward hook
        """
        norm = self.compute_output_gradient_norm(grad_output)

        stats = self.layer_stats[param_name]

        if stats['ema'] is None:
            stats['ema'] = norm
            stats['ema_sq'] = norm ** 2
        else:
            stats['ema'] = self.alpha * stats['ema'] + (1.0 - self.alpha) * norm
            stats['ema_sq'] = self.alpha * stats['ema_sq'] + (1.0 - self.alpha) * (norm ** 2)

        stats['grad_history'].append(norm)
        if len(stats['grad_history']) > 200:
            stats['grad_history'].pop(0)

        stats['count'] += 1

    def estimate_commitment(self, param_name: str) -> float:
        """
        Estimate commitment for a layer (0.0 = fully active, 1.0 = fully committed).

        Based on coefficient of variation of output gradient norms.
        - If CoV < threshold, layer is dormant -> commitment ~= 1.0
        - If CoV > threshold, layer is active -> commitment ~= 0.0

        Args:
            param_name: Layer name

        Returns:
            Commitment score in [0, 1].
        """
        if param_name not in self.layer_stats:
            return 0.0

        stats = self.layer_stats[param_name]

        if stats['ema'] is None or stats['count'] < 10:
            return 0.0

        variance = stats['ema_sq'] - stats['ema'] ** 2
        variance = max(variance, 0.0)
        std_dev = math.sqrt(variance)

        if stats['ema'] < 1e-8:
            cov = 1.0
        else:
            cov = std_dev / (stats['ema'] + 1e-10)

        if cov < self.cv_threshold:
            commitment = 1.0
        else:
            commitment = max(0.0, 1.0 - (cov / (2.0 * self.cv_threshold)))

        return commitment

    def get_all_commitments(self) -> Dict[str, float]:
        """
        Get commitment estimates for all monitored layers.

        Returns:
            Dict mapping layer name -> commitment score [0, 1].
        """
        return {
            name: self.estimate_commitment(name)
            for name in self.layer_stats.keys()
        }

    def get_layer_stats(self, param_name: str) -> Optional[Dict]:
        """
        Get detailed statistics for a specific layer.

        Args:
            param_name: Layer name

        Returns:
            Dict with 'ema', 'ema_sq', 'count', 'cov', 'commitment', or None.
        """
        if param_name not in self.layer_stats:
            return None

        stats = self.layer_stats[param_name]

        if stats['ema'] is None:
            return None

        variance = max(stats['ema_sq'] - stats['ema'] ** 2, 0.0)
        std_dev = math.sqrt(variance)
        cov = std_dev / (stats['ema'] + 1e-10)

        return {
            'ema': stats['ema'],
            'ema_sq': stats['ema_sq'],
            'count': stats['count'],
            'std_dev': std_dev,
            'cov': cov,
            'commitment': self.estimate_commitment(param_name),
        }

    def check_uncommit_trigger(
        self,
        param_name: str,
        spike_factor: float = 10.0,
    ) -> bool:
        """
        Check if layer should be uncommitted due to sudden gradient spike.

        If gradient norm exceeds 10x its EMA suddenly, uncommit (reactivate).

        Args:
            param_name: Layer name
            spike_factor: Multiplier threshold for spike detection (default 10x)

        Returns:
            True if uncommit should be triggered.
        """
        if param_name not in self.layer_stats:
            return False

        stats = self.layer_stats[param_name]

        if stats['ema'] is None or len(stats['grad_history']) < 10:
            return False

        recent_norms = stats['grad_history'][-10:]
        recent_max = max(recent_norms)

        if recent_max > spike_factor * stats['ema']:
            return True

        return False

    def reset_layer(self, param_name: str) -> None:
        """
        Reset statistics for a specific layer (e.g., after uncommitting).

        Args:
            param_name: Layer name
        """
        if param_name in self.layer_stats:
            self.layer_stats[param_name] = {
                'ema': None,
                'ema_sq': None,
                'count': 0,
                'grad_history': [],
            }
            if param_name in self.committed_layers:
                self.committed_layers.remove(param_name)

    def get_summary(self) -> Dict:
        """
        Get summary statistics across all monitored layers.

        Returns:
            Dict with aggregated stats and per-layer details.
        """
        commitments = self.get_all_commitments()

        if not commitments:
            return {
                'num_layers': 0,
                'committed_count': 0,
                'committed_fraction': 0.0,
                'avg_commitment': 0.0,
                'layers': {},
            }

        committed_count = sum(1 for c in commitments.values() if c > 0.5)
        avg_commitment = sum(commitments.values()) / len(commitments)

        layers_detail = {
            name: self.get_layer_stats(name)
            for name in commitments.keys()
        }

        return {
            'num_layers': len(commitments),
            'committed_count': committed_count,
            'committed_fraction': committed_count / len(commitments),
            'avg_commitment': avg_commitment,
            'layers': layers_detail,
        }

    def step(self) -> None:
        """Increment internal step counter."""
        self.step_count += 1

    def create_backward_hook(self, module_name: str):
        """
        Create a backward hook function for torch.nn.Module.register_backward_hook().

        Usage:
            probe = MetabolicProbe()
            layer = model.ssm_blocks[0]
            hook = probe.create_backward_hook("ssm_blocks.0")
            layer.register_backward_hook(hook)

        Args:
            module_name: Name to associate with this module

        Returns:
            Hook function that accepts (module, grad_input, grad_output).
        """
        def backward_hook(module, grad_input, grad_output):
            self.update(module_name, grad_output)
            return None

        return backward_hook
