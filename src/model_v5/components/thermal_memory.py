"""
Thermal Memory Manager - Temperature-based resource allocation for ternary training.

References:
- R4-A6-P8: Thermal memory classifies parameters into temperature bands (HOT, WARM, COOL, COLD)
  based on activity metrics (gradient norms, weight flip rates).

Principle:
  HOT (active) parameters: full precision, frequent updates, large caches
  COLD (committed) parameters: lower precision, rare updates, compressed caches

This enables dynamic resource allocation: hot parameters get full training resources,
cold parameters are compressed to save memory.

Temperature metric combines:
  - Gradient norm: high gradients = active parameter
  - Weight flip rate: frequent ternary flips = exploring state space
  - Velocity: rate of change of commitment score
"""

from typing import Dict, Optional, List
import math


class ThermalMemoryManager:
    """
    Manages parameter classification based on temperature (activity level).
    Temperature is computed from gradient statistics and flip rates.
    """

    def __init__(
        self,
        ema_decay: float = 0.99,
        hot_threshold: float = 0.7,
        warm_threshold: float = 0.4,
        cool_threshold: float = 0.1,
    ):
        """
        Args:
            ema_decay: Exponential moving average decay for temperature smoothing
            hot_threshold: Temperature >= this -> HOT
            warm_threshold: Temperature in [warm, hot) -> WARM
            cool_threshold: Temperature in [cool, warm) -> COOL
            cool_threshold < temp: COLD
        """
        assert 0 < ema_decay < 1, "ema_decay must be in (0, 1)"
        assert cool_threshold < warm_threshold < hot_threshold, "Thresholds must be ordered"

        self.ema_decay = ema_decay
        self.hot_threshold = hot_threshold
        self.warm_threshold = warm_threshold
        self.cool_threshold = cool_threshold

        # Track per-parameter temperature history
        self._temperature_history = {}  # param_id -> float (current EMA temperature)
        self._grad_norm_history = {}  # param_id -> float (EMA of abs grad norm)
        self._flip_rate_history = {}  # param_id -> float (EMA of flip frequency)

    def compute_temperature(
        self,
        param_name: str,
        grad_norm: float,
        flip_rate: float,
        norm_grad: float = 1.0,
        norm_flip: float = 1.0,
    ) -> float:
        """
        Compute temperature for a parameter based on its gradient and flip activity.

        Temperature = 0.6 * normalized_grad_norm + 0.4 * normalized_flip_rate
        Higher temperature = more active = more "heat"

        Args:
            param_name: Parameter identifier for history tracking
            grad_norm: L2 norm of gradient (or running average)
            flip_rate: Frequency of ternary weight flips (0 to 1)
            norm_grad: Normalization factor for grad_norm (e.g., mean grad norm across all params)
            norm_flip: Normalization factor for flip_rate (e.g., mean flip rate)

        Returns:
            Temperature in [0, 1]: 0 = cold (inactive), 1 = hot (very active)
        """
        if norm_grad <= 0:
            norm_grad = 1.0
        if norm_flip <= 0:
            norm_flip = 1.0

        normalized_grad = min(grad_norm / norm_grad, 1.0)
        normalized_flip = min(flip_rate / norm_flip, 1.0)

        raw_temperature = 0.6 * normalized_grad + 0.4 * normalized_flip

        # Apply EMA smoothing
        if param_name in self._temperature_history:
            old_temp = self._temperature_history[param_name]
            temperature = self.ema_decay * old_temp + (1 - self.ema_decay) * raw_temperature
        else:
            temperature = raw_temperature

        self._temperature_history[param_name] = temperature
        self._grad_norm_history[param_name] = normalized_grad
        self._flip_rate_history[param_name] = normalized_flip

        return temperature

    def classify_thermal_state(self, temperature: float) -> str:
        """
        Classify a parameter's thermal state based on its temperature.

        Args:
            temperature: Temperature value in [0, 1]

        Returns:
            One of: 'HOT', 'WARM', 'COOL', 'COLD'
        """
        if temperature >= self.hot_threshold:
            return 'HOT'
        elif temperature >= self.warm_threshold:
            return 'WARM'
        elif temperature >= self.cool_threshold:
            return 'COOL'
        else:
            return 'COLD'

    def get_resource_allocation(self, thermal_state: str) -> Dict:
        """
        Return resource allocation recommendation for a thermal state.

        Allocation includes:
        - bits_per_param: Precision (higher = more bits, more memory)
        - update_frequency: How often to update (higher = more frequent)
        - cache_priority: Cache allocation preference (higher = prioritize)
        - optimizer_state: Whether to maintain full optimizer state (Adam vs SGD)

        Args:
            thermal_state: One of 'HOT', 'WARM', 'COOL', 'COLD'

        Returns:
            Dict with allocation specifications
        """
        allocations = {
            'HOT': {
                'bits_per_param': 32,  # FP32 full precision
                'update_frequency': 1.0,  # Update every step
                'cache_priority': 1.0,  # High priority in caches
                'optimizer_state': 'full_adam',  # Full Adam: m + v
                'activation_precision': 'FP32',
                'gradient_precision': 'FP32',
            },
            'WARM': {
                'bits_per_param': 16,  # FP16
                'update_frequency': 1.0,  # Update every step
                'cache_priority': 0.7,  # Medium-high priority
                'optimizer_state': 'full_adam',  # Full Adam
                'activation_precision': 'FP16',
                'gradient_precision': 'FP16',
            },
            'COOL': {
                'bits_per_param': 8,  # INT8 quantized
                'update_frequency': 0.5,  # Update every 2 steps
                'cache_priority': 0.3,  # Low priority
                'optimizer_state': 'momentum_only',  # SGD: momentum only
                'activation_precision': 'INT8',
                'gradient_precision': 'INT8',
            },
            'COLD': {
                'bits_per_param': 2,  # Packed ternary
                'update_frequency': 0.1,  # Update every 10 steps (or not at all)
                'cache_priority': 0.0,  # No cache allocation
                'optimizer_state': 'none',  # No optimizer state
                'activation_precision': 'INT2',
                'gradient_precision': 'INT2',
            },
        }

        return allocations.get(thermal_state, allocations['HOT'])

    def get_thermal_map(self) -> Dict:
        """
        Return complete thermal state map for all tracked parameters.

        Returns:
            Dict mapping param_name -> {temperature, state, allocation}
        """
        thermal_map = {}

        for param_name, temperature in self._temperature_history.items():
            state = self.classify_thermal_state(temperature)
            allocation = self.get_resource_allocation(state)

            thermal_map[param_name] = {
                'temperature': temperature,
                'state': state,
                'grad_norm': self._grad_norm_history.get(param_name, 0.0),
                'flip_rate': self._flip_rate_history.get(param_name, 0.0),
                'allocation': allocation,
            }

        return thermal_map

    def get_thermal_statistics(self) -> Dict:
        """
        Return aggregated thermal statistics across all parameters.

        Returns:
            Dict with distribution of thermal states and average temperatures
        """
        if not self._temperature_history:
            return {
                'count_hot': 0,
                'count_warm': 0,
                'count_cool': 0,
                'count_cold': 0,
                'mean_temperature': 0.0,
                'max_temperature': 0.0,
                'min_temperature': 1.0,
            }

        temps = list(self._temperature_history.values())
        states = [self.classify_thermal_state(t) for t in temps]

        return {
            'count_hot': states.count('HOT'),
            'count_warm': states.count('WARM'),
            'count_cool': states.count('COOL'),
            'count_cold': states.count('COLD'),
            'mean_temperature': sum(temps) / len(temps),
            'max_temperature': max(temps),
            'min_temperature': min(temps),
            'total_params': len(temps),
        }
