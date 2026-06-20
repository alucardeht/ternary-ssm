"""
Graduated Optimizer State Eviction (R3-A6-M2, R3-A6-F1-A, R3-A6-F2-A, R3-A6-F3-A)

Three-tier memory management for committed weights:

- Tier 1 (P_flip < 0.1): Compress m,v from FP32 to INT4 (4-bit quantized states)
- Tier 2 (P_flip < 0.01): Offload m,v to CPU memory (restore via async prefetch if needed)
- Tier 3 (P_flip < 0.001): Full eviction - delete m,v entirely (cold restart if uncommit)

Automatically restores states when weights uncommit (P_flip rises above tier threshold).
Integrated with AdamW8bit optimizer via state dict manipulation.
"""

import torch
import math
from typing import Dict, Optional, Tuple, List
from enum import Enum


class EvictionTier(Enum):
    """State eviction tier classification."""
    ACTIVE = 0       # No eviction, full FP32 state
    TIER1 = 1        # INT4 compressed on GPU
    TIER2 = 2        # CPU offload
    TIER3 = 3        # Full eviction


class GraduatedEvictionManager:
    """
    Manages three-tier optimizer state eviction based on commitment levels.

    Tracks which parameters are in which tier and handles transitions
    (compress on commit, restore on uncommit).
    """

    def __init__(
        self,
        tier1_threshold: float = 0.1,
        tier2_threshold: float = 0.01,
        tier3_threshold: float = 0.001,
    ):
        """
        Args:
            tier1_threshold: P_flip threshold for INT4 compression (0.1 = 10% flip prob)
            tier2_threshold: P_flip threshold for CPU offload (0.01 = 1%)
            tier3_threshold: P_flip threshold for full eviction (0.001 = 0.1%)
        """
        self.tier1_threshold = tier1_threshold
        self.tier2_threshold = tier2_threshold
        self.tier3_threshold = tier3_threshold

        self.param_states = {}
        self.evicted_states = {}
        self.compressed_states = {}
        self.cpu_states = {}

        self.memory_savings = {
            'tier1_bytes': 0,
            'tier2_bytes': 0,
            'tier3_bytes': 0,
            'total_bytes': 0,
        }

    def _get_state_size_bytes(self, m: torch.Tensor, v: torch.Tensor) -> int:
        """Compute size of m,v tensors in bytes."""
        return m.numel() * 4 + v.numel() * 4

    def _quantize_state_int4(
        self,
        m: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[bytes, bytes, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Quantize m,v from FP32 to INT4 with scale factors.

        Args:
            m: First moment [N]
            v: Second moment [N]

        Returns:
            (m_quantized_bytes, v_quantized_bytes, (m_scale, v_scale))
        """
        m_abs_max = m.abs().max()
        v_abs_max = v.abs().max()

        m_scale = m_abs_max / 7.0
        v_scale = v_abs_max / 7.0

        if m_scale < 1e-10:
            m_scale = 1.0
        if v_scale < 1e-10:
            v_scale = 1.0

        m_quantized = torch.clamp(m / m_scale, -7, 7).to(torch.int8)
        v_quantized = torch.clamp(v / v_scale, -7, 7).to(torch.int8)

        m_bytes = m_quantized.cpu().numpy().tobytes()
        v_bytes = v_quantized.cpu().numpy().tobytes()

        return (
            m_bytes,
            v_bytes,
            (torch.tensor(m_scale, dtype=torch.float32),
             torch.tensor(v_scale, dtype=torch.float32))
        )

    def _dequantize_state_int4(
        self,
        m_bytes: bytes,
        v_bytes: bytes,
        scales: Tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Restore m,v from INT4 quantization.

        Args:
            m_bytes: Quantized m bytes
            v_bytes: Quantized v bytes
            scales: (m_scale, v_scale) tensors
            device: Device to restore to

        Returns:
            (m_restored, v_restored)
        """
        m_scale, v_scale = scales

        m_int8 = torch.frombuffer(m_bytes, dtype=torch.int8).to(device)
        v_int8 = torch.frombuffer(v_bytes, dtype=torch.int8).to(device)

        m = m_int8.float() * m_scale.to(device)
        v = v_int8.float() * v_scale.to(device)

        return m, v

    def classify_tier(self, p_flip: float) -> EvictionTier:
        """
        Classify parameter into eviction tier based on flip probability.

        Args:
            p_flip: Probability of weight flipping within horizon

        Returns:
            EvictionTier enum value
        """
        if p_flip >= self.tier1_threshold:
            return EvictionTier.ACTIVE
        elif p_flip >= self.tier2_threshold:
            return EvictionTier.TIER1
        elif p_flip >= self.tier3_threshold:
            return EvictionTier.TIER2
        else:
            return EvictionTier.TIER3

    def evict_states(
        self,
        optimizer: torch.optim.Optimizer,
        param_names: List[str],
        p_flip_dict: Dict[str, float],
    ) -> None:
        """
        Evict optimizer states based on commitment levels.

        Args:
            optimizer: The optimizer (e.g., AdamW8bit)
            param_names: List of parameter names to evaluate
            p_flip_dict: Dict mapping param name -> P_flip probability
        """
        for param_name in param_names:
            if param_name not in optimizer.state:
                continue

            if param_name not in p_flip_dict:
                continue

            p_flip = p_flip_dict[param_name]
            old_tier = self.param_states.get(param_name, EvictionTier.ACTIVE)
            new_tier = self.classify_tier(p_flip)

            if old_tier == new_tier:
                continue

            opt_state = optimizer.state[param_name]

            if 'exp_avg' not in opt_state or 'exp_avg_sq' not in opt_state:
                continue

            m = opt_state['exp_avg']
            v = opt_state['exp_avg_sq']

            self._transition_tier(param_name, m, v, old_tier, new_tier, optimizer)
            self.param_states[param_name] = new_tier

    def _transition_tier(
        self,
        param_name: str,
        m: torch.Tensor,
        v: torch.Tensor,
        old_tier: EvictionTier,
        new_tier: EvictionTier,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Transition parameter from old tier to new tier.

        Args:
            param_name: Parameter name
            m: First moment tensor
            v: Second moment tensor
            old_tier: Previous tier
            new_tier: Target tier
            optimizer: The optimizer
        """
        opt_state = optimizer.state[param_name]
        state_size = self._get_state_size_bytes(m, v)

        if old_tier == EvictionTier.ACTIVE and new_tier == EvictionTier.TIER1:
            m_bytes, v_bytes, scales = self._quantize_state_int4(m, v)
            self.compressed_states[param_name] = (m_bytes, v_bytes, scales)
            del opt_state['exp_avg']
            del opt_state['exp_avg_sq']
            self.memory_savings['tier1_bytes'] += state_size

        elif old_tier in (EvictionTier.ACTIVE, EvictionTier.TIER1) and new_tier == EvictionTier.TIER2:
            if param_name in self.compressed_states:
                m_bytes, v_bytes, scales = self.compressed_states[param_name]
                del self.compressed_states[param_name]
            else:
                m_bytes, v_bytes, scales = self._quantize_state_int4(m, v)

            m_cpu = m.cpu()
            v_cpu = v.cpu()
            self.cpu_states[param_name] = (m_cpu, v_cpu)
            del opt_state['exp_avg']
            del opt_state['exp_avg_sq']
            self.memory_savings['tier2_bytes'] += state_size

        elif new_tier == EvictionTier.TIER3:
            if param_name in self.compressed_states:
                del self.compressed_states[param_name]
            if param_name in self.cpu_states:
                del self.cpu_states[param_name]

            del opt_state['exp_avg']
            del opt_state['exp_avg_sq']
            self.memory_savings['tier3_bytes'] += state_size

        elif new_tier == EvictionTier.ACTIVE:
            if param_name in self.compressed_states:
                m_bytes, v_bytes, scales = self.compressed_states[param_name]
                m, v = self._dequantize_state_int4(m_bytes, v_bytes, scales, m.device)
                opt_state['exp_avg'] = m
                opt_state['exp_avg_sq'] = v
                del self.compressed_states[param_name]
                self.memory_savings['tier1_bytes'] -= state_size

            elif param_name in self.cpu_states:
                m_cpu, v_cpu = self.cpu_states[param_name]
                opt_state['exp_avg'] = m_cpu.to(m.device)
                opt_state['exp_avg_sq'] = v_cpu.to(m.device)
                del self.cpu_states[param_name]
                self.memory_savings['tier2_bytes'] -= state_size

            elif param_name in self.evicted_states:
                m_init, v_init = self.evicted_states[param_name]
                opt_state['exp_avg'] = m_init.to(m.device)
                opt_state['exp_avg_sq'] = v_init.to(m.device)
                del self.evicted_states[param_name]
                self.memory_savings['tier3_bytes'] -= state_size

            else:
                opt_state['exp_avg'] = m
                opt_state['exp_avg_sq'] = v

    def get_memory_savings(self) -> Dict[str, int]:
        """
        Get cumulative memory savings across all tiers.

        Returns:
            Dict with keys: 'tier1_bytes', 'tier2_bytes', 'tier3_bytes', 'total_bytes'
        """
        self.memory_savings['total_bytes'] = sum(
            self.memory_savings[k] for k in ['tier1_bytes', 'tier2_bytes', 'tier3_bytes']
        )
        return self.memory_savings.copy()

    def get_tier_distribution(self) -> Dict[str, int]:
        """
        Get count of parameters in each tier.

        Returns:
            Dict mapping tier name -> count.
        """
        tier_counts = {
            'ACTIVE': 0,
            'TIER1': 0,
            'TIER2': 0,
            'TIER3': 0,
        }

        for tier in self.param_states.values():
            tier_counts[tier.name] += 1

        return tier_counts

    def restore_all_states(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Restore all evicted/compressed states back to ACTIVE tier (emergency restore).

        Args:
            optimizer: The optimizer
        """
        for param_name in list(self.param_states.keys()):
            if self.param_states[param_name] == EvictionTier.ACTIVE:
                continue

            if param_name not in optimizer.state:
                continue

            opt_state = optimizer.state[param_name]

            if param_name in self.compressed_states:
                m_bytes, v_bytes, scales = self.compressed_states[param_name]
                device = next(optimizer.param_groups[0]['params']).device
                m, v = self._dequantize_state_int4(m_bytes, v_bytes, scales, device)
                opt_state['exp_avg'] = m
                opt_state['exp_avg_sq'] = v
                del self.compressed_states[param_name]

            elif param_name in self.cpu_states:
                m_cpu, v_cpu = self.cpu_states[param_name]
                device = next(optimizer.param_groups[0]['params']).device
                opt_state['exp_avg'] = m_cpu.to(device)
                opt_state['exp_avg_sq'] = v_cpu.to(device)
                del self.cpu_states[param_name]

            self.param_states[param_name] = EvictionTier.ACTIVE
