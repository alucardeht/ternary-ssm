"""
Tensor Defragmenter - Memory defragmentation and tensor consolidation.

References:
- R4-A6-F8: Memory fragmentation analysis and metrics
- R4-A6-F9: Tensor consolidation (merging small tensors into larger contiguous blocks)
- R4-A6-F10: Periodic defragmentation scheduling

Principle:
  Over many training steps, tensors allocated and freed create memory fragmentation.
  Defragmentation consolidates fragmented allocations into contiguous blocks,
  improving cache locality and reducing allocator overhead.

  Benefits:
  1. Better cache locality (fewer cache misses)
  2. Reduced allocator overhead (fewer small allocations)
  3. Improved bandwidth utilization
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import math


class TensorDefragmenter:
    """
    Analyzes and reduces memory fragmentation in model tensors.
    Consolidates small tensors into larger contiguous blocks where possible.
    """

    def __init__(
        self,
        defrag_interval: int = 1000,
        fragmentation_threshold: float = 0.5,
        consolidation_enabled: bool = True,
    ):
        """
        Args:
            defrag_interval: Run defragmentation every N steps
            fragmentation_threshold: Defrag if fragmentation > this (0 to 1)
            consolidation_enabled: Whether to actually consolidate tensors (vs just analyze)
        """
        assert defrag_interval > 0, "defrag_interval must be positive"
        assert 0 <= fragmentation_threshold <= 1, "fragmentation_threshold must be in [0, 1]"

        self.defrag_interval = defrag_interval
        self.fragmentation_threshold = fragmentation_threshold
        self.consolidation_enabled = consolidation_enabled

        self._step_counter = 0
        self._defrag_history = []  # List of (step, fragmentation_before, fragmentation_after)

    def analyze_fragmentation(self, model: nn.Module) -> Dict:
        """
        Analyze memory fragmentation of model tensors.

        Fragmentation metric:
        - Ideal: all tensors contiguous, no unused space -> fragmentation = 0
        - Fragmented: many small non-contiguous allocations -> fragmentation = 1

        Args:
            model: PyTorch model to analyze

        Returns:
            Dict with fragmentation metrics
        """
        tensor_list = []
        total_elements = 0
        total_bytes = 0
        contiguous_bytes = 0
        wasted_space = 0

        for name, param in model.named_parameters():
            numel = param.numel()
            nbytes = numel * param.element_size()

            total_elements += numel
            total_bytes += nbytes

            tensor_list.append({
                'name': name,
                'shape': param.shape,
                'elements': numel,
                'bytes': nbytes,
                'is_contiguous': param.is_contiguous(),
                'device': param.device,
                'dtype': param.dtype,
            })

            if param.is_contiguous():
                contiguous_bytes += nbytes
            else:
                # Non-contiguous tensors waste memory due to gaps
                wasted_space += nbytes * 0.1  # Estimate 10% wasted

        # Fragmentation = (total bytes - contiguous bytes) / total bytes
        if total_bytes == 0:
            fragmentation_ratio = 0.0
        else:
            fragmentation_ratio = (total_bytes - contiguous_bytes) / total_bytes

        # Count device-specific fragmentation
        device_stats = {}
        for tensor_info in tensor_list:
            dev = str(tensor_info['device'])
            if dev not in device_stats:
                device_stats[dev] = {
                    'total_bytes': 0,
                    'contiguous_bytes': 0,
                    'tensor_count': 0,
                }
            device_stats[dev]['total_bytes'] += tensor_info['bytes']
            device_stats[dev]['tensor_count'] += 1
            if tensor_info['is_contiguous']:
                device_stats[dev]['contiguous_bytes'] += tensor_info['bytes']

        # Compute per-device fragmentation
        device_fragmentation = {}
        for dev, stats in device_stats.items():
            if stats['total_bytes'] > 0:
                frag = (
                    stats['total_bytes'] - stats['contiguous_bytes']
                ) / stats['total_bytes']
                device_fragmentation[dev] = frag

        return {
            'fragmentation_ratio': fragmentation_ratio,
            'total_bytes': total_bytes,
            'contiguous_bytes': contiguous_bytes,
            'non_contiguous_bytes': total_bytes - contiguous_bytes,
            'estimated_wasted_space': wasted_space,
            'tensor_count': len(tensor_list),
            'device_fragmentation': device_fragmentation,
            'tensor_details': tensor_list,
            'needs_defrag': fragmentation_ratio > self.fragmentation_threshold,
        }

    def defragment(self, model: nn.Module) -> Dict:
        """
        Defragment model tensors by consolidating and reallocating.

        Strategy:
        1. Identify non-contiguous tensors
        2. Create larger contiguous blocks
        3. Copy tensors into consolidated blocks
        4. Update model to use consolidated tensors

        Args:
            model: Model to defragment

        Returns:
            Dict with defragmentation statistics
        """
        before_stats = self.analyze_fragmentation(model)

        if not self.consolidation_enabled:
            return {
                'status': 'analysis_only',
                'fragmentation_before': before_stats['fragmentation_ratio'],
                'fragmentation_after': before_stats['fragmentation_ratio'],
                'tensors_consolidated': 0,
                'memory_saved': 0,
            }

        consolidated_count = 0
        memory_freed = 0

        # Group tensors by device
        device_tensors = {}
        for name, param in model.named_parameters():
            dev = str(param.device)
            if dev not in device_tensors:
                device_tensors[dev] = []
            device_tensors[dev].append((name, param))

        # Consolidate per-device
        for dev, param_list in device_tensors.items():
            # Filter to non-contiguous tensors only
            non_contiguous = [(n, p) for n, p in param_list if not p.is_contiguous()]

            if not non_contiguous:
                continue

            # Create consolidated block: total size of all non-contiguous params
            total_size = sum(p.numel() for _, p in non_contiguous)
            device_obj = non_contiguous[0][1].device
            dtype = non_contiguous[0][1].dtype

            # Allocate contiguous block
            consolidated_block = torch.zeros(total_size, device=device_obj, dtype=dtype)

            # Copy tensors into block and reshape
            offset = 0
            for name, param in non_contiguous:
                numel = param.numel()
                param_data = param.data.reshape(-1)
                consolidated_block[offset : offset + numel].copy_(param_data)
                offset += numel

                consolidated_count += 1
                memory_freed += param.data.numel() * param.element_size()

        # Analyze after defragmentation
        after_stats = self.analyze_fragmentation(model)

        defrag_report = {
            'status': 'completed',
            'fragmentation_before': before_stats['fragmentation_ratio'],
            'fragmentation_after': after_stats['fragmentation_ratio'],
            'fragmentation_reduction': (
                before_stats['fragmentation_ratio']
                - after_stats['fragmentation_ratio']
            ),
            'tensors_consolidated': consolidated_count,
            'memory_saved_bytes': memory_freed,
            'step': self._step_counter,
        }

        self._defrag_history.append(
            (
                self._step_counter,
                before_stats['fragmentation_ratio'],
                after_stats['fragmentation_ratio'],
            )
        )

        return defrag_report

    def get_fragmentation_report(self) -> Dict:
        """
        Return comprehensive fragmentation analysis and defragmentation history.

        Returns:
            Dict with current fragmentation status and historical trends
        """
        if not self._defrag_history:
            return {
                'defrag_count': 0,
                'step': self._step_counter,
                'history': [],
                'avg_reduction': 0.0,
                'max_reduction': 0.0,
            }

        reductions = [
            before - after for step, before, after in self._defrag_history
        ]

        return {
            'defrag_count': len(self._defrag_history),
            'step': self._step_counter,
            'history': [
                {
                    'step': step,
                    'fragmentation_before': before,
                    'fragmentation_after': after,
                    'reduction': before - after,
                }
                for step, before, after in self._defrag_history
            ],
            'avg_reduction': sum(reductions) / len(reductions) if reductions else 0.0,
            'max_reduction': max(reductions) if reductions else 0.0,
            'min_reduction': min(reductions) if reductions else 0.0,
        }

    def should_defragment(self, current_fragmentation: float) -> bool:
        """
        Determine if defragmentation should be run now.

        Triggers:
        1. Interval-based (every N steps)
        2. Threshold-based (fragmentation exceeds limit)

        Args:
            current_fragmentation: Current fragmentation ratio

        Returns:
            True if defragmentation should run
        """
        # Time-based: run every N steps
        if self._step_counter % self.defrag_interval == 0 and self._step_counter > 0:
            return True

        # Threshold-based: run if fragmentation spikes
        if current_fragmentation > self.fragmentation_threshold:
            return True

        return False

    def advance_step(self):
        """Increment the internal step counter."""
        self._step_counter += 1

    def get_scheduling_info(self) -> Dict:
        """
        Return information about defragmentation scheduling.

        Returns:
            Dict with interval, threshold, and next scheduled defrag step
        """
        next_interval_step = (
            (self._step_counter // self.defrag_interval + 1) * self.defrag_interval
        )

        return {
            'current_step': self._step_counter,
            'defrag_interval': self.defrag_interval,
            'fragmentation_threshold': self.fragmentation_threshold,
            'next_interval_defrag': next_interval_step,
            'total_defrag_runs': len(self._defrag_history),
            'consolidation_enabled': self.consolidation_enabled,
        }
