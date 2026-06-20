"""Distributed training components for ternary neural networks.

Tier 2-C: Local SGD with majority vote synchronization for ternary weights,
and gradient compression with error feedback for efficient communication.

Key insight: Ternary weights {-1,0,+1} change slowly and discretely,
enabling much larger synchronization windows (H=50-500) compared to FP16 (H=1-10).

Tier 3-D: Distributed extensions for heterogeneous scheduling, NVMe offloading,
optimal synchronization, elastic checkpointing, and load balancing.
"""

from .local_sgd import TernaryLocalSGD
from .gradient_compress import TernaryGradientCompressor
from .extensions import (
    HeterogeneousScheduler,
    NVMeOffloader,
    OptimalSyncSchedule,
    ElasticCheckpointer,
    LoadBalancer,
)

__all__ = [
    'TernaryLocalSGD',
    'TernaryGradientCompressor',
    'HeterogeneousScheduler',
    'NVMeOffloader',
    'OptimalSyncSchedule',
    'ElasticCheckpointer',
    'LoadBalancer',
]
