"""Shim — optim.py moved to optimizers/optim.py.

This re-exports for backward compatibility. The canonical location is src/model_v11/optimizers/optim.py
"""
from ..optimizers.optim import (
    AdamWBF16,
    Muon,
    GradientProjector,
    FiraAdamW,
    CPUOffloadOptimizer,
    AGSCPUOffloadOptimizer,
    InformationTheoreticFreezer,
    SignSGDMomentum,
    compute_layer_alphas,
    compute_lr_multipliers,
)

__all__ = [
    'AdamWBF16', 'Muon', 'GradientProjector', 'FiraAdamW',
    'CPUOffloadOptimizer', 'AGSCPUOffloadOptimizer',
    'InformationTheoreticFreezer', 'SignSGDMomentum',
    'compute_layer_alphas', 'compute_lr_multipliers',
]
