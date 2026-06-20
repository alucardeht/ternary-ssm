# Components for Model v5 architecture

# Paradigm-breaking training alternatives (no backpropagation)
from .predictive_coding import PredictiveCodingTrainer
from .belief_propagation import BeliefPropagationTrainer
from .diff_evolution import DifferentialEvolutionTrainer
from .hebbian_training import ContrastiveHebbianTrainer

# Threshold computation for ternary quantization
from .lagrangian_threshold import (
    lagrangian_threshold,
    lagrangian_threshold_function,
    compute_sparsity_from_threshold,
)

__all__ = [
    "PredictiveCodingTrainer",
    "BeliefPropagationTrainer",
    "DifferentialEvolutionTrainer",
    "ContrastiveHebbianTrainer",
    "lagrangian_threshold",
    "lagrangian_threshold_function",
    "compute_sparsity_from_threshold",
]
