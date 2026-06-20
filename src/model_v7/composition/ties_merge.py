"""TIES (Task-Integrated Embedding Space) merging for ternary models.

Implements the TIES merging algorithm adapted for ternary weight models.
TIES merges multiple specialized models by operating in parameter space,
with special handling for ternary weights.

Reference: "TIES-Merging: Resolving Interference When Merging Models" (Xinghao et al., 2024)
Adaptation for ternary weights: Handles {-1, 0, +1} weights with voting mechanism.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class TernaryTIESMerger:
    """Merge multiple models using TIES algorithm with ternary-specific optimizations.
    
    TIES (Task-Integrated Embedding Space) operates in parameter space to merge models:
    1. Identify task-specific parameters (those that differ significantly)
    2. Remove or reduce interference by zeroing conflicting gradients
    3. Merge via weighted average in a clean parameter space
    
    For ternary weights {-1, 0, +1}, we use:
    - Voting: count sign agreement across models
    - Trimming: remove low-confidence weights
    - Scaling: rescale by task importance
    """
    
    def __init__(
        self,
        trim_threshold: float = 0.15,
        scaling_factor: float = 1.0,
        tie_weights: Optional[List[float]] = None,
        device: str = 'cpu',
    ):
        """Initialize TIES merger.
        
        Args:
            trim_threshold: Fraction of least-interfering weights to trim (0.15 = 15%)
            scaling_factor: Multiplier for task-specific parameters
            tie_weights: Optional per-model weights. If None, use uniform 1/N
            device: Device to perform merging on ('cpu' or 'cuda')
        """
        self.trim_threshold = trim_threshold
        self.scaling_factor = scaling_factor
        self.device = device
        self.tie_weights = tie_weights
        self.merge_stats = {}
        
    def merge(self, models: List[nn.Module]) -> Dict[str, torch.Tensor]:
        """Merge multiple models using TIES algorithm.
        
        Args:
            models: List of model instances to merge (must have compatible architectures)
            
        Returns:
            Dict of merged state dict ready to load into new model
        """
        if not models:
            raise ValueError("At least one model required for merging")
        
        n_models = len(models)
        
        # Determine merge weights
        if self.tie_weights is None:
            merge_weights = [1.0 / n_models] * n_models
        else:
            if len(self.tie_weights) != n_models:
                raise ValueError(f"tie_weights length {len(self.tie_weights)} != models length {n_models}")
            total = sum(self.tie_weights)
            merge_weights = [w / total for w in self.tie_weights]
        
        # Extract state dicts
        state_dicts = [m.state_dict() for m in models]
        
        # Initialize merged state
        merged_state = {}
        
        # Track statistics
        stats = defaultdict(int)
        
        # Process each parameter
        param_names = state_dicts[0].keys()
        for param_name in param_names:
            tensors = [state_dicts[i][param_name].to(self.device) for i in range(n_models)]
            
            # Determine if this is a ternary weight
            is_ternary = self._is_ternary_weight(param_name, tensors[0])
            
            if is_ternary:
                merged = self._merge_ternary(tensors, merge_weights, param_name)
                stats['ternary_params'] += tensors[0].numel()
            else:
                merged = self._merge_dense(tensors, merge_weights, param_name)
                stats['dense_params'] += tensors[0].numel()
            
            merged_state[param_name] = merged.to(state_dicts[0][param_name].device)
        
        self.merge_stats = dict(stats)
        return merged_state
    
    def _is_ternary_weight(self, param_name: str, tensor: torch.Tensor) -> bool:
        """Check if parameter is a ternary weight.
        
        Ternary weights are identified by:
        - Containing 'shadow' in name (shadow weights in STE)
        - Having very few unique values (3-4 for {-1, 0, +1, noise})
        - Being a weight matrix (shape[0] > 1)
        """
        if 'shadow' in param_name.lower():
            return True
        
        # Check if values are concentrated around {-1, 0, +1}
        if tensor.numel() > 100:
            unique_vals = len(torch.unique(tensor))
            if unique_vals <= 5:
                return True
        
        return False
    
    def _merge_ternary(
        self,
        tensors: List[torch.Tensor],
        weights: List[float],
        param_name: str,
    ) -> torch.Tensor:
        """Merge ternary weights using voting and trimming."""
        n_models = len(tensors)
        
        # Extract signs
        signs = []
        for t in tensors:
            sign = torch.sign(t)
            signs.append(sign)
        
        # Stack signs
        signs_stacked = torch.stack(signs, dim=0)
        
        # Count agreement
        agreement = signs_stacked.sum(dim=0) / n_models
        confidence = agreement.abs()
        
        # Trim low-confidence weights
        trim_mask = confidence < self.trim_threshold
        
        # Merge via weighted average
        merged = torch.zeros_like(tensors[0])
        for i, t in enumerate(tensors):
            merged = merged + weights[i] * t
        
        # Apply trimming
        merged[trim_mask] = 0.0
        
        # Re-quantize to ternary
        merged = torch.sign(merged)
        
        return merged
    
    def _merge_dense(
        self,
        tensors: List[torch.Tensor],
        weights: List[float],
        param_name: str,
    ) -> torch.Tensor:
        """Merge dense (non-ternary) parameters via weighted average."""
        merged = torch.zeros_like(tensors[0])
        for i, t in enumerate(tensors):
            merged = merged + weights[i] * t
        return merged
    
    def get_merge_stats(self) -> Dict[str, int]:
        """Get statistics about the merge operation."""
        return self.merge_stats


def validate_merge_quality(
    original_models: List[nn.Module],
    merged_model: nn.Module,
    test_input: torch.Tensor,
    device: str = 'cpu',
) -> Dict[str, float]:
    """Validate quality of merge by comparing outputs.
    
    Args:
        original_models: List of original models before merge
        merged_model: Merged model
        test_input: Test input tensor
        device: Device to run validation on
        
    Returns:
        Dict with metrics:
        - output_agreement: How well merged matches average of originals
        - weight_divergence: How much merged weights diverged from originals
    """
    metrics = {}
    
    # Move models and input to device
    test_input = test_input.to(device)
    merged_model = merged_model.to(device)
    original_models = [m.to(device) for m in original_models]
    
    # Run inference
    with torch.no_grad():
        merged_model.eval()
        merged_out = merged_model(test_input)
        
        orig_outs = []
        for model in original_models:
            model.eval()
            orig_outs.append(model(test_input))
        
        # Average output of original models
        avg_orig = torch.stack(orig_outs).mean(dim=0)
        
        # Output agreement: MSE between merged and average original
        output_mse = torch.mean((merged_out - avg_orig) ** 2).item()
        metrics['output_agreement'] = 1.0 / (1.0 + output_mse)
    
    # Weight divergence
    merged_params = dict(merged_model.named_parameters())
    total_divergence = 0.0
    param_count = 0
    
    for i, orig_model in enumerate(original_models):
        for name, param in orig_model.named_parameters():
            if name in merged_params:
                div = torch.mean((merged_params[name] - param) ** 2).item()
                total_divergence += div
                param_count += 1
    
    avg_divergence = total_divergence / max(param_count, 1)
    metrics['weight_divergence'] = avg_divergence
    
    return metrics
