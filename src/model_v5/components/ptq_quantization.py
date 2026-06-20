"""
Post-Training Quantization (PTQ) with Golden-Section Threshold Optimization

F5: REQuant-Golden-Section-Threshold

Reference: Yang et al., arXiv:2510.04044 (2025)

This module implements optimal clipping threshold (alpha) search for post-training
ternary quantization using golden-section search. The quantization error f(α)
is proven locally convex, allowing O(log n) convergence to global optimum.

Key insight: Unlike training-time threshold (which requires gradient-based updates),
PTQ operates on frozen weights and must find optimal threshold analytically.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


# Golden ratio constant (φ)
PHI = (1 + math.sqrt(5)) / 2  # ≈ 1.618033988749895


def golden_section_search(
    f,
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Tuple[float, float]:
    """
    Golden-section search for finding minimum of function f on interval [a, b].
    
    Golden-section search is an algorithm for finding the minimum of a unimodal
    function by successively narrowing the range of values inside which the
    minimum is known to exist. It achieves O(log n) convergence without requiring
    derivative information.
    
    Args:
        f: Unimodal function to minimize (scalar -> scalar)
        a: Lower bound of search interval
        b: Upper bound of search interval
        tol: Convergence tolerance (search stops when (b-a) < tol)
        max_iter: Maximum number of iterations
        
    Returns:
        Tuple of (x_min, f(x_min)) - optimal point and minimum value
        
    Complexity: O(log((b-a)/tol)) iterations, O(1) space
    
    Reference: 
        - Kiefer (1953) "Sequential Minimax Search for a Maximum"
        - Yang et al., arXiv:2510.04044 (2025) - REQuant
    """
    if a >= b:
        raise ValueError(f"Invalid interval: a={a} >= b={b}")
    
    # Initialize interior points using golden ratio
    # These points divide [a,b] into three segments with ratio φ:1:φ
    inv_phi = 1.0 / PHI  # ≈ 0.618
    
    x1 = b - inv_phi * (b - a)
    x2 = a + inv_phi * (b - a)
    
    f1 = f(x1)
    f2 = f(x2)
    
    for iteration in range(max_iter):
        # Check convergence
        if abs(b - a) < tol:
            break
            
        # Determine which subinterval contains minimum
        if f1 > f2:
            # Minimum is in [x1, b]
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + inv_phi * (b - a)
            f2 = f(x2)
        else:
            # Minimum is in [a, x2]
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - inv_phi * (b - a)
            f1 = f(x1)
    
    # Return the better point
    if f1 <= f2:
        return x1, f1
    else:
        return x2, f2


def compute_ternary_quantization_error(
    weight: torch.Tensor,
    alpha: float,
    group_size: int = 128,
    per_group: bool = True
) -> torch.Tensor:
    """
    Compute MSE quantization error for given clipping threshold alpha.
    
    Ternary quantization: q(w) = sign(w) * I(|w| > alpha)
    
    Args:
        weight: FP32 weight tensor
        alpha: Clipping threshold (all values with |w| < alpha -> 0)
        group_size: Size of quantization groups (for per-group optimization)
        per_group: If True, compute error per group; else global
        
    Returns:
        Mean squared quantization error (scalar or per-group)
        
    Mathematical formulation:
        f(α) = E[(w - Q(w, α))²]
        
        where Q(w, α) = sign(w) if |w| > α, else 0
        
    Note:
        For symmetric ternary ({-1, 0, +1}), the scaling factor is implicitly
        determined by alpha. This differs from learned scale approaches.
    """
    weight_flat = weight.view(-1)
    
    if per_group:
        # Per-group quantization error
        n_groups = weight_flat.numel() // group_size
        if weight_flat.numel() % group_size != 0:
            n_groups += 1
            
        # Pad to divisible
        padded = torch.zeros(n_groups * group_size, device=weight.device)
        padded[:weight_flat.numel()] = weight_flat
        
        # Reshape to groups
        weight_groups = padded.view(n_groups, group_size)
        
        # Compute threshold per group (using group mean as initial alpha)
        group_mean = weight_groups.abs().mean(dim=-1, keepdim=True)
        group_alpha = alpha * group_mean  # Scale alpha relative to group magnitude
        
        # Apply ternary quantization
        threshold = group_alpha.expand_as(weight_groups)
        mask = weight_groups.abs() > threshold
        quantized = torch.sign(weight_groups) * mask.float()
        
        # Scale back to preserve magnitude (ternary uses alpha as scale)
        # q = sign(w) * I(|w| > alpha) * alpha
        # This gives: ||q||_∞ = alpha, matching threshold
        quantized = quantized * group_alpha.expand_as(quantized)
        
        # Compute MSE error
        error = (weight_groups - quantized).pow(2).mean(dim=-1)
        return error.mean()  # Return scalar for optimization
        
    else:
        # Global quantization
        threshold = alpha
        mask = weight_flat.abs() > threshold
        quantized = torch.sign(weight_flat) * mask.float() * alpha
        
        # Compute MSE error
        error = (weight_flat - quantized).pow(2).mean()
        return error


def find_optimal_threshold_gss(
    weight: torch.Tensor,
    group_size: int = 128,
    per_group: bool = True,
    tol: float = 1e-6,
    max_iter: int = 50
) -> Tuple[float, float]:
    """
    Find optimal clipping threshold using golden-section search.
    
    Args:
        weight: FP32 weight tensor to quantize
        group_size: Size of quantization groups
        per_group: If True, optimize per group; else global
        tol: Convergence tolerance
        max_iter: Maximum iterations per group/globally
        
    Returns:
        Tuple of (optimal_alpha, min_error)
        
    Reference: Yang et al., arXiv:2510.04044 (2025) - REQuant
        
    Note:
        - For per-group: returns mean alpha across groups
        - Search interval: [0, max(|w|)] captures full range
    """
    weight_abs_max = weight.abs().max().item()
    
    if per_group:
        # Per-group optimization
        weight_flat = weight.view(-1)
        n_elements = weight_flat.numel()
        n_groups = n_elements // group_size
        if n_elements % group_size != 0:
            n_groups += 1
            
        # Pad to divisible
        padded = torch.zeros(n_groups * group_size, device=weight.device)
        padded[:n_elements] = weight_flat
        weight_groups = padded.view(n_groups, group_size)
        
        # Find optimal alpha for each group
        optimal_alphas = []
        min_errors = []
        
        for g in range(n_groups):
            group_weights = weight_groups[g]
            group_max = group_weights.abs().max().item()
            
            if group_max < 1e-8:
                # Skip zero groups
                continue
            
            # Define error function for this group
            group_mean = group_weights.abs().mean().item()
            
            def group_error(alpha):
                # Scale alpha relative to group magnitude
                scaled_alpha = alpha * group_mean
                if scaled_alpha < 1e-8:
                    return float('inf')
                threshold = scaled_alpha
                mask = group_weights.abs() > threshold
                quantized = torch.sign(group_weights) * mask.float() * scaled_alpha
                error = (group_weights - quantized).pow(2).mean().item()
                return error
            
            # Golden-section search in normalized space [0, 1]
            # alpha_normalized = alpha / group_mean, so search in [0, group_max/group_mean]
            search_max = min(group_max / (group_mean + 1e-8), 2.0)
            
            try:
                alpha_norm, min_err = golden_section_search(
                    group_error,
                    a=0.0,
                    b=search_max,
                    tol=tol,
                    max_iter=max_iter
                )
                optimal_alphas.append(alpha_norm * group_mean)
                min_errors.append(min_err)
            except Exception:
                # Fallback to simple threshold
                optimal_alphas.append(0.5 * group_mean)
                min_errors.append(group_error(0.5))
        
        if optimal_alphas:
            return torch.tensor(optimal_alphas).mean().item(), sum(min_errors) / len(min_errors)
        else:
            return 0.0, 0.0
            
    else:
        # Global optimization
        def global_error(alpha):
            if alpha < 1e-8:
                return float('inf')
            return compute_ternary_quantization_error(
                weight, alpha, group_size=group_size, per_group=False
            ).item()
        
        return golden_section_search(
            global_error,
            a=0.0,
            b=weight_abs_max,
            tol=tol,
            max_iter=max_iter
        )


class REQuantThresholdOptimizer:
    """
    REQuant: Post-training threshold optimizer using golden-section search.
    
    Reference: Yang et al., arXiv:2510.04044 (2025)
    
    This optimizer finds the optimal clipping threshold for frozen weights
    using golden-section search, achieving near-zero accuracy loss at 8/6-bit
    and SOTA 4-bit PTQ.
    
    Usage:
        optimizer = REQuantThresholdOptimizer()
        optimal_alpha = optimizer.find_optimal_threshold(weight)
        quantized = optimizer.quantize(weight, optimal_alpha)
        
    Note:
        - Unlike training-time methods (STE, Learnable Delta), this operates
          on frozen weights and finds threshold analytically
        - O(log n) complexity makes it efficient for large tensors
        - Can be applied per-layer or per-group for better accuracy
    """
    
    def __init__(
        self,
        group_size: int = 128,
        per_group: bool = True,
        tol: float = 1e-6,
        max_iter: int = 50
    ):
        """
        Initialize REQuant threshold optimizer.
        
        Args:
            group_size: Size of quantization groups
            per_group: If True, optimize threshold per group
            tol: Convergence tolerance for golden-section search
            max_iter: Maximum iterations per search
        """
        self.group_size = group_size
        self.per_group = per_group
        self.tol = tol
        self.max_iter = max_iter
        
    def find_optimal_threshold(
        self,
        weight: torch.Tensor
    ) -> float:
        """
        Find optimal clipping threshold for given weight tensor.
        
        Args:
            weight: FP32 weight tensor
            
        Returns:
            Optimal clipping threshold alpha
        """
        optimal_alpha, _ = find_optimal_threshold_gss(
            weight,
            group_size=self.group_size,
            per_group=self.per_group,
            tol=self.tol,
            max_iter=self.max_iter
        )
        return optimal_alpha
        
    def quantize(
        self,
        weight: torch.Tensor,
        alpha: Optional[float] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Quantize weight using optimal or provided threshold.
        
        Args:
            weight: FP32 weight tensor
            alpha: Clipping threshold (if None, find optimal)
            
        Returns:
            Tuple of (quantized_weight, alpha_used)
        """
        if alpha is None:
            alpha = self.find_optimal_threshold(weight)
            
        weight_flat = weight.view(-1)
        
        if self.per_group:
            # Per-group quantization
            n_elements = weight_flat.numel()
            n_groups = n_elements // self.group_size
            if n_elements % self.group_size != 0:
                n_groups += 1
                
            padded = torch.zeros(n_groups * self.group_size, device=weight.device)
            padded[:n_elements] = weight_flat
            weight_groups = padded.view(n_groups, self.group_size)
            
            group_mean = weight_groups.abs().mean(dim=-1, keepdim=True)
            group_alpha = alpha * group_mean
            
            threshold = group_alpha.expand_as(weight_groups)
            mask = weight_groups.abs() > threshold
            quantized = torch.sign(weight_groups) * mask.float() * group_alpha
            
            result = quantized.view_as(weight)
            if n_elements < result.numel():
                result = result[:n_elements].view_as(weight)
        else:
            # Global quantization
            threshold = alpha
            mask = weight_flat.abs() > threshold
            quantized = torch.sign(weight_flat) * mask.float() * alpha
            result = quantized.view_as(weight)
            
        return result, alpha


def requant_layer(
    layer: nn.Linear,
    group_size: int = 128,
    per_group: bool = True,
    layer_name: str = ""
) -> dict:
    """
    Apply REQuant threshold optimization to a linear layer.
    
    Args:
        layer: nn.Linear layer to quantize
        group_size: Size of quantization groups
        per_group: If True, optimize per group
        layer_name: Name for logging
        
    Returns:
        Dictionary with quantization info: alpha, error, shape
    """
    optimizer = REQuantThresholdOptimizer(
        group_size=group_size,
        per_group=per_group
    )
    
    # Get original weight
    weight = layer.weight.data
    
    # Find optimal threshold
    alpha = optimizer.find_optimal_threshold(weight)
    
    # Quantize
    quantized, _ = optimizer.quantize(weight, alpha)
    
    # Compute metrics
    mse = (weight - quantized).pow(2).mean().item()
    
    return {
        "layer_name": layer_name,
        "shape": list(weight.shape),
        "optimal_alpha": alpha,
        "mse": mse,
        "group_size": group_size,
        "per_group": per_group
    }


def requantize_model(
    model: nn.Module,
    group_size: int = 128,
    per_group: bool = True,
    layers: Optional[list] = None
) -> dict:
    """
    Apply REQuant to all linear layers in a model.
    
    Args:
        model: nn.Module to quantize
        group_size: Size of quantization groups
        per_group: If True, optimize per group
        layers: Optional list of layer names to quantize (None = all Linear)
        
    Returns:
        Dictionary mapping layer names to quantization info
    """
    results = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if layers is None or any(l in name for l in layers):
                info = requant_layer(module, group_size, per_group, name)
                results[name] = info
                
    return results
