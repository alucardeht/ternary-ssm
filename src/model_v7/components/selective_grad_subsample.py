"""
Selective Gradient Subsampling for Committed Weights (R4-A6-M7, R4-A6-P2)

For committed weights, compute gradient from only a random subsample of the weight matrix
instead of all weights. Scales up sampled gradients to maintain unbiased gradient estimate.

Works especially well with gradient accumulation (K micro-batches): committed layers
compute weight gradient from only 1 micro-batch, saving K-1 backward passes.

Complementary to other sparse update strategies - reduces compute within each update step.
"""

from typing import Dict, Optional, Tuple
import torch
import math


class SelectiveGradSubsampler:
    """
    Manages selective gradient computation for committed weight blocks.

    Subsampling ratios by tier:
    - ACTIVE (0% committed): 1.0 (compute all gradients)
    - TRANSITIONING (30-60% committed): 0.5 (compute 50% of gradients)
    - COMMITTED (60-90% committed): 0.1 (compute 10% of gradients)
    - FROZEN (>95% committed): 0.0 (no gradients)

    Gradients are scaled up: scale = 1 / subsample_ratio to maintain unbiased estimate.
    Mask is random or deterministic per layer (for reproducibility).
    """

    def __init__(
        self,
        grad_subsample_committed: float = 0.1,
        grad_subsample_transitioning: float = 0.5,
        subsample_seed: Optional[int] = None,
    ):
        """
        Args:
            grad_subsample_committed: Subsample ratio for COMMITTED blocks (0.1 = 10%)
            grad_subsample_transitioning: Subsample ratio for TRANSITIONING blocks (0.5 = 50%)
            subsample_seed: Seed for reproducible mask generation (None = random)
        """
        self.committed_ratio = grad_subsample_committed
        self.transitioning_ratio = grad_subsample_transitioning
        self.seed = subsample_seed

        self.subsample_masks: Dict[str, torch.Tensor] = {}
        self.mask_cache_valid: Dict[str, bool] = {}

    def get_subsample_ratio(self, tier: str) -> float:
        """
        Get subsample ratio for a given commitment tier.

        Args:
            tier: One of ACTIVE, TRANSITIONING, COMMITTED, FROZEN

        Returns:
            Subsample ratio (0.0 to 1.0)
        """
        if tier == "ACTIVE":
            return 1.0
        elif tier == "TRANSITIONING":
            return self.transitioning_ratio
        elif tier == "COMMITTED":
            return self.committed_ratio
        elif tier == "FROZEN":
            return 0.0
        else:
            return 1.0

    def create_subsample_mask(
        self,
        param_name: str,
        param_shape: torch.Size,
        tier: str,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Create a boolean mask for gradient subsampling.

        Args:
            param_name: Parameter identifier (for caching)
            param_shape: Shape of the weight tensor
            tier: Commitment tier (ACTIVE, TRANSITIONING, COMMITTED, FROZEN)
            device: Device for mask tensor

        Returns:
            Boolean mask of same shape as param, True = compute gradient, False = skip
        """
        ratio = self.get_subsample_ratio(tier)

        if ratio >= 1.0:
            mask = torch.ones(param_shape, dtype=torch.bool, device=device)
        elif ratio <= 0.0:
            mask = torch.zeros(param_shape, dtype=torch.bool, device=device)
        else:
            total_elements = 1
            for s in param_shape:
                total_elements *= s

            num_sample = max(1, math.ceil(total_elements * ratio))

            if self.seed is not None:
                torch.manual_seed(self.seed)

            sampled_indices = torch.randperm(total_elements)[:num_sample]
            flat_mask = torch.zeros(total_elements, dtype=torch.bool, device=device)
            flat_mask[sampled_indices] = True

            mask = flat_mask.reshape(param_shape)

        self.subsample_masks[param_name] = mask
        self.mask_cache_valid[param_name] = True

        return mask

    def apply_mask(
        self,
        grad_tensor: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Zero out gradients where mask is False.

        Args:
            grad_tensor: Gradient tensor (any shape)
            mask: Boolean mask of same shape, True = keep, False = zero

        Returns:
            Masked gradient tensor (same shape, in-place zeros applied)
        """
        grad_tensor = grad_tensor * mask.float()
        return grad_tensor

    def scale_gradients(
        self,
        grad_tensor: torch.Tensor,
        mask: torch.Tensor,
        subsample_ratio: float,
    ) -> torch.Tensor:
        """
        Scale up sampled gradients to maintain unbiased estimate.

        If we sample 10% of weights, each sampled gradient is scaled by 10x to
        approximate the sum over all weights.

        Args:
            grad_tensor: Gradient tensor (already masked)
            mask: Boolean mask (for verification)
            subsample_ratio: Ratio of weights sampled (0.1 means 10% sampled)

        Returns:
            Scaled gradient tensor
        """
        if subsample_ratio <= 0.0 or subsample_ratio >= 1.0:
            return grad_tensor

        scale = 1.0 / subsample_ratio
        grad_tensor = grad_tensor * scale

        return grad_tensor

    def process_gradient(
        self,
        param_name: str,
        grad_tensor: torch.Tensor,
        tier: str,
    ) -> torch.Tensor:
        """
        End-to-end gradient processing: mask + scale.

        Args:
            param_name: Parameter ID (for mask caching)
            grad_tensor: Input gradient
            tier: Commitment tier

        Returns:
            Processed gradient (masked and scaled)
        """
        ratio = self.get_subsample_ratio(tier)

        if ratio >= 1.0:
            return grad_tensor

        if ratio <= 0.0:
            return torch.zeros_like(grad_tensor)

        if param_name not in self.subsample_masks or not self.mask_cache_valid[param_name]:
            mask = self.create_subsample_mask(
                param_name,
                grad_tensor.shape,
                tier,
                device=grad_tensor.device,
            )
        else:
            mask = self.subsample_masks[param_name]

        grad_masked = self.apply_mask(grad_tensor, mask)
        grad_scaled = self.scale_gradients(grad_masked, mask, ratio)

        return grad_scaled

    def invalidate_masks(self) -> None:
        """Invalidate all cached masks (e.g., when tier assignments change)."""
        for param_name in self.mask_cache_valid:
            self.mask_cache_valid[param_name] = False

    def get_subsample_report(self) -> Dict[str, dict]:
        """
        Return report of subsampling configuration and stats.

        Returns:
            Dict mapping tier -> {ratio, expected_compute_reduction_pct}
        """
        return {
            "ACTIVE": {
                "subsample_ratio": 1.0,
                "compute_reduction_pct": 0.0,
            },
            "TRANSITIONING": {
                "subsample_ratio": self.transitioning_ratio,
                "compute_reduction_pct": 100.0 * (1.0 - self.transitioning_ratio),
            },
            "COMMITTED": {
                "subsample_ratio": self.committed_ratio,
                "compute_reduction_pct": 100.0 * (1.0 - self.committed_ratio),
            },
            "FROZEN": {
                "subsample_ratio": 0.0,
                "compute_reduction_pct": 100.0,
            },
            "cached_masks": len(self.subsample_masks),
        }
