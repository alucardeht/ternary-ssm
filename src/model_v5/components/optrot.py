"""OptRot - Optimal Rotation for Ternary Weight Quantization.

Learns orthogonal rotation matrices R per TernaryLinear layer that minimize
weight incoherence before quantization. Uses Cayley parameterization to
maintain orthogonality constraint.

Based on: "OptRot: Mitigating Weight Outliers via Data-Free Rotations" (2026)

Key idea:
    W_rotated = R^T @ W  where R = Cayley(S) = (I - S)(I + S)^{-1}
    Minimize ||vec(W_rotated)||_4^4 (L4 norm, proxy for max element)
    Better-distributed weights -> less quantization error -> smaller ternary gap
"""

import torch
import torch.nn as nn
import math


class BlockCayleyRotation(nn.Module):
    """Block-diagonal orthogonal rotation via Cayley parameterization.

    Instead of full NxN rotation (expensive), uses block-diagonal structure
    where each block is block_size x block_size. This reduces parameter
    overhead from O(N^2) to O(N * block_size).

    Cayley map: R = (I - S)(I + S)^{-1} where S is skew-symmetric.
    Guarantees R^T @ R = I (orthogonal) for any S.
    S initialized to zero -> R starts as identity (no rotation initially).
    """

    def __init__(self, dim: int, block_size: int = 64):
        super().__init__()
        self.dim = dim
        self.block_size = min(block_size, dim)
        self.n_blocks = math.ceil(dim / self.block_size)
        # Actual size per block (last block may be smaller)
        self.block_sizes = []
        for i in range(self.n_blocks):
            start = i * self.block_size
            end = min(start + self.block_size, dim)
            self.block_sizes.append(end - start)

        # Skew-symmetric parameters for each block
        # For a kxk skew-symmetric matrix, we need k*(k-1)/2 free parameters
        self.skew_params = nn.ParameterList()
        for bs in self.block_sizes:
            n_params = bs * (bs - 1) // 2
            param = nn.Parameter(torch.zeros(n_params))
            self.skew_params.append(param)

        # Cached rotation matrix (recomputed only when skew_params change)
        self._cached_R = None
        self._cache_dirty = True

    def _skew_symmetric(self, params, size):
        """Build skew-symmetric matrix from free parameters (vectorized)."""
        S = torch.zeros(size, size, device=params.device, dtype=params.dtype)
        triu_row, triu_col = torch.triu_indices(size, size, offset=1)
        S[triu_row, triu_col] = params
        S[triu_col, triu_row] = -params
        return S

    def _orthogonal_from_skew(self, S):
        """Exponential map: R = expm(S) for skew-symmetric S gives orthogonal R.

        Replaces Cayley map (torch.linalg.solve) which deadlocks on Windows CUDA.
        For skew-symmetric S: expm(S)^T = expm(-S) = expm(S)^{-1}, so R is orthogonal.
        """
        return torch.matrix_exp(S)

    def get_rotation(self):
        """Compute the full block-diagonal rotation matrix (cached)."""
        if self._cached_R is not None and not self._cache_dirty:
            return self._cached_R

        blocks = []
        for params, bs in zip(self.skew_params, self.block_sizes):
            if bs == 1:
                blocks.append(torch.ones(1, 1, device=params.device, dtype=params.dtype))
            else:
                S = self._skew_symmetric(params, bs)
                R_block = self._orthogonal_from_skew(S)
                blocks.append(R_block)
        self._cached_R = torch.block_diag(*blocks)
        self._cache_dirty = False
        return self._cached_R

    def invalidate_rotation_cache(self):
        """Mark rotation cache as dirty. Call after optimizer.step()."""
        self._cached_R = None
        self._cache_dirty = True

    def rotate(self, W):
        """Apply rotation: W_rotated = W @ R (rotate columns of W)."""
        R = self.get_rotation()
        return W @ R

    def rotate_inverse(self, W):
        """Apply inverse rotation: W @ R^T (since R orthogonal, R^{-1} = R^T)."""
        R = self.get_rotation()
        return W @ R.t()


def compute_l4_loss(W_rotated):
    """L4 regularization: ||vec(W)||_4^4 / ||vec(W)||_2^4

    Normalized L4 measures how "spread out" weight values are.
    Lower L4 -> more uniform distribution -> better for ternary quantization.
    """
    flat = W_rotated.flatten()
    l4 = flat.pow(4).mean()
    l2_sq = flat.pow(2).mean()
    return l4 / (l2_sq.pow(2) + 1e-8)


def enable_optrot(model, block_size=64):
    """Add OptRot rotation modules to all TernaryLinear layers.

    Args:
        model: The model containing TernaryLinear layers
        block_size: Size of rotation blocks (default 64)

    Returns:
        List of (name, rotation_module) pairs for optimizer
    """
    from .ternary import TernaryLinear
    rotations = []
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            rot = BlockCayleyRotation(module.in_features, block_size)
            rot = rot.to(module.weight.device).to(module.weight.dtype)
            module.optrot = rot
            rotations.append((name, rot))
    if rotations:
        print(f"OptRot: enabled on {len(rotations)} layers, block_size={block_size}")
    return rotations


def collect_optrot_loss(model, lambda_l4=0.01):
    """Collect L4 regularization loss from all OptRot-enabled layers.

    Call this during training to add rotation optimization to the loss.

    Args:
        model: Model with OptRot enabled
        lambda_l4: Weight for L4 regularization term

    Returns:
        Scalar loss term (0.0 if no OptRot layers)
    """
    from .ternary import TernaryLinear
    total_loss = 0.0
    count = 0
    for module in model.modules():
        if isinstance(module, TernaryLinear) and hasattr(module, 'optrot'):
            W_rot = module.optrot.rotate(module.weight.data)
            total_loss = total_loss + compute_l4_loss(W_rot)
            count += 1
    if count > 0:
        return lambda_l4 * total_loss / count
    return torch.tensor(0.0)


def bake_optrot(model):
    """Fuse rotations into weights for inference (zero runtime cost).

    After training, applies R to weights permanently and removes rotation modules.
    Call this before inference/export.
    """
    from .ternary import TernaryLinear
    count = 0
    for module in model.modules():
        if isinstance(module, TernaryLinear) and hasattr(module, 'optrot'):
            with torch.no_grad():
                W_rot = module.optrot.rotate(module.weight.data)
                module.weight.data.copy_(W_rot)
                if hasattr(module, 'shadow_weight') and module.shadow_weight is not None:
                    shadow_rot = module.optrot.rotate(module.shadow_weight.data)
                    module.shadow_weight.data.copy_(shadow_rot)
            delattr(module, 'optrot')
            count += 1
    if count > 0:
        print(f"OptRot: baked rotations into {count} layers (zero runtime cost)")
