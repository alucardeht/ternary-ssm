"""Butterfly factorization for ternary weight matrices (R4-A5-P1).

Decompose an NxN ternary weight matrix into log2(N) sparse butterfly factors,
each with O(N) nonzero entries instead of O(N^2). Total cost: O(N log N) instead
of O(N^2) for dense matmul.

Each butterfly stage is a block-diagonal of 2x2 ternary matrices.
"""

import torch
import torch.nn as nn
import math
from typing import List, Tuple


class ButterflyFactor:
    """Single butterfly stage: block-diagonal of 2x2 ternary matrices.

    Stage k has stride 2^k. For dimension N, there are N/2 blocks per stage.
    Each block is a 2x2 ternary matrix (4 values in {-1, 0, +1}).
    """

    def __init__(self, dim: int, stage: int):
        """
        Args:
            dim: dimension of the matrix (must be power of 2)
            stage: stage index (0 = stride 1, 1 = stride 2, etc.)
        """
        self.dim = dim
        self.stage = stage
        self.stride = 2 ** stage

        n_blocks = dim // 2
        # Each block: 2x2 matrix stored as (a, b; c, d)
        self.block_values = torch.zeros(n_blocks, 2, 2, dtype=torch.int8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply butterfly stage: O(N) add/subtract operations.

        Args:
            x: (batch, ..., dim) or (dim,) tensor

        Returns:
            y: same shape as x after butterfly transformation
        """
        orig_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        else:
            x = x.reshape(-1, self.dim)

        batch = x.shape[0]
        y = x.clone()

        # For each position pair separated by stride
        for i in range(0, self.dim, 2 * self.stride):
            for j in range(i, i + self.stride):
                a_idx = j
                b_idx = j + self.stride

                # Get block index
                block_idx = i // (2 * self.stride)
                block = self.block_values[block_idx]

                # Extract values
                a = x[:, a_idx]
                b = x[:, b_idx]

                # Apply 2x2 ternary transform: [a', b'] = block @ [a, b]
                a_new = block[0, 0] * a + block[0, 1] * b
                b_new = block[1, 0] * a + block[1, 1] * b

                y[:, a_idx] = a_new
                y[:, b_idx] = b_new

        return y.reshape(orig_shape)

    def get_parameter_count(self) -> int:
        """Number of ternary parameters in this stage."""
        return (self.dim // 2) * 4


class ButterflyTernaryLayer(nn.Module):
    """Full butterfly factorization: W = B_1 @ B_2 @ ... @ B_{log2(N)}.

    For N=256: 8 stages x 256 ops = 2K ops vs 65K dense = 32x reduction.
    For N=768: ~10 stages x 768 ops = 7.7K ops vs 590K = 77x reduction.
    """

    def __init__(self, in_features: int, out_features: int):
        """
        Args:
            in_features: input dimension
            out_features: output dimension
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        max_dim = max(in_features, out_features)
        # Compute number of stages needed
        n_stages = max(1, math.ceil(math.log2(max_dim)))

        self.stages = nn.ModuleList([
            self._make_butterfly_factor(max_dim, stage)
            for stage in range(n_stages)
        ])

    def _make_butterfly_factor(self, dim: int, stage: int) -> nn.Module:
        """Create a butterfly factor as a parameterized module."""
        class ButterflyFactorModule(nn.Module):
            def __init__(self, dim: int, stage: int):
                super().__init__()
                self.dim = dim
                self.stage = stage
                self.stride = 2 ** stage
                n_blocks = dim // 2

                # Initialize as identity-like (small perturbations)
                init_values = torch.zeros(n_blocks, 2, 2, dtype=torch.float32)
                init_values[:, 0, 0] = 1.0
                init_values[:, 1, 1] = 1.0
                self.block_values = nn.Parameter(init_values)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                orig_shape = x.shape
                if x.dim() == 1:
                    x_flat = x.unsqueeze(0)
                else:
                    x_flat = x.reshape(-1, self.dim)

                batch = x_flat.shape[0]
                y = x_flat.clone()

                for i in range(0, self.dim, 2 * self.stride):
                    for j in range(i, i + self.stride):
                        a_idx = j
                        b_idx = j + self.stride

                        block_idx = i // (2 * self.stride)
                        block = self.block_values[block_idx]

                        a = x_flat[:, a_idx]
                        b = x_flat[:, b_idx]

                        a_new = block[0, 0] * a + block[0, 1] * b
                        b_new = block[1, 0] * a + block[1, 1] * b

                        y[:, a_idx] = a_new
                        y[:, b_idx] = b_new

                return y.reshape(orig_shape)

        return ButterflyFactorModule(dim, stage)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Cascade through all butterfly stages.

        Args:
            x: (batch, in_features) or similar

        Returns:
            y: (batch, out_features)
        """
        # Pad to max dimension
        max_dim = max(self.in_features, self.out_features)
        if x.shape[-1] < max_dim:
            padding = max_dim - x.shape[-1]
            x = torch.nn.functional.pad(x, (0, padding))

        # Apply cascade of butterfly stages
        for stage in self.stages:
            x = stage(x)

        # Trim to output dimension
        if x.shape[-1] > self.out_features:
            x = x[..., :self.out_features]

        return x

    def from_dense_weight(
        self,
        W: torch.Tensor,
        max_iters: int = 100,
        learning_rate: float = 0.01
    ) -> float:
        """Fit butterfly factors to approximate a dense ternary weight matrix.

        Uses alternating optimization: fix all stages except one, optimize that one.

        Args:
            W: (out_features, in_features) target weight matrix
            max_iters: maximum iterations
            learning_rate: optimization learning rate

        Returns:
            final reconstruction error (MSE)
        """
        device = W.device
        dtype = W.dtype

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for iteration in range(max_iters):
            optimizer.zero_grad()

            # Forward through butterfly
            x_eye = torch.eye(W.shape[1], device=device, dtype=dtype)
            y_reconstructed = self.forward(x_eye)

            # Trim to target shape
            y_reconstructed = y_reconstructed[:W.shape[0], :W.shape[1]]

            # Reconstruction loss
            loss = torch.nn.functional.mse_loss(y_reconstructed, W)

            loss.backward()
            optimizer.step()

            if iteration % 20 == 0:
                pass

        return loss.item()

    @property
    def total_parameters(self) -> int:
        """Total ternary parameters across all stages."""
        return sum(stage[0].dim // 2 * 4 for stage in enumerate(self.stages))

    @property
    def compression_ratio(self) -> float:
        """Ratio of butterfly params to dense params."""
        dense_params = self.in_features * self.out_features
        return self.total_parameters / max(dense_params, 1)

    def quantize_to_ternary(self) -> None:
        """Quantize block values to ternary {-1, 0, +1} via STE."""
        with torch.no_grad():
            for stage in self.stages:
                # Round to nearest ternary value
                quantized = torch.sign(stage.block_values)
                stage.block_values.copy_(quantized.float())
