"""INSTANT: Low-Rank Activation Compression for Memory-Efficient Training.

Projects saved activations into a low-rank subspace before storage in autograd.
Reconstructs approximate activations during backward for gradient computation.

R2-A2-F6: INSTANT (Low-Rank Grad+Activation Compression)

Mathematical basis:
  Standard: y = x @ W^T, saves x (shape [B*L, d_in]) for backward
  INSTANT:  y = x @ W^T, saves x @ P^T (shape [B*L, rank]) for backward
            where P = top-r right singular vectors of X, updated every N steps

Memory savings: rank/d_in compression on saved activations.
Quality: bounded approximation error = sum of discarded singular values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InstantProjector:
    """SVD-based activation projector for low-rank activation compression.

    Maintains a projection matrix P that maps activations from d_in to rank.
    P is updated periodically via SVD of the activation batch.

    Similar to GaLoreProjector but applied to activations instead of gradients.

    Args:
        rank: Target rank for projection (compression dimension)
        update_gap: Recompute SVD every N forward passes (default: 50)
    """

    def __init__(self, rank: int, update_gap: int = 50):
        self.rank = rank
        self.update_gap = update_gap
        self.proj_matrix = None  # shape [rank, d_in] or [d_in, rank]
        self._call_count = 0
        self._project_right = True  # Whether to project on right (x @ P^T)
        self._debug_logged = False

    def project(self, x: torch.Tensor, step: int = None) -> torch.Tensor:
        """Project activation to low-rank subspace.

        Args:
            x: Activation tensor, shape [*, d_in] (last dim is feature dim)
            step: Current training step (used for update scheduling)

        Returns:
            Projected tensor, shape [*, rank]
        """
        d_in = x.shape[-1]
        effective_rank = min(self.rank, d_in)

        # Update projection matrix periodically
        self._call_count += 1
        if self.proj_matrix is None or self._call_count % self.update_gap == 0:
            self._update_projection(x, effective_rank)

        # Project: x @ P^T  -> shape [*, rank]
        return x @ self.proj_matrix.T

    def reconstruct(self, x_proj: torch.Tensor) -> torch.Tensor:
        """Reconstruct activation from low-rank projection.

        Args:
            x_proj: Projected tensor, shape [*, rank]

        Returns:
            Reconstructed tensor, shape [*, d_in]
        """
        # Reconstruct: x_proj @ P -> shape [*, d_in]
        return x_proj @ self.proj_matrix

    def _update_projection(self, x: torch.Tensor, effective_rank: int):
        """Recompute SVD-based projection matrix from activation sample.

        Uses the right singular vectors (V) of the activation matrix.
        V^T[:rank] captures the top-rank directions of variation.

        Args:
            x: Sample activation batch, shape [*, d_in]
            effective_rank: Number of singular vectors to keep
        """
        # Flatten to 2D for SVD: [N, d_in]
        flat = x.detach().reshape(-1, x.shape[-1]).float()

        # Subsample if too large (SVD is O(min(m,n)^2 * max(m,n)))
        max_samples = 1024
        if flat.shape[0] > max_samples:
            indices = torch.randperm(flat.shape[0], device=flat.device)[:max_samples]
            flat = flat[indices]

        try:
            # SVD: flat = U @ diag(S) @ V^T
            # V^T[:rank] = top-rank right singular vectors
            _, _, Vh = torch.linalg.svd(flat, full_matrices=False)
            self.proj_matrix = Vh[:effective_rank].to(x.dtype).to(x.device)
            # proj_matrix shape: [rank, d_in]
        except RuntimeError:
            # SVD can fail on degenerate matrices; fall back to random projection
            if self.proj_matrix is None:
                self.proj_matrix = torch.randn(
                    effective_rank, x.shape[-1],
                    device=x.device, dtype=x.dtype
                ) / math.sqrt(effective_rank)

        if not self._debug_logged:
            print(f"  [INSTANT] Projector updated: d_in={x.shape[-1]} -> rank={effective_rank}, "
                  f"proj_shape={list(self.proj_matrix.shape)}")
            self._debug_logged = True


class InstantLinearFn(torch.autograd.Function):
    """Linear layer forward/backward with low-rank activation compression.

    Forward: computes y = x @ W^T + bias exactly (unchanged output).
             Saves compressed x_proj = x @ P^T instead of full x.
    Backward: reconstructs x_approx = x_proj @ P.
              Computes grad_W = grad_y^T @ x_approx (approximate).
              Computes grad_x = grad_y @ W (exact, does not use saved x).

    R2-A2-F6
    """

    @staticmethod
    def forward(ctx, x, weight, bias, projector):
        """
        Args:
            x: Input activation (B, L, d_in)
            weight: Linear weight (d_out, d_in)
            bias: Optional bias (d_out,) or None
            projector: InstantProjector instance
        """
        # Compute exact output
        y = F.linear(x, weight, bias)

        # Project activation for compressed storage
        x_proj = projector.project(x)  # (B, L, rank)

        # Save compressed activation + weight (need W for grad_x in backward)
        ctx.save_for_backward(x_proj, weight)
        ctx.has_bias = bias is not None
        ctx.projector = projector
        ctx.x_shape = x.shape

        return y

    @staticmethod
    def backward(ctx, grad_output):
        """Compute gradients using reconstructed activation."""
        x_proj, weight = ctx.saved_tensors

        # Reconstruct activation from low-rank projection
        x_approx = ctx.projector.reconstruct(x_proj)  # (B, L, d_in)

        # grad_x = grad_output @ weight  (exact - does not use saved x)
        grad_x = grad_output @ weight  # (B, L, d_in)

        # grad_weight = grad_output^T @ x_approx  (approximate)
        # Reshape for matmul: grad_output is (B, L, d_out), x_approx is (B, L, d_in)
        go_flat = grad_output.reshape(-1, grad_output.shape[-1])  # (B*L, d_out)
        xa_flat = x_approx.reshape(-1, x_approx.shape[-1])        # (B*L, d_in)
        grad_weight = go_flat.T @ xa_flat  # (d_out, d_in)

        # grad_bias = sum of grad_output over batch and seq dims
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))

        # Return: grad for (x, weight, bias, projector)
        return grad_x, grad_weight, grad_bias, None


def instant_linear(x, weight, bias, projector):
    """Functional interface for INSTANT-compressed linear layer.

    Drop-in replacement for F.linear when INSTANT compression is active.

    Args:
        x: Input tensor (B, L, d_in)
        weight: Weight matrix (d_out, d_in)
        bias: Optional bias (d_out,) or None
        projector: InstantProjector for this layer

    Returns:
        Output tensor (B, L, d_out)
    """
    return InstantLinearFn.apply(x, weight, bias, projector)
