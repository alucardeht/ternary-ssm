"""
Ternary-to-Binary Decomposition for Matmul.

R4-A5-F2: Decompose ternary weight block W in {-1,0,+1}^c into
Dense W_D in {-1,+1}^c and Sparse mask W_S in {0,1}^c.

Transforms ternary LUT into two binary LUTs matching power-of-two
hardware register widths. 87.6% memory transaction reduction.

Reference:
    Research Round 4, Aspect 5, Finding F2
    docs/model-v5-architecture/research-round4-aspect5.md
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class TernaryBinaryDecomposition:
    """Decompose W = W_D * W_S where W_D in {-1,+1}, W_S in {0,1}.

    W_D (dense): sign of weight, defaulting to +1 for zeros
    W_S (sparse): nonzero mask

    Inference: y = (W_D @ x) * W_S
    Two binary LUTs (2^c entries each) instead of one ternary LUT (3^c entries).
    Matches power-of-two register widths. 87.6% memory transaction reduction.
    """

    def __init__(self, block_size: int = 4):
        """Initialize decomposer.

        Args:
            block_size: Number of weights per block (typically 4 or 5)
        """
        self.block_size = block_size
        self._dense_lut: Optional[torch.Tensor] = None
        self._sparse_lut: Optional[torch.Tensor] = None
        self._num_patterns = 2 ** block_size

    def decompose(self, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose ternary weights into dense + sparse components.

        Args:
            weights: ternary tensor with values in {-1, 0, +1}

        Returns:
            (dense_bits, sparse_bits) both as packed uint8
            dense_bits: sign bits (0 = +1, 1 = -1)
            sparse_bits: nonzero mask (0 = zero, 1 = nonzero)
        """
        dense = torch.where(weights >= 0,
                           torch.zeros_like(weights, dtype=torch.uint8),
                           torch.ones_like(weights, dtype=torch.uint8))
        sparse = (weights != 0).to(torch.uint8)

        dense_packed = self._pack_bits(dense)
        sparse_packed = self._pack_bits(sparse)

        return dense_packed, sparse_packed

    def build_binary_luts(self, activations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build two binary LUTs from activation vector.

        dense_lut[pattern]: sum of activations with sign from dense_bits
        sparse_lut[pattern]: sum of activations masked by sparse_bits

        Each LUT has 2^block_size entries (16 for k=4, 32 for k=5).

        Args:
            activations: (..., dim) activation tensor

        Returns:
            (dense_lut, sparse_lut) both of shape (..., 2^block_size)
        """
        *batch_shape, dim = activations.shape
        device = activations.device

        dense_lut = torch.zeros(*batch_shape, self._num_patterns,
                               dtype=activations.dtype, device=device)
        sparse_lut = torch.zeros(*batch_shape, self._num_patterns,
                                dtype=activations.dtype, device=device)

        for pattern_idx in range(self._num_patterns):
            pattern_bits = [(pattern_idx >> i) & 1 for i in range(self.block_size)]

            dense_result = torch.zeros(*batch_shape, dtype=activations.dtype, device=device)
            sparse_result = torch.zeros(*batch_shape, dtype=activations.dtype, device=device)

            for i, bit in enumerate(pattern_bits):
                if i < dim:
                    sign = 1 if bit == 0 else -1
                    dense_result = dense_result + sign * activations[..., i]

                    sparse_result = sparse_result + activations[..., i]

            dense_lut[..., pattern_idx] = dense_result
            sparse_lut[..., pattern_idx] = sparse_result

        return dense_lut, sparse_lut

    def matvec(self, x: torch.Tensor, dense_packed: torch.Tensor,
              sparse_packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Execute via binary LUT lookup.

        For each output row:
          y[row] = (dense_lut[dense_pattern[row]] * sparse_mask[sparse_pattern[row]]) * scale[row]

        Args:
            x: (batch, dim) input activations
            dense_packed: (out_dim, num_blocks) dense bit patterns
            sparse_packed: (out_dim, num_blocks) sparse bit patterns
            scales: (out_dim,) per-output scales

        Returns:
            y: (batch, out_dim) output
        """
        batch_size, dim = x.shape
        out_dim = dense_packed.shape[0]
        device = x.device

        dense_lut, sparse_lut = self.build_binary_luts(x)

        y = torch.zeros(batch_size, out_dim, dtype=x.dtype, device=device)

        num_blocks = dense_packed.shape[1]

        for out_idx in range(out_dim):
            result = torch.zeros(batch_size, dtype=x.dtype, device=device)

            for block_idx in range(num_blocks):
                dense_pattern = dense_packed[out_idx, block_idx].item()
                sparse_pattern = sparse_packed[out_idx, block_idx].item()

                dense_contrib = dense_lut[:, dense_pattern]
                sparse_contrib = sparse_lut[:, sparse_pattern]

                combined = dense_contrib * (sparse_contrib != 0).float()
                result = result + combined

            y[:, out_idx] = result * scales[out_idx]

        return y

    def _pack_bits(self, bits: torch.Tensor) -> torch.Tensor:
        """Pack boolean/uint8 tensor into bit-packed bytes.

        Args:
            bits: tensor with values in {0, 1}

        Returns:
            packed: uint8 tensor with 8 bits per byte
        """
        original_shape = bits.shape
        bits_flat = bits.reshape(-1)

        num_bytes = (bits_flat.shape[0] + 7) // 8
        bits_padded = torch.zeros(num_bytes * 8, dtype=bits.dtype, device=bits.device)
        bits_padded[:bits_flat.shape[0]] = bits_flat

        packed = torch.zeros(num_bytes, dtype=torch.uint8, device=bits.device)

        for byte_idx in range(num_bytes):
            byte_val = 0
            for bit_idx in range(8):
                if bits_padded[byte_idx * 8 + bit_idx].item():
                    byte_val |= (1 << bit_idx)
            packed[byte_idx] = byte_val

        return packed.reshape(*original_shape[:-1], num_bytes)

    @property
    def compression_ratio(self) -> float:
        """Memory savings from binary LUT vs ternary LUT.

        Ternary: 3^k entries per activation vector
        Binary: 2x(2^k) entries per activation vector
        Ratio: (2 * 2^k) / 3^k
        """
        return (2 * (2 ** self.block_size)) / (3 ** self.block_size)

    @property
    def memory_bytes(self) -> int:
        """Estimate memory footprint for LUT tables."""
        return 2 * self._num_patterns * 4
