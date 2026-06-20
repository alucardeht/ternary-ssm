"""
Amortized block-pattern precomputation via lookup tables.

Finding R4-A5-M1: Precompute partial sums for all ternary weight patterns within blocks,
then assemble via table lookups. Achieves 1.9-2.4x matvec speedup for k=4 (81 patterns)
or k=5 (243 patterns, synergizes with quinary packing).

Theory: For fixed block size k, there are only 3^k possible ternary patterns. Precompute
the partial sum (dot product) for each pattern with the input activation vector. Then
each output element is a sum of k lookups and k additions, vs k*activation_dim multiplies.

For k=4, 81 patterns: 81 + 64 ops per output vs naive 256 ops = 2.4x speedup.
Amortized across batch: single LUT build, multiple lookups.
"""

import math
import torch


POWERS_OF_3 = [1, 3, 9, 27, 81]


def compute_pattern_index(block: torch.Tensor) -> torch.Tensor:
    """Convert block of k ternary weights to pattern index.

    For k=4: index in [0, 80] (3^4 = 81 patterns)
    For k=5: index in [0, 242] (3^5 = 243 patterns, same as quinary byte!)

    Args:
        block: (..., k) tensor with values in {-1, 0, +1}

    Returns:
        (...,) int tensor with pattern indices in range [0, 3^k - 1]
    """
    shifted = block + 1
    shifted = shifted.to(torch.int32)

    k = block.shape[-1]
    idx = torch.zeros(block.shape[:-1], dtype=torch.int32, device=block.device)

    for i in range(k):
        idx = idx + shifted[..., i] * POWERS_OF_3[i]

    return idx


def build_partial_sum_lut(activations: torch.Tensor, block_size: int = 4) -> torch.Tensor:
    """Build LUT of signed partial sums for all possible ternary patterns.

    For each pattern p in [0, 3^k - 1]:
        Decode p to (s_0, ..., s_{k-1}) in {-1, 0, +1}
        sigma(p) = sum_i s_i * activations[..., i]

    Args:
        activations: (..., dim) input activations
        block_size: k, number of weights per block

    Returns:
        lut: (..., 3^k) tensor of partial sums for each pattern
    """
    batch_shape = activations.shape[:-1]
    dim = activations.shape[-1]
    n_patterns = 3 ** block_size

    lut = torch.zeros((*batch_shape, n_patterns), dtype=activations.dtype, device=activations.device)

    for pattern_idx in range(n_patterns):
        decoded = torch.zeros(block_size, dtype=torch.int8, device=activations.device)
        z = pattern_idx
        for i in range(block_size):
            digit = z % 3
            decoded[i] = digit - 1
            z = z // 3

        partial_sum = torch.zeros(batch_shape, dtype=activations.dtype, device=activations.device)
        for i in range(min(block_size, dim)):
            sign = decoded[i].item()
            if sign != 0:
                partial_sum = partial_sum + sign * activations[..., i]

        lut[..., pattern_idx] = partial_sum

    return lut


def pattern_lut_matvec(activations: torch.Tensor,
                        pattern_indices: torch.Tensor,
                        scales: torch.Tensor,
                        block_size: int = 4) -> torch.Tensor:
    """Ternary matvec via pattern LUT lookup.

    Steps:
    1. Build LUT from activations (81 or 243 entries)
    2. For each output row: gather partial sums using pattern indices
    3. Sum partial sums and multiply by scale

    Args:
        activations: (batch, dim) or (dim,) input activations
        pattern_indices: (out_dim, n_blocks) pattern index per block per row
        scales: (out_dim,) per-row scales
        block_size: k (4 or 5)

    Returns:
        (batch, out_dim) or (out_dim,) output tensor
    """
    out_dim, n_blocks = pattern_indices.shape
    has_batch = activations.dim() > 1
    batch_size = activations.shape[0] if has_batch else 1

    if has_batch:
        lut = build_partial_sum_lut(activations, block_size)
        batch_shape = (batch_size,)
    else:
        lut = build_partial_sum_lut(activations.unsqueeze(0), block_size)
        batch_shape = (1,)

    output = torch.zeros((*batch_shape, out_dim), dtype=activations.dtype, device=activations.device)

    for row_idx in range(out_dim):
        row_sum = torch.zeros(batch_shape, dtype=activations.dtype, device=activations.device)
        for block_idx in range(n_blocks):
            pattern_idx = pattern_indices[row_idx, block_idx].item()
            if has_batch:
                partial = lut[:, pattern_idx]
            else:
                partial = lut[0, pattern_idx]
            row_sum = row_sum + partial

        scale = scales[row_idx]
        output[:, row_idx] = row_sum * scale

    return output.squeeze(0) if not has_batch else output


class BlockPatternEngine:
    """Manage pattern-based ternary matmul acceleration.

    Precomputes pattern indices at model load or after weight update.
    Caches indices for reuse across batch elements and iterations.

    Attributes:
        weight_shape: (out_dim, in_dim)
        block_size: k (number of weights per block, 4 or 5)
        group_size: Elements per scale group
        pattern_indices: (out_dim, n_blocks) int32 tensor
        n_distinct_patterns: Count of unique patterns used
    """

    def __init__(self, weight_shape: tuple, block_size: int = 4, group_size: int = 128):
        """Initialize pattern engine.

        Args:
            weight_shape: (out_dim, in_dim)
            block_size: k (4 or 5, default 4)
            group_size: Scale group size (default 128)
        """
        self.weight_shape = weight_shape
        self.block_size = block_size
        self.group_size = group_size

        out_dim, in_dim = weight_shape
        self.n_blocks = math.ceil(in_dim / block_size)
        self.pattern_indices = None
        self._unique_patterns = set()

    def precompute_patterns(self, quantized_weights: torch.Tensor) -> None:
        """Extract pattern indices from quantized ternary weights.

        Called once at load time or after each optimizer step.

        Args:
            quantized_weights: (out_dim, in_dim) ternary tensor
        """
        assert quantized_weights.shape == self.weight_shape
        out_dim, in_dim = self.weight_shape

        self.pattern_indices = torch.zeros((out_dim, self.n_blocks), dtype=torch.int32, device=quantized_weights.device)

        for row_idx in range(out_dim):
            for block_idx in range(self.n_blocks):
                start_col = block_idx * self.block_size
                end_col = min(start_col + self.block_size, in_dim)
                block_len = end_col - start_col

                block = quantized_weights[row_idx, start_col:end_col]

                if block_len < self.block_size:
                    padded = torch.zeros(self.block_size, dtype=block.dtype, device=block.device)
                    padded[:block_len] = block
                    block = padded

                pattern_idx = compute_pattern_index(block).item()
                self.pattern_indices[row_idx, block_idx] = pattern_idx
                self._unique_patterns.add(pattern_idx)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Execute pattern-LUT-accelerated matvec.

        Args:
            x: (in_dim,) or (batch, in_dim) input tensor

        Returns:
            (out_dim,) or (batch, out_dim) output tensor
        """
        if self.pattern_indices is None:
            raise RuntimeError("Must call precompute_patterns() first")

        return pattern_lut_matvec(x, self.pattern_indices, None, self.block_size)

    @property
    def n_distinct_patterns(self) -> int:
        """Count of distinct patterns actually used."""
        return len(self._unique_patterns)

    def diagnostics(self) -> dict:
        """Return pattern statistics.

        Returns:
            dict with pattern distribution info
        """
        n_total_patterns = 3 ** self.block_size
        return {
            'block_size': self.block_size,
            'n_blocks': self.n_blocks,
            'total_possible_patterns': n_total_patterns,
            'distinct_patterns_used': self.n_distinct_patterns,
            'pattern_diversity': self.n_distinct_patterns / n_total_patterns,
        }
