"""
Structure-of-Arrays (SoA) memory layout for ternary weights.

Finding R4-A5-F10: SoA memory layout + aligned allocation for portable SIMD.

Stores ternary weights in two contiguous arrays (sign_array, zero_mask) with 64-byte
alignment for optimal SIMD loads and cache-line boundaries. Enables portable C kernel
implementations (Highway, xsimd, WASM, RISC-V).

Theory: Separate storage of sign and zero information improves cache behavior and
vectorization compared to per-weight struct layout. 64-byte alignment matches L1 cache
line width on modern processors.
"""

import math
import torch


def pad_to_simd_width(dim: int, width: int = 64) -> int:
    """Pad dimension to multiple of SIMD width."""
    return ((dim + width - 1) // width) * width


def aligned_alloc(size: int, alignment: int = 64) -> torch.Tensor:
    """Allocate 64-byte aligned tensor for SIMD-friendly access.

    Args:
        size: Number of elements to allocate
        alignment: Alignment boundary (default 64 bytes for cache line)

    Returns:
        torch.Tensor allocated with padding to alignment boundary
    """
    padded_size = pad_to_simd_width(size, alignment)
    tensor = torch.zeros(padded_size, dtype=torch.int8)
    return tensor


class SoATernaryStorage:
    """Store ternary weights in Structure-of-Arrays layout.

    Instead of AoS (per-weight struct with sign+zero), use two contiguous arrays:
    - sign_array: int8[] where +1/-1 for nonzero weights, 0 for zero weights
    - zero_mask: uint8[] bitmask (1 bit per weight, 1=nonzero, 0=zero)

    Both arrays are 64-byte aligned for optimal SIMD loads.

    Attributes:
        shape: Original tensor shape
        padded_shape: Shape after padding to alignment boundaries
        device: Compute device ('cpu' or 'cuda')
        sign_array: int8 tensor storing weight signs
        zero_mask: Packed uint8 bitmask storing nonzero indicators
    """

    def __init__(self, shape: tuple, device: str = 'cpu'):
        """Initialize SoA storage for given tensor shape.

        Args:
            shape: Tensor shape (e.g., (out_features, in_features))
            device: Compute device ('cpu' or 'cuda')
        """
        self.shape = shape
        self.device = device

        total_elements = math.prod(shape)
        self.padded_elements = pad_to_simd_width(total_elements, 64)

        self.padded_shape = (self.padded_elements,)

        self.sign_array = torch.zeros(self.padded_elements, dtype=torch.int8, device=device)
        self.zero_mask = torch.zeros(math.ceil(self.padded_elements / 8), dtype=torch.uint8, device=device)

    def from_ternary_tensor(self, weights: torch.Tensor) -> None:
        """Convert standard ternary tensor {-1, 0, +1} to SoA layout.

        Args:
            weights: Ternary tensor with values in {-1, 0, +1}
        """
        assert weights.shape == self.shape, f"Shape mismatch: expected {self.shape}, got {weights.shape}"

        flat_weights = weights.flatten()

        sign_values = flat_weights.sign().to(torch.int8)
        self.sign_array[:len(flat_weights)] = sign_values

        nonzero_mask = (flat_weights != 0).to(torch.uint8)
        self._pack_bitmask(nonzero_mask, self.zero_mask)

    def to_ternary_tensor(self) -> torch.Tensor:
        """Reconstruct standard ternary tensor from SoA layout.

        Returns:
            Ternary tensor with shape matching original input
        """
        nonzero_mask = self._unpack_bitmask(self.zero_mask, self.padded_elements)

        signs = self.sign_array[:math.prod(self.shape)]
        nonzero_flags = nonzero_mask[:math.prod(self.shape)]

        weights = signs * nonzero_flags
        return weights.reshape(self.shape)

    def _pack_bitmask(self, mask: torch.Tensor, packed: torch.Tensor) -> None:
        """Pack boolean mask into uint8 bitmask (8 bits per byte).

        Args:
            mask: Boolean or uint8 tensor
            packed: Output uint8 tensor to store packed bits
        """
        mask_flat = mask.flatten()
        for byte_idx in range(len(packed)):
            byte_val = 0
            for bit_idx in range(8):
                elem_idx = byte_idx * 8 + bit_idx
                if elem_idx < len(mask_flat) and mask_flat[elem_idx] > 0:
                    byte_val |= (1 << bit_idx)
            packed[byte_idx] = byte_val

    def _unpack_bitmask(self, packed: torch.Tensor, n_elements: int) -> torch.Tensor:
        """Unpack uint8 bitmask to boolean tensor.

        Args:
            packed: uint8 tensor with packed bits
            n_elements: Total number of elements to unpack

        Returns:
            Boolean tensor of shape (n_elements,)
        """
        mask = torch.zeros(n_elements, dtype=torch.uint8, device=packed.device)
        for byte_idx in range(len(packed)):
            byte_val = packed[byte_idx].item()
            for bit_idx in range(8):
                elem_idx = byte_idx * 8 + bit_idx
                if elem_idx < n_elements:
                    mask[elem_idx] = (byte_val >> bit_idx) & 1
        return mask

    @property
    def padded_shape_tuple(self) -> tuple:
        """Shape after padding to alignment boundaries."""
        return self.padded_shape

    @property
    def memory_bytes(self) -> int:
        """Total memory footprint including alignment padding.

        Returns:
            Total bytes allocated for sign_array + zero_mask
        """
        sign_bytes = self.padded_elements * 1
        zero_mask_bytes = math.ceil(self.padded_elements / 8)
        return sign_bytes + zero_mask_bytes

    @property
    def original_elements(self) -> int:
        """Number of elements in original (unpadded) tensor."""
        return math.prod(self.shape)
