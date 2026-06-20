"""
Quinary (base-243) packing for ternary weights.

Finding R4-A5-M2: Pack 5 ternary weights per byte using base-243 encoding.

Encodes 5 ternary values {-1, 0, +1} per 8-bit byte using base-243 (3^5 = 243).
Achieves 1.6 bits/trit compared to 2.0 for standard packing - 20% storage reduction.

Theory: Each trit maps to {0, 1, 2} (after adding 1). Five trits form a 5-digit base-3
number with value z = sum_{i=0}^4 (w_i+1) * 3^i, range [0, 242], fits in uint8.

Synergy: Quinary byte value IS the pattern index for M1 block pattern LUT with k=5.
Matches hardware POPCOUNT/SIMD widths (powers of 2). Fits L1 cache (243 bytes per LUT).
"""

import math
import torch


POWERS_OF_3 = [1, 3, 9, 27, 81]


def build_encode_table() -> torch.Tensor:
    """Build encoding table: 5 trits -> 1 byte.

    For each combination of 5 ternary values, compute the base-243 encoding:
    z = sum_{i=0}^{4} (w_i+1) * 3^i where w_i in {-1, 0, +1}

    Returns:
        torch.Tensor of shape [3, 3, 3, 3, 3] -> [0, 242] uint8
    """
    table = torch.zeros((3, 3, 3, 3, 3), dtype=torch.uint8)
    for i0 in range(3):
        for i1 in range(3):
            for i2 in range(3):
                for i3 in range(3):
                    for i4 in range(3):
                        z = i0 * 1 + i1 * 3 + i2 * 9 + i3 * 27 + i4 * 81
                        table[i0, i1, i2, i3, i4] = z
    return table


def build_decode_table() -> torch.Tensor:
    """Build decoding table: 1 byte -> 5 trits.

    For each z in [0, 242], recover original 5 weights in {-1, 0, +1}.
    Reverse of encoding: extract base-3 digits, subtract 1.

    Returns:
        torch.Tensor of shape [243, 5] -> int8 values {-1, 0, +1}
    """
    table = torch.zeros((243, 5), dtype=torch.int8)
    for z in range(243):
        for i in range(5):
            digit = (z // POWERS_OF_3[i]) % 3
            table[z, i] = digit - 1
    return table


ENCODE_TABLE = build_encode_table()
DECODE_TABLE = build_decode_table()


def quinary_pack(weights: torch.Tensor) -> torch.Tensor:
    """Pack ternary weight tensor using 5-in-8 quinary encoding.

    Args:
        weights: Ternary tensor with values in {-1, 0, +1}

    Returns:
        Packed uint8 tensor of shape [ceil(numel / 5)]
    """
    flat_weights = weights.flatten()
    numel = flat_weights.numel()
    n_groups = math.ceil(numel / 5)
    padded_numel = n_groups * 5

    padded = torch.zeros(padded_numel, dtype=torch.int8, device=weights.device)
    padded[:numel] = flat_weights

    shifted = padded + 1
    shifted = shifted.to(torch.uint8)

    packed = torch.zeros(n_groups, dtype=torch.uint8, device=weights.device)
    for group_idx in range(n_groups):
        z = 0
        for i in range(5):
            z += shifted[group_idx * 5 + i].item() * POWERS_OF_3[i]
        packed[group_idx] = z

    return packed


def quinary_unpack(packed: torch.Tensor, original_numel: int) -> torch.Tensor:
    """Unpack quinary-packed bytes back to ternary values.

    Uses 243-entry decode LUT for fast decoding.

    Args:
        packed: uint8 tensor of packed quinary bytes
        original_numel: Number of original elements (before padding)

    Returns:
        Ternary tensor of shape [original_numel] with values in {-1, 0, +1}
    """
    decoded_flat = []
    for z in packed:
        z_val = z.item()
        decoded_row = DECODE_TABLE[z_val]
        decoded_flat.extend(decoded_row.tolist())

    result = torch.tensor(decoded_flat, dtype=torch.int8, device=packed.device)
    return result[:original_numel]


class QuinaryPackedWeights:
    """Manage quinary-packed weight storage with encode/decode.

    Maintains packed representation with optional per-group scales for reconstruction.
    Synergy with M1: packed byte value IS the pattern index for block k=5 LUT.

    Attributes:
        shape: Original weight tensor shape
        group_size: Number of elements per scale group
        packed_data: uint8 tensor of packed bytes
        group_scales: float16 tensor of per-group scales
        original_numel: Total number of elements
    """

    def __init__(self, shape: tuple, group_size: int = 128):
        """Initialize quinary packed weight storage.

        Args:
            shape: Original tensor shape
            group_size: Elements per scale group (default 128)
        """
        self.shape = shape
        self.group_size = group_size
        self.original_numel = math.prod(shape)
        self.n_groups = math.ceil(self.original_numel / 5)
        self.n_scale_groups = math.ceil(self.original_numel / group_size)

        self.packed_data = None
        self.group_scales = None

    def encode(self, weights: torch.Tensor, scales: torch.Tensor = None) -> None:
        """Pack weights and store scales.

        Args:
            weights: Ternary tensor to pack
            scales: Optional per-group scales (float16)
        """
        assert weights.shape == self.shape, f"Shape mismatch: {weights.shape} vs {self.shape}"

        self.packed_data = quinary_pack(weights)

        if scales is not None:
            assert len(scales) == self.n_scale_groups
            self.group_scales = scales.to(torch.float16)
        else:
            self.group_scales = torch.ones(self.n_scale_groups, dtype=torch.float16, device=weights.device)

    def decode(self) -> torch.Tensor:
        """Unpack to full ternary tensor.

        Returns:
            Reconstructed ternary tensor of original shape
        """
        if self.packed_data is None:
            raise RuntimeError("No packed data. Call encode() first.")

        unpacked = quinary_unpack(self.packed_data, self.original_numel)

        if self.group_scales is not None:
            unpacked_float = unpacked.to(torch.float32)
            scaled_flat = []
            for group_idx in range(self.n_scale_groups):
                start_idx = group_idx * self.group_size
                end_idx = min(start_idx + self.group_size, self.original_numel)
                group_values = unpacked_float[start_idx:end_idx]
                scale = self.group_scales[group_idx].item()
                scaled_group = group_values * scale
                scaled_flat.extend(scaled_group.tolist())
            unpacked = torch.tensor(scaled_flat, dtype=torch.float32, device=unpacked.device)

        return unpacked.reshape(self.shape)

    @property
    def compression_ratio(self) -> float:
        """Ratio vs standard 2-bit packing.

        Returns:
            Ratio (should be ~0.8, indicating 20% smaller)
        """
        packed_bits = self.n_groups * 8
        standard_bits = self.original_numel * 2
        return packed_bits / standard_bits

    @property
    def memory_bytes(self) -> int:
        """Total memory including packed data and scales.

        Returns:
            Total bytes allocated
        """
        packed_bytes = self.n_groups
        scales_bytes = self.n_scale_groups * 2 if self.group_scales is not None else 0
        return packed_bytes + scales_bytes
