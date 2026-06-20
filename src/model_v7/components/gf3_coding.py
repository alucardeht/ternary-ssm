"""GF(3) Algebraic Checkpoint Coding (R4-A4-P4).

Ternary weights as GF(3) error-correcting codewords.
Rate R=k/n gives 1/R compression + free integrity check + error correction.
group_size=128 as natural code blocks.
"""
import torch
from typing import Tuple


def gf3_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Addition in GF(3): (a + b) mod 3."""
    return (a + b) % 3

def gf3_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiplication in GF(3): (a * b) mod 3."""
    return (a * b) % 3

def gf3_neg(a: torch.Tensor) -> torch.Tensor:
    """Negation in GF(3): (3 - a) mod 3."""
    return (3 - a) % 3


class GF3RepetitionCode:
    """Simple repetition code over GF(3) for error detection/correction."""

    def __init__(self, repetition: int = 3):
        """
        Args:
            repetition: number of repetitions (3 = majority vote correction)
        """
        self.repetition = repetition

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Encode by repeating each symbol."""
        gf3 = (data.flatten().long() + 1) % 3
        return gf3.repeat(self.repetition)

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        """Decode by majority vote."""
        n = len(encoded) // self.repetition
        chunks = encoded[:n * self.repetition].reshape(self.repetition, n)
        decoded = torch.zeros(n, dtype=torch.long, device=encoded.device)
        for val in range(3):
            count = (chunks == val).sum(dim=0)
            better = count > (chunks == decoded).sum(dim=0)
            decoded[better] = val
        return decoded.float() - 1

    def detect_errors(self, encoded: torch.Tensor) -> dict:
        """Detect positions with inconsistent repetitions."""
        n = len(encoded) // self.repetition
        chunks = encoded[:n * self.repetition].reshape(self.repetition, n)
        consistent = (chunks == chunks[0:1]).all(dim=0)
        n_errors = (~consistent).sum().item()
        return {
            'n_errors': n_errors,
            'error_rate': n_errors / max(n, 1),
            'error_positions': (~consistent).nonzero(as_tuple=True)[0],
        }


class GF3ParityCode:
    """Parity code over GF(3) for lightweight error detection."""

    def __init__(self, block_size: int = 128):
        self.block_size = block_size

    def compute_parity(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute GF(3) parity per block."""
        flat = (weights.flatten().long() + 1) % 3
        n = len(flat)
        n_blocks = (n + self.block_size - 1) // self.block_size

        padded = torch.zeros(n_blocks * self.block_size, dtype=torch.long,
                           device=weights.device)
        padded[:n] = flat

        blocks = padded.reshape(n_blocks, self.block_size)
        parity = blocks.sum(dim=1) % 3
        return parity

    def verify_parity(self, weights: torch.Tensor, expected_parity: torch.Tensor) -> dict:
        """Verify weight integrity using stored parity values."""
        current_parity = self.compute_parity(weights)
        mismatches = (current_parity != expected_parity[:len(current_parity)])
        return {
            'is_valid': not mismatches.any().item(),
            'n_corrupted_blocks': mismatches.sum().item(),
            'corrupted_block_indices': mismatches.nonzero(as_tuple=True)[0].tolist(),
        }
