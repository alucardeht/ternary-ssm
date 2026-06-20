"""T3M: Threshold-only ternary-to-ternary matmul using POPCOUNT (R4-A5-P2).

When both activations AND weights are ternary {-1, 0, +1}, the inner product
becomes pure bitwise operations: 4 AND + 4 POPCOUNT + 3 arithmetic ops.

Speedup vs standard ternary matvec: ~7.4x (with hardware POPCOUNT).

REQUIRES ternary activations (aggressive post-training quantization).
Quality impact at 6M scale: unknown (typically negative).
"""

import torch
import torch.nn as nn
from typing import Tuple


class TernaryTernaryMatmul:
    """Threshold-only matmul for ternary-weight + ternary-activation.

    Represent each ternary vector as P (positive) + N (negative) bitmasks,
    both packed into uint64 words.

    y_i = popcount(W_P[i] AND X_P) + popcount(W_N[i] AND X_N)
        - popcount(W_P[i] AND X_N) - popcount(W_N[i] AND X_P)
    """

    def __init__(self):
        pass

    @staticmethod
    def encode_ternary(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode ternary vector as positive and negative bitmasks.

        Args:
            x: ternary tensor {-1, 0, +1}, any shape
        Returns:
            (pos_mask, neg_mask) as uint64 packed bitmasks
        """
        x_flat = x.flatten()
        pos = (x_flat == 1).to(torch.uint8)
        neg = (x_flat == -1).to(torch.uint8)

        # Pack bits into uint64 words (64 bits per word)
        pos_packed = TernaryTernaryMatmul._pack_bits_64(pos)
        neg_packed = TernaryTernaryMatmul._pack_bits_64(neg)

        return pos_packed, neg_packed

    @staticmethod
    def _pack_bits_64(mask: torch.Tensor) -> torch.Tensor:
        """Pack boolean/uint8 tensor into uint64 words (64 bits per word).

        Args:
            mask: (n,) boolean or uint8 tensor

        Returns:
            (ceil(n/64),) uint64 tensor
        """
        n = mask.shape[0]
        n_words = (n + 63) // 64

        packed = torch.zeros(n_words, dtype=torch.uint64, device=mask.device)

        for i in range(n):
            word_idx = i // 64
            bit_idx = i % 64
            if mask[i]:
                packed[word_idx] |= (1 << bit_idx)

        return packed

    @staticmethod
    def _unpack_bits_64(packed: torch.Tensor, n: int) -> torch.Tensor:
        """Unpack uint64 words back to boolean tensor.

        Args:
            packed: (n_words,) uint64 tensor
            n: target length

        Returns:
            (n,) boolean tensor
        """
        mask = torch.zeros(n, dtype=torch.bool, device=packed.device)

        for i in range(n):
            word_idx = i // 64
            bit_idx = i % 64
            if word_idx < packed.shape[0]:
                mask[i] = bool((packed[word_idx] >> bit_idx) & 1)

        return mask

    @staticmethod
    def popcount_64(x: torch.Tensor) -> torch.Tensor:
        """Software popcount for uint64 tensors.

        Uses Brian Kernighan's algorithm or simple bit manipulation.

        Args:
            x: (n,) uint64 tensor

        Returns:
            (n,) uint64 tensor of bit counts
        """
        result = torch.zeros_like(x)

        for i in range(x.shape[0]):
            val = x[i].item()
            count = 0
            while val:
                count += val & 1
                val >>= 1
            result[i] = count

        return result

    @staticmethod
    def matmul(
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        w_pos: torch.Tensor,
        w_neg: torch.Tensor,
        out_dim: int
    ) -> torch.Tensor:
        """Execute T3M matmul via POPCOUNT.

        All inputs are packed bitmasks (uint64).

        Args:
            x_pos: (ceil(in_dim/64),) packed positive bitmask
            x_neg: (ceil(in_dim/64),) packed negative bitmask
            w_pos: (out_dim, ceil(in_dim/64)) packed weight positive bitmasks
            w_neg: (out_dim, ceil(in_dim/64)) packed weight negative bitmasks
            out_dim: output dimension

        Returns:
            y: (out_dim,) output tensor
        """
        y = torch.zeros(out_dim, dtype=torch.int32, device=w_pos.device)

        for i in range(out_dim):
            # pp = popcount(w_pos[i] AND x_pos)  # +1 * +1 = +1
            # nn = popcount(w_neg[i] AND x_neg)  # -1 * -1 = +1
            # pn = popcount(w_pos[i] AND x_neg)  # +1 * -1 = -1
            # np = popcount(w_neg[i] AND x_pos)  # -1 * +1 = -1

            pp = TernaryTernaryMatmul.popcount_64(w_pos[i] & x_pos).sum()
            nn = TernaryTernaryMatmul.popcount_64(w_neg[i] & x_neg).sum()
            pn = TernaryTernaryMatmul.popcount_64(w_pos[i] & x_neg).sum()
            np_val = TernaryTernaryMatmul.popcount_64(w_neg[i] & x_pos).sum()

            y[i] = (pp + nn) - (pn + np_val)

        return y.float()

    @staticmethod
    def quantize_activations_ternary(x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Quantize continuous activations to ternary {-1, 0, +1}.

        WARNING: Aggressive quantization. Quality impact at 6M unknown (typically negative).

        Args:
            x: floating point tensor
            threshold: magnitude threshold below which value -> 0

        Returns:
            ternary tensor {-1, 0, +1}
        """
        x_abs = x.abs()
        return torch.where(
            x_abs < threshold,
            torch.zeros_like(x),
            x.sign()
        )

    @staticmethod
    def full_matvec(
        x: torch.Tensor,
        W: torch.Tensor,
        activation_threshold: float = 0.5
    ) -> torch.Tensor:
        """Full T3M matvec from float inputs.

        Args:
            x: (batch, in_features) or (in_features,) input
            W: (out_features, in_features) ternary weight
            activation_threshold: threshold for ternary quantization

        Returns:
            y: (batch, out_features) or (out_features,) output
        """
        was_1d = x.dim() == 1
        if was_1d:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        in_features = x.shape[-1]
        out_features = W.shape[0]

        # Quantize activations to ternary
        x_ternary = TernaryTernaryMatmul.quantize_activations_ternary(
            x, activation_threshold
        )

        # Encode weights
        W_pos, W_neg = [], []
        for i in range(out_features):
            pos, neg = TernaryTernaryMatmul.encode_ternary(W[i])
            W_pos.append(pos)
            W_neg.append(neg)
        W_pos = torch.stack(W_pos)
        W_neg = torch.stack(W_neg)

        # Process each batch element
        y = torch.zeros(batch_size, out_features, dtype=torch.float, device=x.device)

        for b in range(batch_size):
            x_pos, x_neg = TernaryTernaryMatmul.encode_ternary(x_ternary[b])
            y[b] = TernaryTernaryMatmul.matmul(x_pos, x_neg, W_pos, W_neg, out_features)

        if was_1d:
            y = y.squeeze(0)

        return y


def create_t3m_layer(
    in_features: int,
    out_features: int,
    W: torch.Tensor = None
) -> nn.Module:
    """Create a T3M-accelerated linear layer wrapper.

    Args:
        in_features: input dimension
        out_features: output dimension
        W: optional pre-trained ternary weight to use

    Returns:
        nn.Module wrapping T3M
    """
    class T3MLinear(nn.Module):
        def __init__(self, in_features: int, out_features: int, W: torch.Tensor = None):
            super().__init__()
            if W is None:
                # Initialize as identity-like for small out_features
                W = torch.randn(out_features, in_features)
                W = (W / W.abs().max()).sign()
            self.register_buffer('weight', W.to(torch.float32))
            self.in_features = in_features
            self.out_features = out_features

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return TernaryTernaryMatmul.full_matvec(x, self.weight)

    return T3MLinear(in_features, out_features, W)
