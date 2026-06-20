"""GF(3) rank factorization for ternary weight matrices (R4-A5-M4).

Decompose a ternary weight matrix W in {-1,0,+1}^{m x n} over GF(3) as
W = U @ V where U is m x r and V is r x n, both ternary. If r << min(m,n),
matvec cost reduces from O(mn) to O(r(m+n)).

GF(3) arithmetic: {-1,0,+1} maps to {0,1,2} in GF(3), where 2 ≡ -1 (mod 3).
"""

import torch
import torch.nn as nn
from typing import Tuple


def gf3_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Addition in GF(3): (a + b) mod 3, with mapping {-1,0,+1} <-> {0,1,2}."""
    # Convert to GF(3) space {0,1,2}
    a_gf3 = (a + 1) % 3
    b_gf3 = (b + 1) % 3
    result_gf3 = (a_gf3 + b_gf3) % 3
    # Convert back to {-1,0,+1}
    return (result_gf3 - 1) % 3 - 1


def gf3_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiplication in GF(3): a * b mod 3."""
    prod = a * b
    # Map: ±2 -> ∓1 (mod 3)
    return torch.where(
        prod.abs() > 1,
        -prod.sign(),
        prod
    )


def gf3_gaussian_elimination(W: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """Gaussian elimination over GF(3) to find rank and factorization.

    Args:
        W: (m, n) ternary matrix with values in {-1, 0, +1}

    Returns:
        rank: the GF(3) rank of W
        U: (m, rank) ternary matrix
        V: (rank, n) ternary matrix
    """
    m, n = W.shape
    device = W.device
    dtype = W.dtype

    # Convert to GF(3) representation {0, 1, 2}
    W_gf3 = ((W + 1) % 3).clone().float()

    # Forward elimination
    pivot_cols = []
    pivot_rows = []
    current_row = 0

    for col in range(n):
        if current_row >= m:
            break

        # Find pivot
        pivot_found = False
        for row in range(current_row, m):
            if W_gf3[row, col] != 0:
                # Swap rows
                W_gf3[[current_row, row]] = W_gf3[[row, current_row]]
                pivot_found = True
                break

        if not pivot_found:
            continue

        pivot_val = W_gf3[current_row, col]
        pivot_inv = pow(int(pivot_val), -1, 3) if pivot_val != 0 else 1

        pivot_cols.append(col)
        pivot_rows.append(current_row)

        # Normalize pivot row
        W_gf3[current_row] = (W_gf3[current_row] * pivot_inv) % 3

        # Eliminate column
        for row in range(m):
            if row != current_row and W_gf3[row, col] != 0:
                factor = W_gf3[row, col]
                W_gf3[row] = (W_gf3[row] - factor * W_gf3[current_row]) % 3

        current_row += 1

    rank = len(pivot_cols)

    # Extract U and V
    U_gf3 = W_gf3[:, :rank].clone()
    V_gf3 = W_gf3[:rank, :].clone()

    # Convert back to {-1, 0, +1}
    U = ((U_gf3 - 1) % 3 - 1).to(dtype).to(device)
    V = ((V_gf3 - 1) % 3 - 1).to(dtype).to(device)

    return rank, U, V


class GF3RankFactorization:
    """Manage GF(3) rank factorization for TernaryLinear layers.

    Post-training optimization: factorization computed once at model load.
    If rank r < mn/(m+n), factorization provides matvec speedup.
    """

    def __init__(self, gf3_min_speedup: float = 1.5):
        """
        Args:
            gf3_min_speedup: minimum speedup threshold to apply factorization
        """
        self._factored_layers = {}
        self.min_speedup = gf3_min_speedup

    def analyze_and_factorize(self, model: nn.Module) -> dict:
        """Analyze all TernaryLinear layers for GF(3) rank factorization.

        Returns dict mapping layer_name -> {
            'original_shape': (m, n),
            'gf3_rank': r,
            'dense_ops': m * n,
            'factored_ops': r * (m + n),
            'speedup': dense_ops / factored_ops,
            'worthwhile': speedup >= min_speedup,
            'U': U tensor,
            'V': V tensor
        }
        """
        results = {}

        for name, module in model.named_modules():
            # Check if this is a ternary linear layer
            if not hasattr(module, 'weight') or not hasattr(module, 'quantize_enabled'):
                continue

            m, n = module.weight.shape
            dense_ops = m * n

            # Factorize via GF(3)
            rank, U, V = gf3_gaussian_elimination(module.weight)

            factored_ops = rank * (m + n)
            speedup = dense_ops / max(factored_ops, 1)

            results[name] = {
                'original_shape': (m, n),
                'gf3_rank': rank,
                'dense_ops': dense_ops,
                'factored_ops': factored_ops,
                'speedup': speedup,
                'worthwhile': speedup >= self.min_speedup,
                'U': U,
                'V': V,
            }

            if speedup >= self.min_speedup:
                self._factored_layers[name] = (U, V)

        return results

    def factored_matvec(
        self,
        x: torch.Tensor,
        layer_name: str,
        W_original: torch.Tensor = None
    ) -> torch.Tensor:
        """Execute matvec via factored form: y = U @ (V @ x).

        Args:
            x: (batch, ..., n) input
            layer_name: name of the layer (for lookup)
            W_original: (m, n) original weight (fallback if layer not factored)

        Returns:
            y: (batch, ..., m) output
        """
        if layer_name not in self._factored_layers:
            # Fallback to dense
            if W_original is None:
                raise ValueError(f"Layer {layer_name} not factored and no fallback weight")
            return torch.nn.functional.linear(x, W_original)

        U, V = self._factored_layers[layer_name]

        # Two ternary matvecs: y = U @ (V @ x)
        intermediate = torch.nn.functional.linear(x, V)  # V: (rank, n) -> (batch, ..., rank)
        y = torch.nn.functional.linear(intermediate, U)  # U: (m, rank) -> (batch, ..., m)

        return y

    def get_compression_stats(self) -> dict:
        """Return compression statistics across factored layers."""
        if not self._factored_layers:
            return {'num_factored_layers': 0, 'total_params_saved': 0}

        total_original = sum(U.numel() + V.numel() for U, V in self._factored_layers.values())
        total_factored = sum(U.numel() + V.numel() for U, V in self._factored_layers.values())

        return {
            'num_factored_layers': len(self._factored_layers),
            'total_original_params': total_original,
            'total_factored_params': total_factored,
            'compression_ratio': total_factored / max(total_original, 1),
        }
