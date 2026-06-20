"""
Adaptive Dataflow Selection (R4-A5-F1).

Auto-select between Activation-Persistent (AP) and Output-Persistent (OP)
dataflow patterns for ternary matmul based on layer dimensions and cache size.

AP (Activation-Persistent): optimal when input dim >> output dim.
Keep activation tiles in registers, stream weight tiles.

OP (Output-Persistent): optimal when output dim >> input dim.
Keep accumulator tiles in registers, stream activation and weight tiles.

Decision is made at model load time based on shape analysis.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Tuple


class DataflowSelector:
    """
    Select optimal dataflow pattern for each TernaryLinear layer.

    AP (Activation-Persistent):
      - Best when K (input dim) is small relative to M (output dim)
      - Minimizes activation re-reads from memory
      - Registers hold activation tile across multiple weight iterations

    OP (Output-Persistent):
      - Best when M is small relative to K
      - Minimizes output write-backs to memory
      - Registers hold accumulator tile across multiple activation iterations

    Selection based on M/K ratio and L1 cache capacity.
    """

    def __init__(self):
        """Initialize DataflowSelector."""
        self._layer_dataflows = {}

    def analyze_model(self, model: nn.Module) -> Dict[str, Dict[str, object]]:
        """
        Analyze all TernaryLinear layers and assign dataflow patterns.

        Walks the model's module tree, identifies layers with quantization
        enabled, computes their M (output_dim) and K (input_dim), and assigns
        either 'AP' or 'OP' based on the M/K ratio.

        Args:
            model: PyTorch model to analyze

        Returns:
            dict mapping layer_name -> {
                'dataflow': 'AP' | 'OP',
                'reason': str explaining decision,
                'shape': (out_dim, in_dim),
                'ratio': M/K
            }
        """
        result = {}

        for name, module in model.named_modules():
            if hasattr(module, "quantize_enabled") and module.quantize_enabled:
                if hasattr(module, "weight"):
                    out_dim, in_dim = module.weight.shape
                    ratio = out_dim / max(in_dim, 1)

                    if ratio > 2.0:
                        dataflow = "AP"
                        reason = f"Wide output (M={out_dim} >> K={in_dim})"
                    elif ratio < 0.5:
                        dataflow = "OP"
                        reason = f"Tall input (M={out_dim} << K={in_dim})"
                    else:
                        dataflow = "OP"
                        reason = f"Square-ish (M/K={ratio:.2f}, default to OP)"

                    self._layer_dataflows[name] = dataflow
                    result[name] = {
                        "dataflow": dataflow,
                        "reason": reason,
                        "shape": (out_dim, in_dim),
                        "ratio": ratio,
                    }

        return result

    def get_dataflow(self, layer_name: str) -> str:
        """
        Get assigned dataflow for a layer.

        Args:
            layer_name: module name in model

        Returns:
            'AP' or 'OP' (defaults to 'OP' if not analyzed)
        """
        return self._layer_dataflows.get(layer_name, "OP")

    def get_tile_params(
        self, layer_name: str, cache_size_kb: int = 32
    ) -> Dict[str, int]:
        """
        Compute optimal tile parameters for assigned dataflow.

        Args:
            layer_name: module name in model
            cache_size_kb: L1 cache size in KB (default: 32)

        Returns:
            dict with tile_m, tile_k for the assigned dataflow
        """
        dataflow = self.get_dataflow(layer_name)
        cache_bytes = cache_size_kb * 1024

        if dataflow == "AP":
            tile_k = min(256, cache_bytes // 8)
            tile_m = 8
        else:
            tile_m = 8
            tile_k = min(64, cache_bytes // (tile_m * 2))

        return {"tile_m": tile_m, "tile_k": tile_k}


def ap_matvec(
    x: torch.Tensor, w: torch.Tensor, tile_k: int = 64
) -> torch.Tensor:
    """
    Activation-Persistent dataflow reference implementation.

    Activation tiles stay in registers, weight tiles are streamed.
    Process input in chunks of tile_k, accumulate partial outputs.

    Args:
        x: (batch, ..., in_dim) input activations
        w: (out_dim, in_dim) ternary weight matrix
        tile_k: size of input tiles (default: 64)

    Returns:
        y: (batch, ..., out_dim) output
    """
    out_dim, in_dim = w.shape
    batch_shape = x.shape[:-1]
    y = torch.zeros(*batch_shape, out_dim, dtype=x.dtype, device=x.device)

    for k_start in range(0, in_dim, tile_k):
        k_end = min(k_start + tile_k, in_dim)
        x_tile = x[..., k_start:k_end]
        w_tile = w[:, k_start:k_end]

        y = y + torch.nn.functional.linear(x_tile, w_tile)

    return y


def op_matvec(
    x: torch.Tensor, w: torch.Tensor, tile_m: int = 8
) -> torch.Tensor:
    """
    Output-Persistent dataflow reference implementation.

    Accumulators stay in registers, activation and weight tiles stream.
    Process output in chunks of tile_m, accumulate each chunk fully.

    Args:
        x: (batch, ..., in_dim) input activations
        w: (out_dim, in_dim) ternary weight matrix
        tile_m: size of output tiles (default: 8)

    Returns:
        y: (batch, ..., out_dim) output
    """
    out_dim, in_dim = w.shape
    batch_shape = x.shape[:-1]
    y = torch.zeros(*batch_shape, out_dim, dtype=x.dtype, device=x.device)

    for m_start in range(0, out_dim, tile_m):
        m_end = min(m_start + tile_m, out_dim)
        w_tile = w[m_start:m_end, :]

        y[..., m_start:m_end] = torch.nn.functional.linear(x, w_tile)

    return y
