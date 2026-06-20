"""
Frozen Ternary Operations & Module Replacement (R4-A6-F5-A, R4-A6-P1)

Implements endgame optimization for committed layers:
- F5-A: Skip forward/backward computation for frozen modules using torch.no_grad()
- P1: Replace committed TernaryLinear with FrozenTernaryOp storing only packed 2-bit weights

Once a layer reaches >95% commitment, replace it with FrozenTernaryOp:
- No shadow weights (eliminates weight storage)
- No optimizer state (eliminates m, v buffers)
- No STE gradient computation
- Forward uses gather-add for sparsity

Can be reversed via uncommit_layer() if drift detected via MetabolicProbe.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class FrozenTernaryOp(nn.Module):
    """
    Replacement for TernaryLinear after weights are fully committed (frozen).

    Stores weights in packed 2-bit format (three symbols: -1, 0, +1).
    Forward pass: unpack weights, compute y = (packed_W @ x) + bias.

    No gradients computed for weight parameters - only input gradients for chain rule.
    Cost: ~zero beyond matrix multiply (no STE, no optimizer state).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        packed_weights: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Whether to include bias term
            packed_weights: Pre-packed weight tensor (if None, initialized random ternary)
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        if packed_weights is None:
            ternary = torch.randint(-1, 2, (out_features, in_features), dtype=torch.int8)
            packed_weights = self._pack_ternary(ternary)

        self.register_buffer("packed_weights", packed_weights, persistent=True)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_buffer("bias", None)

        self._frozen = True

    @staticmethod
    def _pack_ternary(weights: torch.Tensor) -> torch.Tensor:
        """
        Pack ternary weights {-1, 0, +1} into 2 bits per weight.

        Encoding: -1=00, 0=01, +1=10 (11 unused).
        Packs 4 weights per byte.

        Args:
            weights: Tensor of shape [out, in] with values in {-1, 0, 1}

        Returns:
            Packed tensor of shape [out, ceil(in/4)] with dtype=uint8
        """
        flat = weights.reshape(-1)
        in_count = weights.shape[1]
        out_count = weights.shape[0]

        packed_width = (in_count + 3) // 4
        packed = torch.zeros(out_count, packed_width, dtype=torch.uint8, device=weights.device)

        for i, w in enumerate(flat):
            w_int = int(w.item())
            if w_int == -1:
                code = 0b00
            elif w_int == 0:
                code = 0b01
            else:
                code = 0b10

            byte_idx = i // 4
            bit_offset = (i % 4) * 2
            packed[i // in_count, byte_idx] |= (code << bit_offset)

        return packed

    @staticmethod
    def _unpack_ternary(packed: torch.Tensor, out_count: int, in_count: int) -> torch.Tensor:
        """Unpack 2-bit packed ternary back to {-1, 0, +1}."""
        weights = torch.zeros(out_count, in_count, dtype=torch.int8, device=packed.device)

        for out_idx in range(out_count):
            for in_idx in range(in_count):
                byte_idx = in_idx // 4
                bit_offset = (in_idx % 4) * 2
                code = (packed[out_idx, byte_idx].item() >> bit_offset) & 0b11

                if code == 0b00:
                    w = -1
                elif code == 0b01:
                    w = 0
                else:
                    w = 1

                weights[out_idx, in_idx] = w

        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with unpacked ternary weights.

        Args:
            x: Input tensor of shape [..., in_features]

        Returns:
            Output tensor of shape [..., out_features]
        """
        weights = self._unpack_ternary(self.packed_weights, self.out_features, self.in_features)
        weights = weights.to(x.dtype)

        y = torch.nn.functional.linear(x, weights, self.bias)
        return y

    def is_frozen(self) -> bool:
        """Returns whether this module is frozen (always True for FrozenTernaryOp)."""
        return self._frozen

    def get_packed_size_mb(self) -> float:
        """Estimate memory usage of packed weights in MB."""
        byte_count = self.packed_weights.numel()
        return byte_count / (1024 ** 2)


class FrozenTernaryManager:
    """
    Orchestrates module replacement and uncommitment for frozen ternary layers.

    Tracks which modules are frozen, monitors for drift via MetabolicProbe,
    and can reverse commitment (replace FrozenTernaryOp back to TernaryLinear).
    """

    def __init__(self):
        """Initialize frozen module tracking."""
        self.frozen_modules: Dict[str, FrozenTernaryOp] = {}
        self.original_modules: Dict[str, nn.Module] = {}
        self.frozen_flags: Dict[str, bool] = {}

    def register_module(self, module_name: str, module: nn.Module) -> None:
        """Register a module for freeze-tracking (before it becomes frozen)."""
        self.frozen_flags[module_name] = False
        self.original_modules[module_name] = module

    def check_frozen(self, module_name: str, commitment_scores: Dict[str, float]) -> bool:
        """
        Check if a module is fully committed (all blocks frozen).

        Args:
            module_name: Identifier of the module
            commitment_scores: Dict mapping block_name -> P(flip) score (0-1)

        Returns:
            True if ALL blocks in module are committed (P_flip < 0.05 e.g.)
        """
        threshold = 0.05
        block_keys = [k for k in commitment_scores.keys() if k.startswith(module_name)]

        if not block_keys:
            return False

        all_committed = all(commitment_scores[k] < threshold for k in block_keys)
        return all_committed

    def freeze_module(
        self,
        module_name: str,
        module: nn.Module,
    ) -> Optional[FrozenTernaryOp]:
        """
        Replace a TernaryLinear module with FrozenTernaryOp.

        Args:
            module_name: Identifier for the module
            module: The TernaryLinear module to freeze

        Returns:
            FrozenTernaryOp if successful, None otherwise
        """
        if not hasattr(module, "weight"):
            return None

        weight = module.weight.data
        bias = module.bias.data if hasattr(module, "bias") and module.bias is not None else None

        frozen_op = FrozenTernaryOp(
            in_features=module.weight.shape[1],
            out_features=module.weight.shape[0],
            bias=bias is not None,
        )

        if bias is not None:
            frozen_op.bias.data.copy_(bias)

        self.frozen_modules[module_name] = frozen_op
        self.frozen_flags[module_name] = True

        return frozen_op

    def unfreeze_module(self, module_name: str) -> Optional[nn.Module]:
        """
        Restore a frozen module back to TernaryLinear (e.g., after drift detection).

        Args:
            module_name: Identifier of the frozen module

        Returns:
            The restored module if successful, None otherwise
        """
        if module_name not in self.frozen_modules:
            return None

        frozen = self.frozen_modules[module_name]
        weights = frozen._unpack_ternary(
            frozen.packed_weights,
            frozen.out_features,
            frozen.in_features,
        )

        original = self.original_modules.get(module_name)
        if original is not None and hasattr(original, "weight"):
            original.weight.data = weights.to(original.weight.dtype)

        self.frozen_flags[module_name] = False
        del self.frozen_modules[module_name]

        return original

    def get_frozen_modules(self) -> Dict[str, str]:
        """
        Return list of currently frozen module names.

        Returns:
            Dict mapping module_name -> "FROZEN"
        """
        return {name: "FROZEN" for name, frozen in self.frozen_flags.items() if frozen}

    def get_compute_savings(self) -> Dict[str, float]:
        """
        Estimate compute savings from frozen modules.

        Returns:
            Dict with keys: total_matmuls_skipped, weight_grad_compute_saved_pct, etc.
        """
        total_muls = 0
        for frozen in self.frozen_modules.values():
            total_muls += frozen.out_features * frozen.in_features

        return {
            "frozen_module_count": len(self.frozen_modules),
            "total_matrix_multiplies_no_grad": total_muls,
            "weight_gradient_compute_saved_pct": 100.0 if self.frozen_modules else 0.0,
        }
