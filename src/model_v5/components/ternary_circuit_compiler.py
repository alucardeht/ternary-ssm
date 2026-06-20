"""
Ternary Circuit Compilation (R4-A5-P7).

Analyze the composed computation graph of ternary weights across RecursiveSSM's
32 iterations. Paths through any zero at any iteration are provably dead.
Compile an optimized execution plan eliminating all dead computation paths.

Inference-only optimization. Compiles CSR (Compressed Sparse Row) index arrays
for efficient gather-add-subtract execution.
"""

import torch
from typing import List, Dict, Tuple, Optional


class TernaryCircuitCompiler:
    """
    Compile ternary weight matrices into optimized execution plans.

    For RecursiveSSM with shared weights W:
    - W has ~33% zeros (ternary {-1,0,+1})
    - Composed: W^2 has ~55% dead paths (1 - 0.67^2)
    - W^3: ~70% dead (1 - 0.67^3)
    - Through all 32 iterations: massive dead path elimination

    Generates CSR (Compressed Sparse Row) format for efficient sparse
    gather-add-subtract execution instead of dense matmul.
    """

    def __init__(self, n_iterations: int = 32):
        """
        Initialize TernaryCircuitCompiler.

        Args:
            n_iterations: number of recursive iterations (default: 32)
        """
        self.n_iterations = n_iterations
        self._compiled_plan = None
        self._live_mask = None
        self._csr_row_ptr = None
        self._csr_col_idx = None
        self._csr_signs = None
        self._live_fraction = 0.0

    def compile(self, weight_matrices: List[torch.Tensor]) -> Dict[str, object]:
        """
        Compile execution plan from ternary weight matrices.

        Args:
            weight_matrices: list of (out, in) ternary tensors
                For RecursiveSSM: [W_in, W_out] (shared block projections)
                For single layer: [W]

        Returns:
            dict with compilation statistics:
                - live_fraction: fraction of paths that are live (0-1)
                - dead_fraction: fraction provably dead
                - csr_rows: CSR row pointers
                - csr_cols: CSR column indices
                - csr_signs: sign values for live entries (+1/-1)
                - speedup_estimate: expected speedup over dense matmul
        """
        if not weight_matrices:
            raise ValueError("weight_matrices cannot be empty")

        out_dim, in_dim = weight_matrices[-1].shape

        nonzero_masks = [
            (w != 0).float() for w in weight_matrices
        ]

        composed = nonzero_masks[0]
        for mask in nonzero_masks[1:]:
            composed = torch.mm(mask, mask)
            composed = (composed > 0).float()

        live_mask = composed.clone()
        for iteration in range(2, self.n_iterations):
            live_mask = torch.mm(live_mask, composed)
            live_mask = (live_mask > 0).float()

        self._live_mask = live_mask
        self._live_fraction = live_mask.sum().item() / (out_dim * in_dim)

        sign_matrices = [w.sign().to(torch.int8) for w in weight_matrices]
        composed_signs = sign_matrices[0]
        for sign_mat in sign_matrices[1:]:
            composed_signs = torch.mm(composed_signs, sign_mat)

        self._csr_row_ptr = []
        self._csr_col_idx = []
        self._csr_signs = []
        col_offset = 0

        for row in range(out_dim):
            self._csr_row_ptr.append(col_offset)
            for col in range(in_dim):
                if live_mask[row, col] > 0:
                    self._csr_col_idx.append(col)
                    self._csr_signs.append(
                        int(composed_signs[row, col].item())
                    )
                    col_offset += 1

        self._csr_row_ptr.append(col_offset)

        nnz = len(self._csr_col_idx)
        dead_fraction = 1.0 - self._live_fraction
        speedup_estimate = (out_dim * in_dim) / max(nnz, 1)

        self._compiled_plan = {
            "live_fraction": self._live_fraction,
            "dead_fraction": dead_fraction,
            "nnz": nnz,
            "speedup_estimate": speedup_estimate,
        }

        return self._compiled_plan

    def execute(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute compiled plan via CSR gather-add-subtract.

        Args:
            x: (batch, seq, in_features) or (in_features,) input tensor

        Returns:
            y: output after sparse matmul
        """
        if self._compiled_plan is None:
            raise RuntimeError("Must call compile() first")

        if self._csr_row_ptr is None or len(self._csr_row_ptr) == 0:
            raise RuntimeError("Compilation failed: no CSR indices generated")

        out_dim = len(self._csr_row_ptr) - 1
        input_is_1d = x.dim() == 1
        batch_shape = x.shape[:-1] if not input_is_1d else ()
        in_dim = x.shape[-1]

        x_flat = x.reshape(-1, in_dim) if not input_is_1d else x.unsqueeze(0)
        batch_size = x_flat.shape[0]

        y = torch.zeros(
            batch_size, out_dim, dtype=x.dtype, device=x.device
        )

        for row in range(out_dim):
            col_start = self._csr_row_ptr[row]
            col_end = self._csr_row_ptr[row + 1]

            for idx in range(col_start, col_end):
                col = self._csr_col_idx[idx]
                sign = self._csr_signs[idx]

                if sign > 0:
                    y[:, row] += x_flat[:, col]
                else:
                    y[:, row] -= x_flat[:, col]

        if input_is_1d:
            y = y.squeeze(0)
        elif batch_shape:
            y = y.reshape(*batch_shape, out_dim)

        return y

    def diagnostics(self) -> Dict[str, object]:
        """
        Return compilation statistics.

        Returns:
            dict with live_fraction, dead_fraction, nnz, speedup_estimate
        """
        if self._compiled_plan is None:
            return {
                "live_fraction": 0.0,
                "dead_fraction": 0.0,
                "nnz": 0,
                "speedup_estimate": 1.0,
            }
        return self._compiled_plan.copy()
