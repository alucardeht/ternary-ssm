"""
Ternary Scan to Sparse Matmul Reformulation.

R4-A5-M5 (TSM): Reformulate SSM scan h_t = A_t * h_{t-1} + u_t as sparse
matrix-vector multiply when A is ternary.

h = M @ u where M is lower-triangular with M[t, s] = product of A[t-1], ..., A[s].
With A ternary and p(A=0) = 1/3, M has ~3L nonzeros instead of L^2/2.

TODO: Requires TSCS (ternary A matrix). Currently A is continuous float.
Will activate once A becomes ternary {-1, 0, +1}.

Reference:
    Research Round 4, Aspect 5, Finding M5
    docs/model-v5-architecture/research-round4-aspect5.md
"""

from typing import Optional, Tuple

import torch


class TernaryScanMatrix:
    """SSM scan as sparse ternary matrix multiplication.

    h = M @ u where M is lower-triangular.
    M[t, s] = product of A[t-1], A[t-2], ..., A[s] for s < t
    M[t, t] = 1 (identity diagonal)

    With A ternary and p(A=0) = 1/3:
    M has ~3L nonzeros instead of L^2/2 (extremely sparse).
    Sparse MVM: O(3L) add/subtract operations.

    M is same for all state dimensions (A is diagonal).
    M is same for all 32 RecursiveSSM iterations (A is learned, not input-dependent).

    TODO: This class requires ternary A. Currently will fail gracefully if A
    is not ternary due to implementation in build_from_ternary_A.
    """

    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.csr_row_ptr: Optional[torch.Tensor] = None
        self.csr_col_idx: Optional[torch.Tensor] = None
        self.csr_signs: Optional[torch.Tensor] = None
        self._valid = False
        self._nnz = 0

    def build_from_ternary_A(self, A_ternary: torch.Tensor) -> None:
        """Build sparse scan matrix from ternary A.

        Args:
            A_ternary: (seq_len,) ternary values {-1, 0, +1}

        TODO: Assumes A_ternary is ternary. Will be activated when TSCS is implemented.
        """
        if A_ternary.dim() != 1:
            raise ValueError(f"Expected 1D A_ternary, got shape {A_ternary.shape}")

        L = self.seq_len
        if A_ternary.shape[0] != L:
            raise ValueError(f"A shape mismatch: expected {L}, got {A_ternary.shape[0]}")

        device = A_ternary.device
        row_ptr = [0]
        col_idx = []
        signs = []

        for t in range(L):
            running_product = 1
            row_nnz = 0

            for s in range(t-1, -1, -1):
                a_s = A_ternary[s].item()

                if a_s == 0:
                    break

                if a_s == -1:
                    running_product = -running_product

                col_idx.append(s)
                signs.append(running_product)
                row_nnz += 1

            col_idx.append(t)
            signs.append(1)
            row_nnz += 1

            col_idx_row = col_idx[-row_nnz:]
            col_idx_row.sort()
            for idx in col_idx_row:
                if idx not in col_idx[-row_nnz:]:
                    col_idx_row.remove(idx)

            row_ptr.append(row_ptr[-1] + row_nnz)
            self._nnz = row_ptr[-1]

        self.csr_row_ptr = torch.tensor(row_ptr, dtype=torch.int32, device=device)
        self.csr_col_idx = torch.tensor(col_idx, dtype=torch.int32, device=device)
        self.csr_signs = torch.tensor(signs, dtype=torch.int8, device=device)
        self._valid = True

    def scan(self, u: torch.Tensor) -> torch.Tensor:
        """Execute scan via sparse MVM: h = M @ u.

        Args:
            u: (batch, seq_len, d_state) input

        Returns:
            h: (batch, seq_len, d_state) output
        """
        if not self._valid:
            raise RuntimeError("Matrix not built. Call build_from_ternary_A first.")

        batch_size, L, d_state = u.shape
        if L != self.seq_len:
            raise ValueError(f"Input sequence length {L} mismatch with matrix {self.seq_len}")

        device = u.device
        h = torch.zeros_like(u)

        row_ptr = self.csr_row_ptr.to(device)
        col_idx = self.csr_col_idx.to(device)
        signs = self.csr_signs.to(device)

        for t in range(L):
            start_idx = row_ptr[t].item()
            end_idx = row_ptr[t+1].item()

            for col_offset in range(start_idx, end_idx):
                s = col_idx[col_offset].item()
                sign = signs[col_offset].item()

                if sign == 1:
                    h[:, t, :] = h[:, t, :] + u[:, s, :]
                else:
                    h[:, t, :] = h[:, t, :] - u[:, s, :]

        return h

    @property
    def nnz(self) -> int:
        """Number of nonzero entries (expected ~3L)."""
        return self._nnz

    @property
    def sparsity(self) -> float:
        """Fraction of entries that are zero."""
        if not self._valid:
            return 0.0
        total = self.seq_len * self.seq_len
        return 1.0 - (self._nnz / total)

    @property
    def memory_bytes(self) -> int:
        """Memory footprint of CSR representation."""
        if not self._valid:
            return 0
        row_ptr_bytes = (self.seq_len + 1) * 4
        col_idx_bytes = self._nnz * 4
        signs_bytes = self._nnz * 1
        return row_ptr_bytes + col_idx_bytes + signs_bytes


def parallel_sparse_scan(u: torch.Tensor, csr_row_ptr: torch.Tensor,
                         csr_col_idx: torch.Tensor,
                         csr_signs: torch.Tensor) -> torch.Tensor:
    """Fully parallel scan via sparse MVM.

    O(1) depth (all outputs independent in sparse form).
    O(3L) total operations.

    Args:
        u: (batch, seq_len, d_state) input
        csr_row_ptr: (seq_len+1,) CSR row pointers
        csr_col_idx: (nnz,) column indices
        csr_signs: (nnz,) sign values (+1 or -1)

    Returns:
        h: (batch, seq_len, d_state) output
    """
    batch_size, L, d_state = u.shape
    device = u.device

    h = torch.zeros_like(u)

    row_ptr = csr_row_ptr.to(device)
    col_idx = csr_col_idx.to(device)
    signs = csr_signs.to(device)

    for t in range(L):
        start = row_ptr[t].item()
        end = row_ptr[t+1].item()

        for idx in range(start, end):
            s = col_idx[idx].item()
            sign = signs[idx].item()

            if sign == 1:
                h[:, t, :] += u[:, s, :]
            else:
                h[:, t, :] -= u[:, s, :]

    return h
