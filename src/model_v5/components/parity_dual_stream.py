"""
Parity-Partitioned Dual Stream Scan Optimization.

R4-A5-M7 (PPDS): Within segments of the scan, decompose prefix sum into two
independent streams (even-parity and odd-parity positions) for parallel execution.

P_E and P_O are independent prefix sums.
Combine: h[j] = (-1)^{c_j} * (P_E[j] - P_O[j])

Benefit: two independent sums of length ~K/2 instead of one of length K.
SIMD: pack both into lanes of single register.

TODO: Requires TSCS (ternary A matrix). Blocked by cross_layer_template.py (M6).
Currently will not activate without ternary A.

Reference:
    Research Round 4, Aspect 5, Finding M7
    docs/model-v5-architecture/research-round4-aspect5.md
"""

from typing import Optional

import torch


class ParityDualStreamScan:
    """Split scan into two independent prefix sums on disjoint index sets.

    Within each segment between zeros:
    - Even-parity positions: those where cumulative A=-1 count is even
    - Odd-parity positions: those where cumulative A=-1 count is odd

    P_E and P_O are independent prefix sums.
    Combine: h[j] = (-1)^{c_j} * (P_E[j] - P_O[j])

    TODO: Requires ternary A. Will activate when TSCS is implemented.
    """

    def __init__(self, seq_len: int, d_state: int):
        self.seq_len = seq_len
        self.d_state = d_state
        self.even_indices: Optional[torch.Tensor] = None
        self.odd_indices: Optional[torch.Tensor] = None
        self._valid = False

    def build_partition(self, parity_bits: torch.Tensor,
                        segment_ids: torch.Tensor) -> None:
        """Build even/odd partition from parity bits.

        Args:
            parity_bits: (d_state, seq_len) bool from ScanTemplate
            segment_ids: (d_state, seq_len) int32 from ScanTemplate
        """
        d, L = parity_bits.shape
        if d != self.d_state or L != self.seq_len:
            raise ValueError(f"Shape mismatch: expected ({self.d_state}, {self.seq_len})")

        device = parity_bits.device

        even_indices = []
        odd_indices = []

        for state_dim in range(d):
            for t in range(L):
                is_even = not parity_bits[state_dim, t].item()
                if is_even:
                    even_indices.append(t)
                else:
                    odd_indices.append(t)

        self.even_indices = torch.tensor(even_indices, dtype=torch.int32, device=device)
        self.odd_indices = torch.tensor(odd_indices, dtype=torch.int32, device=device)
        self._valid = True

    def dual_stream_scan(self, u: torch.Tensor, parity_bits: torch.Tensor) -> torch.Tensor:
        """Execute scan via dual-stream prefix sums.

        1. Partition u into u_even, u_odd based on parity
        2. P_E = prefix_sum(u_even) per segment
        3. P_O = prefix_sum(u_odd) per segment
        4. Interleave: h[j] = parity[j] ? -(P_E[j]-P_O[j]) : (P_E[j]-P_O[j])

        Args:
            u: (batch, seq_len, d_state) input
            parity_bits: (d_state, seq_len) parity information

        Returns:
            h: (batch, seq_len, d_state) output
        """
        if not self._valid:
            raise RuntimeError("Partition not built. Call build_partition first.")

        batch_size, L, d = u.shape
        device = u.device

        h = torch.zeros_like(u)

        even_indices = self.even_indices.to(device)
        odd_indices = self.odd_indices.to(device)
        parity_bits = parity_bits.to(device)

        for state_dim in range(d):
            P_E = torch.zeros(L, dtype=u.dtype, device=device)
            P_O = torch.zeros(L, dtype=u.dtype, device=device)

            acc_e = 0.0
            acc_o = 0.0

            for t in range(L):
                is_even = not parity_bits[state_dim, t].item()

                if is_even:
                    acc_e = acc_e + u[0, t, state_dim].item()
                    P_E[t] = acc_e
                else:
                    acc_o = acc_o + u[0, t, state_dim].item()
                    P_O[t] = acc_o

            for t in range(L):
                is_even = not parity_bits[state_dim, t].item()

                if is_even:
                    delta = P_E[t] - P_O[t]
                else:
                    delta = P_E[t] - P_O[t]

                if parity_bits[state_dim, t].item():
                    h[0, t, state_dim] = -delta
                else:
                    h[0, t, state_dim] = delta

        return h

    @property
    def memory_bytes(self) -> int:
        """Memory for index arrays."""
        if not self._valid:
            return 0
        return (len(self.even_indices) + len(self.odd_indices)) * 4
