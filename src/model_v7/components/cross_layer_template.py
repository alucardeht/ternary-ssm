"""
Cross-Layer Template Caching for SSM Scan Optimization.

R4-A5-M6 (CLTC): Precompute "scan template" from ternary A matrix that encodes
all combinatorial structure, reducing each layer's scan to 3 element-wise operations.

TODO: Requires TSCS (ternary A matrix) to be implemented for full benefit.
Currently A = -exp(A_log) which is continuous float. This component will activate
once A becomes ternary {-1, 0, +1}.

Reference:
    Research Round 4, Aspect 5, Finding M6
    docs/model-v5-architecture/research-round4-aspect5.md
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class ScanTemplate:
    """Precomputed scan structure for ternary A matrix.

    Given A in {-1, 0, +1}^L per state dimension:
    - Segment boundaries at A=0 positions (reset points)
    - Parity bits: cumulative count of A=-1 positions

    Per-layer scan reduces to:
    1. y[i] = parity[i] ? -u[i] : u[i]           (conditional negate)
    2. S = segmented_prefix_sum(y)                 (standard prefix sum with resets)
    3. h[i] = parity[i] ? -S[i] : S[i]            (conditional negate)

    Total: 5L elementary ops per layer. Template computed ONCE, reused across 32 layers.
    """

    def __init__(self, seq_len: int, d_state: int):
        self.seq_len = seq_len
        self.d_state = d_state
        self.segment_ids: Optional[torch.Tensor] = None
        self.parity_bits: Optional[torch.Tensor] = None
        self._valid = False

    def build_template(self, A_ternary: torch.Tensor) -> None:
        """Build template from ternary A matrix.

        Args:
            A_ternary: (d_state,) or (d_state, seq_len) ternary values
                If 1D: same A for all timesteps (learned diagonal)
                If 2D: input-dependent A (per-timestep)

        TODO: This method assumes A is ternary {-1, 0, +1}.
        Currently A is continuous float. Will activate when TSCS is implemented.
        """
        if A_ternary.dim() == 1:
            A_ternary = A_ternary.unsqueeze(1).expand(-1, self.seq_len)

        d, L = A_ternary.shape
        if d != self.d_state or L != self.seq_len:
            raise ValueError(f"A shape {A_ternary.shape} mismatch with template {(self.d_state, self.seq_len)}")

        segment_ids = torch.zeros((d, L), dtype=torch.int32, device=A_ternary.device)
        parity_bits = torch.zeros((d, L), dtype=torch.bool, device=A_ternary.device)

        for state_dim in range(d):
            seg_id = 0
            parity = False

            for t in range(L):
                a_t = A_ternary[state_dim, t].item()

                if a_t == 0:
                    seg_id += 1
                elif a_t == -1:
                    parity = not parity

                segment_ids[state_dim, t] = seg_id
                parity_bits[state_dim, t] = parity

        self.segment_ids = segment_ids
        self.parity_bits = parity_bits
        self._valid = True

    def apply_scan(self, u: torch.Tensor) -> torch.Tensor:
        """Apply template-accelerated scan.

        Args:
            u: (batch, seq_len, d_state) input sequence

        Returns:
            h: (batch, seq_len, d_state) scan output
        """
        if not self._valid:
            raise RuntimeError("Template not built. Call build_template first.")

        batch_size, L, d = u.shape
        if d != self.d_state or L != self.seq_len:
            raise ValueError(f"Input shape {u.shape} mismatch with template")

        device = u.device
        segment_ids = self.segment_ids.to(device)
        parity_bits = self.parity_bits.to(device)

        parity_mask = parity_bits.float()

        y = torch.where(
            parity_bits.unsqueeze(0),
            -u,
            u
        )

        h = torch.zeros_like(u)

        for state_dim in range(d):
            for t in range(L):
                if t == 0:
                    acc = y[0, 0, state_dim]
                else:
                    if segment_ids[state_dim, t] == segment_ids[state_dim, t-1]:
                        acc = h[0, t-1, state_dim] + y[0, t, state_dim]
                    else:
                        acc = y[0, t, state_dim]

                h[0, t, state_dim] = acc

        h = torch.where(
            parity_bits.unsqueeze(0),
            -h,
            h
        )

        return h

    @property
    def memory_bytes(self) -> int:
        """Template memory: 2L bytes per state dimension (segment_ids + parity_bits)."""
        return 2 * self.seq_len * self.d_state


def segmented_prefix_sum(values: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
    """Compute prefix sum within segments.

    At positions where segment_id changes, reset the accumulator.
    Standard operation - maps to parallel scan with segment barriers.

    Args:
        values: (batch, seq_len, d) values to sum
        segment_ids: (d, seq_len) segment boundary markers

    Returns:
        result: (batch, seq_len, d) cumulative sums within segments
    """
    batch_size, L, d = values.shape
    result = torch.zeros_like(values)

    for state_dim in range(d):
        acc = 0.0
        for t in range(L):
            if t > 0 and segment_ids[state_dim, t] != segment_ids[state_dim, t-1]:
                acc = 0.0

            acc = acc + values[0, t, state_dim].item()
            result[0, t, state_dim] = acc

    return result
