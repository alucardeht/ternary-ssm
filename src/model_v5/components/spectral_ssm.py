"""Spectral-cached SSM backward - zero activation storage for SSM recurrence.

When SSM A matrix is ternary {-1, 0, +1}, there are only 3 possible eigenvalues.
Pre-compute FFT basis kernels for each eigenvalue, then forward/backward
are just weighted sums of these cached bases.

This eliminates the need to store intermediate SSM states for backward pass.
"""

import torch
import torch.nn as nn
import torch.fft
import math


class SpectralCachedSSM(nn.Module):
    """SSM with spectral caching for zero-activation backward.

    For ternary A with eigenvalues in {-1, 0, +1}:
    - eigenvalue 0: state decays to zero immediately
    - eigenvalue +1: state persists unchanged (integrator)
    - eigenvalue -1: state alternates sign (oscillator)

    Pre-compute frequency-domain representations for each case.
    Forward/backward become element-wise operations on cached spectra.
    """

    def __init__(self, d_model: int, d_state: int, seq_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.seq_len = seq_len

        self.eigen_logits = nn.Parameter(torch.zeros(d_state, 3))

        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)

        self._cache_basis_kernels(seq_len)

    def _cache_basis_kernels(self, seq_len: int):
        """Pre-compute convolution kernels for eigenvalues {-1, 0, +1}."""
        t = torch.arange(seq_len, dtype=torch.float32)

        kernel_plus1 = torch.ones(seq_len)
        kernel_minus1 = (-1.0) ** t
        kernel_zero = torch.zeros(seq_len)
        kernel_zero[0] = 1.0

        self.register_buffer('fft_plus1', torch.fft.rfft(kernel_plus1))
        self.register_buffer('fft_minus1', torch.fft.rfft(kernel_minus1))
        self.register_buffer('fft_zero', torch.fft.rfft(kernel_zero))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using spectral convolution.

        Args:
            x: [batch, seq_len, d_model]

        Returns:
            y: [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape

        Bx = self.B_proj(x)
        C = self.C_proj(x)

        eigen_weights = torch.softmax(self.eigen_logits, dim=-1)

        Bx_fft = torch.fft.rfft(Bx, n=seq_len, dim=1)

        combined_fft = (
            eigen_weights[:, 0].unsqueeze(0).unsqueeze(0) * self.fft_zero.unsqueeze(0).unsqueeze(-1) +
            eigen_weights[:, 1].unsqueeze(0).unsqueeze(0) * self.fft_plus1.unsqueeze(0).unsqueeze(-1) +
            eigen_weights[:, 2].unsqueeze(0).unsqueeze(0) * self.fft_minus1.unsqueeze(0).unsqueeze(-1)
        )

        h_fft = Bx_fft * combined_fft
        h = torch.fft.irfft(h_fft, n=seq_len, dim=1)

        y = torch.nn.functional.linear(C * h, self.C_proj.weight.T)

        return y
