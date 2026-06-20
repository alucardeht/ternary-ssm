import torch
import torch.nn as nn


if hasattr(nn, 'RMSNorm'):
    class RMSNorm(nn.RMSNorm):
        """Root Mean Square Layer Normalization (fused PyTorch implementation).

        Uses torch.nn.RMSNorm fused kernel (1 kernel vs 3+ unfused ops).
        """
        def __init__(self, d: int, eps: float = 1e-6):
            super().__init__(d, eps=eps, elementwise_affine=True)
else:
    class RMSNorm(nn.Module):
        """Root Mean Square Layer Normalization (unfused fallback).

        Fallback for PyTorch < 2.4 without torch.nn.RMSNorm.
        """
        def __init__(self, d: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(d))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x * norm * self.weight


class BitShiftNorm(nn.Module):
    """Bit-shift normalization - approximates RMSNorm using power-of-2 scaling.

    Uses STE for the rounding so gradients flow during training.
    At inference on integer hardware, normalization becomes a bit-shift.
    """
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        log2_rms = torch.log2(rms)
        log2_rms_rounded = torch.round(log2_rms)
        log2_rms_q = log2_rms + (log2_rms_rounded - log2_rms).detach()
        rms_q = torch.exp2(log2_rms_q)
        return (x / rms_q) * self.weight
