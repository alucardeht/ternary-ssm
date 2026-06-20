"""Wrapper for fused SSM recurrence CUDA kernel using NVRTC."""

import torch
from pathlib import Path
from .nvrtc_compiler import NVRTCKernel


class FusedSSMKernel:
    def __init__(self):
        self._kernel = NVRTCKernel()
        self._compiled = False

        if self._kernel.available:
            cu_path = str(Path(__file__).parent / 'fused_ssm_step.cu')
            self._compiled = self._kernel.compile_file(cu_path)

    @property
    def available(self) -> bool:
        return self._compiled

    def step(self, h, dt, A, B, C, x, D) -> tuple:
        """Execute fused SSM step via CUDA kernel.

        Args:
            h: (batch, n_heads, d_state, head_dim) - IN/OUT state
            dt: (batch, n_heads) - scalar dt after softplus
            A: (n_heads,) - A matrix log-space (negative values)
            B: (batch, d_state) - input projection
            C: (batch, d_state) - output projection
            x: (batch, n_heads, head_dim) - activation (post-conv+silu)
            D: (n_heads, head_dim) - skip connection weights

        Returns:
            (y, h): output and updated state
        """
        if not self.available:
            return self.step_fallback(h, dt, A, B, C, x, D)

        batch = h.shape[0]
        n_heads = h.shape[1]
        d_state = h.shape[2]
        head_dim = h.shape[3]

        h = h.contiguous()
        dt = dt.contiguous()
        A = A.contiguous()
        B = B.contiguous()
        C = C.contiguous()
        x = x.contiguous()
        D = D.contiguous()

        y = torch.empty(batch, n_heads * head_dim, dtype=torch.float32, device=h.device)

        try:
            grid = (n_heads, batch, 1)
            block = (32, head_dim, 1)

            self._kernel.launch(
                'fused_ssm_step_kernel',
                grid, block,
                [h, y, dt, A, B, C, x, D, batch, n_heads, d_state, head_dim]
            )

            # No synchronize needed — default CUDA stream ensures ordering
            # with subsequent PyTorch ops automatically
            return y, h

        except Exception as e:
            print(f"Warning: CUDA kernel launch failed: {e}. Falling back to PyTorch.")
            return self.step_fallback(h, dt, A, B, C, x, D)

    def step_fallback(self, h, dt, A, B, C, x, D) -> tuple:
        """Pure PyTorch fallback for SSM step."""
        batch = h.shape[0]
        n_heads = h.shape[1]
        d_state = h.shape[2]
        head_dim = h.shape[3]

        dt_expanded = dt.unsqueeze(-1).unsqueeze(-1)
        A_expanded = A.unsqueeze(-1).unsqueeze(-1)
        dA = torch.exp(dt_expanded * A_expanded)

        B_expanded = B.unsqueeze(1).unsqueeze(-1)
        x_expanded = x.unsqueeze(2)
        dBx = dt_expanded * B_expanded * x_expanded

        h[:] = dA * h + dBx

        C_expanded = C.unsqueeze(1).unsqueeze(-1)
        y_ssm = (h * C_expanded).sum(dim=2)

        y = y_ssm.reshape(batch, -1) + D * x.reshape(batch, -1)

        return y, h
