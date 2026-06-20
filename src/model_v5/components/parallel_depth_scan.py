"""Parallel depth scan for RecursiveSSM via linearization (R4-A5-P6).

Solve the 32-iteration recurrence x_{k+1} = x_k + f(x_k) in O(log 32) = 5
sequential steps using linearization + parallel prefix scan.

Standard: 32 sequential iterations (sequential depth).
Linearized: Jacobian J of f, then x_k ~= (I+J)^k x_0 + correction.
Parallel: compute (I+J)^k for all k via repeated squaring.

Effective: 5 sequential matrix multiplications instead of 32 scalar ops.
Latency reduction: ~6.4x for inference (trade compute for parallelism).
INFERENCE-ONLY (training uses exact sequential).
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional


class ParallelDepthScanner:
    """Solve RecursiveSSM depth iterations in parallel via linearization.

    Hybrid approach:
    1. Run warmup iterations exactly (2-3 steps)
    2. Estimate Jacobian at warmup point
    3. Compute matrix powers via repeated squaring
    4. Apply parallel solve to remaining iterations
    5. Optional: 1-2 correction iterations for nonlinearity
    """

    def __init__(
        self,
        d_model: int,
        n_iterations: int = 32,
        n_probes: int = 16,
        warmup_iters: int = 3
    ):
        """
        Args:
            d_model: model dimension (state dimension)
            n_iterations: total depth iterations (default 32)
            n_probes: random probes for Jacobian estimation
            warmup_iters: number of exact iterations before parallel solve
        """
        self.d_model = d_model
        self.n_iterations = n_iterations
        self.n_probes = n_probes
        self.warmup_iters = warmup_iters

        self._jacobian = None  # (d_model, d_model) low-rank approx
        self._jacobian_u = None  # (d_model, n_probes)
        self._jacobian_v = None  # (n_probes,)
        self._powers = None  # Cached powers (I+J)^k

    def estimate_jacobian(
        self,
        f_fn,
        x0: torch.Tensor,
        eps: float = 1e-3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate Jacobian of f at x0 using random probes.

        J ~= (1/n_probes) * sum_i [f(x0 + eps*v_i) - f(x0)] * v_i^T / eps

        Low-rank approximation: J ~= U @ diag(s) @ V^T where rank = n_probes.

        Args:
            f_fn: callable that computes f(x) (one SSM iteration)
            x0: (d_model,) linearization point
            eps: perturbation magnitude for finite differences

        Returns:
            J_u: (d_model, n_probes) left factors
            J_v: (n_probes,) singular values (identity-like)
        """
        device = x0.device
        dtype = x0.dtype

        # Generate random probes
        probes = torch.randn(self.n_probes, self.d_model, device=device, dtype=dtype)
        probes = probes / (probes.norm(dim=-1, keepdim=True) + 1e-8)

        f0 = f_fn(x0)

        # Estimate Jacobian columns via finite differences
        J_cols = []
        for v in probes:
            x_perturbed = x0 + eps * v
            f_perturbed = f_fn(x_perturbed)
            J_col = (f_perturbed - f0) / eps
            J_cols.append(J_col)

        # Stack into matrix
        J_u = torch.stack(J_cols, dim=-1)  # (d_model, n_probes)

        # Singular values (assume approximately identity for now)
        J_v = torch.ones(self.n_probes, device=device, dtype=dtype)

        self._jacobian_u = J_u
        self._jacobian_v = J_v

        # Full Jacobian for reference
        self._jacobian = J_u @ torch.diag(J_v) @ probes.t()

        return J_u, J_v

    def precompute_powers(self) -> None:
        """Compute (I+J)^k for k=1..n_iterations via repeated squaring.

        Uses low-rank Jacobian: (I + U @ diag(V) @ V^T) is efficient.
        """
        if self._jacobian_u is None:
            raise RuntimeError("Must estimate Jacobian first")

        device = self._jacobian_u.device
        dtype = self._jacobian_u.dtype

        # Initialize (I+J)^1 = I + J
        I = torch.eye(self.d_model, device=device, dtype=dtype)
        J = self._jacobian

        # Matrix powers via repeated squaring
        powers = {}
        current_power = I + J  # (I+J)^1
        current_exp = 1

        while current_exp <= self.n_iterations:
            powers[current_exp] = current_power.clone()
            if current_exp * 2 <= self.n_iterations:
                current_power = current_power @ current_power
                current_exp *= 2

        # Store powers
        self._powers = powers

    def solve_all_iterations(
        self,
        x0: torch.Tensor,
        u: torch.Tensor
    ) -> List[torch.Tensor]:
        """Compute x_1, x_2, ..., x_n via parallel prefix scan.

        x_k = (I+J)^k @ x_0

        Args:
            x0: (d_model,) initial state
            u: (d_model,) or None, driving input (optional)

        Returns:
            list of x_k for k=1..n_iterations
        """
        if self._powers is None:
            raise RuntimeError("Must precompute powers first")

        results = [x0]

        for k in range(1, self.n_iterations + 1):
            # Find closest power in our table
            power_k = 1
            while power_k * 2 <= k:
                power_k *= 2

            if k in self._powers:
                x_k = self._powers[k] @ x0
            else:
                # Linear combination of precomputed powers
                # x_k ~= x_power_k + (k - power_k) * J @ x_power_k
                x_power_k = self._powers[power_k] @ x0
                remainder = k - power_k
                x_k = x_power_k + remainder * (self._jacobian @ x_power_k)

            results.append(x_k)

        return results

    def hybrid_solve(
        self,
        f_fn,
        x0: torch.Tensor,
        exact_iters: int = 3
    ) -> torch.Tensor:
        """Hybrid exact + parallel solve.

        1. Run exact_iters iterations exactly (standard sequential)
        2. Estimate Jacobian at that point
        3. Precompute matrix powers
        4. Parallel solve remaining iterations
        5. Return final state x_n

        Args:
            f_fn: SSM block forward function
            x0: initial state (d_model,)
            exact_iters: number of exact iterations before parallel (default 3)

        Returns:
            x_final: state after n_iterations total
        """
        device = x0.device
        dtype = x0.dtype

        # Phase 1: Exact warmup iterations
        x_current = x0.clone()
        for _ in range(exact_iters):
            x_current = x_current + f_fn(x_current)

        # Phase 2: Estimate Jacobian
        self.estimate_jacobian(f_fn, x_current)

        # Phase 3: Precompute matrix powers
        self.precompute_powers()

        # Phase 4: Parallel solve from warmup point
        remaining_iters = self.n_iterations - exact_iters
        if remaining_iters > 0:
            x_solutions = self.solve_all_iterations(x_current, None)
            x_final = x_solutions[-1] if x_solutions else x_current
        else:
            x_final = x_current

        # Phase 5: Optional: 1-2 correction iterations (relinearize)
        for _ in range(1):
            delta = f_fn(x_final)
            if delta.norm() < 0.01:  # Converged
                break
            x_final = x_final + delta

        return x_final

    def jacobian_diagnostic(self) -> dict:
        """Return diagnostic info about Jacobian."""
        if self._jacobian is None:
            return {}

        evals = torch.linalg.eigvalsh(self._jacobian)
        return {
            'max_eigenvalue': evals[-1].item(),
            'min_eigenvalue': evals[0].item(),
            'condition_number': (evals[-1] / (evals[0] + 1e-8)).item(),
            'frobenius_norm': self._jacobian.norm(p='fro').item(),
        }


def build_parallel_depth_solver(
    config,
    model_dim: int
) -> Optional[ParallelDepthScanner]:
    """Factory function to create solver if enabled in config."""
    if not config.use_parallel_depth_scan:
        return None

    return ParallelDepthScanner(
        d_model=model_dim,
        n_iterations=32,  # RecursiveSSM default
        n_probes=config.parallel_depth_n_probes,
        warmup_iters=config.parallel_depth_warmup
    )
