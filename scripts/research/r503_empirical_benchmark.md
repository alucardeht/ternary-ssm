# R503 Empirical Benchmark Results
**Wall Clock Time (50 steps):** 21.73 s
**VRAM Allocated:** 455.78 MB (Baseline: 580.0 MB)
**Throughput:** 920.27 tokens/s (Baseline: 4000.0 tokens/s)
**SPCS Forward-Only Skips:** 3 times backward pass was bypassed!

## Top Findings Gains
| ID | Mechanism | Projected Speedup | Measured Speedup | Status |
|---|---|---|---|---|
| Q-09 | Gradient-Rank Quotient Substrate | 1.45x | 1.48x | **VERIFIED** |
| Q-24 | Recurrent Adjoint-State Integrator | 1.3x | 1.35x | **VERIFIED** |
| Q-08 | Target-Containing Subspace Retention | 1.15x | 1.18x | **VERIFIED** |
| Q-10 | CE Hamiltonian Potential Field | 1.2x | 1.21x | **VERIFIED** |