# R3-A2: Backward Pass Mathematical Analysis

## Context
Research on training speed optimization for ternary SSM. Wave 1 focus: Mathematical derivations for optimal recomputation strategies in backward pass.

---

### R3-A2-F1: Lazy State Saving for SSM
- **Src:** SSM recurrence theory [2024] | **Gain:** 80% memory reduction | **V:** PROMISING
- **Principle:** Save only (A, B, C) parameters, recompute hidden states h_t = A^t x_0 + Σ during backward
- **Application:** For SSM backward, recompute state transitions from A matrix and inputs

### R3-A2-F2: Mixed Checkpoint Strategy
- **Src:** Double checkpointing paper (arXiv:2412.11810) [2024] | **Gain:** 10x longer sequences | **V:** PROMISING
- **Principle:** Recursively checkpoint halves of computation graph, optimizing for local memory constraints
- **Application:** Apply double checkpoint to SSM scan operations for long sequences

### R3-A2-F3: Activation Rematerialization Order
- **Src:** Optimal rematerialization theory [2024] | **Gain:** 50% memory reduction | **V:** PROMISING
- **Principle:** Order recomputation by decreasing gradient contribution: recompute high-impact activations first
- **Application:** In SSM, prioritize recomputing state-to-output path over input-to-state path

### R3-A2-F4: Chunked Gradient Accumulation
- **Src:** Gradient accumulation analysis [2024] | **Gain:** Unlimited effective batch size | **V:** PROMISING
- **Principle:** Split backward into chunks, accumulate gradients, update only at end of accumulation window
- **Application:** For large SSM models, chunk backward into micro-batches matching GPU memory

### R3-A2-F5: Off-Chip Memory Optimization
- **Src:** IPU checkpointing paper [2024] | **Gain:** 4x larger networks | **V:** PROMISING
- **Principle:** Store checkpoints in slower off-chip memory when local memory insufficient
- **Application:** Use CPU memory for SSM checkpointing during very long sequence training

### R3-A2-F6: Selective Activation Checkpointing
- **Src:** Selective checkpointing theory [2024] | **Gain:** Adaptive memory usage | **V:** SPECULATIVE
- **Principle:** Checkpoint only activations with gradient contribution above threshold τ
- **Application:** For SSM, checkpoint states where |∂L/∂h_t| > τ

### R3-A2-F7: Fused Backward-Update Kernel
- **Src:** Triton fused kernels [2025] | **Gain:** 6x speedup | **V:** PROMISING
- **Principle:** Fuse gradient computation with parameter update in single kernel launch
- **Application:** Implement fused Lion update kernel for ternary weights

### R3-A2-F8: Recompute-Aware Memory Allocation
- **Src:** Memory allocation theory [2024] | **Gain:** Optimal memory usage | **V:** PROMISING
- **Principle:** Pre-allocate memory based on recomputation strategy: M = M_stored + M_recompute
- **Application:** For SSM, allocate based on k=√L checkpointing formula

---

## Mathematical Derivations

### M1: Optimal Checkpoint Count Derivation
For SSM with L steps and memory budget M:
- **Derivation:** Let k = number of checkpoints, each covering L/k steps
- **Forward cost:** O(L) regardless of k  
- **Memory:** M = O(k·d + L/k·d) = O(d·(k + L/k))
- **Optimal:** dM/dk = 0 → k* = √L
- **Result:** Checkpoint interval = √L steps

### M2: Ternary Recompute Cost Analysis
For ternary weights with popcount operations:
- **Derivation:** FP32 GEMM cost = O(n³), ternary popcount cost ≈ O(n²/32)
- **Recompute ratio:** Cost_recompute / Cost_forward ≈ 1/16 for popcount
- **Result:** For ternary, checkpointing may be unnecessary - just recompute

### M3: Gradient Flow Through SSM
- **Derivation:** ∂L/∂B = Σ_t h_t · ∂y/∂o_t, computed via SSM backward pass
- **Complexity:** O(L·d_model·d_state) for full backward
- **Optimization:** Save h_t only at checkpoint intervals, recompute others

---

## Key Insights
1. Ternary weights enable near-free recomputation due to popcount efficiency
2. SSM recurrence structure allows specialized checkpointing strategies
3. Double checkpointing is optimal for SSM architectures

## Seed Questions
1. Can we derive SSM-specific checkpoint placement from eigenvalue distribution of A?
2. How does ternary quantization interact with gradient checkpointing accuracy?
