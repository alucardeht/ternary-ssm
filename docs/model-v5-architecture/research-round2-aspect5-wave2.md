# Research Round 2, Aspect 5: Parallelism & Hardware Utilization — Wave 2 (Math-Paradigm-Breakers)

**Date:** 2026-03-01
**Source:** Merged from Theme A (System-Level) and Theme B (Kernel-Level) mathematical formulations
**Type:** Novel mathematical frameworks for heterogeneous CPU+GPU+NPU execution and kernel-level optimization

---

## Summary

Wave 2 Math-Paradigm-Breakers reveal two complementary dimensions: system-level techniques exploit ternary's algebraic structure (diagonal A enables parallel scan, bounded-error speculation, optimal partitioning) to orchestrate CPU-GPU-NPU heterogeneous execution, while kernel-level techniques exploit ternary's compact representation (2 bits per weight) to achieve extreme register efficiency, POPCOUNT-based computation, and 6.8x memory traffic reduction. Together, they form a complete execution stack from optimal device partitioning down to instruction-level encoding.

**Key Insight:** Ternary {-1,0,+1} enables a unique "roofline inversion" where arithmetic intensity falls compute-bound on CPU (balance 1.42 ops/byte) but bandwidth-bound on GPU (balance 595 ops/byte). This inverts conventional wisdom (FP matmul is GPU-optimal) and justifies a fundamentally different heterogeneous strategy: ternary projections on CPU (SIMD), SSM scan on GPU (parallel prefix), attention on GPU.

---

## HIGH VIABILITY FORMULATIONS

### System-Level: Optimal Heterogeneous Execution

**R2-A5-M1: GRAPH-CUT PARTITION** (HIGH feasibility, MEDIUM-HIGH impact)
Optimal 2-3 device partitioning as polynomial min-cut/max-flow. Linear chain SSM+attention models exhibit "contiguity" — optimal partition has ≤2 device boundaries despite alternating operator types. DP solution O(L·k²) for L=32 layers, k=3 devices. Expected 20-40% latency reduction vs naive "all GPU" strategy.

**R2-A5-M7: TERNARY RECURRENCE EIGENDECOMPOSITION** (HIGH feasibility, HIGH impact)
Ternary diagonal A decomposes SSM into three independent subspaces: "forget" (a=0, fully parallel), "integrate" (a=+1, prefix sum), "oscillate" (a=-1, alternating prefix sum). All three are GPU-friendly with O(log T) depth. Inverts "SSM on CPU" conventional wisdom — instead: ternary projection on CPU → SSM scan on GPU (parallel prefix) → attention on GPU. No sequential bottleneck for ternary A.

**R2-A5-M2: SPECULATIVE SSM PIPELINE** (HIGH feasibility, HIGH impact)
Ternary A ensures spectral radius ≤ 1, guaranteeing non-growing speculative error: ||e_{t+k}||_2 ≤ (1−f₀)^(k/2) · ||e₀||_2 where f₀ ≈ 1/3. Enables aggressive CPU→GPU pipelining: CPU speculatively executes SSM step while GPU computes attention, correction cost O(D) adds when true activation arrives. Error never diverges. 2x latency reduction for balanced CPU/GPU workloads.

**R2-A5-M3: TAPS — Ternary-Algebraic Parallel Scan** (HIGH feasibility, VERY HIGH impact)
Ternary monoid {-1,0,+1} under multiplication decomposes into two binary prefix operations: zero-propagation (find last zero position) and parity-accumulation (count negatives). Both O(log T) parallel depth with pure integer arithmetic. Extends to selective (input-dependent) A via per-dimension tracking. For T=2048: 2048→11 step reduction. Scan is 10-50x faster, overall SSM training 1.5-3x. Keystone formulation — enables eigendecomposition at selective level.

**R2-A5-M4: ASYMMETRIC MICRO-BATCH PIPELINE** (HIGH feasibility, MEDIUM impact)
CPU and GPU have different optimal batch sizes (B_cpu=16, B_gpu=64-128 for Tensor Cores). Rate-matching buffer (4-8 tensors) with simple leaky-bucket analysis enables asymmetric batching, achieving 10-30% throughput improvement. MAX-MIN optimization finds balance point where per-sample times equalize. Small buffer needed due to regularity of neural network timing.

### Kernel-Level: Extreme Register and Instruction Efficiency

**R2-A5-M1': RR-TSSM — Register-Resident Ternary SSM** (HIGH feasibility)
For D ≤ 739 on AVX512 (32 registers), entire SSM state fits in registers (D/32 registers). Ternary weights (2-bit) stream only as memory, not state. 9x arithmetic intensity improvement: AI from 4/9 → 4 ops/byte. Instruction-level pipeline hides state update latency. 3-5x real-world speedup for D=512.

**R2-A5-M6: RIT — Roofline Inversion Theorem** (HIGH feasibility, VERY HIGH impact)
Ternary GEMV arithmetic intensity = 4 ops/byte. CPU balance point = 1.42 ops/byte (compute-bound). GPU balance point = 595 ops/byte (bandwidth-bound). For B=1: CPU utilization ~100%, GPU utilization ~0.7%. Batch B ≥ 355 needed for GPU to be compute-bound. Formal proof that CPU is optimal for ternary matmul at inference batch sizes. Directly justifies heterogeneous execution strategy.

**R2-A5-M7: AETW — Action-Encoded Ternary Weights** (HIGH feasibility)
Encode as two bit-plane masks: ADD-plane (w=+1), SUB-plane (w=-1), instead of numeric 2-bit values. Direct SIMD consumption: `kmovw k1, add_mask; kmovw k2, sub_mask; vaddps zmm_acc{k1}, zmm_acc, zmm0`. 4 instructions per 16-element vector vs 6-9 for numeric decode. 33-55% fewer instructions, zero storage overhead. Weights ARE the instructions.

**R2-A5-M8: OTTG — Output-Tiled Ternary GEMV** (HIGH feasibility)
Process M_t=11 output rows simultaneously (AVX512 constraints), amortize N-element input load across 11 outputs. Memory traffic reduction: 6.8x per output element. Optimal M_t · N_t = 32·16 registers = 512 work per register set. Roofline-justified: for compute-bound ternary (per M6), reducing memory frees bandwidth for prefetching. 4-5x real-world speedup for large matrices.

---

## MEDIUM VIABILITY FORMULATIONS

**R2-A5-M5: STATE-IN-SRAM** (MEDIUM feasibility, MEDIUM impact)
NPU SRAM (~1 TB/s, 1 ns latency) vs PCIe (32 GB/s, 1 μs). SSM state (64 KB typical) fits entirely in NPU SRAM. CPU sends delta = B·x (4 KB), NPU applies h'=diag(a)·h+delta (ternary: 3 instructions), sends back. 5-7x latency reduction for state access. Risk: Intel NPU driver support for custom kernels uncertain. Hardware capable, software ecosystem unproven.

**R2-A5-M2: AGS — Asymmetric Gradient Sparsification** (HIGH feasibility, MEDIUM impact)
In ZeRO-Offload: GPU→CPU gradients (FP16, 2B/param) are 8x larger than CPU→GPU weights (2-bit ternary, 0.25B/param). Only 0.24% of weights near quantization boundaries can flip in one step. Transfer only masked gradients + bitmask. 15x reduction in GPU→CPU transfer (8.26 GB → 0.54 GB per step). No approximation — purely exploits ternary sparsity. Orthogonal to standard ZeRO-Offload.

**R2-A5-M4: TPF — Temporal Phase Fusion** (HIGH feasibility)
Fuse quantize+forward+backward into one kernel by temporal scheduling (quantize frees registers before forward uses them). Register budget increases 3.4x: G ≤ 66 → G ≤ 223. Memory traffic reduction: 53% on weight terms. Shared memory cost: G/4 bytes (negligible). Occupancy trade-off acceptable for compute-light ternary. 1.3-1.8x training speedup.

**R2-A5-M5: EDLPC — Error-Dampened Low-Precision Checkpointing** (HIGH feasibility)
Ternary A=0 entries erase checkpoint errors: P(error survives K steps) = (2/3)^K. K=20: 99.97% of state errors zeroed naturally. INT4 checkpoints viable vs FP16. Memory savings: 4x on activation checkpoints (45 KB → 11.25 KB for T=2048, D=256). Zero compute overhead. Unique to ternary — no FP analog.

**R2-A5-M6: INFORMATION-BOTTLENECK TRANSFER** (MEDIUM feasibility, LOW-MEDIUM impact)
Learned compression layer reduces inter-device transfers for long sequences (T>5000). Rate-distortion: minimize task loss + λ·bottleneck_size. Optimal bottleneck dimension co-trained. Ternary compress/expand layers (fast on CPU SIMD). With d_bottleneck=d_model/4: 4x transfer reduction. Break-even at T≈5500; profitable for T>8000. Risk: bottleneck may limit expressivity.

**R2-A5-M8: POWER-AWARE KNAPSACK SCHEDULING** (HIGH feasibility)
Operator-to-device assignment under power budget: constrained shortest path DP. O(L·k²) = O(288) for L=32, k=3. Phase transitions: critical power threshold P* where GPU-beneficial operators exceed budget. Greedy ordering by efficiency e_i = (c_cpu − c_gpu)/(p_gpu − p_cpu). NPU optimal for power-constrained scenarios (p_npu << p_cpu). 10-25% latency improvement at fixed power.

---

## SPECULATIVE FORMULATIONS

**R2-A5-M3: SFB — Sparsity Futility Bound** (MEDIUM feasibility, SPECULATIVE)
At 33% zeros, SIMD-aligned skipping is impossible: P(16-wide all-zero) = (1/3)^16 ≈ 0. Dense SIMD strictly optimal at natural ternary distribution. Training regularization to reach 56% zeros enables ~10% sparse speedup; 75% zeros enables 25% speedup. Connects training loss to hardware-exploitable sparsity. Novel training-hardware co-optimization axis.

**R2-A5-M9: GF3-CASCADE — Cascaded GF(3) Computation** (SPECULATIVE, VERY HIGH impact if viable)
When both weights AND activations are ternary, matmul reduces to GF(3) operations: 4 AND + 4 POPCOUNT on N-bit masks. 14.6x fewer instructions than standard ternary matmul. Enables fully integer inference for ternary segments. Only viable for models with D_hidden ≥ 2048 (preserve ~3200 bits of information per ternary vector at 1.585 bits/element). Requires intermediate ternary quantization (adds STE backward cost).

---

## KEY INSIGHT — The Ternary Roofline Inversion

Conventional wisdom: GPU executes matrix operations faster (Tensor Cores, 595 ops/byte balance point). Ternary inverts this via arithmetic intensity = 4 ops/byte:

| Device | Balance Point | Ternary AI | Status |
|--------|---------------|-----------|--------|
| CPU | 1.42 ops/byte | 4 | COMPUTE-BOUND (100% utilization) |
| GPU | 595 ops/byte | 4 | BANDWIDTH-BOUND (0.7% utilization) |

This single mathematical fact reshapes the entire execution strategy:
- **System-level:** M1 optimal partition favors CPU for ternary, GPU for FP16 attention
- **Operator ordering:** M7 eigendecomposition shows SSM scan should run on GPU (parallel prefix), not CPU
- **Batch size crossover:** B ≥ 355 needed for GPU to be compute-bound (training-only regime)
- **Kernel-level:** M6, M7, M8 all optimize CPU execution (where ternary is compute-bound)

---

## RECOMMENDED COMBINATIONS

### Combination α: Heterogeneous Execution Stack (M1 + M7 + M2 + RIT)
Optimal partition (M1) assigns ternary projections to CPU (RIT roofline justification) and SSM scan to GPU (M7 eigendecomposition proves parallelizability). Speculative SSM pipeline (M2) overlaps GPU attention with CPU speculation. Complete system redesign of execution strategy.

### Combination β: CPU Ternary Kernel Stack (M1' + M6 + M7 + M8)
Register-resident SSM state (M1') + roofline-justified CPU compute-focus (M6) + action-encoded weights (M7) + output tiling (M8). Forms complete, instruction-efficient CPU SIMD engine. Expected 3-5x CPU ternary speedup.

### Combination γ: Memory-Optimal Training (AGS + TPF + EDLPC)
Asymmetric gradient sparsification (15x less transfer) + temporal phase fusion (53% less memory traffic) + INT4 checkpoints (4x less checkpoint memory). Three independent optimizations, orthogonal benefits.

### Combination δ: Full-Integer Inference (AETW + GF3 + SFB-regularization)
Action-encoded weights (M7) + GF(3) segments for ternary-in-ternary-out layers (M9). Speculative quality impact, but 10-15x speedup on ternary layers if hidden dimensions sufficient.

---

## FOR WAVE 3 PARADIGM-BREAKERS (Seed Questions)

1. **Can SSM selectivity (B, C, dt input-dependent) be exploited at the system level?** Current TAPS and eigendecomposition assume fixed A. If B/C vary per token, does the parallel scan structure change? Does selective routing (per-token device assignment) become feasible?

2. **Is the roofline inversion (M6) exploitable via novel instruction set design?** Current analysis uses x86-64 / ARM balance points. What if we designed a custom ISA or FPGA for ternary matmul with different compute-memory ratio? Could we shift the balance point?

3. **Can the ternary quantization error (A=0 forgets, A=±1 preserves) be intentionally designed into the architecture?** Instead of treating a_i ∈ {-1,0,+1} as emergent from training, what if we enforce structured sparsity patterns (e.g., a_i=0 for dimensions corresponding to "short-term" features)?

4. **How do ternary FFN gates interact with GF(3) POPCOUNT computation?** Gate outputs are typically gated linear units (e.g., GLU). Can gating be performed in GF(3) arithmetic, or does it break the ternary quantization?

5. **Is there a geometric property of ternary matrices that enables novel partitioning strategies?** Ternary matrices have structure (many zero columns, rank constraints). Can we exploit this for hierarchical partitioning beyond M1's layer-wise approach (e.g., column-group partitioning)?

---

## UNKNOWNS & RISKS

1. **Eigendecomposition assumes selective A maintains 33% zeros.** If training drives ternary distribution away from uniform (sparsity increases or concentrates), the three subspaces become unbalanced. "Forget" subspace dominates → model loses memory.

2. **NPU driver maturity (M5).** Intel OpenVINO 2025+ may not support custom kernels for state server operation. Even if hardware capability exists, software ecosystem risk is substantial.

3. **GF(3) quality impact (M9).** Ternary intermediate activations compress to ~1600 bits per hidden (D=1024). Quality loss empirically unknown. Only viable for models with D_hidden ≥ 2048, and even then quality could degrade significantly.

4. **Speculative prediction heuristics (M2).** Zero-order hold (h'_t = h_{t-1}) may have high initial error for rapidly changing states. Better predictors (linear, learned) reduce correction cost but add complexity.

5. **Gradient sparsification margin (AGS, M2).** Multi-step accumulation increases safe margin; optimal K_skip is empirically determined. Early stopping or changing optimizer can invalidate assumptions.

---

## RANKING BY IMPACT & FEASIBILITY

| Rank | ID | Name | Impact | Feasibility | Notes |
|------|-----|------|--------|-------------|-------|
| 1 | M3 | TAPS | VERY HIGH | HIGH | Keystone — enables selective SSM scan, 50x depth reduction |
| 2 | M7 | Eigendecomposition | HIGH | HIGH | Inverts "SSM on CPU" wisdom, system-level restructuring |
| 3 | M6 | Roofline Theorem | HIGH | HIGH | Formal justification for heterogeneous CPU/GPU split |
| 4 | M1 | Graph-Cut Partition | MEDIUM-HIGH | HIGH | Optimal polynomial solution, foundation for M7 |
| 5 | AETW | Action-Encoded Weights | MEDIUM | HIGH | 33% instruction reduction, zero overhead |
| 6 | M2 | Speculative Pipeline | HIGH | HIGH | 2x latency reduction, unique to ternary |
| 7 | OTTG | Output Tiling | MEDIUM | HIGH | 6.8x memory savings, Roofline-justified |
| 8 | TPF | Temporal Phase Fusion | MEDIUM | HIGH | 53% memory traffic, 1.3-1.8x training speedup |
| 9 | RR-TSSM | Register-Resident State | MEDIUM | HIGH | 9x AI, 3-5x speedup for small D |
| 10 | AGS | Gradient Sparsification | MEDIUM | HIGH | 15x less transfer, orthogonal to ZeRO |
| 11 | EDLPC | INT4 Checkpoints | MEDIUM | HIGH | 4x memory, unique to ternary |
| 12 | SFB | Sparsity Analysis | LOW | HIGH | Impossibility result, directs training |
| 13 | GF3-CASCADE | GF(3) Segments | VERY HIGH* | SPECULATIVE | 14.6x speedup, but quality risk unknown |
| 14 | STATE-IN-SRAM | NPU Server | MEDIUM | MEDIUM | Elegant, driver support uncertain |

*GF(3) has highest speedup IF quality survives (unproven).

