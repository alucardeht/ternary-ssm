# Research Round 3 — Memory Optimization

## Overview

This document tracks research findings on memory optimization techniques for ternary SSM models. Focus: reducing VRAM/RAM during inference and enabling larger models on constrained hardware (12GB GPU, 32GB RAM, or CPU-only).

## Already Known (from previous phases)

- **Packed 2-bit ternary storage:** 16x compression vs FP32 (implemented in TernaryLinear)
- **DeferredTernaryLinear:** Low-rank delta replaces FP32 shadow weights (95% training memory savings at 1B)
- **KV-cache elimination:** SSM has O(1) state per token vs O(n) for Transformer attention
- **Sliding Window Attention (SWA):** Fixed window=128 bounds attention memory
- **Gradient checkpointing:** Trades compute for memory during training
- **GaLore optimizer:** Low-rank gradient projection reduces optimizer state memory
- **Weight tying:** Embedding + LM head share weights (saves d_model × vocab_size params)
- **Current inference memory:** 876.8 tok/s at ~200MB for 6M model (Phase 23 engine)
- **Current VRAM usage:** 4.4GB peak for 50M ternary training (43% of 12GB limit)

## NEW FINDINGS (Round 3)

### Wave 1: Web Research (2 researchers, Feb 2026)

#### WEIGHT STORAGE & COMPRESSION

| # | Technique | Memory Impact | Applicability |
|---|-----------|--------------|---------------|
| F1 | Basis Sharing (Cross-Layer SVD) | <1.5 bits/weight effective | HIGH |
| F2 | Sherry 1.25-Bit (3:4 Structured Sparsity) | 1.25 bits/weight (21% vs ternary) | HIGH |
| F3 | Memory-Mapped Weights (mmap) | Eliminates load copy, 2x peak reduction | MEDIUM |
| F4 | XStreamVGGT Cache Compression | 4.42x KV cache reduction | LOW (no KV cache in SSM) |
| F5 | HY-1.8B SEQ Quantization | 2 bits/weight symmetric | LOW (heavier than ternary) |
| F6 | Tequila Trapping-Free Ternary | Better quality at 1.58 bits | HIGH |
| F7 | Nacrith CDF-24 Entropy Coding | ~1.59 bits vs 1.6 fixed packing | LOW-MEDIUM |

**F1: Basis Sharing — Cross-Layer SVD Compression** (ICLR 2025, arxiv 2410.03765)
- Decompose weight matrices across layers as linear combination of shared basis vectors + unique coefficients.
- W = Shared_Basis × Layer_Coefficients. SSM layers are repetitive — natural fit.
- Combined with 1.6-bit ternary encoding → <1.5 bits/weight effective storage.
- Code available: TUDa-HWAI/Basis_Sharing on GitHub.

**F2: Sherry 1.25-Bit with 3:4 Structured Sparsity** (AngelSlim/Tencent, Feb 2026)
- Enforce 3:4 structured sparsity: exactly 3 non-zero ±1 in every 4-weight block.
- Pack 4 weights into 5 bits (32 permutations = 2^5). SIMD-friendly.
- LLaMA-3.2 3B: 712MB vs 846MB (15.8% smaller), 45.55 tok/s on Intel i7.
- Requires QAT with Arenas gradient injection during training.

**F3: Memory-Mapped Weights (mmap)** (llama.cpp, MLX discussions 2025)
- OS mmap() maps model files to virtual memory without full RAM copy.
- Eliminates 2x peak during startup. Can run models larger than RAM via paging.
- Throughput collapses when paging (0.025 tok/s for 70GB on 64GB RAM).
- Best for cold-start latency reduction or multi-model serving (shared pages).

**F6: Tequila Trapping-Free Ternary** (AngelSlim 2026, Huang et al. 2025) — HIGH PRIORITY
- Reactivates deadzone weights by repurposing as dynamic biases during training.
- Y = X·Q(W) + C(W) where C(W) = Σ λw_i for deadzone indices.
- Biases merged offline post-training → ZERO inference overhead.
- LLaMA-3.2 1B: 0.519 avg accuracy vs BitNet 0.483 (+7.5%).
- Directly addresses our Phase 18-C deadzone trapping issue.

#### RUNTIME MEMORY (Activations, State, Buffers)

| # | Technique | Memory Impact | Applicability |
|---|-----------|--------------|---------------|
| F8 | CompreSSM (Balanced Truncation) | 75% SSM state reduction | HIGH |
| F9 | SSM State Quantization (QAT) | 6x model compression | HIGH |
| F10 | Activation Checkpointing (Hybrid) | 50% KV cache reduction | MEDIUM |
| F11 | FlashAttention-3 FP8 | 4x attention activation reduction | MEDIUM |
| F12 | In-Place Fused SSM Kernels | 30-50% intermediate buffers | HIGH |
| F13 | Memory Pooling (Pre-Allocated) | 10-20% fragmentation reduction | MEDIUM |
| F14 | NVMe/SSD Offload | Run 2-3x larger models | LOW (future 1B+) |
| F15 | SSM State Low-Rank Compression | 2-4x state compression | LOW (high risk) |
| F16 | Continuous Batching for SSM | 20-40% utilization improvement | MEDIUM |

**F8: CompreSSM — In-Training State Compression** (arxiv 2510.02823, Oct 2025) — HIGH PRIORITY
- Uses Hankel singular value analysis to compress SSM state during training.
- Reduces state N=384 → N=92 (76% reduction) maintaining 98.8% accuracy.
- 3.17x training speedup. Compatible with diagonal A matrix (our architecture).
- Could compress our d_state=64 → d_state=16-24 without quality loss.

**F9: SSM State Quantization for Edge** (arxiv 2506.12480, June 2025)
- Heterogeneous QAT: A matrix + state at 8-bit, B/C/D at 4-bit, activations at 6-bit.
- Clips state to [-50, +50] to prevent divergence. 96% accuracy (vs 40% PTQ).
- CRITICAL INSIGHT: A matrix and state are MOST sensitive — must stay ≥8-bit.
- Our ternary weights already compressed; this addresses RUNTIME state memory.

**F12: In-Place Fused SSM Kernels** (May 2025) — IMMEDIATE WIN
- Fuse SSM recurrence (A*h + B*u) into single kernel with in-place state updates.
- Eliminate intermediate activation materialization. ~30-50% buffer reduction.
- We already have mega-kernels K1-K4 — extend with in-place ops.
- Low complexity: modify existing CUDA kernels.

**F14: NVMe/SSD Offload** (NeurIPS 2025, arxiv 2508.06978)
- Offload inactive layer weights to NVMe during inference.
- NVMe bandwidth 6-7 GB/s vs DRAM 50-100 GB/s. Latency hit.
- Enables 1B+ params on 12GB GPU. Energy-efficient for throughput tasks.
- Future relevance only — current 50M model fits in VRAM.

#### KEY PATTERNS
- SSM state is HIGHLY sensitive to quantization — must stay ≥8-bit even when weights are ternary
- Cross-layer sharing (F1) + structured sparsity (F2) can push below 1.25 bits/weight
- Tequila (F6) solves deadzone trapping with zero inference cost — immediate priority
- CompreSSM (F8) enables aggressive state dimension reduction via control theory
- In-place kernel ops (F12) are low-hanging fruit for our mega-kernel architecture
- mmap (F3) useful for deployment, not research phase

### Wave 2: Mathematical Paradigm Breakers (2 agents, Feb 2026)

#### WEIGHT STORAGE MATHEMATICAL ALTERNATIVES

| # | Technique | Bits/Weight | Exact? | Viability |
|---|-----------|-------------|--------|-----------|
| M1 | GF(3)-Rank Factorization | 0.3-1.5 | YES | MEDIUM |
| M2 | Hamming Covering Code Dictionary | 0.5-1.2 | YES | HIGH |
| M3 | Kronecker Sum Decomposition | 0.3-0.6 | APPROX | MEDIUM |
| M4 | Walsh-Hadamard Spectral | 0.3-1.5 | APPROX | LOW-MEDIUM |
| M5 | Context-Modeled Streaming Arithmetic | 0.6-1.2 | YES | MEDIUM-HIGH |

**M1: GF(3)-Rank Factorization with Integer Correction**
- Treat ternary matrix as element of GF(3) matrix algebra. Compute rank over finite field.
- Factorize W_3 = U_3 × V_3 over GF(3), then correct for integer-to-GF(3) carrying errors.
- y = U_Z × (V_Z × x) + Delta × x, where Delta is sparse (divisible by 3).
- If GF(3)-rank r << min(M,K): storage = r*(M+K)*1.585 bits + sparse correction.
- CRITICAL UNKNOWN: Do trained ternary matrices have low GF(3)-rank? Empirical question.
- Diagnostic: Gaussian elimination mod 3 on trained 50M model (~30 min).

**M2: Hamming Covering Code Row Dictionary** — HIGHEST PRIORITY
- View each row as point in ternary Hamming space Z_3^K.
- Build dictionary of p "codeword" rows. Store dictionary + sparse deltas per row.
- For p=64, K=1280, R_avg=128 (10% differ): ~0.66 bits/weight.
- Precompute dictionary outputs once per GEMV, then sparse corrections per row.
- ALWAYS exact. Worst case degrades to baseline (p=M, no compression).
- Can achieve sub-1-bit/weight if trained matrices have clustered rows.
- BOTH memory compression AND computational speedup possible.

**M3: Kronecker Sum Decomposition in Ternary Domain**
- Decompose W ≈ Σ_s (A_s ⊗ B_s) where A_s, B_s are smaller ternary matrices.
- For S=128, balanced split: ~0.31 bits/weight. Extreme compression.
- BUT: intermediate z = B_s × x is FP16, not ternary → loses add/subtract advantage.
- Trade-off: 20x memory compression vs 7x compute penalty (at S=128).
- Best for memory-constrained deployment where compute overhead acceptable.

**M5: Context-Modeled Streaming Arithmetic Coding** — PRACTICAL
- Store matrix as arithmetically coded bitstream with 2D context model.
- During GEMV, decode each weight on-the-fly (never materialize full matrix).
- Approaches conditional entropy H(W[i,j] | context) bits per weight.
- Decode overhead: 2-4x per weight (comparison + shift).
- On CPU (production target): decode cost may be FASTER due to reduced bandwidth.
- Diagnostic: Measure conditional entropy of trained weights (~15 min).

#### RUNTIME MEMORY MATHEMATICAL ALTERNATIVES

| # | Technique | Compression | Exact? | Viability |
|---|-----------|-------------|--------|-----------|
| M6 | Cross-Head Subspace Projection | 4-8x | APPROX | MEDIUM |
| M7 | Adaptive Block Floating-Point | 2-4x | Near-exact | HIGH |
| M8 | Windowed Rank-1 Decomposition | 1.5-2x | APPROX | LOW |
| M9 | Hierarchical Shared-Private State | 10-13x | YES (trained) | HIGH |
| M10 | Manifold Embedding (Nonlinear) | 10-500x | APPROX | SPECULATIVE |

**M6: Cross-Head Subspace Projection**
- B is shared across all heads → column space of all heads' states is common.
- Project onto low-rank basis U_r of this shared subspace.
- At r=8: 7.88x compression (262K → 33K floats per layer).
- Update: c_i = exp(dt*a_i)*c_i + dt*(U_r^T*B)*x_i^T. Basis updated periodically.
- Depends on effective rank of B-subspace (empirical question).

**M7: Adaptive Block Floating-Point State** — IMMEDIATE WIN
- Replace FP32 state with block floating-point (shared exponent + reduced mantissa).
- Adapt bit width per head based on decay rate: slow heads get more bits, fast less.
- Error bound: ||e_ss|| ≤ (scale/2^b) / (1-alpha_i).
- Periodic re-anchoring (FP32 recompute every K steps) bounds cumulative error.
- Rigorous error analysis. Straightforward implementation.

**M9: Hierarchical Shared-Private State** — BREAKTHROUGH POTENTIAL
- Partition heads into groups. Maintain shared "slow" state + private "fast" residual.
- Shared: full d_state, FP32, group-averaged input. Private: small d_state_fast, FP16.
- G=4 groups, d_state_fast=8: 12.8x compression AND 62.5% LESS compute.
- Justified by shared-B property: all heads see same B projection.
- ARCHITECTURE MODIFICATION — trained natively, zero approximation error.
- Quality impact must be validated via training experiment.

**M10: Manifold Embedding via Takens Theorem** — SPECULATIVE
- SSM state evolves on attractor set with low intrinsic dimensionality d_eff.
- Learn encoder/decoder to map between full state and d_eff coordinates.
- If d_eff=32 vs d_state*head_dim=16384: 512x theoretical compression.
- Requires: (1) measure intrinsic dimensionality, (2) learn dynamics in compressed space.
- Periodic re-anchoring needed to bound drift.

#### KEY CROSS-POLLINATION INSIGHTS
- **M2 (Dictionary) + M5 (Arithmetic Coding):** Dictionary separates predictable/residual; arithmetic coding compresses residuals further
- **M9 (Hierarchical) + M7 (BFP):** Compose for 26-30x total: structural 13x × quantization 2x
- **M6 (Subspace) validates M9:** If B-subspace has low rank → heads ARE redundant → grouping works
- **M7's error formula is universal:** ||e_ss|| ≤ quantization_error / (1-alpha) applies to ALL approximate paths

#### RECOMMENDED DIAGNOSTICS (zero-cost, run on trained 50M model)
1. GF(3) rank of all weight matrices (~30 min)
2. Pairwise Hamming distances + k-medoids clustering (~1 hour)
3. Conditional entropy H(W[i,j] | context) (~15 min)
4. Per-head decay rate distribution from trained A_log values (~5 min)
5. Intrinsic dimensionality of SSM state trajectory (~30 min)

### Wave 3: Paradigm Breakers (2 agents, Feb 2026)

#### RADICAL WEIGHT MEMORY APPROACHES

| # | Technique | Reduction | Key Insight | Feasibility |
|---|-----------|-----------|-------------|-------------|
| P1 | Butterfly-Factored Ternary | 100-1000x weights | Implicit representation via O(n log n) factors | MEDIUM-HIGH |
| P2 | Weight Trajectory Compression | 8-24x exact | Layers are permutation+sign transforms of a base | LIKELY |
| P3 | Boolean Circuit Compilation (CSE) | 10-20x | Model is a circuit, not data — compile it | MEDIUM |
| P4 | Demand-Paged Hypernetwork | 3-7x | Generate weights on-the-fly from tiny network | MEDIUM (GPU) |

**P1: Butterfly-Factored Ternary Matrices** — PARADIGM SHIFT
- Factor W as product of k alternating circulant + diagonal ternary matrices.
- Each circulant defined by ONE row (n values). k=3 factors for n=4096.
- Storage per matrix: 6n ternary values = 6 KB vs 8 MB brute-force = 1300x per matrix.
- Compute: O(n log n) via FFT vs O(n^2) brute-force = 113x fewer operations.
- At 1B: ~2.9 MB ternary weights + ~70 MB FP16 params = ~73 MB total (vs 370 MB = 5x).
- Mathematical foundation: Kaleidoscope/Monarch matrices (Dao 2020).
- Risk: FFT intermediates are FP, partially defeating "no multiply" advantage.
- REQUIRES TRAINING with butterfly factors (not post-hoc).

**P2: Weight Trajectory Compression** — HIGHEST PRIORITY
- Layers are not independent: they evolved from same initialization under same loss.
- Store ONE base matrix W_0 + lightweight transformation T per layer.
- T consists of: row permutation + column permutation + sign modulation.
- W_l[i,j] = s_l[i] * t_l[j] * W_0[sigma_l(i), tau_l(j)] + C_l[i,j] (sparse correction).
- If correction rate k < 5%: storage = base (8 MB) + 48 × corrections (40 MB) = 48 MB.
- EXACT computation. 8-24x compression depending on correction rate.
- ZERO-COST DIAGNOSTIC: Compare Hamming distance between layer weight matrices after permutation alignment (~10 min on existing 50M checkpoint).

**P3: Boolean Circuit Compilation (CSE)**
- After training, the model is a FIXED arithmetic circuit of additions/subtractions.
- Common Subexpression Elimination: shared partial sums across output rows.
- If unique patterns < 50% of rows: 10-20x memory + potential speedup.
- Direct path to FPGA synthesis: compiled circuit maps to LUT configurations.
- On CPU: compiled to sequence of SIMD add/subtract instructions.
- ZERO-COST DIAGNOSTIC: Count unique rows per matrix in trained model (~10 min).

**P4: Demand-Paged Hypernetwork**
- Tiny hypernetwork G (~2.5 MB) generates each layer's weights on demand.
- Only ONE layer's weights in memory at a time (~4 MB buffer).
- Total: 2.5 MB (G) + 4 MB (buffer) + 50 MB (FP16) = ~57 MB at 1B.
- Risk: Generation must be faster than DRAM read. Feasible on GPU, not CPU.
- More of a serving optimization than a research priority.

#### RADICAL RUNTIME MEMORY APPROACHES

| # | Technique | Reduction | Key Insight | Feasibility |
|---|-----------|-----------|-------------|-------------|
| P5 | Cross-Layer State Collapse | 100-5000x state | All layers share ONE dynamical system | LIKELY (modest) |
| P6 | Metabolic Architecture | 1.2-1.5x total | Variable expansion ratio per layer | LIKELY |
| P7 | Linearized State-Machine | 2x + 77K tok/s | Compile 48 layers into one matrix multiply | WILD |

**P5: Cross-Layer State Collapse**
- Instead of 42 independent states, maintain ONE master state H with K decay channels.
- Per-layer: read state via ternary projection P_j, write via Q_j.
- Extreme: K=8, d_master=256 → state = 8 KB (vs 44 MB = 5500x).
- Conservative: groups of 7 layers → 7x state reduction. More realistic.
- Quality risk: layers need enough independent information for representation.

**P6: Metabolic Architecture (Variable-Width Layers)**
- Not all layers need expand=2. Use bell-curve profile across depth.
- Early/late layers narrow (expand=1.0), middle layers wide (expand=2.0).
- Conservative (expand_min=1.0): 21% weight reduction.
- Aggressive (expand_min=0.5): 34% weight reduction.
- Well-validated in vision (hourglass architectures). Requires training.
- Compounds with ALL other compression techniques.

**P7: Linearized State-Machine Inference** — MOONSHOT
- Compile all 48 layers into a single matrix M: new_state = M × [state; embedding].
- For d_eff=512: M is 512×2560 = 2.6 MB. Total model ~203 MB.
- Speed: ONE matrix-vector multiply per token = 77,000 tok/s theoretical.
- State becomes 2 KB → 6 million concurrent sessions in 12 GB.
- Risk: SSM dynamics are linear, but SiLU/softmax make full model nonlinear.
- ZERO-COST DIAGNOSTIC: Measure d_eff via PCA on state trajectories (~1 hour).

#### META-INSIGHTS FROM PARADIGM BREAKERS
> **Key reframing:** "The ACTUAL memory problem is weights (91% at 1B), not state (6-10%)."
> State only dominates DYNAMIC memory. Total memory is dominated by weight storage.
>
> **Most important diagnostic to run (zero-cost, determines everything):**
> 1. GF(3) rank of trained weight matrices (~30 min) — if low, 5x FREE compression
> 2. Inter-layer Hamming distance after permutation alignment (~10 min) — if low, 8-24x compression
> 3. PCA intrinsic dimensionality of SSM states (~1 hour) — determines state compression ceiling
>
> **Killer combination: P1 (Butterfly) + P5 (State Collapse) at 7B:**
> Weights: 2.23 GB → ~15 MB (150x), State: 130 MB → ~32 KB (4000x), FP16: ~270 MB
> Total: 2.9 GB → ~350 MB (8x). With INT8 FP16 params: ~200 MB for a 7B model.

#### CROSS-WAVE SYNERGIES (Memory)
- **F1 (Basis Sharing) + P2 (Trajectory):** SVD-shared basis IS trajectory compression in linear algebra form
- **F6 (Tequila) + P6 (Metabolic):** Fewer parameters in narrow layers → less deadzone trapping
- **M2 (Hamming Dict) + P3 (Circuit CSE):** Dictionary entries ARE common subexpressions
- **M9 (Hierarchical State) + P5 (State Collapse):** Hierarchical IS a conservative version of collapse
- **M7 (Adaptive BFP) + M9 (Hierarchical):** Compose: structural 13x × quantization 2x = 26x

---

## Aspect 2 Summary: Memory

**Total findings:** 16 web research (F1-F16) + 10 mathematical (M1-M10) + 7 paradigm shifts (P1-P7) = **33 findings**

**Top 5 by estimated impact:**
1. **P1 Butterfly-Factored Ternary** — 100-1000x weight compression + O(n log n) compute
2. **P2 Weight Trajectory** — 8-24x exact, zero quality loss, zero-cost diagnostic
3. **M9 Hierarchical Shared-Private State** — 10-13x state, trained natively, REDUCES compute
4. **M2 Hamming Covering Code Dictionary** — Sub-1-bit/weight storage, always exact
5. **M5 Context-Modeled Streaming Arithmetic** — Approaches conditional entropy, lossless

**Zero-cost diagnostics (run on existing 50M checkpoint):**
1. GF(3) rank (30 min) — determines if 5x free compression exists
2. Inter-layer Hamming distance (10 min) — determines if 8-24x trajectory compression works
3. Conditional entropy (15 min) — determines streaming arithmetic potential
4. Pairwise row Hamming distances + k-medoids (1 hour) — determines dictionary compression
5. State intrinsic dimensionality via PCA (1 hour) — determines state compression ceiling
