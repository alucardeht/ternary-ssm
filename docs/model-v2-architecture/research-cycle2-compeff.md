# Research Cycle 2 — Computational Efficiency Findings

## Step 1: Online Research (2 researchers, 2026-02-26)

### Researcher 1: Mixed-Precision & Hardware Co-Design

**R1-CE2-01: TWEO — Transformers Without Extreme Outliers (Nov 2025)**
- Loss regularizer suppressing extreme activation outliers for stable FP8 training
- L_TWEO = (1/L) Σ E[(|A^(l)|/(τ+ε))^p] with τ=3.0, p=4
- FP8 matches BF16 within 0.1% loss, +36% throughput on A100
- Applicable: suppress outliers in FP16 components of ternary training
- Effort: LOW (add regularizer term)

**R1-CE2-02: MoR — Mixture of Representations (Dec 2025)**
- Dynamic per-tensor precision: automatically selects FP8 vs BF16
- 98.38% of tensors in FP8 while preserving quality
- Per-tensor statistics determine optimal precision
- Applicable: ternary weights (2-bit) + dynamic FP8/BF16 activations
- Effort: MEDIUM (per-tensor analysis infrastructure)

**R1-CE2-03: NVFP4 Training — 4-bit Hierarchical Scaling (2025)**
- 4-bit format with 2-level hierarchical scaling (16×16 blocks)
- 1.59x speedup vs BF16, stable convergence (Llama 3 8B, 1T tokens)
- CRITICAL: final 4 transformer layers MUST stay BF16 (diverges otherwise)
- Applicable: ternary weights + NVFP4 activations + BF16 final layers
- Effort: HIGH (new format infrastructure)

**R1-CE2-04: Unified FP8 Train-Inference (FP8-TI)**
- Use same precision for training AND inference rollouts
- Mixed precision (BF16 train + FP8 infer) causes growing mismatch at scale
- CRITICAL for ternary: FP16 shadows in training vs pure ternary at inference = growing gap
- Solution: "unified ternary" — always quantize in forward pass
- Effort: LOW (we already do this via STE, validates our approach)

**R1-CE2-05: MXFP8 — Microscaling FP8 (2025)**
- Per-32-element block scaling (finer than per-tensor)
- Native Blackwell (B100/B200) support via MXFP8 Tensor Cores
- Matches BF16 convergence, outperforms standard FP8
- Applicable: non-ternary ops (SSM state, attention) on future hardware
- Effort: LOW (framework-level, when hardware available)

**R1-CE2-06: E8M0 Scaling — Power-of-2 Scales**
- Replace FP32 scales with powers-of-2 (bit-shift instead of multiply)
- Hardware-friendly on all platforms including CPU inference
- Applicable: simplify ternary group scale computation
- Effort: LOW | Risk: MEDIUM (precision trade-off)

**R1-CE2-07: MemAscend — SSD Offloading Optimization**
- System memory optimization for SSD offloading of optimizer states
- Enables larger batch sizes when VRAM exhausted
- Applicable: 1B training when optimizer exceeds 12GB VRAM
- Effort: MEDIUM

**R1-CE2-08: AutoHete — Heterogeneous Scheduling**
- Automatic CPU/GPU/SSD scheduling for activation checkpointing
- CPU for lightweight ops (quantization), GPU for heavy ops (matmul)
- Applicable: ternary quantization on CPU while GPU does forward/backward
- Effort: HIGH

### Researcher 2: Training Paradigm Innovations

**R2-CE2-01: Token Order Prediction (TOP) — Simpler MTP (2025)**
- Predict token ORDER instead of exact tokens (learning-to-rank loss, ListNet)
- Only 1 extra unembedding layer vs MTP's N transformer blocks
- Outperforms standard MTP on 340M/1.8B/7B across NLP benchmarks
- +2-5% reasoning quality with minimal overhead
- Effort: LOW-MEDIUM

**R2-CE2-02: DeepSeek Sequential MTP — Coherent Multi-Token**
- Sequential MTP heads with causal dependency (vs independent)
- Input: RMSNorm(h_{k-1}) concat RMSNorm(Emb(t_{i+k})) → linear projection
- 85-90% acceptance rate for 2nd token, 1.8x inference via speculative decoding
- Effort: MEDIUM

**R2-CE2-03: Efficient Optimizer Variants (2025-2026)**
- SOAP: improved convergence for quantized training
- Muon: momentum-based, potentially better for ternary
- Prodigy: adaptive LR without manual tuning
- Schedule-Free Adam: eliminates LR scheduling overhead
- Effort: LOW (drop-in replacements) | Need: deeper paper analysis

**R2-CE2-04: Tensor Core INT4 Mapping for Ternary**
- RTX 5070 has native FP4 support; ternary {-1,0,+1} maps to INT4
- Could leverage Tensor Core acceleration for ternary GEMV
- Currently using add/sub kernels; INT4 Tensor Cores might be faster
- Effort: HIGH (kernel rewrite) | Potential: 2x+ compute throughput

**R2-CE2-05: Synthetic Data for Pretraining (Phi-style)**
- Textbook-quality datasets achieve strong results with <10B tokens
- TinyStories, FineWeb-Edu, Dolma curated subsets
- Could reduce training tokens needed, test ternary gap on cleaner data
- Effort: MEDIUM (data pipeline changes)

**R2-CE2-06: Curriculum MTP Training (2025)**
- Forward curriculum: NTP → gradually increase to MTP predictions
- Enables small models (<1B) to leverage MTP effectively
- Applicable: progressive introduction of auxiliary objectives
- Effort: MEDIUM

**R2-CE2-07: Operator Fusion for Training Backward**
- Fused backward passes: combine gradient computation across ops
- Triton-based fused forward+backward in single kernel launch
- Target: sequential scan bottleneck in SSM training
- Effort: HIGH

**R2-CE2-08: Critical Batch Size Theory**
- Optimal batch size where gradient noise ≈ model noise
- Classic theory likely sufficient for current scale
- At 50M: batch=32-64 optimal (confirmed by Phase 5 results)
- Effort: N/A (validated)

### Priority Ranking (Cycle 2 CompEff)

| Rank | Finding | Impact | Effort | Priority |
|------|---------|--------|--------|----------|
| 1 | R2-CE2-01 TOP training | +2-5% quality, minimal overhead | LOW-MEDIUM | HIGH |
| 2 | R1-CE2-04 Unified ternary | Validates our STE approach | LOW | Confirmed |
| 3 | R1-CE2-01 TWEO regularizer | Stabilizes FP16 components | LOW | MEDIUM |
| 4 | R2-CE2-03 New optimizers | Potential convergence gains | LOW | Research needed |
| 5 | R2-CE2-04 Tensor Core INT4 | 2x+ compute on ternary | HIGH | Future hardware |
| 6 | R1-CE2-02 MoR precision | Adaptive activation precision | MEDIUM | MEDIUM |
| 7 | R2-CE2-05 Synthetic data | Faster convergence | MEDIUM | MEDIUM |
| 8 | R1-CE2-03 NVFP4 activations | 1.59x speedup | HIGH | Future hardware |

### Key Patterns

1. **Mixed-precision is going heterogeneous** — per-tensor, per-layer precision selection is the 2025-2026 trend
2. **Train-inference consistency matters at scale** — validates our STE-based approach (always quantize in forward)
3. **TOP > MTP for small models** — single extra layer vs N transformer blocks, better quality/cost ratio
4. **Hardware co-design accelerating** — FP4/FP8 native support means ternary could leverage Tensor Cores
5. **Final layers need higher precision** — NVFP4 diverges without BF16 final layers (may apply to ternary too)

## Step 2: Mathematical Analysis (2 mathematicians, 2026-02-26)

### Mathematician 1: Training Computational Efficiency

**M1-CE2-01: Ternary Backward Kernel — 4.5x Bandwidth Reduction**
- Forward uses add/sub but backward dispatches to cuBLAS FP16
- grad_x = W_q^T * grad_y is TERNARY (same structure as forward, transposed)
- grad_W = grad_y * x^T is FP16 (outer product for shadow weight update)
- Bandwidth: FP16 backward reads 576KB vs ternary backward reads 128KB = 4.5x reduction
- Overall backward speedup: 1.64x (grad_x 4.5x faster, grad_W stays same)
- Full training step speedup: **1.3x** (backward is ~60% of total)
- Stability: exact same gradients (no approximation)
- Effort: MEDIUM (custom CUDA kernel for transposed ternary GEMV)

**M1-CE2-02: SSM Parallel Scan — 1.6-3.0x Training Speedup**
- Sequential scan h_t = A*h_{t-1} + B*x_t reformulated as parallel prefix
- Affine transforms form monoid: (A2,b2)∘(A1,b1) = (A2*A1, A2*b1+b2)
- With diagonal A (Mamba-2): each combine O(d_state), depth O(log T)
- T=1024: depth 10 steps vs 1024 sequential (102x depth reduction)
- Chunked SSD: optimal chunk C=sqrt(T)=32, depth 64 vs 1024
- GPU utilization: 30-50% → 70-85% projected
- SSM scan is 40-50% of training time → **1.6-3.0x overall training speedup**
- Effort: HIGH (fused CUDA kernel for chunked parallel scan + backward)

**M1-CE2-03: INT8 Gradient Quantization — "Free" for Ternary**
- STE gradient noise already ~5-15% relative error (dominates)
- INT8 quantization adds only ~1.5% relative noise (negligible on top of STE)
- Mathematical proof: delta^2_quant << b^2_STE + sigma^2_minibatch
- Memory savings: 50M * (2-1) bytes = **50 MB** gradient memory freed
- KEY INSIGHT: "Noise budget" framework — STE already spent most of the noise budget
- Any additional noise below STE magnitude is essentially free
- Effort: LOW (drop-in INT8 quantization of gradients)

**M1-CE2-04: Optimal Mixed-Precision Allocation — 43% Memory Reduction**
- Optimal per-component precision derived via Lagrange optimization
- FP8 activations: -50% activation memory (medium sensitivity, sufficient precision)
- INT8 gradients: -50% gradient memory (low sensitivity, STE dominates)
- BF16 optimizer states: -50% optimizer memory (m and v track EMAs)
- Shadow weights: MUST stay FP32 (BF16 causes gradient underflow at lr*grad ~5e-6)
- Total: 1500 MB → 847 MB = **43% memory reduction** at 50M
- Enables larger batch sizes or longer sequences within same VRAM
- Effort: MEDIUM (implement each component separately, verify convergence)

**M1-CE2-05: Effective FLOPs Accounting — Ternary = 62.7% of FP16**
- Standard FLOPs assume multiply-accumulate (2 FLOPs/weight)
- Ternary forward: {skip, add, sub} = (1-f_0) FLOPs/weight = 0.667 FLOPs/weight
- Per training step: forward (0.667) + grad_x (0.667) + grad_W (2.0) = 3.333 vs 6.0
- 84% ternary + 16% FP16 → **62.7% of equivalent FP16 FLOPs** per step
- Weight bandwidth efficiency: 2.4x (6.0/2.5 bytes)
- NEW METRIC — Effective Ops Per Byte (EOPB): ternary 2.667 vs FP16 1.0 = 2.667x
- Effort: N/A (analytical result for paper)

**M1-CE2-06: Stochastic Ternary Shadow Accumulation (SPECULATIVE)**
- Replace FP32 shadow weights with INT16 counter + stochastic ternary updates
- E[delta_t] = -lr*g_t/epsilon (correct in expectation)
- Variance issue: diverges without batching (random walk)
- With K=4-16 gradient accumulation: variance manageable
- Memory savings: 168 MB → 84 MB (INT16 vs FP32 for 42M ternary params)
- Effort: HIGH | Risk: convergence uncertain

**M1-CE2-07: Discrete Coordinate Descent — Eliminate Shadow Weights (SPECULATIVE)**
- Second-order STE: score(v) = (v-w)*g + 0.5*(v-w)^2*h for v in {-1,0,+1}
- Diagonal Hessian h = x_j^2 * grad_y_i^2 (free, already computed)
- Probabilistic flipping via Boltzmann: P(v) ∝ exp(-score(v)/T)
- Eliminates shadow weights (168MB), optimizer m+v (336MB) = **504 MB total**
- Major concern: no gradient accumulation without shadow weights
- Effort: HIGH | Risk: convergence uncertain, local minima

### Mathematician 2: Hardware-Aware Efficiency

**M2-CE2-01: Tensor Core INT4 — Only at Scale (d≥2048)**
- RTX 5070 INT4 Tensor Core: ~988 TOPS
- Critical batch for Tensor Core saturation: B_crit = 1470d/(8d-23520)
- At d=512 (50M): IMPOSSIBLE to saturate (negative B_crit)
- At d=4096 (1B): B_crit = 651 (impractical for autoregressive)
- Root cause: ternary weights so small (0.25 bytes/param) that activation I/O dominates
- Add/sub GEMV on shader cores already optimal at d=512
- Effort: HIGH (only justified at 1B+ with batched inference)

**M2-CE2-02: L2 Residency — Entire 50M Model in L2 Cache**
- Model memory: 21.3 MB (layers) + 32 MB (LM head FP16) = 53.3 MB
- RTX 5070 L2: 48 MB → need to quantize LM head
- With INT4 LM head: 21.3 + 8 = 29.3 MB → **FITS in L2**
- L2 bandwidth ~12 TB/s vs DRAM 672 GB/s = **18x effective bandwidth**
- Theoretical max: 322K tok/s (L2 BW ceiling) vs 18K (DRAM ceiling)
- Current 876.8 tok/s = only 4.9% of DRAM ceiling
- With L2 residency: becomes COMPUTE-BOUND (not BW-bound!)
- Effort: MEDIUM (quantize LM head, persistent kernel)

**M2-CE2-03: Zero-Unpack Dual Bitmask (DBMK) Format**
- Current: 2-bit packed → unpack to INT8/FP16 → compute (overhead)
- Proposed: store as separate pos_mask and neg_mask (uint32 per 32 weights)
- Warp-predicated computation: 8 ops per 32 elements vs 64 (unpack+FMA)
- Naturally skips zeros (33% of weights): 21.3 vs 32 iterations
- Same 2 bits/weight storage, eliminates unpack kernel entirely
- Computation operates directly on bitmasks via shift+AND+select+warp_reduce
- Effort: MEDIUM (reformat storage, new kernel)

**M2-CE2-04: Scaling Analysis — Kernel Overhead is THE Bottleneck**
- At 50M + L2 resident: OI=7.76 > L2 ridge(1.28) → COMPUTE-BOUND
- Current bottleneck: 264 kernel launches × 0.76μs = 200μs overhead = **67% of total time**
- Useful compute: ~100μs. Overhead dominates.
- Even at 200M streaming from DRAM: compute-bound at batch≥3 (shader cores)
- Scaling trajectory: GPU util improves with scale (larger matrices amortize launch cost)
- KEY: optimize kernel count, not bandwidth utilization
- Effort: HIGH (mega-kernel fusion, but highest impact)

**M2-CE2-05: Power-of-2 Scales — NOT Worth Pursuing**
- Scale multiply: 1 op per 128 elements = 0.78% of compute (negligible speedup)
- Power-of-2 constraint: RMSE 20.4% relative error on scale factors
- With N=4 sub-levels: RMSE 5.1% but still requires one FP16 multiply
- Verdict: negligible throughput gain, substantial quality loss
- Effort: LOW | Impact: NEGATIVE (quality degradation)

**M2-CE2-06: Multi-GPU Communication — PP over TP**
- TP all-reduce on ACTIVATIONS (not weights) → ternary provides NO benefit
- Communication-to-computation ratio: ternary 8x HIGHER than FP16
- TP efficiency at 50M: 82.5% (2 GPU) vs FP16 97.4%
- PP efficiency: only 1 pipeline boundary vs 24 all-reduces → **24x less communication**
- Minimum model size for efficient TP: >300M (2 GPU), >600M (4 GPU)
- But ternary models fit in fewer GPUs (8x compression) → often don't NEED multi-GPU
- Effort: LOW (architectural choice, not implementation)

### Priority Ranking (Step 2)

| Rank | Finding | Impact | Viability |
|------|---------|--------|-----------|
| 1 | M2-CE2-04 Kernel fusion | Eliminate 67% overhead | HIGH |
| 2 | M1-CE2-02 SSM parallel scan | 1.6-3.0x training speedup | HIGH |
| 3 | M2-CE2-02 L2 residency | 18x effective bandwidth | HIGH |
| 4 | M1-CE2-04 Mixed-precision | 43% memory reduction | HIGH |
| 5 | M1-CE2-01 Ternary backward | 1.3x training step speedup | HIGH |
| 6 | M2-CE2-03 DBMK format | Eliminate unpack overhead | HIGH |
| 7 | M1-CE2-03 INT8 gradients | 50MB free savings | HIGH |
| 8 | M1-CE2-05 Effective FLOPs | Correct metrics for paper | HIGH (analytical) |
| 9 | M2-CE2-06 PP over TP | 24x less multi-GPU comm | HIGH (architectural) |
| 10 | M1-CE2-07 Discrete coord descent | 504MB savings, no shadows | SPECULATIVE |

### Key Mathematical Insights

1. **"Noise Budget" Framework:** STE gradient noise (5-15%) is the dominant noise source. Any additional quantization noise below STE magnitude is FREE. This makes INT8 gradients, FP8 activations, and BF16 optimizer states all viable simultaneously.
2. **Compute-Bound When L2-Resident:** At 50M, with model in L2 cache, the system flips from bandwidth-bound to COMPUTE-bound. This fundamentally changes the optimization target.
3. **Kernel Overhead Dominates:** 67% of current inference time is kernel launch overhead (200μs/300μs). Not bandwidth, not compute. Mega-kernel fusion is THE priority.
4. **Ternary Backward is Half-Exploitable:** grad_x uses ternary structure (4.5x BW reduction), grad_W is FP16 (unavoidable). Combined: 1.3x training speedup.
5. **Multi-GPU Hurts at Small Scale:** Ternary computation is so cheap that communication overhead dominates earlier. PP strongly preferred over TP. Don't use multi-GPU below 300M.
6. **Effective FLOPs = 62.7%:** Ternary training does 37.3% fewer operations than FP16, plus 2.4x less bandwidth. Combined training efficiency advantage = 2.5-4x with all optimizations.

## Step 3: Paradigm Breaking (2 paradigm-breakers, 2026-02-26)

### Paradigm-Breaker 1: Training Paradigm Shifts

**P1-CE2-01: Codebook Ternary Training — 360x Optimizer Memory Reduction**
- Instead of 42M individual shadow weights + Adam state, share K=256 ternary codebook patterns
- Each weight group (G=128) gets codebook index + FP16 scale
- Forward: y_g = alpha_g * (C[i_g] . x_g), backward: gradients aggregate to K entries
- Memory: 504 MB (current) → 1.4 MB (codebook) = **360x reduction** at 50M
- At 1B: 10 GB → 30 MB = 333x reduction. Enables 1B training on 12GB VRAM
- Speed: 4-7x from larger batches + reduced optimizer overhead
- Index reassignment every M=100 steps via K-nearest (K=256, trivial cost)
- Risk: MEDIUM (codebook expressiveness, gradient sharing quality)
- Novelty: PQ for inference is known (GPTQ), PQ AS training representation is NEW

**P1-CE2-02: Teacher Distillation + Combinatorial Layer Matching — 2x Faster**
- Phase 1: Train same-size FP16 model for 20% of total iterations (0.85x per-step)
- Phase 2: Layer-wise quantize to ternary via closed-form matching (1% cost)
- Phase 3: STE fine-tune for 60% of iterations (1.0x per-step)
- Total cost: 17% + 0.5% + 60% = ~48% of current = **2x faster training**
- Key: FP16 as SEARCH ACCELERATOR (finds neighborhood), ternary snaps to discrete optimum
- Not "warmup" (already proven unnecessary) — this is initialization acceleration
- Risk: LOW (each phase uses proven techniques)

**P1-CE2-03: Group-Adaptive TAF — Eliminate Shadow Weights, 5.3x Memory**
- State per weight: w_i in {-1,0,+1} (2-bit) + BF16 accumulator (2 bytes)
- State per group: v_g in FP16 (group variance, Adam-like adaptivity)
- Update: effective_i = EMA(grad) / sqrt(group_variance), flip when > threshold
- No shadow weights (168MB gone), no per-param optimizer (336MB gone)
- Total: 504 MB → 95 MB = **5.3x reduction**
- Speed: 2-3x (primarily from batch size increase with freed memory)
- Key insight: adaptivity at GROUP level (328K groups vs 42M params) preserves Adam-like convergence
- Risk: MEDIUM (existing TAF variant not competitive, but lacked group adaptivity)

**P1-CE2-04: Sign-Mask Decomposition Training — Better Convergence**
- Decompose w = sign(s) * mask(m): two binary sub-problems
- Phase 1 (30% budget): train signs only {-1,+1} (no zeros, faster binary STE)
- Phase 2 (10% budget): one-shot mask selection via importance scoring (SynFlow)
- Phase 3 (60% budget): joint STE fine-tune from better initialization
- Total cost: ~86% of baseline = **15-20% speedup**
- Real win: better convergence because each sub-problem is easier
- Risk: MEDIUM (phase transitions, mask-sign interaction)

**P1-CE2-05: Decoupled Layer-Parallel Training — 24x Backward Memory**
- Each layer trains with LOCAL objective: predict next layer's input
- L_local = ||y_l - sg(x_{l+1})||^2 + lambda * self-consistency
- Backward passes PARALLEL across layers (O(1) vs O(L) memory per layer)
- Activation memory: 24x reduction (800 MB → 33 MB at 50M)
- SSM-specific advantage: state compression = natural local objective
- Speed: 30-40% backward speedup (single GPU), 2-3x with 4 GPUs (layer parallelism)
- Risk: HIGH (representation collapse, target staleness, quality degradation)

### Paradigm-Breaker 2: Hardware Paradigm Shifts

**P2-CE2-01: Bit-Plane Activation Decomposition — True Matmul-Free**
- Quantize activations to 8-bit fixed-point, decompose into B=8 bit-planes
- Each bit-plane × ternary weight = XNOR + AND + POPC (pure integer logic)
- Processes 32 weight-activation pairs per cycle per CUDA core
- INT pipelines and FP pipelines are INDEPENDENT on GPU → overlap with SSM recurrence
- Estimated: 2-3x per GEMV, eliminates ALL floating-point multiply from ternary ops
- Opens path to NPU deployment (integer-only hardware)
- Risk: MEDIUM (INT8 activation quality for SSM state projections)

**P2-CE2-02: Persistent Cooperative Forward-Pass Kernel — 5-8x at 50M**
- Single cudaLaunchCooperativeKernel for ENTIRE forward pass
- Weights loaded into L2 ONCE, stay resident across all tokens
- Inter-layer sync via cooperative_groups::grid::sync() (not CPU return)
- Activations in shared memory, never touch DRAM
- L2 bandwidth 12 TB/s vs DRAM 672 GB/s = 18x effective bandwidth
- Scale limitation: full residency only at <200M (48MB L2)
- Risk: MEDIUM (debugging difficulty, no cuDNN/cuBLAS inside persistent kernel)

**P2-CE2-03: Trit Packing (20 Trits Per uint32) — 25% Bandwidth, Lossless**
- log2(3) = 1.585 bits/weight, current 2-bit wastes 26% bandwidth
- Pack 20 ternary values per uint32 (3^20 = 3.49B < 2^32 = 4.29B)
- Decode via hierarchical LUT: split into 4 groups of 5 trits, 243-entry LUT per group
- LUT fits in L1 (972 bytes), returns pre-computed partial sums
- 25% bandwidth reduction, stacks multiplicatively with ALL other optimizations
- Proven concept: bitnet.cpp uses same technique on CPU
- Risk: LOW (lossless encoding, bounded implementation)

**P2-CE2-04: Adaptive Early Termination — Ternary-Unique, 2-3x LM Head**
- Ternary-unique property: each weight contribution bounded by |x[j]|
- After processing S of K inputs: tight upper/lower bounds on all outputs
- If leader's lower_bound > all others' upper_bound → STOP
- LM head: 80% of tokens terminate after 15-25% of computation = 4-6x speedup
- LM head is 28-33% of compute → **1.6-2.0x net inference speedup**
- IMPOSSIBLE with FP16 weights (unbounded contributions prevent tight bounds)
- Risk: LOW for LM head (exact computation), MEDIUM for hidden layers

**P2-CE2-05: 2:4 Structured Sparsity on Tensor Cores — 100x Batched**
- Current ~42% zeros → train with 2:4 constraint (50% zeros, 8% more)
- NVIDIA hardware sparse acceleration: 2x Tensor Core throughput
- Batch=1: 1.5-2x improvement
- Batch=4-8 (speculative decoding): **10-30x** over Triton GEMV
- Batch=64 (serving): **50-100x** (Tensor Cores at near-peak)
- Makes ternary inference FASTER than FP16 dense at any batch size
- Risk: MEDIUM (quality impact of 8% more zeros, needs Blackwell cuSPARSE)

### Priority Ranking (Step 3)

| Rank | Finding | Impact | Risk | Priority |
|------|---------|--------|------|----------|
| 1 | P2-CE2-03 Trit packing | 25% BW, lossless, stacks | LOW | IMMEDIATE |
| 2 | P1-CE2-01 Codebook training | 360x memory, 4-7x speed | MEDIUM | HIGH |
| 3 | P2-CE2-04 Early termination | 2-3x LM head, ternary-unique | LOW | HIGH |
| 4 | P1-CE2-02 Teacher+matching | 2x training, safe path | LOW | HIGH |
| 5 | P2-CE2-02 Persistent kernel | 5-8x at 50M | MEDIUM | HIGH |
| 6 | P1-CE2-03 GA-TAF | 5.3x memory, no shadows | MEDIUM | MEDIUM |
| 7 | P2-CE2-05 2:4 sparsity | 100x batched | MEDIUM | MEDIUM |
| 8 | P2-CE2-01 Bit-plane XNOR | 2-3x, true matmul-free | MEDIUM | RESEARCH |
| 9 | P1-CE2-04 Sign-mask decomp | 15-20%, better convergence | MEDIUM | RESEARCH |
| 10 | P1-CE2-05 Layer-parallel | 24x backward memory | HIGH | RESEARCH |

### Key Paradigm Shifts

1. **SGD on continuous surrogates is NOT the only option:** Codebook training (P1-CE2-01) and GA-TAF (P1-CE2-03) both bypass shadow weights entirely, treating ternary training as the DISCRETE optimization problem it fundamentally is.
2. **Memory inflation is the real training bottleneck:** 12-16 bytes per ternary weight during training (shadow + Adam) vs 0.25 bytes at inference = 48-64x inflation. Codebook reduces this to ~0.003 bytes/param.
3. **Ternary computation IS conditional set accumulation, NOT linear algebra:** Bit-plane decomposition (P2-CE2-01) and early termination (P2-CE2-04) exploit this by using integer logic and bounded contributions.
4. **25% of current bandwidth is wasted:** Trit packing (P2-CE2-03) exploits log2(3)=1.585 vs 2.0 bits. Free 25% improvement, proven on CPU.
5. **Structured sparsity unlocks Tensor Cores:** 2:4 sparsity (P2-CE2-05) bridges the gap between ternary and hardware acceleration, potentially making ternary FASTER than FP16 for batched inference.
