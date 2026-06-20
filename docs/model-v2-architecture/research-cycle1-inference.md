# Research Cycle 1 — Inference Optimization Findings

## Step 1: Online Research (2 researchers, 2026-02-26)

### Researcher 1: Serving & Framework-Level Optimizations

**R1-IF-01: Marconi — Prefix Caching for Hybrid SSMs (MLSys 2025)**
- First prefix caching system designed for hybrid models (attention + SSM)
- Smart admission/eviction based on reuse forecasts and compute-vs-memory tradeoffs
- 34.4x higher token hit rates, 71.1% lower TTFT vs baseline
- Directly applicable: our 12.5% attention layers need prefix caching
- Effort: MEDIUM (serving framework integration)

**R1-IF-02: FlashAttention-3 with Tensor Memory Accelerator**
- Fused attention with TMA for automated HBM-SRAM movement (Hopper+)
- 1.3-1.4x throughput over FA2 on attention layers
- Applies to our 12.5% attention layers — but requires H100+ hardware
- Effort: LOW (use FA3 library) | Requires: Hopper+ GPU

**R1-IF-03: INT4/INT8 Activation Quantization + Ternary Weights**
- W3A8 combination: ternary weights + INT8 dynamic activations
- 1.5x TPOT improvement, 3.5x memory reduction (Quamba2 benchmarks)
- Also quantize 16% FP16 weights to INT4 at inference
- Effort: MEDIUM (calibration dataset, kernel modifications)

**R1-IF-04: Triton Mega-Kernels / Liger-Kernel Style Fusion**
- Fuse RMSNorm + selective scan + SiLU into single kernel
- 7x for RMSNorm, 8x for RoPE, 3x for fused ops (Liger benchmarks)
- Eliminates intermediate DRAM writes, keeps data in registers
- Effort: HIGH (custom Triton, but Liger templates available)

**R1-IF-05: TP=1 Multi-Instance Serving**
- Run 2-3 model instances on single GPU for batch throughput
- 3.2x throughput for multi-request serving
- Our 6M model easily fits multiple instances in 12GB VRAM
- Effort: LOW (vLLM native support)

**R1-IF-06: Continuous Batching for Hybrid Models**
- SSM O(1) memory per token enables larger batch sizes than Transformers
- 2.65x TTFT, 3x TPOT at 32K context vs Transformer baseline
- Effort: MEDIUM (serving framework modifications)

**R1-IF-07: Speculative Decoding with Tiny SSM Drafter**
- 1M ternary SSM as drafter, 6M as verifier
- SSM fast decode makes drafting cheap
- 2-3x theoretical for decode-heavy workloads
- Effort: HIGH (train drafter model, integrate speculative framework)

**R1-IF-08: KV Cache CPU Offloading for Hybrid Layers**
- Offload attention KV cache to CPU (only 12.5% layers need it)
- Keep SSM state on GPU, attention KV in system RAM
- 4x longer context support, minimal latency impact
- Effort: LOW (vLLM native, llm-d scheduling)

### Researcher 2: Hardware-Level & Compilation Optimizations

**R2-IF-01: Early-Exit for SSMs**
- Add classifier heads at layers 6, 12, 18 (of 24); exit when confident
- 2.5-3x speedup with <1% accuracy loss
- SSMs perfect for adaptive depth (no KV cache complexity)
- Dynamic rebatching handles tokens exiting at different layers
- Effort: MEDIUM (train MLP heads, confidence calibration)

**R2-IF-02: Automated Kernel Fusion Compilers (FusionStitching, Neptune)**
- Merge sequential kernels (matmul → quantize → scan → unquantize) into one
- 2-20x for memory-bound chains, 60% for compute+memory fusion
- Vertical fusion (sequential ops) + horizontal fusion (batch independent ops)
- Effort: HIGH (Triton templates or torch.compile integration)

**R2-IF-03: torch.compile + TorchInductor Graph Fusion**
- Automatic graph fusion via @torch.compile with mode='max-autotune'
- 1.5-2x for pointwise-heavy models, zero-effort optimization
- Effort: TRIVIAL (single decorator, validate correctness)

**R2-IF-04: GDDR7 Memory Bandwidth Optimization**
- RTX 5070: 672 GB/s with PAM3 signaling (50% more data/cycle)
- Ensure 32-byte aligned access, contiguous ternary weight layout
- 10-20% from memory pattern tuning (currently underutilized)
- Effort: LOW (access pattern alignment)

**R2-IF-05: Marlin Kernel Ideas for Packed Weights**
- INT4 Marlin kernels pack weights for GEMM, inspiration for ternary
- 1.5-2x bandwidth savings from packed weight streaming
- Key: train-infer consistency already achieved by our STE approach
- Effort: HIGH (adapt Marlin packing for 2-bit ternary)

**R2-IF-06: ONNX Runtime for Edge Deployment**
- Export to ONNX for DirectML, WebGPU, NPU inference
- 3-4x speedup on edge devices with quantization support
- Enables consumer device reach (laptops, phones)
- Effort: MEDIUM (ONNX export + quantization transforms)

**R2-IF-07: CPU SIMD Ternary Inference (AVX-512)**
- Pack 16 ternary values in 32-bit register, LUT-based multiply
- 5-10x over naive CPU inference, 50+ tok/s on Intel Core Ultra 5
- Use BitNet CPU kernels as template
- Effort: HIGH (custom AVX-512 kernels)

**R2-IF-08: Adaptive Layer Skipping (Per-Token)**
- Skip layers for "easy" tokens based on cosine similarity saturation
- 30-50% layer reduction for typical sequences
- First subword needs full depth, suffixes skip 50%
- Effort: MEDIUM (profiling, dynamic batch handling)

### Priority Ranking (Inference Research)

| Rank | Finding | Speedup | Effort | Priority |
|------|---------|---------|--------|----------|
| 1 | R2-IF-03 torch.compile | 1.5-2x | TRIVIAL | Immediate |
| 2 | R1-IF-04 Mega-kernels | 3-8x/op | HIGH | Phase 23 Step 4 |
| 3 | R2-IF-01 Early-exit | 2.5-3x | MEDIUM | Phase 24+ |
| 4 | R1-IF-03 INT8 activations | 1.5x | MEDIUM | Quick win |
| 5 | R2-IF-08 Layer skipping | 1.3-1.5x | MEDIUM | Stacks with early-exit |
| 6 | R1-IF-06 Continuous batching | 3x | MEDIUM | Serving mode |
| 7 | R2-IF-07 CPU SIMD ternary | 5-10x CPU | HIGH | Edge deployment |
| 8 | R1-IF-01 Marconi prefix cache | 34.4x hits | MEDIUM | Multi-turn serving |

### Key Patterns

1. **Memory bandwidth is THE bottleneck** — kernel fusion, GDDR7 tuning, activation quant all target bandwidth
2. **SSMs have serving advantages** — O(1) memory enables larger batches, no KV cache for 87.5% of layers
3. **Compilation is the meta-strategy** — torch.compile provides 1.5-2x for free, then custom Triton on top
4. **Early exit is high-value** — 2.5-3x with <1% accuracy loss, perfect fit for SSM uniform layer structure
5. **No production ternary inference exists** — we're pioneering; BitNet (2024) is closest reference

## Step 2: Mathematical Analysis (2 mathematicians, 2026-02-26)

### Mathematician 1: Roofline Analysis & Hardware Bounds

**M1-IF-01: Theoretical Bandwidth-Limited Throughput**
- Weight bytes per token: 9.42MB (6M), 49.83MB (50M), 393.4MB (1B)
- Theoretical max B=1: 71,338 tok/s (6M), 13,484 tok/s (50M), 1,708 tok/s (1B)
- System is DEEPLY memory-bound at ALL scales (arithmetic intensity 4 FLOP/byte vs ridge 390)
- Every optimization must maximize effective VRAM bandwidth utilization

**M1-IF-02: Ternary Advantage Scales with Model Size**
- Ternary vs FP16 throughput: 1.75x (6M), 3.27x (50M), 5.60x (1B)
- At 6M: FP16 embeddings dilute ternary compression benefit
- At 1B: embeddings <13% of weight memory, 8x ternary compression yields ~6x advantage
- CRITICAL: ternary value proposition INCREASES with scale

**M1-IF-03: SSM State Is Irrelevant to Performance**
- SSM recurrence: <0.1% of per-token time at ALL scales
- State fits entirely in L2 cache (84KB@6M, 336KB@50M, 5.25MB@1B)
- Optimizing SSM state yields negligible returns
- State update ~0.05μs at 1B vs ~585μs for weight loading

**M1-IF-04: Kernel Launch Overhead Dominates at 6M**
- ~264 kernel launches per token, ~200μs overhead in CUDA Graph
- At 6M: overhead is 2.8x the compute time
- Mega-kernel (26 kernels): overhead drops to 27.5μs → 9,852 tok/s@50M (73% of theoretical)
- Persistent kernel (1 launch): ~5μs → 85-95% of theoretical bandwidth

**M1-IF-05: Batch Size Is "Free" Thanks to SSM**
- Max batch: 6,706 (6M), 820 (50M), 184 (1B) at seq=2048
- 7.7x larger batches than equivalent all-Transformer (SSM O(1) state vs KV cache)
- 50ms latency constraint NEVER binding — even max batch < 1ms latency
- System NEVER becomes compute-bound at achievable batch sizes

**M1-IF-06: 6M Benchmark Is Poor Proxy for Scaled Performance**
- At d=128, GPU cannot achieve meaningful utilization (matrices too small)
- 876.8 tok/s reflects GPU underutilization, not architecture limitation
- Scaling to 50M (d=512) gives disproportionately better tok/s per parameter

**M1-IF-07: Hard Physical Limit at 1B**
- 672 GB/s / 393 MB = 1,708 tok/s — hard bandwidth ceiling at B=1
- To exceed: higher bandwidth HW, weight compression beyond 2-bit, sparse execution, or speculative decoding
- Block-sparse zero skipping (if clustered zeros): ceiling → 1,900-2,200 tok/s

### Mathematician 2: Novel Inference Frameworks

**M2-IF-01: LM Head Is 92.8% of All Weight Reads at 6M (CRITICAL)**
- LM head (tied embeddings): 15.6 MB per token
- All ternary layers combined: only 1.2 MB per token
- IMPLICATION: optimizing ternary GEMV is secondary; LM head is THE bottleneck
- This ratio improves at scale but LM head remains dominant until 200M+

**M2-IF-02: Persistent Mega-Kernel — Model Fits in L2**
- 6M model total weights: 16.8 MB — fits in RTX 5070 L2 cache (~48-96 MB)
- After first token (warm L2): weight reads at ~3 TB/s L2 bandwidth
- Theoretical: 16.8 MB / 3 TB/s = 5.6μs per token = 179,000 tok/s
- Practical with overhead: 2,600-8,800 tok/s (3-10x current)

**M2-IF-03: LSH Vocabulary Projection**
- Cross-polytope LSH with L=24 tables: 99.5%+ recall for top-1 token
- Reduces LM head from 32K dot products to ~750 candidates
- 50.9x compute reduction, ~62x bandwidth reduction
- Combined with persistent kernel: massive impact on total throughput

**M2-IF-04: Product-Quantized Ternary GEMV**
- Decompose weight rows into K=8 sub-vectors, cluster into C=256 codebook entries
- Precompute sub-vector dot products once, lookup per row
- 3.13x computation reduction across all ternary layers
- ALSO 2.67x compression (24KB vs 64KB per matrix)

**M2-IF-05: Batched Speculative Self-Draft**
- Use first 2 of 8 layers as drafter (25% cost per draft token)
- At acceptance rate p=0.7: 2.77 tokens per verification round
- Verification of n tokens is nearly FREE (weights dominate, shared across batch)
- With persistent kernel: 55,200 tok/s projected at p=0.7

**M2-IF-06: GF(3) Rank Decomposition (NOVEL)**
- Ternary matrix over GF(3) = Z/3Z where 2 acts as -1
- If GF(3) rank r << min(m,n): factor W = U*V, GEMV cost O(r(m+n)) vs O(mn)
- At r=32 for in_proj: 12.8x speedup
- ENTIRELY UNEXPLORED in literature — could be genuine contribution
- Requires empirical measurement on trained checkpoint

**M2-IF-07: Fixed-Point SSM Recurrence (Q8.8)**
- Replace FP32 state with INT16 fixed-point: steady-state error ~0.04 (4% of LSB)
- Negligible impact at 6M (SSM is <0.1% of time)
- Value grows at 1B+: 50% state memory reduction
- Periodic re-normalization every ~100 tokens for slow-decaying modes

**M2-IF-08: Optimal Combination Stack**
- Current: 877 tok/s
- + Persistent kernel: ×5 → 4,400 tok/s
- + LSH LM head: ×2 → 8,800 tok/s
- + Speculative (p=0.7): ×2 → 17,600 tok/s
- + PQ GEMV: ×1.5 → 26,400 tok/s
- EXCEEDS 10K target by 2.6×

### Priority Ranking (Mathematical Analysis)

| Rank | Finding | Impact | Actionability |
|------|---------|--------|---------------|
| 1 | M2-IF-01 LM head bottleneck | Redirects ALL effort | Immediate insight |
| 2 | M2-IF-02 Persistent mega-kernel | 3-10x | HIGH (CUDA engineering) |
| 3 | M2-IF-03 LSH vocabulary projection | 2-5x for LM head | MEDIUM (needs embedding analysis) |
| 4 | M1-IF-04 Mega-kernel overhead | 2x at 6M | HIGH (Triton) |
| 5 | M2-IF-05 Speculative self-draft | 1.4-4.2x | MEDIUM (needs trained model) |
| 6 | M2-IF-04 PQ ternary GEMV | 3.13x compute | HIGH (codebook construction) |
| 7 | M2-IF-06 GF(3) decomposition | 1-12.8x (unknown) | SPECULATIVE (novel research) |
| 8 | M1-IF-02 Ternary scales better | Strategic insight | Validates 1B roadmap |

### Key Mathematical Insights

1. **LM head dominance at 6M:** 92.8% of weight reads go to vocabulary projection. This completely changes the optimization priority — ternary GEMV optimization is secondary.

2. **L2 cache opportunity at 6M:** The entire model fits in L2. A persistent kernel could achieve 179K theoretical tok/s by avoiding DRAM entirely after warmup.

3. **Ternary advantage accelerates with scale:** From 1.75x at 6M to 5.6x at 1B — the architecture becomes MORE valuable as models grow.

4. **Never compute-bound:** The compute-bound crossover is beyond VRAM limits at all scales. 100% of effort should go to bandwidth optimization.

5. **Combination stack exceeds target:** Persistent + LSH + Speculative + PQ = 26,400 tok/s at 6M, 2.6× above 10K target.

6. **GF(3) is unexplored territory:** Analyzing trained ternary weights over the finite field GF(3) could yield novel insights and potentially huge speedups via low-rank factorization.

## Step 3: Paradigm Breaking (2 paradigm-breakers, 2026-02-26)

### Paradigm-Breaker 1: Computation Restructuring

**P1-IF-01: SVD-Ternary Factored LM Head**
- Decompose E (32K×256 FP16 = 16MB) via truncated SVD: E = U*S*V^T
- Quantize U to ternary. V^T is tiny FP16 (r×256). S is diagonal.
- Step 1: z = V^T @ x (16KB, fits L1). Step 2: s = S*z. Step 3: logits = U_ternary @ s
- At r=32: 272 KB total vs 16.4 MB original = **60x bandwidth reduction**
- ENTIRE 6M model (including factored LM head) fits in L2 cache → near-zero GDDR7 reads
- Complexity: LOW (post-training SVD, offline) | Risk: LOW (well-understood math)
- Validate: compute SVD of embedding, measure variance captured at r=32,64

**P1-IF-02: Binary Code Token Prediction**
- Replace 32K-way softmax with B=64 independent binary classifications
- Each bit predicted by a single ternary vector (256→1): total LM head = 4 KB (4000x reduction)
- Learned code assignment places similar tokens near in Hamming distance
- Error correction: 49 redundant bits allow BCH 8-bit correction
- Quality risk HIGH (independent bits assumption may fail)
- Complexity: MEDIUM | Risk: HIGH

**P1-IF-03: Delta-Logit Streaming (Video Codec for Logits)**
- Maintain running logits vector; update incrementally via l_{t+1} = l_t + E^T * delta_x
- If delta_x is rank-k (k<<256): only read k columns of E
- At k=32: 2 MB instead of 16.4 MB per token = 8x LM head reduction
- Exact fallback when delta_x is full-rank — ZERO quality risk
- Periodic full recomputation every ~100 tokens prevents drift
- Complexity: LOW | Risk: LOW

**P1-IF-04: SSM State Extrapolation for Multi-Token Amortization**
- Predict next token from SSM state WITHOUT reading layer weights
- h_{t+1} = A*h_t + B*embed(v) — uses only small A,B matrices in registers
- 1.5-1.7x speedup; "drafter" requires ZERO additional model weights
- Unique to SSM — impossible with Transformers
- Verify by checking cosine similarity of extrapolated vs exact hidden state
- Complexity: MEDIUM | Risk: MEDIUM (approximation compounds across layers)

**P1-IF-05: Persistent Shared-Memory Residency**
- Load entire 6M model (~1.5 MB weights) into persistent SM shared memory
- After initial load: ZERO weight bandwidth — transforms bandwidth→compute bound
- One threadblock per SM, persistent (never exits), inter-SM sync via L2
- RTX 5070: 84 SMs × 48 KB = 4 MB shared → model fits
- Eliminates bandwidth bottleneck entirely at 6M scale
- Complexity: HIGH (persistent kernel programming) | Risk: MEDIUM

### Paradigm-Breaker 2: Ternary-Native Algorithms

**P2-IF-01: Ternary LM Head — Embarrassingly Obvious Fix (HIGHEST PRIORITY)**
- The embedding IS the LM head (tied weights) and IS ALREADY trained as ternary via STE
- Making inference LM head ternary is simply matching inference format to training format
- 32K×256×2bits packed = 2.1 MB (+ 64KB scales) vs 16 MB FP16 = **7.6x reduction**
- At 2.1 MB: fits permanently in 48 MB L2 cache → zero GDDR7 reads after warmup
- Reuses existing Triton GEMV kernel — zero new kernel development needed
- Complexity: TRIVIAL (config change) | Risk: VERY LOW
- **This single change eliminates 92.8% bottleneck**

**P2-IF-02: Neuromorphic Race Decoding (Anytime Argmax)**
- Sort hidden dims by |h[d]| descending; process in that order
- Each dim contributes exactly +h[d], -h[d], or 0 (ternary = pure add/sub)
- Early termination when leader's margin exceeds remaining budget R(k)
- IMPOSSIBLE with FP16 — requires ternary's structured accumulation
- Average 40-60% of dimensions needed → 2.5-3x on top of ternary LM head
- Requires column-major weight storage + batched termination check
- Complexity: MEDIUM | Risk: MEDIUM

**P2-IF-03: Residual Delta Projection**
- SSM states evolve smoothly (A near identity) → delta_t sparse
- Only read columns of W where |delta_t[d]| > threshold
- I-frame/P-frame pattern: full recompute every N=8 tokens, deltas between
- At K=32 sparse dims, N=8: 487 KB amortized vs 2.1 MB = 4.3x additional
- Stacks multiplicatively with ternary LM head
- Complexity: LOW-MEDIUM | Risk: MEDIUM (needs delta sparsity validation)

**P2-IF-04: Hadamard-Domain Progressive Refinement**
- WHT is addition/subtraction only — ternary-native orthogonal transform
- Pre-transform: W' = W@H^T. Inference: h' = H@h, logits = W'@h'
- Energy concentrates in low-frequency coefficients → two-pass coarse+fine
- Coarse (32 coefficients): 256 KB. Fine (224 cols for top-256 candidates): 14 KB
- Total: 270 KB vs 2.1 MB = 7.8x additional | Complexity: MEDIUM | Risk: MEDIUM

**P2-IF-05: Predictive Vocabulary Masking**
- Lightweight classifier (33K params) predicts 8/128 active vocab clusters
- 6.25% of vocab scored → 10.8x reduction
- CRITICAL RISK: missing correct cluster = wrong token (must be >99.9% recall)
- Safer as soft mask (multiplicative scaling) than hard mask
- Complexity: MEDIUM | Risk: HIGH (catastrophic failure mode)

### Priority Ranking (Paradigm Breaking)

| Rank | Finding | Impact | Viability | Priority |
|------|---------|--------|-----------|----------|
| 1 | P2-IF-01 Ternary LM Head | 7.6x on 92.8% bottleneck | VERY HIGH | Implement TODAY |
| 2 | P1-IF-01 SVD-Ternary Factor | 60x LM head | HIGH | Validate with SVD analysis |
| 3 | P1-IF-03 Delta-Logit Stream | 4-8x average | HIGH | Zero quality risk |
| 4 | P2-IF-02 Race Decoding | 2.5-3x additional | MEDIUM-HIGH | Ternary-unique algorithm |
| 5 | P1-IF-05 Persistent SM | Zero bandwidth | MEDIUM | Transforms problem regime |
| 6 | P2-IF-03 Residual Delta | 4.3x additional | MEDIUM | Needs empirical validation |
| 7 | P2-IF-04 Hadamard Progressive | 7.8x additional | MEDIUM | Better at 50M+ scale |
| 8 | P1-IF-04 SSM Extrapolation | 1.5-1.7x | MEDIUM | SSM-unique advantage |
| 9 | P1-IF-02 Binary Codes | 4000x | LOW-MEDIUM | High quality risk |
| 10 | P2-IF-05 Vocab Masking | 10.8x | LOW-MEDIUM | Catastrophic failure risk |

### Key Paradigm Insights

1. **THE KILLER FINDING: Ternary LM head is trivially obvious.** The embedding IS already ternary during training (tied weights + STE). Making inference ternary is just matching formats. This alone eliminates the 92.8% bottleneck. Every other LM head optimization stacks on top.

2. **Ternary enables impossible algorithms.** Race decoding, column-wise accumulation, branch-and-bound argmax — these REQUIRE ternary/binary weights. FP16 models cannot use them. Ternary is not "degraded FP16" — it is a DIFFERENT computational regime with unique algorithmic possibilities.

3. **The combination stack is multiplicative:**
   - Ternary LM head: 16MB → 2.1MB (7.6x)
   - + L2 cache residency: ~0 GDDR7 reads
   - + Race decoding (40% dims): 2.1MB → 0.84MB effective (2.5x)
   - + Residual delta (K=32, N=8): 0.84MB → 0.20MB amortized (4.2x)
   - **Total: ~80x LM head bandwidth reduction**

4. **After fixing LM head, bottleneck shifts to ternary body layers.** This is architecturally correct — the model should spend bandwidth on COMPUTATION (thinking), not on vocabulary projection.

5. **Delta-logit streaming is zero-risk.** Exact fallback when deltas aren't sparse. Can be implemented as pure inference optimization with no training changes.

6. **SSM state extrapolation is unique to SSMs.** Transformers can't do this. The SSM recurrence naturally provides a free single-step predictor from the state. This is a native architectural advantage for speculative generation.
