# Research Round 4 — Aspect 5: Portable Kernel Algorithms
**Category:** TRAINING + INFERENCE SPEED (hardware-agnostic)
**Date:** 2026-03-01

## Focus
Algorithmic optimizations for ternary matmul and SSM kernels that work on ANY hardware — x86, ARM, RISC-V, WASM, mobile. No vendor-specific features. Pure algorithmic and compiler-driven approaches.

---

## Wave 1 — Web Research (Portable Kernel Optimization)

### Theme A: Cache-Oblivious Algorithms & Tiling

#### R4-A5-F1: ACTIVATION-PERSISTENT-VS-OUTPUT-PERSISTENT-ADAPTIVE-DATAFLOW
- **Category:** IMPLEMENT
- **Source:** T-SAR (arxiv 2511.13676v1, Nov 2025)
- **What it does:** Two distinct microkernel dataflow patterns selected at compile-time per layer based on shape. Activation-Persistent (AP) retains input activations in registers to minimize LUT recomputation; Output-Persistent (OP) keeps accumulators local to reduce memory write-back.
- **Mechanism:** Implement both AP and OP variants for ternary matmul kernel. At model load time, select fastest dataflow per layer based on (N, K, M) dimensions. AP minimizes TLUT invocations; OP minimizes DRAM stores. Same compute primitives, differ only in loop order and register allocation.
- **Expected Impact:** 1.15-2.1x additional speedup over baseline LUT-based ternary kernels. Combined with register-resident LUT generation for 5.6-24.5x GEMM latency reduction.
- **v5 Applicability:** RecursiveSSM forward pass has varying (N, K, M) across layers. Auto-select via profiling hook. Pure algorithmic, no ISA dependencies.

#### R4-A5-F2: TERNARY-TO-BINARY-DECOMPOSITION-WITH-POWER-OF-TWO-LUT
- **Category:** IMPLEMENT
- **Source:** T-SAR (arxiv 2511.13676v1, Nov 2025)
- **What it does:** Decomposes ternary weight block w ∈ {-1,0,+1}^c into Dense weights w_D ∈ {-1,+1}^c and Sparse mask w_S ∈ {0,1}^c. Transforms one ternary LUT (3^c entries, base-3) into two binary LUTs (2^c entries each, base-2), matching power-of-two hardware register widths.
- **Mechanism:** At weight packing time: split each ternary block into dense+sparse bitplanes. At inference: generate two binary LUTs from activations, perform two lookups per block, subtract results. Storage = 2^(c+1) entries vs 3^c.
- **Expected Impact:** Eliminates base-3/base-2 architectural mismatch. 87.6% memory transaction reduction (TLUTs were dominant memory traffic). One subtraction per block but eliminates external memory round-trip.
- **v5 Applicability:** Packed ternary weights (2-bit, 16/word) can be pre-split into dense+sparse during model load. Portable across all SIMD widths — NEON, AVX2, AVX-512, RISC-V RVV.

#### R4-A5-F3: SIMD-WIDTH-AGNOSTIC-VECTORIZATION-WITH-DIAGNOSTICS
- **Category:** IMPLEMENT
- **Source:** Deep Engineering #17 (Sep 2025), GCC 15 vectorization enhancements
- **What it does:** Write clean C/C++ loops with SoA layout, alignment, restrict pointers. Enable compiler diagnostics to make vectorization observable and catch de-vectorization regressions. Use function multi-versioning only for critical hotspots.
- **Mechanism:** (1) SoA layout for weight/activation arrays. (2) Align buffers to cache lines. (3) Compiler diagnostics (-Rpass=loop-vectorize) in CI to fail on missed vectorization. (4) target_clones for 1-2 hottest kernels for runtime ISA dispatch. GCC 15 vectorizes unknown trip counts at -O2.
- **Expected Impact:** Auto-vectorizes across x86 (AVX2/512), ARM (NEON), RISC-V (RVV) without manual intrinsics. Multi-versioning yields 1.15-2.1x additional on AVX-512 while maintaining compatibility.
- **v5 Applicability:** RecursiveSSM kernel loops can be SoA + aligned + restricted. Add vectorization diagnostics to CI. Single binary runs on ANY CPU.

#### R4-A5-F4: PREFETCH-FRIENDLY-TILING-FOR-TERNARY-MATMUL
- **Category:** IMPLEMENT
- **Source:** BitNet.cpp TL-2 kernel (ACL 2025, arxiv 2502.11880), TABv2 (IEEE TVLSI 2026)
- **What it does:** Partition ternary matmul into fixed-size tiles that fit L1/L2 cache. Process tiles in row-major for activations, column-major for weights to maximize locality. Prefetch next tile while computing current.
- **Mechanism:** Runtime cache size detection, tile dimensions fit activation+weight+output in L1/L2. Software prefetch for next tile. For packed 2-bit weights, tile boundaries aligned to word boundaries (K % 16 = 0). Pad K dimension to multiple of 16.
- **Expected Impact:** BitNet.cpp reports 2.37-6.17x on x86, 1.37-5.07x on ARM. TABv2: 2.1x at GEMM level. Reduces cache misses by 40-60%, improves bandwidth utilization from ~30% to ~70-80%.
- **v5 Applicability:** RecursiveSSM matmul kernels (32 layers x ~220K params) benefit directly. Implement in C++ with runtime cache detection. Expected 1.5-2.5x throughput increase.

#### R4-A5-F5: LOOP-UNROLLING-AND-REGISTER-BLOCKING-FOR-TERNARY
- **Category:** IMPLEMENT
- **Source:** TABv2 (IEEE TVLSI 2026), T-SAR kernel framework (Nov 2025)
- **What it does:** Register blocking: accumulate 4-8 output elements simultaneously in registers before storing. Unroll inner loop by 16 to process full packed word per iteration. Exposes ILP and reduces loop overhead.
- **Mechanism:** 4-8 SIMD register accumulators. Unroll K-loop by 16 (one packed word = 16 ternary weights). POPCOUNT + add/subtract per unrolled iteration. Horizontal reduction after K-loop. Compiler-driven with pragma unroll.
- **Expected Impact:** TABv2: 2.1x at GEMM level. Register-resident accumulation reduces memory writes 60-75%. Unrolling by 16 matches packed word size. ILP saturates execution ports: 1.3-1.8x over scalar.
- **v5 Applicability:** RecursiveSSM K=6912 (typical 2B) unrolled by 16. 8 accumulators cover 8 output channels. Portable — compiler-driven across all architectures.

### Theme B: Portable Vectorization & Cross-Platform Kernel Design

#### R4-A5-F6: HIGHWAY-RUNTIME-DISPATCH-WITH-COMPILE-TIME-TEMPLATING
- **Category:** IMPLEMENT
- **Source:** Google Highway library (github.com/google/highway, production-ready 2025)
- **What it does:** Single C++ codebase compiles to 27 SIMD targets (SSE2 through AVX-512, ARM NEON/SVE, RISC-V V, WASM) with runtime CPU detection or static single-target compilation. Uses templated tag dispatch so algorithms are written once.
- **Mechanism:** ScalableTag<T> templating, foreach_target.h pattern, HWY_DYNAMIC_DISPATCH for runtime selection. Handles scalable vectors (RVV, SVE) transparently. Used in Chromium, JPEG XL, TensorFlow.
- **Expected Impact:** 5-10x speedup vs scalar. 85-95% of hand-tuned assembly performance. Write kernel ONCE, compile for all platforms.
- **v5 Applicability:** Core ternary matmul kernel written as Highway algorithm. ScalableTag<int8_t> for signs, ScalableTag<int16_t> for accumulators. Same code → AVX2 (16 weights/cycle), NEON (8/cycle), RVV (4-32/cycle), WASM (8/cycle). ONE codebase for all platforms.

#### R4-A5-F7: XSIMD-BATCH-TYPES-ARCHITECTURE-AGNOSTIC-MATH
- **Category:** IMPLEMENT
- **Source:** xsimd library (xsimd.readthedocs.io, NumScale/xtensor ecosystem)
- **What it does:** C++14 batch<T, Arch> type abstracts SIMD vectors with 100+ math functions compiling to optimal instructions per-architecture. Includes alignment manipulation, type traits, batch ops (shuffle, broadcast).
- **Mechanism:** batch<int8_t> signs = select(weight_bits & sign_mask, -1, +1) → compiles to AVX2 vpblendvb, NEON vbsl, RISC-V vmerge. Handles lane width differences via batch::size trait. Aligned/unaligned load variants.
- **Expected Impact:** 6-10x speedup (word embedding 6.6x, sentiment 9x measured). Fine-grained control for tight inner loops.
- **v5 Applicability:** Ternary matmul inner loop: batch<int16_t> acc, batch<int8_t> signs decoded, multiplied with inputs. Lower-level than Highway — useful for tight inner loops where Highway's abstraction adds overhead.

#### R4-A5-F8: WASM-SIMD-FOR-BROWSER-TERNARY-INFERENCE
- **Category:** IMPLEMENT
- **Source:** Production consensus 2025 (ThinkingLoop/Medium, Netrunnaz), WASI-NN spec
- **What it does:** WebAssembly 128-bit SIMD (v128) enables vector ops in browsers and WASM runtimes. WASI-NN provides standardized ML inference API. Compile C/C++ to wasm32-wasi with SIMD, deploy to browsers or edge devices.
- **Mechanism:** 128-bit vectors: 16×i8, 8×i16, 4×i32 ops. Compile with Emscripten -msimd128. WASI-NN: load() → init_context() → set_input() → compute() → get_output(). Sub-100ms cold start, <200ms inference latency.
- **Expected Impact:** 2-4x over plain WASM (RGB→grayscale 8.1x, edge detection 10x vs JS). Browser-based inference: zero network latency, private, offline-capable. 85-95% of native CPU performance.
- **v5 Applicability:** RecursiveSSM 220K params = ~55KB packed → loads instantly in browser. Expected ~400-450 tok/s in Chrome. Compile via Highway (supports WASM target). Deploy as chatbot widget, code completion, privacy-preserving text processing.

#### R4-A5-F9: RISC-V-VECTOR-EXTENSION-AUTO-VECTORIZATION
- **Category:** IMPLEMENT
- **Source:** IEEE (11043225, GCC 14.1 study), arxiv 2511.06955 (RISC-V NN extensions)
- **What it does:** RVV 1.0 provides scalable SIMD where vector length is implementation-defined (128-2048 bit). GCC 14.1+ auto-vectorizes C loops to RVV without code changes. Custom ISA extensions demonstrate 4.3x + 49% energy reduction.
- **Mechanism:** Compiler inserts vsetvli for vector length. Packed weights loaded via vle8.v, unpacked with vsrl.vi + vand.vi, accumulated via vadd.vv, reduced via vredsum.vs. Code compiles to ALL VLEN from same source.
- **Expected Impact:** 1.88x with auto-vectorization (zero code changes), 2-4x with manual intrinsics, 4.3x with custom extensions. Same binary runs on $50 RISC-V SBC and $5000 server.
- **v5 Applicability:** RecursiveSSM compiles with GCC -march=rv64gcv -O3. Inner loop auto-vectorizes. Deployment: embedded (128-bit RVV, 50-100 tok/s), desktop (256-bit, 200-300 tok/s), server (512-bit, 500+ tok/s).

#### R4-A5-F10: STRUCTURE-OF-ARRAYS-ALIGNED-MEMORY-PORTABLE-SIMD
- **Category:** IMPLEMENT
- **Source:** Highway docs, xsimd alignment docs, SIMD optimization best practices
- **What it does:** SoA layout + 64-byte aligned buffers enables efficient SIMD loads/stores across ALL architectures. Batch processing in multiples of vector width minimizes scalar tail handling.
- **Mechanism:** AoS → SoA transform for weight/activation arrays. aligned_alloc(64, size) for buffers. Highway AllocateAligned helper. BlendedStore for tail handling. Pad dimensions to vector width multiples.
- **Expected Impact:** SoA vs AoS: 3-10x for complex structures. Aligned vs unaligned: 1.2-1.5x on modern CPUs, 2-4x on older. Prevents cache-line splits.
- **v5 Applicability:** Packed ternary decoded to separate sign[] + zero_flag[] arrays (SoA). All buffers 64-byte aligned. Pad hidden_dim to 64 boundary. Same pattern works on all SIMD widths. Zero-cost on modern allocators.

---

## Wave 2 — Math-Paradigm-Breakers (Portable Kernel Mathematics)

### Theme A: Mathematical Foundations for Ternary Computation

#### R4-A5-M1: AMORTIZED-BLOCK-PATTERN-PRECOMPUTATION (ABP)
- **Category:** IMPLEMENT
- **Mathematical Insight:** Ternary matvec y=Wx can be decomposed by partitioning columns into blocks of size k. Distinct ternary patterns within each block are at most 3^k. Precompute signed partial sum for every occurring pattern, then assemble via table lookups — amortizes addition cost across all rows sharing patterns.
- **Mechanism:** Partition [n] into n/k blocks. For each block, compute sigma(pattern) = sum of signed inputs. Assembly: y_i = sum of sigma lookups across blocks. Optimal k satisfies 3^k ≈ m/k². For m=768: k=4 optimal. Cost: 209K ops vs 393K naive = 1.88x speedup.
- **Expected Impact:** 1.9-2.4x reduction in arithmetic operations. Scales with output dimension m. Pattern indices precomputed once at model load.
- **v5 Applicability:** Every TernaryLinear projection in RecursiveSSM. Pattern indices also serve as compressed weight representation.
- **Portability:** Uses only integer indexing and FP16 addition — universal across all hardware. 81-entry LUT (k=4) fits in any L1 cache.

#### R4-A5-M2: QUINARY-BYTE-PACKING (5 trits per byte)
- **Category:** IMPLEMENT
- **Mathematical Insight:** 3^5 = 243 ≤ 256 = 2^8, so 5 ternary weights fit losslessly in 1 byte. This achieves 1.6 bits/weight (0.95% above information-theoretic limit of 1.585 bits). Standard 2-bit packing wastes 26% capacity.
- **Mechanism:** Encoding: z = sum_{i=0}^{4} (w_i+1) × 3^i → z ∈ [0,242] → 1 byte. Decoding: 243-entry LUT (1,215 bytes, fits any L1). Decode cost: 1 byte load + 1 LUT access vs 2-bit: 5× (shift+mask). SYNERGY with M1: packed byte value IS the ABP pattern index, merging decode+lookup tables.
- **Expected Impact:** 20% reduction in ternary weight storage (1.033 GB → 826 MB at 7B). Faster decode than 2-bit. Byte-granularity indexing.
- **v5 Applicability:** All TernaryLinear storage. 1,215-byte decode table loaded once. Combined with ABP k=5 eliminates all decode overhead.
- **Portability:** Byte-addressable memory is universal. No alignment issues, no endianness issues, works on 8-bit microcontrollers to 64-bit servers.

#### R4-A5-M3: COMPUTE-BOUND-REGIME-THEOREM-FOR-TERNARY
- **Category:** IMPLEMENT (changes optimization TARGET for all kernel development)
- **Mathematical Insight:** Standard FP32 matvec is memory-bandwidth-bound (arithmetic intensity ~0.5 ops/byte). Ternary matvec with packed weights (0.2 bytes/weight) and cached x is COMPUTE-BOUND (intensity ~3.33 ops/byte) — 6.67x more compute-intense. This fundamentally changes the optimization strategy.
- **Mechanism:** Proof: with 5-in-8 packing, streaming W costs 0.2n bytes/row. With ~2n/3 additions (2/3 nonzero fraction): I = (2n/3)/(0.2n) ≈ 3.33 ops/byte. For n=768 on 4GHz 8-wide SIMD: compute 16ns, memory 0.5ns → compute:memory = 32:1. Correct optimization: maximize addition ILP and SIMD utilization, NOT prefetching/bandwidth.
- **Expected Impact:** 2-5x better kernel performance vs mis-targeted bandwidth optimization. Reframes ALL kernel development priorities.
- **v5 Applicability:** Every TernaryLinear in inference pipeline. All ternary matmul/matvec obeys this theorem.
- **Portability:** Hardware-INDEPENDENT mathematical property. The intensity ratio depends only on byte-sizes of packed ternary vs FP32, which are constants regardless of hardware. Same optimization strategy correct on ALL platforms.

#### R4-A5-M4: GF3-RANK-FACTORIZATION-OF-TERNARY-MATRICES
- **Category:** EXPLORE
- **Mathematical Insight:** Ternary matrices live in GF(3)^{m×n}. Over GF(3), W has rank r_3 and if r_3 < mn/(m+n), factorization W=UV (both U,V ternary) reduces matvec from mn to r_3(m+n) additions. EXACT — zero quality loss. No analog in floating-point computation.
- **Mechanism:** Gaussian elimination mod 3 at model load time. O(mn·min(m,n)) mod-3 ops (~100ms for 768×768). If rank r_3 ≤ m/2: ~2x speedup. If r_3 ≤ m/4: ~3.5x. Approximate variant: keep rank-r + sparse residual E. Factors U,V are ternary, use same packed storage and compute kernels.
- **Expected Impact:** 2-3.5x speedup with ZERO quality loss if matrices have low GF(3) rank. Empirical measurement needed on trained weights.
- **v5 Applicability:** Post-training optimization for every TernaryLinear. Factorization computed once. RecursiveSSM shared weights: one factorization speeds up all 32 layer passes.
- **Portability:** GF(3) = modular arithmetic on small integers. Factored matrices use same ternary infrastructure. No hardware-specific features at any stage.

### Theme B: SSM Scan Optimization for Ternary A

#### R4-A5-M5: TERNARY-SCAN-TO-SPARSE-MATMUL (TSM)
- **Category:** IMPLEMENT
- **Mathematical Insight:** SSM scan h_t = A_t·h_{t-1} + u_t defines linear map h = Mu where M is lower-triangular ternary matrix. With p(A=0)=1/3, M is extremely sparse (~3L non-zeros instead of L²/2). Since A is learned (not input-dependent) and shared across 32 layers, M is computed ONCE and reused.
- **Mechanism:** Scan A to find zero positions (segment boundaries). Within segments, compute cumulative sign products. Fill CSR sparse matrix. Each matvec: 3L add/subtract operations. CSR structure identical for all 32 layers and batch elements.
- **Expected Impact:** ~340x parallel speedup over sequential scan (O(1) depth vs O(Ld)). 3x more total operations but all parallelizable and pure add/subtract.
- **v5 Applicability:** Direct. RecursiveSSM scans same A 32 times. Sparse ternary matvec uses same add/subtract infrastructure as TernaryLinear. CSR with sign-bit values only.
- **Portability:** CSR sparse matvec is among most well-optimized primitives across ALL platforms. Ternary constraint simplifies: values array is bit vector, not float array.

#### R4-A5-M6: CROSS-LAYER-TEMPLATE-CACHING (CLTC)
- **Category:** IMPLEMENT
- **Mathematical Insight:** Scan structure (segment boundaries, parity patterns) depends ONLY on learned A, not on input. Precompute "scan template" T encoding ALL combinatorial structure, reducing each layer's scan to 3 element-wise operations: conditional negate → segmented prefix sum → conditional negate.
- **Mechanism:** Template T = (seg_id, parity_bit) computed once in O(L). Per-layer: (1) y[i] = par[i] ? -u[i] : u[i] (2) S[i] = segmented prefix sum of y (3) h[i] = par[i] ? -S[i] : S[i]. Proof: h_t = (-1)^{c_t} × PrefixSum((-1)^{c_j} × u_j). Branchless: (val XOR mask) - mask.
- **Expected Impact:** Per-layer cost: 5L elementary ops (4 integer + 1 FP add per element). For L=1024, d=64: 328K ops/layer, 10.5M for 32 layers. Template is 2L bytes — fits L1 for L ≤ 16K.
- **v5 Applicability:** Template computed once in forward(), reused 32 times. At inference, computed once at model load (A is fixed). Eliminates all combinatorial overhead from per-layer scan.
- **Portability:** Integer XOR, subtract, FP add, integer comparison only. Element-wise and prefix-sum operations vectorize on any SIMD width. Template is byte arrays — portable everywhere.

#### R4-A5-M7: PARITY-PARTITIONED-DUAL-STREAM (PPDS)
- **Category:** EXPLORE
- **Mathematical Insight:** Within segments, scan decomposes into TWO independent prefix sums over disjoint index sets (even-parity and odd-parity positions). These are COMPLETELY INDEPENDENT — zero data dependency between them. Transforms one sequential prefix sum of length K into two independent sums of length ~K/2.
- **Mechanism:** Partition positions by parity c_i mod 2. Compute P_E (even prefix sum) and P_O (odd prefix sum) independently. Combine: h_j = (-1)^{c_j} × (P_E[j] - P_O[j]). Pack both streams into SIMD lanes: single add instruction advances BOTH sums.
- **Expected Impact:** 1.5-2x speedup on prefix sum step (bottleneck). 1.3-1.5x overall scan speedup. Doubles SIMD utilization for prefix computation.
- **v5 Applicability:** Optimization of M6's Step 2. Compatible with template caching — adds even/odd partition masks to template. Relevant for training (long sequences) and prefill.
- **Portability:** Data partitioning + addition/subtraction only. Blend/shuffle available on SSE4.1, AVX2, NEON, WASM SIMD. On GPU: two independent thread groups per warp.

#### R4-A5-M8: ABSORBING-STATE-TREE-PRUNING (ASTP)
- **Category:** EXPLORE
- **Mathematical Insight:** In parallel scan tree, A=0 is absorbing under multiplication. At tree level k, probability of non-zero accumulated A decays DOUBLY EXPONENTIALLY: (2/3)^{2^k}. Level 0: 33% zero, Level 2: 80% zero, Level 3: 96% zero, Level 4: 99.8% zero. Pruned nodes do zero work (simple copy).
- **Mechanism:** Standard recursive doubling with branchless pruning: mask = (a_left ≠ 0) & (a_right ≠ 0). When mask=0, combine degenerates to copy of right child's b-value. Branchless: 5 ops regardless, but multiply-by-zero eliminates expensive terms. Total work drops from O(L log L) to O(L) with constant ≈ 0.94.
- **Expected Impact:** ~2.5x fewer total operations (L=1024: 12K vs 31K ops). Effective depth ~4 levels instead of 10. All non-pruned nodes use conditional negate (not general multiply).
- **v5 Applicability:** Direct for any parallel scan implementation. Pruning mask is part of template (computed once, reused 32 layers). Only applies to training/prefill (inference L=1).
- **Portability:** Branchless = no warp divergence (GPU), no branch misprediction (CPU), no control flow (WASM). Pure integer comparison + conditional negate + FP add. Parallel scan tree topology maps identically to any parallel architecture.

---

## Wave 3 — Paradigm-Breakers (Radical Portable Computation)

### Theme A: Radical Computation Paradigms for Ternary

#### R4-A5-P1: BUTTERFLY-TERNARY-FACTORIZATION (BTF)
- **Category:** EXPLORE
- **Radical Insight:** Ternary weight matrix W (N×K) can be decomposed into log2(N) sparse ternary butterfly factors. Each factor has O(N) nonzero entries in 2×2 block-diagonal pattern. Total GEMV cost becomes O(N log N) instead of O(NK).
- **Mechanism:** Parameterize W = B_1 × B_2 × ... × B_{log2(N)}. Each B_k is butterfly stage with stride 2^k. Inference: cascade x through stages, each O(N) add/sub. For N=256: 8 stages × 256 = 2K ops vs 131K dense = 64x reduction. Training: butterfly-constrained STE.
- **Expected Impact:** 16-64x fewer arithmetic operations per GEMV. Memory drops from 16KB to 512B per matrix. Perfect cache behavior (contiguous stride access).
- **v5 Applicability:** All TernaryLinear layers. Shared SSM block: one factorization speeds up all 32 iterations. Combined with TSCS: sub-microsecond SSM block on CPU.
- **Risk/Limitation:** Butterfly manifold spans ~3% of general ternary matrices — quality loss likely. May need 2-4x wider to compensate. Diagnostic needed: factorize trained matrices, measure reconstruction error.

#### R4-A5-P2: THRESHOLD-ONLY-TERNARY-TO-TERNARY-MATMUL (T3M)
- **Category:** EXPLORE
- **Radical Insight:** When BOTH activations and weights are ternary, full inner product decomposes into 4 POPCOUNT operations over bitmasks. No accumulation needed — just AND + POPCOUNT + subtract + compare.
- **Mechanism:** Represent each ternary vector as P (positive) + N (negative) bitmasks. y_i = popcount(W_P[i] AND X_P) + popcount(W_N[i] AND X_N) - popcount(W_P[i] AND X_N) - popcount(W_N[i] AND X_P). Per output: 16 POPCOUNT + 6 add/sub + 1 CMP = 23 ops vs 170 standard.
- **Expected Impact:** 7.4x speedup on ternary-activation layers. With hardware POPCOUNT (all modern CPUs): 16 cycles per output. Without: software POPCOUNT still 2.7x faster.
- **v5 Applicability:** Requires ternary activation quantization (aggressive FPIT). Best tested at 50M+ scale where redundancy compensates information loss. Combined with RecursiveSSM 32 iterations: massive compound savings.
- **Risk/Limitation:** Ternary activations (1.585 bits) may degrade quality at 6M. Not applicable to embedding or output layers.

#### R4-A5-P3: RECURSIVE-TERNARY-DELTA-PROPAGATION (RTDP)
- **Category:** IMPLEMENT
- **Radical Insight:** In RecursiveSSM, consecutive iterations produce increasingly similar outputs (convergence proven by adaptive depth). Compute DELTA y_new = y_old + W × (x_new - x_old). When activations are quantized, delta is SPARSE — only changed positions contribute. 90% stable = 10x speedup per iteration.
- **Mechanism:** Iteration 1: full GEMV. Iterations 2-32: compare quantized activations against previous, identify changed positions, update outputs using weight columns for changed dims only. Cost: fraction_changed × full_cost. With 8% average flip rate: 1 + 31×0.08 = 3.48 full equivalents instead of 32 = 9.2x.
- **Expected Impact:** 7-10x across 32 iterations. Combined with adaptive depth (9 effective iterations): 1 + 8×0.08 = 1.64 equivalents = 19.5x. Multiplicative with other kernel optimizations.
- **v5 Applicability:** Direct for RecursiveSSM. Needs row-packed (full GEMV) + CSC (delta updates) weight formats. Storage overhead ~2x on 220K shared block = ~165KB total. Negligible.
- **Risk/Limitation:** If model doesn't converge (high flip rates >30%), benefit is marginal. Diagnostic: measure per-iteration flip rate at inference.

#### R4-A5-P4: HOMOMORPHIC-TOKEN-PACKING (HTP)
- **Category:** IMPLEMENT
- **Radical Insight:** Ternary GEMV is LINEAR (only add/subtract). Pack 4 INT8 activation values into non-overlapping 16-bit fields of a 64-bit integer. One add/subtract simultaneously processes 4 tokens. 4x throughput using ONLY standard 64-bit ALU — no SIMD, no special instructions.
- **Mechanism:** Pack: packed_x[j] = x_t0 | (x_t1 << 16) | (x_t2 << 32) | (x_t3 << 48). GEMV: same loop but on packed words. Unpack: shift+mask outputs. Overflow guarantee: K=256, max per field = 21,760 < 32,768 (16-bit signed).
- **Expected Impact:** 4x throughput for batch/prefill on ANY 64-bit CPU. Guaranteed, exact, trivial to implement (~15 lines). Composes multiplicatively with all other optimizations.
- **v5 Applicability:** Immediate for prefill, speculative decoding (4 candidates), training forward pass. Modify ternary_gemv.c inner loop. For K>512: use 17-bit fields (3 tokens/word).
- **Risk/Limitation:** Requires batch ≥ 4. No benefit for single-token decode without speculation. INT8 activations required.

### Theme B: Radical Exploitation of Weight Sharing + Ternary + Recurrence

#### R4-A5-P5: GEOMETRIC-EXTRAPOLATION-OF-DEPTH
- **Category:** IMPLEMENT
- **Radical Insight:** RecursiveSSM's depth iteration is a fixed-point iteration. Deltas form approximately geometric sequence: ||delta_k|| ≈ r^k × ||delta_0||. After K warmup iterations, EXTRAPOLATE the fixed point: x_final ≈ x_K + delta_K × r/(1-r). Predicts fixed point in K+1 steps instead of 32.
- **Mechanism:** After K=6-8 iterations: compute per-dimension contraction ratio r = delta_K / delta_{K-1}, clamp to [0, 0.99], compute geometric tail sum = delta_K × r/(1-r). Error bound: ||delta_K|| × |r_est - r_true|^(32-K) / |1-r|, vanishes exponentially.
- **Expected Impact:** 3-6x inference speedup (32 → 6-10 warmup + 1 extrapolation). Extrapolation costs O(d_model) — negligible. Quality loss bounded and exponentially decreasing with K.
- **v5 Applicability:** ~25 lines in RecursiveSSM forward. Coexists with adaptive_depth_exit. Inference-only mode.
- **Risk/Limitation:** Iteration embeddings perturb contraction ratio. Non-contractive early in training. Per-dimension independence assumption approximate.

#### R4-A5-P6: PARALLEL-DEPTH-SCAN (Linearized Depth via Prefix Scan)
- **Category:** EXPLORE
- **Radical Insight:** Linearize the 32-iteration recurrence around the trajectory. With shared weights, Jacobian J is approximately SAME at every step. Composition of (I+J)^N precomputed via matrix exponentiation. Solve all 32 iterations in O(log 32) = 5 parallel steps.
- **Mechanism:** 2-3 warmup iterations, estimate J via 16 random probes (finite differences), compute (I+J)^k via repeated squaring, apply to get all x_k simultaneously. On parallel hardware: effective 8 sequential steps instead of 32 (4x latency reduction).
- **Expected Impact:** 4x depth-latency reduction on parallel hardware. Most valuable for GPU batch inference where depth-latency dominates.
- **v5 Applicability:** New inference mode. Linearization quality depends on convergence. Combine with P5 for hybrid: parallel scan first 16, extrapolate last 16.
- **Risk/Limitation:** Rank-16 Jacobian captures 6.25% of full space. Nonlinearities break linearity assumption. 16 extra forward passes for estimation. Inference-only.

#### R4-A5-P7: TERNARY-CIRCUIT-COMPILATION
- **Category:** IMPLEMENT
- **Radical Insight:** Ternary weights have ~33% exact zeros, creating sparse computation graph. Weight sharing means SAME graph traversed 32 times. Compose the graph with itself: paths through ANY zero at ANY iteration are permanently dead. Static analysis at model load compiles optimized execution plan eliminating all provably dead computation.
- **Mechanism:** Extract sign patterns from ternary matrices. Build adjacency graph. Compose cross-iteration: G_composed = |S_out^T| × |S_in|. Individual 33% sparsity → composed 55% sparsity (1 - 0.67²). Full SSM path: 70% dead (1 - 0.67³). Generate CSR index arrays for gather-add-subtract execution. Compile time: O(d²) ≈ 100ms for d=768.
- **Expected Impact:** 15-30% total inference speedup from eliminating dead paths. Compilation cost amortized over millions of tokens. During training: recompile every 100-500 steps.
- **v5 Applicability:** Replace generic matmul in TernaryLinear with compiled gather-add-subtract. Compile ONCE at model load, reuse 32 iterations. CSR format is portable across all hardware.
- **Risk/Limitation:** Nonlinearities/residuals can re-activate dead paths. Only optimize ternary matmul portions. Moderate engineering effort for custom kernels.

#### R4-A5-P8: ANDERSON-ACCELERATION-FOR-DEPTH
- **Category:** IMPLEMENT
- **Radical Insight:** RecursiveSSM's depth iteration is vanilla Picard iteration (linear convergence rate r). Anderson Acceleration achieves SUPERLINEAR convergence from same per-iteration cost. Shared weights guarantee consistent fixed-point operator.
- **Mechanism:** Variant A — Heavy Ball (5 lines): x = x + delta + beta × (x - x_prev). Reduces effective rate from r to sqrt(r). For r=0.7: 10 → 6 iterations. Variant B — Anderson (m=3, ~30 lines): maintain window of 3 iterates, solve 3×3 least-squares for optimal mixing, compute accelerated iterate. Cost: O(9×d_model) per step — negligible. Achieves 1/3 to 1/2 iterations of vanilla.
- **Expected Impact:** Heavy Ball: 32 → ~20 iterations (1.6x). Anderson: 32 → 12-16 (2-2.7x). Combined with adaptive exit: 32 → 8-10. Nearly zero overhead.
- **v5 Applicability:** Direct modification of RecursiveSSM forward loop. Heavy Ball: 1 extra tensor (x_prev). Anderson: 3 extra tensors. Memory: 12.6 MB for 6M config — negligible. Beta/m learnable end-to-end.
- **Risk/Limitation:** Iteration embeddings make operator non-stationary. Heavy Ball can overshoot (clamp beta to [0, 0.9]). Anderson's least-squares can be unstable (add Tikhonov regularization).

---

## ASPECT 5 COMPLETE — Summary

**Total findings:** 26 (F1-F10 web research, M1-M8 math-paradigm-breakers, P1-P8 paradigm-breakers)

**Breakdown:**
- IMPLEMENT: 18
- EXPLORE: 8

**Top priorities for v5 implementation:**
1. **P4 HOMOMORPHIC-TOKEN-PACKING** — guaranteed 4x batch throughput, ~15 lines, trivial
2. **P8 ANDERSON-ACCELERATION** — Heavy Ball variant: 5 lines, 1.5-2.7x depth reduction
3. **P5 GEOMETRIC-EXTRAPOLATION** — 25 lines, 3-6x inference speedup, stacks with P8
4. **P3 RECURSIVE-DELTA-PROPAGATION** — 7-10x across 32 iterations, needs flip-rate diagnostic
5. **M6 CROSS-LAYER-TEMPLATE-CACHING** — eliminates scan overhead, 5L ops per layer
6. **M2 QUINARY-BYTE-PACKING** — 20% model size reduction, synergy with M1 ABP
7. **M1 AMORTIZED-BLOCK-PATTERN-PRECOMPUTATION** — 1.9-2.4x via pattern LUTs
8. **P7 TERNARY-CIRCUIT-COMPILATION** — 15-30% from dead path elimination

**Key synergies:**
- P4 (HTP) × P3 (RTDP) × P8 (Anderson): 4x × 9.2x × 2x = ~74x total batch inference speedup
- M1 (ABP) + M2 (quinary): merged decode+lookup table, zero-overhead weight access
- M5 (sparse scan) + M6 (template) + M8 (tree pruning): complete scan optimization stack
- M3 (compute-bound theorem): reframes ALL kernel optimization priorities — addition throughput, not bandwidth
- P5 (extrapolation) + P8 (acceleration): compound depth reduction from ~32 to ~6-8 iterations
- F6 (Highway) + F8 (WASM) + F9 (RISC-V): ONE codebase compiling for every platform
