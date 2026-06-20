# Research Round 2, Aspect 1: Training Throughput — Wave 2 (Math-Paradigm-Breakers)

## Summary

Two math-paradigm-breakers explored novel mathematical formulations for training optimization. Together they produced 10 novel paths, with significant synergies. After deduplication, 8 unique formulations remain.

## HIGH VIABILITY (implement first)

### R2-A1-M1: Margin-Based Sparse Backward
- **Principle:** Only weights near quantization boundaries need accurate gradients. Weights far from boundaries won't change ternary value regardless of gradient.
- **Formulation:** Compute margin m(i,j) = distance to nearest quantization boundary. Only compute full weight gradient (outer product x * g^T) for critical set C = {(i,j) : m(i,j) < delta}. Typically |C| = 5-10% of all weights.
- **Speedup:** 2-3x backward pass (weight gradient is 40-60% of backward, computing 10-20% of entries)
- **Quality:** Provably minimal impact — skipped computations have zero effect on ternary output
- **Feasibility:** MEDIUM — custom backward per layer, sparse outer product
- **Synergies:** Combines with M4 (temporal recycling) and M2 (TMC counters)

### R2-A1-M2: Ternary Momentum Counters (TMC) — Replace Adam
- **Principle:** Adam maintains 96 bits per weight (shadow + m + v) to decide among 3 values. This is 60x information-theoretic overhead. Replace with integer counter tracking directional pressure.
- **Formulation:** Per weight: q_i in {-1,0,+1} (2 bits), mu_i INT16 momentum, s_i INT8 volatility tracker. Update: mu <- decay*mu + quantize(grad). Flip when |mu| > threshold. Adaptive threshold via gradient sign volatility.
- **Memory:** 26 bits/weight vs 96 bits (Adam) = **3.8x compression**. At 7B: 49.6 GB → 13.4 GB optimizer state.
- **Speedup:** All-integer optimizer update (no FP32 math for 59% of params)
- **Quality:** Convergence follows sign-SGD theory. Bounded lattice eliminates divergence risk.
- **Feasibility:** MEDIUM — replace optimizer for ternary params, keep Adam for continuous
- **Synergies:** Natural integration with M3 (sign gradients) and M5 (fused operator)

### R2-A1-M3: Sign-Quantized Gradients for Ternary Layers
- **Principle:** Weight gradient g_ij = dL/dy_i * x_j. For TMC, we only need sign or rough magnitude. sign(g_ij) = sign(dL/dy_i) XOR sign(x_j) — a 1-bit operation.
- **Formulation:** Bitwise outer product: pack 256 sign bits per AVX2 register, XOR gives 256 gradient signs per instruction. 16x throughput vs FP16 multiply. Multi-bit extension (INT4): 4x throughput.
- **Speedup:** 4-16x on gradient computation for ternary layers
- **Quality:** Sign-SGD converges comparably for overparameterized nets. 4-bit sufficient for direction+magnitude.
- **Feasibility:** EASY-MEDIUM — XOR/SIMD implementation straightforward
- **Synergies:** Feeds directly into TMC counters (M2). Enables fused operator (M5).

### R2-A1-M4: Temporal Gradient Recycling
- **Principle:** 95-99% of ternary weights don't change between steps. Gradient computation for unchanged weights produces stale but correlated information. Only compute fresh gradients for recently-flipped + near-boundary weights.
- **Formulation:** Track flipped set F_t via bitmap comparison. Fresh gradient for F_t ∩ C_t (flipped AND near-boundary). EMA + noise for stable weights. Full refresh every K steps.
- **Speedup:** Combined with M1 (margin): 5-8x on weight gradient computation
- **Quality:** Bounded staleness from periodic refresh. Low risk.
- **Feasibility:** EASY-MEDIUM — bitmap tracking is cheap, sparse outer product existing tech
- **Synergies:** Combines perfectly with M1 (margin sparsification)

### R2-A1-M5: Fused Backward-Update Operator
- **Principle:** Standard pipeline: write gradients → read+update optimizer → read+quantize = 3 memory passes. Fuse into single pass: read (upstream grad, activation, counter, weight) → compute+update → write (counter, weight).
- **Formulation:** Single operator F(delta_i, x_j, mu, q, tau) → (mu', q'). Reduces 36.25 bytes/weight memory traffic to 4.5 bytes/weight = **8x reduction**.
- **Speedup:** 15-28% total training speedup (optimizer is bandwidth-bound)
- **Quality:** Identical convergence (pure computation reordering)
- **Feasibility:** MEDIUM — custom CUDA kernel or PyTorch custom Function
- **Synergies:** Fuses M2+M3 into hardware-efficient implementation

## MEDIUM VIABILITY (validate after high-priority)

### R2-A1-M6: Mirror Descent / Single-Logit Optimizer
- **Principle:** Multiplicative Weights Update on the ternary simplex. For ternary: g_{+1} = -g_{-1}, g_0 = 0, so ONE cumulative logit L per weight suffices. p_{+1} = exp(L)/Z, p_{-1} = exp(-L)/Z.
- **Formulation:** L(i,j) += eta * dL/dy * x_i. Weight decision: w = sign(L) * [|L| > threshold]. One FMA per weight per step.
- **Memory:** 2 bytes/weight (FP16 logit) vs 12 bytes (Adam) = **6x reduction**. At 7B: 49.6 GB → 8.3 GB.
- **Quality:** MWU convergence theory applies but needs deep network validation
- **Feasibility:** MEDIUM — elegant but convergence empirically unvalidated for deep nets
- **Note:** More elegant than TMC (M2) but less proven. Consider as TMC alternative if TMC converges.

### R2-A1-M7: Stochastic Coordinate Descent + Gradient Caching
- **Principle:** Full backward pass every M steps, cache gradient. Between refreshes, coordinate descent on random subsets using cached gradients. Staleness bounded by ternary flip rate.
- **Formulation:** Full cycle every M=10 steps. Intermediate: sample K=N/100 coordinates, flip if cached gradient suggests improvement. Cost: (N/M + K) per step vs N standard = 9x reduction.
- **Quality:** Staleness bounded by 2*F*L_smooth where F = flips since refresh
- **Feasibility:** MEDIUM-HIGH — well-understood theory, needs M* tuning
- **Synergies:** Independent of M1-M5, can be combined

### R2-A1-M8: Algebraic Backward Collapse (INT16 Jacobians)
- **Principle:** Product of 2 ternary matrices = integer matrix with entries in [-d, +d], fits INT16. Precompute blocked Jacobian products for faster backward gradient propagation.
- **Formulation:** M_{l,l+1} = W_{l+1}^T * W_l^T (integer matrix via popcount on bit-packed weights). Backward uses L/2 INT16 matmuls instead of L FP16 matmuls. INT16 is 2-4x faster than FP16.
- **Speedup:** 2-4x on input gradient computation
- **Quality:** Exact computation (no approximation)
- **Feasibility:** MEDIUM — needs custom INT16 backward kernels, popcount precomputation
- **Limitation:** Non-linearities between layers prevent full multi-layer collapse

## SPECULATIVE (moonshot)

### R2-A1-M9: Gradient-Free Ternary Training (SPSA / Forward-Only)
- **Principle:** Eliminate backward pass entirely. Sample ternary perturbations, evaluate loss change via forward-only passes. Forward is cheap (add/sub), backward is expensive (FP16 matmul for weight gradients).
- **Formulation:** P perturbations, each flipping K random weights. g_est = (1/P) * Σ (L_p - L_0) * delta_p. For P=20: 20 forward passes cost ~5x one backward. No activation storage needed.
- **Memory:** Eliminates activation storage: 600 MB at 6M, 4.3 GB at 7B
- **Quality:** SPSA convergence O(1/T^{1/3}) vs SGD O(1/T^{1/2}) — slower
- **Feasibility:** HARD — convergence rate degradation may be prohibitive
- **Note:** Valuable at extreme scale where activation memory is the binding constraint

## Key Insight: The Ternary Optimizer Revolution

**The unifying theme across all findings:** Adam is massively over-provisioned for ternary optimization. The ternary constraint reduces the optimization to a sequence of 3-way classification decisions, requiring at most log2(3) = 1.58 bits of information per weight. Current training uses 96 bits per weight in optimizer state — a 60x overhead.

**The recommended stack (M1+M2+M3+M4+M5):**
1. TMC counters replace Adam (3.8x memory reduction)
2. Sign-quantized gradients feed TMC (4-16x gradient compute speedup)
3. Margin-based sparsification + temporal recycling (5-8x fewer gradient computations)
4. Fused operator combines everything into single memory pass (8x bandwidth reduction)

**Estimated combined impact:**
- Optimizer memory: 49.6 GB → 13.4 GB at 7B (3.8x)
- Backward throughput: 3-6x faster
- Optimizer step: 8x faster (bandwidth-bound → compute-bound)
- Total training throughput: estimated 2-4x improvement

## For Paradigm-Breakers (Wave 3)

These findings propose mathematical replacements for Adam, STE, and autograd. Wave 3 should explore:
1. Can the ENTIRE training pipeline be reformulated as integer-only computation?
2. What would a custom training framework (not PyTorch) look like for ternary SSM?
3. Are there biological/neuromorphic training paradigms that naturally fit ternary?
