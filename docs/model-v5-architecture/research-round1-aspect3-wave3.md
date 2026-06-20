# Round 1, Aspect 3: Compute-on-Demand & Adaptive Architectures — Wave 3 (Paradigm)

## Wave 3 — Paradigm-Breakers (2 researchers, Feb 2026)

### COMPUTATION MODEL PARADIGM SHIFTS

#### v5-R1-A3-P1: Ternary Butterfly Factorization
- **Radical idea:** Replace dense ternary matmul O(n²) with product of O(log n) butterfly matrices, each with entries in {-1,0,+1}. Each 2x2 block has 3⁴=81 options, selected via Gumbel-Softmax.
- **Impact:** in_proj: from 33.5M to ~213K ops (157x speedup). Weight storage: from 8.4 MB to 53 KB per projection. For 7B: potentially ~50 MB total weights.
- **Key insight:** in_proj + out_proj = 94% of per-layer compute. The SSM recurrence is only 4%. Everyone optimizes the 4% — this attacks the 94%.
- **Prior art:** Monarch Mixer (2023) proved butterfly works for FP16. Ternary version is natural extension.
- **Combinable with:** P3 (random butterfly = zero storage, zero training), P4 (permutation sharing of butterfly factors)
- **Feasibility:** MEDIUM-HIGH — well-established math, ternary constraint adds novelty
- **Verdict:** PROMISING

#### v5-R1-A3-P2: Resonant SSM with Ternary Spectral Gating
- **Radical idea:** Place A-matrix eigenvalues ON the unit circle (pure rotation, no decay). Information stored as frequency patterns. Ternary B/C = spectral gating: +1 excites, -1 inhibits, 0 ignores specific frequency bands.
- **Impact:** Theoretically UNLIMITED context (no information decay). State capacity: N × 16 = 2048 bits per position (vs exponentially decaying in standard SSM). Could REPLACE sliding window attention.
- **Key insight:** Forgetting controlled by ternary B/C (explicit write/erase) not by eigenvalue magnitude (passive decay). Analogous to neural oscillation theory (theta/gamma bands).
- **Risk:** Marginal stability (|λ|=1 exactly). Need eigenvalue normalization after each gradient step.
- **Feasibility:** MEDIUM — S4/DSS already used complex eigenvalues, but unit circle targeting is new
- **Verdict:** PROMISING

#### v5-R1-A3-P3: Seed-Generated Random Ternary Projections
- **Radical idea:** in_proj is NOT LEARNED. It is a pseudo-random ternary matrix generated from a seed. Zero storage, zero training. Only out_proj, x_proj, SSM params are trained.
- **Justification:** Johnson-Lindenstrauss lemma guarantees random projections preserve distances. Compressed sensing RIP: random ternary matrices faithfully represent sparse signals. Biological precedent: cerebellar granule cells perform random expansion.
- **Impact:** Eliminate 35% of all parameters (~612 MB in 7B). No STE gradient noise for those params. 35% faster training (fewer params to update). Peak memory: O(d) per row (on-the-fly generation) instead of O(d²).
- **Combinable with:** P1 (random butterfly = O(n log n) random projection = best of both worlds)
- **Feasibility:** HIGH — strongest theoretical backing, simplest implementation
- **Verdict:** PROMISING (TOP — test immediately)

#### v5-R1-A3-P4: Permutation-Based Ternary Weight Sharing
- **Radical idea:** Store K=4-8 "archetype" ternary matrices. Derive all 56 layer variants via learned row/column permutations. Permutations preserve all matrix properties (singular values, norm, rank). Application: O(n) index reordering.
- **Impact:** K=8 tiers: 448 MB → 67 MB (6.7x). K=2 tiers: 448 MB → 19 MB (30x). Permutation storage: 2.7 MB total (negligible).
- **Key insight:** Permuted ternary matrix is still ternary. Expressiveness > ALBERT sharing (which uses identical matrices). Acts as regularization.
- **Training:** Sinkhorn operator: continuous doubly-stochastic → hard permutation with temperature annealing.
- **Feasibility:** MEDIUM-HIGH — ALBERT proves sharing works; permutation is strictly more expressive
- **Verdict:** PROMISING

#### v5-R1-A3-P5: SSM-Guided Ternary Weight Flipping (Self-Modifying)
- **Radical idea:** Model modifies its own ternary weights during inference. Flipping -1↔+1 = single bit XOR (free for ternary). Tiny meta-controller (h_t → sparse flip mask). Model literally reconfigures itself per token.
- **Impact:** In-context adaptation without attention. Potential replacement for SWA layers. Unique to ternary — FP16 weight modification requires full-precision compute.
- **Key constraint:** 0.001%-0.01% flip rate per token = bounded perturbation. Over 2048 tokens: 0.03% total weights modified.
- **Connection:** Fast Weight Programmers (Schmidhuber 1991), NTMs, but gradient-FREE at inference.
- **Feasibility:** LOW (MOONSHOT) — training meta-controller through discrete flips is hard
- **Verdict:** SPECULATIVE

### INFERENCE ENGINE PARADIGM SHIFTS

#### v5-R1-A3-P6: Instruction-Stream Compilation (Weights = Code)
- **Radical idea:** Ternary weight W_ij ∈ {-1,0,+1} = one of three CPU instructions: ADD, SUB, NOP. Compile each layer into native machine code. No weight file. Model IS a shared library (.so/.dll). "Loading model" = dlopen().
- **Impact:** Zeros generate NO instructions (60% sparsity = 60% eliminated). Instruction prefetch gets near-optimal behavior (linear code). For MoE: inactive experts = never-fetched code pages. OS demand-paging = free expert scheduler.
- **Key insight:** For ternary, weight value and arithmetic operation are in BIJECTION. This bijection doesn't hold for any quantization >3 levels.
- **Feasibility:** HIGH for CPU (especially d≤2048). Code generation via LLVM or assembler templates.
- **Impact estimate:** 1.5-3x throughput, ZERO additional RAM
- **Verdict:** PROMISING

#### v5-R1-A3-P7: Periodic State Automaton
- **Radical idea:** Ternary A matrix close to signed permutation has finite period T (A^T = I). Precompute powers via repeated squaring: store A¹, A², A⁴, ..., A^(2¹⁶). Any A^s via binary decomposition = log₂(T) matrix-vector multiplies.
- **Novel insight (CROSS-CUTTING):** SSM A matrices should MINIMIZE zeros (each zero ERASES state). MLP weights should MAXIMIZE zeros (speed). Router weights should be LOW-RANK. This "heterogeneous sparsity" principle is NEVER discussed in ternary literature.
- **Impact:** O(log T) for arbitrary-step state update vs O(s) sequential. RAM: 16 extra A copies per layer (negligible for small d_state).
- **Feasibility:** MEDIUM — constraining A to signed permutation may limit expressiveness
- **Verdict:** PROMISING (novel insight about heterogeneous sparsity regardless of main proposal)

#### v5-R1-A3-P8: Bitplane POPCOUNT Engine
- **Radical idea:** Ternary matmul decomposed into bitwise AND + POPCOUNT instead of add/subtract. Input quantized to INT8 → 8 bitplanes. Each row: AND bitplane with pos_mask, POPCOUNT, AND with neg_mask, POPCOUNT, difference × 2^bit.
- **Speedup math:** Standard: n additions per output = 4096. Bitplane AVX-512 VPOPCNTDQ: 2 × 8 × ceil(4096/512) = 128 ops. **32x speedup.** Even scalar POPCNT: 4x.
- **Key constraint:** Requires INT8 activation quantization (well-validated: LLM.int8(), SmoothQuant). Entire forward pass becomes pure integer — no floating point.
- **Combinable with:** P6 (compile POPCOUNT ops into instruction stream = stacked advantage)
- **Feasibility:** HIGH — all hardware primitives exist today, INT8 quantization is proven
- **Verdict:** PROMISING (TOP — implementable immediately, 4-32x throughput)

#### v5-R1-A3-P9: Collapsed Path Network (Layer Fusion)
- **Radical idea:** Product of ternary values along any network path is itself ternary (product of {-1,0,+1} = {-1,0,+1}). Fuse K=2 adjacent layers: W_fused = W₂ · W₁ (integer matrix). Re-ternarize → half the depth.
- **SSM-specific insight:** SSM state evolution is LINEAR (no nonlinearities). A^K computed exactly. Precompute for parallel state computation.
- **Impact:** K=2 fusion: ~1.5-2x throughput, 2-10% quality loss estimated.
- **Feasibility:** LOW-MEDIUM for full fusion (quality risk), HIGH for SSM A^K precompute
- **Verdict:** NEEDS-MORE-WORK

#### v5-R1-A3-P10: Moment-Bypass Routing
- **Radical idea:** For routing decisions, compute output STATISTICS instead of exact values. For ternary row with |P| positive and |N| negative entries, applied to input with mean μ, variance σ²: E[y] = (|P|-|N|)·μ, Var[y] = (|P|+|N|)·σ². Cost: O(1) per output neuron (precomputed counts + input stats).
- **Two-stage routing:** Stage 1: moment-based screening O(n+m). Stage 2: exact computation for 2k candidates O(2kn). For 64 experts, k=2: 13x faster routing.
- **Extension to MoD:** 70-90% of token skip/compute decisions made in O(1) via moments.
- **Feasibility:** MEDIUM — i.i.d. assumption rough but two-stage approach provides safety net
- **Verdict:** PROMISING

## Key Insights — A3 Wave 3

1. **94% of compute is in projections, not SSM** — P1 butterfly attacks the RIGHT bottleneck (157x potential)
2. **Random projections are mathematically justified** — P3 eliminates 35% of params with theoretical guarantees (JL lemma, RIP)
3. **Bitplane POPCOUNT = 4-32x on existing hardware** — P8 is implementable TODAY, no model changes needed
4. **Instruction-stream compilation** — P6 eliminates weight loading entirely for ternary (weights ARE code)
5. **Heterogeneous sparsity principle** — P7's insight: A matrices should be DENSE, MLP SPARSE, Router LOW-RANK
6. **Self-modifying architecture is ternary-exclusive** — P5: weight flipping is free (1 bit XOR), impossible for FP16
7. **P1+P3+P4 combined = 1000x reduction** — Random butterfly with permutation sharing = near-zero storage, O(n log n) compute

## Aspect 3 Complete Summary (Waves 1 + 2 + 3)

### Total Findings: 34
- Wave 1 (Web): 8 findings (F1-F8)
- Wave 2 (Math): 16 formulations (M1-M16)
- Wave 3 (Paradigm): 10 proposals (P1-P10)

### TOP Findings for Implementation Priority

| Rank | Finding | Source | Impact | Viability |
|------|---------|--------|--------|-----------|
| 1 | P8: Bitplane POPCOUNT | Wave 3 | 4-32x throughput | HIGH (today) |
| 2 | M2+M5: Exact Ternary Exit + Hamming | Wave 2 | 2-3x, zero loss | HIGH (exact) |
| 3 | P3: Random Projections | Wave 3 | 35% fewer params | HIGH (JL theorem) |
| 4 | M13: 8x Throughput Ceiling | Wave 2 | Foundation | HIGH (roofline) |
| 5 | P1: Butterfly Factorization | Wave 3 | 157x per projection | MEDIUM-HIGH |
| 6 | P6: Instruction Compilation | Wave 3 | 1.5-3x, zero RAM | HIGH (CPU) |
| 7 | M14: Unified Memory Zero-Copy | Wave 2 | 11x vs discrete | HIGH (hw-dep) |
| 8 | P4: Permutation Sharing | Wave 3 | 6.7-30x compression | MEDIUM-HIGH |
| 9 | M8: Minimum Viable Router | Wave 2 | 2-4x adaptive | HIGH |
| 10 | M15: SSM State Predictor | Wave 2 | Hides NVMe latency | MEDIUM-HIGH |

### Combined Architecture Vision

**Inference Engine:**
Bitplane POPCOUNT (P8) → Instruction-compiled (P6) → Hamming early exit (M2+M5) → Speculative decoding (F2+M9)

**Memory Efficiency:**
Random projections (P3) × Permutation sharing (P4) × Butterfly factorization (P1) × GF(3) basis (A2-M5) × MoE NVMe streaming (A1)

**Adaptive Compute:**
Moment-bypass routing (P10) → MoD skip (F4+M3) → Ternary router (M8) → Early exit (M2+M5)

**Theoretical Maximum (multiplicative):**
POPCOUNT (8x) × Butterfly (10x) × Adaptive depth (2x) × Speculative (2x) × Streaming (5x) = **1,600x** vs naive FP16 autoregressive

**Realistic Target:** 50-200x improvement over FP16, achievable by implementing top-5 findings.
