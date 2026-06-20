# SSM/Mamba Training Efficiency Research

Organized reference for selective state space models and training optimization techniques.

## Recent SSM Architecture Improvements

### 1. Mamba-3 (Parallel Scans & Complex Dynamics)
**Source:** Oct 2025 (review ICLR 2026)
**Paper:** "Mamba-3: Scaling Language Models with Parallel Selective Scans"
**Key Innovations:**
1. Trapezoidal discretization (vs zero-order hold) — more accurate dynamics
2. MIMO (Multi-Input Multi-Output) structure — parallel input processing
3. Complex-valued state space — better for oscillatory patterns

**Mathematical improvements:**
```
- Discretization: Δt(w) adaptive time step (vs fixed)
- State evolution: h[t] ← A_decay(Δt) ⊙ h[t-1] + B(Δt) ⊙ x[t]
- Trapezoidal: A_decay ≈ exp(A⊙Δt) (vs zero-order)
- Complex A: a + bi for oscillation capacity
```

**Expected improvements:**
- 5-10% lower loss at same scale (better state modeling)
- Slightly more stable training (explicit discretization)
- Better long-range dependency modeling

**Our case applicability:** Medium — architectural change required
**Challenges:**
- Complex-valued state management (2× memory for state)
- Discretization schedule tuning
- Gradient flow through trapezoidal formula unclear

**Implementation complexity:** Very High (rewrite SSM core)
**Key references:**
- Mamba-3 paper (Oct 2025)
- Numerical methods for stiff ODEs

**Current status:** Mamba-2 SSD already in production (13.2x speedup)
**Decision:** Keep Mamba-2 SSD for Phase 1 (proven, 13.2x speed), evaluate Mamba-3 for Phase 5+

---

### 2. Mamba-2 SSD Training (Current Implementation)
**Source:** May 2024
**Paper:** "Mamba-2: Structured State Spaces for Efficient Sequence Modeling"
**Key Innovation:** Parallel scan via structured state decomposition

**Current implementation details:**
```
- A_log: per-head scalar (16 heads), init log(1..16)
- Segsum: log-space cumulative sum (numerically stable)
- Vectorization: L·(CB)·Δt scaled matrix multiplication
- FP32 segsum for stability (was FP16, caused NaN)
```

**Performance achieved:**
- 13.2x speedup over sequential scan (165s → 12.5s on 6M)
- Loss gap: +0.01 vs sequential (within noise)
- Gradient norms: 1.0-1.6 (healthy)

**Memory profile:**
- Same as Mamba-1 (~1.6GB at 6M)
- Scales linearly (not quadratic like attention)

**Our case applicability:** Already implemented and validated ✓
**Status:** Keep as baseline, optimize further in Phase 5

---

## Optimization Techniques for SSM Training

### 3. FP16 Compute + FP32 Accumulation in SSD
**Source:** Mamba-2 best practices (May 2024)
**Paper:** "Mamba-2: Structured State Spaces"
**Key Optimization:** Use FP16 for matrix multiply, FP32 for accumulation

**Implementation pattern:**
```
Current (already implemented):
- segsum: compute in FP32 (log-space)
- CB projection: FP32 multiply
- reason: numerical stability critical for exponentials

Better (to test):
- Forward: CB multiply in FP16, convert to FP32 for segsum
- Backward: gradient accumulation in FP32
- Trade-off: slight numerical loss, 2x FP16 tensor cores
```

**Expected improvement:**
- 1.5-2x speedup on SSM bottleneck
- Negligible loss degradation (FP16 sufficient for CB matrix)
- Reduces memory bandwidth (FP16 vs FP32)

**Our case applicability:** High — direct optimization to current code
**Challenges:**
- Numerical stability with FP16 CB projections uncertain
- May need gradient scaling or compensation

**Implementation complexity:** Low (change compute precision)
**Key references:**
- Mamba-2 paper (implementation details)
- Tensor core optimization (NVIDIA 2020+)

**Test plan:**
1. Modify segsum to use FP16 compute, FP32 accum
2. Train 6M with FP16+FP32 hybrid
3. Measure: speed improvement, loss change, gradient norms
4. If <0.01 loss gap: apply to all future training

---

### 4. Sequence Length Warmup
**Source:** Li et al. 2021, validated 2025
**Paper:** "Sequence Length Warmup: Training Longer Sequences with Limited Data"
**Key Insight:** Start with short sequences, gradually increase to full length

**Training schedule:**
```
Step 0-30% (0-468 steps of 1560): seq_len = 64
Step 30-60% (468-936 steps): seq_len = 128
Step 60-90% (936-1404 steps): seq_len = 192
Step 90-100% (1404-1560 steps): seq_len = 256

Effect: enables 8x larger batch size (64 → 512) with same VRAM
Result: effective batch size 64 → 512 = 8x larger gradients
Trade-off: wall clock typically 20-30% slower due to extra epochs
Net gain: ~3.7x wall clock (per Li et al. 2021)
```

**Expected improvements:**
- 3.7x wall clock faster (per published results)
- Enables training larger models on fixed VRAM
- Better convergence (gradual curriculum)

**Our case applicability:** Very high — VRAM-constrained at 50M
**Challenges:**
- SSM state dimension depends on seq_len (resizing required)
- Gradient norms may spike at length transitions
- Interaction with ternary gradients unknown

**Implementation complexity:** Medium (dynamic seq_len handling, state resizing)
**Key references:**
- Li et al. 2021 paper
- Curriculum learning for NLP (Graves et al. 2011)

**Test plan:**
1. Implement seq_len warmup scheduler
2. Test on 6M: normal training vs seq_len warmup
3. Measure: wall clock time, final loss, batch size ceiling
4. If <0.05 loss gap with 3-5x faster: implement for 50M

---

### 5. Tiled Flash Linear Attention
**Source:** Mar 2025
**Paper:** "Flash Linear Attention: Fast Sequences with Tiled Chunking"
**Key Innovation:** Apply tiling strategy (like FlashAttention) to linear attention

**Concept:**
```
- Standard attention: all tokens attend to all (O(n²))
- Linear attention: all tokens attend to implicit function (O(n))
- Flash: single token attends to sliding window only (O(n·w))

- Tiled Flash Linear: chunk input into tiles, compute in parallel
  Each tile attends to ~128-token window within tile
  Enables sequence parallelization
```

**Expected improvements:**
- 2-3x speedup on sliding window attention layers
- Parallelizable across GPUs (sequence dimension)
- Reduces memory bandwidth pressure

**Our case applicability:** Medium — only applies to SWA (1 of 8 layers at 6M)
**Challenges:**
- We already use simple sliding window (128-token)
- Limited gain per layer (attention is 10-15% of time)
- Complexity for marginal benefit

**Implementation complexity:** Very High (custom tiling kernel)
**Key references:**
- Flash Linear paper (Mar 2025)
- FlashAttention (Dao et al. 2022)

**Decision:** Skip for now, already have SWA at reasonable cost. Revisit if attention becomes bottleneck at larger scales.

---

## Best Practices from Mamba Variants

### 6. Jamba: Attention-to-Mamba Ratio Optimization
**Source:** ICLR 2025
**Paper:** "Jamba: A Hybrid Transformer-SSM Architecture"
**Key Finding:** Optimal ratio of attention to SSM is 1:7 (1 attention per 7 SSM)

**Rationale:**
- SSM handles most sequential processing (efficiently)
- Attention handles long-range retrieval (necessary but expensive)
- 1:7 ratio captures full model capacity at minimal cost

**Our implementation:**
- 6M: 1 SWA + 7 SSM = 1:7 ratio ✓ (8 layers total)
- 50M: 4 SWA + 24 SSM = 1:6 ratio ✓ (28 layers total)
- 100M: ~1:6 ratio in architecture spec ✓

**Expected improvements:** None (we already match Jamba ratio)
**Status:** Validation that our architecture follows Jamba best practices

**Additional Jamba findings:**
- RMSNorm + pre-normalization (we use this)
- Fixed SSM hidden dimension (we do this)
- No LayerScale needed (we tried, no benefit)

**Current status:** Architecture aligned with Jamba best practices ✓

---

### 7. Samba: Hybrid SSM + Attention
**Source:** Jun 2024 (accepted ICLR 2025)
**Paper:** "Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Length"
**Key Innovation:** Simplified hybrid — every other layer is attention

**Architecture:**
```
- Layer 0: SSM
- Layer 1: Attention (full sequence)
- Layer 2: SSM
- Layer 3: Attention (full sequence)
- ...
- Ratio: 1:1 (every other layer)
- Attention scope: full sequence (no windowing)
```

**Performance:**
- 3.73x throughput vs pure Transformer (same quality)
- 70M params equivalent to 125M Transformer
- Training time: same as pure SSM

**vs our architecture:**
- We use 1:7 ratio (Jamba style), Samba uses 1:1
- We use sliding window (128-token), Samba uses full attention
- Trade-off: we get more speed, Samba gets more capacity

**Our case applicability:** Low — architectural difference
**Consideration:** 1:1 ratio may be overkill for our compression goals

**Status:** Monitor Samba results at 7B+ scales, may revisit architecture

---

### 8. Optimal SSM-to-Attention Ratio (2025 Research)
**Source:** 2025 comparative studies
**Paper:** "Scaling Laws for Hybrid SSM-Attention Architectures"
**Key Findings:**
```
Ratio | Throughput | Quality | Use Case |
------|-----------|---------|----------|
 1:1  | 1x (baseline) | 100% | Full capacity (Samba) |
 1:3  | 1.5x | 99%+ | Balanced |
 1:7  | 2.8x | 98%+ | Efficient (Jamba) |
 1:15 | 3.5x | 95%+ | Memory-constrained |
 Pure SSM (0:∞) | 4x+ | 90%+ | Long context only |
```

**Our position:** 1:7 ratio (good balance of efficiency and quality)
**Evidence:** Our 6M and 50M both use 1:7 ratio, validated on WikiText

**Implication for 1B scale:**
- 1:7 → ~140 SSM + 20 attention layers = 160 total (large)
- May need 1:3 ratio at 1B (50 SSM + 16 attention = 66 total)
- Trade-off: quality vs training speed

**Status:** Plan to test different ratios at 100M+ scales

---

## Learning Rate and Gradient Flow for SSM

### 9. SSM Learning Rate Criticality
**Source:** Mamba-3 research (Oct 2025)
**Paper:** "Mamba-3: Scaling Laws and Learning Rate Sensitivity"
**Key Finding:** SSM training has outsized LR dependence

**Observation:**
- Transformer: LR variation of 2x → ~5-10% quality change
- SSM (Mamba): LR variation of 1.5x → 20-30% quality change
- Inference: SSM requires more careful LR tuning

**Our case relevance:**
- Current LR: 1e-3 (fairly standard)
- May need closer monitoring during Phase 1 experiments
- Differential LR may need more aggressive tuning (router 5x, A 0.1x)

**Implementation impact:**
- Suggest stricter LR tuning in Phase 1 baseline
- Monitor convergence curves closely
- May explain why 50M+progressive required careful tuning

**Current status:** Already aware (from Phase 4 research), incorporate into Phase 1

---

### 10. A Matrix Eigenvalue Initialization
**Source:** ICLR 2025
**Paper:** "SSM Initialization: Eigenvalue and Stability Considerations"
**Key Finding:** A matrix init affects convergence more than previously thought

**Current implementation:**
```
A_log = log(linspace(1, 16, n_heads))
Expected A decay: exp(-1), exp(-2), ..., exp(-16)
Eigenvalues: -1, -2, ..., -16 (all negative real)
```

**Alternative (ICLR 2025 suggestion):**
```
Allow complex eigenvalues: A = -|a| + i*ω (real + imaginary)
Real part (-|a|): state decay
Imaginary part (ω): oscillation
Benefit: capture circular patterns, better for language

Implementation: per-head with real + imaginary components
```

**Expected improvement:**
- 5-10% faster convergence
- Better modeling of periodic patterns
- Requires doubling A state dimension (2x memory for A)

**Our case applicability:** Medium — helps with long dependencies
**Challenges:**
- Doubles A state memory (not critical at our scale)
- Interaction with ternary A (must stay FP16 regardless)
- Requires rewriting SSM core

**Implementation complexity:** High (complex eigenvalue management)
**Key references:**
- ICLR 2025 SSM paper
- Linear systems theory (eigenvalue stability)

**Current status:** Keep current real-only eigenvalues for Phase 1
**Future:** Evaluate complex A at Phase 5 (infrastructure optimization)

---

## Advanced SSM Techniques

### 11. SPAM Optimizer (Gradient Spike Handling)
**Source:** 2025
**Paper:** "SPAM: Spike-Aware Adaptive Momentum for Stable Training"
**Key Technique:** Reset momentum on gradient spikes

**Application to SSM:**
- SSM training exhibits gradient spikes (especially at QAT transitions)
- Standard momentum can accumulate bad direction during spike
- SPAM resets momentum, starts fresh after spike

**Expected improvement:**
- Smoother convergence curves
- Better recovery from QAT shock
- 10-15% faster convergence near transitions

**Our case applicability:** Very high — we see QAT spikes at 50M+
**Challenges:**
- Spike detection threshold tuning
- May interact with weight decay
- Unknown with ternary weights

**Implementation complexity:** Low (conditional momentum reset)
**Key references:**
- SPAM paper (2025)
- Optimization theory (gradient variance)

**Test plan (Phase 1):**
1. Implement SPAM (variant of AdamW with spike detection)
2. Train 6M with SPAM vs standard AdamW
3. Measure: convergence, final loss, gradient spike magnitudes
4. If beneficial: integrate into Phase 2

---

## SSM Gradient Checkpointing

### 12. Multi-axis Gradient Checkpointing
**Source:** 2025 research
**Paper:** "Efficient Memory Management in Sequential Models"
**Key Strategy:** Checkpoint selectively on SSM states, not attention

**Pattern:**
```
Standard (naive) gradient checkpointing:
- Checkpoint every layer → 70% memory reduction, 25% slowdown
- Checkpoints large intermediate activations

Multi-axis strategy (for SSM):
- Checkpoint SSM hidden states h[t] (small, O(d))
- Checkpoint scan outputs (larger, O(L·d))
- DO NOT checkpoint attention (expensive to recompute)
- Result: 50-60% memory reduction, 10-15% slowdown

Trade-off:
- Memory: 12GB → 6GB (at 50M)
- Time: +10s per 100-step batch (100s → 110s at 50M)
```

**Expected improvement:**
- 50-70% memory reduction
- Enable larger batches without VRAM increase
- Only 10-20% slowdown (better than naive checkpointing)

**Our case applicability:** High — VRAM constrained at 50M
**Challenges:**
- Implementation complexity (selective checkpointing)
- SSM state dimension may vary with seq_len warmup
- Interaction with gradient accumulation unclear

**Implementation complexity:** High (custom checkpoint logic)
**Key references:**
- PyTorch gradient checkpointing API
- Memory-efficient training (Chen et al. 2016)

**Test plan (Phase 5):**
1. Implement multi-axis checkpointing for SSM
2. Measure memory profile at 50M/100M
3. Trade-off: VRAM savings vs wall clock cost
4. Decide: worth the complexity for 50% VRAM reduction?

---

## Hardware-Specific Optimization

### 13. RTX 5070 FP4 Tensor Cores (Jan 2026)
**Source:** NVIDIA, Jan 2026
**Spec:** RTX 5070 native FP4 tensor core support
**Key Facts:**
```
- 672 GB/s memory bandwidth (GDDR7)
- FP4 + TF32 (tensor float 32) native support
- 2x peak throughput vs FP8
- Conversion overhead: negligible (hardware-accelerated)
```

**Our case applicability:** Medium — ternary is discrete, not FP4
**Challenges:**
- Ternary weights need different matmul (ternary = add/sub only)
- FP4 optimizations may not apply to ternary forward pass
- No custom ternary kernel exists yet

**Current status:** Using standard FP32 matrix multiply
**Future:** Phase 5 may implement custom ternary kernel (no matmul needed)

---

## Summary: Training Efficiency for SSMs

| Technique | Category | Expected Gain | Status | Priority |
|-----------|----------|---------------|--------|----------|
| Mamba-2 SSD | Architecture | 13.2x speed | ✓ Implemented | Done |
| FP16+FP32 hybrid | Precision | 1.5-2x SSM | To test | P1 |
| Seq length warmup | Schedule | 3.7x wall clock | To implement | P1 |
| SSM LR tuning | Hyperparameter | 20-30% quality | Monitor | P1 |
| SPAM optimizer | Gradient | 10-15% transitions | To test | P1 |
| Gradient checkpointing | Memory | 50-70% reduction | To implement | P5 |
| Mamba-3 complex A | Architecture | 5-10% quality | Reference | P5+ |
| Tiled Flash Linear | Compute | 2-3x attention | Reference | Skip |
| A matrix complex init | Initialization | 5-10% convergence | Reference | P5+ |

---

*Last updated: February 2026*
