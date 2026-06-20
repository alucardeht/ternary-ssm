# Phase 7.7: Path D — RWKV-7 Gating → Custom SSM Design

**Date Created:** February 14, 2026
**Status:** Planning (architecture research complete, implementation pending)
**Objective:** Explore alternative recurrence architecture (RWKV-7 style gating) as potential path to all-ternary training.

---

## Why Path D?

Our SSM design requires an A matrix (FP16 minimum per Q-S5 research). Ternary in projections works, but the recurrent state evolution is inherently matrix multiply. RWKV-7 uses **element-wise gating instead of matrix multiply**, potentially enabling all-ternary training at the architectural level.

**Hypothesis:** An architecture based on data-dependent element-wise gates (like RWKV-7) is more naturally compatible with ternary quantization than linear SSM recurrence.

---

## The Problem We're Solving

| Aspect | SSD (our baseline) | RWKV-7 (alternative) |
|--------|-------------------|----------------------|
| Recurrence | h[t] = A×h[t-1] + B×x[t] | h[t] = g[t] ⊙ h[t-1] + v[t] |
| Matrix ops | A (FP16 minimum) | None (element-wise only) |
| Ternary-compatible | Projections only | Potentially gates too |
| Typical accuracy | Strong (SSM theory proven) | Strong (RWKV-7 scales to 14B) |
| Inference latency | 1 matrix multiply per token | 1 element-wise add per token |

Key insight: If decay gate g[t] could be ternary (or integer), the entire recurrence becomes integer arithmetic.

---

## Deep Research Findings (Feb 14, 2026)

### RWKV-7 Architecture

**State shape:** 64×64 matrix per head (NOT scalar, surprising)
- 6 heads × 64×64 = 24,576 per position
- Our SSM: 512 scalar state per layer

**Projections per layer:** ~8-10 (W_k, W_v, W_r, W_g, W_a, W_o, etc.)
- Our SSD: 3-4 (W_b, W_c, W_o)
- RWKV-7 more "projection-heavy", SSM more "recurrence-heavy"

**Decay gate function:**
```
w_t = sigmoid(W_w * x_t)  ∈ (0, 1)
     typically learned to be ∈ (0.6, 1)  // stable range
```

**Known constraint (Q-S5, July 2025):**
- SSM decay factor below 8-bit precision causes "dramatic collapse"
- RWKV-7 gates are decay factors — same vulnerability applies
- Below 8-bit (e.g., ternary quantization) likely problematic for gates

### Why Initial Hypothesis Was Incomplete

Original estimate: "Element-wise gating = 10x+ speedup from ternary compute"

**Reality:**
- Element-wise gating is indeed faster than matrix multiply
- BUT: RWKV-7 has 2x more projections than SSD
- Projections still require W matmul (not ternary-simplified by gate design)
- Net gain much smaller than 10x initially estimated

**Revised expectation:** ~2-4x speedup from removing A matrix multiply, offset by extra projections and gate overhead. Roughly neutral compute with better architecture alignment for ternary.

---

## Test Plan: 4 Sequential Phases

### Phase D.1: RWKV-7 Full-Precision Baseline

**Purpose:** Establish if RWKV-7 can match our SSD at 6M scale on WikiText-103.

**Implementation:**
- 6M parameter model, same vocab/embedding as SSD baseline
- 6 heads × 1024 hidden = comparable to SSD 8-layer architecture
- 500 training steps (consistent with Phase 4.5 benchmarks)
- WikiText-103, batch=32

**Metrics:**
| Metric | Target | Notes |
|--------|--------|-------|
| Perplexity | 70-85 | SSD baseline is 77.10 |
| Tok/s | 50K-80K | Expect ~same as SSD |
| VRAM | 3-4GB | Expect ~same as SSD |
| Wall clock | 45-60s | 500 steps benchmark |

**Success criteria:**
- Within 5% of SSD baseline loss (4.1129)
- No convergence issues (smooth loss curve)
- Reasonable throughput (not obviously slower)

**If successful:** → Proceed to D.2
**If worse than 10% loss:** → Investigate root cause (may indicate RWKV-7 incompatible with our settings)

---

### Phase D.2: RWKV-7 Hybrid Ternary

**Purpose:** Test selective ternary (projections ternary, gates FP16) for stability.

**Ternary breakdown:**
- **Ternary:** W_k, W_v, W_r, W_o (projections)
- **FP16:** w_t, a_t (decay gates, attention gates — critical for learning curve)

**Rationale:**
- Gates are "learning knobs" that control state evolution
- Quantizing gates may break learning dynamics (empirically tested at D.3)
- Projections are feature extractors — more amenable to ternary

**Measurements:**
| Metric | Target |
|--------|--------|
| Loss degradation vs D.1 | <5% |
| Gate distributions | w_t ∈ (0.6, 1), a_t typical range |
| Ternary weight distribution | 29/42/29 or learned distribution |
| Training stability | No loss spikes or gradient explosions |

**Key measurement:** Gate values over training
- Are gates staying in [0.6, 1] range?
- Do FP16 gates provide enough expressiveness?
- Or do gates saturate (all 1s, all 0.5s)?

**If successful:** → Proceed to D.3
**If >5% degradation:** → Skip D.3 and go to D.4 (gates not ternary-compatible)

---

### Phase D.3: RWKV-7 Full Ternary Attempt

**Purpose:** Empirically test ternary gates despite Q-S5 theory warnings.

**Gate quantization strategy:**
- Decay gate w_t: normal range (0.6, 1) → ternary approximation
  - Option A: {0.5, 0.75, 1.0} (center around 0.75)
  - Option B: {0.6, 0.8, 1.0} (map to actual range)
  - Option C: {-1, 0, +1} scaled by learned alpha
- Implement via ternary STE on gate logits

**Full ternary model:**
- All projections: TernaryLinear
- All gates: TernaryLinear (or custom ternary gate function)
- Recurrent state: FP16 (state stays high precision)

**Measurements:**
| Metric | Purpose |
|--------|---------|
| Loss vs D.2 | Quantify gate quantization impact |
| Gate distribution | Are ternary gates being used, or collapse to 0/+1? |
| Training stability | Gradient norms, loss spikes, convergence rate |
| Actual vs theoretical degradation | Q-S5 predicts "dramatic collapse" — measure actual |

**Expected:** Visible loss increase (gates sensitive to quantization).

**Data point value:** We'll empirically measure how bad ternary gates actually are. Q-S5 warns of issues below 8-bit, but RWKV-7's narrower range (0.6-1) might be more robust than typical SSM (0-1).

**If <10% degradation:** → Reconsider gate quantization as viable path
**If >15% degradation:** → Gate quantization not practical, proceed to D.4 with FP16 gates

---

### Phase D.4: Custom Simplified SSM

**Purpose:** Design a novel architecture combining all insights from D.1-D.3.

**Design constraints:**
- Maximize ternary compatibility from first principles
- Element-wise or low-projection-count design (not 8-10 projections)
- Maintain comparable loss to SSD/RWKV-7 baselines
- Target: all ternary (100% of trainable params except embeddings)

**Architecture outline (example, will refine after D.1-D.3):**

```
Input x_t (embedding)
  ↓ [Linear] → c_t (candidate state, FP16)
  ↓ [TernaryLinear] → r_t (gate logits)
  ↓ [Sigmoid] → g_t = sigmoid(r_t) (decay gate)

State update (element-wise):
  h_t = g_t ⊙ h_{t-1} + (1-g_t) ⊙ c_t

Output:
  y_t = [TernaryLinear](h_t)
```

**Key differences from RWKV-7:**
- Scalar state (not matrix-valued)
- 2-3 projections (not 8-10)
- Simpler gate function (e.g., single scalar gate per element)
- All ternary except candidate generation

**Benchmark:** 50M model if feasible (memory permitting)
- Compare vs RWKV-7 50M baseline (if D.1 tested at 50M)
- Measure loss, throughput, ternary %
- Test on WikiText-103 (full dataset if time permits)

**Success criteria:**
- <5% loss gap vs SSD (4.1129 at 6M scale)
- 100% ternary weights (except embedding/output heads)
- Integer training feasible (combined with Path C)
- Throughput ≥ SSD (no regression)

---

## Timeline & Sequencing

| Phase | Duration | Blocker | Next |
|-------|----------|---------|------|
| **D.1** | 1-2 days | None | D.2 if success |
| **D.2** | 1 day | D.1 results | D.3 if <5% gap |
| **D.3** | 1 day | D.2 results | D.4 if <15% gap |
| **D.4** | 3-5 days | D.3 data | Phase 8 or pivot |

**Total estimated:** 1 week for full exploration.

**Decision points:**
- If D.1 fails (>10% worse): RWKV-7 incompatible, skip to Phase 8
- If D.3 shows gates work (<10% loss): novel paper contribution, fast-track D.4
- If D.3 shows gates fail (>15% loss): FP16 gates only, proceed D.4 with constraints

---

## Relationship to Other Paths

| Path | Focus | Relationship to D |
|------|-------|-------------------|
| **Path C** | Integer gradient accumulation | Orthogonal (D tests architecture, C tests training) |
| **Path A** | Custom CUDA kernels | Complementary (A for SSD, D for new architecture) |
| **Path B** | JAX/XLA rewrite | Deferred (wait for architecture stability) |

**Combined strategy (if D.4 succeeds):**
- D.4 provides ternary-native architecture
- Path C provides integer training for D.4
- Phase 5 infrastructure (AMP, GradCkpt) applies to D.4
- Phase 3 optimizer (AdamW+WSD) applies to D.4

---

## Success Metrics for Phase D

**Overall:** Path D succeeds if we find an architecture that:
1. Matches SSD baseline quality (within 5% loss)
2. Is naturally compatible with all-ternary training
3. Provides a viable path to full-ternary models

**Specific outcomes:**
- **D.1:** RWKV-7 as a reference point (may not adopt, but informative)
- **D.2:** Determine if gates can be ternary without catastrophic failure
- **D.3:** Empirical data on gate quantization sensitivity
- **D.4:** Novel architecture ready for Phase 8 (μP scaling laws)

**If all D phases fail:** Architecture paradigm shift may not be viable at this scale. Return focus to Path C (integer training on SSD) and Phase 8 (HP transfer).

---

## References

- **RWKV Paper:** "RWKV: Reinventing RNNs for the Transformer Era" (Peng et al., 2023)
- **RWKV-7 Architecture:** (no formal paper yet; based on community implementation)
- **Q-S5 (quantization for SSM):** "Q-S5: Sub-8-bit Quantization of Selective SSMs" (July 2025)
- **HESTIA:** Soft ternary quantization (Phase 2 winner)
- **Integer Training:** Phase C.1 findings on gradient accumulators

---

*Last updated: February 14, 2026 — Planning phase, ready for implementation*
