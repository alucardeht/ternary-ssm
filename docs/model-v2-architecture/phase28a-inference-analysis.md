# Phase 28-A: Inference Implementation — Deep Analysis

## Source Material
- ~39 findings from Research Cycles 1+2 (research-cycle2-inference.md)
- Architecture-enforcer deep review
- Math-paradigm-breaker rigorous validation

## Critical Discovery: LM Head is the Hidden Bottleneck

**None of the 39 research findings identified this.** Both the architect and mathematician independently discovered it.

| Component | Size at 50M | % of Total |
|-----------|-------------|-----------|
| SSM layer ternary weights | ~12 MB | 23% |
| SSM FP16 components (dt, conv, A, norms) | ~1 MB | 2% |
| SWA layer weights + KV cache | ~4.5 MB | 9% |
| **LM Head (FP16, tied embedding)** | **39.1 MB** | **66%** |
| Total | ~57 MB | 100% |

The LM Head is **3.25x larger** than all layer weights combined. It does NOT fit in L2 cache (48 MB). This changes every throughput calculation:

| Scenario | Theoretical Max (50M) |
|----------|----------------------|
| Current (all from HBM) | ~11,260 tok/s |
| Layers in L2, LM head from HBM | ~15,700 tok/s |
| **Factored LM head (all in L2)** | **81,000-107,000 tok/s** |
| Factored + ternary LM head | ~210,000 tok/s |

**Factored LM Head is the #1 priority. Without it, nothing else matters.**

## Corrected Speedup Claims

| Finding | Research Claim | Real Value | Why |
|---------|---------------|-----------|-----|
| Python loop bottleneck | 10-20x | 1.2-1.4x | CUDA Graph already solves 97% of Python overhead |
| Persistent kernel | 50-100x | 17x (no LM fix) / 87x (with) | Limited by HBM for LM Head |
| POPCOUNT on GPU | 10-50x | ~1.0x | Batch=1 is memory-bound, compute savings irrelevant |
| Delta skip | 2-3x | 1.3-1.6x | Branching overhead + realistic sparsity |
| Speculation + TC | 5-12x | 0.4-2.5x | Draft tokens cost as much as full tokens (LM Head) |
| Layer factoring | 6.8x | 1.4-1.8x | Random ternary f≈0.35, trained unlikely >0.5 |

## Non-Multiplicative Composition

Research projected speedups as multiplicative. Reality: many target the SAME bottleneck.

**Redundant pairs (pick one, not both):**
- Mega-kernel vs Persistent kernel → both reduce kernel launch overhead
- Trit-LUT vs POPCOUNT → both optimize GEMV compute

**Realistic composition:**
- Kernel overhead reduction (best approach): 3-5x
- GEMV throughput improvement (best approach): 1.5-3x
- Speculative decoding: 1.5-2x
- GPU-only sampling: 1.2-1.3x
- Delta-skip (composable): 1.3-1.5x
- **Total realistic: 8-30x** (not 50-100x)

## What Research Missed

1. No 50M benchmark exists — all numbers extrapolated from 6M
2. Conv1d state update: ~5 kernel launches per layer, never analyzed
3. RMSNorm: 36-54 kernel launches total, never discussed
4. V2 kernel (multi-row, autotuned) exists but NEVER used
5. SWA layer optimization completely ignored
6. At 1B scale, model doesn't fit L2 → completely different strategy needed
7. POPCOUNT is a CPU optimization, not GPU (wrong target hardware)

## Implementation Plan (Corrected Priority Order)

### Step 0: Measurement (BEFORE any code changes)
- Benchmark 50M on current engine → actual tok/s baseline
- Profile with nsight or timing → actual time breakdown
- Measure activation delta sparsity → validate delta-skip
- Test V2 kernel at 50M → may already be faster

### Step 1: Factored LM Head (HIGHEST PRIORITY — enables everything)
- Factor W_lm (32K×640) into W1 (640×256) @ W2 (256×32K)
- SVD initialization from existing embedding
- Total: 16 MB (fits in L2 alongside 12 MB layers = 28 MB < 48 MB)
- Quality validation: PPL before/after
- Effort: 1-2 days. Impact: 2-4x standalone, enables 87x with persistent kernel

### Step 2: SSM Layer Mega-Kernel
- Fuse ~15 launches per SSM layer into 1 Triton kernel
- Inline ternary GEMV bit-unpacking (Triton can't call other Triton kernels)
- Eliminate ~240 kernel launches at 50M
- Effort: 1-2 weeks. Impact: 2-4x

### Step 3: GPU-Only Sampling
- Move argmax/top-k to GPU, eliminate PCIe sync
- Chain into CUDA Graph or use device-side launch
- Start with greedy, add top-p later
- Effort: 2-3 days. Impact: 1.2-1.4x (certain)

### Step 4: Delta-Activation Layer Skip
- Skip entire layers where activation delta < threshold
- Still update SSM state (approximate: h = dA * h, decay only)
- Periodic full recompute every N=16 tokens
- Depends on Step 0 sparsity measurement
- Effort: 1 week. Impact: 1.3-1.8x

### Step 5: Speculative Self-Draft
- 2-4 layers as drafter, verify K candidates through full model
- REQUIRES factored LM head (Step 1) to be net-positive
- REQUIRES alpha > 0.58 (measure on trained model first)
- Need batch>1 ternary GEMV (V2 kernel or new batched kernel)
- Effort: 2-3 weeks. Impact: 1.5-2x

## Realistic Performance Projection

| Step | Technique | Multiplier | Cumulative (from ~300 tok/s 50M baseline) |
|------|-----------|------------|------------------------------------------|
| 0 | Current engine at 50M | 1x | ~200-400 tok/s (unknown) |
| 1 | Factored LM Head | 2-4x | ~500-1,200 tok/s |
| 2 | SSM Mega-Kernel | 2-3x | ~1,000-3,600 tok/s |
| 3 | GPU-Only Sampling | 1.2-1.3x | ~1,200-4,700 tok/s |
| 4 | Delta Layer Skip | 1.3-1.5x | ~1,500-7,000 tok/s |
| 5 | Speculative Decode | 1.5-2x | ~2,300-14,000 tok/s |

**Conservative: 3,000-5,000 tok/s. Optimistic: 8,000-14,000 tok/s.**

## Scale-Dependent Strategy (from mathematician)

| Scale | Model Size | Fits L2? | Optimal Stack | Max tok/s |
|-------|-----------|----------|---------------|-----------|
| 6M | 5-17 MB | YES | Persistent kernel + GPU loop | 160K-450K |
| 50M | 28 MB (factored) | YES | Factored LM + persistent + speculation | 81K-107K |
| 100M | 36 MB (factored) | YES | Same as 50M | ~83K |
| 370M | 86 MB (factored) | NO | HBM streaming + speculation + TC | ~7.8K |
| 1B | 241 MB (factored) | NO | Disaggregated GPU+CPU | ~4.3K |
