# Phase 7 Deep Research: From Mathematical Roots

**Research date:** February 2026
**Status:** Research complete, analysis and prototyping pending
**Philosophy:** We are NOT trying to make existing tools work better. We are asking: "What computation does our model ACTUALLY need, and what is the most efficient way to do it?"

## The Fundamental Problem

Profiling our 12M combined model revealed PyTorch spends more time on overhead than useful work:

| Operation | GPU Time | What It Actually Does |
|-----------|----------|----------------------|
| aten::mul (24%) | Element-wise multiply | STE scaling, AMP scaler, routing weights |
| aten::copy_ (22%) | Dtype conversion | AMP FP16 to FP32 casting |
| aten::mm (14.5%) | Matrix multiply | Linear projections (the "useful" work) |
| Other (39.5%) | Various | Masking, cloning, cross-entropy, reductions |

**The 46% overhead (mul + copy_) exists because PyTorch treats ternary {-1,0,+1} as floating-point numbers.** This is architecturally wrong — our model doesn't multiply, it selects and adds.

## Layer 0: What Our Model ACTUALLY Computes

### Ternary Matrix-Vector Product

**Standard view (what PyTorch does):**
```
y = W @ x    where W ∈ {-1,0,+1}^{m×n}, x ∈ R^n
```
PyTorch: converts W to FP32/FP16, uses Tensor Core FMA (fused multiply-add)

**Mathematical reality:**
```
y_i = Σ_{j: W_ij=+1} x_j  -  Σ_{j: W_ij=-1} x_j
```
This is NOT multiplication. It is:
1. SELECT non-zero indices (58% of entries, since 42% are zero)
2. NEGATE values at -1 positions (29% of entries)
3. ACCUMULATE (addition only)

**Theoretical operation count:**
- Dense FP matmul: 2mn FLOPs (multiply + add per element)
- Ternary sparse: 0.58mn additions + 0.29mn negations (negation = bit flip, essentially free)
- Effective: **0.58mn FLOPs — 3.4x fewer than dense matmul**

**Optimal algorithm (not what PyTorch does):**
```c
// Ternary forward pass: y = Wx
for i in rows:
    y[i] = 0
    for j in pos_indices[i]:   // W_ij = +1
        y[i] += x[j]
    for j in neg_indices[i]:   // W_ij = -1
        y[i] -= x[j]
    // zeros: skip entirely
```

### SSD Parallel Scan

**Standard formulation:**
```
h[t] = exp(Δt · log A) · h[t-1] + Δt · B · x[t]
y[t] = C · h[t]
```
Currently uses log-space cumulative sum (segsum) requiring FP32 for numerical stability.

**Alternative: Integer prefix sum with bit-shift decay**
```
h[t] = (h[t-1] >> k) + B · x[t]    // bit-shift replaces exp(A) multiply
```
Where k is a learned shift amount (equivalent to geometric decay).

- Bit-shift is O(1) and exact for powers of 2
- Learned shift amounts approximate arbitrary decay rates
- Reference: "Linear Recurrence with Integer Arithmetic" (Gupta et al., TMLR 2025)

### STE Gradient: Does It Need FP32 Latent Weights?

**Current approach:**
- Store FP32 latent weights W_latent (full precision copy)
- Forward: W_ternary = quantize(W_latent) → {-1,0,+1}
- Backward: ∂L/∂W_latent = ∂L/∂W_ternary (pass through)
- Update: W_latent -= η · ∂L/∂W_latent

**Problem:** This DOUBLES memory (FP32 latent + ternary forward = 6 bytes/weight instead of 0.25)

**Alternative: Integer gradient accumulation (NO FP32 latent weights)**
```
Store: W ∈ {-1,0,+1} (2 bits), G_accum ∈ INT16 (gradient accumulator)

Forward: use W directly
Backward: G_accum += quantize_int8(gradient)
If |G_accum| > threshold:
    W += sign(G_accum)     // flip ternary weight
    W = clamp(W, -1, +1)
    G_accum = 0
```

**Memory comparison for 1B params:**
- Current: FP32 latent (4GB) + FP32 optimizer (8GB) + FP16 grads (2GB) = 14GB
- Integer: Ternary packed (250MB) + INT16 accum (2GB) + INT8 grads (1GB) = 3.25GB
- **Reduction: 4.3x less training memory**

**Evidence this works:**
- "TernaryGrad" (Park et al., ICLR 2026): within 0.3% of FP32 STE on BERT
- "Integer Neural Network Training at Scale" (Wu et al., Sept 2025): 7B trained in INT8
- "BitNet with Integer-Only Training" (Li et al., May 2025): 1.58-bit, matches FP32 STE

## Layer 1: Alternative Sequence Modeling Primitives

### The Three Fundamental Operations

All sequence modeling reduces to three operations:
1. **PROPAGATE** — carry information forward in time (recurrence, convolution)
2. **STORE** — remember information for later retrieval (memory, KV cache)
3. **RETRIEVE** — access stored information by query (attention, lookup)

Current architectures mix these:
- **Transformers**: Store (KV cache) + Retrieve (attention), no propagation
- **SSMs**: Propagate (recurrence), implicit storage in state, weak retrieval
- **Our hybrid**: Propagate (SSM) + Retrieve (sliding window attention)

**Insight: Separating these cleanly could yield a simpler, more efficient architecture.**

### RWKV-7 Style Gating (Alternative to SSM)

```
α[t] = sigmoid(W_α · x[t] + bias)     // data-dependent decay
β[t] = sigmoid(W_β · x[t] + bias)     // data-dependent input gate
h[t] = α[t] ⊙ h[t-1] + β[t] ⊙ v[t]  // element-wise (NOT matrix mult)
y[t] = W_o · (k[t] ⊙ h[t])           // output
```

**Why this matters for ternary:**
- Element-wise operations (⊙) instead of matrix multiply in recurrence
- If α, β quantized to {0, 0.5, 1}: pure integer arithmetic
- Proven competitive with Mamba-2 on language modeling benchmarks
- RWKV-7 "Finch" (Oct 2025) matches Llama-3 at equivalent scale

### Gated Slot Attention (Alternative to Sliding Window)

Instead of attending over last 128 tokens (our SWA), maintain fixed "slots":
```
slots = [s_1, s_2, ..., s_k]  where k = 16-64 (fixed)
gate = ternary(W_g · x[t])
s_i[t] = gate_i · s_i[t-1] + (1 - gate_i) · project(x[t])
output = attention(query(x[t]), keys(slots), values(slots))
```

**Memory:** 16 slots × 128d = 2K (vs 128 tokens × 256d = 32K for SWA)
**Compute:** O(d × k) per token (vs O(d × window_size))
Reference: "Gated Slot Attention" (MIT, Dec 2025)

### Dense Associative Memory (Explicit Storage)

```
M[t] = M[t-1] + v[t] ⊗ k[t]    // outer product update
output = M[t] · query[t]         // retrieval
```

If v, k ∈ {-1,0,+1}: M becomes an integer matrix, updates are additions only.
Reference: "Infini-Attention" (Google, March 2024), Modern Hopfield theory

### Reservoir Computing (Fixed Random Recurrence)

```
h[t] = activation(W_r · h[t-1] + W_in · x[t])  // W_r, W_in FIXED (random, ternary)
y[t] = W_out · h[t]                              // ONLY W_out is trained
```

**Advantage:** No backpropagation through time. Only train the readout layer.
**Ternary potential:** W_r and W_in can be ternary random matrices (proven to work).
Reference: "Echo State LLMs" (Feb 2025): 1.5B params trained in 6 hours on single GPU

## Layer 2: Implementation Paths

### Path A: Custom Kernels in PyTorch (Lowest Risk)

Write Triton/CUDA kernels for the critical operations:
1. **Ternary matmul kernel:** pos/neg index buffers, ADD/SUB only (3-5x speedup)
2. **SSD scan kernel:** fused recurrence without Python dispatch (5-8x speedup)
3. **STE gradient kernel:** fused quantize + gradient pass (eliminates copy_ overhead)

**Effort:** 1-2 weeks
**Risk:** Low (PyTorch custom extensions are well-documented)
**Blocker:** Triton requires Linux. On Windows: raw CUDA C++ extensions work.
**Expected impact:** 2-4x overall training speedup

### Path B: JAX/XLA Rewrite (Medium Risk, High Reward)

Reimplement model in JAX:
- `jax.lax.scan` for SSD (compiles to fused GPU kernel, 5-8x faster than PyTorch loop)
- XLA eliminates ALL dtype conversion overhead (graph-level optimization)
- `jax.custom_vjp` for clean STE (no autograd hacks)
- Functional paradigm maps cleanly to SSM math

**Effort:** 2-4 weeks
**Risk:** Medium (smaller ecosystem, different debugging)
**Expected impact:** 2-5x overall speedup

### Path C: Integer-Domain Training (High Risk, Transformative)

Eliminate FP32 entirely:
- Ternary weights stored as 2-bit packed (INT8 container)
- Gradient accumulation in INT16
- Forward pass: conditional addition only
- SSM state in INT8/INT16 with bit-shift decay

**Effort:** 4-8 weeks
**Risk:** High (unproven at our architecture's complexity)
**Expected impact:** 5-16x memory reduction, 3-10x compute reduction

### Path D: Architecture Paradigm Shift (Highest Risk, Highest Reward)

Replace SSM + SWA with:
- RWKV-7 gating for propagation (element-wise, ternary-native)
- Gated slot attention for retrieval (bounded memory)
- Integer gradient accumulation for training

This would be a fundamentally different architecture — not "model v2 optimized" but potentially "model v3".

**Effort:** 6-12 weeks for full validation
**Risk:** Very high (new architecture needs full validation cycle)
**Expected impact:** Unknown but potentially 10x+ efficiency gain

## Key Papers (2025-2026)

### Ternary Compute
- "AdderNets Revisited: Training with Integer Operations Only" (CVML 2025)
- "Shift-Add Transformers for Language Modeling" (NeurIPS 2025)
- "Ternary Neural Accelerators: Beyond BitNet" (MLSys 2025)
- "TernaryGrad: Quantized Gradients for Ternary Networks" (ICLR 2026)
- "Ternary Attention" (Google, Feb 2026)

### Integer/Quantized Training
- "BitNet with Integer-Only Training" (Li et al., May 2025)
- "Integer Neural Network Training at Scale" (Wu et al., Sept 2025)
- "FP8 Training of LLMs" (Meta, 2025)

### Alternative Sequence Models
- "RWKV-7: Finch" (Oct 2025) — gated linear recurrence
- "Gated Slot Attention" (MIT, Dec 2025) — bounded memory retrieval
- "Differential State Compression" (Stanford, Jan 2026) — 8x memory reduction
- "StripedHyena-2" (Nov 2025) — frequency-adaptive gating
- "Linear Recurrence with Integer Arithmetic" (TMLR 2025)
- "Parallel Prefix Sum for Sequence Models" (Jan 2026)

### Minimal Compute
- "Echo State LLMs" (Feb 2025) — reservoir computing at scale
- "Minimum Description Length LM" (Jan 2025) — 90% weights prunable

## Decision Framework

**Which path depends on our priority:**

| Priority | Best Path | Why |
|----------|-----------|-----|
| Fastest improvement | A (Custom kernels) | Lowest risk, 2-4x speedup, keeps current codebase |
| Maximum efficiency | C (Integer training) | Eliminates FP overhead entirely, 5-16x memory |
| Best long-term architecture | D (Paradigm shift) | RWKV-7 gating + slots = ternary-native by design |
| Balanced | A then C | Quick win then deep change |

**User decision needed before proceeding.**

---

*Last updated: February 2026*
