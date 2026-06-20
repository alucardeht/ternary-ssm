# 03 — Mathematical Foundations of Weight Quantization

## Why This Matters for Model v2

If ternary weights ({-1, 0, +1}) work, we eliminate matrix multiplication entirely — the single most expensive operation in neural networks. Instead of float multiply-accumulate, we get add/subtract/skip. CPU becomes faster than GPU. But WHY does reducing 16 bits to 1.58 bits per weight not destroy the model? We need to understand this deeply before building on it.

---

## 1. Why Ternary Works — The Configuration Space Argument

### The Misleading Intuition

Ternary = 1.58 bits/weight. FP16 = 16 bits/weight. That's ~10% of the information. Shouldn't the model be ~10% as good?

**No.** This assumes each weight independently encodes information. Neural networks don't work that way.

### The Collective Configuration Space

For a weight matrix W ∈ R^(m×n) with ternary constraints, the total configuration space is:

```
|Ω_ternary| = 3^(m×n)
```

For a modest 1024×1024 layer: 3^(1,048,576) ≈ 10^500,000 configurations.

For a 3B parameter model: 3^(3,000,000,000) configurations.

This is an astronomically large space. The question isn't "how much info per weight?" but **"how many distinguishable functions can this configuration space represent?"** — and the answer is: far more than needed.

### Overparameterization Makes It Work

**Theorem (informal):** If overparameterization factor k satisfies:
```
k > c · log(1/ε) / log(3)
```
then ternary networks can ε-approximate the target function.

At 3B+ parameters, the configuration mesh is dense enough to approximate virtually any function on compact domains.

**This is why there's a scale threshold** — below ~1B params, the mesh is too sparse for complex tasks.

---

## 2. Information Geometry — Manifold Preservation

### The Neural Network Manifold

Weights define a Riemannian manifold M where:
- Each point = a network configuration
- Metric = Fisher Information Matrix
- Geodesics = optimal training trajectories

### What Ternary Does to the Manifold

Ternary quantization projects continuous M_continuous onto discrete submesh M_discrete.

**What's preserved:**
1. **Local connectivity** — adjacent configs remain close
2. **Global topology** — critical points (minima, saddles) map to nearby discrete points
3. **Geodesic structure** — optimal paths approximated by discrete steps

**Distortion bound:**
```
d_M(F(p₁), F(p₂)) ≤ d_M(p₁, p₂) + 2·max|α·Δ|
```
With proper scaling, distortion is negligible relative to loss landscape curvature.

### Connection to Lottery Ticket Hypothesis

**Strong Lottery Ticket Hypothesis (SLTH):** Randomly initialized overparameterized networks contain subnetworks that achieve target performance.

**Quantization-SLTH Connection (Kumar & Natale, 2024):** Quantization is structured sparsity. Ternary weights SELECT a discrete subset preserving essential manifold structure. The network doesn't need continuous precision — it needs sufficient overparameterization.

### Low-Dimensional Training Manifold

**Key finding:** Training trajectories explore a ~100-1000 dimensional manifold even in billion-parameter networks. Ternary only needs to approximate THIS low-dimensional manifold, not the full parameter space.

**Key insight for Model v2:** 1.58 bits/weight is sufficient because the effective degrees of freedom << total parameters.

---

## 3. The Role of Full-Precision Activations

### BitNet Computation

```
y = σ(W_ternary · x_fp16)

W_ternary ∈ {-1, 0, +1}^(m×n)
x_fp16 ∈ R^n  (BF16)
y ∈ R^m  (BF16)
```

Operation per element:
```
w = +1: y += x    (addition)
w = -1: y -= x    (subtraction)
w =  0: skip      (no-op)
```

**Output is full precision.** Only the routing decisions are ternary.

### Separation of Concerns

- **Weights** store learned STRUCTURE (patterns, filters, associations)
- **Activations** carry input-specific INFORMATION through the network

The ternary weights are **routing functions**, not precise multipliers. They decide WHAT information flows WHERE, while activations carry the actual content at full precision.

### Output Distribution Analysis

For input x ~ N(μ, σ²) and ternary weights W_t:

```
E[W_t · x] ≈ E[W_fp · x]    (unbiased with proper α)
Var[W_t · x] ≈ α² · Var[W_fp · x]    (variance preserved)
```

By Central Limit Theorem, for large fan-in (n >> 1), ternary layer output converges to Gaussian. **No pathological distributions** (no heavy tails, no multi-modality).

### CRITICAL: Implications for SSM State Evolution

If ternary weights project inputs into SSM parameters:
```
A = α_A · W_A,ternary · input    → A is BF16 (full precision)
B = α_B · W_B,ternary · input    → B is BF16 (full precision)
```

**The SSM sees full-precision A, B, C matrices.** Ternary constrains HOW they're computed, not their final precision. Stability depends on learned α and input distribution, not on the ternary constraint itself.

**Caveat:** If ternary projections produce poorly conditioned A matrices, stability degrades. This is a training challenge (learn better projections), not a fundamental limitation.

---

## 4. Quantization Error — Does It Accumulate or Cancel?

### Layer-Wise Error

At layer l:
```
ε^(l) = (W_fp^(l) - W_t^(l)) · σ(z^(l-1))
```

### The Critical Question

Does error grow as O(L) (linear — catastrophic) or O(√L) (sublinear — manageable)?

### Error Cancellation (QEP Framework)

Quantization errors have these properties:
```
E[ΔW^(i)] ≈ 0                    (unbiased with proper scaling)
Cov(ΔW^(i), ΔW^(j)) ≈ 0         (independent layers)
```

By CLT:
```
||ε^(L)||² ≈ Σᵢ ||ΔW^(i)||² · ||G_i||²    (sum of variances)
```

This is **O(√L) growth** — sublinear. Much better than worst-case O(L).

### Error Compensation Strategy

QEP (Quantization Error Propagation) adjusts later layers to compensate:
```
With compensation:    ||ε^(L)||² ≤ C · L^(1/2) · max||ΔW||²
Without:              ||ε^(L)||² ≤ C · L      · max||ΔW||²
```

The √L vs L scaling makes deep ternary networks feasible.

### Conditions for Error Cancellation

1. **Sign diversity** — errors alternate sign across layers
2. **Scale balance** — no single layer dominates
3. **Activation smoothing** — nonlinearities dampen error propagation

**Key insight for Model v2:** Error grows sublinearly, not linearly. A 32-layer ternary network has ~√32 ≈ 5.7x the single-layer error, not 32x. This is manageable.

---

## 5. Training Dynamics — The Straight-Through Estimator

### Why Native Training Is Required

Post-training quantization (FP16 → ternary) fails because:
1. Loss landscape is different for continuous vs discrete weights
2. Representation collapse — lost info can't be recovered
3. Scale incompatibility — FP16 learns magnitudes, ternary learns binary decisions

### The STE Paradox

Quantization is non-differentiable: ∂Quantize(w)/∂w = 0 almost everywhere.

**STE approximation:**
```
Forward:  ŵ = Quantize(w)
Backward: ∂L/∂w ≈ ∂L/∂ŵ    (pass gradient through unchanged)
```

Mathematically **wrong**. Empirically **works**.

### Why STE Works (Coarse Gradient Theory)

**Theorem (Yin et al., ICLR 2019):** The coarse gradient from STE is positively correlated with the true gradient:

```
⟨g_coarse, g_true⟩ > 0
```

STE doesn't need the exact gradient — just a **descent direction**. Because quantization boundaries are stable during training, the coarse approximation suffices.

### Loss Landscape with Ternary Constraints

The loss L_ternary(α, Δ, W_discrete) has:
- **Continuous subspace:** scaling factors α (smooth gradients)
- **Discrete subspace:** weight assignments W (STE approximation)
- **Hybrid optimization:** alternate between exact (α) and approximate (W) updates

### Convergence Guarantee

**Theorem:** Low-precision SGD with STE converges to critical point, but:
- Slower pace than FP16
- Higher steady-state error
- Requires larger batches and careful LR scheduling

**Key insight for Model v2:** Training will take more iterations than standard models. We need to budget for this and use larger batches.

---

## 6. The 1B-3B Scale Threshold

### Empirical Observation

| Scale | Ternary vs FP16 |
|-------|-----------------|
| <1B | Significant degradation |
| 1-3B | Approaching parity |
| 3B+ | Matches FP16 |
| 70B+ | Sometimes EXCEEDS FP16 (regularization effect) |

### Why This Threshold Exists

**Configuration density argument:**
```
Density = 3^n / Volume(Function_Space)
```

As n increases, discrete mesh becomes exponentially denser.

**Spectral density (Random Matrix Theory):**

At n ≈ 1B, eigenvalue spectrum of ternary weight matrices becomes rich enough to represent:
- Long-range dependencies (small eigenvalues)
- High-frequency patterns (large eigenvalues)
- Multimodal distributions (intermediate eigenvalues)

Below this, the spectrum is too sparse.

### Error Scaling Formula

```
ε_total ≈ L^(1/2) · ε_layer(w)
ε_layer(w) ≈ exp(-c · w · log(3))
```

Deeper = more accumulated error (√L). Wider = less per-layer error (exponential in w). The 3B threshold is where width dominates depth-induced error.

**Key insight for Model v2:** Our model needs to be ≥3B parameter-equivalent for ternary weights to match full precision. For smaller validation experiments, we should expect and accept some quality gap.

---

## 7. Beyond Ternary — Is It Optimal?

### Comparison

| Scheme | Bits/Weight | Key Property |
|--------|-------------|-------------|
| Binary {-1,+1} | 1.0 | No zero = can't prune = forced full connectivity |
| **Ternary {-1,0,+1}** | **1.58** | **Zero enables adaptive sparsity + routing** |
| Quinary {-2,-1,0,+1,+2} | 2.32 | Needs multiplication for ±2, marginal gains |
| INT4 | 4.0 | Standard quantization, excellent quality |

### Why Ternary Is Pareto-Optimal

Ternary occupies the Pareto frontier across:
- **Accuracy:** Sufficient for large-scale tasks
- **Efficiency:** Eliminates ALL multiplication (the big hardware win)
- **Simplicity:** Three levels = trivial to implement
- **Sparsity:** Zero enables dynamic pruning

**The zero value is the key differentiator from binary.** It enables:
- Adaptive sparsity (learned pruning)
- Selective routing (skip connections)
- Hierarchical representation

### Learned Quantization Levels

Can learn optimal levels {α₁, α₂, ..., αₖ} instead of fixed {-1,0,+1}. Adds flexibility but:
- More meta-parameters
- Less interpretable
- Harder to accelerate in hardware
- Marginal quality gains

**Key insight for Model v2:** Ternary is the right choice. It eliminates multiplication (the primary efficiency goal) while the zero value provides the routing/sparsity mechanism we need.

---

## 8. Mixed-Precision Strategy

### Not All Layers Are Equal

Optimal precision varies by layer:
- **First layer** (input encoding): Benefits from higher precision
- **Middle layers** (feature extraction): Ternary works well
- **Last layer** (output precision): Benefits from higher precision
- **Router/gate networks**: Need investigation

### Practical Approach

```
Ternary:  Middle layers (bulk of parameters)
FP16:     First layer (input encoding) + last layer (output)
Binary:   Attention masks (already sparse/binary in nature)
```

**Key insight for Model v2:** We can use mixed precision strategically. Ternary for the bulk (95%+ of parameters), higher precision only where mathematically necessary.

---

## 9. The SSM + Ternary Question (CRITICAL)

### Theoretical Analysis

SSM dynamics with ternary-projected parameters:
```
A, B, C = ternary_project(input)    → output is BF16
h(t+1) = A·h(t) + B·u(t)           → state evolution in BF16
y(t) = C·h(t)                       → output in BF16
```

**Stability depends on eigenvalues of A.** A is BF16 (computed from ternary projection), so eigenvalue analysis is standard.

### Potential Issues

1. **Conditioning of A:** Ternary projections might produce ill-conditioned A matrices more often than continuous projections. Monitor condition number during training.

2. **Scale factor sensitivity:** Separate α_A, α_B, α_C for each SSM parameter matrix. These continuous parameters provide the fine-grained control that ternary weights can't.

3. **Initialization:** Use HiPPO initialization for the master weights before quantization. This places eigenvalues optimally.

### Recommended Training Recipe

```
Phase 1 (0-20% of training): FP32 warmup, no quantization
Phase 2 (20-40%): Soft quantization with temperature annealing
Phase 3 (40-100%): Full ternary quantization
```

Plus: EMA of quantized weights for stability. Spectral regularization to encourage well-behaved eigenvalue distributions. Monitor gradient norms and activation statistics at every checkpoint.

### Theoretical Verdict

**No fundamental barrier.** The SSM sees full-precision A, B, C matrices. Ternary constrains the projection, not the state dynamics. With proper scaling, initialization, and training recipe, this should converge.

**The risk is practical, not theoretical:** training instability, poorly conditioned matrices, or gradient noise. All solvable with engineering.

---

## Summary: What We Take Forward to Model v2

### Confirmed

1. **Ternary works** — not a hack, but mathematically grounded in configuration space density + error cancellation + manifold preservation
2. **No matmul needed** — add/subtract/skip captures the essential computation
3. **Full-precision activations** — carry information, weights route it
4. **Error grows O(√L)** — sublinear, manageable for 32+ layer networks
5. **3B+ threshold** — below this, expect quality gap; above, parity with FP16
6. **SSM compatibility** — no fundamental barrier, SSM receives BF16 parameters

### Must Validate Empirically

1. Ternary projection → well-conditioned A matrices (spectral analysis during training)
2. Training convergence speed (expect 2-3x more iterations than FP16)
3. Minimum model size for our target tasks
4. Optimal mixed-precision strategy (which layers need FP16?)

### Design Decisions

1. **Ternary weights** for all middle layers (95%+ of model)
2. **FP16** for first layer, last layer, and possibly router networks
3. **Learned scale factors** (α per layer) — the continuous fine-tuning knob
4. **Native training** — must train in ternary from start, no post-training conversion
5. **Gradual quantization** — FP32 warmup → soft quant → full ternary

---

*Sources: arXiv 2402.17764, 2504.12285, 2411.05882, 2303.01505, 2508.11020, 2106.08295, 2504.09629 + ICLR 2019 (STE theory), NeurIPS 2024, Nature Communications 2025*

*Last updated: February 2026*
