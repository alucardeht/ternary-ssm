# 04 — Mathematical Foundations of Dynamic Routing & Compute Allocation

## Why This Matters for Model v2

Not all tokens need all layers. "the" doesn't need 32 layers of processing. If we can route easy tokens through fewer layers, we save 40-70% of compute per forward pass. But the routing mechanism itself must be cheap, trainable with discrete decisions, and compatible with state evolution.

---

## 1. Why Tokens Don't Need All Layers

### The Math

In Pre-LN architectures, layer contribution decreases toward zero with depth:
```
output = x + f_l(x)
When ||f_l(x)|| ≈ 0, output ≈ x → layer can be skipped
```

### How Often This Happens

- **Deep models:** Later layers frequently contribute near-zero for easy tokens
- **Function words** (the, and, of): Almost always skippable in middle/late layers
- **Content words** (nouns, verbs): Varies by semantic complexity
- **Middle layers** (30-60% of depth): Most frequently have near-zero contributions for easy tokens

### Empirical Evidence

Critical layer analysis on Qwen shows: while specific layers (e.g., layer 23) cause significant degradation when ablated, many others show minimal impact. Adaptive skip connection research shows networks naturally learn near-zero skip weights in later layers.

### Residual Stream Interpretation

With residual connections, large activation spikes persist through the identity path. When a layer adds nothing useful, the residual just passes through — the network already does implicit skipping.

---

## 2. Mixture of Depths (MoD) — The Core Mechanism

### Router Architecture

For layer l and input X_l ∈ R^(S×d):

```
R_l = X_l · w_l ∈ R^S            (router scores: single linear projection)
TopK tokens → process through layer
Rest → skip via residual connection
```

### Token Processing

```
For token i at layer l:
  If R_l[i] in TopK(R_l):
    h_{l+1}[i] = h_l[i] + Block_l(h_l[i])    # Process
  Else:
    h_{l+1}[i] = h_l[i]                        # Skip
```

### Training: The Auxiliary Loss

```
L = L_nll + λ · L_aux

L_aux = (1/E) Σ (c_e/S) · m_e    (load balancing)
```

Without balancing: collapse problem (all tokens route to same layers), underutilization, specialization failure.

### Inference Problem

Top-k needs all token scores simultaneously — impossible in autoregressive generation. Solutions:

1. **Auxiliary classifier:** Train binary decision per token: route if p > 0.5
2. **Threshold-based:** Learn threshold τ_k from training statistics
3. **Learned threshold:** τ_k = w_threshold · std(R_l) + mean(R_l)

---

## 3. Training Routers with Discrete Decisions

### The Core Problem

Route decision is binary (process/skip) but training needs gradients. Top-k has zero gradient almost everywhere:
```
∂TopK(x)/∂x_i = 1 if x_i in top-k, else 0
```

### Solution 1: Gumbel-Softmax

Continuous relaxation of discrete categorical:
```
y_i = exp((log(π_i) + g_i)/τ) / Σ_j exp((log(π_j) + g_j)/τ)

g_i = -log(-log(u_i)), u_i ~ Uniform(0,1)    (Gumbel noise)
```

Temperature annealing: τ starts at 1.0 (smooth), anneals to 0.1 (nearly discrete).

**Property:** Fully differentiable. Gradient:
```
∂y_i/∂log(π_j) = (1/τ) · [1[i=j]·y_i - y_i·y_j]
```

### Solution 2: Straight-Through Estimator (STE)

```
Forward:  y_hard = one_hot(argmax(...))    (discrete)
Backward: gradients as if y_soft was used   (continuous)
```

**Why it works:** STE provides a descent direction (positively correlated with true gradient), even though it's biased. Low variance makes it practical.

**Recent improvement (2024):** Decoupled temperatures — small τ for forward (discrete), larger τ for backward (smooth gradients).

### Solution 3: Policy Gradient (REINFORCE)

```
∇_θ J(θ) = E[∇_θ log π_θ(a|s) · (R - b(s))]
```

Unbiased but very high variance. Needs baseline subtraction, advantage estimation, many samples.

### Solution 4: Optimal Transport (Sinkhorn)

Routing as assignment problem:
```
min_P <P, C> - ε·H(P)    s.t. row/column sum constraints
```

Solved iteratively by Sinkhorn algorithm. Exponential convergence. Fully differentiable.

**Connection to attention:** Softmax attention IS gradient descent on the entropy-regularized OT dual.

### Comparison

| Method | Bias | Variance | Cost | Best For |
|--------|------|----------|------|----------|
| Gumbel-Softmax | Yes | Low | Medium | Dense routing |
| STE | Yes | Medium | **Low** | Sparse routing, inference speed |
| REINFORCE | No | Very High | High | Non-differentiable rewards |
| Optimal Transport | Low | Low | High | Theoretical guarantees |

**Practical consensus (2025):** STE + loss-free load balancing for production. OT for research.

### Load Balancing Without Auxiliary Loss

**Loss-Free Balancing (2024-2025):** Apply expert-wise bias to routing scores:
```
R'_l[i] = R_l[i] + β_e
β_e(t+1) = β_e(t) - η · (load_e(t) - target_load)
```

No gradient interference. Shifts scores without touching primary loss.

**Key finding:** Micro-batch balancing HURTS performance. Global-batch balancing is correct.

---

## 4. Early Exit — When Is "Good Enough"?

### Entropy-Based Criterion

```
H(p) = -Σ p_i · log(p_i)
H_normalized = H(p) / log(C)

Exit when H_normalized < τ
```

Thresholds vary by task:
- Simple classification: τ ≈ 0.5
- Language generation: τ ≈ 0.7-0.8
- Reasoning: τ ≈ 0.6-0.672

### Cross-Layer Consistency (PABEE)

```
patience_score = count of consecutive layers with same argmax prediction
Exit when patience_score = patience_threshold
```

No calibrated confidence needed — just prediction stability.

### Fixed-Point Interpretation (ICLR 2025)

Each transformer layer is one iteration of refinement:
```
h_{l+1} = h_l + f_l(h_l)
```

If f_l(h_l) ≈ 0, we've reached a fixed point. Under Lipschitz continuity:
```
||h_∞ - h_l|| < ε/(1-L_lip)
```

**Key insight for Model v2:** Early exit is mathematically justified as fixed-point detection. We stop when the representation stops changing.

### Training Multi-Exit Networks

```
L = Σ w_i · L_i    where w_i = α^(L-l_i)
```

Higher weight to later exits. Early exits must learn but shouldn't dominate.

---

## 5. Compute Allocation Theory

### Kolmogorov Complexity Connection

Optimal compute per token is proportional to its conditional Kolmogorov complexity:
```
c_i ∝ K(y_i | x_{<i})
```

More complex predictions need more computation.

### Token Difficulty Distribution

Empirical analysis of GPT-family models:
```
20% tokens: Very easy (K ≈ 1-2 bits) — function words, deterministic
60% tokens: Moderate (K ≈ 3-6 bits) — typical content
15% tokens: Hard (K ≈ 7-10 bits) — reasoning, rare words
5%  tokens: Very hard (K > 10 bits) — creative, multi-hop
```

### Optimal Allocation Formula

Given compute budget B and N tokens:
```
c_i = B · K(y_i|x_{<i}) / Σ_j K(y_j|x_{<j})
```

Practical proxy:
```
c_i = B · softmax(difficulty_score_i / temperature)
```

### Resource-Constrained Optimality (KKT)

```
max Σ Accuracy_i(c_i)  s.t. Σ c_i ≤ B
```

At optimum: **marginal accuracy gain per unit compute is equal across all tokens.** Easy tokens hit diminishing returns fast; hard tokens benefit more from extra compute.

**Key insight for Model v2:** We should allocate layers proportional to token difficulty. ~20% of tokens need only ~25% of layers. ~5% need all layers.

---

## 6. Routing + State Evolution Interaction

### The Core Question

When a token skips a layer in an SSM, the state doesn't get updated. Does this break state coherence?

### Mamba Already Does "Soft Skipping"

The selective mechanism with Δ ≈ 0:
```
When Δ_i → 0: h_i^(l) → h_i^(l-1)    (state unchanged)
```

Explicit layer skipping is just the hard version of what Mamba already does. The model can already learn to not update state for certain tokens.

### Token Dynamics in Deep SSMs

**Dichotomy theorem:** In continuous-time limit, only two scenarios:
1. All tokens converge to zero: lim ||x_i(t)|| = 0
2. All tokens diverge to infinity: lim ||x_i(t)|| = ∞

But they diverge at **different rates**:
```
||x_i(t)|| ≈ C_i · exp(λ_max · t)
```

Tokens with smaller C_i contribute less and are safer to skip.

### When Is Skipping Safe?

1. **Low-impact tokens:** ||Δ_i · B_i · u_i|| < ε (state update is negligible)
2. **State saturation:** ||h_i^(l) - h_i^(l-1)|| < ε (state has converged)
3. **Redundant updates:** Similar to another token being processed
4. **Selective forgetting:** Update primarily resets state, not adds info

### Hybrid SSM-Attention Models

Production architectures (Zamba2, Hymba):
```
[SSM, SSM, Attn, SSM, SSM, Attn, ...]
```

SSM for local patterns (cheap, O(1)). Attention for long-range retrieval (expensive, but sparse). 30-50% speedup over pure Transformers.

**Key insight for Model v2:** Dynamic routing in SSM is natural — it's an extension of the selective mechanism. Safe when state updates are small.

---

## 7. Minimum Router Precision

### Can Routers Use Ternary Weights?

The router is a single linear projection: w^T · h_i. With ternary w:
```
w ∈ {-1, 0, +1}^d
Score = Σ_j w[j] · h_i[j]    (only add/subtract/skip — no multiplication!)
```

### Information-Theoretic Minimum

Router makes binary decision (1 bit) from d-dimensional input. Minimum parameters:
```
d ternary values + 1 threshold ≈ 1.58d + 8 bits
```

For d=512: ~100 bytes per layer vs ~2KB full precision. **20x compression.**

### Evidence That Low Precision Works

- 4-bit router weights: Minimal degradation
- 2-bit router weights: <1% accuracy loss
- Ternary: Should be between 1-bit and 2-bit performance

**Why low precision works for routing:**
1. Routing is a coarse decision (easy/hard), not precise computation
2. Errors in individual decisions average out over sequence
3. Learned thresholds compensate for quantization noise

### Practical Finding

Even 1-dimensional projections (d ternary values + 1 threshold) achieve 80-90% of full-precision router performance.

**Key insight for Model v2:** Ternary routers work. The routing decision is inherently low-information (1 bit output), so low-precision inputs are sufficient.

---

## 8. Gradient Flow Through Sparse Paths

### Variable Depth Problem

Different tokens experience different effective depths. This creates:
```
∂L/∂W_l = Σ_i 1[token i routed through l] · ∂L_i/∂W_l
```

If 50% of tokens skip layer l, gradient is halved.

### Why It Doesn't Break Training

**Residual connections provide gradient highway:**
```
∂L/∂h_l = ∂L/∂h_{l+1} · (I + ∂f_l/∂h_l)
         = ∂L/∂h_{l+1} + ∂L/∂h_{l+1} · ∂f_l/∂h_l
```

The identity term guarantees: ||∂L/∂h_0|| ≥ ||∂L/∂h_L||

**Gradient cannot vanish entirely, regardless of routing.**

### Ensemble Interpretation

A residual network with L layers is equivalent to an ensemble of 2^L shallow networks. Variable routing just changes the ensemble weighting.

### Training Stability Conditions

1. **Gradient floor:** Residual connections with proper initialization (f_l initialized with E[||f_l||²] = O(1/L))
2. **Bounded path variance:** Var(effective_depth) should be bounded
3. **Load balancing:** All layers must receive sufficient training signal
4. **Lipschitz smoothness:** Layer normalization + gradient clipping

### Training Phases (Empirical)

1. **Phase 1 (0-10%):** Random routing, all tokens similar paths
2. **Phase 2 (10-50%):** Specialization begins, routing variance increases — **most unstable period**
3. **Phase 3 (50-100%):** Routing stabilizes, performance plateaus

**Best practice:** Warmup without routing for first 10% of training, then gradually increase routing.

---

## Summary: What We Take Forward to Model v2

### Confirmed Design Decisions

1. **MoD-style routing** with single linear projection per layer — simplest effective router
2. **Ternary router weights** — 20x smaller, no multiplication, sufficient precision
3. **STE for training** — low cost, adequate gradient, proven at scale
4. **Loss-free load balancing** — no auxiliary gradient interference
5. **Entropy + consistency exit criteria** — robust, no calibration needed
6. **Residual connections everywhere** — gradient highway, enables safe routing

### Compute Savings Potential

```
20% tokens × ~25% layers = 5% compute
60% tokens × ~50% layers = 30% compute
15% tokens × ~75% layers = 11% compute
5%  tokens × 100% layers = 5% compute
Total: ~51% of original compute (49% savings)
```

### Compatibility with SSM

- Explicit skipping is natural extension of Mamba's selective Δ mechanism
- Safe when state updates are small (which the router learns to detect)
- No additional architectural changes needed

### Open Questions

1. Optimal skip probability schedule during training
2. Best ratio of SSM:attention layers for hybrid retrieval
3. Whether ternary router + ternary model weights interact badly
4. Minimum state dimension that tolerates aggressive skipping

---

*Sources: arXiv 2312.00752, 2409.19606, 2408.15664, 2510.11472, 2501.06802, 2504.15895, 2507.13569, 2502.11880, 2410.19931, 2410.03292 + ICLR 2025, NeurIPS 2024 proceedings*

*Last updated: February 2026*
