# 01 — Mathematical Foundations of Attention

## Why This Matters for Model v2

We need to understand what attention DOES mathematically so we can decide: do we need it at all? Can we achieve the same effect with something cheaper? What are its hard limits that we must work around?

---

## 1. Why Dot-Product Similarity?

### The Math

Q·K^T captures angular similarity between query and key vectors. When unit vectors, equals cos(theta).

**Variance scaling theorem:** When Q, K components are ~N(0,1), Var(Q·K^T) = d_k. Scaling by 1/sqrt(d_k) restores unit variance — prevents softmax saturation and gradient vanishing in high dimensions.

### Why Not Other Similarities?

| Function | Problem |
|----------|---------|
| Cosine | Equivalent to normalized dot-product, extra O(n·d) norm calculations |
| L2 distance | Sensitive to magnitude, doesn't bound outputs, complex gradients |
| Additive (Bahdanau) | O(n·d_a·(d_k+1)+d_a·d_q) complexity, tanh amplifies/suppresses gradients unpredictably |
| Dot-product | O(n·d), uniform gradient propagation — simplest AND best |

### Deeper: Optimal Transport Justification (2024-2025)

Scaled dot-product attention is the **exact solution** to a one-sided Entropic Optimal Transport (EOT) problem:

```
max_{alpha in Delta^n}  <alpha, s(Q,K)> + beta * H(alpha)
```

Where alpha = attention distribution, s(Q,K) = similarity scores, H(alpha) = Shannon entropy, beta = temperature.

**Solution:** alpha* = softmax(s(Q,K)/beta)

This proves attention isn't a heuristic — it's the solution to a principled optimization problem: maximize similarity while maintaining maximal entropy (diversity).

**Source:** arXiv 2508.08369

---

## 2. Why Softmax?

### Properties

- Guarantees sum(alpha_i) = 1 and alpha_i >= 0
- Output = convex combination of value vectors (can't extrapolate beyond convex hull)
- Implements Boltzmann distribution from statistical mechanics
- Maximizes entropy subject to moment constraints

### The Softmax Bottleneck (CRITICAL FOR US)

The low rank of the logit matrix limits representational capacity. As context grows:
- Attention weights collapse toward uniform (1/n) — "vanishing attention" problem
- Maximal attention probability decays with sequence length
- Hampers retrieval of salient tokens in long contexts

**This is a fundamental architectural weakness, not a fixable bug.**

### Alternatives That Work

| Alternative | Property | Trade-off |
|-------------|----------|-----------|
| Sparsemax | Exact zeros for irrelevant tokens | Better for many-label tasks |
| Sigmoid attention (2025) | No token competition, lower sample complexity | Matches softmax performance, eliminates bottleneck |
| Gumbel-softmax | Differentiable sampling | For discrete decisions |

**Key insight for Model v2:** Sigmoid attention eliminates the softmax bottleneck while matching performance. Token-independent scoring (no competition) may be more natural for our architecture.

---

## 3. What V·attention_weights Actually Computes

### Geometric Meaning

Attention finds a point **within the convex hull** of value vectors:

```
output = sum_i alpha_i * v_i    (convex combination, bounded interpolation)
```

**Cannot extrapolate** — only interpolate between existing representations. This is a safety property (stability) but also a limitation (can't express novel combinations beyond the hull).

### The Fixed-Dimensional Bottleneck

Traditional sequence models (RNNs, SSMs) compress all history into fixed-dimensional state — an information bottleneck. Attention creates **multiple pathways** for information flow, bypassing this bottleneck.

**But:** This comes at O(n^2) cost. The question is whether we can get multiple pathways without quadratic cost.

### Can Simpler Operations Work?

"Cost-Effective Attention" research (2024-2025): models with 25-50% fewer parameters show negligible performance cost. "Super Attention" outperforms SDPA by up to 10% while being faster.

**However:** Replacing attention with pure MLP only works for small, fixed-length contexts. Attention's unique value is **variable-length** sequence modeling.

---

## 4. Multi-Head Attention — Why Multiple Heads?

### What Heads Learn

1. **Positional heads:** Attend to neighbors (local patterns)
2. **Syntactic heads:** Subject-verb agreement, grammatical structure
3. **Semantic heads:** Meaning relationships, pronoun resolution
4. **Rare-word heads:** Focus on low-frequency tokens

### Redundancy Is Massive

- **40% of heads** can be pruned with only ~10% performance cost
- First layer uses all heads; later layers rely on 1-2 heads
- 80% self-attention pruning is tolerable
- Encoder-decoder attention is more sensitive (60%+ pruning = catastrophic)

### Game Theory Insight (Feb 2026)

Heads form **emergent coalitions** — coordination within coalitions, reduced coupling across boundaries. Suggests we should design heterogeneous head architecture rather than uniform heads everywhere.

**Key insight for Model v2:** We don't need uniform multi-head attention. Dense first layer + sparse later layers mirrors actual usage. Can we build this heterogeneity into the architecture from the start?

---

## 5. What Attention Wastes

### The Numbers

Analysis across Qwen 2.5, Llama 3.1, Gemma 3 (4B-72B, 16K-128K tokens):

- **Simple sentences:** 90-95% of attention weights near-zero (wasted)
- **Complex sentences:** 70-80% near-zero
- **Average:** ~70-95% of O(n^2) computation is useless

### Patterns in the Waste

- **Vertical-slash patterns** with RoPE (structured, not random)
- **Block-wise organization** — attention clusters into spatial blocks
- Longer sequences tolerate higher sparsity
- No universal optimal sparsity — varies by task/scale/phase

### The Sparsity Plateau

Performance improves rapidly as sparsity decreases from 1% to ~25%, then **plateaus**. This means **25% of attention captures most necessary information**.

**Key insight for Model v2:** If only 5-25% of attention computations matter, we're wasting 75-95% of compute. A mechanism that directly computes only the relevant interactions would be dramatically more efficient.

---

## 6. Fundamental Limits of Attention

### What Attention CANNOT Do (Proven)

| Limitation | Proof |
|------------|-------|
| **Compositional reasoning** | Fails on NC1-complete tasks (Boolean formula evaluation) for realistic parameter budgets |
| **Length generalization** | Trained on 5-digit addition, fails on 6-digit, regardless of model size |
| **Formal languages** | Hard attention: only AC0 languages. Averaging hard attention: only TC0 |
| **Counting** | Cannot reliably count or do multi-step arithmetic |

### The Complexity Hierarchy

```
AC0 ⊂ TC0 ⊂ NC1 ⊂ ... ⊂ P

UHAT (hard attention):     recognizes AC0 only
AHAT (averaging hard):     recognizes TC0
Full Transformer:          bounded by TC0 (with log precision)
SSMs/RNNs:                 can simulate any finite automaton
Turing machines:           P and beyond
```

**Key insight for Model v2:** Transformers are computationally LIMITED to TC0. RNNs/SSMs can simulate arbitrary finite automata. A state-based model has HIGHER theoretical computational power than attention for sequential reasoning.

### The O(n^2) Lower Bound

**Theorem (SETH-based):** Unless the Strong Exponential Time Hypothesis is false, computing self-attention requires Omega(n^2) time. This holds for approximate computation, windowed attention, and various similarity functions.

**This is not a bug to fix — it's a mathematical inevitability for this mechanism.** We need a different mechanism entirely.

---

## 7. Alternatives Preserving Useful Properties

### Linear Attention

Replace softmax(QK^T)V with kernel approximation phi(Q)(phi(K)^T V). Reduces O(n^2 d) to O(n d^2).

**What's lost (critical):**
- Compositional reasoning power
- Multi-step function composition
- High-frequency information (acts as low-pass filter)

**What's preserved:**
- All-to-all token interactions (structural connectivity)
- Differentiability
- Permutation equivariance

**Verdict:** Linear attention alone is insufficient. Hybrid (1 full attention per 3-6 linear) restores quality.

### Sparse Attention

Only compute attention for selected token subsets.

- Larger sparse models outperform smaller dense ones at equivalent cost
- 25% attention captures most necessary information (plateau finding)
- Context-dependent patterns needed (not fixed sparsity)

### The Minimum Viable Mechanism

From all evidence, the minimum that works:
- **Local attention** as default (sliding window, O(n))
- **Strategic global attention** for identified long-range dependencies
- **~25% of full attention budget** captures essential information
- **Heterogeneous per-layer:** dense early, sparse later

---

## Summary: What We Take Forward to Model v2

### Essential Properties to Preserve

1. **Variable-length sequence modeling** — attention's unique advantage over MLP
2. **Multiple information pathways** — not a single bottleneck state
3. **Learnable token-token interactions** — content-based, not position-based
4. **Bounded output** — convex combination ensures stability

### What We Can Eliminate

1. **O(n^2) full pairwise computation** — 75-95% is wasted
2. **Softmax normalization** — sigmoid works equally well without bottleneck
3. **Uniform multi-head** — heterogeneous is better (dense early, sparse late)
4. **KV cache** — if using state-based model, this disappears entirely

### Open Questions for Architecture Design

1. Can state evolution (SSM-like) provide the "multiple pathways" benefit without O(n^2)?
2. If state-based models have higher computational class (arbitrary FA vs TC0), do we even need attention?
3. Is 25% attention budget the minimum, or can we go lower with better routing?
4. Can sigmoid scoring replace softmax in our architecture for simpler gradient flow?

### Critical Takeaway

**Attention's mathematical essence is: adaptive, content-based information routing.** The specific mechanism (dot-product + softmax + value aggregation) is just one implementation. We need the ROUTING, not the specific math. If our state evolution mechanism can do content-based routing with O(1) memory, we may not need attention at all.

---

*Research compiled from: arXiv 2601.03329, 2508.08369, 2508.17821, 1602.02068, 2502.00281, 1905.09418, 2602.00861, 2410.11842, 2504.17768, 2502.18137, 2209.04881, 2510.05364, 2411.09604 + Nature Scientific Reports 2025*

*Last updated: February 2026*
