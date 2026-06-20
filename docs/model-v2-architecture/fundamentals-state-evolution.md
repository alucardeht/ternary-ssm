# 02 — Mathematical Foundations of State Evolution (SSM)

## Why This Matters for Model v2

State Space Models compress ALL sequence history into a fixed-size vector. If we can make this work with ternary weights, we eliminate both the KV cache (O(n) memory) AND matmul (the two biggest costs in Transformers). But we need to understand WHEN and WHY this compression works, and when it breaks.

---

## 1. The Fixed-State Compression Problem

### Core Equation

```
h(t) = A·h(t-1) + B·x(t)
y(t) = C·h(t) + D·x(t)
```

State h(t) ∈ R^N compresses entire history {x(0), ..., x(t)} into N real numbers.

### What Information Is Preserved?

The state acts as a **lossy compression** following the **Principle of Predictive Sufficiency**: an ideal hidden state is a minimal sufficient statistic of the past for predicting the future.

```
P(future | past, h(t)) = P(future | h(t))
```

Sufficient = contains all info needed for future predictions. Minimal = no redundant info stored. Lossy = irrelevant info discarded.

### What Information Is Lost?

**Exponential forgetting:** For eigenvalues |λᵢ| < 1:
```
influence(t-k) ∝ λᵢ^k
```

Information from k steps ago decays exponentially. This creates **recency bias** — distant tokens are "forgotten rapidly while closer tokens dominate."

**Retrieval bottleneck:** Fixed-size state cannot do random access to arbitrary past tokens. This is the fundamental SSM limitation vs attention.

---

## 2. HiPPO Theory — Why Polynomial Projection Works

### The Core Idea

HiPPO (High-order Polynomial Projection Operators) solves: given continuous function f(t), maintain coefficient vector c(t) ∈ R^N that **optimally captures history** from time 0 to t.

```
ċ(t) = A(t)·c(t) + B(t)·f(t)
```

Projects history onto orthogonal polynomial basis (Legendre polynomials).

### Why Polynomials?

1. **Optimal L² approximation** for continuous functions
2. **Closed-form transition matrices** A and B
3. **Timescale robustness** — Scaled Legendre weights all history uniformly

### Different Measures

| Measure | Description | Property |
|---------|-------------|----------|
| Translated Legendre (LegT) | Fixed-length sliding window | Bounded memory of recent ~θ time units |
| Scaled Legendre (LegS) | Uniform weighting of all history | Timescale-agnostic, sees everything equally |

### Connection to Existing Architectures

- **LSTM/GRU**: Low-order (N=1) HiPPO approximations with input-dependent discretization
- **Legendre Memory Unit (LMU)**: Exactly equivalent to HiPPO-LegT

**Recent validation (Dec 2024):** HiPPO-LegS ODE proved well-posed despite singularity, with convergence guarantees for discretization schemes.

**Key insight for Model v2:** HiPPO tells us the OPTIMAL way to compress history. Our state evolution should be informed by this theory, even if we don't use the exact HiPPO matrices.

---

## 3. Stability Theory

### Eigenvalue Conditions

**Discrete-time systems** — all eigenvalues of A must lie within unit circle:

| Eigenvalue Position | Behavior |
|---------------------|----------|
| \|λ\| < 1 | Stable: information fades gracefully |
| \|λ\| = 1 | Marginally stable: information preserved indefinitely |
| \|λ\| > 1 | UNSTABLE: unbounded growth, numerical explosion |
| Complex λ | Oscillatory behavior |

For language modeling: need **real, positive eigenvalues bounded by 1**.

### Memory vs Stability Trade-off

```
memory_length ∝ 1 / (1 - |λ|)
```

Eigenvalues closer to 1 = longer memory BUT closer to instability boundary.

### SSM-Attention Duality

Both compute weighted summation of historical tokens:
- **Attention weights**: content-based (token-to-token similarity)
- **SSM weights**: position-based (exponential decay)

Signal processing view:
- **Attention**: FIR filters (separate parameters per position)
- **SSMs**: IIR filters (feedback + memory, more parameter-efficient)

**Key insight for Model v2:** SSMs and attention are TWO VIEWS of the same underlying operation. We can potentially get the best of both by designing our state evolution to be content-sensitive (like Mamba's selectivity).

---

## 4. Why Diagonal State Matrices?

### Computational Reality

- **Full-rank A**: O(N²) operations per timestep
- **Diagonal A**: O(N) operations per timestep

For diagonal A = diag(λ₁, ..., λₙ):
- Matrix multiply → element-wise multiply
- Matrix exponential → element-wise exponential
- S4D was **30x faster using 400x less memory** than previous SSMs

### DPLR: Diagonal Plus Low-Rank

S4 decomposes A = Λ - p·qᵀ (diagonal + rank-1 correction). This enables fast computation in frequency domain.

### What Expressiveness Is Lost?

**Mamba-2 finding:** Scalar-times-identity A (all diagonal elements same) is equivalent to **masked self-attention with 1-semiseparable causal mask**.

```
Same transformation, two algorithms:
- O(T) recurrence (SSM view)
- O(T²) attention (attention view)
```

**Empirical verdict:** "Authors haven't found evidence that restricted expressivity hurts" — diagonal is sufficient in practice.

### Normal Forms Connection

Diagonal form is the simplest normal form in dynamical systems. PTD (Perturbed Transformation Diagonalization) can diagonalize non-normal HiPPO matrices, yielding models more robust under perturbations.

**Key insight for Model v2:** We should use diagonal state matrices. The O(N) per-token cost is essential, and expressiveness isn't sacrificed.

---

## 5. The Selective Mechanism (Mamba's Key Innovation)

### Static vs Selective

**Static (S4):** A, B are fixed constants.
```
h(t) = A·h(t-1) + B·x(t)    (same A, B for all inputs)
```

**Selective (Mamba):** A, B, Δ are functions of input.
```
h(t) = A(x(t))·h(t-1) + B(x(t))·x(t)    (different per token)
```

Parameterization:
```
B = Linear(x)
C = Linear(x)
Δ = Broadcast(Linear(x))
```

### Why Selectivity Works

**Content-based filtering:** Each token gets customized filtering, not one-size-fits-all.

**Δ as generalized gate:**
- Smaller Δ → finer temporal granularity (attend to details)
- Larger Δ → coarser steps (skip/downweight information)

**Connection to LSTM:**
LSTM: `c(t+1) = α(t)·c(t) + β(t)·f(t)` where α, β depend on input.
This is **N=1 SSM with input-dependent discretization** — HiPPO view of LSTM.

Key difference: LSTM = heuristic design. Mamba = principled derivation from discretized continuous systems.

### Information-Theoretic View

Selectivity encourages **maximally compressing history without losing predictive power** — discarding noise and spurious patterns dynamically based on content.

**Key insight for Model v2:** Selectivity is ESSENTIAL. Static state transitions (S4) can't do content-based filtering. Our architecture MUST have input-dependent state dynamics.

---

## 6. What SSMs Cannot Do (Fundamental Limits)

### The Copy Task

Given [a₁, a₂, ..., aₙ, SEP], output [a₁, a₂, ..., aₙ].

**SSMs fail** because:
```
Required capacity: L·log₂(V) bits    (L = length, V = vocab size)
Available capacity: N·log₂(precision) bits    (N = state dimension)
```

For fixed N, copying arbitrary sequences is **impossible** when L exceeds capacity.

**Trade-off:** If state grows linearly with sequence length, SSMs can copy — but lose their efficiency advantage.

### Multi-Hop Reasoning

**Associative recall** requires random access to memory:
- **Attention:** O(1) random access to any token
- **SSMs:** Sequential access only, exponential decay

**Critical finding (2025):** Mamba relies on **short convolution** for key-value association — fails without it. This suggests convolution is a necessary component, not optional.

### Formal Language Theory

| Class | SSMs | Transformers |
|-------|------|-------------|
| Regular languages | YES (all) | YES |
| Star-free regular | Perfect (exact solutions) | YES |
| Context-free | Partial (without stack simulation) | Partial |
| NC1-complete | Unknown | NO (proven) |

**Key finding:** SSMs and Transformers have OVERLAPPING BUT DISTINCT strengths.

**SSMs excel:** Star-free state tracking, sequential pattern recognition.
**Transformers excel:** In-context learning, few-shot prompting, long-context retrieval.

### The Fundamental Limitation

SSMs are **inherently local-biased**: they lose long-term memory exponentially and struggle to retrieve from distant context. This is architectural, not a training problem.

**Key insight for Model v2:** We CANNOT rely on pure state evolution for long-range retrieval. We either need a hybrid (sparse attention for retrieval) or a fundamentally new retrieval mechanism.

---

## 7. Information-Theoretic Capacity

### Shannon Capacity of State

For N-dimensional state with k-bit precision:
```
Capacity ≤ N·k bits
```

With eigenvalue decay:
```
Effective capacity ≈ Σᵢ log₂(1/(1-λᵢ))
```

States with λᵢ close to 1 contribute most capacity.

### Compression via Hankel Singular Values

Hankel singular values σᵢ measure importance of each state dimension. Balanced truncation: remove dimensions with smallest σᵢ.

**Recent finding (2024-2025):** Dominant Hankel singular values are rank-preserving during training. Truncation yields **1.14-1.4x speedups with minimal accuracy loss.**

### Context vs State Dimension

- Explicit tokens: ~15 bits each
- Latent hidden states: ~40,960 bits each
- Bandwidth difference: ~2,700x

**Implication:** The state has enormous bandwidth — the bottleneck is not raw capacity but **selective retrieval** from the compressed representation.

### Hybrid Approaches in Production

- **Bamba (IBM):** SSM + Transformer, trained on 4K but handles 32K+
- **Samba:** Selective compression + sliding window attention

**Key insight for Model v2:** Pure SSM has sufficient STORAGE capacity but insufficient RETRIEVAL capability. Hybrids solve this by using attention only for retrieval-critical operations.

---

## 8. Continuous-Time vs Discrete-Time

### Continuous SSM
```
dx/dt = Ã·x + B̃·u
y = C̃·x + D̃·u
```

### Discretization Methods

**Zero-Order Hold (ZOH) — Mamba's choice:**
```
A = e^(Δ·Ã)
B = (Ã)⁻¹·(A - I)·Δ·B̃
```
- Exact match for staircase inputs
- Requires matrix exponential (O(N³) general, O(N) diagonal)

**Bilinear Transform (Tustin):**
```
A = (I - Δ/2·Ã)⁻¹·(I + Δ/2·Ã)
B = (I - Δ/2·Ã)⁻¹·(Δ·B̃)
```
- Best frequency-domain match
- Maps left half-plane → unit circle (preserves stability)
- Only matrix multiplication (no exponential)

**For diagonal matrices:** Both become element-wise operations → O(N).

### Learnable Δ

Training Δ as learnable parameter lets the model adaptively select temporal granularity per sequence pattern. This is the source of Mamba's selectivity.

**Key insight for Model v2:** The continuous-time perspective gives us HiPPO initialization (proven optimal), stability guarantees through discretization, and interpretability through differential equations.

---

## 9. State Evolution Alternatives

### Nonlinear State Evolution

**Polynomial:**
```
h(t) = A·h(t-1) + B·x(t) + W·[h(t-1) ⊗ h(t-1)]
```

Pros: captures nonlinear dynamics. Cons: O(N²) cost, harder stability analysis, harder training.

### Neural ODEs
```
dh/dt = f_θ(h(t), x(t), t)
```

Where f_θ is a neural network. Highly expressive but:
- Stiffness issues (rapid variations need tiny time steps)
- Harder to guarantee stability
- Computational cost unpredictable

**Recent advances:** ControlSynth Neural ODEs (NeurIPS 2024) — precise local asymptotic stability without global Lipschitz assumptions.

### Practical Verdict

Most production SSMs (S4, Mamba) remain **linear** because:
1. Stability is well-understood and guaranteed
2. Computational efficiency preserved
3. Expressiveness achieved through depth + selectivity
4. Training dynamics are stable

Nonlinear extensions add expressiveness but at significant cost to stability and efficiency.

**Key insight for Model v2:** We should start with linear state evolution (proven, stable, efficient) and add nonlinearity through selectivity and depth, not through nonlinear state dynamics.

---

## Summary: What We Take Forward to Model v2

### Essential Properties to Preserve

1. **Fixed-size state** — O(1) memory per token, unlimited context
2. **Diagonal state matrix** — O(N) per timestep, sufficient expressiveness
3. **Selectivity** — input-dependent state dynamics (Mamba's key insight)
4. **HiPPO-informed initialization** — optimal polynomial projection for long-range memory
5. **Stable eigenvalues** — all within unit circle, memory controlled by distance from boundary

### What We Must Solve

1. **Retrieval bottleneck** — pure state can't do random access to distant context
2. **Copy task** — fundamental limit of fixed-size state
3. **Multi-hop reasoning** — needs attention-like mechanism or convolution

### Design Decisions for Architecture

1. **Linear state evolution** — proven, stable, efficient. Nonlinearity via selectivity.
2. **Diagonal A** — O(N) cost, no expressiveness loss in practice.
3. **Learned Δ** — input-dependent discretization step (Mamba's approach).
4. **Hybrid retrieval** — sparse attention or convolution for the ~12.5% of operations that need retrieval (following Mamba hybrid ratio).
5. **Continuous-time initialization** — HiPPO for eigenvalue placement, ZOH discretization.

### Open Questions

1. Can ternary weights produce well-conditioned A matrices? (Addressed in 03-quantization)
2. What's the minimum state dimension for our target tasks?
3. Can we design a retrieval mechanism cheaper than attention?
4. Is the 1:8 attention:SSM ratio optimal, or can we go lower?

---

*Sources: arXiv 2503.18970, 2008.07669, 2312.00752, 2412.08595, 2405.17394, 2412.19350, 2510.02823, 2510.22951, 2409.03231, 2406.07522 + NeurIPS 2024, ICLR 2025 proceedings*

*Last updated: February 2026*
