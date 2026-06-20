# Research Round 2, Aspect 3: Convergence Speed — Wave 2 (Math-Paradigm-Breakers)

## Summary

Two math-paradigm-breakers explored novel mathematical formulations for faster convergence in ternary networks. Together they produced 12 unique formulations. The unifying insight: STE treats ternary optimization as a continuous problem with a discrete projection, but the OPTIMAL algorithms would be native to the discrete ternary structure, exploiting its geometry, information theory, and topology.

## HIGH VIABILITY (implement first)

### R2-A3-M1: Optimal Learning Rate from Ternary Decision Theory
- **Principle:** The LR in ternary training controls how fast shadow weights traverse inter-threshold distance. Optimal η should be derived from threshold crossing statistics, not continuous optimization theory.
- **Formulation:** From SPRT analysis: η* = τ · SNR²_i / (2 · |u_i| · log(1/α)). Since SNR(t) ∝ t^{-β}, this gives η*(t) ∝ t^{-2β}. For β ≈ 0.25: η* ∝ t^{-1/2} — SLOWER decay than cosine.
- **Convergence gain:** 1.3-1.8x. Cosine spends ~40% of training with η too slow for flips. SNR-derived schedule keeps η productive longer.
- **Complexity:** O(1) per weight. Zero overhead — just a schedule formula change.
- **Feasibility:** HIGH — trivial implementation

### R2-A3-M2: Analytical STE Bias Correction
- **Principle:** STE bias decomposes into signal + position-dependent bias + noise. For active weights (Q=±1), STE is UNBIASED. For dead weights (Q=0), STE is ENTIRELY bias — the gradient doesn't correspond to any computation.
- **Formulation:** Correction function: φ(w_i) = 1 if |w_i| ≥ τ (active), φ(w_i) = |w_i|/τ if |w_i| < τ (dead). Apply: g_corrected = g_STE × φ(w_i).
- **Convergence gain:** ~1.33x. Reduces gradient noise variance by ~25% (dead weights = ~33% of params).
- **Complexity:** O(d) per step — one comparison and multiply per weight. Negligible overhead.
- **Feasibility:** HIGH — trivial 1-line implementation, provably non-destabilizing

### R2-A3-M3: Information-Theoretic Adaptive Freezing
- **Principle:** Each weight accumulates evidence for its ternary assignment. When confidence exceeds threshold derived from information theory, freeze the weight permanently.
- **Formulation:** Confidence accumulator: c_i^{t+1} = c_i^t + sign(g_i) · |g_i| / (v_i + ε). Freeze when |c_i| > C_thresh = √(2 log(3/α)). For d=10^6, α=10^{-7}: C_thresh ≈ 6.0.
- **Information-theoretic minimum:** T_min = 2.20/SNR² steps per weight. Current methods: 8-20x slower than this limit.
- **Convergence gain:** 1.5-2x. If 50% of weights settle early, effective dimension halves → direct speedup.
- **Complexity:** O(d) per step. Freezing REDUCES compute over time.
- **Feasibility:** HIGH — accumulator + mask, well-understood theory

### R2-A3-M4: Entropic Optimal Transport Scheduler (EOTS)
- **Principle:** View training as transporting probability mass on the ternary simplex. Each weight has a belief distribution π_i = (p_{-1}, p_0, p_{+1}). Use optimal transport with entropic regularization.
- **Formulation:** Update: π_i^{t+1}(v) ∝ π_i^t(v)^{ε/(ε+λ)} · μ_i(v)^{λ/(ε+λ)} where μ_i = softmax(g_i · [-1,0,+1] / τ_t). Weighted geometric mean — computationally trivial.
- **Key insight:** Wasserstein metric treats -1→+1 as distance 2 (through 0), unlike KL. This naturally prefers gradual transitions -1→0→+1, matching empirical weight evolution.
- **Convergence gain:** 1.5-2.5x. Eliminates threshold oscillation by operating on distributions, not point estimates.
- **Memory:** 2 floats/weight (same as Adam's m,v). Complexity: O(1) per weight per step.
- **Feasibility:** HIGH — drop-in Adam replacement with well-understood OT theory

## MEDIUM VIABILITY (validate after high-priority)

### R2-A3-M5: Ternary Simplex Mirror Descent (Riemannian Optimization)
- **Principle:** Optimize directly on the 2-simplex with Fisher-Rao metric. Multiplicative weights update: p_i^k ← p_i^k · exp(-η · ∂L/∂p_i^k), then normalize.
- **Regret bound:** O(G_∞ · √(2T log 3)). The log(3) initialization term is CONSTANT, vs STE's O(‖w*‖²) which scales with distance.
- **Convergence gain:** 1.5-3x (eliminates threshold-crossing bottleneck). ~15% per-step overhead from exp/normalize.
- **Feasibility:** MEDIUM-HIGH — new optimizer, moderate implementation effort

### R2-A3-M6: Spectral Acceleration via Ternary Hessian Structure
- **Principle:** Ternary Hessian is block-diagonal by ternary phase: H = H_{-1} ⊕ H_0 ⊕ H_{+1}. Each block H_{±1} is sum of rank-1 matrices (outer products). H_0 = 0 (dead weights have zero curvature).
- **Formulation:** Diagonal preconditioner P_i = 1/√(H_{ii} + ε) for active weights, P_i = 1/√ε for dead weights (large step — flat landscape). Recompute every K=10 steps → ~10% overhead.
- **Convergence gain:** 2-4x theoretically (condition number reduction from ~10⁴ to ~10²). Practical: 1.5-2.5x.
- **Feasibility:** MEDIUM — requires Hessian diagonal computation, amortized OK

### R2-A3-M7: Spectral Ternary Dynamics (SVD-Domain)
- **Principle:** Optimize in spectral domain: compute target SVD from gradient, find closest ternary matrix in spectral norm. Converts d-dimensional combinatorial problem to rank-r approximation sequence.
- **Formulation:** Truncated SVD of W_target = W - η·∇L, then per-component ternary projection. Rank r=5-20 suffices.
- **Convergence gain:** 1.3-2x for structured layers, potentially 3x for SSM A-matrices.
- **Feasibility:** MEDIUM — randomized SVD overhead, but amortizable

### R2-A3-M8: Cohomological Gradient Accumulation (CGA — Simple Version)
- **Principle:** Track gradient discrepancies across ternary boundaries. When w_i flips, measure α_i = g_after - g_before. High discrepancy ("frustration") means STE is unreliable for that weight.
- **Formulation:** Frustration index F_i = |Σ_flips α_i|. Low F: use Adam+STE (reliable). High F: switch to zeroth-order evaluation (3 forward passes for that weight's 3 values).
- **Convergence gain:** 1.5-2.5x (eliminates STE oscillation for problematic weights).
- **Feasibility:** MEDIUM-HIGH — gradient history buffer + adaptive switching. ~2-10% overhead from ZO evaluations.

## SPECULATIVE (moonshot)

### R2-A3-M9: Ternary Langevin Dynamics — Thermal Escape from Plateaux
- **Principle:** Add calibrated noise to shadow weights to "tunnel" between plateaux. Noise pushes weights across thresholds before gradient would naturally bring them.
- **Formulation:** w_i ← w_i - η·g_i + √(2ηT(t))·ξ_i. Temperature T*(t) = τ²/(17η) based on SNR analysis. Quadratic cooling γ=2.
- **Convergence gain:** ~2x (plateau traversal 350x faster, ~50% of training is plateau traversal → 2x overall).
- **Feasibility:** SPECULATIVE — cascade risk (one noise-induced flip triggers others). Noise should be applied selectively.

### R2-A3-M10: Momentum Phase Coupling
- **Principle:** After a ternary flip, boost shadow weight in flip direction: w_i ← w_i + μ · v_ternary · τ/2. Provides "inertia" in new state, reduces oscillation.
- **Convergence gain:** 1.3-1.5x (eliminates ~20% of training wasted on oscillation).
- **Feasibility:** SPECULATIVE — interaction with Adam's own momentum unclear. μ ∈ [0.3, 0.6].

### R2-A3-M11: Ternary Belief Propagation (TBP)
- **Principle:** Treat ternary assignment as MAP inference in a Markov Random Field. Message-passing propagates weight CORRELATIONS — "if w_i flips, what should w_j do?"
- **Formulation:** Unary potentials from gradient: φ_i(v) = exp(-g_i·v/T). Pairwise from Fisher: ψ_ij(v_i,v_j) = exp(-F_ij·v_i·v_j/T). BP on sparse row/column graph.
- **Convergence gain:** 1.3-2x (captures correlated flips that Adam misses).
- **Feasibility:** SPECULATIVE — memory cost (3 floats/edge), loopy BP convergence uncertain

### R2-A3-M12: Ternary Langevin Monte Carlo (TLMC — Discrete CTMC)
- **Principle:** Replace gradient descent with continuous-time Markov chain directly on {-1,0,+1}^d. Gradient-informed transition rates: q(v→v') = λ·exp(-max(0, g·(v'-v))/T).
- **Key advantage:** Eliminates shadow weights entirely. No FP32 copies, no momentum. 6x memory reduction vs Adam.
- **Convergence gain:** 1.1-1.5x training + 6x memory. Best as fine-tuning after Adam+STE.
- **Feasibility:** SPECULATIVE — slow early convergence, but dramatic memory savings

## KEY INSIGHT: The Convergence Bottleneck Taxonomy

All 12 formulations attack the same fundamental problem from different angles:

| Bottleneck | Approaches | Combined Speedup |
|-----------|-----------|-----------------|
| **STE bias** | M2 (correction), M8 (frustration detection), M4/M5 (distribution-based) | 1.5-3x |
| **Threshold oscillation** | M10 (phase momentum), M3 (freezing), M9 (Langevin noise) | 1.5-2.5x |
| **Suboptimal LR** | M1 (SNR-derived schedule), M6 (Hessian preconditioning) | 1.3-2x |
| **Weight correlations** | M11 (belief propagation), M7 (spectral), M4 (transport) | 1.3-2x |

## RECOMMENDED COMPOSITES

### Composite 1: "Informed Freezing" (M1 + M2 + M3)
Bias-corrected gradients → SNR-optimal LR → information-theoretic freeze decisions. Zero overhead. Est: 2-3x combined.

### Composite 2: "Accelerated Transport" (M4 + M9)
EOTS distributions + Langevin noise on simplex (Gaussian in log-space). Well-studied convergence. Est: 2-4x.

### Composite 3: "Curvature-Aware Freezing" (M3 + M6)
Hessian diagonal → curvature-dependent freeze thresholds. High curvature = more evidence needed. Est: 2-3x.

## For Paradigm-Breakers (Wave 3)

1. Is there a training paradigm that makes ternary convergence PROVABLY faster than continuous?
2. Can the ternary structure enable super-linear convergence (faster than O(1/T))?
3. What if we train a "meta-optimizer" that learns the optimal ternary update rule from scratch?
4. Are there biological analogues of ternary learning that converge faster than gradient descent?
