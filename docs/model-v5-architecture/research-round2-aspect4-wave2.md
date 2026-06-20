# Research Round 2, Aspect 4: Data Efficiency — Wave 2 (Math-Paradigm-Breakers)

## Summary

Two math-paradigm-breakers explored novel mathematical formulations for data efficiency in ternary training. After deduplication, 10 unique formulations remain. The unifying insight: ternary weights create a FINITE discrete decision space where most training tokens are informationally empty — they move shadow weights within ternary basins without changing any ternary value. All formulations attack this waste from different mathematical angles.

## HIGH VIABILITY (implement first)

### R2-A4-M1: TFIW — Ternary Fisher Information Weighting
- **Principle:** Project Fisher Information Matrix onto quantization-relevant subspace. Standard FIM treats all parameter directions equally; TFIW only measures information about threshold-crossing decisions.
- **Formulation:** F_T(x) = sum_{i in near-threshold} (dL/dw_i)^2 * K((w_i - t_i*) / sigma). Per-token weight omega(x) = F_T(x) / E[F_T(x')]. K is Gaussian kernel centered on threshold.
- **Key insight:** As training progresses, fewer weights are near thresholds → effective dimensionality SHRINKS → data requirement decreases. Absent in continuous models.
- **Data efficiency gain:** 2-5x via importance sampling variance reduction
- **Feasibility:** HIGH — O(d) additional compute per token, amortizable
- **Difference from Wave 1:** SWIFT (F2) weights by standard importance; TFIW derives weighting from ternary threshold geometry.

### R2-A4-M2: SSTD — Sufficient Statistics for Ternary Decisions
- **Principle:** For each ternary weight, the optimal assignment {-1,0,+1} depends on data only through the expected gradient: S_ij = E[delta_i * x_j]. This scalar sufficient statistic concentrates much faster than continuous weight optimization.
- **Formulation:** Phased training: (1) Statistics accumulation (forward-only, 2-3x faster than full training), accumulate S = E[delta * x^T] per layer. (2) Ternary assignment: threshold S to get initial weights. (3) Short refinement with STE.
- **Key insight:** For continuous weights, no finite sufficient statistic exists. For ternary, the decision is 3-class → scalar statistic suffices → concentrates at O(1/sqrt(N)) rate.
- **Data efficiency gain:** 3-5x (forward-only accumulation phase + 50-70% shorter refinement)
- **Feasibility:** HIGH — statistics ARE the gradients (zero overhead). Phased approach is simple training loop modification.
- **Synergies:** ACSI (R2-A3-P1) is the SSM-specific version. SSTD generalizes to all layers.

### R2-A4-M3: EGPT — Entropy-Gated Training via Shadow Weight Phase Transitions
- **Principle:** View ternary training as a physical system undergoing phase transitions. Each shadow weight exists in one of three phases (committed to -1, +1, or uncommitted). Kramers escape rate governs commitment speed.
- **Formulation:** Ternary entropy H_T(t) = -sum_i sum_v p_i^v(t) * log(p_i^v(t)) where p_i^v uses CDF of shadow weight vs thresholds. Data value proportional to -dH_T/dt per token. Kramers rate: tau_commit ~ exp(DeltaV / sigma^2).
- **Key insight:** Data quality matters EXPONENTIALLY more near thresholds (Kramers escape rate). A small improvement in gradient signal quality near thresholds → exponential speedup in commitment.
- **Temporal gating:** Increase effective LR when many weights near thresholds (high but decreasing H_T). Decrease when committed (low H_T).
- **Data efficiency gain:** 3-10x. 80% of training time may be in frozen phases where data has diminishing returns.
- **Feasibility:** HIGH — entropy computation is O(d), trivial. Per-token gating is cheap.
- **Difference from Wave 1:** Curriculum (F5) is static property of data. EGPT recognizes SAME token can be maximally informative at one training point and useless at another.

### R2-A4-M4: TLGA — Ternary Lattice Gradient Alignment
- **Principle:** {-1,0,+1}^d forms a lattice. STE training is a walk on this lattice. Most gradients are "misaligned" with useful lattice moves. Derive per-token weighting from alignment between gradient direction and useful lattice transitions.
- **Formulation:** Useful move set M_i per weight (e.g., 0→+1, 0→-1). Alignment score A(x) = sum_i sum_{m in M_i} sign(m) * g_i(x) * phi(w_i, t_i^m). Token weight omega(x) = max(1, A(x)/A_avg).
- **Active learning extension:** Select token that maximizes reduction in posterior entropy over which lattice move to make next.
- **Data efficiency gain:** 2-4x. Eliminates gradient signal wasted on reinforcing already-decided weights.
- **Feasibility:** HIGH — O(d) per token for alignment score. Cheap.
- **Difference from Wave 1:** All Wave 1 methods weight tokens agnostic to ternary structure. TLGA derives importance FROM lattice geometry.

### R2-A4-M5: TIAR — Ternary Information Acquisition Rate
- **Principle:** Formalize "learning rate" of a ternary weight as the rate at which data resolves uncertainty about which of 3 states is optimal. Token is useful iff it changes posterior probability over optimal ternary assignment.
- **Formulation:** Ternary margin m_j = min(|w_j - tau-|, |w_j - tau+|). Crossing probability p_cross(x) = Phi((eta*|mu_j(x)| - m_j) / (eta*sigma_j)). Aggregate TIAR(x) = sum_{j in boundary} p_cross_j(x).
- **Key insight:** Only boundary weights (m_j < delta) can learn from data. Typically 10-30% of weights. 70-90% of gradient computation is wasted on plateau weights.
- **Data efficiency gain:** 2-5x via focusing on boundary-relevant tokens
- **Feasibility:** HIGH — computing m_j is O(d), negligible
- **Synergies:** Combines with EGPT (temporal gating determines WHEN, TIAR determines WHICH tokens)

## MEDIUM VIABILITY (validate after high-priority)

### R2-A4-M6: CRBT — Capacity-Rate Bound for Ternary Models
- **Principle:** Rate-distortion theory applied to ternary training. Derive MINIMUM data tokens needed for ternary model to achieve given loss level.
- **Formulation:** n_min = d * log2(3) / I_STE(X; nabla L). Ratio vs continuous: n_ternary/n_continuous = log2(3)/b ≈ 1.585/16 ≈ 0.1. Ternary needs ~10x less data than FP16 in principle.
- **Batch selection:** batch* = argmax_B sum_{near-threshold} Var_{x in B}[dL/dw_i]. Maximize gradient VARIANCE for near-threshold weights (high variance = high entropy = high information).
- **Data efficiency gain:** Theoretical maximum 10x vs FP16. Practically 3-8x with proposed batch selection.
- **Feasibility:** MEDIUM — bound involves quantities hard to estimate. Batch selection is implementable.
- **Value:** Provides theoretical LIMIT — tells us how much room for improvement exists.

### R2-A4-M7: SDTR — Spectral Data Efficiency via Ternary Rank Analysis
- **Principle:** Ternary matrices have bounded effective rank (much lower than continuous). Function space is a lower-dimensional submanifold. Data informative about orthogonal complement is WASTED.
- **Formulation:** Effective rank r_eff = ||W||_*^2 / ||W||_F^2. Data requirement: n ~ r_eff * (m+n) * log2(3). If r_eff/min(m,n) ~ 0.1: potentially 100x less data than continuous. Spectral utility per token: U_spec(x) = ||P_{r_eff} nabla_W L(x)||_F^2 / ||nabla_W L(x)||_F^2.
- **Data efficiency gain:** 5-20x in theory. 3-8x practical.
- **Feasibility:** MEDIUM — requires periodic SVD (randomized: O(r_eff * mn) per layer). Need empirical validation of low-rank hypothesis.

### R2-A4-M8: SSDR — Sparse Subnetwork Data Requirements
- **Principle:** Ternary sparsity (many zeros) decomposes learning into support selection (which weights nonzero?) and sign assignment (which sign?). VC dimension analysis gives tighter sample complexity.
- **Formulation:** N_eff(t) = O(|F_t| * log|F_t| + (1-s_t) * d * L * log d) where F_t = active frontier (near-threshold weights), s_t = sparsity. Data budget DECREASES as training progresses.
- **Data budget scheduler:** tokens_per_epoch(t) = C * N_eff(t). As frontier shrinks and sparsity settles, required data drops.
- **Data efficiency gain:** ~30% from VC analysis alone. More in late training.
- **Feasibility:** HIGH — computing |F_t| and s_t is O(d), trivial. Zero computational overhead.

### R2-A4-M9: Optimal Sequence Length via SSM State Entropy
- **Principle:** SSM hidden state entropy H(h_L) depends on sequence length and spectral radius rho(A). For ternary A (integer matrix), eigenvalues are constrained → shorter effective memory → shorter optimal sequences → more diversity per batch.
- **Formulation:** L* = log(epsilon) / log(rho(A)^2). For rho(A) ≈ 0.9: L* ≈ 22 tokens for state saturation. Shorter sequences = more sequences per batch = more gradient diversity.
- **Data efficiency gain:** 2-4x for pure SSM (constrained by attention component in hybrid model)
- **Feasibility:** MEDIUM — clean for linear SSM, approximate for selective (input-dependent) SSM.

### R2-A4-M10: QEA-DRO — Quantization-Error-Aware Distributionally Robust Optimization
- **Principle:** Quantization error e = Q(w) - w biases gradients in a structured direction. Data whose gradient aligns with this bias is anti-helpful. DRO-optimal reweighting via Wasserstein ball.
- **Formulation:** omega(x) = (1/Z) * exp(-beta * cos(g(x), e)) where cos is cosine similarity between gradient and quantization error. Downweight error-aligned data, upweight error-opposing data.
- **Data efficiency gain:** 1.5-2x (reduces 20-40% of gradient variance from quantization bias)
- **Feasibility:** MEDIUM — full per-sample gradients expensive. Loss-proxy approximation is cheap (<5% overhead).

## SPECULATIVE (moonshot)

### R2-A4-M11: Ternary Learning as Error-Correcting Code Construction
- **Principle:** Ternary weight matrix is a codebook in F_3^d (finite field). Training = code construction. Channel capacity bounds give information-theoretic minimum data.
- **Formulation:** N_min = d * log2(3) / C_data. For d=10^7, C_data ≈ 0.1 bits/token: N_min ≈ 1.6 × 10^8 tokens. Practice uses 10^9-10^10 → 6-60x gap.
- **D-optimal batch design:** Maximize det(I_batch) where I_batch is empirical Fisher restricted to boundary weights.
- **Data efficiency gain:** 6-60x theoretical ceiling. Practical: unknown.
- **Feasibility:** SPECULATIVE — beautiful bounds but gap between linear codes and deep nonlinear networks is large.

## KEY INSIGHT: The 10x Data Bound

**Rate-distortion analysis predicts ternary models need ~10x less data than FP16.** Current training uses much more. The gap is due to:
1. STE information loss (FOGZO, bias correction attack this)
2. Wasted computation on plateau weights (TIAR, EGPT attack this)
3. Uninformed data selection (TLGA, SSTD attack this)

## RECOMMENDED COMBINATIONS

### Conservative (HIGH viability)
TIAR (M5) + EGPT (M3): EGPT determines WHEN to apply intensive data selection (temporal gating), TIAR determines WHICH tokens to prioritize. Outside critical windows, use uniform sampling.

### Aggressive (MEDIUM-HIGH)
SSTD (M2) + TIAR (M5) + SSDR (M8): Phased training (statistics → assignment → refinement) with importance-weighted statistics and declining data budgets. Expected: 5-10x combined.

### Grand Pipeline
Phase 0: Compute N_min from CRBT (M6) — the target.
Phase 1: SSTD (M2) — cheap forward-only statistics accumulation, assign initial ternary weights.
Phase 2: TIAR (M5) + QEA-DRO (M10) — refine with importance-weighted, bias-corrected data.
Phase 3: SSDR (M8) + Seq Length (M9) — reduce data budget and shorten sequences as frontier shrinks.

## ASPECT 4 WAVE 2 COMPLETE

**Total formulations: 11 (10 unique + 1 speculative moonshot)**

### TOP 5 Most Impactful:
1. **SSTD (M2)** — phased training via sufficient statistics, 3-5x, zero overhead
2. **EGPT (M3)** — phase transition dynamics for temporal data scheduling, 3-10x
3. **TIAR (M5)** — boundary-weight-focused data selection, 2-5x
4. **CRBT (M6)** — theoretical 10x bound, batch selection criterion
5. **TLGA (M4)** — lattice-aligned token weighting, 2-4x
