# Research Round 2, Aspect 4: Data Efficiency — Wave 3 (Paradigm-Breakers)

## Summary

Two paradigm-breakers explored radical alternatives for data-efficient ternary training. After deduplication (DTAS ≈ HTC ≈ HFTA merged, TASA ≈ TGCS merged), 7 unique paradigm proposals remain. The unifying insight: gradient descent with STE is a CONTINUOUS optimization algorithm forced onto a DISCRETE space via continuous relaxation (shadow weights). This impedance mismatch is the ROOT CAUSE of data inefficiency — every token that moves a shadow weight without flipping a ternary value is wasted. Every proposal below eliminates or bypasses this mismatch.

## TIER 1 — HIGH VIABILITY (test first)

### R2-A4-P1: HTC — Hebbian Ternary Construction
- **Paradigm shift:** Eliminate gradient descent entirely. COMPUTE weights from data correlations in a single forward pass instead of iteratively learning them.
- **Key insight:** Each ternary weight w_ij answers: "Does input i positively correlate with output j (+1), negatively (-1), or not significantly (0)?" This is exactly what correlation analysis computes — no gradients needed.
- **Method:** (1) For each layer, compute cross-correlation C_ij = E[(x_i - mu_x)(y_j - mu_y)] from one data pass. (2) Ternary assign: w_ij = sign(C_ij) if |C_ij| > tau, else 0. (3) Process layers greedily (layer L output feeds layer L+1 input). (4) Optional: 500-2000 STE fine-tuning steps for higher-order interactions.
- **Why ternary-specific:** Sign(correlation) IS the ternary decision. Continuous weights need magnitude too (requires much more data). Zero threshold provides automatic feature selection.
- **Data efficiency gain:** 20-100x. Need O(m) samples per layer to estimate correlations (m = layer width). For 6M model: ~100K-500K tokens total vs 10M+ for SGD.
- **Feasibility:** HIGH — embarrassingly simple algorithm. Implementable in a day. Question is purely about quality.
- **Synergies:** Natural initializer for DCDLO (P2) or BPTA (P4). Combined: HTC init + discrete refinement = full gradient-free pipeline.

### R2-A4-P2: DCDLO — Discrete Coordinate Descent with Loss Oracle
- **Paradigm shift:** Optimize directly in {-1, 0, +1}^N. No shadow weights, no STE, no continuous relaxation. Every step changes actual ternary values.
- **Key insight:** Use one backward pass to approximate loss change for ALL possible flips simultaneously. Select top-K most beneficial flips and apply them. 100% of computation produces actual ternary changes (vs 1-5% with SGD/STE).
- **Method:** (1) Compute gradient g_ij via one backward pass. (2) For each weight, evaluate delta_loss ≈ g_ij * (v' - v_current) for alternative ternary values. (3) Flip top-K weights with largest |delta_loss|. (4) Repeat with new data batch.
- **Why ternary-specific:** Continuous coordinate descent needs 1D optimization per coordinate (expensive). Ternary has exactly 3 values — evaluate all three, pick best. O(1) per weight.
- **Data efficiency gain:** 10-20x. Every step guarantees ternary changes. No wasted computation on plateau weights.
- **Memory savings:** No shadow weights, no Adam m/v. 4x memory reduction.
- **Feasibility:** HIGH — 50 lines of code. Replace Adam update with flip-selection.
- **Difference from Wave 2 CFS (R2-A3-P3):** CFS is from Aspect 3 (convergence speed). DCDLO focuses specifically on data efficiency — fewer tokens needed, not faster convergence per se. Mechanism is similar but framing and optimization target differ.

## TIER 2 — MEDIUM-HIGH VIABILITY (test after Tier 1)

### R2-A4-P3: WATS — Weight-Adversarial Token Synthesis
- **Paradigm shift:** Instead of selecting informative tokens from a fixed corpus, GENERATE tokens maximally informative for specific ternary boundary decisions.
- **Key insight:** At any step, only 1-5% of weights are near boundaries. For each boundary weight, specific input patterns maximally resolve the decision. Synthesize those patterns.
- **Method:** (1) Identify boundary weights B = {(i,j) : |w_float_ij - threshold| < delta}. (2) Compute decision gradient g_decision(x) for boundary weights. (3) Generate/retrieve tokens that maximize |g_decision|. (4) Train on synthesized tokens.
- **Generator options:** Gradient-based synthesis in embedding space, template-based combination, or retrieval from reference corpus via approximate nearest neighbor.
- **Data efficiency gain:** 10-50x. Instead of processing 100 random tokens hoping 1 is informative, synthesize 1 guaranteed-informative token.
- **Feasibility:** MEDIUM-HIGH — active learning + adversarial generation are well-established. Ternary-specific formulation (boundary weights + decision gradients) is concrete.
- **Risk:** Synthesized tokens may not cover all linguistic phenomena. Model could excel at boundary resolution but miss broader patterns.

### R2-A4-P4: BPTA — Belief Propagation for Ternary Assignment
- **Paradigm shift:** Replace optimization with probabilistic inference. Compute posterior P(W | Data) and assign each weight to its most probable value.
- **Key insight:** Maintain a belief tensor b_ij(v) = [p(-1), p(0), p(+1)] per weight. Each data batch updates beliefs via gradient-informed Bayesian update. Confidence = max(b) provides principled uncertainty quantification.
- **Method:** (1) Initialize beliefs uniformly: b(v) = 1/3. (2) Per batch: compute gradient, update b_ij(v) *= exp(-g_ij * (v - w_current) / T), normalize. (3) Assign w_ij = argmax_v b_ij(v). (4) Freeze weights where max(b) > 0.95. (5) Focus data on uncertain weights.
- **Why ternary-specific:** Belief over 3 values = 3 floats per weight (tractable). Continuous posterior = infinite parameters (intractable). MAP assignment = argmax of 3 values (trivial).
- **Memory:** 3 floats/weight (12 bytes) = same as Adam but RICHER information (uncertainty, not just momentum).
- **Natural stopping criterion:** Stop when all beliefs exceed confidence threshold.
- **Data efficiency gain:** Approaches MDL bound (~625K tokens for 6M model). Principled adaptive data allocation emerges from beliefs.
- **Feasibility:** MEDIUM-HIGH — computationally equivalent to gradient training. Replace Adam with belief updates.

## TIER 3 — MEDIUM VIABILITY (speculative)

### R2-A4-P5: LOTD — LSH-Optimal Ternary Design
- **Paradigm shift:** Design weights as information-maximizing hash functions tuned to data distribution geometry.
- **Key insight:** Ternary weight vector defines signed hash h(x) = sign(w·x). Optimal hash computed from input-output cross-covariance SVD, then ternary-approximated.
- **Method:** (1) Estimate Sigma_xy = E[(x-mu_x)(y-mu_y)^T]. (2) SVD: Sigma_xy = U*S*V^T. (3) Ternary-approximate each singular vector: keep top-p% entries as ±1, rest as 0.
- **Data efficiency gain:** 10x+. ~65K tokens for reliable covariance estimation at 6M scale.
- **Feasibility:** MEDIUM — works well for linear/SSM layers, uncertain for attention.

### R2-A4-P6: TGCS — Ternary Ground-state via Combinatorial Search
- **Paradigm shift:** Frame weight assignment as finding ground state of a spin system using simulated annealing with parallel tempering.
- **Key insight:** Simulated annealing converges to global optimum (probability 1) on discrete spaces. Data enters through energy function. A FIXED small archive (100K-500K tokens) suffices because randomness is in temperature, not data sampling.
- **Method:** (1) Initialize with HTC/LOTD. (2) Propose random flips. (3) Accept with probability min(1, exp(-delta_E/T)). (4) Anneal T from high to low. (5) Parallel tempering: R replicas at different temperatures, periodic swaps.
- **Why ternary-specific:** Flipping w_ij changes output by exactly (v'-v)*x_i — incremental delta_E is O(1). 3-state Potts model is extremely well-studied in physics.
- **Data efficiency gain:** 50-200x (unlimited reuse of fixed archive). Convergence speed uncertain.
- **Feasibility:** MEDIUM — physics is sound, main risk is convergence speed at scale.

### R2-A4-P7: ICTA — Iterative Compression-based Ternary Assignment
- **Paradigm shift:** Training is LOSSY COMPRESSION. Use compression algorithms (MDL principle) instead of optimization algorithms.
- **Key insight:** MDL optimal model minimizes model_cost + data_given_model_cost. Model cost = N * log2(3) bits. Process weights sequentially, assigning each to minimize conditional entropy of remaining data.
- **Data efficiency gain:** 10-100x theoretical. MDL bound: O(N * H(W)) tokens where H(W) ~ 1.2 bits for ternary.
- **Feasibility:** MEDIUM — elegant theory, sequential processing makes parallelization difficult.

## UNIFYING INSIGHT: The Gradient-Free Pipeline

All proposals converge on a multi-phase gradient-free approach:

| Phase | Method | Tokens | Purpose |
|-------|--------|--------|---------|
| Init | HTC/LOTD (P1/P5) | ~50K-100K | Statistical construction from correlations |
| Refine | BPTA (P4) or DCDLO (P2) | ~200K | Resolve uncertain weights via inference/search |
| Polish | TGCS (P6) or STE fine-tune | ~100K | Escape local optima, higher-order interactions |
| **Total** | | **~350K-400K** | **vs 10M+ for pure SGD = 25-30x** |

## KEY SYNERGIES AND RECOMMENDED COMBINATIONS

### Conservative Path (HIGH viability)
HTC (P1) + DCDLO (P2): Hebbian init + discrete refinement. Entirely gradient-free. Expected: 10-30x data reduction.

### Principled Path (MEDIUM-HIGH viability)
LOTD (P5) + BPTA (P4): SVD-based init + Bayesian refinement with principled uncertainty. Expected: 20-50x. Approaches MDL bound.

### Aggressive Path (MEDIUM viability)
HTC (P1) + TGCS (P6): Construction + simulated annealing on fixed archive. Unlimited data reuse. Expected: 50-200x if annealing converges.

## ASPECT 4 COMPLETE — SUMMARY ACROSS ALL WAVES

### Wave 1 (Web): 12 findings on distillation, augmentation, selection, synthetic data
### Wave 2 (Math): 11 formulations (TFIW, SSTD, EGPT, TLGA, TIAR, CRBT, SDTR, SSDR, SeqLen, QEA-DRO, ECC)
### Wave 3 (Paradigm): 7 radical alternatives (HTC, DCDLO, WATS, BPTA, LOTD, TGCS, ICTA)

**Total unique findings Aspect 4: 30**

### TOP 5 Most Impactful for Our Project:
1. **HTC (P1)** — single-pass correlation-based construction, 20-100x, HIGH viability, implementable in a day
2. **DCDLO (P2)** — discrete coordinate descent, 10-20x, HIGH viability, 50 lines of code
3. **SSTD (M2)** — phased training via sufficient statistics, 3-5x, zero overhead
4. **BPTA (P4)** — Bayesian belief propagation replaces Adam, approaches MDL bound
5. **OPSD (F1)** — on-policy self-distillation, 4-8x, proven technique
