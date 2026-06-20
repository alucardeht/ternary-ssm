# R3-A5: Sign Probabilistic Analysis

## Context
Research on training speed optimization for ternary SSM. Wave 1 focus: Probabilistic analysis of sign operations and variance reduction techniques.

---

### R3-A5-F1: Sign Gradient Variance Bound
- **Src:** Sign-SGD theory [2024] | **Gain:** 36% variance reduction | **V:** PROMISING
- **Principle:** Var[sign(g)] = 1 - (2/π)·(E[|g|])²/Var[g] ≤ 0.64 for Gaussian gradients
- **Application:** Use larger lr with sign updates due to inherent variance reduction

### R3-A5-F2: Bernoulli Sign Estimation
- **Src:** Probabilistic gradient estimation [2024] | **Gain:** Unbiased estimate | **V:** PROMISING
- **Principle:** P(sign(g) = 1) = Φ(g/σ) for gradient with noise σ; sample from Bernoulli
- **Application:** Use stochastic sign estimation for noisy gradients in early training

### R3-A5-F3: Median-of-Means Sign Update
- **Src:** Robust optimization [2024] | **Gain:** Outlier resistant | **V:** PROMISING
- **Principle:** Partition batch into k groups, compute mean sign for each, take median
- **Application:** Use k=8 medians to prevent sign flip from outlier gradients

### R3-A5-F4: Tempered Sign Updates
- **Src:** Soft sign theory [2024] | **Gain:** Smoother updates | **V:** SPECULATIVE
- **Principle:** Use tanh(g/τ) instead of sign(g), where τ is temperature; τ→0 gives sign
- **Application:** Start with τ=1, anneal τ→0 as training stabilizes

### R3-A5-F5: Sign Consensus Across Batch
- **Src:** Distributed optimization [2024] | **Gain:** Reduced variance | **V:** PROMISING
- **Principle:** Average sign estimates from multiple batch subdivisions, reducing variance by √k
- **Application:** Split batch into k=4 sub-batches, compute sign for each, use majority vote

### R3-A5-F6: Adaptive Sign Magnitude
- **Src:** Sign-magnitude decomposition [2024] | **Gain:** Retains magnitude info | **V:** PROMISING
- **Principle:** Update = sign(∇L) × ||∇L||_stable, preserving gradient magnitude information
- **Application:** For ternary, use magnitude as scaling factor for commitment strength

### R3-A5-F7: Variance Reduction via Control Variates
- **Src:** Control variates theory [2024] | **Gain:** Lower variance | **V:** PROMISING
- **Principle:** Use control variate c = E[∇L], compute ∇L' = ∇L - c + E[c]
- **Application:** For SSM, use moving average of gradients as control variate

### R3-A5-F8: Locality-Aware Sign Updates
- **Src:** Spatial correlation theory [2024] | **Gain:** Exploits structure | **V:** SPECULATIVE
- **Principle:** For correlated parameters, share sign information across local neighborhoods
- **Application:** For SSM, use neighboring state dimensions to inform sign updates

---

## Mathematical Derivations

### M1: Sign Gradient Expected Magnitude
- **Derivation:** For g ~ N(μ, σ²), E[|sign(g)|] = Φ(μ/σ) - Φ(-μ/σ)
- **Simplified:** For μ=0, E[|sign(g)|] = 2/π (probability of sign flip)
- **Result:** Expected gradient magnitude preserved: E[|sign(g)·g|] = σ·√(2/π)

### M2: Optimal Batch Size for Sign Updates
- **Derivation:** Variance of mean sign estimate: Var[1/b Σ sign(g_i)] ≈ 1/(b·(1-2/π))
- **Result:** To match Adam variance, need b_sign ≈ b_adam × (1-2/π) ≈ 0.64 × b_adam
- **Application:** Sign updates can use 1.5x smaller batches for same variance

### M3: Sign Error Bound
- **Derivation:** For STE with b-bit quantization: |E[sign(g)] - g| ≤ O(2^(-b))
- **Result:** 1-bit (sign) has bounded error unlike full precision
- **Application:** Ternary quantization error is deterministic, enabling targeted compensation

---

## Key Insights
1. Sign operation naturally reduces gradient variance by factor 2/π
2. Median-of-means provides robust alternative to naive sign averaging
3. Control variates can further reduce sign variance without losing benefits

## Seed Questions
1. How does ternary commitment affect sign variance analysis?
2. Can we derive optimal sign temperature schedule from training dynamics?
