# Research Round 1 — Aspect 2: Ternary Backward Pass Optimization

**Consolidated Findings Document**  
**Date:** 2026-03-03  
**Source Files:** r1-a2-w1-themeA, r1-a2-w1-themeB, r1-a2-w2-math-themeA, r1-a2-w2-math-themeB, r1-a2-w2-paradigm_themeA, r1-a2-w2-paradigm_themeB  
**Total Unique Findings:** 34

---

## Overview

This document consolidates all research findings for **Aspect 2: Ternary Backward Pass Optimization** from Research Round 1. The findings are organized by category (IMPLEMENT, EXPLORE, REJECT) based on expected gain and feasibility for our ternary SSM architecture.

---

## Category: IMPLEMENT

Findings with clear mathematical foundation, measured gains, and zero tradeoffs.

### GPU Kernel Fusion (Theme A)

| ID | Finding | Expected Gain | Key Insight |
|----|---------|---------------|-------------|
| R1-A2-F1 | **Autograd-Function-Backward-Fusion** | 15-30% backward speedup | Fusing threshold computation directly into backward kernel eliminates separate forward result reload |
| R1-A2-F2 | **Triton-Kernel-Fused-STE-Backward** | 1.5-2x memory bandwidth | Fuse ternary threshold + gradient routing + weight update in single kernel launch |
| R1-A2-F3 | **In-Place-Backward-Gradient-Routing** | 50% memory traffic reduction | Modifying gradient tensor in-place avoids allocating separate gradient buffer |
| R1-A2-F4 | **Saved-Tensor-Minimization-Backward** | Eliminates 1 saved tensor per op | Only save ternary weights and threshold; recompute quantization mask in backward |
| R1-A2-F5 | **Lazy-Backward-Activation-Fusion** | 20-40% backward fusion speedup | torch.compile with Inductor fuses `hardtanh(grad)` + `mask * grad` + `matmul` into single kernel |

### Gradient Computation (Theme B)

| ID | Finding | Expected Gain | Key Insight |
|----|---------|---------------|-------------|
| R1-A2-G1 | **Denoising-Dequantization-Transform** | 5.3% accuracy (A1W1) | Ridge-regression-derived transform explicitly corrects quantization error via ∂g/∂q = Cov(xq)/(Var(q)+λ) |
| R1-A2-G2 | **NVFP4-Stochastic-Rounding-Backward** | Matches BF16 baseline | Round-to-nearest forward, stochastic backward — asymmetry breaks correlation between quantization noise and gradient |
| R1-A2-G3 | **Gradient-Norm-Threshold-Awareness** | Theoretical bound | When ||g|| < √3 × σ_quant, quantized training ineffective — provides actionable failure detection metric |
| R1-A2-G4 | **Affine-Dequantization-Shortcut-Formula** | Reduces 4 matrix terms to 1 | Y~ = (s_X ⊙ s_W^T) ⊙ (Q_X · Q_W - q̄_X · q̄_W^T n) + x̄ · w̄^T n |
| R1-A2-G5 | **Extended-STE-Noise-Aware-Training** | 2.2x training speedup | Forward pass includes noise simulation; backward uses identity STE — decouples realistic noise from tractable gradients |

### Mathematical Optimizations (Wave 2)

| ID | Finding | Expected Gain | Key Insight |
|----|---------|---------------|-------------|
| R1-A2-M1 | **OLS-STE Optimal Linear STE** | 3-8% accuracy | α* = Cov(g_true, g_out)/Var(g_out) ≈ 1 - 2p where p is ternary commitment probability |
| R1-A2-M2 | **CC-STE Curvature-Corrected STE** | 2-5% accuracy | g_corrected = g_out - ½ × diag(H) ⊙ (Q(w) - w), diagonal H computed once per step |
| R1-A2-M3 | **SSM-TGS SSM Temporal Gradient Smoothing** | 5-15% gradient noise reduction | g_smoothed[t] = Σ exp(-λk)g[t-k] / Σ exp(-λj), λ = f(|λ_max(A)|) |
| R1-A2-M4 | **GOC Gradient Orthogonal Compensation** | 2-4% accuracy | Decompose gradient into parallel and orthogonal components; β = 1/(1+γ) |

### Information Theory (Wave 2 Theme B)

| ID | Finding | Expected Gain | Key Insight |
|----|---------|---------------|-------------|
| R1-A2-I1 | **Entropy-Weighted-Gradient-Routing** | 8-15% convergence speedup | Route gradient signal proportionally to information entropy — high-entropy gradients get stronger signal |
| R1-A2-I2 | **Mutual-Information-Gradient-Filter** | 20-40% gradient sparsity | Compute I(W;G) ≈ 0.5 × log(1 + σ_wg²/σ_wσ_g); zero-gradient entries where I < threshold |
| R1-A2-I3 | **KL-Divergence-Gradient-Routing** | 5-10% better alignment | Replace identity backward with KL divergence routing — matches target distribution while preserving direction |
| R1-A2-I4 | **Fisher-Information-Weighted-Backward** | 10-20% faster convergence | Weight gradient by Fisher Information: g_F = g / (1 + β × I_F(w)) |

---

## Category: EXPLORE

Promising but requires validation — speculative gains, hardware dependencies, or architectural complexity.

### Hardware-Native Approaches

| ID | Finding | Expected Gain | Key Challenge |
|----|---------|---------------|---------------|
| R1-A2-P1 | **In-Memory-Analog-Ternary-Backward** | 10-100x energy efficiency | Analog drift may erase ternary commitment during training cycles |
| R1-A2-P2 | **Photonic-Reverse-Mode-Gradient-Free** | 50-100x speedup | Light is unidirectional; must implement "virtual reverse" using separate backward-pass photonic circuits |
| R1-A2-P3 | **GPU-Tensor-Core-Inverse-Training** | 40-50% speedup | Ternary quantization breaks linearity — inverse of ternary matmul is not simple matrix inverse |
| R1-A2-P4 | **SRAM-Compute-Forward-Backward-Fusion** | Eliminates backward pass | SRAM CIM has limited precision (4-8bit); ternary-friendly CIM cell design needed |
| R1-A2-P5 | **Persistent-Thread-Backward-Streaming** | 60-70% memory BW reduction | Persistent threads require careful synchronization; SSM state preservation challenge |

### Advanced Mathematical

| ID | Finding | Expected Gain | Key Insight |
|----|---------|---------------|-------------|
| R1-A2-M5 | **SNR-STE Signal-Noise Ratio Routing** | 3-6% accuracy | Adaptively mix identity STE with prior-based gradient based on local SNR |
| R1-A2-M6 | **EWG-STE Entropy-Weighted Gradient** | 1-3% accuracy | Weight gradient updates by inverse quantization entropy |
| R1-A2-I5 | **Channel-Capacity-Backward-Pass** | 30% gradient compression | Model backward pass as channel with capacity C; compress gradient to rate ≤ C |
| R1-A2-I6 | **Particle-Gradient-Estimation-Backward** | 15-25% reduced variance | Sample k gradient particles; weighted average as backward signal |

### Novel Architectures

| ID | Finding | Expected Gain | Key Challenge |
|----|---------|---------------|---------------|
| R1-A2-P6 | **NoProp-Forward-Free-Training** | 2x training speedup | Local objectives may not align with global loss; convergence unclear for SSM recurrence |
| R1-A2-P7 | **Equilibrium-Propagation-SSMTraining** | Eliminates BPTT | Ternary discrete states hinder smooth energy minimization |
| R1-A2-P8 | **STDP-Supervised-Ternary-Update** | Eliminates gradient compute | Ternary weights have only 3 discrete states; timing signal must map to ternary flip decisions |
| R1-A2-P9 | **Forward-Forward-Ternary-Layer** | Streaming training | Ternary weights create hard boundaries; layer stacking may compound quantization error |

---

## Category: REJECT

Findings with fundamental limitations or tradeoffs that contradict our zero-tradeoff constraint.

| ID | Finding | Rejection Reason |
|----|---------|------------------|
| R1-A2-F6 | **Register-Blocking-Ternary-Backward** | Hardware-specific optimization with minimal practical benefit for our scale |
| R1-A2-F8 | **AOTAutograd-STE-Fusion** | Full graph optimization unpredictable; may break convergence guarantees |
| R1-A2-G6 | **Sub-Channel-Quantization-Outlier-Blending** | Adds complexity; outlier handling already covered by threshold awareness |
| R1-A2-G7 | **Dual-Stream-Quantization-Gradient-Path** | Theoretical framework without clear implementation path; adds 2x memory overhead |
| R1-A2-P10 | **Predictive-Coding-Ternary-SSM** | Discrete belief updates may oscillate; requires precision incompatible with ternary |
| R1-A2-P11 | **Neuroevolution-Ternary-Weight-Search** | Search space exponential (3^N); forward pass per candidate prohibitive |
| R1-A2-P12 | **Mixed-Signal-DAC-Backward-Free** | DAC/ADC overhead dominates; analog gradient signals noisy |

---

## Mathematical Summary

### Key Equations

1. **Denoising Dequantization (ridge regression solution):**
   ```
   g(q) = (Cov(xq) / (Var(q) + λ)) × (q - q̄) + x̄
   ```

2. **Optimal Linear STE:**
   ```
   α* = Cov(g_true, g_out) / Var(g_out) ≈ 1 - 2p
   ```

3. **Gradient Norm Threshold:**
   ```
   Quantized training effective when ||g|| > √3 × σ_quant
   ```

4. **Entropy-Weighted Routing:**
   ```
   g'_i = g_i × (1 + α × H(g_i) / H_max)
   ```

5. **Fisher-Weighted Backward:**
   ```
   g_F = g / (1 + β × I_F(w))
   ```

6. **SSM Temporal Gradient Smoothing:**
   ```
   g_smoothed[t] = Σ_{k=0}^K exp(-λk)g[t-k] / Σ exp(-λj)
   ```

---

## Implementation Priority

### Phase 1: Immediate Implementation (High Impact, Low Risk)

1. **R1-A2-F1** — Autograd-Function-Backward-Fusion
2. **R1-A2-F3** — In-Place-Backward-Gradient-Routing
3. **R1-A2-G2** — NVFP4-Stochastic-Rounding-Backward
4. **R1-A2-G3** — Gradient-Norm-Threshold-Awareness
5. **R1-A2-M1** — OLS-STE Optimal Linear STE
6. **R1-A2-I1** — Entropy-Weighted-Gradient-Routing

### Phase 2: Validation Required (Medium Impact)

1. **R1-A2-F2** — Triton-Kernel-Fused-STE-Backward
2. **R1-A2-G1** — Denoising-Dequantization-Transform
3. **R1-A2-M2** — CC-STE Curvature-Corrected STE
4. **R1-A2-I4** — Fisher-Information-Weighted-Backward

### Phase 3: Research (High Impact, High Risk)

1. **R1-A2-P5** — Persistent-Thread-Backward-Streaming
2. **R1-A2-P6** — NoProp-Forward-Free-Training
3. **R1-A2-P8** — STDP-Supervised-Ternary-Update

---

## Validation Checklist

- [ ] F1: Implement autograd function fusion in TernaryLinear
- [ ] F2: Add Triton fused STE backward kernel
- [ ] F3: Implement in-place gradient routing
- [ ] F4: Minimize saved tensors in backward
- [ ] F5: Test Inductor automatic fusion
- [ ] G1: Implement denoising dequantization transform
- [ ] G2: Add stochastic rounding to backward only
- [ ] G3: Monitor gradient norm threshold during training
- [ ] G4: Implement affine shortcut formula
- [ ] G5: Test extended STE with SSM noise model
- [ ] M1: Implement OLS-STE scaling factor
- [ ] M2: Add curvature correction term
- [ ] M3: Implement SSM temporal gradient smoothing
- [ ] M4: Test gradient orthogonal compensation
- [ ] I1: Implement entropy computation in backward hook
- [ ] I2: Add mutual information tracking to optimizer
- [ ] I3: Test KL divergence vs STE on simple tasks
- [ ] I4: Integrate Fisher computation into Adam

---

## Dedup Registry Updates

The following new findings should be added to the dedup-registry.txt:

```
# Round 1, Aspect 2 - Wave 1 (CUDA Kernels)
R1-A2-F1:Autograd-Function-Backward-Fusion
R1-A2-F2:Triton-Kernel-Fused-STE-Backward
R1-A2-F3:In-Place-Backward-Gradient-Routing
R1-A2-F4:Saved-Tensor-Minimization-Backward
R1-A2-F5:Lazy-Backward-Activation-Fusion

# Round 1, Aspect 2 - Wave 1 (Gradient Computation)
R1-A2-G1:Denoising-Dequantization-Transform
R1-A2-G2:NVFP4-Stochastic-Rounding-Backward
R1-A2-G3:Gradient-Norm-Threshold-Awareness
R1-A2-G4:Affine-Dequantization-Shortcut-Formula
R1-A2-G5:Extended-STE-Noise-Aware-Training

# Round 1, Aspect 2 - Wave 2 (Math Theme A)
R1-A2-M1:OLS-STE-Optimal-Linear-STE
R1-A2-M2:CC-STE-Curvature-Corrected-STE
R1-A2-M3:SSM-TGS-SSM-Temporal-Gradient-Smoothing
R1-A2-M4:GOC-Gradient-Orthogonal-Compensation

# Round 1, Aspect 2 - Wave 2 (Math Theme B)
R1-A2-M5:SNR-STE-Signal-Noise-Ratio-Routing
R1-A2-M6:EWG-STE-Entropy-Weighted-Gradient

# Round 1, Aspect 2 - Wave 2 (Information Theory)
R1-A2-I1:Entropy-Weighted-Gradient-Routing
R1-A2-I2:Mutual-Information-Gradient-Filter
R1-A2-I3:Channel-Capacity-Backward-Pass
R1-A2-I4:KL-Divergence-Gradient-Routing
R1-A2-I5:Fisher-Information-Weighted-Backward
R1-A2-I6:Particle-Gradient-Estimation-Backward

# Round 1, Aspect 2 - Wave 2 (Paradigm Theme A)
R1-A2-P1:In-Memory-Analog-Ternary-Backward
R1-A2-P2:Photonic-Reverse-Mode-Gradient-Free
R1-A2-P3:GPU-Tensor-Core-Inverse-Training
R1-A2-P4:SRAM-Compute-Forward-Backward-Fusion
R1-A2-P5:Persistent-Thread-Backward-Streaming

# Round 1, Aspect 2 - Wave 2 (Paradigm Theme B)
R1-A2-P6:NoProp-Forward-Free-Training
R1-A2-P7:Equilibrium-Propagation-SSMTraining
R1-A2-P8:STDP-Supervised-Ternary-Update
R1-A2-P9:Forward-Forward-Ternary-Layer
```

---

*End of Document*
