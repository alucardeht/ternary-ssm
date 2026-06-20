# R3-A6: Precision Error Analysis

## Context
Research on training speed optimization for ternary SSM. Wave 1 focus: Error analysis and rounding theory for numerical precision in training.

---

### R3-A6-F1: Rounding Error Accumulation Bound
- **Src:** Numerical analysis theory [2024] | **Gain:** Error bound prediction | **V:** PROMISING
- **Principle:** After t steps, rounding error ≤ ε√t where ε is per-step rounding error
- **Application:** For FP16, monitor accumulated error, reset weights periodically if error exceeds threshold

### R3-A6-F2: Stochastic Rounding for Training
- **Src:** Stochastic rounding paper [2024] | **Gain:** Unbiased rounding | **V:** PROMISING
- **Principle:** Round to nearest with probability proportional to distance; E[rounded] = original
- **Application:** Use stochastic rounding for gradient accumulation to prevent bias accumulation

### R3-A6-F3: Floating-Point Error Compensation
- **Src:** Error compensation theory [2024] | **Gain:** Reduced drift | **V:** PROMISING
- **Principle:** Maintain error buffer e = exact_sum - rounded_sum; add e to next computation
- **Application:** For FP16 training, implement Kahan summation for gradient accumulation

### R3-A6-F4: Mixed Precision Error Analysis
- **Src:** Mixed precision training [2024] | **Gain:** Optimal precision selection | **V:** PROMISING
- **Principle:** Use FP32 for accumulation, FP16/BF16 for compute, INT8 for inference
- **Application:** Store master weights in FP32, compute in FP16/BF16 for SSM training

### R3-A6-F5: Block Floating-Point Scaling
- **Src:** Block FP theory [2024] | **Gain:** Better dynamic range | **V:** PROMISING
- **Principle:** Each block has shared exponent, mantissa is per-value; balances range and precision
- **Application:** For SSM hidden states, use block size 32 for A matrix, 16 for activations

### R3-A6-F6: Rounding Noise as Regularization
- **Src:** Noise regularization theory [2024] | **Gain:** Implicit regularization | **V:** SPECULATIVE
- **Principle:** Rounding noise adds stochasticity that can prevent overfitting
- **Application:** For ternary SSM, intentional rounding noise may improve generalization

### R3-A6-F7: Precision-Aware Gradient Clipping
- **Src:** Precision clipping theory [2024] | **Gain:** Prevents overflow | **V:** PROMISING
- **Principle:** Set clip threshold based on precision: FP16 max = 65504, BF16 max = 3.4e38
- **Application:** For FP16 SSM training, clip gradients at 10.0 to prevent overflow

### R3-A6-F8: Error Propagation Analysis for SSM
- **Src:** SSM numerical stability [2024] | **Gain:** SSM-specific bounds | **V:** SPECULATIVE
- **Principle:** SSM state evolution amplifies errors: ||error_t|| ≤ ||A||^t × ||error_0||
- **Application:** For stable SSM, ensure ||A|| < 1 (diagonal A with eigenvalues < 1)

---

## Mathematical Derivations

### M1: Optimal Precision for Gradient Magnitude
- **Derivation:** For gradient magnitude g, need log2(|g|) bits for exponent
- **Result:** BF16 (8-bit exponent) handles gradients up to 10^9; FP16 (5-bit) caps at 10^5
- **Application:** Use BF16 for SSM gradients that can span large dynamic range

### M2: Rounding Error in Ternary Quantization
- **Derivation:** For ternary w ∈ {-1, 0, 1}: error = w - w_fp16; |error| ≤ 0.5
- **Result:** Expected squared error = 1/3 for uniform distribution
- **Application:** For committed weights, error is 0; for uncommitted, accept quantization error

### M3: Numerical Stability Condition for SSM
- **Derivation:** For SSM h_t = Ah_{t-1} + Bx_t, numerical error grows as ||A||^t
- **Result:** Stable SSM requires max(|eigenvalue(A)|) < 1 + O(ε)
- **Application:** Initialize A with eigenvalues in (0.9, 1.0) for stable training

---

## Key Insights
1. Stochastic rounding provides unbiased gradient estimates in low precision
2. Error compensation (Kahan summation) prevents precision drift
3. SSM state evolution requires careful eigenvalue management for stability

## Seed Questions
1. How does ternary commitment affect acceptable precision levels?
2. Can we derive optimal precision schedule from training phase?
