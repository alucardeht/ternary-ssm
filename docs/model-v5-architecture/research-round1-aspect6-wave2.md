# Round 1, Aspect 6: Training Speed & Throughput — Wave 2 (Math Paradigm-Breakers)

## Math-PB1: Training Speed & Convergence

### M1: FING — Fisher Information Natural Gradient for Ternary
- **Description:** Model each ternary weight as point on 2-simplex Δ₂ = {(p₋₁, p₀, p₊₁)}. Use natural gradient with Fisher metric to correct STE bias. ΔS_i = -η·g_i/I(S_i).
- **Key insight:** Fisher matrix for ternary is a SCALAR per weight (because shadow weight is 1D). Natural gradient is just per-element division — essentially FREE. Near threshold: smaller, more precise steps (prevent overshooting). Far from threshold: larger steps (accelerate plateau traversal).
- **Speedup:** 1.5-2.5× faster convergence. Per-step cost increase <1%.
- **Why ternary-specific:** 2-simplex is unique to 3-state. Binary gives 1-simplex (Bernoulli), 4-bit gives 15-simplex. Only ternary has scalar Fisher (no matrix inversion needed).
- **Feasibility:** HIGH

### M2: TSCS — Ternary SSM via Segmented Cumulative Sums
- **Description:** For diagonal ternary A, the recurrence h_t = a_i·h_{t-1} + u_t decomposes into 3 regimes: a_i=+1 (prefix sum), a_i=-1 (alternating prefix sum), a_i=0 (memoryless). Each is embarrassingly parallel via torch.cumsum.
- **Key insight:** a_i^n ∈ {-1,0,+1} for ALL n. The three eigenvalues create closed-form power sequences (0^k, 1^k, (-1)^k). This collapses the sequential scan into parallel prefix sums. Backward is reverse prefix sum (also parallel).
- **Speedup:** 3-8× on SSM scan portion. If scan is 40% of training time → 1.8-3.5× overall.
- **Why ternary-specific:** Continuous A requires O(log N) parallel scan with FP32 matrix multiply. ONLY ternary A reduces to cumsums because only {-1,0,+1} eigenvalues give trivially parallelizable power sequences.
- **Feasibility:** HIGH — uses existing torch.cumsum, no custom kernels needed
- **NOTE:** Both math-PBs independently derived this (M2 here = M9/TPC from PB2). Strong convergence signal.

### M3: STCD — Stochastic Ternary Coordinate Descent (Shadow-Free)
- **Description:** Eliminate shadow weights entirely. Use INT8 momentum counters c_i. Each step: c_i += sign(g_i). When |c_i| > K: flip weight, reset counter. Top-K selection for best flips per step.
- **Key insight:** STE converges to continuous relaxation minimum (may not be discrete optimum). STCD converges directly to discrete local minimum. Eliminates STE bias entirely.
- **Speedup:** 2-2.5× total (no STE bias + 75% memory savings enabling larger batch)
- **Why ternary-specific:** Binary has 1 transition per weight. Higher-bit has many candidates (expensive scoring). Ternary has exactly 2 candidates per weight → O(n) scoring.
- **Feasibility:** MEDIUM (discrete local minima risk)

### M4: TCBS — Ternary Critical Batch Scheduling
- **Description:** Optimal batch size = one that maximizes CORRECT threshold crossings per token. Dynamic: B*(t) = B_base / max(ε, fraction_active(t)).
- **Key insight:** "Useful computation" in ternary training = weight threshold crossings. Critical batch size theory reformulated for discrete flips instead of continuous gradient SNR.
- **Speedup:** 1.3-1.8× convergence. Zero computational overhead (periodic O(n) monitoring).
- **Why ternary-specific:** Threshold distance concept only exists for quantized networks. Continuous networks have no thresholds.
- **Feasibility:** HIGH

### M5: IBTP — Integer Backward Training Pipeline
- **Description:** Entire forward+backward in INT8/INT16. Gradient precision analysis: for ternary, only threshold-crossing DIRECTION matters → 4 bits sufficient. Accumulate in INT32, update shadow in INT16 (precision 6e-5).
- **Key insight:** Ternary needs LESS gradient precision than continuous (only threshold crossings matter). INT8 gradients sufficient. INT16 shadow weights (50% memory savings). INT8 tensor cores: 2-4× faster than FP32.
- **Speedup:** 1.3-1.8× total tok/s (backward is 2/3 of training).
- **Why ternary-specific:** For continuous weights, gradient precision directly affects quality. For ternary, precision only needs to resolve threshold crossings — fundamentally lower requirement that DECREASES as training progresses.
- **Feasibility:** HIGH

### M6: FPDV — Flip Potential Data Valuation
- **Description:** Score each training token by its "flip potential" — how many near-threshold weights it activates. Use importance sampling to oversample high-value tokens.
- **Key insight:** FP_proxy(x) = ||M·x||² where M is sparse binary mask of near-threshold weights. Ternary creates BINARY criterion: token either flips a weight or it doesn't. 2-3× variance reduction for useful gradient components.
- **Speedup:** 1.15-1.6× data efficiency (fewer steps to target loss)
- **Why ternary-specific:** Continuous models: ALL tokens provide useful gradient info. Ternary: tokens not causing flips provide ZERO ternary-domain information.
- **Feasibility:** MEDIUM

### M7: GRCF — Gradient Recycling via Flip Coherence
- **Description:** Track gradient direction consistency per weight: c_i = clip(c_i + sign(g_i), -127, +127). Modulate learning rate: η_i = η · φ(c_i) where φ maps coherence to [α_min, α_max].
- **Key insight:** Weights about to flip have COHERENT gradients (consistently pointing toward threshold). Oscillating weights have incoherent gradients. Amplify coherent, dampen incoherent. Replaces FP32 momentum with 1-byte INT8 tracker.
- **Speedup:** 1.3-1.7× convergence. Zero per-step overhead.
- **Why ternary-specific:** Continuous training needs gradient magnitude. Ternary only needs gradient DIRECTION consistency (flip or not). Coherence tracker captures exactly the ternary-relevant signal.
- **Feasibility:** HIGH

### M8: WAHD — Walsh-Hadamard Sublinear Gradient Estimation
- **Description:** Expand loss L(W) in discrete Fourier basis of (Z/3Z)^n. If spectrum concentrated, estimate gradient from O(n·log²n) forward passes via compressed sensing. No backward pass needed.
- **Key insight:** Gradient estimation as sparse recovery problem on the ternary group. All computation uses ternary forward passes (integer-only). Impractical for full gradient (45× slower) but viable for sparse near-threshold gradient.
- **Speedup:** Break-even to 1.5× for sparse gradient of near-threshold weights. Eliminates backward pass entirely.
- **Why ternary-specific:** Z/3Z group structure is unique to ternary. 3-point DFT has clean cube-root-of-unity structure.
- **Feasibility:** SPECULATIVE (moonshot — profound for integer-only hardware)

---

## Math-PB2: Throughput & Computational Efficiency

### M9: TPC — Ternary Prefix-Sum Collapse (= M2 from PB1)
- **DUPLICATE of M2 above.** Both PBs independently derived the same formulation. Strong convergence signal.
- Prefix sum for a=+1, alternating prefix sum for a=-1, identity for a=0.
- Theoretical 186× parallelism improvement (log₂(2048) vs 2048 sequential steps).
- See M2 for full details.

### M10: FDTC — Fourier-Domain Ternary Convolution
- **Description:** Compute SSM recurrence as frequency-domain convolution via FFT. Ternary A gives analytically known transfer function H_i(z) = 1/(z − a_i). Only THREE distinct filters → precompute and batch.
- **Key insight:** Frequency-domain dual of M2's time-domain approach. FFT path: O(T log T) work, O(log T) depth. May be faster than prefix sum for T > 2048 due to cuFFT optimization and better memory access patterns.
- **Speedup:** 2-5× overall (8-20× on scan portion).
- **Why ternary-specific:** Only 3 filter types needed (vs continuous spectrum for FP A). Precomputed filters, maximum batch efficiency.
- **Feasibility:** HIGH — leverages cuFFT (extremely optimized)

### M11: FPIT — Full Pipeline Integer Training (= M5 extended)
- **Description:** Complete INT8/INT16 pipeline analysis. Forward: ternary matvec produces INT32. Backward: INT8 gradients accumulated in INT32. Shadow update: INT16 with stochastic rounding.
- **Key insight:** Weighted throughput analysis: 0.4×4 + 0.1×2 + 0.3×4 + 0.1×2 + 0.08×4 + 0.02×1 = 3.54× average speedup from INT vs FP32.
- **Speedup:** ~3.5× from integer arithmetic. INT8 tensor cores 2-4× faster than FP32.
- **Overlaps with M5** — merged here with more detailed throughput analysis.
- **Feasibility:** HIGH

### M12: TMA — Ternary Momentum Accumulator
- **Description:** Replace FP32 shadow weights + Adam with INT8 counters. c_ij += sign(grad). When |c_ij| ≥ K: flip weight, reset counter. K controls effective LR (K large = low LR).
- **Key insight:** Information content of useful gradient for ternary = log₂(3) = 1.585 bits. FP32 carries 32 bits → 20× more than needed. TMA eliminates waste. Equivalent to signSGD with majority vote (proven convergence).
- **Speedup:** 1.5-2× (4× memory reduction + 8× faster weight update)
- **Why ternary-specific:** Continuous needs gradient magnitude (step size). Ternary needs DIRECTION only (flip or not).
- **Feasibility:** MEDIUM (convergence quality vs FP32 shadow needs empirical validation)

### M13: SLP — Speculative Layer Pipeline with Ternary Correction
- **Description:** Overlap forward passes of consecutive layers by starting next layer with estimated boundary state, then apply cheap ternary correction. Correction for ternary: a_i=0 → free (no dependency), a_i=+1 → add constant, a_i=-1 → alternating constant.
- **Key insight:** Linearity of ternary SSM: h_t = h_t^{zero-init} + a_i^{t-t_boundary} · Δ_boundary. For ternary a_i, correction is trivial (add, negate, or nothing).
- **Speedup:** 2-4× from improved GPU occupancy (all L layers active simultaneously).
- **Why ternary-specific:** Continuous A requires expensive exponential a^n for correction. Ternary: a^n ∈ {-1,0,+1} → correction is free.
- **Feasibility:** MEDIUM (complex scheduling)

### M14: SGFM — Sparse Gradient via Flip-Margin Detection
- **Description:** Only compute weight gradients for the ~5% of weights near a quantization threshold. Non-critical weights (95%) provably cannot flip for multiple steps → skip entirely. Full gradient every R=10 steps as safety.
- **Key insight:** In well-trained ternary, ~95% of shadow weights are far from thresholds → gradient computation is WASTED. Savings: 0.9 × 19% of step = ~17% speedup standalone. Compounds multiplicatively with INT pipeline.
- **Speedup:** ~1.2× standalone. Excellent compounding value.
- **Why ternary-specific:** Continuous weights: ALL need gradients (no stable regions). Ternary: discrete nature creates natural stable regions far from thresholds.
- **Feasibility:** HIGH

### M15: THPT — Ternary Hebbian Perturbation Training (No Backprop)
- **Description:** Propagate error signal backward through ternary layers using ternary matvecs (as cheap as forward). Error = sign(prediction − target). Signal_ij = −sign(e_l[i]) · sign(mean(x_l[j])). Feed into TMA counter.
- **Key insight:** Error propagation through ternary is just sign-flips and zeros — NO FP32 needed. Total cost: ~2× forward pass vs 4-6× for standard (forward + backward). Direct feedback alignment with ACTUAL weights (not random).
- **Speedup:** 2-3× total step time. Combined with M2 + M11: entire training is integer-only with O(log T) depth.
- **Why ternary-specific:** IMPOSSIBLE for continuous weights (error propagation requires FP32 precision). Ternary: propagation is exact via sign-flips and zeros.
- **Feasibility:** SPECULATIVE (sign-only propagation may degrade in deep networks)

### M16: MMSIOR — Spectral Initialization with Online Refinement
- **Description:** Initialize ternary SSM via spectral decomposition of input-output moment statistics (closed-form). B_hat = R_uh · R_xx⁻¹. Optimal A from autocorrelation at lag 1: r_i(1) ≈ +1 → a_i=+1, ≈ −1 → a_i=−1, ≈ 0 → a_i=0.
- **Key insight:** For ternary, spectral estimate only needs to get SIGN correct (which of 3 values). Much easier than precise continuous estimation. Autocorrelation-based A estimation is EXACT for ternary.
- **Speedup:** ~2.5× total training time (5 min init + 3 refinement epochs vs 10 standard epochs)
- **Why ternary-specific:** Aggressive quantization means spectral estimate only needs coarse accuracy. A estimation is trivial (threshold r(1) into 3 bins).
- **Feasibility:** SPECULATIVE (greedy layer-wise estimation for deep networks untested)

---

## Key Convergences

### Independently Derived by Both PBs (Strong Signal)
1. **Prefix-Sum Collapse** (M2 = M9): SSM scan → parallel cumsums for ternary A
2. **Integer Pipeline** (M5 ≈ M11): Entire training in INT8/INT16
3. **Shadow-Free Training** (M3 ≈ M12): Replace FP32 shadow + Adam with INT8 counters

### Composable Stack (Multiplicative Improvements)
| Component | Speedup | Formulation |
|-----------|---------|------------|
| Parallel scan | 4-8× | M2/M9 (prefix sum) or M10 (FFT) |
| Integer pipeline | 3.5× | M5/M11 |
| Layer pipeline | 2-4× | M13 |
| Sparse gradient | 1.2× | M14 |
| Combined theoretical | **33-134×** | All above |
| Practical (50-70% realization) | **15-90×** | Conservative estimate |

At current ~10K tok/s at 50M: **150K-900K tok/s** theoretically achievable.

### Synergy Pairs
- **FING + GRCF:** Fisher direction + coherence step size = optimal ternary optimizer (2 bytes/weight vs 8 for Adam)
- **TSCS + FPIT:** Parallel scan + integer pipeline = entire training integer-only with O(log T) depth
- **TCBS + FPDV:** Optimal batch SIZE + optimal batch CONTENT
- **THPT + MMSIOR:** Spectral init + Hebbian refinement = complete gradient-free training pipeline

### Critical Caveat: Selective vs Fixed A
If A is input-dependent (selective, as in Mamba): M2/M9 prefix-sum collapse does NOT directly apply. However:
- Ternary scan operator still benefits from monoid property (products stay ternary)
- Associative scan with 2-bit operators still much faster than FP32
- If A is fixed (learned weight), ALL formulations apply with full force
