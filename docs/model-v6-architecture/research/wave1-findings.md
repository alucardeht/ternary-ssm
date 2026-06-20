# v6 Wave 1 Findings - Training Speed

**Date:** 2026-03-02
**Focus:** Training speed optimization across 6 aspects

---

## Aspect 1: Energy-Based Training

### Key Findings

**IMPLEMENT: Margin-Based Sparse Backward + Temporal Gradient Recycling (M1+M4)**

- **Principle:** For ternary optimization, gradient importance is proportional to proximity to quantization boundary. Weights far from threshold cannot change ternary value regardless of gradient magnitude — computing their precise gradient is mathematically wasted work.
- **Mechanism:**
  - Compute margin m(i,j) = min distance to nearest quantization boundary
  - Critical set C = {weights where m < delta} — typically 5-10% of all weights
  - For weights in C: compute full gradient (precise outer product)
  - For weights NOT in C: use EMA of past gradients (cheap, O(1) per weight)
  - Full refresh every K steps to prevent gradient drift
- **Expected impact:** 5-8x speedup on weight gradient computation. Weight gradient is 40-60% of backward pass, so total backward speedup ~2-4x.
- **Dependencies:** None. Works with current Adam+STE pipeline.

**IMPLEMENT: Fused Backward-Update Operator (M5)**

- **Principle:** Standard pipeline: (1) backward writes gradients → (2) optimizer reads grads + states → (3) quantize reads weights = 3 memory passes. Fusing into single pass eliminates redundant memory traffic.
- **Mechanism:** Custom torch.autograd.Function combining backward gradient computation with optimizer state update and ternary quantization. Single pass: read(upstream_grad, activation, optimizer_state, shadow_weight) → compute gradient, update optimizer state, re-quantize.
- **Expected impact:** 15-28% total training speedup. Even partial fusion (backward+quantize only) gives 8-12%.
- **Dependencies:** None for partial fusion. Full benefit requires integer optimizer (TMC/AGA).

**EXPLORE: Direct Ternary Flip Search (WalkSAT-style)**

- **Question:** Does stochastic local search in {-1,0,+1}^N converge for language modeling at 6M scale?
- **Potential impact:** REVOLUTIONARY — zero optimizer state (48x memory reduction at 7B), natural discrete optimization.
- **Risk:** HIGH — convergence for language models completely unproven.

---

## Aspect 2: Ternary Backward Pass

### Key Findings

**IMPLEMENT: Ternary-Aware Activation Checkpointing (M7)**

- **Principle:** Ternary layers compute via add/sub (~100x cheaper than matmul). Recomputing their outputs is nearly free. NEVER checkpoint ternary layer activations — only checkpoint expensive layers (SWA attention, non-ternary projections, SSM state boundaries).
- **Expected impact:** 80-85% activation memory reduction. At 7B seq_len=2048: ~800 MB activations reduced to ~120-160 MB.
- **Dependencies:** Uses existing PyTorch checkpoint utilities. Existing gradient_checkpointing flag needs to be made ternary-aware.

**IMPLEMENT: Fused Gradient Cascade — Zero Gradient Buffer (M4)**

- **Principle:** For accumulator-style optimizers (AGA, SPRT), fuse gradient computation directly into accumulator update — never allocate the full gradient tensor.
- **Expected impact:** Eliminates gradient buffer memory (4 bytes/param). At 7B: saves ~28 GB. Combined with AGA: 5.3x total system memory reduction.
- **Dependencies:** Requires an accumulator-style optimizer (AGA already implemented).

**IMPLEMENT: Quamba-SE — Soft-Edge SSM Activation Quantization (F3)**

- **Principle:** SSM activations can be quantized to sub-4-bit during training if quantization boundaries use differentiable sigmoid modulation (soft-edge) instead of hard rounding.
- **Expected impact:** 75% activation memory reduction for SSM layers (FP16 to 2-bit). <1% perplexity degradation.
- **Dependencies:** None. Complementary with M7 — M7 decides WHAT to store, F3 compresses what remains.

**IMPLEMENT: INSTANT — Low-Rank Gradient and Activation Compression (F6)**

- **Principle:** Project gradients and activations into a low-rank subspace during backpropagation. Rank-r projection captures top-r singular directions which dominate gradient information.
- **Expected impact:** 32x activation memory reduction, 15x backward FLOPs reduction.
- **Dependencies:** Independent of weight quantization. Can combine with F3 for maximum coverage.

---

## Aspect 3: Progressive Freezing

### Key Findings

**IMPLEMENT: Information-Theoretic Adaptive Freezing (M3)**

- **Principle:** Each weight accumulates Bayesian evidence for ternary assignment via confidence accumulator: c_i^{t+1} = c_i^t + sign(g_i) * |g_i| / (v_i + epsilon). Freeze when |c_i| > C_thresh = sqrt(2 * log(3/alpha)).
- **Why it applies:** Many weights settle to final assignment early but Adam+STE continues computing full gradients. Freezing high-confidence weights reduces effective dimension progressively.
- **Expected impact:** 1.5-2x convergence speedup if 50% of weights freeze by mid-training.
- **Dependencies:** None. Synergizes with M1 (SNR-derived schedule).

**IMPLEMENT: Optimal LR from Ternary Decision Theory (M1)**

- **Principle:** In ternary training, LR controls threshold crossing rate, not weight magnitude. From SPRT analysis: optimal schedule is eta* ~ t^{-1/2} — SLOWER decay than cosine.
- **Why it applies:** Cosine was designed for continuous optimization. In ternary, loss changes ONLY when weights cross thresholds. Cosine decays too fast in late training — ~40% of training has LR too small.
- **Expected impact:** 1.3-1.8x convergence speedup. Zero overhead — purely a schedule formula change.
- **Dependencies:** None. Can combine with F6 (AdaGC provides stability at higher effective LR).

**IMPLEMENT: AdaGC — Adaptive Per-Tensor Gradient Clipping (F6 from Aspect 3)**

- **Principle:** EMA of gradient norms provides adaptive clip threshold per tensor. Adapts to each tensor's gradient scale without manual tuning.
- **Why it applies:** STE creates gradient discontinuities at ternary thresholds. These cause gradient spikes that destabilize training.
- **Expected impact:** Eliminate loss spikes, +1-2.5% downstream accuracy, 4.5% faster wall-clock convergence.
- **Dependencies:** None. Orthogonal to all other techniques.

---

## Aspect 4: Data Pipeline

### Key Findings

**IMPLEMENT: Ternary Commitment Monitor + Adaptive Training (M3+M8)**

- **Principle:** Ternary training is a sequence of phase transitions. Shadow weights progressively commit to {-1,0,+1}. Ternary entropy H_T(t) = -sum_i sum_v p_i^v(t) * log(p_i^v(t)) monotonically decreases. When H_T plateaus, further training yields diminishing returns.
- **Mechanism:**
  - Every K steps (K=50), compute H_T from shadow weights vs thresholds
  - Track dH_T/dt (rate of commitment)
  - If dH_T/dt < epsilon for M consecutive checks: reduce LR by factor or early-stop
- **Expected impact:** 30-50% training time savings from: (1) early stopping when H_T plateaus, (2) LR scheduling during rapid commitment phases, (3) dynamic batch size reduction as data budget shrinks.
- **Dependencies:** None (uses existing shadow weights and thresholds).

**EXPLORE: TFIW — Ternary Fisher Information Weighting**

- **Question:** Does ternary-specific token weighting (based on gradient magnitude at near-threshold weights) outperform general excess-loss weighting (Rho-1)?
- **Challenge:** Per-token gradient at specific weights requires per-sample gradient computation or approximation.

**EXPLORE: HTC — Hebbian Ternary Construction**

- **Question:** Does correlation-based ternary initialization reduce the number of STE training steps needed?
- **Potential impact:** 5x training reduction (one forward pass computes correlations, assign ternary from sign(C)).

---

## Aspect 5: Sign-Only Optimizer

### Key Findings

**IMPLEMENT: AGS — Asymmetric Gradient Sparsification**

- **Principle:** Only ~0.24% of weights are near a quantization threshold at any step. Gradients for far-from-threshold weights have NO immediate effect on ternary output.
- **Mechanism:** During GPU-to-CPU transfer, send only: (1) gradients for weights within threshold_margin of a ternary boundary, PLUS (2) top-K% gradients by magnitude. Periodic full sync every K_full steps.
- **Expected impact:** 5-10x reduction in GPU-to-CPU gradient transfer bandwidth. 10-30% training speedup at scale with CPU offload active.
- **Dependencies:** CPU offload being active.

**EXPLORE: TSCS Cluster (4 findings requiring ternary A matrix)**

- **TSCS (Ternary State-space Coefficients):** Makes A in {-1,0,+1} instead of continuous. Unlocks:
  - M7: Eigendecomposition — O(T) scan becomes O(log T) parallel
  - M3: TAPS — integer-only binary prefix operations
  - M5: EDLPC — INT4 activation checkpoints (4x memory savings)
  - P1: RTR — reversible SSM (eliminates activation checkpointing entirely)
- **Recommendation:** Run TSCS quality experiment first (100-step smoke test, then 1000-step comparison).

**EXPLORE: Sign-Magnitude Gate replacing SiLU**

- **Principle:** gate(x,y) = sign(x) * y * min(|x|, C) / C. Only shifts + conditional negate.
- **Potential impact:** Enables fully-integer FFN path.

---

## Aspect 6: Hybrid Precision

### Key Findings

**IMPLEMENT: Optimal Threshold Grid (M13)**

- **Principle:** For Gaussian-distributed shadow weights w ~ N(0,sigma^2), Lloyd-Max optimal ternary quantization grid is {-1.224*sigma, 0, +1.224*sigma}. Current threshold_coeff=0.7 (heuristic). Optimal is 0.767.
- **What to change:** One-line: threshold_coeff default from 0.7 to 0.767.
- **Expected impact:** 1-2% MSE reduction in quantization error per layer, compounding across L=8 layers. Zero risk.

**IMPLEMENT: Two-Stage LR + Weight Decay Schedule (F3)**

- **Principle:** (1) LR drop at ~50% of training causes sharp loss decrease (breaks out of local basins). (2) Weight decay cosine to zero at ~67% of training accelerates convergence.
- **Mechanism:**
  - Phase 1 (steps 0-50%): warmup then constant high LR
  - Phase 2 (steps 50%-100%): drop to 0.1x LR, cosine decay to 0
  - Weight decay: start at configured value, cosine to 0 over steps 67%-100%
- **Expected impact:** 5-15% faster convergence to same loss.

**IMPLEMENT: Oscillation Dead Zone — Hysteresis for STE (M14)**

- **Principle:** Weights near thresholds oscillate when gradient signal is ambiguous. Hysteresis width delta* = k * lr * sigma_g prevents flipping when gradient evidence is insufficient.
- **Expected impact:** 3-8% faster convergence.

**IMPLEMENT: Weight Entropy Diagnostic (M7)**

- **Principle:** Shannon entropy H_w = -sum(p_k * log2(p_k)) for k in {-1,0,+1} measures information utilization. Entropy < 0.8 * H_max indicates capacity loss.
- **Expected impact:** Early detection of dead layers. Prevents wasted training (catch in 100 steps vs discover at 1000).

**IMPLEMENT: Commitment Rate Diagnostic (M18)**

- **Principle:** Uncommitted fraction N_unc(t) = count of weights within c_thresh * (lr * sigma_g) of threshold / total weights. Should decrease monotonically.
- **Expected impact:** Early warning for training pathology.

**IMPLEMENT: BBO — Ballot Box Optimizer (P5)**

- **Principle:** INT8 counters (-128 to +127) provide sufficient precision for ternary flip decisions. Each gradient step votes +1 or -1 based on sign; when votes exceed confidence threshold K, weight flips.
- **Expected impact:** 5.3x memory reduction on ternary params (12 bytes -> 1 byte per weight).
- **Risk:** Convergence must be validated against Adam.

---

### Wave 2 Derived

- **GAST-Q: Gradient-Aware Structured Ternary Quantization**
  - Formula: threshold per gradient percentile
  - Speedup: 15-25% faster convergence
  - Derivation: From M1 (margin-based sparse backward), the principle that gradient importance is proportional to proximity to quantization boundary extends naturally to threshold adaptation. Instead of fixed thresholds, compute per-tensor gradient distribution and set thresholds at gradient percentiles {15%, 50%, 85%}. This aligns quantization boundaries with actual gradient flow patterns.
  - Mechanism: During forward pass, compute gradient percentile statistics. During backward, use these to bias threshold placement toward high-gradient-variance regions.

- **TREC: Ternary Recycling with Confidence Weighting**
  - Formula: EMA_weight = alpha * gradient + (1-alpha) * past_gradient
  - Speedup: 3-5x backward speedup for sparse gradients
  - Derivation: Extends M4 (temporal gradient recycling) by adding confidence weighting based on margin. Weights closer to boundaries get higher weight on current gradient; stable weights use heavier EMA averaging.

- **FUSE-2: Fused Energy-Quantization Operator**
  - Formula: single_pass(output, grad, state) -> (updated_output, updated_state)
  - Speedup: 20-35% training speedup
  - Derivation: Extends M5 (fused backward-update) by adding energy-based importance scoring. The fused operator now routes computational effort based on energy (|w - threshold| * |g|), skipping low-energy regions entirely.

---

## Aspect 2: Ternary Backward Pass

### Key Findings

**IMPLEMENT: Ternary-Aware Activation Checkpointing (M7)**

- **Principle:** Ternary layers compute via add/sub (~100x cheaper than matmul). Recomputing their outputs is nearly free. NEVER checkpoint ternary layer activations — only checkpoint expensive layers (SWA attention, non-ternary projections, SSM state boundaries).
- **Expected impact:** 80-85% activation memory reduction. At 7B seq_len=2048: ~800 MB activations reduced to ~120-160 MB.
- **Dependencies:** Uses existing PyTorch checkpoint utilities. Existing gradient_checkpointing flag needs to be made ternary-aware.

**IMPLEMENT: Fused Gradient Cascade — Zero Gradient Buffer (M4)**

- **Principle:** For accumulator-style optimizers (AGA, SPRT), fuse gradient computation directly into accumulator update — never allocate the full gradient tensor.
- **Expected impact:** Eliminates gradient buffer memory (4 bytes/param). At 7B: saves ~28 GB. Combined with AGA: 5.3x total system memory reduction.
- **Dependencies:** Requires an accumulator-style optimizer (AGA already implemented).

**IMPLEMENT: Quamba-SE — Soft-Edge SSM Activation Quantization (F3)**

- **Principle:** SSM activations can be quantized to sub-4-bit during training if quantization boundaries use differentiable sigmoid modulation (soft-edge) instead of hard rounding.
- **Expected impact:** 75% activation memory reduction for SSM layers (FP16 to 2-bit). <1% perplexity degradation.
- **Dependencies:** None. Complementary with M7 — M7 decides WHAT to store, F3 compresses what remains.

**IMPLEMENT: INSTANT — Low-Rank Gradient and Activation Compression (F6)**

- **Principle:** Project gradients and activations into a low-rank subspace during backpropagation. Rank-r projection captures top-r singular directions which dominate gradient information.
- **Expected impact:** 32x activation memory reduction, 15x backward FLOPs reduction.
- **Dependencies:** Independent of weight quantization. Can combine with F3 for maximum coverage.

### Wave 2 Derived

- **STAQ: Structured Ternary Activation Quantization**
  - Formula: quantize(x) = round(x / s) * s where s = percentile(|x|, 95%)
  - Speedup: 75-85% activation memory reduction
  - Derivation: Extends F3 (soft-edge SSM quantization) by using per-channel scale based on gradient-aware percentile. Instead of uniform scaling, compute scale from gradient flow through each channel, prioritizing channels with active gradients.

- **CASCADE-2: Multi-Resolution Gradient Cascade**
  - Formula: gradient = coarse_grad + fine_grad * mask(coarse_magnitude > threshold)
  - Speedup: 2-3x backward speedup
  - Derivation: Combines M7 checkpointing with F6 low-rank compression. Use coarse (1-bit) gradient representation for most layers, refine only in regions where coarse gradient exceeds entropy threshold. Natural extension of ternary principles to activation computation.

- **ZIP: Zero-Initialization Precision Scaling**
  - Formula: scale = sqrt(mean(forward_output^2)) / sqrt(target_precision)
  - Speedup: 40-50% memory bandwidth reduction
  - Derivation: From M7 insight that ternary computation is cheap. Pre-scale activations to optimal precision range before storage, recover on read. Eliminates storage of full-precision activations while maintaining gradient fidelity.

---

## Aspect 3: Progressive Freezing

### Key Findings

**IMPLEMENT: Information-Theoretic Adaptive Freezing (M3)**

- **Principle:** Each weight accumulates Bayesian evidence for ternary assignment via confidence accumulator: c_i^{t+1} = c_i^t + sign(g_i) * |g_i| / (v_i + epsilon). Freeze when |c_i| > C_thresh = sqrt(2 * log(3/alpha)).
- **Why it applies:** Many weights settle to final assignment early but Adam+STE continues computing full gradients. Freezing high-confidence weights reduces effective dimension progressively.
- **Expected impact:** 1.5-2x convergence speedup if 50% of weights freeze by mid-training.
- **Dependencies:** None. Synergizes with M1 (SNR-derived schedule).

**IMPLEMENT: Optimal LR from Ternary Decision Theory (M1)**

- **Principle:** In ternary training, LR controls threshold crossing rate, not weight magnitude. From SPRT analysis: optimal schedule is eta* ~ t^{-1/2} — SLOWER decay than cosine.
- **Why it applies:** Cosine was designed for continuous optimization. In ternary, loss changes ONLY when weights cross thresholds. Cosine decays too fast in late training — ~40% of training has LR too small.
- **Expected impact:** 1.3-1.8x convergence speedup. Zero overhead — purely a schedule formula change.
- **Dependencies:** None. Can combine with F6 (AdaGC provides stability at higher effective LR).

**IMPLEMENT: AdaGC — Adaptive Per-Tensor Gradient Clipping (F6 from Aspect 3)**

- **Principle:** EMA of gradient norms provides adaptive clip threshold per tensor. Adapts to each tensor's gradient scale without manual tuning.
- **Why it applies:** STE creates gradient discontinuities at ternary thresholds. These cause gradient spikes that destabilize training.
- **Expected impact:** Eliminate loss spikes, +1-2.5% downstream accuracy, 4.5% faster wall-clock convergence.
- **Dependencies:** None. Orthogonal to all other techniques.

### Wave 2 Derived

- **CONFIDE: Confidence-Based Dynamic Dimensionality**
  - Formula: freeze_mask = (|confidence| > C_thresh) AND (|gradient| < G_thresh * EMA_gradient)
  - Speedup: 1.8-2.5x convergence speedup
  - Derivation: Extends M3 (information-theoretic freezing) by adding gradient stability check. A weight is truly frozen only when BOTH confidence is high AND gradient variance is low (stable). This prevents premature freezing of weights that have high confidence but are still oscillating.

- **SNR-Flow: Signal-to-Noise Guided Gradient Routing**
  - Formula: effective_lr = base_lr * SNR(w) where SNR = mean(|gradient|) / std(gradient)
  - Speedup: 20-35% faster convergence
  - Derivation: From M1 (optimal LR from ternary decision theory). Instead of uniform LR decay, modulate LR per-weight based on signal-to-noise ratio. High-SNR weights receive full LR for rapid threshold crossing; low-SNR weights receive reduced LR to avoid noise-driven oscillation.

- **AdaFreeze: Adaptive Freezing with Momentum-Aware Release**
  - Formula: if |c_i| > C_thresh: freeze; if |new_gradient| > 2 * |frozen_gradient|: unfreeze
  - Speedup: Prevents 5-10% accuracy loss from over-freezing
  - Derivation: Extends M3 with momentum-aware release mechanism. Once frozen, weights stay frozen but are monitored for gradient divergence. If previously-frozen weight experiences gradient burst (changing context), automatically unfreeze. Combines benefits of freezing with plasticity preservation.

---

## Aspect 4: Data Pipeline

### Key Findings

**IMPLEMENT: Ternary Commitment Monitor + Adaptive Training (M3+M8)**

- **Principle:** Ternary training is a sequence of phase transitions. Shadow weights progressively commit to {-1,0,+1}. Ternary entropy H_T(t) = -sum_i sum_v p_i^v(t) * log(p_i^v(t)) monotonically decreases. When H_T plateaus, further training yields diminishing returns.
- **Mechanism:**
  - Every K steps (K=50), compute H_T from shadow weights vs thresholds
  - Track dH_T/dt (rate of commitment)
  - If dH_T/dt < epsilon for M consecutive checks: reduce LR by factor or early-stop
- **Expected impact:** 30-50% training time savings from: (1) early stopping when H_T plateaus, (2) LR scheduling during rapid commitment phases, (3) dynamic batch size reduction as data budget shrinks.
- **Dependencies:** None (uses existing shadow weights and thresholds).

**EXPLORE: TFIW — Ternary Fisher Information Weighting**

- **Question:** Does ternary-specific token weighting (based on gradient magnitude at near-threshold weights) outperform general excess-loss weighting (Rho-1)?
- **Challenge:** Per-token gradient at specific weights requires per-sample gradient computation or approximation.

**EXPLORE: HTC — Hebbian Ternary Construction**

- **Question:** Does correlation-based ternary initialization reduce the number of STE training steps needed?
- **Potential impact:** 5x training reduction (one forward pass computes correlations, assign ternary from sign(C)).

### Wave 2 Derived

- **TERM: Ternary Entropy-Reweighted Minibatching**
  - Formula: sample_weight = H_T(sample_gradients) / mean(H_T(all_gradients))
  - Speedup: 15-25% sample efficiency
  - Derivation: From M8 (commitment monitor). Instead of tracking global entropy, compute per-sample ternary entropy based on how samples affect the ternary commitment distribution. Samples that increase entropy (push toward uniform) are deprioritized; samples that decrease entropy (push toward committed states) are prioritized.

- **DynaSample: Dynamic Sampling with Gradient Uncertainty**
  - Formula: sample_prob = softmax(uncertainty_score / temperature)
  - Speedup: 20-30% convergence speedup
  - Derivation: Extends TFIW concept using Bayesian uncertainty. For each sample, compute variance across K forward passes with dropout. High-variance samples (uncertain) get higher sampling probability. This focuses training on regions where the model is uncertain, maximizing information gain per sample.

- **QuickCommit: Accelerated Commitment via Targeted Sampling**
  - Formula: target_samples = samples where |w_shadow| < threshold_margin
  - Speedup: 2-3x commitment rate
  - Derivation: From HTC (Hebbian ternary construction). Instead of random sampling, actively select samples that affect weights near ternary boundaries. These "boundary samples" are most effective at driving ternary commitment because they directly influence threshold crossings.

---

## Aspect 5: Sign-Only Optimizer

### Key Findings

**IMPLEMENT: AGS — Asymmetric Gradient Sparsification**

- **Principle:** Only ~0.24% of weights are near a quantization threshold at any step. Gradients for far-from-threshold weights have NO immediate effect on ternary output.
- **Mechanism:** During GPU-to-CPU transfer, send only: (1) gradients for weights within threshold_margin of a ternary boundary, PLUS (2) top-K% gradients by magnitude. Periodic full sync every K_full steps.
- **Expected impact:** 5-10x reduction in GPU-to-CPU gradient transfer bandwidth. 10-30% training speedup at scale with CPU offload active.
- **Dependencies:** CPU offload being active.

**EXPLORE: TSCS Cluster (4 findings requiring ternary A matrix)**

- **TSCS (Ternary State-space Coefficients):** Makes A in {-1,0,+1} instead of continuous. Unlocks:
  - M7: Eigendecomposition — O(T) scan becomes O(log T) parallel
  - M3: TAPS — integer-only binary prefix operations
  - M5: EDLPC — INT4 activation checkpoints (4x memory savings)
  - P1: RTR — reversible SSM (eliminates activation checkpointing entirely)
- **Recommendation:** Run TSCS quality experiment first (100-step smoke test, then 1000-step comparison).

**EXPLORE: Sign-Magnitude Gate replacing SiLU**

- **Principle:** gate(x,y) = sign(x) * y * min(|x|, C) / C. Only shifts + conditional negate.
- **Potential impact:** Enables fully-integer FFN path.

### Wave 2 Derived

- **SIGNUM: Sign-Based Numerical Update Mechanism**
  - Formula: update = sign(gradient) * magnitude_ema / (count + 1)
  - Speedup: 3-4x optimizer state reduction
  - Derivation: Extends AGS (asymmetric gradient sparsification) by replacing Adam's momentum with sign-based counters. Instead of storing full momentum vectors, store only sign counts (how many + vs - gradients). Update magnitude from exponential moving average of gradient magnitudes. Dramatically reduces state while preserving directional information.

- **VOTE: Voting-Based Ternary Optimizer**
  - Formula: flip_decision = sum_i(sign(g_i)) > confidence_threshold
  - Speedup: 10-15x memory reduction
  - Derivation: From BBO (ballot box optimizer) in Aspect 6. Uses accumulated vote counts to make ternary flip decisions. When vote sum exceeds threshold, flip weight. This naturally implements the SPRT principle from M1 — flip only when accumulated evidence crosses decision boundary.

- **BitVote: Bitwise Voting for Massively Parallel Threshold Decisions**
  - Formula: flip_vector = (vote_count > threshold) as int8 tensor
  - Speedup: 50-100x decision speedup via vectorization
  - Derivation: Extends VOTE by leveraging GPU parallelism. Instead of per-weight Python loops, compute all flip decisions in a single vectorized operation using bitwise operations. Essential for scaling to 7B+ parameters where per-weight iteration is prohibitively slow.

---

## Aspect 6: Hybrid Precision

### Key Findings

**IMPLEMENT: Optimal Threshold Grid (M13)**

- **Principle:** For Gaussian-distributed shadow weights w ~ N(0,sigma^2), Lloyd-Max optimal ternary quantization grid is {-1.224*sigma, 0, +1.224*sigma}. Current threshold_coeff=0.7 (heuristic). Optimal is 0.767.
- **What to change:** One-line: threshold_coeff default from 0.7 to 0.767.
- **Expected impact:** 1-2% MSE reduction in quantization error per layer, compounding across L=8 layers. Zero risk.

**IMPLEMENT: Two-Stage LR + Weight Decay Schedule (F3)**

- **Principle:** (1) LR drop at ~50% of training causes sharp loss decrease (breaks out of local basins). (2) Weight decay cosine to zero at ~67% of training accelerates convergence.
- **Mechanism:**
  - Phase 1 (steps 0-50%): warmup then constant high LR
  - Phase 2 (steps 50%-100%): drop to 0.1x LR, cosine decay to 0
  - Weight decay: start at configured value, cosine to 0 over steps 67%-100%
- **Expected impact:** 5-15% faster convergence to same loss.

**IMPLEMENT: Oscillation Dead Zone — Hysteresis for STE (M14)**

- **Principle:** Weights near thresholds oscillate when gradient signal is ambiguous. Hysteresis width delta* = k * lr * sigma_g prevents flipping when gradient evidence is insufficient.
- **Expected impact:** 3-8% faster convergence.

**IMPLEMENT: Weight Entropy Diagnostic (M7)**

- **Principle:** Shannon entropy H_w = -sum(p_k * log2(p_k)) for k in {-1,0,+1} measures information utilization. Entropy < 0.8 * H_max indicates capacity loss.
- **Expected impact:** Early detection of dead layers. Prevents wasted training (catch in 100 steps vs discover at 1000).

**IMPLEMENT: Commitment Rate Diagnostic (M18)**

- **Principle:** Uncommitted fraction N_unc(t) = count of weights within c_thresh * (lr * sigma_g) of threshold / total weights. Should decrease monotonically.
- **Expected impact:** Early warning for training pathology.

**IMPLEMENT: BBO — Ballot Box Optimizer (P5)**

- **Principle:** INT8 counters (-128 to +127) provide sufficient precision for ternary flip decisions. Each gradient step votes +1 or -1 based on sign; when votes exceed confidence threshold K, weight flips.
- **Expected impact:** 5.3x memory reduction on ternary params (12 bytes -> 1 byte per weight).
- **Risk:** Convergence must be validated against Adam.

### Wave 2 Derived

- **GRIDLOCK: Adaptive Grid Locking**
  - Formula: locked_threshold = optimal_threshold * (1 - commitment_rate^2)
  - Speedup: 10-15% convergence speedup
  - Derivation: Extends M13 (optimal threshold grid) by making thresholds adaptive to commitment rate. As weights commit to ternary values, lock thresholds in place to prevent oscillation. Unlock only when commitment rate drops. This combines M14 (oscillation dead zone) with adaptive grid positioning.

- **PRECISION: Progressive Precision Annealing**
  - Formula: precision_schedule = {FP32: 0-20%, FP16: 20-50%, INT8: 50-80%, Ternary: 80-100%}
  - Speedup: 25-40% memory reduction
  - Derivation: Extends F3 (two-stage LR) concept to precision. Start training in FP32, progressively anneal to ternary. Each precision transition acts as a "breakout" event (like LR drop) that helps escape local minima. Natural extension of progressive freezing to precision domain.

- **HYSTEER: Hysteresis with Ternary Energy-Aware Release**
  - Formula: release_threshold = base_hysteresis * (1 + energy(w) / max_energy)
  - Speedup: 5-10% reduction in oscillation
  - Derivation: Extends M14 (oscillation dead zone) by making hysteresis width depend on ternary energy (distance to boundary times gradient magnitude). High-energy weights need wider hysteresis to prevent premature flips; low-energy weights can use narrower hysteresis for faster adaptation.

- **AdaptiveBBO: Adaptive Confidence Thresholding for BBO**
  - Formula: confidence_K = K_base * (1 - commitment_rate) + K_min * commitment_rate
  - Speedup: 15-20% faster convergence vs fixed threshold
  - Derivation: Extends BBO by making confidence threshold adaptive. Early training (low commitment): use lower K for rapid flipping. Late training (high commitment): use higher K to prevent oscillation. This adapts the exploration-exploitation tradeoff to training phase.

---

## Wave 3 Paradigm Shifts

### Aspect 1: Optimizer

**IMO — Ising Model Optimization**

- **Principle:** Treat ternary weights as spins in an Ising model. Weight configuration evolves via Gibbs sampling: P(w_i = s) = sigmoid(beta * E_i(s)) where energy E_i(s) = -sum_j(w_j * gradient_ij * s). Temperature schedule beta(t) controls exploration-exploitation tradeoff.
- **Paradigm shift:** Replaces gradient descent with thermodynamic sampling. Instead of minimizing loss directly, samples from Boltzmann distribution over weight configurations.
- **Expected impact:** Natural escape from local minima, parallelizable weight updates, physics-inspired optimization.
- **Derivation:** From thermodynamic principles — gradient descent is zero-temperature limit. Ising model naturally handles discrete {-1,0,+1} state space without STE approximation.

---

### Aspect 2: Backward Pass

**LSE — Latent State Evolution**

- **Principle:** Learn a small model g(forward_state, shadow_weights) -> gradient_correction that predicts backward-pass gradients. Train g during initial steps, then use for subsequent steps.
- **Paradigm shift:** Decouples backward pass from exact gradient computation. Instead of backpropagating through quantization, predict the gradient correction.
- **Expected impact:** Eliminates STE backward bottleneck. 10-50x faster backward once predictor is trained.
- **Derivation:** From observation that STE gradients have structured patterns. The quantization boundary creates predictable gradient "kinks" that can be modeled by a small auxiliary network.

---

### Aspect 3: Freezing

**TSPU — Ternary-State Progressive Unfreezing**

- **Principle:** Bottom-up unfreezing: start from lowest-level features (embeddings, early layers) and progressively unfreeze toward output. Each layer's state-sensitivity (gradient variance / mean) determines when to unfreeze.
- **Paradigm shift:** Reverses traditional progressive freezing. Start fully frozen, unfreeze based on state evolution maturity rather than confidence or loss.
- **Expected impact:** 2-3x convergence speedup, better feature reuse.
- **Derivation:** From neural tangent kernel theory — early layers have more reusable features. State-sensitivity measures whether a layer's representations have stabilized.

---

### Aspect 4: Data Pipeline

**SSCL — Self-Signal Continuous Learning**

- **Principle:** Model generates its own training signals by: (1) predicting future tokens from current hidden state, (2) using prediction confidence as sample weight, (3) creating synthetic examples from high-uncertainty regions.
- **Paradigm shift:** Removes reliance on external labels. Model continuously learns from its own predictions.
- **Expected impact:** 30-50% sample efficiency improvement, continuous adaptation.
- **Derivation:** From self-supervised learning principles — the model knows where it is uncertain. High-uncertainty tokens are most informative for learning.

---

### Aspect 5: Sign Optimizer

**TERA — Ternary Energy-Resonance Annealing**

- **Principle:** Stochastic resonance in ternary weight space: add noise to shadow weights, then threshold. Noise amplitude follows annealing schedule. Energy function E(w) = loss + lambda * |w - target|^2.
- **Paradigm shift:** Uses noise to escape local minima. Unlike Simulated Annealing which cools uniformly, TERA focuses noise on high-energy (high-loss) regions.
- **Expected impact:** Better convergence to global minima, reduced oscillation.
- **Derivation:** From stochastic resonance — optimal noise levels exist where signal-to-noise ratio is maximized. TERA applies this to discrete ternary optimization.

---

### Aspect 6: Hybrid Precision

**NSDQ — Neural State-Dependent Quantization**

- **Principle:** Quantization precision depends on hidden state. High-uncertainty states (high gradient variance) use finer quantization; stable states use coarser.
- **Paradigm shift:** Dynamic precision based on local computation context. Replaces global precision schedule with per-token adaptive precision.
- **Expected impact:** 40-60% memory reduction with <1% quality loss.
- **Derivation:** From adaptive computation — not all tokens need equal precision. NSDQ extends this to weight quantization, allocating precision where it matters most.

---

## Summary

### 5 Most Promising Techniques with Projected Speedup

| Rank | Technique | Aspect | Projected Speedup | Risk |
|------|-----------|--------|-------------------|------|
| 1 | Margin-Sparse Backward + Temporal Recycling | 1 | 2-4x backward | LOW |
| 2 | Two-Stage LR Schedule | 6 | 5-15% convergence | LOW |
| 3 | Information-Theoretic Freezing | 3 | 1.5-2x convergence | LOW |
| 4 | Ternary-Aware Checkpointing | 2 | 80-85% memory reduction | NONE |
| 5 | Commitment Monitor + Adaptive Training | 4 | 30-50% training time | NONE |

### Implementation Priority

1. **M13 (Optimal Threshold)** — TRIVIAL, one line, zero risk
2. **M7+M18 (Diagnostics)** — LOW effort, training visibility
3. **M1 (LR Schedule)** — 5 lines, zero risk, immediate testability
4. **F6 (AdaGC)** — 20 lines, near-zero risk, proven
5. **M1+M4 (Margin-Sparse + Recycling)** — HIGH priority, 2-4x backward speedup
6. **M3 (Freezing)** — 30 lines, low risk, progressive benefit
7. **M5 (Fused Backward-Update)** — MEDIUM effort, 15-28% speedup

### Key Synergies

- M1 (t^{-1/2} LR) + F6 (AdaGC): Higher LR kept productive longer, AdaGC prevents spikes
- M1 + M3: SNR statistics feed both schedule AND freeze decisions
- M7 + F3: M7 decides what to checkpoint, F3 compresses what remains
- M4 + AGA: M4 fuses gradient into AGA update, eliminating gradient buffer

### Target

**Goal:** 10-15x speedup over v5 training pipeline

