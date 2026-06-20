# Optimizer Research Catalog

Organized reference for optimizer techniques applicable to ternary-weight SSM training. Sorted by expected impact and implementation complexity.

## High-Impact Optimizers (2-5x convergence improvement)

### 1. Muon Optimizer
**Source:** DeepSeek, Nov 2024
**Paper:** "Scalable Training of LLMs with Muon" (to appear, preprint available)
**Key Technique:** Momentum orthogonalized by Newton's method, removes scale adaptivity from Adam

**Mathematical core:**
```
- Newton orthogonalization: M = M - (MM^T - I) / (1 + λ)
- Momentum update with orthogonal constraint
- Adam-like but with orthogonal moment estimates
```

**Expected improvement:** 2-5x convergence speed on Transformer training
**Our case applicability:** Very high — SSM compatible (linearity-preserving)
**Challenges:**
- Novel combination: Muon + STE + ternary weights (no published work)
- Requires orthogonal matrix maintenance (extra compute)
- Interaction with ternary gradient noise unknown

**Implementation complexity:** High (custom orthogonal projection required)
**Key references:**
- DeepSeek Nov 2024 preprint
- Orthogonal group manifold theory (DiffGeo)

**Test plan:**
1. Implement Muon with standard Adam as baseline
2. Train 6M model with Muon + standard STE
3. Compare convergence to AdamW baseline
4. If successful: test Muon + HESTIA (better gradient)

---

### 2. Sophia Optimizer (Hessian-aware)
**Source:** Stanford, May 2023; updated 2024
**Paper:** "Sophia: A Scalable Stochastic Second-Order Optimizer for Language Model Pre-Training"
**Key Technique:** Lightweight second-order — approximate Hessian diagonal via variance of gradients

**Mathematical core:**
```
- h[t] ← (1-β) * h[t-1] + β * g[t]^2  (approx Hessian diagonal)
- θ[t+1] ← θ[t] - α * g[t] / max(h[t], 1)  (divide by Hessian)
- Avoids full Hessian matrix (O(n) not O(n^2))
```

**Expected improvement:** 2x convergence speed on LLM training
**Our case applicability:** Good — Hessian information helps ternary training
**Challenges:**
- Hessian diagonal computation adds 10-15% overhead
- STE gradients are noisy — Hessian estimate less reliable
- Not tested with sub-4-bit quantization

**Implementation complexity:** Medium (variance tracking, clipping)
**Key references:**
- Sophia paper (ICLR 2023)
- Hessian-free methods review (Martens & Grosse 2015)

**Test plan:**
1. Implement Sophia's Hessian diagonal tracking
2. Train 6M with Sophia + STE
3. Measure overhead vs quality improvement
4. Compare to Muon on same 6M

---

### 3. AdEMAMix Optimizer
**Source:** Meta/Inria, Jul 2024
**Paper:** "AdEMAMix: Dual Momentum with Fast Reversion for Robust Optimization"
**Key Technique:** Dual exponential moving averages with different decay rates, enables fast reversion

**Mathematical core:**
```
- Two momentum buffers: m1[t], m2[t] with different β1, β2
- m1[t] ← β1_fast * m1[t-1] + (1-β1_fast) * g[t]
- m2[t] ← β1_slow * m2[t-1] + (1-β1_slow) * g[t]
- Update: θ[t+1] ← θ[t] - α * (w1*m1[t] + w2*m2[t])
```

**Expected improvement:** 1.3-2x convergence speed
**Our case applicability:** Good — independent of quantization
**Challenges:**
- Hyperparameter tuning: β1_fast, β1_slow, w1, w2 (4 new params)
- Optimal values task-dependent
- Limited evidence on ternary quantization

**Implementation complexity:** Low (modify momentum computation)
**Key references:**
- Meta/Inria paper, Jul 2024
- Momentum buffer design (adaptive + fixed decay)

**Test plan:**
1. Implement AdEMAMix in torch optimizer
2. Tune hyperparameters on 6M (grid search β1_fast ∈ {0.1, 0.3, 0.5}, β1_slow ∈ {0.9, 0.95, 0.99})
3. Compare to AdamW baseline
4. If promising: test on 50M

---

### 4. SPAM Optimizer (Spike-Aware Momentum)
**Source:** 2025 (preprint available)
**Paper:** "SPAM: Spike-Aware Adaptive Momentum Optimizer"
**Key Technique:** Momentum reset on gradient spikes, designed for training stability

**Mathematical core:**
```
- Detect spike: if ||g[t]|| > threshold * EMA(||g||)
- Reset momentum: m[t] ← 0 if spike detected
- Otherwise: standard Adam momentum update
- Threshold ≈ 2-3 × exponential moving average
```

**Expected improvement:** Better stability during quantization transitions
**Our case applicability:** Very high — specifically designed for QAT instability
**Challenges:**
- New algorithm (not yet widely adopted)
- Threshold tuning required
- May interact poorly with weight decay

**Implementation complexity:** Low (conditional momentum reset)
**Key references:**
- 2025 preprint (find on arXiv)
- Gradient spike literature (optimization dynamics)

**Test plan:**
1. Implement SPAM on top of AdamW
2. Test on 50M at QAT transition (steps 980-1050)
3. Measure loss spike magnitude and recovery time
4. Compare to standard AdamW recovery

---

## Medium-Impact Optimizers (1.3-2x convergence improvement)

### 5. SOAP Optimizer (Shampoo + Adam)
**Source:** Google DeepMind, Sept 2024
**Paper:** "SOAP: Shampoo Optimizer with Adaptive Preconditioners"
**Key Technique:** Combines Shampoo (second-order preconditioning) with adaptive learning rates

**Mathematical core:**
```
- Shampoo: H[t] ← running covariance estimate of gradients
- Preconditioning: g_precond[t] = H[t]^{-1/2} * g[t]
- Adaptive LR: α[t] = α_base / (β * norm(g[t]) + ε)
```

**Expected improvement:** 1.5-3x convergence on large models
**Our case applicability:** Good — works with any loss function
**Challenges:**
- Shampoo adds 30-50% memory overhead for covariance matrices
- Limited evidence at small scale (6M)
- Preconditioning computation expensive

**Implementation complexity:** Medium-High (matrix inversion, preconditioning)
**Key references:**
- SOAP paper (DeepMind, Sept 2024)
- Shampoo history (K-FAC, natural gradient methods)

**Test plan:**
1. Implement Shampoo preconditioner (can use approximations)
2. Test on 6M (memory cost significant)
3. If VRAM permits: test on 50M
4. Benchmark vs Muon (both second-order)

---

### 6. Schedule-Free AdamW
**Source:** Meta/Apple, Mar 2024
**Paper:** "Schedule-Free Optimizers"
**Key Technique:** Eliminates learning rate schedule entirely, uses sequence averaging

**Mathematical core:**
```
- No schedule: fixed LR throughout training
- Sequence averaging: θ_avg[t] = (1/(t+1)) * Σ θ[i]
- Returns θ_avg instead of latest θ
- Convergence guaranteed without decay schedule
```

**Expected improvement:** Removes LR tuning burden, often 1-1.5x better final loss
**Our case applicability:** Very high — simplifies hyperparameter tuning
**Challenges:**
- Returns averaged parameters (not final checkpoint)
- Requires averaging buffer (marginal memory)
- No evidence with ternary quantization

**Implementation complexity:** Low (averaging instead of schedules)
**Key references:**
- Schedule-Free paper (Meta/Apple, Mar 2024)
- Averaging theory (Polyak averaging, Nesterov)

**Test plan:**
1. Implement Schedule-Free AdamW (trivial modification)
2. Train 6M with fixed LR=1e-3 (no WSD schedule)
3. Compare to WSD+AdamW (same compute budget)
4. If better: use for all downstream experiments

---

### 7. Prodigy Optimizer (Auto-tuning)
**Source:** Carnegie Mellon, Dec 2023
**Paper:** "Prodigy: An Expeditious Optimizer for Large Language Models"
**Key Technique:** Automatically estimates optimal learning rate per parameter layer

**Mathematical core:**
```
- Track D[t] = (1-β) * D[t-1] + β * ||g[t]||^2 / ||θ[t]||^2
- Per-param LR: α[t] = ρ / sqrt(D[t])
- No manual tuning required
```

**Expected improvement:** Automatically optimal LR (1-1.5x time savings from tuning)
**Our case applicability:** Medium — layer-wise adaptation useful for differential LR
**Challenges:**
- New algorithm (limited adoption)
- Interaction with ternary gradients unclear
- Slighly different convergence dynamics than AdamW

**Implementation complexity:** Medium (per-param adaptation tracking)
**Key references:**
- Prodigy paper (CMU, Dec 2023)
- Adaptive learning rate theory

**Test plan:**
1. Implement Prodigy (available in torchopt)
2. Train 6M without manual LR tuning
3. Compare convergence to AdamW with careful tuning
4. Benefit mainly if hyperparameter tuning is expensive (not priority)

---

## Learning Rate Schedules (10-20% improvement)

### 8. WSD Schedule (Warmup-Stable-Decay)
**Source:** Alibaba, Oct 2024
**Paper:** "WSD: Warmup-Stable-Decay Learning Rate Schedule"
**Key Technique:** Three-phase schedule replacing cosine annealing

**Mathematical core:**
```
- Phase 1 (Warmup): LR ramps 0 → LR_max over 10% of training
- Phase 2 (Stable): LR constant at LR_max over 60% of training
- Phase 3 (Decay): LR linearly decreases LR_max → 0 over 30% of training
```

**Expected improvement:** 5-15% better final loss than cosine annealing
**Our case applicability:** Very high — direct replacement
**Challenges:**
- New empirical finding (not well-understood theoretically)
- Phase durations (10/60/30) may need tuning for our case
- Less standard than cosine

**Implementation complexity:** Trivial (three linear segments)
**Key references:**
- Alibaba paper, Oct 2024
- Warmup literature (Goyal et al. 2017)

**Test plan:**
1. Implement WSD as learning rate scheduler
2. Train 6M with WSD (keep everything else constant)
3. Train 6M with cosine (baseline)
4. Compare final validation loss and convergence speed
5. If WSD wins: use for Phase 1 baseline

---

### 9. Cosine Annealing with Warmup (current baseline)
**Source:** Loshchilov & Hutter, 2016
**Paper:** "SGDR: Stochastic Gradient Descent with Warm Restarts"

**Current implementation:** Linear warmup (10% of steps) + cosine decay

**Expected improvement:** Baseline (0x, we have this)
**Implementation complexity:** Already in codebase
**Notes:**
- Well-understood, industry standard
- WSD (above) may improve by 5-15%

---

## Batch Size and Gradient Accumulation

### 10. Critical Batch Size Scaling (Chinchilla Updates)
**Source:** DeepMind, Chinchilla updates 2024
**Paper:** "Critical Batch Sizes for Optimal LLM Training" (2024 research)
**Key Concept:** Optimal batch size depends on model size and task

**Mathematical pattern:**
```
- Critical batch B_crit ≈ α * L^β where L = model layers
- For our SSM: B_crit ≈ 32-64 × grad_accum is reasonable
- Scales roughly: 6M → batch=64, 50M → batch=256, 100M → batch=512
- Grad accumulation: 2-4x to stay within VRAM (12GB at 50M)
```

**Expected improvement:** 10-20% convergence speed from proper batch sizing
**Our case applicability:** High — not yet optimized
**Challenges:**
- VRAM constraint limits batch size at 50M+
- Grad accumulation interacts with BN statistics
- STE may have different batch size sensitivity

**Implementation complexity:** Empirical (test different batch sizes)
**Key references:**
- Chinchilla paper updates (2024)
- Critical batch size theory (Zhang et al. 2019)

**Test plan:**
1. Measure baseline: current batch=32 + grad_accum=2
2. Test batch=64 + grad_accum=1 (same effective)
3. Test batch=32 + grad_accum=4 (larger effective)
4. Measure convergence curves
5. Pick optimal for VRAM constraint

---

## 8-bit Optimizer Memory Optimization

### 11. AdamW8bit (bitsandbytes)
**Source:** bitsandbytes 2024
**Paper:** "8-bit Optimizers via Block-wise Quantization" (bitsandbytes docs)
**Key Technique:** Quantize optimizer states (m, v) to 8-bit while computing in FP32

**Memory reduction:**
```
- Standard AdamW: 2 FP32 buffers per param = 8 bytes * 2 = 16 bytes per param
- AdamW8bit: 2 INT8 buffers + 1 FP32 scale per block = ~2-3 bytes per param
- Reduction: 16 → 3 = 5.3x on optimizer states (total model ~2x)
```

**Expected improvement:** 50% total model memory at 50M scale
**Our case applicability:** Very high — memory-constrained at 50M
**Challenges:**
- Quantized optimizer states may lose precision
- STE gradients already noisy — 8-bit accumulation may degrade
- Unknown interaction: ternary weights + 8-bit momentum

**Implementation complexity:** Low (use bitsandbytes library)
**Key references:**
- bitsandbytes documentation
- Tim Dettmers' work on quantized training (ICLR 2022)

**Test plan:**
1. Train 6M with AdamW8bit vs standard AdamW
2. Measure: VRAM usage, convergence speed, final loss
3. If no degradation: use for 50M (mandatory for VRAM)
4. Monitor gradient norms (may increase with quantized accumulation)

---

## Summary Table

| Optimizer | Expected Gain | Complexity | Test Priority | Novelty Risk |
|-----------|---------------|-----------|---|---|
| Muon | 2-5x | High | P1 (high upside) | 60% (novel combo) |
| Sophia | 2x | Medium | P2 | 30% (known alg) |
| AdEMAMix | 1.3-2x | Low | P3 | 40% (new on ternary) |
| SPAM | Stability | Low | P1 (QAT phase) | 50% (new alg) |
| SOAP | 1.5-3x | High | P3 (if VRAM OK) | 30% (known alg) |
| Schedule-Free | 1-1.5x | Low | P1 (easy win) | 40% (new on ternary) |
| Prodigy | 1x (tuning saved) | Medium | P4 (low priority) | 50% (new) |
| WSD Schedule | 1.05-1.15x | Trivial | P1 (baseline) | 20% (empirical) |
| Critical Batch | 1.1-1.2x | Empirical | P1 (easy test) | Low (known) |
| AdamW8bit | Memory only | Low | P1 (memory crucial) | 40% (STE combo) |

---

**Testing Recommendation:**
1. Phase 1 baseline: WSD + 8-bit optimizer + Schedule-Free AdamW on 6M (easy wins)
2. Phase 2 validation: Muon + SPAM on 50M if Phase 1 successful
3. Phase 3 advanced: Sophia or SOAP only if earlier phases show 20%+ gain

---

*Last updated: February 2026*
