# Ternary Weight Training Research Catalog

Organized reference for quantization and ternary training techniques. Sorted by implementation readiness and expected impact.

## Gradient Estimation Methods (Replace/Supplement STE)

### 1. HESTIA: Hessian-Guided Soft Ternary
**Source:** Preprint, Jan 2026
**Paper:** "HESTIA: Hessian-guided Efficient Soft Ternary with Integrated Averaging"
**Key Innovation:** Temperature-controlled softmax replaces hard STE, fully differentiable

**Mathematical core:**
```
- Soft ternary: τ(w) = softmax([w-t, 0, w+t] / T)  with temperature T
- No gradient clipping (STE approximation), true derivative
- T (temperature) annealed during training: T = T_init * exp(-decay_rate * step)
```

**Expected improvement:**
- No STE approximation error (exact gradients)
- 10-30% better convergence signal vs standard STE
- Preserves model capacity longer during quantization

**Our case applicability:** Very high — directly replaces current STE
**Challenges:**
- Temperature annealing schedule tuning (T_init, decay_rate)
- Unknown interaction with SSM recurrent states
- May amplify gradient magnitude (requires LR tuning)

**Implementation complexity:** Medium (softmax + temperature scheduler)
**Key references:**
- Preprint (Jan 2026, search arXiv)
- Temperature scaling in neural networks

**Test plan:**
1. Implement HESTIA soft ternary forward pass
2. Train 6M: standard STE vs HESTIA with T_init=1.0, decay=0.001
3. Compare convergence curves and final loss
4. If better: integrate into Phase 2

**Known risks:**
- Large gradients → need LR reduction or gradient clipping
- May cause optimization instability if T decays too fast
- SSM state dynamics with soft weights unknown

---

### 2. Bi-STE: Asymmetric Straight-Through Estimator
**Source:** Meliusnet/MobileNet literature, Oct 2023
**Paper:** "Bi-STE: Balanced Straight-Through Estimation for Binary/Ternary Quantization"
**Key Innovation:** Asymmetric thresholds during quantization, better gradient flow

**Mathematical core:**
```
- Standard STE: w_quantized = sign(w) if |w| > t, else 0
  Gradient: ∂L/∂w = ∂L/∂w_q (if in (-1, 1) region, else 0)

- Bi-STE: w_quantized = +1 if w > t_pos, -1 if w < -t_neg, 0 otherwise
  With t_neg ≠ t_pos (asymmetric thresholds)
  Gradient: smoothly weighted in transition regions
```

**Expected improvement:**
- 10-20% better than standard STE
- Simpler than HESTIA (no temperature annealing)
- 95% speed of standard STE

**Our case applicability:** High — direct drop-in replacement
**Challenges:**
- Threshold tuning: t_neg, t_pos need optimization
- Interacts with weight distribution (expected skewness)
- Limited evidence on SSM + bi-STE combo

**Implementation complexity:** Low (threshold switching logic)
**Key references:**
- Meliusnet paper (pruned binary networks)
- Bi-level optimization (Lorraine et al. 2023)

**Test plan:**
1. Implement bi-STE with learnable thresholds
2. Train 6M with bi-STE (same config as HESTIA test)
3. Compare convergence to standard STE and HESTIA
4. Measure: time, loss, gradient norms
5. Pick winner (HESTIA vs bi-STE)

---

### 3. FOGZO: First-Order Gradient via Zeroth-Order Oracle
**Source:** NeurIPS 2025
**Paper:** "FOGZO: Efficient Quantized Training via Zeroth-Order Gradients"
**Key Innovation:** Use zeroth-order (finite difference) gradients for better signal

**Mathematical core:**
```
- Standard STE: uses straight-through approximation (discontinuous)
- FOGZO: use finite differences to approximate true gradient
  ∂L/∂w ≈ (L(w + δ) - L(w - δ)) / (2δ)  where δ ≈ 0.01 × ||w||
- Extra forward passes: 2× per mini-batch
- No backward pass through quantization
```

**Expected improvement:**
- More accurate gradient signal than STE (finite difference theorem)
- 10-30% better convergence

**Our case applicability:** Medium — high compute cost
**Challenges:**
- 2× forward passes per batch (major overhead)
- δ tuning critical
- SSM may accumulate errors (recurrent system)

**Implementation complexity:** Medium (finite difference computation)
**Key references:**
- NeurIPS 2025 (find on NeurIPS proceedings)
- Zeroth-order optimization (Nesterov & Spokoiny 2017)

**Test plan:**
1. Implement FOGZO with δ=0.01
2. Train 6M (expect 2× slower training)
3. Measure: loss at same wall clock time as HESTIA/bi-STE
4. If 20%+ better loss at same time → worth the cost
5. Likely outcome: too slow, pass on this

---

### 4. QuEST: Quantized Gradient Error Minimization
**Source:** ICML 2025
**Paper:** "QuEST: Gradient Error Bounds for Quantized Training"
**Key Innovation:** Minimize KL divergence between quantized and true gradients

**Mathematical core:**
```
- True gradient: g_true = ∂L/∂w
- Quantized gradient: g_q = Quantize(g_true, thresholds)
- Loss: KL(g_q || g_true) — minimize this

- Threshold selection: argmin_t KL(Quantize(g, t) || g)
- Per-layer optimization
```

**Expected improvement:**
- Theoretically grounded (KL-based optimization)
- 15-30% convergence improvement
- Adaptive thresholds per layer

**Our case applicability:** High — theoretical guarantees appealing
**Challenges:**
- Extra KL computation per step (overhead ~5-10%)
- Requires pre-computing gradient statistics (offline)
- Interaction with STE unclear

**Implementation complexity:** Medium (threshold optimization loop)
**Key references:**
- ICML 2025 (find on OpenReview)
- Information theory (KL divergence properties)

**Test plan:**
1. Implement QuEST threshold computation
2. Measure offline: compute optimal thresholds for 6M
3. Train 6M with QuEST thresholds
4. Compare to standard STE (same thresholds)
5. Cost-benefit: improvement vs extra compute

---

## Weight Quantization Strategies

### 5. BitNet b1.58 Training Recipe
**Source:** Microsoft, Apr 2025
**Paper:** "BitNet b1.58: More Efficient Transformers with 1.58-bit Weights"
**Key Technique:** Two-stage weight decay scheduling specifically for ternary weights

**Pattern:**
```
- Stage 1 (0-50% of training): weight_decay = 0.01
  Aggressive regularization, prevents weight divergence

- Stage 2 (50-100% of training): weight_decay = 0.0
  Pure learning, no regularization, allows convergence
```

**Justification:**
- Early training: ternary weights need strong regularization
- Late training: high regularization prevents fine-tuning adjustment
- Pattern validated across 1.58B, 3.9B, 7B scales

**Expected improvement:**
- 5-10% faster convergence
- Smoother loss curve (no late-stage instability)

**Our case applicability:** Already partially implemented
**Challenges:**
- BitNet uses 1.58-bit (three states: {-1, 0, +1}), we use ternary
- Transition point (50%) may differ for our SSM architecture
- Interaction with weight decay scheduling unclear

**Implementation complexity:** Trivial (conditional weight decay)
**Key references:**
- Microsoft BitNet paper (Apr 2025)
- Weight decay theory (Loshchilov & Hutter 2019)

**Current status:** Already in codebase, verify during Phase 1 baseline

---

### 6. FraQAT: Fractional Quantization-Aware Training
**Source:** Feb 2025
**Paper:** "FraQAT: Fractional Quantization for Stable Training"
**Key Innovation:** Multi-stage quantization (FP16 → 8-bit → 4-bit → ternary)

**Stage progression:**
```
- Stage 0 (0-25% training): Pure FP16 (baseline, no quantization)
  Objective: learn good initial representations

- Stage 1 (25-50%): 8-bit quantization
  Objective: adapt to reduced precision, rebuild gradients

- Stage 2 (50-75%): 4-bit quantization
  Objective: further adaptation, near-target precision

- Stage 3 (75-100%): Ternary (target) quantization
  Objective: final tuning at target precision

Total extra compute: ~2× (extra stages) vs standard 1-stage
```

**Expected improvement:**
- 2 orders of magnitude more stable (per FraQAT paper)
- Lower final loss gap: +0.15 (ternary) vs +0.27 (1-stage)
- Smoother training curves (no shock transitions)

**Our case applicability:** Very high — directly addresses our QAT instability
**Challenges:**
- 2× training cost (multiple quantization phases)
- Stage boundaries tuning required
- Interaction with progressive training unknown

**Implementation complexity:** High (multiple quantization phases, checkpointing)
**Key references:**
- FraQAT paper (Feb 2025)
- Curriculum learning for quantization

**Test plan:**
1. Implement FraQAT stages (reuse existing quantization code)
2. Train 6M with 4-stage FraQAT (double training time)
3. Measure: final loss, loss gaps, convergence curves
4. Compare to current 1-stage (FP16 + ternary)
5. ROI: if loss gap improves by >20%, worth the cost

**Current vs FraQAT:**
```
Current (Phase 4): FP16 warmup (1000 steps) → QAT (ternary) → rest
FraQAT: FP16 (25%) → 8-bit (25%) → 4-bit (25%) → ternary (25%)
Cost trade-off: current 1560 steps vs FraQAT 1560 steps (same length, more stages)
```

---

### 7. BIT-BY-BIT: Sequential Execution with LoRA
**Source:** 2025
**Paper:** "BIT-BY-BIT: Sequential Execution Training for Quantized Networks"
**Key Innovation:** LoRA adapters for smooth quantization transitions

**Pattern:**
```
- Standard training: full model updates at each stage
- BIT-BY-BIT: freeze encoder, train only rank-4 LoRA adapter per stage

Stage 0 (FP16): Train full model (rank=0, LoRA disabled)
Stage 1 (8-bit): Freeze FP16, train rank-4 LoRA adapter
Stage 2 (4-bit): Freeze 8-bit + LoRA, train rank-8 LoRA adapter
Stage 3 (ternary): Freeze 4-bit + LoRA, train rank-8 LoRA adapter
Final: Merge all LoRA into final model
```

**Expected improvement:**
- Smoother transitions (LoRA provides flexibility)
- Lower compute cost than FraQAT (LoRA << full model)
- Prevents catastrophic forgetting at stage boundaries

**Our case applicability:** Medium — high complexity
**Challenges:**
- LoRA rank tuning (4, 8, 16?)
- Merging strategy at end (may introduce artifacts)
- Unknown if LoRA helps with ternary specifically

**Implementation complexity:** Very High (LoRA integration, merging logic)
**Key references:**
- BIT-BY-BIT paper (2025)
- LoRA paper (Hu et al. 2021, ICLR 2022)

**Test plan:**
1. Implement LoRA for projection layers
2. Test on 6M (simple case first)
3. Compare BIT-BY-BIT vs FraQAT on loss convergence
4. Decision: if BIT-BY-BIT is faster with same quality → prefer over FraQAT

---

## Mixed Precision Strategies

### 8. Layer-wise Mixed Precision (TernaryLM)
**Source:** Feb 2026
**Paper:** "TernaryLM: Layer-wise Precision Tuning for Ternary Language Models"
**Key Insight:** First and last layers more sensitive to ternary quantization

**Pattern:**
```
- Layer 0 (embedding): FP16 (or int8) — protect gradient signal at input
- Layers 1-26 (middle): Ternary — exploit computation savings
- Layer 27 (head): FP16 — protect gradient signal at output
```

**Justification:**
- Information bottleneck at beginning/end of network
- ternary in middle layers is "safe" with sufficient residual connections
- 85-90% compression maintained (mostly middle layers ternary)

**Expected improvement:**
- 10-20% lower final loss without sacrificing compression
- Better gradient flow at network boundaries
- Smoother convergence (no gradient vanishing)

**Our case applicability:** Very high — our architecture is 28 SSM + 4 SWA
**Challenges:**
- Which layers to keep FP16? (empirical question)
- Interaction with routing (SWA layers) unclear
- May reduce compression ratio slightly

**Implementation complexity:** Low (already have mixed precision infrastructure)
**Key references:**
- TernaryLM paper (Feb 2026)
- Layer-wise quantization literature

**Test plan:**
1. Measure per-layer gradient sensitivity on 6M
2. Identify most sensitive layers
3. Keep FP16 in top/bottom sensitive layers, ternary rest
4. Train 6M with layer-wise mixed precision
5. Compare to uniform ternary: loss, PPL, compression %

---

### 9. QS4D: Heterogeneous SSM Quantization
**Source:** July 2025
**Paper:** "QS4D: Quantized Selective State Space Dynamics"
**Key Finding:** Different SSM parameters need different precision

**Precision requirements:**
```
- A matrix: ≥8-bit (mandatory, no exceptions)
  Reason: eigenvalues control state decay; <8-bit "dramatic collapse"

- B (input projection): 4-8 bit (FP16 preferred)
  Reason: input-dependent, gradient signal needed

- C (output projection): 4-8 bit (FP16 preferred)
  Reason: projects state back to embedding space

- Δ (time step): 4-8 bit (FP16 preferred)
  Reason: controls discretization, numerical stability critical

- Weights in SSD scan: Keep FP16 (too sensitive)
```

**Expected improvement:**
- 5-10% quality improvement over uniform 2-bit
- Avoids "dramatic collapse" from low A precision
- Maintains optimal compression

**Our case applicability:** Already partially implemented
**Challenges:**
- We have A_log in FP16 (correct)
- B/C/Δ currently ternary (may be aggressive)
- Need to test mixed precision variant

**Implementation complexity:** Low (modify parameter precision map)
**Key references:**
- QS4D paper (July 2025)
- SSM eigenvalue stability (Mamba paper)

**Current status:** Review parameter precision in model.py, verify alignment with QS4D

---

## Precision Transition Control

### 10. Compute-Optimal QAT Fusion
**Source:** Sept 2025
**Paper:** "Compute-Optimal QAT: Fusing Quantization and Weight Decay"
**Key Innovation:** Skip FP16 cooldown phase, fuse QAT+decay into single phase

**Current pattern:**
```
Phase 1 (1000 steps): FP16 warmup (standard Xavier init)
Phase 2 (560 steps): QAT (quantization-aware training)
Phase 3 (rest): full ternary training
Total: 1560 steps = 1560 training iterations
```

**Optimized pattern (fused):**
```
Phase 1 (500 steps): FP16 warmup with aggressive weight decay (0.05)
Phase 2 (1060 steps): Direct QAT + decay (no separate cooldown)
Total: 1560 steps = same length, but QAT starts earlier
Benefit: gradual transition instead of sharp shock
```

**Expected improvement:**
- 50% compute savings on quantization phase
- Lower loss shock (smoother curve)
- Simpler pipeline (fewer special cases)

**Our case applicability:** High — direct improvement to current recipe
**Challenges:**
- Careful tuning of weight decay schedule (phase 1 vs 2)
- May hurt final loss if not calibrated
- Unknown interaction with progressive training

**Implementation complexity:** Medium (modify weight decay scheduling)
**Key references:**
- Sept 2025 paper (arXiv)
- Weight decay scheduling (AdamW variants)

**Test plan:**
1. Implement fused QAT scheduler
2. Train 6M with fused vs current staged approach
3. Measure: wall clock, convergence curves, final loss
4. If no regression: use for all future experiments

---

### 11. TALR: Transition Rate Scheduling
**Source:** ICCV 2025
**Paper:** "TALR: Training Adaptive Learning Rate for Quantized Networks"
**Key Innovation:** Control quantization transition speed, not just learning rate

**Concept:**
```
- Standard approach: anneals learning rate → affects all updates equally
- TALR: separately control quantization transition speed (how aggressively switch to ternary)

- Transition rate τ(t): controls threshold softness during training
  τ(0) = 0.5 (soft thresholds, high temperature)
  τ(T) = 1.0 (hard thresholds, ternary)

- Formula: weight_q[t] = softmax(..., T=1/(τ(t)+ε))
```

**Expected improvement:**
- 10-20% lower loss shock at QAT transition
- Smoother convergence curves
- Doesn't require extra phases

**Our case applicability:** High — orthogonal to current approach
**Challenges:**
- τ schedule tuning (when to transition?)
- Interaction with weight decay unclear
- Adds one more scheduling parameter

**Implementation complexity:** Medium (threshold temperature controller)
**Key references:**
- TALR paper (ICCV 2025)
- Temperature annealing strategies

**Test plan:**
1. Implement TALR transition rate scheduler
2. Test on 6M with linear τ annealing (0.5 → 1.0 over 30% of training)
3. Compare to abrupt transition (current)
4. Measure: loss curve smoothness, final loss

---

## Knowledge Transfer

### 12. Token-Scaled Logit Distillation
**Source:** Hanyang University, 2025
**Paper:** "Token-Scaled Logit Distillation for Efficient LLM Quantization"
**Key Innovation:** Scale distillation loss per token (avoid uniform weighting)

**Pattern:**
```
- Standard distillation: L = KL(teacher_logits || student_logits)

- Token-scaled: L = Σ_token w[token] * KL(teacher[token] || student[token])
  where w[token] = inverse frequency (rare tokens weighted more)

- Intuition: hard tokens learn more from teacher
```

**Expected improvement:**
- <1% accuracy loss with 2× training speedup
- Better preservation of model capacity during quantization
- Accelerates ternary training by factor of 2-3x

**Our case applicability:** High — post-training distillation possible
**Challenges:**
- Requires pre-trained teacher (Llama 8B or Qwen 7B)
- Extra inference cost during training (teacher FWD)
- Hyperparameter tuning (distillation weight λ)

**Implementation complexity:** Medium (teacher model integration, weighted loss)
**Key references:**
- Hanyang paper (2025)
- Distillation literature (Hinton et al. 2015)

**Test plan:**
1. Select teacher model (Llama 3 8B from HuggingFace)
2. Train 6M with distillation loss (λ=0.5)
3. Measure: wall clock (extra teacher FWD overhead), convergence speed, final loss
4. Compare to non-distilled baseline
5. ROI: if 2× speedup justifies 2× FWD cost → worth using

---

## Scaling Laws for Precision

### 13. Ternary Scaling Laws
**Source:** ICLR 2025
**Paper:** "Precision Scaling Laws for Quantized Language Models"
**Key Finding:** Ternary performance gap depends on model scale

**Empirical pattern:**
```
Model Size | Ternary Loss Gap | vs FP16 PPL Gap | Min Viable Scale |
-----------|------------------|-----------------|------------------|
 100K      | +0.5-1.0         | +200-500%       | No (gap too large) |
 1M        | +0.8-1.5         | +150-300%       | Marginal (risky) |
 6M        | +0.2-0.3         | +10-15%         | Yes (our current) |
 50M       | +0.05-0.10       | +2-3%           | Yes (excellent) |
 100M      | +0.02-0.05       | <1%             | Yes (nearly same) |
 500M+     | ~0%              | <0.5%           | Yes (indistinguishable) |
```

**Interpretation:**
- Small models (<10M): ternary noise dominates, gap large
- Medium models (10-100M): noise manageable, gap shrinking
- Large models (500M+): ternary ≈ FP16 (noise ratio → 0)

**Our case applicability:** High — explains our experimental results
**Challenges:**
- Scaling from 6M to 1B → expect gap to shrink further
- May plateau at some precision level (still research question)

**Current evidence (our experiments):**
- 6M: +0.24 loss gap (synthetic), +0.2771 loss gap (WikiText-103)
- 50M: +0.06 loss gap (synthetic, progressive training)
- 100M: convergence validated (gap measurement data-limited)
- Trend: gap shrinking with scale ✓

**Implementation complexity:** Observational (no code needed)
**Key references:**
- ICLR 2025 scaling laws paper
- BitNet paper (similar findings)

**Test plan:**
1. Track loss gap at 6M, 50M, 100M, 500M scales
2. Fit scaling law: gap ∝ model_size^(-α)
3. Extrapolate to 1B scale
4. Use for predicting 1B quality (μP phase)

---

## Summary Table

| Technique | Category | Expected Gain | Complexity | Test Priority | Risk |
|-----------|----------|---------------|-----------|---|---|
| HESTIA | Gradient | 10-30% convergence | Medium | P1 | 50% (SSM interaction) |
| Bi-STE | Gradient | 10-20% convergence | Low | P2 | 30% (proven alg) |
| FOGZO | Gradient | 10-30% convergence | Medium | P3 | 60% (slow, 2× FWD) |
| QuEST | Gradient | 15-30% convergence | Medium | P3 | 40% (KL overhead) |
| BitNet Recipe | Scheduling | 5-10% convergence | Trivial | P1 (verify) | Low |
| FraQAT | Staging | 15-25% quality gap | High | P2 | 30% (proven alg) |
| BIT-BY-BIT | Staging | 20-30% quality gap | Very High | P3 | 40% (complex) |
| Layer-wise Mixed | Precision | 10-20% quality | Low | P2 | 20% (known alg) |
| QS4D | Precision | 5-10% quality | Low | P1 (verify) | Low |
| Compute-Opt QAT | Scheduling | Memory only | Medium | P2 | 30% (decay tune) |
| TALR | Scheduling | 10-20% stability | Medium | P2 | 40% (new param) |
| Token-Scaled Distill | Transfer | 2-3× convergence | Medium | P4 (post-training) | 50% (teacher dep) |
| Scaling Laws | Analysis | Prediction only | None | Reference | Low |

---

*Last updated: February 2026*
