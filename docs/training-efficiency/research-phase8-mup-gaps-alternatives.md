# Phase 8: Critical Analysis of μP - Gaps, Failures, and Alternatives

**Research Date:** February 2026
**Status:** Critical pre-implementation analysis
**Purpose:** Identify risks, failure modes, and alternatives BEFORE committing weeks of implementation effort

---

## Executive Summary

This document provides a CRITICAL analysis of Maximal Update Parametrization (μP) for hyperparameter transfer in our ternary SSM model. After extensive research into recent literature (2024-2026), practitioner reports, and failure cases, several significant risks have been identified that may impact our specific architecture.

**Key Finding:** μP is not a silver bullet. Multiple failure modes exist, especially for:
- Mixture-of-Experts and routing architectures (we use MoD)
- Progressive training with changing architecture (we do this)
- Quantization-aware training with dynamic precision (we do this)
- Ternary weight distributions that change with width (we have this)

**Recommendation:** Proceed with caution. Implement verification steps incrementally, have fallback plans, and consider ternary-specific alternatives.

---

## 1. μP Failures and Limitations

### 1.1 Known Failure Cases from Literature

#### Practical Training Setup Failures

**Issue:** μP works well under ideal conditions (constant learning rate, many epochs) but FAILS in practical setups like Llama-style training.

**Source:** [EleutherAI Practitioner's Guide](https://blog.eleuther.ai/mutransfer/)

**Relevance to us:** We're targeting practical training, not academic benchmarks. Our progressive training with temperature annealing is FAR from "ideal conditions."

#### Mixture-of-Experts (MoE) Challenges

**Issue:** "Despite its success on many architectures such as Transformers and ResNets, extending μP to Mixture-of-Experts models remains an open challenge. Learning rate is NOT preserved across different granularities in MoE models."

**Source:** [μ-Parametrization for Mixture of Experts](https://arxiv.org/pdf/2508.09752) (2025)

**Relevance to us:** We use Mixture-of-Depths (MoD) routing. Tokens skip layers, creating DIFFERENT effective depths per token. This is similar to MoE's dynamic compute allocation. The paper specifically calls out "different granularities" as a failure mode.

**Recent progress (2025):** New work shows μP can be adapted for MoE, but requires specialized scaling rules for router networks and expert selection mechanisms.

#### Architectural Modifications Break Transfer

**Issue:** "It is recommended to re-tune hyperparameters whenever you make a change to your model architecture (e.g., attention algorithm, nonlinearity, position embeddings, vocabulary size) or training procedure."

**Source:** [Cerebras Practitioner's Guide](https://www.cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization)

**Relevance to us:** We have:
- Hybrid SSM + sliding window attention (not standard attention)
- Ternary weights (not continuous)
- SwiGLU activation (non-standard)
- No position embeddings (SSM handles position implicitly)
- MoD routing (dynamic architecture)

Every single one of these is a "modification" that may require re-tuning.

#### Batch Size Constraints

**Issue:** "Learning rate transfer to a large model will be sub-optimal if the proxy model is trained with a batch size smaller than the critical batch size."

**Source:** [EleutherAI Guide](https://blog.eleuther.ai/mutransfer/)

**Relevance to us:** On RTX 5070 (12GB VRAM), our 6M model trains with small batches (~64-128). The 1B model may have a MUCH larger critical batch size. We may need gradient accumulation, which introduces another interaction with μP.

#### Optimizer Compatibility

**Issue:** "Trainable normalization gains, optimizer choices like Lion, and per-layer bias inclusion can disrupt hyperparameter transfer if not aligned with μP-prescribed scaling."

**Source:** [Practitioner's Guide](https://blog.eleuther.ai/mutransfer/)

**Current:** We use AdamW (compatible).
**Future risk:** If experiments show Lion performs better, we'd need to re-derive μP rules for it. Lion has different momentum mechanics than Adam.

---

### 1.2 Stability and Numerical Issues

#### μP Does NOT Solve Instability

**Critical quote:** "While adopting μP does not completely solve the problem of instability, practitioners will still need to be mindful of precision, numerical stability, hardware failures, and outlier data."

**Source:** [Cerebras Guide](https://www.cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization)

**Implication:** We still need:
- Gradient clipping (how does this interact with μP scaling?)
- Loss spike detection
- Checkpoint recovery
- Outlier handling in ternary quantization

#### The "Depth Crisis"

**Issue:** "μP faces a critical crisis in the depth dimension: it does not provide stable training and HP transfer across network depths."

**Source:** [Complete Hyperparameter Transfer](https://arxiv.org/html/2512.22382v1) (2025)

**Relevance to us:** MoD routing means tokens see DIFFERENT effective depths:
- Token skips 0 layers: depth = 12
- Token skips 6 layers: depth = 6

μP assumes fixed depth. Our architecture violates this assumption at the token level.

**Potential mitigation:** Recent work on Depth-μP and CompleteP extends μP to depth scaling, but requires additional parameterization complexity.

---

### 1.3 Reported Implementation Bugs and Pitfalls

#### Coordinate Check Passes, μTransfer Still Fails

**Real case from GitHub Issues:** User implemented μP, coordinate check looked good (activation scales stable), but transferring learning rate from 350M model to 1.3B model produced NaN.

**Source:** [microsoft/mup Issue #22](https://github.com/microsoft/mup/issues/22)

**Lesson:** Coordinate check is necessary but NOT sufficient. It catches initialization/scaling bugs but misses optimizer/training dynamic issues.

**What coordinate check DOES catch:**
- Initialization scale errors
- Layer-wise scaling bugs
- Activation explosion/vanishing

**What coordinate check MISSES:**
- Optimizer state scaling issues
- Learning rate transfer problems under specific training regimes
- Batch size interactions
- Architectural edge cases

#### Common Implementation Bugs

From practitioner reports:

1. **Forgetting to vary dimensions in delta model:** If you vary width but forget to vary `d_ffn`, coordinate check will show curves blowing up.

2. **Incorrect groupwise rescaling:** Transformers need per-parameter learning rates and attention scales. Easy to miss one.

3. **Weight tying:** Embedding and output layer weight sharing requires special handling in μP.

**Time cost:** Practitioners report weeks of compute wasted on subtle bugs that only show up at scale.

---

## 2. Alternatives to μP

### 2.1 Neural Scaling Laws (Kaplan, Chinchilla)

#### Chinchilla Scaling Law

**Approach:** For compute-optimal training, scale model size (N) and training tokens (D) equally:
- N ∝ C^0.5
- D ∝ C^0.5
- Optimal ratio: ~20 tokens per parameter

**Source:** [Chinchilla Scaling Law](https://www.emergentmind.com/topics/chinchilla-scaling-law), [LLM Scaling Laws](https://research.aimultiple.com/llm-scaling-laws/)

**Difference from μP:**
- Chinchilla: Tells you HOW MUCH to scale (model size vs data)
- μP: Tells you HOW to scale hyperparameters given a model size

**Complementary, not competing.** We could use Chinchilla to decide 6M → 50M → 200M → 1B progression, and μP (or alternatives) to transfer HPs at each step.

#### Kaplan vs Chinchilla Reconciliation (2024-2025)

**Finding:** A 2024 study reconciled differences between Kaplan and Chinchilla laws. The discrepancy came from:
1. Last layer computational cost
2. Warmup duration
3. Scale-dependent optimizer tuning

With these corrected, both laws agree.

**Source:** [Neural Scaling Law - Wikipedia](https://en.wikipedia.org/wiki/Neural_scaling_law)

**Relevance:** Optimizer tuning is STILL required at different scales, even with good scaling laws. μP doesn't eliminate this, just reduces search space.

---

### 2.2 Recent Extensions to μP (2024-2025)

#### Unit-Scaled Maximal Update Parametrization (u-μP)

**Development:** ICLR 2025 paper combines μP with Unit Scaling to address practical limitations.

**Issues addressed:**
- Inefficient hyperparameter search
- Unclear choices of which HPs to sweep
- Difficulties with low-precision training

**Source:** [u-μP: Unit-Scaled Maximal Update Parametrization](https://arxiv.org/abs/2407.17465)

**Status:** Very recent (2024), limited practitioner reports. Unknown interaction with ternary weights.

#### Complete(d)P Parametrisation

**Extends μP to:**
- Width AND depth (simultaneously)
- Batch size
- Training duration

**Approach:** Uses an adaptation of CompleteP (transformer-specific) with per-layer hyperparameter support.

**Source:** [Complete Hyperparameter Transfer](https://arxiv.org/html/2512.22382v1) (2025)

**Relevance:** Addresses the depth crisis and our MoD variable-depth problem. But adds significant implementation complexity.

**Trade-off:** More comprehensive transfer vs. more complex implementation and debugging.

#### μP for Grouped Query Attention (GQA-μP)

**Development:** First derivation of μP scalings for grouped-query attention.

**Source:** [GQA-μP OpenReview](https://openreview.net/forum?id=UJB2uOS9MR) (2024)

**Relevance:** We don't use GQA, but this shows μP requires SPECIALIZED DERIVATIONS for non-standard architectural components. Our hybrid SSM + attention may need similar treatment.

---

### 2.3 Novel Zero-Shot HP Transfer Methods (2025-2026)

#### μTransfer for Fourier Neural Operators (FNOs)

**Approach:** Extended μP to FNOs (completely different architecture family), enabling zero-shot transfer from small FNOs to billion-parameter FNOs.

**Source:** [Maximal Update Parametrization for FNOs](https://proceedings.mlr.press/v267/li25do.html) (ICML 2025)

**Lesson:** μP CAN be extended to novel architectures, but requires mathematical re-derivation. We may need to derive "μP for ternary SSMs" from first principles.

#### Bayesian Optimization + HP Transfer

**Approach:** Use ViT as auxiliary model, apply Bayesian optimization to find optimal HPs, transfer to target model (Qwen2.5-VL).

**Source:** [Soybean Leaf Disease Recognition](https://www.mdpi.com/2073-4395/16/2/218) (2025)

**Difference from μP:** Model-agnostic, doesn't require specialized parameterization. But requires training multiple proxy models.

**Trade-off:** Flexibility vs. compute cost.

#### Spectral Architecture Search (SPARCS)

**Approach:** Uses spectral parametrization of inter-layer transfer matrix for architecture search and HP tuning.

**Source:** [SPectral ARchiteCture Search](https://www.nature.com/articles/s44387-025-00039-1) (2025)

**Relevance:** Completely different approach based on signal processing theory. May be worth exploring if μP fails for our architecture.

---

## 3. Implementation Pitfalls (What Practitioners Report)

### 3.1 Time and Compute Wasted

**Reported issues:**
- Weeks of compute wasted on coordinate check passing but μTransfer failing
- Subtle optimizer state bugs only visible at 1B+ scale
- NaN losses appearing at step 50k after smooth training

**Source:** [microsoft/mup GitHub Issues](https://github.com/microsoft/mup/issues)

**Mitigation strategies:**
1. Start with TINY models (1M → 3M → 6M) before jumping to 1B
2. Run short training runs (1000 steps) at each scale to verify stability
3. Implement comprehensive logging: loss, gradient norms, weight norms, optimizer state norms

---

### 3.2 Gradient Accumulation and Checkpointing Interactions

#### Gradient Checkpointing Slowdown

**Finding:** Gradient checkpointing slows training by ~20%. Combined with gradient accumulation, memory is saved but training becomes slower.

**Source:** [Gradient Checkpointing Guide](https://python-bloggers.com/2024/09/mastering-gradient-checkpoints-in-pytorch-a-comprehensive-guide/)

**PyTorch 2.1 improvement:** Nested checkpointing reduces memory from O(√n) to O(log n).

**Relevance to us:** We'll need both techniques for 1B model. Need to verify μP transfer holds under these conditions.

**Unknown:** Does gradient accumulation affect μP's learning rate scaling? Literature is silent on this.

---

### 3.3 Platform-Specific Issues (Windows + PyTorch)

#### Windows PyTorch Bugs (2024-2025)

**Reported issues:**
- CPU inference behavior change since PyTorch 2.4.0 on Windows
- Intermittent crashes or driver timeouts on AMD hardware
- DataLoader 7-50x slower than manual batching

**Source:** [PyTorch Windows Issues](https://github.com/pytorch/pytorch/issues)

**Relevance:** We're on Windows + RTX 5070. Need to test μP implementation on our specific platform, not assume Linux-based paper results transfer.

**Mitigation:** Test on multiple PyTorch versions (2.3, 2.4, 2.5+) to identify platform-specific regressions.

---

## 4. Our Specific Architecture Risks

### 4.1 Mixture-of-Depths (MoD) Routing

**The problem:** MoD routing creates token-level variable depth:
- Some tokens see depth = 12 (no skips)
- Some tokens see depth = 6 (skip 6 layers)

**μP assumption:** Fixed depth for all inputs.

**What μP doesn't model:**
- Tokens with different effective depths have different "distance" from input to output
- Gradient flow differs for deep vs shallow paths
- Learning rate that's optimal for deep paths may be too large for shallow paths

**Evidence from MoE research:** "Learning rate is NOT preserved across different granularities in MoE models." (2025)

**Possible mitigations:**
1. Derive MoD-specific scaling rules (extend recent MoE-μP work)
2. Use CompleteP (handles variable depth, but complex)
3. Accept that μP transfer will be approximate, require fine-tuning

**Experiment needed:** Train 6M model with/without MoD, check if optimal LR differs.

---

### 4.2 Progressive Training (Changing Architecture During Training)

**What we do:**
- Start with high `tau` (soft quantization, ~8-bit precision)
- Anneal `tau` → 0.01 (hard quantization, ternary)
- Effective precision CHANGES during training

**μP assumption:** Fixed architecture throughout training.

**Recent work:** Falcon-H1 (2025) used μP with progressive training, but applied "dampening μP multipliers" to handle spikes in SSM blocks.

**Source:** [Falcon-H1 Blog](https://falcon-lm.github.io/blog/falcon-h1/)

**Unknown:** What dampening strategy did they use? Paper doesn't specify exact formulation.

**Risk:** Our `tau` annealing is a CONTINUOUS architecture change. μP was derived for static architectures. Transfer may fail mid-training when effective precision shifts.

**Mitigation:**
1. Keep `tau` annealing schedule identical across all model sizes
2. Test transfer at DIFFERENT `tau` values (soft, medium, hard quantization)
3. Consider per-phase HP tuning (soft phase → hard phase)

---

### 4.3 HESTIA Temperature Annealing

**What we do:** Temperature parameter in router network anneals from high (soft routing) to low (hard routing).

**Interaction with μP:** Router network has its own learning rate. Does it scale differently than main network?

**Evidence:** GQA-μP required specialized scaling for attention mechanisms. Our router is another specialized component.

**Unknown:** Should router LR scale as 1/width (like output layer) or stay constant (like input layer)?

**Experiment needed:**
- Train 6M model with different router LRs
- Check if optimal router:main LR ratio changes with model width

---

### 4.4 Ternary Weight Distribution Changes with Width

**Empirical observation:** At narrow models (~6M), ternary quantization achieves ~30% zeros. At wide models (theoretical), BitNet reports ~89% zeros.

**μP assumption:** Weight distribution SHAPE should be scale-invariant (only magnitude scales).

**What's happening:**
- Narrow model: weights are densely packed, harder to quantize to zero
- Wide model: redundancy increases, more weights are exactly zero

This is a WIDTH-DEPENDENT BEHAVIOR that μP doesn't model.

**Consequence:** Effective capacity per parameter CHANGES with width. A 1B ternary model with 89% zeros has ~110M non-zero weights. A 6M ternary model with 30% zeros has ~4.2M non-zero weights.

Ratio: (110M / 1000M) vs (4.2M / 6M) = 0.11 vs 0.70

Narrow model is DENSER per parameter than wide model.

**Implication for μP:** Learning rate that's optimal for 70% density may be suboptimal for 11% density.

**Mitigation ideas:**
1. Derive scaling law that includes sparsity as a parameter
2. Use quantization threshold (0.7 × mean|W|) as the scaling anchor instead of σ_W
3. Train multiple 6M models with DIFFERENT quantization thresholds (forcing different sparsity levels) to empirically derive sparsity-aware scaling

---

## 5. Novel Approaches We Should Consider

### 5.1 Ternary-Specific Scaling Laws

**Observation:** Recent work exists on ternary scaling laws.

**Source:** [Scaling Laws and Efficient Inference for Ternary Language Models](https://aclanthology.org/2025.acl-long.1294.pdf) (ACL 2025)

**Key findings (from abstract):**
- Ternary models follow power-law scaling
- Empirically, ternary requires ~2× hidden size to match FP16 perplexity
- Scaling relationship between precision, size, and performance

**What we should extract:**
- Are their scaling exponents different from standard neural scaling laws?
- Do they provide HP transfer recommendations?
- Can we adapt their ternary-specific rules to SSMs (their work is on Transformers)?

**Action:** Deep-dive into this paper. It's the MOST relevant to our architecture.

---

### 5.2 Quantization Threshold as Scaling Anchor

**Standard μP:** Uses weight standard deviation (σ_W) as the unit of scale.

**Our proposal:** Use quantization threshold (T = 0.7 × mean|W|) as the unit of scale.

**Rationale:**
- Threshold determines ternary distribution
- Threshold is proportional to mean|W|, which μP already tracks
- Threshold is the ACTUAL parameter that controls model behavior (sparsity, effective precision)

**Scaling rule (proposed):**
```
Initialization: W ~ N(0, σ²/fan_in)  [standard μP]
Quantization threshold: T = 0.7 × E[|W|]
Learning rate for W: lr_W ∝ T / width
  (instead of lr_W ∝ σ_W / width)
```

**Why this might work:**
- Explicitly ties LR to the quantization regime
- Accounts for sparsity changes (higher sparsity → lower T → lower LR)
- Keeps ternary distribution SHAPE consistent across widths

**How to test:**
1. Train 6M, 12M, 24M models with threshold-based LR scaling
2. Compare loss curves to μP-based LR scaling
3. Measure: training stability, convergence speed, final loss

---

### 5.3 Quantization-Aware Scaling Laws (Recent Work)

**Recent findings:**
- QAT has distinct scaling laws from full-precision training
- Quantization error depends on (N, T, G): parameters, tokens, quantization group size
- Optimal compute allocation: divide training between FP phase and QAT phase

**Source:** [Scaling Law for Quantization-Aware Training](https://arxiv.org/pdf/2505.14302), [Compute-Optimal QAT](https://arxiv.org/pdf/2509.22935) (2024-2025)

**Relevance:** We use progressive training (high precision → low precision). This is EXACTLY the "FP phase → QAT phase" paradigm.

**Key insight:** "Optimally dividing training between FP and QAT phases leads to predictable scaling."

**Action:**
1. Extract their optimal phase division formula
2. Apply to our `tau` annealing schedule
3. Check if their scaling laws predict HP transfer for progressive quantization

**Advantage:** This is EMPIRICAL work specifically for quantized models, not a theoretical framework adapted from FP models.

---

### 5.4 Scaling Laws for Precision (ICLR 2025)

**Recent work:** "Scaling Laws for Precision" published at ICLR 2025.

**Source:** [Scaling Laws for Precision](https://pehlevan.seas.harvard.edu/sites/g/files/omnuum6471/files/2025-03/Kumar_etal_ICLR_2025.pdf)

**Claims:** Derives scaling laws for models with extreme quantization (including ternary) from first principles using random feature models.

**What to extract:**
- How does loss scale with (N, T, b) where b = bit-width?
- Does it predict optimal LR for different precisions?
- Can it explain why ternary requires 2× width for same performance?

**Action:** Read paper, check if it provides HP transfer rules for ternary models.

---

## 6. Recommended Research & Validation Plan

### Phase 1: Literature Deep-Dive (1 week)

**High-priority papers:**
1. [Scaling Laws for Ternary Language Models](https://aclanthology.org/2025.acl-long.1294.pdf) (ACL 2025) — MOST relevant
2. [Scaling Laws for Precision](https://pehlevan.seas.harvard.edu/sites/g/files/omnuum6471/files/2025-03/Kumar_etal_ICLR_2025.pdf) (ICLR 2025)
3. [Scaling Law for QAT](https://arxiv.org/pdf/2505.14302) (2024)
4. [μ-Parametrization for MoE](https://arxiv.org/pdf/2508.09752) (2025) — MoD relevance
5. [Complete Hyperparameter Transfer](https://arxiv.org/html/2512.22382v1) (2025) — Depth handling

**Extract from each:**
- Scaling exponents
- HP transfer recommendations
- Failure modes reported
- Experimental validation methodology

---

### Phase 2: Empirical Baseline (1-2 weeks compute)

**Goal:** Establish ground truth BEFORE attempting μP.

**Experiments:**

1. **Width scaling (no μP):**
   - Train 3M, 6M, 12M models with FIXED HPs
   - Measure: final loss, training stability, convergence speed
   - Identify: does performance improve with width? Any instabilities?

2. **MoD impact:**
   - Train 6M with MoD vs without MoD
   - Compare optimal learning rates
   - Hypothesis: MoD requires lower LR (variable depth creates instability)

3. **Sparsity vs width:**
   - Train 3M, 6M, 12M, measure % zeros at convergence
   - Fit curve: sparsity = f(width)
   - Validate: does sparsity increase with width as BitNet reports?

4. **Progressive training phases:**
   - Train 6M, log optimal LR at different `tau` values
   - Check: does optimal LR change during progressive quantization?

**Deliverable:** Empirical scaling curves that μP must match.

---

### Phase 3: μP Implementation (2 weeks)

**Incremental approach:**

1. **Minimal μP (week 1):**
   - Implement standard μP for a SIMPLE model (no MoD, no progressive training)
   - Validate with coordinate check
   - Test μTransfer: 3M → 6M

2. **Add complexity (week 2):**
   - Add MoD (test if transfer still works)
   - Add progressive training (test if transfer breaks at `tau` transition)
   - Implement logging: track all μP scaling factors during training

**Success criteria:**
- Coordinate check: activations stable across widths
- μTransfer: LR from 3M transfers to 6M with <10% final loss degradation
- Stability: no NaNs, no loss spikes at scale transitions

**Failure triggers:**
- Coordinate check passes but μTransfer fails → check optimizer state scaling
- NaNs appear at larger widths → gradient clipping, batch size issues
- Loss spikes during progressive training → `tau` annealing interacts with μP

---

### Phase 4: Alternative Approaches (if μP fails)

**Fallback 1: Ternary-specific scaling law**
- Implement threshold-based LR scaling (Section 5.2)
- Compare to μP on same 3M → 6M → 12M progression

**Fallback 2: Per-phase tuning**
- Accept that progressive training breaks transfer
- Tune HPs separately for soft, medium, hard quantization phases
- Use Bayesian optimization for efficiency

**Fallback 3: Empirical scaling rules**
- Fit power law to our 3M, 6M, 12M results
- Extrapolate to 50M, 200M, 1B
- Validate with short runs at each scale

---

## 7. Critical Questions to Answer Before Implementation

### 7.1 Architecture Compatibility

- [ ] Does μP support token-level variable depth (MoD)?
- [ ] Can we adapt recent MoE-μP work to MoD?
- [ ] Should router network LR scale differently than main network?

### 7.2 Progressive Training Compatibility

- [ ] Does μP transfer hold when effective precision changes during training?
- [ ] What "dampening" did Falcon-H1 use for SSM spikes?
- [ ] Should we use per-phase HP tuning instead of single μP parameterization?

### 7.3 Ternary Quantization Compatibility

- [ ] Does standard μP account for changing sparsity with width?
- [ ] Should we use threshold-based scaling instead of σ-based?
- [ ] What do the ACL 2025 ternary scaling laws recommend?

### 7.4 Implementation Verification

- [ ] Is coordinate check sufficient, or do we need additional verification?
- [ ] How do we test μP on our Windows + PyTorch + RTX 5070 stack?
- [ ] What logging is needed to debug failures at scale?

---

## 8. Sources and References

### Academic Papers (2024-2026)

- [The Practitioner's Guide to the Maximal Update Parameterization - Cerebras](https://www.cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization)
- [The Practitioner's Guide to the Maximal Update Parameterization - EleutherAI](https://blog.eleuther.ai/mutransfer/)
- [u-μP: The Unit-Scaled Maximal Update Parametrization](https://arxiv.org/abs/2407.17465) (ICLR 2025)
- [μ-Parametrization for Mixture of Experts](https://arxiv.org/pdf/2508.09752) (2025)
- [Complete Hyperparameter Transfer](https://arxiv.org/html/2512.22382v1) (2025)
- [Scaling Laws for Ternary Language Models](https://aclanthology.org/2025.acl-long.1294.pdf) (ACL 2025)
- [Scaling Laws for Precision](https://pehlevan.seas.harvard.edu/sites/g/files/omnuum6471/files/2025-03/Kumar_etal_ICLR_2025.pdf) (ICLR 2025)
- [Scaling Law for Quantization-Aware Training](https://arxiv.org/pdf/2505.14302) (2024)
- [Compute-Optimal Quantization-Aware Training](https://arxiv.org/pdf/2509.22935) (2024)
- [Falcon-H1 Technical Report](https://falcon-lm.github.io/blog/falcon-h1/) (2025)
- [GQA-μP: The Maximal Parameterization Update for Grouped Query Attention](https://openreview.net/forum?id=UJB2uOS9MR) (2024)
- [Maximal Update Parametrization for Fourier Neural Operators](https://proceedings.mlr.press/v267/li25do.html) (ICML 2025)

### Scaling Laws Research

- [Chinchilla Scaling Law](https://www.emergentmind.com/topics/chinchilla-scaling-law)
- [Neural Scaling Law - Wikipedia](https://en.wikipedia.org/wiki/Neural_scaling_law)
- [LLM Scaling Laws Analysis](https://research.aimultiple.com/llm-scaling-laws/)
- [Evaluating the Robustness of Chinchilla Compute-Optimal Scaling](https://arxiv.org/html/2509.23963v1) (2024)

### Ternary and Quantization Research

- [A One-Stop Guide to Scaling Laws in LLM Quantization](https://mr-philo.github.io/posts/2025/08/quantization-scaling-law/)
- [BitNet b1.58 Model](https://www.emergentmind.com/topics/bitnet-b1-58-model)
- [Sparsity-control Ternary Weight Networks](https://www.sciencedirect.com/science/article/abs/pii/S0893608021004093)
- [Ternary Neural Network Guide 2025](https://www.shadecoder.com/topics/ternary-neural-network-a-comprehensive-guide-for-2025)
- [Mixed Precision Quantization Based on Information Entropy](https://www.nature.com/articles/s41598-025-91684-8) (2025)

### Optimizer Research

- [Convergence Analysis of Lion Optimizer](https://arxiv.org/html/2508.12327) (2024)
- [A Refined Lion Optimizer for Deep Learning](https://www.nature.com/articles/s41598-025-07112-4) (2025)

### Implementation Reports

- [microsoft/mup GitHub Issues](https://github.com/microsoft/mup/issues)
- [Coord check looks good, but μTransfer is not working - Issue #22](https://github.com/microsoft/mup/issues/22)
- [PyTorch Windows Issues](https://github.com/pytorch/pytorch/issues)
- [Gradient Checkpointing Guide](https://python-bloggers.com/2024/09/mastering-gradient-checkpoints-in-pytorch-a-comprehensive-guide/)

### SSM and Mamba Research

- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
- [State Space Duality (Mamba-2) Part I](https://goombalab.github.io/blog/2024/mamba2-part1-model/)
- [From S4 to Mamba: A Comprehensive Survey](https://arxiv.org/abs/2503.18970) (2025)

---

## 9. Final Recommendations

### DO proceed with μP implementation IF:

1. We accept that it may require custom derivations for our architecture
2. We implement comprehensive verification (coordinate check + μTransfer tests + stability tests)
3. We budget 3-4 weeks for implementation and debugging
4. We have fallback plans (empirical scaling, per-phase tuning)

### DO NOT proceed with μP IF:

1. We need guaranteed results in <2 weeks
2. We cannot afford compute for extensive validation experiments
3. We're unwilling to derive custom scaling rules for MoD/ternary/progressive training

### HYBRID APPROACH (recommended):

1. **Week 1:** Deep-dive into ACL 2025 ternary scaling laws paper
2. **Week 2:** Run empirical baseline (3M, 6M, 12M) to establish ground truth
3. **Week 3:** Implement SIMPLE μP (no MoD, no progressive training) and validate
4. **Week 4:** Add complexity (MoD, progressive training) OR switch to ternary-specific scaling if simple μP fails

**Success is not guaranteed. Prepare for the possibility that μP doesn't fit our architecture and we need a custom solution.**

---

**Next Steps:** Review this analysis, decide on research direction, allocate compute budget for experiments.
