# v5 Implementation Plan — From Research to Reality

## Philosophy: NO LEGO

Each finding is NOT a ready-made component to plug in. The implementation process for EVERY finding is:

1. **DEEP READ** — Agent reads the full finding + source papers, extracts the MATHEMATICAL PRINCIPLE
2. **APPLICABILITY ANALYSIS** — Does this principle apply to our specific case? (ternary {-1,0,+1} weights, diagonal SSM, Mamba-style architecture, 6M test scale)
3. **DERIVE SOLUTION** — If applicable, derive a solution TAILORED to our architecture from the principle. NOT copy the original technique.
4. **DOCUMENT REASONING** — Write WHY it should work, what the expected effect is, and what would PROVE it worked
5. **IMPLEMENT** — Only after mathematical justification is clear
6. **VERIFY** — Debug logging proving the technique activated AND had real computational effect
7. **MEASURE** — Compare against baseline with proper controls

If at step 2 the principle doesn't apply → SKIP and document WHY. No wasted GPU time.

## Batch Structure

Findings are grouped into BATCHES by dependency and impact area. Each batch:
- Contains 3-5 related findings
- Has clear success criteria
- Is tested at 6M scale (100 steps functional, 500 steps quality)
- Each finding within a batch is analyzed INDIVIDUALLY before any implementation
- Batch results are documented before moving to next batch

## Scale Rules
- ALL testing at 6M scale unless user explicitly approves larger
- Functional smoke test: 15-100 steps (does it crash? does it activate?)
- Quality test: 500 steps (does loss improve vs baseline?)
- Speed test: 100 steps with timing (tok/s comparison)
- 50M scale: ONLY when user explicitly decides, NEVER suggested by agents

---

## BATCH 1: Training Speed — Safe Wins (LOW RISK)

**Goal:** Make training measurably faster with zero quality risk.
**Baseline:** Current 6M training speed and loss.
**Expected combined gain:** 4-5× training speedup

### Finding 1.1: BTG — Backward-Ternary GEMV (v5-R1-A6-P6)
- **Principle:** In backprop, gradient w.r.t. input = W^T @ grad_output. For ternary W, W^T is ALSO ternary → backward GEMV uses add/subtract instead of FP16 matmul.
- **What to analyze:**
  - Does our TernaryLinear backward currently use FP16 matmul? (likely yes — PyTorch autograd default)
  - What fraction of training time is backward pass? (typically 60-66%)
  - Can we implement custom backward that uses ternary W^T?
- **Expected gain:** 1.27-1.5× overall training speed
- **Success criteria:** Training tok/s increases by >20%, loss unchanged
- **Risk:** Very low — mathematically exact

### Finding 1.2: PTC — Persistent Ternary Cache (v5-R1-A6-P7)
- **Principle:** Ternary weights change sparsely (<5% flip per step). Re-quantizing ALL weights every step is wasteful. Cache quantized form, track dirty bits, only re-quantize changed weights.
- **What to analyze:**
  - What % of weights actually flip per training step in our model?
  - Where is quantization overhead in our forward pass?
  - Is the overhead significant enough to justify caching?
- **Expected gain:** 1.05-1.15× (depends on flip rate)
- **Success criteria:** Quantization time reduced, same loss trajectory
- **Risk:** Very low — simple bookkeeping

### Finding 1.3: CTRI — Convolutional Training, Recurrent Inference (v5-R1-A6-P9)
- **Principle:** Linear SSM h_t = A*h_{t-1} + B*x_t has DUAL form as causal convolution y = K * x where K is the SSM kernel. Convolution form can use FFT → fully parallel over sequence length. Mathematically EXACT equivalence for linear/diagonal A.
- **What to analyze:**
  - Is our A matrix fixed/diagonal? (YES — ternary diagonal)
  - Does our SSM implementation already support conv mode? (check FLA library)
  - Can we switch to conv mode for training without changing any other code?
  - What is the sequence length? (FFT benefit scales with length)
- **Expected gain:** ~3× training throughput
- **Success criteria:** Training tok/s increases by >2×, loss trajectory identical
- **Risk:** Low — well-established in Mamba/S4 literature
- **Caveat:** If A is input-dependent (selective), conv form doesn't apply. Our A is fixed diagonal → should work.

---

## BATCH 2: Training Speed — Mathematical (MEDIUM RISK)

**Goal:** Apply ternary-specific math optimizations to training.
**Depends on:** Batch 1 results (new baseline with Batch 1 improvements)
**Expected combined gain:** Additional 5-10× on top of Batch 1

### Finding 2.1: TSCS/TPC — Ternary Prefix-Sum Collapse (v5-R1-A6-M2/M9)
- **Principle:** For diagonal ternary A with eigenvalues in {-1,0,+1}, the SSM scan becomes:
  - a_ii = +1: cumulative sum (prefix sum)
  - a_ii = -1: alternating-sign cumulative sum
  - a_ii = 0: just B*x (no state)
  This replaces the sequential associative scan with 3 parallel prefix operations.
- **What to analyze:**
  - What are the actual A eigenvalue distributions in our trained model?
  - Does FLA's scan implementation support custom parallel operations?
  - How much of training time is spent in the scan?
  - Is the prefix-sum equivalent exact or approximate?
- **Expected gain:** 4-8× on scan portion, 1.8-3.5× overall
- **Success criteria:** Scan time reduced >50%, loss trajectory unchanged
- **Risk:** Low (independently derived by 2 researchers)

### Finding 2.2: FING — Fisher Natural Gradient (v5-R1-A6-M1)
- **Principle:** For ternary weights, the Fisher information matrix is SCALAR per weight (not matrix). Near-threshold weights get naturally smaller gradient steps, far-from-threshold weights get larger steps. This is essentially a per-weight adaptive learning rate that's FREE to compute.
- **What to analyze:**
  - What is the Fisher information for our ternary quantization function?
  - How does this interact with our existing optimizer (Adam)?
  - Is this equivalent to some existing technique (e.g., natural gradient, LAMB)?
  - Does it actually change which weights flip vs standard gradient?
- **Expected gain:** 1.5-2.5× faster convergence (same final quality in fewer steps)
- **Success criteria:** Same loss achieved in fewer steps (e.g., 500 steps matches 1000-step baseline)
- **Risk:** Low

### Finding 2.3: FPIT/IBTP — Full Integer Pipeline (v5-R1-A6-M5/M11)
- **Principle:** For ternary training, the forward pass only needs sign(accumulation) at thresholds. The backward pass only needs flip direction. Both are binary decisions → INT8/INT16 arithmetic suffices for the ENTIRE training loop. No FP32 needed anywhere.
- **What to analyze:**
  - Where does FP32 currently appear in our training loop?
  - Can accumulations be done in INT16 without overflow?
  - What precision does gradient accumulation actually need for ternary?
  - Does PyTorch support custom autograd with integer tensors?
- **Expected gain:** 3.5× throughput (INT8 ops are 4× faster on modern GPUs)
- **Success criteria:** Training runs entirely in INT8/16, loss matches FP32 baseline
- **Risk:** Medium — overflow and precision edge cases

---

## BATCH 3: Training Memory — Optimizer Revolution (MEDIUM RISK)

**Goal:** Dramatically reduce optimizer memory to enable larger models.
**Depends on:** Batch 2 (speed improvements reduce iteration time, making memory experiments faster)
**Expected combined gain:** 5-7× memory reduction

### Finding 3.1: AGA — Algebraic Gradient Accumulation (v5-R1-A6-P10)
- **Principle:** For ternary weights, the optimizer only needs to decide IF and WHICH DIRECTION to flip. This is a counting/threshold problem. Replace FP32 shadow weights + Adam states (12 bytes/param) with INT16 flip accumulators (2 bytes/param). When accumulator crosses threshold → flip weight.
- **What to analyze:**
  - How many gradient steps does it typically take for a weight to flip?
  - What is the right threshold? (too low = noisy flips, too high = slow convergence)
  - INT16 range is [-32768, 32767] — is this sufficient?
  - How does this interact with learning rate scheduling?
- **Expected gain:** 6.2× memory reduction
- **Success criteria:** Training memory drops >5×, loss trajectory within 5% of baseline
- **Risk:** Medium — threshold tuning critical

### Finding 3.2: TDO — Threshold-Distance Optimizer (v5-R1-A5-M1)
- **Principle:** Reparametrize each weight as (q, d, s) where q is current ternary value, d is distance to nearest threshold, s is gradient sign. Only weights near thresholds need gradient updates (80% are in "plateau" → skip them).
- **What to analyze:**
  - What does the distance-to-threshold distribution look like in our model?
  - How many weights are actually "near threshold" at any given step?
  - Does skipping plateau weights affect convergence?
  - Is MFPT (mean first passage time) analysis applicable to our case?
- **Expected gain:** 5× memory reduction (skip 80% of optimizer state)
- **Success criteria:** Same convergence with 80% fewer gradient computations
- **Risk:** Low (rigorous analysis in paper)

### Finding 3.3: FOGZO — First-Order-Guided Zeroth-Order (v5-R1-A6-F9)
- **Principle:** STE is biased but fast. ZO (zeroth-order) is unbiased but slow. FOGZO uses STE as primary estimator and ZO as occasional bias corrector. 796× faster than pure ZO while reducing STE bias.
- **What to analyze:**
  - What is the actual STE bias in our model? (measure gradient error)
  - How often should ZO correction be applied? (every N steps)
  - Does bias correction actually improve final quality or just convergence speed?
  - Is the overhead of occasional ZO evaluation worth the bias reduction?
- **Expected gain:** 1-8% quality improvement (PPL), minimal speed cost
- **Success criteria:** Final loss improves >2% with <5% speed overhead
- **Risk:** Medium

---

## BATCH 4: Inference Engine — Speed (LOW-MEDIUM RISK)

**Goal:** Build fast inference path exploiting ternary properties.
**Depends on:** Batch 1-3 (trained model needed for inference testing)
**Expected combined gain:** 10-30× inference speedup

### Finding 4.1: Bitplane POPCOUNT Engine (v5-R1-A3-P8)
- **Principle:** Pack ternary weights as 2-bit pairs (sign, magnitude). Matmul becomes: mask positive weights (AND), POPCOUNT positive contributions, mask negative weights (AND), POPCOUNT negative contributions, subtract. All hardware-native operations.
- **What to analyze:**
  - What is the optimal packing format for our weight layout?
  - Does AVX-512 VPOPCOUNT instruction exist on our hardware?
  - How does this interact with activation quantization (INT8)?
  - What is the theoretical throughput ceiling on our CPU/GPU?
- **Expected gain:** 4-32× throughput (depends on hardware)
- **Success criteria:** Inference tok/s increases >4×, output quality identical
- **Risk:** Low (all primitives exist)

### Finding 4.2: Micro-Stripe Pipelined Inference (v5-R1-A5-P6)
- **Principle:** Don't load entire weight matrix. Stream it in vertical stripes of S columns. Accumulate partial results. Working memory = S × d_model × 2 bits. For S=256, working memory ~852 KB for entire 7B model.
- **What to analyze:**
  - What stripe size minimizes overhead while fitting in L1/L2 cache?
  - How does stripe processing interact with POPCOUNT engine?
  - What is the overhead of accumulating partial results?
  - Does this work for all layer types (SSM, attention, MLP)?
- **Expected gain:** Sub-1 MB working memory, no quality impact
- **Success criteria:** Inference runs with <1 MB peak working memory
- **Risk:** Very low (mathematically identical to full matmul)

### Finding 4.3: Layer Streaming from DRAM (v5-R1-A3-M11)
- **Principle:** Ternary weights are 8× smaller than FP16. Loading ternary from DRAM is FASTER than computing with FP16 already in memory (bandwidth-bound regime). Counterintuitive but proven by roofline analysis.
- **What to analyze:**
  - What is our actual memory bandwidth? (DDR5, measured not spec)
  - At what model size does streaming become faster than in-memory FP16?
  - Can we overlap layer computation with next-layer prefetch?
  - Does this apply to our 6M test model or only at larger scales?
- **Expected gain:** At scale: 5.1× faster than FP16 in-memory
- **Success criteria:** Demonstrate streaming inference at competitive speed
- **Risk:** Low (roofline analysis)

---

## BATCH 5: Architecture Core — Model Quality (MEDIUM RISK)

**Goal:** Implement architectural innovations that improve model quality.
**Depends on:** Batches 1-4 (need fast, memory-efficient training to iterate on architecture)
**Expected: Better quality at same parameter count**

### Finding 5.1: ES-SSM — Expert-Selective SSM (v5-R1-A2-M7)
- **Principle:** Instead of separate SSM state per MoE expert (fragmentation), share ONE SSM state across all experts. Each expert has its own selectivity projection (sparse) but operates on shared state. Coherent memory always maintained.
- **What to analyze:**
  - How does our current SSM handle multiple experts (if we have MoE)?
  - Does shared state limit expert specialization?
  - What is the parameter overhead of per-expert selectivity projections?
  - How does this interact with ternary quantization of projections?
- **Expected gain:** Better quality (coherent state) + memory efficiency (shared state)
- **Success criteria:** Perplexity improves vs independent-state MoE
- **Risk:** Medium

### Finding 5.2: GF(3) Basis Sharing (v5-R1-A2-M5)
- **Principle:** Ternary matrices live in GF(3) (finite field with 3 elements). Store r basis matrices + per-layer coefficients. Reconstruction is EXACT (not approximation). Only requires add/negate operations.
- **What to analyze:**
  - What rank r is needed to represent our weight matrices with acceptable quality?
  - How do we find the basis? (SVD-like decomposition over GF(3))
  - Does sharing across layers actually preserve diversity?
  - What is the compression ratio at different r values?
- **Expected gain:** 4-8× weight compression with zero error
- **Success criteria:** Same model quality with 4× fewer stored parameters
- **Risk:** Low (exact algebra)

### Finding 5.3: T3S — Two-Tier Ternary State (v5-R1-A5-M12)
- **Principle:** SSM state has heterogeneous information density. Top 43 dimensions carry 80% of variance (keep FP16). Bottom 213 dimensions are low-variance (quantize to ternary with self-correcting error). Rate-distortion optimal allocation.
- **What to analyze:**
  - What is the actual variance distribution of our SSM state dimensions?
  - What is the quality impact of quantizing low-variance dimensions?
  - Does the self-correcting mechanism actually prevent error accumulation?
  - What is the optimal hot/cold split ratio for our model?
- **Expected gain:** 4.7× state memory reduction
- **Success criteria:** Same output quality with 4× less state memory
- **Risk:** Low (rate-distortion theory)

---

## BATCH 6: Advanced Training — Paradigm Shifts (HIGH RISK, HIGH REWARD)

**Goal:** Explore fundamentally different training approaches unique to ternary.
**Depends on:** Batches 1-5 (need solid baseline to compare against)
**Expected: Potentially revolutionary, but may also fail**

### Finding 6.1: SGTS — Surrogate-Guided Ternary Search (v5-R1-A6-P3)
- **Principle:** Train cheap surrogate (linear model) to predict loss change from weight flips. Only 2 possible flips per weight → discrete and locally predictable. Surrogate evaluates millions of candidates per second, real model validates periodically.
- **What to analyze:**
  - Can a linear surrogate actually predict flip outcomes accurately?
  - How often does the surrogate need recalibration?
  - How does surrogate training interact with main model training?
  - At what model size does surrogate overhead become negligible?
- **Expected gain:** 10-100× if surrogate is accurate
- **Success criteria:** Surrogate-guided flips improve loss faster than STE
- **Risk:** Medium-High

### Finding 6.2: TWCCP — Constraint Propagation (v5-R1-A6-P5)
- **Principle:** Reformulate training as constraint satisfaction. Each data point constrains weight values. Use arc-consistency propagation to deduce weights without gradients. Ternary (3 values) makes CSP tractable.
- **What to analyze:**
  - How to formulate neural network training as CSP?
  - What constraints does each training example impose?
  - How many weights can be determined by propagation vs search?
  - Does this scale beyond trivial networks?
- **Expected gain:** 10-24× if CSP resolves most weights
- **Success criteria:** Non-trivial fraction of weights determined by propagation
- **Risk:** High

### Finding 6.3: FF Layer-Streaming (v5-R1-A5-P4)
- **Principle:** Forward-Forward algorithm: train each layer independently with local goodness objective. Peak memory = single layer (not full model). For 7B: 132 MB peak.
- **What to analyze:**
  - Does Forward-Forward achieve competitive quality on language modeling?
  - What is the "goodness" function for SSM layers?
  - How does local training affect cross-layer feature coherence?
  - Can we use FF for initialization then fine-tune with backprop?
- **Expected gain:** 636× memory reduction (but quality is the question)
- **Success criteria:** Quality within 20% of backprop baseline
- **Risk:** High (FF hasn't achieved SOTA on LLMs)

---

## Summary Table

| Batch | Focus | Risk | Expected Gain | Dependency |
|-------|-------|------|---------------|------------|
| 1 | Training Speed (safe) | Low | 4-5× speed | None (baseline) |
| 2 | Training Speed (math) | Medium | +5-10× speed | Batch 1 |
| 3 | Training Memory | Medium | 5-7× memory | Batch 2 |
| 4 | Inference Engine | Low-Medium | 10-30× inference | Batches 1-3 |
| 5 | Architecture Core | Medium | Quality + compression | Batches 1-4 |
| 6 | Paradigm Shifts | High | 10-100× potential | Batches 1-5 |

## Aggregate Targets (if batches succeed)

| Metric | Current (v2) | After Batch 1-2 | After Batch 3 | After Batch 4 | v5 Target |
|--------|-------------|-----------------|---------------|---------------|-----------|
| Training tok/s | ~9,750 | ~50,000 | ~50,000 | N/A | 100,000+ |
| Training memory | ~4.4 GB (50M) | ~4.4 GB | ~0.7 GB | N/A | <1 GB for 7B |
| Inference tok/s (GPU) | 876 | N/A | N/A | ~5,000 | 1,000+ |
| Inference tok/s (CPU) | 105 | N/A | N/A | ~500 | 1,000+ |
| Inference RAM | ~4.4 GB | N/A | N/A | <1 GB | <1 GB |
| Val PPL (50M) | 70.77 | Same | Same | Same | <70 |
