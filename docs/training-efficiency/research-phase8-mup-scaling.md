# Phase 8: μP Scaling Laws — Research Notes

**Research Date:** 2026-02-14
**Context:** Applying μP (Maximal Update Parametrization) to our ternary SSM architecture (ternary weights + SSD + MoD routing) to enable hyperparameter transfer from 6M proxy model to 1B+ target without retuning.

---

## 1. What is μP?

### Core Concept

**Maximal Update Parametrization (μP)** is a principled method for parameterizing neural networks that makes optimal hyperparameters **independent of model width**. This enables hyperparameter sweeps on small "proxy" models (e.g., 6M params) and direct transfer to large "target" models (e.g., 1B+ params) without retuning.

**Key Benefits:**
- **Compute savings:** Cerebras reports 93% reduction in HP tuning compute for GPT-3 6.7B (only 7% of pretraining cost)
- **Predictable scaling:** Optimal learning rate remains stable across width variations
- **Better scaling laws:** Tighter loss prediction across model scales due to consistent training dynamics
- **Universal applicability:** Works for Transformers, ResNets, CNNs, Diffusion models, MoE, and Fourier Neural Operators

### The Problem μP Solves

**Standard Parametrization (SP):**
- Activations scale with width → gradient magnitudes change → optimal learning rate drifts
- Each model size requires expensive HP sweeps
- Scaling laws have high variance due to inconsistent dynamics

**μP Solution:**
- Controls activation magnitudes to remain O(1) regardless of width
- Three components per layer: multiplier (a), initialization σ (b), learning rate (c)
- Maintains invariant: pre-activations stay constant across width changes

---

## 2. u-μP (ICLR 2025)

### What u-μP Adds

**Unit-Scaled Maximal Update Parametrization (u-μP)** improves μP by combining it with **Unit Scaling**, a technique that ensures activations, weights, and gradients all begin training **with scale of one**.

**Publication:** ICLR 2025 Spotlight (arXiv:2407.17465)

### Key Improvements Over Standard μP

| Aspect | Standard μP | u-μP |
|--------|-------------|------|
| **Initialization complexity** | 3 factors per layer (A_W, B_W, C_W) | Simplified via abc-symmetry (B_W = 1 universally) |
| **Base-shape hyperparameters** | Required (complicates implementation) | Eliminated |
| **Default hyperparameters** | Require tuning | Near-optimal out-of-the-box |
| **FP8 training** | Requires careful scaling | Works out-of-the-box |
| **Embedding LR rule** | c_emb = constant | c_emb = 1/√fan-out (better transfer) |
| **α hyperparameters** | Per-weight association | Per-operation association (more independent) |

### u-μP Scaling Rules

**Input (Embedding) Layer:**
- Parameter multiplier: A_W = 1
- Initialization: B_W = 1
- Adam LR: C_W = η · (1/√fan-out)

**Hidden Layer:**
- Parameter multiplier: A_W = 1/√fan-in
- Initialization: B_W = 1
- Adam LR: C_W = η · (1/√fan-in)

**Output Layer:**
- Parameter multiplier: A_W = 1/fan-in (forward), 1/√fan-in (backward via cut-edge rule)
- Initialization: B_W = 1
- Adam LR: C_W = η

**Residual connections:**
- Applied as branch multipliers: α_res and α_res-attn-ratio

### Transferable Hyperparameters

**Basic set:**
- η (global learning rate multiplier)

**Extended set (operation-level multipliers):**
- α_emb, α_attn-softmax, α_ffn-act, α_loss-softmax
- α_res, α_res-attn-ratio
- α_out

### Unit Scaling Integration

All tensors initialize with **unit variance** through coordinated scaling factors. This ensures activations, weights, and gradients remain within floating-point representable ranges without explicit dynamic rescaling for most operations — critical for FP8 training.

---

## 3. μP for SSMs

### Current State (2025-2026)

**Direct μP application to SSMs:** No published papers specifically addressing μP for Mamba, S4, or SSD architectures found in 2025-2026 literature.

**Relevant findings:**
1. **μP works on Mamba architectures** (confirmed in howtoscalenn.github.io): "muP works successfully on Mamba architectures"
2. **Learning rate sensitivity:** 2025 research shows learning rate choice plays an outsized role in recurrent model performance — some negative SSM results may reflect suboptimal HP tuning rather than architectural limitations
3. **Initialization matters:** Mamba abandoned S4's complex manual parametrization (HiPPO constraints) in favor of S4D-Real initialization

### Mamba's Initialization Scheme

**S4D-Real initialization:**
- Used for A matrix (state transition matrix)
- Chosen because it has the same spectrum as original S4-LegS HiPPO Matrix
- A is initialized with HiPPO-LegS structure, then diagonalized

**dt_proj bias initialization:**
- Sampled log-uniformly in range (Δ_min = 0.001, Δ_max = 0.1)
- Covers geometric range of timescales (expected length 10 to 1000)
- Implementation: `dt_proj.bias` set using inverse softplus to ensure F.softplus(dt_bias) ∈ [0.001, 0.1]
- Performance tip: Set 1/Δ_min ≈ sequence length (e.g., 0.001 → ~10³ samples)

**State dimension scaling:**
- H (state dimension) typically refers to number of independent SSM copies
- SSMs operate on scalars → H different stacked copies with different parameters
- Challenge: State dimension needs to scale linearly with sequence length for full memorization

### Open Questions for SSM + μP

1. **State dimension (N) scaling:** Does N scale with width under μP, or remain fixed?
   - Current practice: N often fixed (e.g., 16 or 64) regardless of model width
   - μP implication: If N is "matrix-like" dimension, should it scale as n^0.5?

2. **A matrix parametrization:** A is FP16 in our architecture (not ternary)
   - Should A initialization scale with width? (Currently: HiPPO initialization is fixed)
   - Should A learning rate scale differently from other FP16 parameters?

3. **Selective scan parameters (B, C):** These are input-dependent in Mamba
   - B (input → state) and C (state → output) are generated by linear projections
   - Do these projections follow standard hidden layer μP rules?

4. **dt (timestep) scaling:** dt_proj generates input-dependent timesteps
   - Should dt_proj follow hidden layer rules?
   - Does log-uniform bias initialization interact with μP?

---

## 4. μP for Quantized/Ternary Models

### BitNet and Ternary Quantization (2025)

**BitNet b1.58 (Microsoft, 2025):**
- Ternary weights {-1, 0, +1} (1.58 bits per weight on average)
- Trained natively in ternary via Quantization-Aware Training (QAT)
- Uses Straight-Through Estimator (STE) for gradient flow through non-differentiable quantization

**Key findings:**
1. **Scaling laws similar to FP16:** BitNet exhibits similar scaling properties to full-precision models
2. **Width scaling requirement:** 1.58-bit models match FP16 performance when hidden size is **2× larger**
   - Example: 1.58-bit @ width=192 ≈ FP16 @ width=96
3. **Performance gap shrinks with scale:** As model size increases, gap between FP16 and BitNet narrows
4. **Quantization threshold scaling:** Scale = mean(|W|), then quantize(W/scale), then rescale

### HESTIA: Hessian-Guided QAT (2026)

**Problem with standard STE:**
- Hard rounding from training start prematurely discretizes optimization landscape
- Persistent gradient mismatch between latent weights and quantized weights
- Hinders effective optimization of quantized models

**HESTIA's approach (arXiv:2601.20745):**
1. **Temperature-controlled softmax relaxation** instead of rigid step function
   - Maintains gradient flow early in training
   - Progressively hardens quantization
2. **Hessian trace metric** as lightweight curvature signal
   - Drives fine-grained temperature annealing
   - Enables sensitivity-aware discretization across model
3. **Differentiable optimization path** unlike standard STE

**Results:** Average zero-shot improvements of 5.39% and 4.34% on Llama-3.2 ternary QAT

### μP + Quantization: Open Questions

**No published work found** specifically combining μP with ternary/quantized training in 2025-2026 literature.

**Critical questions for our architecture:**

1. **Quantization threshold scaling:**
   - Current practice: scale = mean(|W|) is width-independent
   - Under μP: Weights initialize with σ ∝ 1/√n, so mean(|W|) ∝ 1/√n
   - Should quantization scale factor adjust with width?

2. **STE gradient scaling:**
   - Our triple-STE setup (FP16 latent → FP16 QAT → ternary inference)
   - Does STE backward pass need width-dependent scaling?
   - HESTIA's temperature annealing: does initial temperature scale with width?

3. **Learning rate for quantized layers:**
   - FP16 latent weights follow standard μP rules (lr ∝ 1/n for Adam)
   - But effective update magnitude also affected by quantization granularity
   - Should lr be further adjusted to account for quantization error?

4. **Ternary layers vs FP16 layers:**
   - Our A matrix is FP16, rest is ternary
   - Do FP16 and ternary layers need different μP scaling?
   - Or does "latent weight" perspective unify them?

---

## 5. Practical Implementation

### The mup Library (Microsoft)

**Installation:**
```bash
pip install mup
```

**Core workflow (4 steps):**

#### Step 1: Model Definition

Replace output layer with `MuReadout`:
```python
from mup import MuReadout
readout = MuReadout(width, d_out)
```

For transformers, use `1/d` attention scaling (not `1/√d`):
```python
attention_scores = query @ key.T * 8 / d  # 8/d for backward compat with d=64
```

#### Step 2: Shape Configuration

Create base (width=1), delta (width=2), and target models, then:
```python
from mup import set_base_shapes
set_base_shapes(model, base_model, delta=delta_model)
```

This attaches `infshape` attributes marking dimensions as infinite (scaled) or finite (fixed).

#### Step 3: Initialization

Replace PyTorch initializers with mup versions:
```python
import mup.init as init
init.uniform_(param, -0.1, 0.1)
init.kaiming_normal_(param)
```

#### Step 4: Optimization

Use mup-specific optimizers with automatic width-dependent LR scaling:
```python
from mup import MuAdam, MuSGD
optimizer = MuAdam(model.parameters(), lr=0.001)
optimizer = MuSGD(model.parameters(), lr=0.1)
```

LR automatically scaled by `globalLR / p.infshape.width_mult()`, where multiplier ≈ `fan_in / base_fan_in`.

### Coordinate Check (Critical Validation)

**Purpose:** Verify μP implementation correctness before expensive training runs.

**How it works:**
1. Train models of different widths (e.g., 128, 256, 512, 1024) for 10 steps
2. Record mean absolute activation value per layer type at each step
3. Plot activation magnitudes vs width

**Success criteria:**
- **Flat curves** → μP correctly implemented (activation scale independent of width)
- **Exploding curves** → Bug in μP (forgot to scale some dimension?)
- **Shrinking curves** → Bug in μP (incorrect scaling direction?)

**Implementation:**
```python
from mup.coord_check import get_coord_data, plot_coord_data

df = get_coord_data(models, dataloader)
plot_coord_data(df, save_to='coord_check.png')
```

**Tips:**
- Use **large learning rate** (larger than actual training) to emphasize potential issues
- Test above **critical batch size** to avoid batch-size-dependent failures
- Check **all layer types** separately (embeddings, hidden, attention, output)

### μTransfer Test (Optional Secondary Validation)

**Purpose:** Confirm optimal hyperparameters transfer across widths.

**How it works:**
1. Train proxy models (hidden size ≥ 256) with different widths
2. Sweep learning rates for each width
3. Plot optimal LR vs width

**Success criteria:**
- **μP:** Optimal LR plateau (constant across widths)
- **SP:** Optimal LR declines proportionally with width

### Common Pitfalls

| Pitfall | Consequence | Solution |
|---------|-------------|----------|
| Batch size below critical | Suboptimal LR transfer to large model | Train proxy above critical batch size |
| Incomplete scaling (forgot attention logits) | Training instability persists | Check all layers in coord check |
| Tied embeddings not sharing init | Incorrect scaling | Ensure embedding and unembedding have identical initialization |
| Architecture changes after tuning | Hyperparameters no longer optimal | Re-tune whenever architecture or training procedure changes |
| Forgot to vary d_ffn in delta model | Coord check fails (curves explode/shrink) | Ensure all width-dependent dimensions vary in delta |

### Recommended Proxy Model HP Sweep

**Train proxy for:** 20 tokens per parameter (e.g., 6M params → 120M tokens)

**Tune 4 hyperparameters:**
1. Base initialization standard deviation (σ_0)
2. Base learning rate (η_0)
3. Embedding multiplier (β_emb or α_emb)
4. Output logit multiplier (β_logit or α_out)

**Expected compute savings:** 2× faster to reach same performance compared to standard training.

---

## 6. Standard μP Scaling Rules (Summary)

### Initialization Standard Deviation (σ)

| Layer Type | Formula | Rationale |
|------------|---------|-----------|
| **Hidden layers (n×n)** | σ = σ_base · n^(-0.5) | Control output variance via CLT |
| **Embedding (vocab×n)** | σ = σ_base | Input layer, no width scaling |
| **Unembedding (n×vocab)** | σ = σ_base | Output layer, no width scaling |
| **Residual branches** | σ = σ_base · n^(-0.5) · (2L)^(-0.5) | Account for 2L residual additions |

### Learning Rate Scaling (Adam/AdamW)

| Layer Type | Formula | Notes |
|------------|---------|-------|
| **Hidden layers** | η = η_base · n^(-1) · b̃^(0.5) · d̃^(α_data) | n^(-1) maintains update scale |
| **Embedding** | η = η_base · b̃^(0.5) · d̃^(α_data) | No width scaling |
| **Unembedding/readout** | η = η_base · b̃^(0.5) · d̃^(α_data) | No width scaling |

Where:
- b̃ = batch_size / batch_size_base
- d̃ = dataset_size / dataset_size_base
- α_data ≈ -0.12 (empirically derived)
- Exponent 0.5 accounts for increased training steps with larger batches

### Multiplier Scaling (a)

| Layer Type | Formula |
|------------|---------|
| **Hidden layers** | a = α_hidden · 1 (no scaling) |
| **Unembedding** | a = α_readout · n^(-1) |

Typical values: α_embed ≈ 10, α_hidden ≈ 1

### Attention Mechanism

**QK^T scaling:**
```python
attention_logits = (Q @ K.T) / d_head  # NOT 1/√d_head
```

**Rationale:** After training begins, Q and K become correlated → Law of Large Numbers applies (not Central Limit Theorem).

### Adam/AdamW Hyperparameter Adjustments

| Parameter | Formula |
|-----------|---------|
| **β₁ (momentum)** | β₁_eff = 1 - b̃(1 - β₁) |
| **β₂ (RMSprop)** | β₂_eff = 1 - b̃(1 - β₂) |
| **ε (numerical stability)** | ε_eff = ε_base · b̃^(-0.5) |
| **Weight decay (hidden)** | λ = λ_base · n · b̃^(?) · d̃^(?) |
| **Weight decay (embed)** | λ = λ_base · b̃^(?) · d̃^(?) |

### Special Initializations

**Zero-variance initialization for:**
- Query projection matrices
- Residual output projection layers
- LM head (readout layer)

---

## 7. Application to Our Architecture

### Our Architecture Components

1. **Ternary weights:** {-1, 0, +1} via triple-STE (FP16 latent → FP16 QAT → ternary inference)
2. **SSD (Structured State Space Duality):** Mamba-2 selective scan with:
   - A matrix: FP16, diagonal, initialized via S4D-Real (HiPPO-derived)
   - B, C: Input-dependent projections
   - dt: Input-dependent timestep via dt_proj
3. **MoD routing:** Ternary router (not sparse MoE, just depth-wise routing)
4. **Validated at:** 6M, 50M, 100M parameters
5. **Target:** 1B+ parameters

### Proposed μP Application Strategy

#### Phase 1: Apply μP to FP16 Baseline First

**Rationale:** Isolate μP implementation issues from quantization issues.

**Steps:**
1. Implement standard Transformer with FP16 weights (no ternary, no SSM)
2. Apply μP scaling rules from Section 6
3. Run coordinate check at widths: 128, 256, 512, 1024
4. Validate hyperparameter transfer: train 6M proxy → test on 50M, 100M
5. **Only proceed after coordinate check passes**

#### Phase 2: Add SSD (SSM) Components

**Approach:** Treat SSM as "just another layer type" and apply μP rules.

**Specific rules for SSD components:**

| Component | μP Rule | Justification |
|-----------|---------|---------------|
| **B_proj (input→state)** | Hidden layer rules (σ ∝ 1/√n, lr ∝ 1/n) | Linear projection, fan-in = n |
| **C_proj (state→output)** | Hidden layer rules (σ ∝ 1/√n, lr ∝ 1/n) | Linear projection, fan-in = N (state dim) |
| **dt_proj (timestep)** | Hidden layer rules (σ ∝ 1/√n, lr ∝ 1/n) | Linear projection, fan-in = n |
| **dt_proj.bias** | Special: log-uniform [0.001, 0.1] | Mamba-specific, width-independent |
| **A matrix** | **OPEN QUESTION** | FP16, diagonal, HiPPO-initialized |
| **State dimension (N)** | **OPEN QUESTION** | Fixed vs width-scaled? |

**A matrix scaling options:**

**Option A (Fixed HiPPO):** Keep A initialization fixed regardless of width
- Pro: Matches Mamba's design (HiPPO is architecture-level choice)
- Pro: Simpler implementation
- Con: Might not follow μP theory if A participates in feature learning

**Option B (Scale LR only):** Fixed init, but lr_A scales with width
- Pro: Allows A to adapt while preserving HiPPO structure
- Con: Unclear if this maintains μP guarantees

**Option C (Scale both):** Scale both init and LR
- Pro: Follows μP theory strictly
- Con: Might conflict with HiPPO's mathematical properties

**Recommendation:** Start with Option A, run coordinate check. If A activations explode/shrink, try Option B.

**State dimension N scaling:**

**Current practice:** N fixed (e.g., 16 or 64)
**μP perspective:** If N is treated as "internal hidden dimension", should it scale?

**Recommendation:** Keep N fixed initially (simpler, matches Mamba). If coordinate check fails specifically in SSM layers, experiment with N ∝ √n.

#### Phase 3: Add Ternary Quantization

**Approach:** Apply μP to FP16 latent weights, let quantization be "downstream effect".

**Key insight:** Under triple-STE, FP16 latent weights receive gradients and updates. Ternary weights are just forward-pass quantization → μP applies to latent weights.

**Specific considerations:**

1. **Initialization:**
   - Latent weights: σ ∝ 1/√n (standard μP)
   - Quantization scale: mean(|W_latent|) ∝ 1/√n (automatic from latent init)
   - **Question:** Does this maintain ternary distribution across widths?

2. **Learning rate:**
   - Latent weights: lr ∝ 1/n (standard μP)
   - Effective ternary update: depends on STE gradient and quantization granularity
   - **Question:** Do we need additional scaling to maintain consistent ternary updates?

3. **STE backward pass:**
   - Standard STE: ∂L/∂W_latent = ∂L/∂W_ternary (straight-through)
   - Under μP: Gradient magnitudes should be O(1) if μP is correct
   - **Question:** Does STE need explicit width-dependent scaling?

**Recommendation:**
1. Apply μP to latent weights exactly as FP16 baseline
2. Run coordinate check on **both latent and ternary activations**
3. If ternary activations diverge from latent, investigate quantization scale adjustment

**HESTIA integration (future):**
- HESTIA's temperature annealing might need width-dependent initial temperature
- Start with width-independent temperature, check coord check
- If ternary convergence degrades at larger widths, scale initial temp ∝ 1/√n

#### Phase 4: Add MoD Routing

**Challenge:** μP for MoE/MoD is an open research problem (2025 paper: arXiv:2508.09752).

**Our MoD specifics:**
- Ternary router (not sparse top-k gating, just learned routing scores)
- Depth-wise routing (skip vs compute)
- Router is small ternary network

**Proposed approach (based on MoE μP research):**

1. **Router initialization:** Treat router as "hidden layer" → σ ∝ 1/√n_router
2. **Router learning rate:** lr ∝ 1/n_router (if router width scales) or lr = constant (if router is fixed-width)
3. **Routed layers:** Standard hidden layer rules
4. **Routing score scaling:** Ensure routing logits remain O(1) across widths

**Key difference from sparse MoE:**
- MoE: Multiple experts, router selects subset → load balancing issues
- MoD: Single path, router decides depth → no load balancing
- Implication: MoD might be **easier to apply μP to** than MoE

**Recommendation:** Implement MoD routing after SSM + ternary are working. Run coordinate check on router activations separately.

### Open Questions Summary

| Question | Impact | Recommended Approach |
|----------|--------|---------------------|
| **A matrix init/lr scaling** | SSM stability | Start fixed (Option A), validate via coord check |
| **State dimension N scaling** | SSM capacity vs μP | Keep N fixed, experiment if coord check fails |
| **Ternary quantization scale** | Distribution consistency | Auto from latent init, validate via coord check |
| **STE gradient scaling** | Convergence quality | Standard STE first, adjust if coord check fails |
| **HESTIA temperature scaling** | Ternary convergence at scale | Width-independent initially, scale if needed |
| **MoD router scaling** | Routing stability | Treat as hidden layer, validate separately |

---

## 8. Validation Plan

### Coordinate Check Protocol

**Models to test (all same architecture, varying width):**
- Width 64 (base), Width 128 (delta), Width 256, Width 512

**Layers to monitor separately:**
1. Embedding layer
2. SSD B_proj, C_proj, dt_proj
3. SSD selective scan output
4. MoD router (if applicable)
5. Hidden layers (if any standard MLP)
6. Output/unembedding layer

**Check both:**
- FP16 latent activations
- Ternary quantized activations (forward pass)

**Success criteria:**
- All layer types show flat curves across widths
- Both latent and ternary activations stable
- No explosions after 10 training steps with large LR

### Transfer Validation Protocol

**Stage 1: 6M → 50M transfer**
1. Train 6M proxy with width sweep (find optimal η_0, σ_0, α_emb, α_out)
2. Apply same HPs to 50M model
3. Compare convergence speed and final loss to 50M with independent tuning
4. **Target:** 50M with transferred HPs reaches same loss in ≤ 1.5× steps

**Stage 2: 6M → 100M transfer**
1. Use same 6M-tuned HPs on 100M model
2. **Target:** 100M reaches same loss as independently-tuned 100M in ≤ 2× steps

**Stage 3: 6M → 1B+ transfer (if Stages 1-2 pass)**
1. Scale to 1B using transferred HPs
2. Monitor loss curve for instabilities
3. **Target:** No divergence, smooth convergence

### Failure Modes and Debugging

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Coord check: explosions in SSD layers | A matrix scaling incorrect | Try Option B (scale LR only) |
| Coord check: ternary activations diverge from latent | Quantization scale issue | Adjust quant scale with width |
| Transfer: 50M converges slower than expected | Batch size below critical | Increase proxy batch size |
| Transfer: 100M diverges | Architecture change broke μP | Re-check all layer scaling rules |
| Coord check: router activations explode | Router lr too high | Scale router lr ∝ 1/n_router |

---

## 9. Implementation Checklist

### Pre-Implementation

- [ ] Read full mup library documentation
- [ ] Review Microsoft mup GitHub examples (Transformer, MLP)
- [ ] Study nanoGPT-mup port (EleutherAI fork)
- [ ] Understand u-μP paper (ICLR 2025) for simplified rules

### Phase 1: FP16 Baseline

- [ ] Implement FP16 Transformer (no SSM, no ternary)
- [ ] Replace output layer with MuReadout
- [ ] Change attention scaling to 1/d (not 1/√d)
- [ ] Create base (width=64) and delta (width=128) models
- [ ] Apply set_base_shapes()
- [ ] Replace initializers with mup.init versions
- [ ] Use MuAdam optimizer
- [ ] Run coordinate check: widths 64, 128, 256, 512
- [ ] **GATE:** Must pass coord check before proceeding

### Phase 2: Add SSD

- [ ] Add SSD block with B_proj, C_proj, dt_proj
- [ ] Apply hidden layer μP rules to projections
- [ ] Keep A matrix HiPPO-initialized (fixed)
- [ ] Keep dt_proj.bias log-uniform [0.001, 0.1] (fixed)
- [ ] Keep state dimension N fixed (e.g., 16)
- [ ] Run coordinate check with SSD layers monitored
- [ ] **GATE:** SSD layer activations must be flat

### Phase 3: Add Ternary Quantization

- [ ] Add triple-STE to all layers except A matrix
- [ ] Apply μP to FP16 latent weights
- [ ] Run coordinate check on both latent and ternary activations
- [ ] **GATE:** Both latent and ternary must be stable

### Phase 4: Add MoD Routing

- [ ] Implement ternary router
- [ ] Apply hidden layer μP rules to router
- [ ] Run coordinate check with router activations monitored
- [ ] **GATE:** Router activations stable across widths

### Phase 5: Transfer Validation

- [ ] Train 6M proxy: sweep η_0, σ_0, α_emb, α_out
- [ ] Record optimal HPs
- [ ] Train 50M with transferred HPs
- [ ] Compare to 50M baseline (independent tuning)
- [ ] **GATE:** Transfer must achieve ≥ 80% of baseline efficiency
- [ ] Train 100M with transferred HPs
- [ ] **GATE:** Transfer must achieve ≥ 70% of baseline efficiency

### Phase 6: Scale to 1B+

- [ ] Apply transferred HPs to 1B model
- [ ] Monitor first 1000 steps for divergence
- [ ] Full training run
- [ ] **SUCCESS:** Stable training, no retuning required

---

## 10. Open Questions

### Critical (Block Implementation)

1. **SSM state dimension scaling:** Should N scale with width, or remain fixed?
   - **Investigate:** Literature on SSM capacity vs state dimension
   - **Test:** Coord check with N fixed vs N ∝ √n

2. **A matrix parametrization under μP:** How to handle HiPPO-initialized A?
   - **Investigate:** Whether A participates in feature learning or is structural
   - **Test:** Coord check with fixed A vs scaled A learning rate

3. **Ternary quantization scale width-dependence:** Does scale = mean(|W|) need adjustment?
   - **Investigate:** Ternary distribution moments under μP initialization
   - **Test:** Histogram of ternary values across widths

### Important (Affect Performance)

4. **HESTIA temperature annealing:** Should initial temperature scale with width?
   - **Investigate:** HESTIA paper for width-dependent recommendations
   - **Test:** Convergence quality with fixed vs scaled temperature

5. **MoD router scaling:** How to handle routing scores under μP?
   - **Investigate:** μP-for-MoE paper (arXiv:2508.09752) for insights
   - **Test:** Routing distribution stability across widths

6. **dt_proj bias scaling:** Should log-uniform range adjust with width?
   - **Investigate:** Relationship between Δ and width in Mamba
   - **Test:** SSM timescale consistency across widths

### Nice-to-Have (Optimizations)

7. **Batch size scaling:** How does critical batch size relate to width?
   - **Investigate:** μP literature on batch size transfer
   - **Test:** Loss curves at different batch sizes per width

8. **Weight decay scaling:** Should λ scale with width for ternary models?
   - **Investigate:** Weight decay behavior under quantization
   - **Test:** Overfitting vs underfitting across widths

9. **Gradient clipping:** Should clip threshold scale with width?
   - **Investigate:** Gradient magnitude distribution under μP
   - **Test:** Training stability with fixed vs scaled clip threshold

---

## 11. References

### Core μP Papers

1. **Original μP:** Yang et al. "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" (2022)
   - [https://arxiv.org/abs/2203.03466](https://arxiv.org/abs/2203.03466)

2. **u-μP (ICLR 2025):** "The Unit-Scaled Maximal Update Parametrization"
   - [arXiv:2407.17465](https://arxiv.org/abs/2407.17465)
   - [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/e3130a164f5c724e37271b93bc76dd28-Paper-Conference.pdf)
   - [OpenReview](https://openreview.net/forum?id=P7KRIiLM8T)

3. **μP for Diffusion Transformers (NeurIPS 2025):** "Scaling Diffusion Transformers Efficiently via μP"
   - [GitHub](https://github.com/ML-GSAI/Scaling-Diffusion-Transformers-muP)

4. **μP for MoE (2025):** "μ-Parametrization for Mixture of Experts"
   - [arXiv:2508.09752](https://arxiv.org/abs/2508.09752)

### Practical Guides

5. **Cerebras Practitioner's Guide:** "The Practitioner's Guide to the Maximal Update Parameterization"
   - [https://www.cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization](https://www.cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization)

6. **EleutherAI Guide:** "μTransfer"
   - [https://blog.eleuther.ai/mutransfer/](https://blog.eleuther.ai/mutransfer/)

7. **How To Scale (Interactive):** [https://howtoscalenn.github.io/](https://howtoscalenn.github.io/)

8. **Sam Stevens' μP Explainer:** [https://samuelstevens.me/writing/mup](https://samuelstevens.me/writing/mup)

### Implementation

9. **Microsoft mup library:** [https://github.com/microsoft/mup](https://github.com/microsoft/mup)
   - [PyPI](https://pypi.org/project/mup/)

10. **nanoGPT-mup (EleutherAI):** [https://github.com/EleutherAI/nanoGPT-mup](https://github.com/EleutherAI/nanoGPT-mup)

11. **nanoGPT PR #549 (Karpathy):** [μP implementation and guide](https://github.com/karpathy/nanoGPT/pull/549)

### SSM Papers

12. **Mamba (2023):** "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    - [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
    - [GitHub](https://github.com/state-spaces/mamba)

13. **Mamba-3 (2025):** "Mamba-3: Improved Sequence Modeling using State Space Principles"
    - [OpenReview](https://openreview.net/forum?id=HwCvaJOiCj)

14. **S4 Survey (2025):** "From S4 to Mamba: A Comprehensive Survey on Structured State Space Models"
    - [arXiv:2503.18970](https://arxiv.org/abs/2503.18970)

15. **HiPPO:** "How to Train Your HiPPO: State Space Models with Generalized Orthogonal Basis"
    - [arXiv:2206.12037](https://arxiv.org/abs/2206.12037)
    - [OpenReview](https://openreview.net/pdf?id=klK17OQ3KB)

16. **S4 Annotated:** "The Annotated S4"
    - [ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/annotated-s4/)

17. **SSM Initialization (2025):** "Autocorrelation Matters: Understanding the Role of Initialization Schemes for State Space Models"
    - [arXiv:2411.19455](https://arxiv.org/abs/2411.19455)

### Quantization Papers

18. **BitNet b1.58 (2025):** "BitNet b1.58 2B4T Technical Report"
    - [arXiv:2504.12285](https://arxiv.org/pdf/2504.12285)

19. **PT-BitNet (2025):** "PT-BitNet: Scaling up the 1-Bit large language model with post-training quantization"
    - [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S089360802500735X)

20. **Falcon-Edge (2025):** "Falcon-Edge: A series of powerful, universal, fine-tunable 1.58bit language models"
    - [https://falcon-lm.github.io/blog/falcon-edge/](https://falcon-lm.github.io/blog/falcon-edge/)

21. **HESTIA (2026):** "HESTIA: A Hessian-Guided Differentiable Quantization-Aware Training Framework for Extremely Low-Bit LLMs"
    - [arXiv:2601.20745](https://arxiv.org/abs/2601.20745)

22. **Understanding STE (ICLR 2019):** "Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets"
    - [arXiv:1903.05662](https://arxiv.org/abs/1903.05662)
    - [OpenReview](https://openreview.net/forum?id=Skh4jRcKQ)

23. **Ternary Quantization Survey:** "Ternary Quantization: A Survey"
    - [arXiv:2303.01505](https://arxiv.org/pdf/2303.01505)

24. **Direct Quantized Training (2024):** "Direct Quantized Training of Language Models with Stochastic Rounding"
    - [arXiv:2412.04787](https://arxiv.org/abs/2412.04787)

### Production Systems

25. **IBM Granite 4.0:** Mamba-based production model (2025)
    - [IBM Blog](https://www.ibm.com/think/topics/mamba-model)

26. **Jamba (AI21):** Hybrid Transformer-Mamba with 256K context
    - [Hugging Face](https://huggingface.co/blog/lbourdois/ssm-2022)

27. **Codestral Mamba (Mistral):** Pure SSM code generation model
    - [Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/mamba)

28. **bitnet.cpp:** "Bitnet.cpp: Efficient Edge Inference for Ternary LLMs"
    - [ACL Anthology](https://aclanthology.org/2025.acl-long.457.pdf)

### Additional Resources

29. **EmergentMind μP Topic:** [https://www.emergentmind.com/topics/maximal-update-parameterization-mup](https://www.emergentmind.com/topics/maximal-update-parameterization-mup)

30. **Mamba GitHub Issues - A matrix initialization:** [Issue #167](https://github.com/state-spaces/mamba/issues/167)

31. **Microsoft mup GitHub Issues - Coord check troubleshooting:** [Issue #22](https://github.com/microsoft/mup/issues/22)

---

**Next Steps:**
1. Implement Phase 1 (FP16 baseline) with coordinate check
2. Document coord check results in `validation-mup-coord-check.md`
3. Address open questions through experimentation
4. Update `vision-and-status.md` with Phase 8 progress

**Estimated Timeline:**
- Phase 1 (FP16 baseline): 2-3 days
- Phase 2 (SSD): 3-4 days
- Phase 3 (Ternary): 2-3 days
- Phase 4 (MoD): 1-2 days
- Phase 5 (Transfer validation): 3-5 days
- **Total:** ~2-3 weeks for full μP implementation and validation
