# Phase 8: μP Implementation Landscape — Practical Research

**Research Date:** 2026-02-14
**Target Stack:** PyTorch 2.10+cu128, Windows 11, RTX 5070 (12GB GDDR7, Blackwell)
**Architecture:** Custom (SSM + ternary quantization + MoD routing)

---

## Executive Summary

**TL;DR:** Microsoft's `mup` library is the primary tool but is designed for standard architectures (Transformers, ResNets). For our custom SSM + ternary + MoD architecture, we'll need significant DIY work. Good news: PyTorch 2.10 infrastructure supports all the primitives we need. Bad news: No existing μP implementation covers SSM layers, quantized training, or MoD routing.

**Key Findings:**
1. **mup library:** Active, works with PyTorch 2.x, but Transformer-centric. No SSM/quantization/MoD support.
2. **u-μP (unit-scaled μP):** Newer variant (Graphcore, 2024-2025) adds unit scaling for better low-precision (FP8/FP4) compatibility — highly relevant for RTX 5070.
3. **DIY path is viable:** PyTorch's `param_groups` infrastructure + manual scaling rules = full control.
4. **Blackwell FP4/FP8:** Native support exists, but NO existing μP library leverages it for proxy model training.
5. **Major gaps:** SSM width scaling, ternary weight initialization, MoD router scaling — ALL unsolved.

**Recommendation:** Start with DIY μP for SSM layers using `param_groups`, validate with coordinate check, then explore u-μP for FP8 proxy training on RTX 5070.

---

## 1. Existing μP Libraries (2024-2026)

### 1.1 Microsoft `mup` (Primary)

**Repository:** [microsoft/mup](https://github.com/microsoft/mup)
**Last Activity:** Active (2024-2025), maintained by Microsoft Research
**PyTorch Compatibility:** Works with PyTorch 2.x (no explicit version ceiling documented)
**Windows Support:** Yes (pure Python, no platform-specific code)

#### What It Provides

- **Automatic LR scaling:** `mup.MuAdam`, `mup.MuAdamW`, `mup.MuSGD` with built-in width-based LR adjustments
- **Special layers:** `MuReadout` (output layer with 1/width scaling), `MuSharedReadout` (for tied embeddings)
- **Initialization:** `mup.init.xavier_uniform_`, `mup.init.kaiming_normal_`, etc. (drop-in replacements for `torch.nn.init`)
- **Base shape tracking:** `set_base_shapes(model, base_model)` — attaches `infshape` attribute to each parameter for width multiplier calculation
- **Coordinate check:** Built-in verification tool to ensure μP implementation is correct (plots activation norms across widths)

#### What It Does NOT Provide

- ❌ **SSM layer support:** No guidance for A, B, C parameter scaling in state-space models
- ❌ **Quantized layer support:** No integration with quantization-aware training (QAT)
- ❌ **MoD routing support:** No router scaling rules for mixture-of-depths
- ❌ **Custom architecture docs:** Examples only cover Transformers, ResNets, MLPs

#### Key Implementation Pattern (from mup README)

```python
# 1. Define base model (width=1) and delta model (width=2)
base_model = MyModel(width=64)   # Small
delta_model = MyModel(width=128) # Slightly larger
model = MyModel(width=1024)      # Target

# 2. Set base shapes (attaches infshape to each param)
mup.set_base_shapes(model, base_model, delta=delta_model)

# 3. Replace output layer with MuReadout
# model.output = nn.Linear(width, d_out)  # OLD
model.output = mup.MuReadout(width, d_out)  # NEW

# 4. Use mup.init for initialization (after set_base_shapes)
def init_weights(m):
    if isinstance(m, nn.Linear):
        mup.init.kaiming_normal_(m.weight)

# 5. Use mup optimizer
optimizer = mup.MuAdamW(model.parameters(), lr=0.001)
```

#### Known Issues

- **torch.save limitation:** `infshape` attributes are NOT saved in checkpoints (must call `set_base_shapes` after loading)
- **DataParallel incompatibility:** Use `DistributedDataParallel` instead (DataParallel strips custom attributes)
- **Attention scaling:** Must manually change `1/√d` to `1/d` for Transformers (or `8/d` for backward compat when d=64)

**Source:** [microsoft/mup README](https://github.com/microsoft/mup/blob/main/README.md)

---

### 1.2 EleutherAI `nanoGPT-mup` (Educational)

**Repository:** [EleutherAI/nanoGPT-mup](https://github.com/EleutherAI/nanoGPT-mup)
**Purpose:** Minimal reference implementation + coordinate check examples
**Key Resource:** ["The Practitioner's Guide to the Maximal Update Parameterization"](https://blog.eleuther.ai/mutransfer/) (EleutherAI Blog, Sept 2024)

#### What It Adds Over Microsoft mup

- **Coordinate check examples:** Scripts to reproduce all figures from the practitioner's guide
- **Verification workflows:** Both coordinate check (activation norms) and μTransfer test (optimal HP stability)
- **Extension branches:**
  - `completep`: CompleteP parametrization (2025 paper, extends μP with additional scaling rules for deeper models)
  - `supar`: Sparse μP (2024 paper, μP for sparse training dynamics)

#### Key Insight

nanoGPT-mup shows that coordinate check is the CRITICAL validation step:
- Plot average activation L1 norm across widths (64, 128, 256, 512, ...)
- If implemented correctly, lines should be **horizontal** (stable across width)
- If lines diverge, your μP scaling rules are wrong

**Why this matters for us:** We'll need to implement coordinate check for SSM layers — no existing tool does this.

**Source:** [EleutherAI/nanoGPT-mup](https://github.com/EleutherAI/nanoGPT-mup), [EleutherAI Blog](https://blog.eleuther.ai/mutransfer/)

---

### 1.3 Graphcore `unit-scaling` (u-μP)

**Repository:** [graphcore-research/unit-scaling](https://github.com/graphcore-research/unit-scaling)
**Paper:** "u-μP: The Unit-Scaled Maximal Update Parametrization" (2024-2025)
**Status:** Beta release (active development)

#### What u-μP Adds

**Unit scaling:** Ensures ALL activations, weights, gradients have RMS ≈ 1 at initialization
- **Why:** FP16/FP8 range is ~6e-5 to ~60,000 — unit scaling keeps values in-range without overflow/underflow
- **Result:** Better low-precision training stability

**Key difference from standard μP:**
- Standard μP: Scales LR and initialization to maintain feature learning across widths
- u-μP: ALSO scales activations/gradients to unit variance → better FP8/FP4 compatibility

#### Implementation Pattern

```python
import unit_scaling as uscale

# Wrap layers with unit-scaled versions
# layer = nn.Linear(d_in, d_out)  # Standard
layer = uscale.functional.linear(input, weight, bias)  # Unit-scaled

# Learning rate rules applied as pre-optimizer transformation
# Uses standard PyTorch optimizers (AdamW, etc.)
```

**RTX 5070 relevance:** u-μP is designed for low-precision training (FP8, FP4). Blackwell natively supports both. If we train proxy models in FP8, u-μP might provide better transfer stability.

**Limitation:** Still Transformer-focused, no SSM/quantization examples.

**Source:** [graphcore-research/unit-scaling](https://github.com/graphcore-research/unit-scaling), [u-μP documentation](https://graphcore-research.github.io/unit-scaling/user_guide)

---

### 1.4 Other Variants and Research (2025-2026)

#### CompleteP (2025)

**Paper:** "Don't be lazy: CompleteP enables compute-efficient deep transformers"
**Implementation:** [EleutherAI/nanoGPT-mup (completep branch)](https://github.com/EleutherAI/nanoGPT-mup/tree/completep)
**Concept:** Extends μP with additional scaling rules for **depth** (not just width)

**Key insight:** Standard μP optimizes width scaling. CompleteP adds depth scaling rules so you can transfer HPs when scaling BOTH width AND depth simultaneously.

**Relevance:** Medium — we're designing one architecture, not scaling depth much. But interesting if we later want to test 12-layer vs 24-layer versions.

---

#### μP for Mixture-of-Experts (2025)

**Paper:** "μ-Parametrization for Mixture of Experts" (Aug 2025)
**ArXiv:** [2508.09752](https://arxiv.org/html/2508.09752v1)
**Key contribution:** Extends μP to MoE models with scaling rules for:
- Number of experts
- Expert size
- Router initialization
- Sparsity patterns

**Relevance:** HIGH — we have MoD routing (similar to MoE). Paper derives parameterizations that ensure feature learning across all weights at different model widths AND expert counts.

**Implementation status:** Theory paper, no public code yet (as of Feb 2026).

**What we need from this:** Router scaling rules — how should ternary router weights be initialized and LR-scaled as width changes?

**Source:** [μ-Parametrization for Mixture of Experts](https://arxiv.org/html/2508.09752v1), ["Hyperparameter Transfer with Mixture-of-Experts Layers"](https://arxiv.org/html/2601.20205)

---

## 2. DIY μP Implementation for Custom Architectures

Since no library supports SSM + ternary + MoD, here's the manual approach.

### 2.1 Core Concept Recap

μP = "maximal update parametrization" = specific scaling rules that make optimal HPs stable across width.

**Three ingredients:**
1. **Initialization scaling:** How to initialize weights based on width
2. **Learning rate scaling:** How to adjust LR per parameter group based on width
3. **Output multiplier:** Scalar applied to output layer to control logit scale

**Width multiplier:** `width_mult = target_width / base_width` (e.g., 1024/64 = 16)

---

### 2.2 Standard μP Rules (Transformers)

| Parameter Type | Initialization | Learning Rate | Notes |
|----------------|----------------|---------------|-------|
| **Input embedding** | `std = 1.0` (no width scaling) | `lr = base_lr` | Fixed |
| **Hidden weights** (linear layers) | `std = 1/√fan_in` (Kaiming) | `lr = base_lr / width_mult` | Scaled down |
| **Output layer** (logits) | `std = 1/√fan_in` (Kaiming) | `lr = base_lr / width_mult` | Scaled down |
| **Output multiplier** | N/A | N/A | Apply `output_mult / width_mult` to logits |
| **Attention scaling** | N/A | N/A | Use `1/d` instead of `1/√d` |

**Key insight:** As width increases, LR decreases proportionally so that update magnitudes stay constant.

**Source:** [microsoft/mup README](https://github.com/microsoft/mup/blob/main/README.md), [How To Scale](https://howtoscalenn.github.io/)

---

### 2.3 Adapting μP for SSM Layers (UNSOLVED)

Our architecture has selective SSM layers with parameters A, B, C, D (discretized SSM). **No existing research covers this.**

#### Hypothesis: Treating SSM Parameters

| Parameter | Shape | Role | Proposed μP Rule |
|-----------|-------|------|------------------|
| **A (dynamics)** | `(d_state, d_state)` | State evolution | **Fixed size** (d_state=16) → `lr = base_lr` (no scaling) |
| **B (input→state)** | `(d_state, d_inner)` | Input projection | `fan_out=d_state (fixed), fan_in=d_inner (width)` → `lr = base_lr / width_mult` |
| **C (state→output)** | `(d_inner, d_state)` | Output projection | `fan_out=d_inner (width), fan_in=d_state (fixed)` → `lr = base_lr / width_mult` |
| **D (skip)** | `(d_inner,)` | Residual connection | Width-scaled → `lr = base_lr / width_mult` |
| **Δ (discretization)** | `(d_inner,)` | Time-step | Width-scaled → `lr = base_lr / width_mult` |

**Rationale:**
- **A is not scaled by width** — it's the core dynamics matrix, size determined by state dimension (fixed at 16)
- **B and C scale with d_inner** — they project between width-dependent hidden state and fixed state dimension
- **D and Δ are per-channel** — scale with d_inner

**Validation needed:** Coordinate check across widths (64, 128, 256, 512) to verify activation norms stay stable.

**Initialization:** Use `mup.init.kaiming_normal_` for B, C after `set_base_shapes` — this should automatically adjust std based on fan_in/fan_out.

---

### 2.4 Adapting μP for Ternary Weights (UNSOLVED)

Ternary weights: `{-1, 0, +1}` (1.58-bit quantization)
Training: Straight-Through Estimator (STE) — forward uses quantized, backward uses continuous

**Challenge:** μP assumes continuous weights. How do initialization and LR scaling apply to quantized weights?

#### Proposed Approach: μP on Continuous, Quantize Forward

```python
# During training
W_continuous = nn.Parameter(torch.randn(fan_out, fan_in))  # μP-scaled init
W_ternary = quantize_ternary(W_continuous)  # {-1, 0, +1} for forward
output = F.linear(input, W_ternary)  # Forward uses quantized
loss.backward()  # Backward updates W_continuous (STE)
```

**μP rules apply to W_continuous:**
- Initialize W_continuous with `std = 1/√fan_in` (Kaiming)
- Use `lr = base_lr / width_mult` for W_continuous updates
- Quantization happens AFTER μP initialization, BEFORE forward pass

**Validation:** Coordinate check on **W_ternary activations** (not W_continuous) — we care about quantized behavior.

**Open question:** Does ternary quantization break μP's infinite-width theory? Quantization introduces discretization noise — does this dominate width scaling effects?

**Related research:** [TernaryLM (Feb 2026)](https://arxiv.org/abs/2602.07374) uses "adaptive per-layer scaling factors" for ternary QAT — might need to combine with μP.

---

### 2.5 Adapting μP for MoD Routing (UNSOLVED)

Mixture-of-Depths (MoD): Ternary router decides per-token whether to apply SSM layer or skip.

**Router architecture:**
```python
router_weight: (d_model,) ternary vector
router_logit = input @ router_weight  # Shape: (batch, seq_len)
decision = router_logit > threshold  # Binary: process or skip
```

**μP questions:**
1. How should `router_weight` be initialized as width changes?
2. How should router LR scale with width?
3. Does router threshold need to be width-adjusted?

#### Hypothesis: Treat Router as Output Layer

In MoE literature, routers are typically treated similarly to output layers (they produce logits).

**Proposed rules:**
- **Initialization:** `std = 1/√d_model` (Kaiming, width-scaled)
- **Learning rate:** `lr = base_lr / width_mult` (same as hidden weights)
- **Threshold:** Keep fixed (e.g., 0.0) — logit scale is controlled by initialization

**Validation:** Measure router decision distribution across widths — should maintain ~50% activation rate regardless of width (if that's the target).

**Related research:** [μP for MoE (Aug 2025)](https://arxiv.org/html/2508.09752v1) derives scaling rules for MoE routers. Key insight: router initialization must account for number of experts AND width. For MoD (binary decision), this simplifies to width-only scaling.

---

### 2.6 Manual Per-Parameter LR with `param_groups`

PyTorch's `param_groups` is the mechanism for per-parameter learning rates.

#### Pattern: Grouping Parameters by μP Role

```python
# After model initialization and set_base_shapes
base_lr = 0.001
width_mult = model.d_model / base_model.d_model  # e.g., 1024/64 = 16

param_groups = [
    # Input embedding: no scaling
    {
        'params': [model.embedding.weight],
        'lr': base_lr,
        'name': 'embedding'
    },

    # Hidden weights (SSM B, C, D, Δ): scaled down
    {
        'params': [
            p for n, p in model.named_parameters()
            if any(x in n for x in ['ssm.B', 'ssm.C', 'ssm.D', 'ssm.delta'])
        ],
        'lr': base_lr / width_mult,
        'name': 'ssm_hidden'
    },

    # SSM A matrix: no scaling (state dimension is fixed)
    {
        'params': [p for n, p in model.named_parameters() if 'ssm.A' in n],
        'lr': base_lr,
        'name': 'ssm_dynamics'
    },

    # Output layer: scaled down
    {
        'params': [model.output.weight],
        'lr': base_lr / width_mult,
        'name': 'output'
    },

    # Router: scaled down (hypothesis)
    {
        'params': [p for n, p in model.named_parameters() if 'router' in n],
        'lr': base_lr / width_mult,
        'name': 'router'
    },
]

optimizer = torch.optim.AdamW(param_groups)
```

**Why NOT use mup.MuAdamW?**
- mup library doesn't know about SSM, ternary, or MoD — it would apply wrong scaling
- Manual `param_groups` gives full control to implement custom rules

**Downside:** Must manually update width_mult if architecture changes. mup's `infshape` tracking automates this.

**Hybrid approach:** Use `set_base_shapes` to attach `infshape`, then manually construct `param_groups` using `p.infshape.width_mult()` for each parameter.

**Source:** [PyTorch AdamW param_groups docs](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html), [PyTorch Forums discussion](https://discuss.pytorch.org/t/implementing-differential-learning-rate-by-parameter-groups/32903)

---

### 2.7 Output Multiplier for Custom Architectures

μP uses an output multiplier to control logit scale: `output = model(x) * (output_mult / width_mult)`

**Why:** As width increases, output variance increases. Dividing by `width_mult` keeps logit scale constant.

**How to implement:**

```python
# Option 1: Manual scaling in forward pass
class CustomModel(nn.Module):
    def __init__(self, d_model, base_d_model=64, output_mult=1.0):
        super().__init__()
        self.width_mult = d_model / base_d_model
        self.output_mult = output_mult
        # ... rest of model

    def forward(self, x):
        logits = self.output_layer(x)
        return logits * (self.output_mult / self.width_mult)

# Option 2: Use mup.MuReadout (if architecture allows)
# Handles scaling internally, but requires standard Linear layer
```

**Tuning `output_mult`:** This is the ONE hyperparameter that still needs tuning (but transfers across widths). Typical range: 0.1 to 10.0. Use coordinate check to verify logit scale stability.

**For SSM outputs:** If using SSM as final layer (instead of Linear), apply same multiplier to SSM output: `y = ssm(x) * (output_mult / width_mult)`

**Source:** [Speechmatics blog on MuP](https://blog.speechmatics.com/mup), [mup README](https://github.com/microsoft/mup/blob/main/README.md)

---

### 2.8 Coordinate Check for Custom Architectures

**Coordinate check = verification that μP is implemented correctly.**

**Process:**
1. Train models at multiple widths (64, 128, 256, 512) for a few steps (e.g., 10 steps)
2. For each width, measure activation L1 norm at each layer: `||activations||_1 / num_elements`
3. Plot activation norm vs. width for each layer
4. **Expected:** Lines should be horizontal (stable across width)
5. **If lines diverge:** μP scaling is wrong for that layer

#### Implementation Pattern

```python
import matplotlib.pyplot as plt

widths = [64, 128, 256, 512]
activation_norms = {w: [] for w in widths}

for width in widths:
    model = build_model(d_model=width)
    apply_mup_init(model, base_width=64)

    # Register hooks to capture activations
    activations = {}
    def hook(name):
        def fn(module, input, output):
            activations[name] = output.detach()
        return fn

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, SSMLayer)):
            module.register_forward_hook(hook(name))

    # Forward pass
    x = torch.randn(32, 128, width)  # (batch, seq_len, d_model)
    model(x)

    # Measure norms
    for name, act in activations.items():
        norm = act.abs().mean().item()
        activation_norms[width].append(norm)

# Plot
plt.figure(figsize=(10, 6))
for width in widths:
    plt.plot(activation_norms[width], label=f'width={width}')
plt.xlabel('Layer')
plt.ylabel('Activation L1 Norm')
plt.legend()
plt.title('Coordinate Check (should be horizontal)')
plt.savefig('coordinate_check.png')
```

**For SSM layers:** Capture BOTH:
- SSM internal state activations (after A, B, C application)
- SSM output activations (what gets passed to next layer)

**For ternary layers:** Measure activations AFTER quantization (forward pass), not before.

**Source:** [EleutherAI Practitioner's Guide](https://blog.eleuther.ai/mutransfer/), [nanoGPT-mup examples](https://github.com/EleutherAI/nanoGPT-mup)

---

## 3. PyTorch-Specific Considerations

### 3.1 `torch.compile` Compatibility

**Question:** Does `torch.compile` break μP's output multipliers?

**Answer:** UNKNOWN — no documented cases as of Feb 2026.

**Potential issues:**
- `torch.compile` optimizes computation graphs → might fuse multiplier into layer weights
- If multiplier is a Python scalar (not a tensor), torch.compile might treat it as constant and inline it

**Testing needed:** Compile a μP model, run coordinate check, verify norms stay stable.

**Known torch.compile + hooks issue:** Forward hooks registered AFTER compilation are ignored ([pytorch#117758](https://github.com/pytorch/pytorch/issues/117758)). If using hooks for coordinate check, register BEFORE compiling.

**Recommendation:** Initially develop without `torch.compile`, validate μP works, THEN compile and re-validate.

**Source:** [torch.compile docs](https://docs.pytorch.org/docs/stable/generated/torch.compile.html), [State of torch.compile Aug 2025](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/)

---

### 3.2 AMP (Autocast) + μP Multipliers

**Question:** Does mixed precision (FP16/BF16) interact with μP scaling?

**Answer:** Likely SAFE, but precision considerations exist.

**How AMP works:**
- `torch.cuda.amp.autocast()` casts operations to FP16/BF16 dynamically
- `GradScaler` prevents underflow by scaling loss (and gradients)

**μP interaction:**
- μP's output multiplier is applied in forward pass → autocast will cast it to FP16/BF16
- LR scaling happens in optimizer (FP32 param updates) → no interaction with autocast

**Potential issue:** If output multiplier is very small (e.g., 0.01) and width_mult is large (e.g., 16), the combined scaling `0.01/16 = 0.000625` might cause FP16 underflow.

**Solution:** Use BF16 (better dynamic range) or keep multipliers in FP32:

```python
with torch.cuda.amp.autocast():
    logits = model(x)  # FP16/BF16

# Force multiplier to FP32
output_mult = torch.tensor(0.01, dtype=torch.float32, device=logits.device)
logits = logits.float() * (output_mult / width_mult)  # FP32 scaling
```

**Known issue:** Gradient checkpointing + autocast + LayerNorm can cause dtype mismatches ([pytorch#49738](https://github.com/pytorch/pytorch/issues/49738)). If using both, test carefully.

**Source:** [PyTorch AMP docs](https://docs.pytorch.org/docs/stable/amp.html), [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)

---

### 3.3 Gradient Checkpointing + μP

**Question:** Does gradient checkpointing break μP?

**Answer:** NO — checkpointing is orthogonal to μP.

**How checkpointing works:**
- Saves only subset of activations during forward
- Recomputes activations during backward (trades compute for memory)

**μP interaction:** None — μP scales initialization and LR, checkpointing only affects memory/compute. Activation norms should be identical with or without checkpointing.

**Caveat:** Checkpointing + AMP + custom hooks can cause issues (see above). Test coordinate check with checkpointing enabled.

**Source:** [PyTorch checkpoint docs](https://docs.pytorch.org/docs/stable/checkpoint.html)

---

### 3.4 Per-Parameter LR: `param_groups` vs. Manual Scaling

**Two approaches to per-parameter LR:**

#### Option 1: `param_groups` (Recommended)

```python
optimizer = torch.optim.AdamW([
    {'params': group1_params, 'lr': 0.001},
    {'params': group2_params, 'lr': 0.0001},
])
```

**Pros:**
- Clean, declarative
- Integrates with LR schedulers (each group can have own schedule)
- Saved in optimizer state_dict

**Cons:**
- Must manually group parameters (can be tedious for large models)

---

#### Option 2: Manual Scaling in Optimizer Step

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# After optimizer.step()
with torch.no_grad():
    for n, p in model.named_parameters():
        if 'hidden' in n:
            p.grad *= (1.0 / width_mult)  # Scale gradient
```

**Pros:**
- Simpler setup (single LR)

**Cons:**
- Modifies gradients directly (fragile)
- Doesn't play well with gradient clipping or LR schedulers
- NOT how μP is typically done

**Verdict:** Use `param_groups`. It's the PyTorch-native way and integrates cleanly.

**Source:** [PyTorch optimizer docs](https://docs.pytorch.org/docs/stable/optim.html), [PyTorch Forums](https://discuss.pytorch.org/t/optimizers-good-practices-for-handling-multiple-param-groups/79642)

---

## 4. RTX 5070 Blackwell Considerations

### 4.1 FP4 Native Support

**RTX 5070 specs:**
- 5th-gen Tensor Cores
- Native FP4 (NVFP4), FP6, FP8, BF16, FP16, TF32 support
- FP4 uses dual-level scaling: FP8 micro-blocks + FP32 tensor scale → <1% accuracy loss
- FP4 throughput: 2× FP8 throughput

**μP opportunity:** Train proxy model (width=64) in FP4, transfer HPs to full model (width=1024) in FP8/BF16.

**Current support (Feb 2026):**
- PyTorch 2.11 nightly + CUDA 12.8 supports Blackwell (sm_120)
- Stable PyTorch 2.10 does NOT support sm_120 yet
- TensorRT-LLM supports FP4 for inference (post-training quantization)
- **NO QAT (quantization-aware training) for FP4 in PyTorch yet**

**Feasibility:** Medium-term (6-12 months). Wait for PyTorch stable support + QAT tooling.

**Alternative:** Use FP8 for proxy training (already supported in PyTorch 2.10 with `torch.float8_e4m3fn`, `torch.float8_e5m2`).

**Source:** [RTX 5070 AI benchmarks](https://www.bestgpusforai.com/gpu-comparison/5070-vs-5080), [NVIDIA Blackwell FP4 blog](https://www.edge-ai-vision.com/2025/10/nvidia-blackwell-the-impact-of-nvfp4-for-llm-inference/), [Blackwell GPU Architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)

---

### 4.2 FP8 Training + μP

**FP8 support in PyTorch 2.10:**
- `torch.float8_e4m3fn` (e4m3: 4-bit exponent, 3-bit mantissa, better for forward)
- `torch.float8_e5m2` (e5m2: 5-bit exponent, 2-bit mantissa, better for backward)
- Supported on Blackwell, Hopper (H100), Ada (RTX 4090)

**u-μP advantage:** Designed for low-precision — ensures activations/gradients stay in FP8 range (unit scaling).

**Workflow:**
1. Train proxy model (width=64) in FP8 with u-μP
2. Measure optimal HPs (base_lr, weight_decay, warmup, etc.)
3. Transfer HPs to full model (width=1024)
4. Train full model in BF16/FP8

**Implementation:** Use Graphcore's `unit-scaling` library + manual FP8 casting:

```python
import unit_scaling as uscale

# Wrap layers with unit-scaled versions
hidden = uscale.functional.linear(x, W_fp8, bias)

# Train with AMP
with torch.cuda.amp.autocast(dtype=torch.float8_e4m3fn):
    output = model(x)
```

**Caveat:** `unit-scaling` library is beta (as of Feb 2026) — expect bugs.

**Alternative:** Standard μP in BF16, then quantize to FP8 post-training (loses some accuracy but safer).

**Source:** [PyTorch FP8 support](https://docs.pytorch.org/docs/stable/amp.html), [u-μP library](https://github.com/graphcore-research/unit-scaling), [PyTorch Hardware Acceleration 2025](https://tunguz.github.io/PyTorch_Hardware_2025/)

---

### 4.3 Hardware-Specific Optimizations for μP Scaling Operations

**Question:** Can Blackwell accelerate μP's scaling operations (width_mult divisions, output multipliers)?

**Answer:** Minimal impact — scaling is element-wise (cheap), bottleneck is matmul/conv.

**Scaling operation:** `output = logits * (output_mult / width_mult)` → single scalar multiply per element
- On GPU: fuses into prior operation (kernel fusion)
- Cost: negligible compared to matrix multiply

**Where Blackwell helps:**
- **Matmuls in FP8/FP4:** 2-4× faster than FP16
- **Memory bandwidth:** 672 GB/s GDDR7 → faster activation loading for SSM scans

**μP itself adds ~0.1% overhead** (a few scalar multiplies). The real benefit is HP transfer, not runtime performance.

**Source:** [NVIDIA Blackwell whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)

---

## 5. Gaps in Tooling and Research

### 5.1 What's Missing from μP Ecosystem

| Feature | Status | Impact |
|---------|--------|--------|
| **SSM layer scaling rules** | Not researched | HIGH — critical for our architecture |
| **Quantized (ternary) weight μP** | Not researched | HIGH — ternary is core to our design |
| **MoD router scaling** | Partial (MoE paper exists, no MoD-specific) | MEDIUM — can adapt from MoE |
| **u-μP for SSM** | Not researched | MEDIUM — would improve FP8 proxy training |
| **FP4 QAT + μP** | Not researched | LOW — FP4 QAT doesn't exist yet |
| **Coordinate check for non-Transformer** | No tooling | MEDIUM — must implement ourselves |
| **μP + gradient checkpointing validation** | Not documented | LOW — likely works, needs testing |

---

### 5.2 Opportunity: μP Extension for SSM Architectures

**What:** Open-source library/paper extending μP to selective SSMs (Mamba, custom SSMs)

**Why:** Mamba is increasingly popular, but no scaling laws or μP rules exist for it

**What it would include:**
- Theoretical derivation of μP rules for SSM parameters (A, B, C, D, Δ)
- Coordinate check implementation for SSM layers
- Reference implementation (mup-ssm library?)
- Validation on Mamba-2 and custom SSMs

**Who would use it:** Anyone training SSM models at scale (our use case, plus research community)

**Effort:** Medium — requires mathematical derivation + empirical validation

**Impact:** High — could become standard for SSM training

---

### 5.3 μP + Quantization-Aware Training (QAT)

**Current state (Feb 2026):**
- QAT research is exploding (EfficientQAT, TernaryLM, PT-BitNet)
- μP research exists (standard, u-μP, CompleteP)
- **ZERO overlap** — no one has combined them

**Why it matters:**
- QAT changes gradient flow (STE, fake quantization)
- μP changes update magnitudes (LR scaling)
- Interaction is unknown — do they interfere? complement?

**Open questions:**
1. Does ternary quantization noise break μP's infinite-width assumptions?
2. Do STE gradients require different LR scaling than standard gradients?
3. Can u-μP's unit scaling improve QAT stability in low precision?

**Opportunity:** Research paper + implementation combining μP and QAT for ternary/FP8 training

**Effort:** High — requires theoretical work + extensive experiments

**Impact:** High — ternary LLMs (BitNet, TernaryLM) could benefit enormously from HP transfer

---

### 5.4 What We Need to Build Ourselves

**Immediate needs (for our architecture):**

1. **SSM μP scaling rules**
   - Derive (or empirically test) LR scaling for A, B, C, D, Δ
   - Implement coordinate check for SSM layers
   - Validate on widths 64, 128, 256, 512

2. **Ternary + μP integration**
   - Decide: apply μP to continuous weights, then quantize? Or modify μP for discrete weights?
   - Test coordinate check on ternary activations
   - Validate HP transfer holds with quantization

3. **MoD router scaling**
   - Adapt MoE μP rules to binary MoD routing
   - Test router decision stability across widths

4. **Custom coordinate check tool**
   - Script to train multiple widths, hook activations, plot norms
   - SSM-aware (capture state evolution, not just outputs)
   - Ternary-aware (measure quantized activations)

**Long-term (if successful):**
- Open-source `mup-ssm` library
- Research paper: "Scaling Laws for Ternary State-Space Models"
- Contribute back to μP ecosystem

---

## 6. Practical Recommendations

### 6.1 Phase 1: Validate μP on Standard Layers (2-3 days)

**Goal:** Verify we can implement μP correctly before tackling custom layers

**Steps:**
1. Build toy Transformer (standard Linear layers, no SSM)
2. Use Microsoft `mup` library as-is
3. Run coordinate check across widths 64, 128, 256
4. Verify activation norms are horizontal
5. Validate HP transfer (tune width=64, apply to width=256, check loss curves match)

**Why:** De-risk μP implementation — if we can't get standard layers working, custom layers will be worse.

**Resources:** [nanoGPT-mup examples](https://github.com/EleutherAI/nanoGPT-mup), [EleutherAI guide](https://blog.eleuther.ai/mutransfer/)

---

### 6.2 Phase 2: DIY μP for SSM (1-2 weeks)

**Goal:** Derive and validate μP rules for SSM layers

**Steps:**
1. **Hypothesis formation:** Write down proposed LR scaling rules (see Section 2.3)
2. **Implementation:** Manual `param_groups` with width-based LR scaling
3. **Coordinate check:** Train widths 64, 128, 256, 512 for 100 steps each
4. **Iteration:** If norms diverge, adjust scaling rules and re-test
5. **Validation:** Once norms are stable, run full HP transfer test (tune at 64, apply to 512)

**Key metric:** Loss curves for width=512 with transferred HPs should match loss curves for width=512 with directly-tuned HPs (within ~5%)

**Fallback:** If μP doesn't work for SSM, fall back to standard scaling (LR ∝ 1/√width) — less optimal but proven

---

### 6.3 Phase 3: Integrate Ternary Quantization (1 week)

**Goal:** Verify μP works with quantized forward pass

**Steps:**
1. Add ternary quantization to forward pass (STE for backward)
2. Apply μP initialization to continuous weights (before quantization)
3. Run coordinate check on QUANTIZED activations
4. Adjust if needed (e.g., scale threshold if quantization shifts distribution)

**Expected:** Quantization adds noise but shouldn't break μP — norms should still be stable (maybe with higher variance)

**Fallback:** If quantization breaks μP, consider hybrid approach (train proxy in BF16, quantize only for full model)

---

### 6.4 Phase 4: Explore u-μP for FP8 Proxy Training (1-2 weeks, optional)

**Goal:** Leverage RTX 5070's FP8 support for faster proxy training

**Prerequisites:**
- PyTorch stable support for Blackwell (or use nightly)
- u-μP library tested and debugged

**Steps:**
1. Install `unit-scaling` library
2. Wrap SSM layers with unit-scaled equivalents (may need custom implementation)
3. Train proxy (width=64) in FP8 with u-μP
4. Transfer HPs to full model (width=512) in BF16
5. Compare to standard μP (BF16 proxy) — is transfer quality equivalent?

**Expected benefit:** 2-4× faster proxy training (FP8 vs BF16), no quality loss if u-μP works correctly

**Risk:** Beta software (unit-scaling), SSM support unproven

---

### 6.5 Windows 11 + CUDA 12.8 + mup Library Compatibility

**Current status:**
- PyTorch 2.10 + CUDA 12.8 wheels available for Windows
- Microsoft `mup` library is pure Python (no compiled components) → platform-independent
- Blackwell (sm_120) support requires PyTorch 2.11 nightly OR wait for 2.12 stable (Q2 2026?)

**Setup for immediate work (BF16/FP16):**
```bash
# Windows 11, stable PyTorch (no Blackwell-specific features)
pip install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install mup  # Latest version from PyPI
```

**Setup for Blackwell FP8/FP4 experiments:**
```bash
# Requires nightly PyTorch
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install mup
```

**Known issues:**
- Blackwell support in PyTorch is new (Jan 2026) — expect bugs
- Monitor CUDA out-of-memory more carefully (Blackwell memory management differs from Ampere/Ada)

**Source:** [PyTorch Windows install guide](https://pytorch.org/get-started/locally/), [NVIDIA Blackwell software migration guide](https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus-a-guide-to-cuda-12-8-pytorch-tensorrt-and-llama-cpp/321330)

---

## 7. Summary and Next Steps

### TL;DR

| Question | Answer |
|----------|--------|
| **Can we use existing μP libraries?** | Partially — `mup` works for standard layers, but NO support for SSM/ternary/MoD |
| **DIY μP feasible?** | YES — PyTorch `param_groups` + manual scaling rules + coordinate check |
| **u-μP for FP8?** | Promising but immature (beta software, no SSM support) |
| **RTX 5070 FP4 support?** | Hardware ready, software NOT (FP4 QAT doesn't exist yet) |
| **Biggest gap?** | SSM width scaling rules — ZERO existing research |

### Recommended Path Forward

1. **Week 1:** Validate standard μP (Transformer toy model)
2. **Weeks 2-3:** DIY μP for SSM (derive rules, coordinate check, iterate)
3. **Week 4:** Integrate ternary quantization + validate
4. **Week 5+:** (Optional) Experiment with u-μP + FP8

### What We're Building Ourselves

- **SSM μP scaling rules** (LR, initialization for A, B, C, D, Δ)
- **Ternary + μP integration pattern**
- **MoD router scaling rules**
- **Custom coordinate check tool** (SSM + ternary aware)

### Open Research Questions

1. Do ternary weights break μP's infinite-width assumptions?
2. What's the optimal LR scaling for selective SSM parameters?
3. Can u-μP improve ternary QAT stability in FP8?

---

## Sources

### Primary Libraries
- [Microsoft mup](https://github.com/microsoft/mup)
- [EleutherAI nanoGPT-mup](https://github.com/EleutherAI/nanoGPT-mup)
- [Graphcore unit-scaling (u-μP)](https://github.com/graphcore-research/unit-scaling)

### Key Papers and Blogs
- [The Practitioner's Guide to μP (EleutherAI, Sept 2024)](https://blog.eleuther.ai/mutransfer/)
- [μTransfer: Hyperparameter Tuning of Enormous Neural Networks (Microsoft Research)](https://www.microsoft.com/en-us/research/blog/%C2%B5transfer-a-technique-for-hyperparameter-tuning-of-enormous-neural-networks/)
- [μ-Parametrization for Mixture of Experts (Aug 2025)](https://arxiv.org/html/2508.09752v1)
- [Hyperparameter Transfer with MoE Layers (Jan 2026)](https://arxiv.org/html/2601.20205)
- [TernaryLM (Feb 2026)](https://arxiv.org/abs/2602.07374)
- [Reduce Model Tuning Costs with MuP (Speechmatics)](https://blog.speechmatics.com/mup)

### PyTorch Documentation
- [torch.optim (param_groups)](https://docs.pytorch.org/docs/stable/optim.html)
- [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/amp.html)
- [torch.nn.init (Kaiming, Xavier)](https://docs.pytorch.org/docs/stable/nn.init.html)

### Hardware and Compatibility
- [RTX 5070 vs 5080 for AI (2026)](https://www.bestgpusforai.com/gpu-comparison/5070-vs-5080)
- [NVIDIA Blackwell FP4 for LLM Inference](https://www.edge-ai-vision.com/2025/10/nvidia-blackwell-the-impact-of-nvfp4-for-llm-inference/)
- [NVIDIA Blackwell GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
- [Blackwell Software Migration Guide (NVIDIA Forums)](https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus-a-guide-to-cuda-12-8-pytorch-tensorrt-and-llama-cpp/321330)
- [State of PyTorch Hardware Acceleration 2025](https://tunguz.github.io/PyTorch_Hardware_2025/)

### Quantization Research
- [PyTorch QAT Guide](https://pytorch.org/blog/quantization-aware-training/)
- [EfficientQAT for Large Language Models (2025)](https://openreview.net/forum?id=6Mdvq0bPyG)
- [PT-BitNet: Scaling 1-Bit LLMs (July 2025)](https://www.sciencedirect.com/science/article/abs/pii/S089360802500735X)

---

**End of Research Report**
**Next:** Phase 1 — Validate standard μP, then derive SSM scaling rules.
