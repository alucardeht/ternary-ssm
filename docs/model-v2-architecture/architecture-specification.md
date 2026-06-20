# Architecture Specification: Model v2

## Overview

A hybrid SSM-Attention language model with ternary weights, dynamic routing, and mixed precision. Built from first principles, not a copy of any existing implementation.

**Name:** TBD (to be decided after validation)

---

## 1. High-Level Architecture

```
Input tokens
    ↓
[Embedding Layer] (FP16, tied input/output)
    ↓
[RMSNorm]
    ↓
┌─────────────────────────────────────────┐
│  × N layers (mostly SSM, some SWA)     │
│                                         │
│  LayerNorm(x)                          │
│      ↓                                  │
│  [SSM Block] or [SWA Block + MoD]      │
│      ↓                                  │
│  + residual (from original x)          │
└─────────────────────────────────────────┘
    ↓
[RMSNorm]
    ↓
[Output Head] (tied with embedding, FP16)
    ↓
Logits
```

### Layer Composition

For a model with L total layers:
- **SSM layers:** ~87.5% (7 out of every 8)
- **SWA layers:** ~12.5% (1 out of every 8)
- **Placement:** SWA at strategic positions (evenly spaced)

| Total Layers | SSM Layers | SWA Layers | SWA Positions |
|--------------|------------|------------|---------------|
| 8            | 7          | 1          | 4             |
| 16           | 14         | 2          | 5, 11         |
| 24           | 21         | 3          | 6, 12, 18     |
| 32           | 28         | 4          | 8, 16, 24, 31 |
| 48           | 42         | 6          | 8, 16, 24, 32, 40, 47 |
| 64           | 56         | 8          | 8, 16, 24, 32, 40, 48, 56, 63 |

---

## 2. SSM Block Design (SSD/Mamba-2 Style)

Each SSM block follows selective state space model design with ternary projections.

### Data Flow

```
x (d_model, FP16 activations)
    ↓
[RMSNorm] (FP16 params)
    ↓
[in_proj: TERNARY] x → (z, x_inner)  // split into gate and input
    ↓
[Conv1d] (FP16, kernel=4) on x_inner  // local context
    ↓
[SiLU activation]
    ↓
┌──────────────────────────────────────┐
│ Selective Scan (SSM Core)            │
│                                      │
│ x_proj: TERNARY → (Δ_input, B, C)   │
│ dt_proj: FP16 → Δ (discretization)  │
│ A: FP16 diagonal (eigenvalues)       │
│ D: FP16 (skip connection)            │
│                                      │
│ Discretize:                          │
│   Ā = exp(Δ · A)                     │
│   B̄ = Δ · B                          │
│                                      │
│ Scan:                                │
│   h_t = Ā · h_{t-1} + B̄ · x_t       │
│   y_t = C · h_t + D · x_t           │
└──────────────────────────────────────┘
    ↓
[gate: SiLU(z) ⊙ y]  // gated output
    ↓
[out_proj: TERNARY]
    ↓
+ x (residual from BEFORE RMSNorm)
    ↓
output
```

### Precision Map (SSM Block)

| Component | Precision | Rationale |
|-----------|-----------|-----------|
| in_proj | **TERNARY** | Largest layer, prime candidate (~40% params) |
| out_proj | **TERNARY** | Second largest (~20% params) |
| x_proj | **TERNARY** | Projection for B, C, Δ input (~10% params) |
| dt_proj | **FP16** | Discretization sensitive, small layer |
| A matrix | **FP16** | Eigenvalue stability, ≥8-bit mandatory |
| D vector | **FP16** | Skip connection, critical for stability |
| Conv1d | **FP16** | Small, local patterns matter |
| RMSNorm | **FP16** | Normalization precision critical |

---

## 3. Sliding Window Attention (SWA) Block Design

For the ~12.5% of layers that handle retrieval.

### Data Flow

```
x (d_model, FP16 activations)
    ↓
[RMSNorm] (FP16 params)
    ↓
┌──────────────────────────────────────────────────┐
│ Router Decision (FP16 params)                    │
│   Same mechanism as SSM blocks                   │
└──────────────────────────────────────────────────┘
    ↓ (if processing)
[Q_proj: TERNARY] x_norm → Q
[K_proj: TERNARY] x_norm → K
[V_proj: TERNARY] x_norm → V
    ↓
[RoPE positional encoding] on Q, K (FP16)
    ↓
[Sliding Window Attention]
    window_size = 128 tokens
    scores = (Q · K^T) / √d_head    (computed in FP16)
    weights = softmax(scores)         (FP16)
    output = weights · V              (FP16)
    ↓
[out_proj: TERNARY]
    ↓
[MLP block]
    [up_proj: TERNARY] → SiLU → [gate_proj: TERNARY]
    [down_proj: TERNARY]
    ↓
+ x (residual from BEFORE RMSNorm)
    ↓
output
```

### Precision Map (SWA Block)

| Component | Precision | Rationale |
|-----------|-----------|-----------|
| Q, K, V proj | **TERNARY** | Large projections, ternary-friendly |
| Attention scores | **FP16** | Precision critical for retrieval quality |
| RoPE | **FP16** | Positional encoding needs precision |
| out_proj | **TERNARY** | Standard projection |
| MLP up/gate/down | **TERNARY** | Standard MLP layers |
| RMSNorm | **FP16** | Normalization |
| Router | **FP16** | Same as SSM blocks |

---

## 4. Dynamic Routing (Mixture of Depths)

### Mechanism

Router applies to **SWA layers only** (not SSM layers). Per-token expert-choice routing using top-k selection with Straight-Through Estimator (STE).

```
Router (SWA layers only):
    score = softmax(W_router · LayerNorm(x)) × capacity
    W_router: d_model → n_swa_experts (FP16, NOT ternary)
    capacity: 0.5 (process ~50% tokens per layer)

Training:
    Expert-choice top-k (Raposo et al. 2024) with STE
    select k experts per token: top_k(scores, k=int(n_tokens × capacity))
    Straight-through gradient: pass raw scores through backward

Inference:
    Hard expert-choice: top_k(scores, k=int(n_tokens × capacity))

Skip path:
    x_out = x_in  (pure residual, zero compute)

Process path:
    x_out = x_in + Block_expert(LayerNorm(x_in))
```

### Auxiliary Losses

```
L_total = L_main (no auxiliary losses with expert-choice routing)

Target: ~50% tokens processed per SWA layer
Monitor: Per-layer load balance, gradient flow through skip paths
```

### Expected Compute Savings

| Token Difficulty | % of Tokens | Layers Used | Compute vs Full |
|------------------|-------------|-------------|-----------------|
| Easy             | ~20%        | ~25%        | 0.25×           |
| Moderate         | ~60%        | ~50%        | 0.50×           |
| Hard             | ~20%        | ~100%       | 1.00×           |
| **Weighted avg** |             |             | **~0.51×**      |

---

## 5. Model Configurations by Scale

### Validation Scale (1M params) — Minutes to Train

```
d_model:        256
n_layer:        8 (7 SSM + 1 SWA at position 4)
d_state:        16
d_conv:         4
expand:         2
d_inner:        512 (= expand × d_model)
dt_rank:        16 (= d_model / 16)
n_heads_swa:    4 (for SWA layers, head_dim = 64)
window_size:    128
vocab_size:     8,192
seq_len:        512
tied_embeddings: yes

Estimated params: ~1.2M
Memory (weights): ~0.4 MB (mixed precision)
Training time:   <5 min on RTX 5070
Purpose:         Quick iteration, component testing
```

### Experiment Scale (10M params) — 15-30 Min to Train

```
d_model:        512
n_layer:        16 (14 SSM + 2 SWA at positions 5, 11)
d_state:        16
d_conv:         4
expand:         2
d_inner:        1024
dt_rank:        32
n_heads_swa:    8
window_size:    128
vocab_size:     16,384
seq_len:        1024
tied_embeddings: yes

Estimated params: ~12M
Memory (weights): ~4 MB (mixed precision)
Training time:   15-30 min on RTX 5070
Purpose:         Ternary validation, precision floor testing
```

### Proof of Concept (100M params)

```
d_model:        768
n_layer:        24 (21 SSM + 3 SWA at positions 6, 12, 18)
d_state:        16
d_conv:         4
expand:         2
d_inner:        1536
dt_rank:        48
n_heads_swa:    12
window_size:    128
vocab_size:     32,000
seq_len:        2048
tied_embeddings: yes

Estimated params: ~100M
Memory (weights): ~38 MB (mixed precision)
Memory (training): ~600 MB
Training time:   Hours on RTX 5070
Purpose:         Quality validation vs FP16 baseline
```

### Medium Scale (370M params)

```
d_model:        1024
n_layer:        48 (42 SSM + 6 SWA at positions 8, 16, 24, 32, 40, 47)
d_state:        64 (Mamba-2 style)
d_conv:         4
expand:         2
d_inner:        2048
dt_rank:        64
n_heads_swa:    16
window_size:    128
vocab_size:     32,000
seq_len:        2048
tied_embeddings: yes

Estimated params: ~370M
Memory (weights): ~141 MB (mixed precision)
Memory (training): ~1.7 GB
Purpose:         Pre-production validation, benchmark comparisons
```

### Target Scale (1B params)

```
d_model:        2048
n_layer:        48 (42 SSM + 6 SWA)
d_state:        64
d_conv:         4
expand:         2
d_inner:        4096
dt_rank:        128
n_heads_swa:    32
window_size:    128
vocab_size:     50,000
seq_len:        2048
tied_embeddings: yes

Estimated params: ~1B
Memory (weights): ~380 MB (mixed precision) vs 2 GB FP16
Memory (training): ~3.5 GB (fits RTX 5070)
Purpose:         Production-viable model
```

### Full Scale (7B equivalent)

```
d_model:        4096
n_layer:        64 (56 SSM + 8 SWA)
d_state:        128
d_conv:         4
expand:         2
d_inner:        8192
dt_rank:        256
n_heads_swa:    32
window_size:    128
vocab_size:     50,000
seq_len:        2048
tied_embeddings: yes

Estimated params: ~7B
Memory (weights): ~2.66 GB (mixed precision) vs 14 GB FP16
Memory (inference): ~3 GB (CPU-viable with 32GB RAM)
Purpose:         Final production model
```

---

## 6. Mixed Precision Strategy

### Parameter Distribution

| Component | % of Total Params | Precision | Rationale |
|-----------|-------------------|-----------|-----------|
| in_proj (all layers) | ~35% | TERNARY | Largest, add/subtract/skip |
| out_proj (all layers) | ~18% | TERNARY | Second largest |
| x_proj (SSM layers) | ~8% | TERNARY | B/C/Δ input projection |
| MLP (SWA layers) | ~15% | TERNARY | Standard MLP |
| Q/K/V proj (SWA layers) | ~5% | TERNARY | Attention projections |
| **Subtotal ternary** | **~81%** | | |
| Embeddings (tied) | ~8% | FP16 | Input/output precision |
| A matrices | ~4% | FP16 | Eigenvalue stability |
| dt_proj, Conv1d, D | ~3% | FP16 | SSM-critical |
| RMSNorm (all layers) | ~1% | FP16 | Normalization |
| Router (all layers) | <1% | FP16 | Routing precision |
| Attention scores | 0% (runtime) | FP16 | Computed, not stored |
| **Subtotal FP16** | **~16%** | | |
| **Subtotal TERNARY** | **~84%** | | |

### Compression Ratio

```
FP16 baseline:  params × 2 bytes = 14 GB for 7B
Mixed precision: 0.84 × params × 0.2 + 0.16 × params × 2
               = params × (0.168 + 0.32)
               = params × 0.488 bytes
               = 3.4 GB for 7B

Compression: 14 GB → 3.4 GB = 4.1× reduction
```

*Note: Practical compression may be ~3.5-4× due to alignment and metadata overhead.*

---

## 7. Initialization Strategy

### SSM Components

| Component | Initialization |
|-----------|---------------|
| A matrix (diagonal) | `A = -torch.linspace(1, d_state, d_state)` (HiPPO-inspired, log-spaced) |
| B matrix | Zero init (learned from data) |
| C matrix | Zero init (learned from data) |
| D vector | Ones (identity skip connection) |
| dt_bias | `torch.linspace(log(dt_min), log(dt_max), d_inner)` (log-uniform, dt_min=0.001, dt_max=0.1) |

### Projection Layers (Ternary)

Ternary layers (direct training from step 0):
- **in_proj, out_proj:** Reduced Xavier uniform (gain=1/√n), scaled by 1/√d_model
- **x_proj:** Reduced Xavier uniform (gain=1/√n), scaled by 1/√d_inner
- Group quantization active from step 0: groups of 128 params share scaling factor
- `w_ternary = s · sign(w) · round(clip(|w|/s, 0, 1))` where s = max(|w_group|)
- Adaptive TWN threshold: 0.7 × mean(|W_normalized|) per Experiment 2d

### Attention Components (SWA Layers)

| Component | Initialization |
|-----------|---------------|
| Q, K, V proj | Xavier uniform, scaled by 1/√d_model |
| out_proj | Xavier uniform, scaled by 1/√d_model |
| MLP up/down/gate | Xavier uniform with gain=√(2/3) |

### Router

| Component | Initialization |
|-----------|---------------|
| W_router | Xavier uniform, small scale (0.01) |
| Bias | 0.5 (start with ~50% process probability) |

---

## 8. Training Recipe (Direct Ternary)

Direct ternary training from step 0 — no FP16 warmup phases needed. Validated by B.1 (6M, within 0.2% of progressive) and B.2 (50M, zero instability, perfectly monotonic convergence).

### Training Configuration

```
All ternary-eligible weights: Ternary from step 0 (STE + group quantization)
Keep FP16: A, dt_proj, Conv1d, D, embeddings, norms, router
MoD routing on SWA layers: Active from step 0
Optimizer: AdamW(β₁=0.9, β₂=0.95, ε=1e-8, weight_decay=0.1), FP32 accumulation
LR schedule: WSD (Warmup-Stable-Decay)
Peak LR: 6e-4
Gradient clipping: 1.0 global

Differential LRs:
  Ternary weights: 1.0× peak (6e-4)
  SSM B, C, Δ:    1.0× peak (6e-4)
  SSM A matrix:    0.1× peak (6e-5)
  Router (SWA):    1.0× peak (6e-4)
  Embeddings:      0.5× peak (3e-4)

Monitoring (every 100 steps):
  - Per-layer gradient norms
  - Ternary weight distribution {-1, 0, +1}
  - Router entropy (SWA layers)
  - A eigenvalues

Convergence indicators:
  - Loss decreasing monotonically
  - Ternary distribution ~33% each value (±5%)
  - Gradient norms in [1e-3, 10]
  - Router MeanW increasing toward 0.9+
```

### Batch Sizes by Scale

| Scale | Batch Size | Grad Accum | Effective Batch |
|-------|------------|------------|-----------------|
| 1M   | 256        | 1          | 256             |
| 10M  | 128        | 2          | 256             |
| 100M | 64         | 4          | 256             |
| 370M | 32         | 8          | 256             |
| 1B   | 16         | 16         | 256             |

*Note: Larger effective batch sizes (4-8x) recommended for models with >80% ternary to reduce STE gradient variance.*

---

## 9. Ternary Quantization Details

### Group Quantization

```
Group size: 128 parameters
Scaling: absmax per group

Forward:
    s = max(|w_group|)
    w_scaled = w / s
    w_ternary = sign(w_scaled) · round(clip(|w_scaled|, 0, 1))
    output = s · w_ternary · x  (effectively: s × (add/subtract/skip))

Backward (STE):
    ∂L/∂w = ∂L/∂w_ternary  (pass through, ignore quantization)

Storage:
    Weights: 2 bits per param (encode {-1, 0, +1})
    Scales: 1 FP16 per group (1 per 128 params)
    Overhead: 2/128 = 1.6% from scales
```

### Inference Operations

For ternary weight w ∈ {-1, 0, +1} and FP16 activation x:
```
w = +1: output += s × x   (add)
w =  0: output += 0       (skip)
w = -1: output -= s × x   (subtract)

NO multiply needed for 84% of parameters.
Only multiply needed: FP16 components (A matrix, dt_proj, attention scores)
```

---

## 10. Monitoring System

### Dashboard Metrics (Every N Steps)

| Metric | Frequency | Healthy Range | Red Flag |
|--------|-----------|---------------|----------|
| Training loss | Every step | Decreasing | Spike > 2× recent avg |
| Per-layer gradient norm | Every 100 | [1e-3, 10] | Outside range |
| Router entropy | Every 100 | [0.5·log(K), log(K)] | < 0.3·log(K) |
| Ternary distribution | Every 500 | ~33% each ±7% | >45% any value |
| A eigenvalues max\|λ\| | Every 500 | [0.9, 0.99] | >0.995 or <0.8 |
| State norm \|\|h_t\|\| | Every 100 | < 10 | > 100 |
| Tokens processed/layer | Every 100 | 40-75% | <20% or >90% |
| Validation perplexity | Every 1K | Within 15% of FP baseline | >25% gap |

### Automated Responses

```
Gradient norm spike (>10× avg):
    → Reduce LR 50%
    → Increase per-layer clipping
    → Log diagnostic dump

Router collapse (entropy < 0.3·log(K)):
    → Increase router LR 2×
    → Increase balance loss α: 0.01 → 0.05
    → Add entropy bonus β: 0.001 → 0.01

State explosion (||h_t|| > 100):
    → Clip A eigenvalues to [0.8, 0.99]
    → Reduce A LR to 0.01×
    → Checkpoint + investigate

Loss spike (>3× recent avg):
    → Revert to last checkpoint
    → Reduce global LR 50%
    → Increase batch size if possible
```

---

## 11. Inference Characteristics

### Memory Footprint (Inference Only)

| Scale | Weights | SSM State | SWA KV Cache | Total |
|-------|---------|-----------|--------------|-------|
| 1M   | 0.4 MB  | <1 MB     | <1 MB        | ~2 MB |
| 10M  | 4 MB    | ~1 MB     | ~1 MB        | ~6 MB |
| 100M | 38 MB   | ~5 MB     | ~3 MB        | ~46 MB |
| 1B   | 380 MB  | ~25 MB    | ~12 MB       | ~417 MB |
| 7B   | 2.66 GB | ~130 MB   | ~65 MB       | ~2.9 GB |

*SSM State: d_inner × d_state × n_ssm_layers × dtype*
*SWA KV Cache: 2 × window × d_model × n_swa_layers × dtype (128 tokens)*

### CPU Inference Viability

7B model at 2.9 GB fits comfortably in 32 GB RAM.

**Operations breakdown:**
- ~84% of ops: add/subtract/skip (ternary, CPU-friendly)
- ~12% of ops: FP16 multiply (A matrix, dt_proj, attention scores)
- ~4% of ops: activation functions (SiLU, softmax, exp)

**Expected throughput:** TBD after implementation (depends on ternary kernel optimization)

---

## 12. Key Design Decisions Summary

| Decision | Choice | Why |
|----------|--------|-----|
| SSM variant | SSD/Mamba-2 (parallel scan) | 13.2x speedup, +0.01 loss gap vs sequential (Exp 3.5) |
| Attention type | Sliding window (128 tokens) | Minimal overhead, proven with SSM |
| Attention ratio | ~12.5% (1:7) | Industry standard (Jamba, Hymba) |
| Quantization | Group ternary (groups of 128) | 40% less error than naive ternary |
| Routing | Top-k MoD on SWA layers only (STE) | SSM incompatible with MoD; top-k per Raposo et al. 2024 |
| Normalization | Pre-RMSNorm | Most stable for quantized models |
| Embeddings | Tied, FP16 | Reduce params, preserve quality |
| Optimizer | AdamW with differential LRs | Essential for mixed-precision stability |
| Training | Direct ternary (no warmup phases) | B.1/B.2 validated; progressive unnecessary |
| Residual | Additive, bypass all discrete ops | Gradient highway, non-negotiable |

---

## 13. What Makes This Architecture Novel

1. **Ternary SSM:** No published work has combined group-ternary weights with Mamba-style selective scan at this level of integration (Bi-Mamba used binary, not group-ternary)
2. **Triple combination:** Ternary + SSM + MoD routing together is unexplored territory
3. **Mixed precision design:** Specifically optimized precision map for SSM components (A=FP16, projections=ternary)
4. **Sliding window + SSM hybrid with ternary:** Attention projections also ternary, only scores stay FP16
5. **From scratch:** Understanding principles, not copying implementation

### What's Proven vs What's Novel

| Component | Status |
|-----------|--------|
| Ternary LLMs (BitNet) | ✅ Proven at 3B+ |
| SSM language models (Mamba) | ✅ Proven at 2.8B |
| SSM + attention hybrid (Jamba) | ✅ Proven at 398B |
| MoD routing | ✅ Proven with Transformers |
| Binary SSM (Bi-Mamba) | ✅ Proven at 2.7B |
| Group-ternary SSM | ⚠️ Our novel contribution |
| Ternary + SSM + MoD combined | ⚠️ Our novel contribution |
| Ternary attention projections in hybrid | ⚠️ Our novel contribution |

---

## 14. Experiment Roadmap (Phase 3)

### Experiment 1: Baseline SSM (1M params)
- Pure SSM, all FP16
- Validate training pipeline works
- Establish quality ceiling
- **Task:** Copy task, next-token prediction on TinyStories

### Experiment 2: Ternary Projections (1M params)
- Same model, ternary in_proj/out_proj
- Keep everything else FP16
- Measure quality delta
- **Task:** Same as Exp 1, compare

### Experiment 3: Full Ternary (10M params)
- All ternary-eligible layers quantized
- Mixed precision as specified
- Direct ternary training
- **Task:** TinyStories + wikitext

### Experiment 4: Add SWA (10M params)
- Hybrid model (SSM + SWA layers)
- Ternary + mixed precision
- Test retrieval capability
- **Task:** Copy task, passkey retrieval, language modeling

### Experiment 5: Add Routing (10M params)
- Full architecture (SSM + SWA + MoD)
- All ternary + mixed precision + routing
- Validate combined stability
- **Task:** Full evaluation suite

### Experiment 6: Scale Test (100M params)
- Full architecture at 100M
- Compare against FP16 100M baseline
- Measure quality gap and efficiency gains
- **Task:** Standard LM benchmarks

---

*This specification combines findings from fundamentals and validation documents. Each design choice is backed by mathematical analysis and/or empirical evidence from published research.*

*Last updated: February 2026*
