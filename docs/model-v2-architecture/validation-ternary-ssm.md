# 05 — Validation: Ternary + SSM Viability

## VERDICT: VIABLE — But With Mixed Precision Constraints

Bi-Mamba (Nov 2024) proves 1-bit Mamba works at 780M-2.7B scale with 84-89% compression. BUT: critical SSM components need higher precision.

---

## 1. Prior Art — Someone Already Did Part of This

### Bi-Mamba (Nov 2024) — The Flagship

1-bit quantization on Mamba architecture:
- Scales: 780M, 1.3B, 2.7B parameters
- **Binarizes 90%+ of parameters** (input/output projections)
- **Keeps full-precision:** A, B, C, D matrices, conv1d, Δ_bias, embeddings, layer norms
- Uses FBI-Linear layers: binary weights {-1, +1} with learned scaling factors
- Training: binarization-aware from scratch (NOT post-training)
- Distillation from LLaMA2-7B teacher
- 2.7B model: 49.3% avg accuracy on zero-shot tasks

### Q-S5 (2024) — Binary Recurrent Weights

First solution with **binary recurrent weights solving copy task >1000 timesteps**. Maintains performance comparable to full-precision.

### What This Tells Us

The basic approach works. 90% of parameters can be ternary/binary. The remaining 10% (SSM-critical components) need higher precision. This is still a massive win.

---

## 2. Hard Constraints Discovered

### A Matrix: MUST Be ≥8-bit

**No exceptions found in any paper.**

- Sub-8-bit A: "dramatic performance collapse" (Q-S5)
- 4-bit A: Marginal on simple tasks only
- 1-bit A: "Never converged" on dynamical system tasks

**Why:** Eigenvalues of A determine stability. Quantization shifts eigenvalues outside unit circle → exploding/vanishing states. Error compounds multiplicatively across timesteps.

### B, C, Δ Projections: Unknown Frontier

**No published work has quantized B, C, Δ to ternary and shown it works.**

Bi-Mamba keeps these at full precision. Rationale: "Maintaining these critical SSM components in high precision preserves discretization mechanism and state evolution properties."

This is our KEY RESEARCH QUESTION for validation experiments.

### Post-Training Quantization: FAILS Catastrophically

PTQ on Mamba at <4-bit: perplexity increases 10x. Produces "meaningless symbols."

**Training-aware quantization from scratch is MANDATORY.** No shortcuts.

---

## 3. Quantization Degradation Curve (Mamba-Specific)

| Precision | Quality Loss | Method | Status |
|-----------|-------------|--------|--------|
| 16→8 bit | <1-3% | PTQ works | Production-ready |
| 8→4 bit | 5-15% | Advanced PTQ or QAT | Viable |
| 4→2 bit | 20-40% | QAT required | Experimental |
| 2→1 bit | 40%+ unless trained from scratch | QAT mandatory | Research (Bi-Mamba) |

---

## 4. SSM-Specific Failure Modes

### Activation Outlier Amplification
Parallel Scan operator amplifies outliers through self-multiplication. Standard Hadamard transform (works for Transformers) FAILS for Mamba because data is input-dependent.

### State Cache Precision
State caches (h_t) are highly sensitive. Quantization errors accumulate over timesteps. Minimum: 4-bit state cache (W8A8H4).

### Curse of Memory Under Quantization
For diagonal RNNs: gradient w.r.t. λ scales as (1+λ²)/(1-λ²)³ — cubic divergence. Gradients diverge FASTER than activations near stability boundaries. Recurrent weights need more precision than feedforward weights.

### STE Stalling
At 1-2 bit grids, STE gradients can be too small to change quantized levels → training plateaus. Solutions: sigmoid STE, soft expectations, noise regularization.

---

## 5. Revised Architecture: Mixed Precision Strategy

Based on Bi-Mamba's findings and constraints:

```
TERNARY (1.58-bit) — 90%+ of parameters:
  - Input projections (token → hidden)
  - Output projections (hidden → token)
  - MLP/feed-forward blocks
  - Router networks (MoD)

HIGHER PRECISION — Critical 10%:
  - A matrix: ≥8-bit (eigenvalue stability)
  - B, C, Δ projections: START at FP16, incrementally quantize in experiments
  - Embeddings: FP16 (first layer)
  - Layer norms: FP16 (normalization precision)
  - State cache h(t): ≥4-bit
```

### Compression Still Massive

Even with 10% at higher precision:
```
Standard 7B (FP16):     14 GB
90% ternary + 10% FP16: ~1.9 GB (86% reduction)
90% ternary + 10% INT8:  ~1.5 GB (89% reduction)
```

---

## 6. Validation Experiments to Run

### Experiment 1: Baseline (Bi-Mamba approach)
- Build small SSM (1-10M params) with ternary projections
- Keep A, B, C, Δ at FP16
- Train from scratch on simple task (copy, associative recall)
- **Expected:** Should work (proven by Bi-Mamba)
- **Purpose:** Validate our implementation

### Experiment 2: Incremental B, C, Δ Quantization
- Start from working Experiment 1
- Quantize B projection: FP16 → INT8 → INT4 → ternary
- Measure selectivity diversity at each step
- Repeat for C, then Δ
- **Purpose:** Find the precision floor for each component

### Experiment 3: A Matrix Floor
- Test A at INT8, INT4 (expect failure below 8-bit)
- Validate eigenvalue stability monitoring
- **Purpose:** Confirm 8-bit A constraint in our architecture

### Experiment 4: Combined System
- Ternary projections + best-found B/C/Δ precision + routing
- Copy task, associative recall, simple language modeling
- **Purpose:** Full architecture validation at small scale

### Experiment 5: Scale Test
- If Exp 4 passes, scale to 100M params
- Compare against full-precision baseline
- **Purpose:** Check if findings hold at scale

---

## 7. Training Recipe (From Literature)

```
Phase 0 (optional): Pre-train full-precision teacher for distillation
Phase 1 (0-10%):    FP32 warmup, no quantization
Phase 2 (10-20%):   Quantize projections to ternary (STE)
Phase 3 (20-40%):   Stabilize, monitor eigenvalues and gradient norms
Phase 4 (40-100%):  Full training with all quantization active

Optimizer: Adam (adaptive essential for quantized training)
Normalization: RMSNorm (preferred over LayerNorm for quantized models)
Gradient clipping: Essential
State monitoring: Eigenvalues of A, gradient norms, router entropy, state magnitudes
```

---

## 8. Impact on Model v2 Design

### What Changes From Our Original Vision

1. **NOT 100% ternary** — ~90% ternary, ~10% higher precision
2. **A matrix is special** — needs its own precision tier (≥8-bit)
3. **B, C, Δ are the research frontier** — may stay FP16 or may go lower (our experiments will determine)
4. **Must train from scratch** — no post-training quantization path
5. **Need distillation teacher** — Bi-Mamba used LLaMA2-7B as teacher

### What Stays The Same

1. Ternary weights for the BULK of the model (projections, MLP, router)
2. SSM core with selective state evolution
3. Dynamic routing (MoD) with ternary router
4. CPU-primary inference goal (most operations are add/subtract/skip)
5. No KV cache (SSM state, O(1) memory)

### Net Assessment

**Still revolutionary compared to standard Transformers:**
- ~86-89% weight compression
- Most operations are add/subtract/skip (no matmul for 90% of compute)
- O(1) memory for context (no KV cache)
- CPU inference viable for the ternary portions
- Only ~10% of ops need traditional multiply (A matrix, B/C/Δ)

---

*Sources: Bi-Mamba (arXiv 2411.11843), Q-S5 (arXiv 2406.09477), Q-Mamba (ACL 2025), MambaQuant (arXiv 2501.13484), Quamba (arXiv 2410.13229), TernaryLM (arXiv 2602.07374)*

*Last updated: February 2026*
