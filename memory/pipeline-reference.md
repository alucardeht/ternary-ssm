---
name: Validated training pipeline reference
description: THE current best training pipeline with all confirmed components — READ BEFORE ANY TRAINING EXPERIMENT
type: reference
---

# Validated Training Pipeline — v11

**Last updated:** 2026-03-24
**Source:** STATUS.md + R272 (v11 reference) + R92/R128/R139 (v10 validated)

---

## THE PIPELINE (Current Best)

### Stage 1: Forward Programming (FP)

Train each layer sequentially, one at a time, with all others frozen.

```
For stage k = 0 to (n_layers - 1):
  1. Unpack layer k from INT2 (CPU) → FP32 (GPU)
  2. Train 100-150 steps:
     - Forward through ALL layers (frozen = no_grad)
     - CE loss
     - Backward through layer k ONLY
     - Optimizer step (AdamW8bit, LR=5e-4)
  3. [VALIDATED] D4 Gradient Filter: zero gradients unless sign consistent for 5+ steps
  4. [VALIDATED] ABA Early Stopping: patience=20, min_steps=30, abs_threshold=0.005
  5. [OPTIONAL] OLS recalibration of out_proj (closed-form least squares)
  6. Pack layer k → INT2, move to CPU
```

**Measured (R130, 3.38B model):** 893 MB peak VRAM, 29 min total for 32 stages
**Quality (R139, 100M):** val_loss = 2.3139 nats (FP only)

### Stage 2: mini-BCD (Block Coordinate Descent)

Post-FP refinement on subset of layers.

```
For layer k in target_layers (bottom 25% OR top 25% — R291 pending):
  1. Unpack layer k → FP32 on GPU
  2. Train 50 steps:
     - Full forward through ALL layers
     - CE loss
     - Backward through layer k only
     - LR = 1e-4 (R272) or 5e-6 (v10 BCD)
  3. Pack layer k → INT2, move back to CPU
```

**Measured (R92, 100M):** ~28-32% gap closure vs joint GD
**R272 variant:** bottom 25% layers, LR=1e-4, 50 steps

### Alternative Stage 2: SAC (Sequential Adjoint Correction)

```
3 forward sweeps with adjoint correction.
VRAM: 1.33 GB. Time: ~1 min. Gap closure: 35.1% (R128, 100M).
```

### Alternative Stage 2: SBP (Sequential BackProp)

```
Full GD with layer-by-layer backward + CPU offloading.
VRAM: ~10 GB. Quality: identical to joint GD (+0.0002 nats, R139, 100M).
Time: 5.5x slower than joint GD.
```

---

## VALIDATED COMPONENTS (use these)

| Component | Gain | Source | Status |
|-----------|------|--------|--------|
| FP layer-sequential | Enables 7B in 893 MB | R121/R130 | ACTIVE |
| INT2 storage frozen layers | 83% compression, +0.025 nats | R98/R100 | ACTIVE |
| D4 gradient filter (sign consistency) | Always-on in R272 | R272/R294 | ACTIVE |
| ABA early stopping | Saves ~30% FP steps | R272 | ACTIVE |
| fp32 cache (not PCA) | 2.59x speedup, +0.018 nats | R122 | ACTIVE |
| RNS Embedding | **3335× fewer embedding params** | R297 validated | ACTIVE |
| Predictive Coding | SSM processes surprise (zero-init) | R297 validated | ACTIVE |
| TBD Init (A_log geometric) | Multi-scale temporal processing | R297 validated | ACTIVE |
| TCSD (self-distillation EMA h_t) | Stabilizes representations | R307 validated | ACTIVE (Joint GD) |
| TMA (64-slot ternary memory) | Retrieval without attention | R307 validated | ACTIVE |
| TIL (multi-horizon loss) | Direct gradient to A_log | R307 validated | ACTIVE |
| PSA (predictive state alignment) | Constant SNR all layers | R306 validated | ACTIVE |
| OSP (orthogonal state partition) | Max entropy A eigenspectrum | R307 validated | ACTIVE |
| Eigenspectrum calibration | Data-driven A_log init | R304 validated | ACTIVE |
| CE sum gradient | BitNet confirmed | R308 validated | ACTIVE |
| Joint GD training mode | Standard backprop all layers | R303/R305 | ACTIVE (replaced FP) |
| Streaming CE loss | **6× VRAM reduction** (no logits tensor) | R298 validated | ACTIVE (FP only) |
| BCD cached targets | **50×+ VRAM reduction** (1 layer on GPU) | R298 validated | ACTIVE |
| LR schedule (warmup+cosine) | Standard, no measurement yet | Implemented | ACTIVE |
| Warm start last stage | 10 pre-steps with frozen head | R297 math validated | ACTIVE |
| Proportional steps (2.5× last) | More budget where needed | R297 math validated | ACTIVE |
| INT8 activation cache | -0.028 nats (regularization!) | R118 | VALIDATED |
| mini-BCD bottom 25% | 28-32% gap closure | R92 | ACTIVE |
| SAC (3 sweeps) | 35.1% gap closure | R128 | VALIDATED |
| SBP | = joint GD quality | R139 | VALIDATED |
| Lazy cache refresh | 2-3x FP speedup | R150 | VALIDATED, not implemented |
| K=4 chunk FP | -0.106 nats vs K=1 (shrinks with scale) | R123/R123b | VALIDATED |
| OLS out_proj recalibration | Closed-form, tested in R277 | R277 | DESIGNED, pending results |
| σ_min(I-A_q) attractor regularizer | Prevents repetition fixed points | R309 | ACTIVE |
| Trajectory Diversity Loss (L_traj) | Penalizes state convergence | R310 | ACTIVE |
| Prompt-State Anchoring | Residual injection keeps model on-topic | R311 | ACTIVE |
| EOS Entropy Head | Binary head detecting information exhaustion | R311 | ACTIVE |
| Adaptive Threshold τ_i* | Per-group std-based ternary thresholds | R312 | ACTIVE |
| L_gram Row-Orthogonality | Ternary weight diversity regularizer | R313 | ACTIVE |

## REJECTED COMPONENTS (do NOT use)

| Component | Why rejected | Source |
|-----------|-------------|--------|
| Ternary training (STE) | +1.06 nats overhead | R97 |
| PCA-INT8 cache for TRAINING | +3.04 nats catastrophic | R122 |
| INT4 activations | +0.302 nats dead zone | R118 |
| Reverse BCD | -59% quality | R112 |
| QOTA | +1.20 nats (Taylor invalid) | R108 |
| DEQ-FP | Not viable (distance=0.84) | R110 |
| Combined mods (SLS+Warm+BCD) | -33.4% interference | R115 |
| norm-ternary cache | NaN collapse | R120 |

## PENDING VALIDATION (designed, no results yet)

R276-R294: 14 experiments designed but never run. See experiment-index.md.
Most critical: **R291** (top vs bottom BCD targets — could change mini-BCD strategy).

---

## VALIDATED SPEED GAINS (R244-R271b)

These are CONFIRMED improvements to the pipeline. Must be used in all new experiments.

| Technique | Gain | Source | Status |
|-----------|------|--------|--------|
| Cache H_k incremental | **33x FP speedup** | R262/R265 | VALIDATED 6M |
| ABA early stopping | **47% fewer FP steps**, +0.039 nats | R270 | VALIDATED |
| Batch fix + double forward fix | loss 3.18 vs 3.43 | R257 | VALIDATED |
| mini-BCD (replaces full BCD) | **25% gap closure in 57s** (vs 2h full BCD) | R271b | VALIDATED |
| W_shift rank-8 correction | -0.122 nats (3.5% closure) | R278 | VALIDATED |

**Estimated 7B time with ALL optimizations:**
- FP with cache incremental + ABA: **~13 min**
- Mini-BCD: **~5-10 min**
- **Total: ~20-25 min** (vs baseline 27h+, vs unoptimized FP 29 min)

## REJECTED RECENT (R276-R280)

| Technique | Result | Source |
|-----------|--------|--------|
| TB-BCD (two-phase backward) | 23x SLOWER | R276 |
| Robust noise alpha=0.3 | NEUTRAL | R276 |
| OLS closed-form out_proj | 7.7% (weak) | R277 |
| Rounds vs FP classic | = FP (no benefit) | R279 |

## IMPORTANT NUMBERS

| Metric | Value | Source |
|--------|-------|--------|
| FP gap vs GD (100M) | 0.23 nats (2.31 vs 2.08) | R139 |
| FP gap structural | ~65-70% is inherent to sequential training | R112-R115 |
| FP+1000 steps GD | 89% gap closure | R135 |
| BCD gap closure | 28-32% | R92 |
| mini-BCD gap closure | 25% in 57s | R271b |
| SAC gap closure | 35.1% | R128 |
| Real 7B training time (60B tokens) | ~495 days | R150 analysis |
| FP 7B time (Stage 1, unoptimized) | 29 min | R130 |
| FP 7B time (Stage 1, optimized) | ~13 min (projected) | R262/R270 |
| FP 7B VRAM | 893 MB peak | R130 |
| FP-GD gap by scale | 6M: 7% / 25M: 6.6% / 100M: 17.3% | R134 |
