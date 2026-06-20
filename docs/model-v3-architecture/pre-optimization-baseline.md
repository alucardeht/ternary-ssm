# Pre-Optimization Baseline — Immutable Reference

**Snapshot date:** 2026-02-27
**Purpose:** Immutable reference point for comparing gains from v3 optimization techniques. DO NOT MODIFY THIS FILE.

## What This Captures

State of the model AFTER:
- Phases 1-36 complete (v2 architecture + v3 research implementation)
- CUDA Graph bug fixed (commit 532b185)
- HLM dtype bug fixed (commit 9ebde9c)
- 25 techniques implemented, 24/25 failed, GaLore only success
- All failing techniques DISABLED (flags off)

State BEFORE:
- Any Phase A/B/C optimization techniques (STP, CTCE, SAQ, RLEC, etc.)
- Any inference optimizations (Entropy Screening, POPCOUNT, BSP-Tree, etc.)

## FP16 Reference (Conventional Model)

These are the numbers our ternary model must beat or match.

### Training Quality
| Metric | FP16 Value | Measured On | Notes |
|--------|-----------|-------------|-------|
| Val PPL (5-epoch, 50M) | 31.96 | RTX 5070 | Full FP16 training baseline |
| Loss @100 steps (50M) | — | — | Not measured for FP16 at 100 steps |

### Inference Speed
| Metric | FP16 Value | Hardware | Notes |
|--------|-----------|----------|-------|
| GPU tok/s (6M) | 117.5 | RTX 5070 | Standard PyTorch FP16 |
| CPU tok/s (6M) | 81.9 | Intel 225F | Standard PyTorch FP16 |
| VRAM inference (6M) | 300 MB | RTX 5070 | Full FP16 weights in memory |

## v2 Baseline (Pre-Research Ternary)

Phase 23 ternary SSM model with inference engine, NO research techniques applied.

### Training Quality
| Metric | v2 Value | Scale | Notes |
|--------|---------|-------|-------|
| Val PPL (5-epoch) | 34.33 | 50M | Direct ternary STE, AdamW |
| Loss @100 steps | 7.41 | 50M | Baseline for technique comparison |
| Gap vs FP16 | 7.4% | 50M | (34.33 - 31.96) / 31.96 |

### Inference Speed (6M scale, RTX 5070)
| Metric | v2 Value | Notes |
|--------|---------|-------|
| GPU tok/s (FP16 LM head) | 876.8 | CUDA Graph + Triton GEMV + fused SSM |
| CPU tok/s | 105.6 | Portable engine |
| VRAM inference | 27 MB | 2-bit packed ternary weights |
| vs FP16 GPU speedup | 7.5x | 876.8 / 117.5 |
| vs FP16 CPU speedup | 1.3x | 105.6 / 81.9 |
| vs FP16 VRAM savings | -91% | 27 MB vs 300 MB |

### Training Speed
| Metric | v2 Value | Hardware | Notes |
|--------|---------|----------|-------|
| Training tok/s | 10,738 | RTX 5070 | Direct ternary, AdamW |
| Training tok/s | 24,676 | Modal L4 | Cloud GPU benchmark |
| Training VRAM peak | 4.4 GB | RTX 5070 | 50M params |

## v3 Current State (Post Bug Fixes, Pre-Optimization)

This is the STARTING POINT for new optimizations.

### Training Quality (50M scale, 100 steps)
| Technique | Loss @100 | vs v2 Baseline | Status |
|-----------|----------|----------------|--------|
| v2 Baseline (no techniques) | 7.41 | — | Reference |
| GaLore optimizer | 7.34 | -0.9% (BEST) | ACTIVE — only working technique |
| STEWS ensemble | 7.51 | +1.4% | DISABLED |
| Ternary LM Head | 7.51 | +1.4% (but eliminates FP16 bottleneck) | ACTIVE — architectural necessity |
| MoTE | 7.57 | +2.2% | DISABLED |
| Angular+Boundary reg | 7.66 | +3.4% | DISABLED |
| Optimal Rotation | 7.72 | +4.2% | DEFERRED |
| Deferred Shadow | 7.87 | +6.2% | DISABLED |
| Full Integration | 8.84 | +19.3% | DISABLED — all techniques combined |

**Current best training config:** GaLore + Ternary LM Head = loss ~7.34-7.51 range

### Inference Speed (6M scale, RTX 5070)
| Configuration | GPU tok/s | CPU tok/s | VRAM | Notes |
|--------------|-----------|-----------|------|-------|
| Ternary LM Head (baseline) | 171.2 | — | 294 MB | Reference v3 baseline |
| FP16 LM Head (ablation) | 182.0 | — | 74 MB | Quality baseline without ternary head |
| + CUDA Graph (fixed) | 189.8 | — | ~400 MB | Runner cached on model |
| + Early Exit | 169.6 | — | 530 MB | Neutral on random data |
| + Hierarchical LM Head | 0.97x | — | 3709 MB | Neutral at 6M, bandwidth benefit at scale |
| C Portable ctypes | — | 12.6 | — | Overhead dominates at 6M |

**Current best inference config:** CUDA Graph = 189.8 tok/s GPU

### v3 vs v2 Comparison (HONEST)
| Metric | v2 | v3 Current | Change | Why |
|--------|-----|-----------|--------|-----|
| GPU tok/s | 876.8 | 189.8 | -78% | Ternary LM Head adds 8.2M params (+220MB VRAM) |
| CPU tok/s | 105.6 | 84.1 | -20% | Additional v3 components overhead |
| VRAM inference | 27 MB | 294 MB | +988% | Ternary LM Head weight storage |
| Training loss (best) | 7.41 | 7.34 | -0.9% | GaLore only improvement |
| Training VRAM | 4.4 GB | TBD | TBD | Not re-measured post v3 |

### Key Insight on Speed Regression

The 876.8 → 189.8 tok/s drop is NOT from failed techniques (they're disabled). It's from the **Ternary LM Head** — a deliberate architectural choice that:
- Eliminates the FP16 bottleneck (the ONLY non-ternary component)
- Adds 8.2M params of ternary weights to the output head
- Increases VRAM from 74 MB to 294 MB
- BUT enables all-ternary inference pipeline (prerequisite for POPCOUNT, compiled dispatch, etc.)

Without Ternary LM Head, the FP16 LM head achieves 182.0 tok/s (comparable to v2's engine without CUDA Graph). The v2 876.8 number included CUDA Graph + FP16 LM head.

Fair comparison: v2 FP16-head (876.8) vs v3 FP16-head (182.0) shows engine regression that needs investigation. The CUDA Graph fix brought v3 from 72.5 to 189.8, but that's with Ternary LM Head.

## Targets for Optimization

Based on mathematical analysis (consolidated-analysis-synthesis.md):

### Training Quality Target
- Close 2.8-5.4% of 7.4% PPL gap (techniques STP, CTCE, SAQ, RLEC, GASTE)
- Best case: gap reduced from 7.4% to 2.0-4.6%
- Stretch: full gap closure with SA post-training

### Inference Speed Target
- Conservative pipeline: 15,000-50,000 tok/s at 6M
- Key techniques: Entropy Screening (2-3x), POPCOUNT (10-18x), GF(3) factorization (3-6x)
- LM head is 92.8% of bottleneck — focus there first

### Memory Target
- Inference VRAM: reduce from 294 MB toward v2's 27 MB via better packing
- Training VRAM: maintain ≤4.4 GB at 50M

## Architecture Details (for reproducibility)

| Parameter | Value |
|-----------|-------|
| Model config | config_ternary_6m() / config_ternary_50m() |
| Total params (6M) | 20,180,592 |
| Total params (50M) | ~72M actual |
| Weight encoding | {-1, 0, +1} with per-group scaling |
| Group size | 64 weights per group |
| Architecture | Hybrid SSM (87.5%) + SWA (12.5%) |
| SSM variant | SSD (Mamba-2 style) |
| Vocab size | 32,000 |
| Tokenizer | GPT-2 BPE |
| Training optimizer | AdamW (+ GaLore for best config) |
| Gradient estimator | STE (Straight-Through Estimator) |
| Hardware | RTX 5070 12GB GDDR7, Intel 225F, 32GB DDR5 |
