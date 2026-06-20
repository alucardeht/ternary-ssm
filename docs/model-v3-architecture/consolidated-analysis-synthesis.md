# Consolidated Analysis Synthesis — v3 Post-Mortem & Path Forward

**Generated:** 2026-02-27
**Sources:** 2 math agents + 2 paradigm-breaker agents analyzing ~142 research findings across 6 rounds

## Executive Summary

v3 implemented 25 techniques from research findings using a "lego" approach (plug and test). Result: 24/25 failed, only GaLore succeeded (-0.9% loss). Full integration was +19.3% WORSE.

This document presents the PRINCIPLED analysis: understanding WHY failures happened, deriving NOVEL solutions from mathematical roots, and ranking ALL proposals by impact × feasibility.

**Key discoveries:**
- **STE Equilibrium Fragility:** Explains 24/25 failures — techniques modifying weights/gradients globally disrupt the ~95% locked / ~5% active weight equilibrium
- **Additive Failure:** 19.3% ≈ sum of individual deltas — techniques independently degrade, don't anti-synergize
- **Safe Technique Rule:** Only techniques operating ORTHOGONALLY to STE (optimizer state, post-hoc, inference-only) are safe
- **Speed Paradox:** Ternary advantage is LATENT at 6M, materializes at 1B+ (bandwidth bottleneck crossover at d_model ≈ 8200)
- **Ternary ≠ Damaged Continuous:** It's a GF(3) algebraic structure with unique properties (finite neighborhoods, exact composition, compiled dispatch)

## Research Statistics

| Metric | Value |
|--------|-------|
| Total research findings | ~460+ across 35 docs (Rounds 1-6) |
| Findings organized by principle | 142 across 17 families |
| Techniques implemented in v3 | 25 |
| Techniques that improved quality | 1 (GaLore: -0.9%) |
| Success rate (lego approach) | 4% |
| Novel solutions derived (math analysis) | 14 training + 8 inference + 6 paradigm |
| Combined potential (training quality) | 2.8-5.4% PPL closure of 7.4% gap |
| Combined potential (inference speed) | 15,000-50,000 tok/s at 6M (conservative) |

## Why v3 Failed — Root Causes

### 1. STE Equilibrium Fragility (Core Insight)

STE training reaches equilibrium where ~95% of weights are "locked" (gradient too small to flip ternary value) and ~5% are "active" (at decision boundaries). Any technique modifying loss surface, gradient, or quantization globally disrupts this equilibrium.

**Failed techniques and the rule they violated:**
| Technique | What it modified | Why it failed |
|-----------|-----------------|---------------|
| Angular reg | Weight space (pushed toward attractors) | Conflicts with task gradient direction |
| Boundary reg | Gradient estimation (attenuated 85-95%) | Slowed learning globally, not just at boundaries |
| STEWS ensemble | Gradient estimation (averaged estimators) | Same-sign bias preserved through averaging |
| HESTIA | Quantization schedule (soft → hard) | Incompatible with from-scratch ternary QAT |
| Deferred Shadow | Weight initialization (random→learned) | Delayed convergence, STE can't recover in time |
| MoTE | Architecture (conditional routing) | Router added noise to gradient signal |
| OptRot | Weight transformation (rotation) | Changed weight landscape STE had optimized for |
| Full Integration | All of the above simultaneously | Compounded misalignment → +19.3% |

**The ONE success:** GaLore operates in OPTIMIZER STATE SPACE (low-rank gradient projection). Never touches weights, gradients, or quantization directly. Preserves STE equilibrium completely.

### 2. Additive Failure (Not Anti-Synergy)

```
Individual deltas: STEWS(+1.4%) + TLH(+1.4%) + MoTE(+2.2%) + AngBound(+3.4%) + DeferShadow(+6.2%) ≈ 14.6%
Full integration: +19.3%
Difference: 4.7% (interaction effects, ~25% of total)
```

Implication: Each technique independently adds noise. Fixing one doesn't help others. Only POSITIVE individual techniques should be combined.

### 3. Lego Approach (Process Failure)

- Plugged techniques without understanding mathematical principles
- Tested on/off flags without understanding failure modes
- No mathematical analysis of WHY something should work before implementing
- Result: ~3 months of implementation with 4% success rate

## The Safe Technique Classification

### SAFE for STE Equilibrium (IMPLEMENT)
| Category | Examples | Why Safe |
|----------|----------|----------|
| Optimizer improvements | GaLore, schedule changes | Operate in optimizer state space |
| Post-hoc optimization | SA, VQ-codebook, XOR-Delta | Applied after training, no STE interaction |
| Inference-only | CUDA Graph, compiled dispatch, POPCOUNT | Don't affect training at all |
| Loss-space regularization | STP (threshold smoothing) | Smooths landscape without pushing weights |
| Gradient amplification (not attenuation) | CTCE | Amplifies 5% near boundary, leaves 95% unchanged |
| Architecture (if trained from scratch) | Any, if whole training is ternary | STE adapts to architecture during training |

### UNSAFE for STE Equilibrium (AVOID)
| Category | Examples | Why Unsafe |
|----------|----------|------------|
| Weight regularizers | Angular, Boundary, any R(w) with attractor | Fixed attractors conflict with task gradient |
| Modified STE | STEWS, HESTIA, CAGE, DQT | Change gradient estimation globally |
| Weight transformations | OptRot, Deferred Shadow | Modify weight values STE has optimized |
| Any technique affecting >5% of weights | Most regularizers | Disrupts locked/active equilibrium |

## Ranked Proposals — Training Quality

**Target:** Close 7.4% PPL gap vs FP16

**Gap decomposition:**
- 30% Representation (~2.2% PPL) — Shannon bound: current encoding within 31% of optimal
- 20% Optimization (~1.5% PPL) — STE ignores cross-weight threshold coupling
- 40-60% Duration (~3.0-4.4% PPL) — Only 0.002% of weights cross thresholds per step
- 10% Interaction (~0.7% PPL) — Regularizer-induced gradient misalignment

| Rank | Technique | Expected PPL Closure | Compute Cost | Risk | Time | Gap Component |
|------|-----------|---------------------|--------------|------|------|---------------|
| 1 | **STP** (Stochastic Threshold Perturbation) | 0.3-0.5% | ~0% | Very Low | 1 hour | Duration + Interaction |
| 2 | **CTCE** (Composite Threshold-Crossing Estimator) | 0.6-0.9% | ~0% | Low | 4 hours | Optimization |
| 3 | **SAQ** (Step-Aligned Quantization) | 0.9-2.2% | Negative | Medium | 1 day | Duration |
| 4 | **RLEC** (Residual Low-Rank Error Compensation) | 0.9-1.5% | +0.6-1.6% | Low | 2 days | Representation |
| 5 | **GASTE** (Group-Aware STE) | 0.1-0.3% | ~0% | Low | 1 day | Optimization |
| 6 | **Block-Entropy Regularization** | 0.5-1.5% | ~0% | Low | 4 hours | Duration |
| 7 | **Dual-Rate STE** | 0.5-1.5% | ~0% (saves locked weight computation) | Low | 1 day | Duration |
| 8 | **SA Post-Training** | 0.5-2.0% | Post-hoc | Medium | 2 days | All components |
| 9 | **Shadow-Free Training** | Enables 3.35x model size | Same | HIGH | 1 week | Paradigm shift |

**Combined potential (Rank 1-5):** 2.8-5.4% closure → gap reduced to 2.0-4.6%
**Combined potential (Rank 1-8):** Potentially close entire gap with SA post-training refinement

**Synergies:**
- STP + CTCE: STP smooths landscape, CTCE amplifies at boundaries → strong directed signal
- SAQ + STP: FP16 steps provide clean gradients, ternary steps use STP → temporally separated
- RLEC: Independent parameter space (A, B matrices) → composable with ANY gradient estimator

## Ranked Proposals — Inference Speed

**Target:** Maximize tok/s for autoregressive batch-1 inference
**Bottleneck:** LM Head = 92.8% of weight reads at 6M (32000 vocab × 256 dim)

| Rank | Technique | Expected Speedup | Scope | Risk | Time |
|------|-----------|-----------------|-------|------|------|
| 1 | **Entropy Screening** (pre-filter by support density) | 2-3x LM head | LM head | Zero | 1 hour |
| 2 | **Bit-Serial POPCOUNT** (quantized activations + progressive pruning) | 10-18x LM head | LM head | Low | 1-2 days |
| 3 | **BSP-Tree Decoding** (binary space partition over vocab cones) | 50-200x realistic | LM head | Medium | 1 day |
| 4 | **GF(3) Factorization** (ternary matrix decomposition) | 3-6x body | All layers | Medium | 2h measure + 2 days |
| 5 | **Spectral Biclustering** (block-diagonal decomposition) | 4-8x | Body layers | Medium | 2h measure + 1 day |
| 6 | **Cross-Row Partial Sum Sharing** | 3-5x LM head | LM head | Medium | 2 hours |
| 7 | **VQ-Ternary** (codebook + compiled dispatch) | 5-10x CPU | All layers | Low | 1-2 weeks |
| 8 | **Compiled Inference VM** (weights as instructions) | 4.5x+ CPU | All layers | Low | 1-2 weeks |

**CRITICAL NEGATIVE:** Tensor Core Composition is 10.7x SLOWER — do NOT implement.

**Composable Pipeline (Conservative):**
```
Baseline:            877 tok/s (CUDA Graph)
+ Entropy Screening: ~2x  → 1,754 tok/s
+ POPCOUNT:          ~5x  → 8,770 tok/s
+ GF(3) Body:        ~3x  → 26,310 tok/s
Conservative estimate: 15,000-50,000 tok/s at 6M
```

**Prerequisite Measurements (on existing 6M checkpoint, no training needed):**
| Measurement | Purpose | Time |
|-------------|---------|------|
| Support density (I_v) per vocab entry | Validates Entropy Screening | 10 min |
| GF(3) rank of all ternary matrices | Validates GF(3) Factorization | 2 hours |
| SVD of in_proj, out_proj, LM head | Validates Biclustering + BSP-tree | 30 min |
| Hamming distances between LM head rows | Validates Cross-Row Sharing | 30 min |
| Activation magnitude distribution | Calibrates POPCOUNT bit-serial | 10 min |

## Ranked Proposals — Paradigm Shifts

These are RADICAL alternatives that could fundamentally change the approach:

| Rank | Proposal | What Changes | Impact | Feasibility |
|------|----------|-------------|--------|-------------|
| 1 | **VQ-Ternary** (vector-quantized codebook) | Model storage + inference | 3.9 MB model, 5-10x CPU speed | High (1-2 weeks) |
| 2 | **GF(3) Model Algebra** (exact composition) | Model storage + distribution | 133x family compression, exact merging | Medium-High (1 week) |
| 3 | **Compiled Inference VM** (weights as instructions) | Inference execution model | 4.5x+ CPU, eliminates multiply entirely | High (1-2 weeks) |
| 4 | **XOR-Delta Ensemble** (4 ternary < 1 FP16) | Quality approach | Match FP16 quality at same memory | Medium (2-3 weeks) |
| 5 | **Shadow-Free Graph Descent** (eliminate shadow weights) | Training memory | 3.35x model size in same VRAM | Low (4+ weeks research) |
| 6 | **Error-Correcting Architecture** (coding theory) | Quality approach | Close 30-50% representation gap | Low (4+ weeks research) |

## Recommended Action Plan

### Phase A: Quick Wins (1-2 days total)

**Training:**
1. STP implementation (1 hour) — test at 6M, 100 steps
2. CTCE implementation (4 hours) — test at 6M, 100 steps
3. Block-Entropy regularization (4 hours) — test at 6M, 100 steps

**Inference:**
4. Entropy Screening implementation (1 hour) — benchmark at 6M
5. Prerequisite measurements (GF(3) rank, SVD, support density) — 3 hours compute

**Expected outcome:** 0.9-2.9% PPL closure + 2-3x LM head speed + data to validate next steps

### Phase B: Short-Term (1 week)

**Training:**
6. SAQ implementation (1 day) — test at 6M, then 50M if positive
7. RLEC implementation (2 days) — test at 6M
8. Dual-Rate STE (1 day) — test at 6M

**Inference:**
9. Bit-Serial POPCOUNT (1-2 days) — 10-18x LM head
10. BSP-Tree prototype (1 day) — validate 50-200x estimate

**Expected outcome:** 2.8-5.4% PPL closure + 10-60x LM head speed

### Phase C: Medium-Term (2-4 weeks)

**Training:**
11. SA Post-Training optimization
12. Shadow-Free Training research (1M scale prototype)

**Inference:**
13. VQ-Ternary codebook + compiled dispatch
14. GF(3) Factorization (if rank measurement validates)
15. Compiled Inference VM prototype

**Paradigm:**
16. XOR-Delta Ensemble system
17. GF(3) Model Algebra (exact composition)

## Files Reference

| Document | Content |
|----------|---------|
| `docs/model-v3-architecture/findings-by-principle.md` | 142 findings organized by 17 mathematical principle families |
| `docs/model-v3-architecture/math-analysis-gradient-quality.md` | 6 novel training solutions with mathematical derivations |
| `docs/model-v3-architecture/math-analysis-inference-speed.md` | 8 inference speed paths with complexity analysis |
| `docs/model-v3-architecture/paradigm-analysis-ternary-unique.md` | 6 radical alternatives exploiting ternary as GF(3) |
| `docs/model-v3-architecture/paradigm-analysis-cross-cutting.md` | Cross-cutting synthesis: why v3 failed, safe technique classification |
| `docs/model-v3-architecture/version-comparison.md` | All measured performance numbers v2 vs v3 |

## Post-Mortem: Lessons for Future Implementation

1. **NEVER lego:** Don't plug techniques without mathematical analysis of WHY they should work
2. **Respect STE equilibrium:** Only techniques orthogonal to STE are safe
3. **Test individually first:** Never combine untested techniques
4. **Measure prerequisites:** Validate mathematical assumptions (e.g., GF(3) rank) BEFORE implementing
5. **Debug logging ALWAYS:** Never declare results without proving the technique actually activated
6. **6M benchmarks underestimate ternary advantage:** Speed benefits materialize at 1B+ scale
7. **LM Head dominates inference:** Focus optimization there first (92.8% of weight reads)
