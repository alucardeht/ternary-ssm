# Implementation Triage — Consolidated Findings

## Baselines (Phase 36, 50M params)
- Training loss: 7.41 baseline, 7.34 best (GaLore)
- Inference: GPU 876.8 tok/s, CPU 105.6 tok/s
- VRAM: 4.4GB training, 27MB inference
- Ternary gap: 7.4% at 50M (angular, not radial)
- Architecture: 87.5% SSM + 12.5% SWA, d_model=512, d_state=16, 16 layers, window=128

## Already Tested (do NOT re-implement)
| Technique | Result | Phase |
|-----------|--------|-------|
| GaLore optimizer | -0.95% (BEST) | 28-C |
| Mega-kernel K1-K4 | 1.17x inference | 28-A |
| STEWS ensemble | +1.4% WORSE | 28-B/36 |
| Ternary LM head | +1.4% WORSE | 31/36 |
| MoTE experts | +2.2% WORSE | 30-C/36 |
| Angular reg | +3.4% WORSE | 28-B/36 |
| Boundary reg | +3.4% WORSE | 28-B/36 |
| OptRot rotation | +4.2% WORSE | 30-B/36 |
| Deferred ternary | +6.2% WORSE | 35/36 |
| Full integration | +19.3% WORSE | 36 |
| CAGE gradient | NaN divergence | 10.C |
| DQT stochastic | +16.5% WORSE | 10.A |
| HESTIA 2026 | +110% WORSE | 10.E |
| QZO | Both WORSE, 30% slower | 10.E |

---

## TIER 1: EASY WINS (Implement Now, Testable at 6M)

### Quality/Training (8 findings)

| # | Finding | Lines | Impact | Source |
|---|---------|-------|--------|--------|
| Q1 | CAQ (Curvature-Aware Quantization) | ~4 | 2-4% PPL | R3-Quality M1 |
| Q2 | BDT (Binary Decomposition Training) | drop-in | Better convergence | R4-Training P1 |
| Q3 | BAGR (Boundary-Attentive Gradient Reweight) | ~35 | 1.3-2x faster training | R6-Training M4 |
| Q4 | ASGKS (Analytic Surrogate Gradient) | ~45 | 1.5-3x fewer steps | R6-Training M5 |
| Q5 | Context-as-Precision (slow SSM heads) | trivial | Free scaling axis | R4-Training P5 |
| Q6 | BATR-STE (Boundary-Aware Trust Region) | drop-in | Better convergence | R4-Training M1 |
| Q7 | Cooldown Fusion (QAT schedule) | scheduler | 50% compute savings | R3-Quality F2 |
| Q8 | Evolutionary Discrete Polish | post-train | 1-3% PPL free | Discoveries |

**Q1: CAQ** — Use Adam's second moment (v_hat) as per-weight ternary threshold. High-curvature weights get tighter tolerance. Zero overhead (reuses existing optimizer state). 4 lines of code.

**Q2: BDT** — Decompose ternary w = sign × activity. Train each with independent binary STE. Binary STE has proven O(1/√T) convergence. Zero sparsity emerges naturally. Drop-in replacement for TernaryQuantize.

**Q3: BAGR** — Gradient damping by distance to quantization boundary: g' = g × exp(-dist²/2σ²). Weights deep in a ternary region get minimal gradient (already committed). Boundary weights get full gradient. ~35 lines.

**Q4: ASGKS** — Replace STE with exact derivative of smooth kernel-smoothed quantization. Mathematical INVERSE of STE: gradient=0 at ternary values (committed), maximal at boundaries (uncertain). Temperature annealing. ~45 lines.

**Q5: Context-as-Precision** — Initialize some SSM heads with slow decay (λ=0.99-0.999). Temporal averaging over T_eff=50-500 tokens gives effective precision up to 6.1 bits per ternary weight. Trivial A_log init change. Zero architecture modification.

**Q6: BATR-STE** — Multiply gradient by exp(-β·dist(w, nearest_centroid)). Mathematically cleaner than Phase 20.B Boundary STE (which was tested but cosmetic). This uses exponential, not linear damping. Drop-in STE replacement.

**Q7: Cooldown Fusion** — Apple 757-experiment study: optimal QAT fraction ~40%. Start QAT at constant LR, then decay DURING QAT (not before). Eliminates 9-13% wasted tokens. Scheduler change only.

**Q8: Evolutionary Polish** — Post-training: flip weights guided by gradient statistics. Explores discrete landscape STE cannot reach. Zero risk (non-destructive). Can run on existing 50M model immediately.

### Architecture (3 findings)

| # | Finding | Effort | Impact | Source |
|---|---------|--------|--------|--------|
| A1 | MoA (Mixture of Attention Spans) | config | 3.9x context, 6.6-8.2x decode | R6-Hybrid F9 |
| A2 | Head Gating (per-head sigmoid) | 2 days | 1-2% quality | R6-Hybrid F7 |
| A3 | Sink Tokens (global attention BOS) | 1 day | 30-100% long-context | R6-Hybrid F8 |

**A1: MoA** — ZERO-COST. Change window=128 to [32, 64, 128, 256] per head. Published COLM 2025. 3.9x effective context with SAME average window. No code change beyond config.

**A2: Head Gating** — Sigmoid gate after scaled dot-product, per head. Mitigates attention sinks. ~1% overhead. Tested at 15B scale. For ternary: single scalar multiply per head.

**A3: Sink Tokens** — Position 0 (BOS) + separators get full attention (not windowed). All other positions keep window=128. Proven technique for long-context coherence.

### Inference (2 findings)

| # | Finding | Effort | Impact | Source |
|---|---------|--------|--------|--------|
| I1 | TRD (Row Deduplication) | 30 min | 1.1-1.4x free | R3-Velocity P6 |
| I2 | SSS (Self-Speculative Superposition) | 30 min | 1.5-2.5x | R4-Inference P1 |

**I1: TRD** — Post-training analysis. Count identical ternary weight rows. Compute once, broadcast. ZERO accuracy loss. ZERO training cost. 30-min validation on existing 50M.

**I2: SSS** — Feed expected embedding (Σ P(v)×e_v from top-K) through layers. SSM linearity guarantees E[h_{t+1}] is exact. No separate draft model. IMPOSSIBLE with Transformers. 30-min prototype.

### Memory (2 findings)

| # | Finding | Effort | Impact | Source |
|---|---------|--------|--------|--------|
| M1 | SSD Structural Checkpointing | 1-line | 127x training tensor | Discoveries |
| M2 | MoA (heterogeneous windows) | config | Lower avg attention FLOPS | R6-Hybrid F9 |

**M1: SSD Checkpointing** — Save log_decay input instead of full (T,T) L matrix in backward. Recompute on backward. 127x tensor reduction. +15% recompute overhead. One-line change in ssm.py.

---

## TIER 2: MEDIUM (1-4 weeks, moderate risk)

### Quality/Training (10 findings)

| # | Finding | Effort | Impact | Risk |
|---|---------|--------|--------|------|
| Q9 | Tequila Re-test (bug fixed) | 3-5 days | 7.4→3-4% | isinstance bug now fixed |
| Q10 | TPSQ (Phase-Space Quantization) | ~10 lines | Info-theoretic optimal | Low |
| Q11 | IAQD (Input-Adaptive Quantization) | 23K params | Novel, unexplored | Low |
| Q12 | GWQ (Gradient-Aware Weight Quant) | post-train | 1-2% PPL | Zero risk |
| Q13 | MDTS (Mirror Descent) | ~110 lines | 30-50% gap closure | 1.5x memory |
| Q14 | Bootstrap (Generational) | ~30 lines | 7.4→2.4-4.4% | 5x compute |
| Q15 | QAD Multi-Teacher Distill | 2-3 weeks | 3-5% PPL | Needs FP16 teacher |
| Q16 | Attention Relation Distill | 2-3 weeks | 1.4-2.1% | Needs FP16 teacher |
| Q17 | TCD (Critical Dynamics) | easy | Transformative | Theoretical |
| Q18 | Gumbel+STE noise | easy | 4.5x faster, 98% gap | Needs validation |

**Q9: Tequila** — Phase 18-C had isinstance() bug (silently failed). Bug now fixed. Repurpose deadzone weights as learnable dynamic biases. Tencent 2024 validated >4% gain.

**Q10: TPSQ** — Input-distribution-aware thresholds: threshold_i = base/(1+β·σ_i²). Information-theoretically correct (matched filter theory). ~10 lines.

**Q11: IAQD** — Learn per-feature gate x_corrected = x·sigmoid(W_gate·x + b_gate). Corrects INPUTS not weights. 23K params at 50M (0.046%). Novel approach.

**Q13: MDTS** — Replace STE with categorical distribution optimizer using KL divergence. Provably optimal for discrete distributions. 30-50% gap closure theoretical. 1.5x memory (2 logits/weight).

**Q17: TCD** — Treat model as 3-state Potts at critical point. Add learnable temperature β per layer. Gap IS finite-size scaling with ν=0.5. Unifies PTP, angular gap, STE, max entropy.

### Inference (6 findings)

| # | Finding | Effort | Impact | Risk |
|---|---------|--------|--------|------|
| I3 | WPH (Weight Pattern Hashing) | 1-2 weeks | 2.8-3.6x guaranteed | Low |
| I4 | Columnar Delta Inference | 2-3 weeks | 5-15x | Text-dependent |
| I5 | SCS (Sorted Coordinate Seriation) | 30 min measure | 64-340x if B_avg low | Needs test |
| I6 | RTDC (Row-TSP Differential) | 2-3 weeks | 1.5-38x | Data-dependent |
| I7 | Eigenmode Decoupling | 2-3 weeks | 1.5-3x | Medium |
| I8 | POPCOUNT Attention | 2-4 weeks | 5-10x attention | Activation quant |

**I3: WPH** — Group columns by ternary pattern, pre-sum activations per group. GUARANTEED 2.8x minimum (exact computation). Full-precision activations (no quant error like T-MAC). Dual of T-MAC.

**I5: SCS** — Reorder columns so +1/-1 form contiguous blocks. Compute via prefix sums. 64-340x depends on B_avg. **CRITICAL**: Single 30-min measurement on existing 50M weights determines if viable.

**I8: POPCOUNT Attention** — Quantize Q/K to ternary, compute similarity via bitwise AND + POPCOUNT. 5-10x faster attention. EXACT for ternary values. Three independent findings converge (M2, M6, M9).

### Architecture (2 findings)

| # | Finding | Effort | Impact | Risk |
|---|---------|--------|--------|------|
| A4 | d_state=64 | config | 2-4% quality | 4x state memory |
| A5 | Phase-Space Attention | 2 weeks | 32x KV reduction | Quality unknown |

**A4: d_state=64** — Config change d_state=16→64. More expressive SSM dynamics. Should test at 50M. 4x state memory tradeoff.

**A5: Phase-Space Attention** — Attend in SSM state space (dim=16) instead of token space (dim=512). 32x KV cache reduction. SSM learns coordinate system, attention uses it for retrieval.

### Memory (3 findings)

| # | Finding | Effort | Impact | Risk |
|---|---------|--------|--------|------|
| M3 | Group Codebook Training | 3-4 weeks | 100x optimizer (7B on 12GB!) | Quality TBD |
| M4 | Layer Streaming + mmap | 3-4 weeks | 3.3x VRAM | PCIe latency |
| M5 | XOR-Delta Multi-Model | 4-6 weeks | 8.2x storage | Needs variants |

**M3: Group Codebook** — Vector-quantize weight groups (K=256-65K prototypes). Train codebook only. 7B optimizer: 70→0.7 GB. First time 7B fits on 12GB GPU. Validate: cluster existing 50M weights (30 min).

---

## TIER 3: HARD/DEFER (Scale-dependent or high risk)

### Requires 50M+ Validation
| # | Finding | Impact | Why Defer |
|---|---------|--------|-----------|
| H1 | Discretized Decay (shift-add states) | Entire fwd pass multiply-free | Requires retraining |
| H2 | WFST ternary transitions | Replaces SSM+attention | Major arch change |
| H3 | Compiled Native Binary | CPU 200-400 tok/s at 7B | 8-12 weeks, platform-specific |
| H4 | FP4 Tensor Core Mapping | 2.4-3.2x, beats FP16 | Conflicts with INT8 path |
| H5 | INT8 Full Pipeline | GPU 2-3.3x, unlocks NPU | 4-6 weeks, conflicts FP4 |
| H6 | Tensor-Network Decomposition | 10-100x theoretical | Speculative, needs TT-rank |
| H7 | ARLC (Layer Composition) | 1.5-3x via INT8 cores | Memory cost, complex |
| H8 | CPI (Constraint Propagation) | Up to 6400x on LM head | Bound tightness unknown |

### Requires 1B+ / Theoretical
| # | Finding | Impact | Why Defer |
|---|---------|--------|-----------|
| H9 | Ensemble 4 ternary | +17.7% vs single FP16 | Needs scale for memory advantage |
| H10 | Fisher-Rao power-law breakdown | Gap collapse at 10B | Purely theoretical |
| H11 | Thermodynamic free energy | Ternary > FP16 at 2-5B | Needs scale proof |
| H12 | Precision-Topology Co-Learning | 2-3x via hetero precision | Major training change |
| H13 | Reservoir + Learnable Readout | Instant scaling | Unclear if works for LLMs |

### Similar to Failed Techniques (LOW PRIORITY)
| # | Finding | Similar To | Why Risky |
|---|---------|-----------|-----------|
| X1 | Shadow-Weight Self-Distill | Deferred ternary (+6.2%) | Uses shadow weights |
| X2 | ReSTE (Rectified STE) | HESTIA/BiSTE (failed) | STE variant |
| X3 | LeanQuant Asymmetric | OptRot (+4.2%) | Learned asymmetry |
| X4 | Per-Channel Steering | OptRot (+4.2%) | Learned modulation |

---

## ZERO-COST VALIDATIONS (Run on Existing 50M Model)

| Priority | Experiment | Time | Determines |
|----------|-----------|------|-----------|
| 1 | TRD: Count duplicate ternary rows | 30 min | 1.1-1.4x free speedup |
| 2 | SCS: Measure B_avg after seriation | 30 min | If 64-340x is real |
| 3 | SSS: Expected embedding forward | 30 min | 1.5-2.5x speculation |
| 4 | Codebook clustering (k-means) | 30 min | If 7B fits on 12GB |
| 5 | Evolutionary polish (weight flip) | 1 day | 1-3% free PPL |
| 6 | INT8 activation quantization PPL | 2-3 hrs | Integer pipeline viability |

---

## IMPLEMENTATION ORDER

### Round 1: Easy Wins (1-2 weeks)
1. **A1 MoA** — Zero-cost config change (1 hour)
2. **Q1 CAQ** — 4 lines of code (1 hour)
3. **Q5 Context-as-Precision** — A_log init change (1 hour)
4. **Q2 BDT** — Drop-in TernaryQuantize replacement (2 hours)
5. **Q6 BATR-STE** — Drop-in STE replacement (2 hours)
6. **Q3 BAGR** — Gradient reweighting (half day)
7. **Q4 ASGKS** — Kernel-smoothed gradient (half day)
8. **A2 Head Gating** — Per-head sigmoid (2 days)
9. **A3 Sink Tokens** — Global attention for BOS (1 day)
10. **I1 TRD** — Row dedup analysis (30 min validation)
11. **M1 SSD Checkpointing** — One-line backward change (1 day)
12. **Q8 Evolutionary Polish** — Post-training on existing 50M (1 day)

### Round 2: Medium (2-4 weeks)
13. **Q9 Tequila Re-test** — Bug fixed, 3-5 days
14. **Q10 TPSQ** — 10 lines, info-theoretic (1 day)
15. **Q11 IAQD** — Input-adaptive gates (3 days)
16. **I3 WPH** — Weight pattern hashing (1-2 weeks)
17. **I5 SCS** — Validate B_avg first (30 min), then implement
18. **I8 POPCOUNT Attention** — Validate activation quant first
19. **A4 d_state=64** — Config change + 50M training run

### Round 3: Deeper (4-8 weeks)
20. **Q13 MDTS** — Mirror descent optimizer
21. **Q14 Bootstrap** — Generational self-distillation
22. **Q15 QAD Distillation** — Multi-teacher (needs FP16 teacher)
23. **I4 Columnar Delta** — Delta inference
24. **I6 RTDC** — Row-TSP differential
25. **A5 Phase-Space Attention** — SSM-coordinate attention
26. **M3 Group Codebook** — Optimizer compression

### Final: 50M Validation
- Run ALL Round 1+2 winners combined at 50M
- Compare against Phase 36 baseline (7.41 loss)
- Measure: loss, GPU tok/s, CPU tok/s, VRAM

---

## CROSS-CUTTING SYNERGIES

| Combo | Expected | Why Multiplicative |
|-------|----------|-------------------|
| CAQ + BAGR + ASGKS | 3-6% PPL | All target boundary optimization differently |
| WPH + Columnar Delta | 10-50x inference | Hash only changed columns |
| SCS + RTDC | 64-340x GEMV | Column + row reordering (orthogonal) |
| MoA + Sink Tokens | Better ICL | Heterogeneous windows + global positions |
| Context-as-Precision + d_state=64 | 4-8% PPL | More dimensions × temporal accumulation |
| POPCOUNT Attention + Phase-Space | 32-160x attention | Ternary compute + compressed coordinates |
| Bootstrap + QAD Distillation | 5-10% gap closure | Self-distill + teacher-distill (stacking) |
