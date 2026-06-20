# R9 — Research Round Summary

**Goal:** tokens_to_converge reduction (days → minutes, convergence axis)
**Formula:** total_time = tokens_to_converge × time_per_token
**Date:** 2026-03-06
**Model:** v7 (ternary SSM, Lion, 6M params, 45,225 tok/s baseline)

---

## Totals

| Aspect | Focus | IMPLEMENT | EXPLORE | SKIP | Total |
|--------|-------|-----------|---------|------|-------|
| A1 | Curriculum learning | 15 | 5 | 0 | 20 |
| A2 | Data selection & quality | 13 | 6 | 1 | 20 |
| A3 | LR meta-strategies | 17 | 2 | 1 | 20 |
| A4 | Second-order & gradient quality | 18 | 1 | 1 | 20 |
| A5 | Loss landscape engineering | 15 | 5 | 0 | 20 |
| A6 | Regularization & generalization | 18 | 2 | 0 | 20 |
| **Total** | | **96** | **21** | **3** | **120** |

---

## Critical Insights by Aspect

### A1 — Curriculum Learning
- **PBS-ONLINE-DIFFICULTY-SAMPLER:** replaces static CurriculumSampler (LEGO); online linear predictor with 4 features; 6-13% step reduction
- **DISCRETE-OT-TERNARY-LATTICE:** score = fraction of uncommitted params where sign(g) agrees with sign(mom); Hoeffding bound → 70% alignment halves expected Hamming distance; 10-18%
- **SIGN-COHERENCE-CURRICULUM:** Lion momentum sign coherence as difficulty metric, O(seq_len) at data load time; 8-18%
- **EFFECTIVE-CAPACITY-PACING:** q(t) = max(0.4, 1-f(t)); tracks shrinking VC dim as AutoFreeze commits; novel (no prior work accounts for progressive capacity reduction)
- **SSM-COLDSTART-LOSS-UPWEIGHT:** upweight positions t≥T_mix=64; SSM h≈0 for first 64 tokens → noisy gradients; 5-10%
- **Unified 5-level scheduler:** PBS × commitment × Fisher × OT × capacity pacing

### A2 — Data Selection & Quality
- **Preprocessing pipeline (one-time cost):** n-gram repetition filter → perplexity band [p10,p85] → MinHash dedup → DSIR importance resampling (15-25%)
- **ZLIB-RESIDUAL-MDL-FILTER:** adaptive — mastered sequences drop below threshold automatically as training progresses; 10-20%
- **TERNARY-THRESHOLD-FRONTIER-PRIORITY:** TFA(x) = sum_{j in frontier} sign(g_j)×sign(w_j); directly minimizes commitment hitting time; 8-15%
- **UNCOMMITTED-GRADIENT-FRACTION-RESAMPLING:** I_uncomm = ||g_uncomm||₁/||g_total||₁; weights sampling by STE utility; 5-15%
- **DOREMI-SELF-ONLINE-DOMAIN-WEIGHTS:** reference-free adaptation (rho1.py already implemented but external-ref gated)
- **SSM-BACK-LOADING-INFORMATIVE-TOKENS:** gradient decays as ρ^(T-t); place rare tokens LATE in packed sequence; 3-8%

### A3 — LR Meta-Strategies
- **COMMITMENT-RATE-PID-LR-CONTROLLER:** PID over d(committed_fraction)/dt; target r* = fraction_to_commit/T_budget; K_p=0.2, K_i=0.001; 10-20%
- **UNCOMMITTED-GAP-ADAPTIVE-LR:** lr_t from hitting-time theory over |w_uncomm| distribution; replaces fixed WSD schedule; 10-25%
- **CAPACITY-NORMALIZED-LR:** LR × (n_total/n_free)^0.35 as weights commit; 8-18%
- **LION-SIGN-NOISE-INDEPENDENCE:** Lion sign-error rate is INDEPENDENT of lr (unlike SGD/Adam); lr_max = 2×mean(|g|)
- **FLIP-RATE-ALWAYS-ACTIVE:** existing use_two_stage_flip_lr LEGO → remove guard
- **W1-AgB-F2 SUPERSEDED:** β₁ direction correction from W2 (direction was wrong)
- Combined formula: η = η_schedule × capacity_mult × pid × avalanche_brake × σ_g_decay

### A4 — Second-Order & Gradient Quality
- **Structural result:** STE Hessian is DIAGONAL (piecewise constant → cross-partials vanish). Simplifies all curvature findings.
- **MARS-LION-VARIANCE-REDUCTION:** STORM prev_grad correction; c_t = g + γ×(β₁/(1-β₁))×(g-g_prev); 20-30%
- **COMMITTED-GRAD-NOISE-FLOOR:** committed weights E[g]=0 → pure noise calibration σ²_noise = mean(g_committed²); 8-15%
- **GRAD-CENTRALIZATION:** subtract row-mean from 2D grad; B_eff 1.82→3.48 for Lion; 15-25% early
- **LAPLACE-SIGN-ERROR-MOMENTUM-GATE:** gate exp_avg write when P(sign error)>35%; 8-15%
- **HESTIA-TEMP-STE-HARDENING:** softmax relaxation → progressive hardening; 10-20% first 30% of steps
- **SSM-SPECTRAL-IS-REWEIGHTING:** IS correction ρ^{-(T-t)} per chunk; corrects positional gradient bias for A_log; 5-15%
- **LION-LN-OUTPUT-ADAM-SPLIT:** Adam for LN+lm_head, Lion for matrices; 10-20%

### A5 — Loss Landscape Engineering
- **Critical finding:** A_log, B, dt receive ρ^{T-t}≈0.006 gradient attenuation from CE alone. Auxiliary losses that bypass this are high-value.
- **LEGO fixes:** label_smoothing (default 0.0→0.05), mtp_heads (n=2 always-active)
- **PREDICTIVE-SSM-STATE-CODING:** h_pred = exp(k×A×dt)×h_t; L2 vs h_{t+k}; O(1)-depth gradient to A_log; 8-18%
- **INFONCE-SSM-STATE-MI:** contrastive I(h_t; x_{t+1:k}) using embedding table; gradient reaches B, C, A_log; 10-20%
- **TERNARY-BASIN-DEPTH-LOSS:** L_basin = λ×mean(d_i²), d_i = ||w|-δ_g|; Voronoi push toward nearest attractor; 5-12%
- **SSM-PROJECTION-FIRST-COMMIT:** x_proj amplifies gradient noise 50×; lower n_stable for x_proj; 8-15%
- **SPECTRAL-RADIUS-ENTROPY-LOSS:** L_sp anchors T_mix=64 from R8-A6, forces multi-timescale head specialization; 6-14%
- **TERNARY-MANIFOLD-COHERENCE-REG:** L_coh = ||W_fp - α×W_q||²; Tikhonov pull toward ternary manifold; 8-15%

### A6 — Regularization & Generalization
- **Paper-quality analytical result 1:** f* = log(3)/(1+log(3)) ≈ 0.524 — PAC-Bayes optimal committed fraction for ternary networks (universal constant, independent of n, m, data)
- **Paper-quality analytical result 2:** n_stable(A_log) = n_stable_base × √T ≈ 45× base (T=2048) — A_log commits cost T×log(3) KL vs log(3) for feedforward
- **GAUSSIAN-MIXTURE-PROXIMAL-REG:** GM prior gives exact ternary basin bifurcation at |w|≈0.35; proximal pull at correct strength; 8-15%
- **TERNARY-OT-DISTRIBUTION-MATCHING:** Wasserstein-1 enforces global cardinality balance; when p_0=80%, forces redistribution (vs Voronoi allows 80% at 0); 8-18%
- **NGPT-SHADOW-ROW-UNIT-NORM:** post-step row-L2-norm; makes LSQ scale interpretable; 10-20%
- **HADAMARD-PRE-QUANTIZE:** Walsh-Hadamard → Gaussian → min MSE quantization (JL-lemma); 5-15%
- **PAC-BAYES-OPTIMAL-f*-TARGET:** threshold controller drives f toward 0.524; 5-15%
- **SPECTRAL-ADAPTIVE-SSM-DROPOUT:** p_opt = (1-ρ²)/(2-ρ²); uniform SNR across BPTT depths; 6-12%

---

## Top 10 Highest-Impact Findings (Implementation Priority)

| Rank | ID | Aspect | Gain | Mechanism |
|------|----|--------|------|-----------|
| 1 | MARS-LION-VARIANCE-REDUCTION | A4 | 20-30% steps | STORM correction, no extra backward |
| 2 | INFONCE-SSM-STATE-MI | A5 | 10-20% steps | direct gradient path to A_log, B, C |
| 3 | PREDICTIVE-SSM-STATE-CODING | A5 | 8-18% steps | O(1)-depth gradient to A_log |
| 4 | COMMITMENT-RATE-PID-LR | A3 | 10-20% steps | drives commitment rate to target |
| 5 | UNCOMMITTED-GAP-ADAPTIVE-LR | A3 | 10-25% steps | replaces WSD with geometry-driven schedule |
| 6 | GRAD-CENTRALIZATION | A4 | 15-25% early | B_eff 1.82→3.48 for Lion |
| 7 | DSIR-NGRAM-RESAMPLING | A2 | 15-25% steps | importance-weighted corpus (one-time) |
| 8 | NGPT-SHADOW-ROW-NORM | A6 | 10-20% steps | uniform geometry for quantization |
| 9 | DISCRETE-OT-TERNARY-LATTICE | A1 | 10-18% steps | sign-agreement sampling |
| 10 | TERNARY-OT-DISTRIBUTION-MATCHING | A6 | 8-18% steps | global cardinality balance |

---

## LEGO Flags Found in v7 Codebase (Remove These)

| Flag | Finding | Always-active behavior |
|------|---------|----------------------|
| `label_smoothing=0.0` | R9-A5 | hardcode α=0.05 in CrossEntropyLoss |
| `mtp_num_heads=0` | R9-A5 | hardcode n=2 MTP heads |
| `use_rho1=True` gated on external ref | R9-A2 | self-referential EMA eliminates external dependency |
| `use_two_stage_flip_lr` LEGO | R9-A3 | remove guard, always-active |

---

## Combined Gain Estimate (conservative)

| Axis | Techniques | Conservative Estimate |
|------|-----------|----------------------|
| **tokens_to_converge** | A1+A2+A3+A4+A5+A6 stacked | 3-8× fewer tokens |
| **time_per_token** | R8 already covers | 4-8× faster per step |
| **Combined R8+R9** | | **12-64× total wall-clock reduction** |

*Gains overlap significantly. Smoke tests required per technique. Conservative combined estimate: 10-20×.*

---

## Files

| Aspect | Findings doc |
|--------|-------------|
| A1 | `docs/model-v7-architecture/research/r9-a1-curriculum-findings.md` |
| A2 | `docs/model-v7-architecture/research/r9-a2-data-selection-findings.md` |
| A3 | `docs/model-v7-architecture/research/r9-a3-lr-strategies-findings.md` |
| A4 | `docs/model-v7-architecture/research/r9-a4-gradient-quality-findings.md` |
| A5 | `docs/model-v7-architecture/research/r9-a5-loss-landscape-findings.md` |
| A6 | `docs/model-v7-architecture/research/r9-a6-regularization-findings.md` |

---

## Next Steps

1. **R10:** Effective batch size and memory — gradient accumulation, activation offloading, mixed precision, VRAM reduction
2. **R11:** Ternary-specific acceleration — popcount GEMM, bit-packing, quantization efficiency
3. **Implementation phase:** 75 (R8) + 96 (R9) = 171 IMPLEMENT findings pending
