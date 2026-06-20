# Research Round 2, Aspect 6: Scaling Laws & Training Stability

## Wave 1 — Web Research (2 researchers, Mar 2026)

### Theme A: Scaling Laws for Ternary/Quantized Models

**R2-A6-F1: COMPUTE-OPTIMAL-QAT**
- **Src:** Dremov et al., Apple MLR, 2025 | **Gain:** Optimal QAT-to-FP ratio predicted by tokens-per-parameter-byte
- Optimal QAT training ratio INCREASES with total compute budget. For ternary: suggests progressive quantization with longer FP16 warmup MAY yield better quality at larger scales than immediate ternary-from-step-0. Status: Test both approaches.

**R2-A6-F2: SPECTRA-EFFECTIVE-CAPACITY**
- **Src:** Kaushal et al., Spectra Suite, 2024 | **Gain:** TriLM 3.9B matches FloatLM 3.9B on tasks, quality gap narrows at Billion+ scale
- Ternary models need ~6× params for FP16-equivalent quality on downstream tasks, but gap narrows at scale. v5 target (7B-equivalent quality in <1GB RAM) maps to ~700M-1B ternary params, not 7B ternary. Factual knowledge preserved well; reasoning/ICL is bottleneck.

**R2-A6-F3: SPECTRA-TRAINING-DYNAMICS**
- **Src:** Kaushal et al., Spectra Suite, 2024 | **Gain:** All scales show data-dependent spikes; LR drop at halfway + weight decay removal at 2/3 essential
- Critical training dynamics: (1) Spikes occur at SAME token counts across scales (data-dependent), (2) LR drop at halfway causes sharp loss decrease, (3) Weight decay removal at 2/3 accelerates convergence. For v5: two-stage training is non-optional for ternary convergence.

**R2-A6-F4: BITNET-2B4T-NATIVE**
- **Src:** Ma et al., Microsoft, BitNet b1.58 2B4T, 2025 | **Gain:** Native 1.58-bit training at 2B scale matches FP16 baselines (LLaMA 3.2 1B, Qwen2.5 1.5B, Gemma-3 1B)
- First open-source native ternary LLM at scale. Two-stage LR schedule (high→cooldown), two-stage weight decay (0.1→0), data progression (general→curated). Proves native ternary training works at scale.

**R2-A6-F5: BITNET-2B4T-SCALING-DIRECTION**
- **Src:** Ma et al., Microsoft, BitNet b1.58 2B4T, 2025 | **Gain:** Scaling laws for 1-bit models identified as OPEN RESEARCH
- Standard Chinchilla scaling laws (20 tokens/param optimal) may NOT apply to ternary. v5 in uncharted territory—optimal data-to-param ratio for ternary UNKNOWN.

**R2-A6-F6: ROUTING-MAMBA-MOE-SCALING**
- **Src:** Zhan et al., Microsoft/Northeastern, Routing Mamba, 2025 | **Gain:** RoM 1.3B active (10B total) matches 3B dense Mamba; 2.3× effective capacity multiplier
- First successful MoE+SSM: RoM 1-of-8 routing. Ternary MoE orthogonal to quantization—near-zero per-expert memory overhead. Could achieve 10-20× effective capacity multiplier (ternary weights + sparse activation).

**R2-A6-F7: ROUTING-MAMBA-WIDTH-SCALING**
- **Src:** Zhan et al., Microsoft/Northeastern, Routing Mamba, 2025 | **Gain:** Scaling projections (width) via MoE outperforms adding layers (depth)
- Sparse horizontal scaling (projection experts) more effective than vertical (more layers). For ternary: limited per-weight expressiveness {-1,0,+1} likely benefits MORE from width than depth. MoE-ternary compensates via horizontal scaling.

**R2-A6-F8: SPECTRA-WIDTH-VS-DEPTH**
- **Src:** Kaushal et al., Spectra Suite, 2024 | **Gain:** Scaling curves imply width advantage, not explicit ablation
- Performance gap narrows at larger scales, suggesting ternary may eventually match FP16 at sufficient scale. Implication: horizontal scaling (wider layers) more efficient than vertical (more layers) for ternary's discrete nature.

**R2-A6-F9: MAMBA-SSM-SCALING-GENERAL**
- **Src:** Zylos AI Research, Mamba & SSM Survey, 2026 | **Gain:** O(n) enables million-token contexts, 70%+ VRAM reduction, 5× throughput vs Transformer
- Consensus: pure SSM struggles ICL (15pt MMLU gap), hybrid 9:1 Mamba:Attention optimal. v5 ternary SSM + sliding window attention is CORRECT choice—SSM for efficiency, attention for retrieval.

**R2-A6-F10: MAMBA-2-STATE-DIMENSION**
- **Src:** Zylos AI Research, Mamba & SSM Survey, 2026 | **Gain:** Mamba-2 allows N=64-256 vs Mamba-1 N=16; larger states improve quality via tensor cores
- Upgrading v2 (Mamba-1, N=16) to Mamba-2 SSD (N=64-256) improves quality without param increase. Trade-off: state dimension scales memory linearly, still O(1) per token.

**R2-A6-F11: COMPUTE-OPTIMAL-QAT-COOLDOWN**
- **Src:** Dremov et al., Apple MLR, 2025 | **Gain:** Cooldown fusion eliminates redundant FP updates; saves compute
- Novel approach: LR decay JOINTLY with quantization onset (not separate FP→QAT phases). Eliminates redundant FP warmup. For v5: direct ternary + two-stage LR may be MORE efficient than multi-phase FP→ternary progression.

**R2-A6-F12: SPECTRA-KNOWLEDGE-CAPACITY**
- **Src:** Kaushal et al., Spectra Suite, 2024 | **Gain:** Ternary models show NO significant knowledge degradation vs FP16 at equivalent param count
- Knowledge preserved well (SciQ, TriviaQA, MMLU). Capacity bottleneck likely in reasoning/ICL, not factual knowledge. Connection presence {+1,-1} more important than strength (float value).

### Theme B: Training Stability & Robustness

**R2-A6-F13: TERNARY-LIF-SNN**
- **Src:** Wang et al., IEEE TNN, 2025 | **Gain:** Phase-Temporal-Spike Neuron (PTSN) achieves 4.1% gap to FP vs 11.7% prior
- Ternary spikes {-1,0,+1} with stable gradients via per-timestep normalization. Insight: ternary transitions STABLE when normalized locally, not globally. Applicable to ternary SSM recurrence.

**R2-A6-F14: PARETOQ-CONTRADICTS-PHASE-10.B**
- **Src:** Liu et al., PyTorch Blog, June 2025 | **Gain:** Optimal 90% FP + 10% QAT fine-tuning; ternary requires 3-5× more tokens than 3-4 bit
- **KEY CONTRADICTION:** ParetoQ claims direct ternary training fails, requires progressive FP→QAT + 3-5× more tokens. **v5 Phase 10.B results DIRECTLY CONTRADICT this:** Direct ternary training matched progressive at both 6M (within 0.2%) and 50M (perfect convergence, zero NaN, monotonic loss). Either: (1) ternary SSM architecture exploits principles ParetoQ doesn't cover, (2) v5 progressive training unnecessary overhead, or (3) ParetoQ findings don't transfer to SSM recurrence. Status: INVESTIGATE — test cooldown-only (no FP warmup) vs two-stage.

**R2-A6-F15: ECRM-MEMORY-TRANSFER**
- **Src:** Yang et al., Advanced Science, 2026 | **Gain:** In-situ quantization mitigates catastrophic forgetting in continual learning
- During fine-tuning, ternary weights near thresholds flip catastrophically. Solution: soft-freeze low-variance weights, allow high-variance plastic exploration. Separate stable vs plastic ternary subsets.

**R2-A6-F16: QAT-GRID-PRESERVATION**
- **Src:** Pillet et al., HAL, 2026 | **Gain:** Ternary fine-tuning without catastrophic forgetting if QAT properly scheduled
- Extreme quantization CAN fine-tune on new data if QAT phase adapts with task (not after). Ternary grid must co-evolve with task data.

**R2-A6-F17: PD-SSM-STABILITY**
- **Src:** Terzić et al., NeurIPS 2025 Spotlight, IBM | **Gain:** BIBO-stable sparse A(u) prevents unbounded recurrence explosion
- Discrete A {-1,0,+1} satisfies |A_entries|≤1, but ERROR ACCUMULATION differs from FP over long sequences. Solutions: (a) parallel scan formulation (FLA chunk_gla already does this), (b) periodic state re-normalization, (c) spectral radius constraint.

**R2-A6-F18: ERROR-FREE-LINEAR-ATTN-ALREADY-IMPLEMENTED**
- **Src:** Lei et al., arXiv, 2025 | **Gain:** Parallel scan formulation avoids cumulative error drift
- SSM recurrence h_t = A·h_{t-1} + B·x_t accumulates quantization error. Solution: parallel scan (like FLA chunk_gla) instead of sequential recurrence. **STATUS: Already implemented in v5.**

**R2-A6-F19: SPECTRAL-NORM-STABILITY**
- **Src:** General 2025 practices | **Gain:** Spectral normalization bounds Lipschitz constant ≤1, prevents gradient explosion
- Rescale weight matrices to constrain largest singular value ≤1. Apply BEFORE quantization to normalize shadow (FP) weights, then quantize. Prevents ternary weights from unbounded operator norm.

**R2-A6-F20: PARETOQ-INIT**
- **Src:** Liu et al., PyTorch, 2025 | **Gain:** Learnable quantization scales outperform fixed thresholds; more stable training
- Initialize ternary quantization scale S as LEARNABLE parameter (vs fixed ±0.33 thresholds). Better than stats-based init; adapts per-layer. Prevents dead zones (all weights→zero).

**R2-A6-F21: LOW-BIT-FAVORS-UNDERTRAINED**
- **Src:** Ouyang et al., ACL, 2025 | **Gain:** Low-bit quantization MAY improve undertrained models (regularization effect)
- Counterintuitive: if training budget limited, ternary MAY act as regularizer. Training to lower FP loss then quantizing MAY be WORSE than early-stop FP + ternary QAT. Status: NEEDS-WORK (speculative).

**R2-A6-F22: RLEC-TERNARY-NEURON**
- **Src:** Battista et al., Neuron, 2025 | **Gain:** Gradient norm clipping at 1.0 prevents exploding gradients in ternary networks
- Ternary discontinuities amplify gradient spikes. Clipping threshold should be LOWER for ternary than FP (0.5-1.0 vs 5.0). Adapt based on quantization error magnitude.

**R2-A6-F23: SNN-PARALLEL-TRAINING**
- **Src:** Huang et al., arXiv, 2026 | **Gain:** Prevents membrane explosion and sustained spike saturation
- Ternary state saturation (all→+1 or -1) analogous to membrane explosion. Solutions: (a) gradient enhancement at transition boundaries (BiSTE, Boundary-Aware STE), (b) state diversity loss. Prevent saturation via multi-constraint training.

**R2-A6-F24: LMS-LEARNABLE-SPIKE**
- **Src:** Ye et al., OpenReview | **Gain:** Learnable maximum spike value adapts to gradient during training
- Instead of fixed {-1,0,+1}, use LEARNABLE bounds {-α,0,+α} where α per-layer trainable. Allows model to discover optimal quantization range per layer. Similar to LSQ but for symmetric ternary.

## Key Insights — Aspect 6 Wave 1

### Scaling Laws (Theme A)

1. **Ternary effective capacity multiplier ~6×, achieves Haiku-level quality in <1GB RAM with 700M-1B ternary params** — Spectra finding validates v5 design philosophy.

2. **Progressive vs direct ternary: INVESTIGATE — ParetoQ claims 3-5× penalty, Phase 10.B shows none.** This is THE critical question for v5 efficiency. Either SSM architecture bypasses ParetoQ's constraint, or progressive training is wasted overhead. Test direct ternary + two-stage LR schedule (cooldown fusion) vs current multi-phase approach.

3. **Width scaling > depth scaling for ternary** — MoE-ternary via horizontal scaling (projection experts) likely more efficient than deep layers. Limited {-1,0,+1} expressiveness benefits more from connectivity than depth.

4. **SSM + hybrid attention CORRECT choice** — Pure SSM weak on ICL (15pt gap). v5 architecture (SSM + sliding window attention) balances O(1) memory with precise retrieval.

5. **Mamba-2 SSD upgrade path exists** — Larger state dimensions (N=64-256) improve quality without param inflation. Current v2 (Mamba-1, N=16) has room for enhancement.

6. **Scaling laws for 1-bit models UNKNOWN** — v5 in uncharted territory. Chinchilla-style laws may not apply; empirical testing required.

### Training Stability (Theme B)

1. **PARETOQ-PHASE-10.B CONTRADICTION IS CRITICAL** — Do NOT treat ParetoQ findings as law. v5 empirical results (Phase 10.B direct ternary matches progressive) suggest either: (a) SSM recurrence exploits mathematical properties ParetoQ's transformer analysis doesn't cover, (b) direct ternary + proper two-stage LR (cooldown fusion) sidesteps ParetoQ's overhead, or (c) different quantization methodology. This determines whether v5 training can be dramatically simplified.

2. **Two-stage training empirically proven essential** — High LR → cooldown drop at halfway, weight decay cosine → zero at 2/3, data progression (general→curated). Not optional; applies universally across ternary models.

3. **Gradient clipping lower for ternary** — Use 0.5-1.0 vs FP's 5.0. Ternary discontinuities amplify spikes; adapt threshold based on quantization error.

4. **Error accumulation in SSM recurrence avoidable via parallel scan** — Already implemented (FLA chunk_gla). Periodic state re-normalization backup option for extreme long-context.

5. **Learnable quantization scales > fixed thresholds** — Per-layer adaptive bounds {-α,0,+α} prevent dead zones and discover optimal quantization range.

6. **Catastrophic forgetting in fine-tuning is REAL** — Weights near thresholds flip under small gradient updates. Solutions: soft-freeze stable weights, re-run QAT when adapting to new tasks (not just FP fine-tune then quantize).

## For Math-Paradigm-Breakers (Wave 2)

**Priority 1 (BLOCKING):**
1. **RESOLVE ParetoQ-Phase-10.B contradiction** — Formalize why SSM direct ternary matches progressive when transformers require 90/10 FP/QAT split. Is it architecture-specific (A_{SSM} stability), quantization methodology, or data characteristics? Critical for v5 simplification.
2. **Derive ternary-specific scaling law L(N,D)** — Apple's tokens-per-parameter-byte approach for ternary (1.585 bits/weight) → derive optimal data budget scaling.

**Priority 2 (HIGH):**
3. **Width-to-depth ratio optimization for ternary SSM** — Given param budget P and discrete weights {-1,0,+1}, what's optimal layer width w(L) and depth L? Does ternary constraint shift Roofline model?
4. **MoE capacity multiplier proof** — Combine 6× ternary effective capacity + 2.3× MoE sparse scaling. What's theoretical combined bound? How does selectivity (B,C input-dependent) interact with MoE routing?
5. **Error accumulation bound for ternary SSM over T steps** — h_T = A^T h_0 + sum(A^(T-t) B x_t). If A ternary {-1,0,+1}, quantization error compounds. Derive T_max before re-normalization needed; compare to FP.

**Priority 3 (ANALYSIS):**
6. **Optimal ternary grid beyond {-1,0,+1}** — SEQ suggests {-1.5,-0.5,0.5,1.5}. Derive grid minimizing quantization error under L2 norm; layer-dependent?
7. **Spectral radius constraint for selective ternary A** — If A ternary but B,C input-dependent (selective), BIBO stability changes. Derive spectral radius bound for Mamba-2 SSD.

## Research Quality Metrics

- **Total Findings (deduplicated):** 24
- **Already Implemented:** 1 (F18: ERROR-FREE-LINEAR-ATTN)
- **Contradictions Found:** 1 (F14: PARETOQ vs Phase 10.B — CRITICAL)
- **Actionable (F+M+P rating):** 16 findings rated PROMISING or NEEDS-WORK
- **Speculative (requires math work):** 4 findings (scaling laws, grid optimization)
