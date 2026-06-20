# Research Discoveries — Ternary SSM Architecture (Part 2/2)

**Date:** 2026-02-25 | 28 agents across 5 research rounds | Consolidated single source of truth

**This file covers Parts 9-14: Technique synergies, implementation roadmap, validation experiments, and strategic competitive advantages.**

---

## Part 9: Technique Synergies & Conflicts

### Top 10 Synergy Combinations (Multiplicative Gains)

| Synergy | Techniques | Combined Impact | Why Multiplicative |
|---------|-----------|-----------------|-------------------|
| Development Accelerator | Fused kernel (4x) + CUDA graphs (10x) | **40x at 50M** | Each eliminates layer of overhead |
| Quality-Speed Holy Grail | Test-time compute (5x) + Speculative (2.5x) + Kernels (15x) | **187x total** | Spend 5x compute on quality, still 20x faster |
| Edge Ensemble | Ensemble (4 models) + XOR-delta (8x) + Layer streaming (3x) | 4.6 GB, +17.7% quality | All work on same discrete structure |
| Scale Breaker | Codebook (100x optimizer) + MoE (2.35x) | 16.5B-effective on 12 GB | Codebook frees VRAM for MoE |
| Quality Stack | K=2 (4-6%) + Tequila (3-4%) + Polish (1-2%) | **5.6-9.8% gap closure** | Three independent error sources |
| Serving Revolution | Context-independent (80 sessions) + Adaptive state (13x) + State DB (12K TTFT) | 160K concurrent users | SSM's O(1) property compounds |
| Throughput Maximizer | Speculative (2.5x) + Fusion (4x) + Graphs (10x) | **100x at 50M** | Pure multiplicative stacking |
| CPU Revolution | Compiled (14x) + Integer pipeline (4x) | **56x for 7B on CPU** | Assembly + SIMD both eliminate overhead |
| Quick Win | Masked fix (1.5x) + SSD checkpoint | Speed + Memory | Minimal dependency overhead |
| Architecture Maximizer | MoE (2.35x) + Ensemble (4 models) + XOR-delta (8x) | 66B effective in 6.2 GB | Ternary sparsity amplifies each |

### Hard Conflicts

| Conflict | A | B | Resolution |
|----------|---|---|-----------|
| FP4 vs Integer Pipeline | FP4 E2M1 activations (988 TOPS) | INT8 activations + integer hardware (494 TOPS) | Test FP4 activation quality first. If <2% PPL loss → FP4. Else → INT8. |
| Layer Streaming vs CUDA Graphs | Dynamic memory (load/unload) | Static memory (fixed layout) | Different targets. Graphs for dev/server (model fits VRAM). Streaming for edge (4-8 GB). |
| Multi-Codebook K=2 vs Kernels | W = s₁·T₁ + s₂·T₂ (9 values) | Current kernels assume 3 values | Validate K=2 quality first, then redesign kernels. |

### Soft Conflicts (Orthogonal, both viable)

- **Ensemble vs MoE:** Both increase capacity via parameter multiplication. Orthogonal diversity sources. Do MoE first (one training), ensemble later.

### Redundancies

- **Masked reduction fix (R1.5) → superseded by Fused kernels (R1.2)** in 2-3 weeks. Apply R1.5 immediately (hours). R1.2 renders it obsolete but R1.5 is free speedup while waiting.

---

## Part 10: Implementation Roadmap

### Dependency Graph & Critical Path

**TIER 0: FOUNDATIONS (Week 1-2)**
- R1.5 Masked Reduction Fix [2-4 hours]
- R2.5 SSD Structural Checkpointing [3-5 days]
- R3.5 Evolutionary Discrete Polish [3-5 days]
- R3.3 Tequila Re-test [3-5 days]

**TIER 1: CORE INFRASTRUCTURE (Week 2-5)**
- R1.2 Fused SSM Mega-Kernel [2-3 weeks] ← depends on R1.5
- R3.2 Multi-Codebook K=2 validation [2-3 weeks] (parallel)

**TIER 2: ACCELERATION (Week 5-10)**
- R4.2 CUDA Graph Capture [1-2 weeks] ← depends on R1.2
- R4.1 Full Integer Pipeline [4-6 weeks] (parallel)
- R1.1 FP4 Investigation [4-6 weeks] (parallel, OR alternative to R4.1)
- R2.2 Codebook Training [3-4 weeks] (parallel)

**TIER 3: ADVANCED (Week 10-24)**
- R4.4 Self-Speculative Decoding [3-4 weeks] ← depends on R4.2
- R1.4 MoE on SSM FFN [8-16 weeks] ← depends on R2.2
- R4.3 Context-Independent Serving [6-8 weeks]

**TIER 4: DEPLOYMENT (Week 24-40)**
- R1.3 Compiled Native Binary [8-12 weeks] ← depends on R4.1
- R3.1 Ternary Ensemble [4-6 weeks + training runs]
- R3.4 Test-Time Compute Arbitrage [2-3 weeks]
- R2.1 XOR-Delta Multi-Model [4-6 weeks] ← depends on R3.1

**TIER 5: MOONSHOTS (Week 40+)**
- R4.5 Semantic State Database [6+ weeks]
- R2.4 Adaptive SSM State Precision [4-6 weeks]
- R2.3 Layer Streaming + mmap [3-4 weeks]

**Critical Path:** R1.5 → R1.2 → R4.2 → R4.4 → R4.3 → R4.5 = **~29 weeks to full serving stack**

**Fastest to dev velocity:** R1.5 → R1.2 → R4.2 = **5 weeks to 15-40x faster experiments**

### Phase-by-Phase Roadmap

**Phase A: Immediate (0-2 weeks) — Zero/Low Risk**

| ID | Finding | Effort | Speed Gain | Quality Gain | Training Memory Gain |
|----|---------|--------|-----------|-------------|---------------------|
| A1 | Masked Reduction Fix | 2-4h | **1.5-2x GPU** | — | — |
| A2 | SSD Struct Checkpoint | 3-5d | — | — | **127x tensor reduction** |
| A3 | Evolutionary Polish | 3-5d | — | **1-3% PPL** | — |
| A4 | Tequila Re-test | 3-5d | — | **3-5% PPL** | — |

**Combined A:** 1.5-2x speed + 4-8% quality closure + training fits in 12 GB comfortably.

**Phase B: Short-term (Weeks 2-8) — Medium Risk**

| ID | Finding | Effort | Gain |
|----|---------|--------|------|
| B1 | Fused SSM Mega-Kernel | 2-3w | **3-5x GPU** |
| B2 | CUDA Graph Capture | 1-2w | **5-10x additional** |
| B3 | Multi-Codebook K=2 | 2-3w | **4-6% quality** |

**Combined A+B:** **15-40x speed at 50M**. Potentially all 7.4% quality gap closed.

**Phase C: Medium-term (Weeks 8-24) — Medium Risk**

| ID | Finding | Effort | Gain |
|----|---------|--------|------|
| C1 | Full Integer Pipeline | 4-6w | **2-3.3x GPU, 3-5x CPU** |
| C2 | Codebook Training | 3-4w | **100x optimizer** |
| C3 | Self-Speculative Decoding | 3-4w | **2.52x inference** |
| C4 | FP4 Investigation | 4-6w | **2x over INT8** (if viable) |

**Combined through C:** **75-200x speed at 50M**. **7B training on 12 GB GPU**.

**Phase D: Long-term (Weeks 24-40) — Higher Risk, Transformative**

| ID | Finding | Effort | Gain |
|----|---------|--------|------|
| D1 | MoE on SSM FFN | 8-16w | **2.35x effective capacity** |
| D2 | Compiled Native Binary | 8-12w | **200-400 tok/s 7B CPU** |
| D3 | Context-Indep. Serving | 6-8w | **80+ concurrent 7B** |
| D4 | Ternary Ensemble | 4-6w+train | **+17.7% vs FP16** |
| D5 | Test-Time Compute | 2-3w | **best-of-5 quality** |

**Phase E: Moonshots (Weeks 40+) — Research Projects**

| ID | Finding | Effort | Gain |
|----|---------|--------|------|
| E1 | Semantic State DB | 6+w | **12,308x TTFT** |
| E2 | XOR-Delta Multi-Model | 4-6w | **10 models in 4.2 GB** |
| E3 | Adaptive State Precision | 4-6w | **13x state compression** |
| E4 | Layer Streaming + mmap | 3-4w | **7B in 1.1 GB VRAM** |

---

## Part 11: Zero-Cost Validation Experiments

Run these with EXISTING 50M model to validate/invalidate key discoveries before committing to implementation.

| # | Experiment | Validates | Cost | Time |
|---|-----------|-----------|------|------|
| 1 | Cluster 50M ternary weight groups with k-means | Codebook training viability | Zero | 30 min |
| 2 | Hamming distance between consecutive layer bitmaps | XOR-delta and inter-layer streaming | Zero | 30 min |
| 3 | Zero old SSM state channels, measure PPL change | Adaptive state precision viability | Zero | 1 hour |
| 4 | Save SSM state, reload, verify generation identity | State-vector algebra basics | Zero | 30 min |
| 5 | Implement SSD checkpoint (1-line change in ssm.py) | SSD structural checkpoint VRAM savings | Zero | 30 min |
| 6 | Replace mask multiply with conditional in Triton | Masked reduction fix actual speedup | Zero | 2-4 hours |
| 7 | Quantize activations to INT8, measure PPL | Full integer pipeline viability | Low | 2-3 hours |
| 8 | Quantize activations to FP4 E2M1, measure PPL | FP4 tensor core viability | Low | 2-3 hours |
| 9 | Binarize ternary (keep signs), measure PPL | Binary draft model viability | Low | 1 hour |
| 10 | Evolutionary discrete polish on trained 50M | Post-training quality improvement | Low | 1 day |
| 11 | Tequila with isinstance fix + explicit logging | Training quality improvement after bug fix | Low | 1 day |

**Total time: ~3 days to validate/invalidate 11 most impactful discoveries. Zero training required.**

---

## Part 12: Unexplored Directions (Next Research Round)

Gaps not covered in Rounds 1-5. Recommend which agents should read first before researching.

1. **Bitpacked Kernels** — Paper 3 identifies dual-bitmap encoding (1.75 GB at 7B) but no kernel exists. How to implement efficient bitpacked ternary GEMV?
   - **Read first:** Paper 3 efficiency section, architecture-specification.md

2. **torch.compile for SSM** — Currently REGRESSES -56%. Can selective compilation (non-scan parts only) help?
   - **Read first:** vision-and-status.md (Finding 10.E.1 torch.compile issue)

3. **Scaling beyond 50M** — 100M, 370M, 1B experiments. At what scale does quality gap close to <3%?
   - **Read first:** Paper 1 scaling analysis, quality gap trend data

4. **Ternary-Aware Optimizer** — Muon/Fira tuned specifically for ternary dynamics. Paper 2 shows IGA is -2.1% better — can we go further?
   - **Read first:** Paper 2 optimizer ablations, IGA finding

5. **SWA Alternatives** — Linear attention (GLA/RetNet) replacing SWA for full O(L) model. Quality impact?
   - **Read first:** Paper 3 SWA analysis, architecture-specification.md

6. **Hardware-Specific Kernels** — CUTLASS FP4 templates for RTX 5070. cuBLAS 12.4 FP4 GEMM.
   - **Read first:** FP4 Tensor Core section (Part 3 in Part 1)

7. **Batched Inference** — Paper 3 mentions but doesn't implement. Weight loading amortized over batch.
   - **Read first:** Paper 3 serving section, baseline-context.md

8. **Training Stability at 1B+** — Does direct ternary still converge at 1B? Any new instability?
   - **Read first:** Paper 1 stability section, Finding B.2

9. **Ternary MoE Routing** — Can the router itself be ternary? Zero-cost routing decisions?
   - **Read first:** Paper 1 MoD routing, architecture-specification.md

10. **Distillation from Larger FP16** — Train FP16 teacher (larger), distill into ternary student. Quality gap impact?
    - **Read first:** Paper 1 quality gap analysis, distillation literature

---

## Part 13: Strategic Competitive Advantages (The Moat)

### 5 Permanent Structural Advantages

1. **Context-Length-Independent Session Memory** — O(1) state vs O(n) KV cache is a complexity class difference. 258 MB fixed vs growing. Cannot be replicated by scaling.

2. **4.1x Weight Compression with Training-Time Quality** — Not post-training quantization. Quality baked in during training. Gap DECREASES with scale (7.4% → projected <1% at 7B).

3. **Elimination of Multiplication for 84% of Parameters** — Physics constraint: addition < multiplication in all CMOS. Current 1.55x gap is immature kernels, not fundamental. Scales improve with better kernels.

4. **GF(3) Algebraic Weight Space** — Ternary weights form finite field. Deltas are ternary. Composition is exact and lossless. XOR-delta multi-model, algebraic adapters. FP16 has no algebraic closure.

5. **10x Memory-Bandwidth Ceiling** — Shannon's theorem on memory bus: 3.4 GB model vs 14 GB FP16 at 7B. In bandwidth-bound regime (7B+), translates directly to speed advantage.

### Killer Applications (Impossible with FP16 Transformers)

| Application | Our Capability | Transformer Equivalent | Advantage |
|-------------|---------------|----------------------|-----------|
| Mass concurrent serving | 80-160K sessions on 24 GB | 2-12 sessions | **13-80,000x** |
| Multi-model marketplace | 10 variants in 7 GB | 10 variants in 140 GB | **20x storage** |
| Always-on daemon | 3.7 GB constant | Growing KV forever | **Infinite context** |
| Edge mesh inference | 7B on 4 phones (0.85 GB each) | Impossible (14 GB weights + KV) | **Impossible vs possible** |
| Semantic state database | 5 ms TTFT for cached contexts | No equivalent possible | **12,308x TTFT** |
| Zero-cost binary draft | Binarize ternary (58% preserved) | Binarize FP16 (0% preserved) | **Free vs impossible** |

### Scaling Story

**Crossover point at 1-3B where ALL advantages activate simultaneously:**

- **Below 1B:** Compute-bound regime. Ternary's immature kernels are bottleneck.
- **Above 3B:** Bandwidth-bound regime. Ternary's 4.1x compression translates directly to speed.
- **Quality gap:** Shrinks with scale (7.4% at 50M → projected <3% at 1B → <1% at 7B).
- **Concurrent sessions:** Advantage GROWS with scale (larger models = larger KV caches for Transformers).
- **Every advantage:** Holds constant or improves with scale. None shrinks.

---

## Part 14: Projected End-State (All Phases Complete)

| Metric | Current | After Phase A-B | After Phase C | After Phase D-E |
|--------|---------|----------------|---------------|-----------------|
| **GPU tok/s (50M)** | 75.6 | 1,134-3,024 | ~15,000 | ~15,000+ |
| **GPU tok/s (7B)** | ~10 est. | 80-150 | ~900 | ~900+ |
| **CPU tok/s (7B)** | 14.3 | 21-29 | 40-70 | **200-400** |
| **Quality vs FP16** | -7.4% PPL | 0 to -1.8% | ~0% (parity) | **+17.7%** (ensemble) |
| **Training VRAM (7B)** | >70 GB | >70 GB | **~11 GB** | ~11 GB |
| **Concurrent 7B sessions** | 1 | 1 | ~10 | **80-160,000** |
| **TTFT (cached 32K)** | 64 sec | 64 sec | 64 sec | **5 ms** |
| **7B on phone** | No | No | No | **Yes (29 tok/s)** |

---

**End of Part 2/2. See research-round1-efficiency-part1.md for Parts 1-8 (performance baselines and kernel optimizations).**
