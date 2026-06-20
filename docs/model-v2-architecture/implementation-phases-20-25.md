# Implementation Phases 20-25: Research Findings → Code

**Source:** 210 research discoveries from Rounds 1-4
**Date started:** 2026-02-25
**Testing protocol:** Synthetic data (6M) → WikiText-103 (6M) → Scale to 50M if validated

---

## Phase 20: Quality Quick Wins (est. 1-2 days)

**Goal:** Close 2-5pp of the 7.4% PPL gap with minimal code changes.
**Test scale:** 6M model, 1000 steps, synthetic data first.

| ID | Technique | Source | Expected | Effort |
|----|-----------|--------|----------|--------|
| 20.A | Anti-windup shadow weights | SQ-03 | -1 to -1.8pp | 1 line/weight/step |
| 20.B | Boundary-aware STE gradient | MQ-02 | -1 to -1.8pp | Modify STE function |
| 20.C | Defect initialization (5-10% wrong trits) | SQ-06 | -0.5 to -1pp | 1 line in init |
| 20.D | Marginal returns training (top-K layers) | SQ-08 | -0.8 to -1.3pp | Gradient mask |

**Protocol:**
1. Baseline: Run 6M model 1000 steps synthetic → record loss curve
2. Test each technique INDIVIDUALLY (A/B test vs baseline)
3. Test best combo (stack winners)
4. If validated: repeat on WikiText-103 at 6M
5. Document: loss curves, final loss, convergence speed

**Success criteria:** Any single technique shows >0.5pp improvement consistently.

---

## Phase 21: Quality Medium Techniques (est. 1 week)

**Goal:** Close remaining gap to <2% with architectural additions.
**Prerequisite:** Phase 20 validated techniques applied.

| ID | Technique | Source | Expected | Effort |
|----|-----------|--------|----------|--------|
| 21.A | Low-rank FP16 correction per layer | MQ-01 | -3 to -5.5pp | rank-4 FP16 matrix/layer |
| 21.B | Dendritic gating (element-wise pre-gate) | SQ-02 | -0.9 to -1.5pp | d FP16 params/layer |
| 21.C | Sigma-delta noise shaping in quantization | SQ-01 | -0.7 to -1.1pp | Sequential quant in group |
| 21.D | Bias-corrected STE (variance reduction) | TQ-02 | -2 to -3pp | Running EMA of bias |

**Protocol:** Same A/B testing at 6M synthetic → 6M WikiText.

**Status after Phase 21:** 21.A (SLIGHTLY POSITIVE), 21.B (COSMETIC), 21.C (NEGATIVE), 21.D (NEUTRAL — Finding 92). Meta-conclusion: all interventions below expected magnitude. The gap is fundamental.

---

## Phase 22: Representation Capacity (Quality Round 4) (est. 1 week)

**Goal:** Test if the gap is addressable by changing representation capacity.
**Meta-conclusion:** NEGATIVE — gap is fundamental, not representation-improvable.

| ID | Technique | Source | Expected | Effort |
|----|-----------|--------|----------|--------|
| 22.A | Group size calibration control | Research control | Variable | 1 day |
| 22.B | Block-diagonal ternary DAC | MQ-08 | -1 to -1.5pp | Mixing matrices |

**Protocol:** Same A/B testing at 6M WikiText.

---

## Phase 23: Inference Kernel — GPU (est. 2-3 weeks)

**Goal:** 10-20x GPU inference speedup (75.6 → 750-1500 tok/s at 7B)
**Prerequisite:** Quality phases validated (model architecture stable).

### Steps 1-3 COMPLETE (6M model validation)

**Step 1: Triton GEMV Integration — COMPLETE**
- Replaced on-the-fly dequantization with 2-bit packed Triton GEMV kernel
- VRAM: 75MB → 27MB (-64%) — 2-bit packing vs FP32 shadow weights
- Speed at 6M: 170 tok/s vs 217 tok/s baseline (0.78x) — Triton kernel has higher per-call overhead than cuBLAS at small scale
- Correctness: max abs diff 0.0028 over 10 tokens
- Key insight: At 6M scale, kernel launch overhead dominates (93%), not matmul speed. Triton GEMV benefit grows with model size where bandwidth dominates.
- Bonus: Fixed pre-existing inference bugs (ssm_ssd_step, ssm_mamba1_step, rwkv7_step were feeding un-normed input to in_proj)

**Step 2: In-Place State Mutation — COMPLETE**
- Converted all inference state updates to in-place operations
- h update: `h.mul_(dA).add_(dB_x)` instead of `h_new = dA * h + dB_x`
- conv_state: in-place shift with clone() for overlap safety
- Memory drift: 0.00 MB over 200 tokens (previously grew with each allocation)
- Speed: neutral (~160 tok/s Triton, ~202 tok/s baseline)
- Eliminates ~24,000 CUDA tensor allocations per 1000 tokens at 24-layer model
- Enables CUDA Graph capture by ensuring all tensor addresses are static

**Step 3: CUDA Graph Capture — COMPLETE**
- Wrapped entire model_step in torch.cuda.CUDAGraph for zero-overhead replay
- Challenges solved:
  - SWA RoPE: pre-computed outside graph, passed as static buffers (updated via fill_() before replay)
  - SWA cache write: scatter_() with GPU tensor index instead of Python int fancy indexing
  - Position tracking: CPU-side counter updated outside graph boundary
  - RoPE position overflow: modulo wrapping against cache size
- Result: **876.8 tok/s median (6.11x vs Triton-only, 4.03x vs original baseline)**
- Graph capture time: 5.7ms (amortized over generation)
- Confirms mathematician's prediction: 93% of 6M inference time was kernel launch overhead
- At larger scales (7B), CUDA Graphs give smaller speedup (1.2-2x) since bandwidth dominates

**Speedup progression:**
- Baseline: 217 tok/s
- Step 1 (Triton GEMV): 170 tok/s (0.78x — overhead dominates at small scale)
- Step 2 (In-Place): 160 tok/s (neutral — measurement noise)
- Step 3 (CUDA Graph): **876.8 tok/s (4.03x total speedup)**

#### Step 4a: Kernel E — Fused SSM Recurrence (COMPLETE — NEUTRAL)
- Triton kernel replacing 9-14 PyTorch ops → 1 kernel
- Correctness: PASS (h_diff < 1e-4)
- Speedup: 1.02x single layer, 0.92x full model (within noise)
- **Root cause of neutral result:** SSM recurrence is ~2% of layer compute. Inside CUDA Graph, inter-kernel overhead already ~1-2us. Individual fusion has diminishing returns.
- **Strategic pivot:** Remaining Kernels A-H (individual fusions) would have same issue. Real gains require mega-kernel (fuse everything per layer) or popcount GEMV (fundamentally faster matmul).
- Finding 98 in research-round4-inference.md

#### Step 4b: Hierarchical LM Head — NEGATIVE (2026-02-25)

**Goal:** Replace full vocabulary projection (15.6 MB/token) with two-pass hierarchical projection (~1.1 MB/token) for ~14x bandwidth reduction.

**Implementation:** HierarchicalLMHead class with K-means++ clustering (128 clusters, top-8 fine pass), padded storage for CUDA Graph compatibility, pre-allocated buffers.

**Result:** NEGATIVE. At d_model=256, full matmul completes in 19 us — no bandwidth bottleneck to exploit. Hierarchical path is 18x SLOWER (342 us) due to multi-kernel overhead. Top-1 recall only 21.4% (K-means on random embeddings). Would be viable at d_model≥1024 / vocab≥100K.

**Finding:** 99 | **Commit:** pending

#### Step 4c: Self-Speculative Decoding Validation — PARTIAL (2026-02-25)

**Goal:** Validate whether layer-skip draft models agree with full model predictions often enough for speculative decoding.

**Implementation:** Added `clone_state()` and `max_layers` to `model_step()`. Benchmark compares draft vs full predictions over 200 tokens at 4 draft depths.

**Result:** PARTIAL. Speed scaling validated (linear with depth, 2L=3.82x over 8L). Agreement rate 100% but meaningless on untrained model — needs trained model to validate. Infrastructure ready for real testing at scale.

**Finding:** 100 | **Commit:** pending

#### Step 5: INT8 Activation Pipeline — DEFERRED (2026-02-25)

**Goal:** Quantize activations to INT8 for bandwidth reduction and INT8-only arithmetic.

**Result:** DEFERRED. Micro-benchmark showed naive INT8 quantization costs ~65 us (fixed overhead, 5-6 kernel launches) vs ~15 us for the matmul itself at d_model=256. 452% overhead makes it non-viable at 6M. Deferred to Phase 24 (200M+ scale) where matmul cost amortizes quantization overhead and INT8 Tensor Cores provide real speedup.

**Finding:** 101 | **Commit:** pending

#### Step 6: CPU Inference Baseline — POSITIVE (2026-02-25)

**Goal:** Validate CPU inference viability without custom kernels.

**Result:** POSITIVE. **105.6 tok/s** on Intel Core Ultra 5 225F with PyTorch MKL. Exceeds 50-100 tok/s target. No custom kernels needed — pure MKL with ternary weights as FP32. Theoretical ceiling ~11,500 tok/s (1% utilized). Custom AVX-512 kernel deferred to Phase 24.

**Finding:** 102 | **Commit:** pending

### Steps 4d-7 & Beyond PENDING

| ID | Technique | Source | Expected | Effort |
|----|-----------|--------|----------|--------|
| 23.4b | Mega-kernel SSM + projections | R1.2 | 3-5x additional | Triton kernel |
| 23.6 | Full inference pipeline optimization | Combined | Target: 10-20x total at 7B | Integration |
| 24.A | INT8 activation pipeline | Finding 101 | 2-3x (at 200M+ only) | Deferred |
| 24.B | AVX-512 ternary GEMV (CPU) | Finding 102 | 10-50x (1000-5000 tok/s) | Phase 24 |

**Next step:** Step 4b Mega-kernel (fuse ENTIRE SSM layer including in_proj, conv, gate, out_proj into single Triton kernel to eliminate intermediate tensors and memory round-trips).

---

## Phase 23: Inference Kernel — CPU (est. 2-3 weeks)

**Goal:** 5-10x CPU inference speedup (14.3 → 70-140 tok/s at 7B)

| ID | Technique | Source | Expected | Effort |
|----|-----------|--------|----------|--------|
| 23.A | AVX-512 VNNI ternary GEMM | IB-01 | 3-4x | C/intrinsics kernel |
| 23.B | LUT state transition precompute | MS-02 | 4-6x | LUT generation + lookup |
| 23.C | Prefetch + loop unrolling SSM | IB-08 | 1.7-2x | Compiler intrinsics |
| 23.D | Per-token early exit | MS-06 | 1.6-2.1x | Confidence threshold |

---

## Phase 24: Inference Memory (est. 1-2 weeks)

**Goal:** 3.4x weight compression + 10x state compression

| ID | Technique | Source | Expected | Effort |
|----|-----------|--------|----------|--------|
| 24.A | Factored codebook ternary compression | MPI-03 | 3.4 → 1.0 GB | Codebook design |
| 24.B | Rate-distortion state pruning | MPI-01 | 8 → 2.7 MB/session | Pruning analysis |
| 24.C | Dimension-adaptive state quantization | MPI-07 | 3.3x additional | INT4/INT8 per dim |

---

## Phase 25: System-Level Serving (est. 2-3 weeks)

**Goal:** Production-ready serving with max throughput

| ID | Technique | Source | Expected | Effort |
|----|-----------|--------|----------|--------|
| 25.A | Continuous batching | IA-04 | 1.5-2x throughput | Scheduler |
| 25.B | Memory-based batch sizing | IA-07 | 2-3x throughput | Dynamic batch |
| 25.C | Prefix state caching (CDN-style) | SI-04 | 3.5-7x prompts | Cache system |
| 25.D | JIT kernel specialization | SI-07 | 1.8-2.9x | Profiler + dispatch |

---

## Testing Protocol (ALL PHASES)

### Step 1: Synthetic Proof (6M, synthetic data)
- 1000 steps, batch=32, seq_len=128
- Measure: loss curve, final loss, convergence speed
- Compare: technique ON vs OFF (A/B)

### Step 2: Real Data Validation (6M, WikiText-103)
- Same config but WikiText-103
- Measure: val PPL, loss curve
- Compare: vs FP16 baseline (gap measurement)

### Step 3: Scale Validation (50M, WikiText-103)
- Only for techniques that passed Steps 1-2
- 1000 steps on Modal L4 or local GPU
- Measure: val PPL, throughput, VRAM

### Documentation Per Experiment
- Finding number (continuing from Finding 84+)
- Exact config (model size, steps, LR, technique params)
- Loss curves (saved to results/)
- A/B comparison table
- Conclusion: POSITIVE / NEUTRAL / NEGATIVE

---

## Progress Tracker

| Phase | Status | Findings | Key Result |
|-------|--------|----------|------------|
| 20.A Anti-windup | NEGATIVE | Finding 85 | K=0.01 neutral (-0.005%), K=0.05 worse (+0.16%), K=0.10 worse (+0.24%). Shadow weight drift is useful for exploration. |
| 20.B Boundary STE | NEUTRAL | Finding 86 | Code activates correctly (28 layers verified), but modulation ≈ 0.93 because 88% of weights already near boundary. Shadow weight drift premise INCORRECT at 6M/1000 steps. |
| 20.C Defect init | NEUTRAL | Finding 87 | Defect rates 0-20% all within ±0.001 of baseline. STE compensates injected defects but no gradient stress benefit. Misalignment never self-heals (stabilizes ~15.4%). |
| 20.D Marginal returns | NEGATIVE | Finding 88 | 92.2% of improvement in first 1000 steps. 5000 steps gains only 0.05% extra. Gap NOT from insufficient training duration. |
| Phase 20 Summary | COMPLETE | — | All 4 techniques failed. Gap is fundamental representation issue, not STE/training dynamics. |
| 21.A Low-rank FP16 | SLIGHTLY POSITIVE | Finding 89 | Low-rank FP16 correction verified active. Synthetic: negligible (<0.03%). WikiText-103: -0.19% (rank=2 best). Works but far below expected -3 to -5.5pp. May scale better at 50M. |
| 21.B Dendritic gating | COSMETIC/SLIGHTLY POSITIVE | Finding 90 | Pre-gate: COSMETIC — deep verify proved only 0.001% ternary values change (41/3.7M). Post-gate: +0.83% on WikiText via output scaling (not ternary pattern change). Gate insufficient to cross quantization threshold. |
| 21.C Sigma-delta noise shaping | NEGATIVE | Finding 91 | Sigma-delta successfully modifies ~26% ternary values (verified by deep verify) but WORSENS quality. WikiText loss +0.93-1.07%, reconstruction error +7.8%, throughput -10%. Error feedback loop pushes wrong direction. Fundamental mismatch: ternary has too few levels for error feedback to find better representations. |
| 21.D Bias-corrected STE | NEUTRAL | Finding 92 | BSTE activates correctly (correction 1% → 31.5% over 1000 steps), but per-bin gradient analysis shows STE already unbiased. Throughput cost -7.1%, quality +0.02% WORSE. Gradient bias is not the cause of the quality gap. |
| Phase 21 Summary | COMPLETE | — | All 4 techniques tested. Low magnitude improvements (< 0.2% WikiText max). The gap is fundamental to ternary representation. |
| 22.A Group size calibration | COMPLETE | Finding 93 | CONTROL: gs_128 baseline (5.4969), gs_64 (-0.3%), gs_32 (-0.35%), gs_16 (-0.35%). Floor at gs_32 with negligible gain. Scale granularity ≤0.35% of gap. Confirms gap is ANGULAR not RADIAL. |
| 22.B Block-diagonal DAC | COMPLETE | Finding 94 | block_4: +0.40% WORSE, -25% throughput. block_8: +0.004% neutral, -17% throughput. Mixing matrices learn but provide no loss benefit. Ternary weights already encode sufficient information. |
| Phase 22 Summary | COMPLETE | — | Quality research concluded. Gap confirmed fundamental. No post-hoc fixes work. Next: scaling validation at 200M (predicted 1.0-1.5% gap), then inference engine development. |
| 23.Step 1-3 | COMPLETE | Finding 95-97 | GPU inference: 217 → 877 tok/s (4.03x). Triton GEMV (-64% VRAM), in-place mutation (0 drift), CUDA Graph (6.11x over Triton). 6M validated. |
| 23.Step 4a | COMPLETE | Finding 98 | Kernel E fused SSM recurrence: 9-14 ops → 1 kernel. Impact NEUTRAL (1.02x single layer). Confirms: inside CUDA Graphs, individual fusion has diminishing returns. Real gains require mega-kernel or popcount GEMV. |
| 23.Step 4b | COMPLETE | Finding 99 | Hierarchical LM Head: K-means clustering (128 clusters, top-8 fine) for 14x bandwidth reduction. NEGATIVE — at d_model=256, full matmul (19us) is faster than hierarchical (342us). 6+ kernel launches exceed bandwidth savings. Would work at d_model≥1024. |
| 23.Step 4c-6 | PENDING | — | Mega-kernel (3-5x additional), INT8 pipeline (2-3x), full integration (target 10-20x total at 7B). |
| 24.A-C | Independent | — | Memory optimization (can run parallel) |
| 25.A-D | BLOCKED by 23+ | — | System-level serving |
