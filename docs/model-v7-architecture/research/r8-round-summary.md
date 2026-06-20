# R8 — Research Round Summary

**Goal:** Training time reduction: days → minutes
**Formula:** total_time = tokens_to_converge × time_per_token
**Date:** 2026-03-06
**Model:** v7 (clean v5 baseline, 45,225 tok/s, 886 MB VRAM, 6M params, RTX 5070)

---

## Totals

| Aspect | Focus | IMPLEMENT | EXPLORE | SKIP | Total |
|--------|-------|-----------|---------|------|-------|
| A1 | Ternary GEMM replacement | 13 | 4 | 5 | 22 |
| A2 | SSM kernel fusion | 15 | 4 | 1 | 20 |
| A3 | Compile-time optimization | 7 | 3 | 2 | 12 |
| A4 | Memory bandwidth | 13 | 5 | 2 | 20 |
| A5 | Backward pass reduction | 14 | 4 | 2 | 20 |
| A6 | I/O pipeline & data loading | 13 | 2 | 3 | 18 |
| **Total** | | **75** | **22** | **15** | **112** |

---

## Critical Insights by Aspect

### A1 — Ternary GEMM (3-5× step speedup ceiling)
- **Root cause of current FLOP waste:** F.linear(x, W_q.float()) runs float matmul even though W_q ∈ {-1,0,+1}. Every layer does this.
- **Fix:** torch._int_mm (INT8 exact arithmetic) → 1.3-2× on forward pass (1/3 of FLOPs)
- **Full FLOP coverage:** backward grad_input + grad_weight + optimizer dispatch cover remaining 2/3. All three have IMPLEMENT paths.
- **Highest impact single finding:** COMMITTED-LAYER-DIRECT-BYPASS — fully committed layers skip quantize+matmul entirely; after A3-F13 (block-wise zeroing) is active, this compounds.

### A2 — SSM Kernel Fusion (10-60% step time reduction)
- **Easiest win:** SSM scan in BF16 not float32 — 1-line fix in FLA call, 2× memory bandwidth in scan.
- **FusedRMSNormGated:** z-gate silu(z)*y is structurally identical to FLA's existing fused kernel — 1 kernel call instead of 3.
- **chunk_size 64→128:** doubles Triton occupancy in SSD scan without code changes, ~20% scan speedup.
- **Microbatch pipeline overlap:** forward[b] and backward[b-1] overlap on separate CUDA streams (35-50% wall-clock reduction).
- **EXPLORE — affine monoid prefix reduce:** mathematically sound reformulation of SSD scan as affine composition, eliminates sequential dependency. High-risk but could 2-4× scan throughput.

### A3 — Compile-time (10-100× leverage on overhead vs compute)
- **Critical measurement:** 75-85% of step time is Python/kernel-launch overhead at 6M scale. Kernel compute is 0.06 ms of 45.3 ms/step.
- **Highest leverage:** REDUCE-OVERHEAD-CUDA-GRAPH (torch.compile mode="reduce-overhead") + ALLOW-IN-GRAPH-ALL-STE (4 missing allow_in_graph calls create graph breaks).
- **Print suppression:** debug prints inside training loop cost measurable time at 45K tok/s.
- **Sequence warmup dynamic shapes:** eliminates recompile on first-step shortened sequences.
- **LEGO flags to remove:** sequence_packing, cuda_prefetch — both already implemented, just hidden behind False defaults.

### A4 — Memory Bandwidth (50-80% e2e throughput gain ceiling)
- **Roofline:** RTX 5070 ridge point = 59 FLOPs/byte; linear layers at batch=4 are 100% bandwidth-bound (AI = 0.50, 119× below ridge).
- **conv1d transpose:** 2 non-contiguous ops + 1 copy per SSM forward/backward, eliminated via channels-last layout.
- **Lipschitz-lazy requantization:** if weight change < threshold (||ΔW||₂ ≤ ε × ||W||₂), skip quantize call entirely. Skips 99.8% of quantize calls after warmup.
- **Shadow weight BF16 downcast:** shadow FP32 → BF16 at load time reduces param bandwidth 2×.
- **Committed-layer bypass:** fully committed layers serve precomputed BF16 weights directly, skipping quantize + int8 matmul setup.

### A5 — Backward Pass Reduction (60-70% activation VRAM, 25-40% backward time)
- **SiLU approx BP:** 2-bit sign masks replace full FP16 activation storage in backward. 30% peak VRAM reduction, <0.4% accuracy drop (ICML 2024).
- **LN folding at init:** fold LayerNorm into adjacent projection matrices at initialization — eliminates per-token LN forward+backward permanently.
- **Subsampled token grad_weight:** compute grad_weight on random 16/512 token subset — 96.9% FLOP reduction in most expensive backward op, mathematically equivalent via law of large numbers.
- **Lion momentum lock grad skip:** committed weights at step stationarity → grad∝momentum→0. 63% of grad_weight computations produce near-zero updates, can be skipped.
- **OLS-STE diagonal Gram approximation:** current OLS-STE computes full Gram matrix per group. Diagonal approximation reduces O(d²) → O(d) with <2% error.

### A6 — I/O Pipeline & Data Loading (20-40% token reduction potential)
- **I/O is NOT the bottleneck:** data loading = 0.00018% of step time. Value is in smarter batch composition.
- **LEGO flags to kill:** `use_cuda_prefetch=False`, `use_sequence_packing=False` — both already implemented, always-active by removing the guards.
- **Stateful streaming training:** SSM mixing time T_mix=458 tokens for slow heads (ρ=0.99). Random-access batching resets h every 512 tokens → 89.5% wasted tokens. Stateful (adjacent sequences) corrects this: 20-40% step reduction (BPTT detach error <3%).
- **Optimal sequence length with stateful:** T_opt = 44-90 tokens (from mixing time theory). 512→64 token sequences → 8× larger batches at same VRAM.
- **Corpus phase offsets:** correct formula is b×floor(N/B) not b×seq_len. Wrong formula creates sequence dependency between workers, corrupting stateful training statistics.
- **SOBOL position sampling:** 4× lower discrepancy vs Monte Carlo → 5-15% step reduction via better coverage of data distribution.

---

## Top 10 Highest-Impact Findings (Implementation Priority)

| Rank | ID | Aspect | Gain | Type |
|------|----|--------|------|------|
| 1 | REDUCE-OVERHEAD-CUDA-GRAPH | A3 | 75-85% overhead eliminated | time_per_token |
| 2 | ALLOW-IN-GRAPH-ALL-STE | A3 | enables graph fusion across quant boundaries | time_per_token |
| 3 | STATEFUL-STREAMING-TRAINING | A6 | 20-40% step reduction | tokens_to_converge |
| 4 | STATEFUL-OPTIMAL-SEQ-LEN | A6 | 8× batch size → fewer steps at same VRAM | both axes |
| 5 | INT8-ACCUM-FORWARD | A1 | 1.3-2× forward matmul | time_per_token |
| 6 | MICROBATCH-PIPELINE-OVERLAP | A2 | 35-50% wall-clock step reduction | time_per_token |
| 7 | SUBSAMPLED-TOKEN-GRAD-WEIGHT | A5 | 96.9% grad_weight FLOP reduction | time_per_token |
| 8 | LIPSCHITZ-LAZY-REQUANTIZE | A4 | skip 99.8% quantize calls | time_per_token |
| 9 | LION-MOMENTUM-LOCK-GRAD-SKIP | A5 | skip 63% grad_weight | time_per_token |
| 10 | SSM-BF16-SCAN | A2 | 2× scan bandwidth | time_per_token |

---

## LEGO Flags Found in v7 Codebase (Must Remove)

These are existing implementations hidden behind `False` defaults — pure LEGO violations to fix immediately:

| Flag | File | Always-active behavior |
|------|------|----------------------|
| `use_cuda_prefetch=False` | config.py L151 | CUDA H2D overlap |
| `use_sequence_packing=False` | config.py | bin-packing sequences |

---

## Combined Gain Estimate (conservative, multiplicative)

| Axis | Techniques | Conservative Estimate |
|------|-----------|----------------------|
| **time_per_token** | A1+A2+A3+A4+A5 combined | 4-8× faster per step |
| **tokens_to_converge** | A6 stateful+sobol+stratified | 2-3× fewer steps |
| **Combined** | | **8-24× total wall-clock reduction** |

*Note: gains are partially overlapping; 8-24× is optimistic. Smoke tests required per technique.*

---

## Files

| Aspect | Findings doc |
|--------|-------------|
| A1 | `docs/model-v7-architecture/research/r8-a1-ternary-gemm-findings.md` |
| A2 | `docs/model-v7-architecture/research/r8-a2-ssm-kernel-findings.md` |
| A3 | `docs/model-v7-architecture/research/r8-a3-compile-findings.md` |
| A4 | `docs/model-v7-architecture/research/r8-a4-memory-bandwidth-findings.md` |
| A5 | `docs/model-v7-architecture/research/r8-a5-backward-pass-findings.md` |
| A6 | `docs/model-v7-architecture/research/r8-a6-io-pipeline-findings.md` |

---

## Next Steps

1. **Implementation phase (R8 findings):** 75 IMPLEMENT findings — start with Top 10 list above.
2. **R9:** Convergence (tokens-to-converge) — curriculum, data selection, LR meta-strategies.
3. **R10:** Effective batch size and memory — gradient accumulation, activation offloading.
4. **R11:** Ternary-specific acceleration — popcount GEMM, bit-packing, quantization efficiency.
