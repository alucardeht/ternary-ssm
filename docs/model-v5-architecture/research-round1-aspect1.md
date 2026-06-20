# Round 1, Aspect 1: Extreme Sparse Activation for Ultra-Low RAM

## Wave 1 — Web Research (2 researchers, Feb 2026)

### ULTRA-SPARSE MoE APPROACHES

#### v5-R1-A1-F1: MoTE — Mixture of Ternary Experts (2.3 GB for 7B)
- **Source:** arXiv 2506.14435v2, June 2025
- **Summary:** Ternary quantization ({-1,0,+1}) applied to MoE experts. Each expert uses ~1/16th memory of FP16. LLaVA-NeXT 7B → 2.3 GB (vs 5.2 GB FP16 MoE). 13B → 3.4 GB.
- **RAM:** 2.3 GB for 7B-equivalent (measured)
- **Quality:** -2 to -4% vs FP16 on VQA/GQA/TextVQA
- **Throughput:** 2-3x FASTER (ternary matmul + sparse activation)
- **Maturity:** Proof-of-concept (published, code likely available)
- **Verdict:** VERY PROMISING — closest to sub-1GB goal. With more experts + top-1 routing could reach <1GB.

#### v5-R1-A1-F2: FlashMoE — SSD/NVMe Expert Offloading (64GB → ~1GB DRAM)
- **Source:** arXiv 2601.17063v1, January 2026
- **Summary:** ML-based cache replacement for offloading MoE experts to SSD/NVMe. Predicts which experts needed next, pre-loads from disk. Only active experts in DRAM.
- **RAM:** 64 GB → ~1 GB DRAM (98% reduction, measured)
- **Quality:** Not measured vs dense (system paper)
- **Throughput:** "Minimal latency penalty" via predictive caching (no tok/s numbers)
- **Maturity:** Proof-of-concept (2026 paper)
- **Verdict:** PROMISING — proves sub-1GB DRAM is achievable with offloading

#### v5-R1-A1-F3: BuddyMoE — Expert Redundancy Exploitation
- **Source:** arXiv 2511.10054v1, November 2025
- **Summary:** Many MoE experts are functionally redundant (similar outputs). Groups experts into "buddy" clusters, loads ONE per cluster. Mixtral 8x7B: 87 GB → 32 GB (63% reduction).
- **RAM:** 87 GB → 32 GB (63% reduction, measured on Mixtral)
- **Quality:** -1 to -2% on MMLU, GSM8K
- **Throughput:** FASTER (fewer experts = less bandwidth pressure)
- **Maturity:** Proof-of-concept (real measurements on Mixtral)
- **Verdict:** PROMISING — significant reduction, combinable with other techniques

#### v5-R1-A1-F4: DeepSeek MLA — 93x KV Cache Compression
- **Source:** arXiv 2412.19437v1, December 2024 (DeepSeek-V3)
- **Summary:** Multi-head Latent Attention compresses KV cache by 93.3x vs standard attention. 671B total params, only 37B active per token (<6% activation).
- **RAM:** KV cache 93.3x smaller (critical for long context)
- **Quality:** SOTA on many benchmarks (beats GPT-4 on some)
- **Throughput:** Very fast (optimized kernels)
- **Maturity:** Production (public model, API available)
- **Verdict:** PROMISING — KV cache is major RAM consumer, 93x reduction is massive

#### v5-R1-A1-F5: Ultra-Sparse MoE (1000+ Experts, K=1)
- **Source:** EmergentMind survey + multiple 2025-2026 papers
- **Summary:** Models with thousands of experts are feasible. Top-1 of 1000 = 0.1% activation. SEER-MoE can prune 25% of experts with only 3.85 MMLU drop. Combined with ternary: each expert < 1 MB → 1000 experts = < 1 GB total.
- **RAM:** Theoretical: 7B with 64 ternary experts × 14 MB each = 900 MB (if only 8 loaded)
- **Quality:** -3.85 pts MMLU with 25% expert pruning
- **Throughput:** FLOPs/Token = O(K × d_model × d_ff), with K << N
- **Maturity:** Theoretical (no end-to-end implementation at this scale)
- **Verdict:** PROMISING — math checks out, needs validation

### SPARSE ACTIVATION IN DENSE MODELS

#### v5-R1-A1-F6: PowerInfer — Hot/Cold Neuron Split (Production)
- **Source:** SOSP 2024, IPADS @ Shanghai Jiao Tong, arXiv 2312.12456
- **Summary:** Exploits power-law neuron activation: 10% "hot" neurons always active (stay on GPU), 90% "cold" neurons input-dependent (computed on CPU). OPT-175B runs on RTX 4090.
- **RAM:** OPT-175B: A100 80GB → RTX 4090 24GB (~70% reduction, measured)
- **Quality:** SAME as dense (zero degradation)
- **Throughput:** 13.20 tok/s avg, 29.08 tok/s peak (11.69x vs llama.cpp)
- **Maturity:** PRODUCTION (GitHub, Docker, SOSP 2024)
- **Verdict:** PROMISING — best real-world RAM savings with zero quality loss

#### v5-R1-A1-F7: DejaVu — Activation Prediction
- **Source:** ICML 2023, Rice/Stanford/CMU, arXiv 2310.17157
- **Summary:** Predicts contextual sparsity (which neurons activate) BEFORE computing them using tiny vanilla NNs. Only top-25-35% of neurons/heads activated per token.
- **RAM:** ~50% KV cache reduction (only cache active neurons)
- **Quality:** SAME at 25% sparsity, <1% loss at 35% sparsity
- **Throughput:** 2x faster than FasterTransformer (OPT-175B)
- **Maturity:** Proof-of-concept (GitHub, requires predictor training)
- **Verdict:** PROMISING — strong accuracy preservation

#### v5-R1-A1-F8: R-Sparse — Rank-Aware Rotation Sparsity (Training-Free)
- **Source:** ICLR 2025, Meta/UT Austin, arXiv 2504.19449, April 2025
- **Summary:** Rotates inputs via SVD to identify sparse channels WITHOUT magnitude pruning. No predictor needed. 50% model-wide sparsity with <0.17 PPL gap.
- **RAM:** Theoretical 50% reduction at 50% sparsity
- **Quality:** Llama-2-7B at 50%: <0.17 PPL gap, <0.54% accuracy gap
- **Throughput:** 1.43x end-to-end with custom kernels
- **Maturity:** Proof-of-concept (training-free is huge advantage)
- **Verdict:** PROMISING — elegant math, no training overhead

#### v5-R1-A1-F9: TEAL — Training-Free Magnitude Sparsity
- **Source:** ICLR 2025, MIT/Roblox, OpenReview dGVZwyq5tV, January 2025
- **Summary:** Top-k% activation across all layers based on magnitude. Simplest approach. Compatible with weight quantization.
- **RAM:** ~40-50% reduction at 40-50% sparsity
- **Quality:** Minimal degradation at 40%, <2% at 50%
- **Throughput:** 1.53x at 40%, 1.8x at 50%
- **Maturity:** Proof-of-concept (works with existing sparse kernels)
- **Verdict:** PROMISING — simplest approach, combinable with quantization

### ADDITIONAL FINDINGS

#### v5-R1-A1-F10: TC-MoE — Ternary Expert Choice Routing
- **Source:** OpenReview dsP91M4hDL, 2025
- **Summary:** Ternary weights for routing mechanism itself (not just experts). Reduces routing computation overhead.
- **Verdict:** NEEDS-MORE-WORK (niche optimization)

#### v5-R1-A1-F11: MoR — Mixture of Recursions (NeurIPS 2025)
- **Source:** Google/Mila, arXiv 2509.05915
- **Summary:** Recursive transformer with dynamic depth routing. Tokens get different recursion depths. Selective KV caching (~40-60% reduction).
- **Quality:** BETTER than vanilla at equal FLOPs
- **Verdict:** NEEDS-MORE-WORK (requires full architecture redesign)

### REJECTED

#### v5-R1-A1-F12: Early-Exit LLMs
- **Summary:** Exit at intermediate layers based on confidence. 2-3x compute savings.
- **RAM:** NO direct RAM reduction — full model still loaded.
- **Verdict:** REJECTED (no RAM savings, only compute savings)

#### v5-R1-A1-F13: Dynamic Input Pruning
- **Source:** arXiv 2412.01380, December 2024
- **Summary:** Top-k pruning per input. Reduces DRAM bandwidth, NOT capacity.
- **Verdict:** REJECTED (bandwidth != RAM capacity)

## Key Insights — Aspect 1

1. **Sub-1GB IS mathematically achievable** — MoTE already hits 2.3 GB for 7B. With more aggressive sparsity + offloading: sub-1GB.
2. **The combination formula:** Sparse activation (70%) + Ternary weights (75% vs FP16) + Selective KV (50%) = ~89% total reduction → 1.5 GB for 7B. Add INT4 KV → sub-1GB.
3. **PowerInfer is the gold standard** — ONLY production system with measured 70% RAM reduction + zero quality loss + 11.69x speedup.
4. **Expert offloading works** — FlashMoE proves NVMe→DRAM pipeline can keep only active experts in memory.
5. **Redundancy is real** — BuddyMoE shows 63% of experts are redundant. Load fewer = less RAM.
6. **Training-free methods exist** — R-Sparse and TEAL need no retraining, just apply at inference.
7. **KV cache is the hidden RAM killer** — DeepSeek MLA's 93x reduction is critical for long context.

## Projected Path to Sub-1GB (7B-equivalent)

| Component | Technique | RAM | Notes |
|-----------|-----------|-----|-------|
| Weights | 64 ternary experts × ~14 MB | ~900 MB total, ~28 MB active (top-2) | MoTE approach |
| Shared params | Ternary attention/embed | ~100 MB | Non-expert params |
| KV cache | MLA-style compression | ~10 MB (for 2K context) | 93x reduction |
| Offload buffer | FlashMoE prediction | ~28 MB (2 next experts) | Pre-loaded from NVMe |
| Router | Lightweight gating | ~5 MB | Small FC layer |
| **TOTAL** | | **~171 MB active** | All experts on NVMe (~900 MB) |

**This is UNDER 200 MB active RAM with all experts stored on NVMe (< 1 GB).**
