# Research Round 2, Aspect 2: Training Memory Optimization

## Wave 1 — Web Research (2 researchers, Feb 2026)

### Theme A: Memory-Efficient Training Techniques

#### R2-A2-F1: π-Quant — Irrational Complex Rotation Optimizer Compression
- **Source:** Tian et al., NeurIPS 2025, arXiv:2501.12896
- **Summary:** Maps optimizer state pairs (m, v) into complex space using π-based rotation angles, quantizing the angle instead of raw values. Achieves 3.32-bit average for optimizer states (vs 32-bit FP32).
- **Memory savings:** 41.8% GPU memory reduction for optimizer states. 12 bytes/param → ~5 bytes/param.
- **Quality:** Zero accuracy loss
- **Applicability:** Direct replacement for Adam state storage. For 72M params: 864 MB → 360 MB.
- **Verdict:** PROMISING

#### R2-A2-F2: MoTE — Mixture of Ternary Experts
- **Source:** Wang et al., arXiv:2506.14435, June 2025
- **Summary:** Sparse MoE where routed experts use ternary {-1,0,+1}, shared expert stays FP16. 4.3% accuracy gain vs FP16 MoE at same memory. Enables 4x more experts at same budget.
- **Applicability:** Requires MoE architecture change. Interesting for future scaling.
- **Verdict:** NEEDS-MORE-WORK (architecture change required)

#### R2-A2-F3: Quamba-SE — Soft-Edge Quantization for SSM Activations
- **Source:** arXiv:2601.09451, Jan 2026
- **Summary:** Sub-4-bit activation quantization specifically for State Space Models. Differentiable sigmoid-modulated boundaries (soft-edge) + lookup-table approximation.
- **Memory savings:** 75% activation memory reduction (FP16 → 2-bit activations), <1% perplexity degradation
- **Applicability:** Specifically designed for SSM recurrent states. Soft-edge boundaries better for state evolution stability.
- **Verdict:** PROMISING — SSM-specific, directly applicable

#### R2-A2-F4: Block-wise Ternary Encoding (5-in-1)
- **Source:** EmergentMind synthesis, 2025 papers
- **Summary:** Pack 5 ternary weights per int8 byte (3^5 = 243 patterns fit in byte). Entropy-optimal: log2(243) ≈ 7.92 bits for 5 weights = 1.58 bits/weight.
- **Memory savings:** 80% reduction vs naive int8 storage (1.6 bits/weight vs 8 bits)
- **Quality:** Lossless encoding
- **Verdict:** PROMISING — immediate win for weight storage

### Theme B: Memory Management Systems

#### R2-A2-F5: TERAIO — Lifetime-Aware Tensor Offloading via GPUDirect Storage
- **Source:** Yuan et al., NeurIPS 2025
- **Summary:** Profiles first few training iterations to predict tensor lifetime (active period in GPU memory). Generates optimized offload/prefetch plan. Key insight: 98.3% of allocated memory is inactive at any given iteration, only 1.7% active.
- **Memory savings:** Train models 8-10x larger than GPU VRAM. 1.47x faster than ZeRO-Offload.
- **Applicability:** RTX 5070 PCIe 5.0 (128 GB/s) can hide SSD latency. Solves "7B in <1GB RAM" problem by offloading inactive tensors.
- **Verdict:** PROMISING — keystone for ultra-low RAM architecture

#### R2-A2-F6: INSTANT — Low-Rank Gradient + Activation Compression
- **Source:** Doan et al., ICLR 2026
- **Summary:** Projects gradients and activations into low-rank subspace during backprop. Backward pass runs in compressed representation.
- **Memory savings:** 32x activation memory reduction, 15x FLOPs reduction
- **Applicability:** Synergizes with ternary: 16x (ternary weights) × 32x (INSTANT activations) = 512x combined compression
- **Verdict:** PROMISING — extreme compression when combined with ternary

#### R2-A2-F7: COAT — FP8 Optimizer States
- **Source:** NVlabs/COAT, ICLR 2025
- **Summary:** Replace FP32 optimizer states with FP8. Adam m+v from 8 bytes/param to 1 byte/param. Compatible with FP16 forward/backward.
- **Memory savings:** 12x optimizer memory reduction (12 bytes → 1 byte per param)
- **Quality:** Slight degradation, compensated by per-channel dynamic scaling
- **Applicability:** At 7B: 84 GB → 7 GB optimizer states. Combined with ternary weights (1.75 GB): total model+optimizer = 8.75 GB (fits in 12 GB GPU!)
- **Verdict:** PROMISING — makes 7B training feasible on 12GB GPU

#### R2-A2-F8: FlashOptim — Zero Optimizer State
- **Source:** arXiv:2602.23349, Feb 2026
- **Summary:** Unifies gradient accumulation with momentum, eliminating dedicated optimizer memory. Low-rank decomposition of optimizer states.
- **Memory savings:** Eliminates optimizer memory entirely. Combined with ternary: 0.2 bytes/param total.
- **Applicability:** At 7B: total = 1.4 GB for weights only, zero optimizer. THIS IS THE V5 TARGET.
- **Verdict:** PROMISING — potential keystone, needs full paper access for implementation details

#### R2-A2-F9: D²Prune — Dual Taylor Expansion Pruning
- **Source:** arXiv:2601.09176, Jan 2026
- **Summary:** Structured pruning using second-order Taylor expansions of Hessian + attention distributions. 60% sparsity with <2% quality loss on Llama-3 8B.
- **Applicability:** More valuable for FP16 components (embeddings, LM head). Ternary weights already ultra-compressed.
- **Verdict:** NEEDS-MORE-WORK (diminishing returns on ternary)

## Key Insights — Aspect 2 Wave 1

1. **Optimizer state is the new compression frontier** — Weights are 1.58 bits, but optimizer is 96 bits. π-Quant/COAT/FlashOptim attack this gap.
2. **SSM-specific techniques emerging** — Quamba-SE validates that SSM activation quantization is tractable.
3. **Compression stacking is multiplicative** — Ternary (16x) × INSTANT (32x) × COAT (12x) = 6,144x theoretical.
4. **TERAIO enables the impossible** — 98.3% of GPU memory is inactive → offload → fit 7B in <1GB active memory.
5. **FlashOptim could be the keystone** — Zero optimizer state + ternary weights = 0.2 bytes/param total.

## For Math-Paradigm-Breakers (Wave 2)

1. Can we derive the information-theoretic minimum memory for training a ternary model?
2. Is there a mathematically optimal compression for SSM recurrent states specifically?
3. Can lifetime-aware offloading be formalized as a scheduling/packing optimization?
4. What is the minimum precision needed for gradient propagation in ternary networks?
