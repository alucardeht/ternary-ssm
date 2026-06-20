# Efficiency Techniques — Stacking Optimizations (February 2026)

## Overview

These techniques can be applied on top of ANY architecture (Transformer, Mamba, xLSTM, etc.) to further reduce hardware requirements. They are COMPLEMENTARY — many can be stacked together.

---

## 1. Mixture of Depths (MoD) — Dynamic Layer Skipping

### Mechanism
Not all tokens need all layers. Router network decides per-token which layers to skip.
- Early layers: low skip rate (5-10%)
- Later layers: high skip rate (50-80%)
- Tokens skip via residual connection (no processing)

### Impact
- 40-50% FLOP reduction per forward pass
- Up to 2x faster inference (1.5x typical)
- 44-46% KV cache reduction
- Quality: matches vanilla Transformer at iso-FLOP

### Combines with: MoE (expert routing + layer routing), any architecture
### Status: Research (2024-2025), p-MoD and MoDification implementations exist

---

## 2. Structured Sparsity / Pruning

### Mechanism
Remove weights that contribute minimally. Methods:
- **SparseGPT**: second-order reconstruction, layer-by-layer, no retraining
- **Wanda**: scores by weight_magnitude x input_activation, per-output pruning
- **2:4 Sparsity**: NVIDIA hardware-accelerated (2 non-zero per 4 elements)
- **PruneNet (2025)**: 50% faster than SliceGPT, 84% accuracy without recovery

### Impact
- 50-70% parameter reduction achievable
- 90-95% quality retention with modern methods
- 1.27-2x inference speedup
- 7B -> ~3.5B effective parameters

### Quality Threshold
Can prune 50-60% before significant degradation. Beyond 70%, quality drops sharply.

### Combines with: quantization (prune first, then quantize), all architectures
### Status: Production (SparseGPT, Wanda widely used post-training)

---

## 3. Knowledge Distillation

### Mechanism
Large teacher model trains smaller student to mimic output distributions.
- Progressive: 70B -> 30B -> 13B -> 7B -> 3B (staged)
- HPM-KD (2025): hierarchical multi-teacher with adaptive hyperparameters

### Impact
- Moderate compression (30-50% reduction): 97-100% quality retention
- Extreme compression (20x+): 80-90% retention
- Example: DeepSeek-R1 671B -> 7B distill = 92.8% MATH-500

### Theoretical Limit
Student cannot exceed teacher's information content. Emerging reasoning abilities are hardest to preserve in small models.

### Combines with: quantization, pruning, any architecture
### Status: Production (DeepSeek-R1-Distill, many commercial deployments)

---

## 4. Dynamic Inference / Early Exit (LayerSkip)

### Mechanism
Simple tokens exit after early layers; complex tokens use all layers. Meta's LayerSkip:
- Training: layer dropout (low early, high late) + shared exit projection
- Inference: self-speculative decoding (early layers draft, later verify)
- No separate draft model needed

### Impact
- 1.82-2.16x speedup (task-dependent)
- Reduced KV cache (fewer active layers)
- Quality matches full model on most tasks

### Meta Released Checkpoints
Llama-2 (7B, 13B, 70B), CodeLlama (7B, 34B) — all with LayerSkip

### Combines with: speculative decoding, MoD, any architecture
### Status: Production (PyTorch torchtune Dec 2024, HuggingFace Nov 2024)

---

## 5. Weight Sharing / Parameter Tying

### Mechanism
Share weights across multiple layers. ALBERT: all layers share same parameters + factorized embeddings.

### Impact
- 10-20x fewer parameters (32 layers sharing = 1 layer in VRAM)
- ~8x reduction on embeddings (factorized)
- Quality: comparable or better on NLP benchmarks
- Training: slower convergence (more steps needed)

### Combines with: quantization, any architecture
### Status: Production (ALBERT), but not mainstream in latest LLMs

---

## 6. KV Cache Compression

### Mechanism
Quantize KV cache to lower precision. Not just weights — the runtime cache itself.
- INT8 KV: 2x cache reduction, minimal quality loss
- INT4 KV: 4x cache reduction, slight degradation
- INT2 KV (MiniKV 2025): 8x reduction, moderate degradation
- Hybrid: critical channels INT4, rest INT2

### Impact (7B model)
```
Baseline FP16: weights 14GB + KV 4.5GB = 18.5GB
INT4 all:      weights 3.5GB + KV 1.1GB = 4.6GB (75% savings)
```

### Combines with: weight quantization, any KV-cache architecture (not SSMs/RNNs)
### Status: Production (vLLM, llama.cpp, Ollama all support)

---

## 7. Tensor Decomposition

### Mechanism
Break large weight matrices into products of smaller matrices.
- Tucker decomposition: core tensor + factor matrices
- CP decomposition: sum of rank-1 tensors
- Kronecker products: efficient multiplication
- TensLoRA (2025): tensor-based LoRA, 10-100x fewer parameters than standard LoRA

### Impact
- 1-2 orders of magnitude fewer parameters (theory)
- Speed: faster matmuls from fewer parameters
- Quality: depends on rank selection (sweet spot: 10-30% of original rank)

### Combines with: LoRA/QLoRA, any architecture
### Status: Research (TensLoRA promising for fine-tuning)

---

## 8. Sub-Quadratic Attention

### Flash Attention 3 (2025)
- Still O(n²) but IO-optimized
- 1.5-2x faster than FA2 (FP16)
- 75% H100 utilization (vs 35% FA2)
- FP8: 1.2 PFLOPS, 2.6x smaller error
- 2x longer context on same hardware

### Linear Attention
- FLA framework: RetNet, GLA, Mamba, HGRN, RWKV6 integrated
- True O(n) complexity
- 5-10% quality degradation typical

### Sliding Window Attention
- Each token attends to fixed window (e.g., 4096)
- O(n x window) = O(n)
- Used by Mistral 7B, Gemma2
- 2x speedup for 16K sequences, 50% cache reduction

### Ring Attention
- Distributed attention across devices
- Enables millions of tokens
- Passes needle-in-haystack test

### Combines with: any Transformer-based architecture
### Status: Flash Attention 3 production-ready; linear/sparse are research/specialized

---

## Stacking Strategy

### Maximum VRAM Reduction (Production-Ready Stack)
```
1. Start with distilled model (if available)
2. Apply AWQ INT4 quantization
3. Enable INT4 KV cache
4. Use Flash Attention 3
5. Consider LayerSkip checkpoints

Combined result (7B):
  Weights: 14GB -> 3.5GB (INT4)
  KV Cache: 4.5GB -> 1.1GB (INT4 KV)
  Speed: 2x (LayerSkip)
  Quality: ~85-90% of original
  Total: ~4.6GB
```

### Aggressive Stack (Research)
```
1. Pruning 50% (SparseGPT)
2. INT4 quantization
3. INT4 KV cache
4. MoD (skip layers)
5. Early exit

Combined result (7B):
  Weights: 1.75GB
  KV Cache: 0.6GB
  Total: ~2.35GB (87% reduction)
  Speed: 2x faster
  Quality: ~70-75% of original
```

### NVIDIA Proven Pipeline
```
Pruning + Distillation = 4x model size reduction
+ INT8 quantization = 14x total reduction
```

---

*Last updated: February 2026*
