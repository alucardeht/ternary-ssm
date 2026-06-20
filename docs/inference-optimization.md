# Inference Optimization — Maximum Performance When Using Models (February 2026)

## Three Sub-Areas

Inference optimization has three distinct stages:
1. **Model Optimization** — compress/optimize the model BEFORE serving
2. **Runtime Optimization** — techniques applied DURING token generation
3. **Serving** — how to expose the model via API

---

## 1. Model Optimization (Before Serving)

### Quantization

| Format | Quality vs FP32 | Size (7B) | Speed | Best For |
|--------|-----------------|-----------|-------|----------|
| FP16 | 100% | 14 GB | Baseline | Reference |
| FP4 native (Blackwell) | ~95% | ~3.5 GB | Fastest (tensor core) | RTX 5070 specifically |
| AWQ (4-bit) | 95% | ~4 GB | Fast | Quality-critical work |
| Q4_K_M (GGUF) | 92% | 4-5 GB | Good | Universal, Ollama |
| Q5_K_M (GGUF) | 94% | 5-6 GB | Good | Higher quality if VRAM allows |
| GPTQ (4-bit) | 90% | ~4 GB | Fastest batch | vLLM production serving |
| EXL2 (4.65bpw) | Configurable | ~3.8 GB | Fast | Fine-grained control |
| INT2 | ~70% | ~1.7 GB | Fast | Experimental |

### Post-Training Pruning

Remove weights that contribute minimally to output:

**SparseGPT**: Layer-by-layer, no retraining, second-order reconstruction
**Wanda**: weight_magnitude × input_activation scoring, per-output pruning
**PruneNet (2025)**: 50% faster than SliceGPT, 84% accuracy without recovery

```
7B → 50% pruning → 3.5B effective
  90-95% quality retained
  No retraining needed
  1.3-2x inference speedup
```

Quality threshold: can prune 50-60% before significant degradation. Beyond 70%, quality drops sharply.

### 2:4 Structured Sparsity (NVIDIA Hardware)
- 2 non-zero values per 4 elements
- Hardware-accelerated on Ampere+ (RTX 5070 supports this)
- Theoretical 2x matmul speedup (1.27x practical)

### Stacking: Pruning + Quantization

NVIDIA proven pipeline:
```
7B FP16 (14 GB)
  → Prune 50%: 3.5B effective
  → INT4 quantization: ~1.75 GB
  = 14x total reduction
  = ~85% quality
```

With distillation:
```
13B → Distill to 7B → Prune 50% → INT4
  = 1.75 GB with 13B-origin quality at ~80-85%
```

---

## 2. Runtime Optimization (During Inference)

### Operator Fusion

Reduce VRAM round-trips by combining operations into single kernels:

```
Without fusion: 11 VRAM round-trips per layer × 32 = 352 total
With fusion:     5 VRAM round-trips per layer × 32 = 160 total
= 55% fewer memory operations
= 20-30% faster inference
```

Fused groups:
- [RMSNorm + Q/K/V projection] → 1 kernel
- [Attention] → Flash Attention (1 kernel)
- [O_proj + RMSNorm] → 1 kernel
- [Gate + Up + SiLU] → 1 kernel
- [Down + residual add] → 1 kernel

### Flash Attention 3 (2025)

Still O(n²) but IO-optimized:
- 1.5-2x faster than Flash Attention 2
- 75% GPU utilization (vs 35% FA2)
- FP8 support: 1.2 PFLOPS
- 2x longer context on same hardware
- Production-ready (PyTorch, vLLM)

### KV Cache Compression

| Format | Cache Reduction | Quality Impact |
|--------|----------------|----------------|
| INT8 KV | 2x | Minimal |
| INT4 KV | 4x | Slight |
| MiniKV (INT2 hybrid) | 8x | Moderate |

```
7B Q4 + INT4 KV Cache:
  Weights: 3.5 GB
  KV Cache (32K ctx): 1.1 GB (was 4.5 GB)
  TOTAL: 4.6 GB (75% reduction from FP16 baseline)
```

### Early Exit / LayerSkip (Meta)

Simple tokens exit after early layers, complex tokens use all 32:

```
Token "the": exits at layer 8   → 75% less compute
Token "however": exits layer 16 → 50% less compute
Token "epistemologically": all 32 layers → full compute

Average: 1.8-2.2x speedup, quality maintained
```

Meta released LayerSkip checkpoints: Llama-2 (7B, 13B, 70B), CodeLlama (7B, 34B).
Integrated in PyTorch torchtune and HuggingFace transformers.

### Mixture of Depths (MoD)

Router decides per-token which layers to process:
- 40-50% FLOP reduction
- 44-46% KV cache reduction
- Quality matches at iso-FLOP
- Status: Research (2024-2025)

### Speculative Decoding (NPU as Draft Model)

The killer optimization for heterogeneous hardware:

```
1. NPU runs small model (3B INT8, ~2GB in system RAM)
   Drafts 5 tokens ahead (~5ms, ~5W power)

2. GPU takes 5 drafted tokens
   Verifies ALL in single forward pass (~6ms)

3. Acceptance rate: 60-75% (3B draft → 8B target)
   Result: ~3-4 tokens accepted per GPU forward pass

WITHOUT speculative: 1 token per 6.7ms = 150 tok/s
WITH speculative:    ~3.5 tokens per 6.7ms = ~525 tok/s (theoretical)
Realistic: 350-450 tok/s

NPU and GPU run in PARALLEL — zero interference
NPU: 5W | GPU: 250W
```

### KV Cache Overflow to RAM

```
Standard: KV limited to VRAM = ~53K tokens
Custom paged KV:
  Hot pages (recent): VRAM (~4GB) = ~32K tokens
  Cold pages (old):   RAM (~8GB)  = ~64K tokens
  Async prefetch via PCIe 5.0 (63 GB/s)
  Total: ~96K tokens effective
  Extra latency: ~2μs per cold page access (negligible)
```

---

## 3. Serving

| Server | Tok/s (single) | Batch Throughput | Setup | Best For |
|--------|----------------|------------------|-------|----------|
| Ollama | 50-70 | ~484 tok/s | Zero config, native Windows | Dev, exploration |
| llama.cpp | 40-60 | ~200 tok/s | Minimal deps | Portability |
| vLLM | — | ~8,033 tok/s | Docker/WSL2 | Production API |
| SGLang | — | ~6,395 tok/s | Docker/WSL2 | High throughput |
| TensorRT-LLM | 350+ | ~1,406 tok/s | Complex CUDA SDK | Max RTX perf |

### Recommendation Path
Phase 1 (Explore): Ollama (zero setup)
Phase 4 (Production): SGLang or vLLM (concurrent requests, batching)

---

## Cumulative Gains (All Techniques Stacked)

### Standard Approach
```
Ollama + Qwen 3 8B Q4_K_M on RTX 5070
  Decode: 80-120 tok/s
  Context: ~53K tokens
  VRAM: ~10 GB
  Power: 250W
```

### Optimized Stack (Production-Ready Techniques)
```
Pruning 50% + AWQ + INT4 KV + Flash Attention 3 + LayerSkip
  Decode: ~200-300 tok/s
  Context: ~96K tokens (KV overflow to RAM)
  VRAM: ~4-5 GB
  Power: 250W
  Improvement: ~2.5x
```

### Maximum Custom Engine (Including Speculative Decode)
```
FP4 native + Operator fusion + CPU pipeline + Speculative decode (NPU)
  Decode: 350-450 tok/s
  Context: ~96K tokens
  VRAM: ~4-5 GB
  Power: 255W (GPU + NPU)
  Improvement: ~3-4x
```

### Theoretical Model v2 (BitNet + SSM + MoD + Early Exit)
```
  Decode: 40+ tok/s on CPU ALONE (no GPU)
  Context: Unlimited (O(1) memory)
  VRAM: 0 GB (runs on CPU)
  Memory: ~1-2 GB system RAM
  Power: ~15W
  Paradigm: GPU becomes unnecessary
```

---

## Hardware Utilization Map

| Hardware | Training | Inference | Best Role |
|----------|----------|-----------|-----------|
| RTX 5070 (12GB VRAM) | Primary (QLoRA, AdaLomo) | Primary (fast decode) | Heavy lifting |
| Intel NPU (13 TOPS) | NOT USABLE | Speculative decode draft model | Background, low-power |
| CPU (6P+4E cores) | NOT PRACTICAL | Layer pipeline (+13%) | Minor assist |
| RAM (32GB) | Dataset caching | KV cache overflow | Storage |

---

*Last updated: February 2026*
