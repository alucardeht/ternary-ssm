# R117+ Hardware Theory — CPU+GPU Hybrid + Ternary Activations

## Origin (2026-03-13, user insight)

User observed that training is 100% GPU-focused, but our model is TERNARY — GPUs are optimized for float matmul which we DON'T need. Meanwhile CPU (32GB RAM, 14 cores, AVX-512, DDR5 90 GB/s) sits completely idle.

## The Problem: Wasted Hardware

| Resource       | Available   | Used in Training | Wasted |
|---------------|------------|-----------------|--------|
| GPU VRAM      | 12 GB      | 4.53 GB         | 62%    |
| GPU compute   | 30+ TFLOPs | matmul on ternary (wrong tool) | ~90% |
| **CPU RAM**   | **32 GB**  | **~0**          | **100%** |
| **CPU compute** | **AVX-512, 14 cores** | **~0** | **100%** |
| **DDR5 bandwidth** | **~90 GB/s** | **~0** | **100%** |

## Step 1: CPU+GPU Hybrid Pipeline

### Concept
- **CPU (32 GB RAM):** stores ALL frozen layers (INT2, ~4.5 GB at 7B). Runs inference through frozen ternary layers (add/sub/zero operations = memory-bound, DDR5 handles it).
- **GPU (12 GB VRAM):** ONLY the active layer (fp32 ~0.81 GB) + optimizer (~1.63 GB) + activations (~0.5 GB) = **~3 GB.**
- **Transfer:** H_k from CPU→GPU via PCIe 5.0 (32 GB/s) = ~1ms per batch at 7B.

### VRAM Impact
| Scenario           | GPU VRAM | CPU RAM  | Max Model |
|-------------------|---------|---------|-----------|
| Current (all GPU) | 4.53 GB | ~0      | 7B        |
| Hybrid CPU+GPU    | ~3 GB   | ~20 GB  | 7B trivial, 30B possible |

### Speed Estimate (7B)
- Cache advancement (CPU, 1 ternary layer × 1000 batches): ~1 sec/stage × 32 = 32 sec
- Training (GPU, 1 layer fp32): ~15-25ms/step × 1000 × 32 = 8-13 min
- Total: ~9-14 min

### Problem Identified
Cache of H_k in fp32 is FAT: 1000 batches × (4, 512, 4096) × 4 bytes = 32 GB → bad tradeoff.
Even fp16 = 16 GB. Using 20 GB RAM to save a few minutes = NOT a good trade (user feedback).

## Step 2: Ternary/Compact Activations (User Theory)

### The Core Insight
Weights are ternary (2 bits). Activations are fp32 (32 bits). **16× mismatch.**

If activations were ALSO ternary (or very compact):
- H_k cache: 1000 × (4, 512, 4096) × 2 bits = **250 MB** (vs 16-32 GB)
- **64-128× smaller cache**
- CPU RAM easily holds this
- Transfer CPU→GPU becomes negligible

### What This Enables
- Entire model (weights + activations) in compact format
- CPU does ternary × ternary = lookup tables, not matmul
- Operations become: read weight (+1/-1/0), read activation (+1/-1/0), output = simple table
- Memory-bound, and DDR5 90 GB/s is plenty
- GPU might become unnecessary for inference entirely

### Open Questions (to explore in R118+)
1. How to quantize activations to ternary without destroying gradient signal?
2. During TRAINING, backward pass needs continuous gradients — can we quantize forward activations but keep backward in fp32? (like STE for weights)
3. What's the quality impact? (R97 showed ternary WEIGHTS cost +1.06 nats — activations might be different)
4. Can we use intermediate formats (INT4, INT8) instead of full ternary as stepping stones?
5. For the ACTIVE layer being trained: activations must be fp32 for gradient. But for FROZEN layers in FP: activations could be quantized since no gradient flows through them.

### Key Distinction (from user)
> "CERTAMENTE isso tambem cabe de irmos fazendo descobertas e melhorando a eficiencia"

Approach is INCREMENTAL:
1. First: enable all hardware (CPU+GPU hybrid) — even with fp32 activations
2. Then: compress activations progressively (fp16 → int8 → int4 → ternary)
3. Each step = measurable gain, building on previous

## Step 3: Full Ternary Pipeline (Long-term Vision)

If everything is ternary:
- 7B model weights: ~1.75 GB (2 bits × 7B params)
- Activations per layer: (4, 512, 4096) × 2 bits = 2 MB (vs 32 MB fp32)
- Full cache 1000 batches: 2 GB (vs 32 GB fp32)
- Operations: ternary lookup tables, no matmul
- Hardware: CPU is IDEAL (add/sub/zero = native ops, AVX-512 does 512 bits/cycle)
- GPU optional for training backward pass only

### Training Time Estimate (fully optimized)
- 7B, ternary everything, CPU+GPU: potentially **<5 minutes**
- 30B would fit in 32 GB RAM and train in <30 min

## Hardware Reference (User's System)

- **GPU:** RTX 5070 — 12 GB GDDR7, 672 GB/s, Blackwell
- **CPU:** Intel Core Ultra 5 225F — 6P+8E cores, no iGPU, AVX-512(?), DDR5
- **RAM:** 32 GB DDR5 (~90 GB/s)
- **Target:** portable, CPU-first, GPU optional

## Implementation Plan

### R117: CPU+GPU Hybrid (immediate)
- Move frozen INT2 layers to CPU RAM
- Forward frozen layers on CPU
- Transfer H_k to GPU for active layer training
- Measure: VRAM reduction + time impact

### R118: Activation Quantization (next)
- Quantize frozen layer activations (H_k) to int8/int4
- Measure quality impact (should be small for frozen layers)
- Measure cache size reduction

### R119: Ternary Activations (experimental)
- Explore ternary activation quantization
- STE-style: ternary forward, fp32 backward
- Only for frozen layers initially

### R120+: Full Pipeline Integration
- Combine all optimizations
- 7B end-to-end benchmark
