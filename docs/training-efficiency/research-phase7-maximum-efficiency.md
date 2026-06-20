# Phase 7 Research: Maximum Training Efficiency

**Research date:** February 2026
**Status:** Research complete, testing pending

## Overview

Phase 7 targets squeezing every possible optimization from the training pipeline to train the largest model possible on RTX 5070 12GB + 32GB DDR5 RAM (Windows 11, PyTorch 2.10.0+cu128).

## Scaling Math — Maximum Trainable Model Size

### Memory Breakdown (Training Mode)

For our ternary architecture (~87% ternary, ~13% FP16):

| Component | Formula | Notes |
|-----------|---------|-------|
| Weights (latent FP16) | ~2N bytes | STE needs FP16 latent for ALL params |
| Optimizer (AdamW 8-bit) | ~2N bytes | bitsandbytes 8-bit moments |
| Gradients (AMP FP16) | ~2N bytes | With AMP enabled |
| Activations (GradCkpt) | ~10-15% of full | Recompute on backward |

### VRAM Estimates by Scale

With all optimizations (AMP + GradCkpt + 8-bit optimizer):

| Scale | VRAM | Batch Size | Time/3 epochs (WikiText-103) | Verdict |
|-------|------|-----------|------------------------------|---------|
| 50M | ~800 MB | 32+ | ~13h | Easy |
| 100M | ~1.4 GB | 16-32 | ~17h | Easy |
| 500M | ~5 GB | 8-16 | ~1.7 days | Comfortable |
| 1B | ~9 GB | 4-8 | ~2.5 days | Tight but feasible |
| 3B | ~24 GB | — | — | Impossible (2x VRAM) |
| 7B | ~52 GB | — | — | Impossible |

**Answer:** Max realistic = **1B params** (tight), **500M comfortable**.

### Why 3B+ is Impossible

- Activations alone at 3B: ~6 GB (even with GradCkpt)
- Weights (latent): ~6 GB, Optimizer: ~6 GB, Gradients: ~6 GB
- Total: ~24 GB = 2x our VRAM
- CPU offloading helps optimizer (-6 GB) but still 18 GB > 12 GB

### Ternary Training Memory Reality

- **Inference:** 4x smaller (ternary storage) — TRUE
- **Training:** Only ~1.5x less VRAM vs FP16 — latent weights + optimizer dominate
- FP16 latent weights are MANDATORY for STE gradient updates during training

## Techniques — Adopt

### ~~1. ONNX Runtime Training (ORTModule)~~ — REJECTED

- **Source:** onnxruntime.ai/training
- **Initial promise:** +30-40% training throughput via graph optimization
- **Why rejected (detailed review Feb 2026):**
  1. **CUDA 12.8 incompatible** — official builds support up to CUDA 12.1
  2. **PyTorch 2.10 incompatible** — ORTModule lags by 3-6 months, supports up to 2.3
  3. **SM_120 (Blackwell) not supported** — kernel compilation targets older compute capabilities
  4. **STE gradient risk** — ONNX graph fusion may eliminate quantization node in backward pass, causing silent training failure
  5. **SSM scan not optimizable** — ONNX cannot parallelize Python loops (our core bottleneck)
  6. **Realistic speedup only 5-10%** — the 30-40% figure is for BERT/GPT transformers, not SSMs
  7. **Windows stability** — 10-15% crash rate on Windows (DLL loading errors)
- **Conclusion:** High risk, low reward. Not worth pursuing until CUDA 12.8 + PyTorch 2.10 + Blackwell are officially supported.

### 2. torchao bitnet_training()

- **Source:** PyTorch AO 0.6.1+ (github.com/pytorch/ao)
- **What:** Official PyTorch recipe for ternary training. `quantize_(model, bitnet_training())`. Packed 2-bit weights, INT8 activation kernels via `gemm_lowbit`.
- **Benefit:** Memory (2-bit packed storage) + Speed (INT8 Tensor Cores)
- **Complexity:** Easy (one-line API)
- **Status:** NEEDS TESTING — verify compatibility with our TernaryLinear/SSD/MoD

### 3. FSDP2 CPU Offload

- **Source:** PyTorch 2.10 native (torch.distributed.fsdp.fully_shard)
- **What:** Offloads optimizer states, gradients, and optionally parameters to CPU RAM. Single GPU supported.
- **Benefit:** Eliminates optimizer VRAM (~2N bytes). With 8-bit optimizer already saving 50%, this moves remaining 50% to CPU = ~75% total optimizer VRAM gone.
- **Complexity:** Medium
- **Compatibility:** PyTorch native, Windows-friendly, requires fsdp_use_orig_params=True for custom autograd
- **Status:** NEEDS TESTING — verify STE compatibility with offloaded gradients

### 4. Selective Activation Offloading

- **Source:** torchtune v0.4.0+, TorchTitan 2025
- **What:** Large activations offload to CPU, small ones recompute (extends GradCkpt). Async prefetching hides latency.
- **Benefit:** +2-3x model size on top of GradCkpt
- **Complexity:** Medium
- **Status:** Deferred — test after FSDP2 offload

### 5. CUDA Allocator Config

- **What:** `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` reduces memory fragmentation
- **Benefit:** Reduces OOM from fragmentation (not total VRAM reduction)
- **Complexity:** 1 line env var
- **Status:** ADOPT IMMEDIATELY — no downside

### 6. DataLoader Tweaks

- **What:** `persistent_workers=True`, `pin_memory=True`, `num_workers=4-8`
- **Benefit:** Faster epoch starts, faster CPU→GPU transfer
- **Complexity:** Trivial
- **Status:** ADOPT IMMEDIATELY — no downside

### 7. Batch Size Maximization

- **Source:** NeurIPS 2025 "Small Batch Size Training for Language Models"
- **Finding:** Small batches are NOT harmful. Maximize batch_size to fill VRAM, reduce grad_accum.
- **Why:** Fewer optimizer steps = faster wall-clock, same final quality
- **Status:** Apply during testing — try batch_size=64-128 with grad_accum=1-2

## Techniques — Investigate

### 1. ZenFlow (DeepSpeed, Aug 2025)

- **What:** Stall-free offloading via top-k gradient prioritization. Up to 5x speedup vs ZeRO-Offload.
- **Risk:** Bleeding edge, Windows stability unknown
- **Status:** Test AFTER FSDP2 baseline established

### 2. CUTLASS 4.3+ Ternary GEMM

- **What:** Check if CUTLASS has ternary-specific GEMM kernels for SM_120 (Blackwell)
- **Risk:** May not exist yet
- **Status:** Research documentation first

### 3. Ternary Binary-Decomposition (ICML 2025)

- **What:** Transform ternary matrix → 2 binary matrices, use efficient binary matmul
- **Risk:** Needs custom CUDA kernel implementation
- **Status:** Low priority — only if kernel perf matters

### 4. Microsoft BitNet GPU Kernel

- **Source:** github.com/microsoft/BitNet (May 2025 GPU, Jan 2026 CPU)
- **What:** Fused RMSNorm + BitLinear in SRAM. dp4a dot products for 2-bit × 8-bit.
- **Relevance:** Inference-focused but fusion technique applicable to training forward pass
- **Status:** Investigate for forward pass optimization

## Techniques — Rejected

| Technique | Reason |
|-----------|--------|
| SuperOffload (Oct 2025) | Requires GH200 hardware (NVLink-C2C), not PCIe |
| Layer-by-layer GPU processing | Too slow vs FSDP offload |
| Reduced-precision gradients | Gradient shrinkage violates stability requirements |
| Rust/C++ training loop | SSM scan is GPU-bound, not language-bound |
| DTR (Dynamic Tensor Rematerialization) | 2021 prototype, no production integration |
| Split ternary/latent CPU/GPU storage | Transfer overhead ~125ms/backward for 1B params |
| FP4 native for ternary | FP4 Tensor Cores do 4-bit float matmul, not 2-bit int additions |
| mamba-ssm native Windows | Triton dependency blocks it; use WSL or pure PyTorch |
| torch.compile GPU on Windows | Still blocked (CPU backend only on Windows) |
| T-MAC | CPU/NPU inference only |
| Gradient-free training | Too experimental, conflicts with STE-based training |

## Key Insights

1. **ORTModule is the torch.compile replacement for Windows** — same graph optimization, no Triton
2. **torchao has official BitNet training recipe** — we may be able to replace our custom TernaryLinear
3. **FP4 on RTX 5070 doesn't help ternary** — ternary forward is addition-based, not matmul
4. **Activation memory dominates at scale** — 50-60% of total, GradCkpt MANDATORY
5. **Training memory ≠ inference memory** — ternary saves 4x at inference but only ~1.5x during training
6. **PCIe 5.0 offloading is viable** — real benchmarks show only ~6% perf impact for AI workloads

## Testing Plan (Revised Feb 2026)

ORTModule rejected after detailed compatibility review. Revised plan:

### Phase 7.1: Profiling (establish baseline) — COMPLETE

Profiled 12M combined model (ternary+routing+SSD, AMP, batch=32, seq_len=256).
Peak VRAM: 5243 MB. Self CUDA time: 346ms over 10 active steps.

**CUDA time breakdown (where GPU spends time):**

| Operation | CUDA Time | % Total | What |
|-----------|-----------|---------|------|
| aten::mul | 84ms | 24.3% | Element-wise multiply (STE scaling, AMP scaler, routing weights) |
| aten::copy_ | 75ms | 21.6% | Dtype conversions (AMP FP16↔FP32) |
| aten::mm | 50ms | 14.5% | Matrix multiply (linear projections) |
| masked_fill | 31ms | 9.0% | Sliding window attention mask |
| clone | 26ms | 7.5% | Tensor cloning |
| cross_entropy | 23ms | 6.7% | Loss (vocab=32000) |
| sum | 19ms | 5.5% | Reductions (norms, means) |
| DtoD memcpy | 16ms | 4.6% | Device-to-device copies |

**CPU time breakdown:**

| Operation | CPU Time | % Total | What |
|-----------|----------|---------|------|
| cudaStreamSynchronize | 127ms | 22.8% | GPU sync from .item() calls |
| cudaLaunchKernel | 53ms | 9.6% | Kernel launch overhead |
| TernaryQuantize | 49ms | 8.8% | Custom STE autograd |
| aten::linear | 38ms | 6.8% | Linear dispatch |

**Key insights:**
1. Bottleneck is NOT matmul (only 14.5%) — it's element-wise ops (mul 24%) and dtype copies (22%)
2. AMP dtype conversions (copy_) are expensive — FP16↔FP32 casting overhead
3. item() syncs waste 23% CPU time — reduce logging frequency
4. TernaryQuantize is Python-dispatch heavy (9% CPU)
5. Cross-entropy at vocab=32000 is non-trivial (6.7%)

**Optimization targets (by expected impact):**
1. Reduce copy_ (dtype conversions): minimize unnecessary AMP autocasting → 10-15%
2. Fuse element-wise ops (mul): STE + quantization fusion → 5-10%
3. Reduce item() sync: less frequent logging → 5-10% CPU
4. DataLoader: pin_memory, persistent_workers → 2-5%

### Phase 7.2: Quick Wins (config changes only) — COMPLETE (ALL NEGATIVE)

Tested on 12M combined model (ternary+routing+SSD, WSD+AMP), 500 steps, WikiText-103.

| Variant | Tok/s | Time | VRAM | Loss | vs Baseline |
|---------|-------|------|------|------|-------------|
| Baseline (b32, default DL) | 99,017 | 41.4s | ~2.0 GB | 6.2986 | — |
| DL Optimized (workers=2, pin_memory) | 92,094 | 44.5s | ~2.2 GB | 6.3046 | -7.0% WORSE |
| Batch 64 | 57,980 | 141.3s | ~4.0 GB | 6.2666 | -41.4% WORSE |

**All negative for throughput. No quick config wins available.**

**Why DataLoader workers hurt (-7%):**
- Windows uses spawn (not fork) for multiprocessing — high overhead per worker
- WikiText-103 with memmap is already fast (single-process IO)
- Worker startup cost > data loading benefit at this scale
- **Recommendation: keep num_workers=0 on Windows with memmap datasets**

**Why batch=64 hurts (-41%):**
- SSM scan is sequential per token — larger batch adds proportionally more serial work
- Per-token cost: 17.2μs (b64) vs 10.1μs (b32) — 70% more time per token
- Loss is slightly better (6.27 vs 6.30) due to larger effective batch reducing gradient noise
- But throughput loss far outweighs quality gain
- **Recommendation: keep batch=32 (or smaller) for SSM-dominant architectures**

**CUDA allocator (expandable_segments:True):** Neutral. No OOM events to compare against.

**Conclusion:** Config-level optimizations don't help because the bottleneck is internal:
- dtype conversions (aten::copy_ 22%) from AMP FP16↔FP32 casting
- element-wise ops (aten::mul 24%) from STE scaling and routing
- These require model-level code changes, not config tweaks

### Phase 7.3: torchao BitNet Training — COMPLETE (REJECTED)

**Research finding: torchao bitnet_training() is NOT ternary — it's INT8 QAT.**

| Feature | Our TernaryLinear | torchao bitnet_training() | torchao ternary (experimental) |
|---------|-------------------|--------------------------|-------------------------------|
| Values | {-1, 0, +1} | INT8 (-128 to 127) | {-1, 0, +1} |
| Threshold | TWN adaptive (0.7×mean) | Learned scale factors | Adaptive (basic) |
| STE types | Standard, HESTIA, bi-STE | Not STE (dequantize path) | Standard STE only |
| Group-wise | Yes (group_size=128) | Per-tensor | Per-tensor only |
| Gradual threshold | Yes | No | No |
| Pressure blend | Yes | No | No |
| Storage | FP16 during training | INT8 packed | 2-bit packed |

**Why rejected:**
1. `bitnet_training()` is wrong quantization scheme (INT8, not ternary)
2. `int8_weight_only_quantized_training()` (experimental ternary) lacks HESTIA, gradual threshold, pressure blend
3. No group-wise quantization in torchao (we use group_size=128)
4. Our TernaryLinear validated at -12.4% loss improvement (Phase 4) — torchao would lose this
5. torchao ternary has no training speedup (0.9-1.1x)

**Potential future use:** torchao `AffineQuantizedTensor` for 2-bit packed storage (memory-only benefit).
**Recommendation:** Keep our custom TernaryLinear. It is more advanced than torchao for ternary training.

### Phase 7.4: Deep Foundations Research — COMPLETE

**Pivot:** Instead of incremental fixes (FSDP2, dtype tweaks), went deep into mathematical roots.
**Full details:** `research-phase7-deep-foundations.md`

**Key discoveries:**
1. **Ternary matmul is NOT multiplication** — it's selection + addition, 3.4x fewer FLOPs than dense matmul
2. **FP32 latent weights NOT necessary** — integer gradient accumulation (INT16) works, proven in 2025-2026 papers
3. **SSM log-space cumsum replaceable** — integer prefix sum with bit-shift decay
4. **RWKV-7 gating is ternary-native** — element-wise ops instead of matrix multiply in recurrence
5. **46% PyTorch overhead is STRUCTURAL** — framework fights ternary architecture fundamentally

**Four paths identified:**
- **Path A: Custom kernels in PyTorch** — 2-4x speedup, 1-2 weeks, lowest risk
- **Path B: JAX/XLA rewrite** — 2-5x speedup, cleaner SSM math, medium risk
- **Path C: Integer-domain training** — 5-16x memory reduction, eliminate FP32 entirely
- **Path D: Architecture paradigm shift** — RWKV-7 gating + slot memory, potentially 10x+

### Phase 7.5: Prototype Testing — IN PROGRESS

**Path C.1 (Integer Gradient Accumulation): PARTIALLY TESTED**
- Model learns (loss 10.4 → 6.32) but gap vs STE baseline is +6.8%
- CRITICAL FINDING: All threshold values (0.1 to 2.0) produce nearly identical loss
- Weight flips are NOT the learning mechanism — alpha scale parameter does all the work
- 9 flips (t=2.0) achieves same loss as 51K flips (t=0.1)
- Random ternary {-1,0,+1} + learned scalar = significant learning, but pattern evolution needed
- Update rule needs refinement — more aggressive flipping or per-group scales
- **Full details:** `testing-roadmap-deep-foundations.md`

**Remaining tests:**
- Path C.1 refinement: lower thresholds, per-group alpha, 50M scale
- Path A: Custom CUDA kernels for ternary matmul
- Path D: RWKV-7 gating layer prototype

## References

### Offloading & Memory
- DeepSpeed ZeRO-Offload: deepspeed.ai/tutorials/zero-offload/
- ZenFlow (Aug 2025): pytorch.org/blog/zenflow-stall-free-offloading-engine-for-llm-training/
- FSDP2 CPU Offload: docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
- torchtune Memory Optimizations: docs.pytorch.org/torchtune/0.5/tutorials/memory_optimizations.html
- STAlloc (EUROSYS '26): arxiv.org/pdf/2507.16274

### Kernels & Implementations
- torchao BitNet: github.com/pytorch/ao
- Microsoft BitNet: github.com/microsoft/BitNet
- CUTLASS 4.x: github.com/NVIDIA/cutlass
- T-MAC: github.com/microsoft/T-MAC
- Ternary Binary-Decomposition (ICML 2025): arxiv.org/html/2411.06360v3
- ONNX Runtime Training: onnxruntime.ai/training

### Scaling & Profiling
- NeurIPS 2025 Small Batch: arxiv.org/abs/2507.07101
- Chinchilla Scaling: arxiv.org/abs/2203.15556
- PyTorch Profiler: docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
- FFCV: github.com/libffcv/ffcv

---

*Last updated: February 2026*
