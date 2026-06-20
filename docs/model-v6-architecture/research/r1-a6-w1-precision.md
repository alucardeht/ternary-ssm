# Wave 1 - Aspect 6: Hybrid Precision (FP8, INT8)

## Research Context
- **Architecture:** Ternary SSM (State Space Model) with weights in {-1, 0, +1}
- **Focus:** Training speed optimization through hybrid precision
- **Date filter:** 2024-2026 papers only

## Findings

### ID: FP8-HOPPER-TRAINING
- **Src:** https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/ [2024] | **Gain:** Up to 2x throughput vs BF16 | **V:** PROMISING
- **Principle:** FP8 uses E4M3 (forward pass) and E5M2 (backward) formats with scaling factors to prevent overflow
- **Application:** Use NVIDIA Transformer Engine with FP8 recipe for H100 GPUs

### ID: MXFP8-BLOCK-SCALING
- **Src:** https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/ [2025] | **Gain:** Per-block scaling reduces quantization error | **V:** PROMISING
- **Principle:** MXFP8 uses block-level scaling (32 values per block) vs tensor-level, better handling of dynamic range
- **Application:** Use MXFP8 on Blackwell GPUs for finer-grained precision control

### ID: FP8-VS-BF16
- **Src:** https://acecloud.ai/blog/fp8-vs-bf16-mixed-precision-tensor-cores/ [2024] | **Gain:** 1.5-2x speedup vs BF16 on H100 | **V:** PROMISING
- **Principle:** FP8 has 4x fewer bits than BF16, enabling higher Tensor Core throughput
- **Application:** Start with BF16 as baseline, then add FP8 for selected layers on H100

### ID: FP8-DELAYED-SCALING
- **Src:** https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/ [2024] | **Gain:** Stable scaling factor | **V:** PROMISING
- **Principle:** Delayed scaling uses history of max values to compute scale, avoiding contamination from transient spikes
- **Application:** Use delayed scaling (delay=16) for FP8 training stability

### ID: FP8-PER-TENSOR-CURRENT
- **Src:** https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/ [2024] | **Gain:** Adaptive scaling per step | **V:** SPECULATIVE
- **Principle:** Per-tensor current scaling determines scale from current forward pass statistics
- **Application:** Use for workloads with rapidly changing activation magnitudes

### ID: INT8-TRAINING-WITH-SCALE
- **Src:** Research on INT8 training [2024] | **Gain:** 2x memory reduction | **V:** SPECULATIVE
- **Principle:** INT8 requires careful scaling but enables 2x memory reduction vs FP16
- **Application:** Use per-channel scaling for INT8 weights and activations

### ID: TENSOR-CORE-OPTIMIZATION
- **Src:** https://markaicode.com/tensorflow-h100-tensor-core-optimization/ [2024] | **Gain:** Up to 4x speedup | **V:** PROMISING
- **Principle:** Tensor Cores accelerate matrix ops in reduced precision, require dimensions multiples of 8
- **Application:** Ensure model dimensions are multiples of 8 for FP16/BF16, 16 for FP8

### ID: BF16-AS-BASELINE
- **Src:** https://acecloud.ai/blog/fp8-vs-bf16-mixed-precision-tensor-cores/ [2024] | **Gain:** Safe default precision | **V:** PROMISING
- **Principle:** BF16 has same exponent range as FP32 but fewer mantissa bits, providing stability
- **Application:** Use BF16 as default mixed precision, add FP8 only on H100 for selected layers

### ID: FP8-RECIPES
- **Src:** https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/ [2024] | **Gain:** Optimized scaling strategies | **V:** PROMISING
- **Principle:** FP8 recipes define when to use tensor vs block scaling, delayed vs current scaling
- **Application:** Start with delayed scaling recipe, tune per-layer based on convergence

### ID: MXFP8-NEMOTRON
- **Src:** https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/ [2025] | **Gain:** Matches BF16 convergence | **V:** PROMISING
- **Principle:** MXFP8 on Blackwell achieves similar perplexity as BF16 with 2x speedup
- **Application:** For 8B+ models, MXFP8 provides best throughput/accuracy tradeoff

## Key Insights
- FP8 on H100 provides 1.5-2x speedup vs BF16 with proper scaling
- MXFP8 on Blackwell enables finer-grained precision with per-block scaling
- BF16 should be baseline for most training, FP8 added selectively
- Tensor Core optimization requires dimension alignment (multiples of 8)
- Scaling recipes (delayed/current, tensor/block) critical for convergence

## Seed Questions for Wave 2
1. How does FP8 interact with ternary weight STE gradients during backward?
2. Can block-level FP8 scaling be combined with ternary weight quantization?
