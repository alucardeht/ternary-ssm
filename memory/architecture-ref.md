---
name: Model architecture quick reference
description: v11 model architecture — block types, dimensions, configs, weight representation
type: reference
---

# Model v11 Architecture Reference

**Base:** Forked from v10 final (R121)

## Architecture Type

SSM hybrid: ~87.5% Mamba-2 SSD + ~12.5% Sliding Window Attention (SWA)

## Block Types

1. **SSM Block (SelectiveSSMSSD):** Mamba-2 SSD with chunked scan (training) / recurrent O(1) (inference)
2. **SWA Block:** Full transformer block (attention + SwiGLU FFN), window=128 tokens
3. **Normalization:** BitShiftNorm (default) or RMSNorm
4. **Pattern:** pre-norm + residual on all layers

## Configurations

| Config | d_model | n_layer | SWA layers | ~Params |
|--------|---------|---------|------------|---------|
| 6M (dev) | 256 | 8 | 1 (idx 4) | ~6M |
| 50M | 640 | 18 | 2 (idx 6,12) | ~50M |
| 100M | 768 | 24 | 3 (idx 6,12,18) | ~100M |
| 7B PoC | 4096 | 32 | ? | 3.38B (vocab=256) |

**IMPORTANT:** The "7B PoC" used vocab=256 (byte-level). With vocab=32K it would be ~3.9B params. True 7B needs larger dimensions.

## Weight Representation

- **Training:** FP32 shadow weights (standard gradient descent)
- **Frozen layers:** INT2 packed (4 weights/byte, {-1,0,+1})
- **Inference:** INT2 (2-bit ternary, TRNM format)
- **Ternary training (STE):** REJECTED (+1.06 nats overhead, R97)

## Inference

- SSM layers: O(1) per token (recurrent mode, no KV cache)
- SWA layers: O(window=128) memory
- No fixed context limit (SSM state is constant size)
- Export format: TRNM (2-bit packed, CRC32C, 64-byte aligned)

## Kernels Available

| Kernel | Target | Status |
|--------|--------|--------|
| Triton GEMV | GPU, B=1 | Complete |
| T-SAR CPU scatter | CPU, B=1 | Complete |
| C Portable (AVX2) | CPU | Complete |
| POPCOUNT (phase 1) | Simulation | 10x SLOWER than matmul |
| POPCOUNT (phase 2, CUDA __popc) | GPU | NOT IMPLEMENTED |

## What's Active vs Orphan

**Active in forward pass:** SSM blocks, SWA blocks, BitShiftNorm, residual connections, embedding, lm_head
**Available but off by default:** RWKV7, MoE, MoD router, InstantProjector, EigenmodeMixed
**Utility only:** PTQ quantization, ternary serialization, ACSI init, SSM aux losses
