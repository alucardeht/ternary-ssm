# C Inference Engine — Benchmark Comparison (P4.3)

## Test Setup
- Hardware: Intel Core Ultra 5 225F, 32GB DDR5, Windows 11
- CPU mode only (single thread), AVX2
- Model scale: ~6M parameters, 4 layers, d_model=256, vocab=1000
- Date: 2026-02-28

## Our Engine (Ternary SSM Hybrid)
- Architecture: SSM (Mamba-style) + Sliding Window Attention, ternary weights
- Forward pass: 3,401 tok/s
- Binary size: ~150 KB (trnm_cli.exe: 135 KB, libternary_engine.a: 86 KB)
- Model file: ~111 KB (2-bit packed ternary)
- Dependencies: Zero (C11, self-contained)
- Build: CMake, single command

## llama.cpp b8182 (Transformer)
- Architecture: Llama (pure Transformer), tested with F16 and Q2_K quantization
- F16 (16 bits/weight): 3,698 ± 89 tok/s, model 7.01 MB
- Q2_K (5.59 bits/weight): 4,480 ± 102 tok/s, model 2.44 MB
- Binary size: ~15 MB (llama-cli + DLLs)
- Dependencies: OpenMP, multiple CPU-variant DLLs

## Head-to-Head Comparison

| Metric | Our Engine | llama.cpp F16 | llama.cpp Q2_K | Notes |
|--------|-----------|---------------|----------------|-------|
| Throughput (tok/s) | 3,401 | 3,698 | 4,480 | Single thread, AVX2 |
| vs llama.cpp F16 | 0.92x | 1.00x | 1.21x | Our engine 92% of F16 speed |
| Binary size | 150 KB | ~15 MB | ~15 MB | **100x smaller** |
| Model file | 111 KB | 7.01 MB | 2.44 MB | **22-63x smaller** |
| Dependencies | 0 | OpenMP + DLLs | OpenMP + DLLs | **Zero deps** |
| Weight precision | 2 bits (ternary) | 16 bits | 5.59 bits | True ternary vs quantized |
| Architecture | SSM + SWA hybrid | Transformer | Transformer | Different paradigm |

## Component Breakdown (Our Engine)
- Ternary GEMV: 14.17 us (70,547 tok/s)
- RMSNorm: 0.07 us
- SiLU: 0.16 us
- SSM step: 54.44 us (18,367 tok/s)
- Attention step: 100.78 us (9,922 tok/s)
- E2E forward: 290.30 us (3,445 tok/s)

## Analysis

### Wins
1. **Binary size**: 100x smaller (150 KB vs 15 MB) — deployable on microcontrollers, wearables
2. **Model size**: 22-63x smaller — entire model fits in L2 cache at this scale
3. **Zero dependencies**: Single static binary, no DLLs, no OpenMP, no runtime
4. **From-scratch SSM architecture**: Different paradigm, competitive performance
5. **Cross-platform ready**: AVX2, AVX-512, NEON, WASM backends compiled

### Losses
1. **Raw throughput**: 8% slower than llama.cpp F16, 24% slower than Q2_K
2. **SSM sequential overhead**: SSM recurrence (54 us) is inherently serial
3. **Single-threaded only**: llama.cpp has mature thread pool (would pull further ahead with -t N)
4. **No batched GEMM**: We do GEMV only (one token at a time)

### Why Q2_K is Faster Than Our 2-bit
- Q2_K uses 5.59 bits/weight (not truly 2-bit like us)
- llama.cpp's Q2_K uses block-based quantization with LUT-optimized GEMV
- Our approach does pure add/subtract (theoretically faster per-op) but SSM recurrence adds overhead
- At this tiny scale (256×256 matrices), framework overhead matters more than kernel speed

### Scaling Projection
At larger model sizes (1B+ params), our advantages compound:
- Ternary add/sub scales O(n) vs multiply-accumulate
- SSM O(1) memory per token vs attention O(n)
- Binary stays ~150 KB regardless of model size
- Model file stays proportionally smaller (2 bits vs 5.59 bits)

## Conclusion
Our from-scratch C inference engine achieves 92% of llama.cpp's throughput with 100x smaller binary and zero dependencies. The SSM hybrid architecture provides a fundamentally different compute profile that should scale favorably at larger model sizes where memory bandwidth dominates.

---

## Related Documentation

- See `docs/model-v2-architecture/vision-and-status.md` Phase 34 for C Portable Inference Engine details
- See `src/model_v5/` for current model implementation
