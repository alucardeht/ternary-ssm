# Theoretical Performance Analysis — Custom Engine vs Standard Frameworks

## Status: Initial Analysis (February 2026)
**Verdict: ~3x improvement identified so far, but this is conservative. Further research needed to push boundaries.**

## How LLM Inference Actually Works

### The Fundamental Operation
Everything an LLM does reduces to matrix multiplications (matmul). Each token generated involves:

```
Input: vector of 4096 dimensions (one token)
       ×
Weight: matrix 4096 × 4096 (one layer)
       =
Output: vector of 4096 dimensions
```

A 7B model has ~32 layers, each with ~7 matmuls. To generate a single token:
```
32 layers × ~7 matmuls per layer = ~224 large matrix multiplications
```

No magic. Just giant matrix multiplications. GPU, NPU, CPU — any hardware that does fast matmul works. CUDA is not special — it's just NVIDIA's API to access cores that do matmul. Vulkan Compute, OpenCL, SYCL do the same thing.

### Two Phases: Prefill vs Decode

**Prefill (processing the prompt):**
```
Input: N tokens at once (e.g., 500 token prompt)
Operation: BATCH matmul (500 × 4096) × (4096 × 4096)
Bottleneck: COMPUTE (needs lots of math)
GPU excels here: massive parallelism
```

**Decode (generating token by token):**
```
Input: 1 token at a time
Operation: VECTOR matmul (1 × 4096) × (4096 × 4096)
Bottleneck: MEMORY BANDWIDTH (must READ entire weight matrix to multiply by 1 vector)
```

Critical insight: **decode is memory-bandwidth bound, not compute bound**. You read 4.5GB of weights from memory to do a calculation that uses 0.1% of the GPU's compute capacity. CUDA cores sit idle waiting for data.

```
RTX 5070:
  Compute: 988 TOPS     ← capacity
  Bandwidth: 672 GB/s   ← actual bottleneck

To generate 1 token with Q4 model (4.5GB):
  Minimum time = 4.5GB ÷ 672 GB/s = 6.7ms = ~150 tok/s theoretical ceiling
```

### KV Cache: Why Long Context Eats VRAM

Each processed token generates two vectors: Key and Value (from the attention mechanism). They must be stored for future tokens to reference.

```
KV Cache per token = 2 × num_layers × kv_heads × head_dim × precision
For Qwen 3 8B (32 layers, 8 KV heads via GQA, 128 dim, FP16):
  = 2 × 32 × 8 × 128 × 2 bytes = 128 KB per token

For 4096 token context:  = 4096 × 128 KB = ~512 MB
For 53K tokens:          = 53K × 128 KB  = ~6.5 GB (fills remaining VRAM)
For 128K tokens:         = 128K × 128 KB = ~16 GB (impossible in 12GB)
```

### Attention Mechanism

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

- Query (Q): "What am I looking for?" — current token's vector
- Key (K): "What do I offer?" — vector from each previous token
- Value (V): "What's my content?" — actual information from each token

The current token multiplies its Q against all previous Ks, generating relevance scores. Then uses those scores to get a weighted average of Vs.

Multi-Head Attention: Instead of one global attention, splits into 32 parallel "heads", each looking at 128 dimensions (4096 ÷ 32). Each head learns to attend to different things (syntax, semantics, position, etc).

GQA (Grouped Query Attention): Modern models (Llama 3, Qwen 3) share K and V across groups of heads. Instead of 32 KV heads, they use 8. Reduces KV cache by 4x.

## Scenario 1: Standard Framework (Ollama/llama.cpp, RTX 5070 Only)

### Decode Performance
```
Model: Qwen 3 8B Q4_K_M = ~4.5 GB
RTX 5070 bandwidth = 672 GB/s
Time per token = 4.5 / 672 = 6.7ms
Theoretical ceiling = ~150 tok/s

Overhead losses:
- CUDA kernel launch overhead
- KV cache reads (grows with context)
- CPU↔GPU synchronization
- Softmax, RMSNorm, activation functions

Realistic: 80-120 tok/s
```

### Prefill Performance
```
RTX 5070 FP16: 31 TFLOPS
500 token prompt: ~50-100ms
~2000 tok/s effective
```

### Maximum Context
```
VRAM total:          12 GB
Model Q4:           -4.5 GB
CUDA overhead:      -1.0 GB
Remaining for cache:  6.5 GB
KV per token (GQA):   128 KB
Max context: 6.5GB / 128KB = ~53,000 tokens
```

### Summary
```
Decode:       80-120 tok/s
Prefill:      ~2000 tok/s
Max context:  ~53K tokens
Power:        250W
Hardware:     GPU only (NPU and CPU idle)
```

## Scenario 2: Custom Heterogeneous Engine

### Gain 1: Native FP4 on Blackwell (+28%)

RTX 5070 has hardware FP4 support (5th-gen tensor cores). Ollama/llama.cpp use Q4_K_M (INT4 via software dequantization).

```
Q4_K_M (software dequant):
  - Read 4-bit → dequantize to FP16 in SM → compute in FP16
  - Model size: 4.5GB
  - Time: 4.5 / 672 = 6.7ms

FP4 native (hardware):
  - Tensor cores process FP4 DIRECTLY, no dequantization
  - Model size: ~3.5GB (more efficient packing)
  - Less data to read + compute integrated in tensor core
  - Time: 3.5 / 672 = 5.2ms

Result: 150 → ~192 tok/s (+28%)
```

Nobody does this properly yet because native FP4 is exclusive to Blackwell (Jan 2025). Frameworks haven't fully exploited it.

### Gain 2: Operator Fusion (+25%)

Each transformer layer does sequential operations:

```
Standard (no fusion):
  RMSNorm → READ/WRITE VRAM
  Q_proj  → READ/WRITE VRAM
  K_proj  → READ/WRITE VRAM
  V_proj  → READ/WRITE VRAM
  Attention → READ/WRITE VRAM
  O_proj  → READ/WRITE VRAM
  RMSNorm → READ/WRITE VRAM
  Gate    → READ/WRITE VRAM
  Up      → READ/WRITE VRAM
  SiLU    → READ/WRITE VRAM
  Down    → READ/WRITE VRAM

= 11 VRAM round-trips per layer × 32 layers = 352 round-trips
```

```
With fusion (custom kernels):
  [RMSNorm + Q/K/V_proj] → 1 READ/WRITE    (fused into 1 kernel)
  [Attention]             → 1 READ/WRITE    (Flash Attention)
  [O_proj + RMSNorm]      → 1 READ/WRITE    (fused)
  [Gate + Up + SiLU]      → 1 READ/WRITE    (fused)
  [Down + residual]       → 1 READ/WRITE    (fused)

= 5 round-trips per layer × 32 = 160 round-trips
```

Reduces intermediate reads/writes by ~55%. Activations stay in registers/shared memory instead of round-tripping to VRAM.

```
Realistic gain: ~20-30% less time in memory overhead
192 → ~240 tok/s
```

### Gain 3: Speculative Decoding with NPU (Game-Changer, +80-120%)

The most important technique. This is where the NPU truly shines.

```
PROBLEM with normal decode:
  GPU generates 1 token → reads 4.5GB → generates 1 token → reads 4.5GB → ...
  Each token costs a full read of the model.

SPECULATIVE DECODING:
  1. NPU runs SMALL model (3B, ~2GB in INT8)
     and "drafts" next 5 tokens at once
     Cost: ~5ms on NPU (in background, ~5W)

  2. GPU takes the 5 drafted tokens and VERIFIES all
     at once in a single forward pass of the large model (8B)
     Cost: 1 forward pass (~6ms), but verifies 5 tokens

  3. If draft model got 3 of 5 correct:
     Result: 4 accepted tokens (3 correct + 1 from large model)
     in 1 GPU forward pass

  WITHOUT speculative: 1 token per forward pass = 6.7ms/token
  WITH speculative: ~4 tokens per forward pass = 6.7ms/4 = 1.7ms/token
```

```
Typical acceptance rate of 3B draft → 8B target: 60-75%
With γ=5 (5 draft tokens):
  Accepted tokens per step = 1 + 0.7 × 5 × acceptance_efficiency ≈ 3-4

Effective throughput:
  240 tok/s × 3.5 (speculative multiplier) ÷ ~1.5 (verification overhead)
  ≈ 400-560 tok/s theoretical

Realistic (with overheads): 350-450 tok/s
```

Why NPU is perfect for this:
- Draft model runs in parallel with GPU (zero interference)
- Consumes 5W (vs 250W for GPU)
- Uses system RAM (not VRAM)
- While GPU verifies batch N, NPU already generates draft for batch N+1

### Gain 4: KV Cache Overflow to RAM (+context)

```
Standard:
  KV cache limited to remaining VRAM = ~53K tokens

Custom (Paged KV with overflow):
  Hot pages (recent tokens):  VRAM (~4GB) = ~32K tokens
  Cold pages (older tokens):  RAM (~8GB)  = ~64K tokens
  Async prefetch via PCIe 5.0 (63 GB/s)

  Total: ~96K tokens effective context

  Cost: ~0.1ms extra latency when accessing cold page
  (PCIe 5.0: 128KB / 63 GB/s = 2μs, negligible)
```

### Gain 5: GPU↔CPU Layer Pipeline (Modest, +10-15%)

Splitting layers between GPU and CPU:

```
GPU bandwidth (VRAM):  672 GB/s
CPU bandwidth (RAM):   102.4 GB/s (DDR5-6400 dual channel)

Optimal split (proportional to bandwidth):
  GPU: 28 layers (87.5%) → 3.94GB / 672 = 5.86ms
  CPU: 4 layers (12.5%)  → 0.56GB / 102.4 = 5.47ms

With cross-token pipeline:
  Time per token = max(5.86, 5.47) = 5.86ms = ~170 tok/s

vs GPU-only: 150 tok/s = +13%
```

Note: NPU does NOT help here — it shares RAM bandwidth with CPU, so giving layers to NPU actually hurts (competes for bandwidth).

## Cumulative Gains Breakdown

```
Base (bandwidth limit):          150 tok/s   (1.0x)
+ FP4 native:                    192 tok/s   (1.28x)
+ Operator fusion:               240 tok/s   (1.60x)
+ CPU pipeline:                  270 tok/s   (1.80x)
+ Speculative decoding (NPU):    400 tok/s   (2.67x)
+ Minor optimizations:           450 tok/s   (3.0x)

Total theoretical improvement: ~3x
```

## Where NPU Helps vs Doesn't

| Use Case | NPU Helps? | Why |
|----------|-----------|-----|
| Speculative decoding (draft model) | **Yes, significantly** | Runs in parallel, zero cost to GPU |
| Decode layers | **No** | Competes for RAM bandwidth with CPU |
| Always-on background model | **Yes** | 5W vs 250W, saves energy |
| Prefill | **No** | GPU is orders of magnitude faster |

## Training (Fine-tuning) Impact

```
Standard QLoRA:     ~11.5 GB VRAM, 4-6h for 5K examples
Custom engine:
  - Optimizer states offload to RAM: frees ~1.5GB
  - Optimized gradient accumulation: -10% overhead
  - Custom mixed precision: -15% time

Realistic speedup: ~1.3x (not the main focus)
```

## Open Questions for Further Research

1. Can we push beyond 3x? Areas to investigate:
   - Custom SRAM/register tiling strategies for Blackwell SMs
   - Asynchronous weight prefetching between layers
   - Custom tokenizer optimized for specific use cases
   - Batch-optimized attention for multi-request serving
   - Dynamic precision switching per-layer based on token importance
   - Weight sharing / pruning at inference time
   - Custom memory allocator that eliminates fragmentation
   - Exploiting PCIe 5.0 DMA for zero-copy transfers
   - NVIDIA's Blackwell-specific SM architecture optimizations
   - Intel AMX (Advanced Matrix Extensions) on Arrow Lake P-cores
   - Compiler-level optimizations (auto-vectorization, loop unrolling for matmul)
   - Novel attention algorithms beyond Flash Attention
   - Mixture of quantizations within a single layer
   - Hardware-aware model pruning (structured sparsity matching SM warp size)

2. The current 3x estimate may be conservative because:
   - We haven't fully explored Blackwell-specific features
   - Speculative decoding acceptance rates could be higher with better draft models
   - Operator fusion depth could go beyond what's outlined
   - We're not yet considering hardware-level optimizations (warp scheduling, memory coalescing)
   - Novel algorithms from 2025-2026 papers haven't been evaluated yet

---

*This analysis is Phase 1. Further deep research into each optimization area is needed before architecture design.*
*Last updated: February 2026*
