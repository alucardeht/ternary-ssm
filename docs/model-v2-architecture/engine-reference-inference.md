# Engine Reference: Inference Pipeline

> Technical reference for how the inference engine works internally.
> Updated after each implementation step. For research agents: read THIS instead of code.

## Overview

Ternary SSM inference engine for batch=1 autoregressive generation.
Current performance: 876.8 tok/s at 6M scale (4.03x over naive baseline).

## Architecture Stack

```
generate() / generate_with_cuda_graph()
  └── CUDAGraphRunner (captures entire forward pass)
       └── model_step_graphable() (one token, all layers)
            ├── SSM layers: ssm_ssd_step_fused() (7 of 8 layers at 6M)
            │    ├── RMSNorm + in_proj (ternary GEMV)
            │    ├── Conv1d state update + SiLU
            │    ├── x_proj (ternary GEMV) → dt, B, C
            │    ├── dt_proj + softplus
            │    ├── **Fused SSM recurrence** (Triton kernel) ← Kernel E
            │    └── Gate + out_proj (ternary GEMV) + residual
            └── SWA layer: swa_step_graphable() (1 of 8 layers at 6M)
                 ├── Q/K/V projection (ternary GEMVs)
                 ├── RoPE (pre-computed outside graph)
                 ├── KV cache update (scatter_)
                 ├── Attention scores + softmax
                 └── MLP (SwiGLU with ternary GEMVs)
```

## Key Files

| File | Purpose |
|------|---------|
| `src/model_v2/inference.py` | All inference functions, CUDAGraphRunner |
| `src/model_v2/kernels/triton/ternary_gemv.py` | 2-bit packed ternary GEMV |
| `src/model_v2/kernels/triton/fused_ssm_recurrence.py` | Fused SSM state update |
| `src/model_v2/model.py` | Model architecture, generate() |
| `src/model_v2/config.py` | Scale configs (6M, 50M, etc.) |

## Tensor Shapes (6M scale: config_wikitext_6m)

| Tensor | Shape | Dtype | Location |
|--------|-------|-------|----------|
| x_t (layer input) | (1, 256) | FP16 | Between layers |
| h (SSM state) | (1, 16, 16, 32) | FP32 | Per SSM layer |
| conv_state | (1, 512, 4) | FP16 | Per SSM layer |
| k/v_cache | (1, 4, 128, 64) | FP16 | Per SWA layer |
| in_proj output | (1, 1024) | FP16 | → split to x_inner(512) + z(512) |
| x_dbc (x_proj out) | (1, 48) | FP16 | → split to dt(16) + B(16) + C(16) |
| dt (after dt_proj) | (1, 512) | FP16 | Per-channel discretization |

## Config Parameters

| Param | 6M | 50M | Purpose |
|-------|-----|------|---------|
| d_model | 256 | 640 | Hidden dimension |
| d_inner | 512 | 1280 | SSM inner dimension |
| d_state | 16 | 16 | State dimension per head |
| n_heads (SSM) | 16 | 40 | Number of SSM heads |
| head_dim | 32 | 32 | d_inner / n_heads |
| d_conv | 4 | 4 | Conv1d kernel size |
| dt_rank | 16 | 16 | dt projection rank |
| layers | 8 | 24 | Total layers |
| SSM layers | 7 | ~21 | Layers using SSM |
| SWA layers | 1 | ~3 | Layers using sliding window attention |

## Component Details

### Ternary GEMV (TernaryTritonLinear)

**How it works:** Weights stored as 2-bit packed bitmaps (32 weights per int32). Two bitmaps per row: `nonzero_mask` (which weights are ±1) and `sign_mask` (which of those are -1). FP16 scale per 128-weight group.

**Computation:** For each output element: extract bits → conditional add/subtract input → multiply by scale.

**Edge cases handled:**
- `correction_A/B`: Low-rank FP16 correction (Phase 21.A)
- `block_mixing`: Cross-channel mixing (if block_size > 0)
- `gate_pre`: Pre-quantization gating (folded into packed weights at conversion)
- `gate_post`: Post-quantization gating (applied after GEMV)

**VRAM:** ~64% less than FP16 F.linear (2-bit vs 16-bit weights)

**Speed at 6M:** 0.78x vs F.linear (kernel launch overhead dominates, not matmul speed)

### Fused SSM Recurrence (Kernel E)

**File:** `src/model_v2/kernels/triton/fused_ssm_recurrence.py`
**Status:** Implemented, correctness verified, impact NEUTRAL

**How it works:** Single Triton kernel replaces 9-14 PyTorch ops for SSM state update. Grid: (n_heads, batch). Each program handles one head's state vector.

**Algorithm per program:**
1. Load dt_head (scalar) and A_head (scalar)
2. Compute dA = exp(dt * A) — decay factor
3. Load x_head (head_dim FP32 values)
4. Loop over d_state (16 iterations, unrolled):
   - Load h[s, :], B[s], C[s]
   - h_new = dA * h + dt * B * x
   - Store h_new (in-place)
   - y_acc += C * h_new
5. Add D skip: y += D * x
6. Store output

**Shapes:** h=(1,16,16,32) FP32, dt=(1,16) FP32, x=(1,16,32) FP32
**Register usage:** ~32 per thread (head_dim=32 for h + intermediates). 768 threads total = 19% of SM register file.

**Impact:** Neutral. SSM recurrence is ~2% of layer compute. Inside CUDA Graph, inter-kernel overhead already minimal. Individual fusion doesn't help.

**Lesson:** Inside CUDA Graphs, fusing small elementwise operations has diminishing returns. The bottleneck is ternary GEMV bandwidth, not SSM arithmetic.

### CUDA Graph Capture (CUDAGraphRunner)

**How it works:** Records the ENTIRE model_step_graphable() call as a CUDA Graph, then replays it with a single CUDA API call per token. Eliminates all kernel launch overhead.

**What's OUTSIDE the graph (computed per-token before replay):**
- RoPE cos/sin values (position-dependent)
- KV cache write indices (position-dependent)
- Token ID (copied into static buffer)

**What's INSIDE the graph (static computation):**
- All layer computations (SSM, SWA, norms, projections)
- All memory reads/writes to state tensors

**Constraints:**
- All tensor shapes and memory addresses must be fixed
- No Python control flow (if/else based on values)
- No dynamic memory allocation
- No CPU-GPU synchronization (.item(), .cpu(), print of tensor values)

**Performance:** 876.8 tok/s (6.11x over Triton-only, 4.03x over baseline)

### In-Place State Mutation

**Pattern:** `h.mul_(dA).add_(dB_x)` instead of `h_new = dA * h + dB_x`

**Why:** CUDA Graphs require static tensor addresses. If we allocate new tensors, the addresses change and the graph is invalid. In-place operations modify the same memory.

**Applied to:** h (SSM state), conv_state (conv buffer), k_cache/v_cache (SWA cache)

**Memory drift:** 0.00 MB over 200 tokens (verified)

## Kernel Launch Count (before/after fusion)

| Component | Before Fusion | After Kernel E | Notes |
|-----------|--------------|----------------|-------|
| SSM recurrence (per layer) | 9-14 | 1 | Kernel E (implemented, neutral impact) |
| Conv1d + SiLU (per layer) | 5-6 | 5-6 | Kernel A (pending) |
| Gate + out_proj (per layer) | 5 | 5 | Kernel B (pending) |
| Total per token | ~215 | ~159-163 | After Kernel E only |

## Performance History

| Step | Technique | Speed | vs Baseline |
|------|-----------|-------|-------------|
| Baseline | F.linear (naive) | 217.7 tok/s | 1.00x |
| Step 1 | Triton GEMV | 170.0 tok/s | 0.78x |
| Step 2 | In-place state | 160.2 tok/s | 0.74x |
| Step 3 | CUDA Graph | 876.8 tok/s | 4.03x |
| Step 4a | Kernel E (fused SSM) | ~877 tok/s | ~4.03x (neutral) |

## Hierarchical LM Head (Experimental — NEGATIVE at 6M)

**Status:** Implemented but DISABLED. Not beneficial at current scale.

**Location:** `src/model_v2/components/hierarchical_lm_head.py`

**Concept:** Two-pass vocabulary projection:
1. Coarse: hidden @ centroids.T → (1, 128) cluster scores
2. Fine: score only tokens in top-K clusters (~2000 tokens)
3. Scatter: place fine scores into pre-allocated logits buffer

**Activation:** `prepare_for_inference(model, device, hierarchical_lm=True)`

**Why disabled:** At d_model=256 / vocab=32K, full matmul (19 us) is faster than hierarchical (342 us). The overhead of 6+ kernel launches exceeds bandwidth savings. Would be viable at d_model≥1024.

**Key tensors:**
- `centroids`: (128, 256) FP16 — cluster center vectors
- `padded_embeddings`: (128, max_cluster_size, 256) FP16 — padded for uniform access
- `logits_buf`: (1, 32000) FP16 — pre-allocated output (filled with -inf each call)
- `valid_mask`: (128, max_cluster_size) bool — masks padding positions

## INT8 Activation Quantization (Deferred — Phase 24)

**Status:** NOT implemented. Deferred to 200M+ scale.

**Why deferred:** At d_model=256, naive INT8 quantization costs 65 us (fixed overhead from abs().max → scale → divide → round → clamp → cast = 5-6 kernel launches). The matmul itself is only 15 us. 452% overhead kills any INT8 benefit.

**When to revisit:**
- d_model ≥ 1024 (matmul cost >> quantize cost)
- Fused quantize-matmul kernel (single kernel launch)
- Phase 24 scaling to 200M+

**Design notes for future:**
- Keep SSM state in FP32 (recurrent error accumulation risk)
- Quantize ONLY GEMV inputs (non-recurrent path)
- Per-channel quantization preferred over per-tensor (lower error)
- Integration point: inside `TernaryTritonLinear.forward()` (transparent to rest of pipeline)

---

## Speculative Decoding Infrastructure (Ready for Testing)

**Status:** Infrastructure implemented. Needs trained model for real validation.

### `clone_state(state)` — Deep Clone InferenceState
**Location:** `inference.py`, after `init_state`
- Deep-copies all tensors in `layer_states` (prevents in-place mutation contamination)
- Clones `v_first` if present
- Required for speculative decoding: draft and full paths must start from same state

### `model_step(..., max_layers=N)` — Early Exit
**Location:** `inference.py`, `model_step` function
- When `max_layers` is set, only first N layers execute
- `final_norm` + LM head projection still run on truncated hidden state
- State updated only for executed layers; remaining layers untouched
- NOT available in `model_step_graphable` (CUDA Graph requires fixed graph)

### Speed Scaling (measured at 6M, 8 layers):
| Draft Depth | Speed | Speedup over Full |
|-------------|-------|-------------------|
| 2 layers | 706 tok/s | 3.82x |
| 4 layers | 379 tok/s | 2.05x |
| 6 layers | 238 tok/s | 1.29x |
| 8 layers (full) | 185 tok/s | 1.00x |

## CPU Inference (Baseline — No Custom Kernels)

**Status:** Working. 105.6 tok/s at 6M scale with PyTorch MKL.

**Usage:**
```python
model, info = prepare_for_inference(model, device='cpu', backend='none')
state = init_state(model, batch_size=1, device='cpu')
logits, state = model_step(model, token, state)
```

**How it works:** Without Triton backend, TernaryLinear layers use standard PyTorch `F.linear(x, W_quantized)` where W_quantized contains {-1, 0, +1} as FP32. MKL (AVX-512) handles the matmul. SSM state stays FP32.

**Performance at 6M (8 layers, d_model=256):**
- 105.6 tok/s (9.47 ms/tok)
- Latency breakdown: Python overhead ~60%, MKL matmul ~25%, SSM/misc ~15%
- Weight data: 1.5 MB packed ternary (fits in L2 cache)

**Optimization roadmap (Phase 24):**
- Custom AVX-512 kernel: replace MKL FP32 matmul with ternary-aware add/sub → 10-50x
- C++ model_step: eliminate Python loop overhead → 3-5x
- Cache blocking: for 50M+ models where weights exceed L3 → 2x

---

## Limitations and Known Issues

- **6M scale only tested.** Kernels designed for d_model=256..640 range. 7B (d_model=4096) would need tiling.
- **Batch=1 only.** All kernels assume batch_size=1 for autoregressive generation.
- **Head_dim must be ≤ 64.** Triton programs use head_dim-sized vectors in registers.
- **NVRTC CUDA kernel exists but fragile.** `fused_ssm_step.cu` requires NVRTC runtime compilation. Triton preferred.
- **SWA RoPE pre-computed outside graph.** Position wrapping uses modulo on cache size (256 default).
