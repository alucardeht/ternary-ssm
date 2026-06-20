# v12 C Implementation — STATUS (2026-03-30) [FINAL]

## FASE ATUAL: GPU Pipeline Complete — Run-018 Validation in Progress

### Throughput Evolution (FINAL)
| Batch | Original (CPU) | Current (GPU) | Speedup |
|-------|---------------|---------------|---------|
| 1     | 226 tok/s     | 708 tok/s     | 3.1x    |
| 4     | 286 tok/s     | **1249 tok/s**| **5.5x**|
| 8     | —             | 1110 tok/s    | —       |
| 16    | —             | ~1150 tok/s   | —       |

**FINAL NUMBER: batch=4 → 1249 tok/s (5.5×)**

### What Was Implemented (2026-03-29 / 03-30 / 03-30 session 2 / FINAL)
- generate.c: autoregressive text generation from checkpoint (CPU-only)
- Batch>1: 13 offset bugs fixed + SSM scan shared mem [512->1024]
- GPU sampled CE loss: 3 shaders (sampled_ce_forward, sampled_ce_scatter_dh, norm_grad_reduce)
- GPU dlm_head scatter: CAS-loop atomicAdd shader
- GPU A-backward: 2 shaders (a_backward_dgate, a_backward_dx_cv)
- GPU conv1d+gate backward: shader created but regresses due to vk_fill_zero overhead
- Batch command recording: staging ring allocator + deferred downloads (8 batch sections)
- Eliminated unnecessary GPU->CPU downloads (Tier 1+2)
- Eliminated d_residual backward download: -96 MB/step
- Eval loop every 500 steps
- Geometric init (128 timescales tau 1-512), B* balanced init (B[i]=sqrt(1-A[i]))
- tok/s + VRAM logging
- Fixed: gradient scaling for batch>1, static thread-safety in loss.c, Makefile clean
- Fixed: repack off-by-one, gradient accumulation bug, shared scratch buffer bug, norm gradient accumulation, VK_MAX_BUFFERS overflow, MTP 3 bugs, orth_sum silent disable
- GPU optimizer: ALL 13 param groups active on GPU; CPU optimizer fully skipped
- Runtime --batch flag: batch size configurable at launch without recompile (--batch N)
- GPU CSR rebuild: ternary CSR index rebuilt on GPU after mvue_resample
- Checkpoint save: GPU→CPU sync before writing
- VK_MAX_BUFFERS raised to 768 (571 in use)
- Gradient accumulation: fixed accumulation bug; 16 steps active

### GPU Shaders (40 total)
| Shader | Purpose | Status |
|--------|---------|--------|
| ternary_matvec | CSR gather-sum | ACTIVE |
| fp32_gemv | batched GEMV | ACTIVE |
| fp32_gemm | C = A^T @ B | ACTIVE |
| residual_add | y += x | ACTIVE |
| ssm_scan | Blelloch parallel prefix | ACTIVE |
| ssm_scan_backward | suffix scan | ACTIVE |
| ssm_bwd_elementwise | post-scan gradients | ACTIVE |
| elementwise_gates | fused gate+conv1d+Direct-B | ACTIVE |
| output_project_wht | C@h GEMV + WHT | ACTIVE |
| conv1d_bwd_weights | weight gradient reduction | ACTIVE |
| rms_norm_forward | RMS norm | ACTIVE |
| rms_norm_backward | RMS norm backward | ACTIVE |
| wht_inplace | Walsh-Hadamard | ACTIVE |
| token_gather | embedding lookup | ACTIVE |
| sampled_ce_forward | sampling + dots + loss + d_logits | ACTIVE |
| sampled_ce_scatter_dh | d_h gradient scatter | ACTIVE |
| norm_grad_reduce | d_norm_w/b reduction | ACTIVE |
| dlm_head_scatter | lm_head gradient with CAS atomicAdd | ACTIVE |
| a_backward_dgate | d_dot + dA/db_A reduction | ACTIVE |
| a_backward_dx_cv | W_A^T @ d_dot per-timestep | ACTIVE |
| embed_e_backward | embedding gradient scatter | ACTIVE |
| pc_gather_prev | gather previous hidden state for PC | ACTIVE |
| residual_sub | residual subtraction for PC error | ACTIVE |
| pc_negate_gather | negate and gather for PC backward | ACTIVE |
| residual_scatter_add | scatter-add for residual path | ACTIVE |
| broadcast_fill | fill buffer with broadcast value | ACTIVE |
| mtp_loss | multi-token prediction loss | ACTIVE |
| direct_b_backward | Direct-B weight gradient | ACTIVE |
| dc_bilinear_reduce | bilinear reduction for D/C gradients | ACTIVE |
| welford_update | Welford online mean/var for orth_sum | ACTIVE |
| dlm_head_scatter (new) | lm_head gradient with CAS atomicAdd | ACTIVE |
| adam_dense | dense Adam update for norm/A/b_A/conv_w/W_A | ACTIVE |
| grad_scale | gradient scaling for batch>1 | ACTIVE |
| factored_adam_reduce | factored Adam: row/col gradient reduction | ACTIVE |
| factored_adam_moment | factored Adam: moment update | ACTIVE |
| factored_adam_update | factored Adam: parameter update | ACTIVE |
| mvue_resample | MVUE ternary resampling | ACTIVE |
| csr_count | count nonzeros per row for CSR rebuild on GPU | ACTIVE |
| csr_scatter | scatter nonzero indices into CSR structure on GPU | ACTIVE |
| conv1d_bwd_input_gate | conv1d transpose + gate backward | CREATED (disabled, overhead issue) |
| elementwise_gates_backward | legacy, has t=0 boundary bug | DISABLED |

### VRAM Usage
| Batch | VRAM | Notes |
|-------|------|-------|
| 1     | 271 MB (2.3%) | — |
| 4     | 902 MB (7.6%) | forward: ZERO downloads |
| 8     | 1303 MB (10.9%) | — |
| 16    | 1774 MB (14.9%) | forward: ZERO downloads (all GPU paths active) |

### Remaining CPU Bottlenecks
1. MTP forward scan + loss (sequential recurrence, 1.4ms — not worth GPU-ifying)
2. Embedding backward d_embed_E (1B FLOPs, needs scatter atomics)
3. Direct-B backward (small)
4. GP-KL Welford update (small)
5. Downloads still needed for CPU backward ops (~2 per layer; d_residual eliminated)

### GPU Optimizer Status (FINAL)

All 13 param groups GPU-resident. CPU optimizer fully skipped.

| Param Group | Optimizer | Status |
|-------------|-----------|--------|
| A | dense Adam | ACTIVE |
| B | dense Adam | ACTIVE |
| C | dense Adam | ACTIVE |
| b_A | dense Adam | ACTIVE |
| conv_w | dense Adam | ACTIVE |
| W_A | dense Adam | ACTIVE |
| norm_w | dense Adam | ACTIVE |
| norm_b | dense Adam | ACTIVE |
| embed_E | factored ternary Adam | ACTIVE |
| embed_P | factored ternary Adam | ACTIVE |
| W_in | factored ternary Adam | ACTIVE |
| W_out | factored ternary Adam | ACTIVE |
| lm_head | factored ternary Adam | ACTIVE |

### Correctness Audit (2026-03-30)

| Component | Status | Details |
|-----------|--------|---------|
| Bilinear recurrence | VERIFIED | Algebraic equivalence confirmed |
| PC forward/backward | VERIFIED | 7/7 checks passed |
| MTP | FIXED | 3 bugs: w2 scaling, UV backprop, A clamp |
| Welford orth_sum | FIXED | Was silently disabled — now active |
| Repack | FIXED | Off-by-one in INT2 repack loop |
| Gradient accumulation | FIXED | Incorrect accumulation across micro-steps |
| Shared scratch buffer | FIXED | Aliasing bug between concurrent shader dispatches |
| Norm gradient accumulation | FIXED | Norm grad not accumulated correctly over batch |
| VK_MAX_BUFFERS | FIXED | Overflow — limit raised / binding slots freed |
| Sampled CE | VERIFIED | 0.8% deviation from CPU reference |
| B* init | VERIFIED | Correct for diagonal SSM architecture |

### Pipeline Summary (FINAL)

| Phase | Downloads | Notes |
|-------|-----------|-------|
| Forward | ZERO | All activations GPU-resident |
| Backward | Minimal | ~2 per layer (CPU ops only); d_residual eliminated (-96 MB/step) |
| Optimizer | ZERO | All 13 groups GPU-resident; CPU optimizer fully skipped |
| CSR rebuild | GPU | mvue_resample → csr_count → csr_scatter, no CPU roundtrip |
| Checkpoint save | GPU→CPU sync | Explicit sync before file write |

### Run-018 Validation

- **Status:** IN PROGRESS
- **Config:** 600 steps, batch=4
- **Goal:** Confirm 1249 tok/s stable over full run; validate loss convergence at final GPU pipeline state

### Prior Art Survey (2026-03-29)
- Slender-Mamba (COLING 2025) and BitMamba-2 precede as ternary SSMs
- NOVEL: MVUE gradient, SSGC optimizer, self-gated credit truncation, tight frame init
- Full survey: .workspace/research/r354-prior-art-survey.md

### Build
```
cd src/model_v12
make USE_VULKAN=1        # GPU build
make generate            # inference-only build
```

### Config
```
V12_N_LAYERS=6, V12_D_MODEL=1024, V12_D_EMBED=128, V12_D_STATE=128
V12_VOCAB_SIZE=32064, V12_SEQ_LEN=1024, V12_BATCH_SIZE=4
V12_GRAD_ACCUM_STEPS=4, V12_LR_MAX=0.0003, V12_LR_TOTAL_STEPS=18000
```
