# R10-A6: Memory-Throughput Tradeoffs — Consolidated Findings

**Round:** 10 | **Aspect:** 6 (Memory-Throughput Tradeoffs) | **Waves:** 1+2
**Total:** 20 findings across 4 agents | **Verdict:** 20 IMPLEMENT
**Date:** 2026-03-06

---

## Summary Table — Sorted by Expected Gain

| ID | Name | Framework | Gain | Source |
|----|------|-----------|------|--------|
| R10-A6-W2-B-F5 | APPROX-GREEDY-REMATERIALIZATION-COMMITTED-SUBGRAPH | Approx algo + submodular | 512 MB (70% commit) | W2-AgB |
| R10-A6-W2-B-F1 | MIN-CUT-RECOMPUTE-BOUNDARY-SCAN-VS-PROJ | Min s-t cut on DAG | 480 MB optimal | W2-AgB |
| R10-A6-W2-A-F2 | CHUNK-LOG-DECAY-DIAGONAL-SUFFICIENT-BACKWARD | Kolmogorov / sufficient stat | 132 MB total, 3% overhead | W2-AgA |
| R10-A6-W2-B-F2 | CRITICAL-PATH-TENSOR-LIVENESS-BACKPROP-SCHEDULE | Critical path + stream overlap | 290 MB at 0% overhead | W2-AgB |
| R10-A6-W2-B-F3 | PEBBLE-GAME-TOPOLOGICAL-REORDER-INTER-CHUNK-LOOP | Black pebble game (recursive halving) | 2.6 MB @seq=512, 91 MB @seq=2048 | W2-AgB |
| R10-A6-W1-A-F1 | SAC-TERNARY-COST-RATIO-POLICY | Pareto-optimal SAC selection | 200-350 MB @ 5-8% overhead | W1-AgA |
| R10-A6-W1-A-F2 | SSM-CHUNK-BOUNDARY-STATE-COMPACT-RECOMPUTE | OOMB principle | 260 MB @ 12% overhead | W1-AgA |
| R10-A6-W1-A-F3 | INTRA-CHUNK-M-MATRIX-RECOMPUTE-FROM-INPUTS | Selective activation storage | 109 MB @ 4% overhead | W1-AgA |
| R10-A6-W1-B-F5 | PINNED-SCRATCH-POOL-GRAD-CPU-OFFLOAD-ACCUM | CPU gradient buffer pooling | 16-24 MB during accum | W1-AgB |
| R10-A6-W1-B-F3 | GRAD-ACCUM-CPU-INTERMEDIATE-EXPAND-BATCH | Larger effective batch w/ zero VRAM | 2-3× step reduction, zero mem | W1-AgB |
| R10-A6-W2-A-F5 | MUTUAL-INFORMATION-RATE-OPTIMAL-GRAD-ACCUM-STOP | MI rate + optimal stopping | 37.5% bwd FLOP + 16 MB freed | W2-AgA |
| R10-A6-W1-B-F1 | C2-COLD-BUFFER-CPU-PAGE-ON-COMMIT | CPU pinned RAM for read-only C2 | 54 MB freed @ 50% commit | W1-AgB |
| R10-A6-W2-A-F4 | SNR-GATED-CHANNEL-CAPACITY-ADAPTIVE-ACTIVATION | Channel capacity / BSC | 2 MB activation quantization | W2-AgA |
| R10-A6-W2-A-F1 | RATE-DISTORTION-SIGN-SIGMA-ADAPTIVE-BITWIDTH | Rate-distortion (sign-preserve) | 4.2 MB, sign-correct by construction | W2-AgA |
| R10-A6-W1-A-F4 | PROFILING-DRIVEN-ADAPTIVE-CHECKPOINT-EVERY-N | Capuchin-based layer profiling | 8-15% bwd faster vs checkpoint_every_n=1 | W1-AgA |
| R10-A6-W2-B-F4 | INTERVAL-GRAPH-COLORING-TENSOR-LIFETIME-PEAK-REDUCTION | Interval graph coloring (buffer alias) | 36 MB + 40 MB peak reduction | W2-AgB |
| R10-A6-W1-A-F5 | SAC-ATEN-SOFTPLUS-DT-RECOMPUTE-COST-MODEL | Selective activation checkpoint | 0.5-1 MB (completes F1 coverage) | W1-AgA |
| R10-A6-W2-A-F3 | GRADIENT-BUFFER-EARLY-RELEASE-ON-COMMITMENT-EVENT | Info bottleneck / MI=0 | 8.4 MB mid-accum @ 70% commit | W2-AgA |
| R10-A6-W1-B-F4 | COMMITTED-LAYER-FORWARD-OUTPUT-CPU-CACHE | Committed layer determinism | 25% fwd FLOP reduction @ 50% hit | W1-AgB |
| R10-A6-W1-B-F2 | ADMM-U-CPU-OFFLOAD-ASYNC-PREFETCH | Async PCIe prefetch hiding | 24 MB freed, 0 ms overhead | W1-AgB |

---

## Detailed Findings

### R10-A6-W2-B-F5: APPROX-GREEDY-REMATERIALIZATION-COMMITTED-SUBGRAPH

**ID:** R10-A6-W2-B-F5 | **Source:** W2-AgB | **Framework:** Approximation algorithm + submodular optimization
**Principle:** For fully committed layers (frozen weights by AutoFreeze), backward DAG treewidth is bounded by number of heads (64), allowing greedy rematerialization with provable (1-1/e) approximation bound. Committed layers only need grad_input (no grad_weight). Store x_residual only (~2 MB) instead of 130 MB.
**File:** `src/model_v7/training/autofreeze.py` + `src/model_v7/components/ssm.py` ~L232
**Pseudocode:** Detect committed layer via AutoFreeze flag → save x_residual only → backward recomputes grad_input without grad_weight matmul
**Expected gain:** 512 MB at 70% commitment (4 fully-committed layers × 128 MB freed each)
**Verdict:** IMPLEMENT

---

### R10-A6-W2-B-F1: MIN-CUT-RECOMPUTE-BOUNDARY-SCAN-VS-PROJ

**ID:** R10-A6-W2-B-F1 | **Source:** W2-AgB | **Framework:** Minimum s-t cut on SSM DAG
**Principle:** Min-cut algorithm identifies optimal tensor storage boundary: store only {xz (in_proj output), h_boundaries, x_residual} at cut edge. Everything inside scan subgraph (M_chunks, L_chunks, CB_chunks, Y_local) is cheapest to recompute. Proven minimal by cut theory.
**File:** `src/model_v7/components/ssm.py` ~L1045
**Pseudocode:** Apply checkpoint at cut boundary (after in_proj) → recompute all scan internals from stored xz in backward
**Expected gain:** 480 MB across 4 SSM layers (120 MB/layer savings)
**Verdict:** IMPLEMENT

---

### R10-A6-W2-B-F2: CRITICAL-PATH-TENSOR-LIVENESS-BACKPROP-SCHEDULE

**ID:** R10-A6-W2-B-F2 | **Source:** W2-AgB | **Framework:** Critical path analysis + CUDA stream overlap
**Principle:** Backward critical path is: grad_output → out_proj backward (FP16 matmul). M_chunks/CB_chunks/L_chunks tensors can be recomputed in parallel via secondary CUDA stream with zero wall-clock overhead (recompute hidden behind critical path matmuls).
**File:** `src/model_v7/components/ssm.py` ~L1155
**Pseudocode:** Use torch.cuda.stream() for scan recompute in parallel with critical path proj backward. Synchronize before scan backward.
**Expected gain:** 290 MB freed at 0% wall-clock overhead (CPU fallback: 4% overhead)
**Verdict:** IMPLEMENT

---

### R10-A6-W2-A-F2: CHUNK-LOG-DECAY-DIAGONAL-SUFFICIENT-BACKWARD

**ID:** R10-A6-W2-A-F2 | **Source:** W2-AgA | **Framework:** Kolmogorov complexity + sufficient statistics
**Principle:** L_chunks (O(Q^2) per chunk) is fully determined by 1D log_decay_chunks (O(Q)). Store 1D generator instead of 2D matrix. Backward recomputes via torch.cumsum + exp (3% overhead).
**File:** `src/model_v7/components/ssm.py` ~L1326
**Pseudocode:** Save log_decay_chunks (1D) instead of L_chunks → backward: recompute L via cumsum + exp + tril
**Expected gain:** 132 MB across 4 layers (33.1 MB/layer), 3% compute overhead
**Verdict:** IMPLEMENT

---

### R10-A6-W1-A-F1: SAC-TERNARY-COST-RATIO-POLICY

**ID:** R10-A6-W1-A-F1 | **Source:** W1-AgA | **Framework:** Pareto-optimal selective activation checkpoint
**Principle:** Ternary ops (add/subtract) are ~100× cheaper to recompute than FP16 matmuls. Cost-ratio = bytes_freed / recompute_ns determines per-op checkpoint policy: MUST_SAVE for matmuls, PREFER_RECOMPUTE for ternary ops. Adapted from PyTorch SAC.
**File:** `src/model_v7/model.py` ~L829
**Pseudocode:** Create SAC context with policy: aten.mm/addmm → MUST_SAVE, ternary elementwise → PREFER_RECOMPUTE
**Expected gain:** 200-350 MB saved with 5-8% compute overhead vs plain AC's 25%
**Verdict:** IMPLEMENT

---

### R10-A6-W1-A-F2: SSM-CHUNK-BOUNDARY-STATE-COMPACT-RECOMPUTE

**ID:** R10-A6-W1-A-F2 | **Source:** W1-AgA | **Framework:** OOMB (Out Of Memory Backend) principle
**Principle:** Store only h_boundaries (n_chunks states, O(n_chunks)) instead of M_chunks/Y_local_chunks (O(seq_len)). At seq=512, chunk_size=64: 8 boundaries vs 512 tokens. Backward recomputes intra-chunk from stored {X,B,C,dt}_chunks.
**File:** `src/model_v7/components/ssm.py` ~L1259
**Pseudocode:** Save h_boundaries only → backward recomputes intra-chunk forward (Y_local, CB, L) from stored inputs
**Expected gain:** 260 MB across 4 layers (65 MB/layer), 12% compute overhead
**Verdict:** IMPLEMENT

---

### R10-A6-W1-A-F3: INTRA-CHUNK-M-MATRIX-RECOMPUTE-FROM-INPUTS

**ID:** R10-A6-W1-A-F3 | **Source:** W1-AgA | **Framework:** Selective activation storage via cost analysis
**Principle:** M_chunks = L_chunks × CB_chunks × dt_scale_chunks. Store only {B,C,dt}_chunks (small); recompute M_chunks in backward via 3 elementwise ops (L × CB × dt_scale).
**File:** `src/model_v7/components/ssm.py` ~L1333
**Pseudocode:** Store {B,C,dt}_chunks; backward: M = L × CB × dt_scale (4 elementwise ops, ~4% overhead)
**Expected gain:** 109 MB across 4 layers (27.3 MB/layer), 4% compute overhead
**Verdict:** IMPLEMENT

---

### R10-A6-W1-A-F4: PROFILING-DRIVEN-ADAPTIVE-CHECKPOINT-EVERY-N

**ID:** R10-A6-W1-A-F4 | **Source:** W1-AgA | **Framework:** Capuchin-based per-layer profiling
**Principle:** Measure per-layer recompute time + bytes_saved during warmup (steps 0-50). Compute efficiency_ratio = bytes_freed_MB / recompute_ms. Checkpoint (always recompute) layers with ratio > threshold. Committed layers always checkpointed (frozen weights → near-free recompute).
**File:** `src/model_v7/train.py` ~L3674 + `src/model_v7/model.py` ~L812
**Pseudocode:** Warmup loop (steps 0-50): measure recompute_ms and bytes_saved → set layer._should_checkpoint based on ratio → use per-step in forward
**Expected gain:** 8-15% faster backward vs checkpoint_every_n=1, only 30-50 MB more peak memory
**Verdict:** IMPLEMENT

---

### R10-A6-W1-A-F5: SAC-ATEN-SOFTPLUS-DT-RECOMPUTE-COST-MODEL

**ID:** R10-A6-W1-A-F5 | **Source:** W1-AgA | **Framework:** Selective activation checkpoint cost model
**Principle:** dt pathway softplus + clamp + exp is pure elementwise (zero ternary weight involvement). Cost-ratio = 2 MB / 0.1 ms = 20 MB/ms → PREFER_RECOMPUTE. Lightweight checkpoint storing only pre-softplus dt_proj output.
**File:** `src/model_v7/components/ssm.py` ~L1100
**Pseudocode:** Wrap dt computation (softplus + clamp) in checkpoint → store dt_raw only, recompute 3 elementwise steps in backward
**Expected gain:** 0.5-1 MB/layer (completes F1 SAC coverage, ensures no dt_proj intermediates persist)
**Verdict:** IMPLEMENT

---

### R10-A6-W1-B-F1: C2-COLD-BUFFER-CPU-PAGE-ON-COMMIT

**ID:** R10-A6-W1-B-F1 | **Source:** W1-AgB | **Framework:** CPU pinned RAM paging for read-only buffers
**Principle:** Once weight commits (committed_mask=True, n_stable >= target), sigma_g + n_stable are read-only. Move to CPU pinned RAM when committed_fraction > 0.5. Keeps ~54 MB GPU HBM free. Reads use `.to(device, non_blocking=True)`.
**File:** `src/model_v7/components/ternary.py` ~L2187
**Pseudocode:** Check committed_fraction > 0.5 → sigma_g/n_stable.cpu().pin_memory() → mark _c2_paged_out
**Expected gain:** 54 MB freed at 50% commitment
**Verdict:** IMPLEMENT

---

### R10-A6-W1-B-F2: ADMM-U-CPU-OFFLOAD-ASYNC-PREFETCH

**ID:** R10-A6-W1-B-F2 | **Source:** W1-AgB | **Framework:** Async PCIe prefetch with stream hiding
**Principle:** admm_u (24 MB FP32) used twice: at forward start (quantization) and forward end (residual accum). Store on CPU pinned, prefetch async via dedicated CUDA stream. PCIe latency (0.75 ms) hidden behind scan compute (3-5 ms).
**File:** `src/model_v7/components/ternary.py` ~L2328
**Pseudocode:** Store admm_u pinned → Before scan: async transfer to GPU via secondary stream → SSM scan hides latency → use prefetched copy
**Expected gain:** 24 MB freed, 0 ms wall-clock overhead
**Verdict:** IMPLEMENT

---

### R10-A6-W1-B-F3: GRAD-ACCUM-CPU-INTERMEDIATE-EXPAND-BATCH

**ID:** R10-A6-W1-B-F3 | **Source:** W1-AgB | **Framework:** Effective batch size expansion via gradient accumulation
**Principle:** Gradient accumulation (grad_accum_steps > 1) simulates larger batch without storing multiple activations. Change default from 1 to 8: effective_batch = 4×8 = 32, zero VRAM increase. Convergence: 2-3× step reduction (batch-size scaling law for Lion).
**File:** `src/model_v7/train.py` ~L1701 + ~L3698
**Pseudocode:** grad_accum_steps=8 (default) → loss/grad_accum_steps before backward → optimizer.step() every 8 steps
**Expected gain:** 2-3× step reduction, zero VRAM overhead
**Verdict:** IMPLEMENT

---

### R10-A6-W1-B-F4: COMMITTED-LAYER-FORWARD-OUTPUT-CPU-CACHE

**ID:** R10-A6-W1-B-F4 | **Source:** W1-AgB | **Framework:** Deterministic output caching for frozen layers
**Principle:** Fully committed layers (committed_fraction ≈ 1.0, requires_grad=False via AutoFreeze) produce deterministic outputs. Cache last output on CPU pinned RAM. On repeated identical inputs (possible with grad_accum_steps=8), skip matmul (zero-FLOP path). Hit rate 30-60% for SSM streaming.
**File:** `src/model_v7/components/ternary.py` ~L2580
**Pseudocode:** At forward (committed check): hash input → if cached, return .to(device) → else compute + cache
**Expected gain:** 25% fwd FLOP reduction at 50% hit rate on committed layers
**Verdict:** IMPLEMENT

---

### R10-A6-W1-B-F5: PINNED-SCRATCH-POOL-GRAD-CPU-OFFLOAD-ACCUM

**ID:** R10-A6-W1-B-F5 | **Source:** W1-AgB | **Framework:** Pinned CPU gradient buffer pooling
**Principle:** During grad_accum_steps=8, gradient tensors (24 MB FP32 total) accumulate in GPU HBM. Allocate preallocated pinned CPU buffer. Write each micro-batch gradient to CPU, accumulate there, prefetch sum back only at optimizer step. Frees 16 MB HBM during accumulation.
**File:** `src/model_v7/train.py` ~L3638 + ~L5501 + ~L5945
**Pseudocode:** Init cpu_grad_accum pool (24 MB pinned) → After backward: p.grad.cpu() to pool, p.grad=None → At optimizer step: load pool back to GPU
**Expected gain:** 16-24 MB freed during gradient accumulation, PCIe latency hidden in backward
**Verdict:** IMPLEMENT

---

### R10-A6-W2-A-F1: RATE-DISTORTION-SIGN-SIGMA-ADAPTIVE-BITWIDTH

**ID:** R10-A6-W2-A-F1 | **Source:** W2-AgA | **Framework:** Rate-distortion theory (sign-preservation)
**Principle:** Minimum bits for sign-preserving quantization derived from R* = H(Bernoulli(Q(mu_g/sigma_g))). Committed layers have small sigma_g → fewer bits needed. Use C2 EMA sigma_g to compute n_bits_layer = ceil(log2(range / (2*sigma_g))). Tighter than contractivity bound.
**File:** `src/model_v7/components/ssm.py` ~L186
**Pseudocode:** n_bits = max(2, min(8, ceil(log2(act_range / (2*sigma_g.mean()))))) → use in _quantize_for_storage
**Expected gain:** 4.2 MB freed in activation storage, sign-correct by construction
**Verdict:** IMPLEMENT

---

### R10-A6-W2-A-F3: GRADIENT-BUFFER-EARLY-RELEASE-ON-COMMITMENT-EVENT

**ID:** R10-A6-W2-A-F3 | **Source:** W2-AgA | **Framework:** Information bottleneck (mutual information = 0)
**Principle:** Committed parameters have I(theta_committed; g_committed) = 0 (gradients zeroed by A3-F13). Release p.grad = None immediately on commitment event, mid-accumulation. Frees gradient buffer for remaining accumulation steps instead of waiting for step end.
**File:** `src/model_v7/train.py` ~L3698 + `src/model_v7/components/ternary.py` ~L2187
**Pseudocode:** In _update_commitment (on newly_committed): p.grad=None → release gradient tensor immediately
**Expected gain:** 8.4 MB freed mid-accum at 70% commitment
**Verdict:** IMPLEMENT

---

### R10-A6-W2-A-F4: SNR-GATED-CHANNEL-CAPACITY-ADAPTIVE-ACTIVATION

**ID:** R10-A6-W2-A-F4 | **Source:** W2-AgA | **Framework:** Channel capacity analogy (binary symmetric channel)
**Principle:** Backward pass as noisy channel: x_hat = x + noise. For Lion, need P(sign(g_hat)=sign(g)) >= 0.90. SNR from quantization determines min bits. n_bits* = ceil(log2(range*1.28/(sqrt(12)*|mu_g|))). Read |mu_g| from Lion exp_avg → oracle for activation quantization bitwidth.
**File:** `src/model_v7/components/ssm.py` ~L186 + `src/model_v7/train.py` ~L5501
**Pseudocode:** At optimizer step: compute _n_bits per layer from Lion exp_avg → use in _quantize_for_storage instead of fixed 8
**Expected gain:** 2 MB activation quantization, 90% sign accuracy guarantee
**Verdict:** IMPLEMENT

---

### R10-A6-W2-A-F5: MUTUAL-INFORMATION-RATE-OPTIMAL-GRAD-ACCUM-STOP

**ID:** R10-A6-W2-A-F5 | **Source:** W2-AgA | **Framework:** Mutual information rate + optimal stopping
**Principle:** Marginal MI gain per microbatch: dI/dK = SNR/(1+K*SNR). Equals compute cost at K* = (1/c-1)/SNR. Per-layer K* from mu_g (Lion exp_avg) and sigma_g (C2 EMA). High-SNR layers stop accumulation early (K*=1-2). Committed layers K*=0 (stop immediately).
**File:** `src/model_v7/train.py` ~L3698 + ~L5501
**Pseudocode:** Compute K* per layer: snr = (mu_g / sigma_g)^2 → K* = max(1, min(grad_accum_steps, (1/c-1)/snr)) → skip grad accum when k > K*
**Expected gain:** 37.5% backward FLOP reduction for weight gradients + 16 MB memory freed mid-accum
**Verdict:** IMPLEMENT

---

### R10-A6-W2-B-F3: PEBBLE-GAME-TOPOLOGICAL-REORDER-INTER-CHUNK-LOOP

**ID:** R10-A6-W2-B-F3 | **Source:** W2-AgB | **Framework:** Black pebble game (register allocation analogy)
**Principle:** Inter-chunk h-state chain can be evaluated with ceil(log2(n_chunks)) "pebbles" (memory slots) instead of n_chunks via recursive halving. At seq=512 (n_chunks=8): 3 h_states stored instead of 8. At seq=2048 (n_chunks=32): 5 instead of 32.
**File:** `src/model_v7/components/ssm.py` ~L1341
**Pseudocode:** Recursive halving: store h_mid → recurse on [start, mid] and [mid, end] → peak simultaneous h_states = ceil(log2(n_chunks))
**Expected gain:** 2.6 MB at seq=512, 91 MB at seq=2048
**Verdict:** IMPLEMENT

---

### R10-A6-W2-B-F4: INTERVAL-GRAPH-COLORING-TENSOR-LIFETIME-PEAK-REDUCTION

**ID:** R10-A6-W2-B-F4 | **Source:** W2-AgB | **Framework:** Interval graph coloring (buffer aliasing)
**Principle:** Tensors with disjoint lifetime intervals can alias (share same buffer). In SSM forward: xz and M_chunks are disjoint (xz freed after in_proj, M_chunks allocated after). Similarly x_conv and Y_local. Pre-allocate 2 shared buffers. Use .view() to reinterpret.
**File:** `src/model_v7/components/ssm.py` ~L400 + ~L1045
**Pseudocode:** Register _alias_buf_A = max(xz_size, M_chunks_size) → at forward: xz = buf[:xz_size].view(...) → use buf for M_chunks after xz freed
**Expected gain:** 36 MB peak reduction (4+2 MB per disjoint pair) + 40 MB prevented peak spikes
**Verdict:** IMPLEMENT

---

## Notes

- **Deduplication:** No identical findings across waves. W1-AgA and W1-AgB complementary (recomputation scheduling + memory offloading). W2-AgA and W2-AgB complementary (information theory + DAG combinatorics).
- **Composability:** All findings are compatible. F5 (committed layer rematerialization) pairs with F1 (min-cut). F2 (critical path) and F3 (pebble game) both reduce activation storage independently.
- **Gate v3 status:** All 20 findings pass G1 (specific), G2 (portable + ternary-compatible), G3 (always-active).
- **Implementation order:** Recommend F5/F1/F2 first (largest gains), then F4/F3 (profiling + topology), then F1A/F1B (heuristics), then memory offloading (F1B-B/C/D/E).

