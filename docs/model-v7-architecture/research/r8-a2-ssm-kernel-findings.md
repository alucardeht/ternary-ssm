# R8-A2: SSM Kernel Fusion Findings — Complete Synthesis

**Research Round:** R8 (Advanced kernel optimization + algorithm restructuring)
**Aspect:** A2 (SSM Kernel Fusion)
**Synthesis Date:** 2026-03-06
**Waves:** W1 (kernel fusion 5x2 findings) + W2 (paradigm breakers 5x2 findings)

---

## Summary: 20 Findings — 15 IMPLEMENT, 4 EXPLORE, 1 SKIP

**Key insight:** SSM bottlenecks fall into three categories:
1. **Kernel-level fusion** (W1-AgA): Fuse existing separate ops into Triton kernels
2. **Training backward optimization** (W1-AgB): Eliminate or compress intermediate states
3. **Algorithm restructuring** (W2): Break sequential dependencies via mathematical insight

**Implementation strategy:** Tier 1 findings are independent and can be parallelized. Tier 2 (EXPLORE) findings require empirical validation before scaling to production training.

---

## Tier 1 — IMPLEMENT (ready, high confidence)

### F1-K1: FLA-FUSED-NORM-GATE-IN-SSM
**Tier:** IMPLEMENT
**Principle:** The z-gate operation `silu(z) * y` is structurally identical to FLA's `FusedRMSNormGated` kernel. Fusing norm+gate avoids a separate kernel launch and one VRAM roundtrip for the activation tensor.

**Application:** `src/model_v7/components/ssm.py` — `SelectiveSSMSSD.forward()` lines 1176-1178

**Pseudocode:**
```python
# In __init__: self.out_gate = FusedRMSNormGated(d_inner, activation='silu')
# In forward:  act_z = self.out_gate(y, z); output = self.out_proj(act_z)
# Replaces: act_z = self.act_fn(z); act_z = act_z * y
```

**Expected gain:** 3-7% end-to-end throughput (kernel launch overhead at small batch sizes)
**Dependencies:** None
**Synergies:** Works alongside F2, F3 (other kernel fusion targets)

---

### F2-K2: SCAN-BF16-STATES-REMOVE-FLOAT-UPCAST
**Tier:** IMPLEMENT
**Principle:** FLA's `chunk_simple_gla` is called with explicit `.float()` upcasts despite BF16 autocast being active. IBM's study shows FP16 states give 16% speedup over FP32 with <0.3% output mismatch. Remove explicit upcasts to let BF16 flow through scan.

**Application:** `src/model_v7/components/ssm.py` — `SelectiveSSMSSD._ssd_scan_fla()` line 1252

**Pseudocode:**
```python
# Remove .float() casts:
# o, _ = chunk_simple_gla(q, k, v, g, scale=1.0, output_final_state=False)
# Add verification: print(f"[BF16-SCAN] dtype={q.dtype}")
```

**Expected gain:** 4-8% end-to-end throughput
**Dependencies:** None
**Synergies:** Orthogonal to F1; both reduce kernel overhead

---

### F3-K3: DEPTHWISE-CONV1D-PRELOAD-IN-SCAN-CHUNK
**Tier:** IMPLEMENT
**Principle:** Depthwise conv1d before SSD scan is a separate PyTorch op requiring full VRAM roundtrip. IBM flagged conv+norm as "appealing candidates for next fusion." Preload d_conv boundary tokens at chunk start; compute conv1d intra-chunk, eliminating separate kernel launch.

**Application:** `src/model_v7/components/ssm.py` — `SelectiveSSMSSD.forward()` lines 1065-1067 and `_ssd_scan_fla()` refactor

**Pseudocode:**
```python
# For each chunk i: boundary = x_inner[:, max(0, i*C-d_conv):i*C, :]
# causal_conv_chunk = F.conv1d(boundary.T, conv_weight, padding=0).T[:, -C:, :]
# Pass causal_conv_chunk as x to chunk_simple_gla (replaces pre-computed x_conv)
```

**Expected gain:** 2-5% end-to-end throughput (eliminates 4 kernel launches per 4 layers)
**Dependencies:** Requires chunk_size configuration (F4)
**Synergies:** Combines with F4 for optimal chunk size

---

### F4-K4: FLA-CHUNK-SIZE-128-OPTIMAL-FOR-FUSED-SCAN
**Tier:** IMPLEMENT
**Principle:** IBM's fused kernel study identified chunk_size=128 as optimal (down from 256 unfused) due to reduced register pressure and better cache utilization. v7 currently uses 64 (conservative). Update default to 128.

**Application:** `src/model_v7/components/ssm.py` — `SelectiveSSMSSD.__init__()` line 480 + `_adaptive_chunk_size()` fallback

**Pseudocode:**
```python
# In __init__: self.chunk_size = 128  (instead of 64)
# Verify: print(f"[CHUNK-SIZE] chunk_size={self.chunk_size} at step=0")
```

**Expected gain:** 2-6% end-to-end throughput (5-15% on scan portion)
**Dependencies:** None (backward compatible)
**Synergies:** Pairs with F3 (conv1d preload benefits from larger chunks)

---

### F5-K5: FLA-CHUNK-SIMPLE-GLA-REPLACE-WITH-DIRECT-MAMBA2-KERNEL
**Tier:** IMPLEMENT
**Principle:** v7 uses `chunk_simple_gla` (Simple GLA kernel) as a proxy for SSD scan via variable substitution. FLA provides `mamba_chunk_scan_combined` (direct SSD kernel), eliminating proxy reshapes and dtype inconsistencies.

**Application:** `src/model_v7/components/ssm.py` — `SelectiveSSMSSD._ssd_scan_fla()` lines 1222-1258

**Pseudocode:**
```python
# from fla.ops.mamba2.chunk import mamba_chunk_scan_combined  (try/except for fallback)
# y = mamba_chunk_scan_combined(x, dt, A, B, C, D=None, z=None, chunk_size=128)
# Fallback to chunk_simple_gla proxy if import fails
```

**Expected gain:** 2-5% end-to-end throughput (5-10% on scan from removing proxy overhead)
**Dependencies:** FLA library (already required)
**Synergies:** Directly benefits from F2 (BF16) and F4 (chunk_size)

---

### F6-B1: FLA-FUSED-RECURRENT-BACKWARD
**Tier:** IMPLEMENT
**Principle:** Replace `chunk_simple_gla` with `fused_recurrent_simple_gla`. Avoids materializing O(T*N) intermediate hidden states by recomputing them during backward. Direct SSM analog of Flash Attention's recomputation.

**Application:** `src/model_v7/components/ssm.py` — `SelectiveSSMSSD._ssd_scan_fla()` line 1252 (kernel function swap)

**Pseudocode:**
```python
# from fla.ops.simple_gla import fused_recurrent_simple_gla
# o, _ = fused_recurrent_simple_gla(q.float(), k.float(), v.float(), g.float(),
#                                    scale=1.0, output_final_state=False)
```

**Expected gain:**
- Memory: 15% VRAM reduction in backward (enables batch=6 from batch=4)
- Throughput: 10-25% from reduced memory bandwidth; conservative 15% per backward pass

**Dependencies:** FLA version >= 0.7 (check for `fused_recurrent_simple_gla` availability)
**Synergies:** Pairs with F1-F5 (all operate on same scan path); mutually exclusive with F7 but compatible via try/except
**Note:** If unavailable, falls back to `chunk_simple_gla` — zero overhead

---

### F7-B2: REVERSE-SCAN-FUSED-TRITON-BACKWARD
**Tier:** IMPLEMENT
**Principle:** The backward of linear recurrence h_t = A*h_{t-1} + B*x_t is itself a reverse linear recurrence on dh. Fused Triton backward kernel computes dh_t via reverse scan, keeping h_t only in registers. Eliminates storing full h trajectory.

**Application:** `src/model_v7/kernels/triton/fused_ssm_recurrence.py` — add `_fused_ssm_backward_kernel` alongside forward, wrap in `autograd.Function`

**Pseudocode:**
```python
@triton.jit
def _fused_ssm_backward_kernel(dh_ptr, d_inputs_ptr, ...):
    # Scan BACKWARDS: dh_{t-1} = A*dh_t; dk_t = dh_t * v_t; dv_t = dh_t * k_t
    for t in reversed(range(T)):
        dh = tl.load(dh_ptr + t*...) + C_t * d_output_t
        tl.store(d_inputs_ptr + t*..., dh_prev(dh, dA))
        dh = dA * dh  # reverse recurrence
```

**Expected gain:** 5-12% backward throughput (50MB VRAM saved per seq_len=512 for 6M model)
**Dependencies:** Requires custom Triton kernel (medium complexity); fallback to autograd.Function without custom kernel
**Synergies:** Orthogonal to F6 (targets different backward path: recurrent vs chunked)

---

### F8-B3: CHUNKED-BOUNDARY-CHECKPOINT-SQRT-T
**Tier:** IMPLEMENT
**Principle:** Current `QuantizedCheckpointSSDScan` quantizes all inputs. More efficient: save only O(sqrt(T)) inter-chunk boundary hidden states h_{k*chunk_size}. Recompute intra-chunk h_t during backward from boundary. Storage: O(sqrt(T)*N) instead of O(T*N).

**Application:** `src/model_v7/components/ssm.py` — replace outer `QuantizedCheckpointSSDScan` strategy in backward

**Pseudocode:**
```python
# Forward: save boundary states every sqrt(T) steps
boundaries = []
for chunk in range(0, T, chunk_size):
    h_boundary = h.clone()  # save at chunk boundaries only
    h = run_chunk_forward(h, inputs[chunk:chunk+chunk_size])
    boundaries.append(h_boundary)
# Backward: recompute each chunk from its boundary
for chunk in reversed(range(n_chunks)):
    h = boundaries[chunk]
    h_traj = recompute_chunk(h, inputs[chunk:chunk+chunk_size])  # local only
    d_chunk = backward_through_chunk(h_traj, d_output[chunk:chunk+chunk_size])
```

**Expected gain:**
- Memory: 50-70% activation memory reduction (262MB → 12MB per sample for 6M model)
- Throughput: enables batch=6-8 on 12GB (~1.5-2× more gradient signal per wall-clock minute)

**Dependencies:** None (pure PyTorch)
**Synergies:** Combines with F6 or F7 for complete backward memory optimization

---

### F9-B4: FUSED-LIGER-CE-FOR-SSM-OUTPUT
**Tier:** IMPLEMENT
**Principle:** Output projection (SSM hidden → vocab) is the largest VRAM allocation (logit tensor = batch*seq_len*vocab_size). Fused Linear+Cross-Entropy computes forward+backward in one Triton kernel, never materializing full logit tensor. Liger achieves 84% VRAM reduction + 20% throughput on output layer.

**Application:** `src/model_v7/train.py` — replace `nn.CrossEntropyLoss` with `LigerFusedLinearCrossEntropyLoss`

**Pseudocode:**
```python
# from liger_kernel.transformers import apply_liger_kernel_to_cross_entropy
# loss_fn = LigerFusedLinearCrossEntropyLoss()  # fuses output proj + CE in one kernel
# loss = loss_fn(hidden_states.view(-1, d_model), lm_head.weight, targets.view(-1))
# Replaces: loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
```

**Expected gain:**
- Memory: 15-25% total VRAM reduction (logits dominate backward memory, 206MB eliminated for batch=4)
- Throughput: 10-20% on output layer from fewer memory bandwidth round-trips

**Dependencies:** liger-kernel library (already used in train.py:1170)
**Synergies:** Orthogonal to F1-F8 (targets loss computation, not SSM scan)

---

### F10-M1: GATE-ABSORBED-INTO-C-PROJECTION
**Tier:** IMPLEMENT
**Principle:** Current: scan produces y_t, THEN gate: output = silu(z_t)*y_t. Mathematical identity: silu(z_t)*(C_t^T*h_t) = (silu(z_t)*C_t)^T*h_t. Gate absorbed into query projection, reduces 2 ops (silu + multiply) to 1 (scaled query before kernel).

**Application:** `src/model_v7/components/ssm.py` — `SelectiveSSMSSD._ssd_scan_fla()` lines 1241-1252 and `forward()` lines 1176-1177

**Pseudocode:**
```python
# In forward(), compute gate BEFORE scan call:
act_z = self.act_fn(z)  # shape (batch, seq_len, d_inner)
# In _ssd_scan_fla: fuse gate into q before kernel:
q = C[:, :, None, :].expand(...) * act_z_heads  # elementwise scale before kernel
o, _ = chunk_simple_gla(q.float(), k.float(), v.float(), g.float(), ...)
# Remove: act_z = self.act_fn(z); act_z = act_z * y  (old lines 1176-1177)
output = self.out_proj(o.reshape(...))
```

**Expected gain:** 2-4% end-to-end throughput (eliminates silu kernel + multiply, reduces 3 ops to 2)
**Dependencies:** None
**Synergies:** Complements F1 (both reduce kernel launches around SSM output)

---

### F11-M2: TSCS-A-ZERO-PARALLEL-PREFIX-SCAN
**Tier:** IMPLEMENT
**Principle:** For TSCS heads where A=0 (37.5% of n_heads default): decay_t = exp(0) = 1 for all t. This is pure cumulative sum, computable via parallel prefix (torch.cumsum) instead of sequential loop. The loop scales linearly with n_heads; removing 37.5% saves proportional time.

**Application:** `src/model_v7/components/ssm.py` — `SelectiveSSMSSD._ssd_scan_chunked()` lines 1341-1380 (add pre-pass)

**Pseudocode:**
```python
# At start of _ssd_scan_chunked, BEFORE the sequential loop:
zero_heads = (A == 0)  # bool mask, shape (n_heads,)
if zero_heads.any():
    # For A=0 heads: cumsum replaces sequential scan
    u_zero = dt_heads[:, :, zero_heads, None] * (B[:, :, None, :] * X_reshaped[:, zero_heads, :, :head_dim])
    h_zero = torch.cumsum(u_zero, dim=1)  # parallel prefix
    Y_zero = torch.einsum('bths,btn->bthd', h_zero, C[..., zero_heads.sum(), d_state])
    # Mask out A=0 heads in main sequential loop
```

**Expected gain:**
- For TSCS-enabled models: 10-20% reduction in _ssd_scan_chunked time
- End-to-end: 3-8% when TSCS is active (always-on per project config)

**Dependencies:** Requires TSCS enabled (already enabled per memory/MEMORY.md: "TSCS reset fraction = 37.5%")
**Synergies:** Orthogonal to F5 (direct Mamba2 kernel may obsolete chunked scan, but F11 applies to fallback path)

---

### F12-M3: DT-HEADS-MEAN-ELIMINATION-VIA-EXACT-PER-HEAD-DT
**Tier:** IMPLEMENT
**Principle:** Current dt aggregation uses hardcoded `.mean(-1)` to produce per-head scalar from head_dim values, losing information. Replace with learnable linear projection (head_dim → 1 per head). Adds negligible parameters (256 for n_heads=16) and enables fused einsum (mean + A-scale in single op).

**Application:** `src/model_v7/components/ssm.py` — `SelectiveSSMSSD.__init__()` + `_ssd_scan_fla()` line 1238

**Pseudocode:**
```python
# In __init__: (adds 256 learnable params for n_heads=16, head_dim=16)
self.dt_agg_weight = nn.Parameter(torch.ones(n_heads, head_dim) / head_dim)
# In _ssd_scan_fla and _ssd_scan_chunked, replace .mean(-1):
dt_reshaped = dt.view(batch, seq_len, n_heads, head_dim)
dt_heads = torch.einsum('bthd,hd->bth', dt_reshaped, self.dt_agg_weight)
g = A[None, None, :] * dt_heads
print(f"[DT-AGG] weight norm={self.dt_agg_weight.norm():.4f}")
```

**Expected gain:**
- Throughput: 1-2% from fused einsum
- Convergence: potential 5-15% reduction in steps to target (speculative, depends on information discarded by mean)

**Dependencies:** None (pure parameter addition)
**Synergies:** Works with any scan backend (F5 or F11)
**Note:** Convergence gain marked as exploratory, throughput gain is reliable

---

### F13-M4: SEMISEPARABLE-CB-VIA-COMMITTED-WEIGHT-PRECOMPUTATION
**Tier:** IMPLEMENT
**Principle:** C and B both come from ternary x_proj applied to same input. For committed weight rows (AutoFreeze), W_B and W_C are frozen. Precompute cross-projection kernel P = W_C^T*W_B (integer matrix) once commitment stabilizes. Use P for CB matrix instead of recomputing B,C from scratch each step.

**Application:** `src/model_v7/components/ssm.py` — `SelectiveSSMSSD._ssd_scan_chunked()` einsum line 1330 + register_buffer for P_committed

**Pseudocode:**
```python
# In __init__: (committed cross-projection matrix, updated when commitment changes)
self.register_buffer('P_committed', None)  # set externally by C2 tracker when mask stabilizes
# In _ssd_scan_chunked (before CB einsum):
if self.P_committed is not None:
    CB_fast = torch.einsum('bti,ij,bsj->bts', x_conv_chunk, self.P_committed, x_conv_chunk)
    # Replaces: CB_chunks = torch.einsum('bktn,bksn->bkts', C_chunks, B_chunks)
    print(f"[CB-PRECOMP] Using precomputed P_committed, shape={self.P_committed.shape}")
```

**Expected gain:** 3-8% gain in _ssd_scan_chunked after 500+ steps when commitment is meaningful (zero overhead at init)
**Dependencies:** C2 commitment tracking (already always-on)
**Synergies:** Scales with AutoFreeze progress; complements commitment-based optimizations

---

### F14-M5: AFFINE-MONOID-INTER-CHUNK-PARALLEL-REDUCE
**Tier:** IMPLEMENT
**Principle:** Inter-chunk sequential loop propagates h via affine updates h_k = a_k*h_{k-1} + b_k. These compose associatively: (a2,b2)∘(a1,b1) = (a2*a1, a2*b1+b2). Parallel prefix over (a_k,b_k) pairs computes all cumulative h_k in O(log n_chunks) steps instead of O(n_chunks) sequential. For n_chunks=8: 3 parallel rounds vs 8 iterations.

**Application:** `src/model_v7/components/ssm.py` — `SelectiveSSMSSD._ssd_scan_chunked()` replace sequential loop lines 1341-1380

**Pseudocode:**
```python
# 1. Precompute all (a_k, b_k) in parallel:
a_chunks = torch.exp(log_decay_chunks.sum(-1))  # (batch, n_chunks, n_heads)
b_chunks = _compute_all_chunk_updates_batched(B_chunks, C_chunks, X_chunks, dt_chunks, a_chunks)
# 2. Parallel prefix reduce over n_chunks using affine monoid:
h_starts = _affine_prefix_scan(a_chunks, b_chunks, h_init)  # O(log n_chunks) rounds
# 3. Compute all Y_chunks simultaneously:
Y = _apply_all_h_starts_batched(h_starts, Y_local_chunks, C_chunks, a_chunks)
print(f"[AFFINE-PREFIX] n_chunks={n_chunks}, rounds={math.ceil(math.log2(n_chunks))}")
```

**Expected gain:**
- For non-FLA fallback path (_ssd_scan_chunked): 5-15% speedup on loop (8→3 iterations for 8 chunks)
- End-to-end: 2-6% if chunked scan used; higher on CPU
- Benefit scales with n_chunks; most significant for longer sequences

**Dependencies:** None (pure PyTorch)
**Synergies:**
- Pairs with F11 (both optimize _ssd_scan_chunked loop; can be unified)
- Orthogonal to F5 (direct Mamba2 path bypasses chunked scan entirely)

---

### F15-T1: GRAD-SNR-ADAPTIVE-LAYER-SKIP
**Tier:** IMPLEMENT
**Principle:** Track gradient SNR per layer via EMA: SNR_l(t) = E[||grad_l||^2] / Var[||grad_l||]. When SNR_l < 1, variance dominates and updates are noise. Zero out grad for that layer (skip optimizer.step for it), but keep grad_input flowing (chain rule intact). Expected loss decrease under noise is negative; skipping achieves zero variance harm.

**Application:** `src/model_v7/train.py` — training_step() + per-layer gradient post-hook; `src/model_v7/components/ssm.py` — track per-layer SNR

**Pseudocode:**
```python
# In training_step, after loss.backward():
for layer_idx, layer in enumerate(model.blocks):
    g_norm = sum(p.grad.norm().item()**2 for p in layer.parameters() if p.grad is not None)**0.5
    grad_mu[layer_idx] = beta * grad_mu[layer_idx] + (1-beta) * g_norm
    grad_nu[layer_idx] = beta * grad_nu[layer_idx] + (1-beta) * g_norm**2
    snr = grad_mu[layer_idx]**2 / (grad_nu[layer_idx] - grad_mu[layer_idx]**2 + 1e-8)
    if snr < 1.0 and step > warmup:
        for p in layer.parameters():
            p.grad = None  # zero grad => optimizer.step() skips this layer
```

**Expected gain:** 5-10% optimizer.step() time reduction (scales with layer count and low-SNR fraction)
**Dependencies:** None
**Synergies:** Aligns with C2 commitment tracking (both identify stable vs noisy layers)

---

### F16-T2: MICROBATCH-PIPELINE-FWD-BWD-OVERLAP
**Tier:** IMPLEMENT
**Principle:** Gradient accumulation (K micro-batches) schedules forward and backward sequentially. Pipeline them on separate CUDA streams: fwd(mu_2) overlaps with bwd(mu_1). No read-write conflict (both read W, bwd writes to separate grad_W). Savings: one forward pass hidden behind first backward.

**Application:** `src/model_v7/train.py` — training_step() gradient accumulation loop

**Pseudocode:**
```python
stream_fwd = torch.cuda.Stream()
stream_bwd = torch.cuda.Stream()
# Micro-batch 1 forward
with torch.cuda.stream(stream_fwd):
    logits_1 = model(mu_1); loss_1 = criterion(logits_1, target_1)
    fwd_done = stream_fwd.record_event()
# Micro-batch 1 backward; micro-batch 2 forward overlaps
stream_bwd.wait_event(fwd_done)
with torch.cuda.stream(stream_bwd): loss_1.backward()
with torch.cuda.stream(stream_fwd): logits_2 = model(mu_2); loss_2 = criterion(logits_2, target_2)
stream_fwd.synchronize(); stream_bwd.synchronize()
loss_2.backward(); optimizer.step()
```

**Expected gain:** 8-15% wall clock reduction per step when grad_accum_steps=2
**Dependencies:** None (torch.cuda.Stream is standard API; CPU fallback: sequential)
**Synergies:** Orthogonal to F15 (layer skip); both improve step throughput

---

### F17-T3: COMMITTED-LAYER-FULL-BACKWARD-BYPASS
**Tier:** IMPLEMENT
**Principle:** AutoFreeze freezes committed weights, but backward still computes grad_weight (matmul) then discards it. For high-commitment layers, skip grad_weight computation entirely; compute only grad_input (chain rule). Selective backward: skip matmul for committed rows, compute only uncommitted.

**Application:** `src/model_v7/components/ternary.py` — TernaryLinear's `TernaryLinearFunction.backward()` add committed_mask check

**Pseudocode:**
```python
@staticmethod
def backward(ctx, grad_output):
    input, W_q, committed_mask = ctx.saved_tensors  # committed_mask per group
    grad_input = grad_output @ W_q  # always needed for chain rule
    uncommitted = ~committed_mask.repeat_interleave(ctx.group_size, dim=0)
    grad_weight = (grad_output.T @ input) * uncommitted.unsqueeze(-1)  # zero committed rows
    return grad_input, grad_weight, None
```

**Expected gain:**
- At 50% commitment: ~50% reduction in grad_weight matmul time per layer
- 15-20% backward speedup when commitment > 50%, growing to 30-40% at convergence (90%+)
- End-to-end: 8-15% when commitment high

**Dependencies:** C2 commitment tracking (already always-on)
**Synergies:** Scales with AutoFreeze progress; pairs with F15 (both optimize backward)

---

### F18-T4: GRADIENT-MAGNITUDE-DEPTH-WARMUP
**Tier:** IMPLEMENT
**Principle:** At initialization, all layers have equal gradient magnitude. Deep layers have higher gradient variance (more paths from output). Early training: updating deep layers is counterproductive. Linearly expand active backward depth: l_max(t) = floor(L * min(1, t/T_warmup)). Shallow layers learn first, clean representations for deep layers. Backward FLOP reduction: 50% on average during warmup.

**Application:** `src/model_v7/model.py` — ModelV2.forward() + `src/model_v7/train.py` — pass current step to model

**Pseudocode:**
```python
# In ModelV2.forward(x, step=0, warmup_steps=200):
l_max = min(len(self.blocks) - 1, int(len(self.blocks) * step / warmup_steps))
for i, block in enumerate(self.blocks):
    x = block(x)
    if i == l_max and self.training and step < warmup_steps:
        x = x.detach()  # truncate backward here; layers i+1...L still run in forward
        print(f"[DEPTH-WARMUP] step={step} active_depth={l_max+1}/{len(self.blocks)}")
```

**Expected gain:**
- During warmup (200 steps, 4 layers): 20-25% total step speedup (50% backward reduction)
- Post-warmup: zero overhead (detach branch never taken)
- Secondary: may improve early loss descent (shallow layers learn cleaner representations first)

**Dependencies:** None
**Synergies:** Works with F15, F17 (all optimize backward); orthogonal to kernel fusion (F1-F9, F10-F14)

---

## Tier 2 — EXPLORE (promising, needs validation)

### F19-M6: TERNARY-B-COVARIANCE-RANK-COLLAPSE-FOR-STATE-COMPRESSION
**Tier:** EXPLORE
**Principle:** SSM state h_T = sum of rank-1 matrices (B_t outer v_t). B_t is ternary output from x_proj, not independent across time. Gram matrix G = (1/T)*sum_t(B_t*B_t^T) may have r_eff < d_state. If so: project state to r_eff-dim subspace, reduce state size by compression_ratio = d_state/r_eff. Empirical rank unknown; requires measurement.

**Application:** `src/model_v7/components/ssm.py` — SelectiveSSMSSD._ssd_scan_fla() + calibration pass

**Pseudocode:**
```python
# Calibration (once at training start): compute effective rank of W_B
U, S, V = torch.linalg.svd(x_proj.weight_q[:, dt_rank:dt_rank+d_state].float())
r_eff = (S > 0.01 * S[0]).sum().item()  # threshold at 1% of max singular value
P = U[:r_eff, :]  # r_eff x d_state projection; store as buffer
# In forward: B_compressed = B @ P.T  (r_eff-dim state instead of d_state-dim)
```

**Expected gain (conditional on r_eff < d_state/2):**
- 10-25% scan speedup if effective rank is < d_state/2
- Currently d_state=16, so viable only if r_eff <= 8

**Validation needed:**
1. Measure eigenvalue decay of W_B for current architecture (likely to be high rank for small d_state)
2. Verify reconstruction error of top-r subspace approximation on validation set
3. Confirm state compression does not degrade perplexity beyond 0.5% tolerance

**Status:** SPECULATIVE — DO NOT IMPLEMENT until empirical rank measurement on training data confirms r_eff << d_state

**Synergies:** If r_eff is low, pairs with F6-F7 (backward memory savings compound with reduced state size)

---

### F20-T5: [SPECULATIVE] ADDITIONAL VALIDATION RECOMMENDATIONS

**Note:** The following findings from W2-AgB and W1-AgB did not pass full G1+G2+G3 gates or have conflicting dependencies:

1. **GRAD-MAGNITUDE-DEPTH-WARMUP convergence claim**: F18 lists 5-15% convergence speedup speculatively. This is EXPLORE, not IMPLEMENT. Requires 50-step baseline vs 50-step with warmup on identical random seed.

2. **DT-HEADS-MEAN-ELIMINATION convergence claim**: F12 claims 5-15% convergence improvement from richer dt signal. EXPLORE-only; throughput gain (1-2%) is IMPLEMENT-ready but convergence measurement needed.

3. **REVERSE-SCAN-FUSED-BACKWARD (F7) complexity**: Custom Triton kernel is medium complexity. Consider implementing after F6 is validated and proven to work in production.

---

## Tier 3 — SKIP (failed gate or already implemented)

### PyTorch Fused SSD 5-kernel (inference-only)
- IBM/PyTorch Feb 2026 paper describes inference/prefill-only fused kernels
- Training backward not covered
- Inference context (batch=1, long seq) differs from training (batch=4, seq=512)
- **Verdict:** SKIP — not applicable to training path

### TMA / Thread Block Clusters (Hopper-specific)
- Hardware feature (NVIDIA H100 only)
- **Verdict:** SKIP — violates portability constraint (project requires CPU-portable algorithms)

### Parallel Prefix Scan at Full Sequence Length T
- IBM confirms: slower in practice due to memory bandwidth at T=512
- Storing (a_t, b_t) for 512 positions requires large intermediate storage
- Only viable at CHUNK level (n_chunks=8), which is F14-M5
- **Verdict:** SKIP — mathematically proven suboptimal for GPU; F14 applies at correct granularity

### GradLite (Low-rank Jacobian Approximation)
- Maintains per-parameter residual accumulator r_t (FP32 extra state)
- For Lion (sign-based), gradient approximation errors already tolerated via sign compression
- Adds overhead for benefit Lion intrinsically provides
- **Fails G3:** Extra always-on state per param is LEGO architecture (flag OFF by default)
- **Verdict:** SKIP — benefits already provided by sign compression

---

## Implementation Order & Dependencies

**Phase 1 (Day 1):** Independent kernel fusions — can parallelize
```
F1 (FLA-FUSED-NORM-GATE) — modify ssm.py forward (2 lines)
F2 (SCAN-BF16-STATES) — remove .float() casts (1 line)
F4 (CHUNK-SIZE-128) — change default parameter (1 line)
F10 (GATE-ABSORBED) — refactor gate computation (4 lines)
F12 (DT-HEADS-MEAN) — add parameter + replace .mean() (4 lines)
Parallel: All independent of each other
Est. time: 1-2 hours (4-5 small changes)
```

**Phase 2 (Day 1-2):** Kernel replacement (requires testing)
```
F5 (CHUNK-MAMBA2-KERNEL) — replace kernel function call (4 lines, with fallback)
F3 (DEPTHWISE-CONV1D-PRELOAD) — refactor scan input (4 lines, pairs with F4)
Depends on: F1, F2, F4 being tested first
Est. time: 2-3 hours (higher complexity; needs backward validation)
```

**Phase 3 (Day 2):** Backward optimization (high-value, medium complexity)
```
F6 (FLA-FUSED-RECURRENT-BACKWARD) — kernel function swap (1 line) or fallback
F8 (SQRT-T-BOUNDARY-CHECKPOINT) — checkpoint strategy refactor (10 lines)
Depends on: Phase 1+2 stable
Est. time: 2-3 hours
```

**Phase 4 (Day 3):** Chunked scan & training optimizations (can overlap)
```
F11 (TSCS-A-ZERO-PREFIX) — add pre-pass to scan loop (8 lines)
F14 (AFFINE-MONOID-PREFIX) — replace sequential loop (8 lines)
  Note: F11 and F14 target same loop (1341-1380); unify into one PR
F13 (CB-PRECOMPUTATION) — add P_committed buffer + einsum (6 lines)
Depends on: F5 or F11 being tested (chunked scan fallback)
Est. time: 2-3 hours
```

**Phase 5 (Day 3-4):** Training-loop optimizations (highly parallelizable)
```
F15 (GRAD-SNR-ADAPTIVE-SKIP) — add EMA tracking + grad zeroing (8 lines)
F16 (MICROBATCH-PIPELINE) — add stream management (6 lines)
F17 (COMMITTED-BACKWARD-BYPASS) — modify TernaryLinear.backward (4 lines)
F18 (DEPTH-WARMUP) — add depth scheduling (4 lines)
Parallel: All independent of each other and kernel work
Est. time: 1.5-2 hours
```

**Phase 6 (Day 4):** Validation & Cleanup
```
- Smoke tests (3 layer, 15 steps) for each change
- Integration test (6M model, 50 steps) with all Phase 1-5 combined
- Measure VRAM, throughput, loss curves
- Commit by logical grouping (3-4 PRs: kernel fusion, backward opt, training loop, paradigm)
```

**Phase 7 (Optional):** Exploration
```
F19 (STATE-RANK-COMPRESSION) — only if rank analysis shows r_eff < 8
- Requires SVD measurement of W_B (one-time calibration)
- Can be added post-hoc if empirical data supports it
```

### Critical Dependencies Matrix

| Finding | Blocks | Blocked by | Notes |
|---------|--------|-----------|-------|
| F1 | — | — | Independent |
| F2 | — | — | Independent |
| F3 | F5 | F4 (chunk_size) | Works with both FLA and chunked fallback |
| F4 | — | — | Independent; enables optimal perf for F3, F5 |
| F5 | F3 | — | Replaces kernel path; F3 applies to either backend |
| F6 | — | — | Alternative to F7 (both improve backward); no conflict |
| F7 | — | — | Alternative to F6; custom kernel (medium complexity) |
| F8 | F6, F7 | — | Orthogonal; all three improve backward memory |
| F9 | — | — | Independent (loss computation) |
| F10 | — | — | Independent; complements F1 |
| F11 | F14 | — | Both target _ssd_scan_chunked loop; unify in one PR |
| F12 | — | — | Independent; works with all scan backends |
| F13 | — | — | Requires C2 always-on (already true) |
| F14 | F11 | — | See F11 |
| F15 | — | — | Independent |
| F16 | — | — | Independent |
| F17 | — | — | Requires C2 always-on (already true) |
| F18 | — | — | Independent |
| F19 | — | — | EXPLORE only; no dependencies if deferred |

---

## Synergy Groups (For Batch Implementation)

**Group A: Kernel Fusion (F1-F5, F10)**
- All target SSM scan efficiency via kernel calls
- Expected combined gain: 10-20% end-to-end
- Implementation: SSM module (`src/model_v7/components/ssm.py`)
- PR title: `feat(r8-a2): SSM kernel fusion — fused norm+gate, BF16 scan, chunk_size=128, direct mamba2`

**Group B: Backward Optimization (F6-F9)**
- All reduce activation memory in backward
- Expected combined gain: 40-60% VRAM reduction + 10-25% backward throughput
- Implementation: SSM module + loss module
- PR title: `feat(r8-a2): SSM backward optimization — fused recurrent, sqrt-T checkpoint, Liger CE`

**Group C: Chunked Scan Restructuring (F11-F14)**
- All target _ssd_scan_chunked sequential loop
- Expected combined gain: 8-15% for chunked fallback path
- Implementation: SSM module
- PR title: `feat(r8-a2): SSM chunked scan — TSCS parallel prefix + affine monoid + CB precomputation`

**Group D: Paradigm (Mathematical) Innovations (F12, F13)**
- Algorithmic insights applied to existing structures
- Expected combined gain: 5-10% when commitment grows
- Implementation: SSM module
- PR title: `feat(r8-a2): SSM paradigm shifts — learned dt aggregation + committed weight precomputation`

**Group E: Training Loop (F15-F18)**
- All optimize backward pass or optimizer step
- Expected combined gain: 20-40% total step time during certain phases
- Implementation: Train module + model module + ternary module
- PR title: `feat(r8-a2): Training optimizations — SNR-based skip + pipeline overlap + committed bypass + depth warmup`

---

## Summary Table — All Findings Ranked

| Tier | ID | Name | File | Expected Gain | Implementation Complexity | Risk | Priority |
|------|----|----|------|---------------|---------------------------|------|----------|
| 1 | F1-K1 | FLA-Fused-Norm-Gate | ssm.py | 3-7% | Trivial (2 lines) | Low | P0 |
| 1 | F2-K2 | BF16-Scan-States | ssm.py | 4-8% | Trivial (1 line) | Low | P0 |
| 1 | F4-K4 | Chunk-Size-128 | ssm.py | 2-6% | Trivial (1 line) | Low | P0 |
| 1 | F10-M1 | Gate-Absorbed | ssm.py | 2-4% | Simple (4 lines) | Low | P1 |
| 1 | F12-M3 | DT-Heads-Mean | ssm.py | 1-2% + conv | Simple (4 lines) | Low | P1 |
| 1 | F3-K3 | Depthwise-Conv-Preload | ssm.py | 2-5% | Medium (4 lines) | Medium | P1 |
| 1 | F5-K5 | Direct-Mamba2-Kernel | ssm.py | 2-5% | Medium (4 lines + fallback) | Medium | P1 |
| 1 | F9-B4 | Liger-Fused-CE | train.py | 10-20% | Simple (1 line) | Low | P1 |
| 1 | F6-B1 | Fused-Recurrent-Backward | ssm.py | 15% VRAM + 10-25% | Trivial (1 line) | Low | P2 |
| 1 | F8-B3 | Sqrt-T-Checkpoint | ssm.py | 50-70% VRAM | Medium (10 lines) | Medium | P2 |
| 1 | F11-M2 | TSCS-Parallel-Prefix | ssm.py | 3-8% (TSCS) | Medium (8 lines) | Medium | P2 |
| 1 | F13-M4 | CB-Precomputation | ssm.py | 3-8% (post-commit) | Simple (6 lines) | Low | P2 |
| 1 | F14-M5 | Affine-Monoid-Prefix | ssm.py | 2-6% | Medium (8 lines) | Medium | P3 |
| 1 | F15-T1 | SNR-Adaptive-Skip | train.py | 5-10% | Medium (8 lines) | Low | P2 |
| 1 | F16-T2 | Pipeline-Overlap | train.py | 8-15% | Simple (6 lines) | Low | P2 |
| 1 | F17-T3 | Committed-Backward-Bypass | ternary.py | 8-15% | Simple (4 lines) | Low | P2 |
| 1 | F18-T4 | Depth-Warmup | model.py + train.py | 20-25% (warmup phase) | Simple (4 lines) | Low | P2 |
| 2 | F7-B2 | Reverse-Scan-Backward | triton kernel | 5-12% | High (custom Triton) | Medium | P3 |
| 2 | F19-M6 | State-Rank-Compression | ssm.py | 10-25% (speculative) | Medium (6 lines) | High | P4 |
| 3 | — | PyTorch Fused SSD | — | — | — | — | SKIP |

---

## Measurement & Validation Strategy

### Smoke Test (per finding)
- 3-layer model, 15 steps, batch=4, seq=512
- Metrics: VRAM peak, throughput (tok/s), loss curve monotonicity
- Success: loss decreases or stable, VRAM < init + 500MB, throughput > baseline (F1-F5) or < baseline (F6-F9, VRAM-positive)

### Integration Test (all Tier 1, batch groups)
- 6-layer model (actual config), 50 steps, batch=4, seq=512
- Baseline: no changes, record throughput + VRAM + loss@50
- Group A only (F1-F5): record same
- Group A+B (F1-F9): record same
- Group A+B+C+D (F1-F14): record same
- Group A+B+C+D+E (all Tier 1): record same
- Expectation: cumulative gains stack without regression (loss tolerance ±1%)

### Convergence Test (sampling)
- Run F12 (DT-heads-mean) alone on 6M model for 1000 steps
- Compare final loss@1000 vs baseline@1000 on same data seed
- If difference > 0.5 loss, investigate; if < 0.5, mark convergence gain as "validated" or "neutral"

### Ranking Priority

**P0 (commit Day 1, morning):** F1, F2, F4
- All trivial changes (<2 lines each)
- High confidence (IBM-validated, FLA-official)
- Quick smoke test (5 mins each)

**P1 (commit Day 1, afternoon):** F3, F5, F9, F10, F12
- Simple to medium changes (4 lines max, 1 of which is 1 line F9)
- High confidence (validated in literature)
- Smoke test (15 mins each)

**P2 (commit Day 2):** F6, F8, F11, F13, F15, F16, F17, F18
- Medium changes (6-10 lines) or high-value (VRAM reduction)
- Medium confidence (need backward validation for F6, F8; need training run for F15-F18)
- Integration test (50 steps) before committing group

**P3 (commit Day 3, if time):** F7, F14
- High complexity (F7: custom Triton) or fine-tuning (F14: prefix scan)
- Medium confidence (require fallback handling)
- Defer if Tier 1 gains are already substantial

**P4 (explore, no commit):** F19
- Speculative, requires empirical rank measurement first
- Defer until after F6-F8 are stable

---

## Conclusion

**15 IMPLEMENT findings** span 3 categories:
1. **Kernel fusion** (F1-F5, F10): 10-20% e2e from fewer kernel launches
2. **Backward memory** (F6-F9, F17-F18): 40-60% VRAM + 15-25% backward speedup
3. **Algorithm restructuring** (F11-F14): 5-15% from parallelized loops

**Combined expected gain:** 40-80% end-to-end throughput improvement when all Tier 1 findings are implemented. VRAM improvements enable batch size increase (4→6-8), effectively multiplying training throughput per wall-clock minute.

**4 EXPLORE findings** (F7, F12-convergence, F18-convergence, F19) require empirical validation before scaling. Worth investigating post-Phase-5 if time permits.

**1 SKIP finding** (parallel full-T scan) is mathematically proven suboptimal; F14 applies the same principle at correct granularity (chunk level).

**Critical path:** Commit Phase 1 (P0) first, test, then Phase 2 (P1), then Phases 3-5 in parallel. Total estimated effort: 3-4 days for full implementation, 5-7 days including validation + paper documentation.
