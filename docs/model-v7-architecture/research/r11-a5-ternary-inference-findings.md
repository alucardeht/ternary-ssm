# R11-A5 — Ternary Inference Acceleration (Wave 1+2 Synthesis)

**Round:** R11 | **Aspect:** A5 | **Source:** W1-A, W1-B, W2-A, W2-B | **Synthesized:** 2026-03-06

---

## Summary Table

| ID | Finding | Gain | Source | Verdict |
|----|----|----|----|-----|
| F1 | INT2 ternary weight packing @ eval | 8× memory BW, 2-4× throughput | W1-A | IMPLEMENT |
| F2 | LSQ scale absorption into weights | 10-20% speedup | W1-A | IMPLEMENT |
| F3 | Bitmask unpack cache @ eval | 25-40% bitmask overhead | W1-A | IMPLEMENT |
| F4 | L2 cache residency for GEMV | ~32× memory efficiency | W1-A | IMPLEMENT |
| F5 | SSM recurrent step bypass (seq_len=1) | 3-8× per-token SSM | W1-A | IMPLEMENT |
| F6 | H-state fast-decay head skip | 25% loop speedup (~31% heads) | W1-B | IMPLEMENT |
| F7 | A_col precompute LUT | 2-5% latency reduction | W1-B | IMPLEMENT |
| F8 | H-state BF16 storage | 2× h-bandwidth reduction | W1-B | IMPLEMENT |
| F9 | Speculative decoding activation replay | 1.5-2× throughput | W1-B | IMPLEMENT |
| F10 | Copy-on-Write h-state beam search | 75% mem reduction (prefix) | W1-B | IMPLEMENT |
| F11 | Sparse index GEMV @ 60% sparsity | 2.5× fewer reads, 3× storage | W2-A | IMPLEMENT |
| F12 | Sorted index cache locality | 15-25% additional via spatial cache | W2-A | IMPLEMENT |
| F13 | Input pruning via magnitude ranking | 30% GEMV reduction, <0.3% loss | W2-A | IMPLEMENT |
| F14 | Arithmetic coding (ANS) ternary weight | 29% size reduction vs INT2 | W2-A | EXPLORE |
| F15 | Low-rank input batched GEMV (SVD) | 5-8× speedup @ batch>1 | W2-A | IMPLEMENT |
| F16 | Vocab-indexed in_proj precompute | 10-20% latency reduction | W2-B | IMPLEMENT |
| F17 | Conv1d ring-buffer state machine | 5-15% latency reduction | W2-B | IMPLEMENT |
| F18 | Bx_table precomputation (LTI mode) | 30-50% SSM step reduction | W2-B | EXPLORE |
| F19 | FST output cache (B,C,dt memoization) | 8-15% latency (85-90% hit) | W2-B | IMPLEMENT |
| F20 | SSM head saturation early exit | 10-25% additional speedup | W2-B | EXPLORE |

---

## IMPLEMENT Findings (16)

### F1: INT2 Ternary Weight Packing @ Eval
**ID:** R11-A5-W1-A-F1
**Source:** W1-A
**Principle:** At eval mode, ternary weights are constant. Pack to INT2 (2 bits/weight) offline, read 8× less memory than FP16.
**File:** `src/model_v7/components/ternary.py` line ~2715 (eval branch)
**Pseudocode:**
```python
# In TernaryLinear.eval():
self._eval_int2 = pack_int2(self.weight_quantized)
# In forward (eval mode):
w_pos, w_neg = unpack_int2_to_masks(self._eval_int2)
output = (x @ w_pos.T) - (x @ w_neg.T) + bias
```
**Expected gain:** 8× memory bandwidth vs FP16; 2-4× throughput on CPU (885 KB vs 7 MB for 6M).
**G1/G2/G3:** ✓ pass/pass/pass

---

### F2: LSQ Scale Absorption Into Weights
**ID:** R11-A5-W1-A-F2
**Source:** W1-A
**Principle:** Absorb asymmetric LSQ scales into ternary GEMV weights at eval time (constant scales).
**File:** `src/model_v7/components/ternary.py` line ~2801 (eval override)
**Pseudocode:**
```python
# In TernaryLinear.eval():
self._eval_scaled_pos = (w_pos * sp).to(torch.float16)
self._eval_scaled_neg = (w_neg * sn).to(torch.float16)
# In forward (eval): output = F.linear(x, scaled_pos - scaled_neg, bias)
```
**Expected gain:** 10-20% speedup (eliminates per-group scale broadcast/multiply).

---

### F3: Bitmask Unpack Cache @ Eval
**ID:** R11-A5-W1-A-F3
**Source:** W1-A
**Principle:** Cache unpacked float masks at eval entry (weights fixed); skip 32-bit unpack loop per call.
**File:** `src/model_v7/components/bitmask_utils.py` line ~68 + `ternary.py` line ~2801
**Pseudocode:**
```python
# In TernaryLinear.eval():
self._eval_w_merged = unpack_and_merge_bitmask(self.bitmask_pos, self.bitmask_neg)
# In forward (eval): output = F.linear(x, self._eval_w_merged, bias)
```
**Expected gain:** 25-40% bitmask_gemv speedup (saves ~400 unpack ops/pass for 4-layer autoregressive).

---

### F4: L2 Cache Residency For GEMV
**ID:** R11-A5-W1-A-F4
**Source:** W1-A
**Principle:** At seq_len=1 inference, entire 6M INT2 ternary weights (885 KB) fit in L2 cache (~1-4 MB per core).
**File:** `src/model_v7/components/ternary.py` line ~2801 (ensure contiguous)
**Pseudocode:**
```python
# In TernaryLinear.eval():
self._eval_int2 = pack_int2(w).contiguous()
if device.type == 'cpu': self._eval_int2 = self._eval_int2.pin_memory()
```
**Expected gain:** ~32× memory efficiency vs FP16 DRAM (L2 @200GB/s vs DRAM @50GB/s).

---

### F5: SSM Recurrent Step Bypass (seq_len=1)
**ID:** R11-A5-W1-A-F5
**Source:** W1-A
**Principle:** At seq_len=1 eval, skip FLA scan kernel; run O(1) raw recurrence directly.
**File:** `src/model_v7/components/ssm.py` (new `step_recurrent` method)
**Pseudocode:**
```python
# In SelectiveSSMSSD.forward (eval, seq_len=1):
B_t = self.x_proj_B(x[:, 0, :])
h_next = A_t * self.h_cache + B_t * x_inner
self.h_cache = h_next
return C_t^T @ h_next
```
**Expected gain:** 3-8× SSM per-token reduction (~0.3ms scan → ~0.04ms recurrence).

---

### F6: H-State Fast-Decay Head Skip
**ID:** R11-A5-W1-B-F1
**Source:** W1-B
**Principle:** ~31% of heads have |A_bar| < 1e-3 (fast decay). Skip h read/write for stateless mode.
**File:** `src/model_v7/components/ssm.py` line ~903 (_lti_sequential)
**Pseudocode:**
```python
# Precompute at eval: fast_mask = (A_col < 1e-3)  # ~31% heads
# Per step: y_fast = direct compute; h updates only for ~69% heads
```
**Expected gain:** ~25% loop speedup (eliminate 20/64 head updates per token per layer).

---

### F7: A_col Precompute LUT
**ID:** R11-A5-W1-B-F2
**Source:** W1-B
**Principle:** Cache A_col = exp(-exp(A_log)) at eval entry (frozen parameter).
**File:** `src/model_v7/components/ssm.py` line ~887 (_lti_sequential)
**Expected gain:** 2-5% latency reduction (eliminates transcendental per n_heads per token).

---

### F8: H-State BF16 Storage
**ID:** R11-A5-W1-B-F3
**Source:** W1-B
**Principle:** Store h in BF16 at inference (4 MB per layer → 2 MB). Bounded error: O(T_eff * eps_bf16).
**File:** `src/model_v7/components/ssm.py` line ~895, ~1300 (h allocation dtype)
**Expected gain:** 2× h-bandwidth reduction; 1.5-2× speedup for h-update portion.

---

### F9: Speculative Decoding Activation Replay
**ID:** R11-A5-W1-B-F4
**Source:** W1-B
**Principle:** Store lightweight activations (~6KB/token), recompute h on rejection instead of full h snapshot (24 MB).
**File:** `src/model_v7/components/ssm.py` line ~879
**Expected gain:** 1.5-2× throughput in speculative generation (660× memory reduction per step).

---

### F10: Copy-on-Write H-State Beam Search
**ID:** R11-A5-W1-B-F5
**Source:** W1-B
**Principle:** Share h reference across beams until divergence; copy only on write at first mismatch.
**File:** `src/model_v7/components/ssm.py` line ~895, ~1300
**Expected gain:** 75% h-mem reduction during prefix sharing (24 MB → 12 MB for beam_size=4 with BF16).

---

### F11: Sparse Index GEMV @ 60% Sparsity
**ID:** R11-A5-W2-A-F1
**Source:** W2-A
**Principle:** Precompute ±1 nonzero indices; GEMV becomes gather-and-add (no multiply).
**File:** `src/model_v7/components/ternary.py` line ~2801 (eval override)
**Pseudocode:**
```python
# At eval: self._eval_pos_idx, self._eval_neg_idx = precompute indices
# Forward: y[i] = x[pos_idx[i]].sum() - x[neg_idx[i]].sum()
```
**Expected gain:** 2.5× fewer memory reads @ 60% sparsity; 3× storage (472 KB vs 1.5 MB weights).

---

### F12: Sorted Index Cache Locality
**ID:** R11-A5-W2-A-F2
**Source:** W2-A
**Principle:** Store nonzero indices sorted by column; gathers use sequential cache lines.
**File:** `src/model_v7/components/ternary.py` line ~2801
**Expected gain:** 15-25% additional throughput via hardware prefetcher engagement (on top of F11).

---

### F13: Input Pruning Via Magnitude Ranking
**ID:** R11-A5-W2-A-F3
**Source:** W2-A
**Principle:** Bottom 30% of |x| contribute <5% energy; prune them from GEMV (Laplacian tail bound).
**File:** `src/model_v7/components/ternary.py` line ~2715 (eval forward)
**Pseudocode:**
```python
# Before GEMV: threshold = quantile(x.abs(), 0.30); x_pruned = x * (x.abs() >= threshold)
```
**Expected gain:** ~25% GEMV speedup; cumulative <0.3% perplexity loss @ 6 layers.

---

### F15: Low-Rank Input Batched GEMV (SVD)
**ID:** R11-A5-W2-A-F5
**Source:** W2-A
**Principle:** For batch>1, decompose X ≈ U Σ V^T (rank r_eff << K); compute W × V then scale/multiply.
**File:** `src/model_v7/components/ternary.py` line ~2715 (batch>1 path)
**Pseudocode:**
```python
# U, S, Vh = SVD(x, full_matrices=False); r = rank for 99% variance
# M = ternary_GEMV(W, V_r); y = (U @ M) * S
```
**Expected gain:** 5-8× speedup for batch=8, r_eff=10 (76× FLOP reduction on ternary GEMV).

---

### F16: Vocab-Indexed in_proj Precompute
**ID:** R11-A5-W2-B-F1
**Source:** W2-B
**Principle:** Precompute in_proj(embedding[tok]) for all V vocab entries at eval entry.
**File:** `src/model_v7/model.py` line ~270 (in ModelV2.eval())
**Pseudocode:**
```python
# In ModelV2.eval(): self._inproj_cache = in_proj(all_embeddings)  # (V, 2*d_inner)
# In forward: xz = self._inproj_cache[tok_id]  # O(1) lookup
```
**Expected gain:** 10-20% latency reduction (eliminates in_proj GEMV per token, 25.7 MB cache).

---

### F17: Conv1d Ring-Buffer State Machine
**ID:** R11-A5-W2-B-F2
**Source:** W2-B
**Principle:** Replace full conv1d forward with persistent ring buffer + d_conv-1 multiply-accumulates per token.
**File:** `src/model_v7/components/ssm.py` line ~1065 (conv1d call)
**Pseudocode:**
```python
# Register: self._conv_buf = torch.zeros(1, d_inner, d_conv-1)
# Per step: shift ring, compute output = kernel_weight * [buf | x_new]
```
**Expected gain:** 5-15% latency reduction (eliminates conv1d kernel launch overhead).

---

### F19: FST Output Cache (B, C, dt Memoization)
**ID:** R11-A5-W2-B-F4
**Source:** W2-B
**Principle:** Cache (tok_t, conv_state_hash) → (B_t, C_t, dt_t) for common token patterns.
**File:** `src/model_v7/components/ssm.py` line ~1097 (x_proj call)
**Pseudocode:**
```python
# cache_key = (tok_id, hash(conv_buf));
# if key in cache: use cached; else: compute x_proj, store
```
**Expected gain:** 8-15% latency (85-90% hit rate on top-1000 tokens, bounded to 32K entries ~10 MB).

---

## EXPLORE Findings (4)

### F14: Arithmetic Coding (ANS) Ternary Weight
**ID:** R11-A5-W2-A-F4
**Source:** W2-A
**Principle:** Entropy-optimal ternary compression: 1.42 bits/weight (ANS) vs 2.0 bits (INT2).
**File:** `src/model_v7/components/ternary.py` line ~2010 (_pack_ternary_weights)
**Pseudocode:**
```python
# In eval(): self._eval_ans_stream = ternary_ans_encode(weight_quantized)
# In forward: w_row = ternary_ans_decode(stream[i])  # inline decode
```
**Expected gain:** 29% weight reduction vs INT2 (1.07 MB vs 1.5 MB @ 6M); ~8µs savings per model forward.
**Verdict:** EXPLORE — Requires ANS decoder library; benchmark decode throughput vs INT2 unpack overhead.

---

### F18: Bx_table Precomputation (LTI Mode)
**ID:** R11-A5-W2-B-F3
**Source:** W2-B
**Principle:** Precompute B_t * x_conv for all V tokens (vocab-indexed table); eliminates per-token einsum.
**File:** `src/model_v7/components/ssm.py` line ~909 (_lti_sequential)
**Expected gain:** 30-50% SSM step reduction (in LTI mode).
**Verdict:** EXPLORE — Table size: 3.3 GB @ d_state=16 (feasible for small test variant); 26 GB @ full SSD d_state=128 (infeasible).
**Note:** Viable for d_state=16 test variant only; mark as speculative for 6M config.

---

### F20: SSM Head Saturation Early Exit
**ID:** R11-A5-W2-B-F5
**Source:** W2-B
**Principle:** Detect heads with |Δh| < ε (saturated); use steady-state formula h/(1-A_bar) instead of full update.
**File:** `src/model_v7/components/ssm.py` line ~907 (_lti_sequential), ~1388 (_ssd_scan_chunked)
**Pseudocode:**
```python
# Per step: delta_rel = |h_t - h_{t-1}| / |h_{t-1}|; sat_mask = (delta_rel < 1e-3)
# For saturated heads: h = Bx / (1 - A_col)  # steady-state formula
```
**Expected gain:** 10-25% additional speedup on >64-token sequences (20-40% heads saturate by token 50).
**Verdict:** EXPLORE — Requires runtime saturation detection; benchmark overhead vs savings.

---

*Synthesis complete: 20 findings (16 IMPLEMENT, 4 EXPLORE). All evaluated via Gate v3 (Actionability, Applicability, Always-Active). No LEGO patterns detected.*
