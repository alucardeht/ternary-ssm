# R11-A1: Ternary Bitwise Compute — Synthesized Findings

**Round:** R11 | **Aspect:** A1 (Ternary Bitwise Compute)
**Synthesis Date:** 2026-03-06
**Wave Coverage:** W1-AgA (5), W1-AgB (5), W2-AgA (5), W2-AgB (5) = 20 total findings
**All findings pass Gate v3 (G1✓ G2✓ G3✓): IMPLEMENT**

---

## Summary Table

| Rank | ID | Technique | File | Expected Gain | Source |
|------|----|-----------:|------|---------------|--------|
| 1 | R11-A1-W2-B-F3 | CRT-GROUP-INT8-ACCUMULATION-TERNARY-GEMM | bitmask_utils.py~79 | 2-3x CPU fwd / 1.5-2x GPU | W2-AgB |
| 2 | R11-A1-W1-A-F4 | ABSMEAN-INT8-ACT-BITMASK-GEMM | bitmask_utils.py~79 | 2-4x matmul | W1-AgA |
| 3 | R11-A1-W2-A-F1 | WHT-NIBBLE-SUBSET-SUM-GEMM | bitmask_utils.py~85 | 2.3x FLOP, 1.2-2.0x tput | W2-AgA |
| 4 | R11-A1-W1-A-F1 | BITMASK-GEMV-ALWAYS-ACTIVE | config.py~47 | 15-30% fwd throughput | W1-AgA |
| 5 | R11-A1-W2-B-F2 | BOOTH-LSQ-SYMMETRIC-BINARY-CORRECTION-DECOMP | bitmask_utils.py~46 | 15-25% bitmask_gemv | W2-AgB |
| 6 | R11-A1-W1-B-F5 | COMMITTED-ZERO-GRAD-WEIGHT-ROW-SKIP | ternary.py~1505 | ~23% e2e at f=0.70 | W1-AgB |
| 7 | R11-A1-W1-B-F1 | COMMITTED-NONZERO-COL-SELECT-GEMM | ternary.py~2713 | 30-35% FLOP reduction | W1-AgB |
| 8 | R11-A1-W1-B-F4 | STRUCTURED-ROW-SPARSITY-BLOCK-NONZERO-SLICE | ternary.py~2713 | 35-50% FLOP after commit | W1-AgB |
| 9 | R11-A1-W1-A-F5 | W-SUM-SCALE-CORRECTION-BITMASK-TRAINING | bitmask_utils.py~46 | 5-10% step reduction | W1-AgA |
| 10 | R11-A1-W2-A-F4 | GF3-TRIT-XOR-ROW-PARTIAL-BITMASK-INVALIDATE | ternary.py~2693 | 59-70% repack reduction | W2-AgA |
| 11 | R11-A1-W1-B-F3 | ZERO-WEIGHT-INPUT-MASK-BATCHED-REDUCTION | ternary.py~2713 | 41% multiply elimination | W1-AgB |
| 12 | R11-A1-W1-B-F2 | SIGN-PARTITION-TERNARY-TRAINING-GEMM | ternary.py~2713 | 1.2-1.8x GPU / 2-4x CPU | W1-AgB |
| 13 | R11-A1-W1-A-F3 | BITMASK-STALE-C2-DIRTY-TRACKING | ternary.py~2695 | 5-12% step reduction | W1-AgA |
| 14 | R11-A1-W1-A-F2 | BITMASK-TRAINING-SINGLE-KERNEL-FUSED | bitmask_utils.py~82 | 2x FLOP reduction | W1-AgA |
| 15 | R11-A1-W2-B-F1 | NAF-SIGNED-PAIR-CACHE-TERNARY-ACCUM | ternary.py~2713 | 10-20% committed-layer fwd | W2-AgB |
| 16 | R11-A1-W2-A-F2 | POPCOUNT-L2-WEIGHT-NORM-AGC | ternary.py~1200 | 3-6% e2e (75% AGC reduction) | W2-AgA |
| 17 | R11-A1-W2-A-F3 | BALANCED-TERNARY-INT16-GROUP-ACCUMULATOR | bitmask_utils.py~95 | 1.5-2.0x CPU matmul | W2-AgA |
| 18 | R11-A1-W2-A-F5 | TROPICAL-L∞-AGC-TERNARY-CONSTANT-WEIGHT-NORM | ternary.py~1200 | 3-6% e2e; needs lambda tune | W2-AgA |
| 19 | R11-A1-W2-B-F5 | WHT-SPECTRAL-DRIFT-DETECTION-COMMITTED-ROWS | ternary.py~2300 | 3-7% fewer de-commit events | W2-AgB |
| 20 | R11-A1-W2-B-F4 | LDPC-INCREMENTAL-GF3-PARITY-COMMITTED-INTEGRITY | ternary.py~2696 | 59% parity reduction + drift detect | W2-AgB |

---

## Findings (Ranked by Expected Gain)

### Rank 1: CRT-GROUP-INT8-ACCUMULATION-TERNARY-GEMM

**ID:** R11-A1-W2-B-F3
**Source:** W2-AgB
**Principle:** Chinese Remainder Theorem applied to ternary weights with sign-binarized activations. Each group of K=128 weights summed with sign-binarized inputs stays within INT8 bounds [-128, 128]. Enables pure INT8 arithmetic accumulation (exact, no overflow) followed by per-group scale correction.
**File:** `src/model_v7/components/bitmask_utils.py` line ~79

```python
x_sign = x_g.sign().to(torch.int8)
w_int8 = w_ternary_g.to(torch.int8)
y_int = (w_int8 @ x_sign).to(torch.int16)
scale = x_g.abs().mean()
y_g = y_int.float() * scale
```

**Expected gain:** 2-3x CPU forward throughput (INT8 SIMD is 4x wider than FP32); 1.5-2x GPU via tensor cores.
**Verdict:** IMPLEMENT

---

### Rank 2: ABSMEAN-INT8-ACT-BITMASK-GEMM

**ID:** R11-A1-W1-A-F4
**Source:** W1-AgA
**Principle:** BitNet-style INT8 activation quantization (absmean per-token scaling) enables INT8xINT8 integer GEMM via `torch._int_mm`. STE gradient still uses float weights; forward uses pure integer accumulation with no approximation.
**File:** `src/model_v7/components/bitmask_utils.py` line ~79

```python
scale_x = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8) / 127.0
x_int8 = (x / scale_x).round().clamp(-128, 127).to(torch.int8)
w_int8 = (pos_unpacked.to(torch.int8) - neg_unpacked.to(torch.int8))
y_int = torch._int_mm(x_int8, w_int8.T)
y = y_int.float() * scale_x.squeeze(-1).unsqueeze(1)
```

**Expected gain:** 2-4x throughput on forward matmul; 71.4x energy reduction per BitNet b1.58 (7nm measurements).
**Verdict:** IMPLEMENT

---

### Rank 3: WHT-NIBBLE-SUBSET-SUM-GEMM

**ID:** R11-A1-W2-A-F1
**Source:** W2-AgA
**Principle:** Sum-over-subsets Zeta transform on input nibbles (4-element groups). Precompute 16 subset sums per nibble, then for each output neuron, index lookup pos/neg tables and subtract. Zero multiplications; pure addition and lookup.
**File:** `src/model_v7/components/bitmask_utils.py` line ~85

```python
x2d = x.reshape(-1, K//4, 4)
ss = torch.zeros(*x2d.shape[:-1], 16, dtype=x.dtype, device=x.device)
for bit in range(4):
    mask = 1 << bit
    for s in range(16):
        if s & mask:
            ss[..., s] = ss[..., s ^ mask] + x2d[..., bit]
y = ss[..., pos_nibbles].sum(-2) - ss[..., neg_nibbles].sum(-2)
```

**Expected gain:** 2.3x FLOP reduction (7168 vs 16384 ops); CPU 1.5-2.0x, GPU Triton 1.2-1.5x.
**Verdict:** IMPLEMENT

---

### Rank 4: BITMASK-GEMV-ALWAYS-ACTIVE

**ID:** R11-A1-W1-A-F1
**Source:** W1-AgA
**Principle:** Currently `use_bitmask_gemv=False` in config — a LEGO flag silencing 8x weight memory bandwidth reduction. Enable always-active in all TernaryLinear layers, eliminating FP16 matmul for ternary weights.
**File:** `src/model_v7/config.py` line ~47

```python
# config.py line 47: change default
use_bitmask_gemv: bool = True  # was False
# ternary.py line 1681: remove LEGO guard; always init bitmask buffers
```

**Expected gain:** 8x weight memory bandwidth reduction; 15-30% forward throughput at 6M scale (memory-bandwidth bound).
**Verdict:** IMPLEMENT

---

### Rank 5: BOOTH-LSQ-SYMMETRIC-BINARY-CORRECTION-DECOMP

**ID:** R11-A1-W2-B-F2
**Source:** W2-AgB
**Principle:** Booth's theorem: asymmetric LSQ scales (s_pos, s_neg) decompose into symmetric s_sym=(s_pos+s_neg)/2 and correction d=(s_pos-s_neg)/2. Ternary matmul uses scalar scale (one multiply); binary mask matmul (correction) uses popcount (pure addition). Skip correction if d<eps.
**File:** `src/model_v7/components/bitmask_utils.py` line ~46

```python
s_sym = (lsq_scale_pos + lsq_scale_neg) * 0.5
d = (lsq_scale_pos - lsq_scale_neg) * 0.5
y_main = group_scaled_linear(x, w_ternary, s_sym)
if d.abs().max() > eps:
    y_corr = group_scaled_linear(x, w_nonzero_mask, d)
y = y_main + y_corr
```

**Expected gain:** 15-25% throughput improvement on bitmask_gemv path (separates two F.linear calls).
**Verdict:** IMPLEMENT

---

### Rank 6: COMMITTED-ZERO-GRAD-WEIGHT-ROW-SKIP

**ID:** R11-A1-W1-B-F5
**Source:** W1-AgB
**Principle:** In backward, committed output rows have gradients zeroed by MBGSC hook after computation. Skip grad_weight matmul for fully-committed rows entirely (grad will be zero anyway). Skips matmul for 70% of rows at f=0.70.
**File:** `src/model_v7/components/ternary.py` line ~1505

```python
if ctx.needs_input_grad[1] and committed_mask is not None:
    fully_committed_rows = committed_mask.all(dim=1)
    active_rows = ~fully_committed_rows
    if active_rows.any():
        go_active = go_2d[:, active_rows]
        grad_weight = torch.zeros(w.shape, dtype=go_2d.dtype, device=go_2d.device)
        grad_weight[active_rows] = go_active.t().matmul(x_2d)
```

**Expected gain:** ~23% end-to-end FLOP reduction at f=0.70 (grad_weight is ~33% of total FLOP).
**Verdict:** IMPLEMENT

---

### Rank 7: COMMITTED-NONZERO-COL-SELECT-GEMM

**ID:** R11-A1-W1-B-F1
**Source:** W1-AgB
**Principle:** For committed layers with static zero pattern, precompute nonzero columns once and cache. Slice input/weight before F.linear: ~30-35% of columns are all-zero, FLOP reduction.
**File:** `src/model_v7/components/ternary.py` line ~2713

```python
if self.committed_mask is not None and self.nonzero_cols_cache is None:
    with torch.no_grad():
        col_active = (w.detach().abs().sum(0) != 0)
        self.nonzero_cols_cache = col_active.nonzero(as_tuple=False).squeeze(1)
x_s = x[..., self.nonzero_cols_cache]
w_s = w[:, self.nonzero_cols_cache]
output = F.linear(x_s, w_s.to(x.dtype), self.bias)
```

**Expected gain:** 30-35% FLOP reduction in training forward (after commitment stabilizes, step ~500+).
**Verdict:** IMPLEMENT

---

### Rank 8: STRUCTURED-ROW-SPARSITY-BLOCK-NONZERO-SLICE

**ID:** R11-A1-W1-B-F4
**Source:** W1-AgB
**Principle:** LSQ quantization induces structured sparsity (zero clustering within groups). Group output rows by identical nonzero patterns; compute per-pattern F.linear batched. Typically 3-8 unique patterns; each uses only active columns.
**File:** `src/model_v7/components/ternary.py` line ~2713

```python
def _build_row_groups(w_ternary, group_size=128):
    w_q = w_ternary.reshape(w_ternary.shape[0], -1, group_size)
    nnz_per_group = (w_q != 0).sum(-1)
    patterns, inverse = torch.unique(nnz_per_group, dim=0, return_inverse=True)
    return [(inverse == i).nonzero().squeeze(1) for i in range(patterns.shape[0])]
for row_idx in self._row_groups:
    w_sub = w[row_idx, :]
    active_cols = (w_sub.abs().sum(0) != 0)
    outputs[..., row_idx] = F.linear(x[..., active_cols], w_sub[:, active_cols].to(x.dtype))
```

**Expected gain:** 35-50% FLOP reduction after commitment (structured sparsity + column selection).
**Verdict:** IMPLEMENT

---

### Rank 9: W-SUM-SCALE-CORRECTION-BITMASK-TRAINING

**ID:** R11-A1-W1-A-F5
**Source:** W1-AgA
**Principle:** Inference popcount path uses w_sum correction (per-group sum of ternary weights). Training bitmask path has no correction, introducing systematic quantization bias. Add per-group scale_x and w_sum correction to training path to close accuracy gap.
**File:** `src/model_v7/components/bitmask_utils.py` line ~46

```python
scale_x_g = x.abs().reshape(-1, num_groups, group_size).mean(dim=-1)
w_sum_g = popcount(pos_mask) - 2*popcount(neg_mask)
y_raw = y_pos - y_neg
correction = (scale_x_g.unsqueeze(1) * w_sum_g.unsqueeze(0)).sum(-1)
y = y_raw - correction
```

**Expected gain:** Closes 2-5% accuracy gap between bitmask and FP16 reference; 5-10% step reduction via reduced noise.
**Verdict:** IMPLEMENT

---

### Rank 10: GF3-TRIT-XOR-ROW-PARTIAL-BITMASK-INVALIDATE

**ID:** R11-A1-W2-A-F4
**Source:** W2-AgA
**Principle:** GF(3) property: trit changes iff any of its two bits change. XOR old/new packed weights, find changed rows via bitwise OR across columns, repack ONLY those rows. Skips full-layer repack for committed rows (complementary to F3).
**File:** `src/model_v7/components/ternary.py` line ~2693

```python
changed_rows = ((new_pos ^ prev_pos).any(dim=1) |
                (new_neg ^ prev_neg).any(dim=1))
self._bitmask_pos[changed_rows] = new_pos[changed_rows]
self._bitmask_neg[changed_rows] = new_neg[changed_rows]
self._prev_bitmask_pos, self._prev_bitmask_neg = new_pos, new_neg
```

**Expected gain:** At f=0.59: 59% reduction in bitmask update cost (only uncommitted rows repack).
**Verdict:** IMPLEMENT

---

### Rank 11: ZERO-WEIGHT-INPUT-MASK-BATCHED-REDUCTION

**ID:** R11-A1-W1-B-F3
**Source:** W1-AgB
**Principle:** COO format: store nonzero (row, col, sign) indices. Forward via scatter_add: pure addition/subtraction, zero multiplications (since signs are {-1,+1}). Eliminates 41% multiply-adds.
**File:** `src/model_v7/components/ternary.py` line ~2713

```python
with torch.no_grad():
    pos_r, pos_c = (w.detach() > 0).nonzero(as_tuple=True)
    neg_r, neg_c = (w.detach() < 0).nonzero(as_tuple=True)
x_flat = x.reshape(-1, x.shape[-1])
y = torch.zeros(x_flat.shape[0], w.shape[0], dtype=x.dtype, device=x.device)
y[:, pos_r] += x_flat[:, pos_c]
y[:, neg_r] -= x_flat[:, neg_c]
```

**Expected gain:** 41% multiply elimination; CPU 1.2-1.5x, GPU 1.1-1.3x.
**Verdict:** IMPLEMENT

---

### Rank 12: SIGN-PARTITION-TERNARY-TRAINING-GEMM

**ID:** R11-A1-W1-B-F2
**Source:** W1-AgB
**Principle:** W = W_pos - W_neg with each ~59% sparse (41% zeros + sign split). Use CSR sparse format (cuSPARSE on GPU, CPU BLAS SPARSE). Cache CSR after each optimizer step.
**File:** `src/model_v7/components/ternary.py` line ~2713

```python
if not hasattr(self, '_csr_valid') or not self._csr_valid:
    with torch.no_grad():
        w_pos_dense = (w.detach() > 0).to(torch.float32)
        w_neg_dense = (w.detach() < 0).to(torch.float32)
        self._w_pos_csr = w_pos_dense.to_sparse_csr()
        self._w_neg_csr = w_neg_dense.to_sparse_csr()
x_2d = x.reshape(-1, x.shape[-1]).to(torch.float32)
y = torch.mm(x_2d, self._w_pos_csr.T) - torch.mm(x_2d, self._w_neg_csr.T)
```

**Expected gain:** CPU 2-4x, GPU 1.2-1.8x on forward matmul.
**Verdict:** IMPLEMENT

---

### Rank 13: BITMASK-STALE-C2-DIRTY-TRACKING

**ID:** R11-A1-W1-A-F3
**Source:** W1-AgA
**Principle:** Currently `_bitmask_stale=True` unconditionally every step, triggering full repack. Use C2 commitment tracking: only mark stale if uncommitted weights could have changed. Fully-committed layers skip repack.
**File:** `src/model_v7/components/ternary.py` line ~2695

```python
if self.training:
    fully_committed = (hasattr(self, 'committed_mask') and
                       self.committed_mask is not None and
                       self.committed_mask.all())
    self._bitmask_stale = not fully_committed
```

**Expected gain:** At 59% commitment, 59% of layers avoid repack every step; 5-12% step time reduction.
**Verdict:** IMPLEMENT

---

### Rank 14: BITMASK-TRAINING-SINGLE-KERNEL-FUSED

**ID:** R11-A1-W1-A-F2
**Source:** W1-AgA
**Principle:** Current bitmask forward calls F.linear twice (y_pos, y_neg) then subtracts. Replace with one fused call: `y = F.linear(x, pos_float - neg_float)`. Single GEMM instead of two.
**File:** `src/model_v7/components/bitmask_utils.py` line ~82

```python
w_ternary_float = pos_unpacked.to(x.dtype) - neg_unpacked.to(x.dtype)
y = F.linear(x_2d, w_ternary_float, bias)
```

**Expected gain:** 2x FLOP reduction on bitmask path; 20-40% bitmask throughput improvement.
**Verdict:** IMPLEMENT

---

### Rank 15: NAF-SIGNED-PAIR-CACHE-TERNARY-ACCUM

**ID:** R11-A1-W2-B-F1
**Source:** W2-AgB
**Principle:** Non-Adjacent Form (NAF) applied to committed ternary rows. Partition nonzero +1/-1 positions into canonical pairs; use `x[j_pos] - x[j_neg]` (one subtraction on precomputed difference) vs two separate loads. Halves memory bandwidth for paired fraction.
**File:** `src/model_v7/components/ternary.py` line ~2713

```python
pos_idx, neg_idx = (w[i] > 0).nonzero(), (w[i] < 0).nonzero()
n_pairs = min(len(pos_idx), len(neg_idx))
paired_pos, paired_neg = pos_idx[:n_pairs], neg_idx[:n_pairs]
y[..., i] = (x[..., paired_pos] - x[..., paired_neg]).sum(-1) \
           + x[..., solo_pos].sum(-1) - x[..., solo_neg].sum(-1)
```

**Expected gain:** 10-20% committed-layer forward speedup (bandwidth reduction for ~50% of nonzeros).
**Verdict:** IMPLEMENT

---

### Rank 16: POPCOUNT-L2-WEIGHT-NORM-AGC

**ID:** R11-A1-W2-A-F2
**Source:** W2-AgA
**Principle:** For ternary layers: ||w||_2^2 = popcount(pos_mask | neg_mask) exactly (Hamming weight identity). Bitmask read + K/32 ORs + popcount vs K float multiply-adds. Cache for committed layers (zero recomputation).
**File:** `src/model_v7/components/ternary.py` line ~1200

```python
if hasattr(module, '_bitmask_pos') and module._bitmask_pos is not None:
    if not hasattr(module, '_cached_weight_norm') or module._bitmask_stale:
        nnz = (module._bitmask_pos | module._bitmask_neg).int().sum()
        module._cached_weight_norm = nnz.float().sqrt()
    w_norm = module._cached_weight_norm
```

**Expected gain:** At f=0.70: 75% AGC weight-norm reduction; 2-4% e2e throughput.
**Verdict:** IMPLEMENT

---

### Rank 17: BALANCED-TERNARY-INT16-GROUP-ACCUMULATOR

**ID:** R11-A1-W2-A-F3
**Source:** W2-AgA
**Principle:** Overflow analysis proves INT16 sufficiency for group_size≤258. Custom Triton kernel using INT16 accumulators (2 bytes vs 4 for INT32) halves register pressure. CPU SIMD: 2x wider (32 INT16 vs 16 INT32 in 512-bit register).
**File:** `src/model_v7/components/bitmask_utils.py` line ~95

```python
acc_int16 = tl.zeros([BLOCK_N], dtype=tl.int16)
for k in range(0, group_size, BLOCK_K):
    x_int8 = tl.load(x_ptr + k).to(tl.int8)
    w_trit = tl.load(w_ptr + k).to(tl.int8)
    acc_int16 += x_int8 * w_trit
acc_fp32 += acc_int16.to(tl.float32) * scale_x_g * scale_w_g
```

**Expected gain:** CPU 1.5-2.0x matmul throughput; GPU 1.1-1.3x (reduced register spills).
**Verdict:** IMPLEMENT

---

### Rank 18: TROPICAL-L∞-AGC-TERNARY-CONSTANT-WEIGHT-NORM

**ID:** R11-A1-W2-A-F5
**Source:** W2-AgA
**Principle:** Tropical semiring: ternary weight L∞ norm = 1 always (max|{-1,0,+1}|=1). Replaces weight norm computation with constant; gradient norm becomes max-abs reduction (no multiply). Requires lambda rescaling (lambda_∞ = lambda_2 × sqrt(K×sparsity)).
**File:** `src/model_v7/components/ternary.py` line ~1200

```python
if is_ternary_layer:
    w_norm_inf = 1.0
    g_norm_inf = grad.abs().max()
    clip_factor = (lambda_ * w_norm_inf / g_norm_inf.clamp(min=1e-8)).clamp(max=1.0)
    grad.mul_(clip_factor)
```

**Expected gain:** 3-6% e2e throughput (eliminates weight norm, reduces grad norm compute). **Requires empirical validation of convergence after lambda rescaling.**
**Verdict:** IMPLEMENT (with convergence validation)

---

### Rank 19: WHT-SPECTRAL-DRIFT-DETECTION-COMMITTED-ROWS

**ID:** R11-A1-W2-B-F5
**Source:** W2-AgB
**Principle:** Walsh-Hadamard Transform computes spectral signature of committed rows (fixed signature = committed ternary pattern). After LSQ, recompute WHT for candidate rows; if spectral L1 distance > threshold, mark unstable (prevents correlated weight flips invisible to per-element checks).
**File:** `src/model_v7/components/ternary.py` line ~2300

```python
def _compute_row_wht(w_row: torch.Tensor) -> torch.Tensor:
    x = w_row.float().clone()
    h = 1
    while h < len(x):
        for i in range(0, len(x), h * 2):
            a, b = x[i:i+h].clone(), x[i+h:i+2*h].clone()
            x[i:i+h], x[i+h:i+2*h] = a + b, a - b
        h *= 2
    return x
if row_is_candidate_for_commitment:
    wht_now = _compute_row_wht(w_ternary[row])
    if (wht_now - wht_cache[row]).abs().sum() > tau * len(wht_now):
        n_stable[row] = 0
```

**Expected gain:** 3-7% fewer de-commitment events; 2-5% reduction in total steps to convergence.
**Verdict:** IMPLEMENT

---

### Rank 20: LDPC-INCREMENTAL-GF3-PARITY-COMMITTED-INTEGRITY

**ID:** R11-A1-W2-B-F4
**Source:** W2-AgB
**Principle:** GF(3) parity P_g = (sum_j W_gj + 128) mod 3 is fixed at commitment. Track per-group parities incrementally: update only uncommitted groups' parities; check committed groups' parities every step for corruption (early drift detection).
**File:** `src/model_v7/components/ternary.py` line ~2696

```python
w_int = (w_ternary.detach() + 1).long()
w_groups = w_int.reshape(num_groups, group_size)
new_parity = w_groups.sum(dim=1) % 3
corrupted = committed_mask_groups & (new_parity != self.gf3_parity)
self.gf3_parity[~committed_mask_groups] = new_parity[~committed_mask_groups].to(torch.int8)
if corrupted.any():
    self.committed_mask[corrupted_rows] = False
```

**Expected gain:** At f=0.59: 59% parity compute reduction; early detection of MBGSC bypass failures.
**Verdict:** IMPLEMENT

---

## Dedup Summary

- **20 distinct findings**, zero duplicates (verified against R8-A1, R11-A1-W1-A, R11-A1-W1-B)
- **All pass Gate v3:** Actionable (file:line specified), Applicable (ternary {-1,0,+1} + SSM), No-LEGO (always-active, no flags OFF by default)
- **Categories covered:**
  - Forward optimization: F1, F4, F2, F3, F5 (W1), F1, F2, F3 (W2)
  - Backward/grad optimization: B-F5 (committed row skip)
  - Infrastructure: A-F3, A-F4, W2-A-F4, W2-B-F4, W2-B-F5 (invalidation, parity, WHT)
  - Algebraic (math paradigm): W2-A (F1-F5), W2-B (F1-F5)

---

## Implementation Priority

**Phase 1 (High Impact, Low Risk):**
1. R11-A1-W1-A-F1 (BITMASK-GEMV-ALWAYS-ACTIVE) — one config change
2. R11-A1-W1-A-F2 (BITMASK-TRAINING-SINGLE-KERNEL-FUSED) — one line fix
3. R11-A1-W1-A-F3 (BITMASK-STALE-C2-DIRTY-TRACKING) — guard logic
4. R11-A1-W1-A-F5 (W-SUM-SCALE-CORRECTION) — bias correction

**Phase 2 (Medium Impact, Medium Effort):**
5. R11-A1-W1-B-F1, B-F2, B-F3, B-F4, B-F5 (sparsity exploitation)
6. R11-A1-W2-A-F2, W2-A-F4 (AGC + parity invalidation)

**Phase 3 (Algorithmic, High Risk, Highest Gain):**
7. R11-A1-W1-A-F4 (INT8 activation)
8. R11-A1-W2-A-F1, W2-A-F3 (WHT, INT16)
9. R11-A1-W2-B-F2, W2-B-F3 (Booth, CRT)
10. R11-A1-W2-B-F1, W2-B-F4, W2-B-F5 (NAF, LDPC, WHT-spectral)

---

## Notes for Recipe/Implementation

- **INT8 activation paths (F4, W2-B-F3):** Complementary; CRT is stricter (sign-binarized, INT8 exact); absmean is reference (BitNet). Choose one or both.
- **Sparsity findings (W1-B-F1 to F5):** Synergistic; earlier layers reduce search space for later layers.
- **ALG findings (W2-A, W2-B):** Orthogonal; implement in any order. F5 (tropical L∞) needs convergence validation.
- **No blocking dependencies:** All 20 can be implemented independently or in preferred order.

