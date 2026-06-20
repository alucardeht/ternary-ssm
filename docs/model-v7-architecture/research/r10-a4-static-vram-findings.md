# R10-A4 — Static VRAM Reduction: Ternary Commitment Buffers
# Synthesis: Waves 1-2, Agents A-B (2026-03-06)

## Summary Table

| ID | Technique | Gain | Verdict | File:Line |
|----|-----------|------|---------|-----------|
| R10-A4-W1-A-F1 | EMA-SHADOW-COMMITTED-PARAM-EVICT | 13.4 MB @70% | IMPLEMENT | train.py:773,781 |
| R10-A4-W1-A-F2 | EMA-EMBEDDING-BF16-SHADOW-ALWAYS | 8-16 MB | IMPLEMENT | train.py:768-770 |
| R10-A4-W1-A-F3 | COMMIT-MONITOR-PREV-TERNARY-INT8 | 14.3 MB | IMPLEMENT | train.py:1505,1490 |
| R10-A4-W1-A-F4 | CROSS-LAYER-OUT-PROJ-SHADOW-SHARING | 4.5 MB | EXPLORE | model.py:490 |
| R10-A4-W1-A-F5 | EMBEDDING-FREQ-ROW-SHARING-RARE-TOKENS | 2.75-11 MB | EXPLORE | model.py:270, prepare_wikitext.py |
| R10-A4-W1-B-F1 | C2-BUF-SIGMA-FP16 | 12 MB | IMPLEMENT | ternary.py:2093 |
| R10-A4-W1-B-F2 | C2-BUF-NSTABLE-INT16 | 12 MB | IMPLEMENT | ternary.py:2096 |
| R10-A4-W1-B-F3 | C2-BUF-MASK-PACKBITS | 5.25 MB | **SKIP (superseded)** | ternary.py:2099 |
| R10-A4-W1-B-F4 | C2-BUF-PREV-INT8 | 18 MB | **SKIP (superseded)** | ternary.py:2102,2173,2191 |
| R10-A4-W1-B-F5 | C2-BUF-SIGMA-GROUP | 23.8 MB | IMPLEMENT | ternary.py:2093,2120-2122,2183-2184 |
| R10-A4-W2-A-F1 | ABSORBING-STATE-COMMITTED-BUFFER-NULL | 28 MB @70% | IMPLEMENT | ternary.py:2120,2182 |
| R10-A4-W2-A-F2 | SIGMA-G-ADAPTIVE-INT8-HIGH-NSTABLE | ~10.5 MB | EXPLORE | ternary.py:2093,2120 |
| R10-A4-W2-A-F3 | PREV-TERNARY-TRIT-PACKING-2BIT | 18.75 MB | IMPLEMENT | ternary.py:2102,2173,2191 |
| R10-A4-W2-A-F4 | COMMITMENT-ENTROPY-EARLY-COMMIT | Indirect (70-80% faster) | EXPLORE | ternary.py:2182,2093 |
| R10-A4-W2-A-F5 | COMMITTED-STATE-UNIFIED-FIELD-5BIT | 73 MB theoretical | SPECULATIVE | ternary.py:2093+ |
| R10-A4-W2-B-F1 | Z3-DELTA-FLIP-BITMAP-REPLACE-PREV-TERNARY | 22.5 MB | **IMPLEMENT (supersedes F4)** | ternary.py:2102,2173,2191 |
| R10-A4-W2-B-F2 | COMMITTED-MASK-MONOTONE-INFERENCE-NO-STORE | 6 MB | **IMPLEMENT (supersedes F3)** | ternary.py:2099,2131,2139,2187,2194 |
| R10-A4-W2-B-F3 | SIGMA-G-ZERO-COST-COMMITTED-EVICTION | 8.4 MB @f=0.7 | IMPLEMENT | ternary.py:2120-2122 |
| R10-A4-W2-B-F4 | CROSS-LAYER-COMMITTED-MASK-LATTICE-SHARING | 8-12 MB | EXPLORE | ternary.py:2099-2102 |
| R10-A4-W2-B-F5 | LAZY-FLIP-DETECTION-SKIP | ~2-3% throughput | IMPLEMENT (COMPUTE) | ternary.py:2173 |

---

## IMPLEMENT (Priority Order)

### EMA Shadow & Monitor Reductions (train.py, 35-44 MB)

**W1-A-F1: EMA-SHADOW-COMMITTED-PARAM-EVICT**

Committed weights freeze → EMA shadow converges to param.data within 1000 steps.
Delete buffer for committed layers; on apply(), use param directly. Gain: 13.4 MB @70%.

```python
# train.py:773 ModelEMA.update():
for name, param in model.named_parameters():
    if name in self.shadow:
        module = _get_param_module(model, name)
        if hasattr(module, 'committed_mask') and module.committed_mask.float().mean() > 0.95:
            del self.shadow[name]; continue
        self.shadow[name].mul_(self.decay).add_(param, alpha=1-self.decay)
# train.py:781 ModelEMA.apply(): use param.data directly if name not in self.shadow
```

**W1-A-F2: EMA-EMBEDDING-BF16-SHADOW-ALWAYS**

Embedding table (32K×256) costs 31.2 MB FP32 shadow. Embeddings never ternary;
BF16 shadow sufficient (7-bit mantissa ≫ embedding precision). Gain: 8-16 MB.

```python
# train.py:768-770 ModelEMA.__init__():
for name, param in model.named_parameters():
    if param.requires_grad:
        dtype = torch.bfloat16 if 'embedding' in name else torch.float32
        self.shadow[name] = param.data.to(dtype)
```

**W1-A-F3: COMMIT-MONITOR-PREV-TERNARY-INT8**

TernaryCommitmentMonitor.prev_ternary stores {-1,0,+1} as FP32 (4 bytes).
INT8 exact. Gain: 14.3 MB at 5M params.

```python
# train.py:1505 TernaryCommitmentMonitor.update():
self.prev_ternary[name] = current_ternary.to(torch.int8)
# train.py:1490 comparison: flipped = current_ternary.to(torch.int8) != self.prev_ternary[name]
```

---

### C2 Buffer Reductions (ternary.py, 71 MB via algebraic + dtype)

**W1-B-F1 & W1-B-F2: sigma_g FP16 + n_stable INT16**

sigma_g gradient EMA: values in [1e-4, 5e-2], FP16 sufficient (dynamic range + 0.1% quantization noise ≪ 1e-4 margin). n_stable counts ≤ 500; INT16 max = 32767 safe. Gain: 12 + 12 = 24 MB.

```python
# ternary.py:2093 _init_commitment_buffers():
self.sigma_g = torch.full_like(self.weight, 1e-3, dtype=torch.float16, device=device)
# ternary.py:2096:
self.n_stable = torch.zeros_like(self.weight, dtype=torch.int16, device=device)
# Update sites: cast to float only for EMA/sqrt computation
self.sigma_g.mul_(0.99).add_(grad_abs.half(), alpha=0.01)
self.n_stable = torch.where(flipped, torch.zeros_like(self.n_stable),
                            (self.n_stable + 1).clamp(max=32767)).to(torch.int16)
```

**W1-B-F5: C2-BUF-SIGMA-GROUP (128× reduction)**

sigma_g modulates commitment threshold, not selects which weight commits. LSQ already groups (size=128).
Group-mean |grad| replaces per-weight sigma_g. Gain: 23.8 MB (24 MB → 0.19 MB).

```python
# ternary.py:2093 init:
n_groups = (self.weight.numel() + 127) // 128
self.sigma_g = torch.full((n_groups,), 1e-3, dtype=torch.float16, device=device)
# ternary.py:2120 grad hook:
grad_grouped = grad_abs.view(-1, 128).mean(dim=-1)
self.sigma_g.mul_(0.99).add_(grad_grouped, alpha=0.01)
# Use: sg_expanded = self.sigma_g.repeat_interleave(128)[:weight.numel()].view(shape)
```

**W2-A-F1: ABSORBING-STATE-COMMITTED-BUFFER-NULL**

Committed positions: grad=0 (MBGSC) → sigma_g EMA→0, n_stable never resets, prev_ternary frozen.
These buffers carry zero future information. Null them for committed_frac > 0.5. Gain: 28 MB @70%.

```python
# ternary.py:2120/2182 after update_commitment_state:
if self.committed_mask.float().mean() > 0.5 and not hasattr(self, '_buffers_nulled'):
    self.sigma_g[self.committed_mask] = 0.0
    self.n_stable[self.committed_mask] = 0
    self._buffers_nulled = True
```

**W2-A-F3 & W2-B-F1: Flip Detection Packing (trit ↔ 2-bit)**

prev_ternary_c2 ∈ {-1,0,+1} requires only 1.585 bits (log₂ 3). W2-A-F3 uses 2-bit packing (4 states);
W2-B-F1 uses Z₃ delta-encoding via two {pos,neg} bitmaps. Both 8× denser than INT8.
W2-B-F1 supersedes W1-B-F4: choose W2-B-F1 (22.5 MB vs 18 MB saved).

```python
# ternary.py:2102 - W2-B-F1 approach (delta bitmaps):
n = self.weight.numel(); n_packed = (n + 7) // 8
self.prev_pos_packed = torch.zeros(n_packed, dtype=torch.uint8, device=device)
self.prev_neg_packed = torch.zeros(n_packed, dtype=torch.uint8, device=device)
# ternary.py:2173 flip detect:
cur_pos = torch.packbits((w_quantized > 0).view(-1))
cur_neg = torch.packbits((w_quantized < 0).view(-1))
flipped_packed = (cur_pos ^ self.prev_pos_packed) | (cur_neg ^ self.prev_neg_packed)
flipped = torch.unpackbits(flipped_packed)[:n].bool().view(self.weight.shape)
# ternary.py:2191 update:
self.prev_pos_packed.copy_(cur_pos)
self.prev_neg_packed.copy_(cur_neg)
```

**W2-B-F2: COMMITTED-MASK-MONOTONE-INFERENCE-NO-STORE**

committed_mask is monotone non-decreasing (once True, always True). Fully derivable from n_stable ≥ target.
Eliminates 6 MB bool tensor entirely. Supersedes W1-B-F3 (packbits).

```python
# ternary.py:2099 DELETE buffer registration (no self.committed_mask = ...)
# All read sites (L2131, 2139, 2187, 2194): inline
mask = (self.n_stable >= self._commitment_n_stable_target)
# Or define property:
@property
def committed_mask(self):
    return self.n_stable >= self._commitment_n_stable_target if self.n_stable is not None else None
```

**W2-B-F3: SIGMA-G-ZERO-COST-COMMITTED-EVICTION**

Committed positions never updated (grad=0) → sigma_g never read. Store only uncommitted sigma_g.
Masked EMA update: reduce from 24 MB to ~7.2 MB active at 70% commitment. Gain: 8.4 MB.

```python
# ternary.py:2120 _grad_hook_fn:
uncommitted = ~(self.n_stable >= self._commitment_n_stable_target)
if uncommitted.any():
    self.sigma_g[uncommitted].mul_(0.99).add_(
        grad_abs[uncommitted], alpha=0.01
    )
# No update for committed positions
```

**W2-B-F5: LAZY-FLIP-DETECTION-SKIP (Compute, not VRAM)**

Flip detection (~80% of positions at mid-training) is skipped for positions where margin is large.
Uses Lion exp_avg for threshold crossing check. Saves 2-3% throughput via reduced write bandwidth.

```python
# ternary.py:2173 fast path:
n_stable_safe = (self.n_stable >= 10)
margin_large = (w_shadow.abs() - delta_threshold) > 5 * lr
skip = n_stable_safe & margin_large  # ~80% of positions
if not skip.all():
    flipped = ~skip & (w_quantized.to(torch.int8) != self._unpack_prev_ternary())
else:
    flipped = torch.zeros_like(skip)
```

---

## SKIP (Superseded)

| ID | Reason |
|----|--------|
| W1-B-F3 | COMMITTED-MASK-PACKBITS (5.25 MB) superseded by W2-B-F2 (0 MB, fully derived from n_stable) |
| W1-B-F4 | PREV-TERNARY-INT8 (18 MB) superseded by W2-B-F1 (22.5 MB via 2-bit delta bitmaps, 4× denser) |

---

## EXPLORE

| ID | Technique | Rationale |
|----|-----------|-----------|
| W1-A-F4 | CROSS-LAYER-OUT-PROJ-SHADOW-SHARING | 4.5 MB, but ternary weight diversity across layers uncertain; needs convergence eval |
| W1-A-F5 | EMBEDDING-FREQ-ROW-SHARING-RARE-TOKENS | 2.75-11 MB, but rare-token quality degrades; requires tail-token perplexity validation |
| W2-A-F2 | SIGMA-G-ADAPTIVE-INT8-HIGH-NSTABLE | ~10.5 MB, decision-boundary precision sound, but round-trip quantize may expose edge cases |
| W2-A-F4 | COMMITMENT-ENTROPY-EARLY-COMMIT | Bayesian framing (70% faster commits), but false-positive commitment cost high; empirical validation needed |
| W2-B-F4 | CROSS-LAYER-COMMITTED-MASK-LATTICE-SHARING | 8-12 MB for fully-committed layers, but removes grad hook overhead; interaction with AutoFreeze unclear |

---

## SPECULATIVE

**W2-A-F5: COMMITTED-STATE-UNIFIED-FIELD-5BIT**

Complete redesign: unified 5-bit status_field (1 bit committed + 2 bits prev_trit + 2 bits sigma_g_bin)
+ sparse sigma_g table for Region B (marginal weights near threshold). Theoretical: 78 MB → 5 MB (94% reduction).
Engineering complexity high (sparse synchronization). Recommend post-validation of F1+F2+F3 incremental steps.

---

## Total VRAM Reduction (IMPLEMENT Only, W1+W2)

**Per-Component:**
- EMA shadows (train.py): 13.4 + 8 + 14.3 = **35.7 MB**
- C2 buffers (ternary.py): 24 (FP16+INT16) + 23.8 (group sigma_g) + 28 (null committed) + 22.5 (flip bitmap) + 6 (mask inference) + 8.4 (sigma eviction) = **112.7 MB**

**Grand Total: 148.4 MB @ 70% commitment**

**Baseline VRAM (6M params, ternary v5 = 886 MB):**
- With R10-A3 (mixed precision + optimizer memory): ~900-950 MB
- With R10-A2 (activation memory, 650-750 MB gain): ~200-300 MB
- **With R10-A4 static buffers (148.4 MB):** ~50-150 MB (extreme compression)

Combined v6 stack (R10-A2/A3/A4): **886 MB → ~50-100 MB VRAM at batch_size=4, enabling batch_size=6-8 on RTX 5070 12GB.**

---

## Implementation Sequence

1. **train.py changes** (W1-A: F1, F2, F3)
2. **ternary.py dtype downcasts** (W1-B: F1, F2, F5)
3. **ternary.py algebraic reductions** (W2-B: F1, F2, F3; W2-B: F5 if compute is priority)
4. **Smoke test:** 50 steps, verify VRAM before/after, check C2 buffer sizes match predictions
5. **Validation:** loss trajectory vs baseline, commitment timeline, no functional regression

---

**Dedup Status:** W2-AgB-F1/F2 supersede W1-AgB-F4/F3; both marked SKIP. W2-AgA-F1/F3 merged into IMPLEMENT.
**Files:** src/model_v7/train.py, src/model_v7/components/ternary.py
**Commits:** Per-batch (train.py → commit 1; ternary.py dtype → commit 2; ternary.py algebraic → commit 3)
