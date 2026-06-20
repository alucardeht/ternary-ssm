# R11-A3 — Ternary Quantization Efficiency Synthesis

**Round:** R11 | **Aspect:** A3 | **Waves:** 1-2 (4 agents) | **Date:** 2026-03-06
**Total findings:** 19 (1 superseded) | **All pass Gate v3**

---

## Summary Table

| ID | Gain | Source | Verdict |
|----|------|--------|---------|
| W2-A-F2 | 20-30% | Bayesian threshold | IMPLEMENT |
| W1-B-F1 | 35-45% | Committed skip | IMPLEMENT |
| W2-B-F1 | 6-14% | IFT coupling | IMPLEMENT |
| W2-B-F3 | 5-12% | Bilevel hypergradient | IMPLEMENT |
| W1-A-F2 | 8-15% | MSE-optimal init | IMPLEMENT |
| W2-A-F1 | 4-8% | Boundary score | IMPLEMENT |
| W2-B-F2 | 4-10% | Margin-proportional STE | IMPLEMENT |
| W1-A-F1 | 5-12% | Sqrt-N scale norm | IMPLEMENT |
| W1-A-F4 | 6-12% | Scale EMA smooth | IMPLEMENT |
| W2-A-F3 | 5-10% | MLE scale correction | IMPLEMENT |
| W2-A-F4 | 3-6% | Order-stat EMA | IMPLEMENT |
| W2-B-F4 | 2-6% | Lion projection | IMPLEMENT |
| W2-A-F5 | 6-10% | Fisher-weighted EMA | IMPLEMENT |
| W2-B-F5 | 2-5% | Manifold projection | IMPLEMENT |
| W1-A-F3 | 5-10% | Group mean offset | IMPLEMENT |
| W1-A-F5 | 4-8% | Committed scale grad zero | IMPLEMENT |
| W1-B-F2 | 20-30% | Kthvalue (SUPERSEDED) | SKIP |
| W1-B-F3 | 8-12% | STE backward fuse | IMPLEMENT |
| W1-B-F4 | 2-3% | Lazy commit | IMPLEMENT |
| W1-B-F5 | 1-2% | Sparse grad compute | IMPLEMENT |

---

## IMPLEMENT Findings (18)

### W2-A-F2: BAYESIAN-POSTERIOR-OPTIMAL-HALF-SCALE-THRESHOLD
**ID:** R11-A3-W2-A-F2 | **Source:** W2-AgA | **Gain:** 20-30% forward + 3-6% step
**File:** `src/model_v7/components/ternary.py` line ~415
**Principle:** Laplace MAP threshold is exactly 0.5 in normalized coordinates. Eliminates sort+cumsum pipeline entirely (O(K) vs O(K log K)).
**Pseudocode:**
```python
BASE_THRESHOLD = 0.5
if sigma_g is not None:
    sg_grouped = sigma_g.reshape(-1, group_size).mean(dim=-1)
    sg_rel = sg_grouped / sg_grouped.mean().clamp(min=1e-8)
    threshold = (0.5 / (1.0 + 0.3 * sg_rel)).unsqueeze(-1)
```
**Verdict:** IMPLEMENT — Replaces energy threshold block. No flag.

---

### W1-B-F1: COMMITTED-GROUP-QUANTIZE-SKIP
**ID:** R11-A3-W1-B-F1 | **Source:** W1-AgB | **Gain:** 35-45% quantize time, ~5-8% e2e
**File:** `src/model_v7/components/ternary.py` line ~393
**Principle:** Fully-committed groups (59% at current state) have fixed ternary assignments. Cache and reuse, skip sort+cumsum for those groups.
**Pseudocode:**
```python
if committed_mask is not None:
    committed_groups = committed_mask_grouped.all(dim=-1)
    active = ~committed_groups
    # Full pipeline only on uncommitted groups (~41%)
```
**Verdict:** IMPLEMENT — Conditional skip in forward. No flag.

---

### W2-B-F1: IFT-SCALE-POST-STEP-WEIGHT-COUPLING-CORRECTION
**ID:** R11-A3-W2-B-F1 | **Source:** W2-AgB | **Gain:** 6-14% step reduction
**File:** `src/model_v7/components/ternary.py` line ~2596 (post-step hook)
**Principle:** After weight update, optimal scale shifts by Δs* = -mean(Δw)/s. Apply IFT correction immediately to eliminate 1-3 step lag.
**Pseudocode:**
```python
dw = weight.data - prev_weight
dw_grouped = dw.reshape(-1, group_size)
dw_mean_pos = (dw_grouped * pos_mask.float()).sum(-1) / n_pos
lsq_scale_pos.data -= dw_mean_pos / lsq_scale_pos.abs().clamp(min=1e-8)
```
**Verdict:** IMPLEMENT — Post-step hook. No flag.

---

### W2-B-F3: BILEVEL-HYPERGRADIENT-WEIGHT-CORRECTION
**ID:** R11-A3-W2-B-F3 | **Source:** W2-AgB | **Gain:** 5-12% step reduction (early training)
**File:** `src/model_v7/components/ternary.py` line ~492 (backward)
**Principle:** Weight gradient corrected for scale coupling: Δgrad_w_i = go_i * (grad_sp_g * s_g) / n_nonzero_pos_g.
**Pseudocode:**
```python
bilevel_corr_pos = (grad_sp * sp.squeeze(-1)) / n_nonzero_pos.clamp(min=1)
bilevel_flat_g = pos_mask.float() * bilevel_corr_pos.unsqueeze(-1)
grad_weight = grad_weight + go_flat * bilevel_flat
```
**Verdict:** IMPLEMENT — Extends backward. No flag.

---

### W1-A-F2: MSE-OPTIMAL-SCALE-INIT-RESET
**ID:** R11-A3-W1-A-F2 | **Source:** W1-AgA | **Gain:** 8-15% step reduction (early)
**File:** `src/model_v7/components/ternary.py` line ~1759 and ~2166
**Principle:** TWN/QuEST result: MSE-optimal scale s* = mean(|w_nonzero|). Reset on de-commit instead of max-abs init.
**Pseudocode:**
```python
pos_vals = grouped_init.clamp(min=0)
_init_sp = pos_vals.sum(-1) / pos_vals.gt(0).float().sum(-1).clamp(min=1)
_init_sp = _init_sp.clamp(min=1e-3)
```
**Verdict:** IMPLEMENT — Replace initialization path. No flag.

---

### W2-A-F1: BOUNDARY-SCORE-AUGMENTED-LSQ-GRAD
**ID:** R11-A3-W2-A-F1 | **Source:** W2-AgA | **Gain:** 4-8% step reduction
**File:** `src/model_v7/components/ternary.py` line ~494 (backward)
**Principle:** Complete score function includes boundary term from zero-assigned weights near threshold. Corrects scale underestimation bias.
**Pseudocode:**
```python
zero_mask = ~pos_mask & ~neg_mask
near_pos = (normalized.abs() - 0.5).abs() < 0.15
boundary_pos = (zero_mask & near_pos).float()
grad_sp = grad_sp + (go_grouped * boundary_pos).sum(dim=-1) * 0.1
```
**Verdict:** IMPLEMENT — Extends backward. No flag.

---

### W2-B-F2: MARGIN-PROPORTIONAL-STE-ALPHA-CURVATURE
**ID:** R11-A3-W2-B-F2 | **Source:** W2-AgB | **Gain:** 4-10% step reduction
**File:** `src/model_v7/components/ternary.py` line ~483 (backward)
**Principle:** STE alpha should reflect margin curvature: low margin → high alpha, high margin → low alpha.
**Pseudocode:**
```python
margin = ctx.saved_tensors[...]
alpha_i = ols_alpha_base * (1.0 - (margin / (sp_flat * m_commit)).clamp(0, 1))
grad_weight = grad_output * alpha_i
```
**Verdict:** IMPLEMENT — Element-wise alpha in backward. No flag.

---

### W1-A-F1: SQRT-N-SCALE-GRADIENT-NORMALIZATION
**ID:** R11-A3-W1-A-F1 | **Source:** W1-AgA | **Gain:** 5-12% step reduction
**File:** `src/model_v7/components/ternary.py` line ~494
**Principle:** LSQ paper: scale gradient / sqrt(n_nonzero_per_group) prevents magnitude collapse and cross-layer LR sensitivity.
**Pseudocode:**
```python
n_nonzero_pos = pos_mask.float().sum(dim=-1).clamp(min=1.0)
grad_sp = (go_grouped * pos_mask.float()).sum(dim=-1) / n_nonzero_pos.sqrt()
```
**Verdict:** IMPLEMENT — Extends gradient computation. No flag.

---

### W1-A-F4: SCALE-EMA-SMOOTHING-REPLACES-GRADIENT-UPDATE
**ID:** R11-A3-W1-A-F4 | **Source:** W1-AgA | **Gain:** 6-12% step reduction
**File:** `src/model_v7/components/ternary.py` line ~1765 and ~2596
**Principle:** Scale gradients are noisy (computed over ~51 weights per group). EMA shadow (β=0.9) reduces oscillation. Use EMA in forward, not raw param.
**Pseudocode:**
```python
self.register_buffer("lsq_scale_pos_ema", _init_sp.clone())
with torch.no_grad():
    m.lsq_scale_pos_ema.mul_(0.9).add_(m.lsq_scale_pos.abs(), alpha=0.1)
sp = lsq_scale_pos_ema.clamp(min=1e-8).unsqueeze(-1)
```
**Verdict:** IMPLEMENT — Register buffer + post-step hook. No flag.

---

### W2-A-F3: NONZERO-FRACTION-MLE-SCALE-CORRECTION
**ID:** R11-A3-W2-A-F3 | **Source:** W2-AgA | **Gain:** 5-10% step reduction
**File:** `src/model_v7/components/ternary.py` line ~2594 (post-quantize in forward)
**Principle:** MLE: s_MLE = -2*log(f_nz)*mean(|w_nonzero|). Corrects scale drift every step with β=0.05.
**Pseudocode:**
```python
f_nz = (grouped.abs() > 0.05).float().mean(dim=-1).clamp(min=0.05, max=0.95)
mean_nz = (grouped.abs() * nz_mask.float()).sum(-1) / nz_mask.float().sum(-1).clamp(min=1)
s_mle = -2.0 * f_nz.log() * mean_nz
lsq_scale_pos.data.lerp_(s_mle.clamp(min=1e-4), 0.05)
```
**Verdict:** IMPLEMENT — Post-quantize correction. No flag.

---

### W2-A-F4: ORDER-STAT-VARIANCE-ADAPTIVE-EMA
**ID:** R11-A3-W2-A-F4 | **Source:** W2-AgA | **Gain:** 3-6% step reduction
**File:** `src/model_v7/components/ternary.py` line ~432 (after energy_threshold)
**Principle:** Threshold variance is inversely proportional to local density. High density → low EMA (fast); low density → high EMA (slow smoothing).
**Pseudocode:**
```python
near_thr = ((abs_normalized - energy_threshold.unsqueeze(-1)).abs() < 0.1).float()
local_density = near_thr.mean(dim=-1) / 0.2
order_stat_var = 0.4 * (1 - 0.4) / (group_size * local_density.clamp(min=1e-3)**2)
beta_g = (1.0 - order_stat_var.clamp(max=0.9)).clamp(min=0.05)
smoothed_thr = beta_g * ctx._thr_ema + (1 - beta_g) * energy_threshold
```
**Verdict:** IMPLEMENT — Adaptive EMA in forward. No flag.

---

### W2-B-F4: LION-SCALE-POSITIVE-CONE-PROJECTION
**ID:** R11-A3-W2-B-F4 | **Source:** W2-AgB | **Gain:** 2-6% step reduction
**File:** `src/model_v7/train.py` line ~3698 (post optimizer.step)
**Principle:** Scale constrained to R+. After Lion step, project to max(s, 1e-4) and zero momentum if boundary hit.
**Pseudocode:**
```python
hit_boundary = param.data < 1e-4
param.data.clamp_(min=1e-4)
if hit_boundary.any():
    state = optimizer.state.get(param, {})
    if 'exp_avg' in state:
        state['exp_avg'][hit_boundary] = 0.0
```
**Verdict:** IMPLEMENT — Post-step constraint projection. No flag.

---

### W2-A-F5: FISHER-WEIGHTED-SCALE-EMA
**ID:** R11-A3-W2-A-F5 | **Source:** W2-AgA | **Gain:** 6-10% step reduction
**File:** `src/model_v7/components/ternary.py` line ~1765 (post-optimizer hook)
**Principle:** Fisher information for scale is I(s) = n_nz * λ² / (1-f_nz). EMA coefficient β_t ∝ I_t. High information → fast update; low info → slow.
**Pseudocode:**
```python
f_nz = (grouped.abs() > 0.05).float().mean(dim=-1)
fi_weight = (4.0 * f_nz * (1.0 - f_nz)).clamp(min=0.02)
beta = 1.0 - fi_weight * 0.5
lsq_scale_pos_ema.mul_(beta).add_(lsq_scale_pos.abs(), alpha=1-beta)
```
**Verdict:** IMPLEMENT — Per-group adaptive EMA post-optimizer. No flag.

---

### W2-B-F5: COMMITTED-WEIGHT-MANIFOLD-PROJECTION
**ID:** R11-A3-W2-B-F5 | **Source:** W2-AgB | **Gain:** 2-5% effective (fewer de-commit events)
**File:** `src/model_v7/components/ternary.py` line ~2596 (post-step)
**Principle:** Lion momentum can de-commit weights spuriously. Project committed weights back: committed-to-+1 → clamp(min=T+eps).
**Pseudocode:**
```python
T = self._current_threshold
committed_pos = (self.committed_mask & (self._last_ternary > 0)).reshape(-1)
w = self.weight.data.reshape(-1)
w[committed_pos] = w[committed_pos].clamp(min=T + 1e-3)
```
**Verdict:** IMPLEMENT — Post-step projection. No flag.

---

### W1-A-F3: PER-GROUP-MEAN-OFFSET-BIAS-CORRECTION
**ID:** R11-A3-W1-A-F3 | **Source:** W1-AgA | **Gain:** 5-10% step reduction
**File:** `src/model_v7/components/ternary.py` line ~393 (forward)
**Principle:** Ternary quantization has systematic bias when group mean is nonzero. Center per group, quantize, add back offset.
**Pseudocode:**
```python
group_mean = grouped.mean(dim=-1, keepdim=True)
grouped_centered = grouped - group_mean
normalized = grouped_centered / s_norm
# ... threshold + quantize logic ...
w_asymmetric = pos_mask.float() * sp - neg_mask.float() * sn + group_mean
```
**Verdict:** IMPLEMENT — Offset in quantize path. No flag.

---

### W1-A-F5: COMMITTED-GROUP-SCALE-GRAD-ZERO
**ID:** R11-A3-W1-A-F5 | **Source:** W1-AgA | **Gain:** 4-8% step reduction (scale params)
**File:** `src/model_v7/components/ternary.py` line ~492 (backward)
**Principle:** Fully-committed groups produce zero weight gradients. Their scale gradients are pure noise. Zero grad_sp/grad_sn for committed groups before optimizer.
**Pseudocode:**
```python
cm_grouped = committed_mask.reshape(-1, group_size)
group_committed = cm_grouped.all(dim=-1)
grad_sp = grad_sp * (~group_committed).float()
grad_sn = grad_sn * (~group_committed).float()
```
**Verdict:** IMPLEMENT — Mask in backward. No flag.

---

### W1-B-F3: STE-BACKWARD-COMMITTED-MASK-ZERO-FUSED
**ID:** R11-A3-W1-B-F3 | **Source:** W1-AgB | **Gain:** 8-12% backward speedup, ~2-4% e2e
**File:** `src/model_v7/components/ternary.py` line ~483 (backward)
**Principle:** STE backward multiplies by alpha, then hook zeros committed weights. Fuse into single masked multiply.
**Pseudocode:**
```python
active_mask = (~committed_mask_flat).float() if committed_mask else 1.0
grad_weight = grad_output * ols_alpha * active_mask
```
**Verdict:** IMPLEMENT — Single-op STE backward. No flag.

---

### W1-B-F4: UPDATE-COMMITMENT-STATE-LAZY
**ID:** R11-A3-W1-B-F4 | **Source:** W1-AgB | **Gain:** 2-3% e2e
**File:** `src/model_v7/components/ternary.py` line ~2172 (update_commitment_state)
**Principle:** Committed weights' margin is stable (gradient is zero anyway). Skip margin recomputation for ~59% of weights, only update ~41% uncommitted.
**Pseudocode:**
```python
active = ~self.committed_mask
flipped = torch.zeros_like(self.committed_mask)
flipped[active] = (w_quantized[active] != self.prev_ternary_c2[active])
self.n_stable[active] = torch.where(flipped[active], 0, self.n_stable[active]+1)
```
**Verdict:** IMPLEMENT — Masked update in commitment state. No flag.

---

### W1-B-F5: SPARSE-SCALE-GRAD-COMMITTED-GROUP-SKIP
**ID:** R11-A3-W1-B-F5 | **Source:** W1-AgB | **Gain:** 1-2% e2e
**File:** `src/model_v7/components/ternary.py` line ~492 (backward)
**Principle:** Committed weights get zero gradients. Their groups' scale gradients are zero. Skip masked-sum for committed groups.
**Pseudocode:**
```python
active_groups = ~committed_groups_mask
grad_sp_full[active_groups] = (go_active * pos_mask[active_groups].float()).sum(-1)
```
**Verdict:** IMPLEMENT — Sparse reduction in backward. No flag.

---

## SKIP Finding (1)

### W1-B-F2: KTHVALUE-THRESHOLD (SUPERSEDED)
**ID:** R11-A3-W1-B-F2 | **Source:** W1-AgB | **Verdict:** SKIP
**Reason:** Subsumed by W2-A-F2. W2-A-F2 eliminates sort+cumsum entirely via Bayesian MAP threshold = 0.5 (O(K) scalar arithmetic). W1-B-F2 only replaces sort with kthvalue (still O(K) but no improvement over the Bayesian approach). **Use W2-A-F2 instead.**

---

## Implementation Priority

**Batch 1 (forward critical path):**
1. W2-A-F2 — Bayesian threshold (replaces 415-434 block entirely)
2. W1-B-F1 — Committed skip (conditional cache)
3. W1-A-F3 — Group mean offset (in LSQTernaryQuantize.forward ~393)

**Batch 2 (backward + scale params):**
4. W2-B-F1 — IFT coupling (post-step)
5. W2-B-F3 — Bilevel hypergradient (backward extend)
6. W1-A-F2 — MSE-optimal init (init + de-commit paths)
7. W2-A-F1 — Boundary score (backward extend)

**Batch 3 (EMA + optimizer):**
8. W1-A-F4 — Scale EMA buffer + post-step hook
9. W2-A-F3 — MLE correction (post-quantize)
10. W2-A-F5 — Fisher EMA (post-step hook)

**Batch 4 (efficiency + margins):**
11. W1-A-F1 — Sqrt-N normalization (backward)
12. W2-B-F2 — Margin-proportional alpha (backward)
13. W1-A-F5 — Committed scale grad zero (backward)
14. W1-B-F3 — Fused STE backward (backward)
15. W1-B-F4 — Lazy commitment (commitment state)
16. W1-B-F5 — Sparse scale grad (backward)

**Batch 5 (constraints + projection):**
17. W2-A-F4 — Order-stat EMA (forward)
18. W2-B-F4 — Lion projection (post-step)
19. W2-B-F5 — Manifold projection (post-step)

