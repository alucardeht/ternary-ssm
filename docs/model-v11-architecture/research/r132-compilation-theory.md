# R132 — Compilation Training: Complete Theory

**Date:** 2026-03-14
**Status:** Theory validated against code. Ready for implementation.
**Source analysis files:** `.workspace/analysis/r132-integrated-math.md`, `r132-compilation-theory-v2.md`, `r132-theory-validation.md`

---

## 1. Problem Statement

Given sequential layer-by-layer training (Forward Propagation), can we replace gradient descent for LINEAR projections with closed-form solutions, keeping GD only for NONLINEAR components?

**Layer k structure:**
```
H_{k+1} = H_k + out_proj(SSM(in_proj(LN(H_k))) * gate(LN(H_k)))
```

**Parameter split:**
- Linear (closed-form eligible, ~91-94% of params): W_in (in_proj), W_out (out_proj)
- Nonlinear (GD required, ~6-9% of params): A_log, x_proj, dt_proj, D

---

## 2. What FAILED and WHY (Proven)

### 2a. OLS with Self-Generated Targets is Tautological (Theorem)

If we hook (X, Y) pairs where Y = W_current @ X, then OLS recovers:

    W_OLS = (C_XX + λI)^{-1} C_XX × W_current = P_λ × W_current

This is a RIDGE SHRINKAGE of current weights. Contains zero new information. With rank@90% ≈ 17-20 and d=256, effective shrinkage factor is ~17/256 ≈ 0.07. Scale collapses: α_OLS ≈ 0.07 × α_P1.

**Proof:** Y^(b) = W @ X^(b) by construction. Substituting into normal equations:
X^T Y = X^T (WX) = C_XX W^T → W_OLS = W^T × C_XX × (C_XX + λI)^{-1}. QED.

### 2b. EM is Circular (Corollary)

E-step: forward with current weights → Y^(b) = W_current @ X^(b).
M-step: OLS from (X, Y) → recovers W_current (up to shrinkage).
Fixed point: W → 0 in null space of C_XX. Converges in 1-2 steps to scale-collapsed solution.

### 2c. Ternarization After OLS Causes Scale Collapse

OLS produces shrunken weights. Ternarize: sign(W_OLS). Scale α = mean(|W_OLS|) ≈ 0.07 × α_P1.
Effective weight ≈ 0.07 of original magnitude. Layer becomes near-identity. Cascades through N layers.

### 2d. These Three Anti-Patterns Are PROVEN HARMFUL
1. **OLS with self-generated targets** → tautological (Theorem 2a)
2. **EM iteration** → circular (Corollary 2b), converges to zero
3. **Ternarize after OLS** → scale collapse (Section 2c)

---

## 3. What WORKS: Natural Gradient (Non-Tautological)

### 3a. Breaking the Tautology

The tautology is broken by using an EXTERNAL signal: the loss gradient G^(b) = ∂L/∂H_{k+1}.

This gradient is already computed during backprop in Phase 1. It encodes information about the END-TO-END loss through all downstream layers — external to the current layer's weights.

### 3b. Natural Gradient Formula

For W_out (out_proj):

    W_out* = W_fp32 + η_NG × (C_{vv} + λI)^{-1} × C_{Gv}^T

Where:
- C_{vv} = (1/N) Σ_b v_k^T v_k — activation covariance (d_inner × d_inner)
- C_{Gv} = (1/N) Σ_b G^(b)^T v_k — gradient-activation cross-correlation (d_model × d_inner)
- v_k = SSM_output * gate — the input to out_proj
- G^(b) = ∂L/∂H_{k+1} — loss gradient at layer output

This is the EXACT one-step natural gradient update. Equivalent to 5-10 GD steps in one matrix solve. Non-tautological because G^(b) is external signal from CE loss.

For W_in (in_proj), analogous formula with u_k = LN(H_k) and G_in = gradient at in_proj output.

### 3c. CCA Compilation (Optional Enhancement)

CCA = Canonical Correlation Analysis between activations U_k and targets T.
SVD of C_{UT} = U_k^T T / N gives optimal projection directions.
These directions maximize information transfer from H_k to targets.

    W_out_CCA = M[:, 0:r_cca] × (C_{vv}+λI)^{-1} × C_{Gv}^T

This IS the "compiler": deterministic spectral decomposition against data distribution. No GD.

---

## 4. Code Mapping (Validated)

| Theory Construct | Code Location | How to Access |
|---|---|---|
| v_k (SSM out × gate) | `act_z` in ssm.py L540-546 / L1748-1804 | Hook: `out_proj.register_forward_hook(λ m,inp,out: store(inp[0]))` |
| G^(b) (∂L/∂H_{k+1}) | Output of ModelBlock.forward | Hook: `h_kp1.register_hook(λ grad: store(grad))` |
| W_fp32 shadow | `module.weight` (TernaryLinear) | Direct: `out_proj.weight.data += η * delta` |
| SSM params for GD | `A_log`, `x_proj`, `dt_proj`, `D` | Separate nn.Parameter, independently freezeable |
| Projections for closed-form | `in_proj`, `out_proj` | TernaryLinear instances |

**Memory cost of covariance matrices:**
- 6M (d=256, d_inner=512): C_{vv}=1MB, C_{Gv}=512KB. Total ~1.5MB.
- 7B (d=4096, d_inner=8192): C_{vv}=256MB, C_{Gv}=128MB. Total ~384MB. With PCA rank-20: trivial.

**Cholesky solve cost:**
- 6M: instant (<1ms)
- 7B direct: ~27-55 seconds (one per stage)
- 7B with PCA rank-20: instant (O(20³) = 8000 FLOPs)

---

## 5. The Integrated Pipeline

```
Per stage k (k = 0, 1, ..., N-1):

  ┌─ Phase 1: SSM Warmup (50 AdamW steps) ──────────────────────┐
  │ FREEZE: in_proj.weight, out_proj.weight                      │
  │ TRAIN:  A_log, x_proj, dt_proj, D (+ lm_head, final_norm)   │
  │ Deep gradients through frozen downstream layers k+1..N-1     │
  │ COLLECT (last 10 steps): C_{vv}, C_{Gv} via forward+backward │
  │   hooks on out_proj input (v_k) and layer output grad (G^b)  │
  └──────────────────────────────────────────────────────────────┘
             │
             ▼
  ┌─ Gate Check ─────────────────────────────────────────────────┐
  │ If loss_reduction < 3% in Phase 1: SKIP Phase 2             │
  └──────────────────────────────────────────────────────────────┘
             │
             ▼
  ┌─ Phase 2: Natural Gradient (one matrix solve) ───────────────┐
  │ For out_proj:                                                 │
  │   δ_out = solve(C_{vv} + λI, C_{Gv}^T)^T                   │
  │   out_proj.weight.data += η_NG × δ_out                      │
  │                                                               │
  │ Recompute v_k with updated out_proj (one forward pass)       │
  │                                                               │
  │ For in_proj:                                                  │
  │   δ_in = solve(C_{uu} + λI, C_{G_in,u}^T)^T                │
  │   in_proj.weight.data += η_NG × δ_in                        │
  │                                                               │
  │ Scale check: if ||δ|| > 2×||W||, REJECT (keep Phase 1 W)    │
  └──────────────────────────────────────────────────────────────┘
             │
             ▼
  ┌─ Phase 3: ELIMINATED ───────────────────────────────────────┐
  │ EM proven circular (Corollary 2b). Not executed.             │
  └──────────────────────────────────────────────────────────────┘
             │
             ▼
  ┌─ Phase 4: Joint Fine-Tuning (50 AdamW steps) ───────────────┐
  │ TRAIN: ALL params in layer k (+ lm_head, final_norm)         │
  │ Deep gradients through frozen downstream layers               │
  │ STE handles ternarization in forward pass automatically       │
  │ Cross-stage LR correction: η_k = η_base × σ^{-k}            │
  └──────────────────────────────────────────────────────────────┘
             │
             ▼
  ┌─ Post-Stage ─────────────────────────────────────────────────┐
  │ Pack layer k to INT2, move to CPU                             │
  │ Rebuild H_{k+1} cache with final weights (bf16, CPU)         │
  └──────────────────────────────────────────────────────────────┘

  Total per stage: 100 GD steps + 1 matrix solve
  Total GD: 100 × N (vs 150 × N in pure FP)
  Closed-form handles: ~91-94% of layer parameters
```

---

## 6. Key Design Decisions and Rationale

### 6a. Why Phase 1 Freezes Projections
SSM dynamics (A, B, C) are nonlinear — cannot be solved closed-form. Training SSM first with random projections gives a rough dynamics foundation. Natural gradient then optimally adjusts projections to work with the trained SSM.

### 6b. Why NO Ternarization in Phase 2
Phase 2 updates the fp32 SHADOW weights. The next forward pass (in Phase 4) will ternarize via STE. Ternarizing explicitly after OLS causes scale collapse (proven in Section 2c). Let STE handle it naturally.

### 6c. Why EM is Eliminated
EM with self-generated targets converges to scale-collapsed zero in 1-2 iterations (proven in Section 2b). With external-signal natural gradient, the update is one-shot and exact. No iteration needed.

### 6d. Why Collect on LAST 10 Steps
Early training steps have high-variance gradients. Last 10 steps are near-converged, giving stable C_{vv} and C_{Gv} estimates. 10 steps × batch_size × seq_len gives ~5000-20000 samples — sufficient for rank-20 subspace.

### 6e. Cross-Stage Coupling
Phase 1 trains SSM with random projections → Phase 2 adjusts projections → Phase 4 fine-tunes jointly. The coupling is handled by Phase 4's joint optimization. Phase 2 provides a better starting point (natural gradient optimal direction) than random projections for Phase 4.

---

## 7. Expected Comparison

| Metric | R130 (pure FP) | R132 (compilation) |
|---|---|---|
| GD steps per stage | 150 | 100 (50 SSM + 50 joint) |
| Closed-form ops | 0 | 1 Cholesky solve per stage |
| Params by GD | 100% | ~6-9% (Phase 1) then 100% (Phase 4) |
| VRAM overhead | PCA-INT8 cache | +1.5MB (6M) / +384MB (7B) for covariance |
| Expected quality | 2.43 nats (stage 0) | ≥ 2.43 nats (natural gradient adds information) |

---

## 8. Remaining Open Questions

1. **η_NG step size:** Theory says "equivalent to 5-10 GD steps" but exact η_NG needs tuning. Start with η_NG = 1.0 (full natural gradient step) and scale down if diverges.

2. **50 GD steps enough for SSM warmup?** SSM has ~6-9% of params. 50 steps on fewer params should suffice, but verify empirically.

3. **Phase 1→Phase 2 coupling:** SSM trained on random projections. After Phase 2 changes projections, SSM dynamics are stale. Phase 4 handles this but initial mismatch may slow convergence. Monitor SSM param changes in Phase 4 vs Phase 1.

4. **CCA enhancement:** Optional. Natural gradient is the minimum viable. CCA adds optimal direction selection. Implement CCA only after natural gradient is confirmed working.
