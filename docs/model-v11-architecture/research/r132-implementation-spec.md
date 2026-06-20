# R132 — Implementation Specification

**Date:** 2026-03-14
**Theory:** `docs/model-v11-architecture/research/r132-compilation-theory.md`
**Target file:** `src/model_v11/experiments/r132_compilation_training.py` (rewrite from scratch)
**Based on:** R130 structure (`src/model_v11/experiments/r130_cached_7b_pipeline.py`)

---

## Overview

Rewrite R132 from scratch. DO NOT patch the existing buggy file. Fresh implementation based on validated theory.

---

## File Structure

```python
# r132_compilation_training.py

# === CONSTANTS ===
SMALL_CONFIG = { n_layer=8, d_model=256, d_inner=512, d_state=16, ... }
LARGE_CONFIG = { n_layer=32, d_model=4096, d_inner=8192, ... }

# === PHASE 0: Model Setup (reuse from R130) ===
def init_model(cfg) -> model
def pack_layer_to_int2(model, k)
def unpack_layer_from_int2(model, k)

# === COVARIANCE COLLECTION ===
class CovarianceCollector:
    """Accumulates C_{vv} and C_{Gv} during training via hooks."""
    def __init__(self, d_model, d_inner, device)
    def register_hooks(self, layer_block, h_kp1_tensor)
    def remove_hooks(self)
    def accumulate(self, enabled=True)  # toggle collection
    def get_matrices(self) -> (C_vv, C_Gv)
    def reset(self)

# === PHASE 1: SSM Warmup ===
def run_phase1_ssm_warmup(model, stage_k, hk_cache, data, cfg) -> (loss_start, loss_end, collector)

# === PHASE 2: Natural Gradient ===
def run_phase2_natural_gradient(model, stage_k, collector, cfg) -> applied

# === PHASE 4: Joint Fine-Tuning ===
def run_phase4_joint_finetune(model, stage_k, hk_cache, data, cfg) -> (loss_start, loss_end)

# === STAGE LOOP ===
def run_compilation_training(cfg)

# === MAIN ===
if __name__ == '__main__': ...
```

---

## Detailed Specifications

### CovarianceCollector

```python
class CovarianceCollector:
    """
    Collects C_{vv} and C_{Gv} for natural gradient computation.

    C_{vv} = (1/N) Σ v_k^T v_k    where v_k = out_proj input (SSM_out * gate)
    C_{Gv} = (1/N) Σ G^T v_k      where G = ∂L/∂H_{k+1} (loss gradient at layer output)

    v_k captured via forward hook on out_proj.
    G captured via backward hook on h_{k+1} tensor.
    """

    # Storage:
    #   C_vv: (d_inner, d_inner) float32, on device
    #   C_Gv: (d_model, d_inner) float32, on device
    #   n_samples: int

    # Forward hook on out_proj:
    #   inp[0] is v_k, shape (batch, seq_len, d_inner)
    #   Flatten to (B*T, d_inner), accumulate C_vv += v_flat.T @ v_flat
    #   Store v_flat temporarily for pairing with G

    # Backward hook on h_{k+1}:
    #   grad is G^(b), shape (batch, seq_len, d_model)
    #   Flatten to (B*T, d_model), accumulate C_Gv += G_flat.T @ v_flat_stored

    # CRITICAL: v_k and G must be from the SAME forward/backward pass.
    # Forward hook fires first (stores v_flat), backward hook fires after (uses stored v_flat).
```

### Phase 1: SSM Warmup (50 steps)

```
INPUTS:
  - model with layer k unpacked to GPU (fp32)
  - hk_cache: bf16 tensor on CPU, shape (M_CACHE, batch, seq_len, d_model)
  - data iterator for targets

FREEZE:
  - model.layers[k].block.in_proj.weight.requires_grad_(False)
  - model.layers[k].block.out_proj.weight.requires_grad_(False)

TRAIN:
  - model.layers[k].block.A_log (nn.Parameter)
  - model.layers[k].block.x_proj (TernaryLinear — GD on fp32 shadow via STE)
  - model.layers[k].block.dt_proj (nn.Linear — direct GD, fp32)
  - model.layers[k].block.D (nn.Parameter)
  - model.lm_head (persistent across stages)
  - model.final_norm (persistent across stages)

OPTIMIZER: AdamW(trainable_params, lr=stage_lr(k), weight_decay=0.01)
  - stage_lr(k) = lr_base × σ^{-k}, σ=0.93
  - 5-step linear warmup

FORWARD (per step):
  1. h = hk_cache[step % M_CACHE].to(device, dtype=torch.bfloat16)
  2. Load targets y for this batch
  3. Forward through layer k: h_out = model.layers[k](h)
  4. Forward through frozen downstream layers k+1..N-1 WITH gradient:
     - Move each frozen layer to GPU
     - model.layers[j].eval()  # suppress inner SAC checkpoint
     - h_out = torch.utils.checkpoint.checkpoint(model.layers[j], h_out, use_reentrant=False)
  5. logits = model.lm_head(model.final_norm(h_out))
  6. loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))
  7. loss.backward()
  8. optimizer.step(), optimizer.zero_grad()
  9. Move frozen layers back to CPU AFTER backward

COVARIANCE COLLECTION (last 10 steps only):
  - Step 40: collector = CovarianceCollector(d_model, d_inner, device)
  - Step 40: collector.register_hooks(layer_block, h_kp1_tensor)
  - Steps 40-49: hooks accumulate C_{vv} and C_{Gv}
  - Step 49: collector.remove_hooks()

OUTPUT:
  - loss_start, loss_end (for gate check)
  - collector (containing C_{vv} and C_{Gv})

VERIFICATION:
  - Print: [P1] stage=K step=S/50 loss=L.LLLL
  - Print: [P1-GRAD] layer_k.A_log grad_norm=G.GGGG (verify SSM training)
  - Print: [P1-FREEZE] in_proj.requires_grad=False, out_proj.requires_grad=False (verify freeze)
```

### Phase 2: Natural Gradient (one solve)

```
INPUTS:
  - model with layer k on GPU
  - collector with C_{vv} and C_{Gv}

GATE CHECK:
  - loss_reduction = (loss_start - loss_end) / loss_start
  - If loss_reduction < 0.03: SKIP Phase 2, return applied=False

SOLVE for out_proj:
  1. C_vv, C_Gv = collector.get_matrices()
  2. C_vv += λ × I   (λ = 1e-6)
  3. δ_out = torch.linalg.solve(C_vv, C_Gv.T).T   # shape (d_model, d_inner)
  4. Scale check: if ||δ_out|| > 2 × ||out_proj.weight||: REJECT
  5. out_proj.weight.data += η_NG × δ_out.to(out_proj.weight.dtype)

SOLVE for in_proj (analogous):
  - Requires C_{uu} and C_{G_in, u}
  - C_{uu} accumulated from in_proj forward hook (inp[0] = u_k = LN(H_k))
  - G_in = gradient at in_proj output (backward hook on in_proj output tensor)
  - Same solve pattern

  NOTE: For V1 implementation, solve ONLY out_proj (simpler, fewer hooks).
  Add in_proj natural gradient in V2 if out_proj works.

η_NG:
  - Start with η_NG = 1.0 (full natural gradient step)
  - If loss worsens after Phase 2, reduce to 0.5, 0.1

VERIFICATION:
  - Print: [P2] stage=K ||C_vv||=V.VV ||C_Gv||=G.GG cond(C_vv)=C.CC
  - Print: [P2] δ_out: ||δ||=D.DD, ||W||=W.WW, ratio=R.RR
  - Print: [P2] applied=True/False (gate check or scale check rejection)
```

### Phase 4: Joint Fine-Tuning (50 steps)

```
INPUTS:
  - model with layer k on GPU (weights updated by Phase 2)
  - hk_cache
  - data iterator

UNFREEZE:
  - model.layers[k].block.in_proj.weight.requires_grad_(True)
  - model.layers[k].block.out_proj.weight.requires_grad_(True)
  - ALL params in layer k now trainable

OPTIMIZER: AdamW(all_layer_k_params + lm_head + final_norm, lr=stage_lr_p4(k))
  - stage_lr_p4(k) = lr_refine × σ^{-k}, lr_refine = 3e-5
  - LR decay within phase: lr × 0.7^(step/50)

FORWARD: Same as Phase 1 (deep gradients through frozen downstream)

VERIFICATION:
  - Print: [P4] stage=K step=S/50 loss=L.LLLL
  - Print: [P4-GRAD] out_proj.weight grad_norm=G.GGGG (verify projections training)
  - Print: [P4-GRAD] A_log grad_norm=G.GGGG (verify SSM still training)
```

### Stage Loop

```
def run_compilation_training(cfg):
    model = init_model(cfg)
    # Pack all layers to INT2, move emb+lm_head+norm to GPU

    hk_cache = build_initial_cache(model, data)  # bf16, CPU

    eval_loss_initial = eval_loss(model, data)
    print(f"[INIT] val_loss={eval_loss_initial:.4f}")

    for stage_k in range(cfg.n_layer):
        unpack_layer_from_int2(model, stage_k)
        model.layers[stage_k].to(device)

        # Phase 1: SSM warmup
        loss_s, loss_e, collector = run_phase1_ssm_warmup(
            model, stage_k, hk_cache, data, cfg
        )

        # Phase 2: Natural gradient
        applied = run_phase2_natural_gradient(
            model, stage_k, collector, cfg
        )

        # Phase 4: Joint fine-tuning
        loss4_s, loss4_e = run_phase4_joint_finetune(
            model, stage_k, hk_cache, data, cfg
        )

        # Post: rebuild cache, pack to INT2
        hk_cache[stage_k + 1] = compute_next_cache(model, hk_cache[stage_k], stage_k)
        pack_layer_to_int2(model, stage_k)
        model.layers[stage_k].to('cpu')

        # Checkpoint every 4 stages
        if stage_k % 4 == 0 or stage_k == cfg.n_layer - 1:
            save_checkpoint(model, stage_k)

        eval_val = eval_loss(model, data)
        print(f"[STAGE {stage_k}] P1: {loss_s:.4f}→{loss_e:.4f} | "
              f"P2: applied={applied} | P4: {loss4_s:.4f}→{loss4_e:.4f} | "
              f"val={eval_val:.4f}")

    print(f"[DONE] final_val={eval_loss(model, data):.4f}")
```

---

## What to Reuse from R130

| Component | Reuse? | Notes |
|---|---|---|
| INT2 pack/unpack | YES | Copy exactly |
| Stage loop structure | YES | Adapt for 3-phase instead of single |
| bf16 cache | YES | Replace PCA-INT8 with direct bf16 |
| Model initialization | YES | Standard random init (NO zeroing!) |
| Data loading (WikiText-103) | YES | Copy exactly |
| eval_loss function | YES | Copy exactly |
| Checkpoint saving | YES | Copy exactly |
| PCA-INT8 codec | NO | Removed (bf16 direct cache) |
| Global cosine LR | NO | Replaced with per-stage correction |

---

## Config

```python
SMALL_CONFIG = {
    'n_layer': 8, 'd_model': 256, 'd_inner': 512,
    'd_state': 16, 'n_heads': 4, 'vocab_size': 256,
    'seq_len': 256, 'batch_size': 4,
    'phase1_steps': 50, 'phase4_steps': 50,
    'lr_bootstrap': 5e-4, 'lr_refine': 3e-5,
    'sigma': 0.93, 'lambda_reg': 1e-6,
    'eta_ng': 1.0, 'gate_threshold': 0.03,
    'M_CACHE': 150,
}
```

---

## Verification Checklist

Before declaring ANY result:

- [ ] `[P1]` logs appear with decreasing loss
- [ ] `[P1-FREEZE]` confirms in_proj/out_proj frozen
- [ ] `[P1-GRAD]` shows nonzero grad_norm on A_log
- [ ] `[P2]` shows C_vv and C_Gv norms (not zero, not NaN)
- [ ] `[P2]` δ_out ratio < 2.0 (scale check passes)
- [ ] `[P4]` loss continues decreasing from Phase 2 starting point
- [ ] `[P4-GRAD]` shows nonzero grad_norm on out_proj AND A_log
- [ ] Val loss after each stage is monotonically improving (or at worst flat)
- [ ] No NaN anywhere
- [ ] VRAM peak reported and < 10GB

---

## Anti-Patterns (DO NOT IMPLEMENT)

1. **OLS with hooks on current weights** → tautological, proven
2. **EM iteration** → circular, proven
3. **Explicit ternarization after closed-form** → scale collapse, proven
4. **Zeroing layer weights** → destroys initialization
5. **Implementing phases independently** → they interact, design as integrated system
6. **Plain SGD for bootstrap** → too slow, use AdamW
7. **Collecting covariance on ALL steps** → high variance early; last 10 steps only
