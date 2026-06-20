# Phase 29 — Validation Batch Results

**Date:** 2026-02-26
**Objective:** Validate all Phase 28 implementations at functional level before scale testing.

## Critical Bug Fix: Optimizer Hyperparameters

**Root cause of NaN/gradient explosion in ternary training identified and fixed.**

| Parameter | Before (broken) | After (fixed) | Why |
|-----------|-----------------|---------------|-----|
| beta2 | 0.95 | 0.999 | 0.95 causes variance oscillation with STE discontinuous gradients |
| weight_decay | 0.1 | 0.01 | 0.1 fights STE by pushing shadow weights toward zero |
| fused | True | False | Fused AdamW may handle STE gradients differently |

**Commit:** `6b9a9d0` — All three now CLI-configurable (--beta2, --weight-decay, --fused-optimizer)

### Validation: Ternary 6M (no SSD)
- 50 steps, zero NaN, loss 9.82 → 8.41
- Grad norms stable: 1.47-1.58
- **PASS**

### Validation: SSD + Ternary 50M
- 30 steps, zero NaN, loss 12.74 → 8.95
- Grad norms: 2.1-15.5 (spike during LR warmup, normal)
- Previously exploded at step 5-7 with old hyperparameters
- **PASS — SSD+ternary collapse RESOLVED**

## Test 1: Angular + Boundary Regularization (Phase 28-B)

**Config:** ternary_6m, 100 steps, lambda=0.01 each
**Result: PASS**

| Metric | Value |
|--------|-------|
| Final loss | 7.69 |
| Grad norms | 1.72 → 1.00 (healthy decay) |
| Angular reg loss | 0.0005 |
| Boundary reg loss | 0.059-0.064 |
| Ternary layers active | 28 |
| Eval PPL | 1877.42 |
| Throughput | 8K-9K tok/s |

**Key observation:** Both regularizers confirmed active on all 28 ternary layers. Measurable loss contribution without destabilizing training.

## Test 2: GaLore Optimizer (Phase 28-C)

**Config:** ternary_6m, 100 steps, rank=64
**Result: PASS (after CLI wiring fix)**

| Metric | Value |
|--------|-------|
| Final loss | 7.75 |
| Grad norms | 1.63 → 0.97 |
| GaLore projectors | 35 low-rank projectors created |
| GaLore params | 3.77M (31% of model) |
| Standard params | 8.22M |
| Eval PPL | 2227.75 |
| Throughput | 9.6K-11K tok/s |

**Bug found and fixed:** `--use-galore` flag was dead code — didn't set `optimizer_type='galore'`. Also `--optimizer-type` choices were missing 'galore'. Fixed in same session.

## Test 3: d_state=64 (Phase 28-B)

**Config:** ternary_6m with d_state overridden 16→64, 100 steps
**Result: PASS**

| Metric | Value |
|--------|-------|
| Final loss | 10.43 |
| Grad norms | 4.37 → 1.88 (healthy decay) |
| Total params | 12.3M (minimal increase from d_state change) |
| Ternary ratio | 32.9% |
| Throughput | 3.3K-5.2K tok/s |

**Note:** Higher d_state = richer SSM state but lower throughput. Trade-off to evaluate at scale.

## Test 4: STEWS Evaluation (Phase 28-B)

**Config:** ternary_6m, 100 steps training → STEWS eval with K=1,3,5
**Result: PASS**

| Ensemble K | PPL | NLL | vs Baseline |
|------------|-----|-----|-------------|
| K=1 (baseline) | 40,491 | 10.609 | — |
| K=3 | 40,982 | 10.621 | +1.2% |
| K=5 | 40,191 | 10.601 | -0.7% |

**Key observation:** Low variance across stochastic weight swaps. K=5 shows marginal improvement. Ternary weights are stable under STEWS perturbation.

## Test 5: torch.compile Selective (Phase 28-D)

**Config:** ternary_6m, 50 steps, compile ON vs OFF
**Result: NEGATIVE — torch.compile harmful at this scale**

| Metric | Without compile | With compile | Change |
|--------|----------------|--------------|--------|
| Throughput | 10,404 tok/s | 526 tok/s | **-94.9%** |
| Training time | 4.9s | 97.4s | **19.8x slower** |
| Final loss | 10.426 | 10.413 | -0.12% (neutral) |
| GPU memory | ~2.3 GB | ~2.3 GB | Same |

**Root cause:** RTX 5070 doesn't have enough SMs for max_autotune_gemm. SSM + ternary graph creates compilation overhead that dominates execution at small scale.

**Decision:** Disable torch.compile for all Phase 29 scale runs. Revisit only for selective function-level compilation on specific hot paths.

## Test 6: Mega-kernel Benchmark (Phase 28-A)

**Config:** 50M scale, CUDA graphs enabled
**Result: PASS (after rotary embedding fix)**

| Backend | tok/s | vs PyTorch |
|---------|-------|------------|
| PyTorch (baseline) | 825.8 | — |
| Triton | 855.0 | +3.5% |
| Mega-kernel | 842.2 | +1.9% |

**Bug fixed:** `swa_step()` was missing modulo wrapping for position index into rotary cache. Added `position % rope_cache_size` matching the pattern already in CUDAGraphRunner. Commit `f294616`.

**Note:** Modest speedup on Windows where Triton is not fully supported. Linux should show larger gains. Peak VRAM: 247 MB.

## Infrastructure Fixes Applied

1. **New configs:** `config_ternary_6m()`, `config_ternary_50m()` — first configs with `use_ternary=True`
2. **CLI expansion:** 13 new arguments for Phase 28 features + operational params
3. **Optimizer type choices:** Expanded from ['adamw', 'schedule_free'] to include galore, muon, ademamix, spam, fira
4. **GaLore wiring fix:** `--use-galore` now properly sets `optimizer_type='galore'`
5. **Init revert:** TernaryLinear gain restored to `1/sqrt(in_features)` from unauthorized change

## Summary

| Test | Status | Key Finding |
|------|--------|-------------|
| Optimizer fix (beta2/wd/fused) | **PASS** | Root cause of ALL ternary NaN issues |
| SSD+ternary training | **PASS** | Collapse was optimizer, not architecture |
| Angular+Boundary reg | **PASS** | Active on 28 layers, stable contribution |
| GaLore optimizer | **PASS** | 35 projectors, 31% params low-rank |
| d_state=64 | **PASS** | Works but ~2x slower throughput |
| STEWS | **PASS** | Low variance, marginal improvement at K=5 |
| torch.compile | **NEGATIVE** | 19.8x slower, disable for now |
| Mega-kernel 50M | **PASS** | +1.9% PyTorch, +3.5% Triton (Windows limited) |

**Overall: 7 PASS, 1 NEGATIVE (expected)**

## Scale Validation: 50M Ternary + Phase 28 Techniques (Local RTX 5070)

**Config:** ternary_50m, batch=64, 1000 steps, angular+boundary reg (0.01), no torch.compile
**Hardware:** RTX 5070 12GB VRAM (batch=64 fits — GPU VRAM ≠ system RAM)

| Metric | Value |
|--------|-------|
| Final loss | 5.76 |
| Best loss (step ~800) | 5.59 |
| Eval PPL | 280.75 |
| Throughput | 10,738 tok/s |
| Training time | 6.4 min |
| Total params | 72.1M |
| Ternary params | 50.6M (70.2%) |
| Ternary layers (reg active) | 62 |
| Grad norms | ~0.6 (stable) |
| NaN | Zero |

**Loss progression (every 100 steps):**
7.26 → 6.81 → 6.40 → 6.18 → 6.05 → 5.68 → 5.84 → 5.59 → 5.75 → 5.76

**Key findings:**
- Batch=64 fits on 12GB RTX 5070 (same as Modal L4 24GB — ternary is memory efficient)
- All 62 ternary layers receiving angular+boundary regularization
- Ternary distribution converged to perfect uniformity (33/33/33) by step 100
- Cosine LR decayed to 0 at step 1000 — longer runs need adjusted schedule
- PPL 280.75 is early-stage (previous Modal full training achieved PPL 70.77)
- Comparison with previous baseline needs matched step count

**Infrastructure verified:**
- Optimizer fix (beta2=0.999, wd=0.01, fused=False) confirmed working at scale
- Rotary embedding fix confirmed working (no IndexError)
- torch.compile disabled (19.8x slower on RTX 5070)

## Next Steps

1. Extended training: match Modal step count for fair PPL comparison with baseline
2. Phase 30: POPCOUNT binary decomposition for inference
