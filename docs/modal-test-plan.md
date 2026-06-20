# Modal.com Test Plan — Ternary SSM Language Model

## Overview

Tests organized by priority. All tests use our existing codebase — Modal wraps our training scripts with GPU access and persistent storage.

**Modal Setup:**
- Install: `pip install modal && modal token new`
- Volumes: `ternary-checkpoints` (results), `training-data` (WikiText-103 tokenized)
- Upload data: `modal volume put training-data ./data/wikitext/ /data/wikitext/`

## Test 1: FP16 Baseline (CRITICAL — No comparison without this)

**Purpose:** Train the EXACT same architecture with FP16 weights instead of ternary. Gives us clean comparison numbers.

**What changes:** `config.use_ternary = False` (everything else identical)

**Metrics to compare:**
| Metric | Ternary (measured) | FP16 (to measure) |
|--------|-------------------|-------------------|
| Val PPL (1 epoch) | 78 | ? |
| Train VRAM | 2,580 MB | ? |
| Inference VRAM | 313 MB | ? |
| Train throughput | 10,000 tok/s | ? |
| Inference tok/s | 47 tok/s | ? |
| Weight storage | 5.9 bits/param | 16 bits/param |

**Config:**
- Model: config_wikitext_50m() with use_ternary=False, use_routing=True, use_ssd=True
- Data: WikiText-103, 1 epoch (50K steps)
- Optimizer: AdamW, lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95)
- Regularization: dropout=0.1, shuffle=True
- BF16 autocast + torch.compile
- Eval every 5000 steps on val split, final eval on test split

**GPU:** A100-80GB (cheaper, sufficient for 50M)
**Estimated time:** ~30 min on A100 (vs 1.8h on RTX 5070)
**Estimated cost:** ~$1.25 (0.5h x $2.50/h)

---

## Test 2: Multi-Epoch Ternary (5 epochs)

**Purpose:** Train ternary model to convergence on WikiText-103. Currently PPL 78 after 1 epoch — need final converged PPL.

**Config:**
- Model: config_wikitext_50m() with use_ternary=True, use_routing=True, use_ssd=True
- Data: WikiText-103, 5 epochs (250K steps)
- Same optimizer/regularization as Test 1
- Checkpoint every 10K steps, eval every 5K steps
- Resume from existing 50K checkpoint OR train fresh

**GPU:** A100-80GB
**Estimated time:** ~2.5h on A100
**Estimated cost:** ~$6.25

---

## Test 3: Multi-Epoch FP16 Baseline (5 epochs)

**Purpose:** Compare converged FP16 vs converged ternary. The definitive comparison.

**Config:** Same as Test 2 but use_ternary=False

**GPU:** A100-80GB
**Estimated time:** ~2.5h on A100
**Estimated cost:** ~$6.25

---

## Test 4: Scaling — 100M Ternary

**Purpose:** Validate that ternary SSM scales to 100M. Check training stability, PPL, VRAM.

**Config:**
- Need to define config_wikitext_100m() (d_model=768, n_layer=24, etc.)
- WikiText-103, 1 epoch
- Same optimizer settings (may need LR adjustment)

**GPU:** A100-80GB or H100
**Estimated time:** ~2-4h
**Estimated cost:** ~$5-16

---

## Test 5: Scaling — 300M Ternary

**Purpose:** Show the architecture works at 300M. This is where efficiency gains really matter.

**Config:**
- Need config_wikitext_300m() (d_model=1024, n_layer=32, etc.)
- WikiText-103, 1 epoch
- May need gradient accumulation for memory

**GPU:** H100 (80GB needed for 300M)
**Estimated time:** ~6-12h
**Estimated cost:** ~$24-47

---

## Test 6: Scaling — 100M and 300M FP16 Baselines

**Purpose:** FP16 baselines at each scale for direct comparison.

**Cost:** Similar to Tests 4-5

---

## Execution Order

```
Priority 1 (immediate):
  Test 1: FP16 baseline 50M (1 epoch)     ~$1.25, 30 min
  Test 2: Multi-epoch ternary 50M          ~$6.25, 2.5h

Priority 2 (after Priority 1 results):
  Test 3: Multi-epoch FP16 50M             ~$6.25, 2.5h

Priority 3 (scaling validation):
  Test 4: 100M ternary (1 epoch)           ~$5-16
  Test 5: 300M ternary (1 epoch)           ~$24-47
  Test 6: 100M + 300M FP16 baselines       ~$29-63

Total estimated cost: ~$75-140
```

## Cost Summary

| Test | GPU | Time | Cost |
|------|-----|------|------|
| 1. FP16 baseline 50M | A100 | 0.5h | $1.25 |
| 2. Multi-epoch ternary 50M | A100 | 2.5h | $6.25 |
| 3. Multi-epoch FP16 50M | A100 | 2.5h | $6.25 |
| 4. 100M ternary | A100/H100 | 2-4h | $5-16 |
| 5. 300M ternary | H100 | 6-12h | $24-47 |
| 6. 100M+300M FP16 | H100 | 8-16h | $29-63 |
| **Total** | | | **$75-140** |

## Checkpoint Naming Convention

```
{model_size}_{type}_{epochs}e_{steps}s.pt
Examples:
  50m_ternary_1e_50000s.pt
  50m_fp16_1e_50000s.pt
  50m_ternary_5e_250000s.pt
  100m_ternary_1e_30000s.pt
  300m_ternary_1e_15000s.pt
```

## Metrics to Collect (Every Test)

1. Val PPL (every 5K steps) + Test PPL (final)
2. Training loss curve
3. Peak VRAM usage
4. Training throughput (tok/s)
5. Wall-clock time per epoch
6. Ternary weight distribution (% of -1, 0, +1) — ternary tests only
7. Gradient norms (for stability monitoring)

## What We Already Have (Local Results)

| Test | Result | Source |
|------|--------|--------|
| 50M ternary, 1 epoch | Val PPL 78, Test PPL 81 | Finding 37 |
| 50M ternary, inference | 47 tok/s GPU, 14 CPU | Finding 38 |
| 50M training VRAM | 2,580 MB | Finding 37 |
| 50M inference VRAM | 313 MB | Finding 38 |
| BF16+compile speedup | 2.23x throughput | Finding F |
