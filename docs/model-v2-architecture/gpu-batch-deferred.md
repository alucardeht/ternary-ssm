# GPU Batch — Overnight Execution Plan

**Status:** Ready to execute. User runs before sleeping.
**Total estimated time:** ~5h
**Updated:** 2026-02-27 — Removed Q14 (NEUTRAL) and Q16 (NEGLIGIBLE) based on synthetic testing

## Execution Order

### Step 1: FP16 Teacher Training (~2h)
**Purpose:** Train full-precision 50M baseline for distillation
**Unlocks:** Step 2 (Q15 QAD)

```bash
python -m src.model_v2.train --config scale_50m --max-steps 5000 --device cuda --data-dir data/wikitext103 --save-dir checkpoints/fp16_teacher_50m
```

**Verify:** Check that `checkpoints/fp16_teacher_50m/` contains model checkpoint and loss < 4.0

### Step 2: Q15 QAD Distillation with Real Teacher (~1h)
**Purpose:** Knowledge distillation from FP16 teacher → ternary student
**Depends on:** Step 1 complete
**Expected:** 3-5% PPL improvement over baseline ternary

```bash
python -m src.model_v2.experiments.bench_distillation_unified --config scale_50m --teacher-path checkpoints/fp16_teacher_50m --device cuda --max-steps 2000 --mode q15_only
```

**Note:** Script may need adaptation for real teacher path. Check args.

### Step 3: T10 Final 50M Validation (~2h)
**Purpose:** Combine all winning techniques at 50M scale
**Techniques to include:**
- GaLore (best loss in Phase 36: 7.34)
- STEWS (7.51, close to baseline)
- TernaryLMHead (7.51, same as STEWS)
- Q15 QAD distillation (if Step 2 shows improvement)

```bash
python -m src.model_v2.train --config scale_50m --max-steps 5000 --device cuda --data-dir data/wikitext103 --save-dir checkpoints/t10_final_50m --use-galore --use-stews --use-ternary-lm-head
```

**Note:** Exact flags depend on what's implemented in train.py. Verify available args first.

## Results from Synthetic Testing (6M, 15 steps)

These results informed which techniques to include/exclude:

| Technique | Result | Status |
|-----------|--------|--------|
| Q15 QAD (Logit KL) | -50% loss | **STRONG** — include at scale |
| Q14 Bootstrap | No iterative improvement | **NEUTRAL** — excluded |
| Q16 Attention MSE | Adds ~0 over Q15 | **NEGLIGIBLE** — excluded |
| M3 Codebook | 1.14× at best | **MARGINAL** — excluded |
| M4 mmap | 99.97% memory reduction | **VALIDATED** — deployment tool, not training |
| M5 XOR-Delta | 1.53× for variants | **VIABLE** — post-training tool, not overnight |

## After Overnight Batch

1. Check all logs for NaN/divergence
2. Compare T10 loss vs Phase 36 baseline (7.41)
3. Record GPU throughput (tok/s) and VRAM peak
4. Update vision-and-status.md with final results
5. Decision: scale to 100M or iterate on architecture
