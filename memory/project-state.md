---
name: Current project state
description: ENTRY POINT — read this FIRST after /compact to know where we are and what to do next
type: project
---

# Project State

## Operational Note (2026-04-14)

This file is still the stable technical baseline, but it is no longer the newest
operational truth.

Before acting on "where we stopped" or "what to do next", read:

1. `.workspace/codex/STATE.md`
2. `.workspace/codex/QUEUE.md`
3. `memory/project_overnight-autonomous-plan.md`
4. `.workspace/research/ROUND-BACKLOG-consolidated.md`

Use this file for the baseline architecture / phase snapshot. Use the continuity
files above for the live handoff from the overnight loop and the current
implementation frontier.

### Key files:
- MACRO plan: src/model_v12/docs/MACRO-PLAN.md
- Architecture: src/model_v12/docs/architecture.md
- Math: src/model_v12/docs/math-innovations.md
- Build: gcc command in MACRO-PLAN.md Phase 2.3
- Dataset: data/chat_combined/train.bin (896M tokens)

### Build command:
```bash
gcc -std=c99 -O2 -Wall -mavx2 -mfma -DUSE_MMAP=0 \
  -I src/model_v12/src -I src/model_v12/src/core -I src/model_v12/src/io \
  src/model_v12/src/core/ternary_matmul.c \
  src/model_v12/src/core/ssm_scan.c \
  src/model_v12/src/core/loss.c \
  src/model_v12/src/core/optimizer.c \
  src/model_v12/src/core/model.c \
  src/model_v12/src/io/dataset.c \
  src/model_v12/src/train.c \
  -o src/model_v12/train -lm
```

### IMPORTANT:
- types.h has DEV config (small dims). Restore before prod training.
- Optimizer uses plain SGD (MVUE (w-s) removed — was too slow)
- D4 filter disabled (was blocking all updates steps 0-4)
- gp_kl_stub.c DELETED (loss.c has real implementation)

---

**Last updated:** 2026-04-15

## What This Project Is

A ternary SSM language model (Mamba-2 SSD + SWA hybrid) trained with Forward Programming (layer-sequential) + mini-BCD, designed to train 7B on consumer GPU (12 GB VRAM, RTX 5070).

**Core innovation:** FP trains one layer at a time (893 MB VRAM for 3.38B model) instead of joint GD (~56 GB). INT2 storage compresses frozen layers 16x.

## The 3 Barriers (2026-03-14)

1. **VRAM: SOLVED** — INT2 frozen + FP layer-sequential = 893 MB/stage (R130)
2. **TEMPO: NOT SOLVED** — 60B tokens minimum = 495 days on RTX 5070 (R150). Optimized FP = ~20-25 min for 1 pass, but real training needs many passes.
3. **QUALIDADE: NOT SOLVED** — FP gap is structural (65-70%). Each layer trains ISOLATED. mini-BCD closes 25%, SAC 35%. FP+1000 GD steps = 89% closure (R135).

## Current Phase

**V12 C IMPLEMENTATION IN PROGRESS (2026-04-15 refresh)**
**V13 C IMPLEMENTATION IN PROGRESS (2026-04-15 refresh)**

v11 FROZEN (tag v11-final). v12 is frozen.
- **Active Development Track:** Native **Ternary VSA Fractional Engine** (`src/model_v13/`).
- **Frozen Baseline:** Native Ternary SSM (`src/model_v12/`) is now frozen for benchmark comparisons.
- **Active Research Round:** **R506 Implementation Phase** (Dynamic Topology and VSA Binding).

### Current implementation frontier (R368 M1, 2026-04-15)
- `S4-safe-gpu-only` landed: dead GPU-only Vulkan plumbing was removed from the active path and validated by a clean Vulkan rebuild plus a 10-step smoke run.
- `train.c` now exposes `--seed`, `--log-every`, and `--loss-log-hex`, which makes per-step deterministic logging and reproducibility checks possible without changing training math.
- Strong post-`S4` artifacts live in `.workspace/research/references/r368-m1/`: two 100-step runs with `seed=42` produced 100/100 identical `[S5]` lines, and the `[M1 RECOMPUTE]` probe appeared 10,800 times per run.
- The previously observed exit code 1 came from the strong-validation harness or wrapper, not from the binary itself; direct `train.exe` runs returned 0 and ended at `[TRAIN] Complete`.
- `N1` is now landed: the default `USE_VULKAN=1` backward path uses the recompute path without needing `V12_M1_SLICE_A/B`.
- A stronger acceptance check was added for `N1`: the default path matched the prior validated reference on 100/100 `[S5]` lines, and rebuilding with old `V12_M1_SLICE_A/B` flags produced no behavioral diff.
- `N2` is now landed: the remaining GPU forward-only gate cache path was removed (`fwd_a_eff_gate` / `a_eff_gate_out`), and two new 100-step `seed=42` runs still matched the post-`N1` reference on 100/100 `[S5]` lines.
- Formal closure of the original `S5` criterion remains historically blocked because the repo does not contain a named pre-`S1` reference log for `R368 M1`.

### Immediate next action
- Choose the next frontier explicitly: either continue with the host-side follow-on of `R368` (`N3`) or pivot to the larger `R386` owner-inversion refactor.
- Keep the historical pre-`S1` baseline issue separate from `N1/N2`; do not claim zero-delta equivalence versus pre-`S1`, only versus the best validated reference currently available in-repo.

### What EXISTS and WORKS in v12:
- types.h: all structs (571 lines)
- ternary_matmul.c: CSR add/sub + AVX2 gather (3/3 tests pass)
- ssm_scan.c: forward + backward (5/6 tests pass)
- loss.c: CE + GP-KL Welford (3/3 tests pass)
- optimizer.c: plain SGD + crystallization (4/4 tests pass)
- dataset.c: load + xoshiro RNG (6/6 tests pass)
- model.c: alloc/free + CSR rebuild
- train.c: forward + backward + train_step + main

### CONFIRMED WORKING:
- Compiles: gcc -std=c99 -O2 -mavx2 -mfma -DUSE_MMAP=0 (7 .c files)
- Trains: loss 5.556 → 5.541 over 2000 steps (tiny dev config)
- Dev config: 2 layers, d=64, d_state=16, vocab=256, seq=64
- AVX2 ternary_matvec active

### NEXT STEPS (Phase 2 from MACRO-PLAN.md):
1. Build transposed CSR for fast backward (lm_head backward is 10s at full dims)
2. Restore full dimensions (6L, d=1024, vocab=32064)
3. Create Makefile
4. Optimize: SIMD for all matmuls, batch matmul

### All active features in v2 training:
MVUE piecewise linear, natural gradient Fisher, GP-KL online Welford,
Fisher coupling, boundary filter hooks, MTP-SSM A^k, WHT 1×/layer,
MUX, eigenspectrum 4-cluster, capacity matching, deadzone bias,
crystallization skip, direct-B, EOS head, contraction window,
co-spectral subspace, ternary_aware_checkpointing, OSP full-dataset calibration

### Previously implemented and active in train_production.py:
- BPE 32K tokenizer (vocab=32007)
- ABA Welch t-test (replaced point comparison)
- V-PRIME fixed eval batch (deterministic measurement)
- LR schedule (warmup + cosine decay)
- Warm start last stage (20 pre-steps, frozen head)
- lm_head reinit before last stage
- Proportional steps (2.5× budget for last stage)
- Streaming CE loss (6× VRAM reduction)
- BCD cached targets (50×+ VRAM reduction)
- CE sum gradient (not mean — BitNet confirmed, -0.72 nats)
- Predictive coding rank=24 (down from 64)
- PSA (Predictive State Alignment — constant SNR all layers)
- TIL (Temporal Information Loss — direct gradient to A_log)
- B* init (RNS→eigenspace alignment layer 0)
- RNS embedding + TBD init
- Configurable FP order (forward/reverse/bisection)
- Intermediate checkpoint between FP and BCD

### ALL VALIDATED FINDINGS IMPLEMENTED:
- TCSD: ACTIVE in Joint GD
- TMA: ACTIVE (64-slot ternary memory addressing)
- OSP: ACTIVE (config propagates correctly)
- Eigenspectrum calibration: ACTIVE (tau=[8,121], r=[0.88,0.99])
- PSA, TIL, CE sum, B* init, rank=24: all ACTIVE
- σ_min(I-A_q) attractor regularizer: ACTIVE
- Trajectory Diversity Loss (L_traj): ACTIVE
- Prompt-State Anchoring: ACTIVE (src/model_v11/components/prompt_state_anchor.py)
- EOS Entropy Head: ACTIVE (in model.py)
- Adaptive Threshold: ACTIVE (threshold_type=adaptive in TernaryLinear)
- L_gram Row-Orthogonality: ACTIVE

### Deferred Items (need SSM scan changes):
- SITW (State-Impact Token Weighting): requires exposing internal SSM states
- ISSC (Inter-Sequence State Chaining): requires SSM scan modification

### Extraordinary Findings (not yet implemented):
- SSM Ciclotômico (Kronecker's theorem, cyclotomic polynomials for A)
- STDP + Dopamine-gating (training without backprop)
- M₃-Natural Gradient (Fisher-Rao on belief simplex)

### Skipped (justified):
- STS: requires use_ternary=True (incompatible)
- TCDM: relevant for 7B, not urgent at 108M
- DASS/SID/Reverse FP: Joint GD replaced FP
- T-SAR/Base-3: inference/deploy

### Dataset:
- 322M BPE tokens (390K examples, 10 sources)
- `data/chat_full/train.bin` (627 MB)

### OVERNIGHT JOINT GD TRAINING COMPLETE (2026-03-24)
- **Mode: JOINT GD** (not FP — FP stagnated at 6.3 nats)
- Model: 108M params, 6 layers, d=1024, d_state=128
- Dataset: `data/chat_full/train.bin` — 322M BPE tokens, 390K examples (10 sources)
- Output: `data/checkpoints/overnight_joint/`
- Batch: 16, VRAM: ~1.76 GB, ~195ms/step
- Steps: 55,000 (~3h estimated)
- Checkpoints every 2000 steps
- Eval every 500 steps (V-PRIME fixed batch)
- Checks programmed: 5/15/30/60/120 min
- **Note:** This training used baseline configuration. R309-R313 implementations were added AFTER this run. A new training run is needed to test all improvements together.

### Previous results (same session):
- FP v1 (100 steps/stage): 6.34 nats in 6 min — garbage output
- FP v2 (500 steps/stage): 6.26 nats in 6 min — ABA stopped early, same result
- Joint GD smoke (20 steps): 6.64 nats in 4s — already near FP's 445-step result
- **Conclusion: FP gap is structural at 6 layers. Joint GD is the correct approach.**

After training: run test battery from `.workspace/plans/model-test-battery.md`

## What Works (Confirmed)

- FP trains 3.38B in 29 min unoptimized, **~13 min projected with cache+ABA** (R130/R262/R270)
- Cache H_k incremental: 33x FP speedup (R262/R265)
- ABA early stopping: 47% fewer steps (R270)
- mini-BCD: 25% gap closure in 57s (R271b)
- SAC: 35.1% gap closure at 100M (R128)
- SBP: = joint GD quality at 100M (R139)
- FP+1000 GD steps: 89% gap closure (R135)
- R132 Compilation Training: 6M 5.73→3.19 nats (R132)
- INT2 storage: +0.025 nats, negligible (R98)
- D4 gradient filter: always-on in v11 reference (R272)
- W_shift rank-8: 3.5% additional closure (R278)
- 6M standard GD: loss 1.93, 166 tok/s inference (R296)

## What Doesn't Work

- Ternary training (STE): +1.06 nats overhead (R97)
- PCA cache for training: +3.04 nats catastrophic (R122)
- Ternary post-training (PTQ): PPL 8814 even with GPTQ (R229/R241)
- Transformer→SSM conversion: converges to RANDOM (R230)
- K-FAC for SSM: 50-100x worse than Adam (R133b)
- TB-BCD: 23x slower (R276)
- Combined mods (SLS+Warm+BCD): interfere -33.4% (R115)
- Real 7B end-to-end: 495 DAYS (R150) — NEEDS paradigm shift

## Critical Open Questions

1. Does Stage 2 (mini-BCD/SAC) work at 7B scale? (never tested)
2. Should BCD target top or bottom layers? (R291 designed, not run)
3. Can compilation training (R132) scale? (6M worked, 7B unknown)
4. How to get from 495 days → hours for real 7B training?
5. Can we use the optimized pipeline (cache+ABA+mini-BCD) at 7B?

## MANDATORY WORKFLOW — After ANY Implementation

```
1. IMPLEMENT → test → debug → confirm it works
2. DOCUMENT → baselines.md + experiment-index.md + implementation-log.md
3. UPDATE PRODUCTION FILES → integrate gain into train_production.py / run_production.py
4. SMOKE TEST → run production files (10 steps) to verify they work
5. COMMIT → code + docs + production files together
6. THEN continue to next task
```

**NEVER skip steps 2-5.** NEVER continue to next research/implementation without updating production files first. See `memory/feedback_production-ready-sempre.md` for full rationale.

**Production files:**
- `src/model_v11/train_production.py` — ALWAYS up-to-date with ALL validated gains
- `src/model_v11/run_production.py` — ALWAYS ready to load and use a trained model
- `src/model_v11/results/` — ALL training run results saved here

## Key Files

- Pipeline: `memory/pipeline-reference.md`
- Baselines: `memory/baselines.md`
- Architecture: `memory/architecture-ref.md`
- Experiments: `memory/experiment-index.md`
- Doc system: `memory/doc-organization.md`
- v11 status: `docs/model-v11-architecture/STATUS.md`
- v11 impl log: `docs/model-v11-architecture/implementation/implementation-log.md`
- v11 code: `src/model_v11/` (ACTIVE)
- v10 code: `src/model_v10/` (FROZEN, read-only)

## Quick Decision Tree

- "What's the pipeline?" → `memory/pipeline-reference.md`
- "What numbers do we have?" → `memory/baselines.md`
- "Did we test X?" → `memory/experiment-index.md`
- "What's the model?" → `memory/architecture-ref.md`
- "What's the status?" → this file + `docs/model-v11-architecture/STATUS.md`
