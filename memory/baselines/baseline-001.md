# Measured Baselines — All Numbers

**Last updated:** 2026-05-21

---

## Dataset V2 (2026-03-25)
- **224K examples, 80M BPE tokens, 152 MB**
- Mix: math=45%, PT=34%, chat=17%, knowledge=4%
- Sources: GSM8K, Orca-Math, MathInstruct, FineWeb-Edu, Alpaca-PT, OPUS EN-PT + existing chat
- Tokenizer: BPE 32K (vocab=32064)
- Curation: removed duplicates, filtered low-quality, balanced domains

---

## Quality (Loss in nats)

### 100M scale (d=768, n=24, WikiText-103)

| Method | Val Loss | Gap Closure | Source |
|--------|----------|-------------|--------|
| Joint GD (baseline) | 2.0858 | 100% | R139 |
| SBP | 2.0860 | 35.2% | R139 |
| FP + SAC | 2.0848 | 35.1% | R139 |
| FP only | 2.3139 | 0% | R139 |

### 6M scale (d=256, n=8)

| Method | Val Loss | Dataset | Source |
|--------|----------|---------|--------|
| Standard GD 10k steps | 1.93 | curated 2.96 MB | R296 |
| FP K=1 (75 steps/stage) | 2.1786 | synthetic | R123 |
| FP K=4 (300 steps/stage) | 2.0729 | synthetic | R123 |

### 3.38B scale (d=4096, n=32, vocab=256, byte-level)

| Method | Val Loss | Source |
|--------|----------|--------|
| FP Stage 0 only | 3.3889 | R121 |
| FP all 32 stages | 4.54 | R130 |

**Note:** 3.38B model uses vocab=256 (byte-level), not vocab=32K. Loss NOT comparable to LLaMA 2.

---

## VRAM

| Config | VRAM | Source |
|--------|------|--------|
| 3.38B init (emb+head+norm only) | 8 MB | R121 |
| 3.38B FP per stage (R121) | 1.33 GB | R121 |
| 3.38B FP per stage (R130, cached) | 893 MB | R130 |
| 100M SBP | ~292 MB | R139 |
| 100M joint GD | ~694 MB | R139 |
| 7B projected SBP | ~10 GB | R139 extrapolation |
| 7B projected joint GD | ~25+ GB | R139 extrapolation |

---

## Throughput

| Config | tok/s | Source |
|--------|-------|--------|
| 6M inference (RTX 5070) | 166 | R296 |
| 7B inference (theoretical) | 80-150 | docs projection |
| 3.38B FP total (32 stages) | 29.4 min | R130 |
| FP speedup with cache | 43x vs uncached | R130 vs R121 |
| FP speedup with lazy refresh | 2-3x additional | R150 |

---

## Compression

| Technique | Compression | Quality impact | Source |
|-----------|-------------|---------------|--------|
| INT2 weight storage | 83% (244→41 MB/layer) | +0.025 nats | R98/R100 |
| PCA-INT8 cache (inference) | 51x | +0.114 nats | R120 |
| INT8 activation cache (training) | 4x vs fp32 | -0.028 nats (BETTER) | R118 |
| FP16 activation cache | 2x vs fp32 | -0.038 nats (BETTER) | R118 |

---

## Scale Projections (7B)

| Metric | Projected | Source |
|--------|-----------|--------|
| INT2 model on disk | 4.53 GB | R100 |
| INT2 all layers on CPU | ~1.3 GB | R130 |
| VRAM inference (INT2) | ~4.5-5.5 GB | R100 extrapolation |
| Training time (FP, 60B tokens) | ~495 days | R150 analysis |
| Training time with lazy refresh | ~165-247 days | R150 with 2-3x speedup |

---

## v12 Stability And Boundary Measurements

| Run | Dataset / shape | Key result | Source |
|-----|------------------|------------|--------|
| R430 baseline | `wiki_5mb.bin`, 1000 steps, batch 1, seed 42780, teaching off | Eval improved through step 500 (`5.6195`) then diverged by step 600 (`32.6945`) and step 1000 (`14588.0723`) | R430 overnight validation |
| R435 boundary receipt, pre-fix | `wiki_5mb.bin`, 502 steps, batch 1, seed 42780, teaching off | Fixed-probe receipt localized the jump to immediate materialization: `L_a=5.67003918`, `L_b=5.68601799`, `L_c=45.3098907`, `L_d=44.7653694` | R435 implementation closeout |
| R435 boundary receipt, post-fix | Same as above | Periodic boundary no longer jumps after CSR rebuild: `L_a=5.67004013`, `L_b=5.68601704`, `L_c=5.69009066`, `L_d=5.66146755` | R435 implementation closeout |
| R435 production smoke, post-fix | `wiki_5mb.bin`, 600 steps, batch 1, seed 42780, teaching off | Eval stayed stable past the boundary: step 500 `5.6195`, step 600 `4.9604`; activation logged `[R435_BOUNDARY_FIX] sampled_resample=0 csr_rebuild=1` | R435 implementation closeout |

R435 is a stability fix and does not change the official 7B time projection,
VRAM/RAM baseline, inference speed, or any measured training-speed claim.

---

## v11 Experiment Results (R122-R296)

| Exp | What | Key Result | Status |
|-----|------|------------|--------|
| R122 | Cached FP | fp32 +0.018 ✅, PCA-INT8 +3.04 ❌ | CONFIRMED/REJECTED |
| R123/b | K-Layer chunk | K=4 -0.106 nats (shrinks w/ scale) | CONFIRMED |
| R128 | SAC | 35.1% gap closure, 3 sweeps | CONFIRMED |
| R130 | Cached 7B | 29 min, 893 MB VRAM | CONFIRMED |
| R132 | Compilation training | 6M: 5.73→3.19 nats | CONFIRMED |
| R133b | K-FAC vs Adam | K-FAC 50-100× WORSE | REJECTED |
| R134 | FP-GD scale gap | 6M: 7% / 25M: 6.6% / 100M: 17.3% | CONFIRMED |
| R135 | FP+joint GD | 89% gap closure (1000 steps) | CONFIRMED |
| R136 | LSGD G=4 | 13% (predicted 77%) FAILED | REJECTED |
| R139 | SBP | = joint GD (+0.0002 nats) | CONFIRMED |
| R257 | Batch fix | loss 3.18 vs 3.43 | CONFIRMED |
| R262/R265 | Cache H_k incremental | **33× FP speedup** | CONFIRMED |
| R270 | ABA early stopping | **47% fewer steps**, +0.039 nats | CONFIRMED |
| R271b | Mini-BCD | **25% gap closure in 57s** | CONFIRMED |
| R276 | TB-BCD | 23× slower | REJECTED |
| R277 | OLS out_proj | 7.7% (weak) | NEUTRAL |
| R278 | W_shift rank-8 | -0.122 nats (3.5%) | CONFIRMED |
| R279 | Rounds vs FP | = FP (no improvement) | NEUTRAL |
| R296 | Full GD 6M curated | loss 1.93, 166 tok/s | CONFIRMED |

## FLOPs Analysis (R132, 7B)

| Metric | Value |
|--------|-------|
| F_layer_fwd | 1.79 TFLOPs (61% in_proj, 31% out_proj) |
| F_lmhead_fwd | 2.15 TFLOPs (LARGER than 1 layer!) |
| Pipeline 100 steps/stage | 393 PFLOPs, 26.2M tokens, ~24 min RTX5070 |
| Pipeline 2T tokens | 30,000 PFLOPs, ~12 years RTX5070 |
| Conventional 7B 2T | 84,000 PFLOPs, ~6 days on 1000 A100s |

## First Real Training Run (100M, 2026-03-23)

| Metric | Value |
|--------|-------|
| Model | 91.5M params, d=768, n=24, vocab=256 (byte-level) |
| Dataset | 438 MB (122K examples: smoltalk+glaive+everyday) |
| FP | 853 steps (64.5% saved by ABA), 86 min, loss 5.54→3.42 |
| Eval baseline | 3.34 nats |
| mini-BCD | 50 steps, 65s, loss 3.13→2.54 (-0.80 nats, +24%) |
| Final loss | **2.54 nats** (byte-level — garbage output) |
| Inference | 74.7 tok/s, 372 MB VRAM |
| VRAM FP peak | 6.5 GB (batch=64) |
| VRAM BCD peak | 380 MB (batch=4) |

## OVERNIGHT JOINT GD (2026-03-24, 55K steps) — BEST MODEL

| Metric | Value |
|--------|-------|
| Model | 108M (6L, d=1024, d_state=128, BPE 32K) |
| Training | **Joint GD** (not FP — FP stagnates at 6.3) |
| Dataset | 322M BPE tokens (390K examples, 10 sources) |
| Steps | 55,000 |
| Time | **2.6 hours** |
| Train loss | 2.11 nats |
| **Eval loss (V-PRIME)** | **1.86 nats** |
| VRAM train | 1.76 GB |
| VRAM inference | **368 MB** |
| Throughput | **379 tok/s** |
| Checkpoint | `data/checkpoints/overnight_joint/model_checkpoint.pt` |

**Generation quality (108M, 1.86 nats):**
- Greeting: coherent English, off-topic but grammatical (3/5)
- Math: correct reasoning FORMAT but wrong answer (15+27≠27)
- Explanation: mentions correct concepts, confused execution
- Code: **generates valid Python with def/for/if/return** (4/5)

**Key insight:** FP stagnated at 6.3 nats regardless of steps (structural gap).
Joint GD reached 1.86 nats — 3.4× better in same time.

## Overnight Training v1 (2026-03-24, 100 steps/stage)

| Metric | Value |
|--------|-------|
| Model | 108M (6L, d=1024, d_state=128, BPE 32K) |
| Dataset | 322M BPE tokens (390K examples, 10 sources) |
| FP | 445 steps (25.8% saved ABA), 352s, loss 10.37→6.30 |
| Eval | 6.34 nats |
| BCD | -0.003 nats (negligible) |
| Total time | **353s (~6 min)** |
| VRAM peak | 948 MB |
| Generation quality | Garbage (repetitive, no coherence) |
| **Verdict** | 100 steps/stage insufficient. Need 500+ |

## R306-R308 Implementations (2026-03-24)

| Change | Smoke Loss | Delta vs Baseline | Source |
|--------|-----------|-------------------|--------|
| CE sum gradient (not mean) | **8.45** | **-0.72 nats** vs 9.17 | BitNet b1.58, R306 |
| predictive_coding_rank 64→24 | (combined above) | ~0 (marginal) | R304 validated |

## R300 Diagnostics — Layer Contribution (2026-03-24)

| Layer | BI (Block Influence) | Probe Loss | Interpretation |
|-------|---------------------|------------|----------------|
| 0 | **0.8536** | 4.87 | Does ALL the work |
| 1 | 0.0586 | 11.42 | Minimal contribution |
| 2-22 | **<0.05** (all) | 10-89 | NEAR-IDENTITY, redundant |
| 23 | 0.2189 | 16.10 | Moderate (lm_head) |

**21 of 24 layers are near-identity.** Layers 2-22 DESTROY information (probe loss explodes).
Model is effectively 2-3 layers with 21 passthrough layers.
Confirms: 24 layers is massively over-deep for this dataset + FP training.

## R297 Quality Improvements (2026-03-23)

| Improvement | Impact | Source |
|-------------|--------|--------|
| RNS Embedding | 3335× fewer embedding params (7.3K vs 24.6M) | R297 validated |
| Predictive Coding | SSM processes surprise not raw content | R297 validated |
| TBD Init | Geometric multi-scale A_log init | R297 validated |
| Smoke test FP reduction | **10.4%** (vs 3.5% without improvements) | Measured |

## Streaming CE + Cached BCD (2026-03-23)

| Metric | Before (standard CE) | After (streaming CE) | Reduction |
|--------|---------------------|---------------------|-----------|
| VRAM stage0 step0 (100M, batch=16) | 10,419 MB | **1,752 MB** | **6×** |
| VRAM stage23 step0 | ~6,575 MB | **2,295 MB** | **3×** |
| BCD VRAM (100M, all layers) | ~10 GB (OOM) | **~150-200 MB** | **50×+** |

Streaming CE: validated, exact (no approximation), replaces F.cross_entropy at all sites.
BCD cached targets: MSE local training, INT8 compressed targets, 1 layer on GPU at a time.

## R309-R313 Implementations (2026-03-24)

| Change | Description | Source |
|--------|-------------|--------|
| σ_min(I-A_q) regularizer | Prevents attractor fixed points → less repetition | R309 |
| Trajectory Diversity Loss | Penalizes state convergence across sequences | R310 |
| Prompt-State Anchoring | Residual injection keeps model on-topic | R311 |
| EOS Entropy Head | Focal BCE binary head for info exhaustion | R311 |
| Adaptive Threshold τ_i* | Per-group std-based ternary thresholds | R312 |
| L_gram regularizer | Row-orthogonality for ternary weight diversity | R313 |

**Note:** These were verified via smoke test (Joint GD, 1 step each). Full training needed to measure impact.

**Training status:** Items 1-6 implemented AFTER the OVERNIGHT JOINT GD training (2026-03-24, 55K steps). That training run used baseline configuration without these additions. New training run needed to test all improvements together.

---

**Issues found (first real training run):**
- vocab=256 byte-level = each char is 1 token = 5× less efficient than BPE 32K
- ABA possibly too aggressive (stopping at 31 steps with loss still decreasing)
- Cache incremental (33× speedup) NOT integrated into train_production.py
- No LR schedule (flat LR)
- Model saw only ~3% of dataset (14M of 438M tokens)
- BCD batch_size=4 too conservative (380 MB used of 12.8 GB available)

## Key Ratios

| Comparison | Value | Source |
|------------|-------|--------|
| Our VRAM vs joint GD | 62x less (893 MB vs ~56 GB) | R130 |
| Our VRAM vs SBP | ~11x less (893 MB vs ~10 GB) | R130 vs R139 |
| INT2 vs FP32 storage | 16x compression | R100 |
| FP gap (100M) | 0.2281 nats (11%) | R139 |
| BCD closes | 28-32% of gap | R92 |
| mini-BCD closes | 25% in 57s | R271b |
| SAC closes | 35.1% of gap | R128 |
| FP+1000 GD steps | 89% of gap | R135 |
| Cache incremental speedup | 33× | R262/R265 |
| ABA step reduction | 47% | R270 |

## Important Conclusions (Import Experiments R218-R237)

- Transformer→SSM conversion: IMPOSSIBLE (softmax gap proven, R230)
- Ternary post-training: IMPOSSIBLE without fine-tune (R229/R241, PPL 8814 with GPTQ)
- Sliding window W=384: gap +4.8% (works, R228d)
- GQA makes KV cache cheap: 1 GB for 32B model with W=4096
- Best path: train from scratch with FP+BCD, NOT convert from transformer

---

## R337 Convergence Benchmark (2026-03-25)

| Metric | Overnight (baseline) | R337 (5500 steps) |
|--------|---------------------|-------------------|
| Steps | 55,000 | 5,500 |
| Time | 2.6h (156 min) | **21 min** (1271s) |
| Tokens seen | 225M | 11.3M (5% of dataset) |
| Train loss | 2.11 | 2.24 |
| **Eval loss** | **1.86** | **8.59** |
| ms/step | 195ms | 264ms |
| batch × seq | 16×256 = 4096 tok/step | 4×512 = 2048 tok/step |
| VRAM | 1.76 GB | 2.6 GB |
| use_ternary | False | True |
| Optimizer | AdamW | FactoredAdamW (natural gradient) |
| LR | 3e-4 | 0.1 |

**Key findings:**
- Train loss converges 10× faster (same train loss in 1/10 the steps)
- Eval gap (8.59 vs 1.86) is DATA COVERAGE: model saw 5% of dataset vs 70%
- Natural gradient tolerates 333× higher LR than Adam (measured: 3e-4→0.1 stable)
- Init loss 2.89 (eigenspectrum+B*+TBD) vs 10.4 (random) = 3.6× better start

---

## Overnight Training V2 (2026-03-25)

| Metric | Overnight V1 (baseline) | Overnight V2 |
|--------|------------------------|-------------|
| Steps | 55,000 | 18,000 |
| Time | 2.6h | 4h35min |
| ms/step | 195ms | 918ms |
| Tokens seen | 225M | 92M (1.15 epochs) |
| Train loss | 2.11 | **2.38** |
| Eval loss | **1.86** | **8.50** |
| batch × seq | 16×256=4096 | 10×512=5120 |
| use_ternary | False | **True** |
| Dataset | chat_full (322M, chat-heavy) | chat_v2 (80M, curated) |
| Features | None | ALL (MVUE, nat-grad, GP-KL, hooks, CSP, etc.) |

**Analysis:**
- Train loss comparable (2.38 vs 2.11) despite ternary + smaller dataset
- Eval gap (8.50 vs 1.86) is DATA COVERAGE: 92M tokens vs 225M tokens
- Eval stabilized at 8.50 from step ~8000 (no improvement after)
- Crystallization: grew during training (boundary filter 75%→76.5%)
- All features ran successfully for 18K steps without crash
- CSP calibrated at step 50 (co-spectral subspace active)
- OSP calibrated from full dataset at startup

**Key insight:** Eval is bottlenecked by dataset coverage, not model capacity.
The curated dataset (80M tokens) is too small for 78M params.
Next: either more data, or train on the larger chat_full (322M tokens) with new features.

---

## Code Audit Findings (2026-03-26)

Pipeline bugs discovered — invalidate R337 benchmark numbers:
- INT32 ternary forward was OFF (F.linear fp32 fallback)
- MVUE + natural gradient double-counting (Fisher^{-2})
- 4 loss conflicts, GRAM diagnostic useless
- Eval gap partly from state leakage + MVUE-det mismatch
- All numbers (918ms, 8.50 eval) measured on BUGGY pipeline

**All 11 bugs FIXED in commit 9f5f29a. Pipeline clean. Measured: 749ms/step, VRAM 2.5GB.**

---

## R368 M1 Runtime Cleanup + Deterministic Validation (2026-04-15)

| Metric | Value | Source |
|--------|-------|--------|
| S4-safe-gpu-only cleanup | Landed; dead GPU-only Vulkan plumbing removed from the active path; validated by clean Vulkan rebuild + 10-step smoke | R368 |
| Deterministic validation controls | `train.c` gained `--seed`, `--log-every`, `--loss-log-hex` for reproducible per-step logging | R368 |
| Post-S4 reproducibility | Two 100-step runs with `seed=42` produced 100/100 identical `[S5]` lines | R368 |
| `[M1 RECOMPUTE]` probe hits | 10,800 per run | R368 |
| Exit status attribution | Strong-validation harness observed exit code 1, but direct `train.exe` returned 0 and both logs ended at `[TRAIN] Complete` | R368 |
| N1 default path validation | Default `USE_VULKAN=1` build matched the prior validated reference on 100/100 `[S5]` lines | R368 |
| Old slice flags after N1 | Rebuilding with `-DV12_M1_SLICE_A -DV12_M1_SLICE_B` produced `diffFromDefault = 0` and `diffFromRef = 0` on 100-step `seed=42` `[S5]` logs | R368 |
| N1 smoke exit codes | Probe smoke and default smoke both returned exit code 0 | R368 |
| N1 activation proof | `[M1 RECOMPUTE]` fired on the defaultized path; old slice flags are now behavioral no-ops | R368 |
| N2 zero-hit removal | `fwd_a_eff_gate` and `a_eff_gate_out` now produce zero hits in `src/**/*.c`, `src/**/*.h`, and `src/**/*.comp` | R368 |
| N2 determinism | Two new 100-step `seed=42` default Vulkan runs matched each other on 100/100 `[S5]` lines | R368 |
| N2 equivalence vs post-N1 reference | Both new 100-step runs matched the post-`N1` validated reference with `diffFromRef = 0` | R368 |
| N2 runtime status | Clean Vulkan rebuild and 10-step smoke returned exit code 0 after removing the GPU forward-only gate cache path | R368 |
| Artifacts | `.workspace/research/references/r368-m1/` | R368 |

**Note:** The original formal `S5` closure versus a named pre-`S1` reference remains historically blocked. The repo does not contain that baseline, so no pre/post-`S1` zero-delta equivalence is claimed here. `N1` only proves that the default path now matches the best validated reference currently available in-repo.

---

## R386 Runtime Closure (2026-04-18)

| Family | Exact bytes | Exact MiB | Validated in |
|--------|------------:|----------:|--------------|
| `lm_head` | `394002432` | `375.750000` | `.work/jobs/r386-impl-01/validation.md` |
| `w_pred` | `12599296` | `12.015625` | `.work/jobs/r386-impl-02/validation.md` |
| `norm` | `32768` | `0.031250` | `.work/jobs/r386-impl-02/validation.md` |
| `embed` | `68021760` | `64.869629` | `.work/jobs/r386-impl-03/validation.md` |
| `W_in` | `6346752` | `6.052734` | `.work/jobs/r386-impl-04b/artifacts/fresh.log`, `.work/jobs/r386-impl-04b/validation.md` |
| `W_out` | `6346752` | `6.052734` | `.work/jobs/r386-impl-04b/artifacts/fresh.log`, `.work/jobs/r386-impl-04b/validation.md` |
| **Total** | **`487349760`** | **`464.772949`** | closed from slices 1-4 evidence |

The final exact runtime closeout exceeds the earlier research lower bound of `~405.75 MiB`; the current-`v12` validated runtime truth is `487349760` bytes (`~464.77 MiB`) of steady-state omitted host residency.

Validated continuity facts:

- checkpoint format on the validated resumed path is `v8`
- embed hashes were preserved on save/resume:
  - `embed_E`: `m=0xb83d3cfa09db0383`, `v=0xb83d3cfa09db0383`
  - `embed_P`: `m=0x0feb61957bfd0383`, `v=0x0feb61957bfd0383`
- per-layer persisted hashes were preserved on save/resume:
  - `W_in`: `aggregate=0xffae11df367a3b83`
  - `W_out`: `aggregate=0xffae11df367a3b83`
- real repack-boundary proof on the validated resumed path:
  - `[R386] repack boundary step=500 W_in/W_out path=GPU steady_state_host=off`

Runtime invariant:

- steady-state GPU ownership is now validated for the six mutable training-state families `lm_head`, `w_pred`, `norm`, `embed`, `W_in`, and `W_out`
- host mirrors are materialized only at named boundaries such as checkpoint save, checkpoint load/rebuild before GPU re-upload, explicit CPU fallback entry, and explicit debug/hash probe paths

---

## R364 Closeout Measurement (2026-04-26)

Source: `R364`

Artifacts:

- `.work/jobs/r364-impl-01/validation-slice2b.md`
- `.work/jobs/r364-impl-01/mathematician-gate.md`

Measured paired run facts:

- baseline command shape: `train.exe test_data.bin 12 --batch 1 --seed 42 --log-every 1 --teaching-off`
- baseline elapsed: `45.612s`
- baseline final step/loss: step `11`, loss `14.5662`
- R364 command shape: baseline run plus `--r364-pilot --r364-warmup 1 --r364-compile-every 2 --r364-compiled-steps 1`
- R364 elapsed: `74.318s`
- R364 final step/loss: step `11`, loss `14.5166`
- earliest R364 target reach: step `11`, same as the final reported step
- total compile time: `20214.019 ms` across `6` compiles
- exact native linear boundary result: the largest exactly linear native subtree is `lm_head` at `Cache.norm_out`
- disposition: no matched-objective speed signal; stopped as an implementation speed path; disabled diagnostic flag retained

Interpretation:

- Slice 2B showed that the implemented `lm_head` pilot is stable on short batch-1 Vulkan probes but slower than the baseline under matched command shape.
- The mathematician gate blocks expansion beyond `lm_head` under the current formulation because larger native upper-subtree extensions are not exactly linear and the locally linear additions are too small to plausibly repay the measured compile cost.

---

## R408 Corpus Scaffold Residual Pilot (2026-05-08)

Source: `R408`

Artifact:

- `.workspace/research/r408-speed-paradigm-rotor/corpus-scaffold-gate-closeout.md`

Measured low-scale gate facts:

| target | baseline | scaffold | result |
|---|---:|---:|---|
| CE `6.080136325208` | `0.277s`, `4,959,849` residual terms | `0.227s`, `0` residual terms | `1.220265159273x` wall-clock pass |
| CE `5.849073213003` | `0.563s`, `12,000,000` residual terms | `0.623s`, `7,000,000` residual terms | `1.71x` fewer residual terms, but wall-clock fail |
| CE `5.729831243270` | `1.004s`, `20,000,000` residual terms | `1.127s`, `16,000,000` residual terms | `1.25x` fewer residual terms, but wall-clock fail |

Memory estimates from the standalone CPU gate:

| run | allocated bytes estimate |
|---|---:|
| easy target | `537640448` bytes (`~512.7 MiB`) |
| 12M residual target | `806075904` bytes (`~768.7 MiB`) |
| 20M residual target | `1342946816` bytes (`~1.25 GiB`) |

Interpretation:

- This is a gate measurement, not a production training baseline.
- The scaffold has real original-CE and sample-efficiency signal.
- The current count-residual representation does not justify production
  integration because hard-target wall-clock fails after build and lookup
  overhead are charged.
- First real-path landing added an opt-in corpus bigram residual prior to the
  Model, CPU sampled-CE path, and Vulkan sampled-CE shader via
  `--r408-scaffold`, `--r408-scaffold-alpha`, `--r408-scaffold-scale`,
  `--r408-scaffold-terms`, and `--r408-scaffold-stride`.
- The Vulkan path uses a sparse open-addressed `(context,target)` count table
  instead of a dense `V x V` table. Activation must emit
  `[R408_SCAFFOLD_GPU] active=1 mode=vulkan-sparse-hash-log-prior ...`.
- Current GPU implementation status: production opt-in path landed and build
  validated. This is not yet a measured GPU speed baseline.
- Current measured build validation: `mingw32-make USE_VULKAN=1 shaders` and
  `mingw32-make USE_VULKAN=1 train` pass after wiring the sparse R408 buffers
  into the Vulkan sampled-CE descriptor layout and push constants.
- Current activation smoke:
  `train.exe test_data.bin 0 --batch 1 --seed 42 --log-every 1 --teaching-off --r408-scaffold --r408-scaffold-terms 1024`
  returned exit code `0`, emitted
  `[R408_SCAFFOLD_GPU] active=1 mode=vulkan-sparse-hash-log-prior pair_capacity=65536 unique_pairs=1015 terms=1024`,
  and reported `[VK] allocated: 728.73 MB` before the zero-step loop.
- Current R408 probe build validation:
  `mingw32-make -B r408_exact_multiplicity_gate r408_corpus_scaffold_gate r408_corpus_scaffold_residual_pilot`
  passes.
- Runtime GPU smoke must stay minimal on the active user PC: batch `1`, shortest
  step path available, inspect the activation log before any larger run.

---

## R409 CE-Equivalent Logit Control Variate Real-Path Smoke (2026-05-08)

Source: `R409`

Artifact:

- `.workspace/research/r409-speed-quality-training-paradigm/synthesis.md`

Implementation:

- Production-facing flags: `--corpus-logit-prior`,
  `--corpus-logit-prior-alpha`, `--corpus-logit-prior-scale`,
  `--corpus-logit-prior-terms`, `--corpus-logit-prior-stride`, and
  `--logit-prior-gate`.
- Legacy `--r408-scaffold*` aliases remain accepted as compatibility shims, but
  new public logs and the gate use mechanism names.
- The Vulkan sampled-CE shader now writes both candidate CE and baseline sampled
  CE in the same pass, allowing a real-path residual/base ratio without changing
  the training objective family.

Build validation:

- `mingw32-make USE_VULKAN=1 shaders` passed.
- `mingw32-make USE_VULKAN=1 train` passed.
- Post-hygiene zero-step activation smoke with mechanism-named interface passed:
  `train.exe test_data.bin 0 --batch 1 --seed 42 --log-every 1 --teaching-off --corpus-logit-prior --corpus-logit-prior-terms 1024 --logit-prior-gate`
  returned exit code `0`, emitted `[CORPUS_LOGIT_PRIOR_GPU] active=1`, used
  `NVIDIA GeForce RTX 5070`, and allocated `728.74 MB` before the zero-step
  loop.

Paired one-step smoke, batch `1`, `test_data.bin`, seed `42`, teaching off:

| path | command shape | loss | tok/s | VRAM |
|---|---|---:|---:|---:|
| baseline | `train.exe test_data.bin 1 --batch 1 --seed 42 --log-every 1 --teaching-off` | `11.7143` | `1231` | `728 MB` |
| corpus logit prior + gate | `train.exe test_data.bin 1 --batch 1 --seed 42 --log-every 1 --teaching-off --corpus-logit-prior --corpus-logit-prior-terms 1024 --logit-prior-gate` | `11.6785` | `1235` | `729 MB` |

Gate line from the candidate run:

- `[LOGIT_PRIOR_GATE] step=1 base_loss=10.379663 residual_loss=10.335491 residual_ratio=0.995744 batches=4`

Interpretation:

- Activation is confirmed on the RTX 5070 Vulkan path.
- The low-scale residual/base sampled-CE ratio is finite and favorable:
  residual CE is `0.4256%` lower than baseline sampled CE for this tiny
  one-step smoke.
- This smoke validates the real-path instrumentation and opt-in production
  embedding. It is not enough to project a 10% training-time gain by itself.
- The paired smoke shows no meaningful VRAM penalty at this scale (`+1 MB`) and
  no measurable throughput penalty inside one-step noise. No quality tradeoff is
  introduced by the gate path because the reported training loss is still CE on
  the realized logits.

---

## R409 Proof-Carrying Program Kernel Real-Path Smoke (2026-05-08)

Source: `R409`

Artifact:

- `.workspace/research/r409-speed-quality-training-paradigm/program-kernel-implementation.md`

Implementation:

- Production-facing flags: `--program-kernel`, `--program-kernel-gate`,
  `--program-kernel-logit`, `--program-kernel-min-count`,
  `--program-kernel-terms`, and `--program-kernel-stride`.
- The kernel builds a corpus-owned deterministic context-to-target table and
  abstains on conflicting or under-supported contexts.
- Vulkan sampled CE now consumes the table in the GPU path and records baseline
  sampled CE versus kernel-augmented sampled CE when the gate is active.

Build validation:

- `mingw32-make USE_VULKAN=1 shaders` passed.
- `mingw32-make USE_VULKAN=1 train` passed.

Current-corpus one-step smoke, batch `1`, `test_data.bin`, seed `42`, teaching off:

| path | command shape | routed entries | loss | tok/s | VRAM |
|---|---|---:|---:|---:|---:|
| baseline | `train.exe test_data.bin 1 --batch 1 --seed 42 --log-every 1 --teaching-off` | n/a | `11.7143` | `1260` | `728 MB` |
| program kernel gate | `... --program-kernel-gate --program-kernel-terms 4096 --program-kernel-min-count 1 --program-kernel-logit 4.0` | `0` | `11.7143` | `1256` | `728 MB` |

Interpretation:

- `test_data.bin` built `entries=0 conflicts=256`, so this fixture does not
  exercise the exact-kernel mechanism. This is a no-signal activation check, not
  a finding rejection.

Synthetic exact-domain smoke:

| fixture | command shape | base sampled CE | kernel sampled CE | ratio | routed/correct | VRAM |
|---|---|---:|---:|---:|---:|---:|
| `.work/tmp/program_kernel_repeat_A.bin` | `... --program-kernel-gate --program-kernel-terms 2048 --program-kernel-min-count 2 --program-kernel-logit 8.0` | `46.534023` | `38.534023` | `0.828083` | `4092/4092` | `728.24 MB` |

Gate line:

- `[PROGRAM_KERNEL_GATE] step=1 base_loss=46.534023 kernel_loss=38.534023 kernel_ratio=0.828083 batches=4 routed=4092 correct=4092 incorrect=0 route_accuracy=1.000000`

Interpretation:

- Activation is confirmed on the RTX 5070 Vulkan path.
- On a verified exact route domain, sampled CE drops by `17.1917%` with perfect
  route accuracy and negligible VRAM increase versus the current one-step
  baseline allocation (`727.99 MB -> 728.24 MB`).
- This is a real-path mechanism validation, not yet a broad-corpus
  training-time claim. Broader gains require richer exact kernels with nonzero
  coverage on real corpora.

---

## R409 State Transition Atlas Real-Path Smoke (2026-05-08)

Source: `R409`

Artifact:

- `.workspace/research/r409-speed-quality-training-paradigm/state-transition-atlas-implementation.md`

Implementation:

- Production-facing flags: `--state-transition-atlas`,
  `--state-transition-atlas-gate`, `--state-transition-atlas-logit`,
  `--state-transition-atlas-min-count`, `--state-transition-atlas-terms`, and
  `--state-transition-atlas-stride`.
- The atlas builds a corpus-owned exact local state table keyed by
  `(previous_token, current_token)` and abstains on conflict or insufficient
  support.
- CPU sampled CE and Vulkan sampled CE consume the atlas as an additive logit
  path, while the GPU gate records baseline sampled CE versus atlas sampled CE.

Build validation:

- `mingw32-make USE_VULKAN=1 shaders` passed.
- `mingw32-make USE_VULKAN=1 train` passed.

Paired one-step smoke, batch `1`, `test_data.bin`, seed `42`, teaching off:

| path | command shape | entries | loss | tok/s | VRAM |
|---|---|---:|---:|---:|---:|
| baseline | `train.exe test_data.bin 1 --batch 1 --seed 42 --log-every 1 --teaching-off` | n/a | `11.7143` | `1253` | `727.99 MB` |
| state transition atlas gate | `... --state-transition-atlas-gate --state-transition-atlas-terms 4096 --state-transition-atlas-min-count 1 --state-transition-atlas-logit 8.0` | `3857` | `9.4774` | `1258` | `728.74 MB` |

Gate line:

- `[STATE_TRANSITION_ATLAS_GATE] step=1 base_loss=10.379663 atlas_loss=2.901261 atlas_ratio=0.279514 batches=4 routed=1321 correct=1158 incorrect=163 route_accuracy=0.876609`

Interpretation:

- Activation is confirmed on the RTX 5070 Vulkan path.
- On the current small fixture, the exact local state atlas has nonzero
  real-corpus coverage and reduces sampled CE by `72.0486%` inside the gate.
- The paired smoke shows negligible VRAM increase (`+0.75 MB`) and no
  throughput penalty inside one-step noise.
- This is a low-scale real-path measurement, not yet a broad-corpus
  training-time claim. Broader gains require richer exact state coverage and
  matched target-time validation.

---

## R409 Counterfactual Minimal-Pair Packets Real-Path Gate (2026-05-08)

Source: `R409`

Artifact:

- `.workspace/research/r409-speed-quality-training-paradigm/counterfactual-packets-implementation.md`

Implementation:

- Production-facing flags: `--counterfactual-packets`,
  `--counterfactual-packets-gate`, `--counterfactual-packets-min-count`,
  `--counterfactual-packets-terms`, and `--counterfactual-packets-stride`.
- The packet ledger builds a corpus-owned local state table keyed by
  `(previous_token, current_token)` and stores the first two distinct observed
  next-token targets plus support counts.
- The current GPU path is a gate only: it records paired packet coverage and
  balance while preserving original CE as the unchanged objective.

Build validation:

- `mingw32-make USE_VULKAN=1 train` passed.

One-step smoke, batch `1`, `test_data.bin`, seed `42`, teaching off:

| path | command shape | packet states | loss | tok/s | VRAM |
|---|---|---:|---:|---:|---:|
| counterfactual packets gate | `... --counterfactual-packets-gate --counterfactual-packets-terms 4096 --counterfactual-packets-min-count 1` | `118` | `11.7143` | `1258` | `727.99 MB` |

Gate line:

- `[COUNTERFACTUAL_PACKETS_GATE] step=1 packets=65 primary=5 secondary=60 other=3 abstain=4020 coverage=0.052838 balance=0.074074 batches=4`

Interpretation:

- Activation is confirmed on the RTX 5070 Vulkan path.
- The gate exposes nonzero local paired packet signal on the current fixture,
  but coverage and balance are weak in this tiny one-step smoke.
- The reported training objective remains original CE. No quality tradeoff is
  introduced by the gate path, and no broad-corpus training-time gain is claimed.

---

## R409 Certified Silent-Span Real-Path Gate (2026-05-08)

Source: `R409`

Artifact:

- `.workspace/research/r409-speed-quality-training-paradigm/certified-silent-span-implementation.md`

Implementation:

- Production-facing flags: `--certified-silent-span-gate`,
  `--certified-silent-span-max-loss`, `--certified-silent-span-min-len`,
  `--certified-silent-span-cert-cost`, and
  `--certified-silent-span-repair-cost`.
- The gate uses per-token sampled CE from the Vulkan CE path and certifies a
  token as silent only when the CE-derived omitted logit-gradient bound is below
  the configured threshold.
- The gate records `silent_fraction`, `grad_bound`, and
  `priced_ratio = 1 - silent_fraction + cert_cost + repair_cost` while keeping
  original CE unchanged.

Build validation:

- `mingw32-make USE_VULKAN=1 train` passed.

One-step smoke, batch `1`, `test_data.bin`, seed `42`, teaching off:

| path | command shape | certified tokens | loss | tok/s | VRAM |
|---|---|---:|---:|---:|---:|
| certified silent-span gate | `... --certified-silent-span-gate` | `0 / 4092` | `11.7143` | `1258` | `727.99 MB` |

Gate line:

- `[CERTIFIED_SILENT_SPAN_GATE] step=1 tokens=0 positions=4092 spans=0 fraction=0.000000 grad_bound=0.000000 priced_ratio=1.000000 batches=4`

Interpretation:

- Activation is confirmed on the RTX 5070 Vulkan path.
- With the conservative default threshold `max_loss=0.10`, the current early
  one-step fixture exposes no certified silent mass.
- The finding is not falsified by this smoke. The current implementation lands
  the real-path certificate and pricing hook; a useful speed projection requires
  nonzero silent mass on later/easier training states or a stronger endpoint
  certificate.

---

## R409 Error-Causal Lattice Real-Path Gate (2026-05-08)

Source: `R409`

Artifact:

- `.workspace/research/r409-speed-quality-training-paradigm/error-causal-lattice-implementation.md`

Implementation:

- Production-facing flags: `--error-causal-lattice`,
  `--error-causal-lattice-gate`, `--error-causal-lattice-min-count`,
  `--error-causal-lattice-terms`, and `--error-causal-lattice-stride`.
- The gate builds corpus-owned target/prerequisite atom support counts and
  measures coverage-preserving orderability on the live GPU training stream.
- The current path does not change sampling, scheduling, or CE.

Build validation:

- `mingw32-make USE_VULKAN=1 train` passed.

One-step smoke, batch `1`, `test_data.bin`, seed `42`, teaching off:

| path | command shape | target/prereq atoms | loss | tok/s | VRAM |
|---|---|---:|---:|---:|---:|
| error-causal lattice gate | `... --error-causal-lattice-gate --error-causal-lattice-terms 4096 --error-causal-lattice-min-count 1` | `256 / 256` | `11.7143` | `1266` | `727.99 MB` |

Gate line:

- `[ERROR_CAUSAL_LATTICE_GATE] step=1 positions=4092 target_covered=4092 prereq_ready=4092 orderable=4092 rare_positions=0 rare_orderable=0 target_coverage=1.000000 prereq_readiness=1.000000 orderable_fraction=1.000000 rare_coverage=1.000000 batches=4`

Interpretation:

- Activation is confirmed on the RTX 5070 Vulkan path.
- The one-step fixture is fully covered by the atom ledger at `min_count=1`.
- This is a prerequisite-lattice infrastructure result, not a scheduling-speed
  result. A training-time claim requires an equal-priced curriculum pilot.

---

## R409 Residual Active Sampling Real-Path Gate (2026-05-08)

Source: `R409`

Artifact:

- `.workspace/research/r409-speed-quality-training-paradigm/residual-active-sampling-implementation.md`

Implementation:

- Production-facing flags: `--residual-active-sampling-gate`,
  `--residual-active-sampling-floor`, and
  `--residual-active-sampling-target-fraction`.
- The gate reads live Vulkan sampled-CE losses and prices a loss-proportional
  unbiased importance sampler with an ESS and variance proxy.
- The current path does not change sampling, gradients, or CE.

Build validation:

- `mingw32-make USE_VULKAN=1 train` passed.

One-step smoke, batch `1`, `test_data.bin`, seed `42`, teaching off:

| path | command shape | ESS fraction | variance proxy | loss | tok/s | VRAM |
|---|---|---:|---:|---:|---:|---:|
| residual active sampling gate | `... --residual-active-sampling-gate --residual-active-sampling-target-fraction 0.5` | `0.999996` | `2.000007` | `11.7143` | `1249` | `727.99 MB` |

Gate line:

- `[RESIDUAL_ACTIVE_SAMPLING_GATE] step=1 positions=4092 mean_loss=10.379663 score_cv=0.001910 ess_fraction=0.999996 target_fraction=0.500000 max_weight=1.005682 variance_proxy=2.000007 batches=4`

Interpretation:

- Activation is confirmed on the RTX 5070 Vulkan path.
- The tiny early fixture has a nearly uniform residual-score distribution, so
  half-token active sampling would be unbiased in expectation but carries about
  a `2.0x` variance proxy at the chosen fraction.
- This is a sampler-pricing infrastructure result, not a training-time speed
  result. A gain claim requires an actual importance-weighted sampler pilot.

---

## R409 Exact Transition Quotient Real-Path Gate (2026-05-08)

Source: `R409`

Artifact:

- `.workspace/research/r409-speed-quality-training-paradigm/exact-transition-quotient-implementation.md`

Implementation:

- Production-facing flags: `--exact-transition-quotient`,
  `--exact-transition-quotient-gate`, `--exact-transition-quotient-min-count`,
  `--exact-transition-quotient-terms`, and
  `--exact-transition-quotient-stride`.
- The gate builds a corpus-owned exact `(previous token, current token, target
  token)` multiplicity ledger and prices objective-identical quotient mass.
- The current path does not change sampling, gradients, or CE.

Build validation:

- `mingw32-make USE_VULKAN=1 train` passed.

One-step smoke, batch `1`, `test_data.bin`, seed `42`, teaching off:

| path | command shape | quotient ratio | saved fraction | loss | tok/s | VRAM |
|---|---|---:|---:|---:|---:|---:|
| exact transition quotient gate | `... --exact-transition-quotient-gate --exact-transition-quotient-terms 4096 --exact-transition-quotient-min-count 1` | `0.999512` | `0.000816` | `11.7143` | `1303` | `727.99 MB` |

Gate lines:

- `[EXACT_TRANSITION_QUOTIENT] active=1 mode=exact-local-triple-multiplicity min_count=1 stride=1 terms=4096 unique_terms=4094 supported_mass=4096 quotient_ratio=0.999512 table_capacity=65536`
- `[EXACT_TRANSITION_QUOTIENT_GATE] step=1 corpus_ratio=0.999512 positions=4088 supported=1224 live_coverage=0.299413 saved_terms=1 saved_fraction=0.000816 mean_multiplicity=1.000817 batches=4`

Interpretation:

- Activation is confirmed on the RTX 5070 Vulkan path.
- The current one-step fixture exposes almost no exact duplicate local triple
  mass: the exact quotient is effectively `1.0`.
- This closes the implementation as real-path landed but weak on the current
  data; no training-time speed claim is made.

## R410 Prefix Frontier Adjoint Real-Path Gate (2026-05-08)

Source: `R410`

Artifact:

- `.workspace/research/r410-speed-full-path/prefix-frontier-implementation.md`

Implementation:

- Production-facing flags: `--prefix-frontier-adjoint-gate`,
  `--prefix-frontier-min-count`, `--prefix-frontier-terms`,
  `--prefix-frontier-stride`, `--prefix-frontier-overhead`, and
  `--prefix-frontier-repair-cost`.
- The gate builds an exact local frontier multiplicity table keyed by
  `(previous_token, current_token)` and measures sampled-CE adjoint energy from
  live Vulkan `ce_d_logits` rows on reusable frontiers.
- The current path is a measurement gate only and leaves original CE unchanged.

Build validation:

- `mingw32-make USE_VULKAN=1 train` passed.

One-step smoke, batch `1`, `test_data.bin`, seed `42`, teaching off:

| path | command shape | priced ratio | loss | tok/s | VRAM |
|---|---|---:|---:|---:|---:|
| prefix frontier gate | `... --prefix-frontier-adjoint-gate --prefix-frontier-terms 4096 --prefix-frontier-min-count 2` | `0.999711` | `11.7143` | `1220` | `728 MB` |

Gate line:

- `[PREFIX_FRONTIER_GATE] step=1 positions=4088 supported=69 reuse_fraction=0.017123 energy_fraction=0.016879 mean_multiplicity=2.014493 saved_terms=70 priced_ratio=0.999711 batches=4`

Interpretation:

- Activation is confirmed on the RTX 5070 Vulkan path.
- The current tiny fixture produces finite metrics but weak useful frontier
  reuse after pricing. This does not clear the 10% material-gain bar and does
  not project a training-time speedup.
- The finding is not theoretically rejected by this smoke because the fixture
  has low frontier multiplicity and the current gate measures sampled-CE output
  adjoint eligibility, not a full recurrent-prefix adjoint cache.

---

## R410 Active State Subspace Real-Path Gate (2026-05-08)

Source: `R410`

Artifact:

- `.workspace/research/r410-speed-full-path/active-state-subspace-implementation.md`

Implementation:

- Production-facing flags: `--active-state-subspace-gate`,
  `--active-state-subspace-k`, and `--active-state-subspace-overhead`.
- The gate measures live Vulkan final-residual gradient energy after RMSNorm
  backward and prices a top-k active subspace against reversible-tail regret.
- The current path is a measurement gate only and leaves original CE unchanged.

Build validation:

- `mingw32-make USE_VULKAN=1 train` passed.

One-step smokes, batch `1`, `test_data.bin`, seed `42`, teaching off:

| path | command shape | k | top-k energy fraction | tail regret | priced ratio | loss | tok/s | VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| active state subspace gate | `... --active-state-subspace-gate --active-state-subspace-k 16` | `16` | `0.022658` | `0.977342` | `43.823940` | `11.7143` | `1206` | `728 MB` |
| active state subspace gate | `... --active-state-subspace-gate --active-state-subspace-k 128` | `128` | `0.160920` | `0.839079` | `5.991030` | `11.7143` | `1205` | `728 MB` |

Gate lines:

- `[ACTIVE_STATE_SUBSPACE_GATE] step=1 positions=4096 k=16 k_over_d=0.015625 energy_fraction=0.022658 tail_regret=0.977342 priced_ratio=43.823940 batches=4`
- `[ACTIVE_STATE_SUBSPACE_GATE] step=1 positions=4096 k=128 k_over_d=0.125000 energy_fraction=0.160920 tail_regret=0.839079 priced_ratio=5.991030 batches=4`

Interpretation:

- Activation is confirmed on the RTX 5070 Vulkan path.
- The implemented final-residual-gradient proxy is identifying and unfavorable:
  the top-k active state does not carry enough gradient energy to price below
  the current path on this smoke.
- This closes the implemented proxy as diagnostic-only. It does not reject every
  possible SSM-internal active-state formulation because the gate does not yet
  expose full per-layer SSM state-adjoint bases.

---

## R410 Absorbed State-Atlas Real-Path Gate (2026-05-08)

Source: `R410`

Artifact:

- `.workspace/research/r410-speed-full-path/absorbed-state-atlas-implementation.md`

Implementation:

- Production-facing flags: `--absorbed-state-atlas-gate`,
  `--absorbed-state-atlas-overhead`,
  `--absorbed-state-atlas-memory-cost`,
  `--absorbed-state-atlas-repair-cost`, and
  `--absorbed-state-atlas-kappa`.
- The gate reuses the exact local state-transition atlas and prices the
  residual sampled-CE burden after absorbing atlas-supported transitions.
- The current path is opt-in and leaves the default training path unchanged.

Build validation:

- `mingw32-make USE_VULKAN=1 train` passed.

One-step smoke, batch `1`, `test_data.bin`, seed `42`, teaching off:

| path | command shape | residual ratio | priced ratio | coverage | route accuracy | loss | tok/s | VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| absorbed state atlas gate | `... --absorbed-state-atlas-gate --state-transition-atlas-terms 4096 --state-transition-atlas-min-count 1 --state-transition-atlas-logit 8.0` | `0.279514` | `0.279514` | `0.323141` | `0.876609` | `9.4774` | `1238` | `729 MB` |

Gate lines:

- `[STATE_TRANSITION_ATLAS_GATE] step=1 base_loss=10.379663 atlas_loss=2.901261 atlas_ratio=0.279514 batches=4 routed=1321 correct=1158 incorrect=163 route_accuracy=0.876609`
- `[ABSORBED_STATE_ATLAS_GATE] step=1 base_loss=10.379663 residual_loss=2.901261 residual_ratio=0.279514 priced_ratio=0.279514 coverage=0.323141 route_accuracy=0.876609 batches=4 routed=1321 correct=1158 incorrect=163`

Interpretation:

- Activation is confirmed on the RTX 5070 Vulkan path.
- The low-scale gate clears the material-gain bar for sampled-CE residual
  burden: `priced_ratio=0.279514`, or roughly `3.58x` less priced residual
  burden with zero configured overhead.
- This is not yet a broad-corpus matched-training-time claim. It is a real-path
  decomposition signal that should be expanded with larger corpora and explicit
  atlas cost pricing.

---

## R410 Macro-Transcript Real-Path Gate (2026-05-08)

Source: `R410`

Artifact:

- `.workspace/research/r410-speed-full-path/macro-transcript-implementation.md`

Implementation:

- Production-facing flags: `--macro-transcript-gate`,
  `--macro-transcript-min-count`, `--macro-transcript-terms`,
  `--macro-transcript-stride`, `--macro-transcript-hardness`,
  `--macro-transcript-codec-cost`, and `--macro-transcript-decode-cost`.
- The gate builds an exact adjacent-pair dictionary and measures greedy
  reversible macro-token transcript compression on the real GPU training stream.
- The current path is a measurement gate only and leaves original CE unchanged.

Build validation:

- `mingw32-make USE_VULKAN=1 train` passed.

One-step smokes, batch `1`, `test_data.bin`, seed `42`, teaching off:

| path | command shape | min count | supported fraction | length ratio | priced ratio | loss | tok/s | VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| macro transcript gate | `... --macro-transcript-gate --macro-transcript-terms 4096 --macro-transcript-min-count 2` | `2` | `0.058838` | `0.983398` | `0.983398` | `11.7143` | `1262` | `728 MB` |
| macro transcript gate | `... --macro-transcript-gate --macro-transcript-terms 4096 --macro-transcript-min-count 1` | `1` | `1.000000` | `0.810791` | `0.810791` | `11.7143` | `1250` | `728 MB` |

Gate lines:

- `[MACRO_TRANSCRIPT_GATE] step=1 positions=4096 ops=4028 pair_macros=68 singletons=3960 pair_fraction=0.016882 length_ratio=0.983398 priced_ratio=0.983398 batches=4`
- `[MACRO_TRANSCRIPT_GATE] step=1 positions=4096 ops=3321 pair_macros=775 singletons=2546 pair_fraction=0.233363 length_ratio=0.810791 priced_ratio=0.810791 batches=4`

Interpretation:

- Activation is confirmed on the RTX 5070 Vulkan path.
- Repeated-support mode (`min_count=2`) is weak on this fixture and does not
  clear the 10% bar.
- Broad exact reversible coverage (`min_count=1`) clears the structural bar
  with `length_ratio=0.810791`, or about `1.23x` fewer transcript operations
  before explicit codec-memory pricing.
- This should stay opt-in until larger-corpus dictionary cost and matched-CE
  behavior are measured.

---

## R412 Quality-Content Gates (2026-05-09)

Source: `R412`

Artifact:

- `.workspace/research/r412-speed-quality-content/quality-content-implementation.md`

Implementation:

- `QC-1 Exact Window-Multiset Sufficient Statistic` landed behind
  `--exact-window-multiset-gate` with configurable window count, stride,
  minimum multiplicity, overhead, and noise pricing.
- `QC-2 Reversible Deterministic Expansion Factors` landed behind
  `--reversible-deterministic-expansion-gate` with configurable term limit,
  minimum deterministic run length, overhead, and noise pricing.
- Both gates are opt-in measurement paths on the real Vulkan training stream and
  leave original CE unchanged by default.

Build validation:

- `mingw32-make USE_VULKAN=1 train` passed.

One-step smokes, batch `1`, `test_data.bin`, seed `42`, teaching off:

| path | command shape | key ratio | live signal | audits | loss | tok/s | VRAM |
|---|---|---:|---:|---|---:|---:|---:|
| EWMS gate | `... --exact-window-multiset-gate --exact-window-multiset-windows 0 --exact-window-multiset-min-count 1` | `q=0.999936` | `saved_fraction=0.000733`, `live_coverage=1.000000` | `objective=1`, `gradient=1`, `collisions=0` | `11.7143` | `1234` | `728 MB` |
| RDEF gate | `... --reversible-deterministic-expansion-gate --reversible-deterministic-expansion-terms 8192 --reversible-deterministic-expansion-min-run 2` | `priced_ratio=1.000000` | `lambda=0.000000`, `eta=0.000000` | `inverse=1`, `inverse_failures=0` | `11.7143` | `1245` | `728 MB` |
| combined gates | both flags above | `EWMS q=0.999936`, `RDEF priced_ratio=1.000000` | `EWMS saved_fraction=0.000733`, `RDEF covered=0` | all passed | `11.7143` | `1244` | `728 MB` |

Gate lines:

- `[EXACT_WINDOW_MULTISET] active=1 mode=exact-window-sufficient-statistic min_count=1 stride=1 windows=8977 units=9183471 unique_units=9182886 duplicate_units=585 supported_mass=9183471 q=0.999936 overhead=0.000000 noise=0.000000 priced_ratio=0.999936 collisions=0 objective_audit=1 gradient_audit=1 table_capacity=33554432`
- `[EXACT_WINDOW_MULTISET_GATE] step=1 q=0.999936 positions=4092 supported=4092 live_coverage=1.000000 saved_terms=3 saved_fraction=0.000733 mean_multiplicity=1.000733 priced_ratio=0.999936 objective_audit=1 gradient_audit=1 collisions=0 batches=4`
- `[REVERSIBLE_DETERMINISTIC_EXPANSION] active=1 mode=fixed-repeat-run-transducer min_run=2 terms=8192 tokens=8192 covered=0 spans=0 lambda=0.000000 eta=0.000000 overhead=0.000000 noise=0.000000 priced_ratio=1.000000 inverse_failures=0 inverse_audit=1`
- `[REVERSIBLE_DETERMINISTIC_EXPANSION_GATE] step=1 positions=4096 covered=0 spans=0 coord_units=0 lambda=0.000000 eta=0.000000 priced_ratio=1.000000 inverse_audit=1 inverse_failures=0 batches=4`

Interpretation:

- Activation is confirmed on the RTX 5070 Vulkan path.
- EWMS now measures sliding model-visible windows rather than only aligned
  block starts, so the live stream is covered in the smoke. The tiny fixture has
  negligible duplicate mass and does not clear the R412 material-gain bar.
- RDEF inverse realization is active and exact, but the tiny fixture exposes no
  repeated deterministic separator spans and does not clear the R412
  material-gain bar.
- No broad training-time speedup is claimed from this closeout. Promotion
  requires a larger corpus/pricing run before any CE skipping or objective
  replacement.

---

## R405 S1 DPTH Training Gate (2026-05-11)

Source: `R405 S1`

Artifact:

- `.workspace/research/r405-s1-dpth-training-gate/summary.md`

Validated command shape:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\r405_run_dpth_training_gate.ps1 -Steps 390 -Batch 1 -Seed 40531 -LogEvery 30 -EvalEvery 130 -EvalBatches 4 -DpthPool 4 -PollSeconds 15 -JobName nightly-20260511-r405-dpth-390-seed40531
```

Operational validation:

- Monitored runner compiled `train.exe`, ran A0/A1/A2 sequentially, captured
  `gpu-telemetry.csv`, verified per-arm completion and analyzer output, and
  ended with no project `train.exe` left alive.
- GPU telemetry stayed inside the safety envelope: max memory about `3023 MB`,
  max temperature `66 C`, and max power `109.21 W`.

Scientific result:

| arm | held-out CE progress | priced progress proxy |
|---|---:|---:|
| A0 baseline | `26.450100000000003` | `14195.56520769231` |
| A1 DPTH packet | `25.420599999999997` | `13556.942623375431` |
| A2 loss-control | `25.4639` | `13632.920620668458` |

Analyzer decision:

```text
DPTH_PACKET_FAILS_TRAINING_GATE
packet_vs_baseline = 0.9550125285627358
packet_vs_loss_control = 0.9944268730518508
```

Interpretation:

- The CPU oracle pre-gate justified a real GPU packet-arm validation, but the
  current cheap DPTH packet sampler does not beat either control.
- A prior 520-step monitored run with seed `40521` completed safely but ended
  `INSUFFICIENT` because all arms had negative first-to-last held-out CE
  progress after a late rebound. Best-checkpoint audit at step 390 still did
  not clear the material gate: `packet_vs_baseline_best=1.0468304977413336`
  and `packet_vs_loss_best=1.0026448101536183`.
- This closes the current DPTH packet realization as not paper-positive and not
  training-speed positive. It does not prove a global no-go for all future
  boundary-event or hitting-time mechanisms.

---

## R414 Certified Artifact Boundary Gate (2026-05-09)

Source: `R414`

Artifact:

- `.workspace/research/r414-original-ce-witness/certified-artifact-boundary-implementation.md`

Implementation:

- `Certified Abstaining Artifact Boundary` landed behind
  `--certified-artifact-boundary-gate`.
- The first proof-rule family is `copy-current-proof-transducer`: a context
  token is certified only when the configured corpus prefix proves every
  observed continuation after that token is the same token, with no
  counterexample and at least `min_count` support.
- The route representation stores context-level counts, false counts, and a
  certification mask. It is not a train-position target table.
- The gate leaves ordinary original CE active for every abstained context.

Build validation:

- `mingw32-make USE_VULKAN=1 train` passed.

One-step smokes, batch `1`, seed `42`, teaching off:

| path | command shape | `b_K` | false route | CE residual | CE budget | priced ratio | loss | tok/s | VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real fixture | `test_data.bin ... --certified-artifact-boundary-gate --certified-artifact-boundary-terms 4096 --certified-artifact-boundary-min-count 1` | `0.000000` | `0.000000` | `0.000000000` | `0.001000000` | `1.000000` | `11.7143` | `1280` | `728 MB` |
| synthetic copy-current fixture | `.work/r414-certified-artifact-boundary/copy_current_u16.bin ... --certified-artifact-boundary-gate --certified-artifact-boundary-terms 4096 --certified-artifact-boundary-min-count 1` | `1.000000` | `0.000000` | `0.001000500` | `0.001000000` | `0.000000` | `11.7250` | `1275` | `728 MB` |

Gate lines:

- `[CERTIFIED_ARTIFACT_BOUNDARY_GATE] step=1 positions=4092 routed=0 correct=0 incorrect=0 abstain=4092 b_k=0.000000 false_route_rate=0.000000 ce_residual=0.000000000 ce_budget=0.001000000 priced_ratio=1.000000 route_bytes=288576 obligation_audit=1 non_lookup_audit=1 batches=4`
- `[CERTIFIED_ARTIFACT_BOUNDARY_GATE] step=1 positions=4092 routed=4092 correct=4092 incorrect=0 abstain=0 b_k=1.000000 false_route_rate=0.000000 ce_residual=0.001000500 ce_budget=0.001000000 priced_ratio=0.000000 route_bytes=288576 obligation_audit=1 non_lookup_audit=1 batches=4`

Interpretation:

- Activation is confirmed on the RTX 5070 Vulkan path.
- The real fixture has no certified copy-current route mass, so the training
  projection is unchanged: `495d` remains `495d`, and lazy `165-247d` remains
  `165-247d`.
- The synthetic positive fixture proves the route path can certify and route
  `100%` of positions with zero false routes, but the default CE budget is just
  below the exact full-routing residual for `epsilon=0.001`.
- No broad training-time speedup is claimed. Promotion requires a stronger
  proof-rule family or corpus pass with nonzero real `b_K`, zero false routes,
  CE residual inside budget, and `(1 - b_K) * kappa_R + h_total <= 0.90`.

---

## R415 Credit-Time Equalized Dynamics Gate (2026-05-09)

Source: `R415`

Artifact:

- `.workspace/research/r415-speed-dual/credit-time-equalized-dynamics-implementation.md`

Implementation:

- `SSM Credit-Time Equalized Dynamics` landed behind
  `--credit-time-equalized-dynamics`.
- The optional gate is enabled with `--credit-time-equalized-dynamics-gate`.
- The path keeps original CE gradients and applies an SSM credit-time diagonal
  preconditioner to CPU-owned SSM dynamics gradients in the Vulkan training
  path.
- The default training path is unchanged.

Build validation:

- `mingw32-make USE_VULKAN=1 train` passed.

One-step smokes, batch `1`, seed `42`, teaching off:

| path | command shape | activation | loss | tok/s | VRAM | gate result |
|---|---|---|---:|---:|---:|---|
| baseline | `test_data.bin 1 --batch 1 --seed 42 --log-every 1 --teaching-off` | ordinary Vulkan path | `11.7143` | `1270` | `728 MB` | n/a |
| credit-time | `test_data.bin 1 --batch 1 --seed 42 --log-every 1 --teaching-off --credit-time-equalized-dynamics --credit-time-equalized-dynamics-gate` | `[CREDIT_TIME_EQUALIZED_DYNAMICS] active=1` | `11.7143` | `1280` | `728 MB` | `priced_ratio=1.393748` |

Gate line:

- `[CREDIT_TIME_EQUALIZED_DYNAMICS_GATE] step=1 apply=1 params=732 mean_scale=0.899283 min_scale=0.521392 max_scale=2.000000 descent_gain=0.717490 line_shrink=1.000000 net_descent_gain=0.717490 priced_ratio=1.393748 batches=1`

Interpretation:

- Activation is confirmed on the RTX 5070 Vulkan path.
- The current default credit-time scaling fails the material gain gate on the
  small real fixture: `priced_ratio=1.393748`, above the `0.90` material bar.
- The baseline projection is unchanged: `495d` remains `495d`, and lazy
  `165-247d` remains `165-247d`.
- If this measured ratio were applied directly, it would project to `689.4d`
  and lazy `229.5-344.3d`, so no speedup is claimed.
- This is a negative gate for the landed configuration, not a global
  falsification of all possible SSM credit-time dynamics variants.

---

## R417 Boundary Liability Ledger Gate (2026-05-10)

Source: `R417`

Artifact:

- `.workspace/research/r417-paper-speed-synthesis/boundary-liability-ledger-implementation.md`

Implementation:

- `Boundary Liability Ledger` landed a first opt-in real-path gate behind
  `--boundary-liability-ledger-gate`.
- The gate measures post-RMSNorm original-CE final-residual gradient energy on
  the existing Vulkan path, intersects it with the live `boundary_mask`, and
  counts only boundary rows that are also gradient-null witnesses as removable
  candidates.
- The default training path is unchanged.

Build validation:

- `mingw32-make USE_VULKAN=1 train` passed.

Default-disabled smoke:

- `train.exe test_data.bin 0 --batch 1 --seed 42 --log-every 1 --teaching-off`
  passed and emitted no `BOUNDARY_LIABILITY_LEDGER_GATE` lines.

One-step activation smoke, batch `1`, seed `42`, teaching off:

| path | command shape | loss | tok/s | VRAM | gate result |
|---|---|---:|---:|---:|---|
| boundary liability ledger | `test_data.bin 1 --batch 1 --seed 42 --log-every 1 --teaching-off --boundary-liability-ledger-gate` | `11.7143` | `1232` | `728 MB` | `priced_gain=0.000000`, `pass=0` |

Gate line:

- `[BOUNDARY_LIABILITY_LEDGER_GATE] step=1 positions=4096 boundary_positions=11 boundary_fraction=0.002686 boundary_energy_fraction=0.002543 null_fraction=0.000977 boundary_null_fraction=0.000000 false_discharge_energy=0.000000000 overhead=0.000000 priced_gain=0.000000 pass=0 batches=4`

Expanded sensitivity checks:

| run | final boundary fraction | final boundary energy fraction | final null fraction | final boundary-null fraction | final priced gain | pass |
|---|---:|---:|---:|---:|---:|---:|
| `test_data.bin`, 5 steps, default epsilon | `0.004004` | `0.003990` | `0.000977` | `0.000000` | `0.000000` | `0` |
| `text_data.bin`, 3 steps, default epsilon | `0.005290` | `0.005112` | `0.000977` | `0.000000` | `0.000000` | `0` |
| `text_data.bin`, 3 steps, epsilon `0.001` | `0.005290` | `0.005112` | `0.000977` | `0.000000` | `0.000000` | `0` |
| `text_data.bin`, 3 steps, epsilon `0.1` | `0.005290` | `0.005112` | `0.000977` | `0.000000` | `0.000000` | `0` |
| `wiki_1mb.bin`, 3 steps, default epsilon | `0.003825` | `0.003767` | `0.000977` | `0.000000` | `0.000000` | `0` |
| `wiki_1mb.bin`, 10 steps, post live-SSM repair | `0.004248` | `0.004119` | `0.000977` | `0.000000` | `0.000000` | `0` |
| `text_data.bin`, 10 steps, post live-SSM repair | `0.005347` | `0.005463` | `0.000977` | `0.000000` | `0.000000` | `0` |

Interpretation:

- Activation is confirmed on the RTX 5070 Vulkan path.
- The gate is identifying: it observes real reset-boundary rows and real
  original-CE gradient energy.
- The first final-residual-gradient/null-boundary realization is negative across
  bounded validation. No checked boundary row is also a gradient-null witness,
  so the priced removable fraction is zero.
- The post live-SSM repair retest preserved this result on `wiki_1mb.bin` and
  `text_data.bin`, while a same-path FP probe confirmed nonzero SSM residual
  activity (`mean_effect_ratio=0.056895446`, `identifying=1`) on `wiki_1mb.bin`.
- The constant `null_fraction=0.000977` across corpus and epsilon settings is
  consistent with structurally CE-unused tail rows, not useful boundary work.
- No speedup, baseline change, or paper-ready claim is supported.
- This is not a global falsification of the R417/R415 boundary-liability line.
  It is specifically a failed realization of final-residual null-boundary
  removal. Reopening should require either substantially stronger warmup/corpus
  evidence or a richer SSM transition/adjoint boundary-flux witness with
  residual reconstruction and false-discharge harm priced.

---

