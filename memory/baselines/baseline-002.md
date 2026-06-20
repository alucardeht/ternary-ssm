# Measured Baselines — All Numbers

**Last updated:** 2026-05-21

---

## R417 Optimizer Integrity Continuation (2026-05-10)

Source: `R417`

Artifact:

- `.workspace/research/r417-paper-speed-synthesis/optimizer-integrity-continuation.md`

Implementation:

- `Optimizer Integrity Gate` remains behind `--optimizer-integrity-gate`.
- The underlying default Vulkan path now performs real GPU factored-Adam updates
  for all GPU-owned ternary groups: six `W_in`, six `W_out`, `lm_head`, and
  `W_pred`.
- Vulkan buffer handle semantics were repaired: slot `0` is now
  `VK_BUFFER_NULL`, real allocations start at slot `1`, and both `0` and
  `UINT32_MAX` are invalid. This prevents `calloc`-initialized optional handles
  from aliasing the first real buffer.
- `lm_head` uses an explicit vocab-to-hot row map for hot-row moment semantics.
- The factored Adam moment path now accumulates multi-workgroup-safe row/column
  second-moment sums for host-side `mean_v` computation.
- The repair is a training-fidelity fix, not a speed gain claim.

Build validation:

- `mingw32-make USE_VULKAN=1 shaders` passed.
- `mingw32-make USE_VULKAN=1 train` passed.

Initial failing gate before repair:

| path | command shape | loss | tok/s | VRAM | gate result |
|---|---|---:|---:|---:|---|
| optimizer integrity gate, pre-repair | `test_data.bin 1 --batch 1 --seed 42 --log-every 1 --teaching-off --optimizer-integrity-gate` | `11.7143` | `1268` | `728 MB` | `missing_update_groups=14`, `pass=0` |

Gate lines:

- `[OPTIMIZER_INTEGRITY_GATE] step=1 owned_ternary_groups=14 resident_ternary_groups=14 cpu_skip_groups=14 gpu_update_groups=0 missing_update_groups=14 pass=0 reason=gpu_owned_ternary_groups_have_no_optimizer_dispatch`
- `[OPTIMIZER_INTEGRITY_GATE] step=1 owned_ternary_groups=14 resident_ternary_groups=14 cpu_skip_groups=14 gpu_update_groups=0 missing_update_groups=14 pass=0 batches=1`

Repaired validation:

| run | command shape | gate result | loss | tok/s | VRAM |
|---|---|---|---:|---:|---:|
| one-step repaired | `test_data.bin 1 --batch 1 --seed 42 --log-every 1 --teaching-off --optimizer-integrity-gate` | `gpu_update_groups=14`, `missing_update_groups=0`, `pass=1` | `40.8025` | `1292` | `740 MB` |
| three-step repaired | `test_data.bin 3 --batch 1 --seed 42 --log-every 1 --teaching-off --optimizer-integrity-gate` | pass on steps `1`, `2`, and `3` | `40.8025 -> 39.7609 -> 39.6481` | `1304-1309` | `740 MB` |

Repaired gate line:

- `[OPTIMIZER_INTEGRITY_GATE] step=1 owned_ternary_groups=14 resident_ternary_groups=14 cpu_skip_groups=14 gpu_update_groups=14 missing_update_groups=0 pass=1 reason=all_gpu_owned_ternary_groups_updated`

Interpretation:

- This was a real compiled-path training-fidelity blocker, not intentional GPU
  freezing and not a warmup-only artifact.
- The optimizer contract is now repaired enough to unblock R417 multi-step
  optimizer-dynamics gates from this specific failure mode.
- The old `loss=11.7143` optimizer-integrity smoke is not a valid baseline for
  this issue. It was contaminated by the false-valid handle behavior: an
  unallocated `opt_dw_pred` handle could alias buffer slot `0` and zero
  `pred_A` through `gpu_zero_grad_accumulators()`.
- No speedup, baseline projection change, or paper-ready claim is made by this
  repair alone.

---

## R417 Evidence-Rate Commitment Gate (2026-05-10)

Source: `R417`

Artifact:

- `.workspace/research/r417-paper-speed-synthesis/evidence-rate-commitment-gate.md`

Implementation:

- `Evidence-Rate Commitment Gate` landed behind
  `--evidence-rate-commitment-gate`.
- Modes are selected with `--evidence-rate-commitment-mode current|tsa|gbar`.
- The gate compares the repaired current Vulkan ternary optimizer path against
  direct TSA-like and `g_bar` commitment modes.
- The implementation records evidence-rate telemetry on the GPU path: nonzero
  gradient elements, active/failed commitments, boundary hits, sign
  stability/conflict, virtual descent, applied descent, applied harm, priced
  ratio, and pass/fail.
- Validation found and repaired a descriptor-layout telemetry bug: the reduce
  shader used 12 bindings while its layout declared 11, and the update shader
  bound 10 descriptors while its layout declared 6.

Build validation:

- `mingw32-make USE_VULKAN=1 shaders` passed.
- `mingw32-make USE_VULKAN=1 train` passed.

Validation:

| mode | command shape | gate result | loss | tok/s | VRAM |
|---|---|---|---:|---:|---:|
| current | `test_data.bin 3 --batch 1 --seed 42 --log-every 1 --teaching-off --optimizer-integrity-gate --evidence-rate-commitment-gate --evidence-rate-commitment-mode current` | identifying telemetry, `priced_ratio=inf`, `pass=0` for direct TSA/gbar virtual alternatives | `40.8025 -> 39.7609 -> 39.6481` | `1272-1277` | `740 MB` |
| tsa | same fixture with `--evidence-rate-commitment-mode tsa` | `applied_commits=62850 -> 105575 -> 132488`, `applied_harm=1591.89172 -> 1550.93945 -> 1823.34033`, `priced_ratio=inf`, `pass=0` | `40.8025 -> 41.7554 -> 43.0045` | `825-843` | `740 MB` |
| gbar | same fixture with `--evidence-rate-commitment-mode gbar` | `applied_commits=62850 -> 94099 -> 106578`, `applied_harm=1591.89392 -> 1550.5 -> 1814.28271`, `priced_ratio=inf`, `pass=0` | `40.8025 -> 41.7554 -> 43.0045` | `813-867` | `740 MB` |
| current, post live-SSM repair | `wiki_1mb.bin 3 ... --evidence-rate-commitment-mode current` | finite telemetry, no applied commits, `pass=0` | `41.5045 -> 40.4159 -> 39.6436` | n/a | `740 MB` |
| tsa, post live-SSM repair | `wiki_1mb.bin 3 ... --evidence-rate-commitment-mode tsa --evidence-rate-force-repack` | `applied_descent=-27607.9492`, `applied_harm=82363.75`, `pass=0` after selected-arm repair | `41.5045 -> 306.0152 -> 249.3623` | n/a | `740 MB` |
| gbar, post live-SSM repair | `wiki_1mb.bin 3 ... --evidence-rate-commitment-mode gbar --evidence-rate-force-repack` | `applied_descent=-74304.8516`, `applied_harm=74304.8203`, `pass=0` after selected-arm repair | `41.5045 -> 388.0175 -> 797.0929` | n/a | `740 MB` |

Key gate lines:

- current step 1: `elements=35454976`, `nonzero=4943872`,
  `active=62850`, `failed=4881022`, `tsa_descent=-1591.89099`,
  `gbar_descent=-1591.89087`, `priced_ratio=inf`, `pass=0`.
- current step 3: `nonzero=4947967`, `active=116892`,
  `tsa_descent=-1728.35022`, `gbar_descent=-1728.92468`,
  `priced_ratio=inf`, `pass=0`.
- TSA step 3: `commitment_fraction=1.000000`,
  `applied_descent=-1070.55847`, `applied_harm=1823.34033`,
  `priced_ratio=inf`, `pass=0`.
- gbar step 3: `commitment_fraction=0.804368`,
  `applied_descent=-1814.28235`, `applied_harm=1814.28271`,
  `priced_ratio=inf`, `pass=0`.

Interpretation:

- The gate is now identifying on the repaired native Vulkan path.
- The tested direct TSA-like and `g_bar` commitment realizations fail the
  original-CE-per-priced-time gate and increase loss over the same 3-step
  fixture.
- A post live-SSM repair `wiki_1mb.bin` retest preserved the failure and
  exposed a selected-arm pass bug. Applied modes now pass only if the selected
  applied arm has positive descent and `applied_harm <= applied_descent`; after
  this repair, harmful `tsa` and `gbar` runs fail with
  `reason=applied_mode_harm_or_no_positive_descent`.
- No speedup, baseline projection change, or paper-ready positive claim is
  supported.
- This fails the direct commitment realization, not the broader native
  evidence-accumulation paper idea. Reopening requires a stricter commitment
  certificate, trust region, delayed-commit rule, reset/repair pricing, or
  sign-stability predictor that avoids the observed harmful direct flips.

---

## R417 FP Interface Gap Probe (2026-05-10)

Source: `R417`

Artifact:

- `.workspace/research/r417-paper-speed-synthesis/fp-interface-gap-probe.md`

Implementation:

- `FP Interface Gap Probe` landed behind `--fp-interface-gap-probe`.
- `--fp-interface-gap-effect-epsilon <float>` controls the near-identity
  threshold; default is `0.05`.
- The probe leaves CE, sampling, and optimizer behavior unchanged.
- It measures residual layer effect, downstream CE-gradient RMS, projection
  alignment, `x_proj_rms`, `x_cv_rms`, `b_eff_rms`, `h_rms`, `y_ssm_rms`, and
  `wout_nnz` on the current Vulkan training path.

Build validation:

- `mingw32-make USE_VULKAN=1 train` passed.

Validation:

| fixture | command shape | loss | VRAM | gate result |
|---|---|---:|---:|---|
| `test_data.bin` initial | `1 --batch 1 --seed 42 --log-every 1 --teaching-off --optimizer-integrity-gate --fp-interface-gap-probe` | `40.8025` | `740.14 MB` | `layers=24`, `near_identity_layers=24`, `mean_effect_ratio=0`, `max_grad_rms=0.0196544528`, `identifying=1`, `batches=4` |
| `text_data.bin` initial | same flags | `40.5320` | `740.14 MB` | `layers=24`, `near_identity_layers=24`, `mean_effect_ratio=0`, `max_grad_rms=0.0189722609`, `identifying=1`, `batches=4` |
| `test_data.bin` post-repair | same flags | `40.7839` | `740 MB` | `layers=24`, `near_identity_layers=22`, `mean_effect_ratio=0.0426213033`, `min_effect_ratio=0.0371272266`, `identifying=1`, `batches=4` |
| `text_data.bin` post-repair | same flags | `40.6226` | `740 MB` | `layers=24`, `near_identity_layers=16`, `mean_effect_ratio=0.0502656996`, `min_effect_ratio=0.0414666608`, `identifying=1`, `batches=4` |
| `wiki_1mb.bin` post-repair | same flags | `41.5045` | `740 MB` | `layers=24`, `near_identity_layers=4`, `mean_effect_ratio=0.056895446`, `min_effect_ratio=0.04626159`, `identifying=1`, `batches=4` |
| `test_data.bin` post-repair short train | `3 --batch 1 --seed 42 --log-every 1 --teaching-off --optimizer-integrity-gate` | `40.7839 -> 39.7562 -> 39.7009` | `740 MB` | optimizer integrity passed every checked step |

Key observations:

- `optimizer-integrity-gate` passed on the checked path with
  `gpu_update_groups=14`, `missing_update_groups=0`, `pass=1`.
- `W_out` is not empty: `wout_nnz` is about `19.7k` per SSM layer.
- Downstream CE gradient is finite at the residual boundary:
  `grad_rms` is about `0.01965` on `test_data.bin` and about `0.01897` on
  `text_data.bin`.

- Initial diagnostics showed `y_ssm_rms=0`, `delta_rms=0`, `effect_ratio=0`,
  and `near_identity=1` for all `24` captured layer/batch records on
  `test_data.bin` and `text_data.bin`.
- Follow-up instrumentation showed `x_proj_rms>0`, but `x_cv_rms=0` and
  `h_rms=0`, localizing the zero to the conv/gate input.
- `shadow_w_conv` was zero-initialized while the live forward path now uses
  conv. A small tap-0 pass-through initialization repaired the dead SSM path.

Interpretation:

- This is a repaired path-fidelity blocker and support probe, not a speedup.
- The old zero-SSM observation should not be used as a paper-grade negative
  claim; it was caused by stale initialization assumptions after conv entered
  the live forward path.
- Future FP, stagewise, or statistics-first SSM claims must run on the repaired
  live SSM residual path and clear held-out real-corpus, warmup-aware, priced
  wall-clock validation before being promoted.

## R415 CE Boundary-Flux Sheaf Endpoint Gate (2026-05-10)

Source: `R415`

Artifact:

- `.workspace/research/r415-speed-dual/boundary-flux-sheaf-implementation.md`

Implementation:

- `CE Boundary-Flux Sheaf Endpoint Gate` landed behind
  `--boundary-flux-sheaf-gate`.
- The gate measures live Vulkan SSM transition-credit energy from
  `dA_per_step` on reset-boundary and immediate pre-boundary rows.
- The first attempt found `total_energy=0` in `dA_per_step`, which made the
  probe non-identifying. The path-fidelity cause was the Vulkan SSM backward
  using diagonal `C_bilinear` for the recurrent adjoint even though forward
  output projection uses the full `C` matrix.
- The repair adds `output_project_backward` on the GPU path to compute
  `C^T @ d_y` after inverse WHT, and fixes `ssm_bwd_elementwise` descriptor
  binding to update all 13 storage buffers.

Build validation:

- `mingw32-make USE_VULKAN=1 shaders` passed.
- `mingw32-make USE_VULKAN=1 train` passed.

Validation:

| corpus | steps | final loss path | final endpoint fraction | final energy fraction | final priced gain | identifying | pass | VRAM |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `test_data.bin` | 1 | `40.7839` | `0.004395` | `0.004146` | `0.000017` | `1` | `0` | `740MB` |
| `wiki_1mb.bin` | 3 | `41.5045 -> 40.4121 -> 39.6361` | `0.006592` | `0.006474` | `0.000042` | `1` | `0` | `740MB` |
| `text_data.bin` | 3 | `40.6226 -> 39.5302 -> 39.6909` | `0.009603` | `0.005891` | `0.000035` | `1` | `0` | `740MB` |

Interpretation:

- Activation is confirmed on the RTX 5070 Vulkan path.
- The repaired gate is identifying: `dA_per_step` now has nonzero total
  transition-credit energy.
- The endpoint-only realization is negative. Reset/pre-reset endpoints carry
  only about `0.4%` to `0.65%` of checked transition-credit energy, below the
  10% material-gain floor even before overhead.
- No speedup, baseline change, or paper-ready positive claim is supported.
- This is not a global falsification of CE Boundary-Flux Sheaf. Reopen only
  with a richer segment-level adjoint tube or residual reconstruction witness,
  plus false-equivalence, false-skip harm, refresh, dispatch, and memory costs.

## R415 Certified Gradient-Null Witness Gate (2026-05-10)

Source: `R415`

Artifact:

- `.workspace/research/r415-speed-dual/gradient-null-witness-implementation.md`

Implementation:

- `Certified Gradient-Null Witness Gate` landed behind
  `--gradient-null-witness-gate`.
- The gate audits CPU- and GPU-resident original-CE gradient tensors after
  accumulation and scaling, before optimizers consume them.
- A tensor group is certified-null only when its full gradient RMS is below
  `--gradient-null-witness-epsilon`.
- The first raw element-count reading was biased: raw null parameter coverage
  was about `10.8%`, but sparse vocabulary/embedding tensors cannot be priced
  as dense backward work. The landed gate separates raw `element_fraction` from
  work-priced `saved_work_fraction`, and the pass condition uses the latter.

Build validation:

- `mingw32-make USE_VULKAN=1 train` passed.

Validation:

| corpus | steps | final loss path | raw element fraction | saved work fraction | omitted fraction | priced gain | identifying | pass | VRAM |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `test_data.bin` | 1 | `40.7839` | `0.108029` | `0.046156` | `0` | `0.046156` | `1` | `0` | `740MB` |
| `wiki_1mb.bin` | 3 | `41.5045 -> 40.4120 -> 39.6361` | `0.108035` | `0.046192` | `0` | `0.046192` | `1` | `0` | `740MB` |
| `text_data.bin` | 3 | `40.6226 -> 39.5302 -> 39.6909` | `0.108035` | `0.046192` | `0` | `0.046192` | `1` | `0` | `740MB` |

Interpretation:

- Activation is confirmed on the RTX 5070 Vulkan path.
- The gate is identifying: full-gradient tensor groups and total gradient
  energy are observed.
- Exact-null tensor groups are persistent and have zero omitted-gradient energy
  under the tensor-group certificate.
- The cheap tensor-group realization is negative after work pricing. Corrected
  saved-work coverage is about `4.6%`, below the 10% material-gain floor before
  witness build, verification, refresh, and memory costs.
- No speedup, baseline change, or paper-ready positive claim is supported.
- Reopen only with finer trainable contexts or recurrent segments whose
  work-priced saved fraction exceeds 10% while preserving the omitted-gradient
  audit.

---

## R416 CE-Normal Ternary Chamber Gate (2026-05-10)

Source: `R416`

Artifact:

- `.workspace/research/r416-training-speed-dual/ce-normal-chamber-gate-implementation.md`

Implementation:

- `CE-Normal Ternary Chamber` landed as an opt-in GPU force-bound audit behind
  `--ce-normal-chamber-gate`.
- The gate reads GPU-owned ternary shadow logits and gradients after GPU
  gradient scaling and before the GPU ternary optimizer.
- It separates stochastic cells (`|shadow| < 1`), deterministic saturated
  cells (`|shadow| >= 1`), already-crystallized cells
  (`|shadow| >= V12_CRYSTALLIZATION_TAU`), and candidate chamber cells
  (`1 <= |shadow| < V12_CRYSTALLIZATION_TAU`).
- It reports work-priced no-crossing fractions for `K=1`, `K=4`, and `K=16`.
- The default training path and original CE objective are unchanged.

Build validation:

- `mingw32-make USE_VULKAN=1 train` passed.

Validation:

| corpus | steps | loss path | chamber result | identifying | pass | VRAM |
|---|---:|---|---|---:|---:|---:|
| `test_data.bin` | 1 | `40.7839` | `stochastic=35454976`, `saturated=0`, `candidates=0`, `safe_k16=0`, `priced_k16=0` | `0` | `0` | `740MB` |
| `test_data.bin` | 10 | starts `40.7839` | every checked step stayed `saturated=0`, `candidates=0`, `safe_k16=0`, `priced_k16=0` | `0` | `0` | `740MB` |
| `text_data.bin` | 3 | starts `40.6226` | every checked step stayed `saturated=0`, `candidates=0`, `safe_k16=0`, `priced_k16=0` | `0` | `0` | `740MB` |
| `wiki_1mb.bin` | 3 | starts `41.5045` | every checked step stayed `saturated=0`, `candidates=0`, `safe_k16=0`, `priced_k16=0` | `0` | `0` | `740MB` |

Interpretation:

- Activation is confirmed on the RTX 5070 Vulkan path.
- The current early from-scratch training state exposes no deterministic ternary
  chamber cells; every audited GPU-owned ternary shadow remains in
  `|shadow| < 1`.
- This is a non-identifying gate result for the current state, not a rejection
  of the broader R416 chamber hypothesis.
- No speedup, baseline change, or paper-ready positive claim is supported.
- Reopen only when a compatible mature v12 checkpoint or a budgeted warmup first
  shows `candidate_work_fraction > 0`, then add missed-boundary replay,
  optimizer-state drift accounting, residual/original-CE harm audit, and full
  overhead pricing.

---

## R416 Boundary-Cotangent Credit Gate (2026-05-10)

Source: `R416`

Artifact:

- `.workspace/research/r416-training-speed-dual/boundary-cotangent-credit-gate-implementation.md`

Implementation:

- `Boundary-Cotangent Credit` landed as an opt-in GPU transition-credit
  reconstruction audit behind `--boundary-cotangent-credit-gate`.
- The gate reads the repaired Vulkan `gpu_dA_per_step` transition credit, keeps
  boundary/pre-boundary rows plus window endpoints as anchors, linearly
  reconstructs non-anchor rows, and compares the reconstruction to the exact
  original-CE transition credit.
- It prices anchor work, verification work, residual lost descent, and overhead.
- The default training path and original CE objective are unchanged.

Build validation:

- `mingw32-make USE_VULKAN=1 train` passed.

Validation:

| corpus | steps | loss path | anchor fraction | cosine | residual fraction | priced ratio | unverified priced ratio | identifying | pass | VRAM |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_data.bin` | 1 | `40.7839` | `0.005371` | `0.016161` | `1.394330` | `2.399701` | `1.399701` | `1` | `0` | `740MB` |
| `text_data.bin` | 1 | `40.6226` | `0.011230` | `-0.025884` | `1.545871` | `2.557101` | `1.557101` | `1` | `0` | `740MB` |
| `wiki_1mb.bin` | 1 | `41.5045` | `0.005859` | `0.149229` | `1.118282` | `2.124142` | `1.124142` | `1` | `0` | `740MB` |
| `wiki_1mb.bin` | 3 | `41.5045 -> 40.4121 -> 39.6361` | `0.007568` | `0.137153` | `1.150172` | `2.157741` | `1.157741` | `1` | `0` | `740MB` |

Interpretation:

- Activation is confirmed on the RTX 5070 Vulkan path.
- The repaired SSM backward path exposes nonzero exact transition credit, so the
  gate is identifying.
- The boundary-anchor-linear reconstruction is negative: compact anchors do not
  reconstruct the recurrent transition-credit cotangent. Cosine is low or
  negative and residual fraction stays above `1.0`.
- Full verification would also erase any speed claim, but residual alone kills
  this realization because even the unverified priced ratio stays above `1.0`.
- This fails only the first boundary-anchor-linear realization, not the broader
  forward-eligibility family.
- No speedup, baseline change, or paper-ready positive claim is supported.

---

## R416 Variance-Priced Credit Allocation Gate (2026-05-10)

Source: `R416`

Artifact:

- `.workspace/research/r416-training-speed-dual/variance-priced-credit-gate-implementation.md`

Implementation:

- `Variance-Priced Credit Allocation` landed as an opt-in GPU audit behind
  `--variance-priced-credit-gate`.
- The gate treats live `gpu_dA_per_step` rows as recurrent credit items, scores
  them by norm, simulates a deterministic unbiased sampled update, and audits
  it against the exact full transition-credit update.
- The default training path and original CE objective are unchanged.

Build validation:

- `mingw32-make USE_VULKAN=1 train` passed.

Default paid validation:

| corpus | steps | loss path | sampled fraction | norm ESS fraction | alignment | variance proxy | priced work | descent per paid work | pass | VRAM |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_data.bin` | 1 | `40.7839` | `0.246582` | `0.880166` | `0.980506` | `0.552685` | `2.246582` | `0.190432` | `0` | `740MB` |
| `text_data.bin` | 1 | `40.6226` | `0.245605` | `0.849285` | `0.993189` | `0.028781` | `2.245605` | `0.429465` | `0` | `740MB` |
| `wiki_1mb.bin` | 1 | `41.5045` | `0.245809` | `0.891970` | `0.990792` | `0.014523` | `2.245809` | `0.434707` | `0` | `740MB` |
| `wiki_1mb.bin` | 3 | `41.5045 -> 40.4121 -> 39.6361` | `0.247206` | `0.842905` | `0.991684` | `0.015712` | `2.247206` | `0.434304` | `0` | `740MB` |

Sensitivity:

- With `--variance-priced-score-fraction 0 --variance-priced-verify-fraction 0`,
  `wiki_1mb.bin` three-step final produced `descent_per_paid_work=3.948010` and
  `pass=1`.

Interpretation:

- Activation is confirmed on the RTX 5070 Vulkan path.
- The sampled transition-credit estimator is aligned on `text_data.bin` and
  `wiki_1mb.bin`; variance itself is not the immediate blocker there.
- The current fully scored and fully verified realization is negative because
  scoring plus exact verification cost more than the work removed.
- This is a conditional positive signal for the estimator math, not a deployed
  speedup. Reopen only with a cheap or amortized scoring path plus a cheaper
  audit/certificate that preserves original CE.
- No baseline change or deployed speedup is claimed.

---

## R419 TSSI CPU Pre-Gate And Residual Pilot (2026-05-11)

Source: `R419`

Artifact:

- `.workspace/research/r419-dataset-import-ternary/gate-closeout.md`

Measurement:

| metric | value |
|---|---:|
| dataset | `src/model_v12/wiki_1mb.bin` |
| tokenization scope | byte-level v12 shard, `vocab=256` |
| tokens | `500000` |
| split | `300000 train / 100000 calib / 100000 held-out` |
| feature dimension | `256` |
| selected feature mode | `bigram` |
| selected density | `0.05` |
| selected scale | `2.4` |
| realized density | `0.0496978759765625` |
| uniform CE | `5.545177444479562` |
| held-out CE | `2.949221092857901` |
| random held-out CE mean | `5.722178622339609` |
| random held-out CE min | `5.648630028780756` |
| sign alignment | `0.9941664108074916` |
| weighted sign alignment | `0.9986051321029663` |
| weighted random sign alignment mean | `0.45805180817842484` |
| decision | `PASS_PRE_GATE` |

Residual original-CE pilot:

| metric | value |
|---|---:|
| dataset | `src/model_v12/wiki_1mb.bin` |
| tokenization scope | byte-level v12 shard, `vocab=256` |
| tokens | `500000` |
| feature mode | `bigram` |
| density | `0.05` |
| scale | `2.4` |
| pilot steps | `1000` |
| batch size | `1024` |
| learning rate | `1.0` |
| seed | `777` |
| target CE | `2.9492211233757843` |
| TSSI target total seconds | `0.8049524002708495` |
| best control target total seconds | `2.8539996999315917` |
| target speedup vs best control | `3.5455508909238374` |
| TSSI final CE | `2.5494919515235193` |
| best control final CE | `2.78477980855248` |
| same-budget CE advantage | `0.2352878570289607` |
| sign-shuffled target reached | no |
| decision | `PASS_RESIDUAL_PILOT_MATERIAL_HEADSTART` |

Deterministic SSM-state proxy:

| metric | value |
|---|---:|
| dataset | `src/model_v12/wiki_1mb.bin` |
| tokenization scope | byte-level v12 shard, `vocab=256` |
| tokens | `500000` |
| feature mode | `ssm_state` |
| density | `0.01` |
| scale | `1.6` |
| held-out CE | `3.303536691372773` |
| random held-out CE mean | `5.581430338229163` |
| weighted sign alignment | `1.0` |
| seed 777 target speedup vs best control | `3.6053478120822366` |
| seed 777 same-budget CE advantage | `0.05614244344347741` |
| seed 419 target speedup vs best control | `3.5931313221153545` |
| seed 419 same-budget CE advantage | `0.05664628606735311` |
| decision | `PASS_RESIDUAL_PILOT_MATERIAL_HEADSTART` |

Live Vulkan SSM trace gate:

| metric | value |
|---|---:|
| dataset | `src/model_v12/wiki_1mb.bin` |
| tokenization scope | byte-level v12 shard, `vocab=256` |
| train trace | `10` steps, batch `1`, seed `777` |
| trace shape | `10 x 6 x 1 x 1024 x 128` |
| activation | `[R395_TRACE] hidden_cache_materialized source=vulkan_fwd_h_cache` |
| nonzero state values | `7864320 / 7864320` |
| state abs mean | `0.0001635154039831832` |
| selected layer mode | `all` |
| selected density | `0.05` |
| selected scale | `3.2` |
| realized density | `0.049530029296875` |
| calibration CE | `5.342514858450941` |
| held-out CE | `5.346169927602639` |
| random held-out CE mean | `5.54442637453797` |
| random held-out CE min | `5.5440281246181575` |
| sign alignment | `0.987779831587595` |
| weighted sign alignment | `0.9970957040786743` |
| weighted random sign alignment mean | `0.5062233805656433` |
| residual pilot decision | `PASS_RESIDUAL_PILOT_INITIAL_AND_FINAL_CE` |
| TSSI target total seconds | `80.27210727770999` |
| TSSI final CE | `5.345970772811278` |
| best control final CE | `5.544401972408052` |
| same-budget CE advantage | `0.19843119959677402` |
| target speedup vs best control | null |
| decision | `PARTIAL_SIGNAL_LIVE_TRACE_CE_ONLY` |

Record-disjoint live-trace rotations:

| metric | value |
|---|---:|
| split mode | whole trace records |
| rotations | `0,2,4,6,8` |
| held-out record pairs | `(8,9), (0,1), (2,3), (4,5), (6,7)` |
| all-layer rotations passed | `5 / 5` |
| all-layer CE margin vs best random min, min | `0.189190727524743` |
| all-layer CE margin vs best random min, mean | `0.204293772528715` |
| all-layer CE margin vs best random min, max | `0.238266453598485` |
| last-layer rotations passed | `5 / 5` |
| last-layer mean CE margin vs best random min | `0.0327855562301441` |
| mean-layer rotations passed | `5 / 5` |
| mean-layer mean CE margin vs best random min | `0.013581927197886` |

Interpretation:

- This is a positive CPU-only identifying pre-gate for Ternary
  Sufficient-Statistic Import, plus positive residual pilots in the readout-like
  byte/bigram scope, deterministic recurrent `ssm_state` proxy, and live Vulkan
  SSM hidden-state trace scope. The record-disjoint rotations reduce the
  flattened-example leakage and small-heldout-suffix concern.
- It is not a production GPU speedup, not an integrated production-path result,
  and not a baseline projection change. It is now a scoped paper candidate for
  ternary corpus-statistic import, but not a paper-ready speedup claim.
- The next required gate is an amortized online TSSI compiler or opt-in
  production import path that preserves the live all-layer SSM signal without
  full trace export, with zero/random/sign-shuffled controls and compiler cost
  included.

---

## R420 Online TSSI Compiler Research Gate (2026-05-12)

Source: `R420`

Artifacts:

- `.workspace/research/r420-tssi-training-impact/online-tssi-compiler-implementation.md`
- `.workspace/research/r420-tssi-training-impact/online-tssi-live-gpu-gate.md`
- `.workspace/research/r420-tssi-training-impact/online-tssi-score-accumulator.md`
- `.workspace/research/r420-tssi-training-impact/online-tssi-compact-residual-pricing.md`
- `.workspace/research/r420-tssi-training-impact/online-tssi-smoke-small.json`
- `.workspace/research/r420-tssi-training-impact/online-tssi-smoke-ssm-state.json`
- `.workspace/research/r420-tssi-training-impact/online-tssi-compact-smoke-small.json`
- `.workspace/research/r420-tssi-training-impact/online-tssi-compact-smoke-ssm-state.json`
- `.workspace/research/r420-tssi-training-impact/online-tssi-compact-smoke-ssm-state-a025.json`

Measurement:

| metric | byte/bigram smoke | ssm_state proxy smoke |
|---|---:|---:|
| dataset | `data/wikitext103/train.bin` | `data/wikitext103/train.bin` |
| tokens | `20000` | `30000` |
| split | `12000 / 4000 / 4000` | `18000 / 6000 / 6000` |
| feature dimension | `32` | `64` |
| density | `0.05` | `0.05` |
| scale | `2.4` | `2.4` |
| online accumulate seconds | `0.005763199646025896` | `0.057553000282496214` |
| realize seconds | `0.013827500399202108` | `0.022945300210267305` |
| verify seconds | `0.00803190004080534` | `0.019011899828910828` |
| offline equivalence max abs | `9.313225746154785e-10` | `3.814697265625e-06` |
| TSSI held-out CE | `9.3417275827746` | `9.176069506050487` |
| zero initial CE | `10.375489258056701` | `10.375489257975287` |
| random initial CE | `10.487458778757189` | `10.472506037790152` |
| sign-shuffled initial CE | `11.301582597797887` | `10.748371066401692` |
| same-budget CE advantage | `1.0336118575542326` | `1.1990446894143822` |
| decision | `PASS_ONLINE_TSSI_RESIDUAL_GATE` | `PASS_ONLINE_TSSI_RESIDUAL_GATE` |

Interpretation:

- The opt-in script `scripts/research/r420_online_tssi_compiler_gate.py`
  validates that TSSI sufficient statistics can be accumulated online in
  process and reproduce the offline compiler without hidden trace export.
- This is a positive implementation-style research gate for the compiler
  accounting path, not a production GPU speedup, not a baseline projection
  change, and not yet the live all-layer Vulkan hidden-state import path.
- The next required object is a GPU-resident hidden-state accumulator or
  equivalent live moment buffer that avoids hidden-cache trace export and avoids
  adding a second state-materialization pass.
- Addendum live Vulkan audit: `--online-tssi-live-gate` on the RTX 5070,
  `data/wikitext103/train.bin`, one step, batch `1`, log every `1`, teaching
  off, emitted `state_layer_fraction=1.000000`,
  `moment_layer_fraction=1.000000`, `sampled_ce_buffers=4`,
  `missing_state_score_layers=24`, `no_hidden_cache_materialization=1`,
  `identifying=0`, `pass=0`, `batches=4`. This confirms live GPU state and
  moment residency, but the target-conditioned state-score buffer is still
  missing. No speedup or baseline projection change is claimed.
- Addendum target-score accumulator: `--online-tssi-live-gate` on the RTX 5070,
  `data/wikitext103/train.bin`, one step, batch `1`, log every `1`, teaching
  off, emitted `state_layer_fraction=1.000000`,
  `moment_layer_fraction=1.000000`, `score_layer_fraction=1.000000`,
  `sampled_ce_buffers=4`, `score_updates=24552`,
  `missing_state_score_layers=0`, `no_hidden_cache_materialization=1`,
  `identifying=1`, `pass=1`, `batches=4`. This confirms the live compact
  target-conditioned state-score accumulator. No speedup or baseline projection
  change is claimed.
- Addendum compact residual pricing: `scripts/research/r420_online_tssi_compiler_gate.py`
  with `--compact-accumulator-pricing` selects active dimensions by compact
  absolute state score and compares TSSI against zero/random/sign-shuffled
  controls with build/realize/verify/select cost charged. Byte/bigram
  `20000`-token smoke at active fraction `0.50` produced compact target CE
  `9.30684111457552` versus full TSSI CE `9.3417275827746` and same-budget CE
  advantage `1.0685678011299693`. Deterministic `ssm_state` `30000`-token
  smoke at active fraction `0.50` produced compact target CE
  `9.108312444977809` versus full TSSI CE `9.176069506050487` and same-budget
  CE advantage `1.2669224167569606`; active fraction `0.25` kept same-budget
  CE advantage `1.1324311701689869`. This is a positive compact-pricing proxy,
  not a production speedup or baseline projection change.

---

## R422 Original-CE Credit Debt Gate (2026-05-12)

Source: `R422`

Artifacts:

- `.workspace/research/r422-extraordinary-paradigm-horizons/original-ce-credit-debt-implementation.md`
- `scripts/research/original_ce_credit_debt_gate.py`
- `.workspace/research/r422-extraordinary-paradigm-horizons/original-ce-credit-debt-test.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/original-ce-credit-debt-text.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/original-ce-credit-debt-pattern.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/original-ce-credit-debt-small.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/original-ce-credit-debt-text-128.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/original-ce-credit-debt-observer-smoke2.out`
- `.workspace/research/r422-extraordinary-paradigm-horizons/original-ce-credit-debt-observer-text_data.out`
- `.workspace/research/r422-extraordinary-paradigm-horizons/original-ce-credit-debt-observer-wiki_1mb.out`

Measurement:

| scope | result |
|---|---|
| CPU gate | `PASS_LOCAL_DIRECT_CREDIT_GATE_SPARSE_SURFACE_NOT_PASSED` on all five real-token smokes |
| CPU local progress | `0.944811` to `0.978798` |
| CPU local residual | `0.158299` to `0.222034` |
| CPU whole-step estimate | `1.229014x` to `1.243235x`, projection only |
| sparse surface | failed; best surface progress stayed around `0.350304` to `0.455205` with dense residual |
| real v12 observer | activated on RTX 5070 with `objective=unchanged_ce`, `observer_only=1`, `no_training_change=1` |
| real v12 local progress | `0.192524` to `0.210093` |
| real v12 local residual | `0.805090` to `0.812126` |
| real v12 local priced ratio | `2.534235` to `2.764009` |
| real v12 status | current embedding fails promotion; no speedup or baseline change |

Interpretation:

- CPU evidence preserves local direct original-CE credit as a submechanism.
- The stronger sparse active-surface certificate failed.
- The real v12 observer is now identifying, but the current live local-credit
  sketch is too weak to replace dense recurrent credit.
- No production training behavior changed and no measured speedup is claimed.

---

## R422 Reversible State Ledger Gate (2026-05-12)

Source: `R422`

Artifacts:

- `.workspace/research/r422-extraordinary-paradigm-horizons/reversible-state-ledger-implementation.md`
- `scripts/research/r422_reversible_state_ledger_gate.py`
- `.workspace/research/r422-extraordinary-paradigm-horizons/reversible-state-ledger-main.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/reversible-state-ledger-highload.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/reversible-state-ledger-dictionary-control.json`

Measurement:

| scope | decision | key result |
|---|---|---|
| main structured binding gate | `PASS_REVERSIBLE_STATE_LEDGER_GATE` | decode `0.999685`; CE gain vs keyed same-state `1.190161`; vs bitmatched state `0.602692`; vs 4x state `0.338157`; false accept `0.000000` |
| high-load control | `FAIL_UNSTABLE_REVERSIBLE_DECODE` | decode falls to `0.952444`, below the `0.98` gate |
| dictionary-collapse control | `FAIL_HIDDEN_RETRIEVAL_MEMORY_CONTROL` | short-key full dictionary memory ratio `0.562855`, so the ledger is not admissible in that regime |

Interpretation:

- Positive CPU structured-task gate for proof-carrying reversible state.
- The mechanism is scope-limited: it needs a load certificate and must fail when
  a complete retrieval/dictionary table is cheaper than the certified ledger.
- No production v12 path, no GPU speedup, and no broad training-quality baseline
  change is claimed.

---

## R422 Lossless Packet CE Gate (2026-05-12)

Source: `R422`

Artifacts:

- `.workspace/research/r422-extraordinary-paradigm-horizons/lossless-packet-ce-implementation.md`
- `scripts/research/r422_lossless_packet_ce_gate.py`
- `.workspace/research/r422-extraordinary-paradigm-horizons/lossless-packet-wiki-main.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/lossless-packet-text.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/lossless-packet-testdata.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/lossless-packet-cyberjutsu-test.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/lossless-packet-books-tokenized.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/lossless-packet-combined10-val.json`

Measurement:

| shard | decision | packet ratio | priced ratio | macro coverage | shuffled margin |
|---|---|---:|---:|---:|---:|
| `wiki_1mb.bin` | `FAIL_SHUFFLED_PACKET_CONTROL` | `0.330825` | `0.379896` | `0.947188` | `0.088216` |
| `text_data.bin` | `PASS_LOSSLESS_PACKET_CE_GATE` | `0.126667` | `0.186660` | `1.000000` | `0.439863` |
| `test_data.bin` | `FAIL_PRICED_WORK_RATIO` | `1.000000` | `1.050067` | `0.000000` | `-0.000340` |
| `books_tokenized/train.bin` | `FAIL_SHUFFLED_PACKET_CONTROL` | `0.747225` | `0.804606` | `0.417562` | `0.077427` |
| `combined_10/val.bin` | `FAIL_PACKET_COUNT_REDUCTION` | `0.820533` | `0.877461` | `0.321433` | `0.066296` |
| `cyberjutsu/test_processed/tokens.bin` | `PASS_LOSSLESS_PACKET_CE_GATE` | `0.304200` | `0.381289` | `0.833400` | `0.196056` |

Interpretation:

- Positive only as a structured text/code-like packet signal.
- Exact reconstruction and zero chain-rule identity error held in all retained
  runs.
- Broad-corpus packet training is not validated: wiki/books/combined either
  fail shuffled controls or fail packet-count reduction.
- No production v12 path, GPU speedup, or broad baseline change is claimed.

---

## R422 Capability Closure Gate (2026-05-12)

Source: `R422`

Artifacts:

- `.workspace/research/r422-extraordinary-paradigm-horizons/capability-closure-implementation.md`
- `scripts/research/r422_capability_closure_gate.py`
- `.workspace/research/r422-extraordinary-paradigm-horizons/capability-closure-main.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/capability-closure-budget240.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/capability-closure-budget480.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/capability-closure-budget1440.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/capability-closure-concepts32.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/capability-closure-seed1001.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/capability-closure-seed1002.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/capability-closure-seed1003.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/capability-closure-seed1004.json`

Measurement:

| run | decision | score gain | total gain | exception gain | normal regression | capability/budget ratio |
|---|---|---:|---:|---:|---:|---:|
| default budget `1200` | `FAIL_CAPABILITY_GAIN` | `0.245750` | `0.073893` | `0.564912` | `0.000000` | `1.325820` |
| stress budget `240` | `FAIL_CAPABILITY_GAIN` | `0.050925` | `-0.218786` | `0.551818` | `0.334506` | `1.083030` |
| mid budget `480` | `PASS_CAPABILITY_CLOSURE_GATE` | `0.340043` | `0.098705` | `0.788242` | `0.004854` | `1.529262` |
| loose budget `1440` | `FAIL_CAPABILITY_GAIN` | `0.220797` | `0.066442` | `0.507455` | `0.000000` | `1.286670` |
| `32` concepts, budget `960` | `FAIL_CAPABILITY_GAIN` | `0.226879` | `0.069770` | `0.518652` | `0.000000` | `1.298472` |
| seed `1001`, budget `480` | `PASS_CAPABILITY_CLOSURE_GATE` | `0.331320` | `0.087891` | `0.783403` | `0.012069` | `1.512311` |
| seed `1002`, budget `480` | `PASS_CAPABILITY_CLOSURE_GATE` | `0.333996` | `0.089410` | `0.788229` | `0.012591` | `1.518670` |
| seed `1003`, budget `480` | `PASS_CAPABILITY_CLOSURE_GATE` | `0.336300` | `0.091435` | `0.791050` | `0.012591` | `1.523999` |
| seed `1004`, budget `480` | `PASS_CAPABILITY_CLOSURE_GATE` | `0.336948` | `0.091942` | `0.791961` | `0.012591` | `1.525504` |

Interpretation:

- Positive scoped verifier gate for capability closure plus minimal causal
  intervention ledgers.
- The signal is budget-windowed: too little budget harms normal-rule closure,
  while looser budgets let closure-only controls catch up enough to miss strict
  total-gain thresholds.
- No production v12 path, GPU speedup, broad corpus claim, or broad model-quality
  baseline change is claimed.

---

## R422 Executable Micro-Theory Gate (2026-05-12)

Source: `R422`

Artifacts:

- `.workspace/research/r422-extraordinary-paradigm-horizons/executable-micro-theory-implementation.md`
- `scripts/research/r422_executable_micro_theory_gate.py`
- `.workspace/research/r422-extraordinary-paradigm-horizons/executable-micro-theory-main.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/executable-micro-theory-reuse100.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/executable-micro-theory-reuse200-long.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/executable-micro-theory-budget320.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/executable-micro-theory-budget720.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/executable-micro-theory-seed1001.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/executable-micro-theory-seed1002.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/executable-micro-theory-seed1003.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/executable-micro-theory-seed1004.json`

Measurement:

| run | decision | total gain | long gain | correct/paid ratio | state checks |
|---|---|---:|---:|---:|---:|
| main | `PASS_EXECUTABLE_MICRO_THEORY_GATE` | `0.138802` | `0.160821` | `1.946677` | `1.000000` |
| reuse `100` | `PASS_EXECUTABLE_MICRO_THEORY_GATE` | `0.163568` | `0.200326` | `2.204753` | `1.000000` |
| reuse `200` long | `PASS_EXECUTABLE_MICRO_THEORY_GATE` | `0.187760` | `0.236332` | `2.357733` | `1.000000` |
| budget `320` | `FAIL_TOTAL_CAPABILITY_GAIN` | `-0.049089` | `-0.013424` | `1.753905` | `1.000000` |
| budget `720` | `PASS_EXECUTABLE_MICRO_THEORY_GATE` | `0.163824` | `0.199583` | `2.336864` | `1.000000` |
| seed `1001`, reuse `100` | `PASS_EXECUTABLE_MICRO_THEORY_GATE` | `0.165964` | `0.199542` | `2.210643` | `1.000000` |
| seed `1002`, reuse `100` | `PASS_EXECUTABLE_MICRO_THEORY_GATE` | `0.168021` | `0.201553` | `2.217738` | `1.000000` |
| seed `1003`, reuse `100` | `PASS_EXECUTABLE_MICRO_THEORY_GATE` | `0.170833` | `0.204253` | `2.228479` | `1.000000` |
| seed `1004`, reuse `100` | `PASS_EXECUTABLE_MICRO_THEORY_GATE` | `0.169531` | `0.202630` | `2.226108` | `1.000000` |

Interpretation:

- Positive scoped CPU verifier gate for executable local-theory objects.
- The effect is amortization- and coverage-dependent: the under-budget `320`
  control fails because not enough domains receive a micro-theory object.
- No production v12 path, GPU speedup, broad corpus claim, or broad model-quality
  baseline change is claimed.

---

## R422 Event-Causal Credit Ledger Closeout (2026-05-13)

Source: `R422`

Artifacts:

- `.workspace/research/r422-extraordinary-paradigm-horizons/event-causal-credit-ledger-implementation.md`
- `.workspace/research/r422-extraordinary-paradigm-horizons/event-causal-credit-events-testdata.out`
- `memory/baselines.md` prior R416/R417/R388 gate sections

Measurement:

| evidence object | result | interpretation |
|---|---|---|
| current GPU event smoke | `CE_NORMAL_CHAMBER_GATE` active on `test_data.bin`, `groups=14`, `elements=35454976`, `saturated=0`, `candidates=0`, `priced_k16=0`, `identifying=0`, optimizer integrity `pass=1`, loss `40.7839`, VRAM `740MB` | early from-scratch chamber exposes no material event mass; non-identifying for event-causal credit |
| R416 boundary-cotangent credit | `wiki_1mb.bin` three-step final `residual_fraction=1.150172`, `priced_ratio=2.157741`, `pass=0` | boundary-anchor-linear event/transition-credit realization fails |
| R416 variance-priced credit | `wiki_1mb.bin` three-step final default-paid `descent_per_paid_work=0.434304`, `pass=0`; zero score/verify sensitivity `3.948010`, `pass=1` | aligned credit-item signal is pricing-blocked by score and verify cost |
| R417 evidence-rate commitment | direct TSA and `gbar` modes increase loss severely after live-SSM repair | direct event commitment is harmful under current realization |
| R388 flip-refresh | forced early gate `eta=0.029575`, `projected_refresh_gain=1.030476`, `pass=0`; normal-refresh attempt invalid due process overlap | no valid mature flip-event evidence yet |

Interpretation:

- `R422-CA-02 Event-Causal Credit Ledger` is a training-dynamics finding, so CPU
  evidence cannot close it.
- Current v12/GPU event-adjacent evidence does not support promotion or a speed
  claim.
- The result is parked, not globally rejected: future mature/post-warmup event
  traces or cheap in-step score/verify witnesses can still reopen it.
- No production behavior, baseline speed, VRAM/RAM target, or paper-ready
  positive claim changes.

---

## R422 Target-Preserving Entropy-Residual Curriculum Gate (2026-05-13)

Source: `R422`

Artifacts:

- `scripts/research/r422_entropy_residual_curriculum_gate.py`
- `.workspace/research/r422-extraordinary-paradigm-horizons/entropy-residual-curriculum-implementation.md`
- `.workspace/research/r422-extraordinary-paradigm-horizons/entropy-residual-curriculum-*.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/entropy-residual-grid/*.json`

Measurement:

| shard / sweep | result | interpretation |
|---|---|---|
| `test_data.bin`, order `2`, paid `0.35` | `PASS`, best support-priced entropy residual, equivalent-uniform `2.6442x`, CE gain vs uniform `0.031219`, debt error `0` | positive control only; small fixture is memorization-sensitive |
| `text_data.bin`, order `2`, paid `0.35` | `FAIL`, best entropy residual, equivalent-uniform `0.9798x`, CE gain vs uniform `-0.005552`, debt error `0` | current residual band does not beat uniform on this real text shard |
| `wiki_1mb.bin`, order `2`, paid `0.35` | `FAIL`, best entropy residual, equivalent-uniform `0.9836x`, CE gain vs uniform `-0.005472`, debt error `0` | order-2 residual bands are not enough |
| `wiki_1mb.bin`, order `3`, paid `0.20-0.50` | `PASS`, support-priced entropy residual, equivalent-uniform `1.2410x-1.5247x`, CE gain vs uniform `0.073251-0.083422` | material order-3 residual-scheduler signal |
| `wiki_5mb.bin` offset sweep, order `3` | pass at offsets `0`, `240000`, and `600000`; partial/near-miss at `360000`; fail at `120000` and `480000` | signal is real but offset-sensitive |

Interpretation:

- `R422-TO-04 Target-Preserving Curriculum By Entropy Residual` survives as a
  partial positive training-material/curriculum signal.
- The useful policy is support-priced entropy residual at context order `3`, not
  plain entropy residual and not a generic loss-only curriculum.
- Every retained run preserves the explicit CE debt identity:
  `paid_events + deferred_events = original_pool_events`.
- No v12/GPU sampler, training behavior change, wall-clock speedup, VRAM/RAM
  change, or broad baseline change is claimed.
- Promotion requires an opt-in v12/GPU sampler or data-stream scheduler that
  preserves deferred CE debt and beats uniform/random/easy/loss-only controls
  after sampler overhead, variance, warmup behavior, and optimizer dynamics are
  priced.

---

## R422 Frontier-Only Capability Stream Gate (2026-05-13)

Source: `R422`

Artifacts:

- `scripts/research/r422_frontier_capability_stream_gate.py`
- `.workspace/research/r422-extraordinary-paradigm-horizons/frontier-capability-stream-implementation.md`
- `.workspace/research/r422-extraordinary-paradigm-horizons/frontier-capability-stream-budget*.json`
- `.workspace/research/r422-extraordinary-paradigm-horizons/frontier-capability-grid/*.json`

Measurement:

| budget | decision | best control | score gain vs best | score gain vs uniform | frontier fraction | paid ratio |
|---:|---|---|---:|---:|---:|---:|
| `360` | `FAIL_LIVE_FRONTIER_BAND_DENSITY` | `hard_stream` | `0.023904` | `0.298952` | `0.328686` | `1.025421` |
| `480` | `FAIL_LIVE_FRONTIER_BAND_DENSITY` | `hard_stream` | `-0.002404` | `0.195200` | `0.238521` | `0.997793` |
| `720` | `FAIL_LIVE_FRONTIER_BAND_DENSITY` | `hard_stream` | `-0.002772` | `0.082862` | `0.153981` | `0.997154` |

Interpretation:

- `R422-CI-01 Frontier-Only Capability Stream` is closed for the current
  realization as `closed_current_frontier_stream_collapses_to_hard_curriculum`.
- The live verifier-band stream consistently beats uniform, so live difficulty
  selection remains useful.
- The standalone frontier-only claim fails because `hard_stream` is always the
  best control and true frontier-band density remains below threshold across the
  budget/band grid.
- No v12/GPU training behavior, measured speedup, VRAM/RAM change, or broad
  model baseline changes.
- Reopen only with a sharper frontier definition that beats hard, uniform, easy,
  and loss-only controls while preserving residual original-CE or normal
  capability guards.

---

## R422 Spatial Canvas Internal Code Gate (2026-05-13)

Source: `R422`

Artifacts:

- `scripts/research/r422_spatial_canvas_gate.py`
- `.workspace/research/r422-extraordinary-paradigm-horizons/spatial-canvas-implementation.md`
- `.workspace/research/r422-extraordinary-paradigm-horizons/spatial-canvas-*.json`

Measurement:

| run | chain range | distractors | decision | depth gain | work ratio | answer accuracy | reconstruction |
|---|---:|---:|---|---:|---:|---:|---:|
| main | `8-96` | `24` | `PASS_SPATIAL_CANVAS_GATE` | `62.343181x` | `0.064033` | `1.000000` | `1.000000` |
| short | `4-32` | `12` | `PASS_SPATIAL_CANVAS_GATE` | `30.248633x` | `0.159813` | `1.000000` | `1.000000` |
| long | `64-192` | `48` | `PASS_SPATIAL_CANVAS_GATE` | `109.215043x` | `0.026056` | `1.000000` | `1.000000` |

Interpretation:

- `R422-MS-03 Spatial Canvas Internal Code` passes as a synthetic structural
  substrate gate.
- The positive object is exact canonical text-canvas-text encoding plus local
  pointer-jumping propagation, not a decorative visual embedding.
- No real-text compiler, production v12 path, GPU speedup, VRAM/RAM change, or
  broad model-quality baseline change is claimed.
- Promotion requires residual-certified real-text parsing, model-native canvas
  integration, controls against direct retrieval/dictionary solutions, and full
  pricing of parser, residual text, canvas memory, update rounds, and readout.

---

## R422 Capability Equivalence-Class Objective Gate (2026-05-13)

Source: `R422`

Artifacts:

- `scripts/research/r422_capability_equivalence_gate.py`
- `.workspace/research/r422-extraordinary-paradigm-horizons/capability-equivalence-implementation.md`
- `.workspace/research/r422-extraordinary-paradigm-horizons/capability-equivalence-*.json`

Measurement:

| run | surfaces/class | noise | decision | class CE | flat surface CE | class gain vs flat | random gain | residual guard |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| main | `24` | `0.20` | `PASS_CAPABILITY_EQUIVALENCE_GATE` | `0.361304` | `7.307117` | `6.945813` | `3.943808` | `0.000000` |
| surface stress | `8` | `0.20` | `PASS_CAPABILITY_EQUIVALENCE_GATE` | `0.357866` | `6.378997` | `6.021131` | `3.900639` | `0.000000` |
| noisy control | `24` | `0.40` | `PASS_CAPABILITY_EQUIVALENCE_GATE` | `0.361304` | `7.307117` | `6.945813` | `3.918371` | `0.000000` |

Interpretation:

- `R422-TO-02 Capability Equivalence-Class Objective` passes as a controlled
  material/proof gate.
- Exact factorization identity error is numerical zero (`8.88e-16` in retained
  runs), and residual surface CE does not degrade relative to flat surface CE.
- No real extractor, v12/GPU training path, measured speedup, VRAM/RAM change,
  or broad model-quality baseline change is claimed.
- Promotion requires a real deterministic extractor for proof/program/answer or
  action classes plus residual surface settlement and controls against noisy,
  random, and shortcut class definitions.

---

## R422 Reversible Corpus Language Gate (2026-05-13)

Source: `R422`

Artifacts:

- `scripts/research/r422_reversible_corpus_language_gate.py`
- `.workspace/research/r422-extraordinary-paradigm-horizons/reversible-corpus-language-implementation.md`
- `.workspace/research/r422-extraordinary-paradigm-horizons/reversible-corpus-*.json`

Measurement:

| run | facts | decision | object token ratio | query work ratio | shuffled control | reconstruction |
|---|---:|---|---:|---:|---:|---:|
| main | `12-72` | `PASS_REVERSIBLE_CORPUS_LANGUAGE_GATE` | `0.581728` | `0.114326` | `0.012500` | `1.000000` |
| small | `4-16` | `FAIL_SHUFFLED_GRAPH_CONTROL` | `0.610384` | `0.114248` | `0.075000` | `1.000000` |
| large | `64-160` | `PASS_REVERSIBLE_CORPUS_LANGUAGE_GATE` | `0.575366` | `0.114387` | `0.012500` | `1.000000` |

Interpretation:

- `R422-TM-01 Reversible Corpus Language` passes as a controlled material gate
  for medium/large dense records.
- The small-record control fails because shuffled graph answers hit too often by
  chance, so the mechanism should not be promoted for sparse records.
- No real parser, v12/GPU training path, measured speedup, VRAM/RAM change, or
  broad model-quality baseline change is claimed.
- Promotion requires non-canonical real-text parsing, exact reconstruction or
  explicit residual source CE, inverse/consistency tasks versus same-token plain
  corpus, and controls against direct lookup, compression-only, shuffled graph,
  and hidden residual storage.

## R422 Ternary Cellular Automaton Language Core Gate (2026-05-13)

Source: `R422`

Artifacts:

- `scripts/research/r422_ternary_ca_language_gate.py`
- `.workspace/research/r422-extraordinary-paradigm-horizons/ternary-ca-language-implementation.md`
- `.workspace/research/r422-extraordinary-paradigm-horizons/ternary-ca-*.json`

Measurement:

| run | background | decision | log accuracy | linear accuracy | log compute ratio | linear compute ratio |
|---|---|---|---:|---:|---:|---:|
| main `16-256` | `zero` | `FAIL_FINITE_CONE_REQUIRES_LINEAR_STEPS` | `0.000000` | `1.000000` | `7.364583` | `139.343750` |
| short `8-64` | `zero` | `FAIL_FINITE_CONE_REQUIRES_LINEAR_STEPS` | `0.000000` | `1.000000` | `5.437500` | `35.895833` |
| long `128-512` | `zero` | `FAIL_FINITE_CONE_REQUIRES_LINEAR_STEPS` | `0.000000` | `1.000000` | `8.770833` | `336.875000` |
| random control `16-256` | `random` | `FAIL_ACTIVE_COMPUTE_PRICING` | `0.552083` | `0.552083` | `7.177083` | `127.291667` |

Interpretation:

- `R422-MS-04 Ternary Cellular Automaton Language Core` is closed for the
  current radius-1 local CA realization.
- Clean dependency transport proves the finite-cone blocker: logarithmic update
  schedules cannot carry the nonlocal dependency, while linear schedules solve
  it only by paying length-proportional active compute.
- The random-background control also fails without a collision/decoding
  contract.
- No production v12 path, GPU speedup, VRAM/RAM change, or broad model-quality
  baseline change is claimed.
- Reopen only with hierarchy, nonlocal shortcuts, multiscale CA, active
  compression, or pointer-jumping-like routes plus full pricing.

## R422 Belief-Ledger Answerer Gate (2026-05-13)

Source: `R422`

Artifacts:

- `scripts/research/r422_belief_ledger_answerer_gate.py`
- `.workspace/research/r422-extraordinary-paradigm-horizons/belief-ledger-answerer-implementation.md`
- `.workspace/research/r422-extraordinary-paradigm-horizons/belief-ledger-*.json`

Measurement:

| run | conflict | decision | mode score | wrong rate | pass rate | ledger / best QPP | ledger tokens | scan tokens |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| main | `0.35` | `PASS_BELIEF_LEDGER_ANSWERER_GATE` | `0.940544` | `0.027500` | `0.643750` | `1.250376` | `6.712500` | `31.143333` |
| high conflict | `0.55` | `FAIL_CONFIDENT_WRONG_RATE` | `0.934132` | `0.031667` | `0.610000` | `1.204037` | `6.780000` | `32.353333` |
| low repeat | `0.35` | `FAIL_CONFIDENT_WRONG_RATE` | `0.893750` | `0.047222` | `0.697222` | `0.295021` | `6.605556` | `30.933333` |
| low conflict | `0.10` | `PASS_BELIEF_LEDGER_ANSWERER_GATE` | `0.941272` | `0.022083` | `0.773750` | `1.317231` | `6.452500` | `28.800000` |

Interpretation:

- `R422-MS-06 Belief-Ledger Answerer` survives as a partial positive
  material/compiler candidate for repeated moderate-conflict QA.
- The current realization fails under heavier contradiction and low query reuse.
- No production v12 path, GPU speedup, VRAM/RAM change, or broad model-quality
  baseline change is claimed.
- Promotion requires a real corpus/task compiler, residual render CE, answer/defer
  calibration under held-out paraphrase/conflict variants, and controls against
  recency, majority, always-defer, and hidden retrieval.

## R422 Dual Artifact Gate (2026-05-13)

Source: `R422`

Artifacts:

- `scripts/research/r422_dual_artifact_gate.py`
- `.workspace/research/r422-extraordinary-paradigm-horizons/dual-artifact-implementation.md`
- `.workspace/research/r422-extraordinary-paradigm-horizons/dual-artifact-*.json`

Measurement:

| run | ops | styles | modulus | decision | factorized accuracy | sparse monolithic accuracy | train token ratio | compact interface ratio |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| main | `64` | `32` | `16` | `PASS_DUAL_ARTIFACT_GATE` | `1.000000` | `0.000000` | `0.102844` | `0.111937` |
| low style | `64` | `4` | `16` | `PASS_DUAL_ARTIFACT_GATE` | `1.000000` | `0.000000` | `0.522196` | `0.118830` |
| no style | `64` | `1` | `16` | `FAIL_TRAINING_TOKEN_PRICING` | `1.000000` | `0.000000` | `2.446875` | `0.150000` |
| wide answer | `64` | `32` | `256` | `PASS_DUAL_ARTIFACT_GATE` | `1.000000` | `0.000000` | `0.102304` | `0.184764` |

Interpretation:

- `R422-MS-07 Dual Artifact: Fluent Shell + Competence Core` passes as a
  controlled structural/material gate when reusable fluency surface amortizes
  the shell.
- The no-style control fails, so the split is not universally cheaper.
- The leaky-interface control is rejected by interface pricing.
- No production v12 path, GPU speedup, VRAM/RAM change, or broad model-quality
  baseline change is claimed.

## R422 Local Semantic Atlas Gate (2026-05-13)

Source: `R422`

Artifacts:

- `scripts/research/r422_local_semantic_atlas_gate.py`
- `.workspace/research/r422-extraordinary-paradigm-horizons/local-semantic-atlas-implementation.md`
- `.workspace/research/r422-extraordinary-paradigm-horizons/local-semantic-atlas-*.json`

Measurement:

| run | domains | dim | decision | atlas accuracy | monolithic | topic | retrieval | priced ratio | atlas error |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| main | `24` | `16` | `PASS_LOCAL_SEMANTIC_ATLAS_GATE` | `1.000000` | `0.000000` | `0.000000` | `0.000000` | `0.106771` | `0.000000` |
| dim32 | `24` | `32` | `PASS_LOCAL_SEMANTIC_ATLAS_GATE` | `1.000000` | `0.000000` | `0.000000` | `0.000000` | `0.084635` | `0.000000` |
| small | `8` | `16` | `PASS_LOCAL_SEMANTIC_ATLAS_GATE` | `1.000000` | `0.000000` | `0.000000` | `0.000000` | `0.171875` | `0.000000` |
| aligned control | `24` | `16` | `FAIL_TOPIC_OR_MONOLITHIC_CONTROL` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.106771` | `0.000000` |
| noise `0.03` | `24` | `16` | `PASS_LOCAL_SEMANTIC_ATLAS_GATE` | `1.000000` | `0.000000` | `0.000000` | `0.000000` | `0.106771` | `0.019852` |
| noise `0.12` | `24` | `16` | `FAIL_ATLAS_TRANSFER_ACCURACY` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.106771` | `0.139647` |

Interpretation:

- `R422-SM-02 Local Semantic Atlas` passes as an exact chart-transfer structural
  gate when local maps are accurate.
- It is not a topic/MoE result: topic labels, raw monolithic averaging, and
  retrieval-like local deltas fail under incompatible charts.
- Aligned charts and high atlas noise bound the claim.
- No production v12 path, GPU speedup, VRAM/RAM change, or broad model-quality
  baseline change is claimed.

## R422 Proof-Carrying Memory Cells Gate (2026-05-13)

Source: `R422`

Artifacts:

- `scripts/research/r422_proof_memory_cells_gate.py`
- `.workspace/research/r422-extraordinary-paradigm-horizons/proof-memory-cells-implementation.md`
- `.workspace/research/r422-extraordinary-paradigm-horizons/proof-memory-*.json`

Measurement:

| run | valid | conflict | decision | mode score | false recall | proof CE | best control CE | CE gain |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| main | `0.55` | `0.30` | `PASS_PROOF_MEMORY_CELLS_GATE` | `1.000000` | `0.000000` | `0.045551` | `0.849277` | `0.803726` |
| high conflict | `0.40` | `0.45` | `PASS_PROOF_MEMORY_CELLS_GATE` | `1.000000` | `0.000000` | `0.047085` | `0.780234` | `0.733148` |
| mostly valid | `0.80` | `0.10` | `PASS_PROOF_MEMORY_CELLS_GATE` | `1.000000` | `0.000000` | `0.042987` | `0.509811` | `0.466824` |
| small cells | `0.55` | `0.30` | `PASS_PROOF_MEMORY_CELLS_GATE` | `1.000000` | `0.000000` | `0.045638` | `0.860785` | `0.815146` |

Interpretation:

- `R422-SM-03 Proof-Carrying Memory Cells` passes as a controlled local
  activation-certificate gate.
- The result extends reversible state storage with a distinct proof-gated
  activation condition.
- No real memory-cell compiler, production v12 path, GPU speedup, VRAM/RAM
  change, or broad model-quality baseline change is claimed.

