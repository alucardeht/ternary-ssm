# Measured Baselines — All Numbers

**Last updated:** 2026-05-21

---

## R428 Monitored Longer Validation - R427 Growth Plus Speed-To-Quality Stack (2026-05-15)

Source: `R428`

Artifacts:

- `.workspace/training/rounds/r428.ps1 -Mode integrated -Profile exclusive`
- `.work/jobs/overnight/r428-speed-validation-20260515-043529/baseline_full.status.json`
- `.work/jobs/overnight/r428-direct-s300-20260515-064218/launch.json`
- `.work/jobs/training-round-presets/r427/integrated-20260515-064220/r427_integrated_training_analysis.json`
- `.work/jobs/training-round-presets/r428/integrated-20260515-064219/r428_speed_to_quality_analysis.json`
- failed staged-400 source log: `.work/jobs/training-round-presets/r427/integrated-20260515-061538/r427_staged_source.log`

Measurement:

| object | result |
|---|---|
| baseline arm | `wiki_5mb.bin`, 700 steps, batch `1`, seed `42780`, teaching off |
| baseline elapsed | `5268.165s` / `87.80 min` |
| baseline telemetry | max `63 C`, `2512 MiB`, `112.26 W`, max util `100%` |
| baseline best eval | step `500`, eval loss `5.6195` |
| baseline final eval | step `700`, eval loss `96.5459`; late instability after step `500` |
| R428 successful arm | staged `300` smaller-width steps plus lifted full-width continuation for `700` additional steps |
| R428 staged source | completed; eval losses `20.3894`, `19.3960`, `19.5194`; `~1530-1546 tok/s`, `357 MB` |
| R428 checkpoint lift | completed; source `512/64/64`, target `1024/128/128`, `copied_entries=211`, `validated_flags=18`, `status=ok` |
| R428 continuation | completed from resumed step `300`; first row loss `39.3961`, last row loss `52068.6914`; `~509-633 tok/s`, `740 MB` |
| R428 best eval after lift | step `500`, eval loss `11.2388` |
| R428 final eval | step `1000`, eval loss `94831.8906`; strong late divergence |
| R428 operational decision | `PASS_R428_SPEED_TO_QUALITY_READY` from controller, but only as executable readiness |
| R427 integrated decision | `PASS_R427_INTEGRATED_TRAINING_PRESET` from analyzer |
| live positive internal signals | readout quotient net proxy `3.000000`, interaction pass `1`, genotype pass `1`, variance pass `1` |
| non-promoted / failed internal signals | boundary pass `0`, original-CE local/surface pass `0/0`, receipt pass `0` |
| staged-400 attempt | failed near source step `380` with assertion `model->layers[k].moments.n > 0` in `src/core/loss.c:174` |
| monitor issue | external overnight monitor false-stalled R428 because wrapper stdout did not advance while nested R427 logs were active |

Interpretation:

- This is the first longer real execution of the R428 control layer over the
  R427 no-reset growth ladder.
- The central capability target is confirmed operationally: a smaller compiled
  v12 model can train, save, lift into the full-width model, and continue
  training without resetting from scratch.
- The current R428 recipe does not improve matched-quality training time. It is
  worse than the baseline at the best shared checkpoint region: baseline best
  eval loss `5.6195` at step `500`, versus R428 best post-lift eval loss
  `11.2388` at step `500`.
- Both arms expose late instability at the current learning-rate ceiling, but
  R428 diverges much harder after global step `800`, ending at eval loss
  `94831.8906`.
- Do not multiply R428 isolated projections into a new official training-day
  projection. The isolated gates and internal proxies remain useful, but the
  composed training recipe must be repaired before it can claim a real
  production speedup.
- The next speed-critical target is not another broad projection pass; it is a
  stabilized growth schedule: lower or stage-local LR after lift, catch-up-debt
  stopping, better staged/full split, and a monitor that follows nested R427
  logs instead of wrapper stdout only.

Status: `MIXED_REAL_LONG_VALIDATION_GROWTH_CAPABILITY_CONFIRMED_SPEED_NOT_CONFIRMED`.

---

## R428 Pre-Next-Round Hardening - Monitor, Staged-400, Post-Lift LR (2026-05-15)

Source: `R428`

Artifacts:

- `.workspace/training/overnight/r428_speed_validation_monitor.ps1`
- `.workspace/training/rounds/r427.ps1`
- `.workspace/training/rounds/r428.ps1`
- `scripts/research/r427_integrated_training_analyzer.py`
- `src/model_v12/src/core/loss.c`
- `src/model_v12/src/core/loss.h`
- `src/model_v12/src/train.c`
- `.work/jobs/overnight/r428-speed-validation-20260515-113714/r428_integrated.status.json`
- `.work/jobs/training-round-presets/r427/integrated-20260515-114054/r427_integrated_training_analysis.json`

Measurement:

| object | result |
|---|---|
| build | `mingw32-make USE_VULKAN=1 train` passed before staged-400 validation |
| monitor smoke | `.workspace/training/overnight/r428_speed_validation_monitor.ps1 -SmokeOnly ...` completed in `74.041s`, max `52 C`, max `2396 MiB`, max util `100%`, no lingering `train.exe` |
| nested heartbeat | monitor now sums wrapper log plus recent nested R427/R428 `.log` / `.json` writes, preventing false stall while nested jobs progress |
| LR override activation | nested continuation emitted `[LR_BASE_OVERRIDE] active=1 lr_base=0.000119999997 checkpoint_resumed=1 step=1` in the smoke run |
| staged-400 diagnostic | `.workspace/training/rounds/r427.ps1 -Mode integrated -StagedSteps 400 -FullSteps 0 ... -ContinuationLrBase 0.00012` reached step `400`, saved staged checkpoint, and emitted `[TRAIN] Complete` |
| staged-400 evals | eval loss `20.3895` at step `100`, `19.3945` at step `200`, `19.5192` at step `300`, `18.6385` at step `400` |
| staged-400 throughput / VRAM | about `1497-1552 tok/s`, `357 MB` |
| analyzer | existing run reclassified as `PASS_R427_INTEGRATED_STAGED_DIAGNOSTIC` after zero-continuation diagnostic handling |
| GP/KL crash repair | layers with no accumulated moments are skipped in GP/KL loss, so checkpoint-loaded/staged eval paths no longer abort on `moments.n == 0` |

Interpretation:

- This closes the operational blockers found by the overnight R428 validation
  before opening another research round.
- The old staged-400 failure was a path-fidelity bug in the GP/KL diagnostic
  assumption, not a falsification of staged growth.
- The staged-400 diagnostic does not validate live credit/result lines because
  `FullSteps=0`; use nonzero continuation steps for training-preset readiness
  or speed-to-quality claims.
- Lower post-lift LR is now wired into serious R428 profile presets, but it is
  not yet a measured stability repair until a longer continuation run confirms
  reduced divergence.

Status: `PRE_NEXT_ROUND_HARDENING_COMPLETE_GROWTH_PATH_READY_FOR_NEXT_LONG_VALIDATION`.

---

## R429 No-Tradeoff Stack Smoke (2026-05-15)

Source: `R429`

Artifacts:

- `.workspace/training/rounds/r429.ps1`
- `scripts/research/r429_no_tradeoff_stack_controller.py`
- `.workspace/research/r429-extraordinary-speed-memory-frontier/implementation-todo.md`
- `.workspace/research/r429-extraordinary-speed-memory-frontier/implementation-closeout.md`
- `.work/jobs/training-round-presets/r429/integrated-20260515-173430/r429_no_tradeoff_stack_analysis.json`

Measurement:

| object | result |
|---|---|
| decision | `PASS_R429_NO_TRADEOFF_STACK_SMOKE` |
| command | `.workspace/training/rounds/r429.ps1 -Mode integrated -Profile interactive -StagedSteps 1 -FullSteps 1 -Batch 1 -EvalEvery 0 -MaterialTokens 65536` |
| staged source throughput | `1532 tok/s` |
| full continuation throughput | `504 tok/s` |
| staged/full throughput ratio | `3.039683x` |
| readout net speedup proxy | `3.000000` |
| interaction-safe update | `pass=1` |
| genotype growth | `pass=1` |
| variance-priced credit | `pass=1` |
| R429 decision-equivalence witness | `pass=1`, `mismatch_updates=0`, `certified_fraction=0.303030`, `avoided_work_fraction=0.046156` |
| no-tradeoff contract | exact equivalence, exact reconstruction, certified irrelevance, or full fallback |

Projection:

| class | speedup | projected 7B time from R150 `495d` |
|---|---:|---:|
| conservative implemented stack | `1.80x-2.60x` | `275.0-190.4 days` |
| strong implemented stack | `2.80x-4.50x` | `176.8-110.0 days` |

Interpretation:

- This is an executable no-tradeoff control stack over R428/R427.
- The stack covers all five R429 aspects and now includes the native Aspect A
  decision-equivalence witness in C.
- The projection is overlap-discounted and must not be multiplied across
  aspects.
- This does not change the official `495d` baseline, does not prove a
  matched-target 7B speedup, and does not claim production inference speed.

Status: `PASS_R429_FIVE_ASPECT_NATIVE_WITNESS_STACK_SMOKE_PROJECTED_ONLY`.

---

## R430 Formal Stack Smoke (2026-05-15)

Source: `R430`

Artifacts:

- `.workspace/training/rounds/r430.ps1`
- `scripts/research/r430_formal_stack_controller.py`
- `.workspace/research/r430-sub-10-day-frontier/implementation-closeout.md`
- `.work/jobs/training-round-presets/r430/integrated-20260515-205152/r430_formal_stack_analysis.json`
- `.work/jobs/training-round-presets/r430/integrated-20260515-205152/r430_denominator_receipts.json`

Measurement:

| object | result |
|---|---|
| decision | `PASS_R430_FORMAL_STACK_READY` |
| command | `.workspace/training/rounds/r430.ps1 -Mode integrated -Profile interactive -StagedSteps 1 -FullSteps 1 -Batch 1 -EvalEvery 0 -MaterialTokens 65536` |
| nested R429 decision | `PASS_R429_NO_TRADEOFF_STACK_SMOKE` |
| nested R428 decision | `PASS_R428_SPEED_TO_QUALITY_READY` |
| denominator receipts | `11` total, `10` active |
| strict speed-countable receipts | `0` |
| strict anchor | R429 lower conservative endpoint `275.0 days` |
| required factor to `<10 days` from strict anchor | `27.5x` |

Interpretation:

- This is the executable formal R430 layer over R429/R428/R427.
- It proves the R430 preset and receipt controller are wired for a serious
  monitored training run.
- It does not change the official `495d` baseline, does not prove a measured
  matched-target 7B speedup, and does not claim production inference speed,
  VRAM reduction, RAM reduction, or a strict sub-10-day result.
- Provisional C/D planning deltas are stored as non-strict gate receipts and
  must not be reported as measured speed.

Status: `PASS_R430_FORMAL_STACK_READY`.

---

## R436 Amortized CE Martingale Verifier Gate (2026-05-20)

Source: `R436`

Artifacts:

- `.workspace/training/rounds/r436.ps1`
- `scripts/research/r436_martingale_verifier_gate.py`
- `.workspace/research/r436-real-training-speed-round/implementation-closeout.md`
- `.work/jobs/training-round-presets/r436/martingale-20260520-121114/r436_martingale_verifier_analysis.json`

Measurement:

| object | result |
|---|---|
| decision | `PASS_R436_AMORTIZED_CE_MARTINGALE_VERIFIER_GATE` |
| command | `.workspace/training/rounds/r436.ps1 -Mode martingale -Steps 3 -Batch 1 -Seed 43600 -LogEvery 1 -EvalEvery 0` |
| sampled fraction | `0.247654` |
| amortized overhead | `0.120000` |
| alignment | `0.990347` |
| variance proxy | `0.018748` |
| certified progress proxy | `0.971599` |
| residual debt | `0.028401` |
| debt ratio | `0.029231` |
| priced ratio | `0.396055` |
| projected speedup if promoted | `2.524902x` |
| train smoke | step `0` loss `40.9837`, step `2` loss `39.7002`, `740 MB` VRAM |

Interpretation:

- This is a positive opt-in GPU observer/pricing gate for the R436 martingale
  verifier.
- It re-prices the live `VARIANCE_PRICED_CREDIT_GATE` estimator under an
  amortized original-CE audit contract.
- It does not apply sampled updates to training and does not change default
  training behavior.
- It does not prove matched-target speedup, update the official 7B projection,
  or change VRAM/RAM/inference baselines.

Status: `POSITIVE_GPU_OBSERVER_GATE_NOT_SPEED_CLAIM`.

---

## R436 Real Training Stack Activation (2026-05-20)

Source: `R436`

Artifacts:

- `.workspace/training/rounds/r436.ps1`
- `.workspace/training/rounds/r427.ps1`
- `.workspace/research/r436-real-training-speed-round/implementation-closeout.md`
- `.work/jobs/training-round-presets/r436/real-20260520-123412/r436_real_training_stack_analysis.json`
- `.work/jobs/training-round-presets/r427/integrated-20260520-123412/r427_lifted_growth_continuation.log`

Measurement:

| object | result |
|---|---|
| decision | `PASS_R436_REAL_TRAINING_STACK_ACTIVATION` |
| command | `.workspace/training/rounds/r436.ps1 -Mode real -StagedSteps 1 -FullSteps 1 -Batch 1 -Seed 42780 -LogEvery 1 -EvalEvery 0 -EvalBatches 1` |
| martingale training activation | `active=1`, `training_change=1` |
| martingale applied layers | step `2`: `training_layers=24`, `applied_layers=24`, `fallback_layers=0` |
| variance-priced support | `sampled_fraction=0.245076`, `priced_work=0.365077`, `descent_per_paid_work=2.660161`, `pass=1` |
| readout quotient support | `coverage=1.000000`, `priced_work_ratio=0.333333`, `net_speedup_estimate=3.000000` |
| online TSSI support | `identifying=1`, `pass=1`, `score_updates=24552` |
| segment cotangent observer | `identifying=1`, `pass=0`; support only |
| original-CE debt observer | `identifying=1`, `local_pass=0`, `surface_pass=0`; support only |
| full continuation smoke | step `0` loss `40.5444`, `480 tok/s`, `740 MB`; completed |

Interpretation:

- This is the first R436 path where the martingale-gated sampled transition
  credit changes opt-in training behavior.
- The preset composes over the R427 integrated path, so staged source, lift,
  and full-width continuation are part of the real activation smoke.
- This does not prove matched-target speedup, does not update the official 7B
  projection, and does not change VRAM/RAM/inference baselines.
- A-family recurrent cotangent moment, Green operator, and segment tube remain
  observer/support-only in this preset; no dense recurrent backward work is
  removed yet.

Status: `PASS_R436_REAL_TRAINING_STACK_ACTIVATION_NOT_SPEED_CLAIM`.

---

## R436 Direct Martingale Speed Sanity (2026-05-20)

Source: `R436`

Artifacts:

- `.work/jobs/training-round-presets/r436/speed-sanity-20260520-125229/speed_sanity_summary.json`
- `.work/jobs/training-round-presets/r436/speed-sanity-20260520-125229/baseline.log`
- `.work/jobs/training-round-presets/r436/speed-sanity-20260520-125229/candidate_martingale.log`

Measurement:

| object | baseline | candidate |
|---|---:|---:|
| command shape | `wiki_1mb.bin 5`, batch `1`, seed `43621`, teaching off, eval off | same plus martingale/variance flags |
| elapsed | `42.166s` | `43.540s` |
| avg throughput | `508.4 tok/s` | `490.8 tok/s` |
| final loss | `39.6595` | `39.6595` |
| final VRAM | `740 MB` | `740 MB` |
| martingale applied | no | yes |

Ratios:

- candidate elapsed / baseline elapsed: `1.032585`
- candidate avg tok/s / baseline avg tok/s: `0.965382`
- candidate final loss minus baseline final loss: `0.000000`

Interpretation:

- This is an immediate same-fixture speed sanity for the applied C1 bridge.
- The current applied martingale path is activation-positive but throughput
  negative on this short fixture: about `3.3%` slower by elapsed and about
  `3.5%` lower by logged throughput.
- It is therefore not ready to be treated as speed-improving until the bridge
  overhead is removed or amortized and this sanity is rerun.
- No official 7B projection, matched-target speedup, VRAM/RAM baseline, or
  inference baseline changes.

Status: `R436_DIRECT_MARTINGALE_SPEED_SANITY_NEGATIVE_OVERHEAD`.

---

## R436 Martingale Bridge Row-Skip Overhead Repair (2026-05-20)

Source: `R436`

Artifacts:

- `.work/jobs/training-round-presets/r436/speed-abba-row-skip-20260520-133352/speed_abba_summary.json`
- `.work/jobs/training-round-presets/r436/speed-abba-row-skip-20260520-133352/baseline_a.log`
- `.work/jobs/training-round-presets/r436/speed-abba-row-skip-20260520-133352/candidate_a.log`
- `.work/jobs/training-round-presets/r436/speed-abba-row-skip-20260520-133352/candidate_b.log`
- `.work/jobs/training-round-presets/r436/speed-abba-row-skip-20260520-133352/baseline_b.log`

Measurement:

| object | baseline avg | candidate avg |
|---|---:|---:|
| command shape | `wiki_1mb.bin 5`, batch `1`, seed `43631`, teaching off, eval off | same plus martingale/variance flags |
| elapsed | `41.976s` | `42.366s` |
| throughput | `510.5 tok/s` | `504.5 tok/s` |
| final loss | `40.4880` | `40.4880` |
| final VRAM | `740 MB` | `740 MB` |
| martingale applied | no | yes |

Ratios:

- candidate elapsed / baseline elapsed: `1.009291`
- candidate throughput / baseline throughput: `0.988247`
- baseline elapsed range: `0.069s`
- candidate elapsed range: `0.119s`

Interpretation:

- The row-skip repair prevents the CPU martingale fallback from paying dense
  `d_state^2` `A/bA/WA` work for rows whose sampled transition credit was
  zeroed by the martingale gate.
- This reduces the immediate bridge penalty from the previous `1.032585x`
  elapsed ratio to `1.009291x`, but it still does not produce a speedup.
- Runtime activation was real: candidate logs emitted
  `[AMORTIZED_CE_MARTINGALE_TRAINING] active=1`, all five steps had
  `fallback_layers=0`, and variance-priced credit remained identifying/pass.
- The remaining blocker is placement and work removal, not activation: the
  current path still runs the full GPU SSM backward, syncs/downloads
  `gpu_dA_per_step`, then applies the martingale bridge. It also keeps the
  verifier scan and CPU/GPU bridge costs. This embedding cannot justify the
  earlier observer-only `projected_speedup_if_promoted` as measured training
  speed.
- No official 7B projection, matched-target speedup, VRAM/RAM baseline, or
  inference baseline changes.

Status: `R436_ROW_SKIP_OVERHEAD_REPAIR_PARTIAL_STILL_SPEED_NEGATIVE`.

---

## R436 Heldout Speed Validation Workflow Repair (2026-05-20)

Source: `R436`

Artifacts:

- `src/model_v12/src/train.c`
- `scripts/research/r436_speed_validation_workflow.py`
- `.workspace/training/rounds/r436.ps1`
- `.work/jobs/training-round-presets/r436/speed-validate-20260520-143145/r436_speed_validation_analysis.json`

Measurement:

| object | baseline avg | candidate avg |
|---|---:|---:|
| command shape | split train prefix / heldout suffix, `1` step, batch `1`, seed `43690`, heldout eval batches `1` | same plus martingale/variance flags |
| initial heldout loss | `42.234341` | `42.234341` |
| final heldout loss | `40.886475` | `40.886478` |
| heldout progress | `1.347866` | `1.347863` |
| elapsed | `11.740164s` | `11.780199s` |
| candidate activation | no | yes |

Ratios:

- candidate elapsed / baseline elapsed: `1.003410`
- candidate progress-rate / baseline progress-rate: `0.996556`
- speed gain fraction: `-0.003410`
- progress-rate gain fraction: `-0.003444`
- configured noise floor: `0.030000`

Interpretation:

- The validation instrument was repaired to measure non-overlapping heldout CE
  progress, not train loss or same-fixture memorization.
- The trainer now supports inline heldout evaluation through
  `--heldout-dataset`, `--heldout-eval-batches`, and
  `--heldout-eval-seed`, avoiding the fragile final checkpoint save/reload path.
- This tiny smoke was heldout-identifying, candidate activation was real, and
  heldout quality did not regress.
- The measured candidate delta is inside the configured noise floor and is
  slightly slower. It is therefore a valid no-decision speed smoke, not a
  speedup.
- No official 7B projection, matched-target speedup, VRAM/RAM baseline, or
  inference baseline changes.

Status: `NO_DECISION_R436_SPEED_VALIDATION_WITHIN_NOISE_MARGIN`.

---

## R437 ASQI Synthetic Trace Gate (2026-05-20)

Source: `R437`

Artifacts:

- `scripts/research/r437_asqi_trace_gate.py`
- `.workspace/training/rounds/r437.ps1`
- `.workspace/research/r437-mathematical-training-speed-transform/implementation-closeout.md`
- `.work/jobs/training-round-presets/r437/diagnostic-20260520-171812/r437_asqi_trace_gate_analysis.json`
- `.work/jobs/training-round-presets/r437/manual-asqi-10seed/r437_asqi_trace_gate_analysis.json`

Measurement:

| object | official preset | repeated stress |
|---|---:|---:|
| command shape | `.workspace/training/rounds/r437.ps1 -Mode diagnostic -Seed 43700 -SeedCount 5` | `python scripts/research/r437_asqi_trace_gate.py --seed 43720 --seed-count 10` |
| decision | `PASS_R437_ASQI_TRACE_GATE` | `PASS_R437_ASQI_TRACE_GATE` |
| positive pass count | `5/5` | `10/10` |
| positive heldout residual mean | `0.005567` | `0.004373` |
| positive priced route mean | `0.265567` | `0.264373` |
| positive projected speedup proxy mean | `3.765827x` | `3.782667x` |
| pricing-fail pass count | `0/5` | `0/10` |
| residual-fail pass count | `0/5` | `0/10` |
| random-control pass count | `0/5` | `0/10` |

Interpretation:

- This is a CPU synthetic diagnostic gate for the R437 `Adjoint-State Quotient
  Integrator`, not a real training-speed run.
- The condition where speed is supposed to exist was explicitly exercised:
  heldout recurrent cotangents generated by forward-owned trace features.
- The gate is identifying: the positive condition passed below the promotion
  residual threshold `0.10` and priced route `<0.90`, while residual, pricing,
  and random-alignment controls all failed.
- The first attempted fixture failed the positive condition because the probe
  generated nonlinear cotangents while the gate estimator was linear. That was
  treated as a probe-validity failure and repaired before accepting the result.
- No official 7B projection, matched-target speedup, VRAM/RAM baseline,
  inference baseline, heldout CE progress result, GPU path, or default
  training behavior changes.

Status: `PASS_R437_ASQI_SYNTHETIC_TRACE_GATE_NOT_SPEED_CLAIM`.

---

## R438 A1 Lagged CE-Impulse GPU Bridge And Heldout Speed Validation (2026-05-20)

Source: `R438`

Artifacts:

- `src/model_v12/src/core/types.h`
- `src/model_v12/src/train.c`
- `src/model_v12/src/gpu/gpu_pipeline.c`
- `scripts/research/r438_speed_validation_workflow.py`
- `.workspace/training/rounds/r438.ps1`
- `.workspace/research/r438-gpu-native-training-speed-frontier/implementation-closeout.md`
- `.work/jobs/training-round-presets/r438/real-20260520-231558/r438_a1_real_activation_analysis.json`
- `.work/jobs/training-round-presets/r438/speed-validate-20260520-231628/r438_speed_validation_analysis.json`

Measurement:

| object | baseline | A1 candidate |
|---|---:|---:|
| ABBA arm count | `2` | `2` |
| elapsed mean | `13.828484550002031s` | `14.896718400006648s` |
| heldout progress mean | `1.4227180000000033` | `1.4227180000000033` |
| progress-rate mean | `0.1029099500228019` | `0.09550826613266525` |
| heldout loss mean | `40.811623` | `40.811623` |
| final VRAM | `740 MB` | `740 MB` |
| A1 active/pass/applied | `false` | `true` |

Ratios:

- candidate elapsed / baseline elapsed: `1.07724880091828`
- candidate progress-rate / baseline progress-rate: `0.9280761103421327`
- speed gain fraction: `-0.0772488009182799`
- progress-rate gain fraction: `-0.07192388965786733`

Interpretation:

- `H=4` failed activation and is not the accepted A1 preset.
- `H=8` passed real GPU activation with `transition_residual_fraction=0.046273`,
  `dA_residual_fraction=0.004227`, `dbA_residual_fraction=0.005401`,
  `dWA_residual_fraction=0.000000` over `64` sampled active columns, and
  `applied_layers=24`.
- The ABBA heldout-progress validation was identifying and candidate activation
  was real, but the current embedding was speed-negative.
- The negative speed result is explained by placement: dense recurrent backward
  is still paid before A1 audits and applies the ledger path.
- No official 7B projection, matched-target speedup, VRAM/RAM baseline,
  inference baseline, or default-path promotion changes.

Status: `R438_A1_APPLIED_REAL_GPU_PATH_SPEED_NEGATIVE_CURRENT_EMBEDDING`.

---

## R438 B1 Executable Multiplicity Packet Signature Gate (2026-05-20)

Source: `R438`

Artifacts:

- `src/model_v12/src/core/types.h`
- `src/model_v12/src/train.c`
- `src/model_v12/src/gpu/gpu_pipeline.c`
- `.workspace/training/rounds/r438_b1.ps1`
- `.workspace/research/r438-gpu-native-training-speed-frontier/b1-implementation-closeout.md`
- `.work/jobs/training-round-presets/r438-b1/sweep-20260520-233715/r438_b1_packet_analysis.json`

Measurement:

| state scale | transition scale | accepted mass | residual fraction | packet work fraction | priced ratio | pass |
|---:|---:|---:|---:|---:|---:|---|
| `16` | `64` | `0.967651` | `1.397539` | `0.066524` | `1.455875` | `0` |
| `64` | `256` | `0.967651` | `1.397539` | `0.066524` | `1.455875` | `0` |
| `1024` | `4096` | `0.873454` | `1.261412` | `0.155409` | `1.375207` | `0` |
| `4096` | `16384` | `0.190999` | `0.750237` | `0.432254` | `1.111864` | `0` |
| `16384` | `65536` | `0.004517` | `0.607980` | `0.486486` | `1.100555` | `0` |

Interpretation:

- The gate ran on the real v12 GPU path and emitted explicit activation logs.
- No tested signature cleared the required combination of accepted mass
  `>=0.20`, residual fraction `<=0.10`, and priced ratio `<0.90`.
- Coarse signatures found large multiplicity, but residual exceeded the
  original accepted gradient energy.
- Fine signatures reduced residual only to `0.607980`, while accepted mass
  collapsed to `0.004517`.
- Because the identifying gate failed, no ABBA speed validation or applied
  packet training path is justified for this signature form.
- No official 7B projection, matched-target speedup, VRAM/RAM baseline,
  inference baseline, or default-path promotion changes.

Status: `R438_B1_FAILED_REAL_GPU_SIGNATURE_GATE_CURRENT_SIGNATURE`.

---

## T002 Transformer KV Margin Sweep (2026-05-21)

Source: `T002`

Artifacts:

- `src/transformer_residency/kv_margin.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t002-latest/summary.json`
- `.workspace/research/t002-transformer-kv-margin-efficiency/extraordinary-synthesis.md`

Validation:

```powershell
python -m unittest tests.test_transformer_residency_probe
python scripts/research/t001_transformer_residency_probe.py --t002-kv-margin-sweep --out-dir .work/jobs/t001-transformer-residency-probe/t002-latest
```

Both passed. The runtime emitted `T002_KV_MARGIN_SWEEP_ACTIVE`.

Corrected denominator:

- Official Qwen3-32B `head_dim=128` is now represented in `qwen3_32b_shape`.
- FP16 KV denominator is `262144` bytes/token = `256 KiB/token`.
- FP16 KV costs: 4k `1.0 GiB`, 8k `2.0 GiB`, 16k `4.0 GiB`, 32k `8.0 GiB`.

Synthetic before/after:

| scenario | context | total GiB | slack GiB | tok/s proxy | speedup vs corrected before | status |
|---|---:|---:|---:|---:|---:|---|
| corrected current T001 + FP16 KV | 4k | `12.949` | `-0.949` | `41.84` | `1.000x` | does not fit |
| static arena + light tighten + INT8 KV | 8k | `11.557` | `0.443` | `45.42` | `1.086x` | synthetic pass |
| stronger tighten + tiered INT4 KV | 16k | `10.739` | `1.261` | `49.07` | `1.173x` | synthetic pass |
| quotient/KIVI-style INT2 KV | 32k | `10.839` | `1.161` | `49.07` | `1.173x` | synthetic pass, harder quality gate |
| exact self-spec on strong 16k stack | 16k | `10.739` | `1.261` | `66.25` | `1.583x` | synthetic pass |

Candidate inventory: `32` total, `28` strong, `23` near-no-tradeoff strong, `4`
medium.

Status: `PASS_T002_KV_MARGIN_SWEEP_SYNTHETIC_PROJECTION_GATE`. This is a
synthetic denominator and wiring gate, not a measured Qwen3 VRAM run, not a
real model tok/s benchmark, and not a default-path promotion.

---

## T002 Real CUDA KV Bandwidth Benchmark (2026-05-21)

Source: `T002`

Artifacts:

- `scripts/research/t002_kv_real_benchmark.py`
- `.work/jobs/t002-kv-real-benchmark/latest/summary.json`
- `.work/jobs/t002-kv-real-benchmark/large-buffer/summary.json`

Validation:

```powershell
python scripts/research/t002_kv_real_benchmark.py --out-dir .work/jobs/t002-kv-real-benchmark/latest --buffer-gib 1.0 --repeats 24 --warmup 4
python scripts/research/t002_kv_real_benchmark.py --out-dir .work/jobs/t002-kv-real-benchmark/large-buffer --buffer-gib 2.0 --repeats 12 --warmup 3
```

Both runs emitted `T002_REAL_CUDA_KV_BENCHMARK_ACTIVE` and passed.

Measured local bandwidth and repriced tok/s proxy:

| run | buffer | bandwidth | corrected before proxy | strong 16k proxy | tok/s-push proxy |
|---|---:|---:|---:|---:|---:|
| latest | `1.0 GiB` | `514.568 GiB/s` | `43.06 tok/s` | `50.50 tok/s` | `68.18 tok/s` |
| large-buffer | `2.0 GiB` | `512.546 GiB/s` | `42.89 tok/s` | `50.31 tok/s` | `67.91 tok/s` |

Preflight on the latest run: RTX 5070, PyTorch `2.10.0+cu128`, `10.776 GiB`
free according to `torch.cuda.mem_get_info`, `11.940 GiB` total. With a
`0.25 GiB` safety margin, the strong 16k residency plan is not safe to fully
allocate while Windows/apps are already using the GPU, even though it fits the
12 GiB denominator.

Status: `PASS_T002_REAL_CUDA_KV_BANDWIDTH_BENCHMARK`. This is a real CUDA
bandwidth/allocation-scale benchmark and supports the `~66 tok/s` proxy on this
machine. It is not a real Qwen3-32B inference benchmark, not a quality result,
and not proof that the exact self-spec verifier has been implemented.

---

## T008 Sliced Compressed Generation Smoke (2026-05-21)

Source: `T008`

Artifacts:

- `src/transformer_residency/t008_compressed_generation_smoke.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t008-compressed-generation-smoke/latest/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t008-compressed-generation-smoke/layers-0-3/summary.json`

Validation:

```powershell
python -m unittest tests.test_transformer_residency_probe
python scripts/research/t001_transformer_residency_probe.py --t008-compressed-generation-smoke --local-model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137 --out-dir .work/jobs/t001-transformer-residency-probe/t008-compressed-generation-smoke/latest
python scripts/research/t001_transformer_residency_probe.py --t008-compressed-generation-smoke --local-model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137 --generation-layer-indices 0,1,2,3 --out-dir .work/jobs/t001-transformer-residency-probe/t008-compressed-generation-smoke/layers-0-3
```

All emitted `T008_COMPRESSED_GENERATION_SMOKE_ACTIVE` and passed.

Measured smoke facts:

| slice | bits/weight | Q4 savings | avg logits rRMSE | max logits rRMSE | top-1 agreement | sliced compressed tok/s |
|---|---:|---:|---:|---:|---:|---:|
| layers `[0, 1]` | `3.2360` | `19.0993%` | `0.4939` | `0.5145` | `0.25` | `865.6884` |
| layers `[0, 1, 2, 3]` | `3.1365` | `21.5875%` | `0.4942` | `0.5174` | `0.50` | `634.9710` |

Generated-token smoke:

- layers `[0, 1]`: compressed generated tokens `[478, 141, 440, 163]`;
  exact top-1 tokens `[141, 103, 440, 490]`.
- layers `[0, 1, 2, 3]`: compressed generated tokens `[478, 141, 440, 163]`;
  exact top-1 tokens `[478, 103, 440, 490]`.

Status: `PASS_T008_SLICED_COMPRESSED_GENERATION_SMOKE`. This is a sliced
micro-generation measurement over real Qwen tensors. It is not a full
hidden-width/all-layer Qwen3-32B generation benchmark, not tokenizer-compatible
text generation, not a GPU-kernel result, not a VRAM residency measurement, and
not a default-path promotion.

---

## T009-T013 Qwen Runtime Scaler Bridge (2026-05-21)

Source: `T009-T013`

Artifacts:

- `src/transformer_residency/t009_t013_runtime_scaler.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t009-t013-runtime-scaler/latest/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t009-t013-runtime-scaler/latest/summary.md`

Validation:

```powershell
python -m unittest tests.test_transformer_residency_probe
python scripts/research/t001_transformer_residency_probe.py --t009-t013-runtime-scaler --local-model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137 --out-dir .work/jobs/t001-transformer-residency-probe/t009-t013-runtime-scaler/latest
```

The real smoke emitted `T009_T013_QWEN_RUNTIME_SCALER_ACTIVE` and passed.

Measured smoke facts:

| stage | scope | bits/weight | Q4 savings | avg projection rRMSE | pass count |
|---|---|---:|---:|---:|---:|
| T009 | full-width layer `0` projections | `3.0569` | `23.5783%` | `0.4367` | `7/7` |
| T010 | layers `[0, 1, 2, 3]`, row limit `512` | `3.2427` | `18.9313%` | `0.4494` | `28/28` |

Tokenizer and KV facts:

- T011 loaded the local Hugging Face tokenizer and round-tripped
  `Local compressed Qwen smoke:`.
- T012 prices Qwen3-32B FP16 KV at `256 KiB/token`.
- T012 16k tiered KV with 1024 exact recent tokens and 4-bit old KV is
  `1.1875 GiB`, saving `70.3125%` versus FP16 KV.

Full-model-equivalent pricing:

- candidate weights: `12.4382 GiB`
- candidate total with KV and runtime overhead at 16k: `13.4382 GiB`
- Q4 weights plus FP16 KV at 16k: `19.8057 GiB`
- candidate savings versus Q4+FP16KV: `6.3675 GiB`
- candidate fits 12 GiB: `false`
- memory-bound full-model-equivalent tok/s: `43.2359`
- Q4+FP16KV tok/s: `33.6917`
- speedup versus Q4+FP16KV: `1.2833x`

Status: `PASS_T009_T013_RUNTIME_SCALER_BRIDGE_WITH_12GIB_PRICING_BLOCKER`.
This is a full-width selected-projection and full-model-equivalent pricing
bridge over real Qwen artifacts. It is not measured full Qwen3-32B generation
throughput, not a GPU-kernel result, not measured VRAM residency, and not a
default-path promotion.

---

## T014 Denominator Tightening Gate (2026-05-21)

Source: `T014-A`

Artifacts:

- `src/transformer_residency/t014_denominator_tightening.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t014-denominator-tightening/latest/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t014-denominator-tightening/latest/summary.md`

Validation:

```powershell
python -m unittest tests.test_transformer_residency_probe
python scripts\research\t001_transformer_residency_probe.py --t014-denominator-tightening --local-model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137 --out-dir .work\jobs\t001-transformer-residency-probe\t014-denominator-tightening\latest
```

The real smoke emitted `T014_DENOMINATOR_TIGHTENING_ACTIVE` and passed.

Measured smoke facts:

| gate | scope | bits/weight | Q4 savings | avg projection rRMSE | pass count |
|---|---|---:|---:|---:|---:|
| T014 MLP | full-width layer `0` MLP projections | `2.4500` | `38.7500%` | `0.4848` | `3/3` |

Packed arena receipt:

- current overhead: `0.5500 GiB`
- packed overhead: `0.2500 GiB`
- saved: `0.3000 GiB`
- pass: `true`

Candidate pricing:

| candidate | total GiB | slack GiB | saved vs current T013 | tok/s proxy | speedup vs Q4+FP16KV |
|---|---:|---:|---:|---:|---:|
| current T013 | `13.4382` | `-1.4382` | `0.0000` | `43.2359` | `1.2833x` |
| T014-A / F15 | `11.8424` | `0.1576` | `1.5958` | `48.5250` | `1.4403x` |
| F21 near-term | `11.1773` | `0.8227` | `2.2609` | `50.5046` | `1.4990x` |
| F22 strong | `10.7187` | `1.2813` | `2.7195` | `52.6168` | `1.5617x` |
| F23 stretch | `10.0648` | `1.9352` | `3.3734` | `55.3292` | `1.6422x` |

Status: `PASS_T014_A_DENOMINATOR_GATE_NOT_FULL_INFERENCE`.
This is a real-weight denominator and MLP projection gate. It is not full
all-layer Qwen text inference, not measured full-model tok/s, not measured GPU
residency, and not a default-path promotion.

---

## T015 Packed Qwen Streamed Inference Smoke (2026-05-21)

Source: `T015`

Artifacts:

- `src/transformer_residency/t015_packed_qwen_inference.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t015-packed-qwen-inference/layer1/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t015-packed-qwen-inference/all64/summary.json`

Validation:

```powershell
python -m unittest tests.test_transformer_residency_probe
python scripts\research\t001_transformer_residency_probe.py --t015-packed-qwen-inference --local-model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137 --t015-prompt-text "Local compressed Qwen smoke:" --t015-max-new-tokens 1 --t015-layer-limit 1 --t015-device auto --t015-lm-head-chunk-rows 2048 --out-dir .work\jobs\t001-transformer-residency-probe\t015-packed-qwen-inference\layer1
python scripts\research\t001_transformer_residency_probe.py --t015-packed-qwen-inference --local-model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137 --t015-prompt-text "Local compressed Qwen smoke:" --t015-max-new-tokens 1 --t015-device auto --t015-lm-head-chunk-rows 2048 --out-dir .work\jobs\t001-transformer-residency-probe\t015-packed-qwen-inference\all64
```

The real smokes emitted `T015_PACKED_QWEN_STREAMED_INFERENCE_ACTIVE` and passed.

Measured smoke facts:

| smoke | layers | generated token | text | elapsed | tok/s | peak CUDA allocated | peak CUDA reserved |
|---|---:|---:|---|---:|---:|---:|---:|
| layer-1 | `1/1` | `89637` | `HomeAs` | `8.7862s` | `0.1138` | `0.4982 GiB` | `0.6680 GiB` |
| all-64 | `64/64` | `220` | `" "` | `212.6454s` | `0.0047` | `0.4982 GiB` | `0.6680 GiB` |

T014-A denominator used by T015:

- total: `11.8424 GiB`
- slack: `0.1576 GiB`

Status: `PASS_T015_FUNCTIONAL_ALL_LAYER_STREAMED_INFERENCE_SLOW`.
This proves functional all-layer token generation through the project
compressed-weight path over local Qwen safetensors. It is not optimized packed
residency throughput; the current path streams and reconstructs weights on the
fly, so useful tok/s requires a persistent packed GPU arena or cached project
format.

---

## T016 Persistent Packed Cache Smoke (2026-05-21)

Source: `T016`

Artifacts:

- `src/transformer_residency/t016_persistent_packed_cache.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t016-persistent-packed-cache/layer1-build/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t016-persistent-packed-cache/layer1-hit/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t016-persistent-packed-cache/all64-cache-fill/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t016-persistent-packed-cache/all64-cache-hit/summary.json`

Validation:

```powershell
python -m unittest tests.test_transformer_residency_probe
python scripts\research\t001_transformer_residency_probe.py --t016-persistent-packed-cache --local-model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137 --t016-prompt-text "Local compressed Qwen smoke:" --t016-max-new-tokens 1 --t016-layer-limit 1 --t016-device auto --t016-lm-head-chunk-rows 2048 --t016-refresh-cache --out-dir .work\jobs\t001-transformer-residency-probe\t016-persistent-packed-cache\layer1-build
python scripts\research\t001_transformer_residency_probe.py --t016-persistent-packed-cache --local-model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137 --t016-prompt-text "Local compressed Qwen smoke:" --t016-max-new-tokens 1 --t016-layer-limit 1 --t016-device auto --t016-lm-head-chunk-rows 2048 --out-dir .work\jobs\t001-transformer-residency-probe\t016-persistent-packed-cache\layer1-hit
python scripts\research\t001_transformer_residency_probe.py --t016-persistent-packed-cache --local-model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137 --t016-prompt-text "Local compressed Qwen smoke:" --t016-max-new-tokens 1 --t016-layer-limit 0 --t016-device auto --t016-lm-head-chunk-rows 2048 --out-dir .work\jobs\t001-transformer-residency-probe\t016-persistent-packed-cache\all64-cache-fill
python scripts\research\t001_transformer_residency_probe.py --t016-persistent-packed-cache --local-model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137 --t016-prompt-text "Local compressed Qwen smoke:" --t016-max-new-tokens 1 --t016-layer-limit 0 --t016-device auto --t016-lm-head-chunk-rows 2048 --out-dir .work\jobs\t001-transformer-residency-probe\t016-persistent-packed-cache\all64-cache-hit
```

The real smokes emitted `T016_PERSISTENT_PACKED_CACHE_ACTIVE` and passed.

Measured smoke facts:

| smoke | layers | cache state | built | loaded | generated token | text | elapsed | tok/s | peak CUDA allocated | peak CUDA reserved |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| layer-1 build | `1/1` | refresh/build | `82` | `0` | `89637` | `HomeAs` | `10.3344s` | `0.096764` | `1.0080 GiB` | `1.3320 GiB` |
| layer-1 hit | `1/1` | hot cache | `0` | `82` | `89637` | `HomeAs` | `2.5044s` | `0.399302` | `1.0080 GiB` | `1.3320 GiB` |
| all-64 fill | `64/64` | partial warm fill | `441` | `82` | `220` | `" "` | `229.9812s` | `0.004348` | `1.0080 GiB` | `1.3320 GiB` |
| all-64 hit | `64/64` | full hot cache | `0` | `523` | `220` | `" "` | `43.9065s` | `0.022776` | `1.0080 GiB` | `1.3320 GiB` |

The full cache contains `523` files under
`.work/cache/t016-persistent-packed-qwen/ede6860eaccc` and reads `32.8003 GiB`
on the hot all-64 run. T016 hot all-64 is about `4.85x` faster than T015
streamed all-64 (`0.022776 / 0.0047 tok/s`) while preserving generated token
`220` and text `" "`.

T014-A denominator used by T016:

- total: `11.8424 GiB`
- slack: `0.1576 GiB`

Status: `PASS_T016_PERSISTENT_PACKED_CACHE_PRE_KERNEL_BOUNDARY`.
This proves persistent compressed cache reuse for all-layer local Qwen
generation through the project path. It is not final packed GPU residency
throughput: the hot run still pays `torch.load` plus dense reconstruction before
matmul, with `23.5721s` load time and `19.0640s` reconstruction time.

---

## T017 Packed Linear Kernel Boundary (2026-05-21)

Source: `T017`

Artifacts:

- `src/transformer_residency/t017_packed_linear_kernel.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t017-packed-linear-kernel/synthetic-512/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t017-packed-linear-kernel/synthetic-attn-qwenish/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t017-packed-linear-kernel/synthetic-mlp-qwenish/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t017-kernel-qwen-inference/layer1-hit-repeat/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t017-kernel-qwen-inference/all64-hit-final/summary.json`

Validation:

```powershell
python -m unittest tests.test_transformer_residency_probe
python scripts\research\t001_transformer_residency_probe.py --t017-packed-linear-kernel --t017-rows 8192 --t017-cols 5120 --t017-tokens 6 --t017-repair-fraction 0.06 --t017-iterations 5 --out-dir .work\jobs\t001-transformer-residency-probe\t017-packed-linear-kernel\synthetic-attn-qwenish
python scripts\research\t001_transformer_residency_probe.py --t017-packed-linear-kernel --t017-rows 25600 --t017-cols 5120 --t017-tokens 6 --t017-repair-fraction 0.0155 --t017-iterations 3 --out-dir .work\jobs\t001-transformer-residency-probe\t017-packed-linear-kernel\synthetic-mlp-qwenish
python scripts\research\t001_transformer_residency_probe.py --t017-kernel-qwen-inference --local-model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137 --t017-prompt-text "Local compressed Qwen smoke:" --t017-max-new-tokens 1 --t017-layer-limit 0 --t017-device auto --t017-lm-head-chunk-rows 2048 --out-dir .work\jobs\t001-transformer-residency-probe\t017-kernel-qwen-inference\all64-hit-final
```

The real smokes emitted `T017_PACKED_LINEAR_KERNEL_ACTIVE` and passed.

Synthetic kernel gates:

| gate | rows | cols | tokens | repairs/row | speedup vs reconstruct+matmul | rRMSE |
|---|---:|---:|---:|---:|---:|---:|
| small | `512` | `512` | `4` | `16` | `2.5119x` | `2.78e-7` |
| attention-like | `8192` | `5120` | `6` | `308` | `3.8805x` | `7.24e-7` |
| MLP-like | `25600` | `5120` | `6` | `80` | `4.7723x` | `8.40e-7` |

Measured Qwen smoke facts:

| smoke | layers | generated token | text | elapsed | tok/s | kernel seconds | load seconds | peak CUDA allocated |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| layer-1 hot repeat | `1/1` | `89637` | `HomeAs` | `1.5295s` | `0.653822` | `0.5191s` | `0.5626s` | `0.1476 GiB` |
| all-64 hot final | `64/64` | `220` | `" "` | `35.6344s` | `0.028063` | `5.6611s` | `28.0466s` | `0.1476 GiB` |

Comparisons:

- versus T016 all-64 hot-cache: `1.231x` faster end-to-end and about `85.4%`
  lower peak CUDA allocation
- versus T015 all-64 streamed: about `5.97x` faster end-to-end

Status: `PASS_T017_PACKED_LINEAR_KERNEL_BOUNDARY_IO_BOUND`.
This proves direct packed-linear kernels for the current Qwen projection and
lm_head path. It is not useful steady-state throughput yet because packed-cache
loading dominates the all-64 run: `28.0466s` load time versus `5.6611s` kernel
time.

---

## T020 Device Credit Flow Static Gate (2026-05-25)

Source: `T020`

Artifacts:

- `src/transformer_residency/t020_device_credit_flow.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t020-device-credit-flow/b-math-w02-09/summary.json`
- `.workspace/research/t020-execution-shape-extraordinary/b-math-w02-09-implementation-log.md`

Validation:

```powershell
python -m py_compile .\src\transformer_residency\t017_packed_linear_kernel.py .\src\transformer_residency\t018_resident_payload_runtime.py .\src\transformer_residency\t019_binary_payload_runtime.py .\src\transformer_residency\t020_device_credit_flow.py .\scripts\research\t001_transformer_residency_probe.py
python -m unittest tests.test_transformer_residency_probe.TransformerResidencyProbeTests.test_t020_device_credit_flow_gate_prices_overlap_and_memory
python scripts\research\t001_transformer_residency_probe.py --t020-device-credit-flow --out-dir .work\jobs\t001-transformer-residency-probe\t020-device-credit-flow\b-math-w02-09
```

The gate emitted `T020_DEVICE_CREDIT_FLOW_ACTIVE` and passed.

Projected comparison against the T017 all-64 hot final:

| item | before | after | status |
|---|---:|---:|---|
| total seconds per token | `35.6344` | `30.086522` | projected static gate |
| tokens/s | `0.028063` | `0.0332377457` | projected static gate |
| speedup | `1.0000x` | `1.184397x` | projected, clears 10% floor |
| payload load seconds | `28.0466` | `22.3855` residual after overlap | projected static gate |
| live credit memory | `0.1576 GiB` slack | `0.1250 GiB` live credit window | fits 12 GiB envelope |

Status: `PASS_T020_DEVICE_CREDIT_FLOW_STATIC_GATE`.
This is a projection and opt-in receipt integration, not a measured new Qwen
runtime speedup. It prices the next scheduler/coalescing work against explicit
transfer, compute, sync, accumulator, and 12 GiB memory constraints.

---

## T020 Packed Work Coalescing Static Gate (2026-05-25)

Source: `T020`

Artifacts:

- `src/transformer_residency/t020_packed_work_coalescing.py`
- `src/transformer_residency/t017_packed_linear_kernel.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t020-packed-work-coalescing/b-pb-w02-11/summary.json`
- `.workspace/research/t020-execution-shape-extraordinary/b-pb-w02-11-implementation-log.md`

Validation:

```powershell
python -m py_compile .\src\transformer_residency\t017_packed_linear_kernel.py .\src\transformer_residency\t018_resident_payload_runtime.py .\src\transformer_residency\t019_binary_payload_runtime.py .\src\transformer_residency\t020_packed_work_coalescing.py .\scripts\research\t001_transformer_residency_probe.py
python -m unittest tests.test_transformer_residency_probe.TransformerResidencyProbeTests.test_t020_packed_work_coalescing_gate_maps_residual_destinations
python scripts\research\t001_transformer_residency_probe.py --t020-packed-work-coalescing --out-dir .work\jobs\t001-transformer-residency-probe\t020-packed-work-coalescing\b-pb-w02-11
```

The gate emitted `T020_PACKED_WORK_COALESCING_ACTIVE` and passed.

Structural comparison:

| item | before | after | status |
|---|---:|---:|---|
| legal residual-destination outputs per block | `0` fused | `2` fused | static gate |
| separate branch tensors for legal outputs | `2/block` | `0/block` | implemented opt-in |
| residual add boundaries across 64 layers | `128` standalone add sites | `128` fused into packed kernels | implemented opt-in |
| dense branch write+reread traffic | `5.0 MiB/token` | `0 MiB/token` for legal outputs | structural traffic gate |
| T017 all-64 tok/s | `0.028063` | not remeasured | payload-load denominator still dominant |

Status: `PASS_T020_PACKED_WORK_COALESCING_STATIC_GATE`.
This is structural proof and opt-in packed-kernel integration, not a measured
new Qwen runtime speedup. The next runtime measurement should wait until the
payload-load denominator is reduced enough that residual-branch coalescing is
not hidden under `28.0466s` of packed payload load.

---

## T020 Execution-Order Slab Arena Static Gate (2026-05-25)

Source: `T020`

Artifacts:

- `src/transformer_residency/t020_execution_order_slab.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t020-execution-order-slab/a-math-w01-01/summary.json`
- `.workspace/research/t020-execution-shape-extraordinary/a-math-w01-01-implementation-log.md`

Validation:

```powershell
python -m py_compile .\src\transformer_residency\t020_execution_order_slab.py .\scripts\research\t001_transformer_residency_probe.py .\tests\test_transformer_residency_probe.py
python -m unittest tests.test_transformer_residency_probe.TransformerResidencyProbeTests.test_t020_execution_order_slab_gate_reduces_acquisition_units
python scripts\research\t001_transformer_residency_probe.py --t020-execution-order-slab --out-dir .work\jobs\t001-transformer-residency-probe\t020-execution-order-slab\a-math-w01-01
```

The gate emitted `T020_EXECUTION_ORDER_SLAB_ARENA_ACTIVE` and passed.

Acquisition comparison:

| item | before | after | status |
|---|---:|---:|---|
| layer-1 acquisition objects | `82` | `1` | modeled gate |
| offset entries | `0` explicit slab offsets | `82` | modeled gate |
| acquisition seconds | `0.5214` | `0.1608750756` | modeled gate |
| acquisition speedup | `1.0000x` | `3.241024x` | clears `2x` kill gate |
| T017 all-64 tok/s | `0.028063` | not remeasured | slab file not integrated yet |

Status: `PASS_T020_EXECUTION_ORDER_SLAB_ARENA_STATIC_GATE`.
This is modeled acquisition-only pricing and an isolated CLI proof object, not
a measured new Qwen runtime speedup, not a production slab reader, and not a
default-path promotion.

---

## T020 Immutable Arena Receipt Static Gate (2026-05-25)

Source: `T020`

Artifacts:

- `src/transformer_residency/t020_immutable_arena_receipt.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t020-immutable-arena-receipt/a-math-w01-15/summary.json`
- `.workspace/research/t020-execution-shape-extraordinary/a-math-w01-15-implementation-log.md`

Validation:

```powershell
python -m py_compile .\src\transformer_residency\t020_immutable_arena_receipt.py .\scripts\research\t001_transformer_residency_probe.py .\tests\test_transformer_residency_probe.py
python -m unittest tests.test_transformer_residency_probe.TransformerResidencyProbeTests.test_t020_immutable_arena_receipt_gate_binds_hot_path_identity
python scripts\research\t001_transformer_residency_probe.py --t020-immutable-arena-receipt --out-dir .work\jobs\t001-transformer-residency-probe\t020-immutable-arena-receipt\a-math-w01-15
```

The gate emitted `T020_IMMUTABLE_ARENA_RECEIPT_ACTIVE` and passed.

Manifest comparison:

| item | before | after | status |
|---|---:|---:|---|
| layer-1 dynamic discovery seconds | `0.5214` | `0.005784` | modeled manifest gate |
| speedup vs dynamic discovery | `1.0000x` | `90.145228x` | modeled metadata/receipt gate |
| descriptor count | original payload objects | `82` compact descriptors | manifest root |
| hot-path original payload loads for verification | required | `0` | receipt gate |
| compression rerun/header scan in verification | allowed by old path | `0` | receipt gate |
| T017 all-64 tok/s | `0.028063` | not remeasured | slab reader not integrated yet |

Status: `PASS_T020_IMMUTABLE_ARENA_RECEIPT_STATIC_GATE`.
This is modeled manifest/receipt verification pricing and an isolated CLI proof
object, not a measured new Qwen runtime speedup, not a production slab reader,
and not a default-path promotion.

---

## T020 Conversion-Time Arena Compiler Static Gate (2026-05-25)

Source: `T020`

Artifacts:

- `src/transformer_residency/t020_conversion_time_arena_compiler.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t020-conversion-time-arena-compiler/a-pb-w02-14/summary.json`
- `.workspace/research/t020-execution-shape-extraordinary/a-pb-w02-14-implementation-log.md`

Validation:

```powershell
python -m py_compile .\src\transformer_residency\t020_conversion_time_arena_compiler.py .\scripts\research\t001_transformer_residency_probe.py .\tests\test_transformer_residency_probe.py
python -m unittest tests.test_transformer_residency_probe.TransformerResidencyProbeTests.test_t020_conversion_time_arena_compiler_gate_proves_descriptor_equality
python scripts\research\t001_transformer_residency_probe.py --t020-conversion-time-arena-compiler --out-dir .work\jobs\t001-transformer-residency-probe\t020-conversion-time-arena-compiler\a-pb-w02-14
```

The gate emitted `T020_CONVERSION_TIME_ARENA_COMPILER_ACTIVE` and passed.

Compiler comparison:

| item | before | after | status |
|---|---:|---:|---|
| visible interpretation denominator | `0.5214s` | `0.180184s` | modeled compiler gate |
| deleted denominator seconds | `0` | `0.341216s` | compile-time artifact |
| saved fraction | `0.0000` | `0.654423` | modeled compiler gate |
| speedup vs visible subfrontier | `1.0000x` | `2.893709x` | modeled compiler gate |
| runtime interpreter required | `true` | `false` | compiled views |
| compiled descriptor equality | not available | `true` over `8` descriptors | proof object |
| T017 all-64 tok/s | `0.028063` | not remeasured | compiled reader not integrated yet |

Status: `PASS_T020_CONVERSION_TIME_ARENA_COMPILER_STATIC_GATE`.
This is modeled compiler/descriptor pricing and an isolated CLI proof object,
not a measured new Qwen runtime speedup, not a production slab writer, and not a
default-path promotion.

---

## T020 Affine Packed View Static Gate (2026-05-25)

Source: `T020`

Artifacts:

- `src/transformer_residency/t020_affine_packed_view.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t020-affine-packed-view/a-math-w02-03/summary.json`
- `.workspace/research/t020-execution-shape-extraordinary/a-math-w02-03-implementation-log.md`

Validation:

```powershell
python -m py_compile .\src\transformer_residency\t020_affine_packed_view.py .\scripts\research\t001_transformer_residency_probe.py .\tests\test_transformer_residency_probe.py
python -m unittest tests.test_transformer_residency_probe.TransformerResidencyProbeTests.test_t020_affine_packed_view_gate_matches_materialized_values
python scripts\research\t001_transformer_residency_probe.py --t020-affine-packed-view --out-dir .work\jobs\t001-transformer-residency-probe\t020-affine-packed-view\a-math-w02-03
```

The gate emitted `T020_AFFINE_PACKED_VIEW_ACTIVE` and passed.

Affine-view comparison:

| item | before | after | status |
|---|---:|---:|---|
| materialization required | `true` | `false` | affine view |
| coordinates checked | `0` | `512` | exhaustive synthetic rectangle |
| max absolute error | not applicable | `0.0` | equality proof |
| net deleted seconds | `0` | `0.1500s` | modeled materialization deletion |
| saved fraction vs visible denominator | `0.0000` | `0.287687` | modeled affine gate |
| speedup vs visible subfrontier | `1.0000x` | `1.403877x` | modeled affine gate |
| T017 all-64 tok/s | `0.028063` | not remeasured | kernel view not integrated yet |

Status: `PASS_T020_AFFINE_PACKED_VIEW_STATIC_GATE`.
This is a synthetic mathematical equality gate and modeled materialization
pricing, not a measured new Qwen runtime speedup, not a production kernel view,
and not a default-path promotion.

---

## T020 Recompute-On-Reject Static Gate (2026-05-25)

Source: `T020`

Artifacts:

- `src/transformer_residency/t020_recompute_on_reject.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t020-recompute-on-reject/c-math-w02-02/summary.json`
- `.workspace/research/t020-execution-shape-extraordinary/c-math-w02-02-implementation-log.md`

Validation:

```powershell
python -m py_compile .\src\transformer_residency\t020_recompute_on_reject.py .\scripts\research\t001_transformer_residency_probe.py .\tests\test_transformer_residency_probe.py
python -m unittest tests.test_transformer_residency_probe.TransformerResidencyProbeTests.test_t020_recompute_on_reject_gate_defers_most_branch_positions
python scripts\research\t001_transformer_residency_probe.py --t020-recompute-on-reject --out-dir .work\jobs\t001-transformer-residency-probe\t020-recompute-on-reject\c-math-w02-02
```

The gate emitted `T020_RECOMPUTE_ON_REJECT_ACTIVE` and passed.

Branch-policy comparison:

| item | before | after | status |
|---|---:|---:|---|
| branch policy cost | `24.15552` | `12.32928` | synthetic branch tree |
| deleted rejected-branch cost | `0` | `11.82624` | synthetic branch tree |
| speedup vs eager KV writes | `1.0000x` | `1.959200x` | synthetic branch tree |
| stored branch positions | `16` | `4` | store/defer policy |
| deferred branch positions | `0` | `12` | exact recompute receipts |
| stored fraction | `1.0000` | `0.2500` | below `0.80` kill threshold |

Status: `PASS_T020_RECOMPUTE_ON_REJECT_STATIC_GATE`.
This is a synthetic branch-policy gate, not a measured decode/KV runtime
speedup, not a production speculative decoder, and not a default-path promotion.

---

## T020 Expected-Accepted-Span Static Gate (2026-05-25)

Source: `T020`

Artifacts:

- `src/transformer_residency/t020_expected_accepted_span.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t020-expected-accepted-span/c-math-w01-02/summary.json`
- `.workspace/research/t020-execution-shape-extraordinary/c-math-w01-02-implementation-log.md`

Validation:

```powershell
python -m py_compile .\src\transformer_residency\t020_expected_accepted_span.py .\scripts\research\t001_transformer_residency_probe.py .\tests\test_transformer_residency_probe.py
python -m unittest tests.test_transformer_residency_probe.TransformerResidencyProbeTests.test_t020_expected_accepted_span_gate_selects_legal_span
python scripts\research\t001_transformer_residency_probe.py --t020-expected-accepted-span --out-dir .work\jobs\t001-transformer-residency-probe\t020-expected-accepted-span\c-math-w01-02
```

The gate emitted `T020_EXPECTED_ACCEPTED_SPAN_ACTIVE` and passed.

Accepted-token denominator comparison:

| item | before | after | status |
|---|---:|---:|---|
| selected span length | `1` | `4` | high-confidence profile |
| expected accepted tokens | `0.92` | `3.26148096` | symbolic acceptance profile |
| cost per accepted token | `1.0178260869565217` | `0.7171838658227213` | denominator table |
| speedup vs one-token denominator | `1.0000x` | `1.419198x` | symbolic span table |
| collapse-after-first selected span | `1` | `1` | safe fallback |

Status: `PASS_T020_EXPECTED_ACCEPTED_SPAN_STATIC_GATE`.
This is a symbolic accepted-token denominator gate, not a measured decode/KV
runtime speedup, not a production speculative decoder, and not a default-path
promotion.

---

## T020 Prefix Transaction Ledger Static Gate (2026-05-25)

Source: `T020`

Artifacts:

- `src/transformer_residency/t020_prefix_transaction_ledger.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t020-prefix-transaction-ledger/c-math-w01-01/summary.json`
- `.workspace/research/t020-execution-shape-extraordinary/c-math-w01-01-implementation-log.md`

Validation:

```powershell
python -m py_compile .\src\transformer_residency\t020_prefix_transaction_ledger.py .\scripts\research\t001_transformer_residency_probe.py .\tests\test_transformer_residency_probe.py
python -m unittest tests.test_transformer_residency_probe.TransformerResidencyProbeTests.test_t020_prefix_transaction_ledger_gate_matches_serial_prefixes
python scripts\research\t001_transformer_residency_probe.py --t020-prefix-transaction-ledger --out-dir .work\jobs\t001-transformer-residency-probe\t020-prefix-transaction-ledger\c-math-w01-01
```

The gate emitted `T020_PREFIX_TRANSACTION_LEDGER_ACTIVE` and passed.

Prefix visibility comparison:

| item | before | after | status |
|---|---:|---:|---|
| average mutation cost | `2.12` | `0.24` | symbolic accept lengths |
| deleted mutation cost | `0` | `1.8800000000000001` | symbolic accept lengths |
| speedup vs eager mutation | `1.0000x` | `8.833333x` | modeled transaction ledger |
| rollback copy or compaction required | `true` | `false` | visibility ledger |
| accepted-prefix cases matching serial KV | `0/5` | `5/5` | symbolic receipt enumeration |

Status: `PASS_T020_PREFIX_TRANSACTION_LEDGER_STATIC_GATE`.
This is a symbolic prefix-visibility gate, not a measured decode/KV runtime
speedup, not a production speculative decoder, and not a default-path promotion.

---

## T020 Acceptance-Weighted KV Materialization Static Gate (2026-05-25)

Source: `T020`

Artifacts:

- `src/transformer_residency/t020_acceptance_weighted_kv_materialization.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t020-acceptance-weighted-kv-materialization/c-math-w01-10/summary.json`
- `.workspace/research/t020-execution-shape-extraordinary/c-math-w01-10-implementation-log.md`

Validation:

```powershell
python -m py_compile .\src\transformer_residency\t020_acceptance_weighted_kv_materialization.py .\scripts\research\t001_transformer_residency_probe.py .\tests\test_transformer_residency_probe.py
python -m unittest tests.test_transformer_residency_probe.TransformerResidencyProbeTests.test_t020_acceptance_weighted_kv_materialization_gate_prices_delay
python scripts\research\t001_transformer_residency_probe.py --t020-acceptance-weighted-kv-materialization --out-dir .work\jobs\t001-transformer-residency-probe\t020-acceptance-weighted-kv-materialization\c-math-w01-10
```

The gate emitted `T020_ACCEPTANCE_WEIGHTED_KV_MATERIALIZATION_ACTIVE` and
passed.

Delayed materialization comparison:

| item | before | after | status |
|---|---:|---:|---|
| materialization cost at `p=0.92` | `20.8` | `17.938089031680004` | symbolic branch tree |
| deleted materialization cost | `0` | `2.8619109683199966` | symbolic branch tree |
| speedup vs eager materialization | `1.0000x` | `1.159544x` | delayed exact KV |
| materializes rejected suffix | `true` | `false` | acceptance-weighted policy |
| break-even acceptance probability | not applicable | `0.97` | no gain beyond this |

Status: `PASS_T020_ACCEPTANCE_WEIGHTED_KV_MATERIALIZATION_STATIC_GATE`.
This is a symbolic delayed-materialization gate, not a measured decode/KV
runtime speedup, not a production speculative decoder, and not a default-path
promotion.

---

## T020 Basis-Shared KV Transform Static Gate (2026-05-25)

Source: `T020`

Artifacts:

- `src/transformer_residency/t020_basis_shared_kv_transform.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t020-basis-shared-kv-transform/c-math-w01-05/summary.json`
- `.workspace/research/t020-execution-shape-extraordinary/c-math-w01-05-implementation-log.md`

Validation:

```powershell
python -m py_compile .\src\transformer_residency\t020_basis_shared_kv_transform.py .\scripts\research\t001_transformer_residency_probe.py .\tests\test_transformer_residency_probe.py
python -m unittest tests.test_transformer_residency_probe.TransformerResidencyProbeTests.test_t020_basis_shared_kv_transform_gate_preserves_dot_products
python scripts\research\t001_transformer_residency_probe.py --t020-basis-shared-kv-transform --out-dir .work\jobs\t001-transformer-residency-probe\t020-basis-shared-kv-transform\c-math-w01-05
```

The gate emitted `T020_BASIS_SHARED_KV_TRANSFORM_ACTIVE` and passed.

Basis-transform comparison:

| item | before | after | status |
|---|---:|---:|---|
| representation crossing cost | `0.38` | `0.13` | diagonal dual basis |
| deleted crossing cost | `0` | `0.25` | diagonal dual basis |
| speedup vs representation crossing | `1.0000x` | `2.923077x` | symbolic basis gate |
| max dot-product error | not applicable | `8.881784197001252e-16` | exact dual-basis check |
| numerical debt vs margin | not checked | `2e-06 / 0.001` | stable enough |
| dense inverse required | possible | `false` | diagonal scale class |
| canonical and transformed KV both required | possible | `false` | no dual KV storage |

Status: `PASS_T020_BASIS_SHARED_KV_TRANSFORM_STATIC_GATE`.
This is an algebraic transform-class gate, not a measured decode/KV runtime
speedup, not a runtime KV storage change, and not a default-path promotion.

---

## T020 Ring-Exact Tail Receipts Static Gate (2026-05-25)

Source: `T020`

Artifacts:

- `src/transformer_residency/t020_ring_exact_tail_receipts.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t020-ring-exact-tail-receipts/c-math-w01-11/summary.json`
- `.workspace/research/t020-execution-shape-extraordinary/c-math-w01-11-implementation-log.md`

Validation:

```powershell
python -m py_compile .\src\transformer_residency\t020_ring_exact_tail_receipts.py .\scripts\research\t001_transformer_residency_probe.py .\tests\test_transformer_residency_probe.py
python -m unittest tests.test_transformer_residency_probe.TransformerResidencyProbeTests.test_t020_ring_exact_tail_receipts_gate_declares_adversarial_debt
python scripts\research\t001_transformer_residency_probe.py --t020-ring-exact-tail-receipts --out-dir .work\jobs\t001-transformer-residency-probe\t020-ring-exact-tail-receipts\c-math-w01-11
```

The gate emitted `T020_RING_EXACT_TAIL_RECEIPTS_ACTIVE` and passed.

Ring-tail comparison:

| item | before | after | status |
|---|---:|---:|---|
| active context cost | `16384.0` | `4904.200000000001` | symbolic context policy |
| deleted active context cost | `0` | `11479.8` | symbolic context policy |
| speedup vs full active context | `1.0000x` | `3.340810x` | symbolic context policy |
| GPU KV GiB | `4.0` | `1.0498046875` | exact ring only |
| exact ring tokens | `16384` | `4300` | ring-tail partition |
| far-anchor quality claim | possible | declared debt | no overclaim |

Status: `PASS_T020_RING_EXACT_TAIL_RECEIPTS_STATIC_GATE`.
This is a symbolic ring-tail feasibility gate, not a measured long-context
runtime speedup, not a quality claim for far-tail retrieval, and not a
default-path promotion.

---

## T020 Cumulative Execution-Shape Static Gate (2026-05-25)

Source: `T020`

Artifacts:

- `src/transformer_residency/t020_cumulative_execution_shape.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t020-cumulative-execution-shape/summary.json`
- `.workspace/research/t020-execution-shape-extraordinary/t020-cumulative-execution-shape-implementation-log.md`

Validation:

```powershell
python -m py_compile .\src\transformer_residency\t020_cumulative_execution_shape.py .\scripts\research\t001_transformer_residency_probe.py .\tests\test_transformer_residency_probe.py
python -m unittest tests.test_transformer_residency_probe.TransformerResidencyProbeTests.test_t020_cumulative_execution_shape_gate_declares_stack_speedup
python scripts\research\t001_transformer_residency_probe.py --t020-cumulative-execution-shape --out-dir .work\jobs\t001-transformer-residency-probe\t020-cumulative-execution-shape
```

The gate emitted `T020_CUMULATIVE_EXECUTION_SHAPE_ACTIVE` and passed.

Cumulative stack comparison:

| item | before | after | status |
|---|---:|---:|---|
| tokens/s proxy | `0.028063` | `10971.500528746225` | declared cumulative proxy |
| cumulative speedup | `1.0000x` | `390959.645396x` | composed T020 stack |
| numeric speedup components | `0` | `11` | all speedups >= 10 percent |
| structural components | `0` | `1` | packed coalescing traffic deletion |
| packed branch traffic deleted | `0 MiB/token` | `5.0 MiB/token` | structural packed-kernel gain |

Status: `PASS_T020_CUMULATIVE_EXECUTION_SHAPE_STATIC_GATE`.
This is the declared cumulative T020 proxy speedup from the implemented gates.
Production tok/s measurement remains a later validation step after runtime
decoder-path promotion.

---
