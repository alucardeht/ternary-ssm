# Measured Baselines — All Numbers

**Last updated:** 2026-05-21

---

## R422 Concept-Addressed Memory Gate (2026-05-13)

Source: `R422`

Artifacts:

- `scripts/research/r422_concept_addressed_memory_gate.py`
- `.workspace/research/r422-extraordinary-paradigm-horizons/concept-addressed-memory-implementation.md`
- `.workspace/research/r422-extraordinary-paradigm-horizons/concept-addressed-*.json`

Measurement:

| run | concepts | attr noise | decision | concept accuracy | nearest neighbor | soft prompt | topic bucket | work ratio |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| main | `128` | `0.00` | `PASS_CONCEPT_ADDRESSED_MEMORY_GATE` | `1.000000` | `0.358643` | `0.000000` | `0.150146` | `0.013021` |
| noise `0.05` | `128` | `0.05` | `FAIL_CONCEPT_ADDRESS_TRANSFER` | `0.606689` | `0.250732` | `0.000000` | `0.144043` | `0.013021` |
| noise `0.20` | `128` | `0.20` | `FAIL_CONCEPT_ADDRESS_TRANSFER` | `0.092285` | `0.126709` | `0.000000` | `0.122559` | `0.013021` |
| small | `32` | `0.00` | `FAIL_RETRIEVAL_OR_TOPIC_CONTROL` | `1.000000` | `0.777344` | `0.000000` | `0.292969` | `0.041667` |

Interpretation:

- `R422-SM-04 Concept-Addressed Non-Retrieval Memory` survives only as an
  exact/certified concept-address gate.
- It is brittle to address noise and fails when the concept family is small
  enough for nearest-neighbor controls to become too strong.
- No real concept-address observer, production v12 path, GPU speedup, VRAM/RAM
  change, or broad model-quality baseline change is claimed.

## R422 Compressed Palace State Gate (2026-05-13)

Source: `R422`

Artifacts:

- `scripts/research/r422_compressed_palace_state_gate.py`
- `.workspace/research/r422-extraordinary-paradigm-horizons/compressed-palace-state-implementation.md`
- `.workspace/research/r422-extraordinary-paradigm-horizons/compressed-palace-*.json`

Measurement:

| run | payload | axes | values | hops | decision | palace acc | nearest acc | topic acc | seen acc | flat oracle acc | palace work ratio |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| main | compositional | `5` | `9` | `6` | `PASS_COMPRESSED_PALACE_STATE_GATE` | `1.000000` | `0.187256` | `0.015137` | `0.183838` | `1.000000` | `0.001016` |
| random payload | random | `5` | `9` | `6` | `FAIL_NONCOMPOSITIONAL_PAYLOAD_CONTROL` | `0.011230` | `0.199219` | `0.017578` | `0.187744` | `1.000000` | `0.001016` |
| small flat | compositional | `4` | `4` | `3` | `FAIL_COMPRESSED_PALACE_STATE_GATE` | `1.000000` | `0.536133` | `0.069336` | `0.536133` | `1.000000` | `0.109375` |

Interpretation:

- `R422-SM-05 Compressed Palace State` passes as a compact compositional
  semantic-state navigation gate.
- The result does not hold for random non-compositional payloads, and small
  domains weaken the pricing/control separation.
- No real compiler, production v12 path, GPU speedup, VRAM/RAM change, or broad
  model-quality baseline change is claimed.

## R422 Reconstructive Anchor Memory Gate (2026-05-13)

Source: `R422`

Artifacts:

- `scripts/research/r422_reconstructive_anchor_memory_gate.py`
- `.workspace/research/r422-extraordinary-paradigm-horizons/reconstructive-anchor-memory-implementation.md`
- `.workspace/research/r422-extraordinary-paradigm-horizons/reconstructive-anchor-*.json`

Measurement:

| run | mode | variables | rank | anchor dim | decision | anchor acc | partial acc | lossy acc | surface acc | full memory acc | anchor work ratio |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| main | lowrank | `256` | `8` | `8` | `PASS_RECONSTRUCTIVE_ANCHOR_MEMORY_GATE` | `1.000000` | `0.004150` | `0.004395` | `0.002686` | `1.000000` | `0.031250` |
| random variables | random | `256` | `8` | `8` | `FAIL_NONRECONSTRUCTIVE_RANDOM_VARIABLE_CONTROL` | `0.005615` | `0.003418` | `0.002686` | `0.002441` | `1.000000` | `0.031250` |
| insufficient anchors | lowrank | `256` | `16` | `8` | `FAIL_INSUFFICIENT_ANCHOR_DIMENSION` | `0.004883` | `0.004395` | `0.005127` | `0.004883` | `1.000000` | `0.031250` |
| small expensive | lowrank | `48` | `8` | `8` | `FAIL_RECONSTRUCTIVE_ANCHOR_MEMORY_GATE` | `1.000000` | `0.002686` | `0.003174` | `0.004883` | `1.000000` | `0.166667` |

Interpretation:

- `R422-SM-06 Reconstructive Anchor Memory` passes as a low-rank delayed
  variable reconstruction gate.
- It fails when variables are random, when required factors are omitted, or when
  the delayed variable surface is too small to make anchors cheap.
- No real observer, production v12 path, GPU speedup, VRAM/RAM change, or broad
  model-quality baseline change is claimed.

## R422 Proof-Carrying Continuation Target Gate (2026-05-13)

Source: `R422`

Artifacts:

- `scripts/research/r422_proof_carrying_continuation_gate.py`
- `.workspace/research/r422-extraordinary-paradigm-horizons/proof-carrying-continuation-implementation.md`
- `.workspace/research/r422-extraordinary-paradigm-horizons/proof-continuation-*.json`

Measurement:

| run | mode | decision | identity error | false accept | cert length ratio | priced ratio | valid set reduction | weak false accept | long cert ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| main | strong | `PASS_PROOF_CARRYING_CONTINUATION_GATE` | `0.000000` | `0.000000` | `0.300000` | `0.330000` | `0.001003` | `1.000000` | `1.000000` |
| weak | weak | `FAIL_WEAK_CHECKER_CONTROL` | `0.000000` | `1.000000` | `0.100000` | `0.130000` | `0.500000` | `1.000000` | `1.000000` |
| long | long | `FAIL_LONG_CERTIFICATE_CONTROL` | `0.000000` | `0.000000` | `1.000000` | `1.030000` | `0.001003` | `1.000000` | `1.000000` |
| short target | strong | `FAIL_PROOF_CARRYING_CONTINUATION_GATE` | `0.000000` | `0.000000` | `0.600000` | `0.630000` | `0.001003` | `1.000000` | `1.000000` |

Interpretation:

- `R422-TO-03 Proof-Carrying Continuation Target` passes as a deterministic
  certificate-domain gate with exact checker identity and zero false accepts.
- Weak checkers and target-copy certificates fail their controls.
- No real compiler, production v12 path, GPU speedup, VRAM/RAM change, or broad
  model-quality baseline change is claimed.

## R422 Latent Transcript Marginal CE Gate (2026-05-13)

Source: `R422`

Artifacts:

- `scripts/research/r422_latent_transcript_marginal_ce_gate.py`
- `.workspace/research/r422-extraordinary-paradigm-horizons/latent-transcript-marginal-ce-implementation.md`
- `.workspace/research/r422-extraordinary-paradigm-horizons/latent-transcript-*.json`

Measurement:

| run | steps | transcripts | decision | canonical CE | exact CE | gain nats | top-k CE | top-k gap | weak CE | weak false mass | work ratio |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| main | `8` | `256` | `PASS_LATENT_TRANSCRIPT_MARGINAL_CE_GATE` | `4.663963` | `1.652778` | `3.011186` | `3.495572` | `1.842794` | `0.000000` | `0.778023` | `0.500000` |
| top-k | `8` | `256` | `FAIL_LOOSE_VARIATIONAL_TOPK_BOUND` | `4.663963` | `1.652778` | `3.011186` | `3.495572` | `1.842794` | `0.000000` | `0.778023` | `0.500000` |
| weak | `8` | `256` | `FAIL_WEAK_LATENT_CHECKER_CONTROL` | `4.663963` | `1.652778` | `3.011186` | `3.495572` | `1.842794` | `0.000000` | `0.778023` | `0.500000` |
| too large | `11` | `2048` | `FAIL_ENUMERATION_COST_CONTROL` | `6.387229` | `1.779736` | `4.607493` | `4.666301` | `2.886566` | `0.000000` | `0.808592` | `4.000000` |

Interpretation:

- `R422-TO-05 Latent Transcript Marginal CE` passes only as an exact tiny-domain
  marginalization proof object.
- Loose top-k bounds, weak checkers, and unpriced larger enumeration fail.
- No real marginalizer, production v12 path, GPU speedup, VRAM/RAM change, or
  broad model-quality baseline change is claimed.

## R422 Macro-Decision State Transition Objective Gate (2026-05-13)

Source: `R422`

Artifacts:

- `scripts/research/r422_macro_decision_state_transition_gate.py`
- `.workspace/research/r422-extraordinary-paradigm-horizons/macro-decision-state-transition-implementation.md`
- `.workspace/research/r422-extraordinary-paradigm-horizons/macro-decision-*.json`

Measurement:

| run | extractor noise | decision | macro state | macro residual | task success | surface control | state-only residual | residual-only task | work ratio |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| main | `0.00` | `PASS_MACRO_DECISION_STATE_TRANSITION_GATE` | `1.000000` | `1.000000` | `1.000000` | `0.033203` | `0.000000` | `0.033203` | `0.454545` |
| noisy | `0.08` | `FAIL_DETERMINISTIC_STATE_EXTRACTION` | `0.925537` | `1.000000` | `0.925537` | `0.029785` | `0.000000` | `0.029785` | `0.454545` |
| high noise | `0.25` | `FAIL_DETERMINISTIC_STATE_EXTRACTION` | `0.737305` | `1.000000` | `0.737305` | `0.032715` | `0.000000` | `0.032715` | `0.454545` |

Interpretation:

- `R422-TO-06 Macro-Decision State Transition Objective` passes as a
  deterministic state-transition plus residual-text material gate.
- Noisy extraction, state-only, residual-only, and surface-template controls
  fail.
- No real extractor, production v12 path, GPU speedup, VRAM/RAM change, or broad
  model-quality baseline change is claimed.

R422 Credit Market With CE Arbitrage addendum (2026-05-13): added `scripts/research/r422_credit_market_ce_arbitrage_gate.py` as a settled CE-bid allocation material gate for `R422-CA-03`. The settled run passed with market gain/work `2.9884627626529077`, gradient-norm `2.5831951970503857`, random `1.4075415959680777`, oracle `3.117638694278782`, market/grad `1.1568861563637451`, market/random `2.1231790031736186`, selected bad-bidder rate `0.00042877398303689063`, and raw no-settlement bad-bidder rate `0.32219963591825845`. The no-settlement control failed with raw bad-bidder rate `0.5606346871713396`; the high-noise miscalibrated-bid control failed with market/grad `0.5481774019806871`. Status: `POSITIVE_CREDIT_MARKET_SETTLEMENT_GATE_PENDING_GPU_ALLOCATOR`; no production v12 allocator, GPU speedup, VRAM/RAM change, or baseline change is claimed.

R422 Reversible Cotangent Transport addendum (2026-05-13): `R422-CA-05` is partially positive through the existing `R356 F1` exact reversible backward contract. The Makefile test target was repaired, the fixture cache semantics were corrected to store post-update `h[t]`, and `test_r356_f1.exe` passed `6 / 6` tests with one expected GPU parity skip. Batch-1, batch-2, and no-bilinear gradient checks matched canonical backward with cosine `1.00000000`; unstable replay was refused by the stability probe. Status: `PARTIAL_POSITIVE_EXACT_REVERSIBLE_COTANGENT_CONTRACT_GPU_REPLAY_MISSING`; no production GPU replay, GPU speedup, VRAM/RAM change, or baseline change is claimed.

R422 Obligation Passing addendum (2026-05-13): added `scripts/research/r422_obligation_passing_gate.py` as an observer-only gate for `R422-CA-07`. It reuses exact R405 state-erasure traces and tests whether a compact recent-write obligation handle selects originating CE liability beyond auxiliary CE, prior predictor, random, and raw block norm controls. The gate passed with min narrow signed value/work `1.7924426954366401`, min narrow precision `0.72`, min material positive share `0.1423510116443785`, min material signed value/work `1.2653992773888014`, max auxiliary material signed value/work `-0.033830572319908325`, and min obligation margin over raw block norm `0.6415197799917727`. Status: `POSITIVE_OBLIGATION_PASSING_OBSERVER_GATE_PENDING_GPU_INTEGRATION`; no production v12/GPU obligation emitter, training behavior change, measured speedup, VRAM/RAM change, or baseline change is claimed.

R422 Proof-Carrying Examples addendum (2026-05-13): added `scripts/research/r422_proof_carrying_examples_gate.py` as an independent proof-material compiler gate for `R422-TM-02`. The main 4096-example run passed with answer/support/failure accuracy `1.0`, contradiction and insufficient-premise accuracy `1.0`, corrupted proof reject rate `0.986328125`, random support accept rate `0.01708984375`, proof work ratio `0.1111111111111111`, full-scan work gain `9.0x`, and certificate token ratio `0.18181818181818182`. The small 8-item control still passed with proof work ratio `0.25` and full-scan gain `4.0x`. Status: `POSITIVE_PROOF_CARRYING_EXAMPLES_MATERIAL_GATE_PENDING_REAL_COMPILER`; no production v12 path, GPU speedup, VRAM/RAM change, or baseline change is claimed.

R422 Executable Corpus Packets addendum (2026-05-13): added `scripts/research/r422_executable_corpus_packets_gate.py` as a deterministic executable packet compiler gate for `R422-TM-03`. Main run passed with executable held-out accuracy `1.0`, text-only `0.05078125`, trace-nearest `0.02880859375`, hidden lookup `0.0`, shuffled code `0.02001953125`, mismatch detection `1.0`, corrupted trace accept `0.0`, packet work ratio `0.18494897959183673`, and gain versus best control `19.692307692307693x`. Dense-trace run also passed with packet work ratio `0.18181818181818182` and gain versus best control `44.52173913043478x`. Status: `POSITIVE_EXECUTABLE_CORPUS_PACKETS_MATERIAL_GATE_PENDING_REAL_COMPILER`; no production v12 path, GPU speedup, VRAM/RAM change, or baseline change is claimed.

R422 Exception-Ledger Curriculum addendum (2026-05-13): added `scripts/research/r422_exception_ledger_curriculum_gate.py` as a rule/exception/repair curriculum gate for `R422-TM-04`. Dense exception ledgers passed with exception rate `0.2650146484375`, ledger total/normal/exception accuracy `1.0`, total gain versus best control `0.1318359375`, exception gain `0.49746660525103636`, contradiction avoidance `1.0`, and ledger work ratio `0.1158021712907117`. Sparse/main ledgers failed total gain at `0.072265625` despite exception gain `0.5421245421245422`. Status: `PARTIAL_POSITIVE_EXCEPTION_LEDGER_DENSE_BOUNDARY_GATE_PENDING_REAL_COMPILER`; no production v12 path, GPU speedup, VRAM/RAM change, or baseline change is claimed.

R422 Macro-Span CE Material addendum (2026-05-13): added `scripts/research/r422_macro_span_ce_material_gate.py` as a delayed dependency/entity-state macro obligation gate for `R422-TM-05`. Main run passed with macro accuracy `1.0`, outline-only accuracy `0.1612548828125`, local-section accuracy `0.09588623046875`, shuffled-arc accuracy `0.0584716796875`, no future leakage, work ratio `0.09375`, and gain versus best admissible control `0.8387451171875`. Small run also passed with work ratio `0.1875` and gain `0.783447265625`. Status: `POSITIVE_MACRO_SPAN_CE_MATERIAL_GATE_PENDING_REAL_COMPILER`; no production v12 path, GPU speedup, VRAM/RAM change, or baseline change is claimed.

R422 Spatial-Symbolic Knowledge Sheets addendum (2026-05-13): added `scripts/research/r422_spatial_symbolic_sheets_gate.py` as a support math-contract gate for `R422-TM-06`. Main run passed with missing-fill, transpose, and text recovery accuracy all `1.0`; row-only `0.26220703125`, column-only `0.1095703125`, bag `0.1107421875`, shuffled `0.11875`, work ratio `0.037037037037037035`, and gain `0.73779296875`. Small run also passed with gain `0.65625`. Status: `POSITIVE_SUPPORT_SPATIAL_SYMBOLIC_SHEET_MATH_CONTRACT`; no production v12 path, GPU speedup, VRAM/RAM change, or baseline change is claimed.

R422 Repair Episode Distillation addendum (2026-05-13): added `scripts/research/r422_repair_episode_distillation_gate.py` as a verifier-generated repair episode material gate for `R422-CI-02`. Main run passed with first-pass exception accuracy `0.005859375`, second-pass exception accuracy `1.0`, held-out transfer `1.0`, best admissible control `0.0341796875`, dense-oracle work `1943.0`, repair work ratio `0.09222851260936703`, and gain `0.9658203125`. Small run also passed with gain `0.9224489795918367` and repair work ratio `0.22779661016949143`. Status: `POSITIVE_REPAIR_EPISODE_MATERIAL_GATE_PENDING_REAL_COMPILER`; no production v12 path, GPU speedup, VRAM/RAM change, or baseline change is claimed.

R422 Synthetic Workbench Worlds addendum (2026-05-13): added `scripts/research/r422_synthetic_workbench_worlds_gate.py` as an exact state/action/natural-residual material gate for `R422-CI-03`. Main run passed with answer accuracy `1.0`, long-composition accuracy `1.0`, state-update accuracy `1.0`, residual natural guard `1.0`, strongest admissible control `0.72802734375`, work ratio `0.022764661420123605`, and gain `0.27197265625`. Small run also passed with strongest admissible control `0.7847222222222222`, work ratio `0.03007518796992481`, and gain `0.2152777777777778`. Status: `POSITIVE_SYNTHETIC_WORKBENCH_WORLDS_MATERIAL_GATE_PENDING_REAL_COMPILER`; no production v12 path, GPU speedup, VRAM/RAM change, or baseline change is claimed.

R422 Evidence-Deferral Contract addendum (2026-05-13): added `scripts/research/r422_evidence_deferral_gate.py` as an answer/defer/request-evidence information-set certificate gate for `R422-CI-04`. Main run passed with mode accuracy `1.0`, answerable pass rate `1.0`, answer accuracy `1.0`, missing-request accuracy `1.0`, conflict-defer accuracy `1.0`, confident wrong rate `0.0`, utility gain versus best control `0.4224609375000255`, and certificate work ratio `0.5671792152704135`. Small run also passed with utility gain `0.44707031249999274`. Status: `POSITIVE_EVIDENCE_DEFERRAL_CONTRACT_MATERIAL_GATE_PENDING_REAL_COMPILER`; no production v12 path, GPU speedup, VRAM/RAM change, or baseline change is claimed.

R422 Compositional Skill Closure addendum (2026-05-13): added `scripts/research/r422_compositional_skill_closure_gate.py` as an atomic-operation plus closure-proof syllabus gate for `R422-CI-05`. Main run passed with unseen composition accuracy `1.0`, order-sensitive accuracy `1.0`, closure proof accuracy `1.0`, best admissible control `0.035400390625`, work ratio `0.0126953125`, and gain `0.964599609375`. Small run also passed with best admissible control `0.052083333333333336`, work ratio `0.0234375`, and gain `0.9479166666666666`. Status: `POSITIVE_COMPOSITIONAL_SKILL_CLOSURE_MATERIAL_GATE_PENDING_REAL_COMPILER`; no production v12 path, GPU speedup, VRAM/RAM change, or baseline change is claimed.

R422 Minimal Causal Intervention addendum (2026-05-13): added `scripts/research/r422_minimal_causal_intervention_gate.py` as a minimal certified causal intervention material gate for `R422-CI-06`. Main, small, and wide runs passed against the hard `distractor_flip_pairs` control. Main metrics: causal/counterfactual/invariance/base guard all `1.0`, score gain `0.6702962239583333`, counterfactual gain `0.7916666666666666`, invariance gain `0.7841796875`, base regression `0.0`, and correct-per-paid ratio `3.03770621586685`. Small score gain was `0.6695746527777777`; wide score gain was `0.714794921875`. Status: `POSITIVE_MINIMAL_CAUSAL_INTERVENTION_MATERIAL_GATE_PENDING_REAL_COMPILER`; no production v12 path, GPU speedup, VRAM/RAM change, or baseline change is claimed.

R422 Local Trace Autopsy addendum (2026-05-13): added `scripts/research/r422_local_trace_autopsy_gate.py` as a verified local trace autopsy material gate for `R422-CI-07`. Main, small, and wide runs passed as workflow-specific support material. Main metrics: final-state accuracy `1.0`, repair-action accuracy `1.0`, error localization `1.0`, verifier pass `1.0`, best control `plain_instruction_tokens`, score gain `0.8263671874999998`, final-state gain `1.0`, repair gain `1.0`, localization gain `0.1318359375`, and work ratio `0.005273437499999979`. Status: `POSITIVE_WORKFLOW_SPECIFIC_TRACE_AUTOPSY_MATERIAL_GATE`; no production v12 path, GPU speedup, VRAM/RAM change, broad model-quality baseline change, or general model-training claim is made.

## v12 C Binary (2026-03-26, DEV CONFIG)

| Metric | Value |
|--------|-------|
| Language | C99 + AVX2 SIMD, zero dependencies |
| Model | 2 layers, d=64, d_state=16, vocab=256, seq=64 |
| Loss | 5.556 → 5.541 (2000 steps, plain SGD, LR=0.1) |
| Binary size | ~90 KB |
| Compile time | <1s |
| AVX2 ternary_matvec | ACTIVE |
| Tests passing | 6/6 core modules |
| Status | **Loss decreasing confirmed** |

**Note:** dev config only. Full dims (6L/d=1024/vocab=32K) need Phase 2 speed fixes first.
R423 Project Growth Gates addendum (2026-05-14): added `scripts/research/r423_project_growth_gates.py` and `.workspace/training/rounds/r423.ps1` as the first executable bridge for R423 Aspect A native project-specific growth findings. The diagnostic preset `.workspace/training/rounds/r423.ps1 -Mode diagnostic -MaxTokens 50000 -Seed 42300` used `src/model_v12/wiki_1mb.bin`, emitted `[R423_PROJECT_GROWTH_GATES] active=1`, covered all 30 counted Aspect A items, and returned `pass_count=7`, `fail_count=1`, `invalid_count=0`, `no_gpu=1`, `no_training_integration=1`. Passing gate families and priced proxy ratios: microstep factorization `0.5335000002505357`, retraction-fiber split `0.045`, sign-lattice covering `0.7548958550163025`, readout null-denominator `0.3145825511891145`, genotype exception pressure `0.600066233342293`, receipt challenge `0.26923828125`, and interaction release safety `0.12494000522353818`. Hankel residual release failed the current finite-horizon sketch realization with `relative_gain=0.009814348284426679`, `priced_ratio=1.0401856517155732`, and projected proxy speedup `0.961366846726548x`. This is a CPU proof-object/proxy implementation only: no production training behavior, GPU speedup, VRAM/RAM change, or baseline training-time change is claimed. The canonical full 7B/60B projection remains `495d`, and lazy-refresh remains `165-247d`.

R423 Native Growth Readout Quotient addendum (2026-05-14): added the real-training bridge `--r423-native-growth-readout-quotient` and updated `.workspace/training/rounds/r423.ps1 -Mode real` to promote the R423 readout/null-denominator family through the existing Macro CE Quotient gradient-weighting path. `mingw32-make USE_VULKAN=1 train` passed with one pre-existing signedness warning at `src/train.c:4946`. The bounded GPU compare smoke `.workspace/training/rounds/r423.ps1 -Mode compare -Steps 2 -Batch 1 -Seed 42300 -LogEvery 1 -EvalEvery 2 -EvalBatches 1` wrote `.work/jobs/training-round-presets/r423/compare-20260514-003320/`. Baseline: step losses `41.3925 -> 39.4595`, eval loss `39.9249`, tok/s `516`, VRAM `740MB`. R423 bridge: emitted `[R423_NATIVE_GROWTH_READOUT_QUOTIENT] requested=1` and `active=1`, with `contexts=1204`, `classes=527`, `quotient_ratio=0.437708`, training enabled, and step-2 result `coverage=0.997432`, `priced_work_ratio=0.440276`, `net_speedup_estimate=2.271302`, `training_coverage=0.994990`, `training_effective_weight=0.999023`, tok/s `513-514`, VRAM `740MB`, eval loss `39.9249`. This is a real training-path activation and sample-efficiency/priced-work proxy, not a measured wall-clock speedup; the full 7B/60B projection remains `495d` until a longer matched-target run proves otherwise.

R423 Native Growth State Selector addendum (2026-05-14): added `--r423-native-growth-state-selector-out` and `--r423-native-growth-state-selector-json`, plus `.workspace/training/rounds/r423.ps1 -Mode state-selector`, as the first real growth-mask bridge for the sign-lattice / retraction-fiber direction. Build `mingw32-make USE_VULKAN=1 train` passed. The bounded GPU smoke `.workspace/training/rounds/r423.ps1 -Mode state-selector -Steps 2 -Batch 1 -Seed 42300 -LogEvery 1 -EvalEvery 2 -EvalBatches 1 -StateActiveFraction 0.5` wrote `.work/jobs/training-round-presets/r423/state-selector-20260514-004412/`. Export arm emitted `[R423_NATIVE_GROWTH_STATE_SELECTOR_EXPORT] requested=1` and `status=ok`, wrote an R423 selector with `64/128` active recurrent-state dimensions, and completed with eval loss `39.9249`, tok/s `512-515`, VRAM `740MB`. Train arm emitted `[R423_NATIVE_GROWTH_STATE_SELECTOR] requested=1` and `active=1`; each training application masked `393216` `W_in` gradient values and `393216` `W_out` gradient values, completed with eval loss `39.9249`, tok/s `512-513`, VRAM `740MB`, and `[TRAIN] Complete`. This is a real shape-preserving growth-control training path, not a measured speedup or memory reduction; the full 7B/60B projection remains `495d` until a longer matched-target run proves otherwise.

R423 Native Growth Microstep Birth addendum (2026-05-14): added `--r423-native-growth-microstep-birth-gate` and `.workspace/training/rounds/r423.ps1 -Mode microstep` as a real v12 parameter certificate for exact two-substep birth. Build `mingw32-make USE_VULKAN=1 train` passed with the existing signedness warning at `src/train.c:5083`. The bounded GPU smoke `.workspace/training/rounds/r423.ps1 -Mode microstep -Steps 2 -Batch 1 -Seed 42300 -LogEvery 1 -EvalEvery 2 -EvalBatches 1` wrote `.work/jobs/training-round-presets/r423/microstep-20260514-005403/`. The log emitted `[R423_NATIVE_GROWTH_MICROSTEP_BIRTH_GATE] requested=1` and `active=1`, checked `49152` real v12 layer/state synthetic transition states over `6` layers and `64` synthetic steps, reached `max_state_error=4.4408920985e-15`, `max_logit_error=4.4408920985e-15`, `micro_feature_ratio=0.9855846892`, `min_A=0.367879450321`, `max_A=0.998048782349`, and `pass=1`. The two training steps completed with losses `41.3925 -> 39.4595`, eval loss `39.9249`, tok/s `515-516`, VRAM `740MB`, and `[TRAIN] Complete`. This is an exact-birth certificate, not a released paid microstep or measured speedup; the full 7B/60B projection remains `495d`.

R423 Native Growth Receipt Challenge addendum (2026-05-14): added `--r423-native-growth-receipt-challenge-gate` and `.workspace/training/rounds/r423.ps1 -Mode receipt` as a real GPU-gradient observer for compact receipt/challenge retirement. Build `mingw32-make USE_VULKAN=1 train` passed. The bounded GPU smoke `.workspace/training/rounds/r423.ps1 -Mode receipt -Steps 2 -Batch 1 -Seed 42300 -LogEvery 1 -EvalEvery 2 -EvalBatches 1` wrote `.work/jobs/training-round-presets/r423/receipt-20260514-010640/`. The log emitted `[R423_NATIVE_GROWTH_RECEIPT_CHALLENGE_GATE] requested=1` and observed real `W_in` / `W_out` GPU gradients with `valid_states=768`; step 0 had `caught_harm_fraction=0.259740`, `accepted_harm_fraction=0.740260`, `priced_ratio=0.990781`, `primary_value=-0.532603`, `pass=0`; step 1 had `caught_harm_fraction=0.194805`, `accepted_harm_fraction=0.805195`, `priced_ratio=1.055716`, `primary_value=-0.649452`, `pass=0`. The two training steps completed with losses `41.3925 -> 39.4595`, eval loss `39.9249`, tok/s `515`, VRAM `740MB`, and `[TRAIN] Complete`. This is a real observer and implementation check, but the current stride-64 receipt statistic is not promoted as a release/retirement gain; no speedup, VRAM/RAM reduction, or baseline training-time change is claimed.

R423 Native Growth Interaction Release addendum (2026-05-14): added `--r423-native-growth-interaction-release-gate` and `.workspace/training/rounds/r423.ps1 -Mode interaction` as a real GPU-gradient observer for interaction-safe release. Build `mingw32-make USE_VULKAN=1 train` passed. The bounded GPU smoke `.workspace/training/rounds/r423.ps1 -Mode interaction -Steps 2 -Batch 1 -Seed 42300 -LogEvery 1 -EvalEvery 2 -EvalBatches 1` wrote `.work/jobs/training-round-presets/r423/interaction-20260514-011417/`. The log emitted `[R423_NATIVE_GROWTH_INTERACTION_RELEASE_GATE] requested=1` and observed live `dW_A` interaction gradients plus recurrent `A`: step 0 had `valid_releases=384`, `selected=76`, `commutator_harm_correlation=0.847282`, `safe_harm=3.18233946e-11`, `control_harm=3.1523563e-09`, `harm_reduction=0.989905`, `priced_ratio=0.050095`, `pass=1`; step 1 had `commutator_harm_correlation=0.879831`, `safe_harm=1.22252592e-11`, `control_harm=8.89652653e-10`, `harm_reduction=0.986258`, `priced_ratio=0.053742`, `pass=1`. The two training steps completed with losses `41.3925 -> 39.4595`, eval loss `39.9249`, tok/s `512-515`, VRAM `740MB`, and `[TRAIN] Complete`. This is a positive real-path observer for selecting safer interaction updates, not yet an applied release policy or measured wall-clock speedup; the full 7B/60B projection remains `495d` until a longer matched-target run or applied selector proves otherwise.

R423 Native Growth Genotype Exception addendum (2026-05-14): added `--r423-native-growth-genotype-exception-gate` and `.workspace/training/rounds/r423.ps1 -Mode genotype` as a real GPU-gradient observer for genotype-plus-exception pressure. Build `mingw32-make USE_VULKAN=1 train` passed. The bounded GPU smoke `.workspace/training/rounds/r423.ps1 -Mode genotype -Steps 2 -Batch 1 -Seed 42300 -LogEvery 1 -EvalEvery 2 -EvalBatches 1` wrote `.work/jobs/training-round-presets/r423/genotype-20260514-012051/`. The log emitted `[R423_NATIVE_GROWTH_GENOTYPE_EXCEPTION_GATE] requested=1` and observed live `dW_A` interaction gradients: step 0 had `matrices=3`, `rank=3`, `dense_bits=49152`, `genotype_bits=2304`, `exceptions=3`, `random_exceptions=17302`, `coverage=0.999939`, `exception_density=0.000061`, `description_ratio=0.047791`, `random_description_ratio=5.327026`, `pressure_gain=0.952209`, `priced_ratio=0.082791`, `pass=1`; step 1 had `exceptions=0`, `random_exceptions=17427`, `coverage=1.000000`, `exception_density=0.000000`, `description_ratio=0.046875`, `random_description_ratio=5.365173`, `pressure_gain=0.953125`, `priced_ratio=0.081875`, `pass=1`. The two training steps completed with losses `41.3925 -> 39.4595`, eval loss `39.9249`, tok/s `514`, VRAM `740MB`, and `[TRAIN] Complete`. This is a positive real-path observer for compact interaction-gradient structure, not yet an applied compressed update policy or measured wall-clock speedup; the full 7B/60B projection remains `495d` until a matched-target run or applied compact update proves otherwise.

R423 Native Growth Genotype Compact Update addendum (2026-05-14): added `--r423-native-growth-genotype-compact-update` and `.workspace/training/rounds/r423.ps1 -Mode genotype-applied` as the first applied genotype/exception training path. Build `mingw32-make USE_VULKAN=1 train` passed. The bounded GPU smoke `.workspace/training/rounds/r423.ps1 -Mode genotype-applied -Steps 2 -Batch 1 -Seed 42300 -LogEvery 1 -EvalEvery 2 -EvalBatches 1` wrote `.work/jobs/training-round-presets/r423/genotype-applied-20260514-012932/`. The log emitted `[R423_NATIVE_GROWTH_GENOTYPE_COMPACT_UPDATE] requested=1` and applied conservative live `dW_A` exception suppression only when the compact genotype criterion passed: step 0 compacted `3` of `49152` entries, `suppressed_abs_fraction=2.77272127e-05`, `description_ratio=0.047791`, `priced_ratio=0.092791`, `applied=1`, `pass=1`; step 1 compacted `0` entries, `description_ratio=0.046875`, `priced_ratio=0.091875`, `applied=1`, `pass=1`. The two training steps completed with losses `41.3925 -> 39.4595`, eval loss `39.9249`, tok/s `512`, VRAM `740MB`, and `[TRAIN] Complete`. This is a real opt-in training behavior and compactness-pressure bridge, not a measured wall-clock speedup, VRAM/RAM reduction, or compressed-kernel result; the full 7B/60B projection remains `495d` until a longer matched-target or compressed-update run proves otherwise.

R423 Native Growth Interaction Safe Update addendum (2026-05-14): added `--r423-native-growth-interaction-safe-update` and `.workspace/training/rounds/r423.ps1 -Mode interaction-applied` as the first applied interaction-safe training path. Build `mingw32-make USE_VULKAN=1 train` passed. The bounded GPU smoke `.workspace/training/rounds/r423.ps1 -Mode interaction-applied -Steps 2 -Batch 1 -Seed 42300 -LogEvery 1 -EvalEvery 2 -EvalBatches 1` wrote `.work/jobs/training-round-presets/r423/interaction-applied-20260514-013511/`. The log emitted `[R423_NATIVE_GROWTH_INTERACTION_SAFE_UPDATE] requested=1` and damped high-commutator live `dW_A` rows only when the safety criterion passed: step 0 damped `308` rows / `39424` entries, `observer_harm_reduction=0.989905`, `applied_harm_reduction=0.748587`, `priced_ratio=0.075095`, `applied=1`, `pass=1`; step 1 damped `308` rows / `39424` entries, `observer_harm_reduction=0.986258`, `applied_harm_reduction=0.748139`, `priced_ratio=0.078742`, `applied=1`, `pass=1`. The two training steps completed with losses `41.3925 -> 39.4595`, eval loss `39.9249`, tok/s `515-516`, VRAM `740MB`, and `[TRAIN] Complete`. This is a real opt-in training behavior and sample-efficiency/quality-per-update candidate, not a measured wall-clock speedup, VRAM/RAM reduction, or sparse-kernel result; the full 7B/60B projection remains `495d` until a matched-target run proves otherwise.

R423 Native Growth Applied Stack addendum (2026-05-14): updated `.workspace/training/rounds/r423.ps1 -Mode real` so the round's default real preset activates the R423 readout quotient, interaction-safe update, and genotype compact update paths that do not require an external selector artifact. Build `mingw32-make USE_VULKAN=1 train` passed. The bounded compare `.workspace/training/rounds/r423.ps1 -Mode compare -Steps 2 -Batch 1 -Seed 42300 -LogEvery 1 -EvalEvery 2 -EvalBatches 1` wrote `.work/jobs/training-round-presets/r423/compare-20260514-014008/`: baseline losses `41.3925 -> 39.4595`, eval loss `39.9249`, tok/s `508-509`, VRAM `740MB`; applied stack losses `41.3925 -> 39.4595`, eval loss `39.9249`, tok/s `508-512`, VRAM `740MB`, readout quotient step-2 `priced_work_ratio=0.440276`, `net_speedup_estimate=2.271302`, interaction safe update `applied_harm_reduction=0.748136-0.748638`, and genotype compact update `pass=1`. A follow-up 1-step real smoke `.workspace/training/rounds/r423.ps1 -Mode real -Steps 1 -Batch 1 -Seed 42300 -LogEvery 1 -EvalEvery 1 -EvalBatches 1` wrote `.work/jobs/training-round-presets/r423/real-20260514-014202/` and verified the parser-safe `[R423_NATIVE_GROWTH_GENOTYPE_COMPACT_UPDATE_SUMMARY]` line. This is the current reproducible R423 applied stack, but no final matched-target wall-clock speedup, VRAM/RAM reduction, or 7B/60B training-time reduction is measured; the full projection remains `495d` until a real longer run proves sample-efficiency progress or a sparse/compact optimizer reduces the denominator.
## R423 Native Growth Applied Stack Overnight Validation (2026-05-14)

Source: `R423`

Artifacts:

- `.workspace/research/r423-incremental-evolving-models/overnight-validation-2026-05-14.md`
- `.work/jobs/overnight/r423-applied-stack-20260514-032956/`

Measurement:

| item | baseline | R423 applied stack |
|---|---:|---:|
| dataset | `wiki_5mb.bin` | `wiki_5mb.bin` |
| target steps | `700` | `700` |
| completed logged step | `700` complete | `600` partial; activation through step `602` |
| duration | `93.80` min | `180.13` min, stopped by `max_minutes_180` |
| max temperature | `62 C` | `68 C` |
| max VRAM telemetry | `2172 MiB` | `2167 MiB` |
| max power | `95.01 W` | `107.43 W` |
| matched step 600 eval loss | `31.4752` | `31.4097` |
| process hygiene | clean | time-stop killed the active command and one project `train.exe` orphan; post-run check clean |

Activation:

- Applied stack emitted `31` `R423_NATIVE_GROWTH_READOUT_QUOTIENT_RESULT`
  lines, `604` `R423_NATIVE_GROWTH_INTERACTION_SAFE_UPDATE` lines, and `603`
  `R423_NATIVE_GROWTH_GENOTYPE_COMPACT_UPDATE_SUMMARY` lines.
- Late readout proxy at step `601` reported `priced_work_ratio=0.425750`,
  `net_speedup_estimate=2.348795`, `training_coverage=0.997218`, and
  `training_effective_weight=0.988406`.

Interpretation:

- This is a partial positive long-window activation/stability result: the R423
  applied stack remains connected to the real RTX 5070 Vulkan training path past
  warmup, and the run stayed inside the safety envelope.
- This is not a measured production wall-clock speedup. The current
  instrumented implementation has material real overhead and did not complete
  the 700-step applied arm inside the time guard.
- The baseline projection is unchanged: `495d` remains `495d`, and lazy
  `165-247d` remains `165-247d` until a matched-target run proves otherwise.

---
## R425 Conventional Update Objects Empirical Gate (2026-05-14)

Source: `R425`

Artifacts:

- `scripts/research/r425_update_objects_empirical_gate.py`
- `.workspace/research/r425-conventional-high-impact-paper/empirical-gate-closeout.md`
- `.workspace/research/r425-conventional-high-impact-paper/empirical-gate-result.json`
- `.workspace/research/r425-conventional-high-impact-paper/empirical-gate-result-d64.json`
- `.workspace/research/r425-conventional-high-impact-paper/empirical-gate-result-wiki5mb-d64.json`
- `.workspace/research/r425-conventional-high-impact-paper/empirical-gate-paperlite-5seed.json`
- `.workspace/research/r425-conventional-high-impact-paper/empirical-gate-sensitivity-c0005.json`
- `.workspace/research/r425-conventional-high-impact-paper/empirical-gate-sensitivity-c005.json`
- `.workspace/research/r425-conventional-high-impact-paper/empirical-gate-sensitivity-build10.json`

Measurement:

| run | decision | dataset | d_model | tokens | object/control ratio | checkpoint delta | raw replay | selected replay | distillation | gradient capsule | shuffled delta |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `empirical-gate-result` | `CHECKPOINT_DELTA_POSITIVE_CAPSULE_WEAK` | `wiki_1mb.bin` | 48 | 60000 | `2.487` | `0.600784` | `0.241524` | `0.232166` | `0.199657` | `0.190418` | `0.200517` |
| `empirical-gate-result-d64` | `CHECKPOINT_DELTA_POSITIVE_CAPSULE_WEAK` | `wiki_1mb.bin` | 64 | 80000 | `2.456` | `1.206092` | `0.491047` | `0.467791` | `0.385313` | `0.394254` | `0.411670` |
| `empirical-gate-result-wiki5mb-d64` | `CHECKPOINT_DELTA_POSITIVE_CAPSULE_WEAK` | `wiki_5mb.bin` | 64 | 150000 | `2.454` | `1.215255` | `0.495229` | `0.474986` | `0.390310` | `0.396957` | `0.415321` |
| `empirical-gate-paperlite-5seed` | `CHECKPOINT_DELTA_POSITIVE_CAPSULE_WEAK` | `wiki_5mb.bin` | 64 | 180000 | `2.468` | `1.165835` | `0.472301` | `0.451653` | `0.368169` | `0.383020` | `0.400597` |
| `empirical-gate-sensitivity-c0005` | `CHECKPOINT_DELTA_POSITIVE_CAPSULE_WEAK` | `wiki_1mb.bin` | 48 | 60000 | `1.871` | `0.451797` | `0.241524` | `0.232166` | `0.199657` | `0.190217` | `0.200422` |
| `empirical-gate-sensitivity-c005` | `CHECKPOINT_DELTA_POSITIVE_CAPSULE_WEAK` | `wiki_1mb.bin` | 48 | 60000 | `2.638` | `0.637100` | `0.241524` | `0.232166` | `0.199657` | `0.190430` | `0.200160` |
| `empirical-gate-sensitivity-build10` | `CHECKPOINT_DELTA_POSITIVE_CAPSULE_WEAK` | `wiki_1mb.bin` | 48 | 60000 | `1.379` | `0.333108` | `0.241524` | `0.232166` | `0.199657` | `0.111077` | `0.111178` |

Interpretation:

- Sparse checkpoint transition objects produced a real empirical positive in
  small conventional Transformer gates.
- The positive survives shuffled-delta ablation, replay controls,
  gradient-alignment selected replay, distillation, model-width change, shard
  change, compression sensitivity, and a pessimistic build-cost sensitivity.
- The naive gradient transaction capsule realization is weak and does not prove
  the broader update-object family.
- This is not a production speedup, not a v12 speed result, and not yet a
  submission-grade paper claim. It is a paper seed for `Checkpoint Delta Replay
  Objects`.

---

## R425 Mature Capsule Validation (2026-05-14)

Source: `R425`

Artifacts:

- `scripts/research/r425_update_objects_empirical_gate.py`
- `.workspace/research/r425-conventional-high-impact-paper/capsule-mature-closeout.md`
- `.workspace/research/r425-conventional-high-impact-paper/capsule-mature-gate-5seed.json`
- `.workspace/research/r425-conventional-high-impact-paper/capsule-mature-source48-5seed.json`
- `.workspace/research/r425-conventional-high-impact-paper/capsule-mature-source48-c005-5seed.json`
- `.workspace/research/r425-conventional-high-impact-paper/capsule-mature-source48-c0005-5seed.json`
- `.workspace/research/r425-conventional-high-impact-paper/capsule-mature-source48-c005-build10-5seed.json`
- `.workspace/research/r425-conventional-high-impact-paper/capsule-mature-d96-source48-c005-3seed.json`
- `.workspace/research/r425-conventional-high-impact-paper/paper-core-d96-source48-c005-5seed.json`

Measurement:

| run | source/replay | density | model | seeds | best capsule | capsule score | raw replay | shuffled capsule | capsule/raw | seed wins | verdict |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---|
| `capsule-mature-gate-5seed` | `24/12` | `0.02` | d64 | 5 | `optimizer_transaction_capsule` | `0.506960` | `0.472301` | `0.380432` | `1.073x` | `5/5` | weak positive |
| `capsule-mature-source48-5seed` | `48/12` | `0.02` | d64 | 5 | `optimizer_transaction_capsule` | `0.635740` | `0.489593` | `0.398688` | `1.298x` | `5/5` | positive |
| `capsule-mature-source48-c005-5seed` | `48/12` | `0.05` | d64 | 5 | `optimizer_transaction_capsule` | `0.904204` | `0.489593` | `0.399149` | `1.847x` | `5/5` | strong positive |
| `capsule-mature-source48-c0005-5seed` | `48/12` | `0.005` | d64 | 5 | `optimizer_transaction_capsule` | `0.483575` | `0.489593` | `0.398317` | `0.988x` | `1/5` | over-compressed failure |
| `capsule-mature-source48-c005-build10-5seed` | `48/12` | `0.05` | d64 | 5 | `optimizer_transaction_capsule` | `0.561091` | `0.489593` | `0.247686` | `1.146x` | `5/5` | material but below strict paper threshold |
| `capsule-mature-d96-source48-c005-3seed` | `48/12` | `0.05` | d96 | 3 | `optimizer_transaction_capsule` | `2.267360` | `1.261991` | `0.971832` | `1.797x` | `3/3` | positive scale check |
| `paper-core-d96-source48-c005-5seed` | `48/12` | `0.05` | d96 | 5 | `optimizer_transaction_capsule` | `2.231399` | `1.250833` | `0.987417` | `1.784x` | `5/5` | positive paper-core check |

Interpretation:

- The naive gradient-only capsule variants are not supported as a strong paper
  claim.
- `optimizer_transaction_capsule` is a positive conditional survivor: it wins
  when the source interval is longer than the replay budget and density is at
  least `0.02` in these gates.
- The strongest d64 setting was density `0.05`, with `1.847x` capsule/raw and
  `5/5` seed wins. A d96 paper-core scale check stayed positive at `1.784x`
  with `5/5` seed wins.
- The capsule is not robust to extreme over-compression (`0.005`) and becomes
  material-but-below-threshold under pessimistic build cost `10`.
- This is still a conventional small-Transformer paper gate, not a production
  v12 training speedup.

---

## R426 Theoretical Performance Gates (2026-05-14)

Source: `R426`

Artifacts:

- `scripts/research/r426_theoretical_performance_gates.py`
- `scripts/research/r426_checkpoint_lift.py`
- `.workspace/training/rounds/r426.ps1`
- `.workspace/research/r426-theoretical-performance-after-growth/implementation-closeout.md`
- `.work/jobs/training-round-presets/r426/diagnostic-20260514-163251/r426_theoretical_performance_gates.json`
- `.work/jobs/training-round-presets/r426/staged-build-20260514-164024/r426_staged_width_smoke.log`

Measurement:

| gate | decision | key zero/error metric | paid ratio | projected promoted ratio | interpretation |
|---|---|---:|---:|---:|---|
| `staged_width_structural_birth_gate` | `PASS_STAGED_WIDTH_STRUCTURAL_BIRTH_GATE` | `max_logit_error=0.0` | `0.6790625000000001` | `1.472618499769903x` | exact dormant birth plus phase-area accounting passed the CPU proof gate |
| `reachability_null_inactive_cone_gate` | `PASS_REACHABILITY_NULL_INACTIVE_CONE_GATE` | `max_omitted_grad=0.0` | `0.67` | `1.4925373134328357x` | structural non-participation certificate passed the CPU shadow-gradient audit |

Staged-width build smoke:

| build | dimensions | steps | batch | VRAM | throughput | interpretation |
|---|---|---:|---:|---:|---:|---|
| isolated v12 Vulkan staged-width binary | `D_MODEL=512`, `D_STATE=64`, `D_EMBED=64`, `N_HEADS=8` | `1` | `1` | `357 MB` | `1412 tok/s` | true smaller-realized binary compiled through `EXTRA_CFLAGS`, not full-width masking |

Matched 100-step smoke:

| arm | dimensions | command elapsed | completed | avg tok/s | final tok/s | avg VRAM | final loss |
|---|---|---:|---|---:|---:|---:|---:|
| full-width default v12 | `D_MODEL=1024`, `D_STATE=128`, `D_EMBED=128`, `N_HEADS=16` | `828.375s` | yes | `497.765` | `500` | `740 MB` | `36.4733` |
| isolated staged-width v12 | `D_MODEL=512`, `D_STATE=64`, `D_EMBED=64`, `N_HEADS=8` | `300.254s` including isolated copy/build | yes | `1443.031` | `1409` | `357 MB` | `20.3213` |

Derived comparison:

| metric | value |
|---|---:|
| staged/full average throughput ratio | `2.899x` |
| staged/full command elapsed ratio | `2.759x` |
| staged/full average VRAM ratio | `0.482x` |
| average VRAM saved | `383 MB` |

Staged growth / checkpoint lift 100-step smoke:

| arm | dimensions / path | elapsed | avg tok/s | avg VRAM | final loss | status |
|---|---|---:|---:|---:|---:|---|
| full-width 100-step baseline | `1024/128/128`, default v12 | `828.375s` | `497.765` | `740 MB` | `36.4733` | completed |
| staged 70 | `512/64/64`, isolated v12 | `219.769s` | `1427.174` parsed over 69/70 step logs | `357 MB` | `20.5976` | completed and saved staged checkpoint |
| checkpoint lift | overlap copy into full-width template | `~0.900s` | n/a | n/a | n/a | completed, `copied_entries=211`, `validated_flags=18` |
| full continuation 30 | lifted full-width checkpoint, `--checkpoint-load-only` | `247.957s` | `501.367` | `740 MB` | `37.3936` | completed from loaded step 70 to step 100 |

Derived staged growth / checkpoint lift comparison:

| metric | value |
|---|---:|
| elapsed excluding template creation | `468.626s` |
| elapsed including template creation | `476.744s` |
| elapsed ratio vs full-width 100-step baseline, excluding template creation | `1.768x` |
| elapsed ratio vs full-width 100-step baseline, including template creation | `1.738x` |
| weighted average VRAM over 70 staged + 30 full steps | `471.9 MB` |
| peak VRAM | `740 MB` |
| final loss delta vs full-width 100-step baseline | `+0.9203` nats |

Interpretation:

- This is a CPU proof-object gate suite for the two R426 validated families.
- The staged-width smoke is a GPU/Vulkan one-step build/run check for an
  isolated smaller binary. It exists to support no-reset model growth and does
  not promote the default production binary.
- The matched 100-step smoke is a short runtime comparison, not a
  matched-quality training result. The staged arm has lower loss because it is a
  smaller architecture, so the loss values are not a quality win claim.
- The staged growth / checkpoint lift smoke proves the primary capability: a
  smaller compiled v12 artifact can be lifted into a larger full-width v12
  checkpoint and continue GPU training from loaded step 70 to step 100 without
  resetting from scratch. This is the first concrete no-reset growth bridge
  toward a future `100M -> 300M -> 1B`-style ladder.
- The `1.738x-1.768x` short elapsed ratio and lower average VRAM are secondary
  bonuses from this exact smoke, not the central claim. The lifted arm ended
  `0.9203` nats worse after only 30 full-width catch-up steps, so the remaining
  validation is growth maturity/catch-up quality, not existence of the growth
  transition.
- The full-width resume path currently requires `--checkpoint-load-only` for
  this continuation test: loading and training are clean, but saving a final
  checkpoint after a resumed full-width step still crashes at the final-save
  lm_head materialization boundary.
- The CPU proof gates are not production training integration, not a measured
  wall-clock speedup, not a VRAM/RAM change, and not a baseline change.
- Full-size tensors plus selector masks must not be counted as implemented
  no-reset size growth.

---

## R427 Unified Growth-Ladder Activation Smoke (2026-05-15)

Source: `R427`

Artifacts:

- `.workspace/training/rounds/r427.ps1`
- `scripts/research/r427_growth_ladder_analyzer.py`
- `.workspace/research/r427-7b-local-quality-breakthrough/implementation-closeout.md`
- `.work/jobs/training-round-presets/r427/real-20260515-011234/r427_growth_ladder_analysis.json`

Measurement:

| arm/object | result |
|---|---|
| staged source | `1` step, loss `21.4925`, `1519 tok/s`, `357 MB`, checkpoint saved |
| full template | zero-step full-width envelope checkpoint saved |
| checkpoint lift | `[R426_CHECKPOINT_LIFT] ... status=ok` |
| lifted continuation | resumed from step `1`, completed `1` step, loss `40.5208`, `506 tok/s`, `740 MB` |
| analyzer | `PASS_R427_UNIFIED_GROWTH_LADDER_ACTIVATION` |

Growth-stack activation inside lifted continuation:

| mechanism | measured smoke signal |
|---|---|
| readout quotient | coverage `1.000000`, priced ratio `0.437708`, net proxy `2.284630` |
| interaction-safe update | `applied=1`, `pass=1`, priced ratio `0.077723` |
| genotype compact update | `applied=1`, `pass=1`, priced ratio `0.091875` |

Interpretation:

- This is an activation and lineage smoke for the first R427 block.
- It proves the executable path:
  `smaller compiled v12 -> full-width template -> lifted checkpoint -> resumed
  full-width training with live growth mechanisms`.
- It does not change the official training-time projection, quality target,
  production speed baseline, or VRAM/RAM baseline.
- The preset intentionally uses a zero-step full-width template because the
  template is only the checkpoint envelope. A one-step full-width template run
  still hits the known final-save fragility, so post-continuation checkpoint
  persistence remains a blocker for chained large growth.

---

## R427 Segment-Credit Certificate Smoke (2026-05-15)

Source: `R427`

Artifacts:

- `.workspace/training/rounds/r427.ps1 -Mode segment-credit`
- `scripts/research/r427_segment_credit_analyzer.py`
- `.workspace/research/r427-7b-local-quality-breakthrough/implementation-closeout.md`
- `.work/jobs/training-round-presets/r427/segment-credit-20260515-012132/r427_segment_credit_analysis.json`

Measurement:

| object | result |
|---|---|
| train smoke | `1` step, loss `41.3771`, `511 tok/s`, `740 MB` |
| analyzer | `PASS_R427_SEGMENT_CREDIT_CERTIFICATE_ACTIVATION` |
| disposition | `observer_activation_with_material_positive_signal` |
| variance-priced credit | `pass=1`, alignment `0.985817`, priced work `0.364670`, descent per paid work `2.665013` |
| original-CE surface credit | surface priced ratio `0.429917`, surface whole-step proxy `1.588699`, strict `surface_pass=0` because residual remained high |
| boundary cotangent tube | finite identifying metrics, `pass=0`, residual fraction `1.163308` |
| receipt challenge | finite real GPU receipt metrics, `pass=0`, caught harm fraction `0.207792`, priced ratio `1.042729` |

Interpretation:

- This is a real GPU observer/certificate activation for R427 block 2.
- It proves finite segment-credit pricing objects exist in the live training
  path.
- It does not change the official training-time projection, quality target,
  production speed baseline, or VRAM/RAM baseline.
- The only material pass in this smoke is variance-priced credit. Original-CE
  surface credit is a promising proxy signal, while boundary cotangent and
  receipt challenge remain finite but non-promoted observers in this
  configuration.

---

## R427 Capability-Dense Material Compiler Gate (2026-05-15)

Source: `R427`

Artifacts:

- `.workspace/training/rounds/r427.ps1 -Mode capability-material`
- `scripts/research/r427_capability_material_gate.py`
- `.workspace/research/r427-7b-local-quality-breakthrough/implementation-closeout.md`
- `.work/jobs/training-round-presets/r427/capability-material-20260515-013116/r427_capability_material_gate.json`
- `.work/jobs/training-round-presets/r427/capability-material-20260515-013116/r427_capability_material.jsonl`

Measurement:

| object | result |
|---|---|
| decision | `PASS_R427_CAPABILITY_MATERIAL_GATE` |
| strong sub-gates | `7/7` |
| material held-out accuracy | `1.000000` |
| strong raw-control held-out accuracy | `0.741567` |
| random-selector held-out accuracy | `0.000000` |
| material cost | `209.970000` |
| raw-control cost | `210.000000` |
| material rows available / used | `480 / 75` |
| atom / exception / rotation coverage | `1.000000 / 1.000000 / 1.000000` |

Interpretation:

- This is a CPU material/compiler gate for R427 block 3.
- It proves that the B-V1 through B-V7 capability-material families can be
  represented as one project-owned material artifact with priced held-out
  capability progress.
- It does not change the official training-time projection, quality target,
  production speed baseline, or VRAM/RAM baseline.
- The random-selector ratios are intentionally not used as production gain
  claims; the more informative sub-gates compare against the raw rule-inference
  control and clear `1.275776x`, `1.311219x`, and `1.348688x`.

---

## R427 Local Recurrent Artifact Shape Gate (2026-05-15)

Source: `R427`

Artifacts:

- `.workspace/training/rounds/r427.ps1 -Mode artifact-shape`
- `scripts/research/r427_local_artifact_shape_gate.py`
- `.workspace/research/r427-7b-local-quality-breakthrough/implementation-closeout.md`
- `.work/jobs/training-round-presets/r427/artifact-shape-20260515-013536/r427_local_artifact_shape_gate.json`
- `.work/jobs/training-round-presets/r427/artifact-shape-20260515-013536/r427_local_artifact_shape.json`

Measurement:

| object | result |
|---|---|
| decision | `PASS_R427_LOCAL_ARTIFACT_SHAPE_GATE` |
| sub-gates | `5/5` |
| recurrent transcript | replay CE-gap proxy `0.0`, state bytes `192`, KV-sidecar avoided proxy `12288` bytes |
| low-rank ternary delta | artifact ratio `0.097222`, reconstructed delta fraction `1.0` |
| periodic surfaces | scheduled/dense cost ratio `0.599119`, max alias residual `0.008` |
| quotient head | quotient/dense cost ratio `0.187500`, refiner fraction `0.125` |
| projection ledger | active fraction `0.2578125`, exception density `0.08`, retained sensitivity mass `0.617870` |

Interpretation:

- This is a CPU artifact-shape and cost-model gate for R427 block 4.
- It validates the compact final artifact family targeted by R427: fixed
  recurrent state, rank-bounded ternary delta, scheduled surfaces, quotient
  head, and projection ledger.
- It does not change the official training-time projection, quality target,
  production inference speed baseline, or VRAM/RAM baseline.

---

## R427 Integrated Trainable Preset Smoke (2026-05-15)

Source: `R427`

Artifacts:

- `.workspace/training/rounds/r427.ps1 -Mode integrated`
- `scripts/research/r427_materialize_training_bin.py`
- `scripts/research/r427_integrated_training_analyzer.py`
- `.workspace/research/r427-7b-local-quality-breakthrough/implementation-closeout.md`
- `.work/jobs/training-round-presets/r427/integrated-20260515-020528/r427_integrated_training_analysis.json`
- `.work/jobs/training-round-presets/r427/integrated-20260515-020528/r427_materialized_training_summary.json`

Measurement:

| object | result |
|---|---|
| decision | `PASS_R427_INTEGRATED_TRAINING_PRESET` |
| material gate | `PASS_R427_CAPABILITY_MATERIAL_GATE`, material accuracy `1.000000`, control accuracy `0.707341`, `7/7` sub-gates |
| artifact gate | `PASS_R427_LOCAL_ARTIFACT_SHAPE_GATE`, `5/5` sub-gates |
| materialized dataset | `65536` tokens, `131072` bytes, SHA-256 `c8c93c56b9b22bdfa3afb722874b6d6b98ee68a20eb760d416bb20fbaf795f72` |
| staged source | one step completed, loss `20.6404`, `1525 tok/s`, `357 MB` |
| full continuation | resumed from step `1`, one step completed, loss `40.5444`, `504 tok/s`, `740 MB` |
| live growth signals | readout quotient net proxy `3.000000`, interaction update `pass=1`, genotype update `pass=1` |
| live credit signals | boundary/original/receipt finite, variance-priced credit `pass=1` |

Interpretation:

- This is the first integrated executable R427 training preset: material
  compiler, artifact descriptor, deterministic training-bin materialization,
  no-reset checkpoint lift, resumed full-width training, growth stack, and
  segment-credit observers all run together.
- It makes R427 ready for a serious monitored training run from a coherent
  preset.
- It does not change the official `495d` projection, final 7B quality target,
  matched-target speed baseline, production inference speed baseline, or
  VRAM/RAM baseline.

---

## R428 Speed-To-Quality Control Layer Smoke (2026-05-15)

Source: `R428`

Artifacts:

- `.workspace/training/rounds/r428.ps1 -Mode plan`
- `.workspace/training/rounds/r428.ps1 -Mode integrated`
- `scripts/research/r428_speed_to_quality_controller.py`
- `.workspace/research/r428-real-training-speed-last-round/synthesis.md`
- `.work/jobs/training-round-presets/r428/integrated-20260515-032732/r428_speed_to_quality_analysis.json`
- `.work/jobs/training-round-presets/r427/integrated-20260515-032733/r427_integrated_training_analysis.json`

Measurement:

| object | result |
|---|---|
| decision | `PASS_R428_SPEED_TO_QUALITY_READY` |
| requested finding floor | `15/15` findings, `11/10` strong findings |
| profile | `interactive` |
| staged source | one step completed at `1537 tok/s` |
| full continuation | one step completed at `508 tok/s` |
| staged/full throughput ratio | `3.025591x` |
| readout quotient proxy | `3.000000` |
| interaction / genotype / variance | `pass=1 / pass=1 / pass=1` |
| required readiness checks | all true |
| first attempted seed | `42800` failed before GPU training with R427 material gate `5/7` |
| default seed after repair | `42780` |

Interpretation:

- This is a speed-to-quality control layer over the R427 integrated trainable
  preset. It records the half-extraordinary R428 finding set, profile policy,
  and readiness checks for a later serious monitored run.
- The round met the requested research floor: `15` findings with `11` strong
  findings. Isolated strong projections range from `1.15x` to conditional
  `2.5x`, but they are intentionally not multiplied into a headline training
  claim.
- The passing smoke proves the R428 controller can analyze a coherent R427
  integrated run and that the live growth/credit readiness signals are present.
- It does not change the official `495d` projection, final 7B quality target,
  matched-target speed baseline, production inference speed baseline, or
  VRAM/RAM baseline. Those require a serious monitored matched-target run.

---

