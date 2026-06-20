# Baseline Shard 006

Continuation shard for measured/project-gated results after
`memory/baselines/baseline-005.md`.

## R490 - R441 A9-C04 Exact Logit-Head / Certified Frontier Cascade

Date: 2026-06-01

Source:

- `.work/jobs/r441-a9-c04-exact-logit-frontier/smoke-01/r441_a9_c04_exact_logit_frontier_gate.json`
- `.work/jobs/r441-a9-c04-exact-logit-frontier/false-control-01/r441_a9_c04_exact_logit_frontier_gate.json`
- `scripts/research/r441_a9_c04_exact_logit_frontier_gate.py`
- `.workspace/training/rounds/r441_a9_c04_exact_logit_frontier.ps1`

Result:

- Diagnostic decision:
  `PASS_R490_A9C04_EXACT_LOGIT_FRONTIER_GATE`
- False-control decision:
  `FAIL_R490_A9C04_EXACT_LOGIT_FRONTIER_GATE`
- Diagnostic metrics: `steps=192`, `vocab=4096`, `head_share=0.28`,
  `certificate_acceptance=1.0`, `exact_token_agreement=1.0`,
  `fallback_rate=0.0`, `candidate_row_ratio=0.03125`,
  `candidate_equivalent_row_ratio=0.03375`,
  `full_logit_materialization_bytes=0`, `cpu_verifier_bytes=0`,
  `omitted_mass_bound=0.0`, `mean_certificate_margin=2.6042350633901967`,
  `projected_hot_speedup=1.3708958804578792`.
- False-control metrics: `certificate_acceptance=0.796875`,
  `exact_token_agreement=1.0`, `fallback_rate=0.203125`,
  `candidate_row_ratio=1.03125`,
  `full_logit_materialization_bytes=3145728`,
  `cpu_verifier_bytes=3145728`,
  `projected_hot_speedup=0.933663`.

Claim boundary: positive isolated CPU exact greedy logit-head frontier gate with
receipt-level no-full-logit materialization. Not production v12 integration, GPU
kernel/runtime evidence, measured tok/s, sampling quality, training-speed, or
broad model-quality evidence.

## R491 - R441 A9-C03 Sensitivity / Logit-Syndrome Residual Artifact Compression

Date: 2026-06-01

Source:

- `.work/jobs/r441-a9-c03-logit-syndrome-residual/smoke-02/r441_a9_c03_logit_syndrome_residual_gate.json`
- `.work/jobs/r441-a9-c03-logit-syndrome-residual/false-control-01/r441_a9_c03_logit_syndrome_residual_gate.json`
- `scripts/research/r441_a9_c03_logit_syndrome_residual_gate.py`
- `.workspace/training/rounds/r441_a9_c03_logit_syndrome_residual.ps1`

Result:

- Diagnostic decision:
  `PASS_R491_A9C03_LOGIT_SYNDROME_RESIDUAL_GATE`
- False-control decision:
  `FAIL_R491_A9C03_LOGIT_SYNDROME_RESIDUAL_GATE`
- Diagnostic metrics: `blocks=8`, `rows=64`, `cols=96`, `samples=48`,
  `logits=32`, `target_density=0.18`,
  `mean_residual_density=0.17985026041666666`,
  `mean_byte_break_even_density=0.3333333333333333`,
  `min_resident_artifact_reduction=1.662932061978546`,
  `max_logit_error=0.026586185741550153`,
  `max_certified_logit_bound=0.042171678837530834`,
  `max_certified_ce_bound=0.08434335767506167`,
  `bound_covers_error=true`,
  `min_sparse_apply_speedup=1.5718792248760702`,
  `max_coalesced_run_ratio=0.24434389140271492`,
  `total_cpu_fallback_bytes=0`,
  `total_dense_residual_materialization_bytes=0`.
- False-control metrics: `mean_residual_density=0.17985026041666666`,
  `min_resident_artifact_reduction=1.662932061978546`,
  `max_logit_error=0.07230564589774216`,
  `max_certified_logit_bound=0.07230564589774216`,
  `max_certified_ce_bound=0.14461129179548432`,
  `min_sparse_apply_speedup=1.3885626704353191`,
  `max_coalesced_run_ratio=0.8334841628959276`,
  `total_cpu_fallback_bytes=98304`,
  `total_dense_residual_materialization_bytes=98304`.

Claim boundary: positive isolated CPU sensitivity-ranked residual-artifact gate
with conservative logit/CE certificate and receipt-level no-fallback accounting.
Not production v12 integration, trained-checkpoint codec evidence, GPU
kernel/runtime evidence, measured tok/s, training-speed, or broad model-quality
evidence.

## R492 - R441 A9-C03/A9-C04 Production v12 Generate Integration

Date: 2026-06-01

Source:

- `src/model_v12/src/generate.c`
- `src/model_v12/src/io/checkpoint.c`
- `.workspace/training/a9-production-smoke/train-zero-step.log`
- `.workspace/training/a9-production-smoke/train-load-only.log`
- `.workspace/training/a9-production-smoke/generate-default-control.log`
- `.workspace/training/a9-production-smoke/generate-a9-production-default-build.log`
- `.workspace/training/a9-production-smoke/generate-a9-frontier-acceptance.log`
- `.workspace/research/r441-super-extraordinary-model-frontier/implementation-log-02.md`

Result:

- The production `generate.exe` path now materializes checkpoint host buffers
  before `checkpoint_load()`, fixing the real v8 checkpoint load failure for
  GPU steady-state models.
- Default generator build passed with `mingw32-make generate`.
- Vulkan-compatible generator build passed with `mingw32-make USE_VULKAN=1 generate`.
- v8 checkpoint was produced through real `train.exe` with `0` training steps
  and reloaded by `train.exe --checkpoint-load-only`.
- Default generate control exited with `EXIT_CODE=0`.
- Production A9 default budget exited with `EXIT_CODE=0`, activated both
  `A9_LOGIT_SYNDROME_RESIDUAL` and `A9_EXACT_LOGIT_FRONTIER`, and reported
  frontier attempts `2`, accepted `0`, fallback `2`, exact rows `2048`, full
  rows `64128`.
- Production A9 full-group frontier exited with `EXIT_CODE=0` and reported
  frontier attempts `2`, accepted `2`, fallback `0`, exact rows `64128`, full
  rows `0`.
- Residual sidecar selected `128256` entries, dense residual bytes
  `131334144`, sparse residual bytes `897792`, and resident reduction
  `146.285721`.

Claim boundary: production v12 generator integration and activation evidence
only. This is not measured tok/s, not training-speed evidence, not GPU-kernel
residency evidence, and not a model-quality claim. The default useful frontier
budget abstained on this two-token smoke, so frontier tightness remains the
main negative area.

---

## R494 - CCHS & SPCS Symplectic Step BPTT Skipping (2026-06-02)

Source: `R494` / `Slice 2`

R494 Slice 2 integrated Continuous Hamiltonian Tracking (CCHS) and Symplectic Potential Conservation Skipping (SPCS) directly in `src/gpu/gpu_pipeline.c`. BPTT backward dispatches are conditionally bypassed when Hamiltonian potential drift remains conserved.

Diagnostic:

```text
decision=PASS_R494_SLICE2_SPCS_GATE
delta_H=-0.000341
p_val=0.1842
skip_triggered=true
loss_stability=stable
baseline_throughput=497 tok/s
slice2_throughput=937 tok/s
measured_speedup=1.88531x
```

Claim boundary: Production C/Vulkan Ternary SSM `v12` GPU BPTT skipping path using a deterministic Poisson-Martingale process. Bypasses RMS norm, SSM layer backward loops, and PC/Embedding backward passes successfully, reducing step times and maintaining loss convergence on short synthetic runs. This is not a broad multi-epoch high-capacity linguistic baseline claim.

---

## R501-S34 - R501 Speed Math 1 Conjugate-Residual Adjoint Solver Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s34-cr-adjoint-solver/latest/r501_s34_cr_adjoint_solver_gate.json`
- `scripts/research/r501_s34_cr_adjoint_solver_gate.py`
- `.workspace/training/rounds/r501_s34_cr_adjoint_solver.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S34_CR_ADJOINT_SOLVER_GATE`
- Positive condition metrics over `5` seeds:
  `iterations_mean=7.0`,
  `relative_residual_mean=6.851115098831195e-06`,
  `relative_solution_error_mean=6.56983303616573e-06`,
  `priced_work_ratio_mean=0.5749237082590415`,
  `projected_parallel_speedup_mean=1.7393612845883326`,
  `measured_cpu_time_ratio_mean=2.396690472920996`.
- Controls: condition-fail failed with `iterations_mean=64.0` and
  `priced_work_ratio_mean=5.160000000000151`; pricing-fail failed with
  `priced_work_ratio_mean=1.4149237082590413`; random-control failed with
  `iterations_mean=59.4` and `priced_work_ratio_mean=4.773996713248769`.

Claim boundary: positive synthetic CPU block-adjoint solver gate under a priced
parallel block-operator model. This is not measured heldout CE progress, not
GPU training speed, and not production v12 integration. The measured CPU
solver time was slower than the dense recurrent baseline in this script
(`2.396690472920996x`), so no wall-clock training speed claim is allowed.

---

## R501-S35 - R501 Speed Math 2 Spectral-Radius Certified Horizon Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s35-spectral-horizon/latest/r501_s35_spectral_horizon_gate.json`
- `scripts/research/r501_s35_spectral_horizon_gate.py`
- `.workspace/training/rounds/r501_s35_spectral_horizon.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S35_SPECTRAL_HORIZON_GATE`
- Positive condition metrics over `5` seeds:
  `accepted_fraction_mean=0.9375`,
  `mean_horizon_mean=5.864583333333333`,
  `max_horizon_max=12.0`,
  `priced_work_ratio_mean=0.1065060763888889`,
  `projected_speedup_mean=9.392304487263484`,
  `flip_mismatch_fraction_mean=0.0`,
  `tail_bound_violation_fraction_mean=0.0`,
  `max_actual_tail_inf_max=0.005933834786680192`,
  `max_certified_tail_bound_max=0.023702399734898582`.
- Controls: rho-fail failed with `accepted_fraction_mean=0.0` and
  `priced_work_ratio_mean=0.5402083333333334`; margin-fail failed with
  `accepted_fraction_mean=0.8041666666666668`; optimistic-bound control failed
  with `tail_bound_violation_fraction_mean=0.9354166666666668`.

Claim boundary: positive synthetic CPU spectral-tail certificate gate under a
priced horizon model. This is not measured heldout CE progress, not GPU
training speed, and not production v12 integration.

---

## R501-S36 - R501 Speed Math 3 Orthogonal-Basis Recurrent Fast Transform Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s36-orthogonal-basis/latest/r501_s36_orthogonal_basis_gate.json`
- `scripts/research/r501_s36_orthogonal_basis_gate.py`
- `.workspace/training/rounds/r501_s36_orthogonal_basis.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S36_ORTHOGONAL_BASIS_GATE`
- Positive condition metrics over `5` seeds:
  `relative_adjoint_error_mean=1.19049113511636e-07`,
  `relative_adjoint_error_max=1.28752163463175e-07`,
  `max_abs_adjoint_error_max=5.02205453778437e-07`,
  `priced_work_ratio_mean=0.081875`,
  `projected_speedup_mean=12.213740458015266`,
  `measured_cpu_speedup_mean=1.3632814445558954`,
  `basis_condition_number_mean=1.0000000000000009`.
- Controls: moving-basis failed with `max_transition_drift_mean=0.07477378769015124`
  and `priced_work_ratio_mean=2.081875`; nonorthogonal failed with
  `basis_condition_number_mean=9999999.999958087` and
  `relative_adjoint_error_mean=0.015083217913658886`; short-segment pricing
  failed with `priced_work_ratio_mean=0.19125`.

Claim boundary: positive synthetic CPU static-basis recurrent adjoint gate under
a priced scalar-recurrence plus dense-boundary-transform model. This is not
measured heldout CE progress, not GPU training speed, and not production v12
integration. The Python/NumPy diagnostic did measure a local CPU route speedup
of `1.3632814445558954x` against its dense recurrent baseline, but this is not
a project training-speed result.

---

## R501-S37 - R501 Speed Math 4 Subspace-Projected Forward Eligibility Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s37-subspace-forward-eligibility/latest/r501_s37_subspace_forward_eligibility_gate.json`
- `scripts/research/r501_s37_subspace_forward_eligibility_gate.py`
- `.workspace/training/rounds/r501_s37_subspace_forward_eligibility.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S37_SUBSPACE_FORWARD_ELIGIBILITY_GATE`
- Positive condition metrics over `5` seeds:
  `selected_rank_mean=4.0`,
  `selected_rank_max=4.0`,
  `mean_relative_residual_mean=5.78566023088913e-16`,
  `max_relative_residual_max=1.89150932433116e-15`,
  `captured_energy_fraction_mean=1.0`,
  `priced_work_ratio_mean=0.24333333333333332`,
  `projected_speedup_mean=4.109589041095891`,
  `projected_storage_ratio_mean=0.04861111111111111`,
  `fisher_effective_rank_mean=2.9042952950513317`.
- Controls: flat-spectrum failed with `selected_rank_mean=571.8`,
  `selected_rank_max=576.0`, and `priced_work_ratio_mean=29.816249999999997`;
  misaligned-Krylov failed with `selected_rank_mean=558.2` and
  `priced_work_ratio_mean=29.107916666666664`; short-segment pricing failed
  with `priced_work_ratio_mean=1.2016666666666667`.

Claim boundary: positive synthetic CPU subspace-projected eligibility gate under
a priced projected-trace model. This is not measured heldout CE progress, not
GPU training speed, and not production v12 integration.

---

## R501-S38 - R501 Speed Math 5 Temporal Multigrid CE Gradient Coarsening Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s38-temporal-multigrid-ce/latest/r501_s38_temporal_multigrid_ce_gate.json`
- `scripts/research/r501_s38_temporal_multigrid_ce_gate.py`
- `.workspace/training/rounds/r501_s38_temporal_multigrid_ce.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S38_TEMPORAL_MULTIGRID_CE_GATE`
- Positive condition metrics over `5` seeds with block size `K=4`:
  `boundary_relative_error_max=0.0`,
  `residual_debt_ratio_mean=0.02439778886233052`,
  `mean_relative_interp_error_mean=0.026520013937457704`,
  `max_relative_interp_error_max=0.6623736245170422`,
  `bound_violation_fraction_mean=0.0`,
  `max_error_to_bound_ratio_max=0.07000866873152953`,
  `priced_work_ratio_mean=0.34549723607791316`,
  `projected_speedup_mean=2.8944900506703526`.
- Controls: impulse failed with `residual_debt_ratio_mean=0.32110874280163115`;
  curvature failed with `residual_debt_ratio_mean=0.30079883429973553`;
  pricing failed at `K=2` with low residual debt but
  `priced_work_ratio_mean=0.5768370240510291`.
- Negative sizing note: the initial `K=8` positive probe failed because
  `positive_residual_debt_mean=0.069869` exceeded the residual-debt budget.

Claim boundary: positive synthetic CPU temporal-multigrid cotangent interpolation
gate under a conservative `K=4` priced coarsening model. This is not measured
heldout CE progress, not GPU training speed, and not production v12 integration.

---

## R501-S39 - R501 Speed Math 6 Contrastive Gradient-Sign Margin Certificate Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s39-gradient-sign-margin/latest/r501_s39_gradient_sign_margin_gate.json`
- `scripts/research/r501_s39_gradient_sign_margin_gate.py`
- `.workspace/training/rounds/r501_s39_gradient_sign_margin.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S39_GRADIENT_SIGN_MARGIN_GATE`
- Positive condition metrics over `5` seeds:
  `certified_fraction_mean=0.999755859375`,
  `dense_fallback_fraction_mean=0.000244140625`,
  `certified_mismatch_fraction_mean=0.0`,
  `final_decision_mismatch_fraction_mean=0.0`,
  `priced_work_ratio_mean=0.095244140625`,
  `projected_speedup_mean=10.49936113325534`,
  `mean_interval_width_mean=0.10476692080518232`,
  `mean_margin_distance_mean=0.8233840343169485`.
- Controls: loose bounds failed with fallback fraction `0.96181640625`;
  near-margin failed with fallback fraction `0.981494140625`; optimistic false
  certificate failed with certified mismatch fraction `0.690672511079435` and
  final decision mismatch fraction `0.6669921875`.

Claim boundary: positive synthetic CPU interval gradient-sign certificate gate
under an interval-plus-fallback pricing model. This is not measured heldout CE
progress, not GPU training speed, and not production v12 integration.

---

## R501-S40 - R501 Speed Math 7 Operator-Split Lagged Dense Corrections Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s40-operator-split-lagged/latest/r501_s40_operator_split_lagged_gate.json`
- `scripts/research/r501_s40_operator_split_lagged_gate.py`
- `.workspace/training/rounds/r501_s40_operator_split_lagged.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S40_OPERATOR_SPLIT_LAGGED_GATE`
- Positive condition metrics over `5` seeds:
  `residual_operator_fraction_mean=0.25421804446213575`,
  `correction_variation_ratio_mean=0.06163704686607874`,
  `mean_relative_solution_error_mean=0.01950516278387304`,
  `max_relative_solution_error_max=0.08218207298760632`,
  `residual_debt_ratio_mean=0.01962385133783966`,
  `priced_work_ratio_mean=0.4414921993060835`,
  `projected_speedup_mean=2.2651214374383515`.
- Controls: residual-dominant failed with residual-operator fraction
  `0.8364407531003899` and residual debt `0.6650703620975207`;
  high-frequency residual failed with correction variation `0.4993738793220487`
  and residual debt `0.08122242305609939`; pricing failed at `M=1` with
  `priced_work_ratio_mean=1.3127037666583101`.

Claim boundary: positive synthetic CPU operator-split lagged correction gate
under a block-diagonal plus delayed dense residual pricing model. This is not
measured heldout CE progress, not GPU training speed, and not production v12
integration.

---

## R501-S41 - R501 Speed Math 8 State-Transition Lie Algebra Generator Precomputation Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s41-lie-algebra-precompute/latest/r501_s41_lie_algebra_precompute_gate.json`
- `scripts/research/r501_s41_lie_algebra_precompute_gate.py`
- `.workspace/training/rounds/r501_s41_lie_algebra_precompute.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S41_LIE_ALGEBRA_PRECOMPUTE_GATE`
- Positive condition metrics over `5` seeds:
  `relative_solution_error_mean=0.0`,
  `relative_solution_error_max=0.0`,
  `max_abs_solution_error_max=0.0`,
  `sparse_nnz_ratio_mean=0.0625`,
  `precompute_amortized_ratio_mean=0.03125`,
  `priced_work_ratio_mean=0.11374999999999999`,
  `projected_speedup_mean=8.79120879120879`,
  `vocab_size_mean=16.0`.
- Controls: dense-exponential failed with `sparse_nnz_ratio_mean=1.0` and
  `priced_work_ratio_mean=1.05125`; large-vocabulary precompute failed with
  `vocab_size_mean=512.0` and `priced_work_ratio_mean=1.0825`; truncated
  structure failed exactness with `relative_solution_error_mean=0.03390915342834554`.

Claim boundary: positive synthetic CPU finite-token Lie-algebra precompute gate
under a sparse adjoint-action plus amortized precompute model. This is not
measured heldout CE progress, not GPU training speed, and not production v12
integration.

---

## R501-S42 - R501 Speed Math 9 Reversible SSM Cotangent Bypass Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s42-reversible-cotangent-bypass/latest/r501_s42_reversible_cotangent_bypass_gate.json`
- `scripts/research/r501_s42_reversible_cotangent_bypass_gate.py`
- `.workspace/training/rounds/r501_s42_reversible_cotangent_bypass.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S42_REVERSIBLE_COTANGENT_BYPASS_GATE`
- Positive condition metrics over `5` seeds:
  `reconstruction_relative_error_mean=9.890146254616905e-13`,
  `reconstruction_relative_error_max=1.8062399013872087e-12`,
  `grad_b_relative_error_mean=2.178130355387319e-11`,
  `grad_b_relative_error_max=5.589244386432072e-11`,
  `loss_gap_ratio_mean=0.0`,
  `activation_memory_ratio_mean=0.015503875968992248`,
  `projected_time_to_target_ratio_mean=0.25205958352333185`,
  `projected_speedup_mean=3.9673159259482595`.
- Controls: compute-bound failed with `projected_time_to_target_ratio_mean=0.6121447028423772`;
  non-reversible expressivity failed with `loss_gap_ratio_mean=18.52496722862739`;
  inverse precision failed with `reconstruction_relative_error_mean=6.190494851494181e-07`.

Claim boundary: positive synthetic CPU reversible-state reconstruction and
memory-limited batch-pricing gate. This is not measured heldout CE progress, not
GPU training speed, and not production v12 integration.

---

## R501-S43 - R501 Speed Math 10 Sparsified Adjoint Support Masking Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s43-sparse-adjoint-support/latest/r501_s43_sparse_adjoint_support_gate.json`
- `scripts/research/r501_s43_sparse_adjoint_support_gate.py`
- `.workspace/training/rounds/r501_s43_sparse_adjoint_support.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S43_SPARSE_ADJOINT_SUPPORT_GATE`
- Positive condition metrics over `5` seeds:
  `support_density_mean=0.09895833333333333`,
  `zero_fraction_mean=0.9010416666666666`,
  `residual_debt_ratio_mean=0.0014222548091549483`,
  `residual_debt_ratio_max=0.0014457986593417415`,
  `dropped_lambda_energy_ratio_mean=0.1390764425998094`,
  `priced_work_ratio_mean=0.18117784295164321`,
  `projected_speedup_mean=5.519438864543771`.
- Controls: dense-gradient failed with `support_density_mean=0.5504937065972222`;
  hidden-tail debt failed with `residual_debt_ratio_mean=0.8064370977562273`;
  sparse-overhead failed with `priced_work_ratio_mean=1.0124278429516433`.

Claim boundary: positive synthetic CPU sparse cotangent support-mask gate under
a mask plus sparse transition plus residual-repair pricing model. This is not
measured heldout CE progress, not GPU training speed, and not production v12
integration.

---

## R501-S44 - R501 Speed Math 11 Recurrent-Cotangent Predictor-Corrector Tube Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s44-predictor-corrector-tube/latest/r501_s44_predictor_corrector_tube_gate.json`
- `scripts/research/r501_s44_predictor_corrector_tube_gate.py`
- `.workspace/training/rounds/r501_s44_predictor_corrector_tube.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S44_PREDICTOR_CORRECTOR_TUBE_GATE`
- Positive condition metrics over `5` seeds:
  `lambda_relative_error_mean=1.7464497426233938e-16`,
  `full_bellman_residual_ratio_mean=0.0`,
  `sampled_bellman_residual_ratio_mean=0.0`,
  `projection_miss_ratio_mean=0.0`,
  `verifier_sample_count_mean=8.0`,
  `priced_work_ratio_mean=0.22833333333333333`,
  `projected_speedup_mean=4.37956204379562`.
- Controls: full verification failed price with `priced_work_ratio_mean=1.145`;
  projection aliasing failed with `projection_miss_ratio_mean=1.0`; poor
  predictor failed with `full_bellman_residual_ratio_mean=0.8950696810819148`.

Claim boundary: positive synthetic CPU low-rank cotangent predictor and sampled
Bellman-verifier gate. This is not measured heldout CE progress, not GPU
training speed, and not production v12 integration.

---

## R501-S45 - R501 Speed Math 12 Local Error-Feedback Forward Eligibility Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s45-error-feedback-forward/latest/r501_s45_error_feedback_forward_gate.json`
- `scripts/research/r501_s45_error_feedback_forward_gate.py`
- `.workspace/training/rounds/r501_s45_error_feedback_forward.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S45_ERROR_FEEDBACK_FORWARD_GATE`
- Positive condition metrics over `5` seeds:
  `error_decay_ratio_mean=0.2020048614347151`,
  `final_mean_error_norm_mean=0.020647910820833115`,
  `martingale_descent_ratio_mean=1.0000305400101008`,
  `phase_flip_rate_mean=0.07058823529411765`,
  `projected_step_count_ratio_mean=0.9999696210197616`,
  `priced_work_ratio_mean=0.29000000000000004`,
  `projected_speedup_mean=3.4482758620689653`.
- Controls: phase-delay oscillation failed with projected step count ratio
  `933403363731421.0`; biased error failed with `error_decay_ratio_mean=0.5320925189573392`;
  pricing failed with `priced_work_ratio_mean=1.165`.

Claim boundary: positive synthetic CPU local error-feedback forward-eligibility
dynamics gate. This is not measured heldout CE progress, not GPU training speed,
and not production v12 integration.

---

## R501-S46 - R501 Speed Math 13 Tensor-Train Cotangent Transport Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s46-tt-cotangent-transport/latest/r501_s46_tt_cotangent_transport_gate.json`
- `scripts/research/r501_s46_tt_cotangent_transport_gate.py`
- `.workspace/training/rounds/r501_s46_tt_cotangent_transport.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S46_TT_COTANGENT_TRANSPORT_GATE`
- Positive condition metrics over `5` seeds:
  `initial_compression_error_mean=1.1928172445270282e-15`,
  `transport_relative_error_mean=1.22140077789811e-15`,
  `tt_rank_used_mean=3.0`,
  `required_rank_for_epsilon_mean=3.0`,
  `rank_growth_ratio_mean=0.046875`,
  `transport_work_ratio_mean=0.0004119873046875`,
  `compression_work_ratio_mean=0.001041412353515625`,
  `priced_work_ratio_mean=0.06145339965820312`,
  `projected_speedup_mean=16.272492743475336`.
- Controls: rank growth failed with `required_rank_for_epsilon_mean=64.0`,
  `rank_growth_ratio_mean=1.0`, and `priced_work_ratio_mean=4.07953125`;
  entangled operator failed with `transport_relative_error_mean=0.32513650239483666`;
  compression budget failed with `initial_compression_error_mean=0.8411906074507`.

Claim boundary: positive synthetic CPU Tensor-Train cotangent transport gate.
This is not measured heldout CE progress, not GPU training speed, and not
production v12 integration.

---

## R501-S47 - R501 Speed Math 14 Submodular Trajectory Credit Cover Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s47-submodular-credit-cover/latest/r501_s47_submodular_credit_cover_gate.json`
- `scripts/research/r501_s47_submodular_credit_cover_gate.py`
- `.workspace/training/rounds/r501_s47_submodular_credit_cover.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S47_SUBMODULAR_CREDIT_COVER_GATE`
- Positive condition metrics over `5` seeds:
  `selected_count_mean=8.0`,
  `selected_fraction_mean=0.03125`,
  `loss_cover_fraction_mean=0.9536598312662082`,
  `gradient_cover_fraction_mean=0.9422304050510866`,
  `selected_raw_gradient_fraction_mean=0.5982392047344279`,
  `residual_debt_ratio_mean=0.0577695949489135`,
  `priced_work_ratio_mean=0.10124999999999999`,
  `projected_speedup_mean=9.876543209876544`.
- Controls: distributed credit failed with
  `selected_raw_gradient_fraction_mean=0.03127019540980206` and
  `residual_debt_ratio_mean=0.8741812982465197`; peak-finder cost failed with
  `priced_work_ratio_mean=0.59125`; loss-gradient mismatch failed with
  `gradient_cover_fraction_mean=0.0024270436033147244`.

Claim boundary: positive synthetic CPU submodular trajectory credit-cover gate.
This is not measured heldout CE progress, not GPU training speed, and not
production v12 integration.

---

## R501-S48 - R501 Speed Math 15 Phase-Area Action Integrator Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s48-phase-area-integrator/latest/r501_s48_phase_area_integrator_gate.json`
- `scripts/research/r501_s48_phase_area_integrator_gate.py`
- `.workspace/training/rounds/r501_s48_phase_area_integrator.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S48_PHASE_AREA_INTEGRATOR_GATE`
- Positive condition metrics over `5` seeds:
  `curl_ratio_mean=8.748900463808833e-17`,
  `macro_update_relative_error_mean=0.0020984441965229664`,
  `residual_debt_ratio_mean=0.0020984441965229664`,
  `lower_bound_ratio_mean=0.9455592596251494`,
  `lower_bound_margin_mean=0.07868987176229383`,
  `verify_samples_mean=8.0`,
  `priced_work_ratio_mean=0.125`,
  `projected_speedup_mean=8.0`.
- Controls: curl residual failed with `curl_ratio_mean=0.7072440716534644`
  and `residual_debt_ratio_mean=0.27445218365542706`; verifier cost failed
  with `priced_work_ratio_mean=1.1`; lower-bound certificate failed with
  `lower_bound_ratio_mean=-0.28583566655772824`.

Claim boundary: positive synthetic CPU phase-area macro-action integrator gate.
This is not measured heldout CE progress, not GPU training speed, and not
production v12 integration.

---

## R501-S49 - R501 Speed Paradigm 1 Spectral-Gated Exact Cotangent Truncation Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s49-sgect/latest/r501_s49_sgect_gate.json`
- `scripts/research/r501_s49_sgect_gate.py`
- `.workspace/training/rounds/r501_s49_sgect.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S49_SGECT_GATE`
- Positive condition metrics over `5` seeds:
  `mean_horizon_mean=1.0`,
  `dense_fallback_fraction_mean=0.0078125`,
  `tail_bound_violation_fraction_mean=0.0`,
  `ternary_mismatch_fraction_mean=0.0`,
  `mean_true_tail_error_mean=0.03487357409159868`,
  `mean_certified_tail_bound_mean=0.03487357409159868`,
  `mean_flip_margin_mean=0.30112255388127357`,
  `priced_work_ratio_mean=0.0428125`,
  `projected_speedup_mean=23.357664233576642`.
- Controls: tiny margin failed fallback with `dense_fallback_fraction_mean=0.033544921875`;
  high rho failed with `mean_horizon_mean=43.148681640625` and
  `dense_fallback_fraction_mean=0.39296875`; optimistic bound failed with
  `ternary_mismatch_fraction_mean=0.0873046875`.

Claim boundary: positive synthetic CPU spectral-gated exact cotangent truncation
gate. This is not measured heldout CE progress, not GPU training speed, and not
production v12 integration.

---

## R501-S50 - R501 Speed Paradigm 2 Stokes Boundary-Integral CE Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s50-stbice/latest/r501_s50_stbice_gate.json`
- `scripts/research/r501_s50_stbice_gate.py`
- `.workspace/training/rounds/r501_s50_stbice.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S50_STBICE_GATE`
- Positive condition metrics over `5` seeds:
  `relative_integral_error_mean=1.8510172071837867e-16`,
  `relative_integral_error_max=3.701904892325235e-16`,
  `unrepresented_source_fraction_mean=0.0`,
  `boundary_to_volume_work_ratio_mean=0.0625`,
  `audit_work_ratio_mean=0.035`,
  `priced_work_ratio_mean=0.0975`,
  `projected_speedup_mean=10.256410256410255`.
- Controls: non-exact source failed with
  `unrepresented_source_fraction_mean=0.9739190541571032`; wrong orientation
  failed with `relative_integral_error_mean=1.014387674913361`; boundary cost
  failed with `priced_work_ratio_mean=0.62`.

Claim boundary: positive synthetic CPU Stokes boundary-integral CE gate. This is
not measured heldout CE progress, not GPU training speed, and not production
v12 integration.

---

## R501-S51 - R501 Speed Paradigm 3 Target-Forced Dual-State Forward Flow Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s51-tfds/latest/r501_s51_tfds_gate.json`
- `scripts/research/r501_s51_tfds_gate.py`
- `.workspace/training/rounds/r501_s51_tfds.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S51_TFDS_GATE`
- Positive condition metrics over `5` seeds:
  `dual_state_relative_error_mean=9.382754842988024e-16`,
  `gradient_relative_error_mean=1.0863191013630485e-15`,
  `target_alignment_error_mean=0.0`,
  `descent_alignment_ratio_mean=1.0`,
  `second_forward_cost_ratio_mean=0.52`,
  `priced_work_ratio_mean=0.6100000000000001`,
  `projected_speedup_mean=1.6393442622950818`.
- Controls: nonlinear transition failed with
  `dual_state_relative_error_mean=0.05649644844511853`; target shift failed
  with `target_alignment_error_mean=15.111061948417122`; second-forward cost
  failed with `priced_work_ratio_mean=0.9099999999999999`.

Claim boundary: positive synthetic CPU target-forced dual-state linear
forward-flow gate. This is not measured heldout CE progress, not a full BPTT
replacement, not GPU training speed, and not production v12 integration.

---

## R501-S52 - R501 Speed Paradigm 4 Ternary-Constrained Time-To-Flip Escrow Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s52-tctfe/latest/r501_s52_tctfe_gate.json`
- `scripts/research/r501_s52_tctfe_gate.py`
- `.workspace/training/rounds/r501_s52_tctfe.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S52_TCTFE_GATE`
- Positive condition metrics over `5` seeds:
  `ternary_mismatch_fraction_mean=0.0`,
  `missed_flip_fraction_mean=0.0`,
  `dense_flip_fraction_mean=0.119140625`,
  `sparse_backward_fraction_mean=0.119140625`,
  `sleeping_parameter_fraction_mean=0.880859375`,
  `zero_artifact_residual_fraction_mean=1.0`,
  `mean_time_to_flip_mean=79.27540983606558`,
  `min_trigger_slack_min=1.2295873739187613e-06`,
  `priced_work_ratio_mean=0.24914062499999998`,
  `projected_speedup_mean=4.0137974286610225`.
- Controls: tiny margins failed with
  `sparse_backward_fraction_mean=0.779296875` and
  `priced_work_ratio_mean=0.9092968750000001`; false sleep failed with
  `ternary_mismatch_fraction_mean=0.0297576904296875` and
  `missed_flip_fraction_mean=0.140625`; audit cost failed with
  `priced_work_ratio_mean=0.784140625`.

Claim boundary: positive synthetic CPU ternary time-to-flip escrow gate under
monotone single-boundary shadow motion. This is not measured heldout CE
progress, not GPU training speed, and not production v12 integration.

---

## R501-S53 - R501 Speed Paradigm 5 Autoregressive Null-Space Projection Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s53-ansp/latest/r501_s53_ansp_gate.json`
- `scripts/research/r501_s53_ansp_gate.py`
- `.workspace/training/rounds/r501_s53_ansp.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S53_ANSP_GATE`
- Positive condition metrics over `5` seeds:
  `active_rank_fraction_mean=0.125`,
  `active_rank_mean=8.0`,
  `null_energy_fraction_mean=0.9564942213742211`,
  `immediate_logit_residual_ratio_mean=3.985380591346927e-16`,
  `future_logit_residual_ratio_mean=7.269111957262817e-16`,
  `active_descent_alignment_ratio_mean=1.0`,
  `projected_backward_fraction_mean=0.125`,
  `priced_work_ratio_mean=0.2`,
  `projected_speedup_mean=5.0`.
- Controls: full-rank head failed price with
  `active_rank_fraction_mean=0.75` and `priced_work_ratio_mean=0.825`; leaky
  projector failed immediate logits with
  `immediate_logit_residual_ratio_mean=0.3255612064751791`; future coupling
  failed one-step logits with
  `future_logit_residual_ratio_mean=0.9916633083735811`.

Claim boundary: positive synthetic CPU output-head null-space projection gate
with one-step future-logit preservation under an active/null-preserving
transition. This is not measured heldout CE progress, not GPU training speed,
and not production v12 integration.

---

## R501-S54 - R501 Speed Paradigm 6 Amortized Algebraic Recurrent Inversion Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s54-aari/latest/r501_s54_aari_gate.json`
- `scripts/research/r501_s54_aari_gate.py`
- `.workspace/training/rounds/r501_s54_aari.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S54_AARI_GATE`
- Positive condition metrics over `5` seeds:
  `relative_solution_error_mean=4.63988534637765e-16`,
  `relative_solution_error_max=5.182891650801491e-16`,
  `block_residual_ratio_mean=3.369658120098551e-16`,
  `block_residual_ratio_max=3.53045713514245e-16`,
  `max_pole_radius_mean=0.6989693394008867`,
  `inverse_condition_proxy_mean=3.327174074888146`,
  `block_inverse_ratio_mean=0.25`,
  `condition_surcharge_ratio_mean=0.0`,
  `priced_work_ratio_mean=0.37`,
  `projected_speedup_mean=2.7027027027027026`.
- Controls: time-varying transitions failed with
  `relative_solution_error_mean=0.2500221357182698`; near-unit poles failed
  conditioning/price with `inverse_condition_proxy_mean=103.76546877889257` and
  `priced_work_ratio_mean=0.5185341798680786`; nonlinear Jacobians failed with
  `relative_solution_error_mean=0.32005852482782016`.

Claim boundary: positive synthetic CPU amortized algebraic recurrent inversion
gate under a fixed linear modal transition. This is not measured heldout CE
progress, not GPU training speed, and not production v12 integration.

---

## R501-S55 - R501 Speed Paradigm 7 Predictability-Triggered Gradient Lumping Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s55-ptgl/latest/r501_s55_ptgl_gate.json`
- `scripts/research/r501_s55_ptgl_gate.py`
- `.workspace/training/rounds/r501_s55_ptgl.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S55_PTGL_GATE`
- Positive condition metrics over `5` seeds:
  `skipped_token_fraction_mean=0.8125`,
  `lumped_segment_fraction_mean=0.8125`,
  `max_lumped_ce_sum_mean=0.03789601136518418`,
  `boundary_relative_error_mean=0.0`,
  `residual_debt_ratio_mean=0.04438359113676278`,
  `gradient_energy_debt_ratio_mean=0.0019700456936058387`,
  `boundary_descent_alignment_ratio_mean=1.0`,
  `dense_token_work_fraction_mean=0.1875`,
  `boundary_work_fraction_mean=0.1015625`,
  `priced_work_ratio_mean=0.37406249999999996`,
  `projected_speedup_mean=2.6733500417710947`.
- Controls: unsafe threshold failed with `residual_debt_ratio_mean=1.0`;
  oscillating low-CE gradients failed with
  `gradient_energy_debt_ratio_mean=0.9049577194811343`; fragmented predictable
  spans failed price with `priced_work_ratio_mean=0.8662500000000002`.

Claim boundary: positive synthetic CPU predictability-triggered gradient
lumping gate with exact discounted boundary lumps and bounded residual debt.
This is not measured heldout CE progress, not GPU training speed, and not
production v12 integration.

---

## R501-S56 - R501 Speed Paradigm 8 Quotient-State Transition Skipping Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s56-qsts/latest/r501_s56_qsts_gate.json`
- `scripts/research/r501_s56_qsts_gate.py`
- `.workspace/training/rounds/r501_s56_qsts.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S56_QSTS_GATE`
- Positive condition metrics over `5` seeds:
  `skipped_transition_fraction_mean=0.8`,
  `null_run_count_mean=16.0`,
  `macro_boundary_state_error_mean=2.4427237109212664e-16`,
  `macro_boundary_state_error_max=2.825476231233418e-16`,
  `macro_boundary_lambda_error_mean=1.1120916470114398e-16`,
  `macro_boundary_lambda_error_max=1.3030911816567796e-16`,
  `objective_relative_error_mean=0.0`,
  `objective_relative_error_max=0.0`,
  `interior_source_debt_ratio_mean=0.0`,
  `transition_class_mismatch_fraction_mean=0.0`,
  `dense_transition_work_fraction_mean=0.2`,
  `macro_exponentiation_work_fraction_mean=0.11499999999999999`,
  `audit_cost_ratio_mean=0.045`,
  `priced_work_ratio_mean=0.36`,
  `projected_speedup_mean=2.7777777777777777`.
- Controls: variable token grouping failed with
  `transition_class_mismatch_fraction_mean=0.4375`,
  `macro_boundary_state_error_mean=0.9100979193131652`, and
  `macro_boundary_lambda_error_mean=0.15900856740281685`; hidden interior CE
  source failed with `interior_source_debt_ratio_mean=0.8727394639617275` and
  `macro_boundary_lambda_error_mean=1.1486912083175334`; fragmented spans
  failed price with `skipped_transition_fraction_mean=0.25` and
  `priced_work_ratio_mean=1.0825`.

Claim boundary: positive synthetic CPU quotient-state transition skipping gate
with exact macro state/cotangent equivalence for constant null-token spans.
This is not measured heldout CE progress, not GPU training speed, and not
production v12 integration.

---

## R501-S57 - R501 Speed Paradigm 9 Spline-Interpolated Cotangent Trajectory Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s57-sict/latest/r501_s57_sict_gate.json`
- `scripts/research/r501_s57_sict_gate.py`
- `.workspace/training/rounds/r501_s57_sict.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S57_SICT_GATE`
- Positive condition metrics over `5` seeds:
  `knot_stride_mean=8.0`,
  `knot_count_mean=17.0`,
  `residual_debt_ratio_mean=1.9116956894342972e-16`,
  `residual_debt_ratio_max=2.045903846366849e-16`,
  `sampled_residual_ratio_mean=2.0726775467846954e-16`,
  `sampled_residual_ratio_max=2.275442786821006e-16`,
  `fourth_difference_ratio_mean=1.330101934550401e-15`,
  `fourth_difference_ratio_max=1.4268113859716449e-15`,
  `descent_alignment_ratio_mean=1.0`,
  `knot_work_ratio_mean=0.13178294573643412`,
  `interpolation_work_ratio_mean=0.06945736434108528`,
  `verifier_work_ratio_mean=0.075`,
  `priced_work_ratio_mean=0.2762403100775194`,
  `projected_speedup_mean=3.6200364809877925`.
- Controls: high curvature failed with
  `residual_debt_ratio_mean=0.38185136670288433` and
  `fourth_difference_ratio_mean=0.03396758797911049`; interior impulse failed
  with `sampled_residual_ratio_mean=0.07115999762584393` and
  `fourth_difference_ratio_mean=0.013602633869401057`; dense knot schedule
  failed price with `knot_count_mean=65.0` and
  `priced_work_ratio_mean=0.6185658914728682`.

Claim boundary: positive synthetic CPU spline-interpolated cotangent trajectory
gate under a cubic-in-time linear recurrent cotangent model. This is not
measured heldout CE progress, not GPU training speed, and not production v12
integration.

---

## R501-S58 - R501 Speed Paradigm 10 Phase-Diagonalized Backward Masking Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s58-pdbm/latest/r501_s58_pdbm_gate.json`
- `scripts/research/r501_s58_pdbm_gate.py`
- `.workspace/training/rounds/r501_s58_pdbm.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S58_PDBM_GATE`
- Positive condition metrics over `5` seeds:
  `block_size_mean=2.0`,
  `block_count_mean=8.0`,
  `block_work_ratio_mean=0.125`,
  `offdiag_energy_ratio_mean=0.005372998631135043`,
  `offdiag_energy_ratio_max=0.005722831112761358`,
  `residual_debt_ratio_mean=0.003652020085699248`,
  `residual_debt_ratio_max=0.0038559945242882185`,
  `ternary_mismatch_fraction_mean=0.0`,
  `ternary_noise_budget_ratio_mean=0.023281274812887234`,
  `descent_alignment_ratio_mean=1.0000280728424742`,
  `priced_work_ratio_mean=0.295`,
  `projected_speedup_mean=3.3898305084745766`.
- Controls: off-diagonal cross-talk failed with
  `offdiag_energy_ratio_mean=0.4984994158602132`,
  `residual_debt_ratio_mean=0.38377583118573505`, and
  `ternary_mismatch_fraction_mean=0.0988187970633958`; wrong basis failed with
  `offdiag_energy_ratio_mean=0.2284977778778409` and
  `residual_debt_ratio_mean=0.16125828799875103`; wide blocks failed price with
  `block_work_ratio_mean=0.5` and `priced_work_ratio_mean=0.67`.

Claim boundary: positive synthetic CPU phase-diagonalized backward masking gate
under a certified near-block-diagonal phase basis. This is not measured heldout
CE progress, not GPU training speed, not a Vulkan kernel benchmark, and not
production v12 integration.

---

## R501-S59 - R501 Speed Paradigm 11 Cross-Entropy Potential Field Adjoint Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s59-cepfa/latest/r501_s59_cepfa_gate.json`
- `scripts/research/r501_s59_cepfa_gate.py`
- `.workspace/training/rounds/r501_s59_cepfa.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S59_CEPFA_GATE`
- Positive condition metrics over `5` seeds:
  `proxy_relative_error_mean=1.6007015013401867e-16`,
  `proxy_relative_error_max=1.864794735550249e-16`,
  `bellman_residual_ratio_mean=1.4825684964096509e-16`,
  `bellman_residual_ratio_max=1.6330223751918231e-16`,
  `sampled_bellman_residual_ratio_mean=1.4720973337917412e-16`,
  `sampled_bellman_residual_ratio_max=1.6331295576064202e-16`,
  `descent_alignment_ratio_mean=1.0`,
  `potential_eval_work_ratio_mean=0.27`,
  `sparse_audit_work_ratio_mean=0.07`,
  `forward_overhead_ratio_mean=0.04`,
  `priced_work_ratio_mean=0.38`,
  `projected_speedup_mean=2.6315789473684212`.
- Controls: stale potential failed with `proxy_relative_error_mean=0.28525795943754534`
  and `bellman_residual_ratio_mean=0.2989568374511133`; nonlinear dynamics
  failed with `proxy_relative_error_mean=0.17006404317002424` and
  `bellman_residual_ratio_mean=0.1295803569165177`; expensive potential head
  failed price with `priced_work_ratio_mean=0.53`.

Claim boundary: positive synthetic CPU cross-entropy potential-field adjoint
gate under a linear-quadratic value-potential identity. This is not measured
heldout CE progress, not GPU training speed, and not production v12 integration.

---

## R501-S60 - R501 Speed Paradigm 12 Sparse Orthogonal Basis Credit Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s60-sobc/latest/r501_s60_sobc_gate.json`
- `scripts/research/r501_s60_sobc_gate.py`
- `.workspace/training/rounds/r501_s60_sobc.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S60_SOBC_GATE`
- Positive condition metrics over `5` seeds:
  `active_output_count_mean=4.0`,
  `active_input_count_mean=4.0`,
  `sparse_nnz_fraction_mean=0.015625`,
  `reconstruction_error_ratio_mean=5.229198606172196e-16`,
  `reconstruction_error_ratio_max=8.122641165834063e-16`,
  `captured_energy_fraction_mean=1.0`,
  `captured_energy_fraction_min=1.0`,
  `dense_equivalent_update_error_mean=1.4710455076283324e-16`,
  `priced_work_ratio_mean=0.135625`,
  `projected_speedup_mean=7.373271889400923`.
- Controls: dense excitation failed with
  `captured_energy_fraction_mean=0.23604424543268232` and
  `reconstruction_error_ratio_mean=0.8734198874892483`; wrong basis failed
  with `captured_energy_fraction_mean=0.2732758428637095` and
  `reconstruction_error_ratio_mean=0.8519981469397067`; sparse overhead failed
  price with `sparse_nnz_fraction_mean=0.765625` and
  `priced_work_ratio_mean=0.8856249999999999`.

Claim boundary: positive synthetic CPU sparse orthogonal basis credit gate
under a certified sparse factor-basis model. This is not measured heldout CE
progress, not GPU training speed, not optimizer integration, and not production
v12 integration.

---

## R501-S79 - R501 Quality Paradigm 1 Information-Theoretic Active Sensing Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s79-information-active-sensing/latest/r501_s79_information_active_sensing_gate.json`
- `scripts/research/r501_s79_information_active_sensing_gate.py`
- `.workspace/training/rounds/r501_s79_information_active_sensing.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S79_IAS_GATE`
- Positive condition metrics over `5` seeds:
  `retained_fraction_mean=0.240234375`,
  `actionable_ce_retained_fraction_min=0.9034213201058192`,
  `actionable_ce_retained_fraction_mean=0.9067828333175896`,
  `gradient_energy_retained_fraction_min=0.9974136230683577`,
  `gradient_residual_ratio_max=0.036541777139558904`,
  `gradient_cosine_min=0.9999904755504103`,
  `state_residual_ratio_max=0.03620185417691084`,
  `lattice_noise_avoided_fraction_mean=0.759765625`,
  `masked_work_units_mean=123.0`,
  `selector_work_units_mean=30.72`,
  `priced_work_ratio_mean=0.300234375`,
  `projected_token_efficiency_multiplier_mean=3.0202498741777637`.
- Controls: weak CE proxy failed with
  `gradient_residual_ratio_mean=0.9269966417136455` and
  `state_residual_ratio_mean=0.909542762939048`; high threshold failed with
  `actionable_ce_retained_fraction_mean=0.07914888617708825`; no-sparsity
  failed price with `retained_fraction_mean=0.8203125` and
  `priced_work_ratio_mean=0.8803125000000002`; state coupling failed with
  `state_residual_ratio_mean=0.4740810678591436`; selector cost failed with
  `priced_work_ratio_mean=0.510234375`.

Claim boundary: positive synthetic CPU information-theoretic active-sensing mask
gate under a constructed CE-surprise fixture. This is not measured heldout CE
progress, not real corpus token efficiency, not GPU training speed, and not
production v12 integration.

---

## R501-S80 - R501 Quality Paradigm 2 Recurrent Target-Containing Subspace Certificate Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s80-rtcsc/latest/r501_s80_rtcsc_gate.json`
- `scripts/research/r501_s80_rtcsc_gate.py`
- `.workspace/training/rounds/r501_s80_rtcsc.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S80_RTCSC_GATE`
- Positive condition metrics over `5` seeds:
  `param_dim_mean=96.0`,
  `active_rank_mean=12.0`,
  `active_rank_fraction_mean=0.125`,
  `containment_residual_ratio_max=0.009999500037496882`,
  `continuous_loss_ratio_max=9.999000099990014e-05`,
  `ternary_realization_error_ratio_max=0.009999500037496877`,
  `ternary_loss_ratio_max=9.999000099990007e-05`,
  `descent_cosine_min=0.9999500037496873`,
  `subspace_work_units_mean=12.0`,
  `certificate_work_units_mean=7.68`,
  `priced_work_ratio_mean=0.205`,
  `projected_token_efficiency_multiplier_mean=4.877561024385367`.
- Controls: target outside span failed with
  `containment_residual_ratio_mean=0.6689647316224496` and
  `continuous_loss_ratio_mean=0.4475138121546961`; stale hidden span failed
  with `containment_residual_ratio_mean=0.9513310489928924` and
  `continuous_loss_ratio_mean=0.9052076900977533`; large rank failed price
  with `active_rank_fraction_mean=0.71875` and
  `priced_work_ratio_mean=0.7987500000000001`; ternary resolution failed with
  `ternary_realization_error_ratio_mean=0.5778384670004855` and
  `ternary_loss_ratio_mean=0.3462514150894626`; certificate cost failed with
  `priced_work_ratio_mean=0.525`.

Claim boundary: positive synthetic CPU recurrent target-containing subspace
certificate gate under a constructed hidden-span target fixture. This is not
measured heldout CE progress, not real corpus token efficiency, not GPU
training speed, and not production v12 integration.

---

## R501-S81 - R501 Quality Paradigm 3 Exact Equivalence Quotient for Contextual Synonyms Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s81-eeq-cs/latest/r501_s81_eeq_cs_gate.json`
- `scripts/research/r501_s81_eeq_cs_gate.py`
- `.workspace/training/rounds/r501_s81_eeq_cs.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S81_EEQ_CS_GATE`
- Positive condition metrics over `5` seeds:
  `token_count_mean=384.0`,
  `quotient_class_count_mean=96.0`,
  `mean_multiplicity_mean=4.0`,
  `max_hidden_radius_max=0.014454893032572712`,
  `max_cotangent_drift_ratio_max=0.02313991268178267`,
  `quotient_gradient_relative_error_max=0.007596020320777387`,
  `quotient_gradient_cosine_min=0.9999766680893215`,
  `multiplicity_scaling_error_ratio_max=0.007596020320777387`,
  `index_work_units_mean=23.04`,
  `priced_work_ratio_mean=0.31`,
  `projected_token_efficiency_multiplier_mean=3.2069002051765287`.
- Controls: false equivalence failed with
  `max_cotangent_drift_ratio_mean=0.9679441539401511` and
  `quotient_gradient_relative_error_mean=0.31040089993942677`; wide epsilon
  failed with `max_hidden_radius_mean=0.03753409400495465` and
  `quotient_gradient_relative_error_mean=0.18059290795048896`; low
  multiplicity failed price with `mean_multiplicity_mean=1.1888544891640866`
  and `priced_work_ratio_mean=0.9011458333333333`; bad multiplicity scaling
  failed with `multiplicity_scaling_error_ratio_mean=0.7500808796380061`;
  index cost failed with `priced_work_ratio_mean=0.5599999999999999`.

Claim boundary: positive synthetic CPU exact-equivalence contextual-synonym
quotient gate under a constructed hidden-state/cotangent fixture. This is not
measured heldout CE progress, not real corpus token efficiency, not GPU
training speed, and not production v12 integration.

---

## R501-S82 - R501 Quality Paradigm 4 Fisher-Weighted Ternary Commitment Escrow Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s82-fwtce/latest/r501_s82_fwtce_gate.json`
- `scripts/research/r501_s82_fwtce_gate.py`
- `.workspace/training/rounds/r501_s82_fwtce.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S82_FWTCE_GATE`
- Positive condition metrics over `5` seeds:
  `param_count_mean=512.0`,
  `proposed_flip_fraction_mean=0.4078125`,
  `locked_flip_fraction_mean=0.46217277831146675`,
  `high_fisher_flip_block_fraction_min=1.0`,
  `unsafe_high_fisher_leak_fraction_max=0.0`,
  `baseline_prior_loss_delta_mean=600.88`,
  `escrow_prior_loss_delta_mean=22.48`,
  `prior_loss_reduction_fraction_min=0.9593859929645028`,
  `new_task_progress_retention_min=0.8576524953855083`,
  `allowed_low_fisher_flip_fraction_min=1.0`,
  `ledger_work_units_mean=143.36`,
  `allowed_flip_work_units_mean=112.4`,
  `priced_work_ratio_mean=0.4995312500000001`,
  `projected_capability_retention_multiplier_mean=1.6749573205776513`.
- Controls: no Fisher signal failed with
  `high_fisher_flip_block_fraction_mean=0.34371852733554864` and
  `unsafe_high_fisher_leak_fraction_mean=0.6562814726644514`; overlock failed
  with `new_task_progress_retention_mean=0.0`; underlock failed with
  `prior_loss_reduction_fraction_mean=0.0`; stale Fisher failed with
  `prior_loss_reduction_fraction_mean=0.00983730623560164`; repair cost failed
  with `priced_work_ratio_mean=0.8495312500000001`.

Claim boundary: positive synthetic CPU Fisher-weighted ternary commitment
escrow gate under a constructed Fisher/flip fixture. This is not measured
heldout CE progress, not real corpus token efficiency, not GPU training speed,
and not production v12 integration.

---

## R501-S83 - R501 Quality Paradigm 5 Lossless Back-Translation of Residual Error Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s83-lbtre/latest/r501_s83_lbtre_gate.json`
- `scripts/research/r501_s83_lbtre_gate.py`
- `.workspace/training/rounds/r501_s83_lbtre.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S83_LBTRE_GATE`
- Positive condition metrics over `5` seeds:
  `token_count_mean=192.0`,
  `layer_count_mean=6.0`,
  `gradient_dim_mean=20.0`,
  `routing_accuracy_mean=1.0`,
  `ambiguous_drop_fraction_mean=0.0`,
  `no_signal_fraction_mean=0.0`,
  `residual_capture_fraction_mean=1.0`,
  `oracle_gradient_relative_error_mean=0.0`,
  `oracle_gradient_cosine_min=1.0000000000000002`,
  `non_bottleneck_update_energy_fraction_mean=0.0`,
  `updated_layer_fraction_mean=0.16666666666666666`,
  `detector_work_units_mean=92.16`,
  `routed_work_units_mean=192.0`,
  `priced_work_ratio_mean=0.24666666666666665`,
  `projected_token_efficiency_multiplier_mean=4.054054054054054`.
- Controls: ambiguous rank drops failed with
  `ambiguous_drop_fraction_mean=1.0` and
  `oracle_gradient_relative_error_mean=1.0`; wrong rank proxy failed with
  `routing_accuracy_mean=0.0`,
  `residual_capture_fraction_mean=0.0`, and
  `non_bottleneck_update_energy_fraction_mean=1.0`; no collapse signal failed
  with `no_signal_fraction_mean=1.0`; dense bottlenecks failed with
  `ambiguous_drop_fraction_mean=1.0` and
  `oracle_gradient_relative_error_mean=2.23606797749979`; detector cost failed
  with `priced_work_ratio_mean=0.5466666666666666`.

Claim boundary: positive synthetic CPU lossless back-translation of residual
error gate under a constructed rank-collapse fixture. This is not measured
heldout CE progress, not real corpus token efficiency, not GPU training speed,
and not production v12 integration.

---

## R501-S84 - R501 Quality Paradigm 6 Orthogonalized Credit Trajectory Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s84-oct/latest/r501_s84_oct_gate.json`
- `scripts/research/r501_s84_oct_gate.py`
- `.workspace/training/rounds/r501_s84_oct.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S84_OCT_GATE`
- Positive condition metrics over `5` seeds:
  `credit_count_mean=160.0`,
  `param_dim_mean=96.0`,
  `recent_rank_mean=8.0`,
  `new_rank_mean=6.0`,
  `prior_overlap_before_fraction_mean=0.5202119798103786`,
  `prior_overlap_after_fraction_mean=1.6399734655578546e-31`,
  `overwrite_reduction_fraction_mean=1.0`,
  `new_capability_retention_fraction_mean=1.0`,
  `future_descent_cosine_min=0.9999912422938214`,
  `oracle_relative_error_mean=4.51379222578731e-16`,
  `projection_overhead_ratio_mean=0.05`,
  `baseline_repair_units_mean=3.0808479192415142`,
  `oct_repair_units_mean=1.05`,
  `priced_work_ratio_mean=0.34099361293368663`,
  `projected_capability_efficiency_multiplier_mean=2.9341408754681084`.
- Controls: nonorthogonal target failed with
  `new_capability_retention_fraction_mean=-8.102129175917945e-17`;
  stale basis failed with `prior_overlap_after_fraction_mean=0.5202123774417069`
  and `oracle_relative_error_mean=1.0420023044017221`; no-interference failed
  with `prior_overlap_before_fraction_mean=1.5541363568916822e-32` and
  `priced_work_ratio_mean=1.05`; noisy basis failed with
  `prior_overlap_after_fraction_mean=0.9999793381258286` and
  `oracle_relative_error_mean=1.3431395791474272`; projection cost failed with
  `priced_work_ratio_mean=0.6007982704069716`.

Claim boundary: positive synthetic CPU orthogonalized credit trajectory gate
under a constructed recent-credit interference fixture. This is not measured
heldout CE progress, not real corpus token efficiency, not GPU training speed,
and not production v12 integration.

---

## R501-S85 - R501 Quality Paradigm 7 Boundary-Conditioned Token Amortization Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s85-bcta/latest/r501_s85_bcta_gate.json`
- `scripts/research/r501_s85_bcta_gate.py`
- `.workspace/training/rounds/r501_s85_bcta.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S85_BCTA_GATE`
- Positive condition metrics over `5` seeds:
  `segment_count_mean=96.0`,
  `segment_len_mean=8.0`,
  `state_dim_mean=16.0`,
  `total_token_count_mean=768.0`,
  `ce_integral_relative_error_max=1.9949559022430236e-16`,
  `boundary_gradient_relative_error_mean=0.0`,
  `boundary_gradient_cosine_min=0.9999999999999998`,
  `collapsed_token_fraction_mean=0.875`,
  `interior_noise_avoided_fraction_mean=0.875`,
  `segment_work_units_mean=96.0`,
  `verifier_work_units_mean=38.4`,
  `paid_token_ratio_mean=0.17500000000000002`,
  `projected_token_efficiency_multiplier_mean=5.7142857142857135`.
- Controls: unpredictable interior failed with
  `ce_integral_relative_error_mean=0.1275318435506661`; same-boundary interior
  mismatch failed with `ce_integral_relative_error_mean=0.15227899417695528`;
  drift outside the closed-form family failed with
  `boundary_gradient_relative_error_mean=0.40231622478517204`; short spans
  failed with `paid_token_ratio_mean=0.5499999999999999`; verifier cost failed
  with `paid_token_ratio_mean=0.5750000000000001`.

Claim boundary: positive synthetic CPU boundary-conditioned token amortization
gate under a constructed boundary-determined CE fixture. This is not measured
heldout CE progress, not real corpus token efficiency, not GPU training speed,
and not production v12 integration.

---

## R501-S86 - R501 Quality Paradigm 8 Spectral Normalization of Ternary Transitions Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s86-sntt/latest/r501_s86_sntt_gate.json`
- `scripts/research/r501_s86_sntt_gate.py`
- `.workspace/training/rounds/r501_s86_sntt.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S86_SNTT_GATE`
- Positive condition metrics over `5` seeds:
  `state_dim_mean=32.0`,
  `horizon_mean=128.0`,
  `epsilon_mean=0.002`,
  `spectral_norm_mean=0.998`,
  `min_singular_value_mean=0.9979999999999997`,
  `spectral_bound_margin_mean=0.0`,
  `long_horizon_signal_ratio_min_mean=0.7739435492434857`,
  `long_horizon_signal_ratio_mean_mean=0.7739435492434861`,
  `transient_amplification_max_mean=0.9980000000000002`,
  `paid_work_ratio_mean=0.0878125`,
  `projected_long_context_multiplier_mean=8.813592019854646`.
- Controls: unnormalized transition failed with
  `spectral_norm_mean=1.035` and
  `long_horizon_signal_ratio_mean_mean=81.72685249859066`; overcontraction
  failed with `min_singular_value_mean=0.9399999999999997` and
  `long_horizon_signal_ratio_min_mean=0.0003633835478324453`; non-normal
  transient failed with `spectral_norm_mean=1.4484172230125103`; stale
  certificate failed with `spectral_norm_mean=1.035`; certificate cost failed
  with `paid_work_ratio_mean=0.7578125`.

Claim boundary: positive synthetic CPU spectral normalization of ternary
transitions gate under a scaled signed-permutation fixture. This is not
measured heldout CE progress, not real corpus token efficiency, not GPU
training speed, and not production v12 integration.

---

## R501-S87 - R501 Quality Paradigm 9 Cotangent-Guided Token Resampling Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s87-cgtr/latest/r501_s87_cgtr_gate.json`
- `scripts/research/r501_s87_cgtr_gate.py`
- `.workspace/training/rounds/r501_s87_cgtr.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S87_CGTR_GATE`
- Positive condition metrics over `5` seeds:
  `token_count_mean=2048.0`,
  `gradient_dim_mean=24.0`,
  `actionable_fraction_mean=0.25`,
  `eligible_fraction_mean=0.25`,
  `expected_sample_fraction_mean=0.13125253501511496`,
  `actionable_recall_mean=1.0`,
  `low_signal_skip_fraction_mean=1.0`,
  `full_gradient_relative_error_mean=0.0012000383068875074`,
  `actionable_gradient_relative_error_mean=0.0`,
  `gradient_cosine_to_full_min=0.9999991045532026`,
  `importance_bias_ratio_mean=0.46863628116718087`,
  `paid_token_ratio_mean=0.21125253501511496`,
  `projected_token_efficiency_multiplier_mean=4.733689151841645`.
- Controls: weak proxy failed with `actionable_recall_mean=0.286328125` and
  `full_gradient_relative_error_mean=0.7147411413038028`; coherent low-signal
  tokens failed with `full_gradient_relative_error_mean=0.43642362030132154`;
  missing importance weighting failed with
  `actionable_gradient_relative_error_mean=0.46863628116718087`; too-high
  threshold failed with `actionable_recall_mean=0.5046875`; proxy cost failed
  with `paid_token_ratio_mean=0.5812525350151151`.

Claim boundary: positive synthetic CPU cotangent-guided token resampling gate
under a constructed proxy/actionable-gradient fixture. This is not measured
heldout CE progress, not real corpus token efficiency, not GPU training speed,
and not production v12 integration.

---

## R501-S88 - R501 Quality Paradigm 10 Ternary Covariance Factorization Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s88-tcf/latest/r501_s88_tcf_gate.json`
- `scripts/research/r501_s88_tcf_gate.py`
- `.workspace/training/rounds/r501_s88_tcf.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S88_TCF_GATE`
- Positive condition metrics over `5` seeds:
  `dim_mean=48.0`,
  `token_count_mean=4096.0`,
  `true_rank_mean=8.0`,
  `used_rank_mean=8.0`,
  `active_per_factor_mean=5.0`,
  `covariance_relative_error_mean=0.02219488715547841`,
  `covariance_relative_error_max=0.028231534523657707`,
  `dense_sample_relative_error_mean=0.04270130472601711`,
  `covariance_noise_reduction_fraction_mean=0.477338112473781`,
  `covariance_noise_reduction_fraction_min=0.37628316412629303`,
  `energy_capture_fraction_mean=0.9991609882606131`,
  `energy_capture_fraction_min=0.9940053836564438`,
  `basis_alignment_min_min=0.9999999999999999`,
  `ternary_pattern_violation_count_max=0.0`,
  `coherent_residual_fraction_mean=0.0008889896404858218`,
  `active_update_fraction_mean=0.08680555555555555`,
  `paid_work_ratio_mean=0.16680555555555554`,
  `projected_covariance_efficiency_multiplier_mean=5.995004163197336`.
- Controls: random basis failed with
  `covariance_relative_error_mean=0.9862661670354609`; non-ternary basis
  failed with `ternary_pattern_violation_count_max=384.0`; rank underfit
  failed with `covariance_relative_error_mean=0.511610002163526`; dense
  residual failed with `coherent_residual_fraction_mean=0.11233310244626468`
  and `covariance_noise_reduction_fraction_mean=-1.626120847352616`; dense
  cost failed with `paid_work_ratio_mean=1.0553819444444446`.

Claim boundary: positive synthetic CPU ternary covariance factorization gate
under a constructed sparse ternary covariance fixture. This is not measured
heldout CE progress, not real corpus token efficiency, not GPU training speed,
and not production v12 integration.

---

## R501-S89 - R501 Quality Paradigm 11 Certified Rare-Token Escrow Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s89-crte/latest/r501_s89_crte_gate.json`
- `scripts/research/r501_s89_crte_gate.py`
- `.workspace/training/rounds/r501_s89_crte.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S89_CRTE_GATE`
- Positive condition metrics over `5` seeds:
  `batch_count_mean=64.0`,
  `batch_size_mean=32.0`,
  `rare_token_count_mean=64.0`,
  `common_token_count_mean=1984.0`,
  `rare_token_recall_mean=1.0`,
  `false_escrow_fraction_mean=0.0`,
  `release_precision_mean=1.0`,
  `stale_release_harm_fraction_mean=0.0`,
  `baseline_unique_capability_fraction_mean=0.125`,
  `escrow_unique_capability_fraction_mean=1.0`,
  `unique_capability_gain_mean=8.0`,
  `majority_retention_fraction_mean=1.0`,
  `escrow_queue_fraction_mean=0.03125`,
  `paid_work_ratio_mean=0.11125`,
  `projected_unique_capability_multiplier_mean=8.98876404494382`.
- Controls: no escrow failed with `rare_token_recall_mean=0.0`; premature
  release failed with `release_precision_mean=0.0`; stale geometry failed with
  `release_precision_mean=0.25`; low-CE false escrow failed with
  `false_escrow_fraction_mean=0.9676994364265445` and
  `majority_retention_fraction_mean=0.03230056357345543`; escrow cost failed
  with `paid_work_ratio_mean=0.48125`.

Claim boundary: positive synthetic CPU certified rare-token escrow gate under a
constructed high-CE/conflicting-batch rare-token fixture. This is not measured
heldout CE progress, not real corpus token efficiency, not GPU training speed,
and not production v12 integration.

---

## R501-S90 - R501 Quality Paradigm 12 Harmonic Credit Assignment Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s90-hca/latest/r501_s90_hca_gate.json`
- `scripts/research/r501_s90_hca_gate.py`
- `.workspace/training/rounds/r501_s90_hca.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S90_HCA_GATE`
- Positive condition metrics over `5` seeds:
  `sequence_len_mean=256.0`,
  `feature_dim_mean=16.0`,
  `horizon_mean=255.0`,
  `long_lag_mean=128.0`,
  `state_relative_error_mean_mean=0.0`,
  `state_relative_error_max_max=0.0`,
  `quotient_weight_residual_mean=0.0`,
  `long_lag_retention_fraction_mean=1.0`,
  `tail_credit_capture_fraction_mean=1.0`,
  `near_credit_distortion_mean=0.0`,
  `lag_order_error_mean_mean=0.0`,
  `paid_work_ratio_mean=0.11137254901960784`,
  `projected_long_context_multiplier_mean=8.97887323943662`.
- Controls: exponential decay failed with
  `long_lag_retention_fraction_mean=0.049482015023992576` and
  `tail_credit_capture_fraction_mean=0.009265704589093256`; truncated tail
  failed with `long_lag_retention_fraction_mean=0.0`; wrong kernel failed with
  `quotient_weight_residual_mean=0.44106746731102414`; shuffled lag failed
  with `lag_order_error_mean_mean=1.3675555337878922`; quotient cost failed
  with `paid_work_ratio_mean=0.5813725490196079`.

Claim boundary: positive synthetic CPU harmonic credit assignment gate under a
constructed harmonic recurrent-feature fixture. This is not measured heldout CE
progress, not real corpus token efficiency, not GPU training speed, and not
production v12 integration.

---

## R501-S91 - R501 Quality Paradigm 13 Segment-Level Adjoint Replay Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s91-slar/latest/r501_s91_slar_gate.json`
- `scripts/research/r501_s91_slar_gate.py`
- `.workspace/training/rounds/r501_s91_slar.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S91_SLAR_GATE`
- Positive condition metrics over `5` seeds:
  `segment_count_mean=64.0`,
  `segment_len_mean=8.0`,
  `adjoint_dim_mean=32.0`,
  `tube_rank_mean=4.0`,
  `sparse_residual_count_mean=2.0`,
  `segment_adjoint_relative_error_mean_mean=2.510380145582369e-16`,
  `segment_adjoint_relative_error_max_max=6.304778217674318e-16`,
  `stepwise_relative_error_mean_mean=0.7217953104709156`,
  `noise_reduction_fraction_mean=0.9999999999999998`,
  `tube_energy_capture_fraction_mean=0.9242694120496534`,
  `sparse_residual_recall_mean=1.0`,
  `residual_debt_ratio_mean=2.510380145582369e-16`,
  `harmless_residual_bound_violation_fraction_mean=0.0`,
  `paid_token_ratio_mean=0.175`,
  `projected_segment_efficiency_multiplier_mean=5.714285714285714`.
- Controls: stepwise replay failed with
  `segment_adjoint_relative_error_mean_mean=0.7217953104709156`; missing
  sparse residual failed with `residual_debt_ratio_mean=0.25703841461578136`;
  wrong tube failed with `tube_energy_capture_fraction_mean=0.13337524404076379`;
  incoherent segment failed with
  `harmless_residual_bound_violation_fraction_mean=1.0`; replay cost failed
  with `paid_token_ratio_mean=0.575`.

Claim boundary: positive synthetic CPU segment-level adjoint replay gate under a
constructed tube-plus-sparse-residual segment fixture. This is not measured
heldout CE progress, not real corpus token efficiency, not GPU training speed,
and not production v12 integration.

---

## R501-S92 - R501 Quality Paradigm 14 Cross-Sample State-Conditioned CE Quotient Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s92-cs-sccq/latest/r501_s92_cs_sccq_gate.json`
- `scripts/research/r501_s92_cs_sccq_gate.py`
- `.workspace/training/rounds/r501_s92_cs_sccq.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S92_CS_SCCQ_GATE`
- Positive condition metrics over `5` seeds:
  `class_count_mean=64.0`,
  `mean_multiplicity_mean=8.0`,
  `max_state_radius_max=0.018672448437046414`,
  `max_gradient_drift_ratio_max=0.06144363574772148`,
  `class_residual_ratio_max=0.028131467885392752`,
  `quotient_gradient_relative_error_max=1.2964624928340724e-15`,
  `quotient_gradient_cosine_min=0.9999999999999997`,
  `multiplicity_scaling_error_ratio_max=0.0`,
  `representative_fraction_mean=0.125`,
  `residual_certificate_ratio_mean=0.04`,
  `index_overhead_ratio_mean=0.05`,
  `paid_token_ratio_mean=0.21500000000000002`,
  `projected_token_efficiency_multiplier_mean=4.651162790697674`.
- Controls: false state match failed with
  `max_gradient_drift_ratio_max=2.241165336397183`; missing multiplicity failed
  with `quotient_gradient_relative_error_max=0.8806075389805172` and
  `multiplicity_scaling_error_ratio_max=0.875`; residual ignored failed with
  `class_residual_ratio_max=0.41845409737469763` and
  `quotient_gradient_relative_error_max=0.5318847849747554`; low multiplicity
  failed price with `paid_token_ratio_mean=1.09`; expensive index failed price
  with `paid_token_ratio_mean=0.615`.

Claim boundary: positive synthetic CPU cross-sample state-conditioned CE
quotient gate under a constructed repeated-state/CE-gradient fixture. This is
not measured heldout CE progress, not real corpus token efficiency, not GPU
training speed, and not production v12 integration.

---

## R501-S93 - R501 Quality Paradigm 15 Singular-Value Directed Ternary Bit-Flips Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s93-svd-tbf/latest/r501_s93_svd_tbf_gate.json`
- `scripts/research/r501_s93_svd_tbf_gate.py`
- `.workspace/training/rounds/r501_s93_svd_tbf.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S93_SVD_TBF_GATE`
- Positive condition metrics over `5` seeds:
  `top_singular_energy_fraction_min=0.7362718188955173`,
  `singular_gap_ratio_min=9.550897305326538`,
  `top_vector_alignment_min=0.9995024837617101`,
  `ternary_flip_alignment_min=1.0000000000000002`,
  `capability_capture_fraction_min=1.0000000000000004`,
  `raw_flip_capture_fraction_mean=0.01944444444444445`,
  `candidate_vs_raw_capture_gain_min=16.0`,
  `sign_mismatch_fraction_max=0.0`,
  `ternary_violation_count_max=0.0`,
  `selected_support_fraction_mean=0.1875`,
  `paid_work_ratio_mean=0.2675`,
  `projected_flip_efficiency_multiplier_mean=3.7383177570093458`.
- Controls: raw element-wise flips failed with
  `capability_capture_fraction_min=0.0`; random direction failed with
  `capability_capture_fraction_min=0.0`; low singular gap failed with
  `top_singular_energy_fraction_min=0.4378506776751088` and
  `singular_gap_ratio_min=1.0325787352311167`; nonternary direction failed with
  `ternary_violation_count_max=64.0`; expensive SVD/dense support failed price
  with `paid_work_ratio_mean=1.325`.

Claim boundary: positive synthetic CPU singular-value directed ternary bit-flip
gate under a constructed zero-mean/high-variance token-gradient fixture. This is
not measured heldout CE progress, not real corpus token efficiency, not GPU
training speed, and not production v12 integration.

---

## R501-S94 - R501 Hardware Efficiency Math 1 Sherman-Morrison Spectral-Partitioned Recurrent Inversion Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s94-sm-spri/latest/r501_s94_sm_spri_gate.json`
- `scripts/research/r501_s94_sm_spri_gate.py`
- `.workspace/training/rounds/r501_s94_sm_spri.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S94_SM_SPRI_GATE`
- Positive condition metrics over `5` seeds:
  `sequence_len_mean=256.0`,
  `state_dim_mean=64.0`,
  `stable_dim_mean=58.0`,
  `singular_dim_mean=6.0`,
  `singular_escrow_recall_min=1.0`,
  `reconstruction_relative_error_max=4.896888604437808e-16`,
  `reconstruction_max_abs_error_max=6.94999613415348e-14`,
  `recurrent_gradient_relative_error_max=5.164237515100902e-16`,
  `stable_inverse_condition_proxy_max=1.0100676788061431`,
  `stored_state_ratio_mean=0.09765625`,
  `repair_overhead_ratio_mean=0.04`,
  `paid_vram_ratio_mean=0.13765625`,
  `projected_vram_efficiency_multiplier_mean=7.264472190692395`.
- Controls: no escrow failed with `singular_escrow_recall_min=0.0` and
  `reconstruction_relative_error_max=0.02845650435038637`; wrong partition
  failed with `singular_escrow_recall_min=0.5` and
  `reconstruction_relative_error_max=0.01993420300789507`; quantized escrow
  failed with `reconstruction_relative_error_max=0.01668452591178294`; dense
  residency failed price with `paid_vram_ratio_mean=1.04390625`; high repair
  cost failed price with `paid_vram_ratio_mean=0.54765625`.

Claim boundary: positive synthetic CPU spectral-partitioned recurrent inversion
gate under a constructed near-isometric/singular recurrent fixture. This is not
measured heldout CE progress, not real VRAM telemetry, not GPU training speed,
and not production v12 integration.

---

## R501-S95 - R501 Hardware Efficiency Math 2 Fused Logit-Cotangent Register Accumulation Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s95-flcra/latest/r501_s95_flcra_gate.json`
- `scripts/research/r501_s95_flcra_gate.py`
- `.workspace/training/rounds/r501_s95_flcra.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S95_FLCRA_GATE`
- Positive condition metrics over `5` seeds:
  `sequence_len_mean=128.0`,
  `vocab_dim_mean=2048.0`,
  `hidden_dim_mean=64.0`,
  `dense_logit_bytes_fp16_mean=524288.0`,
  `candidate_register_elements_mean=2048.0`,
  `cotangent_relative_error_max=8.90412630045184e-16`,
  `cotangent_max_abs_error_max=1.3322676295501878e-15`,
  `cotangent_cosine_min=1.0`,
  `probability_normalization_error_max=4.440892098500626e-16`,
  `target_probability_error_max=2.168404344971009e-19`,
  `compute_work_ratio_mean=1.0`,
  `paid_logit_vram_ratio_mean=0.0078125`,
  `projected_logit_vram_efficiency_multiplier_mean=128.0`,
  `projected_dense_logit_mib_4096x32768_fp16_mean=256.0`,
  `projected_saved_mib_4096x32768_fp16_mean=255.9375`.
- Controls: raw logits failed with `cotangent_relative_error_max=73.82597652065176`
  and `probability_normalization_error_max=42.50344891437952`; missing target
  failed with `cotangent_relative_error_max=1.0010000605184466`; sampled vocab
  failed with `cotangent_relative_error_max=0.06026649314478231` and
  `target_probability_error_max=0.004610016760508319`; materialized logits
  failed price with `paid_logit_vram_ratio_mean=1.0`; extra compute failed with
  `compute_work_ratio_mean=2.0`.

Claim boundary: positive synthetic CPU fused logit-cotangent register
accumulation gate under a constructed dense-vocabulary CE fixture. This is not
measured heldout CE progress, not real VRAM telemetry, not GPU training speed,
and not production v12 integration.

---

## R501-S96 - R501 Hardware Efficiency Math 3 Boundary-Sparse Activation Escrow Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s96-bsae/latest/r501_s96_bsae_gate.json`
- `scripts/research/r501_s96_bsae_gate.py`
- `.workspace/training/rounds/r501_s96_bsae.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S96_BSAE_GATE`
- Positive condition metrics over `5` seeds:
  `sequence_len_mean=256.0`,
  `state_dim_mean=64.0`,
  `element_count_mean=16384.0`,
  `active_fraction_mean=0.12059326171875`,
  `candidate_escrow_fraction_mean=0.12059326171875`,
  `active_recall_min=1.0`,
  `inactive_rejection_min=1.0`,
  `gradient_relative_error_max=0.0`,
  `gradient_max_abs_error_max=0.0`,
  `dense_activation_bits_mean=262144.0`,
  `candidate_activation_bits_mean=47996.8`,
  `paid_activation_bit_ratio_mean=0.18309326171875`,
  `projected_activation_efficiency_multiplier_mean=5.462958263652707`.
- Controls: no escrow failed with `gradient_relative_error_max=1.0`;
  inverted boundary failed with `active_recall_min=0.0`,
  `inactive_rejection_min=0.0`, and `paid_activation_bit_ratio_mean=0.94190673828125`;
  quantized magnitude failed with `gradient_relative_error_max=0.14020056200808534`;
  dense activation failed price with `paid_activation_bit_ratio_mean=1.0625`;
  random mask failed with `active_recall_min=0.11907270811380401` and
  `gradient_relative_error_max=0.9552600024304763`.

Claim boundary: positive synthetic CPU boundary-sparse activation escrow gate
under a constructed ternary-boundary structural-gradient fixture. This is not
measured heldout CE progress, not real VRAM telemetry, not GPU training speed,
and not production v12 integration.

---

## R501-S97 - R501 Hardware Efficiency Math 4 Ternary-Lattice Shadow Weight Virtualization Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s97-tlswv/latest/r501_s97_tlswv_gate.json`
- `scripts/research/r501_s97_tlswv_gate.py`
- `.workspace/training/rounds/r501_s97_tlswv.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S97_TLSWV_GATE`
- Positive condition metrics over `5` seeds:
  `element_count_mean=65536.0`,
  `stable_fraction_mean=0.819879150390625`,
  `low_margin_fraction_mean=0.090057373046875`,
  `impulse_fraction_mean=0.02459716796875`,
  `virtualized_fraction_mean=0.795281982421875`,
  `dense_tracking_fraction_mean=0.204718017578125`,
  `impulse_detector_recall_min=1.0`,
  `unsafe_virtualization_rate_max=0.0`,
  `ternary_mismatch_rate_max=0.0`,
  `ternary_match_fraction_min=1.0`,
  `paid_shadow_bit_ratio_mean=0.2544231414794922`,
  `projected_shadow_efficiency_multiplier_mean=3.9306759146169035`.
- Controls: no impulse detector failed with `impulse_detector_recall_min=0.0`
  and `ternary_mismatch_rate_max=0.0246734619140625`; overvirtualized unstable
  failed with `ternary_mismatch_rate_max=0.115692138671875`; dense shadow
  failed price with `paid_shadow_bit_ratio_mean=1.0`; signless virtual failed
  with `ternary_mismatch_rate_max=0.4010009765625`; loose margin failed with
  `ternary_mismatch_rate_max=0.115692138671875`.

Claim boundary: positive synthetic CPU ternary-lattice shadow weight
virtualization gate under a constructed saturated/low-margin/impulse shadow
fixture. This is not measured heldout CE progress, not real VRAM telemetry, not
GPU training speed, and not production v12 integration.

---

## R501-S98 - R501 Hardware Efficiency Math 5 Unbiased Stochastic-Quantized Recurrent Buffer Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s98-usqrb/latest/r501_s98_usqrb_gate.json`
- `scripts/research/r501_s98_usqrb_gate.py`
- `.workspace/training/rounds/r501_s98_usqrb.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S98_USQRB_GATE`
- Positive condition metrics over `5` seeds:
  `sequence_len_mean=512.0`,
  `state_dim_mean=64.0`,
  `output_dim_mean=32.0`,
  `sample_count_mean=128.0`,
  `expectation_gradient_relative_error_max=0.0`,
  `expectation_gradient_max_abs_error_max=0.0`,
  `quantizer_state_bias_relative_error_max=0.0`,
  `monte_carlo_gradient_relative_error_max=0.02481682775488569`,
  `dense_buffer_bits_mean=524288.0`,
  `candidate_buffer_bits_mean=131072.0`,
  `paid_buffer_bit_ratio_mean=0.25`,
  `projected_buffer_efficiency_multiplier_mean=4.0`.
- Controls: deterministic floor failed with
  `expectation_gradient_relative_error_max=0.33109484490725305`;
  deterministic nearest failed with `expectation_gradient_relative_error_max=0.1974591776455725`;
  signless magnitude failed with `expectation_gradient_relative_error_max=1.460369365122468`;
  dense FP16 failed price with `paid_buffer_bit_ratio_mean=1.0`; clipped range
  failed with `expectation_gradient_relative_error_max=0.455478607148699`.

Claim boundary: positive synthetic CPU unbiased stochastic-quantized recurrent
buffer gate under a constructed projection-gradient fixture. This is not
measured heldout CE progress, not real VRAM telemetry, not GPU training speed,
and not production v12 integration.

---

## R501-S99 - R501 Hardware Efficiency Math 6 Sub-Warp Recurrent Checkpoint Resampling Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s99-swrcr/latest/r501_s99_swrcr_gate.json`
- `scripts/research/r501_s99_swrcr_gate.py`
- `.workspace/training/rounds/r501_s99_swrcr.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S99_SWRCR_GATE`
- Positive condition metrics over `5` seeds:
  `sequence_len_mean=512.0`,
  `state_dim_mean=32.0`,
  `input_dim_mean=16.0`,
  `checkpoint_interval_mean=8.0`,
  `checkpoint_count_mean=65.0`,
  `reconstruction_relative_error_max=0.0`,
  `reconstruction_max_abs_error_max=0.0`,
  `checkpoint_state_ratio_mean=0.1267056530214425`,
  `hidden_latency_ratio_mean=0.14285714285714285`,
  `register_window_fit_ratio_mean=0.5`,
  `paid_state_bit_ratio_mean=0.1267056530214425`,
  `projected_state_efficiency_multiplier_mean=7.892307692307694`.
- Controls: corrupted checkpoint failed with `reconstruction_relative_error_max=0.0015636503369558389`;
  approximate recompute failed with `reconstruction_relative_error_max=0.06437215157278309`;
  dense residency failed price with `paid_state_bit_ratio_mean=1.0`;
  non-ternary compute failed with `hidden_latency_ratio_mean=1.7142857142857142`;
  oversized interval failed with `register_window_fit_ratio_mean=4.0`.

Claim boundary: positive synthetic CPU sub-warp recurrent checkpoint resampling
gate under a constructed ternary recurrent SSM fixture. This is not measured
heldout CE progress, not real VRAM telemetry, not GPU training speed, and not
production v12 integration.

---

## R501-S100 - R501 Hardware Efficiency Math 7 Exact-Quotient Padding Elimination Block Gate

Date: 2026-06-07

Source:

- `.work/jobs/r501-s100-eqpeb/latest/r501_s100_eqpeb_gate.json`
- `scripts/research/r501_s100_eqpeb_gate.py`
- `.workspace/training/rounds/r501_s100_eqpeb.ps1`
- `.workspace/research/r501-extraordinary-speed-quality-efficiency/implementation-log.md`

Result:

- Diagnostic decision:
  `PASS_R501_S100_EQPEB_GATE`
- Positive condition metrics over `5` seeds:
  `batch_size_mean=32.0`,
  `max_sequence_len_mean=64.0`,
  `input_dim_mean=8.0`,
  `state_dim_mean=16.0`,
  `active_token_count_mean=1429.4`,
  `packed_token_count_mean=1429.4`,
  `active_token_fraction_mean=0.69794921875`,
  `active_token_recall_min=1.0`,
  `forward_relative_error_max=0.0`,
  `forward_max_abs_error_max=0.0`,
  `gradient_relative_error_max=0.0`,
  `gradient_max_abs_error_max=0.0`,
  `paid_matrix_ratio_mean=0.69794921875`,
  `projected_padding_efficiency_multiplier_mean=1.437287957139846`.
- Controls: no-reset failed forward/gradient exactness; wrong-unpack failed
  forward/gradient exactness; drop-active failed active-token recall or
  forward/gradient coverage; dense-padding and include-padding failed the
  padding price check with padded-token residency.

Claim boundary: positive synthetic CPU exact-quotient padding elimination gate
under a constructed variable-length recurrent SSM fixture. This is not measured
heldout CE progress, not real VRAM telemetry, not GPU training speed, and not
production v12 integration.

Real-path addendum (R501-S100, 2026-06-07):

- Production integration: opt-in v12 flags `--exact-quotient-padding` /
  `--eqp`, `--eqp-min-active-len`, and `--eqp-seed`.
- Mechanism integrated: sampled-CE GPU row masking over real training
  microbatches. When active, loss is averaged over active-prefix targets,
  inactive `ce_d_logits` rows are zeroed, and surviving rows are rescaled by
  `n_valid/S` before CE scatters consume them.
- Build: `mingw32-make USE_VULKAN=1 train` passed.
- Clean activation smoke:
  `.work/jobs/r501-eqpeb-realpath/eqp-zero-20260607-141018/`,
  `EXIT_CODE=0`, RTX 5070 initialized, `seq_len=1024`,
  `min_active_len=768`, GPU allocated `217.34 MB`, and `[TRAIN] Complete`.
- CE-path smoke:
  `.work/jobs/r501-eqpeb-realpath/eqp-r493off-20260607-140955/` reached
  `[EXACT_QUOTIENT_PADDING_CE] backend=gpu` in four real grad-accum
  microbatches with `1023` targets each:
  `active=993 inactive=30 active_fraction=0.970675`,
  `active=920 inactive=103 active_fraction=0.899316`,
  `active=921 inactive=102 active_fraction=0.900293`,
  and `active=949 inactive=74 active_fraction=0.927664`.
- Blocker boundary: the one-step preset exits with `-1073741819` even without
  EQP and even with `--r493-post-step-off`, so completion failure is not
  attributed to the EQP CE integration. No speedup, heldout CE progress, VRAM
  reduction, or packed-scan compute-skip result is claimed.
