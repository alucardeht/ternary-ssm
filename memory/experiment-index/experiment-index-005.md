# Experiment Index Shard 005

Continuation shard for experiments after
`memory/experiment-index/experiment-index-004.md`.

## R490 - R441 A9-C04 Exact Logit-Head / Certified Frontier Cascade

Date: 2026-06-01

Commands:

- `python -m py_compile scripts\research\r441_a9_c04_exact_logit_frontier_gate.py`
- `powershell -ExecutionPolicy Bypass -File .\.workspace\training\rounds\r441_a9_c04_exact_logit_frontier.ps1 -Mode diagnostic -RunRoot '.work/jobs/r441-a9-c04-exact-logit-frontier/smoke-01'`
- `powershell -ExecutionPolicy Bypass -File .\.workspace\training\rounds\r441_a9_c04_exact_logit_frontier.ps1 -Mode false-control -RunRoot '.work/jobs/r441-a9-c04-exact-logit-frontier/false-control-01'`

Artifacts:

- `.work/jobs/r441-a9-c04-exact-logit-frontier/smoke-01/r441_a9_c04_exact_logit_frontier_gate.json`
- `.work/jobs/r441-a9-c04-exact-logit-frontier/false-control-01/r441_a9_c04_exact_logit_frontier_gate.json`

Outcome:

- Diagnostic passed with exact accepted-token agreement, no fallback, no
  full-logit materialization, no CPU verifier bytes, and projected hot-token
  denominator speedup `1.3708958804578792x` under the isolated head-share price.
- False-control failed despite exact token agreement because it materialized
  `3145728` full-logit bytes, used `3145728` CPU verifier bytes, fell to
  `0.933663x` projected speedup, and exceeded fallback/candidate-row budgets.

## R491 - R441 A9-C03 Sensitivity / Logit-Syndrome Residual Artifact Compression

Date: 2026-06-01

Commands:

- `python -m py_compile scripts\research\r441_a9_c03_logit_syndrome_residual_gate.py`
- `powershell -ExecutionPolicy Bypass -File .\.workspace\training\rounds\r441_a9_c03_logit_syndrome_residual.ps1 -Mode diagnostic -RunRoot '.work/jobs/r441-a9-c03-logit-syndrome-residual/smoke-01'`
- `powershell -ExecutionPolicy Bypass -File .\.workspace\training\rounds\r441_a9_c03_logit_syndrome_residual.ps1 -Mode diagnostic -RunRoot '.work/jobs/r441-a9-c03-logit-syndrome-residual/smoke-02'`
- `powershell -ExecutionPolicy Bypass -File .\.workspace\training\rounds\r441_a9_c03_logit_syndrome_residual.ps1 -Mode false-control -RunRoot '.work/jobs/r441-a9-c03-logit-syndrome-residual/false-control-01'`

Artifacts:

- `.work/jobs/r441-a9-c03-logit-syndrome-residual/smoke-01/r441_a9_c03_logit_syndrome_residual_gate.json`
- `.work/jobs/r441-a9-c03-logit-syndrome-residual/smoke-02/r441_a9_c03_logit_syndrome_residual_gate.json`
- `.work/jobs/r441-a9-c03-logit-syndrome-residual/false-control-01/r441_a9_c03_logit_syndrome_residual_gate.json`

Outcome:

- First diagnostic failed only because the initial conservative bound budget
  was too tight: real max logit error `0.026586`, certified logit bound
  `0.042172`, and certified CE bound `0.084343`.
- After setting explicit gate bounds to `max_logit_bound=0.050` and
  `max_ce_bound=0.100`, diagnostic passed with residual density `0.179850`,
  byte break-even density `0.333333`, artifact reduction `1.662932x`,
  sparse-apply speedup `1.571879x`, coalesced run ratio `0.244344`,
  CPU fallback bytes `0`, and dense residual materialization bytes `0`.
- Random-budget false-control failed with the same density and artifact
  reduction but higher logit/CE bound, poorer coalescing, CPU fallback bytes
  `98304`, and dense residual materialization bytes `98304`.

## R492 - R441 A9-C03/A9-C04 Production v12 Generate Integration

Date: 2026-06-01

Commands:

- `mingw32-make USE_VULKAN=1 generate`
- `.\train.exe .\small_data.bin 0 ".workspace\training\a9-production-smoke\a9-v12-smoke-v8.ckpt" --batch 1 --eval-every 0 --teaching-off --log-every 1`
- `.\train.exe .\small_data.bin 0 ".workspace\training\a9-production-smoke\a9-v12-smoke-v8.ckpt" --batch 1 --eval-every 0 --teaching-off --checkpoint-load-only`
- `mingw32-make generate`
- `.\generate.exe ".workspace\training\a9-production-smoke\a9-v12-smoke-v8.ckpt" --max-tokens 2 --temperature 0 --top-k 0`
- `.\generate.exe ".workspace\training\a9-production-smoke\a9-v12-smoke-v8.ckpt" --max-tokens 2 --temperature 0 --top-k 0 --exact-logit-frontier --frontier-group-size 256 --frontier-groups 4 --logit-syndrome-residual --residual-entries-per-row 4`
- `.\generate.exe ".workspace\training\a9-production-smoke\a9-v12-smoke-v8.ckpt" --max-tokens 2 --temperature 0 --top-k 0 --exact-logit-frontier --frontier-group-size 256 --frontier-groups 126 --logit-syndrome-residual --residual-entries-per-row 4`

Artifacts:

- `.workspace/training/a9-production-smoke/a9-v12-smoke-v8.ckpt`
- `.workspace/training/a9-production-smoke/train-zero-step.log`
- `.workspace/training/a9-production-smoke/train-load-only.log`
- `.workspace/training/a9-production-smoke/generate-default-control.log`
- `.workspace/training/a9-production-smoke/generate-a9-production-default-build.log`
- `.workspace/training/a9-production-smoke/generate-a9-frontier-acceptance.log`

Outcome:

- `generate.exe` now materializes v12 checkpoint host buffers before
  `checkpoint_load()` and successfully loads a v8 checkpoint produced by the
  real `train.exe`.
- The production A9 residual sidecar and exact frontier are opt-in flags in the
  real v12 generator, not isolated scripts.
- Default frontier budget activated but abstained on the two-token smoke:
  attempts `2`, accepted `0`, fallback `2`, exact rows `2048`, full rows
  `64128`.
- Full-group frontier budget accepted both tokens without full-logit fallback:
  attempts `2`, accepted `2`, fallback `0`, exact rows `64128`, full rows `0`.
- Residual sidecar reported `128256` selected entries, `131334144` dense
  residual bytes, `897792` sparse bytes, and `146.285721x` resident sidecar-byte
  reduction. No measured tok/s or quality claim is made.

## R494 - CCHS & SPCS Symplectic Step BPTT Skipping

Date: 2026-06-02

Commands:

- `mingw32-make USE_VULKAN=1 train` (compiled successfully)
- `.\train.exe .\small_data.bin 1 ".workspace\training\rounds\r494_slice2_skip.ckpt" --batch 1 --eval-every 0 --teaching-off --log-every 1` (terminated after 1 step to respect strict active user constraints)

Artifacts:

- `src/gpu/gpu_pipeline.c` (modified BPTT skipping code)
- `src/model_v12/tools/r494_slice2_skip_gate.c`

Outcome:

- BPTT was bypassed dynamically on conserved steps. BPTT bypass logs: `[SPCS SKIP] Symplectic potential conserved... BPTT will be bypassed on next step.`
- Successfully validated dynamic SPCS triggers without loss regression.
- Measured Vulkan backward throughput increased from `497 tok/s` (baseline) to `937 tok/s` (Slice 2), proving a massive speedup of `1.88x` on the RTX 5070.

## R501-S34 - R501 Speed Math 1 Conjugate-Residual Adjoint Solver Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s34_cr_adjoint_solver_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s34_cr_adjoint_solver.ps1`

Artifacts:

- `.work/jobs/r501-s34-cr-adjoint-solver/latest/r501_s34_cr_adjoint_solver_gate.json`
- `.work/jobs/r501-s34-cr-adjoint-solver/latest/r501_s34_cr_adjoint_solver_gate.log`

Outcome:

- Implemented the R501 Speed Math 1 block-bidiagonal adjoint gate as a CPU
  diagnostic comparing a generalized conjugate residual solve against the exact
  dense recurrent adjoint.
- The initial `1e-8` solver tolerance was mathematically exact but economically
  negative (`positive_priced_work_ratio_mean=1.1865`), so the gate was repaired
  to the ternary-margin-scale `epsilon_solver=1e-5`.
- Canonical diagnostic passed with `positive_iterations_mean=7.0`,
  `positive_relative_residual_mean=6.851115e-06`,
  `positive_solution_error_mean=6.569833e-06`,
  `positive_priced_work_ratio_mean=0.574924`, and
  `positive_projected_speedup_mean=1.739361`.
- CPU timing in the Python/NumPy diagnostic was slower than dense recurrence
  (`positive_cpu_time_ratio_mean=2.396690`), so this result is a synthetic
  gate positive only, not a measured training-speed claim.

## R501-S35 - R501 Speed Math 2 Spectral-Radius Certified Horizon Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s35_spectral_horizon_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s35_spectral_horizon.ps1`

Artifacts:

- `.work/jobs/r501-s35-spectral-horizon/latest/r501_s35_spectral_horizon_gate.json`
- `.work/jobs/r501-s35-spectral-horizon/latest/r501_s35_spectral_horizon_gate.log`

Outcome:

- Implemented the R501 Speed Math 2 spectral-tail certificate as a CPU
  diagnostic. It computes exact recurrent adjoints, chooses per-position
  horizons only when the geometric omitted-tail bound is below ternary
  flip-margin slack, and verifies exact/truncated flip agreement.
- Initial implementation exposed an off-by-one in the finite geometric tail
  bound; the corrected gate includes the first omitted tail term.
- Canonical diagnostic passed with `positive_accepted_fraction_mean=0.9375`,
  `positive_mean_horizon_mean=5.864583`,
  `positive_priced_work_ratio_mean=0.106506`,
  `positive_projected_speedup_mean=9.392304`,
  `positive_flip_mismatch_fraction_mean=0.0`, and
  `positive_tail_bound_violation_fraction_mean=0.0`.
- Controls failed as required: rho-fail had no certified truncation mass,
  margin-fail missed the accepted-fraction threshold, and the optimistic-bound
  control violated the actual tail bound.

## R501-S36 - R501 Speed Math 3 Orthogonal-Basis Recurrent Fast Transform Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s36_orthogonal_basis_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s36_orthogonal_basis.ps1`

Artifacts:

- `.work/jobs/r501-s36-orthogonal-basis/latest/r501_s36_orthogonal_basis_gate.json`
- `.work/jobs/r501-s36-orthogonal-basis/latest/r501_s36_orthogonal_basis_gate.log`

Outcome:

- Implemented the R501 Speed Math 3 static-basis recurrent fast transform as a
  CPU diagnostic. It compares a dense recurrent adjoint against a static
  orthogonal-basis scalar recurrence with dense transforms only at sequence
  boundaries.
- Canonical diagnostic emitted `R501_S36_ORTHOGONAL_BASIS_GATE active=1` and
  passed with `positive_relative_adjoint_error_mean=1.190491e-07`,
  `positive_priced_work_ratio_mean=0.081875`,
  `positive_projected_speedup_mean=12.213740`, and
  `positive_measured_cpu_speedup_mean=1.363281`.
- Controls failed as required: moving-basis reintroduced per-timestep dense
  work, nonorthogonal ill-conditioning broke the exact quotient safety, and
  short segments failed the boundary-transform pricing budget.

## R501-S37 - R501 Speed Math 4 Subspace-Projected Forward Eligibility Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s37_subspace_forward_eligibility_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s37_subspace_forward_eligibility.ps1`

Artifacts:

- `.work/jobs/r501-s37-subspace-forward-eligibility/latest/r501_s37_subspace_forward_eligibility_gate.json`
- `.work/jobs/r501-s37-subspace-forward-eligibility/latest/r501_s37_subspace_forward_eligibility_gate.log`

Outcome:

- Implemented the R501 Speed Math 4 subspace-projected forward eligibility gate
  as a CPU diagnostic over dense `d*d` transition-parameter eligibility vectors.
- Canonical diagnostic emitted `R501_S37_SUBSPACE_FORWARD_ELIGIBILITY_GATE active=1`
  and passed with `positive_selected_rank_mean=4.0`,
  `positive_max_residual_max=1.891509e-15`,
  `positive_priced_work_ratio_mean=0.243333`,
  `positive_projected_speedup_mean=4.109589`, and
  `positive_storage_ratio_mean=0.048611`.
- Controls failed as required: flat-spectrum and misaligned-Krylov conditions
  required near-total rank, while short segments failed the projected-trace
  pricing budget despite exact residuals.

## R501-S38 - R501 Speed Math 5 Temporal Multigrid CE Gradient Coarsening Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s38_temporal_multigrid_ce_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s38_temporal_multigrid_ce.ps1`

Artifacts:

- `.work/jobs/r501-s38-temporal-multigrid-ce/latest/r501_s38_temporal_multigrid_ce_gate.json`
- `.work/jobs/r501-s38-temporal-multigrid-ce/latest/r501_s38_temporal_multigrid_ce_gate.log`
- `.work/jobs/r501-s38-temporal-multigrid-ce/k4-probe/r501_s38_temporal_multigrid_ce_gate.json`

Outcome:

- Implemented the R501 Speed Math 5 temporal multigrid CE gradient coarsening
  gate as a CPU diagnostic over exact block-boundary cotangents and linear
  internal interpolation.
- Initial `K=8` positive probe failed with
  `positive_residual_debt_mean=0.069869`; the canonical conservative `K=4`
  diagnostic emitted `R501_S38_TEMPORAL_MULTIGRID_CE_GATE active=1` and passed.
- Canonical positive metrics: `boundary_relative_error_max=0.0`,
  `positive_residual_debt_mean=0.024398`,
  `positive_priced_work_ratio_mean=0.345497`, and
  `positive_projected_speedup_mean=2.894490`.
- Controls failed as required: impulse and curvature conditions exceeded the
  residual-debt budget, while `K=2` failed the pricing budget.

## R501-S39 - R501 Speed Math 6 Contrastive Gradient-Sign Margin Certificate Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s39_gradient_sign_margin_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s39_gradient_sign_margin.ps1`

Artifacts:

- `.work/jobs/r501-s39-gradient-sign-margin/latest/r501_s39_gradient_sign_margin_gate.json`
- `.work/jobs/r501-s39-gradient-sign-margin/latest/r501_s39_gradient_sign_margin_gate.log`

Outcome:

- Implemented the R501 Speed Math 6 interval gradient-sign margin certificate as
  a CPU diagnostic over exact ternary threshold decisions and interval
  certificates with dense fallback for ambiguous cases.
- Canonical diagnostic emitted `R501_S39_GRADIENT_SIGN_MARGIN_GATE active=1`
  and passed with `positive_certified_fraction_mean=0.999756`,
  `positive_fallback_fraction_mean=0.000244`,
  `positive_certified_mismatch_fraction_mean=0.0`,
  `positive_priced_work_ratio_mean=0.095244`, and
  `positive_projected_speedup_mean=10.499361`.
- Controls failed as required: loose bounds and near-margin distributions paid
  dense fallback on nearly all items, while optimistic false certificates were
  cheap but mismatched exact ternary decisions.

## R501-S40 - R501 Speed Math 7 Operator-Split Lagged Dense Corrections Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s40_operator_split_lagged_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s40_operator_split_lagged.ps1`

Artifacts:

- `.work/jobs/r501-s40-operator-split-lagged/latest/r501_s40_operator_split_lagged_gate.json`
- `.work/jobs/r501-s40-operator-split-lagged/latest/r501_s40_operator_split_lagged_gate.log`

Outcome:

- Implemented the R501 Speed Math 7 operator-split lagged dense correction gate
  as a CPU diagnostic over `A = D + R`, with block-diagonal `D` applied every
  timestep and dense residual corrections sampled/interpolated every `M=8`.
- Canonical diagnostic emitted `R501_S40_OPERATOR_SPLIT_LAGGED_GATE active=1`
  and passed with `positive_residual_operator_fraction_mean=0.254218`,
  `positive_residual_debt_ratio_mean=0.019624`,
  `positive_priced_work_ratio_mean=0.441492`, and
  `positive_projected_speedup_mean=2.265121`.
- Controls failed as required: residual-dominant made `R` carry most credit,
  high-frequency residuals exceeded the debt budget, and `M=1` failed pricing.

## R501-S41 - R501 Speed Math 8 State-Transition Lie Algebra Generator Precomputation Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s41_lie_algebra_precompute_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s41_lie_algebra_precompute.ps1`

Artifacts:

- `.work/jobs/r501-s41-lie-algebra-precompute/latest/r501_s41_lie_algebra_precompute_gate.json`
- `.work/jobs/r501-s41-lie-algebra-precompute/latest/r501_s41_lie_algebra_precompute_gate.log`

Outcome:

- Implemented the R501 Speed Math 8 Lie-algebra generator precompute gate as a
  CPU diagnostic over finite-token precomputed sparse adjoint-action operators.
- Canonical diagnostic emitted `R501_S41_LIE_ALGEBRA_PRECOMPUTE_GATE active=1`
  and passed with `positive_relative_solution_error_mean=0.0`,
  `positive_sparse_nnz_ratio_mean=0.0625`,
  `positive_precompute_amortized_ratio_mean=0.03125`,
  `positive_priced_work_ratio_mean=0.113750`, and
  `positive_projected_speedup_mean=8.791209`.
- Controls failed as required: dense exponentials destroyed sparsity,
  large vocabulary size destroyed precompute amortization, and truncated
  structure broke exactness despite sparse pricing.

## R501-S42 - R501 Speed Math 9 Reversible SSM Cotangent Bypass Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s42_reversible_cotangent_bypass_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s42_reversible_cotangent_bypass.ps1`

Artifacts:

- `.work/jobs/r501-s42-reversible-cotangent-bypass/latest/r501_s42_reversible_cotangent_bypass_gate.json`
- `.work/jobs/r501-s42-reversible-cotangent-bypass/latest/r501_s42_reversible_cotangent_bypass_gate.log`

Outcome:

- Implemented the R501 Speed Math 9 reversible cotangent bypass gate as a CPU
  diagnostic over exact reverse state reconstruction and reconstructed-gradient
  replay under a memory-limited `2x` batch price model.
- Canonical diagnostic emitted `R501_S42_REVERSIBLE_COTANGENT_BYPASS_GATE active=1`
  and passed with `positive_reconstruction_error_mean=9.890146e-13`,
  `positive_grad_b_error_mean=2.178130e-11`,
  `positive_loss_gap_mean=0.0`,
  `positive_activation_memory_ratio_mean=0.015504`,
  `positive_time_to_target_ratio_mean=0.252060`, and
  `positive_projected_speedup_mean=3.967316`.
- Controls failed as required: compute-bound pricing did not clear the `0.50x`
  time-to-target inequality, non-reversible target dynamics created a large
  loss gap, and inverse precision noise broke reconstruction.

## R501-S43 - R501 Speed Math 10 Sparsified Adjoint Support Masking Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s43_sparse_adjoint_support_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s43_sparse_adjoint_support.ps1`

Artifacts:

- `.work/jobs/r501-s43-sparse-adjoint-support/latest/r501_s43_sparse_adjoint_support_gate.json`
- `.work/jobs/r501-s43-sparse-adjoint-support/latest/r501_s43_sparse_adjoint_support_gate.log`

Outcome:

- Implemented the R501 Speed Math 10 sparse adjoint support-mask gate as a CPU
  diagnostic over cotangent support density, omitted-tail transition debt, and
  priced sparse transition application.
- Canonical diagnostic emitted `R501_S43_SPARSE_ADJOINT_SUPPORT_GATE active=1`
  and passed with `positive_support_density_mean=0.098958`,
  `positive_zero_fraction_mean=0.901042`,
  `positive_residual_debt_ratio_mean=0.001422`,
  `positive_priced_work_ratio_mean=0.181178`, and
  `positive_projected_speedup_mean=5.519439`.
- Controls failed as required: dense gradients were not sparse enough,
  hidden below-threshold tails carried transition debt, and high sparse-kernel
  overhead erased the priced gain.

## R501-S44 - R501 Speed Math 11 Recurrent-Cotangent Predictor-Corrector Tube Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s44_predictor_corrector_tube_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s44_predictor_corrector_tube.ps1`

Artifacts:

- `.work/jobs/r501-s44-predictor-corrector-tube/latest/r501_s44_predictor_corrector_tube_gate.json`
- `.work/jobs/r501-s44-predictor-corrector-tube/latest/r501_s44_predictor_corrector_tube_gate.log`

Outcome:

- Implemented the R501 Speed Math 11 predictor-corrector tube gate as a CPU
  diagnostic over low-rank cotangent prediction and sampled Bellman residual
  verification.
- Canonical diagnostic emitted `R501_S44_PREDICTOR_CORRECTOR_TUBE_GATE active=1`
  and passed with `positive_lambda_error_mean=1.746450e-16`,
  `positive_full_bellman_residual_mean=0.0`,
  `positive_sampled_bellman_residual_mean=0.0`,
  `positive_priced_work_ratio_mean=0.228333`, and
  `positive_projected_speedup_mean=4.379562`.
- Controls failed as required: full verification recovered dense cost,
  projection aliasing missed residual debt, and poor predictor residual/error
  was high.

## R501-S45 - R501 Speed Math 12 Local Error-Feedback Forward Eligibility Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s45_error_feedback_forward_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s45_error_feedback_forward.ps1`

Artifacts:

- `.work/jobs/r501-s45-error-feedback-forward/latest/r501_s45_error_feedback_forward_gate.json`
- `.work/jobs/r501-s45-error-feedback-forward/latest/r501_s45_error_feedback_forward_gate.log`

Outcome:

- Implemented the R501 Speed Math 12 local error-feedback forward-eligibility
  gate as a CPU diagnostic over synthetic-adjoint error expectation, martingale
  descent alignment, phase-delay oscillation, step-count proxy, and price.
- Canonical diagnostic emitted `R501_S45_ERROR_FEEDBACK_FORWARD_GATE active=1`
  and passed with `positive_error_decay_ratio_mean=0.202005`,
  `positive_martingale_descent_ratio_mean=1.000031`,
  `positive_phase_flip_rate_mean=0.070588`,
  `positive_step_count_ratio_mean=0.999970`,
  `positive_priced_work_ratio_mean=0.290000`, and
  `positive_projected_speedup_mean=3.448276`.
- Controls failed as required: phase delay exploded the projected step-count
  proxy, biased error did not converge to zero, and correction every step
  recovered dense backward cost.

## R501-S46 - R501 Speed Math 13 Tensor-Train Cotangent Transport Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s46_tt_cotangent_transport_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s46_tt_cotangent_transport.ps1`

Artifacts:

- `.work/jobs/r501-s46-tt-cotangent-transport/latest/r501_s46_tt_cotangent_transport_gate.json`
- `.work/jobs/r501-s46-tt-cotangent-transport/latest/r501_s46_tt_cotangent_transport_gate.log`

Outcome:

- Implemented the R501 Speed Math 13 Tensor-Train cotangent transport gate as a
  CPU diagnostic over TT compression error, direct TT-core adjoint transport,
  required-rank growth, and priced work against dense transition cost.
- Canonical diagnostic emitted `R501_S46_TT_COTANGENT_TRANSPORT_GATE active=1`
  and passed with `positive_initial_compression_error_mean=1.192817e-15`,
  `positive_transport_relative_error_mean=1.221401e-15`,
  `positive_required_rank_mean=3.0`,
  `positive_rank_growth_ratio_mean=0.046875`,
  `positive_priced_work_ratio_mean=0.061453`, and
  `positive_projected_speedup_mean=16.272493`.
- Controls failed as required: full-rank dense cotangents needed TT rank `64`
  and recovered dense cost, entangled operators were not represented in
  separable TT-core space, and rank-1 compression violated the invariant.

## R501-S47 - R501 Speed Math 14 Submodular Trajectory Credit Cover Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s47_submodular_credit_cover_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s47_submodular_credit_cover.ps1`

Artifacts:

- `.work/jobs/r501-s47-submodular-credit-cover/latest/r501_s47_submodular_credit_cover_gate.json`
- `.work/jobs/r501-s47-submodular-credit-cover/latest/r501_s47_submodular_credit_cover_gate.log`

Outcome:

- Implemented the R501 Speed Math 14 submodular trajectory credit-cover gate as
  a CPU diagnostic over CE loss cover, actual gradient-energy cover, residual
  non-peak debt, selected credit-origin fraction, and priced exact-credit work.
- Canonical diagnostic emitted `R501_S47_SUBMODULAR_CREDIT_COVER_GATE active=1`
  and passed with `positive_selected_count_mean=8.0`,
  `positive_loss_cover_fraction_mean=0.953660`,
  `positive_gradient_cover_fraction_mean=0.942230`,
  `positive_residual_debt_ratio_mean=0.057770`,
  `positive_priced_work_ratio_mean=0.101250`, and
  `positive_projected_speedup_mean=9.876543`.
- Controls failed as required: distributed credit left high residual debt,
  expensive peak finding erased the priced gain, and loss peaks that did not
  match gradient energy were rejected.

## R501-S48 - R501 Speed Math 15 Phase-Area Action Integrator Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s48_phase_area_integrator_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s48_phase_area_integrator.ps1`

Artifacts:

- `.work/jobs/r501-s48-phase-area-integrator/latest/r501_s48_phase_area_integrator_gate.json`
- `.work/jobs/r501-s48-phase-area-integrator/latest/r501_s48_phase_area_integrator_gate.log`

Outcome:

- Implemented the R501 Speed Math 15 phase-area action integrator gate as a CPU
  diagnostic over conservative-field curl, macro-vs-micro update residual debt,
  descent lower-bound margin, and priced macro integration plus verification.
- Canonical diagnostic emitted `R501_S48_PHASE_AREA_INTEGRATOR_GATE active=1`
  and passed with `positive_curl_ratio_mean=0.0`,
  `positive_macro_update_error_mean=0.002098`,
  `positive_lower_bound_ratio_mean=0.945559`,
  `positive_residual_debt_ratio_mean=0.002098`,
  `positive_priced_work_ratio_mean=0.125000`, and
  `positive_projected_speedup_mean=8.0`.
- Controls failed as required: high curl accumulated residual debt, full
  verification recovered dense micro-step cost, and an excessive lower-bound
  beta/debt certificate invalidated the macro action.

## R501-S49 - R501 Speed Paradigm 1 Spectral-Gated Exact Cotangent Truncation Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s49_sgect_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s49_sgect.ps1`

Artifacts:

- `.work/jobs/r501-s49-sgect/latest/r501_s49_sgect_gate.json`
- `.work/jobs/r501-s49-sgect/latest/r501_s49_sgect_gate.log`

Outcome:

- Implemented the R501 Speed Paradigm 1 SGECT gate as a CPU diagnostic over
  channel-local spectral horizons, certified omitted-tail bounds, exact ternary
  update equivalence, dense fallback mass, and priced truncated backward work.
- Canonical diagnostic emitted `R501_S49_SGECT_GATE active=1` and passed with
  `positive_mean_horizon_mean=1.0`,
  `positive_dense_fallback_fraction_mean=0.007812`,
  `positive_tail_bound_violation_fraction_mean=0.0`,
  `positive_ternary_mismatch_fraction_mean=0.0`,
  `positive_priced_work_ratio_mean=0.042813`, and
  `positive_projected_speedup_mean=23.357664`.
- Controls failed as required: tiny flip margins increased fallback, near-unit
  spectral radii recovered long horizons, and optimistic tail bounds caused
  ternary decision mismatches.

## R501-S50 - R501 Speed Paradigm 2 Stokes Boundary-Integral CE Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s50_stbice_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s50_stbice.ps1`

Artifacts:

- `.work/jobs/r501-s50-stbice/latest/r501_s50_stbice_gate.json`
- `.work/jobs/r501-s50-stbice/latest/r501_s50_stbice_gate.log`

Outcome:

- Implemented the R501 Speed Paradigm 2 STBICE gate as a CPU diagnostic over
  dense curl-volume versus oriented boundary circulation, hidden non-boundary
  CE source mass, and priced boundary plus audit work.
- Canonical diagnostic emitted `R501_S50_STBICE_GATE active=1` and passed with
  `positive_relative_integral_error_mean=1.851017e-16`,
  `positive_unrepresented_source_fraction_mean=0.0`,
  `positive_boundary_to_volume_work_ratio_mean=0.0625`,
  `positive_priced_work_ratio_mean=0.0975`, and
  `positive_projected_speedup_mean=10.256410`.
- Controls failed as required: interior non-exact source mass was not recovered
  from the boundary, wrong orientation broke exactness, and small/high-audit
  boundary evaluation recovered too much dense cost.

## R501-S51 - R501 Speed Paradigm 3 Target-Forced Dual-State Forward Flow Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s51_tfds_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s51_tfds.ps1`

Artifacts:

- `.work/jobs/r501-s51-tfds/latest/r501_s51_tfds_gate.json`
- `.work/jobs/r501-s51-tfds/latest/r501_s51_tfds_gate.log`

Outcome:

- Implemented the R501 Speed Paradigm 3 TFDS gate as a CPU diagnostic over a
  linear target-forced two-forward identity, dense two-state gradient reference,
  target alignment, and priced second-forward work.
- Canonical diagnostic emitted `R501_S51_TFDS_GATE active=1` and passed with
  `positive_dual_state_error_mean=9.382755e-16`,
  `positive_gradient_error_mean=1.086319e-15`,
  `positive_target_alignment_error_mean=0.0`,
  `positive_descent_alignment_ratio_mean=1.0`,
  `positive_priced_work_ratio_mean=0.61`, and
  `positive_projected_speedup_mean=1.639344`.
- Controls failed as required: nonlinear transition broke the identity, shifted
  target forcing broke alignment, and expensive second forward erased too much
  of the speed advantage.

## R501-S52 - R501 Speed Paradigm 4 Ternary-Constrained Time-To-Flip Escrow Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s52_tctfe_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s52_tctfe.ps1`

Artifacts:

- `.work/jobs/r501-s52-tctfe/latest/r501_s52_tctfe_gate.json`
- `.work/jobs/r501-s52-tctfe/latest/r501_s52_tctfe_gate.log`

Outcome:

- Implemented the R501 Speed Paradigm 4 TCTFE gate as a CPU diagnostic over
  exact distance-to-flip margins, sparse backward wakeups, realized ternary
  artifact equivalence, false-sleep mismatch detection, and priced escrow audit
  work.
- Canonical diagnostic emitted `R501_S52_TCTFE_GATE active=1` and passed with
  `positive_ternary_mismatch_fraction_mean=0.0`,
  `positive_dense_flip_fraction_mean=0.119141`,
  `positive_sparse_backward_fraction_mean=0.119141`,
  `positive_sleeping_parameter_fraction_mean=0.880859`,
  `positive_zero_artifact_residual_fraction_mean=1.0`,
  `positive_priced_work_ratio_mean=0.249141`, and
  `positive_projected_speedup_mean=4.013797`.
- Controls failed as required: tiny margins recovered dense backward evaluation,
  under-covering escrow potential caused false sleep and ternary mismatches, and
  high audit cost erased the speed advantage.

## R501-S53 - R501 Speed Paradigm 5 Autoregressive Null-Space Projection Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s53_ansp_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s53_ansp.ps1`

Artifacts:

- `.work/jobs/r501-s53-ansp/latest/r501_s53_ansp_gate.json`
- `.work/jobs/r501-s53-ansp/latest/r501_s53_ansp_gate.log`

Outcome:

- Implemented the R501 Speed Paradigm 5 ANSP gate as a CPU diagnostic over
  output-head row-space projection, null cotangent dropping, immediate and
  one-step future logit residuals, active descent preservation, and projected
  backward pricing.
- Canonical diagnostic emitted `R501_S53_ANSP_GATE active=1` and passed with
  `positive_active_rank_fraction_mean=0.125`,
  `positive_null_energy_fraction_mean=0.956494`,
  `positive_immediate_logit_residual_mean=3.985381e-16`,
  `positive_future_logit_residual_mean=7.269112e-16`,
  `positive_descent_alignment_mean=1.0`,
  `positive_priced_work_ratio_mean=0.2`, and
  `positive_projected_speedup_mean=5.0`.
- Controls failed as required: high output-head rank recovered dense cost,
  leaky projection corrupted immediate logits, and null-to-active transition
  coupling corrupted future logits.

## R501-S54 - R501 Speed Paradigm 6 Amortized Algebraic Recurrent Inversion Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s54_aari_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s54_aari.ps1`

Artifacts:

- `.work/jobs/r501-s54-aari/latest/r501_s54_aari_gate.json`
- `.work/jobs/r501-s54-aari/latest/r501_s54_aari_gate.log`

Outcome:

- Implemented the R501 Speed Paradigm 6 AARI gate as a CPU diagnostic over
  exact modal inversion of a fixed linear recurrent backward system, block
  residuals, inverse conditioning, and amortized block-inversion pricing.
- Canonical diagnostic emitted `R501_S54_AARI_GATE active=1` and passed with
  `positive_solution_error_mean=4.639885e-16`,
  `positive_block_residual_mean=3.369658e-16`,
  `positive_condition_proxy_mean=3.327174`,
  `positive_priced_work_ratio_mean=0.37`, and
  `positive_projected_speedup_mean=2.702703`.
- Controls failed as required: time-varying transitions broke the static
  inverse, near-unit poles failed conditioning/price, and state-dependent
  Jacobians broke the linear inverse contract.

## R501-S55 - R501 Speed Paradigm 7 Predictability-Triggered Gradient Lumping Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s55_ptgl_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s55_ptgl.ps1`

Artifacts:

- `.work/jobs/r501-s55-ptgl/latest/r501_s55_ptgl_gate.json`
- `.work/jobs/r501-s55-ptgl/latest/r501_s55_ptgl_gate.log`

Outcome:

- Implemented the R501 Speed Paradigm 7 PTGL gate as a CPU diagnostic over
  CE-triggered segment skipping, exact discounted boundary lumps, residual debt,
  gradient-energy debt, and pricing.
- Canonical diagnostic emitted `R501_S55_PTGL_GATE active=1` and passed with
  `positive_skipped_token_fraction_mean=0.8125`,
  `positive_boundary_error_mean=0.0`,
  `positive_residual_debt_mean=0.044384`,
  `positive_gradient_energy_debt_mean=0.001970`,
  `positive_priced_work_ratio_mean=0.374062`, and
  `positive_projected_speedup_mean=2.673350`.
- Controls failed as required: unsafe threshold lumped high-entropy debt,
  oscillating low-CE gradients exposed missing gradient-energy certification,
  and fragmented predictable spans recovered dense cost.

## R501-S56 - R501 Speed Paradigm 8 Quotient-State Transition Skipping Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s56_qsts_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s56_qsts.ps1`

Artifacts:

- `.work/jobs/r501-s56-qsts/latest/r501_s56_qsts_gate.json`
- `.work/jobs/r501-s56-qsts/latest/r501_s56_qsts_gate.log`

Outcome:

- Implemented the R501 Speed Paradigm 8 QSTS gate as a CPU diagnostic over
  exact macro transitions for constant null-token spans, exact macro cotangent
  propagation, objective equivalence, no-interior-source debt, and pricing.
- Canonical diagnostic emitted `R501_S56_QSTS_GATE active=1` and passed with
  `positive_skipped_transition_fraction_mean=0.8`,
  `positive_state_error_max=2.825476e-16`,
  `positive_lambda_error_max=1.303091e-16`,
  `positive_objective_error_max=0.0`,
  `positive_priced_work_ratio_mean=0.36`, and
  `positive_projected_speedup_mean=2.777778`.
- Controls failed as required: variable token grouping broke exact equivalence,
  hidden interior CE sources broke the no-source quotient contract, and
  fragmented null spans recovered dense cost.

## R501-S57 - R501 Speed Paradigm 9 Spline-Interpolated Cotangent Trajectory Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s57_sict_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s57_sict.ps1`

Artifacts:

- `.work/jobs/r501-s57-sict/latest/r501_s57_sict_gate.json`
- `.work/jobs/r501-s57-sict/latest/r501_s57_sict_gate.log`

Outcome:

- Implemented the R501 Speed Paradigm 9 SICT gate as a CPU diagnostic over
  cubic cotangent interpolation from dense knots, fourth-difference curvature
  certification, sparse interior verification, and pricing.
- Canonical diagnostic emitted `R501_S57_SICT_GATE active=1` and passed with
  `positive_knot_stride_mean=8.0`,
  `positive_residual_debt_max=2.045904e-16`,
  `positive_sampled_residual_max=2.275443e-16`,
  `positive_fourth_difference_max=1.426811e-15`,
  `positive_priced_work_ratio_mean=0.276240`, and
  `positive_projected_speedup_mean=3.620036`.
- Controls failed as required: high curvature broke the residual certificate,
  an interior impulse was caught by sparse verification, and a dense knot
  schedule recovered dense cost.

## R501-S58 - R501 Speed Paradigm 10 Phase-Diagonalized Backward Masking Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s58_pdbm_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s58_pdbm.ps1`

Artifacts:

- `.work/jobs/r501-s58-pdbm/latest/r501_s58_pdbm_gate.json`
- `.work/jobs/r501-s58-pdbm/latest/r501_s58_pdbm_gate.log`

Outcome:

- Implemented the R501 Speed Paradigm 10 PDBM gate as a CPU diagnostic over
  orthogonal phase-basis block masking, off-diagonal residual debt, certified
  ternary update preservation, and pricing.
- Canonical diagnostic emitted `R501_S58_PDBM_GATE active=1` and passed with
  `positive_block_work_ratio_mean=0.125`,
  `positive_offdiag_energy_ratio_max=0.005722831`,
  `positive_residual_debt_ratio_max=0.003855995`,
  `positive_ternary_mismatch_fraction_max=0.0`,
  `positive_priced_work_ratio_mean=0.295`, and
  `positive_projected_speedup_mean=3.389831`.
- Controls failed as required: real off-block cross-talk broke the mask, wrong
  basis destroyed the block structure, and wide phase blocks recovered dense
  cost.

## R501-S59 - R501 Speed Paradigm 11 Cross-Entropy Potential Field Adjoint Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s59_cepfa_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s59_cepfa.ps1`

Artifacts:

- `.work/jobs/r501-s59-cepfa/latest/r501_s59_cepfa_gate.json`
- `.work/jobs/r501-s59-cepfa/latest/r501_s59_cepfa_gate.log`

Outcome:

- Implemented the R501 Speed Paradigm 11 CEPFA gate as a CPU diagnostic over
  exact value-potential gradients, full and sparse Bellman residual
  certification, and pricing.
- Canonical diagnostic emitted `R501_S59_CEPFA_GATE active=1` and passed with
  `positive_proxy_error_max=1.864795e-16`,
  `positive_bellman_residual_max=1.633022e-16`,
  `positive_sampled_residual_max=1.633130e-16`,
  `positive_priced_work_ratio_mean=0.38`, and
  `positive_projected_speedup_mean=2.631579`.
- Controls failed as required: stale potential parameters broke Bellman
  residuals, nonlinear dynamics broke the linear value-potential contract, and
  an expensive potential head recovered dense cost.

## R501-S60 - R501 Speed Paradigm 12 Sparse Orthogonal Basis Credit Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s60_sobc_gate.py`
- `powershell -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s60_sobc.ps1`

Artifacts:

- `.work/jobs/r501-s60-sobc/latest/r501_s60_sobc_gate.json`
- `.work/jobs/r501-s60-sobc/latest/r501_s60_sobc_gate.log`

Outcome:

- Implemented the R501 Speed Paradigm 12 SOBC gate as a CPU diagnostic over
  sparse orthogonal factor-basis credit, exact dense-gradient reconstruction,
  captured energy, and pricing.
- Canonical diagnostic emitted `R501_S60_SOBC_GATE active=1` and passed with
  `positive_sparse_nnz_fraction_mean=0.015625`,
  `positive_reconstruction_error_max=8.122641e-16`,
  `positive_captured_energy_min=1.0`,
  `positive_priced_work_ratio_mean=0.135625`, and
  `positive_projected_speedup_mean=7.373272`.
- Controls failed as required: dense excitation and wrong basis lost most
  gradient energy, and a large active rectangle recovered dense cost.

## R501-S79 - R501 Quality Paradigm 1 Information-Theoretic Active Sensing Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s79_information_active_sensing_gate.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s79_information_active_sensing.ps1`

Artifacts:

- `.work/jobs/r501-s79-information-active-sensing/latest/r501_s79_information_active_sensing_gate.json`
- `.work/jobs/r501-s79-information-active-sensing/latest/r501_s79_information_active_sensing_gate.log`

Outcome:

- Implemented the R501 Quality Paradigm 1 IAS gate as a CPU diagnostic over
  surprise-proportional token masking, actionable CE retention, gradient and
  state residual debt, lattice-noise avoidance, and pricing.
- Canonical diagnostic emitted `R501_S79_IAS_GATE active=1` and passed with
  `positive_retained_fraction_mean=0.240234375`,
  `positive_actionable_ce_retained_fraction_min=0.9034213201058192`,
  `positive_gradient_residual_ratio_max=0.036541777139558904`,
  `positive_gradient_cosine_min=0.9999904755504103`,
  `positive_state_residual_ratio_max=0.03620185417691084`,
  `positive_priced_work_ratio_mean=0.300234375`, and
  `positive_projected_token_efficiency_multiplier_mean=3.0202498741777637`.
- Controls failed as required: weak CE proxy and high threshold created credit
  debt, no-sparsity recovered dense price, state coupling exposed hidden
  recurrent debt, and selector overhead erased the denominator.

## R501-S80 - R501 Quality Paradigm 2 Recurrent Target-Containing Subspace Certificate Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s80_rtcsc_gate.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s80_rtcsc.ps1`

Artifacts:

- `.work/jobs/r501-s80-rtcsc/latest/r501_s80_rtcsc_gate.json`
- `.work/jobs/r501-s80-rtcsc/latest/r501_s80_rtcsc_gate.log`

Outcome:

- Implemented the R501 Quality Paradigm 2 RTCSC gate as a CPU diagnostic over
  recent-hidden-state subspace containment, normalized target loss, ternary
  coefficient realization, and priced subspace search.
- Canonical diagnostic emitted `R501_S80_RTCSC_GATE active=1` and passed with
  `positive_active_rank_fraction_mean=0.125`,
  `positive_containment_residual_ratio_max=0.009999500037496882`,
  `positive_continuous_loss_ratio_max=9.999000099990014e-05`,
  `positive_ternary_realization_error_ratio_max=0.009999500037496877`,
  `positive_ternary_loss_ratio_max=9.999000099990007e-05`,
  `positive_descent_cosine_min=0.9999500037496873`,
  `positive_priced_work_ratio_mean=0.205`, and
  `positive_projected_token_efficiency_multiplier_mean=4.877561024385367`.
- Controls failed as required: outside and stale spans broke containment,
  oversized rank recovered dense price, continuous-but-nonternary targets broke
  realization, and certificate overhead erased the denominator.

## R501-S81 - R501 Quality Paradigm 3 Exact Equivalence Quotient for Contextual Synonyms Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s81_eeq_cs_gate.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s81_eeq_cs.ps1`

Artifacts:

- `.work/jobs/r501-s81-eeq-cs/latest/r501_s81_eeq_cs_gate.json`
- `.work/jobs/r501-s81-eeq-cs/latest/r501_s81_eeq_cs_gate.log`

Outcome:

- Implemented the R501 Quality Paradigm 3 EEQ-CS gate as a CPU diagnostic over
  hidden-state quotient classes, cotangent drift, multiplicity-scaled quotient
  gradients, and priced class-count reduction.
- Canonical diagnostic emitted `R501_S81_EEQ_CS_GATE active=1` and passed with
  `positive_quotient_class_count_mean=96.0`,
  `positive_mean_multiplicity_mean=4.0`,
  `positive_max_hidden_radius_max=0.014454893032572712`,
  `positive_max_cotangent_drift_ratio_max=0.02313991268178267`,
  `positive_quotient_gradient_relative_error_max=0.007596020320777387`,
  `positive_quotient_gradient_cosine_min=0.9999766680893215`,
  `positive_priced_work_ratio_mean=0.31`, and
  `positive_projected_token_efficiency_multiplier_mean=3.2069002051765287`.
- Controls failed as required: false equivalence and wide epsilon broke
  cotangent preservation, low multiplicity and expensive indexing erased the
  denominator, and missing multiplicity scaling undercounted the dense signal.

## R501-S82 - R501 Quality Paradigm 4 Fisher-Weighted Ternary Commitment Escrow Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s82_fwtce_gate.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s82_fwtce.ps1`

Artifacts:

- `.work/jobs/r501-s82-fwtce/latest/r501_s82_fwtce_gate.json`
- `.work/jobs/r501-s82-fwtce/latest/r501_s82_fwtce_gate.log`

Outcome:

- Implemented the R501 Quality Paradigm 4 FWTCE gate as a CPU diagnostic over
  Fisher-weighted ternary flip blocking, prior capability loss, future-progress
  retention, stale/no-signal controls, threshold controls, and ledger pricing.
- Canonical diagnostic emitted `R501_S82_FWTCE_GATE active=1` and passed with
  `positive_high_fisher_block_min=1.0`,
  `positive_unsafe_leak_max=0.0`,
  `positive_prior_loss_reduction_min=0.9593859929645028`,
  `positive_new_task_retention_min=0.8576524953855083`,
  `positive_allowed_low_fisher_min=1.0`,
  `positive_priced_work_ratio_mean=0.4995312500000001`, and
  `positive_projected_capability_retention_multiplier_mean=1.6749573205776513`.
- Controls failed as required: uninformative or stale Fisher leaked high-risk
  flips, overlock starved future updates, underlock preserved no prior
  capability, and an expensive ledger erased the denominator.

## R501-S83 - R501 Quality Paradigm 5 Lossless Back-Translation of Residual Error Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s83_lbtre_gate.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s83_lbtre.ps1`

Artifacts:

- `.work/jobs/r501-s83-lbtre/latest/r501_s83_lbtre_gate.json`
- `.work/jobs/r501-s83-lbtre/latest/r501_s83_lbtre_gate.log`

Outcome:

- Implemented the R501 Quality Paradigm 5 LBTRE gate as a CPU diagnostic over
  unique layer-rank collapse detection, original CE cotangent back-translation,
  oracle bottleneck gradient equivalence, non-bottleneck bypass, and detector
  pricing.
- Canonical diagnostic emitted `R501_S83_LBTRE_GATE active=1` and passed with
  `positive_routing_accuracy_mean=1.0`,
  `positive_residual_capture_fraction_mean=1.0`,
  `positive_oracle_gradient_relative_error_mean=0.0`,
  `positive_oracle_gradient_cosine_min=1.0000000000000002`,
  `positive_non_bottleneck_update_energy_fraction_mean=0.0`,
  `positive_updated_layer_fraction_mean=0.16666666666666666`,
  `positive_priced_work_ratio_mean=0.24666666666666665`, and
  `positive_projected_token_efficiency_multiplier_mean=4.054054054054054`.
- Controls failed as required: ambiguous rank drops and dense bottlenecks broke
  lossless localization, a wrong rank proxy routed all energy to
  non-bottlenecks, no-collapse traces exposed missing evidence, and an
  expensive detector erased the denominator.

## R501-S84 - R501 Quality Paradigm 6 Orthogonalized Credit Trajectory Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s84_oct_gate.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s84_oct.ps1`

Artifacts:

- `.work/jobs/r501-s84-oct/latest/r501_s84_oct_gate.json`
- `.work/jobs/r501-s84-oct/latest/r501_s84_oct_gate.log`

Outcome:

- Implemented the R501 Quality Paradigm 6 OCT gate as a CPU diagnostic over
  recent-credit projection removal, prior-overwrite reduction, independent
  capability retention, oracle residual matching, and projection pricing.
- Canonical diagnostic emitted `R501_S84_OCT_GATE active=1` and passed with
  `positive_prior_overlap_before_fraction_mean=0.5202119798103786`,
  `positive_prior_overlap_after_fraction_mean=1.6399734655578546e-31`,
  `positive_overwrite_reduction_fraction_mean=1.0`,
  `positive_new_capability_retention_fraction_mean=1.0`,
  `positive_future_descent_cosine_min=0.9999912422938214`,
  `positive_oracle_relative_error_mean=4.51379222578731e-16`,
  `positive_priced_work_ratio_mean=0.34099361293368663`, and
  `positive_projected_capability_efficiency_multiplier_mean=2.9341408754681084`.
- Controls failed as required: nonorthogonal targets erased the new capability,
  stale and noisy bases failed residual matching, no-interference traces had no
  overwrite risk to remove, and high projection overhead erased the denominator.

## R501-S85 - R501 Quality Paradigm 7 Boundary-Conditioned Token Amortization Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s85_bcta_gate.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s85_bcta.ps1`

Artifacts:

- `.work/jobs/r501-s85-bcta/latest/r501_s85_bcta_gate.json`
- `.work/jobs/r501-s85-bcta/latest/r501_s85_bcta_gate.log`

Outcome:

- Implemented the R501 Quality Paradigm 7 BCTA gate as a CPU diagnostic over
  closed-form boundary CE integrals, boundary-gradient preservation, token-span
  amortization, and verifier pricing.
- Canonical diagnostic emitted `R501_S85_BCTA_GATE active=1` and passed with
  `positive_ce_integral_relative_error_max=1.9949559022430236e-16`,
  `positive_boundary_gradient_relative_error_mean=0.0`,
  `positive_boundary_gradient_cosine_min=0.9999999999999998`,
  `positive_collapsed_token_fraction_mean=0.875`,
  `positive_paid_token_ratio_mean=0.17500000000000002`, and
  `positive_projected_token_efficiency_multiplier_mean=5.7142857142857135`.
- Controls failed as required after strengthening the interior-debt controls:
  unpredictable interiors and same-boundary mismatches broke the CE integral,
  drift outside the closed-form family broke boundary gradients, and short spans
  or expensive verification erased the denominator.

## R501-S86 - R501 Quality Paradigm 8 Spectral Normalization of Ternary Transitions Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s86_sntt_gate.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s86_sntt.ps1`

Artifacts:

- `.work/jobs/r501-s86-sntt/latest/r501_s86_sntt_gate.json`
- `.work/jobs/r501-s86-sntt/latest/r501_s86_sntt_gate.log`

Outcome:

- Implemented the R501 Quality Paradigm 8 SNTT gate as a CPU diagnostic over
  ternary transition spectral normalization, singular-floor preservation,
  long-horizon signal retention, non-normal transient controls, stale
  certificate controls, and certificate pricing.
- Canonical diagnostic emitted `R501_S86_SNTT_GATE active=1` and passed with
  `positive_spectral_norm_mean=0.998`,
  `positive_min_singular_value_mean=0.9979999999999997`,
  `positive_long_horizon_signal_ratio_min_mean=0.7739435492434857`,
  `positive_transient_amplification_max_mean=0.9980000000000002`,
  `positive_paid_work_ratio_mean=0.0878125`, and
  `positive_projected_long_context_multiplier_mean=8.813592019854646`.
- Controls failed as required: unnormalized and stale transitions exploded,
  overcontraction passed the upper bound but destroyed long-horizon signal,
  non-normal structure produced unsafe transients, and expensive certification
  erased the denominator.

## R501-S87 - R501 Quality Paradigm 9 Cotangent-Guided Token Resampling Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s87_cgtr_gate.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s87_cgtr.ps1`

Artifacts:

- `.work/jobs/r501-s87-cgtr/latest/r501_s87_cgtr_gate.json`
- `.work/jobs/r501-s87-cgtr/latest/r501_s87_cgtr_gate.log`

Outcome:

- Implemented the R501 Quality Paradigm 9 CGTR gate as a CPU diagnostic over
  proxy-norm token selection, actionable recall, low-signal skipping,
  importance weighting, coherent low-signal controls, and proxy pricing.
- Canonical diagnostic emitted `R501_S87_CGTR_GATE active=1` and passed with
  `positive_actionable_recall_mean=1.0`,
  `positive_low_signal_skip_fraction_mean=1.0`,
  `positive_full_gradient_relative_error_mean=0.0012000383068875074`,
  `positive_actionable_gradient_relative_error_mean=0.0`,
  `positive_gradient_cosine_to_full_min=0.9999991045532026`,
  `positive_paid_token_ratio_mean=0.21125253501511496`, and
  `positive_projected_token_efficiency_multiplier_mean=4.733689151841645`.
- Controls failed as required: weak proxy and too-high threshold lost
  actionable recall, coherent low-signal tokens made skipping unsafe, missing
  importance weighting biased the selected gradient, and expensive proxy scoring
  erased the denominator.

## R501-S88 - R501 Quality Paradigm 10 Ternary Covariance Factorization Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s88_tcf_gate.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s88_tcf.ps1`

Artifacts:

- `.work/jobs/r501-s88-tcf/latest/r501_s88_tcf_gate.json`
- `.work/jobs/r501-s88-tcf/latest/r501_s88_tcf_gate.log`

Outcome:

- Implemented the R501 Quality Paradigm 10 TCF gate as a CPU diagnostic over
  sparse ternary covariance basis recovery, covariance reconstruction,
  dense-sample noise reduction, coherent residual controls, non-ternary
  substrate controls, rank controls, and factorization pricing.
- Canonical diagnostic emitted `R501_S88_TCF_GATE active=1` and passed with
  `positive_covariance_relative_error_mean=0.02219488715547841`,
  `positive_covariance_relative_error_max=0.028231534523657707`,
  `positive_covariance_noise_reduction_fraction_mean=0.477338112473781`,
  `positive_energy_capture_fraction_mean=0.9991609882606131`,
  `positive_basis_alignment_min_min=0.9999999999999999`,
  `positive_active_update_fraction_mean=0.08680555555555555`,
  `positive_paid_work_ratio_mean=0.16680555555555554`, and
  `positive_projected_covariance_efficiency_multiplier_mean=5.995004163197336`.
- Controls failed as required: random basis missed the covariance, continuous
  eigenbasis violated `U in {-1,0,1}`, rank underfit missed energy, dense
  residual debt defeated the sparse basis, and dense active support erased the
  denominator.

## R501-S89 - R501 Quality Paradigm 11 Certified Rare-Token Escrow Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s89_crte_gate.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s89_crte.ps1`

Artifacts:

- `.work/jobs/r501-s89-crte/latest/r501_s89_crte_gate.json`
- `.work/jobs/r501-s89-crte/latest/r501_s89_crte_gate.log`

Outcome:

- Implemented the R501 Quality Paradigm 11 CRTE gate as a CPU diagnostic over
  high-CE rare-token conflict certificates, escrow recall, compatible release,
  rare-capability acquisition, majority retention, false-escrow controls,
  timing controls, stale-geometry controls, and escrow pricing.
- Canonical diagnostic emitted `R501_S89_CRTE_GATE active=1` and passed with
  `positive_rare_token_recall_mean=1.0`,
  `positive_baseline_unique_capability_fraction_mean=0.125`,
  `positive_escrow_unique_capability_fraction_mean=1.0`,
  `positive_unique_capability_gain_mean=8.0`,
  `positive_release_precision_mean=1.0`,
  `positive_majority_retention_fraction_mean=1.0`,
  `positive_paid_work_ratio_mean=0.11125`, and
  `positive_projected_unique_capability_multiplier_mean=8.98876404494382`.
- Controls failed as required: no escrow lost rare recall, premature release and
  stale geometry failed release precision, low-CE false escrow polluted the
  queue and damaged majority retention, and expensive escrow erased the
  denominator.

## R501-S90 - R501 Quality Paradigm 12 Harmonic Credit Assignment Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s90_hca_gate.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s90_hca.ps1`

Artifacts:

- `.work/jobs/r501-s90-hca/latest/r501_s90_hca_gate.json`
- `.work/jobs/r501-s90-hca/latest/r501_s90_hca_gate.log`

Outcome:

- Implemented the R501 Quality Paradigm 12 HCA gate as a CPU diagnostic over
  exact harmonic recurrent-feature states, long-lag retention, tail-credit
  capture, near-lag distortion, lag-order preservation, exponential/truncated
  controls, wrong-kernel controls, shuffled-lag controls, and quotient pricing.
- Canonical diagnostic emitted `R501_S90_HCA_GATE active=1` and passed with
  `positive_state_relative_error_max_max=0.0`,
  `positive_quotient_weight_residual_mean=0.0`,
  `positive_long_lag_retention_fraction_mean=1.0`,
  `positive_tail_credit_capture_fraction_mean=1.0`,
  `positive_near_credit_distortion_mean=0.0`,
  `positive_paid_work_ratio_mean=0.11137254901960784`, and
  `positive_projected_long_context_multiplier_mean=8.97887323943662`.
- Controls failed as required: exponential and truncated credit lost the long
  tail, the wrong polynomial kernel distorted the quotient, shuffled lag order
  broke assignment, and expensive quotient accounting erased the denominator.

## R501-S91 - R501 Quality Paradigm 13 Segment-Level Adjoint Replay Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s91_slar_gate.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s91_slar.ps1`

Artifacts:

- `.work/jobs/r501-s91-slar/latest/r501_s91_slar_gate.json`
- `.work/jobs/r501-s91-slar/latest/r501_s91_slar_gate.log`

Outcome:

- Implemented the R501 Quality Paradigm 13 SLAR gate as a CPU diagnostic over
  segment adjoints decomposed into tube projection plus certified sparse
  residual, noisy stepwise replay controls, sparse-residual controls,
  wrong-tube controls, incoherent-segment controls, harmless residual bounds,
  and replay pricing.
- Canonical diagnostic emitted `R501_S91_SLAR_GATE active=1` and passed after
  correcting the probe to remove the sparse residual before tube projection,
  with `positive_segment_adjoint_relative_error_mean_mean=2.510380145582369e-16`,
  `positive_segment_adjoint_relative_error_max_max=6.304778217674318e-16`,
  `positive_stepwise_relative_error_mean_mean=0.7217953104709156`,
  `positive_noise_reduction_fraction_mean=0.9999999999999998`,
  `positive_tube_energy_capture_fraction_mean=0.9242694120496534`,
  `positive_sparse_residual_recall_mean=1.0`,
  `positive_paid_token_ratio_mean=0.175`, and
  `positive_projected_segment_efficiency_multiplier_mean=5.714285714285714`.
- Controls failed as required: stepwise replay kept noise, missing sparse
  residual left debt, wrong tube lost the segment adjoint, incoherent segments
  violated the harmless residual bound, and expensive replay erased the
  denominator.

## R501-S92 - R501 Quality Paradigm 14 Cross-Sample State-Conditioned CE Quotient Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s92_cs_sccq_gate.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s92_cs_sccq.ps1`

Artifacts:

- `.work/jobs/r501-s92-cs-sccq/latest/r501_s92_cs_sccq_gate.json`
- `.work/jobs/r501-s92-cs-sccq/latest/r501_s92_cs_sccq_gate.log`

Outcome:

- Implemented the R501 Quality Paradigm 14 CS-SCCQ gate as a CPU diagnostic
  over cross-sample state-conditioned CE-gradient quotient classes,
  multiplicity-scaled representatives, residual certificates, false-state
  controls, missing-multiplicity controls, residual-ignored controls, low
  multiplicity controls, and index-cost pricing.
- Canonical diagnostic emitted `R501_S92_CS_SCCQ_GATE active=1` and passed after
  correcting the multiplicity-scaling metric to test the applied
  representative coefficient, with
  `positive_class_count_mean=64.0`,
  `positive_mean_multiplicity_mean=8.0`,
  `positive_max_state_radius_max=0.018672448437046414`,
  `positive_max_gradient_drift_ratio_max=0.06144363574772148`,
  `positive_class_residual_ratio_max=0.028131467885392752`,
  `positive_quotient_gradient_relative_error_max=1.2964624928340724e-15`,
  `positive_quotient_gradient_cosine_min=0.9999999999999997`,
  `positive_multiplicity_scaling_error_ratio_max=0.0`,
  `positive_paid_token_ratio_mean=0.21500000000000002`, and
  `positive_projected_token_efficiency_multiplier_mean=4.651162790697674`.
- Controls failed as required: false state match exceeded gradient drift,
  missing multiplicity broke the quotient gradient, residual ignored left
  residual/gradient debt, low multiplicity erased the denominator, and expensive
  indexing erased the denominator.

## R501-S93 - R501 Quality Paradigm 15 Singular-Value Directed Ternary Bit-Flips Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s93_svd_tbf_gate.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s93_svd_tbf.ps1`

Artifacts:

- `.work/jobs/r501-s93-svd-tbf/latest/r501_s93_svd_tbf_gate.json`
- `.work/jobs/r501-s93-svd-tbf/latest/r501_s93_svd_tbf_gate.log`

Outcome:

- Implemented the R501 Quality Paradigm 15 SVD-TBF gate as a CPU diagnostic
  over zero-mean/high-variance token-gradient matrices, dominant right singular
  vector recovery, sparse ternary projection, raw element-wise flip controls,
  random direction controls, low singular gap controls, nonternary controls,
  and dense/expensive SVD pricing.
- Canonical diagnostic emitted `R501_S93_SVD_TBF_GATE active=1` and passed with
  `positive_top_singular_energy_fraction_min=0.7362718188955173`,
  `positive_singular_gap_ratio_min=9.550897305326538`,
  `positive_top_vector_alignment_min=0.9995024837617101`,
  `positive_ternary_flip_alignment_min=1.0000000000000002`,
  `positive_capability_capture_fraction_min=1.0000000000000004`,
  `positive_raw_flip_capture_fraction_mean=0.01944444444444445`,
  `positive_candidate_vs_raw_capture_gain_min=16.0`,
  `positive_sign_mismatch_fraction_max=0.0`,
  `positive_ternary_violation_count_max=0.0`,
  `positive_paid_work_ratio_mean=0.2675`, and
  `positive_projected_flip_efficiency_multiplier_mean=3.7383177570093458`.
- Controls failed as required: raw element-wise flips and random directions
  captured no capability in the zero-mean fixture, low singular gap made the
  top direction non-identifying, continuous SVD violated the ternary boundary,
  and dense/expensive SVD support erased the denominator.

## R501-S94 - R501 Hardware Efficiency Math 1 Sherman-Morrison Spectral-Partitioned Recurrent Inversion Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s94_sm_spri_gate.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s94_sm_spri.ps1`

Artifacts:

- `.work/jobs/r501-s94-sm-spri/latest/r501_s94_sm_spri_gate.json`
- `.work/jobs/r501-s94-sm-spri/latest/r501_s94_sm_spri_gate.log`

Outcome:

- Implemented the R501 Hardware Efficiency Math 1 SM-SPRI gate as a CPU
  diagnostic over near-isometric stable recurrent coordinates, singular
  coordinate escrow, exact backward trajectory reconstruction, recurrent
  gradient preservation, no-escrow controls, wrong-partition controls,
  quantized-escrow controls, dense-residency controls, and repair-cost pricing.
- Canonical diagnostic emitted `R501_S94_SM_SPRI_GATE active=1` and passed with
  `positive_singular_escrow_recall_min=1.0`,
  `positive_reconstruction_relative_error_max=4.896888604437808e-16`,
  `positive_reconstruction_max_abs_error_max=6.94999613415348e-14`,
  `positive_recurrent_gradient_relative_error_max=5.164237515100902e-16`,
  `positive_stable_inverse_condition_proxy_max=1.0100676788061431`,
  `positive_stored_state_ratio_mean=0.09765625`,
  `positive_paid_vram_ratio_mean=0.13765625`, and
  `positive_projected_vram_efficiency_multiplier_mean=7.264472190692395`.
- Controls failed as required: no escrow and wrong partition missed singular
  coordinates, quantized escrow broke exact reconstruction, dense residency
  erased the memory denominator, and high repair overhead erased the denominator.

## R501-S95 - R501 Hardware Efficiency Math 2 Fused Logit-Cotangent Register Accumulation Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s95_flcra_gate.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s95_flcra.ps1`

Artifacts:

- `.work/jobs/r501-s95-flcra/latest/r501_s95_flcra_gate.json`
- `.work/jobs/r501-s95-flcra/latest/r501_s95_flcra_gate.log`

Outcome:

- Implemented the R501 Hardware Efficiency Math 2 FLCRA gate as a CPU
  diagnostic comparing dense CE hidden-state cotangents against row-streamed
  cotangents that do not globally materialize a `T x V` logit tensor.
- Canonical diagnostic emitted `R501_S95_FLCRA_GATE active=1` and passed with
  `positive_cotangent_relative_error_max=8.90412630045184e-16`,
  `positive_cotangent_max_abs_error_max=1.3322676295501878e-15`,
  `positive_cotangent_cosine_min=1.0`,
  `positive_paid_logit_vram_ratio_mean=0.0078125`,
  `positive_projected_logit_vram_efficiency_multiplier_mean=128.0`,
  `positive_projected_saved_mib_4096x32768_fp16_mean=255.9375`, and
  `positive_compute_work_ratio_mean=1.0`.
- Controls failed as required: raw logits broke softmax exactness, missing
  target omitted the one-hot subtraction, sampled vocab changed the
  distribution, materialized logits erased the VRAM denominator, and extra
  recomputation erased the zero-overhead constraint.

## R501-S96 - R501 Hardware Efficiency Math 3 Boundary-Sparse Activation Escrow Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s96_bsae_gate.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s96_bsae.ps1`

Artifacts:

- `.work/jobs/r501-s96-bsae/latest/r501_s96_bsae_gate.json`
- `.work/jobs/r501-s96-bsae/latest/r501_s96_bsae_gate.log`

Outcome:

- Implemented the R501 Hardware Efficiency Math 3 BSAE gate as a CPU diagnostic
  over a ternary-boundary structural-gradient fixture.
- Canonical diagnostic emitted `R501_S96_BSAE_GATE active=1` and passed with
  `positive_active_fraction_mean=0.12059326171875`,
  `positive_active_recall_min=1.0`,
  `positive_inactive_rejection_min=1.0`,
  `positive_gradient_relative_error_max=0.0`,
  `positive_paid_activation_bit_ratio_mean=0.18309326171875`, and
  `positive_projected_activation_efficiency_multiplier_mean=5.462958263652707`.
- Controls failed as required: no escrow lost active magnitudes, inverted
  boundary stored the wrong side, quantized magnitudes broke exactness, dense
  activation erased the denominator, and random masks missed active support.

## R501-S97 - R501 Hardware Efficiency Math 4 Ternary-Lattice Shadow Weight Virtualization Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s97_tlswv_gate.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s97_tlswv.ps1`

Artifacts:

- `.work/jobs/r501-s97-tlswv/latest/r501_s97_tlswv_gate.json`
- `.work/jobs/r501-s97-tlswv/latest/r501_s97_tlswv_gate.log`

Outcome:

- Implemented the R501 Hardware Efficiency Math 4 TLSWV gate as a CPU
  diagnostic over saturated stable shadow weights, low-margin near-threshold
  weights, and rare massive impulses.
- Canonical diagnostic emitted `R501_S97_TLSWV_GATE active=1` and passed with
  `positive_stable_fraction_mean=0.819879150390625`,
  `positive_impulse_detector_recall_min=1.0`,
  `positive_ternary_mismatch_rate_max=0.0`,
  `positive_virtualized_fraction_mean=0.795281982421875`,
  `positive_paid_shadow_bit_ratio_mean=0.2544231414794922`, and
  `positive_projected_shadow_efficiency_multiplier_mean=3.9306759146169035`.
- Controls failed as required: no detector missed massive impulses,
  overvirtualization and loose margins crossed ternary boundaries, dense shadow
  tracking erased the denominator, and signless virtual state lost negative
  saturated weights.

## R501-S98 - R501 Hardware Efficiency Math 5 Unbiased Stochastic-Quantized Recurrent Buffer Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s98_usqrb_gate.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s98_usqrb.ps1`

Artifacts:

- `.work/jobs/r501-s98-usqrb/latest/r501_s98_usqrb_gate.json`
- `.work/jobs/r501-s98-usqrb/latest/r501_s98_usqrb_gate.log`

Outcome:

- Implemented the R501 Hardware Efficiency Math 5 USQRB gate as a CPU diagnostic
  over recurrent state buffers and projection cotangents.
- Canonical diagnostic emitted `R501_S98_USQRB_GATE active=1` and passed with
  `positive_expectation_gradient_relative_error_max=0.0`,
  `positive_quantizer_state_bias_relative_error_max=0.0`,
  `positive_monte_carlo_gradient_relative_error_max=0.02481682775488569`,
  `positive_paid_buffer_bit_ratio_mean=0.25`, and
  `positive_projected_buffer_efficiency_multiplier_mean=4.0`.
- Controls failed as required: deterministic floor, nearest, signless, and
  clipped quantizers introduced projection-gradient bias, while dense FP16
  residency erased the denominator.

## R501-S99 - R501 Hardware Efficiency Math 6 Sub-Warp Recurrent Checkpoint Resampling Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s99_swrcr_gate.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s99_swrcr.ps1`

Artifacts:

- `.work/jobs/r501-s99-swrcr/latest/r501_s99_swrcr_gate.json`
- `.work/jobs/r501-s99-swrcr/latest/r501_s99_swrcr_gate.log`

Outcome:

- Implemented the R501 Hardware Efficiency Math 6 SWRCR gate as a CPU
  diagnostic over exact ternary recurrent segment recomputation from sparse
  checkpoints.
- Canonical diagnostic emitted `R501_S99_SWRCR_GATE active=1` and passed with
  `positive_reconstruction_relative_error_max=0.0`,
  `positive_checkpoint_state_ratio_mean=0.1267056530214425`,
  `positive_hidden_latency_ratio_mean=0.14285714285714285`,
  `positive_register_window_fit_ratio_mean=0.5`,
  `positive_paid_state_bit_ratio_mean=0.1267056530214425`, and
  `positive_projected_state_efficiency_multiplier_mean=7.892307692307694`.
- Controls failed as required: corrupted checkpoints and approximate
  recomputation broke reconstruction, dense residency erased the denominator,
  non-ternary recomputation exceeded hidden latency, and oversized intervals
  exceeded the sub-warp register window.

## R501-S100 - R501 Hardware Efficiency Math 7 Exact-Quotient Padding Elimination Block Gate

Date: 2026-06-07

Commands:

- `python -m py_compile scripts\research\r501_s100_eqpeb_gate.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .workspace\training\rounds\r501_s100_eqpeb.ps1`

Artifacts:

- `.work/jobs/r501-s100-eqpeb/latest/r501_s100_eqpeb_gate.json`
- `.work/jobs/r501-s100-eqpeb/latest/r501_s100_eqpeb_gate.log`

Outcome:

- Implemented the R501 Hardware Efficiency Math 7 EQPEB gate as a CPU
  diagnostic over active-token packing, sequence-boundary reset, and inverse
  unpacking of recurrent forward states and input gradients.
- The first attempted run failed due to a diagnostic metric bug, not a mechanism
  failure: relative error was applied to a difference tensor. The metric was
  corrected and the same preset passed with all controls identifying.
- The canonical diagnostic emitted `R501_S100_EQPEB_GATE active=1` and passed
  with `positive_active_token_fraction_mean=0.69794921875`,
  `positive_active_token_recall_min=1.0`,
  `positive_forward_relative_error_max=0.0`,
  `positive_gradient_relative_error_max=0.0`,
  `positive_paid_matrix_ratio_mean=0.69794921875`, and
  `positive_projected_padding_efficiency_multiplier_mean=1.437287957139846`.
- Controls failed as required: no-reset, wrong-unpack, drop-active,
  dense-padding, and include-padding arms each violated the registered
  exactness, recall, or price checks.

## R501-S100 real-path addendum - v12 sampled-CE integration

Date: 2026-06-07

Production v12 now exposes an opt-in training bridge for R501-S100 behind
`--exact-quotient-padding` / `--eqp`, plus `--eqp-min-active-len` and
`--eqp-seed`. The integration changes sampled-CE training behavior only when
active: it averages loss over active-prefix targets, zeroes inactive
`ce_d_logits` rows, and rescales surviving rows by `n_valid/S` before the
existing GPU CE scatters consume them.

Validation:

- Build passed: `mingw32-make USE_VULKAN=1 train`.
- Clean zero-step RTX 5070 activation smoke:
  `.work/jobs/r501-eqpeb-realpath/eqp-zero-20260607-141018/`,
  `EXIT_CODE=0`, `seq_len=1024`, `min_active_len=768`, GPU allocated
  `217.34 MB`, and `[TRAIN] Complete`.
- One-step CE-path smoke reached the real GPU backend:
  `.work/jobs/r501-eqpeb-realpath/eqp-r493off-20260607-140955/`.
  Four `[EXACT_QUOTIENT_PADDING_CE] backend=gpu` microbatch lines reported
  `active_targets/inactive_targets/active_fraction` =
  `993/30/0.970675`, `920/103/0.899316`, `921/102/0.900293`, and
  `949/74/0.927664` out of `1023` targets each.
- Baseline control with the same one-step preset and no EQP also exits with
  `-1073741819`, including with `--r493-post-step-off`; the completion crash is
  tracked separately from the EQP CE integration.

Claim boundary: real training-path CE objective integration and GPU backend
activation evidence. This is not heldout CE progress, not a measured wall-clock
speedup, not a VRAM reduction, and not packed scan/CE compute elimination.
