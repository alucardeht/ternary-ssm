# Model Mental Map

This is the compact architecture map researchers should carry into every new
round. It is not a replacement for source code, ledgers, or measured baselines.

Current state: the active implementation is the native `v12` C/Vulkan training
path. The memory-fit problem is largely solved for the research target; the open
frontier is still time-to-quality and original-CE-preserving capability gain.

## Macro Architecture

This view is for orienting a round before choosing an aspect.

```mermaid
flowchart TD
  "Corpus / generated training stream" --> "Tokenizer and fixed sequence batches"
  "Corpus / generated training stream" --> "Optional corpus-statistic compiler"
  "Optional corpus-statistic compiler" --> "Ternary sign/zero import artifact"
  "Ternary sign/zero import artifact" --> "Native v12 training executable"
  "Tokenizer and fixed sequence batches" --> "Native v12 training executable"
  "Native v12 training executable" --> "Vulkan GPU resident training surfaces"
  "Native v12 training executable" --> "CPU-owned or host-visible SSM dynamics surfaces"
  "Vulkan GPU resident training surfaces" --> "Forward and backward sampled/original CE path"
  "CPU-owned or host-visible SSM dynamics surfaces" --> "SSM recurrent transition and dynamics gradients"
  "Forward and backward sampled/original CE path" --> "Optimizer / staged update policy"
  "SSM recurrent transition and dynamics gradients" --> "Optimizer / staged update policy"
  "Optimizer / staged update policy" --> "Ternary or compressed realized artifact"
  "Ternary or compressed realized artifact" --> "Local inference artifact"
  "Research proof objects and ledgers" --> "Native v12 training executable"
  "Research proof objects and ledgers" --> "Optimizer / staged update policy"
```

Operational notes:

- Training is GPU-based by default. CPU-only probes can support diagnosis, but
  they do not close a training finding unless explicitly scoped that way.
- R419 adds an optional pre-training research object: a corpus-statistic
  compiler that may write ternary sign/zero commitments before residual original
  CE training. Its CPU pre-gate and first residual pilot passed on byte-level
  real shards in a readout-like byte/bigram scope; a deterministic recurrent
  `ssm_state` proxy also passed. The live Vulkan SSM trace and record-disjoint
  rotations make it a scoped paper candidate for corpus-statistic import, but
  it is not a default path, not a production GPU speedup, and not admissible if
  it becomes hidden lookup or iterative CE training in disguise.
- Current Vulkan smokes commonly show about `728 MB` VRAM for the small
  `test_data.bin` one-step fixture; this is not the full 7B training budget.
- Experimental research paths must stay opt-in, emit activation logs, preserve
  the default baseline path, and compare against the latest relevant baseline.

## Conceptual Architecture

This view is for understanding what the model is trying to become.

```mermaid
flowchart LR
  "Text / byte stream" --> "Embedding and compact input representation"
  "Embedding and compact input representation" --> "Ternary SSM recurrent core"
  "Ternary SSM recurrent core" --> "State trajectory h_0...h_T"
  "State trajectory h_0...h_T" --> "Predictive channels and auxiliary memory"
  "Predictive channels and auxiliary memory" --> "Output logits"
  "Output logits" --> "Original next-token CE"
  "Original next-token CE" --> "Credit assignment object"
  "Credit assignment object" --> "Training dynamics"
  "Training dynamics" --> "Realized local model"
```

Research should not confuse these layers:

- A content object changes what information is presented or compressed.
- A CE witness object proves that some original-CE work can be removed,
  discharged, or replaced.
- A dynamics object changes how the same original-CE objective moves the model.
- A runtime object changes execution cost, but is not a research finding unless
  it changes time-to-target or unlocks a deeper project-owned mechanism.

## Mathematical Architecture

This view is for deriving candidates before agents narrow onto a local trick.

```mermaid
flowchart TD
  "Dataset D and token examples z" --> "Objective L(theta) = E_D ell(theta; z)"
  "theta = realized trainable artifact" --> "Objective L(theta) = E_D ell(theta; z)"
  "SSM recurrence h_t = F_theta(h_{t-1}, x_t)" --> "Logits y_t = G_theta(h_t)"
  "Logits y_t = G_theta(h_t)" --> "Objective L(theta) = E_D ell(theta; z)"
  "Gradient / cotangent g = grad_theta L" --> "Update u"
  "Objective L(theta) = E_D ell(theta; z)" --> "Gradient / cotangent g = grad_theta L"
  "Update u" --> "New artifact theta'"
  "New artifact theta'" --> "Quality target L(theta') <= L_target + epsilon"
  "Mechanism M" --> "Build score train fail repair verify realize memory costs"
  "Build score train fail repair verify realize memory costs" --> "T_M(epsilon)"
  "Quality target L(theta') <= L_target + epsilon" --> "T_M(epsilon)"
```

Canonical speed inequality:

```text
T_M(epsilon) <= 0.90 * T_0(epsilon)
```

Headline inequality:

```text
T_M(epsilon) <= 0.50 * T_0(epsilon)
```

The headline `2x` threshold is not the discard threshold. A no-tradeoff
`1.2x`, `1.5x`, or `1.7x` path is material if it clears the `10%` floor and all
costs are priced.

## Current Concrete Baseline

Use measured/projected numbers only through `memory/baselines.md`.

- `R150`: canonical 7B / 60B-token projection is about `495d`.
- `R150`: lazy refresh projection is about `165-247d`.
- Recent one-step RTX 5070 Vulkan fixture baseline: `loss=11.7143`,
  `VRAM=728 MB`; exact source depends on the active implementation closeout.
- `R415 SSM Credit-Time Equalized Dynamics` is landed opt-in but negative on
  the cheap default gate: `priced_ratio=1.393748`; no baseline speedup is
  claimed.
- `R415 CE Boundary-Flux Sheaf` endpoint-only transition-credit gate is landed
  opt-in but negative after the Vulkan SSM adjoint repair: endpoint
  `energy_fraction=0.004146-0.006474`, `pass=0`; no baseline speedup is
  claimed.
- `R415 Certified Gradient-Null Witness` tensor-group gate is landed opt-in and
  identifying, but negative after work pricing: raw null element fraction is
  about `0.108`, corrected saved work is about `0.0462`, `pass=0`; no baseline
  speedup is claimed.
- `R416 CE-Normal Ternary Chamber` force-bound audit is landed opt-in but
  non-identifying on current early from-scratch smokes: checked GPU-owned
  ternary shadows remained stochastic with `saturated=0` and `candidates=0`.
- `R416 Boundary-Cotangent Credit` boundary-anchor-linear audit is landed opt-in
  and identifying on repaired Vulkan `dA_per_step`, but negative: compact
  anchors do not reconstruct recurrent transition credit (`wiki_1mb.bin`
  three-step final `residual_fraction=1.150172`, `priced_ratio=2.157741`,
  `pass=0`).
- `R416 Variance-Priced Credit Allocation` audit is landed opt-in and
  identifying. The current full-score/full-verify realization is negative
  (`wiki_1mb.bin` three-step final `descent_per_paid_work=0.434304`,
  `pass=0`), but a zero score/verify sensitivity gives
  `descent_per_paid_work=3.948010`, so the estimator math remains conditionally
  interesting if scoring and verification can be made cheap or amortized.
- `R419 Ternary Sufficient-Statistic Import` is a scoped paper candidate, not a
  speedup claim:
  on `wiki_1mb.bin` byte-level split, a bigram-feature ternary import reached
  held-out CE `2.949221092857901` versus uniform CE `5.545177444479562` and
  random held-out CE mean `5.722178622339609`, with weighted sign alignment
  `0.9986051321029663`. A residual original-CE pilot then measured TSSI target
  time `0.8049524002708495s` versus best control target time
  `2.8539996999315917s`, with same-budget CE advantage
  `0.2352878570289607`. A deterministic `ssm_state` recurrent proxy also passed
  at `500000` tokens in two seeds: target speedup around `3.6x` versus best
  control and same-budget CE advantage around `0.056` nats. A live Vulkan SSM
  trace gate then materialized `vulkan_fwd_h_cache` into R395 traces and found
  best `all`-layer held-out CE `5.346169927602639` versus random held-out CE
  mean `5.54442637453797`, weighted sign alignment `0.9970957040786743`, and
  same-budget CE advantage `0.19843119959677402`. Record-disjoint live-trace
  rotations passed all held-out record pairs, with all-layer CE margin versus
  best random min mean `0.204293772528715`. No production GPU speedup, baseline
  change, or integrated model-native import path is claimed. `R420` attempted
  to turn this into training impact through an online compiler, live selector,
  direct production selector-mask consumer, active subspace, stability
  thresholds, residual active sampling, and CE-certified corpus atoms. It
  closed with no new speedup or training-impact paper claim: the selector-mask
  production path was mixed/negative, the selector artifact was not
  sign-bearing enough for an honest production sign-shuffled arm, and the
  remaining extension gates failed their closeout thresholds.

## Current Research Frontier

The next research or implementation round should start from the post-R415 state:

- `CE Boundary-Flux Sheaf` endpoint-only form is closed negative; the broader
  object can reopen only as a segment-level adjoint/residual sheaf with priced
  false-discharge harm.
- `Certified Gradient-Null Witness Object` tensor-group form is closed
  negative; it can reopen only with finer trainable-context or recurrent-segment
  certificates.
- `Forward Eligibility / Boundary-Cotangent Credit` boundary-anchor-linear form
  is closed negative; the family can reopen only with a different eligibility
  basis, segment-level adjoint tube, or residual-sparse repair object whose
  residual clears budget before verification and memory costs are priced.
- `Variance-Priced Credit Allocation` is pricing-blocked rather than dead:
  reopen only as a cheap-score/cheap-verify certificate path, then rerun exact
  gradient audit and real wall-clock pricing.
- `CE-Normal Ternary Chamber` is blocked on mature-chamber evidence; it can
  reopen only after the real GPU path shows nonzero deterministic chamber work.
- `SSM Credit-Time Equalized Dynamics` is implemented and activated, but the
  current default scaling failed the cheap real-path material gate.

The immediate frontier is therefore not "make another local optimizer tweak".
It is:

1. build R434 original-CE speed gates that settle removed recurrent cotangent,
   quotient, or ternary commitment work into CE receipts and residual debt
   before a speed claim is counted;
2. find or validate a cheaper original-CE witness object;
3. find a dynamics change that improves original-CE progress per priced time
   without proxy loss or hidden repair;
4. make any content/quality route translate into faster matched target, not just
   nicer local data.
5. keep TSSI framed as a scoped corpus-statistic import paper object unless a
   genuinely new sign-bearing model-native artifact appears; do not continue
   the closed R420 selector-mask, stability-threshold, residual-sampling, or
   simple-atom paths as active training-impact items.

R434 protocol speed refinement:

- `R433 Original-CE Speed Carriers` is preserved only as historical process
  context. R434 supersedes it as the protocol-compliant source of truth.
- `Interior Cotangent Tube With Sparse Residual Settlement` is the first
  gate-order speed object. It tests whether exact segment cotangents can be
  represented by an interior SSM tube plus sparse residual settlement with
  `residual_fraction <= 0.10` and `paid_ratio < 0.90`.
- `Recurrent Cotangent Sufficient Statistics For Exact CE Work Removal` is the
  second recurrent denominator gate. It requires `g(S) = P_theta(A_S) + r_S`
  with bounded residual, material accepted mass, and low refresh cost.
- `State-Conditioned CE Quotient With Residual Debt` is the current quotient
  gate. It pays representative original-CE work for model-visible state cells
  and charges all residual verification, replay, repair, and memory costs.
- `Ternary Sign-Lattice Escrow / Ternary Commitment Escrow` is the current
  ternary-artifact gate. It commits sign/zero coordinates only when original-CE
  margin dominates certified repair radius: `m_i - B_i * rho_i > 0`.
- No R434 result is measured, landed, production-ready, or a baseline/projection
  change. All four positive items are `validated_for_gate`.

R436 real training update:

- `Amortized CE Martingale Verifier` is now an opt-in real training bridge, not
  only an observer. The flag `--amortized-ce-martingale-training` lets the live
  martingale-priced transition-credit sample affect the recurrent `A/bA/WA`
  update when the local gate clears, with exact fallback and unchanged default
  behavior.
- `.workspace/training/rounds/r436.ps1 -Mode real` is the current R436 training
  entrypoint. It composes the martingale bridge over R427 integrated staged
  growth, checkpoint lift, full-width continuation, readout quotient,
  variance-priced support, online TSSI support, and credit observers.
- The activation smoke passed, but it is not a matched-target speedup, not a
  7B projection update, and not a VRAM/RAM or inference baseline change.
- The A-family recurrent cotangent moment, Green operator, and segment-tube
  objects remain support/observer-only until a separate implementation removes
  dense recurrent backward work with residual settlement.

R437 mathematical speed update:

- R437 identifies the current denominator blocker as placement: R436 computes
  and certifies sampled recurrent credit after dense recurrent backward has
  already been paid.
- The new primary speed object is `Adjoint-State Quotient Integrator`, which
  attempts to compute recurrent cotangent as a forward-owned adjoint basis plus
  coefficient integrator:
  `lambda_t = B_theta(q_t) a_t + r_t`.
- This keeps dense backward as calibration/audit/residual settlement rather
  than the default path. It is only `validated_for_gate`, not measured or
  implemented.
- Secondary R437 transforms are `CE Hamiltonian Potential Field` and
  `Curvature-Bounded Macro-Action Integrator`; `Ternary Cotangent Event-Time
  Measure` is only a gate hypothesis.

## Researcher Preflight

Every round brief should point researchers here and force them to restate:

1. macro path: where the proposed mechanism enters the artifact;
2. conceptual path: whether it is content, witness, dynamics, runtime, or a
   different substrate;
3. mathematical path: what objective, residual, denominator, and realization map
   it changes;
4. latest baseline: before/after time, VRAM/RAM, and quality target;
5. active post-implementation delta from the previous round.
