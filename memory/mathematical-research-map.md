# Mathematical Research Map

This file defines how research should be framed mathematically before it becomes
algorithm or code. Use it together with `memory/model-mental-map.md`: that file
draws the architecture, while this file defines the proof objects and
inequalities a round must respect.

## Base Objective

Let the target be original held-out next-token CE:

```text
L(theta) = E_{z ~ D}[ell(theta; z)].
```

The real training target is not local loss improvement. It is time to matched
quality:

```text
T_0(epsilon) = inf { t : L(theta_t) <= L_target + epsilon }.
```

For any mechanism `M`, all costs must be priced:

```text
T_M =
  T_build
  + T_score
  + T_train
  + T_failed
  + T_repair
  + T_verify
  + T_realize.
```

## Current Additional Gate: R438 GPU-Native Training Speed Frontier

R438 defines the current speed-frontier proof objects. They are projection-only
until real v12 GPU training-denominator gates pass.

The primary credit object is the lagged CE-impulse eligibility ledger:

```text
lambda_t = sum_d Phi_{t,t+d}^T ell_h(t+d)
tail <= E_t * rho^(H+1)/(1-rho)
```

It is admissible only if:

```text
C_recurrent_backward_saved >
  H * C_local_jvp
  + C_ring
  + C_tail_audit/K
  + C_mem.
```

The update-lagged Green lease is:

```text
g_rec(theta) = G_s z + r_s
||r_s|| <= e_fit + L_G ||theta - theta_s||
```

The gradient-rank quotient substrate is:

```text
g(theta) = B(theta) alpha + r
-g_original^T B alpha >= -epsilon_descent
```

The executable multiplicity packet gate is:

```text
g_C = n_C * g_rep + r_C
```

for state/transition signature cells `C` emitted by the real model path.

The phase-area dynamics gate is:

```text
T_effective =
  integral active_capacity(t) dt
  + release_debt
  + catchup_debt
  + verify
  + repair.
```

It is admissible only when old-function error remains within budget:

```text
old_function_error <= epsilon_old.
```

The first executable queue after R438 is:

1. A1 lagged CE-impulse eligibility GPU observer/path.
2. B1 executable multiplicity packet GPU signature gate.
3. C9 CE transaction block observer.
4. B10 gradient-rank quotient substrate observer.
5. C10 phase-area isoperimetric training observer.

CPU-only probes may support diagnosis. They do not close R438 training-speed
findings.

Implementation update: the first A1 bridge has now been landed and tested on the
real v12 GPU path. `H=4` failed the activation gate. `H=8` passed activation with
low residuals and applied all recurrent update layers, but ABBA heldout-progress
speed validation was speed-negative in the current embedding. The mathematical
object remains interesting, but a speed-positive A1 requires moving the ledger
before the dense recurrent backward is paid and keeping the realization
GPU-resident.

B1 implementation update: the first executable multiplicity packet signature
gate was also tested on the real v12 GPU path. The current state/transition hash
signature failed: high-mass packet cells had residual above original gradient
energy, and low-residual signatures lost nearly all mass. B1 should not be
promoted unless a different signature or residual object changes that
mass/residual frontier.

## Current Additional Gate: R419 Ternary Sufficient-Statistic Import

The dataset-import framing is admissible only as a CE-sufficient statistic or
certificate import. The current R419 gate is:

```text
R_TSSI =
  h_build
  + h_realize
  + h_verify
  + kappa_res * H_res/H_0
  <= 0.90
```

with the identifying checks:

```text
A_sign > A_random + margin
CE(theta_I) < CE(theta_ref) on held-out data
```

where `theta_I` is the ternary sign/zero import compiled from corpus moments in
a fixed SSM feature map. The residual model must still train under original CE.

Literal raw corpus import is a scoped no-go unless it reduces to a
CE-sufficient statistic, exact/certified quotient, or residual-preserving
certificate. Otherwise it is lookup, training in disguise, or insufficient
initialization.

2026-05-11 gate status: the first CPU-only identifying pre-gate, residual
original-CE pilot, deterministic `ssm_state` recurrent proxy pilot, and live
Vulkan SSM hidden-state trace gate passed as partial signals on available v12
byte-level real shards. On `wiki_1mb.bin` with `500000`
tokens, the best train/calib-selected bigram import reached held-out CE
`2.949221092857901` versus uniform CE `5.545177444479562` and random held-out
CE mean `5.722178622339609`. Weighted sign alignment was
`0.9986051321029663` versus random `0.45805180817842484`. The byte/bigram
residual pilot charged TSSI build time and measured target CE
`2.9492211233757843`: TSSI reached the target at `0.8049524002708495` seconds,
the best control reached it at `2.8539996999315917` seconds, target speedup vs
best control was `3.5455508909238374`, and same-budget CE advantage was
`0.2352878570289607`. The `ssm_state` proxy, a deterministic diagonal recurrent
state map, also passed on `500000` tokens with density `0.01` and scale `1.6`:
seed `777` measured target speedup `3.6053478120822366` and same-budget CE
advantage `0.05614244344347741`; seed `419` measured target speedup
`3.5931313221153545` and same-budget CE advantage `0.05664628606735311`. The
live Vulkan SSM trace gate repaired R395 hidden-state materialization from
`vulkan_fwd_h_cache`; the `10`-step batch-1 trace had `7864320 / 7864320`
nonzero state values, and the best `all`-layer import reached held-out CE
`5.346169927602639` versus random held-out CE mean `5.54442637453797`, weighted
sign alignment `0.9970957040786743`, and residual same-budget CE advantage
`0.19843119959677402`. A follow-up record-disjoint anti-memorization gate held
out whole trace records under rotations `0,2,4,6,8`, so each two-record block
was held out once. All five all-layer rotations passed; the CE margin versus
the best density-matched random control had min `0.189190727524743`, mean
`0.204293772528715`, and max `0.238266453598485`. `last` and `mean` layer
summaries also passed all rotations with weaker mean margins
`0.0327855562301441` and `0.013581927197886`. This is now a scoped paper
candidate for corpus moment import plus residual training over actual Vulkan
SSM states. It is not a production speedup or `validated_speed_finding` until
the signal is converted into an amortized online compiler or opt-in production
import path with full `R_TSSI` pricing.

2026-05-12 R420 closeout status: the first online-compiler continuation did
not convert TSSI into a production speed or training-impact paper claim. The
online compiler, live score accumulator, live selector export, and Python
selector consumer are positive support for the research object, but the direct
production selector-mask consumer was mixed/negative (`tssi=39.3419`,
`zero=39.0047`, `random=36.7838` held-out CE after one step). The live selector
artifact contains active dimensions plus aggregate per-dimension signed and
absolute scores, but no per-token/per-logit ternary commitments; therefore it
cannot realize a sign-bearing production import or honest sign-shuffled arm.
R420 also closed the active-subspace, stability-threshold, residual-sampling,
and simple corpus-atom extensions negative under their current gates. TSSI
remains a scoped corpus-statistic import paper object, not a speed finding.

A headline speed finding should satisfy:

```text
T_M(epsilon) <= 0.5 * T_0(epsilon)
```

with no quality tradeoff.

This `0.5 * T_0` inequality is the aspirational `2x` paradigm bar, not the
minimum material-gain bar. The implementation/research minimum remains:

```text
T_M(epsilon) <= 0.9 * T_0(epsilon)
```

with no quality tradeoff. A result between `0.5 * T_0` and `0.9 * T_0` must be
kept alive as a material candidate or next-gate finding when costs are priced
and the mechanism is project-owned.

## Quality Preservation Tests

The strongest result is exact objective equivalence:

```text
L_M(phi) = L(realize(phi)).
```

When exact equivalence is not available, a candidate needs a certificate that
the residual is harmless:

```text
L(realize(phi_M)) <= L_target + epsilon_quality.
```

Update-level arguments must separate useful CE direction from harmful residual:

```text
u_M = alpha * u_0 + b
```

and the residual must not spend more quality than the saved time is worth:

```text
-grad L(theta)^T b - curvature_penalty(b) >= -saved_time_budget_value.
```

## Useful Mathematical Objects

Prefer discoveries that produce one of these objects:

- exact quotient of repeated or equivalent CE terms;
- sufficient statistic for original CE gradient/progress;
- reversible coordinate transform of the data/objective;
- target-preserving parameter description;
- sparse exception certificate;
- target-containing subspace certificate;
- scaffold that removes initial gap while preserving full residual freedom;
- conserved quantity or monotone progress measure;
- low-cost proof that a large active surface is unnecessary.

## Three-Layer Round Model

Every new candidate should be expressible on all three layers.

### 1. Macro Layer

```text
A_theta = realized artifact
M = proposed mechanism
T_M = T_build + T_score + T_train + T_failed + T_repair + T_verify + T_realize + T_memory
```

Question:

```text
Where does M enter the native v12 training/inference artifact, and what cost
term does it reduce?
```

### 2. Conceptual Layer

Classify the mechanism before validating it:

```text
M in {content, CE witness, dynamics, realization, runtime support, new substrate}
```

Positive research normally needs `content`, `CE witness`, `dynamics`,
`realization`, or `new substrate`. A pure runtime-support mechanism is a
product optimization unless it directly unlocks one of those objects.

### 3. Mathematical Layer

A candidate must name one of:

```text
L_M(phi) = L(realize(phi))
```

or

```text
L(realize(phi_M)) <= L_target + epsilon_quality
```

or an update-level certificate:

```text
u_M = alpha * u_0 + b
-grad L(theta)^T b - curvature_penalty(b) >= -saved_time_budget_value
```

Then it must price:

```text
T_M(epsilon) / T_0(epsilon)
```

and clear at least:

```text
T_M(epsilon) <= 0.90 * T_0(epsilon)
```

## Candidate Families And Inequalities

### Lossless Latent Transcript

```text
r_len * m_lat * rho_hard + h_codec <= 0.5
```

Needs exact reversible decoding and original-text CE through the codec.

### Sparse Active-Surface SSM

```text
(a + o_route + o_balance) / q_sparse <= 0.5
```

Needs original-CE progress retention and bounded route/balance overhead.

### Genotype + Exception Ledger

```text
p_phi + c_E * s_E + h_G <= 0.5
```

Needs target-preserving short description and sparse exceptions.

### Corpus Scaffold + Residual CE

```text
(1 - gamma) * kappa + h_S <= 0.5
```

Needs scaffold gap removal, full residual parameter freedom, and priced build
cost.

### Exact Multiplicity Quotient

```text
U/N + h <= 0.5
```

Needs exact objective-identical CE terms, not approximate similarity.

## Current Post-R415 Proof Objects

These are the latest active or recently landed objects that researchers must
not accidentally rediscover as new.

### R434 Protocol Speed Gates

R433 is historical process context only. R434 supersedes it as the
protocol-compliant source of truth for this speed-candidate family.

First gate-order object:

```text
g_exact(segment)
  = P_tube(A_segment, h_0, h_T, interior_stats) + r_sparse(segment)
```

with:

```text
residual_fraction <= 0.10
paid_ratio < 0.90
```

This is `Interior Cotangent Tube With Sparse Residual Settlement`. It reopens
recurrent credit through interior SSM cotangent tubes, not endpoint summaries or
boundary-anchor-linear reconstruction.

Second recurrent denominator object:

```text
g(S) = P_theta(A_S) + r_S
```

This is `Recurrent Cotangent Sufficient Statistics For Exact CE Work Removal`.
It is admissible only if the statistic reconstructs CE cotangent contributions
with low residual, material accepted mass, and refresh/replay/audit/memory costs
below dense recurrent backward work.

Quotient object:

```text
g_C = n_C * g_rep + r_C
L(theta) = L_Q(theta) + R_Q(theta)
```

This is `State-Conditioned CE Quotient With Residual Debt`. It can remove work
only when model-visible state signatures create high-multiplicity original-CE
cells after residual verification, replay, repair, failed-cell, and memory costs
are charged.

Ternary escrow object:

```text
m_i - B_i * rho_i > 0
```

This is `Ternary Sign-Lattice Escrow / Ternary Commitment Escrow`. It commits
sign/zero coordinates only when original-CE margin dominates a certified repair
radius; otherwise coordinates remain trainable.

Status: `validated_for_gate` theory round only. No measured speedup, baseline
change, VRAM/RAM change, implementation, GPU validation, warmup, or production
claim exists from R434.

### CE Boundary-Flux Sheaf

```text
g_original,c(theta) = B_c(theta) a_c + e_c(theta)
sum_c ||e_c(theta)||_* <= epsilon_grad
```

Material condition:

```text
f_flux * (1 - kappa_flux)
  > 0.10 + h_build + h_score + h_verify + h_dispatch + h_refresh + h_mem
```

Status: endpoint-only Vulkan gate landed negative after R415/R417 live-path
repairs. The gate now observes nonzero SSM transition-credit energy through
`dA_per_step`, but boundary/pre-boundary endpoint mass stayed below 1% on
bounded real-path smokes and did not clear the 10% material bar. Do not promote
the endpoint-only realization. The broader sheaf can reopen only as a
segment-level adjoint/residual reconstruction object with false-equivalence,
false-skip, refresh, dispatch, and memory pricing.

### Certified Gradient-Null Witness Object

```text
||g_i(theta)|| <= ||g_i(theta_0)|| + H_i rho
b_i + H_i rho_i <= tau_i
sum_i tau_i <= epsilon_grad
```

Status: cheap tensor-group gate landed negative after R415/R417 live-path
repairs. The gate observes exact-null full-gradient tensor groups with zero
omitted-gradient energy, but raw parameter-element coverage was biased by
sparse vocabulary/embedding tensors. Corrected work-priced saved fraction is
about `0.0462`, below the 10% material bar before build, verify, refresh, or
memory overhead. Reopen only with finer trainable-context or recurrent-segment
certificates.

### CE-Normal Ternary Chamber

```text
eta K ||g_B||_inf <= rho_B - sigma_B
```

Material condition:

```text
f_inactive * kappa_backward
  > 0.10 / lambda_train + h_margin + h_block_route + h_verify + h_memory
```

Status: opt-in GPU force-bound audit landed after R416, but the current early
from-scratch Vulkan path is non-identifying for this object. Checked
`test_data.bin`, `text_data.bin`, and `wiki_1mb.bin` smokes reported
`saturated=0` and `candidates=0` across GPU-owned ternary groups, so all
audited shadows remained in the stochastic zone `|shadow| < 1`. Do not promote
or reject the theory from this state. Reopen only when a mature checkpoint or
budgeted warmup first shows `candidate_work_fraction > 0`, then add
missed-boundary replay, optimizer-state drift accounting, original-CE harm
audit, and full overhead pricing.

### Boundary-Cotangent Credit

```text
g_exact = exact recurrent transition-credit cotangent
u_B = reconstruction from boundary/pre-boundary plus endpoint anchors
```

Gate contract:

```text
residual_fraction = ||g_exact - u_B||^2 / ||g_exact||^2
R_paid = anchor_fraction + verify_fraction + overhead + residual_fraction
```

Required:

```text
residual_fraction <= 0.10
R_paid < 0.90
CE_descent_per_paid_work > 1
```

Status: opt-in GPU audit landed after R416 as
`--boundary-cotangent-credit-gate`. The first boundary-anchor-linear
realization is identifying but negative. On repaired Vulkan `dA_per_step`,
compact anchors are present, but cosine is low and residual exceeds the exact
energy on checked corpora. `wiki_1mb.bin` three-step final:
`anchor_fraction=0.007568`, `cosine=0.137153`,
`residual_fraction=1.150172`, `priced_ratio=2.157741`,
`unverified_priced_ratio=1.157741`, `pass=0`. This closes only the
boundary-anchor-linear realization. Reopen the broader forward-eligibility
family only with a different eligibility basis, segment-level adjoint tube, or
residual-sparse repair object that first reduces residual below budget before
verification and memory costs are priced.

### Variance-Priced Credit Allocation

```text
g = sum_i g_i
E[ghat] = g
p_i proportional to ||g_i|| / sqrt(c_i)
```

Gate contract:

```text
descent_per_paid_work
  = max(0, alignment - variance_proxy)
    / (sampled_fraction + score_fraction + verify_fraction + overhead)
```

Required:

```text
descent_per_paid_work > 1.111
```

Status: opt-in GPU audit landed after R416 as
`--variance-priced-credit-gate`. The first audit is identifying and shows that
the sampled transition-credit estimator can stay well aligned on real corpora.
Default paid pricing fails because score and exact verification are each charged
as a full pass. `wiki_1mb.bin` three-step final with default pricing:
`sampled_fraction=0.247206`, `alignment=0.991684`,
`variance_proxy=0.015712`, `priced_work=2.247206`,
`descent_per_paid_work=0.434304`, `pass=0`. With
`score_fraction=0` and `verify_fraction=0`, the same run reaches
`descent_per_paid_work=3.948010`, `pass=1`. This is a conditional estimator
signal, not a deployed speedup. Reopen only by deriving cheap or amortized
scoring plus cheaper verification/certification while preserving original CE.

### R436 Amortized CE Martingale Training Bridge

The R436 bridge promotes the cheap-verifier line into an opt-in training
operator. Let `g` be the exact recurrent transition-credit term and `ghat_S`
the norm-priced Horvitz-Thompson sampled estimator. The bridge may apply the
sampled recurrent update only when the local martingale certificate clears:

```text
descent_per_paid_work(ghat_S) >= tau

audit_debt(ghat_S, g) + residual_debt
  <= epsilon * certified_progress
```

The live implementation applies this only behind
`--amortized-ce-martingale-training`, with exact fallback. The R436 real stack
activation smoke passed with applied sampled recurrent credit, but it is only a
training-entrypoint proof:

```text
PASS_R436_REAL_TRAINING_STACK_ACTIVATION
```

It does not yet establish:

- matched-target wall-clock speedup;
- dense recurrent backward-work removal;
- official 7B projection movement;
- default-path promotion.

The remaining A-family mathematical frontier is still to expose a moment,
Green-operator, or segment-tube object whose residual settlement removes dense
recurrent backward work under original CE.

### R437 Mathematical Training Speed Transform

R437 sharpens the post-R436 frontier. The current martingale bridge activates
after dense recurrent backward has already been paid, so its speed projection
cannot become wall-clock speed until the recurrent credit object is moved before
or instead of dense backward.

Primary new gate-order object:

```text
lambda_t = B_theta(q_t) a_t + r_t

g_theta = g_ASQI + r_theta
```

This is `Adjoint-State Quotient Integrator`. It is admissible only if a
forward-owned adjoint basis `B_theta(q_t)` and coefficient integrator `a_t`
reconstruct heldout exact recurrent cotangents with bounded residual, while
dense backward is used only for calibration, sparse audit, or residual
settlement.

Gate contract:

```text
residual_fraction_heldout <= 0.20   # weak first trace gate
residual_fraction_heldout <= 0.10   # promotion target

R_ASQI =
  (C_basis_forward
   + C_coeff_integrator
   + C_residual_settlement
   + C_audit
   + C_memory)
  / C_bwd_rec
  < 0.90
```

Secondary R437 gate objects:

```text
V_theta(h_t, t)
  ~= E[future CE | h_t, t]

delta_t =
  V_theta(h_t, t)
  - ell_t(theta)
  - V_theta(F_theta(h_t, x_t), t+1)
```

This is `CE Hamiltonian Potential Field`. It is high-uncertainty
`validated_for_gate` only if the Bellman-gradient residual is audited under
original CE and fitting/refresh does not become dense or teacher-like.

```text
D_lower(U) =
  -g(theta)^T U
  - 0.5 * beta ||U||^2
  - interaction_debt(U)
  - residual_audit_debt(U)
```

This is `Curvature-Bounded Macro-Action Integrator`. It can reduce paid step
count only when a macro action's original-CE descent receipt beats the dense
micro-step path after curvature, interaction, rejected-action, and verification
costs.

`Ternary Cotangent Event-Time Measure` remains a `gate_hypothesis`, not an
active validated-for-gate finding, until a forward-owned event measure predicts
low-credit regions without dense backward and with false-low-event cotangent
energy below budget.

No R437 item is measured, implemented, production-ready, a default-path change,
or an official 7B projection update. The queue consequence is to gate ASQI
first before returning to R436 long speed validation.

### SSM Credit-Time Equalized Dynamics

```text
c_j = sum_{k=0}^{H} |lambda_j|^(2k)
Delta theta = -eta G_c^(-1/2) grad L(theta)
```

Descent contract:

```text
L(theta + Delta theta)
  <= L(theta) + g^T Delta theta + (beta / 2) ||Delta theta||^2
```

Status: opt-in real-path implementation exists, but the default cheap gate is
negative on the current fixture:

```text
net_descent_gain = 0.717490
priced_ratio = 1.393748
```

Do not promote this parameterization. Reopen only with a new normalization,
trust region, or parameterization that clears `R_time <= 0.90`.

## Scoped No-Go Boundary

Strict validation cannot be obtained for teacher-free training-speed mechanisms
that lack an exact empirical-risk quotient, exact target-preserving parameter
description, exact scaffold gap-removal certificate, exact target-containing
subspace, or equally strong proof object, while building model-dependent objects
only from the same signal available to baseline training.

Such methods either:

1. discard/reweight/compress CE information without proving the residual
   harmless, leaving quality risk; or
2. compute enough information to certify harmlessness, leaving certificate cost
   unbounded below the speed budget.

This no-go is scoped, not global. It tells future research what kind of proof
object is required.

Post-R415 refinement:

- copy-current proof routing is not enough without real route mass;
- raw duplicate quotients are not enough without exact model-visible objective
  equivalence;
- proxy objectives with later repair are not enough without exact equality or a
  bounded harmless residual;
- local throughput is not a speed finding unless it lowers time-to-matched
  original-CE quality after all costs.

## Research Discipline

Mathematics first does not mean code can fail mysteriously later. If the math is
truly correct and fully embedded into the real artifact boundary, implementation
failure should be a bug, missing assumption, or wrong embedding, not a conceptual
surprise.

Therefore every theoretical result must include:

- artifact boundary;
- exact objective or certified residual;
- priced denominator;
- implementation embedding;
- smallest identifying gate;
- known ways it can be killed.
