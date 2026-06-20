# R15 — Training as Compilation: Round Summary
**Date:** 2026-03-07 | **Agents:** paradigm-breaker + math-paradigm-breaker

## The Central Insight ("Square Wheel")

Ternary inference 1B: 250MB, CPU, additions only. Near-zero compute.
Ternary training 1B: 6–14 bytes/param, 100B tokens, iterative backprop. **Absurd.**

We are using continuous optimization machinery to solve a fundamentally discrete problem:
deciding, for each weight, whether it should be -1, 0, or +1.

**The compilation analogy:**
- Training = compilation
- Dataset = source code
- Model = binary

A compiler does not run the program 100B times adjusting parameters.
It statically analyzes structure → produces optimal binary in one pass.
Training should be analogous: statistical analysis of data → ternary weights directly.

## Key Mathematical Result

Both agents converged on the same finding:

> **Second-order statistics (covariances) of the data are sufficient to determine
> optimal ternary weights per layer via closed-form solve — without gradient descent.**

The entire SGD + STE apparatus is **CONVENTION**, not mathematical necessity.

What is required: final weights in {-1,0,+1}, SSM structure, portability.
What is conventional (and removable): continuous relaxation, global backprop, 100B+ token iterations, STE.

## Techniques Identified

### T1: Layer-Wise Covariance Solve (LCS) — PRIORITY 1
**Source:** math-paradigm-breaker Path 2 + paradigm-breaker pseudoinverse
**Viability:** HIGH | **Token reduction:** ~10,000×

Closed-form per-layer weight recovery:
```
W_L* = Σ_{y,h} · Σ_{h,h}^{-1}    (population regression coefficient)
```

For Mamba-2 input-dependent parameters:
- `W_B*`: regress innovations `Δh_t = h_t - A(x_t)h_{t-1}` on `x_t`
- `W_C*`: regress `y_t` on `h_t`
- `W_A*`: log-linearize `A_k(x) = -exp(W_A·x)` → regress log-ratio on `x_t`

Single forward pass to accumulate statistics. Solve per layer: O(d³). No backprop.

**Required experiment (GO/NO-GO for all R15):**
Take a 6M model trained with gradient descent → compute LCS weights from 100K tokens → compare:
- Cosine similarity between W_gd and W_lcs per layer
- Loss of W_lcs vs W_gd on validation set
- If similarity > 0.7 AND loss < 110% of W_gd: path-independence confirmed → pipeline works

### T2: Compressed Sensing / LASSO Recovery — PRIORITY 2
**Source:** math-paradigm-breaker Path 3
**Viability:** MEDIUM-HIGH | **Token reduction:** ~50,000×

Ternary weights are S-sparse (S ≈ 0.4N). CS theory guarantees exact recovery of S-sparse vector
from O(S log N) linear measurements — far fewer than 100B tokens.

Theoretical minimum tokens (from CS bounds):
- 6M config: ~200K tokens
- 7B config: ~1.6M tokens

Complementary to T1: use LCS to get float estimate, then LASSO to project to ternary constraint.

**Immediate test (no training needed, ~1h):** Verify RIP condition on WikiText-103.
If restricted isometry constant δ_{2S} < 0.41 (Candès bound) → exact recovery guaranteed.

### T3: Koopman Operator / Extended DMD — PRIORITY 3
**Source:** math-paradigm-breaker Path 4
**Viability:** MEDIUM | **Token reduction:** ~1,000×

Key insight: Extended DMD with lifted state `z̃_t = [h_t; exp(τ_t); W_B·x_t; ...]` converts
Mamba-2's nonlinear input-dependent dynamics into a **linear** system in the extended space.
Then DMD identifies the Koopman operator from data in a single eigendecomposition.

**Critical difference from N4SID (which failed):** N4SID requires time-invariant linear systems.
Mamba-2 has input-dependent A, B, C. The Koopman lifting absorbs this nonlinearity.

Bonus: using Koopman eigenvectors as a basis for T1 makes covariance matrices diagonal → better conditioning.

### T4: Analytic Compilation Pipeline (ACP) — THE COMPILER
**Source:** Synthesis of T1+T2+T3

```
Step 1 — K-EDMD: O(T·d̃²)        → Koopman eigenstructure, W_A init
Step 2 — LCS in Koopman basis: O(d̃³/layer)  → W_B*, W_C* in float
Step 3 — CS-LASSO: O(mn·log(1/ε)/layer)     → ternary assignment
Step 4 — Verification pass: O(T_small)        → flip ambiguous weights

TOTAL: O(T·d² + n_layers·d³), ONE PASS, ZERO gradient descent, ZERO backprop
Token requirement: ~10M (vs 100B current) → 10,000× speedup
```

### T5: Spectral Moment Matching — ALTERNATIVE
**Source:** math-paradigm-breaker Path 1 | **Viability:** MEDIUM
Match spectral distribution (singular values) of weight matrices to data operator,
using free probability / random matrix theory. Single-pass, one eigendecomposition.

### T6: Discrete Optimal Transport — MOONSHOT
**Source:** math-paradigm-breaker Path 5 | **Viability:** SPECULATIVE
If the conjecture holds that the global OT plan factorizes into per-weight partitioning problems,
training reduces to O(T log T) sorting operations. No gradient, no backprop. Needs theoretical development.

## Implementation Order

| Priority | Action | Cost | Output |
|----------|--------|------|--------|
| **1** | **GO/NO-GO experiment** | 20 min, 100K tokens, 6M config | Confirms or refutes path-independence hypothesis |
| **2** | Implement LCS (T1) | ~1 day coding, 1 day validation | First gradient-free ternary layer training |
| **3** | Test RIP for CS-LASSO (T2) | 1h, no training | Validates T2 feasibility |
| **4** | Implement ACP (T4 = T1+T2+T3) | ~3 days | Full analytic compilation pipeline |
| 5 | Thermodynamic SA | ~1 day | Fallback: reduces VRAM, still iterative |

## Relationship to R14

R14 techniques (DSST, STCD-MI, Memory Shedding) solve **VRAM** but not **training time**.
They optimize the square wheel — they don't replace it.

**If R15 T1 experiment succeeds:** R14 techniques become secondary (ACP eliminates the iteration).
**If R15 T1 experiment fails:** R14 becomes Plan B (still valuable for VRAM reduction).

## What Was Already Tried and Failed (Do Not Retry)

- IFCD: direct gradient flip without accumulation → does not converge (noise dominates)
- Hebbian: assumes weight symmetry → SSM is asymmetric, does not apply
- N4SID linear SysID: blocked by input-dependent A,B,C in Mamba-2
- SysID via beam search: N forward passes per step → prohibitive overhead
- 25 LEGO flags: 3.5% success rate
- R13 incremental techniques (ADMM, Gumbel-STE, Kalman Hessian): improved quality but regressed throughput
