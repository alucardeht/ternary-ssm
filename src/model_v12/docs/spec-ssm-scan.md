# Specification: src/core/ssm_scan.c

Version: 1.0
Status: READY FOR IMPLEMENTATION
Depends on: architecture.md
Implements: architecture.md §2 (SSM Scan), §5 (GP-KL Welford), §4 partial (MTP state projection)

---

## 1. Scope and Boundaries

This file owns exactly one responsibility: the SSM recurrence kernel and its
direct derivatives. It does NOT own:

- in_proj / out_proj (ternary matmul — separate file)
- conv1d (separate file)
- GP-KL loss aggregation (separate file reads the Welford structs this file fills)
- MTP loss computation (separate file; this file only provides the `A^k` power application)

What this file produces:

- `y[t] = C @ h[t]` for every t in the sequence
- Updated hidden state `h` after the full sequence (for autoregressive inference)
- Welford moment accumulators filled in a single pass (no second pass needed)
- Gradient tensors for dA, dB, dC, dx when called from the backward path

---

## 2. Mathematical Derivation

### 2.1 Base Recurrence

The architecture defines (architecture.md §2):

```
h[t] = A ⊙ h[t-1] + B * x[t]      (element-wise ⊙, A diagonal)
y[t] = C @ h[t]
```

`A` is a diagonal matrix stored as a 1-D vector of length `d_state`. The
element-wise product eliminates all off-diagonal multiplications, reducing the
per-step cost from O(d_state^2) to O(d_state).

### 2.2 Input-Dependent dt (Mamba-2 Style)

`A` in the stored form is `A_log` (log-space, negative values for stability).
At each time step a scalar `dt[t]` (computed by the caller, e.g. via a linear
projection of x[t]) modulates the effective decay:

```
A_eff[t][i] = exp(dt[t] * A_log[i])       for i in 0..d_state-1
```

When `dt` is NULL the caller signals a time-invariant A: compute `A_eff[i] =
exp(A_log[i])` once before the loop and reuse it for every step.

### 2.3 Full Per-Step Update

```
// for each state dimension i:
h[i] = A_eff[t][i] * h[i]  +  B[i] * x[t]
```

`B` is input-dependent in the general case: `B[t][i]` is produced by a
separate projection. When `B` is time-invariant (static mode), the pointer
points to a single row of length `d_state` and the stride argument is 0.

### 2.4 Output Projection

```
y[t][j] = sum_i  C[j][i] * h[i]          j in 0..d_inner-1
```

`C` is `[d_inner × d_state]`. This is the only non-trivial GEMV in the scan
(d_inner=128, d_state=128 per architecture.md — a square 128×128 GEMV per
step).

### 2.5 Backward Pass (Chain Rule)

Given upstream gradient `dL/dy[t]` (shape `[seq_len × d_inner]`):

```
dh[t] = C^T @ (dL/dy[t])  +  A_eff[t] ⊙ dh[t+1]     (reverse scan)
dB[t] = dh[t] ⊙ x[t]
dC    += dL/dy[t] ⊗ h[t]                               (outer product, accumulated)
dA_log[i] += sum_t  dt[t] * A_eff[t][i] * h[t-1][i] * dh[t][i]
dx[t] = B[t]^T @ dh[t]    (if B is static: B^T ⊙ dh[t])
```

The backward scan runs in reverse order over t. h[t-1] values must be
available; they are stored during the forward pass in the h_cache buffer (see
§4.4).

---

## 3. Data Layout Contracts

All arrays are row-major (C order). Caller owns allocation and lifetime.

| Symbol | Shape | Type | Notes |
|--------|-------|------|-------|
| `A_log` | `[d_state]` | `float` | Log-space diagonal, negative for decay |
| `dt` | `[seq_len]` | `float` | Per-step timescale; NULL = time-invariant |
| `B` | `[seq_len × d_state]` | `float` | Row stride = d_state; stride=0 for static |
| `C` | `[d_inner × d_state]` | `float` | Output projection matrix |
| `x` | `[seq_len × d_inner]` | `float` | Input after in_proj+conv1d |
| `h` | `[d_state]` | `float` | State vector, modified in-place |
| `y` | `[seq_len × d_inner]` | `float` | Output, written by forward |
| `h_cache` | `[seq_len × d_state]` | `float` | Per-step states saved for backward; NULL = inference (do not save) |
| `moments` | `GpKlMoments*` | struct | Welford accumulators; NULL = skip GP-KL |

Batch versions add a leading `batch_size` dimension. All batch loops are
independent and parallelisable.

---

## 3b. Co-Spectral Subspace (Rank-20)

After 50 warmup steps, compute PCA of SSM output (rank-20) to establish a shared
low-rank subspace across all layers.

```c
typedef struct {
    float *U;          // [d_state × 20] basis vectors
    float *cov_accum;  // [d_state × d_state] online covariance accumulator
    int    n_samples;  // warmup step counter
    int    calibrated; // flag: set to 1 after 50 steps
} CoSpectralSubspace;
```

**Calibration (steps 0-50):**

1. Accumulate online covariance of SSM output `h[t]`:
   ```
   cov_accum += h[t] @ h[t]^T  / n_samples
   ```

2. At step 50: compute PCA, extract top-20 eigenvectors into `U`.

**Forward pass (after calibration):**

After SSM scan completes, apply straight-through projection:
```
Forward:  h_full = full d_state output (unchanged)
Backward: project d_h onto rank-20 subspace: d_h_proj = U @ U^T @ d_h
```

This constrains inter-layer gradient flow to the rank-20 subspace while preserving
forward computation at full dimensionality.

---

## 4. Function Signatures

### 4.1 ssm_scan_forward

```c
void ssm_scan_forward(
    const float *A_log,    /* [d_state] */
    const float *dt,       /* [seq_len] or NULL */
    const float *B,        /* [seq_len × d_state] or [d_state] if static */
    int          B_stride, /* d_state (dynamic) or 0 (static) */
    const float *C,        /* [d_inner × d_state] */
    const float *x,        /* [seq_len × d_inner] */
    float       *h,        /* [d_state] in/out */
    float       *y,        /* [seq_len × d_inner] out */
    float       *h_cache,  /* [seq_len × d_state] or NULL */
    GpKlMoments *moments,  /* or NULL */
    int seq_len,
    int d_state,
    int d_inner
);
```

Behaviour:
- If `dt != NULL`: compute `A_eff[t]` inside the loop per step.
- If `dt == NULL`: compute `A_eff` once (length `d_state`) before the loop.
- If `h_cache != NULL`: store `h` after each update (before y computation) at
  `h_cache[t * d_state .. (t+1)*d_state]`.
- If `moments != NULL`: call `gp_kl_welford_update(moments, h, d_state)` after
  each state update (see §6).
- y[t] = C @ h is a GEMV of shape `[d_inner] = [d_inner × d_state] @ [d_state]`.

### 4.2 ssm_scan_backward

```c
void ssm_scan_backward(
    const float *A_log,    /* [d_state] */
    const float *dt,       /* [seq_len] or NULL */
    const float *B,        /* [seq_len × d_state] or [d_state] */
    int          B_stride,
    const float *C,        /* [d_inner × d_state] */
    const float *x,        /* [seq_len × d_inner] */
    const float *h_cache,  /* [seq_len × d_state] REQUIRED */
    const float *dy,       /* [seq_len × d_inner] upstream gradient */
    float       *dA_log,   /* [d_state] accumulated gradient out */
    float       *dB,       /* [seq_len × d_state] or [d_state] gradient out */
    float       *dC,       /* [d_inner × d_state] accumulated gradient out */
    float       *dx,       /* [seq_len × d_inner] gradient out */
    float       *dh_out,   /* [d_state] gradient w.r.t. initial h, or NULL */
    int seq_len,
    int d_state,
    int d_inner
);
```

Behaviour:
- Runs the reverse scan (t = seq_len-1 down to 0).
- `dA_log`, `dC` are accumulated (caller must zero-initialise before calling if
  they span multiple sequences).
- `dB` layout matches `B`: dynamic gets a full `[seq_len × d_state]` tensor;
  static gets a single `[d_state]` vector (accumulated over all t).
- Does NOT call GP-KL accumulators (backward is pure gradient, no statistics).

### 4.3 ssm_scan_forward_batch

```c
void ssm_scan_forward_batch(
    const float *A_log,    /* [d_state] shared across batch */
    const float *dt,       /* [batch_size × seq_len] or NULL */
    const float *B,        /* [batch_size × seq_len × d_state] or static */
    int          B_stride,
    const float *C,        /* [d_inner × d_state] shared across batch */
    const float *x,        /* [batch_size × seq_len × d_inner] */
    float       *h,        /* [batch_size × d_state] in/out */
    float       *y,        /* [batch_size × seq_len × d_inner] */
    float       *h_cache,  /* [batch_size × seq_len × d_state] or NULL */
    GpKlMoments *moments,  /* [batch_size] array or NULL */
    int batch_size,
    int seq_len,
    int d_state,
    int d_inner
);
```

Implementation rule: this is a loop over `b in 0..batch_size-1` calling
`ssm_scan_forward` with pointer arithmetic. No new logic. The loop is the
only addition.

When `moments != NULL` the caller passes an array of `batch_size`
`GpKlMoments` structs; each batch element updates its own struct.

### 4.4 ssm_scan_backward_batch

```c
void ssm_scan_backward_batch(
    /* mirrors ssm_scan_forward_batch parameter layout */
    /* dA_log and dC are batch-accumulated: caller zeros, this adds */
    int batch_size,
    /* ... */
);
```

Same pattern as the forward batch: loop over b, call `ssm_scan_backward`.
Gradient accumulation into `dA_log` and `dC` is additive across the batch
(they are shared parameters).

### 4.5 ssm_scan_forward_wht

```c
void ssm_scan_forward_wht(
    const float *A_log,
    const float *dt,
    const float *B,
    int          B_stride,
    const float *C,
    const float *x,
    float       *h,
    float       *y,
    float       *h_cache,
    GpKlMoments *moments,
    int seq_len,
    int d_state,
    int d_inner
);
```

**WHT Applied to ALL SSM Layers.** Pre-scan and post-scan (2× per layer, not per chunk).

Identical to `ssm_scan_forward` except:

1. Before entering the scan loop, apply a Walsh-Hadamard Transform to `h` (the
   initial state vector): `h = WHT(h) / sqrt(d_state)`.
2. After the full scan loop completes, apply the inverse WHT to `h`:
   `h = WHT(h) / sqrt(d_state)` (WHT is its own inverse up to normalisation).
3. The scan loop itself is identical to `ssm_scan_forward`.

Rationale: the single WHT per call (not per step) costs O(d_state log d_state)
= O(128 * 7) = 896 ops vs O(T * d_state) = O(512 * 128) = 65536 ops for the
scan. Overhead is ~1.4%. The WHT mixes state dimensions before and after the
temporal scan, providing cross-dimension information flow without adding
per-step cost (architecture.md §5, WHT mixing note).

**In training:** All 6 layers use `ssm_scan_forward_wht` (never the non-WHT variant).

WHT implementation: in-place, radix-2, iterative. `d_state` must be a power of
two (128 = 2^7, satisfied by architecture.md). No separate file needed; a
`static inline wht_inplace(float *v, int n)` lives at the top of `ssm_scan.c`.

---

## 5. Internal Structure and Implementation Order

### 5.1 Pre-loop: A_eff Precomputation

```c
float A_eff[MAX_D_STATE];   // stack allocation, MAX_D_STATE = 256 (covers 128)

if (dt == NULL) {
    for (int i = 0; i < d_state; i++)
        A_eff[i] = expf(A_log[i]);
    // reuse every step
}
```

When dt is non-NULL, `A_eff` is recomputed inside the loop:

```c
for (int i = 0; i < d_state; i++)
    A_eff[i] = expf(dt[t] * A_log[i]);
```

`expf` is the standard C99 single-precision exponential. No fast approximation
unless the SIMD backend provides one (see §7.2).

### 5.2 Main Scan Loop

```c
for (int t = 0; t < seq_len; t++) {
    const float *x_t  = x + t * d_inner;
    const float *B_t  = B + t * B_stride;   // B_stride=0 → static B
    float       *y_t  = y + t * d_inner;

    // 1. A_eff (only if dt != NULL)
    if (dt != NULL)
        for (int i = 0; i < d_state; i++)
            A_eff[i] = expf(dt[t] * A_log[i]);

    // 2. State update: h[i] = A_eff[i] * h[i] + B_t[i] * x_t[?]
    //    See §5.3 for B*x term shape.
    for (int i = 0; i < d_state; i++)
        h[i] = A_eff[i] * h[i] + B_t[i] * x_t[i % d_inner];
    // NOTE: when d_state == d_inner (both 128 here) this is h[i] = ... + B_t[i]*x_t[i]

    // 3. Save state
    if (h_cache != NULL)
        memcpy(h_cache + t * d_state, h, d_state * sizeof(float));

    // 4. GP-KL Welford update
    if (moments != NULL)
        gp_kl_welford_update(moments, h, d_state, t);

    // 5. Output: y_t = C @ h  (GEMV)
    for (int j = 0; j < d_inner; j++) {
        float acc = 0.0f;
        const float *C_row = C + j * d_state;
        for (int i = 0; i < d_state; i++)
            acc += C_row[i] * h[i];
        y_t[j] = acc;
    }
}
```

### 5.3 B * x Term Clarification

In the architecture's data flow: `in_proj` maps `d_model=1024 → d_state=128`,
so `x` entering the scan has shape `[seq_len × d_state]` with `d_inner=128`.
Since `d_state == d_inner == 128` in this architecture, the term `B[i] * x[i]`
is a simple element-wise product: dimension i of the state is driven by
dimension i of the (projected) input, gated by B[i]. No reshape needed.

If a future variant uses `d_inner != d_state`, the term becomes
`sum_j B[i][j] * x[j]` — a dot product per state dimension. This is handled
by a runtime check: if `d_inner == d_state` use the fast path (element-wise),
otherwise use the dot-product path.

### 5.4 Chunked Processing

For sequences longer than the L1 cache working set, process in chunks of
`CHUNK = 64` steps. Within each chunk, `h`, `A_eff`, `B_t`, `x_t` all fit in
L1 (128 * 4 * 4 arrays = 8 KB, well within 32 KB L1).

```c
#define SSM_CHUNK 64

for (int t0 = 0; t0 < seq_len; t0 += SSM_CHUNK) {
    int t_end = (t0 + SSM_CHUNK < seq_len) ? t0 + SSM_CHUNK : seq_len;
    ssm_scan_chunk(/* args */, t0, t_end);
}
```

The chunk function is static and contains the inner loop from §5.2. This
chunking is transparent to the caller.

---

## 6. GP-KL Welford Integration

The Welford accumulators are defined in a header (e.g. `gp_kl.h`). This file
does not define the struct, only calls the update function. The contract:

```c
// Defined in gp_kl.h — referenced here, not owned here
typedef struct {
    int    n;           // sample count
    float  mean[MAX_D_STATE];
    float  M2[MAX_D_STATE];   // for variance (Welford)
    float  lag1_prev[MAX_D_STATE];
    float  lag1_sum[MAX_D_STATE];   // for lag-1 autocorrelation
    float  stability_sum;           // E[|h_t - h_{t-1}|^2]
    float  h_prev[MAX_D_STATE];     // previous state snapshot
    float  orth_pairs[32][2];       // 32 random pair indices (integers stored as float for alignment)
    float  orth_sum[32];            // E[<h_i, h_j>] for 32 pairs
} GpKlMoments;

// Called from ssm_scan_forward once per step, after state update
void gp_kl_welford_update(GpKlMoments *m, const float *h, int d_state, int t);
```

The `gp_kl_welford_update` call is the ONLY GP-KL logic that runs inside the
scan loop. All 5 moments (mean, variance, lag-1 autocorr, stability,
orthogonality) are accumulated inside that function. This file does not
implement the GP-KL math, only calls the hook.

Cost per step: approximately `5 * d_state` floating-point ops = 640 ops vs
the scan step cost of approximately `d_state + d_inner * d_state` = 128 + 16384
= 16512 ops. The Welford overhead is ~3.9% per step.

---

## 7. SIMD Strategy

### 7.1 Backend Selection (Compile-Time)

```c
#if defined(__AVX2__)
  #define SSM_SIMD_AVX2
#elif defined(__ARM_NEON)
  #define SSM_SIMD_NEON
#else
  #define SSM_SIMD_SCALAR
#endif
```

All three backends produce identical numerical output. The SIMD backends are
optimisations, not separate code paths for the caller.

### 7.2 AVX2 State Update (Inner Loop Replacement)

The state update loop (§5.2, step 2) operates on `d_state=128` floats. AVX2
processes 8 floats per instruction.

```c
// Processes 8 state dimensions per iteration (128/8 = 16 iterations)
__m256 va   = _mm256_loadu_ps(A_eff + i);
__m256 vh   = _mm256_loadu_ps(h + i);
__m256 vb   = _mm256_loadu_ps(B_t + i);
__m256 vx   = _mm256_loadu_ps(x_t + i);    // valid when d_state == d_inner
__m256 res  = _mm256_fmadd_ps(va, vh, _mm256_mul_ps(vb, vx));
_mm256_storeu_ps(h + i, res);
```

One `_mm256_fmadd_ps` per 8 state dimensions: `A_eff * h + B * x` in a single
fused op. No scalar fallback needed for `d_state=128` (divisible by 8).

### 7.3 AVX2 GEMV for Output (y = C @ h)

The GEMV `[128 × 128] @ [128]` is the most expensive operation per step.
Approach: process 4 rows of C simultaneously using 4 accumulators.

```c
// 4 output elements (j, j+1, j+2, j+3) computed in parallel
__m256 acc0 = _mm256_setzero_ps();
__m256 acc1 = _mm256_setzero_ps();
__m256 acc2 = _mm256_setzero_ps();
__m256 acc3 = _mm256_setzero_ps();

for (int i = 0; i < d_state; i += 8) {
    __m256 vh = _mm256_loadu_ps(h + i);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(C + (j+0)*d_state + i), vh, acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(C + (j+1)*d_state + i), vh, acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(C + (j+2)*d_state + i), vh, acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(C + (j+3)*d_state + i), vh, acc3);
}

y_t[j+0] = hsum256(acc0);
y_t[j+1] = hsum256(acc1);
y_t[j+2] = hsum256(acc2);
y_t[j+3] = hsum256(acc3);
```

`hsum256` is a standard 3-instruction horizontal sum of an `__m256` register.
This inner loop runs 128/8 = 16 times for 4 output rows, then the outer loop
advances by 4 rows. Total outer iterations: 128/4 = 32.

### 7.4 NEON Fallback

Replace `__m256` / `_mm256_*` with `float32x4_t` / `vfmaq_f32`. Process
4 floats per register instead of 8. All logic is identical. No separate
NEON-specific loop restructuring needed for this dimension (128 is divisible
by 4).

### 7.5 Scalar Fallback

The loops from §5.2 without any intrinsics. Compiler will auto-vectorise
with `-O2 -march=native` when the SIMD backends are disabled. This is the
reference implementation for correctness verification.

---

## 8. MTP State Power Application

Architecture §4 defines:

```
h_future = h_t ⊙ A^k + U @ V^T @ h_t
```

The `A^k` application is a per-element power of the diagonal:

```c
void ssm_state_power(
    const float *A_log,   /* [d_state] */
    const float *h,       /* [d_state] input state */
    float       *h_out,   /* [d_state] output: h ⊙ exp(k * A_log) */
    int          k,       /* lookahead steps */
    int          d_state
);
```

Implementation:

```c
for (int i = 0; i < d_state; i++)
    h_out[i] = h[i] * expf((float)k * A_log[i]);
```

When `dt` is non-NULL, `A_eff` varies per step and `A^k` is not well-defined
as a single vector power. In that case, the MTP branch uses the time-invariant
stored `A_log` (treating it as the expected per-step decay rate) and accepts
that as an approximation. This is documented in a code comment, not silently
assumed.

This function is declared and defined in `ssm_scan.c` because it directly
uses `A_log` and shares the `expf`-based A computation pattern. It is not
a scan — it is a single vector operation.

---

## 9. Error Handling and Assertions

All parameter validation is `assert()`-based (C99 `<assert.h>`). This is a
hot-path kernel; runtime bounds checking belongs in the caller, not here.

Required asserts:
```c
assert(A_log != NULL);
assert(x != NULL);
assert(h != NULL);
assert(y != NULL);
assert(d_state > 0 && d_state <= MAX_D_STATE);
assert(d_inner > 0);
assert(seq_len > 0);
assert(d_state % 8 == 0);  // required for AVX2 alignment; 128 satisfies this
```

On backward: additionally `assert(h_cache != NULL)`.

---

## 10. Header Interface

The public header `ssm_scan.h` exposes exactly these symbols (no internals):

```c
#ifndef SSM_SCAN_H
#define SSM_SCAN_H

#include "gp_kl.h"   // for GpKlMoments

#define MAX_D_STATE 256
#define SSM_CHUNK    64

void ssm_scan_forward(
    const float *A_log, const float *dt,
    const float *B, int B_stride,
    const float *C, const float *x,
    float *h, float *y, float *h_cache,
    GpKlMoments *moments,
    int seq_len, int d_state, int d_inner
);

void ssm_scan_backward(
    const float *A_log, const float *dt,
    const float *B, int B_stride,
    const float *C, const float *x,
    const float *h_cache, const float *dy,
    float *dA_log, float *dB, float *dC, float *dx, float *dh_out,
    int seq_len, int d_state, int d_inner
);

void ssm_scan_forward_batch(
    const float *A_log, const float *dt,
    const float *B, int B_stride,
    const float *C, const float *x,
    float *h, float *y, float *h_cache,
    GpKlMoments *moments,
    int batch_size, int seq_len, int d_state, int d_inner
);

void ssm_scan_backward_batch(
    const float *A_log, const float *dt,
    const float *B, int B_stride,
    const float *C, const float *x,
    const float *h_cache, const float *dy,
    float *dA_log, float *dB, float *dC, float *dx, float *dh_out,
    int batch_size, int seq_len, int d_state, int d_inner
);

void ssm_scan_forward_wht(
    const float *A_log, const float *dt,
    const float *B, int B_stride,
    const float *C, const float *x,
    float *h, float *y, float *h_cache,
    GpKlMoments *moments,
    int seq_len, int d_state, int d_inner
);

void ssm_state_power(
    const float *A_log, const float *h,
    float *h_out, int k, int d_state
);

#endif /* SSM_SCAN_H */
```

---

## 11. Implementation Order for code-writer

The following sequence minimises integration risk. Each step is independently
testable before proceeding.

**Step 1 — Scalar forward (no SIMD, no GP-KL, no h_cache)**
Implement `ssm_scan_forward` with scalar loops, `dt=NULL` path only, static B.
Write a unit test: 2-step sequence, d_state=4, d_inner=4, verify by hand.

**Step 2 — h_cache and dynamic dt**
Add `h_cache` saving and the `dt != NULL` branch. Unit test: verify h_cache
values match manually computed h[t] at each step.

**Step 3 — Dynamic B (B_stride > 0)**
Add the `B_stride` pointer offset. Unit test: confirm dynamic B changes the
output differently from static B.

**Step 4 — Scalar backward**
Implement `ssm_scan_backward`. Verify with finite-difference gradient check:
perturb each input element by ε=1e-4 and compare to analytical gradient.
Tolerance: max relative error < 1e-3.

**Step 5 — Batch wrappers**
Implement `ssm_scan_forward_batch` and `ssm_scan_backward_batch` as loops.
Test: batch of 4, result must equal 4 independent calls to the single-sequence
version.

**Step 6 — WHT variant**
Implement `wht_inplace`, add `ssm_scan_forward_wht`. Test: WHT applied twice
(forward then backward normalisation) must recover the original vector.

**Step 7 — ssm_state_power**
Implement and test: `k=0` must return `h` unchanged (exp(0)=1).

**Step 8 — GP-KL hook**
Add `gp_kl_welford_update` calls once `gp_kl.h` is available. Test:
Welford mean over a constant sequence must equal that constant.

**Step 9 — AVX2 SIMD**
Replace scalar inner loops in the forward scan and GEMV with AVX2 intrinsics
(§7.2 and §7.3). Correctness check: AVX2 output must match scalar output
exactly (same floats, not just approximately — fmadd is deterministic).

**Step 10 — Chunked processing**
Wrap the main loop in the chunk iterator (§5.4). Test: seq_len=200 (not a
multiple of 64) must give the same result as the unchunked version.

---

## 12. Performance Targets

Based on architecture.md model dimensions (d_state=128, d_inner=128,
seq_len=512) per forward call:

| Operation | FLOPs | Bottleneck |
|-----------|-------|-----------|
| State update (per step) | 2 × 128 = 256 | memory bandwidth (128 * 3 loads) |
| A_eff expf (per step, dt != NULL) | 128 expf calls | expf throughput |
| GEMV y=C@h (per step) | 2 × 128 × 128 = 32768 | compute |
| Total per sequence | 512 × (256 + 32768) ≈ 16.9M | GEMV-bound |

With AVX2 at 8 FP ops/cycle at 4 GHz: 32 GFLOP/s single-threaded throughput.
Expected forward time for one sequence: ~0.5 ms scalar, ~0.1 ms AVX2.

Memory: working set per chunk = 64 steps × (128 + 128 + 128) floats × 4B = 98
KB. Fits in L2 (512 KB typical). h vector (512 B) stays in L1 for the entire
sequence.

---

## 13. What This File Does NOT Contain

Explicitly out of scope to prevent scope creep:

- `expf` approximation tables — use standard `<math.h>`
- Any memory allocation — all buffers are caller-provided
- Threading / OpenMP — caller decides parallelism
- CUDA kernels — separate `ssm_scan_cuda.cu`
- GP-KL loss computation — only Welford update hook
- Ternary matmul for in_proj/out_proj/C — separate `ternary_matmul.c`
- conv1d — separate file
- MTP loss — only `ssm_state_power` helper
