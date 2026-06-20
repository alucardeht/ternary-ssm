# spec-training.md — Training Loop Specification

**File:** `src/model_v12/src/train.c`
**Language:** C99
**Dependencies:** `model.h`, `ssm.h`, `ops.h` (ternary matmul, SSM scan)

---

## Overview

`train.c` owns the complete forward → loss → backward → optimizer → repack cycle.
It is not a general autograd engine. Every gradient formula is closed-form and specific
to this architecture: ternary {-1, 0, +1} weights, diagonal SSM A, MVUE quantization.

No dynamic dispatch. No feature flags. All 5 subsystems (CE loss, GP-KL loss, MVUE
backward, crystallization skip, LR schedule) are always active.

---

## Data Structures Required from `model.h`

```c
// Per-layer SSM parameters (FP32 A, B, C; INT2 W_in, W_out; FP32 shadow)
typedef struct {
    float   *A;            // [d_state]          diagonal, in (0,1)
    float   *B;            // [d_state]
    float   *C;            // [d_state]
    uint8_t *W_in_int2;    // [d_model * d_state / 4]  INT2 packed
    uint8_t *W_out_int2;   // [d_state * d_model / 4]  INT2 packed
    float   *W_in_shadow;  // [d_model * d_state]      FP32 logits
    float   *W_out_shadow; // [d_state * d_model]      FP32 logits
    float   *W_in_sampled; // [d_model * d_state]      current sampled {-1,0,+1}
    float   *W_out_sampled;// [d_state * d_model]      current sampled {-1,0,+1}
    float   *conv_w;       // [4 * d_state]            conv1d kernel, ternary
    float   *conv_w_shadow;
    // Gradients (accumulated for one step, zeroed after optimizer_step)
    float   *dW_in;        // [d_model * d_state]
    float   *dW_out;       // [d_state * d_model]
    float   *dA;           // [d_state]
    float   *dB;           // [d_state]
    float   *dC;           // [d_state]
} SSMLayer;

typedef struct {
    SSMLayer layers[N_LAYERS];   // N_LAYERS = 6
    float   *embed;              // [vocab * d_model] FP32 (RNS output)
    float   *lm_head;            // [d_model * vocab] FP32 shadow
    float   *lm_head_sampled;    // sampled ternary
    float   *norm_w;             // [d_model]  final layer norm scale
    float   *norm_b;             // [d_model]  final layer norm bias
    int      step;               // global step counter
    // Welford accumulators for GP-KL (per layer)
    float   *gp_mean;            // [N_LAYERS * d_state]
    float   *gp_M2;              // [N_LAYERS * d_state]
    float   *gp_autocov;         // [N_LAYERS * d_state]
    float   *gp_stability;       // [N_LAYERS * d_state]
    float   *gp_orth;            // [N_LAYERS]  scalar per layer
    int     *gp_count;           // [N_LAYERS]
} Model;

// Forward pass activations (allocated once, reused every step)
typedef struct {
    float *h;        // [N_LAYERS * seq_len * d_state]  SSM hidden states
    float *x_proj;   // [N_LAYERS * seq_len * d_state]  W_in output
    float *x_conv;   // [N_LAYERS * seq_len * d_state]  conv1d output
    float *residual; // [N_LAYERS * seq_len * d_model]  pre-skip activations
    float *logits;   // [seq_len * vocab]
} Cache;
```

---

## Subsystem 1 — Cross-Entropy Loss

### `compute_ce_loss`

```c
float compute_ce_loss(
    const float    *logits,   // [seq_len * vocab]  raw pre-softmax
    const uint16_t *targets,  // [seq_len]          target token IDs
    int             vocab,
    int             seq_len
);
```

**Algorithm:**

For each position t = 0 .. seq_len-2 (predict token t+1 from logits at t):

1. Find max logit for numerical stability (log-sum-exp trick):
   ```
   max_logit = max(logits[t * vocab + 0 .. vocab-1])
   ```

2. Compute log-softmax for the target token only (no need to materialize full softmax):
   ```
   sum_exp = 0
   for v in 0..vocab-1:
       sum_exp += exp(logits[t*vocab + v] - max_logit)
   log_prob = logits[t*vocab + targets[t+1]] - max_logit - log(sum_exp)
   ```

3. Accumulate:
   ```
   ce_loss -= log_prob  (nats, positive)
   ```

4. Return `ce_loss / (seq_len - 1)`.

**Memory:** O(1) extra (no softmax buffer). Single pass over logits.

**Numerical contract:** `sum_exp` is always >= 1.0 (target term is included). No log(0).

**Implementation note:** vocab = 32064. Inner loop over 32064 floats per token. With AVX2,
use 8-wide float reduction. For SCALAR fallback, plain loop is fine (compiler will vectorize).

---

## Subsystem 2 — GP-KL Unified Loss

### `update_gp_moments`

Called once per forward pass, after SSM scan, before lm_head.

```c
void update_gp_moments(
    Model       *model,
    const float *h_all,    // [N_LAYERS * seq_len * d_state]  all hidden states
    int          seq_len
);
```

**Algorithm (per layer k):**

Uses Welford online accumulator (see math-innovations.md §9). At each step:

```
For t = 0..seq_len-1, for each dim i in 0..d_state-1:
    x = h_all[k * seq_len * d_state + t * d_state + i]
    count[k] += 1
    delta = x - mean[k*d_state + i]
    mean[k*d_state + i] += delta / count[k]
    delta2 = x - mean[k*d_state + i]
    M2[k*d_state + i] += delta * delta2
    variance[i] = M2[i] / count[k]

Lag-1 autocov (approximate, online):
    For t = 1..seq_len-1:
        autocov[k*d_state + i] += h[t][i] * h[t-1][i]  (unnormalized)

Stability:
    For t = 1..seq_len-1:
        diff = h[t][i] - h[t-1][i]
        stability[k*d_state + i] += diff * diff

Orthogonality (32 random pairs, fixed indices chosen at init):
    gp_orth[k] += dot(h[pair_a][k], h[pair_b][k])^2  (scalar per layer)
```

Accumulators persist across steps (not reset). They converge to true statistics over training.

### `compute_gp_kl_loss`

```c
float compute_gp_kl_loss(
    const Model *model
);
```

**Targets (from architecture.md §5):**

| Moment | Target |
|--------|--------|
| Mean | 0.0 (zero-centered activations) |
| Variance | 1.0 (unit variance) |
| Lag-1 autocorr | 0.0 (no temporal bias) |
| Stability | <= 0.1 (bounded change per step) |
| Orthogonality | 0.0 (independent dimensions) |

**Loss formulation (MSE to target, per layer, summed):**

```
For each layer k:
    mu     = mean(gp_mean[k * d_state .. (k+1)*d_state - 1])
    var_   = mean(M2[k*...] / count[k])
    autocr = mean(gp_autocov[k*...]) / (var_ + 1e-8)
    stab   = mean(gp_stability[k*...]) / count[k]
    orth   = gp_orth[k] / (32 * count[k])

    l_mean  = mu * mu
    l_var   = (var_ - 1.0f) * (var_ - 1.0f)
    l_acorr = autocr * autocr
    l_stab  = fmaxf(stab - 0.1f, 0.0f) * fmaxf(stab - 0.1f, 0.0f)
    l_orth  = orth * orth

    gp_loss += l_mean + l_var + l_acorr + l_stab + l_orth
```

Return `GP_KL_BETA * gp_loss` where `GP_KL_BETA = 0.01f`.

### Subsystem 2b — Contraction Window Loss

Add attractor penalty to the total loss:

```c
float compute_contraction_loss(
    const Model *model
);
```

**Rationale:** Prevents near-unit eigenvalues that cause repetitive output generation.

**Algorithm:**

For each layer k, compute minimum singular value of `(I - A_q)` where `A_q` is the quantized A:

```c
// For diagonal A, σ_min(I - A) = min_i(1 - A[i])
float sigma_min = 1.0f - max_A;  // max element of A

// Add attractor penalty
float L_attractor = 0.0f;
if (sigma_min > 0.0f) {
    L_attractor = -CONTRACTION_BETA * logf(sigma_min);
} else {
    L_attractor = CONTRACTION_PENALTY_INF;  // high penalty for divergence
}
```

Return `L_attractor` to be added to total loss.

**Combined loss** (called by `train_step`):
```
total_loss = ce_loss + gp_kl_loss + contraction_loss
```

---

## Subsystem 3 — Backward Pass (Manual, Architecture-Specific)

### `compute_backward`

```c
void compute_backward(
    Model          *model,
    Cache          *cache,
    const uint16_t *targets,
    int             seq_len
);
```

This is NOT a general autograd. It is the closed-form backward for the specific graph:
`tokens → embed → [SSM × 6] → norm → lm_head → CE loss`.

Gradients accumulated into `model->layers[k].dW_*`, `model->layers[k].dA/dB/dC`.

---

#### 3a. Gradient of CE loss w.r.t. logits

```
dL/d_logits[t][v] = softmax(logits[t])[v] - 1[v == targets[t+1]]
```

For each t, compute full softmax (needed here), subtract 1 at target index.
Store in `d_logits[seq_len * vocab]` (caller allocates on stack or scratch buffer).

**Numerically:** use the same max_logit computed in forward. Do NOT recompute max.

---

#### 3b. Gradient through lm_head (ternary W_lm)

```
d_hidden[t] = W_lm_sampled^T @ d_logits[t]     [d_model]
d_W_lm     += d_logits[t] @ hidden[seq_len-1][t]^T  [d_model * vocab]
```

`W_lm_sampled` is the sampled ternary matrix. Use CSR-lite for transpose matmul:
```
For each output neuron i in 0..d_model-1:
    d_hidden[t][i] = sum(d_logits[t][pos_indices[i]]) - sum(d_logits[t][neg_indices[i]])
```

---

#### 3c. Gradient through final layer norm

Standard layer norm backward. Given norm scale `gamma`, bias `beta`:
```
y = gamma * (x - mu) / sigma + beta
dy/dx = gamma / sigma * (1 - 1/N - (x-mu)^2/N*sigma^2)
```

Pass `d_pre_norm` back through.

---

#### 3d. Gradient through SSM stack (layers 5 → 0)

For each layer k (reverse order):

**3d-i. Gradient through W_out (ternary)**
```
d_ssm_out[t] = W_out_sampled^T @ d_residual[t]   [d_state]
d_W_out     += d_ssm_out[t] @ x_conv[k][t]^T      (outer product, accumulated)
```
CSR-lite transpose: same as lm_head above.

**3d-ii. Gradient through SSM scan (linear recurrence)**

The SSM scan is `h[t] = A ⊙ h[t-1] + B * x[t]`, output `y[t] = C @ h[t]`.

Backward through linear SSM (BPTT, reverse-time scan):

```
d_h[T-1] = C^T * d_ssm_out[T-1]
For t = T-2 .. 0:
    d_h[t] = C^T * d_ssm_out[t] + A ⊙ d_h[t+1]   (reverse recurrence)
```

This is cheap: A is diagonal, so `A ⊙ d_h[t+1]` is an element-wise multiply.
Cost: T * d_state multiplies (no matmul).

Gradients for A, B, C:
```
dA[i]  += sum_t( d_h[t][i] * h[t-1][i] )      for t = 1..T-1
dB[i]  += sum_t( d_h[t][i] * x_conv[k][t] )
dC[i]  += sum_t( d_ssm_out[t][i] * h[t][i] )   (via chain rule through y[t]=C@h[t])
```

**Note:** A, B, C are FP32, not ternary. Standard gradient, no MVUE needed here.

**3d-iii. Gradient through conv1d**

```
d_x_conv[t] = B^T * d_h[t]
d_x_proj = conv1d_backward(d_x_conv, conv_w)   [standard 1D conv backward]
```
Conv kernel is ternary: use same CSR-lite pattern for conv backward.

**3d-iv. Gradient through W_in (ternary)**
```
d_residual_in[t] = W_in_sampled^T @ d_x_proj[t]
d_W_in          += d_x_proj[t] @ residual[k][t]^T
```

**3d-v. Skip connection**
```
d_residual[k-1][t] = d_residual[k][t] + d_residual_in[t]
```
(The skip `h_{k+1} = h_k + SSM(h_k)` means gradient flows to both branches.)

---

#### 3e. What we do NOT backprop through

- The RNS embedding: embedding vectors are exact integers, no gradient needed.
  The input tokens are discrete. Embed lookup is treated as a constant during backward.
- The GP-KL loss backward: gradients flow only through the CE path. The GP-KL
  accumulators are updated in the forward pass and contribute to loss value, but their
  backward signal is implicitly captured through the hidden states `h`. In practice,
  GP-KL backward = `d_h[t] += GP_KL_BETA * d_gp_per_h[t]` (mean/variance gradients
  w.r.t. h[t]). This is a 3-line addition to step 3d-ii:

  ```
  // GP-KL gradient contribution to d_h[t]
  float gp_mu   = gp_mean[k * d_state + i];
  float gp_var  = gp_M2[k * d_state + i] / (gp_count[k] + 1e-8f);
  d_h[t][i] += GP_KL_BETA * (2.0f * gp_mu / gp_count[k]             // mean term
                           + 2.0f * (gp_var - 1.0f) * d_h[t][i]     // var term
                           );
  ```

  This is the only place GP-KL touches the backward graph.

---

## Subsystem 4 — Optimizer (MVUE + Crystallization)

### `optimizer_step`

```c
void optimizer_step(
    Model *model,
    float  lr
);
```

**Applied to all ternary weight matrices:** W_in, W_out, W_lm, conv kernels.
**NOT applied to:** A, B, C (FP32, plain SGD), norm weights (plain SGD).

---

#### 4a. MVUE Update for Ternary Logits

For each shadow logit `s` and its sampled value `w_sampled` (both stored in Model):

```c
float soft_boundary_mask(float s) {
    float abs_s = fabsf(s);
    if (abs_s < 0.5f)  return 1.0f;
    if (abs_s > 1.5f)  return 0.0f;
    return expf(-10.0f * (abs_s - 1.0f) * (abs_s - 1.0f));
}

// For each ternary logit s with accumulated gradient dL/dW:
float mvue_grad = (w_sampled - s) * soft_boundary_mask(s);
// mvue_grad is the natural gradient on the ternary manifold
// dL/dW from backward is the loss-signal gradient
// Combined update:
float update = dW * mvue_grad;  // hadamard: loss direction × natural geometry
```

Wait — this needs clarification. The MVUE gradient replaces Adam's second-moment
correction. The relationship is:

- `dW` (from backward): `∂L/∂w_sampled` — how the loss changes with sampled weights.
- `mvue_grad = (w_sampled - s) · mask`: the score function of the ternary distribution.

The full update (from math-innovations.md §1 and §2):
```
s_new = s - lr * dW * (w_sampled - s) * mask(s)
```

This is `lr=0.1` applied to the product of loss gradient and MVUE score function.
The mask is the soft boundary (Innovation #7), NOT the hard 1[|s|<1].

**Why this works:** `(w_sampled - s)` has the same sign as the natural gradient
direction on the ternary manifold. Multiplying by `dW` points in the direction of
decreasing loss along that manifold.

**Important — do NOT apply Fisher^-1 separately.** MVUE already encodes the natural
gradient (see math-innovations.md §2, lessons-learned.md §2). Double correction
worsens loss by 0.34 nats (empirically validated R97).

---

#### 4b. D4 Temporal Sign Consistency Filter

Before applying the update, check sign consistency over 5 consecutive steps:

```c
// Per-weight sign history buffer (circular, size 5)
// Stored in Model: int8_t *sign_history[n_weights]

int sign_current = (dW > 0) ? 1 : ((dW < 0) ? -1 : 0);
int t = step % 5;  // circular index
sign_history[w_idx][t] = sign_current;

// Check consistency: same sign for all 5 recent steps
int consistent = 1;
for (int i = 0; i < 5; i++) {
    if (sign_history[w_idx][i] != sign_current || sign_current == 0) {
        consistent = 0;
        break;
    }
}

if (!consistent) {
    continue;  // skip gradient update
}
```

Alternatively, use sign-product form (more efficient):
```c
// Accumulate sign product: P_t = P_{t-1} * sign(g_t) * sign(g_{t-5})
// High sign consistency → P_t remains ±1
// Gradient update only if |P_t| = 1.0 (perfect agreement)
```

#### 4c. Crystallization Skip

Before applying the update, check:

```c
if (fabsf(s) > CRYSTALLIZATION_THRESHOLD) {
    continue;  // skip — weight has converged
}
// threshold = 3.0f  (see math-innovations.md §5)
```

Crystallized weights: ~20-30% after 10K steps. They are never updated again.

**Order of operations:**
1. Check crystallization first (skip if |s| > 3).
2. Compute MVUE grad (soft boundary mask).
3. Apply update.
4. Do NOT clamp shadow logit after update (allow free movement).

---

#### 4d. FP32 Parameters (A, B, C, norm)

Plain SGD, no MVUE:
```c
A[i] -= lr * dA[i];
B[i] -= lr * dB[i];
C[i] -= lr * dC[i];
norm_w[i] -= lr * d_norm_w[i];
norm_b[i] -= lr * d_norm_b[i];
```

Clamp A to (0, 1) after update (stability invariant):
```c
A[i] = fmaxf(1e-4f, fminf(1.0f - 1e-4f, A[i]));
```

---

#### 4e. Zero Gradients

After optimizer_step, zero all accumulated gradients:
```c
memset(layer->dW_in,  0, d_model * d_state * sizeof(float));
memset(layer->dW_out, 0, d_state * d_model * sizeof(float));
memset(layer->dA,     0, d_state * sizeof(float));
memset(layer->dB,     0, d_state * sizeof(float));
memset(layer->dC,     0, d_state * sizeof(float));
```

---

#### 4f. INT2 Repack (Periodic)

Every `REPACK_INTERVAL = 100` steps:
```c
if (model->step % REPACK_INTERVAL == 0) {
    for (int k = 0; k < N_LAYERS; k++) {
        repack_shadow_to_int2(model->layers[k].W_in_shadow,
                              model->layers[k].W_in_int2,
                              d_model * d_state);
        repack_shadow_to_int2(model->layers[k].W_out_shadow,
                              model->layers[k].W_out_int2,
                              d_state * d_model);
        // Also resample W_in_sampled and W_out_sampled from updated shadow
        resample_ternary(model->layers[k].W_in_shadow,
                         model->layers[k].W_in_sampled,
                         d_model * d_state);
    }
}
```

`repack_shadow_to_int2`: round shadow logit s to nearest {-1, 0, +1} then pack 4 per byte.
`resample_ternary`: sample new w ~ MVUE(s) for next forward pass.

---

## Subsystem 5 — LR Schedule

### `compute_lr`

```c
float compute_lr(int step, int warmup_steps, int total_steps, float lr_max);
```

**Warmup + cosine decay (from architecture.md §6):**

```
LR_MAX = 0.1f
WARMUP  = 1000 steps  (1% of expected 100K training)
TOTAL   = 100000 steps

if step < warmup_steps:
    lr = lr_max * (float)step / warmup_steps          // linear warmup

else:
    t = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = lr_max * 0.5f * (1.0f + cosf(M_PI * t))     // cosine decay to 0
```

**Final LR at end of schedule:** 0.0 (cosine reaches zero).
**Minimum LR floor:** `fmaxf(lr, lr_max * 0.01f)` — never go below 1% of peak.

---

## Top-Level Entry Point

### `train_step`

```c
void train_step(
    Model          *model,
    Cache          *cache,
    const uint16_t *tokens,    // [seq_len] token IDs
    int             seq_len,
    float           lr
);
```

**Sequence:**

```
1. Forward pass:
   a. embed_rns(tokens, embed_out)              // RNS lookup, exact integers
   b. For k = 0..N_LAYERS-1:
       x_proj[k] = ternary_matmul(h_in, W_in)  // CSR-lite, no multiplications
       x_conv[k] = conv1d(x_proj[k], conv_w)   // kernel=4
       h[k]      = ssm_scan(x_conv[k], A, B, C) // diagonal recurrence
       y_out[k]  = ternary_matmul(h[k], W_out)
       h_next    = h_in + y_out[k]              // skip connection
       update_gp_moments(model, h[k], seq_len)  // Welford accumulators
   c. final_norm(h_next)
   d. logits = ternary_matmul(h_norm, W_lm)

2. Loss:
   ce   = compute_ce_loss(logits, tokens, vocab, seq_len)
   gp   = compute_gp_kl_loss(model)
   loss = ce + gp

3. Backward:
   compute_backward(model, cache, tokens, seq_len)

4. Optimizer:
   optimizer_step(model, lr)

5. Increment step:
   model->step += 1
```

**Note on MTP (Multi-step prediction):** MTP contributes a second CE loss term. For
each of 3 prediction layers (alternating layers 0, 2, 4), compute:
```
h_future = h[k] ⊙ A^2 + U @ V^T @ h[k]    // 2-step lookahead
logits_mtp = lm_head(h_future)
mtp_loss += ce_loss(logits_mtp, tokens[t+2..], vocab, seq_len-2)
```
MTP backward follows the same path as main CE backward, using the same `W_lm`
(shared head). Gradients to U and V are standard FP32 SGD (no MVUE needed since
U, V are FP32, not ternary).

Total loss:
```
loss = ce_main + 0.3f * mtp_loss + gp_kl_loss
```
Weight 0.3 for MTP (subordinate signal, not dominant).

---

## Function Signatures (Complete)

```c
// ---- Loss ---------------------------------------------------------------
float compute_ce_loss(const float *logits, const uint16_t *targets,
                      int vocab, int seq_len);

float compute_gp_kl_loss(const Model *model);

void  update_gp_moments(Model *model, const float *h_all, int seq_len);

// ---- Backward -----------------------------------------------------------
void  compute_backward(Model *model, Cache *cache,
                       const uint16_t *targets, int seq_len);

// ---- Optimizer ----------------------------------------------------------
void  optimizer_step(Model *model, float lr);

// ---- LR Schedule --------------------------------------------------------
float compute_lr(int step, int warmup_steps, int total_steps, float lr_max);

// ---- MVUE helpers (inline or static) ------------------------------------
static inline float soft_boundary_mask(float s);
static inline float sample_ternary(float s, float uniform_rand);
static inline float round_ternary(float s);

// ---- Top-level ----------------------------------------------------------
void  train_step(Model *model, Cache *cache,
                 const uint16_t *tokens, int seq_len, float lr);
```

---

## Constants

```c
#define N_LAYERS                 6
#define D_MODEL               1024
#define D_STATE                128
#define N_HEADS                 16
#define VOCAB_SIZE           32064
#define SEQ_LEN                512
#define LR_MAX                 0.1f
#define LR_WARMUP_STEPS       1000
#define LR_TOTAL_STEPS      100000
#define LR_MIN_FACTOR          0.01f
#define GP_KL_BETA             0.01f
#define CRYSTALLIZATION_TAU    3.0f
#define REPACK_INTERVAL         100
#define MTP_WEIGHT             0.3f
#define SOFT_BOUNDARY_SCALE    10.0f   // coefficient in exp(-k*(|s|-1)^2)
```

---

## Memory Responsibilities

`train.c` does NOT allocate. All buffers are owned by the caller:
- `Model` — allocated and initialized by `model.c`
- `Cache` — allocated once by caller, reused across all `train_step` calls
- `tokens` — owned by data loader

`Cache` scratch buffers required:
| Buffer | Size | Notes |
|--------|------|-------|
| `d_logits` | `seq_len * vocab * sizeof(float)` | CE backward scratch |
| `d_h` | `N_LAYERS * seq_len * d_state * sizeof(float)` | SSM backward |
| `d_x_conv` | `N_LAYERS * seq_len * d_state * sizeof(float)` | conv backward |
| `d_x_proj` | `N_LAYERS * seq_len * d_state * sizeof(float)` | W_in backward |
| `d_residual` | `N_LAYERS * seq_len * d_model * sizeof(float)` | skip backward |

Total scratch: ~2 × forward activation size = ~66 MB at batch=1, seq=512.

---

## Invariants

These must hold at all times. Violations are bugs, not edge cases:

1. `A[i] ∈ (0, 1)` for all i — enforced by clamp in optimizer_step.
2. `|s_sampled| ∈ {-1, 0, +1}` — enforced by `sample_ternary`.
3. `gp_count[k] > 0` before `compute_gp_kl_loss` — first call must follow at least one
   `update_gp_moments`.
4. `dW_*` zeroed before each forward pass begins (zeroed at end of `optimizer_step`).
5. Crystallization skip applied BEFORE gradient update (never skip after).
6. Soft boundary mask applied AFTER checking crystallization (not before).

---

## What This Is NOT

- Not a general autograd. Do not add dynamic ops or tape recording.
- Not a batched trainer. `train_step` processes one sequence at a time.
  Batching is handled at the caller level by looping and accumulating gradients.
- Not a distributed trainer. Single-process, single-thread (SIMD inside kernels).
- Not compatible with frameworks (PyTorch, JAX). This is standalone C.

---

## Implementation Order for code-writer

1. `soft_boundary_mask`, `sample_ternary`, `round_ternary` (inline helpers, ~15 lines)
2. `compute_lr` (10 lines)
3. `compute_ce_loss` (log-sum-exp, ~25 lines)
4. `update_gp_moments` (Welford, ~40 lines)
5. `compute_gp_kl_loss` (MSE to targets, ~30 lines)
6. `compute_backward` (most complex — ~150 lines; implement in this sub-order:
   a. CE gradient w.r.t. logits
   b. lm_head transpose
   c. final norm backward
   d. layer loop (W_out → SSM scan → conv → W_in → skip), layers 5 → 0
   e. GP-KL contribution to d_h
   f. MTP gradient (additive to existing lm_head gradient)
   )
7. `optimizer_step` (MVUE + crystallization + A-clamp + repack, ~60 lines)
8. `train_step` (orchestrator, ~30 lines)

Each function is independently testable. Unit test order mirrors implementation order.
