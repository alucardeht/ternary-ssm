/*
 * ssm_scan.h — Public interface for SSM recurrence kernels
 *
 * Implements the diagonal SSM scan primitive used in v12 layers.
 * All sequences are scanned with O(d_state) per-step cost via element-wise
 * state updates and GEMV output projection.
 *
 * Reference: src/model_v12/docs/spec-ssm-scan.md
 */

#ifndef SSM_SCAN_H
#define SSM_SCAN_H

#include <stdint.h>   /* uint8_t */
#include "types.h"    /* GpKlMoments, constants */
#include <math.h>     /* floorf, fabsf */

static inline float decompress_e5m2(uint8_t val) {
    uint32_t sign = (val & 0x80u) << 24;
    uint32_t exp = (val & 0x7Cu) >> 2;
    uint32_t mant = val & 0x03u;

    if (exp == 0) {
        if (mant == 0) {
            union { uint32_t u; float f; } uval;
            uval.u = sign;
            return uval.f;
        }
        float f_val = (float)mant * 0.25f * 0.00006103515625f; /* 2^-14 * (m/4) */
        return (val & 0x80u) ? -f_val : f_val;
    } else if (exp == 31) {
        union { uint32_t u; float f; } uval;
        if (mant == 0) {
            uval.u = sign | 0x7F800000u;
        } else {
            uval.u = sign | 0x7FC00000u;
        }
        return uval.f;
    } else {
        int real_exp = (int)exp - 15 + 127;
        if (real_exp <= 0) {
            return 0.0f;
        } else if (real_exp >= 255) {
            union { uint32_t u; float f; } uval;
            uval.u = sign | 0x7F800000u;
            return uval.f;
        }
        union { uint32_t u; float f; } uval;
        uval.u = sign | ((uint32_t)real_exp << 23) | (mant << 21);
        return uval.f;
    }
}

static inline uint8_t compress_e5m2_stochastic(float val, uint32_t *rng_state) {
    union { float f; uint32_t u; } uval;
    uval.f = val;
    uint32_t sign = (uval.u & 0x80000000u) >> 24; /* bit 7 of uint8 */
    float abs_val = fabsf(val);
    
    union { float f; uint32_t u; } abs_uval;
    abs_uval.f = abs_val;
    uint32_t f_exp = (abs_uval.u & 0x7F800000u) >> 23;
    uint32_t f_mant = abs_uval.u & 0x007FFFFFu;

    if (f_exp == 0) {
        return (uint8_t)sign;
    } else if (f_exp == 255) {
        uint8_t e5m2_mant = (f_mant != 0) ? 1u : 0u;
        return (uint8_t)(sign | 0x7Cu | e5m2_mant);
    }

    int real_exp = (int)f_exp - 112;
    if (real_exp <= 0) {
        float scaled = abs_val * 65536.0f;
        float f = floorf(scaled);
        float r = scaled - f;
        *rng_state = *rng_state * 1664525u + 1013904223u;
        float u = (float)(*rng_state >> 8) * (1.0f / 16777216.0f);
        int m = (int)((u < r) ? (f + 1.0f) : f);
        if (m <= 0) {
            return (uint8_t)sign;
        } else if (m >= 4) {
            return (uint8_t)(sign | (1u << 2));
        } else {
            return (uint8_t)(sign | m);
        }
    } else if (real_exp >= 31) {
        return (uint8_t)(sign | 0x7Cu);
    } else {
        uint32_t floor_m = f_mant >> 21;
        float r = (float)(f_mant & 0x001FFFFFu) / 2097152.0f;
        *rng_state = *rng_state * 1664525u + 1013904223u;
        float u = (float)(*rng_state >> 8) * (1.0f / 16777216.0f);
        uint32_t m = (u < r) ? (floor_m + 1) : floor_m;
        if (m == 4) {
            real_exp++;
            m = 0;
            if (real_exp >= 31) {
                return (uint8_t)(sign | 0x7Cu);
            }
        }
        return (uint8_t)(sign | ((uint32_t)real_exp << 2) | m);
    }
}

static inline float decompress_e4m3(uint8_t val) {
    uint32_t sign = (val & 0x80u) << 24;
    uint32_t exp = (val & 0x78u) >> 3;
    uint32_t mant = val & 0x07u;

    if (exp == 0) {
        if (mant == 0) {
            union { uint32_t u; float f; } uval;
            uval.u = sign;
            return uval.f;
        }
        float f_val = (float)mant * 0.125f * 0.015625f; /* 2^-6 * (m/8) = 2^-9 * m */
        return (val & 0x80u) ? -f_val : f_val;
    } else if (exp == 15 && mant == 7) {
        union { uint32_t u; float f; } uval;
        uval.u = sign | 0x7FC00000u;
        return uval.f;
    } else {
        int real_exp = (int)exp - 7 + 127;
        if (real_exp <= 0) {
            return 0.0f;
        } else if (real_exp >= 255) {
            union { uint32_t u; float f; } uval;
            uval.u = sign | 0x7F800000u;
            return uval.f;
        }
        union { uint32_t u; float f; } uval;
        uval.u = sign | ((uint32_t)real_exp << 23) | (mant << 20);
        return uval.f;
    }
}

static inline uint8_t compress_e4m3_stochastic(float val, uint32_t *rng_state) {
    union { float f; uint32_t u; } uval;
    uval.f = val;
    uint32_t sign = (uval.u & 0x80000000u) >> 24;
    float abs_val = fabsf(val);
    
    union { float f; uint32_t u; } abs_uval;
    abs_uval.f = abs_val;
    uint32_t f_exp = (abs_uval.u & 0x7F800000u) >> 23;
    uint32_t f_mant = abs_uval.u & 0x007FFFFFu;

    if (f_exp == 0) {
        return (uint8_t)sign;
    } else if (f_exp == 255) {
        return (uint8_t)(sign | 0x78u | 0x07u);
    }

    int real_exp = (int)f_exp - 120;
    if (real_exp <= 0) {
        float scaled = abs_val * 512.0f;
        float f = floorf(scaled);
        float r = scaled - f;
        *rng_state = *rng_state * 1664525u + 1013904223u;
        float u = (float)(*rng_state >> 8) * (1.0f / 16777216.0f);
        int m = (int)((u < r) ? (f + 1.0f) : f);
        if (m <= 0) {
            return (uint8_t)sign;
        } else if (m >= 8) {
            return (uint8_t)(sign | (1u << 3));
        } else {
            return (uint8_t)(sign | m);
        }
    } else if (real_exp >= 15) {
        return (uint8_t)(sign | 0x78u | 0x06u);
    } else {
        uint32_t floor_m = f_mant >> 20;
        float r = (float)(f_mant & 0x000FFFFFu) / 1048576.0f;
        *rng_state = *rng_state * 1664525u + 1013904223u;
        float u = (float)(*rng_state >> 8) * (1.0f / 16777216.0f);
        uint32_t m = (u < r) ? (floor_m + 1) : floor_m;
        if (m == 8) {
            real_exp++;
            m = 0;
            if (real_exp >= 15) {
                return (uint8_t)(sign | 0x78u | 0x06u);
            }
        }
        return (uint8_t)(sign | ((uint32_t)real_exp << 3) | m);
    }
}


/*
 * ssm_scan_forward
 *
 * Forward SSM scan over a sequence. Updates the hidden state h in-place
 * and writes output y. Optionally saves state snapshots for backward
 * and accumulates GP-KL statistics.
 *
 * Parameters:
 *   A:        [d_state] direct decay coefficients (A[i] in (0,1), not log-space)
 *   A_per_step: [seq_len × d_state] per-timestep A values (input-dependent A gate), or NULL
 *   dt:       [seq_len] per-step timescale, or NULL for time-invariant A
 *   B:        [d_state] input gate (static, time-invariant)
 *   B_stride: 0 = static B (reuse same row each step)
 *   C:        [d_inner * d_state] output projection (row-major)
 *   x:        [seq_len * d_inner] input sequence
 *   h:        [d_state] hidden state (in/out, updated in-place)
 *   y:        [seq_len * d_inner] output (written completely)
 *   h_cache:  [seq_len * d_state] saved states for backward, or NULL for inference
 *   moments:  GpKlMoments accumulator, or NULL to skip GP-KL
 *   boundary_mask: [seq_len] uint8, 1 = zero h before this step, or NULL to disable
 *   seq_len:  sequence length
 *   d_state:  hidden state dimension (128 in v12)
 *   d_inner:  output dimension (128 in v12, equals d_state)
 */
void ssm_scan_forward(
    const float   *A,
    const float   *A_per_step,   /* [seq_len × d_state] or NULL */
    const float   *dt,
    const float   *B,
    int            B_stride,
    const float   *C,
    const float   *x,
    float         *h,
    float         *y,
    float         *h_cache,
    GpKlMoments   *moments,
    const uint8_t *boundary_mask,
    int seq_len,
    int d_state,
    int d_inner,
    const float   *C_bilinear,   /* [d_state] bilinear gate, or NULL to disable */
    const float   *u,            /* [d_state] low-rank vector u, or NULL */
    const float   *v             /* [d_state] low-rank vector v, or NULL */
);

/*
 * ssm_scan_backward
 *
 * Reverse BPTT over the SSM recurrence. Computes gradients w.r.t. A, B, C, x,
 * and the initial hidden state h.
 *
 * Parameters:
 *   A, A_per_step, dt, B, B_stride, C, x: same as forward
 *   h_cache:  [seq_len * d_state] saved states from forward (REQUIRED)
 *   dy:       [seq_len * d_inner] upstream gradient dL/dy
 *   boundary_mask: [seq_len] uint8, 1 = zero dh before backward at this step, or NULL to disable
 *   dA:       [d_state] gradient accumulator (caller must zero before calling)
 *   dA_per_step: [seq_len × d_state] per-timestep gradient, or NULL (written when A_per_step is non-NULL)
 *   dB:       [d_state] if B_stride==0, else [seq_len * d_state]
 *   dC:       [d_inner * d_state] gradient accumulator (caller must zero)
 *   dx:       [seq_len * d_inner] gradient w.r.t. input
 *   dh_out:   [d_state] gradient w.r.t. initial h, or NULL
 *   seq_len, d_state, d_inner: same as forward
 */
void ssm_scan_backward(
    const float   *A,
    const float   *A_per_step,   /* [seq_len × d_state] or NULL */
    const float   *dt,
    const float   *B,
    int            B_stride,
    const float   *C,
    const float   *x,
    const float   *h_cache,
    const float   *dy,
    const uint8_t *boundary_mask,
    float         *dA,
    float         *dA_per_step,  /* [seq_len × d_state] or NULL */
    float         *dB,
    float         *dC,
    float         *dx,
    float         *dh_out,
    int seq_len,
    int d_state,
    int d_inner,
    const float   *C_bilinear,   /* [d_state] bilinear gate, or NULL to disable */
    float         *dC_bilinear,  /* [d_state] gradient accumulator, or NULL */
    const float   *u,            /* [d_state] low-rank vector u, or NULL */
    const float   *v,            /* [d_state] low-rank vector v, or NULL */
    float         *du,           /* [d_state] gradient accumulator, or NULL */
    float         *dv,            /* [d_state] gradient accumulator, or NULL */
    const float   *dh_direct     /* [seq_len * d_state] or NULL */
);

/*
 * ssm_scan_forward_batch
 *
 * Batch wrapper for ssm_scan_forward. Processes batch_size independent sequences.
 * All calls share A, A_per_step, dt, B, B_stride, C. Moments array has one struct per batch element.
 */
void ssm_scan_forward_batch(
    const float   *A,
    const float   *A_per_step,   /* [seq_len × d_state] or NULL (shared across batch) */
    const float   *dt,
    const float   *B,
    int            B_stride,
    const float   *C,
    const float   *x,
    float         *h,
    float         *y,
    float         *h_cache,
    GpKlMoments   *moments,
    const uint8_t *boundary_mask,
    int batch_size,
    int seq_len,
    int d_state,
    int d_inner,
    const float   *C_bilinear,   /* [d_state] bilinear gate, or NULL to disable */
    const float   *u,            /* [d_state] low-rank vector u, or NULL */
    const float   *v             /* [d_state] low-rank vector v, or NULL */
);

/*
 * ssm_scan_backward_batch
 *
 * Batch wrapper for ssm_scan_backward. dA and dC are accumulated across
 * the batch (caller must zero before calling). dB, dx, and dA_per_step are per-batch.
 */
void ssm_scan_backward_batch(
    const float   *A,
    const float   *A_per_step,   /* [seq_len × d_state] or NULL (shared) */
    const float   *dt,
    const float   *B,
    int            B_stride,
    const float   *C,
    const float   *x,
    const float   *h_cache,
    const float   *dy,
    const uint8_t *boundary_mask,
    float         *dA,
    float         *dA_per_step,  /* [batch_size × seq_len × d_state] or NULL (per-batch written) */
    float         *dB,
    float         *dC,
    float         *dx,
    float         *dh_out,
    int batch_size,
    int seq_len,
    int d_state,
    int d_inner,
    const float   *C_bilinear,   /* [d_state] bilinear gate, or NULL to disable */
    float         *dC_bilinear,  /* [d_state] gradient accumulator, or NULL */
    const float   *u,            /* [d_state] low-rank vector u, or NULL */
    const float   *v,            /* [d_state] low-rank vector v, or NULL */
    float         *du,           /* [d_state] gradient accumulator, or NULL */
    float         *dv,           /* [d_state] gradient accumulator, or NULL */
    const float   *dh_direct     /* [batch_size * seq_len * d_state] or NULL */
);

/*
 * ssm_scan_backward_reversible
 *
 * Reverse BPTT over the SSM recurrence, eliminating h_cache by reconstructing h[t-1]
 * from h[t] on-the-fly using the invertible SSM recurrence.
 *
 * Requires:
 *   h_final: final hidden state h[seq_len-1] saved from ssm_scan_forward.
 *   B_eff:   per-timestep B (Direct-B gate values, same as used in forward).
 *   x:       per-timestep conv output (same as used in forward).
 *   dt must be NULL (time-invariant case only; v12 always uses dt=NULL).
 *
 * Parameters:
 *   A:        [d_state] direct decay coefficients (A[i] in (0,1))
 *   B_eff:    [seq_len × d_state] per-timestep B values
 *   x:        [seq_len × d_state] input to SSM scan
 *   h_final:  [d_state] final hidden state from forward pass
 *   dy:       [seq_len × d_inner] upstream gradient dL/dy
 *   boundary_mask: [seq_len] uint8, 1 = zero dh at this step, or NULL to disable
 *   dA:       [d_state] gradient accumulator (caller must zero before calling)
 *   dB:       [seq_len × d_state] per-timestep B gradient
 *   dC:       [d_inner × d_state] gradient accumulator (caller must zero)
 *   dx:       [seq_len × d_inner] gradient w.r.t. input
 *   dh_out:   [d_state] gradient w.r.t. initial h, or NULL
 *   C:        [d_inner × d_state] output projection matrix
 *   seq_len, d_state, d_inner: dimensions
 */
void ssm_scan_backward_reversible(
    const float   *A,
    const float   *B_eff,
    const float   *x,
    const float   *h_final,
    const float   *dy,
    const uint8_t *boundary_mask,
    float         *dA,
    float         *dB,
    float         *dC,
    float         *dx,
    float         *dh_out,
    const float   *C,
    int seq_len,
    int d_state,
    int d_inner
);

/*
 * ssm_scan_backward_reversible_batch
 *
 * Batch wrapper for ssm_scan_backward_reversible. dA and dC are accumulated
 * across the batch (caller must zero before calling). dB and dx are per-batch.
 */
void ssm_scan_backward_reversible_batch(
    const float   *A,
    const float   *B_eff,
    const float   *x,
    const float   *h_final,
    const float   *dy,
    const uint8_t *boundary_mask,
    float         *dA,
    float         *dB,
    float         *dC,
    float         *dx,
    float         *dh_out,
    const float   *C,
    int batch_size,
    int seq_len,
    int d_state,
    int d_inner
);

/*
 * ssm_scan_backward_checkpointed
 *
 * Gradient checkpointed backward: saves only V12_CKPT_COUNT checkpoints in
 * forward (14× memory reduction), re-runs forward per segment in backward.
 * Numerically identical to ssm_scan_backward — no approximation.
 *
 * Parameters:
 *   A, A_per_step, dt, B, B_stride, C, x: same as ssm_scan_backward
 *   h_ckpt:   [V12_CKPT_COUNT × d_state] checkpoint array (not full h_cache)
 *   dy, boundary_mask: same as ssm_scan_backward
 *   dA, dA_per_step, dB, dC, dx, dh_out: same as ssm_scan_backward
 *   seq_len, d_state, d_inner: dimensions (seq_len must be divisible by V12_CKPT_STRIDE)
 */
void ssm_scan_backward_checkpointed(
    const float   *A,
    const float   *A_per_step,   /* [seq_len × d_state] or NULL */
    const float   *dt,
    const float   *B,
    int            B_stride,
    const float   *C,
    const float   *x,
    const float   *h_ckpt,
    const float   *dy,
    const uint8_t *boundary_mask,
    float         *dA,
    float         *dA_per_step,  /* [seq_len × d_state] or NULL */
    float         *dB,
    float         *dC,
    float         *dx,
    float         *dh_out,
    int seq_len,
    int d_state,
    int d_inner,
    const float   *C_bilinear,   /* [d_state] bilinear gate, or NULL to disable */
    float         *dC_bilinear,  /* [d_state] gradient accumulator, or NULL */
    const float   *u,            /* [d_state] low-rank vector u, or NULL */
    const float   *v,            /* [d_state] low-rank vector v, or NULL */
    float         *du,           /* [d_state] gradient accumulator, or NULL */
    float         *dv            /* [d_state] gradient accumulator, or NULL */
);

/*
 * ssm_scan_backward_checkpointed_batch
 *
 * Batch wrapper for ssm_scan_backward_checkpointed. dA and dC are accumulated
 * across the batch (caller must zero before calling). dB, dx, and dA_per_step are per-batch.
 */
void ssm_scan_backward_checkpointed_batch(
    const float   *A,
    const float   *A_per_step,   /* [seq_len × d_state] or NULL (shared) */
    const float   *dt,
    const float   *B,
    int            B_stride,
    const float   *C,
    const float   *x,
    const float   *h_ckpt,
    const float   *dy,
    const uint8_t *boundary_mask,
    float         *dA,
    float         *dA_per_step,  /* [batch_size × seq_len × d_state] or NULL (per-batch) */
    float         *dB,
    float         *dC,
    float         *dx,
    float         *dh_out,
    int batch_size,
    int seq_len,
    int d_state,
    int d_inner,
    const float   *C_bilinear,   /* [d_state] bilinear gate, or NULL to disable */
    float         *dC_bilinear,  /* [d_state] gradient accumulator, or NULL */
    const float   *u,            /* [d_state] low-rank vector u, or NULL */
    const float   *v,            /* [d_state] low-rank vector v, or NULL */
    float         *du,           /* [d_state] gradient accumulator, or NULL */
    float         *dv            /* [d_state] gradient accumulator, or NULL */
);

/*
 * ssm_scan_forward_wht
 *
 * SSM forward scan with Walsh-Hadamard Transform applied before and after.
 * WHT mixes state dimensions for cross-dimension information flow.
 * Identical to ssm_scan_forward except for pre/post WHT applications.
 */
void ssm_scan_forward_wht(
    const float *A,
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

/*
 * ssm_state_power
 *
 * Apply k-step state decay: h_out[i] = h[i] * A[i]^k
 * Used by MTP (multi-step prediction) to project state forward k steps.
 *
 * Parameters:
 *   A:       [d_state] direct decay coefficients
 *   h:       [d_state] input state
 *   h_out:   [d_state] output: h[i] * A[i]^k
 *   k:       lookahead steps (non-negative)
 *   d_state: state dimension
 */
void ssm_state_power(
    const float *A,
    const float *h,
    float       *h_out,
    int          k,
    int          d_state
);

/*
 * wht_inplace
 *
 * Radix-2 iterative Walsh-Hadamard Transform applied in-place.
 * Transforms a float vector of length n (must be power of 2).
 * WHT is self-inverse with proper normalisation: wht(wht(v)) = v.
 * Used to mix SSM output dimensions per-timestep before W_out projection.
 */
void wht_inplace(float *v, int n);

/*
 * ssm_scan_forward_with_stability_probe
 *
 * R356 F1 — slice S0 — forward stability-probe instrumentation.
 *
 * Non-invasive wrapper around ssm_scan_forward that optionally computes
 * the cumulative per-channel quantity
 *     stability_probe_cum[i] = Sum_t ln(max(|A_tot_i[t]|, 1e-30))
 * where A_tot_i[t] = A_eff[t][i] + (C_bilinear ? C_bilinear[i]*x[t][i] : 0).
 *
 * This metric is the validator-amendment stability bound for the F1 reverse
 * recurrence: err(h[0]) <= err(h[T]) * exp(max_i stability_probe_cum[i]).
 *
 * When stability_probe_cum is NULL the function is a thin pass-through to
 * ssm_scan_forward with zero behavioural change. When non-NULL, the buffer
 * (length d_state) is overwritten with the cumulative sum and a single
 * [F1 PROBE] summary line is printed at the end of the scan.
 *
 * Parameters mirror ssm_scan_forward plus:
 *   stability_probe_cum: [d_state] cumulative probe output, or NULL to disable
 */
void ssm_scan_forward_with_stability_probe(
    const float   *A,
    const float   *A_per_step,
    const float   *dt,
    const float   *B,
    int            B_stride,
    const float   *C,
    const float   *x,
    float         *h,
    float         *y,
    float         *h_cache,
    GpKlMoments   *moments,
    const uint8_t *boundary_mask,
    int seq_len,
    int d_state,
    int d_inner,
    const float   *C_bilinear,
    float         *stability_probe_cum
);

/*
 * ssm_scan_forward_batch_with_stability_probe
 *
 * Batch wrapper for ssm_scan_forward_with_stability_probe.
 * stability_probe_cum layout: [batch_size * d_state] (per-batch), or NULL.
 */
void ssm_scan_forward_batch_with_stability_probe(
    const float   *A,
    const float   *A_per_step,
    const float   *dt,
    const float   *B,
    int            B_stride,
    const float   *C,
    const float   *x,
    float         *h,
    float         *y,
    float         *h_cache,
    GpKlMoments   *moments,
    const uint8_t *boundary_mask,
    int batch_size,
    int seq_len,
    int d_state,
    int d_inner,
    const float   *C_bilinear,
    float         *stability_probe_cum
);

/*
 * ssm_scan_backward_f1
 *
 * R356 F1 — slice S1 — CPU reference backward with on-the-fly h reconstruction.
 *
 * Reverse BPTT over the SSM recurrence that eliminates h_cache by reconstructing
 * h[t-1] from h[t] via the per-channel scalar inversion
 *     A_tot_i[t] = A_eff[t][i] + (C_bilinear ? C_bilinear[i]*x[t][i] : 0)
 *     h[t-1][i]  = (h[t][i] - B_eff[t][i] * x[t][i]) / A_tot_i[t]
 *
 * Parameters mirror ssm_scan_backward, except:
 *   h_final: [d_state] hidden state AFTER the last step (replaces h_cache)
 *   C_bilinear, dC_bilinear: bilinear gate and its gradient accumulator
 *   stability_probe_cum: [d_state] cumulative probe from
 *     ssm_scan_forward_with_stability_probe, or NULL to skip the gate.
 *     When non-NULL and max_i(probe) >= ln(10), the function refuses to run
 *     (logs a warning and returns early) — the caller must dispatch to
 *     ssm_scan_backward_checkpointed instead.
 *
 * NOT a replacement for ssm_scan_backward_reversible. That function implements
 * the rejected pure-diagonal (divide-by-A) form and is kept only for audit.
 * The two must not be merged.
 */
void ssm_scan_backward_f1(
    const float   *A,
    const float   *A_per_step,   /* [seq_len × d_state] or NULL */
    const float   *dt,
    const float   *B,
    int            B_stride,
    const float   *C,
    const float   *x,
    const float   *h_final,      /* [d_state] state AFTER last step */
    const float   *dy,
    const uint8_t *boundary_mask,
    float         *dA,
    float         *dA_per_step,  /* [seq_len × d_state] or NULL */
    float         *dB,
    float         *dC,
    float         *dx,
    float         *dh_out,
    const float   *C_bilinear,   /* [d_state] or NULL */
    float         *dC_bilinear,  /* [d_state] or NULL */
    const float   *stability_probe_cum, /* [d_state] or NULL */
    int seq_len,
    int d_state,
    int d_inner,
    const float   *u,            /* [d_state] or NULL */
    const float   *v,            /* [d_state] or NULL */
    float         *du,           /* [d_state] or NULL */
    float         *dv            /* [d_state] or NULL */
);

/*
 * ssm_scan_backward_f1_batch
 *
 * R356 F1 — slice S2 — batch wrapper for ssm_scan_backward_f1.
 *
 * Mirrors ssm_scan_backward_batch. dA, dC, and dC_bilinear are accumulated
 * across the batch (caller must zero before calling). dB, dx, dA_per_step,
 * and dh_out are per-batch.
 *
 * h_final: [batch_size × d_state] — one post-last-step state per batch element.
 * stability_probe_cum: [batch_size × d_state] or NULL — one probe per batch
 * element, matching ssm_scan_forward_batch_with_stability_probe's layout.
 */
void ssm_scan_backward_f1_batch(
    const float   *A,
    const float   *A_per_step,   /* [seq_len × d_state] or NULL (shared) */
    const float   *dt,
    const float   *B,
    int            B_stride,
    const float   *C,
    const float   *x,
    const float   *h_final,      /* [batch_size × d_state] */
    const float   *dy,
    const uint8_t *boundary_mask,
    float         *dA,
    float         *dA_per_step,  /* [batch_size × seq_len × d_state] or NULL */
    float         *dB,
    float         *dC,
    float         *dx,
    float         *dh_out,
    const float   *C_bilinear,   /* [d_state] or NULL */
    float         *dC_bilinear,  /* [d_state] or NULL */
    const float   *stability_probe_cum, /* [batch_size × d_state] or NULL */
    int batch_size,
    int seq_len,
    int d_state,
    int d_inner,
    const float   *u,            /* [d_state] or NULL */
    const float   *v,            /* [d_state] or NULL */
    float         *du,           /* [d_state] or NULL */
    float         *dv            /* [d_state] or NULL */
);

/*
 * gp_kl_welford_update
 *
 * Online Welford accumulator update for one SSM layer at one timestep.
 * Called by the SSM scan loop (ssm_scan_forward) after computing h[t].
 * NOT called from loss.c — declared here so ssm_scan.c can include loss.h.
 *
 * m    — the GpKlMoments accumulator for this layer (read+write)
 * h    — h[t], the SSM hidden state vector at this timestep [d_state floats]
 * d_state — must equal V12_D_STATE (128). Passed explicitly to avoid coupling.
 */
void gp_kl_welford_update(GpKlMoments *m, const float *h, int d_state, int t);

#endif /* SSM_SCAN_H */
