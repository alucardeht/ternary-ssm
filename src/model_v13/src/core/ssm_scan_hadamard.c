/*
 * ssm_scan_hadamard.c — Hadamard-Diagonal-Hadamard (HDH) Spectral Masks
 *
 * Replaces dense convolution matrices with O(N log N) Walsh-Hadamard Transforms 
 * and ternary diagonal masking.
 * Validated in R505 (Candidate A02).
 */

#include "ssm_scan.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

void wht_inplace(float *v, int n) {
    assert(n > 0 && (n & (n - 1)) == 0);
    for (int h = 1; h < n; h *= 2) {
        for (int i = 0; i < n; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = v[j];
                float b = v[j + h];
                v[j]     = a + b;
                v[j + h] = a - b;
            }
        }
    }
    float scale = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; i++) {
        v[i] *= scale;
    }
}

static int hadamard_initialized = 0;

void ssm_scan_forward(
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
    const float   *u,
    const float   *v
) {
    if (!hadamard_initialized) {
        printf("[R505] Hadamard-Diagonal-Hadamard Scan active\n");
        hadamard_initialized = 1;
    }

    float h_temp[V12_MAX_D_STATE];
    memcpy(h_temp, h, d_state * sizeof(float));

    for (int t = 0; t < seq_len; t++) {
        if (boundary_mask && boundary_mask[t]) {
            memset(h_temp, 0, d_state * sizeof(float));
        }

        const float *x_t = x + t * d_inner;
        const float *B_t = B + t * B_stride;

        // 1. WHT (Pre-transform)
        wht_inplace(h_temp, d_state);

        // 2. Ternary Diagonal Masking
        for (int i = 0; i < d_state; i++) {
            float sigma = (A[i] > 0.5f) ? 1.0f : ((A[i] < 0.1f) ? -1.0f : 0.0f);
            h_temp[i] = sigma * h_temp[i] + B_t[i] * x_t[i % d_inner];
        }

        // 3. WHT (Post-transform)
        wht_inplace(h_temp, d_state);

        if (moments) {
            gp_kl_welford_update(moments, h_temp, d_state, t);
        }

        if (h_cache) {
            memcpy(h_cache + t * d_state, h_temp, d_state * sizeof(float));
        }

        float *y_t = y + t * d_inner;
        for (int j = 0; j < d_inner; j++) {
            float acc = 0.0f;
            const float *C_row = C + j * d_state;
            for (int i = 0; i < d_state; i++) acc += C_row[i] * h_temp[i];
            y_t[j] = acc;
        }
    }
    memcpy(h, h_temp, d_state * sizeof(float));
}

void ssm_scan_backward(
    const float   *A, const float   *A_per_step, const float   *dt, const float   *B, int B_stride,
    const float   *C, const float   *x, const float   *h_cache, const float   *dy, const uint8_t *boundary_mask,
    float         *dA, float         *dA_per_step, float         *dB, float         *dC, float         *dx,
    float         *dh_out, int seq_len, int d_state, int d_inner, const float   *C_bilinear, float *dC_bilinear,
    const float   *u, const float   *v, float *du, float *dv, const float *dh_direct
) {
    float dh[V12_MAX_D_STATE] = {0};

    for (int t = seq_len - 1; t >= 0; t--) {
        const float *x_t = x + t * d_inner;
        const float *dy_t = dy + t * d_inner;
        const float *h_t = h_cache + t * d_state;

        for (int i = 0; i < d_state; i++) {
            float acc = 0.0f;
            for (int j = 0; j < d_inner; j++) acc += C[j * d_state + i] * dy_t[j];
            dh[i] += acc;
        }

        for (int j = 0; j < d_inner; j++) {
            for (int i = 0; i < d_state; i++) dC[j * d_state + i] += dy_t[j] * h_t[i];
        }

        if (B_stride == 0) {
            for (int i = 0; i < d_state; i++) dB[i] += dh[i] * x_t[i % d_inner];
        } else {
            float *dB_t = dB + t * d_state;
            for (int i = 0; i < d_state; i++) dB_t[i] = dh[i] * x_t[i % d_inner];
        }

        float *dx_t = dx + t * d_inner;
        memset(dx_t, 0, d_inner * sizeof(float));
        const float *B_t = B + t * B_stride;
        for (int i = 0; i < d_state; i++) dx_t[i % d_inner] += B_t[i] * dh[i];

        wht_inplace(dh, d_state);
        for (int i = 0; i < d_state; i++) {
            float sigma = (A[i] > 0.5f) ? 1.0f : ((A[i] < 0.1f) ? -1.0f : 0.0f);
            dh[i] *= sigma;
        }
        wht_inplace(dh, d_state);
    }
    if (dh_out) memcpy(dh_out, dh, d_state * sizeof(float));
}

void ssm_scan_forward_batch(
    const float *A, const float *A_per_step, const float *dt, const float *B, int B_stride,
    const float *C, const float *x, float *h, float *y, float *h_cache,
    GpKlMoments *moments, const uint8_t *boundary_mask, int batch_size,
    int seq_len, int d_state, int d_inner, const float *C_bilinear, const float *u, const float *v)
{
    for (int b = 0; b < batch_size; b++) {
        ssm_scan_forward(A, A_per_step, dt, B, B_stride, C, x + b * seq_len * d_inner,
                         h + b * d_state, y + b * seq_len * d_inner,
                         h_cache ? (h_cache + b * seq_len * d_state) : NULL,
                         moments ? (moments + b) : NULL, boundary_mask, seq_len, d_state, d_inner, C_bilinear, u, v);
    }
}

void ssm_scan_backward_batch(
    const float *A, const float *A_per_step, const float *dt, const float *B, int B_stride,
    const float *C, const float *x, const float *h_cache, const float *dy, const uint8_t *boundary_mask,
    float *dA, float *dA_per_step, float *dB, float *dC, float *dx, float *dh_out,
    int batch_size, int seq_len, int d_state, int d_inner, const float *C_bilinear,
    float *dC_bilinear, const float *u, const float *v, float *du, float *dv, const float *dh_direct)
{
    for (int b = 0; b < batch_size; b++) {
        ssm_scan_backward(A, A_per_step, dt, B, B_stride, C, x + b * seq_len * d_inner,
                          h_cache + b * seq_len * d_state, dy + b * seq_len * d_inner, boundary_mask, dA,
                          dA_per_step ? (dA_per_step + b * seq_len * d_state) : NULL,
                          dB + (B_stride ? b * seq_len * d_state : 0), dC, dx + b * seq_len * d_inner,
                          dh_out ? (dh_out + b * d_state) : NULL, seq_len, d_state, d_inner, C_bilinear, dC_bilinear, u, v, du, dv,
                          dh_direct ? (dh_direct + b * seq_len * d_state) : NULL);
    }
}
