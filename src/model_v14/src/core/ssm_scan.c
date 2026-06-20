/*
 * ssm_scan.c — SSM recurrence kernel implementation
 *
 * Implements diagonal SSM scan with element-wise state updates and GEMV output.
 * Supports forward and backward passes, batching, and optional GP-KL statistics.
 *
 * CRITICAL: A is stored as direct decay coefficients (A[i] in (0,1)),
 * NOT log-space. The spec pseudocode used A_log for clarity; this implementation
 * uses the actual stored representation from types.h SSMLayer.A.
 */

#include <math.h>      /* expf, logf, sqrtf, fabsf */
#include <stdio.h>     /* printf for [F1 PROBE] instrumentation */
#include <string.h>    /* memcpy, memset */
#include <assert.h>    /* assert */
#include <stdlib.h>    /* malloc, free, calloc */
#include "ssm_scan.h"  /* includes types.h */

#if defined(_OPENMP)
  #include <omp.h>
#else
  #define omp_get_max_threads() 1
  #define omp_get_thread_num() 0
#endif

/* SIMD backend selection (compile-time) */
#if defined(__AVX2__)
  #include <immintrin.h>
  #define SSM_SIMD_AVX2
#elif defined(__ARM_NEON)
  #include <arm_neon.h>
  #define SSM_SIMD_NEON
#else
  #define SSM_SIMD_SCALAR
#endif

/* ============================================================================
 * INTERNAL HELPERS
 * ============================================================================ */

/*
 * wht_inplace — Radix-2 iterative Walsh-Hadamard Transform
 *
 * In-place transformation of a float vector of length n (must be power of 2).
 * Applies the butterfly operations, then normalises by dividing by sqrt(n).
 * WHT is its own inverse: wht(wht(v)) = v (with proper normalisation).
 */
void wht_inplace(float *v, int n) {
    assert(n > 0 && (n & (n - 1)) == 0);  /* n is power of 2 */

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

#ifdef SSM_SIMD_AVX2
static inline float hsum256(__m256 v) {
    __m256 hi64 = _mm256_permute2f128_ps(v, v, 0x1);
    __m256 sum64 = _mm256_add_ps(v, hi64);
    __m128 sum128 = _mm256_castps256_ps128(sum64);
    __m128 hi32 = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sum32 = _mm_add_ps(sum128, hi32);
    hi32 = _mm_shuffle_ps(sum32, sum32, _MM_SHUFFLE(1, 0, 3, 2));
    sum32 = _mm_add_ps(sum32, hi32);
    return _mm_cvtss_f32(sum32);
}
#endif

/* ============================================================================
 * FORWARD SCAN
 * ============================================================================ */

#if V12_MRLC_ENABLED
static inline void apply_mrlc(float *h, int d_state) {
    float energy = 0.0f;
    for (int i = 0; i < d_state; i++) {
        energy += h[i] * h[i];
    }
    if (energy > V12_MRLC_ENERGY_LIMIT && energy > 0.0f) {
        float scale = sqrtf(V12_MRLC_ENERGY_LIMIT / energy);
        for (int i = 0; i < d_state; i++) {
            h[i] *= scale;
        }
    }
}
#endif


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
    assert(A != NULL);
    assert(x != NULL);
    assert(h != NULL);
    assert(y != NULL);
    assert(d_state > 0 && d_state <= V12_MAX_D_STATE);
    assert(d_inner > 0);
    assert(seq_len > 0);
    assert(d_state % 8 == 0);

#if V12_SISSF_ENABLED
    {
        // Symplectic Integrator State-Space Flow (SIS-SF) forward scan
        float A_eff[V12_MAX_D_STATE];
        if (A_per_step == NULL && dt == NULL) {
            for (int i = 0; i < d_state; i++) A_eff[i] = A[i];
        }
        for (int t = 0; t < seq_len; t++) {
            const float *x_t = x + t * d_inner;
            const float *B_t = B + t * B_stride;
            float       *y_t = y + t * d_inner;
            if (boundary_mask != NULL && boundary_mask[t]) {
                memset(h, 0, (size_t)d_state * sizeof(float));
            }
            if (A_per_step != NULL) {
                const float *a_t = A_per_step + t * d_state;
                for (int i = 0; i < d_state; i++) A_eff[i] = a_t[i];
            } else if (dt != NULL) {
                for (int i = 0; i < d_state; i++) {
                    A_eff[i] = expf(dt[t] * logf(A[i]));
                }
            }
            if (h_cache != NULL && seq_len % V12_CKPT_STRIDE == 0) {
                if (t % V12_CKPT_STRIDE == 0) {
                    size_t ckpt_off = (size_t)(t / V12_CKPT_STRIDE) * d_state;
                    memcpy(h_cache + ckpt_off, h, d_state * sizeof(float));
                }
            }
            
            float w_prev = 0.0f;
            if (u != NULL && v != NULL) {
                for (int i = 0; i < d_state; i++) w_prev += v[i] * h[i];
            }

            float dt_val = (dt != NULL) ? dt[t] : 1.0f;
            int half_d = d_state / 2;

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < half_d; i++) {
                float q_prev = h[i];
                float p_prev = h[i + half_d];
                float w_q = A_eff[i];
                float w_x = B_t[i];
                float m_inv = A_eff[i + half_d];
                float x_val = x_t[i];

                float v_prev = w_q * q_prev + w_x * x_val;
                float tanh_v_prev = tanhf(v_prev);
                float grad_q_prev = -w_q * (1.0f - tanh_v_prev * tanh_v_prev);
                float p_half = p_prev - (dt_val * 0.5f) * grad_q_prev;

                float q_new = q_prev + dt_val * m_inv * p_half;
                if (u != NULL && v != NULL) {
                    q_new += w_prev * u[i];
                }

                float v_new = w_q * q_new + w_x * x_val;
                float tanh_v_new = tanhf(v_new);
                float grad_q_new = -w_q * (1.0f - tanh_v_new * tanh_v_new);
                float p_new = p_half - (dt_val * 0.5f) * grad_q_new;
                if (u != NULL && v != NULL) {
                    p_new += w_prev * u[i + half_d];
                }

                h[i] = q_new;
                h[i + half_d] = p_new;
            }

#if V12_MRLC_ENABLED
            apply_mrlc(h, d_state);
#endif

            if (moments != NULL) {
                gp_kl_welford_update(moments, h, d_state, t);
            }
            if (h_cache != NULL && seq_len % V12_CKPT_STRIDE != 0) {
                memcpy(h_cache + (size_t)t * d_state, h, d_state * sizeof(float));
            }

            // Output projection Y = C @ H
            float h_q05[V12_MAX_D_STATE];
            float p4_sum = 0.0f;
            for (int i = 0; i < d_state; i++) {
                float h2 = h[i] * h[i];
                p4_sum += h2 * h2;
                h_q05[i] = h[i];
            }
            float norm_p4 = powf(p4_sum, 0.25f);
            float q05_scale = 1.0f / fmaxf(1.0f, norm_p4 / 10.0f);
            for (int i = 0; i < d_state; i++) h_q05[i] *= q05_scale;
            for (int j = 0; j < d_inner; j++) {
                float acc = 0.0f;
                const float *C_row = C + j * d_state;
                for (int i = 0; i < d_state; i++) acc += C_row[i] * h_q05[i];
                y_t[j] = acc;
            }
        }
        if (h_cache != NULL && seq_len % V12_CKPT_STRIDE == 0) {
            size_t final_ckpt_off = (size_t)(seq_len / V12_CKPT_STRIDE) * d_state;
            memcpy(h_cache + final_ckpt_off, h, d_state * sizeof(float));
        }
        return;
    }
#endif

    int L = (int)sqrtf((float)seq_len);
    if (L < 1) L = 1;
    int M = (seq_len + L - 1) / L;

    float *A_eff_seq = malloc((size_t)seq_len * d_state * sizeof(float));
    float *B_seq = malloc((size_t)seq_len * d_state * sizeof(float));
    float *H = malloc((size_t)M * d_state * sizeof(float));
    float *h_temp_moments = moments ? malloc((size_t)seq_len * d_state * sizeof(float)) : NULL;

    if (!A_eff_seq || !B_seq || !H || (moments && !h_temp_moments)) {
        free(A_eff_seq); free(B_seq); free(H); free(h_temp_moments);
        
        // Fallback to sequential scan
        float A_eff[V12_MAX_D_STATE];
        if (A_per_step == NULL && dt == NULL) {
            for (int i = 0; i < d_state; i++) A_eff[i] = A[i];
        }
        for (int t = 0; t < seq_len; t++) {
            const float *x_t  = x + t * d_inner;
            const float *B_t  = B + t * B_stride;
            float       *y_t  = y + t * d_inner;
            if (boundary_mask != NULL && boundary_mask[t]) {
                memset(h, 0, (size_t)d_state * sizeof(float));
            }
            if (A_per_step != NULL) {
                const float *a_t = A_per_step + t * d_state;
                for (int i = 0; i < d_state; i++) A_eff[i] = a_t[i];
            } else if (dt != NULL) {
                for (int i = 0; i < d_state; i++) {
                    A_eff[i] = expf(dt[t] * logf(A[i]));
                }
            }
            if (h_cache != NULL && seq_len % V12_CKPT_STRIDE == 0) {
                if (t % V12_CKPT_STRIDE == 0) {
                    size_t ckpt_off = (size_t)(t / V12_CKPT_STRIDE) * d_state;
                    memcpy(h_cache + ckpt_off, h, d_state * sizeof(float));
                }
            }
            float w_prev = 0.0f;
            if (u != NULL && v != NULL) {
                for (int i = 0; i < d_state; i++) w_prev += v[i] * h[i];
            }
            if (C_bilinear != NULL) {
                for (int i = 0; i < d_state; i++) {
                    float h_prev_i = h[i];
                    h[i] = A_eff[i] * h_prev_i + B_t[i] * x_t[i % d_inner] + C_bilinear[i] * h_prev_i * x_t[i % d_inner];
                    if (u != NULL && v != NULL) h[i] += w_prev * u[i];
                }
            } else {
                for (int i = 0; i < d_state; i++) {
                    h[i] = A_eff[i] * h[i] + B_t[i] * x_t[i % d_inner];
                    if (u != NULL && v != NULL) h[i] += w_prev * u[i];
                }
            }
#if V12_MRLC_ENABLED
            apply_mrlc(h, d_state);
#endif
            if (moments != NULL) {
                gp_kl_welford_update(moments, h, d_state, t);
            }
            if (h_cache != NULL && seq_len % V12_CKPT_STRIDE != 0) {
                memcpy(h_cache + (size_t)t * d_state, h, d_state * sizeof(float));
            }
            float h_q05[V12_MAX_D_STATE];
            float p4_sum = 0.0f;
            for (int i = 0; i < d_state; i++) {
                float h2 = h[i] * h[i];
                p4_sum += h2 * h2;
                h_q05[i] = h[i];
            }
            float norm_p4 = powf(p4_sum, 0.25f);
            float q05_scale = 1.0f / fmaxf(1.0f, norm_p4 / 10.0f);
            for (int i = 0; i < d_state; i++) h_q05[i] *= q05_scale;
            for (int j = 0; j < d_inner; j++) {
                float acc = 0.0f;
                const float *C_row = C + j * d_state;
                for (int i = 0; i < d_state; i++) acc += C_row[i] * h_q05[i];
                y_t[j] = acc;
            }
        }
        if (h_cache != NULL && seq_len % V12_CKPT_STRIDE == 0) {
            size_t final_ckpt_off = (size_t)(seq_len / V12_CKPT_STRIDE) * d_state;
            memcpy(h_cache + final_ckpt_off, h, d_state * sizeof(float));
        }
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < seq_len; t++) {
        if (A_per_step != NULL) {
            const float *a_t = A_per_step + t * d_state;
            for (int i = 0; i < d_state; i++) A_eff_seq[t * d_state + i] = a_t[i];
        } else if (dt != NULL) {
            for (int i = 0; i < d_state; i++) {
                A_eff_seq[t * d_state + i] = expf(dt[t] * logf(A[i]));
            }
        } else {
            for (int i = 0; i < d_state; i++) A_eff_seq[t * d_state + i] = A[i];
        }
        const float *B_t = B + t * B_stride;
        for (int i = 0; i < d_state; i++) B_seq[t * d_state + i] = B_t[i];
    }

    memcpy(H, h, d_state * sizeof(float));
    float h_temp[V12_MAX_D_STATE];
    memcpy(h_temp, h, d_state * sizeof(float));

    for (int t = 0; t < seq_len; t++) {
        if (boundary_mask != NULL && boundary_mask[t]) {
            memset(h_temp, 0, d_state * sizeof(float));
        }
        float w_prev = 0.0f;
        if (u != NULL && v != NULL) {
            for (int i = 0; i < d_state; i++) w_prev += v[i] * h_temp[i];
        }
        const float *x_t = x + t * d_inner;
        const float *B_t = B_seq + t * d_state;
        for (int i = 0; i < d_state; i++) {
            float h_prev_i = h_temp[i];
            float a_val = A_eff_seq[t * d_state + i];
            if (C_bilinear != NULL) {
                a_val += C_bilinear[i] * x_t[i % d_inner];
            }
            h_temp[i] = a_val * h_prev_i + B_t[i] * x_t[i % d_inner];
            if (u != NULL && v != NULL) {
                h_temp[i] += w_prev * u[i];
            }
        }
        if ((t + 1) % L == 0 && (t + 1) < seq_len) {
            int b = (t + 1) / L;
            memcpy(H + b * d_state, h_temp, d_state * sizeof(float));
        }
    }

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < M; b++) {
        int t0 = b * L;
        int t1 = (b + 1) * L;
        if (t1 > seq_len) t1 = seq_len;
        
        float h_local[V12_MAX_D_STATE];
        memcpy(h_local, H + b * d_state, d_state * sizeof(float));
        
        for (int t = t0; t < t1; t++) {
            if (h_cache != NULL && seq_len % V12_CKPT_STRIDE == 0) {
                if (t % V12_CKPT_STRIDE == 0) {
                    size_t ckpt_off = (size_t)(t / V12_CKPT_STRIDE) * d_state;
                    memcpy(h_cache + ckpt_off, h_local, d_state * sizeof(float));
                }
            }
            
            if (boundary_mask != NULL && boundary_mask[t]) {
                memset(h_local, 0, d_state * sizeof(float));
            }
            
            float w_prev = 0.0f;
            if (u != NULL && v != NULL) {
                for (int i = 0; i < d_state; i++) w_prev += v[i] * h_local[i];
            }
            
            const float *x_t = x + t * d_inner;
            const float *B_t = B_seq + t * d_state;
            for (int i = 0; i < d_state; i++) {
                float h_prev_i = h_local[i];
                float a_val = A_eff_seq[t * d_state + i];
                if (C_bilinear != NULL) {
                    a_val += C_bilinear[i] * x_t[i % d_inner];
                }
                h_local[i] = a_val * h_prev_i + B_t[i] * x_t[i % d_inner];
                if (u != NULL && v != NULL) {
                    h_local[i] += w_prev * u[i];
                }
            }
            
#if V12_MRLC_ENABLED
            apply_mrlc(h_local, d_state);
#endif

            if (h_temp_moments != NULL) {
                memcpy(h_temp_moments + t * d_state, h_local, d_state * sizeof(float));
            }
            
            if (h_cache != NULL && seq_len % V12_CKPT_STRIDE != 0) {
                memcpy(h_cache + (size_t)t * d_state, h_local, d_state * sizeof(float));
            }
            
            float *y_t = y + t * d_inner;
            float h_q05[V12_MAX_D_STATE];
            float p4_sum = 0.0f;
            for (int i = 0; i < d_state; i++) {
                float h2 = h_local[i] * h_local[i];
                p4_sum += h2 * h2;
                h_q05[i] = h_local[i];
            }
            float norm_p4 = powf(p4_sum, 0.25f);
            float q05_scale = 1.0f / fmaxf(1.0f, norm_p4 / 10.0f);
            for (int i = 0; i < d_state; i++) h_q05[i] *= q05_scale;
#ifdef SSM_SIMD_AVX2
            for (int j = 0; j < d_inner; j += 4) {
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();
                for (int i = 0; i < d_state; i += 8) {
                    __m256 vh = _mm256_loadu_ps(h_q05 + i);
                    if (j < d_inner) {
                        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(C + j * d_state + i), vh, acc0);
                    }
                    if (j + 1 < d_inner) {
                        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(C + (j + 1) * d_state + i), vh, acc1);
                    }
                    if (j + 2 < d_inner) {
                        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(C + (j + 2) * d_state + i), vh, acc2);
                    }
                    if (j + 3 < d_inner) {
                        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(C + (j + 3) * d_state + i), vh, acc3);
                    }
                }
                if (j < d_inner) y_t[j] = hsum256(acc0);
                if (j + 1 < d_inner) y_t[j + 1] = hsum256(acc1);
                if (j + 2 < d_inner) y_t[j + 2] = hsum256(acc2);
                if (j + 3 < d_inner) y_t[j + 3] = hsum256(acc3);
            }
#else
            for (int j = 0; j < d_inner; j++) {
                float acc = 0.0f;
                const float *C_row = C + j * d_state;
                for (int i = 0; i < d_state; i++) acc += C_row[i] * h_q05[i];
                y_t[j] = acc;
            }
#endif
        }
    }

    if (moments != NULL && h_temp_moments != NULL) {
        for (int t = 0; t < seq_len; t++) {
            gp_kl_welford_update(moments, h_temp_moments + t * d_state, d_state, t);
        }
    }

    if (h_cache != NULL && seq_len % V12_CKPT_STRIDE == 0) {
        size_t final_ckpt_off = (size_t)(seq_len / V12_CKPT_STRIDE) * d_state;
        memcpy(h_cache + final_ckpt_off, h_temp, d_state * sizeof(float));
    }

    memcpy(h, h_temp, d_state * sizeof(float));

    free(A_eff_seq);
    free(B_seq);
    free(H);
    free(h_temp_moments);
}

/* ============================================================================
 * BACKWARD SCAN
 * ============================================================================ */

void ssm_scan_backward(
    const float   *A,
    const float   *A_per_step,
    const float   *dt,
    const float   *B,
    int            B_stride,
    const float   *C,
    const float   *x,
    const float   *h_cache,
    const float   *dy,
    const uint8_t *boundary_mask,
    float         *dA,
    float         *dA_per_step,
    float         *dB,
    float         *dC,
    float         *dx,
    float         *dh_out,
    int seq_len,
    int d_state,
    int d_inner,
    const float   *C_bilinear,
    float         *dC_bilinear,
    const float   *u,
    const float   *v,
    float         *du,
    float         *dv,
    const float   *dh_direct
) {
    assert(h_cache != NULL);
    assert(dA != NULL);
    assert(dB != NULL);
    assert(dC != NULL);
    assert(dx != NULL);
    assert(d_state > 0 && d_state <= V12_MAX_D_STATE);
    assert(d_inner > 0);
    assert(seq_len > 0);

#if V12_SISSF_ENABLED
    {
        // Symplectic Integrator State-Space Flow (SIS-SF) backward scan
        float dh[V12_MAX_D_STATE];
        memset(dh, 0, d_state * sizeof(float));
        float A_eff[V12_MAX_D_STATE];
        if (A_per_step == NULL && dt == NULL) {
            for (int i = 0; i < d_state; i++) A_eff[i] = A[i];
        }
        int half_d = d_state / 2;

        float local_h[V12_MAX_D_STATE];
        if (h_cache != NULL) {
            // Only read the final state if provided (in practice, finding 14 drops caching,
            // so we'd only pass the final state). Since the CPU fallback currently passes a full
            // array, we just take the last element.
            memcpy(local_h, h_cache + (seq_len - 1) * d_state, d_state * sizeof(float));
        } else {
            memset(local_h, 0, d_state * sizeof(float));
        }

        for (int t = seq_len - 1; t >= 0; t--) {
            const float *x_t   = x + t * d_inner;
            const float *B_t   = B + t * B_stride;
            const float *dy_t  = dy + t * d_inner;
            int boundary_at_t = (boundary_mask != NULL && boundary_mask[t]);
            
            if (A_per_step != NULL) {
                const float *a_t = A_per_step + t * d_state;
                for (int i = 0; i < d_state; i++) A_eff[i] = a_t[i];
            } else if (dt != NULL) {
                for (int i = 0; i < d_state; i++) {
                    A_eff[i] = expf(dt[t] * logf(A[i]));
                }
            }
            
            // Output projection gradient: dh += C^T dy_t
            for (int i = 0; i < d_state; i++) {
                float acc = 0.0f;
                for (int j = 0; j < d_inner; j++) acc += C[j * d_state + i] * dy_t[j];
                dh[i] += acc;
            }
            if (dh_direct != NULL) {
                const float *dh_dir_t = dh_direct + t * d_state;
                for (int i = 0; i < d_state; i++) dh[i] += dh_dir_t[i];
            }
            for (int j = 0; j < d_inner; j++) {
                for (int i = 0; i < d_state; i++) dC[j * d_state + i] += dy_t[j] * local_h[i];
            }
            
            float dt_val = (dt != NULL) ? dt[t] : 1.0f;
            float next_dh[V12_MAX_D_STATE];
            memset(next_dh, 0, d_state * sizeof(float));
            float prev_h[V12_MAX_D_STATE];

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < half_d; i++) {
                float dq_n = dh[i];
                float dp_n = dh[i + half_d];

                float q_n = local_h[i];
                float p_n = local_h[i + half_d];

                float w_q = A_eff[i];
                float w_x = B_t[i];
                float m_inv = A_eff[i + half_d];
                float x_val = x_t[i];

                // 1. Reconstruct state backwards:
                // g1(q_n) = -w_q * (1 - tanh^2(w_q*q_n + w_x*x_val))
                float v1 = w_q * q_n + w_x * x_val;
                float tanh_v1 = tanhf(v1);
                float g1 = -w_q * (1.0f - tanh_v1 * tanh_v1);
                float p_half = p_n + (dt_val * 0.5f) * g1;

                float q_p = q_n - dt_val * m_inv * p_half;

                float v0 = w_q * q_p + w_x * x_val;
                float tanh_v0 = tanhf(v0);
                float g0 = -w_q * (1.0f - tanh_v0 * tanh_v0);
                float p_p = p_half + (dt_val * 0.5f) * g0;

                prev_h[i] = q_p;
                prev_h[i + half_d] = p_p;

                // 2. Backward gradients
                // dp_half = dp_n; dg1 = -0.5*dt * dp_n
                float dp_half = dp_n;
                float dg1 = -(dt_val * 0.5f) * dp_n;

                // dw_q, dx, dq_n from g1
                float dw_q = dg1 * (-(1.0f - tanh_v1 * tanh_v1));
                float dtanh_v1 = dg1 * (2.0f * w_q * tanh_v1);
                float dv1 = dtanh_v1 * (1.0f - tanh_v1 * tanh_v1);

                dw_q += dv1 * q_n;
                float dq_n_acc = dq_n + dv1 * w_q;
                float dw_x = dv1 * x_val;
                float dx_val = dv1 * w_x;

                float dq_p = dq_n_acc;
                float dm_inv = dq_n_acc * (dt_val * p_half);
                dp_half += dq_n_acc * (dt_val * m_inv);

                float dp_p = dp_half;
                float dg0 = -(dt_val * 0.5f) * dp_half;

                dw_q += dg0 * (-(1.0f - tanh_v0 * tanh_v0));
                float dtanh_v0 = dg0 * (2.0f * w_q * tanh_v0);
                float dv0 = dtanh_v0 * (1.0f - tanh_v0 * tanh_v0);

                dw_q += dv0 * q_p;
                dq_p += dv0 * w_q;
                dw_x += dv0 * x_val;
                dx_val += dv0 * w_x;

                if (A_per_step != NULL) {
                    if (dA_per_step != NULL && !boundary_at_t) {
                        float *da_t = dA_per_step + t * d_state;
                        da_t[i] = dw_q;
                        da_t[i + half_d] = dm_inv;
                    }
                } else {
                    if (!boundary_at_t) {
                        dA[i] += dw_q;
                        dA[i + half_d] += dm_inv;
                    }
                }

                if (B_stride == 0) {
                    dB[i] += dw_x;
                } else {
                    float *dB_t = dB + t * d_state;
                    dB_t[i] = dw_x;
                }

                dx[t * d_inner + i] += dx_val;

                if (!boundary_at_t) {
                    next_dh[i] = dq_p;
                    next_dh[i + half_d] = dp_p;
                }
            }

            if (boundary_at_t) {
                memset(dh, 0, d_state * sizeof(float));
            } else {
                memcpy(dh, next_dh, d_state * sizeof(float));
                memcpy(local_h, prev_h, d_state * sizeof(float));
            }
        }
        if (dh_out != NULL) {
            memcpy(dh_out, dh, d_state * sizeof(float));
        }
        return;
    }
#endif

    int L = (int)sqrtf((float)seq_len);
    if (L < 1) L = 1;
    int M = (seq_len + L - 1) / L;

    float *DH = malloc((size_t)M * d_state * sizeof(float));
    float *A_eff_seq = malloc((size_t)seq_len * d_state * sizeof(float));
    float *B_seq = malloc((size_t)seq_len * d_state * sizeof(float));

    int max_threads = omp_get_max_threads();
    float *dA_threads = calloc((size_t)max_threads * d_state, sizeof(float));
    float *dC_threads = calloc((size_t)max_threads * d_inner * d_state, sizeof(float));
    float *dC_bil_threads = C_bilinear ? calloc((size_t)max_threads * d_state, sizeof(float)) : NULL;
    float *du_threads = u ? calloc((size_t)max_threads * d_state, sizeof(float)) : NULL;
    float *dv_threads = v ? calloc((size_t)max_threads * d_state, sizeof(float)) : NULL;

    if (!DH || !A_eff_seq || !B_seq || !dA_threads || !dC_threads || 
        (C_bilinear && !dC_bil_threads) || (u && !du_threads) || (v && !dv_threads)) {
        free(DH); free(A_eff_seq); free(B_seq);
        free(dA_threads); free(dC_threads); free(dC_bil_threads);
        free(du_threads); free(dv_threads);

        // Fallback to sequential scan
        float dh[V12_MAX_D_STATE];
        memset(dh, 0, d_state * sizeof(float));
        float A_eff[V12_MAX_D_STATE];
        if (A_per_step == NULL && dt == NULL) {
            for (int i = 0; i < d_state; i++) A_eff[i] = A[i];
        }
        for (int t = seq_len - 1; t >= 0; t--) {
            const float *x_t   = x + t * d_inner;
            const float *B_t   = B + t * B_stride;
            const float *dy_t  = dy + t * d_inner;
            const float *h_t   = h_cache + t * d_state;
            const float *h_prev = (t > 0) ? (h_cache + (t - 1) * d_state) : NULL;
            int boundary_at_t = (boundary_mask != NULL && boundary_mask[t]);
            if (A_per_step != NULL) {
                const float *a_t = A_per_step + t * d_state;
                for (int i = 0; i < d_state; i++) A_eff[i] = a_t[i];
            } else if (dt != NULL) {
                for (int i = 0; i < d_state; i++) {
                    A_eff[i] = expf(dt[t] * logf(A[i]));
                }
            }
            for (int i = 0; i < d_state; i++) {
                float acc = 0.0f;
                for (int j = 0; j < d_inner; j++) acc += C[j * d_state + i] * dy_t[j];
                dh[i] += acc;
            }
            if (dh_direct != NULL) {
                const float *dh_dir_t = dh_direct + t * d_state;
                for (int i = 0; i < d_state; i++) dh[i] += dh_dir_t[i];
            }
            for (int j = 0; j < d_inner; j++) {
                for (int i = 0; i < d_state; i++) dC[j * d_state + i] += dy_t[j] * h_t[i];
            }
            if (A_per_step != NULL) {
                if (dA_per_step != NULL && h_prev != NULL && !boundary_at_t) {
                    float *da_t = dA_per_step + t * d_state;
                    for (int i = 0; i < d_state; i++) da_t[i] = dh[i] * h_prev[i];
                }
            } else if (dt != NULL && h_prev != NULL && !boundary_at_t) {
                for (int i = 0; i < d_state; i++) dA[i] += dt[t] * A_eff[i] * h_prev[i] * dh[i];
            } else if (h_prev != NULL && !boundary_at_t) {
                for (int i = 0; i < d_state; i++) dA[i] += h_prev[i] * dh[i];
            }
            if (B_stride == 0) {
                for (int i = 0; i < d_state; i++) dB[i] += dh[i] * x_t[i % d_inner];
            } else {
                float *dB_t = dB + t * d_state;
                for (int i = 0; i < d_state; i++) dB_t[i] = dh[i] * x_t[i % d_inner];
            }
            float *dx_t = dx + t * d_inner;
            memset(dx_t, 0, d_inner * sizeof(float));
            for (int i = 0; i < d_state; i++) dx_t[i % d_inner] += B_t[i] * dh[i];
            if (C_bilinear != NULL && h_prev != NULL && !boundary_at_t) {
                for (int i = 0; i < d_state; i++) {
                    float h_prev_i = h_prev[i];
                    dC_bilinear[i] += dh[i] * h_prev_i * x_t[i % d_inner];
                    dx_t[i % d_inner] += C_bilinear[i] * h_prev_i * dh[i];
                }
            }
            float dw_prev = 0.0f;
            float w_prev = 0.0f;
            if (u != NULL && v != NULL) {
                for (int i = 0; i < d_state; i++) dw_prev += dh[i] * u[i];
                if (h_prev != NULL && !boundary_at_t) {
                    for (int i = 0; i < d_state; i++) w_prev += v[i] * h_prev[i];
                    if (du != NULL) {
                        for (int i = 0; i < d_state; i++) du[i] += dh[i] * w_prev;
                    }
                    if (dv != NULL) {
                        for (int i = 0; i < d_state; i++) dv[i] += dw_prev * h_prev[i];
                    }
                }
            }
            if (boundary_mask != NULL && boundary_mask[t]) {
                memset(dh, 0, (size_t)d_state * sizeof(float));
            }
            for (int i = 0; i < d_state; i++) {
                float dh_factor = A_eff[i];
                if (C_bilinear != NULL) dh_factor += C_bilinear[i] * x_t[i % d_inner];
                dh[i] = dh_factor * dh[i];
                if (u != NULL && v != NULL && h_prev != NULL && !boundary_at_t) {
                    dh[i] += dw_prev * v[i];
                }
            }
        }
        if (dh_out != NULL) {
            memcpy(dh_out, dh, d_state * sizeof(float));
        }
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < seq_len; t++) {
        if (A_per_step != NULL) {
            const float *a_t = A_per_step + t * d_state;
            for (int i = 0; i < d_state; i++) A_eff_seq[t * d_state + i] = a_t[i];
        } else if (dt != NULL) {
            for (int i = 0; i < d_state; i++) {
                A_eff_seq[t * d_state + i] = expf(dt[t] * logf(A[i]));
            }
        } else {
            for (int i = 0; i < d_state; i++) A_eff_seq[t * d_state + i] = A[i];
        }
        const float *B_t = B + t * B_stride;
        for (int i = 0; i < d_state; i++) B_seq[t * d_state + i] = B_t[i];
    }

#if V12_DADD_ENABLED
    int use_dadd = (u == NULL && v == NULL);
#else
    int use_dadd = 0;
#endif

    if (use_dadd) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < d_state; i++) {
            float dh_val = 0.0f;
            for (int t = seq_len - 1; t >= 0; t--) {
                const float *dy_t = dy + t * d_inner;
                float acc = 0.0f;
                for (int j = 0; j < d_inner; j++) {
                    acc += C[j * d_state + i] * dy_t[j];
                }
                dh_val += acc;
                if (dh_direct != NULL) {
                    dh_val += dh_direct[t * d_state + i];
                }
                if (boundary_mask != NULL && boundary_mask[t]) {
                    dh_val = 0.0f;
                }
                // Finding 43: Nilpotent Residual Algebraic Collapse (NRAC)
                if ((seq_len - 1 - t) % 32 == 31) {
                    dh_val = 0.0f;
                }
                const float *x_t = x + t * d_inner;
                float dh_factor = A_eff_seq[t * d_state + i];
                if (C_bilinear != NULL) {
                    dh_factor += C_bilinear[i] * x_t[i % d_inner];
                }
                dh_val = dh_factor * dh_val;

                if (t % L == 0 && t > 0) {
                    int b = t / L;
                    DH[(b - 1) * d_state + i] = dh_val;
                }
                if (t == 0 && dh_out != NULL) {
                    dh_out[i] = dh_val;
                }
            }
        }
        memset(DH + (M - 1) * d_state, 0, d_state * sizeof(float));

#if V12_DADD_VALIDATION
        // Parity check against sequential run
        float *DH_ref = malloc((size_t)M * d_state * sizeof(float));
        float dh_temp_ref[V12_MAX_D_STATE];
        memset(dh_temp_ref, 0, d_state * sizeof(float));
        if (DH_ref) {
            memset(DH_ref + (M - 1) * d_state, 0, d_state * sizeof(float));
            for (int t = seq_len - 1; t >= 0; t--) {
                const float *dy_t = dy + t * d_inner;
                for (int i = 0; i < d_state; i++) {
                    float acc = 0.0f;
                    for (int j = 0; j < d_inner; j++) acc += C[j * d_state + i] * dy_t[j];
                    dh_temp_ref[i] += acc;
                }
                if (dh_direct != NULL) {
                    const float *dh_dir_t = dh_direct + t * d_state;
                    for (int i = 0; i < d_state; i++) dh_temp_ref[i] += dh_dir_t[i];
                }
                if (boundary_mask != NULL && boundary_mask[t]) {
                    memset(dh_temp_ref, 0, d_state * sizeof(float));
                }
                // Finding 43: Nilpotent Residual Algebraic Collapse (NRAC)
                if ((seq_len - 1 - t) % 32 == 31) {
                    memset(dh_temp_ref, 0, d_state * sizeof(float));
                }
                const float *x_t = x + t * d_inner;
                for (int i = 0; i < d_state; i++) {
                    float dh_factor = A_eff_seq[t * d_state + i];
                    if (C_bilinear != NULL) {
                        dh_temp_ref[i] = (A_eff_seq[t * d_state + i] + C_bilinear[i] * x_t[i % d_inner]) * dh_temp_ref[i];
                        continue;
                    }
                    dh_temp_ref[i] = dh_factor * dh_temp_ref[i];
                }
                if (t % L == 0 && t > 0) {
                    int b = t / L;
                    memcpy(DH_ref + (b - 1) * d_state, dh_temp_ref, d_state * sizeof(float));
                }
            }
            float max_diff = 0.0f;
            for (int k = 0; k < M * d_state; k++) {
                float diff = fabsf(DH[k] - DH_ref[k]);
                if (diff > max_diff) max_diff = diff;
            }
            if (max_diff > 1e-6f) {
                printf("[DADD VALIDATION ERROR] max_diff=%.8e\n", (double)max_diff);
            } else {
                printf("[DADD VALIDATION OK] max_diff=%.8e\n", (double)max_diff);
            }
            free(DH_ref);
        }
#endif
    } else {
        float dh_temp[V12_MAX_D_STATE];
        memset(dh_temp, 0, d_state * sizeof(float));
        memset(DH + (M - 1) * d_state, 0, d_state * sizeof(float));

        for (int t = seq_len - 1; t >= 0; t--) {
            const float *dy_t = dy + t * d_inner;
            for (int i = 0; i < d_state; i++) {
                float acc = 0.0f;
                for (int j = 0; j < d_inner; j++) acc += C[j * d_state + i] * dy_t[j];
                dh_temp[i] += acc;
            }
            if (dh_direct != NULL) {
                const float *dh_dir_t = dh_direct + t * d_state;
                for (int i = 0; i < d_state; i++) dh_temp[i] += dh_dir_t[i];
            }
            float dw_prev = 0.0f;
            if (u != NULL && v != NULL) {
                for (int i = 0; i < d_state; i++) dw_prev += dh_temp[i] * u[i];
            }
            if (boundary_mask != NULL && boundary_mask[t]) {
                memset(dh_temp, 0, d_state * sizeof(float));
            }
            // Finding 43: Nilpotent Residual Algebraic Collapse (NRAC)
            if ((seq_len - 1 - t) % 32 == 31) {
                memset(dh_temp, 0, d_state * sizeof(float));
            }
            const float *x_t = x + t * d_inner;
            for (int i = 0; i < d_state; i++) {
                float dh_factor = A_eff_seq[t * d_state + i];
                if (C_bilinear != NULL) {
                    dh_factor += C_bilinear[i] * x_t[i % d_inner];
                }
                dh_temp[i] = dh_factor * dh_temp[i];
                if (u != NULL && v != NULL && t > 0) {
                    int boundary_at_t = (boundary_mask != NULL && boundary_mask[t]);
                    if (!boundary_at_t) dh_temp[i] += dw_prev * v[i];
                }
            }
            if (t % L == 0 && t > 0) {
                int b = t / L;
                memcpy(DH + (b - 1) * d_state, dh_temp, d_state * sizeof(float));
            }
        }
        if (dh_out != NULL) {
            memcpy(dh_out, dh_temp, d_state * sizeof(float));
        }
    }

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < M; b++) {
        int tid = omp_get_thread_num();
        float *my_dA = dA_threads + tid * d_state;
        float *my_dC = dC_threads + tid * d_inner * d_state;
        float *my_dC_bil = dC_bil_threads ? (dC_bil_threads + tid * d_state) : NULL;
        float *my_du = du_threads ? (du_threads + tid * d_state) : NULL;
        float *my_dv = dv_threads ? (dv_threads + tid * d_state) : NULL;
        
        int t0 = b * L;
        int t1 = (b + 1) * L;
        if (t1 > seq_len) t1 = seq_len;
        
        float dh_local[V12_MAX_D_STATE];
        memcpy(dh_local, DH + b * d_state, d_state * sizeof(float));
        
        for (int t = t1 - 1; t >= t0; t--) {
            const float *x_t = x + t * d_inner;
            const float *B_t = B_seq + t * d_state;
            const float *dy_t = dy + t * d_inner;
            const float *h_t = h_cache + t * d_state;
            const float *h_prev = (t > 0) ? (h_cache + (t - 1) * d_state) : NULL;
            int boundary_at_t = (boundary_mask != NULL && boundary_mask[t]);
            
            for (int i = 0; i < d_state; i++) {
                float acc = 0.0f;
                for (int j = 0; j < d_inner; j++) acc += C[j * d_state + i] * dy_t[j];
                dh_local[i] += acc;
            }
            if (dh_direct != NULL) {
                const float *dh_dir_t = dh_direct + t * d_state;
                for (int i = 0; i < d_state; i++) dh_local[i] += dh_dir_t[i];
            }
            for (int j = 0; j < d_inner; j++) {
                for (int i = 0; i < d_state; i++) my_dC[j * d_state + i] += dy_t[j] * h_t[i];
            }
            if (A_per_step != NULL) {
                if (dA_per_step != NULL && h_prev != NULL && !boundary_at_t) {
                    float *da_t = dA_per_step + t * d_state;
                    for (int i = 0; i < d_state; i++) da_t[i] = dh_local[i] * h_prev[i];
                }
            } else if (dt != NULL && h_prev != NULL && !boundary_at_t) {
                for (int i = 0; i < d_state; i++) {
                    my_dA[i] += dt[t] * A_eff_seq[t * d_state + i] * h_prev[i] * dh_local[i];
                }
            } else if (h_prev != NULL && !boundary_at_t) {
                for (int i = 0; i < d_state; i++) {
                    my_dA[i] += h_prev[i] * dh_local[i];
                }
            }
            if (B_stride == 0) {
                #pragma omp critical(dB_crit)
                {
                    for (int i = 0; i < d_state; i++) dB[i] += dh_local[i] * x_t[i % d_inner];
                }
            } else {
                float *dB_t = dB + t * d_state;
                for (int i = 0; i < d_state; i++) dB_t[i] = dh_local[i] * x_t[i % d_inner];
            }
            float *dx_t = dx + t * d_inner;
            memset(dx_t, 0, d_inner * sizeof(float));
            for (int i = 0; i < d_state; i++) dx_t[i % d_inner] += B_t[i] * dh_local[i];
            if (C_bilinear != NULL && h_prev != NULL && !boundary_at_t) {
                for (int i = 0; i < d_state; i++) {
                    float h_prev_i = h_prev[i];
                    if (my_dC_bil) {
                        my_dC_bil[i] += dh_local[i] * h_prev_i * x_t[i % d_inner];
                    }
                    dx_t[i % d_inner] += C_bilinear[i] * h_prev_i * dh_local[i];
                }
            }
            float dw_prev = 0.0f;
            float w_prev = 0.0f;
            if (u != NULL && v != NULL) {
                for (int i = 0; i < d_state; i++) dw_prev += dh_local[i] * u[i];
                if (h_prev != NULL && !boundary_at_t) {
                    for (int i = 0; i < d_state; i++) w_prev += v[i] * h_prev[i];
                    if (my_du) {
                        for (int i = 0; i < d_state; i++) my_du[i] += dh_local[i] * w_prev;
                    }
                    if (my_dv) {
                        for (int i = 0; i < d_state; i++) my_dv[i] += dw_prev * h_prev[i];
                    }
                }
            }
            if (boundary_mask != NULL && boundary_mask[t]) {
                memset(dh_local, 0, d_state * sizeof(float));
            }
            for (int i = 0; i < d_state; i++) {
                float dh_factor = A_eff_seq[t * d_state + i];
                if (C_bilinear != NULL) {
                    dh_factor += C_bilinear[i] * x_t[i % d_inner];
                }
                dh_local[i] = dh_factor * dh_local[i];
                if (u != NULL && v != NULL && h_prev != NULL && !boundary_at_t) {
                    dh_local[i] += dw_prev * v[i];
                }
            }
        }
    }

    for (int tid = 0; tid < max_threads; tid++) {
        for (int i = 0; i < d_state; i++) dA[i] += dA_threads[tid * d_state + i];
        for (int i = 0; i < d_inner * d_state; i++) dC[i] += dC_threads[tid * d_inner * d_state + i];
        if (C_bilinear && dC_bilinear) {
            for (int i = 0; i < d_state; i++) dC_bilinear[i] += dC_bil_threads[tid * d_state + i];
        }
        if (u && du) {
            for (int i = 0; i < d_state; i++) du[i] += du_threads[tid * d_state + i];
        }
        if (v && dv) {
            for (int i = 0; i < d_state; i++) dv[i] += dv_threads[tid * d_state + i];
        }
    }

    /* dh_out copy handled within the DADD/fallback branches */

    free(DH);
    free(A_eff_seq);
    free(B_seq);
    free(dA_threads);
    free(dC_threads);
    free(dC_bil_threads);
    free(du_threads);
    free(dv_threads);
}

/* ============================================================================
 * BATCH WRAPPERS
 * ============================================================================ */

void ssm_scan_forward_batch(
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
    const float   *u,
    const float   *v
) {
    for (int b = 0; b < batch_size; b++) {
        const float   *dt_b     = (dt != NULL) ? (dt + b * seq_len) : NULL;
        const float   *B_b      = (B_stride > 0) ? (B + b * seq_len * d_state) : B;
        const float   *x_b      = x + b * seq_len * d_inner;
        float         *h_b      = h + b * d_state;
        float         *y_b      = y + b * seq_len * d_inner;
        float         *hcache_b = (h_cache != NULL) ? (h_cache + b * seq_len * d_state) : NULL;
        GpKlMoments   *mom_b    = moments;
        const uint8_t *bmask_b  = (boundary_mask != NULL) ? (boundary_mask + b * seq_len) : NULL;

        ssm_scan_forward(A, A_per_step, dt_b, B_b, B_stride, C, x_b, h_b, y_b, hcache_b, mom_b,
                         bmask_b, seq_len, d_state, d_inner, C_bilinear, u, v);
    }
}

void ssm_scan_backward_batch(
    const float   *A,
    const float   *A_per_step,
    const float   *dt,
    const float   *B,
    int            B_stride,
    const float   *C,
    const float   *x,
    const float   *h_cache,
    const float   *dy,
    const uint8_t *boundary_mask,
    float         *dA,
    float         *dA_per_step,
    float         *dB,
    float         *dC,
    float         *dx,
    float         *dh_out,
    int batch_size,
    int seq_len,
    int d_state,
    int d_inner,
    const float   *C_bilinear,
    float         *dC_bilinear,
    const float   *u,
    const float   *v,
    float         *du,
    float         *dv,
    const float   *dh_direct
) {
    for (int b = 0; b < batch_size; b++) {
        const float   *dt_b       = (dt != NULL) ? (dt + b * seq_len) : NULL;
        const float   *B_b        = (B_stride > 0) ? (B + b * seq_len * d_state) : B;
        const float   *x_b        = x + b * seq_len * d_inner;
        const float   *hcache_b   = h_cache + b * seq_len * d_state;
        const float   *dy_b       = dy + b * seq_len * d_inner;
        const uint8_t *bmask_b    = (boundary_mask != NULL) ? (boundary_mask + b * seq_len) : NULL;
        float         *dB_b       = (B_stride > 0) ? (dB + b * seq_len * d_state) : dB;
        float         *dx_b       = dx + b * seq_len * d_inner;
        float         *dA_ps_b    = (dA_per_step != NULL) ? (dA_per_step + b * seq_len * d_state) : NULL;
        float         *dh_out_b   = (dh_out != NULL) ? (dh_out + b * d_state) : NULL;
        const float   *dh_dir_b   = (dh_direct != NULL) ? (dh_direct + b * seq_len * d_state) : NULL;

        ssm_scan_backward(A, A_per_step, dt_b, B_b, B_stride, C, x_b, hcache_b, dy_b,
                          bmask_b, dA, dA_ps_b, dB_b, dC, dx_b, dh_out_b,
                          seq_len, d_state, d_inner, C_bilinear, dC_bilinear,
                          u, v, du, dv, dh_dir_b);
    }
}

/* ============================================================================
 * WHT VARIANT
 * ============================================================================ */

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
) {
    wht_inplace(h, d_state);

    ssm_scan_forward(A, NULL, dt, B, B_stride, C, x, h, y, h_cache, moments,
                     NULL, seq_len, d_state, d_inner, NULL, NULL, NULL);

    wht_inplace(h, d_state);
}

/* ============================================================================
 * REVERSIBLE BACKWARD SCAN
 * ============================================================================ */

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
) {
    assert(A != NULL);
    assert(B_eff != NULL);
    assert(x != NULL);
    assert(h_final != NULL);
    assert(dy != NULL);
    assert(dA != NULL);
    assert(dB != NULL);
    assert(dC != NULL);
    assert(dx != NULL);
    assert(C != NULL);
    assert(d_state > 0 && d_state <= V12_MAX_D_STATE);
    assert(d_inner > 0);
    assert(seq_len > 0);

    float dh[V12_MAX_D_STATE];
    memset(dh, 0, d_state * sizeof(float));

    float inv_A[V12_MAX_D_STATE];
    for (int i = 0; i < d_state; i++) {
        float a_safe = fmaxf(A[i], 1e-6f);
        inv_A[i] = 1.0f / a_safe;
    }

    float h_cur[V12_MAX_D_STATE];
    memcpy(h_cur, h_final, d_state * sizeof(float));

    float h_prev_buf[V12_MAX_D_STATE];

    for (int t = seq_len - 1; t >= 0; t--) {
        const float *B_eff_t = B_eff + t * d_state;
        const float *x_t     = x + t * d_inner;
        const float *dy_t    = dy + t * d_inner;

        if (t == 0) {
            memset(h_prev_buf, 0, d_state * sizeof(float));
        } else if (boundary_mask != NULL && boundary_mask[t]) {
            memset(h_prev_buf, 0, d_state * sizeof(float));
        } else {
            for (int i = 0; i < d_state; i++) {
                h_prev_buf[i] = (h_cur[i] - B_eff_t[i] * x_t[i % d_inner]) * inv_A[i];
            }
        }

        /* dh += C^T @ dy_t (transposed GEMV) */
        for (int i = 0; i < d_state; i++) {
            float acc = 0.0f;
            for (int j = 0; j < d_inner; j++) {
                acc += C[j * d_state + i] * dy_t[j];
            }
            dh[i] += acc;
        }

        /* dC += dy_t ⊗ h_cur (outer product, accumulated) */
        for (int j = 0; j < d_inner; j++) {
            for (int i = 0; i < d_state; i++) {
                dC[j * d_state + i] += dy_t[j] * h_cur[i];
            }
        }

        /* dA[i] += h_prev[i] * dh[i] (dt == NULL branch) */
        for (int i = 0; i < d_state; i++) {
            dA[i] += h_prev_buf[i] * dh[i];
        }

        /* dB[t][i] = dh[i] * x_t[i % d_inner] (B_stride == d_state) */
        float *dB_t = dB + t * d_state;
        for (int i = 0; i < d_state; i++) {
            dB_t[i] = dh[i] * x_t[i % d_inner];
        }

        /* dx[t][i] = B_eff_t[i] * dh[i] */
        float *dx_t = dx + t * d_inner;
        memset(dx_t, 0, d_inner * sizeof(float));
        for (int i = 0; i < d_state; i++) {
            dx_t[i % d_inner] += B_eff_t[i] * dh[i];
        }

        /* Boundary: gradient does not flow back across a reset boundary */
        if (boundary_mask != NULL && boundary_mask[t]) {
            memset(dh, 0, (size_t)d_state * sizeof(float));
        }

        /* Propagate dh backward through A */
        for (int i = 0; i < d_state; i++) {
            dh[i] = A[i] * dh[i];
        }

        memcpy(h_cur, h_prev_buf, d_state * sizeof(float));
    }

    if (dh_out != NULL) {
        memcpy(dh_out, dh, d_state * sizeof(float));
    }
}

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
) {
    for (int b = 0; b < batch_size; b++) {
        const float   *B_eff_b   = B_eff + b * seq_len * d_state;
        const float   *x_b       = x + b * seq_len * d_inner;
        const float   *h_final_b = h_final + b * d_state;
        const float   *dy_b      = dy + b * seq_len * d_inner;
        const uint8_t *bmask_b   = (boundary_mask != NULL) ? (boundary_mask + b * seq_len) : NULL;
        float         *dB_b      = dB + b * seq_len * d_state;
        float         *dx_b      = dx + b * seq_len * d_inner;
        float         *dh_out_b  = (dh_out != NULL) ? (dh_out + b * d_state) : NULL;

        ssm_scan_backward_reversible(
            A, B_eff_b, x_b, h_final_b, dy_b,
            bmask_b, dA, dB_b, dC, dx_b, dh_out_b, C,
            seq_len, d_state, d_inner
        );
    }
}

/* ============================================================================
 * CHECKPOINTED BACKWARD SCAN
 * ============================================================================ */

void ssm_scan_backward_checkpointed(
    const float   *A,
    const float   *A_per_step,
    const float   *dt,
    const float   *B,
    int            B_stride,
    const float   *C,
    const float   *x,
    const float   *h_ckpt,
    const float   *dy,
    const uint8_t *boundary_mask,
    float         *dA,
    float         *dA_per_step,
    float         *dB,
    float         *dC,
    float         *dx,
    float         *dh_out,
    int seq_len,
    int d_state,
    int d_inner,
    const float   *C_bilinear,
    float         *dC_bilinear,
    const float   *u,
    const float   *v,
    float         *du,
    float         *dv
) {
    assert(h_ckpt != NULL);
    assert(dA != NULL && dB != NULL && dC != NULL && dx != NULL);
    assert(d_state > 0 && d_state <= V12_MAX_D_STATE);
    assert(seq_len > 0 && seq_len % V12_CKPT_STRIDE == 0);

    const int K = V12_CKPT_STRIDE;
    const int n_seg = seq_len / K;

    float dh[V12_MAX_D_STATE];
    memset(dh, 0, d_state * sizeof(float));

    /* A_eff initialization */
    float A_eff[V12_MAX_D_STATE];
    if (A_per_step == NULL && dt == NULL) {
        for (int i = 0; i < d_state; i++) A_eff[i] = A[i];
    }

    /* Stack buffer: reconstructed h values for current segment */
    float h_local[V12_CKPT_STRIDE][V12_MAX_D_STATE];

    for (int seg = n_seg - 1; seg >= 0; seg--) {
        const int t0 = seg * K;
        const int t1 = t0 + K;                    /* exclusive */

        /* --- Phase 1: re-run forward to reconstruct h[t0..t1-1] --- */
        const float *h_entry = h_ckpt + (size_t)seg * d_state;
        float h_run[V12_MAX_D_STATE];
        memcpy(h_run, h_entry, d_state * sizeof(float));

        for (int t = t0; t < t1; t++) {
            const float *B_t = B + (size_t)t * B_stride;
            const float *x_t = x + (size_t)t * d_inner;

            if (boundary_mask != NULL && boundary_mask[t]) {
                memset(h_run, 0, d_state * sizeof(float));
            }

            /* Select A_eff: A_per_step takes priority over dt */
            if (A_per_step != NULL) {
                const float *a_t = A_per_step + t * d_state;
                for (int i = 0; i < d_state; i++) A_eff[i] = a_t[i];
            } else if (dt != NULL) {
                /* dt path: recompute A_eff per timestep */
                for (int i = 0; i < d_state; i++) {
                    A_eff[i] = expf(dt[t] * logf(A[i]));
                }
            }
            /* else: A_eff already set (time-invariant) */

            float w_prev = 0.0f;
            if (u != NULL && v != NULL) {
                for (int i = 0; i < d_state; i++) {
                    w_prev += v[i] * h_run[i];
                }
            }

            if (C_bilinear != NULL) {
                for (int i = 0; i < d_state; i++) {
                    float h_prev_i = h_run[i];
                    h_run[i] = A_eff[i] * h_prev_i + B_t[i] * x_t[i % d_inner];
                    h_run[i] += C_bilinear[i] * h_prev_i * x_t[i % d_inner];
                    if (u != NULL && v != NULL) {
                        h_run[i] += w_prev * u[i];
                    }
                }
            } else {
                for (int i = 0; i < d_state; i++) {
                    h_run[i] = A_eff[i] * h_run[i] + B_t[i] * x_t[i % d_inner];
                    if (u != NULL && v != NULL) {
                        h_run[i] += w_prev * u[i];
                    }
                }
            }

            /* Store h AFTER update at position (t - t0) */
            memcpy(h_local[t - t0], h_run, d_state * sizeof(float));
        }

        /* --- Phase 2: backward through [t1-1 .. t0] using reconstructed h --- */
        for (int t = t1 - 1; t >= t0; t--) {
            const int   li    = t - t0;           /* local index in h_local */
            const float *x_t  = x  + (size_t)t * d_inner;
            const float *B_t  = B  + (size_t)t * B_stride;
            const float *dy_t = dy + (size_t)t * d_inner;
            const float *h_t  = h_local[li];

            /* h_prev: h BEFORE the state update at time t */
            const float *h_prev;
            int boundary_at_t = (boundary_mask != NULL && boundary_mask[t]);
            float h_zero[V12_MAX_D_STATE];
            if (boundary_at_t) {
                /* forward reset h to 0 before this token's update */
                memset(h_zero, 0, d_state * sizeof(float));
                h_prev = h_zero;
            } else if (li > 0) {
                h_prev = h_local[li - 1];
            } else {
                /* li == 0: entry state for this segment = checkpoint */
                h_prev = h_entry;
            }

            /* Select A_eff: A_per_step takes priority over dt */
            if (A_per_step != NULL) {
                const float *a_t = A_per_step + t * d_state;
                for (int i = 0; i < d_state; i++) A_eff[i] = a_t[i];
            } else if (dt != NULL) {
                for (int i = 0; i < d_state; i++) {
                    A_eff[i] = expf(dt[t] * logf(A[i]));
                }
            }
            /* else: A_eff already set (time-invariant) */

            /* dh += C^T @ dy_t */
            for (int i = 0; i < d_state; i++) {
                float acc = 0.0f;
                for (int j = 0; j < d_inner; j++) {
                    acc += C[j * d_state + i] * dy_t[j];
                }
                dh[i] += acc;
            }

            /* dC += dy_t ⊗ h_t */
            for (int j = 0; j < d_inner; j++) {
                for (int i = 0; i < d_state; i++) {
                    dC[j * d_state + i] += dy_t[j] * h_t[i];
                }
            }

            /* dA gradient: when A_per_step != NULL, write to dA_per_step; otherwise accumulate to dA */
            if (A_per_step != NULL) {
                if (dA_per_step != NULL && t > 0 && !boundary_at_t) {
                    float *da_t = dA_per_step + t * d_state;
                    for (int i = 0; i < d_state; i++) {
                        da_t[i] = dh[i] * h_prev[i];
                    }
                }
                /* dA (static) does not accumulate when A_per_step is active */
            } else if (dt != NULL && t > 0 && !boundary_at_t) {
                for (int i = 0; i < d_state; i++) {
                    dA[i] += dt[t] * A_eff[i] * h_prev[i] * dh[i];
                }
            } else if (t > 0 && !boundary_at_t) {
                for (int i = 0; i < d_state; i++) {
                    dA[i] += h_prev[i] * dh[i];
                }
            }

            /* dB[t][i] = dh[i] * x_t[i % d_inner] */
            if (B_stride == 0) {
                for (int i = 0; i < d_state; i++) dB[i] += dh[i] * x_t[i % d_inner];
            } else {
                float *dB_t = dB + (size_t)t * d_state;
                for (int i = 0; i < d_state; i++) dB_t[i] = dh[i] * x_t[i % d_inner];
            }

            /* dx[t][i] = B_t[i] * dh[i] */
            float *dx_t = dx + (size_t)t * d_inner;
            memset(dx_t, 0, d_inner * sizeof(float));
            for (int i = 0; i < d_state; i++) dx_t[i % d_inner] += B_t[i] * dh[i];

            /* Bilinear recurrence gradients: h[t] += C_bilinear[i] * h_prev[i] * x_t[i % d_inner] */
            if (C_bilinear != NULL && dC_bilinear != NULL && t > 0 && !boundary_at_t) {
                for (int i = 0; i < d_state; i++) {
                    float h_prev_i = h_prev[i];
                    dC_bilinear[i] += dh[i] * h_prev_i * x_t[i % d_inner];
                    dx_t[i % d_inner] += C_bilinear[i] * h_prev_i * dh[i];
                }
            }

            float dw_prev = 0.0f;
            float w_prev = 0.0f;
            if (u != NULL && v != NULL) {
                for (int i = 0; i < d_state; i++) {
                    dw_prev += dh[i] * u[i];
                }
                if (t > 0 && !boundary_at_t) {
                    for (int i = 0; i < d_state; i++) {
                        w_prev += v[i] * h_prev[i];
                    }
                    if (du != NULL) {
                        for (int i = 0; i < d_state; i++) {
                            du[i] += dh[i] * w_prev;
                        }
                    }
                    if (dv != NULL) {
                        for (int i = 0; i < d_state; i++) {
                            dv[i] += dw_prev * h_prev[i];
                        }
                    }
                }
            }

            /* Boundary: gradient does not flow back across a reset */
            if (boundary_at_t) {
                memset(dh, 0, d_state * sizeof(float));
            }

            /* Propagate dh backward through A (+ bilinear term if active) */
            for (int i = 0; i < d_state; i++) {
                float dh_factor = A_eff[i];
                if (C_bilinear != NULL) dh_factor += C_bilinear[i] * x_t[i % d_inner];
                dh[i] = dh_factor * dh[i];
                if (u != NULL && v != NULL && t > 0 && !boundary_at_t) {
                    dh[i] += dw_prev * v[i];
                }
            }
        }
        /* dh now carries gradient to propagate into segment seg-1 */
    }

    if (dh_out != NULL) {
        memcpy(dh_out, dh, d_state * sizeof(float));
    }
}

void ssm_scan_backward_checkpointed_batch(
    const float   *A,
    const float   *A_per_step,
    const float   *dt,
    const float   *B,
    int            B_stride,
    const float   *C,
    const float   *x,
    const float   *h_ckpt,
    const float   *dy,
    const uint8_t *boundary_mask,
    float         *dA,
    float         *dA_per_step,
    float         *dB,
    float         *dC,
    float         *dx,
    float         *dh_out,
    int batch_size,
    int seq_len,
    int d_state,
    int d_inner,
    const float   *C_bilinear,
    float         *dC_bilinear,
    const float   *u,
    const float   *v,
    float         *du,
    float         *dv
) {
    const int ckpt_count = seq_len / V12_CKPT_STRIDE + 1;
    for (int b = 0; b < batch_size; b++) {
        const float   *dt_b       = (dt != NULL) ? (dt + b * seq_len) : NULL;
        const float   *B_b        = (B_stride > 0) ? (B + b * seq_len * d_state) : B;
        const float   *x_b        = x + b * seq_len * d_inner;
        const float   *hckpt_b    = h_ckpt + b * ckpt_count * d_state;
        const float   *dy_b       = dy + b * seq_len * d_inner;
        const uint8_t *bmask_b    = (boundary_mask != NULL) ? (boundary_mask + b * seq_len) : NULL;
        float         *dB_b       = (B_stride > 0) ? (dB + b * seq_len * d_state) : dB;
        float         *dx_b       = dx + b * seq_len * d_inner;
        float         *dA_ps_b    = (dA_per_step != NULL) ? (dA_per_step + b * seq_len * d_state) : NULL;
        float         *dh_out_b   = (dh_out != NULL) ? (dh_out + b * d_state) : NULL;

        ssm_scan_backward_checkpointed(
            A, A_per_step, dt_b, B_b, B_stride, C, x_b, hckpt_b, dy_b,
            bmask_b, dA, dA_ps_b, dB_b, dC, dx_b, dh_out_b,
            seq_len, d_state, d_inner, C_bilinear, dC_bilinear,
            u, v, du, dv
        );
    }
}

/* ============================================================================
 * MTP HELPER
 * ============================================================================ */

void ssm_state_power(
    const float *A,
    const float *h,
    float       *h_out,
    int          k,
    int          d_state
) {
    assert(A != NULL);
    assert(h != NULL);
    assert(h_out != NULL);
    assert(k >= 0);
    assert(d_state > 0);

    /* h_out[i] = h[i] * A[i]^k = h[i] * exp(k * log(A[i]))
     * When k=0: exp(0) = 1, so h_out = h (identity case) */
    for (int i = 0; i < d_state; i++) {
        h_out[i] = h[i] * expf((float)k * logf(A[i]));
    }
}

/* ============================================================================
 * R356 F1 — SLICE S0: FORWARD STABILITY PROBE (non-invasive instrumentation)
 *
 * Computes the cumulative quantity
 *     stability_probe_cum[i] = Sum_{t=0..seq_len-1} ln(max(|A_tot_i[t]|, 1e-30))
 * where
 *     A_tot_i[t] = A_eff[t][i] + (C_bilinear ? C_bilinear[i] * x[t][i] : 0)
 *
 * This metric is the validator-amendment stability bound for the F1 reverse
 * recurrence: err(h[0]) <= err(h[T]) * exp(max_i stability_probe_cum[i]).
 * The runtime selector (S3) uses it to decide whether the F1 backward is safe.
 *
 * This function runs a pre-pass that consumes only A, A_per_step, dt, x and
 * C_bilinear — none of which depend on the evolving hidden state — then
 * delegates to ssm_scan_forward for the actual scan. Semantics of the scan
 * are unchanged: the probe is purely additive and produces no behavioural
 * side-effects when stability_probe_cum is NULL (in that case this function
 * is a straight forward-to forward delegation).
 *
 * When stability_probe_cum is non-NULL the caller opts in to the [F1 PROBE]
 * summary line, emitted once per call at the end of the scan.
 * ============================================================================ */

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
) {
    if (stability_probe_cum != NULL) {
        assert(A != NULL);
        assert(x != NULL);
        assert(d_state > 0 && d_state <= V12_MAX_D_STATE);
        assert(d_inner > 0);
        assert(seq_len > 0);

        for (int i = 0; i < d_state; i++) {
            stability_probe_cum[i] = 0.0f;
        }

        float A_eff[V12_MAX_D_STATE];
        if (A_per_step == NULL && dt == NULL) {
            for (int i = 0; i < d_state; i++) A_eff[i] = A[i];
        }

        for (int t = 0; t < seq_len; t++) {
            const float *x_t = x + t * d_inner;

            if (A_per_step != NULL) {
                const float *a_t = A_per_step + t * d_state;
                for (int i = 0; i < d_state; i++) A_eff[i] = a_t[i];
            } else if (dt != NULL) {
                for (int i = 0; i < d_state; i++) {
                    A_eff[i] = expf(dt[t] * logf(A[i]));
                }
            }

            if (C_bilinear != NULL) {
                for (int i = 0; i < d_state; i++) {
                    float a_tot = A_eff[i] + C_bilinear[i] * x_t[i % d_inner];
                    float mag = fabsf(a_tot);
                    if (mag < 1e-30f) mag = 1e-30f;
                    stability_probe_cum[i] += logf(mag);
                }
            } else {
                for (int i = 0; i < d_state; i++) {
                    float mag = fabsf(A_eff[i]);
                    if (mag < 1e-30f) mag = 1e-30f;
                    stability_probe_cum[i] += logf(mag);
                }
            }
        }

        float cmax = stability_probe_cum[0];
        float cmin = stability_probe_cum[0];
        double csum = 0.0;
        for (int i = 0; i < d_state; i++) {
            float v = stability_probe_cum[i];
            if (v > cmax) cmax = v;
            if (v < cmin) cmin = v;
            csum += (double)v;
        }
        float cmean = (float)(csum / (double)d_state);

        /* Verification anchor per handoff §7.4: one summary line per call.
         * Fields: terminating step, representative channel 0, max/min/mean
         * across channels, and the per-channel max across the sequence
         * (which equals max over channels of the cumulative probe at t=T). */
        printf("[F1 PROBE] step=%d ch0_cum=%.6f max_cum=%.6f min_cum=%.6f "
               "mean_cum=%.6f per_channel_max=%.6f threshold_ln10=%.6f\n",
               seq_len,
               (double)stability_probe_cum[0],
               (double)cmax,
               (double)cmin,
               (double)cmean,
               (double)cmax,
               (double)logf(10.0f));
    }

    ssm_scan_forward(A, A_per_step, dt, B, B_stride, C, x, h, y, h_cache,
                     moments, boundary_mask, seq_len, d_state, d_inner,
                     C_bilinear, NULL, NULL);
}

/* ============================================================================
 * R356 F1 — SLICE S1: CPU REFERENCE BACKWARD (diagonal fast path)
 *
 * ssm_scan_backward_f1
 *
 * Reverse BPTT over the SSM recurrence that reconstructs h[t-1] on the fly
 * from h[t] via the per-channel inversion
 *
 *     A_tot_i[t] = A_eff[t][i] + (C_bilinear ? C_bilinear[i]*x[t][i] : 0)
 *     h[t-1][i]  = (h[t][i] - B_eff[t][i] * x[t][i]) / A_tot_i[t]
 *
 * This is the validator-accepted "Finding 1" form. Unlike
 * ssm_scan_backward_reversible (the rejected pure-diagonal form left in
 * the tree for audit), this function divides by (A_eff + C_bl*x), not by
 * A alone, and therefore correctly inverts the bilinear recurrence.
 *
 * Under the current v12 block, C_bilinear is a per-channel diagonal
 * perturbation (u_i v_i, not a rank-1 outer product uv^T), so the inversion
 * reduces to a scalar divide per channel. A future rank-1 form would need
 * the Sherman-Morrison correction
 *     A_tot^-1 = D^-1 - (D^-1 u v^T D^-1) / (1 + v^T D^-1 u)
 * (see TODO below). That path is intentionally NOT implemented in S1 — the
 * current v12 bilinear is diagonal and the general form is future-proofing.
 *
 * Parameters mirror ssm_scan_backward exactly, except:
 *   - h_cache is REPLACED by h_final (the state AFTER the last step).
 *   - stability_probe_cum is an input: if non-NULL, the function checks
 *     max_i(probe) < ln(10) before running and refuses with a warning
 *     on exceed (the caller, S3, should dispatch to _checkpointed instead).
 *
 * NOT touched: ssm_scan_backward_reversible. Do not merge the two.
 * ============================================================================ */

void ssm_scan_backward_f1(
    const float   *A,
    const float   *A_per_step,
    const float   *dt,
    const float   *B,
    int            B_stride,
    const float   *C,
    const float   *x,
    const float   *h_final,
    const float   *dy,
    const uint8_t *boundary_mask,
    float         *dA,
    float         *dA_per_step,
    float         *dB,
    float         *dC,
    float         *dx,
    float         *dh_out,
    const float   *C_bilinear,
    float         *dC_bilinear,
    const float   *stability_probe_cum,
    int seq_len,
    int d_state,
    int d_inner,
    const float   *u,
    const float   *v,
    float         *du,
    float         *dv
) {
    assert(A != NULL);
    assert(x != NULL);
    assert(h_final != NULL);
    assert(dy != NULL);
    assert(dA != NULL);
    assert(dB != NULL);
    assert(dC != NULL);
    assert(dx != NULL);
    assert(d_state > 0 && d_state <= V12_MAX_D_STATE);
    assert(d_inner > 0);
    assert(seq_len > 0);

    /* Validator amendment — cumulative stability gate (handoff §3.2).
     * If the probe is supplied and exceeds ln(10) on any channel, the
     * reverse walk is considered unsafe: log loudly and refuse to run.
     * The runtime selector (S3) is expected to catch this first; this
     * is the last line of defence. */
    if (stability_probe_cum != NULL) {
        const float stab_threshold = logf(10.0f);
        float probe_max = stability_probe_cum[0];
        int   probe_argmax = 0;
        for (int i = 1; i < d_state; i++) {
            if (stability_probe_cum[i] > probe_max) {
                probe_max = stability_probe_cum[i];
                probe_argmax = i;
            }
        }
        if (probe_max >= stab_threshold) {
            fprintf(stderr,
                "[F1 GATE] REFUSED: stability probe exceeds threshold "
                "(max_cum=%.6f at channel %d, threshold ln(10)=%.6f). "
                "Caller must dispatch to ssm_scan_backward_checkpointed.\n",
                (double)probe_max, probe_argmax, (double)stab_threshold);
            return;
        }
    }

    float dh[V12_MAX_D_STATE];
    memset(dh, 0, d_state * sizeof(float));

    float A_eff[V12_MAX_D_STATE];
    if (A_per_step == NULL && dt == NULL) {
        /* Time-invariant: A is already direct decay, copy once */
        for (int i = 0; i < d_state; i++) A_eff[i] = A[i];
    }

    /* h_cur tracks the state AT time t (the "post-update" state of token t).
     * We start at h_final = h[seq_len-1] and walk back to h[0]. */
    float h_cur[V12_MAX_D_STATE];
    float h_prev_buf[V12_MAX_D_STATE];
    memcpy(h_cur, h_final, d_state * sizeof(float));

    /* Near-zero denominator exception counter (handoff §5 risk #3). */
    const float denom_eps = 1e-6f;
    int denom_clamp_count = 0;

    for (int t = seq_len - 1; t >= 0; t--) {
        const float *x_t   = x  + (size_t)t * d_inner;
        const float *B_t   = B  + (size_t)t * B_stride;
        const float *dy_t  = dy + (size_t)t * d_inner;

        /* Select A_eff — three branches, must match the forward's choice. */
        if (A_per_step != NULL) {
            const float *a_t = A_per_step + (size_t)t * d_state;
            for (int i = 0; i < d_state; i++) A_eff[i] = a_t[i];
        } else if (dt != NULL) {
            for (int i = 0; i < d_state; i++) {
                A_eff[i] = expf(dt[t] * logf(A[i]));
            }
        }
        /* else: A_eff already set (time-invariant) */

        /* Reconstruct h_prev via the scalar-divide inversion.
         *
         *   h[t-1][i] = (h[t][i] - B_eff[t][i] * x[t][i]) / A_tot_i[t]
         *
         * Boundary at t: the forward reset h to 0 BEFORE processing token t,
         * so the true h_prev at this step is zero (no inversion needed). At
         * t == 0 there is no prior step; convention is h_prev = 0. */
        int boundary_at_t = (boundary_mask != NULL && boundary_mask[t]);
        if (t == 0 || boundary_at_t) {
            memset(h_prev_buf, 0, d_state * sizeof(float));
        } else if (u != NULL && v != NULL) {
            float temp_du[V12_MAX_D_STATE];
            float temp_d_u[V12_MAX_D_STATE];
            float dot_v_d_u = 0.0f;
            float dot_v_du = 0.0f;
            for (int i = 0; i < d_state; i++) {
                float a_tot = A_eff[i];
                if (C_bilinear != NULL) {
                    a_tot += C_bilinear[i] * x_t[i % d_inner];
                }
                float a_mag = fabsf(a_tot);
                if (a_mag < denom_eps) {
                    a_tot = (a_tot < 0.0f) ? -denom_eps : denom_eps;
                    denom_clamp_count++;
                }
                temp_du[i] = (h_cur[i] - B_t[i] * x_t[i % d_inner]) / a_tot;
                temp_d_u[i] = u[i] / a_tot;
                dot_v_d_u += v[i] * temp_d_u[i];
                dot_v_du += v[i] * temp_du[i];
            }
            float denom = 1.0f + dot_v_d_u;
            if (fabsf(denom) < 1e-6f) {
                denom = (denom < 0.0f) ? -1e-6f : 1e-6f;
            }
            float k_factor = dot_v_du / denom;
            for (int i = 0; i < d_state; i++) {
                h_prev_buf[i] = temp_du[i] - k_factor * temp_d_u[i];
            }
        } else if (C_bilinear != NULL) {
            for (int i = 0; i < d_state; i++) {
                float a_tot = A_eff[i] + C_bilinear[i] * x_t[i % d_inner];
                float a_mag = fabsf(a_tot);
                if (a_mag < denom_eps) {
                    /* Clamp magnitude while preserving sign; count exception. */
                    a_tot = (a_tot < 0.0f) ? -denom_eps : denom_eps;
                    denom_clamp_count++;
                }
                h_prev_buf[i] = (h_cur[i] - B_t[i] * x_t[i % d_inner]) / a_tot;
            }
            /* TODO(F1-SM): when C_bilinear becomes a true rank-1 outer
             * product u v^T (d_state x d_inner dense) rather than the
             * current per-channel diagonal u_i v_i form, replace the
             * scalar divide above with the Sherman-Morrison inversion:
             *     D     = diag(A_eff)
             *     du    = (h_cur - b) / D     ; with b = B_t .* x_t
             *     d_u   = u / D
             *     denom = 1 + v . d_u
             *     k     = (v . du) / denom
             *     h_prev_buf = du - k * d_u
             * The current v12 block does not exercise this path. */
        } else {
            for (int i = 0; i < d_state; i++) {
                float a_tot = A_eff[i];
                float a_mag = fabsf(a_tot);
                if (a_mag < denom_eps) {
                    a_tot = (a_tot < 0.0f) ? -denom_eps : denom_eps;
                    denom_clamp_count++;
                }
                h_prev_buf[i] = (h_cur[i] - B_t[i] * x_t[i % d_inner]) / a_tot;
            }
        }

        /* From here on the gradient update mirrors ssm_scan_backward line by
         * line, using h_cur as h_t and h_prev_buf as h_prev. */

        /* dh += C^T @ dy_t */
        for (int i = 0; i < d_state; i++) {
            float acc = 0.0f;
            for (int j = 0; j < d_inner; j++) {
                acc += C[j * d_state + i] * dy_t[j];
            }
            dh[i] += acc;
        }

        /* dC += dy_t ⊗ h_cur (h_cur == h_t) */
        for (int j = 0; j < d_inner; j++) {
            for (int i = 0; i < d_state; i++) {
                dC[j * d_state + i] += dy_t[j] * h_cur[i];
            }
        }

        /* dA gradient — three branches matching ssm_scan_backward. h_prev_buf
         * is defined for all t here (zero at t==0 or at a boundary). */
        if (A_per_step != NULL) {
            if (dA_per_step != NULL && t > 0 && !boundary_at_t) {
                float *da_t = dA_per_step + (size_t)t * d_state;
                for (int i = 0; i < d_state; i++) {
                    da_t[i] = dh[i] * h_prev_buf[i];
                }
            }
            /* dA (static) does not accumulate when A_per_step is active */
        } else if (dt != NULL && t > 0 && !boundary_at_t) {
            for (int i = 0; i < d_state; i++) {
                dA[i] += dt[t] * A_eff[i] * h_prev_buf[i] * dh[i];
            }
        } else if (t > 0 && !boundary_at_t) {
            for (int i = 0; i < d_state; i++) {
                dA[i] += h_prev_buf[i] * dh[i];
            }
        }

        /* dB[i] += dh[i] * x_t[i % d_inner] */
        if (B_stride == 0) {
            for (int i = 0; i < d_state; i++) {
                dB[i] += dh[i] * x_t[i % d_inner];
            }
        } else {
            float *dB_t = dB + (size_t)t * d_state;
            for (int i = 0; i < d_state; i++) {
                dB_t[i] = dh[i] * x_t[i % d_inner];
            }
        }

        /* dx[t][i] = B_t[i] * dh[i] */
        float *dx_t = dx + (size_t)t * d_inner;
        memset(dx_t, 0, d_inner * sizeof(float));
        for (int i = 0; i < d_state; i++) {
            dx_t[i % d_inner] += B_t[i] * dh[i];
        }

        /* Bilinear recurrence gradients */
        if (C_bilinear != NULL && t > 0 && !boundary_at_t) {
            for (int i = 0; i < d_state; i++) {
                float h_prev_i = h_prev_buf[i];
                if (dC_bilinear != NULL) {
                    dC_bilinear[i] += dh[i] * h_prev_i * x_t[i % d_inner];
                }
                dx_t[i % d_inner] += C_bilinear[i] * h_prev_i * dh[i];
            }
        }

        /* u and v gradient updates */
        float dw_prev = 0.0f;
        float w_prev = 0.0f;
        if (u != NULL && v != NULL) {
            for (int i = 0; i < d_state; i++) {
                dw_prev += dh[i] * u[i];
            }
            if (t > 0 && !boundary_at_t) {
                for (int i = 0; i < d_state; i++) {
                    w_prev += v[i] * h_prev_buf[i];
                }
                if (du != NULL) {
                    for (int i = 0; i < d_state; i++) {
                        du[i] += dh[i] * w_prev;
                    }
                }
                if (dv != NULL) {
                    for (int i = 0; i < d_state; i++) {
                        dv[i] += dw_prev * h_prev_buf[i];
                    }
                }
            }
        }

        /* Boundary: gradient does not flow back across a reset boundary */
        if (boundary_at_t) {
            memset(dh, 0, (size_t)d_state * sizeof(float));
        }

        /* Propagate dh backward through the A gate (+ bilinear term if active) */
        for (int i = 0; i < d_state; i++) {
            float dh_factor = A_eff[i];
            if (C_bilinear != NULL) dh_factor += C_bilinear[i] * x_t[i % d_inner];
            dh[i] = dh_factor * dh[i];
        }
        if (u != NULL && v != NULL && t > 0 && !boundary_at_t) {
            for (int i = 0; i < d_state; i++) {
                dh[i] += dw_prev * v[i];
            }
        }

        /* Slide the reverse walk: next iteration's h_cur is this step's h_prev. */
        memcpy(h_cur, h_prev_buf, d_state * sizeof(float));
    }

    if (dh_out != NULL) {
        memcpy(dh_out, dh, d_state * sizeof(float));
    }

    if (denom_clamp_count > 0) {
        fprintf(stderr,
            "[F1 WARN] denominator |A_tot| clamped to %.1e on %d occasions "
            "(seq_len=%d, d_state=%d). Inversion may be degraded near those "
            "steps; consider dispatching to ssm_scan_backward_checkpointed.\n",
            (double)denom_eps, denom_clamp_count, seq_len, d_state);
    }
}

/* ============================================================================
 * R356 F1 — SLICE S2: CPU BATCH WRAPPER
 *
 * ssm_scan_backward_f1_batch
 *
 * Pattern-identical to ssm_scan_backward_batch: iterates the batch dimension
 * and calls ssm_scan_backward_f1 with per-item strided pointers. dA, dC, and
 * dC_bilinear are SHARED accumulators across the batch and are NOT split.
 * h_final is a per-item d_state slice (no seq_len factor); stability_probe_cum
 * matches ssm_scan_forward_batch_with_stability_probe's [batch × d_state]
 * layout.
 * ============================================================================ */

void ssm_scan_backward_f1_batch(
    const float   *A,
    const float   *A_per_step,
    const float   *dt,
    const float   *B,
    int            B_stride,
    const float   *C,
    const float   *x,
    const float   *h_final,
    const float   *dy,
    const uint8_t *boundary_mask,
    float         *dA,
    float         *dA_per_step,
    float         *dB,
    float         *dC,
    float         *dx,
    float         *dh_out,
    const float   *C_bilinear,
    float         *dC_bilinear,
    const float   *stability_probe_cum,
    int batch_size,
    int seq_len,
    int d_state,
    int d_inner,
    const float   *u,
    const float   *v,
    float         *du,
    float         *dv
) {
    for (int b = 0; b < batch_size; b++) {
        const float   *dt_b       = (dt != NULL) ? (dt + b * seq_len) : NULL;
        const float   *B_b        = (B_stride > 0) ? (B + b * seq_len * d_state) : B;
        const float   *x_b        = x + b * seq_len * d_inner;
        const float   *h_final_b  = h_final + b * d_state;
        const float   *dy_b       = dy + b * seq_len * d_inner;
        const uint8_t *bmask_b    = (boundary_mask != NULL) ? (boundary_mask + b * seq_len) : NULL;
        float         *dB_b       = (B_stride > 0) ? (dB + b * seq_len * d_state) : dB;
        float         *dx_b       = dx + b * seq_len * d_inner;
        float         *dA_ps_b    = (dA_per_step != NULL) ? (dA_per_step + b * seq_len * d_state) : NULL;
        float         *dh_out_b   = (dh_out != NULL) ? (dh_out + b * d_state) : NULL;
        const float   *probe_b    = (stability_probe_cum != NULL)
                                        ? (stability_probe_cum + b * d_state)
                                        : NULL;

        ssm_scan_backward_f1(
            A, A_per_step, dt_b, B_b, B_stride, C, x_b, h_final_b, dy_b,
            bmask_b, dA, dA_ps_b, dB_b, dC, dx_b, dh_out_b,
            C_bilinear, dC_bilinear, probe_b,
            seq_len, d_state, d_inner,
            u, v, du, dv);
    }
}

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
) {
    for (int b = 0; b < batch_size; b++) {
        const float   *dt_b     = (dt != NULL) ? (dt + b * seq_len) : NULL;
        const float   *B_b      = (B_stride > 0) ? (B + b * seq_len * d_state) : B;
        const float   *x_b      = x + b * seq_len * d_inner;
        float         *h_b      = h + b * d_state;
        float         *y_b      = y + b * seq_len * d_inner;
        float         *hcache_b = (h_cache != NULL) ? (h_cache + b * seq_len * d_state) : NULL;
        GpKlMoments   *mom_b    = moments;
        const uint8_t *bmask_b  = (boundary_mask != NULL) ? (boundary_mask + b * seq_len) : NULL;
        float         *probe_b  = (stability_probe_cum != NULL)
                                      ? (stability_probe_cum + b * d_state)
                                      : NULL;

        ssm_scan_forward_with_stability_probe(
            A, A_per_step, dt_b, B_b, B_stride, C, x_b, h_b, y_b, hcache_b,
            mom_b, bmask_b, seq_len, d_state, d_inner, C_bilinear, probe_b);
    }
}
