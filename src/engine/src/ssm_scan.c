#include "ssm_scan.h"
#include "ternary_gemv.h"
#include "float_gemv.h"
#include "activations.h"
#include "simd_detect.h"
#include "ternary_types.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef ENGINE_AVX2
#include <immintrin.h>
#endif

ssm_state_t* ssm_state_alloc(int32_t n_heads, int32_t d_state, int32_t head_dim, int32_t d_conv) {
    ssm_state_t* s = (ssm_state_t*)calloc(1, sizeof(ssm_state_t));
    s->n_heads = n_heads;
    s->d_state = d_state;
    s->head_dim = head_dim;
    s->d_inner = n_heads * head_dim;
    s->d_conv = d_conv;
    s->h = (float*)calloc((size_t)n_heads * d_state * head_dim, sizeof(float));
    s->conv_buf = (float*)calloc((size_t)s->d_inner * d_conv, sizeof(float));
    return s;
}

void ssm_state_free(ssm_state_t* state) {
    if (state) {
        free(state->h);
        free(state->conv_buf);
        free(state);
    }
}

void ssm_state_reset(ssm_state_t* state) {
    if (state) {
        memset(state->h, 0, (size_t)state->n_heads * state->d_state * state->head_dim * sizeof(float));
        memset(state->conv_buf, 0, (size_t)state->d_inner * state->d_conv * sizeof(float));
    }
}

ssm_layer_t* ssm_layer_alloc(int32_t d_model, int32_t d_inner, int32_t d_state,
                              int32_t d_conv, int32_t dt_rank, int32_t n_heads, int use_tscs) {
    ssm_layer_t* l = (ssm_layer_t*)calloc(1, sizeof(ssm_layer_t));
    l->d_model = d_model;
    l->d_inner = d_inner;
    l->d_state = d_state;
    l->d_conv = d_conv;
    l->dt_rank = dt_rank;
    l->n_heads = n_heads;
    l->head_dim = d_inner / n_heads;
    l->use_tscs = use_tscs;

    l->in_proj = ternary_matrix_alloc(2 * d_inner, d_model);
    l->x_proj = ternary_matrix_alloc(dt_rank + 2 * d_state, d_inner);
    l->out_proj = ternary_matrix_alloc(d_model, d_inner);

    l->dt_weight = (float*)calloc((size_t)d_inner * dt_rank, sizeof(float));
    l->dt_bias = (float*)calloc((size_t)d_inner, sizeof(float));
    l->conv_weight = (float*)calloc((size_t)d_inner * d_conv, sizeof(float));
    l->conv_bias = (float*)calloc((size_t)d_inner, sizeof(float));
    l->A = (float*)calloc((size_t)n_heads, sizeof(float));
    l->D = (float*)calloc((size_t)d_inner, sizeof(float));

    if (use_tscs) {
        l->A_ternary = (int8_t*)calloc((size_t)n_heads, sizeof(int8_t));
    }

    return l;
}

void ssm_layer_free(ssm_layer_t* layer) {
    if (layer) {
        ternary_matrix_free(layer->in_proj);
        ternary_matrix_free(layer->x_proj);
        ternary_matrix_free(layer->out_proj);
        free(layer->dt_weight);
        free(layer->dt_bias);
        free(layer->conv_weight);
        free(layer->conv_bias);
        free(layer->A);
        free(layer->D);
        free(layer->A_ternary);
        free(layer);
    }
}

size_t ssm_scratch_size(const ssm_layer_t* layer) {
    return (size_t)(4 * layer->d_inner + layer->dt_rank + 2 * layer->d_state + layer->n_heads) * sizeof(float);
}

void ssm_conv1d_step_scalar(float* y, float* conv_buf, const float* x,
                            const float* weight, const float* bias,
                            int32_t d_inner, int32_t d_conv) {
    for (int32_t i = 0; i < d_inner; i++) {
        for (int32_t k = 0; k < d_conv - 1; k++) {
            conv_buf[i * d_conv + k] = conv_buf[i * d_conv + k + 1];
        }
        conv_buf[i * d_conv + (d_conv - 1)] = x[i];

        float acc = 0.0f;
        for (int32_t k = 0; k < d_conv; k++) {
            acc += conv_buf[i * d_conv + k] * weight[i * d_conv + k];
        }
        y[i] = acc + (bias ? bias[i] : 0.0f);
    }
}

#ifdef ENGINE_AVX2
void ssm_conv1d_step_avx2(float* y, float* conv_buf, const float* x,
                          const float* weight, const float* bias,
                          int32_t d_inner, int32_t d_conv) {
    for (int32_t i = 0; i < d_inner; i++) {
        for (int32_t k = 0; k < d_conv - 1; k++) {
            conv_buf[i * d_conv + k] = conv_buf[i * d_conv + k + 1];
        }
        conv_buf[i * d_conv + (d_conv - 1)] = x[i];

        __m256 acc = _mm256_setzero_ps();
        int32_t k = 0;

        for (; k + 7 < d_conv; k += 8) {
            __m256 c_vec = _mm256_loadu_ps(&conv_buf[i * d_conv + k]);
            __m256 w_vec = _mm256_loadu_ps(&weight[i * d_conv + k]);
            acc = _mm256_fmadd_ps(c_vec, w_vec, acc);
        }

        float result[8];
        _mm256_storeu_ps(result, acc);
        float sum = result[0] + result[1] + result[2] + result[3]
                  + result[4] + result[5] + result[6] + result[7];

        for (; k < d_conv; k++) {
            sum += conv_buf[i * d_conv + k] * weight[i * d_conv + k];
        }

        y[i] = sum + (bias ? bias[i] : 0.0f);
    }
}
#endif

typedef void (*ssm_conv1d_fn)(float*, float*, const float*, const float*, const float*, int32_t, int32_t);
typedef void (*ssm_scan_fn)(float*, float*, const float*, const float*, const float*, const float*, const float*, float*, int32_t, int32_t, int32_t);
typedef void (*ssm_scan_ternary_fn)(float*, float*, const float*, const float*, const int8_t*, const float*, const float*, float*, int32_t, int32_t, int32_t);

static ssm_conv1d_fn g_conv1d_kernel = NULL;
static ssm_scan_fn g_scan_kernel = NULL;
static ssm_scan_ternary_fn g_scan_ternary_kernel = NULL;

void ssm_conv1d_step(float* y, float* conv_buf, const float* x,
                     const float* weight, const float* bias,
                     int32_t d_inner, int32_t d_conv) {
    if (!g_conv1d_kernel) ssm_scan_init();
    g_conv1d_kernel(y, conv_buf, x, weight, bias, d_inner, d_conv);
}

void ssm_scan_step_scalar(float* y, float* h, const float* x, const float* dt,
                          const float* A, const float* B, const float* C, float* dt_heads,
                          int32_t n_heads, int32_t d_state, int32_t head_dim) {
    for (int32_t p = 0; p < n_heads; p++) {
        float sum = 0.0f;
        for (int32_t d = 0; d < head_dim; d++) {
            sum += dt[p * head_dim + d];
        }
        dt_heads[p] = sum / (float)head_dim;
    }

    for (int32_t p = 0; p < n_heads; p++) {
        float dA = expf(A[p] * dt_heads[p]);
        float dt_p = dt_heads[p];
        for (int32_t n = 0; n < d_state; n++) {
            float b_scaled = dt_p * B[n];
            int32_t h_off = (p * d_state + n) * head_dim;
            for (int32_t d = 0; d < head_dim; d++) {
                h[h_off + d] = dA * h[h_off + d] + b_scaled * x[p * head_dim + d];
            }
        }
    }

    for (int32_t p = 0; p < n_heads; p++) {
        for (int32_t d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (int32_t n = 0; n < d_state; n++) {
                acc += C[n] * h[(p * d_state + n) * head_dim + d];
            }
            y[p * head_dim + d] = acc;
        }
    }
}

#ifdef ENGINE_AVX2
void ssm_scan_step_avx2(float* y, float* h, const float* x, const float* dt,
                        const float* A, const float* B, const float* C, float* dt_heads,
                        int32_t n_heads, int32_t d_state, int32_t head_dim) {
    for (int32_t p = 0; p < n_heads; p++) {
        float sum = 0.0f;
        for (int32_t d = 0; d < head_dim; d++) {
            sum += dt[p * head_dim + d];
        }
        dt_heads[p] = sum / (float)head_dim;
    }

    for (int32_t p = 0; p < n_heads; p++) {
        float dA = expf(A[p] * dt_heads[p]);
        float dt_p = dt_heads[p];

        __m256 dA_vec = _mm256_set1_ps(dA);

        for (int32_t n = 0; n < d_state; n++) {
            float b_scaled = dt_p * B[n];
            __m256 bs_vec = _mm256_set1_ps(b_scaled);
            int32_t h_off = (p * d_state + n) * head_dim;

            int32_t d = 0;
            for (; d + 7 < head_dim; d += 8) {
                __m256 h_vec = _mm256_loadu_ps(&h[h_off + d]);
                __m256 x_vec = _mm256_loadu_ps(&x[p * head_dim + d]);

                __m256 result = _mm256_fmadd_ps(dA_vec, h_vec, _mm256_mul_ps(bs_vec, x_vec));
                _mm256_storeu_ps(&h[h_off + d], result);
            }

            for (; d < head_dim; d++) {
                h[h_off + d] = dA * h[h_off + d] + b_scaled * x[p * head_dim + d];
            }
        }
    }

    for (int32_t p = 0; p < n_heads; p++) {
        for (int32_t d = 0; d < head_dim; d++) {
            __m256 acc = _mm256_setzero_ps();

            int32_t n = 0;
            for (; n + 7 < d_state; n += 8) {
                __m256 c_vec = _mm256_loadu_ps(&C[n]);
                __m256 h_vec = _mm256_loadu_ps(&h[(p * d_state + n) * head_dim + d]);
                acc = _mm256_fmadd_ps(c_vec, h_vec, acc);
            }

            float result[8];
            _mm256_storeu_ps(result, acc);
            float sum = result[0] + result[1] + result[2] + result[3]
                      + result[4] + result[5] + result[6] + result[7];

            for (; n < d_state; n++) {
                sum += C[n] * h[(p * d_state + n) * head_dim + d];
            }

            y[p * head_dim + d] = sum;
        }
    }
}
#endif

void ssm_scan_step(float* y, float* h, const float* x, const float* dt,
                   const float* A, const float* B, const float* C, float* dt_heads,
                   int32_t n_heads, int32_t d_state, int32_t head_dim) {
    if (!g_scan_kernel) ssm_scan_init();
    g_scan_kernel(y, h, x, dt, A, B, C, dt_heads, n_heads, d_state, head_dim);
}

void ssm_scan_step_ternary_scalar(float* y, float* h, const float* x, const float* dt,
                                  const int8_t* A, const float* B, const float* C, float* dt_heads,
                                  int32_t n_heads, int32_t d_state, int32_t head_dim) {
    for (int32_t p = 0; p < n_heads; p++) {
        float sum = 0.0f;
        for (int32_t d = 0; d < head_dim; d++) {
            sum += dt[p * head_dim + d];
        }
        dt_heads[p] = sum / (float)head_dim;
    }

    for (int32_t p = 0; p < n_heads; p++) {
        int8_t a_val = A[p];
        float dt_p = dt_heads[p];

        for (int32_t n = 0; n < d_state; n++) {
            float b_scaled = dt_p * B[n];
            int32_t h_off = (p * d_state + n) * head_dim;

            for (int32_t d = 0; d < head_dim; d++) {
                float x_d = x[p * head_dim + d];
                if (a_val == 1) {
                    h[h_off + d] = h[h_off + d] + b_scaled * x_d;
                } else if (a_val == -1) {
                    h[h_off + d] = -h[h_off + d] + b_scaled * x_d;
                } else {
                    h[h_off + d] = b_scaled * x_d;
                }
            }
        }
    }

    for (int32_t p = 0; p < n_heads; p++) {
        for (int32_t d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (int32_t n = 0; n < d_state; n++) {
                acc += C[n] * h[(p * d_state + n) * head_dim + d];
            }
            y[p * head_dim + d] = acc;
        }
    }
}

#ifdef ENGINE_AVX2
void ssm_scan_step_ternary_avx2(float* y, float* h, const float* x, const float* dt,
                                const int8_t* A, const float* B, const float* C, float* dt_heads,
                                int32_t n_heads, int32_t d_state, int32_t head_dim) {
    for (int32_t p = 0; p < n_heads; p++) {
        float sum = 0.0f;
        for (int32_t d = 0; d < head_dim; d++) {
            sum += dt[p * head_dim + d];
        }
        dt_heads[p] = sum / (float)head_dim;
    }

    for (int32_t p = 0; p < n_heads; p++) {
        int8_t a_val = A[p];
        float dt_p = dt_heads[p];

        __m256 a_vec;
        if (a_val == 1) {
            a_vec = _mm256_set1_ps(1.0f);
        } else if (a_val == -1) {
            a_vec = _mm256_set1_ps(-1.0f);
        } else {
            a_vec = _mm256_set1_ps(0.0f);
        }

        for (int32_t n = 0; n < d_state; n++) {
            float b_scaled = dt_p * B[n];
            __m256 bs_vec = _mm256_set1_ps(b_scaled);
            int32_t h_off = (p * d_state + n) * head_dim;

            int32_t d = 0;
            for (; d + 7 < head_dim; d += 8) {
                __m256 h_vec = _mm256_loadu_ps(&h[h_off + d]);
                __m256 x_vec = _mm256_loadu_ps(&x[p * head_dim + d]);

                __m256 h_mult = _mm256_mul_ps(a_vec, h_vec);
                __m256 result = _mm256_add_ps(h_mult, _mm256_mul_ps(bs_vec, x_vec));

                _mm256_storeu_ps(&h[h_off + d], result);
            }

            for (; d < head_dim; d++) {
                float x_d = x[p * head_dim + d];
                if (a_val == 1) {
                    h[h_off + d] = h[h_off + d] + b_scaled * x_d;
                } else if (a_val == -1) {
                    h[h_off + d] = -h[h_off + d] + b_scaled * x_d;
                } else {
                    h[h_off + d] = b_scaled * x_d;
                }
            }
        }
    }

    for (int32_t p = 0; p < n_heads; p++) {
        for (int32_t d = 0; d < head_dim; d++) {
            __m256 acc = _mm256_setzero_ps();

            int32_t n = 0;
            for (; n + 7 < d_state; n += 8) {
                __m256 c_vec = _mm256_loadu_ps(&C[n]);
                __m256 h_vec = _mm256_loadu_ps(&h[(p * d_state + n) * head_dim + d]);
                acc = _mm256_fmadd_ps(c_vec, h_vec, acc);
            }

            float result[8];
            _mm256_storeu_ps(result, acc);
            float sum = result[0] + result[1] + result[2] + result[3]
                      + result[4] + result[5] + result[6] + result[7];

            for (; n < d_state; n++) {
                sum += C[n] * h[(p * d_state + n) * head_dim + d];
            }

            y[p * head_dim + d] = sum;
        }
    }
}
#endif

void ssm_scan_step_ternary(float* y, float* h, const float* x, const float* dt,
                           const int8_t* A, const float* B, const float* C, float* dt_heads,
                           int32_t n_heads, int32_t d_state, int32_t head_dim) {
    if (!g_scan_ternary_kernel) ssm_scan_init();
    g_scan_ternary_kernel(y, h, x, dt, A, B, C, dt_heads, n_heads, d_state, head_dim);
}

void ssm_step(float* output, ssm_state_t* state, const ssm_layer_t* layer,
              const float* input, float* scratch) {
    int32_t d_inner = layer->d_inner;
    int32_t dt_rank = layer->dt_rank;
    int32_t d_state = layer->d_state;
    int32_t n_heads = layer->n_heads;

    float* xz = scratch;
    float* x_dbc = scratch + 2 * d_inner;
    float* dt = scratch + 2 * d_inner + dt_rank + 2 * d_state;
    float* y = scratch + 3 * d_inner + dt_rank + 2 * d_state;
    float* dt_h = scratch + 4 * d_inner + dt_rank + 2 * d_state;

    ternary_gemv(xz, layer->in_proj, input, NULL, 2 * d_inner, layer->d_model);

    float* x_conv = xz;
    float* z = xz + d_inner;

    ssm_conv1d_step(x_conv, state->conv_buf, x_conv, layer->conv_weight, layer->conv_bias, d_inner, layer->d_conv);

    act_silu(x_conv, x_conv, d_inner);

    ternary_gemv(x_dbc, layer->x_proj, x_conv, NULL, dt_rank + 2 * d_state, d_inner);

    float* dt_raw = x_dbc;
    float* B = x_dbc + dt_rank;
    float* C = x_dbc + dt_rank + d_state;

    float_gemv(dt, layer->dt_weight, dt_raw, layer->dt_bias, d_inner, dt_rank);

    act_softplus(dt, dt, d_inner);

    if (layer->use_tscs) {
        ssm_scan_step_ternary(y, state->h, x_conv, dt, layer->A_ternary, B, C, dt_h,
                              n_heads, d_state, layer->head_dim);
    } else {
        ssm_scan_step(y, state->h, x_conv, dt, layer->A, B, C, dt_h,
                      n_heads, d_state, layer->head_dim);
    }

    for (int32_t i = 0; i < d_inner; i++) {
        y[i] += x_conv[i] * layer->D[i];
    }

    act_silu_mul(y, y, z, d_inner);

    ternary_gemv(output, layer->out_proj, y, NULL, layer->d_model, d_inner);
}

void ssm_scan_init(void) {
    simd_detect_init();
    engine_backend_t backend = simd_get_backend();

    switch (backend) {
#ifdef ENGINE_AVX2
    case ENGINE_BACKEND_AVX2:
        g_conv1d_kernel = ssm_conv1d_step_avx2;
        g_scan_kernel = ssm_scan_step_avx2;
        g_scan_ternary_kernel = ssm_scan_step_ternary_avx2;
        break;
#endif
    default:
        g_conv1d_kernel = ssm_conv1d_step_scalar;
        g_scan_kernel = ssm_scan_step_scalar;
        g_scan_ternary_kernel = ssm_scan_step_ternary_scalar;
        break;
    }
}
