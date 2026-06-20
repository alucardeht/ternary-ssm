#include "attention.h"
#include "ternary_gemv.h"
#include "activations.h"
#include "simd_detect.h"
#include "simd_math.h"
#include "ternary_types.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef ENGINE_AVX2
#include <immintrin.h>
#endif

static int g_attn_initialized = 0;
static void (*rope_fn)(float*, float*, const float*, int32_t, int32_t, int32_t) = NULL;
static void (*scores_fn)(float*, const float*, const float*, int32_t, int32_t, int32_t, int32_t, float) = NULL;
static void (*softmax_fn)(float*, int32_t) = NULL;
static void (*wsum_fn)(float*, const float*, const float*, int32_t, int32_t, int32_t, int32_t) = NULL;

void attn_init(void) {
    if (g_attn_initialized) return;

    simd_detect_init();
    engine_backend_t backend = simd_get_backend();

    switch (backend) {
#ifdef ENGINE_AVX2
    case ENGINE_BACKEND_AVX2:
        rope_fn = attn_rope_apply_avx2;
        scores_fn = attn_scores_avx2;
        softmax_fn = attn_softmax_avx2;
        wsum_fn = attn_weighted_sum_avx2;
        break;
#endif
    default:
        rope_fn = attn_rope_apply_scalar;
        scores_fn = attn_scores_scalar;
        softmax_fn = attn_softmax_scalar;
        wsum_fn = attn_weighted_sum_scalar;
        break;
    }

    g_attn_initialized = 1;
}

attn_kv_cache_t* attn_kv_cache_alloc(int32_t window_size, int32_t n_heads, int32_t head_dim) {
    attn_kv_cache_t* cache = (attn_kv_cache_t*)calloc(1, sizeof(attn_kv_cache_t));

    size_t cache_size = (size_t)n_heads * window_size * head_dim;
    cache->k_cache = (float*)calloc(cache_size, sizeof(float));
    cache->v_cache = (float*)calloc(cache_size, sizeof(float));

    cache->write_pos = 0;
    cache->length = 0;
    cache->abs_pos = 0;
    cache->n_heads = n_heads;
    cache->head_dim = head_dim;
    cache->window_size = window_size;

    return cache;
}

void attn_kv_cache_free(attn_kv_cache_t* cache) {
    if (cache) {
        free(cache->k_cache);
        free(cache->v_cache);
        free(cache);
    }
}

void attn_kv_cache_reset(attn_kv_cache_t* cache) {
    if (cache) {
        size_t cache_size = (size_t)cache->n_heads * cache->window_size * cache->head_dim;
        memset(cache->k_cache, 0, cache_size * sizeof(float));
        memset(cache->v_cache, 0, cache_size * sizeof(float));
        cache->write_pos = 0;
        cache->length = 0;
        cache->abs_pos = 0;
    }
}

attn_layer_t* attn_layer_alloc(int32_t d_model, int32_t n_heads, int32_t window_size) {
    attn_layer_t* layer = (attn_layer_t*)calloc(1, sizeof(attn_layer_t));

    layer->d_model = d_model;
    layer->n_heads = n_heads;
    layer->window_size = window_size;
    layer->head_dim = d_model / n_heads;
    layer->mlp_hidden = (int32_t)(d_model * 2.67f);

    layer->q_proj = ternary_matrix_alloc(d_model, d_model);
    layer->k_proj = ternary_matrix_alloc(d_model, d_model);
    layer->v_proj = ternary_matrix_alloc(d_model, d_model);
    layer->out_proj = ternary_matrix_alloc(d_model, d_model);

    layer->mlp_gate = ternary_matrix_alloc(layer->mlp_hidden, d_model);
    layer->mlp_up = ternary_matrix_alloc(layer->mlp_hidden, d_model);
    layer->mlp_down = ternary_matrix_alloc(d_model, layer->mlp_hidden);

    layer->attn_norm_weight = (float*)calloc((size_t)d_model, sizeof(float));
    layer->mlp_norm_weight = (float*)calloc((size_t)d_model, sizeof(float));
    layer->rope_inv_freq = (float*)calloc((size_t)(layer->head_dim / 2), sizeof(float));

    for (int32_t i = 0; i < layer->head_dim / 2; i++) {
        layer->rope_inv_freq[i] = 1.0f / powf(10000.0f, 2.0f * i / layer->head_dim);
    }

    return layer;
}

void attn_layer_free(attn_layer_t* layer) {
    if (layer) {
        ternary_matrix_free(layer->q_proj);
        ternary_matrix_free(layer->k_proj);
        ternary_matrix_free(layer->v_proj);
        ternary_matrix_free(layer->out_proj);
        ternary_matrix_free(layer->mlp_gate);
        ternary_matrix_free(layer->mlp_up);
        ternary_matrix_free(layer->mlp_down);
        free(layer->attn_norm_weight);
        free(layer->mlp_norm_weight);
        free(layer->rope_inv_freq);
        free(layer);
    }
}

size_t attn_scratch_size(const attn_layer_t* layer) {
    int32_t d = layer->d_model;
    int32_t ws = layer->window_size;
    int32_t mlp_h = layer->mlp_hidden;
    return (size_t)(6 * d + ws + 2 * mlp_h) * sizeof(float);
}

void attn_rope_apply_scalar(float* q, float* k, const float* inv_freq,
                            int32_t n_heads, int32_t head_dim, int32_t pos) {
    int32_t half_dim = head_dim / 2;
    float freq_pos = (float)pos;

    for (int32_t h = 0; h < n_heads; h++) {
        float* q_head = q + h * head_dim;
        float* k_head = k + h * head_dim;

        for (int32_t i = 0; i < half_dim; i++) {
            float freq = freq_pos * inv_freq[i];
            float cos_v = cosf(freq);
            float sin_v = sinf(freq);

            float q_i = q_head[i];
            float q_ih = q_head[i + half_dim];
            float k_i = k_head[i];
            float k_ih = k_head[i + half_dim];

            q_head[i] = q_i * cos_v - q_ih * sin_v;
            q_head[i + half_dim] = q_i * sin_v + q_ih * cos_v;

            k_head[i] = k_i * cos_v - k_ih * sin_v;
            k_head[i + half_dim] = k_i * sin_v + k_ih * cos_v;
        }
    }
}

#ifdef ENGINE_AVX2
void attn_rope_apply_avx2(float* q, float* k, const float* inv_freq,
                          int32_t n_heads, int32_t head_dim, int32_t pos) {
    int32_t half_dim = head_dim / 2;
    float freq_pos = (float)pos;

    for (int32_t h = 0; h < n_heads; h++) {
        float* q_head = q + h * head_dim;
        float* k_head = k + h * head_dim;

        int32_t i = 0;
        for (; i + 8 <= half_dim; i += 8) {
            float cos_vals[8], sin_vals[8];
            for (int j = 0; j < 8; j++) {
                float freq = freq_pos * inv_freq[i + j];
                cos_vals[j] = cosf(freq);
                sin_vals[j] = sinf(freq);
            }

            __m256 cos_v = _mm256_loadu_ps(cos_vals);
            __m256 sin_v = _mm256_loadu_ps(sin_vals);

            __m256 q_i = _mm256_loadu_ps(q_head + i);
            __m256 q_ih = _mm256_loadu_ps(q_head + i + half_dim);
            __m256 k_i = _mm256_loadu_ps(k_head + i);
            __m256 k_ih = _mm256_loadu_ps(k_head + i + half_dim);

            __m256 q_i_new = _mm256_fmadd_ps(q_i, cos_v, _mm256_mul_ps(_mm256_mul_ps(q_ih, sin_v), _mm256_set1_ps(-1.0f)));
            __m256 q_ih_new = _mm256_fmadd_ps(q_i, sin_v, _mm256_mul_ps(q_ih, cos_v));
            __m256 k_i_new = _mm256_fmadd_ps(k_i, cos_v, _mm256_mul_ps(_mm256_mul_ps(k_ih, sin_v), _mm256_set1_ps(-1.0f)));
            __m256 k_ih_new = _mm256_fmadd_ps(k_i, sin_v, _mm256_mul_ps(k_ih, cos_v));

            _mm256_storeu_ps(q_head + i, q_i_new);
            _mm256_storeu_ps(q_head + i + half_dim, q_ih_new);
            _mm256_storeu_ps(k_head + i, k_i_new);
            _mm256_storeu_ps(k_head + i + half_dim, k_ih_new);
        }

        for (; i < half_dim; i++) {
            float freq = freq_pos * inv_freq[i];
            float cos_v = cosf(freq);
            float sin_v = sinf(freq);

            float q_i = q_head[i];
            float q_ih = q_head[i + half_dim];
            float k_i = k_head[i];
            float k_ih = k_head[i + half_dim];

            q_head[i] = q_i * cos_v - q_ih * sin_v;
            q_head[i + half_dim] = q_i * sin_v + q_ih * cos_v;

            k_head[i] = k_i * cos_v - k_ih * sin_v;
            k_head[i + half_dim] = k_i * sin_v + k_ih * cos_v;
        }
    }
}
#endif

void attn_scores_scalar(float* scores, const float* q_head, const float* k_cache_head,
                        int32_t head_dim, int32_t cache_len, int32_t cache_start,
                        int32_t window_size, float scale) {
    for (int32_t p = 0; p < cache_len; p++) {
        int32_t pos = (cache_start + p) % window_size;
        float dot = 0.0f;

        for (int32_t d = 0; d < head_dim; d++) {
            dot += q_head[d] * k_cache_head[pos * head_dim + d];
        }

        scores[p] = dot * scale;
    }
}

#ifdef ENGINE_AVX2
void attn_scores_avx2(float* scores, const float* q_head, const float* k_cache_head,
                      int32_t head_dim, int32_t cache_len, int32_t cache_start,
                      int32_t window_size, float scale) {
    for (int32_t p = 0; p < cache_len; p++) {
        int32_t pos = (cache_start + p) % window_size;
        __m256 dot_v = _mm256_set1_ps(0.0f);

        int32_t d = 0;
        for (; d + 8 <= head_dim; d += 8) {
            __m256 q = _mm256_loadu_ps(q_head + d);
            __m256 k = _mm256_loadu_ps(k_cache_head + pos * head_dim + d);
            dot_v = _mm256_fmadd_ps(q, k, dot_v);
        }

        float dot_arr[8];
        _mm256_storeu_ps(dot_arr, dot_v);
        float dot = dot_arr[0] + dot_arr[1] + dot_arr[2] + dot_arr[3] +
                    dot_arr[4] + dot_arr[5] + dot_arr[6] + dot_arr[7];

        for (; d < head_dim; d++) {
            dot += q_head[d] * k_cache_head[pos * head_dim + d];
        }

        scores[p] = dot * scale;
    }
}
#endif

void attn_softmax_scalar(float* x, int32_t n) {
    float max_val = x[0];
    for (int32_t i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    float inv_sum = 1.0f / sum;
    for (int32_t i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
}

#ifdef ENGINE_AVX2
void attn_softmax_avx2(float* x, int32_t n) {
    float max_val = x[0];
    for (int32_t i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    __m256 max_v = _mm256_set1_ps(max_val);
    float sum = 0.0f;

    int32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xi = _mm256_loadu_ps(x + i);
        __m256 shifted = _mm256_sub_ps(xi, max_v);
        __m256 exp_vals = _mm256_set_ps(expf(shifted[7]), expf(shifted[6]), expf(shifted[5]), expf(shifted[4]),
                                         expf(shifted[3]), expf(shifted[2]), expf(shifted[1]), expf(shifted[0]));
        _mm256_storeu_ps(x + i, exp_vals);

        float temp[8];
        _mm256_storeu_ps(temp, exp_vals);
        for (int j = 0; j < 8; j++) sum += temp[j];
    }

    for (; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    __m256 inv_sum = _mm256_set1_ps(1.0f / sum);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xi = _mm256_loadu_ps(x + i);
        __m256 normalized = _mm256_mul_ps(xi, inv_sum);
        _mm256_storeu_ps(x + i, normalized);
    }

    for (; i < n; i++) {
        x[i] /= sum;
    }
}
#endif

void attn_weighted_sum_scalar(float* out, const float* scores, const float* v_cache_head,
                              int32_t head_dim, int32_t cache_len, int32_t cache_start,
                              int32_t window_size) {
    memset(out, 0, (size_t)head_dim * sizeof(float));

    for (int32_t p = 0; p < cache_len; p++) {
        int32_t pos = (cache_start + p) % window_size;
        float score = scores[p];

        for (int32_t d = 0; d < head_dim; d++) {
            out[d] += score * v_cache_head[pos * head_dim + d];
        }
    }
}

#ifdef ENGINE_AVX2
void attn_weighted_sum_avx2(float* out, const float* scores, const float* v_cache_head,
                            int32_t head_dim, int32_t cache_len, int32_t cache_start,
                            int32_t window_size) {
    memset(out, 0, (size_t)head_dim * sizeof(float));

    for (int32_t p = 0; p < cache_len; p++) {
        int32_t pos = (cache_start + p) % window_size;
        __m256 score = _mm256_set1_ps(scores[p]);

        int32_t d = 0;
        for (; d + 8 <= head_dim; d += 8) {
            __m256 out_v = _mm256_loadu_ps(out + d);
            __m256 v = _mm256_loadu_ps(v_cache_head + pos * head_dim + d);
            __m256 prod = _mm256_mul_ps(score, v);
            __m256 result = _mm256_add_ps(out_v, prod);
            _mm256_storeu_ps(out + d, result);
        }

        for (; d < head_dim; d++) {
            out[d] += scores[p] * v_cache_head[pos * head_dim + d];
        }
    }
}
#endif

void attn_rope_apply(float* q, float* k, const float* inv_freq,
                     int32_t n_heads, int32_t head_dim, int32_t pos) {
    if (!g_attn_initialized) attn_init();
    rope_fn(q, k, inv_freq, n_heads, head_dim, pos);
}

void attn_scores(float* scores, const float* q_head, const float* k_cache_head,
                 int32_t head_dim, int32_t cache_len, int32_t cache_start,
                 int32_t window_size, float scale) {
    if (!g_attn_initialized) attn_init();
    scores_fn(scores, q_head, k_cache_head, head_dim, cache_len, cache_start, window_size, scale);
}

void attn_softmax(float* x, int32_t n) {
    if (!g_attn_initialized) attn_init();
    softmax_fn(x, n);
}

void attn_weighted_sum(float* out, const float* scores, const float* v_cache_head,
                       int32_t head_dim, int32_t cache_len, int32_t cache_start,
                       int32_t window_size) {
    if (!g_attn_initialized) attn_init();
    wsum_fn(out, scores, v_cache_head, head_dim, cache_len, cache_start, window_size);
}

void attn_step(float* output, attn_kv_cache_t* cache, const attn_layer_t* layer,
               const float* input, float* scratch) {
    int32_t d = layer->d_model;
    int32_t ws = layer->window_size;
    int32_t mlp_h = layer->mlp_hidden;
    int32_t n_heads = layer->n_heads;
    int32_t head_dim = layer->head_dim;

    float* x_norm = scratch;
    float* q = scratch + d;
    float* k_tmp = scratch + 2 * d;
    float* v_tmp = scratch + 3 * d;
    float* scores = scratch + 4 * d;
    float* attn_out = scratch + 4 * d + ws;
    float* x_res1 = scratch + 5 * d + ws;
    float* mlp_buf1 = scratch + 6 * d + ws;
    float* mlp_buf2 = scratch + 6 * d + ws + mlp_h;

    act_rmsnorm(x_norm, input, layer->attn_norm_weight, d, 1e-5f);

    ternary_gemv(q, layer->q_proj, x_norm, NULL, d, d);
    ternary_gemv(k_tmp, layer->k_proj, x_norm, NULL, d, d);
    ternary_gemv(v_tmp, layer->v_proj, x_norm, NULL, d, d);

    attn_rope_apply(q, k_tmp, layer->rope_inv_freq, n_heads, head_dim, cache->abs_pos);

    for (int32_t h = 0; h < n_heads; h++) {
        float* k_cache_h = cache->k_cache + h * ws * head_dim;
        float* v_cache_h = cache->v_cache + h * ws * head_dim;
        float* k_tmp_h = k_tmp + h * head_dim;
        float* v_tmp_h = v_tmp + h * head_dim;

        memcpy(k_cache_h + cache->write_pos * head_dim, k_tmp_h, (size_t)head_dim * sizeof(float));
        memcpy(v_cache_h + cache->write_pos * head_dim, v_tmp_h, (size_t)head_dim * sizeof(float));
    }

    cache->write_pos = (cache->write_pos + 1) % ws;
    cache->length = cache->length < ws ? cache->length + 1 : ws;
    cache->abs_pos++;

    int32_t cache_start = cache->write_pos;
    memset(attn_out, 0, (size_t)d * sizeof(float));

    for (int32_t h = 0; h < n_heads; h++) {
        float* q_head = q + h * head_dim;
        float* k_cache_h = cache->k_cache + h * ws * head_dim;
        float* v_cache_h = cache->v_cache + h * ws * head_dim;
        float* attn_out_h = attn_out + h * head_dim;
        float* scores_h = scores;

        float scale = 1.0f / sqrtf((float)head_dim);
        attn_scores(scores_h, q_head, k_cache_h, head_dim, cache->length, cache_start, ws, scale);
        attn_softmax(scores_h, cache->length);
        attn_weighted_sum(attn_out_h, scores_h, v_cache_h, head_dim, cache->length, cache_start, ws);
    }

    ternary_gemv(q, layer->out_proj, attn_out, NULL, d, d);

    for (int32_t i = 0; i < d; i++) {
        x_res1[i] = input[i] + q[i];
    }

    act_rmsnorm(x_norm, x_res1, layer->mlp_norm_weight, d, 1e-5f);

    ternary_gemv(mlp_buf1, layer->mlp_gate, x_norm, NULL, mlp_h, d);
    ternary_gemv(mlp_buf2, layer->mlp_up, x_norm, NULL, mlp_h, d);

    act_silu_mul(mlp_buf1, mlp_buf1, mlp_buf2, mlp_h);

    ternary_gemv(x_norm, layer->mlp_down, mlp_buf1, NULL, d, mlp_h);

    for (int32_t i = 0; i < d; i++) {
        output[i] = x_res1[i] + x_norm[i];
    }
}
