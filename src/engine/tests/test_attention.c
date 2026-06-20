#include "attention.h"
#include "ternary_types.h"
#include "ternary_gemv.h"
#include "simd_detect.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("  FAIL: %s\n", #cond); \
        return 0; \
    } \
} while(0)

#define APPROX(a, b, eps) (fabsf((a) - (b)) < (eps))
#define ASSERT_APPROX(a, b, eps) do { \
    if (!APPROX((a), (b), (eps))) { \
        printf("  FAIL: %f != %f (eps=%f)\n", (float)(a), (float)(b), (float)(eps)); \
        return 0; \
    } \
} while(0)

int tests_passed = 0;
int tests_total = 0;

int test_kv_cache_alloc_reset(void) {
    printf("test_kv_cache_alloc_reset...");
    tests_total++;

    attn_kv_cache_t* cache = attn_kv_cache_alloc(4, 2, 4);
    ASSERT(cache != NULL);
    ASSERT(cache->window_size == 4);
    ASSERT(cache->n_heads == 2);
    ASSERT(cache->head_dim == 4);
    ASSERT(cache->k_cache != NULL);
    ASSERT(cache->v_cache != NULL);
    ASSERT(cache->write_pos == 0);
    ASSERT(cache->length == 0);
    ASSERT(cache->abs_pos == 0);

    cache->k_cache[0] = 1.5f;
    cache->v_cache[10] = -2.0f;
    attn_kv_cache_reset(cache);
    ASSERT(cache->k_cache[0] == 0.0f);
    ASSERT(cache->v_cache[10] == 0.0f);

    attn_kv_cache_free(cache);
    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_kv_cache_ring_buffer(void) {
    printf("test_kv_cache_ring_buffer...");
    tests_total++;

    attn_kv_cache_t* cache = attn_kv_cache_alloc(4, 1, 2);
    ASSERT(cache != NULL);

    for (int32_t i = 0; i < 6; i++) {
        int32_t pos = cache->write_pos;
        cache->k_cache[pos * 2 + 0] = (float)(i * 10);
        cache->k_cache[pos * 2 + 1] = (float)(i * 10 + 1);

        cache->write_pos = (cache->write_pos + 1) % cache->window_size;
        cache->abs_pos++;
        if (cache->length < cache->window_size) {
            cache->length++;
        }
    }

    ASSERT(cache->write_pos == 2);
    ASSERT(cache->length == 4);
    ASSERT(cache->abs_pos == 6);

    attn_kv_cache_free(cache);
    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_rope_position_zero(void) {
    printf("test_rope_position_zero...");
    tests_total++;

    int32_t n_heads = 1;
    int32_t head_dim = 4;

    float* inv_freq = (float*)malloc(head_dim / 2 * sizeof(float));
    inv_freq[0] = 1.0f;
    inv_freq[1] = 0.01f;

    float* q = (float*)malloc(head_dim * sizeof(float));
    float* k = (float*)malloc(head_dim * sizeof(float));
    for (int32_t i = 0; i < head_dim; i++) {
        q[i] = (float)(i + 1);
        k[i] = (float)(i + 1);
    }

    float* q_orig = (float*)malloc(head_dim * sizeof(float));
    float* k_orig = (float*)malloc(head_dim * sizeof(float));
    memcpy(q_orig, q, head_dim * sizeof(float));
    memcpy(k_orig, k, head_dim * sizeof(float));

    attn_rope_apply(q, k, inv_freq, n_heads, head_dim, 0);

    for (int32_t i = 0; i < head_dim; i++) {
        ASSERT_APPROX(q[i], q_orig[i], 1e-5f);
        ASSERT_APPROX(k[i], k_orig[i], 1e-5f);
    }

    free(inv_freq);
    free(q);
    free(k);
    free(q_orig);
    free(k_orig);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_rope_known_values(void) {
    printf("test_rope_known_values...");
    tests_total++;

    int32_t n_heads = 1;
    int32_t head_dim = 4;

    float* inv_freq = (float*)malloc(head_dim / 2 * sizeof(float));
    inv_freq[0] = 1.0f;
    inv_freq[1] = 0.01f;

    float* q = (float*)malloc(head_dim * sizeof(float));
    float* k = (float*)malloc(head_dim * sizeof(float));
    q[0] = 1.0f;
    q[1] = 2.0f;
    q[2] = 3.0f;
    q[3] = 4.0f;
    for (int32_t i = 0; i < head_dim; i++) {
        k[i] = q[i];
    }

    attn_rope_apply(q, k, inv_freq, n_heads, head_dim, 1);

    float cos1 = cosf(1.0f);
    float sin1 = sinf(1.0f);
    float cos001 = cosf(0.01f);
    float sin001 = sinf(0.01f);

    float q_out_0 = 1.0f * cos1 - 3.0f * sin1;
    float q_out_1 = 2.0f * cos001 - 4.0f * sin001;
    float q_out_2 = 3.0f * cos1 + 1.0f * sin1;
    float q_out_3 = 4.0f * cos001 + 2.0f * sin001;

    ASSERT_APPROX(q[0], q_out_0, 1e-3f);
    ASSERT_APPROX(q[1], q_out_1, 1e-3f);
    ASSERT_APPROX(q[2], q_out_2, 1e-3f);
    ASSERT_APPROX(q[3], q_out_3, 1e-3f);

    free(inv_freq);
    free(q);
    free(k);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_score_computation(void) {
    printf("test_score_computation...");
    tests_total++;

    int32_t head_dim = 4;
    int32_t cache_len = 3;
    int32_t window_size = 4;
    int32_t cache_start = 0;
    float scale = 0.5f;

    float* q_head = (float*)malloc(head_dim * sizeof(float));
    q_head[0] = 1.0f;
    q_head[1] = 0.0f;
    q_head[2] = 0.0f;
    q_head[3] = 0.0f;

    float* k_cache_head = (float*)malloc(window_size * head_dim * sizeof(float));
    memset(k_cache_head, 0, window_size * head_dim * sizeof(float));

    k_cache_head[0 * head_dim + 0] = 1.0f;
    k_cache_head[1 * head_dim + 0] = 0.0f;
    k_cache_head[1 * head_dim + 1] = 1.0f;
    k_cache_head[2 * head_dim + 0] = 0.5f;
    k_cache_head[2 * head_dim + 1] = 0.5f;

    float* scores = (float*)malloc(cache_len * sizeof(float));
    attn_scores(scores, q_head, k_cache_head, head_dim, cache_len, cache_start, window_size, scale);

    ASSERT_APPROX(scores[0], 0.5f, 1e-5f);
    ASSERT_APPROX(scores[1], 0.0f, 1e-5f);
    ASSERT_APPROX(scores[2], 0.25f, 1e-5f);

    free(q_head);
    free(k_cache_head);
    free(scores);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_softmax_basic(void) {
    printf("test_softmax_basic...");
    tests_total++;

    int32_t n = 3;
    float* x = (float*)malloc(n * sizeof(float));
    x[0] = 1.0f;
    x[1] = 2.0f;
    x[2] = 3.0f;

    attn_softmax(x, n);

    float sum = x[0] + x[1] + x[2];
    ASSERT_APPROX(sum, 1.0f, 1e-5f);
    ASSERT(x[0] > 0.0f && x[1] > 0.0f && x[2] > 0.0f);
    ASSERT(x[0] < x[1] && x[1] < x[2]);

    free(x);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_softmax_single(void) {
    printf("test_softmax_single...");
    tests_total++;

    int32_t n = 1;
    float* x = (float*)malloc(n * sizeof(float));
    x[0] = 5.0f;

    attn_softmax(x, n);

    ASSERT_APPROX(x[0], 1.0f, 1e-5f);

    free(x);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_softmax_large_values(void) {
    printf("test_softmax_large_values...");
    tests_total++;

    int32_t n = 3;
    float* x = (float*)malloc(n * sizeof(float));
    x[0] = 1000.0f;
    x[1] = 1001.0f;
    x[2] = 1002.0f;

    attn_softmax(x, n);

    float sum = x[0] + x[1] + x[2];
    ASSERT(!isnan(sum) && !isinf(sum));
    ASSERT_APPROX(sum, 1.0f, 1e-5f);

    free(x);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_weighted_sum_uniform(void) {
    printf("test_weighted_sum_uniform...");
    tests_total++;

    int32_t head_dim = 2;
    int32_t cache_len = 4;
    int32_t window_size = 4;
    int32_t cache_start = 0;

    float* scores = (float*)malloc(cache_len * sizeof(float));
    for (int32_t i = 0; i < cache_len; i++) {
        scores[i] = 0.25f;
    }

    float* v_cache_head = (float*)malloc(window_size * head_dim * sizeof(float));
    memset(v_cache_head, 0, window_size * head_dim * sizeof(float));

    v_cache_head[0 * head_dim + 0] = 1.0f;
    v_cache_head[0 * head_dim + 1] = 0.0f;
    v_cache_head[1 * head_dim + 0] = 0.0f;
    v_cache_head[1 * head_dim + 1] = 1.0f;
    v_cache_head[2 * head_dim + 0] = 2.0f;
    v_cache_head[2 * head_dim + 1] = 0.0f;
    v_cache_head[3 * head_dim + 0] = 0.0f;
    v_cache_head[3 * head_dim + 1] = 2.0f;

    float* out = (float*)calloc(head_dim, sizeof(float));
    attn_weighted_sum(out, scores, v_cache_head, head_dim, cache_len, cache_start, window_size);

    ASSERT_APPROX(out[0], 0.75f, 1e-5f);
    ASSERT_APPROX(out[1], 0.75f, 1e-5f);

    free(scores);
    free(v_cache_head);
    free(out);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_attn_step_smoke(void) {
    printf("test_attn_step_smoke...");
    tests_total++;

    int32_t d_model = 8;
    int32_t n_heads = 2;
    int32_t window_size = 4;

    attn_layer_t* layer = attn_layer_alloc(d_model, n_heads, window_size);
    ASSERT(layer != NULL);

    for (int32_t i = 0; i < layer->d_model; i++) {
        layer->attn_norm_weight[i] = 1.0f;
        layer->mlp_norm_weight[i] = 1.0f;
    }

    int32_t head_dim = layer->head_dim;
    int32_t dense_size = d_model * d_model;
    int8_t* dense = (int8_t*)calloc(dense_size, sizeof(int8_t));
    for (int32_t i = 0; i < d_model; i++) {
        dense[i * d_model + i] = 1;
    }
    ternary_matrix_pack(layer->q_proj, dense);
    ternary_matrix_pack(layer->k_proj, dense);
    ternary_matrix_pack(layer->v_proj, dense);
    ternary_matrix_pack(layer->out_proj, dense);

    int32_t mlp_hidden = layer->mlp_hidden;
    int32_t mlp_gate_size = mlp_hidden * d_model;
    int32_t mlp_up_size = mlp_hidden * d_model;
    int32_t mlp_down_size = d_model * mlp_hidden;
    int8_t* mlp_gate_dense = (int8_t*)calloc(mlp_gate_size, sizeof(int8_t));
    int8_t* mlp_up_dense = (int8_t*)calloc(mlp_up_size, sizeof(int8_t));
    int8_t* mlp_down_dense = (int8_t*)calloc(mlp_down_size, sizeof(int8_t));

    for (int32_t i = 0; i < d_model; i++) {
        if (i < mlp_hidden) {
            mlp_gate_dense[i * d_model + i] = 1;
            mlp_up_dense[i * d_model + i] = 1;
        }
        if (i < d_model) {
            mlp_down_dense[i * mlp_hidden + i] = 1;
        }
    }

    ternary_matrix_pack(layer->mlp_gate, mlp_gate_dense);
    ternary_matrix_pack(layer->mlp_up, mlp_up_dense);
    ternary_matrix_pack(layer->mlp_down, mlp_down_dense);

    attn_kv_cache_t* cache = attn_kv_cache_alloc(window_size, n_heads, head_dim);
    ASSERT(cache != NULL);

    size_t scratch_sz = attn_scratch_size(layer);
    float* scratch = (float*)malloc(scratch_sz);

    float* input = (float*)malloc(d_model * sizeof(float));
    for (int32_t i = 0; i < d_model; i++) {
        input[i] = 1.0f;
    }

    float* output = (float*)malloc(d_model * sizeof(float));
    attn_step(output, cache, layer, input, scratch);

    ASSERT(!isnan(output[0]) && !isinf(output[0]));
    ASSERT(output[0] != 0.0f);

    free(dense);
    free(mlp_gate_dense);
    free(mlp_up_dense);
    free(mlp_down_dense);
    free(scratch);
    free(input);
    free(output);
    attn_kv_cache_free(cache);
    attn_layer_free(layer);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_attn_step_multi_token(void) {
    printf("test_attn_step_multi_token...");
    tests_total++;

    int32_t d_model = 8;
    int32_t n_heads = 2;
    int32_t window_size = 4;

    attn_layer_t* layer = attn_layer_alloc(d_model, n_heads, window_size);
    ASSERT(layer != NULL);

    for (int32_t i = 0; i < d_model; i++) {
        layer->attn_norm_weight[i] = 1.0f;
        layer->mlp_norm_weight[i] = 1.0f;
    }

    int32_t head_dim = layer->head_dim;
    int32_t dense_size = d_model * d_model;
    int8_t* dense = (int8_t*)calloc(dense_size, sizeof(int8_t));
    for (int32_t i = 0; i < d_model; i++) {
        dense[i * d_model + i] = 1;
    }
    ternary_matrix_pack(layer->q_proj, dense);
    ternary_matrix_pack(layer->k_proj, dense);
    ternary_matrix_pack(layer->v_proj, dense);
    ternary_matrix_pack(layer->out_proj, dense);

    int32_t mlp_hidden = layer->mlp_hidden;
    int32_t mlp_gate_size = mlp_hidden * d_model;
    int32_t mlp_up_size = mlp_hidden * d_model;
    int32_t mlp_down_size = d_model * mlp_hidden;
    int8_t* mlp_gate_dense = (int8_t*)calloc(mlp_gate_size, sizeof(int8_t));
    int8_t* mlp_up_dense = (int8_t*)calloc(mlp_up_size, sizeof(int8_t));
    int8_t* mlp_down_dense = (int8_t*)calloc(mlp_down_size, sizeof(int8_t));

    for (int32_t i = 0; i < d_model; i++) {
        if (i < mlp_hidden) {
            mlp_gate_dense[i * d_model + i] = 1;
            mlp_up_dense[i * d_model + i] = 1;
        }
        if (i < d_model) {
            mlp_down_dense[i * mlp_hidden + i] = 1;
        }
    }

    ternary_matrix_pack(layer->mlp_gate, mlp_gate_dense);
    ternary_matrix_pack(layer->mlp_up, mlp_up_dense);
    ternary_matrix_pack(layer->mlp_down, mlp_down_dense);

    attn_kv_cache_t* cache = attn_kv_cache_alloc(window_size, n_heads, head_dim);
    ASSERT(cache != NULL);

    size_t scratch_sz = attn_scratch_size(layer);
    float* scratch = (float*)malloc(scratch_sz);

    float* input = (float*)malloc(d_model * sizeof(float));
    for (int32_t i = 0; i < d_model; i++) {
        input[i] = 1.0f;
    }

    float* output = (float*)malloc(d_model * sizeof(float));

    for (int step = 0; step < 10; step++) {
        attn_step(output, cache, layer, input, scratch);
        ASSERT(!isnan(output[0]) && !isinf(output[0]));
    }

    ASSERT(cache->length == window_size);
    ASSERT(cache->abs_pos == 10);

    free(dense);
    free(mlp_gate_dense);
    free(mlp_up_dense);
    free(mlp_down_dense);
    free(scratch);
    free(input);
    free(output);
    attn_kv_cache_free(cache);
    attn_layer_free(layer);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_layer_alloc_free(void) {
    printf("test_layer_alloc_free...");
    tests_total++;

    int32_t d_model = 16;
    int32_t n_heads = 4;
    int32_t window_size = 8;

    attn_layer_t* layer = attn_layer_alloc(d_model, n_heads, window_size);
    ASSERT(layer != NULL);
    ASSERT(layer->d_model == d_model);
    ASSERT(layer->n_heads == n_heads);
    ASSERT(layer->window_size == window_size);
    ASSERT(layer->head_dim == 4);
    ASSERT(layer->mlp_hidden == (int32_t)(d_model * 2.67f));

    attn_layer_free(layer);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int main(void) {
    printf("=== Attention Tests ===\n\n");

    simd_detect_init();
    simd_print_caps();
    printf("\n");

    attn_init();

    test_kv_cache_alloc_reset();
    test_kv_cache_ring_buffer();
    test_rope_position_zero();
    test_rope_known_values();
    test_score_computation();
    test_softmax_basic();
    test_softmax_single();
    test_softmax_large_values();
    test_weighted_sum_uniform();
    test_attn_step_smoke();
    test_attn_step_multi_token();
    test_layer_alloc_free();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
