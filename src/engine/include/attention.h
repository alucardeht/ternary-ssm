#ifndef ATTENTION_H
#define ATTENTION_H

#include "ternary_types.h"
#include <stdint.h>

typedef struct {
    float* k_cache;
    float* v_cache;
    int32_t write_pos;
    int32_t length;
    int32_t abs_pos;
    int32_t n_heads;
    int32_t head_dim;
    int32_t window_size;
} attn_kv_cache_t;

typedef struct {
    ternary_matrix_t* q_proj;
    ternary_matrix_t* k_proj;
    ternary_matrix_t* v_proj;
    ternary_matrix_t* out_proj;
    ternary_matrix_t* mlp_gate;
    ternary_matrix_t* mlp_up;
    ternary_matrix_t* mlp_down;
    float* attn_norm_weight;
    float* mlp_norm_weight;
    float* rope_inv_freq;
    int32_t d_model;
    int32_t n_heads;
    int32_t head_dim;
    int32_t window_size;
    int32_t mlp_hidden;
} attn_layer_t;

attn_kv_cache_t* attn_kv_cache_alloc(int32_t window_size, int32_t n_heads, int32_t head_dim);
void attn_kv_cache_free(attn_kv_cache_t* cache);
void attn_kv_cache_reset(attn_kv_cache_t* cache);

attn_layer_t* attn_layer_alloc(int32_t d_model, int32_t n_heads, int32_t window_size);
void attn_layer_free(attn_layer_t* layer);
size_t attn_scratch_size(const attn_layer_t* layer);

void attn_step(float* output, attn_kv_cache_t* cache, const attn_layer_t* layer,
               const float* input, float* scratch);

void attn_rope_apply(float* q, float* k, const float* inv_freq,
                     int32_t n_heads, int32_t head_dim, int32_t pos);
void attn_rope_apply_scalar(float* q, float* k, const float* inv_freq,
                            int32_t n_heads, int32_t head_dim, int32_t pos);

void attn_scores(float* scores, const float* q_head, const float* k_cache_head,
                 int32_t head_dim, int32_t cache_len, int32_t cache_start,
                 int32_t window_size, float scale);
void attn_scores_scalar(float* scores, const float* q_head, const float* k_cache_head,
                        int32_t head_dim, int32_t cache_len, int32_t cache_start,
                        int32_t window_size, float scale);

void attn_softmax(float* x, int32_t n);
void attn_softmax_scalar(float* x, int32_t n);

void attn_weighted_sum(float* out, const float* scores, const float* v_cache_head,
                       int32_t head_dim, int32_t cache_len, int32_t cache_start,
                       int32_t window_size);
void attn_weighted_sum_scalar(float* out, const float* scores, const float* v_cache_head,
                              int32_t head_dim, int32_t cache_len, int32_t cache_start,
                              int32_t window_size);

void attn_init(void);

#ifdef ENGINE_AVX2
void attn_rope_apply_avx2(float* q, float* k, const float* inv_freq,
                          int32_t n_heads, int32_t head_dim, int32_t pos);
void attn_scores_avx2(float* scores, const float* q_head, const float* k_cache_head,
                      int32_t head_dim, int32_t cache_len, int32_t cache_start,
                      int32_t window_size, float scale);
void attn_softmax_avx2(float* x, int32_t n);
void attn_weighted_sum_avx2(float* out, const float* scores, const float* v_cache_head,
                            int32_t head_dim, int32_t cache_len, int32_t cache_start,
                            int32_t window_size);
#endif

#endif /* ATTENTION_H */
