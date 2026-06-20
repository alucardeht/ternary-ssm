#ifndef SSM_SCAN_H
#define SSM_SCAN_H

#include "ternary_types.h"
#include <stdint.h>

typedef struct {
    float*    h;
    float*    conv_buf;
    int32_t   n_heads;
    int32_t   d_state;
    int32_t   head_dim;
    int32_t   d_inner;
    int32_t   d_conv;
} ssm_state_t;

ssm_state_t* ssm_state_alloc(int32_t n_heads, int32_t d_state, int32_t head_dim, int32_t d_conv);
void ssm_state_free(ssm_state_t* state);
void ssm_state_reset(ssm_state_t* state);

typedef struct {
    ternary_matrix_t* in_proj;
    ternary_matrix_t* x_proj;
    ternary_matrix_t* out_proj;
    float*    dt_weight;
    float*    dt_bias;
    float*    conv_weight;
    float*    conv_bias;
    float*    A;
    int8_t*   A_ternary;
    float*    D;
    int       use_tscs;
    int32_t   d_model, d_inner, d_state, d_conv, dt_rank, n_heads, head_dim;
} ssm_layer_t;

ssm_layer_t* ssm_layer_alloc(int32_t d_model, int32_t d_inner, int32_t d_state,
                              int32_t d_conv, int32_t dt_rank, int32_t n_heads, int use_tscs);
void ssm_layer_free(ssm_layer_t* layer);
size_t ssm_scratch_size(const ssm_layer_t* layer);

void ssm_step(float* output, ssm_state_t* state, const ssm_layer_t* layer,
              const float* input, float* scratch);

void ssm_conv1d_step(float* y, float* conv_buf, const float* x,
                     const float* weight, const float* bias,
                     int32_t d_inner, int32_t d_conv);
void ssm_conv1d_step_scalar(float* y, float* conv_buf, const float* x,
                            const float* weight, const float* bias,
                            int32_t d_inner, int32_t d_conv);

void ssm_scan_step(float* y, float* h, const float* x, const float* dt,
                   const float* A, const float* B, const float* C, float* dt_heads,
                   int32_t n_heads, int32_t d_state, int32_t head_dim);
void ssm_scan_step_scalar(float* y, float* h, const float* x, const float* dt,
                          const float* A, const float* B, const float* C, float* dt_heads,
                          int32_t n_heads, int32_t d_state, int32_t head_dim);

void ssm_scan_step_ternary(float* y, float* h, const float* x, const float* dt,
                           const int8_t* A, const float* B, const float* C, float* dt_heads,
                           int32_t n_heads, int32_t d_state, int32_t head_dim);
void ssm_scan_step_ternary_scalar(float* y, float* h, const float* x, const float* dt,
                                  const int8_t* A, const float* B, const float* C, float* dt_heads,
                                  int32_t n_heads, int32_t d_state, int32_t head_dim);

#ifdef ENGINE_AVX2
void ssm_conv1d_step_avx2(float* y, float* conv_buf, const float* x,
                          const float* weight, const float* bias,
                          int32_t d_inner, int32_t d_conv);
void ssm_scan_step_avx2(float* y, float* h, const float* x, const float* dt,
                        const float* A, const float* B, const float* C, float* dt_heads,
                        int32_t n_heads, int32_t d_state, int32_t head_dim);
void ssm_scan_step_ternary_avx2(float* y, float* h, const float* x, const float* dt,
                                const int8_t* A, const float* B, const float* C, float* dt_heads,
                                int32_t n_heads, int32_t d_state, int32_t head_dim);
#endif

void ssm_scan_init(void);

#endif
