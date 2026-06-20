#ifndef VULKAN_BACKEND_H
#define VULKAN_BACKEND_H

#include "ternary_types.h"
#include "model_io.h"
#include <stdint.h>
#include <stdbool.h>

typedef struct vk_context_s vk_context_t;

/* Lifecycle */
bool vk_available(void);
vk_context_t* vk_init(void);
void vk_shutdown(vk_context_t* ctx);

/* Weight upload (call once after model load). Returns 0=ok, -1=fail */
int vk_upload_weights(vk_context_t* ctx, const trnm_model_t* model);

/* Dispatch — all offsets are byte offsets into weight_buf or scratch_buf */
void vk_ternary_gemv(vk_context_t* ctx,
    uint32_t y_off, uint32_t W_off, uint32_t scale_off,
    uint32_t x_off, uint32_t bias_off, int32_t rows, int32_t cols);

void vk_float_gemv(vk_context_t* ctx,
    uint32_t y_off, uint32_t W_off, uint32_t x_off,
    uint32_t bias_off, int32_t rows, int32_t cols);

void vk_rmsnorm(vk_context_t* ctx,
    uint32_t y_off, uint32_t x_off, uint32_t w_off, int32_t n, float eps);

/* mode: 0=SiLU 1=ReLU 2=mul 3=add 4=residual 5=silu_mul 6=softplus */
void vk_elementwise(vk_context_t* ctx,
    uint32_t y_off, uint32_t a_off, uint32_t b_off, int32_t n, int32_t mode);

void vk_rope(vk_context_t* ctx,
    uint32_t q_off, uint32_t k_off, uint32_t freq_off,
    int32_t n_heads, int32_t head_dim, int32_t pos);

void vk_attention(vk_context_t* ctx,
    uint32_t out_off, uint32_t q_off, uint32_t k_cache_off,
    uint32_t v_cache_off, uint32_t scores_off,
    int32_t n_heads, int32_t head_dim, int32_t cache_len,
    int32_t cache_start, int32_t window_size, float scale);

void vk_conv1d_step(vk_context_t* ctx,
    uint32_t y_off, uint32_t conv_buf_off, uint32_t x_off,
    uint32_t w_off, uint32_t bias_off, int32_t d_inner, int32_t d_conv);

/* CPU<->GPU transfer */
void vk_upload_to_scratch(vk_context_t* ctx, uint32_t off, const void* data, size_t bytes);
void vk_download_from_scratch(vk_context_t* ctx, uint32_t off, void* data, size_t bytes);
void vk_submit_and_wait(vk_context_t* ctx);

/* Query */
const char* vk_device_name(const vk_context_t* ctx);
uint64_t vk_device_memory(const vk_context_t* ctx);

#endif
