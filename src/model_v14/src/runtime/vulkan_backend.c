#define V12_VK_DISABLE_DOWNLOAD_MACRO
#include "vulkan_backend.h"
#undef V12_VK_DISABLE_DOWNLOAD_MACRO
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct VulkanContext {
    int initialized;
    size_t allocated;
    VkRuntimeProfile profile;
};

int vk_init(VulkanContext **ctx) {
    printf("[VK] Vulkan backend not compiled (need -DV12_HAS_VULKAN + Vulkan SDK)\n");
    printf("[VK] Falling back to CPU\n");
    *ctx = NULL;
    return -1;
}

void vk_destroy(VulkanContext *ctx) {
    if (ctx) free(ctx);
}

VkBufferHandle vk_alloc(VulkanContext *ctx, size_t bytes, VkBufFlags flags) {
    return VK_BUFFER_NULL;
}

void vk_free(VulkanContext *ctx, VkBufferHandle handle) {
}

int vk_upload(VulkanContext *ctx, VkBufferHandle dst, const void *src, size_t bytes) {
    return -1;
}

int vk_download_tagged(VulkanContext *ctx, void *dst, VkBufferHandle src, size_t bytes,
                       const char *file, int line, const char *func) {
    (void)ctx;
    (void)dst;
    (void)src;
    (void)bytes;
    (void)file;
    (void)line;
    (void)func;
    return -1;
}

int vk_download(VulkanContext *ctx, void *dst, VkBufferHandle src, size_t bytes) {
    return vk_download_tagged(ctx, dst, src, bytes, NULL, 0, NULL);
}

int vk_upload_async(VulkanContext *ctx, VkBufferHandle dst, const void *src, size_t bytes) {
    return -1;
}

int vk_copy_buffer(VulkanContext *ctx, VkBufferHandle dst, VkBufferHandle src, size_t bytes) {
    return -1;
}

int vk_dispatch_ternary_matvec(VulkanContext *ctx,
                                const VkTernaryCSR *csr,
                                VkBufferHandle x,
                                VkBufferHandle y,
                                int batch_size) {
    return -1;
}

int vk_dispatch_ssm_scan(VulkanContext *ctx,
                          VkBufferHandle A_eff, VkBufferHandle B_eff,
                          VkBufferHandle x, VkBufferHandle C_bilinear,
                          VkBufferHandle h_out,
                          int seq_len, int d_state, int batch_size) {
    return -1;
}

int vk_dispatch_ssm_scan_aas(VulkanContext *ctx,
                              VkBufferHandle A_eff, VkBufferHandle B_eff,
                              VkBufferHandle x, VkBufferHandle C_bilinear,
                              VkBufferHandle ssm_u, VkBufferHandle ssm_v,
                              VkBufferHandle h_out,
                              int seq_len, int d_state, int batch_size) {
    return -1;
}

int vk_dispatch_fp32_gemv(VulkanContext *ctx,
                           VkBufferHandle A,
                           VkBufferHandle x,
                           VkBufferHandle y,
                           uint32_t rows,
                           uint32_t cols,
                           uint32_t batch_size,
                           uint32_t x_stride,
                           uint32_t y_stride,
                           int accumulate,
                           int transposed) {
    return -1;
}

int vk_dispatch_fp32_gemm(VulkanContext *ctx,
                           VkBufferHandle A,
                           VkBufferHandle B,
                           VkBufferHandle C,
                           uint32_t T,
                           uint32_t M,
                           uint32_t N,
                           int accumulate) {
    return -1;
}

int vk_dispatch_residual_add(VulkanContext *ctx,
                              VkBufferHandle x,
                              VkBufferHandle y,
                              uint32_t n) {
    return -1;
}

int vk_dispatch_elementwise_gates(VulkanContext *ctx,
                                   VkBufferHandle x_proj,
                                   VkBufferHandle w_conv,
                                   VkBufferHandle B,
                                   VkBufferHandle A,
                                   VkBufferHandle W_A,
                                   VkBufferHandle b_A,
                                   VkBufferHandle a_buf,
                                   VkBufferHandle b_eff,
                                   VkBufferHandle x_cv,
                                   VkBufferHandle x_gated_out,
                                   VkBufferHandle mask_val_out,
                                   uint32_t seq_len,
                                   uint32_t d_state,
                                   float tau,
                                   float scale,
                                   uint32_t batch_size) {
    return -1;
}

int vk_dispatch_output_project_wht(VulkanContext *ctx,
                                    VkBufferHandle h_out,
                                    VkBufferHandle C_mat,
                                    VkBufferHandle y_ssm,
                                    uint32_t seq_len,
                                    uint32_t d_state,
                                    uint32_t batch_size) {
    return -1;
}

int vk_dispatch_output_project_backward(VulkanContext *ctx,
                                         VkBufferHandle d_y,
                                         VkBufferHandle C_mat,
                                         VkBufferHandle d_h,
                                         uint32_t seq_len,
                                         uint32_t d_state,
                                         uint32_t batch_size) {
    return -1;
}

int vk_dispatch_ssm_scan_backward(VulkanContext *ctx,
                                   VkBufferHandle A_eff,
                                   VkBufferHandle v_precomputed,
                                   VkBufferHandle dh_out,
                                   int seq_len, int d_state,
                                   int batch_size, uint32_t horizon) {
    return -1;
}

int vk_dispatch_ssm_bwd_elementwise(VulkanContext *ctx,
                                     VkBufferHandle dh,
                                     VkBufferHandle h_cache,
                                     VkBufferHandle x_cv,
                                     VkBufferHandle B_eff,
                                     VkBufferHandle C_bl,
                                     VkBufferHandle a_eff_gate,
                                     VkBufferHandle A_vec,
                                     VkBufferHandle dA_ps,
                                     VkBufferHandle d_dot,
                                     VkBufferHandle dB_seq,
                                     VkBufferHandle d_x_cv,
                                     VkBufferHandle dA_out,
                                     VkBufferHandle dbA_out,
                                     VkBufferHandle factor,
                                     int seq_len, int d_state,
                                     int has_bilinear,
                                     int batch_size) {
    return -1;
}

int vk_dispatch_ssm_bwd_replay_f1(VulkanContext *ctx,
                                   VkBufferHandle A,
                                   VkBufferHandle A_per_step,
                                   VkBufferHandle dt,
                                   VkBufferHandle B_eff,
                                   VkBufferHandle x,
                                   VkBufferHandle C_bl,
                                   VkBufferHandle h_final,
                                   VkBufferHandle boundary,
                                   VkBufferHandle h_repl,
                                   int seq_len, int d_state,
                                   int has_bilinear,
                                   int a_mode,
                                   int has_boundary,
                                   int batch_size) {
    return -1;
}

int vk_dispatch_conv1d_bwd_weights(VulkanContext *ctx,
                                    VkBufferHandle d_x_cv,
                                    VkBufferHandle x_gated,
                                    VkBufferHandle d_conv_w,
                                    int seq_len, int d_state, int K,
                                    int batch_size) {
    return -1;
}

int vk_dispatch_rms_norm_forward(VulkanContext *ctx,
                                  VkBufferHandle x,
                                  VkBufferHandle gamma,
                                  VkBufferHandle beta,
                                  VkBufferHandle y,
                                  VkBufferHandle rms_out,
                                  int seq_len, int d, float eps,
                                  int batch_size) {
    return -1;
}

int vk_dispatch_rms_norm_backward(VulkanContext *ctx,
                                   VkBufferHandle h_final,
                                   VkBufferHandle d_out,
                                   VkBufferHandle rms,
                                   VkBufferHandle norm_w,
                                   int seq_len, int d,
                                   int batch_size) {
    return -1;
}

int vk_dispatch_wht_inplace(VulkanContext *ctx,
                             VkBufferHandle data,
                             uint32_t seq_len,
                             uint32_t d_state,
                             int batch_size) {
    return -1;
}

int vk_dispatch_token_gather(VulkanContext *ctx,
                              VkBufferHandle embed_E,
                              VkBufferHandle tokens,
                              VkBufferHandle out,
                              uint32_t seq_len,
                              uint32_t d_embed,
                              int batch_size) {
    return -1;
}

int vk_dispatch_sampled_ce_forward(VulkanContext *ctx,
    VkBufferHandle row_ptr_pos, VkBufferHandle pos_indices,
    VkBufferHandle row_ptr_neg, VkBufferHandle neg_indices,
    VkBufferHandle norm_out, VkBufferHandle tokens,
    VkBufferHandle unigram_cdf, VkBufferHandle log_unigram,
    VkBufferHandle token_freq,
    VkBufferHandle sample_ids_out, VkBufferHandle d_logits_out,
    VkBufferHandle loss_out, VkBufferHandle base_loss_out,
    VkBufferHandle prior_ctx_counts, VkBufferHandle prior_pair_keys,
    VkBufferHandle prior_pair_counts, VkBufferHandle prior_unigram,
    VkBufferHandle program_targets, VkBufferHandle program_counts,
    VkBufferHandle atlas_keys, VkBufferHandle atlas_targets,
    VkBufferHandle atlas_counts, VkBufferHandle ema_logits,
    uint32_t seq_len, uint32_t batch_size, uint32_t rng_seed,
    uint32_t corpus_prior_enabled, uint32_t corpus_prior_pair_capacity,
    float corpus_prior_alpha, float corpus_prior_scale,
    uint32_t logit_prior_gate_enabled,
    uint32_t program_kernel_enabled, float program_kernel_logit_boost,
    uint32_t program_kernel_gate_enabled,
    uint32_t state_atlas_enabled, uint32_t state_atlas_capacity,
    float state_atlas_logit_boost, uint32_t state_atlas_gate_enabled,
    float martingale_lambda) {
    (void)base_loss_out;
    (void)prior_ctx_counts;
    (void)prior_pair_keys;
    (void)prior_pair_counts;
    (void)prior_unigram;
    (void)token_freq;
    (void)ema_logits;
    (void)martingale_lambda;
    (void)program_targets;
    (void)program_counts;
    (void)atlas_keys;
    (void)atlas_targets;
    (void)atlas_counts;
    (void)corpus_prior_enabled;
    (void)corpus_prior_pair_capacity;
    (void)corpus_prior_alpha;
    (void)corpus_prior_scale;
    (void)logit_prior_gate_enabled;
    (void)program_kernel_enabled;
    (void)program_kernel_logit_boost;
    (void)program_kernel_gate_enabled;
    (void)state_atlas_enabled;
    (void)state_atlas_capacity;
    (void)state_atlas_logit_boost;
    (void)state_atlas_gate_enabled;
    return -1;
}

int vk_dispatch_sampled_ce_scatter_dh(VulkanContext *ctx,
    VkBufferHandle row_ptr_pos, VkBufferHandle pos_indices,
    VkBufferHandle row_ptr_neg, VkBufferHandle neg_indices,
    VkBufferHandle sample_ids, VkBufferHandle d_logits,
    VkBufferHandle d_h,
    uint32_t seq_len, uint32_t d_model, uint32_t n_samples, uint32_t batch_size) {
    return -1;
}

int vk_dispatch_norm_grad_reduce(VulkanContext *ctx,
    VkBufferHandle h_final, VkBufferHandle d_h,
    VkBufferHandle rms, VkBufferHandle d_norm_w, VkBufferHandle d_norm_b,
    uint32_t seq_len, uint32_t d_model, uint32_t total_positions) {
    return -1;
}

int vk_dispatch_dlm_head_scatter(VulkanContext *ctx,
    VkBufferHandle sample_ids, VkBufferHandle d_logits,
    VkBufferHandle norm_out, VkBufferHandle dlm_head,
    uint32_t seq_len, uint32_t d_model, uint32_t n_samples, uint32_t batch_size) {
    return -1;
}

int vk_dispatch_a_gate_recompute(VulkanContext *ctx,
    VkBufferHandle x_cv,
    VkBufferHandle W_A,
    VkBufferHandle b_A,
    VkBufferHandle A_vec,
    VkBufferHandle a_gate_scratch,
    VkBufferHandle a_eff_scratch,
    uint32_t seq_len, uint32_t d_state, uint32_t batch_size) {
    (void)ctx; (void)x_cv; (void)W_A; (void)b_A; (void)A_vec;
    (void)a_gate_scratch; (void)a_eff_scratch;
    (void)seq_len; (void)d_state; (void)batch_size;
    return -1;
}

int vk_dispatch_a_backward_dx_cv(VulkanContext *ctx,
    VkBufferHandle d_dot, VkBufferHandle W_A, VkBufferHandle d_x_cv,
    uint32_t seq_len, uint32_t ds, uint32_t batch_size) {
    return -1;
}

int vk_dispatch_conv1d_bwd_input_gate(VulkanContext *ctx,
    VkBufferHandle d_x_cv, VkBufferHandle w_conv,
    VkBufferHandle x_proj, VkBufferHandle mask_val,
    VkBufferHandle d_xp, VkBufferHandle d_tau,
    uint32_t seq_len, uint32_t d_state, float tau, float scale,
    uint32_t batch_size) {
    return -1;
}

int vk_dispatch_embed_e_backward(VulkanContext *ctx,
    VkBufferHandle d_h, VkBufferHandle embed_P,
    VkBufferHandle tokens, VkBufferHandle d_embed_E,
    uint32_t total_positions, uint32_t d_model, uint32_t d_embed) {
    return -1;
}

int vk_dispatch_dc_bilinear_reduce(VulkanContext *ctx,
    VkBufferHandle dh,
    VkBufferHandle h_cache,
    VkBufferHandle x_cv,
    VkBufferHandle dc_bl,
    uint32_t seq_len, uint32_t d_state, uint32_t batch_size) {
    return -1;
}

int vk_dispatch_direct_b_backward(VulkanContext *ctx,
    VkBufferHandle dB_seq,
    VkBufferHandle x_cv,
    VkBufferHandle B_vec,
    VkBufferHandle d_x_cv,
    VkBufferHandle dB_accum,
    uint32_t seq_len, uint32_t d_state, uint32_t batch_size) {
    (void)ctx; (void)dB_seq; (void)x_cv; (void)B_vec;
    (void)d_x_cv; (void)dB_accum;
    (void)seq_len; (void)d_state; (void)batch_size;
    return -1;
}

int vk_dispatch_online_tssi_score_update(VulkanContext *ctx,
    VkBufferHandle h_cache,
    VkBufferHandle d_logits,
    VkBufferHandle score_moments,
    uint32_t batch_size, uint32_t seq_len, uint32_t d_state,
    uint32_t n_samples) {
    (void)ctx; (void)h_cache; (void)d_logits; (void)score_moments;
    (void)batch_size; (void)seq_len; (void)d_state; (void)n_samples;
    return -1;
}

int vk_dispatch_broadcast_fill(VulkanContext *ctx,
    VkBufferHandle src, VkBufferHandle dst,
    uint32_t total, uint32_t d_state) {
    (void)ctx; (void)src; (void)dst; (void)total; (void)d_state;
    return -1;
}

int vk_dispatch_mtp_loss(VulkanContext *ctx,
    VkBufferHandle h_mtp, VkBufferHandle mtp_U, VkBufferHandle mtp_V,
    VkBufferHandle A_vec, VkBufferHandle d_mtp_U, VkBufferHandle d_mtp_V,
    VkBufferHandle d_y_ssm, VkBufferHandle loss_acc,
    uint32_t seq_len, uint32_t d_state, uint32_t mtp_k, uint32_t rank,
    float mtp_weight, uint32_t batch_size) {
    (void)ctx; (void)h_mtp; (void)mtp_U; (void)mtp_V; (void)A_vec;
    (void)d_mtp_U; (void)d_mtp_V; (void)d_y_ssm; (void)loss_acc;
    (void)seq_len; (void)d_state; (void)mtp_k; (void)rank;
    (void)mtp_weight; (void)batch_size;
    return -1;
}

int vk_dispatch_pc_gather_prev(VulkanContext *ctx,
    VkBufferHandle src, VkBufferHandle dst,
    uint32_t batch_size, uint32_t seq_len, uint32_t d) {
    (void)ctx; (void)src; (void)dst; (void)batch_size; (void)seq_len; (void)d;
    return -1;
}

int vk_dispatch_residual_sub(VulkanContext *ctx,
    VkBufferHandle pred, VkBufferHandle res,
    uint32_t batch_size, uint32_t seq_len, uint32_t d) {
    (void)ctx; (void)pred; (void)res; (void)batch_size; (void)seq_len; (void)d;
    return -1;
}

int vk_dispatch_pc_negate_gather(VulkanContext *ctx,
    VkBufferHandle src, VkBufferHandle dst,
    uint32_t batch_size, uint32_t seq_len, uint32_t d) {
    (void)ctx; (void)src; (void)dst; (void)batch_size; (void)seq_len; (void)d;
    return -1;
}

int vk_dispatch_residual_scatter_add(VulkanContext *ctx,
    VkBufferHandle pred, VkBufferHandle res,
    uint32_t batch_size, uint32_t seq_len, uint32_t d) {
    (void)ctx; (void)pred; (void)res; (void)batch_size; (void)seq_len; (void)d;
    return -1;
}

int vk_fill_zero(VulkanContext *ctx, VkBufferHandle buf, size_t bytes) {
    return -1;
}

int vk_dispatch_csr_count(VulkanContext *ctx,
    VkBufferHandle sampled, VkBufferHandle count_pos, VkBufferHandle count_neg,
    uint32_t rows, uint32_t cols) {
    (void)ctx; (void)sampled; (void)count_pos; (void)count_neg; (void)rows; (void)cols;
    return -1;
}

int vk_dispatch_csr_scatter(VulkanContext *ctx,
    VkBufferHandle sampled, VkBufferHandle rpp, VkBufferHandle rpn,
    VkBufferHandle pi, VkBufferHandle ni,
    uint32_t rows, uint32_t cols) {
    (void)ctx; (void)sampled; (void)rpp; (void)rpn; (void)pi; (void)ni;
    (void)rows; (void)cols;
    return -1;
}

void vk_barrier(VulkanContext *ctx) {
}

void vk_sync(VulkanContext *ctx) {
}

void vk_begin_batch(VulkanContext *ctx) { (void)ctx; }
void vk_end_batch(VulkanContext *ctx) { (void)ctx; }
int vk_batch_active(VulkanContext *ctx) { (void)ctx; return 0; }

void vk_profile_reset(VulkanContext *ctx) {
    if (!ctx) return;
    memset(&ctx->profile, 0, sizeof(ctx->profile));
}

void vk_profile_snapshot(VulkanContext *ctx, VkRuntimeProfile *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!ctx) return;
    *out = ctx->profile;
}

void vk_print_stats(VulkanContext *ctx) {
    if (!ctx) {
        printf("[VK] No GPU context\n");
        return;
    }
    printf("[VK] Allocated: %.1f MB\n", (double)ctx->allocated / (1024*1024));
}

size_t vk_allocated_bytes(VulkanContext *ctx) {
    return ctx ? ctx->allocated : 0;
}





