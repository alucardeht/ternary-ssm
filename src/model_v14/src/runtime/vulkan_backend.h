#ifndef V12_VULKAN_BACKEND_H
#define V12_VULKAN_BACKEND_H

#include <stddef.h>
#include <stdint.h>
#ifdef V12_HAS_VULKAN
#include <vulkan/vulkan.h>
#endif

typedef struct VulkanContext VulkanContext;
typedef struct {
    uint32_t index;
} VkAsyncTransferHandle;

#define VK_TRANSFER_NULL ((VkAsyncTransferHandle){0u})
#define VK_TRANSFER_VALID(h) ((h).index != 0u && (h).index != UINT32_MAX)


typedef struct {
    uint32_t index;
} VkBufferHandle;

#define VK_BUFFER_NULL ((VkBufferHandle){0u})
#define VK_BUFFER_VALID(h) ((h).index != 0u && (h).index != UINT32_MAX)

typedef enum {
    VK_BUF_STORAGE    = 0x01,
    VK_BUF_TRANSFER_SRC = 0x02,
    VK_BUF_TRANSFER_DST = 0x04,
    VK_BUF_HOST_VISIBLE = 0x08,
} VkBufFlags;

typedef struct {
    VkBufferHandle pos_indices;
    VkBufferHandle neg_indices;
    VkBufferHandle row_ptr_pos;
    VkBufferHandle row_ptr_neg;
    uint32_t       rows;
    uint32_t       cols;
} VkTernaryCSR;

typedef struct {
    uint64_t dispatch_count;
    uint64_t submit_count;
    uint64_t wait_count;
    uint64_t submit_ns;
    uint64_t wait_ns;
    uint64_t dispatch_submit_ns;
    uint64_t dispatch_wait_ns;
    uint64_t dispatch_record_ns;
    uint64_t dispatch_reset_ns;
    uint64_t gpu_cmd_ns;
    uint64_t timed_submit_count;
    uint64_t incomplete_timing_count;
    uint32_t gpu_timing_supported;
} VkRuntimeProfile;

typedef enum {
    VK_R370_SCOPE_UNKNOWN = 0,
    VK_R370_SCOPE_PARTIAL = 1,
    VK_R370_SCOPE_COMPLETE = 2,
} VkR370ScopeStatus;

typedef struct {
    double cpu_step_ms;
    double submit_ms;
    double wait_ms;
    double gpu_cmd_ms;
    double host_gap_ms;
    double dispatch_record_ms;
    double dispatch_submit_ms;
    double dispatch_wait_ms;
    double dispatch_wait_non_gpu_ms;
    double dispatch_reset_ms;
    double profile_snapshot_ms;
    double d_scope_ms;
    double p_scope;
    double other_host_ms;
    double optimistic_speedup;
    double net_delta_optimistic;
    int gpu_timing_available;
    VkR370ScopeStatus scope_status;
} VkR370Summary;

static inline const char *vk_r370_scope_status_string(VkR370ScopeStatus status) {
    switch (status) {
        case VK_R370_SCOPE_COMPLETE: return "complete";
        case VK_R370_SCOPE_PARTIAL:  return "partial";
        default:                     return "unknown";
    }
}

static inline void vk_profile_summarize_r370(const VkRuntimeProfile *profile,
                                             double cpu_step_ms,
                                             double profile_snapshot_ms,
                                             VkR370Summary *out) {
    if (!out) return;

    out->cpu_step_ms = cpu_step_ms;
    out->submit_ms = 0.0;
    out->wait_ms = 0.0;
    out->gpu_cmd_ms = 0.0;
    out->host_gap_ms = 0.0;
    out->dispatch_record_ms = 0.0;
    out->dispatch_submit_ms = 0.0;
    out->dispatch_wait_ms = 0.0;
    out->dispatch_wait_non_gpu_ms = 0.0;
    out->dispatch_reset_ms = 0.0;
    out->profile_snapshot_ms = profile_snapshot_ms;
    out->d_scope_ms = 0.0;
    out->p_scope = 0.0;
    out->other_host_ms = 0.0;
    out->optimistic_speedup = 1.0;
    out->net_delta_optimistic = 0.0;
    out->gpu_timing_available = 0;
    out->scope_status = VK_R370_SCOPE_UNKNOWN;

    if (!profile) return;

    out->submit_ms = (double)profile->submit_ns / 1e6;
    out->wait_ms = (double)profile->wait_ns / 1e6;
    out->gpu_cmd_ms = (double)profile->gpu_cmd_ns / 1e6;
    out->dispatch_record_ms = (double)profile->dispatch_record_ns / 1e6;
    out->dispatch_submit_ms = (double)profile->dispatch_submit_ns / 1e6;
    out->dispatch_wait_ms = (double)profile->dispatch_wait_ns / 1e6;
    out->dispatch_reset_ms = (double)profile->dispatch_reset_ns / 1e6;

    if (out->dispatch_record_ms > 0.0 ||
        out->dispatch_submit_ms > 0.0 ||
        out->dispatch_wait_ms > 0.0 ||
        out->dispatch_reset_ms > 0.0) {
        out->scope_status = VK_R370_SCOPE_PARTIAL;
    }

    if (!profile->gpu_timing_supported ||
        profile->timed_submit_count == 0 ||
        profile->incomplete_timing_count != 0 ||
        cpu_step_ms <= 0.0) {
        return;
    }

    out->gpu_timing_available = 1;
    out->scope_status = VK_R370_SCOPE_COMPLETE;
    out->host_gap_ms = cpu_step_ms - out->gpu_cmd_ms;
    if (out->host_gap_ms < 0.0) out->host_gap_ms = 0.0;
    out->dispatch_wait_non_gpu_ms = out->dispatch_wait_ms - out->gpu_cmd_ms;
    if (out->dispatch_wait_non_gpu_ms < 0.0) out->dispatch_wait_non_gpu_ms = 0.0;
    out->d_scope_ms = out->dispatch_record_ms +
                      out->dispatch_submit_ms +
                      out->dispatch_wait_non_gpu_ms +
                      out->dispatch_reset_ms;
    if (out->d_scope_ms > out->host_gap_ms) {
        out->d_scope_ms = out->host_gap_ms;
    }
    out->other_host_ms = out->host_gap_ms - out->d_scope_ms;
    if (out->other_host_ms < 0.0) out->other_host_ms = 0.0;
    out->p_scope = out->d_scope_ms / cpu_step_ms;
    if (out->p_scope < 0.0) out->p_scope = 0.0;
    if (out->p_scope >= 1.0) {
        out->p_scope = 0.999999;
    }
    out->net_delta_optimistic = out->p_scope;
    out->optimistic_speedup = 1.0 / (1.0 - out->p_scope);
}

#define VK_MAX_BUFFERS 768
#define VK_MAX_PIPELINES 128

int vk_init(VulkanContext **ctx);
int vk_is_recording(VulkanContext *ctx);
void vk_set_recording(VulkanContext *ctx, int val);
void vk_profile_mark_dispatch(VulkanContext *ctx);

void vk_bind_descriptor_set(VulkanContext *ctx, int slot);
void vk_bind_ssm_scan_buffers(VulkanContext *ctx, VkBufferHandle A_eff, VkBufferHandle B_eff, VkBufferHandle x, VkBufferHandle C_bilinear, VkBufferHandle h_out);
#ifdef V12_HAS_VULKAN
VkBuffer vk_get_buffer(VulkanContext *ctx, VkBufferHandle h);

int vk_get_pipe_ssm_scan(VulkanContext *ctx);

VkDescriptorSet vk_get_descriptor_set(VulkanContext *ctx, int pipe_idx);
VkDevice vk_get_device(VulkanContext *ctx);

VkCommandBuffer vk_get_command_buffer(VulkanContext *ctx);
VkPipelineLayout vk_get_pipeline_layout(VulkanContext *ctx, int pipe_idx);
VkPipeline vk_get_pipeline(VulkanContext *ctx, int pipe_idx);
#endif

void vk_destroy(VulkanContext *ctx);

VkBufferHandle vk_alloc(VulkanContext *ctx, size_t bytes, VkBufFlags flags);

void vk_free(VulkanContext *ctx, VkBufferHandle handle);

void* vk_map(VulkanContext *ctx, VkBufferHandle handle);
void vk_unmap(VulkanContext *ctx, VkBufferHandle handle);
VkAsyncTransferHandle vk_create_async_transfer(VulkanContext *ctx);
void vk_destroy_async_transfer(VulkanContext *ctx, VkAsyncTransferHandle handle);
int vk_begin_async_transfer(VulkanContext *ctx, VkAsyncTransferHandle handle);
int vk_cmd_copy_buffer(VulkanContext *ctx, VkAsyncTransferHandle handle, VkBufferHandle dst, VkBufferHandle src, size_t bytes);
int vk_submit_async_transfer(VulkanContext *ctx, VkAsyncTransferHandle handle);
int vk_wait_async_transfer(VulkanContext *ctx, VkAsyncTransferHandle handle);


int vk_upload(VulkanContext *ctx, VkBufferHandle dst, const void *src, size_t bytes);

int vk_download_tagged(VulkanContext *ctx, void *dst, VkBufferHandle src, size_t bytes,
                       const char *file, int line, const char *func);

#if defined(V12_DEBUG_VK_DEFERRED) && !defined(V12_VK_DISABLE_DOWNLOAD_MACRO)
#define vk_download(ctx, dst, src, bytes) \
    vk_download_tagged((ctx), (dst), (src), (bytes), __FILE__, __LINE__, __func__)
#else
int vk_download(VulkanContext *ctx, void *dst, VkBufferHandle src, size_t bytes);
#endif

int vk_upload_async(VulkanContext *ctx, VkBufferHandle dst, const void *src, size_t bytes);

int vk_copy_buffer(VulkanContext *ctx, VkBufferHandle dst, VkBufferHandle src, size_t bytes);
int vk_copy_buffer_offsets(VulkanContext *ctx, VkBufferHandle dst, size_t dst_offset, VkBufferHandle src, size_t src_offset, size_t bytes);

int vk_dispatch_ternary_matvec(VulkanContext *ctx,
                                const VkTernaryCSR *csr,
                                VkBufferHandle x,
                                VkBufferHandle y,
                                int batch_size);

int vk_dispatch_ssm_scan(VulkanContext *ctx,
                          VkBufferHandle A_eff,       /* [batch × seq_len × d_state] */
                          VkBufferHandle B_eff,       /* [batch × seq_len × d_state] */
                          VkBufferHandle x,           /* [batch × seq_len × d_state] */
                          VkBufferHandle C_bilinear,  /* [d_state] or VK_BUFFER_NULL */
                          VkBufferHandle h_out,       /* [batch × seq_len × d_state] output */
                          int seq_len,
                          int d_state,
                          int batch_size);

int vk_dispatch_ssm_scan_aas(VulkanContext *ctx,
                              VkBufferHandle A_eff,
                              VkBufferHandle B_eff,
                              VkBufferHandle x,
                              VkBufferHandle C_bilinear,
                              VkBufferHandle ssm_u,
                              VkBufferHandle ssm_v,
                              VkBufferHandle h_out,
                              int seq_len,
                              int d_state,
                              int batch_size);

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
                           int transposed);

int vk_dispatch_q10_adam_update(VulkanContext *ctx, VkBufferHandle W, VkBufferHandle M, VkBufferHandle V, VkBufferHandle G, uint32_t ds, float lr, float beta1, float beta2, float eps);
int vk_dispatch_q10_value_forward(VulkanContext *ctx, VkBufferHandle h, VkBufferHandle W_V, VkBufferHandle V, uint32_t bseq, uint32_t ds);
int vk_dispatch_q10_value_backward(VulkanContext *ctx, VkBufferHandle h, VkBufferHandle V, VkBufferHandle loss, VkBufferHandle W_V, VkBufferHandle dh, VkBufferHandle dW_V, uint32_t batch_size, uint32_t seq_len, uint32_t ds, float lambda);

int vk_dispatch_forward_linear(VulkanContext *ctx, VkBufferHandle shadow, VkBufferHandle x, VkBufferHandle y, uint32_t rows, uint32_t cols, uint32_t bseq, float beta);
int vk_dispatch_backward_linear(VulkanContext *ctx, VkBufferHandle shadow, VkBufferHandle dy, VkBufferHandle dx, uint32_t rows, uint32_t cols, uint32_t bseq, float beta, uint32_t accumulate);

int vk_dispatch_random_projector(VulkanContext *ctx, VkBufferHandle P, uint32_t bseq, uint32_t k, uint32_t seed);
int vk_dispatch_cos_sim_audit(VulkanContext *ctx, VkBufferHandle A, VkBufferHandle B, VkBufferHandle stats, uint32_t N);

int vk_dispatch_fp32_gemm(VulkanContext *ctx,
                           VkBufferHandle A,
                           VkBufferHandle B,
                           VkBufferHandle C,
                           uint32_t T,
                           uint32_t M,
                           uint32_t N,
                           int accumulate);

int vk_dispatch_residual_add(VulkanContext *ctx,
                              VkBufferHandle x,
                              VkBufferHandle y,
                              uint32_t n);

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
                                   uint32_t batch_size);

int vk_dispatch_output_project_wht(VulkanContext *ctx,
                                    VkBufferHandle h_out,
                                    VkBufferHandle C_mat,
                                    VkBufferHandle y_ssm,
                                    uint32_t seq_len,
                                    uint32_t d_state,
                                    uint32_t batch_size);

int vk_dispatch_output_project_backward(VulkanContext *ctx,
                                         VkBufferHandle d_y,
                                         VkBufferHandle C_mat,
                                         VkBufferHandle d_h,
                                         uint32_t seq_len,
                                         uint32_t d_state,
                                         uint32_t batch_size);

int vk_dispatch_ssm_scan_backward(VulkanContext *ctx,
                                   VkBufferHandle A_eff,
                                   VkBufferHandle v_precomputed,
                                   VkBufferHandle dh_out,
                                   int seq_len, int d_state,
                                   int batch_size, uint32_t horizon);

int vk_dispatch_ssm_scan_backward_cr(VulkanContext *ctx,
                                      VkBufferHandle A_eff,
                                      VkBufferHandle G,
                                      VkBufferHandle Lambda,
                                      int seq_len, int d_state,
                                      int batch_size, uint32_t horizon);

int vk_dispatch_ssm_bwd_project_dh(VulkanContext *ctx,
                                   VkBufferHandle dh,
                                   VkBufferHandle h_cache,
                                   int seq_len, int d_state, int batch_size);

int vk_dispatch_ssm_bwd_dh_norm(VulkanContext *ctx,
                                VkBufferHandle dh,
                                VkBufferHandle factor,
                                int seq_len, int d_state, int batch_size);

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
                                     int batch_size);

int vk_dispatch_ssm_fused(VulkanContext *ctx,
                          VkBufferHandle A_eff,
                          VkBufferHandle B_eff,
                          VkBufferHandle x_cv,
                          VkBufferHandle v_pre,
                          VkBufferHandle C_bl,
                          VkBufferHandle a_eff_gate,
                          VkBufferHandle A_vec,
                          VkBufferHandle dA_ps,
                          VkBufferHandle d_dot,
                          VkBufferHandle dB_seq,
                          VkBufferHandle d_x_cv,
                          VkBufferHandle dA_out,
                          VkBufferHandle dbA_out,
                          int seq_len, int d_state,
                          int has_bilinear,
                          int batch_size);

/* R356 F1 slice S5 — GPU replay shader dispatch for Sherman-Morrison /
 * diagonal bilinear reverse walk. Produces h_repl in place of h_cache so the
 * backward chain can skip storing the per-timestep hidden-state cache.
 *
 * Push constants (matching ssm_bwd_replay_f1.comp):
 *   a_mode: 0 = time-invariant A, 1 = A_per_step, 2 = dt-scaled
 *   has_bilinear: 1 if C_bl active
 *   has_boundary: 1 if boundary buffer populated (uint32 per token)
 *
 * Inactive-mode bindings (A_per_step, dt, boundary) may be passed as
 * small dummy buffers — the shader ignores them when the corresponding
 * push-constant flag is zero.
 *
 * Returns 0 on success, -1 on any failure (invalid buffer, uninitialised
 * context, shader module missing). Caller is expected to fall back when
 * this returns non-zero.
 */
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
                                   int batch_size);

int vk_dispatch_conv1d_bwd_weights(VulkanContext *ctx,
                                    VkBufferHandle d_x_cv,
                                    VkBufferHandle x_gated,
                                    VkBufferHandle d_conv_w,
                                    int seq_len, int d_state, int K,
                                    int batch_size);

int vk_dispatch_rms_norm_forward(VulkanContext *ctx,
                                  VkBufferHandle x,
                                  VkBufferHandle gamma,
                                  VkBufferHandle beta,
                                  VkBufferHandle y,
                                  VkBufferHandle rms_out,
                                  int seq_len, int d, float eps,
                                  int batch_size);

int vk_dispatch_rms_norm_backward(VulkanContext *ctx,
                                   VkBufferHandle h_final,
                                   VkBufferHandle d_out,  /* in-place */
                                   VkBufferHandle rms,
                                   VkBufferHandle norm_w,
                                   int seq_len, int d,
                                   int batch_size);

int vk_dispatch_wht_inplace(VulkanContext *ctx,
                             VkBufferHandle data,  /* [batch × seq_len × d_state] read/write */
                             uint32_t seq_len,
                             uint32_t d_state,
                             int batch_size);

int vk_dispatch_token_gather(VulkanContext *ctx,
                              VkBufferHandle embed_E,  /* [vocab × d_embed] */
                              VkBufferHandle tokens,   /* [batch × seq_len] uint32 */
                              VkBufferHandle out,      /* [batch × seq_len × d_embed] */
                              uint32_t seq_len,
                              uint32_t d_embed,
                              int batch_size);

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
    float martingale_lambda);

int vk_dispatch_sampled_ce_scatter_dh(VulkanContext *ctx,
    VkBufferHandle row_ptr_pos, VkBufferHandle pos_indices,
    VkBufferHandle row_ptr_neg, VkBufferHandle neg_indices,
    VkBufferHandle sample_ids, VkBufferHandle d_logits,
    VkBufferHandle d_h,
    uint32_t seq_len, uint32_t d_model, uint32_t n_samples, uint32_t batch_size);

int vk_dispatch_norm_grad_reduce(VulkanContext *ctx,
    VkBufferHandle h_final, VkBufferHandle d_h,
    VkBufferHandle rms, VkBufferHandle d_norm_w, VkBufferHandle d_norm_b,
    uint32_t seq_len, uint32_t d_model, uint32_t total_positions);

int vk_dispatch_dlm_head_scatter(VulkanContext *ctx,
    VkBufferHandle sample_ids, VkBufferHandle d_logits,
    VkBufferHandle norm_out, VkBufferHandle dlm_head,
    uint32_t seq_len, uint32_t d_model, uint32_t n_samples, uint32_t batch_size);

/* R368 M1 — recompute a_eff_gate from sticky x_conv into
 * transient scratch buffers. Mirrors elementwise_gates.comp Step 4.
 *
 * Slice A: writes `a_gate_scratch[idx] = sigmoid(W_A @ x_cv + b_A)`.
 * Slice B: also writes `a_eff_scratch[idx] = A[i] * sigmoid(...)`,
 *          bit-identical to the forward fwd_a_buf. Both outputs are
 *          always produced and are consumed by the default Vulkan
 *          backward path. `V12_DEBUG_M1` remains probe-only. */
int vk_dispatch_a_gate_recompute(VulkanContext *ctx,
    VkBufferHandle x_cv,
    VkBufferHandle W_A,
    VkBufferHandle b_A,
    VkBufferHandle A_vec,
    VkBufferHandle a_gate_scratch,
    VkBufferHandle a_eff_scratch,
    uint32_t seq_len, uint32_t d_state, uint32_t batch_size);

int vk_dispatch_a_backward_dx_cv(VulkanContext *ctx,
    VkBufferHandle d_dot, VkBufferHandle W_A, VkBufferHandle d_x_cv,
    uint32_t seq_len, uint32_t ds, uint32_t batch_size);

int vk_dispatch_conv1d_bwd_input_gate(VulkanContext *ctx,
    VkBufferHandle d_x_cv, VkBufferHandle w_conv,
    VkBufferHandle x_proj, VkBufferHandle mask_val,
    VkBufferHandle d_xp, VkBufferHandle d_tau,
    uint32_t seq_len, uint32_t d_state, float tau, float scale,
    uint32_t batch_size);

int vk_dispatch_embed_e_backward(VulkanContext *ctx,
    VkBufferHandle d_h, VkBufferHandle embed_P,
    VkBufferHandle tokens, VkBufferHandle d_embed_E,
    uint32_t total_positions, uint32_t d_model, uint32_t d_embed);

int vk_dispatch_dc_bilinear_reduce(VulkanContext *ctx,
    VkBufferHandle dh,
    VkBufferHandle h_cache,
    VkBufferHandle x_cv,
    VkBufferHandle dc_bl,
    uint32_t seq_len, uint32_t d_state, uint32_t batch_size);

int vk_dispatch_direct_b_backward(VulkanContext *ctx,
    VkBufferHandle dB_seq,
    VkBufferHandle x_cv,
    VkBufferHandle B_vec,
    VkBufferHandle d_x_cv,
    VkBufferHandle dB_accum,
    uint32_t seq_len, uint32_t d_state, uint32_t batch_size);

int vk_dispatch_welford_update(VulkanContext *ctx,
    VkBufferHandle h_cache,
    VkBufferHandle moments,
    uint32_t batch_size, uint32_t seq_len, uint32_t d_state);

int vk_dispatch_eligibility(VulkanContext *ctx,
    VkBufferHandle h_cache, VkBufferHandle cov_ema,
    uint32_t batch_size, uint32_t seq_len, uint32_t d_state, float beta);

int vk_dispatch_apply_whitening(VulkanContext *ctx,
    VkBufferHandle h_cache, VkBufferHandle whitening_matrix,
    uint32_t batch_size, uint32_t seq_len, uint32_t d_state);

int vk_dispatch_online_tssi_score_update(VulkanContext *ctx,
    VkBufferHandle h_cache,
    VkBufferHandle d_logits,
    VkBufferHandle score_moments,
    uint32_t batch_size, uint32_t seq_len, uint32_t d_state,
    uint32_t n_samples);

int vk_dispatch_broadcast_fill(VulkanContext *ctx,
    VkBufferHandle src,      /* [d_state] constant vector */
    VkBufferHandle dst,      /* [total * d_state] output  */
    uint32_t total,          /* batch * seq_len           */
    uint32_t d_state);

int vk_dispatch_mtp_loss(VulkanContext *ctx,
    VkBufferHandle h_mtp,    /* [bseq * d_state]          */
    VkBufferHandle mtp_U,    /* [d_state * rank]          */
    VkBufferHandle mtp_V,    /* [d_state * rank]          */
    VkBufferHandle A_vec,    /* [d_state]                 */
    VkBufferHandle d_mtp_U,  /* [d_state * rank] uint CAS */
    VkBufferHandle d_mtp_V,  /* [d_state * rank] uint CAS */
    VkBufferHandle d_y_ssm,  /* [bseq * d_state]          */
    VkBufferHandle loss_acc, /* [1] uint CAS              */
    uint32_t seq_len,
    uint32_t d_state,
    uint32_t mtp_k,
    uint32_t rank,
    float    mtp_weight,
    uint32_t batch_size);

int vk_dispatch_pc_gather_prev(VulkanContext *ctx,
    VkBufferHandle src, VkBufferHandle dst,
    uint32_t batch_size, uint32_t seq_len, uint32_t d);

int vk_dispatch_residual_sub(VulkanContext *ctx,
    VkBufferHandle pred, VkBufferHandle res,
    uint32_t batch_size, uint32_t seq_len, uint32_t d);

int vk_dispatch_pc_negate_gather(VulkanContext *ctx,
    VkBufferHandle src, VkBufferHandle dst,
    uint32_t batch_size, uint32_t seq_len, uint32_t d);

int vk_dispatch_residual_scatter_add(VulkanContext *ctx,
    VkBufferHandle pred, VkBufferHandle res,
    uint32_t batch_size, uint32_t seq_len, uint32_t d);

int vk_dispatch_adam_dense(VulkanContext *ctx,
    VkBufferHandle param, VkBufferHandle grad,
    VkBufferHandle m,     VkBufferHandle v,
    uint32_t N, float lr,
    float beta1, float beta2, float eps,
    float inv_bc1, float inv_bc2,
    float clamp_lo, float clamp_hi);

int vk_dispatch_grad_scale(VulkanContext *ctx,
    VkBufferHandle grad,
    uint32_t N, float scale);

int vk_dispatch_factored_adam_reduce(VulkanContext *ctx,
    VkBufferHandle grad,       VkBufferHandle shadow,
    VkBufferHandle sampled,    VkBufferHandle grad_accum,
    VkBufferHandle g_row_bits, VkBufferHandle g_col_bits,
    VkBufferHandle gq_row_bits, VkBufferHandle gq_col_bits,
    VkBufferHandle cnt_row,    VkBufferHandle cnt_col,
    VkBufferHandle row_map,    VkBufferHandle evidence_stats,
    VkBufferHandle holo_U,     VkBufferHandle holo_V,
    uint32_t rows, uint32_t cols, uint32_t N,
    uint32_t row_map_mode,
    float crystallization_tau, float bi_ste_factor,
    float ema_gamma, float ema_threshold,
    uint32_t evidence_gate, float boundary_eps, uint32_t rank);

int vk_dispatch_factored_adam_moment(VulkanContext *ctx,
    VkBufferHandle m_row,      VkBufferHandle m_col,
    VkBufferHandle v_row,      VkBufferHandle v_col,
    VkBufferHandle g_row_bits, VkBufferHandle g_col_bits,
    VkBufferHandle gq_row_bits, VkBufferHandle gq_col_bits,
    VkBufferHandle cnt_row,    VkBufferHandle cnt_col,
    VkBufferHandle mean_v_out,
    uint32_t rows, uint32_t cols,
    float beta1, float beta2);

int vk_dispatch_factored_adam_update(VulkanContext *ctx,
    VkBufferHandle shadow,     VkBufferHandle grad_accum,
    VkBufferHandle m_row,      VkBufferHandle m_col,
    VkBufferHandle v_row,      VkBufferHandle v_col,
    VkBufferHandle row_map,    VkBufferHandle grad,
    VkBufferHandle sampled,    VkBufferHandle evidence_stats,
    uint32_t rows, uint32_t cols, uint32_t N,
    uint32_t row_map_mode,
    float lr, float eps, float inv_bc1, float inv_bc2,
    float mean_v, float crystallization_tau, float ema_threshold,
    uint32_t evidence_gate, uint32_t evidence_mode);

int vk_dispatch_mvue_resample(VulkanContext *ctx,
    VkBufferHandle shadow, VkBufferHandle sampled, VkBufferHandle flip_counts,
    uint32_t flip_index, uint32_t N, uint32_t rng_seed);

int vk_dispatch_csr_count(VulkanContext *ctx,
    VkBufferHandle sampled, VkBufferHandle count_pos, VkBufferHandle count_neg,
    uint32_t rows, uint32_t cols);

int vk_dispatch_csr_scatter(VulkanContext *ctx,
    VkBufferHandle sampled, VkBufferHandle rpp, VkBufferHandle rpn,
    VkBufferHandle pi, VkBufferHandle ni,
    uint32_t rows, uint32_t cols);

int vk_fill_zero(VulkanContext *ctx, VkBufferHandle buf, size_t bytes);

void vk_barrier(VulkanContext *ctx);

void vk_sync(VulkanContext *ctx);

void vk_batch_checkpoint(VulkanContext *ctx);
void vk_begin_batch(VulkanContext *ctx);
void vk_end_batch(VulkanContext *ctx);
int vk_batch_active(VulkanContext *ctx);

void vk_profile_reset(VulkanContext *ctx);
void vk_profile_snapshot(VulkanContext *ctx, VkRuntimeProfile *out);

void vk_print_stats(VulkanContext *ctx);

size_t vk_allocated_bytes(VulkanContext *ctx);

#endif

void vk_dispatch_q16_project_layer_dh(VulkanContext *ctx, VkBufferHandle bwd_d_h_in, VkBufferHandle bwd_d_h_in_prev_layer, VkBufferHandle sums_buf, size_t elements);
void vk_dispatch_ef03_ring_checkpoint(VulkanContext *ctx, VkBufferHandle h, VkBufferHandle sum, uint32_t bseq, uint32_t ds, uint32_t K);
void vk_dispatch_sp04_warcp(VulkanContext *ctx, VkBufferHandle dh, VkBufferHandle dx, uint32_t total_elements);
void vk_dispatch_sp05_dfmsp(VulkanContext *ctx, VkBufferHandle dh, VkBufferHandle dx, uint32_t total_elements, uint32_t seed);
void vk_dispatch_sp06_bihd(VulkanContext *ctx, VkBufferHandle h, uint32_t total_elements, float delta);

int vk_dispatch_sp07_spbsc(VulkanContext *ctx, VkBufferHandle fwd_h, VkBufferHandle bwd_dh, uint32_t total_elements, float tau, float discount);
int vk_dispatch_sp08_qrf3ac(VulkanContext *ctx, VkBufferHandle fwd_h, VkBufferHandle basis, VkBufferHandle ternary, uint32_t total_elements);
int vk_dispatch_sp11_tmsheaf(VulkanContext *ctx, VkBufferHandle in_A, VkBufferHandle in_B, VkBufferHandle out_sim, uint32_t num_tokens, uint32_t ds);
void vk_dispatch_e10_pscq_compress(VulkanContext *ctx, VkBufferHandle in_data, VkBufferHandle out_data, uint32_t total_elements);
void vk_dispatch_e10_pscq_decompress(VulkanContext *ctx, VkBufferHandle in_data, VkBufferHandle out_data, uint32_t total_elements);
int vk_dispatch_sp09_htr1p(VulkanContext *ctx, VkBufferHandle bwd_dh, VkBufferHandle fwd_h, VkBufferHandle htr1p_u, VkBufferHandle htr1p_v, uint32_t total_elements, float gamma);


#ifdef V12_HAS_VULKAN

void vk_dispatch_vsa_fwht_forward(VulkanContext *ctx, VkBufferHandle state_in, VkBufferHandle state_out, uint32_t num_tokens, uint32_t ds);
void vk_dispatch_vsa_bent_recurrence(VulkanContext *ctx, VkBufferHandle state_in, VkBufferHandle key_in, VkBufferHandle state_out, uint32_t num_tokens, uint32_t ds);
void vk_dispatch_vsa_fractional_laplacian(VulkanContext *ctx, VkBufferHandle in_data, VkBufferHandle alpha_in, VkBufferHandle out_data, uint32_t num_tokens, uint32_t ds);
void vk_dispatch_vsa_dyadic_translation(VulkanContext *ctx, VkBufferHandle in_data, VkBufferHandle shifts_in, VkBufferHandle out_data, uint32_t num_tokens, uint32_t ds);
void vk_dispatch_vsa_dyadic_ternary_convolution(VulkanContext *ctx, VkBufferHandle in_data, VkBufferHandle weight_data, VkBufferHandle out_data, uint32_t num_tokens, uint32_t ds);
void vk_dispatch_vsa_dynamic_permutation_routing(VulkanContext *ctx, VkBufferHandle in_data, VkBufferHandle shift_data, VkBufferHandle out_data, uint32_t num_tokens, uint32_t ds);
VkPipelineLayout vk_get_global_pipeline_layout(VulkanContext *ctx);
#endif

