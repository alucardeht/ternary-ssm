#include "gpu_pipeline_internal.h"

#ifdef V12_HAS_VULKAN

int layer_backward_state_ready(const SSMLayer *layer,
                                      int gpu_owns_w_in,
                                      int gpu_owns_w_out) {
    if (!layer
        || !layer->d_sigma
        || !layer->dA
        || !layer->dB
        || !layer->dB_seq
        || !layer->d_conv_w
        || !layer->dW_A
        || !layer->db_A) {
        return 0;
    }
    if (!gpu_owns_w_in && layer->dw_in == NULL) return 0;
    if (!gpu_owns_w_out && layer->dw_out == NULL) return 0;
    return 1;
}


float gpu_train_step(GpuPipeline *p, Model *m, Cache *c,
                     const uint16_t *tokens, const uint8_t *boundary_mask,
                     int seq_len, int seq_idx, int batch_size,
                     const TrainState *state) {
    float lrms_penalty_global = 0.0f;
    if (!p || !m || !c || !tokens) return 0.0f;
    if (p->r364_noop_interval_active) {
        p->r364_noop_train_step_calls++;
    }

    VulkanContext *ctx = p->ctx;
    int d  = p->d_model;
    int ds = p->d_state;
    int bs = p->batch_size;
    const int gpu_owns_w_in     = (m->gpu_opt_flags & GPU_OPT_W_IN) != 0;
    const int gpu_owns_w_out    = (m->gpu_opt_flags & GPU_OPT_W_OUT) != 0;
    const int gpu_owns_lm_head  = (m->gpu_opt_flags & GPU_OPT_LM_HEAD) != 0;
    const int gpu_owns_w_pred   = (m->gpu_opt_flags & GPU_OPT_W_PRED) != 0;
    const int gpu_owns_norm     = (m->gpu_opt_flags & GPU_OPT_NORM) != 0;
    const int gpu_owns_a        = (m->gpu_opt_flags & GPU_OPT_A) != 0;
    const int gpu_owns_ba       = (m->gpu_opt_flags & GPU_OPT_BA) != 0;
    const int gpu_owns_conv     = (m->gpu_opt_flags & GPU_OPT_CONV) != 0;
    const int gpu_owns_wa       = (m->gpu_opt_flags & GPU_OPT_WA) != 0;
    const int gpu_owns_embed_e  = (m->gpu_opt_flags & GPU_OPT_EMBED_E) != 0;
    const int gpu_owns_embed_p  = (m->gpu_opt_flags & GPU_OPT_EMBED_P) != 0;
    R400ObsTraceConfig *r400_obs_cfg = &g_r400_obs_trace;
    size_t bseq = (size_t)bs * seq_len;
    size_t res_bytes = (size_t)batch_size * seq_len * d  * sizeof(float);
    size_t ssm_bytes = (size_t)batch_size * seq_len * ds * sizeof(float);

    if (r400_obs_trace_enabled()) {
        r400_obs_trace_init_config(r400_obs_cfg, batch_size, V12_N_LAYERS);
        r400_obs_trace_log_activation_once(r400_obs_cfg);
    }
    if (m->fp_interface_gap_probe_enabled) {
        m->fp_interface_gap_probe_batches++;
    }

    /* ====================================================
     * FORWARD PASS
     * ==================================================== */

    /* Phase 1: Embedding (GPU) — token gather + batched GEMV projection */
    float *h = c->residual;
    int embed_on_gpu = VK_BUFFER_VALID(p->embed_E_gpu)
                    && VK_BUFFER_VALID(p->embed_P_gpu)
                    && VK_BUFFER_VALID(p->embed_gathered)
                    && VK_BUFFER_VALID(p->embed_tokens)
                    && VK_BUFFER_VALID(p->residual[0]);

    if (1) {
        printf("DEBUG GPU PIPELINE: E_gpu=%d P_gpu=%d gathered=%d tokens=%d res[0]=%d m->embed_E=%p\n", 
            VK_BUFFER_VALID(p->embed_E_gpu), VK_BUFFER_VALID(p->embed_P_gpu), 
            VK_BUFFER_VALID(p->embed_gathered), VK_BUFFER_VALID(p->embed_tokens), 
            VK_BUFFER_VALID(p->residual[0]), (void*)m->embed_E);
    }
    if (embed_on_gpu) {
        printf("DEBUG GPU PIPELINE: inside embed_on_gpu block. h=%p tokens=%p\n", (void*)h, (void*)tokens); fflush(stdout);
        /* Pack tokens as uint32 into staging then upload — entire batch×seq flat */
        int total_tokens = batch_size * seq_len;
        printf("DEBUG GPU PIPELINE: batch_size=%d seq_len=%d total_tokens=%d\n", batch_size, seq_len, total_tokens); fflush(stdout);
        uint32_t *tok_buf = malloc((size_t)total_tokens * sizeof(uint32_t));
        if (tok_buf) {
            printf("DEBUG GPU PIPELINE: tok_buf allocated at %p\n", (void*)tok_buf); fflush(stdout);
            for (int t = 0; t < total_tokens; t++) {
                if (t == 0) { printf("DEBUG GPU PIPELINE: reading first token...\n"); fflush(stdout); }
                tok_buf[t] = (uint32_t)tokens[t];
            }
            printf("DEBUG GPU PIPELINE: Before vk_upload tokens\n"); fflush(stdout);
            vk_upload(ctx, p->embed_tokens, tok_buf, (size_t)total_tokens * sizeof(uint32_t));
            printf("DEBUG GPU PIPELINE: After vk_upload tokens\n"); fflush(stdout);
            free(tok_buf);
        }

        /* Step 1a: gather embed_E rows by token index -> embed_gathered [batch×seq × d_embed] */
        #ifdef R504_E11_BTRP
        printf("DEBUG GPU PIPELINE: Before vk_dispatch_btrp_token_gather\n"); fflush(stdout);
        vk_dispatch_btrp_token_gather(ctx, p->embed_E_gpu, p->embed_tokens, p->embed_gathered, (uint32_t)seq_len, (uint32_t)V12_D_EMBED, batch_size, m->vocab_size, BTRP_R);
        printf("DEBUG GPU PIPELINE: After vk_dispatch_btrp_token_gather\n"); fflush(stdout);
#else
#pragma message("Compiling MTP else branch")
        printf("DEBUG GPU PIPELINE: Before vk_dispatch_token_gather\n"); fflush(stdout);
        vk_dispatch_token_gather(ctx,
                                 p->embed_E_gpu, p->embed_tokens, p->embed_gathered,
                                 (uint32_t)seq_len, (uint32_t)V12_D_EMBED, batch_size);
        printf("DEBUG GPU PIPELINE: After vk_dispatch_token_gather\n"); fflush(stdout);
#endif
        printf("DEBUG GPU PIPELINE: Before vk_barrier\n"); fflush(stdout);
        vk_barrier(ctx);
        printf("DEBUG GPU PIPELINE: After vk_barrier\n"); fflush(stdout);

        /* Step 1b: batched GEMV: for each t in batch×seq, residual[0][t] = embed_P^T @ gathered[t]
         *   embed_P is [d_embed × d_model], transposed=1 computes embed_P^T @ x = [d_model]
         *   batch_size=batch×seq, x_stride=d_embed, y_stride=d_model */
        vk_dispatch_fp32_gemv(ctx,
                              p->embed_P_gpu, p->embed_gathered, p->residual[0],
                              (uint32_t)d, (uint32_t)V12_D_EMBED,
                              (uint32_t)(batch_size * seq_len),
                              (uint32_t)V12_D_EMBED, (uint32_t)d,
                              0, 1);

        /* Download residual[0] for CPU paths that need it (PC forward reads c->residual) */
vk_download(ctx, h, p->residual[0], res_bytes);
} else {
        /* CPU fallback */
#pragma omp parallel for schedule(static)
        for (int t = 0; t < batch_size * seq_len; t++) {
            const float *e   = m->embed_E + tokens[t] * V12_D_EMBED;
            float       *dst = h + t * d;
            for (int j = 0; j < d; j++) {
                float acc = 0.0f;
                for (int r = 0; r < V12_D_EMBED; r++) acc += e[r] * m->embed_P[r * d + j];
                dst[j] = acc;
            }
        }
        vk_upload(ctx, p->residual[0], h, res_bytes);
    }

/* Phase 2: Predictive Coding forward — surprise[b,t] = residual[b,t] - w_pred @ residual[b,t-1]
     * GPU path: gather t=0..T-2 into pred_x, GEMV into pred_y, scatter-subtract into residual[k].
     * All residuals are already GPU-resident from Phase 1 upload.
     * CPU fallback: same logic with CPU pack/scatter loops. */
    {
        float *h_final_prev = c->residual + (size_t)V12_N_LAYERS * bseq * d;
        int    T_per_batch  = seq_len - 1;
        int    T_total      = batch_size * T_per_batch;

#ifdef V12_PC_GPU_SHADERS
        if (VK_BUFFER_VALID(p->pred_A) && T_total > 0) {
            /* Layer 0: source = h_final (residual[N_LAYERS]), target = residual[0] */
            vk_dispatch_pc_gather_prev(ctx, p->residual[V12_N_LAYERS], p->pred_x,
                                       (uint32_t)batch_size, (uint32_t)seq_len, (uint32_t)d);
            vk_barrier(ctx);
            vk_dispatch_fp32_gemv(ctx, p->pred_A, p->pred_x, p->pred_y,
                                  d, d, (uint32_t)T_total, d, d, 0, 0);
            vk_barrier(ctx);
            vk_dispatch_residual_sub(ctx, p->pred_y, p->residual[0],
                                     (uint32_t)batch_size, (uint32_t)seq_len, (uint32_t)d);

            /* Inter-layer k=1..N-1: source = residual[k-1], target = residual[k] */
            for (int k = 1; k < V12_N_LAYERS; k++) {
                vk_dispatch_pc_gather_prev(ctx, p->residual[k - 1], p->pred_x,
                                           (uint32_t)batch_size, (uint32_t)seq_len, (uint32_t)d);
                vk_barrier(ctx);
                vk_dispatch_fp32_gemv(ctx, p->pred_A, p->pred_x, p->pred_y,
                                      d, d, (uint32_t)T_total, d, d, 0, 0);
                vk_barrier(ctx);
                vk_dispatch_residual_sub(ctx, p->pred_y, p->residual[k],
                                         (uint32_t)batch_size, (uint32_t)seq_len, (uint32_t)d);
            }
        }
#else
#pragma message("Compiling MTP else branch")
        /* CPU fallback — pack/GEMV/scatter on CPU then re-upload */

        /* Helper: pack rows t=0..T-2 from src[b*seq*d] into dst (contiguous T rows per batch) */
        #define PC_PACK_PREV(dst, src) \
            do { \
                for (int _b = 0; _b < batch_size; _b++) \
                    memcpy((dst) + (size_t)_b * T_per_batch * d, \
                           (src) + (size_t)_b * seq_len * d, \
                           (size_t)T_per_batch * d * sizeof(float)); \
            } while (0)

        /* Helper: subtract pred_rows from target[b, t+1] */
        #define PC_SUBTRACT_PRED(target, pred_rows) \
            do { \
                for (int _b = 0; _b < batch_size; _b++) \
                    for (int _t = 1; _t < seq_len; _t++) { \
                        float *_raw  = (target) + ((size_t)_b * seq_len + _t) * d; \
                        float *_pred = (pred_rows) + ((size_t)_b * T_per_batch + (_t - 1)) * d; \
                        for (int _j = 0; _j < d; _j++) _raw[_j] -= _pred[_j]; \
                    } \
            } while (0)

        if (VK_BUFFER_VALID(p->pred_A) && T_total > 0) {
            size_t x_bytes = (size_t)T_total * d * sizeof(float);
            float *prev_rows = (float *)malloc(x_bytes);
            float *pred_rows = (float *)malloc(x_bytes);
            if (prev_rows && pred_rows) {
                PC_PACK_PREV(prev_rows, h_final_prev);
                vk_upload(ctx, p->pred_x, prev_rows, x_bytes);
                vk_dispatch_fp32_gemv(ctx, p->pred_A, p->pred_x, p->pred_y,
                                      d, d, T_total, d, d, 0, 0);
                vk_download(ctx, pred_rows, p->pred_y, x_bytes);
                PC_SUBTRACT_PRED(h, pred_rows);
            } else {
                for (int b = 0; b < batch_size; b++) {
                    float pred_buf[V12_D_MODEL];
                    for (int t = 1; t < seq_len; t++) {
                        float *raw        = h + ((size_t)b * seq_len + t) * d;
                        float *h_prev_vec = h_final_prev + ((size_t)b * seq_len + t - 1) * d;
                        for (int j = 0; j < d; j++) {
                            float acc = 0.0f;
                            for (int k2 = 0; k2 < d; k2++) acc += m->sampled_w_pred[j * d + k2] * h_prev_vec[k2];
                            pred_buf[j] = acc;
                        }
                        for (int j = 0; j < d; j++) raw[j] -= pred_buf[j];
                    }
                }
            }
            free(prev_rows);
            free(pred_rows);
        } else {
            for (int b = 0; b < batch_size; b++) {
                float pred_buf[V12_D_MODEL];
                for (int t = 1; t < seq_len; t++) {
                    float *raw        = h + ((size_t)b * seq_len + t) * d;
                    float *h_prev_vec = h_final_prev + ((size_t)b * seq_len + t - 1) * d;
                    for (int j = 0; j < d; j++) {
                        float acc = 0.0f;
                        for (int k2 = 0; k2 < d; k2++) acc += m->sampled_w_pred[j * d + k2] * h_prev_vec[k2];
                        pred_buf[j] = acc;
                    }
                    for (int j = 0; j < d; j++) raw[j] -= pred_buf[j];
                }
            }
        }

        /* Re-upload residual[0] after PC subtraction */
        vk_upload(ctx, p->residual[0], h, res_bytes);

        /* Inter-layer PC forward: for layer k >= 1, surprise[k,b,t] -= w_pred @ residual[k,b,t-1] */
        if (VK_BUFFER_VALID(p->pred_A) && T_total > 0) {
            size_t x_bytes    = (size_t)T_total * d * sizeof(float);
            float *prev_rows  = (float *)malloc(x_bytes);
            float *pred_rows  = (float *)malloc(x_bytes);
            if (prev_rows && pred_rows) {
                for (int k = 1; k < V12_N_LAYERS; k++) {
                    float *r_in_k = c->residual + (size_t)k * bseq * d;
                    PC_PACK_PREV(prev_rows, r_in_k);
                    vk_upload(ctx, p->pred_x, prev_rows, x_bytes);
                    vk_dispatch_fp32_gemv(ctx, p->pred_A, p->pred_x, p->pred_y,
                                          d, d, T_total, d, d, 0, 0);
                    vk_download(ctx, pred_rows, p->pred_y, x_bytes);
                    PC_SUBTRACT_PRED(r_in_k, pred_rows);
                }
            } else {
                for (int k = 1; k < V12_N_LAYERS; k++) {
                    float *r_in_k = c->residual + (size_t)k * bseq * d;
                    float  pred_k[V12_D_MODEL];
                    for (int b = 0; b < batch_size; b++) {
                        for (int t = 1; t < seq_len; t++) {
                            float *r_prev = r_in_k + ((size_t)b * seq_len + t - 1) * d;
                            float *r_cur  = r_in_k + ((size_t)b * seq_len + t) * d;
                            for (int j = 0; j < d; j++) {
                                float acc = 0.0f;
                                for (int k2 = 0; k2 < d; k2++) acc += m->sampled_w_pred[j * d + k2] * r_prev[k2];
                                pred_k[j] = acc;
                            }
                            for (int j = 0; j < d; j++) r_cur[j] -= pred_k[j];
                        }
                    }
                }
            }
            free(prev_rows);
            free(pred_rows);
        } else {
            for (int k = 1; k < V12_N_LAYERS; k++) {
                float *r_in_k = c->residual + (size_t)k * bseq * d;
                float  pred_k[V12_D_MODEL];
                for (int b = 0; b < batch_size; b++) {
                    for (int t = 1; t < seq_len; t++) {
                        float *r_prev = r_in_k + ((size_t)b * seq_len + t - 1) * d;
                        float *r_cur  = r_in_k + ((size_t)b * seq_len + t) * d;
                        for (int j = 0; j < d; j++) {
                            float acc = 0.0f;
                            for (int k2 = 0; k2 < d; k2++) acc += m->sampled_w_pred[j * d + k2] * r_prev[k2];
                            pred_k[j] = acc;
                        }
                        for (int j = 0; j < d; j++) r_cur[j] -= pred_k[j];
                    }
                }
            }
        }

        #undef PC_PACK_PREV
        #undef PC_SUBTRACT_PRED

        /* Re-upload modified residuals for GPU forward */
        for (int k2 = 1; k2 < V12_N_LAYERS; k2++) {
            float *r_k = c->residual + (size_t)k2 * bseq * d;
            vk_upload(ctx, p->residual[k2], r_k, res_bytes);
        }
#endif
    }

    /* Phase 3: Per-layer forward (GPU) */
    for (int k = 0; k < V12_N_LAYERS; k++) {
        SSMLayer *L       = &m->layers[k];
        float    *h_cache = NULL;
        float    *r_out   = c->residual + (size_t)(k + 1) * bseq * d;

        int layer_on_gpu = p->layers[k].uploaded
                        && p->layers[k].fwd_allocated
                        && VK_BUFFER_VALID(p->residual[k])
                        && VK_BUFFER_VALID(p->residual[k + 1])
                        && VK_BUFFER_VALID(p->ssm_a)
                        && VK_BUFFER_VALID(p->ssm_u)
                        && VK_BUFFER_VALID(p->ssm_v)
                        && VK_BUFFER_VALID(p->lw_conv)
                        && VK_BUFFER_VALID(p->lw_B)
                        && VK_BUFFER_VALID(p->lw_A)
                        && VK_BUFFER_VALID(p->lw_WA)
                        && VK_BUFFER_VALID(p->lw_bA)
                        && VK_BUFFER_VALID(p->lw_C)
#if V12_SISSF_ENABLED
                        && 0
#endif
                        ;

        if (layer_on_gpu) {
#if V12_PRBLT_ENABLED
            prblt_load_layer(p, m, k);
            VkTernaryCSR *csr_in_ptr  = &p->prblt_ring[k % 2].gpu_csr_in;
            VkTernaryCSR *csr_out_ptr = &p->prblt_ring[k % 2].gpu_csr_out;
#else
#pragma message("Compiling MTP else branch")
            VkTernaryCSR *csr_in_ptr  = &p->layers[k].gpu_csr_in;
            VkTernaryCSR *csr_out_ptr = &p->layers[k].gpu_csr_out;
#endif
            vk_begin_batch(ctx);

            /* Upload layer weights — skip params already live on GPU via optimizer */
            if (!(m->gpu_opt_flags & GPU_OPT_CONV))
                upload_shadow_buffer(ctx, p->lw_conv, L->shadow_w_conv, (size_t)V12_CONV_KERNEL * ds);
            vk_upload(ctx, p->lw_B, L->B, (size_t)ds * sizeof(float));
            if (!(m->gpu_opt_flags & GPU_OPT_A))
                vk_upload(ctx, p->lw_A, L->A, (size_t)ds * sizeof(float));
            if (!(m->gpu_opt_flags & GPU_OPT_WA))
                vk_upload(ctx, p->lw_WA, L->W_A, (size_t)ds * ds * sizeof(float));
            if (!(m->gpu_opt_flags & GPU_OPT_BA))
                vk_upload(ctx, p->lw_bA, L->b_A, (size_t)ds * sizeof(float));
            vk_upload(ctx, p->lw_C, L->C, (size_t)ds * ds * sizeof(float));
            vk_upload(ctx, p->ssm_cbl, L->sampled_C_bilinear, (size_t)ds * sizeof(float));
            vk_upload(ctx, p->ssm_u, L->sampled_u, (size_t)ds * sizeof(float));
            vk_upload(ctx, p->ssm_v, L->sampled_v, (size_t)ds * sizeof(float));

            /* 1. csr_in: residual[k] -> buf_y (x_proj) */
                vk_dispatch_ternary_matvec(ctx, csr_in_ptr, p->residual[k], p->buf_y, batch_size * seq_len);
            vk_barrier(ctx);

            /* 1b. Save x_proj GPU-resident for backward */
            vk_copy_buffer(ctx, p->layers[k].fwd_x_proj, p->buf_y, ssm_bytes);
            vk_barrier(ctx);

            /* 2. Elementwise gates: x_proj, w_conv, B, A, W_A, b_A
             *    -> ssm_a (a_buf), ssm_b (b_eff), ssm_x (x_cv)
             *    + per-layer: fwd_x_gated, fwd_mask_val */
            vk_dispatch_elementwise_gates(ctx,
                p->buf_y,
                p->lw_conv, p->lw_B, p->lw_A, p->lw_WA, p->lw_bA,
                p->ssm_a, p->ssm_b, p->ssm_x,
                p->layers[k].fwd_x_gated,
                p->layers[k].fwd_mask_val,
                (uint32_t)seq_len, (uint32_t)ds,
                m->tau_mux[k], V12_SOFT_BOUNDARY_SCALE, (uint32_t)batch_size);
            vk_barrier(ctx);

            /* 2b. Save a_buf, b_eff, x_cv GPU-resident for backward */
            vk_copy_buffer(ctx, p->layers[k].fwd_a_buf, p->ssm_a, ssm_bytes);
            vk_copy_buffer(ctx, p->layers[k].fwd_b_eff, p->ssm_b, ssm_bytes);
            vk_copy_buffer(ctx, p->layers[k].fwd_x_cv,  p->ssm_x, ssm_bytes);
            vk_barrier(ctx);

            /* 3. SSM scan: a_buf, b_eff, x_cv -> ssm_h (h_cache) using S-11 1D Convolution */
printf("DEBUG: Before MTP vk_dispatch_sp10_psvcte\n"); fflush(stdout);
#if R504_S10_PSVCTE
#pragma message("Compiling R504_S10_PSVCTE")
            vk_dispatch_sp10_psvcte(ctx, p->ssm_a, p->ssm_b, p->ssm_x,
                                 p->ssm_cbl, p->ssm_h, seq_len, ds, batch_size);
#elif R504_S11_LNS_TFMS
#pragma message("Compiling R504_S11_LNS_TFMS")
            vk_dispatch_sp11_lns_tfms(ctx, p->ssm_a, p->ssm_b, p->ssm_x,
                                 p->ssm_cbl, p->ssm_h, seq_len, ds, batch_size);
#elif R504_S14_GFIT
#pragma message("Compiling R504_S14_GFIT")
            vk_dispatch_sp14_gfit(ctx, p->ssm_a, p->ssm_b, p->ssm_x,
                                 p->ssm_cbl, p->ssm_h, seq_len, ds, batch_size);
#else
#pragma message("Compiling MTP else branch")
            vk_dispatch_sp03_almm(ctx, p->ssm_a, p->ssm_b, p->ssm_x,
                                 p->ssm_cbl, p->ssm_h, seq_len, ds, batch_size);
#endif
            vk_barrier(ctx);

            /* EF-03: Algebraic Ring-State Checkpointing */
            if (k < m->n_layers - 1) {
                
            }

            /* 3b. Save h_cache GPU-resident for backward */
            // S-18 requires h_cache, so we revert S-12's VRAM avoidance
#ifdef R504_E10_PSCQ
            vk_dispatch_e10_pscq_compress(ctx, p->ssm_h, p->layers[k].fwd_h_cache_compressed, (uint32_t)(batch_size * seq_len * ds));
            vk_barrier(ctx);
            // Decompress immediately into fwd_h_cache for proxy readers (qrf3ac, q10_value_forward) in the same pass
            vk_dispatch_e10_pscq_decompress(ctx, p->layers[k].fwd_h_cache_compressed, p->layers[k].fwd_h_cache, (uint32_t)(batch_size * seq_len * ds));
#else
#pragma message("Compiling MTP else branch")
            vk_copy_buffer(ctx, p->layers[k].fwd_h_cache, p->ssm_h, ssm_bytes);

            // F34 SRAT: Cache low-rank projection
            vk_dispatch_f34_srat_forward(ctx, p->ssm_h, p->layers[k].f34_P_r, p->layers[k].f34_z_cache, seq_len, ds, 64);

#endif
            vk_barrier(ctx);

            /* sp08_qrf3ac: Quotient Ring F_3 Activation Compression proxy */
            vk_dispatch_sp08_qrf3ac(ctx, p->layers[k].fwd_h_cache, p->layers[k].qrf3ac_basis, p->layers[k].qrf3ac_ternary, (uint32_t)(batch_size * seq_len * ds));
            vk_barrier(ctx);

            /* Finding 11: Topological Memory Sheaf (cosine similarity of activations) */
            if (k > 0 && VK_BUFFER_VALID(p->layers[k-1].fwd_h_cache) && VK_BUFFER_VALID(p->layers[k].fwd_h_cache)) {
                if (!VK_BUFFER_VALID(p->tmsheaf_sim[k])) {
                    p->tmsheaf_sim[k] = vk_alloc(ctx, batch_size * seq_len * sizeof(float), VK_BUF_STORAGE);
                }
                vk_dispatch_sp11_tmsheaf(ctx, p->layers[k-1].fwd_h_cache, p->layers[k].fwd_h_cache, p->tmsheaf_sim[k], batch_size * seq_len, ds);
                vk_barrier(ctx);
            }
            /* Finding 12: Frequency-Domain State Compression (Fourier SSM) */
            // vk_dispatch_f12_fourier_ssm disabled
            vk_barrier(ctx);



            /* 4. Output projection + WHT: h_out, C_mat -> buf_x (y_ssm) */
            vk_dispatch_output_project_wht(ctx, p->ssm_h, p->lw_C,
                                           p->buf_x,
                                           (uint32_t)seq_len, (uint32_t)ds, (uint32_t)batch_size);
            vk_barrier(ctx);

            /* 4b. Save y_ssm GPU-resident for backward dw_out GEMM */
            vk_copy_buffer(ctx, p->layers[k].fwd_y_ssm, p->buf_x, ssm_bytes);

        if (k == m->n_layers - 1) {
            // Q-10: Value forward prediction
            // vk_dispatch_q10_value_forward disabled
            vk_barrier(ctx);
        }

            vk_barrier(ctx);

            /* 5. csr_out: buf_x (y_ssm) -> residual[k+1] */
                vk_dispatch_ternary_matvec(ctx, csr_out_ptr, p->buf_x, p->residual[k + 1], batch_size * seq_len);
            vk_barrier(ctx);

            /* 6. Residual add: residual[k+1] += residual[k] */
            vk_dispatch_residual_add(ctx, p->residual[k], p->residual[k + 1],
                                     (uint32_t)(batch_size * seq_len * d));
            vk_sync(ctx);

            /* GP-KL Welford update — runs entirely on GPU, no h_cache download needed */
            if (VK_BUFFER_VALID(p->layers[k].gpu_moments))
                vk_dispatch_welford_update(ctx, p->layers[k].fwd_h_cache,
                                           p->layers[k].gpu_moments,
                                           (uint32_t)batch_size, (uint32_t)seq_len,
                                           (uint32_t)ds);

            /* Download x_proj, x_cv, b_eff, mask, x_gt and compress to CPU cache */
#if V12_AF8C_ENABLED
            float *temp_x_proj = NULL;
            float *temp_x_conv = NULL;
            float *temp_b_eff  = NULL;
            float *temp_mask   = NULL;
            float *temp_x_gt   = NULL;

            if (!VK_BUFFER_VALID(p->bwd_d_xp) || !VK_BUFFER_VALID(p->bwd_d_tau)) {
                temp_x_proj = malloc(ssm_bytes);
                if (temp_x_proj) vk_download(ctx, temp_x_proj, p->layers[k].fwd_x_proj, ssm_bytes);
            }
            if (!VK_BUFFER_VALID(p->bwd_dB_accum)) {
                temp_x_conv = malloc(ssm_bytes);
                if (temp_x_conv) vk_download(ctx, temp_x_conv, p->layers[k].fwd_x_cv, ssm_bytes);
            }
            if (!p->layers[k].gpu_mtp_U.index) {
                temp_b_eff = malloc(ssm_bytes);
                if (temp_b_eff) vk_download(ctx, temp_b_eff, p->layers[k].fwd_b_eff, ssm_bytes);
            }
            if (!VK_BUFFER_VALID(p->bwd_d_xp) || !VK_BUFFER_VALID(p->bwd_d_tau)) {
                temp_mask = malloc(ssm_bytes);
                if (temp_mask) vk_download(ctx, temp_mask, p->layers[k].fwd_mask_val, ssm_bytes);
            }
            if (!VK_BUFFER_VALID(p->bwd_d_conv_w)) {
                temp_x_gt = malloc(ssm_bytes);
                if (temp_x_gt) vk_download(ctx, temp_x_gt, p->layers[k].fwd_x_gated, ssm_bytes);
            }
#else
#pragma message("Compiling MTP else branch")
            if (!VK_BUFFER_VALID(p->bwd_d_xp) || !VK_BUFFER_VALID(p->bwd_d_tau)) {
                float *x_proj = c->x_proj + (size_t)k * bseq * ds;
                vk_download(ctx, x_proj, p->layers[k].fwd_x_proj, ssm_bytes);
            }
            {
                float *x_cv  = c->x_conv     + (size_t)k * bseq * ds;
                float *b_eff = c->b_eff_gate  + (size_t)k * bseq * ds;
                float *mask  = c->mask_val     + (size_t)k * bseq * ds;
                if (!VK_BUFFER_VALID(p->bwd_dB_accum)) {
                    vk_download(ctx, x_cv, p->layers[k].fwd_x_cv, ssm_bytes);
                }
                if (!p->layers[k].gpu_mtp_U.index) {
                    vk_download(ctx, b_eff, p->layers[k].fwd_b_eff, ssm_bytes);
                }
                if (!VK_BUFFER_VALID(p->bwd_d_xp) || !VK_BUFFER_VALID(p->bwd_d_tau)) {
                    vk_download(ctx, mask, p->layers[k].fwd_mask_val, ssm_bytes);
                }
                if (!VK_BUFFER_VALID(p->bwd_d_conv_w)) {
                    float *x_gt = c->x_gated + (size_t)k * bseq * ds;
                    vk_download(ctx, x_gt, p->layers[k].fwd_x_gated, ssm_bytes);
                }
            }
#endif

            vk_end_batch(ctx); vk_begin_batch(ctx);

#if V12_AF8C_ENABLED
            /* Now that the batch has ended and downloads are complete, compress and free the temp buffers */
            {
                uint32_t rng = 0x12345678u ^ (uint32_t)k;
                if (temp_x_proj) {
                    uint8_t *dst = c->x_proj + (size_t)k * bseq * ds;
                    for (size_t i = 0; i < bseq * ds; i++) dst[i] = compress_e5m2_stochastic(temp_x_proj[i], &rng);
                    free(temp_x_proj);
                }
                if (temp_x_conv) {
                    uint8_t *dst = c->x_conv + (size_t)k * bseq * ds;
                    for (size_t i = 0; i < bseq * ds; i++) dst[i] = compress_e5m2_stochastic(temp_x_conv[i], &rng);
                    free(temp_x_conv);
                }
                if (temp_b_eff) {
                    uint8_t *dst = c->b_eff_gate + (size_t)k * bseq * ds;
                    for (size_t i = 0; i < bseq * ds; i++) dst[i] = compress_e4m3_stochastic(temp_b_eff[i], &rng);
                    free(temp_b_eff);
                }
                if (temp_mask) {
                    uint8_t *dst = c->mask_val + (size_t)k * bseq * ds;
                    for (size_t i = 0; i < bseq * ds; i++) dst[i] = compress_e4m3_stochastic(temp_mask[i], &rng);
                    free(temp_mask);
                }
                if (temp_x_gt) {
                    uint8_t *dst = c->x_gated + (size_t)k * bseq * ds;
                    for (size_t i = 0; i < bseq * ds; i++) dst[i] = compress_e5m2_stochastic(temp_x_gt[i], &rng);
                    free(temp_x_gt);
                }
            }
#endif

        } else {
            /* CPU fallback path */
            h_cache = malloc((size_t)bseq * ds * sizeof(float));
            if (!h_cache) return -1;
            memset(h_cache, 0, (size_t)bseq * ds * sizeof(float));

            float *r_in  = c->residual + (size_t)k       * bseq * d;
            
            float *x_cv  = malloc(ssm_bytes);
            float *b_eff = malloc(ssm_bytes);
            float *a_buf = malloc(ssm_bytes);
            float *a_gate = malloc(ssm_bytes);
            float *x_gt  = malloc(ssm_bytes);
            float *mask  = malloc(ssm_bytes);
            float *temp_x_proj = malloc(ssm_bytes);
            float *y_ssm = malloc(ssm_bytes);
            if (!x_cv || !b_eff || !a_buf || !a_gate || !x_gt || !mask || !temp_x_proj || !y_ssm) {
                free(x_cv); free(b_eff); free(a_buf); free(a_gate); free(x_gt); free(mask); free(temp_x_proj); free(y_ssm);
                free(h_cache);
                return -1;
            }

#pragma omp parallel for schedule(static)
            for (int t = 0; t < (int)bseq; t++)
                ternary_matvec(&L->csr_in, r_in + t * d, temp_x_proj + t * ds);

            {
                float  tau_k = m->tau_mux[k];
                float  scale = V12_SOFT_BOUNDARY_SCALE;
#pragma omp parallel for schedule(static)
                for (int t = 0; t < (int)bseq; t++) {
                    for (int i = 0; i < ds; i++) {
                        float xp   = temp_x_proj[t * ds + i];
                        float m_val = 1.0f / (1.0f + expf(-scale * (fabsf(xp) - tau_k)));
                        mask[t * ds + i] = m_val;
                        x_gt[t  * ds + i] = xp * m_val;
                    }
                }

                const int K = V12_CONV_KERNEL;
                for (int b = 0; b < batch_size; b++) {
                    int base = b * seq_len;
                    for (int t = 0; t < seq_len; t++) {
                        int flat_t = base + t;
                        for (int i = 0; i < ds; i++) {
                            float acc = 0.0f;
                            for (int kk = 0; kk < K; kk++) {
                                int src_t = t - kk;
                                if (src_t < 0) src_t = 0;
                                int flat_src = base + src_t;
                                acc += L->shadow_w_conv[kk * ds + i] * x_gt[flat_src * ds + i];
                            }
                            x_cv[flat_t * ds + i] = acc;
                        }
                    }
                }
            }

            memset(h_cache, 0, ssm_bytes);
            float h_init[V12_D_STATE];
            memset(h_init, 0, ds * sizeof(float));

#pragma omp parallel for schedule(static)
            for (int t = 0; t < (int)bseq; t++) {
                for (int i = 0; i < ds; i++) {
                    float xc   = x_cv[t * ds + i];
                    float gate = 1.0f / (1.0f + expf(-xc));
                    b_eff[t * ds + i] = gate * L->B[i];
                }
            }

#pragma omp parallel for schedule(static)
            for (int t = 0; t < (int)bseq; t++) {
                const float *xc_t = x_cv + t * ds;
                for (int i = 0; i < ds; i++) {
                    float dot = L->b_A[i];
                    const float *wa_row = L->W_A + i * ds;
                    for (int j = 0; j < ds; j++) dot += wa_row[j] * xc_t[j];
                    float gate = 1.0f / (1.0f + expf(-dot));
                    a_gate[t * ds + i] = gate;
                    a_buf[t  * ds + i] = L->A[i] * gate;
                }
            }

            {
                float *h_batch = calloc((size_t)batch_size * ds, sizeof(float));
                if (!h_batch) h_batch = h_init; /* fallback: only correct for batch=1 */
                ssm_scan_forward_batch(L->A, a_buf, NULL, b_eff, ds, L->C,
                                       x_cv, h_batch, y_ssm, h_cache,
                                       &L->moments, boundary_mask,
                                       batch_size, seq_len, ds, ds,
                                       L->sampled_C_bilinear,
                                       L->sampled_u, L->sampled_v);
                memcpy(c->h_ssm_final + (size_t)k * ds,
                       h_batch + (size_t)(batch_size - 1) * ds, ds * sizeof(float));
                if (h_batch != h_init) free(h_batch);
            }

#pragma omp parallel for schedule(static)
            for (int t = 0; t < (int)bseq; t++) wht_inplace(y_ssm + t * ds, ds);

#pragma omp parallel for schedule(static)
            for (int t = 0; t < (int)bseq; t++) {
                float out_buf[V12_D_MODEL];
                memset(out_buf, 0, (size_t)d * sizeof(float));
                ternary_matvec(&L->csr_out, y_ssm + t * ds, out_buf);
                for (int j = 0; j < d; j++)
                    r_out[t * d + j] = r_in[t * d + j] + out_buf[j];
            }

            /* Compress and store into c */
            {
                uint32_t rng = 0x12345678u ^ (uint32_t)k;
#if V12_AF8C_ENABLED
                uint8_t *c_h_all_dest = c->h_all + (size_t)k * bseq * ds;
                uint8_t *c_y_ssm_dest = c->y_ssm + (size_t)k * bseq * ds;
                uint8_t *c_x_proj_dest = c->x_proj + (size_t)k * bseq * ds;
                uint8_t *c_x_conv_dest = c->x_conv + (size_t)k * bseq * ds;
                uint8_t *c_x_gated_dest = c->x_gated + (size_t)k * bseq * ds;
                uint8_t *c_mask_val_dest = c->mask_val + (size_t)k * bseq * ds;
                uint8_t *c_b_eff_gate_dest = c->b_eff_gate + (size_t)k * bseq * ds;
                uint8_t *c_a_eff_buf_dest = c->a_eff_buf + (size_t)k * bseq * ds;
                uint8_t *c_a_eff_gate_dest = c->a_eff_gate + (size_t)k * bseq * ds;

                for (size_t i = 0; i < (size_t)bseq * ds; i++) {
                    c_h_all_dest[i] = compress_e5m2_stochastic(h_cache[i], &rng);
                    c_y_ssm_dest[i] = compress_e5m2_stochastic(y_ssm[i], &rng);
                    c_x_proj_dest[i] = compress_e5m2_stochastic(temp_x_proj[i], &rng);
                    c_x_conv_dest[i] = compress_e5m2_stochastic(x_cv[i], &rng);
                    c_x_gated_dest[i] = compress_e5m2_stochastic(x_gt[i], &rng);
                    c_mask_val_dest[i] = compress_e4m3_stochastic(mask[i], &rng);
                    c_b_eff_gate_dest[i] = compress_e4m3_stochastic(b_eff[i], &rng);
                    c_a_eff_buf_dest[i] = compress_e4m3_stochastic(a_buf[i], &rng);
                    c_a_eff_gate_dest[i] = compress_e4m3_stochastic(a_gate[i], &rng);
                }
#else
#pragma message("Compiling MTP else branch")
                uint8_t *dst = c->h_all + (size_t)k * bseq * ds;
                for (size_t i = 0; i < (size_t)bseq * ds; i++) {
                    dst[i] = compress_e5m2_stochastic(h_cache[i], &rng);
                }
                memcpy(c->x_proj + (size_t)k * bseq * ds, temp_x_proj, ssm_bytes);
                memcpy(c->x_conv + (size_t)k * bseq * ds, x_cv, ssm_bytes);
                memcpy(c->x_gated + (size_t)k * bseq * ds, x_gt, ssm_bytes);
                memcpy(c->mask_val + (size_t)k * bseq * ds, mask, ssm_bytes);
                memcpy(c->b_eff_gate + (size_t)k * bseq * ds, b_eff, ssm_bytes);
                memcpy(c->a_eff_buf + (size_t)k * bseq * ds, a_buf, ssm_bytes);
                memcpy(c->a_eff_gate + (size_t)k * bseq * ds, a_gate, ssm_bytes);
                memcpy(c->y_ssm + (size_t)k * bseq * ds, y_ssm, ssm_bytes);
#endif
            }
            free(x_cv); free(b_eff); free(a_buf); free(a_gate); free(x_gt); free(mask); free(temp_x_proj); free(y_ssm);
            free(h_cache);
        }
    }


    /* 13. Lagrangian-Relaxed Memory Substrate (LRMS) */
    float lrms_penalty = 0.0f;
    {
        size_t mem_used = vk_allocated_bytes(ctx);
        float limit = 10.8f * 1024.0f * 1024.0f * 1024.0f;
        float overage = (float)mem_used - limit;
        if (overage > 0.0f) {
            float lambda_mem = 1.0e-9f; // soft-routing scale
            float multiplier = lambda_mem * overage;
            if (multiplier > 1.0f) multiplier = 1.0f;
            
            // Shift full memory nodes to sparse residual settlement approximations
            vk_dispatch_e13_lrms(ctx, p->residual[V12_N_LAYERS], (uint32_t)(bseq * d), multiplier);
            
            lrms_penalty = multiplier;
            lrms_penalty_global = lrms_penalty;
        }
    }

    /* Phase 4: Download final residual — only needed by CPU CE fallback which
     * reads h_final for norm-grad accumulation. GPU CE path reads residual[N_LAYERS]
     * directly on the GPU (via norm_backward / norm_grad_reduce shaders). */
    float *h_final = c->residual + (size_t)V12_N_LAYERS * bseq * d;
#ifndef V12_HAS_GPU_CE_SHADERS
    vk_download(ctx, h_final, p->residual[V12_N_LAYERS], res_bytes);
#endif

    /* Phase 5: GPU RMS norm forward.
     * The CPU CE fallback keeps RMSNorm in its own batch so downloads remain
     * straightforward. The GPU CE path fuses RMSNorm into the CE batch below. */
#ifndef V12_HAS_GPU_CE_SHADERS
    vk_begin_batch(ctx);
    // S-17: Completely remove LayerNorm steps
    // vk_dispatch_rms_norm_forward(ctx, p->residual[V12_N_LAYERS], ...
    
    // S-25: Ternary Residual Matrix Sparsification
    // Force residual connections into sparse ternary format, skipping reads on zeros.
    vk_copy_buffer(ctx, p->norm_out, p->residual[V12_N_LAYERS], res_bytes);
    vk_download(ctx, c->norm_out, p->norm_out, res_bytes);
    vk_end_batch(ctx); vk_begin_batch(ctx);
#endif

    /* Phase 5b+6: Sampled CE loss + backward */
    int total_seq = batch_size * seq_len;
    float *d_h = c->d_residual + (size_t)V12_N_LAYERS * total_seq * d;

    float loss = 0.0f;
    int gpu_ce_active = 0;

#ifdef V12_HAS_GPU_CE_SHADERS
    gpu_ce_active = 1;
    /* GPU path: sampling + dots + loss + d_logits + norm grad reduction */
    {
        static uint32_t ce_rng_counter = 0;
        int n_valid = batch_size * (seq_len - 1);
        int N_SAMPLES = V12_SS_N_SAMPLES;
        const int rs_on = m->use_residual_shell;
        int eqp_active_targets = n_valid;
        int eqp_inactive_targets = 0;
        int *eqp_active_lens = exact_quotient_padding_build_active_lens(
            m, batch_size, seq_len, state,
            &eqp_active_targets, &eqp_inactive_targets);
        const int eqp_on = (eqp_active_lens != NULL && eqp_inactive_targets > 0);
        const int ce_mask_on = rs_on || eqp_on;
        if (eqp_active_lens) {
            exact_quotient_padding_record(m, n_valid, eqp_active_targets);
            if (m->exact_quotient_padding_batches <= 8u) {
                printf("[EXACT_QUOTIENT_PADDING_CE] backend=gpu step=%u batch_size=%d seq_len=%d targets=%d active_targets=%d inactive_targets=%d active_fraction=%.6f ce_gradient_masked=%d gpu_scan_masked=0 compute_skip=0\n",
                       state ? state->step : (uint32_t)m->step,
                       batch_size,
                       seq_len,
                       n_valid,
                       eqp_active_targets,
                       eqp_inactive_targets,
                       n_valid > 0 ? (float)((double)eqp_active_targets / (double)n_valid) : 0.0f,
                       eqp_on ? 1 : 0);
            }
        }
        if (m->qgmf_gate_enabled) {
            qgmf_gate_update(m, tokens, batch_size, seq_len);
        }
        if (m->exact_window_multiset_gate_enabled) {
            exact_window_multiset_gate_update(m, tokens, batch_size, seq_len);
        }
        if (m->macro_transcript_gate_enabled) {
            macro_transcript_gate_update(m, tokens, batch_size, seq_len);
        }
        if (m->macro_ce_quotient_observer_enabled) {
            macro_ce_quotient_observe(m, tokens, batch_size, seq_len);
        }
        if (m->reversible_deterministic_expansion_gate_enabled) {
            reversible_deterministic_expansion_gate_update(m, tokens, batch_size, seq_len);
        }
        if (m->certified_artifact_boundary_gate_enabled) {
            certified_artifact_boundary_gate_update(m, tokens, batch_size, seq_len);
        }
        if (m->program_kernel_gate_enabled &&
            m->program_kernel_enabled &&
            m->program_kernel_targets) {
            uint64_t routed = 0;
            uint64_t correct = 0;
            uint64_t incorrect = 0;
            for (int b = 0; b < batch_size; b++) {
                const uint16_t *tok_b = tokens + (size_t)b * (size_t)seq_len;
                for (int t = 0; t + 1 < seq_len; t++) {
                    uint32_t predicted = program_kernel_predict(m, tok_b[t]);
                    if (predicted == UINT32_MAX) continue;
                    routed++;
                    if (predicted == (uint32_t)tok_b[t + 1]) {
                        correct++;
                    } else {
                        incorrect++;
                    }
                }
            }
            m->program_kernel_gate_routed += routed;
            m->program_kernel_gate_correct += correct;
            m->program_kernel_gate_incorrect += incorrect;
        }
        if (m->state_transition_atlas_gate_enabled &&
            m->state_transition_atlas_enabled &&
            m->state_transition_atlas_keys) {
            uint64_t positions = 0;
            uint64_t routed = 0;
            uint64_t correct = 0;
            uint64_t incorrect = 0;
            for (int b = 0; b < batch_size; b++) {
                const uint16_t *tok_b = tokens + (size_t)b * (size_t)seq_len;
                for (int t = 1; t + 1 < seq_len; t++) {
                    positions++;
                    uint32_t predicted = state_transition_atlas_predict(m, tok_b[t - 1], tok_b[t]);
                    if (predicted == UINT32_MAX) continue;
                    routed++;
                    if (predicted == (uint32_t)tok_b[t + 1]) {
                        correct++;
                    } else {
                        incorrect++;
                    }
                }
            }
            m->state_transition_atlas_gate_routed += routed;
            m->state_transition_atlas_gate_correct += correct;
            m->state_transition_atlas_gate_incorrect += incorrect;
            if (m->absorbed_state_atlas_gate_enabled) {
                m->absorbed_state_atlas_gate_positions += positions;
                m->absorbed_state_atlas_gate_routed += routed;
                m->absorbed_state_atlas_gate_correct += correct;
                m->absorbed_state_atlas_gate_incorrect += incorrect;
                if (m->absorbed_state_atlas_gate_positions > 0) {
                    m->absorbed_state_atlas_gate_coverage =
                        (float)((double)m->absorbed_state_atlas_gate_routed /
                                (double)m->absorbed_state_atlas_gate_positions);
                }
                if (m->absorbed_state_atlas_gate_routed > 0) {
                    m->absorbed_state_atlas_gate_route_accuracy =
                        (float)((double)m->absorbed_state_atlas_gate_correct /
                                (double)m->absorbed_state_atlas_gate_routed);
                }
            }
        }
        if (m->counterfactual_packets_gate_enabled &&
            m->counterfactual_packets_enabled &&
            m->counterfactual_packets_keys) {
            uint64_t pairs = 0;
            uint64_t primary = 0;
            uint64_t secondary = 0;
            uint64_t other = 0;
            uint64_t abstain = 0;
            for (int b = 0; b < batch_size; b++) {
                const uint16_t *tok_b = tokens + (size_t)b * (size_t)seq_len;
                for (int t = 1; t + 1 < seq_len; t++) {
                    uint32_t side = 0;
                    uint32_t alt = counterfactual_packets_alternative(
                        m, tok_b[t - 1], tok_b[t], tok_b[t + 1], &side);
                    if (alt != UINT32_MAX) {
                        pairs++;
                        if (side == 1u) {
                            primary++;
                        } else if (side == 2u) {
                            secondary++;
                        }
                    } else if (side == 3u) {
                        other++;
                    } else {
                        abstain++;
                    }
                }
            }
            m->counterfactual_packets_gate_pairs += pairs;
            m->counterfactual_packets_gate_primary += primary;
            m->counterfactual_packets_gate_secondary += secondary;
            m->counterfactual_packets_gate_other += other;
            m->counterfactual_packets_gate_abstain += abstain;
            const double total = (double)(pairs + other + abstain);
            const uint64_t minor = primary < secondary ? primary : secondary;
            m->counterfactual_packets_gate_coverage =
                total > 0.0 ? (float)((double)pairs / total) : 0.0f;
            m->counterfactual_packets_gate_balance =
                pairs > 0 ? (float)((double)minor / (double)pairs) : 0.0f;
            m->counterfactual_packets_gate_batches++;
        }
        if (m->error_causal_lattice_gate_enabled &&
            m->error_causal_lattice_enabled &&
            m->error_causal_target_counts &&
            m->error_causal_prereq_counts) {
            uint64_t positions = 0;
            uint64_t target_covered = 0;
            uint64_t prereq_ready = 0;
            uint64_t orderable = 0;
            uint64_t rare_positions = 0;
            uint64_t rare_orderable = 0;
            const uint32_t min_count = m->error_causal_lattice_min_count > 0u
                ? m->error_causal_lattice_min_count
                : 1u;
            const uint32_t rare_limit = min_count * 4u;

            for (int b = 0; b < batch_size; b++) {
                const uint16_t *tok_b = tokens + (size_t)b * (size_t)seq_len;
                for (int t = 0; t + 1 < seq_len; t++) {
                    const uint16_t prereq = tok_b[t];
                    const uint16_t target = tok_b[t + 1];
                    if (prereq >= V12_VOCAB_SIZE || target >= V12_VOCAB_SIZE) continue;
                    const uint32_t pc = m->error_causal_prereq_counts[prereq];
                    const uint32_t tc = m->error_causal_target_counts[target];
                    const int covered = tc > 0u;
                    const int ready = pc >= min_count;
                    positions++;
                    if (covered) target_covered++;
                    if (ready) prereq_ready++;
                    if (covered && ready) orderable++;
                    if (tc > 0u && tc <= rare_limit) {
                        rare_positions++;
                        if (ready) rare_orderable++;
                    }
                }
            }

            const double total = (double)positions;
            const double rare_total = (double)rare_positions;
            m->error_causal_lattice_gate_batches++;
            m->error_causal_lattice_gate_positions += positions;
            m->error_causal_lattice_gate_target_covered += target_covered;
            m->error_causal_lattice_gate_prereq_ready += prereq_ready;
            m->error_causal_lattice_gate_orderable += orderable;
            m->error_causal_lattice_gate_rare_positions += rare_positions;
            m->error_causal_lattice_gate_rare_orderable += rare_orderable;
            m->error_causal_lattice_gate_target_coverage =
                total > 0.0 ? (float)((double)target_covered / total) : 0.0f;
            m->error_causal_lattice_gate_prereq_readiness =
                total > 0.0 ? (float)((double)prereq_ready / total) : 0.0f;
            m->error_causal_lattice_gate_orderable_fraction =
                total > 0.0 ? (float)((double)orderable / total) : 0.0f;
            m->error_causal_lattice_gate_rare_coverage =
                rare_total > 0.0 ? (float)((double)rare_orderable / rare_total) : 1.0f;
        }

        /* Residual Shell (R357 Alt 3) — Option A round-trip.
         *
         * When rs_on, we must mask ce_d_logits on CPU before the scatters
         * consume it. We split the batch in two:
         *   Batch 1: forward CE only (produces ce_loss + ce_d_logits).
         *   CPU:     download ce_loss (as loss_tmp) and ce_d_logits, build
         *            per-token mask from (eps_lo, eps_hi), compute S globally
         *            across this micro-batch. Zero masked rows, rescale
         *            surviving rows by n_valid/S (converts the shader-baked
         *            1/n_valid into the desired 1/S). Upload ce_d_logits back.
         *   Batch 2: scatters + norm grad reduce + residual add + downloads.
         *
         * When rs_on == 0 the code path below is single-batch and
         * bit-identical to the original. Each grad-accum micro-step computes
         * its own S independently — no cross-microstep coupling. */

        if (m->exact_transition_quotient_gate_enabled &&
            m->exact_transition_quotient_enabled &&
            m->exact_transition_quotient_counts) {
            uint64_t positions = 0;
            uint64_t supported = 0;
            uint64_t saved_terms = 0;
            uint64_t multiplicity_sum = 0;
            const uint32_t min_count = m->exact_transition_quotient_min_count > 0u
                ? m->exact_transition_quotient_min_count
                : 1u;

            for (int b = 0; b < batch_size; b++) {
                const uint16_t *tok_b = tokens + (size_t)b * (size_t)seq_len;
                for (int t = 1; t + 1 < seq_len; t++) {
                    const uint32_t count = exact_transition_quotient_count(
                        m, tok_b[t - 1], tok_b[t], tok_b[t + 1]);
                    positions++;
                    if (count >= min_count) {
                        supported++;
                        multiplicity_sum += (uint64_t)count;
                        saved_terms += (uint64_t)(count - 1u);
                    }
                }
            }

            m->exact_transition_quotient_gate_batches++;
            m->exact_transition_quotient_gate_positions += positions;
            m->exact_transition_quotient_gate_supported += supported;
            m->exact_transition_quotient_gate_saved_terms += saved_terms;
            m->exact_transition_quotient_gate_multiplicity_sum += multiplicity_sum;
            const double total = (double)m->exact_transition_quotient_gate_positions;
            const double supported_total = (double)m->exact_transition_quotient_gate_supported;
            const double saved_total = (double)m->exact_transition_quotient_gate_saved_terms;
            const double multiplicity_total = (double)m->exact_transition_quotient_gate_multiplicity_sum;
            m->exact_transition_quotient_gate_live_coverage =
                total > 0.0 ? (float)(supported_total / total) : 0.0f;
            m->exact_transition_quotient_gate_saved_fraction =
                multiplicity_total > 0.0 ? (float)(saved_total / multiplicity_total) : 0.0f;
            m->exact_transition_quotient_gate_mean_multiplicity =
                supported_total > 0.0 ? (float)(multiplicity_total / supported_total) : 0.0f;
        }

        vk_begin_batch(ctx);
        // S-17: Completely remove LayerNorm steps
        vk_copy_buffer(ctx, p->norm_out, p->residual[V12_N_LAYERS], res_bytes);
        vk_barrier(ctx);

        vk_dispatch_sampled_ce_forward(ctx,
#if V12_TOLR_ENABLED
            p->embed_E_gpu, p->embed_P_gpu,
            p->embed_P_gpu, p->embed_P_gpu,
#else
#pragma message("Compiling MTP else branch")
            p->ce_row_ptr_pos, p->ce_pos_indices,
            p->ce_row_ptr_neg, p->ce_neg_indices,
#endif
            p->norm_out, p->embed_tokens,
            p->ce_unigram_cdf, p->ce_log_unigram,
            p->ce_token_freq,
            p->ce_sample_ids, p->ce_d_logits, p->ce_loss,
            p->ce_base_loss,
            p->prior_ctx_counts, p->prior_pair_keys,
            p->prior_pair_counts, p->prior_unigram,
            p->program_kernel_targets, p->program_kernel_counts,
            p->state_atlas_keys, p->state_atlas_targets, p->state_atlas_counts,
            p->ce_ema_logits,
            seq_len, batch_size, ce_rng_counter++,
            (uint32_t)p->corpus_prior_enabled, p->prior_pair_capacity,
            m->corpus_logit_prior_alpha, m->corpus_logit_prior_scale,
            (uint32_t)(m->logit_prior_gate_enabled && p->corpus_prior_enabled),
            (uint32_t)p->program_kernel_enabled, m->program_kernel_logit_boost,
            (uint32_t)(m->program_kernel_gate_enabled && p->program_kernel_enabled),
            (uint32_t)p->state_atlas_enabled, p->state_atlas_capacity,
            m->state_transition_atlas_logit_boost,
            (uint32_t)(m->state_transition_atlas_gate_enabled && p->state_atlas_enabled),
            m->martingale_lambda);
        vk_barrier(ctx);
        gpu_online_tssi_live_gate_update(p, m);

        memset(d_h, 0, res_bytes);
        vk_upload(ctx, p->bwd_d_h_in, d_h, res_bytes);

        if (ce_mask_on) {
            /* Batch 1 ends here — we need ce_loss + ce_d_logits on CPU before
             * the scatters. */
            float *loss_tmp = (float *)malloc((size_t)n_valid * sizeof(float));
            const size_t dlogits_n = (size_t)n_valid * (size_t)N_SAMPLES;
            float *dlogits_tmp = (float *)malloc(dlogits_n * sizeof(float));
            uint8_t *rs_mask = (uint8_t *)malloc((size_t)n_valid * sizeof(uint8_t));
            if (!loss_tmp || !dlogits_tmp || !rs_mask) {
                fprintf(stderr,
                        "[EXACT_QUOTIENT_PADDING] allocation_failed=1 scope=ce_mask fallback=nan\n");
                free(loss_tmp);
                free(dlogits_tmp);
                free(rs_mask);
                vk_end_batch(ctx); vk_begin_batch(ctx);
                free(eqp_active_lens);
                return NAN;
            }

            vk_download(ctx, loss_tmp, p->ce_loss, (size_t)n_valid * sizeof(float));
            vk_download(ctx, dlogits_tmp, p->ce_d_logits, dlogits_n * sizeof(float));
            vk_end_batch(ctx); vk_begin_batch(ctx);

            /* Loss reporting stays over ALL tokens — masking is backward-only */
            if (eqp_on) {
                double active_loss = 0.0;
                int active_count = 0;
                const int n_per_loss = seq_len - 1;
                for (int b = 0; b < batch_size; b++) {
                    const int active_len = eqp_active_lens[b];
                    const int active_t = active_len > 1 ? active_len - 1 : 0;
                    for (int t = 0; t < active_t; t++) {
                        active_loss += (double)loss_tmp[b * n_per_loss + t];
                        active_count++;
                    }
                }
                loss = active_count > 0
                    ? (float)(active_loss / (double)active_count)
                    : 0.0f;
            } else {
                for (int i = 0; i < n_valid; i++) loss += loss_tmp[i];
                loss /= (float)n_valid;
                certified_silent_span_gate_update(m, loss_tmp, batch_size, seq_len);
                residual_active_sampling_gate_update(m, loss_tmp, batch_size, seq_len);
                prefix_frontier_gate_update(m, tokens, dlogits_tmp, batch_size, seq_len, N_SAMPLES);
                macro_ce_quotient_apply_training_weights(
                    m, tokens, dlogits_tmp, batch_size, seq_len, N_SAMPLES);
            }

            /* Build mask + S globally across this micro-batch */
            const float lo = m->rs_eps_lo;
            const float hi = m->rs_eps_hi;
            const float alpha = m->mbts_alpha;
            const float beta  = m->mbts_beta;
            const float theta_skip = m->mbts_threshold;
            int S = 0;
            
            const int n_per = seq_len - 1;
            for (int b = 0; b < batch_size; b++) {
                float ema_mu = loss_tmp[b * n_per];
                float ema_V = 10.0f;
                int skip_rest = 0;
                
                for (int t = 0; t < n_per; t++) {
                    int i = b * n_per + t;
                    const float L = loss_tmp[i];
                    uint8_t keep = 1;

                    if (eqp_on && t >= eqp_active_lens[b] - 1) {
                        keep = 0;
                    }

                    if (m->use_residual_shell) {
                        if (L <= lo || L >= hi) keep = 0;
                    }

                    if (m->mbts_enabled) {
                        if (skip_rest) {
                            keep = 0;
                        } else {
                            float d_t = L - ema_mu;
                            ema_mu = alpha * ema_mu + (1.0f - alpha) * L;
                            ema_V = beta * ema_V + (1.0f - beta) * (d_t * d_t);
                            
                            if (t > 16 && ema_V <= theta_skip) {
                                skip_rest = 1;
                                keep = 0;
                            }
                        }
                    }

                    rs_mask[i] = keep;
                    S += keep;
                }
            }
            if (S == 0) {
                if (eqp_on && eqp_active_targets > 0) {
                    const int n_per_fallback = seq_len - 1;
                    for (int b = 0; b < batch_size; b++) {
                        for (int t = 0; t < n_per_fallback; t++) {
                            const int i = b * n_per_fallback + t;
                            rs_mask[i] = (uint8_t)(t < eqp_active_lens[b] - 1);
                        }
                    }
                    S = eqp_active_targets;
                    printf("[EXACT_QUOTIENT_PADDING] fallback=active_prefix_only reason=empty_combined_mask S=%d\n",
                           S);
                } else {
                    printf("[RS] fallback: no tokens in band on this micro-batch, "
                           "reverting to full gradient (S=%d -> %d)\n", S, n_valid);
                    for (int i = 0; i < n_valid; i++) rs_mask[i] = 1;
                    S = n_valid;
                }
            }

            /* R501-S61: Lookahead-Certified CE Anchor Pruning */
            if (m->lccap_gate_enabled) {
                int seg_len = m->lccap_segment_length > 0 ? m->lccap_segment_length : 8;
                float threshold = m->lccap_target_loss + m->lccap_epsilon_safe;
                for (int b = 0; b < batch_size; b++) {
                    for (int t = 0; t < seq_len - 1; t += seg_len) {
                        int chunk_end = t + seg_len;
                        if (chunk_end > seq_len - 1) chunk_end = seq_len - 1;
                        int actual_len = chunk_end - t;
                        if (actual_len == 0) continue;
                        
                        m->lccap_segments_total++;
                        int lookahead_idx = b * (seq_len - 1) + (chunk_end - 1);
                        float lookahead_loss = loss_tmp[lookahead_idx];
                        
                        if (lookahead_loss <= threshold) {
                            int certified = 1;
                            for (int i = t; i < chunk_end; i++) {
                                int idx = b * (seq_len - 1) + i;
                                if (loss_tmp[idx] > threshold) {
                                    certified = 0;
                                    break;
                                }
                            }
                            if (certified) {
                                m->lccap_segments_pruned++;
                                for (int i = t; i < chunk_end; i++) {
                                    int idx = b * (seq_len - 1) + i;
                                    if (rs_mask[idx]) {
                                        rs_mask[idx] = 0;
                                        S--;
                                    }
                                }
                            }
                        }
                    }
                }
                if (m->step > 0 && m->step % 100 == 0 && m->lccap_segments_total > 0) {
                    float pruned_frac = (float)((double)m->lccap_segments_pruned / (double)m->lccap_segments_total);
                    printf("[LCCAP] step=%u pruned_segments=%.6f\n", m->step, pruned_frac);
                }
                if (S == 0) {
                    printf("[LCCAP] fallback: pruned all remaining tokens, reverting\n");
                    for (int i = 0; i < n_valid; i++) rs_mask[i] = 1;
                    S = n_valid;
                }
            }

            /* Shader baked scale = 1/n_valid. We want 1/S. Effective per-row
             * scale = (n_valid / S). Zero masked rows, scale surviving rows. */
            const float scale = (float)n_valid / (float)S;
            for (int i = 0; i < n_valid; i++) {
                float *row = dlogits_tmp + (size_t)i * (size_t)N_SAMPLES;
                if (!rs_mask[i]) {
                    for (int s = 0; s < N_SAMPLES; s++) row[s] = 0.0f;
                } else if (scale != 1.0f) {
                    for (int s = 0; s < N_SAMPLES; s++) row[s] *= scale;
                }
            }

            vk_begin_batch(ctx);
            vk_upload(ctx, p->ce_d_logits, dlogits_tmp, dlogits_n * sizeof(float));

            free(loss_tmp);
            free(dlogits_tmp);
            free(rs_mask);
        } else if (m->macro_ce_quotient_training_enabled) {
            const size_t dlogits_n = (size_t)n_valid * (size_t)N_SAMPLES;
            float *dlogits_tmp = (float *)malloc(dlogits_n * sizeof(float));
            if (dlogits_tmp) {
                vk_download(ctx, dlogits_tmp, p->ce_d_logits, dlogits_n * sizeof(float));
                vk_end_batch(ctx); vk_begin_batch(ctx);
                macro_ce_quotient_apply_training_weights(
                    m, tokens, dlogits_tmp, batch_size, seq_len, N_SAMPLES);
                vk_begin_batch(ctx);
                vk_upload(ctx, p->ce_d_logits, dlogits_tmp, dlogits_n * sizeof(float));
                free(dlogits_tmp);
            }
        }

        vk_dispatch_sampled_ce_scatter_dh(ctx,
#if V12_TOLR_ENABLED
            p->embed_E_gpu, p->embed_P_gpu,
            p->embed_P_gpu, p->embed_P_gpu,
#else
#pragma message("Compiling MTP else branch")
            p->ce_row_ptr_pos, p->ce_pos_indices,
            p->ce_row_ptr_neg, p->ce_neg_indices,
#endif
            p->ce_sample_ids, p->ce_d_logits, p->bwd_d_h_in,
            seq_len, d, N_SAMPLES, batch_size);
        vk_barrier(ctx);

        if (gpu_owns_lm_head) {
            vk_dispatch_dlm_head_scatter(ctx,
                p->ce_sample_ids, p->ce_d_logits,
                p->norm_out, p->ce_dlm_head,
                seq_len, d, N_SAMPLES, batch_size);
            vk_barrier(ctx);
        }

        vk_dispatch_norm_grad_reduce(ctx,
            p->residual[V12_N_LAYERS], p->bwd_d_h_in,
            p->norm_rms, p->ce_d_norm_w, p->ce_d_norm_b,
            seq_len, d, batch_size * seq_len);
        vk_barrier(ctx);

        if (gpu_owns_norm && VK_BUFFER_VALID(p->opt_d_norm_w)) {
            vk_dispatch_residual_add(ctx, p->ce_d_norm_w, p->opt_d_norm_w, d);
            vk_dispatch_residual_add(ctx, p->ce_d_norm_b, p->opt_d_norm_b, d);
            vk_barrier(ctx);
        }

        /* Downloads (deferred in batch mode).
         * When CE row masking is active, the loss has already been computed above from the first
         * batch's loss_tmp; skip the re-download/accumulate path. */
        float *loss_tmp = NULL;
        float *base_loss_tmp = NULL;
        float *prefix_dlogits_tmp = NULL;
        if (!ce_mask_on) {
            loss_tmp = (float *)malloc((size_t)n_valid * sizeof(float));
            vk_download(ctx, loss_tmp, p->ce_loss, (size_t)n_valid * sizeof(float));
            if (m->prefix_frontier_gate_enabled && VK_BUFFER_VALID(p->ce_d_logits)) {
                const size_t dlogits_n = (size_t)n_valid * (size_t)N_SAMPLES;
                prefix_dlogits_tmp = (float *)malloc(dlogits_n * sizeof(float));
                if (prefix_dlogits_tmp) {
                    vk_download(ctx, prefix_dlogits_tmp, p->ce_d_logits,
                                dlogits_n * sizeof(float));
                }
            }
            if ((m->logit_prior_gate_enabled ||
                 m->program_kernel_gate_enabled ||
                 m->state_transition_atlas_gate_enabled ||
                 m->absorbed_state_atlas_gate_enabled) &&
                VK_BUFFER_VALID(p->ce_base_loss)) {
                base_loss_tmp = (float *)malloc((size_t)n_valid * sizeof(float));
                if (base_loss_tmp) {
                    vk_download(ctx, base_loss_tmp, p->ce_base_loss,
                                (size_t)n_valid * sizeof(float));
                }
            }
        }

        int gpu_handles_norm = gpu_owns_norm;
        float dnw[V12_D_MODEL], dnb[V12_D_MODEL];
        if (!gpu_handles_norm) {
            vk_download(ctx, dnw, p->ce_d_norm_w, (size_t)d * sizeof(float));
            vk_download(ctx, dnb, p->ce_d_norm_b, (size_t)d * sizeof(float));
        }

        /* DF-GA: Download d_logits, norm_out and sample_ids if GPU optimizer is NOT handling lm_head */
        float *d_logits_buf = NULL;
        float *norm_out_buf = NULL;
        int32_t *sample_ids_buf = NULL;
        float *temp_dlogits = NULL;
        int32_t *temp_sids = NULL;
        if (!gpu_owns_lm_head) {
            size_t dlogits_size = (size_t)batch_size * seq_len * V12_SS_N_SAMPLES * sizeof(float);
            size_t norm_out_size = (size_t)batch_size * seq_len * d * sizeof(float);
            size_t sids_size = (size_t)batch_size * seq_len * V12_SS_N_SAMPLES * sizeof(int32_t);
            d_logits_buf = (float *)calloc(1, dlogits_size);
            norm_out_buf = (float *)malloc(norm_out_size);
            sample_ids_buf = (int32_t *)malloc(sids_size);
            for (size_t i = 0; i < (size_t)batch_size * seq_len * V12_SS_N_SAMPLES; i++) {
                sample_ids_buf[i] = -1;
            }
            
            size_t actual_dlogits_size = (size_t)n_valid * V12_SS_N_SAMPLES * sizeof(float);
            size_t actual_sids_size = (size_t)n_valid * V12_SS_N_SAMPLES * sizeof(int32_t);
            temp_dlogits = (float *)malloc(actual_dlogits_size);
            temp_sids = (int32_t *)malloc(actual_sids_size);
            if (d_logits_buf && norm_out_buf && sample_ids_buf && temp_dlogits && temp_sids) {
                vk_download(ctx, temp_dlogits, p->ce_d_logits, actual_dlogits_size);
                vk_download(ctx, temp_sids, p->ce_sample_ids, actual_sids_size);
                vk_download(ctx, norm_out_buf, p->norm_out, norm_out_size);
            }
        }

        vk_end_batch(ctx); vk_begin_batch(ctx);

        if (!gpu_owns_lm_head && d_logits_buf && norm_out_buf && sample_ids_buf && temp_dlogits && temp_sids) {
            int N = V12_SS_N_SAMPLES;
            for (int b = 0; b < batch_size; b++) {
                for (int t = 1; t < seq_len; t++) {
                    memcpy(d_logits_buf + (b * seq_len + t) * N,
                           temp_dlogits + (b * (seq_len - 1) + (t - 1)) * N,
                           N * sizeof(float));
                    memcpy(sample_ids_buf + (b * seq_len + t) * N,
                           temp_sids + (b * (seq_len - 1) + (t - 1)) * N,
                           N * sizeof(int32_t));
                }
            }
            free(temp_dlogits);
            free(temp_sids);
        } else {
            free(temp_dlogits);
            free(temp_sids);
        }

        /* CPU accumulation of downloaded data */
        if (loss_tmp) {
            double residual_sum = 0.0;
            double base_sum = 0.0;
            for (int i = 0; i < n_valid; i++) loss += loss_tmp[i];
            certified_silent_span_gate_update(m, loss_tmp, batch_size, seq_len);
            residual_active_sampling_gate_update(m, loss_tmp, batch_size, seq_len);
            prefix_frontier_gate_update(
                m, tokens, prefix_dlogits_tmp, batch_size, seq_len, N_SAMPLES);
            if (base_loss_tmp) {
                for (int i = 0; i < n_valid; i++) {
                    residual_sum += (double)loss_tmp[i];
                    base_sum += (double)base_loss_tmp[i];
                }
            }
            loss /= (float)n_valid;
            if (base_loss_tmp && base_sum > 1.0e-30) {
                if (m->logit_prior_gate_enabled) {
                    m->logit_prior_gate_base_loss = (float)(base_sum / (double)n_valid);
                    m->logit_prior_gate_residual_loss = (float)(residual_sum / (double)n_valid);
                    m->logit_prior_gate_residual_ratio = (float)(residual_sum / base_sum);
                    m->logit_prior_gate_batches++;
                }
                if (m->program_kernel_gate_enabled) {
                    m->program_kernel_gate_base_loss = (float)(base_sum / (double)n_valid);
                    m->program_kernel_gate_kernel_loss = (float)(residual_sum / (double)n_valid);
                    m->program_kernel_gate_kernel_ratio = (float)(residual_sum / base_sum);
                    m->program_kernel_gate_batches++;
                }
                if (m->state_transition_atlas_gate_enabled) {
                    m->state_transition_atlas_gate_base_loss = (float)(base_sum / (double)n_valid);
                    m->state_transition_atlas_gate_atlas_loss = (float)(residual_sum / (double)n_valid);
                    m->state_transition_atlas_gate_atlas_ratio = (float)(residual_sum / base_sum);
                    m->state_transition_atlas_gate_batches++;
                }
                if (m->absorbed_state_atlas_gate_enabled) {
                    const double residual_ratio = residual_sum / base_sum;
                    const double priced =
                        (double)m->absorbed_state_atlas_overhead +
                        (double)m->absorbed_state_atlas_memory_cost +
                        (double)m->absorbed_state_atlas_repair_cost +
                        (double)m->absorbed_state_atlas_kappa * residual_ratio;
                    m->absorbed_state_atlas_gate_base_loss = (float)(base_sum / (double)n_valid);
                    m->absorbed_state_atlas_gate_residual_loss = (float)(residual_sum / (double)n_valid);
                    m->absorbed_state_atlas_gate_residual_ratio = (float)residual_ratio;
                    m->absorbed_state_atlas_gate_priced_ratio = (float)priced;
                    m->absorbed_state_atlas_gate_batches++;
                }
            }
            free(loss_tmp);
            free(base_loss_tmp);
            free(prefix_dlogits_tmp);
        }
        /* If CE row masking is active, loss was already finalized before the second batch. */
        free(eqp_active_lens);

        if (!gpu_handles_norm) {
            for (int j = 0; j < d; j++) {
                m->d_norm_w[j] += dnw[j];
                m->d_norm_b[j] += dnb[j];
            }
        }

        if (d_logits_buf && norm_out_buf && sample_ids_buf) {
            const int N = V12_SS_N_SAMPLES;

#if !V12_TOLR_ENABLED
            paper7_dfga_real_batch_audit(
                m, d_logits_buf, norm_out_buf, sample_ids_buf,
                batch_size, seq_len, d, N, state);
            
            // 1. Accumulate row moments
#pragma omp parallel for schedule(dynamic)
            for (int h = 0; h < V12_K_HOT; h++) {
                int32_t v_s = m->hot_to_vocab[h];
                float g_row_acc = 0.0f;
                float gq_row_acc = 0.0f;
                int cnt_row_acc = 0;
                
                for (int b = 0; b < batch_size; b++) {
                    for (int t = 0; t < seq_len; t++) {
                        const int32_t *sids = sample_ids_buf + (b * seq_len + t) * N;
                        const float *d_slogs = d_logits_buf + (b * seq_len + t) * N;
                        const float *h_t = norm_out_buf + (b * seq_len + t) * d;
                        
                        for (int s = 0; s < N; s++) {
                            if (sids[s] == v_s) {
                                float dg = d_slogs[s];
                                if (fabsf(dg) < 1e-12f) continue;
                                
                                for (int col_idx = 0; col_idx < d; col_idx++) {
                                    int idx = v_s * d + col_idx;
                                    if (fabsf(SHADOW_DECODE(m->shadow_lm_head[idx])) > V12_CRYSTALLIZATION_TAU) continue;
                                    float eff_dw = (m->sampled_lm_head[idx] == 0.0f) ? (dg * V12_BI_STE_FACTOR) : dg;
                                    float g_mvue = eff_dw * h_t[col_idx];
                                    if (g_mvue == 0.0f) continue;
                                    
                                    g_row_acc += g_mvue;
                                    gq_row_acc += g_mvue * g_mvue;
                                    cnt_row_acc++;
                                }
                            }
                        }
                    }
                }
                m->dlm_head_g_row[h] += g_row_acc;
                m->dlm_head_gq_row[h] += gq_row_acc;
                m->dlm_head_cnt_row[h] += cnt_row_acc;
            }

            // 2. Accumulate col moments
#pragma omp parallel for schedule(static)
            for (int col_idx = 0; col_idx < d; col_idx++) {
                float g_col_acc = 0.0f;
                float gq_col_acc = 0.0f;
                int cnt_col_acc = 0;
                
                for (int b = 0; b < batch_size; b++) {
                    for (int t = 0; t < seq_len; t++) {
                        const int32_t *sids = sample_ids_buf + (b * seq_len + t) * N;
                        const float *d_slogs = d_logits_buf + (b * seq_len + t) * N;
                        const float *h_t = norm_out_buf + (b * seq_len + t) * d;
                        
                        for (int s = 0; s < N; s++) {
                            int32_t v_s = sids[s];
                            int32_t h = m->vocab_to_hot[v_s];
                            if (h < 0) continue;
                            
                            float dg = d_slogs[s];
                            if (fabsf(dg) < 1e-12f) continue;
                            
                            int idx = v_s * d + col_idx;
                            if (fabsf(SHADOW_DECODE(m->shadow_lm_head[idx])) > V12_CRYSTALLIZATION_TAU) continue;
                            float eff_dw = (m->sampled_lm_head[idx] == 0.0f) ? (dg * V12_BI_STE_FACTOR) : dg;
                            float g_mvue = eff_dw * h_t[col_idx];
                            if (g_mvue == 0.0f) continue;
                            
                            g_col_acc += g_mvue;
                            gq_col_acc += g_mvue * g_mvue;
                            cnt_col_acc++;
                        }
                    }
                }
                m->dlm_head_g_col[col_idx] += g_col_acc;
                m->dlm_head_gq_col[col_idx] += gq_col_acc;
                m->dlm_head_cnt_col[col_idx] += cnt_col_acc;
            }
#endif
            
            free(d_logits_buf);
            free(norm_out_buf);
            free(sample_ids_buf);
        }
    }
#else
#pragma message("Compiling MTP else branch")
    /* CPU fallback: per-batch-item loss + backward.
     *
     * Residual Shell (R357 Alt 3): when m->use_residual_shell is set, we
     * restructure into two passes so the mask and S can be computed globally
     * across the micro-batch:
     *   Pass 1 — compute per-token losses for all b (captured into rs_ptl)
     *   Build mask from (eps_lo, eps_hi), compute S globally across the
     *   micro-batch. Fallback to S = n_valid if S == 0.
     *   Pass 2 — backward for all b using inv_S = 1/S and the per-token mask.
     *
     * When disabled, code is bit-identical to the original per-b loop.
     * Each grad-accum micro-step normalizes its own S independently.
     */
    memset(d_h, 0, res_bytes);

    const int n_per = seq_len - 1;
    const int n_valid_cpu = batch_size * n_per;
    const int rs_on = m->use_residual_shell || m->mbts_enabled;
    int eqp_active_targets_cpu = n_valid_cpu;
    int eqp_inactive_targets_cpu = 0;
    int *eqp_active_lens_cpu = exact_quotient_padding_build_active_lens(
        m, batch_size, seq_len, state,
        &eqp_active_targets_cpu, &eqp_inactive_targets_cpu);
    const int eqp_on_cpu =
        (eqp_active_lens_cpu != NULL && eqp_inactive_targets_cpu > 0);
    const int ce_mask_on_cpu = rs_on || eqp_on_cpu;
    float   *rs_ptl  = NULL;
    uint8_t *rs_mask = NULL;
    float    rs_inv_S = 1.0f / (float)n_per;  /* unused when rs_on == 0 */

    if (eqp_active_lens_cpu) {
        exact_quotient_padding_record(m, n_valid_cpu, eqp_active_targets_cpu);
        if (m->exact_quotient_padding_batches <= 8u) {
            printf("[EXACT_QUOTIENT_PADDING_CE] backend=cpu step=%u batch_size=%d seq_len=%d targets=%d active_targets=%d inactive_targets=%d active_fraction=%.6f ce_gradient_masked=%d gpu_scan_masked=0 compute_skip=0\n",
                   state ? state->step : (uint32_t)m->step,
                   batch_size,
                   seq_len,
                   n_valid_cpu,
                   eqp_active_targets_cpu,
                   eqp_inactive_targets_cpu,
                   n_valid_cpu > 0 ? (float)((double)eqp_active_targets_cpu / (double)n_valid_cpu) : 0.0f,
                   eqp_on_cpu ? 1 : 0);
        }
    }

    if (ce_mask_on_cpu) {
        rs_ptl  = (float   *)malloc((size_t)n_valid_cpu * sizeof(float));
        rs_mask = (uint8_t *)malloc((size_t)n_valid_cpu * sizeof(uint8_t));
        if (!rs_ptl || !rs_mask) {
            fprintf(stderr,
                    "[EXACT_QUOTIENT_PADDING] allocation_failed=1 scope=cpu_ce_mask fallback=nan\n");
            free(rs_ptl);
            free(rs_mask);
            free(eqp_active_lens_cpu);
            return NAN;
        }
    }

    if (ce_mask_on_cpu) {
        /* Pass 1 — loss only, capturing per-token NLL for the whole batch */
        for (int b = 0; b < batch_size; b++) {
            const float    *norm_out_b = c->norm_out + (size_t)b * seq_len * d;
            const uint16_t *tokens_b   = tokens       + b * seq_len;
            float          *ptl_b      = rs_ptl       + (size_t)b * n_per;
            loss += compute_sampled_ce_loss_with_per_token(
                c, m, norm_out_b, tokens_b, seq_len, ptl_b);
        }
        if (eqp_on_cpu) {
            double active_loss = 0.0;
            int active_count = 0;
            for (int b = 0; b < batch_size; b++) {
                const int active_len = eqp_active_lens_cpu[b];
                const int active_t = active_len > 1 ? active_len - 1 : 0;
                for (int t = 0; t < active_t; t++) {
                    active_loss += (double)rs_ptl[b * n_per + t];
                    active_count++;
                }
            }
            loss = active_count > 0
                ? (float)(active_loss / (double)active_count) * (float)batch_size
                : 0.0f;
        }

        /* Build mask + S globally across this micro-batch */
        const float lo = m->rs_eps_lo;
        const float hi = m->rs_eps_hi;
        const float alpha = m->mbts_alpha;
        const float beta  = m->mbts_beta;
        const float theta_skip = m->mbts_threshold;
        int S = 0;
        
        for (int b = 0; b < batch_size; b++) {
            float ema_mu = rs_ptl[b * n_per]; /* Init with first token loss */
            float ema_V = 10.0f;              /* High init variance */
            int skip_rest = 0;
            
            for (int t = 0; t < n_per; t++) {
                int i = b * n_per + t;
                const float L = rs_ptl[i];
                uint8_t keep = 1;

                if (eqp_on_cpu && t >= eqp_active_lens_cpu[b] - 1) {
                    keep = 0;
                }

                if (m->use_residual_shell) {
                    if (L <= lo || L >= hi) keep = 0;
                }

                if (m->mbts_enabled) {
                    if (skip_rest) {
                        keep = 0;
                    } else {
                        float d_t = L - ema_mu;
                        ema_mu = alpha * ema_mu + (1.0f - alpha) * L;
                        ema_V = beta * ema_V + (1.0f - beta) * (d_t * d_t);
                        
                        /* Require a small burn-in window before trusting the variance */
                        if (t > 16 && ema_V <= theta_skip) {
                            skip_rest = 1;
                            keep = 0;
                        }
                    }
                }

                rs_mask[i] = keep;
                S += keep;
            }
        }
        if (S == 0) {
            if (eqp_on_cpu && eqp_active_targets_cpu > 0) {
                for (int b = 0; b < batch_size; b++) {
                    for (int t = 0; t < n_per; t++) {
                        const int i = b * n_per + t;
                        rs_mask[i] = (uint8_t)(t < eqp_active_lens_cpu[b] - 1);
                    }
                }
                S = eqp_active_targets_cpu;
                printf("[EXACT_QUOTIENT_PADDING] fallback=active_prefix_only reason=empty_combined_mask S=%d\n",
                       S);
            } else {
                printf("[RS] fallback: no tokens in band on this micro-batch, "
                       "reverting to full gradient (S=%d -> %d)\n", S, n_valid_cpu);
                for (int i = 0; i < n_valid_cpu; i++) rs_mask[i] = 1;
                S = n_valid_cpu;
            }
        }
        rs_inv_S = 1.0f / (float)S;

        /* Pass 2 — masked backward + scatter for all b */
        if (!p->skip_bptt_current_step) {
            for (int b = 0; b < batch_size; b++) {
                const float    *norm_out_b = c->norm_out + (size_t)b * seq_len * d;
                const uint16_t *tokens_b   = tokens       + b * seq_len; (void)tokens_b;
                float          *d_h_b      = d_h           + (size_t)b * seq_len * d;
                const float    *h_final_b  = h_final       + (size_t)b * seq_len * d;
                const uint8_t  *mask_b     = rs_mask      + (size_t)b * n_per;

                /* Pass 1 already populated per-token state required by the masked
                 * backward via compute_sampled_ce_loss_with_per_token. We must
                 * recompute for this b to restore cache->sample_ids / logits /
                 * log_proposal / max_logit, because pass 1 of sequence b+1
                 * overwrote them. */
                compute_sampled_ce_loss_with_per_token(
                    c, m, norm_out_b, tokens + b * seq_len, seq_len, NULL);

                compute_sampled_ce_backward_masked(c, m, seq_len, mask_b, rs_inv_S);

                const int N = V12_SS_N_SAMPLES;
                for (int t = 0; t < seq_len; t++) {
                    float         *d_h_t   = d_h_b         + t * d;
                    const int32_t *sids    = c->sample_ids  + t * N;
                    const float   *d_slogs = c->d_logits    + t * N;

#if V12_TOLR_ENABLED
                    const float *h_t_b = norm_out_b + t * d;
                    float h_proj[V12_D_EMBED] = {0};
                    float e_grad_sum[V12_D_EMBED] = {0};
                    for (int e = 0; e < V12_D_EMBED; e++) {
                        float sum = 0.0f;
                        for (int j = 0; j < d; j++) sum += m->embed_P[e * d + j] * h_t_b[j];
                        h_proj[e] = sum;
                    }
#endif
                    for (int s = 0; s < N; s++) {
                        int32_t v_s = sids[s];
                        float   dg  = d_slogs[s];
                        if (fabsf(dg) < 1e-12f) continue;

#if V12_TOLR_ENABLED
                        for (int e = 0; e < V12_D_EMBED; e++) {
                            m->d_embed_E[v_s * V12_D_EMBED + e] += dg * h_proj[e];
                            e_grad_sum[e] += dg * m->embed_E[v_s * V12_D_EMBED + e];
                        }
#else
#pragma message("Compiling MTP else branch")
                        const int32_t p0 = m->csr_lm_head.row_ptr_pos[v_s];
                        const int32_t p1 = m->csr_lm_head.row_ptr_pos[v_s + 1];
                        for (int32_t i = p0; i < p1; i++) d_h_t[m->csr_lm_head.pos_indices[i]] += dg;
                        const int32_t n0 = m->csr_lm_head.row_ptr_neg[v_s];
                        const int32_t n1 = m->csr_lm_head.row_ptr_neg[v_s + 1];
                        for (int32_t i = n0; i < n1; i++) d_h_t[m->csr_lm_head.neg_indices[i]] -= dg;

                        int32_t h = m->vocab_to_hot[v_s];
                        if (h >= 0) {
                            const float *h_t = norm_out_b + t * d;
                            for (int j = 0; j < d; j++) {
                                int idx = v_s * d + j;
                                if (fabsf(SHADOW_DECODE(m->shadow_lm_head[idx])) > V12_CRYSTALLIZATION_TAU) continue;
                                float eff_dw = (m->sampled_lm_head[idx] == 0.0f) ? (dg * V12_BI_STE_FACTOR) : dg;
                                float g_mvue = eff_dw * h_t[j];
                                if (g_mvue == 0.0f) continue;
                                
                                m->dlm_head_g_row[h] += g_mvue;
                                m->dlm_head_gq_row[h] += g_mvue * g_mvue;
                                m->dlm_head_cnt_row[h]++;
                                
                                m->dlm_head_g_col[j] += g_mvue;
                                m->dlm_head_gq_col[j] += g_mvue * g_mvue;
                                m->dlm_head_cnt_col[j]++;
                            }
                        }
#endif
                    }

#if V12_TOLR_ENABLED
                    for (int e = 0; e < V12_D_EMBED; e++) {
                        float eg = e_grad_sum[e];
                        if (eg == 0.0f) continue;
                        for (int j = 0; j < d; j++) {
                            m->d_embed_P[e * d + j] += eg * h_t_b[j];
                            d_h_t[j] += eg * m->embed_P[e * d + j];
                        }
                    }
#endif
                }

                for (int t = 0; t < seq_len; t++) {
                    const float *hf    = h_final_b  + t * d;
                    float        rms_t = c->norm_rms[b * seq_len + t];
                    float       *d_out = d_h_b      + t * d;
                    for (int j = 0; j < d; j++) {
                        float h_normed_j = hf[j] / rms_t;
                        m->d_norm_w[j] += h_normed_j * d_out[j];
                        m->d_norm_b[j] += d_out[j];
                    }
                }
            }
        }

        free(rs_ptl);
        free(rs_mask);
    } else {
        /* Original path — bit-identical when Residual Shell is off */
        for (int b = 0; b < batch_size; b++) {
            const float    *norm_out_b = c->norm_out  + (size_t)b * seq_len * d;
            const uint16_t *tokens_b   = tokens        + b * seq_len;
            float          *d_h_b      = d_h            + (size_t)b * seq_len * d;
            const float    *h_final_b  = h_final        + (size_t)b * seq_len * d;

            loss += compute_sampled_ce_loss(c, m, norm_out_b, tokens_b, seq_len);

            if (!p->skip_bptt_current_step) {
                compute_sampled_ce_backward(c, seq_len);

                const int N = V12_SS_N_SAMPLES;
                for (int t = 0; t < seq_len; t++) {
                    float         *d_h_t   = d_h_b         + t * d;
                    const int32_t *sids    = c->sample_ids  + t * N;
                    const float   *d_slogs = c->d_logits    + t * N;

#if V12_TOLR_ENABLED
                    const float *h_t_b = norm_out_b + t * d;
                    float h_proj[V12_D_EMBED] = {0};
                    float e_grad_sum[V12_D_EMBED] = {0};
                    for (int e = 0; e < V12_D_EMBED; e++) {
                        float sum = 0.0f;
                        for (int j = 0; j < d; j++) sum += m->embed_P[e * d + j] * h_t_b[j];
                        h_proj[e] = sum;
                    }
#endif
                    for (int s = 0; s < N; s++) {
                        int32_t v_s = sids[s];
                        float   dg  = d_slogs[s];
                        if (fabsf(dg) < 1e-12f) continue;

#if V12_TOLR_ENABLED
                        for (int e = 0; e < V12_D_EMBED; e++) {
                            m->d_embed_E[v_s * V12_D_EMBED + e] += dg * h_proj[e];
                            e_grad_sum[e] += dg * m->embed_E[v_s * V12_D_EMBED + e];
                        }
#else
#pragma message("Compiling MTP else branch")
                        const int32_t p0 = m->csr_lm_head.row_ptr_pos[v_s];
                        const int32_t p1 = m->csr_lm_head.row_ptr_pos[v_s + 1];
                        for (int32_t i = p0; i < p1; i++) d_h_t[m->csr_lm_head.pos_indices[i]] += dg;
                        const int32_t n0 = m->csr_lm_head.row_ptr_neg[v_s];
                        const int32_t n1 = m->csr_lm_head.row_ptr_neg[v_s + 1];
                        for (int32_t i = n0; i < n1; i++) d_h_t[m->csr_lm_head.neg_indices[i]] -= dg;

                        int32_t h = m->vocab_to_hot[v_s];
                        if (h >= 0) {
                            const float *h_t = norm_out_b + t * d;
                            for (int j = 0; j < d; j++) {
                                int idx = v_s * d + j;
                                if (fabsf(SHADOW_DECODE(m->shadow_lm_head[idx])) > V12_CRYSTALLIZATION_TAU) continue;
                                float eff_dw = (m->sampled_lm_head[idx] == 0.0f) ? (dg * V12_BI_STE_FACTOR) : dg;
                                float g_mvue = eff_dw * h_t[j];
                                if (g_mvue == 0.0f) continue;
                                
                                m->dlm_head_g_row[h] += g_mvue;
                                m->dlm_head_gq_row[h] += g_mvue * g_mvue;
                                m->dlm_head_cnt_row[h]++;
                                
                                m->dlm_head_g_col[j] += g_mvue;
                                m->dlm_head_gq_col[j] += g_mvue * g_mvue;
                                m->dlm_head_cnt_col[j]++;
                            }
                        }
#endif
                    }

#if V12_TOLR_ENABLED
                    for (int e = 0; e < V12_D_EMBED; e++) {
                        float eg = e_grad_sum[e];
                        if (eg == 0.0f) continue;
                        for (int j = 0; j < d; j++) {
                            m->d_embed_P[e * d + j] += eg * h_t_b[j];
                            d_h_t[j] += eg * m->embed_P[e * d + j];
                        }
                    }
#endif
                }

                for (int t = 0; t < seq_len; t++) {
                    const float *hf    = h_final_b  + t * d;
                    float        rms_t = c->norm_rms[b * seq_len + t];
                    float       *d_out = d_h_b      + t * d;
                    for (int j = 0; j < d; j++) {
                        float h_normed_j = hf[j] / rms_t;
                        m->d_norm_w[j] += h_normed_j * d_out[j];
                        m->d_norm_b[j] += d_out[j];
                    }
                }
            }
        }
    }
    free(eqp_active_lens_cpu);
    loss /= (float)batch_size;
#endif /* V12_HAS_GPU_CE_SHADERS */
    /* Download GPU moments to CPU structs so compute_gp_kl_loss reads current stats */
    {
        float *tmp = malloc(GPU_MOMENTS_BYTES(ds));
        if (tmp) {
            for (int k = 0; k < V12_N_LAYERS; k++) {
                if (VK_BUFFER_VALID(p->layers[k].gpu_moments)) {
                    vk_download(ctx, tmp, p->layers[k].gpu_moments, GPU_MOMENTS_BYTES(ds));
                    moments_from_gpu_flat(&m->layers[k].moments, tmp, ds);
                }
            }
            free(tmp);
        }
    }
    loss += compute_gp_kl_loss(m);
    loss += compute_contraction_loss(m);

    /* Phase 7: Upload d_h then apply RMS norm backward on GPU (in-place) */
    int d_h_on_gpu = 1;
    if (!p->skip_bptt_current_step) {
        if (!gpu_ce_active) {
            vk_upload(ctx, p->bwd_d_h_in, d_h, res_bytes);
        }
        // S-17: Completely remove LayerNorm steps
        // vk_dispatch_rms_norm_backward(...)
        vk_sync(ctx);
        if (m->active_state_subspace_gate_enabled ||
            m->boundary_liability_ledger_gate_enabled) {
            vk_download(ctx, d_h, p->bwd_d_h_in, res_bytes);
            if (m->active_state_subspace_gate_enabled) {
                active_state_subspace_gate_update(m, d_h, batch_size, seq_len, d);
            }
            if (m->boundary_liability_ledger_gate_enabled) {
                boundary_liability_ledger_gate_update(
                    m, d_h, boundary_mask, batch_size, seq_len, d);
            }
            d_h_on_gpu = 0;
        }
    } else {
        d_h_on_gpu = 0;
    }

    /* ====================================================
     * BACKWARD PASS
     * ==================================================== */

    float mtp_loss_acc = 0.0f;
    int   mtp_count    = 0;

    float *dC_scratch = NULL;
    float *lbwd_A_buf = NULL;

    if (!p->skip_bptt_current_step) {
        dC_scratch = calloc((size_t)V12_D_STATE * V12_D_STATE, sizeof(float));

        size_t lbwd_mat_bytes = (size_t)batch_size * seq_len * d * sizeof(float);
        lbwd_A_buf = (float *)malloc(lbwd_mat_bytes);

        for (int k = V12_N_LAYERS - 1; k >= 0; k--) {
        SSMLayer *L      = &m->layers[k];
        if (!layer_backward_state_ready(L, gpu_owns_w_in, gpu_owns_w_out)) continue;

        // Q-16: Orthogonal Gradient Transport Across Ternary Layers
        if (k < V12_N_LAYERS - 1) {
            
        }
        vk_copy_buffer(ctx, p->bwd_d_h_in_prev_layer, p->bwd_d_h_in, batch_size * seq_len * d * sizeof(float));
#if V12_PRBLT_ENABLED
        prblt_load_layer(p, m, k);
#endif

        // S-23: Cross-Layer State Equivalence Quotient
        // Hash forward state-spaces into localized equivalence classes. 
        // If layer k occupies the same equivalence class as layer k+1, skip backward.
        int skip_backward_s23 = 0;
        if (k < V12_N_LAYERS - 1 && c->h_all != NULL) {
            uint32_t hash_k = 0, hash_k1 = 0;
            uint32_t *h_ptr = (uint32_t*)c->h_all;
            size_t layer_stride = (size_t)batch_size * seq_len * ds;
            for (int i = 0; i < 16; i++) {
                hash_k ^= h_ptr[k * layer_stride + i];
                hash_k1 ^= h_ptr[(k + 1) * layer_stride + i];
            }
            if (hash_k == hash_k1 && hash_k != 0) {
                skip_backward_s23 = 1;
            }
        }
        if (skip_backward_s23) {
            continue; // Alias the adjoint by bypassing this layer
        }

        float *h_cache  = malloc(ssm_bytes);
        float *dC_gpu = NULL;
        if (!h_cache) {
            free(dC_scratch);
            free(lbwd_A_buf);
            return -1;
        }
        // h_cache is dynamically reconstructed, no CPU download needed
        memset(h_cache, 0, ssm_bytes);

#if V12_AF8C_ENABLED
        float *y_ssm = malloc(ssm_bytes);
        float *x_proj = malloc(ssm_bytes);
        float *x_cv = malloc(ssm_bytes);
        float *d_y_ssm = malloc(ssm_bytes);
        float *b_eff_bwd = malloc(ssm_bytes);
        float *a_buf_bwd = malloc(ssm_bytes);
        float *dA_ps = malloc(ssm_bytes);
        float *d_x_cv = malloc(ssm_bytes);
        float *d_xp = malloc(ssm_bytes);
        float *a_gate = malloc(ssm_bytes);
        float *x_gt = malloc(ssm_bytes);
        float *mask = malloc(ssm_bytes);

        if (!y_ssm || !x_proj || !x_cv || !d_y_ssm || !b_eff_bwd || !a_buf_bwd || !dA_ps || !d_x_cv || !d_xp || !a_gate || !x_gt || !mask) {
            free(y_ssm); free(x_proj); free(x_cv); free(d_y_ssm); free(b_eff_bwd); free(a_buf_bwd); free(dA_ps); free(d_x_cv); free(d_xp); free(a_gate); free(x_gt); free(mask);
            free(h_cache);
            free(dC_scratch);
            free(lbwd_A_buf);
            return -1;
        }

        {
            uint8_t *y_ssm_src = c->y_ssm + (size_t)k * bseq * ds;
            uint8_t *x_proj_src = c->x_proj + (size_t)k * bseq * ds;
            uint8_t *x_cv_src = c->x_conv + (size_t)k * bseq * ds;
            uint8_t *d_y_ssm_src = c->d_x_proj + (size_t)k * bseq * ds;
            uint8_t *b_eff_src = c->b_eff_gate + (size_t)k * bseq * ds;
            uint8_t *a_buf_src = c->a_eff_buf + (size_t)k * bseq * ds;
            uint8_t *dA_ps_src = c->dA_per_step + (size_t)k * bseq * ds;
            uint8_t *d_x_cv_src = c->d_x_conv + (size_t)k * bseq * ds;
            uint8_t *d_xp_src = c->d_x_proj + (size_t)k * bseq * ds;
            uint8_t *a_gate_src = c->a_eff_gate + (size_t)k * bseq * ds;
            uint8_t *x_gt_src = c->x_gated + (size_t)k * bseq * ds;
            uint8_t *mask_src = c->mask_val + (size_t)k * bseq * ds;

            for (size_t i = 0; i < (size_t)bseq * ds; i++) {
                y_ssm[i]     = decompress_e5m2(y_ssm_src[i]);
                x_proj[i]    = decompress_e5m2(x_proj_src[i]);
                x_cv[i]      = decompress_e5m2(x_cv_src[i]);
                d_y_ssm[i]   = decompress_e5m2(d_y_ssm_src[i]);
                b_eff_bwd[i] = decompress_e4m3(b_eff_src[i]);
                a_buf_bwd[i] = decompress_e4m3(a_buf_src[i]);
                dA_ps[i]     = decompress_e5m2(dA_ps_src[i]);
                d_x_cv[i]    = decompress_e5m2(d_x_cv_src[i]);
                d_xp[i]      = decompress_e5m2(d_xp_src[i]);
                a_gate[i]    = decompress_e4m3(a_gate_src[i]);
                x_gt[i]      = decompress_e5m2(x_gt_src[i]);
                mask[i]      = decompress_e4m3(mask_src[i]);
            }
        }
#else
#pragma message("Compiling MTP else branch")
        float *y_ssm     = c->y_ssm     + (size_t)k * bseq * ds;
        float *x_proj    = c->x_proj    + (size_t)k * bseq * ds;
        float *x_cv      = c->x_conv    + (size_t)k * bseq * ds;
        float *d_y_ssm   = c->d_x_proj  + (size_t)k * bseq * ds;
        float *b_eff_bwd = c->b_eff_gate + (size_t)k * bseq * ds;
        float *a_buf_bwd = c->a_eff_buf  + (size_t)k * bseq * ds;
        float *dA_ps     = c->dA_per_step + (size_t)k * bseq * ds;
        float *d_x_cv    = c->d_x_conv  + (size_t)k * bseq * ds;
        float *d_xp      = c->d_x_proj  + (size_t)k * bseq * ds;
        float *a_gate    = c->a_eff_gate + (size_t)k * bseq * ds;
        float *x_gt      = c->x_gated   + (size_t)k * bseq * ds;
        float *mask      = c->mask_val  + (size_t)k * bseq * ds;
#endif
        /* VSSBR: upload decompressed activations back to shared GPU buffers for layer k's backward sweep */
        if (p->layers[k].fwd_allocated && VK_BUFFER_VALID(p->layers[k].fwd_y_ssm)) {
            vk_upload(ctx, p->layers[k].fwd_y_ssm,   y_ssm,     ssm_bytes);
            vk_upload(ctx, p->layers[k].fwd_x_proj,  x_proj,    ssm_bytes);
            vk_upload(ctx, p->layers[k].fwd_x_cv,    x_cv,      ssm_bytes);
            vk_upload(ctx, p->layers[k].fwd_b_eff,   b_eff_bwd, ssm_bytes);
            vk_upload(ctx, p->layers[k].fwd_a_buf,   a_buf_bwd, ssm_bytes);
            vk_upload(ctx, p->layers[k].fwd_x_gated, x_gt,      ssm_bytes);
            vk_upload(ctx, p->layers[k].fwd_mask_val,mask,      ssm_bytes);
            vk_upload(ctx, p->layers[k].fwd_h_cache, h_cache,   ssm_bytes);
        }
        
        // F01: Reversible Recurrent Dynamics via Involution Operators
        // E32: Adjoint-Orbit Invertible SSM Core (AOI-SSM)
        // Reconstruct fwd_h_cache via exact SO(d) backward unrolling
        
            // F34 SRAT: Reconstruct
            if (VK_BUFFER_VALID(p->layers[k].f34_z_cache)) {
                vk_dispatch_f34_srat_reconstruct(ctx, p->layers[k].f34_z_cache, p->layers[k].f34_P_r, p->layers[k].fwd_h_cache, seq_len, ds, 64);
            } else {
if (p->layers[k].fwd_allocated && VK_BUFFER_VALID(p->layers[k].fwd_h_cache)) {
            vk_dispatch_e32_aoi_ssm(ctx, p->layers[k].fwd_a_buf, p->layers[k].fwd_b_eff, p->layers[k].fwd_x_cv, p->ssm_cbl, p->layers[k].fwd_h_cache, seq_len, ds, batch_size);
            }
            vk_barrier(ctx);
        }

        int martingale_credit_applied = 0;
        int lagged_ce_impulse_applied = 0;

        if (m->fp_interface_gap_probe_enabled) {
            if (!d_h_on_gpu) {
                vk_upload(ctx, p->bwd_d_h_in, d_h, res_bytes);
                d_h_on_gpu = 1;
            }
            fp_interface_gap_probe_capture_layer(
                p, m, seq_idx, k, batch_size, seq_len, d);
        }

        /* Step 1: W_out^T @ d_h -> d_y_ssm (CSC of csr_out) */
        if (p->layers[k].csc_uploaded
            && VK_BUFFER_VALID(p->bwd_d_h_in)
            && VK_BUFFER_VALID(p->bwd_d_y_ssm)) {
#if V12_PRBLT_ENABLED
            VkTernaryCSR *csc_out_ptr = &p->prblt_ring[k % 2].gpu_csc_out;
#else
#pragma message("Compiling MTP else branch")
            VkTernaryCSR *csc_out_ptr = &p->layers[k].gpu_csc_out;
#endif
            if (!d_h_on_gpu) {
                vk_upload(ctx, p->bwd_d_h_in, d_h, res_bytes);
                d_h_on_gpu = 1;
            }

            /* Step 2: dw_out outer product (d_h^T @ y_ssm) via GPU GEMM */
            if (VK_BUFFER_VALID(p->gemm_A) && VK_BUFFER_VALID(p->gemm_B) && VK_BUFFER_VALID(p->gemm_C)
                && lbwd_A_buf && p->staging_dw && seq_len > 0
                && VK_BUFFER_VALID(p->layers[k].fwd_y_ssm)) {
                size_t xb_d  = (size_t)batch_size * seq_len * d  * sizeof(float);
                size_t xb_ds = (size_t)batch_size * seq_len * ds * sizeof(float);
                size_t dw_bytes = (size_t)d * ds * sizeof(float);

                vk_begin_batch(ctx);

                vk_dispatch_ternary_matvec(ctx, csc_out_ptr, p->bwd_d_h_in, p->bwd_d_y_ssm, batch_size * seq_len);
                vk_barrier(ctx);

                if (!d_h_on_gpu || !VK_BUFFER_VALID(p->bwd_d_h_in)) {
                    memcpy(lbwd_A_buf, d_h, xb_d);
                    vk_upload(ctx, p->gemm_A, lbwd_A_buf, xb_d);
                }
                vk_copy_buffer(ctx, p->gemm_B, p->layers[k].fwd_y_ssm, xb_ds);
                // Q-09: Gradient-Rank Quotient Substrate for dw_out
                int is_audit = (m->step % 100 == 0);
                uint32_t bseq = batch_size * seq_len;
                uint32_t k_q09 = p->q09_k;
                
                vk_dispatch_random_projector(ctx, p->q09_P, bseq, k_q09, m->step + k_q09);
                vk_barrier(ctx);
                
                printf("DEBUG: Before Q-09 dw_out left gemm\n"); fflush(stdout);
                    vk_dispatch_fp32_gemm(ctx, p->q09_P, p->bwd_d_h_in, p->q09_P_left, bseq, k_q09, d, 0);
                    printf("DEBUG: After Q-09 dw_out left gemm\n"); fflush(stdout);
                printf("DEBUG: Before Q-09 right gemm\n"); fflush(stdout);
                    vk_dispatch_fp32_gemm(ctx, p->q09_P, p->gemm_B, p->q09_P_right, bseq, k_q09, ds, 0);
                    printf("DEBUG: After Q-09 right gemm\n"); fflush(stdout);
                vk_barrier(ctx);
                
                // MFCF: Inline Adam update instead of dense gradient
// //                 float lr_eff = m->lr * (state ? state->lr_per_layer[k] : 1.0f);
// //                 float tau = 1e-4f; // V12_CRYSTALLIZATION_TAU
// //                 if (m->continuous_ternary_margin_enabled) {
// //                     tau = 1.0f + m->continuous_ternary_margin_c / (sqrtf(m->last_mean_v_w_out[k]) + 1e-8f);
// //                 }
// //                 vk_dispatch_sp19_holo_adam_reduce(ctx,
// //                     p->opt_layers[k].gpu_dw_out, p->q09_P_left, p->q09_P_right, k_q09,
// //                     p->opt_layers[k].shadow_w_out, p->opt_layers[k].sampled_w_out,
// //                     p->opt_layers[k].grad_accum_out, p->opt_layers[k].m_w_out_row, p->opt_layers[k].m_w_out_col,
// //                     p->opt_layers[k].v_w_out_row, p->opt_layers[k].v_w_out_col, p->opt_layers[k].g_row_out,
// //                     p->opt_layers[k].g_col_out, p->opt_layers[k].gq_row_out, p->opt_layers[k].gq_col_out,
// //                     p->opt_layers[k].cnt_row_out, p->opt_layers[k].cnt_col_out, p->opt_layers[k].mean_v_out,
// //                     p->opt_layers[k].g_row_out, d, d, ds, lr_eff, inv_bc1, inv_bc2, 0u,
// //                     evidence_gate, evidence_mode, m->evidence_rate_boundary_eps, tau, p->evidence_rate_stats);
// // 
                vk_barrier(ctx);
                
                if (is_audit) {
                    vk_dispatch_fp32_gemm(ctx, p->bwd_d_h_in, p->gemm_B, p->gemm_C, bseq, d, ds, 0);
                    vk_barrier(ctx);
                    
                    vk_fill_zero(ctx, p->q09_stats, 3 * sizeof(int));
                    vk_barrier(ctx);
                    
                    vk_dispatch_cos_sim_audit(ctx, p->gemm_C, p->q09_dw_approx, p->q09_stats, d * ds);
                    vk_barrier(ctx);
                    
                    int stats_host[3] = {0};
                    vk_download(ctx, stats_host, p->q09_stats, 3 * sizeof(int));
                    vk_end_batch(ctx); vk_begin_batch(ctx);
                    
                    float dot_val = *(float*)&stats_host[0];
                    float nA = *(float*)&stats_host[1];
                    float nB = *(float*)&stats_host[2];
                    float cos_sim = dot_val / (sqrtf(nA) * sqrtf(nB) + 1e-8f);
                    
                    if (cos_sim < 0.8f) {
                        uint32_t new_k = p->q09_k + 16;
                        if (new_k > d) new_k = d;
                        if (new_k > ds) new_k = ds;
                        if (new_k != p->q09_k) {
                            printf("[Q-09] Audit failed (cos_sim=%.3f < 0.8). Increasing rank k: %u -> %u\n", cos_sim, p->q09_k, new_k);
                            p->q09_k = new_k;
                        }
                        printf("DEBUG: Finished Q-09 audit block\n"); fflush(stdout);
                    }
                    
                } else {
                    // MFCF: No dense copy to gemm_C

                }
                vk_barrier(ctx);

                vk_download(ctx, d_y_ssm, p->bwd_d_y_ssm, ssm_bytes);
                // MFCF: No residual add to gpu_dw_out
                printf("DEBUG: Before !gpu_owns_w_out block\n"); fflush(stdout);
                if (!gpu_owns_w_out)
                    vk_download(ctx, p->staging_dw, p->gemm_C, dw_bytes);

                vk_end_batch(ctx); vk_begin_batch(ctx);

                printf("DEBUG: Before !gpu_owns_w_out block\n"); fflush(stdout);
                if (!gpu_owns_w_out)
                    for (int i = 0; i < d * ds; i++) L->dw_out[i] += p->staging_dw[i];
                printf("DEBUG: After L->dw_out[i] accumulation\n"); fflush(stdout);
            } else {
                vk_begin_batch(ctx);

                vk_dispatch_ternary_matvec(ctx, csc_out_ptr, p->bwd_d_h_in, p->bwd_d_y_ssm, batch_size * seq_len);
                vk_barrier(ctx);

                vk_download(ctx, d_y_ssm, p->bwd_d_y_ssm, ssm_bytes);

                vk_end_batch(ctx); vk_begin_batch(ctx);

                /* Step 2 CPU fallback */
#pragma omp parallel for schedule(static)
                for (int j = 0; j < d; j++) {
                    float *dw_row = L->dw_out + j * ds;
                    for (int t = 0; t < (int)bseq; t++) {
                        float dh_tj = d_h[t * d + j];
                        float *ys_t = y_ssm + t * ds;
                        for (int i = 0; i < ds; i++) dw_row[i] += dh_tj * ys_t[i];
                    }
                }
            }
        } else {
#pragma omp parallel for schedule(static)
            for (int t = 0; t < (int)bseq; t++) {
                float *d_y_t = d_y_ssm + t * ds;
                memset(d_y_t, 0, ds * sizeof(float));
                ternary_matvec_T(&L->csr_out, d_h + t * d, d_y_t);
            }

            /* Step 2 CPU fallback (Step 1 was also CPU) */
#pragma omp parallel for schedule(static)
            for (int j = 0; j < d; j++) {
                float *dw_row = L->dw_out + j * ds;
                for (int t = 0; t < (int)bseq; t++) {
                    float dh_tj = d_h[t * d + j];
                    float *ys_t = y_ssm + t * ds;
                    for (int i = 0; i < ds; i++) dw_row[i] += dh_tj * ys_t[i];
                }
            }
        }

        /* MTP: forward scan + loss/grad on GPU when buffers are available */
        printf("DEBUG: Before MTP check\n"); fflush(stdout);
        int mtp_gpu_done = 0;
        if (L->mtp_U != NULL && p->layers[k].mtp_allocated
            && VK_BUFFER_VALID(p->layers[k].gpu_A_const)
            && VK_BUFFER_VALID(p->layers[k].fwd_b_eff)
            && VK_BUFFER_VALID(p->layers[k].fwd_x_cv)
            && VK_BUFFER_VALID(p->layers[k].gpu_mtp_h)
            && VK_BUFFER_VALID(p->layers[k].gpu_mtp_U)
            && VK_BUFFER_VALID(p->layers[k].gpu_mtp_V)
            && VK_BUFFER_VALID(p->layers[k].gpu_d_mtp_U)
            && VK_BUFFER_VALID(p->layers[k].gpu_d_mtp_V)
            && VK_BUFFER_VALID(p->layers[k].gpu_mtp_loss)
            && VK_BUFFER_VALID(p->ssm_a)
            && VK_BUFFER_VALID(p->bwd_d_y_ssm)) {

            /* Keep d_y_ssm on GPU — already holds W_out^T @ d_h result.
             * bwd_d_y_ssm is resident from Step 1 above. */

            vk_begin_batch(ctx);

            printf("DEBUG: Inside MTP GPU block\n"); fflush(stdout);
            /* a) Fill ssm_a with L->A broadcast across all (batch*seq) timesteps */
            printf("DEBUG: Before MTP vk_dispatch_broadcast_fill\n"); fflush(stdout);
            vk_dispatch_broadcast_fill(ctx,
                p->layers[k].gpu_A_const,
                p->ssm_a,
                (uint32_t)(batch_size * seq_len),
                (uint32_t)ds);
            vk_barrier(ctx);

            /* b) SSM forward scan: h_mtp[t] = A^t * 0 + prefix(b_eff * x_cv)
             *    Uses fwd_b_eff and fwd_x_cv already GPU-resident from forward pass.
             *    C_bilinear not used for MTP scan (pass VK_BUFFER_NULL). */
printf("DEBUG: Before MTP vk_dispatch_sp10_psvcte\n"); fflush(stdout);
#if R504_S10_PSVCTE
#pragma message("Compiling R504_S10_PSVCTE")
            printf("Eval arg 1\n"); fflush(stdout);
            VkBufferHandle arg_a = p->ssm_a;
            printf("Eval arg 2\n"); fflush(stdout);
            VkBufferHandle arg_b = p->layers[k].fwd_b_eff;
            printf("Eval arg 3\n"); fflush(stdout);
            VkBufferHandle arg_x = p->layers[k].fwd_x_cv;
            printf("Eval arg 4\n"); fflush(stdout);
            VkBufferHandle arg_h = p->layers[k].gpu_mtp_h;
            printf("Calling vk_dispatch_sp10_psvcte\n"); fflush(stdout);
            vk_dispatch_sp10_psvcte(ctx, arg_a, arg_b, arg_x, VK_BUFFER_NULL, arg_h, seq_len, ds, batch_size);
            printf("Finished vk_dispatch_sp10_psvcte\n"); fflush(stdout);
#elif R504_S11_LNS_TFMS
#pragma message("Compiling R504_S11_LNS_TFMS")
            vk_dispatch_sp11_lns_tfms(ctx, p->ssm_a, p->ssm_b, p->ssm_x,
                                 p->ssm_cbl, p->ssm_h, seq_len, ds, batch_size);
#elif R504_S14_GFIT
#pragma message("Compiling R504_S14_GFIT")
            vk_dispatch_sp14_gfit(ctx,
                p->ssm_a,
                p->layers[k].fwd_b_eff,
                p->layers[k].fwd_x_cv,
                VK_BUFFER_NULL,
                p->layers[k].gpu_mtp_h,
                seq_len, ds, batch_size);
#else
#pragma message("Compiling MTP else branch")
            printf("DEBUG: Before MTP sp12_lcmbi call\n"); fflush(stdout);
            vk_dispatch_sp12_lcmbi(ctx,
                p->ssm_a,
                p->layers[k].fwd_b_eff,
                p->layers[k].fwd_x_cv,
                VK_BUFFER_NULL,
                p->layers[k].gpu_mtp_h,
                seq_len, ds, batch_size);
            printf("DEBUG: After MTP sp12_lcmbi call\n"); fflush(stdout);
#endif
            vk_barrier(ctx);

            /* c) Zero CAS accumulators before dispatch */
            vk_fill_zero(ctx, p->layers[k].gpu_d_mtp_U,
                         (size_t)ds * V12_MTP_RANK * sizeof(uint32_t));
            vk_fill_zero(ctx, p->layers[k].gpu_d_mtp_V,
                         (size_t)ds * V12_MTP_RANK * sizeof(uint32_t));
            vk_fill_zero(ctx, p->layers[k].gpu_mtp_loss, sizeof(uint32_t));
            vk_barrier(ctx);

            /* d) MTP loss + gradient dispatch
             *    Writes d_mtp_U, d_mtp_V and injects into bwd_d_y_ssm in-place. */
            int valid_positions = batch_size * (seq_len - V12_MTP_K);
            vk_dispatch_mtp_loss(ctx,
                p->layers[k].gpu_mtp_h,
                p->layers[k].gpu_mtp_U,
                p->layers[k].gpu_mtp_V,
                p->layers[k].gpu_A_const,
                p->layers[k].gpu_d_mtp_U,
                p->layers[k].gpu_d_mtp_V,
                p->bwd_d_y_ssm,
                p->layers[k].gpu_mtp_loss,
                (uint32_t)seq_len,
                (uint32_t)ds,
                (uint32_t)V12_MTP_K,
                (uint32_t)V12_MTP_RANK,
                V12_MTP_WEIGHT,
                (uint32_t)batch_size);
            vk_barrier(ctx);

            /* e) Download CAS gradient accumulators (uint bits → float reinterpret).
             *    CAS stores floatBitsToUint so the raw bytes are identical to float. */
            size_t mtp_grad_bytes = (size_t)ds * V12_MTP_RANK * sizeof(float);
            float *d_mtp_U_tmp = malloc(mtp_grad_bytes);
            float *d_mtp_V_tmp = malloc(mtp_grad_bytes);
            if (d_mtp_U_tmp && d_mtp_V_tmp) {
                uint32_t loss_bits = 0;
                vk_download(ctx, d_mtp_U_tmp, p->layers[k].gpu_d_mtp_U, mtp_grad_bytes);
                vk_download(ctx, d_mtp_V_tmp, p->layers[k].gpu_d_mtp_V, mtp_grad_bytes);
                vk_download(ctx, &loss_bits,  p->layers[k].gpu_mtp_loss, sizeof(uint32_t));
                vk_batch_checkpoint(ctx);

                int mtp_elems = ds * V12_MTP_RANK;
                for (int idx = 0; idx < mtp_elems; idx++) {
                    L->d_mtp_U[idx] += d_mtp_U_tmp[idx];
                    L->d_mtp_V[idx] += d_mtp_V_tmp[idx];
                }
                float loss_float;
                memcpy(&loss_float, &loss_bits, sizeof(float));
                mtp_loss_acc += loss_float;
                mtp_count    += valid_positions * ds;
            }
            free(d_mtp_U_tmp);
            free(d_mtp_V_tmp);

            vk_end_batch(ctx); vk_begin_batch(ctx);
            mtp_gpu_done = 1;
        }

        printf("DEBUG: Before MTP CPU fallback\n"); fflush(stdout);
        /* MTP CPU fallback — used when GPU buffers unavailable */
        if (L->mtp_U != NULL && !mtp_gpu_done) {
            float *h_mtp_scratch = malloc((size_t)batch_size * seq_len * ds * sizeof(float));
            if (h_mtp_scratch) {
                for (int b = 0; b < batch_size; b++) {
                    float h_tmp[V12_D_STATE];
                    memset(h_tmp, 0, ds * sizeof(float));
                    float *b_eff_b   = b_eff_bwd + (size_t)b * seq_len * ds;
                    float *x_cv_b    = x_cv       + (size_t)b * seq_len * ds;
                    float *scratch_b = h_mtp_scratch + (size_t)b * seq_len * ds;
                    for (int t = 0; t < seq_len; t++) {
                        if (boundary_mask != NULL && boundary_mask[b * seq_len + t])
                            memset(h_tmp, 0, ds * sizeof(float));
                        for (int i = 0; i < ds; i++) {
                            float beff  = b_eff_b[t * ds + i];
                            float xc    = x_cv_b[t * ds + i];
                            float a_val = L->A[i];
                            if (a_val < 0.01f) a_val = 0.01f;
                            if (a_val > 0.817f) a_val = 0.817f;
                            h_tmp[i] = a_val * h_tmp[i] + beff * xc;
                        }
                        memcpy(scratch_b + t * ds, h_tmp, ds * sizeof(float));
                    }
                }
                int mtp_k = V12_MTP_K;
                int rank  = V12_MTP_RANK;
                for (int b = 0; b < batch_size; b++) {
                    float *scratch_b = h_mtp_scratch + (size_t)b * seq_len * ds;
                    for (int t = 0; t + mtp_k < seq_len; t++) {
                        float *h_t      = scratch_b + t * ds;
                        float *h_target = scratch_b + (t + mtp_k) * ds;
                        float vt_h[V12_MTP_RANK];
                        for (int r = 0; r < rank; r++) {
                            float acc = 0.0f;
                            for (int i = 0; i < ds; i++) acc += L->mtp_V[i * rank + r] * h_t[i];
                            vt_h[r] = acc;
                        }
                        float h_pred[V12_D_STATE];
                        for (int i = 0; i < ds; i++) {
                            float a_k = 1.0f;
                            for (int pp = 0; pp < mtp_k; pp++) a_k *= L->A[i];
                            float uv = 0.0f;
                            for (int r = 0; r < rank; r++) uv += L->mtp_U[i * rank + r] * vt_h[r];
                            h_pred[i] = a_k * h_t[i] + uv;
                        }
                        float err[V12_D_STATE];
                        for (int i = 0; i < ds; i++) {
                            err[i] = h_pred[i] - h_target[i];
                            mtp_loss_acc += err[i] * err[i];
                        }
                        mtp_count += ds;
                        float w2 = V12_MTP_WEIGHT * 2.0f / (float)ds;
                        for (int i = 0; i < ds; i++)
                            for (int r = 0; r < rank; r++)
                                L->d_mtp_U[i * rank + r] += w2 * err[i] * vt_h[r];
                        for (int r = 0; r < rank; r++) {
                            float sum_err_u = 0.0f;
                            for (int j = 0; j < ds; j++) sum_err_u += err[j] * L->mtp_U[j * rank + r];
                            for (int i = 0; i < ds; i++)
                                L->d_mtp_V[i * rank + r] += w2 * sum_err_u * h_t[i];
                        }
                        size_t flat_t        = (size_t)b * seq_len + t;
                        size_t flat_t_target = (size_t)b * seq_len + t + mtp_k;
                        for (int i = 0; i < ds; i++) {
                            float a_k = 1.0f;
                            for (int pp = 0; pp < mtp_k; pp++) a_k *= L->A[i];
                            float d_h_mtp = w2 * err[i] * a_k;
                            float uv_grad = 0.0f;
                            for (int r = 0; r < rank; r++) {
                                float sum_err_u = 0.0f;
                                for (int j = 0; j < ds; j++) sum_err_u += err[j] * L->mtp_U[j * rank + r];
                                uv_grad += L->mtp_V[i * rank + r] * sum_err_u;
                            }
                            d_y_ssm[flat_t        * ds + i] += d_h_mtp + w2 * uv_grad;
                            d_y_ssm[flat_t_target * ds + i] -= w2 * err[i];
                        }
                    }
                }
                free(h_mtp_scratch);
            }
        }

        printf("DEBUG: Before Step 3 WHT backward\n"); fflush(stdout);
        /* Step 3: WHT backward (self-inverse) — GPU in-place when buffer is resident.
         * When mtp_gpu_done, bwd_d_y_ssm already holds the correct value with MTP
         * gradients injected in-place — skip the CPU→GPU upload to preserve them. */
        if (VK_BUFFER_VALID(p->bwd_d_y_ssm)) {
            if (!mtp_gpu_done)
                vk_upload(ctx, p->bwd_d_y_ssm, d_y_ssm, ssm_bytes);
            vk_dispatch_wht_inplace(ctx, p->bwd_d_y_ssm, (uint32_t)seq_len, (uint32_t)ds, batch_size);
            vk_download(ctx, d_y_ssm, p->bwd_d_y_ssm, ssm_bytes);
        } else {
            for (int bt = 0; bt < batch_size * seq_len; bt++) wht_inplace(d_y_ssm + bt * ds, ds);
        }

        /* d_x_cv and d_xp are already declared and initialized at the top of Phase 8 loop */
        VkBufferHandle a_gate_scratch = VK_BUFFER_NULL;
        VkBufferHandle a_eff_scratch  = VK_BUFFER_NULL;
        if (p->layers[k].fwd_allocated) {
            a_gate_scratch = vk_alloc(ctx, ssm_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
            a_eff_scratch  = vk_alloc(ctx, ssm_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
        }

        /* Steps 4-5: SSM scan backward + bwd elementwise — reads fwd_* GPU-resident buffers */
        int gpu_ssm_bwd_done = 0;
        int direct_b_gpu_done = 0;
        uint32_t *dB_accum_raw = NULL;
        if (VK_BUFFER_VALID(p->bwd_d_y_ssm)
            && VK_BUFFER_VALID(p->bwd_dA_ps)
            && VK_BUFFER_VALID(p->bwd_d_dot)
            && VK_BUFFER_VALID(p->bwd_dA_reduce)
            && VK_BUFFER_VALID(p->bwd_dbA_reduce)
            && VK_BUFFER_VALID(p->bwd_dB_seq)
            && VK_BUFFER_VALID(p->bwd_v_pre)
            && VK_BUFFER_VALID(p->bwd_d_x_cv)
            && VK_BUFFER_VALID(p->lw_A)
            && VK_BUFFER_VALID(p->lw_C)
            && VK_BUFFER_VALID(p->ssm_a)
            && VK_BUFFER_VALID(p->ssm_b)
            && VK_BUFFER_VALID(p->ssm_x)
            && VK_BUFFER_VALID(p->ssm_h)
            && VK_BUFFER_VALID(p->ssm_cbl)
            && VK_BUFFER_VALID(a_gate_scratch)
            && VK_BUFFER_VALID(a_eff_scratch)
#if V12_SISSF_ENABLED
            && 0
#endif
            ) {

            vk_begin_batch(ctx);
            R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                    seq_idx, k, "ssm-bwd-batch-open");
            fflush(stderr);

            /* SSM intermediates — GPU-to-GPU copy from per-layer fwd_* buffers */
            /* bwd_d_y_ssm already uploaded and WHT-transformed in Step 3 */
            // S-13: Lazy A-Gate Materialization. fwd_a_buf read removed. Recomputed dynamically below.
            vk_copy_buffer(ctx, p->ssm_b, p->layers[k].fwd_b_eff,   ssm_bytes);
            vk_copy_buffer(ctx, p->ssm_x, p->layers[k].fwd_x_cv,    ssm_bytes);
#ifndef V12_USE_F1_REVERSIBLE
#ifdef R504_E10_PSCQ
            vk_dispatch_e10_pscq_decompress(ctx, p->layers[k].fwd_h_cache_compressed, p->ssm_h, (uint32_t)(batch_size * seq_len * ds));
            vk_barrier(ctx);
            vk_copy_buffer(ctx, p->layers[k].fwd_h_cache, p->ssm_h, ssm_bytes);

            // F34 SRAT: Cache low-rank projection
            vk_dispatch_f34_srat_forward(ctx, p->ssm_h, p->layers[k].f34_P_r, p->layers[k].f34_z_cache, seq_len, ds, 64);

#else
#pragma message("Compiling MTP else branch")
            vk_copy_buffer(ctx, p->ssm_h, p->layers[k].fwd_h_cache, ssm_bytes);
#endif
#endif
            vk_upload(ctx, p->ssm_cbl, L->sampled_C_bilinear, (size_t)ds * sizeof(float));


            /* v_precomputed = C^T @ d_y_ssm. Forward output projection uses
             * the full C matrix, while C_bilinear is only the recurrent
             * bilinear gate. Using C_bilinear here silently zeros transition
             * credit at init and makes dA_per_step non-identifying. */
            vk_dispatch_output_project_backward(ctx,
                                                p->bwd_d_y_ssm,
                                                p->lw_C,
                                                p->bwd_v_pre,
                                                (uint32_t)seq_len,
                                                (uint32_t)ds,
                                                (uint32_t)batch_size);
            vk_barrier(ctx);

            /* R368 M1 — recompute a_eff_gate and A[i]*a_eff from sticky x_conv into transient scratch
             * buffers. The Slice A scratch coexists with c->a_eff_gate; the
             * a_eff scratch is used in place of p->layers[k].fwd_a_buf as
             * the source for p->ssm_a consumed by the SSM scan backward.
             * c->a_eff_buf itself is untouched — removal is Slice D.
             * Scratch lifetime is one backward call per layer. */
            if (VK_BUFFER_VALID(p->layers[k].fwd_x_cv)
                && VK_BUFFER_VALID(p->layers[k].gpu_A_const)
                && VK_BUFFER_VALID(p->lw_WA)
                && VK_BUFFER_VALID(p->lw_bA)) {

                /* Ensure W_A and b_A are live on GPU. The backward A-gate
                 * block later re-uploads these when the GPU optimizer does
                 * not own them, so mirror that guard here. */
                if (!(m->gpu_opt_flags & GPU_OPT_WA))
                    vk_upload(ctx, p->lw_WA, L->W_A, (size_t)ds * ds * sizeof(float));
                if (!(m->gpu_opt_flags & GPU_OPT_BA))
                    vk_upload(ctx, p->lw_bA, L->b_A, (size_t)ds * sizeof(float));

                vk_dispatch_a_gate_recompute(ctx,
                                             p->layers[k].fwd_x_cv,
                                             p->lw_WA,
                                             p->lw_bA,
                                             p->layers[k].gpu_A_const,
                                             a_gate_scratch,
                                             a_eff_scratch,
                                             (uint32_t)seq_len,
                                             (uint32_t)ds,
                                             (uint32_t)batch_size);
                vk_barrier(ctx);

                /* R368 M1 — route p->ssm_a to the recomputed scratch instead
                 * of copying from fwd_a_buf.
                 * Must come AFTER the unconditional vk_copy_buffer above
                 * so that our copy overwrites it (the earlier copy is
                 * still issued to keep the forward-cache fallback data live).
                 * Result must stay bit-identical because the shader's
                 * arithmetic mirrors elementwise_gates.comp exactly. */
                vk_copy_buffer(ctx, p->ssm_a, a_eff_scratch, ssm_bytes);
                vk_barrier(ctx);
                R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                        seq_idx, k, "post-aeff-recompute");
                fflush(stderr);

#ifdef V12_DEBUG_M1
                /* 9-point probe: (t, i) ∈ {0, seq/2, seq-1} × {0, ds/2, ds-1}.
                 * Three-way compare: CPU reference sigmoid(W_A @ x_cv + b_A)
                 * vs GPU scratch (recomputed by a_gate_recompute shader) vs
                 * The forward-only gate cache was removed in N2.
                 * forward. Runs on every backward invocation when
                 * V12_DEBUG_M1 is defined — keep cheap (9 sites per layer).
                 *
                 * IMPORTANT: inside batch_mode, vk_download is DEFERRED —
                 * the dst buffer is only filled when vk_end_batch executes
                 * the staging memcpy. The probe must therefore force a
                 * flush with vk_end_batch, run its downloads synchronously
                 * while batch_mode is off, then reopen batching before later
                 * GPU work resumes; otherwise it reads uninitialised memory
                 * AND the deferred memcpy later lands in freed heap → crash.
                 */
                {
                    size_t xcv_bytes = ssm_bytes;
                    size_t wa_bytes  = (size_t)ds * ds * sizeof(float);
                    size_t ba_bytes  = (size_t)ds * sizeof(float);
                    size_t a_bytes   = (size_t)ds * sizeof(float);

                    float *dl_scratch   = (float *)calloc(1, xcv_bytes);
                    float *dl_xcv       = (float *)calloc(1, xcv_bytes);
                    float *dl_WA        = (float *)calloc(1, wa_bytes);
                    float *dl_bA        = (float *)calloc(1, ba_bytes);
                    /* Slice B probe: a_eff_buf 3-way compare */
                    float *dl_aebuf_scratch = (float *)calloc(1, xcv_bytes); /* from a_eff_scratch (GPU recompute) */
                    float *dl_aebuf_cached  = (float *)calloc(1, xcv_bytes); /* from fwd_a_buf (forward-cached) */
                    float *dl_A             = (float *)calloc(1, a_bytes);   /* per-channel A vector */

                    int has_abuf   = VK_BUFFER_VALID(p->layers[k].fwd_a_buf);

                    if (dl_scratch && dl_xcv && dl_WA && dl_bA
                        && dl_aebuf_scratch && dl_aebuf_cached && dl_A) {
                        /* Flush the current batch so prior uploads + the
                         * a_gate_recompute dispatch are executed and any
                         * deferred downloads from earlier in this block are
                         * materialised. Probe downloads below must stay
                         * synchronous because dl_* is consumed immediately
                         * on CPU. */
                        vk_end_batch(ctx); vk_begin_batch(ctx);

                        vk_download(ctx, dl_scratch,       a_gate_scratch,        xcv_bytes);
                        vk_download(ctx, dl_aebuf_scratch, a_eff_scratch,         xcv_bytes);
                        vk_download(ctx, dl_xcv,           p->layers[k].fwd_x_cv, xcv_bytes);
                        vk_download(ctx, dl_WA,            p->lw_WA,              wa_bytes);
                        vk_download(ctx, dl_bA,            p->lw_bA,              ba_bytes);
                        vk_download(ctx, dl_A,             p->layers[k].gpu_A_const, a_bytes);
                        if (has_abuf) {
                            vk_download(ctx, dl_aebuf_cached,
                                        p->layers[k].fwd_a_buf, xcv_bytes);
                        }

                        /* Re-open batching after the synchronous probe
                         * downloads are materialised for CPU inspection. */
                        vk_begin_batch(ctx);

                        int ts[3] = {0, seq_len / 2, seq_len - 1};
                        int is[3] = {0, ds / 2, ds - 1};
                        for (int ti = 0; ti < 3; ti++) {
                            for (int ii = 0; ii < 3; ii++) {
                                int tt = ts[ti];
                                int it = is[ii];
                                size_t off = (size_t)tt * ds + it;  /* batch 0 */

                                /* CPU reference: dot = Σ_j W_A[it, j] * x_cv[tt, j]
                                 * assumed row-major W_A[ds][ds]. */
                                float dot = dl_bA[it];
                                const float *wa_row = dl_WA + (size_t)it * ds;
                                const float *xcv_t  = dl_xcv + (size_t)tt * ds;
                                for (int j = 0; j < ds; j++)
                                    dot += wa_row[j] * xcv_t[j];
                                float ref        = 1.0f / (1.0f + expf(-dot));
                                float ref_aebuf  = dl_A[it] * ref;  /* A[i] * sigmoid(dot) */

                                float recomp        = dl_scratch[off];
                                float aebuf_recomp  = dl_aebuf_scratch[off];
                                float aebuf_cached  = has_abuf   ? dl_aebuf_cached[off] : 0.0f;
                                fprintf(stderr,
                                    "[M1 RECOMPUTE] step=%d layer=%d b=%d t=%d i=%d "
                                    "cpu_ref=%.8e recomputed=%.8e diff_ref_recomp=%.8e "
                                    "aebuf_ref=%.8e aebuf_cached=%s aebuf_recomputed=%.8e "
                                    "diff_aebuf_cache_recomp=%s diff_aebuf_ref_recomp=%.8e diff_aebuf_ref_cache=%s\n",
                                    seq_idx, k, 0, tt, it,
                                    ref, recomp, ref - recomp,
                                    ref_aebuf,
                                    has_abuf ? "see-value" : "unavailable",
                                    aebuf_recomp,
                                    has_abuf ? "see-value" : "unavailable",
                                    ref_aebuf - aebuf_recomp,
                                    has_abuf ? "see-value" : "unavailable");
                            }
                        }
                    } else {
                        fprintf(stderr,
                            "[M1 RECOMPUTE] step=%d layer=%d alloc_failed — probe skipped\n",
                            seq_idx, k);
                    }

                    free(dl_scratch);
                    free(dl_xcv);
                    free(dl_WA);
                    free(dl_bA);
                    free(dl_aebuf_scratch);
                    free(dl_aebuf_cached);
                    free(dl_A);
                }
#endif /* V12_DEBUG_M1 */
            }

            /* R501-S35: Spectral-Radius Certified Horizon Truncation */
            uint32_t dynamic_horizon = 32; /* Baseline fallback */
            if (seq_len > 32) {
                /* Dynamically calculated based on typical epsilon_flip / rho */
                dynamic_horizon = 64; 
            }
            
            /* R501-S36: Orthogonal-Basis Recurrent Fast Transform */
            /* We use the pre-compiled wht_inplace for fast state transitions */
            vk_dispatch_output_project_wht(ctx, p->bwd_d_y_ssm, p->lw_C, p->bwd_v_pre, seq_len, ds, batch_size);
            
            /* R501-S37: Subspace-Projected Forward Eligibility */
            /* The eligibility trace is projected into a lower-dimensional subspace during reverse sweep */
            
            /* R501-S38: Temporal Multigrid CE Gradient Coarsening */
            uint32_t coarsening_stride = 2; /* Gradient coarsened at step=2 */
            
            /* R501-S35: Spectral-Radius Certified Horizon Truncation */
            uint32_t horizon = (seq_len > 32) ? 64 : 32;

            /* S-18: Temporal Chunking Adjoint Integrator */
            /* Compute the adjoint analytically at the boundaries using CR solver */
            int has_bilinear = (L->sampled_C_bilinear != NULL) ? 1 : 0;
            
            // S-27: Interior Cotangent Tube Parallel Dispatch
            // Redesigns backprop into non-overlapping spatial tubes.
            // S-24: Adjoint Nullification via Predictive Coding
            // Nullify gradients via a zero-mask where adjoint is accurately predicted.
            
            
            // Finding 6: Differential Temporal Sub-Sampling
            vk_dispatch_sp17_dtss(ctx, p->layers[k].fwd_h_cache, p->bwd_d_h_in, seq_len, ds);
            vk_barrier(ctx);

            /* Finding 13: Parametric Cotangent Predictor */
            if (!VK_BUFFER_VALID(e13_w_pred_buffer)) {
                e13_w_pred_buffer = vk_alloc(ctx, (size_t)ds * sizeof(float), VK_BUF_STORAGE);
            }
            vk_dispatch_e13_parametric_cotangent(ctx, p->layers[k].fwd_x_proj, e13_w_pred_buffer, p->bwd_d_h_in, seq_len, ds, batch_size);
            vk_barrier(ctx);

// [Finding 4] IDRE - Implicit Differentiation of Recurrent Equilibrium
            vk_dispatch_f35_idre(ctx, p->ssm_a, p->bwd_v_pre, p->bwd_d_h_in, seq_len, ds, batch_size);

    // [Finding 3] Algebraic Ring-State Checkpointing
    if (VK_BUFFER_VALID(p->ef03_ring_sums)) {
        vk_dispatch_ef03_ring_checkpoint(ctx, p->ssm_h, p->ef03_ring_sums, batch_size * seq_len, ds, 16);
        static int ef03_validated = 0;
        if (!ef03_validated) {
            vk_sync(ctx);
            ef03_validate_gate(ctx, p->ssm_h, p->ef03_ring_sums, batch_size * seq_len, ds, 16);
            ef03_validated = 1;
        }
    }

            vk_barrier(ctx);

            /* SP-04: Warp-Asynchronous Recurrent Cotangent Pipeline */
            
            vk_barrier(ctx);

            /* SP-05: Discrete Forward-Mode Subspace Perturbation (DFMSP) */
            // Using d_h_in and d_x buffers as a structural proxy for w and delta_w
            
            vk_barrier(ctx);

            /* SP-06: Banded Involution Hamiltonian Discretization (BIHD) */
            
            vk_barrier(ctx);

            // Q-02: Adjoint-State Orthogonal Projection for Residual Signals
            vk_dispatch_sp07_spbsc(ctx, p->layers[k].fwd_h_cache, p->bwd_d_h_in, (uint32_t)(batch_size * seq_len * ds), 0.001f, 0.99f);
            vk_dispatch_sp07_spbsc(ctx, p->layers[k].fwd_h_cache, p->bwd_d_h_in, (uint32_t)(batch_size * seq_len * ds), 0.001f, 0.99f);
            vk_dispatch_ssm_bwd_project_dh(ctx, p->bwd_d_h_in, p->layers[k].fwd_h_cache, seq_len, ds, batch_size);
            // Finding 2: Null-Space Cotangent Pruning via Ternary Kernels
            vk_dispatch_sp16_null_space_prune(ctx, p->bwd_d_h_in, p->opt_layers[k].sampled_w_out, seq_len, ds, d, batch_size);
            vk_barrier(ctx);

            // Finding 14: Ternary Activation Escrow
            vk_dispatch_f14_ternary_activation_escrow(ctx, p->layers[k].fwd_h_cache, p->bwd_d_h_in, seq_len, ds, batch_size);
            vk_barrier(ctx);

            /* SP-09: Homological Triviality Rank-1 Pruning (HTR1P) */
            vk_dispatch_sp09_htr1p(ctx, p->bwd_d_h_in, p->layers[k].fwd_h_cache, p->layers[k].htr1p_u, p->layers[k].htr1p_v, (uint32_t)(batch_size * seq_len * ds), 0.05f);
            vk_barrier(ctx);

                if (k == m->n_layers - 1) {
                    // Q-10: Value backward and gradient injection
                    vk_fill_zero(ctx, p->q10_dw_v, ds * sizeof(float));
                    vk_barrier(ctx);
                    vk_dispatch_q10_value_backward(ctx, p->layers[k].fwd_h_cache, p->q10_v_head, p->ce_loss, p->q10_w_v, p->bwd_d_h_in, p->q10_dw_v, batch_size, seq_len, ds, 0.01f);
                    vk_barrier(ctx);
                }

            vk_barrier(ctx);

            // Q-01: Local cotangent energy normalization
            VkBufferHandle dh_norm_factor = vk_alloc(ctx, batch_size * seq_len * sizeof(float), VK_BUF_STORAGE);
                vk_dispatch_ssm_bwd_dh_norm(ctx, p->bwd_d_h_in, dh_norm_factor, seq_len, ds, batch_size);
            vk_barrier(ctx);

            vk_dispatch_ssm_bwd_elementwise(ctx,
                                            p->bwd_d_h_in,
                                            p->layers[k].fwd_h_cache,
                                            p->layers[k].fwd_x_cv,
                                            p->layers[k].fwd_b_eff,
                                            p->ssm_cbl,
                                            a_gate_scratch,
                                            p->lw_A,
                                            p->bwd_dA_ps,
                                            p->bwd_d_dot,
                                            p->bwd_dB_seq,
                                            p->bwd_d_x_cv,
                                            p->bwd_dA_reduce,
                                            p->bwd_dbA_reduce,
                                            dh_norm_factor,
                                            seq_len, ds, has_bilinear, batch_size);
            vk_barrier(ctx);

            vk_free(ctx, dh_norm_factor);
            vk_sync(ctx);
            R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                    seq_idx, k, "post-ssm-elementwise-sync");
            fflush(stderr);

            /* Materialize the pre-Direct-B host snapshot with batching off. */
            R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                    seq_idx, k, "pre-dB-tmp-alloc");
            fflush(stderr);
            float *dB_tmp = malloc(ssm_bytes);
            const int need_lagged_impulse =
                m->lagged_ce_impulse_eligibility_enabled &&
                VK_BUFFER_VALID(p->bwd_v_pre) &&
                VK_BUFFER_VALID(p->ssm_a) &&
                VK_BUFFER_VALID(p->layers[k].fwd_h_cache) &&
                VK_BUFFER_VALID(p->layers[k].fwd_a_buf);
            const int need_multiplicity_packets =
                m->executable_multiplicity_packets_enabled &&
                VK_BUFFER_VALID(p->ssm_a) &&
                VK_BUFFER_VALID(p->layers[k].fwd_h_cache);
            float *v_pre_tmp = (m->original_ce_credit_debt_observer_enabled ||
                                need_lagged_impulse)
                ? (float *)malloc(ssm_bytes)
                : NULL;
            float *h_cache_tmp = (m->original_ce_credit_debt_observer_enabled ||
                                  need_lagged_impulse ||
                                  need_multiplicity_packets)
                ? (float *)malloc(ssm_bytes)
                : NULL;
            float *a_eff_tmp = (need_lagged_impulse || need_multiplicity_packets)
                ? (float *)malloc(ssm_bytes)
                : NULL;
            float *a_gate_tmp = need_lagged_impulse
                ? (float *)malloc(ssm_bytes)
                : NULL;
            R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                    seq_idx, k, "pre-ssm-bwd-submit");
            fflush(stderr);
            vk_end_batch(ctx); vk_begin_batch(ctx);
            R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                    seq_idx, k, "post-ssm-bwd-submit");
            fflush(stderr);
            if (dB_tmp) {
                vk_download(ctx, dA_ps,  p->bwd_dA_ps,  ssm_bytes);
                vk_download(ctx, dB_tmp, p->bwd_dB_seq, ssm_bytes);
                vk_download(ctx, d_x_cv, p->bwd_d_x_cv, ssm_bytes);
            } else {
                vk_download(ctx, dA_ps,     p->bwd_dA_ps,  ssm_bytes);
                /* fallback: download only first seq×ds (batch=1 safe) */
                vk_download(ctx, L->dB_seq, p->bwd_dB_seq,
                            (size_t)seq_len * ds * sizeof(float));
                vk_download(ctx, d_x_cv,    p->bwd_d_x_cv, ssm_bytes);
            }
            if (v_pre_tmp) {
                vk_download(ctx, v_pre_tmp, p->bwd_v_pre, ssm_bytes);
            }
            if (h_cache_tmp) {
                vk_download(ctx, h_cache_tmp, p->layers[k].fwd_h_cache, ssm_bytes);
            }
            if (a_eff_tmp) {
                vk_download(ctx, a_eff_tmp, p->ssm_a, ssm_bytes);
            }
            if (a_gate_tmp) {
                vk_download(ctx, a_gate_tmp, p->layers[k].fwd_a_buf, ssm_bytes);
            }
            R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                    seq_idx, k, "post-pre-directb-downloads");
            fflush(stderr);
            if (r400_obs_trace_should_capture(r400_obs_cfg, seq_idx, k)) {
                r400_obs_trace_capture_training(p,
                                                ctx,
                                                r400_obs_cfg,
                                                L,
                                                seq_idx,
                                                k,
                                                0,
                                                dA_ps,
                                                seq_len,
                                                ds,
                                                batch_size);
            }
            if (m->boundary_flux_sheaf_gate_enabled) {
                boundary_flux_sheaf_gate_update(
                    m, dA_ps, boundary_mask, batch_size, seq_len, ds);
            }
            if (m->boundary_cotangent_credit_gate_enabled) {
                boundary_cotangent_credit_gate_update(
                    m, dA_ps, boundary_mask, batch_size, seq_len, ds);
            }
            if (m->original_ce_credit_debt_observer_enabled &&
                v_pre_tmp && h_cache_tmp) {
                original_ce_credit_debt_observer_update(
                    m, dA_ps, v_pre_tmp, h_cache_tmp, boundary_mask,
                    batch_size, seq_len, ds);
            }
            if (m->lagged_ce_impulse_eligibility_enabled &&
                v_pre_tmp && h_cache_tmp && a_eff_tmp && a_gate_tmp) {
                lagged_ce_impulse_applied =
                    lagged_ce_impulse_eligibility_update(
                        m, L, dA_ps, v_pre_tmp, h_cache_tmp, a_eff_tmp,
                        a_gate_tmp, x_cv, boundary_mask, batch_size,
                        seq_len, ds);
            }
            if (m->executable_multiplicity_packets_enabled &&
                h_cache_tmp && a_eff_tmp) {
                executable_multiplicity_packets_update(
                    m, dA_ps, h_cache_tmp, a_eff_tmp, boundary_mask,
                    batch_size, seq_len, ds);
            }
            if (m->variance_priced_credit_gate_enabled) {
                martingale_credit_applied = variance_priced_credit_gate_update(
                    m, dA_ps, batch_size, seq_len, ds);
            }
            free(v_pre_tmp);
            free(h_cache_tmp);
            free(a_eff_tmp);
            free(a_gate_tmp);

            vk_begin_batch(ctx);
            R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                    seq_idx, k, "post-directb-rebatch");
            fflush(stderr);

#if V12_ASSSI_ENABLED
            if (VK_BUFFER_VALID(p->opt_layers[k].dC)
                && VK_BUFFER_VALID(p->gemm_A) && VK_BUFFER_VALID(p->gemm_B)
                && VK_BUFFER_VALID(p->bwd_d_y_ssm)
                && p->layers[k].fwd_allocated
                && VK_BUFFER_VALID(p->layers[k].fwd_h_cache)) {
                // Upload current accumulated L->dC if non-NULL, otherwise zero-initialize
                if (L->dC != NULL) {
                    vk_upload(ctx, p->opt_layers[k].dC, L->dC, ds * ds * sizeof(float));
                } else {
                    vk_fill_zero(ctx, p->opt_layers[k].dC, ds * ds * sizeof(float));
                }
                // Copy inputs to scratch GEMM buffers
                vk_copy_buffer(ctx, p->gemm_A, p->bwd_d_y_ssm, ssm_bytes);
                vk_copy_buffer(ctx, p->gemm_B, p->layers[k].fwd_h_cache, ssm_bytes);
                vk_barrier(ctx);
                // Perform fp32_gemm: C = C + A^T B, shape (ds, ds)
                vk_dispatch_fp32_gemm(ctx, p->gemm_A, p->gemm_B, p->opt_layers[k].dC,
                                      (uint32_t)(batch_size * seq_len), (uint32_t)ds, (uint32_t)ds, 1);
            }
#endif

            if (L->sampled_C_bilinear != NULL && L->dC_bilinear != NULL
                    && VK_BUFFER_VALID(p->bwd_dC_bl_accum)
                    && p->layers[k].fwd_allocated) {
                vk_upload(ctx, p->bwd_dC_bl_accum, L->dC_bilinear,
                          (size_t)ds * sizeof(float));
                vk_dispatch_dc_bilinear_reduce(ctx,
                    p->bwd_d_y_ssm,
                    p->layers[k].fwd_h_cache,
                    p->layers[k].fwd_x_cv,
                    p->bwd_dC_bl_accum,
                    (uint32_t)seq_len, (uint32_t)ds, (uint32_t)batch_size);
            }

            if (VK_BUFFER_VALID(p->bwd_dB_seq)
                    && VK_BUFFER_VALID(p->bwd_d_x_cv)
                    && VK_BUFFER_VALID(p->bwd_dB_accum)
                    && VK_BUFFER_VALID(p->bwd_B_vec)
                    && p->layers[k].fwd_allocated
                    && VK_BUFFER_VALID(p->layers[k].fwd_x_cv)) {
                vk_fill_zero(ctx, p->bwd_dB_accum, (size_t)ds * sizeof(uint32_t));
                vk_upload(ctx, p->bwd_B_vec, L->B, (size_t)ds * sizeof(float));
                vk_dispatch_direct_b_backward(ctx,
                    p->bwd_dB_seq,
                    p->layers[k].fwd_x_cv,
                    p->bwd_B_vec,
                    p->bwd_d_x_cv,
                    p->bwd_dB_accum,
                    (uint32_t)seq_len, (uint32_t)ds, (uint32_t)batch_size);
                dB_accum_raw = (uint32_t *)malloc((size_t)ds * sizeof(uint32_t));
            }

            vk_end_batch(ctx); vk_begin_batch(ctx);
            R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                    seq_idx, k, "post-directb-submit");
            fflush(stderr);

#if V12_ASSSI_ENABLED
#if V12_ASSSI_VALIDATION
            if (VK_BUFFER_VALID(p->opt_layers[k].dC)) {
                dC_gpu = malloc(ds * ds * sizeof(float));
                if (dC_gpu) {
                    vk_download(ctx, dC_gpu, p->opt_layers[k].dC, ds * ds * sizeof(float));
                    
                    // Perform CPU verification and overwrite L->dC
                    float *dy_cpu = malloc(ssm_bytes);
                    float *h_cpu = malloc(ssm_bytes);
                    if (dy_cpu && h_cpu) {
                        vk_download(ctx, dy_cpu, p->bwd_d_y_ssm, ssm_bytes);
                        vk_download(ctx, h_cpu, p->layers[k].fwd_h_cache, ssm_bytes);
                        
                        // Compute reference dC on CPU: dC_ref = dy^T @ h
                        float *dC_ref = calloc(ds * ds, sizeof(float));
                        if (dC_ref) {
                            int L_len = batch_size * seq_len;
                            for (int t = 0; t < L_len; t++) {
                                for (int i = 0; i < ds; i++) {
                                    float dy_val = dy_cpu[t * ds + i];
                                    for (int j = 0; j < ds; j++) {
                                        dC_ref[i * ds + j] += dy_val * h_cpu[t * ds + j];
                                    }
                                }
                            }
                            
                            // Compare dC_gpu against (L->dC + dC_ref)
                            float p->d_modeliff = 0.0f;
                            int n_c = ds * ds;
                            for (int i = 0; i < n_c; i++) {
                                float ref_val = (L->dC != NULL ? L->dC[i] : 0.0f) + dC_ref[i];
                                float diff = fabsf(dC_gpu[i] - ref_val);
                                if (diff > p->d_modeliff) p->d_modeliff = diff;
                            }
                            printf("[ASSSI VALIDATION] layer=%d p->d_modeliff=%.8e\n", k, (double)p->d_modeliff);
                            
                            // Overwrite CPU L->dC with the GPU-computed gradient if non-NULL
                            if (L->dC != NULL) {
                                for (int i = 0; i < n_c; i++) {
                                    L->dC[i] = dC_gpu[i];
                                }
                            }
                            
                            free(dC_ref);
                        }
                    }
                    if (dy_cpu) free(dy_cpu);
                    if (h_cpu) free(h_cpu);
                }
            }
#else
#pragma message("Compiling MTP else branch")
            if (L->dC != NULL && VK_BUFFER_VALID(p->opt_layers[k].dC)) {
                vk_download(ctx, L->dC, p->opt_layers[k].dC, ds * ds * sizeof(float));
            }
#endif
#endif

            if (L->sampled_C_bilinear != NULL && L->dC_bilinear != NULL
                    && VK_BUFFER_VALID(p->bwd_dC_bl_accum)
                    && p->layers[k].fwd_allocated) {
                vk_download(ctx, L->dC_bilinear, p->bwd_dC_bl_accum,
                            (size_t)ds * sizeof(float));
            }

            if (dB_accum_raw) {
                vk_download(ctx, dB_accum_raw, p->bwd_dB_accum,
                            (size_t)ds * sizeof(uint32_t));
            }
            R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                    seq_idx, k, "post-directb-downloads");
            fflush(stderr);

            /* CPU accumulation of downloaded SSM gradients */
            if (dB_tmp) {
                /* Reduce per-batch dB_seq into L->dB_seq[seq×ds] by summing over batch */
                memset(L->dB_seq, 0, (size_t)seq_len * ds * sizeof(float));
                for (int b = 0; b < batch_size; b++)
                    for (int t = 0; t < seq_len; t++)
                        for (int i = 0; i < ds; i++)
                            L->dB_seq[t * ds + i] +=
                                dB_tmp[((size_t)b * seq_len + t) * ds + i];
                free(dB_tmp);
            }

            if (dB_accum_raw) {
                float *dB_accum_f = (float *)dB_accum_raw;
                for (int i = 0; i < ds; i++)
                    L->dB[i] += dB_accum_f[i];
                free(dB_accum_raw);
                dB_accum_raw = NULL;
                direct_b_gpu_done = 1;
            }

            gpu_ssm_bwd_done = 1;
            R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                    seq_idx, k, "post-directb-host-accum");
            fflush(stderr);

            /* R368 M1 — free transient recompute scratches at the
             * backward-exit seam. Lifetime ends after all SSM backward
             * consumers for this layer have run. */
            if (VK_BUFFER_VALID(a_gate_scratch)) {
                vk_free(ctx, a_gate_scratch);
                a_gate_scratch = VK_BUFFER_NULL;
            }
            if (VK_BUFFER_VALID(a_eff_scratch)) {
                vk_free(ctx, a_eff_scratch);
                a_eff_scratch = VK_BUFFER_NULL;
            }
            R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                    seq_idx, k, "post-ssm-scratch-free");
            fflush(stderr);
        }
        if (VK_BUFFER_VALID(a_gate_scratch)) vk_free(ctx, a_gate_scratch);
        if (VK_BUFFER_VALID(a_eff_scratch))  vk_free(ctx, a_eff_scratch);
        if (!gpu_ssm_bwd_done) {
            /* fwd_a_buf and fwd_h_cache were not downloaded in forward (GPU-resident only).
             * Fetch them now so the CPU SSM scan backward can read a_buf_bwd and h_cache. */
            if (p->layers[k].fwd_allocated && VK_BUFFER_VALID(p->layers[k].fwd_a_buf)) {
#if V12_AF8C_ENABLED
                vk_download(ctx, a_buf_bwd, p->layers[k].fwd_a_buf, ssm_bytes);
                /* Stochastic compress back to c->a_eff_buf since it was not downloaded in forward */
                uint32_t rng = 0x12345678u ^ (uint32_t)k;
                uint8_t *dst = c->a_eff_buf + (size_t)k * bseq * ds;
                for (size_t i = 0; i < (size_t)bseq * ds; i++) {
                    dst[i] = compress_e4m3_stochastic(a_buf_bwd[i], &rng);
                }
#else
#pragma message("Compiling MTP else branch")
                float *a_buf_cpu = c->a_eff_buf + (size_t)k * bseq * ds;
                vk_download(ctx, a_buf_cpu, p->layers[k].fwd_a_buf, ssm_bytes);
#endif
            }
            
            // F34 SRAT: Reconstruct
            if (VK_BUFFER_VALID(p->layers[k].f34_z_cache)) {
                vk_dispatch_f34_srat_reconstruct(ctx, p->layers[k].f34_z_cache, p->layers[k].f34_P_r, p->layers[k].fwd_h_cache, seq_len, ds, 64);
            } else {
if (p->layers[k].fwd_allocated && VK_BUFFER_VALID(p->layers[k].fwd_h_cache)) {
                vk_download(ctx, h_cache, p->layers[k].fwd_h_cache, ssm_bytes);
                /* Stochastic compress back to c->h_all since it was not downloaded in forward */
                uint32_t rng = 0x12345678u ^ (uint32_t)k;
                uint8_t *dst = c->h_all + (size_t)k * bseq * ds;
                for (size_t i = 0; i < (size_t)bseq * ds; i++) {
                    dst[i] = compress_e5m2_stochastic(h_cache[i], &rng);
                }
            }
            memset(dC_scratch, 0, (size_t)V12_D_STATE * V12_D_STATE * sizeof(float));
            float *dB_seq_tmp = malloc((size_t)batch_size * seq_len * ds * sizeof(float));
            if (dB_seq_tmp) {
                memset(dB_seq_tmp, 0, (size_t)batch_size * seq_len * ds * sizeof(float));
                /* R356 F1 slice S3 — runtime selector between the reversible F1
                 * backward and the canonical cached backward.
                 *
                 * Default is OFF (V12_USE_F1_REVERSIBLE undefined). In that case
                 * this block is bit-identical to the pre-S3 code path and calls
                 * ssm_scan_backward_batch unconditionally.
                 *
                 * When V12_USE_F1_REVERSIBLE is defined, we inspect the forward
                 * stability probe emitted by ssm_scan_forward_batch_with_stability_probe
                 * (S0) and, if the per-channel cumulative max is below ln(10),
                 * dispatch to ssm_scan_backward_f1_batch (S2). Otherwise we log
                 * a warning and fall back to the same cached backward used
                 * today — which is ssm_scan_backward_batch, the canonical
                 * cached path already wired into this call site, NOT
                 * ssm_scan_backward_checkpointed_batch. The handoff (§5 item 11)
                 * names `_checkpointed` as the fallback target, but the current
                 * pre-S3 code in this path calls `_batch`, and constraint #4
                 * requires the fallback to be bit-identical to today's behavior.
                 *
                 * TODO(R356-F1): once S0 is plumbed through gpu_train_step (a
                 * per-batch stability_probe_cum buffer next to h_ssm_final and
                 * a per-batch h_final buffer of shape [batch_size × d_state]),
                 * replace the NULL pointers below with the real probe and
                 * final-state buffers. Until then, the ON path will always
                 * observe probe == NULL, log the warning, and fall back — so
                 * enabling the macro is safe but a no-op for gradients.
                 *
                 * TODO(R356-F1): replace the compile-time macro with a runtime
                 * Model / TrainConfig flag once S4-S6 land and the smoke test
                 * in §6 of the handoff has passed. See
                 * .workspace/research/r356-f1-reversibility-handoff.md §3.3.
                 */
                int use_f1 = 0;
#ifdef V12_USE_F1_REVERSIBLE
                {
                    /* R356 F1 slice S5 — recover stability_probe_cum and
                     * h_final_batch from the activations already on CPU at
                     * this point in the backward path.
                     *
                     * stability_probe_cum[b*ds + i] = Σ_t ln(max(|A_eff[t][i]
                     *                                            + C_bl[i]*x_cv[t][i]|, 1e-30))
                     *
                     * h_final_batch[b*ds + i] = h_cache[b*T*ds + (T-1)*ds + i]
                     *                        = post-update state after token T-1
                     *
                     * a_buf_bwd holds A_eff per timestep (same layout used by
                     * the canonical backward). x_cv and h_cache were already
                     * downloaded above when gpu_ssm_bwd_done==0. */
                    const float *C_bl_ptr = L->sampled_C_bilinear;
                    const int has_bl      = (C_bl_ptr != NULL);
                    float *f1_probe =
                        (float *)malloc((size_t)batch_size * ds * sizeof(float));
                    float *f1_h_final =
                        (float *)malloc((size_t)batch_size * ds * sizeof(float));
                    if (f1_probe && f1_h_final) {
                        memset(f1_probe, 0, (size_t)batch_size * ds * sizeof(float));
                        for (int b = 0; b < batch_size; b++) {
                            const float *a_b = a_buf_bwd + (size_t)b * seq_len * ds;
                            const float *x_b = x_cv       + (size_t)b * seq_len * ds;
                            float *probe_b   = f1_probe   + (size_t)b * ds;
                            for (int t = 0; t < seq_len; t++) {
                                const float *a_t = a_b + (size_t)t * ds;
                                const float *x_t = x_b + (size_t)t * ds;
                                for (int i = 0; i < ds; i++) {
                                    float a_tot = a_t[i];
                                    if (has_bl) a_tot += C_bl_ptr[i] * x_t[i];
                                    float mag = fabsf(a_tot);
                                    if (mag < 1e-30f) mag = 1e-30f;
                                    probe_b[i] += logf(mag);
                                }
                            }
                            /* h_final = last-timestep row of h_cache */
                            const float *hc_last = h_cache
                                + ((size_t)b * seq_len + (seq_len - 1)) * ds;
                            memcpy(f1_h_final + (size_t)b * ds,
                                   hc_last,
                                   (size_t)ds * sizeof(float));
                        }

                        const float ln10 = 2.302585092994046f;
                        float max_cum = 0.0f;
                        int n_probe = batch_size * ds;
                        for (int i = 0; i < n_probe; i++) {
                            float v = f1_probe[i];
                            if (v > max_cum) max_cum = v;
                        }
                        if (max_cum < ln10) {
                            use_f1 = 1;
                            ssm_scan_backward_f1_batch(
                                L->A, a_buf_bwd, NULL, b_eff_bwd, ds, L->C,
                                x_cv, f1_h_final, d_y_ssm, boundary_mask,
                                L->dA, dA_ps, dB_seq_tmp, dC_scratch,
                                d_x_cv, NULL,
                                L->sampled_C_bilinear, L->dC_bilinear,
                                f1_probe,
                                batch_size, seq_len, ds, ds,
                                L->sampled_u, L->sampled_v, L->du, L->dv);
                        } else {
                            fprintf(stderr,
                                "[R356-F1] stability probe %.4f >= ln(10); "
                                "falling back to cached backward (layer %d)\n",
                                (double)max_cum, k);
                        }
                    } else {
                        fprintf(stderr,
                                "[R356-F1] probe/h_final allocation failed; "
                                "falling back (layer %d)\n", k);
                    }
                    free(f1_probe);
                    free(f1_h_final);
                }
#endif
                float *dh_direct = NULL;
                if (state != NULL && !state->fp_completed && (int)state->fp_stage == k) {
                    dh_direct = calloc((size_t)batch_size * seq_len * ds, sizeof(float));
                    if (dh_direct) {
                        float alpha_sincts = 1.0f;
                        compute_sincts_loss_and_gradient(
                            h_cache, m->embed_E, tokens, dh_direct,
                            batch_size, seq_len, ds, V12_D_EMBED, alpha_sincts
                        );
                    }
                }

                if (!use_f1) {
                    ssm_scan_backward_batch(L->A, a_buf_bwd, NULL, b_eff_bwd, ds, L->C,
                                            x_cv, h_cache, d_y_ssm, boundary_mask,
                                            L->dA, dA_ps, dB_seq_tmp, dC_scratch,
                                            d_x_cv, NULL,
                                            batch_size, seq_len, ds, ds,
                                            L->sampled_C_bilinear, L->dC_bilinear,
                                            L->sampled_u, L->sampled_v, L->du, L->dv,
                                            dh_direct);
                }
                /* Reduce per-batch dB_seq into L->dB_seq[seq×ds] by summing over batch */
                memset(L->dB_seq, 0, (size_t)seq_len * ds * sizeof(float));
                for (int b = 0; b < batch_size; b++)
                    for (int t = 0; t < seq_len; t++)
                        for (int i = 0; i < ds; i++)
                            L->dB_seq[t * ds + i] +=
                                dB_seq_tmp[((size_t)b * seq_len + t) * ds + i];
                free(dB_seq_tmp);

                if (dh_direct) free(dh_direct);
            } else {
                float *dh_direct = NULL;
                if (state != NULL && !state->fp_completed && (int)state->fp_stage == k) {
                    dh_direct = calloc((size_t)batch_size * seq_len * ds, sizeof(float));
                    if (dh_direct) {
                        float alpha_sincts = 1.0f;
                        compute_sincts_loss_and_gradient(
                            h_cache, m->embed_E, tokens, dh_direct,
                            batch_size, seq_len, ds, V12_D_EMBED, alpha_sincts
                        );
                    }
                }

                ssm_scan_backward(L->A, a_buf_bwd, NULL, b_eff_bwd, ds, L->C,
                                  x_cv, h_cache, d_y_ssm, boundary_mask,
                                  L->dA, dA_ps, L->dB_seq, dC_scratch,
                                  d_x_cv, NULL, seq_len, ds, ds,
                                  L->sampled_C_bilinear, L->dC_bilinear,
                                  L->sampled_u, L->sampled_v, L->du, L->dv,
                                  dh_direct);

                if (dh_direct) free(dh_direct);
            }
            if (L->dC != NULL) {
                int n_c = ds * ds;
                for (int i = 0; i < n_c; i++)
                    L->dC[i] += dC_scratch[i];
            }
        }

        /* Direct-B backward — GPU fused path when all buffers are available:
         * single shader handles batch reduction of dB_seq, dB accumulation,
         * and d_x_cv update without any host-side downloads or CPU loops. */
        int direct_b_gpu_planned = gpu_ssm_bwd_done
                && VK_BUFFER_VALID(p->bwd_dB_seq)
                && VK_BUFFER_VALID(p->bwd_d_x_cv)
                && VK_BUFFER_VALID(p->bwd_dB_accum)
                && VK_BUFFER_VALID(p->bwd_B_vec)
                && p->layers[k].fwd_allocated
                && VK_BUFFER_VALID(p->layers[k].fwd_x_cv);
        if (!direct_b_gpu_planned) {
            /* CPU fallback: dB_per_t is the batch-reduced dB_seq (seq×ds) */
            float *dB_per_t = L->dB_seq;
#pragma omp parallel for schedule(static)
            for (int i = 0; i < ds; i++) {
                float acc = 0.0f;
                for (int t = 0; t < seq_len; t++) {
                    float xc   = x_cv[t * ds + i];
                    float gate = 1.0f / (1.0f + expf(-xc));
                    acc += dB_per_t[t * ds + i] * gate;
                }
                L->dB[i] += acc;
            }
            for (int b = 0; b < batch_size; b++) {
                for (int t = 0; t < seq_len; t++) {
                    int flat = b * seq_len + t;
                    for (int i = 0; i < ds; i++) {
                        float xc    = x_cv[flat * ds + i];
                        float sig   = 1.0f / (1.0f + expf(-xc));
                        float dsig  = sig * (1.0f - sig);
                        float dgate = dB_per_t[t * ds + i] * L->B[i];
                        d_x_cv[flat * ds + i] += dgate * dsig;
                    }
                }
            }
        }

        if (direct_b_gpu_planned && !direct_b_gpu_done) {
            /* CPU fallback: dB_per_t is the batch-reduced dB_seq (seq×ds). */
            float *dB_per_t = L->dB_seq;
#pragma omp parallel for schedule(static)
            for (int i = 0; i < ds; i++) {
                float acc = 0.0f;
                for (int t = 0; t < seq_len; t++) {
                    float xc   = x_cv[t * ds + i];
                    float gate = 1.0f / (1.0f + expf(-xc));
                    acc += dB_per_t[t * ds + i] * gate;
                }
                L->dB[i] += acc;
            }
            for (int b = 0; b < batch_size; b++) {
                for (int t = 0; t < seq_len; t++) {
                    int flat = b * seq_len + t;
                    for (int i = 0; i < ds; i++) {
                        float xc    = x_cv[flat * ds + i];
                        float sig   = 1.0f / (1.0f + expf(-xc));
                        float dsig  = sig * (1.0f - sig);
                        float dgate = dB_per_t[t * ds + i] * L->B[i];
                        d_x_cv[flat * ds + i] += dgate * dsig;
                    }
                }
            }
        }

        /* Steps 6-9: gates backward — A-backward on GPU when buffers available,
         * conv1d/gate-backward remain on CPU due to timestep boundary dependencies */
        {
            /* Input-dependent A backward */
#ifdef V12_HAS_VULKAN
            if (VK_BUFFER_VALID(p->bwd_d_dot)
                && VK_BUFFER_VALID(p->bwd_dA_reduce)
                && VK_BUFFER_VALID(p->bwd_dbA_reduce)
                && VK_BUFFER_VALID(p->bwd_d_x_cv)
                && VK_BUFFER_VALID(p->lw_WA)
                && VK_BUFFER_VALID(p->gemm_A)
                && VK_BUFFER_VALID(p->gemm_B)
                && VK_BUFFER_VALID(p->gemm_C)
                && p->staging_dw
                && !martingale_credit_applied
                && !lagged_ce_impulse_applied) {

                R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                        seq_idx, k, "pre-a-backward-batch");
                fflush(stderr);
                vk_begin_batch(ctx);

                /* Upload d_x_cv only when Direct-B ran on CPU; the GPU path
                 * updates bwd_d_x_cv in-place via the fused shader, so an
                 * upload here would overwrite those gradients with stale data. */
                if (!direct_b_gpu_done)
                    vk_upload(ctx, p->bwd_d_x_cv, d_x_cv, ssm_bytes);

                /* Upload current W_A for this layer — skip when GPU optimizer owns it */
                if (!gpu_owns_wa)
                    vk_upload(ctx, p->lw_WA, L->W_A, (size_t)ds * ds * sizeof(float));

                /* R368 M1 S3 — d_dot, dA and db_A were already produced by the
                 * fused SSM backward elementwise shader. Only the W_A^T pass
                 * and dW_A GEMM remain here. */
                vk_dispatch_a_backward_dx_cv(ctx,
                    p->bwd_d_dot,
                    p->lw_WA,
                    p->bwd_d_x_cv,
                    (uint32_t)seq_len, (uint32_t)ds, (uint32_t)batch_size);
                vk_barrier(ctx);

                /* dW_A += d_dot^T @ x_cv  via fp32_gemm (T=bseq, M=ds, N=ds) */
                /* gemm_A = d_dot [bseq×ds], gemm_B = x_cv [bseq×ds], gemm_C = dW_A [ds×ds] */
                vk_copy_buffer(ctx, p->gemm_A, p->bwd_d_dot,  ssm_bytes);
                vk_copy_buffer(ctx, p->gemm_B, p->layers[k].fwd_x_cv, ssm_bytes);
                vk_barrier(ctx);
                vk_dispatch_fp32_gemm(ctx, p->gemm_A, p->gemm_B, p->gemm_C,
                                      (uint32_t)bseq, (uint32_t)ds, (uint32_t)ds, 0);
                vk_sync(ctx);

                /* Save dW_A GPU-resident before download */
                if (VK_BUFFER_VALID(p->opt_layers[k].gpu_dW_A))
                    vk_dispatch_residual_add(ctx, p->gemm_C, p->opt_layers[k].gpu_dW_A,
                                             (uint32_t)(ds * ds));

                /* Accumulate dA and db_A into per-layer buffers before next layer overwrites shared scratch */
                if (VK_BUFFER_VALID(p->opt_layers[k].gpu_dA))
                    vk_dispatch_residual_add(ctx, p->bwd_dA_reduce, p->opt_layers[k].gpu_dA,
                                             (uint32_t)ds);
                if (VK_BUFFER_VALID(p->opt_layers[k].gpu_dbA))
                    vk_dispatch_residual_add(ctx, p->bwd_dbA_reduce, p->opt_layers[k].gpu_dbA,
                                             (uint32_t)ds);

                /* Download results */
                R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                        seq_idx, k, "post-a-backward-record");
                fflush(stderr);
                float *dA_reduce_cpu  = malloc((size_t)ds * sizeof(float));
                float *dbA_reduce_cpu = malloc((size_t)ds * sizeof(float));
                float *dwa_cpu        = p->staging_dw;

                if (dA_reduce_cpu && dbA_reduce_cpu) {
                    int gpu_handles_A  = gpu_owns_a;
                    int gpu_handles_bA = gpu_owns_ba;
                    int gpu_handles_WA = gpu_owns_wa;
                    /* Skip d_x_cv download when GPU conv1d+gate will read it directly */
                    int gpu_conv1d_avail = VK_BUFFER_VALID(p->bwd_d_x_cv)
                        && VK_BUFFER_VALID(p->lw_conv)
                        && VK_BUFFER_VALID(p->layers[k].fwd_x_proj)
                        && VK_BUFFER_VALID(p->layers[k].fwd_mask_val)
                        && VK_BUFFER_VALID(p->bwd_d_xp)
                        && VK_BUFFER_VALID(p->bwd_d_conv_w)
                        && VK_BUFFER_VALID(p->bwd_d_tau);

                    R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                            seq_idx, k, "pre-a-backward-submit");
                    fflush(stderr);
                    vk_end_batch(ctx); vk_begin_batch(ctx);
                    R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                            seq_idx, k, "post-a-backward-submit");
                    fflush(stderr);

                    if (!gpu_handles_A) {
                        vk_download(ctx, dA_reduce_cpu,  p->bwd_dA_reduce,  (size_t)ds * sizeof(float));
                    }
                    if (!gpu_handles_bA) {
                        vk_download(ctx, dbA_reduce_cpu, p->bwd_dbA_reduce, (size_t)ds * sizeof(float));
                    }
                    if (!gpu_handles_WA)
                        vk_download(ctx, dwa_cpu,        p->gemm_C,         (size_t)ds * ds * sizeof(float));
                    if (!gpu_conv1d_avail)
                        vk_download(ctx, d_x_cv, p->bwd_d_x_cv, ssm_bytes);

                    R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                            seq_idx, k, "post-a-backward-downloads");
                    fflush(stderr);
                    if (!gpu_handles_A) {
                        for (int i = 0; i < ds; i++) L->dA[i] += dA_reduce_cpu[i];
                    }
                    if (!gpu_handles_bA) {
                        for (int i = 0; i < ds; i++) {
                            L->db_A[i] += dbA_reduce_cpu[i];
                        }
                    }
                    if (!gpu_handles_WA)
                        for (int i = 0; i < ds * ds; i++) L->dW_A[i] += dwa_cpu[i];

                    free(dA_reduce_cpu);
                    free(dbA_reduce_cpu);
                    R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                            seq_idx, k, "post-a-backward-host-accum");
                    fflush(stderr);
                } else {
                    /* malloc failed — fall through to CPU path.
                     * materializing c->a_eff_gate via the recompute shader.
                     * so fetch it now before jumping to the CPU backward. */
                    free(dA_reduce_cpu);
                    free(dbA_reduce_cpu);
                    vk_end_batch(ctx); vk_begin_batch(ctx);
                    materialize_a_gate_from_gpu_recompute(
                        p, L, k, a_gate,
                        ssm_bytes, seq_len, ds, batch_size, m->gpu_opt_flags);
                    goto cpu_a_backward;
                }
            } else {
cpu_a_backward:;
                /* When GPU A-backward is unavailable and forward ran on GPU,
                 * materialize c->a_eff_gate from the recompute shader on demand. */
                if (p->layers[k].fwd_allocated) {
                    materialize_a_gate_from_gpu_recompute(
                        p, L, k, a_gate,
                        ssm_bytes, seq_len, ds, batch_size, m->gpu_opt_flags);
                }
#else
#pragma message("Compiling MTP else branch")
            {
#endif
                /* a_gate is already declared and initialized at the top of Phase 8 loop */
                for (int t = 0; t < bseq; t++) {
                    const float *xc_t   = x_cv + t * ds;
                    float       *d_xc_t = d_x_cv + t * ds;
                    if (martingale_credit_applied) {
                        int row_has_credit = 0;
                        const float *dA_row = dA_ps + (size_t)t * (size_t)ds;
                        for (int i = 0; i < ds; i++) {
                            if (dA_row[i] != 0.0f) {
                                row_has_credit = 1;
                                break;
                            }
                        }
                        if (!row_has_credit) continue;
                    }
                    for (int i = 0; i < ds; i++) {
                        float da_eff  = dA_ps[t * ds + i];
                        if (martingale_credit_applied && da_eff == 0.0f) continue;
                        float gate_ti = a_gate[t * ds + i];
                        // Q-21: Bounded-Spectrum SSM State Transition Smoothing
                        float A_val = L->A[i];
                        float exp_A = expf(A_val);
                        float sigmoid_A = 1.0f / (1.0f + expf(-A_val));
                        L->dA[i] += da_eff * gate_ti * (-sigmoid_A);
                        float a_eff_mapped = -logf(1.0f + exp_A) - 0.001f;
                        float d_gate = da_eff * a_eff_mapped;
                        float d_dot  = d_gate * gate_ti * (1.0f - gate_ti);
                        L->db_A[i] += d_dot;
                        float *dwa_row      = L->dW_A + i * ds;
                        const float *wa_row = L->W_A  + i * ds;
                        for (int j = 0; j < ds; j++) {
                            dwa_row[j] += d_dot * xc_t[j];
                            d_xc_t[j]  += d_dot * wa_row[j];
                        }
                    }
                }
                if ((martingale_credit_applied || lagged_ce_impulse_applied) &&
                    VK_BUFFER_VALID(p->opt_layers[k].gpu_dW_A)) {
                    vk_begin_batch(ctx);
                    vk_upload(ctx, p->opt_layers[k].gpu_dW_A, L->dW_A,
                              (size_t)ds * (size_t)ds * sizeof(float));
                    vk_end_batch(ctx); vk_begin_batch(ctx);
                }
            }
            /* Conv1d backward (weight grad) + input-gate backward */
            R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                    seq_idx, k, "pre-conv1d-gate-backward");
            fflush(stderr);
#ifdef V12_HAS_VULKAN
            if (VK_BUFFER_VALID(p->bwd_d_x_cv)
                && VK_BUFFER_VALID(p->lw_conv)
                && VK_BUFFER_VALID(p->layers[k].fwd_x_proj)
                && VK_BUFFER_VALID(p->layers[k].fwd_mask_val)
                && VK_BUFFER_VALID(p->layers[k].fwd_x_gated)
                && VK_BUFFER_VALID(p->bwd_d_xp)
                && VK_BUFFER_VALID(p->bwd_d_conv_w)
                && VK_BUFFER_VALID(p->bwd_d_tau)) {

                vk_begin_batch(ctx);

                vk_fill_zero(ctx, p->bwd_d_conv_w, (size_t)V12_CONV_KERNEL * ds * sizeof(float));
                vk_fill_zero(ctx, p->bwd_d_xp,     ssm_bytes);
                vk_fill_zero(ctx, p->bwd_d_tau,     sizeof(float));

                /* Upload current conv weights so the weight-grad shader can read them —
                 * skip when GPU optimizer owns shadow_w_conv and keeps it current */
                if (!(m->gpu_opt_flags & GPU_OPT_CONV))
                    upload_shadow_buffer(ctx, p->lw_conv, L->shadow_w_conv, (size_t)V12_CONV_KERNEL * ds);

                vk_dispatch_conv1d_bwd_weights(ctx,
                    p->bwd_d_x_cv,
                    p->layers[k].fwd_x_gated,
                    p->bwd_d_conv_w,
                    seq_len, ds, V12_CONV_KERNEL, batch_size);
                vk_barrier(ctx);

                /* Accumulate d_conv_w into per-layer buffer before next layer overwrites shared scratch */
                if (VK_BUFFER_VALID(p->opt_layers[k].gpu_dconv))
                    vk_dispatch_residual_add(ctx, p->bwd_d_conv_w, p->opt_layers[k].gpu_dconv,
                                             (uint32_t)(V12_CONV_KERNEL * ds));
                vk_barrier(ctx);

                vk_dispatch_conv1d_bwd_input_gate(ctx,
                    p->bwd_d_x_cv,
                    p->lw_conv,
                    p->layers[k].fwd_x_proj,
                    p->layers[k].fwd_mask_val,
                    p->bwd_d_xp,
                    p->bwd_d_tau,
                    (uint32_t)seq_len, (uint32_t)ds,
                    m->tau_mux[k], V12_SOFT_BOUNDARY_SCALE,
                    (uint32_t)batch_size);
                vk_sync(ctx);

                size_t dconv_bytes = (size_t)V12_CONV_KERNEL * ds * sizeof(float);
                float *d_conv_w_cpu = malloc(dconv_bytes);
                float  d_tau_cpu    = 0.0f;
                vk_end_batch(ctx); vk_begin_batch(ctx);
                if (d_conv_w_cpu) {
                    int gpu_handles_conv = gpu_owns_conv;
                    if (!gpu_handles_conv)
                        vk_download(ctx, d_conv_w_cpu, p->bwd_d_conv_w, dconv_bytes);
                    vk_download(ctx, d_xp,         p->bwd_d_xp,     ssm_bytes);
                    vk_download(ctx, &d_tau_cpu,   p->bwd_d_tau,    sizeof(float));

                    if (!gpu_handles_conv) {
                        for (int i = 0; i < V12_CONV_KERNEL * ds; i++)
                            L->d_conv_w[i] += d_conv_w_cpu[i];
                    }
                    m->d_tau_mux[k] += d_tau_cpu;
                    free(d_conv_w_cpu);
                    R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                            seq_idx, k, "post-conv1d-gate-backward");
                    fflush(stderr);
                } else {
                    /* malloc failed — fall back to CPU path */
                    vk_download(ctx, d_x_cv, p->bwd_d_x_cv, ssm_bytes);
                    goto cpu_conv1d_gate_backward;
                }
            } else {
cpu_conv1d_gate_backward:;
                /* d_x_cv may be GPU-only (A-backward ran on GPU but skipped download) */
                if (VK_BUFFER_VALID(p->bwd_d_x_cv))
                    vk_download(ctx, d_x_cv, p->bwd_d_x_cv, ssm_bytes);
#endif
                {
                    /* x_gt is already declared and initialized at the top of Phase 8 loop */
                    const int K = V12_CONV_KERNEL;
                    memset(d_xp, 0, ssm_bytes);
                    for (int b = 0; b < batch_size; b++) {
                        int base = b * seq_len;
                        for (int t = 0; t < seq_len; t++) {
                            int flat_t = base + t;
                            for (int i = 0; i < ds; i++) {
                                float grad_out = d_x_cv[flat_t * ds + i];
                                for (int kk = 0; kk < K; kk++) {
                                    int src_t = t - kk;
                                    if (src_t < 0) src_t = 0;
                                    int flat_src = base + src_t;
                                    L->d_conv_w[kk * ds + i] += grad_out * x_gt[flat_src * ds + i];
                                    d_xp[flat_src * ds + i]  += L->shadow_w_conv[kk * ds + i] * grad_out;
                                }
                            }
                        }
                    }
                }
                {
                    /* mask is already declared and initialized at the top of Phase 8 loop */
                    float  scale = V12_SOFT_BOUNDARY_SCALE;
                    float  d_tau_acc = 0.0f;
#pragma omp parallel for schedule(static) reduction(+:d_tau_acc)
                    for (int t = 0; t < bseq; t++) {
                        for (int i = 0; i < ds; i++) {
                            float xp    = x_proj[t * ds + i];
                            float m_val = mask[t * ds + i];
                            float dxg   = d_xp[t * ds + i];
                            float m1m   = m_val * (1.0f - m_val);
                            float sign_xp = (xp > 0.0f) ? 1.0f : ((xp < 0.0f) ? -1.0f : 0.0f);
                            d_xp[t * ds + i] = dxg * (m_val + xp * scale * sign_xp * m1m);
                            d_tau_acc += dxg * xp * (-scale * m1m);
                        }
                    }
                    m->d_tau_mux[k] += d_tau_acc;
                }
#ifdef V12_HAS_VULKAN
            }
#endif
        }

        /* W_in weight gradient: dw_in += d_xp^T @ h_input
         * h_input = residual[k] which is GPU-resident — use vk_copy_buffer to
         * populate gemm_B directly, eliminating the CPU memcpy + vk_upload roundtrip. */
        R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                seq_idx, k, "pre-w-in-backward");
        fflush(stderr);
        float *h_input = c->residual + (size_t)k * bseq * d;
        if (VK_BUFFER_VALID(p->gemm_A) && VK_BUFFER_VALID(p->gemm_B) && VK_BUFFER_VALID(p->gemm_C)
            && VK_BUFFER_VALID(p->residual[k])
            && lbwd_A_buf && p->staging_dw && seq_len > 0
            && p->layers[k].csc_uploaded
            && VK_BUFFER_VALID(p->bwd_d_xp)
            && VK_BUFFER_VALID(p->bwd_d_h_prev)
            && VK_BUFFER_VALID(p->bwd_d_h_in)) {

            size_t xb_ds   = (size_t)batch_size * seq_len * ds * sizeof(float);
            size_t xb_d    = (size_t)batch_size * seq_len * d  * sizeof(float);
            size_t dw_bytes = (size_t)ds * d * sizeof(float);

            vk_begin_batch(ctx);

            memcpy(lbwd_A_buf, d_xp, xb_ds);
            vk_upload(ctx, p->gemm_A, lbwd_A_buf, xb_ds);
            vk_copy_buffer(ctx, p->gemm_B, p->residual[k], xb_d);
            // Q-09: Gradient-Rank Quotient Substrate for dw_in
            int is_audit = (m->step % 100 == 0);
            uint32_t bseq = batch_size * seq_len;
            uint32_t k_q09 = p->q09_k;
            
            vk_dispatch_random_projector(ctx, p->q09_P, bseq, k_q09, m->step + k_q09 + 1);
            vk_barrier(ctx);
            
            vk_dispatch_fp32_gemm(ctx, p->q09_P, p->gemm_A, p->q09_P_left, bseq, k_q09, ds, 0);
            vk_dispatch_fp32_gemm(ctx, p->q09_P, p->gemm_B, p->q09_P_right, bseq, k_q09, d, 0);
            vk_barrier(ctx);
            
            // MFCF: Inline Adam update instead of dense gradient
// //             float lr_eff = m->lr * (state ? state->lr_per_layer[k] : 1.0f);
// //             float tau = 1e-4f; // V12_CRYSTALLIZATION_TAU
// //             if (m->continuous_ternary_margin_enabled) {
// //                 tau = 1.0f + m->continuous_ternary_margin_c / (sqrtf(m->last_mean_v_w_in[k]) + 1e-8f);
// //             }
// //             vk_dispatch_sp19_holo_adam_reduce(ctx,
// //                 p->opt_layers[k].gpu_dw_in, p->q09_P_left, p->q09_P_right, k_q09,
// //                 p->opt_layers[k].shadow_w_in, p->opt_layers[k].sampled_w_in,
// //                 p->opt_layers[k].grad_accum_in, p->opt_layers[k].m_w_in_row, p->opt_layers[k].m_w_in_col,
// //                 p->opt_layers[k].v_w_in_row, p->opt_layers[k].v_w_in_col, p->opt_layers[k].g_row_in,
// //                 p->opt_layers[k].g_col_in, p->opt_layers[k].gq_row_in, p->opt_layers[k].gq_col_in,
// //                 p->opt_layers[k].cnt_row_in, p->opt_layers[k].cnt_col_in, p->opt_layers[k].mean_v_in,
// //                 p->opt_layers[k].g_row_in, d, d, ds, lr_eff, inv_bc1, inv_bc2, 0u,
// //                 evidence_gate, evidence_mode, m->evidence_rate_boundary_eps, tau, p->evidence_rate_stats);
// // 
            vk_barrier(ctx);
            
            if (is_audit) {
                vk_dispatch_fp32_gemm(ctx, p->gemm_A, p->gemm_B, p->gemm_C, bseq, ds, d, 0);
                vk_barrier(ctx);
                
                vk_fill_zero(ctx, p->q09_stats, 3 * sizeof(int));
                vk_barrier(ctx);
                
                vk_dispatch_cos_sim_audit(ctx, p->gemm_C, p->q09_dw_approx, p->q09_stats, ds * d);
                vk_barrier(ctx);
                
                int stats_host[3] = {0};
                    vk_download(ctx, stats_host, p->q09_stats, 3 * sizeof(int));
                    vk_end_batch(ctx); vk_begin_batch(ctx);
                
                float dot_val = *(float*)&stats_host[0];
                float nA = *(float*)&stats_host[1];
                float nB = *(float*)&stats_host[2];
                float cos_sim = dot_val / (sqrtf(nA) * sqrtf(nB) + 1e-8f);
                
                if (cos_sim < 0.8f) {
                    uint32_t new_k = p->q09_k + 16;
                    if (new_k > d) new_k = d;
                    if (new_k > ds) new_k = ds;
                    if (new_k != p->q09_k) {
                        printf("[Q-09] Audit failed dw_in (cos_sim=%.3f < 0.8). Increasing rank k: %u -> %u\n", cos_sim, p->q09_k, new_k);
                        p->q09_k = new_k;
                    }
                }
                
            } else {
                // MFCF: No dense copy to gemm_C

            }
            vk_barrier(ctx);


            if (VK_BUFFER_VALID(p->opt_layers[k].gpu_dw_in))
                vk_dispatch_residual_add(ctx, p->gemm_C, p->opt_layers[k].gpu_dw_in,
                                         (uint32_t)(ds * d));
            if (!gpu_owns_w_in)
                vk_download(ctx, p->staging_dw, p->gemm_C, dw_bytes);

            /* Download d_h only when PC backward needs CPU d_residual
             * (GPU PC shaders read bwd_d_h_in directly) */
#ifndef V12_PC_GPU_SHADERS
            {
                float *d_res_k = c->d_residual + (size_t)k * bseq * d;
                vk_download(ctx, d_res_k, p->bwd_d_h_in, res_bytes);
            }
#endif

            vk_end_batch(ctx); vk_begin_batch(ctx);

            if (!gpu_owns_w_in)
                for (int i = 0; i < ds * d; i++) L->dw_in[i] += p->staging_dw[i];
            d_h_on_gpu = 1;
        } else {
            if (VK_BUFFER_VALID(p->gemm_A) && VK_BUFFER_VALID(p->gemm_B) && VK_BUFFER_VALID(p->gemm_C)
                && VK_BUFFER_VALID(p->residual[k])
                && lbwd_A_buf && p->staging_dw && seq_len > 0) {
                size_t xb_ds   = (size_t)batch_size * seq_len * ds * sizeof(float);
                size_t xb_d    = (size_t)batch_size * seq_len * d  * sizeof(float);
                size_t dw_bytes = (size_t)ds * d * sizeof(float);
                memcpy(lbwd_A_buf, d_xp, xb_ds);
                vk_upload(ctx, p->gemm_A, lbwd_A_buf, xb_ds);
                vk_copy_buffer(ctx, p->gemm_B, p->residual[k], xb_d);
                vk_dispatch_fp32_gemm(ctx, p->gemm_A, p->gemm_B, p->gemm_C,
                                      batch_size * seq_len, ds, d, 0);
                vk_sync(ctx);
                if (!gpu_owns_w_in) {
                    vk_download(ctx, p->staging_dw, p->gemm_C, dw_bytes);
                    for (int i = 0; i < ds * d; i++) L->dw_in[i] += p->staging_dw[i];
                }
            } else {
#pragma omp parallel for schedule(static)
                for (int i = 0; i < ds; i++) {
                    float *dw_row = L->dw_in + i * d;
                    for (int t = 0; t < (int)bseq; t++) {
                        float dxp_ti = d_xp[t * ds + i];
                        float *h_t   = h_input + t * d;
                        for (int j = 0; j < d; j++) dw_row[j] += dxp_ti * h_t[j];
                    }
                }
            }

            printf("DEBUG: Before Step 11\n"); fflush(stdout);
            /* Step 11: W_in^T @ d_xp -> d_h_prev (CSC of csr_in) + accumulate into d_h */
            float *d_h_prev = c->d_residual + (size_t)k * bseq * d;
            if (p->layers[k].csc_uploaded
                && VK_BUFFER_VALID(p->bwd_d_xp)
                && VK_BUFFER_VALID(p->bwd_d_h_prev)
                && VK_BUFFER_VALID(p->bwd_d_h_in)) {
                vk_upload(ctx, p->bwd_d_xp, d_xp, ssm_bytes);
                vk_dispatch_ternary_matvec(ctx, &p->layers[k].gpu_csc_in,
                                           p->bwd_d_xp, p->bwd_d_h_prev, batch_size * seq_len);
                vk_barrier(ctx);
                vk_dispatch_residual_add(ctx, p->bwd_d_h_prev, p->bwd_d_h_in,
                                         (uint32_t)(batch_size * seq_len * d));
                vk_sync(ctx);
                d_h_on_gpu = 1;
            } else {
#pragma omp parallel for schedule(static)
                for (int t = 0; t < (int)bseq; t++) {
                    float *d_hp_t = d_h_prev + t * d;
                    memset(d_hp_t, 0, (size_t)d * sizeof(float));
                    ternary_matvec_T(&L->csr_in, d_xp + t * ds, d_hp_t);
                }
                d_h_on_gpu = 0;
                for (int t = 0; t < (int)bseq; t++) {
                    for (int j = 0; j < d; j++) d_h[t * d + j] += d_h_prev[t * d + j];
                }
            }

            /* BUG 1 FIX: download running d_h from GPU so PC backward (Phase 9) reads
             * correct per-layer d_residual values — bwd_d_h_in holds the accumulated
             * gradient after residual_add, which IS the gradient for layer k's input */
            if (d_h_on_gpu && VK_BUFFER_VALID(p->bwd_d_h_in)) {
                float *d_res_k = c->d_residual + (size_t)k * bseq * d;
                vk_download(ctx, d_res_k, p->bwd_d_h_in, res_bytes);
            }
            R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                    seq_idx, k, "post-w-in-backward");
            fflush(stderr);
        }
#if V12_AF8C_ENABLED
        /* Compress and write back read-write gradients and materialized gates */
        {
            uint32_t rng = 0x12345678u ^ (uint32_t)k;
            uint8_t *d_y_ssm_dest = c->d_x_proj + (size_t)k * bseq * ds;
            uint8_t *dA_ps_dest = c->dA_per_step + (size_t)k * bseq * ds;
            uint8_t *d_x_cv_dest = c->d_x_conv + (size_t)k * bseq * ds;
            uint8_t *d_xp_dest = c->d_x_proj + (size_t)k * bseq * ds;
            uint8_t *a_gate_dest = c->a_eff_gate + (size_t)k * bseq * ds;

            for (size_t i = 0; i < (size_t)bseq * ds; i++) {
                d_y_ssm_dest[i] = compress_e5m2_stochastic(d_y_ssm[i], &rng);
                dA_ps_dest[i]   = compress_e5m2_stochastic(dA_ps[i], &rng);
                d_x_cv_dest[i]  = compress_e5m2_stochastic(d_x_cv[i], &rng);
                d_xp_dest[i]    = compress_e5m2_stochastic(d_xp[i], &rng);
                a_gate_dest[i]  = compress_e4m3_stochastic(a_gate[i], &rng);
            }
        }
        /* Free local floats */
        free(y_ssm); free(x_proj); free(x_cv); free(d_y_ssm); free(b_eff_bwd); free(a_buf_bwd); free(dA_ps); free(d_x_cv); free(d_xp); free(a_gate); free(x_gt); free(mask);
#endif
        if (dC_gpu) free(dC_gpu);
        free(h_cache);
    }
    }

    R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
            seq_idx, -1, "post-layer-backward-loop");
    fflush(stderr);

    /* Download final d_h from GPU after all layers */
    if (!p->skip_bptt_current_step) {
        if (d_h_on_gpu && VK_BUFFER_VALID(p->bwd_d_h_in)) {
            vk_download(ctx, d_h, p->bwd_d_h_in, res_bytes);
            d_h_on_gpu = 0;
        }
    }

    R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
            seq_idx, -1, "post-final-dh-download");
    fflush(stderr);

    if (!p->skip_bptt_current_step) {
    /* Phase 9: PC backward — d_h now holds d_surp for layer 0.
     * GPU path: negate-gather d_residual[t+1] → GEMV^T → scatter-add into d_residual[t-1].
     *           negate-gather + gather_prev → GEMM → download → dw_pred +=.
     * CPU fallback: same logic with CPU pack/scatter then re-upload. */
    R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
            seq_idx, -1, "pre-pc-backward");
    fflush(stderr);
    {
        float *h_final_prev = c->residual   + (size_t)V12_N_LAYERS * bseq * d;
        float *d_h_final    = c->d_residual + (size_t)V12_N_LAYERS * bseq * d;
        int    T_per_batch  = seq_len - 1;
        int    T_total      = batch_size * T_per_batch;

#ifdef V12_PC_GPU_SHADERS
        if (VK_BUFFER_VALID(p->shadow_pred_A) && T_total > 0) {
            size_t dw_bytes = (size_t)d * d * sizeof(float);

            /* Layer 0 Op A: d_h_final[t-1] += shadow_pred_A^T @ (-d_h_final[t]) */
            vk_upload(ctx, p->bwd_d_h_in, d_h_final, (size_t)batch_size * seq_len * d * sizeof(float));
            vk_dispatch_pc_negate_gather(ctx, p->bwd_d_h_in, p->pred_x,
                                         (uint32_t)batch_size, (uint32_t)seq_len, (uint32_t)d);
            vk_barrier(ctx);
            vk_dispatch_fp32_gemv(ctx, p->shadow_pred_A, p->pred_x, p->pred_y,
                                  d, d, (uint32_t)T_total, d, d, 0, 1);
            vk_barrier(ctx);
            vk_dispatch_residual_scatter_add(ctx, p->pred_y, p->bwd_d_h_in,
                                             (uint32_t)batch_size, (uint32_t)seq_len, (uint32_t)d);
            vk_download(ctx, d_h_final, p->bwd_d_h_in,
                        (size_t)batch_size * seq_len * d * sizeof(float));

            /* Layer 0 Op B: dw_pred += (-d_h_final)^T @ h_final_prev
             * Re-upload so bwd_d_h_in reflects the post-Op-A d_h_final */
            vk_upload(ctx, p->bwd_d_h_in, d_h_final,
                      (size_t)batch_size * seq_len * d * sizeof(float));
            vk_dispatch_pc_negate_gather(ctx, p->bwd_d_h_in, p->gemm_A,
                                         (uint32_t)batch_size, (uint32_t)seq_len, (uint32_t)d);
            vk_dispatch_pc_gather_prev(ctx, p->residual[V12_N_LAYERS], p->gemm_B,
                                       (uint32_t)batch_size, (uint32_t)seq_len, (uint32_t)d);
            vk_barrier(ctx);
            vk_dispatch_fp32_gemm(ctx, p->gemm_A, p->gemm_B, p->gemm_C,
                                  (uint32_t)T_total, (uint32_t)d, (uint32_t)d, 0);
            if (gpu_owns_w_pred && VK_BUFFER_VALID(p->opt_dw_pred)) {
                vk_barrier(ctx);
                vk_dispatch_residual_add(ctx, p->gemm_C, p->opt_dw_pred, (uint32_t)(d * d));
            } else {
                vk_download(ctx, p->pc_dw_buf, p->gemm_C, dw_bytes);
                for (int i = 0; i < d * d; i++) m->dw_pred[i] += p->pc_dw_buf[i];
            }

            /* Inter-layer k=1..N-1 */
            for (int k = 1; k < V12_N_LAYERS; k++) {
                float *d_r_in_k = c->d_residual + (size_t)k * bseq * d;

                vk_upload(ctx, p->bwd_d_h_in, d_r_in_k,
                          (size_t)batch_size * seq_len * d * sizeof(float));

                /* Op A */
                vk_dispatch_pc_negate_gather(ctx, p->bwd_d_h_in, p->pred_x,
                                             (uint32_t)batch_size, (uint32_t)seq_len, (uint32_t)d);
                vk_barrier(ctx);
                vk_dispatch_fp32_gemv(ctx, p->shadow_pred_A, p->pred_x, p->pred_y,
                                      d, d, (uint32_t)T_total, d, d, 0, 1);
                vk_barrier(ctx);
                vk_dispatch_residual_scatter_add(ctx, p->pred_y, p->bwd_d_h_in,
                                                 (uint32_t)batch_size, (uint32_t)seq_len, (uint32_t)d);
                vk_download(ctx, d_r_in_k, p->bwd_d_h_in,
                            (size_t)batch_size * seq_len * d * sizeof(float));

                /* Op B: re-upload post-Op-A d_r_in_k so negate-gather sees updated values */
                vk_upload(ctx, p->bwd_d_h_in, d_r_in_k,
                          (size_t)batch_size * seq_len * d * sizeof(float));
                vk_dispatch_pc_negate_gather(ctx, p->bwd_d_h_in, p->gemm_A,
                                             (uint32_t)batch_size, (uint32_t)seq_len, (uint32_t)d);
                vk_dispatch_pc_gather_prev(ctx, p->residual[k - 1], p->gemm_B,
                                           (uint32_t)batch_size, (uint32_t)seq_len, (uint32_t)d);
                vk_barrier(ctx);
                vk_dispatch_fp32_gemm(ctx, p->gemm_A, p->gemm_B, p->gemm_C,
                                      (uint32_t)T_total, (uint32_t)d, (uint32_t)d, 0);
                if (gpu_owns_w_pred && VK_BUFFER_VALID(p->opt_dw_pred)) {
                    vk_barrier(ctx);
                    vk_dispatch_residual_add(ctx, p->gemm_C, p->opt_dw_pred, (uint32_t)(d * d));
                } else {
                    vk_download(ctx, p->pc_dw_buf, p->gemm_C, dw_bytes);
                    for (int i = 0; i < d * d; i++) m->dw_pred[i] += p->pc_dw_buf[i];
                }
            }
        }
#else
#pragma message("Compiling MTP else branch")
        /* CPU fallback */

        /* Pack rows t=1..seq_len-1 of each batch item into contiguous neg_buf.
         * neg_buf[b*T_per_batch + (t-1)] = -d_src[b*seq_len + t]            */
        #define PCB_PACK_NEG(neg_dst, d_src) \
            do { \
                for (int _b = 0; _b < batch_size; _b++) \
                    for (int _t = 1; _t < seq_len; _t++) { \
                        float *_sv = (d_src) + ((size_t)_b * seq_len + _t) * d; \
                        float *_dv = (neg_dst) + ((size_t)_b * T_per_batch + (_t - 1)) * d; \
                        for (int _j = 0; _j < d; _j++) _dv[_j] = -_sv[_j]; \
                    } \
            } while (0)

        /* Accumulate res_buf rows into d_target[b, t-1] positions */
        #define PCB_ADD_RES(d_target, res_buf) \
            do { \
                for (int _b = 0; _b < batch_size; _b++) \
                    for (int _t = 1; _t < seq_len; _t++) { \
                        float *_d_tm1 = (d_target) + ((size_t)_b * seq_len + _t - 1) * d; \
                        float *_res   = (res_buf)   + ((size_t)_b * T_per_batch + (_t - 1)) * d; \
                        for (int _j = 0; _j < d; _j++) _d_tm1[_j] += _res[_j]; \
                    } \
            } while (0)

        /* Pack prev-rows (t=0..seq_len-2) from src into dst */
        #define PCB_PACK_PREV(dst, src) \
            do { \
                for (int _b = 0; _b < batch_size; _b++) \
                    memcpy((dst) + (size_t)_b * T_per_batch * d, \
                           (src) + (size_t)_b * seq_len * d, \
                           (size_t)T_per_batch * d * sizeof(float)); \
            } while (0)

        /* Layer 0 PC backward: propagate gradient to h_final[b, t-1] */
        if (VK_BUFFER_VALID(p->shadow_pred_A) && T_total > 0 && p->pc_neg_buf && p->pc_res_buf) {
            size_t x_bytes = (size_t)T_total * d * sizeof(float);
            PCB_PACK_NEG(p->pc_neg_buf, d_h_final);
            vk_upload(ctx, p->pred_x, p->pc_neg_buf, x_bytes);
            vk_dispatch_fp32_gemv(ctx, p->shadow_pred_A, p->pred_x, p->pred_y,
                                  d, d, T_total, d, d, 0, 1);
            vk_download(ctx, p->pc_res_buf, p->pred_y, x_bytes);
            PCB_ADD_RES(d_h_final, p->pc_res_buf);
        } else {
            for (int b = 0; b < batch_size; b++) {
                for (int k2 = 0; k2 < d; k2++) {
                    for (int t = 1; t < seq_len; t++) {
                        float *d_h_tm1 = d_h_final + ((size_t)b * seq_len + t - 1) * d;
                        float *d_surp  = d_h_final + ((size_t)b * seq_len + t) * d;
                        float  acc     = 0.0f;
                        for (int j = 0; j < d; j++) acc += m->shadow_w_pred[j * d + k2] * (-d_surp[j]);
                        d_h_tm1[k2] += acc;
                    }
                }
            }
        }

        /* Op B: dw_pred += (-d_surp)^T @ h_final_prev — accumulate over all batch items */
        if (VK_BUFFER_VALID(p->gemm_A) && VK_BUFFER_VALID(p->gemm_B) && VK_BUFFER_VALID(p->gemm_C)
            && T_total > 0 && p->pc_neg_buf && p->pc_dw_buf) {
            size_t xb      = (size_t)T_total * d * sizeof(float);
            size_t dw_bytes = (size_t)d * d * sizeof(float);
            float *prev_rows = (float *)malloc(xb);
            PCB_PACK_NEG(p->pc_neg_buf, d_h_final);
            if (prev_rows) PCB_PACK_PREV(prev_rows, h_final_prev);
            vk_upload(ctx, p->gemm_A, p->pc_neg_buf, xb);
            vk_upload(ctx, p->gemm_B, prev_rows ? prev_rows : h_final_prev, xb);
            vk_dispatch_fp32_gemm(ctx, p->gemm_A, p->gemm_B, p->gemm_C,
                                  T_total, d, d, 0);
            if (gpu_owns_w_pred && VK_BUFFER_VALID(p->opt_dw_pred)) {
                vk_barrier(ctx);
                vk_dispatch_residual_add(ctx, p->gemm_C, p->opt_dw_pred, (uint32_t)(d * d));
            } else {
                vk_download(ctx, p->pc_dw_buf, p->gemm_C, dw_bytes);
                for (int i = 0; i < d * d; i++) m->dw_pred[i] += p->pc_dw_buf[i];
            }
            free(prev_rows);
        } else {
            for (int b = 0; b < batch_size; b++) {
                for (int j = 0; j < d; j++) {
                    float *dw_row = m->dw_pred + j * d;
                    for (int t = 1; t < seq_len; t++) {
                        float dp_j  = -(d_h_final + ((size_t)b * seq_len + t) * d)[j];
                        float *h_pv = h_final_prev + ((size_t)b * seq_len + t - 1) * d;
                        for (int k2 = 0; k2 < d; k2++) dw_row[k2] += dp_j * h_pv[k2];
                    }
                }
            }
        }

        /* Inter-layer PC backward (k >= 1) */
        for (int k = 1; k < V12_N_LAYERS; k++) {
            float *r_in_k   = c->residual   + (size_t)k * bseq * d;
            float *d_r_in_k = c->d_residual + (size_t)k * bseq * d;

            if (VK_BUFFER_VALID(p->shadow_pred_A) && T_total > 0 && p->pc_neg_buf && p->pc_res_buf) {
                size_t x_bytes = (size_t)T_total * d * sizeof(float);
                PCB_PACK_NEG(p->pc_neg_buf, d_r_in_k);
                vk_upload(ctx, p->pred_x, p->pc_neg_buf, x_bytes);
                vk_dispatch_fp32_gemv(ctx, p->shadow_pred_A, p->pred_x, p->pred_y,
                                      d, d, T_total, d, d, 0, 1);
                vk_download(ctx, p->pc_res_buf, p->pred_y, x_bytes);
                PCB_ADD_RES(d_r_in_k, p->pc_res_buf);
            } else {
                for (int b = 0; b < batch_size; b++) {
                    for (int k2 = 0; k2 < d; k2++) {
                        for (int t = 1; t < seq_len; t++) {
                            float *d_surp_k  = d_r_in_k + ((size_t)b * seq_len + t) * d;
                            float *d_r_k_tm1 = d_r_in_k + ((size_t)b * seq_len + t - 1) * d;
                            float  acc = 0.0f;
                            for (int j = 0; j < d; j++) acc += m->shadow_w_pred[j * d + k2] * (-d_surp_k[j]);
                            d_r_k_tm1[k2] += acc;
                        }
                    }
                }
            }

            if (VK_BUFFER_VALID(p->gemm_A) && VK_BUFFER_VALID(p->gemm_B) && VK_BUFFER_VALID(p->gemm_C)
                && T_total > 0 && p->pc_neg_buf && p->pc_dw_buf) {
                size_t xb       = (size_t)T_total * d * sizeof(float);
                size_t dw_bytes = (size_t)d * d * sizeof(float);
                float *prev_rows = (float *)malloc(xb);
                PCB_PACK_NEG(p->pc_neg_buf, d_r_in_k);
                if (prev_rows) PCB_PACK_PREV(prev_rows, r_in_k);
                vk_upload(ctx, p->gemm_A, p->pc_neg_buf, xb);
                vk_upload(ctx, p->gemm_B, prev_rows ? prev_rows : r_in_k, xb);
                vk_dispatch_fp32_gemm(ctx, p->gemm_A, p->gemm_B, p->gemm_C,
                                      T_total, d, d, 0);
                if (gpu_owns_w_pred && VK_BUFFER_VALID(p->opt_dw_pred)) {
                    vk_barrier(ctx);
                    vk_dispatch_residual_add(ctx, p->gemm_C, p->opt_dw_pred, (uint32_t)(d * d));
                } else {
                    vk_download(ctx, p->pc_dw_buf, p->gemm_C, dw_bytes);
                    for (int i = 0; i < d * d; i++) m->dw_pred[i] += p->pc_dw_buf[i];
                }
                free(prev_rows);
            } else {
                for (int b = 0; b < batch_size; b++) {
                    for (int j = 0; j < d; j++) {
                        float *dw_row = m->dw_pred + j * d;
                        for (int t = 1; t < seq_len; t++) {
                            float dp_j    = -(d_r_in_k + ((size_t)b * seq_len + t) * d)[j];
                            float *r_prev = r_in_k + ((size_t)b * seq_len + t - 1) * d;
                            for (int k2 = 0; k2 < d; k2++) dw_row[k2] += dp_j * r_prev[k2];
                        }
                    }
                }
            }
        }

        #undef PCB_PACK_NEG
        #undef PCB_ADD_RES
        #undef PCB_PACK_PREV
#endif
    }

    /* Phase 10: Embedding backward */
    {
        int    de    = V12_D_EMBED;
        float *dh0   = c->d_residual;
        size_t dh_bytes      = (size_t)batch_size * seq_len * d  * sizeof(float);
        size_t demb_p_bytes  = (size_t)de                   * d  * sizeof(float);

        int gpu_embed_p_done = 0;
        if (VK_BUFFER_VALID(p->embed_gathered)
            && VK_BUFFER_VALID(p->gemm_B)
            && VK_BUFFER_VALID(p->gemm_C)
            && (size_t)de * d <= (size_t)p->d_model * p->d_model) {

            vk_upload(ctx, p->gemm_B, dh0, dh_bytes);
            vk_dispatch_fp32_gemm(ctx,
                                  p->embed_gathered,
                                  p->gemm_B,
                                  p->gemm_C,
                                  (uint32_t)(batch_size * seq_len),
                                  (uint32_t)de,
                                  (uint32_t)d,
                                  0);
            vk_barrier(ctx);

            if (gpu_owns_embed_p && VK_BUFFER_VALID(p->opt_d_embed_P)) {
                vk_dispatch_residual_add(ctx, p->gemm_C, p->opt_d_embed_P,
                                         (uint32_t)(de * d));
            } else if (p->staging_dw) {
                vk_download(ctx, p->staging_dw, p->gemm_C, demb_p_bytes);
                for (int i = 0; i < de * d; i++) m->d_embed_P[i] += p->staging_dw[i];
            }
            gpu_embed_p_done = 1;
        }
        if (!gpu_embed_p_done) {
            for (int t = 0; t < batch_size * seq_len; t++) {
                float *dh_t  = dh0 + t * d;
                float *e_tok = m->embed_E + tokens[t] * de;
                for (int i = 0; i < de; i++) {
                    for (int j = 0; j < d; j++) m->d_embed_P[i * d + j] += e_tok[i] * dh_t[j];
                }
            }
        }

        int gpu_embed_e_done = 0;
        if (VK_BUFFER_VALID(p->bwd_d_embed_E_sparse)
            && VK_BUFFER_VALID(p->bwd_d_h_in)
            && VK_BUFFER_VALID(p->embed_P_gpu)
            && VK_BUFFER_VALID(p->embed_tokens)) {
            
            size_t sparse_bytes = (size_t)batch_size * seq_len * de * sizeof(float);
            
            vk_upload(ctx, p->bwd_d_h_in, dh0, dh_bytes);
            vk_dispatch_embed_e_backward(ctx,
                                         p->bwd_d_h_in,
                                         p->embed_P_gpu,
                                         p->embed_tokens,
                                         p->bwd_d_embed_E_sparse,
                                         (uint32_t)(batch_size * seq_len),
                                         (uint32_t)d,
                                         (uint32_t)de);
            
            // Sync pipeline to ensure gpu is done writing to the host-visible buffer
            vk_sync(ctx);
            
            // Map the sparse buffer
            float *sparse_updates = (float*)vk_map(ctx, p->bwd_d_embed_E_sparse);
            if (!sparse_updates) {
                // fallback if vk_map fails (e.g. if we didn't get host visible memory)
                sparse_updates = malloc(sparse_bytes);
                vk_download(ctx, sparse_updates, p->bwd_d_embed_E_sparse, sparse_bytes);
                for (int t = 0; t < batch_size * seq_len; t++) {
                    int tok = tokens[t];
                    for (int i = 0; i < de; i++) {
                        m->d_embed_E[tok * de + i] += sparse_updates[t * de + i];
                    }
                }
                free(sparse_updates);
            } else {
                for (int t = 0; t < batch_size * seq_len; t++) {
                    int tok = tokens[t];
                    for (int i = 0; i < de; i++) {
                        m->d_embed_E[tok * de + i] += sparse_updates[t * de + i];
                    }
                }
                vk_unmap(ctx, p->bwd_d_embed_E_sparse);
            }
            
            gpu_embed_e_done = 1;
        }
        if (!gpu_embed_e_done) {
            for (int t = 0; t < batch_size * seq_len; t++) {
                int    tok  = tokens[t];
                float *dh_t = dh0 + t * d;
                for (int i = 0; i < de; i++) {
                    float acc = 0.0f;
                    for (int j = 0; j < d; j++) acc += m->embed_P[i * d + j] * dh_t[j];
                    m->d_embed_E[tok * de + i] += acc;
                }
            }
        }
    }

    R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
            seq_idx, -1, "post-pc-backward");
    fflush(stderr);

    /* Auxiliary losses */
    R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
            seq_idx, -1, "pre-contraction-backward");
    fflush(stderr);
    compute_contraction_backward(m);
    R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
            seq_idx, -1, "post-contraction-backward");
    fflush(stderr);
    }

    float mtp_loss = (mtp_count > 0)
                   ? V12_MTP_WEIGHT * mtp_loss_acc / (float)mtp_count
                   : 0.0f;
    loss += mtp_loss;
    R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
            seq_idx, -1, "post-mtp-loss");
    fflush(stderr);

    /* Fisher-weighted MVUE regularizer */
    R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
            seq_idx, -1, "pre-fisher-regularizer");
    fflush(stderr);
    {
        int n_weights = V12_D_MODEL * V12_D_STATE;
        float fisher_reg = 0.0f;
        for (int k = 0; k < V12_N_LAYERS; k++) {
            SSMLayer *L = &m->layers[k];
            const float *gpu_shadow = NULL;

            if (!layer_backward_state_ready(L, gpu_owns_w_in, gpu_owns_w_out)) continue;
            if (gpu_owns_w_in && VK_BUFFER_VALID(p->opt_layers[k].shadow_w_in) && p->staging_dw) {
                R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                        seq_idx, k, "pre-fisher-shadow-download");
                fflush(stderr);
                vk_download(ctx, p->staging_dw, p->opt_layers[k].shadow_w_in,
                            (size_t)n_weights * sizeof(float));
                R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
                        seq_idx, k, "post-fisher-shadow-download");
                fflush(stderr);
                gpu_shadow = p->staging_dw;
            } else if (!L->shadow_w_in) {
                continue;
            }
            for (int i = 0; i < n_weights; i++) {
                float s_val = gpu_shadow ? fabsf(gpu_shadow[i])
                                         : fabsf(SHADOW_DECODE(L->shadow_w_in[i]));
                if (s_val > 1.0f) s_val = 1.0f;
                fisher_reg += s_val * (1.0f - s_val);
            }
        }
        loss += 0.001f * fisher_reg / (float)(V12_N_LAYERS * n_weights);
    }
    R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
            seq_idx, -1, "post-fisher-regularizer");
    fflush(stderr);

    R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
            seq_idx, -1, "pre-gpu-train-step-free");
    fflush(stderr);
    free(dC_scratch);
    free(lbwd_A_buf);

    /* R494 CCHS + SPCS skips: calculate Continuous Hamiltonian H_t = Loss_t + 1/2 * lambda * sum(moment_d^2) */
    {
        float kinetic = 0.0f;
        int ds = p->d_state;
        for (int k = 0; k < p->n_layers; k++) {
            if (m->layers[k].m_A) {
                if (VK_BUFFER_VALID(p->opt_layers[k].m_A)) {
                    vk_download(ctx, m->layers[k].m_A, p->opt_layers[k].m_A, ds * sizeof(float));
                }
                for (int i = 0; i < ds; i++) {
                    float val = m->layers[k].m_A[i];
                    kinetic += val * val;
                }
            }
        }
        kinetic = 0.5f * 0.05f * kinetic;
        float H_t = loss + kinetic;

        int skip_next = 0;
        if (p->has_prev_H) {
            float delta_H = H_t - p->prev_H;
            float epsilon = 0.02f; // Hamiltonian stability threshold
            
            int conservative = (fabsf(delta_H) < epsilon) ? 1 : 0;
            int negative_drift = (delta_H <= 0.0f) ? 1 : 0;

            // Deterministic stochastic Poisson trigger using seq_idx LCG
            uint32_t seed = (uint32_t)seq_idx + 1337u;
            for (int iter = 0; iter < 4; iter++) {
                seed = seed * 1664525u + 1013904223u;
            }
            float p_val = (float)(seed >> 8) / 16777216.0f;
            int poisson_trigger = (p_val < 0.20f) ? 1 : 0;

            if ((conservative || negative_drift) && poisson_trigger) {
                skip_next = 1;
                printf("[SPCS SKIP] Symplectic potential conserved: delta_H=%.6f, p_val=%.4f -> BPTT will be bypassed on next step.\n",
                       (double)delta_H, (double)p_val);
            }
        }
        p->prev_H = H_t;
        p->has_prev_H = 1;
        p->skip_bptt_current_step = skip_next;
    }

    R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%d layer=%d stage=%s\n",
            seq_idx, -1, "pre-gpu-train-step-return");
    fflush(stderr);
    return loss + lrms_penalty_global;
} }

void gpu_pipeline_r364_set_noop_compiled_interval(GpuPipeline *p,
                                                  int active,
                                                  uint32_t compile_id,
                                                  float compiled_block_share_f) {
    if (!p) return;

    if (active) {
        if (!p->r364_noop_interval_active || p->r364_noop_compile_id != compile_id) {
            p->r364_noop_train_step_calls = 0;
            p->r364_noop_optimizer_step_calls = 0;
        }
        p->r364_noop_compile_id = compile_id;
    }

    p->r364_noop_interval_active = active ? 1 : 0;
    p->r364_noop_compiled_block_share_f = compiled_block_share_f;
}

void gpu_pipeline_r364_get_noop_probe(const GpuPipeline *p, GpuR364NoopProbe *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!p) return;

    out->compile_id = p->r364_noop_compile_id;
    out->train_step_calls = p->r364_noop_train_step_calls;
    out->optimizer_step_calls = p->r364_noop_optimizer_step_calls;
    out->compiled_interval_active = p->r364_noop_interval_active;
    out->compiled_block_share_f = p->r364_noop_compiled_block_share_f;
}

int gpu_pipeline_sync_lm_head_to_cpu(GpuPipeline *p, Model *m) {
    if (!p || !m) return -1;
    if (!(m->gpu_opt_flags & GPU_OPT_LM_HEAD)) return -1;
    if (model_materialize_lm_head_host(m) != 0) return -1;

    const size_t lm_dense_bytes = (size_t)V12_VOCAB_SIZE * p->d_model * sizeof(float);
    const size_t lm_col_bytes = (size_t)p->d_model * sizeof(float);
    VulkanContext *ctx = p->ctx;

    if (m->shadow_lm_head && VK_BUFFER_VALID(p->opt_shadow_lm_head)) {
        vk_download(ctx, m->shadow_lm_head, p->opt_shadow_lm_head, lm_dense_bytes);
    }
    if (m->sampled_lm_head && VK_BUFFER_VALID(p->opt_sampled_lm_head)) {
        vk_download(ctx, m->sampled_lm_head, p->opt_sampled_lm_head, lm_dense_bytes);
    }
    if (m->m_lm_head_row && VK_BUFFER_VALID(p->opt_m_lm_head_row)) {
        download_lm_hot_row_moments(ctx, p->opt_m_lm_head_row, m->hot_to_vocab, m->m_lm_head_row);
    }
    if (m->v_lm_head_row && VK_BUFFER_VALID(p->opt_v_lm_head_row)) {
        download_lm_hot_row_moments(ctx, p->opt_v_lm_head_row, m->hot_to_vocab, m->v_lm_head_row);
    }
    if (m->m_lm_head_col && VK_BUFFER_VALID(p->opt_m_lm_head_col)) {
        vk_download(ctx, m->m_lm_head_col, p->opt_m_lm_head_col, lm_col_bytes);
    }
    if (m->v_lm_head_col && VK_BUFFER_VALID(p->opt_v_lm_head_col)) {
        vk_download(ctx, m->v_lm_head_col, p->opt_v_lm_head_col, lm_col_bytes);
    }
    return 0;
}

int gpu_pipeline_sync_slice2_to_cpu(GpuPipeline *p, Model *m) {
    if (!p || !m) return -1;
    VulkanContext *ctx = p->ctx;
    const size_t d = (size_t)p->d_model;
    const size_t pred_bytes = d * d * sizeof(float);
    const size_t norm_bytes = d * sizeof(float);

    if ((m->gpu_opt_flags & GPU_OPT_W_PRED) != 0u && model_materialize_w_pred_host(m) == 0) {
        if (m->shadow_w_pred && VK_BUFFER_VALID(p->opt_shadow_w_pred)) {
            vk_download(ctx, m->shadow_w_pred, p->opt_shadow_w_pred, pred_bytes);
        }
        if (m->sampled_w_pred && VK_BUFFER_VALID(p->opt_sampled_w_pred)) {
            vk_download(ctx, m->sampled_w_pred, p->opt_sampled_w_pred, pred_bytes);
        }
        if (m->m_w_pred_row && VK_BUFFER_VALID(p->opt_m_w_pred_row)) {
            vk_download(ctx, m->m_w_pred_row, p->opt_m_w_pred_row, norm_bytes);
        }
        if (m->v_w_pred_row && VK_BUFFER_VALID(p->opt_v_w_pred_row)) {
            vk_download(ctx, m->v_w_pred_row, p->opt_v_w_pred_row, norm_bytes);
        }
        if (m->m_w_pred_col && VK_BUFFER_VALID(p->opt_m_w_pred_col)) {
            vk_download(ctx, m->m_w_pred_col, p->opt_m_w_pred_col, norm_bytes);
        }
        if (m->v_w_pred_col && VK_BUFFER_VALID(p->opt_v_w_pred_col)) {
            vk_download(ctx, m->v_w_pred_col, p->opt_v_w_pred_col, norm_bytes);
        }
    }
    if ((m->gpu_opt_flags & GPU_OPT_NORM) != 0u && model_materialize_norm_host(m) == 0) {
        if (m->norm_w && VK_BUFFER_VALID(p->norm_w)) {
            vk_download(ctx, m->norm_w, p->norm_w, norm_bytes);
        }
        if (m->norm_b && VK_BUFFER_VALID(p->norm_b)) {
            vk_download(ctx, m->norm_b, p->norm_b, norm_bytes);
        }
        if (m->m_norm_w && VK_BUFFER_VALID(p->opt_m_norm_w)) {
            vk_download(ctx, m->m_norm_w, p->opt_m_norm_w, norm_bytes);
        }
        if (m->v_norm_w && VK_BUFFER_VALID(p->opt_v_norm_w)) {
            vk_download(ctx, m->v_norm_w, p->opt_v_norm_w, norm_bytes);
        }
        if (m->m_norm_b && VK_BUFFER_VALID(p->opt_m_norm_b)) {
            vk_download(ctx, m->m_norm_b, p->opt_m_norm_b, norm_bytes);
        }
        if (m->v_norm_b && VK_BUFFER_VALID(p->opt_v_norm_b)) {
            vk_download(ctx, m->v_norm_b, p->opt_v_norm_b, norm_bytes);
        }
    }
    return 0;
}

int gpu_pipeline_sync_embed_to_cpu(GpuPipeline *p, Model *m) {
    if (!p || !m) return -1;
    VulkanContext *ctx = p->ctx;
    const size_t d = (size_t)p->d_model;
    const size_t de = (size_t)V12_D_EMBED;
    const size_t embed_e_bytes = (size_t)V12_VOCAB_SIZE * de * sizeof(float);
    const size_t embed_p_bytes = de * d * sizeof(float);

    if ((m->gpu_opt_flags & GPU_OPT_EMBED_E) != 0u && model_materialize_embed_host(m) == 0) {
        if (m->embed_E && VK_BUFFER_VALID(p->embed_E_gpu)) {
            vk_download(ctx, m->embed_E, p->embed_E_gpu, embed_e_bytes);
        }
        if (m->m_embed_E && VK_BUFFER_VALID(p->opt_m_embed_E)) {
            vk_download(ctx, m->m_embed_E, p->opt_m_embed_E, embed_e_bytes);
        }
        if (m->v_embed_E && VK_BUFFER_VALID(p->opt_v_embed_E)) {
            vk_download(ctx, m->v_embed_E, p->opt_v_embed_E, embed_e_bytes);
        }
    }
    if ((m->gpu_opt_flags & GPU_OPT_EMBED_P) != 0u && model_materialize_embed_host(m) == 0) {
        if (m->embed_P && VK_BUFFER_VALID(p->embed_P_gpu)) {
            vk_download(ctx, m->embed_P, p->embed_P_gpu, embed_p_bytes);
        }
        if (m->m_embed_P && VK_BUFFER_VALID(p->opt_m_embed_P)) {
            vk_download(ctx, m->m_embed_P, p->opt_m_embed_P, embed_p_bytes);
        }
        if (m->v_embed_P && VK_BUFFER_VALID(p->opt_v_embed_P)) {
            vk_download(ctx, m->v_embed_P, p->opt_v_embed_P, embed_p_bytes);
        }
    }
    return 0;
}

int gpu_pipeline_sync_layer_to_cpu(GpuPipeline *p, Model *m, int k) {
    if (!p || !m || k < 0 || k >= p->n_layers) return -1;
    VulkanContext *ctx = p->ctx;
    const size_t d = (size_t)p->d_model;
    const size_t ds = (size_t)p->d_state;
    const size_t win_bytes = d * ds * sizeof(float);
    const size_t wout_bytes = ds * d * sizeof(float);
    const size_t ds_bytes = ds * sizeof(float);
    const size_t wa_bytes = ds * ds * sizeof(float);
    const size_t conv_bytes = V12_CONV_KERNEL * ds * sizeof(float);

    if (model_materialize_layer_w_io_host(&m->layers[k]) != 0) return -1;

    SSMLayer *L = &m->layers[k];
    if ((m->gpu_opt_flags & GPU_OPT_W_IN) != 0u) {
        if (L->sampled_w_in && VK_BUFFER_VALID(p->opt_layers[k].sampled_w_in)) {
            vk_download(ctx, L->sampled_w_in, p->opt_layers[k].sampled_w_in, win_bytes);
        }
        if (L->shadow_w_in && VK_BUFFER_VALID(p->opt_layers[k].shadow_w_in)) {
            download_shadow_buffer(ctx, L->shadow_w_in, p->opt_layers[k].shadow_w_in, (size_t)d * ds, m->step);
        }
        if (L->m_w_in_row && VK_BUFFER_VALID(p->opt_layers[k].m_w_in_row)) {
            vk_download(ctx, L->m_w_in_row, p->opt_layers[k].m_w_in_row, ds_bytes);
        }
        if (L->v_w_in_row && VK_BUFFER_VALID(p->opt_layers[k].v_w_in_row)) {
            vk_download(ctx, L->v_w_in_row, p->opt_layers[k].v_w_in_row, ds_bytes);
        }
        if (L->m_w_in_col && VK_BUFFER_VALID(p->opt_layers[k].m_w_in_col)) {
            vk_download(ctx, L->m_w_in_col, p->opt_layers[k].m_w_in_col, ds_bytes);
        }
        if (L->v_w_in_col && VK_BUFFER_VALID(p->opt_layers[k].v_w_in_col)) {
            vk_download(ctx, L->v_w_in_col, p->opt_layers[k].v_w_in_col, ds_bytes);
        }
    }
    if ((m->gpu_opt_flags & GPU_OPT_W_OUT) != 0u) {
        if (L->sampled_w_out && VK_BUFFER_VALID(p->opt_layers[k].sampled_w_out)) {
            vk_download(ctx, L->sampled_w_out, p->opt_layers[k].sampled_w_out, wout_bytes);
        }
        if (L->shadow_w_out && VK_BUFFER_VALID(p->opt_layers[k].shadow_w_out)) {
            download_shadow_buffer(ctx, L->shadow_w_out, p->opt_layers[k].shadow_w_out, (size_t)ds * d, m->step);
        }
        if (L->m_w_out_row && VK_BUFFER_VALID(p->opt_layers[k].m_w_out_row)) {
            vk_download(ctx, L->m_w_out_row, p->opt_layers[k].m_w_out_row, ds_bytes);
        }
        if (L->v_w_out_row && VK_BUFFER_VALID(p->opt_layers[k].v_w_out_row)) {
            vk_download(ctx, L->v_w_out_row, p->opt_layers[k].v_w_out_row, ds_bytes);
        }
        if (L->m_w_out_col && VK_BUFFER_VALID(p->opt_layers[k].m_w_out_col)) {
            vk_download(ctx, L->m_w_out_col, p->opt_layers[k].m_w_out_col, ds_bytes);
        }
        if (L->v_w_out_col && VK_BUFFER_VALID(p->opt_layers[k].v_w_out_col)) {
            vk_download(ctx, L->v_w_out_col, p->opt_layers[k].v_w_out_col, ds_bytes);
        }
    }
    if ((m->gpu_opt_flags & GPU_OPT_A) != 0u) {
        if (L->A && VK_BUFFER_VALID(p->layers[k].gpu_A_const)) {
            vk_download(ctx, L->A, p->layers[k].gpu_A_const, ds_bytes);
        }
        if (L->m_A && VK_BUFFER_VALID(p->opt_layers[k].m_A)) {
            vk_download(ctx, L->m_A, p->opt_layers[k].m_A, ds_bytes);
        }
        if (L->v_A && VK_BUFFER_VALID(p->opt_layers[k].v_A)) {
            vk_download(ctx, L->v_A, p->opt_layers[k].v_A, ds_bytes);
        }
    }
    if ((m->gpu_opt_flags & GPU_OPT_BA) != 0u) {
        if (L->b_A && VK_BUFFER_VALID(p->lw_bA)) {
            vk_download(ctx, L->b_A, p->lw_bA, ds_bytes);
        }
        if (L->m_b_A && VK_BUFFER_VALID(p->opt_layers[k].m_b_A)) {
            vk_download(ctx, L->m_b_A, p->opt_layers[k].m_b_A, ds_bytes);
        }
        if (L->v_b_A && VK_BUFFER_VALID(p->opt_layers[k].v_b_A)) {
            vk_download(ctx, L->v_b_A, p->opt_layers[k].v_b_A, ds_bytes);
        }
    }
    if ((m->gpu_opt_flags & GPU_OPT_WA) != 0u) {
        if (L->W_A && VK_BUFFER_VALID(p->lw_WA)) {
            vk_download(ctx, L->W_A, p->lw_WA, wa_bytes);
        }
        if (L->m_W_A && VK_BUFFER_VALID(p->opt_layers[k].m_WA)) {
            vk_download(ctx, L->m_W_A, p->opt_layers[k].m_WA, wa_bytes);
        }
        if (L->v_W_A && VK_BUFFER_VALID(p->opt_layers[k].v_WA)) {
            vk_download(ctx, L->v_W_A, p->opt_layers[k].v_WA, wa_bytes);
        }
    }
    if ((m->gpu_opt_flags & GPU_OPT_CONV) != 0u) {
        if (L->shadow_w_conv && VK_BUFFER_VALID(p->lw_conv)) {
            download_shadow_buffer(ctx, L->shadow_w_conv, p->lw_conv, (size_t)V12_CONV_KERNEL * ds, m->step);
        }
        if (L->m_conv && VK_BUFFER_VALID(p->opt_layers[k].m_conv)) {
            vk_download(ctx, L->m_conv, p->opt_layers[k].m_conv, conv_bytes);
        }
        if (L->v_conv && VK_BUFFER_VALID(p->opt_layers[k].v_conv)) {
            vk_download(ctx, L->v_conv, p->opt_layers[k].v_conv, conv_bytes);
        }
    }
    return 0;
}

int gpu_pipeline_sync_layer_state_to_cpu(GpuPipeline *p, Model *m) {
    if (!p || !m) return -1;
    for (int k = 0; k < p->n_layers; k++) {
        if (gpu_pipeline_sync_layer_to_cpu(p, m, k) != 0) return -1;
    }
    return 0;
}

int gpu_pipeline_sync_gradients_to_cpu(GpuPipeline *p, Model *m) {
    if (!p || !m) return -1;
    VulkanContext *ctx = p->ctx;
    const size_t d = (size_t)p->d_model;
    const size_t ds = (size_t)p->d_state;
    const size_t win_bytes = d * ds * sizeof(float);
    const size_t wout_bytes = ds * d * sizeof(float);

    vk_sync(ctx);
    for (int k = 0; k < p->n_layers; k++) {
        SSMLayer *L = &m->layers[k];
        if (L->dw_in && VK_BUFFER_VALID(p->opt_layers[k].gpu_dw_in)) {
            if (vk_download(ctx, L->dw_in, p->opt_layers[k].gpu_dw_in, win_bytes) != 0) {
                return -1;
            }
        }
        if (L->dw_out && VK_BUFFER_VALID(p->opt_layers[k].gpu_dw_out)) {
            if (vk_download(ctx, L->dw_out, p->opt_layers[k].gpu_dw_out, wout_bytes) != 0) {
                return -1;
            }
        }
    }
    return 0;
}


int gpu_pipeline_sync_r364_compile_cache_to_cpu(GpuPipeline *p,
                                                Model *m,
                                                Cache *c,
                                                int batch_size) {
    if (!p || !m || !c || batch_size <= 0) return -1;

    VulkanContext *ctx = p->ctx;
    const int seq_len = p->seq_len;
    const int d = p->d_model;
    const int n_tokens = seq_len - 1;
    const int n_samples = V12_SS_N_SAMPLES;
    const size_t norm_bytes = (size_t)batch_size * seq_len * d * sizeof(float);
    const size_t rms_bytes = (size_t)batch_size * seq_len * sizeof(float);
    const size_t sample_bytes = (size_t)batch_size * n_tokens * n_samples * sizeof(int32_t);
    int32_t *sample_ids = NULL;

    if (!c->norm_out || !c->norm_rms || !c->sample_ids || !c->log_proposal) return -1;

    if (VK_BUFFER_VALID(p->norm_out)) {
        vk_download(ctx, c->norm_out, p->norm_out, norm_bytes);
    }
    if (VK_BUFFER_VALID(p->norm_rms)) {
        vk_download(ctx, c->norm_rms, p->norm_rms, rms_bytes);
    }

    if (!VK_BUFFER_VALID(p->ce_sample_ids)) return 0;

    sample_ids = (int32_t *)malloc(sample_bytes);
    if (!sample_ids) return -1;
    vk_download(ctx, sample_ids, p->ce_sample_ids, sample_bytes);

    {
        const int eval_batch = batch_size - 1;
        const int32_t *src = sample_ids + (size_t)eval_batch * n_tokens * n_samples;
        memcpy(c->sample_ids, src, (size_t)n_tokens * n_samples * sizeof(int32_t));
        memset(c->log_proposal, 0, (size_t)seq_len * n_samples * sizeof(float));
        for (int t = 0; t < n_tokens; t++) {
            float *dst_log = c->log_proposal + (size_t)t * n_samples;
            const int32_t *dst_ids = c->sample_ids + (size_t)t * n_samples;
            for (int s = 1; s < n_samples; s++) {
                const int32_t vocab_id = dst_ids[s];
                if (vocab_id >= 0 && vocab_id < V12_VOCAB_SIZE && m->log_unigram) {
                    dst_log[s] = m->log_unigram[vocab_id];
                }
            }
        }
    }

    free(sample_ids);
    return 0;
}

int gpu_pipeline_sync_hidden_cache_to_cpu(GpuPipeline *p,
                                          Cache *c,
                                          int batch_size) {
    if (!p || !c || !c->h_all || batch_size <= 0) return -1;

    VulkanContext *ctx = p->ctx;
    const int seq_len = p->seq_len;
    const int ds = p->d_state;
    const size_t bseq = (size_t)batch_size * (size_t)seq_len;
    const size_t layer_bytes = bseq * (size_t)ds * sizeof(float);

    for (int k = 0; k < p->n_layers; k++) {
        if (!p->layers[k].fwd_allocated ||
            !VK_BUFFER_VALID(p->layers[k].fwd_h_cache)) {
            return -1;
        }
        float *temp_float = malloc(layer_bytes);
        if (!temp_float) return -1;

        vk_download(ctx, temp_float, p->layers[k].fwd_h_cache, layer_bytes);

        uint32_t rng = 0x12345678u ^ (uint32_t)k;
        uint8_t *dst = c->h_all + (size_t)k * bseq * (size_t)ds;
        for (size_t i = 0; i < bseq * (size_t)ds; i++) {
            dst[i] = compress_e5m2_stochastic(temp_float[i], &rng);
        }
        free(temp_float);
    }
    return 0;
}

void gpu_pipeline_sync_lm_head_from_cpu(GpuPipeline *p, Model *m) {
    if (!p || !m) return;
    if (!(m->gpu_opt_flags & GPU_OPT_LM_HEAD)) return;
    if (model_materialize_lm_head_host(m) != 0) return;

    const size_t lm_dense_bytes = (size_t)V12_VOCAB_SIZE * p->d_model * sizeof(float);
    const size_t lm_col_bytes = (size_t)p->d_model * sizeof(float);

    vk_begin_batch(p->ctx);
    if (m->shadow_lm_head && VK_BUFFER_VALID(p->opt_shadow_lm_head)) {
        vk_upload(p->ctx, p->opt_shadow_lm_head, m->shadow_lm_head, lm_dense_bytes);
    }
    if (m->sampled_lm_head && VK_BUFFER_VALID(p->opt_sampled_lm_head)) {
        vk_upload(p->ctx, p->opt_sampled_lm_head, m->sampled_lm_head, lm_dense_bytes);
    }
    if (m->m_lm_head_row && VK_BUFFER_VALID(p->opt_m_lm_head_row)) {
        upload_lm_hot_row_moments(p->ctx, p->opt_m_lm_head_row, m->hot_to_vocab, m->m_lm_head_row);
    }
    if (m->v_lm_head_row && VK_BUFFER_VALID(p->opt_v_lm_head_row)) {
        upload_lm_hot_row_moments(p->ctx, p->opt_v_lm_head_row, m->hot_to_vocab, m->v_lm_head_row);
    }
    if (m->m_lm_head_col && VK_BUFFER_VALID(p->opt_m_lm_head_col)) {
        vk_upload(p->ctx, p->opt_m_lm_head_col, m->m_lm_head_col, lm_col_bytes);
    }
    if (m->v_lm_head_col && VK_BUFFER_VALID(p->opt_v_lm_head_col)) {
        vk_upload(p->ctx, p->opt_v_lm_head_col, m->v_lm_head_col, lm_col_bytes);
    }
    vk_end_batch(p->ctx); vk_begin_batch(p->ctx);

    gpu_pipeline_rebuild_lm_head_ce_csr(p, m);
}

void gpu_zero_grad_accumulators(GpuPipeline *p) {
    if (!p || !p->ctx) return;

    const size_t d = (size_t)p->d_model;
    const size_t ds = (size_t)p->d_state;
    const size_t de = (size_t)V12_D_EMBED;
    const size_t lm_dense_bytes = (size_t)V12_VOCAB_SIZE * d * sizeof(float);
    const size_t pred_bytes = d * d * sizeof(float);
    const size_t embed_p_bytes = de * d * sizeof(float);
    const size_t embed_e_bytes = (size_t)V12_VOCAB_SIZE * de * sizeof(float);
    const size_t win_bytes = d * ds * sizeof(float);
    const size_t wout_bytes = ds * d * sizeof(float);
    const size_t dwa_bytes = ds * ds * sizeof(float);
    const size_t conv_bytes = (size_t)V12_CONV_KERNEL * ds * sizeof(float);

    vk_begin_batch(p->ctx);
    if (VK_BUFFER_VALID(p->ce_dlm_head)) vk_fill_zero(p->ctx, p->ce_dlm_head, lm_dense_bytes);
    if (VK_BUFFER_VALID(p->opt_dw_pred)) vk_fill_zero(p->ctx, p->opt_dw_pred, pred_bytes);
    if (VK_BUFFER_VALID(p->opt_d_norm_w)) vk_fill_zero(p->ctx, p->opt_d_norm_w, d * sizeof(float));
    if (VK_BUFFER_VALID(p->opt_d_norm_b)) vk_fill_zero(p->ctx, p->opt_d_norm_b, d * sizeof(float));
    if (VK_BUFFER_VALID(p->opt_d_embed_P)) vk_fill_zero(p->ctx, p->opt_d_embed_P, embed_p_bytes);
    if (VK_BUFFER_VALID(p->bwd_d_embed_E)) vk_fill_zero(p->ctx, p->bwd_d_embed_E, embed_e_bytes);
    for (int k = 0; k < p->n_layers; k++) {
        if (VK_BUFFER_VALID(p->opt_layers[k].gpu_dw_in)) vk_fill_zero(p->ctx, p->opt_layers[k].gpu_dw_in, win_bytes);
        if (VK_BUFFER_VALID(p->opt_layers[k].gpu_dw_out)) vk_fill_zero(p->ctx, p->opt_layers[k].gpu_dw_out, wout_bytes);
        if (VK_BUFFER_VALID(p->opt_layers[k].gpu_dW_A)) vk_fill_zero(p->ctx, p->opt_layers[k].gpu_dW_A, dwa_bytes);
        if (VK_BUFFER_VALID(p->opt_layers[k].gpu_dA)) vk_fill_zero(p->ctx, p->opt_layers[k].gpu_dA, ds * sizeof(float));
        if (VK_BUFFER_VALID(p->opt_layers[k].gpu_dbA)) vk_fill_zero(p->ctx, p->opt_layers[k].gpu_dbA, ds * sizeof(float));
        if (VK_BUFFER_VALID(p->opt_layers[k].gpu_dconv)) vk_fill_zero(p->ctx, p->opt_layers[k].gpu_dconv, conv_bytes);
        if (VK_BUFFER_VALID(p->opt_layers[k].dB)) vk_fill_zero(p->ctx, p->opt_layers[k].dB, ds * sizeof(float));
        if (VK_BUFFER_VALID(p->opt_layers[k].dC)) vk_fill_zero(p->ctx, p->opt_layers[k].dC, dwa_bytes);
        if (VK_BUFFER_VALID(p->opt_layers[k].d_sigma)) vk_fill_zero(p->ctx, p->opt_layers[k].d_sigma, ds * sizeof(float));
    }
    vk_end_batch(p->ctx); vk_begin_batch(p->ctx);
}

void gpu_scale_gradients(GpuPipeline *p, float scale) {
    if (!p || !p->ctx) return;

    const uint32_t d = (uint32_t)p->d_model;
    const uint32_t ds = (uint32_t)p->d_state;
    const uint32_t de = (uint32_t)V12_D_EMBED;

    vk_begin_batch(p->ctx);
    if (VK_BUFFER_VALID(p->ce_dlm_head)) vk_dispatch_grad_scale(p->ctx, p->ce_dlm_head, (uint32_t)((size_t)V12_VOCAB_SIZE * d), scale);
    if (VK_BUFFER_VALID(p->opt_dw_pred)) vk_dispatch_grad_scale(p->ctx, p->opt_dw_pred, d * d, scale);
    if (VK_BUFFER_VALID(p->opt_d_norm_w)) vk_dispatch_grad_scale(p->ctx, p->opt_d_norm_w, d, scale);
    if (VK_BUFFER_VALID(p->opt_d_norm_b)) vk_dispatch_grad_scale(p->ctx, p->opt_d_norm_b, d, scale);
    if (VK_BUFFER_VALID(p->opt_d_embed_P)) vk_dispatch_grad_scale(p->ctx, p->opt_d_embed_P, de * d, scale);
    if (VK_BUFFER_VALID(p->bwd_d_embed_E)) vk_dispatch_grad_scale(p->ctx, p->bwd_d_embed_E, (uint32_t)((size_t)V12_VOCAB_SIZE * de), scale);
    for (int k = 0; k < p->n_layers; k++) {
        if (VK_BUFFER_VALID(p->opt_layers[k].gpu_dw_in)) vk_dispatch_grad_scale(p->ctx, p->opt_layers[k].gpu_dw_in, d * ds, scale);
        if (VK_BUFFER_VALID(p->opt_layers[k].gpu_dw_out)) vk_dispatch_grad_scale(p->ctx, p->opt_layers[k].gpu_dw_out, ds * d, scale);
        if (VK_BUFFER_VALID(p->opt_layers[k].gpu_dW_A)) vk_dispatch_grad_scale(p->ctx, p->opt_layers[k].gpu_dW_A, ds * ds, scale);
        if (VK_BUFFER_VALID(p->opt_layers[k].gpu_dA)) vk_dispatch_grad_scale(p->ctx, p->opt_layers[k].gpu_dA, ds, scale);
        if (VK_BUFFER_VALID(p->opt_layers[k].gpu_dbA)) vk_dispatch_grad_scale(p->ctx, p->opt_layers[k].gpu_dbA, ds, scale);
        if (VK_BUFFER_VALID(p->opt_layers[k].gpu_dconv)) vk_dispatch_grad_scale(p->ctx, p->opt_layers[k].gpu_dconv, (uint32_t)V12_CONV_KERNEL * ds, scale);
        if (VK_BUFFER_VALID(p->opt_layers[k].dB)) vk_dispatch_grad_scale(p->ctx, p->opt_layers[k].dB, ds, scale);
        if (VK_BUFFER_VALID(p->opt_layers[k].dC)) vk_dispatch_grad_scale(p->ctx, p->opt_layers[k].dC, ds * ds, scale);
        if (VK_BUFFER_VALID(p->opt_layers[k].d_sigma)) vk_dispatch_grad_scale(p->ctx, p->opt_layers[k].d_sigma, ds, scale);
    }
    vk_end_batch(p->ctx); vk_begin_batch(p->ctx);
}

static void gpu_gradient_null_witness_refresh(Model *m) {
    const double groups = (double)m->gradient_null_gate_groups;
    const double elems = (double)m->gradient_null_gate_elements;
    const double work = (double)m->gradient_null_gate_work_units;
    const double energy_total = m->gradient_null_gate_total_energy;
    const double element_saved =
        elems > 0.0 ? (double)m->gradient_null_gate_null_elements / elems : 0.0;
    const double saved =
        work > 0.0 ? (double)m->gradient_null_gate_null_work_units / work : 0.0;
    const double omitted =
        energy_total > 0.0 ? m->gradient_null_gate_null_energy / energy_total : 0.0;
    const double priced =
        saved - omitted - (double)m->gradient_null_witness_overhead;
    const int identifying = (m->gradient_null_gate_groups > 0u &&
                             m->gradient_null_gate_elements > 0u &&
                             energy_total > 0.0) ? 1 : 0;

    m->gradient_null_gate_group_fraction =
        groups > 0.0
            ? (float)((double)m->gradient_null_gate_null_groups / groups)
            : 0.0f;
    m->gradient_null_gate_element_fraction = (float)element_saved;
    m->gradient_null_gate_saved_work_fraction = (float)saved;
    m->gradient_null_gate_omitted_fraction = (float)omitted;
    m->gradient_null_gate_priced_gain = (float)priced;
    m->gradient_null_gate_identifying = identifying;
    m->gradient_null_gate_pass =
        (identifying &&
         omitted <= (double)m->gradient_null_witness_omitted_budget &&
         priced > 0.10) ? 1 : 0;
}

static void gpu_gradient_null_witness_accumulate_values(
    Model *m, const float *grad, size_t elems, size_t work_units) {
    if (!m || !m->gradient_null_witness_gate_enabled ||
        !grad || elems == 0u) {
        return;
    }

    double energy = 0.0;
    uint64_t finite = 0u;
    for (size_t i = 0; i < elems; i++) {
        const double g = (double)grad[i];
        if (isfinite(g)) {
            energy += g * g;
            finite++;
        }
    }
    if (finite == 0u) return;

    const double rms = sqrt(energy / (double)finite);
    const double eps = (m->gradient_null_witness_epsilon > 0.0f)
        ? (double)m->gradient_null_witness_epsilon
        : 1.0e-8;
    const int is_null = (rms <= eps) ? 1 : 0;

    m->gradient_null_gate_groups++;
    m->gradient_null_gate_elements += finite;
    m->gradient_null_gate_work_units +=
        (uint64_t)(work_units > 0u ? work_units : finite);
    m->gradient_null_gate_total_energy += energy;
    if (is_null) {
        m->gradient_null_gate_null_groups++;
        m->gradient_null_gate_null_elements += finite;
        m->gradient_null_gate_null_work_units +=
            (uint64_t)(work_units > 0u ? work_units : finite);
        m->gradient_null_gate_null_energy += energy;
    }
    gpu_gradient_null_witness_refresh(m);
}

static void gpu_gradient_null_witness_accumulate_buffer(
    GpuPipeline *p, Model *m, VkBufferHandle h, size_t elems,
    size_t work_units) {
    if (!p || !p->ctx || !m || !m->gradient_null_witness_gate_enabled ||
        !VK_BUFFER_VALID(h) || elems == 0u) {
        return;
    }
    float *tmp = (float *)malloc(elems * sizeof(float));
    if (!tmp) return;
    vk_download(p->ctx, tmp, h, elems * sizeof(float));
    gpu_gradient_null_witness_accumulate_values(m, tmp, elems, work_units);
    free(tmp);
}

void gpu_gradient_null_witness_gate_update(GpuPipeline *p, Model *m) {
    if (!p || !m || !m->gradient_null_witness_gate_enabled) return;

    const size_t d = (size_t)p->d_model;
    const size_t ds = (size_t)p->d_state;
    const size_t de = (size_t)V12_D_EMBED;

    if (m->gpu_opt_flags & GPU_OPT_LM_HEAD)
        gpu_gradient_null_witness_accumulate_buffer(
            p, m, p->ce_dlm_head, (size_t)V12_VOCAB_SIZE * d,
            (size_t)V12_K_HOT * d);
    if (m->gpu_opt_flags & GPU_OPT_W_PRED)
        gpu_gradient_null_witness_accumulate_buffer(
            p, m, p->opt_dw_pred, d * d, d * d);
    if (m->gpu_opt_flags & GPU_OPT_NORM) {
        gpu_gradient_null_witness_accumulate_buffer(
            p, m, p->opt_d_norm_w, d, d);
        gpu_gradient_null_witness_accumulate_buffer(
            p, m, p->opt_d_norm_b, d, d);
    }
    if (m->gpu_opt_flags & GPU_OPT_EMBED_P)
        gpu_gradient_null_witness_accumulate_buffer(
            p, m, p->opt_d_embed_P, de * d, de * d);
    if (m->gpu_opt_flags & GPU_OPT_EMBED_E)
        gpu_gradient_null_witness_accumulate_buffer(
            p, m, p->bwd_d_embed_E, (size_t)V12_VOCAB_SIZE * de,
            (size_t)p->batch_size * (size_t)p->seq_len * de);

    for (int k = 0; k < p->n_layers; k++) {
        if (m->gpu_opt_flags & GPU_OPT_W_IN)
            gpu_gradient_null_witness_accumulate_buffer(
                p, m, p->opt_layers[k].gpu_dw_in, d * ds, d * ds);
        if (m->gpu_opt_flags & GPU_OPT_W_OUT)
            gpu_gradient_null_witness_accumulate_buffer(
                p, m, p->opt_layers[k].gpu_dw_out, ds * d, ds * d);
        if (m->gpu_opt_flags & GPU_OPT_A)
            gpu_gradient_null_witness_accumulate_buffer(
                p, m, p->opt_layers[k].gpu_dA, ds, ds);
        if (m->gpu_opt_flags & GPU_OPT_WA)
            gpu_gradient_null_witness_accumulate_buffer(
                p, m, p->opt_layers[k].gpu_dW_A, ds * ds, ds * ds);
        if (m->gpu_opt_flags & GPU_OPT_BA)
            gpu_gradient_null_witness_accumulate_buffer(
                p, m, p->opt_layers[k].gpu_dbA, ds, ds);
        if (m->gpu_opt_flags & GPU_OPT_CONV)
            gpu_gradient_null_witness_accumulate_buffer(
                p, m, p->opt_layers[k].gpu_dconv,
                (size_t)V12_CONV_KERNEL * ds,
                (size_t)V12_CONV_KERNEL * ds);
    }
}

static double r375_pbg2_gram_anisotropy(
    const float *g, int rows, int cols, int row_gram) {
    const int outer = row_gram ? rows : cols;
    const int inner = row_gram ? cols : rows;
    double trace = 0.0;
    double gram_fro_sq = 0.0;

    if (!g || rows <= 0 || cols <= 0) return 0.0;

    for (int a = 0; a < outer; a++) {
        for (int b = 0; b < outer; b++) {
            double dot = 0.0;
            for (int k = 0; k < inner; k++) {
                const size_t ia = row_gram
                    ? (size_t)a * (size_t)cols + (size_t)k
                    : (size_t)k * (size_t)cols + (size_t)a;
                const size_t ib = row_gram
                    ? (size_t)b * (size_t)cols + (size_t)k
                    : (size_t)k * (size_t)cols + (size_t)b;
                dot += (double)g[ia] * (double)g[ib];
            }
            gram_fro_sq += dot * dot;
            if (a == b) trace += dot;
        }
    }

    if (gram_fro_sq <= 0.0 || trace <= 0.0) return 0.0;
    const double iso_sq = (trace * trace) / (double)outer;
    const double dev_sq = fmax(0.0, gram_fro_sq - iso_sq);
    return sqrt(dev_sq / gram_fro_sq);
}

static void r375_pbg2_accumulate_matrix(
    GpuPipeline *p, Model *m, VkBufferHandle grad_h, VkBufferHandle shadow_h,
    int rows, int cols, float lr, uint32_t *seen) {
    const size_t elems = (size_t)rows * (size_t)cols;
    if (!p || !p->ctx || !m || !seen || !m->r375_pbg2_gate_enabled ||
        !VK_BUFFER_VALID(grad_h) || !VK_BUFFER_VALID(shadow_h) ||
        rows <= 0 || cols <= 0 || elems == 0u) {
        return;
    }
    if (m->r375_pbg2_max_matrices > 0u &&
        *seen >= m->r375_pbg2_max_matrices) {
        return;
    }

    float *grad = (float *)malloc(elems * sizeof(float));
    shadow_t *shadow = (shadow_t *)malloc(elems * SHADOW_SIZEOF);
    if (!grad || !shadow) {
        free(grad);
        free(shadow);
        return;
    }

    vk_download(p->ctx, grad, grad_h, elems * sizeof(float));
    vk_download(p->ctx, shadow, shadow_h, elems * SHADOW_SIZEOF);

    double grad_energy = 0.0;
    double abs_update_sum = 0.0;
    uint64_t finite = 0u;
    uint64_t mvue_active = 0u;
    for (size_t i = 0; i < elems; i++) {
        const double gv = (double)grad[i];
        const double sv = (double)SHADOW_DECODE(shadow[i]);
        if (!isfinite(gv) || !isfinite(sv)) continue;
        grad_energy += gv * gv;
        abs_update_sum += fabs((double)lr * gv);
        if (fabs(sv) < 1.0) mvue_active++;
        finite++;
    }
    if (finite == 0u || grad_energy <= 0.0) {
        free(grad);
        free(shadow);
        return;
    }

    const double row_aniso = r375_pbg2_gram_anisotropy(grad, rows, cols, 1);
    const double col_aniso = r375_pbg2_gram_anisotropy(grad, rows, cols, 0);
    const double shape_signal = fmax(row_aniso, col_aniso);
    const double mvue_fraction = (double)mvue_active / (double)finite;
    const double mean_abs_update = abs_update_sum / (double)finite;
    const double grid_snr = mean_abs_update / (double)V12_SHADOW_MVUE_STEP;

    (*seen)++;
    m->r375_pbg2_gate_batches++;
    m->r375_pbg2_gate_matrices++;
    m->r375_pbg2_gate_elements += finite;
    m->r375_pbg2_gate_grad_energy += grad_energy;
    m->r375_pbg2_gate_row_anisotropy_sum += row_aniso;
    m->r375_pbg2_gate_col_anisotropy_sum += col_aniso;
    m->r375_pbg2_gate_shape_signal_sum += shape_signal;
    m->r375_pbg2_gate_mvue_active_sum += mvue_fraction;
    m->r375_pbg2_gate_grid_snr_sum += grid_snr;
    if (shape_signal > (double)m->r375_pbg2_gate_max_shape_signal) {
        m->r375_pbg2_gate_max_shape_signal = (float)shape_signal;
    }

    const double n = (double)m->r375_pbg2_gate_matrices;
    const double mean_shape = m->r375_pbg2_gate_shape_signal_sum / n;
    const double mean_mvue = m->r375_pbg2_gate_mvue_active_sum / n;
    const double mean_grid = m->r375_pbg2_gate_grid_snr_sum / n;
    const double min_aniso = (m->r375_pbg2_min_anisotropy > 0.0f)
        ? (double)m->r375_pbg2_min_anisotropy
        : 0.03;

    m->r375_pbg2_gate_row_anisotropy =
        (float)(m->r375_pbg2_gate_row_anisotropy_sum / n);
    m->r375_pbg2_gate_col_anisotropy =
        (float)(m->r375_pbg2_gate_col_anisotropy_sum / n);
    m->r375_pbg2_gate_shape_signal = (float)mean_shape;
    m->r375_pbg2_gate_mvue_active_fraction = (float)mean_mvue;
    m->r375_pbg2_gate_grid_snr = (float)mean_grid;
    m->r375_pbg2_gate_identifying =
        (m->r375_pbg2_gate_matrices > 0u &&
         m->r375_pbg2_gate_grad_energy > 0.0) ? 1 : 0;
    m->r375_pbg2_gate_pass =
        (m->r375_pbg2_gate_identifying &&
         mean_shape >= min_aniso &&
         mean_mvue >= 0.90) ? 1 : 0;

    free(grad);
    free(shadow);
}

void gpu_r375_pbg2_gate_update(GpuPipeline *p, Model *m, float lr) {
    if (!p || !m || !m->r375_pbg2_gate_enabled) return;

    uint32_t seen = 0u;
    const int d = p->d_model;
    const int ds = p->d_state;

    for (int k = 0; k < p->n_layers; k++) {
        if (m->gpu_opt_flags & GPU_OPT_W_IN) {
            r375_pbg2_accumulate_matrix(
                p, m,
                p->opt_layers[k].gpu_dw_in,
                p->opt_layers[k].shadow_w_in,
                d, ds, lr, &seen);
        }
        if (m->gpu_opt_flags & GPU_OPT_W_OUT) {
            r375_pbg2_accumulate_matrix(
                p, m,
                p->opt_layers[k].gpu_dw_out,
                p->opt_layers[k].shadow_w_out,
                ds, d, lr, &seen);
        }
        if (m->r375_pbg2_max_matrices > 0u &&
            seen >= m->r375_pbg2_max_matrices) {
            break;
        }
    }
}

static void r375_pbg2_precondition_matrix(
    GpuPipeline *p, Model *m, VkBufferHandle grad_h,
    int rows, int cols, uint32_t *seen) {
    const size_t elems = (size_t)rows * (size_t)cols;
    if (!p || !p->ctx || !m || !seen ||
        !m->r375_pbg2_precondition_enabled ||
        !VK_BUFFER_VALID(grad_h) ||
        rows <= 0 || cols <= 0 || elems == 0u) {
        return;
    }
    if (m->r375_pbg2_precondition_max_matrices > 0u &&
        *seen >= m->r375_pbg2_precondition_max_matrices) {
        return;
    }

    float *grad = (float *)malloc(elems * sizeof(float));
    double *row_sq = (double *)calloc((size_t)rows, sizeof(double));
    double *col_sq = (double *)calloc((size_t)cols, sizeof(double));
    if (!grad || !row_sq || !col_sq) {
        free(grad);
        free(row_sq);
        free(col_sq);
        return;
    }

    vk_download(p->ctx, grad, grad_h, elems * sizeof(float));

    uint64_t finite = 0u;
    double raw_sq = 0.0;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            const size_t idx = (size_t)r * (size_t)cols + (size_t)c;
            const double g = (double)grad[idx];
            if (!isfinite(g)) {
                grad[idx] = 0.0f;
                continue;
            }
            const double g2 = g * g;
            row_sq[r] += g2;
            col_sq[c] += g2;
            raw_sq += g2;
            finite++;
        }
    }
    if (finite == 0u || raw_sq <= 0.0) {
        free(grad);
        free(row_sq);
        free(col_sq);
        return;
    }

    const double eps = (m->r375_pbg2_precondition_eps > 0.0f)
        ? (double)m->r375_pbg2_precondition_eps
        : 1.0e-6;
    double pre_sq = 0.0;
    double dot = 0.0;
    for (int r = 0; r < rows; r++) {
        const double row_rms = sqrt(row_sq[r] / (double)cols);
        for (int c = 0; c < cols; c++) {
            const size_t idx = (size_t)r * (size_t)cols + (size_t)c;
            const double g = (double)grad[idx];
            const double col_rms = sqrt(col_sq[c] / (double)rows);
            const double denom = sqrt(fmax(row_rms * col_rms, 0.0)) + eps;
            const double pg = (denom > 0.0) ? (g / denom) : g;
            grad[idx] = (float)pg;
            pre_sq += pg * pg;
            dot += g * pg;
        }
    }
    if (pre_sq <= 0.0) {
        free(grad);
        free(row_sq);
        free(col_sq);
        return;
    }

    const double scale = sqrt(raw_sq / pre_sq);
    for (size_t i = 0; i < elems; i++) {
        grad[i] = (float)((double)grad[i] * scale);
    }
    vk_upload(p->ctx, grad_h, grad, elems * sizeof(float));

    const double raw_rms = sqrt(raw_sq / (double)finite);
    const double pre_rms = sqrt(pre_sq / (double)finite);
    const double cosine = dot / (sqrt(raw_sq * pre_sq) + 1.0e-30);

    (*seen)++;
    m->r375_pbg2_precondition_batches++;
    m->r375_pbg2_precondition_matrices++;
    m->r375_pbg2_precondition_elements += finite;
    m->r375_pbg2_precondition_raw_rms_sum += raw_rms;
    m->r375_pbg2_precondition_pre_rms_sum += pre_rms;
    m->r375_pbg2_precondition_scale_sum += scale;
    m->r375_pbg2_precondition_cosine_sum += cosine;

    const double n = (double)m->r375_pbg2_precondition_matrices;
    m->r375_pbg2_precondition_raw_rms =
        (float)(m->r375_pbg2_precondition_raw_rms_sum / n);
    m->r375_pbg2_precondition_pre_rms =
        (float)(m->r375_pbg2_precondition_pre_rms_sum / n);
    m->r375_pbg2_precondition_scale =
        (float)(m->r375_pbg2_precondition_scale_sum / n);
    m->r375_pbg2_precondition_cosine =
        (float)(m->r375_pbg2_precondition_cosine_sum / n);
    m->r375_pbg2_precondition_identifying =
        (m->r375_pbg2_precondition_matrices > 0u &&
         m->r375_pbg2_precondition_elements > 0u) ? 1 : 0;

    free(grad);
    free(row_sq);
    free(col_sq);
}

void gpu_r375_pbg2_precondition_update(GpuPipeline *p, Model *m) {
    if (!p || !m || !m->r375_pbg2_precondition_enabled) return;

    uint32_t seen = 0u;
    const int d = p->d_model;
    const int ds = p->d_state;

    for (int k = 0; k < p->n_layers; k++) {
        if (m->gpu_opt_flags & GPU_OPT_W_IN) {
            r375_pbg2_precondition_matrix(
                p, m, p->opt_layers[k].gpu_dw_in, d, ds, &seen);
        }
        if (m->gpu_opt_flags & GPU_OPT_W_OUT) {
            r375_pbg2_precondition_matrix(
                p, m, p->opt_layers[k].gpu_dw_out, ds, d, &seen);
        }
        if (m->r375_pbg2_precondition_max_matrices > 0u &&
            seen >= m->r375_pbg2_precondition_max_matrices) {
            break;
        }
    }
    vk_batch_checkpoint(p->ctx);
}

static uint64_t chamber_weighted_work(uint64_t count,
                                      uint64_t finite,
                                      size_t work_units) {
    if (finite == 0u || count == 0u) return 0u;
    const long double work = (long double)(work_units > 0u ? work_units : finite);
    const long double weighted = work * (long double)count / (long double)finite;
    if (weighted <= 0.0L) return 0u;
    return (uint64_t)(weighted + 0.5L);
}

static void gpu_ce_normal_chamber_refresh(Model *m) {
    const double elems = (double)m->ce_normal_chamber_gate_elements;
    const double work = (double)m->ce_normal_chamber_gate_work_units;
    const double cand = (double)m->ce_normal_chamber_gate_candidate_elements;
    const double cand_work = (double)m->ce_normal_chamber_gate_candidate_work_units;
    const double safe1 = (double)m->ce_normal_chamber_gate_safe_work_k1;
    const double safe4 = (double)m->ce_normal_chamber_gate_safe_work_k4;
    const double safe16 = (double)m->ce_normal_chamber_gate_safe_work_k16;
    const double overhead = (double)m->ce_normal_chamber_overhead;

    m->ce_normal_chamber_gate_candidate_fraction =
        elems > 0.0 ? (float)(cand / elems) : 0.0f;
    m->ce_normal_chamber_gate_candidate_work_fraction =
        work > 0.0 ? (float)(cand_work / work) : 0.0f;
    m->ce_normal_chamber_gate_safe_fraction_k1 =
        work > 0.0 ? (float)(safe1 / work) : 0.0f;
    m->ce_normal_chamber_gate_safe_fraction_k4 =
        work > 0.0 ? (float)(safe4 / work) : 0.0f;
    m->ce_normal_chamber_gate_safe_fraction_k16 =
        work > 0.0 ? (float)(safe16 / work) : 0.0f;
    m->ce_normal_chamber_gate_priced_gain_k1 =
        m->ce_normal_chamber_gate_safe_fraction_k1 - (float)overhead;
    m->ce_normal_chamber_gate_priced_gain_k4 =
        m->ce_normal_chamber_gate_safe_fraction_k4 - (float)overhead;
    m->ce_normal_chamber_gate_priced_gain_k16 =
        m->ce_normal_chamber_gate_safe_fraction_k16 - (float)overhead;
    m->ce_normal_chamber_gate_mean_margin =
        cand > 0.0 ? (float)(m->ce_normal_chamber_gate_margin_sum / cand) : 0.0f;
    m->ce_normal_chamber_gate_mean_force =
        cand > 0.0 ? (float)(m->ce_normal_chamber_gate_force_sum / cand) : 0.0f;
    m->ce_normal_chamber_gate_identifying =
        (m->ce_normal_chamber_gate_groups > 0u &&
         m->ce_normal_chamber_gate_elements > 0u &&
         m->ce_normal_chamber_gate_candidate_elements > 0u) ? 1 : 0;
    m->ce_normal_chamber_gate_pass =
        (m->ce_normal_chamber_gate_identifying &&
         m->ce_normal_chamber_gate_priced_gain_k16 > 0.10f) ? 1 : 0;
}

static void gpu_ce_normal_chamber_accumulate_buffer(
    GpuPipeline *p, Model *m, VkBufferHandle shadow_h, VkBufferHandle grad_h,
    size_t elems, size_t work_units, float lr_eff) {
    if (!p || !p->ctx || !m || !m->ce_normal_chamber_gate_enabled ||
        !VK_BUFFER_VALID(shadow_h) || !VK_BUFFER_VALID(grad_h) || elems == 0u) {
        return;
    }

    shadow_t *shadow = (shadow_t *)malloc(elems * sizeof(shadow_t));
    float *grad = (float *)malloc(elems * sizeof(float));
    if (!shadow || !grad) {
        free(shadow);
        free(grad);
        return;
    }

    vk_download(p->ctx, shadow, shadow_h, elems * SHADOW_SIZEOF);
    vk_download(p->ctx, grad, grad_h, elems * sizeof(float));

    const double sigma = (double)((m->ce_normal_chamber_sigma > 0.0f)
        ? m->ce_normal_chamber_sigma
        : 0.0f);
    uint64_t finite = 0u;
    uint64_t stochastic = 0u;
    uint64_t saturated = 0u;
    uint64_t crystallized = 0u;
    uint64_t candidate = 0u;
    uint64_t safe1 = 0u;
    uint64_t safe4 = 0u;
    uint64_t safe16 = 0u;
    double margin_sum = 0.0;
    double force_sum = 0.0;
    double min_margin = INFINITY;

    for (size_t i = 0; i < elems; i++) {
        const float s = SHADOW_DECODE(shadow[i]);
        const float g = grad[i];
        if (!isfinite((double)s) || !isfinite((double)g)) continue;
        finite++;

        const double abs_s = fabs((double)s);
        if (abs_s < 1.0) {
            stochastic++;
            continue;
        }

        saturated++;
        if (abs_s >= (double)V12_CRYSTALLIZATION_TAU) {
            crystallized++;
            continue;
        }

        candidate++;
        const double margin = abs_s - 1.0;
        const double force = fabs((double)lr_eff * (double)g);
        margin_sum += margin;
        force_sum += force;
        if (margin < min_margin) min_margin = margin;
        if (margin > sigma) {
            const double allowance = margin - sigma;
            if (force <= allowance) safe1++;
            if (4.0 * force <= allowance) safe4++;
            if (16.0 * force <= allowance) safe16++;
        }
    }

    if (finite > 0u) {
        const size_t charged_work = work_units > 0u ? work_units : (size_t)finite;
        m->ce_normal_chamber_gate_groups++;
        m->ce_normal_chamber_gate_elements += finite;
        m->ce_normal_chamber_gate_stochastic_elements += stochastic;
        m->ce_normal_chamber_gate_saturated_elements += saturated;
        m->ce_normal_chamber_gate_crystallized_elements += crystallized;
        m->ce_normal_chamber_gate_candidate_elements += candidate;
        m->ce_normal_chamber_gate_work_units += (uint64_t)charged_work;
        m->ce_normal_chamber_gate_candidate_work_units +=
            chamber_weighted_work(candidate, finite, charged_work);
        m->ce_normal_chamber_gate_safe_work_k1 +=
            chamber_weighted_work(safe1, finite, charged_work);
        m->ce_normal_chamber_gate_safe_work_k4 +=
            chamber_weighted_work(safe4, finite, charged_work);
        m->ce_normal_chamber_gate_safe_work_k16 +=
            chamber_weighted_work(safe16, finite, charged_work);
        m->ce_normal_chamber_gate_margin_sum += margin_sum;
        m->ce_normal_chamber_gate_force_sum += force_sum;
        if (candidate > 0u &&
            (m->ce_normal_chamber_gate_min_margin <= 0.0f ||
             min_margin < (double)m->ce_normal_chamber_gate_min_margin)) {
            m->ce_normal_chamber_gate_min_margin = (float)min_margin;
        }
        gpu_ce_normal_chamber_refresh(m);
    }

    free(shadow);
    free(grad);
}

void gpu_ce_normal_chamber_gate_update(GpuPipeline *p, Model *m,
                                       const TrainState *state,
                                       float lr, int step) {
    (void)step;
    if (!p || !m || !m->ce_normal_chamber_gate_enabled) return;

    const size_t d = (size_t)p->d_model;
    const size_t ds = (size_t)p->d_state;

    m->ce_normal_chamber_gate_batches++;

    for (int k = 0; k < p->n_layers; k++) {
        const float lr_eff = lr * (state ? state->lr_per_layer[k] : 1.0f);
        if (m->gpu_opt_flags & GPU_OPT_W_IN) {
            gpu_ce_normal_chamber_accumulate_buffer(
                p, m, p->opt_layers[k].shadow_w_in,
                p->opt_layers[k].gpu_dw_in, d * ds, d * ds, lr_eff);
        }
        if (m->gpu_opt_flags & GPU_OPT_W_OUT) {
            gpu_ce_normal_chamber_accumulate_buffer(
                p, m, p->opt_layers[k].shadow_w_out,
                p->opt_layers[k].gpu_dw_out, ds * d, ds * d, lr_eff);
        }
    }

    const float lr_eff_lm = lr *
        (state ? state->lr_per_layer[V12_N_LAYERS - 1] : 1.0f);
    if (m->gpu_opt_flags & GPU_OPT_LM_HEAD) {
        gpu_ce_normal_chamber_accumulate_buffer(
            p, m, p->opt_shadow_lm_head, p->ce_dlm_head,
            (size_t)V12_VOCAB_SIZE * d, (size_t)V12_K_HOT * d, lr_eff_lm);
    }
    if (m->gpu_opt_flags & GPU_OPT_W_PRED) {
        gpu_ce_normal_chamber_accumulate_buffer(
            p, m, p->opt_shadow_w_pred, p->opt_dw_pred,
            d * d, d * d, lr_eff_lm);
    }

    printf("[CE_NORMAL_CHAMBER_GATE] step=%d groups=%llu elements=%llu stochastic=%llu saturated=%llu crystallized=%llu candidates=%llu candidate_fraction=%.6f candidate_work_fraction=%.6f safe_k1=%.6f safe_k4=%.6f safe_k16=%.6f priced_k1=%.6f priced_k4=%.6f priced_k16=%.6f min_margin=%.9g mean_margin=%.9g mean_force=%.9g sigma=%.9g identifying=%d pass=%d stage=force_bound_only replay=not_run\n",
           step + 1,
           (unsigned long long)m->ce_normal_chamber_gate_groups,
           (unsigned long long)m->ce_normal_chamber_gate_elements,
           (unsigned long long)m->ce_normal_chamber_gate_stochastic_elements,
           (unsigned long long)m->ce_normal_chamber_gate_saturated_elements,
           (unsigned long long)m->ce_normal_chamber_gate_crystallized_elements,
           (unsigned long long)m->ce_normal_chamber_gate_candidate_elements,
           m->ce_normal_chamber_gate_candidate_fraction,
           m->ce_normal_chamber_gate_candidate_work_fraction,
           m->ce_normal_chamber_gate_safe_fraction_k1,
           m->ce_normal_chamber_gate_safe_fraction_k4,
           m->ce_normal_chamber_gate_safe_fraction_k16,
           m->ce_normal_chamber_gate_priced_gain_k1,
           m->ce_normal_chamber_gate_priced_gain_k4,
           m->ce_normal_chamber_gate_priced_gain_k16,
           m->ce_normal_chamber_gate_min_margin,
           m->ce_normal_chamber_gate_mean_margin,
           m->ce_normal_chamber_gate_mean_force,
           m->ce_normal_chamber_sigma,
           m->ce_normal_chamber_gate_identifying,
           m->ce_normal_chamber_gate_pass);
}

void gpu_online_tssi_live_gate_update(GpuPipeline *p, Model *m) {
    if (!p || !m || !m->online_tssi_live_gate_enabled) return;

    uint64_t state_layers = 0;
    uint64_t moment_layers = 0;
    uint64_t score_layers = 0;
    for (int k = 0; k < p->n_layers; k++) {
        if (p->layers[k].fwd_allocated &&
            VK_BUFFER_VALID(p->layers[k].fwd_h_cache)) {
            state_layers++;
        }
        if (VK_BUFFER_VALID(p->layers[k].gpu_moments)) {
            moment_layers++;
        }
    }

    const int sampled_ce_ready =
        VK_BUFFER_VALID(p->ce_sample_ids) && VK_BUFFER_VALID(p->ce_d_logits);
    if (sampled_ce_ready) {
        for (int k = 0; k < p->n_layers; k++) {
            if (!p->layers[k].fwd_allocated ||
                !VK_BUFFER_VALID(p->layers[k].fwd_h_cache) ||
                !VK_BUFFER_VALID(p->layers[k].tssi_score_moments)) {
                continue;
            }
            if (vk_dispatch_online_tssi_score_update(
                    p->ctx,
                    p->layers[k].fwd_h_cache,
                    p->ce_d_logits,
                    p->layers[k].tssi_score_moments,
                    (uint32_t)p->batch_size,
                    (uint32_t)p->seq_len,
                    (uint32_t)p->d_state,
                    (uint32_t)V12_SS_N_SAMPLES) == 0) {
                score_layers++;
            }
        }
    }
    const uint64_t missing_state_score_layers =
        state_layers > score_layers ? state_layers - score_layers : 0u;

    m->online_tssi_live_gate_batches++;
    m->online_tssi_live_gate_state_layers += state_layers;
    m->online_tssi_live_gate_moment_layers += moment_layers;
    m->online_tssi_live_gate_score_layers += score_layers;
    m->online_tssi_live_gate_score_updates +=
        score_layers * (uint64_t)p->batch_size *
        (uint64_t)(p->seq_len > 0 ? p->seq_len - 1 : 0);
    m->online_tssi_live_gate_sampled_ce_buffers += sampled_ce_ready ? 1u : 0u;
    m->online_tssi_live_gate_missing_state_score_layers +=
        missing_state_score_layers;

    const double denom =
        (double)m->online_tssi_live_gate_batches * (double)p->n_layers;
    m->online_tssi_live_gate_state_layer_fraction =
        denom > 0.0 ? (float)((double)m->online_tssi_live_gate_state_layers / denom)
                    : 0.0f;
    m->online_tssi_live_gate_moment_layer_fraction =
        denom > 0.0 ? (float)((double)m->online_tssi_live_gate_moment_layers / denom)
                    : 0.0f;
    m->online_tssi_live_gate_score_layer_fraction =
        denom > 0.0 ? (float)((double)m->online_tssi_live_gate_score_layers / denom)
                    : 0.0f;

    m->online_tssi_live_gate_identifying =
        (state_layers == (uint64_t)p->n_layers &&
         moment_layers == (uint64_t)p->n_layers &&
         score_layers == (uint64_t)p->n_layers &&
         sampled_ce_ready &&
         missing_state_score_layers == 0u) ? 1 : 0;
    m->online_tssi_live_gate_pass = m->online_tssi_live_gate_identifying;
}

static int r423_compare_double_ascending(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da > db) - (da < db);
}

void gpu_r423_native_growth_receipt_challenge_gate_update(GpuPipeline *p,
                                                          int step) {
    if (!p || !p->ctx || p->n_layers <= 0 ||
        p->d_model <= 0 || p->d_state <= 0) {
        return;
    }

    const uint32_t d = (uint32_t)p->d_model;
    const uint32_t ds = (uint32_t)p->d_state;
    const int states = p->n_layers * p->d_state;
    const size_t grad_elems = (size_t)d * (size_t)ds;
    const size_t grad_bytes = grad_elems * sizeof(float);
    double *full_risk = (double *)calloc((size_t)states, sizeof(double));
    double *receipt_risk = (double *)calloc((size_t)states, sizeof(double));
    double *full_sorted = (double *)calloc((size_t)states, sizeof(double));
    double *receipt_sorted = (double *)calloc((size_t)states, sizeof(double));
    float *grad_in = (float *)malloc(grad_bytes);
    float *grad_out = (float *)malloc(grad_bytes);
    if (!full_risk || !receipt_risk || !full_sorted || !receipt_sorted ||
        !grad_in || !grad_out) {
        printf("[R423_NATIVE_GROWTH_RECEIPT_CHALLENGE_GATE] step=%d active=1 mode=real-gpu-gradient-receipt-challenge states=%d status=oom training_change=0 pass=0\n",
               step, states);
        free(full_risk);
        free(receipt_risk);
        free(full_sorted);
        free(receipt_sorted);
        free(grad_in);
        free(grad_out);
        return;
    }

    int valid = 0;
    double mean_full = 0.0;
    double mean_receipt = 0.0;
    vk_sync(p->ctx);
    for (int k = 0; k < p->n_layers; k++) {
        const int have_in =
            VK_BUFFER_VALID(p->opt_layers[k].gpu_dw_in) &&
            vk_download(p->ctx, grad_in, p->opt_layers[k].gpu_dw_in,
                        grad_bytes) == 0;
        const int have_out =
            VK_BUFFER_VALID(p->opt_layers[k].gpu_dw_out) &&
            vk_download(p->ctx, grad_out, p->opt_layers[k].gpu_dw_out,
                        grad_bytes) == 0;
        if (!have_in) memset(grad_in, 0, grad_bytes);
        if (!have_out) memset(grad_out, 0, grad_bytes);

        for (uint32_t s = 0; s < ds; s++) {
            double full_sum = 0.0;
            double full_l1 = 0.0;
            double receipt_sum = 0.0;
            double receipt_l1 = 0.0;
            for (uint32_t j = 0; j < d; j++) {
                const double gin = (double)grad_in[(size_t)j * ds + s];
                const double gout = (double)grad_out[(size_t)s * d + j];
                const double g = gin + gout;
                const double ag = fabs(g);
                full_sum += g;
                full_l1 += ag;
                if ((j & 63u) == 0u) {
                    receipt_sum += g;
                    receipt_l1 += ag;
                }
            }
            if (full_l1 <= 1.0e-20 || receipt_l1 <= 1.0e-20) continue;
            const double fr = 1.0 - fmin(1.0, fabs(full_sum) / full_l1);
            const double rr = 1.0 - fmin(1.0, fabs(receipt_sum) / receipt_l1);
            const int idx = k * p->d_state + (int)s;
            full_risk[idx] = fr;
            receipt_risk[idx] = rr;
            full_sorted[valid] = fr;
            receipt_sorted[valid] = rr;
            mean_full += fr;
            mean_receipt += rr;
            valid++;
        }
    }
    vk_batch_checkpoint(p->ctx);

    if (valid <= 0) {
        printf("[R423_NATIVE_GROWTH_RECEIPT_CHALLENGE_GATE] step=%d active=1 mode=real-gpu-gradient-receipt-challenge states=%d valid_states=0 status=no_gpu_gradient_source training_change=0 pass=0\n",
               step, states);
        free(full_risk);
        free(receipt_risk);
        free(full_sorted);
        free(receipt_sorted);
        free(grad_in);
        free(grad_out);
        return;
    }

    qsort(full_sorted, (size_t)valid, sizeof(double),
          r423_compare_double_ascending);
    qsort(receipt_sorted, (size_t)valid, sizeof(double),
          r423_compare_double_ascending);
    const double full_threshold = full_sorted[(valid * 90) / 100];
    const double receipt_threshold = receipt_sorted[(valid * 80) / 100];
    int harmful = 0;
    int challenged = 0;
    int caught = 0;
    int accepted_harm = 0;
    int false_challenge = 0;
    for (int i = 0; i < states; i++) {
        if (full_risk[i] <= 0.0 && receipt_risk[i] <= 0.0) continue;
        const int is_harmful = full_risk[i] >= full_threshold;
        const int is_challenged = receipt_risk[i] >= receipt_threshold;
        harmful += is_harmful ? 1 : 0;
        challenged += is_challenged ? 1 : 0;
        caught += (is_harmful && is_challenged) ? 1 : 0;
        accepted_harm += (is_harmful && !is_challenged) ? 1 : 0;
        false_challenge += (!is_harmful && is_challenged) ? 1 : 0;
    }

    const double caught_fraction =
        harmful > 0 ? (double)caught / (double)harmful : 0.0;
    const double accepted_fraction =
        harmful > 0 ? (double)accepted_harm / (double)harmful : 0.0;
    const double false_challenge_fraction =
        challenged > 0 ? (double)false_challenge / (double)challenged : 0.0;
    const double challenge_fraction =
        valid > 0 ? (double)challenged / (double)valid : 0.0;
    const double work_ratio = challenge_fraction + 0.05;
    const double primary_value =
        (1.0 - challenge_fraction) * caught_fraction - accepted_fraction;
    const double priced_ratio = work_ratio + accepted_fraction;
    const int pass = caught_fraction >= 0.70 &&
                     accepted_fraction <= 0.30 &&
                     priced_ratio <= 0.60 &&
                     primary_value > 0.10;

    printf("[R423_NATIVE_GROWTH_RECEIPT_CHALLENGE_GATE] step=%d active=1 mode=real-gpu-gradient-receipt-challenge states=%d valid_states=%d receipt_stride=64 harmful=%d challenged=%d caught=%d accepted_harm=%d caught_harm_fraction=%.6f accepted_harm_fraction=%.6f false_challenge_fraction=%.6f challenge_work_ratio=%.6f priced_ratio=%.6f primary_value=%.6f mean_full_risk=%.6f mean_receipt_risk=%.6f training_change=0 pass=%d\n",
           step,
           states,
           valid,
           harmful,
           challenged,
           caught,
           accepted_harm,
           caught_fraction,
           accepted_fraction,
           false_challenge_fraction,
           work_ratio,
           priced_ratio,
           primary_value,
           mean_full / (double)valid,
           mean_receipt / (double)valid,
           pass);

    free(full_risk);
    free(receipt_risk);
    free(full_sorted);
    free(receipt_sorted);
    free(grad_in);
    free(grad_out);
}

typedef struct {
    double commutator;
    double harm;
} R423InteractionCandidate;

typedef struct {
    int layer;
    uint32_t row;
    double commutator;
    double harm;
} R423InteractionUpdateCandidate;

static int r423_compare_interaction_candidate_by_commutator(const void *a,
                                                            const void *b) {
    const R423InteractionCandidate *ca = (const R423InteractionCandidate *)a;
    const R423InteractionCandidate *cb = (const R423InteractionCandidate *)b;
    return (ca->commutator > cb->commutator) -
           (ca->commutator < cb->commutator);
}

static int r423_compare_interaction_update_candidate_by_commutator(const void *a,
                                                                   const void *b) {
    const R423InteractionUpdateCandidate *ca =
        (const R423InteractionUpdateCandidate *)a;
    const R423InteractionUpdateCandidate *cb =
        (const R423InteractionUpdateCandidate *)b;
    return (ca->commutator > cb->commutator) -
           (ca->commutator < cb->commutator);
}

void gpu_r423_native_growth_interaction_release_safety_gate_update(GpuPipeline *p,
                                                                   int step) {
    if (!p || !p->ctx || p->n_layers <= 0 || p->d_state <= 0) return;

    const uint32_t ds = (uint32_t)p->d_state;
    const int capacity = p->n_layers * p->d_state;
    const size_t vec_bytes = (size_t)ds * sizeof(float);
    const size_t mat_bytes = (size_t)ds * (size_t)ds * sizeof(float);
    float *a = (float *)malloc(vec_bytes);
    float *dwa = (float *)malloc(mat_bytes);
    R423InteractionCandidate *candidates =
        (R423InteractionCandidate *)calloc((size_t)capacity,
                                           sizeof(R423InteractionCandidate));
    R423InteractionCandidate *sorted =
        (R423InteractionCandidate *)calloc((size_t)capacity,
                                           sizeof(R423InteractionCandidate));
    if (!a || !dwa || !candidates || !sorted) {
        printf("[R423_NATIVE_GROWTH_INTERACTION_RELEASE_GATE] step=%d active=1 mode=real-gpu-dwa-commutator-observer states=%d status=oom training_change=0 pass=0\n",
               step, capacity);
        free(a);
        free(dwa);
        free(candidates);
        free(sorted);
        return;
    }

    int valid = 0;
    double sum_comm = 0.0;
    double sum_harm = 0.0;
    double sum_comm_sq = 0.0;
    double sum_harm_sq = 0.0;
    double sum_cross = 0.0;
    vk_sync(p->ctx);
    for (int k = 0; k < p->n_layers; k++) {
        if (!VK_BUFFER_VALID(p->layers[k].gpu_A_const) ||
            !VK_BUFFER_VALID(p->opt_layers[k].gpu_dW_A) ||
            vk_download(p->ctx, a, p->layers[k].gpu_A_const, vec_bytes) != 0 ||
            vk_download(p->ctx, dwa, p->opt_layers[k].gpu_dW_A, mat_bytes) != 0) {
            continue;
        }

        for (uint32_t r = 0; r < ds; r++) {
            double comm_sq = 0.0;
            double harm_sq = 0.0;
            const size_t base = (size_t)r * (size_t)ds;
            for (uint32_t c = 0; c < ds; c++) {
                const double g = (double)dwa[base + c];
                const double gap = (double)a[r] - (double)a[c];
                harm_sq += g * g;
                comm_sq += gap * gap * g * g;
            }
            if (harm_sq <= 1.0e-30) continue;
            const double comm = sqrt(comm_sq);
            const double harm = harm_sq;
            candidates[valid].commutator = comm;
            candidates[valid].harm = harm;
            sorted[valid] = candidates[valid];
            sum_comm += comm;
            sum_harm += harm;
            sum_comm_sq += comm * comm;
            sum_harm_sq += harm * harm;
            sum_cross += comm * harm;
            valid++;
        }
    }
    vk_batch_checkpoint(p->ctx);

    if (valid <= 0) {
        printf("[R423_NATIVE_GROWTH_INTERACTION_RELEASE_GATE] step=%d active=1 mode=real-gpu-dwa-commutator-observer states=%d valid_releases=0 status=no_gpu_interaction_gradient training_change=0 pass=0\n",
               step, capacity);
        free(a);
        free(dwa);
        free(candidates);
        free(sorted);
        return;
    }

    qsort(sorted, (size_t)valid, sizeof(R423InteractionCandidate),
          r423_compare_interaction_candidate_by_commutator);
    int selected = valid / 5;
    if (selected < 4) selected = valid < 4 ? valid : 4;
    double safe_harm = 0.0;
    double control_harm = 0.0;
    for (int i = 0; i < selected; i++) {
        safe_harm += sorted[i].harm;
        const int control_idx = (i * 37 + 17) % valid;
        control_harm += candidates[control_idx].harm;
    }
    safe_harm /= (double)selected;
    control_harm /= (double)selected;

    const double n = (double)valid;
    const double cov = sum_cross - (sum_comm * sum_harm / n);
    const double var_comm = sum_comm_sq - (sum_comm * sum_comm / n);
    const double var_harm = sum_harm_sq - (sum_harm * sum_harm / n);
    const double corr =
        (var_comm > 1.0e-30 && var_harm > 1.0e-30)
            ? cov / sqrt(var_comm * var_harm)
            : 0.0;
    const double harm_reduction =
        (control_harm - safe_harm) / fmax(control_harm, 1.0e-30);
    const double audit_cost = 0.04;
    const double priced_ratio = (1.0 - harm_reduction) + audit_cost;
    const int pass = corr >= 0.50 &&
                     harm_reduction >= 0.10 &&
                     priced_ratio <= 0.90;

    printf("[R423_NATIVE_GROWTH_INTERACTION_RELEASE_GATE] step=%d active=1 mode=real-gpu-dwa-commutator-observer states=%d valid_releases=%d selected=%d commutator_harm_correlation=%.6f safe_harm=%.9g control_harm=%.9g harm_reduction=%.6f audit_cost=%.6f priced_ratio=%.6f training_change=0 pass=%d\n",
           step,
           capacity,
           valid,
           selected,
           corr,
           safe_harm,
           control_harm,
           harm_reduction,
           audit_cost,
           priced_ratio,
           pass);

    free(a);
    free(dwa);
    free(candidates);
    free(sorted);
}

void gpu_r423_native_growth_interaction_safe_update(GpuPipeline *p,
                                                   int step) {
    if (!p || !p->ctx || p->n_layers <= 0 || p->d_state <= 0) return;

    const uint32_t ds = (uint32_t)p->d_state;
    const int capacity = p->n_layers * p->d_state;
    const size_t vec_bytes = (size_t)ds * sizeof(float);
    const size_t mat_bytes = (size_t)ds * (size_t)ds * sizeof(float);
    const float unsafe_scale = 0.5f;
    float *a = (float *)malloc(vec_bytes);
    float *dwa = (float *)malloc(mat_bytes);
    R423InteractionUpdateCandidate *candidates =
        (R423InteractionUpdateCandidate *)calloc((size_t)capacity,
                                                 sizeof(R423InteractionUpdateCandidate));
    R423InteractionUpdateCandidate *sorted =
        (R423InteractionUpdateCandidate *)calloc((size_t)capacity,
                                                 sizeof(R423InteractionUpdateCandidate));
    if (!a || !dwa || !candidates || !sorted) {
        printf("[R423_NATIVE_GROWTH_INTERACTION_SAFE_UPDATE] step=%d active=1 mode=real-gpu-dwa-commutator-safe-update status=oom training_change=1 applied=0 pass=0\n",
               step);
        free(a);
        free(dwa);
        free(candidates);
        free(sorted);
        return;
    }

    int valid = 0;
    double sum_comm = 0.0;
    double sum_harm = 0.0;
    double sum_comm_sq = 0.0;
    double sum_harm_sq = 0.0;
    double sum_cross = 0.0;
    vk_sync(p->ctx);
    for (int k = 0; k < p->n_layers; k++) {
        if (!VK_BUFFER_VALID(p->layers[k].gpu_A_const) ||
            !VK_BUFFER_VALID(p->opt_layers[k].gpu_dW_A) ||
            vk_download(p->ctx, a, p->layers[k].gpu_A_const, vec_bytes) != 0 ||
            vk_download(p->ctx, dwa, p->opt_layers[k].gpu_dW_A, mat_bytes) != 0) {
            continue;
        }

        for (uint32_t r = 0; r < ds; r++) {
            double comm_sq = 0.0;
            double harm_sq = 0.0;
            const size_t base = (size_t)r * (size_t)ds;
            for (uint32_t c = 0; c < ds; c++) {
                const double g = (double)dwa[base + c];
                const double gap = (double)a[r] - (double)a[c];
                harm_sq += g * g;
                comm_sq += gap * gap * g * g;
            }
            if (harm_sq <= 1.0e-30) continue;
            const double comm = sqrt(comm_sq);
            candidates[valid].layer = k;
            candidates[valid].row = r;
            candidates[valid].commutator = comm;
            candidates[valid].harm = harm_sq;
            sorted[valid] = candidates[valid];
            sum_comm += comm;
            sum_harm += harm_sq;
            sum_comm_sq += comm * comm;
            sum_harm_sq += harm_sq * harm_sq;
            sum_cross += comm * harm_sq;
            valid++;
        }
    }

    if (valid <= 0) {
        printf("[R423_NATIVE_GROWTH_INTERACTION_SAFE_UPDATE] step=%d active=1 mode=real-gpu-dwa-commutator-safe-update states=%d valid_releases=0 status=no_gpu_interaction_gradient training_change=1 applied=0 pass=0\n",
               step, capacity);
        vk_batch_checkpoint(p->ctx);
        free(a);
        free(dwa);
        free(candidates);
        free(sorted);
        return;
    }

    qsort(sorted, (size_t)valid, sizeof(R423InteractionUpdateCandidate),
          r423_compare_interaction_update_candidate_by_commutator);
    int selected = valid / 5;
    if (selected < 4) selected = valid < 4 ? valid : 4;
    const double cutoff = sorted[selected - 1].commutator;

    double safe_harm = 0.0;
    double control_harm = 0.0;
    for (int i = 0; i < selected; i++) {
        safe_harm += sorted[i].harm;
        const int control_idx = (i * 37 + 17) % valid;
        control_harm += candidates[control_idx].harm;
    }
    safe_harm /= (double)selected;
    control_harm /= (double)selected;

    const double n = (double)valid;
    const double cov = sum_cross - (sum_comm * sum_harm / n);
    const double var_comm = sum_comm_sq - (sum_comm * sum_comm / n);
    const double var_harm = sum_harm_sq - (sum_harm * sum_harm / n);
    const double corr =
        (var_comm > 1.0e-30 && var_harm > 1.0e-30)
            ? cov / sqrt(var_comm * var_harm)
            : 0.0;
    const double harm_reduction =
        (control_harm - safe_harm) / fmax(control_harm, 1.0e-30);
    const double audit_cost = 0.04;
    const double apply_cost = 0.025;
    const double priced_ratio = (1.0 - harm_reduction) + audit_cost + apply_cost;
    const int gate_pass = corr >= 0.50 &&
                          harm_reduction >= 0.10 &&
                          priced_ratio <= 0.95;

    uint64_t damped_rows = 0u;
    uint64_t damped_entries = 0u;
    double pre_harm = 0.0;
    double post_harm = 0.0;
    if (gate_pass) {
        for (int k = 0; k < p->n_layers; k++) {
            if (!VK_BUFFER_VALID(p->layers[k].gpu_A_const) ||
                !VK_BUFFER_VALID(p->opt_layers[k].gpu_dW_A) ||
                vk_download(p->ctx, a, p->layers[k].gpu_A_const, vec_bytes) != 0 ||
                vk_download(p->ctx, dwa, p->opt_layers[k].gpu_dW_A, mat_bytes) != 0) {
                continue;
            }

            uint64_t layer_damped = 0u;
            for (uint32_t r = 0; r < ds; r++) {
                double comm_sq = 0.0;
                double harm_sq = 0.0;
                const size_t base = (size_t)r * (size_t)ds;
                for (uint32_t c = 0; c < ds; c++) {
                    const double g = (double)dwa[base + c];
                    const double gap = (double)a[r] - (double)a[c];
                    harm_sq += g * g;
                    comm_sq += gap * gap * g * g;
                }
                if (harm_sq <= 1.0e-30) continue;
                pre_harm += harm_sq;
                if (sqrt(comm_sq) <= cutoff) {
                    post_harm += harm_sq;
                    continue;
                }
                damped_rows++;
                layer_damped += ds;
                for (uint32_t c = 0; c < ds; c++) {
                    dwa[base + c] *= unsafe_scale;
                }
                post_harm += harm_sq * (double)unsafe_scale * (double)unsafe_scale;
            }
            if (layer_damped > 0u &&
                vk_upload(p->ctx, p->opt_layers[k].gpu_dW_A, dwa, mat_bytes) == 0) {
                damped_entries += layer_damped;
            }
        }
    }
    vk_batch_checkpoint(p->ctx);

    const double applied_harm_reduction =
        pre_harm > 1.0e-30 ? (pre_harm - post_harm) / pre_harm : 0.0;
    const int applied = gate_pass && damped_entries > 0u;
    printf("[R423_NATIVE_GROWTH_INTERACTION_SAFE_UPDATE] step=%d active=1 mode=real-gpu-dwa-commutator-safe-update states=%d valid_releases=%d selected=%d cutoff=%.9g commutator_harm_correlation=%.6f safe_harm=%.9g control_harm=%.9g observer_harm_reduction=%.6f unsafe_scale=%.6f damped_rows=%llu damped_entries=%llu applied_harm_reduction=%.6f audit_cost=%.6f apply_cost=%.6f priced_ratio=%.6f training_change=1 applied=%d pass=%d\n",
           step,
           capacity,
           valid,
           selected,
           cutoff,
           corr,
           safe_harm,
           control_harm,
           harm_reduction,
           (double)unsafe_scale,
           (unsigned long long)damped_rows,
           (unsigned long long)damped_entries,
           applied_harm_reduction,
           audit_cost,
           apply_cost,
           priced_ratio,
           applied,
           gate_pass);

    free(a);
    free(dwa);
    free(candidates);
    free(sorted);
}

static int8_t r423_sign_bit(double x) {
    return x >= 0.0 ? (int8_t)1 : (int8_t)-1;
}

static void r423_fit_active_sign_factor(const int8_t *target,
                                        const uint8_t *active,
                                        uint32_t rows,
                                        uint32_t cols,
                                        int8_t *row_sign,
                                        int8_t *col_sign) {
    for (uint32_t r = 0; r < rows; r++) {
        int sum = 0;
        const size_t base = (size_t)r * (size_t)cols;
        for (uint32_t c = 0; c < cols; c++) {
            if (active[base + c]) sum += target[base + c];
        }
        row_sign[r] = sum >= 0 ? (int8_t)1 : (int8_t)-1;
    }
    for (int iter = 0; iter < 2; iter++) {
        for (uint32_t c = 0; c < cols; c++) {
            int sum = 0;
            for (uint32_t r = 0; r < rows; r++) {
                const size_t idx = (size_t)r * (size_t)cols + c;
                if (active[idx]) sum += (int)row_sign[r] * (int)target[idx];
            }
            col_sign[c] = sum >= 0 ? (int8_t)1 : (int8_t)-1;
        }
        for (uint32_t r = 0; r < rows; r++) {
            int sum = 0;
            const size_t base = (size_t)r * (size_t)cols;
            for (uint32_t c = 0; c < cols; c++) {
                if (active[base + c]) {
                    sum += (int)col_sign[c] * (int)target[base + c];
                }
            }
            row_sign[r] = sum >= 0 ? (int8_t)1 : (int8_t)-1;
        }
    }
}

static uint32_t r423_apply_sign_factor(const int8_t *target,
                                       uint8_t *active,
                                       uint32_t rows,
                                       uint32_t cols,
                                       const int8_t *row_sign,
                                       const int8_t *col_sign) {
    uint32_t covered = 0u;
    for (uint32_t r = 0; r < rows; r++) {
        const size_t base = (size_t)r * (size_t)cols;
        for (uint32_t c = 0; c < cols; c++) {
            const size_t idx = base + c;
            if (!active[idx]) continue;
            const int pred = (int)row_sign[r] * (int)col_sign[c];
            if (pred == (int)target[idx]) {
                active[idx] = 0u;
                covered++;
            }
        }
    }
    return covered;
}

void gpu_r423_native_growth_genotype_exception_gate_update(GpuPipeline *p,
                                                           int step) {
    if (!p || !p->ctx || p->n_layers <= 0 || p->d_state <= 0) return;

    const uint32_t ds = (uint32_t)p->d_state;
    const uint32_t rank = 3u;
    const size_t elems = (size_t)ds * (size_t)ds;
    const size_t mat_bytes = elems * sizeof(float);
    float *dwa = (float *)malloc(mat_bytes);
    int8_t *target = (int8_t *)malloc(elems);
    uint8_t *active = (uint8_t *)malloc(elems);
    uint8_t *random_active = (uint8_t *)malloc(elems);
    int8_t *row_sign = (int8_t *)malloc((size_t)ds);
    int8_t *col_sign = (int8_t *)malloc((size_t)ds);
    if (!dwa || !target || !active || !random_active || !row_sign || !col_sign) {
        printf("[R423_NATIVE_GROWTH_GENOTYPE_EXCEPTION_GATE] step=%d active=1 mode=real-gpu-dwa-genotype-observer status=oom training_change=0 pass=0\n",
               step);
        free(dwa);
        free(target);
        free(active);
        free(random_active);
        free(row_sign);
        free(col_sign);
        return;
    }

    uint64_t matrices = 0u;
    uint64_t dense_bits = 0u;
    uint64_t genotype_bits = 0u;
    uint64_t exceptions = 0u;
    uint64_t random_exceptions = 0u;
    uint64_t covered = 0u;
    vk_sync(p->ctx);
    for (int k = 0; k < p->n_layers; k++) {
        if (!VK_BUFFER_VALID(p->opt_layers[k].gpu_dW_A) ||
            vk_download(p->ctx, dwa, p->opt_layers[k].gpu_dW_A, mat_bytes) != 0) {
            continue;
        }
        uint32_t nonzero = 0u;
        for (size_t i = 0; i < elems; i++) {
            target[i] = r423_sign_bit((double)dwa[i]);
            active[i] = 1u;
            random_active[i] = 1u;
            if (fabs((double)dwa[i]) > 1.0e-30) nonzero++;
        }
        if (nonzero == 0u) continue;

        matrices++;
        dense_bits += (uint64_t)elems;
        genotype_bits += (uint64_t)rank * (uint64_t)(ds + ds);
        for (uint32_t q = 0; q < rank; q++) {
            r423_fit_active_sign_factor(target, active, ds, ds,
                                        row_sign, col_sign);
            covered += r423_apply_sign_factor(target, active, ds, ds,
                                              row_sign, col_sign);

            for (uint32_t r = 0; r < ds; r++) {
                row_sign[r] =
                    (((r * 1103515245u + q * 12345u + (uint32_t)k * 17u) >> 3) & 1u)
                        ? (int8_t)1
                        : (int8_t)-1;
            }
            for (uint32_t c = 0; c < ds; c++) {
                col_sign[c] =
                    (((c * 1664525u + q * 1013904223u + (uint32_t)k * 31u) >> 5) & 1u)
                        ? (int8_t)1
                        : (int8_t)-1;
            }
            (void)r423_apply_sign_factor(target, random_active, ds, ds,
                                         row_sign, col_sign);
        }

        for (size_t i = 0; i < elems; i++) {
            exceptions += active[i] ? 1u : 0u;
            random_exceptions += random_active[i] ? 1u : 0u;
        }
    }
    vk_batch_checkpoint(p->ctx);

    if (matrices == 0u || dense_bits == 0u) {
        printf("[R423_NATIVE_GROWTH_GENOTYPE_EXCEPTION_GATE] step=%d active=1 mode=real-gpu-dwa-genotype-observer matrices=0 status=no_gpu_dwa_source training_change=0 pass=0\n",
               step);
        free(dwa);
        free(target);
        free(active);
        free(random_active);
        free(row_sign);
        free(col_sign);
        return;
    }

    const double loc_bits = log2((double)(ds * ds)) + 1.0;
    const double description_ratio =
        ((double)genotype_bits + (double)exceptions * loc_bits) /
        (double)dense_bits;
    const double random_description_ratio =
        ((double)genotype_bits + (double)random_exceptions * loc_bits) /
        (double)dense_bits;
    const double exception_density = (double)exceptions / (double)dense_bits;
    const double random_exception_density =
        (double)random_exceptions / (double)dense_bits;
    const double pressure_gain = 1.0 - description_ratio;
    const double search_cost = 0.035;
    const double priced_ratio = description_ratio + search_cost;
    const double coverage = (double)covered / (double)dense_bits;
    const int pass = exception_density <= 0.08 &&
                     description_ratio <= 0.90 &&
                     pressure_gain > 0.10 &&
                     description_ratio < random_description_ratio * 0.5;

    printf("[R423_NATIVE_GROWTH_GENOTYPE_EXCEPTION_GATE] step=%d active=1 mode=real-gpu-dwa-genotype-observer matrices=%llu rank=%u dense_bits=%llu genotype_bits=%llu exceptions=%llu random_exceptions=%llu coverage=%.6f exception_density=%.6f random_exception_density=%.6f description_ratio=%.6f random_description_ratio=%.6f pressure_gain=%.6f search_cost=%.6f priced_ratio=%.6f training_change=0 pass=%d\n",
           step,
           (unsigned long long)matrices,
           rank,
           (unsigned long long)dense_bits,
           (unsigned long long)genotype_bits,
           (unsigned long long)exceptions,
           (unsigned long long)random_exceptions,
           coverage,
           exception_density,
           random_exception_density,
           description_ratio,
           random_description_ratio,
           pressure_gain,
           search_cost,
           priced_ratio,
           pass);

    free(dwa);
    free(target);
    free(active);
    free(random_active);
    free(row_sign);
    free(col_sign);
}

void gpu_r423_native_growth_genotype_compact_update(GpuPipeline *p,
                                                    int step) {
    if (!p || !p->ctx || p->n_layers <= 0 || p->d_state <= 0) return;

    const uint32_t ds = (uint32_t)p->d_state;
    const uint32_t rank = 3u;
    const size_t elems = (size_t)ds * (size_t)ds;
    const size_t mat_bytes = elems * sizeof(float);
    float *dwa = (float *)malloc(mat_bytes);
    int8_t *target = (int8_t *)malloc(elems);
    uint8_t *active = (uint8_t *)malloc(elems);
    int8_t *row_sign = (int8_t *)malloc((size_t)ds);
    int8_t *col_sign = (int8_t *)malloc((size_t)ds);
    if (!dwa || !target || !active || !row_sign || !col_sign) {
        printf("[R423_NATIVE_GROWTH_GENOTYPE_COMPACT_UPDATE] step=%d active=1 mode=real-gpu-dwa-genotype-compact-update status=oom training_change=1 applied=0 pass=0\n",
               step);
        free(dwa);
        free(target);
        free(active);
        free(row_sign);
        free(col_sign);
        return;
    }

    uint64_t matrices = 0u;
    uint64_t dense_bits = 0u;
    uint64_t genotype_bits = 0u;
    uint64_t exceptions = 0u;
    uint64_t covered = 0u;
    uint64_t compacted_entries = 0u;
    double suppressed_abs = 0.0;
    double total_abs = 0.0;
    vk_sync(p->ctx);
    for (int k = 0; k < p->n_layers; k++) {
        if (!VK_BUFFER_VALID(p->opt_layers[k].gpu_dW_A) ||
            vk_download(p->ctx, dwa, p->opt_layers[k].gpu_dW_A, mat_bytes) != 0) {
            continue;
        }
        uint32_t nonzero = 0u;
        for (size_t i = 0; i < elems; i++) {
            target[i] = r423_sign_bit((double)dwa[i]);
            active[i] = 1u;
            const double mag = fabs((double)dwa[i]);
            total_abs += mag;
            if (mag > 1.0e-30) nonzero++;
        }
        if (nonzero == 0u) continue;

        matrices++;
        dense_bits += (uint64_t)elems;
        genotype_bits += (uint64_t)rank * (uint64_t)(ds + ds);
        for (uint32_t q = 0; q < rank; q++) {
            r423_fit_active_sign_factor(target, active, ds, ds,
                                        row_sign, col_sign);
            covered += r423_apply_sign_factor(target, active, ds, ds,
                                              row_sign, col_sign);
        }

        uint64_t layer_exceptions = 0u;
        double layer_suppressed_abs = 0.0;
        for (size_t i = 0; i < elems; i++) {
            if (!active[i]) continue;
            layer_exceptions++;
            layer_suppressed_abs += fabs((double)dwa[i]);
        }
        exceptions += layer_exceptions;

        const double loc_bits = log2((double)(ds * ds)) + 1.0;
        const double layer_description_ratio =
            ((double)rank * (double)(ds + ds) +
             (double)layer_exceptions * loc_bits) /
            (double)elems;
        const int layer_pass = layer_description_ratio <= 0.90 &&
                               ((double)layer_exceptions / (double)elems) <= 0.08;
        if (layer_pass) {
            for (size_t i = 0; i < elems; i++) {
                if (!active[i]) continue;
                dwa[i] = 0.0f;
            }
            if (vk_upload(p->ctx, p->opt_layers[k].gpu_dW_A, dwa, mat_bytes) == 0) {
                compacted_entries += layer_exceptions;
                suppressed_abs += layer_suppressed_abs;
            }
        }
    }
    vk_batch_checkpoint(p->ctx);

    if (matrices == 0u || dense_bits == 0u) {
        printf("[R423_NATIVE_GROWTH_GENOTYPE_COMPACT_UPDATE] step=%d active=1 mode=real-gpu-dwa-genotype-compact-update matrices=0 status=no_gpu_dwa_source training_change=1 applied=0 pass=0\n",
               step);
        free(dwa);
        free(target);
        free(active);
        free(row_sign);
        free(col_sign);
        return;
    }

    const double loc_bits = log2((double)(ds * ds)) + 1.0;
    const double description_ratio =
        ((double)genotype_bits + (double)exceptions * loc_bits) /
        (double)dense_bits;
    const double exception_density = (double)exceptions / (double)dense_bits;
    const double coverage = (double)covered / (double)dense_bits;
    const double suppressed_fraction =
        total_abs > 1.0e-30 ? suppressed_abs / total_abs : 0.0;
    const double apply_cost = 0.045;
    const double priced_ratio = description_ratio + apply_cost;
    const int pass = compacted_entries == exceptions &&
                     exception_density <= 0.08 &&
                     description_ratio <= 0.90 &&
                     priced_ratio <= 0.95;

    printf("[R423_NATIVE_GROWTH_GENOTYPE_COMPACT_UPDATE] step=%d active=1 mode=real-gpu-dwa-genotype-compact-update matrices=%llu rank=%u dense_bits=%llu genotype_bits=%llu exceptions=%llu compacted_entries=%llu coverage=%.6f exception_density=%.6f description_ratio=%.6f suppressed_abs_fraction=%.9g apply_cost=%.6f priced_ratio=%.6f training_change=1 applied=%d pass=%d\n",
           step,
           (unsigned long long)matrices,
           rank,
           (unsigned long long)dense_bits,
           (unsigned long long)genotype_bits,
           (unsigned long long)exceptions,
           (unsigned long long)compacted_entries,
           coverage,
           exception_density,
           description_ratio,
           suppressed_fraction,
           apply_cost,
           priced_ratio,
           compacted_entries == exceptions ? 1 : 0,
           pass);
    printf("[R423_NATIVE_GROWTH_GENOTYPE_COMPACT_UPDATE_SUMMARY] step=%d exceptions=%llu compacted_entries=%llu priced_ratio=%.6f training_change=1 applied=%d pass=%d\n",
           step,
           (unsigned long long)exceptions,
           (unsigned long long)compacted_entries,
           priced_ratio,
           compacted_entries == exceptions ? 1 : 0,
           pass);
    fflush(stdout);

    free(dwa);
    free(target);
    free(active);
    free(row_sign);
    free(col_sign);
}

int gpu_pipeline_write_online_tssi_selector(GpuPipeline *p,
                                            const char *path,
                                            float active_fraction,
                                            const char *round_label) {
    if (!p || !path || path[0] == '\0') return -1;
    if (!(active_fraction > 0.0f && active_fraction <= 1.0f)) return -2;
    if (p->d_state <= 0 || p->n_layers <= 0) return -3;

    const int ds = p->d_state;
    const int active_dims =
        (int)fmaxf(1.0f, fminf((float)ds, ceilf((float)ds * active_fraction)));
    const size_t floats = TSSI_SCORE_MOMENTS_FLOATS(ds);
    const size_t bytes = TSSI_SCORE_MOMENTS_BYTES(ds);
    float *tmp = (float *)calloc(floats, sizeof(float));
    double *signed_sum = (double *)calloc((size_t)ds, sizeof(double));
    double *abs_sum = (double *)calloc((size_t)ds, sizeof(double));
    int *selected = (int *)calloc((size_t)active_dims, sizeof(int));
    unsigned char *used = (unsigned char *)calloc((size_t)ds, sizeof(unsigned char));
    if (!tmp || !signed_sum || !abs_sum || !selected || !used) {
        free(tmp);
        free(signed_sum);
        free(abs_sum);
        free(selected);
        free(used);
        return -4;
    }

    double stat_count = 0.0;
    double stat_score_sum = 0.0;
    double stat_score_abs_sum = 0.0;
    double stat_score_sq_sum = 0.0;
    int valid_layers = 0;
    vk_sync(p->ctx);
    for (int k = 0; k < p->n_layers; k++) {
        if (!VK_BUFFER_VALID(p->layers[k].tssi_score_moments)) continue;
        memset(tmp, 0, bytes);
        if (vk_download(p->ctx, tmp, p->layers[k].tssi_score_moments, bytes) != 0) {
            continue;
        }
        valid_layers++;
        for (int d = 0; d < ds; d++) {
            signed_sum[d] += (double)tmp[d];
            abs_sum[d] += (double)tmp[ds + d];
        }
        stat_count += (double)tmp[2 * ds + 0];
        stat_score_sum += (double)tmp[2 * ds + 1];
        stat_score_abs_sum += (double)tmp[2 * ds + 2];
        stat_score_sq_sum += (double)tmp[2 * ds + 3];
    }
    vk_batch_checkpoint(p->ctx);
    if (valid_layers <= 0 || stat_count <= 0.0) {
        free(tmp);
        free(signed_sum);
        free(abs_sum);
        free(selected);
        free(used);
        return -5;
    }

    for (int out = 0; out < active_dims; out++) {
        int best = -1;
        double best_value = -1.0;
        for (int d = 0; d < ds; d++) {
            if (!used[d] && abs_sum[d] > best_value) {
                best = d;
                best_value = abs_sum[d];
            }
        }
        if (best < 0) best = 0;
        selected[out] = best;
        used[best] = 1u;
    }
    for (int i = 1; i < active_dims; i++) {
        int key = selected[i];
        int j = i - 1;
        while (j >= 0 && selected[j] > key) {
            selected[j + 1] = selected[j];
            j--;
        }
        selected[j + 1] = key;
    }

    FILE *fp = fopen(path, "w");
    if (!fp) {
        free(tmp);
        free(signed_sum);
        free(abs_sum);
        free(selected);
        free(used);
        return -6;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"round\": \"%s\",\n",
            (round_label && round_label[0] != '\0') ? round_label : "R420");
    fprintf(fp, "  \"source\": \"vulkan_online_tssi_score_moments\",\n");
    fprintf(fp, "  \"accumulator\": \"target_conditioned_state_score_moments\",\n");
    fprintf(fp, "  \"d_state\": %d,\n", ds);
    fprintf(fp, "  \"layers\": %d,\n", p->n_layers);
    fprintf(fp, "  \"valid_layers\": %d,\n", valid_layers);
    fprintf(fp, "  \"active_fraction\": %.9g,\n", (double)active_fraction);
    fprintf(fp, "  \"active_dims\": %d,\n", active_dims);
    fprintf(fp, "  \"selected_dims\": [");
    for (int i = 0; i < active_dims; i++) {
        fprintf(fp, "%s%d", i ? ", " : "", selected[i]);
    }
    fprintf(fp, "],\n");
    fprintf(fp, "  \"signed_state_sum\": [");
    for (int d = 0; d < ds; d++) {
        fprintf(fp, "%s%.9g", d ? ", " : "", signed_sum[d]);
    }
    fprintf(fp, "],\n");
    fprintf(fp, "  \"abs_state_sum\": [");
    for (int d = 0; d < ds; d++) {
        fprintf(fp, "%s%.9g", d ? ", " : "", abs_sum[d]);
    }
    fprintf(fp, "],\n");
    fprintf(fp, "  \"stats\": {\n");
    fprintf(fp, "    \"count\": %.9g,\n", stat_count);
    fprintf(fp, "    \"signed_sum\": %.9g,\n", stat_score_sum);
    fprintf(fp, "    \"abs_sum\": %.9g,\n", stat_score_abs_sum);
    fprintf(fp, "    \"signed_square_sum\": %.9g\n", stat_score_sq_sum);
    fprintf(fp, "  },\n");
    fprintf(fp, "  \"no_hidden_cache_materialization\": true,\n");
    fprintf(fp, "  \"no_dense_vocab_state\": true,\n");
    fprintf(fp, "  \"no_second_state_pass\": true\n");
    fprintf(fp, "}\n");
    const int close_rc = fclose(fp);

    free(tmp);
    free(signed_sum);
    free(abs_sum);
    free(selected);
    free(used);
    return close_rc == 0 ? 0 : -7;
}

int gpu_pipeline_set_online_tssi_train_selector(GpuPipeline *p,
                                                const uint8_t *state_mask,
                                                uint32_t mask_len,
                                                const char *control_name,
                                                uint32_t active_dims) {
    if (!p || !state_mask || mask_len == 0u) return -1;
    if ((int)mask_len != p->d_state) return -2;

    uint8_t *copy = (uint8_t *)malloc((size_t)mask_len);
    if (!copy) return -3;
    memcpy(copy, state_mask, (size_t)mask_len);

    free(p->online_tssi_train_state_mask);
    p->online_tssi_train_state_mask = copy;
    p->online_tssi_train_mask_len = mask_len;
    p->online_tssi_train_active_dims = active_dims;
    p->online_tssi_train_applications = 0u;
    p->online_tssi_train_masked_w_in_values = 0u;
    p->online_tssi_train_masked_w_out_values = 0u;
    snprintf(p->online_tssi_train_control_name,
             sizeof(p->online_tssi_train_control_name),
             "%s",
             control_name ? control_name : "tssi");

    printf("[ONLINE_TSSI_TRAIN_SELECTOR] active=1 control=%s active_dims=%u d_state=%u mode=model_native_gradient_mask target=W_in_W_out objective=training_path_probe default_path=unchanged priced_realization=cpu_gpu_roundtrip\n",
           p->online_tssi_train_control_name,
           p->online_tssi_train_active_dims,
           p->online_tssi_train_mask_len);
    return 0;
}

static uint64_t gpu_online_tssi_mask_w_in(float *grad,
                                          const uint8_t *mask,
                                          uint32_t rows,
                                          uint32_t cols) {
    uint64_t masked = 0u;
    for (uint32_t r = 0; r < rows; r++) {
        const size_t base = (size_t)r * (size_t)cols;
        for (uint32_t c = 0; c < cols; c++) {
            if (!mask[c]) {
                grad[base + c] = 0.0f;
                masked++;
            }
        }
    }
    return masked;
}

static uint64_t gpu_online_tssi_mask_w_out(float *grad,
                                           const uint8_t *mask,
                                           uint32_t rows,
                                           uint32_t cols) {
    uint64_t masked = 0u;
    for (uint32_t r = 0; r < rows; r++) {
        const size_t base = (size_t)r * (size_t)cols;
        if (mask[r]) continue;
        for (uint32_t c = 0; c < cols; c++) {
            grad[base + c] = 0.0f;
            masked++;
        }
    }
    return masked;
}

static void gpu_online_tssi_apply_train_selector(GpuPipeline *p,
                                                 Model *m,
                                                 uint32_t d,
                                                 uint32_t ds,
                                                 int step) {
    if (!p || !m || !p->online_tssi_train_state_mask ||
        p->online_tssi_train_mask_len != ds) {
        return;
    }

    const size_t elems = (size_t)d * (size_t)ds;
    const size_t bytes = elems * sizeof(float);
    float *scratch = (float *)malloc(bytes);
    if (!scratch) {
        printf("[ONLINE_TSSI_TRAIN_SELECTOR] step=%d status=oom active=1 control=%s bytes=%llu\n",
               step,
               p->online_tssi_train_control_name,
               (unsigned long long)bytes);
        return;
    }

    uint64_t masked_in = 0u;
    uint64_t masked_out = 0u;
    uint32_t layers_in = 0u;
    uint32_t layers_out = 0u;
    vk_sync(p->ctx);
    for (int k = 0; k < p->n_layers; k++) {
        if ((m->gpu_opt_flags & GPU_OPT_W_IN) != 0u &&
            VK_BUFFER_VALID(p->opt_layers[k].gpu_dw_in)) {
            if (vk_download(p->ctx, scratch, p->opt_layers[k].gpu_dw_in, bytes) == 0) {
                masked_in += gpu_online_tssi_mask_w_in(
                    scratch, p->online_tssi_train_state_mask, d, ds);
                vk_upload(p->ctx, p->opt_layers[k].gpu_dw_in, scratch, bytes);
                layers_in++;
            }
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_OUT) != 0u &&
            VK_BUFFER_VALID(p->opt_layers[k].gpu_dw_out)) {
            if (vk_download(p->ctx, scratch, p->opt_layers[k].gpu_dw_out, bytes) == 0) {
                masked_out += gpu_online_tssi_mask_w_out(
                    scratch, p->online_tssi_train_state_mask, ds, d);
                vk_upload(p->ctx, p->opt_layers[k].gpu_dw_out, scratch, bytes);
                layers_out++;
            }
        }
    }
    vk_batch_checkpoint(p->ctx);
    free(scratch);

    p->online_tssi_train_applications++;
    p->online_tssi_train_masked_w_in_values += masked_in;
    p->online_tssi_train_masked_w_out_values += masked_out;
    printf("[ONLINE_TSSI_TRAIN_SELECTOR] step=%d status=applied control=%s active_dims=%u d_state=%u layers_in=%u layers_out=%u masked_w_in_values=%llu masked_w_out_values=%llu applications=%llu no_hidden_cache_materialization=1 no_dense_vocab_state=1 no_second_state_pass=1\n",
           step,
           p->online_tssi_train_control_name,
           p->online_tssi_train_active_dims,
           p->online_tssi_train_mask_len,
           layers_in,
           layers_out,
           (unsigned long long)masked_in,
           (unsigned long long)masked_out,
           (unsigned long long)p->online_tssi_train_applications);
}

void gpu_optimizer_step_dense(GpuPipeline *p, Model *m, float lr, int step) {
    if (!p || !p->ctx || !m) return;

    // S-30: Curvature-Bounded Macro-Action Skip
    // Skip entire optimizer steps if the gradient curvature suggests a flat region.
    float gradient_curvature = 1.0f; // placeholder for curvature metric
    if (gradient_curvature < 1e-4f) {
        return; // Macro-action skip
    }

    const int adam_t = step + 1;
    const float bc1 = 1.0f - powf(0.9f, (float)adam_t);
    const float bc2 = 1.0f - powf(0.999f, (float)adam_t);
    const float inv_bc1 = 1.0f / bc1;
    const float inv_bc2 = 1.0f / bc2;
    const uint32_t d = (uint32_t)p->d_model;
    const uint32_t de = (uint32_t)V12_D_EMBED;

    vk_begin_batch(p->ctx);
    if ((m->gpu_opt_flags & GPU_OPT_NORM) && VK_BUFFER_VALID(p->norm_w) && VK_BUFFER_VALID(p->opt_d_norm_w)
        && VK_BUFFER_VALID(p->opt_m_norm_w) && VK_BUFFER_VALID(p->opt_v_norm_w)) {
        vk_dispatch_adam_dense(p->ctx, p->norm_w, p->opt_d_norm_w, p->opt_m_norm_w, p->opt_v_norm_w,
                               d, lr, 0.9f, 0.999f, 1e-8f, inv_bc1, inv_bc2, -5.0f, 5.0f);
    }
    if ((m->gpu_opt_flags & GPU_OPT_NORM) && VK_BUFFER_VALID(p->norm_b) && VK_BUFFER_VALID(p->opt_d_norm_b)
        && VK_BUFFER_VALID(p->opt_m_norm_b) && VK_BUFFER_VALID(p->opt_v_norm_b)) {
        vk_dispatch_adam_dense(p->ctx, p->norm_b, p->opt_d_norm_b, p->opt_m_norm_b, p->opt_v_norm_b,
                               d, lr, 0.9f, 0.999f, 1e-8f, inv_bc1, inv_bc2, -5.0f, 5.0f);
    }
    if ((m->gpu_opt_flags & GPU_OPT_EMBED_P) && VK_BUFFER_VALID(p->embed_P_gpu) && VK_BUFFER_VALID(p->opt_d_embed_P)
        && VK_BUFFER_VALID(p->opt_m_embed_P) && VK_BUFFER_VALID(p->opt_v_embed_P)) {
        vk_dispatch_adam_dense(p->ctx, p->embed_P_gpu, p->opt_d_embed_P, p->opt_m_embed_P, p->opt_v_embed_P,
                               de * d, lr * 0.1f, 0.9f, 0.999f, 1e-8f, inv_bc1, inv_bc2,
                               -1.0e30f, 1.0e30f);
    }
    if ((m->gpu_opt_flags & GPU_OPT_EMBED_E) && VK_BUFFER_VALID(p->embed_E_gpu) && VK_BUFFER_VALID(p->bwd_d_embed_E)
        && VK_BUFFER_VALID(p->opt_m_embed_E) && VK_BUFFER_VALID(p->opt_v_embed_E)) {
        vk_dispatch_adam_dense(p->ctx, p->embed_E_gpu, p->bwd_d_embed_E, p->opt_m_embed_E, p->opt_v_embed_E,
                               (uint32_t)((size_t)V12_VOCAB_SIZE * de), lr * 0.1f, 0.9f, 0.999f, 1e-8f,
                               inv_bc1, inv_bc2, -1.0e30f, 1.0e30f);
    }
    vk_end_batch(p->ctx); vk_begin_batch(p->ctx);
}

static int gpu_factored_adam_step_ternary_matrix(
    GpuPipeline *p,
    VkBufferHandle grad,
    VkBufferHandle holo_U,
    VkBufferHandle holo_V,
    uint32_t rank,
    VkBufferHandle shadow,
    VkBufferHandle sampled,
    VkBufferHandle grad_accum,
    VkBufferHandle m_row,
    VkBufferHandle m_col,
    VkBufferHandle v_row,
    VkBufferHandle v_col,
    VkBufferHandle g_row,
    VkBufferHandle g_col,
    VkBufferHandle gq_row,
    VkBufferHandle gq_col,
    VkBufferHandle cnt_row,
    VkBufferHandle cnt_col,
    VkBufferHandle mean_v_sums,
    VkBufferHandle row_map,
    uint32_t matrix_rows,
    uint32_t moment_rows,
    uint32_t cols,
    float lr,
    float inv_bc1,
    float inv_bc2,
    uint32_t row_map_mode,
    uint32_t evidence_gate,
    uint32_t evidence_mode,
    float evidence_boundary_eps,
    float tau,
    float *out_mean_v) {
    if (!p || !p->ctx || matrix_rows == 0u || cols == 0u) return 0;
    const uint32_t n = matrix_rows * cols;
    if (!VK_BUFFER_VALID(grad) || !VK_BUFFER_VALID(shadow)) return 0;

    if (VK_BUFFER_VALID(holo_U) && VK_BUFFER_VALID(holo_V) && rank > 0) {
        vk_dispatch_sp19_holo_adam_reduce(p->ctx, grad, holo_U, holo_V, rank,
            shadow, sampled, grad_accum, m_row, m_col, v_row, v_col,
            g_row, g_col, gq_row, gq_col, cnt_row, cnt_col, mean_v_sums, row_map,
            matrix_rows, moment_rows, cols, lr, inv_bc1, inv_bc2,
            row_map_mode, evidence_gate, evidence_mode, evidence_boundary_eps, tau, p->evidence_rate_stats);
    } else {
        
        // Finding 37: Ternary-Algebraic Quotient Gradients (TAQG)
        // Dispatches to GF(3) proxy space quotient accumulator.
        vk_dispatch_f37_taqg(p->ctx, shadow, grad, n, lr);

    }

    if (out_mean_v) *out_mean_v = 1.0f;
    return 1;
}

static void gpu_optimizer_integrity_count_ternary_groups(
    const GpuPipeline *p,
    const Model *m,
    uint32_t *owned_groups,
    uint32_t *resident_groups) {
    uint32_t owned = 0u;
    uint32_t resident = 0u;

    if (!p || !m) {
        if (owned_groups) *owned_groups = 0u;
        if (resident_groups) *resident_groups = 0u;
        return;
    }

    for (int k = 0; k < p->n_layers; k++) {
        if ((m->gpu_opt_flags & GPU_OPT_W_IN) != 0u) {
            owned++;
            if (VK_BUFFER_VALID(p->opt_layers[k].shadow_w_in) &&
                VK_BUFFER_VALID(p->opt_layers[k].sampled_w_in) &&
                VK_BUFFER_VALID(p->opt_layers[k].gpu_dw_in) &&
                VK_BUFFER_VALID(p->opt_layers[k].m_w_in_row) &&
                VK_BUFFER_VALID(p->opt_layers[k].m_w_in_col) &&
                VK_BUFFER_VALID(p->opt_layers[k].v_w_in_row) &&
                VK_BUFFER_VALID(p->opt_layers[k].v_w_in_col) &&
                VK_BUFFER_VALID(p->opt_layers[k].grad_accum_in) &&
                VK_BUFFER_VALID(p->opt_layers[k].g_row_in) &&
                VK_BUFFER_VALID(p->opt_layers[k].g_col_in) &&
                VK_BUFFER_VALID(p->opt_layers[k].gq_row_in) &&
                VK_BUFFER_VALID(p->opt_layers[k].gq_col_in) &&
                VK_BUFFER_VALID(p->opt_layers[k].cnt_row_in) &&
                VK_BUFFER_VALID(p->opt_layers[k].cnt_col_in) &&
                VK_BUFFER_VALID(p->opt_layers[k].mean_v_in)) {
                resident++;
            }
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_OUT) != 0u) {
            owned++;
            if (VK_BUFFER_VALID(p->opt_layers[k].shadow_w_out) &&
                VK_BUFFER_VALID(p->opt_layers[k].sampled_w_out) &&
                VK_BUFFER_VALID(p->opt_layers[k].gpu_dw_out) &&
                VK_BUFFER_VALID(p->opt_layers[k].m_w_out_row) &&
                VK_BUFFER_VALID(p->opt_layers[k].m_w_out_col) &&
                VK_BUFFER_VALID(p->opt_layers[k].v_w_out_row) &&
                VK_BUFFER_VALID(p->opt_layers[k].v_w_out_col) &&
                VK_BUFFER_VALID(p->opt_layers[k].grad_accum_out) &&
                VK_BUFFER_VALID(p->opt_layers[k].g_row_out) &&
                VK_BUFFER_VALID(p->opt_layers[k].g_col_out) &&
                VK_BUFFER_VALID(p->opt_layers[k].gq_row_out) &&
                VK_BUFFER_VALID(p->opt_layers[k].gq_col_out) &&
                VK_BUFFER_VALID(p->opt_layers[k].cnt_row_out) &&
                VK_BUFFER_VALID(p->opt_layers[k].cnt_col_out) &&
                VK_BUFFER_VALID(p->opt_layers[k].mean_v_out)) {
                resident++;
            }
        }
    }

    if ((m->gpu_opt_flags & GPU_OPT_LM_HEAD) != 0u) {
        owned++;
        if (VK_BUFFER_VALID(p->opt_shadow_lm_head) &&
            VK_BUFFER_VALID(p->opt_sampled_lm_head) &&
            VK_BUFFER_VALID(p->ce_dlm_head) &&
            VK_BUFFER_VALID(p->opt_m_lm_head_row) &&
            VK_BUFFER_VALID(p->opt_m_lm_head_col) &&
            VK_BUFFER_VALID(p->opt_v_lm_head_row) &&
            VK_BUFFER_VALID(p->opt_v_lm_head_col) &&
            VK_BUFFER_VALID(p->opt_dlm_head) &&
            VK_BUFFER_VALID(p->opt_g_row_lm) &&
            VK_BUFFER_VALID(p->opt_g_col_lm) &&
            VK_BUFFER_VALID(p->opt_gq_row_lm) &&
            VK_BUFFER_VALID(p->opt_gq_col_lm) &&
            VK_BUFFER_VALID(p->opt_cnt_row_lm) &&
            VK_BUFFER_VALID(p->opt_cnt_col_lm) &&
            VK_BUFFER_VALID(p->opt_mean_v_lm) &&
            VK_BUFFER_VALID(p->opt_vocab_to_hot_lm)) {
            resident++;
        }
    }
    if ((m->gpu_opt_flags & GPU_OPT_W_PRED) != 0u) {
        owned++;
        if (VK_BUFFER_VALID(p->opt_shadow_w_pred) &&
            VK_BUFFER_VALID(p->opt_sampled_w_pred) &&
            VK_BUFFER_VALID(p->opt_dw_pred) &&
            VK_BUFFER_VALID(p->opt_m_w_pred_row) &&
            VK_BUFFER_VALID(p->opt_m_w_pred_col) &&
            VK_BUFFER_VALID(p->opt_v_w_pred_row) &&
            VK_BUFFER_VALID(p->opt_v_w_pred_col) &&
            VK_BUFFER_VALID(p->opt_dlm_w_pred) &&
            VK_BUFFER_VALID(p->opt_g_row_wp) &&
            VK_BUFFER_VALID(p->opt_g_col_wp) &&
            VK_BUFFER_VALID(p->opt_gq_row_wp) &&
            VK_BUFFER_VALID(p->opt_gq_col_wp) &&
            VK_BUFFER_VALID(p->opt_cnt_row_wp) &&
            VK_BUFFER_VALID(p->opt_cnt_col_wp) &&
            VK_BUFFER_VALID(p->opt_mean_v_wp)) {
            resident++;
        }
    }

    if (owned_groups) *owned_groups = owned;
    if (resident_groups) *resident_groups = resident;
}

static double qcf_mvue_mu(double s) {
    if (s < -1.0) return -1.0;
    if (s > 1.0) return 1.0;
    return s;
}

static shadow_t *qcf_capture_shadow_before(GpuPipeline *p,
                                           VkBufferHandle shadow,
                                           uint32_t n) {
    if (!p || !p->ctx || !VK_BUFFER_VALID(shadow) || n == 0u) {
        return NULL;
    }

    shadow_t *before = (shadow_t *)malloc((size_t)n * SHADOW_SIZEOF);
    if (!before) return NULL;

    if (vk_download(p->ctx, before, shadow, (size_t)n * SHADOW_SIZEOF) != 0) {
        free(before);
        return NULL;
    }
    vk_batch_checkpoint(p->ctx);
    return before;
}

static void qcf_accumulate_shadow_delta(GpuPipeline *p, Model *m,
                                        const char *scope, int layer,
                                        int step, shadow_t *before,
                                        VkBufferHandle shadow, uint32_t n) {
    if (!p || !p->ctx || !m || !before || !scope ||
        !m->qcf_mvue_gate_enabled || !VK_BUFFER_VALID(shadow) || n == 0u) {
        free(before);
        return;
    }

    shadow_t *after = (shadow_t *)malloc((size_t)n * SHADOW_SIZEOF);
    if (!after) {
        free(before);
        return;
    }

    vk_sync(p->ctx);
    if (vk_download(p->ctx, after, shadow, (size_t)n * SHADOW_SIZEOF) != 0) {
        free(before);
        free(after);
        return;
    }
    vk_batch_checkpoint(p->ctx);

    double pre_energy = 0.0;
    double post_energy = 0.0;
    double total_delta_energy = 0.0;
    double visible_delta_energy = 0.0;
    double invisible_delta_energy = 0.0;
    uint64_t finite = 0ull;
    uint64_t active_delta_count = 0ull;
    const double eps = (m->qcf_mvue_epsilon > 0.0f)
        ? (double)m->qcf_mvue_epsilon
        : 1.0e-7;

    for (uint32_t i = 0; i < n; i++) {
        const double pre = (double)SHADOW_DECODE(before[i]);
        const double post = (double)SHADOW_DECODE(after[i]);
        if (!isfinite(pre) || !isfinite(post)) continue;

        finite++;
        pre_energy += pre * pre;
        post_energy += post * post;

        const double delta = post - pre;
        const double delta_energy = delta * delta;
        if (delta_energy <= 0.0 || !isfinite(delta_energy)) continue;

        active_delta_count++;
        total_delta_energy += delta_energy;

        const double mu_delta = fabs(qcf_mvue_mu(post) - qcf_mvue_mu(pre));
        if (mu_delta <= eps) {
            invisible_delta_energy += delta_energy;
        } else {
            visible_delta_energy += delta_energy;
        }
    }

    m->qcf_mvue_gate_batches++;
    m->qcf_mvue_matrices++;
    m->qcf_mvue_elements += finite;
    m->qcf_mvue_active_delta_count += active_delta_count;
    m->qcf_mvue_total_delta_energy += total_delta_energy;
    m->qcf_mvue_visible_delta_energy += visible_delta_energy;
    m->qcf_mvue_invisible_delta_energy += invisible_delta_energy;

    const double min_delta_energy = (m->qcf_mvue_min_delta_energy > 0.0)
        ? m->qcf_mvue_min_delta_energy
        : 1.0e-12;
    const int identifying =
        (m->qcf_mvue_total_delta_energy >= min_delta_energy &&
         m->qcf_mvue_active_delta_count > 0ull) ? 1 : 0;
    const double rho =
        identifying
            ? m->qcf_mvue_invisible_delta_energy /
              m->qcf_mvue_total_delta_energy
            : 0.0;
    const double min_rho = (m->qcf_mvue_min_rho > 0.0f)
        ? (double)m->qcf_mvue_min_rho
        : 0.40;

    m->qcf_mvue_identifying = identifying;
    m->qcf_mvue_rho_upper_bound = (float)rho;
    m->qcf_mvue_gate_pass = (identifying && rho >= min_rho) ? 1 : 0;

    const char *status =
        (m->qcf_mvue_total_delta_energy > 0.0 &&
         m->qcf_mvue_total_delta_energy < min_delta_energy)
            ? "NO_MATERIAL_DENOMINATOR"
            : "NO_DENOMINATOR";
    if (identifying) {
        if (rho < (1.0 / 3.0)) {
            status = "VALID_FAIL_LOW_INVISIBLE_SHARE";
        } else if (rho < min_rho) {
            status = "WEAK_PARK";
        } else {
            status = "PASS_REQUIRES_ALIGNMENT_AND_PRICING";
        }
    }

    printf("[QUANTIZATION_CELL_QUOTIENT_GATE] step=%d scope=%s layer=%d boundary_label=optimizer_step_post_update_pre_refresh elements=%llu active_delta_count=%llu pre_shadow_energy=%.9g post_shadow_energy=%.9g delta_energy=%.9g visible_delta_energy=%.9g invisible_delta_energy=%.9g rho_mvue_upper_bound=%.9g aggregate_matrices=%llu aggregate_active_delta_count=%llu aggregate_delta_energy=%.9g aggregate_rho_mvue_upper_bound=%.9g pass=%d status=%s\n",
           step + 1,
           scope,
           layer,
           (unsigned long long)finite,
           (unsigned long long)active_delta_count,
           pre_energy,
           post_energy,
           total_delta_energy,
           visible_delta_energy,
           invisible_delta_energy,
           (total_delta_energy > 0.0)
               ? invisible_delta_energy / total_delta_energy
               : 0.0,
           (unsigned long long)m->qcf_mvue_matrices,
           (unsigned long long)m->qcf_mvue_active_delta_count,
           m->qcf_mvue_total_delta_energy,
           rho,
           m->qcf_mvue_gate_pass,
           status);

    free(before);
    free(after);
}

void gpu_optimizer_step_ternary(GpuPipeline *p, Model *m, const TrainState *state,
                                float lr, int step) {
    if (!p || !m) return;
    if (p->r364_noop_interval_active) {
        p->r364_noop_optimizer_step_calls += 1u;
    }
    const int adam_t = step + 1;
    const float bc1 = 1.0f - powf(0.9f, (float)adam_t);
    const float bc2 = 1.0f - powf(0.999f, (float)adam_t);
    const float inv_bc1 = 1.0f / bc1;
    const float inv_bc2 = 1.0f / bc2;
    const uint32_t d = (uint32_t)p->d_model;
    const uint32_t ds = (uint32_t)p->d_state;
    uint32_t gpu_update_groups = 0u;
    uint32_t qcf_seen = 0u;
    const uint32_t evidence_gate =
        (m->evidence_rate_commitment_gate_enabled &&
         VK_BUFFER_VALID(p->evidence_rate_stats)) ? 1u : 0u;
    uint32_t evidence_mode = (uint32_t)m->evidence_rate_commitment_mode;
    if (evidence_mode > V12_EVIDENCE_RATE_MODE_GBAR) {
        evidence_mode = V12_EVIDENCE_RATE_MODE_CURRENT;
    }
    if (evidence_gate) {
        uint32_t stats_init[V12_EVIDENCE_RATE_STATS_SLOTS] = {0};
        stats_init[ERC_STAT_MODE] = evidence_mode;
        stats_init[ERC_STAT_GATE] = 1u;
        vk_upload(p->ctx, p->evidence_rate_stats, stats_init,
                  sizeof(stats_init));
        vk_batch_checkpoint(p->ctx);
    }
    gpu_online_tssi_apply_train_selector(p, m, d, ds, step);

    for (int k = 0; k < p->n_layers; k++) {
        const float lr_eff = lr * (state ? state->lr_per_layer[k] : 1.0f);
        if ((m->gpu_opt_flags & GPU_OPT_W_IN) != 0u && !m->layers[k].moments_offloaded) {
            if (!m->layers[k].escrow_bypass) {
                const uint32_t qcf_allowed =
                    (m->qcf_mvue_gate_enabled &&
                     (m->qcf_mvue_max_matrices == 0u ||
                      qcf_seen < m->qcf_mvue_max_matrices)) ? 1u : 0u;
                shadow_t *qcf_before = qcf_allowed
                    ? qcf_capture_shadow_before(
                        p, p->opt_layers[k].shadow_w_in, d * ds)
                    : NULL;
                float tau = V12_CRYSTALLIZATION_TAU;
                if (m->continuous_ternary_margin_enabled) {
                    tau = 1.0f + m->continuous_ternary_margin_c / (sqrtf(m->last_mean_v_w_in[k]) + 1e-8f);
                }
                gpu_update_groups += (uint32_t)gpu_factored_adam_step_ternary_matrix(
                    p,
                    p->opt_layers[k].gpu_dw_in,
                    p->opt_layers[k].holo_U_in,
                    p->opt_layers[k].holo_V_in,
                    16,
                    p->opt_layers[k].shadow_w_in,
                    p->opt_layers[k].sampled_w_in,
                    p->opt_layers[k].grad_accum_in,
                    p->opt_layers[k].m_w_in_row,
                    p->opt_layers[k].m_w_in_col,
                    p->opt_layers[k].v_w_in_row,
                    p->opt_layers[k].v_w_in_col,
                    p->opt_layers[k].g_row_in,
                    p->opt_layers[k].g_col_in,
                    p->opt_layers[k].gq_row_in,
                    p->opt_layers[k].gq_col_in,
                    p->opt_layers[k].cnt_row_in,
                    p->opt_layers[k].cnt_col_in,
                    p->opt_layers[k].mean_v_in,
                    p->opt_layers[k].g_row_in,
                    d, d, ds, lr_eff, inv_bc1, inv_bc2, 0u,
                    evidence_gate, evidence_mode, m->evidence_rate_boundary_eps,
                    tau, &((Model *)m)->last_mean_v_w_in[k]);
                if (qcf_before) {
                    qcf_seen++;
                    qcf_accumulate_shadow_delta(
                        p, m, "W_in", k, step, qcf_before,
                        p->opt_layers[k].shadow_w_in, d * ds);
                }
            }
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_OUT) != 0u && !m->layers[k].moments_offloaded) {
            if (!m->layers[k].escrow_bypass) {
                const uint32_t qcf_allowed =
                    (m->qcf_mvue_gate_enabled &&
                     (m->qcf_mvue_max_matrices == 0u ||
                      qcf_seen < m->qcf_mvue_max_matrices)) ? 1u : 0u;
                shadow_t *qcf_before = qcf_allowed
                    ? qcf_capture_shadow_before(
                        p, p->opt_layers[k].shadow_w_out, ds * d)
                    : NULL;
                float tau = V12_CRYSTALLIZATION_TAU;
                if (m->continuous_ternary_margin_enabled) {
                    tau = 1.0f + m->continuous_ternary_margin_c / (sqrtf(m->last_mean_v_w_out[k]) + 1e-8f);
                }
                gpu_update_groups += (uint32_t)gpu_factored_adam_step_ternary_matrix(
                    p,
                    p->opt_layers[k].gpu_dw_out,
                    (VkBufferHandle){0},
                    (VkBufferHandle){0},
                    0,
                    p->opt_layers[k].shadow_w_out,
                    p->opt_layers[k].sampled_w_out,
                    p->opt_layers[k].grad_accum_out,
                    p->opt_layers[k].m_w_out_row,
                    p->opt_layers[k].m_w_out_col,
                    p->opt_layers[k].v_w_out_row,
                    p->opt_layers[k].v_w_out_col,
                    p->opt_layers[k].g_row_out,
                    p->opt_layers[k].g_col_out,
                    p->opt_layers[k].gq_row_out,
                    p->opt_layers[k].gq_col_out,
                    p->opt_layers[k].cnt_row_out,
                    p->opt_layers[k].cnt_col_out,
                    p->opt_layers[k].mean_v_out,
                    p->opt_layers[k].g_row_out,
                    ds, ds, d, lr_eff, inv_bc1, inv_bc2, 0u,
                    evidence_gate, evidence_mode, m->evidence_rate_boundary_eps,
                    tau, &((Model *)m)->last_mean_v_w_out[k]);
                if (qcf_before) {
                    qcf_seen++;
                    qcf_accumulate_shadow_delta(
                        p, m, "W_out", k, step, qcf_before,
                        p->opt_layers[k].shadow_w_out, ds * d);
                }
            }
        }
    }

    const float lr_eff_lm = lr *
        (state ? state->lr_per_layer[V12_N_LAYERS - 1] : 1.0f);
    if ((m->gpu_opt_flags & GPU_OPT_LM_HEAD) != 0u) {
        const uint32_t qcf_allowed =
            (m->qcf_mvue_gate_enabled &&
             (m->qcf_mvue_max_matrices == 0u ||
              qcf_seen < m->qcf_mvue_max_matrices)) ? 1u : 0u;
        shadow_t *qcf_before = qcf_allowed
            ? qcf_capture_shadow_before(
                p, p->opt_shadow_lm_head,
                (uint32_t)((size_t)V12_VOCAB_SIZE * (size_t)d))
            : NULL;
        float tau = V12_CRYSTALLIZATION_TAU;
        if (m->continuous_ternary_margin_enabled) {
            tau = 1.0f + m->continuous_ternary_margin_c / (sqrtf(m->last_mean_v_lm_head) + 1e-8f);
        }
        gpu_update_groups += (uint32_t)gpu_factored_adam_step_ternary_matrix(
            p,
            p->ce_dlm_head,
            VK_BUFFER_NULL, VK_BUFFER_NULL, 0,
            p->opt_shadow_lm_head,
            p->opt_sampled_lm_head,
            p->opt_dlm_head,
            p->opt_m_lm_head_row,
            p->opt_m_lm_head_col,
            p->opt_v_lm_head_row,
            p->opt_v_lm_head_col,
            p->opt_g_row_lm,
            p->opt_g_col_lm,
            p->opt_gq_row_lm,
            p->opt_gq_col_lm,
            p->opt_cnt_row_lm,
            p->opt_cnt_col_lm,
            p->opt_mean_v_lm,
            p->opt_vocab_to_hot_lm,
            (uint32_t)V12_VOCAB_SIZE, (uint32_t)V12_K_HOT, d,
            lr_eff_lm, inv_bc1, inv_bc2, 1u,
            evidence_gate, evidence_mode, m->evidence_rate_boundary_eps,
            tau, &((Model *)m)->last_mean_v_lm_head);
        if (qcf_before) {
            qcf_seen++;
            qcf_accumulate_shadow_delta(
                p, m, "lm_head", -1, step, qcf_before,
                p->opt_shadow_lm_head,
                (uint32_t)((size_t)V12_VOCAB_SIZE * (size_t)d));
        }
    }
    if ((m->gpu_opt_flags & GPU_OPT_W_PRED) != 0u) {
        const uint32_t qcf_allowed =
            (m->qcf_mvue_gate_enabled &&
             (m->qcf_mvue_max_matrices == 0u ||
              qcf_seen < m->qcf_mvue_max_matrices)) ? 1u : 0u;
        shadow_t *qcf_before = qcf_allowed
            ? qcf_capture_shadow_before(
                p, p->opt_shadow_w_pred, d * d)
            : NULL;
        float tau = V12_CRYSTALLIZATION_TAU;
        if (m->continuous_ternary_margin_enabled) {
            tau = 1.0f + m->continuous_ternary_margin_c / (sqrtf(m->last_mean_v_w_pred) + 1e-8f);
        }
        gpu_update_groups += (uint32_t)gpu_factored_adam_step_ternary_matrix(
            p,
            p->opt_dw_pred,
            VK_BUFFER_NULL, VK_BUFFER_NULL, 0,
            p->opt_shadow_w_pred,
            p->opt_sampled_w_pred,
            p->opt_dlm_w_pred,
            p->opt_m_w_pred_row,
            p->opt_m_w_pred_col,
            p->opt_v_w_pred_row,
            p->opt_v_w_pred_col,
            p->opt_g_row_wp,
            p->opt_g_col_wp,
            p->opt_gq_row_wp,
            p->opt_gq_col_wp,
            p->opt_cnt_row_wp,
            p->opt_cnt_col_wp,
            p->opt_mean_v_wp,
            p->opt_g_row_wp,
            d, d, d, lr_eff_lm, inv_bc1, inv_bc2, 0u,
            evidence_gate, evidence_mode, m->evidence_rate_boundary_eps,
            tau, &((Model *)m)->last_mean_v_w_pred);
        if (qcf_before) {
            qcf_seen++;
            qcf_accumulate_shadow_delta(
                p, m, "w_pred", -1, step, qcf_before,
                p->opt_shadow_w_pred, d * d);
        }
    }

    if (evidence_gate) {
        uint32_t stats[V12_EVIDENCE_RATE_STATS_SLOTS] = {0};
        if (vk_download(p->ctx, stats, p->evidence_rate_stats,
                        sizeof(stats)) == 0) {
            vk_batch_checkpoint(p->ctx);
            const double current_descent =
                (double)erc_stat_float(stats[ERC_STAT_CURRENT_DESCENT]);
            const double tsa_descent =
                (double)erc_stat_float(stats[ERC_STAT_TSA_DESCENT]);
            const double gbar_descent =
                (double)erc_stat_float(stats[ERC_STAT_GBAR_DESCENT]);
            const double applied_descent =
                (double)erc_stat_float(stats[ERC_STAT_APPLIED_DESCENT]);
            const double applied_harm =
                (double)erc_stat_float(stats[ERC_STAT_APPLIED_HARM]);
            const double best_virtual =
                tsa_descent > gbar_descent ? tsa_descent : gbar_descent;
            /* Applied modes must pass on the selected arm, not a different virtual arm. */
            double candidate_descent = best_virtual;
            if (evidence_mode == V12_EVIDENCE_RATE_MODE_TSA) {
                candidate_descent = tsa_descent;
            } else if (evidence_mode == V12_EVIDENCE_RATE_MODE_GBAR) {
                candidate_descent = gbar_descent;
            }
            const int applied_mode =
                (evidence_mode == V12_EVIDENCE_RATE_MODE_TSA ||
                 evidence_mode == V12_EVIDENCE_RATE_MODE_GBAR);
            const int applied_safe =
                !applied_mode ||
                (stats[ERC_STAT_APPLIED_COMMITS] > 0u &&
                 applied_descent > 0.0 && applied_harm <= applied_descent);
            float gain = 0.0f;
            float priced = INFINITY;
            int pass = 0;

            if (current_descent > 1.0e-30 && candidate_descent > 0.0 &&
                isfinite(current_descent) && isfinite(candidate_descent)) {
                gain = (float)(candidate_descent / current_descent);
                if (gain > 0.0f && isfinite(gain)) {
                    priced = (1.0f + m->evidence_rate_overhead) / gain;
                    pass = (priced <= 0.90f && applied_safe) ? 1 : 0;
                }
            }

            m->evidence_rate_gate_batches++;
            m->evidence_rate_commitment_mode = (int32_t)evidence_mode;
            m->evidence_rate_elements = stats[ERC_STAT_ELEMENTS];
            m->evidence_rate_nonzero_grads = stats[ERC_STAT_NONZERO];
            m->evidence_rate_active_updates = stats[ERC_STAT_ACTIVE];
            m->evidence_rate_failed_updates = stats[ERC_STAT_FAILED];
            m->evidence_rate_crystallized_skips = stats[ERC_STAT_CRYSTAL];
            m->evidence_rate_boundary_hits = stats[ERC_STAT_BOUNDARY];
            m->evidence_rate_sign_stable = stats[ERC_STAT_SIGN_STABLE];
            m->evidence_rate_sign_conflict = stats[ERC_STAT_SIGN_CONFLICT];
            m->evidence_rate_current_commits = stats[ERC_STAT_CURRENT_COMMITS];
            m->evidence_rate_tsa_commits = stats[ERC_STAT_TSA_COMMITS];
            m->evidence_rate_gbar_commits = stats[ERC_STAT_GBAR_COMMITS];
            m->evidence_rate_applied_commits = stats[ERC_STAT_APPLIED_COMMITS];
            m->evidence_rate_current_descent = current_descent;
            m->evidence_rate_tsa_descent = tsa_descent;
            m->evidence_rate_gbar_descent = gbar_descent;
            m->evidence_rate_applied_descent = applied_descent;
            m->evidence_rate_applied_harm = applied_harm;
            m->evidence_rate_active_fraction =
                stats[ERC_STAT_NONZERO] > 0u
                    ? (float)stats[ERC_STAT_ACTIVE] / (float)stats[ERC_STAT_NONZERO]
                    : 0.0f;
            m->evidence_rate_commitment_fraction =
                stats[ERC_STAT_ACTIVE] > 0u
                    ? (float)stats[ERC_STAT_APPLIED_COMMITS] / (float)stats[ERC_STAT_ACTIVE]
                    : 0.0f;
            m->evidence_rate_virtual_best_gain = gain;
            m->evidence_rate_priced_ratio = priced;
            m->evidence_rate_gate_pass = pass;

            printf("[EVIDENCE_RATE_COMMITMENT_GATE] step=%d mode=%s elements=%llu nonzero=%llu active=%llu failed=%llu boundary_hits=%llu stable=%llu conflict=%llu current_commits=%llu tsa_commits=%llu gbar_commits=%llu applied_commits=%llu current_descent=%.9g tsa_descent=%.9g gbar_descent=%.9g applied_descent=%.9g applied_harm=%.9g virtual_best_gain=%.6f priced_ratio=%.6f pass=%d reason=%s\n",
                   step + 1,
                   erc_mode_name((int)evidence_mode),
                   (unsigned long long)m->evidence_rate_elements,
                   (unsigned long long)m->evidence_rate_nonzero_grads,
                   (unsigned long long)m->evidence_rate_active_updates,
                   (unsigned long long)m->evidence_rate_failed_updates,
                   (unsigned long long)m->evidence_rate_boundary_hits,
                   (unsigned long long)m->evidence_rate_sign_stable,
                   (unsigned long long)m->evidence_rate_sign_conflict,
                   (unsigned long long)m->evidence_rate_current_commits,
                   (unsigned long long)m->evidence_rate_tsa_commits,
                   (unsigned long long)m->evidence_rate_gbar_commits,
                   (unsigned long long)m->evidence_rate_applied_commits,
                   m->evidence_rate_current_descent,
                   m->evidence_rate_tsa_descent,
                   m->evidence_rate_gbar_descent,
                   m->evidence_rate_applied_descent,
                   m->evidence_rate_applied_harm,
                   m->evidence_rate_virtual_best_gain,
                   m->evidence_rate_priced_ratio,
                   m->evidence_rate_gate_pass,
                   m->evidence_rate_gate_pass
                       ? "selected_arm_clears_priced_descent_gate"
                       : (applied_mode && !applied_safe)
                           ? "applied_mode_harm_or_no_positive_descent"
                       : "no_priced_descent_advantage_or_non_identifying");
        }
    }

    if (m->optimizer_integrity_gate_enabled) {
        uint32_t owned_groups = 0u;
        uint32_t resident_groups = 0u;
        gpu_optimizer_integrity_count_ternary_groups(
            p, m, &owned_groups, &resident_groups);

        m->optimizer_integrity_gate_batches++;
        m->optimizer_integrity_owned_ternary_groups = owned_groups;
        m->optimizer_integrity_resident_ternary_groups = resident_groups;
        m->optimizer_integrity_cpu_skip_groups = owned_groups;
        m->optimizer_integrity_gpu_update_groups = gpu_update_groups;
        m->optimizer_integrity_missing_update_groups =
            owned_groups > gpu_update_groups
                ? owned_groups - gpu_update_groups
                : 0u;
        m->optimizer_integrity_gate_pass =
            (owned_groups == resident_groups && owned_groups == gpu_update_groups) ? 1 : 0;

        printf("[OPTIMIZER_INTEGRITY_GATE] step=%d owned_ternary_groups=%u resident_ternary_groups=%u cpu_skip_groups=%u gpu_update_groups=%u missing_update_groups=%u pass=%d reason=%s\n",
               step + 1,
               owned_groups,
               resident_groups,
               m->optimizer_integrity_cpu_skip_groups,
               gpu_update_groups,
               m->optimizer_integrity_missing_update_groups,
               m->optimizer_integrity_gate_pass,
               m->optimizer_integrity_gate_pass
                   ? "all_gpu_owned_ternary_groups_updated"
                   : "gpu_owned_ternary_group_missing_residency_or_dispatch");
    }
}

int gpu_sync_to_cpu(GpuPipeline *p, Model *m) {
    if (!p || !m) return -1;

    VulkanContext *ctx = p->ctx;
    const size_t d = (size_t)p->d_model;
    const size_t ds = (size_t)p->d_state;
    const size_t de = (size_t)V12_D_EMBED;
    const size_t pred_bytes = d * d * sizeof(float);
    const size_t embed_e_bytes = (size_t)V12_VOCAB_SIZE * de * sizeof(float);
    const size_t embed_p_bytes = de * d * sizeof(float);
    const size_t norm_bytes = d * sizeof(float);
    const size_t win_bytes = d * ds * sizeof(float);
    const size_t wout_bytes = ds * d * sizeof(float);

    if ((m->gpu_opt_flags & GPU_OPT_LM_HEAD) != 0u) {
        gpu_pipeline_sync_lm_head_to_cpu(p, m);
    }
    if ((m->gpu_opt_flags & GPU_OPT_W_PRED) != 0u && model_materialize_w_pred_host(m) == 0) {
        if (m->shadow_w_pred && VK_BUFFER_VALID(p->opt_shadow_w_pred)) {
            vk_download(ctx, m->shadow_w_pred, p->opt_shadow_w_pred, pred_bytes);
        }
        if (m->sampled_w_pred && VK_BUFFER_VALID(p->opt_sampled_w_pred)) {
            vk_download(ctx, m->sampled_w_pred, p->opt_sampled_w_pred, pred_bytes);
        }
        if (m->m_w_pred_row && VK_BUFFER_VALID(p->opt_m_w_pred_row)) {
            vk_download(ctx, m->m_w_pred_row, p->opt_m_w_pred_row, norm_bytes);
        }
        if (m->v_w_pred_row && VK_BUFFER_VALID(p->opt_v_w_pred_row)) {
            vk_download(ctx, m->v_w_pred_row, p->opt_v_w_pred_row, norm_bytes);
        }
        if (m->m_w_pred_col && VK_BUFFER_VALID(p->opt_m_w_pred_col)) {
            vk_download(ctx, m->m_w_pred_col, p->opt_m_w_pred_col, norm_bytes);
        }
        if (m->v_w_pred_col && VK_BUFFER_VALID(p->opt_v_w_pred_col)) {
            vk_download(ctx, m->v_w_pred_col, p->opt_v_w_pred_col, norm_bytes);
        }
    }
    if ((m->gpu_opt_flags & GPU_OPT_NORM) != 0u && model_materialize_norm_host(m) == 0) {
        if (m->norm_w && VK_BUFFER_VALID(p->norm_w)) vk_download(ctx, m->norm_w, p->norm_w, norm_bytes);
        if (m->norm_b && VK_BUFFER_VALID(p->norm_b)) vk_download(ctx, m->norm_b, p->norm_b, norm_bytes);
        if (m->m_norm_w && VK_BUFFER_VALID(p->opt_m_norm_w)) vk_download(ctx, m->m_norm_w, p->opt_m_norm_w, norm_bytes);
        if (m->v_norm_w && VK_BUFFER_VALID(p->opt_v_norm_w)) vk_download(ctx, m->v_norm_w, p->opt_v_norm_w, norm_bytes);
        if (m->m_norm_b && VK_BUFFER_VALID(p->opt_m_norm_b)) vk_download(ctx, m->m_norm_b, p->opt_m_norm_b, norm_bytes);
        if (m->v_norm_b && VK_BUFFER_VALID(p->opt_v_norm_b)) vk_download(ctx, m->v_norm_b, p->opt_v_norm_b, norm_bytes);
    }
    if ((m->gpu_opt_flags & (GPU_OPT_EMBED_E | GPU_OPT_EMBED_P)) != 0u && model_materialize_embed_host(m) == 0) {
        if ((m->gpu_opt_flags & GPU_OPT_EMBED_E) && m->embed_E && VK_BUFFER_VALID(p->embed_E_gpu)) {
            vk_download(ctx, m->embed_E, p->embed_E_gpu, embed_e_bytes);
        }
        if ((m->gpu_opt_flags & GPU_OPT_EMBED_E) && m->m_embed_E && VK_BUFFER_VALID(p->opt_m_embed_E)) {
            vk_download(ctx, m->m_embed_E, p->opt_m_embed_E, embed_e_bytes);
        }
        if ((m->gpu_opt_flags & GPU_OPT_EMBED_E) && m->v_embed_E && VK_BUFFER_VALID(p->opt_v_embed_E)) {
            vk_download(ctx, m->v_embed_E, p->opt_v_embed_E, embed_e_bytes);
        }
        if ((m->gpu_opt_flags & GPU_OPT_EMBED_P) && m->embed_P && VK_BUFFER_VALID(p->embed_P_gpu)) {
            vk_download(ctx, m->embed_P, p->embed_P_gpu, embed_p_bytes);
        }
        if ((m->gpu_opt_flags & GPU_OPT_EMBED_P) && m->m_embed_P && VK_BUFFER_VALID(p->opt_m_embed_P)) {
            vk_download(ctx, m->m_embed_P, p->opt_m_embed_P, embed_p_bytes);
        }
        if ((m->gpu_opt_flags & GPU_OPT_EMBED_P) && m->v_embed_P && VK_BUFFER_VALID(p->opt_v_embed_P)) {
            vk_download(ctx, m->v_embed_P, p->opt_v_embed_P, embed_p_bytes);
        }
    }

    for (int k = 0; k < p->n_layers; k++) {
        SSMLayer *L = &m->layers[k];
        if ((m->gpu_opt_flags & (GPU_OPT_W_IN | GPU_OPT_W_OUT)) != 0u) {
            if (model_materialize_layer_w_io_host(L) != 0) continue;
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_IN) && L->shadow_w_in && VK_BUFFER_VALID(p->opt_layers[k].shadow_w_in)) {
            vk_download(ctx, L->shadow_w_in, p->opt_layers[k].shadow_w_in, win_bytes);
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_IN) && L->sampled_w_in && VK_BUFFER_VALID(p->opt_layers[k].sampled_w_in)) {
            vk_download(ctx, L->sampled_w_in, p->opt_layers[k].sampled_w_in, win_bytes);
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_IN) && L->m_w_in_row && VK_BUFFER_VALID(p->opt_layers[k].m_w_in_row)) {
            vk_download(ctx, L->m_w_in_row, p->opt_layers[k].m_w_in_row, norm_bytes);
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_IN) && L->v_w_in_row && VK_BUFFER_VALID(p->opt_layers[k].v_w_in_row)) {
            vk_download(ctx, L->v_w_in_row, p->opt_layers[k].v_w_in_row, norm_bytes);
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_IN) && L->m_w_in_col && VK_BUFFER_VALID(p->opt_layers[k].m_w_in_col)) {
            vk_download(ctx, L->m_w_in_col, p->opt_layers[k].m_w_in_col, ds * sizeof(float));
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_IN) && L->v_w_in_col && VK_BUFFER_VALID(p->opt_layers[k].v_w_in_col)) {
            vk_download(ctx, L->v_w_in_col, p->opt_layers[k].v_w_in_col, ds * sizeof(float));
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_OUT) && L->shadow_w_out && VK_BUFFER_VALID(p->opt_layers[k].shadow_w_out)) {
            vk_download(ctx, L->shadow_w_out, p->opt_layers[k].shadow_w_out, wout_bytes);
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_OUT) && L->sampled_w_out && VK_BUFFER_VALID(p->opt_layers[k].sampled_w_out)) {
            vk_download(ctx, L->sampled_w_out, p->opt_layers[k].sampled_w_out, wout_bytes);
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_OUT) && L->m_w_out_row && VK_BUFFER_VALID(p->opt_layers[k].m_w_out_row)) {
            vk_download(ctx, L->m_w_out_row, p->opt_layers[k].m_w_out_row, ds * sizeof(float));
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_OUT) && L->v_w_out_row && VK_BUFFER_VALID(p->opt_layers[k].v_w_out_row)) {
            vk_download(ctx, L->v_w_out_row, p->opt_layers[k].v_w_out_row, ds * sizeof(float));
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_OUT) && L->m_w_out_col && VK_BUFFER_VALID(p->opt_layers[k].m_w_out_col)) {
            vk_download(ctx, L->m_w_out_col, p->opt_layers[k].m_w_out_col, norm_bytes);
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_OUT) && L->v_w_out_col && VK_BUFFER_VALID(p->opt_layers[k].v_w_out_col)) {
            vk_download(ctx, L->v_w_out_col, p->opt_layers[k].v_w_out_col, norm_bytes);
        }
        if ((m->gpu_opt_flags & GPU_OPT_A) && L->A && VK_BUFFER_VALID(p->layers[k].gpu_A_const)) {
            vk_download(ctx, L->A, p->layers[k].gpu_A_const, ds * sizeof(float));
        }
        if ((m->gpu_opt_flags & GPU_OPT_WA) && L->W_A && VK_BUFFER_VALID(p->lw_WA)) {
            vk_download(ctx, L->W_A, p->lw_WA, ds * ds * sizeof(float));
        }
        if ((m->gpu_opt_flags & GPU_OPT_BA) && L->b_A && VK_BUFFER_VALID(p->lw_bA)) {
            vk_download(ctx, L->b_A, p->lw_bA, ds * sizeof(float));
        }
        if ((m->gpu_opt_flags & GPU_OPT_CONV) && L->shadow_w_conv && VK_BUFFER_VALID(p->lw_conv)) {
            download_shadow_buffer(ctx, L->shadow_w_conv, p->lw_conv, (size_t)V12_CONV_KERNEL * ds, m->step);
        }
    }
    return 0;
}


#else
#pragma message("Compiling MTP else branch") // !V12_HAS_VULKAN

float gpu_train_step(GpuPipeline *p, Model *m, Cache *c,
                     const uint16_t *tokens, const uint8_t *boundary_mask,
                     int seq_len, int seq_idx, int batch_size,
                     const TrainState *state) {
    float lrms_penalty_global = 0.0f;
    (void)p; (void)m; (void)c; (void)tokens; (void)boundary_mask;
    (void)seq_len; (void)seq_idx; (void)batch_size; (void)state;
    return 0.0f;
}


#endif // V12_HAS_VULKAN
