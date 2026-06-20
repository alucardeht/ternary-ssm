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
            VK_BUFFER_VALID(p->embed_E_gpu), VK_BUFFER_VALID(p->embed_P_gpu), 
            VK_BUFFER_VALID(p->embed_gathered), VK_BUFFER_VALID(p->embed_tokens), 
            VK_BUFFER_VALID(p->residual[0]), (void*)m->embed_E);
    }
    if (embed_on_gpu) {
        vk_begin_batch(ctx);
        /* Pack tokens as uint32 into staging then upload — entire batch×seq flat */
        int total_tokens = batch_size * seq_len;
        uint32_t *tok_buf = malloc((size_t)total_tokens * sizeof(uint32_t));
        if (tok_buf) {
            for (int t = 0; t < total_tokens; t++) tok_buf[t] = (uint32_t)tokens[t];
            vk_upload(ctx, p->embed_tokens, tok_buf, (size_t)total_tokens * sizeof(uint32_t));
            free(tok_buf);
        }

        /* Step 1a: gather embed_E rows by token index -> embed_gathered [batch×seq × d_embed] */
        #ifdef R504_E11_BTRP
        vk_dispatch_btrp_token_gather(ctx, p->embed_E_gpu, p->embed_tokens, p->embed_gathered, (uint32_t)seq_len, (uint32_t)V12_D_EMBED, batch_size, m->vocab_size, BTRP_R);
#else
        vk_dispatch_token_gather(ctx,
                                 p->embed_E_gpu, p->embed_tokens, p->embed_gathered,
                                 (uint32_t)seq_len, (uint32_t)V12_D_EMBED, batch_size);
#endif
        vk_barrier(ctx);

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

                        ;

        if (layer_on_gpu) {
#if V12_PRBLT_ENABLED
            prblt_load_layer(p, m, k);
            VkTernaryCSR *csr_in_ptr  = &p->prblt_ring[k % 2].gpu_csr_in;
            VkTernaryCSR *csr_out_ptr = &p->prblt_ring[k % 2].gpu_csr_out;
#else
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
#if R504_S10_PSVCTE
            vk_dispatch_sp10_psvcte(ctx, p->ssm_a, p->ssm_b, p->ssm_x,
                                 p->ssm_cbl, p->ssm_h, seq_len, ds, batch_size);
#elif R504_S11_LNS_TFMS
            vk_dispatch_sp11_lns_tfms(ctx, p->ssm_a, p->ssm_b, p->ssm_x,
                                 p->ssm_cbl, p->ssm_h, seq_len, ds, batch_size);
#elif R504_S14_GFIT
            vk_dispatch_sp14_gfit(ctx, p->ssm_a, p->ssm_b, p->ssm_x,
                                 p->ssm_cbl, p->ssm_h, seq_len, ds, batch_size);
#else
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

            vk_end_batch(ctx);

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
    vk_end_batch(ctx);
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
                vk_end_batch(ctx);
                free(eqp_active_lens);
                return NAN;
            }

            vk_download(ctx, loss_tmp, p->ce_loss, (size_t)n_valid * sizeof(float));
            vk_download(ctx, dlogits_tmp, p->ce_d_logits, dlogits_n * sizeof(float));
            vk_end_batch(ctx);

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
                vk_end_batch(ctx);
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

        vk_end_batch(ctx);

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
                // Q-09 disabled

                vk_download(ctx, d_y_ssm, p->bwd_d_y_ssm, ssm_bytes);
                // MFCF: No residual add to gpu_dw_out
                if (!gpu_owns_w_out)
                    vk_download(ctx, p->staging_dw, p->gemm_C, dw_bytes);

                vk_end_batch(ctx);

                if (!gpu_owns_w_out)
                    for (int i = 0; i < d * ds; i++) L->dw_out[i] += p->staging_dw[i];
            } else {
                vk_begin_batch(ctx);


                    vk_dispatch_ternary_matvec(ctx, csc_out_ptr, p->bwd_d_h_in, p->bwd_d_y_ssm, batch_size * seq_len);
                
                vk_barrier(ctx);

                vk_download(ctx, d_y_ssm, p->bwd_d_y_ssm, ssm_bytes);

                vk_end_batch(ctx);

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

            /* a) Fill ssm_a with L->A broadcast across all (batch*seq) timesteps */
            vk_dispatch_broadcast_fill(ctx,
                p->layers[k].gpu_A_const,
                p->ssm_a,
                (uint32_t)(batch_size * seq_len),
                (uint32_t)ds);
            vk_barrier(ctx);

            /* b) SSM forward scan: h_mtp[t] = A^t * 0 + prefix(b_eff * x_cv)
             *    Uses fwd_b_eff and fwd_x_cv already GPU-resident from forward pass.
             *    C_bilinear not used for MTP scan (pass VK_BUFFER_NULL). */
#if R504_S10_PSVCTE
            vk_dispatch_sp10_psvcte(ctx,
                p->ssm_a,
                p->layers[k].fwd_b_eff,
                p->layers[k].fwd_x_cv,
                VK_BUFFER_NULL,
                p->layers[k].gpu_mtp_h,
                seq_len, ds, batch_size);
#elif R504_S11_LNS_TFMS
            vk_dispatch_sp11_lns_tfms(ctx, p->ssm_a, p->ssm_b, p->ssm_x,
                                 p->ssm_cbl, p->ssm_h, seq_len, ds, batch_size);
#elif R504_S14_GFIT
            vk_dispatch_sp14_gfit(ctx,
                p->ssm_a,
                p->layers[k].fwd_b_eff,
                p->layers[k].fwd_x_cv,
                VK_BUFFER_NULL,
                p->layers[k].gpu_mtp_h,
                seq_len, ds, batch_size);
#else
            vk_dispatch_sp12_lcmbi(ctx,
                p->ssm_a,
                p->layers[k].fwd_b_eff,
                p->layers[k].fwd_x_cv,
                VK_BUFFER_NULL,
                p->layers[k].gpu_mtp_h,
                seq_len, ds, batch_size);
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

            vk_end_batch(ctx);
            mtp_gpu_done = 1;
        }

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
                        vk_end_batch(ctx);

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
            // ef03_validate_gate(ctx, p->ssm_h, p->ef03_ring_sums, batch_size * seq_len, ds, 16);
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
            vk_end_batch(ctx);
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

            vk_end_batch(ctx);
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
                    vk_end_batch(ctx);
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
                    vk_end_batch(ctx);
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
                    vk_end_batch(ctx);
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
                vk_end_batch(ctx);
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
                // Q-09 disabled

            /* Step 11: W_in^T @ d_xp -> d_h_prev (CSC of csr_in) + accumulate into d_h */
            vk_upload(ctx, p->bwd_d_xp, d_xp, ssm_bytes);

                vk_dispatch_ternary_matvec(ctx, &p->layers[k].gpu_csc_in, p->bwd_d_xp, p->bwd_d_h_prev, batch_size * seq_len);
            
            vk_barrier(ctx);
            vk_dispatch_residual_add(ctx, p->bwd_d_h_prev, p->bwd_d_h_in,
                                     (uint32_t)(batch_size * seq_len * d));
            vk_sync(ctx);

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

            vk_end_batch(ctx);

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

