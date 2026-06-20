GpuPipeline *gpu_pipeline_create(VulkanContext *ctx, Model *m, int seq_len, int batch_size) {
    GpuPipeline *p = calloc(1, sizeof(GpuPipeline));
    if (!p) return NULL;

    p->ctx = ctx;
    p->seq_len = seq_len;
    p->batch_size = batch_size;
    p->d_model = V12_D_MODEL;
    p->d_state = V12_D_STATE;
    p->n_layers = V12_N_LAYERS;

    int d = V12_D_MODEL;
    int ds = V12_D_STATE;
    size_t res_bytes = (size_t)batch_size * seq_len * d * sizeof(float);
    size_t ssm_bytes = (size_t)batch_size * seq_len * ds * sizeof(float);
    size_t pred_bytes = (size_t)d * d * sizeof(float);
    size_t pred_xb = (size_t)batch_size * (seq_len - 1) * d * sizeof(float);
    size_t gemm_bytes = (size_t)batch_size * seq_len * d * sizeof(float);
    size_t gemm_c_bytes = (size_t)d * d * sizeof(float);

    /* Embedding GPU buffers */
    size_t embed_E_bytes   = (size_t)V12_VOCAB_SIZE * V12_D_EMBED * sizeof(float);
    size_t embed_P_bytes   = (size_t)V12_D_EMBED    * d           * sizeof(float);
    size_t embed_gath_bytes = (size_t)batch_size * seq_len * V12_D_EMBED * sizeof(float);
    size_t embed_tok_bytes  = (size_t)batch_size * seq_len              * sizeof(uint32_t);

    p->embed_E_gpu    = vk_alloc(ctx, embed_E_bytes,    VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->embed_P_gpu    = vk_alloc(ctx, embed_P_bytes,    VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->embed_gathered = vk_alloc(ctx, embed_gath_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->embed_tokens   = vk_alloc(ctx, embed_tok_bytes,  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);

    vk_upload(ctx, p->embed_E_gpu, m->embed_E, embed_E_bytes);
    vk_upload(ctx, p->embed_P_gpu, m->embed_P, embed_P_bytes);

    /* Shared scratch */
    p->buf_x = vk_alloc(ctx, res_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
    p->buf_y = vk_alloc(ctx, res_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);

    /* Residual stream */
    for (int k = 0; k <= V12_N_LAYERS; k++) {
        p->residual[k] = vk_alloc(ctx, res_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
    }

    /* Predictive coding */
    p->pred_A        = vk_alloc(ctx, pred_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->shadow_pred_A = vk_alloc(ctx, pred_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->pred_x        = vk_alloc(ctx, pred_xb, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->pred_y        = vk_alloc(ctx, pred_xb, VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC | VK_BUF_TRANSFER_DST);

    /* Initial weight uploads */
    vk_upload(ctx, p->pred_A,        m->sampled_w_pred, pred_bytes);
    upload_shadow_buffer(ctx, p->shadow_pred_A, m->shadow_w_pred,  (size_t)d * d);

    /* GEMM scratch */
    p->gemm_A = vk_alloc(ctx, gemm_bytes,   VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->gemm_B = vk_alloc(ctx, gemm_bytes,   VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->gemm_C = vk_alloc(ctx, gemm_c_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);

    p->q09_k = 32;
    p->q09_P = vk_alloc(ctx, (size_t)batch_size * seq_len * p->d_model * sizeof(float), VK_BUF_STORAGE);
    p->q09_P_left = vk_alloc(ctx, (size_t)p->d_model * p->d_model * sizeof(float), VK_BUF_STORAGE);
    p->q09_P_right = vk_alloc(ctx, (size_t)p->d_model * p->d_model * sizeof(float), VK_BUF_STORAGE);
    p->q09_stats = vk_alloc(ctx, 3 * sizeof(int), VK_BUF_STORAGE | VK_BUF_HOST_VISIBLE | VK_BUF_TRANSFER_DST);
    p->q09_dw_approx = vk_alloc(ctx, (size_t)p->d_model * p->d_model * sizeof(float), VK_BUF_STORAGE);

    p->q10_w_v = vk_alloc(ctx, (size_t)p->d_state * sizeof(float), VK_BUF_STORAGE);
    p->q10_dw_v = vk_alloc(ctx, (size_t)p->d_state * sizeof(float), VK_BUF_STORAGE);
    p->q10_v_head = vk_alloc(ctx, (size_t)batch_size * seq_len * sizeof(float), VK_BUF_STORAGE);
    p->q10_w_v_m = vk_alloc(ctx, (size_t)p->d_state * sizeof(float), VK_BUF_STORAGE);
    p->q10_w_v_v = vk_alloc(ctx, (size_t)p->d_state * sizeof(float), VK_BUF_STORAGE);
    vk_fill_zero(ctx, p->q10_w_v, (size_t)p->d_state * sizeof(float));
    vk_fill_zero(ctx, p->q10_w_v_m, (size_t)p->d_state * sizeof(float));
    vk_fill_zero(ctx, p->q10_w_v_v, (size_t)p->d_state * sizeof(float));

    /* SSM transients */
    p->ssm_a   = vk_alloc(ctx, ssm_bytes,          VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
    p->ssm_b   = vk_alloc(ctx, ssm_bytes,          VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->ssm_x   = vk_alloc(ctx, ssm_bytes,          VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->ssm_h   = vk_alloc(ctx, ssm_bytes,          VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
    p->ssm_cbl = vk_alloc(ctx, ds * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->ssm_u   = vk_alloc(ctx, ds * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->ssm_v   = vk_alloc(ctx, ds * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->ef03_ring_sums = vk_alloc(ctx, (size_t)(batch_size * seq_len / 16 + 1) * ds * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);

    /* Layer weights */
    p->lw_conv = vk_alloc(ctx, V12_CONV_KERNEL * ds * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->lw_B    = vk_alloc(ctx, ds * sizeof(float),                   VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->lw_A    = vk_alloc(ctx, ds * sizeof(float),                   VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->lw_WA   = vk_alloc(ctx, ds * ds * sizeof(float),              VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->lw_bA   = vk_alloc(ctx, ds * sizeof(float),                   VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->lw_C    = vk_alloc(ctx, ds * ds * sizeof(float),              VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);

    /* VSSBR: shared forward activation buffers allocated once and reused across all layers */
    VkBufferHandle shared_fwd_x_proj   = vk_alloc(ctx, ssm_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
    VkBufferHandle shared_fwd_x_gated  = vk_alloc(ctx, ssm_bytes, VK_BUF_STORAGE);
    VkBufferHandle shared_fwd_mask_val = vk_alloc(ctx, ssm_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
    VkBufferHandle shared_fwd_x_cv     = vk_alloc(ctx, ssm_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
    VkBufferHandle shared_fwd_b_eff    = vk_alloc(ctx, ssm_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
    VkBufferHandle shared_fwd_a_buf    = vk_alloc(ctx, ssm_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
    VkBufferHandle shared_fwd_h_cache  = vk_alloc(ctx, ssm_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
#ifdef R504_E10_PSCQ
    VkBufferHandle shared_fwd_h_cache_compressed = vk_alloc(ctx, ssm_bytes / 4, VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC | VK_BUF_TRANSFER_DST);
#endif
    VkBufferHandle shared_qrf3ac_basis = vk_alloc(ctx, ssm_bytes, VK_BUF_STORAGE);
    VkBufferHandle shared_qrf3ac_ternary = vk_alloc(ctx, ssm_bytes, VK_BUF_STORAGE);
    VkBufferHandle shared_htr1p_u = vk_alloc(ctx, ssm_bytes, VK_BUF_STORAGE);
    VkBufferHandle shared_htr1p_v = vk_alloc(ctx, ssm_bytes, VK_BUF_STORAGE);
    VkBufferHandle shared_fwd_y_ssm    = vk_alloc(ctx, ssm_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);

    /* Per-layer: CSR + CSC + forward activation buffers */
    for (int k = 0; k < V12_N_LAYERS; k++) {
        SSMLayer *L = &m->layers[k];

        /* CSR */
#if !V12_PRBLT_ENABLED
        upload_csr_to_gpu(ctx, &p->layers[k].gpu_csr_in,  &L->csr_in);
        upload_csr_to_gpu(ctx, &p->layers[k].gpu_csr_out, &L->csr_out);
#endif
        p->layers[k].uploaded = 1;

        /* CSC (transposed, for backward matvec) */
#if !V12_PRBLT_ENABLED
        TernaryCSC csc_in, csc_out;
        if (ternary_csc_from_csr(&csc_in, &L->csr_in) == 0) {
            upload_csc_to_gpu(ctx, &p->layers[k].gpu_csc_in, &csc_in);
            ternary_csc_free(&csc_in);
        }
        if (ternary_csc_from_csr(&csc_out, &L->csr_out) == 0) {
            upload_csc_to_gpu(ctx, &p->layers[k].gpu_csc_out, &csc_out);
            ternary_csc_free(&csc_out);
        }
#endif
        p->layers[k].csc_uploaded = 1;

        /* Forward activation buffers (VSSBR: shared reference) */
        p->layers[k].fwd_x_proj      = shared_fwd_x_proj;
        p->layers[k].fwd_x_gated     = shared_fwd_x_gated;
        p->layers[k].fwd_mask_val    = shared_fwd_mask_val;
        p->layers[k].fwd_x_cv        = shared_fwd_x_cv;
        p->layers[k].fwd_b_eff       = shared_fwd_b_eff;
        p->layers[k].fwd_a_buf       = shared_fwd_a_buf;
        p->layers[k].fwd_h_cache     = shared_fwd_h_cache;

        p->layers[k].f34_P_r = vk_alloc(ctx, ds * 64 * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->layers[k].f34_z_cache = vk_alloc(ctx, batch_size * seq_len * 64 * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);

#ifdef R504_E10_PSCQ
        p->layers[k].fwd_h_cache_compressed = shared_fwd_h_cache_compressed;
#endif
        p->layers[k].qrf3ac_basis    = shared_qrf3ac_basis;
        p->layers[k].qrf3ac_ternary  = shared_qrf3ac_ternary;
        p->layers[k].htr1p_u         = shared_htr1p_u;
        p->layers[k].htr1p_v         = shared_htr1p_v;
        p->layers[k].fwd_y_ssm       = shared_fwd_y_ssm;
        p->layers[k].gpu_moments     = vk_alloc(ctx, GPU_MOMENTS_BYTES(ds),
                                                VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
        p->layers[k].tssi_score_moments =
            vk_alloc(ctx, TSSI_SCORE_MOMENTS_BYTES(ds),
                     VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
        if (VK_BUFFER_VALID(p->layers[k].tssi_score_moments)) {
            vk_fill_zero(ctx, p->layers[k].tssi_score_moments,
                         TSSI_SCORE_MOMENTS_BYTES(ds));
        }
        {
            float *tmp = calloc(GPU_MOMENTS_FLOATS(ds), sizeof(float));
            if (tmp) {
                moments_to_gpu_flat(&L->moments, tmp, ds);
                vk_upload(ctx, p->layers[k].gpu_moments, tmp, GPU_MOMENTS_BYTES(ds));
                free(tmp);
            }
        }
        p->layers[k].fwd_allocated   = 1;

        /* MTP buffers — only for layers that have MTP weights (even layers) */
        p->layers[k].gpu_mtp_U    = VK_BUFFER_NULL;
        p->layers[k].gpu_mtp_V    = VK_BUFFER_NULL;
        p->layers[k].gpu_A_const  = VK_BUFFER_NULL;
        p->layers[k].gpu_d_mtp_U  = VK_BUFFER_NULL;
        p->layers[k].gpu_d_mtp_V  = VK_BUFFER_NULL;
        p->layers[k].gpu_mtp_h    = VK_BUFFER_NULL;
        p->layers[k].gpu_mtp_loss = VK_BUFFER_NULL;
        p->layers[k].mtp_allocated = 0;

        if (L->mtp_U != NULL) {
            size_t mtp_bytes = (size_t)ds * V12_MTP_RANK * sizeof(float);
            size_t mtp_uint_bytes = (size_t)ds * V12_MTP_RANK * sizeof(uint32_t);

            p->layers[k].gpu_mtp_U    = vk_alloc(ctx, mtp_bytes,
                                                  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
            p->layers[k].gpu_mtp_V    = vk_alloc(ctx, mtp_bytes,
                                                  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
            p->layers[k].gpu_A_const  = vk_alloc(ctx, (size_t)ds * sizeof(float),
                                                  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
            p->layers[k].gpu_d_mtp_U  = vk_alloc(ctx, mtp_uint_bytes,
                                                  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
            p->layers[k].gpu_d_mtp_V  = vk_alloc(ctx, mtp_uint_bytes,
                                                  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
            p->layers[k].gpu_mtp_h    = vk_alloc(ctx, ssm_bytes,
                                                  VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
            p->layers[k].gpu_mtp_loss = vk_alloc(ctx, sizeof(uint32_t),
                                                  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);

            vk_upload(ctx, p->layers[k].gpu_mtp_U,   L->mtp_U,   mtp_bytes);
            vk_upload(ctx, p->layers[k].gpu_mtp_V,   L->mtp_V,   mtp_bytes);
            vk_upload(ctx, p->layers[k].gpu_A_const,  L->A,  (size_t)ds * sizeof(float));

            p->layers[k].mtp_allocated = 1;
        }
    }

    /* Backward scratch */
    p->bwd_d_y_ssm  = vk_alloc(ctx, ssm_bytes,                          VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
    p->bwd_d_xp     = vk_alloc(ctx, ssm_bytes,                          VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
    p->bwd_d_x_cv   = vk_alloc(ctx, ssm_bytes,                          VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
    p->bwd_dA_ps    = vk_alloc(ctx, ssm_bytes,                          VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
    p->bwd_dB_seq   = vk_alloc(ctx, ssm_bytes,                          VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
    p->bwd_v_pre    = vk_alloc(ctx, ssm_bytes,                          VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
    p->bwd_d_h_in   = vk_alloc(ctx, res_bytes,                          VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
    p->bwd_h_final  = vk_alloc(ctx, (size_t)batch_size * ds * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
    p->bwd_d_h_prev = vk_alloc(ctx, res_bytes,                          VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
    p->bwd_d_embed_E   = vk_alloc(ctx, (size_t)V12_VOCAB_SIZE * V12_D_EMBED * sizeof(float),
                                  VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC | VK_BUF_TRANSFER_DST);
    p->bwd_d_conv_w    = vk_alloc(ctx, V12_CONV_KERNEL * ds * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
    p->bwd_d_tau       = vk_alloc(ctx, sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC | VK_BUF_TRANSFER_DST);
    p->bwd_d_dot       = vk_alloc(ctx, ssm_bytes,                            VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
    p->bwd_dA_reduce   = vk_alloc(ctx, (size_t)ds * sizeof(float),           VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
    p->bwd_dbA_reduce  = vk_alloc(ctx, (size_t)ds * sizeof(float),           VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
    p->bwd_dC_bl_accum = vk_alloc(ctx, (size_t)ds * sizeof(float),           VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC | VK_BUF_TRANSFER_DST);
    p->bwd_dB_accum    = vk_alloc(ctx, (size_t)ds * sizeof(uint32_t),        VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC | VK_BUF_TRANSFER_DST);
    p->bwd_B_vec       = vk_alloc(ctx, (size_t)ds * sizeof(float),           VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);

#ifdef R503_PHASE_1_EFFICIENCY
    /* R503 E-07 Phase-Area Aliasing 
     * In a full implementation, we alias the memory here. We will set a flag.
     * p->bwd_d_y_ssm = p->ssm_h; // Alias state and adjoint
     */
    
    /* R503 E-10 Token-Driven Sparse Embedding Allocation 
     * We allocate a smaller sparse buffer.
     */
    p->bwd_d_embed_E_sparse = vk_alloc(ctx, (size_t)batch_size * 1024 * V12_D_EMBED * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
    
    /* R503 E-13 Forward-Adjoint Trajectory Interleaving (FATI)
     * Setup FATI chunk sizes 
     */
    p->fati_chunk_size = 128;
#endif
    /* RMS norm weights + outputs */
    p->norm_w   = vk_alloc(ctx, (size_t)d * sizeof(float),                         VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->norm_b   = vk_alloc(ctx, (size_t)d * sizeof(float),                         VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->norm_out  = vk_alloc(ctx, res_bytes,                                          VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
    p->norm_rms  = vk_alloc(ctx, (size_t)batch_size * seq_len * sizeof(float),      VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
    vk_upload(ctx, p->norm_w, m->norm_w, (size_t)d * sizeof(float));
    vk_upload(ctx, p->norm_b, m->norm_b, (size_t)d * sizeof(float));

    /* Optimizer state for norm_w and norm_b (plain Adam, zero-initialized) */
    p->opt_m_norm_w = vk_alloc(ctx, (size_t)d * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->opt_v_norm_w = vk_alloc(ctx, (size_t)d * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->opt_m_norm_b = vk_alloc(ctx, (size_t)d * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->opt_v_norm_b = vk_alloc(ctx, (size_t)d * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->opt_d_norm_w = vk_alloc(ctx, (size_t)d * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    p->opt_d_norm_b = vk_alloc(ctx, (size_t)d * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    vk_fill_zero(ctx, p->opt_m_norm_w, (size_t)d * sizeof(float));
    vk_fill_zero(ctx, p->opt_v_norm_w, (size_t)d * sizeof(float));
    vk_fill_zero(ctx, p->opt_m_norm_b, (size_t)d * sizeof(float));
    vk_fill_zero(ctx, p->opt_v_norm_b, (size_t)d * sizeof(float));
    vk_fill_zero(ctx, p->opt_d_norm_w, (size_t)d * sizeof(float));
    vk_fill_zero(ctx, p->opt_d_norm_b, (size_t)d * sizeof(float));

    /* Optimizer state for embed_P [d_embed × d_model] — plain Adam */
    {
        size_t demb_p_bytes = (size_t)V12_D_EMBED * d * sizeof(float);
        p->opt_m_embed_P = vk_alloc(ctx, demb_p_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_v_embed_P = vk_alloc(ctx, demb_p_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_d_embed_P = vk_alloc(ctx, demb_p_bytes,
                                    VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
        vk_fill_zero(ctx, p->opt_m_embed_P, demb_p_bytes);
        vk_fill_zero(ctx, p->opt_v_embed_P, demb_p_bytes);
        vk_fill_zero(ctx, p->opt_d_embed_P, demb_p_bytes);
    }

    /* Optimizer state for embed_E [vocab_size × d_embed] — dense Adam */
    {
        size_t demb_e_bytes = (size_t)V12_VOCAB_SIZE * V12_D_EMBED * sizeof(float);
        p->opt_m_embed_E = vk_alloc(ctx, demb_e_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_v_embed_E = vk_alloc(ctx, demb_e_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        vk_fill_zero(ctx, p->opt_m_embed_E, demb_e_bytes);
        vk_fill_zero(ctx, p->opt_v_embed_E, demb_e_bytes);
    }

    /* Per-layer dense Adam state for A [ds], b_A [ds], W_A [ds×ds], conv_w [K×ds],
     * and saved gradient buffers for dw_out, dw_in, dW_A */
    for (int k = 0; k < V12_N_LAYERS; k++) {
        size_t ds_bytes   = (size_t)ds * sizeof(float);
        size_t dw_bytes   = (size_t)ds * d  * sizeof(float);   /* d_state × d_model */
        size_t dwa_bytes  = (size_t)ds * ds * sizeof(float);   /* d_state × d_state */
        size_t conv_bytes = (size_t)V12_CONV_KERNEL * ds * sizeof(float);

        p->opt_layers[k].m_A    = vk_alloc(ctx, ds_bytes,   VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].v_A    = vk_alloc(ctx, ds_bytes,   VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].m_b_A  = vk_alloc(ctx, ds_bytes,   VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].v_b_A  = vk_alloc(ctx, ds_bytes,   VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].m_conv = vk_alloc(ctx, conv_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].v_conv = vk_alloc(ctx, conv_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        
        p->opt_layers[k].mprm_Sigma_M = vk_alloc(ctx, 16 * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].mprm_Sigma_V = vk_alloc(ctx, 16 * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        vk_fill_zero(ctx, p->opt_layers[k].mprm_Sigma_M, 16 * sizeof(float));
        vk_fill_zero(ctx, p->opt_layers[k].mprm_Sigma_V, 16 * sizeof(float));

        p->opt_layers[k].m_WA   = vk_alloc(ctx, dwa_bytes,  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].v_WA   = vk_alloc(ctx, dwa_bytes,  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);

        p->opt_layers[k].m_B    = vk_alloc(ctx, ds_bytes,   VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].v_B    = vk_alloc(ctx, ds_bytes,   VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].dB     = vk_alloc(ctx, ds_bytes,   VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);

        p->opt_layers[k].m_C    = vk_alloc(ctx, dwa_bytes,  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].v_C    = vk_alloc(ctx, dwa_bytes,  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].dC     = vk_alloc(ctx, dwa_bytes,  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);

        /* Saved gradient buffers — written during backward by vk_copy_buffer,
         * read by optimizer (dw_in/dw_out reserved for factored_adam_ternary). */
        p->opt_layers[k].gpu_dw_out = vk_alloc(ctx, 4, // MFCF drops dense buffer to 0 bytes

                                                VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
        p->opt_layers[k].gpu_dw_in  = vk_alloc(ctx, 4, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
        p->opt_layers[k].holo_U_in = vk_alloc(ctx, ds * 16 * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
        p->opt_layers[k].holo_V_in = vk_alloc(ctx, 16 * d * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
        p->opt_layers[k].gpu_dW_A   = vk_alloc(ctx, dwa_bytes,
                                                VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].gpu_dA     = vk_alloc(ctx, ds_bytes,
                                                VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].gpu_dbA    = vk_alloc(ctx, ds_bytes,
                                                VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].gpu_dconv  = vk_alloc(ctx, conv_bytes,
                                                VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);

        vk_fill_zero(ctx, p->opt_layers[k].m_A,      ds_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].v_A,      ds_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].m_b_A,    ds_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].v_b_A,    ds_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].m_conv,   conv_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].v_conv,   conv_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].m_WA,     dwa_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].v_WA,     dwa_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].gpu_dA,   ds_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].gpu_dbA,  ds_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].gpu_dconv, conv_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].m_B,      ds_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].v_B,      ds_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].dB,       ds_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].m_C,      dwa_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].v_C,      dwa_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].dC,       dwa_bytes);

        /* Ternary W_in [d_model × d_state]: shadow logits + factored Adam moments */
        size_t win_bytes  = (size_t)d  * ds * sizeof(float);   /* d_model × d_state */
        size_t wout_bytes = (size_t)ds * d  * sizeof(float);   /* d_state × d_model */
        size_t mrow_in_bytes  = (size_t)d  * sizeof(float);    /* d_model */
        size_t mcol_in_bytes  = (size_t)ds * sizeof(float);    /* d_state */
        size_t mrow_out_bytes = (size_t)ds * sizeof(float);    /* d_state */
        size_t mcol_out_bytes = (size_t)d  * sizeof(float);    /* d_model */

        p->opt_layers[k].shadow_w_in = vk_alloc(ctx, win_bytes,
                                                 VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
#if V12_ZMMA_ENABLED
        p->opt_layers[k].m_w_in_row  = VK_BUFFER_NULL;
        p->opt_layers[k].v_w_in_row  = VK_BUFFER_NULL;
        p->opt_layers[k].m_w_in_col  = VK_BUFFER_NULL;
        p->opt_layers[k].v_w_in_col  = VK_BUFFER_NULL;
#else
        p->opt_layers[k].m_w_in_row  = vk_alloc(ctx, mrow_in_bytes,
                                                 VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].v_w_in_row  = vk_alloc(ctx, mrow_in_bytes,
                                                 VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].m_w_in_col  = vk_alloc(ctx, mcol_in_bytes,
                                                 VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].v_w_in_col  = vk_alloc(ctx, mcol_in_bytes,
                                                 VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
#endif

        /* Ternary W_out [d_state × d_model]: shadow logits + factored Adam moments */
        p->opt_layers[k].shadow_w_out = vk_alloc(ctx, wout_bytes,
                                                  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
#if V12_ZMMA_ENABLED
        p->opt_layers[k].m_w_out_row  = VK_BUFFER_NULL;
        p->opt_layers[k].v_w_out_row  = VK_BUFFER_NULL;
        p->opt_layers[k].m_w_out_col  = VK_BUFFER_NULL;
        p->opt_layers[k].v_w_out_col  = VK_BUFFER_NULL;
#else
        p->opt_layers[k].m_w_out_row  = vk_alloc(ctx, mrow_out_bytes,
                                                  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].v_w_out_row  = vk_alloc(ctx, mrow_out_bytes,
                                                  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].m_w_out_col  = vk_alloc(ctx, mcol_out_bytes,
                                                  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].v_w_out_col  = vk_alloc(ctx, mcol_out_bytes,
                                                  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
#endif

        /* Upload shadow logits from CPU model (non-NULL in training mode) */
        SSMLayer *Ltern = &m->layers[k];
        if (Ltern->shadow_w_in)
            upload_shadow_buffer(ctx, p->opt_layers[k].shadow_w_in,  Ltern->shadow_w_in,  (size_t)d * ds);
        if (Ltern->shadow_w_out)
            upload_shadow_buffer(ctx, p->opt_layers[k].shadow_w_out, Ltern->shadow_w_out, (size_t)ds * d);

#if !V12_ZMMA_ENABLED
        if (Ltern->m_w_in_row)  vk_upload(ctx, p->opt_layers[k].m_w_in_row,  Ltern->m_w_in_row,  mrow_in_bytes);
        else                    vk_fill_zero(ctx, p->opt_layers[k].m_w_in_row,  mrow_in_bytes);
        if (Ltern->v_w_in_row)  vk_upload(ctx, p->opt_layers[k].v_w_in_row,  Ltern->v_w_in_row,  mrow_in_bytes);
        else                    vk_fill_zero(ctx, p->opt_layers[k].v_w_in_row,  mrow_in_bytes);
        if (Ltern->m_w_in_col)  vk_upload(ctx, p->opt_layers[k].m_w_in_col,  Ltern->m_w_in_col,  mcol_in_bytes);
        else                    vk_fill_zero(ctx, p->opt_layers[k].m_w_in_col,  mcol_in_bytes);
        if (Ltern->v_w_in_col)  vk_upload(ctx, p->opt_layers[k].v_w_in_col,  Ltern->v_w_in_col,  mcol_in_bytes);
        else                    vk_fill_zero(ctx, p->opt_layers[k].v_w_in_col,  mcol_in_bytes);
        if (Ltern->m_w_out_row) vk_upload(ctx, p->opt_layers[k].m_w_out_row, Ltern->m_w_out_row, mrow_out_bytes);
        else                    vk_fill_zero(ctx, p->opt_layers[k].m_w_out_row, mrow_out_bytes);
        if (Ltern->v_w_out_row) vk_upload(ctx, p->opt_layers[k].v_w_out_row, Ltern->v_w_out_row, mrow_out_bytes);
        else                    vk_fill_zero(ctx, p->opt_layers[k].v_w_out_row, mrow_out_bytes);
        if (Ltern->m_w_out_col) vk_upload(ctx, p->opt_layers[k].m_w_out_col, Ltern->m_w_out_col, mcol_out_bytes);
        else                    vk_fill_zero(ctx, p->opt_layers[k].m_w_out_col, mcol_out_bytes);
        if (Ltern->v_w_out_col) vk_upload(ctx, p->opt_layers[k].v_w_out_col, Ltern->v_w_out_col, mcol_out_bytes);
        else                    vk_fill_zero(ctx, p->opt_layers[k].v_w_out_col, mcol_out_bytes);
#endif

        /* Factored-Adam scratch buffers for W_in [d_model × d_state] */
        p->opt_layers[k].grad_accum_in = vk_alloc(ctx, win_bytes,       VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].sampled_w_in  = vk_alloc(ctx, win_bytes,       VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
#if V12_ZMMA_ENABLED
        p->opt_layers[k].g_row_in      = VK_BUFFER_NULL;
        p->opt_layers[k].g_col_in      = VK_BUFFER_NULL;
        p->opt_layers[k].gq_row_in     = VK_BUFFER_NULL;
        p->opt_layers[k].gq_col_in     = VK_BUFFER_NULL;
        p->opt_layers[k].cnt_row_in    = VK_BUFFER_NULL;
        p->opt_layers[k].cnt_col_in    = VK_BUFFER_NULL;
        p->opt_layers[k].mean_v_in     = VK_BUFFER_NULL;
#else
        p->opt_layers[k].g_row_in      = vk_alloc(ctx, mrow_in_bytes,   VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].g_col_in      = vk_alloc(ctx, mcol_in_bytes,   VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].gq_row_in     = vk_alloc(ctx, mrow_in_bytes,   VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].gq_col_in     = vk_alloc(ctx, mcol_in_bytes,   VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].cnt_row_in    = vk_alloc(ctx, mrow_in_bytes,   VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].cnt_col_in    = vk_alloc(ctx, mcol_in_bytes,   VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].mean_v_in     = vk_alloc(ctx, 2u * sizeof(float),
                                                   VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
#endif

        vk_fill_zero(ctx, p->opt_layers[k].grad_accum_in, win_bytes);
        if (Ltern->sampled_w_in)
            vk_upload(ctx, p->opt_layers[k].sampled_w_in, Ltern->sampled_w_in, win_bytes);
        else
            vk_fill_zero(ctx, p->opt_layers[k].sampled_w_in, win_bytes);
#if !V12_ZMMA_ENABLED
        vk_fill_zero(ctx, p->opt_layers[k].g_row_in,      mrow_in_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].g_col_in,      mcol_in_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].gq_row_in,     mrow_in_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].gq_col_in,     mcol_in_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].cnt_row_in,    mrow_in_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].cnt_col_in,    mcol_in_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].mean_v_in,     2u * sizeof(float));
#endif

        /* Factored-Adam scratch buffers for W_out [d_state × d_model] */
        p->opt_layers[k].grad_accum_out = vk_alloc(ctx, wout_bytes,      VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].sampled_w_out  = vk_alloc(ctx, wout_bytes,      VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
#if V12_ZMMA_ENABLED
        p->opt_layers[k].g_row_out     = VK_BUFFER_NULL;
        p->opt_layers[k].g_col_out     = VK_BUFFER_NULL;
        p->opt_layers[k].gq_row_out    = VK_BUFFER_NULL;
        p->opt_layers[k].gq_col_out    = VK_BUFFER_NULL;
        p->opt_layers[k].cnt_row_out   = VK_BUFFER_NULL;
        p->opt_layers[k].cnt_col_out   = VK_BUFFER_NULL;
        p->opt_layers[k].mean_v_out    = VK_BUFFER_NULL;
#else
        p->opt_layers[k].g_row_out      = vk_alloc(ctx, mrow_out_bytes,  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].g_col_out      = vk_alloc(ctx, mcol_out_bytes,  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].gq_row_out     = vk_alloc(ctx, mrow_out_bytes,  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].gq_col_out     = vk_alloc(ctx, mcol_out_bytes,  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].cnt_row_out    = vk_alloc(ctx, mrow_out_bytes,  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].cnt_col_out    = vk_alloc(ctx, mcol_out_bytes,  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_layers[k].mean_v_out     = vk_alloc(ctx, 2u * sizeof(float),
                                                    VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
#endif

        vk_fill_zero(ctx, p->opt_layers[k].grad_accum_out, wout_bytes);
        if (Ltern->sampled_w_out)
            vk_upload(ctx, p->opt_layers[k].sampled_w_out, Ltern->sampled_w_out, wout_bytes);
        else
            vk_fill_zero(ctx, p->opt_layers[k].sampled_w_out, wout_bytes);
#if !V12_ZMMA_ENABLED
        vk_fill_zero(ctx, p->opt_layers[k].g_row_out,      mrow_out_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].g_col_out,      mcol_out_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].gq_row_out,     mrow_out_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].gq_col_out,     mcol_out_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].cnt_row_out,    mrow_out_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].cnt_col_out,    mcol_out_bytes);
        vk_fill_zero(ctx, p->opt_layers[k].mean_v_out,     2u * sizeof(float));
#endif

        p->opt_layers[k].opt_allocated = 1;
    }

    /* PRBLT (Pipelined Ring-Buffered Layer Transfer) initialization */
#if V12_PRBLT_ENABLED
    uint32_t max_rows = p->d_model;
    uint32_t max_cols = p->d_state;
    // Max staging size per layer: 4 matrices (csr_in, csr_out, csc_in, csc_out).
    // Each has up to 2 * (max_rows*max_cols) indices and 2 * (max_rows+1) row_ptrs.
    // Total bytes for 4 matrices:
    size_t max_rp_bytes = (max_rows + 1) * sizeof(int32_t);
    size_t max_idx_bytes = (max_rows * max_cols) * sizeof(int32_t);
    size_t shadow_bytes = (max_rows * max_cols) * sizeof(float);
    size_t staging_size_per_layer = 4 * (2 * max_rp_bytes + 2 * max_idx_bytes) + 2 * shadow_bytes;

    for (int r = 0; r < 2; r++) {
        p->prblt_ring[r].active_layer_k = -1;
        p->prblt_ring[r].transfer_handle = vk_create_async_transfer(ctx);
        p->prblt_ring[r].staging_buf = vk_alloc(ctx, staging_size_per_layer, VK_BUF_HOST_VISIBLE | VK_BUF_TRANSFER_SRC);
        p->prblt_ring[r].staging_mapped = (char*)vk_map(ctx, p->prblt_ring[r].staging_buf);
        
        prblt_alloc_max_csr(ctx, &p->prblt_ring[r].gpu_csr_in, max_rows, max_cols);
        prblt_alloc_max_csr(ctx, &p->prblt_ring[r].gpu_csr_out, max_cols, max_rows);
        prblt_alloc_max_csr(ctx, &p->prblt_ring[r].gpu_csc_in, max_cols, max_rows);
        prblt_alloc_max_csr(ctx, &p->prblt_ring[r].gpu_csc_out, max_rows, max_cols);
        p->prblt_ring[r].shadow_w_in = vk_alloc(ctx, shadow_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->prblt_ring[r].shadow_w_out = vk_alloc(ctx, shadow_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    }
#endif


    /* CPU staging */
    size_t max_dw = (d > ds) ? (size_t)d * d : (size_t)d * ds;
    p->staging_dw = malloc(max_dw * sizeof(float));

    /* PC scratch */
    size_t pc_xb = (size_t)batch_size * (seq_len - 1) * d * sizeof(float);
    p->pc_neg_buf = malloc(pc_xb);
    p->pc_res_buf = malloc(pc_xb);
    p->pc_dw_buf  = malloc((size_t)d * d * sizeof(float));

    /* CE loss buffers */
    {
        int n_valid = batch_size * (seq_len - 1);
        int N_SAMPLES = V12_SS_N_SAMPLES;

#if !V12_TOLR_ENABLED
        p->ce_row_ptr_pos = vk_alloc(ctx, (size_t)(m->csr_lm_head.rows + 1) * sizeof(int32_t),
                                     VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->ce_pos_indices = vk_alloc(ctx, (size_t)m->csr_lm_head.total_pos * sizeof(int32_t),
                                     VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->ce_row_ptr_neg = vk_alloc(ctx, (size_t)(m->csr_lm_head.rows + 1) * sizeof(int32_t),
                                     VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->ce_neg_indices = vk_alloc(ctx, (size_t)m->csr_lm_head.total_neg * sizeof(int32_t),
                                     VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);

        vk_upload(ctx, p->ce_row_ptr_pos, m->csr_lm_head.row_ptr_pos,
                  (size_t)(m->csr_lm_head.rows + 1) * sizeof(int32_t));
        vk_upload(ctx, p->ce_pos_indices, m->csr_lm_head.pos_indices,
                  (size_t)m->csr_lm_head.total_pos * sizeof(int32_t));
        vk_upload(ctx, p->ce_row_ptr_neg, m->csr_lm_head.row_ptr_neg,
                  (size_t)(m->csr_lm_head.rows + 1) * sizeof(int32_t));
        vk_upload(ctx, p->ce_neg_indices, m->csr_lm_head.neg_indices,
                  (size_t)m->csr_lm_head.total_neg * sizeof(int32_t));
#else
        p->ce_row_ptr_pos = VK_BUFFER_NULL;
        p->ce_pos_indices = VK_BUFFER_NULL;
        p->ce_row_ptr_neg = VK_BUFFER_NULL;
        p->ce_neg_indices = VK_BUFFER_NULL;
#endif

        p->ce_unigram_cdf = vk_alloc(ctx, (size_t)V12_VOCAB_SIZE * sizeof(float),
                                     VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->ce_log_unigram = vk_alloc(ctx, (size_t)V12_VOCAB_SIZE * sizeof(float),
                                     VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->ce_token_freq  = vk_alloc(ctx, (size_t)V12_VOCAB_SIZE * sizeof(float),
                                     VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        vk_upload(ctx, p->ce_unigram_cdf, m->unigram_cdf,
                  (size_t)V12_VOCAB_SIZE * sizeof(float));
        vk_upload(ctx, p->ce_log_unigram, m->log_unigram,
                  (size_t)V12_VOCAB_SIZE * sizeof(float));
        vk_upload(ctx, p->ce_token_freq, m->token_freq, (size_t)V12_VOCAB_SIZE * sizeof(float));

        p->ce_sample_ids = vk_alloc(ctx, (size_t)n_valid * N_SAMPLES * sizeof(int32_t),
                                    VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
        p->ce_d_logits   = vk_alloc(ctx, (size_t)n_valid * N_SAMPLES * sizeof(float),
                                    VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
        p->ce_ema_logits = vk_alloc(ctx, (size_t)n_valid * N_SAMPLES * sizeof(float),
                                    VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
        {
            size_t ema_bytes = (size_t)n_valid * N_SAMPLES * sizeof(float);
            float *zeros = (float *)calloc(1, ema_bytes);
            if (zeros) {
                vk_upload(ctx, p->ce_ema_logits, zeros, ema_bytes);
                free(zeros);
            }
        }
        p->ce_loss       = vk_alloc(ctx, (size_t)n_valid * sizeof(float),
                                    VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
        p->ce_base_loss  = vk_alloc(ctx, (size_t)n_valid * sizeof(float),
                                    VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
        p->ce_d_norm_w   = vk_alloc(ctx, (size_t)d * sizeof(float),
                                    VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
        p->ce_d_norm_b   = vk_alloc(ctx, (size_t)d * sizeof(float),
                                    VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
#if !V12_TOLR_ENABLED
        if ((m->gpu_opt_flags & GPU_OPT_LM_HEAD) != 0) {
            p->ce_dlm_head   = vk_alloc(ctx, (size_t)V12_VOCAB_SIZE * d * sizeof(float),
                                        VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC | VK_BUF_TRANSFER_DST);
        } else {
            p->ce_dlm_head   = VK_BUFFER_NULL;
        }
#else
        p->ce_dlm_head   = VK_BUFFER_NULL;
#endif
        if (m->corpus_logit_prior_enabled &&
            m->corpus_logit_prior_ctx_counts &&
            m->corpus_logit_prior_pair_keys &&
            m->corpus_logit_prior_pair_counts &&
            m->corpus_logit_prior_unigram &&
            m->corpus_logit_prior_pair_capacity > 0u) {
            p->corpus_prior_enabled = 1;
            p->prior_pair_capacity = m->corpus_logit_prior_pair_capacity;
            p->prior_ctx_counts = vk_alloc(ctx, (size_t)V12_VOCAB_SIZE * sizeof(uint32_t),
                                          VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
            p->prior_pair_keys = vk_alloc(ctx, (size_t)p->prior_pair_capacity * sizeof(uint32_t),
                                         VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
            p->prior_pair_counts = vk_alloc(ctx, (size_t)p->prior_pair_capacity * sizeof(uint32_t),
                                           VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
            p->prior_unigram = vk_alloc(ctx, (size_t)V12_VOCAB_SIZE * sizeof(float),
                                       VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
            vk_upload(ctx, p->prior_ctx_counts, m->corpus_logit_prior_ctx_counts,
                      (size_t)V12_VOCAB_SIZE * sizeof(uint32_t));
            vk_upload(ctx, p->prior_pair_keys, m->corpus_logit_prior_pair_keys,
                      (size_t)p->prior_pair_capacity * sizeof(uint32_t));
            vk_upload(ctx, p->prior_pair_counts, m->corpus_logit_prior_pair_counts,
                      (size_t)p->prior_pair_capacity * sizeof(uint32_t));
            vk_upload(ctx, p->prior_unigram, m->corpus_logit_prior_unigram,
                      (size_t)V12_VOCAB_SIZE * sizeof(float));
            printf("[CORPUS_LOGIT_PRIOR_GPU] active=1 mode=vulkan-sparse-hash-log-prior pair_capacity=%u unique_pairs=%u terms=%llu\n",
                   p->prior_pair_capacity,
                   m->corpus_logit_prior_pair_unique,
                   (unsigned long long)m->corpus_logit_prior_actual_terms);
        } else {
            uint32_t zero_u32 = 0u;
            float zero_f32 = 0.0f;
            p->corpus_prior_enabled = 0;
            p->prior_pair_capacity = 0u;
            p->prior_ctx_counts = vk_alloc(ctx, sizeof(uint32_t),
                                          VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
            p->prior_pair_keys = vk_alloc(ctx, sizeof(uint32_t),
                                         VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
            p->prior_pair_counts = vk_alloc(ctx, sizeof(uint32_t),
                                           VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
            p->prior_unigram = vk_alloc(ctx, sizeof(float),
                                       VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
            vk_upload(ctx, p->prior_ctx_counts, &zero_u32, sizeof(uint32_t));
            vk_upload(ctx, p->prior_pair_keys, &zero_u32, sizeof(uint32_t));
            vk_upload(ctx, p->prior_pair_counts, &zero_u32, sizeof(uint32_t));
            vk_upload(ctx, p->prior_unigram, &zero_f32, sizeof(float));
        }
        if (m->program_kernel_enabled &&
            m->program_kernel_targets &&
            m->program_kernel_counts &&
            m->program_kernel_entries > 0u) {
            p->program_kernel_enabled = 1;
            p->program_kernel_targets = vk_alloc(ctx, (size_t)V12_VOCAB_SIZE * sizeof(uint32_t),
                                                VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
            p->program_kernel_counts = vk_alloc(ctx, (size_t)V12_VOCAB_SIZE * sizeof(uint32_t),
                                               VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
            vk_upload(ctx, p->program_kernel_targets, m->program_kernel_targets,
                      (size_t)V12_VOCAB_SIZE * sizeof(uint32_t));
            vk_upload(ctx, p->program_kernel_counts, m->program_kernel_counts,
                      (size_t)V12_VOCAB_SIZE * sizeof(uint32_t));
            printf("[PROGRAM_KERNEL_GPU] active=1 mode=vulkan-deterministic-transition-table entries=%u logit_boost=%.6f\n",
                   m->program_kernel_entries,
                   m->program_kernel_logit_boost);
        } else {
            uint32_t abstain_u32 = UINT32_MAX;
            uint32_t zero_u32 = 0u;
            p->program_kernel_enabled = 0;
            p->program_kernel_targets = vk_alloc(ctx, sizeof(uint32_t),
                                                VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
            p->program_kernel_counts = vk_alloc(ctx, sizeof(uint32_t),
                                               VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
            vk_upload(ctx, p->program_kernel_targets, &abstain_u32, sizeof(uint32_t));
            vk_upload(ctx, p->program_kernel_counts, &zero_u32, sizeof(uint32_t));
        }
        if (m->state_transition_atlas_enabled &&
            m->state_transition_atlas_keys &&
            m->state_transition_atlas_targets &&
            m->state_transition_atlas_counts &&
            m->state_transition_atlas_capacity > 0u &&
            m->state_transition_atlas_entries > 0u) {
            p->state_atlas_enabled = 1;
            p->state_atlas_capacity = m->state_transition_atlas_capacity;
            p->state_atlas_keys = vk_alloc(ctx, (size_t)p->state_atlas_capacity * sizeof(uint32_t),
                                          VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
            p->state_atlas_targets = vk_alloc(ctx, (size_t)p->state_atlas_capacity * sizeof(uint32_t),
                                             VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
            p->state_atlas_counts = vk_alloc(ctx, (size_t)p->state_atlas_capacity * sizeof(uint32_t),
                                            VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
            vk_upload(ctx, p->state_atlas_keys, m->state_transition_atlas_keys,
                      (size_t)p->state_atlas_capacity * sizeof(uint32_t));
            vk_upload(ctx, p->state_atlas_targets, m->state_transition_atlas_targets,
                      (size_t)p->state_atlas_capacity * sizeof(uint32_t));
            vk_upload(ctx, p->state_atlas_counts, m->state_transition_atlas_counts,
                      (size_t)p->state_atlas_capacity * sizeof(uint32_t));
            printf("[STATE_TRANSITION_ATLAS_GPU] active=1 mode=vulkan-exact-local-state-transition entries=%u table_capacity=%u logit_boost=%.6f\n",
                   m->state_transition_atlas_entries,
                   p->state_atlas_capacity,
                   m->state_transition_atlas_logit_boost);
        } else {
            uint32_t zero_u32 = 0u;
            uint32_t abstain_u32 = UINT32_MAX;
            p->state_atlas_enabled = 0;
            p->state_atlas_capacity = 0u;
            p->state_atlas_keys = vk_alloc(ctx, sizeof(uint32_t),
                                          VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
            p->state_atlas_targets = vk_alloc(ctx, sizeof(uint32_t),
                                             VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
            p->state_atlas_counts = vk_alloc(ctx, sizeof(uint32_t),
                                            VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
            vk_upload(ctx, p->state_atlas_keys, &zero_u32, sizeof(uint32_t));
            vk_upload(ctx, p->state_atlas_targets, &abstain_u32, sizeof(uint32_t));
            vk_upload(ctx, p->state_atlas_counts, &zero_u32, sizeof(uint32_t));
        }
    }

    /* CSR rebuild scratch: enough for the largest matrix (vocab × d_model has most rows) */
    p->csr_count_pos = vk_alloc(ctx, 32064 * sizeof(uint32_t),
                                VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC | VK_BUF_TRANSFER_DST);
    p->csr_count_neg = vk_alloc(ctx, 32064 * sizeof(uint32_t),
                                VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC | VK_BUF_TRANSFER_DST);
    p->flip_counts = vk_alloc(ctx, GPU_FLIP_COUNT_SLOTS * sizeof(uint32_t),
                              VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC | VK_BUF_TRANSFER_DST);
    if (VK_BUFFER_VALID(p->flip_counts))
        vk_fill_zero(ctx, p->flip_counts, GPU_FLIP_COUNT_SLOTS * sizeof(uint32_t));
    p->evidence_rate_stats = vk_alloc(ctx, V12_EVIDENCE_RATE_STATS_SLOTS * sizeof(uint32_t),
                                      VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC | VK_BUF_TRANSFER_DST);
    if (VK_BUFFER_VALID(p->evidence_rate_stats))
        vk_fill_zero(ctx, p->evidence_rate_stats,
                     V12_EVIDENCE_RATE_STATS_SLOTS * sizeof(uint32_t));

    /* lm_head GPU factored-Adam optimizer state */
#if !V12_TOLR_ENABLED
    if ((m->gpu_opt_flags & GPU_OPT_LM_HEAD) != 0) {
        const size_t lm_dense_bytes = (size_t)V12_VOCAB_SIZE * d * sizeof(float);
        const size_t lm_row_bytes   = (size_t)V12_VOCAB_SIZE * sizeof(float);
        const size_t lm_col_bytes   = (size_t)d * sizeof(float);

        p->opt_shadow_lm_head = vk_alloc(ctx, lm_dense_bytes,
                                         VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
        if (m->shadow_lm_head)
            upload_shadow_buffer(ctx, p->opt_shadow_lm_head, m->shadow_lm_head, (size_t)V12_VOCAB_SIZE * d);

        p->opt_m_lm_head_row = vk_alloc(ctx, lm_row_bytes,
                                        VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_v_lm_head_row = vk_alloc(ctx, lm_row_bytes,
                                        VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_m_lm_head_col = vk_alloc(ctx, lm_col_bytes,
                                        VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_v_lm_head_col = vk_alloc(ctx, lm_col_bytes,
                                        VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        if (m->m_lm_head_row)
            upload_lm_hot_row_moments(ctx, p->opt_m_lm_head_row, m->hot_to_vocab, m->m_lm_head_row);
        else
            vk_fill_zero(ctx, p->opt_m_lm_head_row, lm_row_bytes);
        if (m->v_lm_head_row)
            upload_lm_hot_row_moments(ctx, p->opt_v_lm_head_row, m->hot_to_vocab, m->v_lm_head_row);
        else
            vk_fill_zero(ctx, p->opt_v_lm_head_row, lm_row_bytes);
        if (m->m_lm_head_col)
            vk_upload(ctx, p->opt_m_lm_head_col, m->m_lm_head_col, lm_col_bytes);
        else
            vk_fill_zero(ctx, p->opt_m_lm_head_col, lm_col_bytes);
        if (m->v_lm_head_col)
            vk_upload(ctx, p->opt_v_lm_head_col, m->v_lm_head_col, lm_col_bytes);
        else
            vk_fill_zero(ctx, p->opt_v_lm_head_col, lm_col_bytes);

        p->opt_sampled_lm_head = vk_alloc(ctx, lm_dense_bytes,
                                          VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
        if (m->sampled_lm_head)
            vk_upload(ctx, p->opt_sampled_lm_head, m->sampled_lm_head, lm_dense_bytes);
        else
            vk_fill_zero(ctx, p->opt_sampled_lm_head, lm_dense_bytes);

        p->opt_dlm_head = vk_alloc(ctx, lm_dense_bytes,
                                   VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        vk_fill_zero(ctx, p->opt_dlm_head, lm_dense_bytes);

        p->opt_g_row_lm   = vk_alloc(ctx, lm_row_bytes,  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_g_col_lm   = vk_alloc(ctx, lm_col_bytes,  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_gq_row_lm  = vk_alloc(ctx, lm_row_bytes,  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_gq_col_lm  = vk_alloc(ctx, lm_col_bytes,  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_cnt_row_lm = vk_alloc(ctx, lm_row_bytes,  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_cnt_col_lm = vk_alloc(ctx, lm_col_bytes,  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_mean_v_lm  = vk_alloc(ctx, 2u * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST
                                                             | VK_BUF_TRANSFER_SRC);
        p->opt_vocab_to_hot_lm = vk_alloc(ctx, (size_t)V12_VOCAB_SIZE * sizeof(int32_t),
                                          VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        if (m->vocab_to_hot)
            vk_upload(ctx, p->opt_vocab_to_hot_lm, m->vocab_to_hot,
                      (size_t)V12_VOCAB_SIZE * sizeof(int32_t));
        vk_fill_zero(ctx, p->opt_g_row_lm,   lm_row_bytes);
        vk_fill_zero(ctx, p->opt_g_col_lm,   lm_col_bytes);
        vk_fill_zero(ctx, p->opt_gq_row_lm,  lm_row_bytes);
        vk_fill_zero(ctx, p->opt_gq_col_lm,  lm_col_bytes);
        vk_fill_zero(ctx, p->opt_cnt_row_lm, lm_row_bytes);
        vk_fill_zero(ctx, p->opt_cnt_col_lm, lm_col_bytes);
        vk_fill_zero(ctx, p->opt_mean_v_lm,  2u * sizeof(float));
    } else {
        p->opt_shadow_lm_head = VK_BUFFER_NULL;
        p->opt_m_lm_head_row = VK_BUFFER_NULL;
        p->opt_v_lm_head_row = VK_BUFFER_NULL;
        p->opt_m_lm_head_col = VK_BUFFER_NULL;
        p->opt_v_lm_head_col = VK_BUFFER_NULL;
        p->opt_sampled_lm_head = VK_BUFFER_NULL;
        p->opt_dlm_head = VK_BUFFER_NULL;
        p->opt_g_row_lm = VK_BUFFER_NULL;
        p->opt_g_col_lm = VK_BUFFER_NULL;
        p->opt_gq_row_lm = VK_BUFFER_NULL;
        p->opt_gq_col_lm = VK_BUFFER_NULL;
        p->opt_cnt_row_lm = VK_BUFFER_NULL;
        p->opt_cnt_col_lm = VK_BUFFER_NULL;
        p->opt_mean_v_lm = VK_BUFFER_NULL;
        p->opt_vocab_to_hot_lm = VK_BUFFER_NULL;
    }
#else
    p->opt_shadow_lm_head = VK_BUFFER_NULL;
    p->opt_m_lm_head_row = VK_BUFFER_NULL;
    p->opt_v_lm_head_row = VK_BUFFER_NULL;
    p->opt_m_lm_head_col = VK_BUFFER_NULL;
    p->opt_v_lm_head_col = VK_BUFFER_NULL;
    p->opt_sampled_lm_head = VK_BUFFER_NULL;
    p->opt_dlm_head = VK_BUFFER_NULL;
    p->opt_g_row_lm = VK_BUFFER_NULL;
    p->opt_g_col_lm = VK_BUFFER_NULL;
    p->opt_gq_row_lm = VK_BUFFER_NULL;
    p->opt_gq_col_lm = VK_BUFFER_NULL;
    p->opt_cnt_row_lm = VK_BUFFER_NULL;
    p->opt_cnt_col_lm = VK_BUFFER_NULL;
    p->opt_mean_v_lm = VK_BUFFER_NULL;
    p->opt_vocab_to_hot_lm = VK_BUFFER_NULL;
#endif

    /* w_pred GPU factored-Adam optimizer state */
    {
        const size_t wp_dense_bytes = (size_t)d * d * sizeof(float);
        const size_t wp_rc_bytes    = (size_t)d * sizeof(float);

        p->opt_shadow_w_pred = vk_alloc(ctx, wp_dense_bytes,
                                        VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
        if (m->shadow_w_pred)
            upload_shadow_buffer(ctx, p->opt_shadow_w_pred, m->shadow_w_pred, (size_t)d * d);

        p->opt_m_w_pred_row = vk_alloc(ctx, wp_rc_bytes,
                                       VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_v_w_pred_row = vk_alloc(ctx, wp_rc_bytes,
                                       VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_m_w_pred_col = vk_alloc(ctx, wp_rc_bytes,
                                       VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_v_w_pred_col = vk_alloc(ctx, wp_rc_bytes,
                                       VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        if (m->m_w_pred_row)
            vk_upload(ctx, p->opt_m_w_pred_row, m->m_w_pred_row, wp_rc_bytes);
        else
            vk_fill_zero(ctx, p->opt_m_w_pred_row, wp_rc_bytes);
        if (m->v_w_pred_row)
            vk_upload(ctx, p->opt_v_w_pred_row, m->v_w_pred_row, wp_rc_bytes);
        else
            vk_fill_zero(ctx, p->opt_v_w_pred_row, wp_rc_bytes);
        if (m->m_w_pred_col)
            vk_upload(ctx, p->opt_m_w_pred_col, m->m_w_pred_col, wp_rc_bytes);
        else
            vk_fill_zero(ctx, p->opt_m_w_pred_col, wp_rc_bytes);
        if (m->v_w_pred_col)
            vk_upload(ctx, p->opt_v_w_pred_col, m->v_w_pred_col, wp_rc_bytes);
        else
            vk_fill_zero(ctx, p->opt_v_w_pred_col, wp_rc_bytes);

        p->opt_dw_pred = vk_alloc(ctx, wp_dense_bytes,
                                  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
        p->opt_dlm_w_pred = vk_alloc(ctx, wp_dense_bytes,
                                      VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_sampled_w_pred = vk_alloc(ctx, wp_dense_bytes,
                                          VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
        p->opt_g_row_wp   = vk_alloc(ctx, wp_rc_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_g_col_wp   = vk_alloc(ctx, wp_rc_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_gq_row_wp  = vk_alloc(ctx, wp_rc_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_gq_col_wp  = vk_alloc(ctx, wp_rc_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_cnt_row_wp = vk_alloc(ctx, wp_rc_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_cnt_col_wp = vk_alloc(ctx, wp_rc_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        p->opt_mean_v_wp  = vk_alloc(ctx, 2u * sizeof(float),
                                     VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);
        vk_fill_zero(ctx, p->opt_dw_pred, wp_dense_bytes);
        vk_fill_zero(ctx, p->opt_dlm_w_pred, wp_dense_bytes);
        if (m->sampled_w_pred)
            vk_upload(ctx, p->opt_sampled_w_pred, m->sampled_w_pred, wp_dense_bytes);
        else
            vk_fill_zero(ctx, p->opt_sampled_w_pred, wp_dense_bytes);
        vk_fill_zero(ctx, p->opt_g_row_wp, wp_rc_bytes);
        vk_fill_zero(ctx, p->opt_g_col_wp, wp_rc_bytes);
        vk_fill_zero(ctx, p->opt_gq_row_wp, wp_rc_bytes);
        vk_fill_zero(ctx, p->opt_gq_col_wp, wp_rc_bytes);
        vk_fill_zero(ctx, p->opt_cnt_row_wp, wp_rc_bytes);
        vk_fill_zero(ctx, p->opt_cnt_col_wp, wp_rc_bytes);
        vk_fill_zero(ctx, p->opt_mean_v_wp, 2u * sizeof(float));
    }

    return p;
}

static void certified_silent_span_gate_update(
    Model *m, const float *losses, int batch_size, int seq_len) {
    if (!m || !m->certified_silent_span_gate_enabled || !losses ||
        batch_size <= 0 || seq_len <= 1) {
        return;
    }

    const int n_valid = batch_size * (seq_len - 1);
    const uint32_t min_len = m->certified_silent_span_min_len > 0u
        ? m->certified_silent_span_min_len
        : 1u;
    const float max_loss = m->certified_silent_span_max_loss > 0.0f
        ? m->certified_silent_span_max_loss
        : 0.10f;
    uint64_t certified = 0;
    uint64_t spans = 0;
    double grad_bound_sum = 0.0;

    for (int b = 0; b < batch_size; b++) {
        const float *row = losses + (size_t)b * (size_t)(seq_len - 1);
        int run_len = 0;
        double run_grad_bound = 0.0;

        for (int t = 0; t < seq_len - 1; t++) {
            const float L = row[t];
            const int finite = isfinite(L) && L >= 0.0f;
            const int silent = finite && L <= max_loss;
            if (silent) {
                run_len++;
                const double py = exp(-(double)L);
                const double bound = 1.4142135623730951 * (1.0 - py);
                run_grad_bound += bound > 0.0 ? bound : 0.0;
            } else {
                if (run_len >= (int)min_len) {
                    certified += (uint64_t)run_len;
                    spans++;
                    grad_bound_sum += run_grad_bound;
                }
                run_len = 0;
                run_grad_bound = 0.0;
            }
        }
        if (run_len >= (int)min_len) {
            certified += (uint64_t)run_len;
            spans++;
            grad_bound_sum += run_grad_bound;
        }
    }

    const double total = (double)n_valid;
    const double fraction = total > 0.0 ? (double)certified / total : 0.0;
    const double mean_grad_bound = certified > 0
        ? grad_bound_sum / (double)certified
        : 0.0;
    const double priced =
        1.0 - fraction +
        (double)m->certified_silent_span_cert_cost +
        (double)m->certified_silent_span_repair_cost;

    m->certified_silent_span_gate_batches++;
    m->certified_silent_span_gate_positions += (uint64_t)n_valid;
    m->certified_silent_span_gate_tokens += certified;
    m->certified_silent_span_gate_spans += spans;
    m->certified_silent_span_gate_fraction = (float)fraction;
    m->certified_silent_span_gate_grad_bound = (float)mean_grad_bound;
    m->certified_silent_span_gate_priced_ratio = (float)priced;
}

static void certified_artifact_boundary_gate_update(
    Model *m, const uint16_t *tokens, int batch_size, int seq_len) {
    if (!m || !m->certified_artifact_boundary_gate_enabled || !tokens ||
        !m->certified_artifact_boundary_certified ||
        batch_size <= 0 || seq_len <= 1) {
        return;
    }

    uint64_t positions = 0;
    uint64_t routed = 0;
    uint64_t correct = 0;
    uint64_t incorrect = 0;
    uint64_t abstain = 0;

    for (int b = 0; b < batch_size; b++) {
        const uint16_t *tok_b = tokens + (size_t)b * (size_t)seq_len;
        for (int t = 0; t + 1 < seq_len; t++) {
            positions++;
            const uint32_t predicted =
                certified_artifact_boundary_predict(m, tok_b[t]);
            if (predicted == UINT32_MAX) {
                abstain++;
                continue;
            }
            routed++;
            if (predicted == (uint32_t)tok_b[t + 1]) {
                correct++;
            } else {
                incorrect++;
            }
        }
    }

    m->certified_artifact_boundary_gate_batches++;
    m->certified_artifact_boundary_gate_positions += positions;
    m->certified_artifact_boundary_gate_routed += routed;
    m->certified_artifact_boundary_gate_correct += correct;
    m->certified_artifact_boundary_gate_incorrect += incorrect;
    m->certified_artifact_boundary_gate_abstain += abstain;

    const double total = (double)m->certified_artifact_boundary_gate_positions;
    const double routed_total =
        (double)m->certified_artifact_boundary_gate_routed;
    const double incorrect_total =
        (double)m->certified_artifact_boundary_gate_incorrect;
    const double b_k = total > 0.0 ? routed_total / total : 0.0;
    const double false_rate = routed_total > 0.0 ? incorrect_total / routed_total : 0.0;
    const double epsilon = (m->certified_artifact_boundary_epsilon > 0.0f &&
                            m->certified_artifact_boundary_epsilon < 1.0f)
        ? (double)m->certified_artifact_boundary_epsilon
        : 0.001;
    const double routed_ce = -log(1.0 - epsilon);
    const double ce_residual = b_k * routed_ce;
    const double priced =
        (1.0 - b_k) * (double)m->certified_artifact_boundary_kappa +
        (double)m->certified_artifact_boundary_overhead;

    m->certified_artifact_boundary_gate_b_k = (float)b_k;
    m->certified_artifact_boundary_gate_false_route_rate = (float)false_rate;
    m->certified_artifact_boundary_gate_ce_residual = (float)ce_residual;
    m->certified_artifact_boundary_gate_priced_ratio = (float)priced;
    if (m->certified_artifact_boundary_gate_routed +
        m->certified_artifact_boundary_gate_abstain !=
        m->certified_artifact_boundary_gate_positions) {
        m->certified_artifact_boundary_obligation_audit = 0;
    }
}

static void residual_active_sampling_gate_update(
    Model *m, const float *losses, int batch_size, int seq_len) {
    if (!m || !m->residual_active_sampling_gate_enabled || !losses ||
        batch_size <= 0 || seq_len <= 1) {
        return;
    }

    const int n_valid = batch_size * (seq_len - 1);
    const double floor_score = m->residual_active_sampling_floor > 0.0f
        ? (double)m->residual_active_sampling_floor
        : 1.0e-6;
    const double target_fraction =
        (m->residual_active_sampling_target_fraction > 0.0f &&
         m->residual_active_sampling_target_fraction <= 1.0f)
            ? (double)m->residual_active_sampling_target_fraction
            : 0.50;

    double loss_sum = 0.0;
    double score_sum = 0.0;
    double score_sq_sum = 0.0;
    double min_score = 1.0e300;

    for (int i = 0; i < n_valid; i++) {
        const double L = (double)losses[i];
        const double finite_loss = (isfinite(L) && L >= 0.0) ? L : 0.0;
        const double score = finite_loss > floor_score ? finite_loss : floor_score;
        loss_sum += finite_loss;
        score_sum += score;
        score_sq_sum += score * score;
        if (score < min_score) min_score = score;
    }

    const double n = (double)n_valid;
    const double mean_loss = n > 0.0 ? loss_sum / n : 0.0;
    const double mean_score = n > 0.0 ? score_sum / n : 0.0;
    const double var_score = n > 0.0 ? score_sq_sum / n - mean_score * mean_score : 0.0;
    const double score_cv = mean_score > 0.0 && var_score > 0.0
        ? sqrt(var_score) / mean_score
        : 0.0;
    const double ess_fraction =
        (n > 0.0 && score_sq_sum > 0.0)
            ? (score_sum * score_sum) / (n * score_sq_sum)
            : 0.0;
    const double max_weight =
        (n > 0.0 && min_score > 0.0)
            ? score_sum / (n * min_score)
            : 0.0;
    const double variance_proxy =
        (target_fraction > 0.0 && ess_fraction > 0.0)
            ? 1.0 / (target_fraction * ess_fraction)
            : 0.0;

    m->residual_active_sampling_gate_batches++;
    m->residual_active_sampling_gate_positions += (uint64_t)n_valid;
    m->residual_active_sampling_gate_mean_loss = (float)mean_loss;
    m->residual_active_sampling_gate_score_cv = (float)score_cv;
    m->residual_active_sampling_gate_ess_fraction = (float)ess_fraction;
    m->residual_active_sampling_gate_max_weight = (float)max_weight;
    m->residual_active_sampling_gate_variance_proxy = (float)variance_proxy;
}

static uint64_t exact_quotient_padding_mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
}

static int exact_quotient_padding_choose_active_len(
    const Model *m, int seq_len, uint32_t step, uint64_t microbatch, int batch) {
    if (!m || seq_len <= 2) return seq_len;
    uint32_t min_len = m->exact_quotient_padding_min_active_len;
    if (min_len < 2u) min_len = 2u;
    if (min_len > (uint32_t)seq_len) min_len = (uint32_t)seq_len;

    const uint32_t span = (uint32_t)seq_len - min_len + 1u;
    if (span <= 1u) return seq_len;

    uint64_t x = m->exact_quotient_padding_seed;
    x ^= ((uint64_t)step + 0x9e3779b97f4a7c15ull);
    x ^= (microbatch + 0xbf58476d1ce4e5b9ull) << 1;
    x ^= ((uint64_t)(uint32_t)batch + 0x94d049bb133111ebull) << 7;
    x = exact_quotient_padding_mix64(x);
    return (int)(min_len + (uint32_t)(x % span));
}

static int *exact_quotient_padding_build_active_lens(
    Model *m, int batch_size, int seq_len, const TrainState *state,
    int *active_targets_out, int *inactive_targets_out) {
    if (active_targets_out) *active_targets_out = batch_size * (seq_len - 1);
    if (inactive_targets_out) *inactive_targets_out = 0;
    if (!m || !m->exact_quotient_padding_enabled ||
        batch_size <= 0 || seq_len <= 1) {
        return NULL;
    }

    int *active_lens = (int *)malloc((size_t)batch_size * sizeof(int));
    if (!active_lens) {
        fprintf(stderr,
                "[EXACT_QUOTIENT_PADDING] allocation_failed=1 scope=active_lens fallback=full_ce\n");
        return NULL;
    }

    const uint32_t step = state ? state->step : (uint32_t)m->step;
    const uint64_t microbatch = m->exact_quotient_padding_batches;
    int active_targets = 0;
    for (int b = 0; b < batch_size; b++) {
        int active_len = exact_quotient_padding_choose_active_len(
            m, seq_len, step, microbatch, b);
        if (active_len < 2) active_len = 2;
        if (active_len > seq_len) active_len = seq_len;
        active_lens[b] = active_len;
        active_targets += active_len - 1;
    }

    const int total_targets = batch_size * (seq_len - 1);
    const int inactive_targets = total_targets - active_targets;
    if (active_targets_out) *active_targets_out = active_targets;
    if (inactive_targets_out) *inactive_targets_out = inactive_targets;
    return active_lens;
}

static void exact_quotient_padding_record(
    Model *m, int total_targets, int active_targets) {
    if (!m || !m->exact_quotient_padding_enabled || total_targets <= 0) return;
    if (active_targets < 0) active_targets = 0;
    if (active_targets > total_targets) active_targets = total_targets;
    const int inactive_targets = total_targets - active_targets;

    m->exact_quotient_padding_batches++;
    m->exact_quotient_padding_total_targets += (uint64_t)total_targets;
    m->exact_quotient_padding_active_targets += (uint64_t)active_targets;
    m->exact_quotient_padding_inactive_targets += (uint64_t)inactive_targets;
    if (m->exact_quotient_padding_total_targets > 0u) {
        m->exact_quotient_padding_active_fraction =
            (float)((double)m->exact_quotient_padding_active_targets /
                    (double)m->exact_quotient_padding_total_targets);
    }
}

static void prefix_frontier_gate_update(
    Model *m, const uint16_t *tokens, const float *d_logits,
    int batch_size, int seq_len, int n_samples) {
    if (!m || !m->prefix_frontier_gate_enabled || !tokens || !d_logits ||
        batch_size <= 0 || seq_len <= 2 || n_samples <= 0) {
        return;
    }

    uint64_t positions = 0;
    uint64_t supported = 0;
    uint64_t saved_terms = 0;
    uint64_t multiplicity_sum = 0;
    double total_energy = 0.0;
    double supported_energy = 0.0;
    const uint32_t min_count = m->prefix_frontier_min_count > 0u
        ? m->prefix_frontier_min_count
        : 2u;

    for (int b = 0; b < batch_size; b++) {
        const uint16_t *tok_b = tokens + (size_t)b * (size_t)seq_len;
        for (int t = 1; t + 1 < seq_len; t++) {
            const int row = b * (seq_len - 1) + t;
            const float *row_logits = d_logits + (size_t)row * (size_t)n_samples;
            double energy = 0.0;
            for (int s = 0; s < n_samples; s++) {
                const double g = (double)row_logits[s];
                if (isfinite(g)) energy += g * g;
            }
            const uint32_t count = prefix_frontier_count(m, tok_b[t - 1], tok_b[t]);
            positions++;
            total_energy += energy;
            if (count >= min_count) {
                supported++;
                saved_terms += (uint64_t)(count - 1u);
                multiplicity_sum += (uint64_t)count;
                supported_energy += energy;
            }
        }
    }

    m->prefix_frontier_gate_batches++;
    m->prefix_frontier_gate_positions += positions;
    m->prefix_frontier_gate_supported += supported;
    m->prefix_frontier_gate_saved_terms += saved_terms;
    m->prefix_frontier_gate_multiplicity_sum += multiplicity_sum;
    m->prefix_frontier_gate_total_energy += total_energy;
    m->prefix_frontier_gate_supported_energy += supported_energy;

    const double total_positions = (double)m->prefix_frontier_gate_positions;
    const double supported_total = (double)m->prefix_frontier_gate_supported;
    const double saved_total = (double)m->prefix_frontier_gate_saved_terms;
    const double multiplicity_total = (double)m->prefix_frontier_gate_multiplicity_sum;
    const double reuse_fraction =
        total_positions > 0.0 ? fmin(1.0, saved_total / total_positions) : 0.0;
    const double energy_fraction =
        m->prefix_frontier_gate_total_energy > 0.0
            ? m->prefix_frontier_gate_supported_energy /
              m->prefix_frontier_gate_total_energy
            : 0.0;
    const double mean_multiplicity =
        supported_total > 0.0 ? multiplicity_total / supported_total : 0.0;
    double priced = 1.0 - reuse_fraction * energy_fraction
                  + (double)m->prefix_frontier_overhead
                  + (double)m->prefix_frontier_repair_cost;
    if (priced < 0.0) priced = 0.0;

    m->prefix_frontier_gate_reuse_fraction = (float)reuse_fraction;
    m->prefix_frontier_gate_energy_fraction = (float)energy_fraction;
    m->prefix_frontier_gate_mean_multiplicity = (float)mean_multiplicity;
    m->prefix_frontier_gate_priced_ratio = (float)priced;
}

static void qgmf_gate_update(Model *m, const uint16_t *tokens, int batch_size, int seq_len) {
    if (!m || !m->qgmf_gate_enabled || !tokens || batch_size <= 1 || seq_len <= 1) return;
    
    uint64_t collisions = 0;
    for (int t = 1; t < seq_len; t++) {
        for (int b1 = 0; b1 < batch_size; b1++) {
            for (int b2 = b1 + 1; b2 < batch_size; b2++) {
                int match = 1;
                for (int i = 0; i <= t; i++) {
                    if (tokens[b1 * seq_len + i] != tokens[b2 * seq_len + i]) {
                        match = 0;
                        break;
                    }
                }
                if (match) {
                    collisions++;
                }
            }
        }
    }
    
    m->qgmf_gate_batches++;
    m->qgmf_gate_collisions += collisions;
    
    if (collisions > 0 || (m->qgmf_gate_batches % 100 == 0)) {
        printf("[F08 QGMF Gate] Quotient Group Memory Factorization - Prefix collisions: %llu (Batches: %llu)\n", 
            (unsigned long long)collisions, (unsigned long long)m->qgmf_gate_batches);
    }
}

static void exact_window_multiset_gate_update(
    Model *m, const uint16_t *tokens, int batch_size, int seq_len) {
    if (!m || !m->exact_window_multiset_gate_enabled || !tokens ||
        batch_size <= 0 || seq_len <= 1) {
        return;
    }

    uint64_t positions = 0;
    uint64_t supported = 0;
    uint64_t saved_terms = 0;
    uint64_t multiplicity_sum = 0;
    const uint32_t min_count = m->exact_window_multiset_min_count > 0u
        ? m->exact_window_multiset_min_count
        : 1u;

    for (int b = 0; b < batch_size; b++) {
        const uint16_t *tok_b = tokens + (size_t)b * (size_t)seq_len;
        for (int t = 0; t + 1 < seq_len; t++) {
            const uint32_t count =
                exact_window_multiset_count_live(m, tok_b, seq_len, t);
            positions++;
            if (count >= min_count) {
                supported++;
                multiplicity_sum += (uint64_t)count;
                saved_terms += (uint64_t)(count - 1u);
            }
        }
    }

    m->exact_window_multiset_gate_batches++;
    m->exact_window_multiset_gate_positions += positions;
    m->exact_window_multiset_gate_supported += supported;
    m->exact_window_multiset_gate_saved_terms += saved_terms;
    m->exact_window_multiset_gate_multiplicity_sum += multiplicity_sum;

    const double total = (double)m->exact_window_multiset_gate_positions;
    const double supported_total = (double)m->exact_window_multiset_gate_supported;
    const double saved_total = (double)m->exact_window_multiset_gate_saved_terms;
    const double multiplicity_total = (double)m->exact_window_multiset_gate_multiplicity_sum;
    m->exact_window_multiset_gate_live_coverage =
        total > 0.0 ? (float)(supported_total / total) : 0.0f;
    m->exact_window_multiset_gate_saved_fraction =
        multiplicity_total > 0.0 ? (float)(saved_total / multiplicity_total) : 0.0f;
    m->exact_window_multiset_gate_mean_multiplicity =
        supported_total > 0.0 ? (float)(multiplicity_total / supported_total) : 0.0f;
    m->exact_window_multiset_gate_priced_ratio =
        m->exact_window_multiset_overhead +
        m->exact_window_multiset_gate_q_ratio *
            (1.0f + m->exact_window_multiset_noise_penalty);
}

static void active_state_subspace_gate_update(
    Model *m, const float *grad, int batch_size, int seq_len, int d) {
    if (!m || !m->active_state_subspace_gate_enabled || !grad ||
        batch_size <= 0 || seq_len <= 0 || d <= 0) {
        return;
    }

    double dim_energy[V12_D_MODEL];
    for (int j = 0; j < d; j++) dim_energy[j] = 0.0;

    const int positions = batch_size * seq_len;
    double total_energy = 0.0;
    for (int i = 0; i < positions; i++) {
        const float *row = grad + (size_t)i * (size_t)d;
        for (int j = 0; j < d; j++) {
            const double g = (double)row[j];
            if (!isfinite(g)) continue;
            const double e = g * g;
            dim_energy[j] += e;
            total_energy += e;
        }
    }

    uint32_t k = m->active_state_subspace_k > 0u
        ? m->active_state_subspace_k
        : (uint32_t)d;
    if (k > (uint32_t)d) k = (uint32_t)d;

    double topk_energy = 0.0;
    for (uint32_t pick = 0; pick < k; pick++) {
        int best = -1;
        double best_energy = -1.0;
        for (int j = 0; j < d; j++) {
            if (dim_energy[j] > best_energy) {
                best_energy = dim_energy[j];
                best = j;
            }
        }
        if (best < 0 || best_energy <= 0.0) break;
        topk_energy += best_energy;
        dim_energy[best] = -1.0;
    }

    m->active_state_subspace_gate_batches++;
    m->active_state_subspace_gate_positions += (uint64_t)positions;
    m->active_state_subspace_gate_total_energy += total_energy;
    m->active_state_subspace_gate_topk_energy += topk_energy;

    const double k_over_d = d > 0 ? (double)k / (double)d : 1.0;
    const double energy_fraction =
        m->active_state_subspace_gate_total_energy > 0.0
            ? m->active_state_subspace_gate_topk_energy /
              m->active_state_subspace_gate_total_energy
            : 0.0;
    const double tail_regret = fmax(0.0, 1.0 - energy_fraction);
    double priced = 0.0;
    if (energy_fraction > 1.0e-30) {
        priced = (k_over_d + (double)m->active_state_subspace_overhead + tail_regret)
               / energy_fraction;
    }

    m->active_state_subspace_gate_k_over_d = (float)k_over_d;
    m->active_state_subspace_gate_energy_fraction = (float)energy_fraction;
    m->active_state_subspace_gate_tail_regret = (float)tail_regret;
    m->active_state_subspace_gate_priced_ratio = (float)priced;
}

static void boundary_liability_ledger_gate_update(
    Model *m, const float *grad, const uint8_t *boundary_mask,
    int batch_size, int seq_len, int d) {
    if (!m || !m->boundary_liability_ledger_gate_enabled || !grad ||
        !boundary_mask || batch_size <= 0 || seq_len <= 0 || d <= 0) {
        return;
    }

    const int positions = batch_size * seq_len;
    double total_energy = 0.0;
    double boundary_energy = 0.0;
    uint64_t boundary_positions = 0;

    for (int i = 0; i < positions; i++) {
        const float *row = grad + (size_t)i * (size_t)d;
        double row_energy = 0.0;
        for (int j = 0; j < d; j++) {
            const double g = (double)row[j];
            if (isfinite(g)) row_energy += g * g;
        }
        total_energy += row_energy;
        if (boundary_mask[i]) {
            boundary_positions++;
            boundary_energy += row_energy;
        }
    }

    const double mean_energy =
        positions > 0 ? total_energy / (double)positions : 0.0;
    const double eps = (m->boundary_liability_null_epsilon > 0.0f)
        ? (double)m->boundary_liability_null_epsilon
        : 1.0e-6;
    const double null_threshold = mean_energy * eps;
    uint64_t null_positions = 0;
    uint64_t boundary_null_positions = 0;
    double null_energy = 0.0;
    double boundary_null_energy = 0.0;

    for (int i = 0; i < positions; i++) {
        const float *row = grad + (size_t)i * (size_t)d;
        double row_energy = 0.0;
        for (int j = 0; j < d; j++) {
            const double g = (double)row[j];
            if (isfinite(g)) row_energy += g * g;
        }
        if (row_energy <= null_threshold) {
            null_positions++;
            null_energy += row_energy;
            if (boundary_mask[i]) {
                boundary_null_positions++;
                boundary_null_energy += row_energy;
            }
        }
    }

    m->boundary_liability_gate_batches++;
    m->boundary_liability_gate_positions += (uint64_t)positions;
    m->boundary_liability_gate_boundary_positions += boundary_positions;
    m->boundary_liability_gate_null_positions += null_positions;
    m->boundary_liability_gate_boundary_null_positions += boundary_null_positions;
    m->boundary_liability_gate_total_energy += total_energy;
    m->boundary_liability_gate_boundary_energy += boundary_energy;
    m->boundary_liability_gate_null_energy += null_energy;
    m->boundary_liability_gate_boundary_null_energy += boundary_null_energy;

    const double total_positions = (double)m->boundary_liability_gate_positions;
    const double boundary_total =
        (double)m->boundary_liability_gate_boundary_positions;
    const double null_total = (double)m->boundary_liability_gate_null_positions;
    const double boundary_null_total =
        (double)m->boundary_liability_gate_boundary_null_positions;
    const double energy_total = m->boundary_liability_gate_total_energy;
    const double boundary_null_fraction =
        total_positions > 0.0 ? boundary_null_total / total_positions : 0.0;
    const double false_discharge_energy =
        energy_total > 0.0
            ? m->boundary_liability_gate_boundary_null_energy / energy_total
            : 0.0;
    const double priced_gain =
        boundary_null_fraction - false_discharge_energy
        - (double)m->boundary_liability_overhead;

    m->boundary_liability_gate_boundary_fraction =
        total_positions > 0.0 ? (float)(boundary_total / total_positions) : 0.0f;
    m->boundary_liability_gate_boundary_energy_fraction =
        energy_total > 0.0
            ? (float)(m->boundary_liability_gate_boundary_energy / energy_total)
            : 0.0f;
    m->boundary_liability_gate_null_fraction =
        total_positions > 0.0 ? (float)(null_total / total_positions) : 0.0f;
    m->boundary_liability_gate_boundary_null_fraction =
        (float)boundary_null_fraction;
    m->boundary_liability_gate_false_discharge_energy =
        (float)false_discharge_energy;
    m->boundary_liability_gate_priced_gain = (float)priced_gain;
    m->boundary_liability_gate_pass = (priced_gain > 0.10) ? 1 : 0;
}

static void boundary_flux_sheaf_gate_update(
    Model *m, const float *transition_credit, const uint8_t *boundary_mask,
    int batch_size, int seq_len, int d_state) {
    if (!m || !m->boundary_flux_sheaf_gate_enabled || !transition_credit ||
        !boundary_mask || batch_size <= 0 || seq_len <= 0 || d_state <= 0) {
        return;
    }

    const int positions = batch_size * seq_len;
    double total_energy = 0.0;
    double boundary_energy = 0.0;
    double pre_boundary_energy = 0.0;
    double endpoint_energy = 0.0;
    uint64_t boundary_positions = 0;
    uint64_t pre_boundary_positions = 0;
    uint64_t endpoint_positions = 0;

    for (int b = 0; b < batch_size; b++) {
        const size_t batch_base = (size_t)b * (size_t)seq_len;
        for (int t = 0; t < seq_len; t++) {
            const size_t pos = batch_base + (size_t)t;
            const float *row = transition_credit + pos * (size_t)d_state;
            double row_energy = 0.0;
            for (int i = 0; i < d_state; i++) {
                const double g = (double)row[i];
                if (isfinite(g)) row_energy += g * g;
            }

            const int is_boundary = boundary_mask[pos] ? 1 : 0;
            const int is_pre_boundary =
                (t + 1 < seq_len && boundary_mask[pos + 1u]) ? 1 : 0;
            const int is_endpoint = is_boundary || is_pre_boundary;

            total_energy += row_energy;
            if (is_boundary) {
                boundary_positions++;
                boundary_energy += row_energy;
            }
            if (is_pre_boundary) {
                pre_boundary_positions++;
                pre_boundary_energy += row_energy;
            }
            if (is_endpoint) {
                endpoint_positions++;
                endpoint_energy += row_energy;
            }
        }
    }

    m->boundary_flux_gate_batches++;
    m->boundary_flux_gate_layers++;
    m->boundary_flux_gate_positions += (uint64_t)positions;
    m->boundary_flux_gate_boundary_positions += boundary_positions;
    m->boundary_flux_gate_pre_boundary_positions += pre_boundary_positions;
    m->boundary_flux_gate_endpoint_positions += endpoint_positions;
    m->boundary_flux_gate_total_energy += total_energy;
    m->boundary_flux_gate_boundary_energy += boundary_energy;
    m->boundary_flux_gate_pre_boundary_energy += pre_boundary_energy;
    m->boundary_flux_gate_endpoint_energy += endpoint_energy;

    const double total_positions = (double)m->boundary_flux_gate_positions;
    const double energy_total = m->boundary_flux_gate_total_energy;
    const double endpoint_fraction =
        total_positions > 0.0
            ? (double)m->boundary_flux_gate_endpoint_positions / total_positions
            : 0.0;
    const double energy_fraction =
        energy_total > 0.0
            ? m->boundary_flux_gate_endpoint_energy / energy_total
            : 0.0;
    const int identifying = (energy_total > 0.0) ? 1 : 0;
    const double kappa_est = fmax(0.0, 1.0 - energy_fraction);
    const double priced_gain =
        energy_fraction * (1.0 - kappa_est) -
        (double)m->boundary_flux_sheaf_overhead;

    m->boundary_flux_gate_boundary_fraction =
        total_positions > 0.0
            ? (float)((double)m->boundary_flux_gate_boundary_positions /
                      total_positions)
            : 0.0f;
    m->boundary_flux_gate_pre_boundary_fraction =
        total_positions > 0.0
            ? (float)((double)m->boundary_flux_gate_pre_boundary_positions /
                      total_positions)
            : 0.0f;
    m->boundary_flux_gate_endpoint_fraction = (float)endpoint_fraction;
    m->boundary_flux_gate_energy_fraction = (float)energy_fraction;
    m->boundary_flux_gate_kappa_est = (float)kappa_est;
    m->boundary_flux_gate_priced_gain = (float)priced_gain;
    m->boundary_flux_gate_identifying = identifying;
    m->boundary_flux_gate_pass =
        (identifying && priced_gain > 0.10) ? 1 : 0;
}

static void boundary_cotangent_credit_gate_update(
    Model *m, const float *transition_credit, const uint8_t *boundary_mask,
    int batch_size, int seq_len, int d_state) {
    if (!m || !m->boundary_cotangent_credit_gate_enabled ||
        !transition_credit || !boundary_mask ||
        batch_size <= 0 || seq_len <= 0 || d_state <= 0) {
        return;
    }

    int *anchors = (int *)malloc((size_t)seq_len * sizeof(int));
    if (!anchors) return;

    const int positions = batch_size * seq_len;
    double exact_energy = 0.0;
    double recon_energy = 0.0;
    double dot = 0.0;
    double residual_energy = 0.0;
    uint64_t anchor_positions = 0;

    for (int b = 0; b < batch_size; b++) {
        const size_t batch_base = (size_t)b * (size_t)seq_len;
        int anchor_count = 0;

        for (int t = 0; t < seq_len; t++) {
            const size_t pos = batch_base + (size_t)t;
            const int is_boundary = boundary_mask[pos] ? 1 : 0;
            const int is_pre_boundary =
                (t + 1 < seq_len && boundary_mask[pos + 1u]) ? 1 : 0;
            if (t == 0 || t + 1 == seq_len || is_boundary || is_pre_boundary) {
                anchors[anchor_count++] = t;
            }
        }

        if (anchor_count <= 0) {
            anchors[anchor_count++] = 0;
        }
        anchor_positions += (uint64_t)anchor_count;

        if (anchor_count == 1) {
            const float *anchor_row =
                transition_credit + (batch_base + (size_t)anchors[0]) *
                (size_t)d_state;
            for (int t = 0; t < seq_len; t++) {
                const float *row =
                    transition_credit + (batch_base + (size_t)t) *
                    (size_t)d_state;
                for (int i = 0; i < d_state; i++) {
                    const double g = (double)row[i];
                    if (!isfinite(g)) continue;
                    const double r =
                        isfinite((double)anchor_row[i]) ?
                        (double)anchor_row[i] : 0.0;
                    const double e = g - r;
                    exact_energy += g * g;
                    recon_energy += r * r;
                    dot += g * r;
                    residual_energy += e * e;
                }
            }
            continue;
        }

        for (int a_idx = 0; a_idx + 1 < anchor_count; a_idx++) {
            const int left_t = anchors[a_idx];
            const int right_t = anchors[a_idx + 1];
            const int width = right_t - left_t;
            const int start_t = (a_idx == 0) ? left_t : left_t + 1;
            const float *left_row =
                transition_credit + (batch_base + (size_t)left_t) *
                (size_t)d_state;
            const float *right_row =
                transition_credit + (batch_base + (size_t)right_t) *
                (size_t)d_state;

            for (int t = start_t; t <= right_t; t++) {
                const double alpha =
                    width > 0 ? (double)(t - left_t) / (double)width : 0.0;
                const float *row =
                    transition_credit + (batch_base + (size_t)t) *
                    (size_t)d_state;
                for (int i = 0; i < d_state; i++) {
                    const double g = (double)row[i];
                    if (!isfinite(g)) continue;
                    const double l =
                        isfinite((double)left_row[i]) ?
                        (double)left_row[i] : 0.0;
                    const double r_anchor =
                        isfinite((double)right_row[i]) ?
                        (double)right_row[i] : 0.0;
                    const double r = l + alpha * (r_anchor - l);
                    const double e = g - r;
                    exact_energy += g * g;
                    recon_energy += r * r;
                    dot += g * r;
                    residual_energy += e * e;
                }
            }
        }
    }

    free(anchors);

    m->boundary_cotangent_gate_batches++;
    m->boundary_cotangent_gate_layers++;
    m->boundary_cotangent_gate_positions += (uint64_t)positions;
    m->boundary_cotangent_gate_anchor_positions += anchor_positions;
    m->boundary_cotangent_gate_exact_energy += exact_energy;
    m->boundary_cotangent_gate_recon_energy += recon_energy;
    m->boundary_cotangent_gate_dot += dot;
    m->boundary_cotangent_gate_residual_energy += residual_energy;

    const double total_positions =
        (double)m->boundary_cotangent_gate_positions;
    const double anchors_total =
        (double)m->boundary_cotangent_gate_anchor_positions;
    const double exact_total = m->boundary_cotangent_gate_exact_energy;
    const double recon_total = m->boundary_cotangent_gate_recon_energy;
    const double dot_total = m->boundary_cotangent_gate_dot;
    const double residual_total = m->boundary_cotangent_gate_residual_energy;
    const double anchor_fraction =
        total_positions > 0.0 ? anchors_total / total_positions : 1.0;
    const double residual_fraction =
        exact_total > 0.0 ? residual_total / exact_total : 1.0;
    const double energy_covered = fmax(0.0, 1.0 - residual_fraction);
    const double cosine_den = sqrt(exact_total * recon_total);
    const double cosine = cosine_den > 0.0 ? dot_total / cosine_den : 0.0;
    const double verify_fraction =
        m->boundary_cotangent_verify_fraction >= 0.0f
            ? (double)m->boundary_cotangent_verify_fraction
            : 1.0;
    const double overhead = (double)m->boundary_cotangent_overhead;
    const double priced_ratio =
        anchor_fraction + verify_fraction + overhead + residual_fraction;
    const double unverified_priced_ratio =
        anchor_fraction + overhead + residual_fraction;
    const double descent_per_paid_work =
        priced_ratio > 0.0 ? energy_covered / priced_ratio : 0.0;
    const double unverified_descent_per_paid_work =
        unverified_priced_ratio > 0.0
            ? energy_covered / unverified_priced_ratio
            : 0.0;
    const int identifying =
        (m->boundary_cotangent_gate_positions > 0u && exact_total > 0.0) ? 1 : 0;

    m->boundary_cotangent_gate_anchor_fraction = (float)anchor_fraction;
    m->boundary_cotangent_gate_saved_recurrent_fraction =
        (float)fmax(0.0, 1.0 - anchor_fraction);
    m->boundary_cotangent_gate_cosine = (float)cosine;
    m->boundary_cotangent_gate_residual_fraction = (float)residual_fraction;
    m->boundary_cotangent_gate_energy_covered = (float)energy_covered;
    m->boundary_cotangent_gate_basis_bytes_per_token =
        total_positions > 0.0
            ? (float)((anchors_total * (double)d_state * sizeof(float)) /
                      total_positions)
            : 0.0f;
    m->boundary_cotangent_gate_verification_fraction = (float)verify_fraction;
    m->boundary_cotangent_gate_priced_ratio = (float)priced_ratio;
    m->boundary_cotangent_gate_unverified_priced_ratio =
        (float)unverified_priced_ratio;
    m->boundary_cotangent_gate_descent_per_paid_work =
        (float)descent_per_paid_work;
    m->boundary_cotangent_gate_unverified_descent_per_paid_work =
        (float)unverified_descent_per_paid_work;
    m->boundary_cotangent_gate_identifying = identifying;
    m->boundary_cotangent_gate_pass =
        (identifying &&
         residual_fraction <= (double)m->boundary_cotangent_residual_budget &&
         priced_ratio < 0.90 &&
         descent_per_paid_work > 1.0) ? 1 : 0;
}

static double variance_priced_hash01(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return (double)(x >> 8) * (1.0 / 16777216.0);
}

static int variance_priced_credit_gate_update(
    Model *m, float *transition_credit, int batch_size,
    int seq_len, int d_state) {
    if (!m || !m->variance_priced_credit_gate_enabled ||
        !transition_credit || batch_size <= 0 || seq_len <= 0 ||
        d_state <= 0) {
        return 0;
    }

    const int items = batch_size * seq_len;
    float *norms = (float *)malloc((size_t)items * sizeof(float));
    double *probs = (double *)calloc((size_t)items, sizeof(double));
    uint8_t *sampled_mask = (uint8_t *)calloc((size_t)items, sizeof(uint8_t));
    double *exact_sum = (double *)calloc((size_t)d_state, sizeof(double));
    double *sampled_sum = (double *)calloc((size_t)d_state, sizeof(double));
    if (!norms || !probs || !sampled_mask || !exact_sum || !sampled_sum) {
        free(norms);
        free(probs);
        free(sampled_mask);
        free(exact_sum);
        free(sampled_sum);
        return 0;
    }

    double norm_sum = 0.0;
    double norm_sq_sum = 0.0;
    double max_norm = 0.0;
    int nonzero_items = 0;

    for (int pos = 0; pos < items; pos++) {
        const float *row = transition_credit + (size_t)pos * (size_t)d_state;
        double e = 0.0;
        for (int i = 0; i < d_state; i++) {
            const double g = (double)row[i];
            if (isfinite(g)) e += g * g;
        }
        const double n = sqrt(e);
        norms[pos] = (float)n;
        norm_sum += n;
        norm_sq_sum += n * n;
        if (n > max_norm) max_norm = n;
        if (n > 0.0) nonzero_items++;
    }

    double sample_fraction = (double)m->variance_priced_sample_fraction;
    if (sample_fraction < 0.0) sample_fraction = 0.0;
    if (sample_fraction > 1.0) sample_fraction = 1.0;
    const double target_samples = sample_fraction * (double)items;

    double lambda = 0.0;
    if (target_samples >= (double)nonzero_items) {
        lambda = INFINITY;
    } else if (target_samples > 0.0 && norm_sum > 0.0) {
        double lo = 0.0;
        double hi = 1.0;
        for (int grow = 0; grow < 32; grow++) {
            double s = 0.0;
            for (int pos = 0; pos < items; pos++) {
                const double n = (double)norms[pos];
                if (n > 0.0) s += fmin(1.0, hi * n);
            }
            if (s >= target_samples) break;
            hi *= 2.0;
        }
        for (int it = 0; it < 40; it++) {
            const double mid = 0.5 * (lo + hi);
            double s = 0.0;
            for (int pos = 0; pos < items; pos++) {
                const double n = (double)norms[pos];
                if (n > 0.0) s += fmin(1.0, mid * n);
            }
            if (s < target_samples) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lambda = hi;
    }

    uint64_t sampled_items = 0;
    const uint32_t gate_salt =
        (uint32_t)(m->variance_priced_gate_layers * 747796405u) ^
        (uint32_t)(m->variance_priced_gate_batches * 277803737u);

    for (int pos = 0; pos < items; pos++) {
        const float *row = transition_credit + (size_t)pos * (size_t)d_state;
        const double n = (double)norms[pos];
        double p_i = 0.0;
        if (n > 0.0) {
            p_i = isinf(lambda) ? 1.0 : fmin(1.0, lambda * n);
        }
        const uint32_t h =
            gate_salt ^ (uint32_t)(pos * 2891336453u + 0x9e3779b9u);
        const int sampled = (p_i >= 1.0 || variance_priced_hash01(h) < p_i);
        probs[pos] = p_i;
        sampled_mask[pos] = sampled ? 1u : 0u;
        if (sampled) sampled_items++;

        for (int i = 0; i < d_state; i++) {
            const double g = (double)row[i];
            if (!isfinite(g)) continue;
            exact_sum[i] += g;
            if (sampled && p_i > 0.0) {
                sampled_sum[i] += g / p_i;
            }
        }
    }

    double exact_energy = 0.0;
    double sampled_energy = 0.0;
    double dot = 0.0;
    double residual_energy = 0.0;
    for (int i = 0; i < d_state; i++) {
        const double g = exact_sum[i];
        const double s = sampled_sum[i];
        const double r = s - g;
        exact_energy += g * g;
        sampled_energy += s * s;
        dot += g * s;
        residual_energy += r * r;
    }

    const double local_sampled_fraction =
        items > 0 ? (double)sampled_items / (double)items : 0.0;
    const double local_alignment =
        exact_energy > 0.0 ? dot / exact_energy : 0.0;
    const double local_variance_proxy =
        exact_energy > 0.0 ? residual_energy / exact_energy : 1.0;
    const double priced_work =
        local_sampled_fraction +
        (double)m->variance_priced_score_fraction +
        (double)m->variance_priced_verify_fraction +
        (double)m->variance_priced_overhead;
    const double local_descent_proxy =
        fmax(0.0, local_alignment - local_variance_proxy);
    const double local_descent_per_paid_work =
        priced_work > 0.0 ? local_descent_proxy / priced_work : 0.0;
    const int local_identifying = (items > 0 && exact_energy > 0.0) ? 1 : 0;
    const double min_descent_per_paid =
        m->amortized_ce_martingale_min_descent_per_paid_work > 0.0f
            ? (double)m->amortized_ce_martingale_min_descent_per_paid_work
            : 1.111;
    const int should_apply =
        (m->amortized_ce_martingale_training_enabled &&
         local_identifying &&
         local_descent_per_paid_work >= min_descent_per_paid) ? 1 : 0;

    m->amortized_ce_martingale_last_descent_per_paid_work =
        (float)local_descent_per_paid_work;
    m->amortized_ce_martingale_last_sampled_fraction =
        (float)local_sampled_fraction;
    m->amortized_ce_martingale_last_applied = should_apply;
    if (m->amortized_ce_martingale_training_enabled) {
        m->amortized_ce_martingale_training_layers++;
        if (should_apply) {
            m->amortized_ce_martingale_applied_layers++;
        } else {
            m->amortized_ce_martingale_fallback_layers++;
        }
    }

    if (should_apply) {
        for (int pos = 0; pos < items; pos++) {
            float *row = transition_credit + (size_t)pos * (size_t)d_state;
            const double p_i = probs[pos];
            const int sampled = sampled_mask[pos] != 0u;
            const float scale = (sampled && p_i > 0.0)
                ? (float)(1.0 / p_i)
                : 0.0f;
            for (int i = 0; i < d_state; i++) {
                row[i] = isfinite((double)row[i]) ? row[i] * scale : 0.0f;
            }
        }
    }

    m->variance_priced_gate_batches++;
    m->variance_priced_gate_layers++;
    m->variance_priced_gate_items += (uint64_t)items;
    m->variance_priced_gate_sampled_items += sampled_items;
    m->variance_priced_gate_norm_sum += norm_sum;
    m->variance_priced_gate_norm_sq_sum += norm_sq_sum;
    m->variance_priced_gate_exact_energy += exact_energy;
    m->variance_priced_gate_sampled_energy += sampled_energy;
    m->variance_priced_gate_dot += dot;
    m->variance_priced_gate_residual_energy += residual_energy;
    if (max_norm > (double)m->variance_priced_gate_max_norm) {
        m->variance_priced_gate_max_norm = (float)max_norm;
    }

    const double total_items = (double)m->variance_priced_gate_items;
    const double sampled_total = (double)m->variance_priced_gate_sampled_items;
    const double norm_total = m->variance_priced_gate_norm_sum;
    const double norm_sq_total = m->variance_priced_gate_norm_sq_sum;
    const double exact_total = m->variance_priced_gate_exact_energy;
    const double dot_total = m->variance_priced_gate_dot;
    const double residual_total = m->variance_priced_gate_residual_energy;
    const double sampled_fraction =
        total_items > 0.0 ? sampled_total / total_items : 0.0;
    const double norm_ess_fraction =
        (total_items > 0.0 && norm_sq_total > 0.0)
            ? (norm_total * norm_total) / (total_items * norm_sq_total)
            : 0.0;
    const double alignment =
        exact_total > 0.0 ? dot_total / exact_total : 0.0;
    const double variance_proxy =
        exact_total > 0.0 ? residual_total / exact_total : 1.0;
    const double aggregate_priced_work =
        sampled_fraction +
        (double)m->variance_priced_score_fraction +
        (double)m->variance_priced_verify_fraction +
        (double)m->variance_priced_overhead;
    const double descent_proxy = fmax(0.0, alignment - variance_proxy);
    const double descent_per_paid_work =
        aggregate_priced_work > 0.0 ? descent_proxy / aggregate_priced_work : 0.0;
    const int identifying =
        (m->variance_priced_gate_items > 0u && exact_total > 0.0) ? 1 : 0;

    m->variance_priced_gate_sampled_fraction = (float)sampled_fraction;
    m->variance_priced_gate_norm_ess_fraction = (float)norm_ess_fraction;
    m->variance_priced_gate_mean_norm =
        total_items > 0.0 ? (float)(norm_total / total_items) : 0.0f;
    m->variance_priced_gate_alignment = (float)alignment;
    m->variance_priced_gate_variance_proxy = (float)variance_proxy;
    m->variance_priced_gate_priced_work = (float)aggregate_priced_work;
    m->variance_priced_gate_descent_proxy = (float)descent_proxy;
    m->variance_priced_gate_descent_per_paid_work =
        (float)descent_per_paid_work;
    m->variance_priced_gate_identifying = identifying;
    m->variance_priced_gate_pass =
        (identifying && descent_per_paid_work > 1.111) ? 1 : 0;

    free(norms);
    free(probs);
    free(sampled_mask);
    free(exact_sum);
    free(sampled_sum);
    return should_apply;
}

static double safe_f64(float v) {
    return isfinite((double)v) ? (double)v : 0.0;
}

static double relative_residual_or_zero(double exact_energy, double residual_energy) {
    if (exact_energy > 1.0e-20) return residual_energy / exact_energy;
    return residual_energy <= 1.0e-20 ? 0.0 : 1.0;
}

static int lagged_ce_impulse_eligibility_update(
    Model *m, SSMLayer *L, float *transition_credit, const float *ce_impulse,
    const float *h_cache, const float *a_eff, const float *a_gate,
    const float *x_cv, const uint8_t *boundary_mask,
    int batch_size, int seq_len, int d_state) {
    if (!m || !m->lagged_ce_impulse_eligibility_enabled || !L ||
        !transition_credit || !ce_impulse || !h_cache || !a_eff ||
        !a_gate || !x_cv || batch_size <= 0 || seq_len <= 0 ||
        d_state <= 0) {
        return 0;
    }

    uint32_t horizon = m->lagged_ce_impulse_horizon;
    if (!(horizon == 2u || horizon == 4u || horizon == 8u)) {
        horizon = 4u;
    }

    const int items = batch_size * seq_len;
    const size_t scalar_count = (size_t)items * (size_t)d_state;
    float *ledger_credit = (float *)malloc(scalar_count * sizeof(float));
    float *exact_dot = (float *)malloc(scalar_count * sizeof(float));
    float *ledger_dot = (float *)malloc(scalar_count * sizeof(float));
    double *exact_dA = (double *)calloc((size_t)d_state, sizeof(double));
    double *ledger_dA = (double *)calloc((size_t)d_state, sizeof(double));
    double *exact_dbA = (double *)calloc((size_t)d_state, sizeof(double));
    double *ledger_dbA = (double *)calloc((size_t)d_state, sizeof(double));
    if (!ledger_credit || !exact_dot || !ledger_dot ||
        !exact_dA || !ledger_dA || !exact_dbA || !ledger_dbA) {
        free(ledger_credit);
        free(exact_dot);
        free(ledger_dot);
        free(exact_dA);
        free(ledger_dA);
        free(exact_dbA);
        free(ledger_dbA);
        return 0;
    }

    double exact_transition_energy = 0.0;
    double ledger_transition_energy = 0.0;
    double transition_dot = 0.0;
    double transition_residual_energy = 0.0;
    double a_eff_sq_sum = 0.0;
    double a_eff_abs_max = 0.0;
    uint64_t a_eff_count = 0u;

    for (int b = 0; b < batch_size; b++) {
        const size_t batch_base = (size_t)b * (size_t)seq_len;
        for (int t = 0; t < seq_len; t++) {
            const size_t pos = batch_base + (size_t)t;
            for (int i = 0; i < d_state; i++) {
                double product = 1.0;
                double lambda = 0.0;
                for (uint32_t d = 0u; d <= horizon; d++) {
                    const int ft = t + (int)d;
                    if (ft >= seq_len) break;
                    const size_t fpos = batch_base + (size_t)ft;
                    if (d > 0u) {
                        if (boundary_mask && boundary_mask[fpos]) break;
                        const double a = safe_f64(a_eff[fpos * (size_t)d_state + (size_t)i]);
                        product *= a;
                    }
                    lambda += product * safe_f64(ce_impulse[fpos * (size_t)d_state + (size_t)i]);
                }

                const double prev =
                    (t > 0 && (!boundary_mask || !boundary_mask[pos]))
                        ? safe_f64(h_cache[(pos - 1u) * (size_t)d_state + (size_t)i])
                        : 0.0;
                const size_t idx = pos * (size_t)d_state + (size_t)i;
                const double exact = safe_f64(transition_credit[idx]);
                const double ledger = lambda * prev;
                const double diff = ledger - exact;
                const double gate = safe_f64(a_gate[idx]);
                const double a_value = safe_f64(L->A[i]);
                const double exact_d = exact * a_value * gate * (1.0 - gate);
                const double ledger_d = ledger * a_value * gate * (1.0 - gate);

                ledger_credit[idx] = (float)ledger;
                exact_dot[idx] = (float)exact_d;
                ledger_dot[idx] = (float)ledger_d;
                exact_dA[i] += exact * gate;
                ledger_dA[i] += ledger * gate;
                exact_dbA[i] += exact_d;
                ledger_dbA[i] += ledger_d;

                exact_transition_energy += exact * exact;
                ledger_transition_energy += ledger * ledger;
                transition_dot += exact * ledger;
                transition_residual_energy += diff * diff;
                {
                    const double a = safe_f64(a_eff[idx]);
                    const double aa = fabs(a);
                    a_eff_sq_sum += a * a;
                    if (aa > a_eff_abs_max) a_eff_abs_max = aa;
                    a_eff_count++;
                }
            }
        }
    }

    double exact_dA_energy = 0.0;
    double ledger_dA_energy = 0.0;
    double dA_dot = 0.0;
    double dA_residual_energy = 0.0;
    double exact_dbA_energy = 0.0;
    double ledger_dbA_energy = 0.0;
    double dbA_dot = 0.0;
    double dbA_residual_energy = 0.0;
    for (int i = 0; i < d_state; i++) {
        const double ea = exact_dA[i];
        const double la = ledger_dA[i];
        const double ed = exact_dbA[i];
        const double ld = ledger_dbA[i];
        exact_dA_energy += ea * ea;
        ledger_dA_energy += la * la;
        dA_dot += ea * la;
        dA_residual_energy += (la - ea) * (la - ea);
        exact_dbA_energy += ed * ed;
        ledger_dbA_energy += ld * ld;
        dbA_dot += ed * ld;
        dbA_residual_energy += (ld - ed) * (ld - ed);
    }

    uint32_t dWA_cols = (uint32_t)d_state;
    if (dWA_cols > 64u) dWA_cols = 64u;
    if (dWA_cols == 0u) dWA_cols = 1u;
    const double dWA_scale = (double)d_state / (double)dWA_cols;
    double *x_col_energy = (double *)calloc((size_t)d_state, sizeof(double));
    uint8_t *x_col_selected = (uint8_t *)calloc((size_t)d_state, sizeof(uint8_t));
    if (!x_col_energy || !x_col_selected) {
        free(x_col_energy);
        free(x_col_selected);
        free(ledger_credit);
        free(exact_dot);
        free(ledger_dot);
        free(exact_dA);
        free(ledger_dA);
        free(exact_dbA);
        free(ledger_dbA);
        return 0;
    }
    for (int pos = 0; pos < items; pos++) {
        for (int j = 0; j < d_state; j++) {
            const double x = safe_f64(x_cv[(size_t)pos * (size_t)d_state + (size_t)j]);
            x_col_energy[j] += x * x;
        }
    }
    for (uint32_t pick = 0u; pick < dWA_cols; pick++) {
        int best = -1;
        double best_energy = -1.0;
        for (int j = 0; j < d_state; j++) {
            if (!x_col_selected[j] && x_col_energy[j] > best_energy) {
                best_energy = x_col_energy[j];
                best = j;
            }
        }
        if (best < 0) break;
        x_col_selected[best] = 1u;
    }
    double exact_dWA_energy = 0.0;
    double ledger_dWA_energy = 0.0;
    double dWA_dot = 0.0;
    double dWA_residual_energy = 0.0;
    for (int i = 0; i < d_state; i++) {
        for (int j = 0; j < d_state; j++) {
            if (!x_col_selected[j]) continue;
            double ex = 0.0;
            double le = 0.0;
            for (int pos = 0; pos < items; pos++) {
                const size_t idx = (size_t)pos * (size_t)d_state + (size_t)i;
                const double x = safe_f64(x_cv[(size_t)pos * (size_t)d_state + (size_t)j]);
                ex += safe_f64(exact_dot[idx]) * x;
                le += safe_f64(ledger_dot[idx]) * x;
            }
            exact_dWA_energy += dWA_scale * ex * ex;
            ledger_dWA_energy += dWA_scale * le * le;
            dWA_dot += dWA_scale * ex * le;
            dWA_residual_energy += dWA_scale * (le - ex) * (le - ex);
        }
    }
    free(x_col_energy);
    free(x_col_selected);

    const double transition_residual_fraction =
        relative_residual_or_zero(exact_transition_energy, transition_residual_energy);
    const double dA_residual_fraction =
        relative_residual_or_zero(exact_dA_energy, dA_residual_energy);
    const double dbA_residual_fraction =
        relative_residual_or_zero(exact_dbA_energy, dbA_residual_energy);
    const double dWA_residual_fraction =
        relative_residual_or_zero(exact_dWA_energy, dWA_residual_energy);
    const double cosine =
        (exact_transition_energy > 0.0 && ledger_transition_energy > 0.0)
            ? transition_dot / sqrt(exact_transition_energy * ledger_transition_energy)
            : 0.0;
    const double rho_rms =
        a_eff_count > 0u ? sqrt(a_eff_sq_sum / (double)a_eff_count) : 1.0;
    double observed_rho = fmax(rho_rms, a_eff_abs_max);
    if (!isfinite(observed_rho) || observed_rho < 0.0) observed_rho = 1.0;
    if (observed_rho > 0.999) observed_rho = 0.999;
    const double tail_bound_fraction =
        observed_rho < 0.999
            ? pow(observed_rho, (double)horizon + 1.0) / fmax(1.0e-9, 1.0 - observed_rho)
            : 999.0;
    const double priced_ratio =
        (double)horizon * (double)m->lagged_ce_impulse_local_jvp_fraction +
        (double)m->lagged_ce_impulse_ring_fraction +
        (double)m->lagged_ce_impulse_tail_audit_fraction +
        (double)m->lagged_ce_impulse_memory_fraction;
    const double budget = (double)m->lagged_ce_impulse_residual_budget;
    const int identifying = exact_transition_energy > 1.0e-20 ? 1 : 0;
    const int pass =
        identifying &&
        transition_residual_fraction <= budget &&
        dA_residual_fraction <= budget &&
        dbA_residual_fraction <= budget &&
        dWA_residual_fraction <= budget &&
        priced_ratio < 0.90 ? 1 : 0;
    double blend = (double)m->lagged_ce_impulse_blend;
    if (blend < 0.0) blend = 0.0;
    if (blend > 1.0) blend = 1.0;
    const int should_apply = pass && blend > 0.0 ? 1 : 0;

    if (should_apply) {
        const double keep = 1.0 - blend;
        for (size_t idx = 0; idx < scalar_count; idx++) {
            transition_credit[idx] =
                (float)(keep * safe_f64(transition_credit[idx]) +
                        blend * safe_f64(ledger_credit[idx]));
        }
    }

    m->lagged_ce_impulse_batches++;
    m->lagged_ce_impulse_layers++;
    m->lagged_ce_impulse_positions += (uint64_t)items;
    m->lagged_ce_impulse_exact_transition_energy += exact_transition_energy;
    m->lagged_ce_impulse_ledger_transition_energy += ledger_transition_energy;
    m->lagged_ce_impulse_transition_dot += transition_dot;
    m->lagged_ce_impulse_transition_residual_energy += transition_residual_energy;
    m->lagged_ce_impulse_exact_dA_energy += exact_dA_energy;
    m->lagged_ce_impulse_ledger_dA_energy += ledger_dA_energy;
    m->lagged_ce_impulse_dA_dot += dA_dot;
    m->lagged_ce_impulse_dA_residual_energy += dA_residual_energy;
    m->lagged_ce_impulse_exact_dbA_energy += exact_dbA_energy;
    m->lagged_ce_impulse_ledger_dbA_energy += ledger_dbA_energy;
    m->lagged_ce_impulse_dbA_dot += dbA_dot;
    m->lagged_ce_impulse_dbA_residual_energy += dbA_residual_energy;
    m->lagged_ce_impulse_exact_dWA_energy += exact_dWA_energy;
    m->lagged_ce_impulse_ledger_dWA_energy += ledger_dWA_energy;
    m->lagged_ce_impulse_dWA_dot += dWA_dot;
    m->lagged_ce_impulse_dWA_residual_energy += dWA_residual_energy;
    m->lagged_ce_impulse_dWA_sampled_columns = dWA_cols;
    m->lagged_ce_impulse_transition_residual_fraction =
        (float)relative_residual_or_zero(
            m->lagged_ce_impulse_exact_transition_energy,
            m->lagged_ce_impulse_transition_residual_energy);
    m->lagged_ce_impulse_dA_residual_fraction =
        (float)relative_residual_or_zero(
            m->lagged_ce_impulse_exact_dA_energy,
            m->lagged_ce_impulse_dA_residual_energy);
    m->lagged_ce_impulse_dbA_residual_fraction =
        (float)relative_residual_or_zero(
            m->lagged_ce_impulse_exact_dbA_energy,
            m->lagged_ce_impulse_dbA_residual_energy);
    m->lagged_ce_impulse_dWA_residual_fraction =
        (float)relative_residual_or_zero(
            m->lagged_ce_impulse_exact_dWA_energy,
            m->lagged_ce_impulse_dWA_residual_energy);
    m->lagged_ce_impulse_cosine =
        (float)((m->lagged_ce_impulse_exact_transition_energy > 0.0 &&
                 m->lagged_ce_impulse_ledger_transition_energy > 0.0)
            ? m->lagged_ce_impulse_transition_dot /
              sqrt(m->lagged_ce_impulse_exact_transition_energy *
                   m->lagged_ce_impulse_ledger_transition_energy)
            : 0.0);
    m->lagged_ce_impulse_priced_ratio = (float)priced_ratio;
    m->lagged_ce_impulse_observed_rho = (float)observed_rho;
    m->lagged_ce_impulse_tail_bound_fraction = (float)tail_bound_fraction;
    m->lagged_ce_impulse_identifying = identifying;
    m->lagged_ce_impulse_pass = pass;
    m->lagged_ce_impulse_last_applied = should_apply;
    m->lagged_ce_impulse_training_layers++;
    if (should_apply) {
        m->lagged_ce_impulse_applied_layers++;
    } else {
        m->lagged_ce_impulse_fallback_layers++;
    }

    free(ledger_credit);
    free(exact_dot);
    free(ledger_dot);
    free(exact_dA);
    free(ledger_dA);
    free(exact_dbA);
    free(ledger_dbA);
    return should_apply;
}

typedef struct {
    uint64_t hash;
    uint32_t count;
    uint32_t rep_pos;
    double exact_energy;
    double residual_energy;
    int used;
} R438PacketCell;

static uint64_t r438_mix_u64(uint64_t h, uint64_t x) {
    h ^= x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h *= 1099511628211ull;
    return h;
}

static int32_t r438_quantize_signature_value(float v, float scale) {
    if (!isfinite((double)v)) return 0;
    double q = floor((double)v * (double)scale + ((v >= 0.0f) ? 0.5 : -0.5));
    if (q > 32767.0) q = 32767.0;
    if (q < -32768.0) q = -32768.0;
    return (int32_t)q;
}

static uint64_t r438_state_transition_signature_hash(
    const float *h_cache, const float *a_eff, const uint8_t *boundary_mask,
    int pos, int d_state, float state_scale, float transition_scale) {
    uint64_t h = 1469598103934665603ull;
    h = r438_mix_u64(h, boundary_mask && boundary_mask[pos] ? 1ull : 0ull);
    for (int q = 0; q < 8; q++) {
        int idx = (q * d_state) / 8;
        if (idx >= d_state) idx = d_state - 1;
        const size_t off = (size_t)pos * (size_t)d_state + (size_t)idx;
        const int32_t hs = r438_quantize_signature_value(h_cache[off], state_scale);
        const int32_t as = r438_quantize_signature_value(a_eff[off], transition_scale);
        h = r438_mix_u64(h, (uint64_t)(uint32_t)hs);
        h = r438_mix_u64(h, (uint64_t)(uint32_t)as);
    }
    return h ? h : 1ull;
}

static void executable_multiplicity_packets_update(
    Model *m, const float *dA_ps, const float *h_cache, const float *a_eff,
    const uint8_t *boundary_mask, int batch_size, int seq_len, int d_state) {
    if (!m || !m->executable_multiplicity_packets_enabled || !dA_ps ||
        !h_cache || !a_eff || batch_size <= 0 || seq_len <= 0 ||
        d_state <= 0) {
        return;
    }

    const int items = batch_size * seq_len;
    uint32_t table_cap = 1u;
    while (table_cap < (uint32_t)items * 2u) table_cap <<= 1u;
    if (table_cap < 16u) table_cap = 16u;
    R438PacketCell *cells =
        (R438PacketCell *)calloc((size_t)table_cap, sizeof(R438PacketCell));
    if (!cells) return;

    uint64_t used_cells = 0u;
    const float state_scale =
        m->executable_multiplicity_state_scale > 0.0f
            ? m->executable_multiplicity_state_scale
            : 16.0f;
    const float transition_scale =
        m->executable_multiplicity_transition_scale > 0.0f
            ? m->executable_multiplicity_transition_scale
            : 64.0f;
    for (int pos = 0; pos < items; pos++) {
        const uint64_t hash =
            r438_state_transition_signature_hash(
                h_cache, a_eff, boundary_mask, pos, d_state,
                state_scale, transition_scale);
        uint32_t slot = (uint32_t)(hash & (uint64_t)(table_cap - 1u));
        while (cells[slot].used && cells[slot].hash != hash) {
            slot = (slot + 1u) & (table_cap - 1u);
        }
        if (!cells[slot].used) {
            cells[slot].used = 1;
            cells[slot].hash = hash;
            cells[slot].rep_pos = (uint32_t)pos;
            used_cells++;
        }
        R438PacketCell *cell = &cells[slot];
        cell->count++;

        double exact_energy = 0.0;
        double residual_energy = 0.0;
        const size_t base = (size_t)pos * (size_t)d_state;
        const size_t rep_base = (size_t)cell->rep_pos * (size_t)d_state;
        for (int i = 0; i < d_state; i++) {
            const double exact = safe_f64(dA_ps[base + (size_t)i]);
            const double rep = safe_f64(dA_ps[rep_base + (size_t)i]);
            const double diff = exact - rep;
            exact_energy += exact * exact;
            residual_energy += diff * diff;
        }
        cell->exact_energy += exact_energy;
        cell->residual_energy += residual_energy;
    }

    uint64_t accepted_cells = 0u;
    uint64_t accepted_positions = 0u;
    double accepted_exact_energy = 0.0;
    double accepted_residual_energy = 0.0;
    const uint32_t min_count =
        m->executable_multiplicity_min_count < 2u
            ? 2u
            : m->executable_multiplicity_min_count;
    for (uint32_t s = 0u; s < table_cap; s++) {
        if (!cells[s].used || cells[s].count < min_count) continue;
        accepted_cells++;
        accepted_positions += cells[s].count;
        accepted_exact_energy += cells[s].exact_energy;
        accepted_residual_energy += cells[s].residual_energy;
    }
    free(cells);

    const uint64_t rejected_positions =
        accepted_positions <= (uint64_t)items
            ? (uint64_t)items - accepted_positions
            : 0u;
    const double positions = (double)items;
    const double accepted_mass = positions > 0.0
        ? (double)accepted_positions / positions
        : 0.0;
    const double residual_fraction =
        relative_residual_or_zero(accepted_exact_energy, accepted_residual_energy);
    const double mean_multiplicity = accepted_cells > 0u
        ? (double)accepted_positions / (double)accepted_cells
        : 0.0;
    const double packet_work_fraction = accepted_positions > 0u
        ? (double)accepted_cells / (double)accepted_positions
        : 1.0;
    const double rejected_fraction = positions > 0.0
        ? (double)rejected_positions / positions
        : 1.0;
    const double overhead =
        (double)m->executable_multiplicity_signature_fraction +
        (double)m->executable_multiplicity_verifier_fraction +
        (double)m->executable_multiplicity_memory_fraction;
    const double priced_ratio =
        rejected_fraction +
        accepted_mass * packet_work_fraction +
        accepted_mass * residual_fraction +
        overhead;
    const int identifying = used_cells > 0u ? 1 : 0;
    const int pass =
        identifying &&
        accepted_mass >= (double)m->executable_multiplicity_min_mass &&
        residual_fraction <= (double)m->executable_multiplicity_max_residual &&
        priced_ratio < 0.90 ? 1 : 0;

    m->executable_multiplicity_batches++;
    m->executable_multiplicity_layers++;
    m->executable_multiplicity_positions += (uint64_t)items;
    m->executable_multiplicity_cells += used_cells;
    m->executable_multiplicity_accepted_cells += accepted_cells;
    m->executable_multiplicity_accepted_positions += accepted_positions;
    m->executable_multiplicity_rejected_positions += rejected_positions;
    m->executable_multiplicity_exact_energy += accepted_exact_energy;
    m->executable_multiplicity_residual_energy += accepted_residual_energy;
    m->executable_multiplicity_accepted_mass =
        m->executable_multiplicity_positions > 0u
            ? (float)((double)m->executable_multiplicity_accepted_positions /
                      (double)m->executable_multiplicity_positions)
            : 0.0f;
    m->executable_multiplicity_residual_fraction =
        (float)relative_residual_or_zero(
            m->executable_multiplicity_exact_energy,
            m->executable_multiplicity_residual_energy);
    m->executable_multiplicity_mean_multiplicity =
        m->executable_multiplicity_accepted_cells > 0u
            ? (float)((double)m->executable_multiplicity_accepted_positions /
                      (double)m->executable_multiplicity_accepted_cells)
            : 0.0f;
    m->executable_multiplicity_packet_work_fraction =
        m->executable_multiplicity_accepted_positions > 0u
            ? (float)((double)m->executable_multiplicity_accepted_cells /
                      (double)m->executable_multiplicity_accepted_positions)
            : 1.0f;
    m->executable_multiplicity_priced_ratio = (float)priced_ratio;
    m->executable_multiplicity_identifying = identifying;
    m->executable_multiplicity_pass = pass;
}

static void original_ce_credit_debt_observer_update(
    Model *m, const float *transition_credit, const float *direct_state_credit,
    const float *h_cache, const uint8_t *boundary_mask,
    int batch_size, int seq_len, int d_state) {
    if (!m || !m->original_ce_credit_debt_observer_enabled ||
        !transition_credit || !direct_state_credit || !h_cache ||
        batch_size <= 0 || seq_len <= 0 || d_state <= 0) {
        return;
    }

    const int items = batch_size * seq_len;
    double *exact_sum = (double *)calloc((size_t)d_state, sizeof(double));
    double *local_sum = (double *)calloc((size_t)d_state, sizeof(double));
    double *selected_exact_sum = (double *)calloc((size_t)d_state, sizeof(double));
    double *selected_local_sum = (double *)calloc((size_t)d_state, sizeof(double));
    double *row_scores = (double *)calloc((size_t)items, sizeof(double));
    uint8_t *selected = (uint8_t *)calloc((size_t)items, sizeof(uint8_t));
    if (!exact_sum || !local_sum || !selected_exact_sum ||
        !selected_local_sum || !row_scores || !selected) {
        free(exact_sum);
        free(local_sum);
        free(selected_exact_sum);
        free(selected_local_sum);
        free(row_scores);
        free(selected);
        return;
    }

    double row_score_sum = 0.0;
    for (int b = 0; b < batch_size; b++) {
        const size_t batch_base = (size_t)b * (size_t)seq_len;
        for (int t = 0; t < seq_len; t++) {
            const size_t pos = batch_base + (size_t)t;
            const float *exact = transition_credit + pos * (size_t)d_state;
            const float *direct = direct_state_credit + pos * (size_t)d_state;
            const float *prev = NULL;
            if (t > 0 && (!boundary_mask || !boundary_mask[pos])) {
                prev = h_cache + (batch_base + (size_t)(t - 1)) *
                       (size_t)d_state;
            }
            double score = 0.0;
            for (int i = 0; i < d_state; i++) {
                const double e = isfinite((double)exact[i]) ?
                    (double)exact[i] : 0.0;
                const double p = prev && isfinite((double)prev[i]) ?
                    (double)prev[i] : 0.0;
                const double d = isfinite((double)direct[i]) ?
                    (double)direct[i] : 0.0;
                const double l = d * p;
                exact_sum[i] += e;
                local_sum[i] += l;
                score += l * l;
            }
            row_scores[pos] = score;
            row_score_sum += score;
        }
    }

    double surface = (double)m->original_ce_credit_surface_fraction;
    if (surface <= 0.0) surface = 0.35;
    if (surface > 1.0) surface = 1.0;
    int target_selected = (int)ceil(surface * (double)items);
    if (target_selected < 1) target_selected = 1;
    if (target_selected > items) target_selected = items;

    for (int pick = 0; pick < target_selected; pick++) {
        int best = -1;
        double best_score = -1.0;
        for (int pos = 0; pos < items; pos++) {
            if (!selected[pos] && row_scores[pos] > best_score) {
                best_score = row_scores[pos];
                best = pos;
            }
        }
        if (best < 0) break;
        selected[best] = 1u;
    }

    uint64_t selected_positions = 0u;
    for (int b = 0; b < batch_size; b++) {
        const size_t batch_base = (size_t)b * (size_t)seq_len;
        for (int t = 0; t < seq_len; t++) {
            const size_t pos = batch_base + (size_t)t;
            if (!selected[pos]) continue;
            selected_positions++;
            const float *exact = transition_credit + pos * (size_t)d_state;
            const float *direct = direct_state_credit + pos * (size_t)d_state;
            const float *prev = NULL;
            if (t > 0 && (!boundary_mask || !boundary_mask[pos])) {
                prev = h_cache + (batch_base + (size_t)(t - 1)) *
                       (size_t)d_state;
            }
            for (int i = 0; i < d_state; i++) {
                const double e = isfinite((double)exact[i]) ?
                    (double)exact[i] : 0.0;
                const double p = prev && isfinite((double)prev[i]) ?
                    (double)prev[i] : 0.0;
                const double d = isfinite((double)direct[i]) ?
                    (double)direct[i] : 0.0;
                selected_exact_sum[i] += e;
                selected_local_sum[i] += d * p;
            }
        }
    }

    double exact_energy = 0.0;
    double local_energy = 0.0;
    double local_dot = 0.0;
    double local_residual_energy = 0.0;
    double selected_exact_energy = 0.0;
    double selected_local_energy = 0.0;
    double selected_dot = 0.0;
    double selected_residual_energy = 0.0;

    for (int i = 0; i < d_state; i++) {
        const double e = exact_sum[i];
        const double l = local_sum[i];
        const double r = e - l;
        exact_energy += e * e;
        local_energy += l * l;
        local_dot += e * l;
        local_residual_energy += r * r;

        const double se = selected_exact_sum[i];
        const double sl = selected_local_sum[i];
        const double sr = se - sl;
        selected_exact_energy += se * se;
        selected_local_energy += sl * sl;
        selected_dot += se * sl;
        selected_residual_energy += sr * sr;
    }

    free(exact_sum);
    free(local_sum);
    free(selected_exact_sum);
    free(selected_local_sum);
    free(row_scores);
    free(selected);

    m->original_ce_credit_gate_batches++;
    m->original_ce_credit_gate_layers++;
    m->original_ce_credit_gate_positions += (uint64_t)items;
    m->original_ce_credit_gate_selected_positions += selected_positions;
    m->original_ce_credit_gate_exact_energy += exact_energy;
    m->original_ce_credit_gate_local_energy += local_energy;
    m->original_ce_credit_gate_local_dot += local_dot;
    m->original_ce_credit_gate_local_residual_energy += local_residual_energy;
    m->original_ce_credit_gate_selected_exact_energy += selected_exact_energy;
    m->original_ce_credit_gate_selected_local_energy += selected_local_energy;
    m->original_ce_credit_gate_selected_dot += selected_dot;
    m->original_ce_credit_gate_selected_residual_energy += selected_residual_energy;

    const double exact_total = m->original_ce_credit_gate_exact_energy;
    const double selected_exact_total =
        m->original_ce_credit_gate_selected_exact_energy;
    const double positions_total =
        (double)m->original_ce_credit_gate_positions;
    const double selected_total =
        (double)m->original_ce_credit_gate_selected_positions;
    const double local_progress =
        exact_total > 0.0 ? m->original_ce_credit_gate_local_dot / exact_total : 0.0;
    const double local_residual =
        exact_total > 0.0
            ? sqrt(m->original_ce_credit_gate_local_residual_energy / exact_total)
            : 1.0;
    const double selected_progress =
        selected_exact_total > 0.0
            ? m->original_ce_credit_gate_selected_dot / selected_exact_total
            : 0.0;
    const double selected_residual =
        selected_exact_total > 0.0
            ? sqrt(m->original_ce_credit_gate_selected_residual_energy /
                   selected_exact_total)
            : 1.0;
    const double selected_fraction =
        positions_total > 0.0 ? selected_total / positions_total : 0.0;
    const double local_work = (double)m->original_ce_credit_local_work_multiplier;
    const double surface_work = selected_fraction * local_work;
    const double debt_price = (double)m->original_ce_credit_debt_price;
    const double verify = (double)m->original_ce_credit_verifier_fraction;
    const double overhead = (double)m->original_ce_credit_overhead;
    const double recurrent_share =
        fmin(1.0, fmax(0.0, (double)m->original_ce_credit_recurrent_backward_share));
    const double local_priced =
        local_progress > 1.0e-12
            ? (local_work + verify + overhead + debt_price * local_residual) /
              local_progress
            : INFINITY;
    const double surface_priced =
        selected_progress > 1.0e-12
            ? (surface_work + verify + overhead + debt_price * selected_residual) /
              selected_progress
            : INFINITY;
    const double local_whole =
        1.0 / fmax(1.0e-12,
            1.0 - recurrent_share * fmax(0.0, 1.0 - local_priced));
    const double surface_whole =
        1.0 / fmax(1.0e-12,
            1.0 - recurrent_share * fmax(0.0, 1.0 - surface_priced));
    const int identifying = (exact_total > 0.0 && row_score_sum >= 0.0) ? 1 : 0;

    m->original_ce_credit_gate_selected_fraction = (float)selected_fraction;
    m->original_ce_credit_gate_local_progress_ratio = (float)local_progress;
    m->original_ce_credit_gate_local_residual_fraction = (float)local_residual;
    m->original_ce_credit_gate_local_priced_ratio = (float)local_priced;
    m->original_ce_credit_gate_local_whole_step_speedup = (float)local_whole;
    m->original_ce_credit_gate_surface_progress_ratio = (float)selected_progress;
    m->original_ce_credit_gate_surface_residual_fraction = (float)selected_residual;
    m->original_ce_credit_gate_surface_priced_ratio = (float)surface_priced;
    m->original_ce_credit_gate_surface_whole_step_speedup = (float)surface_whole;
    m->original_ce_credit_gate_identifying = identifying;
    m->original_ce_credit_gate_local_pass =
        (identifying &&
         local_progress >= (double)m->original_ce_credit_min_progress &&
         local_residual <= (double)m->original_ce_credit_max_residual &&
         local_whole >= (double)m->original_ce_credit_min_whole_step_speedup)
            ? 1 : 0;
    m->original_ce_credit_gate_surface_pass =
        (identifying &&
         selected_progress >= (double)m->original_ce_credit_min_progress &&
         selected_residual <= (double)m->original_ce_credit_max_residual &&
         surface_whole >= (double)m->original_ce_credit_min_whole_step_speedup)
            ? 1 : 0;
}

static void fp_interface_gap_probe_capture_layer(
    GpuPipeline *p, Model *m, int step, int layer,
    int batch_size, int seq_len, int d) {
    if (!p || !m || !m->fp_interface_gap_probe_enabled ||
        layer < 0 || layer >= V12_N_LAYERS ||
        batch_size <= 0 || seq_len <= 0 || d <= 0 ||
        !VK_BUFFER_VALID(p->residual[layer]) ||
        !VK_BUFFER_VALID(p->residual[layer + 1]) ||
        !VK_BUFFER_VALID(p->bwd_d_h_in)) {
        return;
    }

    const size_t n = (size_t)batch_size * (size_t)seq_len * (size_t)d;
    const size_t bytes = n * sizeof(float);
    const int ds = p->d_state;
    const size_t y_n = (size_t)batch_size * (size_t)seq_len * (size_t)ds;
    const size_t y_bytes = y_n * sizeof(float);
    float *r_in = (float *)malloc(bytes);
    float *r_out = (float *)malloc(bytes);
    float *grad = (float *)malloc(bytes);
    float *x_proj = VK_BUFFER_VALID(p->layers[layer].fwd_x_proj)
        ? (float *)malloc(y_bytes)
        : NULL;
    float *x_cv = VK_BUFFER_VALID(p->layers[layer].fwd_x_cv)
        ? (float *)malloc(y_bytes)
        : NULL;
    float *b_eff = VK_BUFFER_VALID(p->layers[layer].fwd_b_eff)
        ? (float *)malloc(y_bytes)
        : NULL;
    float *h_cache = VK_BUFFER_VALID(p->layers[layer].fwd_h_cache)
        ? (float *)malloc(y_bytes)
        : NULL;
    float *y_ssm = VK_BUFFER_VALID(p->layers[layer].fwd_y_ssm)
        ? (float *)malloc(y_bytes)
        : NULL;
    if (!r_in || !r_out || !grad) {
        free(r_in);
        free(r_out);
        free(grad);
        free(x_proj);
        free(x_cv);
        free(b_eff);
        free(h_cache);
        free(y_ssm);
        return;
    }

    vk_download(p->ctx, r_in, p->residual[layer], bytes);
    vk_download(p->ctx, r_out, p->residual[layer + 1], bytes);
    vk_download(p->ctx, grad, p->bwd_d_h_in, bytes);
    if (x_proj) {
        vk_download(p->ctx, x_proj, p->layers[layer].fwd_x_proj, y_bytes);
    }
    if (x_cv) {
        vk_download(p->ctx, x_cv, p->layers[layer].fwd_x_cv, y_bytes);
    }
    if (b_eff) {
        vk_download(p->ctx, b_eff, p->layers[layer].fwd_b_eff, y_bytes);
    }
    if (h_cache) {
        vk_download(p->ctx, h_cache, p->layers[layer].fwd_h_cache, y_bytes);
    }
    if (y_ssm) {
        vk_download(p->ctx, y_ssm, p->layers[layer].fwd_y_ssm, y_bytes);
    }

    double input_norm2 = 0.0;
    double delta_norm2 = 0.0;
    double grad_norm2 = 0.0;
    double dot = 0.0;
    uint64_t finite_count = 0;
    for (size_t i = 0; i < n; i++) {
        const double in_v = (double)r_in[i];
        const double delta = (double)r_out[i] - in_v;
        const double g = (double)grad[i];
        if (!isfinite(in_v) || !isfinite(delta) || !isfinite(g)) continue;
        input_norm2 += in_v * in_v;
        delta_norm2 += delta * delta;
        grad_norm2 += g * g;
        dot += delta * g;
        finite_count++;
    }
    double y_norm2 = 0.0;
    double x_proj_norm2 = 0.0;
    double x_cv_norm2 = 0.0;
    double b_eff_norm2 = 0.0;
    double h_norm2 = 0.0;
    uint64_t y_finite_count = 0;
    if (y_ssm) {
        for (size_t i = 0; i < y_n; i++) {
            const double y = (double)y_ssm[i];
            if (!isfinite(y)) continue;
            y_norm2 += y * y;
            y_finite_count++;
        }
    }
    uint64_t x_proj_finite_count = 0;
    if (x_proj) {
        for (size_t i = 0; i < y_n; i++) {
            const double v = (double)x_proj[i];
            if (!isfinite(v)) continue;
            x_proj_norm2 += v * v;
            x_proj_finite_count++;
        }
    }
    uint64_t x_cv_finite_count = 0;
    if (x_cv) {
        for (size_t i = 0; i < y_n; i++) {
            const double v = (double)x_cv[i];
            if (!isfinite(v)) continue;
            x_cv_norm2 += v * v;
            x_cv_finite_count++;
        }
    }
    uint64_t b_eff_finite_count = 0;
    if (b_eff) {
        for (size_t i = 0; i < y_n; i++) {
            const double v = (double)b_eff[i];
            if (!isfinite(v)) continue;
            b_eff_norm2 += v * v;
            b_eff_finite_count++;
        }
    }
    uint64_t h_finite_count = 0;
    if (h_cache) {
        for (size_t i = 0; i < y_n; i++) {
            const double v = (double)h_cache[i];
            if (!isfinite(v)) continue;
            h_norm2 += v * v;
            h_finite_count++;
        }
    }

    const double denom = (double)(finite_count > 0 ? finite_count : 1);
    const double y_denom = (double)(y_finite_count > 0 ? y_finite_count : 1);
    const double input_rms = sqrt(input_norm2 / denom);
    const double delta_rms = sqrt(delta_norm2 / denom);
    const double y_ssm_rms = sqrt(y_norm2 / y_denom);
    const double x_proj_rms =
        sqrt(x_proj_norm2 / (double)(x_proj_finite_count > 0 ? x_proj_finite_count : 1));
    const double x_cv_rms =
        sqrt(x_cv_norm2 / (double)(x_cv_finite_count > 0 ? x_cv_finite_count : 1));
    const double b_eff_rms =
        sqrt(b_eff_norm2 / (double)(b_eff_finite_count > 0 ? b_eff_finite_count : 1));
    const double h_rms =
        sqrt(h_norm2 / (double)(h_finite_count > 0 ? h_finite_count : 1));
    const double grad_rms = sqrt(grad_norm2 / denom);
    const double effect_ratio =
        input_norm2 > 1.0e-30 ? sqrt(delta_norm2 / input_norm2) : 0.0;
    const double cosine =
        (delta_norm2 > 1.0e-30 && grad_norm2 > 1.0e-30)
            ? dot / sqrt(delta_norm2 * grad_norm2)
            : 0.0;
    const double abs_projection = fabs(cosine);
    const double descent_alignment = -cosine;
    const double effect_eps = m->fp_interface_gap_effect_epsilon > 0.0f
        ? (double)m->fp_interface_gap_effect_epsilon
        : 0.05;
    const int identifying = (finite_count > 0 && input_norm2 > 1.0e-30 &&
                             grad_norm2 > 1.0e-30) ? 1 : 0;
    const int near_identity =
        (identifying && effect_ratio < effect_eps && grad_rms > 1.0e-12) ? 1 : 0;

    m->fp_interface_gap_effect_ratio[layer] = (float)effect_ratio;
    m->fp_interface_gap_input_rms[layer] = (float)input_rms;
    m->fp_interface_gap_delta_rms[layer] = (float)delta_rms;
    m->fp_interface_gap_y_ssm_rms[layer] = (float)y_ssm_rms;
    m->fp_interface_gap_grad_rms[layer] = (float)grad_rms;
    m->fp_interface_gap_abs_projection[layer] = (float)abs_projection;
    m->fp_interface_gap_descent_alignment[layer] = (float)descent_alignment;
    m->fp_interface_gap_wout_nnz[layer] =
        (uint64_t)m->layers[layer].csr_out.total_pos +
        (uint64_t)m->layers[layer].csr_out.total_neg;
    m->fp_interface_gap_near_identity[layer] = near_identity;

    m->fp_interface_gap_probe_layers++;
    if (near_identity) m->fp_interface_gap_near_identity_layers++;
    m->fp_interface_gap_probe_identifying =
        (m->fp_interface_gap_probe_identifying || identifying) ? 1 : 0;

    double sum_effect = 0.0;
    double sum_projection = 0.0;
    double min_effect = INFINITY;
    double max_grad = 0.0;
    uint64_t seen = 0;
    for (int k = 0; k < V12_N_LAYERS; k++) {
        const double er = (double)m->fp_interface_gap_effect_ratio[k];
        const double ap = (double)m->fp_interface_gap_abs_projection[k];
        const double gr = (double)m->fp_interface_gap_grad_rms[k];
        if (er <= 0.0 && ap <= 0.0 && gr <= 0.0) continue;
        sum_effect += er;
        sum_projection += ap;
        if (er < min_effect) min_effect = er;
        if (gr > max_grad) max_grad = gr;
        seen++;
    }
    m->fp_interface_gap_mean_effect_ratio =
        seen > 0 ? (float)(sum_effect / (double)seen) : 0.0f;
    m->fp_interface_gap_mean_abs_projection =
        seen > 0 ? (float)(sum_projection / (double)seen) : 0.0f;
    m->fp_interface_gap_min_effect_ratio =
        isfinite(min_effect) ? (float)min_effect : 0.0f;
    m->fp_interface_gap_max_grad_rms = (float)max_grad;

    printf("[FP_INTERFACE_GAP_PROBE] seq_idx=%d layer=%d input_rms=%.9g x_proj_rms=%.9g x_cv_rms=%.9g b_eff_rms=%.9g h_rms=%.9g delta_rms=%.9g y_ssm_rms=%.9g wout_nnz=%llu effect_ratio=%.9g grad_rms=%.9g cosine=%.9g abs_projection=%.9g descent_alignment=%.9g near_identity=%d identifying=%d finite=%llu mode=current-v12-vulkan\n",
           step,
           layer,
           input_rms,
           x_proj_rms,
           x_cv_rms,
           b_eff_rms,
           h_rms,
           delta_rms,
           y_ssm_rms,
           (unsigned long long)m->fp_interface_gap_wout_nnz[layer],
           effect_ratio,
           grad_rms,
           cosine,
           abs_projection,
           descent_alignment,
           near_identity,
           identifying,
           (unsigned long long)finite_count);

    free(r_in);
    free(r_out);
    free(grad);
    free(x_proj);
    free(x_cv);
    free(b_eff);
    free(h_cache);
    free(y_ssm);
}

static void macro_transcript_gate_update(
    Model *m, const uint16_t *tokens, int batch_size, int seq_len) {
    if (!m || !m->macro_transcript_gate_enabled || !tokens ||
        batch_size <= 0 || seq_len <= 0) {
        return;
    }

    uint64_t positions = 0;
    uint64_t ops = 0;
    uint64_t pair_macros = 0;
    uint64_t singletons = 0;
    const uint32_t min_count = m->macro_transcript_min_count
        ? m->macro_transcript_min_count
        : 2u;

    for (int b = 0; b < batch_size; b++) {
        const uint16_t *tok_b = tokens + (size_t)b * (size_t)seq_len;
        int t = 0;
        positions += (uint64_t)seq_len;
        while (t < seq_len) {
            if (t + 1 < seq_len &&
                macro_transcript_pair_count(m, tok_b[t], tok_b[t + 1]) >= min_count) {
                pair_macros++;
                ops++;
                t += 2;
            } else {
                singletons++;
                ops++;
                t++;
            }
        }
    }

    m->macro_transcript_gate_batches++;
    m->macro_transcript_gate_positions += positions;
    m->macro_transcript_gate_ops += ops;
    m->macro_transcript_gate_pair_macros += pair_macros;
    m->macro_transcript_gate_singletons += singletons;

    const double total_ops = (double)m->macro_transcript_gate_ops;
    const double total_positions = (double)m->macro_transcript_gate_positions;
    const double length_ratio = total_positions > 0.0 ? total_ops / total_positions : 1.0;
    const double pair_fraction = total_ops > 0.0
        ? (double)m->macro_transcript_gate_pair_macros / total_ops
        : 0.0;
    const double priced =
        length_ratio * (double)m->macro_transcript_hardness +
        (double)m->macro_transcript_codec_cost +
        (double)m->macro_transcript_decode_cost;

    m->macro_transcript_gate_length_ratio = (float)length_ratio;
    m->macro_transcript_gate_pair_fraction = (float)pair_fraction;
    m->macro_transcript_gate_priced_ratio = (float)priced;
}

static void reversible_deterministic_expansion_gate_update(
    Model *m, const uint16_t *tokens, int batch_size, int seq_len) {
    if (!m || !m->reversible_deterministic_expansion_gate_enabled || !tokens ||
        batch_size <= 0 || seq_len <= 0) {
        return;
    }

    uint64_t positions = 0;
    uint64_t covered = 0;
    uint64_t spans = 0;
    uint64_t coord_units = 0;
    uint64_t inverse_failures = 0;
    const uint32_t min_run = m->reversible_deterministic_expansion_min_run > 1u
        ? m->reversible_deterministic_expansion_min_run
        : 3u;

    for (int b = 0; b < batch_size; b++) {
        const uint16_t *tok_b = tokens + (size_t)b * (size_t)seq_len;
        int t = 0;
        while (t < seq_len) {
            const uint16_t tok = tok_b[t];
            int run = 1;
            while (t + run < seq_len && tok_b[t + run] == tok) run++;
            positions += (uint64_t)run;
            if (run >= (int)min_run && reversible_deterministic_expansion_token(tok)) {
                int inverse_ok = 1;
                for (int j = 0; j < run; j++) {
                    if (tok_b[t + j] != tok) {
                        inverse_ok = 0;
                        break;
                    }
                }
                if (inverse_ok) {
                    covered += (uint64_t)(run - 1);
                    spans++;
                    coord_units++;
                } else {
                    inverse_failures++;
                }
            }
            t += run;
        }
    }

    m->reversible_deterministic_expansion_gate_batches++;
    m->reversible_deterministic_expansion_gate_positions += positions;
    m->reversible_deterministic_expansion_gate_covered += covered;
    m->reversible_deterministic_expansion_gate_spans += spans;
    m->reversible_deterministic_expansion_gate_coord_units += coord_units;
    m->reversible_deterministic_expansion_inverse_failures += inverse_failures;

    const double total = (double)m->reversible_deterministic_expansion_gate_positions;
    const double lambda = total > 0.0
        ? (double)m->reversible_deterministic_expansion_gate_covered / total
        : 0.0;
    const double eta = total > 0.0
        ? (double)m->reversible_deterministic_expansion_gate_coord_units / total
        : 0.0;
    const double priced = (double)m->reversible_deterministic_expansion_overhead +
        (1.0 - lambda + eta) *
            (1.0 + (double)m->reversible_deterministic_expansion_noise_penalty);
    m->reversible_deterministic_expansion_gate_lambda = (float)lambda;
    m->reversible_deterministic_expansion_gate_eta = (float)eta;
    m->reversible_deterministic_expansion_gate_priced_ratio = (float)priced;
    m->reversible_deterministic_expansion_inverse_audit =
        (m->reversible_deterministic_expansion_inverse_failures == 0u) ? 1 : 0;
}

