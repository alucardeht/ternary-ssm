#include "gpu_pipeline_internal.h"

#ifdef V12_HAS_VULKAN

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

void gpu_pipeline_destroy(GpuPipeline *p) {
    if (!p) return;

    VulkanContext *ctx = p->ctx;

    for (int k = 0; k < V12_N_LAYERS; k++) {
        /* Ternary W_in / W_out shadow + factored moments */
        if (VK_BUFFER_VALID(p->opt_layers[k].shadow_w_in))  vk_free(ctx, p->opt_layers[k].shadow_w_in);
        if (VK_BUFFER_VALID(p->opt_layers[k].m_w_in_row))   vk_free(ctx, p->opt_layers[k].m_w_in_row);
        if (VK_BUFFER_VALID(p->opt_layers[k].v_w_in_row))   vk_free(ctx, p->opt_layers[k].v_w_in_row);
        if (VK_BUFFER_VALID(p->opt_layers[k].m_w_in_col))   vk_free(ctx, p->opt_layers[k].m_w_in_col);
        if (VK_BUFFER_VALID(p->opt_layers[k].v_w_in_col))   vk_free(ctx, p->opt_layers[k].v_w_in_col);
        if (VK_BUFFER_VALID(p->opt_layers[k].shadow_w_out)) vk_free(ctx, p->opt_layers[k].shadow_w_out);
        if (VK_BUFFER_VALID(p->opt_layers[k].m_w_out_row))  vk_free(ctx, p->opt_layers[k].m_w_out_row);
        if (VK_BUFFER_VALID(p->opt_layers[k].v_w_out_row))  vk_free(ctx, p->opt_layers[k].v_w_out_row);
        if (VK_BUFFER_VALID(p->opt_layers[k].m_w_out_col))  vk_free(ctx, p->opt_layers[k].m_w_out_col);
        if (VK_BUFFER_VALID(p->opt_layers[k].v_w_out_col))  vk_free(ctx, p->opt_layers[k].v_w_out_col);

        /* Factored-Adam scratch: W_in */
        if (VK_BUFFER_VALID(p->opt_layers[k].grad_accum_in)) vk_free(ctx, p->opt_layers[k].grad_accum_in);
        if (VK_BUFFER_VALID(p->opt_layers[k].sampled_w_in))  vk_free(ctx, p->opt_layers[k].sampled_w_in);
        if (VK_BUFFER_VALID(p->opt_layers[k].g_row_in))      vk_free(ctx, p->opt_layers[k].g_row_in);
        if (VK_BUFFER_VALID(p->opt_layers[k].g_col_in))      vk_free(ctx, p->opt_layers[k].g_col_in);
        if (VK_BUFFER_VALID(p->opt_layers[k].gq_row_in))     vk_free(ctx, p->opt_layers[k].gq_row_in);
        if (VK_BUFFER_VALID(p->opt_layers[k].gq_col_in))     vk_free(ctx, p->opt_layers[k].gq_col_in);
        if (VK_BUFFER_VALID(p->opt_layers[k].cnt_row_in))    vk_free(ctx, p->opt_layers[k].cnt_row_in);
        if (VK_BUFFER_VALID(p->opt_layers[k].cnt_col_in))    vk_free(ctx, p->opt_layers[k].cnt_col_in);
        if (VK_BUFFER_VALID(p->opt_layers[k].mean_v_in))     vk_free(ctx, p->opt_layers[k].mean_v_in);

        /* Factored-Adam scratch: W_out */
        if (VK_BUFFER_VALID(p->opt_layers[k].grad_accum_out)) vk_free(ctx, p->opt_layers[k].grad_accum_out);
        if (VK_BUFFER_VALID(p->opt_layers[k].sampled_w_out))  vk_free(ctx, p->opt_layers[k].sampled_w_out);
        if (VK_BUFFER_VALID(p->opt_layers[k].g_row_out))      vk_free(ctx, p->opt_layers[k].g_row_out);
        if (VK_BUFFER_VALID(p->opt_layers[k].g_col_out))      vk_free(ctx, p->opt_layers[k].g_col_out);
        if (VK_BUFFER_VALID(p->opt_layers[k].gq_row_out))     vk_free(ctx, p->opt_layers[k].gq_row_out);
        if (VK_BUFFER_VALID(p->opt_layers[k].gq_col_out))     vk_free(ctx, p->opt_layers[k].gq_col_out);
        if (VK_BUFFER_VALID(p->opt_layers[k].cnt_row_out))    vk_free(ctx, p->opt_layers[k].cnt_row_out);
        if (VK_BUFFER_VALID(p->opt_layers[k].cnt_col_out))    vk_free(ctx, p->opt_layers[k].cnt_col_out);
        if (VK_BUFFER_VALID(p->opt_layers[k].mean_v_out))     vk_free(ctx, p->opt_layers[k].mean_v_out);

        if (VK_BUFFER_VALID(p->layers[k].tssi_score_moments)) vk_free(ctx, p->layers[k].tssi_score_moments);
    }

    /* Free CE loss buffers */
    if (VK_BUFFER_VALID(p->ce_row_ptr_pos)) vk_free(ctx, p->ce_row_ptr_pos);
    if (VK_BUFFER_VALID(p->ce_pos_indices)) vk_free(ctx, p->ce_pos_indices);
    if (VK_BUFFER_VALID(p->ce_row_ptr_neg)) vk_free(ctx, p->ce_row_ptr_neg);
    if (VK_BUFFER_VALID(p->ce_neg_indices)) vk_free(ctx, p->ce_neg_indices);
    if (VK_BUFFER_VALID(p->ce_unigram_cdf)) vk_free(ctx, p->ce_unigram_cdf);
    if (VK_BUFFER_VALID(p->ce_log_unigram)) vk_free(ctx, p->ce_log_unigram);
    if (VK_BUFFER_VALID(p->ce_token_freq)) vk_free(ctx, p->ce_token_freq);
    if (VK_BUFFER_VALID(p->ce_sample_ids))  vk_free(ctx, p->ce_sample_ids);
    if (VK_BUFFER_VALID(p->ce_d_logits))    vk_free(ctx, p->ce_d_logits);
    if (VK_BUFFER_VALID(p->ce_ema_logits))  vk_free(ctx, p->ce_ema_logits);
    if (VK_BUFFER_VALID(p->ce_loss))        vk_free(ctx, p->ce_loss);
    if (VK_BUFFER_VALID(p->ce_base_loss))   vk_free(ctx, p->ce_base_loss);
    if (VK_BUFFER_VALID(p->ce_d_norm_w))    vk_free(ctx, p->ce_d_norm_w);
    if (VK_BUFFER_VALID(p->ce_d_norm_b))    vk_free(ctx, p->ce_d_norm_b);
    if (VK_BUFFER_VALID(p->ce_dlm_head))    vk_free(ctx, p->ce_dlm_head);
    if (VK_BUFFER_VALID(p->prior_ctx_counts))  vk_free(ctx, p->prior_ctx_counts);
    if (VK_BUFFER_VALID(p->prior_pair_keys))   vk_free(ctx, p->prior_pair_keys);
    if (VK_BUFFER_VALID(p->prior_pair_counts)) vk_free(ctx, p->prior_pair_counts);
    if (VK_BUFFER_VALID(p->prior_unigram))     vk_free(ctx, p->prior_unigram);
    if (VK_BUFFER_VALID(p->program_kernel_targets)) vk_free(ctx, p->program_kernel_targets);
    if (VK_BUFFER_VALID(p->program_kernel_counts))  vk_free(ctx, p->program_kernel_counts);
    if (VK_BUFFER_VALID(p->state_atlas_keys))    vk_free(ctx, p->state_atlas_keys);
    if (VK_BUFFER_VALID(p->state_atlas_targets)) vk_free(ctx, p->state_atlas_targets);
    if (VK_BUFFER_VALID(p->state_atlas_counts))  vk_free(ctx, p->state_atlas_counts);

    if (VK_BUFFER_VALID(p->csr_count_pos)) vk_free(ctx, p->csr_count_pos);
    if (VK_BUFFER_VALID(p->csr_count_neg)) vk_free(ctx, p->csr_count_neg);
    if (VK_BUFFER_VALID(p->flip_counts))   vk_free(ctx, p->flip_counts);
    if (VK_BUFFER_VALID(p->evidence_rate_stats)) vk_free(ctx, p->evidence_rate_stats);

    /* Free lm_head optimizer state */
    if (VK_BUFFER_VALID(p->opt_shadow_lm_head))  vk_free(ctx, p->opt_shadow_lm_head);
    if (VK_BUFFER_VALID(p->opt_m_lm_head_row))   vk_free(ctx, p->opt_m_lm_head_row);
    if (VK_BUFFER_VALID(p->opt_v_lm_head_row))   vk_free(ctx, p->opt_v_lm_head_row);
    if (VK_BUFFER_VALID(p->opt_m_lm_head_col))   vk_free(ctx, p->opt_m_lm_head_col);
    if (VK_BUFFER_VALID(p->opt_v_lm_head_col))   vk_free(ctx, p->opt_v_lm_head_col);
    if (VK_BUFFER_VALID(p->opt_dlm_head))         vk_free(ctx, p->opt_dlm_head);
    if (VK_BUFFER_VALID(p->opt_sampled_lm_head))  vk_free(ctx, p->opt_sampled_lm_head);
    if (VK_BUFFER_VALID(p->opt_g_row_lm))         vk_free(ctx, p->opt_g_row_lm);
    if (VK_BUFFER_VALID(p->opt_g_col_lm))         vk_free(ctx, p->opt_g_col_lm);
    if (VK_BUFFER_VALID(p->opt_gq_row_lm))        vk_free(ctx, p->opt_gq_row_lm);
    if (VK_BUFFER_VALID(p->opt_gq_col_lm))        vk_free(ctx, p->opt_gq_col_lm);
    if (VK_BUFFER_VALID(p->opt_cnt_row_lm))       vk_free(ctx, p->opt_cnt_row_lm);
    if (VK_BUFFER_VALID(p->opt_cnt_col_lm))       vk_free(ctx, p->opt_cnt_col_lm);
    if (VK_BUFFER_VALID(p->opt_mean_v_lm))        vk_free(ctx, p->opt_mean_v_lm);
    if (VK_BUFFER_VALID(p->opt_vocab_to_hot_lm))  vk_free(ctx, p->opt_vocab_to_hot_lm);

    /* Free w_pred optimizer state */
    if (VK_BUFFER_VALID(p->opt_shadow_w_pred))  vk_free(ctx, p->opt_shadow_w_pred);
    if (VK_BUFFER_VALID(p->opt_m_w_pred_row))   vk_free(ctx, p->opt_m_w_pred_row);
    if (VK_BUFFER_VALID(p->opt_v_w_pred_row))   vk_free(ctx, p->opt_v_w_pred_row);
    if (VK_BUFFER_VALID(p->opt_m_w_pred_col))   vk_free(ctx, p->opt_m_w_pred_col);
    if (VK_BUFFER_VALID(p->opt_v_w_pred_col))   vk_free(ctx, p->opt_v_w_pred_col);
    if (VK_BUFFER_VALID(p->opt_dw_pred))        vk_free(ctx, p->opt_dw_pred);
    if (VK_BUFFER_VALID(p->opt_dlm_w_pred))     vk_free(ctx, p->opt_dlm_w_pred);
    if (VK_BUFFER_VALID(p->opt_sampled_w_pred)) vk_free(ctx, p->opt_sampled_w_pred);
    if (VK_BUFFER_VALID(p->opt_g_row_wp))       vk_free(ctx, p->opt_g_row_wp);
    if (VK_BUFFER_VALID(p->opt_g_col_wp))       vk_free(ctx, p->opt_g_col_wp);
    if (VK_BUFFER_VALID(p->opt_gq_row_wp))      vk_free(ctx, p->opt_gq_row_wp);
    if (VK_BUFFER_VALID(p->opt_gq_col_wp))      vk_free(ctx, p->opt_gq_col_wp);
    if (VK_BUFFER_VALID(p->opt_cnt_row_wp))     vk_free(ctx, p->opt_cnt_row_wp);
    if (VK_BUFFER_VALID(p->opt_cnt_col_wp))     vk_free(ctx, p->opt_cnt_col_wp);
    if (VK_BUFFER_VALID(p->opt_mean_v_wp))      vk_free(ctx, p->opt_mean_v_wp);

    /* Free CPU staging */
    free(p->online_tssi_train_state_mask);
    free(p->staging_dw);
    free(p->pc_neg_buf);
    free(p->pc_res_buf);
    free(p->pc_dw_buf);

    /* Free SSM scan transients */
    if (VK_BUFFER_VALID(p->ssm_a))   vk_free(ctx, p->ssm_a);
    if (VK_BUFFER_VALID(p->ssm_b))   vk_free(ctx, p->ssm_b);
    if (VK_BUFFER_VALID(p->ssm_x))   vk_free(ctx, p->ssm_x);
    if (VK_BUFFER_VALID(p->ssm_h))   vk_free(ctx, p->ssm_h);
    if (VK_BUFFER_VALID(p->ssm_cbl)) vk_free(ctx, p->ssm_cbl);
    if (VK_BUFFER_VALID(p->ssm_u))   vk_free(ctx, p->ssm_u);
    if (VK_BUFFER_VALID(p->ssm_v))   vk_free(ctx, p->ssm_v);
    if (VK_BUFFER_VALID(p->ef03_ring_sums)) vk_free(ctx, p->ef03_ring_sums);

    /* VSSBR: Free shared forward activation buffers (referenced in layer 0) */
    if (p->n_layers > 0) {
        if (VK_BUFFER_VALID(p->layers[0].fwd_x_proj))   vk_free(ctx, p->layers[0].fwd_x_proj);
        if (VK_BUFFER_VALID(p->layers[0].fwd_x_gated))  vk_free(ctx, p->layers[0].fwd_x_gated);
        if (VK_BUFFER_VALID(p->layers[0].fwd_mask_val)) vk_free(ctx, p->layers[0].fwd_mask_val);
        if (VK_BUFFER_VALID(p->layers[0].fwd_x_cv))     vk_free(ctx, p->layers[0].fwd_x_cv);
        if (VK_BUFFER_VALID(p->layers[0].fwd_b_eff))    vk_free(ctx, p->layers[0].fwd_b_eff);
        if (VK_BUFFER_VALID(p->layers[0].fwd_a_buf))    vk_free(ctx, p->layers[0].fwd_a_buf);
        if (VK_BUFFER_VALID(p->layers[0].fwd_h_cache))  vk_free(ctx, p->layers[0].fwd_h_cache);
        if (VK_BUFFER_VALID(p->layers[0].fwd_y_ssm))    vk_free(ctx, p->layers[0].fwd_y_ssm);
    }

    /* PRBLT ring buffer cleanup */
#if V12_PRBLT_ENABLED
    for (int r = 0; r < 2; r++) {
        if (VK_TRANSFER_VALID(p->prblt_ring[r].transfer_handle)) {
            vk_wait_async_transfer(ctx, p->prblt_ring[r].transfer_handle);
            vk_destroy_async_transfer(ctx, p->prblt_ring[r].transfer_handle);
        }
        if (VK_BUFFER_VALID(p->prblt_ring[r].staging_buf)) {
            vk_unmap(ctx, p->prblt_ring[r].staging_buf);
            vk_free(ctx, p->prblt_ring[r].staging_buf);
        }
        prblt_free_max_csr(ctx, &p->prblt_ring[r].gpu_csr_in);
        prblt_free_max_csr(ctx, &p->prblt_ring[r].gpu_csr_out);
        prblt_free_max_csr(ctx, &p->prblt_ring[r].gpu_csc_in);
        prblt_free_max_csr(ctx, &p->prblt_ring[r].gpu_csc_out);
    }
#endif
    for (int i = 0; i < 2; i++) {
        if (VK_BUFFER_VALID(p->prblt_ring[i].shadow_w_in)) {
            vk_free(ctx, p->prblt_ring[i].shadow_w_in);
        }
        if (VK_BUFFER_VALID(p->prblt_ring[i].shadow_w_out)) {
            vk_free(ctx, p->prblt_ring[i].shadow_w_out);
        }
        free_vk_csr(ctx, &p->prblt_ring[i].gpu_csr_in);
        free_vk_csr(ctx, &p->prblt_ring[i].gpu_csr_out);
        free_vk_csr(ctx, &p->prblt_ring[i].gpu_csc_in);
        free_vk_csr(ctx, &p->prblt_ring[i].gpu_csc_out);
    }

    vk_print_stats(ctx);
    free(p);
}


void gpu_pipeline_print_stats(GpuPipeline *p) {
    if (!p) return;
    vk_print_stats(p->ctx);
}


#else // !V12_HAS_VULKAN

GpuPipeline *gpu_pipeline_create(VulkanContext *ctx, Model *m, int seq_len, int batch_size) {
    (void)ctx; (void)m; (void)seq_len; (void)batch_size;
    return NULL;
}

void gpu_pipeline_destroy(GpuPipeline *p) { (void)p; }

void gpu_pipeline_print_stats(GpuPipeline *p)                                     { (void)p; }


#endif // V12_HAS_VULKAN
