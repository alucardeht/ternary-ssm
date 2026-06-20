#include "gpu_pipeline_internal.h"

#ifdef V12_HAS_VULKAN

void gpu_upload_shadows(VulkanContext *ctx, VkBufferHandle dst, const int16_t *src_shadow, size_t count) {
    if (!src_shadow || !VK_BUFFER_VALID(dst)) return;
    float *temp = malloc(count * sizeof(float));
    if (temp) {
        for (size_t i = 0; i < count; i++) {
            temp[i] = shadow_decode(src_shadow[i]);
        }
        vk_upload(ctx, dst, temp, count * sizeof(float));
        free(temp);
    }
}

void gpu_download_shadows(VulkanContext *ctx, int16_t *dst_shadow, VkBufferHandle src, size_t count, uint64_t step) {
    if (!dst_shadow || !VK_BUFFER_VALID(src)) return;
    float *temp = malloc(count * sizeof(float));
    if (temp) {
        vk_download(ctx, temp, src, count * sizeof(float));
        for (size_t i = 0; i < count; i++) {
            uint32_t seed = (uint32_t)(step ^ i ^ 0x9e3779b9u);
            seed = seed * 1664525u + 1013904223u;
            dst_shadow[i] = stochastic_round_shadow(temp[i], seed);
        }
        free(temp);
    }
}

void upload_shadow_buffer(VulkanContext *ctx, VkBufferHandle dst, const shadow_t *src, size_t count) {
#if V12_F2_INT8_SHADOWS
    gpu_upload_shadows(ctx, dst, src, count);
#else
    vk_upload(ctx, dst, src, count * sizeof(float));
#endif
}

void download_shadow_buffer(VulkanContext *ctx, shadow_t *dst, VkBufferHandle src, size_t count, uint64_t step) {
#if V12_F2_INT8_SHADOWS
    gpu_download_shadows(ctx, dst, src, count, step);
#else
    (void)step;
    vk_download(ctx, dst, src, count * sizeof(float));
#endif
}

int upload_csr_to_gpu(VulkanContext *ctx, VkTernaryCSR *dst, const TernaryCSR *src) {
#if S01_SUBGROUP_PACKED_TERNARY
    // S-21: Ternary Weight Bit-Interleaving Dispatch
    // Pack 2-bit ternary weights into interleaved 32-bit uints.
    // Each weight takes 2 bits: bit 0 = mag, bit 1 = sign.
    uint32_t words_per_row = (src->cols + 15) / 16;
    size_t packed_bytes = (size_t)src->rows * words_per_row * sizeof(uint32_t);
    uint32_t *packed_buf = calloc((size_t)src->rows * words_per_row, sizeof(uint32_t));
    if (!packed_buf) {
        return -1;
    }
    
    for (int r = 0; r < src->rows; r++) {
        for (int k = src->row_ptr_pos[r]; k < src->row_ptr_pos[r+1]; k++) {
            int c = src->pos_indices[k];
            packed_buf[r * words_per_row + c / 16] |= (1u << ((c % 16) * 2));
        }
        for (int k = src->row_ptr_neg[r]; k < src->row_ptr_neg[r+1]; k++) {
            int c = src->neg_indices[k];
            packed_buf[r * words_per_row + c / 16] |= (1u << ((c % 16) * 2)); // mag
            packed_buf[r * words_per_row + c / 16] |= (1u << ((c % 16) * 2 + 1)); // sign
        }
    }

    dst->pos_indices = vk_alloc(ctx, packed_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    dst->neg_indices = vk_alloc(ctx, 4, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST); // unused, allocate 4 bytes
    dst->row_ptr_pos = vk_alloc(ctx, 4, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    dst->row_ptr_neg = vk_alloc(ctx, 4, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    dst->rows = (uint32_t)src->rows;
    dst->cols = (uint32_t)src->cols;

    int ok = 1;
    ok &= (vk_upload(ctx, dst->pos_indices, packed_buf, packed_bytes) == 0);
    
    uint32_t dummy = 0;
    ok &= (vk_upload(ctx, dst->neg_indices, &dummy, 4) == 0);
    ok &= (vk_upload(ctx, dst->row_ptr_pos, &dummy, 4) == 0);
    ok &= (vk_upload(ctx, dst->row_ptr_neg, &dummy, 4) == 0);
    
    
    
    return ok ? 0 : -1;
#else
    size_t pos_bytes = (size_t)src->total_pos * sizeof(int32_t);
    size_t neg_bytes = (size_t)src->total_neg * sizeof(int32_t);
    size_t rp_bytes  = (size_t)(src->rows + 1) * sizeof(int32_t);

    dst->pos_indices = vk_alloc(ctx, pos_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    dst->neg_indices = vk_alloc(ctx, neg_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    dst->row_ptr_pos = vk_alloc(ctx, rp_bytes,  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    dst->row_ptr_neg = vk_alloc(ctx, rp_bytes,  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    dst->rows = (uint32_t)src->rows;
    dst->cols = (uint32_t)src->cols;

    int ok = 1;
    ok &= (vk_upload(ctx, dst->pos_indices, src->pos_indices, pos_bytes) == 0);
    ok &= (vk_upload(ctx, dst->neg_indices, src->neg_indices, neg_bytes) == 0);
    ok &= (vk_upload(ctx, dst->row_ptr_pos, src->row_ptr_pos, rp_bytes) == 0);
    ok &= (vk_upload(ctx, dst->row_ptr_neg, src->row_ptr_neg, rp_bytes) == 0);
    return ok ? 0 : -1;
#endif
}

int upload_csc_to_gpu(VulkanContext *ctx, VkTernaryCSR *dst, const TernaryCSC *csc) {
    /* CSC viewed as CSR of W^T: swap rows/cols */
#if S01_SUBGROUP_PACKED_TERNARY
    uint32_t eff_rows = csc->cols;
    uint32_t eff_cols = csc->rows;
    // S-21: Ternary Weight Bit-Interleaving Dispatch
    uint32_t words_per_row = (eff_cols + 15) / 16;
    size_t packed_bytes = (size_t)eff_rows * words_per_row * sizeof(uint32_t);
    uint32_t *packed_buf = calloc((size_t)eff_rows * words_per_row, sizeof(uint32_t));
    if (!packed_buf) {
        return -1;
    }
    
    for (uint32_t c = 0; c < csc->cols; c++) {
        for (int k = csc->col_ptr_pos[c]; k < csc->col_ptr_pos[c+1]; k++) {
            int r = csc->pos_row_idx[k];
            packed_buf[c * words_per_row + r / 16] |= (1u << ((r % 16) * 2));
        }
        for (int k = csc->col_ptr_neg[c]; k < csc->col_ptr_neg[c+1]; k++) {
            int r = csc->neg_row_idx[k];
            packed_buf[c * words_per_row + r / 16] |= (1u << ((r % 16) * 2));
            packed_buf[c * words_per_row + r / 16] |= (1u << ((r % 16) * 2 + 1));
        }
    }

    dst->pos_indices = vk_alloc(ctx, packed_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    dst->neg_indices = vk_alloc(ctx, 4, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST); // unused
    dst->row_ptr_pos = vk_alloc(ctx, 4, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    dst->row_ptr_neg = vk_alloc(ctx, 4, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    dst->rows = eff_rows;
    dst->cols = eff_cols;

    int ok = 1;
    
    
    
    return ok ? 0 : -1;
#else
    size_t rp_bytes = (size_t)(csc->cols + 1) * sizeof(int32_t);
    size_t pi_bytes = (size_t)csc->total_pos * sizeof(int32_t);
    size_t ni_bytes = (size_t)csc->total_neg * sizeof(int32_t);

    dst->rows = (uint32_t)csc->cols;  /* transposed */
    dst->cols = (uint32_t)csc->rows;
    dst->row_ptr_pos = vk_alloc(ctx, rp_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    dst->pos_indices = vk_alloc(ctx, pi_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    dst->row_ptr_neg = vk_alloc(ctx, rp_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    dst->neg_indices = vk_alloc(ctx, ni_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);

    vk_upload(ctx, dst->row_ptr_pos, csc->col_ptr_pos, rp_bytes);
    vk_upload(ctx, dst->pos_indices, csc->pos_row_idx, pi_bytes);
    vk_upload(ctx, dst->row_ptr_neg, csc->col_ptr_neg, rp_bytes);
    vk_upload(ctx, dst->neg_indices, csc->neg_row_idx, ni_bytes);
    return 0;
#endif
}

void gpu_csr_rebuild_from_sampled(GpuPipeline *p, VkBufferHandle sampled,
                                          VkTernaryCSR *gpu_csr,
                                          uint32_t rows, uint32_t cols);

void gpu_pipeline_rebuild_lm_head_ce_csr(GpuPipeline *p, Model *m) {
#if V12_TOLR_ENABLED
    (void)p; (void)m;
    return;
#else
    if (!p || !m) return;

    if ((m->gpu_opt_flags & GPU_OPT_LM_HEAD)
        && VK_BUFFER_VALID(p->opt_sampled_lm_head)
        && VK_BUFFER_VALID(p->ce_row_ptr_pos)
        && VK_BUFFER_VALID(p->ce_row_ptr_neg)
        && VK_BUFFER_VALID(p->ce_pos_indices)
        && VK_BUFFER_VALID(p->ce_neg_indices)) {
        VkTernaryCSR ce_csr = {
            .pos_indices = p->ce_pos_indices,
            .neg_indices = p->ce_neg_indices,
            .row_ptr_pos = p->ce_row_ptr_pos,
            .row_ptr_neg = p->ce_row_ptr_neg,
            .rows = (uint32_t)V12_VOCAB_SIZE,
            .cols = (uint32_t)p->d_model,
        };
        gpu_csr_rebuild_from_sampled(p, p->opt_sampled_lm_head, &ce_csr,
                                     (uint32_t)V12_VOCAB_SIZE,
                                     (uint32_t)p->d_model);
        return;
    }

    if (VK_BUFFER_VALID(p->ce_row_ptr_pos)) {
        vk_free(p->ctx, p->ce_row_ptr_pos);
        p->ce_row_ptr_pos = vk_alloc(p->ctx,
            (size_t)(m->csr_lm_head.rows + 1) * sizeof(int32_t),
            VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        vk_upload(p->ctx, p->ce_row_ptr_pos, m->csr_lm_head.row_ptr_pos,
                  (size_t)(m->csr_lm_head.rows + 1) * sizeof(int32_t));
    }
    if (VK_BUFFER_VALID(p->ce_pos_indices)) {
        vk_free(p->ctx, p->ce_pos_indices);
        p->ce_pos_indices = vk_alloc(p->ctx,
            (size_t)m->csr_lm_head.total_pos * sizeof(int32_t),
            VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        vk_upload(p->ctx, p->ce_pos_indices, m->csr_lm_head.pos_indices,
                  (size_t)m->csr_lm_head.total_pos * sizeof(int32_t));
    }
    if (VK_BUFFER_VALID(p->ce_row_ptr_neg)) {
        vk_free(p->ctx, p->ce_row_ptr_neg);
        p->ce_row_ptr_neg = vk_alloc(p->ctx,
            (size_t)(m->csr_lm_head.rows + 1) * sizeof(int32_t),
            VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        vk_upload(p->ctx, p->ce_row_ptr_neg, m->csr_lm_head.row_ptr_neg,
                  (size_t)(m->csr_lm_head.rows + 1) * sizeof(int32_t));
    }
    if (VK_BUFFER_VALID(p->ce_neg_indices)) {
        vk_free(p->ctx, p->ce_neg_indices);
        p->ce_neg_indices = vk_alloc(p->ctx,
            (size_t)m->csr_lm_head.total_neg * sizeof(int32_t),
            VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
        vk_upload(p->ctx, p->ce_neg_indices, m->csr_lm_head.neg_indices,
                  (size_t)m->csr_lm_head.total_neg * sizeof(int32_t));
    }
#endif
}

void prblt_alloc_max_csr(VulkanContext *ctx, VkTernaryCSR *csr, uint32_t max_rows, uint32_t max_cols) {
    size_t rp_bytes = (max_rows + 1) * sizeof(int32_t);
    size_t idx_bytes = (max_rows * max_cols) * sizeof(int32_t);
    csr->pos_indices = vk_alloc(ctx, idx_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    csr->neg_indices = vk_alloc(ctx, idx_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    csr->row_ptr_pos = vk_alloc(ctx, rp_bytes,  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    csr->row_ptr_neg = vk_alloc(ctx, rp_bytes,  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    csr->rows = max_rows;
    csr->cols = max_cols;
}

void prblt_free_max_csr(VulkanContext *ctx, VkTernaryCSR *csr) {
    vk_free(ctx, csr->pos_indices);
    vk_free(ctx, csr->neg_indices);
    vk_free(ctx, csr->row_ptr_pos);
    vk_free(ctx, csr->row_ptr_neg);
}


void gpu_pipeline_sync_weights(GpuPipeline *p, Model *m) {
    if (!p || !m) return;
    size_t pred_bytes   = (size_t)p->d_model * p->d_model * sizeof(float);
    size_t embed_E_bytes = (size_t)V12_VOCAB_SIZE * V12_D_EMBED * sizeof(float);
    size_t embed_P_bytes = (size_t)V12_D_EMBED    * p->d_model  * sizeof(float);

    vk_begin_batch(p->ctx);

    if (!(m->gpu_opt_flags & GPU_OPT_W_PRED)) {
        vk_upload(p->ctx, p->pred_A,        m->sampled_w_pred, pred_bytes);
        upload_shadow_buffer(p->ctx, p->shadow_pred_A, m->shadow_w_pred,  (size_t)p->d_model * p->d_model);
    }
    if (!(m->gpu_opt_flags & GPU_OPT_NORM)) {
        vk_upload(p->ctx, p->norm_w, m->norm_w, (size_t)p->d_model * sizeof(float));
    }
    if (!(m->gpu_opt_flags & GPU_OPT_NORM)) {
        vk_upload(p->ctx, p->norm_b, m->norm_b, (size_t)p->d_model * sizeof(float));
    }
    if (VK_BUFFER_VALID(p->embed_E_gpu) && !(m->gpu_opt_flags & GPU_OPT_EMBED_E)) {
        vk_upload(p->ctx, p->embed_E_gpu, m->embed_E, embed_E_bytes);
    }
    if (VK_BUFFER_VALID(p->embed_P_gpu) && !(m->gpu_opt_flags & GPU_OPT_EMBED_P)) {
        vk_upload(p->ctx, p->embed_P_gpu, m->embed_P, embed_P_bytes);
    }
    if (VK_BUFFER_VALID(p->q10_w_v)) {
        int ds = V12_D_STATE;
        float lr = 0.001f; /* p->learning_rate */
        vk_dispatch_q10_adam_update(p->ctx, p->q10_w_v, p->q10_w_v_m, p->q10_w_v_v, p->q10_dw_v, ds, lr, 0.9f, 0.999f, 1e-8f);
    }


    vk_end_batch(p->ctx);
}

void gpu_pipeline_sync_materialized_state(GpuPipeline *p, Model *m) {
    if (!p || !m) return;

    const size_t win_bytes      = (size_t)V12_D_MODEL * V12_D_STATE * sizeof(float);
    const size_t wout_bytes     = (size_t)V12_D_STATE * V12_D_MODEL * sizeof(float);
    const size_t pred_bytes     = (size_t)p->d_model * p->d_model * sizeof(float);
    const size_t lm_dense_bytes = (size_t)V12_VOCAB_SIZE * p->d_model * sizeof(float);
    const size_t lm_row_bytes   = (size_t)V12_K_HOT * sizeof(float);
    const size_t embed_E_bytes  = (size_t)V12_VOCAB_SIZE * V12_D_EMBED * sizeof(float);
    const size_t embed_P_bytes  = (size_t)V12_D_EMBED * p->d_model * sizeof(float);
    const size_t ds_bytes       = (size_t)V12_D_STATE * sizeof(float);
    const size_t d_bytes        = (size_t)V12_D_MODEL * sizeof(float);

    gpu_pipeline_sync_weights(p, m);

    for (int k = 0; k < p->n_layers; k++) {
        SSMLayer *L = &m->layers[k];

        if ((m->gpu_opt_flags & GPU_OPT_W_IN) && L->shadow_w_in
            && VK_BUFFER_VALID(p->opt_layers[k].shadow_w_in)) {
            upload_shadow_buffer(p->ctx, p->opt_layers[k].shadow_w_in, L->shadow_w_in, (size_t)V12_D_MODEL * V12_D_STATE);
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_IN) && L->sampled_w_in
            && VK_BUFFER_VALID(p->opt_layers[k].sampled_w_in)) {
            vk_upload(p->ctx, p->opt_layers[k].sampled_w_in, L->sampled_w_in, win_bytes);
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_IN) && L->m_w_in_row
            && VK_BUFFER_VALID(p->opt_layers[k].m_w_in_row)) {
            vk_upload(p->ctx, p->opt_layers[k].m_w_in_row, L->m_w_in_row, d_bytes);
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_IN) && L->v_w_in_row
            && VK_BUFFER_VALID(p->opt_layers[k].v_w_in_row)) {
            vk_upload(p->ctx, p->opt_layers[k].v_w_in_row, L->v_w_in_row, d_bytes);
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_IN) && L->m_w_in_col
            && VK_BUFFER_VALID(p->opt_layers[k].m_w_in_col)) {
            vk_upload(p->ctx, p->opt_layers[k].m_w_in_col, L->m_w_in_col, ds_bytes);
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_IN) && L->v_w_in_col
            && VK_BUFFER_VALID(p->opt_layers[k].v_w_in_col)) {
            vk_upload(p->ctx, p->opt_layers[k].v_w_in_col, L->v_w_in_col, ds_bytes);
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_OUT) && L->shadow_w_out
            && VK_BUFFER_VALID(p->opt_layers[k].shadow_w_out)) {
            upload_shadow_buffer(p->ctx, p->opt_layers[k].shadow_w_out, L->shadow_w_out, (size_t)V12_D_STATE * V12_D_MODEL);
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_OUT) && L->sampled_w_out
            && VK_BUFFER_VALID(p->opt_layers[k].sampled_w_out)) {
            vk_upload(p->ctx, p->opt_layers[k].sampled_w_out, L->sampled_w_out, wout_bytes);
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_OUT) && L->m_w_out_row
            && VK_BUFFER_VALID(p->opt_layers[k].m_w_out_row)) {
            vk_upload(p->ctx, p->opt_layers[k].m_w_out_row, L->m_w_out_row, ds_bytes);
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_OUT) && L->v_w_out_row
            && VK_BUFFER_VALID(p->opt_layers[k].v_w_out_row)) {
            vk_upload(p->ctx, p->opt_layers[k].v_w_out_row, L->v_w_out_row, ds_bytes);
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_OUT) && L->m_w_out_col
            && VK_BUFFER_VALID(p->opt_layers[k].m_w_out_col)) {
            vk_upload(p->ctx, p->opt_layers[k].m_w_out_col, L->m_w_out_col, d_bytes);
        }
        if ((m->gpu_opt_flags & GPU_OPT_W_OUT) && L->v_w_out_col
            && VK_BUFFER_VALID(p->opt_layers[k].v_w_out_col)) {
            vk_upload(p->ctx, p->opt_layers[k].v_w_out_col, L->v_w_out_col, d_bytes);
        }
    }

    if ((m->gpu_opt_flags & GPU_OPT_LM_HEAD) && m->shadow_lm_head
        && VK_BUFFER_VALID(p->opt_shadow_lm_head)) {
        upload_shadow_buffer(p->ctx, p->opt_shadow_lm_head, m->shadow_lm_head, (size_t)V12_VOCAB_SIZE * p->d_model);
    }
    if ((m->gpu_opt_flags & GPU_OPT_LM_HEAD) && m->sampled_lm_head
        && VK_BUFFER_VALID(p->opt_sampled_lm_head)) {
        vk_upload(p->ctx, p->opt_sampled_lm_head, m->sampled_lm_head, lm_dense_bytes);
    }
    if ((m->gpu_opt_flags & GPU_OPT_LM_HEAD) && m->m_lm_head_row
        && VK_BUFFER_VALID(p->opt_m_lm_head_row)) {
        upload_lm_hot_row_moments(p->ctx, p->opt_m_lm_head_row, m->hot_to_vocab, m->m_lm_head_row);
    }
    if ((m->gpu_opt_flags & GPU_OPT_LM_HEAD) && m->v_lm_head_row
        && VK_BUFFER_VALID(p->opt_v_lm_head_row)) {
        upload_lm_hot_row_moments(p->ctx, p->opt_v_lm_head_row, m->hot_to_vocab, m->v_lm_head_row);
    }
    if ((m->gpu_opt_flags & GPU_OPT_LM_HEAD) && m->m_lm_head_col
        && VK_BUFFER_VALID(p->opt_m_lm_head_col)) {
        vk_upload(p->ctx, p->opt_m_lm_head_col, m->m_lm_head_col, d_bytes);
    }
    if ((m->gpu_opt_flags & GPU_OPT_LM_HEAD) && m->v_lm_head_col
        && VK_BUFFER_VALID(p->opt_v_lm_head_col)) {
        vk_upload(p->ctx, p->opt_v_lm_head_col, m->v_lm_head_col, d_bytes);
    }

    if ((m->gpu_opt_flags & GPU_OPT_W_PRED) && model_w_pred_host_is_materialized(m)
        && m->shadow_w_pred
        && VK_BUFFER_VALID(p->opt_shadow_w_pred)) {
        upload_shadow_buffer(p->ctx, p->opt_shadow_w_pred, m->shadow_w_pred, (size_t)p->d_model * p->d_model);
    }
    if ((m->gpu_opt_flags & GPU_OPT_W_PRED) && model_w_pred_host_is_materialized(m)
        && m->sampled_w_pred
        && VK_BUFFER_VALID(p->opt_sampled_w_pred)) {
        vk_upload(p->ctx, p->opt_sampled_w_pred, m->sampled_w_pred, pred_bytes);
    }
    if ((m->gpu_opt_flags & GPU_OPT_W_PRED) && model_w_pred_host_is_materialized(m)
        && m->m_w_pred_row
        && VK_BUFFER_VALID(p->opt_m_w_pred_row)) {
        vk_upload(p->ctx, p->opt_m_w_pred_row, m->m_w_pred_row, d_bytes);
    }
    if ((m->gpu_opt_flags & GPU_OPT_W_PRED) && model_w_pred_host_is_materialized(m)
        && m->v_w_pred_row
        && VK_BUFFER_VALID(p->opt_v_w_pred_row)) {
        vk_upload(p->ctx, p->opt_v_w_pred_row, m->v_w_pred_row, d_bytes);
    }
    if ((m->gpu_opt_flags & GPU_OPT_W_PRED) && model_w_pred_host_is_materialized(m)
        && m->m_w_pred_col
        && VK_BUFFER_VALID(p->opt_m_w_pred_col)) {
        vk_upload(p->ctx, p->opt_m_w_pred_col, m->m_w_pred_col, d_bytes);
    }
    if ((m->gpu_opt_flags & GPU_OPT_W_PRED) && model_w_pred_host_is_materialized(m)
        && m->v_w_pred_col
        && VK_BUFFER_VALID(p->opt_v_w_pred_col)) {
        vk_upload(p->ctx, p->opt_v_w_pred_col, m->v_w_pred_col, d_bytes);
    }
    if ((m->gpu_opt_flags & GPU_OPT_W_PRED) && model_w_pred_host_is_materialized(m)
        && m->sampled_w_pred && m->shadow_w_pred) {
        vk_upload(p->ctx, p->pred_A,        m->sampled_w_pred, pred_bytes);
        vk_upload(p->ctx, p->shadow_pred_A, m->shadow_w_pred,  pred_bytes);
    }

    if ((m->gpu_opt_flags & GPU_OPT_NORM) && model_norm_host_is_materialized(m) && m->norm_w)
        vk_upload(p->ctx, p->norm_w, m->norm_w, (size_t)p->d_model * sizeof(float));
    if ((m->gpu_opt_flags & GPU_OPT_NORM) && model_norm_host_is_materialized(m) && m->norm_b)
        vk_upload(p->ctx, p->norm_b, m->norm_b, (size_t)p->d_model * sizeof(float));
    if ((m->gpu_opt_flags & GPU_OPT_NORM) && model_norm_host_is_materialized(m)
        && m->m_norm_w && VK_BUFFER_VALID(p->opt_m_norm_w))
        vk_upload(p->ctx, p->opt_m_norm_w, m->m_norm_w, d_bytes);
    if ((m->gpu_opt_flags & GPU_OPT_NORM) && model_norm_host_is_materialized(m)
        && m->v_norm_w && VK_BUFFER_VALID(p->opt_v_norm_w))
        vk_upload(p->ctx, p->opt_v_norm_w, m->v_norm_w, d_bytes);
    if ((m->gpu_opt_flags & GPU_OPT_NORM) && model_norm_host_is_materialized(m)
        && m->m_norm_b && VK_BUFFER_VALID(p->opt_m_norm_b))
        vk_upload(p->ctx, p->opt_m_norm_b, m->m_norm_b, d_bytes);
    if ((m->gpu_opt_flags & GPU_OPT_NORM) && model_norm_host_is_materialized(m)
        && m->v_norm_b && VK_BUFFER_VALID(p->opt_v_norm_b))
        vk_upload(p->ctx, p->opt_v_norm_b, m->v_norm_b, d_bytes);
    if (model_embed_host_is_materialized(m)) {
        if ((m->gpu_opt_flags & GPU_OPT_EMBED_E) && m->embed_E && VK_BUFFER_VALID(p->embed_E_gpu))
            vk_upload(p->ctx, p->embed_E_gpu, m->embed_E, embed_E_bytes);
        if ((m->gpu_opt_flags & GPU_OPT_EMBED_E) && m->m_embed_E && VK_BUFFER_VALID(p->opt_m_embed_E))
            vk_upload(p->ctx, p->opt_m_embed_E, m->m_embed_E, embed_E_bytes);
        if ((m->gpu_opt_flags & GPU_OPT_EMBED_E) && m->v_embed_E && VK_BUFFER_VALID(p->opt_v_embed_E))
            vk_upload(p->ctx, p->opt_v_embed_E, m->v_embed_E, embed_E_bytes);
        if ((m->gpu_opt_flags & GPU_OPT_EMBED_P) && m->embed_P && VK_BUFFER_VALID(p->embed_P_gpu))
            vk_upload(p->ctx, p->embed_P_gpu, m->embed_P, embed_P_bytes);
        if ((m->gpu_opt_flags & GPU_OPT_EMBED_P) && m->m_embed_P && VK_BUFFER_VALID(p->opt_m_embed_P))
        if ((m->gpu_opt_flags & GPU_OPT_EMBED_P) && m->v_embed_P && VK_BUFFER_VALID(p->opt_v_embed_P))
            vk_upload(p->ctx, p->opt_v_embed_P, m->v_embed_P, embed_P_bytes);
    }
#if V12_PRBLT_ENABLED
    p->prblt_ring[0].active_layer_k = -1;
    p->prblt_ring[1].active_layer_k = -1;
#endif
}

void gpu_csr_rebuild_from_sampled(GpuPipeline *p, VkBufferHandle sampled,
                                          VkTernaryCSR *gpu_csr,
                                          uint32_t rows, uint32_t cols) {
    VulkanContext *ctx = p->ctx;
    const int caller_batch_active = vk_batch_active(ctx);

    uint32_t *cnt_p = malloc(rows * sizeof(uint32_t));
    uint32_t *cnt_n = malloc(rows * sizeof(uint32_t));
    if (!cnt_p || !cnt_n) {
        free(cnt_p);
        free(cnt_n);
        return;
    }

    if (caller_batch_active) {
        vk_end_batch(ctx);
    }

    vk_begin_batch(ctx);
    vk_dispatch_csr_count(ctx, sampled, p->csr_count_pos, p->csr_count_neg, rows, cols);
    vk_end_batch(ctx);

    vk_download(ctx, cnt_p, p->csr_count_pos, rows * sizeof(uint32_t));
    vk_download(ctx, cnt_n, p->csr_count_neg, rows * sizeof(uint32_t));

    uint32_t *rpp = malloc((rows + 1) * sizeof(uint32_t));
    uint32_t *rpn = malloc((rows + 1) * sizeof(uint32_t));
    if (!rpp || !rpn) {
        free(cnt_p);
        free(cnt_n);
        free(rpp);
        free(rpn);
        if (caller_batch_active) {
            vk_begin_batch(ctx);
        }
        return;
    }
    rpp[0] = 0; rpn[0] = 0;
    for (uint32_t i = 0; i < rows; i++) {
        rpp[i+1] = rpp[i] + cnt_p[i];
        rpn[i+1] = rpn[i] + cnt_n[i];
    }

    vk_begin_batch(ctx);
    vk_upload(ctx, gpu_csr->row_ptr_pos, rpp, (rows + 1) * sizeof(uint32_t));
    vk_upload(ctx, gpu_csr->row_ptr_neg, rpn, (rows + 1) * sizeof(uint32_t));

    vk_dispatch_csr_scatter(ctx, sampled, gpu_csr->row_ptr_pos, gpu_csr->row_ptr_neg,
                             gpu_csr->pos_indices, gpu_csr->neg_indices, rows, cols);
    vk_end_batch(ctx);

    if (caller_batch_active) {
        vk_begin_batch(ctx);
    }

    gpu_csr->rows = rows;
    gpu_csr->cols = cols;

    free(cnt_p); free(cnt_n); free(rpp); free(rpn);
}

static int r388_mvue_resample_with_gate(GpuPipeline *p, Model *m,
                                        VkBufferHandle shadow,
                                        VkBufferHandle sampled,
                                        uint32_t flip_index,
                                        uint32_t n,
                                        uint32_t rng_seed) {
    if (!p || !m) return -1;
    if (!m->r388_flip_refresh_gate_enabled) {
        return vk_dispatch_mvue_resample(p->ctx, shadow, sampled, p->flip_counts,
                                         flip_index, n, rng_seed);
    }

    if (!VK_BUFFER_VALID(sampled) || n == 0u) {
        return vk_dispatch_mvue_resample(p->ctx, shadow, sampled, p->flip_counts,
                                         flip_index, n, rng_seed);
    }

    const size_t bytes = (size_t)n * sizeof(float);
    float *before = (float *)malloc(bytes);
    float *after = (float *)malloc(bytes);
    if (!before || !after) {
        free(before);
        free(after);
        return vk_dispatch_mvue_resample(p->ctx, shadow, sampled, p->flip_counts,
                                         flip_index, n, rng_seed);
    }

    const int caller_batch_active = vk_batch_active(p->ctx);
    if (caller_batch_active) {
        vk_end_batch(p->ctx);
    }

    int ok = (vk_download(p->ctx, before, sampled, bytes) == 0);
    const int rc = vk_dispatch_mvue_resample(p->ctx, shadow, sampled, p->flip_counts,
                                             flip_index, n, rng_seed);
    vk_sync(p->ctx);
    ok = ok && (rc == 0) && (vk_download(p->ctx, after, sampled, bytes) == 0);

    if (caller_batch_active) {
        vk_begin_batch(p->ctx);
    }

    if (ok) {
        uint32_t block_elems = m->r388_flip_refresh_block_elements;
        if (block_elems == 0u) block_elems = 1024u;

        const uint64_t blocks =
            ((uint64_t)n + (uint64_t)block_elems - 1ull) / (uint64_t)block_elems;
        uint64_t omittable_blocks = 0ull;
        uint64_t flips = 0ull;

        for (uint64_t b = 0; b < blocks; b++) {
            const uint64_t start = b * (uint64_t)block_elems;
            uint64_t end = start + (uint64_t)block_elems;
            if (end > (uint64_t)n) end = (uint64_t)n;

            uint64_t block_flips = 0ull;
            for (uint64_t i = start; i < end; i++) {
                if (before[i] != after[i]) {
                    block_flips++;
                }
            }
            if (block_flips == 0ull) {
                omittable_blocks++;
            }
            flips += block_flips;
        }

        m->r388_flip_refresh_gate_batches++;
        m->r388_flip_refresh_matrices++;
        m->r388_flip_refresh_elements += (uint64_t)n;
        m->r388_flip_refresh_blocks += blocks;
        m->r388_flip_refresh_omittable_blocks += omittable_blocks;
        m->r388_flip_refresh_flips += flips;

        const double eta =
            m->r388_flip_refresh_blocks > 0ull
                ? (double)m->r388_flip_refresh_omittable_blocks /
                  (double)m->r388_flip_refresh_blocks
                : 0.0;
        const double flip_fraction =
            m->r388_flip_refresh_elements > 0ull
                ? (double)m->r388_flip_refresh_flips /
                  (double)m->r388_flip_refresh_elements
                : 0.0;
        m->r388_flip_refresh_eta = (float)eta;
        m->r388_flip_refresh_flip_fraction = (float)flip_fraction;
        m->r388_flip_refresh_projected_refresh_gain =
            eta < 0.999999 ? (float)(1.0 / (1.0 - eta)) : 1000000.0f;
        m->r388_flip_refresh_identifying = 1;
        m->r388_flip_refresh_pass =
            (eta >= (double)m->r388_flip_refresh_min_eta) ? 1 : 0;
    }

    free(before);
    free(after);
    return rc;
}

void gpu_pipeline_reupload_csr(GpuPipeline *p, Model *m, uint32_t rng_seed) {
    if (!p) return;
    (void)rng_seed;
    for (int k = 0; k < p->n_layers; k++) {
        SSMLayer *L = &m->layers[k];

        if (VK_BUFFER_VALID(p->opt_layers[k].sampled_w_in) &&
            VK_BUFFER_VALID(p->layers[k].gpu_csr_in.row_ptr_pos)) {
            gpu_csr_rebuild_from_sampled(p, p->opt_layers[k].sampled_w_in,
                                         &p->layers[k].gpu_csr_in,
                                         (uint32_t)L->csr_in.rows,
                                         (uint32_t)L->csr_in.cols);
        }
        if (VK_BUFFER_VALID(p->opt_layers[k].sampled_w_out) &&
            VK_BUFFER_VALID(p->layers[k].gpu_csr_out.row_ptr_pos)) {
            gpu_csr_rebuild_from_sampled(p, p->opt_layers[k].sampled_w_out,
                                         &p->layers[k].gpu_csr_out,
                                         (uint32_t)L->csr_out.rows,
                                         (uint32_t)L->csr_out.cols);
        }

        if (!VK_BUFFER_VALID(p->opt_layers[k].sampled_w_in) ||
            !VK_BUFFER_VALID(p->opt_layers[k].sampled_w_out)) {
            gpu_pipeline_reupload_layer(p, m, k);
        }
    }

    gpu_pipeline_rebuild_lm_head_ce_csr(p, m);
}

void gpu_pipeline_resample_and_reupload_csr(GpuPipeline *p, Model *m, uint32_t rng_seed) {
    if (!p) return;
    for (int k = 0; k < p->n_layers; k++) {
        SSMLayer *L = &m->layers[k];

        if (VK_BUFFER_VALID(p->opt_layers[k].shadow_w_in) &&
            VK_BUFFER_VALID(p->opt_layers[k].sampled_w_in) &&
            VK_BUFFER_VALID(p->flip_counts)) {
            r388_mvue_resample_with_gate(p, m, p->opt_layers[k].shadow_w_in,
                                         p->opt_layers[k].sampled_w_in,
                                         (uint32_t)(2 * k),
                                         (uint32_t)((size_t)L->csr_in.rows *
                                                    (size_t)L->csr_in.cols),
                                         rng_seed ^ (uint32_t)(0x9E3779B9u + (uint32_t)k));
        }
        if (VK_BUFFER_VALID(p->opt_layers[k].shadow_w_out) &&
            VK_BUFFER_VALID(p->opt_layers[k].sampled_w_out) &&
            VK_BUFFER_VALID(p->flip_counts)) {
            r388_mvue_resample_with_gate(p, m, p->opt_layers[k].shadow_w_out,
                                         p->opt_layers[k].sampled_w_out,
                                         (uint32_t)(2 * k + 1),
                                         (uint32_t)((size_t)L->csr_out.rows *
                                                    (size_t)L->csr_out.cols),
                                         rng_seed ^ (uint32_t)(0x85EBCA6Bu + (uint32_t)k));
        }

        if (VK_BUFFER_VALID(p->opt_layers[k].sampled_w_in) &&
            VK_BUFFER_VALID(p->layers[k].gpu_csr_in.row_ptr_pos)) {
            gpu_csr_rebuild_from_sampled(p, p->opt_layers[k].sampled_w_in,
                                         &p->layers[k].gpu_csr_in,
                                         (uint32_t)L->csr_in.rows,
                                         (uint32_t)L->csr_in.cols);
        }
        if (VK_BUFFER_VALID(p->opt_layers[k].sampled_w_out) &&
            VK_BUFFER_VALID(p->layers[k].gpu_csr_out.row_ptr_pos)) {
            gpu_csr_rebuild_from_sampled(p, p->opt_layers[k].sampled_w_out,
                                         &p->layers[k].gpu_csr_out,
                                         (uint32_t)L->csr_out.rows,
                                         (uint32_t)L->csr_out.cols);
        }

        if (!VK_BUFFER_VALID(p->opt_layers[k].sampled_w_in) ||
            !VK_BUFFER_VALID(p->opt_layers[k].sampled_w_out)) {
            gpu_pipeline_reupload_layer(p, m, k);
        }
    }

    if (VK_BUFFER_VALID(p->opt_shadow_lm_head) &&
        VK_BUFFER_VALID(p->opt_sampled_lm_head) &&
        VK_BUFFER_VALID(p->flip_counts)) {
        r388_mvue_resample_with_gate(p, m, p->opt_shadow_lm_head,
                                     p->opt_sampled_lm_head,
                                     (uint32_t)(V12_N_LAYERS * 2),
                                     (uint32_t)((size_t)V12_VOCAB_SIZE *
                                                (size_t)V12_D_MODEL),
                                     rng_seed ^ 0xC2B2AE35u);
    }
    if (VK_BUFFER_VALID(p->opt_shadow_w_pred) &&
        VK_BUFFER_VALID(p->opt_sampled_w_pred) &&
        VK_BUFFER_VALID(p->flip_counts)) {
        r388_mvue_resample_with_gate(p, m, p->opt_shadow_w_pred,
                                     p->opt_sampled_w_pred,
                                     (uint32_t)(V12_N_LAYERS * 2 + 1),
                                     (uint32_t)((size_t)V12_D_MODEL *
                                                (size_t)V12_D_MODEL),
                                     rng_seed ^ 0x27D4EB2Fu);
    }
    gpu_pipeline_rebuild_lm_head_ce_csr(p, m);
}

void prblt_load_layer(GpuPipeline *p, Model *m, int k) {
    if (!p || !m || k < 0 || k >= p->n_layers) return;
    int r = k % 2;
    if (p->prblt_ring[r].active_layer_k == k) {
        return; /* Already loaded */
    }

    VulkanContext *ctx = p->ctx;
    SSMLayer *L = &m->layers[k];
    size_t d = (size_t)p->d_model;
    size_t ds = (size_t)p->d_state;

    /* Free old CSR/CSC buffers in this slot */
    free_vk_csr(ctx, &p->prblt_ring[r].gpu_csr_in);
    free_vk_csr(ctx, &p->prblt_ring[r].gpu_csr_out);
    free_vk_csr(ctx, &p->prblt_ring[r].gpu_csc_in);
    free_vk_csr(ctx, &p->prblt_ring[r].gpu_csc_out);

    /* Upload/Pack new CSR/CSC */
    upload_csr_to_gpu(ctx, &p->prblt_ring[r].gpu_csr_in,  &L->csr_in);
    upload_csr_to_gpu(ctx, &p->prblt_ring[r].gpu_csr_out, &L->csr_out);

    TernaryCSC csc_in, csc_out;
    if (ternary_csc_from_csr(&csc_in, &L->csr_in) == 0) {
        upload_csc_to_gpu(ctx, &p->prblt_ring[r].gpu_csc_in, &csc_in);
        ternary_csc_free(&csc_in);
    }
    if (ternary_csc_from_csr(&csc_out, &L->csr_out) == 0) {
        upload_csc_to_gpu(ctx, &p->prblt_ring[r].gpu_csc_out, &csc_out);
        ternary_csc_free(&csc_out);
    }

    /* If training, upload shadow weights */


    p->prblt_ring[r].active_layer_k = k;
}

void gpu_pipeline_reupload_layer(GpuPipeline *p, Model *m, int k) {
    if (!p || !m || k < 0 || k >= p->n_layers) return;
    int r = -1;
    for (int i = 0; i < 2; i++) {
        if (p->prblt_ring[i].active_layer_k == k) {
            r = i;
            break;
        }
    }
    if (r == -1) return;
    VulkanContext *ctx = p->ctx;
    SSMLayer *L = &m->layers[k];
    size_t d = (size_t)p->d_model;
    size_t ds = (size_t)p->d_state;
    upload_shadow_buffer(ctx, p->prblt_ring[r].shadow_w_in, L->shadow_w_in, d * ds);
    upload_shadow_buffer(ctx, p->prblt_ring[r].shadow_w_out, L->shadow_w_out, ds * d);
}

void gpu_pipeline_offload_moments(GpuPipeline *p, Model *m, int k) {
    if (!p || !m || k < 0 || k >= p->n_layers) return;
    VulkanContext *ctx = p->ctx;
    SSMLayer *L = &m->layers[k];
    const size_t d = (size_t)p->d_model;
    const size_t ds = (size_t)p->d_state;
    const size_t norm_bytes = d * sizeof(float);
    
    /* 1. Download GPU-resident moments to CPU */
    if (L->m_w_in_row && VK_BUFFER_VALID(p->opt_layers[k].m_w_in_row)) {
        vk_download(ctx, L->m_w_in_row, p->opt_layers[k].m_w_in_row, norm_bytes);
    }
    if (L->v_w_in_row && VK_BUFFER_VALID(p->opt_layers[k].v_w_in_row)) {
        vk_download(ctx, L->v_w_in_row, p->opt_layers[k].v_w_in_row, norm_bytes);
    }
    if (L->m_w_in_col && VK_BUFFER_VALID(p->opt_layers[k].m_w_in_col)) {
        vk_download(ctx, L->m_w_in_col, p->opt_layers[k].m_w_in_col, ds * sizeof(float));
    }
    if (L->v_w_in_col && VK_BUFFER_VALID(p->opt_layers[k].v_w_in_col)) {
        vk_download(ctx, L->v_w_in_col, p->opt_layers[k].v_w_in_col, ds * sizeof(float));
    }
    if (L->m_w_out_row && VK_BUFFER_VALID(p->opt_layers[k].m_w_out_row)) {
        vk_download(ctx, L->m_w_out_row, p->opt_layers[k].m_w_out_row, ds * sizeof(float));
    }
    if (L->v_w_out_row && VK_BUFFER_VALID(p->opt_layers[k].v_w_out_row)) {
        vk_download(ctx, L->v_w_out_row, p->opt_layers[k].v_w_out_row, ds * sizeof(float));
    }
    if (L->m_w_out_col && VK_BUFFER_VALID(p->opt_layers[k].m_w_out_col)) {
        vk_download(ctx, L->m_w_out_col, p->opt_layers[k].m_w_out_col, norm_bytes);
    }
    if (L->v_w_out_col && VK_BUFFER_VALID(p->opt_layers[k].v_w_out_col)) {
        vk_download(ctx, L->v_w_out_col, p->opt_layers[k].v_w_out_col, norm_bytes);
    }

    /* 2. Free GPU buffers and set them to VK_BUFFER_NULL */
    if (VK_BUFFER_VALID(p->opt_layers[k].m_w_in_row)) {
        vk_free(ctx, p->opt_layers[k].m_w_in_row);
        p->opt_layers[k].m_w_in_row = VK_BUFFER_NULL;
    }
    if (VK_BUFFER_VALID(p->opt_layers[k].v_w_in_row)) {
        vk_free(ctx, p->opt_layers[k].v_w_in_row);
        p->opt_layers[k].v_w_in_row = VK_BUFFER_NULL;
    }
    if (VK_BUFFER_VALID(p->opt_layers[k].m_w_in_col)) {
        vk_free(ctx, p->opt_layers[k].m_w_in_col);
        p->opt_layers[k].m_w_in_col = VK_BUFFER_NULL;
    }
    if (VK_BUFFER_VALID(p->opt_layers[k].v_w_in_col)) {
        vk_free(ctx, p->opt_layers[k].v_w_in_col);
        p->opt_layers[k].v_w_in_col = VK_BUFFER_NULL;
    }
    if (VK_BUFFER_VALID(p->opt_layers[k].m_w_out_row)) {
        vk_free(ctx, p->opt_layers[k].m_w_out_row);
        p->opt_layers[k].m_w_out_row = VK_BUFFER_NULL;
    }
    if (VK_BUFFER_VALID(p->opt_layers[k].v_w_out_row)) {
        vk_free(ctx, p->opt_layers[k].v_w_out_row);
        p->opt_layers[k].v_w_out_row = VK_BUFFER_NULL;
    }
    if (VK_BUFFER_VALID(p->opt_layers[k].m_w_out_col)) {
        vk_free(ctx, p->opt_layers[k].m_w_out_col);
        p->opt_layers[k].m_w_out_col = VK_BUFFER_NULL;
    }
    if (VK_BUFFER_VALID(p->opt_layers[k].v_w_out_col)) {
        vk_free(ctx, p->opt_layers[k].v_w_out_col);
        p->opt_layers[k].v_w_out_col = VK_BUFFER_NULL;
    }
}


#else // !V12_HAS_VULKAN

void gpu_pipeline_sync_weights(GpuPipeline *p, Model *m)                          { (void)p; (void)m; }

void gpu_pipeline_sync_materialized_state(GpuPipeline *p, Model *m)               { (void)p; (void)m; }

int gpu_pipeline_sync_lm_head_to_cpu(GpuPipeline *p, Model *m)                   { (void)p; (void)m; return -1; }

int gpu_pipeline_sync_r364_compile_cache_to_cpu(GpuPipeline *p,
                                                Model *m,
                                                Cache *c,
                                                int batch_size) {
    (void)p;
    (void)m;
    (void)c;
    (void)batch_size;
    return -1;
}

int gpu_pipeline_sync_hidden_cache_to_cpu(GpuPipeline *p,
                                          Cache *c,
                                          int batch_size) {
    (void)p;
    (void)c;
    (void)batch_size;
    return -1;
}

void gpu_pipeline_sync_lm_head_from_cpu(GpuPipeline *p, Model *m)                 { (void)p; (void)m; }

void gpu_pipeline_reupload_csr(GpuPipeline *p, Model *m, uint32_t rng_seed)       { (void)p; (void)m; (void)rng_seed; }

void gpu_pipeline_resample_and_reupload_csr(GpuPipeline *p, Model *m, uint32_t rng_seed) { (void)p; (void)m; (void)rng_seed; }

void gpu_pipeline_reupload_layer(GpuPipeline *p, Model *m, int k)                 { (void)p; (void)m; (void)k; }

int gpu_sync_to_cpu(GpuPipeline *p, Model *m)                                    { (void)p; (void)m; return -1; }


#endif // V12_HAS_VULKAN
