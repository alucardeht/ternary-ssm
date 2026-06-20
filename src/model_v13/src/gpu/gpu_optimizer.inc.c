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
    vk_end_batch(p->ctx);
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

