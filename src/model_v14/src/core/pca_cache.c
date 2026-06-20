#include "pca_cache.h"
#include "model.h"
#include "ternary_matmul.h"
#include "ssm_scan.h"
#include "../io/dataset.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

ResidualCache *pca_cache_alloc(void) {
    ResidualCache *rc = calloc(1, sizeof(ResidualCache));
    if (!rc) return NULL;
    rc->active_entry = -1;
    for (int i = 0; i < V12_MAX_LAYERS; i++) {
        rc->entries[i].calibrated = 0;
    }
    return rc;
}

void pca_cache_free(ResidualCache *rc) {
    if (!rc) return;
    for (int i = 0; i < V12_MAX_LAYERS; i++) {
        free(rc->entries[i].basis);
        free(rc->entries[i].mean);
        free(rc->entries[i].codes);
        free(rc->entries[i].scales);
    }
    free(rc);
}

static void forward_pass_frozen_prefix(Model *m, Cache *c,
                                       const uint16_t *tokens, int seq_len,
                                       int layer_idx) {
    if (!m || !c || !tokens) return;

    int d = V12_D_MODEL;
    int ds = V12_D_STATE;

    float *h = c->residual;
    float out_buf[V12_D_MODEL];
    float h_init[V12_D_STATE];

    for (int t = 0; t < seq_len; t++) {
        {
            const float *e = m->embed_E + (size_t)tokens[t] * V12_D_EMBED;
            float *dst = h + t * d;
            for (int j = 0; j < d; j++) {
                float acc = 0.0f;
                for (int r = 0; r < V12_D_EMBED; r++) acc += e[r] * m->embed_P[r * d + j];
                dst[j] = acc;
            }
        }
    }

    {
        float *h_final_prev = c->residual + (size_t)V12_N_LAYERS * seq_len * d;
        float  pred_buf[V12_D_MODEL];

        for (int t = 0; t < seq_len; t++) {
            float *raw = h + t * d;

            if (t == 0) {

            } else {
                float *h_prev_vec = h_final_prev + (t - 1) * d;

                for (int j = 0; j < d; j++) {
                    float acc = 0.0f;
                    for (int k2 = 0; k2 < d; k2++) {
                        acc += m->sampled_w_pred[j * d + k2] * h_prev_vec[k2];
                    }
                    pred_buf[j] = acc;
                }

                for (int j = 0; j < d; j++) {
                    raw[j] -= pred_buf[j];
                }
            }
        }
    }

    for (int k = 1; k <= layer_idx && k < V12_N_LAYERS; k++) {
        float *r_in_k = c->residual + (size_t)k * seq_len * d;
        float  pred_k[V12_D_MODEL];

        for (int t = 1; t < seq_len; t++) {
            float *r_prev = r_in_k + (t - 1) * d;
            float *r_cur  = r_in_k + t * d;

            for (int j = 0; j < d; j++) {
                float acc = 0.0f;
                for (int k2 = 0; k2 < d; k2++) {
                    acc += m->sampled_w_pred[j * d + k2] * r_prev[k2];
                }
                pred_k[j] = acc;
            }

            for (int j = 0; j < d; j++) {
                r_cur[j] -= pred_k[j];
            }
        }
    }

    for (int k = 0; k <= layer_idx && k < V12_N_LAYERS; k++) {
        SSMLayer *L = &m->layers[k];
        float *r_in = c->residual + (size_t)k * seq_len * d;
        float *r_out = c->residual + (size_t)(k + 1) * seq_len * d;

        float *x_proj = malloc((size_t)seq_len * ds * sizeof(float));
        float *x_cv   = malloc((size_t)seq_len * ds * sizeof(float));
        float *x_gt   = malloc((size_t)seq_len * ds * sizeof(float));
        float *mask   = malloc((size_t)seq_len * ds * sizeof(float));
        float *b_eff  = malloc((size_t)seq_len * ds * sizeof(float));
        float *y_ssm  = malloc((size_t)seq_len * ds * sizeof(float));
        float *h_cache = malloc((size_t)seq_len * ds * sizeof(float));
        if (!x_proj || !x_cv || !x_gt || !mask || !b_eff || !y_ssm || !h_cache) {
            free(x_proj); free(x_cv); free(x_gt); free(mask); free(b_eff); free(y_ssm); free(h_cache);
            return;
        }

        for (int t = 0; t < seq_len; t++) {
            float *src = (k == 0) ? h + t * d : r_in + t * d;
            memcpy(r_in + t * d, src, d * sizeof(float));
        }

        for (int t = 0; t < seq_len; t++) {
            ternary_matvec(&L->csr_in, r_in + t * d, x_proj + t * ds);
        }

        {
            float scale = V12_SOFT_BOUNDARY_SCALE;
            float tau_k = m->tau_mux[k];
            for (int t = 0; t < seq_len; t++) {
                for (int i = 0; i < ds; i++) {
                    float xp = x_proj[t * ds + i];
                    float m_val = 1.0f / (1.0f + expf(-scale * (fabsf(xp) - tau_k)));
                    mask[t * ds + i] = m_val;
                    x_gt[t * ds + i] = xp * m_val;
                }
            }
        }

        {
            const int K = V12_CONV_KERNEL;
            for (int t = 0; t < seq_len; t++) {
                for (int i = 0; i < ds; i++) {
                    float acc = 0.0f;
                    for (int kk = 0; kk < K; kk++) {
                        int src_t = t - kk;
                        if (src_t < 0) src_t = 0;
                        acc += L->shadow_w_conv[kk * ds + i]
                             * x_gt[src_t * ds + i];
                    }
                    x_cv[t * ds + i] = acc;
                }
            }
        }

        memset(h_init, 0, (size_t)ds * sizeof(float));
        memset(h_cache, 0, (size_t)seq_len * ds * sizeof(float));

        for (int t = 0; t < seq_len; t++) {
            for (int i = 0; i < ds; i++) {
                float xc = x_cv[t * ds + i];
                float gate = 1.0f / (1.0f + expf(-xc));
                b_eff[t * ds + i] = gate * L->B[i];
            }
        }

        ssm_scan_forward(
            L->A, NULL, NULL, b_eff, ds, L->C,
            x_cv,
            h_init,
            y_ssm,
            h_cache,
            &L->moments,
            NULL,
            seq_len, ds, ds,
            NULL,  /* C_bilinear = NULL for cache reconstruction */
            L->sampled_u, L->sampled_v
        );

        /* Compress and store into c->h_all */
        {
            uint32_t rng = 0x12345678u ^ (uint32_t)k;
            uint8_t *h_all_dest = c->h_all + (size_t)k * seq_len * ds;
            for (size_t i = 0; i < (size_t)seq_len * ds; i++) {
                h_all_dest[i] = compress_e5m2_stochastic(h_cache[i], &rng);
            }
        }
#if V12_AF8C_ENABLED
        /* Compress and store into c->y_ssm */
        {
            uint32_t rng = 0x12345678u ^ (uint32_t)k;
            uint8_t *y_ssm_dest = c->y_ssm + (size_t)k * seq_len * ds;
            for (size_t i = 0; i < (size_t)seq_len * ds; i++) {
                y_ssm_dest[i] = compress_e5m2_stochastic(y_ssm[i], &rng);
            }
        }
#endif
        free(h_cache);

        for (int t = 0; t < seq_len; t++) {
            wht_inplace(y_ssm + t * ds, ds);
        }

        for (int t = 0; t < seq_len; t++) {
            ternary_matvec(&L->csr_out, y_ssm + t * ds, out_buf);
            for (int j = 0; j < d; j++) {
                r_out[t * d + j] = r_in[t * d + j] + out_buf[j];
            }
        }

        free(x_proj);
        free(x_cv);
        free(x_gt);
        free(mask);
        free(b_eff);
        free(y_ssm);
    }
}

int pca_cache_build_entry(Model *m, ResidualCache *rc, Dataset *ds,
                          int layer_idx, int seq_len, int n_seqs) {
    if (!m || !rc || !ds || layer_idx < 0 || layer_idx >= V12_N_LAYERS) {
        return -1;
    }

    int d = V12_D_MODEL;
    int rank = V12_PCA_CACHE_RANK;

    Cache *c = calloc(1, sizeof(Cache));
    if (!c) return -1;

    size_t residual_size = (size_t)(V12_N_LAYERS + 1) * seq_len * d;
    c->residual = calloc(residual_size, sizeof(float));
#if V12_AF8C_ENABLED
    c->x_proj = malloc((size_t)V12_N_LAYERS * seq_len * V12_D_STATE * sizeof(uint8_t));
    c->x_conv = malloc((size_t)V12_N_LAYERS * seq_len * V12_D_STATE * sizeof(uint8_t));
    c->x_gated = malloc((size_t)V12_N_LAYERS * seq_len * V12_D_STATE * sizeof(uint8_t));
    c->mask_val = malloc((size_t)V12_N_LAYERS * seq_len * V12_D_STATE * sizeof(uint8_t));
    c->b_eff_gate = malloc((size_t)V12_N_LAYERS * seq_len * V12_D_STATE * sizeof(uint8_t));
    c->y_ssm = malloc((size_t)V12_N_LAYERS * seq_len * V12_D_STATE * sizeof(uint8_t));
#else
    c->x_proj = malloc((size_t)V12_N_LAYERS * seq_len * V12_D_STATE * sizeof(float));
    c->x_conv = malloc((size_t)V12_N_LAYERS * seq_len * V12_D_STATE * sizeof(float));
    c->x_gated = malloc((size_t)V12_N_LAYERS * seq_len * V12_D_STATE * sizeof(float));
    c->mask_val = malloc((size_t)V12_N_LAYERS * seq_len * V12_D_STATE * sizeof(float));
    c->b_eff_gate = malloc((size_t)V12_N_LAYERS * seq_len * V12_D_STATE * sizeof(float));
    c->y_ssm = malloc((size_t)V12_N_LAYERS * seq_len * V12_D_STATE * sizeof(float));
#endif
    c->h_all = malloc((size_t)V12_N_LAYERS * seq_len * V12_D_STATE * sizeof(uint8_t));
    c->norm_out = malloc((size_t)seq_len * d * sizeof(float));
    c->norm_rms = malloc((size_t)seq_len * sizeof(float));
    c->res_cache = NULL;

    if (!c->residual || !c->x_proj || !c->x_conv || !c->x_gated ||
        !c->mask_val || !c->b_eff_gate || !c->y_ssm || !c->h_all ||
        !c->norm_out || !c->norm_rms) {
        fprintf(stderr, "[PCA] OOM allocating temp cache\n");
        free(c->residual);
        free(c->x_proj);
        free(c->x_conv);
        free(c->x_gated);
        free(c->mask_val);
        free(c->b_eff_gate);
        free(c->y_ssm);
        free(c->h_all);
        free(c->norm_out);
        free(c->norm_rms);
        free(c);
        return -1;
    }

    float *mean = malloc((size_t)d * sizeof(float));
    if (!mean) {
        fprintf(stderr, "[PCA] OOM for mean\n");
        free(c->residual);
        free(c->x_proj);
        free(c->x_conv);
        free(c->x_gated);
        free(c->mask_val);
        free(c->b_eff_gate);
        free(c->y_ssm);
        free(c->h_all);
        free(c->norm_out);
        free(c->norm_rms);
        free(c);
        return -1;
    }
    memset(mean, 0, (size_t)d * sizeof(float));

    dataset_reset_cursor(ds);

    for (int s = 0; s < n_seqs; s++) {
        uint16_t tokens[V12_SEQ_LEN];
        dataset_next_sequential(ds, tokens, 1, seq_len);
        forward_pass_frozen_prefix(m, c, tokens, seq_len, layer_idx);

        float *r_out = c->residual + (size_t)(layer_idx + 1) * seq_len * d;
        for (int t = 0; t < seq_len; t++) {
            for (int j = 0; j < d; j++) {
                mean[j] += r_out[t * d + j];
            }
        }
    }

    float inv_count = 1.0f / (float)(n_seqs * seq_len);
    for (int j = 0; j < d; j++) {
        mean[j] *= inv_count;
    }

    float *cov = malloc((size_t)d * d * sizeof(float));
    if (!cov) {
        fprintf(stderr, "[PCA] OOM for covariance\n");
        free(mean);
        free(c->residual);
        free(c->x_proj);
        free(c->x_conv);
        free(c->x_gated);
        free(c->mask_val);
        free(c->b_eff_gate);
        free(c->y_ssm);
        free(c->h_all);
        free(c->norm_out);
        free(c->norm_rms);
        free(c);
        return -1;
    }
    memset(cov, 0, (size_t)d * d * sizeof(float));

    dataset_reset_cursor(ds);

    for (int s = 0; s < n_seqs; s++) {
        uint16_t tokens[V12_SEQ_LEN];
        dataset_next_sequential(ds, tokens, 1, seq_len);
        forward_pass_frozen_prefix(m, c, tokens, seq_len, layer_idx);

        float *r_out = c->residual + (size_t)(layer_idx + 1) * seq_len * d;

        for (int t = 0; t < seq_len; t++) {
            for (int i = 0; i < d; i++) {
                float ci = r_out[t * d + i] - mean[i];
                for (int j = i; j < d; j++) {
                    float cj = r_out[t * d + j] - mean[j];
                    cov[i * d + j] += ci * cj;
                }
            }
        }
    }

    int total_vecs = n_seqs * seq_len;
    for (int i = 0; i < d; i++) {
        for (int j = i; j < d; j++) {
            cov[i * d + j] /= (float)total_vecs;
            cov[j * d + i] = cov[i * d + j];
        }
    }

    float *U = malloc((size_t)d * rank * sizeof(float));
    float *q = malloc((size_t)d * sizeof(float));
    float *aq = malloc((size_t)d * sizeof(float));

    if (!U || !q || !aq) {
        fprintf(stderr, "[PCA] OOM for eigenvector computation\n");
        free(U);
        free(q);
        free(aq);
        free(cov);
        free(mean);
        free(c->residual);
        free(c->x_proj);
        free(c->x_conv);
        free(c->x_gated);
        free(c->mask_val);
        free(c->b_eff_gate);
        free(c->y_ssm);
        free(c->h_all);
        free(c->norm_out);
        free(c->norm_rms);
        free(c);
        return -1;
    }

    memset(U, 0, (size_t)d * rank * sizeof(float));

    uint64_t rng = 0x853c49e6626f59d5ULL;

    for (int r = 0; r < rank && r < d; r++) {
        for (int j = 0; j < d; j++) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            q[j] = ((float)((rng >> 32) & 0x7FFFFFFF) / 1073741824.0f) * 0.01f + ((j == r) ? 1.0f : 0.0f);
        }

        for (int iter = 0; iter < 50; iter++) {
            for (int i = 0; i < d; i++) {
                aq[i] = 0.0f;
                for (int j = 0; j < d; j++) {
                    aq[i] += cov[i * d + j] * q[j];
                }
            }

            for (int p = 0; p < r; p++) {
                float dot = 0.0f;
                for (int j = 0; j < d; j++) {
                    dot += aq[j] * U[j * rank + p];
                }
                for (int j = 0; j < d; j++) {
                    aq[j] -= dot * U[j * rank + p];
                }
            }

            float norm = 0.0f;
            for (int j = 0; j < d; j++) {
                norm += aq[j] * aq[j];
            }
            norm = sqrtf(norm + 1e-10f);
            for (int j = 0; j < d; j++) {
                q[j] = aq[j] / norm;
            }
        }

        for (int j = 0; j < d; j++) {
            U[j * rank + r] = q[j];
        }
    }

    free(q);
    free(aq);
    free(cov);

    ResidualCacheEntry *e = &rc->entries[layer_idx];
    e->basis = malloc((size_t)d * rank * sizeof(float));
    e->mean = malloc((size_t)d * sizeof(float));
    e->codes = malloc((size_t)n_seqs * seq_len * rank * sizeof(int8_t));
    e->scales = malloc((size_t)n_seqs * seq_len * sizeof(float));

    if (!e->basis || !e->mean || !e->codes || !e->scales) {
        fprintf(stderr, "[PCA] OOM allocating cache entry\n");
        free(e->basis);
        free(e->mean);
        free(e->codes);
        free(e->scales);
        free(U);
        free(mean);
        free(c->residual);
        free(c->x_proj);
        free(c->x_conv);
        free(c->x_gated);
        free(c->mask_val);
        free(c->b_eff_gate);
        free(c->y_ssm);
        free(c->h_all);
        free(c->norm_out);
        free(c->norm_rms);
        free(c);
        return -1;
    }

    memcpy(e->basis, U, (size_t)d * rank * sizeof(float));
    memcpy(e->mean, mean, (size_t)d * sizeof(float));

    dataset_reset_cursor(ds);

    float total_mse = 0.0f;
    int total_n = 0;

    for (int s = 0; s < n_seqs; s++) {
        uint16_t tokens[V12_SEQ_LEN];
        dataset_next_sequential(ds, tokens, 1, seq_len);
        forward_pass_frozen_prefix(m, c, tokens, seq_len, layer_idx);

        float *r_out = c->residual + (size_t)(layer_idx + 1) * seq_len * d;

        if (s % 50 == 0 || r_out[0] != r_out[0]) {
            fprintf(stderr, "[PCA-DBG] compress seq=%d/%d layer=%d r_out[0]=%e\n",
                    s, n_seqs, layer_idx, r_out[0]);
            fflush(stderr);
        }

        for (int t = 0; t < seq_len; t++) {
            float centered[V12_D_MODEL];
            for (int j = 0; j < d; j++) {
                centered[j] = r_out[t * d + j] - mean[j];
            }

            float coeffs[V12_PCA_CACHE_RANK];
            float abs_max = 0.0f;
            for (int r = 0; r < rank; r++) {
                float dot = 0.0f;
                for (int j = 0; j < d; j++) {
                    dot += U[j * rank + r] * centered[j];
                }
                coeffs[r] = dot;
                if (fabsf(dot) > abs_max) {
                    abs_max = fabsf(dot);
                }
            }

            float scale = (abs_max < 1e-12f) ? 1.0f : abs_max;
            e->scales[s * seq_len + t] = scale;

            int8_t *dst = e->codes + ((size_t)s * seq_len + t) * rank;
            float inv_scale = 127.0f / scale;
            for (int r = 0; r < rank; r++) {
                float q = coeffs[r] * inv_scale;
                if (q > 127.0f) q = 127.0f;
                if (q < -127.0f) q = -127.0f;
                dst[r] = (int8_t)(q >= 0.0f ? (int)(q + 0.5f) : (int)(q - 0.5f));
            }

            float reconstructed[V12_D_MODEL];
            for (int j = 0; j < d; j++) {
                reconstructed[j] = mean[j];
                for (int r = 0; r < rank; r++) {
                    reconstructed[j] += U[j * rank + r] * coeffs[r];
                }
            }

            for (int j = 0; j < d; j++) {
                float err = r_out[t * d + j] - reconstructed[j];
                total_mse += err * err;
            }
            total_n++;
        }
    }

    fprintf(stderr, "[PCA-DBG] compression done. total_n=%d total_mse=%e\n", total_n, total_mse);
    fflush(stderr);
    if (total_n > 0) {
        total_mse /= (float)total_n;
        printf("[PCA] Layer %d: MSE reconstruction=%.6f\n", layer_idx, total_mse);
    }
    fprintf(stderr, "[PCA-DBG] setting metadata and cleaning up\n"); fflush(stderr);

    e->layer_idx = layer_idx;
    e->n_seqs = n_seqs;
    e->seq_len = seq_len;
    e->d_model = d;
    e->calibrated = 1;

    if (layer_idx >= rc->active_entry) {
        rc->active_entry = layer_idx;
    }

    printf("[PCA] Built cache for layer %d: %d seqs × %d pos × rank %d = %zu KB\n",
           layer_idx, n_seqs, seq_len, rank,
           ((size_t)n_seqs * seq_len * (rank + 4) / 1024));

    free(U);
    free(mean);
    free(c->residual);
    free(c->x_proj);
    free(c->x_conv);
    free(c->x_gated);
    free(c->mask_val);
    free(c->b_eff_gate);
    free(c->y_ssm);
    free(c->h_all);
    free(c->norm_out);
    free(c->norm_rms);
    free(c);

    return 0;
}

void pca_cache_load_entry(const ResidualCacheEntry *entry,
                          int seq_idx, int t,
                          float *out_buf, int d_model, int rank) {
    if (!entry || !out_buf || entry->seq_len == 0) {
        return;
    }

    float scale = entry->scales[seq_idx * entry->seq_len + t];
    const int8_t *codes_t = entry->codes + ((size_t)seq_idx * entry->seq_len + t) * rank;
    float inv_scale = scale / 127.0f;

    for (int j = 0; j < d_model && j < entry->d_model; j++) {
        float acc = entry->mean[j];
        for (int r = 0; r < rank; r++) {
            acc += entry->basis[j * rank + r] * (float)codes_t[r] * inv_scale;
        }
        out_buf[j] = acc;
    }
}
