#include "y_cache.h"
#include "model.h"
#include "ternary_matmul.h"
#include "ssm_scan.h"
#include "../io/dataset.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

YCache *y_cache_alloc(void) {
    YCache *yc = calloc(1, sizeof(YCache));
    if (!yc) return NULL;
    yc->active_entry = -1;
    for (int i = 0; i < V12_MAX_LAYERS; i++) {
        yc->entries[i].calibrated = 0;
    }
    return yc;
}

void y_cache_free(YCache *yc) {
    if (!yc) return;
    for (int i = 0; i < V12_MAX_LAYERS; i++) {
        free(yc->entries[i].y);
        free(yc->entries[i].r0);
    }
    free(yc);
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

int y_cache_build_entry(Model *m, YCache *yc, Dataset *ds,
                        int layer_idx, int seq_len, int n_seqs) {
    if (!m || !yc || !ds || layer_idx < 0 || layer_idx >= V12_N_LAYERS) {
        return -1;
    }

    int d = V12_D_MODEL;
    int ds_val = V12_D_STATE;

    Cache *c = calloc(1, sizeof(Cache));
    if (!c) return -1;

    size_t residual_size = (size_t)(V12_N_LAYERS + 1) * seq_len * d;
    c->residual = calloc(residual_size, sizeof(float));
#if V12_AF8C_ENABLED
    c->x_proj = malloc((size_t)V12_N_LAYERS * seq_len * ds_val * sizeof(uint8_t));
    c->x_conv = malloc((size_t)V12_N_LAYERS * seq_len * ds_val * sizeof(uint8_t));
    c->x_gated = malloc((size_t)V12_N_LAYERS * seq_len * ds_val * sizeof(uint8_t));
    c->mask_val = malloc((size_t)V12_N_LAYERS * seq_len * ds_val * sizeof(uint8_t));
    c->b_eff_gate = malloc((size_t)V12_N_LAYERS * seq_len * ds_val * sizeof(uint8_t));
    c->y_ssm = malloc((size_t)V12_N_LAYERS * seq_len * ds_val * sizeof(uint8_t));
#else
    c->x_proj = malloc((size_t)V12_N_LAYERS * seq_len * ds_val * sizeof(float));
    c->x_conv = malloc((size_t)V12_N_LAYERS * seq_len * ds_val * sizeof(float));
    c->x_gated = malloc((size_t)V12_N_LAYERS * seq_len * ds_val * sizeof(float));
    c->mask_val = malloc((size_t)V12_N_LAYERS * seq_len * ds_val * sizeof(float));
    c->b_eff_gate = malloc((size_t)V12_N_LAYERS * seq_len * ds_val * sizeof(float));
    c->y_ssm = malloc((size_t)V12_N_LAYERS * seq_len * ds_val * sizeof(float));
#endif
    c->h_all = malloc((size_t)V12_N_LAYERS * seq_len * ds_val * sizeof(uint8_t));
    c->norm_out = malloc((size_t)seq_len * d * sizeof(float));
    c->norm_rms = malloc((size_t)seq_len * sizeof(float));
    c->res_cache = NULL;
    c->y_cache = NULL;

    if (!c->residual || !c->x_proj || !c->x_conv || !c->x_gated ||
        !c->mask_val || !c->b_eff_gate || !c->y_ssm || !c->h_all ||
        !c->norm_out || !c->norm_rms) {
        fprintf(stderr, "[Y-CACHE] OOM allocating temp cache\n");
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

    dataset_reset_cursor(ds);

    YCacheEntry *entry = &yc->entries[layer_idx];
    entry->y = malloc((size_t)n_seqs * seq_len * ds_val * sizeof(float));
    entry->r0 = malloc((size_t)n_seqs * seq_len * d * sizeof(float));

    if (!entry->y || !entry->r0) {
        fprintf(stderr, "[Y-CACHE] OOM allocating cache entry\n");
        free(entry->y);
        free(entry->r0);
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

    dataset_reset_cursor(ds);

    for (int s = 0; s < n_seqs; s++) {
        uint16_t tokens[V12_SEQ_LEN];
        dataset_next_sequential(ds, tokens, 1, seq_len);
        forward_pass_frozen_prefix(m, c, tokens, seq_len, layer_idx);

        float *r0_src = c->residual + (size_t)0 * seq_len * d;
        float *r0_dst = entry->r0 + (size_t)s * seq_len * d;
        memcpy(r0_dst, r0_src, (size_t)seq_len * d * sizeof(float));

#if V12_AF8C_ENABLED
        uint8_t *y_src = c->y_ssm + (size_t)layer_idx * seq_len * ds_val;
        float *y_dst = entry->y + (size_t)s * seq_len * ds_val;
        for (size_t i = 0; i < (size_t)seq_len * ds_val; i++) {
            y_dst[i] = decompress_e5m2(y_src[i]);
        }
#else
        float *y_src = c->y_ssm + (size_t)layer_idx * seq_len * ds_val;
        float *y_dst = entry->y + (size_t)s * seq_len * ds_val;
        memcpy(y_dst, y_src, (size_t)seq_len * ds_val * sizeof(float));
#endif

        if (s % 50 == 0) {
            fprintf(stderr, "[Y-CACHE] Built seq=%d/%d layer=%d\n",
                    s, n_seqs, layer_idx);
            fflush(stderr);
        }
    }

    entry->layer_idx = layer_idx;
    entry->n_seqs = n_seqs;
    entry->seq_len = seq_len;
    entry->d_state = ds_val;
    entry->d_model = d;
    entry->calibrated = 1;

    yc->active_entry = layer_idx;

    size_t y_size_mb = (size_t)n_seqs * seq_len * ds_val * sizeof(float) / 1048576;
    size_t r0_size_mb = (size_t)n_seqs * seq_len * d * sizeof(float) / 1048576;

    printf("[Y-CACHE] Built layer %d: %d seqs × %d pos × d_state=%d"
           " y=%.1f MB r0=%.1f MB\n",
           layer_idx, n_seqs, seq_len, ds_val,
           (float)y_size_mb,
           (float)r0_size_mb);

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

void y_cache_load_entry(const YCache *yc, const Model *m,
                        int seq_idx, int t,
                        float *out_buf, int d_model) {
    if (!yc || !m || !out_buf) return;

    int active = yc->active_entry;
    if (active < 0) return;

    const YCacheEntry *ye = &yc->entries[active];
    int K = active + 1;
    int ds = ye->d_state;

    const float *r0 = ye->r0 + ((size_t)seq_idx * ye->seq_len + t) * d_model;
    memcpy(out_buf, r0, d_model * sizeof(float));

    float tmp[V12_D_MODEL];
    for (int k = 0; k < K; k++) {
        const YCacheEntry *yk_entry = &yc->entries[k];
        if (!yk_entry->calibrated) continue;

        const float *y_kt = yk_entry->y + ((size_t)seq_idx * yk_entry->seq_len + t) * ds;
        ternary_matvec(&m->layers[k].csr_out, y_kt, tmp);
        for (int j = 0; j < d_model; j++) {
            out_buf[j] += tmp[j];
        }
    }
}
