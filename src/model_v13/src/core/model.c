/*
 * model.c — Model and cache memory management
 *
 * Implements allocation and deallocation for Model and Cache structs.
 * All allocations use malloc(...) for SIMD-friendly access,
 * or calloc(...) for gradient accumulators (no SIMD requirements).
 *
 * Error handling: On failure, the alloc functions return NULL after
 * freeing any partial allocations. Callers must check for NULL.
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include "types.h"
#include "ternary_matmul.h"
#include "model.h"
#include "optimizer.h"

/* Forward declarations */
void model_free(Model *m);
void cache_free(Cache *c);


/* =========================================================================
 * rns_embed_init_E — Initialize RNS embedding table for all tokens.
 *
 * For each token t and dimension j (over d_embed space):
 *   group = j / dims_per_mod
 *   m     = RNS_MODULI[group]
 *   r     = t % m
 *   embed_E[t * d_embed + j] = (2.0f * r / (m - 1)) - 1.0f * scale
 *
 * Writes to: float *embed_E, shape [vocab_size × d_embed]
 * Scale normalizes by d_embed (not d_model).
 * ========================================================================= */
static void rns_embed_init_E(float *embed_E) {
    static const int RNS_MODULI[8] = {5, 7, 8, 9, 11, 13, 17, 19};
    const int dims_per_mod = V12_D_EMBED / 8;

    for (int t = 0; t < V12_VOCAB_SIZE; t++) {
        for (int j = 0; j < V12_D_EMBED; j++) {
            int group = j / dims_per_mod;
            int m     = RNS_MODULI[group];
            int r     = t % m;
            float raw = (2.0f * (float)r / (float)(m - 1)) - 1.0f;
            embed_E[t * V12_D_EMBED + j] = raw * (1.0f / sqrtf((float)V12_D_EMBED));
        }
    }
}


/* =========================================================================
 * model_build_hot_mask — Classify tokens as hot (top-K) or cold
 *
 * Counts raw token frequencies and marks the top K_HOT as hot.
 * Builds three lookup structures: hot_mask, vocab_to_hot, hot_to_vocab.
 * Called after sampled_softmax_build_unigram during model_alloc.
 * ========================================================================= */

static int cv_cmp_desc(const void *a, const void *b) {
    typedef struct { uint32_t cnt; int32_t v; } CV;
    const CV *ca = (const CV *)a;
    const CV *cb = (const CV *)b;
    if (ca->cnt != cb->cnt) return (ca->cnt > cb->cnt) ? -1 : 1;
    return ca->v - cb->v;
}

void model_build_hot_mask(Model *m, const uint16_t *tokens, size_t n_toks) {
    typedef struct { uint32_t cnt; int32_t v; } CV;

    uint32_t *counts = (uint32_t *)calloc(V12_VOCAB_SIZE, sizeof(uint32_t));
    if (!counts) {
        fprintf(stderr, "[HOT-COLD] count alloc failed\n");
        return;
    }

    for (size_t i = 0; i < n_toks; i++) {
        if (tokens[i] < V12_VOCAB_SIZE) counts[tokens[i]]++;
    }

    CV *pairs = (CV *)malloc(V12_VOCAB_SIZE * sizeof(CV));
    if (!pairs) {
        fprintf(stderr, "[HOT-COLD] pair alloc failed\n");
        free(counts);
        return;
    }

    for (int v = 0; v < V12_VOCAB_SIZE; v++) {
        pairs[v].cnt = counts[v];
        pairs[v].v   = v;
    }

    qsort(pairs, V12_VOCAB_SIZE, sizeof(CV), cv_cmp_desc);

    memset(m->hot_mask, 0, V12_VOCAB_SIZE);
    for (int h = 0; h < V12_K_HOT; h++) {
        int32_t v = pairs[h].v;
        m->hot_mask[v]     = 1;
        m->vocab_to_hot[v] = h;
        m->hot_to_vocab[h] = v;
    }

    free(pairs);
    free(counts);

    printf("[HOT-COLD] built: K_HOT=%d (%.2f%% of vocab)\n",
           V12_K_HOT, 100.0f * V12_K_HOT / V12_VOCAB_SIZE);
}


/* Tight frame init: 128 orthogonal rows in R^1024 via Gram-Schmidt.
 * Maximizes I(X; WX) by equalizing eigenvalues of WW^T.
 * R352 F1+F5 CONFIRMED: 10×+ more independent info vs random. */
static void tight_frame_init(shadow_t *shadow, int n_rows, int n_cols, uint64_t seed) {
    float *buf = (float *)malloc((size_t)n_rows * n_cols * sizeof(float));
    if (!buf) return;  /* fallback: shadow stays at calloc zeros */

    /* Step 1: Random Gaussian fill (Box-Muller from xorshift) */
    uint64_t rng = seed;
    for (int i = 0; i < n_rows * n_cols; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        float u1 = ((float)(rng & 0xFFFF) + 1.0f) / 65537.0f;
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        float u2 = ((float)(rng & 0xFFFF)) / 65536.0f;
        buf[i] = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    }

    /* Step 2: Modified Gram-Schmidt orthogonalization */
    for (int i = 0; i < n_rows; i++) {
        float *row_i = buf + (size_t)i * n_cols;
        for (int j = 0; j < i; j++) {
            float *row_j = buf + (size_t)j * n_cols;
            float dot = 0.0f, norm_j = 0.0f;
            for (int c = 0; c < n_cols; c++) {
                dot += row_i[c] * row_j[c];
                norm_j += row_j[c] * row_j[c];
            }
            if (norm_j > 1e-12f) {
                float s = dot / norm_j;
                for (int c = 0; c < n_cols; c++)
                    row_i[c] -= s * row_j[c];
            }
        }
    }

    /* Step 3: Normalize + scale to shadow logit range [-0.6, 0.6] */
    for (int i = 0; i < n_rows; i++) {
        float *row_i = buf + (size_t)i * n_cols;
        float norm = 0.0f;
        for (int c = 0; c < n_cols; c++) norm += row_i[c] * row_i[c];
        norm = sqrtf(norm + 1e-12f);

        float max_abs = 0.0f;
        for (int c = 0; c < n_cols; c++) {
            float a = fabsf(row_i[c] / norm);
            if (a > max_abs) max_abs = a;
        }

        float final_scale = 0.6f / (max_abs + 1e-12f);
        for (int c = 0; c < n_cols; c++) {
            float v = (row_i[c] / norm) * final_scale;
#if V12_F2_INT8_SHADOWS
            uint32_t r = shadow_rng(0, V12_SHADOW_MAT_W_IN, i * n_cols + c, seed);
            SHADOW_STORE(shadow[i * n_cols + c], v, r);
#else
            shadow[i * n_cols + c] = v;
#endif
        }
    }

    free(buf);
}

static void init_depthwise_conv_identity(shadow_t *shadow, int layer_id) {
    const int n_conv = V12_CONV_KERNEL * V12_D_STATE;
    const float identity_scale = 0.01f;
    for (int i = 0; i < n_conv; i++) {
        const int tap = i / V12_D_STATE;
        const float v = (tap == 0) ? identity_scale : 0.0f;
#if V12_F2_INT8_SHADOWS
        uint32_t r = shadow_rng(layer_id, V12_SHADOW_MAT_W_CONV, i, 0ULL);
        SHADOW_STORE(shadow[i], v, r);
#else
        shadow[i] = v;
#endif
    }
}

/* =========================================================================
 * model_alloc — Complete Model allocation for training mode
 * ========================================================================= */

Model *model_alloc(void) {
    Model *m = calloc(1, sizeof(Model));
    if (!m) return NULL;

    m->n_layers = V12_N_LAYERS;
    m->d_model = V12_D_MODEL;
    m->d_state = V12_D_STATE;
    m->vocab_size = V12_VOCAB_SIZE;
    m->conv_kernel = V12_CONV_KERNEL;
    m->mtp_rank = V12_MTP_RANK;
    m->step = 0;

    /* Residual Shell defaults — OFF by default, sane band values ready to use.
     * Runtime-only; not persisted. See types.h for the full caveat. */
    m->use_residual_shell = 0;
    m->rs_eps_lo          = 0.5f;
    m->rs_eps_hi          = 15.0f;
    m->exact_quotient_padding_enabled = 0;
    m->exact_quotient_padding_min_active_len = 0u;
    m->exact_quotient_padding_seed = 0x9e3779b97f4a7c15ull;

    /* --- Factored embedding (ALBERT-style) -------------------------------- */
    /* embed_E: RNS-initialized, [vocab_size × d_embed]                      */
    /* embed_P: projection, [d_embed × d_model]                              */
    /* embed_int2: materialized projected vector, INT2-packed, for inference  */

    m->embed_int2 = malloc(V12_VOCAB_SIZE * V12_D_MODEL / V12_INT2_WEIGHTS_PER_BYTE);
    if (!m->embed_int2) { model_free(m); return NULL; }
    m->csr_embed = NULL;

    m->embed_E = malloc((size_t)V12_VOCAB_SIZE * V12_D_EMBED * sizeof(float));
    if (!m->embed_E) { model_free(m); return NULL; }

    m->embed_P = malloc((size_t)V12_D_EMBED * V12_D_MODEL * sizeof(float));
    if (!m->embed_P) { model_free(m); return NULL; }

    m->d_embed_E = calloc((size_t)V12_VOCAB_SIZE * V12_D_EMBED, sizeof(float));
    if (!m->d_embed_E) { model_free(m); return NULL; }

    m->d_embed_P = calloc((size_t)V12_D_EMBED * V12_D_MODEL, sizeof(float));
    if (!m->d_embed_P) { model_free(m); return NULL; }

    m->sa_embed_E = calloc((size_t)V12_VOCAB_SIZE * V12_D_EMBED, sizeof(int8_t));
    if (!m->sa_embed_E) { model_free(m); return NULL; }

    m->m_embed_E_row = calloc(V12_VOCAB_SIZE, sizeof(float));
    if (!m->m_embed_E_row) { model_free(m); return NULL; }

    m->m_embed_E_col = calloc(V12_D_EMBED, sizeof(float));
    if (!m->m_embed_E_col) { model_free(m); return NULL; }

    m->v_embed_E_row = calloc(V12_VOCAB_SIZE, sizeof(float));
    if (!m->v_embed_E_row) { model_free(m); return NULL; }

    m->v_embed_E_col = calloc(V12_D_EMBED, sizeof(float));
    if (!m->v_embed_E_col) { model_free(m); return NULL; }

    m->m_embed_P = calloc((size_t)V12_D_EMBED * V12_D_MODEL, sizeof(float));
    if (!m->m_embed_P) { model_free(m); return NULL; }

    m->v_embed_P = calloc((size_t)V12_D_EMBED * V12_D_MODEL, sizeof(float));
    if (!m->v_embed_P) { model_free(m); return NULL; }

    rns_embed_init_E(m->embed_E);

    /* Initialize embed_P with Xavier uniform */
    {
        uint64_t rng = 0xDEADBEEFCAFE1234ULL;
        float scale = sqrtf(6.0f / (V12_D_EMBED + V12_D_MODEL));
        for (int i = 0; i < V12_D_EMBED * V12_D_MODEL; i++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            float u = (float)(rng >> 11) * (1.0f / (float)(1ULL << 53));
            m->embed_P[i] = (2.0f * u - 1.0f) * scale;
        }
    }

    /* Materialize projected float vectors, then pack to INT2 for inference */
    {
        float proj_buf[V12_D_MODEL];
        for (int t = 0; t < V12_VOCAB_SIZE; t++) {
            /* proj_buf = embed_E[t] @ embed_P  (1×128 @ 128×1024 → 1×1024) */
            const float *e = m->embed_E + t * V12_D_EMBED;
            for (int j = 0; j < V12_D_MODEL; j++) {
                float acc = 0.0f;
                for (int r = 0; r < V12_D_EMBED; r++) {
                    acc += e[r] * m->embed_P[r * V12_D_MODEL + j];
                }
                proj_buf[j] = acc;
            }
            /* Pack to INT2 */
            for (int b = 0; b < V12_D_MODEL / V12_INT2_WEIGHTS_PER_BYTE; b++) {
                uint8_t packed = 0;
                for (int k = 0; k < V12_INT2_WEIGHTS_PER_BYTE; k++) {
                    int j = b * V12_INT2_WEIGHTS_PER_BYTE + k;
                    float v = proj_buf[j];
                    uint8_t bits;
                    if      (v >  0.33f) bits = 0x01;
                    else if (v < -0.33f) bits = 0x03;
                    else                  bits = 0x00;
                    packed |= (bits << (k * 2));
                }
                m->embed_int2[t * (V12_D_MODEL / V12_INT2_WEIGHTS_PER_BYTE) + b] = packed;
            }
        }
    }

    /* --- lm_head (hot/cold split) ----------------------------------------- */
    m->lm_head_int2 = malloc(V12_D_MODEL * V12_VOCAB_SIZE / V12_INT2_WEIGHTS_PER_BYTE);
    if (!m->lm_head_int2) { model_free(m); return NULL; }
    memset(&m->csr_lm_head, 0, sizeof(TernaryCSR));

    /* Hot/cold mask allocation (filled after unigram build) */
    m->hot_mask = calloc(V12_VOCAB_SIZE, sizeof(uint8_t));
    if (!m->hot_mask) { model_free(m); return NULL; }
    m->vocab_to_hot = malloc(V12_VOCAB_SIZE * sizeof(int32_t));
    if (!m->vocab_to_hot) { model_free(m); return NULL; }
    memset(m->vocab_to_hot, -1, V12_VOCAB_SIZE * sizeof(int32_t));
    m->hot_to_vocab = malloc(V12_K_HOT * sizeof(int32_t));
    if (!m->hot_to_vocab) { model_free(m); return NULL; }

    /* --- Final layer norm ------------------------------------------------- */
    m->norm_w = calloc(V12_D_MODEL, sizeof(float));
    if (!m->norm_w) { model_free(m); return NULL; }

    m->norm_b = calloc(V12_D_MODEL, sizeof(float));
    if (!m->norm_b) { model_free(m); return NULL; }

    m->d_norm_w = calloc(V12_D_MODEL, sizeof(float));
    if (!m->d_norm_w) { model_free(m); return NULL; }

    m->d_norm_b = calloc(V12_D_MODEL, sizeof(float));
    if (!m->d_norm_b) { model_free(m); return NULL; }

    for (int j = 0; j < V12_D_MODEL; j++) {
        m->norm_w[j] = 1.0f;
    }

    /* --- Predictive coding W_pred ----------------------------------------- */
    m->w_pred_int2 = malloc(V12_D_MODEL * V12_D_MODEL / V12_INT2_WEIGHTS_PER_BYTE);
    if (!m->w_pred_int2) { model_free(m); return NULL; }
    memset(&m->csr_w_pred, 0, sizeof(TernaryCSR));

    m->shadow_w_pred = calloc(V12_D_MODEL * V12_D_MODEL, SHADOW_SIZEOF);
    if (!m->shadow_w_pred) { model_free(m); return NULL; }

    m->dw_pred = calloc(V12_D_MODEL * V12_D_MODEL, sizeof(float));
    if (!m->dw_pred) { model_free(m); return NULL; }

    m->crystal_mask_pred = calloc(V12_D_MODEL * V12_D_MODEL, sizeof(uint8_t));
    if (!m->crystal_mask_pred) { model_free(m); return NULL; }

    m->sampled_w_pred = calloc((size_t)V12_D_MODEL * V12_D_MODEL, sizeof(float));
    if (!m->sampled_w_pred) { model_free(m); return NULL; }

    /* === ADAM MOMENT BUFFERS — Model level ================================= */

    /* Factored Adam moments — lm_head (hot tokens only after split)
     * m/v_lm_head_row: [K_HOT] (first/second moment per hot token index)
     * m/v_lm_head_col: [d_model] (first/second moment per d_model dimension) */
#if !V12_TOLR_ENABLED
    m->m_lm_head_row = calloc(V12_K_HOT, sizeof(float));
    if (!m->m_lm_head_row) { model_free(m); return NULL; }
    m->m_lm_head_col = calloc(V12_D_MODEL, sizeof(float));
    if (!m->m_lm_head_col) { model_free(m); return NULL; }
    m->v_lm_head_row = calloc(V12_K_HOT, sizeof(float));
    if (!m->v_lm_head_row) { model_free(m); return NULL; }
    m->v_lm_head_col = calloc(V12_D_MODEL, sizeof(float));
    if (!m->v_lm_head_col) { model_free(m); return NULL; }

    /* Direct Factored Gradient Accumulation (DF-GA) arrays */
    m->dlm_head_g_row = calloc(V12_K_HOT, sizeof(float));
    if (!m->dlm_head_g_row) { model_free(m); return NULL; }
    m->dlm_head_g_col = calloc(V12_D_MODEL, sizeof(float));
    if (!m->dlm_head_g_col) { model_free(m); return NULL; }
    m->dlm_head_gq_row = calloc(V12_K_HOT, sizeof(float));
    if (!m->dlm_head_gq_row) { model_free(m); return NULL; }
    m->dlm_head_gq_col = calloc(V12_D_MODEL, sizeof(float));
    if (!m->dlm_head_gq_col) { model_free(m); return NULL; }
    m->dlm_head_cnt_row = calloc(V12_K_HOT, sizeof(int32_t));
    if (!m->dlm_head_cnt_row) { model_free(m); return NULL; }
    m->dlm_head_cnt_col = calloc(V12_D_MODEL, sizeof(int32_t));
    if (!m->dlm_head_cnt_col) { model_free(m); return NULL; }
#endif

    /* Factored Adam moments — w_pred */
    m->m_w_pred_row = calloc(V12_D_MODEL, sizeof(float));
    if (!m->m_w_pred_row) { model_free(m); return NULL; }
    m->m_w_pred_col = calloc(V12_D_MODEL, sizeof(float));
    if (!m->m_w_pred_col) { model_free(m); return NULL; }
    m->v_w_pred_row = calloc(V12_D_MODEL, sizeof(float));
    if (!m->v_w_pred_row) { model_free(m); return NULL; }
    m->v_w_pred_col = calloc(V12_D_MODEL, sizeof(float));
    if (!m->v_w_pred_col) { model_free(m); return NULL; }

    m->m_norm_w = calloc(V12_D_MODEL, sizeof(float));
    if (!m->m_norm_w) { model_free(m); return NULL; }
    m->v_norm_w = calloc(V12_D_MODEL, sizeof(float));
    if (!m->v_norm_w) { model_free(m); return NULL; }
    m->m_norm_b = calloc(V12_D_MODEL, sizeof(float));
    if (!m->m_norm_b) { model_free(m); return NULL; }
    m->v_norm_b = calloc(V12_D_MODEL, sizeof(float));
    if (!m->v_norm_b) { model_free(m); return NULL; }

    /* --- tau_mux (ternary multiplexing threshold) ------------------------- */
    for (int k = 0; k < V12_N_LAYERS; k++) {
        m->tau_mux[k] = 0.1f;
        m->d_tau_mux[k] = 0.0f;
    }

    /* --- Per-layer allocations -------------------------------------------- */
    for (int k = 0; k < V12_N_LAYERS; k++) {
        SSMLayer *L = &m->layers[k];

        /* INT2 packed weights */
        L->w_in_int2 = malloc(V12_D_MODEL * V12_D_STATE / V12_INT2_WEIGHTS_PER_BYTE);
        if (!L->w_in_int2) { model_free(m); return NULL; }

        L->w_out_int2 = malloc(V12_D_STATE * V12_D_MODEL / V12_INT2_WEIGHTS_PER_BYTE);
        if (!L->w_out_int2) { model_free(m); return NULL; }

        L->w_conv_int2 = malloc(V12_CONV_KERNEL * V12_D_STATE / V12_INT2_WEIGHTS_PER_BYTE);
        if (!L->w_conv_int2) { model_free(m); return NULL; }

        memset(&L->csr_in, 0, sizeof(TernaryCSR));
        memset(&L->csr_out, 0, sizeof(TernaryCSR));

        /* SSM dynamics */
        L->A = malloc(V12_D_STATE * sizeof(float));
        if (!L->A) { model_free(m); return NULL; }

        L->B = calloc(V12_D_STATE, sizeof(float));
        if (!L->B) { model_free(m); return NULL; }

        L->C = malloc(V12_D_STATE * V12_D_STATE * sizeof(float));
        if (!L->C) { model_free(m); return NULL; }

        for (int i = 0; i < V12_D_STATE; i++) {
            float tau = 1.0f + (512.0f - 1.0f) * (float)i / (float)(V12_D_STATE - 1);
            L->A[i] = expf(-1.0f / tau);
            L->B[i] = sqrtf(1.0f - L->A[i]);
        }
        memset(L->C, 0, V12_D_STATE * V12_D_STATE * sizeof(float));
        for (int i = 0; i < V12_D_STATE; i++) {
            L->C[i * V12_D_STATE + i] = 1.0f;
        }

        /* Output scaling vector (weight tying) */
        L->sigma = malloc(V12_D_STATE * sizeof(float));
        if (!L->sigma) { model_free(m); return NULL; }
        for (int i = 0; i < V12_D_STATE; i++) L->sigma[i] = 1.0f;

        L->d_sigma = calloc(V12_D_STATE, sizeof(float));
        if (!L->d_sigma) { model_free(m); return NULL; }

        L->m_sigma = calloc(V12_D_STATE, sizeof(float));
        if (!L->m_sigma) { model_free(m); return NULL; }
        L->v_sigma = calloc(V12_D_STATE, sizeof(float));
        if (!L->v_sigma) { model_free(m); return NULL; }

        /* MTP matrices (only on layers 0, 2, 4) */
        if (k % 2 == 0) {
            L->mtp_U = calloc(V12_D_STATE * V12_MTP_RANK, sizeof(float));
            if (!L->mtp_U) { model_free(m); return NULL; }

            L->mtp_V = calloc(V12_D_STATE * V12_MTP_RANK, sizeof(float));
            if (!L->mtp_V) { model_free(m); return NULL; }

            L->d_mtp_U = calloc((size_t)V12_D_STATE * V12_MTP_RANK, sizeof(float));
            if (!L->d_mtp_U) { model_free(m); return NULL; }

            L->d_mtp_V = calloc((size_t)V12_D_STATE * V12_MTP_RANK, sizeof(float));
            if (!L->d_mtp_V) { model_free(m); return NULL; }
        } else {
            L->mtp_U = NULL;
            L->mtp_V = NULL;
            L->d_mtp_U = NULL;
            L->d_mtp_V = NULL;
        }

        /* Shadow weights — R356 F2: shadow_t / SHADOW_SIZEOF select fp32 or int16 */
        L->shadow_w_in = malloc(V12_D_MODEL * V12_D_STATE * SHADOW_SIZEOF);
        if (!L->shadow_w_in) { model_free(m); return NULL; }

        L->shadow_w_out = malloc(V12_D_STATE * V12_D_MODEL * SHADOW_SIZEOF);
        if (!L->shadow_w_out) { model_free(m); return NULL; }

        L->shadow_w_conv = calloc(V12_CONV_KERNEL * V12_D_STATE, SHADOW_SIZEOF);
        if (!L->shadow_w_conv) { model_free(m); return NULL; }
        init_depthwise_conv_identity(L->shadow_w_conv, k);

        /* Input-dependent A gate: W_A [d_state × d_state] and b_A [d_state] */
        L->W_A = malloc(V12_D_STATE * V12_D_STATE * sizeof(float));
        if (!L->W_A) { model_free(m); return NULL; }
        memset(L->W_A, 0, V12_D_STATE * V12_D_STATE * sizeof(float));

        L->b_A = calloc(V12_D_STATE, sizeof(float));
        if (!L->b_A) { model_free(m); return NULL; }
        for (int i = 0; i < V12_D_STATE; i++) {
            L->b_A[i] = 2.0f;
        }

        L->shadow_w_A = malloc(V12_D_STATE * V12_D_STATE * SHADOW_SIZEOF);
        if (!L->shadow_w_A) { model_free(m); return NULL; }
        memset(L->shadow_w_A, 0, V12_D_STATE * V12_D_STATE * SHADOW_SIZEOF);

        L->dW_A = calloc(V12_D_STATE * V12_D_STATE, sizeof(float));
        if (!L->dW_A) { model_free(m); return NULL; }

        L->db_A = calloc(V12_D_STATE, sizeof(float));
        if (!L->db_A) { model_free(m); return NULL; }

        L->m_W_A = calloc(V12_D_STATE * V12_D_STATE, sizeof(float));
        if (!L->m_W_A) { model_free(m); return NULL; }

        L->v_W_A = calloc(V12_D_STATE * V12_D_STATE, sizeof(float));
        if (!L->v_W_A) { model_free(m); return NULL; }

        L->m_b_A = calloc(V12_D_STATE, sizeof(float));
        if (!L->m_b_A) { model_free(m); return NULL; }

        L->v_b_A = calloc(V12_D_STATE, sizeof(float));
        if (!L->v_b_A) { model_free(m); return NULL; }

        tight_frame_init(L->shadow_w_in, V12_D_STATE, V12_D_MODEL, 12345ULL + (uint64_t)k * 17);

        uint64_t rng_w_out = 54321ULL + (uint64_t)k * 31;
        for (size_t i = 0; i < (size_t)V12_D_STATE * V12_D_MODEL; i++) {
            rng_w_out ^= rng_w_out << 13;
            rng_w_out ^= rng_w_out >> 7;
            rng_w_out ^= rng_w_out << 17;
            float v = ((float)(rng_w_out & 0xFFFF) / 65536.0f - 0.5f) * 0.6f;
#if V12_F2_INT8_SHADOWS
            uint32_t r = shadow_rng(k, V12_SHADOW_MAT_W_OUT, (int)i, 0ULL);
            SHADOW_STORE(L->shadow_w_out[i], v, r);
#else
            L->shadow_w_out[i] = v;
#endif
        }

        /* Sampled ternary weights */
        L->sampled_w_in = malloc(V12_D_MODEL * V12_D_STATE * sizeof(float));
        if (!L->sampled_w_in) { model_free(m); return NULL; }
        memset(L->sampled_w_in, 0, V12_D_MODEL * V12_D_STATE * sizeof(float));

        L->sampled_w_out = malloc(V12_D_STATE * V12_D_MODEL * sizeof(float));
        if (!L->sampled_w_out) { model_free(m); return NULL; }
        memset(L->sampled_w_out, 0, V12_D_STATE * V12_D_MODEL * sizeof(float));

        /* Crystallization masks */
        L->crystal_mask_in = calloc(V12_D_MODEL * V12_D_STATE, sizeof(uint8_t));
        if (!L->crystal_mask_in) { model_free(m); return NULL; }

        L->crystal_mask_out = calloc(V12_D_STATE * V12_D_MODEL, sizeof(uint8_t));
        if (!L->crystal_mask_out) { model_free(m); return NULL; }

        /* EMA-SPRT gradient accumulators */
        L->grad_accum_in = calloc(V12_D_MODEL * V12_D_STATE, sizeof(float));
        if (!L->grad_accum_in) { model_free(m); return NULL; }

        L->grad_accum_out = calloc(V12_D_STATE * V12_D_MODEL, sizeof(float));
        if (!L->grad_accum_out) { model_free(m); return NULL; }

        /* Gradient accumulators */
        L->dw_in = calloc(V12_D_MODEL * V12_D_STATE, sizeof(float));
        if (!L->dw_in) { model_free(m); return NULL; }

        L->dw_out = calloc(V12_D_STATE * V12_D_MODEL, sizeof(float));
        if (!L->dw_out) { model_free(m); return NULL; }

        L->dA = calloc(V12_D_STATE, sizeof(float));
        if (!L->dA) { model_free(m); return NULL; }

        L->dB = calloc(V12_D_STATE, sizeof(float));
        if (!L->dB) { model_free(m); return NULL; }

        L->dB_seq = calloc((size_t)V12_SEQ_LEN * V12_D_STATE, sizeof(float));
        if (!L->dB_seq) { model_free(m); return NULL; }

        L->dC = NULL;  /* V-free: C is identity, not trained */

        L->d_conv_w = calloc(V12_CONV_KERNEL * V12_D_STATE, sizeof(float));
        if (!L->d_conv_w) { model_free(m); return NULL; }

        /* === ADAM MOMENT BUFFERS — Per-layer ================================= */

        /* Factored Adam moments — w_in */
        L->m_w_in_row = calloc(V12_D_MODEL, sizeof(float));
        if (!L->m_w_in_row) { model_free(m); return NULL; }
        L->m_w_in_col = calloc(V12_D_STATE, sizeof(float));
        if (!L->m_w_in_col) { model_free(m); return NULL; }
        L->v_w_in_row = calloc(V12_D_MODEL, sizeof(float));
        if (!L->v_w_in_row) { model_free(m); return NULL; }
        L->v_w_in_col = calloc(V12_D_STATE, sizeof(float));
        if (!L->v_w_in_col) { model_free(m); return NULL; }

        /* Factored Adam moments — w_out */
        L->m_w_out_row = calloc(V12_D_STATE, sizeof(float));
        if (!L->m_w_out_row) { model_free(m); return NULL; }
        L->m_w_out_col = calloc(V12_D_MODEL, sizeof(float));
        if (!L->m_w_out_col) { model_free(m); return NULL; }
        L->v_w_out_row = calloc(V12_D_STATE, sizeof(float));
        if (!L->v_w_out_row) { model_free(m); return NULL; }
        L->v_w_out_col = calloc(V12_D_MODEL, sizeof(float));
        if (!L->v_w_out_col) { model_free(m); return NULL; }

        /* Adam moments — A, B */
        L->m_A = calloc(V12_D_STATE, sizeof(float));
        if (!L->m_A) { model_free(m); return NULL; }
        L->v_A = calloc(V12_D_STATE, sizeof(float));
        if (!L->v_A) { model_free(m); return NULL; }
        L->m_B = calloc(V12_D_STATE, sizeof(float));
        if (!L->m_B) { model_free(m); return NULL; }
        L->v_B = calloc(V12_D_STATE, sizeof(float));
        if (!L->v_B) { model_free(m); return NULL; }

        /* Adam moments — C (V-free: not trained, set to NULL) */
        L->m_C = NULL;
        L->v_C = NULL;

        /* Adam moments — conv */
        L->m_conv = calloc(V12_CONV_KERNEL * V12_D_STATE, sizeof(float));
        if (!L->m_conv) { model_free(m); return NULL; }
        L->v_conv = calloc(V12_CONV_KERNEL * V12_D_STATE, sizeof(float));
        if (!L->v_conv) { model_free(m); return NULL; }

        /* Adam moments — MTP (only even layers) */
        if (k % 2 == 0) {
            L->m_mtp_U = calloc((size_t)V12_D_STATE * V12_MTP_RANK, sizeof(float));
            if (!L->m_mtp_U) { model_free(m); return NULL; }
            L->v_mtp_U = calloc((size_t)V12_D_STATE * V12_MTP_RANK, sizeof(float));
            if (!L->v_mtp_U) { model_free(m); return NULL; }
            L->m_mtp_V = calloc((size_t)V12_D_STATE * V12_MTP_RANK, sizeof(float));
            if (!L->m_mtp_V) { model_free(m); return NULL; }
            L->v_mtp_V = calloc((size_t)V12_D_STATE * V12_MTP_RANK, sizeof(float));
            if (!L->v_mtp_V) { model_free(m); return NULL; }
        } else {
            L->m_mtp_U = NULL;
            L->v_mtp_U = NULL;
            L->m_mtp_V = NULL;
            L->v_mtp_V = NULL;
        }

        /* CoSpectralSubspace */
        L->cospec.n_samples = 0;
        L->cospec.calibrated = 0;
        L->cospec.U = NULL;

        L->cospec.cov_accum = calloc(V12_D_STATE * V12_D_STATE, sizeof(float));
        if (!L->cospec.cov_accum) { model_free(m); return NULL; }

        /* GpKlMoments — calloc zeroes it; initialize orthogonal pair indices */
        memset(&L->moments, 0, sizeof(GpKlMoments));

        uint64_t rng_state = 42ULL;
        for (int p = 0; p < 32; p++) {
            L->moments.orth_pair_a[p] = (float)(xorshift64(&rng_state) % V12_D_STATE);
            L->moments.orth_pair_b[p] = (float)(xorshift64(&rng_state) % V12_D_STATE);
        }

        /* Bilinear recurrence gate C_bilinear (R348 F6) — init 0 = no-op at start */
        L->C_bilinear = calloc(V12_D_STATE, sizeof(float));
        if (!L->C_bilinear) { model_free(m); return NULL; }
        L->shadow_C_bilinear = calloc(V12_D_STATE, SHADOW_SIZEOF);
        if (!L->shadow_C_bilinear) { model_free(m); return NULL; }
        L->sampled_C_bilinear = calloc(V12_D_STATE, sizeof(float));
        if (!L->sampled_C_bilinear) { model_free(m); return NULL; }
        L->dC_bilinear = calloc(V12_D_STATE, sizeof(float));
        if (!L->dC_bilinear) { model_free(m); return NULL; }
        L->m_C_bilinear = calloc(V12_D_STATE, sizeof(float));
        if (!L->m_C_bilinear) { model_free(m); return NULL; }
        L->v_C_bilinear = calloc(V12_D_STATE, sizeof(float));
        if (!L->v_C_bilinear) { model_free(m); return NULL; }

        /* Low-Rank Dyadic Perturbations u and v (R495 Aspect A01) */
        L->u = calloc(V12_D_STATE, sizeof(float));
        if (!L->u) { model_free(m); return NULL; }
        L->shadow_u = calloc(V12_D_STATE, SHADOW_SIZEOF);
        if (!L->shadow_u) { model_free(m); return NULL; }
        L->sampled_u = calloc(V12_D_STATE, sizeof(float));
        if (!L->sampled_u) { model_free(m); return NULL; }
        L->du = calloc(V12_D_STATE, sizeof(float));
        if (!L->du) { model_free(m); return NULL; }
        L->m_u = calloc(V12_D_STATE, sizeof(float));
        if (!L->m_u) { model_free(m); return NULL; }
        L->v_u = calloc(V12_D_STATE, sizeof(float));
        if (!L->v_u) { model_free(m); return NULL; }

        L->v = calloc(V12_D_STATE, sizeof(float));
        if (!L->v) { model_free(m); return NULL; }
        L->shadow_v = calloc(V12_D_STATE, SHADOW_SIZEOF);
        if (!L->shadow_v) { model_free(m); return NULL; }
        L->sampled_v = calloc(V12_D_STATE, sizeof(float));
        if (!L->sampled_v) { model_free(m); return NULL; }
        L->dv = calloc(V12_D_STATE, sizeof(float));
        if (!L->dv) { model_free(m); return NULL; }
        L->m_v = calloc(V12_D_STATE, sizeof(float));
        if (!L->m_v) { model_free(m); return NULL; }
        L->v_v = calloc(V12_D_STATE, sizeof(float));
        if (!L->v_v) { model_free(m); return NULL; }

        /* Alive indices for lazy optimizer — NULL until first repack */
        L->alive_idx_in = NULL;
        L->n_alive_in = 0;
        L->alive_idx_out = NULL;
        L->n_alive_out = 0;

    }

    /* Initialize sampled weights from shadow logits */
    uint64_t sample_rng = 99999ULL;
    for (int k = 0; k < V12_N_LAYERS; k++) {
        mvue_sample_weights(&m->layers[k], &sample_rng);
    }

    /* Initialize sampled_w_pred similarly */
    for (size_t idx = 0; idx < (size_t)V12_D_MODEL * V12_D_MODEL; idx++) {
        sample_rng ^= sample_rng << 13;
        sample_rng ^= sample_rng >> 7;
        sample_rng ^= sample_rng << 17;
        float u = (float)(sample_rng >> 11) * (1.0f / (float)(1ULL << 53));
        m->sampled_w_pred[idx] = sample_ternary_from_logit_mvue(SHADOW_DECODE(m->shadow_w_pred[idx]), u);
    }

    /* Build CSR indices from freshly sampled weights */
    for (int k = 0; k < V12_N_LAYERS; k++) {
        if (model_rebuild_layer_csr(&m->layers[k]) != 0) {
            model_free(m);
            return NULL;
        }
    }
    return m;
}

/* =========================================================================
 * model_rebuild_layer_csr — Convert sampled float weights to int8 and build CSR.
 *
 * Called once after model_alloc and again after every MVUE repack
 * (every V12_REPACK_INTERVAL steps via optimizer_step).
 *
 * sampled_w_in  layout: [d_state × d_model], row=d_state, col=d_model
 * sampled_w_out layout: [d_model × d_state], row=d_model, col=d_state
 *
 * Returns 0 on success, -1 on allocation failure.
 * ========================================================================= */
int model_rebuild_layer_csr(SSMLayer *L) {
    int n_in  = V12_D_STATE * V12_D_MODEL;
    int n_out = V12_D_MODEL * V12_D_STATE;

    int8_t *buf = malloc(n_in > n_out ? n_in : n_out);
    if (!buf) return -1;

    for (int i = 0; i < n_in; i++) {
        buf[i] = (int8_t)(int)L->sampled_w_in[i];
    }
    ternary_csr_free(&L->csr_in);
    if (ternary_csr_init(&L->csr_in, buf, V12_D_STATE, V12_D_MODEL) != 0) {
        free(buf);
        return -1;
    }

    for (int i = 0; i < n_out; i++) {
        buf[i] = (int8_t)(int)L->sampled_w_out[i];
    }
    ternary_csr_free(&L->csr_out);
    if (ternary_csr_init(&L->csr_out, buf, V12_D_MODEL, V12_D_STATE) != 0) {
        free(buf);
        return -1;
    }

    free(buf);
    return 0;
}

/* =========================================================================
 * model_rebuild_lm_head_csr — Convert sampled_lm_head float weights to CSR.
 *
 * sampled_lm_head layout as used by forward loop: [vocab × d_model]
 * rows = V12_VOCAB_SIZE, cols = V12_D_MODEL
 *
 * Returns 0 on success, -1 on allocation failure.
 * ========================================================================= */
int model_rebuild_lm_head_csr(Model *m) {
#if V12_TOLR_ENABLED
    if (!m || !m->embed_E || !m->embed_P) return -1;
    
    int n = V12_VOCAB_SIZE * V12_D_MODEL;
    int8_t *buf = malloc(n);
    if (!buf) return -1;

    for (int t = 0; t < V12_VOCAB_SIZE; t++) {
        const float *e = m->embed_E + t * V12_D_EMBED;
        for (int j = 0; j < V12_D_MODEL; j++) {
            float acc = 0.0f;
            for (int r = 0; r < V12_D_EMBED; r++) {
                acc += e[r] * m->embed_P[r * V12_D_MODEL + j];
            }
            if      (acc >  0.33f) buf[t * V12_D_MODEL + j] = 1;
            else if (acc < -0.33f) buf[t * V12_D_MODEL + j] = -1;
            else                   buf[t * V12_D_MODEL + j] = 0;
        }
        
        /* Pack directly into embed_int2 as well to keep them in sync */
        for (int b = 0; b < V12_D_MODEL / V12_INT2_WEIGHTS_PER_BYTE; b++) {
            uint8_t packed = 0;
            for (int k = 0; k < V12_INT2_WEIGHTS_PER_BYTE; k++) {
                int j = b * V12_INT2_WEIGHTS_PER_BYTE + k;
                int8_t v = buf[t * V12_D_MODEL + j];
                uint8_t bits;
                if      (v == 1)  bits = 0x01;
                else if (v == -1) bits = 0x03;
                else              bits = 0x00;
                packed |= (bits << (k * 2));
            }
            m->embed_int2[t * (V12_D_MODEL / V12_INT2_WEIGHTS_PER_BYTE) + b] = packed;
        }
    }

    ternary_csr_free(&m->csr_lm_head);
    int rc = ternary_csr_init(&m->csr_lm_head, buf,
                               V12_VOCAB_SIZE, V12_D_MODEL);
    free(buf);
    return rc;
#else
    if (!m || !m->sampled_lm_head) return -1;

    int n = V12_VOCAB_SIZE * V12_D_MODEL;

    int8_t *buf = malloc(n);
    if (!buf) return -1;

    for (int i = 0; i < n; i++) {
        buf[i] = (int8_t)(int)m->sampled_lm_head[i];
    }

    ternary_csr_free(&m->csr_lm_head);
    int rc = ternary_csr_init(&m->csr_lm_head, buf,
                               V12_VOCAB_SIZE, V12_D_MODEL);
    free(buf);
    return rc;
#endif
}

static int model_alloc_lm_head_host_buffers(Model *m) {
#if V12_TOLR_ENABLED
    return 0;
#else
    const size_t lm_elems = (size_t)V12_VOCAB_SIZE * V12_D_MODEL;

    if (!m) return -1;

    if (!m->shadow_lm_head) {
        m->shadow_lm_head = malloc(lm_elems * SHADOW_SIZEOF);
        if (!m->shadow_lm_head) return -1;
    }
    if (!m->sampled_lm_head) {
        m->sampled_lm_head = malloc(lm_elems * sizeof(float));
        if (!m->sampled_lm_head) return -1;
    }

    return 0;
#endif
}

int model_initialize_lm_head_host(Model *m) {
    uint64_t lmhead_rng = 123ULL;
    uint64_t sample_rng = 99999ULL;

    if (model_alloc_lm_head_host_buffers(m) != 0) {
        return -1;
    }

#if !V12_TOLR_ENABLED
    const size_t lm_elems = (size_t)V12_VOCAB_SIZE * V12_D_MODEL;
    for (size_t i = 0; i < lm_elems; i++) {
        lmhead_rng ^= lmhead_rng << 13;
        lmhead_rng ^= lmhead_rng >> 7;
        lmhead_rng ^= lmhead_rng << 17;
        float v = ((float)(lmhead_rng & 0xFFFF) / 65536.0f - 0.5f) * 0.6f;
#if V12_F2_INT8_SHADOWS
        uint32_t r = shadow_rng(0, V12_SHADOW_MAT_LM_HEAD, (int)i, 0ULL);
        SHADOW_STORE(m->shadow_lm_head[i], v, r);
#else
        m->shadow_lm_head[i] = v;
#endif
    }
#endif

    for (int k = 0; k < V12_N_LAYERS; k++) {
        mvue_sample_weights(&m->layers[k], &sample_rng);
    }

#if !V12_TOLR_ENABLED
    for (size_t i = 0; i < lm_elems; i++) {
        sample_rng ^= sample_rng << 13;
        sample_rng ^= sample_rng >> 7;
        sample_rng ^= sample_rng << 17;
        float u = (float)(sample_rng >> 11) * (1.0f / (float)(1ULL << 53));
        m->sampled_lm_head[i] = sample_ternary_from_logit_mvue(SHADOW_DECODE(m->shadow_lm_head[i]), u);
    }
#endif

    return model_rebuild_lm_head_csr(m);
}

int model_materialize_lm_head_host(Model *m) {
    return model_alloc_lm_head_host_buffers(m);
}

static int model_alloc_layer_w_io_host_buffers(SSMLayer *layer) {
    const size_t win_elems = (size_t)V12_D_MODEL * V12_D_STATE;
    const size_t wout_elems = (size_t)V12_D_STATE * V12_D_MODEL;

    if (!layer) return -1;

    if (!layer->shadow_w_in) {
        layer->shadow_w_in = (shadow_t *)calloc(win_elems, SHADOW_SIZEOF);
        if (!layer->shadow_w_in) return -1;
    }
    if (!layer->sampled_w_in) {
        layer->sampled_w_in = (float *)calloc(win_elems, sizeof(float));
        if (!layer->sampled_w_in) return -1;
    }
    if (!layer->m_w_in_row) {
        layer->m_w_in_row = (float *)calloc(V12_D_MODEL, sizeof(float));
        if (!layer->m_w_in_row) return -1;
    }
    if (!layer->m_w_in_col) {
        layer->m_w_in_col = (float *)calloc(V12_D_STATE, sizeof(float));
        if (!layer->m_w_in_col) return -1;
    }
    if (!layer->v_w_in_row) {
        layer->v_w_in_row = (float *)calloc(V12_D_MODEL, sizeof(float));
        if (!layer->v_w_in_row) return -1;
    }
    if (!layer->v_w_in_col) {
        layer->v_w_in_col = (float *)calloc(V12_D_STATE, sizeof(float));
        if (!layer->v_w_in_col) return -1;
    }

    if (!layer->shadow_w_out) {
        layer->shadow_w_out = (shadow_t *)calloc(wout_elems, SHADOW_SIZEOF);
        if (!layer->shadow_w_out) return -1;
    }
    if (!layer->sampled_w_out) {
        layer->sampled_w_out = (float *)calloc(wout_elems, sizeof(float));
        if (!layer->sampled_w_out) return -1;
    }
    if (!layer->m_w_out_row) {
        layer->m_w_out_row = (float *)calloc(V12_D_STATE, sizeof(float));
        if (!layer->m_w_out_row) return -1;
    }
    if (!layer->m_w_out_col) {
        layer->m_w_out_col = (float *)calloc(V12_D_MODEL, sizeof(float));
        if (!layer->m_w_out_col) return -1;
    }
    if (!layer->v_w_out_row) {
        layer->v_w_out_row = (float *)calloc(V12_D_STATE, sizeof(float));
        if (!layer->v_w_out_row) return -1;
    }
    if (!layer->v_w_out_col) {
        layer->v_w_out_col = (float *)calloc(V12_D_MODEL, sizeof(float));
        if (!layer->v_w_out_col) return -1;
    }

    return 0;
}

int model_materialize_layer_w_io_host(SSMLayer *layer) {
    return model_alloc_layer_w_io_host_buffers(layer);
}

void model_release_layer_w_io_host(SSMLayer *layer) {
    if (!layer) return;

    free(layer->shadow_w_in);
    free(layer->sampled_w_in);
    free(layer->m_w_in_row);
    free(layer->m_w_in_col);
    free(layer->v_w_in_row);
    free(layer->v_w_in_col);

    free(layer->shadow_w_out);
    free(layer->sampled_w_out);
    free(layer->m_w_out_row);
    free(layer->m_w_out_col);
    free(layer->v_w_out_row);
    free(layer->v_w_out_col);

    layer->shadow_w_in = NULL;
    layer->sampled_w_in = NULL;
    layer->m_w_in_row = NULL;
    layer->m_w_in_col = NULL;
    layer->v_w_in_row = NULL;
    layer->v_w_in_col = NULL;

    layer->shadow_w_out = NULL;
    layer->sampled_w_out = NULL;
    layer->m_w_out_row = NULL;
    layer->m_w_out_col = NULL;
    layer->v_w_out_row = NULL;
    layer->v_w_out_col = NULL;
}

static void model_init_embed_P(float *embed_P) {
    uint64_t rng = 0xDEADBEEFCAFE1234ULL;
    float scale = sqrtf(6.0f / (V12_D_EMBED + V12_D_MODEL));
    for (int i = 0; i < V12_D_EMBED * V12_D_MODEL; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        float u = (float)(rng >> 11) * (1.0f / (float)(1ULL << 53));
        embed_P[i] = (2.0f * u - 1.0f) * scale;
    }
}

static int model_alloc_embed_host_buffers(Model *m) {
    if (!m) return -1;

    if (!m->embed_E) {
        m->embed_E = (float *)malloc((size_t)V12_VOCAB_SIZE * V12_D_EMBED * sizeof(float));
        if (!m->embed_E) return -1;
        rns_embed_init_E(m->embed_E);
    }
    if (!m->embed_P) {
        m->embed_P = (float *)malloc((size_t)V12_D_EMBED * V12_D_MODEL * sizeof(float));
        if (!m->embed_P) return -1;
        model_init_embed_P(m->embed_P);
    }
    if (!m->d_embed_E) {
        m->d_embed_E = (float *)calloc((size_t)V12_VOCAB_SIZE * V12_D_EMBED, sizeof(float));
        if (!m->d_embed_E) return -1;
    }
    if (!m->d_embed_P) {
        m->d_embed_P = (float *)calloc((size_t)V12_D_EMBED * V12_D_MODEL, sizeof(float));
        if (!m->d_embed_P) return -1;
    }
    if (!m->m_embed_E) {
        m->m_embed_E = (float *)calloc((size_t)V12_VOCAB_SIZE * V12_D_EMBED, sizeof(float));
        if (!m->m_embed_E) return -1;
    }
    if (!m->v_embed_E) {
        m->v_embed_E = (float *)calloc((size_t)V12_VOCAB_SIZE * V12_D_EMBED, sizeof(float));
        if (!m->v_embed_E) return -1;
    }
    if (!m->m_embed_P) {
        m->m_embed_P = (float *)calloc((size_t)V12_D_EMBED * V12_D_MODEL, sizeof(float));
        if (!m->m_embed_P) return -1;
    }
    if (!m->v_embed_P) {
        m->v_embed_P = (float *)calloc((size_t)V12_D_EMBED * V12_D_MODEL, sizeof(float));
        if (!m->v_embed_P) return -1;
    }
    if (!m->m_embed_E_row) {
        m->m_embed_E_row = (float *)calloc(V12_VOCAB_SIZE, sizeof(float));
        if (!m->m_embed_E_row) return -1;
    }
    if (!m->m_embed_E_col) {
        m->m_embed_E_col = (float *)calloc(V12_D_EMBED, sizeof(float));
        if (!m->m_embed_E_col) return -1;
    }
    if (!m->v_embed_E_row) {
        m->v_embed_E_row = (float *)calloc(V12_VOCAB_SIZE, sizeof(float));
        if (!m->v_embed_E_row) return -1;
    }
    if (!m->v_embed_E_col) {
        m->v_embed_E_col = (float *)calloc(V12_D_EMBED, sizeof(float));
        if (!m->v_embed_E_col) return -1;
    }

    return 0;
}

int model_materialize_embed_host(Model *m) {
    return model_alloc_embed_host_buffers(m);
}

static int model_alloc_w_pred_host_buffers(Model *m) {
    const size_t pred_elems = (size_t)V12_D_MODEL * V12_D_MODEL;

    if (!m) return -1;

    if (!m->shadow_w_pred) {
        m->shadow_w_pred = calloc(pred_elems, SHADOW_SIZEOF);
        if (!m->shadow_w_pred) return -1;
    }
    if (!m->sampled_w_pred) {
        m->sampled_w_pred = calloc(pred_elems, sizeof(float));
        if (!m->sampled_w_pred) return -1;
    }
    if (!m->dw_pred) {
        m->dw_pred = calloc(pred_elems, sizeof(float));
        if (!m->dw_pred) return -1;
    }
    if (!m->m_w_pred_row) {
        m->m_w_pred_row = calloc(V12_D_MODEL, sizeof(float));
        if (!m->m_w_pred_row) return -1;
    }
    if (!m->m_w_pred_col) {
        m->m_w_pred_col = calloc(V12_D_MODEL, sizeof(float));
        if (!m->m_w_pred_col) return -1;
    }
    if (!m->v_w_pred_row) {
        m->v_w_pred_row = calloc(V12_D_MODEL, sizeof(float));
        if (!m->v_w_pred_row) return -1;
    }
    if (!m->v_w_pred_col) {
        m->v_w_pred_col = calloc(V12_D_MODEL, sizeof(float));
        if (!m->v_w_pred_col) return -1;
    }

    return 0;
}

static int model_alloc_norm_host_buffers(Model *m) {
    if (!m) return -1;

    if (!m->norm_w) {
        m->norm_w = calloc(V12_D_MODEL, sizeof(float));
        if (!m->norm_w) return -1;
    }
    if (!m->norm_b) {
        m->norm_b = calloc(V12_D_MODEL, sizeof(float));
        if (!m->norm_b) return -1;
    }
    if (!m->d_norm_w) {
        m->d_norm_w = calloc(V12_D_MODEL, sizeof(float));
        if (!m->d_norm_w) return -1;
    }
    if (!m->d_norm_b) {
        m->d_norm_b = calloc(V12_D_MODEL, sizeof(float));
        if (!m->d_norm_b) return -1;
    }
    if (!m->m_norm_w) {
        m->m_norm_w = calloc(V12_D_MODEL, sizeof(float));
        if (!m->m_norm_w) return -1;
    }
    if (!m->v_norm_w) {
        m->v_norm_w = calloc(V12_D_MODEL, sizeof(float));
        if (!m->v_norm_w) return -1;
    }
    if (!m->m_norm_b) {
        m->m_norm_b = calloc(V12_D_MODEL, sizeof(float));
        if (!m->m_norm_b) return -1;
    }
    if (!m->v_norm_b) {
        m->v_norm_b = calloc(V12_D_MODEL, sizeof(float));
        if (!m->v_norm_b) return -1;
    }

    return 0;
}

int model_materialize_w_pred_host(Model *m) {
    return model_alloc_w_pred_host_buffers(m);
}

void model_release_w_pred_host(Model *m) {
    if (!m) return;

    free(m->shadow_w_pred);
    free(m->sampled_w_pred);
    free(m->dw_pred);
    free(m->m_w_pred_row);
    free(m->m_w_pred_col);
    free(m->v_w_pred_row);
    free(m->v_w_pred_col);

    m->shadow_w_pred = NULL;
    m->sampled_w_pred = NULL;
    m->dw_pred = NULL;
    m->m_w_pred_row = NULL;
    m->m_w_pred_col = NULL;
    m->v_w_pred_row = NULL;
    m->v_w_pred_col = NULL;
}

void model_release_embed_host(Model *m) {
    if (!m) return;

    free(m->embed_E);
    free(m->embed_P);
    free(m->d_embed_E);
    free(m->d_embed_P);
    free(m->sa_embed_E);
    free(m->sa_embed_P);
    free(m->m_embed_E_row);
    free(m->m_embed_E_col);
    free(m->v_embed_E_row);
    free(m->v_embed_E_col);

    m->embed_E = NULL;
    m->embed_P = NULL;
    m->d_embed_E = NULL;
    m->d_embed_P = NULL;
    m->sa_embed_E = NULL;
    m->sa_embed_P = NULL;
    m->m_embed_E_row = NULL;
    m->m_embed_E_col = NULL;
    m->v_embed_E_row = NULL;
    m->v_embed_E_col = NULL;
}

int model_materialize_norm_host(Model *m) {
    return model_alloc_norm_host_buffers(m);
}

void model_release_norm_host(Model *m) {
    if (!m) return;

    free(m->norm_w);
    free(m->norm_b);
    free(m->d_norm_w);
    free(m->d_norm_b);
    free(m->m_norm_w);
    free(m->v_norm_w);
    free(m->m_norm_b);
    free(m->v_norm_b);

    m->norm_w = NULL;
    m->norm_b = NULL;
    m->d_norm_w = NULL;
    m->d_norm_b = NULL;
    m->m_norm_w = NULL;
    m->v_norm_w = NULL;
    m->m_norm_b = NULL;
    m->v_norm_b = NULL;
}

int model_w_pred_host_is_materialized(const Model *m) {
    return m
        && m->shadow_w_pred
        && m->sampled_w_pred
        && m->dw_pred
        && m->m_w_pred_row
        && m->m_w_pred_col
        && m->v_w_pred_row
        && m->v_w_pred_col;
}

int model_embed_host_is_materialized(const Model *m) {
    return m
        && m->embed_E
        && m->embed_P
        && m->d_embed_E
        && m->d_embed_P
        && m->sa_embed_E
        && m->sa_embed_P
        && m->m_embed_E_row
        && m->m_embed_E_col
        && m->v_embed_E_row
        && m->v_embed_E_col;
}

int model_norm_host_is_materialized(const Model *m) {
    return m
        && m->norm_w
        && m->norm_b
        && m->d_norm_w
        && m->d_norm_b
        && m->m_norm_w
        && m->v_norm_w
        && m->m_norm_b
        && m->v_norm_b;
}

int model_layer_w_io_host_is_materialized(const SSMLayer *layer) {
    return layer
        && layer->shadow_w_in
        && layer->sampled_w_in
        && layer->m_w_in_row
        && layer->m_w_in_col
        && layer->v_w_in_row
        && layer->v_w_in_col
        && layer->shadow_w_out
        && layer->sampled_w_out
        && layer->m_w_out_row
        && layer->m_w_out_col
        && layer->v_w_out_row
        && layer->v_w_out_col;
}

void model_release_lm_head_host(Model *m) {
    if (!m) return;
#if !V12_TOLR_ENABLED
    free(m->shadow_lm_head);
    free(m->sampled_lm_head);
    free(m->dlm_head_g_row);
    free(m->dlm_head_g_col);
    free(m->dlm_head_gq_row);
    free(m->dlm_head_gq_col);
    free(m->dlm_head_cnt_row);
    free(m->dlm_head_cnt_col);
    m->shadow_lm_head = NULL;
    m->sampled_lm_head = NULL;
    m->dlm_head_g_row = NULL;
    m->dlm_head_g_col = NULL;
    m->dlm_head_gq_row = NULL;
    m->dlm_head_gq_col = NULL;
    m->dlm_head_cnt_row = NULL;
    m->dlm_head_cnt_col = NULL;
#endif
}

/* =========================================================================
 * model_free — Complete Model deallocation
 * ========================================================================= */

/*
 * sampled_softmax_build_unigram — Build unigram proposal distribution for NCE.
 *
 * Scans all tokens to compute frequency counts, applies 3/4 power smoothing
 * (standard NCE practice: rare tokens are sampled more), normalizes to form
 * a valid CDF and log-probability table.
 *
 * Call once after dataset load, before first training step.
 */
void sampled_softmax_build_unigram(Model *m, const uint16_t *tokens, size_t n_toks) {
    int V = V12_VOCAB_SIZE;

    if (!m->unigram_cdf)   m->unigram_cdf  = malloc(V * sizeof(float));
    if (!m->log_unigram)   m->log_unigram  = malloc(V * sizeof(float));
    if (!m->token_freq)    m->token_freq   = calloc(V, sizeof(float));

    float *freq = calloc(V, sizeof(float));
    for (size_t i = 0; i < n_toks; i++) {
        freq[tokens[i]] += 1.0f;
        if (tokens[i] < V) m->token_freq[tokens[i]] += 1.0f;
    }

    float total = 0.0f;
    for (int v = 0; v < V; v++) {
        freq[v] = powf(freq[v] + 1.0f, 0.75f);
        total += freq[v];
    }

    float cdf = 0.0f;
    for (int v = 0; v < V; v++) {
        float q = freq[v] / total;
        m->log_unigram[v] = logf(q + 1e-30f);
        cdf += q;
        m->unigram_cdf[v] = cdf;
    }
    m->unigram_cdf[V - 1] = 1.0f;

    free(freq);
}

static uint32_t corpus_logit_prior_key(uint16_t context, uint16_t target) {
    return ((uint32_t)context << 16) | (uint32_t)target;
}

static uint32_t corpus_logit_prior_mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static void corpus_logit_prior_clear(Model *m) {
    if (!m) return;
    free(m->corpus_logit_prior_ctx_counts);
    free(m->corpus_logit_prior_pair_keys);
    free(m->corpus_logit_prior_pair_counts);
    free(m->corpus_logit_prior_unigram);
    m->corpus_logit_prior_ctx_counts = NULL;
    m->corpus_logit_prior_pair_keys = NULL;
    m->corpus_logit_prior_pair_counts = NULL;
    m->corpus_logit_prior_unigram = NULL;
    m->corpus_logit_prior_actual_terms = 0;
    m->corpus_logit_prior_pair_capacity = 0;
    m->corpus_logit_prior_pair_unique = 0;
}

void corpus_logit_prior_configure(
    Model *m, int enabled, float alpha, float scale,
    uint64_t requested_terms, uint32_t stride) {
    if (!m) return;
    corpus_logit_prior_clear(m);
    m->corpus_logit_prior_enabled = enabled ? 1 : 0;
    m->corpus_logit_prior_alpha = (alpha > 0.0f) ? alpha : 100.0f;
    m->corpus_logit_prior_scale = (scale >= 0.0f) ? scale : 1.0f;
    m->corpus_logit_prior_requested_terms = requested_terms;
    m->corpus_logit_prior_stride = (stride > 0u) ? stride : 1u;
}

static int corpus_logit_prior_rehash(Model *m, uint32_t new_capacity) {
    uint32_t *old_keys = m->corpus_logit_prior_pair_keys;
    uint32_t *old_counts = m->corpus_logit_prior_pair_counts;
    uint32_t old_capacity = m->corpus_logit_prior_pair_capacity;

    uint32_t *new_keys = (uint32_t *)calloc(new_capacity, sizeof(uint32_t));
    uint32_t *new_counts = (uint32_t *)calloc(new_capacity, sizeof(uint32_t));
    if (!new_keys || !new_counts) {
        free(new_keys);
        free(new_counts);
        return -1;
    }

    m->corpus_logit_prior_pair_keys = new_keys;
    m->corpus_logit_prior_pair_counts = new_counts;
    m->corpus_logit_prior_pair_capacity = new_capacity;
    m->corpus_logit_prior_pair_unique = 0;

    const uint32_t mask = new_capacity - 1u;
    for (uint32_t i = 0; i < old_capacity; i++) {
        if (old_counts[i] == 0u) continue;
        uint32_t key = old_keys[i];
        uint32_t slot = corpus_logit_prior_mix32(key) & mask;
        while (new_counts[slot] != 0u) {
            slot = (slot + 1u) & mask;
        }
        new_keys[slot] = key;
        new_counts[slot] = old_counts[i];
        m->corpus_logit_prior_pair_unique++;
    }

    free(old_keys);
    free(old_counts);
    return 0;
}

static int corpus_logit_prior_add_pair(Model *m, uint16_t context, uint16_t target) {
    if (m->corpus_logit_prior_pair_capacity == 0u) {
        if (corpus_logit_prior_rehash(m, 65536u) != 0) return -1;
    } else if ((uint64_t)(m->corpus_logit_prior_pair_unique + 1u) * 2u >=
               (uint64_t)m->corpus_logit_prior_pair_capacity) {
        if (corpus_logit_prior_rehash(m, m->corpus_logit_prior_pair_capacity << 1) != 0) {
            return -1;
        }
    }

    const uint32_t key = corpus_logit_prior_key(context, target);
    const uint32_t mask = m->corpus_logit_prior_pair_capacity - 1u;
    uint32_t slot = corpus_logit_prior_mix32(key) & mask;
    while (m->corpus_logit_prior_pair_counts[slot] != 0u) {
        if (m->corpus_logit_prior_pair_keys[slot] == key) {
            if (m->corpus_logit_prior_pair_counts[slot] != UINT32_MAX) {
                m->corpus_logit_prior_pair_counts[slot]++;
            }
            return 0;
        }
        slot = (slot + 1u) & mask;
    }

    m->corpus_logit_prior_pair_keys[slot] = key;
    m->corpus_logit_prior_pair_counts[slot] = 1u;
    m->corpus_logit_prior_pair_unique++;
    return 0;
}

int corpus_logit_prior_build(Model *m, const uint16_t *tokens, size_t n_toks) {
    if (!m || !m->corpus_logit_prior_enabled) return 0;
    if (!tokens || n_toks < 2) return -1;

    corpus_logit_prior_clear(m);
    const int V = V12_VOCAB_SIZE;

    m->corpus_logit_prior_ctx_counts = (uint32_t *)calloc((size_t)V, sizeof(uint32_t));
    m->corpus_logit_prior_unigram = (float *)malloc((size_t)V * sizeof(float));
    uint64_t *uni_counts = (uint64_t *)calloc((size_t)V, sizeof(uint64_t));
    if (!m->corpus_logit_prior_ctx_counts || !m->corpus_logit_prior_unigram || !uni_counts) {
        free(uni_counts);
        corpus_logit_prior_clear(m);
        return -1;
    }

    uint64_t uni_total = 0;
    for (size_t i = 0; i < n_toks; i++) {
        if (tokens[i] < V12_VOCAB_SIZE) {
            uni_counts[tokens[i]]++;
            uni_total++;
        }
    }
    const double uni_den = (double)uni_total + (double)V;
    for (int v = 0; v < V; v++) {
        m->corpus_logit_prior_unigram[v] =
            (float)(((double)uni_counts[v] + 1.0) / uni_den);
    }

    const uint32_t stride = m->corpus_logit_prior_stride ? m->corpus_logit_prior_stride : 1u;
    const uint64_t requested = m->corpus_logit_prior_requested_terms;
    const uint64_t max_terms = requested ? requested : UINT64_MAX;

    for (size_t i = 0; i + 1 < n_toks && m->corpus_logit_prior_actual_terms < max_terms; i += stride) {
        const uint16_t ctx = tokens[i];
        const uint16_t tgt = tokens[i + 1];
        if (ctx >= V12_VOCAB_SIZE || tgt >= V12_VOCAB_SIZE) continue;
        if (m->corpus_logit_prior_ctx_counts[ctx] != UINT32_MAX) {
            m->corpus_logit_prior_ctx_counts[ctx]++;
        }
        if (corpus_logit_prior_add_pair(m, ctx, tgt) != 0) {
            free(uni_counts);
            corpus_logit_prior_clear(m);
            return -1;
        }
        m->corpus_logit_prior_actual_terms++;
    }

    free(uni_counts);
    printf("[CORPUS_LOGIT_PRIOR] active=1 mode=corpus-bigram-log-prior alpha=%.3f scale=%.6f stride=%u terms=%llu unique_pairs=%u table_capacity=%u\n",
           m->corpus_logit_prior_alpha,
           m->corpus_logit_prior_scale,
           m->corpus_logit_prior_stride,
           (unsigned long long)m->corpus_logit_prior_actual_terms,
           m->corpus_logit_prior_pair_unique,
           m->corpus_logit_prior_pair_capacity);
    return 0;
}

float corpus_logit_prior_logit(const Model *m, uint16_t context, uint16_t target) {
    if (!m || !m->corpus_logit_prior_enabled || !m->corpus_logit_prior_unigram ||
        !m->corpus_logit_prior_ctx_counts || !m->corpus_logit_prior_pair_counts ||
        !m->corpus_logit_prior_pair_keys || m->corpus_logit_prior_pair_capacity == 0u) {
        return 0.0f;
    }
    if (context >= V12_VOCAB_SIZE || target >= V12_VOCAB_SIZE) {
        return 0.0f;
    }

    const float p_uni = fmaxf(m->corpus_logit_prior_unigram[target], 1e-30f);
    const uint32_t ctx_count = m->corpus_logit_prior_ctx_counts[context];
    if (ctx_count == 0u) {
        return 0.0f;
    }

    uint32_t pair_count = 0u;
    const uint32_t key = corpus_logit_prior_key(context, target);
    const uint32_t mask = m->corpus_logit_prior_pair_capacity - 1u;
    uint32_t slot = corpus_logit_prior_mix32(key) & mask;
    while (m->corpus_logit_prior_pair_counts[slot] != 0u) {
        if (m->corpus_logit_prior_pair_keys[slot] == key) {
            pair_count = m->corpus_logit_prior_pair_counts[slot];
            break;
        }
        slot = (slot + 1u) & mask;
    }

    const float alpha = (m->corpus_logit_prior_alpha > 0.0f) ? m->corpus_logit_prior_alpha : 100.0f;
    const float numerator = (float)pair_count + alpha * p_uni;
    const float denominator = (float)ctx_count + alpha;
    const float p_prior = fmaxf(numerator / denominator, 1e-30f);
    return m->corpus_logit_prior_scale * logf(p_prior / p_uni);
}

static void program_kernel_clear(Model *m) {
    if (!m) return;
    free(m->program_kernel_targets);
    free(m->program_kernel_counts);
    m->program_kernel_targets = NULL;
    m->program_kernel_counts = NULL;
    m->program_kernel_actual_terms = 0;
    m->program_kernel_entries = 0;
    m->program_kernel_conflict_contexts = 0;
    m->program_kernel_gate_base_loss = 0.0f;
    m->program_kernel_gate_kernel_loss = 0.0f;
    m->program_kernel_gate_kernel_ratio = 0.0f;
    m->program_kernel_gate_batches = 0;
    m->program_kernel_gate_routed = 0;
    m->program_kernel_gate_correct = 0;
    m->program_kernel_gate_incorrect = 0;
}

void program_kernel_configure(
    Model *m, int enabled, int gate_enabled, float logit_boost,
    uint32_t min_count, uint64_t requested_terms, uint32_t stride) {
    if (!m) return;
    program_kernel_clear(m);
    m->program_kernel_enabled = enabled ? 1 : 0;
    m->program_kernel_gate_enabled = gate_enabled ? 1 : 0;
    m->program_kernel_logit_boost = (logit_boost > 0.0f) ? logit_boost : 4.0f;
    m->program_kernel_min_count = (min_count > 0u) ? min_count : 2u;
    m->program_kernel_requested_terms = requested_terms;
    m->program_kernel_stride = (stride > 0u) ? stride : 1u;
}

int program_kernel_build(Model *m, const uint16_t *tokens, size_t n_toks) {
    if (!m || !m->program_kernel_enabled) return 0;
    if (!tokens || n_toks < 2) return -1;

    program_kernel_clear(m);
    const int V = V12_VOCAB_SIZE;
    m->program_kernel_targets = (uint32_t *)malloc((size_t)V * sizeof(uint32_t));
    m->program_kernel_counts = (uint32_t *)calloc((size_t)V, sizeof(uint32_t));
    uint8_t *conflict = (uint8_t *)calloc((size_t)V, sizeof(uint8_t));
    if (!m->program_kernel_targets || !m->program_kernel_counts || !conflict) {
        free(conflict);
        program_kernel_clear(m);
        return -1;
    }
    for (int v = 0; v < V; v++) {
        m->program_kernel_targets[v] = UINT32_MAX;
    }

    const uint32_t stride = m->program_kernel_stride ? m->program_kernel_stride : 1u;
    const uint64_t requested = m->program_kernel_requested_terms;
    const uint64_t max_terms = requested ? requested : UINT64_MAX;

    for (size_t i = 0; i + 1 < n_toks && m->program_kernel_actual_terms < max_terms; i += stride) {
        const uint16_t ctx = tokens[i];
        const uint16_t tgt = tokens[i + 1];
        if (ctx >= V12_VOCAB_SIZE || tgt >= V12_VOCAB_SIZE) continue;

        if (m->program_kernel_counts[ctx] != UINT32_MAX) {
            m->program_kernel_counts[ctx]++;
        }
        if (m->program_kernel_targets[ctx] == UINT32_MAX) {
            m->program_kernel_targets[ctx] = (uint32_t)tgt;
        } else if (m->program_kernel_targets[ctx] != (uint32_t)tgt) {
            conflict[ctx] = 1u;
        }
        m->program_kernel_actual_terms++;
    }

    for (int v = 0; v < V; v++) {
        const int has_context = m->program_kernel_counts[v] > 0u;
        if (has_context && conflict[v]) {
            m->program_kernel_conflict_contexts++;
        }
        if (!has_context ||
            conflict[v] ||
            m->program_kernel_counts[v] < m->program_kernel_min_count ||
            m->program_kernel_targets[v] == UINT32_MAX) {
            m->program_kernel_targets[v] = UINT32_MAX;
            m->program_kernel_counts[v] = 0u;
            continue;
        }
        m->program_kernel_entries++;
    }

    free(conflict);
    printf("[PROGRAM_KERNEL] active=1 mode=deterministic-transition-table logit_boost=%.6f min_count=%u stride=%u terms=%llu entries=%u conflicts=%u\n",
           m->program_kernel_logit_boost,
           m->program_kernel_min_count,
           m->program_kernel_stride,
           (unsigned long long)m->program_kernel_actual_terms,
           m->program_kernel_entries,
           m->program_kernel_conflict_contexts);
    return 0;
}

uint32_t program_kernel_predict(const Model *m, uint16_t context) {
    if (!m || !m->program_kernel_enabled || !m->program_kernel_targets ||
        context >= V12_VOCAB_SIZE) {
        return UINT32_MAX;
    }
    return m->program_kernel_targets[context];
}

float program_kernel_logit(const Model *m, uint16_t context, uint16_t target) {
    const uint32_t predicted = program_kernel_predict(m, context);
    if (predicted == UINT32_MAX || predicted != (uint32_t)target) {
        return 0.0f;
    }
    return m->program_kernel_logit_boost;
}

static void certified_artifact_boundary_clear(Model *m) {
    if (!m) return;
    free(m->certified_artifact_boundary_counts);
    free(m->certified_artifact_boundary_false_counts);
    free(m->certified_artifact_boundary_certified);
    m->certified_artifact_boundary_counts = NULL;
    m->certified_artifact_boundary_false_counts = NULL;
    m->certified_artifact_boundary_certified = NULL;
    m->certified_artifact_boundary_actual_terms = 0;
    m->certified_artifact_boundary_candidate_contexts = 0;
    m->certified_artifact_boundary_certified_contexts = 0;
    m->certified_artifact_boundary_corpus_routed = 0;
    m->certified_artifact_boundary_corpus_abstain = 0;
    m->certified_artifact_boundary_corpus_false_routes = 0;
    m->certified_artifact_boundary_route_bytes = 0;
    m->certified_artifact_boundary_gate_batches = 0;
    m->certified_artifact_boundary_gate_positions = 0;
    m->certified_artifact_boundary_gate_routed = 0;
    m->certified_artifact_boundary_gate_correct = 0;
    m->certified_artifact_boundary_gate_incorrect = 0;
    m->certified_artifact_boundary_gate_abstain = 0;
    m->certified_artifact_boundary_gate_b_k = 0.0f;
    m->certified_artifact_boundary_gate_false_route_rate = 0.0f;
    m->certified_artifact_boundary_gate_ce_residual = 0.0f;
    m->certified_artifact_boundary_gate_priced_ratio = 0.0f;
    m->certified_artifact_boundary_obligation_audit = 0;
    m->certified_artifact_boundary_non_lookup_audit = 0;
}

void certified_artifact_boundary_configure(
    Model *m, int gate_enabled, uint32_t min_count,
    uint64_t requested_terms, uint32_t stride,
    float epsilon, float ce_budget, float overhead, float kappa) {
    if (!m) return;
    certified_artifact_boundary_clear(m);
    m->certified_artifact_boundary_gate_enabled = gate_enabled ? 1 : 0;
    m->certified_artifact_boundary_min_count = min_count > 0u ? min_count : 2u;
    m->certified_artifact_boundary_requested_terms = requested_terms;
    m->certified_artifact_boundary_stride = stride > 0u ? stride : 1u;
    m->certified_artifact_boundary_epsilon =
        (epsilon > 0.0f && epsilon < 1.0f) ? epsilon : 0.001f;
    m->certified_artifact_boundary_ce_budget = ce_budget >= 0.0f ? ce_budget : 0.001f;
    m->certified_artifact_boundary_overhead = overhead >= 0.0f ? overhead : 0.0f;
    m->certified_artifact_boundary_kappa = kappa > 0.0f ? kappa : 1.0f;
}

int certified_artifact_boundary_build(Model *m, const uint16_t *tokens, size_t n_toks) {
    if (!m || !m->certified_artifact_boundary_gate_enabled) return 0;
    if (!tokens || n_toks < 2) return -1;

    const uint32_t min_count = m->certified_artifact_boundary_min_count
        ? m->certified_artifact_boundary_min_count
        : 2u;
    const uint64_t requested = m->certified_artifact_boundary_requested_terms;
    const uint32_t stride = m->certified_artifact_boundary_stride
        ? m->certified_artifact_boundary_stride
        : 1u;
    const float epsilon = (m->certified_artifact_boundary_epsilon > 0.0f &&
                           m->certified_artifact_boundary_epsilon < 1.0f)
        ? m->certified_artifact_boundary_epsilon
        : 0.001f;
    const float ce_budget = m->certified_artifact_boundary_ce_budget >= 0.0f
        ? m->certified_artifact_boundary_ce_budget
        : 0.001f;
    const float overhead = m->certified_artifact_boundary_overhead >= 0.0f
        ? m->certified_artifact_boundary_overhead
        : 0.0f;
    const float kappa = m->certified_artifact_boundary_kappa > 0.0f
        ? m->certified_artifact_boundary_kappa
        : 1.0f;

    certified_artifact_boundary_clear(m);
    m->certified_artifact_boundary_gate_enabled = 1;
    m->certified_artifact_boundary_min_count = min_count;
    m->certified_artifact_boundary_requested_terms = requested;
    m->certified_artifact_boundary_stride = stride;
    m->certified_artifact_boundary_epsilon = epsilon;
    m->certified_artifact_boundary_ce_budget = ce_budget;
    m->certified_artifact_boundary_overhead = overhead;
    m->certified_artifact_boundary_kappa = kappa;

    const int V = V12_VOCAB_SIZE;
    m->certified_artifact_boundary_counts =
        (uint32_t *)calloc((size_t)V, sizeof(uint32_t));
    m->certified_artifact_boundary_false_counts =
        (uint32_t *)calloc((size_t)V, sizeof(uint32_t));
    m->certified_artifact_boundary_certified =
        (uint8_t *)calloc((size_t)V, sizeof(uint8_t));
    if (!m->certified_artifact_boundary_counts ||
        !m->certified_artifact_boundary_false_counts ||
        !m->certified_artifact_boundary_certified) {
        certified_artifact_boundary_clear(m);
        return -1;
    }

    const uint64_t max_terms = requested ? requested : UINT64_MAX;
    for (size_t i = 0;
         i + 1 < n_toks && m->certified_artifact_boundary_actual_terms < max_terms;
         i += stride) {
        const uint16_t ctx = tokens[i];
        const uint16_t tgt = tokens[i + 1];
        if (ctx >= V12_VOCAB_SIZE || tgt >= V12_VOCAB_SIZE) continue;

        if (m->certified_artifact_boundary_counts[ctx] != UINT32_MAX) {
            m->certified_artifact_boundary_counts[ctx]++;
        }
        if (tgt != ctx &&
            m->certified_artifact_boundary_false_counts[ctx] != UINT32_MAX) {
            m->certified_artifact_boundary_false_counts[ctx]++;
        }
        m->certified_artifact_boundary_actual_terms++;
    }

    for (int v = 0; v < V; v++) {
        const uint32_t count = m->certified_artifact_boundary_counts[v];
        const uint32_t false_count = m->certified_artifact_boundary_false_counts[v];
        if (count > 0u) {
            m->certified_artifact_boundary_candidate_contexts++;
        }
        if (count >= min_count && false_count == 0u) {
            m->certified_artifact_boundary_certified[v] = 1u;
            m->certified_artifact_boundary_certified_contexts++;
            m->certified_artifact_boundary_corpus_routed += (uint64_t)count;
        } else {
            m->certified_artifact_boundary_corpus_false_routes += (uint64_t)false_count;
        }
    }

    if (m->certified_artifact_boundary_actual_terms >=
        m->certified_artifact_boundary_corpus_routed) {
        m->certified_artifact_boundary_corpus_abstain =
            m->certified_artifact_boundary_actual_terms -
            m->certified_artifact_boundary_corpus_routed;
    }
    m->certified_artifact_boundary_route_bytes =
        (uint64_t)V * (uint64_t)(2u * sizeof(uint32_t) + sizeof(uint8_t));
    m->certified_artifact_boundary_obligation_audit =
        (m->certified_artifact_boundary_corpus_routed +
         m->certified_artifact_boundary_corpus_abstain ==
         m->certified_artifact_boundary_actual_terms) ? 1 : 0;
    m->certified_artifact_boundary_non_lookup_audit = 1;

    printf("[CERTIFIED_ARTIFACT_BOUNDARY] active=1 mode=copy-current-proof-transducer min_count=%u stride=%u terms=%llu candidate_contexts=%u certified_contexts=%u corpus_routed=%llu corpus_abstain=%llu corpus_false_routes=%llu route_bytes=%llu obligation_audit=%d non_lookup_audit=%d epsilon=%.9f ce_budget=%.9f overhead=%.6f kappa=%.6f\n",
           m->certified_artifact_boundary_min_count,
           m->certified_artifact_boundary_stride,
           (unsigned long long)m->certified_artifact_boundary_actual_terms,
           m->certified_artifact_boundary_candidate_contexts,
           m->certified_artifact_boundary_certified_contexts,
           (unsigned long long)m->certified_artifact_boundary_corpus_routed,
           (unsigned long long)m->certified_artifact_boundary_corpus_abstain,
           (unsigned long long)m->certified_artifact_boundary_corpus_false_routes,
           (unsigned long long)m->certified_artifact_boundary_route_bytes,
           m->certified_artifact_boundary_obligation_audit,
           m->certified_artifact_boundary_non_lookup_audit,
           m->certified_artifact_boundary_epsilon,
           m->certified_artifact_boundary_ce_budget,
           m->certified_artifact_boundary_overhead,
           m->certified_artifact_boundary_kappa);
    return 0;
}

uint32_t certified_artifact_boundary_predict(const Model *m, uint16_t context) {
    if (!m || !m->certified_artifact_boundary_gate_enabled ||
        !m->certified_artifact_boundary_certified ||
        context >= V12_VOCAB_SIZE ||
        !m->certified_artifact_boundary_certified[context]) {
        return UINT32_MAX;
    }
    return (uint32_t)context;
}

static uint32_t state_transition_atlas_key(uint16_t prev, uint16_t context) {
    return ((uint32_t)prev << 16) | (uint32_t)context;
}

static void state_transition_atlas_clear(Model *m) {
    if (!m) return;
    free(m->state_transition_atlas_keys);
    free(m->state_transition_atlas_targets);
    free(m->state_transition_atlas_counts);
    m->state_transition_atlas_keys = NULL;
    m->state_transition_atlas_targets = NULL;
    m->state_transition_atlas_counts = NULL;
    m->state_transition_atlas_actual_terms = 0;
    m->state_transition_atlas_capacity = 0;
    m->state_transition_atlas_entries = 0;
    m->state_transition_atlas_conflicts = 0;
    m->state_transition_atlas_gate_base_loss = 0.0f;
    m->state_transition_atlas_gate_atlas_loss = 0.0f;
    m->state_transition_atlas_gate_atlas_ratio = 0.0f;
    m->state_transition_atlas_gate_batches = 0;
    m->state_transition_atlas_gate_routed = 0;
    m->state_transition_atlas_gate_correct = 0;
    m->state_transition_atlas_gate_incorrect = 0;
    m->absorbed_state_atlas_gate_base_loss = 0.0f;
    m->absorbed_state_atlas_gate_residual_loss = 0.0f;
    m->absorbed_state_atlas_gate_residual_ratio = 0.0f;
    m->absorbed_state_atlas_gate_coverage = 0.0f;
    m->absorbed_state_atlas_gate_route_accuracy = 0.0f;
    m->absorbed_state_atlas_gate_priced_ratio = 0.0f;
    m->absorbed_state_atlas_gate_batches = 0;
    m->absorbed_state_atlas_gate_positions = 0;
    m->absorbed_state_atlas_gate_routed = 0;
    m->absorbed_state_atlas_gate_correct = 0;
    m->absorbed_state_atlas_gate_incorrect = 0;
}

void state_transition_atlas_configure(
    Model *m, int enabled, int gate_enabled, float logit_boost,
    uint32_t min_count, uint64_t requested_terms, uint32_t stride) {
    if (!m) return;
    state_transition_atlas_clear(m);
    m->state_transition_atlas_enabled = enabled ? 1 : 0;
    m->state_transition_atlas_gate_enabled = gate_enabled ? 1 : 0;
    m->state_transition_atlas_logit_boost = (logit_boost > 0.0f) ? logit_boost : 8.0f;
    m->state_transition_atlas_min_count = (min_count > 0u) ? min_count : 2u;
    m->state_transition_atlas_requested_terms = requested_terms;
    m->state_transition_atlas_stride = (stride > 0u) ? stride : 1u;
}

static int state_transition_atlas_rehash(Model *m, uint32_t new_capacity) {
    uint32_t *old_keys = m->state_transition_atlas_keys;
    uint32_t *old_targets = m->state_transition_atlas_targets;
    uint32_t *old_counts = m->state_transition_atlas_counts;
    uint32_t old_capacity = m->state_transition_atlas_capacity;

    uint32_t *new_keys = (uint32_t *)calloc(new_capacity, sizeof(uint32_t));
    uint32_t *new_targets = (uint32_t *)malloc((size_t)new_capacity * sizeof(uint32_t));
    uint32_t *new_counts = (uint32_t *)calloc(new_capacity, sizeof(uint32_t));
    if (!new_keys || !new_targets || !new_counts) {
        free(new_keys);
        free(new_targets);
        free(new_counts);
        return -1;
    }
    for (uint32_t i = 0; i < new_capacity; i++) new_targets[i] = UINT32_MAX;

    m->state_transition_atlas_keys = new_keys;
    m->state_transition_atlas_targets = new_targets;
    m->state_transition_atlas_counts = new_counts;
    m->state_transition_atlas_capacity = new_capacity;

    const uint32_t mask = new_capacity - 1u;
    for (uint32_t i = 0; i < old_capacity; i++) {
        if (old_counts[i] == 0u) continue;
        uint32_t key = old_keys[i];
        uint32_t slot = corpus_logit_prior_mix32(key) & mask;
        while (new_counts[slot] != 0u) slot = (slot + 1u) & mask;
        new_keys[slot] = key;
        new_targets[slot] = old_targets[i];
        new_counts[slot] = old_counts[i];
    }

    free(old_keys);
    free(old_targets);
    free(old_counts);
    return 0;
}

static int state_transition_atlas_add(Model *m, uint32_t key, uint16_t target) {
    if (m->state_transition_atlas_capacity == 0u) {
        if (state_transition_atlas_rehash(m, 65536u) != 0) return -1;
    }

    if ((uint64_t)(m->state_transition_atlas_entries + 1u) * 2u >=
        (uint64_t)m->state_transition_atlas_capacity) {
        if (state_transition_atlas_rehash(m, m->state_transition_atlas_capacity << 1) != 0) {
            return -1;
        }
    }

    const uint32_t mask = m->state_transition_atlas_capacity - 1u;
    uint32_t slot = corpus_logit_prior_mix32(key) & mask;
    while (m->state_transition_atlas_counts[slot] != 0u) {
        if (m->state_transition_atlas_keys[slot] == key) {
            if (m->state_transition_atlas_counts[slot] != UINT32_MAX) {
                m->state_transition_atlas_counts[slot]++;
            }
            if (m->state_transition_atlas_targets[slot] != (uint32_t)target) {
                m->state_transition_atlas_targets[slot] = UINT32_MAX;
            }
            return 0;
        }
        slot = (slot + 1u) & mask;
    }

    m->state_transition_atlas_keys[slot] = key;
    m->state_transition_atlas_targets[slot] = (uint32_t)target;
    m->state_transition_atlas_counts[slot] = 1u;
    m->state_transition_atlas_entries++;
    return 0;
}

int state_transition_atlas_build(Model *m, const uint16_t *tokens, size_t n_toks) {
    if (!m || !m->state_transition_atlas_enabled) return 0;
    if (!tokens || n_toks < 3) return -1;

    state_transition_atlas_clear(m);
    const uint32_t stride = m->state_transition_atlas_stride ? m->state_transition_atlas_stride : 1u;
    const uint64_t requested = m->state_transition_atlas_requested_terms;
    const uint64_t max_terms = requested ? requested : UINT64_MAX;

    for (size_t i = 1; i + 1 < n_toks && m->state_transition_atlas_actual_terms < max_terms; i += stride) {
        const uint16_t prev = tokens[i - 1];
        const uint16_t ctx = tokens[i];
        const uint16_t tgt = tokens[i + 1];
        if (prev >= V12_VOCAB_SIZE || ctx >= V12_VOCAB_SIZE || tgt >= V12_VOCAB_SIZE) continue;
        if (state_transition_atlas_add(m, state_transition_atlas_key(prev, ctx), tgt) != 0) {
            state_transition_atlas_clear(m);
            return -1;
        }
        m->state_transition_atlas_actual_terms++;
    }

    m->state_transition_atlas_entries = 0;
    const uint32_t min_count = m->state_transition_atlas_min_count ? m->state_transition_atlas_min_count : 2u;
    for (uint32_t i = 0; i < m->state_transition_atlas_capacity; i++) {
        if (m->state_transition_atlas_counts[i] == 0u) continue;
        if (m->state_transition_atlas_targets[i] == UINT32_MAX) {
            m->state_transition_atlas_conflicts++;
        }
        if (m->state_transition_atlas_targets[i] == UINT32_MAX ||
            m->state_transition_atlas_counts[i] < min_count) {
            m->state_transition_atlas_targets[i] = UINT32_MAX;
            continue;
        }
        m->state_transition_atlas_entries++;
    }

    printf("[STATE_TRANSITION_ATLAS] active=1 mode=exact-local-state-transition logit_boost=%.6f min_count=%u stride=%u terms=%llu entries=%u conflicts=%u table_capacity=%u\n",
           m->state_transition_atlas_logit_boost,
           m->state_transition_atlas_min_count,
           m->state_transition_atlas_stride,
           (unsigned long long)m->state_transition_atlas_actual_terms,
           m->state_transition_atlas_entries,
           m->state_transition_atlas_conflicts,
           m->state_transition_atlas_capacity);
    return 0;
}

uint32_t state_transition_atlas_predict(const Model *m, uint16_t prev, uint16_t context) {
    if (!m || !m->state_transition_atlas_enabled ||
        !m->state_transition_atlas_keys || !m->state_transition_atlas_targets ||
        !m->state_transition_atlas_counts || m->state_transition_atlas_capacity == 0u) {
        return UINT32_MAX;
    }
    if (prev >= V12_VOCAB_SIZE || context >= V12_VOCAB_SIZE) return UINT32_MAX;

    const uint32_t key = state_transition_atlas_key(prev, context);
    const uint32_t mask = m->state_transition_atlas_capacity - 1u;
    uint32_t slot = corpus_logit_prior_mix32(key) & mask;
    for (uint32_t probe = 0; probe < m->state_transition_atlas_capacity; probe++) {
        const uint32_t count = m->state_transition_atlas_counts[slot];
        if (count == 0u) return UINT32_MAX;
        if (m->state_transition_atlas_keys[slot] == key) {
            return m->state_transition_atlas_targets[slot];
        }
        slot = (slot + 1u) & mask;
    }
    return UINT32_MAX;
}

float state_transition_atlas_logit(
    const Model *m, uint16_t prev, uint16_t context, uint16_t target) {
    const uint32_t predicted = state_transition_atlas_predict(m, prev, context);
    if (predicted == UINT32_MAX || predicted != (uint32_t)target) return 0.0f;
    return m->state_transition_atlas_logit_boost;
}

static void counterfactual_packets_clear(Model *m) {
    if (!m) return;
    free(m->counterfactual_packets_keys);
    free(m->counterfactual_packets_target_a);
    free(m->counterfactual_packets_target_b);
    free(m->counterfactual_packets_count_a);
    free(m->counterfactual_packets_count_b);
    free(m->counterfactual_packets_count_other);
    m->counterfactual_packets_keys = NULL;
    m->counterfactual_packets_target_a = NULL;
    m->counterfactual_packets_target_b = NULL;
    m->counterfactual_packets_count_a = NULL;
    m->counterfactual_packets_count_b = NULL;
    m->counterfactual_packets_count_other = NULL;
    m->counterfactual_packets_actual_terms = 0;
    m->counterfactual_packets_capacity = 0;
    m->counterfactual_packets_entries = 0;
    m->counterfactual_packets_candidate_states = 0;
    m->counterfactual_packets_other_targets = 0;
    m->counterfactual_packets_gate_batches = 0;
    m->counterfactual_packets_gate_pairs = 0;
    m->counterfactual_packets_gate_primary = 0;
    m->counterfactual_packets_gate_secondary = 0;
    m->counterfactual_packets_gate_other = 0;
    m->counterfactual_packets_gate_abstain = 0;
    m->counterfactual_packets_gate_coverage = 0.0f;
    m->counterfactual_packets_gate_balance = 0.0f;
}

void counterfactual_packets_configure(
    Model *m, int enabled, int gate_enabled,
    uint32_t min_count, uint64_t requested_terms, uint32_t stride) {
    if (!m) return;
    counterfactual_packets_clear(m);
    m->counterfactual_packets_enabled = enabled ? 1 : 0;
    m->counterfactual_packets_gate_enabled = gate_enabled ? 1 : 0;
    m->counterfactual_packets_min_count = (min_count > 0u) ? min_count : 1u;
    m->counterfactual_packets_requested_terms = requested_terms;
    m->counterfactual_packets_stride = (stride > 0u) ? stride : 1u;
}

static int counterfactual_packets_rehash(Model *m, uint32_t new_capacity) {
    uint32_t *old_keys = m->counterfactual_packets_keys;
    uint32_t *old_a = m->counterfactual_packets_target_a;
    uint32_t *old_b = m->counterfactual_packets_target_b;
    uint32_t *old_ca = m->counterfactual_packets_count_a;
    uint32_t *old_cb = m->counterfactual_packets_count_b;
    uint32_t *old_co = m->counterfactual_packets_count_other;
    uint32_t old_capacity = m->counterfactual_packets_capacity;

    uint32_t *new_keys = (uint32_t *)calloc(new_capacity, sizeof(uint32_t));
    uint32_t *new_a = (uint32_t *)malloc((size_t)new_capacity * sizeof(uint32_t));
    uint32_t *new_b = (uint32_t *)malloc((size_t)new_capacity * sizeof(uint32_t));
    uint32_t *new_ca = (uint32_t *)calloc(new_capacity, sizeof(uint32_t));
    uint32_t *new_cb = (uint32_t *)calloc(new_capacity, sizeof(uint32_t));
    uint32_t *new_co = (uint32_t *)calloc(new_capacity, sizeof(uint32_t));
    if (!new_keys || !new_a || !new_b || !new_ca || !new_cb || !new_co) {
        free(new_keys);
        free(new_a);
        free(new_b);
        free(new_ca);
        free(new_cb);
        free(new_co);
        return -1;
    }
    for (uint32_t i = 0; i < new_capacity; i++) {
        new_a[i] = UINT32_MAX;
        new_b[i] = UINT32_MAX;
    }

    m->counterfactual_packets_keys = new_keys;
    m->counterfactual_packets_target_a = new_a;
    m->counterfactual_packets_target_b = new_b;
    m->counterfactual_packets_count_a = new_ca;
    m->counterfactual_packets_count_b = new_cb;
    m->counterfactual_packets_count_other = new_co;
    m->counterfactual_packets_capacity = new_capacity;

    const uint32_t mask = new_capacity - 1u;
    for (uint32_t i = 0; i < old_capacity; i++) {
        if (!old_ca || old_ca[i] == 0u) continue;
        const uint32_t key = old_keys[i];
        uint32_t slot = corpus_logit_prior_mix32(key) & mask;
        while (new_ca[slot] != 0u) slot = (slot + 1u) & mask;
        new_keys[slot] = key;
        new_a[slot] = old_a[i];
        new_b[slot] = old_b[i];
        new_ca[slot] = old_ca[i];
        new_cb[slot] = old_cb[i];
        new_co[slot] = old_co[i];
    }

    free(old_keys);
    free(old_a);
    free(old_b);
    free(old_ca);
    free(old_cb);
    free(old_co);
    return 0;
}

static int counterfactual_packets_add(Model *m, uint32_t key, uint16_t target) {
    if (m->counterfactual_packets_capacity == 0u) {
        if (counterfactual_packets_rehash(m, 65536u) != 0) return -1;
    }
    if ((uint64_t)(m->counterfactual_packets_candidate_states + 1u) * 2u >=
        (uint64_t)m->counterfactual_packets_capacity) {
        if (counterfactual_packets_rehash(m, m->counterfactual_packets_capacity << 1) != 0) {
            return -1;
        }
    }

    const uint32_t mask = m->counterfactual_packets_capacity - 1u;
    uint32_t slot = corpus_logit_prior_mix32(key) & mask;
    while (m->counterfactual_packets_count_a[slot] != 0u) {
        if (m->counterfactual_packets_keys[slot] == key) {
            const uint32_t tgt = (uint32_t)target;
            if (m->counterfactual_packets_target_a[slot] == tgt) {
                m->counterfactual_packets_count_a[slot]++;
            } else if (m->counterfactual_packets_target_b[slot] == tgt) {
                m->counterfactual_packets_count_b[slot]++;
            } else if (m->counterfactual_packets_target_b[slot] == UINT32_MAX) {
                m->counterfactual_packets_target_b[slot] = tgt;
                m->counterfactual_packets_count_b[slot] = 1u;
            } else {
                m->counterfactual_packets_count_other[slot]++;
                m->counterfactual_packets_other_targets++;
            }
            return 0;
        }
        slot = (slot + 1u) & mask;
    }

    m->counterfactual_packets_keys[slot] = key;
    m->counterfactual_packets_target_a[slot] = (uint32_t)target;
    m->counterfactual_packets_target_b[slot] = UINT32_MAX;
    m->counterfactual_packets_count_a[slot] = 1u;
    m->counterfactual_packets_count_b[slot] = 0u;
    m->counterfactual_packets_count_other[slot] = 0u;
    m->counterfactual_packets_candidate_states++;
    return 0;
}

int counterfactual_packets_build(Model *m, const uint16_t *tokens, size_t n_toks) {
    if (!m || !m->counterfactual_packets_enabled) return 0;
    if (!tokens || n_toks < 3) return -1;

    counterfactual_packets_clear(m);
    const uint32_t stride = m->counterfactual_packets_stride ? m->counterfactual_packets_stride : 1u;
    const uint64_t requested = m->counterfactual_packets_requested_terms;
    const uint64_t max_terms = requested ? requested : UINT64_MAX;

    for (size_t i = 1; i + 1 < n_toks && m->counterfactual_packets_actual_terms < max_terms; i += stride) {
        const uint16_t prev = tokens[i - 1];
        const uint16_t ctx = tokens[i];
        const uint16_t tgt = tokens[i + 1];
        if (prev >= V12_VOCAB_SIZE || ctx >= V12_VOCAB_SIZE || tgt >= V12_VOCAB_SIZE) continue;
        if (counterfactual_packets_add(m, state_transition_atlas_key(prev, ctx), tgt) != 0) {
            counterfactual_packets_clear(m);
            return -1;
        }
        m->counterfactual_packets_actual_terms++;
    }

    m->counterfactual_packets_entries = 0;
    const uint32_t min_count = m->counterfactual_packets_min_count ? m->counterfactual_packets_min_count : 1u;
    for (uint32_t i = 0; i < m->counterfactual_packets_capacity; i++) {
        if (m->counterfactual_packets_count_a[i] == 0u) continue;
        if (m->counterfactual_packets_target_b[i] == UINT32_MAX) continue;
        if (m->counterfactual_packets_count_a[i] < min_count ||
            m->counterfactual_packets_count_b[i] < min_count) {
            continue;
        }
        m->counterfactual_packets_entries++;
    }

    printf("[COUNTERFACTUAL_PACKETS] active=1 mode=local-state-first-two-target-packets min_count=%u stride=%u terms=%llu pair_states=%u candidate_states=%u other_targets=%llu table_capacity=%u\n",
           m->counterfactual_packets_min_count,
           m->counterfactual_packets_stride,
           (unsigned long long)m->counterfactual_packets_actual_terms,
           m->counterfactual_packets_entries,
           m->counterfactual_packets_candidate_states,
           (unsigned long long)m->counterfactual_packets_other_targets,
           m->counterfactual_packets_capacity);
    return 0;
}

uint32_t counterfactual_packets_alternative(
    const Model *m, uint16_t prev, uint16_t context, uint16_t target, uint32_t *side) {
    if (side) *side = 0u;
    if (!m || !m->counterfactual_packets_enabled ||
        !m->counterfactual_packets_keys ||
        !m->counterfactual_packets_target_a ||
        !m->counterfactual_packets_target_b ||
        !m->counterfactual_packets_count_a ||
        !m->counterfactual_packets_count_b ||
        m->counterfactual_packets_capacity == 0u) {
        return UINT32_MAX;
    }
    if (prev >= V12_VOCAB_SIZE || context >= V12_VOCAB_SIZE || target >= V12_VOCAB_SIZE) {
        return UINT32_MAX;
    }

    const uint32_t key = state_transition_atlas_key(prev, context);
    const uint32_t mask = m->counterfactual_packets_capacity - 1u;
    const uint32_t min_count = m->counterfactual_packets_min_count ? m->counterfactual_packets_min_count : 1u;
    uint32_t slot = corpus_logit_prior_mix32(key) & mask;
    for (uint32_t probe = 0; probe < m->counterfactual_packets_capacity; probe++) {
        const uint32_t count_a = m->counterfactual_packets_count_a[slot];
        if (count_a == 0u) return UINT32_MAX;
        if (m->counterfactual_packets_keys[slot] == key) {
            const uint32_t a = m->counterfactual_packets_target_a[slot];
            const uint32_t b = m->counterfactual_packets_target_b[slot];
            const uint32_t count_b = m->counterfactual_packets_count_b[slot];
            if (b == UINT32_MAX || count_a < min_count || count_b < min_count) {
                return UINT32_MAX;
            }
            if ((uint32_t)target == a) {
                if (side) *side = 1u;
                return b;
            }
            if ((uint32_t)target == b) {
                if (side) *side = 2u;
                return a;
            }
            if (side) *side = 3u;
            return UINT32_MAX;
        }
        slot = (slot + 1u) & mask;
    }
    return UINT32_MAX;
}

static void error_causal_lattice_clear(Model *m) {
    if (!m) return;
    free(m->error_causal_target_counts);
    free(m->error_causal_prereq_counts);
    m->error_causal_target_counts = NULL;
    m->error_causal_prereq_counts = NULL;
    m->error_causal_lattice_actual_terms = 0;
    m->error_causal_lattice_target_atoms = 0;
    m->error_causal_lattice_prereq_atoms = 0;
    m->error_causal_lattice_gate_batches = 0;
    m->error_causal_lattice_gate_positions = 0;
    m->error_causal_lattice_gate_target_covered = 0;
    m->error_causal_lattice_gate_prereq_ready = 0;
    m->error_causal_lattice_gate_orderable = 0;
    m->error_causal_lattice_gate_rare_positions = 0;
    m->error_causal_lattice_gate_rare_orderable = 0;
    m->error_causal_lattice_gate_target_coverage = 0.0f;
    m->error_causal_lattice_gate_prereq_readiness = 0.0f;
    m->error_causal_lattice_gate_orderable_fraction = 0.0f;
    m->error_causal_lattice_gate_rare_coverage = 0.0f;
}

void error_causal_lattice_configure(
    Model *m, int enabled, int gate_enabled,
    uint32_t min_count, uint64_t requested_terms, uint32_t stride) {
    if (!m) return;
    error_causal_lattice_clear(m);
    m->error_causal_lattice_enabled = enabled ? 1 : 0;
    m->error_causal_lattice_gate_enabled = gate_enabled ? 1 : 0;
    m->error_causal_lattice_min_count = min_count > 0u ? min_count : 1u;
    m->error_causal_lattice_requested_terms = requested_terms;
    m->error_causal_lattice_stride = stride > 0u ? stride : 1u;
}

int error_causal_lattice_build(Model *m, const uint16_t *tokens, size_t n_toks) {
    if (!m || !m->error_causal_lattice_enabled) return 0;
    if (!tokens || n_toks < 2) return -1;

    error_causal_lattice_clear(m);
    m->error_causal_target_counts =
        (uint32_t *)calloc((size_t)V12_VOCAB_SIZE, sizeof(uint32_t));
    m->error_causal_prereq_counts =
        (uint32_t *)calloc((size_t)V12_VOCAB_SIZE, sizeof(uint32_t));
    if (!m->error_causal_target_counts || !m->error_causal_prereq_counts) {
        error_causal_lattice_clear(m);
        return -1;
    }

    const uint32_t stride = m->error_causal_lattice_stride
        ? m->error_causal_lattice_stride
        : 1u;
    const uint64_t requested = m->error_causal_lattice_requested_terms;
    const uint64_t max_terms = requested ? requested : UINT64_MAX;

    for (size_t i = 0; i + 1 < n_toks && m->error_causal_lattice_actual_terms < max_terms; i += stride) {
        const uint16_t prereq = tokens[i];
        const uint16_t target = tokens[i + 1];
        if (prereq >= V12_VOCAB_SIZE || target >= V12_VOCAB_SIZE) continue;
        m->error_causal_prereq_counts[prereq]++;
        m->error_causal_target_counts[target]++;
        m->error_causal_lattice_actual_terms++;
    }

    for (uint32_t i = 0; i < (uint32_t)V12_VOCAB_SIZE; i++) {
        if (m->error_causal_target_counts[i] > 0u) {
            m->error_causal_lattice_target_atoms++;
        }
        if (m->error_causal_prereq_counts[i] > 0u) {
            m->error_causal_lattice_prereq_atoms++;
        }
    }

    printf("[ERROR_CAUSAL_LATTICE] active=1 mode=target-prerequisite-atom-support min_count=%u stride=%u terms=%llu target_atoms=%u prereq_atoms=%u\n",
           m->error_causal_lattice_min_count,
           m->error_causal_lattice_stride,
           (unsigned long long)m->error_causal_lattice_actual_terms,
           m->error_causal_lattice_target_atoms,
           m->error_causal_lattice_prereq_atoms);
    return 0;
}

static uint64_t exact_transition_quotient_key(
    uint16_t prev, uint16_t context, uint16_t target) {
    return ((uint64_t)prev << 32)
         | ((uint64_t)context << 16)
         | (uint64_t)target;
}

static uint32_t exact_transition_quotient_hash(uint64_t key) {
    const uint32_t lo = (uint32_t)key;
    const uint32_t hi = (uint32_t)(key >> 32);
    return corpus_logit_prior_mix32(lo ^ corpus_logit_prior_mix32(hi));
}

static void exact_transition_quotient_clear(Model *m) {
    if (!m) return;
    free(m->exact_transition_quotient_keys);
    free(m->exact_transition_quotient_counts);
    m->exact_transition_quotient_keys = NULL;
    m->exact_transition_quotient_counts = NULL;
    m->exact_transition_quotient_actual_terms = 0;
    m->exact_transition_quotient_capacity = 0;
    m->exact_transition_quotient_unique_terms = 0;
    m->exact_transition_quotient_supported_mass = 0;
    m->exact_transition_quotient_gate_batches = 0;
    m->exact_transition_quotient_gate_positions = 0;
    m->exact_transition_quotient_gate_supported = 0;
    m->exact_transition_quotient_gate_saved_terms = 0;
    m->exact_transition_quotient_gate_multiplicity_sum = 0;
    m->exact_transition_quotient_gate_corpus_ratio = 0.0f;
    m->exact_transition_quotient_gate_live_coverage = 0.0f;
    m->exact_transition_quotient_gate_saved_fraction = 0.0f;
    m->exact_transition_quotient_gate_mean_multiplicity = 0.0f;
}

void exact_transition_quotient_configure(
    Model *m, int enabled, int gate_enabled,
    uint32_t min_count, uint64_t requested_terms, uint32_t stride) {
    if (!m) return;
    exact_transition_quotient_clear(m);
    m->exact_transition_quotient_enabled = enabled ? 1 : 0;
    m->exact_transition_quotient_gate_enabled = gate_enabled ? 1 : 0;
    m->exact_transition_quotient_min_count = min_count > 0u ? min_count : 1u;
    m->exact_transition_quotient_requested_terms = requested_terms;
    m->exact_transition_quotient_stride = stride > 0u ? stride : 1u;
}

static int exact_transition_quotient_rehash(Model *m, uint32_t new_capacity) {
    uint64_t *old_keys = m->exact_transition_quotient_keys;
    uint32_t *old_counts = m->exact_transition_quotient_counts;
    uint32_t old_capacity = m->exact_transition_quotient_capacity;

    uint64_t *new_keys = (uint64_t *)calloc((size_t)new_capacity, sizeof(uint64_t));
    uint32_t *new_counts = (uint32_t *)calloc((size_t)new_capacity, sizeof(uint32_t));
    if (!new_keys || !new_counts) {
        free(new_keys);
        free(new_counts);
        return -1;
    }

    m->exact_transition_quotient_keys = new_keys;
    m->exact_transition_quotient_counts = new_counts;
    m->exact_transition_quotient_capacity = new_capacity;

    if (old_keys && old_counts) {
        const uint32_t mask = new_capacity - 1u;
        for (uint32_t i = 0; i < old_capacity; i++) {
            const uint32_t count = old_counts[i];
            if (count == 0u) continue;
            const uint64_t key = old_keys[i];
            uint32_t slot = exact_transition_quotient_hash(key) & mask;
            while (new_counts[slot] != 0u) {
                slot = (slot + 1u) & mask;
            }
            new_keys[slot] = key;
            new_counts[slot] = count;
        }
    }

    free(old_keys);
    free(old_counts);
    return 0;
}

static int exact_transition_quotient_add(Model *m, uint64_t key) {
    if (m->exact_transition_quotient_capacity == 0u) {
        if (exact_transition_quotient_rehash(m, 65536u) != 0) return -1;
    }
    if ((uint64_t)(m->exact_transition_quotient_unique_terms + 1u) * 2u >=
        (uint64_t)m->exact_transition_quotient_capacity) {
        if (exact_transition_quotient_rehash(
                m, m->exact_transition_quotient_capacity << 1) != 0) {
            return -1;
        }
    }

    const uint32_t mask = m->exact_transition_quotient_capacity - 1u;
    uint32_t slot = exact_transition_quotient_hash(key) & mask;
    while (m->exact_transition_quotient_counts[slot] != 0u) {
        if (m->exact_transition_quotient_keys[slot] == key) {
            m->exact_transition_quotient_counts[slot]++;
            return 0;
        }
        slot = (slot + 1u) & mask;
    }

    m->exact_transition_quotient_keys[slot] = key;
    m->exact_transition_quotient_counts[slot] = 1u;
    m->exact_transition_quotient_unique_terms++;
    return 0;
}

int exact_transition_quotient_build(Model *m, const uint16_t *tokens, size_t n_toks) {
    if (!m || !m->exact_transition_quotient_enabled) return 0;
    if (!tokens || n_toks < 3) return -1;

    exact_transition_quotient_clear(m);
    const uint32_t stride = m->exact_transition_quotient_stride
        ? m->exact_transition_quotient_stride
        : 1u;
    const uint64_t requested = m->exact_transition_quotient_requested_terms;
    const uint64_t max_terms = requested ? requested : UINT64_MAX;

    for (size_t i = 1; i + 1 < n_toks && m->exact_transition_quotient_actual_terms < max_terms; i += stride) {
        const uint16_t prev = tokens[i - 1];
        const uint16_t ctx = tokens[i];
        const uint16_t tgt = tokens[i + 1];
        if (prev >= V12_VOCAB_SIZE || ctx >= V12_VOCAB_SIZE || tgt >= V12_VOCAB_SIZE) continue;
        const uint64_t key = exact_transition_quotient_key(prev, ctx, tgt);
        if (exact_transition_quotient_add(m, key) != 0) {
            exact_transition_quotient_clear(m);
            return -1;
        }
        m->exact_transition_quotient_actual_terms++;
    }

    const uint32_t min_count = m->exact_transition_quotient_min_count
        ? m->exact_transition_quotient_min_count
        : 1u;
    for (uint32_t i = 0; i < m->exact_transition_quotient_capacity; i++) {
        const uint32_t count = m->exact_transition_quotient_counts[i];
        if (count >= min_count) {
            m->exact_transition_quotient_supported_mass += (uint64_t)count;
        }
    }
    m->exact_transition_quotient_gate_corpus_ratio =
        m->exact_transition_quotient_actual_terms > 0
            ? (float)((double)m->exact_transition_quotient_unique_terms /
                      (double)m->exact_transition_quotient_actual_terms)
            : 0.0f;

    printf("[EXACT_TRANSITION_QUOTIENT] active=1 mode=exact-local-triple-multiplicity min_count=%u stride=%u terms=%llu unique_terms=%u supported_mass=%llu quotient_ratio=%.6f table_capacity=%u\n",
           m->exact_transition_quotient_min_count,
           m->exact_transition_quotient_stride,
           (unsigned long long)m->exact_transition_quotient_actual_terms,
           m->exact_transition_quotient_unique_terms,
           (unsigned long long)m->exact_transition_quotient_supported_mass,
           m->exact_transition_quotient_gate_corpus_ratio,
           m->exact_transition_quotient_capacity);
    return 0;
}

uint32_t exact_transition_quotient_count(
    const Model *m, uint16_t prev, uint16_t context, uint16_t target) {
    if (!m || !m->exact_transition_quotient_enabled ||
        !m->exact_transition_quotient_keys ||
        !m->exact_transition_quotient_counts ||
        m->exact_transition_quotient_capacity == 0u) {
        return 0u;
    }
    if (prev >= V12_VOCAB_SIZE || context >= V12_VOCAB_SIZE || target >= V12_VOCAB_SIZE) {
        return 0u;
    }

    const uint64_t key = exact_transition_quotient_key(prev, context, target);
    const uint32_t mask = m->exact_transition_quotient_capacity - 1u;
    uint32_t slot = exact_transition_quotient_hash(key) & mask;
    for (uint32_t probe = 0; probe < m->exact_transition_quotient_capacity; probe++) {
        const uint32_t count = m->exact_transition_quotient_counts[slot];
        if (count == 0u) return 0u;
        if (m->exact_transition_quotient_keys[slot] == key) return count;
        slot = (slot + 1u) & mask;
    }
    return 0u;
}

static uint64_t macro_ce_context_key(uint16_t left, uint16_t right) {
    return ((uint64_t)left << 16) | (uint64_t)right;
}

static uint64_t macro_ce_triple_key(uint64_t context_key, uint16_t target) {
    return (context_key << 16) | (uint64_t)target;
}

static uint16_t macro_ce_triple_target(uint64_t triple_key) {
    return (uint16_t)(triple_key & 0xffffu);
}

static uint64_t macro_ce_triple_context(uint64_t triple_key) {
    return triple_key >> 16;
}

static uint64_t macro_ce_class_key(
    uint16_t top1, uint16_t top2, uint32_t total, uint32_t top1_count,
    uint32_t bins) {
    if (bins == 0u) bins = 4u;
    double dominance = total > 0u ? (double)top1_count / (double)total : 0.0;
    uint32_t bin = (uint32_t)(dominance * (double)bins);
    if (bin >= bins) bin = bins - 1u;
    return 1ull
         | ((uint64_t)top1 << 24)
         | ((uint64_t)top2 << 8)
         | (uint64_t)(bin & 0xffu);
}

static uint32_t macro_ce_hash64(uint64_t key) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return (uint32_t)(key ^ (key >> 32));
}

static int macro_ce_count_rehash(
    uint64_t **keys, uint32_t **counts, uint32_t *capacity,
    uint32_t new_capacity) {
    uint64_t *old_keys = *keys;
    uint32_t *old_counts = *counts;
    uint32_t old_capacity = *capacity;
    uint64_t *new_keys = (uint64_t *)calloc((size_t)new_capacity, sizeof(uint64_t));
    uint32_t *new_counts = (uint32_t *)calloc((size_t)new_capacity, sizeof(uint32_t));
    if (!new_keys || !new_counts) {
        free(new_keys);
        free(new_counts);
        return -1;
    }

    *keys = new_keys;
    *counts = new_counts;
    *capacity = new_capacity;

    if (old_keys && old_counts) {
        const uint32_t mask = new_capacity - 1u;
        for (uint32_t i = 0; i < old_capacity; i++) {
            if (old_counts[i] == 0u) continue;
            uint32_t slot = macro_ce_hash64(old_keys[i]) & mask;
            while (new_counts[slot] != 0u) slot = (slot + 1u) & mask;
            new_keys[slot] = old_keys[i];
            new_counts[slot] = old_counts[i];
        }
    }

    free(old_keys);
    free(old_counts);
    return 0;
}

static int macro_ce_count_add(
    uint64_t **keys, uint32_t **counts, uint32_t *capacity,
    uint32_t *unique, uint64_t key, uint32_t amount,
    uint32_t *out_count) {
    if (*capacity == 0u) {
        if (macro_ce_count_rehash(keys, counts, capacity, 65536u) != 0) return -1;
    }
    if ((uint64_t)(*unique + 1u) * 2u >= (uint64_t)(*capacity)) {
        if (macro_ce_count_rehash(keys, counts, capacity, (*capacity) << 1) != 0) {
            return -1;
        }
    }

    const uint32_t mask = (*capacity) - 1u;
    uint32_t slot = macro_ce_hash64(key) & mask;
    while ((*counts)[slot] != 0u) {
        if ((*keys)[slot] == key) {
            (*counts)[slot] += amount;
            if (out_count) *out_count = (*counts)[slot];
            return 0;
        }
        slot = (slot + 1u) & mask;
    }

    (*keys)[slot] = key;
    (*counts)[slot] = amount;
    (*unique)++;
    if (out_count) *out_count = amount;
    return 0;
}

static uint32_t macro_ce_count_get(
    const uint64_t *keys, const uint32_t *counts,
    uint32_t capacity, uint64_t key) {
    if (!keys || !counts || capacity == 0u) return 0u;
    const uint32_t mask = capacity - 1u;
    uint32_t slot = macro_ce_hash64(key) & mask;
    for (uint32_t probe = 0; probe < capacity; probe++) {
        const uint32_t count = counts[slot];
        if (count == 0u) return 0u;
        if (keys[slot] == key) return count;
        slot = (slot + 1u) & mask;
    }
    return 0u;
}

static int macro_ce_context_rehash(Model *m, uint32_t new_capacity) {
    uint64_t *old_keys = m->macro_ce_quotient_context_keys;
    uint32_t *old_totals = m->macro_ce_quotient_context_totals;
    uint64_t *old_classes = m->macro_ce_quotient_context_class_keys;
    uint16_t *old_top1_token = m->macro_ce_quotient_context_top1_token;
    uint16_t *old_top2_token = m->macro_ce_quotient_context_top2_token;
    uint32_t *old_top1_count = m->macro_ce_quotient_context_top1_count;
    uint32_t *old_top2_count = m->macro_ce_quotient_context_top2_count;
    uint32_t old_capacity = m->macro_ce_quotient_context_capacity;

    uint64_t *new_keys = (uint64_t *)calloc((size_t)new_capacity, sizeof(uint64_t));
    uint32_t *new_totals = (uint32_t *)calloc((size_t)new_capacity, sizeof(uint32_t));
    uint64_t *new_classes = (uint64_t *)calloc((size_t)new_capacity, sizeof(uint64_t));
    uint16_t *new_top1_token = (uint16_t *)calloc((size_t)new_capacity, sizeof(uint16_t));
    uint16_t *new_top2_token = (uint16_t *)calloc((size_t)new_capacity, sizeof(uint16_t));
    uint32_t *new_top1_count = (uint32_t *)calloc((size_t)new_capacity, sizeof(uint32_t));
    uint32_t *new_top2_count = (uint32_t *)calloc((size_t)new_capacity, sizeof(uint32_t));
    if (!new_keys || !new_totals || !new_classes || !new_top1_token ||
        !new_top2_token || !new_top1_count || !new_top2_count) {
        free(new_keys);
        free(new_totals);
        free(new_classes);
        free(new_top1_token);
        free(new_top2_token);
        free(new_top1_count);
        free(new_top2_count);
        return -1;
    }

    m->macro_ce_quotient_context_keys = new_keys;
    m->macro_ce_quotient_context_totals = new_totals;
    m->macro_ce_quotient_context_class_keys = new_classes;
    m->macro_ce_quotient_context_top1_token = new_top1_token;
    m->macro_ce_quotient_context_top2_token = new_top2_token;
    m->macro_ce_quotient_context_top1_count = new_top1_count;
    m->macro_ce_quotient_context_top2_count = new_top2_count;
    m->macro_ce_quotient_context_capacity = new_capacity;

    if (old_keys && old_totals) {
        const uint32_t mask = new_capacity - 1u;
        for (uint32_t i = 0; i < old_capacity; i++) {
            if (old_totals[i] == 0u) continue;
            uint32_t slot = macro_ce_hash64(old_keys[i]) & mask;
            while (new_totals[slot] != 0u) slot = (slot + 1u) & mask;
            new_keys[slot] = old_keys[i];
            new_totals[slot] = old_totals[i];
            new_classes[slot] = old_classes ? old_classes[i] : 0u;
            new_top1_token[slot] = old_top1_token ? old_top1_token[i] : 0u;
            new_top2_token[slot] = old_top2_token ? old_top2_token[i] : 0u;
            new_top1_count[slot] = old_top1_count ? old_top1_count[i] : 0u;
            new_top2_count[slot] = old_top2_count ? old_top2_count[i] : 0u;
        }
    }

    free(old_keys);
    free(old_totals);
    free(old_classes);
    free(old_top1_token);
    free(old_top2_token);
    free(old_top1_count);
    free(old_top2_count);
    return 0;
}

static int macro_ce_context_slot(Model *m, uint64_t key, int create) {
    if (!m) return -1;
    if (m->macro_ce_quotient_context_capacity == 0u) {
        if (!create) return -1;
        if (macro_ce_context_rehash(m, 65536u) != 0) return -1;
    }
    if (create &&
        (uint64_t)(m->macro_ce_quotient_unique_contexts_raw + 1u) * 2u >=
        (uint64_t)m->macro_ce_quotient_context_capacity) {
        if (macro_ce_context_rehash(
                m, m->macro_ce_quotient_context_capacity << 1) != 0) {
            return -1;
        }
    }

    const uint32_t mask = m->macro_ce_quotient_context_capacity - 1u;
    uint32_t slot = macro_ce_hash64(key) & mask;
    for (uint32_t probe = 0; probe < m->macro_ce_quotient_context_capacity; probe++) {
        const uint32_t total = m->macro_ce_quotient_context_totals[slot];
        if (total == 0u) {
            if (!create) return -1;
            m->macro_ce_quotient_context_keys[slot] = key;
            m->macro_ce_quotient_unique_contexts_raw++;
            return (int)slot;
        }
        if (m->macro_ce_quotient_context_keys[slot] == key) return (int)slot;
        slot = (slot + 1u) & mask;
    }
    return -1;
}

static void macro_ce_context_update_top(
    Model *m, uint32_t slot, uint16_t target, uint32_t target_count) {
    if (m->macro_ce_quotient_context_top1_count[slot] == 0u ||
        target == m->macro_ce_quotient_context_top1_token[slot] ||
        target_count > m->macro_ce_quotient_context_top1_count[slot]) {
        if (target != m->macro_ce_quotient_context_top1_token[slot]) {
            m->macro_ce_quotient_context_top2_token[slot] =
                m->macro_ce_quotient_context_top1_token[slot];
            m->macro_ce_quotient_context_top2_count[slot] =
                m->macro_ce_quotient_context_top1_count[slot];
        }
        m->macro_ce_quotient_context_top1_token[slot] = target;
        m->macro_ce_quotient_context_top1_count[slot] = target_count;
    } else if (target != m->macro_ce_quotient_context_top1_token[slot] &&
               (m->macro_ce_quotient_context_top2_count[slot] == 0u ||
                target == m->macro_ce_quotient_context_top2_token[slot] ||
                target_count > m->macro_ce_quotient_context_top2_count[slot])) {
        m->macro_ce_quotient_context_top2_token[slot] = target;
        m->macro_ce_quotient_context_top2_count[slot] = target_count;
    }
}

static void macro_ce_quotient_clear(Model *m) {
    if (!m) return;
    free(m->macro_ce_quotient_context_keys);
    free(m->macro_ce_quotient_context_totals);
    free(m->macro_ce_quotient_context_class_keys);
    free(m->macro_ce_quotient_context_top1_token);
    free(m->macro_ce_quotient_context_top2_token);
    free(m->macro_ce_quotient_context_top1_count);
    free(m->macro_ce_quotient_context_top2_count);
    free(m->macro_ce_quotient_triple_keys);
    free(m->macro_ce_quotient_triple_counts);
    free(m->macro_ce_quotient_class_keys);
    free(m->macro_ce_quotient_class_totals);
    free(m->macro_ce_quotient_class_target_keys);
    free(m->macro_ce_quotient_class_target_counts);
    free(m->macro_ce_quotient_random_keys);
    free(m->macro_ce_quotient_random_totals);
    free(m->macro_ce_quotient_random_target_keys);
    free(m->macro_ce_quotient_random_target_counts);

    m->macro_ce_quotient_context_keys = NULL;
    m->macro_ce_quotient_context_totals = NULL;
    m->macro_ce_quotient_context_class_keys = NULL;
    m->macro_ce_quotient_context_top1_token = NULL;
    m->macro_ce_quotient_context_top2_token = NULL;
    m->macro_ce_quotient_context_top1_count = NULL;
    m->macro_ce_quotient_context_top2_count = NULL;
    m->macro_ce_quotient_triple_keys = NULL;
    m->macro_ce_quotient_triple_counts = NULL;
    m->macro_ce_quotient_class_keys = NULL;
    m->macro_ce_quotient_class_totals = NULL;
    m->macro_ce_quotient_class_target_keys = NULL;
    m->macro_ce_quotient_class_target_counts = NULL;
    m->macro_ce_quotient_random_keys = NULL;
    m->macro_ce_quotient_random_totals = NULL;
    m->macro_ce_quotient_random_target_keys = NULL;
    m->macro_ce_quotient_random_target_counts = NULL;

    m->macro_ce_quotient_actual_terms = 0;
    m->macro_ce_quotient_context_capacity = 0;
    m->macro_ce_quotient_triple_capacity = 0;
    m->macro_ce_quotient_class_capacity = 0;
    m->macro_ce_quotient_class_target_capacity = 0;
    m->macro_ce_quotient_random_capacity = 0;
    m->macro_ce_quotient_random_target_capacity = 0;
    m->macro_ce_quotient_unique_contexts_raw = 0;
    m->macro_ce_quotient_unique_triples = 0;
    m->macro_ce_quotient_unique_contexts = 0;
    m->macro_ce_quotient_quotient_classes = 0;
    m->macro_ce_quotient_random_buckets = 0;
    m->macro_ce_quotient_observer_batches = 0;
    m->macro_ce_quotient_observer_positions = 0;
    m->macro_ce_quotient_observer_covered = 0;
    m->macro_ce_quotient_context_ce_sum = 0.0;
    m->macro_ce_quotient_quotient_ce_sum = 0.0;
    m->macro_ce_quotient_random_ce_sum = 0.0;
    m->macro_ce_quotient_coverage = 0.0f;
    m->macro_ce_quotient_context_ce = 0.0f;
    m->macro_ce_quotient_quotient_ce = 0.0f;
    m->macro_ce_quotient_random_ce = 0.0f;
    m->macro_ce_quotient_ce_excess = 0.0f;
    m->macro_ce_quotient_random_advantage = 0.0f;
    m->macro_ce_quotient_ratio = 0.0f;
    m->macro_ce_quotient_residual_price = 0.0f;
    m->macro_ce_quotient_priced_work_ratio = 0.0f;
    m->macro_ce_quotient_net_speedup_estimate = 0.0f;
    m->macro_ce_quotient_training_batches = 0;
    m->macro_ce_quotient_training_rows = 0;
    m->macro_ce_quotient_training_covered = 0;
    m->macro_ce_quotient_training_coverage = 0.0f;
    m->macro_ce_quotient_training_effective_weight = 1.0f;
}

void macro_ce_quotient_configure(
    Model *m, int observer_enabled, int training_enabled, uint32_t min_support,
    uint64_t requested_terms, uint32_t stride,
    float build_fraction, float alpha,
    float residual_price_scale, float overhead,
    uint32_t signature_bins, uint32_t seed,
    float training_weight) {
    if (!m) return;
    macro_ce_quotient_clear(m);
    m->macro_ce_quotient_observer_enabled = (observer_enabled || training_enabled) ? 1 : 0;
    m->macro_ce_quotient_training_enabled = training_enabled ? 1 : 0;
    m->macro_ce_quotient_min_support = min_support > 0u ? min_support : 4u;
    m->macro_ce_quotient_requested_terms = requested_terms;
    m->macro_ce_quotient_stride = stride > 0u ? stride : 1u;
    m->macro_ce_quotient_build_fraction =
        (build_fraction > 0.0f && build_fraction <= 1.0f) ? build_fraction : 0.7f;
    m->macro_ce_quotient_alpha = alpha > 0.0f ? alpha : 0.05f;
    m->macro_ce_quotient_residual_price_scale =
        residual_price_scale >= 0.0f ? residual_price_scale : 1.0f;
    m->macro_ce_quotient_overhead = overhead >= 0.0f ? overhead : 0.0f;
    m->macro_ce_quotient_signature_bins = signature_bins > 0u ? signature_bins : 4u;
    m->macro_ce_quotient_seed = seed != 0u ? seed : 42140u;
    m->macro_ce_quotient_training_weight =
        (training_weight >= 0.0f && training_weight <= 1.0f) ? training_weight : 0.5f;
}

int macro_ce_quotient_build(Model *m, const uint16_t *tokens, size_t n_toks) {
    if (!m || !m->macro_ce_quotient_observer_enabled) return 0;
    if (!tokens || n_toks < 8u) return -1;

    macro_ce_quotient_clear(m);
    m->macro_ce_quotient_observer_enabled = 1;
    const uint32_t stride = m->macro_ce_quotient_stride
        ? m->macro_ce_quotient_stride
        : 1u;
    const uint32_t min_support = m->macro_ce_quotient_min_support
        ? m->macro_ce_quotient_min_support
        : 4u;
    const uint64_t requested = m->macro_ce_quotient_requested_terms;
    const uint64_t max_terms = requested ? requested : UINT64_MAX;
    size_t train_limit = (size_t)((double)n_toks * (double)m->macro_ce_quotient_build_fraction);
    if (train_limit > n_toks) train_limit = n_toks;
    if (train_limit < 4u) train_limit = n_toks;

    for (size_t i = 2; i < train_limit && m->macro_ce_quotient_actual_terms < max_terms; i += stride) {
        const uint16_t left = tokens[i - 2];
        const uint16_t right = tokens[i - 1];
        const uint16_t target = tokens[i];
        if (left >= V12_VOCAB_SIZE || right >= V12_VOCAB_SIZE || target >= V12_VOCAB_SIZE) {
            continue;
        }
        const uint64_t ctx_key = macro_ce_context_key(left, right);
        const int ctx_slot_i = macro_ce_context_slot(m, ctx_key, 1);
        if (ctx_slot_i < 0) {
            macro_ce_quotient_clear(m);
            return -1;
        }
        const uint32_t ctx_slot = (uint32_t)ctx_slot_i;
        m->macro_ce_quotient_context_totals[ctx_slot]++;

        uint32_t triple_count = 0;
        if (macro_ce_count_add(
                &m->macro_ce_quotient_triple_keys,
                &m->macro_ce_quotient_triple_counts,
                &m->macro_ce_quotient_triple_capacity,
                &m->macro_ce_quotient_unique_triples,
                macro_ce_triple_key(ctx_key, target), 1u, &triple_count) != 0) {
            macro_ce_quotient_clear(m);
            return -1;
        }
        macro_ce_context_update_top(m, ctx_slot, target, triple_count);
        m->macro_ce_quotient_actual_terms++;
    }

    uint32_t class_unique = 0;
    for (uint32_t i = 0; i < m->macro_ce_quotient_context_capacity; i++) {
        const uint32_t total = m->macro_ce_quotient_context_totals[i];
        if (total < min_support) continue;
        const uint64_t class_key = macro_ce_class_key(
            m->macro_ce_quotient_context_top1_token[i],
            m->macro_ce_quotient_context_top2_token[i],
            total,
            m->macro_ce_quotient_context_top1_count[i],
            m->macro_ce_quotient_signature_bins);
        m->macro_ce_quotient_context_class_keys[i] = class_key;
        if (macro_ce_count_add(
                &m->macro_ce_quotient_class_keys,
                &m->macro_ce_quotient_class_totals,
                &m->macro_ce_quotient_class_capacity,
                &class_unique,
                class_key, total, NULL) != 0) {
            macro_ce_quotient_clear(m);
            return -1;
        }
        m->macro_ce_quotient_unique_contexts++;
    }
    m->macro_ce_quotient_quotient_classes = class_unique;
    m->macro_ce_quotient_random_buckets = class_unique > 0u ? class_unique : 1u;

    uint32_t class_target_unique = 0;
    uint32_t random_unique = 0;
    uint32_t random_target_unique = 0;
    for (uint32_t i = 0; i < m->macro_ce_quotient_triple_capacity; i++) {
        const uint32_t count = m->macro_ce_quotient_triple_counts[i];
        if (count == 0u) continue;
        const uint64_t triple_key = m->macro_ce_quotient_triple_keys[i];
        const uint64_t ctx_key = macro_ce_triple_context(triple_key);
        const uint16_t target = macro_ce_triple_target(triple_key);
        const int ctx_slot_i = macro_ce_context_slot(m, ctx_key, 0);
        if (ctx_slot_i < 0) continue;
        const uint32_t ctx_slot = (uint32_t)ctx_slot_i;
        const uint64_t class_key = m->macro_ce_quotient_context_class_keys[ctx_slot];
        if (class_key == 0u) continue;
        if (macro_ce_count_add(
                &m->macro_ce_quotient_class_target_keys,
                &m->macro_ce_quotient_class_target_counts,
                &m->macro_ce_quotient_class_target_capacity,
                &class_target_unique,
                macro_ce_triple_key(class_key, target), count, NULL) != 0) {
            macro_ce_quotient_clear(m);
            return -1;
        }
        const uint64_t random_bucket =
            (uint64_t)(macro_ce_hash64(ctx_key ^ (uint64_t)m->macro_ce_quotient_seed) %
                       m->macro_ce_quotient_random_buckets) + 1ull;
        if (macro_ce_count_add(
                &m->macro_ce_quotient_random_keys,
                &m->macro_ce_quotient_random_totals,
                &m->macro_ce_quotient_random_capacity,
                &random_unique,
                random_bucket, count, NULL) != 0) {
            macro_ce_quotient_clear(m);
            return -1;
        }
        if (macro_ce_count_add(
                &m->macro_ce_quotient_random_target_keys,
                &m->macro_ce_quotient_random_target_counts,
                &m->macro_ce_quotient_random_target_capacity,
                &random_target_unique,
                macro_ce_triple_key(random_bucket, target), count, NULL) != 0) {
            macro_ce_quotient_clear(m);
            return -1;
        }
    }

    m->macro_ce_quotient_ratio =
        m->macro_ce_quotient_unique_contexts > 0u
            ? (float)((double)m->macro_ce_quotient_quotient_classes /
                      (double)m->macro_ce_quotient_unique_contexts)
            : 0.0f;

    printf("[R421_MACRO_CE_QUOTIENT] active=1 mode=order2-predictive-quotient observer=%d training=%d min_support=%u stride=%u terms=%llu build_fraction=%.6f contexts=%u classes=%u quotient_ratio=%.6f random_buckets=%u alpha=%.6f residual_price_scale=%.6f overhead=%.6f training_weight=%.6f\n",
           m->macro_ce_quotient_observer_enabled ? 1 : 0,
           m->macro_ce_quotient_training_enabled ? 1 : 0,
           m->macro_ce_quotient_min_support,
           m->macro_ce_quotient_stride,
           (unsigned long long)m->macro_ce_quotient_actual_terms,
           m->macro_ce_quotient_build_fraction,
           m->macro_ce_quotient_unique_contexts,
           m->macro_ce_quotient_quotient_classes,
           m->macro_ce_quotient_ratio,
           m->macro_ce_quotient_random_buckets,
           m->macro_ce_quotient_alpha,
           m->macro_ce_quotient_residual_price_scale,
           m->macro_ce_quotient_overhead,
           m->macro_ce_quotient_training_weight);
    return 0;
}

static double macro_ce_from_counts(uint32_t count, uint32_t total, float alpha) {
    const double a = alpha > 0.0f ? (double)alpha : 0.05;
    const double denom = (double)total + a * (double)V12_VOCAB_SIZE;
    const double p = ((double)count + a) / (denom > 0.0 ? denom : 1.0);
    return -log(p > 1.0e-12 ? p : 1.0e-12);
}

void macro_ce_quotient_observe(
    Model *m, const uint16_t *tokens, int batch_size, int seq_len) {
    if (!m || !m->macro_ce_quotient_observer_enabled || !tokens ||
        batch_size <= 0 || seq_len <= 2 ||
        m->macro_ce_quotient_quotient_classes == 0u) {
        return;
    }

    uint64_t positions = 0;
    uint64_t covered = 0;
    double context_ce_sum = 0.0;
    double quotient_ce_sum = 0.0;
    double random_ce_sum = 0.0;

    for (int b = 0; b < batch_size; b++) {
        const uint16_t *tok_b = tokens + (size_t)b * (size_t)seq_len;
        for (int t = 2; t < seq_len; t++) {
            const uint16_t left = tok_b[t - 2];
            const uint16_t right = tok_b[t - 1];
            const uint16_t target = tok_b[t];
            if (left >= V12_VOCAB_SIZE || right >= V12_VOCAB_SIZE || target >= V12_VOCAB_SIZE) {
                continue;
            }
            positions++;
            const uint64_t ctx_key = macro_ce_context_key(left, right);
            const int ctx_slot_i = macro_ce_context_slot(m, ctx_key, 0);
            if (ctx_slot_i < 0) continue;
            const uint32_t ctx_slot = (uint32_t)ctx_slot_i;
            const uint64_t class_key = m->macro_ce_quotient_context_class_keys[ctx_slot];
            const uint32_t ctx_total = m->macro_ce_quotient_context_totals[ctx_slot];
            if (class_key == 0u || ctx_total < m->macro_ce_quotient_min_support) continue;

            const uint64_t triple_key = macro_ce_triple_key(ctx_key, target);
            const uint32_t ctx_count = macro_ce_count_get(
                m->macro_ce_quotient_triple_keys,
                m->macro_ce_quotient_triple_counts,
                m->macro_ce_quotient_triple_capacity,
                triple_key);
            const uint32_t class_total = macro_ce_count_get(
                m->macro_ce_quotient_class_keys,
                m->macro_ce_quotient_class_totals,
                m->macro_ce_quotient_class_capacity,
                class_key);
            const uint32_t class_count = macro_ce_count_get(
                m->macro_ce_quotient_class_target_keys,
                m->macro_ce_quotient_class_target_counts,
                m->macro_ce_quotient_class_target_capacity,
                macro_ce_triple_key(class_key, target));
            const uint64_t random_bucket =
                (uint64_t)(macro_ce_hash64(ctx_key ^ (uint64_t)m->macro_ce_quotient_seed) %
                           m->macro_ce_quotient_random_buckets) + 1ull;
            const uint32_t random_total = macro_ce_count_get(
                m->macro_ce_quotient_random_keys,
                m->macro_ce_quotient_random_totals,
                m->macro_ce_quotient_random_capacity,
                random_bucket);
            const uint32_t random_count = macro_ce_count_get(
                m->macro_ce_quotient_random_target_keys,
                m->macro_ce_quotient_random_target_counts,
                m->macro_ce_quotient_random_target_capacity,
                macro_ce_triple_key(random_bucket, target));
            if (class_total == 0u || random_total == 0u) continue;

            covered++;
            context_ce_sum += macro_ce_from_counts(ctx_count, ctx_total, m->macro_ce_quotient_alpha);
            quotient_ce_sum += macro_ce_from_counts(class_count, class_total, m->macro_ce_quotient_alpha);
            random_ce_sum += macro_ce_from_counts(random_count, random_total, m->macro_ce_quotient_alpha);
        }
    }

    m->macro_ce_quotient_observer_batches++;
    m->macro_ce_quotient_observer_positions += positions;
    m->macro_ce_quotient_observer_covered += covered;
    m->macro_ce_quotient_context_ce_sum += context_ce_sum;
    m->macro_ce_quotient_quotient_ce_sum += quotient_ce_sum;
    m->macro_ce_quotient_random_ce_sum += random_ce_sum;

    const double total_positions = (double)m->macro_ce_quotient_observer_positions;
    const double total_covered = (double)m->macro_ce_quotient_observer_covered;
    const double context_ce = total_covered > 0.0
        ? m->macro_ce_quotient_context_ce_sum / total_covered
        : 0.0;
    const double quotient_ce = total_covered > 0.0
        ? m->macro_ce_quotient_quotient_ce_sum / total_covered
        : 0.0;
    const double random_ce = total_covered > 0.0
        ? m->macro_ce_quotient_random_ce_sum / total_covered
        : 0.0;
    const double coverage = total_positions > 0.0 ? total_covered / total_positions : 0.0;
    const double ce_excess = quotient_ce - context_ce;
    const double residual_price =
        (ce_excess > 0.0 ? ce_excess : 0.0) *
        (double)m->macro_ce_quotient_residual_price_scale;
    const double priced =
        (double)m->macro_ce_quotient_ratio +
        residual_price +
        (1.0 - coverage) +
        (double)m->macro_ce_quotient_overhead;

    m->macro_ce_quotient_coverage = (float)coverage;
    m->macro_ce_quotient_context_ce = (float)context_ce;
    m->macro_ce_quotient_quotient_ce = (float)quotient_ce;
    m->macro_ce_quotient_random_ce = (float)random_ce;
    m->macro_ce_quotient_ce_excess = (float)ce_excess;
    m->macro_ce_quotient_random_advantage = (float)(random_ce - quotient_ce);
    m->macro_ce_quotient_residual_price = (float)residual_price;
    m->macro_ce_quotient_priced_work_ratio = (float)priced;
    m->macro_ce_quotient_net_speedup_estimate =
        priced > 1.0e-12 ? (float)(1.0 / priced) : 0.0f;
}

static int macro_ce_quotient_row_is_covered(
    Model *m, const uint16_t *tok_b, int t) {
    const uint16_t left = tok_b[t - 1];
    const uint16_t right = tok_b[t];
    const uint16_t target = tok_b[t + 1];
    if (left >= V12_VOCAB_SIZE || right >= V12_VOCAB_SIZE || target >= V12_VOCAB_SIZE) {
        return 0;
    }
    const uint64_t ctx_key = macro_ce_context_key(left, right);
    const int ctx_slot_i = macro_ce_context_slot(m, ctx_key, 0);
    if (ctx_slot_i < 0) return 0;
    const uint32_t ctx_slot = (uint32_t)ctx_slot_i;
    const uint64_t class_key = m->macro_ce_quotient_context_class_keys[ctx_slot];
    const uint32_t ctx_total = m->macro_ce_quotient_context_totals[ctx_slot];
    if (class_key == 0u || ctx_total < m->macro_ce_quotient_min_support) return 0;
    const uint32_t class_total = macro_ce_count_get(
        m->macro_ce_quotient_class_keys,
        m->macro_ce_quotient_class_totals,
        m->macro_ce_quotient_class_capacity,
        class_key);
    const uint32_t class_count = macro_ce_count_get(
        m->macro_ce_quotient_class_target_keys,
        m->macro_ce_quotient_class_target_counts,
        m->macro_ce_quotient_class_target_capacity,
        macro_ce_triple_key(class_key, target));
    return class_total > 0u && class_count > 0u;
}

void macro_ce_quotient_apply_training_weights(
    Model *m, const uint16_t *tokens, float *d_logits,
    int batch_size, int seq_len, int n_samples) {
    if (!m || !m->macro_ce_quotient_training_enabled || !tokens || !d_logits ||
        batch_size <= 0 || seq_len <= 2 || n_samples <= 0 ||
        m->macro_ce_quotient_quotient_classes == 0u) {
        return;
    }

    const int n_valid = batch_size * (seq_len - 1);
    uint64_t covered = 0;
    const float quotient_weight = m->macro_ce_quotient_training_weight;

    for (int b = 0; b < batch_size; b++) {
        const uint16_t *tok_b = tokens + (size_t)b * (size_t)seq_len;
        for (int t = 1; t + 1 < seq_len; t++) {
            if (macro_ce_quotient_row_is_covered(m, tok_b, t)) {
                covered++;
            }
        }
    }

    if (covered == 0u) {
        m->macro_ce_quotient_training_batches++;
        m->macro_ce_quotient_training_rows += (uint64_t)n_valid;
        m->macro_ce_quotient_training_effective_weight = 1.0f;
        return;
    }

    const double denom = (double)(n_valid - (int)covered) +
                         (double)covered * (double)quotient_weight;
    const float renorm = denom > 1.0e-12 ? (float)((double)n_valid / denom) : 1.0f;

    for (int b = 0; b < batch_size; b++) {
        const uint16_t *tok_b = tokens + (size_t)b * (size_t)seq_len;
        for (int t = 0; t + 1 < seq_len; t++) {
            const int row_index = b * (seq_len - 1) + t;
            float row_weight = renorm;
            if (t >= 1 && macro_ce_quotient_row_is_covered(m, tok_b, t)) {
                row_weight *= quotient_weight;
            }
            if (row_weight != 1.0f) {
                float *row = d_logits + (size_t)row_index * (size_t)n_samples;
                for (int s = 0; s < n_samples; s++) row[s] *= row_weight;
            }
        }
    }

    m->macro_ce_quotient_training_batches++;
    m->macro_ce_quotient_training_rows += (uint64_t)n_valid;
    m->macro_ce_quotient_training_covered += covered;
    m->macro_ce_quotient_training_coverage =
        m->macro_ce_quotient_training_rows > 0u
            ? (float)((double)m->macro_ce_quotient_training_covered /
                      (double)m->macro_ce_quotient_training_rows)
            : 0.0f;
    m->macro_ce_quotient_training_effective_weight = quotient_weight * renorm;
}

static uint64_t ewms_mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static uint64_t ewms_hash_unit(const uint16_t *window, int position) {
    uint64_t h = 1469598103934665603ULL;
    h ^= (uint64_t)(uint32_t)position + 0x9e3779b97f4a7c15ULL;
    h *= 1099511628211ULL;
    const uint16_t reset = (position == 0 || window[position - 1] == (uint16_t)'\n') ? 1u : 0u;
    h ^= reset;
    h *= 1099511628211ULL;
    for (int i = 0; i <= position + 1; i++) {
        h ^= (uint64_t)window[i] + 0x10000ULL;
        h *= 1099511628211ULL;
    }
    return ewms_mix64(h);
}

static void exact_window_multiset_clear(Model *m) {
    if (!m) return;
    free(m->exact_window_multiset_hashes);
    free(m->exact_window_multiset_starts);
    free(m->exact_window_multiset_positions);
    free(m->exact_window_multiset_counts);
    m->exact_window_multiset_hashes = NULL;
    m->exact_window_multiset_starts = NULL;
    m->exact_window_multiset_positions = NULL;
    m->exact_window_multiset_counts = NULL;
    m->exact_window_multiset_tokens = NULL;
    m->exact_window_multiset_n_tokens = 0;
    m->exact_window_multiset_actual_units = 0;
    m->exact_window_multiset_capacity = 0;
    m->exact_window_multiset_unique_units = 0;
    m->exact_window_multiset_duplicate_units = 0;
    m->exact_window_multiset_collision_count = 0;
    m->exact_window_multiset_gate_batches = 0;
    m->exact_window_multiset_gate_positions = 0;
    m->exact_window_multiset_gate_supported = 0;
    m->exact_window_multiset_gate_saved_terms = 0;
    m->exact_window_multiset_gate_multiplicity_sum = 0;
    m->exact_window_multiset_gate_q_ratio = 0.0f;
    m->exact_window_multiset_gate_live_coverage = 0.0f;
    m->exact_window_multiset_gate_saved_fraction = 0.0f;
    m->exact_window_multiset_gate_mean_multiplicity = 0.0f;
    m->exact_window_multiset_gate_priced_ratio = 0.0f;
    m->exact_window_multiset_objective_audit = 0;
    m->exact_window_multiset_gradient_audit = 0;
}

void exact_window_multiset_configure(
    Model *m, int gate_enabled, uint32_t min_count,
    uint64_t requested_windows, uint32_t stride,
    float overhead, float noise_penalty) {
    if (!m) return;
    exact_window_multiset_clear(m);
    m->exact_window_multiset_gate_enabled = gate_enabled ? 1 : 0;
    m->exact_window_multiset_min_count = min_count > 0u ? min_count : 1u;
    m->exact_window_multiset_requested_windows = requested_windows;
    m->exact_window_multiset_stride = stride > 0u ? stride : 1u;
    m->exact_window_multiset_overhead = overhead >= 0.0f ? overhead : 0.0f;
    m->exact_window_multiset_noise_penalty = noise_penalty >= 0.0f ? noise_penalty : 0.0f;
}

static int ewms_units_equal_corpus(
    const Model *m, uint64_t a_start, uint16_t a_pos,
    uint64_t b_start, uint16_t b_pos, int seq_len) {
    if (!m || !m->exact_window_multiset_tokens || a_pos != b_pos) return 0;
    if (a_start + (uint64_t)a_pos + 1u >= (uint64_t)m->exact_window_multiset_n_tokens ||
        b_start + (uint64_t)b_pos + 1u >= (uint64_t)m->exact_window_multiset_n_tokens ||
        a_pos + 1 >= (uint16_t)seq_len) {
        return 0;
    }
    const uint16_t *a = m->exact_window_multiset_tokens + (size_t)a_start;
    const uint16_t *b = m->exact_window_multiset_tokens + (size_t)b_start;
    const int reset_a = (a_pos == 0u || a[a_pos - 1u] == (uint16_t)'\n') ? 1 : 0;
    const int reset_b = (b_pos == 0u || b[b_pos - 1u] == (uint16_t)'\n') ? 1 : 0;
    if (reset_a != reset_b) return 0;
    for (uint16_t i = 0; i <= a_pos + 1u; i++) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

static int ewms_units_equal_live(
    const Model *m, const uint16_t *window, int seq_len, int position,
    uint64_t stored_start, uint16_t stored_pos) {
    if (!m || !window || !m->exact_window_multiset_tokens ||
        position < 0 || position + 1 >= seq_len ||
        stored_pos != (uint16_t)position ||
        stored_start + (uint64_t)stored_pos + 1u >= (uint64_t)m->exact_window_multiset_n_tokens) {
        return 0;
    }
    const uint16_t *stored = m->exact_window_multiset_tokens + (size_t)stored_start;
    const int reset_live = (position == 0 || window[position - 1] == (uint16_t)'\n') ? 1 : 0;
    const int reset_stored = (stored_pos == 0u || stored[stored_pos - 1u] == (uint16_t)'\n') ? 1 : 0;
    if (reset_live != reset_stored) return 0;
    for (int i = 0; i <= position + 1; i++) {
        if (window[i] != stored[i]) return 0;
    }
    return 1;
}

static int exact_window_multiset_rehash(Model *m, uint32_t new_capacity, int seq_len) {
    uint64_t *old_hashes = m->exact_window_multiset_hashes;
    uint64_t *old_starts = m->exact_window_multiset_starts;
    uint16_t *old_positions = m->exact_window_multiset_positions;
    uint32_t *old_counts = m->exact_window_multiset_counts;
    uint32_t old_capacity = m->exact_window_multiset_capacity;

    uint64_t *new_hashes = (uint64_t *)calloc((size_t)new_capacity, sizeof(uint64_t));
    uint64_t *new_starts = (uint64_t *)calloc((size_t)new_capacity, sizeof(uint64_t));
    uint16_t *new_positions = (uint16_t *)calloc((size_t)new_capacity, sizeof(uint16_t));
    uint32_t *new_counts = (uint32_t *)calloc((size_t)new_capacity, sizeof(uint32_t));
    if (!new_hashes || !new_starts || !new_positions || !new_counts) {
        free(new_hashes);
        free(new_starts);
        free(new_positions);
        free(new_counts);
        return -1;
    }

    m->exact_window_multiset_hashes = new_hashes;
    m->exact_window_multiset_starts = new_starts;
    m->exact_window_multiset_positions = new_positions;
    m->exact_window_multiset_counts = new_counts;
    m->exact_window_multiset_capacity = new_capacity;

    if (old_hashes && old_counts) {
        const uint32_t mask = new_capacity - 1u;
        for (uint32_t i = 0; i < old_capacity; i++) {
            const uint32_t count = old_counts[i];
            if (count == 0u) continue;
            uint32_t slot = (uint32_t)ewms_mix64(old_hashes[i]) & mask;
            while (new_counts[slot] != 0u) {
                if (new_hashes[slot] == old_hashes[i] &&
                    ewms_units_equal_corpus(m, new_starts[slot], new_positions[slot],
                                            old_starts[i], old_positions[i], seq_len)) {
                    new_counts[slot] += count;
                    goto next_old;
                }
                slot = (slot + 1u) & mask;
            }
            new_hashes[slot] = old_hashes[i];
            new_starts[slot] = old_starts[i];
            new_positions[slot] = old_positions[i];
            new_counts[slot] = count;
next_old:
            ;
        }
    }

    free(old_hashes);
    free(old_starts);
    free(old_positions);
    free(old_counts);
    return 0;
}

static int exact_window_multiset_add(
    Model *m, uint64_t start, uint16_t position, uint64_t hash, int seq_len) {
    if (m->exact_window_multiset_capacity == 0u) {
        if (exact_window_multiset_rehash(m, 65536u, seq_len) != 0) return -1;
    }
    if ((uint64_t)(m->exact_window_multiset_unique_units + 1u) * 2u >=
        (uint64_t)m->exact_window_multiset_capacity) {
        if (exact_window_multiset_rehash(
                m, m->exact_window_multiset_capacity << 1, seq_len) != 0) {
            return -1;
        }
    }

    const uint32_t mask = m->exact_window_multiset_capacity - 1u;
    uint32_t slot = (uint32_t)ewms_mix64(hash) & mask;
    while (m->exact_window_multiset_counts[slot] != 0u) {
        if (m->exact_window_multiset_hashes[slot] == hash) {
            if (ewms_units_equal_corpus(
                    m, m->exact_window_multiset_starts[slot],
                    m->exact_window_multiset_positions[slot],
                    start, position, seq_len)) {
                if (m->exact_window_multiset_counts[slot] != UINT32_MAX) {
                    m->exact_window_multiset_counts[slot]++;
                }
                return 0;
            }
            m->exact_window_multiset_collision_count++;
        }
        slot = (slot + 1u) & mask;
    }

    m->exact_window_multiset_hashes[slot] = hash;
    m->exact_window_multiset_starts[slot] = start;
    m->exact_window_multiset_positions[slot] = position;
    m->exact_window_multiset_counts[slot] = 1u;
    m->exact_window_multiset_unique_units++;
    return 0;
}

int exact_window_multiset_build(
    Model *m, const uint16_t *tokens, size_t n_toks, int seq_len) {
    if (!m || !m->exact_window_multiset_gate_enabled) return 0;
    if (!tokens || seq_len < 2 || n_toks < (size_t)seq_len) return -1;

    const uint32_t min_count = m->exact_window_multiset_min_count
        ? m->exact_window_multiset_min_count
        : 1u;
    const uint32_t stride = m->exact_window_multiset_stride
        ? m->exact_window_multiset_stride
        : 1u;
    const uint64_t requested = m->exact_window_multiset_requested_windows;
    const uint64_t max_windows = requested ? requested : UINT64_MAX;

    exact_window_multiset_clear(m);
    m->exact_window_multiset_gate_enabled = 1;
    m->exact_window_multiset_min_count = min_count;
    m->exact_window_multiset_stride = stride;
    m->exact_window_multiset_requested_windows = requested;
    m->exact_window_multiset_tokens = tokens;
    m->exact_window_multiset_n_tokens = n_toks;

    uint64_t windows = 0;
    const size_t step = (size_t)stride;
    for (size_t start = 0;
         start + (size_t)seq_len <= n_toks && windows < max_windows;
         start += step) {
        const uint16_t *window = tokens + start;
        for (int pos = 0; pos + 1 < seq_len; pos++) {
            const uint64_t hash = ewms_hash_unit(window, pos);
            if (exact_window_multiset_add(m, (uint64_t)start, (uint16_t)pos, hash, seq_len) != 0) {
                exact_window_multiset_clear(m);
                return -1;
            }
            m->exact_window_multiset_actual_units++;
        }
        windows++;
        if (step == 0u) break;
    }

    uint64_t weighted_total = 0;
    uint64_t supported_mass = 0;
    for (uint32_t i = 0; i < m->exact_window_multiset_capacity; i++) {
        const uint32_t count = m->exact_window_multiset_counts[i];
        if (count == 0u) continue;
        weighted_total += (uint64_t)count;
        if (count >= min_count) supported_mass += (uint64_t)count;
    }
    m->exact_window_multiset_duplicate_units =
        m->exact_window_multiset_actual_units >= (uint64_t)m->exact_window_multiset_unique_units
            ? m->exact_window_multiset_actual_units - (uint64_t)m->exact_window_multiset_unique_units
            : 0u;
    m->exact_window_multiset_gate_q_ratio =
        m->exact_window_multiset_actual_units > 0
            ? (float)((double)m->exact_window_multiset_unique_units /
                      (double)m->exact_window_multiset_actual_units)
            : 1.0f;
    m->exact_window_multiset_gate_priced_ratio =
        m->exact_window_multiset_overhead +
        m->exact_window_multiset_gate_q_ratio *
            (1.0f + m->exact_window_multiset_noise_penalty);
    m->exact_window_multiset_objective_audit =
        (weighted_total == m->exact_window_multiset_actual_units) ? 1 : 0;
    m->exact_window_multiset_gradient_audit =
        m->exact_window_multiset_objective_audit;

    printf("[EXACT_WINDOW_MULTISET] active=1 mode=exact-window-sufficient-statistic min_count=%u stride=%u windows=%llu units=%llu unique_units=%u duplicate_units=%llu supported_mass=%llu q=%.6f overhead=%.6f noise=%.6f priced_ratio=%.6f collisions=%llu objective_audit=%d gradient_audit=%d table_capacity=%u\n",
           min_count,
           stride,
           (unsigned long long)windows,
           (unsigned long long)m->exact_window_multiset_actual_units,
           m->exact_window_multiset_unique_units,
           (unsigned long long)m->exact_window_multiset_duplicate_units,
           (unsigned long long)supported_mass,
           m->exact_window_multiset_gate_q_ratio,
           m->exact_window_multiset_overhead,
           m->exact_window_multiset_noise_penalty,
           m->exact_window_multiset_gate_priced_ratio,
           (unsigned long long)m->exact_window_multiset_collision_count,
           m->exact_window_multiset_objective_audit,
           m->exact_window_multiset_gradient_audit,
           m->exact_window_multiset_capacity);
    return 0;
}

uint32_t exact_window_multiset_count_live(
    const Model *m, const uint16_t *window, int seq_len, int position) {
    if (!m || !m->exact_window_multiset_gate_enabled || !window ||
        !m->exact_window_multiset_hashes || !m->exact_window_multiset_counts ||
        m->exact_window_multiset_capacity == 0u ||
        position < 0 || position + 1 >= seq_len) {
        return 0u;
    }

    const uint64_t hash = ewms_hash_unit(window, position);
    const uint32_t mask = m->exact_window_multiset_capacity - 1u;
    uint32_t slot = (uint32_t)ewms_mix64(hash) & mask;
    for (uint32_t probe = 0; probe < m->exact_window_multiset_capacity; probe++) {
        const uint32_t count = m->exact_window_multiset_counts[slot];
        if (count == 0u) return 0u;
        if (m->exact_window_multiset_hashes[slot] == hash &&
            ewms_units_equal_live(
                m, window, seq_len, position,
                m->exact_window_multiset_starts[slot],
                m->exact_window_multiset_positions[slot])) {
            return count;
        }
        slot = (slot + 1u) & mask;
    }
    return 0u;
}

static void prefix_frontier_clear(Model *m) {
    if (!m) return;
    free(m->prefix_frontier_keys);
    free(m->prefix_frontier_counts);
    m->prefix_frontier_keys = NULL;
    m->prefix_frontier_counts = NULL;
    m->prefix_frontier_actual_terms = 0;
    m->prefix_frontier_capacity = 0;
    m->prefix_frontier_unique_terms = 0;
    m->prefix_frontier_supported_mass = 0;
    m->prefix_frontier_gate_batches = 0;
    m->prefix_frontier_gate_positions = 0;
    m->prefix_frontier_gate_supported = 0;
    m->prefix_frontier_gate_saved_terms = 0;
    m->prefix_frontier_gate_multiplicity_sum = 0;
    m->prefix_frontier_gate_total_energy = 0.0;
    m->prefix_frontier_gate_supported_energy = 0.0;
    m->prefix_frontier_gate_reuse_fraction = 0.0f;
    m->prefix_frontier_gate_energy_fraction = 0.0f;
    m->prefix_frontier_gate_mean_multiplicity = 0.0f;
    m->prefix_frontier_gate_priced_ratio = 0.0f;
}

void prefix_frontier_configure(
    Model *m, int gate_enabled, uint32_t min_count,
    uint64_t requested_terms, uint32_t stride,
    float overhead, float repair_cost) {
    if (!m) return;
    prefix_frontier_clear(m);
    m->prefix_frontier_gate_enabled = gate_enabled ? 1 : 0;
    m->prefix_frontier_min_count = min_count > 0u ? min_count : 2u;
    m->prefix_frontier_requested_terms = requested_terms;
    m->prefix_frontier_stride = stride > 0u ? stride : 1u;
    m->prefix_frontier_overhead = overhead >= 0.0f ? overhead : 0.0f;
    m->prefix_frontier_repair_cost = repair_cost >= 0.0f ? repair_cost : 0.0f;
}

static int prefix_frontier_rehash(Model *m, uint32_t new_capacity) {
    uint32_t *old_keys = m->prefix_frontier_keys;
    uint32_t *old_counts = m->prefix_frontier_counts;
    uint32_t old_capacity = m->prefix_frontier_capacity;

    uint32_t *new_keys = (uint32_t *)calloc((size_t)new_capacity, sizeof(uint32_t));
    uint32_t *new_counts = (uint32_t *)calloc((size_t)new_capacity, sizeof(uint32_t));
    if (!new_keys || !new_counts) {
        free(new_keys);
        free(new_counts);
        return -1;
    }

    m->prefix_frontier_keys = new_keys;
    m->prefix_frontier_counts = new_counts;
    m->prefix_frontier_capacity = new_capacity;

    if (old_keys && old_counts) {
        const uint32_t mask = new_capacity - 1u;
        for (uint32_t i = 0; i < old_capacity; i++) {
            const uint32_t count = old_counts[i];
            if (count == 0u) continue;
            const uint32_t key = old_keys[i];
            uint32_t slot = corpus_logit_prior_mix32(key) & mask;
            while (new_counts[slot] != 0u) slot = (slot + 1u) & mask;
            new_keys[slot] = key;
            new_counts[slot] = count;
        }
    }

    free(old_keys);
    free(old_counts);
    return 0;
}

static int prefix_frontier_add(Model *m, uint32_t key) {
    if (m->prefix_frontier_capacity == 0u) {
        if (prefix_frontier_rehash(m, 65536u) != 0) return -1;
    }
    if ((uint64_t)(m->prefix_frontier_unique_terms + 1u) * 2u >=
        (uint64_t)m->prefix_frontier_capacity) {
        if (prefix_frontier_rehash(m, m->prefix_frontier_capacity << 1) != 0) {
            return -1;
        }
    }

    const uint32_t mask = m->prefix_frontier_capacity - 1u;
    uint32_t slot = corpus_logit_prior_mix32(key) & mask;
    while (m->prefix_frontier_counts[slot] != 0u) {
        if (m->prefix_frontier_keys[slot] == key) {
            m->prefix_frontier_counts[slot]++;
            return 0;
        }
        slot = (slot + 1u) & mask;
    }

    m->prefix_frontier_keys[slot] = key;
    m->prefix_frontier_counts[slot] = 1u;
    m->prefix_frontier_unique_terms++;
    return 0;
}

int prefix_frontier_build(Model *m, const uint16_t *tokens, size_t n_toks) {
    if (!m || !m->prefix_frontier_gate_enabled) return 0;
    if (!tokens || n_toks < 3) return -1;

    prefix_frontier_clear(m);
    const uint32_t stride = m->prefix_frontier_stride ? m->prefix_frontier_stride : 1u;
    const uint64_t requested = m->prefix_frontier_requested_terms;
    const uint64_t max_terms = requested ? requested : UINT64_MAX;

    for (size_t i = 1; i + 1 < n_toks && m->prefix_frontier_actual_terms < max_terms; i += stride) {
        const uint16_t prev = tokens[i - 1];
        const uint16_t ctx = tokens[i];
        if (prev >= V12_VOCAB_SIZE || ctx >= V12_VOCAB_SIZE) continue;
        if (prefix_frontier_add(m, state_transition_atlas_key(prev, ctx)) != 0) {
            prefix_frontier_clear(m);
            return -1;
        }
        m->prefix_frontier_actual_terms++;
    }

    const uint32_t min_count = m->prefix_frontier_min_count
        ? m->prefix_frontier_min_count
        : 2u;
    for (uint32_t i = 0; i < m->prefix_frontier_capacity; i++) {
        const uint32_t count = m->prefix_frontier_counts[i];
        if (count >= min_count) {
            m->prefix_frontier_supported_mass += (uint64_t)count;
        }
    }

    const double reuse_fraction = m->prefix_frontier_actual_terms > 0
        ? 1.0 - ((double)m->prefix_frontier_unique_terms /
                 (double)m->prefix_frontier_actual_terms)
        : 0.0;
    printf("[PREFIX_FRONTIER] active=1 mode=exact-local-frontier-multiplicity min_count=%u stride=%u terms=%llu unique_terms=%u supported_mass=%llu corpus_reuse_fraction=%.6f table_capacity=%u\n",
           m->prefix_frontier_min_count,
           m->prefix_frontier_stride,
           (unsigned long long)m->prefix_frontier_actual_terms,
           m->prefix_frontier_unique_terms,
           (unsigned long long)m->prefix_frontier_supported_mass,
           reuse_fraction,
           m->prefix_frontier_capacity);
    return 0;
}

uint32_t prefix_frontier_count(const Model *m, uint16_t prev, uint16_t context) {
    if (!m || !m->prefix_frontier_gate_enabled ||
        !m->prefix_frontier_keys || !m->prefix_frontier_counts ||
        m->prefix_frontier_capacity == 0u) {
        return 0u;
    }
    if (prev >= V12_VOCAB_SIZE || context >= V12_VOCAB_SIZE) return 0u;

    const uint32_t key = state_transition_atlas_key(prev, context);
    const uint32_t mask = m->prefix_frontier_capacity - 1u;
    uint32_t slot = corpus_logit_prior_mix32(key) & mask;
    for (uint32_t probe = 0; probe < m->prefix_frontier_capacity; probe++) {
        const uint32_t count = m->prefix_frontier_counts[slot];
        if (count == 0u) return 0u;
        if (m->prefix_frontier_keys[slot] == key) return count;
        slot = (slot + 1u) & mask;
    }
    return 0u;
}

static void macro_transcript_clear(Model *m) {
    if (!m) return;
    free(m->macro_transcript_keys);
    free(m->macro_transcript_counts);
    m->macro_transcript_keys = NULL;
    m->macro_transcript_counts = NULL;
    m->macro_transcript_actual_terms = 0;
    m->macro_transcript_capacity = 0;
    m->macro_transcript_unique_pairs = 0;
    m->macro_transcript_supported_mass = 0;
    m->macro_transcript_gate_batches = 0;
    m->macro_transcript_gate_positions = 0;
    m->macro_transcript_gate_ops = 0;
    m->macro_transcript_gate_pair_macros = 0;
    m->macro_transcript_gate_singletons = 0;
    m->macro_transcript_gate_pair_fraction = 0.0f;
    m->macro_transcript_gate_length_ratio = 0.0f;
    m->macro_transcript_gate_priced_ratio = 0.0f;
}

void macro_transcript_configure(
    Model *m, int gate_enabled, uint32_t min_count,
    uint64_t requested_terms, uint32_t stride,
    float hardness, float codec_cost, float decode_cost) {
    if (!m) return;
    macro_transcript_clear(m);
    m->macro_transcript_gate_enabled = gate_enabled ? 1 : 0;
    m->macro_transcript_min_count = (min_count > 0u) ? min_count : 2u;
    m->macro_transcript_requested_terms = requested_terms;
    m->macro_transcript_stride = (stride > 0u) ? stride : 1u;
    m->macro_transcript_hardness = (hardness > 0.0f) ? hardness : 1.0f;
    m->macro_transcript_codec_cost = (codec_cost >= 0.0f) ? codec_cost : 0.0f;
    m->macro_transcript_decode_cost = (decode_cost >= 0.0f) ? decode_cost : 0.0f;
}

static int macro_transcript_rehash(Model *m, uint32_t new_capacity) {
    uint32_t *old_keys = m->macro_transcript_keys;
    uint32_t *old_counts = m->macro_transcript_counts;
    uint32_t old_capacity = m->macro_transcript_capacity;

    uint32_t *new_keys = (uint32_t *)calloc((size_t)new_capacity, sizeof(uint32_t));
    uint32_t *new_counts = (uint32_t *)calloc((size_t)new_capacity, sizeof(uint32_t));
    if (!new_keys || !new_counts) {
        free(new_keys);
        free(new_counts);
        return -1;
    }

    m->macro_transcript_keys = new_keys;
    m->macro_transcript_counts = new_counts;
    m->macro_transcript_capacity = new_capacity;

    if (old_keys && old_counts) {
        const uint32_t mask = new_capacity - 1u;
        for (uint32_t i = 0; i < old_capacity; i++) {
            const uint32_t count = old_counts[i];
            if (count == 0u) continue;
            const uint32_t key = old_keys[i];
            uint32_t slot = corpus_logit_prior_mix32(key) & mask;
            while (new_counts[slot] != 0u) slot = (slot + 1u) & mask;
            new_keys[slot] = key;
            new_counts[slot] = count;
        }
    }

    free(old_keys);
    free(old_counts);
    return 0;
}

static int macro_transcript_add(Model *m, uint32_t key) {
    if (m->macro_transcript_capacity == 0u) {
        if (macro_transcript_rehash(m, 65536u) != 0) return -1;
    }
    if ((uint64_t)(m->macro_transcript_unique_pairs + 1u) * 2u >=
        (uint64_t)m->macro_transcript_capacity) {
        if (macro_transcript_rehash(m, m->macro_transcript_capacity << 1) != 0) {
            return -1;
        }
    }

    const uint32_t mask = m->macro_transcript_capacity - 1u;
    uint32_t slot = corpus_logit_prior_mix32(key) & mask;
    while (m->macro_transcript_counts[slot] != 0u) {
        if (m->macro_transcript_keys[slot] == key) {
            m->macro_transcript_counts[slot]++;
            return 0;
        }
        slot = (slot + 1u) & mask;
    }

    m->macro_transcript_keys[slot] = key;
    m->macro_transcript_counts[slot] = 1u;
    m->macro_transcript_unique_pairs++;
    return 0;
}

int macro_transcript_build(Model *m, const uint16_t *tokens, size_t n_toks) {
    if (!m || !m->macro_transcript_gate_enabled) return 0;
    if (!tokens || n_toks < 2) return -1;

    macro_transcript_clear(m);
    const uint32_t stride = m->macro_transcript_stride ? m->macro_transcript_stride : 1u;
    const uint64_t requested = m->macro_transcript_requested_terms;
    const uint64_t max_terms = requested ? requested : UINT64_MAX;

    for (size_t i = 0; i + 1 < n_toks && m->macro_transcript_actual_terms < max_terms; i += stride) {
        const uint16_t left = tokens[i];
        const uint16_t right = tokens[i + 1];
        if (left >= V12_VOCAB_SIZE || right >= V12_VOCAB_SIZE) continue;
        if (macro_transcript_add(m, state_transition_atlas_key(left, right)) != 0) {
            macro_transcript_clear(m);
            return -1;
        }
        m->macro_transcript_actual_terms++;
    }

    const uint32_t min_count = m->macro_transcript_min_count
        ? m->macro_transcript_min_count
        : 2u;
    uint32_t supported_pairs = 0;
    for (uint32_t i = 0; i < m->macro_transcript_capacity; i++) {
        const uint32_t count = m->macro_transcript_counts[i];
        if (count >= min_count) {
            m->macro_transcript_supported_mass += (uint64_t)count;
            supported_pairs++;
        }
    }

    const double supported_fraction = m->macro_transcript_actual_terms > 0
        ? (double)m->macro_transcript_supported_mass /
          (double)m->macro_transcript_actual_terms
        : 0.0;
    printf("[MACRO_TRANSCRIPT] active=1 mode=exact-reversible-bigram-codec min_count=%u stride=%u terms=%llu unique_pairs=%u supported_pairs=%u supported_mass=%llu supported_fraction=%.6f table_capacity=%u\n",
           m->macro_transcript_min_count,
           m->macro_transcript_stride,
           (unsigned long long)m->macro_transcript_actual_terms,
           m->macro_transcript_unique_pairs,
           supported_pairs,
           (unsigned long long)m->macro_transcript_supported_mass,
           supported_fraction,
           m->macro_transcript_capacity);
    return 0;
}

uint32_t macro_transcript_pair_count(const Model *m, uint16_t left, uint16_t right) {
    if (!m || !m->macro_transcript_gate_enabled ||
        !m->macro_transcript_keys || !m->macro_transcript_counts ||
        m->macro_transcript_capacity == 0u) {
        return 0u;
    }
    if (left >= V12_VOCAB_SIZE || right >= V12_VOCAB_SIZE) return 0u;

    const uint32_t key = state_transition_atlas_key(left, right);
    const uint32_t mask = m->macro_transcript_capacity - 1u;
    uint32_t slot = corpus_logit_prior_mix32(key) & mask;
    for (uint32_t probe = 0; probe < m->macro_transcript_capacity; probe++) {
        const uint32_t count = m->macro_transcript_counts[slot];
        if (count == 0u) return 0u;
        if (m->macro_transcript_keys[slot] == key) return count;
        slot = (slot + 1u) & mask;
    }
    return 0u;
}

int reversible_deterministic_expansion_token(uint16_t token) {
    switch (token) {
        case (uint16_t)' ':
        case (uint16_t)'\n':
        case (uint16_t)'\t':
        case (uint16_t)'\r':
        case (uint16_t)'-':
        case (uint16_t)'=':
        case (uint16_t)'_':
        case (uint16_t)'*':
        case (uint16_t)'#':
        case (uint16_t)'/':
        case (uint16_t)'\\':
        case (uint16_t)'|':
        case (uint16_t)'.':
        case (uint16_t)',':
        case (uint16_t)':':
        case (uint16_t)';':
        case (uint16_t)')':
        case (uint16_t)']':
        case (uint16_t)'}':
        case (uint16_t)'>':
            return 1;
        default:
            return 0;
    }
}

static void reversible_deterministic_expansion_clear(Model *m) {
    if (!m) return;
    m->reversible_deterministic_expansion_actual_terms = 0;
    m->reversible_deterministic_expansion_corpus_tokens = 0;
    m->reversible_deterministic_expansion_corpus_covered = 0;
    m->reversible_deterministic_expansion_corpus_spans = 0;
    m->reversible_deterministic_expansion_inverse_failures = 0;
    m->reversible_deterministic_expansion_gate_batches = 0;
    m->reversible_deterministic_expansion_gate_positions = 0;
    m->reversible_deterministic_expansion_gate_covered = 0;
    m->reversible_deterministic_expansion_gate_spans = 0;
    m->reversible_deterministic_expansion_gate_coord_units = 0;
    m->reversible_deterministic_expansion_gate_lambda = 0.0f;
    m->reversible_deterministic_expansion_gate_eta = 0.0f;
    m->reversible_deterministic_expansion_gate_priced_ratio = 0.0f;
    m->reversible_deterministic_expansion_inverse_audit = 0;
}

void reversible_deterministic_expansion_configure(
    Model *m, int gate_enabled, uint32_t min_run,
    uint64_t requested_terms, float overhead, float noise_penalty) {
    if (!m) return;
    reversible_deterministic_expansion_clear(m);
    m->reversible_deterministic_expansion_gate_enabled = gate_enabled ? 1 : 0;
    m->reversible_deterministic_expansion_min_run = min_run > 1u ? min_run : 3u;
    m->reversible_deterministic_expansion_requested_terms = requested_terms;
    m->reversible_deterministic_expansion_overhead = overhead >= 0.0f ? overhead : 0.0f;
    m->reversible_deterministic_expansion_noise_penalty =
        noise_penalty >= 0.0f ? noise_penalty : 0.0f;
}

int reversible_deterministic_expansion_build(
    Model *m, const uint16_t *tokens, size_t n_toks) {
    if (!m || !m->reversible_deterministic_expansion_gate_enabled) return 0;
    if (!tokens || n_toks == 0) return -1;

    const uint32_t min_run = m->reversible_deterministic_expansion_min_run > 1u
        ? m->reversible_deterministic_expansion_min_run
        : 3u;
    const uint64_t requested = m->reversible_deterministic_expansion_requested_terms;
    const size_t scan_end = requested && requested < (uint64_t)n_toks
        ? (size_t)requested
        : n_toks;

    reversible_deterministic_expansion_clear(m);
    m->reversible_deterministic_expansion_gate_enabled = 1;
    m->reversible_deterministic_expansion_min_run = min_run;
    m->reversible_deterministic_expansion_requested_terms = requested;

    size_t i = 0;
    while (i < scan_end) {
        const uint16_t tok = tokens[i];
        size_t run = 1;
        while (i + run < scan_end && tokens[i + run] == tok) run++;
        m->reversible_deterministic_expansion_actual_terms += (uint64_t)run;
        m->reversible_deterministic_expansion_corpus_tokens += (uint64_t)run;
        if (run >= (size_t)min_run && reversible_deterministic_expansion_token(tok)) {
            int inverse_ok = 1;
            for (size_t j = 0; j < run; j++) {
                if (tokens[i + j] != tok) {
                    inverse_ok = 0;
                    break;
                }
            }
            if (inverse_ok) {
                m->reversible_deterministic_expansion_corpus_covered += (uint64_t)(run - 1u);
                m->reversible_deterministic_expansion_corpus_spans++;
            } else {
                m->reversible_deterministic_expansion_inverse_failures++;
            }
        }
        i += run;
    }

    const double positions = (double)m->reversible_deterministic_expansion_corpus_tokens;
    const double lambda = positions > 0.0
        ? (double)m->reversible_deterministic_expansion_corpus_covered / positions
        : 0.0;
    const double eta = positions > 0.0
        ? (double)m->reversible_deterministic_expansion_corpus_spans / positions
        : 0.0;
    const double priced = (double)m->reversible_deterministic_expansion_overhead +
        (1.0 - lambda + eta) *
            (1.0 + (double)m->reversible_deterministic_expansion_noise_penalty);
    m->reversible_deterministic_expansion_gate_lambda = (float)lambda;
    m->reversible_deterministic_expansion_gate_eta = (float)eta;
    m->reversible_deterministic_expansion_gate_priced_ratio = (float)priced;
    m->reversible_deterministic_expansion_inverse_audit =
        (m->reversible_deterministic_expansion_inverse_failures == 0u) ? 1 : 0;

    printf("[REVERSIBLE_DETERMINISTIC_EXPANSION] active=1 mode=fixed-repeat-run-transducer min_run=%u terms=%llu tokens=%llu covered=%llu spans=%llu lambda=%.6f eta=%.6f overhead=%.6f noise=%.6f priced_ratio=%.6f inverse_failures=%llu inverse_audit=%d\n",
           min_run,
           (unsigned long long)m->reversible_deterministic_expansion_actual_terms,
           (unsigned long long)m->reversible_deterministic_expansion_corpus_tokens,
           (unsigned long long)m->reversible_deterministic_expansion_corpus_covered,
           (unsigned long long)m->reversible_deterministic_expansion_corpus_spans,
           m->reversible_deterministic_expansion_gate_lambda,
           m->reversible_deterministic_expansion_gate_eta,
           m->reversible_deterministic_expansion_overhead,
           m->reversible_deterministic_expansion_noise_penalty,
           m->reversible_deterministic_expansion_gate_priced_ratio,
           (unsigned long long)m->reversible_deterministic_expansion_inverse_failures,
           m->reversible_deterministic_expansion_inverse_audit);
    return 0;
}

void model_free(Model *m) {
    if (!m) return;

    free(m->embed_int2);
    free(m->embed_E);
    free(m->embed_P);
    free(m->d_embed_E);
    free(m->d_embed_P);
    free(m->sa_embed_E);
    free(m->m_embed_E_row);
    free(m->m_embed_E_col);
    free(m->v_embed_E_row);
    free(m->v_embed_E_col);
    free(m->sa_embed_P);
    if (m->csr_embed) {
        ternary_csr_free(m->csr_embed);
        free(m->csr_embed);
    }

    free(m->lm_head_int2);
    ternary_csr_free(&m->csr_lm_head);
    free(m->shadow_lm_head);
    free(m->sampled_lm_head);
    free(m->dlm_head_g_row);
    free(m->dlm_head_g_col);
    free(m->dlm_head_gq_row);
    free(m->dlm_head_gq_col);
    free(m->dlm_head_cnt_row);
    free(m->dlm_head_cnt_col);
    free(m->hot_mask);
    free(m->vocab_to_hot);
    free(m->hot_to_vocab);

    free(m->norm_w);
    free(m->norm_b);
    free(m->d_norm_w);
    free(m->d_norm_b);

    free(m->w_pred_int2);
    ternary_csr_free(&m->csr_w_pred);
    free(m->shadow_w_pred);
    free(m->sampled_w_pred);
    free(m->dw_pred);
    free(m->crystal_mask_pred);

    free(m->m_lm_head_row);
    free(m->m_lm_head_col);
    free(m->v_lm_head_row);
    free(m->v_lm_head_col);
    free(m->m_w_pred_row);
    free(m->m_w_pred_col);
    free(m->v_w_pred_row);
    free(m->v_w_pred_col);
    free(m->m_norm_w);
    free(m->v_norm_w);
    free(m->m_norm_b);
    free(m->v_norm_b);

    for (int k = 0; k < V12_N_LAYERS; k++) {
        SSMLayer *L = &m->layers[k];

        free(L->w_in_int2);
        free(L->w_out_int2);
        free(L->w_conv_int2);
        ternary_csr_free(&L->csr_in);
        ternary_csr_free(&L->csr_out);

        free(L->A);
        free(L->B);
        free(L->C);
        free(L->sigma);
        free(L->d_sigma);
        free(L->m_sigma);
        free(L->v_sigma);

        free(L->mtp_U);
        free(L->mtp_V);
        free(L->d_mtp_U);
        free(L->d_mtp_V);

        free(L->shadow_w_in);
        free(L->shadow_w_out);
        free(L->shadow_w_conv);

        free(L->W_A);
        free(L->b_A);
        free(L->shadow_w_A);

        free(L->sampled_w_in);
        free(L->sampled_w_out);

        free(L->crystal_mask_in);
        free(L->crystal_mask_out);

        free(L->grad_accum_in);
        free(L->grad_accum_out);

        free(L->dw_in);
        free(L->dw_out);
        free(L->dA);
        free(L->dB);
        free(L->dB_seq);
        if (L->dC) { free(L->dC); L->dC = NULL; }
        free(L->d_conv_w);
        free(L->dW_A);
        free(L->db_A);

        free(L->m_w_in_row);
        free(L->m_w_in_col);
        free(L->v_w_in_row);
        free(L->v_w_in_col);
        free(L->m_w_out_row);
        free(L->m_w_out_col);
        free(L->v_w_out_row);
        free(L->v_w_out_col);
        free(L->m_A);
        free(L->v_A);
        free(L->m_B);
        free(L->v_B);
        if (L->m_C) { free(L->m_C); L->m_C = NULL; }
        if (L->v_C) { free(L->v_C); L->v_C = NULL; }
        free(L->m_conv);
        free(L->v_conv);
        free(L->m_W_A);
        free(L->v_W_A);
        free(L->m_b_A);
        free(L->v_b_A);
        free(L->m_mtp_U);
        free(L->v_mtp_U);
        free(L->m_mtp_V);
        free(L->v_mtp_V);

        free(L->cospec.U);
        free(L->cospec.cov_accum);

        free(L->C_bilinear);
        free(L->shadow_C_bilinear);
        free(L->sampled_C_bilinear);
        free(L->dC_bilinear);
        free(L->m_C_bilinear);
        free(L->v_C_bilinear);

        free(L->u);
        free(L->shadow_u);
        free(L->sampled_u);
        free(L->du);
        free(L->m_u);
        free(L->v_u);

        free(L->v);
        free(L->shadow_v);
        free(L->sampled_v);
        free(L->dv);
        free(L->m_v);
        free(L->v_v);

        free(L->alive_idx_in);
        free(L->alive_idx_out);
    }

    free(m->unigram_cdf);
    free(m->log_unigram);
    free(m->corpus_logit_prior_ctx_counts);
    free(m->corpus_logit_prior_pair_keys);
    free(m->corpus_logit_prior_pair_counts);
    free(m->corpus_logit_prior_unigram);
    free(m->program_kernel_targets);
    free(m->program_kernel_counts);
    free(m->certified_artifact_boundary_counts);
    free(m->certified_artifact_boundary_false_counts);
    free(m->certified_artifact_boundary_certified);
    free(m->state_transition_atlas_keys);
    free(m->state_transition_atlas_targets);
    free(m->state_transition_atlas_counts);
    free(m->counterfactual_packets_keys);
    free(m->counterfactual_packets_target_a);
    free(m->counterfactual_packets_target_b);
    free(m->counterfactual_packets_count_a);
    free(m->counterfactual_packets_count_b);
    free(m->counterfactual_packets_count_other);
    free(m->error_causal_target_counts);
    free(m->error_causal_prereq_counts);
    free(m->exact_transition_quotient_keys);
    free(m->exact_transition_quotient_counts);
    free(m->macro_ce_quotient_context_keys);
    free(m->macro_ce_quotient_context_totals);
    free(m->macro_ce_quotient_context_class_keys);
    free(m->macro_ce_quotient_context_top1_token);
    free(m->macro_ce_quotient_context_top2_token);
    free(m->macro_ce_quotient_context_top1_count);
    free(m->macro_ce_quotient_context_top2_count);
    free(m->macro_ce_quotient_triple_keys);
    free(m->macro_ce_quotient_triple_counts);
    free(m->macro_ce_quotient_class_keys);
    free(m->macro_ce_quotient_class_totals);
    free(m->macro_ce_quotient_class_target_keys);
    free(m->macro_ce_quotient_class_target_counts);
    free(m->macro_ce_quotient_random_keys);
    free(m->macro_ce_quotient_random_totals);
    free(m->macro_ce_quotient_random_target_keys);
    free(m->macro_ce_quotient_random_target_counts);
    free(m->exact_window_multiset_hashes);
    free(m->exact_window_multiset_starts);
    free(m->exact_window_multiset_positions);
    free(m->exact_window_multiset_counts);
    free(m->prefix_frontier_keys);
    free(m->prefix_frontier_counts);
    free(m->macro_transcript_keys);
    free(m->macro_transcript_counts);

    free(m);
}

/* =========================================================================
 * pack_layer_to_int2 — Quantize and pack shadow weights to INT2 storage.
 *
 * Applies round_ternary to shadow_w_in, shadow_w_out, shadow_w_conv
 * and packs into w_*_int2 using INT2 encoding (4 weights per byte).
 * ========================================================================= */

void pack_layer_to_int2(SSMLayer *L) {
    int n_in   = V12_D_MODEL * V12_D_STATE;
    int n_out  = V12_D_STATE * V12_D_MODEL;
    int n_conv = V12_CONV_KERNEL * V12_D_STATE;

    int cnt_pos = 0, cnt_neg = 0, cnt_zero = 0;

    /* Pack w_in */
    for (int b = 0; b < n_in / V12_INT2_WEIGHTS_PER_BYTE; b++) {
        uint8_t packed = 0;
        for (int k = 0; k < V12_INT2_WEIGHTS_PER_BYTE; k++) {
            float q = round_ternary(SHADOW_DECODE(L->shadow_w_in[b * V12_INT2_WEIGHTS_PER_BYTE + k]));
            uint8_t bits = (q > 0.5f) ? 0x01 : (q < -0.5f) ? 0x03 : 0x00;
            packed |= (bits << (k * 2));

            if (q > 0.5f) cnt_pos++;
            else if (q < -0.5f) cnt_neg++;
            else cnt_zero++;
        }
        L->w_in_int2[b] = packed;
    }

    printf("[INT2-DBG] w_in distribution: +1=%d  -1=%d  0=%d  total=%d\n",
           cnt_pos, cnt_neg, cnt_zero, n_in);

    /* Pack w_out */
    for (int b = 0; b < n_out / V12_INT2_WEIGHTS_PER_BYTE; b++) {
        uint8_t packed = 0;
        for (int k = 0; k < V12_INT2_WEIGHTS_PER_BYTE; k++) {
            float q = round_ternary(SHADOW_DECODE(L->shadow_w_out[b * V12_INT2_WEIGHTS_PER_BYTE + k]));
            uint8_t bits = (q > 0.5f) ? 0x01 : (q < -0.5f) ? 0x03 : 0x00;
            packed |= (bits << (k * 2));
        }
        L->w_out_int2[b] = packed;
    }

    /* Pack w_conv */
    for (int b = 0; b < n_conv / V12_INT2_WEIGHTS_PER_BYTE; b++) {
        uint8_t packed = 0;
        for (int k = 0; k < V12_INT2_WEIGHTS_PER_BYTE; k++) {
            float q = round_ternary(SHADOW_DECODE(L->shadow_w_conv[b * V12_INT2_WEIGHTS_PER_BYTE + k]));
            uint8_t bits = (q > 0.5f) ? 0x01 : (q < -0.5f) ? 0x03 : 0x00;
            packed |= (bits << (k * 2));
        }
        L->w_conv_int2[b] = packed;
    }

    printf("[INT2] Packed layer: %d w_in + %d w_out + %d w_conv → %d bytes\n",
           n_in, n_out, n_conv, (n_in + n_out + n_conv) / V12_INT2_WEIGHTS_PER_BYTE);
}

/* =========================================================================
 * unpack_layer_from_int2 — Reconstruct fp32 shadow weights from INT2 storage.
 *
 * Extracts ternary {-1, 0, +1} values from packed INT2 buffers.
 * Used when reactivating a frozen layer for BCD or joint training.
 * ========================================================================= */

void unpack_layer_from_int2(SSMLayer *L) {
    int n_in   = V12_D_MODEL * V12_D_STATE;
    int n_out  = V12_D_STATE * V12_D_MODEL;
    int n_conv = V12_CONV_KERNEL * V12_D_STATE;

    /* Unpack w_in */
    for (int b = 0; b < n_in / V12_INT2_WEIGHTS_PER_BYTE; b++) {
        uint8_t packed = L->w_in_int2[b];
        for (int k = 0; k < V12_INT2_WEIGHTS_PER_BYTE; k++) {
            uint8_t bits = (packed >> (k * 2)) & 0x03;
            float v = (bits == 0x01) ?  1.0f : (bits == 0x03) ? -1.0f : 0.0f;
#if V12_F2_INT8_SHADOWS
            uint32_t r = shadow_rng(0, V12_SHADOW_MAT_W_IN,
                                    b * V12_INT2_WEIGHTS_PER_BYTE + k, 0ULL);
            SHADOW_STORE(L->shadow_w_in[b * V12_INT2_WEIGHTS_PER_BYTE + k], v, r);
#else
            L->shadow_w_in[b * V12_INT2_WEIGHTS_PER_BYTE + k] = v;
#endif
        }
    }

    /* Unpack w_out */
    for (int b = 0; b < n_out / V12_INT2_WEIGHTS_PER_BYTE; b++) {
        uint8_t packed = L->w_out_int2[b];
        for (int k = 0; k < V12_INT2_WEIGHTS_PER_BYTE; k++) {
            uint8_t bits = (packed >> (k * 2)) & 0x03;
            float v = (bits == 0x01) ?  1.0f : (bits == 0x03) ? -1.0f : 0.0f;
#if V12_F2_INT8_SHADOWS
            uint32_t r = shadow_rng(0, V12_SHADOW_MAT_W_OUT,
                                    b * V12_INT2_WEIGHTS_PER_BYTE + k, 0ULL);
            SHADOW_STORE(L->shadow_w_out[b * V12_INT2_WEIGHTS_PER_BYTE + k], v, r);
#else
            L->shadow_w_out[b * V12_INT2_WEIGHTS_PER_BYTE + k] = v;
#endif
        }
    }

    /* Unpack w_conv */
    for (int b = 0; b < n_conv / V12_INT2_WEIGHTS_PER_BYTE; b++) {
        uint8_t packed = L->w_conv_int2[b];
        for (int k = 0; k < V12_INT2_WEIGHTS_PER_BYTE; k++) {
            uint8_t bits = (packed >> (k * 2)) & 0x03;
            float v = (bits == 0x01) ?  1.0f : (bits == 0x03) ? -1.0f : 0.0f;
#if V12_F2_INT8_SHADOWS
            uint32_t r = shadow_rng(0, V12_SHADOW_MAT_W_CONV,
                                    b * V12_INT2_WEIGHTS_PER_BYTE + k, 0ULL);
            SHADOW_STORE(L->shadow_w_conv[b * V12_INT2_WEIGHTS_PER_BYTE + k], v, r);
#else
            L->shadow_w_conv[b * V12_INT2_WEIGHTS_PER_BYTE + k] = v;
#endif
        }
    }

    printf("[INT2] Unpacked layer to shadow fp32\n");
}

/* =========================================================================
 * layer_offload_fp32 — Free fp32 scaffolding for frozen layer
 *
 * Frees all training-only fp32 buffers. Sentinel: dw_in == NULL after free.
 * ========================================================================= */

void layer_offload_fp32(SSMLayer *L) {
    free(L->sampled_w_in); L->sampled_w_in = NULL;
    free(L->sampled_w_out); L->sampled_w_out = NULL;
    free(L->crystal_mask_in); L->crystal_mask_in = NULL;
    free(L->crystal_mask_out); L->crystal_mask_out = NULL;
    free(L->grad_accum_in); L->grad_accum_in = NULL;
    free(L->grad_accum_out); L->grad_accum_out = NULL;
    free(L->dw_in);  L->dw_in = NULL;
    free(L->dw_out); L->dw_out = NULL;
    free(L->dA);     L->dA = NULL;
    free(L->dB);     L->dB = NULL;
    free(L->dB_seq); L->dB_seq = NULL;
    if (L->dC) { free(L->dC); L->dC = NULL; }
    free(L->d_conv_w); L->d_conv_w = NULL;
    free(L->m_w_in_row); L->m_w_in_row = NULL;
    free(L->m_w_in_col); L->m_w_in_col = NULL;
    free(L->v_w_in_row); L->v_w_in_row = NULL;
    free(L->v_w_in_col); L->v_w_in_col = NULL;
    free(L->m_w_out_row); L->m_w_out_row = NULL;
    free(L->m_w_out_col); L->m_w_out_col = NULL;
    free(L->v_w_out_row); L->v_w_out_row = NULL;
    free(L->v_w_out_col); L->v_w_out_col = NULL;
    free(L->m_A); L->m_A = NULL;
    free(L->v_A); L->v_A = NULL;
    free(L->m_B); L->m_B = NULL;
    free(L->v_B); L->v_B = NULL;
    if (L->m_C) { free(L->m_C); L->m_C = NULL; }
    if (L->v_C) { free(L->v_C); L->v_C = NULL; }
    free(L->m_conv); L->m_conv = NULL;
    free(L->v_conv); L->v_conv = NULL;
    free(L->d_mtp_U); L->d_mtp_U = NULL;
    free(L->d_mtp_V); L->d_mtp_V = NULL;
    free(L->m_mtp_U); L->m_mtp_U = NULL;
    free(L->v_mtp_U); L->v_mtp_U = NULL;
    free(L->m_mtp_V); L->m_mtp_V = NULL;
    free(L->v_mtp_V); L->v_mtp_V = NULL;
    printf("[OFFLOAD] Layer fp32 freed (sentinel: dw_in=NULL)\n");
}

/* =========================================================================
 * layer_restore_fp32 — Restore fp32 scaffolding for unfrozen layer
 *
 * Re-allocates all training-only fp32 buffers, zero-initialized.
 * Then unpacks INT2 weights into shadow arrays.
 * ========================================================================= */

void layer_restore_fp32(SSMLayer *L) {
    int n_in = V12_D_MODEL * V12_D_STATE;
    int n_out = V12_D_STATE * V12_D_MODEL;
    int n_conv = V12_CONV_KERNEL * V12_D_STATE;

    if (L->shadow_w_in == NULL) {
        L->shadow_w_in = calloc(n_in, SHADOW_SIZEOF);
        L->shadow_w_out = calloc(n_out, SHADOW_SIZEOF);
        L->shadow_w_conv = calloc(n_conv, SHADOW_SIZEOF);
        unpack_layer_from_int2(L);
        printf("[RESTORE] Layer fp32 restored from INT2 (lossy — shadows were freed)\n");
    } else {
        if (L->shadow_w_conv == NULL) {
            L->shadow_w_conv = calloc(n_conv, SHADOW_SIZEOF);
        }
        printf("[RESTORE] Layer fp32 restored from retained shadows (lossless)\n");
    }

    L->sampled_w_in = calloc(n_in, sizeof(float));
    L->sampled_w_out = calloc(n_out, sizeof(float));
    L->crystal_mask_in = calloc(n_in, sizeof(uint8_t));
    L->crystal_mask_out = calloc(n_out, sizeof(uint8_t));
    L->grad_accum_in = calloc(n_in, sizeof(float));
    L->grad_accum_out = calloc(n_out, sizeof(float));
    L->dw_in = calloc(n_in, sizeof(float));
    L->dw_out = calloc(n_out, sizeof(float));
    L->dA = calloc(V12_D_STATE, sizeof(float));
    L->dB = calloc(V12_D_STATE, sizeof(float));
    L->dB_seq = calloc(V12_SEQ_LEN * V12_D_STATE, sizeof(float));
    L->dC = NULL;  /* V-free: not trained */
    L->d_conv_w = calloc(n_conv, sizeof(float));
    L->m_w_in_row = calloc(V12_D_MODEL, sizeof(float));
    L->m_w_in_col = calloc(V12_D_STATE, sizeof(float));
    L->v_w_in_row = calloc(V12_D_MODEL, sizeof(float));
    L->v_w_in_col = calloc(V12_D_STATE, sizeof(float));
    L->m_w_out_row = calloc(V12_D_STATE, sizeof(float));
    L->m_w_out_col = calloc(V12_D_MODEL, sizeof(float));
    L->v_w_out_row = calloc(V12_D_STATE, sizeof(float));
    L->v_w_out_col = calloc(V12_D_MODEL, sizeof(float));
    L->m_A = calloc(V12_D_STATE, sizeof(float));
    L->v_A = calloc(V12_D_STATE, sizeof(float));
    L->m_B = calloc(V12_D_STATE, sizeof(float));
    L->v_B = calloc(V12_D_STATE, sizeof(float));
    L->m_C = NULL;  /* V-free: not trained */
    L->v_C = NULL;  /* V-free: not trained */
    L->m_conv = calloc(n_conv, sizeof(float));
    L->v_conv = calloc(n_conv, sizeof(float));

    /* Restore MTP gradient and Adam moment buffers for even layers */
    if (L->mtp_U != NULL) {
        L->d_mtp_U = calloc((size_t)V12_D_STATE * V12_MTP_RANK, sizeof(float));
        L->d_mtp_V = calloc((size_t)V12_D_STATE * V12_MTP_RANK, sizeof(float));
        L->m_mtp_U = calloc((size_t)V12_D_STATE * V12_MTP_RANK, sizeof(float));
        L->v_mtp_U = calloc((size_t)V12_D_STATE * V12_MTP_RANK, sizeof(float));
        L->m_mtp_V = calloc((size_t)V12_D_STATE * V12_MTP_RANK, sizeof(float));
        L->v_mtp_V = calloc((size_t)V12_D_STATE * V12_MTP_RANK, sizeof(float));
    }

#if V12_F2_INT8_SHADOWS
    /* sampled_* stays FP32; shadow_* is int16 — decode element-wise. */
    for (int i = 0; i < n_in; i++)  L->sampled_w_in[i]  = SHADOW_DECODE(L->shadow_w_in[i]);
    for (int i = 0; i < n_out; i++) L->sampled_w_out[i] = SHADOW_DECODE(L->shadow_w_out[i]);
#else
    memcpy(L->sampled_w_in, L->shadow_w_in, n_in * sizeof(float));
    memcpy(L->sampled_w_out, L->shadow_w_out, n_out * sizeof(float));
#endif
}

/* =========================================================================
 * cache_alloc — Allocate forward and backward cache
 * ========================================================================= */

Cache *cache_alloc(int batch, int seq) {
    if (seq != V12_SEQ_LEN) {
        fprintf(stderr, "cache_alloc: seq=%d but V12_SEQ_LEN=%d\n", seq, V12_SEQ_LEN);
        return NULL;
    }
    if (batch < 1) {
        fprintf(stderr, "cache_alloc: batch=%d must be >= 1\n", batch);
        return NULL;
    }

    Cache *c = calloc(1, sizeof(Cache));
    if (!c) return NULL;

    int n_layers = V12_N_LAYERS;
    int d_state = V12_D_STATE;
    int d_model = V12_D_MODEL;
    int vocab = V12_VOCAB_SIZE;
    int bseq = batch * seq;

    /* --- Forward activations ---------------------------------------------- */

    c->h_all = malloc((size_t)n_layers * bseq * d_state * sizeof(uint8_t));
    if (!c->h_all) { cache_free(c); return NULL; }

    c->h_ssm_final = malloc(n_layers * d_state * sizeof(float));
    if (!c->h_ssm_final) { cache_free(c); return NULL; }

#if V12_AF8C_ENABLED
    c->x_proj = malloc((size_t)n_layers * bseq * d_state * sizeof(uint8_t));
    if (!c->x_proj) { cache_free(c); return NULL; }

    c->x_conv = malloc((size_t)n_layers * bseq * d_state * sizeof(uint8_t));
    if (!c->x_conv) { cache_free(c); return NULL; }

    c->x_gated = malloc((size_t)n_layers * bseq * d_state * sizeof(uint8_t));
    if (!c->x_gated) { cache_free(c); return NULL; }

    c->mask_val = malloc((size_t)n_layers * bseq * d_state * sizeof(uint8_t));
    if (!c->mask_val) { cache_free(c); return NULL; }

    c->b_eff_gate = malloc((size_t)n_layers * bseq * d_state * sizeof(uint8_t));
    if (!c->b_eff_gate) { cache_free(c); return NULL; }

    c->a_eff_gate = malloc((size_t)n_layers * bseq * d_state * sizeof(uint8_t));
    if (!c->a_eff_gate) { cache_free(c); return NULL; }

    c->a_eff_buf = malloc((size_t)n_layers * bseq * d_state * sizeof(uint8_t));
    if (!c->a_eff_buf) { cache_free(c); return NULL; }

    c->dA_per_step = malloc((size_t)n_layers * bseq * d_state * sizeof(uint8_t));
    if (!c->dA_per_step) { cache_free(c); return NULL; }
#else
    c->x_proj = malloc((size_t)n_layers * bseq * d_state * sizeof(float));
    if (!c->x_proj) { cache_free(c); return NULL; }

    c->x_conv = malloc((size_t)n_layers * bseq * d_state * sizeof(float));
    if (!c->x_conv) { cache_free(c); return NULL; }

    c->x_gated = malloc((size_t)n_layers * bseq * d_state * sizeof(float));
    if (!c->x_gated) { cache_free(c); return NULL; }

    c->mask_val = malloc((size_t)n_layers * bseq * d_state * sizeof(float));
    if (!c->mask_val) { cache_free(c); return NULL; }

    c->b_eff_gate = malloc((size_t)n_layers * bseq * d_state * sizeof(float));
    if (!c->b_eff_gate) { cache_free(c); return NULL; }

    c->a_eff_gate = malloc((size_t)n_layers * bseq * d_state * sizeof(float));
    if (!c->a_eff_gate) { cache_free(c); return NULL; }

    c->a_eff_buf = malloc((size_t)n_layers * bseq * d_state * sizeof(float));
    if (!c->a_eff_buf) { cache_free(c); return NULL; }

    c->dA_per_step = malloc((size_t)n_layers * bseq * d_state * sizeof(float));
    if (!c->dA_per_step) { cache_free(c); return NULL; }
#endif

    c->residual = malloc((size_t)(n_layers + 1) * bseq * d_model * sizeof(float));
    if (!c->residual) { cache_free(c); return NULL; }

    const int N = V12_SS_N_SAMPLES;

    c->logits = malloc((size_t)seq * N * sizeof(float));
    if (!c->logits) { cache_free(c); return NULL; }

    c->sample_ids = malloc((size_t)seq * N * sizeof(int32_t));
    if (!c->sample_ids) { cache_free(c); return NULL; }

    c->log_proposal = malloc((size_t)seq * N * sizeof(float));
    if (!c->log_proposal) { cache_free(c); return NULL; }

#if V12_AF8C_ENABLED
    c->y_ssm = malloc((size_t)n_layers * bseq * d_state * sizeof(uint8_t));
    if (!c->y_ssm) { cache_free(c); return NULL; }
#else
    c->y_ssm = malloc((size_t)n_layers * bseq * d_state * sizeof(float));
    if (!c->y_ssm) { cache_free(c); return NULL; }
#endif

    /* --- Layer norm cache (batch × seq × d_model) -------------------------- */

    c->norm_out = malloc((size_t)bseq * d_model * sizeof(float));
    if (!c->norm_out) { cache_free(c); return NULL; }

    c->norm_rms = malloc((size_t)bseq * sizeof(float));
    if (!c->norm_rms) { cache_free(c); return NULL; }

    /* --- Backward scratch buffers ------------------------------------------ */

    c->d_logits = malloc((size_t)seq * N * sizeof(float));
    if (!c->d_logits) { cache_free(c); return NULL; }

#if V12_AF8C_ENABLED
    c->d_h = malloc((size_t)n_layers * bseq * d_state * sizeof(uint8_t));
    if (!c->d_h) { cache_free(c); return NULL; }

    c->d_x_conv = malloc((size_t)n_layers * bseq * d_state * sizeof(uint8_t));
    if (!c->d_x_conv) { cache_free(c); return NULL; }

    c->d_x_proj = malloc((size_t)n_layers * bseq * d_state * sizeof(uint8_t));
    if (!c->d_x_proj) { cache_free(c); return NULL; }
#else
    c->d_h = malloc((size_t)n_layers * bseq * d_state * sizeof(float));
    if (!c->d_h) { cache_free(c); return NULL; }

    c->d_x_conv = malloc((size_t)n_layers * bseq * d_state * sizeof(float));
    if (!c->d_x_conv) { cache_free(c); return NULL; }

    c->d_x_proj = malloc((size_t)n_layers * bseq * d_state * sizeof(float));
    if (!c->d_x_proj) { cache_free(c); return NULL; }
#endif

    c->d_residual = malloc((size_t)(n_layers + 1) * bseq * d_model * sizeof(float));
    if (!c->d_residual) { cache_free(c); return NULL; }

    /* --- Cached max logits (scratch, reused per batch item) ---------------- */

    c->max_logit = calloc(seq, sizeof(float));
    if (!c->max_logit) { cache_free(c); return NULL; }

    return c;
}

/* =========================================================================
 * cache_free — Deallocate forward and backward cache
 * ========================================================================= */

void cache_free(Cache *c) {
    if (!c) return;

    free(c->h_all);
    free(c->h_ssm_final);
    free(c->x_proj);
    free(c->x_conv);
    free(c->x_gated);
    free(c->mask_val);
    free(c->residual);
    free(c->logits);
    free(c->sample_ids);
    free(c->log_proposal);
    free(c->y_ssm);

    free(c->d_logits);
    free(c->d_h);
    free(c->d_x_conv);
    free(c->d_x_proj);
    free(c->d_residual);

    free(c->max_logit);
    free(c->b_eff_gate);
    free(c->a_eff_gate);
    free(c->a_eff_buf);
    free(c->dA_per_step);

    free(c->norm_out);
    free(c->norm_rms);

    free(c);
}
