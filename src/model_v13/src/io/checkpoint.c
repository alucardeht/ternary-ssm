#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "../core/types.h"
#include "../core/model.h"
#include "../core/optimizer.h"
#include "checkpoint.h"

typedef struct {
    uint32_t step;
    float    lr_base;
    float    lr_per_layer[V12_MAX_LAYERS];
    float    loss_history[V12_LOSS_HISTORY_LEN];
    uint32_t loss_history_head;
    float    eval_gap;
    uint32_t fp_stage;
    uint8_t  fp_completed;
    uint8_t  bcd_active;
    uint8_t  bcd_cycle;
    uint8_t  bcd_layer;
    uint64_t rng_state;
    float    crystal_fraction;
    uint8_t  _reserved[4];
} TrainStateV8Compat;

static void train_state_from_v8_compat(TrainState *dst,
                                       const TrainStateV8Compat *src) {
    memset(dst, 0, sizeof(*dst));
    dst->step = src->step;
    dst->lr_base = src->lr_base;
    memcpy(dst->lr_per_layer, src->lr_per_layer, sizeof(dst->lr_per_layer));
    memcpy(dst->loss_history, src->loss_history, sizeof(dst->loss_history));
    dst->loss_history_head = src->loss_history_head;
    dst->eval_gap = src->eval_gap;
    dst->fp_stage = src->fp_stage;
    dst->fp_completed = src->fp_completed;
    dst->bcd_active = src->bcd_active;
    dst->bcd_cycle = src->bcd_cycle;
    dst->bcd_layer = src->bcd_layer;
    dst->rng_state = src->rng_state;
    dst->dataset_cursor = 0;
    memset(dst->dataset_rng, 0, sizeof(dst->dataset_rng));
    dst->dataset_state_valid = 0;
    dst->crystal_fraction = src->crystal_fraction;
}


/* --- Checked read/write macros ------------------------------------------------ */

static int _ckpt_write_idx = 0;
static int _ckpt_read_idx = 0;

#define WRITE(ptr, size, fp) \
    do { \
        _ckpt_write_idx++; \
        if ((ptr) == NULL) { \
            fprintf(stderr, "[CKPT] NULL ptr at write #%d (size=%zu)\n", _ckpt_write_idx, (size_t)(size)); \
            goto io_error; \
        } \
        if (fwrite((ptr), 1, (size), (fp)) != (size)) { \
            fprintf(stderr, "[CKPT] fwrite failed at write #%d (size=%zu, ftell=%ld)\n", _ckpt_write_idx, (size_t)(size), ftell(fp)); \
            goto io_error; \
        } \
    } while(0)

#define READ(ptr, size, fp) \
    do { \
        _ckpt_read_idx++; \
        if ((ptr) == NULL) { \
            fprintf(stderr, "[CKPT] NULL ptr at read #%d (size=%zu)\n", \
                    _ckpt_read_idx, (size_t)(size)); \
            goto io_error; \
        } \
        if (fread((ptr), 1, (size), (fp)) != (size)) { \
            fprintf(stderr, "[CKPT] fread failed at read #%d (size=%zu, ftell=%ld)\n", \
                    _ckpt_read_idx, (size_t)(size), ftell(fp)); \
            goto io_error; \
        } \
    } while(0)


/* === Chunked write/read for large buffers (Windows CRT ~64MB limit workaround) === */

static int write_chunked(FILE *fp, const void *ptr, size_t total) {
    const size_t CHUNK = 64 * 1024 * 1024;
    const uint8_t *p = (const uint8_t *)ptr;
    size_t remaining = total;
    while (remaining > 0) {
        size_t n = (remaining < CHUNK) ? remaining : CHUNK;
        if (fwrite(p, 1, n, fp) != n) return -1;
        p += n;
        remaining -= n;
    }
    return 0;
}

static int read_chunked(FILE *fp, void *ptr, size_t total) {
    const size_t CHUNK = 64 * 1024 * 1024;
    uint8_t *p = (uint8_t *)ptr;
    size_t remaining = total;
    while (remaining > 0) {
        size_t n = (remaining < CHUNK) ? remaining : CHUNK;
        if (fread(p, 1, n, fp) != n) return -1;
        p += n;
        remaining -= n;
    }
    return 0;
}

#define WRITE_LARGE(ptr, size, fp) \
    do { \
        _ckpt_write_idx++; \
        if ((ptr) == NULL) { \
            fprintf(stderr, "[CKPT] NULL ptr at write #%d (size=%zu, LARGE)\n", _ckpt_write_idx, (size_t)(size)); \
            goto io_error; \
        } \
        if (write_chunked((fp), (ptr), (size)) != 0) { \
            fprintf(stderr, "[CKPT] write_chunked failed at write #%d (size=%zu, ftell=%ld)\n", _ckpt_write_idx, (size_t)(size), ftell(fp)); \
            goto io_error; \
        } \
    } while(0)

#define READ_LARGE(ptr, size, fp) \
    do { \
        _ckpt_read_idx++; \
        if ((ptr) == NULL) { \
            fprintf(stderr, "[CKPT] NULL ptr at read #%d (size=%zu, LARGE)\n", \
                    _ckpt_read_idx, (size_t)(size)); \
            goto io_error; \
        } \
        if (read_chunked((fp), (ptr), (size)) != 0) { \
            fprintf(stderr, "[CKPT] read_chunked failed at read #%d (size=%zu, ftell=%ld)\n", \
                    _ckpt_read_idx, (size_t)(size), ftell(fp)); \
            goto io_error; \
        } \
    } while(0)


/* =========================================================================
 * checkpoint_save
 * ========================================================================= */

int checkpoint_save(const char *path, const Model *m, const TrainState *s) {
    if (!path || !m || !s) {
        return -1;
    }

    char tmp_path[4096];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);

    FILE *fp = fopen(tmp_path, "wb");
    if (!fp) {
        fprintf(stderr, "[CKPT] Cannot open %s for writing\n", tmp_path);
        return -1;
    }

    _ckpt_write_idx = 0;

    uint32_t magic = V12_CHECKPOINT_MAGIC;
    uint32_t version = V12_CHECKPOINT_VERSION;
    WRITE(&magic, sizeof(uint32_t), fp);
    WRITE(&version, sizeof(uint32_t), fp);

    /* TrainState blob: direct write (pointer-free) */
    WRITE(s, sizeof(TrainState), fp);

    /* Dimension header */
    WRITE(&m->n_layers, sizeof(uint32_t), fp);
    WRITE(&m->d_model, sizeof(uint32_t), fp);
    WRITE(&m->d_state, sizeof(uint32_t), fp);
    WRITE(&m->vocab_size, sizeof(uint32_t), fp);
    WRITE(&m->conv_kernel, sizeof(uint32_t), fp);
    WRITE(&m->mtp_rank, sizeof(uint32_t), fp);

    int d = (int)m->d_model;
    int de = (int)V12_D_EMBED;
    int ds = (int)m->d_state;
    int V = (int)m->vocab_size;
    int K = (int)m->conv_kernel;
    int R = (int)m->mtp_rank;
    int nl = (int)m->n_layers;

    /* Per-layer parameters */
    for (int k = 0; k < nl; k++) {
        const SSMLayer *L = &m->layers[k];

        WRITE(L->A, ds * sizeof(float), fp);
        WRITE(L->B, ds * sizeof(float), fp);
        WRITE(L->C, ds * ds * sizeof(float), fp);
        WRITE(L->shadow_w_in, d * ds * sizeof(float), fp);
        WRITE(L->sampled_w_in, d * ds * sizeof(float), fp);
        WRITE(L->shadow_w_out, ds * d * sizeof(float), fp);
        WRITE(L->sampled_w_out, ds * d * sizeof(float), fp);
        WRITE(L->shadow_w_conv, K * ds * sizeof(float), fp);

        uint8_t has_mtp = (L->mtp_U != NULL) ? 1 : 0;
        WRITE(&has_mtp, 1, fp);

        if (has_mtp) {
            WRITE(L->mtp_U, ds * R * sizeof(float), fp);
            WRITE(L->mtp_V, ds * R * sizeof(float), fp);
        }

        /* === V3: Factored Adam moment buffers === */
        WRITE(L->m_w_in_row, d * sizeof(float), fp);
        WRITE(L->m_w_in_col, ds * sizeof(float), fp);
        WRITE(L->v_w_in_row, d * sizeof(float), fp);
        WRITE(L->v_w_in_col, ds * sizeof(float), fp);
        WRITE(L->m_w_out_row, ds * sizeof(float), fp);
        WRITE(L->m_w_out_col, d * sizeof(float), fp);
        WRITE(L->v_w_out_row, ds * sizeof(float), fp);
        WRITE(L->v_w_out_col, d * sizeof(float), fp);
        WRITE(L->m_A, ds * sizeof(float), fp);
        WRITE(L->v_A, ds * sizeof(float), fp);
        WRITE(L->m_B, ds * sizeof(float), fp);
        WRITE(L->v_B, ds * sizeof(float), fp);

        uint8_t has_C_moments = (L->m_C != NULL) ? 1 : 0;
        WRITE(&has_C_moments, 1, fp);
        if (has_C_moments) {
            WRITE(L->m_C, ds * ds * sizeof(float), fp);
            WRITE(L->v_C, ds * ds * sizeof(float), fp);
        }

        WRITE(L->m_conv, K * ds * sizeof(float), fp);
        WRITE(L->v_conv, K * ds * sizeof(float), fp);

        uint8_t has_mtp_m = (L->m_mtp_U != NULL) ? 1 : 0;
        WRITE(&has_mtp_m, 1, fp);
        if (has_mtp_m) {
            WRITE(L->m_mtp_U, ds * R * sizeof(float), fp);
            WRITE(L->v_mtp_U, ds * R * sizeof(float), fp);
            WRITE(L->m_mtp_V, ds * R * sizeof(float), fp);
            WRITE(L->v_mtp_V, ds * R * sizeof(float), fp);
        }

        WRITE(L->W_A, ds * ds * sizeof(float), fp);
        WRITE(L->b_A, ds * sizeof(float), fp);

        WRITE(L->m_W_A, ds * ds * sizeof(float), fp);
        WRITE(L->v_W_A, ds * ds * sizeof(float), fp);
        WRITE(L->m_b_A, ds * sizeof(float), fp);
        WRITE(L->v_b_A, ds * sizeof(float), fp);
    }

    /* Model-level parameters */
    WRITE_LARGE(m->shadow_lm_head, d * V * sizeof(float), fp);
    WRITE_LARGE(m->sampled_lm_head, d * V * sizeof(float), fp);
    WRITE(m->shadow_w_pred, d * d * sizeof(float), fp);
    WRITE(m->sampled_w_pred, d * d * sizeof(float), fp);
    WRITE(m->norm_w, d * sizeof(float), fp);
    WRITE(m->norm_b, d * sizeof(float), fp);
    WRITE(m->embed_E, V * de * sizeof(float), fp);
    WRITE(m->embed_P, de * d * sizeof(float), fp);
    WRITE(m->m_embed_E, V * de * sizeof(float), fp);
    WRITE(m->v_embed_E, V * de * sizeof(float), fp);
    WRITE(m->m_embed_P, de * d * sizeof(float), fp);
    WRITE(m->v_embed_P, de * d * sizeof(float), fp);
    WRITE(m->tau_mux, nl * sizeof(float), fp);

    /* === V3: Factored Adam moment buffers === */
    WRITE(m->m_lm_head_row, V12_K_HOT * sizeof(float), fp);
    WRITE(m->m_lm_head_col, d * sizeof(float), fp);
    WRITE(m->v_lm_head_row, V12_K_HOT * sizeof(float), fp);
    WRITE(m->v_lm_head_col, d * sizeof(float), fp);
    WRITE(m->m_w_pred_row, d * sizeof(float), fp);
    WRITE(m->m_w_pred_col, d * sizeof(float), fp);
    WRITE(m->v_w_pred_row, d * sizeof(float), fp);
    WRITE(m->v_w_pred_col, d * sizeof(float), fp);
    WRITE(m->m_norm_w, d * sizeof(float), fp);
    WRITE(m->v_norm_w, d * sizeof(float), fp);
    WRITE(m->m_norm_b, d * sizeof(float), fp);
    WRITE(m->v_norm_b, d * sizeof(float), fp);

    fclose(fp);

    /* Atomic rename */
    remove(path);
    if (rename(tmp_path, path) != 0) {
        fprintf(stderr, "[CKPT] rename %s -> %s failed\n", tmp_path, path);
        remove(tmp_path);
        return -1;
    }

    printf("[CKPT] Saved step=%u → %s\n", s->step, path);
    if (version < V12_CHECKPOINT_VERSION) {
        printf("[CKPT] compatibility_load=1 source_format=%u target_format=%u\n",
               version, V12_CHECKPOINT_VERSION);
    }
    return 0;

io_error:
    fprintf(stderr, "[CKPT] I/O error writing %s\n", tmp_path);
    fclose(fp);
    remove(tmp_path);
    return -1;
}


/* =========================================================================
 * checkpoint_load
 * ========================================================================= */

int checkpoint_load(const char *path, Model *m, TrainState *s) {
    if (!path || !m || !s) {
        return -1;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        /* File not found is not an error (first run) */
        return -1;
    }

    _ckpt_read_idx = 0;

    uint32_t magic = 0;
    uint32_t version = 0;
    READ(&magic, sizeof(uint32_t), fp);
    READ(&version, sizeof(uint32_t), fp);

    if (magic != V12_CHECKPOINT_MAGIC) {
        fprintf(stderr, "[CKPT] Bad magic in %s (got 0x%08X, expected 0x%08X)\n",
                path, magic, V12_CHECKPOINT_MAGIC);
        fclose(fp);
        return -1;
    }

    if (version < V12_CHECKPOINT_MIN_SUPPORTED_VERSION ||
        version > V12_CHECKPOINT_VERSION) {
        fprintf(stderr, "[CKPT] Unsupported version %u in %s (supported %u..%u)\n",
                version, path, V12_CHECKPOINT_MIN_SUPPORTED_VERSION,
                V12_CHECKPOINT_VERSION);
        fclose(fp);
        return -1;
    }

    if (version >= 9U) {
        READ(s, sizeof(TrainState), fp);
    } else {
        TrainStateV8Compat state_v8;
        READ(&state_v8, sizeof(state_v8), fp);
        train_state_from_v8_compat(s, &state_v8);
        printf("[CKPT] WARNING: dataset_cursor absent in format v%u; restored to 0\n",
               version);
    }

    uint32_t n_layers = 0;
    uint32_t d_model = 0;
    uint32_t d_state = 0;
    uint32_t vocab_size = 0;
    uint32_t conv_kernel = 0;
    uint32_t mtp_rank = 0;

    READ(&n_layers, sizeof(uint32_t), fp);
    READ(&d_model, sizeof(uint32_t), fp);
    READ(&d_state, sizeof(uint32_t), fp);
    READ(&vocab_size, sizeof(uint32_t), fp);
    READ(&conv_kernel, sizeof(uint32_t), fp);
    READ(&mtp_rank, sizeof(uint32_t), fp);

    /* Validate dimensions match current build */
    if (n_layers != m->n_layers ||
        d_model != m->d_model ||
        d_state != m->d_state ||
        vocab_size != m->vocab_size ||
        conv_kernel != m->conv_kernel ||
        mtp_rank != m->mtp_rank) {
        fprintf(stderr,
            "[CKPT] Dimension mismatch in %s\n"
            "  file:  nl=%u d=%u ds=%u V=%u K=%u R=%u\n"
            "  build: nl=%u d=%u ds=%u V=%u K=%u R=%u\n",
            path,
            n_layers, d_model, d_state, vocab_size, conv_kernel, mtp_rank,
            m->n_layers, m->d_model, m->d_state, m->vocab_size,
            m->conv_kernel, m->mtp_rank);
        fclose(fp);
        return -1;
    }

    int d = (int)d_model;
    int de = (int)V12_D_EMBED;
    int ds = (int)d_state;
    int V = (int)vocab_size;
    int K = (int)conv_kernel;
    int R = (int)mtp_rank;
    int nl = (int)n_layers;

    for (int k = 0; k < nl; k++) {
        SSMLayer *L = &m->layers[k];

        READ(L->A, ds * sizeof(float), fp);
        READ(L->B, ds * sizeof(float), fp);
        READ(L->C, ds * ds * sizeof(float), fp);
        READ(L->shadow_w_in, d * ds * sizeof(float), fp);
        READ(L->sampled_w_in, d * ds * sizeof(float), fp);
        READ(L->shadow_w_out, ds * d * sizeof(float), fp);
        READ(L->sampled_w_out, ds * d * sizeof(float), fp);
        READ(L->shadow_w_conv, K * ds * sizeof(float), fp);

        uint8_t has_mtp = 0;
        READ(&has_mtp, 1, fp);

        if (has_mtp) {
            if (L->mtp_U == NULL || L->mtp_V == NULL) {
                fprintf(stderr,
                    "[CKPT] Layer %d: checkpoint has MTP but model does not\n", k);
                fclose(fp);
                return -1;
            }
            READ(L->mtp_U, ds * R * sizeof(float), fp);
            READ(L->mtp_V, ds * R * sizeof(float), fp);
        }

        /* === V3: Factored Adam moment buffers === */
        if (version >= 3U) {
            READ(L->m_w_in_row, d * sizeof(float), fp);
            READ(L->m_w_in_col, ds * sizeof(float), fp);
            READ(L->v_w_in_row, d * sizeof(float), fp);
            READ(L->v_w_in_col, ds * sizeof(float), fp);
            READ(L->m_w_out_row, ds * sizeof(float), fp);
            READ(L->m_w_out_col, d * sizeof(float), fp);
            READ(L->v_w_out_row, ds * sizeof(float), fp);
            READ(L->v_w_out_col, d * sizeof(float), fp);
            READ(L->m_A, ds * sizeof(float), fp);
            READ(L->v_A, ds * sizeof(float), fp);
            READ(L->m_B, ds * sizeof(float), fp);
            READ(L->v_B, ds * sizeof(float), fp);

            uint8_t has_C_moments = 0;
            READ(&has_C_moments, 1, fp);
            if (has_C_moments) {
                if (L->m_C == NULL || L->v_C == NULL) {
                    fprintf(stderr,
                        "[CKPT] Layer %d: checkpoint has C moments but model does not\n", k);
                    fclose(fp);
                    return -1;
                }
                READ(L->m_C, ds * ds * sizeof(float), fp);
                READ(L->v_C, ds * ds * sizeof(float), fp);
            }

            READ(L->m_conv, K * ds * sizeof(float), fp);
            READ(L->v_conv, K * ds * sizeof(float), fp);

            uint8_t has_mtp_m = 0;
            READ(&has_mtp_m, 1, fp);
            if (has_mtp_m) {
                if (L->m_mtp_U == NULL || L->v_mtp_U == NULL ||
                    L->m_mtp_V == NULL || L->v_mtp_V == NULL) {
                    fprintf(stderr,
                        "[CKPT] Layer %d: checkpoint has MTP moments but model does not\n", k);
                    fclose(fp);
                    return -1;
                }
                READ(L->m_mtp_U, ds * R * sizeof(float), fp);
                READ(L->v_mtp_U, ds * R * sizeof(float), fp);
                READ(L->m_mtp_V, ds * R * sizeof(float), fp);
                READ(L->v_mtp_V, ds * R * sizeof(float), fp);
            }
        }

        if (version >= 5U) {
            READ(L->W_A, ds * ds * sizeof(float), fp);
            READ(L->b_A, ds * sizeof(float), fp);

            READ(L->m_W_A, ds * ds * sizeof(float), fp);
            READ(L->v_W_A, ds * ds * sizeof(float), fp);
            READ(L->m_b_A, ds * sizeof(float), fp);
            READ(L->v_b_A, ds * sizeof(float), fp);
        }
    }

    READ_LARGE(m->shadow_lm_head, d * V * sizeof(float), fp);
    READ_LARGE(m->sampled_lm_head, d * V * sizeof(float), fp);
    READ(m->shadow_w_pred, d * d * sizeof(float), fp);
    READ(m->sampled_w_pred, d * d * sizeof(float), fp);
    READ(m->norm_w, d * sizeof(float), fp);
    READ(m->norm_b, d * sizeof(float), fp);
    READ(m->embed_E, V * de * sizeof(float), fp);
    READ(m->embed_P, de * d * sizeof(float), fp);
    READ(m->m_embed_E, V * de * sizeof(float), fp);
    READ(m->v_embed_E, V * de * sizeof(float), fp);
    READ(m->m_embed_P, de * d * sizeof(float), fp);
    READ(m->v_embed_P, de * d * sizeof(float), fp);
    READ(m->tau_mux, nl * sizeof(float), fp);

    /* === V3: Factored Adam moment buffers === */
    if (version >= 3U) {
        READ(m->m_lm_head_row, V12_K_HOT * sizeof(float), fp);
        READ(m->m_lm_head_col, d * sizeof(float), fp);
        READ(m->v_lm_head_row, V12_K_HOT * sizeof(float), fp);
        READ(m->v_lm_head_col, d * sizeof(float), fp);
        READ(m->m_w_pred_row, d * sizeof(float), fp);
        READ(m->m_w_pred_col, d * sizeof(float), fp);
        READ(m->v_w_pred_row, d * sizeof(float), fp);
        READ(m->v_w_pred_col, d * sizeof(float), fp);
        READ(m->m_norm_w, d * sizeof(float), fp);
        READ(m->v_norm_w, d * sizeof(float), fp);
        READ(m->m_norm_b, d * sizeof(float), fp);
        READ(m->v_norm_b, d * sizeof(float), fp);
    }

    /* Reset eval_gap on load — it's informational and should be recomputed */
    s->eval_gap = 0.0f;
    printf("[CKPT] WARNING: eval_gap reset to 0\n");

    fclose(fp);
    printf("[CKPT] Loaded step=%u from %s (format v%u%s)\n",
           s->step, path, version,
           version < V12_CHECKPOINT_VERSION ? " — moments zero-init" : "");
    return 0;

io_error:
    fprintf(stderr, "[CKPT] I/O error reading %s\n", path);
    fclose(fp);
    return -1;
}


/* =========================================================================
 * checkpoint_rebuild
 * ========================================================================= */

int checkpoint_rebuild(Model *m, TrainState *s) {
    if (!m || !s) {
        return -1;
    }

    uint64_t rng = s->rng_state;

    /* Sampled weights are loaded from checkpoint — do NOT resample.
     * Just rebuild CSR indices from the loaded sampled values. */

    /* Rebuild CSR indices for each layer */
    for (int k = 0; k < (int)m->n_layers; k++) {
        if (model_rebuild_layer_csr(&m->layers[k]) != 0) {
            fprintf(stderr, "[CKPT] CSR rebuild failed for layer %d\n", k);
            return -1;
        }
    }

    /* Rebuild CSR for lm_head */
    if (model_rebuild_lm_head_csr(m) != 0) {
        fprintf(stderr, "[CKPT] lm_head CSR rebuild failed\n");
        return -1;
    }

    /* Update crystallization masks from loaded shadow logits */
    for (int k = 0; k < (int)m->n_layers; k++) {
        update_crystallization(&m->layers[k], k, s, (int)s->step);
    }

    /* Sync RNG back to TrainState for deterministic continuation */
    s->rng_state = rng;

    printf("[CKPT] Rebuild complete: %u layers, crystal_fraction=%.3f\n",
           m->n_layers, s->crystal_fraction);
    return 0;
}
