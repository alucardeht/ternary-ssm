#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "core/model.h"
#include "core/ssm_scan.h"
#include "core/ternary_matmul.h"
#include "io/checkpoint.h"
#include "io/dataset.h"

#define R405_SCHEMA "r405_tcfvl_gate.v1"
#define R405_ANALYZE_SCORE "abs_shadow_plus_alt_gap_v1"
#define R405_DEFAULT_CONTEXTS 64
#define R405_MIN_CONTEXTS 64
#define R405_MAX_CONTEXTS 256
#define R405_DEFAULT_BATCH_SIZE 4
#define R405_DEFAULT_SEED 405ULL
#define R405_DEFAULT_CANDIDATES 8
#define R405_DEFAULT_TOP_K 10
#define R405_CONTEXT_TOKENS (V12_SEQ_LEN - 1)
#define R405_EPS 1.0e-12

typedef enum {
    FAMILY_W_IN = 0,
    FAMILY_W_OUT = 1,
    FAMILY_LM_HEAD = 2,
    FAMILY_W_PRED = 3
} CandidateFamily;

typedef struct {
    const char *dataset_path;
    const char *checkpoint_path;
    const char *output_path;
    int contexts;
    int batch_size;
    uint64_t seed;
    int candidates;
    int top_k;
    const char *ce_mode;
    int path_check_only;
    int target_aware_candidates;
    int per_context_deltas;
} GateConfig;

typedef struct {
    uint16_t tokens[V12_SEQ_LEN];
} ContextWindow;

typedef struct {
    float h[V12_D_STATE];
    float conv_buf[V12_CONV_KERNEL][V12_D_STATE];
    int conv_pos;
} LayerRuntimeState;

typedef struct {
    float prev_inputs[V12_N_LAYERS][V12_D_MODEL];
    int has_prev;
    LayerRuntimeState layers[V12_N_LAYERS];
} SequenceRuntimeState;

typedef struct {
    CandidateFamily family;
    int layer;
    size_t flat_index;
    int row;
    int col;
    int current_value;
    float abs_shadow;
} CandidateCell;

typedef struct {
    CandidateCell cell;
    int alternative_value;
    double delta_ce;
    double flipped_ce;
    double score;
} RankedRecord;

typedef struct {
    size_t supported_records;
    size_t excluded_w_pred;
    size_t forward_eval_count;
    int restore_failures;
} RunSummary;

static uint64_t r405_mix64(uint64_t x)
{
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

static uint64_t r405_rng_next(uint64_t *state)
{
    *state = r405_mix64(*state + 0x9E3779B97F4A7C15ULL);
    return *state;
}

static double r405_rng_unit(uint64_t *state)
{
    return (double)(r405_rng_next(state) >> 11) * (1.0 / (double)(1ULL << 53));
}

static int r405_mkdir_single(const char *path)
{
#ifdef _WIN32
    if (_mkdir(path) == 0 || errno == EEXIST) {
        return 0;
    }
#else
    if (mkdir(path, 0777) == 0 || errno == EEXIST) {
        return 0;
    }
#endif
    return -1;
}

static int r405_ensure_parent_dirs(const char *path)
{
    char scratch[1024];
    size_t len;
    size_t i;

    if (!path) {
        return -1;
    }
    len = strlen(path);
    if (len == 0 || len >= sizeof(scratch)) {
        return -1;
    }
    memcpy(scratch, path, len + 1);
    for (i = 0; i < len; ++i) {
        if (scratch[i] != '/' && scratch[i] != '\\') {
            continue;
        }
        if (i == 0) {
            continue;
        }
        if (i == 2 && scratch[1] == ':') {
            continue;
        }
        scratch[i] = '\0';
        if (r405_mkdir_single(scratch) != 0) {
            return -1;
        }
        scratch[i] = path[i];
    }
    return 0;
}

static const char *r405_family_name(CandidateFamily family)
{
    switch (family) {
        case FAMILY_W_IN: return "W_in";
        case FAMILY_W_OUT: return "W_out";
        case FAMILY_LM_HEAD: return "lm_head";
        case FAMILY_W_PRED: return "W_pred";
        default: return "unknown";
    }
}

static int r405_parse_i32(const char *text, int *out)
{
    char *end = NULL;
    long value;
    if (!text || !*text) {
        return 0;
    }
    value = strtol(text, &end, 10);
    if (!end || *end != '\0') {
        return 0;
    }
    *out = (int)value;
    return 1;
}

static int r405_parse_u64(const char *text, uint64_t *out)
{
    char *end = NULL;
    unsigned long long value;
    if (!text || !*text) {
        return 0;
    }
    value = strtoull(text, &end, 10);
    if (!end || *end != '\0') {
        return 0;
    }
    *out = (uint64_t)value;
    return 1;
}

static void r405_usage(const char *program)
{
    fprintf(stderr,
            "usage: %s <dataset.bin> [--checkpoint PATH] --output PATH "
            "[--contexts 64-256] [--batch-size <=4] [--seed N] "
            "[--candidates N] [--top-k N] [--ce-mode full] [--path-check-only] "
            "[--target-aware-candidates] [--per-context-deltas]\n",
            program);
}

static int r405_parse_args(int argc, char **argv, GateConfig *cfg)
{
    int i;

    memset(cfg, 0, sizeof(*cfg));
    cfg->contexts = R405_DEFAULT_CONTEXTS;
    cfg->batch_size = R405_DEFAULT_BATCH_SIZE;
    cfg->seed = R405_DEFAULT_SEED;
    cfg->candidates = R405_DEFAULT_CANDIDATES;
    cfg->top_k = R405_DEFAULT_TOP_K;
    cfg->ce_mode = "full";

    if (argc < 2) {
        return -1;
    }
    cfg->dataset_path = argv[1];

    for (i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--checkpoint") == 0 && i + 1 < argc) {
            cfg->checkpoint_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            cfg->output_path = argv[++i];
        } else if (strcmp(argv[i], "--contexts") == 0 && i + 1 < argc) {
            if (!r405_parse_i32(argv[++i], &cfg->contexts)) {
                return -1;
            }
        } else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
            if (!r405_parse_i32(argv[++i], &cfg->batch_size)) {
                return -1;
            }
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!r405_parse_u64(argv[++i], &cfg->seed)) {
                return -1;
            }
        } else if (strcmp(argv[i], "--candidates") == 0 && i + 1 < argc) {
            if (!r405_parse_i32(argv[++i], &cfg->candidates)) {
                return -1;
            }
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            if (!r405_parse_i32(argv[++i], &cfg->top_k)) {
                return -1;
            }
        } else if (strcmp(argv[i], "--ce-mode") == 0 && i + 1 < argc) {
            cfg->ce_mode = argv[++i];
        } else if (strcmp(argv[i], "--path-check-only") == 0) {
            cfg->path_check_only = 1;
        } else if (strcmp(argv[i], "--target-aware-candidates") == 0) {
            cfg->target_aware_candidates = 1;
        } else if (strcmp(argv[i], "--per-context-deltas") == 0) {
            cfg->per_context_deltas = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            return 1;
        } else {
            return -1;
        }
    }

    if (!cfg->output_path) {
        return -1;
    }
    if (strcmp(cfg->ce_mode, "full") != 0) {
        return -1;
    }
    if (cfg->contexts < R405_MIN_CONTEXTS || cfg->contexts > R405_MAX_CONTEXTS) {
        return -1;
    }
    if (cfg->batch_size <= 0 || cfg->batch_size > 4) {
        return -1;
    }
    if (cfg->candidates <= 0 || cfg->top_k <= 0) {
        return -1;
    }
    return 0;
}

static int r405_build_w_pred_csr(Model *m)
{
    int8_t *dense;
    size_t elems;
    size_t i;
    int rc;

    if (!m || !m->sampled_w_pred) {
        return -1;
    }
    elems = (size_t)V12_D_MODEL * V12_D_MODEL;
    dense = (int8_t *)malloc(elems);
    if (!dense) {
        return -1;
    }
    for (i = 0; i < elems; ++i) {
        dense[i] = (int8_t)(int)m->sampled_w_pred[i];
    }
    ternary_csr_free(&m->csr_w_pred);
    rc = ternary_csr_init(&m->csr_w_pred, dense, V12_D_MODEL, V12_D_MODEL);
    free(dense);
    return rc;
}

static void r405_embed_token(const Model *m, uint16_t token, float *out)
{
    const float *e = m->embed_E + (size_t)token * V12_D_EMBED;
    int j;
    for (j = 0; j < V12_D_MODEL; ++j) {
        float acc = 0.0f;
        int r;
        for (r = 0; r < V12_D_EMBED; ++r) {
            acc += e[r] * m->embed_P[r * V12_D_MODEL + j];
        }
        out[j] = acc;
    }
}

static void r405_predictive_subtract(const Model *m, const float *prev, float *dst)
{
    float pred[V12_D_MODEL];
    memset(pred, 0, sizeof(pred));
    ternary_matvec(&m->csr_w_pred, prev, pred);
    for (int j = 0; j < V12_D_MODEL; ++j) {
        dst[j] -= pred[j];
    }
}

static void r405_apply_rms_norm(const Model *m, const float *x, float *out)
{
    float sum_sq = 0.0f;
    for (int j = 0; j < V12_D_MODEL; ++j) {
        sum_sq += x[j] * x[j];
    }
    {
        float inv_rms = 1.0f / sqrtf(sum_sq / (float)V12_D_MODEL + 1.0e-5f);
        for (int j = 0; j < V12_D_MODEL; ++j) {
            out[j] = x[j] * inv_rms * m->norm_w[j];
            if (m->norm_b) {
                out[j] += m->norm_b[j];
            }
        }
    }
}

static double r405_full_ce_from_norm(const Model *m, const float *norm_out, uint16_t target)
{
    float max_logit = -FLT_MAX;
    float target_logit = 0.0f;
    double sum_exp = 0.0;

    for (int v = 0; v < V12_VOCAB_SIZE; ++v) {
        float logit = ternary_dot_row(&m->csr_lm_head, v, norm_out);
        if (logit > max_logit) {
            max_logit = logit;
        }
        if (v == (int)target) {
            target_logit = logit;
        }
    }
    for (int v = 0; v < V12_VOCAB_SIZE; ++v) {
        float logit = ternary_dot_row(&m->csr_lm_head, v, norm_out);
        sum_exp += exp((double)(logit - max_logit));
    }
    return -((double)target_logit - (double)max_logit) + log(sum_exp);
}

static void r405_forward_layer_token(const SSMLayer *layer,
                                     LayerRuntimeState *state,
                                     const float *residual_in,
                                     float *residual_out,
                                     float tau_k)
{
    float x_proj[V12_D_STATE];
    float x_gated[V12_D_STATE];
    float x_conv[V12_D_STATE];
    float y_ssm[V12_D_STATE];
    float out_buf[V12_D_MODEL];
    const float *c_bilinear = layer->sampled_C_bilinear ? layer->sampled_C_bilinear : layer->C_bilinear;

    memset(x_proj, 0, sizeof(x_proj));
    ternary_matvec(&layer->csr_in, residual_in, x_proj);

    for (int i = 0; i < V12_D_STATE; ++i) {
        float mask = 1.0f / (1.0f + expf(-V12_SOFT_BOUNDARY_SCALE * (fabsf(x_proj[i]) - tau_k)));
        x_gated[i] = x_proj[i] * mask;
        state->conv_buf[state->conv_pos % V12_CONV_KERNEL][i] = x_gated[i];
    }

    memset(x_conv, 0, sizeof(x_conv));
    for (int kk = 0; kk < V12_CONV_KERNEL; ++kk) {
        int src_slot = ((state->conv_pos - kk) % V12_CONV_KERNEL + V12_CONV_KERNEL) % V12_CONV_KERNEL;
        for (int i = 0; i < V12_D_STATE; ++i) {
            x_conv[i] += SHADOW_DECODE(layer->shadow_w_conv[kk * V12_D_STATE + i]) *
                         state->conv_buf[src_slot][i];
        }
    }
    state->conv_pos += 1;

    for (int i = 0; i < V12_D_STATE; ++i) {
        float b_gate = 1.0f / (1.0f + expf(-x_conv[i]));
        float a_pre = layer->b_A[i];
        float h_prev = state->h[i];
        float a_eff;

        for (int j = 0; j < V12_D_STATE; ++j) {
            a_pre += layer->W_A[i * V12_D_STATE + j] * x_conv[j];
        }
        a_eff = layer->A[i] * (1.0f / (1.0f + expf(-a_pre)));
        state->h[i] = a_eff * h_prev + (b_gate * layer->B[i]) * x_conv[i];
        if (c_bilinear) {
            state->h[i] += c_bilinear[i] * h_prev * x_conv[i];
        }
    }

    for (int i = 0; i < V12_D_STATE; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < V12_D_STATE; ++j) {
            acc += layer->C[i * V12_D_STATE + j] * state->h[j];
        }
        y_ssm[i] = acc;
    }
    wht_inplace(y_ssm, V12_D_STATE);

    memset(out_buf, 0, sizeof(out_buf));
    ternary_matvec(&layer->csr_out, y_ssm, out_buf);
    for (int j = 0; j < V12_D_MODEL; ++j) {
        residual_out[j] = residual_in[j] + out_buf[j];
    }
}

static double r405_evaluate_context_ce(const Model *m, const ContextWindow *ctx)
{
    SequenceRuntimeState runtime;
    float layer_inputs[V12_N_LAYERS][V12_D_MODEL];
    float residual[V12_D_MODEL];
    float residual_next[V12_D_MODEL];
    float norm_out[V12_D_MODEL];

    memset(&runtime, 0, sizeof(runtime));

    for (int t = 0; t < R405_CONTEXT_TOKENS; ++t) {
        r405_embed_token(m, ctx->tokens[t], residual);
        if (runtime.has_prev) {
            r405_predictive_subtract(m, runtime.prev_inputs[0], residual);
        }
        memcpy(layer_inputs[0], residual, sizeof(residual));

        for (int k = 0; k < V12_N_LAYERS; ++k) {
            r405_forward_layer_token(&m->layers[k],
                                     &runtime.layers[k],
                                     layer_inputs[k],
                                     residual_next,
                                     m->tau_mux[k]);
            if (k + 1 < V12_N_LAYERS) {
                if (runtime.has_prev) {
                    r405_predictive_subtract(m, runtime.prev_inputs[k + 1], residual_next);
                }
                memcpy(layer_inputs[k + 1], residual_next, sizeof(residual_next));
            }
        }

        for (int k = 0; k < V12_N_LAYERS; ++k) {
            memcpy(runtime.prev_inputs[k], layer_inputs[k], sizeof(runtime.prev_inputs[k]));
        }
        runtime.has_prev = 1;
    }

    r405_apply_rms_norm(m, residual_next, norm_out);
    return r405_full_ce_from_norm(m, norm_out, ctx->tokens[V12_SEQ_LEN - 1]);
}

static double r405_evaluate_mean_ce(const Model *m,
                                    const ContextWindow *contexts,
                                    int n_contexts,
                                    RunSummary *summary)
{
    double total = 0.0;
    for (int i = 0; i < n_contexts; ++i) {
        total += r405_evaluate_context_ce(m, &contexts[i]);
        if (summary) {
            summary->forward_eval_count += 1;
        }
    }
    return total / (double)n_contexts;
}

static double r405_score_record(const CandidateCell *cell, int alternative_value)
{
    double family_bias = 0.0;
    switch (cell->family) {
        case FAMILY_W_IN: family_bias = 0.03; break;
        case FAMILY_W_OUT: family_bias = 0.02; break;
        case FAMILY_LM_HEAD: family_bias = 0.01; break;
        default: family_bias = 0.0; break;
    }
    return (double)cell->abs_shadow
         + 0.10 * fabs((double)alternative_value - (double)cell->current_value)
         + family_bias
         + ((cell->layer >= 0) ? 0.001 * (double)cell->layer : 0.0);
}

static int r405_candidate_equals(const CandidateCell *a, const CandidateCell *b)
{
    return a->family == b->family && a->layer == b->layer && a->flat_index == b->flat_index;
}

static int r405_append_candidate(CandidateCell *cells, int count, const CandidateCell *cell)
{
    for (int i = 0; i < count; ++i) {
        if (r405_candidate_equals(&cells[i], cell)) {
            return count;
        }
    }
    cells[count] = *cell;
    return count + 1;
}

static int r405_append_target_aware_lm_head_candidate(const Model *m,
                                                      const ContextWindow *contexts,
                                                      int n_contexts,
                                                      int ordinal,
                                                      CandidateCell *cells,
                                                      int count)
{
    CandidateCell cell;
    int row;
    int col;
    size_t local;

    if (!m || !contexts || n_contexts <= 0) {
        return count;
    }

    memset(&cell, 0, sizeof(cell));
    cell.family = FAMILY_LM_HEAD;
    cell.layer = -1;

    if (ordinal == 0) {
        row = (int)contexts[0].tokens[V12_SEQ_LEN - 1];
        col = 0;
    } else {
        int ctx_index = ordinal % n_contexts;
        int col_cycle = ordinal / n_contexts;
        row = (int)contexts[ctx_index].tokens[V12_SEQ_LEN - 1];
        col = col_cycle % V12_D_MODEL;
    }

    local = (size_t)row * V12_D_MODEL + (size_t)col;
    cell.flat_index = local;
    cell.row = row;
    cell.col = col;
    cell.current_value = (int)m->sampled_lm_head[local];
    cell.abs_shadow = fabsf(SHADOW_DECODE(m->shadow_lm_head[local]));
    if (cell.current_value < -1 || cell.current_value > 1) {
        return count;
    }
    return r405_append_candidate(cells, count, &cell);
}

static int r405_sample_random_candidates(const Model *m, const GateConfig *cfg, CandidateCell **out_cells)
{
    CandidateCell *cells;
    int count = 0;
    int attempts = 0;
    int max_attempts = cfg->candidates * 32;
    uint64_t rng = cfg->seed ^ 0x405CA11ULL;

    cells = (CandidateCell *)calloc((size_t)cfg->candidates, sizeof(CandidateCell));
    if (!cells) {
        return -1;
    }

    while (count < cfg->candidates && attempts < max_attempts) {
        CandidateCell cell;
        CandidateFamily family = (CandidateFamily)(attempts % 3);
        uint64_t sample = r405_rng_next(&rng);
        memset(&cell, 0, sizeof(cell));
        cell.family = family;
        cell.layer = -1;

        if (family == FAMILY_W_IN) {
            size_t flat = (size_t)(sample % ((uint64_t)V12_N_LAYERS * V12_D_STATE * V12_D_MODEL));
            int layer = (int)(flat / ((size_t)V12_D_STATE * V12_D_MODEL));
            size_t local = flat % ((size_t)V12_D_STATE * V12_D_MODEL);
            const SSMLayer *l = &m->layers[layer];
            cell.layer = layer;
            cell.flat_index = local;
            cell.row = (int)(local / V12_D_MODEL);
            cell.col = (int)(local % V12_D_MODEL);
            cell.current_value = (int)l->sampled_w_in[local];
            cell.abs_shadow = fabsf(SHADOW_DECODE(l->shadow_w_in[local]));
        } else if (family == FAMILY_W_OUT) {
            size_t flat = (size_t)(sample % ((uint64_t)V12_N_LAYERS * V12_D_MODEL * V12_D_STATE));
            int layer = (int)(flat / ((size_t)V12_D_MODEL * V12_D_STATE));
            size_t local = flat % ((size_t)V12_D_MODEL * V12_D_STATE);
            const SSMLayer *l = &m->layers[layer];
            cell.layer = layer;
            cell.flat_index = local;
            cell.row = (int)(local / V12_D_STATE);
            cell.col = (int)(local % V12_D_STATE);
            cell.current_value = (int)l->sampled_w_out[local];
            cell.abs_shadow = fabsf(SHADOW_DECODE(l->shadow_w_out[local]));
        } else {
            size_t local = (size_t)(sample % ((uint64_t)V12_VOCAB_SIZE * V12_D_MODEL));
            cell.flat_index = local;
            cell.row = (int)(local / V12_D_MODEL);
            cell.col = (int)(local % V12_D_MODEL);
            cell.current_value = (int)m->sampled_lm_head[local];
            cell.abs_shadow = fabsf(SHADOW_DECODE(m->shadow_lm_head[local]));
        }

        if (cell.current_value < -1 || cell.current_value > 1) {
            attempts += 1;
            continue;
        }
        count = r405_append_candidate(cells, count, &cell);
        attempts += 1;
    }

    *out_cells = cells;
    return count;
}

static int r405_sample_target_aware_candidates(const Model *m,
                                               const GateConfig *cfg,
                                               const ContextWindow *contexts,
                                               int n_contexts,
                                               CandidateCell **out_cells)
{
    CandidateCell *cells;
    int count = 0;
    int ordinal = 0;
    int max_ordinals;

    cells = (CandidateCell *)calloc((size_t)cfg->candidates, sizeof(CandidateCell));
    if (!cells) {
        return -1;
    }

    max_ordinals = cfg->candidates * ((n_contexts > 0) ? n_contexts : 1) * 2;
    while (count < cfg->candidates && ordinal < max_ordinals) {
        count = r405_append_target_aware_lm_head_candidate(m, contexts, n_contexts, ordinal, cells, count);
        ordinal += 1;
    }

    *out_cells = cells;
    return count;
}

static int r405_sample_candidates(const Model *m,
                                  const GateConfig *cfg,
                                  const ContextWindow *contexts,
                                  int n_contexts,
                                  CandidateCell **out_cells)
{
    if (cfg->target_aware_candidates) {
        return r405_sample_target_aware_candidates(m, cfg, contexts, n_contexts, out_cells);
    }
    return r405_sample_random_candidates(m, cfg, out_cells);
}

static int r405_set_candidate_value(Model *m,
                                    const CandidateCell *cell,
                                    int new_value,
                                    int *restore_ok)
{
    *restore_ok = 0;
    if (cell->family == FAMILY_W_IN) {
        m->layers[cell->layer].sampled_w_in[cell->flat_index] = (float)new_value;
        if (model_rebuild_layer_csr(&m->layers[cell->layer]) != 0) {
            return -1;
        }
    } else if (cell->family == FAMILY_W_OUT) {
        m->layers[cell->layer].sampled_w_out[cell->flat_index] = (float)new_value;
        if (model_rebuild_layer_csr(&m->layers[cell->layer]) != 0) {
            return -1;
        }
    } else if (cell->family == FAMILY_LM_HEAD) {
        m->sampled_lm_head[cell->flat_index] = (float)new_value;
        if (model_rebuild_lm_head_csr(m) != 0) {
            return -1;
        }
    } else {
        return -1;
    }
    *restore_ok = 1;
    return 0;
}

static int r405_restore_candidate(Model *m, const CandidateCell *cell)
{
    int restore_ok = 0;
    return r405_set_candidate_value(m, cell, cell->current_value, &restore_ok);
}

static int r405_pick_path_check_value(int current_value)
{
    if (current_value >= 1) {
        return -1;
    }
    return 1;
}

static int r405_run_path_check(FILE *out,
                               Model *m,
                               const ContextWindow *ctx,
                               RunSummary *summary)
{
    CandidateCell cell;
    const int max_columns = (V12_D_MODEL < 32) ? V12_D_MODEL : 32;
    const int target_row = (int)ctx->tokens[V12_SEQ_LEN - 1];
    const double baseline_ce = r405_evaluate_context_ce(m, ctx);
    double max_abs_delta = 0.0;
    double hit_delta = 0.0;
    double hit_ce = baseline_ce;
    int hit_column = -1;
    int tested_columns = 0;

    if (summary) {
        summary->forward_eval_count += 1;
    }

    memset(&cell, 0, sizeof(cell));
    cell.family = FAMILY_LM_HEAD;
    cell.layer = -1;
    cell.row = target_row;

    for (int col = 0; col < max_columns; ++col) {
        int restore_ok = 0;
        int current_value;
        int alt_value;
        double flipped_ce;
        double delta_ce;
        size_t flat_index = (size_t)target_row * V12_D_MODEL + (size_t)col;

        cell.flat_index = flat_index;
        cell.col = col;
        current_value = (int)m->sampled_lm_head[flat_index];
        cell.current_value = current_value;
        alt_value = r405_pick_path_check_value(current_value);

        if (r405_set_candidate_value(m, &cell, alt_value, &restore_ok) != 0) {
            return -1;
        }

        flipped_ce = r405_evaluate_context_ce(m, ctx);
        delta_ce = baseline_ce - flipped_ce;
        tested_columns += 1;
        if (summary) {
            summary->forward_eval_count += 1;
        }
        if (fabs(delta_ce) > max_abs_delta) {
            max_abs_delta = fabs(delta_ce);
        }

        if (r405_restore_candidate(m, &cell) != 0) {
            if (summary) {
                summary->restore_failures += 1;
            }
            return -1;
        }

        if (fabs(delta_ce) > 1.0e-9) {
            hit_column = col;
            hit_delta = delta_ce;
            hit_ce = flipped_ce;
            break;
        }
    }

    if (hit_column >= 0) {
        if (fprintf(out,
                    "{\"kind\":\"path_check\",\"target_row\":%d,"
                    "\"tested_columns\":%d,\"max_abs_delta\":%.17g,"
                    "\"path_identifying\":true,\"baseline_ce\":%.17g,"
                    "\"hit_column\":%d,\"hit_delta_ce\":%.17g,\"hit_ce\":%.17g,"
                    "\"decision_reason\":\"lm_head_target_row_delta_detected\"}\n",
                    target_row,
                    tested_columns,
                    max_abs_delta,
                    baseline_ce,
                    hit_column,
                    hit_delta,
                    hit_ce) < 0) {
            return -1;
        }
    } else {
        if (fprintf(out,
                    "{\"kind\":\"path_check\",\"target_row\":%d,"
                    "\"tested_columns\":%d,\"max_abs_delta\":%.17g,"
                    "\"path_identifying\":false,\"baseline_ce\":%.17g,"
                    "\"hit_column\":null,\"hit_delta_ce\":0,\"hit_ce\":%.17g,"
                    "\"decision_reason\":\"invalid_probe_path_blocked_no_lm_head_target_delta\"}\n",
                    target_row,
                    tested_columns,
                    max_abs_delta,
                    baseline_ce,
                    hit_ce) < 0) {
            return -1;
        }
    }

    if (hit_column >= 0) {
        if (fprintf(out,
                    "{\"kind\":\"path_check_summary\",\"path_identifying\":true,"
                    "\"decision\":\"PATH_IDENTIFYING\",\"hit_column\":%d,"
                    "\"tested_columns\":%d}\n",
                    hit_column,
                    tested_columns) < 0) {
            return -1;
        }
    } else {
        if (fprintf(out,
                    "{\"kind\":\"path_check_summary\",\"path_identifying\":false,"
                    "\"decision\":\"INVALID_PROBE\",\"reason\":\"path_blocked\","
                    "\"tested_columns\":%d}\n",
                    tested_columns) < 0) {
            return -1;
        }
    }

    return 0;
}

static int r405_rank_compare(const void *lhs, const void *rhs)
{
    const RankedRecord *a = (const RankedRecord *)lhs;
    const RankedRecord *b = (const RankedRecord *)rhs;
    if (a->score < b->score) return 1;
    if (a->score > b->score) return -1;
    if (a->delta_ce < b->delta_ce) return 1;
    if (a->delta_ce > b->delta_ce) return -1;
    return 0;
}

static int r405_record_conflicts(const RankedRecord *a, const RankedRecord *b)
{
    return r405_candidate_equals(&a->cell, &b->cell);
}

static int r405_collect_unique_records(const RankedRecord *source,
                                       int source_count,
                                       int wanted,
                                       RankedRecord *selected)
{
    int selected_count = 0;
    for (int i = 0; i < source_count && selected_count < wanted; ++i) {
        int conflict = 0;
        for (int j = 0; j < selected_count; ++j) {
            if (r405_record_conflicts(&source[i], &selected[j])) {
                conflict = 1;
                break;
            }
        }
        if (!conflict) {
            selected[selected_count++] = source[i];
        }
    }
    return selected_count;
}

static int r405_collect_unique_random_records(const RankedRecord *source,
                                              int source_count,
                                              int wanted,
                                              uint64_t *rng_state,
                                              RankedRecord *selected)
{
    int selected_count = 0;
    int attempts = 0;
    int max_attempts = source_count * 16;

    while (selected_count < wanted && attempts < max_attempts) {
        int idx = (int)(r405_rng_next(rng_state) % (uint64_t)source_count);
        int conflict = 0;
        for (int j = 0; j < selected_count; ++j) {
            if (r405_record_conflicts(&source[idx], &selected[j])) {
                conflict = 1;
                break;
            }
        }
        if (!conflict) {
            selected[selected_count++] = source[idx];
        }
        attempts += 1;
    }

    if (selected_count < wanted) {
        selected_count = r405_collect_unique_records(source, source_count, wanted, selected);
    }
    return selected_count;
}

static int r405_apply_joint_selection(Model *m, const RankedRecord *records, int n_records)
{
    for (int i = 0; i < n_records; ++i) {
        int restore_ok = 0;
        if (r405_set_candidate_value(m, &records[i].cell, records[i].alternative_value, &restore_ok) != 0) {
            return -1;
        }
    }
    return 0;
}

static void r405_restore_joint_selection(Model *m, const RankedRecord *records, int n_records)
{
    for (int i = 0; i < n_records; ++i) {
        (void)r405_restore_candidate(m, &records[i].cell);
    }
}

static void r405_collect_contexts(Dataset *ds, const GateConfig *cfg, ContextWindow *contexts)
{
    dataset_seed(ds, cfg->seed);
    for (int i = 0; i < cfg->contexts; ++i) {
        dataset_sample_batch(ds, contexts[i].tokens, 1, V12_SEQ_LEN);
    }
}

static int r405_write_meta_fixed(FILE *out, const GateConfig *cfg, int checkpoint_loaded)
{
    if (cfg->checkpoint_path) {
        return fprintf(out,
                       "{\"kind\":\"meta\",\"schema\":\"%s\",\"activation_flag\":true,"
                       "\"activation_line_emitted\":true,\"source\":\"v12-cpu-forward\","
                       "\"ce_mode\":\"full\",\"batch_size\":%d,\"contexts\":%d,"
                       "\"candidate_count\":%d,\"top_k\":%d,\"seed\":%llu,"
                       "\"path_check_only\":%s,"
                       "\"target_aware_candidates\":%s,"
                       "\"per_context_deltas\":%s,"
                       "\"checkpoint_loaded\":%s,\"checkpoint_path\":\"%s\","
                       "\"no_training_integration\":true,\"score_name\":\"%s\"}\n",
                       R405_SCHEMA,
                       cfg->batch_size,
                       cfg->contexts,
                       cfg->candidates,
                       cfg->top_k,
                       (unsigned long long)cfg->seed,
                       cfg->path_check_only ? "true" : "false",
                       cfg->target_aware_candidates ? "true" : "false",
                       cfg->per_context_deltas ? "true" : "false",
                       checkpoint_loaded ? "true" : "false",
                       cfg->checkpoint_path,
                       R405_ANALYZE_SCORE) < 0 ? -1 : 0;
    }
    return fprintf(out,
                   "{\"kind\":\"meta\",\"schema\":\"%s\",\"activation_flag\":true,"
                   "\"activation_line_emitted\":true,\"source\":\"v12-cpu-forward\","
                   "\"ce_mode\":\"full\",\"batch_size\":%d,\"contexts\":%d,"
                    "\"candidate_count\":%d,\"top_k\":%d,\"seed\":%llu,"
                    "\"path_check_only\":%s,"
                    "\"target_aware_candidates\":%s,"
                    "\"per_context_deltas\":%s,"
                    "\"checkpoint_loaded\":%s,\"checkpoint_path\":null,"
                   "\"no_training_integration\":true,\"score_name\":\"%s\"}\n",
                   R405_SCHEMA,
                   cfg->batch_size,
                   cfg->contexts,
                   cfg->candidates,
                   cfg->top_k,
                    (unsigned long long)cfg->seed,
                    cfg->path_check_only ? "true" : "false",
                    cfg->target_aware_candidates ? "true" : "false",
                    cfg->per_context_deltas ? "true" : "false",
                    checkpoint_loaded ? "true" : "false",
                   R405_ANALYZE_SCORE) < 0 ? -1 : 0;
}

int main(int argc, char **argv)
{
    GateConfig cfg;
    Dataset *ds = NULL;
    Model *m = NULL;
    TrainState state;
    ContextWindow *contexts = NULL;
    CandidateCell *cells = NULL;
    RankedRecord *ranked = NULL;
    double *baseline_context_ce = NULL;
    FILE *out = NULL;
    RunSummary summary;
    double baseline_ce;
    int checkpoint_loaded = 0;
    int cell_count;
    int ranked_count = 0;
    int rc;

    memset(&summary, 0, sizeof(summary));
    memset(&state, 0, sizeof(state));

    rc = r405_parse_args(argc, argv, &cfg);
    if (rc == 1) {
        r405_usage(argv[0]);
        return 0;
    }
    if (rc != 0) {
        r405_usage(argv[0]);
        return 2;
    }
    if (cfg.batch_size > 4) {
        fprintf(stderr, "[R405_TCFVL_GATE] invalid batch_size=%d\n", cfg.batch_size);
        return 2;
    }

    printf("[R405_TCFVL_GATE] active=1 mode=diagnostic-only source=v12-cpu-forward "
           "ce_mode=full contexts=%d batch_size=%d candidates=%d output_path=%s "
           "target_aware_candidates=%d no_training_integration=1\n",
           cfg.contexts,
           cfg.batch_size,
           cfg.candidates,
           cfg.output_path,
           cfg.target_aware_candidates);
    fflush(stdout);

    if (r405_ensure_parent_dirs(cfg.output_path) != 0) {
        fprintf(stderr, "[R405_TCFVL_GATE] failed output_path=%s errno=%d\n", cfg.output_path, errno);
        return 1;
    }

    ds = dataset_load(cfg.dataset_path);
    if (!ds) {
        fprintf(stderr, "[R405_TCFVL_GATE] failed dataset=%s\n", cfg.dataset_path);
        return 1;
    }

    m = model_alloc();
    if (!m) {
        fprintf(stderr, "[R405_TCFVL_GATE] model_alloc failed\n");
        dataset_free(ds);
        return 1;
    }

    if (model_initialize_lm_head_host(m) != 0) {
        fprintf(stderr, "[R405_TCFVL_GATE] model_initialize_lm_head_host failed\n");
        model_free(m);
        dataset_free(ds);
        return 1;
    }

    if (cfg.checkpoint_path) {
        if (checkpoint_load(cfg.checkpoint_path, m, &state) != 0) {
            fprintf(stderr, "[R405_TCFVL_GATE] checkpoint_load failed path=%s\n", cfg.checkpoint_path);
            model_free(m);
            dataset_free(ds);
            return 1;
        }
        if (checkpoint_rebuild(m, &state) != 0) {
            fprintf(stderr, "[R405_TCFVL_GATE] checkpoint_rebuild failed path=%s\n", cfg.checkpoint_path);
            model_free(m);
            dataset_free(ds);
            return 1;
        }
        checkpoint_loaded = 1;
    }

    if (r405_build_w_pred_csr(m) != 0) {
        fprintf(stderr, "[R405_TCFVL_GATE] csr_w_pred rebuild failed\n");
        model_free(m);
        dataset_free(ds);
        return 1;
    }

    contexts = (ContextWindow *)calloc((size_t)cfg.contexts, sizeof(ContextWindow));
    if (!contexts) {
        fprintf(stderr, "[R405_TCFVL_GATE] contexts allocation failed\n");
        model_free(m);
        dataset_free(ds);
        return 1;
    }
    r405_collect_contexts(ds, &cfg, contexts);

    out = fopen(cfg.output_path, "wb");
    if (!out) {
        fprintf(stderr, "[R405_TCFVL_GATE] failed output_path=%s errno=%d\n", cfg.output_path, errno);
        free(contexts);
        model_free(m);
        dataset_free(ds);
        return 1;
    }
    if (r405_write_meta_fixed(out, &cfg, checkpoint_loaded) != 0) {
        fprintf(stderr, "[R405_TCFVL_GATE] failed writing meta\n");
        fclose(out);
        free(contexts);
        model_free(m);
        dataset_free(ds);
        return 1;
    }
    fflush(out);

    if (cfg.path_check_only) {
        printf("[R405_TCFVL_GATE] path_check_only=1 scope=lm_head_target_row column_scan_limit=32\n");
        fflush(stdout);
        if (r405_run_path_check(out, m, &contexts[0], &summary) != 0) {
            fprintf(stderr, "[R405_TCFVL_GATE] path_check_only failed\n");
            fclose(out);
            free(contexts);
            model_free(m);
            dataset_free(ds);
            return 1;
        }
        fflush(out);
        fclose(out);
        free(contexts);
        model_free(m);
        dataset_free(ds);
        return 0;
    }

    if (cfg.per_context_deltas) {
        baseline_context_ce = (double *)calloc((size_t)cfg.contexts, sizeof(double));
        if (!baseline_context_ce) {
            fprintf(stderr, "[R405_TCFVL_GATE] baseline_context_ce allocation failed\n");
            fclose(out);
            free(contexts);
            model_free(m);
            dataset_free(ds);
            return 1;
        }
        baseline_ce = 0.0;
        for (int c = 0; c < cfg.contexts; ++c) {
            baseline_context_ce[c] = r405_evaluate_context_ce(m, &contexts[c]);
            summary.forward_eval_count += 1u;
            baseline_ce += baseline_context_ce[c];
        }
        baseline_ce /= (double)cfg.contexts;
    } else {
        baseline_ce = r405_evaluate_mean_ce(m, contexts, cfg.contexts, &summary);
    }

    cell_count = r405_sample_candidates(m, &cfg, contexts, cfg.contexts, &cells);
    if (cell_count < 0) {
        fprintf(stderr, "[R405_TCFVL_GATE] candidate allocation failed\n");
        fclose(out);
        free(baseline_context_ce);
        free(contexts);
        model_free(m);
        dataset_free(ds);
        return 1;
    }
    ranked = (RankedRecord *)calloc((size_t)cell_count * 2U, sizeof(RankedRecord));
    if (!ranked) {
        fprintf(stderr, "[R405_TCFVL_GATE] ranked allocation failed\n");
        free(cells);
        fclose(out);
        free(baseline_context_ce);
        free(contexts);
        model_free(m);
        dataset_free(ds);
        return 1;
    }

    summary.excluded_w_pred = 1;

    for (int i = 0; i < cell_count; ++i) {
        const CandidateCell *cell = &cells[i];
        for (int alt = -1; alt <= 1; ++alt) {
            int restore_ok = 0;
            double flipped_ce;
            double delta_ce;
            if (alt == cell->current_value) {
                continue;
            }
            if (r405_set_candidate_value(m, cell, alt, &restore_ok) != 0) {
                summary.restore_failures += 1;
                continue;
            }
            if (cfg.per_context_deltas) {
                flipped_ce = 0.0;
                for (int c = 0; c < cfg.contexts; ++c) {
                    double context_flipped_ce = r405_evaluate_context_ce(m, &contexts[c]);
                    double context_delta = baseline_context_ce[c] - context_flipped_ce;
                    summary.forward_eval_count += 1u;
                    flipped_ce += context_flipped_ce;
                    if (fprintf(out,
                                "{\"kind\":\"context_delta\",\"family\":\"%s\",\"layer\":%d,"
                                "\"flat_index\":%llu,\"row\":%d,\"col\":%d,\"context\":%d,"
                                "\"current_value\":%d,\"alternative_value\":%d,"
                                "\"baseline_ce\":%.17g,\"flipped_ce\":%.17g,"
                                "\"delta_ce\":%.17g,\"abs_shadow\":%.9g,\"score\":%.9g}\n",
                                r405_family_name(cell->family),
                                cell->layer,
                                (unsigned long long)cell->flat_index,
                                cell->row,
                                cell->col,
                                c,
                                cell->current_value,
                                alt,
                                baseline_context_ce[c],
                                context_flipped_ce,
                                context_delta,
                                cell->abs_shadow,
                                r405_score_record(cell, alt)) < 0) {
                        fclose(out);
                        free(baseline_context_ce);
                        free(ranked);
                        free(cells);
                        free(contexts);
                        model_free(m);
                        dataset_free(ds);
                        return 1;
                    }
                }
                flipped_ce /= (double)cfg.contexts;
            } else {
                flipped_ce = r405_evaluate_mean_ce(m, contexts, cfg.contexts, &summary);
            }
            delta_ce = baseline_ce - flipped_ce;
            if (fprintf(out,
                        "{\"kind\":\"record\",\"family\":\"%s\",\"layer\":%d,"
                        "\"flat_index\":%llu,\"row\":%d,\"col\":%d,"
                        "\"current_value\":%d,\"alternative_value\":%d,"
                        "\"baseline_ce\":%.17g,\"flipped_ce\":%.17g,\"delta_ce\":%.17g,"
                        "\"abs_shadow\":%.9g,\"score\":%.9g,\"restore_ok\":true}\n",
                        r405_family_name(cell->family),
                        cell->layer,
                        (unsigned long long)cell->flat_index,
                        cell->row,
                        cell->col,
                        cell->current_value,
                        alt,
                        baseline_ce,
                        flipped_ce,
                        delta_ce,
                        cell->abs_shadow,
                        r405_score_record(cell, alt)) < 0) {
    fclose(out);
    free(ranked);
                free(cells);
                free(contexts);
                model_free(m);
                dataset_free(ds);
                return 1;
            }
            ranked[ranked_count].cell = *cell;
            ranked[ranked_count].alternative_value = alt;
            ranked[ranked_count].delta_ce = delta_ce;
            ranked[ranked_count].flipped_ce = flipped_ce;
            ranked[ranked_count].score = r405_score_record(cell, alt);
            ranked_count += 1;
            summary.supported_records += 1;
            fflush(out);
            if (r405_restore_candidate(m, cell) != 0) {
                summary.restore_failures += 1;
            }
        }
    }

    qsort(ranked, (size_t)ranked_count, sizeof(RankedRecord), r405_rank_compare);
    if (ranked_count > 0) {
        int wanted_top_k = cfg.top_k < ranked_count ? cfg.top_k : ranked_count;
        double indep_sum = 0.0;
        double joint_delta = 0.0;
        double random_sum = 0.0;
        double random_joint = 0.0;
        double interaction_harm = 0.0;
        RankedRecord *top_pick = NULL;
        RankedRecord *random_pick = NULL;
        uint64_t random_rng = cfg.seed ^ 0xBADC0FFEEULL;
        int top_k = 0;
        int random_k = 0;

        top_pick = (RankedRecord *)calloc((size_t)wanted_top_k, sizeof(RankedRecord));
        random_pick = (RankedRecord *)calloc((size_t)wanted_top_k, sizeof(RankedRecord));
        if (!top_pick || !random_pick) {
            free(top_pick);
            free(random_pick);
            fclose(out);
            free(baseline_context_ce);
            free(ranked);
            free(cells);
            free(contexts);
            model_free(m);
            dataset_free(ds);
            return 1;
        }

        top_k = r405_collect_unique_records(ranked, ranked_count, wanted_top_k, top_pick);
        for (int i = 0; i < top_k; ++i) {
            indep_sum += top_pick[i].delta_ce;
        }
        if (top_k > 0 && r405_apply_joint_selection(m, top_pick, top_k) == 0) {
            double joint_ce = r405_evaluate_mean_ce(m, contexts, cfg.contexts, &summary);
            joint_delta = baseline_ce - joint_ce;
            if (fabs(indep_sum) > R405_EPS) {
                interaction_harm = (indep_sum - joint_delta) / fabs(indep_sum);
            }
            r405_restore_joint_selection(m, top_pick, top_k);
        }

        random_k = r405_collect_unique_random_records(
            ranked, ranked_count, wanted_top_k, &random_rng, random_pick);
        for (int i = 0; i < random_k; ++i) {
            random_sum += random_pick[i].delta_ce;
        }
        if (random_k > 0 && r405_apply_joint_selection(m, random_pick, random_k) == 0) {
            double random_joint_ce = r405_evaluate_mean_ce(m, contexts, cfg.contexts, &summary);
            random_joint = baseline_ce - random_joint_ce;
            r405_restore_joint_selection(m, random_pick, random_k);
        }

        fprintf(out,
                "{\"kind\":\"interaction\",\"selection\":\"score_topk\","
                "\"set_id\":0,\"k\":%d,\"independent_delta_sum\":%.17g,"
                "\"joint_delta\":%.17g,\"interaction_harm\":%.17g}\n",
                top_k, indep_sum, joint_delta, interaction_harm);
        fflush(out);
        fprintf(out,
                "{\"kind\":\"interaction\",\"selection\":\"random_baseline\","
                "\"set_id\":1,\"k\":%d,\"independent_delta_sum\":%.17g,"
                "\"joint_delta\":%.17g,\"interaction_harm\":%.17g}\n",
                random_k, random_sum, random_joint,
                (fabs(random_sum) > R405_EPS) ? ((random_sum - random_joint) / fabs(random_sum)) : 0.0);
        fflush(out);
        free(top_pick);
        free(random_pick);
    }

    fprintf(out,
            "{\"kind\":\"summary\",\"baseline_ce\":%.17g,"
            "\"supported_records\":%llu,\"excluded_w_pred\":%llu,"
            "\"restore_failures\":%d,\"forward_eval_count\":%llu,"
            "\"supported_families\":[\"W_in\",\"W_out\",\"lm_head\"],"
            "\"excluded_families\":[\"W_pred\"]}\n",
            baseline_ce,
            (unsigned long long)summary.supported_records,
            (unsigned long long)summary.excluded_w_pred,
            summary.restore_failures,
            (unsigned long long)summary.forward_eval_count);
    fflush(out);

    fclose(out);
    free(ranked);
    free(cells);
    free(baseline_context_ce);
    free(contexts);
    model_free(m);
    dataset_free(ds);
    return 0;
}
