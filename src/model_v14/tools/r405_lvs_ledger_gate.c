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

#define R405_LVS_SCHEMA "r405_lvs_ledger_gate.v1"
#define R405_LVS_DEFAULT_CONTEXTS 128
#define R405_LVS_DEFAULT_PREFIX_TOKENS 256
#define R405_LVS_DEFAULT_HORIZON 8
#define R405_LVS_DEFAULT_GROUPS 32
#define R405_LVS_DEFAULT_BLOCKS_PER_CONTEXT 4
#define R405_LVS_DEFAULT_SEED 40502ULL
#define R405_LVS_MAX_CONTEXTS 512
#define R405_LVS_MIN_PREFIX_TOKENS 8
#define R405_LVS_MAX_GROUPS 64
#define R405_LVS_MIN_GROUPS 16

typedef struct {
    const char *dataset_path;
    const char *checkpoint_path;
    const char *output_path;
    int contexts;
    int prefix_tokens;
    int horizon;
    int groups;
    int blocks_per_context;
    uint64_t seed;
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
    int layer;
    int group;
    int start;
    int end;
    double block_norm;
    double recent_write_norm;
    double predictor_score;
    double baseline_loss;
    double erased_loss;
    double delta_ce;
} ErasureRecord;

static uint64_t mix64(uint64_t x)
{
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

static uint64_t rng_next(uint64_t *state)
{
    *state = mix64(*state + 0x9E3779B97F4A7C15ULL);
    return *state;
}

static int mkdir_single(const char *path)
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

static int ensure_parent_dirs(const char *path)
{
    char scratch[1024];
    size_t len = strlen(path);
    if (len >= sizeof(scratch)) {
        return -1;
    }
    memcpy(scratch, path, len + 1u);
    for (size_t i = 0; i < len; ++i) {
        if (scratch[i] == '\\') {
            scratch[i] = '/';
        }
    }
    for (size_t i = 1; i < len; ++i) {
        if (scratch[i] == '/') {
            scratch[i] = '\0';
            if (strlen(scratch) > 0u && mkdir_single(scratch) != 0) {
                return -1;
            }
            scratch[i] = '/';
        }
    }
    return 0;
}

static int parse_i32(const char *text, int *out)
{
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (!text || *text == '\0' || !end || *end != '\0') {
        return 0;
    }
    if (value < INT32_MIN || value > INT32_MAX) {
        return 0;
    }
    *out = (int)value;
    return 1;
}

static int parse_u64(const char *text, uint64_t *out)
{
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (!text || *text == '\0' || !end || *end != '\0') {
        return 0;
    }
    *out = (uint64_t)value;
    return 1;
}

static void usage(const char *program)
{
    fprintf(stderr,
            "usage: %s <dataset.bin> --output PATH [--checkpoint PATH] "
            "[--contexts 8-512] [--prefix-tokens N] [--horizon N] "
            "[--groups 16-64] [--blocks-per-context N] [--seed N]\n",
            program);
}

static int parse_args(int argc, char **argv, GateConfig *cfg)
{
    memset(cfg, 0, sizeof(*cfg));
    cfg->contexts = R405_LVS_DEFAULT_CONTEXTS;
    cfg->prefix_tokens = R405_LVS_DEFAULT_PREFIX_TOKENS;
    cfg->horizon = R405_LVS_DEFAULT_HORIZON;
    cfg->groups = R405_LVS_DEFAULT_GROUPS;
    cfg->blocks_per_context = R405_LVS_DEFAULT_BLOCKS_PER_CONTEXT;
    cfg->seed = R405_LVS_DEFAULT_SEED;
    if (argc < 2) {
        return -1;
    }
    cfg->dataset_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            cfg->output_path = argv[++i];
        } else if (strcmp(argv[i], "--checkpoint") == 0 && i + 1 < argc) {
            cfg->checkpoint_path = argv[++i];
        } else if (strcmp(argv[i], "--contexts") == 0 && i + 1 < argc) {
            if (!parse_i32(argv[++i], &cfg->contexts)) return -1;
        } else if (strcmp(argv[i], "--prefix-tokens") == 0 && i + 1 < argc) {
            if (!parse_i32(argv[++i], &cfg->prefix_tokens)) return -1;
        } else if (strcmp(argv[i], "--horizon") == 0 && i + 1 < argc) {
            if (!parse_i32(argv[++i], &cfg->horizon)) return -1;
        } else if (strcmp(argv[i], "--groups") == 0 && i + 1 < argc) {
            if (!parse_i32(argv[++i], &cfg->groups)) return -1;
        } else if (strcmp(argv[i], "--blocks-per-context") == 0 && i + 1 < argc) {
            if (!parse_i32(argv[++i], &cfg->blocks_per_context)) return -1;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!parse_u64(argv[++i], &cfg->seed)) return -1;
        } else if (strcmp(argv[i], "--help") == 0) {
            return 1;
        } else {
            return -1;
        }
    }
    if (!cfg->output_path) {
        return -1;
    }
    if (cfg->contexts <= 0 || cfg->contexts > R405_LVS_MAX_CONTEXTS) {
        return -1;
    }
    if (cfg->prefix_tokens < R405_LVS_MIN_PREFIX_TOKENS ||
        cfg->prefix_tokens + cfg->horizon + 1 >= V12_SEQ_LEN) {
        return -1;
    }
    if (cfg->horizon <= 0 || cfg->horizon > 64) {
        return -1;
    }
    if (cfg->groups < R405_LVS_MIN_GROUPS || cfg->groups > R405_LVS_MAX_GROUPS ||
        cfg->groups > V12_D_STATE) {
        return -1;
    }
    if (cfg->blocks_per_context <= 0 || cfg->blocks_per_context > 8) {
        return -1;
    }
    return 0;
}

static int build_w_pred_csr(Model *m)
{
    int8_t *dense;
    size_t elems;
    int rc;

    if (!m || !m->sampled_w_pred) {
        return -1;
    }
    elems = (size_t)V12_D_MODEL * V12_D_MODEL;
    dense = (int8_t *)malloc(elems);
    if (!dense) {
        return -1;
    }
    for (size_t i = 0; i < elems; ++i) {
        dense[i] = (int8_t)(int)m->sampled_w_pred[i];
    }
    ternary_csr_free(&m->csr_w_pred);
    rc = ternary_csr_init(&m->csr_w_pred, dense, V12_D_MODEL, V12_D_MODEL);
    free(dense);
    return rc;
}

static void embed_token(const Model *m, uint16_t token, float *out)
{
    const float *e = m->embed_E + (size_t)token * V12_D_EMBED;
    for (int j = 0; j < V12_D_MODEL; ++j) {
        float acc = 0.0f;
        for (int r = 0; r < V12_D_EMBED; ++r) {
            acc += e[r] * m->embed_P[r * V12_D_MODEL + j];
        }
        out[j] = acc;
    }
}

static void predictive_subtract(const Model *m, const float *prev, float *dst)
{
    float pred[V12_D_MODEL];
    memset(pred, 0, sizeof(pred));
    ternary_matvec(&m->csr_w_pred, prev, pred);
    for (int j = 0; j < V12_D_MODEL; ++j) {
        dst[j] -= pred[j];
    }
}

static void apply_rms_norm(const Model *m, const float *x, float *out)
{
    float sum_sq = 0.0f;
    for (int j = 0; j < V12_D_MODEL; ++j) {
        sum_sq += x[j] * x[j];
    }
    float inv_rms = 1.0f / sqrtf(sum_sq / (float)V12_D_MODEL + 1.0e-5f);
    for (int j = 0; j < V12_D_MODEL; ++j) {
        out[j] = x[j] * inv_rms * m->norm_w[j];
        if (m->norm_b) {
            out[j] += m->norm_b[j];
        }
    }
}

static double full_ce_from_norm(const Model *m, const float *norm_out, uint16_t target)
{
    float max_logit = -FLT_MAX;
    float target_logit = 0.0f;
    double sum_exp = 0.0;
    for (int v = 0; v < V12_VOCAB_SIZE; ++v) {
        float logit = ternary_dot_row(&m->csr_lm_head, v, norm_out);
        if (logit > max_logit) max_logit = logit;
        if (v == (int)target) target_logit = logit;
    }
    for (int v = 0; v < V12_VOCAB_SIZE; ++v) {
        float logit = ternary_dot_row(&m->csr_lm_head, v, norm_out);
        sum_exp += exp((double)(logit - max_logit));
    }
    return -((double)target_logit - (double)max_logit) + log(sum_exp);
}

static void forward_layer_token(const SSMLayer *layer,
                                LayerRuntimeState *state,
                                const float *residual_in,
                                float *residual_out,
                                float tau_k,
                                float *h_delta_abs_or_null)
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
        for (int j = 0; j < V12_D_STATE; ++j) {
            a_pre += layer->W_A[i * V12_D_STATE + j] * x_conv[j];
        }
        float a_eff = layer->A[i] * (1.0f / (1.0f + expf(-a_pre)));
        state->h[i] = a_eff * h_prev + (b_gate * layer->B[i]) * x_conv[i];
        if (c_bilinear) {
            state->h[i] += c_bilinear[i] * h_prev * x_conv[i];
        }
        if (h_delta_abs_or_null) {
            h_delta_abs_or_null[i] = fabsf(state->h[i] - h_prev);
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

static double forward_one_token(const Model *m,
                                SequenceRuntimeState *runtime,
                                uint16_t input_token,
                                uint16_t target_token,
                                float last_write_abs[V12_N_LAYERS][V12_D_STATE])
{
    float layer_inputs[V12_N_LAYERS][V12_D_MODEL];
    float residual[V12_D_MODEL];
    float residual_next[V12_D_MODEL];
    float norm_out[V12_D_MODEL];

    embed_token(m, input_token, residual);
    if (runtime->has_prev) {
        predictive_subtract(m, runtime->prev_inputs[0], residual);
    }
    memcpy(layer_inputs[0], residual, sizeof(residual));

    for (int k = 0; k < V12_N_LAYERS; ++k) {
        float *delta = last_write_abs ? last_write_abs[k] : NULL;
        forward_layer_token(&m->layers[k],
                            &runtime->layers[k],
                            layer_inputs[k],
                            residual_next,
                            m->tau_mux[k],
                            delta);
        if (k + 1 < V12_N_LAYERS) {
            if (runtime->has_prev) {
                predictive_subtract(m, runtime->prev_inputs[k + 1], residual_next);
            }
            memcpy(layer_inputs[k + 1], residual_next, sizeof(residual_next));
        }
    }

    for (int k = 0; k < V12_N_LAYERS; ++k) {
        memcpy(runtime->prev_inputs[k], layer_inputs[k], sizeof(runtime->prev_inputs[k]));
    }
    runtime->has_prev = 1;

    apply_rms_norm(m, residual_next, norm_out);
    return full_ce_from_norm(m, norm_out, target_token);
}

static double prepare_prefix(const Model *m,
                             const ContextWindow *ctx,
                             int prefix_tokens,
                             SequenceRuntimeState *runtime,
                             float last_write_abs[V12_N_LAYERS][V12_D_STATE])
{
    double local_ce = 0.0;
    memset(runtime, 0, sizeof(*runtime));
    memset(last_write_abs, 0, sizeof(float) * V12_N_LAYERS * V12_D_STATE);
    for (int t = 0; t < prefix_tokens; ++t) {
        float (*delta)[V12_D_STATE] = (t == prefix_tokens - 1) ? last_write_abs : NULL;
        local_ce = forward_one_token(m, runtime, ctx->tokens[t], ctx->tokens[t + 1], delta);
    }
    return local_ce;
}

static double evaluate_future_loss(const Model *m,
                                   const ContextWindow *ctx,
                                   int prefix_tokens,
                                   int horizon,
                                   const SequenceRuntimeState *start_state)
{
    SequenceRuntimeState runtime = *start_state;
    double loss = 0.0;
    for (int s = 0; s < horizon; ++s) {
        int input_idx = prefix_tokens + s;
        int target_idx = input_idx + 1;
        loss += forward_one_token(m, &runtime, ctx->tokens[input_idx], ctx->tokens[target_idx], NULL);
    }
    return loss;
}

static void group_bounds(int group, int groups, int *start, int *end)
{
    int a = (group * V12_D_STATE) / groups;
    int b = ((group + 1) * V12_D_STATE) / groups;
    *start = a;
    *end = b;
}

static double block_norm(const float *values, int start, int end)
{
    double sum = 0.0;
    for (int i = start; i < end; ++i) {
        sum += (double)values[i] * (double)values[i];
    }
    return sqrt(sum);
}

static void erase_block(SequenceRuntimeState *runtime, int layer, int start, int end)
{
    for (int i = start; i < end; ++i) {
        runtime->layers[layer].h[i] = 0.0f;
    }
}

static int write_meta(FILE *out, const GateConfig *cfg, int checkpoint_loaded)
{
    const int smoke_scale = cfg->contexts < 128 ? 1 : 0;
    return fprintf(out,
                   "{\"kind\":\"meta\",\"schema\":\"%s\",\"activation_flag\":true,"
                   "\"activation_line_emitted\":true,\"source\":\"v12-cpu-forward-state-erasure\","
                   "\"ce_mode\":\"full\",\"contexts\":%d,\"prefix_tokens\":%d,"
                   "\"horizon\":%d,\"groups\":%d,\"blocks_per_context\":%d,"
                   "\"seed\":%llu,\"checkpoint_loaded\":%s,\"checkpoint_path\":",
                   R405_LVS_SCHEMA,
                   cfg->contexts,
                   cfg->prefix_tokens,
                   cfg->horizon,
                   cfg->groups,
                   cfg->blocks_per_context,
                   (unsigned long long)cfg->seed,
                   checkpoint_loaded ? "true" : "false") < 0
        ? -1
        : fprintf(out,
                  cfg->checkpoint_path ? "\"%s\"" : "null",
                  cfg->checkpoint_path ? cfg->checkpoint_path : "") < 0
              ? -1
              : fprintf(out,
                        ",\"no_training_integration\":true,\"diagnostic_only\":true,"
                        "\"smoke_scale\":%s}\n",
                        smoke_scale ? "true" : "false") < 0 ? -1 : 0;
}

static void collect_contexts(Dataset *ds, const GateConfig *cfg, ContextWindow *contexts)
{
    dataset_seed(ds, cfg->seed);
    for (int i = 0; i < cfg->contexts; ++i) {
        dataset_sample_batch(ds, contexts[i].tokens, 1, V12_SEQ_LEN);
    }
}

int main(int argc, char **argv)
{
    GateConfig cfg;
    Dataset *ds = NULL;
    Model *m = NULL;
    TrainState state;
    ContextWindow *contexts = NULL;
    FILE *out = NULL;
    int checkpoint_loaded = 0;
    int rc = parse_args(argc, argv, &cfg);
    if (rc == 1) {
        usage(argv[0]);
        return 0;
    }
    if (rc != 0) {
        usage(argv[0]);
        return 2;
    }

    printf("[R405_LVS_LEDGER_GATE] active=1 mode=diagnostic-only source=v12-cpu-forward-state-erasure "
           "ce_mode=full contexts=%d prefix_tokens=%d horizon=%d groups=%d blocks_per_context=%d "
           "output_path=%s no_training_integration=1\n",
           cfg.contexts,
           cfg.prefix_tokens,
           cfg.horizon,
           cfg.groups,
           cfg.blocks_per_context,
           cfg.output_path);
    fflush(stdout);

    if (ensure_parent_dirs(cfg.output_path) != 0) {
        fprintf(stderr, "[R405_LVS_LEDGER_GATE] failed output_path=%s errno=%d\n", cfg.output_path, errno);
        return 1;
    }

    ds = dataset_load(cfg.dataset_path);
    if (!ds) {
        fprintf(stderr, "[R405_LVS_LEDGER_GATE] failed dataset=%s\n", cfg.dataset_path);
        return 1;
    }

    m = model_alloc();
    if (!m) {
        fprintf(stderr, "[R405_LVS_LEDGER_GATE] model_alloc failed\n");
        dataset_free(ds);
        return 1;
    }
    if (model_initialize_lm_head_host(m) != 0) {
        fprintf(stderr, "[R405_LVS_LEDGER_GATE] model_initialize_lm_head_host failed\n");
        model_free(m);
        dataset_free(ds);
        return 1;
    }

    memset(&state, 0, sizeof(state));
    if (cfg.checkpoint_path) {
        if (checkpoint_load(cfg.checkpoint_path, m, &state) != 0 ||
            checkpoint_rebuild(m, &state) != 0) {
            fprintf(stderr, "[R405_LVS_LEDGER_GATE] checkpoint load/rebuild failed path=%s\n", cfg.checkpoint_path);
            model_free(m);
            dataset_free(ds);
            return 1;
        }
        checkpoint_loaded = 1;
    }
    if (build_w_pred_csr(m) != 0) {
        fprintf(stderr, "[R405_LVS_LEDGER_GATE] csr_w_pred rebuild failed\n");
        model_free(m);
        dataset_free(ds);
        return 1;
    }

    contexts = (ContextWindow *)calloc((size_t)cfg.contexts, sizeof(ContextWindow));
    if (!contexts) {
        fprintf(stderr, "[R405_LVS_LEDGER_GATE] contexts allocation failed\n");
        model_free(m);
        dataset_free(ds);
        return 1;
    }
    collect_contexts(ds, &cfg, contexts);

    out = fopen(cfg.output_path, "wb");
    if (!out) {
        fprintf(stderr, "[R405_LVS_LEDGER_GATE] failed output_path=%s errno=%d\n", cfg.output_path, errno);
        free(contexts);
        model_free(m);
        dataset_free(ds);
        return 1;
    }
    if (write_meta(out, &cfg, checkpoint_loaded) != 0) {
        fprintf(stderr, "[R405_LVS_LEDGER_GATE] failed writing meta\n");
        fclose(out);
        free(contexts);
        model_free(m);
        dataset_free(ds);
        return 1;
    }

    uint64_t rng = cfg.seed ^ 0x4C56535F47415445ULL;
    size_t record_count = 0;
    size_t pair_count = 0;
    size_t forward_eval_count = 0;
    double baseline_total = 0.0;

    for (int c = 0; c < cfg.contexts; ++c) {
        SequenceRuntimeState base_state;
        float last_write_abs[V12_N_LAYERS][V12_D_STATE];
        ErasureRecord records[8];
        int n_records = 0;
        double local_ce = prepare_prefix(m, &contexts[c], cfg.prefix_tokens, &base_state, last_write_abs);
        double baseline_loss = evaluate_future_loss(m, &contexts[c], cfg.prefix_tokens, cfg.horizon, &base_state);
        baseline_total += baseline_loss;
        forward_eval_count += (size_t)cfg.prefix_tokens + 1u;

        for (int b = 0; b < cfg.blocks_per_context; ++b) {
            int ordinal = (int)(rng_next(&rng) % (uint64_t)(V12_N_LAYERS * cfg.groups));
            int layer = ordinal / cfg.groups;
            int group = ordinal % cfg.groups;
            int start = 0;
            int end = 0;
            group_bounds(group, cfg.groups, &start, &end);
            SequenceRuntimeState erased_state = base_state;
            erase_block(&erased_state, layer, start, end);
            double erased_loss = evaluate_future_loss(m, &contexts[c], cfg.prefix_tokens, cfg.horizon, &erased_state);
            forward_eval_count += 1u;

            double h_norm = block_norm(base_state.layers[layer].h, start, end);
            double write_norm = block_norm(last_write_abs[layer], start, end);
            double predictor = h_norm + write_norm + 0.01 * local_ce;
            double delta = erased_loss - baseline_loss;
            records[n_records].layer = layer;
            records[n_records].group = group;
            records[n_records].start = start;
            records[n_records].end = end;
            records[n_records].block_norm = h_norm;
            records[n_records].recent_write_norm = write_norm;
            records[n_records].predictor_score = predictor;
            records[n_records].baseline_loss = baseline_loss;
            records[n_records].erased_loss = erased_loss;
            records[n_records].delta_ce = delta;
            n_records += 1;
            record_count += 1u;

            fprintf(out,
                    "{\"kind\":\"record\",\"context\":%d,\"layer\":%d,\"group\":%d,"
                    "\"start\":%d,\"end\":%d,\"local_ce\":%.12g,"
                    "\"block_norm\":%.12g,\"recent_write_norm\":%.12g,"
                    "\"predictor_score\":%.12g,\"baseline_loss\":%.12g,"
                    "\"erased_loss\":%.12g,\"delta_ce\":%.12g}\n",
                    c,
                    layer,
                    group,
                    start,
                    end,
                    local_ce,
                    h_norm,
                    write_norm,
                    predictor,
                    baseline_loss,
                    erased_loss,
                    delta);
        }

        if (n_records >= 2) {
            int first = -1;
            int second = -1;
            for (int i = 0; i < n_records; ++i) {
                if (first < 0 || records[i].predictor_score > records[first].predictor_score) {
                    second = first;
                    first = i;
                } else if (second < 0 || records[i].predictor_score > records[second].predictor_score) {
                    second = i;
                }
            }
            if (first >= 0 && second >= 0) {
                SequenceRuntimeState joint_state = base_state;
                erase_block(&joint_state, records[first].layer, records[first].start, records[first].end);
                erase_block(&joint_state, records[second].layer, records[second].start, records[second].end);
                double joint_loss = evaluate_future_loss(m, &contexts[c], cfg.prefix_tokens, cfg.horizon, &joint_state);
                double joint_delta = joint_loss - baseline_loss;
                double single_sum = records[first].delta_ce + records[second].delta_ce;
                double interaction_harm = (single_sum > 1.0e-12)
                    ? fmax(0.0, single_sum - joint_delta) / single_sum
                    : 0.0;
                forward_eval_count += 1u;
                pair_count += 1u;
                fprintf(out,
                        "{\"kind\":\"pair\",\"context\":%d,\"selection\":\"predictor_top2\","
                        "\"first_layer\":%d,\"first_group\":%d,\"second_layer\":%d,"
                        "\"second_group\":%d,\"single_delta_sum\":%.12g,"
                        "\"joint_delta\":%.12g,\"interaction_harm\":%.12g}\n",
                        c,
                        records[first].layer,
                        records[first].group,
                        records[second].layer,
                        records[second].group,
                        single_sum,
                        joint_delta,
                        interaction_harm);
            }
        }
    }

    fprintf(out,
            "{\"kind\":\"summary\",\"record_count\":%zu,\"pair_count\":%zu,"
            "\"forward_eval_count\":%zu,\"baseline_future_loss_mean\":%.12g}\n",
            record_count,
            pair_count,
            forward_eval_count,
            baseline_total / (double)cfg.contexts);
    fclose(out);
    free(contexts);
    model_free(m);
    dataset_free(ds);
    return 0;
}
