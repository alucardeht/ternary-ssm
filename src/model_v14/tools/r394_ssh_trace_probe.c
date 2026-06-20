#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/core/ssm_scan.h"
#include "../src/core/types.h"

#define R394_SSH_EPS 1.0e-6f
#define R394_MAX_RHOS 16
#define R394_MAX_HORIZONS 16

#define R394_TRACE_RHO_COUNT 4
#define R394_TRACE_SEQ_LEN 32
#define R394_TRACE_CKPT_COUNT (R394_TRACE_SEQ_LEN / V12_CKPT_STRIDE + 1)
#define R394_TRACE_TRAIN_SAMPLES 16
#define R394_TRACE_EVAL_SAMPLES 4
#define R394_TRACE_BATCH_SIZE 4
#define R394_TRACE_STAGE_STEPS 12
#define R394_TRACE_CHECKPOINT_INTERVAL 4
#define R394_TRACE_TOTAL_UPDATES (R394_TRACE_RHO_COUNT * R394_TRACE_STAGE_STEPS)
#define R394_TRACE_CHECKPOINTS (R394_TRACE_TOTAL_UPDATES / R394_TRACE_CHECKPOINT_INTERVAL)

#define R394_TRACE_LR 0.035f
#define R394_TRACE_B_SCALE 0.22f
#define R394_TRACE_SHADOW_CLAMP 3.0f
#define R394_TRACE_DEADZONE 0.18f
#define R394_TRACE_PROGRESS_FLOOR 5.0e-4f
#define R394_TRACE_ENDPOINT_TOL 2.0e-4f
#define R394_TRACE_BOUNDARY_LOSS_TOL 7.5e-4f
#define R394_TRACE_BOUNDARY_STATE_TOL 2.0e-2f
#define R394_TRACE_ALIGNMENT_FLOOR 0.92f

/*
 * The trace probe never supplies GP-KL moments to ssm_scan_forward_batch; that
 * hook stays unreachable in this standalone diagnostic target. Provide a local
 * no-op definition so the probe does not inherit the full loss/LM-head link
 * surface just to satisfy ssm_scan.c.
 */
void gp_kl_welford_update(GpKlMoments *m, const float *h, int d_state, int t) {
    (void)m;
    (void)h;
    (void)d_state;
    (void)t;
}

typedef struct {
    float rhos[R394_MAX_RHOS];
    int horizons[R394_MAX_HORIZONS];
    size_t n_rhos;
    size_t n_horizons;
} ProbeConfig;

typedef struct {
    float log_tau[V12_D_STATE];
    float mean_log_tau;
} LayerSpectrum;

typedef struct {
    float *train_x;
    float *train_target_y;
    float *eval_x;
    float *eval_target_y;
    float *projection_c;
    int8_t target_code[V12_D_STATE];
} TraceData;

typedef struct {
    float loss_start;
    float loss_end;
    float checkpoint_losses[R394_TRACE_CHECKPOINTS];
    float checkpoint_rhos[R394_TRACE_CHECKPOINTS];
    float checkpoint_state_jump[R394_TRACE_CHECKPOINTS];
    float checkpoint_alignment[R394_TRACE_CHECKPOINTS];
    int checkpoint_flips[R394_TRACE_CHECKPOINTS];
    int checkpoint_reversals[R394_TRACE_CHECKPOINTS];
    size_t checkpoint_count;
    int useful_flip_count;
    int useful_flip_reversals;
    float mode_alignment_min;
    float final_rho_boundary_loss_jump;
    float final_rho_boundary_state_jump;
    int final_rho_recovery_window;
    int final_rho_boundary_checkpoint;
    int finite_pass;
    char first_bad[64];
    char restart_provenance[96];
} TraceRun;

static void initialize_layer_a(float *a_values) {
    int i;

    for (i = 0; i < V12_D_STATE; ++i) {
        const float tau = 1.0f + (512.0f - 1.0f) * (float)i / (float)(V12_D_STATE - 1);
        a_values[i] = expf(-1.0f / tau);
    }
}

static float clamp_unit(float value) {
    if (value < R394_SSH_EPS) {
        return R394_SSH_EPS;
    }
    if (value > 1.0f - R394_SSH_EPS) {
        return 1.0f - R394_SSH_EPS;
    }
    return value;
}

static float clamp_symmetric(float value, float limit) {
    if (value < -limit) {
        return -limit;
    }
    if (value > limit) {
        return limit;
    }
    return value;
}

static int compare_float_asc(const void *lhs, const void *rhs) {
    const float a = *(const float *)lhs;
    const float b = *(const float *)rhs;
    if (a < b) {
        return -1;
    }
    if (a > b) {
        return 1;
    }
    return 0;
}

static int compare_int_asc(const void *lhs, const void *rhs) {
    const int a = *(const int *)lhs;
    const int b = *(const int *)rhs;
    if (a < b) {
        return -1;
    }
    if (a > b) {
        return 1;
    }
    return 0;
}

static float percentile_from_sorted(const float *values, size_t n, float q) {
    size_t lo;
    size_t hi;
    float pos;
    float frac;

    if (n == 0) {
        return R394_SSH_EPS;
    }
    if (n == 1) {
        return values[0];
    }

    if (q < 0.0f) {
        q = 0.0f;
    } else if (q > 1.0f) {
        q = 1.0f;
    }

    pos = q * (float)(n - 1);
    lo = (size_t)floorf(pos);
    hi = (size_t)ceilf(pos);
    frac = pos - (float)lo;

    if (hi >= n) {
        hi = n - 1;
    }
    if (lo == hi) {
        return values[lo];
    }
    return values[lo] + frac * (values[hi] - values[lo]);
}

static size_t dedupe_sorted_floats(float *values, size_t n) {
    size_t write_idx = 0;
    size_t i;

    if (n == 0) {
        return 0;
    }

    for (i = 0; i < n; ++i) {
        if (write_idx == 0 || fabsf(values[i] - values[write_idx - 1]) > 1.0e-6f) {
            values[write_idx++] = values[i];
        }
    }
    return write_idx;
}

static size_t dedupe_sorted_ints(int *values, size_t n) {
    size_t write_idx = 0;
    size_t i;

    if (n == 0) {
        return 0;
    }

    for (i = 0; i < n; ++i) {
        if (write_idx == 0 || values[i] != values[write_idx - 1]) {
            values[write_idx++] = values[i];
        }
    }
    return write_idx;
}

static size_t parse_float_list(const char *text, float *out, size_t cap) {
    size_t count = 0;
    char *copy;
    char *token;

    if (!text || !*text || cap == 0) {
        return 0;
    }

    copy = (char *)malloc(strlen(text) + 1);
    if (!copy) {
        return 0;
    }
    strcpy(copy, text);

    token = strtok(copy, ",");
    while (token && count < cap) {
        char *end = NULL;
        double parsed = strtod(token, &end);
        if (end != token) {
            out[count++] = (float)parsed;
        }
        token = strtok(NULL, ",");
    }

    free(copy);
    return count;
}

static size_t parse_int_list(const char *text, int *out, size_t cap) {
    size_t count = 0;
    char *copy;
    char *token;

    if (!text || !*text || cap == 0) {
        return 0;
    }

    copy = (char *)malloc(strlen(text) + 1);
    if (!copy) {
        return 0;
    }
    strcpy(copy, text);

    token = strtok(copy, ",");
    while (token && count < cap) {
        char *end = NULL;
        long parsed = strtol(token, &end, 10);
        if (end != token && parsed > 0L) {
            out[count++] = (int)parsed;
        }
        token = strtok(NULL, ",");
    }

    free(copy);
    return count;
}

static void build_config(ProbeConfig *config) {
    static const float default_rhos[] = {0.25f, 0.50f, 0.75f, 1.00f};
    static const int default_horizons[] = {64, 256, 512};
    const char *rho_env = getenv("R394_SSH_RHOS");
    const char *horizon_env = getenv("R394_SSH_HORIZONS");
    size_t i;

    memset(config, 0, sizeof(*config));

    for (i = 0; i < sizeof(default_rhos) / sizeof(default_rhos[0]); ++i) {
        config->rhos[config->n_rhos++] = default_rhos[i];
    }
    config->n_rhos += parse_float_list(
        rho_env,
        config->rhos + config->n_rhos,
        R394_MAX_RHOS - config->n_rhos
    );
    qsort(config->rhos, config->n_rhos, sizeof(config->rhos[0]), compare_float_asc);
    config->n_rhos = dedupe_sorted_floats(config->rhos, config->n_rhos);

    for (i = 0; i < sizeof(default_horizons) / sizeof(default_horizons[0]); ++i) {
        config->horizons[config->n_horizons++] = default_horizons[i];
    }
    config->n_horizons += parse_int_list(
        horizon_env,
        config->horizons + config->n_horizons,
        R394_MAX_HORIZONS - config->n_horizons
    );
    qsort(config->horizons, config->n_horizons, sizeof(config->horizons[0]), compare_int_asc);
    config->n_horizons = dedupe_sorted_ints(config->horizons, config->n_horizons);
}

static void compute_layer_spectrum(const float *a_values, LayerSpectrum *spectrum) {
    float sum = 0.0f;
    int i;

    for (i = 0; i < V12_D_STATE; ++i) {
        const float a = clamp_unit(a_values[i]);
        const float tau = -1.0f / logf(a);
        spectrum->log_tau[i] = logf(tau);
        sum += spectrum->log_tau[i];
    }

    spectrum->mean_log_tau = sum / (float)V12_D_STATE;
}

static void materialize_homotopy_a(const LayerSpectrum *spectrum, float rho, float *out_a) {
    int i;

    for (i = 0; i < V12_D_STATE; ++i) {
        const float log_tau_rho =
            spectrum->mean_log_tau + rho * (spectrum->log_tau[i] - spectrum->mean_log_tau);
        const float tau_rho = expf(log_tau_rho);
        out_a[i] = expf(-1.0f / tau_rho);
    }
}

static float compute_kappa_eff(const float *a_values, int horizon) {
    float gramian[V12_D_STATE];
    float gramian_active[V12_D_STATE];
    size_t n_active = 0;
    int i;

    for (i = 0; i < V12_D_STATE; ++i) {
        const float a = fabsf(clamp_unit(a_values[i]));
        const float numerator = 1.0f - powf(a, 2.0f * (float)horizon);
        const float denominator = 1.0f - a * a;
        float g = numerator / fmaxf(R394_SSH_EPS, denominator);
        if (g < R394_SSH_EPS) {
            g = R394_SSH_EPS;
        }
        gramian[i] = g;
        if (g > R394_SSH_EPS) {
            gramian_active[n_active++] = g;
        }
    }

    qsort(gramian, V12_D_STATE, sizeof(gramian[0]), compare_float_asc);
    qsort(gramian_active, n_active, sizeof(gramian_active[0]), compare_float_asc);

    return percentile_from_sorted(gramian, V12_D_STATE, 0.95f) /
           fmaxf(R394_SSH_EPS, percentile_from_sorted(gramian_active, n_active, 0.05f));
}

static const char *classify_metric(float r_kappa_best) {
    if (r_kappa_best <= 0.37f) {
        return "metric_strong_candidate";
    }
    if (r_kappa_best <= 0.55f) {
        return "metric_pass_candidate";
    }
    if (r_kappa_best <= 0.75f) {
        return "metric_weak";
    }
    return "metric_kill";
}

static uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static void project_shadow_to_code(const float *shadow, int8_t *code) {
    int i;

    for (i = 0; i < V12_D_STATE; ++i) {
        if (shadow[i] > R394_TRACE_DEADZONE) {
            code[i] = 1;
        } else if (shadow[i] < -R394_TRACE_DEADZONE) {
            code[i] = -1;
        } else {
            code[i] = 0;
        }
    }
}

static int count_code_flips(const int8_t *lhs, const int8_t *rhs) {
    int i;
    int flips = 0;

    for (i = 0; i < V12_D_STATE; ++i) {
        if (lhs[i] != rhs[i]) {
            ++flips;
        }
    }
    return flips;
}

static int count_code_reversals(const int8_t *older, const int8_t *prev, const int8_t *current) {
    int i;
    int reversals = 0;

    for (i = 0; i < V12_D_STATE; ++i) {
        if (current[i] != prev[i] && current[i] == older[i]) {
            ++reversals;
        }
    }
    return reversals;
}

static void mark_first_bad(TraceRun *trace, const char *field) {
    if (trace->first_bad[0] == '\0') {
        size_t n = strlen(field);
        if (n >= sizeof(trace->first_bad)) {
            n = sizeof(trace->first_bad) - 1;
        }
        memcpy(trace->first_bad, field, n);
        trace->first_bad[n] = '\0';
    }
    trace->finite_pass = 0;
}

static int all_finite_array(const float *values, size_t n) {
    size_t i;

    for (i = 0; i < n; ++i) {
        if (!isfinite(values[i])) {
            return 0;
        }
    }
    return 1;
}

static void fill_target_code(const LayerSpectrum *spectrum, int8_t *target_code) {
    int i;

    for (i = 0; i < V12_D_STATE; ++i) {
        const float centered = spectrum->log_tau[i] - spectrum->mean_log_tau;
        const float score =
            0.85f * sinf(1.9f * centered + 0.17f * (float)(i + 1)) +
            0.35f * cosf(0.11f * (float)(i + 3));
        if (score > 0.28f) {
            target_code[i] = 1;
        } else if (score < -0.28f) {
            target_code[i] = -1;
        } else {
            target_code[i] = 0;
        }
    }
}

static void build_projection_matrix(float *projection_c) {
    int row;
    int col;

    for (row = 0; row < V12_D_STATE; ++row) {
        for (col = 0; col < V12_D_STATE; ++col) {
            float value = 0.0f;
            if (row == col) {
                value += 1.0f;
            }
            value += 0.015f * sinf(0.13f * (float)(row + 1) * (float)(col + 1));
            value += 0.010f * cosf(0.07f * (float)(row + 3) + 0.19f * (float)(col + 5));
            projection_c[(size_t)row * V12_D_STATE + col] = value;
        }
    }
}

static void build_sequence_bank(float *x_bank, int sample_count) {
    int sample_idx;
    int t;
    int state_idx;

    for (sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        for (t = 0; t < R394_TRACE_SEQ_LEN; ++t) {
            for (state_idx = 0; state_idx < V12_D_STATE; ++state_idx) {
                const float phase = (float)(sample_idx + 1);
                const float time = (float)(t + 1);
                const float state = (float)(state_idx + 1);
                const float value =
                    0.70f * sinf(0.031f * time * state + 0.17f * phase) +
                    0.45f * cosf(0.049f * (time + phase) * (1.0f + 0.01f * state)) +
                    0.12f * sinf(0.009f * (phase + state) * (time + 3.0f));
                x_bank[((size_t)sample_idx * R394_TRACE_SEQ_LEN + (size_t)t) * V12_D_STATE +
                       (size_t)state_idx] = value;
            }
        }
    }
}

static void build_b_from_code(const int8_t *code, float *b_out) {
    int i;

    for (i = 0; i < V12_D_STATE; ++i) {
        b_out[i] = R394_TRACE_B_SCALE * (float)code[i];
    }
}

static void gather_sequences(
    const float *bank,
    const float *target_bank,
    const unsigned int *indices,
    float *x_batch,
    float *target_batch
) {
    size_t sample_stride = (size_t)R394_TRACE_SEQ_LEN * V12_D_STATE;
    int batch_idx;

    for (batch_idx = 0; batch_idx < R394_TRACE_BATCH_SIZE; ++batch_idx) {
        const unsigned int sample_idx = indices[batch_idx];
        memcpy(
            x_batch + (size_t)batch_idx * sample_stride,
            bank + (size_t)sample_idx * sample_stride,
            sample_stride * sizeof(float)
        );
        memcpy(
            target_batch + (size_t)batch_idx * sample_stride,
            target_bank + (size_t)sample_idx * sample_stride,
            sample_stride * sizeof(float)
        );
    }
}

static int run_ssm_batch(
    const float *a_values,
    const float *b_values,
    const float *projection_c,
    const float *x_batch,
    int batch_size,
    float *y_out,
    float *h_cache_out,
    float *h_final_out
) {
    float *h_state;
    size_t state_count = (size_t)batch_size * V12_D_STATE;
    int batch_idx;

    h_state = (float *)calloc(state_count, sizeof(float));
    if (!h_state) {
        return 0;
    }

    if (h_cache_out == NULL) {
        ssm_scan_forward_batch(
            a_values,
            NULL,
            NULL,
            b_values,
            0,
            projection_c,
            x_batch,
            h_state,
            y_out,
            NULL,
            NULL,
            NULL,
            batch_size,
            R394_TRACE_SEQ_LEN,
            V12_D_STATE,
            V12_D_STATE,
            NULL
        );
    } else {
        const size_t seq_stride = (size_t)R394_TRACE_SEQ_LEN * V12_D_STATE;
        const size_t ckpt_stride = (size_t)R394_TRACE_CKPT_COUNT * V12_D_STATE;

        for (batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            ssm_scan_forward(
                a_values,
                NULL,
                NULL,
                b_values,
                0,
                projection_c,
                x_batch + (size_t)batch_idx * seq_stride,
                h_state + (size_t)batch_idx * V12_D_STATE,
                y_out + (size_t)batch_idx * seq_stride,
                h_cache_out + (size_t)batch_idx * ckpt_stride,
                NULL,
                NULL,
                R394_TRACE_SEQ_LEN,
                V12_D_STATE,
                V12_D_STATE,
                NULL
            );
        }
    }

    memcpy(h_final_out, h_state, state_count * sizeof(float));
    free(h_state);
    return 1;
}

static int build_target_bank(
    const float *a_values,
    const TraceData *data,
    const float *x_bank,
    int sample_count,
    float *target_bank
) {
    float b_target[V12_D_STATE];
    float *y_tmp;
    float *h_final_tmp;
    size_t sample_stride = (size_t)R394_TRACE_SEQ_LEN * V12_D_STATE;
    int sample_idx;

    y_tmp = (float *)malloc(sample_stride * sizeof(float));
    h_final_tmp = (float *)malloc((size_t)V12_D_STATE * sizeof(float));
    if (!y_tmp || !h_final_tmp) {
        free(y_tmp);
        free(h_final_tmp);
        return 0;
    }

    build_b_from_code(data->target_code, b_target);

    for (sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        if (!run_ssm_batch(
                a_values,
                b_target,
                data->projection_c,
                x_bank + (size_t)sample_idx * sample_stride,
                1,
                y_tmp,
                NULL,
                h_final_tmp
            )) {
            free(y_tmp);
            free(h_final_tmp);
            return 0;
        }
        memcpy(
            target_bank + (size_t)sample_idx * sample_stride,
            y_tmp,
            sample_stride * sizeof(float)
        );
    }

    free(y_tmp);
    free(h_final_tmp);
    return 1;
}

static int build_trace_data(const float *rho1_a, const LayerSpectrum *spectrum, TraceData *data) {
    size_t train_bank_size = (size_t)R394_TRACE_TRAIN_SAMPLES * R394_TRACE_SEQ_LEN * V12_D_STATE;
    size_t eval_bank_size = (size_t)R394_TRACE_EVAL_SAMPLES * R394_TRACE_SEQ_LEN * V12_D_STATE;

    memset(data, 0, sizeof(*data));

    data->train_x = (float *)malloc(train_bank_size * sizeof(float));
    data->train_target_y = (float *)malloc(train_bank_size * sizeof(float));
    data->eval_x = (float *)malloc(eval_bank_size * sizeof(float));
    data->eval_target_y = (float *)malloc(eval_bank_size * sizeof(float));
    data->projection_c = (float *)malloc((size_t)V12_D_STATE * V12_D_STATE * sizeof(float));
    if (!data->train_x || !data->train_target_y || !data->eval_x || !data->eval_target_y ||
        !data->projection_c) {
        return 0;
    }

    fill_target_code(spectrum, data->target_code);
    build_projection_matrix(data->projection_c);
    build_sequence_bank(data->train_x, R394_TRACE_TRAIN_SAMPLES);
    build_sequence_bank(data->eval_x, R394_TRACE_EVAL_SAMPLES);

    if (!build_target_bank(rho1_a, data, data->train_x, R394_TRACE_TRAIN_SAMPLES, data->train_target_y)) {
        return 0;
    }
    if (!build_target_bank(rho1_a, data, data->eval_x, R394_TRACE_EVAL_SAMPLES, data->eval_target_y)) {
        return 0;
    }

    return 1;
}

static void free_trace_data(TraceData *data) {
    free(data->train_x);
    free(data->train_target_y);
    free(data->eval_x);
    free(data->eval_target_y);
    free(data->projection_c);
    memset(data, 0, sizeof(*data));
}

static void build_train_order(unsigned int *order) {
    uint64_t state = 0x3945A11CEULL;
    int update_idx;
    int batch_idx;

    for (update_idx = 0; update_idx < R394_TRACE_TOTAL_UPDATES; ++update_idx) {
        for (batch_idx = 0; batch_idx < R394_TRACE_BATCH_SIZE; ++batch_idx) {
            order[update_idx * R394_TRACE_BATCH_SIZE + batch_idx] =
                (unsigned int)(xorshift64(&state) % (uint64_t)R394_TRACE_TRAIN_SAMPLES);
        }
    }
}

static int evaluate_checkpoint(
    const TraceData *data,
    const float *rho1_a,
    const int8_t *code,
    float *out_loss,
    float *out_signature
) {
    float b_eval[V12_D_STATE];
    float *y_eval;
    float *h_cache_eval;
    float *h_final_eval;
    size_t batch_stride = (size_t)R394_TRACE_BATCH_SIZE * R394_TRACE_SEQ_LEN * V12_D_STATE;
    size_t state_count = (size_t)R394_TRACE_BATCH_SIZE * V12_D_STATE;
    size_t ckpt_state_count = (size_t)R394_TRACE_BATCH_SIZE * R394_TRACE_CKPT_COUNT * V12_D_STATE;
    size_t idx;
    float loss_sum = 0.0f;

    y_eval = (float *)malloc(batch_stride * sizeof(float));
    h_cache_eval = (float *)malloc(ckpt_state_count * sizeof(float));
    h_final_eval = (float *)malloc(state_count * sizeof(float));
    if (!y_eval || !h_cache_eval || !h_final_eval) {
        free(y_eval);
        free(h_cache_eval);
        free(h_final_eval);
        return 0;
    }

    build_b_from_code(code, b_eval);
    if (!run_ssm_batch(
            rho1_a,
            b_eval,
            data->projection_c,
            data->eval_x,
            R394_TRACE_BATCH_SIZE,
            y_eval,
            h_cache_eval,
            h_final_eval
        )) {
        free(y_eval);
        free(h_cache_eval);
        free(h_final_eval);
        return 0;
    }

    for (idx = 0; idx < batch_stride; ++idx) {
        const float err = y_eval[idx] - data->eval_target_y[idx];
        loss_sum += 0.5f * err * err;
    }
    *out_loss = loss_sum / (float)batch_stride;

    memset(out_signature, 0, (size_t)V12_D_STATE * sizeof(float));
    for (idx = 0; idx < ckpt_state_count; ++idx) {
        out_signature[idx % V12_D_STATE] += h_cache_eval[idx];
    }
    for (idx = 0; idx < state_count; ++idx) {
        out_signature[idx % V12_D_STATE] += 0.5f * h_final_eval[idx];
    }
    for (idx = 0; idx < (size_t)V12_D_STATE; ++idx) {
        out_signature[idx] /= (float)(R394_TRACE_BATCH_SIZE * (R394_TRACE_CKPT_COUNT + 0.5f));
    }

    free(y_eval);
    free(h_cache_eval);
    free(h_final_eval);
    return all_finite_array(out_signature, (size_t)V12_D_STATE) && isfinite(*out_loss);
}

static float compute_signature_jump(const float *lhs, const float *rhs) {
    float accum = 0.0f;
    int i;

    for (i = 0; i < V12_D_STATE; ++i) {
        accum += fabsf(lhs[i] - rhs[i]);
    }
    return accum / (float)V12_D_STATE;
}

static float compute_signature_alignment(const float *lhs, const float *rhs) {
    float dot = 0.0f;
    float lhs_norm = 0.0f;
    float rhs_norm = 0.0f;
    int i;

    for (i = 0; i < V12_D_STATE; ++i) {
        dot += lhs[i] * rhs[i];
        lhs_norm += lhs[i] * lhs[i];
        rhs_norm += rhs[i] * rhs[i];
    }

    if (lhs_norm < R394_SSH_EPS || rhs_norm < R394_SSH_EPS) {
        return 0.0f;
    }
    return dot / sqrtf(lhs_norm * rhs_norm);
}

static void run_trace_path(
    const TraceData *data,
    const LayerSpectrum *spectrum,
    const float *stage_rhos,
    const unsigned int *order,
    TraceRun *trace
) {
    float shadow[V12_D_STATE];
    int8_t code[V12_D_STATE];
    int8_t prev_code[V12_D_STATE];
    int8_t older_code[V12_D_STATE];
    float prev_signature[V12_D_STATE];
    float current_signature[V12_D_STATE];
    float *x_batch;
    float *target_batch;
    float *y_batch;
    float *h_cache_batch;
    float *h_final_batch;
    float *dy_batch;
    float *dx_batch;
    float *d_c;
    float *d_a;
    float d_b[V12_D_STATE];
    int update_idx;
    int checkpoint_idx = 0;
    size_t batch_stride = (size_t)R394_TRACE_BATCH_SIZE * R394_TRACE_SEQ_LEN * V12_D_STATE;
    size_t state_count = (size_t)R394_TRACE_BATCH_SIZE * V12_D_STATE;
    size_t ckpt_state_count = (size_t)R394_TRACE_BATCH_SIZE * R394_TRACE_CKPT_COUNT * V12_D_STATE;
    float rho1_a[V12_D_STATE];

    memset(trace, 0, sizeof(*trace));
    trace->finite_pass = 1;
    trace->mode_alignment_min = 1.0f;
    trace->final_rho_recovery_window = -1;
    trace->final_rho_boundary_checkpoint = -1;
    memset(shadow, 0, sizeof(shadow));
    memset(code, 0, sizeof(code));
    memset(prev_code, 0, sizeof(prev_code));
    memset(older_code, 0, sizeof(older_code));
    memset(prev_signature, 0, sizeof(prev_signature));

    materialize_homotopy_a(spectrum, 1.0f, rho1_a);

    if (!evaluate_checkpoint(data, rho1_a, code, &trace->loss_start, prev_signature)) {
        mark_first_bad(trace, "eval_rho1_initial");
        trace->loss_end = trace->loss_start;
        return;
    }

    x_batch = (float *)malloc(batch_stride * sizeof(float));
    target_batch = (float *)malloc(batch_stride * sizeof(float));
    y_batch = (float *)malloc(batch_stride * sizeof(float));
    h_cache_batch = (float *)malloc(ckpt_state_count * sizeof(float));
    h_final_batch = (float *)malloc(state_count * sizeof(float));
    dy_batch = (float *)malloc(batch_stride * sizeof(float));
    dx_batch = (float *)malloc(batch_stride * sizeof(float));
    d_c = (float *)malloc((size_t)V12_D_STATE * V12_D_STATE * sizeof(float));
    d_a = (float *)malloc((size_t)V12_D_STATE * sizeof(float));
    if (!x_batch || !target_batch || !y_batch || !h_cache_batch || !h_final_batch || !dy_batch ||
        !dx_batch || !d_c || !d_a) {
        mark_first_bad(trace, "alloc");
        free(x_batch);
        free(target_batch);
        free(y_batch);
        free(h_cache_batch);
        free(h_final_batch);
        free(dy_batch);
        free(dx_batch);
        free(d_c);
        free(d_a);
        trace->loss_end = trace->loss_start;
        return;
    }

    for (update_idx = 0; update_idx < R394_TRACE_TOTAL_UPDATES; ++update_idx) {
        float a_stage[V12_D_STATE];
        float b_stage[V12_D_STATE];
        int stage_idx = update_idx / R394_TRACE_STAGE_STEPS;
        const unsigned int *batch_indices = order + update_idx * R394_TRACE_BATCH_SIZE;
        size_t idx;

        materialize_homotopy_a(spectrum, stage_rhos[stage_idx], a_stage);
        project_shadow_to_code(shadow, code);
        build_b_from_code(code, b_stage);
        gather_sequences(data->train_x, data->train_target_y, batch_indices, x_batch, target_batch);

        if (!run_ssm_batch(
                a_stage,
                b_stage,
                data->projection_c,
                x_batch,
                R394_TRACE_BATCH_SIZE,
                y_batch,
                h_cache_batch,
                h_final_batch
            )) {
            mark_first_bad(trace, "train_forward");
            break;
        }

        for (idx = 0; idx < batch_stride; ++idx) {
            dy_batch[idx] = (y_batch[idx] - target_batch[idx]) / (float)batch_stride;
        }
        if (!all_finite_array(dy_batch, batch_stride)) {
            mark_first_bad(trace, "train_loss");
            break;
        }

        memset(d_a, 0, (size_t)V12_D_STATE * sizeof(float));
        memset(d_b, 0, sizeof(d_b));
        memset(d_c, 0, (size_t)V12_D_STATE * V12_D_STATE * sizeof(float));
        memset(dx_batch, 0, batch_stride * sizeof(float));

        ssm_scan_backward_checkpointed_batch(
            a_stage,
            NULL,
            NULL,
            b_stage,
            0,
            data->projection_c,
            x_batch,
            h_cache_batch,
            dy_batch,
            NULL,
            d_a,
            NULL,
            d_b,
            d_c,
            dx_batch,
            NULL,
            R394_TRACE_BATCH_SIZE,
            R394_TRACE_SEQ_LEN,
            V12_D_STATE,
            V12_D_STATE,
            NULL,
            NULL
        );

        if (!all_finite_array(d_b, (size_t)V12_D_STATE)) {
            mark_first_bad(trace, "dB");
            break;
        }

        for (idx = 0; idx < (size_t)V12_D_STATE; ++idx) {
            shadow[idx] -= R394_TRACE_LR * d_b[idx] * R394_TRACE_B_SCALE;
            shadow[idx] = clamp_symmetric(shadow[idx], R394_TRACE_SHADOW_CLAMP);
        }

        if ((update_idx + 1) % R394_TRACE_CHECKPOINT_INTERVAL == 0) {
            float loss = 0.0f;
            float alignment = 1.0f;
            float state_jump = 0.0f;
            int flips;
            int reversals;

            project_shadow_to_code(shadow, code);
            if (!evaluate_checkpoint(data, rho1_a, code, &loss, current_signature)) {
                mark_first_bad(trace, "eval_rho1_checkpoint");
                break;
            }

            flips = count_code_flips(prev_code, code);
            reversals = count_code_reversals(older_code, prev_code, code);
            if (checkpoint_idx > 0) {
                alignment = compute_signature_alignment(prev_signature, current_signature);
                state_jump = compute_signature_jump(prev_signature, current_signature);
                if (alignment < trace->mode_alignment_min) {
                    trace->mode_alignment_min = alignment;
                }
            }

            trace->checkpoint_losses[checkpoint_idx] = loss;
            trace->checkpoint_rhos[checkpoint_idx] = stage_rhos[stage_idx];
            trace->checkpoint_state_jump[checkpoint_idx] = state_jump;
            trace->checkpoint_alignment[checkpoint_idx] = alignment;
            trace->checkpoint_flips[checkpoint_idx] = flips;
            trace->checkpoint_reversals[checkpoint_idx] = reversals;
            ++trace->checkpoint_count;

            if (checkpoint_idx > 0 &&
                trace->checkpoint_losses[checkpoint_idx - 1] - loss > R394_TRACE_PROGRESS_FLOOR &&
                flips > 0) {
                trace->useful_flip_count += flips;
                trace->useful_flip_reversals += reversals;
            }

            memcpy(older_code, prev_code, sizeof(older_code));
            memcpy(prev_code, code, sizeof(prev_code));
            memcpy(prev_signature, current_signature, sizeof(prev_signature));
            ++checkpoint_idx;
        }
    }

    if (trace->checkpoint_count > 0) {
        trace->loss_end = trace->checkpoint_losses[trace->checkpoint_count - 1];
    } else {
        trace->loss_end = trace->loss_start;
    }

    free(x_batch);
    free(target_batch);
    free(y_batch);
    free(h_cache_batch);
    free(h_final_batch);
    free(dy_batch);
    free(dx_batch);
    free(d_c);
    free(d_a);
}

static void finalize_boundary_metrics(TraceRun *trace) {
    size_t idx;

    if (trace->mode_alignment_min > 1.0f) {
        trace->mode_alignment_min = 1.0f;
    }

    for (idx = 1; idx < trace->checkpoint_count; ++idx) {
        if (fabsf(trace->checkpoint_rhos[idx] - 1.0f) < 1.0e-6f &&
            trace->checkpoint_rhos[idx - 1] < 1.0f) {
            size_t lookahead;
            const float pre_loss = trace->checkpoint_losses[idx - 1];
            const float boundary_loss = trace->checkpoint_losses[idx];

            trace->final_rho_boundary_checkpoint = (int)idx + 1;
            trace->final_rho_boundary_loss_jump = boundary_loss - pre_loss;
            trace->final_rho_boundary_state_jump = trace->checkpoint_state_jump[idx];
            snprintf(
                trace->restart_provenance,
                sizeof(trace->restart_provenance),
                "rho=%.2f->%.2f checkpoint=%d",
                trace->checkpoint_rhos[idx - 1],
                trace->checkpoint_rhos[idx],
                (int)idx + 1
            );

            for (lookahead = idx + 1; lookahead < trace->checkpoint_count; ++lookahead) {
                const float loss_delta = trace->checkpoint_losses[lookahead] - pre_loss;
                const float alignment = trace->checkpoint_alignment[lookahead];
                if (loss_delta <= R394_TRACE_ENDPOINT_TOL &&
                    trace->checkpoint_state_jump[lookahead] <= R394_TRACE_BOUNDARY_STATE_TOL &&
                    alignment >= R394_TRACE_ALIGNMENT_FLOOR) {
                    if (lookahead + 1 < trace->checkpoint_count) {
                        const float next_loss_delta = trace->checkpoint_losses[lookahead + 1] - pre_loss;
                        const float next_alignment = trace->checkpoint_alignment[lookahead + 1];
                        if (next_loss_delta <= R394_TRACE_ENDPOINT_TOL &&
                            trace->checkpoint_state_jump[lookahead + 1] <= R394_TRACE_BOUNDARY_STATE_TOL &&
                            next_alignment >= R394_TRACE_ALIGNMENT_FLOOR) {
                            trace->final_rho_recovery_window = (int)(lookahead - idx);
                            return;
                        }
                    }
                }
            }
            return;
        }
    }

    strcpy(trace->restart_provenance, "rho<1.0 boundary not observed");
}

static const char *classify_endpoint_transfer(
    const TraceRun *baseline,
    const TraceRun *ssh,
    float *delta_vs_baseline
) {
    const float baseline_improvement = baseline->loss_start - baseline->loss_end;
    const float ssh_improvement = ssh->loss_start - ssh->loss_end;

    *delta_vs_baseline = ssh->loss_end - baseline->loss_end;

    if (fabsf(baseline_improvement) < R394_TRACE_PROGRESS_FLOOR &&
        fabsf(ssh_improvement) < R394_TRACE_PROGRESS_FLOOR) {
        return "inconclusive";
    }
    if (*delta_vs_baseline <= R394_TRACE_ENDPOINT_TOL &&
        ssh_improvement >= -R394_TRACE_ENDPOINT_TOL) {
        return "pass";
    }
    return "fail";
}

static const char *classify_ternary_progress(
    const TraceRun *baseline,
    const TraceRun *ssh,
    float *ternary_progress
) {
    const float baseline_improvement = baseline->loss_start - baseline->loss_end;
    const float ssh_improvement = ssh->loss_start - ssh->loss_end;

    *ternary_progress = 0.0f;

    if (ssh->useful_flip_count > 0 && ssh_improvement > R394_TRACE_PROGRESS_FLOOR) {
        *ternary_progress = ssh_improvement / (float)ssh->useful_flip_count;
        return "pass";
    }
    if (fabsf(baseline_improvement) < R394_TRACE_PROGRESS_FLOOR &&
        fabsf(ssh_improvement) < R394_TRACE_PROGRESS_FLOOR &&
        ssh->useful_flip_count == 0) {
        return "inconclusive";
    }
    return "fail";
}

static const char *classify_restart_signature_trace(const TraceRun *trace) {
    const int jump_present =
        trace->final_rho_boundary_loss_jump > R394_TRACE_BOUNDARY_LOSS_TOL ||
        trace->final_rho_boundary_state_jump > R394_TRACE_BOUNDARY_STATE_TOL ||
        trace->mode_alignment_min < R394_TRACE_ALIGNMENT_FLOOR;

    if (trace->final_rho_boundary_checkpoint < 0) {
        return "inconclusive";
    }
    if (!jump_present) {
        return "pass";
    }
    if (trace->final_rho_recovery_window >= 1) {
        return "pass";
    }
    if ((size_t)trace->final_rho_boundary_checkpoint + 2 > trace->checkpoint_count) {
        return "inconclusive";
    }
    return "fail";
}

static const char *classify_branch_decision_trace(
    const char *finite_state,
    const char *endpoint_transfer,
    const char *ternary_progress_state,
    const char *restart_signature_trace
) {
    if (strcmp(finite_state, "pass") != 0 ||
        strcmp(endpoint_transfer, "fail") == 0 ||
        strcmp(ternary_progress_state, "fail") == 0 ||
        strcmp(restart_signature_trace, "fail") == 0) {
        return "kill";
    }
    if (strcmp(endpoint_transfer, "pass") == 0 &&
        strcmp(ternary_progress_state, "pass") == 0 &&
        strcmp(restart_signature_trace, "pass") == 0) {
        return "pass";
    }
    return "inconclusive";
}

int main(void) {
    static const float ssh_path[R394_TRACE_RHO_COUNT] = {0.25f, 0.50f, 0.75f, 1.00f};
    static const float baseline_path[R394_TRACE_RHO_COUNT] = {1.00f, 1.00f, 1.00f, 1.00f};

    ProbeConfig config;
    float layer_a[V12_D_STATE];
    LayerSpectrum spectrum;
    float a_rho[V12_D_STATE];
    float baseline_kappa[R394_MAX_HORIZONS];
    float global_ratios[R394_MAX_RHOS][R394_MAX_HORIZONS];
    float best_ratio = INFINITY;
    float best_rho = 1.0f;
    int worst_horizon = 0;
    int worst_limiting_layer = 0;
    size_t rho_idx;
    size_t horizon_idx;
    TraceData data;
    TraceRun baseline_trace;
    TraceRun ssh_trace;
    unsigned int order[R394_TRACE_TOTAL_UPDATES * R394_TRACE_BATCH_SIZE];
    float delta_vs_baseline = 0.0f;
    float ternary_progress = 0.0f;
    const char *finite_state = "pass";
    const char *endpoint_transfer;
    const char *ternary_progress_state;
    const char *restart_signature_trace;
    const char *branch_progress_decision_trace;
    float rho1_a[V12_D_STATE];

    build_config(&config);
    memset(&data, 0, sizeof(data));

    initialize_layer_a(layer_a);
    compute_layer_spectrum(layer_a, &spectrum);

    for (horizon_idx = 0; horizon_idx < config.n_horizons; ++horizon_idx) {
        materialize_homotopy_a(&spectrum, 1.0f, a_rho);
        baseline_kappa[horizon_idx] = compute_kappa_eff(a_rho, config.horizons[horizon_idx]);
    }

    for (rho_idx = 0; rho_idx < config.n_rhos; ++rho_idx) {
        const float rho = config.rhos[rho_idx];
        float rho_worst_ratio = 0.0f;
        int rho_worst_horizon = 0;

        for (horizon_idx = 0; horizon_idx < config.n_horizons; ++horizon_idx) {
            float global_kappa;

            materialize_homotopy_a(&spectrum, rho, a_rho);
            global_kappa = compute_kappa_eff(a_rho, config.horizons[horizon_idx]);
            global_ratios[rho_idx][horizon_idx] =
                global_kappa / fmaxf(R394_SSH_EPS, baseline_kappa[horizon_idx]);

            printf(
                "[R394_SSH_TRACE] global H=%d rho=%.2f kappa_eff=%.9g r_kappa=%.9g limiting_layer=%d\n",
                config.horizons[horizon_idx],
                rho,
                global_kappa,
                global_ratios[rho_idx][horizon_idx],
                0
            );

            if (rho < 1.0f && global_ratios[rho_idx][horizon_idx] > rho_worst_ratio) {
                rho_worst_ratio = global_ratios[rho_idx][horizon_idx];
                rho_worst_horizon = config.horizons[horizon_idx];
            }
        }

        if (rho < 1.0f && rho_worst_ratio < best_ratio + 1.0e-12f) {
            best_ratio = rho_worst_ratio;
            best_rho = rho;
            worst_horizon = rho_worst_horizon;
            worst_limiting_layer = 0;
        }
    }

    if (!isfinite(best_ratio)) {
        best_ratio = 1.0f;
    }

    materialize_homotopy_a(&spectrum, 1.0f, rho1_a);
    if (!build_trace_data(rho1_a, &spectrum, &data)) {
        fprintf(stderr, "r394_ssh_trace_probe: failed to allocate trace data\n");
        free_trace_data(&data);
        return 1;
    }

    build_train_order(order);
    run_trace_path(&data, &spectrum, baseline_path, order, &baseline_trace);
    run_trace_path(&data, &spectrum, ssh_path, order, &ssh_trace);
    finalize_boundary_metrics(&ssh_trace);

    if (!baseline_trace.finite_pass || !ssh_trace.finite_pass) {
        finite_state = "fail";
    }

    endpoint_transfer = classify_endpoint_transfer(&baseline_trace, &ssh_trace, &delta_vs_baseline);
    ternary_progress_state = classify_ternary_progress(&baseline_trace, &ssh_trace, &ternary_progress);
    restart_signature_trace = classify_restart_signature_trace(&ssh_trace);
    branch_progress_decision_trace = classify_branch_decision_trace(
        finite_state,
        endpoint_transfer,
        ternary_progress_state,
        restart_signature_trace
    );

    printf(
        "r394_ssh_trace metric_decision=%s\n",
        classify_metric(best_ratio)
    );
    printf(
        "r394_ssh_trace r_kappa_best=%.9g rho_best=%.2f horizon_worst=%d limiting_layer=%d\n",
        best_ratio,
        best_rho,
        worst_horizon,
        worst_limiting_layer
    );
    printf(
        "r394_ssh_trace finite_state=%s first_bad=%s\n",
        finite_state,
        strcmp(finite_state, "pass") == 0
            ? "none"
            : (ssh_trace.first_bad[0] ? ssh_trace.first_bad : baseline_trace.first_bad)
    );
    printf(
        "r394_ssh_trace endpoint_transfer=%s delta_vs_baseline=%.9g\n",
        endpoint_transfer,
        delta_vs_baseline
    );
    printf(
        "r394_ssh_trace useful_flip_count=%d useful_flip_reversals=%d\n",
        ssh_trace.useful_flip_count,
        ssh_trace.useful_flip_reversals
    );
    if (strcmp(ternary_progress_state, "pass") == 0) {
        printf("r394_ssh_trace ternary_progress=%.9g\n", ternary_progress);
    } else {
        printf("r394_ssh_trace ternary_progress=%s\n", ternary_progress_state);
    }
    printf(
        "r394_ssh_trace final_rho_boundary_loss_jump=%.9g\n",
        ssh_trace.final_rho_boundary_loss_jump
    );
    printf(
        "r394_ssh_trace final_rho_boundary_state_jump=%.9g\n",
        ssh_trace.final_rho_boundary_state_jump
    );
    printf(
        "r394_ssh_trace final_rho_recovery_window=%d\n",
        ssh_trace.final_rho_recovery_window
    );
    printf(
        "r394_ssh_trace mode_alignment_min=%.9g\n",
        ssh_trace.mode_alignment_min
    );
    printf(
        "r394_ssh_trace restart_signature_trace=%s provenance=%s\n",
        restart_signature_trace,
        ssh_trace.restart_provenance
    );
    printf(
        "r394_ssh_trace branch_progress_decision_trace=%s\n",
        branch_progress_decision_trace
    );

    for (rho_idx = 0; rho_idx < ssh_trace.checkpoint_count; ++rho_idx) {
        printf(
            "r394_ssh_trace checkpoint=%d rho=%.2f loss=%.9g state_jump=%.9g alignment=%.9g flips=%d reversals=%d\n",
            (int)rho_idx + 1,
            ssh_trace.checkpoint_rhos[rho_idx],
            ssh_trace.checkpoint_losses[rho_idx],
            ssh_trace.checkpoint_state_jump[rho_idx],
            ssh_trace.checkpoint_alignment[rho_idx],
            ssh_trace.checkpoint_flips[rho_idx],
            ssh_trace.checkpoint_reversals[rho_idx]
        );
    }

    free_trace_data(&data);
    return strcmp(finite_state, "pass") == 0 ? 0 : 1;
}
