#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/core/types.h"

#define R394_SSH_EPS 1.0e-6f
#define R394_MAX_RHOS 16
#define R394_MAX_HORIZONS 16

#define R394_BRANCH_RHO_COUNT 4
#define R394_BRANCH_CHECKPOINTS 8
#define R394_TRAIN_SAMPLES 64
#define R394_EVAL_SAMPLES 32
#define R394_TOTAL_SAMPLES (R394_TRAIN_SAMPLES + R394_EVAL_SAMPLES)
#define R394_BATCH_SIZE 4
#define R394_STAGE_STEPS 16
#define R394_TOTAL_UPDATES (R394_BRANCH_RHO_COUNT * R394_STAGE_STEPS)
#define R394_CHECKPOINT_INTERVAL 8
#define R394_COORD_COUNT (V12_N_LAYERS * V12_D_STATE)

#define R394_DIAG_LR 0.18f
#define R394_DIAG_SHADOW_CLAMP 2.5f
#define R394_DIAG_DEADZONE 0.18f
#define R394_DIAG_PROGRESS_FLOOR 5.0e-4f
#define R394_DIAG_ENDPOINT_TOL 2.0e-4f
#define R394_DIAG_JUMP_TOL 7.5e-4f
#define R394_DIAG_HIDDEN_JUMP_TOL 1.5e-2f

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
    float *basis[R394_BRANCH_RHO_COUNT];
    float target_values[R394_TOTAL_SAMPLES];
    int8_t target_code[R394_COORD_COUNT];
} DiagnosticData;

typedef struct {
    float loss_start;
    float loss_end;
    float checkpoint_losses[R394_BRANCH_CHECKPOINTS];
    float checkpoint_hidden[R394_BRANCH_CHECKPOINTS];
    float checkpoint_transition_jumps[R394_BRANCH_CHECKPOINTS];
    float checkpoint_rhos[R394_BRANCH_CHECKPOINTS];
    int checkpoint_flips[R394_BRANCH_CHECKPOINTS];
    size_t checkpoint_count;
    int useful_flip_count;
    int finite_pass;
    char first_bad[64];
} DiagnosticTrace;

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

static void materialize_homotopy_a(
    const LayerSpectrum *spectrum,
    float rho,
    float *out_a
) {
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

static void flatten_rho_features(
    const LayerSpectrum *spectra,
    float rho,
    float *out_a_flat,
    float *out_log_tau_flat
) {
    float a_rho[V12_D_STATE];
    int layer_idx;

    for (layer_idx = 0; layer_idx < V12_N_LAYERS; ++layer_idx) {
        int state_idx;
        materialize_homotopy_a(&spectra[layer_idx], rho, a_rho);
        for (state_idx = 0; state_idx < V12_D_STATE; ++state_idx) {
            const int coord = layer_idx * V12_D_STATE + state_idx;
            out_a_flat[coord] = a_rho[state_idx];
            out_log_tau_flat[coord] =
                spectra[layer_idx].mean_log_tau +
                rho * (spectra[layer_idx].log_tau[state_idx] - spectra[layer_idx].mean_log_tau);
        }
    }
}

static int rho_index_from_value(float rho) {
    static const float diag_rhos[R394_BRANCH_RHO_COUNT] = {0.25f, 0.50f, 0.75f, 1.00f};
    int i;
    for (i = 0; i < R394_BRANCH_RHO_COUNT; ++i) {
        if (fabsf(rho - diag_rhos[i]) < 1.0e-6f) {
            return i;
        }
    }
    return -1;
}

static void project_shadow_to_code(const float *shadow, int8_t *code) {
    int coord;
    for (coord = 0; coord < R394_COORD_COUNT; ++coord) {
        if (shadow[coord] > R394_DIAG_DEADZONE) {
            code[coord] = 1;
        } else if (shadow[coord] < -R394_DIAG_DEADZONE) {
            code[coord] = -1;
        } else {
            code[coord] = 0;
        }
    }
}

static int count_code_flips(const int8_t *lhs, const int8_t *rhs) {
    int coord;
    int flips = 0;
    for (coord = 0; coord < R394_COORD_COUNT; ++coord) {
        if (lhs[coord] != rhs[coord]) {
            ++flips;
        }
    }
    return flips;
}

static void mark_first_bad(DiagnosticTrace *trace, const char *field) {
    if (trace->finite_pass && trace->first_bad[0] == '\0') {
        size_t n = strlen(field);
        if (n >= sizeof(trace->first_bad)) {
            n = sizeof(trace->first_bad) - 1;
        }
        memcpy(trace->first_bad, field, n);
        trace->first_bad[n] = '\0';
    }
    trace->finite_pass = 0;
}

static int all_finite_array(const float *values, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        if (!isfinite(values[i])) {
            return 0;
        }
    }
    return 1;
}

static void build_target_code(
    const float *rho1_a,
    const float *rho1_log_tau,
    int8_t *target_code
) {
    float mean = 0.0f;
    float var = 0.0f;
    int coord;

    for (coord = 0; coord < R394_COORD_COUNT; ++coord) {
        mean += rho1_log_tau[coord];
    }
    mean /= (float)R394_COORD_COUNT;

    for (coord = 0; coord < R394_COORD_COUNT; ++coord) {
        const float centered = rho1_log_tau[coord] - mean;
        var += centered * centered;
    }
    var /= (float)R394_COORD_COUNT;
    if (var < R394_SSH_EPS) {
        var = R394_SSH_EPS;
    }

    for (coord = 0; coord < R394_COORD_COUNT; ++coord) {
        const int layer_idx = coord / V12_D_STATE;
        const int state_idx = coord % V12_D_STATE;
        const float centered = (rho1_log_tau[coord] - mean) / sqrtf(var);
        const float score =
            0.85f * sinf(1.7f * centered + 0.21f * (float)(layer_idx + 1)) +
            0.35f * cosf(2.4f * rho1_a[coord] + 0.07f * (float)(state_idx + 1));

        if (score > 0.32f) {
            target_code[coord] = 1;
        } else if (score < -0.32f) {
            target_code[coord] = -1;
        } else {
            target_code[coord] = 0;
        }
    }
}

static void build_basis_for_rho(
    const float *a_flat,
    const float *log_tau_flat,
    float *out_basis
) {
    int sample_idx;

    for (sample_idx = 0; sample_idx < R394_TOTAL_SAMPLES; ++sample_idx) {
        float sum_sq = 0.0f;
        int coord;

        for (coord = 0; coord < R394_COORD_COUNT; ++coord) {
            const int layer_idx = coord / V12_D_STATE;
            const int state_idx = coord % V12_D_STATE;
            const float sample_phase = (float)(sample_idx + 1);
            const float carrier =
                sinf(0.043f * sample_phase * (float)(state_idx + 1) + 1.35f * log_tau_flat[coord]);
            const float envelope =
                cosf(0.071f * (sample_phase + 2.0f) * a_flat[coord] +
                     0.17f * (float)(layer_idx + 1));
            const float bias =
                0.15f * sinf(0.11f * (float)(state_idx + 1) + 0.53f * (float)(layer_idx + 1));
            const float value = carrier + 0.55f * envelope + bias;
            out_basis[(size_t)sample_idx * R394_COORD_COUNT + coord] = value;
            sum_sq += value * value;
        }

        sum_sq = sqrtf(sum_sq / (float)R394_COORD_COUNT);
        if (sum_sq < R394_SSH_EPS) {
            sum_sq = 1.0f;
        }

        for (coord = 0; coord < R394_COORD_COUNT; ++coord) {
            out_basis[(size_t)sample_idx * R394_COORD_COUNT + coord] /= sum_sq;
        }
    }
}

static void build_diagnostic_data(const LayerSpectrum *spectra, DiagnosticData *data) {
    static const float diag_rhos[R394_BRANCH_RHO_COUNT] = {0.25f, 0.50f, 0.75f, 1.00f};
    float *rho_a[R394_BRANCH_RHO_COUNT];
    float *rho_log_tau[R394_BRANCH_RHO_COUNT];
    int rho_idx;
    int sample_idx;

    memset(data, 0, sizeof(*data));

    for (rho_idx = 0; rho_idx < R394_BRANCH_RHO_COUNT; ++rho_idx) {
        rho_a[rho_idx] = (float *)malloc((size_t)R394_COORD_COUNT * sizeof(float));
        rho_log_tau[rho_idx] = (float *)malloc((size_t)R394_COORD_COUNT * sizeof(float));
        data->basis[rho_idx] = (float *)malloc(
            (size_t)R394_TOTAL_SAMPLES * R394_COORD_COUNT * sizeof(float)
        );
        if (!rho_a[rho_idx] || !rho_log_tau[rho_idx] || !data->basis[rho_idx]) {
            int cleanup_idx;
            for (cleanup_idx = 0; cleanup_idx <= rho_idx; ++cleanup_idx) {
                free(rho_a[cleanup_idx]);
                free(rho_log_tau[cleanup_idx]);
                free(data->basis[cleanup_idx]);
                data->basis[cleanup_idx] = NULL;
            }
            return;
        }

        flatten_rho_features(spectra, diag_rhos[rho_idx], rho_a[rho_idx], rho_log_tau[rho_idx]);
        build_basis_for_rho(rho_a[rho_idx], rho_log_tau[rho_idx], data->basis[rho_idx]);
    }

    build_target_code(rho_a[R394_BRANCH_RHO_COUNT - 1], rho_log_tau[R394_BRANCH_RHO_COUNT - 1],
                      data->target_code);

    for (sample_idx = 0; sample_idx < R394_TOTAL_SAMPLES; ++sample_idx) {
        const float *basis =
            data->basis[R394_BRANCH_RHO_COUNT - 1] + (size_t)sample_idx * R394_COORD_COUNT;
        float dot = 0.0f;
        int coord;

        for (coord = 0; coord < R394_COORD_COUNT; ++coord) {
            dot += (float)data->target_code[coord] * basis[coord];
        }
        data->target_values[sample_idx] = dot / sqrtf((float)R394_COORD_COUNT);
    }

    for (rho_idx = 0; rho_idx < R394_BRANCH_RHO_COUNT; ++rho_idx) {
        free(rho_a[rho_idx]);
        free(rho_log_tau[rho_idx]);
    }
}

static void free_diagnostic_data(DiagnosticData *data) {
    int rho_idx;
    for (rho_idx = 0; rho_idx < R394_BRANCH_RHO_COUNT; ++rho_idx) {
        free(data->basis[rho_idx]);
        data->basis[rho_idx] = NULL;
    }
}

static void build_batch_order(unsigned int *order) {
    uint64_t state = 0xC0FFEE123456789ULL;
    int idx;

    for (idx = 0; idx < R394_TOTAL_UPDATES * R394_BATCH_SIZE; ++idx) {
        order[idx] = (unsigned int)(xorshift64(&state) % (uint64_t)R394_TRAIN_SAMPLES);
    }
}

static float compute_prediction(const int8_t *code, const float *basis_row) {
    float dot = 0.0f;
    int coord;
    for (coord = 0; coord < R394_COORD_COUNT; ++coord) {
        dot += (float)code[coord] * basis_row[coord];
    }
    return dot / sqrtf((float)R394_COORD_COUNT);
}

static void evaluate_final_objective(
    const DiagnosticData *data,
    const int8_t *code,
    float *out_loss,
    float *out_hidden_summary,
    DiagnosticTrace *trace,
    const char *loss_field,
    const char *hidden_field
) {
    const float *basis_rho1 = data->basis[R394_BRANCH_RHO_COUNT - 1];
    float loss_sum = 0.0f;
    float hidden_sum = 0.0f;
    int sample_idx;

    *out_loss = INFINITY;
    *out_hidden_summary = INFINITY;

    for (sample_idx = R394_TRAIN_SAMPLES; sample_idx < R394_TOTAL_SAMPLES; ++sample_idx) {
        const float *basis_row = basis_rho1 + (size_t)sample_idx * R394_COORD_COUNT;
        const float pred = compute_prediction(code, basis_row);
        const float err = pred - data->target_values[sample_idx];
        const float loss = 0.5f * err * err;

        if (!isfinite(pred)) {
            mark_first_bad(trace, loss_field);
            return;
        }
        if (!isfinite(loss)) {
            mark_first_bad(trace, loss_field);
            return;
        }

        loss_sum += loss;
        hidden_sum += fabsf(pred);
    }

    *out_loss = loss_sum / (float)R394_EVAL_SAMPLES;
    *out_hidden_summary = hidden_sum / (float)R394_EVAL_SAMPLES;

    if (!isfinite(*out_loss)) {
        mark_first_bad(trace, loss_field);
    }
    if (!isfinite(*out_hidden_summary)) {
        mark_first_bad(trace, hidden_field);
    }
}

static void run_branch_path(
    const DiagnosticData *data,
    const float *stage_rhos,
    const unsigned int *order,
    DiagnosticTrace *trace
) {
    float shadow[R394_COORD_COUNT];
    float grad[R394_COORD_COUNT];
    int8_t code[R394_COORD_COUNT];
    int8_t prev_checkpoint_code[R394_COORD_COUNT];
    int update_idx;
    int checkpoint_idx = 0;

    memset(trace, 0, sizeof(*trace));
    trace->finite_pass = 1;
    trace->first_bad[0] = '\0';
    memset(shadow, 0, sizeof(shadow));
    memset(code, 0, sizeof(code));
    memset(prev_checkpoint_code, 0, sizeof(prev_checkpoint_code));

    evaluate_final_objective(data, code, &trace->loss_start, &trace->checkpoint_hidden[0],
                             trace, "eval_rho1_loss", "hidden_summary");
    if (!trace->finite_pass) {
        trace->loss_end = trace->loss_start;
        return;
    }

    for (update_idx = 0; update_idx < R394_TOTAL_UPDATES; ++update_idx) {
        const int stage_idx = update_idx / R394_STAGE_STEPS;
        const float rho = stage_rhos[stage_idx];
        const int rho_idx = rho_index_from_value(rho);
        const float *basis_matrix;
        int batch_offset;
        int coord;

        if (rho_idx < 0) {
            mark_first_bad(trace, "rho_index");
            break;
        }
        basis_matrix = data->basis[rho_idx];

        memset(grad, 0, sizeof(grad));
        project_shadow_to_code(shadow, code);

        for (batch_offset = 0; batch_offset < R394_BATCH_SIZE; ++batch_offset) {
            const unsigned int sample_idx =
                order[update_idx * R394_BATCH_SIZE + batch_offset];
            const float *basis_row = basis_matrix + (size_t)sample_idx * R394_COORD_COUNT;
            const float pred = compute_prediction(code, basis_row);
            const float err = pred - data->target_values[sample_idx];
            const float scale = err / sqrtf((float)R394_COORD_COUNT);

            if (!isfinite(pred)) {
                mark_first_bad(trace, "train_logits");
                break;
            }
            if (!isfinite(err)) {
                mark_first_bad(trace, "train_loss");
                break;
            }

            for (coord = 0; coord < R394_COORD_COUNT; ++coord) {
                grad[coord] += scale * basis_row[coord];
            }
        }

        if (!trace->finite_pass) {
            break;
        }

        if (!all_finite_array(grad, R394_COORD_COUNT)) {
            mark_first_bad(trace, "gradients");
            break;
        }

        for (coord = 0; coord < R394_COORD_COUNT; ++coord) {
            shadow[coord] -= (R394_DIAG_LR / (float)R394_BATCH_SIZE) * grad[coord];
            shadow[coord] = clamp_symmetric(shadow[coord], R394_DIAG_SHADOW_CLAMP);
        }

        if (!all_finite_array(shadow, R394_COORD_COUNT)) {
            mark_first_bad(trace, "deployed_state");
            break;
        }

        if ((update_idx + 1) % R394_CHECKPOINT_INTERVAL == 0) {
            float eval_loss = 0.0f;
            float hidden_summary = 0.0f;
            int flips;

            project_shadow_to_code(shadow, code);
            flips = count_code_flips(prev_checkpoint_code, code);
            evaluate_final_objective(data, code, &eval_loss, &hidden_summary,
                                     trace, "eval_rho1_loss", "hidden_summary");
            if (!trace->finite_pass) {
                break;
            }

            trace->checkpoint_losses[checkpoint_idx] = eval_loss;
            trace->checkpoint_hidden[checkpoint_idx] = hidden_summary;
            trace->checkpoint_flips[checkpoint_idx] = flips;
            trace->checkpoint_rhos[checkpoint_idx] = rho;
            trace->checkpoint_transition_jumps[checkpoint_idx] =
                (checkpoint_idx == 0) ? 0.0f : (eval_loss - trace->checkpoint_losses[checkpoint_idx - 1]);

            if (checkpoint_idx > 0 &&
                trace->checkpoint_losses[checkpoint_idx - 1] - eval_loss > R394_DIAG_PROGRESS_FLOOR &&
                flips > 0) {
                trace->useful_flip_count += flips;
            }

            memcpy(prev_checkpoint_code, code, sizeof(prev_checkpoint_code));
            ++checkpoint_idx;
        }
    }

    trace->checkpoint_count = (size_t)checkpoint_idx;
    if (trace->checkpoint_count > 0) {
        trace->loss_end = trace->checkpoint_losses[trace->checkpoint_count - 1];
    } else {
        trace->loss_end = trace->loss_start;
    }
}

static const char *classify_endpoint_transfer(
    const DiagnosticTrace *baseline,
    const DiagnosticTrace *ssh,
    float *delta_vs_baseline
) {
    const float baseline_improvement = baseline->loss_start - baseline->loss_end;
    const float ssh_improvement = ssh->loss_start - ssh->loss_end;

    *delta_vs_baseline = ssh->loss_end - baseline->loss_end;

    if (fabsf(baseline_improvement) < R394_DIAG_PROGRESS_FLOOR &&
        fabsf(ssh_improvement) < R394_DIAG_PROGRESS_FLOOR) {
        return "inconclusive";
    }
    if (*delta_vs_baseline <= R394_DIAG_ENDPOINT_TOL &&
        ssh_improvement >= -R394_DIAG_ENDPOINT_TOL) {
        return "pass";
    }
    return "fail";
}

static const char *classify_ternary_progress(
    const DiagnosticTrace *baseline,
    const DiagnosticTrace *ssh,
    float *ternary_progress
) {
    const float baseline_improvement = baseline->loss_start - baseline->loss_end;
    const float ssh_improvement = ssh->loss_start - ssh->loss_end;

    *ternary_progress = 0.0f;

    if (ssh->useful_flip_count > 0 &&
        ssh_improvement > R394_DIAG_PROGRESS_FLOOR) {
        *ternary_progress =
            ssh_improvement / (float)(ssh->useful_flip_count > 0 ? ssh->useful_flip_count : 1);
        return "pass";
    }

    if (fabsf(baseline_improvement) < R394_DIAG_PROGRESS_FLOOR &&
        fabsf(ssh_improvement) < R394_DIAG_PROGRESS_FLOOR &&
        ssh->useful_flip_count == 0) {
        return "inconclusive";
    }

    return "fail";
}

static const char *classify_transition_stability(const DiagnosticTrace *ssh) {
    size_t idx;

    if (ssh->checkpoint_count < 2) {
        return "inconclusive";
    }

    for (idx = 1; idx < ssh->checkpoint_count; ++idx) {
        if (ssh->checkpoint_rhos[idx] > ssh->checkpoint_rhos[idx - 1]) {
            const float prev_loss = ssh->checkpoint_losses[idx - 1];
            const float jump = ssh->checkpoint_losses[idx] - prev_loss;

            if (jump > R394_DIAG_JUMP_TOL) {
                if (idx + 1 >= ssh->checkpoint_count ||
                    ssh->checkpoint_losses[idx + 1] > prev_loss + R394_DIAG_ENDPOINT_TOL) {
                    return "fail";
                }
            }
        }
    }

    return "pass";
}

static const char *classify_restart_signature(const DiagnosticTrace *ssh) {
    size_t idx;

    if (ssh->checkpoint_count < 2) {
        return "inconclusive";
    }

    for (idx = 1; idx < ssh->checkpoint_count; ++idx) {
        if (fabsf(ssh->checkpoint_rhos[idx] - 1.0f) < 1.0e-6f &&
            ssh->checkpoint_rhos[idx - 1] < 1.0f) {
            const float loss_jump = ssh->checkpoint_losses[idx] - ssh->checkpoint_losses[idx - 1];
            const float hidden_jump =
                fabsf(ssh->checkpoint_hidden[idx] - ssh->checkpoint_hidden[idx - 1]);

            if ((loss_jump > R394_DIAG_JUMP_TOL || hidden_jump > R394_DIAG_HIDDEN_JUMP_TOL) &&
                (idx + 1 >= ssh->checkpoint_count ||
                 ssh->checkpoint_losses[idx + 1] >
                     ssh->checkpoint_losses[idx - 1] + R394_DIAG_ENDPOINT_TOL ||
                 fabsf(ssh->checkpoint_hidden[idx + 1] - ssh->checkpoint_hidden[idx - 1]) >
                     R394_DIAG_HIDDEN_JUMP_TOL)) {
                return "fail";
            }
            return "pass";
        }
    }

    return "inconclusive";
}

static const char *classify_branch_decision(
    const char *finite_state,
    const char *transition_stability,
    const char *endpoint_transfer,
    const char *ternary_progress_state,
    const char *restart_signature
) {
    if (strcmp(finite_state, "pass") != 0 ||
        strcmp(transition_stability, "fail") == 0 ||
        strcmp(endpoint_transfer, "fail") == 0 ||
        strcmp(ternary_progress_state, "fail") == 0 ||
        strcmp(restart_signature, "fail") == 0) {
        return "kill";
    }

    if (strcmp(transition_stability, "pass") == 0 &&
        strcmp(endpoint_transfer, "pass") == 0 &&
        strcmp(ternary_progress_state, "pass") == 0 &&
        strcmp(restart_signature, "pass") == 0) {
        return "pass";
    }

    return "inconclusive";
}

int main(void) {
    static const float ssh_path[R394_BRANCH_RHO_COUNT] = {0.25f, 0.50f, 0.75f, 1.00f};
    static const float baseline_path[R394_BRANCH_RHO_COUNT] = {1.00f, 1.00f, 1.00f, 1.00f};

    ProbeConfig config;
    float layer_a[V12_N_LAYERS][V12_D_STATE];
    LayerSpectrum spectra[V12_N_LAYERS];
    float a_rho[V12_D_STATE];
    float layer_baseline[V12_N_LAYERS][R394_MAX_HORIZONS];
    float global_baseline[R394_MAX_HORIZONS];
    float global_ratios[R394_MAX_RHOS][R394_MAX_HORIZONS];
    float best_ratio = INFINITY;
    float best_rho = 1.0f;
    int worst_horizon = 0;
    int worst_limiting_layer = -1;
    size_t rho_idx;
    size_t horizon_idx;
    int layer_idx;
    DiagnosticData data;
    DiagnosticTrace baseline_trace;
    DiagnosticTrace ssh_trace;
    unsigned int order[R394_TOTAL_UPDATES * R394_BATCH_SIZE];
    float delta_vs_baseline = 0.0f;
    float ternary_progress = 0.0f;
    const char *finite_state = "pass";
    const char *endpoint_transfer;
    const char *ternary_progress_state;
    const char *transition_stability;
    const char *restart_signature;
    const char *branch_decision;

    build_config(&config);
    memset(&data, 0, sizeof(data));

    for (layer_idx = 0; layer_idx < V12_N_LAYERS; ++layer_idx) {
        initialize_layer_a(layer_a[layer_idx]);
        compute_layer_spectrum(layer_a[layer_idx], &spectra[layer_idx]);
    }

    for (horizon_idx = 0; horizon_idx < config.n_horizons; ++horizon_idx) {
        float global_kappa = 0.0f;

        for (layer_idx = 0; layer_idx < V12_N_LAYERS; ++layer_idx) {
            materialize_homotopy_a(&spectra[layer_idx], 1.0f, a_rho);
            layer_baseline[layer_idx][horizon_idx] =
                compute_kappa_eff(a_rho, config.horizons[horizon_idx]);
            if (layer_baseline[layer_idx][horizon_idx] > global_kappa) {
                global_kappa = layer_baseline[layer_idx][horizon_idx];
            }
        }

        global_baseline[horizon_idx] = global_kappa;
    }

    for (rho_idx = 0; rho_idx < config.n_rhos; ++rho_idx) {
        const float rho = config.rhos[rho_idx];
        float rho_worst_ratio = 0.0f;
        int rho_worst_horizon = 0;
        int rho_worst_limiting_layer = -1;

        for (horizon_idx = 0; horizon_idx < config.n_horizons; ++horizon_idx) {
            float global_kappa = 0.0f;
            int global_layer = 0;

            for (layer_idx = 0; layer_idx < V12_N_LAYERS; ++layer_idx) {
                float layer_kappa;
                float layer_ratio;

                materialize_homotopy_a(&spectra[layer_idx], rho, a_rho);
                layer_kappa = compute_kappa_eff(a_rho, config.horizons[horizon_idx]);
                layer_ratio = layer_kappa /
                              fmaxf(R394_SSH_EPS, layer_baseline[layer_idx][horizon_idx]);

                printf(
                    "[R394_SSH] rho=%.2f H=%d layer=%d kappa_eff=%.9g r_kappa=%.9g\n",
                    rho,
                    config.horizons[horizon_idx],
                    layer_idx,
                    layer_kappa,
                    layer_ratio
                );

                if (layer_kappa > global_kappa) {
                    global_kappa = layer_kappa;
                    global_layer = layer_idx;
                }
            }

            global_ratios[rho_idx][horizon_idx] =
                global_kappa / fmaxf(R394_SSH_EPS, global_baseline[horizon_idx]);

            if (rho < 1.0f && global_ratios[rho_idx][horizon_idx] > rho_worst_ratio) {
                rho_worst_ratio = global_ratios[rho_idx][horizon_idx];
                rho_worst_horizon = config.horizons[horizon_idx];
                rho_worst_limiting_layer = global_layer;
            }

            printf(
                "[R394_SSH] global H=%d rho=%.2f kappa_eff=%.9g r_kappa=%.9g limiting_layer=%d\n",
                config.horizons[horizon_idx],
                rho,
                global_kappa,
                global_ratios[rho_idx][horizon_idx],
                global_layer
            );
        }

        if (rho < 1.0f && rho_worst_ratio < best_ratio + 1.0e-12f) {
            best_ratio = rho_worst_ratio;
            best_rho = rho;
            worst_horizon = rho_worst_horizon;
            worst_limiting_layer = rho_worst_limiting_layer;
        }
    }

    if (!isfinite(best_ratio)) {
        best_ratio = 1.0f;
    }

    printf(
        "[R394_SSH] metric_decision=%s r_kappa_best=%.9g rho_best=%.2f horizon_worst=%d limiting_layer=%d\n",
        classify_metric(best_ratio),
        best_ratio,
        best_rho,
        worst_horizon,
        worst_limiting_layer
    );
    printf("[R394_SSH] branch_stability=not_measured_slice1\n");
    printf("[R394_SSH] branch_no_final_loss_jump=not_measured_slice1\n");
    printf("[R394_SSH] branch_no_nan_inf_or_hidden_state_divergence=not_measured_slice1\n");
    printf("[R394_SSH] branch_no_persistent_mode_collapse_or_role_reordering=not_measured_slice1\n");
    printf("[R394_SSH] branch_no_equalized_budget_degradation_vs_direct=not_measured_slice1\n");
    printf("[R394_SSH] branch_endpoint_transfer_without_restart_discontinuity=not_measured_slice1\n");
    printf(
        "[R394_SSH] ternary_progress_formula=final_objective_loss_improvement/max(1,useful_ternary_flip_count)\n"
    );
    printf("[R394_SSH] ternary_progress_proxy=not_measured_static_only\n");

    build_diagnostic_data(spectra, &data);
    if (!data.basis[0] || !data.basis[1] || !data.basis[2] || !data.basis[3]) {
        fprintf(stderr, "r394_ssh_probe: failed to allocate diagnostic buffers\n");
        free_diagnostic_data(&data);
        return 1;
    }

    build_batch_order(order);
    run_branch_path(&data, baseline_path, order, &baseline_trace);
    run_branch_path(&data, ssh_path, order, &ssh_trace);

    if (!baseline_trace.finite_pass) {
        finite_state = "fail";
        if (ssh_trace.finite_pass) {
            snprintf(ssh_trace.first_bad, sizeof(ssh_trace.first_bad),
                     "baseline_%s", baseline_trace.first_bad[0] ? baseline_trace.first_bad : "unknown");
        }
    }
    if (!ssh_trace.finite_pass) {
        finite_state = "fail";
    }

    printf(
        "[R394_SSH_BRANCH] baseline final_loss_start=%.9g final_loss_end=%.9g\n",
        baseline_trace.loss_start,
        baseline_trace.loss_end
    );

    for (rho_idx = 0; rho_idx < ssh_trace.checkpoint_count; ++rho_idx) {
        printf(
            "[R394_SSH_BRANCH] ssh rho=%.2f checkpoint=%d eval_rho1_loss=%.9g transition_jump=%.9g\n",
            ssh_trace.checkpoint_rhos[rho_idx],
            (int)rho_idx + 1,
            ssh_trace.checkpoint_losses[rho_idx],
            ssh_trace.checkpoint_transition_jumps[rho_idx]
        );
    }

    printf(
        "[R394_SSH_BRANCH] finite_state=%s first_bad=%s\n",
        finite_state,
        (finite_state[0] == 'p')
            ? "none"
            : (ssh_trace.finite_pass
                   ? (baseline_trace.first_bad[0] ? baseline_trace.first_bad : "unknown")
                   : (ssh_trace.first_bad[0] ? ssh_trace.first_bad : "unknown"))
    );

    transition_stability = classify_transition_stability(&ssh_trace);
    endpoint_transfer = classify_endpoint_transfer(&baseline_trace, &ssh_trace, &delta_vs_baseline);
    ternary_progress_state = classify_ternary_progress(&baseline_trace, &ssh_trace, &ternary_progress);
    restart_signature = classify_restart_signature(&ssh_trace);
    branch_decision = classify_branch_decision(
        finite_state,
        transition_stability,
        endpoint_transfer,
        ternary_progress_state,
        restart_signature
    );

    printf(
        "[R394_SSH_BRANCH] endpoint_transfer=%s delta_vs_baseline=%.9g\n",
        endpoint_transfer,
        delta_vs_baseline
    );
    printf(
        "[R394_SSH_BRANCH] useful_ternary_flip_count=%d\n",
        ssh_trace.useful_flip_count
    );
    if (strcmp(ternary_progress_state, "pass") == 0) {
        printf("[R394_SSH_BRANCH] ternary_progress=%.9g\n", ternary_progress);
    } else {
        printf("[R394_SSH_BRANCH] ternary_progress=%s\n", ternary_progress_state);
    }
    printf("[R394_SSH_BRANCH] transition_stability=%s\n", transition_stability);
    printf("[R394_SSH_BRANCH] restart_signature=%s\n", restart_signature);
    printf("[R394_SSH_BRANCH] branch_progress_decision=%s\n", branch_decision);

    free_diagnostic_data(&data);
    return 0;
}
