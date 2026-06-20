#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define R394_BCT_MAX_STATE 24
#define R394_BCT_MAX_THETA 32
#define R394_BCT_MAX_SUFFIX 64
#define R394_BCT_MAX_VOCAB 16
#define R394_BCT_EPS 1.0e-8f
#define R394_BCT_EPS_COS 1.0e-4f
#define R394_BCT_EPS_M 1.0e-3f

typedef struct {
    int batches;
    int batch_size;
    int boundaries;
    int suffix_window;
    int base_window;
    int dim_state;
    int dim_theta;
    int vocab;
    int timing_repeats;
    float eta;
    float rho_state;
    float eps_step;
} ProbeConfig;

typedef struct {
    int batch_idx;
    int slot_idx;
    int boundary_idx;
    int block_id;
    float theta[R394_BCT_MAX_THETA];
    float prefix[R394_BCT_MAX_STATE];
    float a[R394_BCT_MAX_STATE * R394_BCT_MAX_THETA];
    float w[R394_BCT_MAX_SUFFIX * R394_BCT_MAX_VOCAB * R394_BCT_MAX_STATE];
    float bias[R394_BCT_MAX_SUFFIX * R394_BCT_MAX_VOCAB];
    int targets[R394_BCT_MAX_SUFFIX];
} BoundarySample;

typedef struct {
    int valid;
    int ratio_valid;
    int clipped;
    int heavy_clip;
    float clip_factor;
    float cos_bct;
    float cos_base;
    float delta_cos;
    float ratio_cos;
    float relative_delta;
    float proj_bct;
    float proj_base;
    float delta_d_bct;
    float delta_d_base;
    float predicted_bct;
    float transfer_ratio;
    float time_bct;
    float time_base;
    float d_before;
} BoundaryMetrics;

typedef struct {
    float median_delta_cos;
    float median_ratio_cos;
    float median_relative_delta;
    float median_proj_ratio;
    float median_transfer_ratio;
    float kappa;
    float r_wall;
    float sign_flip_base;
    float sign_flip_bct;
    float invalid_rate;
    float skip_rate;
    float clip_rate;
    float heavy_clip_rate;
    float transfer_risk;
    float delta_cos_iqr;
    float temp_memory_mb;
    int valid_count;
    int ratio_valid_count;
    int invalid_count;
    int skipped_count;
    int clipped_count;
    int heavy_clipped_count;
    int sign_bad_base_count;
    int sign_bad_bct_count;
    int transfer_bad_count;
    int total_samples;
    int suffix_product_count;
    int solve_count;
    int sync_count;
    int materialization_count;
    const char *decision;
    const char *reason;
} ProbeSummary;

static int parse_env_int(const char *name, int default_value, int min_value, int max_value) {
    const char *text = getenv(name);
    long value;
    char *end = NULL;
    if (!text || !*text) {
        return default_value;
    }
    value = strtol(text, &end, 10);
    if (end == text) {
        return default_value;
    }
    if (value < min_value) {
        value = min_value;
    }
    if (value > max_value) {
        value = max_value;
    }
    return (int)value;
}

static float parse_env_float(const char *name, float default_value, float min_value, float max_value) {
    const char *text = getenv(name);
    float value;
    char *end = NULL;
    if (!text || !*text) {
        return default_value;
    }
    value = (float)strtod(text, &end);
    if (end == text) {
        return default_value;
    }
    if (value < min_value) {
        value = min_value;
    }
    if (value > max_value) {
        value = max_value;
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

static float percentile_from_sorted(const float *values, int n, float q) {
    int lo;
    int hi;
    float pos;
    float frac;
    if (n <= 0) {
        return 0.0f;
    }
    if (n == 1) {
        return values[0];
    }
    if (q < 0.0f) {
        q = 0.0f;
    }
    if (q > 1.0f) {
        q = 1.0f;
    }
    pos = q * (float)(n - 1);
    lo = (int)floorf(pos);
    hi = (int)ceilf(pos);
    if (hi >= n) {
        hi = n - 1;
    }
    frac = pos - (float)lo;
    return values[lo] + frac * (values[hi] - values[lo]);
}

static float median_from_values(const float *values, int n) {
    float *copy;
    float result;
    if (n <= 0) {
        return 0.0f;
    }
    copy = (float *)malloc((size_t)n * sizeof(float));
    if (!copy) {
        return 0.0f;
    }
    memcpy(copy, values, (size_t)n * sizeof(float));
    qsort(copy, (size_t)n, sizeof(float), compare_float_asc);
    result = percentile_from_sorted(copy, n, 0.5f);
    free(copy);
    return result;
}

static float iqr_from_values(const float *values, int n) {
    float *copy;
    float q1;
    float q3;
    if (n <= 1) {
        return 0.0f;
    }
    copy = (float *)malloc((size_t)n * sizeof(float));
    if (!copy) {
        return 0.0f;
    }
    memcpy(copy, values, (size_t)n * sizeof(float));
    qsort(copy, (size_t)n, sizeof(float), compare_float_asc);
    q1 = percentile_from_sorted(copy, n, 0.25f);
    q3 = percentile_from_sorted(copy, n, 0.75f);
    free(copy);
    return q3 - q1;
}

static float pseudo_unit(int sample_id, int slot) {
    const float x = sinf(0.131f * (float)(sample_id + 1) * (float)(slot + 3)) +
                    cosf(0.173f * (float)(sample_id + 5) * (float)(slot + 1));
    return 0.5f * x;
}

static void build_config(ProbeConfig *config) {
    config->batches = parse_env_int("R394_BCT_BATCHES", 2, 1, 16);
    config->batch_size = parse_env_int("R394_BCT_BATCH_SIZE", 4, 1, 8);
    config->boundaries = parse_env_int("R394_BCT_BOUNDARIES", 8, 1, 32);
    config->suffix_window = parse_env_int("R394_BCT_SUFFIX_WINDOW", 16, 2, R394_BCT_MAX_SUFFIX);
    config->base_window = parse_env_int("R394_BCT_BASE_WINDOW", config->suffix_window / 4, 1, R394_BCT_MAX_SUFFIX);
    config->dim_state = parse_env_int("R394_BCT_DIM_STATE", 8, 2, R394_BCT_MAX_STATE);
    config->dim_theta = parse_env_int("R394_BCT_DIM_THETA", 12, 2, R394_BCT_MAX_THETA);
    config->vocab = parse_env_int("R394_BCT_VOCAB", 7, 3, R394_BCT_MAX_VOCAB);
    config->timing_repeats = parse_env_int("R394_BCT_TIMING_REPEATS", 64, 1, 1024);
    config->eta = parse_env_float("R394_BCT_ETA", 0.22f, 0.01f, 4.0f);
    config->rho_state = parse_env_float("R394_BCT_RHO_STATE", 0.11f, 0.001f, 4.0f);
    config->eps_step = parse_env_float("R394_BCT_EPS_STEP", 0.035f, 0.001f, 1.0f);
    if (config->base_window > config->suffix_window) {
        config->base_window = config->suffix_window;
    }
}

static int total_boundary_samples(const ProbeConfig *config) {
    return config->batches * config->batch_size * config->boundaries;
}

static void build_sample(const ProbeConfig *config, int sample_id, int batch_idx, int slot_idx, int boundary_idx, BoundarySample *sample) {
    int i;
    int j;
    int e;
    int v;
    sample->batch_idx = batch_idx;
    sample->slot_idx = slot_idx;
    sample->boundary_idx = boundary_idx;
    sample->block_id = sample_id % 2;

    for (j = 0; j < config->dim_theta; ++j) {
        sample->theta[j] = 0.17f * pseudo_unit(sample_id, j) + 0.03f * (float)(sample->block_id + 1);
    }
    for (i = 0; i < config->dim_state; ++i) {
        sample->prefix[i] = 0.11f * pseudo_unit(sample_id + 13, i);
        for (j = 0; j < config->dim_theta; ++j) {
            const int idx = i * config->dim_theta + j;
            sample->a[idx] = 0.10f * pseudo_unit(sample_id + 29, idx) +
                             0.06f * sinf(0.27f * (float)(i + 1) * (float)(j + 2));
        }
    }
    for (e = 0; e < config->suffix_window; ++e) {
        sample->targets[e] = (sample_id + 2 * e + sample->block_id) % config->vocab;
        for (v = 0; v < config->vocab; ++v) {
            const int bias_idx = e * config->vocab + v;
            sample->bias[bias_idx] = 0.07f * pseudo_unit(sample_id + 47, bias_idx) +
                                     0.02f * (float)(v == sample->targets[e]);
            for (i = 0; i < config->dim_state; ++i) {
                const int w_idx = (e * config->vocab + v) * config->dim_state + i;
                sample->w[w_idx] = 0.12f * pseudo_unit(sample_id + 71, w_idx) +
                                   0.08f * sinf(0.19f * (float)(e + 1) * (float)(i + 1)) +
                                   0.05f * cosf(0.23f * (float)(v + 1) * (float)(i + 2));
                if (v == sample->targets[e]) {
                    sample->w[w_idx] += 0.06f * (float)(1 + (i % 3));
                }
            }
        }
    }
}

static void compute_h_b(const ProbeConfig *config, const BoundarySample *sample, float *h_b) {
    int i;
    int j;
    for (i = 0; i < config->dim_state; ++i) {
        float value = sample->prefix[i];
        for (j = 0; j < config->dim_theta; ++j) {
            value += sample->a[i * config->dim_theta + j] * sample->theta[j];
        }
        h_b[i] = value;
    }
}

static float compute_objective_and_lambda(
    const ProbeConfig *config,
    const BoundarySample *sample,
    const float *state,
    int endpoints,
    float *lambda_out
) {
    float logits[R394_BCT_MAX_VOCAB];
    float probs[R394_BCT_MAX_VOCAB];
    float total_ce = 0.0f;
    int i;
    int e;
    int v;
    memset(lambda_out, 0, (size_t)config->dim_state * sizeof(float));
    for (e = 0; e < endpoints; ++e) {
        float max_logit = -INFINITY;
        float sum_exp = 0.0f;
        for (v = 0; v < config->vocab; ++v) {
            float logit = sample->bias[e * config->vocab + v];
            for (i = 0; i < config->dim_state; ++i) {
                logit += sample->w[(e * config->vocab + v) * config->dim_state + i] * state[i];
            }
            logits[v] = logit;
            if (logit > max_logit) {
                max_logit = logit;
            }
        }
        for (v = 0; v < config->vocab; ++v) {
            probs[v] = expf(logits[v] - max_logit);
            sum_exp += probs[v];
        }
        if (sum_exp < R394_BCT_EPS) {
            sum_exp = R394_BCT_EPS;
        }
        for (v = 0; v < config->vocab; ++v) {
            float grad_logit;
            probs[v] /= sum_exp;
            grad_logit = probs[v] - (v == sample->targets[e] ? 1.0f : 0.0f);
            grad_logit /= (float)endpoints;
            for (i = 0; i < config->dim_state; ++i) {
                lambda_out[i] += sample->w[(e * config->vocab + v) * config->dim_state + i] * grad_logit;
            }
        }
        total_ce += -logf(fmaxf(probs[sample->targets[e]], R394_BCT_EPS));
    }
    return total_ce / (float)endpoints;
}

static void state_metric_from_h(const ProbeConfig *config, const float *h_b, float *metric_diag) {
    int i;
    for (i = 0; i < config->dim_state; ++i) {
        metric_diag[i] = 1.0f / (h_b[i] * h_b[i] + R394_BCT_EPS_M);
    }
}

static float construct_target(
    const ProbeConfig *config,
    const float *h_b,
    const float *lambda,
    float *metric_diag,
    float *delta_out,
    float *h_target
) {
    float norm_metric = 0.0f;
    float clip_factor;
    int i;
    for (i = 0; i < config->dim_state; ++i) {
        const float metric_inv = 1.0f / metric_diag[i];
        delta_out[i] = -config->eta * metric_inv * lambda[i];
        norm_metric += delta_out[i] * delta_out[i] * metric_diag[i];
    }
    norm_metric = sqrtf(norm_metric);
    clip_factor = 1.0f;
    if (norm_metric > config->rho_state) {
        clip_factor = config->rho_state / (norm_metric + R394_BCT_EPS);
    }
    for (i = 0; i < config->dim_state; ++i) {
        delta_out[i] *= clip_factor;
        h_target[i] = h_b[i] + delta_out[i];
    }
    return clip_factor;
}

static void project_state_direction_to_theta(
    const ProbeConfig *config,
    const BoundarySample *sample,
    const float *state_direction,
    float scale,
    float *theta_direction
) {
    int i;
    int j;
    for (j = 0; j < config->dim_theta; ++j) {
        float value = 0.0f;
        for (i = 0; i < config->dim_state; ++i) {
            value += sample->a[i * config->dim_theta + j] * state_direction[i];
        }
        theta_direction[j] = scale * value;
    }
}

static float dot_product(const float *lhs, const float *rhs, int n) {
    float total = 0.0f;
    int i;
    for (i = 0; i < n; ++i) {
        total += lhs[i] * rhs[i];
    }
    return total;
}

static float norm_l2(const float *values, int n) {
    return sqrtf(fmaxf(dot_product(values, values, n), 0.0f));
}

static int all_finite(const float *values, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        if (!isfinite(values[i])) {
            return 0;
        }
    }
    return 1;
}

static void compute_theta_to_state_shift(
    const ProbeConfig *config,
    const BoundarySample *sample,
    const float *theta_step,
    float *shift_out
) {
    int i;
    int j;
    for (i = 0; i < config->dim_state; ++i) {
        float value = 0.0f;
        for (j = 0; j < config->dim_theta; ++j) {
            value += sample->a[i * config->dim_theta + j] * theta_step[j];
        }
        shift_out[i] = value;
    }
}

static float compute_ce_only(const ProbeConfig *config, const BoundarySample *sample, const float *state) {
    float lambda_dummy[R394_BCT_MAX_STATE];
    return compute_objective_and_lambda(config, sample, state, config->suffix_window, lambda_dummy);
}

static void collect_boundary_metrics(
    const ProbeConfig *config,
    const BoundarySample *sample,
    BoundaryMetrics *metrics
) {
    float h_b[R394_BCT_MAX_STATE];
    float lambda_ref[R394_BCT_MAX_STATE];
    float lambda_base[R394_BCT_MAX_STATE];
    float metric_diag[R394_BCT_MAX_STATE];
    float delta_state[R394_BCT_MAX_STATE];
    float h_target[R394_BCT_MAX_STATE];
    float state_gap[R394_BCT_MAX_STATE];
    float weighted_gap[R394_BCT_MAX_STATE];
    float u_ref[R394_BCT_MAX_THETA];
    float u_base[R394_BCT_MAX_THETA];
    float u_bct[R394_BCT_MAX_THETA];
    float ref_norm;
    float base_norm;
    float bct_norm;
    float dot_base_ref;
    float dot_bct_ref;
    float d_before;
    float d_after_bct;
    float d_after_base;
    float predicted_bct;
    int i;

    memset(metrics, 0, sizeof(*metrics));
    compute_h_b(config, sample, h_b);
    d_before = compute_objective_and_lambda(config, sample, h_b, config->suffix_window, lambda_ref);
    (void)compute_objective_and_lambda(config, sample, h_b, config->base_window, lambda_base);
    state_metric_from_h(config, h_b, metric_diag);
    metrics->clip_factor = construct_target(config, h_b, lambda_ref, metric_diag, delta_state, h_target);
    metrics->clipped = metrics->clip_factor < 0.999f;
    metrics->heavy_clip = metrics->clip_factor < 0.5f;

    for (i = 0; i < config->dim_state; ++i) {
        state_gap[i] = h_b[i] - h_target[i];
        weighted_gap[i] = metric_diag[i] * state_gap[i];
    }
    project_state_direction_to_theta(config, sample, lambda_ref, -1.0f, u_ref);
    project_state_direction_to_theta(config, sample, lambda_base, -1.0f, u_base);
    project_state_direction_to_theta(config, sample, weighted_gap, -1.0f / config->eta, u_bct);

    if (!all_finite(u_ref, config->dim_theta) ||
        !all_finite(u_base, config->dim_theta) ||
        !all_finite(u_bct, config->dim_theta)) {
        metrics->valid = 0;
        return;
    }

    ref_norm = norm_l2(u_ref, config->dim_theta);
    base_norm = norm_l2(u_base, config->dim_theta);
    bct_norm = norm_l2(u_bct, config->dim_theta);
    if (ref_norm < R394_BCT_EPS || base_norm < R394_BCT_EPS || bct_norm < R394_BCT_EPS) {
        metrics->valid = 0;
        return;
    }

    dot_base_ref = dot_product(u_base, u_ref, config->dim_theta);
    dot_bct_ref = dot_product(u_bct, u_ref, config->dim_theta);
    metrics->cos_base = dot_base_ref / (base_norm * ref_norm + R394_BCT_EPS);
    metrics->cos_bct = dot_bct_ref / (bct_norm * ref_norm + R394_BCT_EPS);
    metrics->delta_cos = metrics->cos_bct - metrics->cos_base;
    metrics->ratio_valid = metrics->cos_base > R394_BCT_EPS_COS;
    if (metrics->ratio_valid) {
        metrics->ratio_cos = metrics->cos_bct / fmaxf(metrics->cos_base, R394_BCT_EPS_COS);
    }
    metrics->relative_delta = metrics->delta_cos / fmaxf(fabsf(metrics->cos_base), R394_BCT_EPS_COS);
    metrics->proj_base = dot_base_ref / (base_norm + R394_BCT_EPS);
    metrics->proj_bct = dot_bct_ref / (bct_norm + R394_BCT_EPS);
    metrics->d_before = d_before;

    d_after_bct = compute_ce_only(config, sample, h_b);
    metrics->time_bct = 0.0f;
    metrics->time_base = 0.0f;
    {
        float elapsed_bct = 0.0f;
        float elapsed_base = 0.0f;
        float theta_norm_bct = norm_l2(u_bct, config->dim_theta);
        float theta_norm_base = norm_l2(u_base, config->dim_theta);
        float theta_step[R394_BCT_MAX_THETA];
        float state_shift[R394_BCT_MAX_STATE];
        float h_after[R394_BCT_MAX_STATE];

        if (theta_norm_bct > R394_BCT_EPS) {
            for (i = 0; i < config->dim_theta; ++i) {
                theta_step[i] = config->eps_step * u_bct[i] / theta_norm_bct;
            }
            compute_theta_to_state_shift(config, sample, theta_step, state_shift);
            for (i = 0; i < config->dim_state; ++i) {
                h_after[i] = h_b[i] + state_shift[i];
            }
            {
                clock_t start_clock = clock();
                d_after_bct = compute_ce_only(config, sample, h_after);
                clock_t end_clock = clock();
                elapsed_bct = (float)(end_clock - start_clock) / (float)CLOCKS_PER_SEC;
            }
        }
        if (theta_norm_base > R394_BCT_EPS) {
            for (i = 0; i < config->dim_theta; ++i) {
                theta_step[i] = config->eps_step * u_base[i] / theta_norm_base;
            }
            compute_theta_to_state_shift(config, sample, theta_step, state_shift);
            for (i = 0; i < config->dim_state; ++i) {
                h_after[i] = h_b[i] + state_shift[i];
            }
            {
                clock_t start_clock = clock();
                d_after_base = compute_ce_only(config, sample, h_after);
                clock_t end_clock = clock();
                elapsed_base = (float)(end_clock - start_clock) / (float)CLOCKS_PER_SEC;
            }
        } else {
            d_after_base = d_before;
        }
        metrics->time_bct = elapsed_bct;
        metrics->time_base = elapsed_base;
    }

    metrics->delta_d_bct = d_before - d_after_bct;
    metrics->delta_d_base = d_before - d_after_base;
    predicted_bct = config->eps_step * dot_bct_ref / (bct_norm + R394_BCT_EPS);
    metrics->predicted_bct = predicted_bct;
    metrics->transfer_ratio = fmaxf(metrics->delta_d_bct, 0.0f) / fmaxf(predicted_bct, R394_BCT_EPS);
    metrics->valid = isfinite(metrics->cos_base) &&
                     isfinite(metrics->cos_bct) &&
                     isfinite(metrics->delta_d_bct) &&
                     isfinite(metrics->delta_d_base);
}

static float seconds_since(clock_t start_clock, clock_t end_clock) {
    return (float)(end_clock - start_clock) / (float)CLOCKS_PER_SEC;
}

static void measure_batch_times(
    const ProbeConfig *config,
    const BoundarySample *samples,
    int batch_idx,
    float *baseline_time_out,
    float *target_time_out
) {
    float sink = 0.0f;
    int repeat_idx;
    int slot_idx;
    int boundary_idx;
    int sample_base = batch_idx * config->batch_size * config->boundaries;
    clock_t start_clock;
    clock_t end_clock;

    start_clock = clock();
    for (repeat_idx = 0; repeat_idx < config->timing_repeats; ++repeat_idx) {
        for (slot_idx = 0; slot_idx < config->batch_size; ++slot_idx) {
            for (boundary_idx = 0; boundary_idx < config->boundaries; ++boundary_idx) {
                float h_b[R394_BCT_MAX_STATE];
                float lambda_base[R394_BCT_MAX_STATE];
                const BoundarySample *sample = &samples[sample_base + slot_idx * config->boundaries + boundary_idx];
                compute_h_b(config, sample, h_b);
                sink += compute_objective_and_lambda(config, sample, h_b, config->base_window, lambda_base);
                sink += lambda_base[0];
            }
        }
    }
    end_clock = clock();
    *baseline_time_out = seconds_since(start_clock, end_clock) / (float)config->timing_repeats;

    start_clock = clock();
    for (repeat_idx = 0; repeat_idx < config->timing_repeats; ++repeat_idx) {
        for (slot_idx = 0; slot_idx < config->batch_size; ++slot_idx) {
            for (boundary_idx = 0; boundary_idx < config->boundaries; ++boundary_idx) {
                float h_b[R394_BCT_MAX_STATE];
                float lambda_ref[R394_BCT_MAX_STATE];
                float metric_diag[R394_BCT_MAX_STATE];
                float delta_state[R394_BCT_MAX_STATE];
                float h_target[R394_BCT_MAX_STATE];
                const BoundarySample *sample = &samples[sample_base + slot_idx * config->boundaries + boundary_idx];
                compute_h_b(config, sample, h_b);
                sink += compute_objective_and_lambda(config, sample, h_b, config->suffix_window, lambda_ref);
                state_metric_from_h(config, h_b, metric_diag);
                sink += construct_target(config, h_b, lambda_ref, metric_diag, delta_state, h_target);
                sink += h_target[0];
            }
        }
    }
    end_clock = clock();
    *target_time_out = seconds_since(start_clock, end_clock) / (float)config->timing_repeats;

    if (sink == 123456.0f) {
        fprintf(stderr, "R394_BCT_INTERNAL sink=%f\n", sink);
    }
}

static void summarize_probe(
    const ProbeConfig *config,
    const BoundaryMetrics *metrics,
    int metrics_count,
    const float *batch_baseline_times,
    const float *batch_target_times,
    ProbeSummary *summary
) {
    float *delta_cos_values = (float *)malloc((size_t)metrics_count * sizeof(float));
    float *ratio_values = (float *)malloc((size_t)metrics_count * sizeof(float));
    float *relative_values = (float *)malloc((size_t)metrics_count * sizeof(float));
    float *proj_ratio_values = (float *)malloc((size_t)metrics_count * sizeof(float));
    float *transfer_values = (float *)malloc((size_t)metrics_count * sizeof(float));
    float *delta_bct_values = (float *)malloc((size_t)metrics_count * sizeof(float));
    float *delta_base_values = (float *)malloc((size_t)metrics_count * sizeof(float));
    float *time_bct_values = (float *)malloc((size_t)metrics_count * sizeof(float));
    float *time_base_values = (float *)malloc((size_t)metrics_count * sizeof(float));
    int delta_count = 0;
    int ratio_count = 0;
    int relative_count = 0;
    int proj_ratio_count = 0;
    int transfer_count = 0;
    int delta_bct_count = 0;
    int delta_base_count = 0;
    int time_bct_count = 0;
    int time_base_count = 0;
    int idx;

    memset(summary, 0, sizeof(*summary));
    summary->total_samples = metrics_count;
    summary->skip_rate = 0.0f;
    summary->temp_memory_mb =
        (float)((size_t)metrics_count * sizeof(BoundarySample) + sizeof(BoundaryMetrics) * (size_t)metrics_count) /
        (1024.0f * 1024.0f);

    for (idx = 0; idx < metrics_count; ++idx) {
        const BoundaryMetrics *item = &metrics[idx];
        if (!item->valid) {
            summary->invalid_count += 1;
            continue;
        }
        summary->valid_count += 1;
        summary->suffix_product_count += config->suffix_window;
        summary->solve_count += 1;
        summary->sync_count += 0;
        summary->materialization_count += 1;
        if (item->clipped) {
            summary->clipped_count += 1;
        }
        if (item->heavy_clip) {
            summary->heavy_clipped_count += 1;
        }
        if (item->proj_base <= 0.0f) {
            summary->sign_bad_base_count += 1;
        }
        if (item->proj_bct <= 0.0f) {
            summary->sign_bad_bct_count += 1;
        }
        if (item->delta_d_bct <= 0.0f || item->delta_d_bct <= item->delta_d_base) {
            summary->transfer_bad_count += 1;
        }

        delta_cos_values[delta_count++] = item->delta_cos;
        relative_values[relative_count++] = item->relative_delta;
        proj_ratio_values[proj_ratio_count++] = item->proj_bct / fmaxf(item->proj_base, R394_BCT_EPS);
        transfer_values[transfer_count++] = item->transfer_ratio;
        delta_bct_values[delta_bct_count++] = item->delta_d_bct;
        delta_base_values[delta_base_count++] = item->delta_d_base;
        if (item->time_bct > 0.0f) {
            time_bct_values[time_bct_count++] = item->time_bct;
        }
        if (item->time_base > 0.0f) {
            time_base_values[time_base_count++] = item->time_base;
        }
        if (item->ratio_valid) {
            ratio_values[ratio_count++] = item->ratio_cos;
        }
    }

    summary->ratio_valid_count = ratio_count;
    summary->median_delta_cos = median_from_values(delta_cos_values, delta_count);
    summary->delta_cos_iqr = iqr_from_values(delta_cos_values, delta_count);
    summary->median_ratio_cos = median_from_values(ratio_values, ratio_count);
    summary->median_relative_delta = median_from_values(relative_values, relative_count);
    summary->median_proj_ratio = median_from_values(proj_ratio_values, proj_ratio_count);
    summary->median_transfer_ratio = median_from_values(transfer_values, transfer_count);
    summary->kappa =
        median_from_values(batch_target_times, config->batches) /
        fmaxf(median_from_values(batch_baseline_times, config->batches), R394_BCT_EPS);
    summary->r_wall =
        (median_from_values(delta_bct_values, delta_bct_count) / fmaxf(median_from_values(time_bct_values, time_bct_count), R394_BCT_EPS)) /
        fmaxf(
            median_from_values(delta_base_values, delta_base_count) /
                fmaxf(median_from_values(time_base_values, time_base_count), R394_BCT_EPS),
            R394_BCT_EPS
        );
    summary->sign_flip_base = (float)summary->sign_bad_base_count / fmaxf((float)summary->valid_count, 1.0f);
    summary->sign_flip_bct = (float)summary->sign_bad_bct_count / fmaxf((float)summary->valid_count, 1.0f);
    summary->invalid_rate = (float)summary->invalid_count / fmaxf((float)metrics_count, 1.0f);
    summary->clip_rate = (float)summary->clipped_count / fmaxf((float)summary->valid_count, 1.0f);
    summary->heavy_clip_rate = (float)summary->heavy_clipped_count / fmaxf((float)summary->valid_count, 1.0f);
    summary->transfer_risk = (float)summary->transfer_bad_count / fmaxf((float)summary->valid_count, 1.0f);

    if (summary->valid_count == 0) {
        summary->decision = "kill";
        summary->reason = "no_valid_boundaries";
    } else if (summary->median_delta_cos < 0.05f) {
        summary->decision = "kill";
        summary->reason = "delta_cos_below_floor";
    } else if (ratio_count > 0 && summary->median_ratio_cos < 1.10f) {
        summary->decision = "kill";
        summary->reason = "relative_gain_below_floor";
    } else if (summary->kappa > 0.30f) {
        summary->decision = "kill";
        summary->reason = "kappa_above_kill";
    } else if (summary->sign_flip_bct > summary->sign_flip_base) {
        summary->decision = "kill";
        summary->reason = "sign_flip_regression";
    } else if (summary->transfer_risk > 0.50f) {
        summary->decision = "kill";
        summary->reason = "endpoint_transfer_failure";
    } else if (summary->median_delta_cos >= 0.10f &&
               (ratio_count == 0 || summary->median_ratio_cos >= 1.20f) &&
               summary->kappa <= 0.15f &&
               summary->r_wall >= 1.70f) {
        summary->decision = "pass";
        summary->reason = summary->r_wall >= 2.0f ? "strong_pass" : "pass_thresholds";
    } else {
        summary->decision = "weak";
        summary->reason = "conditional_signal";
    }

    free(delta_cos_values);
    free(ratio_values);
    free(relative_values);
    free(proj_ratio_values);
    free(transfer_values);
    free(delta_bct_values);
    free(delta_base_values);
    free(time_bct_values);
    free(time_base_values);
}

int main(void) {
    ProbeConfig config;
    BoundarySample *samples;
    BoundaryMetrics *metrics;
    float *batch_baseline_times;
    float *batch_target_times;
    ProbeSummary summary;
    int total_samples_count;
    int batch_idx;
    int slot_idx;
    int boundary_idx;
    int sample_id = 0;

    build_config(&config);
    total_samples_count = total_boundary_samples(&config);

    samples = (BoundarySample *)malloc((size_t)total_samples_count * sizeof(BoundarySample));
    metrics = (BoundaryMetrics *)malloc((size_t)total_samples_count * sizeof(BoundaryMetrics));
    batch_baseline_times = (float *)malloc((size_t)config.batches * sizeof(float));
    batch_target_times = (float *)malloc((size_t)config.batches * sizeof(float));
    if (!samples || !metrics || !batch_baseline_times || !batch_target_times) {
        fprintf(stderr, "r394_bct_probe: allocation failed\n");
        free(samples);
        free(metrics);
        free(batch_baseline_times);
        free(batch_target_times);
        return 1;
    }

    printf(
        "R394_BCT_CONFIG mode=synthetic_reference_only batches=%d batch_size=%d boundaries=%d suffix_window=%d base_window=%d dim_theta=%d dim_state=%d vocab=%d eta=%.6g rho_state=%.6g eps_step=%.6g timing_repeats=%d selected_blocks=2 baseline=truncated_suffix_reference\n",
        config.batches,
        config.batch_size,
        config.boundaries,
        config.suffix_window,
        config.base_window,
        config.dim_theta,
        config.dim_state,
        config.vocab,
        config.eta,
        config.rho_state,
        config.eps_step,
        config.timing_repeats
    );

    for (batch_idx = 0; batch_idx < config.batches; ++batch_idx) {
        for (slot_idx = 0; slot_idx < config.batch_size; ++slot_idx) {
            for (boundary_idx = 0; boundary_idx < config.boundaries; ++boundary_idx) {
                build_sample(&config, sample_id, batch_idx, slot_idx, boundary_idx, &samples[sample_id]);
                collect_boundary_metrics(&config, &samples[sample_id], &metrics[sample_id]);
                sample_id += 1;
            }
        }
        measure_batch_times(&config, samples, batch_idx, &batch_baseline_times[batch_idx], &batch_target_times[batch_idx]);
    }

    summarize_probe(&config, metrics, total_samples_count, batch_baseline_times, batch_target_times, &summary);

    printf(
        "R394_BCT_METRICS median_delta_cos=%.6g relative_cos_gain=%.6g relative_delta=%.6g kappa=%.6g R_wall=%.6g sign_flip_base=%.6g sign_flip_bct=%.6g invalid_rate=%.6g skip_rate=%.6g clip_rate=%.6g heavy_clip_rate=%.6g transfer_risk=%.6g proj_ratio=%.6g transfer_ratio=%.6g delta_cos_iqr=%.6g boundary_count=%d valid_boundaries=%d ratio_valid=%d suffix_product_count=%d solve_count=%d sync_count=%d materialization_count=%d temp_memory_mb=%.6g\n",
        summary.median_delta_cos,
        summary.median_ratio_cos,
        summary.median_relative_delta,
        summary.kappa,
        summary.r_wall,
        summary.sign_flip_base,
        summary.sign_flip_bct,
        summary.invalid_rate,
        summary.skip_rate,
        summary.clip_rate,
        summary.heavy_clip_rate,
        summary.transfer_risk,
        summary.median_proj_ratio,
        summary.median_transfer_ratio,
        summary.delta_cos_iqr,
        summary.total_samples,
        summary.valid_count,
        summary.ratio_valid_count,
        summary.suffix_product_count,
        summary.solve_count,
        summary.sync_count,
        summary.materialization_count,
        summary.temp_memory_mb
    );
    printf("R394_BCT_DECISION %s reason=%s\n", summary.decision, summary.reason);

    free(samples);
    free(metrics);
    free(batch_baseline_times);
    free(batch_target_times);
    return 0;
}
