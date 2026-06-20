#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/optimizer.h"
#include "core/shadow_codec.h"

#define R375_MAX_SHAPES 16
#define R375_MAX_SHAPE_TEXT 256
#define R375_PI 3.14159265358979323846

typedef struct {
    int rows;
    int cols;
} ProbeShape;

typedef struct {
    uint64_t seed;
    int samples;
    int shape_count;
    ProbeShape shapes[R375_MAX_SHAPES];
    char shape_text[R375_MAX_SHAPE_TEXT];
} ProbeConfig;

typedef struct {
    double bias_rel;
    double cos_mean;
    double cos_lcb95;
    double noise_ratio;
    double r_eff;
    double sat_frac;
    double row_energy_cos;
    double col_energy_cos;
    double shape_margin;
    const char *verdict;
} ProbeMetrics;

static uint64_t mix64(uint64_t x)
{
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

static float counter_uniform01(uint64_t seed, uint64_t stream, uint64_t index)
{
    uint64_t word = mix64(seed ^ (stream * 0x9E3779B97F4A7C15ULL) ^ index);
    return (float)((word >> 40) & 0xFFFFFFu) * (1.0f / 16777216.0f);
}

static double dot_f64(const float *a, const float *b, size_t n)
{
    double total = 0.0;
    size_t i;
    for (i = 0; i < n; ++i) {
        total += (double)a[i] * (double)b[i];
    }
    return total;
}

static double norm_sq_f64(const float *a, size_t n)
{
    return dot_f64(a, a, n);
}

static double vector_cosine_f64(const double *a, const double *b, int n)
{
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    int i;
    for (i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a <= 0.0 || norm_b <= 0.0) {
        return 0.0;
    }
    return dot / sqrt(norm_a * norm_b);
}

static float row_basis(int axis, int index, int size)
{
    double x = (double)(index + 1) / (double)(size + 1);
    switch (axis) {
        case 0:
            return (float)(0.85 * sin(2.0 * R375_PI * x) + 0.30 * cos(7.0 * R375_PI * x));
        case 1:
            return (float)(0.75 * cos(3.0 * R375_PI * x) - 0.20 * sin(5.0 * R375_PI * x));
        default:
            return (float)(0.60 * sin(5.0 * R375_PI * x) + 0.40 * cos(11.0 * R375_PI * x));
    }
}

static float col_basis(int axis, int index, int size)
{
    double x = (double)(index + 1) / (double)(size + 1);
    switch (axis) {
        case 0:
            return (float)(0.90 * cos(2.5 * R375_PI * x) + 0.25 * sin(9.0 * R375_PI * x));
        case 1:
            return (float)(0.70 * sin(4.0 * R375_PI * x) - 0.35 * cos(6.0 * R375_PI * x));
        default:
            return (float)(0.55 * cos(6.5 * R375_PI * x) + 0.45 * sin(10.0 * R375_PI * x));
    }
}

static void fill_scales(float *out, int size, double phase)
{
    int i;
    for (i = 0; i < size; ++i) {
        double x = (double)(i + 1) / (double)(size + 1);
        double log_lambda = 1.05 * sin(2.0 * R375_PI * x + phase)
                          + 0.45 * cos(5.0 * R375_PI * x - 0.5 * phase);
        double lambda = exp(log_lambda);
        out[i] = (float)pow(lambda, -0.25);
    }
}

static void build_delta_and_gradient(const ProbeShape *shape, float *grad, float *delta)
{
    int rows = shape->rows;
    int cols = shape->cols;
    int rank_k = 3;
    float *left = (float *)malloc((size_t)rows * (size_t)rank_k * sizeof(float));
    float *right = (float *)malloc((size_t)cols * (size_t)rank_k * sizeof(float));
    float *row_scale = (float *)malloc((size_t)rows * sizeof(float));
    float *col_scale = (float *)malloc((size_t)cols * sizeof(float));
    double max_abs = 0.0;
    size_t idx = 0;
    int r;
    int c;
    int k;

    if (!left || !right || !row_scale || !col_scale) {
        fprintf(stderr, "r375_pbg2_mvue_probe: allocation failed while building synthetic update\n");
        free(left);
        free(right);
        free(row_scale);
        free(col_scale);
        exit(1);
    }

    for (r = 0; r < rows; ++r) {
        for (k = 0; k < rank_k; ++k) {
            left[(size_t)r * (size_t)rank_k + (size_t)k] = row_basis(k, r, rows);
        }
    }
    for (c = 0; c < cols; ++c) {
        for (k = 0; k < rank_k; ++k) {
            right[(size_t)c * (size_t)rank_k + (size_t)k] = col_basis(k, c, cols);
        }
    }
    fill_scales(row_scale, rows, 0.25);
    fill_scales(col_scale, cols, 0.85);

    for (r = 0; r < rows; ++r) {
        for (c = 0; c < cols; ++c, ++idx) {
            double signal = 0.0;
            double noise;
            double mixed;
            double shaped;
            signal += 0.74 * (double)left[(size_t)r * (size_t)rank_k + 0] * (double)right[(size_t)c * (size_t)rank_k + 0];
            signal += 0.21 * (double)left[(size_t)r * (size_t)rank_k + 1] * (double)right[(size_t)c * (size_t)rank_k + 1];
            signal += 0.07 * (double)left[(size_t)r * (size_t)rank_k + 2] * (double)right[(size_t)c * (size_t)rank_k + 2];
            noise = 0.015 * sin(0.013 * (double)(r + 3) * (double)(c + 5))
                  + 0.010 * cos(0.021 * (double)(r + 7) * (double)(c + 1));
            mixed = signal + noise;
            shaped = (double)row_scale[r] * mixed * (double)col_scale[c];
            grad[idx] = (float)mixed;
            delta[idx] = (float)shaped;
            if (fabs(shaped) > max_abs) {
                max_abs = fabs(shaped);
            }
        }
    }

    if (max_abs > 0.0) {
        double scale = 0.82 / max_abs;
        for (idx = 0; idx < (size_t)rows * (size_t)cols; ++idx) {
            grad[idx] = (float)((double)grad[idx] * scale);
            delta[idx] = (float)((double)delta[idx] * scale);
        }
    }

    free(left);
    free(right);
    free(row_scale);
    free(col_scale);
}

static double estimate_spectral_norm_sq(const float *matrix, int rows, int cols)
{
    float *v = (float *)malloc((size_t)cols * sizeof(float));
    float *u = (float *)malloc((size_t)rows * sizeof(float));
    int iter;
    int i;
    int r;
    int c;

    if (!v || !u) {
        fprintf(stderr, "r375_pbg2_mvue_probe: allocation failed during spectral estimate\n");
        free(v);
        free(u);
        exit(1);
    }

    for (i = 0; i < cols; ++i) {
        v[i] = (float)(0.5 + 0.5 * sin(0.37 * (double)(i + 1)));
    }

    for (iter = 0; iter < 24; ++iter) {
        double norm_u = 0.0;
        double norm_v = 0.0;
        for (r = 0; r < rows; ++r) {
            double sum = 0.0;
            const float *row = matrix + (size_t)r * (size_t)cols;
            for (c = 0; c < cols; ++c) {
                sum += (double)row[c] * (double)v[c];
            }
            u[r] = (float)sum;
            norm_u += sum * sum;
        }
        norm_u = sqrt(norm_u);
        if (norm_u <= 0.0) {
            free(v);
            free(u);
            return 0.0;
        }
        for (r = 0; r < rows; ++r) {
            u[r] = (float)((double)u[r] / norm_u);
        }

        for (c = 0; c < cols; ++c) {
            double sum = 0.0;
            for (r = 0; r < rows; ++r) {
                sum += (double)matrix[(size_t)r * (size_t)cols + (size_t)c] * (double)u[r];
            }
            v[c] = (float)sum;
            norm_v += sum * sum;
        }
        norm_v = sqrt(norm_v);
        if (norm_v <= 0.0) {
            free(v);
            free(u);
            return 0.0;
        }
        for (c = 0; c < cols; ++c) {
            v[c] = (float)((double)v[c] / norm_v);
        }
    }

    {
        double sigma_sq = 0.0;
        for (r = 0; r < rows; ++r) {
            double sum = 0.0;
            const float *row = matrix + (size_t)r * (size_t)cols;
            for (c = 0; c < cols; ++c) {
                sum += (double)row[c] * (double)v[c];
            }
            sigma_sq += sum * sum;
        }
        free(v);
        free(u);
        return sigma_sq;
    }
}

static void build_energy_profiles(
    const float *matrix,
    int rows,
    int cols,
    double *row_energy,
    double *col_energy
)
{
    int r;
    int c;
    for (r = 0; r < rows; ++r) {
        row_energy[r] = 0.0;
    }
    for (c = 0; c < cols; ++c) {
        col_energy[c] = 0.0;
    }
    for (r = 0; r < rows; ++r) {
        const float *row = matrix + (size_t)r * (size_t)cols;
        for (c = 0; c < cols; ++c) {
            double v = (double)row[c];
            double sq = v * v;
            row_energy[r] += sq;
            col_energy[c] += sq;
        }
    }
}

static const char *classify_verdict(const ProbeMetrics *metrics)
{
    if (metrics->cos_lcb95 < 0.50 ||
        metrics->noise_ratio > 0.50 ||
        metrics->bias_rel > 0.10 ||
        metrics->r_eff > 32.0 ||
        metrics->sat_frac > 0.01 ||
        metrics->row_energy_cos < 0.75 ||
        metrics->col_energy_cos < 0.75 ||
        metrics->shape_margin < 0.03) {
        return "PARKED";
    }

    if (metrics->cos_lcb95 >= 0.90 &&
        metrics->noise_ratio <= 0.10 &&
        metrics->bias_rel < 0.05 &&
        metrics->r_eff < 10.0 &&
        metrics->sat_frac < 0.001 &&
        metrics->row_energy_cos >= 0.90 &&
        metrics->col_energy_cos >= 0.90 &&
        metrics->shape_margin > 0.10) {
        return "PASS";
    }

    return "WEAK";
}

static void run_mode(
    const ProbeConfig *config,
    const ProbeShape *shape,
    const char *mode,
    int use_int16
)
{
    int rows = shape->rows;
    int cols = shape->cols;
    size_t n = (size_t)rows * (size_t)cols;
    float *grad = (float *)malloc(n * sizeof(float));
    float *delta = (float *)malloc(n * sizeof(float));
    float *mean_t = (float *)calloc(n, sizeof(float));
    int16_t *qbuf = use_int16 ? (int16_t *)malloc(n * sizeof(int16_t)) : NULL;
    double *row_delta = (double *)malloc((size_t)rows * sizeof(double));
    double *col_delta = (double *)malloc((size_t)cols * sizeof(double));
    double *row_grad = (double *)malloc((size_t)rows * sizeof(double));
    double *col_grad = (double *)malloc((size_t)cols * sizeof(double));
    double *row_mean = (double *)malloc((size_t)rows * sizeof(double));
    double *col_mean = (double *)malloc((size_t)cols * sizeof(double));
    double delta_norm_sq;
    double sigma_sq;
    double cos_sum = 0.0;
    double cos_sq_sum = 0.0;
    double noise_sum = 0.0;
    double sat_count = 0.0;
    size_t idx;
    int sample_idx;
    ProbeMetrics metrics;

    if (!grad || !delta || !mean_t || !row_delta || !col_delta || !row_grad || !col_grad || !row_mean || !col_mean || (use_int16 && !qbuf)) {
        fprintf(stderr, "r375_pbg2_mvue_probe: allocation failed for shape %dx%d\n", rows, cols);
        free(grad);
        free(delta);
        free(mean_t);
        free(qbuf);
        free(row_delta);
        free(col_delta);
        free(row_grad);
        free(col_grad);
        free(row_mean);
        free(col_mean);
        exit(1);
    }

    build_delta_and_gradient(shape, grad, delta);
    delta_norm_sq = norm_sq_f64(delta, n);
    sigma_sq = estimate_spectral_norm_sq(delta, rows, cols);

    for (idx = 0; idx < n; ++idx) {
        if (fabsf(delta[idx]) >= 1.0f) {
            sat_count += 1.0;
        }
    }

    for (sample_idx = 0; sample_idx < config->samples; ++sample_idx) {
        double dot_td = 0.0;
        double sample_norm_sq = 0.0;
        double diff_norm_sq = 0.0;
        for (idx = 0; idx < n; ++idx) {
            uint64_t sample_stream = (uint64_t)(sample_idx + 1);
            float sampled;
            if (use_int16) {
                uint32_t qrand = shadow_rng(0, V12_SHADOW_MAT_W_IN, (int)idx, config->seed + sample_stream);
                float u = counter_uniform01(config->seed, 0xA5A5u + sample_stream, idx);
                qbuf[idx] = stochastic_round_shadow(delta[idx], qrand);
                sampled = sample_ternary_from_logit_q(qbuf[idx], u);
            } else {
                float u = counter_uniform01(config->seed, 0x5A5Au + sample_stream, idx);
                sampled = sample_ternary_from_logit(delta[idx], u);
            }
            mean_t[idx] += sampled;
            dot_td += (double)sampled * (double)delta[idx];
            sample_norm_sq += (double)sampled * (double)sampled;
            {
                double diff = (double)sampled - (double)delta[idx];
                diff_norm_sq += diff * diff;
            }
        }

        if (sample_norm_sq > 0.0 && delta_norm_sq > 0.0) {
            double cos_value = dot_td / sqrt(sample_norm_sq * delta_norm_sq);
            cos_sum += cos_value;
            cos_sq_sum += cos_value * cos_value;
        }
        if (delta_norm_sq > 0.0) {
            noise_sum += diff_norm_sq / delta_norm_sq;
        }
    }

    for (idx = 0; idx < n; ++idx) {
        mean_t[idx] = (float)((double)mean_t[idx] / (double)config->samples);
    }

    build_energy_profiles(delta, rows, cols, row_delta, col_delta);
    build_energy_profiles(grad, rows, cols, row_grad, col_grad);
    build_energy_profiles(mean_t, rows, cols, row_mean, col_mean);

    {
        double bias_norm_sq = 0.0;
        for (idx = 0; idx < n; ++idx) {
            double diff = (double)mean_t[idx] - (double)delta[idx];
            bias_norm_sq += diff * diff;
        }
        metrics.bias_rel = (delta_norm_sq > 0.0) ? sqrt(bias_norm_sq / delta_norm_sq) : 0.0;
    }

    metrics.cos_mean = cos_sum / (double)config->samples;
    {
        double var = (cos_sq_sum / (double)config->samples) - metrics.cos_mean * metrics.cos_mean;
        double se;
        if (var < 0.0) {
            var = 0.0;
        }
        se = sqrt(var / (double)config->samples);
        metrics.cos_lcb95 = metrics.cos_mean - 1.96 * se;
    }
    metrics.noise_ratio = noise_sum / (double)config->samples;
    metrics.r_eff = (sigma_sq > 0.0) ? (delta_norm_sq / sigma_sq) : 0.0;
    metrics.sat_frac = sat_count / (double)n;
    metrics.row_energy_cos = vector_cosine_f64(row_mean, row_delta, rows);
    metrics.col_energy_cos = vector_cosine_f64(col_mean, col_delta, cols);
    {
        double row_vs_grad = vector_cosine_f64(row_mean, row_grad, rows);
        double col_vs_grad = vector_cosine_f64(col_mean, col_grad, cols);
        double row_margin = metrics.row_energy_cos - row_vs_grad;
        double col_margin = metrics.col_energy_cos - col_vs_grad;
        metrics.shape_margin = (row_margin < col_margin) ? row_margin : col_margin;
    }
    metrics.verdict = classify_verdict(&metrics);

    printf(
        "[R375 PB-G2] mode=%s shape=%dx%d samples=%d bias_rel=%.6f cos_mean=%.6f cos_lcb95=%.6f noise_ratio=%.6f r_eff=%.6f sat_frac=%.6f row_energy_cos=%.6f col_energy_cos=%.6f shape_margin=%.6f verdict=%s\n",
        mode,
        rows,
        cols,
        config->samples,
        metrics.bias_rel,
        metrics.cos_mean,
        metrics.cos_lcb95,
        metrics.noise_ratio,
        metrics.r_eff,
        metrics.sat_frac,
        metrics.row_energy_cos,
        metrics.col_energy_cos,
        metrics.shape_margin,
        metrics.verdict
    );

    free(grad);
    free(delta);
    free(mean_t);
    free(qbuf);
    free(row_delta);
    free(col_delta);
    free(row_grad);
    free(col_grad);
    free(row_mean);
    free(col_mean);
}

static void set_default_shapes(ProbeConfig *config)
{
    config->shape_count = 3;
    config->shapes[0].rows = 4096;
    config->shapes[0].cols = 128;
    config->shapes[1].rows = 128;
    config->shapes[1].cols = 4096;
    config->shapes[2].rows = 256;
    config->shapes[2].cols = 256;
    snprintf(config->shape_text, sizeof(config->shape_text), "4096x128,128x4096,256x256");
}

static int parse_positive_int(const char *text, int *out)
{
    char *end = NULL;
    long value;
    if (!text || !*text) {
        return 0;
    }
    value = strtol(text, &end, 10);
    if (!end || *end != '\0' || value <= 0 || value > 1000000L) {
        return 0;
    }
    *out = (int)value;
    return 1;
}

static int parse_uint64(const char *text, uint64_t *out)
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

static int parse_shapes(ProbeConfig *config, const char *text)
{
    char buffer[R375_MAX_SHAPE_TEXT];
    char *token;
    int count = 0;

    if (!text || !*text || strlen(text) >= sizeof(buffer)) {
        return 0;
    }
    strcpy(buffer, text);
    token = strtok(buffer, ",");
    while (token) {
        int rows = 0;
        int cols = 0;
        char extra = '\0';
        if (count >= R375_MAX_SHAPES) {
            return 0;
        }
        if (sscanf(token, "%dx%d%c", &rows, &cols, &extra) != 2 || rows <= 0 || cols <= 0) {
            return 0;
        }
        config->shapes[count].rows = rows;
        config->shapes[count].cols = cols;
        count++;
        token = strtok(NULL, ",");
    }
    if (count <= 0) {
        return 0;
    }
    config->shape_count = count;
    snprintf(config->shape_text, sizeof(config->shape_text), "%s", text);
    return 1;
}

static int parse_args(ProbeConfig *config, int argc, char **argv)
{
    int i;
    config->seed = 375378ULL;
    config->samples = 256;
    set_default_shapes(config);

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc || !parse_uint64(argv[++i], &config->seed)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--samples") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &config->samples)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--shapes") == 0) {
            if (i + 1 >= argc || !parse_shapes(config, argv[++i])) {
                return 0;
            }
        } else if (strcmp(argv[i], "--help") == 0) {
            return -1;
        } else {
            return 0;
        }
    }
    return 1;
}

static void print_usage(void)
{
    printf("usage: r375_pbg2_mvue_probe [--seed N] [--samples N] [--shapes RxC,RxC,...]\n");
}

int main(int argc, char **argv)
{
    ProbeConfig config;
    int parse_ok = parse_args(&config, argc, argv);
    int i;

    if (parse_ok == -1) {
        print_usage();
        return 0;
    }
    if (parse_ok == 0) {
        print_usage();
        return 1;
    }

    printf(
        "[R375 PB-G2] activation seed=%llu samples=%d shapes=%s modes=fp32,int16\n",
        (unsigned long long)config.seed,
        config.samples,
        config.shape_text
    );

    for (i = 0; i < config.shape_count; ++i) {
        run_mode(&config, &config.shapes[i], "fp32", 0);
        run_mode(&config, &config.shapes[i], "int16", 1);
    }

    return 0;
}
