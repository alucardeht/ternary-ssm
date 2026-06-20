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

#define R399_DMDC_DEFAULT_SCENARIO "pass_fixture"
#define R399_DMDC_EPS 1e-12
#define R399_DMDC_TERMINAL_BIAS_TOL 2.0
#define R399_DMDC_PREFIX_TOL 3.0
#define R399_DMDC_PHI 0.61803398874989484820
#define R399_DMDC_SQRT2_FRAC 0.41421356237309504880

typedef enum {
    R399_SCHEDULE_QUOTA_LOW_RESIDUAL_ZERO = 0,
    R399_SCHEDULE_QUOTA_BALANCED = 1,
    R399_SCHEDULE_QUOTA_LOW_RESIDUAL_SIGN = 2,
    R399_SCHEDULE_BIASED_ZERO = 3
} R399ScheduleMode;

typedef enum {
    R399_PATTERN_PASS = 0,
    R399_PATTERN_FAIL_NOISE = 1,
    R399_PATTERN_FAIL_RESIDUAL = 2,
    R399_PATTERN_FAIL_OVERHEAD = 3,
    R399_PATTERN_FAIL_BIAS = 4,
    R399_PATTERN_INVALID_NODATA = 5,
    R399_PATTERN_INVALID_ZERO_NOISE = 6
} R399Pattern;

typedef struct {
    const char *name;
    R399Pattern pattern;
    R399ScheduleMode schedule;
    int record_count;
    int steps;
    double overhead;
    const char *expected_decision;
    const char *expected_reason;
} R399ScenarioConfig;

typedef struct {
    const char *scenario;
    const char *output_path;
    uint64_t seed;
} R399CliConfig;

typedef struct {
    double logit;
    double weight;
} R399Record;

typedef struct {
    int records;
    int useful_records;
    int steps;
    uint64_t seed;
    double random_noise_var;
    double det_noise_var;
    double useful_signal_var;
    double f_q;
    double rho;
    double h;
    double lhs;
    double projected_speed;
    double terminal_bias;
    double max_prefix_discrepancy;
    const char *decision;
    const char *reason;
} R399ScenarioMetrics;

static const R399ScenarioConfig R399_SCENARIOS[] = {
    {"pass_fixture",      R399_PATTERN_PASS,               R399_SCHEDULE_QUOTA_LOW_RESIDUAL_ZERO, 96, 4,   0.03, "PASS",    "passed_gate"},
    {"fail_noise_fraction", R399_PATTERN_FAIL_NOISE,      R399_SCHEDULE_QUOTA_LOW_RESIDUAL_SIGN, 96, 4,   0.03, "FAIL",    "insufficient_f_q"},
    {"fail_residual",     R399_PATTERN_FAIL_RESIDUAL,     R399_SCHEDULE_QUOTA_BALANCED,          96, 4,   0.03, "FAIL",    "rho_too_high"},
    {"fail_overhead",     R399_PATTERN_FAIL_OVERHEAD,     R399_SCHEDULE_QUOTA_LOW_RESIDUAL_ZERO, 96, 4,   1.10, "FAIL",    "overhead_too_high"},
    {"fail_bias",         R399_PATTERN_FAIL_BIAS,         R399_SCHEDULE_BIASED_ZERO, 96, 256, 0.01, "FAIL",    "bias_detected"},
    {"invalid_nodata",    R399_PATTERN_INVALID_NODATA,    R399_SCHEDULE_QUOTA_BALANCED,          64, 256, 0.03, "INVALID", "no_useful_records"},
    {"invalid_zero_noise", R399_PATTERN_INVALID_ZERO_NOISE, R399_SCHEDULE_QUOTA_BALANCED,        64, 256, 0.03, "INVALID", "zero_random_noise_denominator"}
};

static const int R399_SCENARIO_COUNT = (int)(sizeof(R399_SCENARIOS) / sizeof(R399_SCENARIOS[0]));

static uint64_t r399_mix64(uint64_t x)
{
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

static double r399_frac(double x)
{
    double y = x - floor(x);
    if (y < 0.0) {
        y += 1.0;
    }
    return y;
}

static double r399_counter_uniform01(uint64_t seed, uint64_t stream, uint64_t index)
{
    uint64_t word = r399_mix64(seed ^ (stream * 0x9E3779B97F4A7C15ULL) ^ index);
    return (double)((word >> 11) & ((1ULL << 53) - 1ULL)) * (1.0 / (double)(1ULL << 53));
}

/* Mirrors sample_ternary_from_logit(...) in src/core/optimizer.h. */
static double r399_sample_ternary_from_logit(double s, double u)
{
    double p_pos;
    double p_neg;
    if (s >= 1.0) {
        return 1.0;
    }
    if (s <= -1.0) {
        return -1.0;
    }
    p_pos = (s > 0.0) ? s : 0.0;
    p_neg = (s < 0.0) ? -s : 0.0;
    if (u < p_pos) {
        return 1.0;
    }
    if (u < p_pos + p_neg) {
        return -1.0;
    }
    return 0.0;
}

static double r399_expected_noise_var(double s)
{
    double abs_s = fabs(s);
    if (abs_s >= 1.0) {
        return 0.0;
    }
    return abs_s - (s * s);
}

static int r399_mkdir_single(const char *path)
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

static int r399_ensure_parent_dirs(const char *path)
{
    char scratch[1024];
    size_t i;
    size_t len;

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
        if (r399_mkdir_single(scratch) != 0) {
            return -1;
        }
        scratch[i] = path[i];
    }
    return 0;
}

static const R399ScenarioConfig *r399_find_scenario(const char *name)
{
    int i;
    for (i = 0; i < R399_SCENARIO_COUNT; ++i) {
        if (strcmp(name, R399_SCENARIOS[i].name) == 0) {
            return &R399_SCENARIOS[i];
        }
    }
    return NULL;
}

static void r399_print_usage(const char *program)
{
    fprintf(stderr,
            "usage: %s [--scenario all|pass_fixture|fail_noise_fraction|fail_residual|fail_overhead|fail_bias|invalid_nodata|invalid_zero_noise] "
            "[--output PATH] [--seed N]\n",
            program);
}

static int r399_parse_uint64(const char *text, uint64_t *out)
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

static int r399_parse_args(int argc, char **argv, R399CliConfig *cfg)
{
    int i;
    cfg->scenario = R399_DMDC_DEFAULT_SCENARIO;
    cfg->output_path = NULL;
    cfg->seed = 399ULL;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--scenario") == 0 && i + 1 < argc) {
            cfg->scenario = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            cfg->output_path = argv[++i];
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!r399_parse_uint64(argv[++i], &cfg->seed)) {
                return -1;
            }
        } else if (strcmp(argv[i], "--help") == 0) {
            return 1;
        } else {
            return -1;
        }
    }

    if (strcmp(cfg->scenario, "all") != 0 && r399_find_scenario(cfg->scenario) == NULL) {
        return -1;
    }
    return 0;
}

static void r399_fill_record(const R399ScenarioConfig *scenario, int index, R399Record *record)
{
    double phase = r399_frac((double)(index + 1) * 0.173 + (double)scenario->pattern * 0.071);
    double sign = ((index & 1) == 0) ? 1.0 : -1.0;

    record->weight = 1.0;
    record->logit = 0.0;

    switch (scenario->pattern) {
        case R399_PATTERN_PASS:
        case R399_PATTERN_FAIL_RESIDUAL:
        case R399_PATTERN_FAIL_OVERHEAD:
            record->logit = sign * (0.14 + 0.04 * phase);
            record->weight = 0.85 + 0.30 * r399_frac((double)(index + 1) * 0.127);
            break;
        case R399_PATTERN_FAIL_NOISE:
            record->logit = sign * (0.80 + 0.08 * phase);
            record->weight = 0.90 + 0.25 * r399_frac((double)(index + 1) * 0.091);
            break;
        case R399_PATTERN_FAIL_BIAS:
            record->logit = sign * (0.12 + 0.06 * phase);
            record->weight = 1.0;
            break;
        case R399_PATTERN_INVALID_NODATA:
            record->logit = sign * (0.22 + 0.10 * phase);
            record->weight = 0.0;
            break;
        case R399_PATTERN_INVALID_ZERO_NOISE:
            record->logit = (index % 3 == 0) ? 0.0 : (((index & 1) == 0) ? 1.0 : -1.0);
            record->weight = 1.0;
            break;
        default:
            record->logit = sign * 0.25;
            record->weight = 1.0;
            break;
    }
}

static double r399_record_phase(
    const R399ScenarioConfig *scenario,
    uint64_t seed,
    int record_index,
    double salt
)
{
    return r399_frac(
        ((double)(seed & 0xFFFFu) * 0.000001)
        + ((double)(record_index + 1) * R399_DMDC_SQRT2_FRAC)
        + ((double)(scenario->pattern + 1) * R399_DMDC_PHI)
        + salt
    );
}

static double r399_schedule_target_mean(const R399ScenarioConfig *scenario, double s)
{
    switch (scenario->schedule) {
        case R399_SCHEDULE_QUOTA_LOW_RESIDUAL_ZERO:
            return 0.0;
        case R399_SCHEDULE_QUOTA_LOW_RESIDUAL_SIGN:
            if (s > 0.0) {
                return 1.0;
            }
            if (s < 0.0) {
                return -1.0;
            }
            return 0.0;
        case R399_SCHEDULE_QUOTA_BALANCED:
            return s;
        case R399_SCHEDULE_BIASED_ZERO:
            return 0.0;
        default:
            return s;
    }
}

static int r399_quota_count(double magnitude, int steps, double phase)
{
    double target = magnitude * (double)steps;
    int base = (int)floor(target);
    double remainder = target - (double)base;
    if (remainder > 0.0 && phase < remainder) {
        base += 1;
    }
    if (base < 0) {
        return 0;
    }
    if (base > steps) {
        return steps;
    }
    return base;
}

static bool r399_quota_hit(int step_index, int steps, int quota, double phase)
{
    double start;
    double end;
    if (quota <= 0) {
        return false;
    }
    if (quota >= steps) {
        return true;
    }
    start = (((double)step_index * (double)quota) / (double)steps) + phase;
    end = ((((double)step_index + 1.0) * (double)quota) / (double)steps) + phase;
    return floor(end) > floor(start);
}

static double r399_deterministic_sample(
    const R399ScenarioConfig *scenario,
    uint64_t seed,
    int record_index,
    int step_index,
    double s
)
{
    double target_mean;
    double magnitude;
    double phase;
    int quota;

    if (scenario->schedule == R399_SCHEDULE_BIASED_ZERO) {
        return 0.0;
    }

    target_mean = r399_schedule_target_mean(scenario, s);
    if (target_mean > 1.0) {
        target_mean = 1.0;
    } else if (target_mean < -1.0) {
        target_mean = -1.0;
    }
    if (target_mean == 0.0) {
        return 0.0;
    }

    magnitude = fabs(target_mean);
    phase = r399_record_phase(scenario, seed, record_index, magnitude * 0.173);
    quota = r399_quota_count(magnitude, scenario->steps, phase);
    if (!r399_quota_hit(step_index, scenario->steps, quota, phase)) {
        return 0.0;
    }
    return (target_mean > 0.0) ? 1.0 : -1.0;
}

static void r399_classify_core(
    const R399ScenarioConfig *scenario,
    R399ScenarioMetrics *metrics
)
{
    double gate_base;
    double required_fq;
    double allowed_rho;
    double allowed_h;
    bool base_gate_passes;
    bool required_fq_valid;
    bool allowed_rho_valid;
    bool allowed_h_valid;
    bool fq_limiter;
    bool rho_limiter;
    bool overhead_limiter;

    metrics->decision = "INVALID";
    metrics->reason = "non_finite_metric";

    if (metrics->useful_records <= 0) {
        metrics->reason = "no_useful_records";
        return;
    }
    if (!isfinite(metrics->random_noise_var) || metrics->random_noise_var <= R399_DMDC_EPS) {
        metrics->reason = "zero_random_noise_denominator";
        return;
    }
    if (!isfinite(metrics->det_noise_var) || !isfinite(metrics->useful_signal_var) ||
        !isfinite(metrics->f_q) || !isfinite(metrics->rho) || !isfinite(metrics->h) ||
        !isfinite(metrics->lhs) || !isfinite(metrics->terminal_bias) ||
        !isfinite(metrics->max_prefix_discrepancy)) {
        metrics->reason = "non_finite_metric";
        return;
    }
    if (metrics->f_q < 0.0 || metrics->f_q > 1.0 || metrics->rho < 0.0 || metrics->h < 0.0) {
        metrics->reason = "metric_out_of_range";
        return;
    }

    if (metrics->terminal_bias > R399_DMDC_TERMINAL_BIAS_TOL ||
        metrics->max_prefix_discrepancy > R399_DMDC_PREFIX_TOL) {
        metrics->decision = "FAIL";
        metrics->reason = "bias_detected";
    } else if (metrics->lhs <= 0.50) {
        metrics->decision = "PASS";
        metrics->reason = "passed_gate";
    } else {
        gate_base = 1.0 - (metrics->f_q * (1.0 - metrics->rho));
        required_fq = DBL_MAX;
        allowed_rho = DBL_MAX;
        allowed_h = DBL_MAX;

        if ((1.0 - metrics->rho) > R399_DMDC_EPS) {
            required_fq = (1.0 - (0.50 / (1.0 + metrics->h))) / (1.0 - metrics->rho);
        }
        if (metrics->f_q > R399_DMDC_EPS) {
            allowed_rho = 1.0 - ((1.0 - (0.50 / (1.0 + metrics->h))) / metrics->f_q);
        }
        if (gate_base > R399_DMDC_EPS) {
            allowed_h = (0.50 / gate_base) - 1.0;
        }

        base_gate_passes = isfinite(gate_base) && gate_base <= 0.50;
        required_fq_valid = isfinite(required_fq) && required_fq >= 0.0 && required_fq <= 1.0;
        allowed_rho_valid = isfinite(allowed_rho) && allowed_rho >= 0.0;
        allowed_h_valid = isfinite(allowed_h) && allowed_h >= 0.0;

        overhead_limiter = base_gate_passes && allowed_h_valid && metrics->h > allowed_h;
        rho_limiter = metrics->rho >= 1.0 || (allowed_rho_valid && metrics->rho > allowed_rho);
        fq_limiter = !base_gate_passes && required_fq_valid && metrics->f_q < required_fq;

        metrics->decision = "FAIL";
        if (overhead_limiter) {
            metrics->reason = "overhead_too_high";
        } else if (rho_limiter) {
            metrics->reason = "rho_too_high";
        } else if (fq_limiter) {
            metrics->reason = "insufficient_f_q";
        } else {
            metrics->reason = "lhs_above_gate";
        }
    }

    if (strcmp(metrics->decision, scenario->expected_decision) != 0 ||
        strcmp(metrics->reason, scenario->expected_reason) != 0) {
        metrics->decision = "INVALID";
        metrics->reason = "self_check_mismatch";
    }
}

static void r399_compute_metrics(
    const R399ScenarioConfig *scenario,
    uint64_t seed,
    R399ScenarioMetrics *metrics
)
{
    R399Record *records;
    double useful_weight_sum = 0.0;
    double useful_signal_sum = 0.0;
    double random_noise_sum = 0.0;
    double det_noise_sum = 0.0;
    double terminal_bias_num = 0.0;
    double terminal_bias_den = 0.0;
    int useful_records = 0;
    int i;

    memset(metrics, 0, sizeof(*metrics));
    metrics->records = scenario->record_count;
    metrics->steps = scenario->steps;
    metrics->seed = seed;
    metrics->h = scenario->overhead;
    metrics->projected_speed = NAN;
    metrics->decision = "INVALID";
    metrics->reason = "runtime_error";

    records = (R399Record *)malloc((size_t)scenario->record_count * sizeof(R399Record));
    if (!records) {
        metrics->reason = "allocation_failed";
        return;
    }

    for (i = 0; i < scenario->record_count; ++i) {
        r399_fill_record(scenario, i, &records[i]);
    }

    for (i = 0; i < scenario->record_count; ++i) {
        const R399Record *record = &records[i];
        double s = record->logit;
        double weight = record->weight;
        double expected_var = weight * r399_expected_noise_var(s);
        double det_mean_sum = 0.0;
        double prefix = 0.0;
        int step;

        if (!isfinite(weight) || weight <= 0.0 || !isfinite(s)) {
            continue;
        }

        useful_records += 1;
        useful_weight_sum += weight;
        useful_signal_sum += weight * s * s;

        for (step = 0; step < scenario->steps; ++step) {
            double random_u = r399_counter_uniform01(
                seed,
                0xA5D3C91EULL + (uint64_t)(i + 1),
                (uint64_t)(step + 1)
            );
            double random_sample = r399_sample_ternary_from_logit(s, random_u);
            double det_sample = r399_deterministic_sample(scenario, seed, i, step, s);
            double random_disc = random_sample - s;
            double det_disc = det_sample - s;
            double prefix_norm;

            random_noise_sum += weight * random_disc * random_disc;
            det_noise_sum += weight * det_disc * det_disc;

            det_mean_sum += det_sample;
            prefix += weight * det_disc;
            prefix_norm = fabs(prefix) / sqrt((expected_var + R399_DMDC_EPS) * (double)(step + 1));
            if (prefix_norm > metrics->max_prefix_discrepancy) {
                metrics->max_prefix_discrepancy = prefix_norm;
            }
        }

        {
            double det_mean = det_mean_sum / (double)scenario->steps;
            double mean_error = det_mean - s;
            terminal_bias_num += weight * mean_error * mean_error;
            terminal_bias_den += (expected_var + R399_DMDC_EPS) / (double)scenario->steps;
        }
    }

    metrics->useful_records = useful_records;

    if (useful_weight_sum > 0.0) {
        metrics->random_noise_var = random_noise_sum / (useful_weight_sum * (double)scenario->steps);
        metrics->det_noise_var = det_noise_sum / (useful_weight_sum * (double)scenario->steps);
        metrics->useful_signal_var = useful_signal_sum / useful_weight_sum;
    } else {
        metrics->random_noise_var = NAN;
        metrics->det_noise_var = NAN;
        metrics->useful_signal_var = NAN;
    }

    if (terminal_bias_den > 0.0) {
        metrics->terminal_bias = sqrt(terminal_bias_num / terminal_bias_den);
    } else {
        metrics->terminal_bias = NAN;
    }

    if (isfinite(metrics->random_noise_var) && isfinite(metrics->useful_signal_var) &&
        (metrics->random_noise_var + metrics->useful_signal_var) > R399_DMDC_EPS) {
        metrics->f_q = metrics->random_noise_var /
                       (metrics->random_noise_var + metrics->useful_signal_var);
    } else {
        metrics->f_q = NAN;
    }

    if (isfinite(metrics->random_noise_var) && metrics->random_noise_var > R399_DMDC_EPS &&
        isfinite(metrics->det_noise_var)) {
        metrics->rho = metrics->det_noise_var / metrics->random_noise_var;
    } else {
        metrics->rho = NAN;
    }

    if (isfinite(metrics->f_q) && isfinite(metrics->rho) && isfinite(metrics->h)) {
        metrics->lhs = (1.0 - metrics->f_q * (1.0 - metrics->rho)) * (1.0 + metrics->h);
    } else {
        metrics->lhs = NAN;
    }

    if (isfinite(metrics->lhs) && metrics->lhs > R399_DMDC_EPS) {
        metrics->projected_speed = 1.0 / metrics->lhs;
    }

    r399_classify_core(scenario, metrics);
    free(records);
}

static void r399_print_metrics_line(const R399ScenarioConfig *scenario, const R399ScenarioMetrics *metrics)
{
    (void)scenario;
    printf(
        "[R399_DMDC_GATE] metrics scenario=%s records=%d useful_records=%d f_q=%.9g rho=%.9g h=%.9g lhs=%.9g projected_speed=%.9g "
        "random_noise_var=%.9g det_noise_var=%.9g useful_signal_var=%.9g terminal_bias=%.9g max_prefix_discrepancy=%.9g decision=%s reason=%s\n",
        scenario->name,
        metrics->records,
        metrics->useful_records,
        metrics->f_q,
        metrics->rho,
        metrics->h,
        metrics->lhs,
        metrics->projected_speed,
        metrics->random_noise_var,
        metrics->det_noise_var,
        metrics->useful_signal_var,
        metrics->terminal_bias,
        metrics->max_prefix_discrepancy,
        metrics->decision,
        metrics->reason
    );
}

static int r399_write_jsonl_record(FILE *handle, const R399ScenarioConfig *scenario, const R399ScenarioMetrics *metrics)
{
    if (fprintf(
            handle,
            "{\"scenario\":\"%s\",\"records\":%d,\"useful_records\":%d,\"steps\":%d,\"seed\":%llu,"
            "\"f_q\":%.17g,\"rho\":%.17g,\"h\":%.17g,\"lhs\":%.17g,\"projected_speed\":%.17g,"
            "\"random_noise_var\":%.17g,\"det_noise_var\":%.17g,\"useful_signal_var\":%.17g,"
            "\"terminal_bias\":%.17g,\"max_prefix_discrepancy\":%.17g,"
            "\"decision\":\"%s\",\"reason\":\"%s\",\"expected_decision\":\"%s\",\"expected_reason\":\"%s\"}\n",
            scenario->name,
            metrics->records,
            metrics->useful_records,
            metrics->steps,
            (unsigned long long)metrics->seed,
            metrics->f_q,
            metrics->rho,
            metrics->h,
            metrics->lhs,
            metrics->projected_speed,
            metrics->random_noise_var,
            metrics->det_noise_var,
            metrics->useful_signal_var,
            metrics->terminal_bias,
            metrics->max_prefix_discrepancy,
            metrics->decision,
            metrics->reason,
            scenario->expected_decision,
            scenario->expected_reason
        ) < 0) {
        return -1;
    }
    return 0;
}

static int r399_run_one(
    const R399ScenarioConfig *scenario,
    uint64_t seed,
    FILE *output_handle
)
{
    R399ScenarioMetrics metrics;
    r399_compute_metrics(scenario, seed, &metrics);
    r399_print_metrics_line(scenario, &metrics);
    if (output_handle && r399_write_jsonl_record(output_handle, scenario, &metrics) != 0) {
        return -1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    R399CliConfig cfg;
    FILE *output_handle = NULL;
    int parse_result;
    int i;

    parse_result = r399_parse_args(argc, argv, &cfg);
    if (parse_result == 1) {
        r399_print_usage(argv[0]);
        return 0;
    }
    if (parse_result != 0) {
        r399_print_usage(argv[0]);
        return 2;
    }

    if (cfg.output_path) {
        if (r399_ensure_parent_dirs(cfg.output_path) != 0) {
            fprintf(stderr, "[R399_DMDC_GATE] failed output_path=%s errno=%d\n", cfg.output_path, errno);
            return 1;
        }
        output_handle = fopen(cfg.output_path, "wb");
        if (!output_handle) {
            fprintf(stderr, "[R399_DMDC_GATE] failed output_path=%s errno=%d\n", cfg.output_path, errno);
            return 1;
        }
    }

    printf("[R399_DMDC_GATE] active mode=diagnostic-only source=standalone no_training_update=1 inequality=\"(1-f_q*(1-rho))*(1+h)<=0.50\"\n");

    if (strcmp(cfg.scenario, "all") == 0) {
        for (i = 0; i < R399_SCENARIO_COUNT; ++i) {
            if (r399_run_one(&R399_SCENARIOS[i], cfg.seed, output_handle) != 0) {
                if (output_handle) {
                    fclose(output_handle);
                }
                fprintf(stderr, "[R399_DMDC_GATE] failed output_path=%s errno=%d\n",
                        cfg.output_path ? cfg.output_path : "(null)", errno);
                return 1;
            }
        }
    } else {
        const R399ScenarioConfig *scenario = r399_find_scenario(cfg.scenario);
        if (!scenario) {
            if (output_handle) {
                fclose(output_handle);
            }
            r399_print_usage(argv[0]);
            return 2;
        }
        if (r399_run_one(scenario, cfg.seed, output_handle) != 0) {
            if (output_handle) {
                fclose(output_handle);
            }
            fprintf(stderr, "[R399_DMDC_GATE] failed output_path=%s errno=%d\n",
                    cfg.output_path ? cfg.output_path : "(null)", errno);
            return 1;
        }
    }

    if (output_handle) {
        fclose(output_handle);
    }
    return 0;
}
