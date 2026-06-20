#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/core/optimizer.h"
#include "../src/core/types.h"

#define R394_RBTC_MAX_K 12
#define R394_RBTC_EPS 1.0e-12
#define R394_RBTC_MEAN_EPS 1.0e-9

typedef enum {
    R394_MODE_SCALAR_FIXED_DW_AUDIT = 0,
    R394_MODE_TINY_BLOCK_EXACT = 1
} ProbeMode;

typedef enum {
    R394_BIN_NEAR_ZERO = 0,
    R394_BIN_MID = 1,
    R394_BIN_EDGE = 2,
    R394_BIN_CRYSTALLIZED = 3
} ShadowBin;

typedef struct {
    ProbeMode mode;
    uint64_t seed;
    int samples;
    int blocks;
    int k;
} ProbeConfig;

typedef struct {
    double active_count;
    double fired_count;
    double excluded_count;
    double baseline_sum;
    double baseline_sum_sq;
    double rbtc_sum;
    double rbtc_sum_sq;
    double mean_delta_sum_sq;
    double baseline_abs_mean;
    double rbtc_abs_mean;
    double support_states_total;
    double support_state_visits;
    double baseline_eval_count;
    int max_support_states_per_block;
} ProbeAccum;

typedef struct {
    const char *denominator_status;
    double baseline_var;
    double rbtc_var;
    double rho_var;
    double baseline_snr;
    double rbtc_snr;
    double snr_lift;
    double bias_rel;
    double conditional_ops_per_coordinate;
    double gamma_cost_proxy;
    double s_net_proxy;
    int zero_collapse_flag;
} ProbeDerived;

static float r394_uniform_signed(uint64_t *state, float lo, float hi) {
    const float u = uniform01(state);
    return lo + (hi - lo) * u;
}

static int r394_parse_int(const char *text, int *out_value) {
    char *end = NULL;
    long parsed;
    if (!text || !*text) {
        return 0;
    }
    parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        return 0;
    }
    *out_value = (int)parsed;
    return 1;
}

static int r394_parse_u64(const char *text, uint64_t *out_value) {
    char *end = NULL;
    unsigned long long parsed;
    if (!text || !*text) {
        return 0;
    }
    parsed = strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        return 0;
    }
    *out_value = (uint64_t)parsed;
    return 1;
}

static const char *r394_mode_name(ProbeMode mode) {
    if (mode == R394_MODE_SCALAR_FIXED_DW_AUDIT) {
        return "scalar_fixed_dw_audit";
    }
    return "tiny_block_exact";
}

static int r394_parse_mode(const char *text, ProbeMode *out_mode) {
    if (strcmp(text, "scalar_fixed_dw_audit") == 0) {
        *out_mode = R394_MODE_SCALAR_FIXED_DW_AUDIT;
        return 1;
    }
    if (strcmp(text, "tiny_block_exact") == 0) {
        *out_mode = R394_MODE_TINY_BLOCK_EXACT;
        return 1;
    }
    return 0;
}

static void r394_build_default_config(ProbeConfig *config) {
    config->mode = R394_MODE_TINY_BLOCK_EXACT;
    config->seed = 394ULL;
    config->samples = 200000;
    config->blocks = 4096;
    config->k = 8;
}

static int r394_parse_args(int argc, char **argv, ProbeConfig *config) {
    int idx;
    r394_build_default_config(config);
    for (idx = 1; idx < argc; ++idx) {
        if (strcmp(argv[idx], "--mode") == 0) {
            if (idx + 1 >= argc || !r394_parse_mode(argv[idx + 1], &config->mode)) {
                return 0;
            }
            ++idx;
        } else if (strcmp(argv[idx], "--seed") == 0) {
            if (idx + 1 >= argc || !r394_parse_u64(argv[idx + 1], &config->seed)) {
                return 0;
            }
            ++idx;
        } else if (strcmp(argv[idx], "--samples") == 0) {
            if (idx + 1 >= argc || !r394_parse_int(argv[idx + 1], &config->samples)) {
                return 0;
            }
            ++idx;
        } else if (strcmp(argv[idx], "--blocks") == 0) {
            if (idx + 1 >= argc || !r394_parse_int(argv[idx + 1], &config->blocks)) {
                return 0;
            }
            ++idx;
        } else if (strcmp(argv[idx], "--k") == 0) {
            if (idx + 1 >= argc || !r394_parse_int(argv[idx + 1], &config->k)) {
                return 0;
            }
            ++idx;
        } else {
            return 0;
        }
    }
    if (config->samples <= 0 || config->blocks <= 0 || config->k <= 0 || config->k > R394_RBTC_MAX_K) {
        return 0;
    }
    return 1;
}

static float r394_pick_shadow(uint64_t *rng, ShadowBin *out_bin) {
    const float u = uniform01(rng);
    float magnitude;
    float sign = (uniform01(rng) < 0.5f) ? -1.0f : 1.0f;
    if (u < 0.25f) {
        *out_bin = R394_BIN_NEAR_ZERO;
        magnitude = r394_uniform_signed(rng, 0.01f, 0.09f);
    } else if (u < 0.60f) {
        *out_bin = R394_BIN_MID;
        magnitude = r394_uniform_signed(rng, 0.18f, 0.72f);
    } else if (u < 0.90f) {
        *out_bin = R394_BIN_EDGE;
        magnitude = r394_uniform_signed(rng, 0.92f, 0.995f);
    } else {
        *out_bin = R394_BIN_CRYSTALLIZED;
        magnitude = r394_uniform_signed(rng, 1.02f, 1.22f);
    }
    return sign * magnitude;
}

static int r394_support_values(float s, float *values, double *probs) {
    if (s >= 1.0f) {
        values[0] = 1.0f;
        probs[0] = 1.0;
        return 1;
    }
    if (s <= -1.0f) {
        values[0] = -1.0f;
        probs[0] = 1.0;
        return 1;
    }
    if (s >= 0.0f) {
        values[0] = 0.0f;
        probs[0] = 1.0 - (double)s;
        values[1] = 1.0f;
        probs[1] = (double)s;
        return 2;
    }
    values[0] = -1.0f;
    probs[0] = (double)(-s);
    values[1] = 0.0f;
    probs[1] = 1.0 + (double)s;
    return 2;
}

static double r394_zero_state_factor(float w) {
    return (w == 0.0f) ? (double)V12_BI_STE_FACTOR : 1.0;
}

static double r394_predict_block(
    int active_k,
    const double *alpha,
    const double *pairwise,
    double bias,
    const float *weights
) {
    double pred = bias;
    int i;
    int j;
    for (i = 0; i < active_k; ++i) {
        pred += alpha[i] * (double)weights[i];
    }
    for (i = 0; i < active_k; ++i) {
        for (j = i + 1; j < active_k; ++j) {
            pred += pairwise[i * active_k + j] * (double)weights[i] * (double)weights[j];
        }
    }
    return pred;
}

static double r394_dw_coordinate(
    int coord_idx,
    int active_k,
    const double *alpha,
    const double *pairwise,
    const float *weights,
    double error
) {
    double feature = alpha[coord_idx];
    int other_idx;
    for (other_idx = 0; other_idx < active_k; ++other_idx) {
        if (other_idx == coord_idx) {
            continue;
        }
        if (coord_idx < other_idx) {
            feature += pairwise[coord_idx * active_k + other_idx] * (double)weights[other_idx];
        } else {
            feature += pairwise[other_idx * active_k + coord_idx] * (double)weights[other_idx];
        }
    }
    return error * feature;
}

static void r394_accumulate_metric(ProbeAccum *accum, double baseline_value, double rbtc_value) {
    accum->active_count += 1.0;
    if (fabs(baseline_value) > R394_RBTC_MEAN_EPS) {
        accum->fired_count += 1.0;
    }
    accum->baseline_sum += baseline_value;
    accum->baseline_sum_sq += baseline_value * baseline_value;
    accum->rbtc_sum += rbtc_value;
    accum->rbtc_sum_sq += rbtc_value * rbtc_value;
    accum->mean_delta_sum_sq += (rbtc_value - baseline_value) * (rbtc_value - baseline_value);
    accum->baseline_abs_mean += fabs(baseline_value);
    accum->rbtc_abs_mean += fabs(rbtc_value);
}

static void r394_run_scalar_fixed_dw_audit(const ProbeConfig *config, ProbeAccum *accum) {
    uint64_t rng = config->seed;
    int sample_idx;
    for (sample_idx = 0; sample_idx < config->samples; ++sample_idx) {
        ShadowBin bin;
        float s = r394_pick_shadow(&rng, &bin);
        float support_values[2];
        double support_probs[2];
        double fixed_dw;
        int support_size;
        int state_idx;
        double conditional = 0.0;
        double state_visits = 0.0;

        if (bin == R394_BIN_CRYSTALLIZED) {
            accum->excluded_count += 1.0;
            continue;
        }

        fixed_dw = (double)r394_uniform_signed(&rng, -1.3f, 1.3f);
        support_size = r394_support_values(s, support_values, support_probs);
        accum->support_states_total += (double)support_size;
        accum->support_state_visits += (double)support_size;
        accum->baseline_eval_count += 1.0;
        if (support_size > accum->max_support_states_per_block) {
            accum->max_support_states_per_block = support_size;
        }

        for (state_idx = 0; state_idx < support_size; ++state_idx) {
            conditional += support_probs[state_idx] * fixed_dw * r394_zero_state_factor(support_values[state_idx]);
            state_visits += 1.0;
        }
        if (state_visits <= 0.0) {
            continue;
        }

        {
            const float sampled = sample_ternary_from_logit(s, uniform01(&rng));
            const double baseline = fixed_dw * r394_zero_state_factor(sampled);
            r394_accumulate_metric(accum, baseline, conditional);
        }
    }
}

static void r394_build_exact_block(
    uint64_t *rng,
    int k,
    int *active_k_out,
    float *s_values,
    double *alpha,
    double *pairwise,
    double *target_out,
    double *bias_out
) {
    int idx;
    int active_k = 0;
    double target = (double)r394_uniform_signed(rng, -1.0f, 1.0f);
    double bias = (double)r394_uniform_signed(rng, -0.35f, 0.35f);
    for (idx = 0; idx < k; ++idx) {
        ShadowBin bin;
        float s = r394_pick_shadow(rng, &bin);
        if (bin == R394_BIN_CRYSTALLIZED) {
            continue;
        }
        s_values[active_k] = s;
        alpha[active_k] = (double)r394_uniform_signed(rng, -1.2f, 1.2f);
        ++active_k;
    }
    if (active_k == 0) {
        *active_k_out = 0;
        *target_out = target;
        *bias_out = bias;
        return;
    }
    for (idx = 0; idx < active_k * active_k; ++idx) {
        pairwise[idx] = 0.0;
    }
    for (idx = 0; idx < active_k; ++idx) {
        int jdx;
        for (jdx = idx + 1; jdx < active_k; ++jdx) {
            pairwise[idx * active_k + jdx] = (double)r394_uniform_signed(rng, -0.55f, 0.55f);
        }
    }
    for (idx = 0; idx < active_k; ++idx) {
        target += 0.08 * alpha[idx] * (double)s_values[idx];
    }
    *active_k_out = active_k;
    *target_out = target;
    *bias_out = bias;
}

static void r394_run_tiny_block_exact(const ProbeConfig *config, ProbeAccum *accum) {
    uint64_t rng = config->seed;
    int block_idx;
    for (block_idx = 0; block_idx < config->blocks; ++block_idx) {
        float s_values[R394_RBTC_MAX_K];
        double alpha[R394_RBTC_MAX_K];
        double pairwise[R394_RBTC_MAX_K * R394_RBTC_MAX_K];
        float sampled_weights[R394_RBTC_MAX_K];
        float enum_weights[R394_RBTC_MAX_K];
        float support_values[R394_RBTC_MAX_K][2];
        double support_probs[R394_RBTC_MAX_K][2];
        int support_sizes[R394_RBTC_MAX_K];
        int state_cursor[R394_RBTC_MAX_K];
        double conditional[R394_RBTC_MAX_K];
        double target = 0.0;
        double bias = 0.0;
        int active_k = 0;
        int idx;
        int support_states = 1;

        r394_build_exact_block(&rng, config->k, &active_k, s_values, alpha, pairwise, &target, &bias);
        accum->excluded_count += (double)(config->k - active_k);
        if (active_k <= 0) {
            continue;
        }

        for (idx = 0; idx < active_k; ++idx) {
            support_sizes[idx] = r394_support_values(s_values[idx], support_values[idx], support_probs[idx]);
            support_states *= support_sizes[idx];
            sampled_weights[idx] = sample_ternary_from_logit(s_values[idx], uniform01(&rng));
            conditional[idx] = 0.0;
        }

        accum->support_states_total += (double)support_states;
        accum->support_state_visits += (double)(support_states * active_k);
        accum->baseline_eval_count += (double)active_k;
        if (support_states > accum->max_support_states_per_block) {
            accum->max_support_states_per_block = support_states;
        }

        memset(state_cursor, 0, sizeof(state_cursor));
        for (;;) {
            double prob = 1.0;
            double pred;
            double error;
            for (idx = 0; idx < active_k; ++idx) {
                enum_weights[idx] = support_values[idx][state_cursor[idx]];
                prob *= support_probs[idx][state_cursor[idx]];
            }
            pred = r394_predict_block(active_k, alpha, pairwise, bias, enum_weights);
            error = pred - target;
            for (idx = 0; idx < active_k; ++idx) {
                const double dw = r394_dw_coordinate(idx, active_k, alpha, pairwise, enum_weights, error);
                conditional[idx] += prob * dw * r394_zero_state_factor(enum_weights[idx]);
            }

            for (idx = active_k - 1; idx >= 0; --idx) {
                state_cursor[idx] += 1;
                if (state_cursor[idx] < support_sizes[idx]) {
                    break;
                }
                state_cursor[idx] = 0;
            }
            if (idx < 0) {
                break;
            }
        }

        {
            const double pred = r394_predict_block(active_k, alpha, pairwise, bias, sampled_weights);
            const double error = pred - target;
            for (idx = 0; idx < active_k; ++idx) {
                const double dw = r394_dw_coordinate(idx, active_k, alpha, pairwise, sampled_weights, error);
                const double baseline = dw * r394_zero_state_factor(sampled_weights[idx]);
                r394_accumulate_metric(accum, baseline, conditional[idx]);
            }
        }
    }
}

static void r394_derive_metrics(const ProbeAccum *accum, ProbeDerived *derived) {
    const double count = accum->active_count;
    const double baseline_mean = (count > 0.0) ? accum->baseline_sum / count : 0.0;
    const double rbtc_mean = (count > 0.0) ? accum->rbtc_sum / count : 0.0;
    const double baseline_mean_sq = baseline_mean * baseline_mean;
    const double rbtc_mean_sq = rbtc_mean * rbtc_mean;
    const double mean_delta = rbtc_mean - baseline_mean;

    if (count <= 0.0) {
        derived->denominator_status = "NO_DATA";
        derived->baseline_var = 0.0;
        derived->rbtc_var = 0.0;
    } else {
        derived->baseline_var = accum->baseline_sum_sq / count - baseline_mean_sq;
        derived->rbtc_var = accum->rbtc_sum_sq / count - rbtc_mean_sq;
        if (derived->baseline_var < 0.0) {
            derived->baseline_var = 0.0;
        }
        if (derived->rbtc_var < 0.0) {
            derived->rbtc_var = 0.0;
        }
        if (accum->fired_count <= 0.0 || derived->baseline_var <= R394_RBTC_EPS) {
            derived->denominator_status = "NO_DENOMINATOR";
        } else {
            derived->denominator_status = "OK";
        }
    }

    derived->rho_var = 1.0 - (derived->rbtc_var / (derived->baseline_var + R394_RBTC_EPS));
    derived->baseline_snr = baseline_mean_sq / (derived->baseline_var + R394_RBTC_EPS);
    derived->rbtc_snr = rbtc_mean_sq / (derived->rbtc_var + R394_RBTC_EPS);
    derived->snr_lift = derived->rbtc_snr / (derived->baseline_snr + R394_RBTC_EPS);
    derived->bias_rel = fabs(mean_delta) / (fabs(baseline_mean) + R394_RBTC_EPS);
    derived->conditional_ops_per_coordinate = accum->support_state_visits / (accum->active_count + R394_RBTC_EPS);
    derived->gamma_cost_proxy = derived->conditional_ops_per_coordinate - 1.0;
    if (derived->gamma_cost_proxy < 0.0) {
        derived->gamma_cost_proxy = 0.0;
    }
    derived->s_net_proxy = derived->snr_lift / (1.0 + derived->gamma_cost_proxy);
    derived->zero_collapse_flag =
        (accum->baseline_abs_mean > R394_RBTC_MEAN_EPS && accum->rbtc_abs_mean <= R394_RBTC_MEAN_EPS) ? 1 : 0;
}

static void r394_decide(
    const ProbeConfig *config,
    const ProbeDerived *derived,
    int independence_shortcut_flag,
    const char **decision_out,
    const char **reason_out
) {
    if (strcmp(derived->denominator_status, "NO_DATA") == 0) {
        *decision_out = "kill";
        *reason_out = "no_active_coordinates";
        return;
    }
    if (strcmp(derived->denominator_status, "NO_DENOMINATOR") == 0) {
        *decision_out = "kill";
        *reason_out = "no_denominator";
        return;
    }
    if (derived->zero_collapse_flag != 0) {
        *decision_out = "kill";
        *reason_out = "zero_collapse";
        return;
    }
    if (config->mode != R394_MODE_TINY_BLOCK_EXACT) {
        *decision_out = "kill";
        *reason_out = "audit_only_mode";
        return;
    }
    if (independence_shortcut_flag != 0) {
        *decision_out = "kill";
        *reason_out = "independence_shortcut";
        return;
    }
    if (derived->bias_rel > 1.0e-3) {
        *decision_out = "kill";
        *reason_out = "bias_rel_above_floor";
        return;
    }
    if (derived->rho_var < 0.33) {
        *decision_out = "kill";
        *reason_out = "rho_var_below_kill";
        return;
    }
    if (derived->s_net_proxy <= 1.10) {
        *decision_out = "kill";
        *reason_out = "s_net_proxy_below_kill";
        return;
    }
    if (derived->rho_var >= 0.50 && derived->s_net_proxy >= 1.50) {
        *decision_out = "pass";
        *reason_out = "exact_thresholds_met";
        return;
    }
    if ((derived->rho_var >= 0.33 && derived->rho_var < 0.50) ||
        (derived->s_net_proxy > 1.10 && derived->s_net_proxy < 1.50)) {
        *decision_out = "weak";
        *reason_out = "conditional_signal";
        return;
    }
    *decision_out = "kill";
    *reason_out = "thresholds_not_met";
}

static void r394_print_usage(void) {
    fprintf(stderr,
            "usage: r394_rbtc_probe [--mode scalar_fixed_dw_audit|tiny_block_exact] "
            "[--seed N] [--samples N] [--blocks N] [--k N]\n");
}

int main(int argc, char **argv) {
    ProbeConfig config;
    ProbeAccum accum;
    ProbeDerived derived;
    const char *decision = "kill";
    const char *reason = "not_run";
    const char *mode_name;
    const char *estimator_identity = "zero_state_ste";
    int independence_shortcut_flag;

    if (!r394_parse_args(argc, argv, &config)) {
        r394_print_usage();
        return 1;
    }

    memset(&accum, 0, sizeof(accum));
    memset(&derived, 0, sizeof(derived));
    mode_name = r394_mode_name(config.mode);
    independence_shortcut_flag = (config.mode == R394_MODE_SCALAR_FIXED_DW_AUDIT) ? 1 : 0;

    printf("[R394][RBTC] activated mode=%s seed=%llu k=%d samples=%d\n",
           mode_name,
           (unsigned long long)config.seed,
           config.k,
           (config.mode == R394_MODE_SCALAR_FIXED_DW_AUDIT) ? config.samples : config.blocks);
    printf("[R394][RBTC] config blocks=%d scalar_samples=%d excluded_count=synthetic_reported\n",
           config.blocks,
           config.samples);

    if (config.mode == R394_MODE_SCALAR_FIXED_DW_AUDIT) {
        r394_run_scalar_fixed_dw_audit(&config, &accum);
    } else {
        r394_run_tiny_block_exact(&config, &accum);
    }

    r394_derive_metrics(&accum, &derived);
    r394_decide(&config, &derived, independence_shortcut_flag, &decision, &reason);

    printf("estimator_identity=%s\n", estimator_identity);
    printf("mode=%s\n", mode_name);
    printf("active_count=%.0f\n", accum.active_count);
    printf("fired_count=%.0f\n", accum.fired_count);
    printf("denominator_status=%s\n", derived.denominator_status);
    printf("baseline_var=%.9g\n", derived.baseline_var);
    printf("rbtc_var=%.9g\n", derived.rbtc_var);
    printf("rho_var=%.9g\n", derived.rho_var);
    printf("baseline_snr=%.9g\n", derived.baseline_snr);
    printf("rbtc_snr=%.9g\n", derived.rbtc_snr);
    printf("snr_lift=%.9g\n", derived.snr_lift);
    printf("bias_rel=%.9g\n", derived.bias_rel);
    printf("zero_collapse_flag=%d\n", derived.zero_collapse_flag);
    printf("independence_shortcut_flag=%d\n", independence_shortcut_flag);
    printf("support_states_total=%.0f\n", accum.support_states_total);
    printf("max_support_states_per_block=%d\n", accum.max_support_states_per_block);
    printf("conditional_ops_per_coordinate=%.9g\n", derived.conditional_ops_per_coordinate);
    printf("gamma_cost_proxy=%.9g\n", derived.gamma_cost_proxy);
    printf("s_net_proxy=%.9g\n", derived.s_net_proxy);
    printf("decision=%s\n", decision);
    printf("reason=%s\n", reason);

    return 0;
}
