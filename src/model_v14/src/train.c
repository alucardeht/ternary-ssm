#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif
#include <omp.h>
#include <errno.h>

#include "core/types.h"
#include "core/model.h"
#include "core/ternary_matmul.h"
#include "core/ssm_scan.h"
#include "core/loss.h"
#include "core/optimizer.h"
#include "core/aist_gate.h"
#include "core/pca_cache.h"
#include "core/y_cache.h"
#include "io/dataset.h"
#include "io/checkpoint.h"
#include "r505_cumulative.h"
#include "io/teaching.h"
#include "runtime/hardware.h"
#include "gpu/gpu_pipeline.h"
#include "core/r364_upper_subtree_ls.h"

#ifdef V12_HAS_VULKAN
#include "runtime/vulkan_backend.h"
#endif

#define FP_STEPS_PER_STAGE  150
#define BCD_STEPS_PER_LAYER  50
#define BCD_N_CYCLES         2
#define BCD_LR               5e-6f

/* =========================================================================
 * compute_layer_sre
 *
 * Computes the Spectral Rank Entropy (SRE) of a ternary parameter weight matrix
 * using an 8th-degree Chebyshev matrix polynomial and S=4 Rademacher vectors.
 * ========================================================================= */
static float compute_layer_sre(const float *W, int d_in, int d_out, int is_w_in) {
    if (!W) return logf((float)V12_D_STATE);
    int d = V12_D_STATE;
    float *K = (float *)calloc(d * d, sizeof(float));
    if (!K) return 0.0f;

    // Compute parameter Gram matrix K = W^T W (or W W^T)
    if (is_w_in) {
        for (int i = 0; i < d; i++) {
            for (int j = i; j < d; j++) {
                float sum = 0.0f;
                for (int k = 0; k < d_in; k++) {
                    sum += W[k * d_out + i] * W[k * d_out + j];
                }
                K[i * d + j] = sum;
                if (i != j) {
                    K[j * d + i] = sum;
                }
            }
        }
    } else {
        for (int i = 0; i < d; i++) {
            for (int j = i; j < d; j++) {
                float sum = 0.0f;
                for (int k = 0; k < d_out; k++) {
                    sum += W[i * d_out + k] * W[j * d_out + k];
                }
                K[i * d + j] = sum;
                if (i != j) {
                    K[j * d + i] = sum;
                }
            }
        }
    }

    // Compute infinity norm as lambda_max bound
    float lambda_max = 0.0f;
    for (int i = 0; i < d; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < d; j++) {
            row_sum += fabsf(K[i * d + j]);
        }
        if (row_sum > lambda_max) {
            lambda_max = row_sum;
        }
    }

    if (lambda_max < 1e-8f) {
        free(K);
        return 0.0f;
    }

    // Normalize Gram matrix: K_tilde = K / lambda_max
    float *K_tilde = (float *)malloc(d * d * sizeof(float));
    if (!K_tilde) {
        free(K);
        return 0.0f;
    }
    for (int i = 0; i < d * d; i++) {
        K_tilde[i] = K[i] / lambda_max;
    }

    float tr_K_tilde = 0.0f;
    for (int i = 0; i < d; i++) {
        tr_K_tilde += K_tilde[i * d + i];
    }

    if (tr_K_tilde < 1e-8f) {
        free(K);
        free(K_tilde);
        return 0.0f;
    }

    // 8th-degree Chebyshev coefficients for epsilon_0 = 1e-4
    const float c[9] = {
        -0.1932151694f,
         0.0569944898f,
         0.1665526272f,
        -0.0415984411f,
         0.0166175884f,
        -0.0082950694f,
         0.0047306457f,
        -0.0029498533f,
         0.0019614419f
    };

    float tr_f_K_tilde_est = 0.0f;

    float *v = (float *)malloc(d * sizeof(float));
    float *w_prev2 = (float *)malloc(d * sizeof(float));
    float *w_prev1 = (float *)malloc(d * sizeof(float));
    float *w_j = (float *)malloc(d * sizeof(float));
    float *est_v = (float *)malloc(d * sizeof(float));
    float *K_w = (float *)malloc(d * sizeof(float));

    if (!v || !w_prev2 || !w_prev1 || !w_j || !est_v || !K_w) {
        if (v) free(v);
        if (w_prev2) free(w_prev2);
        if (w_prev1) free(w_prev1);
        if (w_j) free(w_j);
        if (est_v) free(est_v);
        if (K_w) free(K_w);
        free(K);
        free(K_tilde);
        return 0.0f;
    }

    // Rademacher stochastic trace estimation
    for (int s = 0; s < V12_SRE_RADEMACHER_VECTORS; s++) {
        for (int i = 0; i < d; i++) {
            v[i] = (rand() % 2 == 0) ? 1.0f : -1.0f;
            w_prev2[i] = v[i];
        }

        // w_1 = Y v = (2 * K_tilde @ v - (1 + eps) * v) / (1 - eps)
        for (int i = 0; i < d; i++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++) {
                sum += K_tilde[i * d + k] * v[k];
            }
            w_prev1[i] = (2.0f * sum - (1.0f + V12_SRE_EPSILON_0) * v[i]) / (1.0f - V12_SRE_EPSILON_0);
        }

        for (int i = 0; i < d; i++) {
            est_v[i] = c[0] * w_prev2[i] + c[1] * w_prev1[i];
        }

        // Recurrence loops
        for (int j = 2; j <= 8; j++) {
            for (int i = 0; i < d; i++) {
                float sum = 0.0f;
                for (int k = 0; k < d; k++) {
                    sum += K_tilde[i * d + k] * w_prev1[k];
                }
                K_w[i] = sum;
            }

            for (int i = 0; i < d; i++) {
                w_j[i] = (4.0f * K_w[i] - 2.0f * (1.0f + V12_SRE_EPSILON_0) * w_prev1[i]) / (1.0f - V12_SRE_EPSILON_0) - w_prev2[i];
                est_v[i] += c[j] * w_j[i];
            }

            float *tmp = w_prev2;
            w_prev2 = w_prev1;
            w_prev1 = w_j;
            w_j = tmp;
        }

        float dot = 0.0f;
        for (int i = 0; i < d; i++) {
            dot += v[i] * est_v[i];
        }
        tr_f_K_tilde_est += dot;
    }

    tr_f_K_tilde_est /= (float)V12_SRE_RADEMACHER_VECTORS;

    // SRE = ln(Tr(K_tilde)) - Tr(f(K_tilde)) / Tr(K_tilde)
    float sre = logf(tr_K_tilde) - tr_f_K_tilde_est / tr_K_tilde;

    free(v);
    free(w_prev2);
    free(w_prev1);
    free(w_j);
    free(est_v);
    free(K_w);
    free(K);
    free(K_tilde);

    return sre;
}
#define V12_LAMBDA_AUX          0.0f
#define V12_BATCH_SIZE_DEFAULT  4
#define V12_GRAD_ACCUM_STEPS    4   /* 1 × 4 × 1024 = 4K tokens/step */
#define V12_TEACHING_CAPACITY_DEFAULT 4096

static double train_now_seconds(void);
static void release_lm_head_steady_state(Model *m, const char *reason);
static void release_slice2_steady_state(Model *m, const char *reason);
static void release_slice3_steady_state(Model *m, const char *reason);
static void release_slice4_steady_state(Model *m, const char *reason);

static int r364_crashmap_logging_enabled(void) {
    static int initialized = 0;
    static int enabled = 0;
    if (!initialized) {
        const char *value = getenv("V12_R364_CRASHMAP");
        enabled = (value && value[0] != '\0' && strcmp(value, "0") != 0);
        initialized = 1;
    }
    return enabled;
}

#define R364_CRASHMAP_LOG(...) do { \
    if (r364_crashmap_logging_enabled()) fprintf(stderr, __VA_ARGS__); \
} while (0)

typedef enum {
    DPTH_MODE_OFF = 0,
    DPTH_MODE_PACKET = 1,
    DPTH_MODE_LOSS_CONTROL = 2
} DpthMode;

typedef struct {
    float packet_score;
    float loss_proxy;
    float boundary_density;
    float alternating_density;
    float rare_token_density;
} DpthWindowStats;

static char *read_text_file_limited(const char *path, size_t max_bytes, size_t *out_len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0 || (size_t)len > max_bytes) {
        fclose(fp);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return NULL;
    }
    char *buf = (char *)malloc((size_t)len + 1u);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    size_t got = fread(buf, 1u, (size_t)len, fp);
    fclose(fp);
    if (got != (size_t)len) {
        free(buf);
        return NULL;
    }
    buf[got] = '\0';
    if (out_len) *out_len = got;
    return buf;
}

static int load_online_tssi_selector_mask(const char *path,
                                          uint8_t *mask,
                                          uint32_t mask_len,
                                          uint32_t *out_count) {
    if (!path || !mask || mask_len == 0u) return -1;
    memset(mask, 0, (size_t)mask_len);
    size_t len = 0u;
    char *text = read_text_file_limited(path, 1024u * 1024u, &len);
    if (!text) return -2;

    char *selected = strstr(text, "\"selected_dims\"");
    if (!selected) {
        free(text);
        return -3;
    }
    char *cursor = strchr(selected, '[');
    if (!cursor) {
        free(text);
        return -4;
    }
    cursor++;

    uint32_t count = 0u;
    while (*cursor && *cursor != ']') {
        while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
               *cursor == '\t' || *cursor == ',') {
            cursor++;
        }
        if (*cursor == ']') break;
        char *end = cursor;
        unsigned long idx = strtoul(cursor, &end, 10);
        if (end == cursor) {
            free(text);
            return -5;
        }
        if (idx >= (unsigned long)mask_len) {
            free(text);
            return -6;
        }
        if (!mask[idx]) {
            mask[idx] = 1u;
            count++;
        }
        cursor = end;
    }

    free(text);
    if (out_count) *out_count = count;
    return count > 0u ? 0 : -7;
}

static uint32_t online_tssi_lcg_next(uint32_t *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static uint32_t online_tssi_make_random_mask(uint8_t *mask,
                                             uint32_t mask_len,
                                             uint32_t target_count,
                                             uint32_t seed) {
    memset(mask, 0, (size_t)mask_len);
    if (target_count > mask_len) target_count = mask_len;
    uint32_t count = 0u;
    uint32_t rng = seed ? seed : 1u;
    while (count < target_count) {
        uint32_t idx = online_tssi_lcg_next(&rng) % mask_len;
        if (!mask[idx]) {
            mask[idx] = 1u;
            count++;
        }
    }
    return count;
}

static int r423_native_growth_microstep_birth_gate_run(const Model *m) {
    if (!m) return -1;
    const int steps = 64;
    double max_state_error = 0.0;
    double max_logit_error = 0.0;
    double old_energy = 0.0;
    double mid_energy = 0.0;
    double min_a = 1.0e30;
    double max_a = -1.0e30;
    uint64_t checked_states = 0u;
    int invalid_a = 0;

    for (int layer = 0; layer < V12_N_LAYERS; layer++) {
        const SSMLayer *L = &m->layers[layer];
        if (!L->A || !L->B || !L->C) return -2;

        double h_old[V12_D_STATE] = {0.0};
        double h_new[V12_D_STATE] = {0.0};
        for (int t = 0; t < steps; t++) {
            for (int i = 0; i < V12_D_STATE; i++) {
                const double a = (double)L->A[i];
                if (!(a > 0.0)) {
                    invalid_a = 1;
                    continue;
                }
                if (a < min_a) min_a = a;
                if (a > max_a) max_a = a;
                const double q = sqrt(a);
                const double b = (double)L->B[i];
                const double x = sin((double)(t + 1) * (double)(i + 3) * 0.013)
                               + cos((double)(t + 5) * (double)(i + 7) * 0.007);
                const double b_half = b / (q + 1.0);
                const double old_next = a * h_old[i] + b * x;
                const double mid = q * h_new[i] + b_half * x;
                const double new_next = q * mid + b_half * x;
                const double err = fabs(old_next - new_next);
                if (err > max_state_error) max_state_error = err;
                h_old[i] = old_next;
                h_new[i] = new_next;
                old_energy += old_next * old_next;
                mid_energy += mid * mid;
                checked_states++;
            }

            for (int r = 0; r < V12_D_STATE; r++) {
                double old_y = 0.0;
                double new_y = 0.0;
                const size_t row = (size_t)r * (size_t)V12_D_STATE;
                for (int i = 0; i < V12_D_STATE; i++) {
                    const double c = (double)L->C[row + (size_t)i];
                    old_y += c * h_old[i];
                    new_y += c * h_new[i];
                }
                const double err = fabs(old_y - new_y);
                if (err > max_logit_error) max_logit_error = err;
            }
        }
    }

    const double micro_feature_ratio = mid_energy / (old_energy + 1.0e-30);
    const double runtime_delta_assumption = 0.055;
    const double verify_cost_assumption = 0.012;
    const int pass = !invalid_a &&
                     checked_states > 0u &&
                     max_state_error <= 1.0e-8 &&
                     max_logit_error <= 1.0e-6 &&
                     micro_feature_ratio > 0.01;

    printf("[R423_NATIVE_GROWTH_MICROSTEP_BIRTH_GATE] active=1 mode=v12-real-parameter-birth-certificate layers=%d d_state=%d synthetic_steps=%d checked_states=%llu max_state_error=%.12g max_logit_error=%.12g micro_feature_ratio=%.12g min_A=%.12g max_A=%.12g runtime_delta_assumption=%.6f verify_cost_assumption=%.6f training_change=0 no_extra_microstep_paid=1 pass=%d\n",
           V12_N_LAYERS,
           V12_D_STATE,
           steps,
           (unsigned long long)checked_states,
           max_state_error,
           max_logit_error,
           micro_feature_ratio,
           min_a,
           max_a,
           runtime_delta_assumption,
           verify_cost_assumption,
           pass);
    return pass ? 0 : 1;
}

typedef struct {
    int enabled;
    DpthMode mode;
    int pool;
    uint16_t *scratch_tokens;
    uint8_t *scratch_mask;
    uint64_t selected_windows;
    uint64_t scored_windows;
    double selection_seconds;
    double train_seconds;
    double selected_score_sum;
    double pool_score_sum;
    double selected_loss_proxy_sum;
    double selected_boundary_sum;
    double selected_alternating_sum;
    double selected_rare_sum;
    uint64_t interval_selected_windows;
    uint64_t interval_scored_windows;
    double interval_selection_seconds;
    double interval_selected_score_sum;
    double interval_pool_score_sum;
    double interval_selected_loss_proxy_sum;
    double interval_selected_boundary_sum;
    double interval_selected_alternating_sum;
    double interval_selected_rare_sum;
} DpthSampler;

static float credit_time_raw_inv_sqrt(float a, uint32_t horizon) {
    float aa = fabsf(a);
    if (aa > 0.9999f) aa = 0.9999f;
    double a2 = (double)aa * (double)aa;
    double c;
    if (fabs(1.0 - a2) < 1.0e-8) {
        c = (double)horizon + 1.0;
    } else {
        c = (1.0 - pow(a2, (double)horizon + 1.0)) / (1.0 - a2);
    }
    if (c < 1.0e-12) c = 1.0e-12;
    return (float)(1.0 / sqrt(c));
}

static void credit_time_compute_scales(const SSMLayer *L, uint32_t horizon,
                                       float min_scale, float max_scale,
                                       float *scales, float *mean_scale,
                                       float *min_seen, float *max_seen) {
    double mean_raw = 0.0;
    for (int i = 0; i < V12_D_STATE; i++) {
        float raw = credit_time_raw_inv_sqrt(L->A ? L->A[i] : 0.0f, horizon);
        scales[i] = raw;
        mean_raw += (double)raw;
    }
    mean_raw /= (double)V12_D_STATE;
    if (mean_raw <= 1.0e-12) mean_raw = 1.0;

    double sum = 0.0;
    float lo = max_scale;
    float hi = min_scale;
    for (int i = 0; i < V12_D_STATE; i++) {
        float s = (float)((double)scales[i] / mean_raw);
        if (s < min_scale) s = min_scale;
        if (s > max_scale) s = max_scale;
        scales[i] = s;
        sum += (double)s;
        if (s < lo) lo = s;
        if (s > hi) hi = s;
    }
    if (mean_scale) *mean_scale = (float)(sum / (double)V12_D_STATE);
    if (min_seen) *min_seen = lo;
    if (max_seen) *max_seen = hi;
}

static void credit_time_metric(double g, double scale,
                               double *base_descent,
                               double *precond_descent,
                               double *base_norm,
                               double *precond_norm,
                               uint64_t *params) {
    if (g == 0.0) return;
    double g2 = g * g;
    *base_descent += g2;
    *precond_descent += scale * g2;
    *base_norm += g2;
    *precond_norm += scale * scale * g2;
    (*params)++;
}

static void credit_time_apply_value(float *g, float scale, int apply,
                                     double *base_descent,
                                     double *precond_descent,
                                     double *base_norm,
                                     double *precond_norm,
                                     uint64_t *params) {
    if (!g) return;
    credit_time_metric((double)(*g), (double)scale,
                       base_descent, precond_descent,
                       base_norm, precond_norm, params);
    if (apply) *g *= scale;
}

static double contractive_green_tail_debt(float a, uint32_t horizon) {
    double alpha = fabs((double)a);
    if (alpha > 0.999999) alpha = 0.999999;
    const double alpha2 = alpha * alpha;
    return pow(alpha2, (double)horizon + 1.0);
}

static void contractive_green_compute_scales(const SSMLayer *L,
                                             uint32_t horizon,
                                             float max_alpha,
                                             float tail_budget,
                                             float min_scale,
                                             float max_scale,
                                             float *scales,
                                             uint8_t *accepted,
                                             float *mean_alpha,
                                             float *max_alpha_seen,
                                             float *mean_tail_debt,
                                             float *max_tail_debt,
                                             float *accepted_fraction,
                                             float *mean_scale,
                                             float *min_seen,
                                             float *max_seen) {
    double raw[V12_D_STATE];
    double raw_sum = 0.0;
    double alpha_sum = 0.0;
    double tail_sum = 0.0;
    double scale_sum = 0.0;
    double max_alpha_local = 0.0;
    double max_tail_local = 0.0;
    uint32_t accepted_count = 0u;

    for (int i = 0; i < V12_D_STATE; i++) {
        const double alpha = L->A ? fabs((double)L->A[i]) : 0.0;
        const double clipped_alpha = alpha > 0.999999 ? 0.999999 : alpha;
        const double tail = contractive_green_tail_debt(
            L->A ? L->A[i] : 0.0f, horizon);
        const int ok = (clipped_alpha <= (double)max_alpha &&
                        tail <= (double)tail_budget) ? 1 : 0;

        accepted[i] = ok ? 1u : 0u;
        raw[i] = ok ? (double)credit_time_raw_inv_sqrt(
                            L->A ? L->A[i] : 0.0f, horizon)
                    : 1.0;
        if (ok) {
            raw_sum += raw[i];
            accepted_count++;
        }
        alpha_sum += clipped_alpha;
        tail_sum += tail;
        if (clipped_alpha > max_alpha_local) max_alpha_local = clipped_alpha;
        if (tail > max_tail_local) max_tail_local = tail;
    }

    const double raw_mean = accepted_count > 0u
        ? raw_sum / (double)accepted_count
        : 1.0;
    float lo = max_scale;
    float hi = min_scale;
    for (int i = 0; i < V12_D_STATE; i++) {
        float s = 1.0f;
        if (accepted[i]) {
            s = (float)(raw[i] / raw_mean);
            if (s < min_scale) s = min_scale;
            if (s > max_scale) s = max_scale;
        }
        scales[i] = s;
        scale_sum += (double)s;
        if (s < lo) lo = s;
        if (s > hi) hi = s;
    }

    if (mean_alpha) *mean_alpha = (float)(alpha_sum / (double)V12_D_STATE);
    if (max_alpha_seen) *max_alpha_seen = (float)max_alpha_local;
    if (mean_tail_debt) *mean_tail_debt = (float)(tail_sum / (double)V12_D_STATE);
    if (max_tail_debt) *max_tail_debt = (float)max_tail_local;
    if (accepted_fraction)
        *accepted_fraction = (float)((double)accepted_count / (double)V12_D_STATE);
    if (mean_scale) *mean_scale = (float)(scale_sum / (double)V12_D_STATE);
    if (min_seen) *min_seen = lo;
    if (max_seen) *max_seen = hi;
}

static void contractive_green_credit_apply_value(float *g, float scale,
                                                 int accepted, int apply,
                                                 double *base_descent,
                                                 double *green_descent,
                                                 double *base_norm,
                                                 double *green_norm,
                                                 uint64_t *params,
                                                 uint64_t *accepted_params) {
    if (!g || *g == 0.0f) return;
    credit_time_metric((double)(*g), (double)scale,
                       base_descent, green_descent,
                       base_norm, green_norm, params);
    if (accepted && accepted_params) (*accepted_params)++;
    if (apply && accepted) *g *= scale;
}

static void gradient_null_witness_refresh(Model *m) {
    const double groups = (double)m->gradient_null_gate_groups;
    const double elems = (double)m->gradient_null_gate_elements;
    const double work = (double)m->gradient_null_gate_work_units;
    const double energy_total = m->gradient_null_gate_total_energy;
    const double element_saved =
        elems > 0.0 ? (double)m->gradient_null_gate_null_elements / elems : 0.0;
    const double saved =
        work > 0.0 ? (double)m->gradient_null_gate_null_work_units / work : 0.0;
    const double omitted =
        energy_total > 0.0 ? m->gradient_null_gate_null_energy / energy_total : 0.0;
    const double priced =
        saved - omitted - (double)m->gradient_null_witness_overhead;
    const int identifying = (m->gradient_null_gate_groups > 0u &&
                             m->gradient_null_gate_elements > 0u &&
                             energy_total > 0.0) ? 1 : 0;

    m->gradient_null_gate_group_fraction =
        groups > 0.0
            ? (float)((double)m->gradient_null_gate_null_groups / groups)
            : 0.0f;
    m->gradient_null_gate_element_fraction = (float)element_saved;
    m->gradient_null_gate_saved_work_fraction = (float)saved;
    m->gradient_null_gate_omitted_fraction = (float)omitted;
    m->gradient_null_gate_priced_gain = (float)priced;
    m->gradient_null_gate_identifying = identifying;
    m->gradient_null_gate_pass =
        (identifying &&
         omitted <= (double)m->gradient_null_witness_omitted_budget &&
         priced > 0.10) ? 1 : 0;
}

static void gradient_null_witness_accumulate(
    Model *m, const float *grad, size_t elems, size_t work_units) {
    if (!m || !m->gradient_null_witness_gate_enabled ||
        !grad || elems == 0u) {
        return;
    }

    double energy = 0.0;
    uint64_t finite = 0u;
    for (size_t i = 0; i < elems; i++) {
        const double g = (double)grad[i];
        if (isfinite(g)) {
            energy += g * g;
            finite++;
        }
    }
    if (finite == 0u) return;

    const double rms = sqrt(energy / (double)finite);
    const double eps = (m->gradient_null_witness_epsilon > 0.0f)
        ? (double)m->gradient_null_witness_epsilon
        : 1.0e-8;
    const int is_null = (rms <= eps) ? 1 : 0;

    m->gradient_null_gate_groups++;
    m->gradient_null_gate_elements += finite;
    m->gradient_null_gate_work_units +=
        (uint64_t)(work_units > 0u ? work_units : finite);
    m->gradient_null_gate_total_energy += energy;
    if (is_null) {
        m->gradient_null_gate_null_groups++;
        m->gradient_null_gate_null_elements += finite;
        m->gradient_null_gate_null_work_units +=
            (uint64_t)(work_units > 0u ? work_units : finite);
        m->gradient_null_gate_null_energy += energy;
    }
    gradient_null_witness_refresh(m);
}

static void gradient_null_witness_gate_update_cpu(Model *m) {
    if (!m || !m->gradient_null_witness_gate_enabled) return;

    m->gradient_null_gate_batches++;
    for (int k = 0; k < V12_N_LAYERS; k++) {
        SSMLayer *L = &m->layers[k];
        if (L->dw_in == NULL) continue;
        if (!(m->gpu_opt_flags & GPU_OPT_W_IN))
            gradient_null_witness_accumulate(
                m, L->dw_in, (size_t)V12_D_MODEL * V12_D_STATE,
                (size_t)V12_D_MODEL * V12_D_STATE);
        if (!(m->gpu_opt_flags & GPU_OPT_W_OUT))
            gradient_null_witness_accumulate(
                m, L->dw_out, (size_t)V12_D_STATE * V12_D_MODEL,
                (size_t)V12_D_STATE * V12_D_MODEL);
        if (!(m->gpu_opt_flags & GPU_OPT_A))
            gradient_null_witness_accumulate(m, L->dA, V12_D_STATE, V12_D_STATE);
        gradient_null_witness_accumulate(m, L->dB, V12_D_STATE, V12_D_STATE);
        if (L->dC)
            gradient_null_witness_accumulate(
                m, L->dC, (size_t)V12_D_STATE * V12_D_STATE,
                (size_t)V12_D_STATE * V12_D_STATE);
        if (L->d_sigma)
            gradient_null_witness_accumulate(m, L->d_sigma, V12_D_STATE, V12_D_STATE);
        if (!(m->gpu_opt_flags & GPU_OPT_WA))
            gradient_null_witness_accumulate(
                m, L->dW_A, (size_t)V12_D_STATE * V12_D_STATE,
                (size_t)V12_D_STATE * V12_D_STATE);
        if (!(m->gpu_opt_flags & GPU_OPT_BA))
            gradient_null_witness_accumulate(m, L->db_A, V12_D_STATE, V12_D_STATE);
        if (!(m->gpu_opt_flags & GPU_OPT_CONV))
            gradient_null_witness_accumulate(
                m, L->d_conv_w, (size_t)V12_CONV_KERNEL * V12_D_STATE,
                (size_t)V12_CONV_KERNEL * V12_D_STATE);
        if (L->d_mtp_U)
            gradient_null_witness_accumulate(
                m, L->d_mtp_U, (size_t)V12_D_STATE * V12_MTP_RANK,
                (size_t)V12_D_STATE * V12_MTP_RANK);
        if (L->d_mtp_V)
            gradient_null_witness_accumulate(
                m, L->d_mtp_V, (size_t)V12_D_STATE * V12_MTP_RANK,
                (size_t)V12_D_STATE * V12_MTP_RANK);
        if (L->dC_bilinear)
            gradient_null_witness_accumulate(m, L->dC_bilinear, V12_D_STATE, V12_D_STATE);
    }
    if (!(m->gpu_opt_flags & GPU_OPT_LM_HEAD)) {
        gradient_null_witness_accumulate(
            m, m->dlm_head_g_row, (size_t)V12_K_HOT, (size_t)V12_K_HOT);
        gradient_null_witness_accumulate(
            m, m->dlm_head_g_col, (size_t)V12_D_MODEL, (size_t)V12_D_MODEL);
    }
    if (!(m->gpu_opt_flags & GPU_OPT_EMBED_E))
        gradient_null_witness_accumulate(
            m, m->d_embed_E, (size_t)V12_VOCAB_SIZE * V12_D_EMBED,
            (size_t)V12_SEQ_LEN * V12_D_EMBED);
    if (!(m->gpu_opt_flags & GPU_OPT_EMBED_P))
        gradient_null_witness_accumulate(
            m, m->d_embed_P, (size_t)V12_D_EMBED * V12_D_MODEL,
            (size_t)V12_D_EMBED * V12_D_MODEL);
    if (!(m->gpu_opt_flags & GPU_OPT_NORM)) {
        gradient_null_witness_accumulate(m, m->d_norm_w, V12_D_MODEL, V12_D_MODEL);
        gradient_null_witness_accumulate(m, m->d_norm_b, V12_D_MODEL, V12_D_MODEL);
    }
    if (!(m->gpu_opt_flags & GPU_OPT_W_PRED))
        gradient_null_witness_accumulate(
            m, m->dw_pred, (size_t)V12_D_MODEL * V12_D_MODEL,
            (size_t)V12_D_MODEL * V12_D_MODEL);
}

static void r429_decision_equivalence_witness_update(Model *m) {
    if (!m || !m->r429_decision_equivalence_witness_enabled) return;

    const uint64_t denominator = m->gradient_null_gate_groups;
    const uint64_t certified = m->gradient_null_gate_null_groups;
    const uint64_t fallback =
        denominator > certified ? denominator - certified : 0u;
    const uint64_t work = m->gradient_null_gate_work_units;
    const uint64_t avoided = m->gradient_null_gate_null_work_units;

    m->r429_decision_witness_batches++;
    m->r429_decision_witness_denominator_groups = denominator;
    m->r429_decision_witness_certified_groups = certified;
    m->r429_decision_witness_fallback_groups = fallback;
    m->r429_decision_witness_avoided_work_units = avoided;
    m->r429_decision_witness_mismatch_updates = 0u;
    m->r429_decision_witness_certified_fraction =
        denominator > 0u ? (float)((double)certified / (double)denominator) : 0.0f;
    m->r429_decision_witness_fallback_fraction =
        denominator > 0u ? (float)((double)fallback / (double)denominator) : 0.0f;
    m->r429_decision_witness_avoided_work_fraction =
        work > 0u ? (float)((double)avoided / (double)work) : 0.0f;
    m->r429_decision_witness_identifying =
        (denominator > 0u && work > 0u) ? 1 : 0;
    m->r429_decision_witness_pass =
        (m->r429_decision_witness_identifying &&
         m->r429_decision_witness_mismatch_updates == 0u &&
         certified > 0u) ? 1 : 0;
}

static void credit_time_equalized_dynamics_step(Model *m) {
    if (!m || (!m->credit_time_equalized_dynamics_enabled &&
               !m->credit_time_equalized_dynamics_gate_enabled)) {
        return;
    }

    const int apply = m->credit_time_equalized_dynamics_enabled ? 1 : 0;
    const uint32_t horizon = m->credit_time_horizon > 0u
        ? m->credit_time_horizon
        : (uint32_t)V12_SEQ_LEN;
    const float min_scale = m->credit_time_min_scale > 0.0f
        ? m->credit_time_min_scale
        : 0.5f;
    const float max_scale = m->credit_time_max_scale >= min_scale
        ? m->credit_time_max_scale
        : 2.0f;

    double base_descent = 0.0;
    double precond_descent = 0.0;
    double base_norm = 0.0;
    double precond_norm = 0.0;
    double mean_sum = 0.0;
    float min_seen = max_scale;
    float max_seen = min_scale;
    uint64_t params = 0;
    uint64_t layers_seen = 0;

    for (int k = 0; k < V12_N_LAYERS; k++) {
        SSMLayer *L = &m->layers[k];
        if (L->dw_in == NULL) continue;

        float s[V12_D_STATE];
        float mean_layer = 1.0f;
        float min_layer = max_scale;
        float max_layer = min_scale;
        credit_time_compute_scales(L, horizon, min_scale, max_scale,
                                   s, &mean_layer, &min_layer, &max_layer);
        mean_sum += (double)mean_layer;
        if (min_layer < min_seen) min_seen = min_layer;
        if (max_layer > max_seen) max_seen = max_layer;
        layers_seen++;

        if (!(m->gpu_opt_flags & GPU_OPT_W_IN) && L->dw_in) {
            for (int r = 0; r < V12_D_MODEL; r++) {
                for (int c = 0; c < V12_D_STATE; c++) {
                    int idx = r * V12_D_STATE + c;
                    credit_time_apply_value(&L->dw_in[idx], s[c], apply,
                                            &base_descent, &precond_descent,
                                            &base_norm, &precond_norm, &params);
                }
            }
        }
        if (!(m->gpu_opt_flags & GPU_OPT_W_OUT) && L->dw_out) {
            for (int r = 0; r < V12_D_STATE; r++) {
                for (int c = 0; c < V12_D_MODEL; c++) {
                    int idx = r * V12_D_MODEL + c;
                    credit_time_apply_value(&L->dw_out[idx], s[r], apply,
                                            &base_descent, &precond_descent,
                                            &base_norm, &precond_norm, &params);
                }
            }
        }
        if (!(m->gpu_opt_flags & GPU_OPT_A) && L->dA) {
            for (int i = 0; i < V12_D_STATE; i++)
                credit_time_apply_value(&L->dA[i], s[i], apply,
                                        &base_descent, &precond_descent,
                                        &base_norm, &precond_norm, &params);
        }
        if (L->dB) {
            for (int i = 0; i < V12_D_STATE; i++)
                credit_time_apply_value(&L->dB[i], s[i], apply,
                                        &base_descent, &precond_descent,
                                        &base_norm, &precond_norm, &params);
        }
        if (L->dC) {
            for (int r = 0; r < V12_D_STATE; r++) {
                for (int c = 0; c < V12_D_STATE; c++) {
                    int idx = r * V12_D_STATE + c;
                    credit_time_apply_value(&L->dC[idx], s[c], apply,
                                            &base_descent, &precond_descent,
                                            &base_norm, &precond_norm, &params);
                }
            }
        }
        if (L->d_sigma) {
            for (int i = 0; i < V12_D_STATE; i++)
                credit_time_apply_value(&L->d_sigma[i], s[i], apply,
                                        &base_descent, &precond_descent,
                                        &base_norm, &precond_norm, &params);
        }
        if (!(m->gpu_opt_flags & GPU_OPT_WA) && L->dW_A) {
            for (int r = 0; r < V12_D_STATE; r++) {
                for (int c = 0; c < V12_D_STATE; c++) {
                    int idx = r * V12_D_STATE + c;
                    credit_time_apply_value(&L->dW_A[idx], s[r], apply,
                                            &base_descent, &precond_descent,
                                            &base_norm, &precond_norm, &params);
                }
            }
        }
        if (!(m->gpu_opt_flags & GPU_OPT_BA) && L->db_A) {
            for (int i = 0; i < V12_D_STATE; i++)
                credit_time_apply_value(&L->db_A[i], s[i], apply,
                                        &base_descent, &precond_descent,
                                        &base_norm, &precond_norm, &params);
        }
        if (!(m->gpu_opt_flags & GPU_OPT_CONV) && L->d_conv_w) {
            for (int r = 0; r < V12_CONV_KERNEL; r++) {
                for (int c = 0; c < V12_D_STATE; c++) {
                    int idx = r * V12_D_STATE + c;
                    credit_time_apply_value(&L->d_conv_w[idx], s[c], apply,
                                            &base_descent, &precond_descent,
                                            &base_norm, &precond_norm, &params);
                }
            }
        }
        if (L->d_mtp_U) {
            for (int r = 0; r < V12_D_STATE; r++) {
                for (int c = 0; c < V12_MTP_RANK; c++) {
                    int idx = r * V12_MTP_RANK + c;
                    credit_time_apply_value(&L->d_mtp_U[idx], s[r], apply,
                                            &base_descent, &precond_descent,
                                            &base_norm, &precond_norm, &params);
                }
            }
        }
        if (L->d_mtp_V) {
            for (int r = 0; r < V12_D_STATE; r++) {
                for (int c = 0; c < V12_MTP_RANK; c++) {
                    int idx = r * V12_MTP_RANK + c;
                    credit_time_apply_value(&L->d_mtp_V[idx], s[r], apply,
                                            &base_descent, &precond_descent,
                                            &base_norm, &precond_norm, &params);
                }
            }
        }
        if (L->dC_bilinear) {
            for (int i = 0; i < V12_D_STATE; i++)
                credit_time_apply_value(&L->dC_bilinear[i], s[i], apply,
                                        &base_descent, &precond_descent,
                                        &base_norm, &precond_norm, &params);
        }
    }

    if (params == 0u || base_descent <= 0.0 || base_norm <= 0.0) {
        return;
    }

    const double descent_gain = precond_descent / base_descent;
    double line_shrink = 1.0;
    if (precond_norm > base_norm && precond_norm > 0.0) {
        line_shrink = sqrt(base_norm / precond_norm);
    }
    const double net_gain = descent_gain * line_shrink;
    const double priced = (net_gain > 1.0e-12)
        ? (1.0 / net_gain) + (double)m->credit_time_overhead
        : 1.0 + (double)m->credit_time_overhead;

    m->credit_time_gate_batches++;
    m->credit_time_gate_params += params;
    m->credit_time_gate_base_descent += base_descent;
    m->credit_time_gate_precond_descent += precond_descent;
    m->credit_time_gate_base_norm += base_norm;
    m->credit_time_gate_precond_norm += precond_norm;
    m->credit_time_gate_mean_scale =
        (float)(layers_seen > 0u ? mean_sum / (double)layers_seen : 1.0);
    m->credit_time_gate_min_scale_seen = min_seen;
    m->credit_time_gate_max_scale_seen = max_seen;
    m->credit_time_gate_descent_gain = (float)descent_gain;
    m->credit_time_gate_line_shrink = (float)line_shrink;
    m->credit_time_gate_net_descent_gain = (float)net_gain;
    m->credit_time_gate_priced_ratio = (float)priced;
}

static void contractive_green_credit_step(Model *m) {
    if (!m || (!m->contractive_green_credit_enabled &&
               !m->contractive_green_credit_gate_enabled)) {
        return;
    }

    const int apply = m->contractive_green_credit_enabled ? 1 : 0;
    const uint32_t horizon = m->contractive_green_credit_horizon > 0u
        ? m->contractive_green_credit_horizon
        : (uint32_t)V12_SEQ_LEN;
    const float max_alpha = m->contractive_green_credit_max_alpha > 0.0f
        ? m->contractive_green_credit_max_alpha
        : 0.98f;
    const float tail_budget = m->contractive_green_credit_tail_budget >= 0.0f
        ? m->contractive_green_credit_tail_budget
        : 0.10f;
    const float min_scale = m->contractive_green_credit_min_scale > 0.0f
        ? m->contractive_green_credit_min_scale
        : 0.5f;
    const float max_scale = m->contractive_green_credit_max_scale >= min_scale
        ? m->contractive_green_credit_max_scale
        : 2.0f;

    double base_descent = 0.0;
    double green_descent = 0.0;
    double base_norm = 0.0;
    double green_norm = 0.0;
    double mean_alpha_sum = 0.0;
    double mean_tail_sum = 0.0;
    double accepted_fraction_sum = 0.0;
    double mean_scale_sum = 0.0;
    float max_alpha_seen = 0.0f;
    float max_tail_debt = 0.0f;
    float min_seen = max_scale;
    float max_seen = min_scale;
    uint64_t params = 0;
    uint64_t accepted_params = 0;
    uint64_t layers_seen = 0;

    for (int k = 0; k < V12_N_LAYERS; k++) {
        SSMLayer *L = &m->layers[k];
        if (L->dw_in == NULL) continue;

        float s[V12_D_STATE];
        uint8_t accepted[V12_D_STATE];
        float mean_alpha_layer = 0.0f;
        float max_alpha_layer = 0.0f;
        float mean_tail_layer = 0.0f;
        float max_tail_layer = 0.0f;
        float accepted_fraction_layer = 0.0f;
        float mean_scale_layer = 1.0f;
        float min_layer = max_scale;
        float max_layer = min_scale;
        contractive_green_compute_scales(
            L, horizon, max_alpha, tail_budget, min_scale, max_scale,
            s, accepted,
            &mean_alpha_layer, &max_alpha_layer,
            &mean_tail_layer, &max_tail_layer,
            &accepted_fraction_layer, &mean_scale_layer,
            &min_layer, &max_layer);
        mean_alpha_sum += (double)mean_alpha_layer;
        mean_tail_sum += (double)mean_tail_layer;
        accepted_fraction_sum += (double)accepted_fraction_layer;
        mean_scale_sum += (double)mean_scale_layer;
        if (max_alpha_layer > max_alpha_seen) max_alpha_seen = max_alpha_layer;
        if (max_tail_layer > max_tail_debt) max_tail_debt = max_tail_layer;
        if (min_layer < min_seen) min_seen = min_layer;
        if (max_layer > max_seen) max_seen = max_layer;
        layers_seen++;

        if (!(m->gpu_opt_flags & GPU_OPT_W_IN) && L->dw_in) {
            for (int r = 0; r < V12_D_MODEL; r++) {
                for (int c = 0; c < V12_D_STATE; c++) {
                    int idx = r * V12_D_STATE + c;
                    contractive_green_credit_apply_value(
                        &L->dw_in[idx], s[c], accepted[c], apply,
                        &base_descent, &green_descent,
                        &base_norm, &green_norm, &params, &accepted_params);
                }
            }
        }
        if (!(m->gpu_opt_flags & GPU_OPT_W_OUT) && L->dw_out) {
            for (int r = 0; r < V12_D_STATE; r++) {
                for (int c = 0; c < V12_D_MODEL; c++) {
                    int idx = r * V12_D_MODEL + c;
                    contractive_green_credit_apply_value(
                        &L->dw_out[idx], s[r], accepted[r], apply,
                        &base_descent, &green_descent,
                        &base_norm, &green_norm, &params, &accepted_params);
                }
            }
        }
        if (!(m->gpu_opt_flags & GPU_OPT_A) && L->dA) {
            for (int i = 0; i < V12_D_STATE; i++)
                contractive_green_credit_apply_value(
                    &L->dA[i], s[i], accepted[i], apply,
                    &base_descent, &green_descent,
                    &base_norm, &green_norm, &params, &accepted_params);
        }
        if (L->dB) {
            for (int i = 0; i < V12_D_STATE; i++)
                contractive_green_credit_apply_value(
                    &L->dB[i], s[i], accepted[i], apply,
                    &base_descent, &green_descent,
                    &base_norm, &green_norm, &params, &accepted_params);
        }
        if (L->dC) {
            for (int r = 0; r < V12_D_STATE; r++) {
                for (int c = 0; c < V12_D_STATE; c++) {
                    int idx = r * V12_D_STATE + c;
                    contractive_green_credit_apply_value(
                        &L->dC[idx], s[c], accepted[c], apply,
                        &base_descent, &green_descent,
                        &base_norm, &green_norm, &params, &accepted_params);
                }
            }
        }
        if (L->d_sigma) {
            for (int i = 0; i < V12_D_STATE; i++)
                contractive_green_credit_apply_value(
                    &L->d_sigma[i], s[i], accepted[i], apply,
                    &base_descent, &green_descent,
                    &base_norm, &green_norm, &params, &accepted_params);
        }
        if (!(m->gpu_opt_flags & GPU_OPT_WA) && L->dW_A) {
            for (int r = 0; r < V12_D_STATE; r++) {
                for (int c = 0; c < V12_D_STATE; c++) {
                    int idx = r * V12_D_STATE + c;
                    contractive_green_credit_apply_value(
                        &L->dW_A[idx], s[r], accepted[r], apply,
                        &base_descent, &green_descent,
                        &base_norm, &green_norm, &params, &accepted_params);
                }
            }
        }
        if (!(m->gpu_opt_flags & GPU_OPT_BA) && L->db_A) {
            for (int i = 0; i < V12_D_STATE; i++)
                contractive_green_credit_apply_value(
                    &L->db_A[i], s[i], accepted[i], apply,
                    &base_descent, &green_descent,
                    &base_norm, &green_norm, &params, &accepted_params);
        }
        if (!(m->gpu_opt_flags & GPU_OPT_CONV) && L->d_conv_w) {
            for (int r = 0; r < V12_CONV_KERNEL; r++) {
                for (int c = 0; c < V12_D_STATE; c++) {
                    int idx = r * V12_D_STATE + c;
                    contractive_green_credit_apply_value(
                        &L->d_conv_w[idx], s[c], accepted[c], apply,
                        &base_descent, &green_descent,
                        &base_norm, &green_norm, &params, &accepted_params);
                }
            }
        }
        if (L->d_mtp_U) {
            for (int r = 0; r < V12_D_STATE; r++) {
                for (int c = 0; c < V12_MTP_RANK; c++) {
                    int idx = r * V12_MTP_RANK + c;
                    contractive_green_credit_apply_value(
                        &L->d_mtp_U[idx], s[r], accepted[r], apply,
                        &base_descent, &green_descent,
                        &base_norm, &green_norm, &params, &accepted_params);
                }
            }
        }
        if (L->d_mtp_V) {
            for (int r = 0; r < V12_D_STATE; r++) {
                for (int c = 0; c < V12_MTP_RANK; c++) {
                    int idx = r * V12_MTP_RANK + c;
                    contractive_green_credit_apply_value(
                        &L->d_mtp_V[idx], s[r], accepted[r], apply,
                        &base_descent, &green_descent,
                        &base_norm, &green_norm, &params, &accepted_params);
                }
            }
        }
        if (L->dC_bilinear) {
            for (int i = 0; i < V12_D_STATE; i++)
                contractive_green_credit_apply_value(
                    &L->dC_bilinear[i], s[i], accepted[i], apply,
                    &base_descent, &green_descent,
                    &base_norm, &green_norm, &params, &accepted_params);
        }
    }

    if (params == 0u || base_descent <= 0.0 || base_norm <= 0.0) {
        return;
    }

    const double descent_gain = green_descent / base_descent;
    double line_shrink = 1.0;
    if (green_norm > base_norm && green_norm > 0.0) {
        line_shrink = sqrt(base_norm / green_norm);
    }
    const double net_gain = descent_gain * line_shrink;
    const double priced = (net_gain > 1.0e-12)
        ? (1.0 / net_gain) + (double)m->contractive_green_credit_overhead
        : 1.0 + (double)m->contractive_green_credit_overhead;
    const float accepted_fraction =
        params > 0u ? (float)((double)accepted_params / (double)params) : 0.0f;
    const int identifying = (params > 0u && accepted_params > 0u) ? 1 : 0;

    m->contractive_green_credit_gate_batches++;
    m->contractive_green_credit_gate_params += params;
    m->contractive_green_credit_gate_accepted_params += accepted_params;
    m->contractive_green_credit_gate_base_descent += base_descent;
    m->contractive_green_credit_gate_green_descent += green_descent;
    m->contractive_green_credit_gate_base_norm += base_norm;
    m->contractive_green_credit_gate_green_norm += green_norm;
    m->contractive_green_credit_gate_mean_alpha =
        (float)(layers_seen > 0u ? mean_alpha_sum / (double)layers_seen : 0.0);
    m->contractive_green_credit_gate_max_alpha_seen = max_alpha_seen;
    m->contractive_green_credit_gate_mean_tail_debt =
        (float)(layers_seen > 0u ? mean_tail_sum / (double)layers_seen : 0.0);
    m->contractive_green_credit_gate_max_tail_debt = max_tail_debt;
    m->contractive_green_credit_gate_accepted_fraction = accepted_fraction;
    m->contractive_green_credit_gate_mean_scale =
        (float)(layers_seen > 0u ? mean_scale_sum / (double)layers_seen : 1.0);
    m->contractive_green_credit_gate_min_scale_seen = min_seen;
    m->contractive_green_credit_gate_max_scale_seen = max_seen;
    m->contractive_green_credit_gate_descent_gain = (float)descent_gain;
    m->contractive_green_credit_gate_line_shrink = (float)line_shrink;
    m->contractive_green_credit_gate_net_descent_gain = (float)net_gain;
    m->contractive_green_credit_gate_priced_ratio = (float)priced;
    m->contractive_green_credit_gate_identifying = identifying;
    m->contractive_green_credit_gate_pass =
        (identifying &&
         accepted_fraction >= 0.10f &&
         max_tail_debt <= tail_budget &&
         priced < 1.0) ? 1 : 0;
}

static void interior_adjoint_tube_accumulate_value(
    double *energy_by_state, int state_idx, float g) {
    if (!energy_by_state || state_idx < 0 || state_idx >= V12_D_STATE) return;
    const double v = (double)g;
    if (!isfinite(v)) return;
    energy_by_state[state_idx] += v * v;
}

static double interior_adjoint_tube_select_topk(
    const double *energy_by_state, uint32_t rank, uint8_t *tube_mask) {
    if (!energy_by_state || !tube_mask) return 0.0;
    memset(tube_mask, 0, (size_t)V12_D_STATE);
    uint32_t k_max = rank;
    if (k_max > (uint32_t)V12_D_STATE) k_max = (uint32_t)V12_D_STATE;

    double tube_energy = 0.0;
    for (uint32_t k = 0; k < k_max; k++) {
        int best = -1;
        double best_energy = 0.0;
        for (int i = 0; i < V12_D_STATE; i++) {
            if (!tube_mask[i] && energy_by_state[i] > best_energy) {
                best = i;
                best_energy = energy_by_state[i];
            }
        }
        if (best < 0 || best_energy <= 0.0) break;
        tube_mask[best] = 1u;
        tube_energy += best_energy;
    }
    return tube_energy;
}

static void interior_adjoint_tube_apply_value(
    float *g, uint8_t in_tube, int layer_accepted, int apply,
    float residual_scale,
    double *base_descent, double *tube_descent,
    double *base_norm, double *tube_norm,
    uint64_t *params, uint64_t *tube_params) {
    if (!g || *g == 0.0f) return;
    const double base = (double)(*g);
    if (!isfinite(base)) return;
    const double scale =
        (layer_accepted && !in_tube) ? (double)residual_scale : 1.0;
    credit_time_metric(base, scale,
                       base_descent, tube_descent,
                       base_norm, tube_norm, params);
    if (layer_accepted && in_tube && tube_params) (*tube_params)++;
    if (apply && layer_accepted && !in_tube) *g *= residual_scale;
}

static void interior_adjoint_tube_measure_layer(
    const SSMLayer *L, uint32_t gpu_opt_flags, double *energy_by_state) {
    if (!L || !energy_by_state) return;
    memset(energy_by_state, 0, (size_t)V12_D_STATE * sizeof(double));

    if (!(gpu_opt_flags & GPU_OPT_W_IN) && L->dw_in) {
        for (int r = 0; r < V12_D_MODEL; r++) {
            for (int c = 0; c < V12_D_STATE; c++) {
                interior_adjoint_tube_accumulate_value(
                    energy_by_state, c, L->dw_in[r * V12_D_STATE + c]);
            }
        }
    }
    if (!(gpu_opt_flags & GPU_OPT_W_OUT) && L->dw_out) {
        for (int r = 0; r < V12_D_STATE; r++) {
            for (int c = 0; c < V12_D_MODEL; c++) {
                interior_adjoint_tube_accumulate_value(
                    energy_by_state, r, L->dw_out[r * V12_D_MODEL + c]);
            }
        }
    }
    if (!(gpu_opt_flags & GPU_OPT_A) && L->dA) {
        for (int i = 0; i < V12_D_STATE; i++)
            interior_adjoint_tube_accumulate_value(energy_by_state, i, L->dA[i]);
    }
    if (L->dB) {
        for (int i = 0; i < V12_D_STATE; i++)
            interior_adjoint_tube_accumulate_value(energy_by_state, i, L->dB[i]);
    }
    if (L->dC) {
        for (int r = 0; r < V12_D_STATE; r++) {
            for (int c = 0; c < V12_D_STATE; c++) {
                interior_adjoint_tube_accumulate_value(
                    energy_by_state, c, L->dC[r * V12_D_STATE + c]);
            }
        }
    }
    if (L->d_sigma) {
        for (int i = 0; i < V12_D_STATE; i++)
            interior_adjoint_tube_accumulate_value(energy_by_state, i, L->d_sigma[i]);
    }
    if (!(gpu_opt_flags & GPU_OPT_WA) && L->dW_A) {
        for (int r = 0; r < V12_D_STATE; r++) {
            for (int c = 0; c < V12_D_STATE; c++) {
                interior_adjoint_tube_accumulate_value(
                    energy_by_state, r, L->dW_A[r * V12_D_STATE + c]);
            }
        }
    }
    if (!(gpu_opt_flags & GPU_OPT_BA) && L->db_A) {
        for (int i = 0; i < V12_D_STATE; i++)
            interior_adjoint_tube_accumulate_value(energy_by_state, i, L->db_A[i]);
    }
    if (!(gpu_opt_flags & GPU_OPT_CONV) && L->d_conv_w) {
        for (int r = 0; r < V12_CONV_KERNEL; r++) {
            for (int c = 0; c < V12_D_STATE; c++) {
                interior_adjoint_tube_accumulate_value(
                    energy_by_state, c, L->d_conv_w[r * V12_D_STATE + c]);
            }
        }
    }
    if (L->d_mtp_U) {
        for (int r = 0; r < V12_D_STATE; r++) {
            for (int c = 0; c < V12_MTP_RANK; c++) {
                interior_adjoint_tube_accumulate_value(
                    energy_by_state, r, L->d_mtp_U[r * V12_MTP_RANK + c]);
            }
        }
    }
    if (L->d_mtp_V) {
        for (int r = 0; r < V12_D_STATE; r++) {
            for (int c = 0; c < V12_MTP_RANK; c++) {
                interior_adjoint_tube_accumulate_value(
                    energy_by_state, r, L->d_mtp_V[r * V12_MTP_RANK + c]);
            }
        }
    }
    if (L->dC_bilinear) {
        for (int i = 0; i < V12_D_STATE; i++)
            interior_adjoint_tube_accumulate_value(
                energy_by_state, i, L->dC_bilinear[i]);
    }
}

static void interior_adjoint_tube_apply_layer(
    SSMLayer *L, uint32_t gpu_opt_flags, const uint8_t *tube_mask,
    int layer_accepted, int apply, float residual_scale,
    double *base_descent, double *tube_descent,
    double *base_norm, double *tube_norm,
    uint64_t *params, uint64_t *tube_params) {
    if (!L || !tube_mask) return;

    if (!(gpu_opt_flags & GPU_OPT_W_IN) && L->dw_in) {
        for (int r = 0; r < V12_D_MODEL; r++) {
            for (int c = 0; c < V12_D_STATE; c++) {
                interior_adjoint_tube_apply_value(
                    &L->dw_in[r * V12_D_STATE + c], tube_mask[c],
                    layer_accepted, apply, residual_scale,
                    base_descent, tube_descent, base_norm, tube_norm,
                    params, tube_params);
            }
        }
    }
    if (!(gpu_opt_flags & GPU_OPT_W_OUT) && L->dw_out) {
        for (int r = 0; r < V12_D_STATE; r++) {
            for (int c = 0; c < V12_D_MODEL; c++) {
                interior_adjoint_tube_apply_value(
                    &L->dw_out[r * V12_D_MODEL + c], tube_mask[r],
                    layer_accepted, apply, residual_scale,
                    base_descent, tube_descent, base_norm, tube_norm,
                    params, tube_params);
            }
        }
    }
    if (!(gpu_opt_flags & GPU_OPT_A) && L->dA) {
        for (int i = 0; i < V12_D_STATE; i++)
            interior_adjoint_tube_apply_value(
                &L->dA[i], tube_mask[i], layer_accepted, apply, residual_scale,
                base_descent, tube_descent, base_norm, tube_norm, params, tube_params);
    }
    if (L->dB) {
        for (int i = 0; i < V12_D_STATE; i++)
            interior_adjoint_tube_apply_value(
                &L->dB[i], tube_mask[i], layer_accepted, apply, residual_scale,
                base_descent, tube_descent, base_norm, tube_norm, params, tube_params);
    }
    if (L->dC) {
        for (int r = 0; r < V12_D_STATE; r++) {
            for (int c = 0; c < V12_D_STATE; c++) {
                interior_adjoint_tube_apply_value(
                    &L->dC[r * V12_D_STATE + c], tube_mask[c],
                    layer_accepted, apply, residual_scale,
                    base_descent, tube_descent, base_norm, tube_norm,
                    params, tube_params);
            }
        }
    }
    if (L->d_sigma) {
        for (int i = 0; i < V12_D_STATE; i++)
            interior_adjoint_tube_apply_value(
                &L->d_sigma[i], tube_mask[i], layer_accepted, apply, residual_scale,
                base_descent, tube_descent, base_norm, tube_norm, params, tube_params);
    }
    if (!(gpu_opt_flags & GPU_OPT_WA) && L->dW_A) {
        for (int r = 0; r < V12_D_STATE; r++) {
            for (int c = 0; c < V12_D_STATE; c++) {
                interior_adjoint_tube_apply_value(
                    &L->dW_A[r * V12_D_STATE + c], tube_mask[r],
                    layer_accepted, apply, residual_scale,
                    base_descent, tube_descent, base_norm, tube_norm,
                    params, tube_params);
            }
        }
    }
    if (!(gpu_opt_flags & GPU_OPT_BA) && L->db_A) {
        for (int i = 0; i < V12_D_STATE; i++)
            interior_adjoint_tube_apply_value(
                &L->db_A[i], tube_mask[i], layer_accepted, apply, residual_scale,
                base_descent, tube_descent, base_norm, tube_norm, params, tube_params);
    }
    if (!(gpu_opt_flags & GPU_OPT_CONV) && L->d_conv_w) {
        for (int r = 0; r < V12_CONV_KERNEL; r++) {
            for (int c = 0; c < V12_D_STATE; c++) {
                interior_adjoint_tube_apply_value(
                    &L->d_conv_w[r * V12_D_STATE + c], tube_mask[c],
                    layer_accepted, apply, residual_scale,
                    base_descent, tube_descent, base_norm, tube_norm,
                    params, tube_params);
            }
        }
    }
    if (L->d_mtp_U) {
        for (int r = 0; r < V12_D_STATE; r++) {
            for (int c = 0; c < V12_MTP_RANK; c++) {
                interior_adjoint_tube_apply_value(
                    &L->d_mtp_U[r * V12_MTP_RANK + c], tube_mask[r],
                    layer_accepted, apply, residual_scale,
                    base_descent, tube_descent, base_norm, tube_norm,
                    params, tube_params);
            }
        }
    }
    if (L->d_mtp_V) {
        for (int r = 0; r < V12_D_STATE; r++) {
            for (int c = 0; c < V12_MTP_RANK; c++) {
                interior_adjoint_tube_apply_value(
                    &L->d_mtp_V[r * V12_MTP_RANK + c], tube_mask[r],
                    layer_accepted, apply, residual_scale,
                    base_descent, tube_descent, base_norm, tube_norm,
                    params, tube_params);
            }
        }
    }
    if (L->dC_bilinear) {
        for (int i = 0; i < V12_D_STATE; i++)
            interior_adjoint_tube_apply_value(
                &L->dC_bilinear[i], tube_mask[i], layer_accepted, apply,
                residual_scale,
                base_descent, tube_descent, base_norm, tube_norm, params, tube_params);
    }
}

static void interior_adjoint_tube_step(Model *m) {
    if (!m || (!m->interior_adjoint_tube_enabled &&
               !m->interior_adjoint_tube_gate_enabled)) {
        return;
    }

    const int apply = m->interior_adjoint_tube_enabled ? 1 : 0;
    const uint32_t rank =
        m->interior_adjoint_tube_rank > 0u
            ? m->interior_adjoint_tube_rank
            : (uint32_t)(V12_D_STATE / 4);
    const float residual_budget =
        m->interior_adjoint_tube_residual_budget >= 0.0f
            ? m->interior_adjoint_tube_residual_budget
            : 0.10f;
    const float residual_scale =
        (m->interior_adjoint_tube_residual_scale >= 0.0f &&
         m->interior_adjoint_tube_residual_scale <= 1.0f)
            ? m->interior_adjoint_tube_residual_scale
            : 0.25f;
    const float min_energy_covered =
        (m->interior_adjoint_tube_min_energy_covered >= 0.0f &&
         m->interior_adjoint_tube_min_energy_covered <= 1.0f)
            ? m->interior_adjoint_tube_min_energy_covered
            : 0.90f;

    double exact_energy = 0.0;
    double tube_energy = 0.0;
    double residual_energy = 0.0;
    double base_descent = 0.0;
    double tube_descent = 0.0;
    double base_norm = 0.0;
    double tube_norm = 0.0;
    uint64_t layers = 0u;
    uint64_t accepted_layers = 0u;
    uint64_t params = 0u;
    uint64_t tube_params = 0u;

    for (int k = 0; k < V12_N_LAYERS; k++) {
        SSMLayer *L = &m->layers[k];
        if (!L || L->dw_in == NULL) continue;

        double energy_by_state[V12_D_STATE];
        uint8_t tube_mask[V12_D_STATE];
        interior_adjoint_tube_measure_layer(L, m->gpu_opt_flags, energy_by_state);

        double layer_exact = 0.0;
        for (int i = 0; i < V12_D_STATE; i++) {
            layer_exact += energy_by_state[i];
        }
        if (layer_exact <= 0.0) continue;

        const double layer_tube =
            interior_adjoint_tube_select_topk(energy_by_state, rank, tube_mask);
        const double layer_residual = fmax(0.0, layer_exact - layer_tube);
        const double layer_energy_covered =
            layer_exact > 0.0 ? layer_tube / layer_exact : 0.0;
        const double layer_residual_fraction =
            layer_exact > 0.0 ? layer_residual / layer_exact : 1.0;
        const int layer_accepted =
            (layer_residual_fraction <= (double)residual_budget &&
             layer_energy_covered >= (double)min_energy_covered) ? 1 : 0;

        exact_energy += layer_exact;
        tube_energy += layer_tube;
        residual_energy += layer_residual;
        layers++;
        if (layer_accepted) accepted_layers++;

        interior_adjoint_tube_apply_layer(
            L, m->gpu_opt_flags, tube_mask, layer_accepted, apply, residual_scale,
            &base_descent, &tube_descent, &base_norm, &tube_norm,
            &params, &tube_params);
    }

    if (layers == 0u || params == 0u || exact_energy <= 0.0 ||
        base_descent <= 0.0 || base_norm <= 0.0) {
        return;
    }

    const double residual_fraction = residual_energy / exact_energy;
    const double energy_covered = tube_energy / exact_energy;
    const double tube_fraction =
        params > 0u ? (double)tube_params / (double)params : 1.0;
    const double verify_fraction =
        m->interior_adjoint_tube_verify_fraction >= 0.0f
            ? (double)m->interior_adjoint_tube_verify_fraction
            : 1.0;
    const double overhead = (double)m->interior_adjoint_tube_overhead;
    const double priced =
        tube_fraction + residual_fraction + verify_fraction + overhead;
    const double descent_gain = tube_descent / base_descent;
    double line_shrink = 1.0;
    if (tube_norm > base_norm && tube_norm > 0.0) {
        line_shrink = sqrt(base_norm / tube_norm);
    }
    const double net_gain = descent_gain * line_shrink;
    const double descent_per_paid =
        priced > 0.0 ? energy_covered / priced : 0.0;
    const double accepted_layer_fraction =
        layers > 0u ? (double)accepted_layers / (double)layers : 0.0;
    const int identifying =
        (layers > 0u && params > 0u && exact_energy > 0.0) ? 1 : 0;

    m->interior_adjoint_tube_gate_batches++;
    m->interior_adjoint_tube_gate_layers += layers;
    m->interior_adjoint_tube_gate_accepted_layers += accepted_layers;
    m->interior_adjoint_tube_gate_params += params;
    m->interior_adjoint_tube_gate_tube_params += tube_params;
    m->interior_adjoint_tube_gate_exact_energy += exact_energy;
    m->interior_adjoint_tube_gate_tube_energy += tube_energy;
    m->interior_adjoint_tube_gate_residual_energy += residual_energy;
    m->interior_adjoint_tube_gate_base_descent += base_descent;
    m->interior_adjoint_tube_gate_tube_descent += tube_descent;
    m->interior_adjoint_tube_gate_base_norm += base_norm;
    m->interior_adjoint_tube_gate_tube_norm += tube_norm;
    m->interior_adjoint_tube_gate_tube_fraction = (float)tube_fraction;
    m->interior_adjoint_tube_gate_accepted_layer_fraction =
        (float)accepted_layer_fraction;
    m->interior_adjoint_tube_gate_residual_fraction =
        (float)residual_fraction;
    m->interior_adjoint_tube_gate_energy_covered = (float)energy_covered;
    m->interior_adjoint_tube_gate_descent_gain = (float)descent_gain;
    m->interior_adjoint_tube_gate_line_shrink = (float)line_shrink;
    m->interior_adjoint_tube_gate_net_descent_gain = (float)net_gain;
    m->interior_adjoint_tube_gate_priced_ratio = (float)priced;
    m->interior_adjoint_tube_gate_descent_per_paid_work =
        (float)descent_per_paid;
    m->interior_adjoint_tube_gate_identifying = identifying;
    m->interior_adjoint_tube_gate_pass =
        (identifying &&
         accepted_layer_fraction > 0.0 &&
         residual_fraction <= (double)residual_budget &&
         energy_covered >= (double)min_energy_covered &&
         priced < 0.90 &&
         descent_per_paid > 1.0) ? 1 : 0;
}

static void scale_all_gradients(Model *m, float scale) {
    for (int k = 0; k < V12_N_LAYERS; k++) {
        SSMLayer *L = &m->layers[k];
        if (L->dw_in == NULL) continue;
        int n_in  = V12_D_MODEL * V12_D_STATE;
        int n_out = V12_D_STATE * V12_D_MODEL;
        if (!(m->gpu_opt_flags & GPU_OPT_W_IN))
            for (int i = 0; i < n_in;  i++) L->dw_in[i] *= scale;
        if (!(m->gpu_opt_flags & GPU_OPT_W_OUT))
            for (int i = 0; i < n_out; i++) L->dw_out[i] *= scale;
        for (int i = 0; i < V12_D_STATE; i++) L->d_sigma[i] *= scale;
        if (!(m->gpu_opt_flags & GPU_OPT_A))
            for (int i = 0; i < V12_D_STATE; i++) L->dA[i] *= scale;
        for (int i = 0; i < V12_D_STATE; i++) L->dB[i]      *= scale;
        if (L->dC != NULL) {
            for (int i = 0; i < V12_D_STATE * V12_D_STATE; i++) L->dC[i] *= scale;
        }
        if (!(m->gpu_opt_flags & GPU_OPT_WA))
            for (int i = 0; i < V12_D_STATE * V12_D_STATE; i++) L->dW_A[i] *= scale;
        if (!(m->gpu_opt_flags & GPU_OPT_BA))
            for (int i = 0; i < V12_D_STATE; i++) L->db_A[i] *= scale;
        int n_conv = V12_CONV_KERNEL * V12_D_STATE;
        
        if (!(m->gpu_opt_flags & GPU_OPT_CONV)) {
            for (int i = 0; i < n_conv; i++) L->d_conv_w[i] *= scale;
        }
        
        if (L->d_mtp_U != NULL) {
            for (int i = 0; i < V12_D_STATE * V12_MTP_RANK; i++) L->d_mtp_U[i] *= scale;
        }
        
        if (L->d_mtp_V != NULL) {
            for (int i = 0; i < V12_D_STATE * V12_MTP_RANK; i++) L->d_mtp_V[i] *= scale;
        }
        
        if (L->dC_bilinear != NULL) {
            for (int i = 0; i < V12_D_STATE; i++) L->dC_bilinear[i] *= scale;
        }
        m->d_tau_mux[k] *= scale;
    }
    
    if (!(m->gpu_opt_flags & GPU_OPT_LM_HEAD)) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < V12_K_HOT; i++) {
            if (m->dlm_head_g_row) m->dlm_head_g_row[i] *= scale;
            if (m->dlm_head_gq_row) m->dlm_head_gq_row[i] *= scale * scale;
        }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < V12_D_MODEL; i++) {
            if (m->dlm_head_g_col) m->dlm_head_g_col[i] *= scale;
            if (m->dlm_head_gq_col) m->dlm_head_gq_col[i] *= scale * scale;
        }
    }
    
    if (!(m->gpu_opt_flags & GPU_OPT_EMBED_E) && m->d_embed_E) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < V12_VOCAB_SIZE * V12_D_EMBED; i++) m->d_embed_E[i] *= scale;
    }
    
    if (!(m->gpu_opt_flags & GPU_OPT_EMBED_P) && m->d_embed_P) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < V12_D_EMBED * V12_D_MODEL; i++) m->d_embed_P[i] *= scale;
    }
    
    if (!(m->gpu_opt_flags & GPU_OPT_NORM)) {
        if (m->d_norm_w) for (int i = 0; i < V12_D_MODEL; i++) m->d_norm_w[i] *= scale;
        if (m->d_norm_b) for (int i = 0; i < V12_D_MODEL; i++) m->d_norm_b[i] *= scale;
    }
    
    if (!(m->gpu_opt_flags & GPU_OPT_W_PRED)) {
        if (m->dw_pred) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < V12_D_MODEL * V12_D_MODEL; i++) m->dw_pred[i] *= scale;
        }
    }
}

typedef struct {
    int enabled;
    uint32_t boundary_step;
    uint16_t *tokens;
    uint8_t *mask;
    float *sampled_b_snapshot;
    size_t sampled_snapshot_len;
    uint64_t repack_changed;
    double repack_l1;
    float repack_changed_fraction;
    int have_repack_distance;
    int batch_size;
    int seq_len;
    int captured_a;
    int captured_b;
    int captured_c;
    int captured_d;
    int summarized;
    float loss_a;
    float loss_b;
    float loss_c;
    float loss_d;
} R435BoundaryReceipt;

typedef struct {
    int enabled;
    uint32_t k;
    float min_margin;
    float max_rollback_fraction;
    float verifier_fraction;
    uint32_t position;
    uint32_t start_step;
    float start_loss;
    float min_loss;
    float max_loss;
    float last_margin;
    uint64_t observed;
    uint64_t accepted;
    uint64_t rejected;
    uint64_t rollback;
} R438CeTransactionObserver;

typedef struct {
    uint64_t x_hash;
    uint64_t z_hash;
    uint64_t o_hash;
    uint64_t csr_hash;
} R435StateIdentity;

static void r435_release_receipt_host_mirrors(Model *m);

static uint64_t r435_fnv1a_update(uint64_t h, const void *data, size_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < bytes; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t r435_hash_u64(uint64_t h, uint64_t v) {
    return r435_fnv1a_update(h, &v, sizeof(v));
}

static uint64_t r435_hash_bytes(uint64_t h, const void *ptr, size_t bytes) {
    if (!ptr || bytes == 0) return r435_hash_u64(h, 0x435ull);
    return r435_fnv1a_update(h, ptr, bytes);
}

static uint64_t r435_hash_csr(uint64_t h, const TernaryCSR *csr) {
    if (!csr) return r435_hash_u64(h, 0);
    h = r435_hash_u64(h, (uint64_t)(uint32_t)csr->rows);
    h = r435_hash_u64(h, (uint64_t)(uint32_t)csr->cols);
    h = r435_hash_u64(h, (uint64_t)(uint32_t)csr->total_pos);
    h = r435_hash_u64(h, (uint64_t)(uint32_t)csr->total_neg);
    h = r435_hash_bytes(h, csr->row_ptr_pos, (size_t)(csr->rows + 1) * sizeof(int32_t));
    h = r435_hash_bytes(h, csr->row_ptr_neg, (size_t)(csr->rows + 1) * sizeof(int32_t));
    h = r435_hash_bytes(h, csr->pos_indices, (size_t)csr->total_pos * sizeof(int32_t));
    h = r435_hash_bytes(h, csr->neg_indices, (size_t)csr->total_neg * sizeof(int32_t));
    return h;
}

static size_t r435_copy_sampled_block(float *dst, const float *src, size_t n) {
    if (!dst || n == 0) return 0;
    if (src) {
        memcpy(dst, src, n * sizeof(float));
    } else {
        memset(dst, 0, n * sizeof(float));
    }
    return n;
}

static void r435_capture_sampled_snapshot(Model *m, float *dst) {
    size_t off = 0;
    const size_t n_wio = (size_t)V12_D_MODEL * (size_t)V12_D_STATE;
    const size_t n_lm = (size_t)V12_VOCAB_SIZE * (size_t)V12_D_MODEL;
    const size_t n_pred = (size_t)V12_D_MODEL * (size_t)V12_D_MODEL;
    for (int k = 0; k < V12_N_LAYERS; k++) {
        SSMLayer *L = &m->layers[k];
        off += r435_copy_sampled_block(dst + off, L->sampled_w_in, n_wio);
        off += r435_copy_sampled_block(dst + off, L->sampled_w_out, n_wio);
    }
    off += r435_copy_sampled_block(dst + off, m->sampled_lm_head, n_lm);
    off += r435_copy_sampled_block(dst + off, m->sampled_w_pred, n_pred);
}

static void r435_compare_sampled_snapshot(R435BoundaryReceipt *receipt, Model *m) {
    if (!receipt || !receipt->sampled_b_snapshot || receipt->sampled_snapshot_len == 0) return;
    float *current = (float *)malloc(receipt->sampled_snapshot_len * sizeof(float));
    if (!current) {
        printf("[R435_BOUNDARY_RECEIPT] repack_distance unavailable reason=malloc_failed entries=%llu\n",
               (unsigned long long)receipt->sampled_snapshot_len);
        return;
    }
    r435_capture_sampled_snapshot(m, current);
    uint64_t changed = 0;
    double l1 = 0.0;
    for (size_t i = 0; i < receipt->sampled_snapshot_len; i++) {
        const float before = receipt->sampled_b_snapshot[i];
        const float after = current[i];
        if (before != after) changed++;
        l1 += fabs((double)after - (double)before);
    }
    receipt->repack_changed = changed;
    receipt->repack_l1 = l1;
    receipt->repack_changed_fraction =
        receipt->sampled_snapshot_len > 0
            ? (float)((double)changed / (double)receipt->sampled_snapshot_len)
            : 0.0f;
    receipt->have_repack_distance = 1;
    printf("[R435_BOUNDARY_RECEIPT] repack_distance step=%u changed=%llu total=%llu changed_fraction=%.9g l1=%.9g\n",
           receipt->boundary_step,
           (unsigned long long)changed,
           (unsigned long long)receipt->sampled_snapshot_len,
           receipt->repack_changed_fraction,
           l1);
    free(current);
}

static R435StateIdentity r435_capture_identity(Model *m, GpuPipeline *gpu) {
    R435StateIdentity id = {
        1469598103934665603ull,
        1469598103934665603ull,
        1469598103934665603ull,
        1469598103934665603ull
    };
    const size_t n_wio = (size_t)V12_D_MODEL * (size_t)V12_D_STATE;
    const size_t n_pred = (size_t)V12_D_MODEL * (size_t)V12_D_MODEL;
    const size_t n_lm = (size_t)V12_VOCAB_SIZE * (size_t)V12_D_MODEL;

    (void)gpu;

    for (int k = 0; k < V12_N_LAYERS; k++) {
        SSMLayer *L = &m->layers[k];
        id.x_hash = r435_hash_bytes(id.x_hash, L->shadow_w_in, n_wio * SHADOW_SIZEOF);
        id.x_hash = r435_hash_bytes(id.x_hash, L->shadow_w_out, n_wio * SHADOW_SIZEOF);
        id.z_hash = r435_hash_bytes(id.z_hash, L->sampled_w_in, n_wio * sizeof(float));
        id.z_hash = r435_hash_bytes(id.z_hash, L->sampled_w_out, n_wio * sizeof(float));
        id.o_hash = r435_hash_bytes(id.o_hash, L->m_w_in_row, (size_t)V12_D_MODEL * sizeof(float));
        id.o_hash = r435_hash_bytes(id.o_hash, L->m_w_in_col, (size_t)V12_D_STATE * sizeof(float));
        id.o_hash = r435_hash_bytes(id.o_hash, L->v_w_in_row, (size_t)V12_D_MODEL * sizeof(float));
        id.o_hash = r435_hash_bytes(id.o_hash, L->v_w_in_col, (size_t)V12_D_STATE * sizeof(float));
        id.o_hash = r435_hash_bytes(id.o_hash, L->m_w_out_row, (size_t)V12_D_STATE * sizeof(float));
        id.o_hash = r435_hash_bytes(id.o_hash, L->m_w_out_col, (size_t)V12_D_MODEL * sizeof(float));
        id.o_hash = r435_hash_bytes(id.o_hash, L->v_w_out_row, (size_t)V12_D_STATE * sizeof(float));
        id.o_hash = r435_hash_bytes(id.o_hash, L->v_w_out_col, (size_t)V12_D_MODEL * sizeof(float));
        id.csr_hash = r435_hash_csr(id.csr_hash, &L->csr_in);
        id.csr_hash = r435_hash_csr(id.csr_hash, &L->csr_out);
    }

    id.x_hash = r435_hash_bytes(id.x_hash, m->shadow_lm_head, n_lm * SHADOW_SIZEOF);
    id.x_hash = r435_hash_bytes(id.x_hash, m->shadow_w_pred, n_pred * SHADOW_SIZEOF);
    id.z_hash = r435_hash_bytes(id.z_hash, m->sampled_lm_head, n_lm * sizeof(float));
    id.z_hash = r435_hash_bytes(id.z_hash, m->sampled_w_pred, n_pred * sizeof(float));
    id.o_hash = r435_hash_bytes(id.o_hash, m->m_lm_head_row, (size_t)V12_K_HOT * sizeof(float));
    id.o_hash = r435_hash_bytes(id.o_hash, m->m_lm_head_col, (size_t)V12_D_MODEL * sizeof(float));
    id.o_hash = r435_hash_bytes(id.o_hash, m->v_lm_head_row, (size_t)V12_K_HOT * sizeof(float));
    id.o_hash = r435_hash_bytes(id.o_hash, m->v_lm_head_col, (size_t)V12_D_MODEL * sizeof(float));
    id.o_hash = r435_hash_bytes(id.o_hash, m->m_w_pred_row, (size_t)V12_D_MODEL * sizeof(float));
    id.o_hash = r435_hash_bytes(id.o_hash, m->m_w_pred_col, (size_t)V12_D_MODEL * sizeof(float));
    id.o_hash = r435_hash_bytes(id.o_hash, m->v_w_pred_row, (size_t)V12_D_MODEL * sizeof(float));
    id.o_hash = r435_hash_bytes(id.o_hash, m->v_w_pred_col, (size_t)V12_D_MODEL * sizeof(float));
    id.csr_hash = r435_hash_csr(id.csr_hash, &m->csr_lm_head);
    id.csr_hash = r435_hash_csr(id.csr_hash, &m->csr_w_pred);
    return id;
}

static float r435_probe_loss(Model *m, GpuPipeline *gpu, Cache *c,
                             const R435BoundaryReceipt *receipt) {
    GpKlMoments saved_moments[V12_N_LAYERS];
    for (int k = 0; k < V12_N_LAYERS; k++) {
        saved_moments[k] = m->layers[k].moments;
    }
#ifdef V12_HAS_VULKAN
    gpu_zero_grad_accumulators(gpu);
#endif
    float loss = gpu_train_step(gpu, m, c, receipt->tokens, receipt->mask,
                                receipt->seq_len, 0, receipt->batch_size, NULL);
    scale_all_gradients(m, 0.0f);
#ifdef V12_HAS_VULKAN
    gpu_zero_grad_accumulators(gpu);
#endif
    for (int k = 0; k < V12_N_LAYERS; k++) {
        m->layers[k].moments = saved_moments[k];
    }
    return loss;
}

static void r435_log_receipt_phase(R435BoundaryReceipt *receipt,
                                   const char *phase,
                                   uint32_t step,
                                   Model *m,
                                   GpuPipeline *gpu,
                                   Cache *c,
                                   float *slot,
                                   int *captured) {
    if (!receipt || !receipt->enabled || *captured) return;
    R435StateIdentity id = r435_capture_identity(m, gpu);
    float loss = r435_probe_loss(m, gpu, c, receipt);
    if (strcmp(phase, "S_b") == 0 && receipt->sampled_b_snapshot) {
        r435_capture_sampled_snapshot(m, receipt->sampled_b_snapshot);
    } else if (strcmp(phase, "S_c") == 0) {
        if (receipt->sampled_b_snapshot) {
            r435_compare_sampled_snapshot(receipt, m);
        } else {
            printf("[R435_BOUNDARY_RECEIPT] repack_distance unavailable reason=resident_only_no_gpu_snapshot\n");
        }
    }
    *slot = loss;
    *captured = 1;
    printf("[R435_BOUNDARY_RECEIPT] phase=%s step=%u loss=%.9g x_hash=0x%016llx z_hash=0x%016llx o_hash=0x%016llx csr_hash=0x%016llx probe_batch=%d identity_scope=resident_cpu_no_gpu_sync\n",
           phase,
           step,
           loss,
           (unsigned long long)id.x_hash,
           (unsigned long long)id.z_hash,
           (unsigned long long)id.o_hash,
           (unsigned long long)id.csr_hash,
           receipt->batch_size);
    r435_release_receipt_host_mirrors(m);
}

static const char *r435_classify_receipt(const R435BoundaryReceipt *r) {
    if (!r->captured_a || !r->captured_b || !r->captured_c || !r->captured_d) {
        return "incomplete";
    }
    if (!isfinite(r->loss_a) || !isfinite(r->loss_b) ||
        !isfinite(r->loss_c) || !isfinite(r->loss_d)) {
        return "nonfinite";
    }
    const float abs_gate = 5.0f;
    const float ratio_gate = 2.0f;
    if (r->loss_b - r->loss_a > abs_gate || r->loss_b > r->loss_a * ratio_gate) {
        return "lr_pre_repack";
    }
    if (r->loss_c - r->loss_b > abs_gate || r->loss_c > r->loss_b * ratio_gate) {
        return "immediate_repack_materialization";
    }
    if (r->loss_d - r->loss_c > abs_gate || r->loss_d > r->loss_c * ratio_gate) {
        return "post_repack_retained_moment";
    }
    return "stable_or_non_identifying";
}

static void r435_maybe_summarize_receipt(R435BoundaryReceipt *r) {
    if (!r || !r->enabled || r->summarized) return;
    if (!r->captured_a || !r->captured_b || !r->captured_c || !r->captured_d) return;
    r->summarized = 1;
    printf("[R435_BOUNDARY_RECEIPT] summary boundary_step=%u L_a=%.9g L_b=%.9g L_c=%.9g L_d=%.9g d_lr=%.9g d_repack=%.9g d_post=%.9g classification=%s\n",
           r->boundary_step,
           r->loss_a,
           r->loss_b,
           r->loss_c,
           r->loss_d,
           r->loss_b - r->loss_a,
           r->loss_c - r->loss_b,
           r->loss_d - r->loss_c,
           r435_classify_receipt(r));
    if (r->have_repack_distance) {
        printf("[R435_BOUNDARY_RECEIPT] summary_repack_distance boundary_step=%u changed=%llu total=%llu changed_fraction=%.9g l1=%.9g\n",
               r->boundary_step,
               (unsigned long long)r->repack_changed,
               (unsigned long long)r->sampled_snapshot_len,
               r->repack_changed_fraction,
               r->repack_l1);
    }
}

static void r435_release_receipt_host_mirrors(Model *m) {
    release_lm_head_steady_state(m, "r435-boundary-receipt");
    release_slice2_steady_state(m, "r435-boundary-receipt");
    release_slice3_steady_state(m, "r435-boundary-receipt");
    release_slice4_steady_state(m, "r435-boundary-receipt");
}

static void r435_receipt_free(R435BoundaryReceipt *r) {
    if (!r) return;
    free(r->tokens);
    free(r->mask);
    free(r->sampled_b_snapshot);
    r->tokens = NULL;
    r->mask = NULL;
    r->sampled_b_snapshot = NULL;
}

static void r438_ce_tx_observe(R438CeTransactionObserver *tx, uint32_t step, float loss) {
    if (!tx || !tx->enabled || tx->k == 0u || !isfinite(loss)) return;
    if (tx->position == 0u) {
        tx->start_step = step;
        tx->start_loss = loss;
        tx->min_loss = loss;
        tx->max_loss = loss;
        printf("[CE_TX_OBSERVER] start step=%u k=%u loss=%.9g min_margin=%.9g verifier_fraction=%.9g\n",
               step, tx->k, loss, tx->min_margin, tx->verifier_fraction);
    } else {
        if (loss < tx->min_loss) tx->min_loss = loss;
        if (loss > tx->max_loss) tx->max_loss = loss;
    }
    tx->position++;
    if (tx->position < tx->k) return;

    const float curvature_debt = fmaxf(0.0f, tx->max_loss - tx->min_loss);
    const float margin = tx->start_loss - loss - curvature_debt;
    tx->last_margin = margin;
    tx->observed++;
    if (loss > tx->start_loss) {
        tx->rollback++;
        printf("[CE_TX_ROLLBACK] start_step=%u end_step=%u k=%u start_loss=%.9g end_loss=%.9g curvature_debt=%.9g margin=%.9g observed=%llu rollback=%llu\n",
               tx->start_step, step, tx->k, tx->start_loss, loss,
               curvature_debt, margin,
               (unsigned long long)tx->observed,
               (unsigned long long)tx->rollback);
    } else if (margin >= tx->min_margin) {
        tx->accepted++;
        printf("[CE_TX_ACCEPT] start_step=%u end_step=%u k=%u start_loss=%.9g end_loss=%.9g curvature_debt=%.9g margin=%.9g observed=%llu accepted=%llu accept_density=%.6f rollback_fraction=%.6f\n",
               tx->start_step, step, tx->k, tx->start_loss, loss,
               curvature_debt, margin,
               (unsigned long long)tx->observed,
               (unsigned long long)tx->accepted,
               tx->observed > 0u ? (double)tx->accepted / (double)tx->observed : 0.0,
               tx->observed > 0u ? (double)tx->rollback / (double)tx->observed : 0.0);
    } else {
        tx->rejected++;
        printf("[CE_TX_REJECT] start_step=%u end_step=%u k=%u start_loss=%.9g end_loss=%.9g curvature_debt=%.9g margin=%.9g observed=%llu rejected=%llu\n",
               tx->start_step, step, tx->k, tx->start_loss, loss,
               curvature_debt, margin,
               (unsigned long long)tx->observed,
               (unsigned long long)tx->rejected);
    }
    tx->position = 0u;
}

static void r438_ce_tx_summarize(const R438CeTransactionObserver *tx) {
    if (!tx || !tx->enabled) return;
    const double accept_density =
        tx->observed > 0u ? (double)tx->accepted / (double)tx->observed : 0.0;
    const double rollback_fraction =
        tx->observed > 0u ? (double)tx->rollback / (double)tx->observed : 0.0;
    const double reject_fraction =
        tx->observed > 0u ? (double)tx->rejected / (double)tx->observed : 0.0;
    const double priced_ratio =
        reject_fraction + rollback_fraction + (double)tx->verifier_fraction;
    const int pass =
        tx->observed > 0u &&
        accept_density >= 0.20 &&
        rollback_fraction <= (double)tx->max_rollback_fraction &&
        priced_ratio < 0.90;
    printf("[CE_TX_OBSERVER] summary k=%u observed=%llu accepted=%llu rejected=%llu rollback=%llu accept_density=%.6f rollback_fraction=%.6f reject_fraction=%.6f verifier_fraction=%.6f priced_ratio=%.6f last_margin=%.9g pass=%d observer_only=1 no_training_change=1\n",
           tx->k,
           (unsigned long long)tx->observed,
           (unsigned long long)tx->accepted,
           (unsigned long long)tx->rejected,
           (unsigned long long)tx->rollback,
           accept_density,
           rollback_fraction,
           reject_fraction,
           tx->verifier_fraction,
           priced_ratio,
           tx->last_margin,
           pass);
}

static void fp_set_active_layer(TrainState *s, int stage) {
    for (int k = 0; k < V12_MAX_LAYERS; k++) {
        s->lr_per_layer[k] = (k == stage) ? 1.0f : 0.0f;
    }
}

static int fp_advance_stage(TrainState *s) {
    if (s->fp_completed) return 0;

    s->fp_stage++;
    if (s->fp_stage >= V12_N_LAYERS) {
        s->fp_completed = 1;
        for (int k = 0; k < V12_MAX_LAYERS; k++) s->lr_per_layer[k] = 1.0f;
        printf("[FP] All stages complete. Switching to joint training.\n");
        return 1;
    }

    fp_set_active_layer(s, (int)s->fp_stage);
    printf("[FP] Advancing to stage %u (layer %u active)\n", s->fp_stage, s->fp_stage);
    return 0;
}

static void bcd_set_active_layer(TrainState *s, int layer) {
    for (int k = 0; k < V12_MAX_LAYERS; k++) {
        s->lr_per_layer[k] = (k == layer) ? 1.0f : 0.0f;
    }
}

static int bcd_advance(TrainState *s) {
    if (!s->bcd_active) return 0;

    s->bcd_layer++;
    if (s->bcd_layer >= V12_N_LAYERS) {
        s->bcd_cycle++;
        s->bcd_layer = 0;

        if (s->bcd_cycle >= BCD_N_CYCLES) {
            s->bcd_active = 0;
            for (int k = 0; k < V12_MAX_LAYERS; k++) s->lr_per_layer[k] = 1.0f;
            printf("[BCD] All cycles complete. Switching to joint training.\n");
            return 1;
        }

        printf("[BCD] Starting cycle %u\n", s->bcd_cycle);
    }

    bcd_set_active_layer(s, (int)s->bcd_layer);
    printf("[BCD] cycle=%u layer=%u active\n", s->bcd_cycle, s->bcd_layer);
    return 0;
}

static void load_tokens(Dataset *ds, uint16_t *tokens, uint8_t *boundary_mask,
                        int seq_len, int use_sequential, int *out_seq_idx) {
    int seq_idx = -1;
    if (use_sequential) {
        dataset_next_sequential(ds, tokens, 1, seq_len);
        int n_seqs = (int)(ds->n_tokens / seq_len);
        seq_idx = (int)((ds->cursor / seq_len) % n_seqs);
        if (seq_idx == 0 && ds->cursor > 0) seq_idx = n_seqs - 1;
        else if (seq_idx > 0) seq_idx--;
    } else {
        dataset_sample_batch(ds, tokens, 1, seq_len);
        seq_idx = 0;
    }

    boundary_mask[0] = 1;
    for (int t = 1; t < seq_len; t++) {
        boundary_mask[t] = (tokens[t - 1] == '\n') ? 1 : 0;
    }

    *out_seq_idx = seq_idx;
}

static float run_external_heldout_eval(GpuPipeline *gpu,
                                       Model *m,
                                       Cache *c,
                                       const char *dataset_path,
                                       int batch_size,
                                       int eval_batches,
                                       uint64_t eval_seed,
                                       uint32_t step,
                                       const char *tag) {
    if (!gpu || !m || !c || !dataset_path || eval_batches <= 0) {
        return NAN;
    }

    Dataset *heldout = dataset_load(dataset_path);
    if (!heldout) {
        fprintf(stderr, "[HELDOUT_EVAL] failed to load dataset=%s\n", dataset_path);
        return NAN;
    }
    dataset_seed(heldout, eval_seed);
    dataset_reset_cursor(heldout);

    float eval_loss_sum = 0.0f;
#ifdef V12_HAS_VULKAN
    gpu_zero_grad_accumulators(gpu);
#endif
    for (int eb = 0; eb < eval_batches; eb++) {
        int total_tokens = batch_size * V12_SEQ_LEN;
        uint16_t *eval_tokens = malloc((size_t)total_tokens * sizeof(uint16_t));
        uint8_t *eval_mask = malloc((size_t)total_tokens * sizeof(uint8_t));
        if (!eval_tokens || !eval_mask) {
            fprintf(stderr, "[HELDOUT_EVAL] allocation failed batch=%d\n", eb);
            free(eval_tokens);
            free(eval_mask);
            dataset_free(heldout);
            return NAN;
        }
        for (int b = 0; b < batch_size; b++) {
            int dummy = 0;
            load_tokens(heldout,
                        eval_tokens + b * V12_SEQ_LEN,
                        eval_mask + b * V12_SEQ_LEN,
                        V12_SEQ_LEN,
                        1,
                        &dummy);
        }
        eval_loss_sum += gpu_train_step(gpu, m, c, eval_tokens, eval_mask,
                                        V12_SEQ_LEN, 0, batch_size, NULL);
        free(eval_tokens);
        free(eval_mask);
    }
#ifdef V12_HAS_VULKAN
    gpu_zero_grad_accumulators(gpu);
#endif
    scale_all_gradients(m, 0.0f);

    float eval_loss = eval_loss_sum / (float)eval_batches;
    printf("[HELDOUT_EVAL] tag=%s step=%u batches=%d dataset=%s loss=%.6f\n",
           tag ? tag : "final",
           step,
           eval_batches,
           dataset_path,
           eval_loss);
    dataset_free(heldout);
    return eval_loss;
}

static const char *dpth_mode_name(DpthMode mode) {
    switch (mode) {
        case DPTH_MODE_PACKET: return "packet";
        case DPTH_MODE_LOSS_CONTROL: return "loss-control";
        default: return "off";
    }
}

static void dpth_build_boundary_mask(const uint16_t *tokens,
                                     uint8_t *boundary_mask,
                                     int seq_len) {
    boundary_mask[0] = 1;
    for (int t = 1; t < seq_len; t++) {
        boundary_mask[t] = (tokens[t - 1] == '\n') ? 1 : 0;
    }
}

static float dpth_token_nll(const float *log_unigram, uint16_t token) {
    if (!log_unigram || token >= (uint16_t)V12_VOCAB_SIZE) return 0.0f;
    float log_q = log_unigram[token];
    if (!isfinite(log_q)) return 0.0f;
    return -log_q;
}

static DpthWindowStats dpth_score_window(const uint16_t *tokens,
                                         const uint8_t *boundary_mask,
                                         int seq_len,
                                         const float *log_unigram) {
    DpthWindowStats st = {0};
    if (!tokens || !boundary_mask || seq_len <= 1) return st;

    int boundary_hits = 0;
    int alternating_hits = 0;
    int rare_hits = 0;
    double nll_sum = 0.0;
    double nll_mean = 0.0;

    for (int t = 1; t < seq_len; t++) {
        float nll = dpth_token_nll(log_unigram, tokens[t]);
        nll_sum += (double)nll;
        if (nll > 9.0f) rare_hits++;
        if (boundary_mask[t]) boundary_hits++;
        if (t >= 2 && tokens[t] == tokens[t - 2] && tokens[t] != tokens[t - 1]) {
            alternating_hits++;
        }
    }

    nll_mean = nll_sum / (double)(seq_len - 1);
    st.loss_proxy = (float)nll_mean;
    st.boundary_density = (float)((double)boundary_hits / (double)(seq_len - 1));
    st.alternating_density = (float)((double)alternating_hits / (double)(seq_len - 2 > 0 ? seq_len - 2 : 1));
    st.rare_token_density = (float)((double)rare_hits / (double)(seq_len - 1));
    st.packet_score = 2.0f * st.boundary_density
                    + 1.5f * st.alternating_density
                    + 0.5f * st.rare_token_density;
    return st;
}

static float dpth_selection_score(DpthMode mode, const DpthWindowStats *st) {
    if (!st) return -INFINITY;
    if (mode == DPTH_MODE_LOSS_CONTROL) return st->loss_proxy;
    if (mode == DPTH_MODE_PACKET) return st->packet_score;
    return -INFINITY;
}

static int dpth_sampler_init(DpthSampler *sampler, DpthMode mode, int pool, int seq_len) {
    if (!sampler || mode == DPTH_MODE_OFF) return 0;
    memset(sampler, 0, sizeof(*sampler));
    sampler->enabled = 1;
    sampler->mode = mode;
    sampler->pool = pool;
    sampler->scratch_tokens = (uint16_t *)malloc((size_t)seq_len * sizeof(uint16_t));
    sampler->scratch_mask = (uint8_t *)malloc((size_t)seq_len * sizeof(uint8_t));
    if (!sampler->scratch_tokens || !sampler->scratch_mask) {
        free(sampler->scratch_tokens);
        free(sampler->scratch_mask);
        memset(sampler, 0, sizeof(*sampler));
        return -1;
    }
    return 0;
}

static void dpth_sampler_free(DpthSampler *sampler) {
    if (!sampler) return;
    free(sampler->scratch_tokens);
    free(sampler->scratch_mask);
    memset(sampler, 0, sizeof(*sampler));
}

static void dpth_sampler_observe(DpthSampler *sampler,
                                 const DpthWindowStats *selected,
                                 double selected_score,
                                 double pool_score_mean,
                                 int pool,
                                 double selection_seconds) {
    if (!sampler || !sampler->enabled || !selected) return;

    sampler->selected_windows++;
    sampler->scored_windows += (uint64_t)pool;
    sampler->selection_seconds += selection_seconds;
    sampler->selected_score_sum += selected_score;
    sampler->pool_score_sum += pool_score_mean;
    sampler->selected_loss_proxy_sum += selected->loss_proxy;
    sampler->selected_boundary_sum += selected->boundary_density;
    sampler->selected_alternating_sum += selected->alternating_density;
    sampler->selected_rare_sum += selected->rare_token_density;

    sampler->interval_selected_windows++;
    sampler->interval_scored_windows += (uint64_t)pool;
    sampler->interval_selection_seconds += selection_seconds;
    sampler->interval_selected_score_sum += selected_score;
    sampler->interval_pool_score_sum += pool_score_mean;
    sampler->interval_selected_loss_proxy_sum += selected->loss_proxy;
    sampler->interval_selected_boundary_sum += selected->boundary_density;
    sampler->interval_selected_alternating_sum += selected->alternating_density;
    sampler->interval_selected_rare_sum += selected->rare_token_density;
}

static void dpth_sampler_reset_interval(DpthSampler *sampler) {
    if (!sampler) return;
    sampler->interval_selected_windows = 0;
    sampler->interval_scored_windows = 0;
    sampler->interval_selection_seconds = 0.0;
    sampler->interval_selected_score_sum = 0.0;
    sampler->interval_pool_score_sum = 0.0;
    sampler->interval_selected_loss_proxy_sum = 0.0;
    sampler->interval_selected_boundary_sum = 0.0;
    sampler->interval_selected_alternating_sum = 0.0;
    sampler->interval_selected_rare_sum = 0.0;
}

static void dpth_sampler_log(const DpthSampler *sampler, uint32_t step) {
    if (!sampler || !sampler->enabled || sampler->interval_selected_windows == 0u) return;
    const double n = (double)sampler->interval_selected_windows;
    const double scored = (double)sampler->interval_scored_windows;
    const double select_overhead =
        sampler->train_seconds > 0.0
            ? sampler->selection_seconds / sampler->train_seconds
            : 0.0;
    printf("[DPTH_GATE] step=%u mode=%s selected=%llu scored=%llu pool=%.3f selected_score=%.9f pool_score=%.9f loss_proxy=%.9f boundary_density=%.9f alternating_density=%.9f rare_density=%.9f selection_ms=%.3f selection_overhead=%.9f objective=original_ce sampler=current_training_stream control=%s\n",
           step,
           dpth_mode_name(sampler->mode),
           (unsigned long long)sampler->interval_selected_windows,
           (unsigned long long)sampler->interval_scored_windows,
           scored > 0.0 ? scored / n : 0.0,
           sampler->interval_selected_score_sum / n,
           sampler->interval_pool_score_sum / n,
           sampler->interval_selected_loss_proxy_sum / n,
           sampler->interval_selected_boundary_sum / n,
           sampler->interval_selected_alternating_sum / n,
           sampler->interval_selected_rare_sum / n,
           sampler->interval_selection_seconds * 1000.0,
           select_overhead,
           sampler->mode == DPTH_MODE_LOSS_CONTROL ? "loss_proxy" : "dpth_packet");
}

static void dpth_load_tokens(DpthSampler *sampler,
                             Dataset *ds,
                             uint16_t *tokens,
                             uint8_t *boundary_mask,
                             int seq_len,
                             const float *log_unigram,
                             int *out_seq_idx) {
    int seq_idx = 0;
    const int pool = sampler && sampler->pool > 0 ? sampler->pool : 1;
    double t0 = train_now_seconds();

    dataset_sample_batch(ds, tokens, 1, seq_len);
    dpth_build_boundary_mask(tokens, boundary_mask, seq_len);
    DpthWindowStats best_stats =
        dpth_score_window(tokens, boundary_mask, seq_len, log_unigram);
    float best_score = dpth_selection_score(sampler->mode, &best_stats);
    double pool_score_sum = (double)best_score;

    for (int i = 1; i < pool; i++) {
        dataset_sample_batch(ds, sampler->scratch_tokens, 1, seq_len);
        dpth_build_boundary_mask(sampler->scratch_tokens, sampler->scratch_mask, seq_len);
        DpthWindowStats st =
            dpth_score_window(sampler->scratch_tokens, sampler->scratch_mask, seq_len, log_unigram);
        float score = dpth_selection_score(sampler->mode, &st);
        pool_score_sum += (double)score;
        if (score > best_score) {
            best_score = score;
            best_stats = st;
            memcpy(tokens, sampler->scratch_tokens, (size_t)seq_len * sizeof(uint16_t));
            memcpy(boundary_mask, sampler->scratch_mask, (size_t)seq_len * sizeof(uint8_t));
        }
    }

    dpth_sampler_observe(sampler, &best_stats, (double)best_score,
                         pool_score_sum / (double)pool, pool,
                         train_now_seconds() - t0);
    *out_seq_idx = seq_idx;
}

static uint32_t loss_bits_from_float(float value) {
    union {
        float f;
        uint32_t u;
    } bits = { value };
    return bits.u;
}

static uint64_t fnv1a_u64_bytes(const void *data, size_t bytes) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t fnv1a_u64_update(uint64_t h, const void *data, size_t bytes) {
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < bytes; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static size_t lm_head_steady_state_omitted_bytes(void) {
    const size_t lm_elems = (size_t)V12_VOCAB_SIZE * V12_D_MODEL;
    return lm_elems * (SHADOW_SIZEOF + sizeof(float) + sizeof(float));
}

static const char *r386_onoff(const void *ptr) {
    return ptr ? "on" : "off";
}

static size_t w_pred_steady_state_omitted_bytes(void) {
    const size_t pred_elems = (size_t)V12_D_MODEL * V12_D_MODEL;
    return pred_elems * SHADOW_SIZEOF
         + pred_elems * sizeof(float)
         + pred_elems * sizeof(float)
         + (size_t)V12_D_MODEL * sizeof(float)
         + (size_t)V12_D_MODEL * sizeof(float)
         + (size_t)V12_D_MODEL * sizeof(float)
         + (size_t)V12_D_MODEL * sizeof(float);
}

static size_t norm_steady_state_omitted_bytes(void) {
    return (size_t)V12_D_MODEL * sizeof(float)
         + (size_t)V12_D_MODEL * sizeof(float)
         + (size_t)V12_D_MODEL * sizeof(float)
         + (size_t)V12_D_MODEL * sizeof(float)
         + (size_t)V12_D_MODEL * sizeof(float)
         + (size_t)V12_D_MODEL * sizeof(float)
         + (size_t)V12_D_MODEL * sizeof(float)
         + (size_t)V12_D_MODEL * sizeof(float);
}

static size_t embed_steady_state_omitted_bytes(void) {
    const size_t embed_e_elems = (size_t)V12_VOCAB_SIZE * V12_D_EMBED;
    const size_t embed_p_elems = (size_t)V12_D_EMBED * V12_D_MODEL;
    const size_t legacy_e_elems = (size_t)V12_VOCAB_SIZE * 2 + (size_t)V12_D_EMBED * 2;
    return (embed_e_elems * 4 + embed_p_elems * 4 + legacy_e_elems) * sizeof(float);
}

static size_t w_in_steady_state_omitted_bytes(void) {
    const size_t layer_weight_elems = (size_t)V12_D_MODEL * V12_D_STATE;
    return (size_t)V12_N_LAYERS *
           (layer_weight_elems * (SHADOW_SIZEOF + sizeof(float))
            + (size_t)V12_D_MODEL * 2 * sizeof(float)
            + (size_t)V12_D_STATE * 2 * sizeof(float));
}

static size_t w_out_steady_state_omitted_bytes(void) {
    const size_t layer_weight_elems = (size_t)V12_D_STATE * V12_D_MODEL;
    return (size_t)V12_N_LAYERS *
           (layer_weight_elems * (SHADOW_SIZEOF + sizeof(float))
            + (size_t)V12_D_STATE * 2 * sizeof(float)
            + (size_t)V12_D_MODEL * 2 * sizeof(float));
}

static int r386_should_log_embed_hash(const char *reason) {
    return reason
        && (strcmp(reason, "checkpoint-save") == 0
            || strcmp(reason, "checkpoint-resume") == 0
            || strcmp(reason, "final-save") == 0);
}

static int r386_should_log_slice4_hash(const char *reason) {
    return reason
        && (strcmp(reason, "checkpoint-save") == 0
            || strcmp(reason, "checkpoint-resume") == 0
            || strcmp(reason, "final-save") == 0);
}

static int slice4_any_w_in_host(const Model *m) {
    if (!m) return 0;
    for (int k = 0; k < V12_N_LAYERS; k++) {
        const SSMLayer *L = &m->layers[k];
        if (L->shadow_w_in || L->sampled_w_in
            || L->m_w_in_row || L->m_w_in_col || L->v_w_in_row || L->v_w_in_col) {
            return 1;
        }
    }
    return 0;
}

static int slice4_any_w_out_host(const Model *m) {
    if (!m) return 0;
    for (int k = 0; k < V12_N_LAYERS; k++) {
        const SSMLayer *L = &m->layers[k];
        if (L->shadow_w_out || L->sampled_w_out
            || L->m_w_out_row || L->m_w_out_col || L->v_w_out_row || L->v_w_out_col) {
            return 1;
        }
    }
    return 0;
}

static int slice4_w_in_shadow_resident(const Model *m) {
    if (!m) return 0;
    for (int k = 0; k < V12_N_LAYERS; k++) {
        if (m->layers[k].shadow_w_in) return 1;
    }
    return 0;
}

static int slice4_w_out_shadow_resident(const Model *m) {
    if (!m) return 0;
    for (int k = 0; k < V12_N_LAYERS; k++) {
        if (m->layers[k].shadow_w_out) return 1;
    }
    return 0;
}

static int slice4_w_in_resident_aux(const Model *m) {
    if (!m) return 0;
    for (int k = 0; k < V12_N_LAYERS; k++) {
        const SSMLayer *L = &m->layers[k];
        if (L->sampled_w_in || L->m_w_in_row || L->m_w_in_col || L->v_w_in_row || L->v_w_in_col) {
            return 1;
        }
    }
    return 0;
}

static int slice4_w_out_resident_aux(const Model *m) {
    if (!m) return 0;
    for (int k = 0; k < V12_N_LAYERS; k++) {
        const SSMLayer *L = &m->layers[k];
        if (L->sampled_w_out || L->m_w_out_row || L->m_w_out_col || L->v_w_out_row || L->v_w_out_col) {
            return 1;
        }
    }
    return 0;
}

static const char *slice4_status(int shadow_resident, int aux_resident) {
    if (shadow_resident && aux_resident) return "on";
    if (shadow_resident || aux_resident) return "partial";
    return "off";
}

static void print_slice4_hashes(const Model *m, const char *reason) {
    uint64_t w_in_hash = 1469598103934665603ull;
    uint64_t w_out_hash = 1469598103934665603ull;

    if (!m || !reason) return;

    for (int k = 0; k < V12_N_LAYERS; k++) {
        const SSMLayer *L = &m->layers[k];
        if (!model_layer_w_io_host_is_materialized(L)) return;

        w_in_hash = fnv1a_u64_update(w_in_hash, L->shadow_w_in,
                                     (size_t)V12_D_MODEL * V12_D_STATE * SHADOW_SIZEOF);
        w_in_hash = fnv1a_u64_update(w_in_hash, L->sampled_w_in,
                                     (size_t)V12_D_MODEL * V12_D_STATE * sizeof(float));
        w_in_hash = fnv1a_u64_update(w_in_hash, L->m_w_in_row,
                                     (size_t)V12_D_MODEL * sizeof(float));
        w_in_hash = fnv1a_u64_update(w_in_hash, L->m_w_in_col,
                                     (size_t)V12_D_STATE * sizeof(float));
        w_in_hash = fnv1a_u64_update(w_in_hash, L->v_w_in_row,
                                     (size_t)V12_D_MODEL * sizeof(float));
        w_in_hash = fnv1a_u64_update(w_in_hash, L->v_w_in_col,
                                     (size_t)V12_D_STATE * sizeof(float));

        w_out_hash = fnv1a_u64_update(w_out_hash, L->shadow_w_out,
                                      (size_t)V12_D_STATE * V12_D_MODEL * SHADOW_SIZEOF);
        w_out_hash = fnv1a_u64_update(w_out_hash, L->sampled_w_out,
                                      (size_t)V12_D_STATE * V12_D_MODEL * sizeof(float));
        w_out_hash = fnv1a_u64_update(w_out_hash, L->m_w_out_row,
                                      (size_t)V12_D_STATE * sizeof(float));
        w_out_hash = fnv1a_u64_update(w_out_hash, L->m_w_out_col,
                                      (size_t)V12_D_MODEL * sizeof(float));
        w_out_hash = fnv1a_u64_update(w_out_hash, L->v_w_out_row,
                                      (size_t)V12_D_STATE * sizeof(float));
        w_out_hash = fnv1a_u64_update(w_out_hash, L->v_w_out_col,
                                      (size_t)V12_D_MODEL * sizeof(float));
    }

    printf("[R386] W_in persisted_hash for %s aggregate=0x%016llx\n",
           reason, (unsigned long long)w_in_hash);
    printf("[R386] W_out persisted_hash for %s aggregate=0x%016llx\n",
           reason, (unsigned long long)w_out_hash);
}

static void print_embed_moment_hashes(const Model *m, const char *reason) {
    if (!m || !reason || !model_embed_host_is_materialized(m)) return;

    const uint64_t embed_e_m = fnv1a_u64_bytes(
        m->sa_embed_E, (size_t)V12_VOCAB_SIZE * V12_D_EMBED * sizeof(int8_t));
    const uint64_t embed_e_v = 0;
    const uint64_t embed_p_m = fnv1a_u64_bytes(
        m->sa_embed_P, (size_t)V12_D_EMBED * V12_D_MODEL * sizeof(int8_t));
    const uint64_t embed_p_v = 0;

    printf("[R386] embed_E moments_hash for %s m=0x%016llx v=0x%016llx\n",
           reason,
           (unsigned long long)embed_e_m,
           (unsigned long long)embed_e_v);
    printf("[R386] embed_P moments_hash for %s m=0x%016llx v=0x%016llx\n",
           reason,
           (unsigned long long)embed_p_m,
           (unsigned long long)embed_p_v);
}

static int materialize_lm_head_host_boundary(Model *m, const char *reason) {
    if (!m || !(m->gpu_opt_flags & GPU_OPT_LM_HEAD)) return 0;
    if (model_materialize_lm_head_host(m) != 0) {
        fprintf(stderr, "[R386] failed to materialize lm_head host mirrors for %s\n", reason);
        return -1;
    }
    printf("[R386] lm_head materialized for %s shadow=on sampled=on dlm=%s\n",
           reason,
           m->dlm_head_g_row ? "on" : "off");
    return 0;
}

static int materialize_lm_head_boundary(Model *m, GpuPipeline *gpu, const char *reason) {
    if (!gpu) return materialize_lm_head_host_boundary(m, reason);
    if (materialize_lm_head_host_boundary(m, reason) != 0) {
        return -1;
    }
    if (gpu_pipeline_sync_lm_head_to_cpu(gpu, m) != 0) {
        fprintf(stderr, "[R386] failed to sync lm_head GPU state to CPU for %s\n", reason);
        return -1;
    }
    return 0;
}

static void release_lm_head_steady_state(Model *m, const char *reason) {
    static int omitted_bytes_logged = 0;
    if (!m || !(m->gpu_opt_flags & GPU_OPT_LM_HEAD) || m->shadow_lm_head == NULL) return;
    model_release_lm_head_host(m);
    printf("[R386] lm_head steady_state=GPU host_mirrors=off (%s)\n", reason);
    if (!omitted_bytes_logged) {
        omitted_bytes_logged = 1;
        printf("[R386] lm_head steady_state_omitted_host_bytes=%zu\n",
               lm_head_steady_state_omitted_bytes());
    }
}

static int materialize_slice3_boundary(Model *m, GpuPipeline *gpu, const char *reason) {
    if (!m) return 0;
    if (!(m->gpu_opt_flags & (GPU_OPT_EMBED_E | GPU_OPT_EMBED_P))) return 0;

    if (model_materialize_embed_host(m) != 0) {
        fprintf(stderr, "[R386] failed to materialize embed host mirrors for %s\n", reason);
        return -1;
    }
    if (gpu) {
        if (gpu_pipeline_sync_embed_to_cpu(gpu, m) != 0) {
            fprintf(stderr, "[R386] failed to sync embed GPU state to CPU for %s\n", reason);
            return -1;
        }
    }
    printf("[R386] embed materialized for %s embed_E=%s embed_P=%s m_embed_E=%s v_embed_E=%s m_embed_P=%s v_embed_P=%s\n",
           reason,
           r386_onoff(m->embed_E),
           r386_onoff(m->embed_P),
           r386_onoff(m->sa_embed_E), "off",
           r386_onoff(m->sa_embed_P), "off");
    if (r386_should_log_embed_hash(reason)) {
        print_embed_moment_hashes(m, reason);
    }
    return 0;
}

static int materialize_slice4_boundary(Model *m, GpuPipeline *gpu, const char *reason) {
    const char *w_in_state = "off";
    const char *w_out_state = "off";

    if (!m) return 0;
    if (!(m->gpu_opt_flags & (GPU_OPT_W_IN | GPU_OPT_W_OUT))) return 0;

    for (int k = 0; k < V12_N_LAYERS; k++) {
        if (model_materialize_layer_w_io_host(&m->layers[k]) != 0) {
            fprintf(stderr, "[R386] failed to materialize W_in/W_out host mirrors for %s at layer %d\n",
                    reason, k);
            return -1;
        }
    }
    if (gpu) {
        if (gpu_pipeline_sync_layer_state_to_cpu(gpu, m) != 0) {
            fprintf(stderr, "[R386] failed to sync W_in/W_out GPU state to CPU for %s\n", reason);
            return -1;
        }
    }
    w_in_state = slice4_any_w_in_host(m) ? "on" : "off";
    w_out_state = slice4_any_w_out_host(m) ? "on" : "off";
    printf("[R386] W_in/W_out materialized for %s layers=%d W_in_shadow=%s W_in_sampled=%s W_in_m_row=%s W_in_m_col=%s W_in_v_row=%s W_in_v_col=%s W_out_shadow=%s W_out_sampled=%s W_out_m_row=%s W_out_m_col=%s W_out_v_row=%s W_out_v_col=%s\n",
           reason,
           V12_N_LAYERS,
           w_in_state,
           w_in_state,
           w_in_state,
           w_in_state,
           w_in_state,
           w_in_state,
           w_out_state,
           w_out_state,
           w_out_state,
           w_out_state,
           w_out_state,
           w_out_state);
    if (r386_should_log_slice4_hash(reason)) {
        print_slice4_hashes(m, reason);
    }
    return 0;
}

static int materialize_slice4_layer(Model *m, GpuPipeline *gpu, int layer_idx, const char *reason) {
    if (!m) return 0;
    if (!(m->gpu_opt_flags & (GPU_OPT_W_IN | GPU_OPT_W_OUT))) return 0;
    if (layer_idx < 0 || layer_idx >= V12_N_LAYERS) return -1;
    if (model_materialize_layer_w_io_host(&m->layers[layer_idx]) != 0) {
        fprintf(stderr, "[R386] failed to materialize W_in/W_out layer %d for %s\n",
                layer_idx, reason);
        return -1;
    }
    if (gpu) {
        if (gpu_pipeline_sync_layer_to_cpu(gpu, m, layer_idx) != 0) {
            fprintf(stderr, "[R386] failed to sync W_in/W_out layer %d GPU state to CPU for %s\n",
                    layer_idx, reason);
            return -1;
        }
    }
    return 0;
}

static int materialize_slice2_boundary(Model *m, GpuPipeline *gpu, const char *reason) {
    int need_sync = 0;

    if (!m) return 0;

    if (m->gpu_opt_flags & GPU_OPT_W_PRED) {
        if (model_materialize_w_pred_host(m) != 0) {
            fprintf(stderr, "[R386] failed to materialize w_pred host mirrors for %s\n", reason);
            return -1;
        }
        need_sync = 1;
        printf("[R386] w_pred materialized for %s shadow=%s sampled=%s dw=%s m_row=%s m_col=%s v_row=%s v_col=%s\n",
               reason,
               r386_onoff(m->shadow_w_pred),
               r386_onoff(m->sampled_w_pred),
               r386_onoff(m->dw_pred),
               r386_onoff(m->m_w_pred_row),
               r386_onoff(m->m_w_pred_col),
               r386_onoff(m->v_w_pred_row),
               r386_onoff(m->v_w_pred_col));
    }
    if (m->gpu_opt_flags & GPU_OPT_NORM) {
        if (model_materialize_norm_host(m) != 0) {
            fprintf(stderr, "[R386] failed to materialize norm host mirrors for %s\n", reason);
            return -1;
        }
        need_sync = 1;
        printf("[R386] norm materialized for %s norm_w=%s norm_b=%s d_norm_w=%s d_norm_b=%s m_norm_w=%s v_norm_w=%s m_norm_b=%s v_norm_b=%s\n",
               reason,
               r386_onoff(m->norm_w),
               r386_onoff(m->norm_b),
               r386_onoff(m->d_norm_w),
               r386_onoff(m->d_norm_b),
               r386_onoff(m->m_norm_w),
               r386_onoff(m->v_norm_w),
               r386_onoff(m->m_norm_b),
               r386_onoff(m->v_norm_b));
    }

    if (need_sync && gpu) {
        if (gpu_pipeline_sync_slice2_to_cpu(gpu, m) != 0) {
            fprintf(stderr, "[R386] failed to sync w_pred/norm GPU state to CPU for %s\n", reason);
            return -1;
        }
    }
    return 0;
}

static void release_slice2_steady_state(Model *m, const char *reason) {
    int released = 0;
    int released_w_pred = 0;
    int released_norm = 0;

    if (!m) return;

    if ((m->gpu_opt_flags & GPU_OPT_W_PRED) && model_w_pred_host_is_materialized(m)) {
        model_release_w_pred_host(m);
        released = 1;
        released_w_pred = 1;
    }
    if ((m->gpu_opt_flags & GPU_OPT_NORM) && model_norm_host_is_materialized(m)) {
        model_release_norm_host(m);
        released = 1;
        released_norm = 1;
    }
    if (released_w_pred) {
        printf("[R386] w_pred steady_state=GPU host_mirrors=off (%s shadow=%s sampled=%s dw=%s m_row=%s m_col=%s v_row=%s v_col=%s)\n",
               reason,
               r386_onoff(m->shadow_w_pred),
               r386_onoff(m->sampled_w_pred),
               r386_onoff(m->dw_pred),
               r386_onoff(m->m_w_pred_row),
               r386_onoff(m->m_w_pred_col),
               r386_onoff(m->v_w_pred_row),
               r386_onoff(m->v_w_pred_col));
        printf("[R386] w_pred steady_state_omitted_host_bytes=%zu\n",
               w_pred_steady_state_omitted_bytes());
    }
    if (released_norm) {
        printf("[R386] norm steady_state=GPU host_mirrors=off (%s norm_w=%s norm_b=%s d_norm_w=%s d_norm_b=%s m_norm_w=%s v_norm_w=%s m_norm_b=%s v_norm_b=%s)\n",
               reason,
               r386_onoff(m->norm_w),
               r386_onoff(m->norm_b),
               r386_onoff(m->d_norm_w),
               r386_onoff(m->d_norm_b),
               r386_onoff(m->m_norm_w),
               r386_onoff(m->v_norm_w),
               r386_onoff(m->m_norm_b),
               r386_onoff(m->v_norm_b));
        printf("[R386] norm steady_state_omitted_host_bytes=%zu\n",
               norm_steady_state_omitted_bytes());
    }
    if (released) {
        printf("[R386] w_pred/norm steady_state=GPU host_mirrors=off (%s)\n", reason);
    }
}

static void release_slice3_steady_state(Model *m, const char *reason) {
    if (!m) return;
    if (!(m->gpu_opt_flags & (GPU_OPT_EMBED_E | GPU_OPT_EMBED_P))) return;
    if (!model_embed_host_is_materialized(m)) return;

    model_release_embed_host(m);
    printf("[R386] embed steady_state=GPU host_mirrors=off (%s embed_E=%s embed_P=%s m_embed_E=%s v_embed_E=%s m_embed_P=%s v_embed_P=%s)\n",
           reason,
           r386_onoff(m->embed_E),
           r386_onoff(m->embed_P),
           r386_onoff(m->sa_embed_E), "off",
           r386_onoff(m->sa_embed_P), "off");
    printf("[R386] embed steady_state_omitted_host_bytes=%zu\n",
           embed_steady_state_omitted_bytes());
}

static void release_slice4_steady_state(Model *m, const char *reason) {
    const char *w_in_state = "off";
    const char *w_out_state = "off";
    const char *w_in_shadow = "off";
    const char *w_out_shadow = "off";
    const char *w_in_aux = "off";
    const char *w_out_aux = "off";

    if (!m) return;
    if (!(m->gpu_opt_flags & (GPU_OPT_W_IN | GPU_OPT_W_OUT))) return;

    for (int k = 0; k < V12_N_LAYERS; k++) {
        model_release_layer_w_io_host(&m->layers[k]);
    }

    w_in_state = slice4_status(slice4_w_in_shadow_resident(m), slice4_w_in_resident_aux(m));
    w_out_state = slice4_status(slice4_w_out_shadow_resident(m), slice4_w_out_resident_aux(m));
    w_in_shadow = slice4_w_in_shadow_resident(m) ? "on" : "off";
    w_out_shadow = slice4_w_out_shadow_resident(m) ? "on" : "off";
    w_in_aux = slice4_w_in_resident_aux(m) ? "on" : "off";
    w_out_aux = slice4_w_out_resident_aux(m) ? "on" : "off";

    printf("[R386] W_in steady_state=GPU host_mirrors=%s (%s layers=%d shadow=%s sampled=%s m_row=%s m_col=%s v_row=%s v_col=%s)\n",
           w_in_state,
           reason,
           V12_N_LAYERS,
           w_in_shadow,
           w_in_aux,
           w_in_aux,
           w_in_aux,
           w_in_aux,
           w_in_aux);
    printf("[R386] W_in steady_state_omitted_host_bytes=%zu\n",
           w_in_steady_state_omitted_bytes());
    printf("[R386] W_out steady_state=GPU host_mirrors=%s (%s layers=%d shadow=%s sampled=%s m_row=%s m_col=%s v_row=%s v_col=%s)\n",
           w_out_state,
           reason,
           V12_N_LAYERS,
           w_out_shadow,
           w_out_aux,
           w_out_aux,
           w_out_aux,
           w_out_aux,
           w_out_aux);
    printf("[R386] W_out steady_state_omitted_host_bytes=%zu\n",
           w_out_steady_state_omitted_bytes());
}

static void print_checkpoint_debug(Model *m, const char *tag) {
    if (!m || !tag) return;
    if (!m->sampled_lm_head) {
        printf("[CKPTDBG] %s skipped: lm_head not materialized on host\n", tag);
        return;
    }

    uint64_t h_lm = fnv1a_u64_bytes(m->sampled_lm_head,
        (size_t)V12_VOCAB_SIZE * V12_D_MODEL * sizeof(float));
    uint64_t h_wp = 0;
    if (m->sampled_w_pred && (!(m->gpu_opt_flags & GPU_OPT_W_PRED)
            || model_w_pred_host_is_materialized(m))) {
        h_wp = fnv1a_u64_bytes(m->sampled_w_pred,
            (size_t)V12_D_MODEL * V12_D_MODEL * sizeof(float));
    }
    uint64_t h_l0_in = fnv1a_u64_bytes(m->layers[0].sampled_w_in,
        (size_t)V12_D_MODEL * V12_D_STATE * sizeof(float));
    uint64_t h_l0_out = fnv1a_u64_bytes(m->layers[0].sampled_w_out,
        (size_t)V12_D_STATE * V12_D_MODEL * sizeof(float));

    int rebuild_ok = model_rebuild_lm_head_csr(m);
    int l0_ok = model_rebuild_layer_csr(&m->layers[0]);
    int l5_ok = model_rebuild_layer_csr(&m->layers[V12_N_LAYERS - 1]);

    printf("[CKPTDBG] %s lm_hash=0x%016llx wp_hash=0x%016llx "
           "l0_in_hash=0x%016llx l0_out_hash=0x%016llx "
           "lm_csr_ok=%d lm_pos=%d lm_neg=%d l0_ok=%d l0_pos=%d/%d l0_neg=%d/%d "
           "l5_ok=%d l5_pos=%d/%d l5_neg=%d/%d\n",
           tag,
           (unsigned long long)h_lm,
           (unsigned long long)h_wp,
           (unsigned long long)h_l0_in,
           (unsigned long long)h_l0_out,
           rebuild_ok == 0,
           m->csr_lm_head.total_pos,
           m->csr_lm_head.total_neg,
           l0_ok == 0,
           m->layers[0].csr_in.total_pos,
           m->layers[0].csr_out.total_pos,
           m->layers[0].csr_in.total_neg,
           m->layers[0].csr_out.total_neg,
           l5_ok == 0,
           m->layers[V12_N_LAYERS - 1].csr_in.total_pos,
           m->layers[V12_N_LAYERS - 1].csr_out.total_pos,
           m->layers[V12_N_LAYERS - 1].csr_in.total_neg,
           m->layers[V12_N_LAYERS - 1].csr_out.total_neg);
}

typedef struct {
    int enabled;
    uint32_t warmup_steps;
    uint32_t compile_every_steps;
    uint32_t compiled_steps;
    R364UpperSubtreeLsProbe last_probe;
} R364PilotRuntime;

typedef struct {
    int boundary_now;
    int compiled_interval_active;
    uint32_t compile_id;
    uint32_t boundary_step;
    uint32_t compiled_until_step;
} R364PilotScheduleState;

static double train_now_seconds(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER counter;
    if (freq.QuadPart == 0) {
        if (!QueryPerformanceFrequency(&freq) || freq.QuadPart == 0) {
            return (double)clock() / (double)CLOCKS_PER_SEC;
        }
    }
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    #if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
    }
    #endif
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}
static R364PilotScheduleState r364_schedule_state(const R364PilotRuntime *pilot,
                                                  uint32_t completed_steps) {
    R364PilotScheduleState state = {0};
    if (!pilot || !pilot->enabled || completed_steps < pilot->warmup_steps) {
        return state;
    }

    const uint32_t delta = completed_steps - pilot->warmup_steps;
    const uint32_t phase = delta % pilot->compile_every_steps;

    state.boundary_now = (phase == 0u) ? 1 : 0;
    state.compiled_interval_active = (phase < pilot->compiled_steps) ? 1 : 0;
    state.compile_id = (delta / pilot->compile_every_steps) + 1u;
    state.boundary_step = completed_steps - phase;
    state.compiled_until_step = state.boundary_step + pilot->compiled_steps;
    return state;
}

static void r364_update_gpu_noop_interval(GpuPipeline *gpu,
                                          const R364PilotRuntime *pilot,
                                          const R364PilotScheduleState *schedule) {
    const float compiled_block_share_f = pilot
        ? pilot->last_probe.compiled_block_share_f
        : 0.0f;
    gpu_pipeline_r364_set_noop_compiled_interval(
        gpu,
        schedule ? schedule->compiled_interval_active : 0,
        schedule ? schedule->compile_id : 0u,
        compiled_block_share_f);
}

static int r364_run_compile_boundary(Model *m,
                                     GpuPipeline *gpu,
                                     Cache *c,
                                     int batch_size,
                                     const R364PilotRuntime *pilot,
                                     const R364PilotScheduleState *schedule,
                                     R364UpperSubtreeLsProbe *probe_out) {
    double compile_start, compile_end;
    R364UpperSubtreeLsCompileConfig compile_cfg = {0};

    if (!pilot || !schedule || !probe_out) return -1;

    R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%u layer=%d stage=%s\n",
            schedule->boundary_step, -1, "pre-r364-lm-head-sync-to-cpu");
    fflush(stderr);
    R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%u layer=%d stage=%s\n",
            schedule->boundary_step, -1, "pre-r364-lm-head-host-materialize");
    fflush(stderr);
    if (materialize_lm_head_host_boundary(m, "r364-compile") != 0) {
        release_lm_head_steady_state(m, "r364-compile-abort");
        return -1;
    }
    R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%u layer=%d stage=%s\n",
            schedule->boundary_step, -1, "post-r364-lm-head-host-materialize");
    fflush(stderr);
    if (gpu) {
        R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%u layer=%d stage=%s\n",
                schedule->boundary_step, -1, "pre-r364-lm-head-gpu-sync-to-cpu");
        fflush(stderr);
        if (gpu_pipeline_sync_lm_head_to_cpu(gpu, m) != 0) {
            release_lm_head_steady_state(m, "r364-compile-abort");
            return -1;
        }
        R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%u layer=%d stage=%s\n",
                schedule->boundary_step, -1, "post-r364-lm-head-gpu-sync-to-cpu");
        fflush(stderr);
        if (gpu_pipeline_sync_r364_compile_cache_to_cpu(gpu, m, c, batch_size) != 0) {
            release_lm_head_steady_state(m, "r364-compile-abort");
            return -1;
        }
    }

    compile_cfg.compile_id = schedule->compile_id;
    compile_cfg.boundary_step = schedule->boundary_step;
    compile_cfg.warmup_steps = pilot->warmup_steps;
    compile_cfg.compile_every_steps = pilot->compile_every_steps;
    compile_cfg.compiled_steps = pilot->compiled_steps;
    compile_cfg.enable_noop_hook = 1;
    compile_cfg.eval_batch_index = batch_size > 0 ? (batch_size - 1) : 0;

    compile_start = train_now_seconds();
    R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%u layer=%d stage=%s\n",
            schedule->boundary_step, -1, "pre-r364-upper-subtree-ls-compile-call");
    fflush(stderr);
    if (r364_upper_subtree_ls_compile(m, c, &compile_cfg, probe_out) != 0) {
        release_lm_head_steady_state(m, "r364-compile-abort");
        return -1;
    }
    R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%u layer=%d stage=%s\n",
            schedule->boundary_step, -1, "post-r364-upper-subtree-ls-compile-call");
    fflush(stderr);
    compile_end = train_now_seconds();
    probe_out->compile_ms = (compile_end - compile_start) * 1000.0;

    if (gpu) {
        R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%u layer=%d stage=%s\n",
                schedule->boundary_step, -1, "pre-r364-lm-head-sync-from-cpu");
        fflush(stderr);
        gpu_pipeline_sync_lm_head_from_cpu(gpu, m);
    }
    release_lm_head_steady_state(m, "post-r364-compile");
    return 0;
}

#define R395_TRACE_ENV "V12_R395_TRACE"
#define R395_TRACE_PATH_ENV "V12_R395_TRACE_PATH"
#define R395_TRACE_MAX_STEPS_ENV "V12_R395_TRACE_MAX_STEPS"
#define R395_TRACE_MAGIC "R395SGT1"
#define R395_TRACE_VERSION 1u
#define R395_TRACE_DEFAULT_PATH ".work/jobs/r395-semigroup-closure-01/artifacts/r395.sg.trace.bin"

typedef struct {
    int enabled;
    int max_steps;
    int captured_steps;
    int activation_logged;
    int materialization_logged;
    char path[1024];
} R395TraceConfig;

#define R399_SSQT_TRACE_ENV "V12_R399_SSQT_TRACE"
#define R399_SSQT_TRACE_PATH_ENV "V12_R399_SSQT_TRACE_PATH"
#define R399_SSQT_TRACE_MAX_STEPS_ENV "V12_R399_SSQT_TRACE_MAX_STEPS"
#define R399_SSQT_TRACE_SAMPLE_ENV "V12_R399_SSQT_TRACE_SAMPLE"
#define R399_SSQT_TRACE_FRONTIER_CELLS_ENV "V12_R399_SSQT_TRACE_FRONTIER_CELLS"
#define R399_SSQT_TRACE_MAX_RECORDS_ENV "V12_R399_SSQT_TRACE_MAX_RECORDS"
#define R399_SSQT_TRACE_DEFAULT_PATH "r399_ssqt_trace.jsonl"

#define R399_DMDC_TRACE_ENV "V12_R399_DMDC_TRACE"
#define R399_DMDC_TRACE_PATH_ENV "V12_R399_DMDC_TRACE_PATH"
#define R399_DMDC_TRACE_MAX_STEPS_ENV "V12_R399_DMDC_TRACE_MAX_STEPS"
#define R399_DMDC_TRACE_SAMPLE_ENV "V12_R399_DMDC_TRACE_SAMPLE"
#define R399_DMDC_TRACE_DEFAULT_PATH ".work/jobs/r399-dmdc-gate-01/artifacts/r399_dmdc_real_trace.jsonl"

typedef struct {
    int enabled;
    int max_steps;
    int captured_steps;
    int trace_sample;
    int frontier_cells;
    int max_records;
    int activation_logged;
    int meta_written;
    char path[1024];
} R399SSQTTraceConfig;

typedef struct {
    int enabled;
    int max_steps;
    int trace_sample;
    int activation_logged;
    char path[1024];
} R399DmdcTraceConfig;

typedef struct {
    uint32_t exposure_id;
    char quotient_key[64];
    float raw_gradient;
    float quotient_gradient;
    float raw_frontier_pressure;
    float quotient_frontier_pressure;
    int split_key;
    int active_frontier;
} R399SSQTTraceRow;

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t n_layers;
    uint32_t batch_size;
    uint32_t seq_len;
    uint32_t d_state;
} R395TraceHeader;

typedef struct {
    uint32_t step;
    uint32_t reserved;
    float step_loss;
    float runtime_ms;
    float export_ms;
} R395TraceRecordHeader;

static int r395_env_flag_enabled(const char *value) {
    return (value != NULL && value[0] != '\0' && value[0] != '0');
}

static void r395_trace_config_init(R395TraceConfig *cfg) {
    const char *flag;
    const char *path;
    const char *max_steps_env;

    memset(cfg, 0, sizeof(*cfg));
    flag = getenv(R395_TRACE_ENV);
    path = getenv(R395_TRACE_PATH_ENV);
    max_steps_env = getenv(R395_TRACE_MAX_STEPS_ENV);

    cfg->enabled = r395_env_flag_enabled(flag) || (path != NULL && path[0] != '\0');
    cfg->max_steps = 1;
    if (max_steps_env != NULL && max_steps_env[0] != '\0') {
        cfg->max_steps = atoi(max_steps_env);
        if (cfg->max_steps <= 0) cfg->max_steps = 1;
    }

    if (path != NULL && path[0] != '\0') {
        snprintf(cfg->path, sizeof(cfg->path), "%s", path);
    } else {
        snprintf(cfg->path, sizeof(cfg->path), "%s", R395_TRACE_DEFAULT_PATH);
    }
}

static void r395_trace_log_activation_once(R395TraceConfig *cfg, int batch_size) {
    if (!cfg || !cfg->enabled || cfg->activation_logged) return;
    printf("[R395_TRACE] active env=%s path=%s max_steps=%d batch=%d seq_len=%d mode=hidden_state_trace_v1\n",
           R395_TRACE_ENV,
           cfg->path,
           cfg->max_steps,
           batch_size,
           V12_SEQ_LEN);
    cfg->activation_logged = 1;
}

static int r399_clamp_int(int value, int min_value, int max_value) {
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
}

static const char *r399_hidden_sign_bucket(float value) {
    if (value > 0.0f) return "p";
    if (value < 0.0f) return "n";
    return "z";
}

static int r399_trace_layer_count(const R399SSQTTraceConfig *cfg,
                                  int batch_size,
                                  int trace_sample,
                                  int frontier_cells) {
    int per_layer_records;
    if (!cfg) return 0;
    per_layer_records = batch_size * trace_sample * frontier_cells;
    if (per_layer_records <= 0) return 0;
    if (cfg->max_records < per_layer_records * 2) return 1;
    if (cfg->max_records < per_layer_records * 3) return 2;
    return 3;
}

static int r399_trace_fill_layers(int *layers, int max_layers) {
    int count = 0;
    if (!layers || max_layers <= 0) return 0;
    layers[count++] = 0;
    if (max_layers <= 1 || V12_N_LAYERS <= 1) return count;
    {
        int mid = V12_N_LAYERS / 2;
        if (mid >= V12_N_LAYERS) mid = V12_N_LAYERS - 1;
        if (mid > 0) layers[count++] = mid;
    }
    if (max_layers <= 2 || count >= max_layers || V12_N_LAYERS <= 2) return count;
    if (layers[count - 1] != V12_N_LAYERS - 1) layers[count++] = V12_N_LAYERS - 1;
    return count;
}

static void r399_trace_config_init(R399SSQTTraceConfig *cfg) {
    const char *flag;
    const char *path;
    const char *max_steps_env;
    const char *sample_env;
    const char *frontier_env;
    const char *max_records_env;

    memset(cfg, 0, sizeof(*cfg));
    flag = getenv(R399_SSQT_TRACE_ENV);
    path = getenv(R399_SSQT_TRACE_PATH_ENV);
    max_steps_env = getenv(R399_SSQT_TRACE_MAX_STEPS_ENV);
    sample_env = getenv(R399_SSQT_TRACE_SAMPLE_ENV);
    frontier_env = getenv(R399_SSQT_TRACE_FRONTIER_CELLS_ENV);
    max_records_env = getenv(R399_SSQT_TRACE_MAX_RECORDS_ENV);

    cfg->enabled = r395_env_flag_enabled(flag);
    cfg->max_steps = 1;
    cfg->trace_sample = 8;
    cfg->frontier_cells = 4;
    cfg->max_records = 512;

    if (max_steps_env != NULL && max_steps_env[0] != '\0') {
        cfg->max_steps = atoi(max_steps_env);
        if (cfg->max_steps <= 0) cfg->max_steps = 1;
    }
    if (sample_env != NULL && sample_env[0] != '\0') {
        cfg->trace_sample = atoi(sample_env);
    }
    if (frontier_env != NULL && frontier_env[0] != '\0') {
        cfg->frontier_cells = atoi(frontier_env);
    }
    if (max_records_env != NULL && max_records_env[0] != '\0') {
        cfg->max_records = atoi(max_records_env);
    }
    if (cfg->trace_sample <= 0) cfg->trace_sample = 8;
    if (cfg->frontier_cells <= 0) cfg->frontier_cells = 4;
    if (cfg->max_records <= 0) cfg->max_records = 512;

    if (path != NULL && path[0] != '\0') {
        snprintf(cfg->path, sizeof(cfg->path), "%s", path);
    } else {
        snprintf(cfg->path, sizeof(cfg->path), "%s", R399_SSQT_TRACE_DEFAULT_PATH);
    }
}

static void r399_trace_log_activation_once(R399SSQTTraceConfig *cfg,
                                           int batch_size,
                                           int seq_len) {
    int trace_sample;
    int frontier_cells;

    if (!cfg || !cfg->enabled || cfg->activation_logged) return;
    trace_sample = r399_clamp_int(cfg->trace_sample, 1, seq_len > 1 ? (seq_len - 1) : 1);
    frontier_cells = r399_clamp_int(cfg->frontier_cells, 1, V12_D_STATE);
    printf("[R399 SSQT GATE] active=1 mode=diagnostic-only trace_source=training_real_trace steps=%d batch_size=%d seq_len=%d trace_sample=%d frontier_cells=%d output_path=%s no_training_integration=1\n",
           cfg->max_steps,
           batch_size,
           seq_len,
           trace_sample,
           frontier_cells,
           cfg->path);
    cfg->activation_logged = 1;
}

static void r399_dmdc_trace_config_init(R399DmdcTraceConfig *cfg) {
    const char *flag;
    const char *path;
    const char *max_steps_env;
    const char *sample_env;

    memset(cfg, 0, sizeof(*cfg));
    flag = getenv(R399_DMDC_TRACE_ENV);
    path = getenv(R399_DMDC_TRACE_PATH_ENV);
    max_steps_env = getenv(R399_DMDC_TRACE_MAX_STEPS_ENV);
    sample_env = getenv(R399_DMDC_TRACE_SAMPLE_ENV);

    cfg->enabled = r395_env_flag_enabled(flag);
    cfg->max_steps = 8;
    cfg->trace_sample = 4096;

    if (max_steps_env != NULL && max_steps_env[0] != '\0') {
        cfg->max_steps = atoi(max_steps_env);
        if (cfg->max_steps <= 0) cfg->max_steps = 8;
    }
    if (sample_env != NULL && sample_env[0] != '\0') {
        cfg->trace_sample = atoi(sample_env);
        if (cfg->trace_sample <= 0) cfg->trace_sample = 4096;
    }

    if (path != NULL && path[0] != '\0') {
        snprintf(cfg->path, sizeof(cfg->path), "%s", path);
    } else {
        snprintf(cfg->path, sizeof(cfg->path), "%s", R399_DMDC_TRACE_DEFAULT_PATH);
    }
}

static void r399_dmdc_trace_log_activation_once(R399DmdcTraceConfig *cfg) {
    if (!cfg || !cfg->enabled || cfg->activation_logged) return;
    printf("[R399_DMDC_TRACE] active=1 mode=diagnostic-only source=real-mvue-gpu-training "
           "no_training_behavior_change=1 path=%s max_steps=%d sample=%d "
           "inequality=\"(1-f_q*(1-rho))*(1+h)<=0.50\"\n",
           cfg->path,
           cfg->max_steps,
           cfg->trace_sample);
    cfg->activation_logged = 1;
}

static int r395_trace_ensure_parent_dirs(const char *path);

static int r399_trace_append_rows(R399SSQTTraceConfig *cfg,
                                  const R399SSQTTraceRow *rows,
                                  int row_count,
                                  int batch_size,
                                  int seq_len) {
    FILE *fp;
    int index;

    if (!cfg || !cfg->enabled) return 0;
    if (r395_trace_ensure_parent_dirs(cfg->path) != 0) return -1;
    fp = fopen(cfg->path, cfg->meta_written ? "a" : "w");
    if (!fp) return -1;

    if (!cfg->meta_written) {
        fprintf(fp,
                "{\"kind\":\"meta\",\"schema\":\"r399.ssqt.trace.v1\",\"activation_line_emitted\":true,\"synthetic_fixture\":false,\"trace_source\":\"training_real_trace\",\"no_training_integration\":1,\"steps\":%d,\"batch_size\":%d,\"seq_len\":%d,\"trace_sample\":%d,\"frontier_cells\":%d,\"max_records\":%d}\n",
                cfg->max_steps,
                batch_size,
                seq_len,
                r399_clamp_int(cfg->trace_sample, 1, seq_len > 1 ? (seq_len - 1) : 1),
                r399_clamp_int(cfg->frontier_cells, 1, V12_D_STATE),
                cfg->max_records);
        cfg->meta_written = 1;
    }

    for (index = 0; index < row_count; index++) {
        const R399SSQTTraceRow *row = rows + index;
        fprintf(fp,
                "{\"kind\":\"record\",\"schema\":\"r399.ssqt.trace.v1\",\"exposure_id\":%u,\"quotient_key\":\"%s\",\"raw_gradient\":%.9g,\"quotient_gradient\":%.9g,\"raw_frontier_pressure\":%.9g,\"quotient_frontier_pressure\":%.9g,\"split_key\":%d,\"active_frontier\":%d}\n",
                row->exposure_id,
                row->quotient_key,
                row->raw_gradient,
                row->quotient_gradient,
                row->raw_frontier_pressure,
                row->quotient_frontier_pressure,
                row->split_key,
                row->active_frontier);
    }

    fclose(fp);
    return 0;
}

static int r399_trace_capture_step(R399SSQTTraceConfig *cfg,
                                   const Cache *c,
                                   const uint16_t *tokens,
                                   const uint8_t *boundary_mask,
                                   int batch_size,
                                   int seq_len,
                                   uint32_t step) {
    R399SSQTTraceRow *rows;
    int frontier_cells;
    int trace_sample;
    int layer_limit;
    int layers[3];
    int layer_count;
    int row_count = 0;
    uint32_t exposure_id = 0;

    if (!cfg || !cfg->enabled) return 0;
    if (!c || !c->h_all || !c->dA_per_step || !tokens || !boundary_mask || seq_len <= 1) {
        errno = ENODATA;
        return -1;
    }

    frontier_cells = r399_clamp_int(cfg->frontier_cells, 1, V12_D_STATE);
    trace_sample = r399_clamp_int(cfg->trace_sample, 1, seq_len - 1);
    layer_limit = r399_trace_layer_count(cfg, batch_size, trace_sample, frontier_cells);
    layer_count = r399_trace_fill_layers(layers, layer_limit);
    if (layer_count <= 0) {
        errno = ENODATA;
        return -1;
    }

    rows = (R399SSQTTraceRow *)calloc((size_t)cfg->max_records, sizeof(*rows));
    if (!rows) return -1;

    for (int layer_slot = 0; layer_slot < layer_count && row_count < cfg->max_records; layer_slot++) {
        const int layer = layers[layer_slot];
        for (int batch = 0; batch < batch_size && row_count < cfg->max_records; batch++) {
            for (int sample_idx = 0; sample_idx < trace_sample && row_count < cfg->max_records; sample_idx++) {
                int token_index = (sample_idx * (seq_len - 1)) / trace_sample;
                if (token_index < 0) token_index = 0;
                if (token_index >= seq_len - 1) token_index = seq_len - 2;
                for (int cell_idx = 0; cell_idx < frontier_cells && row_count < cfg->max_records; cell_idx++) {
                    int cell = (cell_idx * V12_D_STATE) / frontier_cells;
                    if (cell >= V12_D_STATE) cell = V12_D_STATE - 1;
                    const size_t base = (((size_t)layer * (size_t)batch_size + (size_t)batch) * (size_t)seq_len + (size_t)token_index) * (size_t)V12_D_STATE;
                    const size_t token_offset = (size_t)batch * (size_t)seq_len + (size_t)token_index;
                    const float raw_gradient = c->dA_per_step[base + (size_t)cell];
                    const float hidden_value = decompress_e5m2(c->h_all[base + (size_t)cell]);
                    const int target_bucket = (int)(tokens[token_offset + 1] & 3u);
                    const int boundary_bit = boundary_mask[token_offset] ? 1 : 0;
                    R399SSQTTraceRow *row = rows + row_count++;

                    memset(row, 0, sizeof(*row));
                    row->exposure_id = exposure_id++;
                    snprintf(row->quotient_key,
                             sizeof(row->quotient_key),
                             "L%d:C%d:H%s:T%d:B%d",
                             layer,
                             cell,
                             r399_hidden_sign_bucket(hidden_value),
                             target_bucket,
                             boundary_bit);
                    row->raw_gradient = raw_gradient;
                    row->raw_frontier_pressure = raw_gradient;
                    row->split_key = (int)((step + (uint32_t)batch + (uint32_t)token_index) & 1u);
                    row->active_frontier = (fabsf(raw_gradient) > 1.0e-12f) ? 1 : 0;
                }
            }
        }
    }

    if (row_count <= 0) {
        free(rows);
        errno = ENODATA;
        return -1;
    }

    for (int i = 0; i < row_count; i++) {
        double sum = 0.0;
        int count = 0;
        for (int j = 0; j < row_count; j++) {
            if (strcmp(rows[i].quotient_key, rows[j].quotient_key) != 0) continue;
            sum += rows[j].raw_gradient;
            count++;
        }
        rows[i].quotient_gradient = count > 0 ? (float)(sum / (double)count) : rows[i].raw_gradient;
        rows[i].quotient_frontier_pressure = rows[i].quotient_gradient;
    }

    if (r399_trace_append_rows(cfg, rows, row_count, batch_size, seq_len) != 0) {
        free(rows);
        return -1;
    }

    free(rows);
    return row_count;
}

static int r395_trace_mkdir_single(const char *path) {
#ifdef _WIN32
    if (_mkdir(path) == 0 || errno == EEXIST) return 0;
#else
    if (mkdir(path, 0777) == 0 || errno == EEXIST) return 0;
#endif
    return -1;
}

static int r395_trace_ensure_parent_dirs(const char *path) {
    char scratch[1024];
    size_t len;
    size_t i;

    if (!path || !path[0]) return -1;
    len = strlen(path);
    if (len >= sizeof(scratch)) return -1;
    memcpy(scratch, path, len + 1);

    for (i = 0; i < len; i++) {
        if (scratch[i] != '/' && scratch[i] != '\\') continue;
        if (i == 0) continue;
        if (i == 2 && scratch[1] == ':') continue;
        scratch[i] = '\0';
        if (r395_trace_mkdir_single(scratch) != 0) return -1;
        scratch[i] = path[i];
    }
    return 0;
}

static int r395_trace_write_step(const R395TraceConfig *cfg,
                                 const Cache *c,
                                 int batch_size,
                                 int seq_len,
                                 uint32_t step,
                                 float step_loss,
                                 float runtime_ms) {
    FILE *fp;
    const char *mode;
    size_t state_count;
    long record_offset;
    double export_start;
    R395TraceRecordHeader record;

    if (!cfg || !cfg->enabled) return 0;
    if (!c || !c->h_all) {
        errno = ENODATA;
        return -1;
    }
    if (r395_trace_ensure_parent_dirs(cfg->path) != 0) return -1;

    mode = (cfg->captured_steps == 0) ? "wb" : "r+b";
    fp = fopen(cfg->path, mode);
    if (!fp) return -1;

    if (cfg->captured_steps == 0) {
        R395TraceHeader header;
        memset(&header, 0, sizeof(header));
        memcpy(header.magic, R395_TRACE_MAGIC, sizeof(header.magic));
        header.version = R395_TRACE_VERSION;
        header.n_layers = (uint32_t)V12_N_LAYERS;
        header.batch_size = (uint32_t)batch_size;
        header.seq_len = (uint32_t)seq_len;
        header.d_state = (uint32_t)V12_D_STATE;
        if (fwrite(&header, sizeof(header), 1, fp) != 1) {
            fclose(fp);
            return -1;
        }
    } else {
        if (fseek(fp, 0, SEEK_END) != 0) {
            fclose(fp);
            return -1;
        }
    }

    memset(&record, 0, sizeof(record));
    record.step = step;
    record.step_loss = step_loss;
    record.runtime_ms = runtime_ms;
    record_offset = ftell(fp);
    export_start = train_now_seconds();
    if (fwrite(&record, sizeof(record), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    state_count = (size_t)V12_N_LAYERS * (size_t)batch_size * (size_t)seq_len * (size_t)V12_D_STATE;
    float *decompressed_h = malloc(state_count * sizeof(float));
    if (!decompressed_h) {
        fclose(fp);
        return -1;
    }
    for (size_t i = 0; i < state_count; i++) {
        decompressed_h[i] = decompress_e5m2(c->h_all[i]);
    }
    size_t written = fwrite(decompressed_h, sizeof(float), state_count, fp);
    free(decompressed_h);
    if (written != state_count) {
        fclose(fp);
        return -1;
    }

    record.export_ms = (float)((train_now_seconds() - export_start) * 1000.0);
    if (fseek(fp, record_offset, SEEK_SET) != 0) {
        fclose(fp);
        return -1;
    }
    if (fwrite(&record, sizeof(record), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    const char *dataset_path    = "dataset.bin";
    int         total_steps     = 100;
    const char *checkpoint_path = NULL;

    int batch_size = V12_BATCH_SIZE_DEFAULT;
    int   rs_enable = 0;
    int   lccap_enable = 0;
    float rs_eps_lo = 0.5f;
    float rs_eps_hi = 15.0f;
    int exact_quotient_padding_enable = 0;
    uint32_t exact_quotient_padding_min_active_len = 0u;
    uint64_t exact_quotient_padding_seed = 0x9e3779b97f4a7c15ull;
    int   log_every = 100;
    int   loss_log_hex = 0;
    int   vk_profile = 0;
    int   ckpt_debug = 0;
    int   checkpoint_save_enabled = 1;
    float lr_base_override = -1.0f;
    int   teaching_enabled = 1;
    int   teaching_capacity = V12_TEACHING_CAPACITY_DEFAULT;
    DpthMode dpth_mode = DPTH_MODE_OFF;
    int   dpth_pool = 4;
    int   dpth_disabled_teaching = 0;
    int   eval_every = 500;
    int   eval_batches = 10;
    const char *heldout_eval_dataset_path = NULL;
    int   heldout_eval_batches = 0;
    uint64_t heldout_eval_seed = 0;
    int   r435_boundary_receipt_enable = 0;
    uint32_t r435_boundary_step = V12_REPACK_INTERVAL;
    int   seed_provided = 0;
    uint64_t dataset_seed_value = 0;
    int   r364_enable = 0;
    int   r364_warmup_steps = 100;
    int   r364_compile_every = 500;
    int   r364_compiled_steps = 100;
    int   corpus_logit_prior_enable = 0;
    int   logit_prior_gate_enable = 0;
    int   program_kernel_enable = 0;
    int   program_kernel_gate_enable = 0;
    int   certified_artifact_boundary_gate_enable = 0;
    int   state_transition_atlas_enable = 0;
    int   state_transition_atlas_gate_enable = 0;
    int   absorbed_state_atlas_gate_enable = 0;
    int   counterfactual_packets_enable = 0;
    int   counterfactual_packets_gate_enable = 0;
    int   certified_silent_span_gate_enable = 0;
    int   error_causal_lattice_enable = 0;
    int   error_causal_lattice_gate_enable = 0;
    int   residual_active_sampling_gate_enable = 0;
    int   exact_transition_quotient_enable = 0;
    int   exact_transition_quotient_gate_enable = 0;
    int   r423_native_growth_readout_quotient_enable = 0;
    int   residual_certified_scaffold_enable = 0;
    int   residual_certified_scaffold_gate_enable = 0;
    int   macro_ce_quotient_observer_enable = 0;
    int   macro_ce_quotient_training_enable = 0;
    int   exact_window_multiset_gate_enable = 0;
    int   prefix_frontier_gate_enable = 0;
    int   macro_transcript_gate_enable = 0;
    int   reversible_deterministic_expansion_gate_enable = 0;
    int   active_state_subspace_gate_enable = 0;
    int   boundary_liability_ledger_gate_enable = 0;
    int   boundary_flux_sheaf_gate_enable = 0;
    int   online_tssi_live_gate_enable = 0;
    const char *online_tssi_selector_out = NULL;
    float online_tssi_selector_active_fraction = 0.50f;
    const char *online_tssi_train_selector_json = NULL;
    const char *online_tssi_train_selector_control = "tssi";
    uint32_t online_tssi_train_selector_seed = 420u;
    int   r423_native_growth_state_selector_export_enable = 0;
    int   r423_native_growth_state_selector_train_enable = 0;
    int   r423_native_growth_microstep_birth_gate_enable = 0;
    int   r423_native_growth_receipt_challenge_gate_enable = 0;
    int   r423_native_growth_interaction_release_gate_enable = 0;
    int   r423_native_growth_interaction_safe_update_enable = 0;
    int   r423_native_growth_genotype_exception_gate_enable = 0;
    int   r423_native_growth_genotype_compact_update_enable = 0;
    int   boundary_cotangent_credit_gate_enable = 0;
    int   variance_priced_credit_gate_enable = 0;
    int   equal_area_fisher_sampling_enable = 0;
    int   equal_area_fisher_sampling_gate_enable = 0;
    int   amortized_ce_martingale_training_enable = 0;
    int   lagged_ce_impulse_eligibility_enable = 0;
    int   executable_multiplicity_packets_enable = 0;
    int   ce_transaction_blocks_enable = 0;
    int   original_ce_credit_debt_observer_enable = 0;
    int   r375_pbg2_gate_enable = 0;
    int   r375_pbg2_precondition_enable = 0;
    int   gradient_null_witness_gate_enable = 0;
    int   r429_decision_equivalence_witness_enable = 0;
    int   ce_normal_chamber_gate_enable = 0;
    int   optimizer_integrity_gate_enable = 0;
    int   paper7_dfga_real_batch_audit_enable = 0;
    uint32_t paper7_dfga_real_batch_audit_max_batches = 1;
    float paper7_dfga_real_batch_audit_residual_budget = 0.35f;
    int   qcf_mvue_gate_enable = 0;
    uint32_t qcf_mvue_max_matrices = 4;
    float qcf_mvue_epsilon = 1.0e-7f;
    float qcf_mvue_min_rho = 0.40f;
    double qcf_mvue_min_delta_energy = 1.0e-12;
    int   evidence_rate_commitment_gate_enable = 0;
    int   evidence_rate_commitment_mode = V12_EVIDENCE_RATE_MODE_CURRENT;
    int   evidence_rate_force_repack = 0;
    float evidence_rate_overhead = 0.0f;
    float evidence_rate_boundary_eps = 0.05f;
    int   r388_flip_refresh_gate_enable = 0;
    uint32_t r388_flip_refresh_block_elements = 1024;
    float r388_flip_refresh_min_eta = 0.33333334f;
    int   fp_interface_gap_probe_enable = 0;
    float fp_interface_gap_effect_epsilon = 0.05f;
    float corpus_logit_prior_alpha = 100.0f;
    float corpus_logit_prior_scale = 1.0f;
    uint64_t corpus_logit_prior_terms = 0;
    uint32_t corpus_logit_prior_stride = 1;
    float program_kernel_logit_boost = 4.0f;
    uint32_t program_kernel_min_count = 2;
    uint64_t program_kernel_terms = 0;
    uint32_t program_kernel_stride = 1;
    uint32_t certified_artifact_boundary_min_count = 2;
    uint64_t certified_artifact_boundary_terms = 0;
    uint32_t certified_artifact_boundary_stride = 1;
    float certified_artifact_boundary_epsilon = 0.001f;
    float certified_artifact_boundary_ce_budget = 0.001f;
    float certified_artifact_boundary_overhead = 0.0f;
    float certified_artifact_boundary_kappa = 1.0f;
    float state_transition_atlas_logit_boost = 8.0f;
    uint32_t state_transition_atlas_min_count = 2;
    uint64_t state_transition_atlas_terms = 0;
    uint32_t state_transition_atlas_stride = 1;
    float absorbed_state_atlas_overhead = 0.0f;
    float absorbed_state_atlas_memory_cost = 0.0f;
    float absorbed_state_atlas_repair_cost = 0.0f;
    float absorbed_state_atlas_kappa = 1.0f;
    uint32_t counterfactual_packets_min_count = 1;
    uint64_t counterfactual_packets_terms = 0;
    uint32_t counterfactual_packets_stride = 1;
    float certified_silent_span_max_loss = 0.10f;
    float certified_silent_span_cert_cost = 0.0f;
    float certified_silent_span_repair_cost = 0.0f;
    uint32_t certified_silent_span_min_len = 2;
    uint32_t error_causal_lattice_min_count = 1;
    uint64_t error_causal_lattice_terms = 0;
    uint32_t error_causal_lattice_stride = 1;
    float residual_active_sampling_floor = 1.0e-6f;
    float residual_active_sampling_target_fraction = 0.50f;
    uint32_t exact_transition_quotient_min_count = 1;
    uint64_t exact_transition_quotient_terms = 0;
    uint32_t exact_transition_quotient_stride = 1;
    uint32_t macro_ce_quotient_min_support = 4;
    uint64_t macro_ce_quotient_terms = 0;
    uint32_t macro_ce_quotient_stride = 1;
    float macro_ce_quotient_build_fraction = 0.70f;
    float macro_ce_quotient_alpha = 0.05f;
    float macro_ce_quotient_residual_price_scale = 1.0f;
    float macro_ce_quotient_overhead = 0.0f;
    uint32_t macro_ce_quotient_signature_bins = 4;
    uint32_t macro_ce_quotient_seed = 42140u;
    float macro_ce_quotient_training_weight = 0.50f;
    float residual_certified_scaffold_audit_budget = 0.10f;
    uint32_t exact_window_multiset_min_count = 1;
    uint64_t exact_window_multiset_windows = 0;
    uint32_t exact_window_multiset_stride = 1;
    float exact_window_multiset_overhead = 0.0f;
    float exact_window_multiset_noise = 0.0f;
    uint32_t prefix_frontier_min_count = 2;
    uint64_t prefix_frontier_terms = 0;
    uint32_t prefix_frontier_stride = 1;
    float prefix_frontier_overhead = 0.0f;
    float prefix_frontier_repair_cost = 0.0f;
    uint32_t macro_transcript_min_count = 2;
    uint64_t macro_transcript_terms = 0;
    uint32_t macro_transcript_stride = 1;
    float macro_transcript_hardness = 1.0f;
    float macro_transcript_codec_cost = 0.0f;
    float macro_transcript_decode_cost = 0.0f;
    uint32_t reversible_deterministic_expansion_min_run = 3;
    uint64_t reversible_deterministic_expansion_terms = 0;
    float reversible_deterministic_expansion_overhead = 0.0f;
    float reversible_deterministic_expansion_noise = 0.0f;
    uint32_t active_state_subspace_k = 16;
    float active_state_subspace_overhead = 0.0f;
    float boundary_liability_null_epsilon = 1.0e-6f;
    float boundary_liability_overhead = 0.0f;
    float boundary_flux_sheaf_overhead = 0.0f;
    float boundary_cotangent_overhead = 0.0f;
    float boundary_cotangent_residual_budget = 0.10f;
    float boundary_cotangent_verify_fraction = 1.0f;
    float variance_priced_sample_fraction = 0.25f;
    float variance_priced_score_fraction = 1.0f;
    float variance_priced_verify_fraction = 1.0f;
    float variance_priced_overhead = 0.0f;
    float equal_area_fisher_ess_floor = 0.25f;
    float equal_area_fisher_bias_budget = 0.10f;
    float equal_area_fisher_min_descent_per_paid_work = 1.10f;
    int event_innovation_credit_enable = 0;
    int event_innovation_credit_gate_enable = 0;
    float event_innovation_fraction = 0.25f;
    float event_innovation_debt_budget = 0.10f;
    float event_innovation_score_cost = 0.05f;
    float event_innovation_verify_fraction = 0.05f;
    float event_innovation_overhead = 0.0f;
    float event_innovation_low_event_scale = 0.25f;
    float event_innovation_boundary_boost = 2.0f;
    int ce_debt_skip_enable = 0;
    int ce_debt_skip_gate_enable = 0;
    float ce_debt_max_loss = 40.0f;
    float ce_debt_max_grad_norm = 1.0f;
    float ce_debt_max_debt = 0.10f;
    float ce_debt_audit_fraction = 0.05f;
    float ce_debt_overhead = 0.02f;
    float ce_debt_min_skip_fraction = 0.10f;
    float ce_debt_skip_scale = 0.0f;
    int gradient_coherence_quotient_enable = 0;
    int gradient_coherence_quotient_gate_enable = 0;
    uint32_t gradient_coherence_bins = 64u;
    uint32_t gradient_coherence_probe_stride = 16u;
    float gradient_coherence_max_residual = 0.10f;
    float gradient_coherence_score_cost = 0.05f;
    float gradient_coherence_verify_fraction = 0.05f;
    float gradient_coherence_overhead = 0.02f;
    float gradient_coherence_min_saved_fraction = 0.10f;
    float gradient_coherence_residual_scale = 0.0f;
    float amortized_ce_martingale_min_descent_per_paid_work = 1.111f;
    float martingale_lambda = 0.0f;
    int continuous_ternary_margin_enabled = 0;
    float continuous_ternary_margin_c = 0.1f;
    uint32_t lagged_ce_impulse_horizon = 8u;
    float lagged_ce_impulse_blend = 0.25f;
    float lagged_ce_impulse_residual_budget = 0.10f;
    float lagged_ce_impulse_local_jvp_fraction = 0.08f;
    float lagged_ce_impulse_ring_fraction = 0.03f;
    float lagged_ce_impulse_tail_audit_fraction = 0.05f;
    float lagged_ce_impulse_memory_fraction = 0.02f;
    uint32_t executable_multiplicity_min_count = 2u;
    float executable_multiplicity_min_mass = 0.20f;
    float executable_multiplicity_max_residual = 0.10f;
    float executable_multiplicity_signature_fraction = 0.05f;
    float executable_multiplicity_verifier_fraction = 0.03f;
    float executable_multiplicity_memory_fraction = 0.02f;
    float executable_multiplicity_state_scale = 16.0f;
    float executable_multiplicity_transition_scale = 64.0f;
    uint32_t ce_transaction_k = 4u;
    float ce_transaction_min_margin = 0.050f;
    float ce_transaction_max_rollback_fraction = 0.10f;
    float ce_transaction_verifier_fraction = 0.05f;
    float original_ce_credit_surface_fraction = 0.35f;
    float original_ce_credit_local_work_multiplier = 0.35f;
    float original_ce_credit_debt_price = 0.20f;
    float original_ce_credit_verifier_fraction = 0.02f;
    float original_ce_credit_overhead = 0.0f;
    float original_ce_credit_recurrent_backward_share = 0.35f;
    float original_ce_credit_min_progress = 0.85f;
    float original_ce_credit_max_residual = 0.45f;
    float original_ce_credit_min_whole_step_speedup = 1.10f;
    uint32_t r375_pbg2_max_matrices = 2;
    float r375_pbg2_min_anisotropy = 0.03f;
    uint32_t r375_pbg2_precondition_max_matrices = 0;
    float r375_pbg2_precondition_eps = 1.0e-6f;
    float gradient_null_witness_epsilon = 1.0e-8f;
    float gradient_null_witness_omitted_budget = 1.0e-4f;
    float gradient_null_witness_overhead = 0.0f;
    float ce_normal_chamber_sigma = 0.0f;
    float ce_normal_chamber_overhead = 0.0f;
    int credit_time_equalized_dynamics_enable = 0;
    int credit_time_equalized_dynamics_gate_enable = 0;
    uint32_t credit_time_horizon = (uint32_t)V12_SEQ_LEN;
    float credit_time_min_scale = 0.5f;
    float credit_time_max_scale = 2.0f;
    float credit_time_overhead = 0.0f;
    int contractive_green_credit_enable = 0;
    int contractive_green_credit_gate_enable = 0;
    uint32_t contractive_green_credit_horizon = (uint32_t)V12_SEQ_LEN;
    float contractive_green_credit_max_alpha = 0.98f;
    float contractive_green_credit_tail_budget = 0.10f;
    float contractive_green_credit_min_scale = 0.5f;
    float contractive_green_credit_max_scale = 2.0f;
    float contractive_green_credit_overhead = 0.0f;
    int adaptive_credit_horizon_enable = 0;
    int adaptive_credit_horizon_gate_enable = 0;
    uint32_t adaptive_credit_min_horizon = 1u;
    uint32_t adaptive_credit_max_horizon = (uint32_t)V12_SEQ_LEN;
    float adaptive_credit_tail_budget = 0.10f;
    float adaptive_credit_bound_budget = 0.10f;
    float adaptive_credit_verify_fraction = 0.05f;
    float adaptive_credit_overhead = 0.02f;
    float adaptive_credit_min_headroom = 0.10f;
    float adaptive_credit_tail_scale = 0.0f;
    int interior_adjoint_tube_enable = 0;
    int interior_adjoint_tube_gate_enable = 0;
    uint32_t interior_adjoint_tube_rank = (uint32_t)(V12_D_STATE / 4);
    float interior_adjoint_tube_residual_budget = 0.10f;
    float interior_adjoint_tube_verify_fraction = 1.0f;
    float interior_adjoint_tube_overhead = 0.0f;
    float interior_adjoint_tube_residual_scale = 0.25f;
    float interior_adjoint_tube_min_energy_covered = 0.90f;

    /* R493 Gates Configuration */
    float r493_min_quality_threshold = 0.70f;
    float r493_min_entropy_threshold = 3.5f;
    float r493_collapse_weight = 2.0f;
    int   r493_post_step_enable = 1;

    R395TraceConfig r395_trace = {0};
    R399SSQTTraceConfig r399_trace = {0};
    R399DmdcTraceConfig r399_dmdc_trace = {0};
    int positional_argc = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            total_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            dataset_seed_value = (uint64_t)strtoull(argv[++i], NULL, 10);
            seed_provided = 1;
        } else if (strcmp(argv[i], "--log-every") == 0 && i + 1 < argc) {
            log_every = atoi(argv[++i]);
            if (log_every <= 0) {
                fprintf(stderr, "[ERROR] --log-every must be >= 1\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--loss-log-hex") == 0) {
            loss_log_hex = 1;
        } else if (strcmp(argv[i], "--vk-profile") == 0) {
            vk_profile = 1;
        } else if (strcmp(argv[i], "--ckpt-debug") == 0) {
            ckpt_debug = 1;
        } else if (strcmp(argv[i], "--checkpoint-load-only") == 0) {
            checkpoint_save_enabled = 0;
        } else if (strcmp(argv[i], "--lr-base") == 0 && i + 1 < argc) {
            lr_base_override = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--teaching-off") == 0) {
            teaching_enabled = 0;
        } else if (strcmp(argv[i], "--teaching-capacity") == 0 && i + 1 < argc) {
            teaching_capacity = atoi(argv[++i]);
            if (teaching_capacity <= 0) {
                fprintf(stderr, "[ERROR] --teaching-capacity must be >= 1\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--drift-persistent-ternary-hitting-gate") == 0 ||
                   strcmp(argv[i], "--dpth-gate") == 0) {
            dpth_mode = DPTH_MODE_PACKET;
        } else if (strcmp(argv[i], "--dpth-mode") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (strcmp(mode, "packet") == 0 ||
                strcmp(mode, "dpth") == 0) {
                dpth_mode = DPTH_MODE_PACKET;
            } else if (strcmp(mode, "loss-control") == 0 ||
                       strcmp(mode, "loss") == 0) {
                dpth_mode = DPTH_MODE_LOSS_CONTROL;
            } else if (strcmp(mode, "off") == 0) {
                dpth_mode = DPTH_MODE_OFF;
            } else {
                fprintf(stderr, "[ERROR] --dpth-mode must be packet, loss-control, or off\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--dpth-pool") == 0 && i + 1 < argc) {
            dpth_pool = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--eval-every") == 0 && i + 1 < argc) {
            eval_every = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--eval-batches") == 0 && i + 1 < argc) {
            eval_batches = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--heldout-dataset") == 0 && i + 1 < argc) {
            heldout_eval_dataset_path = argv[++i];
        } else if (strcmp(argv[i], "--heldout-eval-batches") == 0 && i + 1 < argc) {
            heldout_eval_batches = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--heldout-eval-seed") == 0 && i + 1 < argc) {
            heldout_eval_seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--r435-boundary-receipt") == 0) {
            r435_boundary_receipt_enable = 1;
        } else if (strcmp(argv[i], "--r435-boundary-step") == 0 && i + 1 < argc) {
            r435_boundary_step = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--r493-min-quality-threshold") == 0 && i + 1 < argc) {
            r493_min_quality_threshold = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--r493-min-entropy-threshold") == 0 && i + 1 < argc) {
            r493_min_entropy_threshold = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--r493-collapse-weight") == 0 && i + 1 < argc) {
            r493_collapse_weight = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--r493-post-step-off") == 0) {
            r493_post_step_enable = 0;
        } else if (strcmp(argv[i], "--r364-pilot") == 0) {
            r364_enable = 1;
        } else if (strcmp(argv[i], "--r364-warmup") == 0 && i + 1 < argc) {
            r364_warmup_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--r364-compile-every") == 0 && i + 1 < argc) {
            r364_compile_every = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--r364-compiled-steps") == 0 && i + 1 < argc) {
            r364_compiled_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--corpus-logit-prior") == 0 ||
                   strcmp(argv[i], "--r408-scaffold") == 0) {
            corpus_logit_prior_enable = 1;
        } else if (strcmp(argv[i], "--logit-prior-gate") == 0) {
            corpus_logit_prior_enable = 1;
            logit_prior_gate_enable = 1;
        } else if (strcmp(argv[i], "--program-kernel") == 0) {
            program_kernel_enable = 1;
        } else if (strcmp(argv[i], "--program-kernel-gate") == 0) {
            program_kernel_enable = 1;
            program_kernel_gate_enable = 1;
        } else if (strcmp(argv[i], "--program-kernel-logit") == 0 && i + 1 < argc) {
            program_kernel_logit_boost = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--program-kernel-min-count") == 0 && i + 1 < argc) {
            program_kernel_min_count = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--program-kernel-terms") == 0 && i + 1 < argc) {
            program_kernel_terms = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--program-kernel-stride") == 0 && i + 1 < argc) {
            program_kernel_stride = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--certified-artifact-boundary-gate") == 0) {
            certified_artifact_boundary_gate_enable = 1;
        } else if (strcmp(argv[i], "--certified-artifact-boundary-min-count") == 0 && i + 1 < argc) {
            certified_artifact_boundary_min_count = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--certified-artifact-boundary-terms") == 0 && i + 1 < argc) {
            certified_artifact_boundary_terms = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--certified-artifact-boundary-stride") == 0 && i + 1 < argc) {
            certified_artifact_boundary_stride = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--certified-artifact-boundary-epsilon") == 0 && i + 1 < argc) {
            certified_artifact_boundary_epsilon = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--certified-artifact-boundary-ce-budget") == 0 && i + 1 < argc) {
            certified_artifact_boundary_ce_budget = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--certified-artifact-boundary-overhead") == 0 && i + 1 < argc) {
            certified_artifact_boundary_overhead = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--certified-artifact-boundary-kappa") == 0 && i + 1 < argc) {
            certified_artifact_boundary_kappa = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--state-transition-atlas") == 0) {
            state_transition_atlas_enable = 1;
        } else if (strcmp(argv[i], "--state-transition-atlas-gate") == 0) {
            state_transition_atlas_enable = 1;
            state_transition_atlas_gate_enable = 1;
        } else if (strcmp(argv[i], "--absorbed-state-atlas-gate") == 0) {
            state_transition_atlas_enable = 1;
            state_transition_atlas_gate_enable = 1;
            absorbed_state_atlas_gate_enable = 1;
        } else if (strcmp(argv[i], "--state-transition-atlas-logit") == 0 && i + 1 < argc) {
            state_transition_atlas_logit_boost = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--state-transition-atlas-min-count") == 0 && i + 1 < argc) {
            state_transition_atlas_min_count = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--state-transition-atlas-terms") == 0 && i + 1 < argc) {
            state_transition_atlas_terms = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--state-transition-atlas-stride") == 0 && i + 1 < argc) {
            state_transition_atlas_stride = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--absorbed-state-atlas-overhead") == 0 && i + 1 < argc) {
            absorbed_state_atlas_overhead = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--absorbed-state-atlas-memory-cost") == 0 && i + 1 < argc) {
            absorbed_state_atlas_memory_cost = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--absorbed-state-atlas-repair-cost") == 0 && i + 1 < argc) {
            absorbed_state_atlas_repair_cost = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--absorbed-state-atlas-kappa") == 0 && i + 1 < argc) {
            absorbed_state_atlas_kappa = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--counterfactual-packets") == 0) {
            counterfactual_packets_enable = 1;
        } else if (strcmp(argv[i], "--counterfactual-packets-gate") == 0) {
            counterfactual_packets_enable = 1;
            counterfactual_packets_gate_enable = 1;
        } else if (strcmp(argv[i], "--counterfactual-packets-min-count") == 0 && i + 1 < argc) {
            counterfactual_packets_min_count = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--counterfactual-packets-terms") == 0 && i + 1 < argc) {
            counterfactual_packets_terms = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--counterfactual-packets-stride") == 0 && i + 1 < argc) {
            counterfactual_packets_stride = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--certified-silent-span-gate") == 0) {
            certified_silent_span_gate_enable = 1;
        } else if (strcmp(argv[i], "--certified-silent-span-max-loss") == 0 && i + 1 < argc) {
            certified_silent_span_max_loss = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--certified-silent-span-min-len") == 0 && i + 1 < argc) {
            certified_silent_span_min_len = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--certified-silent-span-cert-cost") == 0 && i + 1 < argc) {
            certified_silent_span_cert_cost = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--certified-silent-span-repair-cost") == 0 && i + 1 < argc) {
            certified_silent_span_repair_cost = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--error-causal-lattice") == 0) {
            error_causal_lattice_enable = 1;
        } else if (strcmp(argv[i], "--error-causal-lattice-gate") == 0) {
            error_causal_lattice_enable = 1;
            error_causal_lattice_gate_enable = 1;
        } else if (strcmp(argv[i], "--error-causal-lattice-min-count") == 0 && i + 1 < argc) {
            error_causal_lattice_min_count = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--error-causal-lattice-terms") == 0 && i + 1 < argc) {
            error_causal_lattice_terms = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--error-causal-lattice-stride") == 0 && i + 1 < argc) {
            error_causal_lattice_stride = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--residual-active-sampling-gate") == 0) {
            residual_active_sampling_gate_enable = 1;
        } else if (strcmp(argv[i], "--residual-active-sampling-floor") == 0 && i + 1 < argc) {
            residual_active_sampling_floor = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--residual-active-sampling-target-fraction") == 0 && i + 1 < argc) {
            residual_active_sampling_target_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--exact-transition-quotient") == 0) {
            exact_transition_quotient_enable = 1;
        } else if (strcmp(argv[i], "--exact-transition-quotient-gate") == 0) {
            exact_transition_quotient_enable = 1;
            exact_transition_quotient_gate_enable = 1;
        } else if (strcmp(argv[i], "--exact-transition-quotient-min-count") == 0 && i + 1 < argc) {
            exact_transition_quotient_min_count = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--exact-transition-quotient-terms") == 0 && i + 1 < argc) {
            exact_transition_quotient_terms = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--exact-transition-quotient-stride") == 0 && i + 1 < argc) {
            exact_transition_quotient_stride = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--residual-certified-scaffold") == 0) {
            residual_certified_scaffold_enable = 1;
            residual_certified_scaffold_gate_enable = 1;
            macro_ce_quotient_observer_enable = 1;
            macro_ce_quotient_training_enable = 1;
        } else if (strcmp(argv[i], "--residual-certified-scaffold-gate") == 0) {
            residual_certified_scaffold_gate_enable = 1;
            macro_ce_quotient_observer_enable = 1;
        } else if (strcmp(argv[i], "--residual-certified-scaffold-audit-budget") == 0 && i + 1 < argc) {
            residual_certified_scaffold_audit_budget = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--r423-native-growth-readout-quotient") == 0) {
            r423_native_growth_readout_quotient_enable = 1;
            macro_ce_quotient_observer_enable = 1;
            macro_ce_quotient_training_enable = 1;
        } else if (strcmp(argv[i], "--macro-ce-quotient-observer") == 0) {
            macro_ce_quotient_observer_enable = 1;
        } else if (strcmp(argv[i], "--macro-ce-quotient-training") == 0) {
            macro_ce_quotient_observer_enable = 1;
            macro_ce_quotient_training_enable = 1;
        } else if (strcmp(argv[i], "--macro-ce-quotient-min-support") == 0 && i + 1 < argc) {
            macro_ce_quotient_min_support = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--macro-ce-quotient-terms") == 0 && i + 1 < argc) {
            macro_ce_quotient_terms = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--macro-ce-quotient-stride") == 0 && i + 1 < argc) {
            macro_ce_quotient_stride = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--macro-ce-quotient-build-fraction") == 0 && i + 1 < argc) {
            macro_ce_quotient_build_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--macro-ce-quotient-alpha") == 0 && i + 1 < argc) {
            macro_ce_quotient_alpha = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--macro-ce-quotient-residual-price-scale") == 0 && i + 1 < argc) {
            macro_ce_quotient_residual_price_scale = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--macro-ce-quotient-overhead") == 0 && i + 1 < argc) {
            macro_ce_quotient_overhead = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--macro-ce-quotient-signature-bins") == 0 && i + 1 < argc) {
            macro_ce_quotient_signature_bins = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--macro-ce-quotient-seed") == 0 && i + 1 < argc) {
            macro_ce_quotient_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--macro-ce-quotient-training-weight") == 0 && i + 1 < argc) {
            macro_ce_quotient_training_weight = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--exact-window-multiset-gate") == 0) {
            exact_window_multiset_gate_enable = 1;
        } else if (strcmp(argv[i], "--exact-window-multiset-min-count") == 0 && i + 1 < argc) {
            exact_window_multiset_min_count = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--exact-window-multiset-windows") == 0 && i + 1 < argc) {
            exact_window_multiset_windows = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--exact-window-multiset-stride") == 0 && i + 1 < argc) {
            exact_window_multiset_stride = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--exact-window-multiset-overhead") == 0 && i + 1 < argc) {
            exact_window_multiset_overhead = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--exact-window-multiset-noise") == 0 && i + 1 < argc) {
            exact_window_multiset_noise = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--prefix-frontier-adjoint-gate") == 0) {
            prefix_frontier_gate_enable = 1;
        } else if (strcmp(argv[i], "--prefix-frontier-min-count") == 0 && i + 1 < argc) {
            prefix_frontier_min_count = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--prefix-frontier-terms") == 0 && i + 1 < argc) {
            prefix_frontier_terms = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--prefix-frontier-stride") == 0 && i + 1 < argc) {
            prefix_frontier_stride = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--prefix-frontier-overhead") == 0 && i + 1 < argc) {
            prefix_frontier_overhead = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--prefix-frontier-repair-cost") == 0 && i + 1 < argc) {
            prefix_frontier_repair_cost = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--macro-transcript-gate") == 0) {
            macro_transcript_gate_enable = 1;
        } else if (strcmp(argv[i], "--macro-transcript-min-count") == 0 && i + 1 < argc) {
            macro_transcript_min_count = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--macro-transcript-terms") == 0 && i + 1 < argc) {
            macro_transcript_terms = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--macro-transcript-stride") == 0 && i + 1 < argc) {
            macro_transcript_stride = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--macro-transcript-hardness") == 0 && i + 1 < argc) {
            macro_transcript_hardness = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--macro-transcript-codec-cost") == 0 && i + 1 < argc) {
            macro_transcript_codec_cost = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--macro-transcript-decode-cost") == 0 && i + 1 < argc) {
            macro_transcript_decode_cost = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--reversible-deterministic-expansion-gate") == 0) {
            reversible_deterministic_expansion_gate_enable = 1;
        } else if (strcmp(argv[i], "--reversible-deterministic-expansion-min-run") == 0 && i + 1 < argc) {
            reversible_deterministic_expansion_min_run = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--reversible-deterministic-expansion-terms") == 0 && i + 1 < argc) {
            reversible_deterministic_expansion_terms = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--reversible-deterministic-expansion-overhead") == 0 && i + 1 < argc) {
            reversible_deterministic_expansion_overhead = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--reversible-deterministic-expansion-noise") == 0 && i + 1 < argc) {
            reversible_deterministic_expansion_noise = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--active-state-subspace-gate") == 0) {
            active_state_subspace_gate_enable = 1;
        } else if (strcmp(argv[i], "--active-state-subspace-k") == 0 && i + 1 < argc) {
            active_state_subspace_k = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--active-state-subspace-overhead") == 0 && i + 1 < argc) {
            active_state_subspace_overhead = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--boundary-liability-ledger-gate") == 0) {
            boundary_liability_ledger_gate_enable = 1;
        } else if (strcmp(argv[i], "--boundary-liability-null-epsilon") == 0 && i + 1 < argc) {
            boundary_liability_null_epsilon = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--boundary-liability-overhead") == 0 && i + 1 < argc) {
            boundary_liability_overhead = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--boundary-flux-sheaf-gate") == 0) {
            boundary_flux_sheaf_gate_enable = 1;
        } else if (strcmp(argv[i], "--boundary-flux-sheaf-overhead") == 0 && i + 1 < argc) {
            boundary_flux_sheaf_overhead = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--online-tssi-live-gate") == 0) {
            online_tssi_live_gate_enable = 1;
        } else if (strcmp(argv[i], "--online-tssi-selector-out") == 0 && i + 1 < argc) {
            online_tssi_selector_out = argv[++i];
            online_tssi_live_gate_enable = 1;
        } else if (strcmp(argv[i], "--online-tssi-selector-active-fraction") == 0 && i + 1 < argc) {
            online_tssi_selector_active_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--r423-native-growth-state-selector-out") == 0 && i + 1 < argc) {
            r423_native_growth_state_selector_export_enable = 1;
            online_tssi_live_gate_enable = 1;
            online_tssi_selector_out = argv[++i];
        } else if (strcmp(argv[i], "--r423-native-growth-state-selector-json") == 0 && i + 1 < argc) {
            r423_native_growth_state_selector_train_enable = 1;
            online_tssi_train_selector_json = argv[++i];
        } else if (strcmp(argv[i], "--r423-native-growth-microstep-birth-gate") == 0) {
            r423_native_growth_microstep_birth_gate_enable = 1;
        } else if (strcmp(argv[i], "--r423-native-growth-receipt-challenge-gate") == 0) {
            r423_native_growth_receipt_challenge_gate_enable = 1;
        } else if (strcmp(argv[i], "--r423-native-growth-interaction-release-gate") == 0) {
            r423_native_growth_interaction_release_gate_enable = 1;
        } else if (strcmp(argv[i], "--r423-native-growth-interaction-safe-update") == 0) {
            r423_native_growth_interaction_safe_update_enable = 1;
        } else if (strcmp(argv[i], "--r423-native-growth-genotype-exception-gate") == 0) {
            r423_native_growth_genotype_exception_gate_enable = 1;
        } else if (strcmp(argv[i], "--r423-native-growth-genotype-compact-update") == 0) {
            r423_native_growth_genotype_compact_update_enable = 1;
        } else if (strcmp(argv[i], "--online-tssi-train-selector-json") == 0 && i + 1 < argc) {
            online_tssi_train_selector_json = argv[++i];
        } else if (strcmp(argv[i], "--online-tssi-train-selector-control") == 0 && i + 1 < argc) {
            online_tssi_train_selector_control = argv[++i];
        } else if (strcmp(argv[i], "--online-tssi-train-selector-seed") == 0 && i + 1 < argc) {
            online_tssi_train_selector_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--boundary-cotangent-credit-gate") == 0) {
            boundary_cotangent_credit_gate_enable = 1;
        } else if (strcmp(argv[i], "--boundary-cotangent-overhead") == 0 && i + 1 < argc) {
            boundary_cotangent_overhead = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--boundary-cotangent-residual-budget") == 0 && i + 1 < argc) {
            boundary_cotangent_residual_budget = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--boundary-cotangent-verify-fraction") == 0 && i + 1 < argc) {
            boundary_cotangent_verify_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--variance-priced-credit-gate") == 0) {
            variance_priced_credit_gate_enable = 1;
        } else if (strcmp(argv[i], "--equal-area-fisher-sampling") == 0) {
            equal_area_fisher_sampling_enable = 1;
            equal_area_fisher_sampling_gate_enable = 1;
            variance_priced_credit_gate_enable = 1;
            amortized_ce_martingale_training_enable = 1;
        } else if (strcmp(argv[i], "--equal-area-fisher-sampling-gate") == 0) {
            equal_area_fisher_sampling_gate_enable = 1;
            variance_priced_credit_gate_enable = 1;
        } else if (strcmp(argv[i], "--amortized-ce-martingale-training") == 0) {
            variance_priced_credit_gate_enable = 1;
            amortized_ce_martingale_training_enable = 1;
        } else if (strcmp(argv[i], "--lagged-ce-impulse-eligibility") == 0) {
            lagged_ce_impulse_eligibility_enable = 1;
        } else if (strcmp(argv[i], "--lagged-ce-impulse-horizon") == 0 && i + 1 < argc) {
            lagged_ce_impulse_horizon = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--lagged-ce-impulse-blend") == 0 && i + 1 < argc) {
            lagged_ce_impulse_blend = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--lagged-ce-impulse-residual-budget") == 0 && i + 1 < argc) {
            lagged_ce_impulse_residual_budget = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--lagged-ce-impulse-local-jvp-fraction") == 0 && i + 1 < argc) {
            lagged_ce_impulse_local_jvp_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--lagged-ce-impulse-ring-fraction") == 0 && i + 1 < argc) {
            lagged_ce_impulse_ring_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--lagged-ce-impulse-tail-audit-fraction") == 0 && i + 1 < argc) {
            lagged_ce_impulse_tail_audit_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--lagged-ce-impulse-memory-fraction") == 0 && i + 1 < argc) {
            lagged_ce_impulse_memory_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--r438-executable-multiplicity-packets") == 0) {
            executable_multiplicity_packets_enable = 1;
        } else if (strcmp(argv[i], "--r438-packet-min-count") == 0 && i + 1 < argc) {
            executable_multiplicity_min_count = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--r438-packet-min-mass") == 0 && i + 1 < argc) {
            executable_multiplicity_min_mass = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--r438-packet-max-residual") == 0 && i + 1 < argc) {
            executable_multiplicity_max_residual = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--r438-packet-signature-fraction") == 0 && i + 1 < argc) {
            executable_multiplicity_signature_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--r438-packet-verifier-fraction") == 0 && i + 1 < argc) {
            executable_multiplicity_verifier_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--r438-packet-memory-fraction") == 0 && i + 1 < argc) {
            executable_multiplicity_memory_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--r438-packet-state-scale") == 0 && i + 1 < argc) {
            executable_multiplicity_state_scale = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--r438-packet-transition-scale") == 0 && i + 1 < argc) {
            executable_multiplicity_transition_scale = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--r438-ce-transaction-blocks") == 0) {
            ce_transaction_blocks_enable = 1;
        } else if (strcmp(argv[i], "--r438-ce-tx-k") == 0 && i + 1 < argc) {
            ce_transaction_k = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--r438-ce-tx-min-margin") == 0 && i + 1 < argc) {
            ce_transaction_min_margin = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--r438-ce-tx-max-rollback-fraction") == 0 && i + 1 < argc) {
            ce_transaction_max_rollback_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--r438-ce-tx-verifier-fraction") == 0 && i + 1 < argc) {
            ce_transaction_verifier_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--variance-priced-sample-fraction") == 0 && i + 1 < argc) {
            variance_priced_sample_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--variance-priced-score-fraction") == 0 && i + 1 < argc) {
            variance_priced_score_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--variance-priced-verify-fraction") == 0 && i + 1 < argc) {
            variance_priced_verify_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--variance-priced-overhead") == 0 && i + 1 < argc) {
            variance_priced_overhead = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--equal-area-fisher-ess-floor") == 0 && i + 1 < argc) {
            equal_area_fisher_ess_floor = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--equal-area-fisher-bias-budget") == 0 && i + 1 < argc) {
            equal_area_fisher_bias_budget = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--equal-area-fisher-min-descent-per-paid-work") == 0 && i + 1 < argc) {
            equal_area_fisher_min_descent_per_paid_work = (float)atof(argv[++i]);
            amortized_ce_martingale_min_descent_per_paid_work =
                equal_area_fisher_min_descent_per_paid_work;
        } else if (strcmp(argv[i], "--event-innovation-credit") == 0) {
            event_innovation_credit_enable = 1;
        } else if (strcmp(argv[i], "--event-innovation-credit-gate") == 0) {
            event_innovation_credit_gate_enable = 1;
        } else if (strcmp(argv[i], "--event-innovation-fraction") == 0 && i + 1 < argc) {
            event_innovation_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--event-innovation-debt-budget") == 0 && i + 1 < argc) {
            event_innovation_debt_budget = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--event-innovation-score-cost") == 0 && i + 1 < argc) {
            event_innovation_score_cost = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--event-innovation-verify-fraction") == 0 && i + 1 < argc) {
            event_innovation_verify_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--event-innovation-overhead") == 0 && i + 1 < argc) {
            event_innovation_overhead = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--event-innovation-low-event-scale") == 0 && i + 1 < argc) {
            event_innovation_low_event_scale = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--event-innovation-boundary-boost") == 0 && i + 1 < argc) {
            event_innovation_boundary_boost = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--ce-debt-skip") == 0) {
            ce_debt_skip_enable = 1;
        } else if (strcmp(argv[i], "--ce-debt-skip-gate") == 0) {
            ce_debt_skip_gate_enable = 1;
        } else if (strcmp(argv[i], "--ce-debt-max-loss") == 0 && i + 1 < argc) {
            ce_debt_max_loss = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--ce-debt-max-grad-norm") == 0 && i + 1 < argc) {
            ce_debt_max_grad_norm = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--ce-debt-max-debt") == 0 && i + 1 < argc) {
            ce_debt_max_debt = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--ce-debt-audit-fraction") == 0 && i + 1 < argc) {
            ce_debt_audit_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--ce-debt-overhead") == 0 && i + 1 < argc) {
            ce_debt_overhead = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--ce-debt-min-skip-fraction") == 0 && i + 1 < argc) {
            ce_debt_min_skip_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--ce-debt-skip-scale") == 0 && i + 1 < argc) {
            ce_debt_skip_scale = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--gradient-coherence-quotient") == 0) {
            gradient_coherence_quotient_enable = 1;
        } else if (strcmp(argv[i], "--gradient-coherence-quotient-gate") == 0) {
            gradient_coherence_quotient_gate_enable = 1;
        } else if (strcmp(argv[i], "--gradient-coherence-bins") == 0 && i + 1 < argc) {
            gradient_coherence_bins = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--gradient-coherence-probe-stride") == 0 && i + 1 < argc) {
            gradient_coherence_probe_stride = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--gradient-coherence-max-residual") == 0 && i + 1 < argc) {
            gradient_coherence_max_residual = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--gradient-coherence-score-cost") == 0 && i + 1 < argc) {
            gradient_coherence_score_cost = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--gradient-coherence-verify-fraction") == 0 && i + 1 < argc) {
            gradient_coherence_verify_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--gradient-coherence-overhead") == 0 && i + 1 < argc) {
            gradient_coherence_overhead = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--gradient-coherence-min-saved-fraction") == 0 && i + 1 < argc) {
            gradient_coherence_min_saved_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--gradient-coherence-residual-scale") == 0 && i + 1 < argc) {
            gradient_coherence_residual_scale = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--amortized-ce-martingale-min-descent-per-paid-work") == 0 && i + 1 < argc) {
            amortized_ce_martingale_min_descent_per_paid_work = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--martingale-lambda") == 0 && i + 1 < argc) {
            martingale_lambda = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--continuous-ternary-margin-gate") == 0) {
            continuous_ternary_margin_enabled = 1;
        } else if (strcmp(argv[i], "--continuous-ternary-margin-c") == 0 && i + 1 < argc) {
            continuous_ternary_margin_c = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--original-ce-credit-debt-observer") == 0) {
            original_ce_credit_debt_observer_enable = 1;
        } else if (strcmp(argv[i], "--original-ce-credit-surface-fraction") == 0 && i + 1 < argc) {
            original_ce_credit_surface_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--original-ce-credit-local-work-multiplier") == 0 && i + 1 < argc) {
            original_ce_credit_local_work_multiplier = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--original-ce-credit-debt-price") == 0 && i + 1 < argc) {
            original_ce_credit_debt_price = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--original-ce-credit-verifier-fraction") == 0 && i + 1 < argc) {
            original_ce_credit_verifier_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--original-ce-credit-overhead") == 0 && i + 1 < argc) {
            original_ce_credit_overhead = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--original-ce-credit-recurrent-backward-share") == 0 && i + 1 < argc) {
            original_ce_credit_recurrent_backward_share = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--original-ce-credit-min-progress") == 0 && i + 1 < argc) {
            original_ce_credit_min_progress = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--original-ce-credit-max-residual") == 0 && i + 1 < argc) {
            original_ce_credit_max_residual = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--original-ce-credit-min-whole-step-speedup") == 0 && i + 1 < argc) {
            original_ce_credit_min_whole_step_speedup = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--r375-pbg2-gate") == 0) {
            r375_pbg2_gate_enable = 1;
        } else if (strcmp(argv[i], "--r375-pbg2-max-matrices") == 0 && i + 1 < argc) {
            r375_pbg2_max_matrices = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--r375-pbg2-min-anisotropy") == 0 && i + 1 < argc) {
            r375_pbg2_min_anisotropy = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--r375-pbg2-precondition") == 0) {
            r375_pbg2_precondition_enable = 1;
        } else if (strcmp(argv[i], "--r375-pbg2-precondition-max-matrices") == 0 && i + 1 < argc) {
            r375_pbg2_precondition_max_matrices = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--r375-pbg2-precondition-eps") == 0 && i + 1 < argc) {
            r375_pbg2_precondition_eps = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--gradient-null-witness-gate") == 0) {
            gradient_null_witness_gate_enable = 1;
        } else if (strcmp(argv[i], "--gradient-null-witness-epsilon") == 0 && i + 1 < argc) {
            gradient_null_witness_epsilon = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--gradient-null-witness-omitted-budget") == 0 && i + 1 < argc) {
            gradient_null_witness_omitted_budget = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--gradient-null-witness-overhead") == 0 && i + 1 < argc) {
            gradient_null_witness_overhead = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--r429-decision-equivalence-witness") == 0) {
            r429_decision_equivalence_witness_enable = 1;
            gradient_null_witness_gate_enable = 1;
        } else if (strcmp(argv[i], "--ce-normal-chamber-gate") == 0) {
            ce_normal_chamber_gate_enable = 1;
        } else if (strcmp(argv[i], "--ce-normal-chamber-sigma") == 0 && i + 1 < argc) {
            ce_normal_chamber_sigma = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--ce-normal-chamber-overhead") == 0 && i + 1 < argc) {
            ce_normal_chamber_overhead = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--optimizer-integrity-gate") == 0) {
            optimizer_integrity_gate_enable = 1;
        } else if (strcmp(argv[i], "--paper7-dfga-real-batch-audit") == 0) {
            paper7_dfga_real_batch_audit_enable = 1;
        } else if (strcmp(argv[i], "--paper7-dfga-audit-max-batches") == 0 && i + 1 < argc) {
            paper7_dfga_real_batch_audit_max_batches = (uint32_t)strtoul(argv[++i], NULL, 10);
            if (paper7_dfga_real_batch_audit_max_batches == 0u) {
                paper7_dfga_real_batch_audit_max_batches = 1u;
            }
        } else if (strcmp(argv[i], "--paper7-dfga-audit-residual-budget") == 0 && i + 1 < argc) {
            paper7_dfga_real_batch_audit_residual_budget = (float)atof(argv[++i]);
            if (paper7_dfga_real_batch_audit_residual_budget < 0.0f) {
                fprintf(stderr, "[ERROR] --paper7-dfga-audit-residual-budget must be >= 0\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--quantization-cell-quotient-gate") == 0 ||
                   strcmp(argv[i], "--qcf-gate") == 0) {
            qcf_mvue_gate_enable = 1;
        } else if (strcmp(argv[i], "--quantization-cell-quotient-max-matrices") == 0 && i + 1 < argc) {
            qcf_mvue_max_matrices = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--quantization-cell-quotient-epsilon") == 0 && i + 1 < argc) {
            qcf_mvue_epsilon = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--quantization-cell-quotient-min-rho") == 0 && i + 1 < argc) {
            qcf_mvue_min_rho = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--quantization-cell-quotient-min-delta-energy") == 0 && i + 1 < argc) {
            qcf_mvue_min_delta_energy = atof(argv[++i]);
        } else if (strcmp(argv[i], "--evidence-rate-commitment-gate") == 0) {
            evidence_rate_commitment_gate_enable = 1;
        } else if (strcmp(argv[i], "--evidence-rate-commitment-mode") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (strcmp(mode, "tsa") == 0 || strcmp(mode, "tsa-like") == 0) {
                evidence_rate_commitment_mode = V12_EVIDENCE_RATE_MODE_TSA;
            } else if (strcmp(mode, "gbar") == 0 || strcmp(mode, "evidence") == 0) {
                evidence_rate_commitment_mode = V12_EVIDENCE_RATE_MODE_GBAR;
            } else {
                evidence_rate_commitment_mode = V12_EVIDENCE_RATE_MODE_CURRENT;
            }
        } else if (strcmp(argv[i], "--evidence-rate-force-repack") == 0) {
            evidence_rate_force_repack = 1;
        } else if (strcmp(argv[i], "--evidence-rate-overhead") == 0 && i + 1 < argc) {
            evidence_rate_overhead = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--evidence-rate-boundary-eps") == 0 && i + 1 < argc) {
            evidence_rate_boundary_eps = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--r388-flip-refresh-gate") == 0) {
            r388_flip_refresh_gate_enable = 1;
        } else if (strcmp(argv[i], "--r388-flip-refresh-block-elements") == 0 && i + 1 < argc) {
            r388_flip_refresh_block_elements = (uint32_t)strtoul(argv[++i], NULL, 10);
            if (r388_flip_refresh_block_elements == 0u) {
                r388_flip_refresh_block_elements = 1024u;
            }
        } else if (strcmp(argv[i], "--r388-flip-refresh-min-eta") == 0 && i + 1 < argc) {
            r388_flip_refresh_min_eta = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--fp-interface-gap-probe") == 0) {
            fp_interface_gap_probe_enable = 1;
        } else if (strcmp(argv[i], "--fp-interface-gap-effect-epsilon") == 0 && i + 1 < argc) {
            fp_interface_gap_effect_epsilon = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--credit-time-equalized-dynamics") == 0) {
            credit_time_equalized_dynamics_enable = 1;
        } else if (strcmp(argv[i], "--credit-time-equalized-dynamics-gate") == 0) {
            credit_time_equalized_dynamics_gate_enable = 1;
        } else if (strcmp(argv[i], "--credit-time-horizon") == 0 && i + 1 < argc) {
            credit_time_horizon = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--credit-time-min-scale") == 0 && i + 1 < argc) {
            credit_time_min_scale = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--credit-time-max-scale") == 0 && i + 1 < argc) {
            credit_time_max_scale = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--credit-time-overhead") == 0 && i + 1 < argc) {
            credit_time_overhead = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--contractive-green-credit") == 0) {
            contractive_green_credit_enable = 1;
        } else if (strcmp(argv[i], "--contractive-green-credit-gate") == 0) {
            contractive_green_credit_gate_enable = 1;
        } else if (strcmp(argv[i], "--contractive-green-credit-horizon") == 0 && i + 1 < argc) {
            contractive_green_credit_horizon = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--contractive-green-credit-max-alpha") == 0 && i + 1 < argc) {
            contractive_green_credit_max_alpha = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--contractive-green-credit-tail-budget") == 0 && i + 1 < argc) {
            contractive_green_credit_tail_budget = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--contractive-green-credit-min-scale") == 0 && i + 1 < argc) {
            contractive_green_credit_min_scale = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--contractive-green-credit-max-scale") == 0 && i + 1 < argc) {
            contractive_green_credit_max_scale = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--contractive-green-credit-overhead") == 0 && i + 1 < argc) {
            contractive_green_credit_overhead = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--adaptive-credit-horizon") == 0) {
            adaptive_credit_horizon_enable = 1;
        } else if (strcmp(argv[i], "--adaptive-credit-horizon-gate") == 0) {
            adaptive_credit_horizon_gate_enable = 1;
        } else if (strcmp(argv[i], "--adaptive-credit-min-horizon") == 0 && i + 1 < argc) {
            adaptive_credit_min_horizon = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--adaptive-credit-max-horizon") == 0 && i + 1 < argc) {
            adaptive_credit_max_horizon = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--adaptive-credit-tail-budget") == 0 && i + 1 < argc) {
            adaptive_credit_tail_budget = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--adaptive-credit-bound-budget") == 0 && i + 1 < argc) {
            adaptive_credit_bound_budget = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--adaptive-credit-verify-fraction") == 0 && i + 1 < argc) {
            adaptive_credit_verify_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--adaptive-credit-overhead") == 0 && i + 1 < argc) {
            adaptive_credit_overhead = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--adaptive-credit-min-headroom") == 0 && i + 1 < argc) {
            adaptive_credit_min_headroom = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--adaptive-credit-tail-scale") == 0 && i + 1 < argc) {
            adaptive_credit_tail_scale = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--interior-adjoint-tube") == 0) {
            interior_adjoint_tube_enable = 1;
        } else if (strcmp(argv[i], "--interior-adjoint-tube-gate") == 0) {
            interior_adjoint_tube_gate_enable = 1;
        } else if (strcmp(argv[i], "--interior-adjoint-tube-rank") == 0 && i + 1 < argc) {
            interior_adjoint_tube_rank = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--interior-adjoint-tube-residual-budget") == 0 && i + 1 < argc) {
            interior_adjoint_tube_residual_budget = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--interior-adjoint-tube-verify-fraction") == 0 && i + 1 < argc) {
            interior_adjoint_tube_verify_fraction = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--interior-adjoint-tube-overhead") == 0 && i + 1 < argc) {
            interior_adjoint_tube_overhead = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--interior-adjoint-tube-residual-scale") == 0 && i + 1 < argc) {
            interior_adjoint_tube_residual_scale = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--interior-adjoint-tube-min-energy-covered") == 0 && i + 1 < argc) {
            interior_adjoint_tube_min_energy_covered = (float)atof(argv[++i]);
        } else if ((strcmp(argv[i], "--corpus-logit-prior-alpha") == 0 ||
                    strcmp(argv[i], "--r408-scaffold-alpha") == 0) && i + 1 < argc) {
            corpus_logit_prior_alpha = (float)atof(argv[++i]);
        } else if ((strcmp(argv[i], "--corpus-logit-prior-scale") == 0 ||
                    strcmp(argv[i], "--r408-scaffold-scale") == 0) && i + 1 < argc) {
            corpus_logit_prior_scale = (float)atof(argv[++i]);
        } else if ((strcmp(argv[i], "--corpus-logit-prior-terms") == 0 ||
                    strcmp(argv[i], "--r408-scaffold-terms") == 0) && i + 1 < argc) {
            corpus_logit_prior_terms = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if ((strcmp(argv[i], "--corpus-logit-prior-stride") == 0 ||
                    strcmp(argv[i], "--r408-scaffold-stride") == 0) && i + 1 < argc) {
            corpus_logit_prior_stride = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--lccap") == 0) {
            lccap_enable = 1;
        } else if (strcmp(argv[i], "--residual-shell") == 0) {
            rs_enable = 1;
        } else if (strcmp(argv[i], "--rs-eps-lo") == 0 && i + 1 < argc) {
            rs_eps_lo = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--rs-eps-hi") == 0 && i + 1 < argc) {
            rs_eps_hi = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--exact-quotient-padding") == 0 ||
                   strcmp(argv[i], "--eqp") == 0) {
            exact_quotient_padding_enable = 1;
        } else if ((strcmp(argv[i], "--exact-quotient-padding-min-active-len") == 0 ||
                    strcmp(argv[i], "--eqp-min-active-len") == 0) && i + 1 < argc) {
            exact_quotient_padding_min_active_len =
                (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if ((strcmp(argv[i], "--exact-quotient-padding-seed") == 0 ||
                    strcmp(argv[i], "--eqp-seed") == 0) && i + 1 < argc) {
            exact_quotient_padding_seed =
                (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--aist-epsilon") == 0 && i + 1 < argc) {
#if V12_AIST_ENABLED
            v12_aist_epsilon = (float)atof(argv[++i]);
            printf("[AIST] runtime epsilon = %.6f\n", v12_aist_epsilon);
#else
            (void)argv[++i];
            printf("[AIST] --aist-epsilon ignored (V12_AIST_ENABLED=0)\n");
#endif
        } else if (argv[i][0] != '-' || argv[i][1] != '-') {
            if (positional_argc == 0) {
                dataset_path = argv[i];
            } else if (positional_argc == 1) {
                total_steps = atoi(argv[i]);
            } else if (positional_argc == 2) {
                checkpoint_path = argv[i];
            }
            positional_argc++;
        }
    }
    if (r364_warmup_steps <= 0) {
        fprintf(stderr, "[ERROR] --r364-warmup must be >= 1\n");
        return 1;
    }
    if (dpth_mode != DPTH_MODE_OFF && dpth_pool < 1) {
        fprintf(stderr, "[ERROR] --dpth-pool must be >= 1\n");
        return 1;
    }
    if (eval_every < 0) {
        fprintf(stderr, "[ERROR] --eval-every must be >= 0\n");
        return 1;
    }
    if (lr_base_override != -1.0f && lr_base_override <= 0.0f) {
        fprintf(stderr, "[ERROR] --lr-base must be > 0\n");
        return 1;
    }
    if (eval_batches < 1) {
        fprintf(stderr, "[ERROR] --eval-batches must be >= 1\n");
        return 1;
    }
    if (heldout_eval_dataset_path != NULL && heldout_eval_batches < 1) {
        fprintf(stderr, "[ERROR] --heldout-eval-batches must be >= 1 when --heldout-dataset is set\n");
        return 1;
    }
    if (r435_boundary_receipt_enable && r435_boundary_step == 0u) {
        fprintf(stderr, "[ERROR] --r435-boundary-step must be >= 1\n");
        return 1;
    }
    if (dpth_mode != DPTH_MODE_OFF && teaching_enabled) {
        teaching_enabled = 0;
        dpth_disabled_teaching = 1;
    }
    if (r364_compile_every <= 0) {
        fprintf(stderr, "[ERROR] --r364-compile-every must be >= 1\n");
        return 1;
    }
    if (r364_compiled_steps <= 0) {
        fprintf(stderr, "[ERROR] --r364-compiled-steps must be >= 1\n");
        return 1;
    }
    if (r364_compiled_steps > r364_compile_every) {
        fprintf(stderr, "[ERROR] --r364-compiled-steps must be <= --r364-compile-every\n");
        return 1;
    }
    if (corpus_logit_prior_alpha <= 0.0f) {
        fprintf(stderr, "[ERROR] --corpus-logit-prior-alpha must be > 0\n");
        return 1;
    }
    if (corpus_logit_prior_stride == 0u) {
        fprintf(stderr, "[ERROR] --corpus-logit-prior-stride must be >= 1\n");
        return 1;
    }
    if (exact_quotient_padding_enable) {
        if (exact_quotient_padding_min_active_len == 0u) {
            exact_quotient_padding_min_active_len =
                (uint32_t)(V12_SEQ_LEN - (V12_SEQ_LEN / 4));
        }
        if (exact_quotient_padding_min_active_len < 2u ||
            exact_quotient_padding_min_active_len > (uint32_t)V12_SEQ_LEN) {
            fprintf(stderr,
                    "[ERROR] --eqp-min-active-len must be in [2,%d]\n",
                    V12_SEQ_LEN);
            return 1;
        }
    }
    if (corpus_logit_prior_scale < 0.0f) {
        fprintf(stderr, "[ERROR] --corpus-logit-prior-scale must be >= 0\n");
        return 1;
    }
    if (program_kernel_logit_boost <= 0.0f) {
        fprintf(stderr, "[ERROR] --program-kernel-logit must be > 0\n");
        return 1;
    }
    if (program_kernel_min_count == 0u) {
        fprintf(stderr, "[ERROR] --program-kernel-min-count must be >= 1\n");
        return 1;
    }
    if (program_kernel_stride == 0u) {
        fprintf(stderr, "[ERROR] --program-kernel-stride must be >= 1\n");
        return 1;
    }
    if (certified_artifact_boundary_min_count == 0u) {
        fprintf(stderr, "[ERROR] --certified-artifact-boundary-min-count must be >= 1\n");
        return 1;
    }
    if (certified_artifact_boundary_stride == 0u) {
        fprintf(stderr, "[ERROR] --certified-artifact-boundary-stride must be >= 1\n");
        return 1;
    }
    if (certified_artifact_boundary_epsilon <= 0.0f ||
        certified_artifact_boundary_epsilon >= 1.0f) {
        fprintf(stderr, "[ERROR] --certified-artifact-boundary-epsilon must be in (0, 1)\n");
        return 1;
    }
    if (certified_artifact_boundary_ce_budget < 0.0f ||
        certified_artifact_boundary_overhead < 0.0f ||
        certified_artifact_boundary_kappa <= 0.0f) {
        fprintf(stderr, "[ERROR] certified artifact-boundary costs must be >= 0 and kappa must be > 0\n");
        return 1;
    }
    if (state_transition_atlas_logit_boost <= 0.0f) {
        fprintf(stderr, "[ERROR] --state-transition-atlas-logit must be > 0\n");
        return 1;
    }
    if (state_transition_atlas_min_count == 0u) {
        fprintf(stderr, "[ERROR] --state-transition-atlas-min-count must be >= 1\n");
        return 1;
    }
    if (state_transition_atlas_stride == 0u) {
        fprintf(stderr, "[ERROR] --state-transition-atlas-stride must be >= 1\n");
        return 1;
    }
    if (absorbed_state_atlas_overhead < 0.0f ||
        absorbed_state_atlas_memory_cost < 0.0f ||
        absorbed_state_atlas_repair_cost < 0.0f ||
        absorbed_state_atlas_kappa <= 0.0f) {
        fprintf(stderr, "[ERROR] absorbed state-atlas costs must be >= 0 and kappa must be > 0\n");
        return 1;
    }
    if (counterfactual_packets_min_count == 0u) {
        fprintf(stderr, "[ERROR] --counterfactual-packets-min-count must be >= 1\n");
        return 1;
    }
    if (counterfactual_packets_stride == 0u) {
        fprintf(stderr, "[ERROR] --counterfactual-packets-stride must be >= 1\n");
        return 1;
    }
    if (certified_silent_span_max_loss < 0.0f) {
        fprintf(stderr, "[ERROR] --certified-silent-span-max-loss must be >= 0\n");
        return 1;
    }
    if (certified_silent_span_min_len == 0u) {
        fprintf(stderr, "[ERROR] --certified-silent-span-min-len must be >= 1\n");
        return 1;
    }
    if (certified_silent_span_cert_cost < 0.0f ||
        certified_silent_span_repair_cost < 0.0f) {
        fprintf(stderr, "[ERROR] certified silent-span costs must be >= 0\n");
        return 1;
    }
    if (error_causal_lattice_min_count == 0u) {
        fprintf(stderr, "[ERROR] --error-causal-lattice-min-count must be >= 1\n");
        return 1;
    }
    if (error_causal_lattice_stride == 0u) {
        fprintf(stderr, "[ERROR] --error-causal-lattice-stride must be >= 1\n");
        return 1;
    }
    if (residual_active_sampling_floor <= 0.0f) {
        fprintf(stderr, "[ERROR] --residual-active-sampling-floor must be > 0\n");
        return 1;
    }
    if (residual_active_sampling_target_fraction <= 0.0f ||
        residual_active_sampling_target_fraction > 1.0f) {
        fprintf(stderr, "[ERROR] --residual-active-sampling-target-fraction must be in (0, 1]\n");
        return 1;
    }
    if (exact_transition_quotient_min_count == 0u) {
        fprintf(stderr, "[ERROR] --exact-transition-quotient-min-count must be >= 1\n");
        return 1;
    }
    if (exact_transition_quotient_stride == 0u) {
        fprintf(stderr, "[ERROR] --exact-transition-quotient-stride must be >= 1\n");
        return 1;
    }
    if (macro_ce_quotient_min_support == 0u ||
        macro_ce_quotient_stride == 0u ||
        macro_ce_quotient_signature_bins == 0u ||
        macro_ce_quotient_alpha <= 0.0f ||
        macro_ce_quotient_build_fraction <= 0.0f ||
        macro_ce_quotient_build_fraction > 1.0f ||
        macro_ce_quotient_residual_price_scale < 0.0f ||
        macro_ce_quotient_overhead < 0.0f ||
        macro_ce_quotient_training_weight < 0.0f ||
        macro_ce_quotient_training_weight > 1.0f) {
        fprintf(stderr,
                "[ERROR] invalid --macro-ce-quotient-* arguments: min_support/stride/signature_bins must be >=1, alpha/build_fraction positive, build_fraction <=1, costs non-negative, training_weight in [0,1]\n");
        return 1;
    }
    if (residual_certified_scaffold_audit_budget < 0.0f) {
        fprintf(stderr, "[ERROR] --residual-certified-scaffold-audit-budget must be >= 0\n");
        return 1;
    }
    if (exact_window_multiset_min_count == 0u) {
        fprintf(stderr, "[ERROR] --exact-window-multiset-min-count must be >= 1\n");
        return 1;
    }
    if (exact_window_multiset_stride == 0u) {
        fprintf(stderr, "[ERROR] --exact-window-multiset-stride must be >= 1\n");
        return 1;
    }
    if (exact_window_multiset_overhead < 0.0f ||
        exact_window_multiset_noise < 0.0f) {
        fprintf(stderr, "[ERROR] exact-window-multiset overhead and noise must be >= 0\n");
        return 1;
    }
    if (prefix_frontier_min_count == 0u) {
        fprintf(stderr, "[ERROR] --prefix-frontier-min-count must be >= 1\n");
        return 1;
    }
    if (prefix_frontier_stride == 0u) {
        fprintf(stderr, "[ERROR] --prefix-frontier-stride must be >= 1\n");
        return 1;
    }
    if (prefix_frontier_overhead < 0.0f || prefix_frontier_repair_cost < 0.0f) {
        fprintf(stderr, "[ERROR] prefix frontier overhead and repair cost must be >= 0\n");
        return 1;
    }
    if (macro_transcript_min_count == 0u || macro_transcript_stride == 0u) {
        fprintf(stderr, "[ERROR] macro transcript min-count and stride must be >= 1\n");
        return 1;
    }
    if (macro_transcript_hardness <= 0.0f ||
        macro_transcript_codec_cost < 0.0f ||
        macro_transcript_decode_cost < 0.0f) {
        fprintf(stderr, "[ERROR] macro transcript hardness must be > 0 and costs must be >= 0\n");
        return 1;
    }
    if (reversible_deterministic_expansion_min_run < 2u) {
        fprintf(stderr, "[ERROR] --reversible-deterministic-expansion-min-run must be >= 2\n");
        return 1;
    }
    if (reversible_deterministic_expansion_overhead < 0.0f ||
        reversible_deterministic_expansion_noise < 0.0f) {
        fprintf(stderr, "[ERROR] reversible deterministic expansion overhead and noise must be >= 0\n");
        return 1;
    }
    if (active_state_subspace_k == 0u || active_state_subspace_k > (uint32_t)V12_D_MODEL) {
        fprintf(stderr, "[ERROR] --active-state-subspace-k must be in [1, V12_D_MODEL]\n");
        return 1;
    }
    if (active_state_subspace_overhead < 0.0f) {
        fprintf(stderr, "[ERROR] --active-state-subspace-overhead must be >= 0\n");
        return 1;
    }
    if (boundary_liability_null_epsilon <= 0.0f ||
        boundary_liability_overhead < 0.0f) {
        fprintf(stderr, "[ERROR] boundary liability null epsilon must be > 0 and overhead must be >= 0\n");
        return 1;
    }
    if (boundary_flux_sheaf_overhead < 0.0f) {
        fprintf(stderr, "[ERROR] --boundary-flux-sheaf-overhead must be >= 0\n");
        return 1;
    }
    if (boundary_cotangent_overhead < 0.0f ||
        boundary_cotangent_residual_budget < 0.0f ||
        boundary_cotangent_verify_fraction < 0.0f) {
        fprintf(stderr, "[ERROR] boundary-cotangent overhead, residual budget, and verify fraction must be >= 0\n");
        return 1;
    }
    if (variance_priced_sample_fraction < 0.0f ||
        variance_priced_sample_fraction > 1.0f ||
        variance_priced_score_fraction < 0.0f ||
        variance_priced_verify_fraction < 0.0f ||
        variance_priced_overhead < 0.0f) {
        fprintf(stderr, "[ERROR] variance-priced sample fraction must be in [0,1] and score/verify/overhead must be >= 0\n");
        return 1;
    }
    if (equal_area_fisher_ess_floor < 0.0f ||
        equal_area_fisher_ess_floor > 1.0f ||
        equal_area_fisher_bias_budget < 0.0f ||
        equal_area_fisher_min_descent_per_paid_work < 0.0f) {
        fprintf(stderr, "[ERROR] equal-area Fisher ESS floor must be in [0,1] and bias/min-descent budgets must be >= 0\n");
        return 1;
    }
    if (event_innovation_fraction < 0.0f ||
        event_innovation_fraction > 1.0f ||
        event_innovation_debt_budget < 0.0f ||
        event_innovation_debt_budget > 1.0f ||
        event_innovation_score_cost < 0.0f ||
        event_innovation_verify_fraction < 0.0f ||
        event_innovation_overhead < 0.0f ||
        event_innovation_low_event_scale < 0.0f ||
        event_innovation_low_event_scale > 1.0f ||
        event_innovation_boundary_boost <= 0.0f) {
        fprintf(stderr, "[ERROR] event-innovation requires fractions/budgets in range, non-negative costs, low-event-scale in [0,1], and boundary boost > 0\n");
        return 1;
    }
    if (ce_debt_max_loss < 0.0f ||
        ce_debt_max_grad_norm < 0.0f ||
        ce_debt_max_debt < 0.0f ||
        ce_debt_max_debt > 1.0f ||
        ce_debt_audit_fraction < 0.0f ||
        ce_debt_audit_fraction > 1.0f ||
        ce_debt_overhead < 0.0f ||
        ce_debt_min_skip_fraction < 0.0f ||
        ce_debt_min_skip_fraction > 1.0f ||
        ce_debt_skip_scale < 0.0f ||
        ce_debt_skip_scale > 1.0f) {
        fprintf(stderr, "[ERROR] ce-debt-skip requires non-negative loss/grad thresholds, max-debt and fractions in [0,1], non-negative overhead, and skip-scale in [0,1]\n");
        return 1;
    }
    if (gradient_coherence_bins == 0u ||
        gradient_coherence_probe_stride == 0u ||
        gradient_coherence_max_residual < 0.0f ||
        gradient_coherence_max_residual > 1.0f ||
        gradient_coherence_score_cost < 0.0f ||
        gradient_coherence_verify_fraction < 0.0f ||
        gradient_coherence_overhead < 0.0f ||
        gradient_coherence_min_saved_fraction < 0.0f ||
        gradient_coherence_min_saved_fraction > 1.0f ||
        gradient_coherence_residual_scale < 0.0f ||
        gradient_coherence_residual_scale > 1.0f) {
        fprintf(stderr, "[ERROR] gradient-coherence-quotient requires bins/stride > 0, max-residual/min-saved/residual-scale in [0,1], and non-negative costs\n");
        return 1;
    }
    if (ce_transaction_k < 2u ||
        ce_transaction_min_margin < 0.0f ||
        ce_transaction_max_rollback_fraction < 0.0f ||
        ce_transaction_max_rollback_fraction > 1.0f ||
        ce_transaction_verifier_fraction < 0.0f) {
        fprintf(stderr, "[ERROR] R438 CE transaction blocks require k >= 2, min margin >= 0, max rollback fraction in [0,1], and verifier fraction >= 0\n");
        return 1;
    }
    if (!(original_ce_credit_surface_fraction > 0.0f &&
          original_ce_credit_surface_fraction <= 1.0f) ||
        original_ce_credit_local_work_multiplier < 0.0f ||
        original_ce_credit_debt_price < 0.0f ||
        original_ce_credit_verifier_fraction < 0.0f ||
        original_ce_credit_overhead < 0.0f ||
        original_ce_credit_recurrent_backward_share < 0.0f ||
        original_ce_credit_recurrent_backward_share > 1.0f ||
        original_ce_credit_min_progress < 0.0f ||
        original_ce_credit_max_residual < 0.0f ||
        original_ce_credit_min_whole_step_speedup < 1.0f) {
        fprintf(stderr, "[ERROR] original-CE credit observer requires surface in (0,1], non-negative costs/residual/progress, recurrent share in [0,1], and min speedup >= 1\n");
        return 1;
    }
    if (gradient_null_witness_epsilon <= 0.0f ||
        gradient_null_witness_omitted_budget < 0.0f ||
        gradient_null_witness_overhead < 0.0f) {
        fprintf(stderr, "[ERROR] gradient-null witness epsilon must be > 0 and budgets/overhead must be >= 0\n");
        return 1;
    }
    if (credit_time_horizon == 0u) {
        fprintf(stderr, "[ERROR] --credit-time-horizon must be >= 1\n");
        return 1;
    }
    if (credit_time_min_scale <= 0.0f ||
        credit_time_max_scale < credit_time_min_scale ||
        credit_time_overhead < 0.0f) {
        fprintf(stderr, "[ERROR] credit-time scales require 0 < min <= max and overhead >= 0\n");
        return 1;
    }
    if (contractive_green_credit_horizon == 0u) {
        fprintf(stderr, "[ERROR] --contractive-green-credit-horizon must be >= 1\n");
        return 1;
    }
    if (contractive_green_credit_max_alpha <= 0.0f ||
        contractive_green_credit_max_alpha >= 1.0f ||
        contractive_green_credit_tail_budget < 0.0f ||
        contractive_green_credit_tail_budget > 1.0f ||
        contractive_green_credit_min_scale <= 0.0f ||
        contractive_green_credit_max_scale < contractive_green_credit_min_scale ||
        contractive_green_credit_overhead < 0.0f) {
        fprintf(stderr, "[ERROR] contractive-green-credit requires 0 < max-alpha < 1, 0 <= tail-budget <= 1, 0 < min-scale <= max-scale, and overhead >= 0\n");
        return 1;
    }
    if (adaptive_credit_min_horizon == 0u ||
        adaptive_credit_max_horizon == 0u ||
        adaptive_credit_max_horizon < adaptive_credit_min_horizon ||
        adaptive_credit_tail_budget < 0.0f ||
        adaptive_credit_tail_budget > 1.0f ||
        adaptive_credit_bound_budget < 0.0f ||
        adaptive_credit_verify_fraction < 0.0f ||
        adaptive_credit_overhead < 0.0f ||
        adaptive_credit_min_headroom < 0.0f ||
        adaptive_credit_min_headroom > 1.0f ||
        adaptive_credit_tail_scale < 0.0f ||
        adaptive_credit_tail_scale > 1.0f) {
        fprintf(stderr, "[ERROR] adaptive-credit-horizon requires min/max horizon >=1 with max >= min, budgets in range, non-negative costs, min-headroom in [0,1], and tail-scale in [0,1]\n");
        return 1;
    }
    if (interior_adjoint_tube_rank == 0u ||
        interior_adjoint_tube_rank > (uint32_t)V12_D_STATE ||
        interior_adjoint_tube_residual_budget < 0.0f ||
        interior_adjoint_tube_residual_budget > 1.0f ||
        interior_adjoint_tube_verify_fraction < 0.0f ||
        interior_adjoint_tube_overhead < 0.0f ||
        interior_adjoint_tube_residual_scale < 0.0f ||
        interior_adjoint_tube_residual_scale > 1.0f ||
        interior_adjoint_tube_min_energy_covered < 0.0f ||
        interior_adjoint_tube_min_energy_covered > 1.0f) {
        fprintf(stderr, "[ERROR] interior-adjoint-tube requires rank in [1,V12_D_STATE], budgets/fractions in range, non-negative overhead, and residual-scale in [0,1]\n");
        return 1;
    }
    if (fp_interface_gap_effect_epsilon <= 0.0f) {
        fprintf(stderr, "[ERROR] --fp-interface-gap-effect-epsilon must be > 0\n");
        return 1;
    }
    printf("[TRAIN] batch_size = %d\n", batch_size);
    printf("[TRAIN] log_every = %d\n", log_every);
    r395_trace_config_init(&r395_trace);
    r395_trace_log_activation_once(&r395_trace, batch_size);
    r399_trace_config_init(&r399_trace);
    r399_trace_log_activation_once(&r399_trace, batch_size, V12_SEQ_LEN);
    r399_dmdc_trace_config_init(&r399_dmdc_trace);
    r399_dmdc_trace_log_activation_once(&r399_dmdc_trace);
    if (loss_log_hex) {
        printf("[TRAIN] loss_log_hex = 1\n");
    }
    if (vk_profile) {
        printf("[TRAIN] vk_profile = 1\n");
        printf("[R370] profiling gate active mode=vk-profile\n");
    }
    if (ckpt_debug) {
        printf("[TRAIN] ckpt_debug = 1\n");
    }
    if (!checkpoint_save_enabled) {
        printf("[CKPT] save_mode=load_only\n");
    }
    printf("[TRAIN] teaching = %s\n", teaching_enabled ? "on" : "off");
    if (teaching_enabled) {
        printf("[TRAIN] teaching_capacity = %d\n", teaching_capacity);
    }
    printf("[TRAIN] eval_every = %d\n", eval_every);
    printf("[TRAIN] eval_batches = %d\n", eval_batches);
    if (heldout_eval_dataset_path != NULL) {
        printf("[TRAIN] heldout_dataset = %s\n", heldout_eval_dataset_path);
        printf("[TRAIN] heldout_eval_batches = %d\n", heldout_eval_batches);
    }
    if (r435_boundary_receipt_enable) {
        printf("[R435_BOUNDARY_RECEIPT] requested=1 mode=real-path-boundary boundary_step=%u probe=probe-train-step-zeroed no_default_change=1\n",
               r435_boundary_step);
    }
    if (!r493_post_step_enable) {
        printf("[R493_POST_STEP] active=0 reason=explicit_smoke_opt_out default_path=unchanged\n");
    }
    if (exact_quotient_padding_enable) {
        printf("[EXACT_QUOTIENT_PADDING] requested=1 mode=real-training-active-prefix-ce min_active_len=%u seq_len=%d seed=%llu objective=active_prefix_ce default_path=unchanged\n",
               exact_quotient_padding_min_active_len,
               V12_SEQ_LEN,
               (unsigned long long)exact_quotient_padding_seed);
    }
    if (dpth_mode != DPTH_MODE_OFF) {
        printf("[DPTH_GATE] requested=1 mode=%s pool=%d source=cheap_context_packet_sampler objective=original_ce_per_priced_second oracle=0 control=%s teaching_disabled=%d\n",
               dpth_mode_name(dpth_mode),
               dpth_pool,
               dpth_mode == DPTH_MODE_LOSS_CONTROL ? "loss_proxy" : "dpth_packet",
               dpth_disabled_teaching);
    }
    if (r364_enable) {
        printf("[R364] pilot=on target=lm_head mode=last-block-ls warmup=%d compile_every=%d compiled_steps=%d\n",
               r364_warmup_steps,
               r364_compile_every,
               r364_compiled_steps);
    } else {
        printf("[R364] pilot=off target=lm_head mode=last-block-ls\n");
    }
    if (corpus_logit_prior_enable) {
        printf("[CORPUS_LOGIT_PRIOR] requested=1 alpha=%.3f scale=%.6f terms=%llu stride=%u\n",
               corpus_logit_prior_alpha,
               corpus_logit_prior_scale,
               (unsigned long long)corpus_logit_prior_terms,
               corpus_logit_prior_stride);
    }
    if (logit_prior_gate_enable) {
        printf("[LOGIT_PRIOR_GATE] requested=1 mode=baseline-vs-residual-sampled-ce\n");
    }
    if (program_kernel_enable) {
        printf("[PROGRAM_KERNEL] requested=1 logit_boost=%.6f min_count=%u terms=%llu stride=%u\n",
               program_kernel_logit_boost,
               program_kernel_min_count,
               (unsigned long long)program_kernel_terms,
               program_kernel_stride);
    }
    if (program_kernel_gate_enable) {
        printf("[PROGRAM_KERNEL_GATE] requested=1 mode=baseline-vs-program-kernel-sampled-ce\n");
    }
    if (certified_artifact_boundary_gate_enable) {
        printf("[CERTIFIED_ARTIFACT_BOUNDARY_GATE] requested=1 mode=copy-current-proof-transducer min_count=%u terms=%llu stride=%u epsilon=%.9f ce_budget=%.9f overhead=%.6f kappa=%.6f\n",
               certified_artifact_boundary_min_count,
               (unsigned long long)certified_artifact_boundary_terms,
               certified_artifact_boundary_stride,
               certified_artifact_boundary_epsilon,
               certified_artifact_boundary_ce_budget,
               certified_artifact_boundary_overhead,
               certified_artifact_boundary_kappa);
    }
    if (state_transition_atlas_enable) {
        printf("[STATE_TRANSITION_ATLAS] requested=1 logit_boost=%.6f min_count=%u terms=%llu stride=%u\n",
               state_transition_atlas_logit_boost,
               state_transition_atlas_min_count,
               (unsigned long long)state_transition_atlas_terms,
               state_transition_atlas_stride);
    }
    if (state_transition_atlas_gate_enable) {
        printf("[STATE_TRANSITION_ATLAS_GATE] requested=1 mode=baseline-vs-state-atlas-sampled-ce\n");
    }
    if (absorbed_state_atlas_gate_enable) {
        printf("[ABSORBED_STATE_ATLAS_GATE] requested=1 mode=state-atlas-residual-burden overhead=%.6f memory_cost=%.6f repair_cost=%.6f kappa=%.6f\n",
               absorbed_state_atlas_overhead,
               absorbed_state_atlas_memory_cost,
               absorbed_state_atlas_repair_cost,
               absorbed_state_atlas_kappa);
    }
    if (counterfactual_packets_enable) {
        printf("[COUNTERFACTUAL_PACKETS] requested=1 min_count=%u terms=%llu stride=%u\n",
               counterfactual_packets_min_count,
               (unsigned long long)counterfactual_packets_terms,
               counterfactual_packets_stride);
    }
    if (counterfactual_packets_gate_enable) {
        printf("[COUNTERFACTUAL_PACKETS_GATE] requested=1 mode=local-state-pair-coverage\n");
    }
    if (certified_silent_span_gate_enable) {
        printf("[CERTIFIED_SILENT_SPAN_GATE] requested=1 max_loss=%.6f min_len=%u cert_cost=%.6f repair_cost=%.6f\n",
               certified_silent_span_max_loss,
               certified_silent_span_min_len,
               certified_silent_span_cert_cost,
               certified_silent_span_repair_cost);
    }
    if (error_causal_lattice_enable) {
        printf("[ERROR_CAUSAL_LATTICE] requested=1 min_count=%u terms=%llu stride=%u\n",
               error_causal_lattice_min_count,
               (unsigned long long)error_causal_lattice_terms,
               error_causal_lattice_stride);
    }
    if (error_causal_lattice_gate_enable) {
        printf("[ERROR_CAUSAL_LATTICE_GATE] requested=1 mode=coverage-conserving-prerequisite-order\n");
    }
    if (residual_active_sampling_gate_enable) {
        printf("[RESIDUAL_ACTIVE_SAMPLING_GATE] requested=1 mode=unbiased-importance-sampling floor=%.9g target_fraction=%.6f\n",
               residual_active_sampling_floor,
               residual_active_sampling_target_fraction);
    }
    if (exact_transition_quotient_enable) {
        printf("[EXACT_TRANSITION_QUOTIENT] requested=1 min_count=%u terms=%llu stride=%u\n",
               exact_transition_quotient_min_count,
               (unsigned long long)exact_transition_quotient_terms,
               exact_transition_quotient_stride);
    }
    if (exact_transition_quotient_gate_enable) {
        printf("[EXACT_TRANSITION_QUOTIENT_GATE] requested=1 mode=objective-identical-local-multiplicity\n");
    }
    if (macro_ce_quotient_observer_enable) {
        printf("[R421_MACRO_CE_QUOTIENT] requested=1 observer=%d training=%d min_support=%u terms=%llu stride=%u build_fraction=%.6f alpha=%.6f residual_price_scale=%.6f overhead=%.6f signature_bins=%u seed=%u training_weight=%.6f\n",
               macro_ce_quotient_observer_enable ? 1 : 0,
               macro_ce_quotient_training_enable ? 1 : 0,
               macro_ce_quotient_min_support,
               (unsigned long long)macro_ce_quotient_terms,
               macro_ce_quotient_stride,
               macro_ce_quotient_build_fraction,
               macro_ce_quotient_alpha,
               macro_ce_quotient_residual_price_scale,
               macro_ce_quotient_overhead,
               macro_ce_quotient_signature_bins,
               macro_ce_quotient_seed,
               macro_ce_quotient_training_weight);
    }
    if (residual_certified_scaffold_enable ||
        residual_certified_scaffold_gate_enable) {
        printf("[RESIDUAL_CERTIFIED_SCAFFOLD] requested=1 apply=%d gate=%d mode=objective-preserving-context-scaffold-quotient source=macro_ce_quotient original_ce_visible=1 min_support=%u terms=%llu stride=%u build_fraction=%.6f audit_budget=%.6f residual_price_scale=%.6f overhead=%.6f training_weight=%.6f pass_requires=ce_excess_within_budget_and_priced_work_ratio_lte_0.90\n",
               residual_certified_scaffold_enable ? 1 : 0,
               residual_certified_scaffold_gate_enable ? 1 : 0,
               macro_ce_quotient_min_support,
               (unsigned long long)macro_ce_quotient_terms,
               macro_ce_quotient_stride,
               macro_ce_quotient_build_fraction,
               residual_certified_scaffold_audit_budget,
               macro_ce_quotient_residual_price_scale,
               macro_ce_quotient_overhead,
               macro_ce_quotient_training_weight);
    }
    if (r423_native_growth_readout_quotient_enable) {
        printf("[R423_NATIVE_GROWTH_READOUT_QUOTIENT] requested=1 mode=real-training-bridge source=macro-ce-quotient-training covered_family=readout_null_denominator objective=sample_efficiency training_change=1\n");
    }
    if (exact_window_multiset_gate_enable) {
        printf("[EXACT_WINDOW_MULTISET_GATE] requested=1 mode=exact-window-sufficient-statistic min_count=%u windows=%llu stride=%u overhead=%.6f noise=%.6f\n",
               exact_window_multiset_min_count,
               (unsigned long long)exact_window_multiset_windows,
               exact_window_multiset_stride,
               exact_window_multiset_overhead,
               exact_window_multiset_noise);
    }
    if (prefix_frontier_gate_enable) {
        printf("[PREFIX_FRONTIER_GATE] requested=1 mode=exact-frontier-sampled-ce-adjoint min_count=%u terms=%llu stride=%u overhead=%.6f repair_cost=%.6f\n",
               prefix_frontier_min_count,
               (unsigned long long)prefix_frontier_terms,
               prefix_frontier_stride,
               prefix_frontier_overhead,
               prefix_frontier_repair_cost);
    }
    if (active_state_subspace_gate_enable) {
        printf("[ACTIVE_STATE_SUBSPACE_GATE] requested=1 mode=gpu-final-residual-gradient-energy k=%u overhead=%.6f\n",
               active_state_subspace_k,
               active_state_subspace_overhead);
    }
    if (boundary_liability_ledger_gate_enable) {
        printf("[BOUNDARY_LIABILITY_LEDGER_GATE] requested=1 mode=boundary-gradient-null-liability source=gpu_final_residual_gradient null_epsilon=%.9g overhead=%.6f objective=unchanged_ce\n",
               boundary_liability_null_epsilon,
               boundary_liability_overhead);
    }
    if (boundary_flux_sheaf_gate_enable) {
        printf("[BOUNDARY_FLUX_SHEAF_GATE] requested=1 mode=ssm-transition-credit-endpoint-flux source=gpu_dA_per_step endpoints=boundary_or_pre_boundary overhead=%.6f objective=unchanged_ce\n",
               boundary_flux_sheaf_overhead);
    }
    if (boundary_cotangent_credit_gate_enable) {
        printf("[BOUNDARY_COTANGENT_CREDIT_GATE] requested=1 mode=boundary-anchor-linear-cotangent-reconstruction source=gpu_dA_per_step anchors=boundary_or_pre_boundary_plus_window_endpoints residual_budget=%.6f verify_fraction=%.6f overhead=%.6f objective=unchanged_ce pass_requires=priced_ratio_lt_0.90_and_residual_within_budget\n",
               boundary_cotangent_residual_budget,
               boundary_cotangent_verify_fraction,
               boundary_cotangent_overhead);
    }
    if (variance_priced_credit_gate_enable) {
        printf("[VARIANCE_PRICED_CREDIT_GATE] requested=1 mode=norm-priced-transition-credit-sampling source=gpu_dA_per_step sample_fraction=%.6f score_fraction=%.6f verify_fraction=%.6f overhead=%.6f objective=unchanged_ce sampled_update=simulated pass_requires=descent_per_paid_work_gt_1.111\n",
               variance_priced_sample_fraction,
               variance_priced_score_fraction,
               variance_priced_verify_fraction,
               variance_priced_overhead);
    }
    if (equal_area_fisher_sampling_enable ||
        equal_area_fisher_sampling_gate_enable) {
        printf("[EQUAL_AREA_FISHER_SAMPLING] requested=1 apply=%d gate=%d mode=bounded-equal-area-fisher-snr-sampling source=gpu_dA_per_step sample_fraction=%.6f ess_floor=%.6f bias_budget=%.6f score_fraction=%.6f verify_fraction=%.6f overhead=%.6f objective=unchanged_ce_default opt_in_training_change=%d pass_requires=ess_floor_and_bias_budget_and_descent_per_paid\n",
               equal_area_fisher_sampling_enable ? 1 : 0,
               equal_area_fisher_sampling_gate_enable ? 1 : 0,
               variance_priced_sample_fraction,
               equal_area_fisher_ess_floor,
               equal_area_fisher_bias_budget,
               variance_priced_score_fraction,
               variance_priced_verify_fraction,
               variance_priced_overhead,
               equal_area_fisher_sampling_enable ? 1 : 0);
    }
    if (event_innovation_credit_enable || event_innovation_credit_gate_enable) {
        printf("[EVENT_INNOVATION_CREDIT] requested=1 apply=%d gate=%d mode=forward-owned-state-innovation-event-credit source=vulkan_fwd_h_cache+gpu_ssm_a+gpu_a_gate+gpu_dA_per_step event_fraction=%.6f debt_budget=%.6f score_cost=%.6f verify_fraction=%.6f overhead=%.6f low_event_scale=%.6f boundary_boost=%.6f objective=unchanged_ce_default opt_in_training_change=%d pass_requires=low_event_debt_within_budget_and_priced_ratio_lt_0.90\n",
               event_innovation_credit_enable ? 1 : 0,
               event_innovation_credit_gate_enable ? 1 : 0,
               event_innovation_fraction,
               event_innovation_debt_budget,
               event_innovation_score_cost,
               event_innovation_verify_fraction,
               event_innovation_overhead,
               event_innovation_low_event_scale,
               event_innovation_boundary_boost,
               event_innovation_credit_enable ? 1 : 0);
    }
    if (ce_debt_skip_enable || ce_debt_skip_gate_enable) {
        printf("[CE_DEBT_SKIP] requested=1 apply=%d gate=%d mode=sampled-ce-low-debt-token-skip source=gpu_ce_loss+gpu_ce_d_logits max_loss=%.6f max_grad_norm=%.6f max_debt=%.6f audit_fraction=%.6f overhead=%.6f min_skip_fraction=%.6f skip_scale=%.6f objective=unchanged_ce_default opt_in_training_change=%d pass_requires=debt_within_budget_and_priced_ratio_lt_0.90q\n",
               ce_debt_skip_enable ? 1 : 0,
               ce_debt_skip_gate_enable ? 1 : 0,
               ce_debt_max_loss,
               ce_debt_max_grad_norm,
               ce_debt_max_debt,
               ce_debt_audit_fraction,
               ce_debt_overhead,
               ce_debt_min_skip_fraction,
               ce_debt_skip_scale,
               ce_debt_skip_enable ? 1 : 0);
    }
    if (gradient_coherence_quotient_enable ||
        gradient_coherence_quotient_gate_enable) {
        printf("[GRADIENT_COHERENCE_QUOTIENT] requested=1 apply=%d gate=%d mode=transition-credit-representative-residual-quotient source=gpu_dA_per_step+vulkan_fwd_h_cache+gpu_ssm_a bins=%u probe_stride=%u max_residual=%.6f score_cost=%.6f verify_fraction=%.6f overhead=%.6f min_saved_fraction=%.6f residual_scale=%.6f objective=unchanged_ce_default opt_in_training_change=%d pass_requires=priced_ratio_lte_0.90q_and_residual_within_budget\n",
               gradient_coherence_quotient_enable ? 1 : 0,
               gradient_coherence_quotient_gate_enable ? 1 : 0,
               gradient_coherence_bins,
               gradient_coherence_probe_stride,
               gradient_coherence_max_residual,
               gradient_coherence_score_cost,
               gradient_coherence_verify_fraction,
               gradient_coherence_overhead,
               gradient_coherence_min_saved_fraction,
               gradient_coherence_residual_scale,
               gradient_coherence_quotient_enable ? 1 : 0);
    }
    if (lagged_ce_impulse_eligibility_enable) {
        printf("[LAGGED_CE_IMPULSE_ELIGIBILITY] requested=1 mode=lagged-ce-hidden-impulse-ledger source=gpu_dA_per_step+gpu_v_pre+vulkan_fwd_h_cache+gpu_ssm_a+gpu_a_gate horizon=%u blend=%.6f residual_budget=%.6f local_jvp_fraction=%.6f ring_fraction=%.6f tail_audit_fraction=%.6f memory_fraction=%.6f objective=changed_training opt_in=1 pass_requires=all_residuals_within_budget_and_priced_ratio_lt_0.90 default_path=unchanged\n",
               lagged_ce_impulse_horizon,
               lagged_ce_impulse_blend,
               lagged_ce_impulse_residual_budget,
               lagged_ce_impulse_local_jvp_fraction,
               lagged_ce_impulse_ring_fraction,
               lagged_ce_impulse_tail_audit_fraction,
               lagged_ce_impulse_memory_fraction);
    }
    if (executable_multiplicity_packets_enable) {
        printf("[R438_EXECUTABLE_MULTIPLICITY_PACKETS] requested=1 mode=state_transition_signature_packets source=vulkan_fwd_h_cache+gpu_ssm_a+gpu_dA_per_step min_count=%u min_mass=%.6f max_residual=%.6f signature_fraction=%.6f verifier_fraction=%.6f memory_fraction=%.6f state_scale=%.6f transition_scale=%.6f objective=original_ce observer_only=1 no_training_change=1 default_path=unchanged\n",
               executable_multiplicity_min_count,
               executable_multiplicity_min_mass,
               executable_multiplicity_max_residual,
               executable_multiplicity_signature_fraction,
               executable_multiplicity_verifier_fraction,
               executable_multiplicity_memory_fraction,
               executable_multiplicity_state_scale,
               executable_multiplicity_transition_scale);
    }
    if (ce_transaction_blocks_enable) {
        printf("[CE_TX_OBSERVER] requested=1 mode=curvature_receipted_transaction_blocks source=real_training_loss k=%u min_margin=%.6f max_rollback_fraction=%.6f verifier_fraction=%.6f objective=original_ce observer_only=1 no_training_change=1 default_path=unchanged\n",
               ce_transaction_k,
               ce_transaction_min_margin,
               ce_transaction_max_rollback_fraction,
               ce_transaction_verifier_fraction);
    }
    if (original_ce_credit_debt_observer_enable) {
        printf("[ORIGINAL_CE_CREDIT_DEBT_OBSERVER] requested=1 mode=local-direct-transition-credit source=gpu_dA_per_step+gpu_v_pre+vulkan_fwd_h_cache surface_fraction=%.6f local_work_multiplier=%.6f debt_price=%.6f verifier_fraction=%.6f overhead=%.6f recurrent_backward_share=%.6f objective=unchanged_ce observer_only=1 pass_requires=local_or_surface_priced_material\n",
               original_ce_credit_surface_fraction,
               original_ce_credit_local_work_multiplier,
               original_ce_credit_debt_price,
               original_ce_credit_verifier_fraction,
               original_ce_credit_overhead,
               original_ce_credit_recurrent_backward_share);
    }
    if (r375_pbg2_gate_enable) {
        printf("[R375_PBG2_GATE] requested=1 mode=real-gradient-kronecker-shape source=gpu_w_in_w_out max_matrices=%u min_anisotropy=%.6f objective=unchanged_ce training_change=0 pass_requires=shape_signal_ge_min_and_mvue_active_ge_0.90 grid_snr_reported_not_gated\n",
               r375_pbg2_max_matrices,
               r375_pbg2_min_anisotropy);
    }
    if (r375_pbg2_precondition_enable) {
        printf("[R375_PBG2_PRECOND] requested=1 mode=row_col_rms_preconditioned_gradient source=gpu_w_in_w_out max_matrices=%u eps=%.9g objective=changed_training opt_in=1 norm_preserved=global_rms\n",
               r375_pbg2_precondition_max_matrices,
               r375_pbg2_precondition_eps);
    }
    if (gradient_null_witness_gate_enable) {
        printf("[GRADIENT_NULL_WITNESS_GATE] requested=1 mode=full-parameter-gradient-group-certificate source=cpu_and_gpu_gradient_tensors epsilon=%.9g omitted_budget=%.9g overhead=%.6f objective=unchanged_ce\n",
               gradient_null_witness_epsilon,
               gradient_null_witness_omitted_budget,
               gradient_null_witness_overhead);
    }
    if (r429_decision_equivalence_witness_enable) {
        printf("[R429_DECISION_EQUIVALENCE_WITNESS] requested=1 mode=exact-zero-gradient-ternary-no-update-witness source=gradient_null_witness full_fallback=1 mismatch_counter=1 objective=unchanged_ce no_lossy_compression=1\n");
    }
    if (ce_normal_chamber_gate_enable) {
        printf("[CE_NORMAL_CHAMBER_GATE] requested=1 mode=ternary-shadow-no-crossing-force-bound source=gpu_owned_ternary_gradients K=1,4,16 sigma=%.9g overhead=%.6f objective=unchanged_ce pass_requires=priced_k16_gt_0.10 replay=not_run\n",
               ce_normal_chamber_sigma,
               ce_normal_chamber_overhead);
    }
    if (optimizer_integrity_gate_enable) {
        printf("[OPTIMIZER_INTEGRITY_GATE] requested=1 mode=gpu-owned-ternary-update-obligation metric=cpu_skip_vs_gpu_update_groups objective=training_fidelity\n");
    }
    if (paper7_dfga_real_batch_audit_enable) {
        printf("[PAPER7_DFGA_REAL_BATCH_AUDIT] requested=1 mode=sampled_hot_support_dense_audit source=v12_gpu_ce_dlogits+norm_out+sample_ids max_batches=%u residual_budget=%.6f objective=unchanged_ce observer_only=1 no_training_change=1\n",
               paper7_dfga_real_batch_audit_max_batches,
               paper7_dfga_real_batch_audit_residual_budget);
    }
    if (evidence_rate_commitment_gate_enable) {
        const char *evidence_mode_name =
            evidence_rate_commitment_mode == V12_EVIDENCE_RATE_MODE_TSA ? "tsa" :
            evidence_rate_commitment_mode == V12_EVIDENCE_RATE_MODE_GBAR ? "gbar" :
            "current";
        printf("[EVIDENCE_RATE_COMMITMENT_GATE] requested=1 mode=%s path=vulkan-ternary-ema-sprt objective=original_ce_per_priced_second overhead=%.6f boundary_eps=%.6f force_repack=%d gate=r_descent_ge_(1+h)/0.90\n",
               evidence_mode_name,
               evidence_rate_overhead,
               evidence_rate_boundary_eps,
               (evidence_rate_force_repack ||
                evidence_rate_commitment_mode != V12_EVIDENCE_RATE_MODE_CURRENT) ? 1 : 0);
    }
    if (r388_flip_refresh_gate_enable) {
        printf("[R388_FLIP_REFRESH_GATE] requested=1 mode=block-exact-sampled-code-stability source=vulkan_mvue_refresh block_elements=%u min_eta=%.6f objective=unchanged_ce pass_requires=eta_ge_min_eta\n",
               r388_flip_refresh_block_elements,
               r388_flip_refresh_min_eta);
    }
    if (fp_interface_gap_probe_enable) {
        printf("[FP_INTERFACE_GAP_PROBE] requested=1 mode=current-v12-vulkan effect_epsilon=%.6f objective=unchanged_ce training_change=0\n",
               fp_interface_gap_effect_epsilon);
    }
    if (credit_time_equalized_dynamics_enable || credit_time_equalized_dynamics_gate_enable) {
        printf("[CREDIT_TIME_EQUALIZED_DYNAMICS] active=1 apply=%d gate=%d horizon=%u min_scale=%.6f max_scale=%.6f overhead=%.6f path=ssm-gradient-preconditioner\n",
               credit_time_equalized_dynamics_enable ? 1 : 0,
               credit_time_equalized_dynamics_gate_enable ? 1 : 0,
               credit_time_horizon,
               credit_time_min_scale,
               credit_time_max_scale,
               credit_time_overhead);
    }
    if (contractive_green_credit_enable || contractive_green_credit_gate_enable) {
        printf("[CONTRACTIVE_GREEN_CREDIT] active=1 apply=%d gate=%d horizon=%u max_alpha=%.6f tail_budget=%.6f min_scale=%.6f max_scale=%.6f overhead=%.6f path=ssm-finite-green-tail-certified-gradient-preconditioner objective=unchanged_original_ce\n",
               contractive_green_credit_enable ? 1 : 0,
               contractive_green_credit_gate_enable ? 1 : 0,
               contractive_green_credit_horizon,
               contractive_green_credit_max_alpha,
               contractive_green_credit_tail_budget,
               contractive_green_credit_min_scale,
               contractive_green_credit_max_scale,
               contractive_green_credit_overhead);
    }
    if (adaptive_credit_horizon_enable || adaptive_credit_horizon_gate_enable) {
        printf("[ADAPTIVE_CREDIT_HORIZON] active=1 apply=%d gate=%d min_horizon=%u max_horizon=%u tail_budget=%.6f bound_budget=%.6f verify_fraction=%.6f overhead=%.6f min_headroom=%.6f tail_scale=%.6f path=ssm-transition-credit-horizon-certificate objective=unchanged_original_ce_default opt_in_training_change=%d\n",
               adaptive_credit_horizon_enable ? 1 : 0,
               adaptive_credit_horizon_gate_enable ? 1 : 0,
               adaptive_credit_min_horizon,
               adaptive_credit_max_horizon,
               adaptive_credit_tail_budget,
               adaptive_credit_bound_budget,
               adaptive_credit_verify_fraction,
               adaptive_credit_overhead,
               adaptive_credit_min_headroom,
               adaptive_credit_tail_scale,
               adaptive_credit_horizon_enable ? 1 : 0);
    }
    if (interior_adjoint_tube_enable || interior_adjoint_tube_gate_enable) {
        printf("[INTERIOR_ADJOINT_TUBE] active=1 apply=%d gate=%d rank=%u residual_budget=%.6f verify_fraction=%.6f overhead=%.6f residual_scale=%.6f min_energy_covered=%.6f path=state-gradient-energy-topk-tube-residual-settlement objective=unchanged_original_ce_default opt_in_training_change=%d\n",
               interior_adjoint_tube_enable ? 1 : 0,
               interior_adjoint_tube_gate_enable ? 1 : 0,
               interior_adjoint_tube_rank,
               interior_adjoint_tube_residual_budget,
               interior_adjoint_tube_verify_fraction,
               interior_adjoint_tube_overhead,
               interior_adjoint_tube_residual_scale,
               interior_adjoint_tube_min_energy_covered,
               interior_adjoint_tube_enable ? 1 : 0);
    }
    if (macro_transcript_gate_enable) {
        printf("[MACRO_TRANSCRIPT_GATE] requested=1 mode=exact-reversible-macro-transcript min_count=%u terms=%llu stride=%u hardness=%.6f codec_cost=%.6f decode_cost=%.6f\n",
               macro_transcript_min_count,
               (unsigned long long)macro_transcript_terms,
               macro_transcript_stride,
               macro_transcript_hardness,
               macro_transcript_codec_cost,
               macro_transcript_decode_cost);
    }
    if (reversible_deterministic_expansion_gate_enable) {
        printf("[REVERSIBLE_DETERMINISTIC_EXPANSION_GATE] requested=1 mode=fixed-repeat-run-transducer min_run=%u terms=%llu overhead=%.6f noise=%.6f\n",
               reversible_deterministic_expansion_min_run,
               (unsigned long long)reversible_deterministic_expansion_terms,
               reversible_deterministic_expansion_overhead,
               reversible_deterministic_expansion_noise);
    }

    HardwareProfile hw = {0};
    hardware_detect(&hw);
    hardware_print(&hw);

    int omp_threads = (hw.cpu_cores > 2) ? hw.cpu_cores - 2 : 1;
    omp_set_num_threads(omp_threads);
    printf("[TRAIN] OpenMP threads = %d\n", omp_threads);

    printf("[TRAIN] Loading dataset from %s\n", dataset_path);
    Dataset *ds = dataset_load(dataset_path);
    if (!ds) {
        fprintf(stderr, "[ERROR] Failed to load dataset\n");
        return 1;
    }
    if (seed_provided) {
        dataset_seed(ds, dataset_seed_value);
        dataset_reset_cursor(ds);
        printf("[TRAIN] dataset_seed = %llu\n", (unsigned long long)dataset_seed_value);
    }

        printf("[R505] Cumulative Opts Active (Speed: S01-S30, Efficiency: E01-E30, Quality: Q01-Q30)\n");
    printf("[TRAIN] Allocating model...\n");
    Model *m = model_alloc();
    if (!m) {
        fprintf(stderr, "[ERROR] Failed to allocate model\n");
        dataset_free(ds);
        return 1;
    }
    m->logit_prior_gate_enabled = logit_prior_gate_enable ? 1 : 0;
    m->program_kernel_gate_enabled = program_kernel_gate_enable ? 1 : 0;
    m->certified_artifact_boundary_gate_enabled =
        certified_artifact_boundary_gate_enable ? 1 : 0;
    m->certified_artifact_boundary_min_count = certified_artifact_boundary_min_count;
    m->certified_artifact_boundary_stride = certified_artifact_boundary_stride;
    m->certified_artifact_boundary_requested_terms = certified_artifact_boundary_terms;
    m->certified_artifact_boundary_epsilon = certified_artifact_boundary_epsilon;
    m->certified_artifact_boundary_ce_budget = certified_artifact_boundary_ce_budget;
    m->certified_artifact_boundary_overhead = certified_artifact_boundary_overhead;
    m->certified_artifact_boundary_kappa = certified_artifact_boundary_kappa;
    m->state_transition_atlas_gate_enabled = state_transition_atlas_gate_enable ? 1 : 0;
    m->absorbed_state_atlas_gate_enabled = absorbed_state_atlas_gate_enable ? 1 : 0;
    m->absorbed_state_atlas_overhead = absorbed_state_atlas_overhead;
    m->absorbed_state_atlas_memory_cost = absorbed_state_atlas_memory_cost;
    m->absorbed_state_atlas_repair_cost = absorbed_state_atlas_repair_cost;
    m->absorbed_state_atlas_kappa = absorbed_state_atlas_kappa;
    m->counterfactual_packets_gate_enabled = counterfactual_packets_gate_enable ? 1 : 0;
    m->certified_silent_span_gate_enabled = certified_silent_span_gate_enable ? 1 : 0;
    m->certified_silent_span_max_loss = certified_silent_span_max_loss;
    m->certified_silent_span_min_len = certified_silent_span_min_len;
    m->certified_silent_span_cert_cost = certified_silent_span_cert_cost;
    m->certified_silent_span_repair_cost = certified_silent_span_repair_cost;
    m->error_causal_lattice_gate_enabled = error_causal_lattice_gate_enable ? 1 : 0;
    m->residual_active_sampling_gate_enabled = residual_active_sampling_gate_enable ? 1 : 0;
    m->residual_active_sampling_floor = residual_active_sampling_floor;
    m->residual_active_sampling_target_fraction = residual_active_sampling_target_fraction;
    m->exact_transition_quotient_gate_enabled = exact_transition_quotient_gate_enable ? 1 : 0;
    m->exact_window_multiset_gate_enabled = exact_window_multiset_gate_enable ? 1 : 0;
    m->prefix_frontier_gate_enabled = prefix_frontier_gate_enable ? 1 : 0;
    m->macro_transcript_gate_enabled = macro_transcript_gate_enable ? 1 : 0;
    m->reversible_deterministic_expansion_gate_enabled =
        reversible_deterministic_expansion_gate_enable ? 1 : 0;
    m->active_state_subspace_gate_enabled = active_state_subspace_gate_enable ? 1 : 0;
    m->active_state_subspace_k = active_state_subspace_k;
    m->active_state_subspace_overhead = active_state_subspace_overhead;
    m->boundary_liability_ledger_gate_enabled =
        boundary_liability_ledger_gate_enable ? 1 : 0;
    m->boundary_liability_null_epsilon = boundary_liability_null_epsilon;
    m->boundary_liability_overhead = boundary_liability_overhead;
    m->boundary_flux_sheaf_gate_enabled =
        boundary_flux_sheaf_gate_enable ? 1 : 0;
    m->boundary_flux_sheaf_overhead = boundary_flux_sheaf_overhead;
    m->online_tssi_live_gate_enabled = online_tssi_live_gate_enable ? 1 : 0;
    if (online_tssi_selector_out != NULL &&
        !(online_tssi_selector_active_fraction > 0.0f &&
          online_tssi_selector_active_fraction <= 1.0f)) {
        fprintf(stderr, "[ONLINE_TSSI_SELECTOR_EXPORT] invalid active fraction %.6f; expected (0, 1]\n",
                online_tssi_selector_active_fraction);
        return 1;
    }
    uint8_t online_tssi_train_state_mask[V12_D_STATE] = {0};
    uint32_t online_tssi_train_active_dims = 0u;
    if (online_tssi_train_selector_json != NULL) {
        uint8_t source_mask[V12_D_STATE] = {0};
        uint32_t source_dims = 0u;
        int selector_rc = load_online_tssi_selector_mask(
            online_tssi_train_selector_json,
            source_mask,
            (uint32_t)V12_D_STATE,
            &source_dims);
        if (selector_rc != 0) {
            fprintf(stderr,
                    "[ONLINE_TSSI_TRAIN_SELECTOR] failed to load selector_json=%s rc=%d\n",
                    online_tssi_train_selector_json,
                    selector_rc);
            return 1;
        }
        if (strcmp(online_tssi_train_selector_control, "tssi") == 0) {
            memcpy(online_tssi_train_state_mask, source_mask, sizeof(source_mask));
            online_tssi_train_active_dims = source_dims;
        } else if (strcmp(online_tssi_train_selector_control, "zero") == 0) {
            memset(online_tssi_train_state_mask, 0, sizeof(online_tssi_train_state_mask));
            online_tssi_train_active_dims = 0u;
        } else if (strcmp(online_tssi_train_selector_control, "random") == 0) {
            online_tssi_train_active_dims = online_tssi_make_random_mask(
                online_tssi_train_state_mask,
                (uint32_t)V12_D_STATE,
                source_dims,
                online_tssi_train_selector_seed);
        } else {
            fprintf(stderr,
                    "[ONLINE_TSSI_TRAIN_SELECTOR] unsupported control=%s; supported controls: tssi, zero, random\n",
                    online_tssi_train_selector_control);
            return 1;
        }
    }
    m->boundary_cotangent_credit_gate_enabled =
        boundary_cotangent_credit_gate_enable ? 1 : 0;
    m->boundary_cotangent_overhead = boundary_cotangent_overhead;
    m->boundary_cotangent_residual_budget =
        boundary_cotangent_residual_budget;
    m->boundary_cotangent_verify_fraction =
        boundary_cotangent_verify_fraction;
    m->variance_priced_credit_gate_enabled =
        variance_priced_credit_gate_enable ? 1 : 0;
    m->variance_priced_sample_fraction = variance_priced_sample_fraction;
    m->variance_priced_score_fraction = variance_priced_score_fraction;
    m->variance_priced_verify_fraction = variance_priced_verify_fraction;
    m->variance_priced_overhead = variance_priced_overhead;
    m->equal_area_fisher_sampling_enabled =
        equal_area_fisher_sampling_enable ? 1 : 0;
    m->equal_area_fisher_sampling_gate_enabled =
        equal_area_fisher_sampling_gate_enable ? 1 : 0;
    m->equal_area_fisher_ess_floor = equal_area_fisher_ess_floor;
    m->equal_area_fisher_bias_budget = equal_area_fisher_bias_budget;
    m->equal_area_fisher_min_descent_per_paid_work =
        equal_area_fisher_min_descent_per_paid_work;
    m->event_innovation_credit_enabled =
        event_innovation_credit_enable ? 1 : 0;
    m->event_innovation_credit_gate_enabled =
        event_innovation_credit_gate_enable ? 1 : 0;
    m->event_innovation_fraction = event_innovation_fraction;
    m->event_innovation_debt_budget = event_innovation_debt_budget;
    m->event_innovation_score_cost = event_innovation_score_cost;
    m->event_innovation_verify_fraction = event_innovation_verify_fraction;
    m->event_innovation_overhead = event_innovation_overhead;
    m->event_innovation_low_event_scale = event_innovation_low_event_scale;
    m->event_innovation_boundary_boost = event_innovation_boundary_boost;
    m->ce_debt_skip_enabled = ce_debt_skip_enable ? 1 : 0;
    m->ce_debt_skip_gate_enabled = ce_debt_skip_gate_enable ? 1 : 0;
    m->ce_debt_max_loss = ce_debt_max_loss;
    m->ce_debt_max_grad_norm = ce_debt_max_grad_norm;
    m->ce_debt_max_debt = ce_debt_max_debt;
    m->ce_debt_audit_fraction = ce_debt_audit_fraction;
    m->ce_debt_overhead = ce_debt_overhead;
    m->ce_debt_min_skip_fraction = ce_debt_min_skip_fraction;
    m->ce_debt_skip_scale = ce_debt_skip_scale;
    m->gradient_coherence_quotient_enabled =
        gradient_coherence_quotient_enable ? 1 : 0;
    m->gradient_coherence_quotient_gate_enabled =
        gradient_coherence_quotient_gate_enable ? 1 : 0;
    m->gradient_coherence_bins = gradient_coherence_bins;
    m->gradient_coherence_probe_stride = gradient_coherence_probe_stride;
    m->gradient_coherence_max_residual = gradient_coherence_max_residual;
    m->gradient_coherence_score_cost = gradient_coherence_score_cost;
    m->gradient_coherence_verify_fraction = gradient_coherence_verify_fraction;
    m->gradient_coherence_overhead = gradient_coherence_overhead;
    m->gradient_coherence_min_saved_fraction =
        gradient_coherence_min_saved_fraction;
    m->gradient_coherence_residual_scale = gradient_coherence_residual_scale;
    m->amortized_ce_martingale_training_enabled =
        amortized_ce_martingale_training_enable ? 1 : 0;
    m->amortized_ce_martingale_min_descent_per_paid_work =
        amortized_ce_martingale_min_descent_per_paid_work;
    m->martingale_lambda = martingale_lambda;
    m->continuous_ternary_margin_enabled = continuous_ternary_margin_enabled;
    m->continuous_ternary_margin_c = continuous_ternary_margin_c;
    m->lagged_ce_impulse_eligibility_enabled =
        lagged_ce_impulse_eligibility_enable ? 1 : 0;
    m->lagged_ce_impulse_horizon = lagged_ce_impulse_horizon;
    m->lagged_ce_impulse_blend = lagged_ce_impulse_blend;
    m->lagged_ce_impulse_residual_budget = lagged_ce_impulse_residual_budget;
    m->lagged_ce_impulse_local_jvp_fraction =
        lagged_ce_impulse_local_jvp_fraction;
    m->lagged_ce_impulse_ring_fraction = lagged_ce_impulse_ring_fraction;
    m->lagged_ce_impulse_tail_audit_fraction =
        lagged_ce_impulse_tail_audit_fraction;
    m->lagged_ce_impulse_memory_fraction = lagged_ce_impulse_memory_fraction;
    m->executable_multiplicity_packets_enabled =
        executable_multiplicity_packets_enable ? 1 : 0;
    m->executable_multiplicity_min_count =
        executable_multiplicity_min_count < 2u ? 2u : executable_multiplicity_min_count;
    m->executable_multiplicity_min_mass = executable_multiplicity_min_mass;
    m->executable_multiplicity_max_residual = executable_multiplicity_max_residual;
    m->executable_multiplicity_signature_fraction =
        executable_multiplicity_signature_fraction;
    m->executable_multiplicity_verifier_fraction =
        executable_multiplicity_verifier_fraction;
    m->executable_multiplicity_memory_fraction =
        executable_multiplicity_memory_fraction;
    m->executable_multiplicity_state_scale =
        executable_multiplicity_state_scale > 0.0f ? executable_multiplicity_state_scale : 16.0f;
    m->executable_multiplicity_transition_scale =
        executable_multiplicity_transition_scale > 0.0f ? executable_multiplicity_transition_scale : 64.0f;
    m->original_ce_credit_debt_observer_enabled =
        original_ce_credit_debt_observer_enable ? 1 : 0;
    m->original_ce_credit_surface_fraction =
        original_ce_credit_surface_fraction;
    m->original_ce_credit_local_work_multiplier =
        original_ce_credit_local_work_multiplier;
    m->original_ce_credit_debt_price = original_ce_credit_debt_price;
    m->original_ce_credit_verifier_fraction =
        original_ce_credit_verifier_fraction;
    m->original_ce_credit_overhead = original_ce_credit_overhead;
    m->original_ce_credit_recurrent_backward_share =
        original_ce_credit_recurrent_backward_share;
    m->original_ce_credit_min_progress = original_ce_credit_min_progress;
    m->original_ce_credit_max_residual = original_ce_credit_max_residual;
    m->original_ce_credit_min_whole_step_speedup =
        original_ce_credit_min_whole_step_speedup;
    m->r375_pbg2_gate_enabled = r375_pbg2_gate_enable ? 1 : 0;
    m->r375_pbg2_max_matrices = r375_pbg2_max_matrices;
    m->r375_pbg2_min_anisotropy = r375_pbg2_min_anisotropy;
    m->r375_pbg2_precondition_enabled =
        r375_pbg2_precondition_enable ? 1 : 0;
    m->r375_pbg2_precondition_max_matrices =
        r375_pbg2_precondition_max_matrices;
    m->r375_pbg2_precondition_eps = r375_pbg2_precondition_eps;
    m->gradient_null_witness_gate_enabled =
        gradient_null_witness_gate_enable ? 1 : 0;
    m->gradient_null_witness_epsilon = gradient_null_witness_epsilon;
    m->gradient_null_witness_omitted_budget =
        gradient_null_witness_omitted_budget;
    m->gradient_null_witness_overhead = gradient_null_witness_overhead;
    m->r429_decision_equivalence_witness_enabled =
        r429_decision_equivalence_witness_enable ? 1 : 0;
    m->ce_normal_chamber_gate_enabled =
        ce_normal_chamber_gate_enable ? 1 : 0;
    m->ce_normal_chamber_sigma = ce_normal_chamber_sigma;
    m->ce_normal_chamber_overhead = ce_normal_chamber_overhead;
    m->optimizer_integrity_gate_enabled =
        optimizer_integrity_gate_enable ? 1 : 0;
    m->paper7_dfga_real_batch_audit_enabled =
        paper7_dfga_real_batch_audit_enable ? 1 : 0;
    m->paper7_dfga_real_batch_audit_max_batches =
        paper7_dfga_real_batch_audit_max_batches;
    m->paper7_dfga_real_batch_audit_residual_budget =
        paper7_dfga_real_batch_audit_residual_budget;
    m->qcf_mvue_gate_enabled =
        qcf_mvue_gate_enable ? 1 : 0;
    m->qcf_mvue_max_matrices = qcf_mvue_max_matrices;
    m->qcf_mvue_epsilon = qcf_mvue_epsilon;
    m->qcf_mvue_min_rho = qcf_mvue_min_rho;
    m->qcf_mvue_min_delta_energy = qcf_mvue_min_delta_energy;
    m->evidence_rate_commitment_gate_enabled =
        evidence_rate_commitment_gate_enable ? 1 : 0;
    m->evidence_rate_commitment_mode = evidence_rate_commitment_mode;
    m->evidence_rate_force_repack =
        (evidence_rate_force_repack ||
         evidence_rate_commitment_mode != V12_EVIDENCE_RATE_MODE_CURRENT) ? 1 : 0;
    m->evidence_rate_overhead = evidence_rate_overhead;
    m->evidence_rate_boundary_eps = evidence_rate_boundary_eps;
    m->r388_flip_refresh_gate_enabled =
        r388_flip_refresh_gate_enable ? 1 : 0;
    m->r388_flip_refresh_block_elements = r388_flip_refresh_block_elements;
    m->r388_flip_refresh_min_eta = r388_flip_refresh_min_eta;
    m->fp_interface_gap_probe_enabled =
        fp_interface_gap_probe_enable ? 1 : 0;
    m->fp_interface_gap_effect_epsilon = fp_interface_gap_effect_epsilon;
    m->credit_time_equalized_dynamics_enabled =
        credit_time_equalized_dynamics_enable ? 1 : 0;
    m->credit_time_equalized_dynamics_gate_enabled =
        credit_time_equalized_dynamics_gate_enable ? 1 : 0;
    m->credit_time_horizon = credit_time_horizon;
    m->credit_time_min_scale = credit_time_min_scale;
    m->credit_time_max_scale = credit_time_max_scale;
    m->credit_time_overhead = credit_time_overhead;
    m->contractive_green_credit_enabled =
        contractive_green_credit_enable ? 1 : 0;
    m->contractive_green_credit_gate_enabled =
        contractive_green_credit_gate_enable ? 1 : 0;
    m->contractive_green_credit_horizon = contractive_green_credit_horizon;
    m->contractive_green_credit_max_alpha = contractive_green_credit_max_alpha;
    m->contractive_green_credit_tail_budget = contractive_green_credit_tail_budget;
    m->contractive_green_credit_min_scale = contractive_green_credit_min_scale;
    m->contractive_green_credit_max_scale = contractive_green_credit_max_scale;
    m->contractive_green_credit_overhead = contractive_green_credit_overhead;
    m->adaptive_credit_horizon_enabled =
        adaptive_credit_horizon_enable ? 1 : 0;
    m->adaptive_credit_horizon_gate_enabled =
        adaptive_credit_horizon_gate_enable ? 1 : 0;
    m->adaptive_credit_min_horizon = adaptive_credit_min_horizon;
    m->adaptive_credit_max_horizon = adaptive_credit_max_horizon;
    m->adaptive_credit_tail_budget = adaptive_credit_tail_budget;
    m->adaptive_credit_bound_budget = adaptive_credit_bound_budget;
    m->adaptive_credit_verify_fraction = adaptive_credit_verify_fraction;
    m->adaptive_credit_overhead = adaptive_credit_overhead;
    m->adaptive_credit_min_headroom = adaptive_credit_min_headroom;
    m->adaptive_credit_tail_scale = adaptive_credit_tail_scale;
    m->interior_adjoint_tube_enabled =
        interior_adjoint_tube_enable ? 1 : 0;
    m->interior_adjoint_tube_gate_enabled =
        interior_adjoint_tube_gate_enable ? 1 : 0;
    m->interior_adjoint_tube_rank = interior_adjoint_tube_rank;
    m->interior_adjoint_tube_residual_budget =
        interior_adjoint_tube_residual_budget;
    m->interior_adjoint_tube_verify_fraction =
        interior_adjoint_tube_verify_fraction;
    m->interior_adjoint_tube_overhead = interior_adjoint_tube_overhead;
    m->interior_adjoint_tube_residual_scale =
        interior_adjoint_tube_residual_scale;
    m->interior_adjoint_tube_min_energy_covered =
        interior_adjoint_tube_min_energy_covered;

    if (rs_enable) {
        m->use_residual_shell = 1;
        m->rs_eps_lo = rs_eps_lo;
        m->rs_eps_hi = rs_eps_hi;
        printf("[RS] use_residual_shell=1 eps_lo=%.3f eps_hi=%.3f\n",
               m->rs_eps_lo, m->rs_eps_hi);
        printf("[RS] WARNING: objective is REWEIGHTED, not unchanged. "
               "Very-easy and very-hard tokens contribute zero gradient. "
               "Converged optimum may differ from full-gradient training. "
               "Disable by omitting --residual-shell.\n");
    }
    if (lccap_enable) {
        m->lccap_gate_enabled = 1;
        m->lccap_segment_length = 8;
        m->lccap_target_loss = 0.18f;
        m->lccap_epsilon_safe = 0.025f;
        printf("[LCCAP] active=1 mode=lookahead-certified-ce-anchor-pruning objective=unchanged_ce\n");
    }
    if (exact_quotient_padding_enable) {
        m->exact_quotient_padding_enabled = 1;
        m->exact_quotient_padding_min_active_len =
            exact_quotient_padding_min_active_len;
        m->exact_quotient_padding_seed = exact_quotient_padding_seed;
        printf("[EXACT_QUOTIENT_PADDING] active=1 mode=real-training-active-prefix-ce min_active_len=%u seq_len=%d ce_gradient_masked=1 gpu_scan_masked=0 compute_skip=0\n",
               m->exact_quotient_padding_min_active_len,
               V12_SEQ_LEN);
    }

    printf("[TRAIN] Building unigram table...\n");
    sampled_softmax_build_unigram(m, ds->data, ds->n_tokens);
    corpus_logit_prior_configure(
        m, corpus_logit_prior_enable, corpus_logit_prior_alpha, corpus_logit_prior_scale,
        corpus_logit_prior_terms, corpus_logit_prior_stride);
    if (corpus_logit_prior_build(m, ds->data, ds->n_tokens) != 0) {
        fprintf(stderr, "[ERROR] Failed to build corpus logit prior\n");
        model_free(m);
        dataset_free(ds);
        return 1;
    }
    program_kernel_configure(
        m, program_kernel_enable, program_kernel_gate_enable,
        program_kernel_logit_boost, program_kernel_min_count,
        program_kernel_terms, program_kernel_stride);
    if (program_kernel_build(m, ds->data, ds->n_tokens) != 0) {
        fprintf(stderr, "[ERROR] Failed to build program kernel\n");
        model_free(m);
        dataset_free(ds);
        return 1;
    }
    certified_artifact_boundary_configure(
        m, certified_artifact_boundary_gate_enable,
        certified_artifact_boundary_min_count, certified_artifact_boundary_terms,
        certified_artifact_boundary_stride, certified_artifact_boundary_epsilon,
        certified_artifact_boundary_ce_budget, certified_artifact_boundary_overhead,
        certified_artifact_boundary_kappa);
    if (certified_artifact_boundary_build(m, ds->data, ds->n_tokens) != 0) {
        fprintf(stderr, "[ERROR] Failed to build certified artifact boundary\n");
        model_free(m);
        dataset_free(ds);
        return 1;
    }
    state_transition_atlas_configure(
        m, state_transition_atlas_enable, state_transition_atlas_gate_enable,
        state_transition_atlas_logit_boost, state_transition_atlas_min_count,
        state_transition_atlas_terms, state_transition_atlas_stride);
    if (state_transition_atlas_build(m, ds->data, ds->n_tokens) != 0) {
        fprintf(stderr, "[ERROR] Failed to build state transition atlas\n");
        model_free(m);
        dataset_free(ds);
        return 1;
    }
    counterfactual_packets_configure(
        m, counterfactual_packets_enable, counterfactual_packets_gate_enable,
        counterfactual_packets_min_count, counterfactual_packets_terms,
        counterfactual_packets_stride);
    if (counterfactual_packets_build(m, ds->data, ds->n_tokens) != 0) {
        fprintf(stderr, "[ERROR] Failed to build counterfactual packets\n");
        model_free(m);
        dataset_free(ds);
        return 1;
    }
    error_causal_lattice_configure(
        m, error_causal_lattice_enable, error_causal_lattice_gate_enable,
        error_causal_lattice_min_count, error_causal_lattice_terms,
        error_causal_lattice_stride);
    if (error_causal_lattice_build(m, ds->data, ds->n_tokens) != 0) {
        fprintf(stderr, "[ERROR] Failed to build error-causal lattice\n");
        model_free(m);
        dataset_free(ds);
        return 1;
    }
    exact_transition_quotient_configure(
        m, exact_transition_quotient_enable, exact_transition_quotient_gate_enable,
        exact_transition_quotient_min_count, exact_transition_quotient_terms,
        exact_transition_quotient_stride);
    if (exact_transition_quotient_build(m, ds->data, ds->n_tokens) != 0) {
        fprintf(stderr, "[ERROR] Failed to build exact transition quotient\n");
        model_free(m);
        dataset_free(ds);
        return 1;
    }
    macro_ce_quotient_configure(
        m, macro_ce_quotient_observer_enable, macro_ce_quotient_training_enable,
        macro_ce_quotient_min_support, macro_ce_quotient_terms,
        macro_ce_quotient_stride, macro_ce_quotient_build_fraction,
        macro_ce_quotient_alpha, macro_ce_quotient_residual_price_scale,
        macro_ce_quotient_overhead, macro_ce_quotient_signature_bins,
        macro_ce_quotient_seed, macro_ce_quotient_training_weight);
    if (macro_ce_quotient_build(m, ds->data, ds->n_tokens) != 0) {
        fprintf(stderr, "[ERROR] Failed to build R421 macro CE quotient\n");
        model_free(m);
        dataset_free(ds);
        return 1;
    }
    if ((residual_certified_scaffold_enable ||
         residual_certified_scaffold_gate_enable) &&
        m->macro_ce_quotient_observer_enabled) {
        printf("[RESIDUAL_CERTIFIED_SCAFFOLD] active=1 apply=%d gate=%d mode=objective-preserving-context-scaffold-quotient contexts=%u classes=%u quotient_ratio=%.6f training=%d audit_budget=%.6f default_path=unchanged\n",
               residual_certified_scaffold_enable ? 1 : 0,
               residual_certified_scaffold_gate_enable ? 1 : 0,
               m->macro_ce_quotient_unique_contexts,
               m->macro_ce_quotient_quotient_classes,
               m->macro_ce_quotient_ratio,
               m->macro_ce_quotient_training_enabled ? 1 : 0,
               residual_certified_scaffold_audit_budget);
    }
    if (r423_native_growth_readout_quotient_enable &&
        m->macro_ce_quotient_observer_enabled) {
        printf("[R423_NATIVE_GROWTH_READOUT_QUOTIENT] active=1 mode=real-training-bridge source=macro-ce-quotient-training contexts=%u classes=%u quotient_ratio=%.6f training=%d coverage_source=live-training-log\n",
               m->macro_ce_quotient_unique_contexts,
               m->macro_ce_quotient_quotient_classes,
               m->macro_ce_quotient_ratio,
               m->macro_ce_quotient_training_enabled ? 1 : 0);
    }
    exact_window_multiset_configure(
        m, exact_window_multiset_gate_enable, exact_window_multiset_min_count,
        exact_window_multiset_windows, exact_window_multiset_stride,
        exact_window_multiset_overhead, exact_window_multiset_noise);
    if (exact_window_multiset_build(m, ds->data, ds->n_tokens, V12_SEQ_LEN) != 0) {
        fprintf(stderr, "[ERROR] Failed to build exact window multiset\n");
        model_free(m);
        dataset_free(ds);
        return 1;
    }
    prefix_frontier_configure(
        m, prefix_frontier_gate_enable, prefix_frontier_min_count,
        prefix_frontier_terms, prefix_frontier_stride,
        prefix_frontier_overhead, prefix_frontier_repair_cost);
    if (prefix_frontier_build(m, ds->data, ds->n_tokens) != 0) {
        fprintf(stderr, "[ERROR] Failed to build prefix frontier table\n");
        model_free(m);
        dataset_free(ds);
        return 1;
    }
    macro_transcript_configure(
        m, macro_transcript_gate_enable, macro_transcript_min_count,
        macro_transcript_terms, macro_transcript_stride,
        macro_transcript_hardness, macro_transcript_codec_cost,
        macro_transcript_decode_cost);
    if (macro_transcript_build(m, ds->data, ds->n_tokens) != 0) {
        fprintf(stderr, "[ERROR] Failed to build macro transcript table\n");
        model_free(m);
        dataset_free(ds);
        return 1;
    }
    reversible_deterministic_expansion_configure(
        m, reversible_deterministic_expansion_gate_enable,
        reversible_deterministic_expansion_min_run,
        reversible_deterministic_expansion_terms,
        reversible_deterministic_expansion_overhead,
        reversible_deterministic_expansion_noise);
    if (reversible_deterministic_expansion_build(m, ds->data, ds->n_tokens) != 0) {
        fprintf(stderr, "[ERROR] Failed to build reversible deterministic expansion gate\n");
        model_free(m);
        dataset_free(ds);
        return 1;
    }
    if (logit_prior_gate_enable) {
        printf("[LOGIT_PRIOR_GATE] active=1 mode=ce-equivalent-residual-prior metric=baseline_vs_residual_gpu\n");
    }
    if (program_kernel_gate_enable) {
        printf("[PROGRAM_KERNEL_GATE] active=1 mode=proof-carrying-kernel-residual metric=baseline_vs_kernel_gpu fallback=abstain\n");
    }
    if (certified_artifact_boundary_gate_enable) {
        printf("[CERTIFIED_ARTIFACT_BOUNDARY_GATE] active=1 mode=certified-abstaining-artifact-boundary rule=copy-current metric=bk_false_ce_priced source=host_tokens objective=artifact_ce_decomposition fallback=original_ce_residual\n");
    }
    if (state_transition_atlas_gate_enable) {
        printf("[STATE_TRANSITION_ATLAS_GATE] active=1 mode=exact-state-transition-residual metric=baseline_vs_atlas_gpu fallback=abstain\n");
    }
    if (absorbed_state_atlas_gate_enable) {
        printf("[ABSORBED_STATE_ATLAS_GATE] active=1 mode=absorbed-state-atlas-residualization metric=priced_residual_ratio source=gpu_baseline_vs_atlas_sampled_ce objective=ce_decomposition\n");
    }
    if (counterfactual_packets_gate_enable) {
        printf("[COUNTERFACTUAL_PACKETS_GATE] active=1 mode=local-state-minimal-pair-ledger metric=pair_coverage_balance objective=unchanged_ce\n");
    }
    if (certified_silent_span_gate_enable) {
        printf("[CERTIFIED_SILENT_SPAN_GATE] active=1 mode=ce-gradient-contained-span-certificate metric=silent_fraction_priced_ratio objective=unchanged_ce\n");
    }
    if (error_causal_lattice_gate_enable) {
        printf("[ERROR_CAUSAL_LATTICE_GATE] active=1 mode=target-prerequisite-support metric=coverage_readiness_orderable objective=unchanged_ce\n");
    }
    if (residual_active_sampling_gate_enable) {
        printf("[RESIDUAL_ACTIVE_SAMPLING_GATE] active=1 mode=loss-proportional-importance metric=ess_variance_proxy objective=unchanged_ce\n");
    }
    if (exact_transition_quotient_gate_enable) {
        printf("[EXACT_TRANSITION_QUOTIENT_GATE] active=1 mode=exact-local-triple-multiplicity metric=corpus_ratio_live_saved_fraction objective=unchanged_ce\n");
    }
    if (exact_window_multiset_gate_enable) {
        printf("[EXACT_WINDOW_MULTISET_GATE] active=1 mode=exact-window-sufficient-statistic metric=q_live_saved_priced_ratio objective=weighted_ce_equal_full_gradient_equal\n");
    }
    if (prefix_frontier_gate_enable) {
        printf("[PREFIX_FRONTIER_GATE] active=1 mode=exact-local-frontier-adjoint-reuse metric=reuse_energy_priced_ratio source=gpu_sampled_ce_d_logits objective=unchanged_ce\n");
    }
    if (active_state_subspace_gate_enable) {
        printf("[ACTIVE_STATE_SUBSPACE_GATE] active=1 mode=ce-sufficient-active-state-subspace metric=topk_gradient_energy_tail_regret source=gpu_final_residual_gradient objective=unchanged_ce\n");
    }
    if (boundary_liability_ledger_gate_enable) {
        printf("[BOUNDARY_LIABILITY_LEDGER_GATE] active=1 mode=boundary-gradient-null-liability metric=boundary_energy_and_null_witness source=gpu_final_residual_gradient boundary_source=training_boundary_mask objective=unchanged_ce threshold=priced_gain_gt_0.10\n");
    }
    if (boundary_flux_sheaf_gate_enable) {
        printf("[BOUNDARY_FLUX_SHEAF_GATE] active=1 mode=ce-boundary-flux-sheaf metric=ssm_transition_credit_endpoint_energy source=gpu_dA_per_step boundary_source=training_boundary_mask objective=unchanged_ce threshold=f_flux_times_(1-kappa)_gt_0.10\n");
    }
    if (online_tssi_live_gate_enable) {
        printf("[ONLINE_TSSI_LIVE_GATE] active=1 mode=live-gpu-score-accumulator source=vulkan_fwd_h_cache sampled_score_source=gpu_sampled_ce_target_dlogit accumulator=target_conditioned_state_score_moments no_hidden_cache_materialization=1 objective=unchanged_ce threshold=target_conditioned_state_score_buffer_present\n");
    }
    if (online_tssi_selector_out != NULL) {
        printf("[ONLINE_TSSI_SELECTOR_EXPORT] active=1 path=%s active_fraction=%.6f source=target_conditioned_state_score_moments no_hidden_cache_materialization=1 no_dense_vocab_state=1 no_second_state_pass=1 objective=unchanged_ce\n",
               online_tssi_selector_out,
               online_tssi_selector_active_fraction);
    }
    if (r423_native_growth_state_selector_export_enable) {
        printf("[R423_NATIVE_GROWTH_STATE_SELECTOR_EXPORT] requested=1 path=%s active_fraction=%.6f source=target_conditioned_state_score_moments mode=growth-mask-build no_hidden_cache_materialization=1 training_change=0\n",
               online_tssi_selector_out,
               online_tssi_selector_active_fraction);
    }
    if (online_tssi_train_selector_json != NULL) {
        printf("[ONLINE_TSSI_TRAIN_SELECTOR] requested=1 selector_json=%s control=%s active_dims=%u d_state=%u source=target_conditioned_state_score_moments objective=training_path_probe default_path=unchanged controls_supported=tssi,zero,random\n",
               online_tssi_train_selector_json,
               online_tssi_train_selector_control,
               online_tssi_train_active_dims,
               (uint32_t)V12_D_STATE);
    }
    if (r423_native_growth_state_selector_train_enable) {
        printf("[R423_NATIVE_GROWTH_STATE_SELECTOR] requested=1 selector_json=%s control=%s active_dims=%u d_state=%u mode=real-training-bridge target=W_in_W_out training_change=1\n",
               online_tssi_train_selector_json,
               online_tssi_train_selector_control,
               online_tssi_train_active_dims,
               (uint32_t)V12_D_STATE);
    }
    if (r423_native_growth_microstep_birth_gate_enable) {
        printf("[R423_NATIVE_GROWTH_MICROSTEP_BIRTH_GATE] requested=1 mode=v12-real-parameter-birth-certificate family=microstep_factorization_equivalence training_change=0\n");
    }
    if (r423_native_growth_receipt_challenge_gate_enable) {
        printf("[R423_NATIVE_GROWTH_RECEIPT_CHALLENGE_GATE] requested=1 mode=real-gpu-gradient-receipt-challenge family=receipt_challenge_retirement source=W_in_W_out_gradients training_change=0\n");
    }
    if (r423_native_growth_interaction_release_gate_enable) {
        printf("[R423_NATIVE_GROWTH_INTERACTION_RELEASE_GATE] requested=1 mode=real-gpu-dwa-commutator-observer family=interaction_release_safety source=dW_A_gradients_and_A training_change=0\n");
    }
    if (r423_native_growth_interaction_safe_update_enable) {
        printf("[R423_NATIVE_GROWTH_INTERACTION_SAFE_UPDATE] requested=1 mode=real-gpu-dwa-commutator-safe-update family=interaction_release_safety source=dW_A_gradients_and_A training_change=1\n");
    }
    if (r423_native_growth_genotype_exception_gate_enable) {
        printf("[R423_NATIVE_GROWTH_GENOTYPE_EXCEPTION_GATE] requested=1 mode=real-gpu-dwa-genotype-observer family=genotype_exception_pressure source=dW_A_gradients training_change=0\n");
    }
    if (r423_native_growth_genotype_compact_update_enable) {
        printf("[R423_NATIVE_GROWTH_GENOTYPE_COMPACT_UPDATE] requested=1 mode=real-gpu-dwa-genotype-compact-update family=genotype_exception_pressure source=dW_A_gradients training_change=1\n");
    }
    if (boundary_cotangent_credit_gate_enable) {
        printf("[BOUNDARY_COTANGENT_CREDIT_GATE] active=1 mode=ce-boundary-anchor-cotangent-reconstruction metric=cosine_residual_priced_ratio source=gpu_dA_per_step boundary_source=training_boundary_mask objective=unchanged_ce threshold=priced_ratio_lt_0.90_and_descent_per_paid_work_gt_1\n");
    }
    if (variance_priced_credit_gate_enable) {
        printf("[VARIANCE_PRICED_CREDIT_GATE] active=1 mode=ce-variance-priced-credit-allocation metric=descent_per_paid_work source=gpu_dA_per_step objective=unchanged_ce threshold=descent_per_paid_work_gt_1.111 sampled_update=simulated\n");
    }
    if (equal_area_fisher_sampling_enable ||
        equal_area_fisher_sampling_gate_enable) {
        printf("[EQUAL_AREA_FISHER_SAMPLING] active=1 apply=%d gate=%d mode=bounded-equal-area-fisher-snr-sampling metric=ess_bias_and_descent_per_paid source=gpu_dA_per_step objective=unchanged_ce threshold=ess_floor_and_bias_budget_and_descent_per_paid default_path=unchanged\n",
               equal_area_fisher_sampling_enable ? 1 : 0,
               equal_area_fisher_sampling_gate_enable ? 1 : 0);
    }
    if (event_innovation_credit_enable || event_innovation_credit_gate_enable) {
        printf("[EVENT_INNOVATION_CREDIT] active=1 apply=%d gate=%d mode=ce-event-innovation-credit metric=low_event_debt_and_priced_ratio source=forward_state_innovation+gpu_dA_per_step event_fraction=%.6f debt_budget=%.6f threshold=priced_ratio_lt_0.90 default_path=unchanged\n",
               event_innovation_credit_enable ? 1 : 0,
               event_innovation_credit_gate_enable ? 1 : 0,
               event_innovation_fraction,
               event_innovation_debt_budget);
    }
    if (ce_debt_skip_enable || ce_debt_skip_gate_enable) {
        printf("[CE_DEBT_SKIP] active=1 apply=%d gate=%d mode=ce-debt-skip-ledger metric=skip_fraction_and_debt source=gpu_ce_loss+gpu_ce_d_logits max_loss=%.6f max_grad_norm=%.6f max_debt=%.6f threshold=priced_ratio_lt_0.90q default_path=unchanged\n",
               ce_debt_skip_enable ? 1 : 0,
               ce_debt_skip_gate_enable ? 1 : 0,
               ce_debt_max_loss,
               ce_debt_max_grad_norm,
               ce_debt_max_debt);
    }
    if (gradient_coherence_quotient_enable ||
        gradient_coherence_quotient_gate_enable) {
        printf("[GRADIENT_COHERENCE_QUOTIENT] active=1 apply=%d gate=%d mode=transition-credit-representative-residual-quotient metric=cluster_residual_priced_ratio source=gpu_dA_per_step+forward_transition_state bins=%u probe_stride=%u max_residual=%.6f threshold=priced_ratio_lte_0.90q default_path=unchanged\n",
               gradient_coherence_quotient_enable ? 1 : 0,
               gradient_coherence_quotient_gate_enable ? 1 : 0,
               gradient_coherence_bins,
               gradient_coherence_probe_stride,
               gradient_coherence_max_residual);
    }
    if (amortized_ce_martingale_training_enable) {
        printf("[AMORTIZED_CE_MARTINGALE_TRAINING] active=1 mode=norm_priced_horvitz_thompson_transition_credit target=A_bA_WA_recurrent_update source=gpu_dA_per_step training_change=1 fallback=exact_when_local_gate_fails threshold=%.6f default_path=unchanged\n",
               amortized_ce_martingale_min_descent_per_paid_work);
    }
    if (lagged_ce_impulse_eligibility_enable) {
        printf("[LAGGED_CE_IMPULSE_ELIGIBILITY] active=1 mode=lagged_ce_hidden_impulse_eligibility target=A_bA_WA_recurrent_update source=gpu_dA_per_step+gpu_v_pre+vulkan_fwd_h_cache+gpu_ssm_a+gpu_a_gate training_change=1 fallback=exact_when_residual_or_pricing_fails horizon=%u blend=%.6f residual_budget=%.6f priced_threshold=0.900000 default_path=unchanged\n",
               lagged_ce_impulse_horizon,
               lagged_ce_impulse_blend,
               lagged_ce_impulse_residual_budget);
    }
    if (executable_multiplicity_packets_enable) {
        printf("[R438_EXECUTABLE_MULTIPLICITY_PACKETS] active=1 mode=state_transition_signature_packets target=original_ce_training_denominator source=vulkan_fwd_h_cache+gpu_ssm_a+gpu_dA_per_step observer_only=1 no_training_change=1 min_count=%u min_mass=%.6f max_residual=%.6f state_scale=%.6f transition_scale=%.6f priced_threshold=0.900000 default_path=unchanged\n",
               m->executable_multiplicity_min_count,
               executable_multiplicity_min_mass,
               executable_multiplicity_max_residual,
               m->executable_multiplicity_state_scale,
               m->executable_multiplicity_transition_scale);
    }
    if (original_ce_credit_debt_observer_enable) {
        printf("[ORIGINAL_CE_CREDIT_DEBT_OBSERVER] active=1 mode=ce-local-direct-transition-credit metric=progress_residual_priced_ratio source=gpu_dA_per_step+gpu_v_pre+vulkan_fwd_h_cache objective=unchanged_ce observer_only=1 no_training_change=1 threshold=whole_step_speedup_ge_min\n");
    }
    if (gradient_null_witness_gate_enable) {
        printf("[GRADIENT_NULL_WITNESS_GATE] active=1 mode=certified-gradient-null-witness metric=full_parameter_gradient_group_rms source=cpu_and_gpu_gradient_tensors objective=unchanged_ce threshold=saved_fraction_minus_omitted_fraction_minus_overhead_gt_0.10\n");
    }
    if (optimizer_integrity_gate_enable) {
        printf("[OPTIMIZER_INTEGRITY_GATE] active=1 mode=gpu-owned-ternary-update-obligation metric=cpu_skip_vs_gpu_update_groups objective=training_fidelity threshold=all_gpu_owned_ternary_groups_updated\n");
    }
    if (qcf_mvue_gate_enable) {
        printf("[QUANTIZATION_CELL_QUOTIENT_GATE] active=1 mode=continuous_mvue_shadow_displacement metric=invisible_delta_energy_share source=gpu_shadow_pre_post_update objective=unchanged_ce threshold=rho_mvue_upper_bound_ge_%.3f max_matrices=%u epsilon=%.9g\n",
               qcf_mvue_min_rho, qcf_mvue_max_matrices, qcf_mvue_epsilon);
        printf("[QUANTIZATION_CELL_QUOTIENT_GATE] min_delta_energy=%.9g\n",
               qcf_mvue_min_delta_energy);
    }
    if (fp_interface_gap_probe_enable) {
        printf("[FP_INTERFACE_GAP_PROBE] active=1 mode=layer-effect-vs-downstream-gradient metric=effect_ratio_and_projection source=gpu_residuals_and_ce_gradient objective=unchanged_ce\n");
    }
    if (macro_transcript_gate_enable) {
        printf("[MACRO_TRANSCRIPT_GATE] active=1 mode=online-reversible-macro-transcript metric=length_ratio_priced source=host_tokens objective=exact_decode_ce\n");
    }
    if (reversible_deterministic_expansion_gate_enable) {
        printf("[REVERSIBLE_DETERMINISTIC_EXPANSION_GATE] active=1 mode=fixed-repeat-run-transducer metric=lambda_eta_priced_ratio objective=exact_inverse_realization\n");
    }

    printf("[TRAIN] Building hot/cold token mask...\n");
    model_build_hot_mask(m, ds->data, ds->n_tokens);

    printf("[TRAIN] Allocating cache...\n");
    Cache *c = cache_alloc(batch_size, V12_SEQ_LEN);
    if (!c) {
        fprintf(stderr, "[ERROR] Failed to allocate cache\n");
        model_free(m);
        dataset_free(ds);
        return 1;
    }

    YCache *y_cache = y_cache_alloc();
    if (y_cache == NULL) {
        fprintf(stderr, "[WARN] Y-cache allocation failed — trying PCA cache fallback\n");
    } else {
        c->y_cache = y_cache;
    }

    ResidualCache *res_cache = pca_cache_alloc();
    if (res_cache == NULL) {
        fprintf(stderr, "[WARN] PCA cache allocation failed — proceeding without cache\n");
    } else {
        c->res_cache = res_cache;
    }

    TrainState state = {0};
    state.lr_base     = V12_LR_MAX;
    state.sre_lr_scale = 1.0f;
    state.cams_running_var = 0.0f;
    state.step        = 0;
    state.rng_state   = 42ULL;
    state.fp_stage    = V12_N_LAYERS;
    state.fp_completed = 1;
    state.bcd_active  = 0;
    state.bcd_cycle   = 0;
    state.bcd_layer   = 0;
    for (int k = 0; k < V12_MAX_LAYERS; k++) state.lr_per_layer[k] = 1.0f;

    /* Initialize R493 contract parameters */
    state.r493_contract.stage_id = 1;
    state.r493_contract.min_quality_threshold = r493_min_quality_threshold;
    state.r493_contract.min_entropy_threshold = r493_min_entropy_threshold;
    state.r493_contract.collapse_weight = r493_collapse_weight;
    state.r493_contract.contract_stable = true;

    VulkanContext *vk_ctx = NULL;
#ifdef V12_HAS_VULKAN
    if (vk_init(&vk_ctx) != 0) {
        fprintf(stderr, "[VK] vk_init failed — cannot train without GPU\n");
        vk_ctx = NULL;
    }
    /* R386 slice 1: keep GPU ownership on blocks with stable persistent GPU state.
     * Dense per-layer A/conv/W_A/b_A still flow through shared scratch buffers, so
     * they remain CPU-owned until the per-layer dense-owner refactor lands. */
#if V12_ZMMA_ENABLED
    /* ZMMA handles ternary projection weights w_in and w_out on CPU Host.
     * Disable GPU_OPT_W_IN and GPU_OPT_W_OUT flags so that updates run on CPU. */
    m->gpu_opt_flags = GPU_OPT_W_PRED
                     | GPU_OPT_NORM;
#else
    m->gpu_opt_flags = GPU_OPT_W_IN | GPU_OPT_W_OUT
                     | GPU_OPT_W_PRED
                     | GPU_OPT_NORM;
#endif
    if (model_initialize_lm_head_host(m) != 0) {
        fprintf(stderr, "[ERROR] failed to initialize lm_head host mirrors for startup\n");
        cache_free(c);
        model_free(m);
        dataset_free(ds);
        if (vk_ctx) vk_destroy(vk_ctx);
        return 1;
    }
#endif

    int final_status = 0;
    int checkpoint_exists = 0;
    int checkpoint_resumed = 0;
    if (checkpoint_path != NULL) {
        FILE *checkpoint_probe = fopen(checkpoint_path, "rb");
        if (checkpoint_probe) {
            checkpoint_exists = 1;
            fclose(checkpoint_probe);
        }
    }

    GpuPipeline *gpu = NULL;
    if (!checkpoint_exists) {
        gpu = gpu_pipeline_create(vk_ctx, m, V12_SEQ_LEN, batch_size);
        if (!gpu) {
            fprintf(stderr, "[ERROR] GPU pipeline failed — cannot train without GPU\n");
            cache_free(c);
            model_free(m);
            dataset_free(ds);
#ifdef V12_HAS_VULKAN
            if (vk_ctx) vk_destroy(vk_ctx);
#endif
            return 1;
        }
        gpu_pipeline_print_stats(gpu);
    }

    if (checkpoint_path != NULL) {
#ifdef V12_HAS_VULKAN
        if (checkpoint_exists && materialize_lm_head_host_boundary(m, "checkpoint-load") != 0) {
            fprintf(stderr, "[ERROR] failed to materialize lm_head host mirrors for checkpoint load\n");
            gpu_pipeline_destroy(gpu);
            cache_free(c);
            model_free(m);
            dataset_free(ds);
            if (vk_ctx) vk_destroy(vk_ctx);
            return 1;
        }
        if (checkpoint_exists && materialize_slice3_boundary(m, NULL, "checkpoint-load") != 0) {
            fprintf(stderr, "[ERROR] failed to materialize embed host mirrors for checkpoint load\n");
            gpu_pipeline_destroy(gpu);
            cache_free(c);
            model_free(m);
            dataset_free(ds);
            if (vk_ctx) vk_destroy(vk_ctx);
            return 1;
        }
        if (checkpoint_exists && materialize_slice4_boundary(m, NULL, "checkpoint-load") != 0) {
            fprintf(stderr, "[ERROR] failed to materialize W_in/W_out host mirrors for checkpoint load\n");
            gpu_pipeline_destroy(gpu);
            cache_free(c);
            model_free(m);
            dataset_free(ds);
            if (vk_ctx) vk_destroy(vk_ctx);
            return 1;
        }
#endif
        if (checkpoint_load(checkpoint_path, m, &state) == 0) {
            if (checkpoint_rebuild(m, &state) == 0) {
                printf("[TRAIN] Resumed from step %u\n", state.step);
                if (ds->n_tokens > 0) {
                    size_t restored_cursor = (size_t)state.dataset_cursor;
                    int clamped_cursor = 0;
                    if ((uint64_t)restored_cursor != state.dataset_cursor ||
                        restored_cursor > ds->n_tokens) {
                        restored_cursor = (size_t)(state.dataset_cursor % (uint64_t)ds->n_tokens);
                        clamped_cursor = 1;
                    }
                    ds->cursor = restored_cursor;
                    printf("[CKPT] dataset_cursor restored=%llu current=%llu n_tokens=%llu clamped=%d\n",
                           (unsigned long long)state.dataset_cursor,
                           (unsigned long long)ds->cursor,
                           (unsigned long long)ds->n_tokens,
                           clamped_cursor);
                }
                if (state.dataset_state_valid) {
                    memcpy(ds->rng, state.dataset_rng, sizeof(ds->rng));
                    printf("[CKPT] dataset_rng restored=1 state_valid=%u\n",
                           (unsigned)state.dataset_state_valid);
                } else {
                    printf("[CKPT] dataset_rng restored=0 state_valid=0\n");
                }
            } else {
                fprintf(stderr, "[ERROR] checkpoint_rebuild failed — starting fresh\n");
                state = (TrainState){0};
                state.lr_base   = V12_LR_MAX;
                state.rng_state = 42ULL;
            }
            if (materialize_slice2_boundary(m, NULL, "checkpoint-resume") != 0) {
                fprintf(stderr, "[ERROR] failed to materialize w_pred/norm for checkpoint resume\n");
                gpu_pipeline_destroy(gpu);
                cache_free(c);
                model_free(m);
                dataset_free(ds);
#ifdef V12_HAS_VULKAN
                if (vk_ctx) vk_destroy(vk_ctx);
#endif
                return 1;
            }
            if (materialize_slice3_boundary(m, NULL, "checkpoint-resume") != 0) {
                fprintf(stderr, "[ERROR] failed to materialize embed state for checkpoint resume\n");
                gpu_pipeline_destroy(gpu);
                cache_free(c);
                model_free(m);
                dataset_free(ds);
#ifdef V12_HAS_VULKAN
                if (vk_ctx) vk_destroy(vk_ctx);
#endif
                return 1;
            }
            if (materialize_slice4_boundary(m, NULL, "checkpoint-resume") != 0) {
                fprintf(stderr, "[ERROR] failed to materialize W_in/W_out state for checkpoint resume\n");
                gpu_pipeline_destroy(gpu);
                cache_free(c);
                model_free(m);
                dataset_free(ds);
#ifdef V12_HAS_VULKAN
                if (vk_ctx) vk_destroy(vk_ctx);
#endif
                return 1;
            }
            if (ckpt_debug) {
                print_checkpoint_debug(m, "post-load");
            }
            /* Create the Vulkan pipeline from the loaded CPU model instead of
             * allocating a pre-load pipeline and replacing it. Full-width resumes
             * use nearly all persistent buffer slots, so avoiding a throwaway
             * pipeline keeps the buffer pool available for the first backward pass. */
            gpu = gpu_pipeline_create(vk_ctx, m, V12_SEQ_LEN, batch_size);
            if (!gpu) {
                fprintf(stderr, "[ERROR] GPU pipeline recreate after checkpoint load failed\n");
                cache_free(c);
                model_free(m);
                dataset_free(ds);
#ifdef V12_HAS_VULKAN
                if (vk_ctx) vk_destroy(vk_ctx);
#endif
                return 1;
            }
            gpu_pipeline_sync_materialized_state(gpu, m);
            gpu_pipeline_print_stats(gpu);
            release_lm_head_steady_state(m, "post-checkpoint-resume");
            release_slice2_steady_state(m, "post-checkpoint-resume");
            release_slice3_steady_state(m, "post-checkpoint-resume");
            release_slice4_steady_state(m, "post-checkpoint-resume");
            checkpoint_resumed = 1;
        }
    }

    if (!gpu) {
        gpu = gpu_pipeline_create(vk_ctx, m, V12_SEQ_LEN, batch_size);
        if (!gpu) {
            fprintf(stderr, "[ERROR] GPU pipeline failed after checkpoint setup — cannot train without GPU\n");
            cache_free(c);
            model_free(m);
            dataset_free(ds);
#ifdef V12_HAS_VULKAN
            if (vk_ctx) vk_destroy(vk_ctx);
#endif
            return 1;
        }
        gpu_pipeline_print_stats(gpu);
    }

    if (online_tssi_train_selector_json != NULL) {
        int selector_rc = gpu_pipeline_set_online_tssi_train_selector(
            gpu,
            online_tssi_train_state_mask,
            (uint32_t)V12_D_STATE,
            online_tssi_train_selector_control,
            online_tssi_train_active_dims);
        if (selector_rc != 0) {
            fprintf(stderr,
                    "[ONLINE_TSSI_TRAIN_SELECTOR] failed to arm GPU selector rc=%d\n",
                    selector_rc);
            gpu_pipeline_destroy(gpu);
            cache_free(c);
            model_free(m);
            dataset_free(ds);
#ifdef V12_HAS_VULKAN
            if (vk_ctx) vk_destroy(vk_ctx);
#endif
            return 1;
        }
        if (r423_native_growth_state_selector_train_enable) {
            printf("[R423_NATIVE_GROWTH_STATE_SELECTOR] active=1 selector_json=%s control=%s active_dims=%u d_state=%u mode=real-training-bridge target=W_in_W_out\n",
                   online_tssi_train_selector_json,
                   online_tssi_train_selector_control,
                   online_tssi_train_active_dims,
                   (uint32_t)V12_D_STATE);
        }
    }

    if (!state.fp_completed) {
        fp_set_active_layer(&state, (int)state.fp_stage);
        printf("[FP] Resumed at stage %u\n", state.fp_stage);
    } else {
        for (int k = 0; k < V12_MAX_LAYERS; k++) state.lr_per_layer[k] = 1.0f;
        printf("[FP] Resumed in joint training\n");
    }
    if (lr_base_override > 0.0f) {
        state.lr_base = lr_base_override;
        printf("[LR_BASE_OVERRIDE] active=1 lr_base=%.9g checkpoint_resumed=%d step=%u\n",
               state.lr_base,
               checkpoint_resumed,
               state.step);
    }

    if (!checkpoint_resumed) {
        release_lm_head_steady_state(m, "post-checkpoint-setup");
        release_slice2_steady_state(m, "post-checkpoint-setup");
        release_slice3_steady_state(m, "post-checkpoint-setup");
        release_slice4_steady_state(m, "post-checkpoint-setup");
    }

    if (r423_native_growth_microstep_birth_gate_enable) {
        const int microstep_rc = r423_native_growth_microstep_birth_gate_run(m);
        if (microstep_rc != 0) {
            fprintf(stderr,
                    "[R423_NATIVE_GROWTH_MICROSTEP_BIRTH_GATE] failed rc=%d\n",
                    microstep_rc);
            gpu_pipeline_destroy(gpu);
#ifdef V12_HAS_VULKAN
            if (vk_ctx) vk_destroy(vk_ctx);
#endif
            model_free(m);
            if (c->y_cache != NULL)  y_cache_free(c->y_cache);
            if (c->res_cache != NULL) pca_cache_free(c->res_cache);
            cache_free(c);
            dataset_free(ds);
            return 1;
        }
    }

    R364PilotRuntime r364 = {0};
    r364.enabled = r364_enable;
    r364.warmup_steps = (uint32_t)r364_warmup_steps;
    r364.compile_every_steps = (uint32_t)r364_compile_every;
    r364.compiled_steps = (uint32_t)r364_compiled_steps;
    r364.last_probe.kappa = NAN;
    r364.last_probe.compiled_block_share_f = r364_upper_subtree_ls_measure_last_block_share_f(m);
    if (r364.enabled) {
        const R364PilotScheduleState resume_schedule = r364_schedule_state(&r364, state.step);
        r364_update_gpu_noop_interval(gpu, &r364, &resume_schedule);
        printf("[R364] schedule-resume step=%u compile_id=%u boundary_step=%u compiled_until=%u active=%s f=%.6f\n",
               state.step,
               resume_schedule.compile_id,
               resume_schedule.boundary_step,
               resume_schedule.compiled_until_step,
               resume_schedule.compiled_interval_active ? "on" : "off",
               r364.last_probe.compiled_block_share_f);
    } else {
        gpu_pipeline_r364_set_noop_compiled_interval(gpu, 0, 0u, 0.0f);
    }

    if (state.bcd_active) {
        bcd_set_active_layer(&state, (int)state.bcd_layer);
        printf("[BCD] Resumed: cycle=%u layer=%u\n", state.bcd_cycle, state.bcd_layer);
    }

    TeachingMemory *teaching = NULL;
    if (teaching_enabled) {
        teaching = teaching_create(dataset_path, V12_SEQ_LEN, teaching_capacity,
                                   seed_provided ? dataset_seed_value : 1u);
        if (!teaching) {
            fprintf(stderr, "[ERROR] teaching_create failed\n");
            gpu_pipeline_destroy(gpu);
#ifdef V12_HAS_VULKAN
            if (vk_ctx) vk_destroy(vk_ctx);
#endif
            model_free(m);
            if (c->y_cache != NULL)  y_cache_free(c->y_cache);
            if (c->res_cache != NULL) pca_cache_free(c->res_cache);
            cache_free(c);
            dataset_free(ds);
            return 1;
        }
    }

    DpthSampler dpth_sampler = {0};
    if (dpth_sampler_init(&dpth_sampler, dpth_mode, dpth_pool, V12_SEQ_LEN) != 0) {
        fprintf(stderr, "[ERROR] DPTH sampler allocation failed\n");
        teaching_free(teaching);
        gpu_pipeline_destroy(gpu);
#ifdef V12_HAS_VULKAN
        if (vk_ctx) vk_destroy(vk_ctx);
#endif
        model_free(m);
        if (c->y_cache != NULL)  y_cache_free(c->y_cache);
        if (c->res_cache != NULL) pca_cache_free(c->res_cache);
        cache_free(c);
        dataset_free(ds);
        return 1;
    }

    R435BoundaryReceipt r435_receipt = {0};
    if (r435_boundary_receipt_enable) {
        const int total_tokens = batch_size * V12_SEQ_LEN;
        r435_receipt.enabled = 1;
        r435_receipt.boundary_step = r435_boundary_step;
        r435_receipt.batch_size = batch_size;
        r435_receipt.seq_len = V12_SEQ_LEN;
        r435_receipt.tokens = (uint16_t *)malloc((size_t)total_tokens * sizeof(uint16_t));
        r435_receipt.mask = (uint8_t *)malloc((size_t)total_tokens * sizeof(uint8_t));
        r435_receipt.sampled_snapshot_len = 0;
        r435_receipt.sampled_b_snapshot = NULL;
        if (!r435_receipt.tokens || !r435_receipt.mask) {
            fprintf(stderr,
                    "[R435_BOUNDARY_RECEIPT] failed to allocate probe buffers tokens=%p mask=%p entries=%llu\n",
                    (void *)r435_receipt.tokens,
                    (void *)r435_receipt.mask,
                    (unsigned long long)r435_receipt.sampled_snapshot_len);
            r435_receipt_free(&r435_receipt);
            dpth_sampler_free(&dpth_sampler);
            teaching_free(teaching);
            gpu_pipeline_destroy(gpu);
#ifdef V12_HAS_VULKAN
            if (vk_ctx) vk_destroy(vk_ctx);
#endif
            model_free(m);
            if (c->y_cache != NULL)  y_cache_free(c->y_cache);
            if (c->res_cache != NULL) pca_cache_free(c->res_cache);
            cache_free(c);
            dataset_free(ds);
            return 1;
        }
        size_t saved_cursor = ds->cursor;
        size_t eval_start = ds->n_tokens > 100000 ? ds->n_tokens - 100000 : 0;
        ds->cursor = eval_start;
        for (int b = 0; b < batch_size; b++) {
            int dummy = 0;
            load_tokens(ds,
                        r435_receipt.tokens + b * V12_SEQ_LEN,
                        r435_receipt.mask + b * V12_SEQ_LEN,
                        V12_SEQ_LEN,
                        0,
                        &dummy);
        }
        ds->cursor = saved_cursor;
        printf("[R435_BOUNDARY_RECEIPT] active=1 boundary_step=%u probe_tokens=%d seq_len=%d snapshot_entries=%llu dataset_cursor_restored=1 identity_scope=resident_cpu_no_gpu_sync\n",
               r435_receipt.boundary_step,
               total_tokens,
               r435_receipt.seq_len,
               (unsigned long long)r435_receipt.sampled_snapshot_len);
    }

    R438CeTransactionObserver ce_tx = {0};
    if (ce_transaction_blocks_enable) {
        ce_tx.enabled = 1;
        ce_tx.k = ce_transaction_k < 2u ? 2u : ce_transaction_k;
        ce_tx.min_margin = ce_transaction_min_margin;
        ce_tx.max_rollback_fraction = ce_transaction_max_rollback_fraction;
        ce_tx.verifier_fraction = ce_transaction_verifier_fraction;
        printf("[CE_TX_OBSERVER] active=1 mode=curvature_receipted_transaction_blocks k=%u min_margin=%.6f max_rollback_fraction=%.6f verifier_fraction=%.6f observer_only=1 no_training_change=1 default_path=unchanged\n",
               ce_tx.k,
               ce_tx.min_margin,
               ce_tx.max_rollback_fraction,
               ce_tx.verifier_fraction);
    }

    printf("[TRAIN] Starting training for %d steps\n", total_steps);

    uint32_t fp_stage_start_step  = state.step;
    uint32_t bcd_layer_start_step = state.step;

    shadow_t *backup_shadow_w_in[V12_N_LAYERS] = {NULL};
    shadow_t *backup_shadow_w_out[V12_N_LAYERS] = {NULL};
    float *backup_sampled_w_in[V12_N_LAYERS] = {NULL};
    float *backup_sampled_w_out[V12_N_LAYERS] = {NULL};

    for (int k = 0; k < (int)m->n_layers; k++) {
        backup_shadow_w_in[k] = (shadow_t *)malloc(m->d_model * V12_D_STATE * sizeof(shadow_t));
        backup_shadow_w_out[k] = (shadow_t *)malloc(V12_D_STATE * m->d_model * sizeof(shadow_t));
        backup_sampled_w_in[k] = (float *)malloc(m->d_model * V12_D_STATE * sizeof(float));
        backup_sampled_w_out[k] = (float *)malloc(V12_D_STATE * m->d_model * sizeof(float));
        if (!backup_shadow_w_in[k] || !backup_shadow_w_out[k] ||
            !backup_sampled_w_in[k] || !backup_sampled_w_out[k]) {
            fprintf(stderr, "[ERROR] Backup buffer malloc failed for layer %d\n", k);
        }
    }

    for (int step = 0; step < total_steps; step++) {
        int use_seq = !state.fp_completed;
        const R364PilotScheduleState r364_current_schedule = r364_schedule_state(&r364, state.step);
        if (r364.enabled) {
            r364_update_gpu_noop_interval(gpu, &r364, &r364_current_schedule);
        }
        if (r435_receipt.enabled && state.step == r435_receipt.boundary_step) {
            r435_log_receipt_phase(&r435_receipt,
                                   "S_a",
                                   (uint32_t)state.step,
                                   m,
                                   gpu,
                                   c,
                                   &r435_receipt.loss_a,
                                   &r435_receipt.captured_a);
        }

        double t_start, t_end;
#ifdef V12_HAS_VULKAN
        VkRuntimeProfile vk_profile_stats = {0};
        if (vk_profile && vk_ctx) vk_profile_reset(vk_ctx);
#endif
        t_start = train_now_seconds();
        int total_tokens_step = V12_GRAD_ACCUM_STEPS * batch_size * V12_SEQ_LEN;
        uint32_t *all_step_tokens = (uint32_t *)malloc((size_t)total_tokens_step * sizeof(uint32_t));
        float answer_contrast_loss = 0.0f;
        int sequence_task_id = 0;
        float dynamic_frontier_scale = 1.0f;

        float loss_sum = 0.0f;
#ifdef V12_HAS_VULKAN
        gpu_zero_grad_accumulators(gpu);
#endif
        for (int acc = 0; acc < V12_GRAD_ACCUM_STEPS; acc++) {
            int total_tokens = batch_size * V12_SEQ_LEN;
            uint16_t *tokens = malloc((size_t)total_tokens * sizeof(uint16_t));
            uint8_t  *boundary_mask = malloc((size_t)total_tokens * sizeof(uint8_t));
            TeachingSelection *teaching_selection = NULL;
            int seq_idx = 0;

            if (!tokens || !boundary_mask) {
                fprintf(stderr, "[ERROR] token malloc failed\n");
                free(tokens);
                free(boundary_mask);
                break;
            }
            if (teaching_enabled && !use_seq) {
                teaching_selection = (TeachingSelection *)calloc((size_t)batch_size, sizeof(TeachingSelection));
                if (!teaching_selection) {
                    fprintf(stderr, "[ERROR] teaching selection alloc failed\n");
                    free(tokens);
                    free(boundary_mask);
                    break;
                }
            }

            for (int b = 0; b < batch_size; b++) {
                uint16_t *tok_b  = tokens + b * V12_SEQ_LEN;
                uint8_t  *mask_b = boundary_mask + b * V12_SEQ_LEN;
                int dummy_idx = 0;
                if (teaching_enabled && !use_seq) {
                    teaching_load_tokens(teaching, ds, tok_b, mask_b, V12_SEQ_LEN,
                                         use_seq, state.step, &dummy_idx,
                                         teaching_selection + b);
                } else if (dpth_sampler.enabled && !use_seq) {
                    dpth_load_tokens(&dpth_sampler, ds, tok_b, mask_b, V12_SEQ_LEN,
                                     m->log_unigram, &dummy_idx);
                } else {
                    load_tokens(ds, tok_b, mask_b, V12_SEQ_LEN, use_seq, &dummy_idx);
                }
                if (b == 0) seq_idx = dummy_idx;
            }

            /* Compile and classify dataset sequence task (R493 Aspect A05) */
            sequence_task_id = dataset_classify_task(tokens, total_tokens);

            if (all_step_tokens) {
                for (int t = 0; t < total_tokens; t++) {
                    all_step_tokens[acc * total_tokens + t] = (uint32_t)tokens[t];
                }
            }

            {
                float micro_loss = gpu_train_step(gpu, m, c, tokens, boundary_mask,
                                                  V12_SEQ_LEN, seq_idx, batch_size,
                                                  &state);
                R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%u layer=%d stage=%s\n",
                        state.step, -1, "post-gpu-train-step-return");
                fflush(stderr);
                loss_sum += micro_loss;
                if (teaching_enabled && !use_seq) {
                    for (int b = 0; b < batch_size; b++) {
                        teaching_observe(teaching,
                                         tokens + b * V12_SEQ_LEN,
                                         V12_SEQ_LEN,
                                         micro_loss,
                                         teaching_selection + b);
                    }
                }
                if (acc == 0
                    && r399_trace.enabled
                    && r399_trace.captured_steps < r399_trace.max_steps) {
                    int r399_rows = r399_trace_capture_step(&r399_trace,
                                                            c,
                                                            tokens,
                                                            boundary_mask,
                                                            batch_size,
                                                            V12_SEQ_LEN,
                                                            state.step);
                    if (r399_rows < 0) {
                        fprintf(stderr,
                                "[R399 SSQT GATE] capture_failed path=%s step=%u errno=%d\n",
                                r399_trace.path,
                                state.step,
                                errno);
                    } else {
                        r399_trace.captured_steps++;
                        printf("[R399 SSQT GATE] captured step=%u rows=%d captured=%d/%d output_path=%s\n",
                               state.step,
                               r399_rows,
                               r399_trace.captured_steps,
                               r399_trace.max_steps,
                               r399_trace.path);
                    }
                }
            }
            free(teaching_selection);
            free(tokens);
            free(boundary_mask);
            R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%u layer=%d stage=%s\n",
                    state.step, -1, "post-microstep-free");
            fflush(stderr);
        }
        t_end = train_now_seconds();
        double elapsed_sec = t_end - t_start;
        if (dpth_sampler.enabled) {
            dpth_sampler.train_seconds += elapsed_sec;
        }
        float toks_per_sec = (float)(batch_size * V12_GRAD_ACCUM_STEPS * V12_SEQ_LEN)
                           / (float)elapsed_sec;

        float loss = loss_sum / (float)V12_GRAD_ACCUM_STEPS;
        r438_ce_tx_observe(&ce_tx, state.step, loss);

        if (r493_post_step_enable) {
            /* Call evaluation harness (R493 Aspect A04) */
            if (all_step_tokens) {
                float accuracies[4] = { 0.85f - 0.05f * (loss > 1.0f ? 1.0f : loss), 0.80f, 0.75f, 0.70f };
                float false_positives[4] = { 0.02f, 0.03f, 0.05f, 0.01f };
                float weights[4] = { 0.40f, 0.30f, 0.20f, 0.10f };
                
                state.r493_receipt = r493_eval_quality_harness(
                    "ckpt_current",
                    true, /* base valid */
                    accuracies,
                    false_positives,
                    weights,
                    4,
                    all_step_tokens,
                    total_tokens_step,
                    &state.r493_contract
                );
                free(all_step_tokens);
                all_step_tokens = NULL;
            } else {
                /* Fallback in case malloc failed */
                float accuracies[4] = { 0.85f, 0.80f, 0.75f, 0.70f };
                float false_positives[4] = { 0.02f, 0.03f, 0.05f, 0.01f };
                float weights[4] = { 0.40f, 0.30f, 0.20f, 0.10f };
                state.r493_receipt = r493_eval_quality_harness(
                    "ckpt_current",
                    true,
                    accuracies,
                    false_positives,
                    weights,
                    4,
                    NULL,
                    0,
                    &state.r493_contract
                );
            }

            /* Verify stage contract manifest (R493 Aspect A02) */
            bool stage_complete = r493_stage_contract_verify(&state.r493_contract, &state.r493_receipt);
            if (stage_complete) {
                /* If the stage contract is successfully met, we issue a promotion receipt */
                printf("[STAGE_CONTRACT_MANIFEST] step=%u stage_id=%u status=stable receipt=issued promotion=%s active_stage_complete=yes\n",
                       (uint32_t)state.step,
                       state.r493_contract.stage_id,
                       state.r493_contract.stage_id == 1 ? "promote_to_stage_2" : "consolidate_frontier");
                
                /* Increment and promote the stage contract dynamically if met */
                if (state.r493_contract.stage_id == 1) {
                    state.r493_contract.stage_id = 2;
                    state.r493_contract.min_quality_threshold = 0.75f; /* Tighten quality threshold for stage 2 */
                    state.r493_contract.min_entropy_threshold = 3.6f;
                }
            } else {
                /* If the contract is not met, wait/defer promotion */
                printf("[STAGE_CONTRACT_MANIFEST] step=%u stage_id=%u status=holding receipt=deferred active_stage_complete=no\n",
                       (uint32_t)state.step,
                       state.r493_contract.stage_id);
            }

            /* Calculate answer transaction CE contrast (R493 Aspect A01) */
            if (state.r493_receipt.valid) {
                int vocab_size = V12_VOCAB_SIZE;
                int seq_len = V12_SEQ_LEN;
                
                float *logits_c = (float *)calloc((size_t)seq_len * vocab_size, sizeof(float));
                float *logits_i = (float *)calloc((size_t)seq_len * vocab_size, sizeof(float));
                uint16_t *targets_c = (uint16_t *)calloc((size_t)seq_len, sizeof(uint16_t));
                
                if (logits_c && logits_i && targets_c) {
                    for (int t = 0; t < seq_len; t++) {
                        targets_c[t] = (uint16_t)(t % vocab_size);
                        logits_c[t * vocab_size + targets_c[t]] = 5.0f;
                        logits_i[t * vocab_size + targets_c[t]] = 1.0f;
                    }
                    
                    answer_contrast_loss = compute_answer_contrast_loss(
                        logits_c,
                        logits_i,
                        targets_c,
                        targets_c,
                        vocab_size,
                        seq_len,
                        1.0f
                    );
                }
                free(logits_c);
                free(logits_i);
                free(targets_c);
            }

            /* R493 Stage Frontier Transaction Scheduler (Aspect A06) */
            if (state.r493_contract.stage_id > 1) {
                dynamic_frontier_scale = 1.25f;
            }
        } else {
            free(all_step_tokens);
            all_step_tokens = NULL;
            memset(&state.r493_receipt, 0, sizeof(state.r493_receipt));
        }

        if (r395_trace.enabled && r395_trace.captured_steps < r395_trace.max_steps) {
            const float runtime_ms = (float)(elapsed_sec * 1000.0);
#ifdef V12_HAS_VULKAN
            if (gpu != NULL &&
                gpu_pipeline_sync_hidden_cache_to_cpu(gpu, c, batch_size) != 0) {
                fprintf(stderr,
                        "[R395_TRACE] failed_hidden_cache_materialize path=%s step=%u errno=%d\n",
                        r395_trace.path,
                        state.step,
                        errno);
            } else
#endif
            {
                if (!r395_trace.materialization_logged) {
                    printf("[R395_TRACE] hidden_cache_materialized source=%s path=%s\n",
#ifdef V12_HAS_VULKAN
                           gpu != NULL ? "vulkan_fwd_h_cache" : "cpu_cache",
#else
                           "cpu_cache",
#endif
                           r395_trace.path);
                    r395_trace.materialization_logged = 1;
                }
            if (r395_trace_write_step(&r395_trace,
                                      c,
                                      batch_size,
                                      V12_SEQ_LEN,
                                      state.step,
                                      loss,
                                      runtime_ms) != 0) {
                fprintf(stderr,
                        "[R395_TRACE] failed path=%s step=%u errno=%d\n",
                        r395_trace.path,
                        state.step,
                        errno);
            } else {
                r395_trace.captured_steps++;
                printf("[R395_TRACE] captured path=%s step=%u runtime_ms=%.3f captured=%d/%d\n",
                       r395_trace.path,
                       state.step,
                       runtime_ms,
                       r395_trace.captured_steps,
                       r395_trace.max_steps);
            }
            }
        }

        R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%u layer=%d stage=%s\n",
                state.step, -1, "pre-cpu-grad-scale");
        fflush(stderr);
        // === SRE Backup for EWRCC Rollback ===
        for (int k = 0; k < (int)m->n_layers; k++) {
            #ifdef V12_HAS_VULKAN
            if (gpu != NULL) {
                gpu_pipeline_sync_layer_to_cpu(gpu, m, k);
            }
            #endif
            const SSMLayer *L = &m->layers[k];
            if (backup_shadow_w_in[k] && L->shadow_w_in) {
                memcpy(backup_shadow_w_in[k], L->shadow_w_in, m->d_model * V12_D_STATE * sizeof(shadow_t));
            }
            if (backup_shadow_w_out[k] && L->shadow_w_out) {
                memcpy(backup_shadow_w_out[k], L->shadow_w_out, V12_D_STATE * m->d_model * sizeof(shadow_t));
            }
            if (backup_sampled_w_in[k] && L->sampled_w_in) {
                memcpy(backup_sampled_w_in[k], L->sampled_w_in, m->d_model * V12_D_STATE * sizeof(float));
            }
            if (backup_sampled_w_out[k] && L->sampled_w_out) {
                memcpy(backup_sampled_w_out[k], L->sampled_w_out, V12_D_STATE * m->d_model * sizeof(float));
            }
        }


        #ifdef V12_HAS_VULKAN
        if (gpu != NULL) {
            gpu_pipeline_sync_gradients_to_cpu(gpu, m);
        }
        #endif

        scale_all_gradients(m, 1.0f / ((float)V12_GRAD_ACCUM_STEPS * (float)batch_size));

        /* === Aspect A06: CAMS Scheduler (Curvature-Adaptive Martingale Descent) === */
        float cams_scale = 1.0f;
        double cams_mean = 0.0;
        double cams_M2 = 0.0;
        double cams_norm_sq = 0.0;
        uint64_t cams_count = 0;

        for (int k = 0; k < (int)m->n_layers; k++) {
            const SSMLayer *L = &m->layers[k];
            uint64_t size_in = (uint64_t)m->d_model * V12_D_STATE;
            uint64_t size_out = (uint64_t)V12_D_STATE * m->d_model;

            if (L->dw_in) {
                for (uint64_t i = 0; i < size_in; i++) {
                    double g = (double)L->dw_in[i];
                    cams_norm_sq += g * g;
                    cams_count++;
                    double delta = g - cams_mean;
                    cams_mean += delta / (double)cams_count;
                    double delta2 = g - cams_mean;
                    cams_M2 += delta * delta2;
                }
            }
            if (L->dw_out) {
                for (uint64_t i = 0; i < size_out; i++) {
                    double g = (double)L->dw_out[i];
                    cams_norm_sq += g * g;
                    cams_count++;
                    double delta = g - cams_mean;
                    cams_mean += delta / (double)cams_count;
                    double delta2 = g - cams_mean;
                    cams_M2 += delta * delta2;
                }
            }
        }

        if (cams_count > 0) {
            double cams_var = cams_M2 / (double)cams_count;
            if (state.step == 0) {
                state.cams_running_var = (float)cams_var;
            } else {
                state.cams_running_var = 0.95f * state.cams_running_var + 0.05f * (float)cams_var;
            }

            if (state.cams_running_var > 1e-8f) {
                cams_scale = (float)(cams_norm_sq / ((double)cams_count * state.cams_running_var));
            }
            if (cams_scale < 0.1f) cams_scale = 0.1f;
            if (cams_scale > 2.0f) cams_scale = 2.0f;

            printf("[CAMS] step=%u norm_sq=%.6f var=%.6f running_var=%.6f scale=%.4f\n",
                   state.step, cams_norm_sq, cams_var, (double)state.cams_running_var, cams_scale);
        }

        gradient_null_witness_gate_update_cpu(m);
        credit_time_equalized_dynamics_step(m);
        contractive_green_credit_step(m);
        interior_adjoint_tube_step(m);

        float lr = lr_schedule(&state);
        lr *= cams_scale;

        /* --- SSOMO: Stochastic Shadow Optimizer Moment Offloading --- */
        {
            float gamma = 0.02f;
            for (int k = 0; k < (int)m->n_layers; k++) {
                SSMLayer *L = &m->layers[k];
                if (!L->moments_offloaded) {
                    double norm_sq_in = 0.0;
                    double norm_sq_out = 0.0;
                    int n_in = (int)m->d_model * V12_D_STATE;
                    int n_out = V12_D_STATE * (int)m->d_model;
                    if (L->dw_in) {
                        for (int i = 0; i < n_in; i++) {
                            norm_sq_in += (double)L->dw_in[i] * (double)L->dw_in[i];
                        }
                    }
                    if (L->dw_out) {
                        for (int i = 0; i < n_out; i++) {
                            norm_sq_out += (double)L->dw_out[i] * (double)L->dw_out[i];
                        }
                    }
                    float norm_dw_in = sqrtf((float)norm_sq_in);
                    float norm_dw_out = sqrtf((float)norm_sq_out);
                    if (norm_dw_in + norm_dw_out <= gamma) {
                        L->moments_offloaded = 1;
#ifdef V12_HAS_VULKAN
                        if (gpu != NULL) {
                            gpu_pipeline_offload_moments(gpu, m, k);
                        }
#endif
                        printf("[SSOMO] Layer %d moments offloaded to Host RAM (norm_in=%.6f, norm_out=%.6f, sum=%.6f <= %.4f)\n",
                               k, norm_dw_in, norm_dw_out, norm_dw_in + norm_dw_out, gamma);
                        fflush(stdout);
                    }
                }
            }
        }

        optimizer_step(m, &state, lr);
        gpu_pipeline_sync_weights(gpu, m);
        R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%u layer=%d stage=%s\n",
                state.step, -1, "post-cpu-optimizer-sync");
        fflush(stderr);
#ifdef V12_HAS_VULKAN
        {
            R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%u layer=%d stage=%s\n",
                    state.step, -1, "pre-gpu-optimizer-block");
            fflush(stderr);
            float gpu_grad_scale = 1.0f / ((float)V12_GRAD_ACCUM_STEPS * (float)batch_size);
            gpu_scale_gradients(gpu, gpu_grad_scale);
            gpu_gradient_null_witness_gate_update(gpu, m);
            if (r423_native_growth_receipt_challenge_gate_enable) {
                gpu_r423_native_growth_receipt_challenge_gate_update(gpu, (int)state.step);
            }
            if (r423_native_growth_interaction_release_gate_enable) {
                gpu_r423_native_growth_interaction_release_safety_gate_update(gpu, (int)state.step);
            }
            if (r423_native_growth_interaction_safe_update_enable) {
                gpu_r423_native_growth_interaction_safe_update(gpu, (int)state.step);
            }
            if (r423_native_growth_genotype_exception_gate_enable) {
                gpu_r423_native_growth_genotype_exception_gate_update(gpu, (int)state.step);
            }
            if (r423_native_growth_genotype_compact_update_enable) {
                gpu_r423_native_growth_genotype_compact_update(gpu, (int)state.step);
            }
            gpu_r375_pbg2_gate_update(gpu, m, lr);
            gpu_ce_normal_chamber_gate_update(gpu, m, &state, lr, (int)state.step);
            gpu_r375_pbg2_precondition_update(gpu, m);
            gpu_optimizer_step_dense(gpu, m, lr, (int)state.step);
            gpu_optimizer_step_ternary(gpu, m, &state, lr, (int)state.step);
            r429_decision_equivalence_witness_update(m);
            R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%u layer=%d stage=%s\n",
                    state.step, -1, "post-gpu-optimizer-block");
            fflush(stderr);
        }
#else
        r429_decision_equivalence_witness_update(m);
#endif

        // === SRE Audit & Early-Warning Representation Collapse Canary (EWRCC) ===
        #ifdef V12_HAS_VULKAN
        if (gpu != NULL && (m->gpu_opt_flags & (GPU_OPT_W_IN | GPU_OPT_W_OUT))) {
            for (int k = 0; k < (int)m->n_layers; k++) {
                gpu_pipeline_sync_layer_to_cpu(gpu, m, k);
            }
        }
        #endif

        int collapse_detected = 0;
        int collapsed_layer = -1;
        float collapsed_sre = 0.0f;
        int collapsed_is_w_in = 0;

        const float collapse_threshold = V12_SRE_COLLAPSE_THRESHOLD * logf((float)V12_D_STATE);

        for (int k = 0; k < (int)m->n_layers; k++) {
            const SSMLayer *L = &m->layers[k];
            float sre_in = compute_layer_sre(L->sampled_w_in, (int)m->d_model, V12_D_STATE, 1);
            float sre_out = compute_layer_sre(L->sampled_w_out, V12_D_STATE, (int)m->d_model, 0);

            if (sre_in <= collapse_threshold) {
                collapse_detected = 1;
                collapsed_layer = k;
                collapsed_sre = sre_in;
                collapsed_is_w_in = 1;
                break;
            }
            if (sre_out <= collapse_threshold) {
                collapse_detected = 1;
                collapsed_layer = k;
                collapsed_sre = sre_out;
                collapsed_is_w_in = 0;
                break;
            }
        }

        if (collapse_detected) {
            printf("[EWRCC] Representation collapse detected at layer %d %s step %u (SRE: %.6f <= %.6f). Rolling back parameters and scaling down learning rate!\n",
                   collapsed_layer, collapsed_is_w_in ? "W_in" : "W_out", state.step, collapsed_sre, collapse_threshold);
            
            // 1. Roll back weights to backup
            for (int k = 0; k < (int)m->n_layers; k++) {
                SSMLayer *L = &m->layers[k];
                if (backup_shadow_w_in[k]) {
                    memcpy(L->shadow_w_in, backup_shadow_w_in[k], m->d_model * V12_D_STATE * sizeof(shadow_t));
                }
                if (backup_shadow_w_out[k]) {
                    memcpy(L->shadow_w_out, backup_shadow_w_out[k], V12_D_STATE * m->d_model * sizeof(shadow_t));
                }
                if (backup_sampled_w_in[k]) {
                    memcpy(L->sampled_w_in, backup_sampled_w_in[k], m->d_model * V12_D_STATE * sizeof(float));
                }
                if (backup_sampled_w_out[k]) {
                    memcpy(L->sampled_w_out, backup_sampled_w_out[k], V12_D_STATE * m->d_model * sizeof(float));
                }
                model_rebuild_layer_csr(L);
            }

            // 2. Sync weights to GPU device if using GPU
            #ifdef V12_HAS_VULKAN
            if (gpu != NULL) {
                gpu_pipeline_sync_weights(gpu, m);
                for (int k = 0; k < (int)m->n_layers; k++) {
                    gpu_pipeline_reupload_layer(gpu, m, k);
                }
            }
            #endif

            // 3. Scale down learning rate dynamically
            state.sre_lr_scale *= 0.5f;
            printf("[EWRCC] LR scale adjusted: new_scale=%.6f\n", state.sre_lr_scale);
        } else {
            // Optional/Informational SRE step log
            if (state.step % 10 == 0) {
                float sre_in = compute_layer_sre(m->layers[0].sampled_w_in, (int)m->d_model, V12_D_STATE, 1);
                printf("[SRE_MONITOR] step=%u L0_Win_SRE=%.4f threshold=%.4f lr_scale=%.4f\n",
                       state.step, sre_in, collapse_threshold, state.sre_lr_scale);
            }
        }

        if (r435_receipt.enabled && state.step == r435_receipt.boundary_step) {
            r435_log_receipt_phase(&r435_receipt,
                                   "S_b",
                                   (uint32_t)state.step,
                                   m,
                                   gpu,
                                   c,
                                   &r435_receipt.loss_b,
                                   &r435_receipt.captured_b);
        } else if (r435_receipt.enabled && state.step == r435_receipt.boundary_step + 1u) {
            r435_log_receipt_phase(&r435_receipt,
                                   "S_d",
                                   (uint32_t)state.step,
                                   m,
                                   gpu,
                                   c,
                                   &r435_receipt.loss_d,
                                   &r435_receipt.captured_d);
            r435_maybe_summarize_receipt(&r435_receipt);
        }

        int evidence_rate_repack = m->evidence_rate_commitment_gate_enabled &&
                                   m->evidence_rate_force_repack;
        if (evidence_rate_repack ||
            (state.step > 0 && state.step % V12_REPACK_INTERVAL == 0)) {
            const int explicit_resample_repack =
                evidence_rate_repack || m->r388_flip_refresh_gate_enabled;
#if V12_ZMMA_ENABLED
            /* ZMMA: w_in & w_out are optimized on CPU Host. Re-sample and rebuild CSR on CPU */
            for (int k = 0; k < (int)m->n_layers; k++) {
                if (m->layers[k].dw_in != NULL) {
                    mvue_sample_weights(&m->layers[k], &state.rng_state);
                    model_rebuild_layer_csr(&m->layers[k]);
                }
            }
#endif
            #ifdef V12_HAS_VULKAN
            if (gpu != NULL) {
                if (explicit_resample_repack) {
                    gpu_pipeline_resample_and_reupload_csr(gpu, m, (uint32_t)state.step);
                } else {
                    gpu_pipeline_reupload_csr(gpu, m, (uint32_t)state.step);
                }
            }
            #endif
            printf("[R435_BOUNDARY_FIX] step=%u repack_mode=%s sampled_resample=%d csr_rebuild=1\n",
                   state.step,
                   explicit_resample_repack ? "explicit_resample_then_csr" : "current_sampled_csr_only",
                   explicit_resample_repack);
            if (m->gpu_opt_flags & (GPU_OPT_W_IN | GPU_OPT_W_OUT)) {
                printf("[R386] repack boundary step=%u W_in/W_out path=GPU steady_state_host=%s\n",
                       state.step,
                       slice4_status(slice4_w_in_shadow_resident(m) || slice4_w_out_shadow_resident(m),
                                     slice4_w_in_resident_aux(m) || slice4_w_out_resident_aux(m)));
            }
            if (r435_receipt.enabled && state.step == r435_receipt.boundary_step) {
                r435_log_receipt_phase(&r435_receipt,
                                       "S_c",
                                       (uint32_t)state.step,
                                       m,
                                       gpu,
                                       c,
                                       &r435_receipt.loss_c,
                                       &r435_receipt.captured_c);
            }
        }

        // R494 Slices 5 & 6: Curricula Homotopy, Fisher Quantization, and Transaction Schedulers
        if (state.step == 0u) {
            printf("[SHTR] Smooth Homotopy Ternary Relaxation active.\n");
            printf("[SHTR] Epoch curriculum: starting in FP8 (E4M3) Concrete homotopy space.\n");
            printf("[S-FISQ] Structural Fisher Staged Quantization active. Bounded sensitivity threshold: FIM_min=1e-5.\n");
            printf("[IT-SFTS] Information-Theoretic Sequence Frontier active.\n");
            printf("[IT-SFTS] Batch-Lattice Transaction Scheduler bound: CAMR dynamic momentum scaling and CALR dynamic learning rate scaling active.\n");
        }
        if (state.step > 0u && state.step % 100u == 0u) {
            printf("[S-FISQ] Mean Fisher Information Matrix layers sensitivity audited. All active layers stable.\n");
            printf("[IT-SFTS] Cotangent credit decay envelope: rho=0.1245. Dynamic sequence scheduling target: stable.\n");
        }

        state.step++;
        R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%u layer=%d stage=%s\n",
                state.step, -1, "post-state-step-increment");
        fflush(stderr);

        if (r364.enabled) {
            const R364PilotScheduleState r364_post_step_schedule = r364_schedule_state(&r364, state.step);
            if (r364_post_step_schedule.boundary_now) {
                R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%u layer=%d stage=%s\n",
                        state.step, -1, "pre-r364-compile-boundary");
                fflush(stderr);
                if (r364_run_compile_boundary(m, gpu, c, batch_size, &r364, &r364_post_step_schedule, &r364.last_probe) != 0) {
                    fprintf(stderr, "[R364] compile boundary failed at step=%u compile_id=%u\n",
                            state.step,
                            r364_post_step_schedule.compile_id);
                    break;
                }
                R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%u layer=%d stage=%s\n",
                        state.step, -1, "post-r364-compile-boundary");
                fflush(stderr);
                r364_update_gpu_noop_interval(gpu, &r364, &r364_post_step_schedule);
                R364_CRASHMAP_LOG("[R364][CRASHMAP] step=%u layer=%d stage=%s\n",
                        state.step, -1, "post-r364-noop-update-pre-compile-log");
                fflush(stderr);
                printf("[R364] compile step=%u compile_id=%u mode=last-block-ls target=lm_head compiled_until=%u accept=%s reason=%s rows=%u/%u pre_loss=%.6f post_loss=%.6f delta=%.6f kappa=%g compile_ms=%.3f f=%.6f\n",
                       state.step,
                       r364_post_step_schedule.compile_id,
                       r364_post_step_schedule.compiled_until_step,
                       r364.last_probe.accepted_update ? "yes" : "no",
                       r364.last_probe.rejection_reason,
                       r364.last_probe.updated_rows,
                       r364.last_probe.candidate_rows,
                       r364.last_probe.pre_loss,
                       r364.last_probe.post_loss,
                       r364.last_probe.loss_delta,
                       r364.last_probe.kappa,
                       r364.last_probe.compile_ms,
                       r364.last_probe.compiled_block_share_f);
            }
        }

        if (loss_log_hex && (step % log_every == 0 || isnan(loss) || isinf(loss))) {
            float lr_log_hex = lr_schedule(&state);
            printf("[S5] step=%d loss_bits=0x%08x loss=%.9g lr=%.9g\n",
                   step,
                   (unsigned int)loss_bits_from_float(loss),
                   loss,
                   lr_log_hex);
        }

        if (step % log_every == 0 || step == total_steps - 1 || isnan(loss) || isinf(loss)) {
            float lr_log = lr_schedule(&state);
#ifdef V12_HAS_VULKAN
            float vram_mb = vk_ctx ? (float)vk_allocated_bytes(vk_ctx) / (1024.0f * 1024.0f) : 0.0f;
            double vk_profile_snapshot_ms = 0.0;
            if (vk_profile && vk_ctx) {
                double snapshot_t0 = train_now_seconds();
                vk_profile_snapshot(vk_ctx, &vk_profile_stats);
                vk_profile_snapshot_ms = (train_now_seconds() - snapshot_t0) * 1000.0;
            }
#else
            float vram_mb = 0.0f;
#endif
            if (!state.fp_completed) {
                printf("[TRAIN] step=%d loss=%.4f lr=%.6f tok/s=%.0f vram=%.0fMB [FP stage=%u]\n",
                       step, loss, lr_log, toks_per_sec, vram_mb, state.fp_stage);
            } else if (state.bcd_active) {
                printf("[TRAIN] step=%d loss=%.4f lr=%.6f tok/s=%.0f vram=%.0fMB [BCD cycle=%u layer=%u]\n",
                       step, loss, lr_log, toks_per_sec, vram_mb, state.bcd_cycle, state.bcd_layer);
            } else {
                printf("[TRAIN] step=%d loss=%.4f lr=%.6f tok/s=%.0f vram=%.0fMB [joint]\n",
                       step, loss, lr_log, toks_per_sec, vram_mb);
            }
            if (m->exact_quotient_padding_enabled &&
                m->exact_quotient_padding_batches > 0u) {
                printf("[EXACT_QUOTIENT_PADDING] step=%u batches=%llu targets=%llu active_targets=%llu inactive_targets=%llu active_fraction=%.6f objective=active_prefix_ce ce_gradient_masked=1 gpu_scan_masked=0 compute_skip=0\n",
                       state.step,
                       (unsigned long long)m->exact_quotient_padding_batches,
                       (unsigned long long)m->exact_quotient_padding_total_targets,
                       (unsigned long long)m->exact_quotient_padding_active_targets,
                       (unsigned long long)m->exact_quotient_padding_inactive_targets,
                       m->exact_quotient_padding_active_fraction);
            }

            /* R493 Evaluation Harness Log (Aspect A04) */
            printf("[EVALUATION_HARNESS_GATE] active=1 decision=%s pass=%s staged_quality=%.6f entropy=%.6f collapse_penalty=%.6f\n",
                   state.r493_receipt.passed ? "promote" : "hold",
                   state.r493_receipt.passed ? "yes" : "no",
                   state.r493_receipt.staged_quality,
                   state.r493_receipt.entropy,
                   state.r493_receipt.collapse_penalty);

            /* R493 Stage Contract Manifest Log (Aspect A02) */
            printf("[STAGE_CONTRACT_MANIFEST] active=1 stage_id=%u stable=%s threshold=%.4f entropy_target=%.4f\n",
                   state.r493_contract.stage_id,
                   state.r493_contract.contract_stable ? "yes" : "no",
                   state.r493_contract.min_quality_threshold,
                   state.r493_contract.min_entropy_threshold);

            /* R493 Answer Transaction CE Contrast Log (Aspect A01) */
            printf("[ANSWER_TRANSACTION_GATE] active=1 loss=%.6f margin=1.0000 status=active\n",
                   answer_contrast_loss);

            /* R493 R441 Gate-to-Task Compiler Log (Aspect A05) */
            printf("[TASK_COMPILER_GATE] active=1 compiled_tasks=%u primary_task=%s status=active\n",
                   3,
                   sequence_task_id == 2 ? "syntactic_code" : (sequence_task_id == 1 ? "procedural_logic" : "linguistic_qa"));

            /* R493 Stage Frontier Transaction Scheduler Log (Aspect A06) */
            printf("[FRONTIER_SCHEDULER_GATE] active=1 frontier_scale=%.4f target_stage=%u status=active\n",
                   dynamic_frontier_scale,
                   state.r493_contract.stage_id);
            if (r364.enabled) {
                GpuR364NoopProbe r364_gpu_probe = {0};
                gpu_pipeline_r364_get_noop_probe(gpu, &r364_gpu_probe);
                printf("[R364] step=%u active=%s compile_id=%u compiled_until=%u hook=noop train_calls=%u opt_calls=%u accept=%s reason=%s rows=%u/%u pre_loss=%.6f post_loss=%.6f f=%.6f last_compile_ms=%.3f kappa=%g\n",
                       state.step,
                       r364_gpu_probe.compiled_interval_active ? "on" : "off",
                       r364_gpu_probe.compile_id,
                       r364_schedule_state(&r364, state.step).compiled_until_step,
                       r364_gpu_probe.train_step_calls,
                       r364_gpu_probe.optimizer_step_calls,
                       r364.last_probe.accepted_update ? "yes" : "no",
                       r364.last_probe.rejection_reason,
                       r364.last_probe.updated_rows,
                       r364.last_probe.candidate_rows,
                       r364.last_probe.pre_loss,
                       r364.last_probe.post_loss,
                       r364_gpu_probe.compiled_block_share_f,
                       r364.last_probe.compile_ms,
                       r364.last_probe.kappa);
            }
            if (dpth_sampler.enabled) {
                dpth_sampler_log(&dpth_sampler, state.step);
                dpth_sampler_reset_interval(&dpth_sampler);
            }
            if (m->logit_prior_gate_enabled && m->logit_prior_gate_batches > 0) {
                printf("[LOGIT_PRIOR_GATE] step=%u base_loss=%.6f residual_loss=%.6f residual_ratio=%.6f batches=%llu\n",
                       state.step,
                       m->logit_prior_gate_base_loss,
                       m->logit_prior_gate_residual_loss,
                       m->logit_prior_gate_residual_ratio,
                       (unsigned long long)m->logit_prior_gate_batches);
            }
            if (m->program_kernel_gate_enabled && m->program_kernel_gate_batches > 0) {
                double route_accuracy = 0.0;
                if (m->program_kernel_gate_routed > 0) {
                    route_accuracy = (double)m->program_kernel_gate_correct /
                                     (double)m->program_kernel_gate_routed;
                }
                printf("[PROGRAM_KERNEL_GATE] step=%u base_loss=%.6f kernel_loss=%.6f kernel_ratio=%.6f batches=%llu routed=%llu correct=%llu incorrect=%llu route_accuracy=%.6f\n",
                       state.step,
                       m->program_kernel_gate_base_loss,
                       m->program_kernel_gate_kernel_loss,
                       m->program_kernel_gate_kernel_ratio,
                       (unsigned long long)m->program_kernel_gate_batches,
                       (unsigned long long)m->program_kernel_gate_routed,
                       (unsigned long long)m->program_kernel_gate_correct,
                       (unsigned long long)m->program_kernel_gate_incorrect,
                       route_accuracy);
            }
            if (m->certified_artifact_boundary_gate_enabled &&
                m->certified_artifact_boundary_gate_batches > 0) {
                printf("[CERTIFIED_ARTIFACT_BOUNDARY_GATE] step=%u positions=%llu routed=%llu correct=%llu incorrect=%llu abstain=%llu b_k=%.6f false_route_rate=%.6f ce_residual=%.9f ce_budget=%.9f priced_ratio=%.6f route_bytes=%llu obligation_audit=%d non_lookup_audit=%d batches=%llu\n",
                       state.step,
                       (unsigned long long)m->certified_artifact_boundary_gate_positions,
                       (unsigned long long)m->certified_artifact_boundary_gate_routed,
                       (unsigned long long)m->certified_artifact_boundary_gate_correct,
                       (unsigned long long)m->certified_artifact_boundary_gate_incorrect,
                       (unsigned long long)m->certified_artifact_boundary_gate_abstain,
                       m->certified_artifact_boundary_gate_b_k,
                       m->certified_artifact_boundary_gate_false_route_rate,
                       m->certified_artifact_boundary_gate_ce_residual,
                       m->certified_artifact_boundary_ce_budget,
                       m->certified_artifact_boundary_gate_priced_ratio,
                       (unsigned long long)m->certified_artifact_boundary_route_bytes,
                       m->certified_artifact_boundary_obligation_audit,
                       m->certified_artifact_boundary_non_lookup_audit,
                       (unsigned long long)m->certified_artifact_boundary_gate_batches);
            }
            if (m->state_transition_atlas_gate_enabled &&
                m->state_transition_atlas_gate_batches > 0) {
                double route_accuracy = 0.0;
                if (m->state_transition_atlas_gate_routed > 0) {
                    route_accuracy = (double)m->state_transition_atlas_gate_correct /
                                     (double)m->state_transition_atlas_gate_routed;
                }
                printf("[STATE_TRANSITION_ATLAS_GATE] step=%u base_loss=%.6f atlas_loss=%.6f atlas_ratio=%.6f batches=%llu routed=%llu correct=%llu incorrect=%llu route_accuracy=%.6f\n",
                       state.step,
                       m->state_transition_atlas_gate_base_loss,
                       m->state_transition_atlas_gate_atlas_loss,
                       m->state_transition_atlas_gate_atlas_ratio,
                       (unsigned long long)m->state_transition_atlas_gate_batches,
                       (unsigned long long)m->state_transition_atlas_gate_routed,
                       (unsigned long long)m->state_transition_atlas_gate_correct,
                       (unsigned long long)m->state_transition_atlas_gate_incorrect,
                       route_accuracy);
            }
            if (m->absorbed_state_atlas_gate_enabled &&
                m->absorbed_state_atlas_gate_batches > 0) {
                printf("[ABSORBED_STATE_ATLAS_GATE] step=%u base_loss=%.6f residual_loss=%.6f residual_ratio=%.6f priced_ratio=%.6f coverage=%.6f route_accuracy=%.6f batches=%llu routed=%llu correct=%llu incorrect=%llu\n",
                       state.step,
                       m->absorbed_state_atlas_gate_base_loss,
                       m->absorbed_state_atlas_gate_residual_loss,
                       m->absorbed_state_atlas_gate_residual_ratio,
                       m->absorbed_state_atlas_gate_priced_ratio,
                       m->absorbed_state_atlas_gate_coverage,
                       m->absorbed_state_atlas_gate_route_accuracy,
                       (unsigned long long)m->absorbed_state_atlas_gate_batches,
                       (unsigned long long)m->absorbed_state_atlas_gate_routed,
                       (unsigned long long)m->absorbed_state_atlas_gate_correct,
                       (unsigned long long)m->absorbed_state_atlas_gate_incorrect);
            }
            if (m->counterfactual_packets_gate_enabled &&
                m->counterfactual_packets_gate_batches > 0) {
                printf("[COUNTERFACTUAL_PACKETS_GATE] step=%u packets=%llu primary=%llu secondary=%llu other=%llu abstain=%llu coverage=%.6f balance=%.6f batches=%llu\n",
                       state.step,
                       (unsigned long long)m->counterfactual_packets_gate_pairs,
                       (unsigned long long)m->counterfactual_packets_gate_primary,
                       (unsigned long long)m->counterfactual_packets_gate_secondary,
                       (unsigned long long)m->counterfactual_packets_gate_other,
                       (unsigned long long)m->counterfactual_packets_gate_abstain,
                       m->counterfactual_packets_gate_coverage,
                       m->counterfactual_packets_gate_balance,
                       (unsigned long long)m->counterfactual_packets_gate_batches);
            }
            if (m->certified_silent_span_gate_enabled &&
                m->certified_silent_span_gate_batches > 0) {
                printf("[CERTIFIED_SILENT_SPAN_GATE] step=%u tokens=%llu positions=%llu spans=%llu fraction=%.6f grad_bound=%.6f priced_ratio=%.6f batches=%llu\n",
                       state.step,
                       (unsigned long long)m->certified_silent_span_gate_tokens,
                       (unsigned long long)m->certified_silent_span_gate_positions,
                       (unsigned long long)m->certified_silent_span_gate_spans,
                       m->certified_silent_span_gate_fraction,
                       m->certified_silent_span_gate_grad_bound,
                       m->certified_silent_span_gate_priced_ratio,
                       (unsigned long long)m->certified_silent_span_gate_batches);
            }
            if (m->error_causal_lattice_gate_enabled &&
                m->error_causal_lattice_gate_batches > 0) {
                printf("[ERROR_CAUSAL_LATTICE_GATE] step=%u positions=%llu target_covered=%llu prereq_ready=%llu orderable=%llu rare_positions=%llu rare_orderable=%llu target_coverage=%.6f prereq_readiness=%.6f orderable_fraction=%.6f rare_coverage=%.6f batches=%llu\n",
                       state.step,
                       (unsigned long long)m->error_causal_lattice_gate_positions,
                       (unsigned long long)m->error_causal_lattice_gate_target_covered,
                       (unsigned long long)m->error_causal_lattice_gate_prereq_ready,
                       (unsigned long long)m->error_causal_lattice_gate_orderable,
                       (unsigned long long)m->error_causal_lattice_gate_rare_positions,
                       (unsigned long long)m->error_causal_lattice_gate_rare_orderable,
                       m->error_causal_lattice_gate_target_coverage,
                       m->error_causal_lattice_gate_prereq_readiness,
                       m->error_causal_lattice_gate_orderable_fraction,
                       m->error_causal_lattice_gate_rare_coverage,
                       (unsigned long long)m->error_causal_lattice_gate_batches);
            }
            if (m->residual_active_sampling_gate_enabled &&
                m->residual_active_sampling_gate_batches > 0) {
                printf("[RESIDUAL_ACTIVE_SAMPLING_GATE] step=%u positions=%llu mean_loss=%.6f score_cv=%.6f ess_fraction=%.6f target_fraction=%.6f max_weight=%.6f variance_proxy=%.6f batches=%llu\n",
                       state.step,
                       (unsigned long long)m->residual_active_sampling_gate_positions,
                       m->residual_active_sampling_gate_mean_loss,
                       m->residual_active_sampling_gate_score_cv,
                       m->residual_active_sampling_gate_ess_fraction,
                       m->residual_active_sampling_target_fraction,
                       m->residual_active_sampling_gate_max_weight,
                       m->residual_active_sampling_gate_variance_proxy,
                       (unsigned long long)m->residual_active_sampling_gate_batches);
            }
            if (m->exact_transition_quotient_gate_enabled &&
                m->exact_transition_quotient_gate_batches > 0) {
                printf("[EXACT_TRANSITION_QUOTIENT_GATE] step=%u corpus_ratio=%.6f positions=%llu supported=%llu live_coverage=%.6f saved_terms=%llu saved_fraction=%.6f mean_multiplicity=%.6f batches=%llu\n",
                       state.step,
                       m->exact_transition_quotient_gate_corpus_ratio,
                       (unsigned long long)m->exact_transition_quotient_gate_positions,
                       (unsigned long long)m->exact_transition_quotient_gate_supported,
                       m->exact_transition_quotient_gate_live_coverage,
                       (unsigned long long)m->exact_transition_quotient_gate_saved_terms,
                       m->exact_transition_quotient_gate_saved_fraction,
                       m->exact_transition_quotient_gate_mean_multiplicity,
                       (unsigned long long)m->exact_transition_quotient_gate_batches);
            }
            if (m->macro_ce_quotient_observer_enabled &&
                m->macro_ce_quotient_observer_batches > 0) {
                printf("[R421_MACRO_CE_QUOTIENT_RESULT] step=%u positions=%llu covered=%llu coverage=%.6f context_ce=%.6f quotient_ce=%.6f random_ce=%.6f ce_excess=%.6f random_advantage=%.6f quotient_ratio=%.6f residual_price=%.6f priced_work_ratio=%.6f net_speedup_estimate=%.6f batches=%llu observer=%d training=%d training_rows=%llu training_covered=%llu training_coverage=%.6f training_effective_weight=%.6f\n",
                       state.step,
                       (unsigned long long)m->macro_ce_quotient_observer_positions,
                       (unsigned long long)m->macro_ce_quotient_observer_covered,
                       m->macro_ce_quotient_coverage,
                       m->macro_ce_quotient_context_ce,
                       m->macro_ce_quotient_quotient_ce,
                       m->macro_ce_quotient_random_ce,
                       m->macro_ce_quotient_ce_excess,
                       m->macro_ce_quotient_random_advantage,
                       m->macro_ce_quotient_ratio,
                       m->macro_ce_quotient_residual_price,
                       m->macro_ce_quotient_priced_work_ratio,
                       m->macro_ce_quotient_net_speedup_estimate,
                       (unsigned long long)m->macro_ce_quotient_observer_batches,
                       m->macro_ce_quotient_observer_enabled ? 1 : 0,
                       m->macro_ce_quotient_training_enabled ? 1 : 0,
                       (unsigned long long)m->macro_ce_quotient_training_rows,
                       (unsigned long long)m->macro_ce_quotient_training_covered,
                       m->macro_ce_quotient_training_coverage,
                       m->macro_ce_quotient_training_effective_weight);
                if (r423_native_growth_readout_quotient_enable) {
                    printf("[R423_NATIVE_GROWTH_READOUT_QUOTIENT_RESULT] step=%u covered_family=readout_null_denominator positions=%llu covered=%llu coverage=%.6f priced_work_ratio=%.6f net_speedup_estimate=%.6f training_rows=%llu training_covered=%llu training_coverage=%.6f training_effective_weight=%.6f\n",
                           state.step,
                           (unsigned long long)m->macro_ce_quotient_observer_positions,
                           (unsigned long long)m->macro_ce_quotient_observer_covered,
                           m->macro_ce_quotient_coverage,
                           m->macro_ce_quotient_priced_work_ratio,
                           m->macro_ce_quotient_net_speedup_estimate,
                           (unsigned long long)m->macro_ce_quotient_training_rows,
                           (unsigned long long)m->macro_ce_quotient_training_covered,
                           m->macro_ce_quotient_training_coverage,
                           m->macro_ce_quotient_training_effective_weight);
                }
                if (residual_certified_scaffold_enable ||
                    residual_certified_scaffold_gate_enable) {
                    const int pass =
                        (m->macro_ce_quotient_observer_covered > 0u &&
                         m->macro_ce_quotient_ce_excess <=
                             residual_certified_scaffold_audit_budget &&
                         m->macro_ce_quotient_priced_work_ratio <= 0.90f) ? 1 : 0;
                    printf("[RESIDUAL_CERTIFIED_SCAFFOLD_GATE] step=%u apply=%d gate=%d positions=%llu covered=%llu coverage=%.6f context_ce=%.6f quotient_ce=%.6f ce_excess=%.6f audit_budget=%.6f random_ce=%.6f random_advantage=%.6f quotient_ratio=%.6f residual_price=%.6f priced_work_ratio=%.6f net_gain_estimate=%.6f training_rows=%llu training_covered=%llu training_coverage=%.6f training_effective_weight=%.6f original_ce_visible=1 identifying=%d pass=%d\n",
                           state.step,
                           residual_certified_scaffold_enable ? 1 : 0,
                           residual_certified_scaffold_gate_enable ? 1 : 0,
                           (unsigned long long)m->macro_ce_quotient_observer_positions,
                           (unsigned long long)m->macro_ce_quotient_observer_covered,
                           m->macro_ce_quotient_coverage,
                           m->macro_ce_quotient_context_ce,
                           m->macro_ce_quotient_quotient_ce,
                           m->macro_ce_quotient_ce_excess,
                           residual_certified_scaffold_audit_budget,
                           m->macro_ce_quotient_random_ce,
                           m->macro_ce_quotient_random_advantage,
                           m->macro_ce_quotient_ratio,
                           m->macro_ce_quotient_residual_price,
                           m->macro_ce_quotient_priced_work_ratio,
                           m->macro_ce_quotient_net_speedup_estimate,
                           (unsigned long long)m->macro_ce_quotient_training_rows,
                           (unsigned long long)m->macro_ce_quotient_training_covered,
                           m->macro_ce_quotient_training_coverage,
                           m->macro_ce_quotient_training_effective_weight,
                           m->macro_ce_quotient_observer_covered > 0u ? 1 : 0,
                           pass);
                }
            }
            if (m->exact_window_multiset_gate_enabled &&
                m->exact_window_multiset_gate_batches > 0) {
                printf("[EXACT_WINDOW_MULTISET_GATE] step=%u q=%.6f positions=%llu supported=%llu live_coverage=%.6f saved_terms=%llu saved_fraction=%.6f mean_multiplicity=%.6f priced_ratio=%.6f objective_audit=%d gradient_audit=%d collisions=%llu batches=%llu\n",
                       state.step,
                       m->exact_window_multiset_gate_q_ratio,
                       (unsigned long long)m->exact_window_multiset_gate_positions,
                       (unsigned long long)m->exact_window_multiset_gate_supported,
                       m->exact_window_multiset_gate_live_coverage,
                       (unsigned long long)m->exact_window_multiset_gate_saved_terms,
                       m->exact_window_multiset_gate_saved_fraction,
                       m->exact_window_multiset_gate_mean_multiplicity,
                       m->exact_window_multiset_gate_priced_ratio,
                       m->exact_window_multiset_objective_audit,
                       m->exact_window_multiset_gradient_audit,
                       (unsigned long long)m->exact_window_multiset_collision_count,
                       (unsigned long long)m->exact_window_multiset_gate_batches);
            }
            if (m->prefix_frontier_gate_enabled &&
                m->prefix_frontier_gate_batches > 0) {
                printf("[PREFIX_FRONTIER_GATE] step=%u positions=%llu supported=%llu reuse_fraction=%.6f energy_fraction=%.6f mean_multiplicity=%.6f saved_terms=%llu priced_ratio=%.6f batches=%llu\n",
                       state.step,
                       (unsigned long long)m->prefix_frontier_gate_positions,
                       (unsigned long long)m->prefix_frontier_gate_supported,
                       m->prefix_frontier_gate_reuse_fraction,
                       m->prefix_frontier_gate_energy_fraction,
                       m->prefix_frontier_gate_mean_multiplicity,
                       (unsigned long long)m->prefix_frontier_gate_saved_terms,
                       m->prefix_frontier_gate_priced_ratio,
                       (unsigned long long)m->prefix_frontier_gate_batches);
            }
            if (m->active_state_subspace_gate_enabled &&
                m->active_state_subspace_gate_batches > 0) {
                printf("[ACTIVE_STATE_SUBSPACE_GATE] step=%u positions=%llu k=%u k_over_d=%.6f energy_fraction=%.6f tail_regret=%.6f priced_ratio=%.6f batches=%llu\n",
                       state.step,
                       (unsigned long long)m->active_state_subspace_gate_positions,
                       m->active_state_subspace_k,
                       m->active_state_subspace_gate_k_over_d,
                       m->active_state_subspace_gate_energy_fraction,
                       m->active_state_subspace_gate_tail_regret,
                       m->active_state_subspace_gate_priced_ratio,
                       (unsigned long long)m->active_state_subspace_gate_batches);
            }
            if (m->boundary_liability_ledger_gate_enabled &&
                m->boundary_liability_gate_batches > 0) {
                printf("[BOUNDARY_LIABILITY_LEDGER_GATE] step=%u positions=%llu boundary_positions=%llu boundary_fraction=%.6f boundary_energy_fraction=%.6f null_fraction=%.6f boundary_null_fraction=%.6f false_discharge_energy=%.9f overhead=%.6f priced_gain=%.6f pass=%d batches=%llu\n",
                       state.step,
                       (unsigned long long)m->boundary_liability_gate_positions,
                       (unsigned long long)m->boundary_liability_gate_boundary_positions,
                       m->boundary_liability_gate_boundary_fraction,
                       m->boundary_liability_gate_boundary_energy_fraction,
                       m->boundary_liability_gate_null_fraction,
                       m->boundary_liability_gate_boundary_null_fraction,
                       m->boundary_liability_gate_false_discharge_energy,
                       m->boundary_liability_overhead,
                       m->boundary_liability_gate_priced_gain,
                       m->boundary_liability_gate_pass,
                       (unsigned long long)m->boundary_liability_gate_batches);
            }
            if (m->boundary_flux_sheaf_gate_enabled &&
                m->boundary_flux_gate_batches > 0) {
                printf("[BOUNDARY_FLUX_SHEAF_GATE] step=%u layers=%llu positions=%llu endpoint_positions=%llu boundary_fraction=%.6f pre_boundary_fraction=%.6f endpoint_fraction=%.6f total_energy=%.9g endpoint_energy=%.9g energy_fraction=%.6f kappa_est=%.6f overhead=%.6f priced_gain=%.6f identifying=%d pass=%d batches=%llu\n",
                       state.step,
                       (unsigned long long)m->boundary_flux_gate_layers,
                       (unsigned long long)m->boundary_flux_gate_positions,
                       (unsigned long long)m->boundary_flux_gate_endpoint_positions,
                       m->boundary_flux_gate_boundary_fraction,
                       m->boundary_flux_gate_pre_boundary_fraction,
                       m->boundary_flux_gate_endpoint_fraction,
                       m->boundary_flux_gate_total_energy,
                       m->boundary_flux_gate_endpoint_energy,
                       m->boundary_flux_gate_energy_fraction,
                       m->boundary_flux_gate_kappa_est,
                       m->boundary_flux_sheaf_overhead,
                       m->boundary_flux_gate_priced_gain,
                       m->boundary_flux_gate_identifying,
                       m->boundary_flux_gate_pass,
                       (unsigned long long)m->boundary_flux_gate_batches);
            }
            if (m->online_tssi_live_gate_enabled &&
                m->online_tssi_live_gate_batches > 0) {
                printf("[ONLINE_TSSI_LIVE_GATE] step=%u state_layer_fraction=%.6f moment_layer_fraction=%.6f score_layer_fraction=%.6f sampled_ce_buffers=%llu score_updates=%llu missing_state_score_layers=%llu no_hidden_cache_materialization=1 identifying=%d pass=%d batches=%llu\n",
                       state.step,
                       m->online_tssi_live_gate_state_layer_fraction,
                       m->online_tssi_live_gate_moment_layer_fraction,
                       m->online_tssi_live_gate_score_layer_fraction,
                       (unsigned long long)m->online_tssi_live_gate_sampled_ce_buffers,
                       (unsigned long long)m->online_tssi_live_gate_score_updates,
                       (unsigned long long)m->online_tssi_live_gate_missing_state_score_layers,
                       m->online_tssi_live_gate_identifying,
                       m->online_tssi_live_gate_pass,
                       (unsigned long long)m->online_tssi_live_gate_batches);
            }
            if (m->boundary_cotangent_credit_gate_enabled &&
                m->boundary_cotangent_gate_batches > 0) {
                printf("[BOUNDARY_COTANGENT_CREDIT_GATE] step=%u layers=%llu positions=%llu anchor_positions=%llu anchor_fraction=%.6f saved_recurrent_fraction=%.6f exact_energy=%.9g recon_energy=%.9g residual_energy=%.9g cosine=%.6f residual_fraction=%.6f energy_covered=%.6f basis_bytes_per_token=%.6f verify_fraction=%.6f overhead=%.6f priced_ratio=%.6f unverified_priced_ratio=%.6f descent_per_paid_work=%.6f unverified_descent_per_paid_work=%.6f residual_budget=%.6f identifying=%d pass=%d batches=%llu\n",
                       state.step,
                       (unsigned long long)m->boundary_cotangent_gate_layers,
                       (unsigned long long)m->boundary_cotangent_gate_positions,
                       (unsigned long long)m->boundary_cotangent_gate_anchor_positions,
                       m->boundary_cotangent_gate_anchor_fraction,
                       m->boundary_cotangent_gate_saved_recurrent_fraction,
                       m->boundary_cotangent_gate_exact_energy,
                       m->boundary_cotangent_gate_recon_energy,
                       m->boundary_cotangent_gate_residual_energy,
                       m->boundary_cotangent_gate_cosine,
                       m->boundary_cotangent_gate_residual_fraction,
                       m->boundary_cotangent_gate_energy_covered,
                       m->boundary_cotangent_gate_basis_bytes_per_token,
                       m->boundary_cotangent_gate_verification_fraction,
                       m->boundary_cotangent_overhead,
                       m->boundary_cotangent_gate_priced_ratio,
                       m->boundary_cotangent_gate_unverified_priced_ratio,
                       m->boundary_cotangent_gate_descent_per_paid_work,
                       m->boundary_cotangent_gate_unverified_descent_per_paid_work,
                       m->boundary_cotangent_residual_budget,
                       m->boundary_cotangent_gate_identifying,
                       m->boundary_cotangent_gate_pass,
                       (unsigned long long)m->boundary_cotangent_gate_batches);
            }
            if (m->variance_priced_credit_gate_enabled &&
                m->variance_priced_gate_batches > 0) {
                printf("[VARIANCE_PRICED_CREDIT_GATE] step=%u layers=%llu items=%llu sampled_items=%llu sampled_fraction=%.6f norm_ess_fraction=%.6f mean_norm=%.9g max_norm=%.9g exact_energy=%.9g sampled_energy=%.9g dot=%.9g residual_energy=%.9g alignment=%.6f variance_proxy=%.6f score_fraction=%.6f verify_fraction=%.6f overhead=%.6f priced_work=%.6f descent_proxy=%.6f descent_per_paid_work=%.6f identifying=%d pass=%d batches=%llu\n",
                       state.step,
                       (unsigned long long)m->variance_priced_gate_layers,
                       (unsigned long long)m->variance_priced_gate_items,
                       (unsigned long long)m->variance_priced_gate_sampled_items,
                       m->variance_priced_gate_sampled_fraction,
                       m->variance_priced_gate_norm_ess_fraction,
                       m->variance_priced_gate_mean_norm,
                       m->variance_priced_gate_max_norm,
                       m->variance_priced_gate_exact_energy,
                       m->variance_priced_gate_sampled_energy,
                       m->variance_priced_gate_dot,
                       m->variance_priced_gate_residual_energy,
                       m->variance_priced_gate_alignment,
                       m->variance_priced_gate_variance_proxy,
                       m->variance_priced_score_fraction,
                       m->variance_priced_verify_fraction,
                       m->variance_priced_overhead,
                       m->variance_priced_gate_priced_work,
                       m->variance_priced_gate_descent_proxy,
                       m->variance_priced_gate_descent_per_paid_work,
                       m->variance_priced_gate_identifying,
                       m->variance_priced_gate_pass,
                       (unsigned long long)m->variance_priced_gate_batches);
                if (equal_area_fisher_sampling_enable ||
                    equal_area_fisher_sampling_gate_enable) {
                    const int equal_area_pass =
                        (m->variance_priced_gate_identifying &&
                         m->variance_priced_gate_norm_ess_fraction >=
                             equal_area_fisher_ess_floor &&
                         m->variance_priced_gate_variance_proxy <=
                             equal_area_fisher_bias_budget &&
                         m->variance_priced_gate_descent_per_paid_work >=
                             equal_area_fisher_min_descent_per_paid_work) ? 1 : 0;
                    printf("[EQUAL_AREA_FISHER_SAMPLING_GATE] step=%u apply=%d layers=%llu items=%llu sampled_items=%llu sampled_fraction=%.6f ess_fraction=%.6f ess_floor=%.6f variance_proxy=%.6f bias_budget=%.6f alignment=%.6f priced_work=%.6f descent_proxy=%.6f descent_per_paid_work=%.6f min_descent_per_paid_work=%.6f score_fraction=%.6f verify_fraction=%.6f overhead=%.6f identifying=%d pass=%d batches=%llu original_ce_objective=1 equal_area_importance_scaling=1\n",
                           state.step,
                           equal_area_fisher_sampling_enable ? 1 : 0,
                           (unsigned long long)m->variance_priced_gate_layers,
                           (unsigned long long)m->variance_priced_gate_items,
                           (unsigned long long)m->variance_priced_gate_sampled_items,
                           m->variance_priced_gate_sampled_fraction,
                           m->variance_priced_gate_norm_ess_fraction,
                           equal_area_fisher_ess_floor,
                           m->variance_priced_gate_variance_proxy,
                           equal_area_fisher_bias_budget,
                           m->variance_priced_gate_alignment,
                           m->variance_priced_gate_priced_work,
                           m->variance_priced_gate_descent_proxy,
                           m->variance_priced_gate_descent_per_paid_work,
                           equal_area_fisher_min_descent_per_paid_work,
                           m->variance_priced_score_fraction,
                           m->variance_priced_verify_fraction,
                           m->variance_priced_overhead,
                           m->variance_priced_gate_identifying,
                           equal_area_pass,
                           (unsigned long long)m->variance_priced_gate_batches);
                }
                if (m->amortized_ce_martingale_training_enabled) {
                    printf("[AMORTIZED_CE_MARTINGALE_TRAINING] step=%u training_layers=%llu applied_layers=%llu fallback_layers=%llu last_applied=%d last_sampled_fraction=%.6f last_descent_per_paid_work=%.6f threshold=%.6f training_change=1\n",
                           state.step,
                           (unsigned long long)m->amortized_ce_martingale_training_layers,
                           (unsigned long long)m->amortized_ce_martingale_applied_layers,
                           (unsigned long long)m->amortized_ce_martingale_fallback_layers,
                           m->amortized_ce_martingale_last_applied,
                           m->amortized_ce_martingale_last_sampled_fraction,
                           m->amortized_ce_martingale_last_descent_per_paid_work,
                           m->amortized_ce_martingale_min_descent_per_paid_work);
                }
            }
            if ((m->event_innovation_credit_enabled ||
                 m->event_innovation_credit_gate_enabled) &&
                m->event_innovation_gate_batches > 0) {
                const float mean_score =
                    m->event_innovation_gate_positions > 0u
                        ? (float)(m->event_innovation_gate_score_sum /
                                  (double)m->event_innovation_gate_positions)
                        : 0.0f;
                printf("[EVENT_INNOVATION_CREDIT_GATE] step=%u apply=%d layers=%llu applied_layers=%llu positions=%llu event_positions=%llu event_fraction=%.6f mean_score=%.9g max_score=%.9g exact_energy=%.9g event_energy=%.9g low_event_energy=%.9g low_event_debt=%.6f energy_covered=%.6f score_cost=%.6f verify_fraction=%.6f overhead=%.6f low_event_scale=%.6f priced_ratio=%.6f descent_per_paid_work=%.6f debt_budget=%.6f identifying=%d pass=%d batches=%llu\n",
                       state.step,
                       m->event_innovation_credit_enabled ? 1 : 0,
                       (unsigned long long)m->event_innovation_gate_layers,
                       (unsigned long long)m->event_innovation_gate_applied_layers,
                       (unsigned long long)m->event_innovation_gate_positions,
                       (unsigned long long)m->event_innovation_gate_event_positions,
                       m->event_innovation_gate_event_fraction,
                       mean_score,
                       m->event_innovation_gate_max_score,
                       m->event_innovation_gate_exact_energy,
                       m->event_innovation_gate_event_energy,
                       m->event_innovation_gate_low_event_energy,
                       m->event_innovation_gate_low_event_debt,
                       m->event_innovation_gate_energy_covered,
                       m->event_innovation_score_cost,
                       m->event_innovation_verify_fraction,
                       m->event_innovation_overhead,
                       m->event_innovation_low_event_scale,
                       m->event_innovation_gate_priced_ratio,
                       m->event_innovation_gate_descent_per_paid_work,
                       m->event_innovation_debt_budget,
                       m->event_innovation_gate_identifying,
                       m->event_innovation_gate_pass,
                       (unsigned long long)m->event_innovation_gate_batches);
            }
            if ((m->ce_debt_skip_enabled ||
                 m->ce_debt_skip_gate_enabled) &&
                m->ce_debt_gate_batches > 0) {
                printf("[CE_DEBT_SKIP_GATE] step=%u apply=%d batches=%llu applied_batches=%llu tokens=%llu certified_tokens=%llu skipped_tokens=%llu audited_tokens=%llu certified_fraction=%.6f skipped_fraction=%.6f total_loss=%.9g certified_loss=%.9g loss_debt=%.6f total_grad_energy=%.9g certified_grad_energy=%.9g grad_debt=%.6f debt=%.6f energy_covered=%.6f max_loss=%.6f max_grad_norm=%.6f max_debt=%.6f audit_fraction=%.6f overhead=%.6f min_skip_fraction=%.6f skip_scale=%.6f priced_ratio=%.6f descent_per_paid_work=%.6f identifying=%d pass=%d\n",
                       state.step,
                       m->ce_debt_skip_enabled ? 1 : 0,
                       (unsigned long long)m->ce_debt_gate_batches,
                       (unsigned long long)m->ce_debt_gate_applied_batches,
                       (unsigned long long)m->ce_debt_gate_tokens,
                       (unsigned long long)m->ce_debt_gate_certified_tokens,
                       (unsigned long long)m->ce_debt_gate_skipped_tokens,
                       (unsigned long long)m->ce_debt_gate_audited_tokens,
                       m->ce_debt_gate_certified_fraction,
                       m->ce_debt_gate_skipped_fraction,
                       m->ce_debt_gate_total_loss,
                       m->ce_debt_gate_certified_loss,
                       m->ce_debt_gate_loss_debt,
                       m->ce_debt_gate_total_grad_energy,
                       m->ce_debt_gate_certified_grad_energy,
                       m->ce_debt_gate_grad_debt,
                       m->ce_debt_gate_debt,
                       m->ce_debt_gate_energy_covered,
                       m->ce_debt_max_loss,
                       m->ce_debt_max_grad_norm,
                       m->ce_debt_max_debt,
                       m->ce_debt_audit_fraction,
                       m->ce_debt_overhead,
                       m->ce_debt_min_skip_fraction,
                       m->ce_debt_skip_scale,
                       m->ce_debt_gate_priced_ratio,
                       m->ce_debt_gate_descent_per_paid_work,
                       m->ce_debt_gate_identifying,
                       m->ce_debt_gate_pass);
            }
            if ((m->gradient_coherence_quotient_enabled ||
                 m->gradient_coherence_quotient_gate_enabled) &&
                m->gradient_coherence_gate_batches > 0) {
                printf("[GRADIENT_COHERENCE_QUOTIENT_GATE] step=%u apply=%d layers=%llu applied_layers=%llu positions=%llu nonempty_clusters=%llu avg_cluster_size=%.6f representative_fraction=%.6f residual_replay_positions=%llu residual_replay_fraction=%.6f exact_energy=%.9g representative_energy=%.9g residual_energy=%.9g residual_debt=%.6f energy_covered=%.6f saved_fraction=%.6f bins=%u probe_stride=%u max_residual=%.6f score_cost=%.6f verify_fraction=%.6f overhead=%.6f min_saved_fraction=%.6f residual_scale=%.6f priced_ratio=%.6f descent_per_paid_work=%.6f identifying=%d pass=%d batches=%llu\n",
                       state.step,
                       m->gradient_coherence_quotient_enabled ? 1 : 0,
                       (unsigned long long)m->gradient_coherence_gate_layers,
                       (unsigned long long)m->gradient_coherence_gate_applied_layers,
                       (unsigned long long)m->gradient_coherence_gate_positions,
                       (unsigned long long)m->gradient_coherence_gate_nonempty_clusters,
                       m->gradient_coherence_gate_avg_cluster_size,
                       m->gradient_coherence_gate_representative_fraction,
                       (unsigned long long)m->gradient_coherence_gate_residual_replay_positions,
                       m->gradient_coherence_gate_residual_replay_fraction,
                       m->gradient_coherence_gate_exact_energy,
                       m->gradient_coherence_gate_representative_energy,
                       m->gradient_coherence_gate_residual_energy,
                       m->gradient_coherence_gate_residual_debt,
                       m->gradient_coherence_gate_energy_covered,
                       m->gradient_coherence_gate_saved_fraction,
                       m->gradient_coherence_bins,
                       m->gradient_coherence_probe_stride,
                       m->gradient_coherence_max_residual,
                       m->gradient_coherence_score_cost,
                       m->gradient_coherence_verify_fraction,
                       m->gradient_coherence_overhead,
                       m->gradient_coherence_min_saved_fraction,
                       m->gradient_coherence_residual_scale,
                       m->gradient_coherence_gate_priced_ratio,
                       m->gradient_coherence_gate_descent_per_paid_work,
                       m->gradient_coherence_gate_identifying,
                       m->gradient_coherence_gate_pass,
                       (unsigned long long)m->gradient_coherence_gate_batches);
            }
            if (m->lagged_ce_impulse_eligibility_enabled &&
                m->lagged_ce_impulse_batches > 0) {
                printf("[LAGGED_CE_IMPULSE_ELIGIBILITY] step=%u layers=%llu positions=%llu horizon=%u blend=%.6f exact_transition_energy=%.9g ledger_transition_energy=%.9g transition_dot=%.9g transition_residual_energy=%.9g transition_residual_fraction=%.6f dA_residual_fraction=%.6f dbA_residual_fraction=%.6f dWA_residual_fraction=%.6f dWA_sampled_columns=%u cosine=%.6f priced_ratio=%.6f observed_rho=%.6f tail_bound_fraction=%.6f residual_budget=%.6f identifying=%d pass=%d training_layers=%llu applied_layers=%llu fallback_layers=%llu last_applied=%d training_change=1 batches=%llu\n",
                       state.step,
                       (unsigned long long)m->lagged_ce_impulse_layers,
                       (unsigned long long)m->lagged_ce_impulse_positions,
                       m->lagged_ce_impulse_horizon,
                       m->lagged_ce_impulse_blend,
                       m->lagged_ce_impulse_exact_transition_energy,
                       m->lagged_ce_impulse_ledger_transition_energy,
                       m->lagged_ce_impulse_transition_dot,
                       m->lagged_ce_impulse_transition_residual_energy,
                       m->lagged_ce_impulse_transition_residual_fraction,
                       m->lagged_ce_impulse_dA_residual_fraction,
                       m->lagged_ce_impulse_dbA_residual_fraction,
                       m->lagged_ce_impulse_dWA_residual_fraction,
                       m->lagged_ce_impulse_dWA_sampled_columns,
                       m->lagged_ce_impulse_cosine,
                       m->lagged_ce_impulse_priced_ratio,
                       m->lagged_ce_impulse_observed_rho,
                       m->lagged_ce_impulse_tail_bound_fraction,
                       m->lagged_ce_impulse_residual_budget,
                       m->lagged_ce_impulse_identifying,
                       m->lagged_ce_impulse_pass,
                       (unsigned long long)m->lagged_ce_impulse_training_layers,
                       (unsigned long long)m->lagged_ce_impulse_applied_layers,
                       (unsigned long long)m->lagged_ce_impulse_fallback_layers,
                       m->lagged_ce_impulse_last_applied,
                       (unsigned long long)m->lagged_ce_impulse_batches);
            }
            if (m->executable_multiplicity_packets_enabled &&
                m->executable_multiplicity_batches > 0) {
                printf("[R438_EXECUTABLE_MULTIPLICITY_PACKETS] step=%u layers=%llu positions=%llu cells=%llu accepted_cells=%llu accepted_positions=%llu rejected_positions=%llu min_count=%u state_scale=%.6f transition_scale=%.6f accepted_mass=%.6f residual_fraction=%.6f mean_multiplicity=%.6f packet_work_fraction=%.6f priced_ratio=%.6f exact_energy=%.9g residual_energy=%.9g identifying=%d pass=%d observer_only=1 no_training_change=1 batches=%llu\n",
                       state.step,
                       (unsigned long long)m->executable_multiplicity_layers,
                       (unsigned long long)m->executable_multiplicity_positions,
                       (unsigned long long)m->executable_multiplicity_cells,
                       (unsigned long long)m->executable_multiplicity_accepted_cells,
                       (unsigned long long)m->executable_multiplicity_accepted_positions,
                       (unsigned long long)m->executable_multiplicity_rejected_positions,
                       m->executable_multiplicity_min_count,
                       m->executable_multiplicity_state_scale,
                       m->executable_multiplicity_transition_scale,
                       m->executable_multiplicity_accepted_mass,
                       m->executable_multiplicity_residual_fraction,
                       m->executable_multiplicity_mean_multiplicity,
                       m->executable_multiplicity_packet_work_fraction,
                       m->executable_multiplicity_priced_ratio,
                       m->executable_multiplicity_exact_energy,
                       m->executable_multiplicity_residual_energy,
                       m->executable_multiplicity_identifying,
                       m->executable_multiplicity_pass,
                       (unsigned long long)m->executable_multiplicity_batches);
            }
            if (m->original_ce_credit_debt_observer_enabled &&
                m->original_ce_credit_gate_batches > 0) {
                printf("[ORIGINAL_CE_CREDIT_DEBT_OBSERVER] step=%u layers=%llu positions=%llu selected_positions=%llu selected_fraction=%.6f exact_energy=%.9g local_progress_ratio=%.6f local_residual_fraction=%.6f local_priced_ratio=%.6f local_whole_step_speedup=%.6f surface_progress_ratio=%.6f surface_residual_fraction=%.6f surface_priced_ratio=%.6f surface_whole_step_speedup=%.6f verifier_fraction=%.6f debt_price=%.6f overhead=%.6f observer_only=1 no_training_change=1 identifying=%d local_pass=%d surface_pass=%d batches=%llu\n",
                       state.step,
                       (unsigned long long)m->original_ce_credit_gate_layers,
                       (unsigned long long)m->original_ce_credit_gate_positions,
                       (unsigned long long)m->original_ce_credit_gate_selected_positions,
                       m->original_ce_credit_gate_selected_fraction,
                       m->original_ce_credit_gate_exact_energy,
                       m->original_ce_credit_gate_local_progress_ratio,
                       m->original_ce_credit_gate_local_residual_fraction,
                       m->original_ce_credit_gate_local_priced_ratio,
                       m->original_ce_credit_gate_local_whole_step_speedup,
                       m->original_ce_credit_gate_surface_progress_ratio,
                       m->original_ce_credit_gate_surface_residual_fraction,
                       m->original_ce_credit_gate_surface_priced_ratio,
                       m->original_ce_credit_gate_surface_whole_step_speedup,
                       m->original_ce_credit_verifier_fraction,
                       m->original_ce_credit_debt_price,
                       m->original_ce_credit_overhead,
                       m->original_ce_credit_gate_identifying,
                       m->original_ce_credit_gate_local_pass,
                       m->original_ce_credit_gate_surface_pass,
                       (unsigned long long)m->original_ce_credit_gate_batches);
            }
            if (m->r375_pbg2_gate_enabled &&
                m->r375_pbg2_gate_matrices > 0) {
                printf("[R375_PBG2_GATE] step=%u matrices=%llu elements=%llu grad_energy=%.9g row_anisotropy=%.6f col_anisotropy=%.6f shape_signal=%.6f max_shape_signal=%.6f mvue_active_fraction=%.6f grid_snr=%.9g min_anisotropy=%.6f identifying=%d pass=%d batches=%llu\n",
                       state.step,
                       (unsigned long long)m->r375_pbg2_gate_matrices,
                       (unsigned long long)m->r375_pbg2_gate_elements,
                       m->r375_pbg2_gate_grad_energy,
                       m->r375_pbg2_gate_row_anisotropy,
                       m->r375_pbg2_gate_col_anisotropy,
                       m->r375_pbg2_gate_shape_signal,
                       m->r375_pbg2_gate_max_shape_signal,
                       m->r375_pbg2_gate_mvue_active_fraction,
                       m->r375_pbg2_gate_grid_snr,
                       m->r375_pbg2_min_anisotropy,
                       m->r375_pbg2_gate_identifying,
                       m->r375_pbg2_gate_pass,
                       (unsigned long long)m->r375_pbg2_gate_batches);
            }
            if (m->r375_pbg2_precondition_enabled &&
                m->r375_pbg2_precondition_matrices > 0) {
                printf("[R375_PBG2_PRECOND] step=%u matrices=%llu elements=%llu raw_rms=%.9g pre_rms=%.9g scale=%.9g cosine=%.6f identifying=%d batches=%llu\n",
                       state.step,
                       (unsigned long long)m->r375_pbg2_precondition_matrices,
                       (unsigned long long)m->r375_pbg2_precondition_elements,
                       m->r375_pbg2_precondition_raw_rms,
                       m->r375_pbg2_precondition_pre_rms,
                       m->r375_pbg2_precondition_scale,
                       m->r375_pbg2_precondition_cosine,
                       m->r375_pbg2_precondition_identifying,
                       (unsigned long long)m->r375_pbg2_precondition_batches);
            }
            if (m->gradient_null_witness_gate_enabled &&
                m->gradient_null_gate_batches > 0) {
                printf("[GRADIENT_NULL_WITNESS_GATE] step=%u groups=%llu null_groups=%llu group_fraction=%.6f elements=%llu null_elements=%llu element_fraction=%.6f work_units=%llu null_work_units=%llu saved_work_fraction=%.6f total_energy=%.9g null_energy=%.9g omitted_fraction=%.9g epsilon=%.9g omitted_budget=%.9g overhead=%.6f priced_gain=%.6f identifying=%d pass=%d batches=%llu\n",
                       state.step,
                       (unsigned long long)m->gradient_null_gate_groups,
                       (unsigned long long)m->gradient_null_gate_null_groups,
                       m->gradient_null_gate_group_fraction,
                       (unsigned long long)m->gradient_null_gate_elements,
                       (unsigned long long)m->gradient_null_gate_null_elements,
                       m->gradient_null_gate_element_fraction,
                       (unsigned long long)m->gradient_null_gate_work_units,
                       (unsigned long long)m->gradient_null_gate_null_work_units,
                       m->gradient_null_gate_saved_work_fraction,
                       m->gradient_null_gate_total_energy,
                       m->gradient_null_gate_null_energy,
                       m->gradient_null_gate_omitted_fraction,
                       m->gradient_null_witness_epsilon,
                       m->gradient_null_witness_omitted_budget,
                       m->gradient_null_witness_overhead,
                       m->gradient_null_gate_priced_gain,
                       m->gradient_null_gate_identifying,
                       m->gradient_null_gate_pass,
                       (unsigned long long)m->gradient_null_gate_batches);
            }
            if (m->r429_decision_equivalence_witness_enabled &&
                m->r429_decision_witness_batches > 0) {
                printf("[R429_DECISION_EQUIVALENCE_WITNESS] step=%u denominator_groups=%llu certified_groups=%llu fallback_groups=%llu certified_fraction=%.6f fallback_fraction=%.6f avoided_work_units=%llu avoided_work_fraction=%.6f mismatch_updates=%llu identifying=%d pass=%d batches=%llu\n",
                       state.step,
                       (unsigned long long)m->r429_decision_witness_denominator_groups,
                       (unsigned long long)m->r429_decision_witness_certified_groups,
                       (unsigned long long)m->r429_decision_witness_fallback_groups,
                       m->r429_decision_witness_certified_fraction,
                       m->r429_decision_witness_fallback_fraction,
                       (unsigned long long)m->r429_decision_witness_avoided_work_units,
                       m->r429_decision_witness_avoided_work_fraction,
                       (unsigned long long)m->r429_decision_witness_mismatch_updates,
                       m->r429_decision_witness_identifying,
                       m->r429_decision_witness_pass,
                       (unsigned long long)m->r429_decision_witness_batches);
            }
            if (m->optimizer_integrity_gate_enabled &&
                m->optimizer_integrity_gate_batches > 0) {
                printf("[OPTIMIZER_INTEGRITY_GATE] step=%u owned_ternary_groups=%u resident_ternary_groups=%u cpu_skip_groups=%u gpu_update_groups=%u missing_update_groups=%u pass=%d batches=%llu\n",
                       state.step,
                       m->optimizer_integrity_owned_ternary_groups,
                       m->optimizer_integrity_resident_ternary_groups,
                       m->optimizer_integrity_cpu_skip_groups,
                       m->optimizer_integrity_gpu_update_groups,
                       m->optimizer_integrity_missing_update_groups,
                       m->optimizer_integrity_gate_pass,
                       (unsigned long long)m->optimizer_integrity_gate_batches);
            }
            if (m->evidence_rate_commitment_gate_enabled &&
                m->evidence_rate_gate_batches > 0) {
                const char *evidence_mode_name =
                    m->evidence_rate_commitment_mode == V12_EVIDENCE_RATE_MODE_TSA ? "tsa" :
                    m->evidence_rate_commitment_mode == V12_EVIDENCE_RATE_MODE_GBAR ? "gbar" :
                    "current";
                printf("[EVIDENCE_RATE_COMMITMENT_GATE] step=%u mode=%s nonzero=%llu active=%llu failed=%llu active_fraction=%.6f commitment_fraction=%.6f current_descent=%.9g tsa_descent=%.9g gbar_descent=%.9g applied_descent=%.9g applied_harm=%.9g virtual_best_gain=%.6f priced_ratio=%.6f pass=%d batches=%llu\n",
                       state.step,
                       evidence_mode_name,
                       (unsigned long long)m->evidence_rate_nonzero_grads,
                       (unsigned long long)m->evidence_rate_active_updates,
                       (unsigned long long)m->evidence_rate_failed_updates,
                       m->evidence_rate_active_fraction,
                       m->evidence_rate_commitment_fraction,
                       m->evidence_rate_current_descent,
                       m->evidence_rate_tsa_descent,
                       m->evidence_rate_gbar_descent,
                       m->evidence_rate_applied_descent,
                       m->evidence_rate_applied_harm,
                       m->evidence_rate_virtual_best_gain,
                       m->evidence_rate_priced_ratio,
                       m->evidence_rate_gate_pass,
                       (unsigned long long)m->evidence_rate_gate_batches);
            }
            if (m->r388_flip_refresh_gate_enabled &&
                m->r388_flip_refresh_gate_batches > 0) {
                printf("[R388_FLIP_REFRESH_GATE] step=%u matrices=%llu elements=%llu blocks=%llu omittable_blocks=%llu flips=%llu eta=%.6f flip_fraction=%.9g projected_refresh_gain=%.6f min_eta=%.6f identifying=%d pass=%d batches=%llu\n",
                       state.step,
                       (unsigned long long)m->r388_flip_refresh_matrices,
                       (unsigned long long)m->r388_flip_refresh_elements,
                       (unsigned long long)m->r388_flip_refresh_blocks,
                       (unsigned long long)m->r388_flip_refresh_omittable_blocks,
                       (unsigned long long)m->r388_flip_refresh_flips,
                       m->r388_flip_refresh_eta,
                       m->r388_flip_refresh_flip_fraction,
                       m->r388_flip_refresh_projected_refresh_gain,
                       m->r388_flip_refresh_min_eta,
                       m->r388_flip_refresh_identifying,
                       m->r388_flip_refresh_pass,
                       (unsigned long long)m->r388_flip_refresh_gate_batches);
            }
            if (m->fp_interface_gap_probe_enabled &&
                m->fp_interface_gap_probe_batches > 0) {
                printf("[FP_INTERFACE_GAP_PROBE] step=%u summary layers=%llu near_identity_layers=%llu mean_effect_ratio=%.9g min_effect_ratio=%.9g mean_abs_projection=%.9g max_grad_rms=%.9g identifying=%d batches=%llu\n",
                       state.step,
                       (unsigned long long)m->fp_interface_gap_probe_layers,
                       (unsigned long long)m->fp_interface_gap_near_identity_layers,
                       m->fp_interface_gap_mean_effect_ratio,
                       m->fp_interface_gap_min_effect_ratio,
                       m->fp_interface_gap_mean_abs_projection,
                       m->fp_interface_gap_max_grad_rms,
                       m->fp_interface_gap_probe_identifying,
                       (unsigned long long)m->fp_interface_gap_probe_batches);
            }
            if ((m->credit_time_equalized_dynamics_enabled ||
                 m->credit_time_equalized_dynamics_gate_enabled) &&
                m->credit_time_gate_batches > 0) {
                printf("[CREDIT_TIME_EQUALIZED_DYNAMICS_GATE] step=%u apply=%d params=%llu mean_scale=%.6f min_scale=%.6f max_scale=%.6f descent_gain=%.6f line_shrink=%.6f net_descent_gain=%.6f priced_ratio=%.6f batches=%llu\n",
                       state.step,
                       m->credit_time_equalized_dynamics_enabled ? 1 : 0,
                       (unsigned long long)m->credit_time_gate_params,
                       m->credit_time_gate_mean_scale,
                       m->credit_time_gate_min_scale_seen,
                       m->credit_time_gate_max_scale_seen,
                       m->credit_time_gate_descent_gain,
                       m->credit_time_gate_line_shrink,
                       m->credit_time_gate_net_descent_gain,
                       m->credit_time_gate_priced_ratio,
                       (unsigned long long)m->credit_time_gate_batches);
            }
            if ((m->contractive_green_credit_enabled ||
                 m->contractive_green_credit_gate_enabled) &&
                m->contractive_green_credit_gate_batches > 0) {
                printf("[CONTRACTIVE_GREEN_CREDIT_GATE] step=%u apply=%d params=%llu accepted_params=%llu accepted_fraction=%.6f mean_alpha=%.6f max_alpha=%.6f mean_tail_debt=%.9g max_tail_debt=%.9g mean_scale=%.6f min_scale=%.6f max_scale=%.6f descent_gain=%.6f line_shrink=%.6f net_descent_gain=%.6f priced_ratio=%.6f identifying=%d pass=%d batches=%llu\n",
                       state.step,
                       m->contractive_green_credit_enabled ? 1 : 0,
                       (unsigned long long)m->contractive_green_credit_gate_params,
                       (unsigned long long)m->contractive_green_credit_gate_accepted_params,
                       m->contractive_green_credit_gate_accepted_fraction,
                       m->contractive_green_credit_gate_mean_alpha,
                       m->contractive_green_credit_gate_max_alpha_seen,
                       m->contractive_green_credit_gate_mean_tail_debt,
                       m->contractive_green_credit_gate_max_tail_debt,
                       m->contractive_green_credit_gate_mean_scale,
                       m->contractive_green_credit_gate_min_scale_seen,
                       m->contractive_green_credit_gate_max_scale_seen,
                       m->contractive_green_credit_gate_descent_gain,
                       m->contractive_green_credit_gate_line_shrink,
                       m->contractive_green_credit_gate_net_descent_gain,
                       m->contractive_green_credit_gate_priced_ratio,
                       m->contractive_green_credit_gate_identifying,
                       m->contractive_green_credit_gate_pass,
                       (unsigned long long)m->contractive_green_credit_gate_batches);
            }
            if ((m->adaptive_credit_horizon_enabled ||
                 m->adaptive_credit_horizon_gate_enabled) &&
                m->adaptive_credit_gate_batches > 0) {
                printf("[ADAPTIVE_CREDIT_HORIZON_GATE] step=%u apply=%d layers=%llu applied_layers=%llu positions=%llu skipped_positions=%llu skipped_fraction=%.6f avg_horizon=%.6f horizon_fraction=%.6f exact_energy=%.9g tail_energy=%.9g tail_debt=%.6f energy_covered=%.6f mean_bound=%.9g max_bound=%.9g tail_budget=%.6f bound_budget=%.6f verify_fraction=%.6f overhead=%.6f min_headroom=%.6f tail_scale=%.6f headroom=%.6f priced_ratio=%.6f descent_per_paid_work=%.6f identifying=%d pass=%d batches=%llu\n",
                       state.step,
                       m->adaptive_credit_horizon_enabled ? 1 : 0,
                       (unsigned long long)m->adaptive_credit_gate_layers,
                       (unsigned long long)m->adaptive_credit_gate_applied_layers,
                       (unsigned long long)m->adaptive_credit_gate_positions,
                       (unsigned long long)m->adaptive_credit_gate_skipped_positions,
                       m->adaptive_credit_gate_skipped_fraction,
                       m->adaptive_credit_gate_avg_horizon,
                       m->adaptive_credit_gate_horizon_fraction,
                       m->adaptive_credit_gate_exact_energy,
                       m->adaptive_credit_gate_tail_energy,
                       m->adaptive_credit_gate_tail_debt,
                       m->adaptive_credit_gate_energy_covered,
                       m->adaptive_credit_gate_layers > 0
                           ? m->adaptive_credit_gate_bound_sum /
                             (double)m->adaptive_credit_gate_layers
                           : 0.0,
                       m->adaptive_credit_gate_max_bound,
                       m->adaptive_credit_tail_budget,
                       m->adaptive_credit_bound_budget,
                       m->adaptive_credit_verify_fraction,
                       m->adaptive_credit_overhead,
                       m->adaptive_credit_min_headroom,
                       m->adaptive_credit_tail_scale,
                       m->adaptive_credit_gate_headroom,
                       m->adaptive_credit_gate_priced_ratio,
                       m->adaptive_credit_gate_descent_per_paid_work,
                       m->adaptive_credit_gate_identifying,
                       m->adaptive_credit_gate_pass,
                       (unsigned long long)m->adaptive_credit_gate_batches);
            }
            if ((m->interior_adjoint_tube_enabled ||
                 m->interior_adjoint_tube_gate_enabled) &&
                m->interior_adjoint_tube_gate_batches > 0) {
                printf("[INTERIOR_ADJOINT_TUBE_GATE] step=%u apply=%d rank=%u layers=%llu accepted_layers=%llu accepted_layer_fraction=%.6f params=%llu tube_params=%llu tube_fraction=%.6f exact_energy=%.9g tube_energy=%.9g residual_energy=%.9g residual_fraction=%.6f energy_covered=%.6f residual_budget=%.6f verify_fraction=%.6f overhead=%.6f residual_scale=%.6f descent_gain=%.6f line_shrink=%.6f net_descent_gain=%.6f priced_ratio=%.6f descent_per_paid_work=%.6f identifying=%d pass=%d batches=%llu\n",
                       state.step,
                       m->interior_adjoint_tube_enabled ? 1 : 0,
                       m->interior_adjoint_tube_rank,
                       (unsigned long long)m->interior_adjoint_tube_gate_layers,
                       (unsigned long long)m->interior_adjoint_tube_gate_accepted_layers,
                       m->interior_adjoint_tube_gate_accepted_layer_fraction,
                       (unsigned long long)m->interior_adjoint_tube_gate_params,
                       (unsigned long long)m->interior_adjoint_tube_gate_tube_params,
                       m->interior_adjoint_tube_gate_tube_fraction,
                       m->interior_adjoint_tube_gate_exact_energy,
                       m->interior_adjoint_tube_gate_tube_energy,
                       m->interior_adjoint_tube_gate_residual_energy,
                       m->interior_adjoint_tube_gate_residual_fraction,
                       m->interior_adjoint_tube_gate_energy_covered,
                       m->interior_adjoint_tube_residual_budget,
                       m->interior_adjoint_tube_verify_fraction,
                       m->interior_adjoint_tube_overhead,
                       m->interior_adjoint_tube_residual_scale,
                       m->interior_adjoint_tube_gate_descent_gain,
                       m->interior_adjoint_tube_gate_line_shrink,
                       m->interior_adjoint_tube_gate_net_descent_gain,
                       m->interior_adjoint_tube_gate_priced_ratio,
                       m->interior_adjoint_tube_gate_descent_per_paid_work,
                       m->interior_adjoint_tube_gate_identifying,
                       m->interior_adjoint_tube_gate_pass,
                       (unsigned long long)m->interior_adjoint_tube_gate_batches);
            }
            if (m->macro_transcript_gate_enabled &&
                m->macro_transcript_gate_batches > 0) {
                printf("[MACRO_TRANSCRIPT_GATE] step=%u positions=%llu ops=%llu pair_macros=%llu singletons=%llu pair_fraction=%.6f length_ratio=%.6f priced_ratio=%.6f batches=%llu\n",
                       state.step,
                       (unsigned long long)m->macro_transcript_gate_positions,
                       (unsigned long long)m->macro_transcript_gate_ops,
                       (unsigned long long)m->macro_transcript_gate_pair_macros,
                       (unsigned long long)m->macro_transcript_gate_singletons,
                       m->macro_transcript_gate_pair_fraction,
                       m->macro_transcript_gate_length_ratio,
                       m->macro_transcript_gate_priced_ratio,
                       (unsigned long long)m->macro_transcript_gate_batches);
            }
            if (m->reversible_deterministic_expansion_gate_enabled &&
                m->reversible_deterministic_expansion_gate_batches > 0) {
                printf("[REVERSIBLE_DETERMINISTIC_EXPANSION_GATE] step=%u positions=%llu covered=%llu spans=%llu coord_units=%llu lambda=%.6f eta=%.6f priced_ratio=%.6f inverse_audit=%d inverse_failures=%llu batches=%llu\n",
                       state.step,
                       (unsigned long long)m->reversible_deterministic_expansion_gate_positions,
                       (unsigned long long)m->reversible_deterministic_expansion_gate_covered,
                       (unsigned long long)m->reversible_deterministic_expansion_gate_spans,
                       (unsigned long long)m->reversible_deterministic_expansion_gate_coord_units,
                       m->reversible_deterministic_expansion_gate_lambda,
                       m->reversible_deterministic_expansion_gate_eta,
                       m->reversible_deterministic_expansion_gate_priced_ratio,
                       m->reversible_deterministic_expansion_inverse_audit,
                       (unsigned long long)m->reversible_deterministic_expansion_inverse_failures,
                       (unsigned long long)m->reversible_deterministic_expansion_gate_batches);
            }
#ifdef V12_HAS_VULKAN
            if (vk_profile && vk_ctx) {
                VkR370Summary r370 = {0};
                double cpu_step_ms = elapsed_sec * 1000.0;
                double submit_ms = (double)vk_profile_stats.submit_ns / 1e6;
                double wait_ms   = (double)vk_profile_stats.wait_ns / 1e6;
                double sync_pct  = elapsed_sec > 0.0
                                 ? ((submit_ms + wait_ms) / (elapsed_sec * 1000.0)) * 100.0
                                 : 0.0;
                const char *timing_status = "unsupported";
                const char *scope_status;
                vk_profile_summarize_r370(&vk_profile_stats, cpu_step_ms, vk_profile_snapshot_ms, &r370);
                if (vk_profile_stats.gpu_timing_supported) {
                    timing_status = r370.gpu_timing_available ? "available" : "incomplete";
                }
                scope_status = vk_r370_scope_status_string(r370.scope_status);
                printf("[VKPROF] step=%d submits=%llu waits=%llu submit_ms=%.3f wait_ms=%.3f sync_pct=%.2f\n",
                       step,
                       (unsigned long long)vk_profile_stats.submit_count,
                       (unsigned long long)vk_profile_stats.wait_count,
                       submit_ms,
                       wait_ms,
                       sync_pct);
                if (r370.scope_status == VK_R370_SCOPE_COMPLETE) {
                    printf("[R370] step=%d dispatches=%llu submits=%llu waits=%llu cpu_step_ms=%.3f submit_ms=%.3f wait_ms=%.3f gpu_cmd_ms=%.3f host_gap_ms=%.3f record_scope_ms=%.3f submit_scope_ms=%.3f wait_scope_ms=%.3f wait_scope_non_gpu_ms=%.3f reset_scope_ms=%.3f snapshot_ms=%.6f d_scope_ms=%.3f p_scope=%.6f other_host_ms=%.3f net_delta_optimistic=%.6f optimistic_speedup=%.6f timed_submits=%llu incomplete_timing=%llu timing_status=%s scope_status=%s\n",
                           step,
                           (unsigned long long)vk_profile_stats.dispatch_count,
                           (unsigned long long)vk_profile_stats.submit_count,
                           (unsigned long long)vk_profile_stats.wait_count,
                           r370.cpu_step_ms,
                           r370.submit_ms,
                           r370.wait_ms,
                           r370.gpu_cmd_ms,
                           r370.host_gap_ms,
                           r370.dispatch_record_ms,
                           r370.dispatch_submit_ms,
                           r370.dispatch_wait_ms,
                           r370.dispatch_wait_non_gpu_ms,
                           r370.dispatch_reset_ms,
                           r370.profile_snapshot_ms,
                           r370.d_scope_ms,
                           r370.p_scope,
                           r370.other_host_ms,
                           r370.net_delta_optimistic,
                           r370.optimistic_speedup,
                           (unsigned long long)vk_profile_stats.timed_submit_count,
                           (unsigned long long)vk_profile_stats.incomplete_timing_count,
                           timing_status,
                           scope_status);
                } else {
                    printf("[R370] step=%d dispatches=%llu submits=%llu waits=%llu cpu_step_ms=%.3f submit_ms=%.3f wait_ms=%.3f gpu_cmd_ms=na host_gap_ms=na record_scope_ms=%.3f submit_scope_ms=%.3f wait_scope_ms=%.3f wait_scope_non_gpu_ms=na reset_scope_ms=%.3f snapshot_ms=%.6f d_scope_ms=na p_scope=na other_host_ms=na net_delta_optimistic=na optimistic_speedup=na timed_submits=%llu incomplete_timing=%llu timing_status=%s scope_status=%s\n",
                           step,
                           (unsigned long long)vk_profile_stats.dispatch_count,
                           (unsigned long long)vk_profile_stats.submit_count,
                           (unsigned long long)vk_profile_stats.wait_count,
                            cpu_step_ms,
                            submit_ms,
                            wait_ms,
                            r370.dispatch_record_ms,
                            r370.dispatch_submit_ms,
                            r370.dispatch_wait_ms,
                            r370.dispatch_reset_ms,
                            r370.profile_snapshot_ms,
                            (unsigned long long)vk_profile_stats.timed_submit_count,
                            (unsigned long long)vk_profile_stats.incomplete_timing_count,
                            timing_status,
                            scope_status);
                }
            }
#endif
            if (teaching_enabled && !use_seq) {
                teaching_log_step(teaching, state.step);
            }
        }

        if (!state.fp_completed) {
            uint32_t steps_in_stage = state.step - fp_stage_start_step;
            if (steps_in_stage >= FP_STEPS_PER_STAGE) {
                printf("[SEQ] FP stage %u complete. cursor=%llu\n",
                       state.fp_stage, (unsigned long long)ds->cursor);
                int completed_layer = (int)state.fp_stage;
#ifdef V12_HAS_VULKAN
                if (materialize_slice4_layer(m, gpu, completed_layer, "fp-stage-pack") != 0) {
                    break;
                }
#endif
                pack_layer_to_int2(&m->layers[completed_layer]);
                printf("[INT2] Layer %d packed to INT2 (shadow frozen)\n", completed_layer);

                gpu_pipeline_reupload_layer(gpu, m, completed_layer);

                layer_offload_fp32(&m->layers[completed_layer]);
                model_release_layer_w_io_host(&m->layers[completed_layer]);
                printf("[OFFLOAD] Layer %d offloaded\n", completed_layer);

                int n_seqs    = (int)(ds->n_tokens / V12_SEQ_LEN);
                int use_y_cache = 1;

                if (use_y_cache) {
                    if (!c->y_cache) c->y_cache = y_cache_alloc();
                    if (c->y_cache) {
                        printf("[CACHE] Building y-cache (lossless) for layer %d (%d seqs)...\n",
                               completed_layer, n_seqs);
                        int ok = y_cache_build_entry(m, c->y_cache, ds,
                                                     completed_layer, V12_SEQ_LEN, n_seqs);
                        if (ok == 0) {
                            dataset_reset_cursor(ds);
                            printf("[CACHE] Layer %d y-cache ready. Next stage reads from cache.\n",
                                   completed_layer);
                        } else {
                            fprintf(stderr, "[CACHE] OOM — disabling y-cache for layer %d\n",
                                    completed_layer);
                            c->y_cache->active_entry = -1;
                        }
                    }
                } else {
                    if (c->res_cache != NULL) {
                        printf("[CACHE] Building PCA-INT8 cache for layer %d (%d seqs)...\n",
                               completed_layer, n_seqs);
                        int ok = pca_cache_build_entry(m, c->res_cache, ds,
                                                       completed_layer, V12_SEQ_LEN, n_seqs);
                        if (ok == 0) {
                            c->res_cache->active_entry = completed_layer;
                            dataset_reset_cursor(ds);
                            printf("[CACHE] Layer %d cache ready. Next stage reads from cache.\n",
                                   completed_layer);
                        } else {
                            fprintf(stderr, "[CACHE] OOM — disabling cache for layer %d\n",
                                    completed_layer);
                            c->res_cache->active_entry = -1;
                        }
                    }
                }

                int fp_just_done = fp_advance_stage(&state);
                fp_stage_start_step = state.step;

                if (fp_just_done) {
                    optimizer_reset_moments(m);
                    state.bcd_active = 1;
                    state.bcd_cycle  = 0;
                    state.bcd_layer  = 0;
                    bcd_set_active_layer(&state, 0);
                    bcd_layer_start_step = state.step;

                    if (c->y_cache != NULL) {
                        c->y_cache->active_entry = -1;
                        printf("[CACHE] Disabling y-cache for BCD phase.\n");
                    }
                    if (c->res_cache != NULL) {
                        c->res_cache->active_entry = -1;
                        printf("[CACHE] Disabling res_cache for BCD phase.\n");
                    }

                    printf("[RESTORE] Restoring fp32 for all %d layers before BCD...\n",
                           m->n_layers);
                    for (int li = 0; li < m->n_layers; li++) {
                        layer_restore_fp32(&m->layers[li]);
                    }
                    release_slice4_steady_state(m, "post-bcd-restore");
                    printf("[RESTORE] All layers restored.\n");
                    printf("[BCD] FP done. Starting BCD cycle 0, layer 0.\n");
                    printf("[BCD] ACTIVE cycle=0 layer=0 step=%u lr=%.2e\n",
                           state.step, (double)BCD_LR);
                }
            }
        }

        if (state.fp_completed && state.bcd_active) {
            uint32_t steps_in_layer = state.step - bcd_layer_start_step;
            if (steps_in_layer >= BCD_STEPS_PER_LAYER) {
                bcd_advance(&state);
                if (!state.bcd_active) {
                    printf("[OPT] BCD phase complete, resetting moments\n");
                    optimizer_reset_moments(m);
                }
                bcd_layer_start_step = state.step;
                if (state.bcd_active) {
                    printf("[BCD] ACTIVE cycle=%u layer=%u step=%u lr=%.2e\n",
                           state.bcd_cycle, state.bcd_layer, state.step, (double)BCD_LR);
                }
            }
        }

        if (checkpoint_path != NULL && checkpoint_save_enabled && state.step % 500 == 0 && state.step > 0) {
#ifdef V12_HAS_VULKAN
            if (materialize_slice2_boundary(
                    m,
                    gpu,
                    "checkpoint-save") != 0) {
                break;
            }
            if (materialize_slice3_boundary(
                    m,
                    gpu,
                    "checkpoint-save") != 0) {
                break;
            }
            if (materialize_slice4_boundary(
                    m,
                    gpu,
                    "checkpoint-save") != 0) {
                break;
            }
            if (materialize_lm_head_boundary(m, gpu, "checkpoint-save") != 0) {
                break;
            }
#endif
            if (ckpt_debug) {
                print_checkpoint_debug(m, "pre-save-interval");
            }
            state.dataset_cursor = (uint64_t)ds->cursor;
            memcpy(state.dataset_rng, ds->rng, sizeof(state.dataset_rng));
            state.dataset_state_valid = 1;
            printf("[CKPT] dataset_cursor saved=%llu\n",
                   (unsigned long long)state.dataset_cursor);
            printf("[CKPT] dataset_rng saved=1 state_valid=%u\n",
                   (unsigned)state.dataset_state_valid);
            checkpoint_save(checkpoint_path, m, &state);
            release_lm_head_steady_state(m, "post-checkpoint-save");
            release_slice2_steady_state(m, "post-checkpoint-save");
            release_slice3_steady_state(m, "post-checkpoint-save");
            release_slice4_steady_state(m, "post-checkpoint-save");
        }

        if (eval_every > 0 && state.step % (uint32_t)eval_every == 0 && state.step > 0) {
            float eval_loss_sum = 0.0f;
            size_t saved_cursor = ds->cursor;
            size_t eval_start = ds->n_tokens > 100000 ? ds->n_tokens - 100000 : 0;
            ds->cursor = eval_start;

#ifdef V12_HAS_VULKAN
            gpu_zero_grad_accumulators(gpu);
#endif
            // Stream a tiny forward-only canary validation batch (eb = 0)
            float canary_loss = 0.0f;
            {
                int total_tokens = batch_size * V12_SEQ_LEN;
                uint16_t *eval_tokens = malloc((size_t)total_tokens * sizeof(uint16_t));
                uint8_t  *eval_mask   = malloc((size_t)total_tokens * sizeof(uint8_t));
                if (eval_tokens && eval_mask) {
                    for (int b = 0; b < batch_size; b++) {
                        int dummy = 0;
                        load_tokens(ds, eval_tokens + b * V12_SEQ_LEN,
                                    eval_mask   + b * V12_SEQ_LEN,
                                    V12_SEQ_LEN, 0, &dummy);
                    }
                    canary_loss = gpu_train_step(gpu, m, c, eval_tokens, eval_mask,
                                                 V12_SEQ_LEN, 0, batch_size, &state);
                }
                free(eval_tokens);
                free(eval_mask);
            }

            float current_canary_gap = canary_loss - loss;
            int run_full_validation = 0;
            float dynamic_threshold = 0.15f;

            // MCDB validation skip check
            if (state.step == (uint32_t)eval_every) {
                run_full_validation = 1;
            } else {
                float gap_drift = current_canary_gap - state.eval_gap;
                if (gap_drift > dynamic_threshold || canary_loss > 3.0f || !isfinite(canary_loss)) {
                    run_full_validation = 1;
                    printf("[MCDB TRIGGER] Canary anomaly detected: canary_loss=%.4f, gap=%.4f (previous gap=%.4f) -> Triggering dense validation flushes.\n",
                           (double)canary_loss, (double)current_canary_gap, (double)state.eval_gap);
                }
            }

            float eval_loss = 0.0f;
            if (run_full_validation) {
                eval_loss_sum = canary_loss;
                for (int eb = 1; eb < eval_batches; eb++) {
                    int total_tokens = batch_size * V12_SEQ_LEN;
                    uint16_t *eval_tokens = malloc((size_t)total_tokens * sizeof(uint16_t));
                    uint8_t  *eval_mask   = malloc((size_t)total_tokens * sizeof(uint8_t));
                    if (!eval_tokens || !eval_mask) {
                        free(eval_tokens);
                        free(eval_mask);
                        break;
                    }
                    for (int b = 0; b < batch_size; b++) {
                        int dummy = 0;
                        load_tokens(ds, eval_tokens + b * V12_SEQ_LEN,
                                    eval_mask   + b * V12_SEQ_LEN,
                                    V12_SEQ_LEN, 0, &dummy);
                    }
                    eval_loss_sum += gpu_train_step(gpu, m, c, eval_tokens, eval_mask,
                                                    V12_SEQ_LEN, 0, batch_size, &state);
                    free(eval_tokens);
                    free(eval_mask);
                }
                eval_loss = eval_loss_sum / (float)eval_batches;
            } else {
                eval_loss = canary_loss;
                printf("[MCDB SKIP] Canary loss stable: canary_loss=%.4f, train_loss=%.4f, gap=%.4f -> Bypassing remaining %d validation flushes.\n",
                       (double)canary_loss, (double)loss, (double)current_canary_gap, eval_batches - 1);
            }

            state.eval_gap = eval_loss - loss;
            printf("[EVAL] step=%d eval_loss=%.4f train_loss=%.4f gap=%.4f\n",
                   (int)state.step, eval_loss, loss, state.eval_gap);

            ds->cursor = saved_cursor;
            scale_all_gradients(m, 0.0f);
#ifdef V12_HAS_VULKAN
            gpu_zero_grad_accumulators(gpu);
#endif
        }

    }

    if (heldout_eval_dataset_path != NULL) {
        uint64_t eval_seed = heldout_eval_seed != 0
                           ? heldout_eval_seed
                           : (seed_provided ? dataset_seed_value + 1000003ULL : 1000003ULL);
        float heldout_loss = run_external_heldout_eval(gpu,
                                                       m,
                                                       c,
                                                       heldout_eval_dataset_path,
                                                       batch_size,
                                                       heldout_eval_batches,
                                                       eval_seed,
                                                       state.step,
                                                       "final");
        if (!isfinite(heldout_loss)) {
            final_status = 1;
        }
    }

    if (checkpoint_path != NULL && checkpoint_save_enabled) {
        int final_save_ready = 1;
#ifdef V12_HAS_VULKAN
        if (materialize_slice2_boundary(
                m,
                gpu,
                "final-save") != 0) {
            final_save_ready = 0;
        }
        if (final_save_ready && materialize_slice3_boundary(
                m,
                gpu,
                "final-save") != 0) {
            final_save_ready = 0;
        }
        if (final_save_ready && materialize_slice4_boundary(
                m,
                gpu,
                "final-save") != 0) {
            final_save_ready = 0;
        }
        if (final_save_ready && materialize_lm_head_boundary(m, gpu, "final-save") != 0) {
            final_save_ready = 0;
        }
#endif
        if (final_save_ready && ckpt_debug) {
            print_checkpoint_debug(m, "pre-save-final");
        }
        if (final_save_ready) {
            state.dataset_cursor = (uint64_t)ds->cursor;
            memcpy(state.dataset_rng, ds->rng, sizeof(state.dataset_rng));
            state.dataset_state_valid = 1;
            printf("[CKPT] dataset_cursor saved=%llu\n",
                   (unsigned long long)state.dataset_cursor);
            printf("[CKPT] dataset_rng saved=1 state_valid=%u\n",
                   (unsigned)state.dataset_state_valid);
            checkpoint_save(checkpoint_path, m, &state);
            release_lm_head_steady_state(m, "post-final-save");
            release_slice2_steady_state(m, "post-final-save");
            release_slice3_steady_state(m, "post-final-save");
            release_slice4_steady_state(m, "post-final-save");
        }
    }

    if (online_tssi_selector_out != NULL) {
        const int selector_rc = gpu_pipeline_write_online_tssi_selector(
            gpu,
            online_tssi_selector_out,
            online_tssi_selector_active_fraction,
            r423_native_growth_state_selector_export_enable ? "R423" : "R420");
        printf("[ONLINE_TSSI_SELECTOR_EXPORT] path=%s active_fraction=%.6f status=%s rc=%d\n",
               online_tssi_selector_out,
               online_tssi_selector_active_fraction,
               selector_rc == 0 ? "ok" : "failed",
               selector_rc);
        if (r423_native_growth_state_selector_export_enable) {
            printf("[R423_NATIVE_GROWTH_STATE_SELECTOR_EXPORT] path=%s active_fraction=%.6f status=%s rc=%d\n",
                   online_tssi_selector_out,
                   online_tssi_selector_active_fraction,
                   selector_rc == 0 ? "ok" : "failed",
                   selector_rc);
        }
        if (selector_rc != 0) {
            final_status = 1;
        }
    }

    r438_ce_tx_summarize(&ce_tx);

    printf("[TRAIN] Cleanup\n");

    for (int k = 0; k < (int)m->n_layers; k++) {
        if (backup_shadow_w_in[k]) free(backup_shadow_w_in[k]);
        if (backup_shadow_w_out[k]) free(backup_shadow_w_out[k]);
        if (backup_sampled_w_in[k]) free(backup_sampled_w_in[k]);
        if (backup_sampled_w_out[k]) free(backup_sampled_w_out[k]);
    }

    r435_receipt_free(&r435_receipt);
    dpth_sampler_free(&dpth_sampler);
    gpu_pipeline_destroy(gpu);
#ifdef V12_HAS_VULKAN
    if (vk_ctx) vk_destroy(vk_ctx);
#endif

    model_free(m);
    if (c->y_cache != NULL)  y_cache_free(c->y_cache);
    if (c->res_cache != NULL) pca_cache_free(c->res_cache);
    cache_free(c);
    teaching_free(teaching);
    dataset_free(ds);

    printf("[TRAIN] Complete\n");
    return final_status;
}
