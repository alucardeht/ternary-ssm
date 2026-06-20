#include "optimizer.h"
#include "optimizer_f3_measure.h"
#include "types.h"
#include "model.h"
#include "aist_gate.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

/* R355 AIST runtime knob. Only the symbol storage lives here; the
 * declaration and the inline gate helpers live in aist_gate.h. When
 * V12_AIST_ENABLED == 0 the symbol is neither declared nor defined,
 * so the optimizer object file is byte-identical to the pre-R355
 * build (enforced by tests/ci_aist_flag_off_parity.sh). */
#if V12_AIST_ENABLED
float v12_aist_epsilon = 0.0f;
#endif

/* === ADAM OPTIMIZER CONSTANTS AND MACRO ================================== */

#define ADAM_BETA1   0.9f
#define ADAM_BETA2   0.999f
#define ADAM_EPS     1e-8f

#define R398_PAFE_WINDOW 16

typedef struct {
    double g[R398_PAFE_WINDOW];
    double t[R398_PAFE_WINDOW];
    int count;
} R398PafeBlockWindow;

static int r398_pafe_probe_active(void) {
    static int initialized = 0;
    static int active = 0;
    if (!initialized) {
        const char *env = getenv("R398_PAFE_PROBE");
        active = (env != NULL && strcmp(env, "0") != 0 && env[0] != '\0');
        initialized = 1;
        if (active) {
            printf("[R398_PAFE_PROBE] active env=R398_PAFE_PROBE window=%d\n",
                   R398_PAFE_WINDOW);
        }
    }
    return active;
}

static double r398_pafe_layer_dot(const Model *model, const SSMLayer *layer,
                                  int layer_idx, int *active_count,
                                  int *excluded_count) {
    double dot = 0.0;
    int count = 0;
    int excluded = 0;

    const int n_in = V12_D_MODEL * V12_D_STATE;
    const int n_out = V12_D_STATE * V12_D_MODEL;

    if (!(model->gpu_opt_flags & GPU_OPT_W_IN)) {
        for (int i = 0; i < n_in; i++) {
            dot += (double)layer->dw_in[i] * (double)SHADOW_DECODE(layer->shadow_w_in[i]);
        }
        count += n_in;
    } else {
        excluded += n_in;
    }

    if (!(model->gpu_opt_flags & GPU_OPT_W_OUT)) {
        for (int i = 0; i < n_out; i++) {
            dot += (double)layer->dw_out[i] * (double)SHADOW_DECODE(layer->shadow_w_out[i]);
        }
        count += n_out;
    } else {
        excluded += n_out;
    }

    for (int i = 0; i < V12_D_STATE; i++) {
        dot += (double)layer->d_sigma[i] * (double)layer->sigma[i];
        dot += (double)layer->dB[i] * (double)layer->B[i];
    }
    count += 2 * V12_D_STATE;

    if (!(model->gpu_opt_flags & GPU_OPT_A)) {
        for (int i = 0; i < V12_D_STATE; i++) {
            dot += (double)layer->dA[i] * (double)layer->A[i];
        }
        count += V12_D_STATE;
    } else {
        excluded += V12_D_STATE;
    }

    if (layer->dC_bilinear != NULL && layer->shadow_C_bilinear != NULL) {
        for (int i = 0; i < V12_D_STATE; i++) {
            dot += (double)layer->dC_bilinear[i] *
                   (double)SHADOW_DECODE(layer->shadow_C_bilinear[i]);
        }
        count += V12_D_STATE;
    }

    if (layer->dC != NULL && layer->C != NULL) {
        const int n_c = V12_D_STATE * V12_D_STATE;
        for (int i = 0; i < n_c; i++) {
            dot += (double)layer->dC[i] * (double)layer->C[i];
        }
        count += n_c;
    }

    if (!(model->gpu_opt_flags & GPU_OPT_CONV)) {
        const int n_conv = V12_CONV_KERNEL * V12_D_STATE;
        for (int i = 0; i < n_conv; i++) {
            dot += (double)layer->d_conv_w[i] *
                   (double)SHADOW_DECODE(layer->shadow_w_conv[i]);
        }
        count += n_conv;
    } else {
        excluded += V12_CONV_KERNEL * V12_D_STATE;
    }

    if (!(model->gpu_opt_flags & GPU_OPT_WA)) {
        const int n_wa = V12_D_STATE * V12_D_STATE;
        for (int i = 0; i < n_wa; i++) {
            dot += (double)layer->dW_A[i] * (double)layer->W_A[i];
        }
        for (int i = 0; i < V12_D_STATE; i++) {
            dot += (double)layer->db_A[i] * (double)layer->b_A[i];
        }
        count += n_wa + V12_D_STATE;
    } else {
        excluded += V12_D_STATE * V12_D_STATE + V12_D_STATE;
    }

    if (layer->mtp_U != NULL && layer->d_mtp_U != NULL &&
        layer->mtp_V != NULL && layer->d_mtp_V != NULL) {
        const int n_mtp = V12_D_STATE * V12_MTP_RANK;
        for (int i = 0; i < n_mtp; i++) {
            dot += (double)layer->d_mtp_U[i] * (double)layer->mtp_U[i];
            dot += (double)layer->d_mtp_V[i] * (double)layer->mtp_V[i];
        }
        count += 2 * n_mtp;
    }

    dot += (double)model->d_tau_mux[layer_idx] * (double)model->tau_mux[layer_idx];
    count += 1;

    if (active_count != NULL) *active_count = count;
    if (excluded_count != NULL) *excluded_count = excluded;
    return dot;
}

static double r398_pafe_median(double *values, int n) {
    for (int i = 1; i < n; i++) {
        double v = values[i];
        int j = i - 1;
        while (j >= 0 && values[j] > v) {
            values[j + 1] = values[j];
            j--;
        }
        values[j + 1] = v;
    }
    if (n <= 0) return 0.0;
    if ((n & 1) != 0) return values[n / 2];
    return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

static void r398_pafe_maybe_summary(int step, R398PafeBlockWindow *windows) {
    double eff[V12_MAX_LAYERS];
    double t_med[V12_MAX_LAYERS];
    double eff_sorted[V12_MAX_LAYERS];
    int valid = 0;
    int positive = 0;
    double total_t = 0.0;
    double top_t = 0.0;

    for (int k = 0; k < V12_N_LAYERS; k++) {
        if (windows[k].count < R398_PAFE_WINDOW) continue;
        double g_tmp[R398_PAFE_WINDOW];
        double t_tmp[R398_PAFE_WINDOW];
        for (int i = 0; i < R398_PAFE_WINDOW; i++) {
            g_tmp[i] = windows[k].g[i];
            t_tmp[i] = windows[k].t[i];
        }
        double g_med = r398_pafe_median(g_tmp, R398_PAFE_WINDOW);
        double t = r398_pafe_median(t_tmp, R398_PAFE_WINDOW);
        if (t <= 0.0) continue;
        valid++;
        total_t += t;
        if (g_med > 0.0) {
            eff[positive] = g_med / t;
            t_med[positive] = t;
            eff_sorted[positive] = eff[positive];
            positive++;
        }
    }

    const char *verdict = "INVALID";
    const char *reason = "none";
    double m = 0.0, h = 0.0, spread = 0.0, s = 0.0, pass_score = 0.0;

    if (valid < 3) {
        reason = "fewer_than_three_valid_blocks";
    } else if (positive <= 0) {
        reason = "no_positive_efficiency_blocks";
    } else {
        double med_tmp[V12_MAX_LAYERS];
        for (int i = 0; i < positive; i++) med_tmp[i] = eff[i];
        m = r398_pafe_median(med_tmp, positive);
        h = eff[0];
        for (int i = 1; i < positive; i++) {
            if (eff[i] > h) h = eff[i];
        }
        if (m <= 0.0 || total_t <= 0.0) {
            reason = "non_positive_median_or_time";
        } else {
            spread = h / m;
            for (int i = 0; i < positive; i++) {
                if (eff[i] >= 1.20 * m) top_t += t_med[i];
            }
            s = top_t / total_t;
            pass_score = 0.5 * s * (spread - 1.0);
            if (spread < 1.20) {
                verdict = "KILL";
                reason = "insufficient_efficiency_spread";
            } else if (s < 0.10) {
                verdict = "KILL";
                reason = "trivial_reallocatable_share";
            } else if (pass_score < 0.10) {
                verdict = "KILL";
                reason = "below_project_floor";
            } else {
                verdict = "PASS";
                reason = "passed_project_floor";
            }
        }
    }

    printf("[R398_PAFE_SUMMARY] step=%d block_count_valid=%d "
           "block_count_positive=%d median_efficiency_M=%.9e "
           "top_efficiency_H=%.9e spread_H_over_M=%.9e "
           "reallocatable_share_S=%.9e pass_score=%.9e verdict=%s "
           "invalid_reasons=%s\n",
           step, valid, positive, m, h, spread, s, pass_score, verdict, reason);

    for (int k = 0; k < V12_N_LAYERS; k++) {
        windows[k].count = 0;
    }
}


#define SAOBM_STEP(sa, g, p, lr) do { \
    int s_val = (int)(sa) + ((g) > 0.0f ? 1 : ((g) < 0.0f ? -1 : 0)); \
    if (s_val > 127) s_val = 127; \
    if (s_val < -127) s_val = -127; \
    (sa) = (int8_t)s_val; \
    (p) -= (lr) * ((float)(sa) / 127.0f); \
} while (0)

/* ADAM_STEP(m, v, g, p, lr, inv_bc1, inv_bc2)
 * Applies one Adam step in-place.
 * m, v: moment accumulators (float, updated in-place)
 * g:    gradient (float, read-only)
 * p:    parameter (float, updated in-place)
 * lr:   effective learning rate for this parameter
 * inv_bc1, inv_bc2: precomputed 1/(1-beta1^t) and 1/(1-beta2^t) for bias correction
 */
#define ADAM_STEP(m, v, g, p, lr, inv_bc1, inv_bc2) do {          \
    (m) = ADAM_BETA1 * (m) + (1.0f - ADAM_BETA1) * (g);           \
    (v) = ADAM_BETA2 * (v) + (1.0f - ADAM_BETA2) * (g) * (g);     \
    float m_hat = (m) * (inv_bc1);                                 \
    float v_hat = (v) * (inv_bc2);                                 \
    (p) -= (lr) * m_hat / (sqrtf(v_hat) + ADAM_EPS);               \
} while (0)

/* FACTORED_ADAM_PARAM_UPDATE(m_row, m_col, v_row, v_col, mean_v,
 *                             p, lr, i, j, inv_bc1, inv_bc2)
 *
 * Applies factored Adam parameter update at element [i,j] of a 2D matrix.
 * m_hat[i,j] = (m_row[i] + m_col[j]) * inv_bc1  (additive factorization)
 * v_hat[i,j] = (v_row[i] * v_col[j] / mean_v) * inv_bc2  (multiplicative)
 * p[i,j] -= lr * m_hat / (sqrt(v_hat) + eps)
 */
#define FACTORED_ADAM_PARAM_UPDATE(m_row, m_col, v_row, v_col, mean_v,  \
                                    p, lr, i, j, inv_bc1, inv_bc2)       \
do {                                                                      \
    float m_hat = ((m_row)[i] + (m_col)[j]) * (inv_bc1);                 \
    float v_approx = (v_row)[i] * (v_col)[j] / ((mean_v) + 1e-30f);     \
    float v_hat = v_approx * (inv_bc2);                                   \
    (p) -= (lr) * m_hat / (sqrtf(v_hat) + ADAM_EPS);                     \
} while (0)

#if V12_CDSVQ_ENABLED
#define CDSVQ_KL_REG(s_f, lr) do { \
    float abs_s = fabsf(s_f); \
    float m = (abs_s > V12_CDSVQ_DELTA) ? abs_s : V12_CDSVQ_DELTA; \
    float exp_s = expf((s_f) - m); \
    float exp_neg_s = expf(-(s_f) - m); \
    float exp_delta = expf(V12_CDSVQ_DELTA - m); \
    float denom = exp_s + exp_neg_s + exp_delta; \
    float p_pos = exp_s / denom; \
    float p_neg = exp_neg_s / denom; \
    (s_f) -= (lr) * V12_CDSVQ_KL_COEFF * (p_pos - p_neg); \
} while(0)
#else
#define CDSVQ_KL_REG(s_f, lr) do { } while(0)
#endif

/* =========================================================================
 * Function 1: mvue_sample_weights
 *
 * Resample sampled_w_in and sampled_w_out from shadow logits using MVUE
 * distribution. Called every V12_REPACK_INTERVAL steps.
 * ========================================================================= */

void mvue_sample_weights(SSMLayer *layer, uint64_t *rng_state) {
    int n_in  = V12_D_MODEL * V12_D_STATE;
    int n_out = V12_D_STATE * V12_D_MODEL;

    /* R356 F2 S5: route every shadow read through SHADOW_DECODE so that
     * the sampler sees the decoded FP32 value regardless of whether the
     * shadow buffer is float (flag OFF, no-op cast) or int16_t (flag ON,
     * grid-decoded). The MVUE math is unchanged in both builds. */
    for (int i = 0; i < n_in; i++) {
        float s = SHADOW_DECODE(layer->shadow_w_in[i]);
        float u = uniform01(rng_state);
        layer->sampled_w_in[i] = sample_ternary_from_logit(s, u);
    }

    for (int i = 0; i < n_out; i++) {
        float s = SHADOW_DECODE(layer->shadow_w_out[i]);
        float u = uniform01(rng_state);
        layer->sampled_w_out[i] = sample_ternary_from_logit(s, u);
    }

    if (layer->shadow_C_bilinear != NULL && layer->sampled_C_bilinear != NULL) {
        for (int i = 0; i < V12_D_STATE; i++) {
            float u = uniform01(rng_state);
            layer->sampled_C_bilinear[i] = sample_ternary_from_logit_mvue(
                SHADOW_DECODE(layer->shadow_C_bilinear[i]), u);
        }
    }
}


/* =========================================================================
 * optimizer_rebuild_alive_idx
 *
 * Scans shadow_w_in and shadow_w_out to build flat alive-index lists.
 * Must be called every V12_REPACK_INTERVAL steps, after update_crystallization.
 *
 * Allocates or reallocates alive_idx_in / alive_idx_out if needed.
 * On first call (alive_idx_in == NULL): allocates n_in ints (worst case = all alive).
 * On subsequent calls: reuses the allocation (size only shrinks or stays same).
 * ========================================================================= */

void optimizer_rebuild_alive_idx(SSMLayer *layer) {
    int n_in  = V12_D_MODEL * V12_D_STATE;
    int n_out = V12_D_STATE * V12_D_MODEL;

    if (layer->alive_idx_in == NULL) {
        layer->alive_idx_in  = (int *)malloc(n_in  * sizeof(int));
        layer->alive_idx_out = (int *)malloc(n_out * sizeof(int));
        if (!layer->alive_idx_in || !layer->alive_idx_out) {
            fprintf(stderr, "[OPT] alive_idx alloc failed\n");
            return;
        }
    }

    /* R356 F2 S6: crystallization gate reads go through SHADOW_DECODE so
     * the |s| > tau test sees the decoded FP32 magnitude. Under flag OFF
     * the macro is a no-op cast and the comparison is bitwise identical
     * to the previous code. Under flag ON the int16 saturation grid is
     * monotone in the encoded code, so the comparison is semantically
     * equivalent within one grid step (~0.07). */
    int cnt = 0;
    for (int i = 0; i < n_in; i++) {
        if (fabsf(SHADOW_DECODE(layer->shadow_w_in[i])) < V12_CRYSTALLIZATION_TAU)
            layer->alive_idx_in[cnt++] = i;
    }
    layer->n_alive_in = cnt;

    cnt = 0;
    for (int i = 0; i < n_out; i++) {
        if (fabsf(SHADOW_DECODE(layer->shadow_w_out[i])) < V12_CRYSTALLIZATION_TAU)
            layer->alive_idx_out[cnt++] = i;
    }
    layer->n_alive_out = cnt;

    printf("[ALIVE-IDX] layer rebuilt: n_alive_in=%d/%d (%.1f%%) n_alive_out=%d/%d (%.1f%%)\n",
           layer->n_alive_in,  n_in,  100.f * layer->n_alive_in  / (float)n_in,
           layer->n_alive_out, n_out, 100.f * layer->n_alive_out / (float)n_out);
}


static float stochastic_round_moment(float x, uint64_t *rng_state) {
    float f = floorf(x);
    float r = x - f;
    float u = uniform01(rng_state);
    return (u < r) ? (f + 1.0f) : f;
}

static void quantize_moment_block(float *m, float *v, int size, uint64_t *rng_state) {
    if (m == NULL || v == NULL) return;
    const int B_s = 64;
    for (int i = 0; i < size; i += B_s) {
        int block_len = (i + B_s <= size) ? B_s : (size - i);
        if (block_len <= 0) break;
        
        float max_m = 0.0f;
        float max_v = 0.0f;
        for (int j = 0; j < block_len; j++) {
            float abs_m = fabsf(m[i + j]);
            float abs_v = fabsf(v[i + j]);
            if (abs_m > max_m) max_m = abs_m;
            if (abs_v > max_v) max_v = abs_v;
        }
        
        float s_m = max_m / 127.0f;
        float s_v = max_v / 127.0f;
        
        for (int j = 0; j < block_len; j++) {
            if (s_m > 0.0f) {
                float val = m[i + j] / s_m;
                float q = stochastic_round_moment(val, rng_state);
                if (q < -128.0f) q = -128.0f;
                if (q > 127.0f) q = 127.0f;
                m[i + j] = q * s_m;
            } else {
                m[i + j] = 0.0f;
            }
            
            if (s_v > 0.0f) {
                float val = v[i + j] / s_v;
                float q = stochastic_round_moment(val, rng_state);
                if (q < -128.0f) q = -128.0f;
                if (q > 127.0f) q = 127.0f;
                v[i + j] = q * s_v;
            } else {
                v[i + j] = 0.0f;
            }
        }
    }
}


/* =========================================================================
 * optimizer_reset_moments
 *
 * Reset first-moment (m) accumulators to zero when transitioning between
 * training phases. Called when switching from FP -> BCD or BCD -> joint.
 * Prevents accumulated momentum direction from exploding when the active
 * parameter set changes.  Second-moment (v) accumulators are KEPT so that
 * Adam's curvature estimate survives the phase boundary and does not waste
 * the first ~100 steps rediscovering per-parameter scale.
 * ========================================================================= */

void optimizer_reset_moments(Model *model) {
    for (int k = 0; k < V12_N_LAYERS; k++) {
        SSMLayer *layer = &model->layers[k];
        if (layer->dw_in == NULL) continue;  /* offloaded layer */

        if (layer->m_w_in_row) memset(layer->m_w_in_row, 0, V12_D_MODEL * sizeof(float));
        if (layer->m_w_in_col) memset(layer->m_w_in_col, 0, V12_D_STATE * sizeof(float));
        /* v_w_in_row, v_w_in_col — inherited (curvature estimate) */
        if (layer->m_w_out_row) memset(layer->m_w_out_row, 0, V12_D_STATE * sizeof(float));
        if (layer->m_w_out_col) memset(layer->m_w_out_col, 0, V12_D_MODEL * sizeof(float));
        /* v_w_out_row, v_w_out_col — inherited (curvature estimate) */
        if (layer->m_A) memset(layer->m_A, 0, V12_D_STATE * sizeof(float));
        /* v_A — inherited */
        if (layer->m_B) memset(layer->m_B, 0, V12_D_STATE * sizeof(float));
        /* v_B — inherited */
        if (layer->m_C != NULL) memset(layer->m_C,     0, V12_D_STATE * V12_D_STATE * sizeof(float));
        /* v_C — inherited */
        if (layer->m_conv) memset(layer->m_conv, 0, V12_CONV_KERNEL * V12_D_STATE * sizeof(float));
        if (layer->m_u) memset(layer->m_u, 0, V12_D_STATE * sizeof(float));
        if (layer->m_v) memset(layer->m_v, 0, V12_D_STATE * sizeof(float));
        /* v_conv — inherited */
    }

    /* Reset lm_head and w_pred first moments; keep second moments (curvature)
     * Note: m_lm_head_row has K_HOT elements (factored for hot tokens only),
     * shadow/sampled/dlm_head have vocab_size × d_model but only hot tokens updated */
#if !V12_TOLR_ENABLED
    memset(model->m_lm_head_row, 0, V12_K_HOT     * sizeof(float));
    memset(model->m_lm_head_col, 0, V12_D_MODEL   * sizeof(float));
    /* v_lm_head_row, v_lm_head_col — inherited */
#endif
    memset(model->m_w_pred_row,  0, V12_D_MODEL    * sizeof(float));
    memset(model->m_w_pred_col,  0, V12_D_MODEL    * sizeof(float));
    /* v_w_pred_row, v_w_pred_col — inherited */
    
    /* Reset norm layer first moments; keep second moments (curvature) */
    memset(model->m_norm_w,  0, V12_D_MODEL * sizeof(float));
    /* v_norm_w — inherited */
    memset(model->m_norm_b,  0, V12_D_MODEL * sizeof(float));
    /* v_norm_b — inherited */
    
    printf("[OPT] Phase transition: m (first moment) reset, v (second moment) inherited\n");
}


/* =========================================================================
 * Function 2: optimizer_step
 *
 * Single gradient step across all parameters with crystallization
 * and MVUE updates.
 * ========================================================================= */

void optimizer_step(Model *model, TrainState *state, float lr) {
    int step = (int)state->step;
    static R398PafeBlockWindow r398_pafe_windows[V12_MAX_LAYERS];
    int r398_pafe_active = r398_pafe_probe_active();

#if V12_ZMMA_ENABLED
    static float zmma_global_var = 1.0f;
    static int zmma_initialized = 0;
    {
        double sum_sq = 0.0;
        uint64_t total_count = 0;
        for (int k = 0; k < V12_N_LAYERS; k++) {
            SSMLayer *L = &model->layers[k];
            if (L->dw_in) {
                uint64_t n_in = (uint64_t)model->d_model * V12_D_STATE;
                for (uint64_t i = 0; i < n_in; i++) {
                    float g = L->dw_in[i];
                    sum_sq += (double)g * g;
                    total_count++;
                }
            }
            if (L->dw_out) {
                uint64_t n_out = (uint64_t)V12_D_STATE * model->d_model;
                for (uint64_t i = 0; i < n_out; i++) {
                    float g = L->dw_out[i];
                    sum_sq += (double)g * g;
                    total_count++;
                }
            }
        }
        if (total_count > 0) {
            float step_var = (float)(sum_sq / (double)total_count);
            if (step_var > 1e-12f) {
                if (!zmma_initialized) {
                    zmma_global_var = step_var;
                    zmma_initialized = 1;
                } else {
                    zmma_global_var = 0.99f * zmma_global_var + 0.01f * step_var;
                }
            }
        }
    }
    float zmma_var_floor = fmaxf(zmma_global_var + 1e-8f, 1e-2f);
#endif

    /* BCD refinement LR override: use small fixed LR instead of schedule */
    if (state->bcd_active) lr = 5e-6f;

    /* Adam bias correction coefficients for this step */
    int adam_t = (int)state->step + 1;
    float bc1 = 1.0f - powf(ADAM_BETA1, (float)adam_t);
    float bc2 = 1.0f - powf(ADAM_BETA2, (float)adam_t);
    float inv_bc1 = 1.0f / bc1;
    float inv_bc2 = 1.0f / bc2;

    if (0) {
        /* R356 F2: read through SHADOW_DECODE so this debug block compiles
         * cleanly under both flag values. */
        float s = SHADOW_DECODE(model->layers[0].shadow_w_in[0]);
        float w = model->layers[0].sampled_w_in[0];
        float dw = model->layers[0].dw_in[0];
        printf("[DEBUG-OPT] s=%f w=%f dw=%e (w-s)=%f update=%e\n",
               s, w, dw, w-s, dw * (w-s));
        printf("[FACTORED-ADAM] m_w_in_row[0]=%.4e m_w_in_col[0]=%.4e "
               "v_w_in_row[0]=%.4e v_w_in_col[0]=%.4e\n",
               model->layers[0].m_w_in_row[0],
               model->layers[0].m_w_in_col[0],
               model->layers[0].v_w_in_row[0],
               model->layers[0].v_w_in_col[0]);
    }

    /* Fisher information LR scaling accumulators.
     * Static: persist across calls, reset only on program restart.
     * grad_norm_acc[k] accumulates sum of squared gradient magnitudes.
     * fisher_acc_count counts steps since last lr_per_layer update.      */
    static float    grad_norm_acc[V12_MAX_LAYERS] = {0.0f};
    static uint32_t fisher_acc_count = 0;
    const uint32_t gpu_opt_wio = model->gpu_opt_flags & (GPU_OPT_W_IN | GPU_OPT_W_OUT);

    /* Global gradient norm clipping: prevent gradient explosion */
    {
        float grad_sq_sum = 0.0f;
        for (int k = 0; k < V12_N_LAYERS; k++) {
            if (model->layers[k].dw_in == NULL) continue;  /* offloaded layer — skip */
            if (!(model->gpu_opt_flags & GPU_OPT_W_IN)) {
                for (int i = 0; i < V12_D_MODEL * V12_D_STATE; i++)
                    grad_sq_sum += model->layers[k].dw_in[i] * model->layers[k].dw_in[i];
            }
            if (!(model->gpu_opt_flags & GPU_OPT_W_OUT)) {
                for (int i = 0; i < V12_D_STATE * V12_D_MODEL; i++)
                    grad_sq_sum += model->layers[k].dw_out[i] * model->layers[k].dw_out[i];
            }
            for (int i = 0; i < V12_D_STATE; i++)
                grad_sq_sum += model->layers[k].d_sigma[i] * model->layers[k].d_sigma[i];
            for (int i = 0; i < V12_D_STATE; i++)
                grad_sq_sum += model->layers[k].dA[i] * model->layers[k].dA[i];
            for (int i = 0; i < V12_D_STATE; i++)
                grad_sq_sum += model->layers[k].dB[i] * model->layers[k].dB[i];
            for (int i = 0; i < V12_D_STATE * V12_D_STATE; i++)
                grad_sq_sum += model->layers[k].dW_A[i] * model->layers[k].dW_A[i];
            for (int i = 0; i < V12_D_STATE; i++)
                grad_sq_sum += model->layers[k].db_A[i] * model->layers[k].db_A[i];
            if (model->layers[k].dC_bilinear != NULL)
                for (int i = 0; i < V12_D_STATE; i++)
                    grad_sq_sum += model->layers[k].dC_bilinear[i] * model->layers[k].dC_bilinear[i];
            if (model->layers[k].du != NULL)
                for (int i = 0; i < V12_D_STATE; i++)
                    grad_sq_sum += model->layers[k].du[i] * model->layers[k].du[i];
            if (model->layers[k].dv != NULL)
                for (int i = 0; i < V12_D_STATE; i++)
                    grad_sq_sum += model->layers[k].dv[i] * model->layers[k].dv[i];
        }
        if (!(model->gpu_opt_flags & GPU_OPT_LM_HEAD) && model->dlm_head_gq_row) {
            float dlm_sq = 0.0f;
            #pragma omp parallel for reduction(+:dlm_sq) schedule(static)
            for (int i = 0; i < V12_K_HOT; i++)
                dlm_sq += model->dlm_head_gq_row[i];
            grad_sq_sum += dlm_sq;
        }

        if (!(model->gpu_opt_flags & GPU_OPT_EMBED_E) && model->d_embed_E) {
            float demb_sq = 0.0f;
            #pragma omp parallel for reduction(+:demb_sq) schedule(static)
            for (int i = 0; i < V12_VOCAB_SIZE * V12_D_EMBED; i++)
                demb_sq += model->d_embed_E[i] * model->d_embed_E[i];
            grad_sq_sum += demb_sq;
        }
        if (!(model->gpu_opt_flags & GPU_OPT_EMBED_P) && model->d_embed_P) {
            for (int i = 0; i < V12_D_EMBED * V12_D_MODEL; i++)
                grad_sq_sum += model->d_embed_P[i] * model->d_embed_P[i];
        }

        float grad_norm = sqrtf(grad_sq_sum + 1e-8f);
        float max_norm = 1.0f;
        if (grad_norm > max_norm) {
            float scale = max_norm / grad_norm;
            for (int k = 0; k < V12_N_LAYERS; k++) {
                if (model->layers[k].dw_in == NULL) continue;  /* offloaded layer — skip */
                if (!(model->gpu_opt_flags & GPU_OPT_W_IN)) {
                    for (int i = 0; i < V12_D_MODEL * V12_D_STATE; i++)
                        model->layers[k].dw_in[i] *= scale;
                }
                if (!(model->gpu_opt_flags & GPU_OPT_W_OUT)) {
                    for (int i = 0; i < V12_D_STATE * V12_D_MODEL; i++)
                        model->layers[k].dw_out[i] *= scale;
                }
                for (int i = 0; i < V12_D_STATE; i++)
                    model->layers[k].d_sigma[i] *= scale;
                for (int i = 0; i < V12_D_STATE; i++)
                    model->layers[k].dA[i] *= scale;
                for (int i = 0; i < V12_D_STATE; i++)
                    model->layers[k].dB[i] *= scale;
                if (model->layers[k].dC != NULL) {
                    for (int i = 0; i < V12_D_STATE * V12_D_STATE; i++)
                        model->layers[k].dC[i] *= scale;
                }
                int n_conv = V12_CONV_KERNEL * V12_D_STATE;
                for (int i = 0; i < n_conv; i++)
                    model->layers[k].d_conv_w[i] *= scale;
                for (int i = 0; i < V12_D_STATE * V12_D_STATE; i++)
                    model->layers[k].dW_A[i] *= scale;
                for (int i = 0; i < V12_D_STATE; i++)
                    model->layers[k].db_A[i] *= scale;
                if (model->layers[k].dC_bilinear != NULL)
                    for (int i = 0; i < V12_D_STATE; i++)
                        model->layers[k].dC_bilinear[i] *= scale;
                if (model->layers[k].du != NULL)
                    for (int i = 0; i < V12_D_STATE; i++)
                        model->layers[k].du[i] *= scale;
                if (model->layers[k].dv != NULL)
                    for (int i = 0; i < V12_D_STATE; i++)
                        model->layers[k].dv[i] *= scale;
            }
            if (!(model->gpu_opt_flags & GPU_OPT_LM_HEAD) && model->dlm_head_g_row) {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < V12_K_HOT; i++) {
                    model->dlm_head_g_row[i] *= scale;
                    model->dlm_head_gq_row[i] *= scale * scale;
                }
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < V12_D_MODEL; i++) {
                    model->dlm_head_g_col[i] *= scale;
                    model->dlm_head_gq_col[i] *= scale * scale;
                }
            }
            if (!(model->gpu_opt_flags & GPU_OPT_EMBED_E) && model->d_embed_E) {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < V12_VOCAB_SIZE * V12_D_EMBED; i++)
                    model->d_embed_E[i] *= scale;
            }
            if (!(model->gpu_opt_flags & GPU_OPT_EMBED_P) && model->d_embed_P) {
                for (int i = 0; i < V12_D_EMBED * V12_D_MODEL; i++)
                    model->d_embed_P[i] *= scale;
            }
            if (!(model->gpu_opt_flags & GPU_OPT_NORM)
                && model->d_norm_w && model->d_norm_b) {
                for (int i = 0; i < V12_D_MODEL; i++) {
                    model->d_norm_w[i] *= scale;
                    model->d_norm_b[i] *= scale;
                }
            }
            if (!(model->gpu_opt_flags & GPU_OPT_W_PRED) && model->dw_pred) {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < V12_D_MODEL * V12_D_MODEL; i++)
                    model->dw_pred[i] *= scale;
            }
        }
    }

    /* R356 F3 measurement hook — read-only, gated by V12_F3_MEASURE compile
     * flag + env var; compiled out in default builds (see header). */
    optimizer_f3_measure_step(model, step);
    if (step == 1100) {
        static int f3_summary_fired = 0;
        if (!f3_summary_fired) {
            optimizer_f3_measure_summary();
            f3_summary_fired = 1;
        }
    }

    for (int k = 0; k < V12_N_LAYERS; k++) {
        SSMLayer *layer = &model->layers[k];
        if (layer->dw_in == NULL) continue;  /* offloaded layer — skip */
        float lr_eff = lr * state->lr_per_layer[k];

        int n_in = V12_D_MODEL * V12_D_STATE;
        int n_out = V12_D_STATE * V12_D_MODEL;
        int r398_pafe_active_count = 0;
        int r398_pafe_excluded_count = 0;
        double r398_pafe_pre_dot = 0.0;
        double r398_pafe_t0 = 0.0;
        if (r398_pafe_active) {
            r398_pafe_pre_dot = r398_pafe_layer_dot(
                model, layer, k, &r398_pafe_active_count, &r398_pafe_excluded_count);
            r398_pafe_t0 = omp_get_wtime();
        }

        /* ---- w_in factored Adam ----------------------------------------- */
        if (!(model->gpu_opt_flags & GPU_OPT_W_IN) || layer->moments_offloaded) {
        {
            int rows = V12_D_MODEL;
            int cols = V12_D_STATE;

            /* Step 1: Accumulate row and col gradient means.
             * Only over non-crystallized, sign-consistent entries. */
            float g_row[V12_D_MODEL] = {0.0f};  /* sum of g over cols per row  */
            float g_col[V12_D_STATE] = {0.0f};  /* sum of g over rows per col  */
            float gq_row[V12_D_MODEL] = {0.0f}; /* sum of g^2 over cols        */
            float gq_col[V12_D_STATE] = {0.0f}; /* sum of g^2 over rows        */
            int   cnt_row[V12_D_MODEL] = {0};
            int   cnt_col[V12_D_STATE] = {0};

            if (layer->alive_idx_in != NULL) {
                /* FAST PATH — iterate only over alive indices */
                for (int ai = 0; ai < layer->n_alive_in; ai++) {
                    int idx = layer->alive_idx_in[ai];
                    int r   = idx / cols;
                    int c   = idx % cols;

                    float dw_i = layer->dw_in[idx];
                    if (dw_i == 0.0f) continue;

                    float eff_dw = (layer->sampled_w_in[idx] == 0.0f) ? (dw_i * V12_BI_STE_FACTOR) : dw_i;
                    float g_mvue = eff_dw;  /* R353: STE (1.57x more efficient than REINFORCE) */

                    /* PHGD Gated Stability Descent (Aspect A03) */
                    {
                        float s_val = SHADOW_DECODE(layer->shadow_w_in[idx]);
                        float dist = fabsf(s_val - 0.5f) < fabsf(s_val + 0.5f) ? fabsf(s_val - 0.5f) : fabsf(s_val + 0.5f);
                        float qbs = 1.0f - expf(-(dist * dist) / (2.0f * V12_PHGD_SIGMA0 * V12_PHGD_SIGMA0));
                        float gate = 1.0f / (1.0f + expf(-((fabsf(g_mvue) * qbs - V12_PHGD_TAU) / V12_PHGD_EPS)));
                        g_mvue *= gate;
                    }

                    layer->grad_accum_in[idx] = V12_EMA_SPRT_GAMMA * layer->grad_accum_in[idx] + g_mvue;
                    if (fabsf(layer->grad_accum_in[idx]) <= V12_EMA_SPRT_THRESHOLD) continue;

                    g_row[r]  += g_mvue;
                    g_col[c]  += g_mvue;
                    gq_row[r] += g_mvue * g_mvue;
                    gq_col[c] += g_mvue * g_mvue;
                    cnt_row[r]++;
                    cnt_col[c]++;
                }
            } else {
                /* SLOW PATH — full scan (first 500 steps only) */
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        int idx = r * cols + c;
                        /* R356 F2 S6: decoded magnitude gate (no-op under flag OFF). */
                        if (fabsf(SHADOW_DECODE(layer->shadow_w_in[idx])) > V12_CRYSTALLIZATION_TAU) continue;

                        float dw_i = layer->dw_in[idx];
                        if (dw_i == 0.0f) continue;

                        float eff_dw = (layer->sampled_w_in[idx] == 0.0f) ? (dw_i * V12_BI_STE_FACTOR) : dw_i;
                        float g_mvue = eff_dw;  /* R353: STE (1.57x more efficient than REINFORCE) */

                        /* PHGD Gated Stability Descent (Aspect A03) */
                        {
                            float s_val = SHADOW_DECODE(layer->shadow_w_in[idx]);
                            float dist = fabsf(s_val - 0.5f) < fabsf(s_val + 0.5f) ? fabsf(s_val - 0.5f) : fabsf(s_val + 0.5f);
                            float qbs = 1.0f - expf(-(dist * dist) / (2.0f * V12_PHGD_SIGMA0 * V12_PHGD_SIGMA0));
                            float gate = 1.0f / (1.0f + expf(-((fabsf(g_mvue) * qbs - V12_PHGD_TAU) / V12_PHGD_EPS)));
                            g_mvue *= gate;
                        }

                        layer->grad_accum_in[idx] = V12_EMA_SPRT_GAMMA * layer->grad_accum_in[idx] + g_mvue;
                        if (fabsf(layer->grad_accum_in[idx]) <= V12_EMA_SPRT_THRESHOLD) continue;

                        g_row[r]  += g_mvue;
                        g_col[c]  += g_mvue;
                        gq_row[r] += g_mvue * g_mvue;
                        gq_col[c] += g_mvue * g_mvue;
                        cnt_row[r]++;
                        cnt_col[c]++;
                    }
                }
            }

#if !V12_ZMMA_ENABLED
            /* Step 2: Update row/col moment vectors using means */
            for (int r = 0; r < rows; r++) {
                float gm  = (cnt_row[r] > 0) ? g_row[r]  / (float)cnt_row[r]  : 0.0f;
                float gqm = (cnt_row[r] > 0) ? gq_row[r] / (float)cnt_row[r]  : 0.0f;
                layer->m_w_in_row[r] = ADAM_BETA1 * layer->m_w_in_row[r] + (1.0f - ADAM_BETA1) * gm;
                layer->v_w_in_row[r] = ADAM_BETA2 * layer->v_w_in_row[r] + (1.0f - ADAM_BETA2) * gqm;
            }
            for (int c = 0; c < cols; c++) {
                float gm  = (cnt_col[c] > 0) ? g_col[c]  / (float)cnt_col[c]  : 0.0f;
                float gqm = (cnt_col[c] > 0) ? gq_col[c] / (float)cnt_col[c]  : 0.0f;
                layer->m_w_in_col[c] = ADAM_BETA1 * layer->m_w_in_col[c] + (1.0f - ADAM_BETA1) * gm;
                layer->v_w_in_col[c] = ADAM_BETA2 * layer->v_w_in_col[c] + (1.0f - ADAM_BETA2) * gqm;
            }

            /* Step 3: Compute mean_v scalar for denominator normalization */
            float sum_vr = 0.0f, sum_vc = 0.0f;
            for (int r = 0; r < rows; r++) sum_vr += layer->v_w_in_row[r];
            for (int c = 0; c < cols; c++) sum_vc += layer->v_w_in_col[c];
            float mean_v = (sum_vr / (float)rows) * (sum_vc / (float)cols);
#else
            float mean_v = 1.0f; /* unused under ZMMA */
#endif

            /* Step 4: Apply parameter update at each active element.
             *
             * R356 F2 S7: under flag ON the shadow buffer is int16-encoded.
             * Decode to a transient FP32 scalar `s_f`, run the Adam macro
             * on that scalar, then stochastically round back into the
             * int16 cell via shadow_rng/stochastic_round_shadow. The clip
             * to [-10, 10] is absorbed into stochastic_round_shadow.
             * The transient never escapes register scope. Under flag OFF
             * the #else branch is byte-identical to the previous build. */
            if (layer->alive_idx_in != NULL) {
                /* FAST PATH — iterate only over alive indices */
                #pragma omp parallel for schedule(static)
                for (int ai = 0; ai < layer->n_alive_in; ai++) {
                    int idx = layer->alive_idx_in[ai];
                    int r   = idx / cols;
                    int c   = idx % cols;

                    if (fabsf(layer->grad_accum_in[idx]) <= V12_EMA_SPRT_THRESHOLD) continue;

#if V12_ZMMA_ENABLED
                    float lr_zmma = lr_eff / sqrtf(zmma_var_floor);
                    float ema_g = layer->grad_accum_in[idx];
                    float step_dir = (ema_g > 0.0f) ? 1.0f : -1.0f;
#if V12_F2_INT8_SHADOWS
                    float s_f = shadow_decode(layer->shadow_w_in[idx]);
                    s_f -= lr_zmma * step_dir;
                    CDSVQ_KL_REG(s_f, lr_eff);
                    uint32_t rnd = shadow_rng(k, V12_SHADOW_MAT_W_IN, idx,
                                              (uint64_t)state->step);
                    layer->shadow_w_in[idx] = stochastic_round_shadow(s_f, rnd);
#else
                    float s_f = layer->shadow_w_in[idx] - lr_zmma * step_dir;
                    CDSVQ_KL_REG(s_f, lr_eff);
                    layer->shadow_w_in[idx] = fmaxf(-10.0f, fminf(10.0f, s_f));
#endif
#else
#if V12_AIST_ENABLED
                    {
                        float s_now = SHADOW_DECODE(layer->shadow_w_in[idx]);
                        float m_hat_ai = (layer->m_w_in_row[r] + layer->m_w_in_col[c]) * inv_bc1;
                        float v_approx_ai = layer->v_w_in_row[r] * layer->v_w_in_col[c] / (mean_v + 1e-30f);
                        float v_hat_ai = v_approx_ai * inv_bc2;
                        float g_ai = lr_eff * m_hat_ai / (sqrtf(v_hat_ai) + ADAM_EPS);
                        if (aist_should_skip(s_now, g_ai, v12_aist_epsilon)) {
                            layer->grad_accum_in[idx] = 0.0f;
                            continue;
                        }
                    }
#endif

#if V12_F2_INT8_SHADOWS
                    float s_f = shadow_decode(layer->shadow_w_in[idx]);
                    FACTORED_ADAM_PARAM_UPDATE(
                        layer->m_w_in_row, layer->m_w_in_col,
                        layer->v_w_in_row, layer->v_w_in_col,
                        mean_v,
                        s_f,
                        lr_eff, r, c, inv_bc1, inv_bc2);
                    uint32_t rnd = shadow_rng(k, V12_SHADOW_MAT_W_IN, idx,
                                              (uint64_t)state->step);
                    layer->shadow_w_in[idx] = stochastic_round_shadow(s_f, rnd);
#else
                    FACTORED_ADAM_PARAM_UPDATE(
                        layer->m_w_in_row, layer->m_w_in_col,
                        layer->v_w_in_row, layer->v_w_in_col,
                        mean_v,
                        layer->shadow_w_in[idx],
                        lr_eff, r, c, inv_bc1, inv_bc2);
                    layer->shadow_w_in[idx] = fmaxf(-10.0f, fminf(10.0f, layer->shadow_w_in[idx]));
#endif
#endif
                    layer->grad_accum_in[idx] = 0.0f;
                }
            } else {
                /* SLOW PATH — full scan (first 500 steps only) */
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        int idx = r * cols + c;
                        /* R356 F2 S6: decoded magnitude gate (no-op under flag OFF). */
                        if (fabsf(SHADOW_DECODE(layer->shadow_w_in[idx])) > V12_CRYSTALLIZATION_TAU) continue;
                        if (fabsf(layer->grad_accum_in[idx]) <= V12_EMA_SPRT_THRESHOLD) continue;

#if V12_ZMMA_ENABLED
                        float lr_zmma = lr_eff / sqrtf(zmma_var_floor);
                        float ema_g = layer->grad_accum_in[idx];
                        float step_dir = (ema_g > 0.0f) ? 1.0f : -1.0f;
#if V12_F2_INT8_SHADOWS
                        float s_f = shadow_decode(layer->shadow_w_in[idx]);
                        s_f -= lr_zmma * step_dir;
                        CDSVQ_KL_REG(s_f, lr_eff);
                        uint32_t rnd = shadow_rng(k, V12_SHADOW_MAT_W_IN, idx,
                                                  (uint64_t)state->step);
                        layer->shadow_w_in[idx] = stochastic_round_shadow(s_f, rnd);
#else
                        float s_f = layer->shadow_w_in[idx] - lr_zmma * step_dir;
                        CDSVQ_KL_REG(s_f, lr_eff);
                        layer->shadow_w_in[idx] = fmaxf(-10.0f, fminf(10.0f, s_f));
#endif
#else
#if V12_AIST_ENABLED
                        {
                            float s_now = SHADOW_DECODE(layer->shadow_w_in[idx]);
                            float m_hat_ai = (layer->m_w_in_row[r] + layer->m_w_in_col[c]) * inv_bc1;
                            float v_approx_ai = layer->v_w_in_row[r] * layer->v_w_in_col[c] / (mean_v + 1e-30f);
                            float v_hat_ai = v_approx_ai * inv_bc2;
                            float g_ai = lr_eff * m_hat_ai / (sqrtf(v_hat_ai) + ADAM_EPS);
                            if (aist_should_skip(s_now, g_ai, v12_aist_epsilon)) {
                                layer->grad_accum_in[idx] = 0.0f;
                                continue;
                            }
                        }
#endif

#if V12_F2_INT8_SHADOWS
                        float s_f = shadow_decode(layer->shadow_w_in[idx]);
                        FACTORED_ADAM_PARAM_UPDATE(
                            layer->m_w_in_row, layer->m_w_in_col,
                            layer->v_w_in_row, layer->v_w_in_col,
                            mean_v,
                            s_f,
                            lr_eff, r, c, inv_bc1, inv_bc2);
                        uint32_t rnd = shadow_rng(k, V12_SHADOW_MAT_W_IN, idx,
                                                  (uint64_t)state->step);
                        layer->shadow_w_in[idx] = stochastic_round_shadow(s_f, rnd);
#else
                        FACTORED_ADAM_PARAM_UPDATE(
                            layer->m_w_in_row, layer->m_w_in_col,
                            layer->v_w_in_row, layer->v_w_in_col,
                            mean_v,
                            layer->shadow_w_in[idx],
                            lr_eff, r, c, inv_bc1, inv_bc2);
                        layer->shadow_w_in[idx] = fmaxf(-10.0f, fminf(10.0f, layer->shadow_w_in[idx]));
#endif
#endif
                        layer->grad_accum_in[idx] = 0.0f;
                    }
                }
            }
        }  /* w_in block */
        }  /* GPU_OPT_W_IN guard */

        /* ---- w_out factored Adam ----------------------------------------- */
        if (!(model->gpu_opt_flags & GPU_OPT_W_OUT) || layer->moments_offloaded) {
        {
            int rows = V12_D_STATE;
            int cols = V12_D_MODEL;

            /* Step 1: Accumulate row and col gradient means.
             * Only over non-crystallized, sign-consistent entries. */
            float g_row[V12_D_STATE] = {0.0f};  /* sum of g over cols per row  */
            float g_col[V12_D_MODEL] = {0.0f};  /* sum of g over rows per col  */
            float gq_row[V12_D_STATE] = {0.0f}; /* sum of g^2 over cols        */
            float gq_col[V12_D_MODEL] = {0.0f}; /* sum of g^2 over rows        */
            int   cnt_row[V12_D_STATE] = {0};
            int   cnt_col[V12_D_MODEL] = {0};

            if (layer->alive_idx_out != NULL) {
                /* FAST PATH — iterate only over alive indices */
                for (int ai = 0; ai < layer->n_alive_out; ai++) {
                    int idx = layer->alive_idx_out[ai];
                    int r   = idx / cols;
                    int c   = idx % cols;

                    float dw_i = layer->dw_out[idx];
                    if (dw_i == 0.0f) continue;

                    float eff_dw = (layer->sampled_w_out[idx] == 0.0f) ? (dw_i * V12_BI_STE_FACTOR) : dw_i;
                    float g_mvue = eff_dw;  /* R353: STE (1.57x more efficient than REINFORCE) */

                    /* PHGD Gated Stability Descent (Aspect A03) */
                    {
                        float s_val = SHADOW_DECODE(layer->shadow_w_out[idx]);
                        float dist = fabsf(s_val - 0.5f) < fabsf(s_val + 0.5f) ? fabsf(s_val - 0.5f) : fabsf(s_val + 0.5f);
                        float qbs = 1.0f - expf(-(dist * dist) / (2.0f * V12_PHGD_SIGMA0 * V12_PHGD_SIGMA0));
                        float gate = 1.0f / (1.0f + expf(-((fabsf(g_mvue) * qbs - V12_PHGD_TAU) / V12_PHGD_EPS)));
                        g_mvue *= gate;
                    }

                    layer->grad_accum_out[idx] = V12_EMA_SPRT_GAMMA * layer->grad_accum_out[idx] + g_mvue;
                    if (fabsf(layer->grad_accum_out[idx]) <= V12_EMA_SPRT_THRESHOLD) continue;

                    g_row[r]  += g_mvue;
                    g_col[c]  += g_mvue;
                    gq_row[r] += g_mvue * g_mvue;
                    gq_col[c] += g_mvue * g_mvue;
                    cnt_row[r]++;
                    cnt_col[c]++;
                }
            } else {
                /* SLOW PATH — full scan (first 500 steps only) */
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        int idx = r * cols + c;
                        /* R356 F2 S6: decoded magnitude gate (no-op under flag OFF). */
                        if (fabsf(SHADOW_DECODE(layer->shadow_w_out[idx])) > V12_CRYSTALLIZATION_TAU) continue;

                        float dw_i = layer->dw_out[idx];
                        if (dw_i == 0.0f) continue;

                        float eff_dw = (layer->sampled_w_out[idx] == 0.0f) ? (dw_i * V12_BI_STE_FACTOR) : dw_i;
                        float g_mvue = eff_dw;  /* R353: STE (1.57x more efficient than REINFORCE) */

                        /* PHGD Gated Stability Descent (Aspect A03) */
                        {
                            float s_val = SHADOW_DECODE(layer->shadow_w_out[idx]);
                            float dist = fabsf(s_val - 0.5f) < fabsf(s_val + 0.5f) ? fabsf(s_val - 0.5f) : fabsf(s_val + 0.5f);
                            float qbs = 1.0f - expf(-(dist * dist) / (2.0f * V12_PHGD_SIGMA0 * V12_PHGD_SIGMA0));
                            float gate = 1.0f / (1.0f + expf(-((fabsf(g_mvue) * qbs - V12_PHGD_TAU) / V12_PHGD_EPS)));
                            g_mvue *= gate;
                        }

                        layer->grad_accum_out[idx] = V12_EMA_SPRT_GAMMA * layer->grad_accum_out[idx] + g_mvue;
                        if (fabsf(layer->grad_accum_out[idx]) <= V12_EMA_SPRT_THRESHOLD) continue;

                        g_row[r]  += g_mvue;
                        g_col[c]  += g_mvue;
                        gq_row[r] += g_mvue * g_mvue;
                        gq_col[c] += g_mvue * g_mvue;
                        cnt_row[r]++;
                        cnt_col[c]++;
                    }
                }
            }

#if !V12_ZMMA_ENABLED
            /* Step 2: Update row/col moment vectors using means */
            for (int r = 0; r < rows; r++) {
                float gm  = (cnt_row[r] > 0) ? g_row[r]  / (float)cnt_row[r]  : 0.0f;
                float gqm = (cnt_row[r] > 0) ? gq_row[r] / (float)cnt_row[r]  : 0.0f;
                layer->m_w_out_row[r] = ADAM_BETA1 * layer->m_w_out_row[r] + (1.0f - ADAM_BETA1) * gm;
                layer->v_w_out_row[r] = ADAM_BETA2 * layer->v_w_out_row[r] + (1.0f - ADAM_BETA2) * gqm;
            }
            for (int c = 0; c < cols; c++) {
                float gm  = (cnt_col[c] > 0) ? g_col[c]  / (float)cnt_col[c]  : 0.0f;
                float gqm = (cnt_col[c] > 0) ? gq_col[c] / (float)cnt_col[c]  : 0.0f;
                layer->m_w_out_col[c] = ADAM_BETA1 * layer->m_w_out_col[c] + (1.0f - ADAM_BETA1) * gm;
                layer->v_w_out_col[c] = ADAM_BETA2 * layer->v_w_out_col[c] + (1.0f - ADAM_BETA2) * gqm;
            }

            /* Step 3: Compute mean_v scalar for denominator normalization */
            float sum_vr = 0.0f, sum_vc = 0.0f;
            for (int r = 0; r < rows; r++) sum_vr += layer->v_w_out_row[r];
            for (int c = 0; c < cols; c++) sum_vc += layer->v_w_out_col[c];
            float mean_v = (sum_vr / (float)rows) * (sum_vc / (float)cols);
#else
            float mean_v = 1.0f; /* unused under ZMMA */
#endif

            /* Step 4: Apply parameter update at each active element.
             * R356 F2 S8: same decode/encode pattern as S7 for w_in. */
            if (layer->alive_idx_out != NULL) {
                /* FAST PATH — iterate only over alive indices */
                #pragma omp parallel for schedule(static)
                for (int ai = 0; ai < layer->n_alive_out; ai++) {
                    int idx = layer->alive_idx_out[ai];
                    int r   = idx / cols;
                    int c   = idx % cols;

                    if (fabsf(layer->grad_accum_out[idx]) <= V12_EMA_SPRT_THRESHOLD) continue;

#if V12_ZMMA_ENABLED
                    float lr_zmma = lr_eff / sqrtf(zmma_var_floor);
                    float ema_g = layer->grad_accum_out[idx];
                    float step_dir = (ema_g > 0.0f) ? 1.0f : -1.0f;
#if V12_F2_INT8_SHADOWS
                    float s_f = shadow_decode(layer->shadow_w_out[idx]);
                    s_f -= lr_zmma * step_dir;
                    CDSVQ_KL_REG(s_f, lr_eff);
                    uint32_t rnd = shadow_rng(k, V12_SHADOW_MAT_W_OUT, idx,
                                              (uint64_t)state->step);
                    layer->shadow_w_out[idx] = stochastic_round_shadow(s_f, rnd);
#else
                    float s_f = layer->shadow_w_out[idx] - lr_zmma * step_dir;
                    CDSVQ_KL_REG(s_f, lr_eff);
                    layer->shadow_w_out[idx] = fmaxf(-10.0f, fminf(10.0f, s_f));
#endif
#else
#if V12_AIST_ENABLED
                    {
                        float s_now = SHADOW_DECODE(layer->shadow_w_out[idx]);
                        float m_hat_ai = (layer->m_w_out_row[r] + layer->m_w_out_col[c]) * inv_bc1;
                        float v_approx_ai = layer->v_w_out_row[r] * layer->v_w_out_col[c] / (mean_v + 1e-30f);
                        float v_hat_ai = v_approx_ai * inv_bc2;
                        float g_ai = lr_eff * m_hat_ai / (sqrtf(v_hat_ai) + ADAM_EPS);
                        if (aist_should_skip(s_now, g_ai, v12_aist_epsilon)) {
                            layer->grad_accum_out[idx] = 0.0f;
                            continue;
                        }
                    }
#endif

#if V12_F2_INT8_SHADOWS
                    float s_f = shadow_decode(layer->shadow_w_out[idx]);
                    FACTORED_ADAM_PARAM_UPDATE(
                        layer->m_w_out_row, layer->m_w_out_col,
                        layer->v_w_out_row, layer->v_w_out_col,
                        mean_v,
                        s_f,
                        lr_eff, r, c, inv_bc1, inv_bc2);
                    uint32_t rnd = shadow_rng(k, V12_SHADOW_MAT_W_OUT, idx,
                                              (uint64_t)state->step);
                    layer->shadow_w_out[idx] = stochastic_round_shadow(s_f, rnd);
#else
                    FACTORED_ADAM_PARAM_UPDATE(
                        layer->m_w_out_row, layer->m_w_out_col,
                        layer->v_w_out_row, layer->v_w_out_col,
                        mean_v,
                        layer->shadow_w_out[idx],
                        lr_eff, r, c, inv_bc1, inv_bc2);
                    layer->shadow_w_out[idx] = fmaxf(-10.0f, fminf(10.0f, layer->shadow_w_out[idx]));
#endif
#endif
                    layer->grad_accum_out[idx] = 0.0f;
                }
            } else {
                /* SLOW PATH — full scan (first 500 steps only) */
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        int idx = r * cols + c;
                        /* R356 F2 S6: decoded magnitude gate (no-op under flag OFF). */
                        if (fabsf(SHADOW_DECODE(layer->shadow_w_out[idx])) > V12_CRYSTALLIZATION_TAU) continue;
                        if (fabsf(layer->grad_accum_out[idx]) <= V12_EMA_SPRT_THRESHOLD) continue;

#if V12_ZMMA_ENABLED
                        float lr_zmma = lr_eff / sqrtf(zmma_var_floor);
                        float ema_g = layer->grad_accum_out[idx];
                        float step_dir = (ema_g > 0.0f) ? 1.0f : -1.0f;
#if V12_F2_INT8_SHADOWS
                        float s_f = shadow_decode(layer->shadow_w_out[idx]);
                        s_f -= lr_zmma * step_dir;
                        CDSVQ_KL_REG(s_f, lr_eff);
                        uint32_t rnd = shadow_rng(k, V12_SHADOW_MAT_W_OUT, idx,
                                                  (uint64_t)state->step);
                        layer->shadow_w_out[idx] = stochastic_round_shadow(s_f, rnd);
#else
                        float s_f = layer->shadow_w_out[idx] - lr_zmma * step_dir;
                        CDSVQ_KL_REG(s_f, lr_eff);
                        layer->shadow_w_out[idx] = fmaxf(-10.0f, fminf(10.0f, s_f));
#endif
#else
#if V12_AIST_ENABLED
                        {
                            float s_now = SHADOW_DECODE(layer->shadow_w_out[idx]);
                            float m_hat_ai = (layer->m_w_out_row[r] + layer->m_w_out_col[c]) * inv_bc1;
                            float v_approx_ai = layer->v_w_out_row[r] * layer->v_w_out_col[c] / (mean_v + 1e-30f);
                            float v_hat_ai = v_approx_ai * inv_bc2;
                            float g_ai = lr_eff * m_hat_ai / (sqrtf(v_hat_ai) + ADAM_EPS);
                            if (aist_should_skip(s_now, g_ai, v12_aist_epsilon)) {
                                layer->grad_accum_out[idx] = 0.0f;
                                continue;
                            }
                        }
#endif

#if V12_F2_INT8_SHADOWS
                        float s_f = shadow_decode(layer->shadow_w_out[idx]);
                        FACTORED_ADAM_PARAM_UPDATE(
                            layer->m_w_out_row, layer->m_w_out_col,
                            layer->v_w_out_row, layer->v_w_out_col,
                            mean_v,
                            s_f,
                            lr_eff, r, c, inv_bc1, inv_bc2);
                        uint32_t rnd = shadow_rng(k, V12_SHADOW_MAT_W_OUT, idx,
                                                  (uint64_t)state->step);
                        layer->shadow_w_out[idx] = stochastic_round_shadow(s_f, rnd);
#else
                        FACTORED_ADAM_PARAM_UPDATE(
                            layer->m_w_out_row, layer->m_w_out_col,
                            layer->v_w_out_row, layer->v_w_out_col,
                            mean_v,
                            layer->shadow_w_out[idx],
                            lr_eff, r, c, inv_bc1, inv_bc2);
                        layer->shadow_w_out[idx] = fmaxf(-10.0f, fminf(10.0f, layer->shadow_w_out[idx]));
#endif
#endif
                        layer->grad_accum_out[idx] = 0.0f;
                    }
                }
            }
        }  /* w_out block */
        }  /* GPU_OPT_W_OUT guard */

        /* ---- sigma (output scaling) Adam -------------------------------- */
        for (int i = 0; i < V12_D_STATE; i++) {
            ADAM_STEP(layer->m_sigma[i], layer->v_sigma[i], layer->d_sigma[i],
                      layer->sigma[i], lr_eff, inv_bc1, inv_bc2);
            /* No clamping needed — sigma can be negative (inverts W_in column) */
        }

        if (!(model->gpu_opt_flags & GPU_OPT_A)) {
        for (int i = 0; i < V12_D_STATE; i++) {
            ADAM_STEP(layer->m_A[i], layer->v_A[i], layer->dA[i],
                      layer->A[i], lr, inv_bc1, inv_bc2);
            layer->A[i] = fmaxf(0.01f, fminf(0.8f, layer->A[i]));
        }
        }  /* GPU_OPT_A guard */

        /* B and C always on CPU — GPU optimizer path disabled (upload overhead) */
        for (int i = 0; i < V12_D_STATE; i++) {
            ADAM_STEP(layer->m_B[i], layer->v_B[i], layer->dB[i],
                      layer->B[i], lr, inv_bc1, inv_bc2);
            layer->B[i] = fmaxf(-2.0f, fminf(2.0f, layer->B[i]));
        }

        /* C_bilinear: MVUE Adam update (R348 F6, Volterra degree-2 gate)
         * R356 F2 S8: decode/encode pair when flag ON. */
        {
            int ds = V12_D_STATE;
            for (int i = 0; i < ds; i++) {
                float dw_i = layer->dC_bilinear[i];
                if (dw_i == 0.0f) continue;
                float eff_dw = (layer->sampled_C_bilinear[i] == 0.0f) ? (dw_i * V12_BI_STE_FACTOR) : dw_i;
                float g_mvue = eff_dw;  /* R353: STE (1.57x more efficient than REINFORCE) */
#if V12_AIST_ENABLED
                {
                    float s_now = SHADOW_DECODE(layer->shadow_C_bilinear[i]);
                    /* Mirror the post-ADAM_STEP moments without mutating them.
                     * ADAM_STEP would set:
                     *   m' = beta1*m + (1-beta1)*g
                     *   v' = beta2*v + (1-beta2)*g*g
                     * Then subtract lr * (m' * inv_bc1) / (sqrt(v' * inv_bc2) + eps). */
                    float m_peek = ADAM_BETA1 * layer->m_C_bilinear[i] + (1.0f - ADAM_BETA1) * g_mvue;
                    float v_peek = ADAM_BETA2 * layer->v_C_bilinear[i] + (1.0f - ADAM_BETA2) * g_mvue * g_mvue;
                    float m_hat_ai = m_peek * inv_bc1;
                    float v_hat_ai = v_peek * inv_bc2;
                    float g_ai = lr_eff * m_hat_ai / (sqrtf(v_hat_ai) + ADAM_EPS);
                    if (aist_should_skip(s_now, g_ai, v12_aist_epsilon)) {
                        continue;
                    }
                }
#endif
#if V12_F2_INT8_SHADOWS
                float s_f = shadow_decode(layer->shadow_C_bilinear[i]);
                ADAM_STEP(layer->m_C_bilinear[i], layer->v_C_bilinear[i],
                          g_mvue, s_f, lr_eff, inv_bc1, inv_bc2);
                uint32_t rnd = shadow_rng(k, V12_SHADOW_MAT_C_BILINEAR, i,
                                          (uint64_t)state->step);
                layer->shadow_C_bilinear[i] = stochastic_round_shadow(s_f, rnd);
#else
                ADAM_STEP(layer->m_C_bilinear[i], layer->v_C_bilinear[i],
                          g_mvue, layer->shadow_C_bilinear[i], lr_eff, inv_bc1, inv_bc2);
                layer->shadow_C_bilinear[i] = fmaxf(-10.0f, fminf(10.0f, layer->shadow_C_bilinear[i]));
#endif
            }
        }

        /* u: MVUE Adam update */
        if (layer->du != NULL && layer->shadow_u != NULL) {
            int ds = V12_D_STATE;
            for (int i = 0; i < ds; i++) {
                float dw_i = layer->du[i];
                if (dw_i == 0.0f) continue;
                float eff_dw = (layer->sampled_u[i] == 0.0f) ? (dw_i * V12_BI_STE_FACTOR) : dw_i;
                float g_mvue = eff_dw;
#if V12_AIST_ENABLED
                {
                    float s_now = SHADOW_DECODE(layer->shadow_u[i]);
                    float m_peek = ADAM_BETA1 * layer->m_u[i] + (1.0f - ADAM_BETA1) * g_mvue;
                    float v_peek = ADAM_BETA2 * layer->v_u[i] + (1.0f - ADAM_BETA2) * g_mvue * g_mvue;
                    float m_hat_ai = m_peek * inv_bc1;
                    float v_hat_ai = v_peek * inv_bc2;
                    float g_ai = lr_eff * m_hat_ai / (sqrtf(v_hat_ai) + ADAM_EPS);
                    if (aist_should_skip(s_now, g_ai, v12_aist_epsilon)) {
                        continue;
                    }
                }
#endif
#if V12_F2_INT8_SHADOWS
                float s_f = shadow_decode(layer->shadow_u[i]);
                ADAM_STEP(layer->m_u[i], layer->v_u[i],
                          g_mvue, s_f, lr_eff, inv_bc1, inv_bc2);
                uint32_t rnd = shadow_rng(k, V12_SHADOW_MAT_U, i,
                                          (uint64_t)state->step);
                layer->shadow_u[i] = stochastic_round_shadow(s_f, rnd);
#else
                ADAM_STEP(layer->m_u[i], layer->v_u[i],
                          g_mvue, layer->shadow_u[i], lr_eff, inv_bc1, inv_bc2);
                layer->shadow_u[i] = fmaxf(-10.0f, fminf(10.0f, layer->shadow_u[i]));
#endif
            }
        }

        /* v: MVUE Adam update */
        if (layer->dv != NULL && layer->shadow_v != NULL) {
            int ds = V12_D_STATE;
            for (int i = 0; i < ds; i++) {
                float dw_i = layer->dv[i];
                if (dw_i == 0.0f) continue;
                float eff_dw = (layer->sampled_v[i] == 0.0f) ? (dw_i * V12_BI_STE_FACTOR) : dw_i;
                float g_mvue = eff_dw;
#if V12_AIST_ENABLED
                {
                    float s_now = SHADOW_DECODE(layer->shadow_v[i]);
                    float m_peek = ADAM_BETA1 * layer->m_v[i] + (1.0f - ADAM_BETA1) * g_mvue;
                    float v_peek = ADAM_BETA2 * layer->v_v[i] + (1.0f - ADAM_BETA2) * g_mvue * g_mvue;
                    float m_hat_ai = m_peek * inv_bc1;
                    float v_hat_ai = v_peek * inv_bc2;
                    float g_ai = lr_eff * m_hat_ai / (sqrtf(v_hat_ai) + ADAM_EPS);
                    if (aist_should_skip(s_now, g_ai, v12_aist_epsilon)) {
                        continue;
                    }
                }
#endif
#if V12_F2_INT8_SHADOWS
                float s_f = shadow_decode(layer->shadow_v[i]);
                ADAM_STEP(layer->m_v[i], layer->v_v[i],
                          g_mvue, s_f, lr_eff, inv_bc1, inv_bc2);
                uint32_t rnd = shadow_rng(k, V12_SHADOW_MAT_V, i,
                                          (uint64_t)state->step);
                layer->shadow_v[i] = stochastic_round_shadow(s_f, rnd);
#else
                ADAM_STEP(layer->m_v[i], layer->v_v[i],
                          g_mvue, layer->shadow_v[i], lr_eff, inv_bc1, inv_bc2);
                layer->shadow_v[i] = fmaxf(-10.0f, fminf(10.0f, layer->shadow_v[i]));
#endif
            }
        }

        /* V-free: C is identity, not trained — skip Adam step */
        if (layer->dC != NULL) {
            int ds = V12_D_STATE;
#if V12_ASPNG_ENABLED
            float B_diag[V12_MAX_D_STATE] = {0};
            float A_diag[V12_MAX_D_STATE] = {0};
            
            for (int r = 0; r < ds; r++) {
                for (int c = 0; c < ds; c++) {
                    int i = r * ds + c;
                    layer->m_C[i] = ADAM_BETA1 * layer->m_C[i] + (1.0f - ADAM_BETA1) * layer->dC[i];
                    layer->v_C[i] = ADAM_BETA2 * layer->v_C[i] + (1.0f - ADAM_BETA2) * layer->dC[i] * layer->dC[i];
                    B_diag[r] += layer->v_C[i];
                    A_diag[c] += layer->v_C[i];
                }
            }
            float sum_vr = 0.0f, sum_vc = 0.0f;
            for (int i = 0; i < ds; i++) {
                B_diag[i] /= ds;
                A_diag[i] /= ds;
                sum_vr += B_diag[i];
                sum_vc += A_diag[i];
            }
            float mean_v = (sum_vr / ds) * (sum_vc / ds);

            for (int r = 0; r < ds; r++) {
                for (int c = 0; c < ds; c++) {
                    int i = r * ds + c;
                    float m_hat = layer->m_C[i] * inv_bc1;
                    float v_approx = B_diag[r] * A_diag[c] / (mean_v + 1e-30f);
                    float v_hat = v_approx * inv_bc2;
                    layer->C[i] -= lr * m_hat / (sqrtf(v_hat) + ADAM_EPS);
                    layer->C[i] = fmaxf(-2.0f, fminf(2.0f, layer->C[i]));
                }
            }
#else
            int d_state_sq = V12_D_STATE * V12_D_STATE;
            for (int i = 0; i < d_state_sq; i++) {
                ADAM_STEP(layer->m_C[i], layer->v_C[i], layer->dC[i],
                          layer->C[i], lr, inv_bc1, inv_bc2);
                layer->C[i] = fmaxf(-2.0f, fminf(2.0f, layer->C[i]));
            }
#endif
        }

        /* Conv1d kernel Adam step (no crystallization, no sign history)
         * R356 F2 S8: decode/encode pair when flag ON. The conv weight clip
         * is the tighter [-2, 2] range, which is enforced explicitly on the
         * FP32 transient before re-encoding (the stochastic rounder's
         * internal [-10, 10] clip is a strict superset and cannot tighten
         * a value that has already been clipped to [-2, 2]). */
        if (!(model->gpu_opt_flags & GPU_OPT_CONV)) {
            const int n_conv = V12_CONV_KERNEL * V12_D_STATE;
            for (int i = 0; i < n_conv; i++) {
#if V12_F2_INT8_SHADOWS
                float s_f = shadow_decode(layer->shadow_w_conv[i]);
                ADAM_STEP(layer->m_conv[i], layer->v_conv[i], layer->d_conv_w[i],
                          s_f, lr_eff, inv_bc1, inv_bc2);
                s_f = fmaxf(-2.0f, fminf(2.0f, s_f));
                uint32_t rnd = shadow_rng(k, V12_SHADOW_MAT_W_CONV, i,
                                          (uint64_t)state->step);
                layer->shadow_w_conv[i] = stochastic_round_shadow(s_f, rnd);
#else
                ADAM_STEP(layer->m_conv[i], layer->v_conv[i], layer->d_conv_w[i],
                          layer->shadow_w_conv[i], lr_eff, inv_bc1, inv_bc2);
                layer->shadow_w_conv[i] = fmaxf(-2.0f, fminf(2.0f, layer->shadow_w_conv[i]));
#endif
            }
        }

        /* Input-dependent A gate: W_A [d_state × d_state] and b_A [d_state] Adam updates */
        if (!(model->gpu_opt_flags & GPU_OPT_WA)) {
            const int n_wa = V12_D_STATE * V12_D_STATE;
            for (int i = 0; i < n_wa; i++) {
                ADAM_STEP(layer->m_W_A[i], layer->v_W_A[i], layer->dW_A[i],
                          layer->W_A[i], lr_eff, inv_bc1, inv_bc2);
                layer->W_A[i] = fmaxf(-2.0f, fminf(2.0f, layer->W_A[i]));
            }

            for (int i = 0; i < V12_D_STATE; i++) {
                ADAM_STEP(layer->m_b_A[i], layer->v_b_A[i], layer->db_A[i],
                          layer->b_A[i], lr_eff, inv_bc1, inv_bc2);
                layer->b_A[i] = fmaxf(-2.0f, fminf(5.0f, layer->b_A[i]));
            }
        }

        /* MTP low-rank projection Adam update */
        if (layer->mtp_U != NULL) {
            int n_mtp = V12_D_STATE * V12_MTP_RANK;
            for (int i = 0; i < n_mtp; i++) {
                ADAM_STEP(layer->m_mtp_U[i], layer->v_mtp_U[i], layer->d_mtp_U[i],
                          layer->mtp_U[i], lr_eff, inv_bc1, inv_bc2);
                ADAM_STEP(layer->m_mtp_V[i], layer->v_mtp_V[i], layer->d_mtp_V[i],
                          layer->mtp_V[i], lr_eff, inv_bc1, inv_bc2);
            }
        }

        /* tau_mux SGD step (plain SGD, no crystallization, stays FP32) */
        model->tau_mux[k] -= lr_eff * model->d_tau_mux[k];
        if (model->tau_mux[k] < 0.0f) model->tau_mux[k] = 0.0f;
        if (model->tau_mux[k] > 2.0f) model->tau_mux[k] = 2.0f;

        if (r398_pafe_active) {
            double r398_pafe_t1 = omp_get_wtime();
            int r398_pafe_post_count = 0;
            int r398_pafe_post_excluded = 0;
            double r398_pafe_post_dot = r398_pafe_layer_dot(
                model, layer, k, &r398_pafe_post_count, &r398_pafe_post_excluded);
            double delta_g_signed = -(r398_pafe_post_dot - r398_pafe_pre_dot);
            double delta_g_positive = delta_g_signed > 0.0 ? delta_g_signed : 0.0;
            double delta_t_ms = (r398_pafe_t1 - r398_pafe_t0) * 1000.0;
            const char *invalid_reason = "none";
            int is_reallocatable = 1;

            if (r398_pafe_active_count <= 0 || r398_pafe_post_count <= 0) {
                invalid_reason = "no_active_cpu_elements";
                is_reallocatable = 0;
            } else if (delta_t_ms <= 0.0) {
                invalid_reason = "non_positive_delta_t";
                is_reallocatable = 0;
            } else if (r398_pafe_excluded_count > 0 || r398_pafe_post_excluded > 0) {
                invalid_reason = "gpu_owned_elements_excluded";
            }

            printf("[R398_PAFE_ROW] step=%d block_id=layer:%d:cpu_optimizer "
                   "block_partition_level=layer_cpu_optimizer "
                   "num_params_or_active_elements=%d action_bytes_or_active_count=%d "
                   "delta_g_signed=%.9e delta_g_positive=%.9e "
                   "delta_t_ms_direct=%.6f shared_t_ms_allocated=0.000000 "
                   "delta_t_ms_total=%.6f is_reallocatable=%d invalid_reason=%s\n",
                   step, k, r398_pafe_active_count, r398_pafe_active_count,
                   delta_g_signed, delta_g_positive, delta_t_ms, delta_t_ms,
                   is_reallocatable, invalid_reason);

            if (is_reallocatable && r398_pafe_windows[k].count < R398_PAFE_WINDOW) {
                int idx = r398_pafe_windows[k].count++;
                r398_pafe_windows[k].g[idx] = delta_g_signed;
                r398_pafe_windows[k].t[idx] = delta_t_ms;
            }
        }

        /* Accumulate diagonal Fisher: sum of squared gradients for this layer.
         * Must happen BEFORE the memset zeros at the end of this function.   */
        {
            float acc = 0.0f;
            int acc_elems = 0;
            /* Note: cannot be NULL here (already checked at top of loop), but guard for clarity */
            if (layer->dw_in != NULL) {
                if (!(model->gpu_opt_flags & GPU_OPT_W_IN)) {
                    for (int i = 0; i < n_in; i++) acc += layer->dw_in[i] * layer->dw_in[i];
                    acc_elems += n_in;
                }
                if (!(model->gpu_opt_flags & GPU_OPT_W_OUT)) {
                    for (int i = 0; i < n_out; i++) acc += layer->dw_out[i] * layer->dw_out[i];
                    acc_elems += n_out;
                }
                if (acc_elems > 0) {
                    grad_norm_acc[k] += acc / (float)acc_elems;
                }
            }
        }
    }

    fisher_acc_count++;
    if (r398_pafe_active && ((step + 1) % R398_PAFE_WINDOW == 0)) {
        r398_pafe_maybe_summary(step, r398_pafe_windows);
    }

    if (!(model->gpu_opt_flags & GPU_OPT_NORM)) {
        int n_norm = V12_D_MODEL;
        for (int i = 0; i < n_norm; i++) {
            ADAM_STEP(model->m_norm_w[i], model->v_norm_w[i], model->d_norm_w[i],
                      model->norm_w[i], lr, inv_bc1, inv_bc2);
            model->norm_w[i] = fmaxf(-5.0f, fminf(5.0f, model->norm_w[i]));

            ADAM_STEP(model->m_norm_b[i], model->v_norm_b[i], model->d_norm_b[i],
                      model->norm_b[i], lr, inv_bc1, inv_bc2);
            model->norm_b[i] = fmaxf(-5.0f, fminf(5.0f, model->norm_b[i]));
        }
    }

    if (!(model->gpu_opt_flags & GPU_OPT_EMBED_P) && model->d_embed_P && model->sa_embed_P) {
        float elr = lr * 0.1f;
        for (int i = 0; i < V12_D_EMBED * V12_D_MODEL; i++) {
            float g = model->d_embed_P[i];
            if (g == 0.0f) continue;
            SAOBM_STEP(model->sa_embed_P[i], g, model->embed_P[i], elr);
        }
    }

    if (!(model->gpu_opt_flags & GPU_OPT_EMBED_E) && model->d_embed_E && model->sa_embed_E) {
        float elr = lr * 0.1f;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < V12_VOCAB_SIZE * V12_D_EMBED; i++) {
            float g = model->d_embed_E[i];
            if (g == 0.0f) continue;
            SAOBM_STEP(model->sa_embed_E[i], g, model->embed_E[i], elr);
        }
    }

    float lr_eff_lm = lr * state->lr_per_layer[V12_N_LAYERS - 1];

    /* ---- lm_head factored Adam (hot tokens only) --------------------- */
#if !V12_TOLR_ENABLED
    if (!(model->gpu_opt_flags & GPU_OPT_LM_HEAD)) {
        int rows = V12_K_HOT;
        int cols = V12_D_MODEL;

        for (int h = 0; h < rows; h++) {
            float gm  = (model->dlm_head_cnt_row[h] > 0) ? model->dlm_head_g_row[h]  / (float)model->dlm_head_cnt_row[h]  : 0.0f;
            float gqm = (model->dlm_head_cnt_row[h] > 0) ? model->dlm_head_gq_row[h] / (float)model->dlm_head_cnt_row[h]  : 0.0f;
            model->m_lm_head_row[h] = ADAM_BETA1 * model->m_lm_head_row[h] + (1.0f-ADAM_BETA1)*gm;
            model->v_lm_head_row[h] = ADAM_BETA2 * model->v_lm_head_row[h] + (1.0f-ADAM_BETA2)*gqm;
        }
        for (int j = 0; j < cols; j++) {
            float gm  = (model->dlm_head_cnt_col[j] > 0) ? model->dlm_head_g_col[j]  / (float)model->dlm_head_cnt_col[j]  : 0.0f;
            float gqm = (model->dlm_head_cnt_col[j] > 0) ? model->dlm_head_gq_col[j] / (float)model->dlm_head_cnt_col[j]  : 0.0f;
            model->m_lm_head_col[j] = ADAM_BETA1 * model->m_lm_head_col[j] + (1.0f-ADAM_BETA1)*gm;
            model->v_lm_head_col[j] = ADAM_BETA2 * model->v_lm_head_col[j] + (1.0f-ADAM_BETA2)*gqm;
        }

        float sum_vr = 0.0f, sum_vc = 0.0f;
        for (int h = 0; h < rows; h++) sum_vr += model->v_lm_head_row[h];
        for (int j = 0; j < cols; j++) sum_vc += model->v_lm_head_col[j];
        float mean_v = (sum_vr / (float)rows) * (sum_vc / (float)cols);

        #pragma omp parallel for schedule(dynamic)
        for (int h = 0; h < rows; h++) {
            if (model->dlm_head_cnt_row[h] == 0) continue;
            int32_t v_s = model->hot_to_vocab[h];
            for (int j = 0; j < cols; j++) {
                int idx = (int)v_s * cols + j;
                /* R356 F2 S6: decoded magnitude gate (no-op under flag OFF). */
                if (fabsf(SHADOW_DECODE(model->shadow_lm_head[idx])) > V12_CRYSTALLIZATION_TAU) continue;

                /* R356 F2 S8: decode/encode pair when flag ON. */
#if V12_F2_INT8_SHADOWS
                float s_f = shadow_decode(model->shadow_lm_head[idx]);
                FACTORED_ADAM_PARAM_UPDATE(
                    model->m_lm_head_row, model->m_lm_head_col,
                    model->v_lm_head_row, model->v_lm_head_col,
                    mean_v,
                    s_f,
                    lr_eff_lm, h, j, inv_bc1, inv_bc2);
                uint32_t rnd = shadow_rng(0, V12_SHADOW_MAT_LM_HEAD, idx,
                                          (uint64_t)state->step);
                model->shadow_lm_head[idx] = stochastic_round_shadow(s_f, rnd);
#else
                FACTORED_ADAM_PARAM_UPDATE(
                    model->m_lm_head_row, model->m_lm_head_col,
                    model->v_lm_head_row, model->v_lm_head_col,
                    mean_v,
                    model->shadow_lm_head[idx],
                    lr_eff_lm, h, j, inv_bc1, inv_bc2);
                model->shadow_lm_head[idx] = fmaxf(-10.0f, fminf(10.0f, model->shadow_lm_head[idx]));
#endif
            }
        }
    }  /* GPU_OPT_LM_HEAD guard */
#endif

    /* ---- w_pred factored Adam ------------------------------------- */
    if (!(model->gpu_opt_flags & GPU_OPT_W_PRED)) {
        int rows = V12_D_MODEL;
        int cols = V12_D_MODEL;

        float g_row[V12_D_MODEL] = {0.0f};
        float g_col[V12_D_MODEL] = {0.0f};
        float gq_row[V12_D_MODEL] = {0.0f};
        float gq_col[V12_D_MODEL] = {0.0f};
        int   cnt_row[V12_D_MODEL] = {0};
        int   cnt_col[V12_D_MODEL] = {0};

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int idx = r * cols + c;
                /* R356 F2 S6: decoded magnitude gate (no-op under flag OFF). */
                if (fabsf(SHADOW_DECODE(model->shadow_w_pred[idx])) > V12_CRYSTALLIZATION_TAU) continue;
                float dw_i = model->dw_pred[idx];
                float eff_dw = (model->sampled_w_pred[idx] == 0.0f) ? (dw_i * V12_BI_STE_FACTOR) : dw_i;
                float g_mvue = eff_dw;  /* R353: STE (1.57x more efficient than REINFORCE) */
                if (g_mvue == 0.0f) continue;
                g_row[r]  += g_mvue;
                g_col[c]  += g_mvue;
                gq_row[r] += g_mvue * g_mvue;
                gq_col[c] += g_mvue * g_mvue;
                cnt_row[r]++;
                cnt_col[c]++;
            }
        }

        for (int r = 0; r < rows; r++) {
            float gm  = (cnt_row[r] > 0) ? g_row[r]  / (float)cnt_row[r]  : 0.0f;
            float gqm = (cnt_row[r] > 0) ? gq_row[r] / (float)cnt_row[r]  : 0.0f;
            model->m_w_pred_row[r] = ADAM_BETA1 * model->m_w_pred_row[r] + (1.0f-ADAM_BETA1)*gm;
            model->v_w_pred_row[r] = ADAM_BETA2 * model->v_w_pred_row[r] + (1.0f-ADAM_BETA2)*gqm;
        }
        for (int c = 0; c < cols; c++) {
            float gm  = (cnt_col[c] > 0) ? g_col[c]  / (float)cnt_col[c]  : 0.0f;
            float gqm = (cnt_col[c] > 0) ? gq_col[c] / (float)cnt_col[c]  : 0.0f;
            model->m_w_pred_col[c] = ADAM_BETA1 * model->m_w_pred_col[c] + (1.0f-ADAM_BETA1)*gm;
            model->v_w_pred_col[c] = ADAM_BETA2 * model->v_w_pred_col[c] + (1.0f-ADAM_BETA2)*gqm;
        }

        float sum_vr = 0.0f, sum_vc = 0.0f;
        for (int r = 0; r < rows; r++) sum_vr += model->v_w_pred_row[r];
        for (int c = 0; c < cols; c++) sum_vc += model->v_w_pred_col[c];
        float mean_v = (sum_vr / (float)rows) * (sum_vc / (float)cols);

        #pragma omp parallel for schedule(static) collapse(2)
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int idx = r * cols + c;
                /* R356 F2 S6: decoded magnitude gate (no-op under flag OFF). */
                if (fabsf(SHADOW_DECODE(model->shadow_w_pred[idx])) > V12_CRYSTALLIZATION_TAU) continue;
                float dw_i = model->dw_pred[idx];
                if (dw_i == 0.0f) continue;
                float eff_dw = (model->sampled_w_pred[idx] == 0.0f) ? (dw_i * V12_BI_STE_FACTOR) : dw_i;
                float g_mvue = eff_dw;  /* R353: STE (1.57x more efficient than REINFORCE) */
                if (g_mvue == 0.0f) continue;

                /* R356 F2 S8: decode/encode pair when flag ON. */
#if V12_F2_INT8_SHADOWS
                float s_f = shadow_decode(model->shadow_w_pred[idx]);
                FACTORED_ADAM_PARAM_UPDATE(
                    model->m_w_pred_row, model->m_w_pred_col,
                    model->v_w_pred_row, model->v_w_pred_col,
                    mean_v,
                    s_f,
                    lr_eff_lm, r, c, inv_bc1, inv_bc2);
                uint32_t rnd = shadow_rng(0, V12_SHADOW_MAT_W_PRED, idx,
                                          (uint64_t)state->step);
                model->shadow_w_pred[idx] = stochastic_round_shadow(s_f, rnd);
#else
                FACTORED_ADAM_PARAM_UPDATE(
                    model->m_w_pred_row, model->m_w_pred_col,
                    model->v_w_pred_row, model->v_w_pred_col,
                    mean_v,
                    model->shadow_w_pred[idx],
                    lr_eff_lm, r, c, inv_bc1, inv_bc2);
                model->shadow_w_pred[idx] = fmaxf(-10.0f, fminf(10.0f, model->shadow_w_pred[idx]));
#endif
            }
        }
    }  /* GPU_OPT_W_PRED guard */

    for (int k = 0; k < V12_N_LAYERS; k++) {
        SSMLayer *layer = &model->layers[k];
        if (layer->dw_in == NULL) continue;  /* offloaded layer — skip */
        memset(layer->dw_in,  0, V12_D_MODEL * V12_D_STATE * sizeof(float));
        memset(layer->dw_out, 0, V12_D_STATE * V12_D_MODEL * sizeof(float));
        memset(layer->d_sigma, 0, V12_D_STATE * sizeof(float));
        memset(layer->dA,     0, V12_D_STATE * sizeof(float));
        memset(layer->dB,     0, V12_D_STATE * sizeof(float));
        memset(layer->dB_seq, 0, V12_SEQ_LEN * V12_D_STATE * sizeof(float));
        if (layer->dC_bilinear != NULL)
            memset(layer->dC_bilinear, 0, V12_D_STATE * sizeof(float));
        if (layer->du != NULL)
            memset(layer->du, 0, V12_D_STATE * sizeof(float));
        if (layer->dv != NULL)
            memset(layer->dv, 0, V12_D_STATE * sizeof(float));
        if (layer->dC != NULL) memset(layer->dC,     0, V12_D_STATE * V12_D_STATE * sizeof(float));
        memset(layer->d_conv_w, 0, V12_CONV_KERNEL * V12_D_STATE * sizeof(float));
        memset(layer->dW_A,   0, V12_D_STATE * V12_D_STATE * sizeof(float));
        memset(layer->db_A,   0, V12_D_STATE * sizeof(float));

        if (model->layers[k].d_mtp_U != NULL) {
            memset(model->layers[k].d_mtp_U, 0, V12_D_STATE * V12_MTP_RANK * sizeof(float));
            memset(model->layers[k].d_mtp_V, 0, V12_D_STATE * V12_MTP_RANK * sizeof(float));
        }
    }

#if !V12_TOLR_ENABLED
    if (model->dlm_head_g_row) {
        memset(model->dlm_head_g_row, 0, V12_K_HOT * sizeof(float));
        memset(model->dlm_head_gq_row, 0, V12_K_HOT * sizeof(float));
        memset(model->dlm_head_cnt_row, 0, V12_K_HOT * sizeof(int32_t));
    }
    if (model->dlm_head_g_col) {
        memset(model->dlm_head_g_col, 0, V12_D_MODEL * sizeof(float));
        memset(model->dlm_head_gq_col, 0, V12_D_MODEL * sizeof(float));
        memset(model->dlm_head_cnt_col, 0, V12_D_MODEL * sizeof(int32_t));
    }
#endif
    if (!(model->gpu_opt_flags & GPU_OPT_EMBED_E) && model->d_embed_E)
        memset(model->d_embed_E, 0, (size_t)V12_VOCAB_SIZE * V12_D_EMBED * sizeof(float));
    if (!(model->gpu_opt_flags & GPU_OPT_EMBED_P) && model->d_embed_P)
        memset(model->d_embed_P, 0, (size_t)V12_D_EMBED * V12_D_MODEL * sizeof(float));
    if (!(model->gpu_opt_flags & GPU_OPT_NORM)) {
        memset(model->d_norm_w, 0, V12_D_MODEL * sizeof(float));
        memset(model->d_norm_b, 0, V12_D_MODEL * sizeof(float));
    }
    if (!(model->gpu_opt_flags & GPU_OPT_W_PRED)) {
        memset(model->dw_pred,  0, V12_D_MODEL * V12_D_MODEL * sizeof(float));
    }

    /* tau_mux gradient reset */
    for (int k = 0; k < V12_N_LAYERS; k++) {
        model->d_tau_mux[k] = 0.0f;
    }

    if (step > 0 && step % V12_REPACK_INTERVAL == 0) {
        /* Update per-layer LR multipliers from accumulated Fisher information.
         * lr_per_layer[k] = mean_F / F_k  (normalized so mean over layers = 1.0)
         * Layers with high gradient magnitude get smaller LR (stability).
         * Layers with low gradient magnitude get larger LR (acceleration).    */
        if (fisher_acc_count > 0) {
            float f[V12_N_LAYERS];
            float mean_f = 0.0f;
            for (int k = 0; k < V12_N_LAYERS; k++) {
                f[k] = grad_norm_acc[k] / (float)fisher_acc_count;
                mean_f += f[k];
            }
            mean_f /= (float)V12_N_LAYERS;

            if (mean_f > 1e-12f && state->fp_completed) {
                for (int k = 0; k < V12_N_LAYERS; k++) {
                    float raw = mean_f / (f[k] + 1e-8f);
                    /* Clamp to [0.1, 10.0] to prevent runaway scaling.           */
                    state->lr_per_layer[k] = fmaxf(0.1f, fminf(10.0f, raw));
                }
            }
            /* else: all gradients near zero OR FP active — leave lr_per_layer unchanged */

            /* Reset accumulators for next window */
            for (int k = 0; k < V12_N_LAYERS; k++) grad_norm_acc[k] = 0.0f;
            fisher_acc_count = 0;

            /* Log scaling factors */
            printf("[FISHER] step=%u lr_per_layer=[", state->step);
            for (int k = 0; k < V12_N_LAYERS; k++) {
                printf("%.3f%s", state->lr_per_layer[k], (k < V12_N_LAYERS-1) ? "," : "");
            }
            printf("] mean_F=%.4e\n", mean_f);
        }

        if (gpu_opt_wio == 0) {
            for (int k = 0; k < V12_N_LAYERS; k++) {
                if (model->layers[k].dw_in == NULL) continue; /* offloaded */
                update_crystallization(&model->layers[k], k, state, step);
            }
            for (int k = 0; k < V12_N_LAYERS; k++) {
                if (model->layers[k].dw_in == NULL) continue; /* offloaded */
                optimizer_rebuild_alive_idx(&model->layers[k]);
            }
            for (int k = 0; k < V12_N_LAYERS; k++) {
                if (model->layers[k].dw_in == NULL) continue; /* offloaded */
                if (model_rebuild_layer_csr(&model->layers[k]) != 0) {
                    fprintf(stderr, "[OPT] CSR rebuild failed at layer %d step %d\n",
                            k, step);
                }
            }
        }
        if (!(model->gpu_opt_flags & GPU_OPT_LM_HEAD)
            && model_rebuild_lm_head_csr(model) != 0) {
            fprintf(stderr, "[OPT] lm_head CSR rebuild failed at step %d\n", step);
        }

        if (!(model->gpu_opt_flags & GPU_OPT_W_PRED)) {
            /* Resample w_pred from shadow logits.
             * R356 F2 S5: route through SHADOW_DECODE (no-op under flag OFF). */
            int n_pred = V12_D_MODEL * V12_D_MODEL;
            for (int i = 0; i < n_pred; i++) {
                float s = SHADOW_DECODE(model->shadow_w_pred[i]);
                float u = uniform01(&state->rng_state);
                model->sampled_w_pred[i] = sample_ternary_from_logit_mvue(s, u);
            }
        }
    }

#if V12_TSMSQ_ENABLED
    if (step == 0) {
        printf("[OPT] TSMSQ (Ternary Shadow-Moment Shared Quantization) active: block_size=64\n");
    }

    /* ---- TSMSQ: Ternary Shadow-Moment Shared Quantization ---- */
    for (int k = 0; k < V12_N_LAYERS; k++) {
        SSMLayer *layer = &model->layers[k];
        if (layer->dw_in == NULL) continue;  /* offloaded layer — skip */

        if (!(model->gpu_opt_flags & GPU_OPT_W_IN)) {
            quantize_moment_block(layer->m_w_in_row, layer->v_w_in_row, V12_D_MODEL, &state->rng_state);
            quantize_moment_block(layer->m_w_in_col, layer->v_w_in_col, V12_D_STATE, &state->rng_state);
        }
        if (!(model->gpu_opt_flags & GPU_OPT_W_OUT)) {
            quantize_moment_block(layer->m_w_out_row, layer->v_w_out_row, V12_D_STATE, &state->rng_state);
            quantize_moment_block(layer->m_w_out_col, layer->v_w_out_col, V12_D_MODEL, &state->rng_state);
        }
        quantize_moment_block(layer->m_sigma, layer->v_sigma, V12_D_STATE, &state->rng_state);
        
        if (!(model->gpu_opt_flags & GPU_OPT_A)) {
            quantize_moment_block(layer->m_A, layer->v_A, V12_D_STATE, &state->rng_state);
        }
        quantize_moment_block(layer->m_B, layer->v_B, V12_D_STATE, &state->rng_state);
        
        if (layer->m_C_bilinear != NULL) {
            quantize_moment_block(layer->m_C_bilinear, layer->v_C_bilinear, V12_D_STATE, &state->rng_state);
        }
        if (layer->m_u != NULL) {
            quantize_moment_block(layer->m_u, layer->v_u, V12_D_STATE, &state->rng_state);
        }
        if (layer->m_v != NULL) {
            quantize_moment_block(layer->m_v, layer->v_v, V12_D_STATE, &state->rng_state);
        }
        if (layer->m_C != NULL) {
            quantize_moment_block(layer->m_C, layer->v_C, V12_D_STATE * V12_D_STATE, &state->rng_state);
        }
        if (!(model->gpu_opt_flags & GPU_OPT_CONV)) {
            quantize_moment_block(layer->m_conv, layer->v_conv, V12_CONV_KERNEL * V12_D_STATE, &state->rng_state);
        }
        if (!(model->gpu_opt_flags & GPU_OPT_WA)) {
            quantize_moment_block(layer->m_W_A, layer->v_W_A, V12_D_STATE * V12_D_STATE, &state->rng_state);
            quantize_moment_block(layer->m_b_A, layer->v_b_A, V12_D_STATE, &state->rng_state);
        }
        if (layer->mtp_U != NULL) {
            quantize_moment_block(layer->m_mtp_U, layer->v_mtp_U, V12_D_STATE * V12_MTP_RANK, &state->rng_state);
            quantize_moment_block(layer->m_mtp_V, layer->v_mtp_V, V12_D_STATE * V12_MTP_RANK, &state->rng_state);
        }
    }

    if (!(model->gpu_opt_flags & GPU_OPT_NORM)) {
        quantize_moment_block(model->m_norm_w, model->v_norm_w, V12_D_MODEL, &state->rng_state);
        quantize_moment_block(model->m_norm_b, model->v_norm_b, V12_D_MODEL, &state->rng_state);
    }
    if (!(model->gpu_opt_flags & GPU_OPT_EMBED_P) && model->m_embed_P) {
        quantize_moment_block(model->m_embed_P, model->v_embed_P, V12_D_EMBED * V12_D_MODEL, &state->rng_state);
    }
    if (!(model->gpu_opt_flags & GPU_OPT_EMBED_E) && model->m_embed_E_col) {
        quantize_moment_block(model->m_embed_E_col, model->v_embed_E_col, V12_D_EMBED, &state->rng_state);
    }
#if !V12_TOLR_ENABLED
    if (!(model->gpu_opt_flags & GPU_OPT_LM_HEAD)) {
        quantize_moment_block(model->m_lm_head_row, model->v_lm_head_row, V12_K_HOT, &state->rng_state);
        quantize_moment_block(model->m_lm_head_col, model->v_lm_head_col, V12_D_MODEL, &state->rng_state);
    }
#endif
    if (!(model->gpu_opt_flags & GPU_OPT_W_PRED)) {
        quantize_moment_block(model->m_w_pred_row, model->v_w_pred_row, V12_D_MODEL, &state->rng_state);
        quantize_moment_block(model->m_w_pred_col, model->v_w_pred_col, V12_D_MODEL, &state->rng_state);
    }
#endif

    if (0) {
        /* R356 F2: read through SHADOW_DECODE for cross-flag compilation. */
        printf("[FACTORED-ADAM] step=10 m_w_in_row[0]=%.4e v_w_in_row[0]=%.4e "
               "shadow_w_in[0]=%.4f\n",
               model->layers[0].m_w_in_row[0],
               model->layers[0].v_w_in_row[0],
               (double)SHADOW_DECODE(model->layers[0].shadow_w_in[0]));
    }

    if (0) {
        printf("[OPT] step=%u lr=%.4f crystal=%.3f\n",
               state->step, lr, state->crystal_fraction);
        printf("[OPT-MUX] step=%u tau=[", state->step);
        for (int k = 0; k < V12_N_LAYERS; k++) {
            printf("%.3f%s", model->tau_mux[k], (k < V12_N_LAYERS-1) ? "," : "");
        }
        printf("]\n");
    }
}

/* =========================================================================
 * Function 3: lr_schedule
 *
 * Warmup + cosine decay learning rate schedule.
 * ========================================================================= */

float lr_schedule(const TrainState *state) {
    int step = (int)state->step;
    float base  = state->lr_base;
    float floor = base * V12_LR_MIN_FACTOR;

    float lr;
    if (step < V12_LR_WARMUP_STEPS) {
        lr = base * (float)step / (float)V12_LR_WARMUP_STEPS;
    } else {
        float t = (float)(step - V12_LR_WARMUP_STEPS)
                / (float)(V12_LR_TOTAL_STEPS - V12_LR_WARMUP_STEPS);
        lr = base * 0.5f * (1.0f + cosf(3.14159265f * t));
    }

    float final_lr = fmaxf(lr, floor);
    if (state->sre_lr_scale > 0.0f) {
        final_lr *= state->sre_lr_scale;
    }
    return final_lr;
}

/* =========================================================================
 * Function 4: update_crystallization
 *
 * Scan shadow logits and set crystallization masks where |s| > tau.
 * ========================================================================= */

void update_crystallization(SSMLayer *layer, int layer_idx,
                             TrainState *state, int step) {
    int n_in  = V12_D_MODEL * V12_D_STATE;

    int total_weights = n_in;
    int frozen_count  = 0;

    /* R356 F2 S6: crystallization gate via SHADOW_DECODE (no-op under flag OFF). */
    for (int i = 0; i < n_in; i++) {
        if (fabsf(SHADOW_DECODE(layer->shadow_w_in[i])) > V12_CRYSTALLIZATION_TAU) {
            layer->crystal_mask_in[i] = 1;
        } else {
            layer->crystal_mask_in[i] = 0;
        }
        frozen_count += layer->crystal_mask_in[i];
    }

    if (layer_idx == 0) {
        state->crystal_fraction = 0.0f;
    }

    float total_model_weights = (float)(2 * n_in) * V12_N_LAYERS;
    state->crystal_fraction += (float)frozen_count / ((float)total_weights * V12_N_LAYERS);
}
