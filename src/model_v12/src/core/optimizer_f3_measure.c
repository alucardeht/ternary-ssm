/* R356 F3 — Gradient covariance rank measurement implementation.
 *
 * Compiled ONLY when V12_F3_MEASURE is defined. Default build leaves this
 * translation unit empty, so linking is unaffected.
 *
 * Statistic measured per step t on G = model->layers[0].dw_in (m x n fp32):
 *   fro_sq       = sum_{i,j} G[i,j]^2
 *   sigma1_sq    = sigma_1(G)^2 via 20 iterations of power iteration
 *   sigma2_sq    = sigma_1(G - sigma1 u1 v1^T)^2 (rank-1 deflation, 20 iters)
 *   ratio1       = sigma1_sq / fro_sq
 *   ratio12      = (sigma1_sq + sigma2_sq) / fro_sq
 *
 * Determinism: seed vector is uniform 1/sqrt(n); no RNG. Run-to-run stats are
 * bitwise reproducible for identical inputs.
 *
 * Gates: compile flag V12_F3_MEASURE AND runtime getenv("V12_F3_MEASURE").
 * Window: step in [F3_STEP_LO, F3_STEP_HI]. Target: layer 0, matrix w_in.
 * GPU-resident path: log-and-skip when GPU_OPT_W_IN is active.
 */

#include "optimizer_f3_measure.h"

#ifdef V12_F3_MEASURE

#include "types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Measurement window (inclusive). */
#define F3_STEP_LO           100
#define F3_STEP_HI           1100
#define F3_POWER_ITERS       20
#define F3_FRO_SQ_EPS        1e-20f

/* Verdict thresholds (from handoff section 5). */
#define F3_PASS_THRESHOLD    0.95f
#define F3_BORDERLINE_LOW    0.70f

/* Module-static aggregators: survive across optimizer_step calls. */
static int    g_enabled_checked = 0;  /* lazy env-var resolution */
static int    g_enabled         = 0;
static int    g_summary_printed = 0;

static int    g_n_samples       = 0;
static double g_ratio1_sum      = 0.0;
static double g_ratio1_sum_sq   = 0.0;
static float  g_ratio1_min      = 1e30f;
static float  g_ratio1_max      = -1e30f;
static int    g_frac_above_cnt  = 0;
static double g_ratio12_sum     = 0.0;
static int    g_gpu_skip_logged = 0;

static int f3_enabled(void) {
    if (!g_enabled_checked) {
        const char *v = getenv("V12_F3_MEASURE");
        g_enabled = (v != NULL && v[0] != '\0' && v[0] != '0');
        g_enabled_checked = 1;
    }
    return g_enabled;
}

/* One power iteration on (m x n) matrix G to extract top singular pair.
 *  v_out <- normalize(G^T u), u_out <- normalize(G v_out)
 * After F3_POWER_ITERS rounds, sigma1 = ||G v||_2. Writes u1, v1, returns
 * sigma1 (non-negative). Uses provided scratch vectors only; never mutates G.
 */
static float f3_power_iteration(const float *G, int m, int n,
                                 float *u, float *v)
{
    /* Deterministic seed: uniform 1/sqrt(n) right singular vector. */
    const float vseed = 1.0f / sqrtf((float)n);
    for (int j = 0; j < n; j++) v[j] = vseed;

    float sigma = 0.0f;
    for (int it = 0; it < F3_POWER_ITERS; it++) {
        /* u = G v */
        for (int i = 0; i < m; i++) {
            float s = 0.0f;
            const float *row = G + (size_t)i * n;
            for (int j = 0; j < n; j++) s += row[j] * v[j];
            u[i] = s;
        }
        float unorm_sq = 0.0f;
        for (int i = 0; i < m; i++) unorm_sq += u[i] * u[i];
        if (unorm_sq < 1e-30f) return 0.0f;
        float inv_unorm = 1.0f / sqrtf(unorm_sq);
        for (int i = 0; i < m; i++) u[i] *= inv_unorm;

        /* v = G^T u */
        for (int j = 0; j < n; j++) v[j] = 0.0f;
        for (int i = 0; i < m; i++) {
            const float *row = G + (size_t)i * n;
            float ui = u[i];
            for (int j = 0; j < n; j++) v[j] += row[j] * ui;
        }
        float vnorm_sq = 0.0f;
        for (int j = 0; j < n; j++) vnorm_sq += v[j] * v[j];
        if (vnorm_sq < 1e-30f) return 0.0f;
        sigma = sqrtf(vnorm_sq);  /* ||G^T u|| = sigma_1 at convergence */
        float inv_vnorm = 1.0f / sigma;
        for (int j = 0; j < n; j++) v[j] *= inv_vnorm;
    }
    return sigma;
}

void optimizer_f3_measure_step(const Model *model, int step)
{
    if (!f3_enabled()) return;
    if (step < F3_STEP_LO || step > F3_STEP_HI) return;
    if (model == NULL) return;

    /* GPU-resident gradient path: log once and skip. The CPU buffer is
     * stale / NULL under GPU_OPT_W_IN. The user must disable that flag for
     * the measurement run — we never silently read undefined memory. */
    if (model->gpu_opt_flags & GPU_OPT_W_IN) {
        if (!g_gpu_skip_logged) {
            fprintf(stdout,
                "[F3-MEASURE] skip reason=gpu_resident flag=GPU_OPT_W_IN "
                "disable_this_flag_to_measure step=%d\n", step);
            fflush(stdout);
            g_gpu_skip_logged = 1;
        }
        return;
    }

    const SSMLayer *layer = &model->layers[0];
    if (layer->dw_in == NULL) return;  /* offloaded — nothing to read */

    const int m = V12_D_MODEL;
    const int n = V12_D_STATE;
    const float *G = layer->dw_in;

    /* Frobenius norm squared. */
    double fro_sq = 0.0;
    const size_t total = (size_t)m * (size_t)n;
    for (size_t i = 0; i < total; i++) fro_sq += (double)G[i] * (double)G[i];

    if (fro_sq < (double)F3_FRO_SQ_EPS) {
        /* Dead / pre-init gradient. Emit a parseable zero row and exclude
         * from the running averages (n_samples unchanged). */
        fprintf(stdout,
            "[F3-MEASURE] step=%d layer=0 matrix=w_in m=%d n=%d "
            "fro_sq=%.6e sigma1_sq=%.6e sigma2_sq=%.6e "
            "ratio1=%.4f ratio12=%.4f\n",
            step, m, n, fro_sq, 0.0, 0.0, 0.0, 0.0);
        fflush(stdout);
        return;
    }

    /* Stack scratch — m,n are known compile-time small (256, 64). */
    float u1[V12_D_MODEL];
    float v1[V12_D_STATE];

    float sigma1 = f3_power_iteration(G, m, n, u1, v1);
    float sigma1_sq = sigma1 * sigma1;

    /* Rank-1 deflation: G' = G - sigma1 * u1 * v1^T. Materialize a copy
     * so the model gradient is not touched. Read-only contract preserved. */
    float *Gd = (float *)malloc(total * sizeof(float));
    float sigma2 = 0.0f;
    if (Gd != NULL) {
        memcpy(Gd, G, total * sizeof(float));
        for (int i = 0; i < m; i++) {
            float a = sigma1 * u1[i];
            float *row = Gd + (size_t)i * n;
            for (int j = 0; j < n; j++) row[j] -= a * v1[j];
        }
        float u2[V12_D_MODEL];
        float v2[V12_D_STATE];
        sigma2 = f3_power_iteration(Gd, m, n, u2, v2);
        free(Gd);
    }
    float sigma2_sq = sigma2 * sigma2;

    float ratio1  = (float)((double)sigma1_sq / fro_sq);
    float ratio12 = (float)(((double)sigma1_sq + (double)sigma2_sq) / fro_sq);

    /* Clamp numerical overshoot to [0,1] — deflation residual can land a
     * hair above 1.0 in single precision. Log the raw value though. */
    fprintf(stdout,
        "[F3-MEASURE] step=%d layer=0 matrix=w_in m=%d n=%d "
        "fro_sq=%.6e sigma1_sq=%.6e sigma2_sq=%.6e "
        "ratio1=%.4f ratio12=%.4f\n",
        step, m, n, fro_sq, (double)sigma1_sq, (double)sigma2_sq,
        (double)ratio1, (double)ratio12);
    fflush(stdout);

    g_n_samples++;
    g_ratio1_sum    += (double)ratio1;
    g_ratio1_sum_sq += (double)ratio1 * (double)ratio1;
    g_ratio12_sum   += (double)ratio12;
    if (ratio1 < g_ratio1_min) g_ratio1_min = ratio1;
    if (ratio1 > g_ratio1_max) g_ratio1_max = ratio1;
    if (ratio1 > F3_PASS_THRESHOLD) g_frac_above_cnt++;
}

void optimizer_f3_measure_summary(void)
{
    if (!f3_enabled()) return;
    if (g_summary_printed) return;
    g_summary_printed = 1;

    if (g_n_samples <= 0) {
        fprintf(stdout,
            "[F3-SUMMARY] layer=0 matrix=w_in window=[%d,%d] n=0 "
            "ratio1_mean=0.0000 ratio1_std=0.0000 "
            "ratio1_min=0.0000 ratio1_max=0.0000 "
            "frac_above_0.95=0.000 ratio12_mean=0.0000 VERDICT=NO_DATA\n",
            F3_STEP_LO, F3_STEP_HI);
        fflush(stdout);
        return;
    }

    double mean  = g_ratio1_sum / (double)g_n_samples;
    double var   = g_ratio1_sum_sq / (double)g_n_samples - mean * mean;
    if (var < 0.0) var = 0.0;
    double stdv  = sqrt(var);
    double mean12 = g_ratio12_sum / (double)g_n_samples;
    double frac  = (double)g_frac_above_cnt / (double)g_n_samples;

    const char *verdict;
    if (mean > (double)F3_PASS_THRESHOLD)         verdict = "PASS_F3";
    else if (mean > (double)F3_BORDERLINE_LOW)    verdict = "BORDERLINE";
    else                                          verdict = "FAIL_F3";

    fprintf(stdout,
        "[F3-SUMMARY] layer=0 matrix=w_in window=[%d,%d] n=%d "
        "ratio1_mean=%.4f ratio1_std=%.4f "
        "ratio1_min=%.4f ratio1_max=%.4f "
        "frac_above_0.95=%.3f ratio12_mean=%.4f VERDICT=%s\n",
        F3_STEP_LO, F3_STEP_HI, g_n_samples,
        mean, stdv,
        (double)g_ratio1_min, (double)g_ratio1_max,
        frac, mean12, verdict);
    fflush(stdout);
}

#endif /* V12_F3_MEASURE */
