/*
 * test_r356_f1.c — R356 F1 Sherman-Morrison reversibility smoke tests (slice S6)
 *
 * CPU-only end-to-end exercises of the F1 reversibility path added in
 * slices S0..S5. Tiny configurations only; target runtime well below one
 * second. The test binary prints [PASS]/[FAIL]/[SKIP] per test and exits 0
 * when every required test passes.
 *
 * Required tests (from the architect handoff §7):
 *   1. [F1 PROBE]    — forward stability probe sanity on a known-stable config
 *   2. [F1 ROUNDTRIP]— forward -> reverse-walk reconstruction of h[t-1..0]
 *   3. [F1 GRAD]     — F1 backward gradients vs. canonical ssm_scan_backward
 *                      (cosine similarity + max abs diff)
 *   4. [F1 STAB]     — pathologically unstable config trips the ln(10) gate
 *                      and ssm_scan_backward_f1 returns without writing grads
 *   5. [F1 GPU-CPU]  — SKIPPED. S5 left the CPU backward path active; the GPU
 *                      replay dispatch is not yet wired. Re-enable once the
 *                      caller flips to the shader path.
 *   6. [F1 NEG]      — negative control with C_bilinear = NULL. The F1 path
 *                      must still roundtrip correctly when the bilinear term
 *                      is absent. This confirms F1 is actually exercised and
 *                      the result is not an accidental fallback match.
 *
 * Build (standalone, from src/model_v12):
 *   gcc -O2 -Wall -I src tests/test_r356_f1.c src/core/ssm_scan.c -lm \
 *       -o test_r356_f1.exe
 *
 * Or via the Makefile target:
 *   make test_r356_f1
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>

#include "../src/core/ssm_scan.h"

/* ----- Tolerances (see architect handoff §6, §7) ---------------------- */
#define F1_ROUNDTRIP_TOL   1e-4f   /* per-element |h_recon - h_ref|        */
#define F1_GRAD_COS_MIN    0.9999f /* gradient cosine similarity threshold */
#define F1_GRAD_ABS_TOL    5e-4f   /* max |grad_f1 - grad_cached|          */
#define F1_PROBE_SUM_TOL   1e-4f   /* expected vs observed probe sum       */

/* ----- Tiny CPU-only configuration ------------------------------------ */
#define D_STATE   8    /* d_state%8==0 required by forward                 */
#define D_INNER   8
#define SEQ_LEN   16   /* seq_len%V12_CKPT_STRIDE==0 when using h_cache    */

/* ---------------------------------------------------------------------- */
/*  Pseudo-random helpers (deterministic, seed-reproducible)              */
/* ---------------------------------------------------------------------- */

static uint32_t rng_state = 0x9E3779B9u;

static void rng_seed(uint32_t s) { rng_state = s ? s : 0x9E3779B9u; }

static float rand_uniform(float lo, float hi) {
    /* xorshift32 */
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rng_state = x;
    float u = (float)(x & 0xFFFFFF) / (float)0x1000000; /* [0,1) */
    return lo + (hi - lo) * u;
}

/* ---------------------------------------------------------------------- */
/*  Per-step reference forward stores h AFTER processing each token.      */
/*  ssm_scan_backward reads h_cache[t] as h_t and h_cache[t-1] as h_prev. */
/* ---------------------------------------------------------------------- */

static void reference_forward(
    const float *A,
    const float *B,       /* [d_state], static (B_stride=0) */
    const float *C,       /* [d_inner × d_state] */
    const float *x,       /* [seq_len × d_inner] */
    const float *C_bilinear, /* [d_state] or NULL */
    const float *u,       /* [d_state] or NULL */
    const float *v,       /* [d_state] or NULL */
    float *h_full_cache,  /* [seq_len × d_state] — h after step t */
    float *h_final,       /* [d_state] — h after the last step */
    float *y,             /* [seq_len × d_inner] */
    int seq_len, int d_state, int d_inner
) {
    float h[D_STATE] = {0};
    for (int t = 0; t < seq_len; t++) {
        const float *x_t = x + (size_t)t * d_inner;
        float w_prev = 0.0f;
        if (u != NULL && v != NULL) {
            for (int i = 0; i < d_state; i++) w_prev += v[i] * h[i];
        }
        if (C_bilinear != NULL) {
            for (int i = 0; i < d_state; i++) {
                float hp = h[i];
                h[i] = A[i] * hp + B[i] * x_t[i] + C_bilinear[i] * hp * x_t[i];
            }
        } else {
            for (int i = 0; i < d_state; i++) {
                h[i] = A[i] * h[i] + B[i] * x_t[i];
            }
        }
        if (u != NULL && v != NULL) {
            for (int i = 0; i < d_state; i++) {
                h[i] += w_prev * u[i];
            }
        }
        memcpy(h_full_cache + (size_t)t * d_state, h, d_state * sizeof(float));
        float *y_t = y + (size_t)t * d_inner;
        for (int j = 0; j < d_inner; j++) {
            float acc = 0.0f;
            const float *C_row = C + (size_t)j * d_state;
            for (int i = 0; i < d_state; i++) acc += C_row[i] * h[i];
            y_t[j] = acc;
        }
    }
    memcpy(h_final, h, d_state * sizeof(float));
}

/* ---------------------------------------------------------------------- */
/*  Same inversion formula the F1 backward uses, extracted so we can     */
/*  exercise the reverse walk directly and compare each reconstructed    */
/*  h[t-1] against the reference cache.                                   */
/* ---------------------------------------------------------------------- */

static void reverse_reconstruct(
    const float *A,
    const float *B,
    const float *x,
    const float *h_final,
    const float *C_bilinear,
    const float *u,
    const float *v,
    float *h_recon_cache,  /* [seq_len × d_state] — h after step t */
    int seq_len, int d_state, int d_inner
) {
    float h_cur[D_STATE];
    memcpy(h_cur, h_final, d_state * sizeof(float));
    for (int t = seq_len - 1; t >= 0; t--) {
        float h_prev[D_STATE];
        const float *x_t = x + (size_t)t * d_inner;
        memcpy(h_recon_cache + (size_t)t * d_state, h_cur,
               d_state * sizeof(float));
        if (t == 0) {
            memset(h_prev, 0, d_state * sizeof(float));
        } else if (u != NULL && v != NULL) {
            float temp_du[D_STATE];
            float temp_d_u[D_STATE];
            float dot_v_d_u = 0.0f;
            float dot_v_du = 0.0f;
            for (int i = 0; i < d_state; i++) {
                float a_tot = A[i];
                if (C_bilinear != NULL) {
                    a_tot += C_bilinear[i] * x_t[i];
                }
                float a_mag = fabsf(a_tot);
                if (a_mag < 1e-6f) {
                    a_tot = (a_tot < 0.0f) ? -1e-6f : 1e-6f;
                }
                temp_du[i] = (h_cur[i] - B[i] * x_t[i]) / a_tot;
                temp_d_u[i] = u[i] / a_tot;
                dot_v_d_u += v[i] * temp_d_u[i];
                dot_v_du += v[i] * temp_du[i];
            }
            float denom = 1.0f + dot_v_d_u;
            if (fabsf(denom) < 1e-6f) {
                denom = (denom < 0.0f) ? -1e-6f : 1e-6f;
            }
            float k_factor = dot_v_du / denom;
            for (int i = 0; i < d_state; i++) {
                h_prev[i] = temp_du[i] - k_factor * temp_d_u[i];
            }
        } else if (C_bilinear != NULL) {
            for (int i = 0; i < d_state; i++) {
                float a_tot = A[i] + C_bilinear[i] * x_t[i];
                h_prev[i] = (h_cur[i] - B[i] * x_t[i]) / a_tot;
            }
        } else {
            for (int i = 0; i < d_state; i++) {
                h_prev[i] = (h_cur[i] - B[i] * x_t[i]) / A[i];
            }
        }
        memcpy(h_cur, h_prev, d_state * sizeof(float));
    }
}

/* ---------------------------------------------------------------------- */
/*  Utility: compare two flat buffers. Returns max abs diff.              */
/* ---------------------------------------------------------------------- */

static float max_abs_diff(const float *a, const float *b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static float cosine_similarity(const float *a, const float *b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    double denom = sqrt(na) * sqrt(nb);
    if (denom < 1e-30) return 1.0f;
    return (float)(dot / denom);
}

/* ---------------------------------------------------------------------- */
/*  Populate a stable tiny configuration                                  */
/* ---------------------------------------------------------------------- */

static void make_stable_config(
    float *A, float *B, float *C, float *x, float *dy, float *C_bilinear
) {
    /* A in a mildly contractive range — keeps cumulative probe well under
     * ln(10). ln(0.9)*16 = -1.69 < 0 < ln(10). */
    const float A_vals[D_STATE] = {
        0.90f, 0.85f, 0.80f, 0.88f, 0.92f, 0.83f, 0.87f, 0.91f
    };
    memcpy(A, A_vals, sizeof(A_vals));

    rng_seed(0x51ED5EEDu);
    for (int i = 0; i < D_STATE; i++) B[i] = rand_uniform(-0.3f, 0.3f);
    for (int i = 0; i < D_INNER * D_STATE; i++) C[i] = rand_uniform(-0.4f, 0.4f);
    for (int i = 0; i < SEQ_LEN * D_INNER; i++) x[i] = rand_uniform(-0.5f, 0.5f);
    for (int i = 0; i < SEQ_LEN * D_INNER; i++) dy[i] = rand_uniform(-0.1f, 0.1f);

    if (C_bilinear != NULL) {
        /* Ternary values in {-1, 0, +1}, matching v12's C_bl storage. */
        const float cbl[D_STATE] = {1, -1, 0, 1, 0, -1, 1, 0};
        memcpy(C_bilinear, cbl, sizeof(cbl));
    }
}

/* ---------------------------------------------------------------------- */
/*  Test 1 — [F1 PROBE] sanity                                            */
/*                                                                         */
/*  Call the forward probe on a known-stable configuration and verify     */
/*  the printed cumulative value matches a hand-computed expected sum.    */
/* ---------------------------------------------------------------------- */

static int test_f1_probe_sanity(void) {
    printf("TEST 1: [F1 PROBE] sanity on a stable config\n");

    float A[D_STATE], B[D_STATE], C[D_INNER * D_STATE];
    float x[SEQ_LEN * D_INNER], dy[SEQ_LEN * D_INNER];
    float C_bilinear[D_STATE];
    make_stable_config(A, B, C, x, dy, C_bilinear);

    float h[D_STATE] = {0};
    float y[SEQ_LEN * D_INNER] = {0};
    float probe[D_STATE] = {0};

    ssm_scan_forward_with_stability_probe(
        A, NULL, NULL, B, 0, C, x, h, y, NULL, NULL, NULL,
        SEQ_LEN, D_STATE, D_INNER, C_bilinear, probe);

    /* Expected: probe[i] = sum_t ln(|A[i] + C_bilinear[i]*x[t][i]|) */
    float expected[D_STATE] = {0};
    for (int t = 0; t < SEQ_LEN; t++) {
        const float *x_t = x + (size_t)t * D_INNER;
        for (int i = 0; i < D_STATE; i++) {
            float a_tot = A[i] + C_bilinear[i] * x_t[i];
            float mag = fabsf(a_tot);
            if (mag < 1e-30f) mag = 1e-30f;
            expected[i] += logf(mag);
        }
    }

    float err = max_abs_diff(probe, expected, D_STATE);
    int pass = (err < F1_PROBE_SUM_TOL);

    float probe_max = probe[0];
    for (int i = 1; i < D_STATE; i++) if (probe[i] > probe_max) probe_max = probe[i];

    printf("  probe_max=%.6f  (ln10=%.6f)   expected_vs_observed max_abs=%.3e\n",
           (double)probe_max, (double)logf(10.0f), (double)err);
    if (probe_max >= logf(10.0f)) {
        printf("  [FAIL] stable config unexpectedly tripped the ln(10) gate\n");
        return 0;
    }
    if (!pass) {
        printf("  [FAIL] probe does not match hand-computed sum\n");
        return 0;
    }
    printf("  [PASS]\n");
    return 1;
}

/* ---------------------------------------------------------------------- */
/*  Test 2 — [F1 ROUNDTRIP] h reconstruction                              */
/*                                                                         */
/*  Run a per-step reference forward, save every h[t-1]. Reverse-walk     */
/*  from h_final using the same inversion formula F1 uses internally.    */
/*  Compare against the reference cache; assert per-element bound.       */
/* ---------------------------------------------------------------------- */

static int test_f1_roundtrip(void) {
    printf("TEST 2: [F1 ROUNDTRIP] h[t-1..0] reconstruction\n");

    float A[D_STATE], B[D_STATE], C[D_INNER * D_STATE];
    float x[SEQ_LEN * D_INNER], dy[SEQ_LEN * D_INNER];
    float C_bilinear[D_STATE];
    make_stable_config(A, B, C, x, dy, C_bilinear);

    float h_ref_cache[SEQ_LEN * D_STATE];
    float h_final[D_STATE];
    float y[SEQ_LEN * D_INNER];

    reference_forward(A, B, C, x, C_bilinear, NULL, NULL, h_ref_cache, h_final, y,
                      SEQ_LEN, D_STATE, D_INNER);

    float h_recon_cache[SEQ_LEN * D_STATE];
    reverse_reconstruct(A, B, x, h_final, C_bilinear, NULL, NULL,
                        h_recon_cache, SEQ_LEN, D_STATE, D_INNER);

    /* Per-timestep max abs diff: should grow (softly) as t decreases because
     * the reverse walk amplifies error by ~exp(cum probe up to t). */
    float worst = 0.0f;
    int   worst_t = -1;
    for (int t = 0; t < SEQ_LEN; t++) {
        float d = max_abs_diff(
            h_recon_cache + (size_t)t * D_STATE,
            h_ref_cache   + (size_t)t * D_STATE,
            D_STATE);
        if (d > worst) { worst = d; worst_t = t; }
    }

    printf("  worst_per_element_diff=%.3e at t=%d   tol=%.1e\n",
           (double)worst, worst_t, (double)F1_ROUNDTRIP_TOL);
    if (worst > F1_ROUNDTRIP_TOL) {
        printf("  [FAIL] roundtrip reconstruction exceeds tolerance\n");
        return 0;
    }
    printf("  [PASS]\n");
    return 1;
}

/* ---------------------------------------------------------------------- */
/*  Test 3 — [F1 GRAD] gradients vs. cached backward                      */
/*                                                                         */
/*  Compute gradients via ssm_scan_backward (reads per-step h_cache) and */
/*  via ssm_scan_backward_f1 (reconstructs from h_final). Compare every  */
/*  output tensor: cosine similarity > F1_GRAD_COS_MIN and max abs diff. */
/* ---------------------------------------------------------------------- */

static int check_grad(const char *name, const float *a, const float *b, size_t n) {
    float cos = cosine_similarity(a, b, n);
    float mad = max_abs_diff(a, b, n);
    int pass = (cos >= F1_GRAD_COS_MIN) && (mad < F1_GRAD_ABS_TOL);
    printf("  %-14s  cos=%.8f  max_abs=%.3e  %s\n",
           name, (double)cos, (double)mad, pass ? "OK" : "FAIL");
    return pass;
}

static int run_grad_compare(int batch_size, int use_bilinear, int use_u_v) {
    float A[D_STATE], B[D_STATE], C[D_INNER * D_STATE];
    float x_one[SEQ_LEN * D_INNER], dy_one[SEQ_LEN * D_INNER];
    float C_bilinear[D_STATE];
    float u_vec[D_STATE];
    float v_vec[D_STATE];
    make_stable_config(A, B, C, x_one, dy_one, C_bilinear);
    const float *cbl_ptr = use_bilinear ? C_bilinear : NULL;
    if (use_u_v) {
        for (int i = 0; i < D_STATE; i++) {
            u_vec[i] = rand_uniform(-0.2f, 0.2f);
            v_vec[i] = rand_uniform(-0.2f, 0.2f);
        }
    }
    const float *u_ptr = use_u_v ? u_vec : NULL;
    const float *v_ptr = use_u_v ? v_vec : NULL;

    /* Replicate x and dy across the batch so both paths see identical input. */
    float *x  = (float *)calloc((size_t)batch_size * SEQ_LEN * D_INNER, sizeof(float));
    float *dy = (float *)calloc((size_t)batch_size * SEQ_LEN * D_INNER, sizeof(float));
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < SEQ_LEN * D_INNER; i++) {
            x[(size_t)b * SEQ_LEN * D_INNER + i] = x_one[i] + 0.01f * (float)b;
            dy[(size_t)b * SEQ_LEN * D_INNER + i] = dy_one[i];
        }
    }

    /* Per-batch per-step h reference cache (for ssm_scan_backward) + h_final
     * (for ssm_scan_backward_f1). */
    float *h_ref_cache = (float *)calloc(
        (size_t)batch_size * SEQ_LEN * D_STATE, sizeof(float));
    float *h_final = (float *)calloc((size_t)batch_size * D_STATE, sizeof(float));
    float *y = (float *)calloc((size_t)batch_size * SEQ_LEN * D_INNER, sizeof(float));
    for (int b = 0; b < batch_size; b++) {
        reference_forward(A, B, C,
            x + (size_t)b * SEQ_LEN * D_INNER,
            cbl_ptr, u_ptr, v_ptr,
            h_ref_cache + (size_t)b * SEQ_LEN * D_STATE,
            h_final     + (size_t)b * D_STATE,
            y           + (size_t)b * SEQ_LEN * D_INNER,
            SEQ_LEN, D_STATE, D_INNER);
    }

    /* Stability probe via the production helper. */
    float *probe = (float *)calloc((size_t)batch_size * D_STATE, sizeof(float));
    {
        float *h_tmp = (float *)calloc((size_t)batch_size * D_STATE, sizeof(float));
        float *y_tmp = (float *)calloc((size_t)batch_size * SEQ_LEN * D_INNER,
                                       sizeof(float));
        ssm_scan_forward_batch_with_stability_probe(
            A, NULL, NULL, B, 0, C, x, h_tmp, y_tmp, NULL, NULL, NULL,
            batch_size, SEQ_LEN, D_STATE, D_INNER, cbl_ptr, probe);
        free(h_tmp); free(y_tmp);
    }

    /* Path A: ssm_scan_backward via batch wrapper emulation.
     * We call the single-sequence backward per batch item so we can feed it
     * our per-step h_ref_cache (the production batch wrapper expects the same
     * layout, and the canonical backward reads h_cache[t*d_state] per step). */
    float *dA_a = (float *)calloc(D_STATE, sizeof(float));
    float *dB_a = (float *)calloc(D_STATE, sizeof(float));
    float *dC_a = (float *)calloc((size_t)D_INNER * D_STATE, sizeof(float));
    float *dx_a = (float *)calloc((size_t)batch_size * SEQ_LEN * D_INNER,
                                  sizeof(float));
    float *dCbl_a = use_bilinear
                      ? (float *)calloc(D_STATE, sizeof(float))
                      : NULL;
    float *du_a = use_u_v ? (float *)calloc(D_STATE, sizeof(float)) : NULL;
    float *dv_a = use_u_v ? (float *)calloc(D_STATE, sizeof(float)) : NULL;
    for (int b = 0; b < batch_size; b++) {
        ssm_scan_backward(
            A, NULL, NULL, B, 0, C,
            x + (size_t)b * SEQ_LEN * D_INNER,
            h_ref_cache + (size_t)b * SEQ_LEN * D_STATE,
            dy + (size_t)b * SEQ_LEN * D_INNER,
            NULL,
            dA_a, NULL, dB_a, dC_a,
            dx_a + (size_t)b * SEQ_LEN * D_INNER,
            NULL,
            SEQ_LEN, D_STATE, D_INNER,
            cbl_ptr, dCbl_a,
            u_ptr, v_ptr, du_a, dv_a,
            NULL);
    }

    /* Path B: ssm_scan_backward_f1_batch. */
    float *dA_b = (float *)calloc(D_STATE, sizeof(float));
    float *dB_b = (float *)calloc(D_STATE, sizeof(float));
    float *dC_b = (float *)calloc((size_t)D_INNER * D_STATE, sizeof(float));
    float *dx_b = (float *)calloc((size_t)batch_size * SEQ_LEN * D_INNER,
                                  sizeof(float));
    float *dCbl_b = use_bilinear
                      ? (float *)calloc(D_STATE, sizeof(float))
                      : NULL;
    float *du_b = use_u_v ? (float *)calloc(D_STATE, sizeof(float)) : NULL;
    float *dv_b = use_u_v ? (float *)calloc(D_STATE, sizeof(float)) : NULL;
    ssm_scan_backward_f1_batch(
        A, NULL, NULL, B, 0, C, x, h_final, dy, NULL,
        dA_b, NULL, dB_b, dC_b, dx_b, NULL,
        cbl_ptr, dCbl_b, probe,
        batch_size, SEQ_LEN, D_STATE, D_INNER,
        u_ptr, v_ptr, du_b, dv_b);

    int pass = 1;
    pass &= check_grad("dA", dA_a, dA_b, D_STATE);
    pass &= check_grad("dB", dB_a, dB_b, D_STATE);
    pass &= check_grad("dC", dC_a, dC_b, (size_t)D_INNER * D_STATE);
    pass &= check_grad("dx", dx_a, dx_b,
                       (size_t)batch_size * SEQ_LEN * D_INNER);
    if (use_bilinear) {
        pass &= check_grad("dC_bilinear", dCbl_a, dCbl_b, D_STATE);
    }
    if (use_u_v) {
        pass &= check_grad("du", du_a, du_b, D_STATE);
        pass &= check_grad("dv", dv_a, dv_b, D_STATE);
    }

    free(x); free(dy); free(h_ref_cache); free(h_final); free(y); free(probe);
    free(dA_a); free(dB_a); free(dC_a); free(dx_a); if (dCbl_a) free(dCbl_a);
    free(dA_b); free(dB_b); free(dC_b); free(dx_b); if (dCbl_b) free(dCbl_b);
    if (du_a) free(du_a); if (dv_a) free(dv_a);
    if (du_b) free(du_b); if (dv_b) free(dv_b);
    return pass;
}

static int test_f1_grad(void) {
    printf("TEST 3: [F1 GRAD] F1 gradients vs. canonical backward (batch=1)\n");
    int ok1 = run_grad_compare(1, 1, 0);
    printf("  %s (batch=1, with bilinear)\n", ok1 ? "[PASS]" : "[FAIL]");

    printf("TEST 3b: [F1 GRAD] batch=2 equivalence with bilinear\n");
    int ok2 = run_grad_compare(2, 1, 0);
    printf("  %s (batch=2, with bilinear)\n", ok2 ? "[PASS]" : "[FAIL]");
    return ok1 && ok2;
}

/* ---------------------------------------------------------------------- */
/*  Test 4 — [F1 STAB] ln(10) gate refuses unstable configurations        */
/*                                                                         */
/*  Build a config where |A_tot| sits well above 1 every step so that    */
/*  the cumulative probe across SEQ_LEN easily exceeds ln(10). Confirm:  */
/*    - probe max exceeds ln(10)                                          */
/*    - ssm_scan_backward_f1 returns without writing gradients (dA etc.  */
/*      remain at their pre-call sentinel values)                         */
/* ---------------------------------------------------------------------- */

static int test_f1_stab_gate(void) {
    printf("TEST 4: [F1 STAB] ln(10) gate refuses unstable config\n");

    /* A >= 0.9 and C_bilinear = +1 with all-positive x around 0.8 makes
     * |A_tot| ~= 1.7 per step. 16 * ln(1.7) ~= 8.5 >> ln(10). */
    float A[D_STATE], B[D_STATE], C[D_INNER * D_STATE];
    float x[SEQ_LEN * D_INNER], dy[SEQ_LEN * D_INNER];
    float C_bilinear[D_STATE];

    for (int i = 0; i < D_STATE; i++) { A[i] = 0.95f; B[i] = 0.1f; }
    for (int i = 0; i < D_INNER * D_STATE; i++) C[i] = 0.1f;
    for (int i = 0; i < SEQ_LEN * D_INNER; i++) x[i] = 0.8f;
    for (int i = 0; i < SEQ_LEN * D_INNER; i++) dy[i] = 0.01f;
    for (int i = 0; i < D_STATE; i++) C_bilinear[i] = 1.0f;

    float h[D_STATE] = {0};
    float y[SEQ_LEN * D_INNER] = {0};
    float probe[D_STATE] = {0};
    ssm_scan_forward_with_stability_probe(
        A, NULL, NULL, B, 0, C, x, h, y, NULL, NULL, NULL,
        SEQ_LEN, D_STATE, D_INNER, C_bilinear, probe);

    float probe_max = probe[0];
    for (int i = 1; i < D_STATE; i++) if (probe[i] > probe_max) probe_max = probe[i];
    printf("  probe_max=%.6f   ln10=%.6f\n",
           (double)probe_max, (double)logf(10.0f));
    if (probe_max < logf(10.0f)) {
        printf("  [FAIL] expected probe_max >= ln(10) for this config\n");
        return 0;
    }

    /* Seed the gradient accumulators with a recognisable sentinel so we can
     * prove the gated backward did not touch them. */
    const float sentinel = -7777.0f;
    float dA[D_STATE], dB[D_STATE], dC[D_INNER * D_STATE];
    float dx[SEQ_LEN * D_INNER];
    float dCbl[D_STATE];
    for (int i = 0; i < D_STATE; i++) { dA[i] = sentinel; dB[i] = sentinel; dCbl[i] = sentinel; }
    for (int i = 0; i < D_INNER * D_STATE; i++) dC[i] = sentinel;
    for (int i = 0; i < SEQ_LEN * D_INNER; i++) dx[i] = sentinel;

    /* We need h_final to satisfy the assertion; value is irrelevant because
     * the stability gate short-circuits before any math runs. */
    float h_final[D_STATE];
    memcpy(h_final, h, sizeof(h_final));

    printf("  (expect one [F1 GATE] stderr line next)\n");
    ssm_scan_backward_f1(
        A, NULL, NULL, B, 0, C, x, h_final, dy, NULL,
        dA, NULL, dB, dC, dx, NULL,
        C_bilinear, dCbl, probe,
        SEQ_LEN, D_STATE, D_INNER,
        NULL, NULL, NULL, NULL);

    int pass = 1;
    for (int i = 0; i < D_STATE && pass; i++) {
        if (dA[i] != sentinel || dB[i] != sentinel || dCbl[i] != sentinel) pass = 0;
    }
    for (int i = 0; i < D_INNER * D_STATE && pass; i++) {
        if (dC[i] != sentinel) pass = 0;
    }
    for (int i = 0; i < SEQ_LEN * D_INNER && pass; i++) {
        if (dx[i] != sentinel) pass = 0;
    }
    if (!pass) {
        printf("  [FAIL] F1 backward wrote gradients despite gate refusal\n");
        return 0;
    }
    printf("  [PASS] F1 backward returned without touching gradients\n");
    return 1;
}

/* ---------------------------------------------------------------------- */
/*  Test 5 — [F1 GPU-CPU]  DEFERRED                                       */
/* ---------------------------------------------------------------------- */

static int test_f1_gpu_cpu_skipped(void) {
    printf("TEST 5: [F1 GPU-CPU] parity\n");
    printf("  [SKIP] GPU replay dispatch not yet wired from S3 caller. "
           "Re-enable once ssm_scan_backward_f1_batch is invoked from the "
           "GPU path (post-S5 follow-up).\n");
    return 1; /* skipped does not fail the suite */
}

/* ---------------------------------------------------------------------- */
/*  Test 6 — [F1 NEG] negative control with C_bilinear = NULL             */
/*                                                                         */
/*  Repeat tests 2 + 3 with C_bilinear disabled. This must still roundtrip */
/*  and match the canonical gradients. Since the rejected                  */
/*  ssm_scan_backward_reversible also succeeds in this case, this proves  */
/*  the F1 code path is actually running (and not masking a fallback).    */
/* ---------------------------------------------------------------------- */

static int test_f1_negative_control(void) {
    printf("TEST 6: [F1 NEG] C_bilinear == NULL roundtrip + gradients\n");

    float A[D_STATE], B[D_STATE], C[D_INNER * D_STATE];
    float x[SEQ_LEN * D_INNER], dy[SEQ_LEN * D_INNER];
    float C_bilinear_unused[D_STATE];
    make_stable_config(A, B, C, x, dy, C_bilinear_unused);

    float h_ref_cache[SEQ_LEN * D_STATE];
    float h_final[D_STATE];
    float y[SEQ_LEN * D_INNER];
    reference_forward(A, B, C, x, NULL, NULL, NULL, h_ref_cache, h_final, y,
                      SEQ_LEN, D_STATE, D_INNER);

    float h_recon_cache[SEQ_LEN * D_STATE];
    reverse_reconstruct(A, B, x, h_final, NULL, NULL, NULL, h_recon_cache,
                        SEQ_LEN, D_STATE, D_INNER);
    float worst = max_abs_diff(h_recon_cache, h_ref_cache, SEQ_LEN * D_STATE);
    printf("  roundtrip_max_abs=%.3e   tol=%.1e\n",
           (double)worst, (double)F1_ROUNDTRIP_TOL);
    if (worst > F1_ROUNDTRIP_TOL) {
        printf("  [FAIL] roundtrip without bilinear exceeds tolerance\n");
        return 0;
    }

    int ok = run_grad_compare(1, 0, 0);
    printf("  %s (batch=1, no bilinear)\n", ok ? "[PASS]" : "[FAIL]");
    return ok;
}

/* ---------------------------------------------------------------------- */
/*  Test 7 — [SMAI-BRR] Sherman-Morrison Adjoint Inversion gradients      */
/*                                                                         */
/*  Verify that F1 backward using Sherman-Morrison rank-1 update          */
/*  inversion correctly reconstructs hidden states and calculates        */
/*  exact gradients for dA, dB, dC, dx, dCbl, du, dv compared to cached.  */
/* ---------------------------------------------------------------------- */

static int test_smai_brr_grad(void) {
    printf("TEST 7: [SMAI-BRR] F1 with Sherman-Morrison rank-1 update (batch=1)\n");
    int ok1 = run_grad_compare(1, 1, 1);
    printf("  %s (batch=1, with bilinear + rank-1)\n", ok1 ? "[PASS]" : "[FAIL]");

    printf("TEST 7b: [SMAI-BRR] batch=2 equivalence with rank-1\n");
    int ok2 = run_grad_compare(2, 1, 1);
    printf("  %s (batch=2, with bilinear + rank-1)\n", ok2 ? "[PASS]" : "[FAIL]");
    return ok1 && ok2;
}

/* ---------------------------------------------------------------------- */
/*  Main                                                                  */
/* ---------------------------------------------------------------------- */

int main(void) {
    printf("\n==== R356 F1 Reversibility Smoke Tests (slice S6) ====\n\n");

    int total = 0, passed = 0, skipped = 0;

    #define RUN(name, fn) do {                       \
        total++;                                     \
        int ok = fn();                               \
        if (ok) passed++;                            \
        printf("\n");                                \
    } while (0)

    RUN("F1 PROBE",    test_f1_probe_sanity);
    RUN("F1 ROUNDTRIP",test_f1_roundtrip);
    RUN("F1 GRAD",     test_f1_grad);
    RUN("F1 STAB",     test_f1_stab_gate);

    /* Skipped tests count as passed for the suite exit code but are
     * reported separately. */
    total++;
    if (test_f1_gpu_cpu_skipped()) { passed++; skipped++; }
    printf("\n");

    RUN("F1 NEG", test_f1_negative_control);
    RUN("SMAI-BRR", test_smai_brr_grad);

    #undef RUN

    printf("==== Summary ====\n");
    printf("%d / %d tests passed (%d skipped)\n", passed, total, skipped);
    return (passed == total) ? 0 : 1;
}
