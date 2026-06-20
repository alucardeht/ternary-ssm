/*
 * test_r357_alt3_residual_shell.c — R357 Alt 3 Residual Shell smoke tests
 *                                   (slice E)
 *
 * CPU-only standalone exercises of the masked CE backward path added in
 * slice B of R357 Alt 3. Uses the non-sampled full-softmax path
 * (compute_ce_loss / compute_ce_backward and their _masked / _with_per_token
 * siblings in src/core/loss.c) on tiny toy configurations. Target runtime
 * well below one second. Prints [RS Cn] [PASS|FAIL] <note> per case and
 * exits 0 iff every required case passes.
 *
 * Required cases (handoff §8, restated for this slice):
 *
 *   [RS C1] mask=1 everywhere with inv_S = 1/n_tokens produces gradients
 *           that are bit-identical (memcmp == 0) to the baseline
 *           compute_ce_backward on the same inputs.
 *   [RS C2] mask is all zero except one token and inv_S = 1.0. The
 *           surviving row must equal (softmax(row) - one_hot(target)) and
 *           every other row must be exactly zero.
 *   [RS C3] Half-and-half batch: two sequences, mask keeps half of the
 *           tokens globally. S == n_kept. Each surviving row must equal
 *           (softmax - one_hot) * (1 / n_kept), and masked rows must be
 *           zero. This confirms the normalization is global across the
 *           (b, t) batch, not per-sequence.
 *   [RS C4] Edge case: S == 0 after thresholding (every token outside the
 *           band). The caller is expected to fall back to all-ones with
 *           S = n_valid and log [RS] fallback. This test emulates the
 *           caller-side fallback (since the mask/S computation lives in
 *           the caller, not in loss.c) and confirms that the backward path
 *           does not divide by zero and produces gradients identical to
 *           the unmasked baseline.
 *   [RS C5] Grad-accum independence: call the masked backward twice on
 *           two separate micro-batches with different masks and different
 *           inv_S values, and confirm each call produces the mask-and-
 *           inv_S-consistent gradients without leaking state from the
 *           previous call. The two micro-batches use different token
 *           distributions and different inv_S values; neither result
 *           depends on the other.
 *
 * Build (standalone, from src/model_v12):
 *   gcc -O2 -Wall -I src tests/test_r357_alt3_residual_shell.c \
 *       src/core/loss.o -lm -o test_r357_alt3.exe
 *
 * Or via the Makefile target:
 *   make test_r357_alt3
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "../src/core/loss.h"
#include "../src/core/types.h"

/* ----- Tiny toy dimensions ------------------------------------------- */
#define RS_SEQ_LEN   16
#define RS_VOCAB     32
#define RS_BATCH     2
#define RS_N_TOKENS  (RS_SEQ_LEN - 1)

/* ----- Tolerances ---------------------------------------------------- */
/* C1 is bit-identical by construction (same arithmetic in the same
 * order with inv_S = 1/n_tokens) — we compare via memcmp. C2..C5 use
 * a small absolute tolerance to absorb compiler-level fp reorderings. */
#define RS_BIT_TOL    0.0f
#define RS_ABS_TOL    1e-6f

/* ---------------------------------------------------------------------- */
/*  Deterministic pseudo-random helpers                                   */
/* ---------------------------------------------------------------------- */

static uint32_t rs_rng_state = 0xC0FFEE00u;

static void rs_seed(uint32_t s) { rs_rng_state = s ? s : 0xC0FFEE00u; }

static float rs_uniform(float lo, float hi) {
    uint32_t x = rs_rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rs_rng_state = x;
    float u = (float)(x & 0xFFFFFF) / (float)0x1000000;
    return lo + (hi - lo) * u;
}

static void fill_random_logits(float *logits, int seq_len, int vocab) {
    for (int i = 0; i < seq_len * vocab; i++) {
        logits[i] = rs_uniform(-2.0f, 2.0f);
    }
}

static void fill_random_targets(uint16_t *targets, int seq_len, int vocab) {
    for (int i = 0; i < seq_len; i++) {
        /* use the same rng to keep reproducibility */
        float u = rs_uniform(0.0f, 1.0f);
        int   v = (int)(u * (float)vocab);
        if (v < 0) v = 0;
        if (v >= vocab) v = vocab - 1;
        targets[i] = (uint16_t)v;
    }
}

/* ---------------------------------------------------------------------- */
/*  Minimal Cache construction for the tests.                             */
/*                                                                         */
/*  compute_ce_loss / compute_ce_backward / compute_ce_backward_masked    */
/*  only touch cache->max_logit (read/write) and cache->d_logits (write). */
/*  Every other Cache field is left NULL/zero. We allocate d_logits at   */
/*  [seq_len × vocab_size] rather than the production [seq_len ×          */
/*  V12_SS_N_SAMPLES] because the full-softmax backward path writes a    */
/*  vocab-wide row per position.                                          */
/* ---------------------------------------------------------------------- */

static void make_test_cache(Cache *cache, float *d_logits_buf,
                            float *max_logit_buf) {
    memset(cache, 0, sizeof(*cache));
    cache->d_logits  = d_logits_buf;
    cache->max_logit = max_logit_buf;
}

/* ---------------------------------------------------------------------- */
/*  Softmax(row) - one_hot(target) reference, scaled by `scale`.         */
/* ---------------------------------------------------------------------- */

static void reference_grad_row(const float *row, int vocab, uint16_t target,
                               float scale, float *out) {
    float max_v = row[0];
    for (int v = 1; v < vocab; v++) if (row[v] > max_v) max_v = row[v];
    float sum_exp = 0.0f;
    for (int v = 0; v < vocab; v++) sum_exp += expf(row[v] - max_v);
    const float inv_sum = scale / sum_exp;
    for (int v = 0; v < vocab; v++) out[v] = expf(row[v] - max_v) * inv_sum;
    out[target] -= scale;
}

static float row_max_abs_diff(const float *a, const float *b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static int row_is_zero(const float *a, int n) {
    for (int i = 0; i < n; i++) if (a[i] != 0.0f) return 0;
    return 1;
}

/* ====================================================================== */
/*  [RS C1] mask=1 everywhere → bit-identical to baseline backward        */
/* ====================================================================== */

static int test_rs_c1(void) {
    printf("[RS C1] mask=1 everywhere vs baseline compute_ce_backward\n");

    float  logits[RS_SEQ_LEN * RS_VOCAB];
    uint16_t targets[RS_SEQ_LEN];
    rs_seed(0xC1C1C1C1u);
    fill_random_logits(logits, RS_SEQ_LEN, RS_VOCAB);
    fill_random_targets(targets, RS_SEQ_LEN, RS_VOCAB);

    float d_logits_base[RS_SEQ_LEN * RS_VOCAB] = {0};
    float d_logits_msk [RS_SEQ_LEN * RS_VOCAB] = {0};
    float max_logit_base[RS_SEQ_LEN] = {0};
    float max_logit_msk [RS_SEQ_LEN] = {0};

    Cache cache_base, cache_msk;
    make_test_cache(&cache_base, d_logits_base, max_logit_base);
    make_test_cache(&cache_msk,  d_logits_msk,  max_logit_msk);

    (void)compute_ce_loss(&cache_base, logits, targets, RS_VOCAB, RS_SEQ_LEN);
    compute_ce_backward(&cache_base, logits, targets, RS_VOCAB, RS_SEQ_LEN);

    /* all-ones mask, inv_S = 1 / (seq_len - 1) to mirror the baseline */
    uint8_t mask[RS_N_TOKENS];
    for (int i = 0; i < RS_N_TOKENS; i++) mask[i] = 1;
    const float inv_S = 1.0f / (float)RS_N_TOKENS;

    (void)compute_ce_loss(&cache_msk, logits, targets, RS_VOCAB, RS_SEQ_LEN);
    compute_ce_backward_masked(&cache_msk, logits, targets,
                               RS_VOCAB, RS_SEQ_LEN, mask, inv_S);

    if (memcmp(d_logits_base, d_logits_msk, sizeof(d_logits_base)) != 0) {
        /* fallback to absolute tolerance report for debugging */
        float mad = row_max_abs_diff(d_logits_base, d_logits_msk,
                                     RS_SEQ_LEN * RS_VOCAB);
        printf("[RS C1] [FAIL] not bit-identical (max_abs=%.3e)\n",
               (double)mad);
        return 0;
    }
    printf("[RS C1] [PASS] memcmp == 0 over %d elements\n",
           RS_SEQ_LEN * RS_VOCAB);
    return 1;
}

/* ====================================================================== */
/*  [RS C2] single surviving token with inv_S = 1.0                       */
/* ====================================================================== */

static int test_rs_c2(void) {
    printf("[RS C2] single-token mask with S=1, inv_S=1.0\n");

    float  logits[RS_SEQ_LEN * RS_VOCAB];
    uint16_t targets[RS_SEQ_LEN];
    rs_seed(0xC2C2C2C2u);
    fill_random_logits(logits, RS_SEQ_LEN, RS_VOCAB);
    fill_random_targets(targets, RS_SEQ_LEN, RS_VOCAB);

    float d_logits[RS_SEQ_LEN * RS_VOCAB] = {0};
    float max_logit[RS_SEQ_LEN] = {0};

    Cache cache;
    make_test_cache(&cache, d_logits, max_logit);

    (void)compute_ce_loss(&cache, logits, targets, RS_VOCAB, RS_SEQ_LEN);

    uint8_t mask[RS_N_TOKENS];
    memset(mask, 0, sizeof(mask));
    const int kept = 5;  /* arbitrary surviving position */
    mask[kept] = 1;
    const float inv_S = 1.0f;

    compute_ce_backward_masked(&cache, logits, targets,
                               RS_VOCAB, RS_SEQ_LEN, mask, inv_S);

    /* Expected surviving row: softmax(row[kept]) - one_hot(target[kept+1]). */
    float expected[RS_VOCAB];
    reference_grad_row(logits + (size_t)kept * RS_VOCAB, RS_VOCAB,
                       targets[kept + 1], 1.0f, expected);
    float mad = row_max_abs_diff(d_logits + (size_t)kept * RS_VOCAB,
                                 expected, RS_VOCAB);
    if (mad > RS_ABS_TOL) {
        printf("[RS C2] [FAIL] surviving row mismatch (max_abs=%.3e)\n",
               (double)mad);
        return 0;
    }

    /* Every other row must be identically zero (including seq_len-1). */
    for (int t = 0; t < RS_SEQ_LEN; t++) {
        if (t == kept) continue;
        if (!row_is_zero(d_logits + (size_t)t * RS_VOCAB, RS_VOCAB)) {
            printf("[RS C2] [FAIL] masked row t=%d not zero\n", t);
            return 0;
        }
    }
    printf("[RS C2] [PASS] surviving row max_abs=%.3e, masked rows zeroed\n",
           (double)mad);
    return 1;
}

/* ====================================================================== */
/*  [RS C3] half-and-half batch with global inv_S = 1 / n_kept            */
/* ====================================================================== */

static int test_rs_c3(void) {
    printf("[RS C3] half-and-half batch, global S = n_kept\n");

    /* Two independent sequences ("batch=2"). */
    float  logits_b0[RS_SEQ_LEN * RS_VOCAB];
    float  logits_b1[RS_SEQ_LEN * RS_VOCAB];
    uint16_t targets_b0[RS_SEQ_LEN];
    uint16_t targets_b1[RS_SEQ_LEN];
    rs_seed(0xC3C3C3C3u);
    fill_random_logits(logits_b0, RS_SEQ_LEN, RS_VOCAB);
    fill_random_targets(targets_b0, RS_SEQ_LEN, RS_VOCAB);
    fill_random_logits(logits_b1, RS_SEQ_LEN, RS_VOCAB);
    fill_random_targets(targets_b1, RS_SEQ_LEN, RS_VOCAB);

    float d_logits_b0[RS_SEQ_LEN * RS_VOCAB] = {0};
    float d_logits_b1[RS_SEQ_LEN * RS_VOCAB] = {0};
    float max_logit_b0[RS_SEQ_LEN] = {0};
    float max_logit_b1[RS_SEQ_LEN] = {0};

    Cache cache_b0, cache_b1;
    make_test_cache(&cache_b0, d_logits_b0, max_logit_b0);
    make_test_cache(&cache_b1, d_logits_b1, max_logit_b1);

    /* Masks: keep every other position globally. Concretely, b=0 keeps
     * even t, b=1 keeps odd t, over t=0..n_tokens-1. */
    uint8_t mask_b0[RS_N_TOKENS];
    uint8_t mask_b1[RS_N_TOKENS];
    int n_kept = 0;
    for (int t = 0; t < RS_N_TOKENS; t++) {
        mask_b0[t] = (t % 2 == 0) ? 1 : 0;
        mask_b1[t] = (t % 2 == 1) ? 1 : 0;
        n_kept += mask_b0[t] + mask_b1[t];
    }
    /* Sanity: across both sequences, exactly n_tokens survive (half each). */
    const int expected_kept = RS_N_TOKENS;
    if (n_kept != expected_kept) {
        printf("[RS C3] [FAIL] setup error: n_kept=%d expected=%d\n",
               n_kept, expected_kept);
        return 0;
    }
    const float inv_S = 1.0f / (float)n_kept;

    /* Forward on both sequences to populate max_logit. */
    (void)compute_ce_loss(&cache_b0, logits_b0, targets_b0,
                          RS_VOCAB, RS_SEQ_LEN);
    (void)compute_ce_loss(&cache_b1, logits_b1, targets_b1,
                          RS_VOCAB, RS_SEQ_LEN);

    /* Masked backward on both. */
    compute_ce_backward_masked(&cache_b0, logits_b0, targets_b0,
                               RS_VOCAB, RS_SEQ_LEN, mask_b0, inv_S);
    compute_ce_backward_masked(&cache_b1, logits_b1, targets_b1,
                               RS_VOCAB, RS_SEQ_LEN, mask_b1, inv_S);

    /* Verify every row of both sequences. */
    float worst = 0.0f;
    for (int sequence = 0; sequence < 2; sequence++) {
        const float *logits = (sequence == 0) ? logits_b0 : logits_b1;
        const uint16_t *tgt = (sequence == 0) ? targets_b0 : targets_b1;
        const uint8_t *mask = (sequence == 0) ? mask_b0 : mask_b1;
        const float *d     = (sequence == 0) ? d_logits_b0 : d_logits_b1;

        for (int t = 0; t < RS_N_TOKENS; t++) {
            const float *d_row = d + (size_t)t * RS_VOCAB;
            if (mask[t]) {
                float expected[RS_VOCAB];
                reference_grad_row(logits + (size_t)t * RS_VOCAB, RS_VOCAB,
                                   tgt[t + 1], inv_S, expected);
                float mad = row_max_abs_diff(d_row, expected, RS_VOCAB);
                if (mad > worst) worst = mad;
            } else {
                if (!row_is_zero(d_row, RS_VOCAB)) {
                    printf("[RS C3] [FAIL] masked row (b=%d,t=%d) not zero\n",
                           sequence, t);
                    return 0;
                }
            }
        }
        /* Last position (t = seq_len-1) must always be zero. */
        if (!row_is_zero(d + (size_t)RS_N_TOKENS * RS_VOCAB, RS_VOCAB)) {
            printf("[RS C3] [FAIL] tail row (b=%d) not zero\n", sequence);
            return 0;
        }
    }
    if (worst > RS_ABS_TOL) {
        printf("[RS C3] [FAIL] surviving rows max_abs=%.3e > tol=%.1e\n",
               (double)worst, (double)RS_ABS_TOL);
        return 0;
    }
    printf("[RS C3] [PASS] n_kept=%d inv_S=%.6f worst_row_max_abs=%.3e\n",
           n_kept, (double)inv_S, (double)worst);
    return 1;
}

/* ====================================================================== */
/*  [RS C4] S == 0 fallback path — caller forces mask to all ones        */
/* ====================================================================== */

static int test_rs_c4(void) {
    printf("[RS C4] S==0 fallback: caller forces mask=1, inv_S=1/n_valid\n");

    float  logits[RS_SEQ_LEN * RS_VOCAB];
    uint16_t targets[RS_SEQ_LEN];
    rs_seed(0xC4C4C4C4u);
    fill_random_logits(logits, RS_SEQ_LEN, RS_VOCAB);
    fill_random_targets(targets, RS_SEQ_LEN, RS_VOCAB);

    /* Compute per-token loss and pretend every token is out of the band.
     * The caller code (gpu_pipeline / CPU fallback) then rewrites the mask
     * to all ones and sets S = n_valid. We mirror that here. */
    float per_token[RS_N_TOKENS] = {0};
    float d_logits_base[RS_SEQ_LEN * RS_VOCAB] = {0};
    float d_logits_fbk [RS_SEQ_LEN * RS_VOCAB] = {0};
    float max_logit_base[RS_SEQ_LEN] = {0};
    float max_logit_fbk [RS_SEQ_LEN] = {0};

    Cache cache_base, cache_fbk;
    make_test_cache(&cache_base, d_logits_base, max_logit_base);
    make_test_cache(&cache_fbk,  d_logits_fbk,  max_logit_fbk);

    /* Forward (also exporting per-token losses for the simulated gate). */
    (void)compute_ce_loss_with_per_token(&cache_fbk, logits, targets,
                                         RS_VOCAB, RS_SEQ_LEN, per_token);

    /* Simulated raw mask = 0 everywhere. */
    uint8_t mask[RS_N_TOKENS];
    memset(mask, 0, sizeof(mask));
    int S = 0;
    for (int t = 0; t < RS_N_TOKENS; t++) S += mask[t];

    /* Caller-side fallback: if S == 0, rewrite mask=1 and S = n_valid. */
    int fallback_triggered = 0;
    if (S == 0) {
        fallback_triggered = 1;
        for (int t = 0; t < RS_N_TOKENS; t++) mask[t] = 1;
        S = RS_N_TOKENS;
    }
    if (!fallback_triggered) {
        printf("[RS C4] [FAIL] fallback did not trigger as expected\n");
        return 0;
    }
    const float inv_S = 1.0f / (float)S;

    compute_ce_backward_masked(&cache_fbk, logits, targets,
                               RS_VOCAB, RS_SEQ_LEN, mask, inv_S);

    /* Unmasked baseline for comparison. */
    (void)compute_ce_loss(&cache_base, logits, targets, RS_VOCAB, RS_SEQ_LEN);
    compute_ce_backward(&cache_base, logits, targets, RS_VOCAB, RS_SEQ_LEN);

    /* The fallback path should match the baseline bit-for-bit because
     * inv_S = 1/n_tokens = 1/(seq_len-1). */
    if (memcmp(d_logits_base, d_logits_fbk, sizeof(d_logits_base)) != 0) {
        float mad = row_max_abs_diff(d_logits_base, d_logits_fbk,
                                     RS_SEQ_LEN * RS_VOCAB);
        printf("[RS C4] [FAIL] fallback grad not bit-identical "
               "to baseline (max_abs=%.3e)\n", (double)mad);
        return 0;
    }
    /* Divide-by-zero guard: inv_S must be finite. */
    if (!isfinite(inv_S) || inv_S <= 0.0f) {
        printf("[RS C4] [FAIL] inv_S=%g is not a safe positive finite\n",
               (double)inv_S);
        return 0;
    }
    printf("[RS C4] [PASS] fallback triggered, S=%d, inv_S=%.6f, "
           "bit-identical to baseline\n", S, (double)inv_S);
    (void)per_token;  /* per-token values only used for the simulated gate */
    return 1;
}

/* ====================================================================== */
/*  [RS C5] grad-accum independence across two micro-batches              */
/* ====================================================================== */

static int test_rs_c5(void) {
    printf("[RS C5] grad-accum independence: two masks, two inv_S values\n");

    /* Micro-batch 1 */
    float logits1[RS_SEQ_LEN * RS_VOCAB];
    uint16_t targets1[RS_SEQ_LEN];
    rs_seed(0xC5AAAAAAu);
    fill_random_logits(logits1, RS_SEQ_LEN, RS_VOCAB);
    fill_random_targets(targets1, RS_SEQ_LEN, RS_VOCAB);

    /* Micro-batch 2 — intentionally different distribution */
    float logits2[RS_SEQ_LEN * RS_VOCAB];
    uint16_t targets2[RS_SEQ_LEN];
    rs_seed(0xC5BBBBBBu);
    fill_random_logits(logits2, RS_SEQ_LEN, RS_VOCAB);
    fill_random_targets(targets2, RS_SEQ_LEN, RS_VOCAB);

    /* Two masks with different shapes and different S counts. */
    uint8_t mask1[RS_N_TOKENS];
    uint8_t mask2[RS_N_TOKENS];
    int S1 = 0, S2 = 0;
    for (int t = 0; t < RS_N_TOKENS; t++) {
        mask1[t] = (t < 3) ? 1 : 0;        /* S1 = 3 */
        mask2[t] = ((t % 2) == 0) ? 1 : 0; /* S2 = ceil(n_tokens/2) */
        S1 += mask1[t];
        S2 += mask2[t];
    }
    const float inv_S1 = 1.0f / (float)S1;
    const float inv_S2 = 1.0f / (float)S2;

    /* Run the two micro-batches through the same Cache object in sequence
     * to exercise the "shared state?" question: the mask backward must
     * overwrite d_logits rather than accumulate into it, so the second
     * call's result must depend only on its own inputs. */
    float d_logits[RS_SEQ_LEN * RS_VOCAB] = {0};
    float max_logit[RS_SEQ_LEN] = {0};
    Cache cache;
    make_test_cache(&cache, d_logits, max_logit);

    /* Micro-batch 1 */
    (void)compute_ce_loss(&cache, logits1, targets1, RS_VOCAB, RS_SEQ_LEN);
    compute_ce_backward_masked(&cache, logits1, targets1,
                               RS_VOCAB, RS_SEQ_LEN, mask1, inv_S1);

    /* Snapshot and independently verify micro-batch 1's gradients. */
    float d1_snapshot[RS_SEQ_LEN * RS_VOCAB];
    memcpy(d1_snapshot, d_logits, sizeof(d1_snapshot));

    float worst1 = 0.0f;
    for (int t = 0; t < RS_N_TOKENS; t++) {
        const float *d_row = d1_snapshot + (size_t)t * RS_VOCAB;
        if (mask1[t]) {
            float expected[RS_VOCAB];
            reference_grad_row(logits1 + (size_t)t * RS_VOCAB, RS_VOCAB,
                               targets1[t + 1], inv_S1, expected);
            float mad = row_max_abs_diff(d_row, expected, RS_VOCAB);
            if (mad > worst1) worst1 = mad;
        } else if (!row_is_zero(d_row, RS_VOCAB)) {
            printf("[RS C5] [FAIL] mb1 masked row t=%d not zero\n", t);
            return 0;
        }
    }

    /* Micro-batch 2 — same cache, different mask / inv_S. */
    (void)compute_ce_loss(&cache, logits2, targets2, RS_VOCAB, RS_SEQ_LEN);
    compute_ce_backward_masked(&cache, logits2, targets2,
                               RS_VOCAB, RS_SEQ_LEN, mask2, inv_S2);

    float worst2 = 0.0f;
    for (int t = 0; t < RS_N_TOKENS; t++) {
        const float *d_row = d_logits + (size_t)t * RS_VOCAB;
        if (mask2[t]) {
            float expected[RS_VOCAB];
            reference_grad_row(logits2 + (size_t)t * RS_VOCAB, RS_VOCAB,
                               targets2[t + 1], inv_S2, expected);
            float mad = row_max_abs_diff(d_row, expected, RS_VOCAB);
            if (mad > worst2) worst2 = mad;
        } else if (!row_is_zero(d_row, RS_VOCAB)) {
            printf("[RS C5] [FAIL] mb2 masked row t=%d not zero\n", t);
            return 0;
        }
    }

    /* Now recompute micro-batch 1 in isolation (fresh cache) and confirm
     * its gradient matches the earlier snapshot — i.e. the mb2 call did
     * not retroactively corrupt anything, and the snapshot itself is
     * independent of any cross-micro-batch state. */
    float d_logits_iso[RS_SEQ_LEN * RS_VOCAB] = {0};
    float max_logit_iso[RS_SEQ_LEN] = {0};
    Cache cache_iso;
    make_test_cache(&cache_iso, d_logits_iso, max_logit_iso);
    (void)compute_ce_loss(&cache_iso, logits1, targets1,
                          RS_VOCAB, RS_SEQ_LEN);
    compute_ce_backward_masked(&cache_iso, logits1, targets1,
                               RS_VOCAB, RS_SEQ_LEN, mask1, inv_S1);

    if (memcmp(d_logits_iso, d1_snapshot, sizeof(d1_snapshot)) != 0) {
        float mad = row_max_abs_diff(d_logits_iso, d1_snapshot,
                                     RS_SEQ_LEN * RS_VOCAB);
        printf("[RS C5] [FAIL] mb1 is not independent of the earlier run "
               "(max_abs=%.3e)\n", (double)mad);
        return 0;
    }

    if (worst1 > RS_ABS_TOL || worst2 > RS_ABS_TOL) {
        printf("[RS C5] [FAIL] per-mb surviving rows too far "
               "(worst1=%.3e worst2=%.3e tol=%.1e)\n",
               (double)worst1, (double)worst2, (double)RS_ABS_TOL);
        return 0;
    }
    printf("[RS C5] [PASS] S1=%d inv_S1=%.6f worst1=%.3e / "
           "S2=%d inv_S2=%.6f worst2=%.3e, independent\n",
           S1, (double)inv_S1, (double)worst1,
           S2, (double)inv_S2, (double)worst2);
    return 1;
}

/* ====================================================================== */
/*  Main                                                                  */
/* ====================================================================== */

int main(void) {
    printf("\n==== R357 Alt 3 Residual Shell Smoke Tests (slice E) ====\n\n");

    int total = 0, passed = 0;

    #define RUN(fn) do { total++; if (fn()) passed++; printf("\n"); } while (0)

    RUN(test_rs_c1);
    RUN(test_rs_c2);
    RUN(test_rs_c3);
    RUN(test_rs_c4);
    RUN(test_rs_c5);

    #undef RUN

    printf("==== Summary ====\n");
    printf("%d / %d cases passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
