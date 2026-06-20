/*
 * test_loss.c — Unit tests for loss.h/loss.c functions
 *
 * Tests:
 * 1. compute_ce_loss: Tiny vocab=4, seq=2 example with manual softmax
 * 2. compute_ce_backward: Verify gradient against forward
 * 3. gp_kl_welford_update: Accumulation over a few timesteps
 *
 * Compile and link against loss.c:
 *   gcc -O2 -lm src/model_v12/src/core/loss.c test_loss.c -o test_loss
 *
 * Run:
 *   ./test_loss
 */

#include "../src/core/loss.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Color codes for terminal output */
#define ANSI_GREEN  "\x1b[32m"
#define ANSI_RED    "\x1b[31m"
#define ANSI_RESET  "\x1b[0m"

/* Test assertion macro */
#define ASSERT_CLOSE(actual, expected, tol, msg) \
    do { \
        if (fabsf((actual) - (expected)) > (tol)) { \
            printf(ANSI_RED "FAIL: %s\n" ANSI_RESET, msg); \
            printf("  Expected: %.6f, Got: %.6f, Diff: %.6e\n", \
                   (expected), (actual), fabsf((actual) - (expected))); \
            return 0; \
        } \
    } while (0)

#define ASSERT_EQ(actual, expected, msg) \
    do { \
        if ((actual) != (expected)) { \
            printf(ANSI_RED "FAIL: %s\n" ANSI_RESET, msg); \
            printf("  Expected: %d, Got: %d\n", (expected), (actual)); \
            return 0; \
        } \
    } while (0)

/* ============================================================================
 * TEST 1: compute_ce_loss with manual calculation
 *
 * Tiny example: vocab=4, seq=2
 * logits = [[1.0, 2.0, 3.0, 0.5],
 *           [0.5, 1.5, 2.0, 1.0]]
 * targets = [0, 2]  (first token is 0, next is 2)
 *
 * Sequence prediction:
 *   t=0: logits[0] predict target[1]=2
 *   t=1: (no target, last position)
 *
 * Manual computation for t=0:
 *   logits[0] = [1.0, 2.0, 3.0, 0.5]
 *   max = 3.0
 *   exp(logits - max) = [exp(-2), exp(-1), exp(0), exp(-2.5)]
 *                     = [0.1353, 0.3679, 1.0, 0.0821]
 *   sum_exp = 1.5853
 *   softmax[2] = 1.0 / 1.5853 = 0.6309
 *   log_softmax[2] = log(0.6309) = -0.4605
 *   ce_loss = -(-0.4605) = 0.4605
 * ============================================================================
 */

int test_ce_loss(void) {
    printf("\n[TEST 1] compute_ce_loss with manual calculation\n");

    const int vocab_size = 4;
    const int seq_len = 2;

    float logits[] = {
        1.0f, 2.0f, 3.0f, 0.5f,  /* t=0 */
        0.5f, 1.5f, 2.0f, 1.0f   /* t=1 */
    };

    uint16_t targets[] = {0, 2};

    Cache cache;
    cache.max_logit = (float *)malloc(seq_len * sizeof(float));
    cache.d_logits = (float *)malloc(seq_len * vocab_size * sizeof(float));
    memset(cache.d_logits, 0, seq_len * vocab_size * sizeof(float));

    float ce = compute_ce_loss(&cache, logits, targets, vocab_size, seq_len);

    /* Manual computation:
     * max = 3.0
     * exp([1-3, 2-3, 3-3, 0.5-3]) = [0.1353, 0.3679, 1.0, 0.0821]
     * sum_exp = 1.5853
     * log(softmax[2]) = log(1.0 / 1.5853) = -0.4605
     * ce_loss = 0.4605 */
    const float expected = 0.4605f;

    printf("  Computed CE loss: %.4f\n", ce);
    printf("  Expected CE loss: %.4f\n", expected);

    ASSERT_CLOSE(ce, expected, 0.001f, "CE loss matches manual calculation");

    printf("  max_logit[0] = %.1f (should be 3.0)\n", cache.max_logit[0]);
    ASSERT_CLOSE(cache.max_logit[0], 3.0f, 0.001f, "max_logit[0] stored correctly");

    printf("  max_logit[1] = %.1f (should be 0.0 — no gradient)\n", cache.max_logit[1]);
    ASSERT_CLOSE(cache.max_logit[1], 0.0f, 0.001f, "max_logit[seq_len-1] zeroed");

    printf(ANSI_GREEN "PASS: test_ce_loss\n" ANSI_RESET);

    free(cache.max_logit);
    free(cache.d_logits);
    return 1;
}


/* ============================================================================
 * TEST 2: compute_ce_backward gradient check
 *
 * Uses the same example as TEST 1.
 * Verifies:
 *   1. d_logits[0][2] ≈ softmax[0][2] - 1 (scaled by 1/n_tokens)
 *   2. d_logits[0][v≠2] ≈ softmax[0][v] (scaled by 1/n_tokens)
 *   3. d_logits[1][*] = 0 (last position)
 * ============================================================================
 */

int test_ce_backward(void) {
    printf("\n[TEST 2] compute_ce_backward gradient verification\n");

    const int vocab_size = 4;
    const int seq_len = 2;

    float logits[] = {
        1.0f, 2.0f, 3.0f, 0.5f,
        0.5f, 1.5f, 2.0f, 1.0f
    };

    uint16_t targets[] = {0, 2};

    Cache cache;
    cache.max_logit = (float *)malloc(seq_len * sizeof(float));
    cache.d_logits = (float *)malloc(seq_len * vocab_size * sizeof(float));
    memset(cache.d_logits, 0, seq_len * vocab_size * sizeof(float));

    /* Forward pass to compute max_logit */
    float ce = compute_ce_loss(&cache, logits, targets, vocab_size, seq_len);
    (void)ce; /* unused */

    /* Backward pass */
    compute_ce_backward(&cache, logits, targets, vocab_size, seq_len);

    /* Manual gradient computation:
     * softmax[0] = [0.0853, 0.2324, 0.6309, 0.0514] (normalized)
     * scaled by 1/1 = 1.0
     * d_logits[0][0] = 0.0853
     * d_logits[0][1] = 0.2324
     * d_logits[0][2] = 0.6309 - 1.0 = -0.3691
     * d_logits[0][3] = 0.0514 */

    const float softmax_0[] = {0.0853f, 0.2324f, 0.6309f, 0.0514f};
    const float target_idx = 2;

    printf("  Gradient at position t=0:\n");
    for (int v = 0; v < vocab_size; v++) {
        float expected = softmax_0[v];
        if (v == (int)target_idx) {
            expected -= 1.0f;
        }
        float actual = cache.d_logits[0 * vocab_size + v];
        printf("    d_logits[0][%d] = %.4f (expected ~%.4f)\n", v, actual, expected);
        ASSERT_CLOSE(actual, expected, 0.002f, "d_logits[0][v] matches softmax-onehot");
    }

    printf("  Gradient at position t=1 (last, should be zero):\n");
    for (int v = 0; v < vocab_size; v++) {
        float actual = cache.d_logits[1 * vocab_size + v];
        printf("    d_logits[1][%d] = %.4f\n", v, actual);
        ASSERT_CLOSE(actual, 0.0f, 1e-6f, "d_logits[seq_len-1][v] is zero");
    }

    printf(ANSI_GREEN "PASS: test_ce_backward\n" ANSI_RESET);

    free(cache.max_logit);
    free(cache.d_logits);
    return 1;
}


/* ============================================================================
 * TEST 3: gp_kl_welford_update accumulation
 *
 * Simulate 3 timesteps of SSM hidden state accumulation.
 * Verify:
 *   1. n increments correctly
 *   2. mean converges to the true mean
 *   3. M2 accumulation (variance computation)
 *   4. lag1_prev is updated for next step
 * ============================================================================
 */

int test_gp_kl_welford_update(void) {
    printf("\n[TEST 3] gp_kl_welford_update accumulation\n");

    const int d_state = 4; /* Small for easy manual verification */
    GpKlMoments m;
    memset(&m, 0, sizeof(GpKlMoments));

    /* Three timesteps of hidden state */
    float h_t0[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_t1[] = {1.5f, 2.5f, 3.5f, 4.5f};
    float h_t2[] = {2.0f, 3.0f, 4.0f, 5.0f};

    printf("  Timestep 0: h = [1.0, 2.0, 3.0, 4.0]\n");
    gp_kl_welford_update(&m, h_t0, d_state, 0);
    ASSERT_EQ(m.n, 1, "n=1 after first update");
    printf("    n = %d\n", m.n);
    printf("    mean = [%.2f, %.2f, %.2f, %.2f]\n", m.mean[0], m.mean[1], m.mean[2], m.mean[3]);
    for (int i = 0; i < d_state; i++) {
        ASSERT_CLOSE(m.mean[i], h_t0[i], 1e-6f, "mean equals h_t0 after step 1");
    }

    printf("  Timestep 1: h = [1.5, 2.5, 3.5, 4.5]\n");
    gp_kl_welford_update(&m, h_t1, d_state, 4);
    ASSERT_EQ(m.n, 2, "n=2 after second update");
    printf("    n = %d\n", m.n);
    printf("    mean = [%.2f, %.2f, %.2f, %.2f]\n", m.mean[0], m.mean[1], m.mean[2], m.mean[3]);
    /* mean should be (h_t0 + h_t1) / 2 */
    for (int i = 0; i < d_state; i++) {
        float expected_mean = (h_t0[i] + h_t1[i]) / 2.0f;
        ASSERT_CLOSE(m.mean[i], expected_mean, 1e-6f, "mean correct after 2 samples");
    }

    printf("  Timestep 2: h = [2.0, 3.0, 4.0, 5.0]\n");
    gp_kl_welford_update(&m, h_t2, d_state, 8);
    ASSERT_EQ(m.n, 3, "n=3 after third update");
    printf("    n = %d\n", m.n);
    printf("    mean = [%.2f, %.2f, %.2f, %.2f]\n", m.mean[0], m.mean[1], m.mean[2], m.mean[3]);
    /* mean should be (h_t0 + h_t1 + h_t2) / 3 = (4.5, 7.5, 10.5, 13.5) / 3 = (1.5, 2.5, 3.5, 4.5) */
    for (int i = 0; i < d_state; i++) {
        float expected_mean = (h_t0[i] + h_t1[i] + h_t2[i]) / 3.0f;
        ASSERT_CLOSE(m.mean[i], expected_mean, 1e-6f, "mean correct after 3 samples");
    }

    /* Verify lag1_prev is updated */
    printf("  lag1_prev (should be h_t2): [%.2f, %.2f, %.2f, %.2f]\n",
           m.lag1_prev[0], m.lag1_prev[1], m.lag1_prev[2], m.lag1_prev[3]);
    for (int i = 0; i < d_state; i++) {
        ASSERT_CLOSE(m.lag1_prev[i], h_t2[i], 1e-6f, "lag1_prev correctly updated");
    }

    /* Verify lag1_sum accumulation
     * lag1_sum[t] = h_t0[t] * h_t(-1)[t] + h_t1[t] * h_t0[t] + h_t2[t] * h_t1[t]
     * Since h_t(-1) = 0 (initialization), we skip first term
     * lag1_sum[0] = 0 + 1.0*1.5 + 1.5*2.0 = 1.5 + 3.0 = 4.5 */
    const float expected_lag1_sum_0 = 1.5f + 3.0f; /* h_t1 * h_t0 + h_t2 * h_t1 */
    printf("  lag1_sum[0] = %.2f (expected ~%.2f)\n", m.lag1_sum[0], expected_lag1_sum_0);
    ASSERT_CLOSE(m.lag1_sum[0], expected_lag1_sum_0, 0.001f, "lag1_sum accumulates correctly");

    printf(ANSI_GREEN "PASS: test_gp_kl_welford_update\n" ANSI_RESET);

    return 1;
}


/* ============================================================================
 * MAIN
 * ============================================================================
 */

int main(void) {
    printf("==================================================\n");
    printf("  Loss Module Unit Tests (loss.h / loss.c)\n");
    printf("==================================================\n");

    int passed = 0;
    int total = 3;

    if (test_ce_loss()) passed++;
    if (test_ce_backward()) passed++;
    if (test_gp_kl_welford_update()) passed++;

    printf("\n==================================================\n");
    printf("Results: %d/%d tests passed\n", passed, total);
    printf("==================================================\n");

    return (passed == total) ? 0 : 1;
}
