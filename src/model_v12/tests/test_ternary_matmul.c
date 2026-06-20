/*
 * test_ternary_matmul.c — Test suite for ternary CSR matmul operations.
 *
 * Tests:
 * 1. Small 4x8 ternary matrix: CSR build and matvec correctness
 * 2. Batched matmul: Y = W @ X with 3 samples
 * 3. 1024x1024 random sparse matrix: 40% sparsity, correctness check
 */

#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

/* Include the ternary matmul header */
#include "../src/core/ternary_matmul.h"
#include "../src/core/types.h"

/* =========================================================================
 * HELPER: Dense reference matvec for correctness checking
 * ========================================================================= */

static void dense_matvec_ref(const int8_t *W, int32_t rows, int32_t cols,
                              const float *x, float *y) {
    for (int32_t i = 0; i < rows; i++) {
        float acc = 0.0f;
        for (int32_t j = 0; j < cols; j++) {
            acc += (float)W[i * cols + j] * x[j];
        }
        y[i] = acc;
    }
}

static void dense_matmul_batch_ref(const int8_t *W, int32_t rows, int32_t cols,
                                    const float *X, float *Y, int32_t batch) {
    for (int32_t i = 0; i < rows; i++) {
        for (int32_t b = 0; b < batch; b++) {
            float acc = 0.0f;
            for (int32_t j = 0; j < cols; j++) {
                acc += (float)W[i * cols + j] * X[j * batch + b];
            }
            Y[i * batch + b] = acc;
        }
    }
}

/* =========================================================================
 * HELPER: Floating-point comparison with tolerance
 * ========================================================================= */

static int float_nearly_equal(float a, float b, float tol) {
    float diff = a > b ? a - b : b - a;
    return diff < tol;
}

/* =========================================================================
 * TEST 1: Small 4x8 CSR build and matvec
 * ========================================================================= */

static void test_small_csr_matvec(void) {
    printf("[TEST 1] Small 4x8 CSR build and matvec\n");

    /* Create a small 4x8 ternary matrix (row-major):
     *  [+1  0 -1  0  0 +1  0  0]
     *  [ 0 +1  0 -1  0  0 +1  0]
     *  [-1  0  0  0 +1  0  0 -1]
     *  [ 0  0 +1  0  0  0 -1 +1]
     */
    int8_t W[] = {
        1,  0, -1,  0,  0,  1,  0,  0,
        0,  1,  0, -1,  0,  0,  1,  0,
       -1,  0,  0,  0,  1,  0,  0, -1,
        0,  0,  1,  0,  0,  0, -1,  1
    };

    float x[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
    float y_ref[4];
    float y_csr[4];

    /* Compute reference */
    dense_matvec_ref(W, 4, 8, x, y_ref);

    /* Build CSR and compute */
    TernaryCSR csr = {0};
    assert(ternary_csr_init(&csr, W, 4, 8) == 0);
    assert(ternary_matvec(&csr, x, y_csr) == 0);

    /* Check correctness */
    for (int i = 0; i < 4; i++) {
        assert(float_nearly_equal(y_ref[i], y_csr[i], 1e-6f));
        printf("  y[%d]: ref=%.1f, csr=%.1f ✓\n", i, y_ref[i], y_csr[i]);
    }

    ternary_csr_free(&csr);
    printf("  PASS\n\n");
}

/* =========================================================================
 * TEST 2: Batched matmul with 3 samples
 * ========================================================================= */

static void test_batched_matmul(void) {
    printf("[TEST 2] Batched matmul (4x8, batch=3)\n");

    int8_t W[] = {
        1,  0, -1,  0,  0,  1,  0,  0,
        0,  1,  0, -1,  0,  0,  1,  0,
       -1,  0,  0,  0,  1,  0,  0, -1,
        0,  0,  1,  0,  0,  0, -1,  1
    };

    /* Column-major batch: X[j * batch + b] = input j for sample b */
    float X[] = {
        /* sample 0 */
        1.0f,  2.0f,  3.0f,  4.0f,
        /* sample 1 */
        2.0f,  3.0f,  4.0f,  5.0f,
        /* sample 2 */
        3.0f,  4.0f,  5.0f,  6.0f,
        /* j=4..7 for samples 0,1,2 */
        5.0f,  6.0f,  7.0f,  8.0f,
        6.0f,  7.0f,  8.0f,  9.0f,
        7.0f,  8.0f,  9.0f, 10.0f,
        8.0f,  9.0f, 10.0f, 11.0f
    };

    int32_t rows = 4, cols = 8, batch = 3;
    float Y_ref[4 * 3] = {0};
    float Y_csr[4 * 3] = {0};

    /* Compute reference: extract columns and compute matvec for each */
    for (int32_t b = 0; b < batch; b++) {
        float x_temp[8];
        for (int32_t j = 0; j < cols; j++) {
            x_temp[j] = X[j * batch + b];
        }
        float y_temp[4];
        dense_matvec_ref(W, rows, cols, x_temp, y_temp);
        for (int32_t i = 0; i < rows; i++) {
            Y_ref[i * batch + b] = y_temp[i];
        }
    }

    /* Build CSR and compute batched matmul */
    TernaryCSR csr = {0};
    assert(ternary_csr_init(&csr, W, rows, cols) == 0);
    assert(ternary_matmul_batch(&csr, X, Y_csr, batch) == 0);

    /* Check correctness */
    for (int32_t i = 0; i < rows; i++) {
        for (int32_t b = 0; b < batch; b++) {
            float ref_val = Y_ref[i * batch + b];
            float csr_val = Y_csr[i * batch + b];
            assert(float_nearly_equal(ref_val, csr_val, 1e-6f));
        }
    }

    printf("  Y_ref: ");
    for (int i = 0; i < 12; i++) printf("%.1f ", Y_ref[i]);
    printf("\n");
    printf("  Y_csr: ");
    for (int i = 0; i < 12; i++) printf("%.1f ", Y_csr[i]);
    printf("\n");

    ternary_csr_free(&csr);
    printf("  PASS\n\n");
}

/* =========================================================================
 * TEST 3: 1024x1024 random sparse matrix
 * ========================================================================= */

static void test_large_sparse_matrix(void) {
    printf("[TEST 3] Large 1024x1024 random sparse matrix (40%% sparsity)\n");

    int32_t rows = 1024, cols = 1024;
    int8_t *W = (int8_t *)malloc(rows * cols * sizeof(int8_t));
    float *x = (float *)malloc(cols * sizeof(float));
    float *y_ref = (float *)malloc(rows * sizeof(float));
    float *y_csr = (float *)malloc(rows * sizeof(float));

    assert(W && x && y_ref && y_csr);

    /* Generate random sparse ternary matrix: 40% nonzero */
    unsigned int seed = 12345;
    for (int32_t i = 0; i < rows * cols; i++) {
        seed = seed * 1103515245 + 12345;
        int rand_val = (seed / 65536) % 100;
        if (rand_val < 40) {
            W[i] = (rand_val < 20) ? 1 : -1;
        } else {
            W[i] = 0;
        }
    }

    /* Generate random input vector */
    seed = 54321;
    for (int32_t j = 0; j < cols; j++) {
        seed = seed * 1103515245 + 12345;
        x[j] = (float)((seed / 65536) % 1000) / 500.0f - 1.0f;
    }

    /* Compute reference (slow, but correct) */
    printf("  Computing reference (1M ops)...\n");
    dense_matvec_ref(W, rows, cols, x, y_ref);

    /* Build CSR and compute */
    printf("  Building CSR...\n");
    TernaryCSR csr = {0};
    assert(ternary_csr_init(&csr, W, rows, cols) == 0);

    printf("  CSR stats: total_pos=%ld, total_neg=%ld\n",
           (long)csr.total_pos, (long)csr.total_neg);

    printf("  Computing CSR matvec...\n");
    assert(ternary_matvec(&csr, x, y_csr) == 0);

    /* Check correctness on a sample of outputs */
    printf("  Spot-checking 10 outputs...\n");
    for (int32_t i = 0; i < 10; i++) {
        int32_t idx = i * (rows / 10);
        assert(float_nearly_equal(y_ref[idx], y_csr[idx], 1e-4f));
        printf("    y[%ld]: ref=%.2f, csr=%.2f ✓\n",
               (long)idx, y_ref[idx], y_csr[idx]);
    }

    ternary_csr_free(&csr);
    free(W);
    free(x);
    free(y_ref);
    free(y_csr);
    printf("  PASS\n\n");
}

/* =========================================================================
 * MAIN: Run all tests
 * ========================================================================= */

int main(void) {
    printf("========================================\n");
    printf("Ternary Matmul CSR Test Suite\n");
    printf("========================================\n\n");

    test_small_csr_matvec();
    test_batched_matmul();
    test_large_sparse_matrix();

    printf("========================================\n");
    printf("All tests passed!\n");
    printf("========================================\n");

    return 0;
}
