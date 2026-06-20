#include "../include/ternary_gemv.h"
#include "../include/simd_detect.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#define EPSILON 1e-5f

static int float_approx_equal(float a, float b, float eps)
{
    float diff = fabsf(a - b);
    float threshold = eps * (fabsf(a) + fabsf(b)) / 2.0f;
    return diff <= threshold;
}

static void print_float_array(const char* label, const float* arr, int n)
{
    printf("%s: [", label);
    for (int i = 0; i < n; i++) {
        printf("%.6f%s", arr[i], i < n - 1 ? ", " : "");
    }
    printf("]\n");
}

static void test_small_matrix(void)
{
    printf("\n=== Test: Small Matrix (4x8) ===\n");

    int rows = 4, cols = 8;
    ternary_matrix_t* W = ternary_matrix_alloc(rows, cols);
    assert(W != NULL);

    /*
     * Create a known weight matrix:
     * W = [
     *   [+1, -1, +1, -1, +1, -1, +1, -1],
     *   [-1, +1, -1, +1, -1, +1, -1, +1],
     *   [+1, +1, +1, +1, -1, -1, -1, -1],
     *   [ 0,  0,  0,  0,  0,  0,  0,  0]
     * ]
     */
    int8_t weights[] = {
        1, -1, 1, -1, 1, -1, 1, -1,
        -1, 1, -1, 1, -1, 1, -1, 1,
        1, 1, 1, 1, -1, -1, -1, -1,
        0, 0, 0, 0, 0, 0, 0, 0
    };

    ternary_matrix_pack(W, weights);

    /* Set scale factors */
    W->scales[0] = 1.0f;
    W->scales[1] = 1.0f;
    W->scales[2] = 1.0f;
    W->scales[3] = 1.0f;

    /* Create input vector: x = [1, 2, 3, 4, 5, 6, 7, 8] */
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    /* Create bias: bias = [0.5, -0.5, 1.0, 2.0] */
    float bias[] = {0.5f, -0.5f, 1.0f, 2.0f};

    /* Compute expected output manually:
     * y[0] = (1*1 + 2*-1 + 3*1 + 4*-1 + 5*1 + 6*-1 + 7*1 + 8*-1) * 1.0 + 0.5
     *      = (1 - 2 + 3 - 4 + 5 - 6 + 7 - 8) * 1.0 + 0.5
     *      = -4 * 1.0 + 0.5 = -3.5
     *
     * y[1] = (-1*1 + 1*2 + -1*3 + 1*4 + -1*5 + 1*6 + -1*7 + 1*8) * 1.0 - 0.5
     *      = (-1 + 2 - 3 + 4 - 5 + 6 - 7 + 8) * 1.0 - 0.5
     *      = 4 * 1.0 - 0.5 = 3.5
     *
     * y[2] = (1*1 + 1*2 + 1*3 + 1*4 + -1*5 + -1*6 + -1*7 + -1*8) * 1.0 + 1.0
     *      = (1 + 2 + 3 + 4 - 5 - 6 - 7 - 8) * 1.0 + 1.0
     *      = -16 * 1.0 + 1.0 = -15.0
     *
     * y[3] = (0*1 + 0*2 + 0*3 + 0*4 + 0*5 + 0*6 + 0*7 + 0*8) * 1.0 + 2.0
     *      = 0 * 1.0 + 2.0 = 2.0
     */
    float expected[] = {-3.5f, 3.5f, -15.0f, 2.0f};

    /* Test scalar kernel */
    float y_scalar[4];
    ternary_gemv_scalar(y_scalar, W, x, bias, rows, cols);
    printf("Scalar result: ");
    print_float_array("", y_scalar, rows);

    for (int i = 0; i < rows; i++) {
        assert(float_approx_equal(y_scalar[i], expected[i], EPSILON));
    }
    printf("Scalar: PASS\n");

#ifdef ENGINE_AVX2
    /* Test AVX2 kernel */
    float y_avx2[4];
    ternary_gemv_avx2(y_avx2, W, x, bias, rows, cols);
    printf("AVX2 result:   ");
    print_float_array("", y_avx2, rows);

    for (int i = 0; i < rows; i++) {
        assert(float_approx_equal(y_avx2[i], expected[i], EPSILON));
    }
    printf("AVX2: PASS\n");

    /* Verify scalar and AVX2 match */
    for (int i = 0; i < rows; i++) {
        assert(float_approx_equal(y_scalar[i], y_avx2[i], EPSILON));
    }
    printf("Scalar vs AVX2: MATCH\n");
#endif

    /* Test dispatched kernel */
    ternary_gemv_init();
    float y_dispatch[4];
    ternary_gemv(y_dispatch, W, x, bias, rows, cols);
    printf("Dispatched result: ");
    print_float_array("", y_dispatch, rows);

    for (int i = 0; i < rows; i++) {
        assert(float_approx_equal(y_dispatch[i], expected[i], EPSILON));
    }
    printf("Dispatched: PASS\n");

    ternary_matrix_free(W);
}

static void test_without_bias(void)
{
    printf("\n=== Test: Without Bias ===\n");

    int rows = 2, cols = 4;
    ternary_matrix_t* W = ternary_matrix_alloc(rows, cols);
    assert(W != NULL);

    /* Simple weights and input */
    int8_t weights[] = {
        1, 1, 1, 1,
        -1, -1, -1, -1
    };
    ternary_matrix_pack(W, weights);

    W->scales[0] = 2.0f;
    W->scales[1] = 0.5f;

    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};

    /* Expected:
     * y[0] = (1 + 2 + 3 + 4) * 2.0 + 0 = 10 * 2.0 = 20.0
     * y[1] = (-1 - 2 - 3 - 4) * 0.5 + 0 = -10 * 0.5 = -5.0
     */
    float expected[] = {20.0f, -5.0f};

    float y[2];
    ternary_gemv(y, W, x, NULL, rows, cols);

    for (int i = 0; i < rows; i++) {
        assert(float_approx_equal(y[i], expected[i], EPSILON));
    }
    printf("Without bias: PASS\n");

    ternary_matrix_free(W);
}

static void test_large_matrix(void)
{
    printf("\n=== Test: Large Matrix (256x256) ===\n");

    int rows = 256, cols = 256;
    ternary_matrix_t* W = ternary_matrix_alloc(rows, cols);
    assert(W != NULL);

    /* Fill with alternating pattern */
    int8_t* weights = (int8_t*)malloc(rows * cols * sizeof(int8_t));
    for (int i = 0; i < rows * cols; i++) {
        int pattern = i % 3;
        if (pattern == 0) weights[i] = 1;
        else if (pattern == 1) weights[i] = -1;
        else weights[i] = 0;
    }
    ternary_matrix_pack(W, weights);

    /* Set scale factors to 1.0 */
    for (int i = 0; i < rows; i++) {
        W->scales[i] = 1.0f;
    }

    /* Create input vector */
    float* x = (float*)malloc(cols * sizeof(float));
    for (int i = 0; i < cols; i++) {
        x[i] = 1.0f + (i % 10) * 0.1f;
    }

    /* Compute with both kernels and verify no crashes */
    float* y_scalar = (float*)malloc(rows * sizeof(float));
    float* y_dispatch = (float*)malloc(rows * sizeof(float));

    ternary_gemv_scalar(y_scalar, W, x, NULL, rows, cols);
    printf("Scalar: completed without crash\n");

    ternary_gemv(y_dispatch, W, x, NULL, rows, cols);
    printf("Dispatched: completed without crash\n");

    /* Verify results match (with larger tolerance for accumulated error) */
    for (int i = 0; i < rows; i++) {
        assert(float_approx_equal(y_scalar[i], y_dispatch[i], 1e-4f));
    }
    printf("Scalar vs Dispatched: MATCH\n");

    free(weights);
    free(x);
    free(y_scalar);
    free(y_dispatch);
    ternary_matrix_free(W);
}

static void test_edge_cases(void)
{
    printf("\n=== Test: Edge Cases ===\n");

    /* Test 1: All zeros */
    {
        int rows = 2, cols = 4;
        ternary_matrix_t* W = ternary_matrix_alloc(rows, cols);

        int8_t weights[] = {0, 0, 0, 0, 0, 0, 0, 0};
        ternary_matrix_pack(W, weights);
        W->scales[0] = 1.0f;
        W->scales[1] = 2.0f;

        float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float bias[] = {5.0f, -3.0f};

        float expected[] = {5.0f, -3.0f};  /* Only bias matters */

        float y[2];
        ternary_gemv(y, W, x, bias, rows, cols);

        for (int i = 0; i < rows; i++) {
            assert(float_approx_equal(y[i], expected[i], EPSILON));
        }
        printf("All-zeros matrix: PASS\n");
        ternary_matrix_free(W);
    }

    /* Test 2: All ones */
    {
        int rows = 2, cols = 4;
        ternary_matrix_t* W = ternary_matrix_alloc(rows, cols);

        int8_t weights[] = {1, 1, 1, 1, 1, 1, 1, 1};
        ternary_matrix_pack(W, weights);
        W->scales[0] = 1.0f;
        W->scales[1] = 1.0f;

        float x[] = {1.0f, 2.0f, 3.0f, 4.0f};

        float expected[] = {10.0f, 10.0f};  /* 1+2+3+4 = 10 */

        float y[2];
        ternary_gemv(y, W, x, NULL, rows, cols);

        for (int i = 0; i < rows; i++) {
            assert(float_approx_equal(y[i], expected[i], EPSILON));
        }
        printf("All-ones matrix: PASS\n");
        ternary_matrix_free(W);
    }

    /* Test 3: Single row */
    {
        int rows = 1, cols = 8;
        ternary_matrix_t* W = ternary_matrix_alloc(rows, cols);

        int8_t weights[] = {1, -1, 1, -1, 1, -1, 1, -1};
        ternary_matrix_pack(W, weights);
        W->scales[0] = 2.0f;

        float x[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        float bias[] = {0.5f};

        float expected[] = {0.5f};  /* (1-1+1-1+1-1+1-1)*2.0 + 0.5 = 0.5 */

        float y[1];
        ternary_gemv(y, W, x, bias, rows, cols);

        assert(float_approx_equal(y[0], expected[0], EPSILON));
        printf("Single-row matrix: PASS\n");
        ternary_matrix_free(W);
    }
}

int main(void)
{
    printf("=== Ternary GEMV Kernel Tests ===\n");

    simd_detect_init();
    const simd_caps_t* caps = simd_get_caps();
    printf("SIMD capabilities:\n");
    printf("  AVX2: %s\n", caps->has_avx2 ? "YES" : "NO");
    printf("  AVX-512: %s\n", caps->has_avx512 ? "YES" : "NO");
    printf("  NEON: %s\n", caps->has_neon ? "YES" : "NO");
    printf("  POPCNT: %s\n", caps->has_popcnt ? "YES" : "NO");

    test_small_matrix();
    test_without_bias();
    test_edge_cases();
    test_large_matrix();

    printf("\n=== All Tests PASSED ===\n");
    return 0;
}
