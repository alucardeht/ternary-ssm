#include "../include/activations.h"
#include "../include/simd_detect.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int tests_passed = 0;
static int tests_total = 0;

#define ASSERT(cond, msg) do { \
    tests_total++; \
    if (!(cond)) { \
        printf("FAIL: %s (line %d)\n", msg, __LINE__); \
    } else { \
        printf("PASS: %s\n", msg); \
        tests_passed++; \
    } \
} while(0)

#define APPROX(a, b, eps) (fabsf((a)-(b)) < (eps))

static void test_rmsnorm_basic(void)
{
    printf("\n=== Test: RMSNorm Basic ===\n");

    int32_t n = 4;
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float weight[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float y[4];
    float eps = 1e-6f;

    float sum_sq = 1.0f + 4.0f + 9.0f + 16.0f;
    float rms_val = 1.0f / sqrtf(sum_sq / n + eps);
    float expected[] = {
        1.0f * rms_val,
        2.0f * rms_val,
        3.0f * rms_val,
        4.0f * rms_val
    };

    act_rmsnorm(y, x, weight, n, eps);

    for (int i = 0; i < n; i++) {
        ASSERT(APPROX(y[i], expected[i], 1e-5f), "RMSNorm element match");
    }
}

static void test_rmsnorm_with_weight(void)
{
    printf("\n=== Test: RMSNorm With Weight ===\n");

    int32_t n = 4;
    float x[] = {2.0f, -1.0f, 3.0f, -2.0f};
    float weight[] = {0.5f, 2.0f, 1.0f, 0.25f};
    float y[4];
    float eps = 1e-6f;

    float sum_sq = 4.0f + 1.0f + 9.0f + 4.0f;
    float rms_val = 1.0f / sqrtf(sum_sq / n + eps);

    act_rmsnorm(y, x, weight, n, eps);

    for (int i = 0; i < n; i++) {
        float expected = x[i] * weight[i] * rms_val;
        ASSERT(APPROX(y[i], expected, 1e-5f), "RMSNorm with weight match");
    }
}

static void test_rmsnorm_large(void)
{
    printf("\n=== Test: RMSNorm Large Vector ===\n");

    int32_t n = 256;
    float* x = (float*)malloc(n * sizeof(float));
    float* weight = (float*)malloc(n * sizeof(float));
    float* y_scalar = (float*)malloc(n * sizeof(float));
    float* y_dispatch = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        x[i] = (i - n/2) * 0.1f;
        weight[i] = 1.0f;
    }

    act_rmsnorm_scalar(y_scalar, x, weight, n, 1e-6f);
    act_rmsnorm(y_dispatch, x, weight, n, 1e-6f);

    for (int i = 0; i < n; i++) {
        ASSERT(APPROX(y_scalar[i], y_dispatch[i], 1e-5f), "RMSNorm large vector match");
    }

    free(x);
    free(weight);
    free(y_scalar);
    free(y_dispatch);
}

static void test_silu_basic(void)
{
    printf("\n=== Test: SiLU Basic ===\n");

    float x[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 5.0f, 10.0f};
    int32_t n = 8;
    float y[8];

    act_silu(y, x, n);

    for (int i = 0; i < n; i++) {
        float sigmoid = 1.0f / (1.0f + expf(-x[i]));
        float expected = x[i] * sigmoid;
        ASSERT(APPROX(y[i], expected, 1e-4f), "SiLU element match");
    }
}

static void test_silu_edge_cases(void)
{
    printf("\n=== Test: SiLU Edge Cases ===\n");

    float x[] = {50.0f, -50.0f, 0.0f};
    int32_t n = 3;
    float y[3];

    act_silu(y, x, n);

    ASSERT(APPROX(y[0], 50.0f, 1e-3f), "SiLU large positive");
    ASSERT(APPROX(y[1], 0.0f, 1e-3f), "SiLU large negative");
    ASSERT(APPROX(y[2], 0.0f, 1e-5f), "SiLU zero");
}

static void test_relu_basic(void)
{
    printf("\n=== Test: ReLU Basic ===\n");

    float x[] = {-3.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, -0.001f, 100.0f};
    int32_t n = 8;
    float y[8];
    float expected[] = {0.0f, 0.0f, 0.0f, 0.5f, 1.0f, 2.0f, 0.0f, 100.0f};

    act_relu(y, x, n);

    for (int i = 0; i < n; i++) {
        ASSERT(APPROX(y[i], expected[i], 1e-5f), "ReLU element match");
    }
}

static void test_mul_basic(void)
{
    printf("\n=== Test: Element-wise Multiply Basic ===\n");

    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {0.5f, -1.0f, 2.0f, 0.0f};
    int32_t n = 4;
    float y[4];
    float expected[] = {0.5f, -2.0f, 6.0f, 0.0f};

    act_mul(y, a, b, n);

    for (int i = 0; i < n; i++) {
        ASSERT(APPROX(y[i], expected[i], 1e-5f), "Multiply element match");
    }
}

static void test_softplus_basic(void)
{
    printf("\n=== Test: Softplus Basic ===\n");

    float x[] = {-5.0f, -1.0f, 0.0f, 1.0f, 5.0f, 20.0f, 25.0f, -20.0f};
    int32_t n = 8;
    float y[8];

    act_softplus(y, x, n);

    for (int i = 0; i < n; i++) {
        float expected;
        if (x[i] > 20.0f) {
            expected = x[i];
        } else if (x[i] < -20.0f) {
            expected = 0.0f;
        } else {
            expected = logf(1.0f + expf(x[i]));
        }
        ASSERT(APPROX(y[i], expected, 1e-3f), "Softplus element match");
    }
}

static void test_silu_mul_fused(void)
{
    printf("\n=== Test: SiLU-Mul Fused ===\n");

    float a[] = {1.0f, 2.0f, -1.0f, 0.0f};
    float b[] = {2.0f, 3.0f, 4.0f, 5.0f};
    int32_t n = 4;
    float y[4];

    act_silu_mul(y, a, b, n);

    for (int i = 0; i < n; i++) {
        float sigmoid = 1.0f / (1.0f + expf(-a[i]));
        float expected = a[i] * sigmoid * b[i];
        ASSERT(APPROX(y[i], expected, 1e-4f), "SiLU-Mul element match");
    }
}

static void test_large_vectors(void)
{
    printf("\n=== Test: Large Vectors (256 elements) ===\n");

    int32_t n = 256;
    float* x = (float*)malloc(n * sizeof(float));
    float* a = (float*)malloc(n * sizeof(float));
    float* b = (float*)malloc(n * sizeof(float));
    float* weight = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        x[i] = (i - n/2) * 0.01f;
        a[i] = cosf(i * 0.1f);
        b[i] = sinf(i * 0.1f);
        weight[i] = 1.0f / sqrtf(n);
    }

    float* y_silu_scalar = (float*)malloc(n * sizeof(float));
    float* y_silu_dispatch = (float*)malloc(n * sizeof(float));
    float* y_relu_scalar = (float*)malloc(n * sizeof(float));
    float* y_relu_dispatch = (float*)malloc(n * sizeof(float));

    act_silu_scalar(y_silu_scalar, x, n);
    act_silu(y_silu_dispatch, x, n);
    act_relu_scalar(y_relu_scalar, x, n);
    act_relu(y_relu_dispatch, x, n);

    int match_count = 0;
    for (int i = 0; i < n; i++) {
        if (APPROX(y_silu_scalar[i], y_silu_dispatch[i], 1e-4f)) match_count++;
    }
    ASSERT(match_count == n, "SiLU scalar vs dispatch large vector");

    match_count = 0;
    for (int i = 0; i < n; i++) {
        if (APPROX(y_relu_scalar[i], y_relu_dispatch[i], 1e-5f)) match_count++;
    }
    ASSERT(match_count == n, "ReLU scalar vs dispatch large vector");

    free(x);
    free(a);
    free(b);
    free(weight);
    free(y_silu_scalar);
    free(y_silu_dispatch);
    free(y_relu_scalar);
    free(y_relu_dispatch);
}

int main(void)
{
    printf("=== Activation Kernel Tests ===\n");

    simd_detect_init();
    simd_print_caps();
    act_init();

    test_rmsnorm_basic();
    test_rmsnorm_with_weight();
    test_rmsnorm_large();
    test_silu_basic();
    test_silu_edge_cases();
    test_relu_basic();
    test_mul_basic();
    test_softplus_basic();
    test_silu_mul_fused();
    test_large_vectors();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
