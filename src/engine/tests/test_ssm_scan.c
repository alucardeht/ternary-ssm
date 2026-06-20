#include "ssm_scan.h"
#include "simd_detect.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("  FAIL: %s\n", #cond); \
        return 0; \
    } \
} while(0)

#define APPROX(a, b, eps) (fabsf((a) - (b)) < (eps))
#define ASSERT_APPROX(a, b, eps) do { \
    if (!APPROX((a), (b), (eps))) { \
        printf("  FAIL: %f != %f (eps=%f)\n", (float)(a), (float)(b), (float)(eps)); \
        return 0; \
    } \
} while(0)

int tests_passed = 0;
int tests_total = 0;

int test_state_alloc_reset(void) {
    printf("test_state_alloc_reset...");
    tests_total++;

    ssm_state_t* s = ssm_state_alloc(2, 2, 4, 4);
    ASSERT(s != NULL);
    ASSERT(s->n_heads == 2);
    ASSERT(s->d_state == 2);
    ASSERT(s->head_dim == 4);
    ASSERT(s->d_inner == 8);
    ASSERT(s->d_conv == 4);
    ASSERT(s->h != NULL);
    ASSERT(s->conv_buf != NULL);

    s->h[0] = 1.5f;
    s->h[5] = -2.0f;
    s->conv_buf[0] = 3.14f;
    ssm_state_reset(s);
    ASSERT(s->h[0] == 0.0f);
    ASSERT(s->h[5] == 0.0f);
    ASSERT(s->conv_buf[0] == 0.0f);

    ssm_state_free(s);
    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_conv1d_step(void) {
    printf("test_conv1d_step...");
    tests_total++;

    int32_t d_inner = 4;
    int32_t d_conv = 4;

    float* y = (float*)calloc((size_t)d_inner, sizeof(float));
    float* conv_buf = (float*)calloc((size_t)d_inner * d_conv, sizeof(float));
    float* x = (float*)calloc((size_t)d_inner, sizeof(float));
    float* weight = (float*)calloc((size_t)d_inner * d_conv, sizeof(float));
    float* bias = (float*)calloc((size_t)d_inner, sizeof(float));

    for (int32_t i = 0; i < d_inner * d_conv; i++) {
        weight[i] = 0.25f;
    }
    for (int32_t i = 0; i < d_inner; i++) {
        bias[i] = 0.0f;
    }

    x[0] = 1.0f; x[1] = 2.0f; x[2] = 3.0f; x[3] = 4.0f;
    ssm_conv1d_step(y, conv_buf, x, weight, bias, d_inner, d_conv);
    ASSERT_APPROX(y[0], 0.25f, 1e-5f);
    ASSERT_APPROX(y[1], 0.5f, 1e-5f);

    for (int32_t i = 0; i < d_inner; i++) x[i] = 1.0f;
    ssm_conv1d_step(y, conv_buf, x, weight, bias, d_inner, d_conv);
    ASSERT_APPROX(y[0], 0.5f, 1e-5f);

    for (int32_t i = 0; i < d_inner; i++) x[i] = 2.0f;
    ssm_conv1d_step(y, conv_buf, x, weight, bias, d_inner, d_conv);
    ASSERT_APPROX(y[0], 1.0f, 1e-5f);

    free(y);
    free(conv_buf);
    free(x);
    free(weight);
    free(bias);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_scan_step_standard(void) {
    printf("test_scan_step_standard...");
    tests_total++;

    int32_t n_heads = 2;
    int32_t d_state = 2;
    int32_t head_dim = 4;

    float* y = (float*)calloc((size_t)n_heads * head_dim, sizeof(float));
    float* h = (float*)calloc((size_t)n_heads * d_state * head_dim, sizeof(float));
    float* x = (float*)calloc((size_t)n_heads * head_dim, sizeof(float));
    float* dt = (float*)calloc((size_t)n_heads * head_dim, sizeof(float));
    float* A = (float*)calloc((size_t)n_heads, sizeof(float));
    float* B = (float*)calloc((size_t)d_state, sizeof(float));
    float* C = (float*)calloc((size_t)d_state, sizeof(float));
    float* dt_heads = (float*)calloc((size_t)n_heads, sizeof(float));

    for (int32_t i = 0; i < n_heads * head_dim; i++) {
        x[i] = 0.1f;
        dt[i] = 0.01f;
    }
    for (int32_t i = 0; i < n_heads; i++) {
        A[i] = -1.0f;
    }
    for (int32_t i = 0; i < d_state; i++) {
        B[i] = 0.5f;
        C[i] = 1.0f;
    }

    ssm_scan_step(y, h, x, dt, A, B, C, dt_heads, n_heads, d_state, head_dim);

    ASSERT(y[0] != 0.0f);
    ASSERT(h[0] != 0.0f);
    ASSERT(dt_heads[0] > 0.0f && dt_heads[0] < 1.0f);

    free(y);
    free(h);
    free(x);
    free(dt);
    free(A);
    free(B);
    free(C);
    free(dt_heads);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_scan_step_ternary(void) {
    printf("test_scan_step_ternary...");
    tests_total++;

    int32_t n_heads = 2;
    int32_t d_state = 2;
    int32_t head_dim = 4;

    float* y = (float*)calloc((size_t)n_heads * head_dim, sizeof(float));
    float* h = (float*)calloc((size_t)n_heads * d_state * head_dim, sizeof(float));
    float* x = (float*)calloc((size_t)n_heads * head_dim, sizeof(float));
    float* dt = (float*)calloc((size_t)n_heads * head_dim, sizeof(float));
    int8_t* A = (int8_t*)calloc((size_t)n_heads, sizeof(int8_t));
    float* B = (float*)calloc((size_t)d_state, sizeof(float));
    float* C = (float*)calloc((size_t)d_state, sizeof(float));
    float* dt_heads = (float*)calloc((size_t)n_heads, sizeof(float));

    for (int32_t i = 0; i < n_heads * head_dim; i++) {
        x[i] = 0.1f;
        dt[i] = 0.01f;
    }
    A[0] = 1;
    A[1] = -1;
    for (int32_t i = 0; i < d_state; i++) {
        B[i] = 0.5f;
        C[i] = 1.0f;
    }

    ssm_scan_step_ternary(y, h, x, dt, A, B, C, dt_heads, n_heads, d_state, head_dim);

    ASSERT(y[0] != 0.0f);
    ASSERT(h[0] != 0.0f);
    ASSERT(dt_heads[0] > 0.0f && dt_heads[0] < 1.0f);

    float h1_before = h[(1 * d_state) * head_dim];
    ssm_scan_step_ternary(y, h, x, dt, A, B, C, dt_heads, n_heads, d_state, head_dim);
    float h1_after = h[(1 * d_state) * head_dim];

    ASSERT(h1_before != h1_after);

    free(y);
    free(h);
    free(x);
    free(dt);
    free(A);
    free(B);
    free(C);
    free(dt_heads);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_scan_multi_step(void) {
    printf("test_scan_multi_step...");
    tests_total++;

    int32_t n_heads = 2;
    int32_t d_state = 2;
    int32_t head_dim = 4;

    float* y = (float*)calloc((size_t)n_heads * head_dim, sizeof(float));
    float* h = (float*)calloc((size_t)n_heads * d_state * head_dim, sizeof(float));
    float* x = (float*)calloc((size_t)n_heads * head_dim, sizeof(float));
    float* dt = (float*)calloc((size_t)n_heads * head_dim, sizeof(float));
    float* A = (float*)calloc((size_t)n_heads, sizeof(float));
    float* B = (float*)calloc((size_t)d_state, sizeof(float));
    float* C = (float*)calloc((size_t)d_state, sizeof(float));
    float* dt_heads = (float*)calloc((size_t)n_heads, sizeof(float));

    for (int32_t i = 0; i < n_heads * head_dim; i++) {
        x[i] = 0.1f;
        dt[i] = 0.01f;
    }
    for (int32_t i = 0; i < n_heads; i++) {
        A[i] = -2.0f;
    }
    for (int32_t i = 0; i < d_state; i++) {
        B[i] = 0.5f;
        C[i] = 1.0f;
    }

    float h_init = h[0];
    for (int step = 0; step < 5; step++) {
        ssm_scan_step(y, h, x, dt, A, B, C, dt_heads, n_heads, d_state, head_dim);
    }
    float h_final = h[0];

    ASSERT(h_final != h_init);
    ASSERT(!isnan(h_final) && !isinf(h_final));

    free(y);
    free(h);
    free(x);
    free(dt);
    free(A);
    free(B);
    free(C);
    free(dt_heads);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int main(void) {
    printf("=== SSM Scan Tests ===\n\n");

    simd_detect_init();
    simd_print_caps();
    printf("\n");

    ssm_scan_init();

    test_state_alloc_reset();
    test_conv1d_step();
    test_scan_step_standard();
    test_scan_step_ternary();
    test_scan_multi_step();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
