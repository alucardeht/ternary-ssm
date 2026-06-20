#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "ternary_types.h"
#include "ternary_gemv.h"
#include "popcount_gemv.h"
#include "simd_detect.h"

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

void test_bitmap_conversion(void) {
    printf("\n--- Test: Bitmap Conversion ---\n");

    ternary_matrix_t* tm = ternary_matrix_alloc(2, 8);
    int8_t dense[] = {1, -1, 0, 1, -1, 0, 1, -1,
                      0, 0, 1, -1, 0, 1, 0, -1};
    ternary_matrix_pack(tm, dense);

    popcount_matrix_t* pm = popcount_matrix_alloc(2, 8);
    popcount_matrix_from_ternary(pm, tm);

    uint64_t expected_val0 = (1ULL<<0)|(1ULL<<1)|(1ULL<<3)|(1ULL<<4)|(1ULL<<6)|(1ULL<<7);
    uint64_t expected_sign0 = (1ULL<<1)|(1ULL<<4)|(1ULL<<7);
    ASSERT(pm->val_bits[0] == expected_val0, "Row 0 val_bits correct");
    ASSERT(pm->sign_bits[0] == expected_sign0, "Row 0 sign_bits correct");

    uint64_t expected_val1 = (1ULL<<2)|(1ULL<<3)|(1ULL<<5)|(1ULL<<7);
    uint64_t expected_sign1 = (1ULL<<3)|(1ULL<<7);
    ASSERT(pm->val_bits[1 * pm->words_per_row] == expected_val1, "Row 1 val_bits correct");
    ASSERT(pm->sign_bits[1 * pm->words_per_row] == expected_sign1, "Row 1 sign_bits correct");

    popcount_matrix_free(pm);
    ternary_matrix_free(tm);
}

void test_activation_quantization(void) {
    printf("\n--- Test: Activation Quantization ---\n");

    float x[] = {0.1f, 0.5f, 0.3f, 0.9f, 0.2f, 0.7f, 0.4f, 0.8f};
    int32_t n = 8;
    uint64_t x_bits[1];
    float threshold;

    popcount_quantize_activations(x, n, x_bits, &threshold);

    ASSERT(fabsf(threshold - 0.4875f) < 0.001f, "Threshold is mean of activations");

    uint64_t expected = (1ULL<<1)|(1ULL<<3)|(1ULL<<5)|(1ULL<<7);
    ASSERT(x_bits[0] == expected, "Activation bits correct");
}

void test_small_popcount_gemv(void) {
    printf("\n--- Test: Small POPCOUNT GEMV (2x8) ---\n");

    ternary_matrix_t* tm = ternary_matrix_alloc(2, 8);
    int8_t dense[] = {1, -1, 0, 1, -1, 0, 1, -1,
                      0, 0, 1, -1, 0, 1, 0, -1};
    ternary_matrix_pack(tm, dense);

    popcount_matrix_t* pm = popcount_matrix_alloc(2, 8);
    popcount_matrix_from_ternary(pm, tm);

    float x[] = {0.1f, 0.5f, 0.3f, 0.9f, 0.2f, 0.7f, 0.4f, 0.8f};
    uint64_t x_bits[1];
    float threshold;
    popcount_quantize_activations(x, 8, x_bits, &threshold);

    float y[2];
    popcount_gemv(pm, x_bits, threshold, y, 2);

    printf("  y[0] = %.4f (expected -0.4875)\n", y[0]);
    printf("  y[1] = %.4f (expected -0.4875)\n", y[1]);
    ASSERT(fabsf(y[0] - (-0.4875f)) < 0.01f, "Row 0 GEMV correct");
    ASSERT(fabsf(y[1] - (-0.4875f)) < 0.01f, "Row 1 GEMV correct");

    popcount_matrix_free(pm);
    ternary_matrix_free(tm);
}

void test_consistency_with_standard_gemv(void) {
    printf("\n--- Test: Consistency with Standard GEMV ---\n");

    int32_t M = 16, N = 64;
    ternary_matrix_t* tm = ternary_matrix_alloc(M, N);

    int8_t* dense = (int8_t*)calloc(M * N, sizeof(int8_t));
    srand(42);
    for (int32_t i = 0; i < M * N; i++) {
        dense[i] = (int8_t)(rand() % 3 - 1);
    }
    ternary_matrix_pack(tm, dense);
    free(dense);

    float x[64];
    for (int32_t i = 0; i < N; i++) {
        x[i] = (float)(rand() % 100) / 100.0f;
    }

    float y_standard[16];
    ternary_gemv(y_standard, tm, x, NULL, M, N);

    popcount_matrix_t* pm = popcount_matrix_alloc(M, N);
    popcount_matrix_from_ternary(pm, tm);

    uint64_t x_bits[1];
    float threshold;
    popcount_quantize_activations(x, N, x_bits, &threshold);

    float y_popcount[16];
    popcount_gemv(pm, x_bits, threshold, y_popcount, M);

    int same_sign = 0;
    for (int32_t i = 0; i < M; i++) {
        if ((y_standard[i] > 0 && y_popcount[i] > 0) ||
            (y_standard[i] < 0 && y_popcount[i] < 0) ||
            (fabsf(y_standard[i]) < 0.01f && fabsf(y_popcount[i]) < 0.01f)) {
            same_sign++;
        }
    }
    float sign_agreement = (float)same_sign / M;
    printf("  Sign agreement: %.1f%% (%d/%d)\n", sign_agreement * 100, same_sign, M);
    ASSERT(sign_agreement >= 0.5f, "POPCOUNT agrees with standard GEMV on sign (>50%)");

    popcount_matrix_free(pm);
    ternary_matrix_free(tm);
}

void test_large_matrix(void) {
    printf("\n--- Test: Large Matrix (256x256) ---\n");

    int32_t M = 256, N = 256;
    ternary_matrix_t* tm = ternary_matrix_alloc(M, N);

    int8_t* dense = (int8_t*)calloc(M * N, sizeof(int8_t));
    srand(123);
    for (int32_t i = 0; i < M * N; i++) {
        dense[i] = (int8_t)(rand() % 3 - 1);
    }
    ternary_matrix_pack(tm, dense);
    free(dense);

    float x[256];
    for (int32_t i = 0; i < N; i++) {
        x[i] = (float)(rand() % 1000) / 1000.0f;
    }

    popcount_matrix_t* pm = popcount_matrix_alloc(M, N);
    popcount_matrix_from_ternary(pm, tm);

    int32_t words = (N + 63) / 64;
    uint64_t* x_bits = (uint64_t*)calloc(words, sizeof(uint64_t));
    float threshold;
    popcount_quantize_activations(x, N, x_bits, &threshold);

    float* y = (float*)calloc(M, sizeof(float));
    popcount_gemv(pm, x_bits, threshold, y, M);

    int finite_count = 0;
    for (int32_t i = 0; i < M; i++) {
        if (isfinite(y[i])) finite_count++;
    }
    ASSERT(finite_count == M, "All 256 outputs are finite");

    printf("  Sample outputs: y[0]=%.4f, y[127]=%.4f, y[255]=%.4f\n", y[0], y[127], y[255]);

    free(y);
    free(x_bits);
    popcount_matrix_free(pm);
    ternary_matrix_free(tm);
}

int main(void) {
    printf("=== POPCOUNT GEMV Tests ===\n");

    simd_detect_init();
    simd_print_caps();
    popcount_gemv_init();

    test_bitmap_conversion();
    test_activation_quantization();
    test_small_popcount_gemv();
    test_consistency_with_standard_gemv();
    test_large_matrix();

    printf("\n=== Results: %d/%d tests passed ===\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
