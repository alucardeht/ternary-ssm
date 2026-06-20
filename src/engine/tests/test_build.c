#include "ternary_types.h"
#include "simd_detect.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  [TEST] %s... ", name)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)

void test_simd_detection(void) {
    TEST("SIMD detection");
    simd_detect_init();
    const simd_caps_t* caps = simd_get_caps();
    /* Just verify it doesn't crash and returns valid struct */
    (void)caps->has_avx2;
    (void)caps->has_avx512;
    (void)caps->has_neon;
    (void)caps->has_popcnt;
    PASS();

    simd_print_caps();
}

void test_ternary_encode_decode(void) {
    TEST("Ternary encode/decode");

    /* Test all 3 values */
    assert(ternary_decode(ternary_encode(0), 0) == 0);
    assert(ternary_decode(ternary_encode(1), 0) == 1);
    assert(ternary_decode(ternary_encode(-1), 0) == -1);

    /* Test pack4 and positional decode */
    uint8_t packed = ternary_pack4(1, -1, 0, 1);
    assert(ternary_decode(packed, 0) == 1);
    assert(ternary_decode(packed, 1) == -1);
    assert(ternary_decode(packed, 2) == 0);
    assert(ternary_decode(packed, 3) == 1);

    PASS();
}

void test_matrix_pack_unpack(void) {
    TEST("Matrix pack/unpack roundtrip");

    int32_t rows = 4, cols = 8;
    ternary_matrix_t* mat = ternary_matrix_alloc(rows, cols);
    assert(mat != NULL);

    /* Create test data */
    int8_t dense[32];
    for (int i = 0; i < 32; i++) {
        dense[i] = (int8_t)((i % 3) - 1);  /* -1, 0, 1, -1, 0, 1, ... */
    }

    /* Pack */
    ternary_matrix_pack(mat, dense);

    /* Unpack and verify */
    int8_t unpacked[32];
    ternary_matrix_unpack(mat, unpacked);

    int mismatch = 0;
    for (int i = 0; i < 32; i++) {
        if (dense[i] != unpacked[i]) mismatch++;
    }

    if (mismatch == 0) {
        PASS();
    } else {
        char msg[64];
        snprintf(msg, sizeof(msg), "%d mismatches in 32 values", mismatch);
        FAIL(msg);
    }

    ternary_matrix_free(mat);
}

void test_matrix_get_set(void) {
    TEST("Matrix get/set individual values");

    int32_t rows = 3, cols = 5;
    ternary_matrix_t* mat = ternary_matrix_alloc(rows, cols);
    assert(mat != NULL);

    /* Set specific values */
    ternary_matrix_set(mat, 0, 0, 1);
    ternary_matrix_set(mat, 0, 1, -1);
    ternary_matrix_set(mat, 0, 2, 0);
    ternary_matrix_set(mat, 1, 3, -1);
    ternary_matrix_set(mat, 2, 4, 1);

    /* Verify */
    int ok = 1;
    ok &= (ternary_matrix_get(mat, 0, 0) == 1);
    ok &= (ternary_matrix_get(mat, 0, 1) == -1);
    ok &= (ternary_matrix_get(mat, 0, 2) == 0);
    ok &= (ternary_matrix_get(mat, 1, 3) == -1);
    ok &= (ternary_matrix_get(mat, 2, 4) == 1);

    if (ok) {
        PASS();
    } else {
        FAIL("Value mismatch after set/get");
    }

    ternary_matrix_free(mat);
}

void test_non_aligned_size(void) {
    TEST("Non-aligned matrix size (not multiple of 4)");

    int32_t rows = 3, cols = 7;  /* 21 values, not multiple of 4 */
    ternary_matrix_t* mat = ternary_matrix_alloc(rows, cols);
    assert(mat != NULL);

    int8_t dense[21];
    for (int i = 0; i < 21; i++) {
        dense[i] = (int8_t)((i % 3) - 1);
    }

    ternary_matrix_pack(mat, dense);

    int8_t unpacked[21];
    ternary_matrix_unpack(mat, unpacked);

    int mismatch = 0;
    for (int i = 0; i < 21; i++) {
        if (dense[i] != unpacked[i]) mismatch++;
    }

    if (mismatch == 0) {
        PASS();
    } else {
        char msg[64];
        snprintf(msg, sizeof(msg), "%d mismatches in 21 values", mismatch);
        FAIL(msg);
    }

    ternary_matrix_free(mat);
}

int main(void) {
    printf("=== Ternary Engine Build Test ===\n\n");

    test_simd_detection();
    printf("\n");
    test_ternary_encode_decode();
    test_matrix_pack_unpack();
    test_matrix_get_set();
    test_non_aligned_size();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
