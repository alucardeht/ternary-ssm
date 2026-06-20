#include "popcount_gemv.h"
#include "simd_detect.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static inline int popcount64_portable(uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (int)((x * 0x0101010101010101ULL) >> 56);
}

#if defined(_MSC_VER)
#include <intrin.h>
static inline int popcount64_hw(uint64_t x) {
    return (int)__popcnt64(x);
}
#elif defined(__GNUC__) || defined(__clang__)
static inline int popcount64_hw(uint64_t x) {
    return __builtin_popcountll(x);
}
#else
static inline int popcount64_hw(uint64_t x) {
    return popcount64_portable(x);
}
#endif

popcount_matrix_t* popcount_matrix_alloc(int32_t rows, int32_t cols) {
    popcount_matrix_t* mat = (popcount_matrix_t*)calloc(1, sizeof(popcount_matrix_t));
    if (!mat) return NULL;

    mat->rows = rows;
    mat->cols = cols;
    mat->words_per_row = (cols + 63) / 64;

    size_t bitmap_size = (size_t)rows * mat->words_per_row * sizeof(uint64_t);
    mat->val_bits = (uint64_t*)calloc(1, bitmap_size);
    mat->sign_bits = (uint64_t*)calloc(1, bitmap_size);
    mat->scales = (float*)calloc(rows, sizeof(float));
    mat->bias = NULL;

    if (!mat->val_bits || !mat->sign_bits || !mat->scales) {
        popcount_matrix_free(mat);
        return NULL;
    }

    for (int32_t i = 0; i < rows; i++) {
        mat->scales[i] = 1.0f;
    }

    return mat;
}

void popcount_matrix_free(popcount_matrix_t* mat) {
    if (!mat) return;
    free(mat->val_bits);
    free(mat->sign_bits);
    free(mat->scales);
    free(mat->bias);
    free(mat);
}

void popcount_matrix_from_ternary(popcount_matrix_t* dst, const ternary_matrix_t* src) {
    for (int32_t r = 0; r < dst->rows && r < src->rows; r++) {
        int32_t wpr = dst->words_per_row;
        uint64_t* val_row = dst->val_bits + (size_t)r * wpr;
        uint64_t* sign_row = dst->sign_bits + (size_t)r * wpr;

        memset(val_row, 0, wpr * sizeof(uint64_t));
        memset(sign_row, 0, wpr * sizeof(uint64_t));

        for (int32_t c = 0; c < dst->cols && c < src->cols; c++) {
            int32_t byte_idx = c / 4;
            int32_t bit_pos = (c % 4) * 2;
            uint8_t packed_byte = src->data[r * ((src->cols + 3) / 4) + byte_idx];
            uint8_t code = (packed_byte >> bit_pos) & 0x03;

            int32_t word_idx = c / 64;
            int32_t bit_idx = c % 64;

            if (code == 0x01) {
                val_row[word_idx] |= (1ULL << bit_idx);
            } else if (code == 0x02) {
                val_row[word_idx] |= (1ULL << bit_idx);
                sign_row[word_idx] |= (1ULL << bit_idx);
            }
        }

        dst->scales[r] = src->scales[r];
    }
}

void popcount_quantize_activations(const float* x, int32_t n, uint64_t* x_bits, float* threshold) {
    float sum = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        sum += x[i];
    }
    *threshold = sum / (float)n;

    int32_t words = (n + 63) / 64;
    memset(x_bits, 0, words * sizeof(uint64_t));

    for (int32_t i = 0; i < n; i++) {
        if (x[i] >= *threshold) {
            x_bits[i / 64] |= (1ULL << (i % 64));
        }
    }
}

void popcount_gemv_scalar(const popcount_matrix_t* W, const uint64_t* x_bits,
                          float x_threshold, float* y, int32_t rows) {
    for (int32_t r = 0; r < rows; r++) {
        int32_t wpr = W->words_per_row;
        const uint64_t* val_row = W->val_bits + (size_t)r * wpr;
        const uint64_t* sign_row = W->sign_bits + (size_t)r * wpr;

        int32_t net = 0;
        for (int32_t w = 0; w < wpr; w++) {
            uint64_t active = x_bits[w] & val_row[w];
            int32_t P = popcount64_portable(active);
            int32_t N = popcount64_portable(active & sign_row[w]);
            net += (P - 2 * N);
        }

        y[r] = W->scales[r] * (float)net * x_threshold;
        if (W->bias) {
            y[r] += W->bias[r];
        }
    }
}

static void popcount_gemv_hw(const popcount_matrix_t* W, const uint64_t* x_bits,
                             float x_threshold, float* y, int32_t rows) {
    for (int32_t r = 0; r < rows; r++) {
        int32_t wpr = W->words_per_row;
        const uint64_t* val_row = W->val_bits + (size_t)r * wpr;
        const uint64_t* sign_row = W->sign_bits + (size_t)r * wpr;

        int32_t net = 0;
        for (int32_t w = 0; w < wpr; w++) {
            uint64_t active = x_bits[w] & val_row[w];
            int32_t P = popcount64_hw(active);
            int32_t N = popcount64_hw(active & sign_row[w]);
            net += (P - 2 * N);
        }

        y[r] = W->scales[r] * (float)net * x_threshold;
        if (W->bias) {
            y[r] += W->bias[r];
        }
    }
}

typedef void (*popcount_gemv_fn)(const popcount_matrix_t*, const uint64_t*, float, float*, int32_t);
static popcount_gemv_fn g_popcount_gemv_kernel = NULL;

void popcount_gemv_init(void) {
    simd_detect_init();
    const simd_caps_t* caps = simd_get_caps();

    if (caps->has_popcnt) {
        g_popcount_gemv_kernel = popcount_gemv_hw;
    } else {
        g_popcount_gemv_kernel = popcount_gemv_scalar;
    }
}

void popcount_gemv(const popcount_matrix_t* W, const uint64_t* x_bits,
                   float x_threshold, float* y, int32_t rows) {
    if (!g_popcount_gemv_kernel) {
        popcount_gemv_init();
    }
    g_popcount_gemv_kernel(W, x_bits, x_threshold, y, rows);
}
