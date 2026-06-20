#ifndef POPCOUNT_GEMV_H
#define POPCOUNT_GEMV_H

#include "ternary_types.h"
#include <stdint.h>

typedef struct {
    uint64_t* val_bits;
    uint64_t* sign_bits;
    float* scales;
    float* bias;
    int32_t rows;
    int32_t cols;
    int32_t words_per_row;
} popcount_matrix_t;

popcount_matrix_t* popcount_matrix_alloc(int32_t rows, int32_t cols);
void popcount_matrix_free(popcount_matrix_t* mat);

void popcount_matrix_from_ternary(popcount_matrix_t* dst, const ternary_matrix_t* src);

void popcount_quantize_activations(const float* x, int32_t n, uint64_t* x_bits, float* threshold);

void popcount_gemv(const popcount_matrix_t* W, const uint64_t* x_bits,
                   float x_threshold, float* y, int32_t rows);

void popcount_gemv_scalar(const popcount_matrix_t* W, const uint64_t* x_bits,
                          float x_threshold, float* y, int32_t rows);

void popcount_gemv_init(void);

#endif
