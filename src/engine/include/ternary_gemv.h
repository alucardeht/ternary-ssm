#ifndef TERNARY_GEMV_H
#define TERNARY_GEMV_H

#include "ternary_types.h"

/*
 * Ternary GEMV: y = W * x + bias
 *
 * W is a packed ternary matrix (rows x cols), x is float input vector (cols),
 * y is float output vector (rows), bias is optional float vector (rows).
 *
 * Instead of float multiply-accumulate, uses conditional add/subtract:
 *   for each row: acc = 0; for each col: if W==+1: acc+=x; elif W==-1: acc-=x;
 *   y[row] = acc * scale[row] + bias[row]
 *
 * Three implementations with runtime dispatch:
 *   1. Scalar (portable fallback)
 *   2. AVX2 (x86, 8 floats at a time)
 *   3. Runtime selection based on hardware capabilities
 */

/* Compute y = W * x + bias using best available kernel */
void ternary_gemv(
    float* y,                    /* Output: [rows] */
    const ternary_matrix_t* W,   /* Weights: packed ternary [rows x cols] */
    const float* x,              /* Input: [cols] */
    const float* bias,           /* Bias: [rows] or NULL */
    int32_t rows,
    int32_t cols
);

/* Individual kernel variants (for benchmarking) */
void ternary_gemv_scalar(
    float* y, const ternary_matrix_t* W, const float* x,
    const float* bias, int32_t rows, int32_t cols
);

#ifdef ENGINE_AVX2
void ternary_gemv_avx2(
    float* y, const ternary_matrix_t* W, const float* x,
    const float* bias, int32_t rows, int32_t cols
);
#endif

/* Initialize GEMV subsystem (call once, selects best kernel) */
void ternary_gemv_init(void);

#endif /* TERNARY_GEMV_H */
