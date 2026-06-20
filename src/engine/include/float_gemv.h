#ifndef FLOAT_GEMV_H
#define FLOAT_GEMV_H

#include <stdint.h>

/*
 * Dense float32 GEMV: y = W * x + bias
 *
 * W is a dense float matrix (rows x cols, row-major layout),
 * x is float input vector (cols),
 * y is float output vector (rows), bias is optional float vector (rows).
 *
 * Standard matrix-vector product: y[row] = sum(W[row][col] * x[col]) + bias[row]
 *
 * Used for dt_proj and other dense float projections in SSM architecture.
 *
 * Three implementations with runtime dispatch:
 *   1. Scalar (portable fallback)
 *   2. AVX2 (x86, 8 floats at a time with FMA)
 *   3. Runtime selection based on hardware capabilities
 */

/* Compute y = W * x + bias using best available kernel */
void float_gemv(
    float* y,           /* Output: [rows] */
    const float* W,     /* Weights: dense float [rows x cols], row-major */
    const float* x,     /* Input: [cols] */
    const float* bias,  /* Bias: [rows] or NULL */
    int32_t rows,
    int32_t cols
);

/* Individual kernel variants (for benchmarking) */
void float_gemv_scalar(
    float* y, const float* W, const float* x,
    const float* bias, int32_t rows, int32_t cols
);

#ifdef ENGINE_AVX2
void float_gemv_avx2(
    float* y, const float* W, const float* x,
    const float* bias, int32_t rows, int32_t cols
);
#endif

/* Initialize GEMV subsystem (call once, selects best kernel) */
void float_gemv_init(void);

#endif /* FLOAT_GEMV_H */
