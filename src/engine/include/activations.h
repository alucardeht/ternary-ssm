#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <stdint.h>

/*
 * Activation function kernels: RMSNorm, SiLU, ReLU, element-wise multiply, softplus.
 *
 * Three implementations with runtime dispatch:
 *   1. Scalar (portable fallback)
 *   2. AVX2 (x86, 8 floats at a time)
 *   3. Runtime selection based on hardware capabilities
 *
 * All functions operate on float* arrays, use int32_t n for vector size.
 * In-place safe: y can alias x.
 */

/* RMSNorm: y[i] = x[i] * weight[i] * rsqrt(mean(x^2) + eps) */
void act_rmsnorm(float* y, const float* x, const float* weight, int32_t n, float eps);
void act_rmsnorm_scalar(float* y, const float* x, const float* weight, int32_t n, float eps);
#ifdef ENGINE_AVX2
void act_rmsnorm_avx2(float* y, const float* x, const float* weight, int32_t n, float eps);
#endif

/* SiLU: y[i] = x[i] / (1 + exp(-x[i])) */
void act_silu(float* y, const float* x, int32_t n);
void act_silu_scalar(float* y, const float* x, int32_t n);
#ifdef ENGINE_AVX2
void act_silu_avx2(float* y, const float* x, int32_t n);
#endif

/* ReLU: y[i] = max(0, x[i]) */
void act_relu(float* y, const float* x, int32_t n);
void act_relu_scalar(float* y, const float* x, int32_t n);
#ifdef ENGINE_AVX2
void act_relu_avx2(float* y, const float* x, int32_t n);
#endif

/* Element-wise multiply: y[i] = a[i] * b[i] */
void act_mul(float* y, const float* a, const float* b, int32_t n);
void act_mul_scalar(float* y, const float* a, const float* b, int32_t n);
#ifdef ENGINE_AVX2
void act_mul_avx2(float* y, const float* a, const float* b, int32_t n);
#endif

/* Softplus: y[i] = log(1 + exp(x[i])), with threshold: x>20 -> x, x<-20 -> 0 */
void act_softplus(float* y, const float* x, int32_t n);
void act_softplus_scalar(float* y, const float* x, int32_t n);
#ifdef ENGINE_AVX2
void act_softplus_avx2(float* y, const float* x, int32_t n);
#endif

/* SiLU-gated multiply (fused): y[i] = silu(a[i]) * b[i] */
void act_silu_mul(float* y, const float* a, const float* b, int32_t n);
void act_silu_mul_scalar(float* y, const float* a, const float* b, int32_t n);
#ifdef ENGINE_AVX2
void act_silu_mul_avx2(float* y, const float* a, const float* b, int32_t n);
#endif

/* Initialize activation subsystem (call once, selects best kernel) */
void act_init(void);

#endif /* ACTIVATIONS_H */
