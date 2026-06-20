#include "../include/activations.h"
#include "../include/simd_detect.h"
#include "simd_math.h"
#include "ternary_types.h"
#include <math.h>
#include <string.h>

#ifdef ENGINE_AVX2
#include <immintrin.h>
#endif

typedef void (*rmsnorm_fn)(float*, const float*, const float*, int32_t, float);
typedef void (*unary_fn)(float*, const float*, int32_t);
typedef void (*binary_fn)(float*, const float*, const float*, int32_t);

static rmsnorm_fn g_rmsnorm = NULL;
static unary_fn g_silu = NULL;
static unary_fn g_relu = NULL;
static binary_fn g_mul = NULL;
static unary_fn g_softplus = NULL;
static binary_fn g_silu_mul = NULL;


void act_rmsnorm_scalar(float* y, const float* x, const float* weight, int32_t n, float eps)
{
    float sum_sq = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = 1.0f / sqrtf(sum_sq / n + eps);

    for (int32_t i = 0; i < n; i++) {
        y[i] = x[i] * weight[i] * rms;
    }
}

#ifdef ENGINE_AVX2
void act_rmsnorm_avx2(float* y, const float* x, const float* weight, int32_t n, float eps)
{
    const __m256 zero = _mm256_setzero_ps();
    __m256 acc0 = zero, acc1 = zero, acc2 = zero, acc3 = zero;

    int32_t i = 0;
    for (; i + 31 < n; i += 32) {
        for (int j = 0; j < 32; j += 8) {
            __m256 xv = _mm256_loadu_ps(&x[i + j]);
            __m256 sq = _mm256_mul_ps(xv, xv);
            acc0 = _mm256_add_ps(acc0, sq);
        }
    }

    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    float sum_sq = 0.0f;
    float buf[8];
    _mm256_storeu_ps(buf, acc);
    for (int j = 0; j < 8; j++) sum_sq += buf[j];

    for (; i < n; i++) {
        sum_sq += x[i] * x[i];
    }

    float rms = 1.0f / sqrtf(sum_sq / n + eps);
    __m256 rms_v = _mm256_set1_ps(rms);

    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xv = _mm256_loadu_ps(&x[i]);
        __m256 wv = _mm256_loadu_ps(&weight[i]);
        __m256 result = _mm256_mul_ps(_mm256_mul_ps(xv, wv), rms_v);
        _mm256_storeu_ps(&y[i], result);
    }

    for (; i < n; i++) {
        y[i] = x[i] * weight[i] * rms;
    }
}
#endif

void act_silu_scalar(float* y, const float* x, int32_t n)
{
    for (int32_t i = 0; i < n; i++) {
        float sigmoid = 1.0f / (1.0f + expf(-x[i]));
        y[i] = x[i] * sigmoid;
    }
}

#ifdef ENGINE_AVX2
void act_silu_avx2(float* y, const float* x, int32_t n)
{
    const __m256 one = _mm256_set1_ps(1.0f);

    int32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xv = _mm256_loadu_ps(&x[i]);
        __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), xv);
        __m256 exp_neg = fast_exp_avx2(neg_x);
        __m256 sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
        __m256 result = _mm256_mul_ps(xv, sigmoid);
        _mm256_storeu_ps(&y[i], result);
    }

    for (; i < n; i++) {
        float sigmoid = 1.0f / (1.0f + expf(-x[i]));
        y[i] = x[i] * sigmoid;
    }
}
#endif

void act_relu_scalar(float* y, const float* x, int32_t n)
{
    for (int32_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

#ifdef ENGINE_AVX2
void act_relu_avx2(float* y, const float* x, int32_t n)
{
    const __m256 zero = _mm256_setzero_ps();

    int32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xv = _mm256_loadu_ps(&x[i]);
        __m256 result = _mm256_max_ps(xv, zero);
        _mm256_storeu_ps(&y[i], result);
    }

    for (; i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}
#endif

void act_mul_scalar(float* y, const float* a, const float* b, int32_t n)
{
    for (int32_t i = 0; i < n; i++) {
        y[i] = a[i] * b[i];
    }
}

#ifdef ENGINE_AVX2
void act_mul_avx2(float* y, const float* a, const float* b, int32_t n)
{
    int32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 av = _mm256_loadu_ps(&a[i]);
        __m256 bv = _mm256_loadu_ps(&b[i]);
        __m256 result = _mm256_mul_ps(av, bv);
        _mm256_storeu_ps(&y[i], result);
    }

    for (; i < n; i++) {
        y[i] = a[i] * b[i];
    }
}
#endif

void act_softplus_scalar(float* y, const float* x, int32_t n)
{
    const float threshold = 20.0f;

    for (int32_t i = 0; i < n; i++) {
        if (x[i] > threshold) {
            y[i] = x[i];
        } else if (x[i] < -threshold) {
            y[i] = 0.0f;
        } else {
            y[i] = logf(1.0f + expf(x[i]));
        }
    }
}

#ifdef ENGINE_AVX2
void act_softplus_avx2(float* y, const float* x, int32_t n)
{
    const __m256 thresh_hi = _mm256_set1_ps(20.0f);
    const __m256 thresh_lo = _mm256_set1_ps(-20.0f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);

    int32_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 xv = _mm256_loadu_ps(&x[i]);

        __m256 exp_x = fast_exp_avx2(xv);
        __m256 one_plus = _mm256_add_ps(one, exp_x);
        __m256 log_val = fast_log_avx2(one_plus);

        __m256 mask_hi = _mm256_cmp_ps(xv, thresh_hi, _CMP_GT_OQ);
        __m256 mask_lo = _mm256_cmp_ps(xv, thresh_lo, _CMP_LT_OQ);
        __m256 result = _mm256_blendv_ps(log_val, xv, mask_hi);
        result = _mm256_blendv_ps(result, zero, mask_lo);

        _mm256_storeu_ps(&y[i], result);
    }

    for (; i < n; i++) {
        if (x[i] > 20.0f) {
            y[i] = x[i];
        } else if (x[i] < -20.0f) {
            y[i] = 0.0f;
        } else {
            y[i] = logf(1.0f + expf(x[i]));
        }
    }
}
#endif

void act_silu_mul_scalar(float* y, const float* a, const float* b, int32_t n)
{
    for (int32_t i = 0; i < n; i++) {
        float sigmoid = 1.0f / (1.0f + expf(-a[i]));
        y[i] = a[i] * sigmoid * b[i];
    }
}

#ifdef ENGINE_AVX2
void act_silu_mul_avx2(float* y, const float* a, const float* b, int32_t n)
{
    const __m256 one = _mm256_set1_ps(1.0f);

    int32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 av = _mm256_loadu_ps(&a[i]);
        __m256 bv = _mm256_loadu_ps(&b[i]);

        __m256 neg_a = _mm256_sub_ps(_mm256_setzero_ps(), av);
        __m256 exp_neg = fast_exp_avx2(neg_a);
        __m256 sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));

        __m256 silu_a = _mm256_mul_ps(av, sigmoid);
        __m256 result = _mm256_mul_ps(silu_a, bv);
        _mm256_storeu_ps(&y[i], result);
    }

    for (; i < n; i++) {
        float sigmoid = 1.0f / (1.0f + expf(-a[i]));
        y[i] = a[i] * sigmoid * b[i];
    }
}
#endif

void act_init(void)
{
    simd_detect_init();
    engine_backend_t backend = simd_get_backend();

    switch (backend) {
    #ifdef ENGINE_AVX2
    case ENGINE_BACKEND_AVX2:
        g_rmsnorm = act_rmsnorm_avx2;
        g_silu = act_silu_avx2;
        g_relu = act_relu_avx2;
        g_mul = act_mul_avx2;
        g_softplus = act_softplus_avx2;
        g_silu_mul = act_silu_mul_avx2;
        break;
    #endif
    default:
        g_rmsnorm = act_rmsnorm_scalar;
        g_silu = act_silu_scalar;
        g_relu = act_relu_scalar;
        g_mul = act_mul_scalar;
        g_softplus = act_softplus_scalar;
        g_silu_mul = act_silu_mul_scalar;
        break;
    }
}

void act_rmsnorm(float* y, const float* x, const float* weight, int32_t n, float eps)
{
    if (!g_rmsnorm) act_init();
    g_rmsnorm(y, x, weight, n, eps);
}

void act_silu(float* y, const float* x, int32_t n)
{
    if (!g_silu) act_init();
    g_silu(y, x, n);
}

void act_relu(float* y, const float* x, int32_t n)
{
    if (!g_relu) act_init();
    g_relu(y, x, n);
}

void act_mul(float* y, const float* a, const float* b, int32_t n)
{
    if (!g_mul) act_init();
    g_mul(y, a, b, n);
}

void act_softplus(float* y, const float* x, int32_t n)
{
    if (!g_softplus) act_init();
    g_softplus(y, x, n);
}

void act_silu_mul(float* y, const float* a, const float* b, int32_t n)
{
    if (!g_silu_mul) act_init();
    g_silu_mul(y, a, b, n);
}
