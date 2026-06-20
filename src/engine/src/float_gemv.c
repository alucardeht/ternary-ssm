#include "float_gemv.h"
#include "simd_detect.h"
#include "simd_math.h"
#include "ternary_types.h"
#include <string.h>
#include <math.h>

#ifdef ENGINE_AVX2
#include <immintrin.h>
#endif


void float_gemv_scalar(
    float* y, const float* W, const float* x,
    const float* bias, int32_t rows, int32_t cols)
{
    for (int32_t r = 0; r < rows; r++) {
        float acc = 0.0f;

        for (int32_t c = 0; c < cols; c++) {
            acc += W[r * cols + c] * x[c];
        }

        y[r] = acc + (bias ? bias[r] : 0.0f);
    }
}


#ifdef ENGINE_AVX2

void float_gemv_avx2(
    float* y, const float* W, const float* x,
    const float* bias, int32_t rows, int32_t cols)
{
    /*
     * Process 8 columns at a time using AVX2.
     *
     * For each row:
     *   Use 4 accumulators to hide FMA latency
     *   Load 8 x values and 8 W values
     *   Use FMA (fused multiply-add) to accumulate: acc += W * x
     *   Sum the 8 accumulators horizontally
     *   Handle remaining columns with scalar fallback
     *   Apply bias
     */

    for (int32_t r = 0; r < rows; r++) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        int32_t c = 0;
        const float* row_ptr = &W[r * cols];

        /* Process 32 columns at a time (4 groups of 8) */
        for (; c + 31 < cols; c += 32) {
            /* Load and accumulate group 0 (columns c to c+7) */
            __m256 xv0 = _mm256_loadu_ps(&x[c]);
            __m256 wv0 = _mm256_loadu_ps(&row_ptr[c]);
            acc0 = _mm256_fmadd_ps(wv0, xv0, acc0);

            /* Load and accumulate group 1 (columns c+8 to c+15) */
            __m256 xv1 = _mm256_loadu_ps(&x[c + 8]);
            __m256 wv1 = _mm256_loadu_ps(&row_ptr[c + 8]);
            acc1 = _mm256_fmadd_ps(wv1, xv1, acc1);

            /* Load and accumulate group 2 (columns c+16 to c+23) */
            __m256 xv2 = _mm256_loadu_ps(&x[c + 16]);
            __m256 wv2 = _mm256_loadu_ps(&row_ptr[c + 16]);
            acc2 = _mm256_fmadd_ps(wv2, xv2, acc2);

            /* Load and accumulate group 3 (columns c+24 to c+31) */
            __m256 xv3 = _mm256_loadu_ps(&x[c + 24]);
            __m256 wv3 = _mm256_loadu_ps(&row_ptr[c + 24]);
            acc3 = _mm256_fmadd_ps(wv3, xv3, acc3);
        }

        /* Process 8 columns at a time for remainder */
        for (; c + 7 < cols; c += 8) {
            __m256 xv = _mm256_loadu_ps(&x[c]);
            __m256 wv = _mm256_loadu_ps(&row_ptr[c]);
            acc0 = _mm256_fmadd_ps(wv, xv, acc0);
        }

        /* Horizontal sum: add all accumulators and their internal 8 floats */
        __m256 sum_vec = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                       _mm256_add_ps(acc2, acc3));
        float result[8];
        _mm256_storeu_ps(result, sum_vec);
        float sum = result[0] + result[1] + result[2] + result[3]
                  + result[4] + result[5] + result[6] + result[7];

        /* Handle remaining columns (< 8) with scalar fallback */
        for (; c < cols; c++) {
            sum += row_ptr[c] * x[c];
        }

        /* Apply bias */
        y[r] = sum + (bias ? bias[r] : 0.0f);
    }
}

#endif /* ENGINE_AVX2 */


/* Function pointer for runtime dispatch */
typedef void (*gemv_fn)(float*, const float*, const float*,
                        const float*, int32_t, int32_t);

static gemv_fn g_float_gemv_kernel = NULL;

void float_gemv_init(void)
{
    simd_detect_init();
    engine_backend_t backend = simd_get_backend();

    switch (backend) {
#ifdef ENGINE_AVX2
    case ENGINE_BACKEND_AVX2:
        g_float_gemv_kernel = float_gemv_avx2;
        break;
#endif
    default:
        g_float_gemv_kernel = float_gemv_scalar;
        break;
    }
}

void float_gemv(
    float* y, const float* W, const float* x,
    const float* bias, int32_t rows, int32_t cols)
{
    if (!g_float_gemv_kernel) {
        float_gemv_init();
    }
    g_float_gemv_kernel(y, W, x, bias, rows, cols);
}
