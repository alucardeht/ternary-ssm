#include "ternary_gemv.h"
#include "simd_detect.h"
#include "simd_math.h"
#include "ternary_types.h"
#include <string.h>
#include <math.h>

#ifdef ENGINE_AVX2
#include <immintrin.h>
#endif

#ifdef ENGINE_AVX512
#include <immintrin.h>
#endif

/*
 * LUT for decoding packed ternary bytes to bit masks.
 * Each byte = 4 weights (2 bits each, LSB first).
 * Encoding: 00=0, 01=+1, 10=-1, 11=reserved(0)
 *
 * pos_lut[byte] = 4-bit mask with 1s where weight = +1
 * neg_lut[byte] = 4-bit mask with 1s where weight = -1
 */

#define DECODE_POS_BIT(byte, i) (((((byte) >> ((i)*2)) & 0x03) == 0x01) ? (1 << (i)) : 0)
#define DECODE_NEG_BIT(byte, i) (((((byte) >> ((i)*2)) & 0x03) == 0x02) ? (1 << (i)) : 0)

#define POS_ENTRY(b) (uint8_t)(DECODE_POS_BIT(b,0) | DECODE_POS_BIT(b,1) | DECODE_POS_BIT(b,2) | DECODE_POS_BIT(b,3))
#define NEG_ENTRY(b) (uint8_t)(DECODE_NEG_BIT(b,0) | DECODE_NEG_BIT(b,1) | DECODE_NEG_BIT(b,2) | DECODE_NEG_BIT(b,3))

#define LUT16(F, base) \
    F(base+0),  F(base+1),  F(base+2),  F(base+3),  \
    F(base+4),  F(base+5),  F(base+6),  F(base+7),  \
    F(base+8),  F(base+9),  F(base+10), F(base+11), \
    F(base+12), F(base+13), F(base+14), F(base+15)

#define LUT256(F) \
    LUT16(F,0),   LUT16(F,16),  LUT16(F,32),  LUT16(F,48),  \
    LUT16(F,64),  LUT16(F,80),  LUT16(F,96),  LUT16(F,112), \
    LUT16(F,128), LUT16(F,144), LUT16(F,160), LUT16(F,176), \
    LUT16(F,192), LUT16(F,208), LUT16(F,224), LUT16(F,240)

static const uint8_t pos_lut[256] = { LUT256(POS_ENTRY) };
static const uint8_t neg_lut[256] = { LUT256(NEG_ENTRY) };


/* Optimized scalar: sequential byte access, inline decode, no function calls */
void ternary_gemv_scalar(
    float* y, const ternary_matrix_t* W, const float* x,
    const float* bias, int32_t rows, int32_t cols)
{
    const int32_t bytes_per_row = cols / 4;

    for (int32_t r = 0; r < rows; r++) {
        float acc = 0.0f;
        const uint8_t* row_data = W->data + (size_t)r * bytes_per_row;

        int32_t c = 0;
        for (int32_t b = 0; b < bytes_per_row; b++) {
            const uint8_t packed = row_data[b];
            uint8_t code;

            code = packed & 0x03;
            if (code == 0x01)      acc += x[c];
            else if (code == 0x02) acc -= x[c];
            c++;

            code = (packed >> 2) & 0x03;
            if (code == 0x01)      acc += x[c];
            else if (code == 0x02) acc -= x[c];
            c++;

            code = (packed >> 4) & 0x03;
            if (code == 0x01)      acc += x[c];
            else if (code == 0x02) acc -= x[c];
            c++;

            code = (packed >> 6) & 0x03;
            if (code == 0x01)      acc += x[c];
            else if (code == 0x02) acc -= x[c];
            c++;
        }

        y[r] = acc * W->scales[r] + (bias ? bias[r] : 0.0f);
    }
}


#ifdef ENGINE_AVX2


/*
 * AVX2 GEMV with masked add/subtract — ZERO float multiplies.
 *
 * For each 8 weights:
 *   1. LUT lookup → 8-bit pos_mask + neg_mask
 *   2. Expand masks to 8x32-bit via broadcast + AND + cmpeq
 *   3. AND input with masks → select relevant x values
 *   4. ADD positives, SUB negatives
 *
 * Main loop processes 16 weights (4 bytes) per iteration with
 * two independent accumulators for ILP.
 */
void ternary_gemv_avx2(
    float* y, const ternary_matrix_t* W, const float* x,
    const float* bias, int32_t rows, int32_t cols)
{
    const __m256i bit_select = _mm256_set_epi32(
        0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01
    );

    const int32_t bytes_per_row = cols / 4;

    for (int32_t r = 0; r < rows; r++) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        const uint8_t* row_data = W->data + (size_t)r * bytes_per_row;

        int32_t c = 0;
        int32_t b = 0;

        /* Main loop: 16 weights per iteration */
        for (; c + 15 < cols; c += 16, b += 4) {
            __m256 xv0 = _mm256_loadu_ps(&x[c]);
            __m256 xv1 = _mm256_loadu_ps(&x[c + 8]);

            uint8_t b0 = row_data[b];
            uint8_t b1 = row_data[b + 1];
            uint8_t b2 = row_data[b + 2];
            uint8_t b3 = row_data[b + 3];

            /* First 8 weights → acc0 */
            {
                uint32_t pm = (uint32_t)pos_lut[b0] | ((uint32_t)pos_lut[b1] << 4);
                uint32_t nm = (uint32_t)neg_lut[b0] | ((uint32_t)neg_lut[b1] << 4);

                __m256i p_bcast = _mm256_set1_epi32((int32_t)pm);
                __m256i p_bits  = _mm256_and_si256(p_bcast, bit_select);
                __m256i p_cmp   = _mm256_cmpeq_epi32(p_bits, bit_select);
                __m256  p_fmask = _mm256_castsi256_ps(p_cmp);

                __m256i n_bcast = _mm256_set1_epi32((int32_t)nm);
                __m256i n_bits  = _mm256_and_si256(n_bcast, bit_select);
                __m256i n_cmp   = _mm256_cmpeq_epi32(n_bits, bit_select);
                __m256  n_fmask = _mm256_castsi256_ps(n_cmp);

                acc0 = _mm256_add_ps(acc0, _mm256_and_ps(xv0, p_fmask));
                acc0 = _mm256_sub_ps(acc0, _mm256_and_ps(xv0, n_fmask));
            }

            /* Second 8 weights → acc1 */
            {
                uint32_t pm = (uint32_t)pos_lut[b2] | ((uint32_t)pos_lut[b3] << 4);
                uint32_t nm = (uint32_t)neg_lut[b2] | ((uint32_t)neg_lut[b3] << 4);

                __m256i p_bcast = _mm256_set1_epi32((int32_t)pm);
                __m256i p_bits  = _mm256_and_si256(p_bcast, bit_select);
                __m256i p_cmp   = _mm256_cmpeq_epi32(p_bits, bit_select);
                __m256  p_fmask = _mm256_castsi256_ps(p_cmp);

                __m256i n_bcast = _mm256_set1_epi32((int32_t)nm);
                __m256i n_bits  = _mm256_and_si256(n_bcast, bit_select);
                __m256i n_cmp   = _mm256_cmpeq_epi32(n_bits, bit_select);
                __m256  n_fmask = _mm256_castsi256_ps(n_cmp);

                acc1 = _mm256_add_ps(acc1, _mm256_and_ps(xv1, p_fmask));
                acc1 = _mm256_sub_ps(acc1, _mm256_and_ps(xv1, n_fmask));
            }
        }

        /* Remainder: 8 weights */
        for (; c + 7 < cols; c += 8, b += 2) {
            __m256 xv = _mm256_loadu_ps(&x[c]);

            uint32_t pm = (uint32_t)pos_lut[row_data[b]] | ((uint32_t)pos_lut[row_data[b+1]] << 4);
            uint32_t nm = (uint32_t)neg_lut[row_data[b]] | ((uint32_t)neg_lut[row_data[b+1]] << 4);

            __m256i p_bcast = _mm256_set1_epi32((int32_t)pm);
            __m256i p_bits  = _mm256_and_si256(p_bcast, bit_select);
            __m256i p_cmp   = _mm256_cmpeq_epi32(p_bits, bit_select);
            __m256  p_fmask = _mm256_castsi256_ps(p_cmp);

            __m256i n_bcast = _mm256_set1_epi32((int32_t)nm);
            __m256i n_bits  = _mm256_and_si256(n_bcast, bit_select);
            __m256i n_cmp   = _mm256_cmpeq_epi32(n_bits, bit_select);
            __m256  n_fmask = _mm256_castsi256_ps(n_cmp);

            acc0 = _mm256_add_ps(acc0, _mm256_and_ps(xv, p_fmask));
            acc0 = _mm256_sub_ps(acc0, _mm256_and_ps(xv, n_fmask));
        }

        /* Combine accumulators + horizontal sum */
        __m256 acc_total = _mm256_add_ps(acc0, acc1);
        float sum = hsum_avx2(acc_total);

        /* Scalar tail for remaining < 8 columns */
        for (; c < cols; c++) {
            int8_t w = ternary_matrix_get(W, r, c);
            if (w == 1)       sum += x[c];
            else if (w == -1) sum -= x[c];
        }

        y[r] = sum * W->scales[r] + (bias ? bias[r] : 0.0f);
    }
}

#endif /* ENGINE_AVX2 */


#ifdef ENGINE_AVX512

static void ternary_gemv_avx512(
    float* y, const ternary_matrix_t* W, const float* x,
    const float* bias, int32_t rows, int32_t cols)
{
    const int32_t bytes_per_row = cols / 4;

    for (int32_t r = 0; r < rows; r++) {
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        const uint8_t* row_data = W->data + (size_t)r * bytes_per_row;

        int32_t c = 0, b = 0;

        for (; c + 31 < cols; c += 32, b += 8) {
            __m512 xv0 = _mm512_loadu_ps(&x[c]);
            __m512 xv1 = _mm512_loadu_ps(&x[c + 16]);

            __mmask16 pm0 = (__mmask16)(
                (uint16_t)pos_lut[row_data[b]]   |
                ((uint16_t)pos_lut[row_data[b+1]] << 4) |
                ((uint16_t)pos_lut[row_data[b+2]] << 8) |
                ((uint16_t)pos_lut[row_data[b+3]] << 12)
            );
            __mmask16 nm0 = (__mmask16)(
                (uint16_t)neg_lut[row_data[b]]   |
                ((uint16_t)neg_lut[row_data[b+1]] << 4) |
                ((uint16_t)neg_lut[row_data[b+2]] << 8) |
                ((uint16_t)neg_lut[row_data[b+3]] << 12)
            );

            acc0 = _mm512_mask_add_ps(acc0, pm0, acc0, xv0);
            acc0 = _mm512_mask_sub_ps(acc0, nm0, acc0, xv0);

            __mmask16 pm1 = (__mmask16)(
                (uint16_t)pos_lut[row_data[b+4]] |
                ((uint16_t)pos_lut[row_data[b+5]] << 4) |
                ((uint16_t)pos_lut[row_data[b+6]] << 8) |
                ((uint16_t)pos_lut[row_data[b+7]] << 12)
            );
            __mmask16 nm1 = (__mmask16)(
                (uint16_t)neg_lut[row_data[b+4]] |
                ((uint16_t)neg_lut[row_data[b+5]] << 4) |
                ((uint16_t)neg_lut[row_data[b+6]] << 8) |
                ((uint16_t)neg_lut[row_data[b+7]] << 12)
            );

            acc1 = _mm512_mask_add_ps(acc1, pm1, acc1, xv1);
            acc1 = _mm512_mask_sub_ps(acc1, nm1, acc1, xv1);
        }

        for (; c + 15 < cols; c += 16, b += 4) {
            __m512 xv = _mm512_loadu_ps(&x[c]);
            __mmask16 pm = (__mmask16)(
                (uint16_t)pos_lut[row_data[b]]   |
                ((uint16_t)pos_lut[row_data[b+1]] << 4) |
                ((uint16_t)pos_lut[row_data[b+2]] << 8) |
                ((uint16_t)pos_lut[row_data[b+3]] << 12)
            );
            __mmask16 nm = (__mmask16)(
                (uint16_t)neg_lut[row_data[b]]   |
                ((uint16_t)neg_lut[row_data[b+1]] << 4) |
                ((uint16_t)neg_lut[row_data[b+2]] << 8) |
                ((uint16_t)neg_lut[row_data[b+3]] << 12)
            );
            acc0 = _mm512_mask_add_ps(acc0, pm, acc0, xv);
            acc0 = _mm512_mask_sub_ps(acc0, nm, acc0, xv);
        }

        __m512 total = _mm512_add_ps(acc0, acc1);
        float sum = _mm512_reduce_add_ps(total);

        for (; c < cols; c++) {
            uint8_t byte_val = row_data[c / 4];
            int shift = (c % 4) * 2;
            uint8_t bits = (byte_val >> shift) & 0x03;
            if (bits == 0x01) sum += x[c];
            else if (bits == 0x02) sum -= x[c];
        }

        y[r] = sum * W->scales[r] + (bias ? bias[r] : 0.0f);
    }
}

#endif /* ENGINE_AVX512 */


typedef void (*gemv_fn)(float*, const ternary_matrix_t*, const float*,
                        const float*, int32_t, int32_t);

static gemv_fn g_gemv_kernel = NULL;

void ternary_gemv_init(void)
{
    simd_detect_init();
    engine_backend_t backend = simd_get_backend();

    switch (backend) {
#ifdef ENGINE_AVX512
    case ENGINE_BACKEND_AVX512:
        g_gemv_kernel = ternary_gemv_avx512;
        break;
#endif
#ifdef ENGINE_AVX2
    case ENGINE_BACKEND_AVX2:
        g_gemv_kernel = ternary_gemv_avx2;
        break;
#endif
    default:
        g_gemv_kernel = ternary_gemv_scalar;
        break;
    }
}

void ternary_gemv(
    float* y, const ternary_matrix_t* W, const float* x,
    const float* bias, int32_t rows, int32_t cols)
{
    if (!g_gemv_kernel) {
        ternary_gemv_init();
    }
    g_gemv_kernel(y, W, x, bias, rows, cols);
}
