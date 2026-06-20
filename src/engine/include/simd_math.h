#ifndef SIMD_MATH_H
#define SIMD_MATH_H

/* Shared SIMD math helpers for all backends */

/* ========== AVX2 ========== */
#ifdef ENGINE_AVX2
#include <immintrin.h>

static inline float hsum_avx2(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sum64 = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sum64);
    __m128 sum32 = _mm_add_ss(sum64, shuf);
    return _mm_cvtss_f32(sum32);
}

static inline __m256 fast_exp_avx2(__m256 x) {
    const __m256 max_val = _mm256_set1_ps(88.0f);
    const __m256 min_val = _mm256_set1_ps(-88.0f);
    const __m256 inv_ln2 = _mm256_set1_ps(1.4426950408f);

    x = _mm256_max_ps(_mm256_min_ps(x, max_val), min_val);
    __m256 t = _mm256_mul_ps(x, inv_ln2);
    __m256i k_int = _mm256_cvttps_epi32(t);
    __m256 k = _mm256_cvtepi32_ps(k_int);
    __m256 r = _mm256_sub_ps(t, k);

    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(0.693147f);
    const __m256 c2 = _mm256_set1_ps(0.240227f);
    const __m256 c3 = _mm256_set1_ps(0.055504f);
    const __m256 c4 = _mm256_set1_ps(0.009618f);
    const __m256 c5 = _mm256_set1_ps(0.001333f);

    __m256 p = c5;
    p = _mm256_fmadd_ps(p, r, c4);
    p = _mm256_fmadd_ps(p, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, c1);
    p = _mm256_fmadd_ps(p, r, c0);

    __m256i ki = _mm256_slli_epi32(k_int, 23);
    __m256i pow2k_bits = _mm256_add_epi32(ki, _mm256_set1_epi32(0x3F800000));
    __m256 pow2k = _mm256_castsi256_ps(pow2k_bits);

    return _mm256_mul_ps(p, pow2k);
}

static inline __m256 fast_log_avx2(__m256 x) {
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 one = _mm256_set1_ps(1.0f);

    __m256i xi = _mm256_castps_si256(x);
    __m256i exponent = _mm256_srli_epi32(xi, 23);
    exponent = _mm256_sub_epi32(exponent, _mm256_set1_epi32(127));
    __m256 e = _mm256_cvtepi32_ps(exponent);

    /* Extract mantissa in [1, 2) */
    __m256i mantissa = _mm256_and_si256(xi, _mm256_set1_epi32(0x007FFFFF));
    mantissa = _mm256_or_si256(mantissa, _mm256_set1_epi32(0x3F800000));
    __m256 m = _mm256_castsi256_ps(mantissa);

    /* Range reduction: if m > sqrt(2), halve m and increment exponent.
       This constrains f to [-0.293, 0.414] instead of [0, 1) */
    __m256 mask = _mm256_cmp_ps(m, _mm256_set1_ps(1.41421356f), _CMP_GT_OQ);
    m = _mm256_blendv_ps(m, _mm256_mul_ps(m, _mm256_set1_ps(0.5f)), mask);
    e = _mm256_add_ps(e, _mm256_and_ps(mask, one));

    __m256 f = _mm256_sub_ps(m, one);

    /* 6th-degree minimax polynomial for ln(1+f) */
    const __m256 a1 = _mm256_set1_ps( 0.9999999995f);
    const __m256 a2 = _mm256_set1_ps(-0.4999999668f);
    const __m256 a3 = _mm256_set1_ps( 0.3333330688f);
    const __m256 a4 = _mm256_set1_ps(-0.2499998808f);
    const __m256 a5 = _mm256_set1_ps( 0.2000062703f);
    const __m256 a6 = _mm256_set1_ps(-0.1666663892f);

    __m256 p = a6;
    p = _mm256_fmadd_ps(p, f, a5);
    p = _mm256_fmadd_ps(p, f, a4);
    p = _mm256_fmadd_ps(p, f, a3);
    p = _mm256_fmadd_ps(p, f, a2);
    p = _mm256_fmadd_ps(p, f, a1);
    p = _mm256_mul_ps(p, f);

    return _mm256_fmadd_ps(e, ln2, p);
}
#endif /* ENGINE_AVX2 */

/* ========== AVX-512 ========== */
#ifdef ENGINE_AVX512
#include <immintrin.h>

static inline float hsum_avx512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}

static inline __m512 fast_exp_avx512(__m512 x) {
    const __m512 max_val = _mm512_set1_ps(88.0f);
    const __m512 min_val = _mm512_set1_ps(-88.0f);
    const __m512 inv_ln2 = _mm512_set1_ps(1.4426950408f);

    x = _mm512_max_ps(_mm512_min_ps(x, max_val), min_val);
    __m512 t = _mm512_mul_ps(x, inv_ln2);
    __m512i k_int = _mm512_cvttps_epi32(t);
    __m512 k = _mm512_cvtepi32_ps(k_int);
    __m512 r = _mm512_sub_ps(t, k);

    const __m512 c0 = _mm512_set1_ps(1.0f);
    const __m512 c1 = _mm512_set1_ps(0.693147f);
    const __m512 c2 = _mm512_set1_ps(0.240227f);
    const __m512 c3 = _mm512_set1_ps(0.055504f);
    const __m512 c4 = _mm512_set1_ps(0.009618f);
    const __m512 c5 = _mm512_set1_ps(0.001333f);

    __m512 p = c5;
    p = _mm512_fmadd_ps(p, r, c4);
    p = _mm512_fmadd_ps(p, r, c3);
    p = _mm512_fmadd_ps(p, r, c2);
    p = _mm512_fmadd_ps(p, r, c1);
    p = _mm512_fmadd_ps(p, r, c0);

    __m512i ki = _mm512_slli_epi32(k_int, 23);
    __m512i pow2k_bits = _mm512_add_epi32(ki, _mm512_set1_epi32(0x3F800000));
    __m512 pow2k = _mm512_castsi512_ps(pow2k_bits);

    return _mm512_mul_ps(p, pow2k);
}
#endif /* ENGINE_AVX512 */

/* ========== NEON ========== */
#ifdef ENGINE_NEON
#include <arm_neon.h>

static inline float hsum_neon(float32x4_t v) {
    float32x2_t sum = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    sum = vpadd_f32(sum, sum);
    return vget_lane_f32(sum, 0);
}

static inline float32x4_t fast_exp_neon(float32x4_t x) {
    const float32x4_t max_val = vdupq_n_f32(88.0f);
    const float32x4_t min_val = vdupq_n_f32(-88.0f);
    const float32x4_t inv_ln2 = vdupq_n_f32(1.4426950408f);

    x = vmaxq_f32(vminq_f32(x, max_val), min_val);
    float32x4_t t = vmulq_f32(x, inv_ln2);
    int32x4_t k_int = vcvtq_s32_f32(t);
    float32x4_t k = vcvtq_f32_s32(k_int);
    float32x4_t r = vsubq_f32(t, k);

    float32x4_t p = vdupq_n_f32(0.001333f);
    p = vfmaq_f32(vdupq_n_f32(0.009618f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.055504f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.240227f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.693147f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);

    int32x4_t ki = vshlq_n_s32(k_int, 23);
    int32x4_t pow2k_bits = vaddq_s32(ki, vdupq_n_s32(0x3F800000));
    float32x4_t pow2k = vreinterpretq_f32_s32(pow2k_bits);

    return vmulq_f32(p, pow2k);
}
#endif /* ENGINE_NEON */

/* ========== WASM SIMD ========== */
#ifdef ENGINE_WASM_SIMD
#include <wasm_simd128.h>

static inline float hsum_wasm(v128_t v) {
    v128_t hi = wasm_i32x4_shuffle(v, v, 2, 3, 0, 1);
    v128_t sum2 = wasm_f32x4_add(v, hi);
    v128_t hi2 = wasm_i32x4_shuffle(sum2, sum2, 1, 0, 3, 2);
    v128_t sum1 = wasm_f32x4_add(sum2, hi2);
    return wasm_f32x4_extract_lane(sum1, 0);
}
#endif /* ENGINE_WASM_SIMD */

#endif /* SIMD_MATH_H */
