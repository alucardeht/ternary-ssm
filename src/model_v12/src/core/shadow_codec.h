/*
 * shadow_codec.h — INT8-style shadow weight codec (R356 F2)
 *
 * Foundation header for R356 F2: int16-encoded shadow weights with split
 * MVUE / saturation grids. Default-OFF behaviour is a no-op layered on top
 * of the existing FP32 shadow buffers — every helper here is inline and
 * pure, so leaving the compile flag at 0 keeps the binary byte-identical
 * to the pre-R356 build.
 *
 * Encoding (when V12_F2_INT8_SHADOWS == 1):
 *
 *   Code range     Zone         Mapping
 *   -127..+127     MVUE active  s = q / 127.0f                  (Δ = 1/127)
 *   ±(128..255)    Saturated    s = sign * (1 + (|q|-128)*9/127) (Δ = 9/127, [1,10])
 *   beyond ±255    Reserved     decoded clamped to ±10
 *
 * One int16_t per weight => 2 bytes/weight, 2× memory reduction vs FP32.
 *
 * This header is intentionally self-contained: it pulls in <stdint.h>,
 * <math.h> and "types.h" only for the V12_F2_INT8_SHADOWS flag.
 *
 * Source of truth: .workspace/research/r356-f2-int8-shadow-handoff.md
 */

#ifndef V12_SHADOW_CODEC_H
#define V12_SHADOW_CODEC_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Grid constants
 *
 * Kept as macros so they can be reused from tests without dragging the
 * inline functions in. The MVUE step is 1/127 (127 levels per side); the
 * saturation step is 9/127 across [1, 10].
 * ------------------------------------------------------------------------- */

#define V12_SHADOW_MVUE_LEVELS    127      /* per side, excluding zero      */
#define V12_SHADOW_MVUE_STEP      (1.0f / 127.0f)
#define V12_SHADOW_SAT_BASE       1.0f     /* saturation grid starts at 1.0 */
#define V12_SHADOW_SAT_TOP        10.0f    /* hard clip ceiling             */
#define V12_SHADOW_SAT_RANGE      (V12_SHADOW_SAT_TOP - V12_SHADOW_SAT_BASE)
#define V12_SHADOW_SAT_LEVELS     127      /* k in [0, 127], code 128..255  */
#define V12_SHADOW_SAT_STEP       (V12_SHADOW_SAT_RANGE / (float)V12_SHADOW_SAT_LEVELS)
#define V12_SHADOW_SAT_CODE_BASE  128      /* lowest saturation magnitude   */
#define V12_SHADOW_SAT_CODE_TOP   255      /* highest saturation magnitude  */
#define V12_SHADOW_CLIP_LO        (-V12_SHADOW_SAT_TOP)
#define V12_SHADOW_CLIP_HI        ( V12_SHADOW_SAT_TOP)

/* -------------------------------------------------------------------------
 * shadow_decode — int16 cell -> fp32 logit
 *
 * Pure function. Branch shape mirrors the handoff (Section 4):
 *   - in [-127, 127]  : uniform MVUE grid
 *   - magnitude in [128, 255] : saturation grid on [1, 10]
 *   - beyond ±255 (reserved) : clipped to ±10
 * ------------------------------------------------------------------------- */
static inline float shadow_decode(int16_t q)
{
    int a = (int)q;
    if (a >= -V12_SHADOW_MVUE_LEVELS && a <= V12_SHADOW_MVUE_LEVELS) {
        return (float)a * V12_SHADOW_MVUE_STEP;
    }
    int sign = (a > 0) ? 1 : -1;
    int mag  = (a > 0) ? a : -a;
    if (mag > V12_SHADOW_SAT_CODE_TOP) {
        mag = V12_SHADOW_SAT_CODE_TOP;
    }
    int k = mag - V12_SHADOW_SAT_CODE_BASE;          /* 0..127 */
    return (float)sign * (V12_SHADOW_SAT_BASE + (float)k * V12_SHADOW_SAT_STEP);
}

/* -------------------------------------------------------------------------
 * stochastic_round_shadow — fp32 logit -> int16 cell, unbiased
 *
 * Hard-clips to [-10, 10] (matches the existing fmaxf/fminf clamp around
 * the FP32 update sites) then stochastically rounds within whichever zone
 * the input falls into. Tower-rule unbiasedness holds in expectation —
 * proof in handoff Section 5.
 *
 * The rand32 input is consumed as a 24-bit uniform; the caller is
 * responsible for producing per-(layer, matrix, idx, step) decorrelated
 * draws via shadow_rng() below or any equivalent counter-based scheme.
 * ------------------------------------------------------------------------- */
static inline int16_t stochastic_round_shadow(float s, uint32_t rand32)
{
    if (s > V12_SHADOW_CLIP_HI) s = V12_SHADOW_CLIP_HI;
    if (s < V12_SHADOW_CLIP_LO) s = V12_SHADOW_CLIP_LO;

    /* 24-bit uniform in [0, 1) */
    float u = (float)(rand32 & 0x00FFFFFFu) * (1.0f / 16777216.0f);

    if (s >= -1.0f && s <= 1.0f) {
        /* MVUE zone: grid step 1/127 */
        float g = s * (float)V12_SHADOW_MVUE_LEVELS;
        float floor_g = floorf(g);
        float frac = g - floor_g;
        int q = (int)floor_g + ((u < frac) ? 1 : 0);
        if (q < -V12_SHADOW_MVUE_LEVELS) q = -V12_SHADOW_MVUE_LEVELS;
        if (q >  V12_SHADOW_MVUE_LEVELS) q =  V12_SHADOW_MVUE_LEVELS;
        return (int16_t)q;
    }

    /* Saturation zone */
    float sign = (s > 0.0f) ? 1.0f : -1.0f;
    float m    = fabsf(s) - V12_SHADOW_SAT_BASE;            /* in [0, 9] */
    float g    = m / V12_SHADOW_SAT_STEP;                   /* in [0, 127] */
    float floor_g = floorf(g);
    float frac = g - floor_g;
    int k = (int)floor_g + ((u < frac) ? 1 : 0);
    if (k < 0) k = 0;
    if (k > V12_SHADOW_SAT_LEVELS) k = V12_SHADOW_SAT_LEVELS;
    int q = V12_SHADOW_SAT_CODE_BASE + k;
    return (int16_t)((sign > 0.0f) ? q : -q);
}

/* -------------------------------------------------------------------------
 * shadow_rng — counter-mixed 32-bit draw
 *
 * Stateless per call. Combines (layer_id, matrix_id, idx, step_id) into a
 * 64-bit seed via a small xorshift mixer and returns the low 32 bits as
 * the random word fed to stochastic_round_shadow. Avoids cross-weight
 * correlation without introducing a new RNG dependency.
 * ------------------------------------------------------------------------- */
static inline uint32_t shadow_rng(int layer_id,
                                  int matrix_id,
                                  int idx,
                                  uint64_t step_id)
{
    uint64_t s = (uint64_t)0x9E3779B97F4A7C15ULL;
    s ^= ((uint64_t)(uint32_t)layer_id  + 0x165667B19E3779F9ULL) * 0xBF58476D1CE4E5B9ULL;
    s ^= ((uint64_t)(uint32_t)matrix_id + 0x94D049BB133111EBULL) * 0xC2B2AE3D27D4EB4FULL;
    s ^= ((uint64_t)(uint32_t)idx       + 0x27D4EB2F165667C5ULL) * 0xD1B54A32D192ED03ULL;
    s ^= step_id * 0x14057B7EF767814FULL;
    s ^= s >> 30;
    s *= 0xBF58476D1CE4E5B9ULL;
    s ^= s >> 27;
    s *= 0x94D049BB133111EBULL;
    s ^= s >> 31;
    return (uint32_t)s;
}

/* Matrix IDs used by shadow_rng to keep RNG draws disjoint across the
 * different shadow buffers. Free to grow as new matrices are quantized. */
enum {
    V12_SHADOW_MAT_W_IN        = 1,
    V12_SHADOW_MAT_W_OUT       = 2,
    V12_SHADOW_MAT_W_CONV      = 3,
    V12_SHADOW_MAT_W_A         = 4,
    V12_SHADOW_MAT_C_BILINEAR  = 5,
    V12_SHADOW_MAT_LM_HEAD     = 6,
    V12_SHADOW_MAT_W_PRED      = 7,
    V12_SHADOW_MAT_U           = 8,
    V12_SHADOW_MAT_V           = 9
};

/* -------------------------------------------------------------------------
 * Bulk converters — fp32 buffer <-> int16 cell buffer
 *
 * Both are inline so they can live in this header without a .c file. They
 * are never invoked when V12_F2_INT8_SHADOWS == 0 (callers are #ifdef'd),
 * so there is no codegen cost in the default build.
 *
 * shadow_to_fp32_bulk:    decode q[] into f[]            (read-only consumers)
 * shadow_from_fp32_bulk:  stochastic-encode f[] into q[] (one-shot init/load)
 *
 * The bulk encoder draws RNG bits via shadow_rng with matrix_id and the
 * caller-supplied layer_id and step_id. For one-shot init paths the
 * caller passes step_id == 0.
 * ------------------------------------------------------------------------- */
static inline void shadow_to_fp32_bulk(const int16_t *q, float *f, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        f[i] = shadow_decode(q[i]);
    }
}

static inline void shadow_from_fp32_bulk(const float *f,
                                         int16_t      *q,
                                         size_t        n,
                                         int           layer_id,
                                         int           matrix_id,
                                         uint64_t      step_id)
{
    for (size_t i = 0; i < n; i++) {
        uint32_t r = shadow_rng(layer_id, matrix_id, (int)i, step_id);
        q[i] = stochastic_round_shadow(f[i], r);
    }
}

/* -------------------------------------------------------------------------
 * sample_ternary_from_logit_q — quantized MVUE sampler
 *
 * Reads the int16 cell directly, decodes to a transient FP32 logit, and
 * applies the standard MVUE three-way sample. The decoded float never
 * touches a persistent buffer — it lives in a register for the life of
 * one sample. Equivalent to the FP32 sampler at grid points; equivalent
 * in expectation off grid (handoff Section 3, "Equivalence proof").
 * ------------------------------------------------------------------------- */
static inline float sample_ternary_from_logit_q(int16_t q, float u)
{
    float s = shadow_decode(q);
    if (s >=  1.0f) return  1.0f;
    if (s <= -1.0f) return -1.0f;
    float p_pos = (s > 0.0f) ?  s : 0.0f;
    float p_neg = (s < 0.0f) ? -s : 0.0f;
    if (u < p_pos)         return  1.0f;
    if (u < p_pos + p_neg) return -1.0f;
    return 0.0f;
}

#ifdef __cplusplus
}
#endif

#endif /* V12_SHADOW_CODEC_H */
