/*
 * aist_gate.h — R355 Active Information Sweep gating (header-only)
 *
 * Default-OFF gate that extends the existing crystallization skip by
 * short-circuiting Adam updates whose expected effect on the sampled
 * ternary weight is below `epsilon`. Sits AFTER the crystallization
 * check in the optimizer hot loops; never replaces it.
 *
 * Core math (see .workspace/research/r355-aist-handoff.md §2.3):
 *
 *   Delta_grid(s) = 1/127        if |s| <= 1        (MVUE zone)
 *                 = 9/127        if 1 < |s| <= 10   (saturation zone)
 *                 = +inf         if |s| >  10       (clipped)
 *
 *   cell_prob_move     = min(1, |g| / Delta_grid(s))
 *   sample_sensitivity = 1                           if |s| < 1 - margin
 *                      = linear decay                if 1 - margin <= |s| < 1
 *                      = 0                           if |s| >= 1
 *   SKIP(i) iff cell_prob_move * sample_sensitivity < epsilon
 *
 * Strict superset property: whenever the current `|s| > V12_CRYSTALLIZATION_TAU`
 * check would skip, `|s| >= 1` also holds, so `sample_sensitivity == 0`
 * and `aist_should_skip` returns 1 for any non-negative epsilon.
 *
 * Flag layering (mirrors R356 F2 `V12_F2_INT8_SHADOWS`):
 *   - Compile flag `V12_AIST_ENABLED` defaults to 0 — gate blocks
 *     preprocessed out, optimizer byte-identical.
 *   - Runtime knob `v12_aist_epsilon` defaults to 0.0f — even a
 *     compiled-in build behaves like baseline until the CLI opts in.
 *
 * Source of truth: .workspace/research/r355-aist-handoff.md
 */

#ifndef V12_AIST_GATE_H
#define V12_AIST_GATE_H

#include <math.h>
#include "shadow_codec.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef V12_AIST_ENABLED
#define V12_AIST_ENABLED 0
#endif

#define V12_AIST_EPSILON_DEFAULT 1e-3f

/* Runtime knob. Declared here and defined once in optimizer.c under the
 * same `#if V12_AIST_ENABLED` guard. When the compile flag is 0 this
 * declaration is still emitted (harmless, never referenced) so unit
 * tests that only include this header can link against a local
 * definition without pulling in the optimizer. */
#if V12_AIST_ENABLED
extern float v12_aist_epsilon;
#endif

/* Local grid step at logit `s`. Matches the R356 F2 codec zones. */
static inline float aist_delta_grid(float s)
{
    float a = fabsf(s);
    if (a <= V12_SHADOW_SAT_BASE) {
        return V12_SHADOW_MVUE_STEP;          /* 1/127  */
    }
    if (a <= V12_SHADOW_SAT_TOP) {
        return V12_SHADOW_SAT_STEP;           /* 9/127  */
    }
    return INFINITY;                          /* clipped */
}

/* Upper bound on Pr[cell moves] under stochastic rounding across one
 * grid boundary. Tight when |g| < Delta_grid. */
static inline float aist_cell_prob_move(float s, float g)
{
    float d = aist_delta_grid(s);
    if (!isfinite(d)) return 0.0f;            /* clipped -> no motion */
    float r = fabsf(g) / d;
    return (r > 1.0f) ? 1.0f : r;
}

/* Three-way MVUE sampler sensitivity to a single-cell move.
 * 1 deep in the interior, linearly decaying across a one-step margin
 * below |s| = 1, and 0 once the sample is deterministically saturated. */
static inline float aist_sample_sensitivity(float s)
{
    float a = fabsf(s);
    if (a >= V12_SHADOW_SAT_BASE) return 0.0f;        /* |s| >= 1 */
    float margin = V12_SHADOW_MVUE_STEP;              /* 1/127    */
    float boundary = V12_SHADOW_SAT_BASE - margin;    /* 1 - 1/127 */
    if (a < boundary) return 1.0f;
    return 1.0f - (a - boundary) / margin;
}

/* Combined AIST skip test. Returns 1 when the proposed step `g` carries
 * less than `epsilon` expected information about the sampled ternary
 * weight at shadow logit `s`. */
static inline int aist_should_skip(float s, float g, float epsilon)
{
    float p_move = aist_cell_prob_move(s, g);
    float p_flip = aist_sample_sensitivity(s);
    return (p_move * p_flip) < epsilon;
}

#ifdef __cplusplus
}
#endif

#endif /* V12_AIST_GATE_H */
