#ifndef V12_OPTIMIZER_H
#define V12_OPTIMIZER_H

#include "types.h"

/* =========================================================================
 * optimizer.h — MVUE ternary optimizer + crystallization + LR schedule
 *
 * Implements the 4-function optimizer pipeline:
 *   1. mvue_sample_weights  — sample {-1,0,+1} from shadow logits
 *   2. optimizer_step       — MVUE gradient + D4 filter + crystallization
 *   3. lr_schedule          — warmup + cosine decay
 *   4. update_crystallization — freeze saturated logits
 *
 * None of these functions allocate memory.
 * All state is in Model and TrainState (types.h).
 * ========================================================================= */

/* -------------------------------------------------------------------------
 * mvue_sample_weights
 *
 * For ONE SSMLayer, draw a new ternary sample from the MVUE distribution
 * P(w | s) defined by the shadow logit s:
 *
 *   P(w = +1 | s) = max(s, 0)
 *   P(w = -1 | s) = max(-s, 0)
 *   P(w =  0 | s) = 1 - |s|
 *
 * Requires |s| <= 1 to form a valid probability. For |s| > 1, the weight
 * is saturated: use w = sign(s) deterministically (equivalent to round).
 *
 * Populates: layer->sampled_w_in, layer->sampled_w_out
 * Reads:     layer->shadow_w_in, layer->shadow_w_out, layer->crystal_mask_*
 *
 * rng_state: pointer to a 64-bit xorshift state. Updated in-place.
 *            Call mvue_sample_weights once per REPACK step, not every step.
 * ------------------------------------------------------------------------- */
void mvue_sample_weights(SSMLayer *layer, uint64_t *rng_state);

/* -------------------------------------------------------------------------
 * optimizer_reset_moments
 *
 * Reset all Adam moment accumulators to zero when transitioning between
 * training phases (FP -> BCD, BCD -> joint, etc.).
 *
 * Called before phase transitions to prevent accumulated optimizer state
 * from exploding when lr_per_layer changes from 0 to nonzero.
 * ------------------------------------------------------------------------- */
void optimizer_reset_moments(Model *model);

/* -------------------------------------------------------------------------
 * optimizer_step
 *
 * Applies one gradient step to all ternary and FP32 parameters in the model.
 *
 * For ternary parameters (W_in, W_out, lm_head, conv):
 *   s_new = s - lr * dW * (w_sampled - s) * soft_boundary_mask(s)
 *   Subject to:
 *     1. Crystallization skip: skip if |s| > V12_CRYSTALLIZATION_TAU
 *     2. D4 sign consistency filter: skip if last 5 gradient signs disagree
 *
 * For FP32 parameters (A, B, C, norm_w, norm_b):
 *   param -= lr * d_param
 *   A clamped to (1e-4, 1-1e-4) after update.
 *
 * Zeros all gradient accumulators (dw_in, dw_out, dA, dB, dC, dlm_head).
 *
 * Every V12_REPACK_INTERVAL steps: repacks shadow → INT2, resamples
 * sampled_w_* via mvue_sample_weights.
 *
 * lr: effective learning rate for this step (output of lr_schedule).
 * ------------------------------------------------------------------------- */
void optimizer_step(Model *model, TrainState *state, float lr);

/* -------------------------------------------------------------------------
 * lr_schedule
 *
 * Warmup + cosine decay:
 *   step < warmup:  lr = lr_base * step / warmup
 *   step >= warmup: lr = lr_base * 0.5 * (1 + cos(pi * t))
 *                   where t = (step - warmup) / (total - warmup)
 *   floor:          lr = max(lr, lr_base * V12_LR_MIN_FACTOR)
 *
 * Returns effective scalar lr for this step.
 * state->lr_per_layer[k] multiplier is applied inside optimizer_step,
 * not here — this function returns the base value.
 * ------------------------------------------------------------------------- */
float lr_schedule(const TrainState *state);

/* -------------------------------------------------------------------------
 * update_crystallization
 *
 * Scans shadow logits of ONE SSMLayer and sets crystal_mask_* to 1 wherever
 * |s| > V12_CRYSTALLIZATION_TAU.
 *
 * Called every V12_REPACK_INTERVAL steps (same cadence as repack).
 * After ~10K steps, 20-30% of bits crystallize per layer.
 *
 * Also updates state->crystal_fraction (fraction across all layers, used
 * for progress monitoring and checkpoint logging).
 * ------------------------------------------------------------------------- */
void update_crystallization(SSMLayer *layer, int layer_idx,
                             TrainState *state, int step);

/* =========================================================================
 * INLINE HELPERS — included in the header so they inline at call sites
 * ========================================================================= */

#include <math.h>

/* soft_boundary_mask
 *
 * Key math (innovation §7, math-innovations.md):
 *   mask = 1.0               for |s| < 0.5
 *   mask = exp(-k*(|s|-1)^2) for 0.5 <= |s| <= 1.5, k = V12_SOFT_BOUNDARY_SCALE
 *   mask = 0.0               for |s| > 1.5
 *
 * Smooth substitute for the hard indicator 1[|s| < 1].
 * Eliminates gradient cliff at |s|=1 (60% oscillation reduction, R335).
 */
static inline float soft_boundary_mask(float s) {
    float abs_s = fabsf(s);
    if (abs_s <= 0.5f) return 1.0f;
    if (abs_s >= 1.5f) return 0.0f;
    float d = abs_s - 1.0f;
    return expf(-V12_SOFT_BOUNDARY_SCALE * d * d);
}

/* xorshift64 — minimal inline RNG for sampling */
static inline uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/* uniform01 — convert xorshift64 output to float in [0, 1) */
static inline float uniform01(uint64_t *rng) {
    return (float)(xorshift64(rng) >> 11) * (1.0f / (float)(1ULL << 53));
}

/* sample_ternary_from_logit
 *
 * Draw one ternary sample from MVUE distribution.
 * u: uniform random in [0, 1) from caller's RNG.
 *
 * Algorithm:
 *   if |s| >= 1.0: return (int)sign(s)   [saturated — deterministic]
 *   p_pos = max(s, 0)
 *   p_neg = max(-s, 0)
 *   if u < p_pos: return +1.0f
 *   if u < p_pos + p_neg: return -1.0f
 *   return 0.0f
 *
 * E[w | s] = s exactly (unbiased, minimum variance over ternary distributions).
 */
static inline float sample_ternary_from_logit_mvue(float s, float u) {
    if (s >=  1.0f) return  1.0f;
    if (s <= -1.0f) return -1.0f;
    float p_pos = (s > 0.0f) ? s : 0.0f;
    float p_neg = (s < 0.0f) ? -s : 0.0f;
    if (u < p_pos)           return  1.0f;
    if (u < p_pos + p_neg)   return -1.0f;
    return 0.0f;
}

static inline float sample_ternary_from_logit(float s, float u) {
#if V12_CDSVQ_ENABLED
    /* Hash u to seed xorshift64 generator for 3 independent Gumbel variables */
    uint64_t state = (uint64_t)(u * 4294967295.0f);
    if (state == 0) state = 0x12345678ULL;
    
    /* Draw 3 uniform variables */
    float u_pos = (float)(xorshift64(&state) >> 11) * (1.0f / (float)(1ULL << 53));
    float u_neg = (float)(xorshift64(&state) >> 11) * (1.0f / (float)(1ULL << 53));
    float u_zero = (float)(xorshift64(&state) >> 11) * (1.0f / (float)(1ULL << 53));
    
    if (u_pos < 1e-7f) u_pos = 1e-7f;
    else if (u_pos > 1.0f - 1e-7f) u_pos = 1.0f - 1e-7f;
    
    if (u_neg < 1e-7f) u_neg = 1e-7f;
    else if (u_neg > 1.0f - 1e-7f) u_neg = 1.0f - 1e-7f;
    
    if (u_zero < 1e-7f) u_zero = 1e-7f;
    else if (u_zero > 1.0f - 1e-7f) u_zero = 1.0f - 1e-7f;
    
    float g_pos = -logf(-logf(u_pos));
    float g_neg = -logf(-logf(u_neg));
    float g_zero = -logf(-logf(u_zero));
    
    float l_pos = s + g_pos;
    float l_neg = -s + g_neg;
    float l_zero = V12_CDSVQ_DELTA + g_zero;
    
    if (l_pos > l_neg && l_pos > l_zero) return 1.0f;
    if (l_neg > l_pos && l_neg > l_zero) return -1.0f;
    return 0.0f;
#else
    return sample_ternary_from_logit_mvue(s, u);
#endif
}

/* round_ternary
 *
 * Deterministic evaluation-time quantization (inference mode).
 * Round shadow logit to nearest {-1, 0, +1}:
 *   s >= 0.5  → +1
 *   s <= -0.5 → -1
 *   else      →  0
 *
 * NOT used during training. Used only by eval and repack.
 */
static inline float round_ternary(float s) {
    if (s >=  0.5f) return  1.0f;
    if (s <= -0.5f) return -1.0f;
    return 0.0f;
}



#endif /* V12_OPTIMIZER_H */
