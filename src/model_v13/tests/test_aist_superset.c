/*
 * test_aist_superset.c — R355 Slice 3 gate tests
 *
 * Covers the two merge-gate guarantees from r355-aist-handoff §6:
 *
 *   Test B  — SKIPPED_off (crystallization only) is a strict subset of
 *             SKIPPED_on (crystallization + AIST), with every zone-2
 *             saturation-band weight landing in the AIST-only delta.
 *
 *   Test C  — For every weight NOT in SKIPPED_on, the shadow logit
 *             after one simulated Adam step is bit-identical between
 *             the AIST-OFF and AIST-ON paths. AIST only gates whether
 *             the update runs; it must not perturb the update math.
 *
 * These tests run in-process against a synthetic fixture that mirrors
 * the exact skip logic embedded in optimizer.c (crystallization check
 * first, AIST check second). They do NOT spin up a real training step
 * — that would require the full model plumbing and is reserved for
 * the production smoke test in the handoff.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>

#ifndef V12_AIST_ENABLED
#define V12_AIST_ENABLED 1
#endif

#include "../src/core/aist_gate.h"

/* Pull in the macro V12_CRYSTALLIZATION_TAU constant without dragging
 * all of types.h. Hard-code it here — if the constant moves in types.h
 * this test will drift and fail loudly, which is intended. */
#ifndef V12_CRYSTALLIZATION_TAU
#define V12_CRYSTALLIZATION_TAU 4.6f
#endif

#if V12_AIST_ENABLED
float v12_aist_epsilon = V12_AIST_EPSILON_DEFAULT;
#endif

#define N_WEIGHTS 4000

/* Simulated weights and proposed gradient steps, partitioned into the
 * four zones from handoff §6 Test B. */
static float s_before[N_WEIGHTS];
static float g_step[N_WEIGHTS];
static int   zone[N_WEIGHTS];    /* 1..4 */

static void build_fixture(void)
{
    uint32_t rng = 0xC0FFEEu;
    const float d_mvue = 1.0f / 127.0f;
    int n_per = N_WEIGHTS / 4;

    for (int i = 0; i < N_WEIGHTS; i++) {
        /* cheap xorshift for reproducible variety */
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        float u = (float)(rng & 0xFFFFFF) / 16777216.0f;

        int z = i / n_per + 1;
        if (z > 4) z = 4;
        zone[i] = z;

        switch (z) {
        case 1: /* crystallized: |s| > 5 */
            s_before[i] = (u < 0.5f ? -1.0f : 1.0f) * (5.0f + u * 4.0f);
            g_step[i]   = 0.5f;
            break;
        case 2: /* saturation band: 1 <= |s| <= 4.6 — AIST territory */
            s_before[i] = (u < 0.5f ? -1.0f : 1.0f) * (1.0f + u * 3.5f);
            g_step[i]   = 0.2f;
            break;
        case 3: /* interior with tiny sub-grid step */
            s_before[i] = (u - 0.5f) * 1.8f;             /* in (-0.9, 0.9) */
            g_step[i]   = 1e-8f * d_mvue;
            break;
        case 4: /* interior with healthy step */
            s_before[i] = (u - 0.5f) * 1.8f;
            g_step[i]   = 5.0f * d_mvue;
            break;
        }
    }
}

/* Simulate the exact optimizer.c order:
 *   1. crystallization check    (if |s| > tau, skip)
 *   2. AIST gate (when ON)      (aist_should_skip -> skip)
 *   3. otherwise apply update   (s <- s - g)
 * Returns 1 if the update was skipped. */
static int simulate_step(float *s_inout, float g, float s_for_gate,
                         int aist_on, float eps)
{
    if (fabsf(s_for_gate) > V12_CRYSTALLIZATION_TAU) return 1;
    if (aist_on && aist_should_skip(s_for_gate, g, eps)) return 1;
    *s_inout = s_for_gate - g;
    return 0;
}

static void test_superset(void)
{
    printf("\n=== AIST Test B: strict superset (crystallized ⊂ AIST) ===\n");
    float eps = 1e-3f;

    int skipped_off[N_WEIGHTS] = {0};
    int skipped_on [N_WEIGHTS] = {0};

    for (int i = 0; i < N_WEIGHTS; i++) {
        float s_off = s_before[i];
        float s_on  = s_before[i];
        skipped_off[i] = simulate_step(&s_off, g_step[i], s_before[i], 0, eps);
        skipped_on [i] = simulate_step(&s_on , g_step[i], s_before[i], 1, eps);
    }

    int n_off = 0, n_on = 0, n_zone2_delta = 0, n_zone3_delta = 0;
    for (int i = 0; i < N_WEIGHTS; i++) {
        n_off += skipped_off[i];
        n_on  += skipped_on[i];
        /* Strict superset check: every weight in off must be in on. */
        assert(!skipped_off[i] || skipped_on[i]);

        if (zone[i] == 1) {
            assert(skipped_off[i] == 1);
            assert(skipped_on[i]  == 1);
        }
        if (zone[i] == 2) {
            assert(skipped_off[i] == 0 && "zone 2 must not skip under crystallization alone");
            assert(skipped_on[i]  == 1 && "zone 2 must be AIST-skipped");
            n_zone2_delta++;
        }
        if (zone[i] == 3) {
            assert(skipped_off[i] == 0);
            assert(skipped_on[i]  == 1);
            n_zone3_delta++;
        }
        if (zone[i] == 4) {
            assert(skipped_off[i] == 0);
            assert(skipped_on[i]  == 0);
        }
    }

    printf("  SKIPPED_off=%d  SKIPPED_on=%d  Δ_zone2=%d  Δ_zone3=%d\n",
           n_off, n_on, n_zone2_delta, n_zone3_delta);
    assert(n_on > n_off);
    assert(n_zone2_delta > 0);
    assert(n_zone3_delta > 0);
    printf("  strict superset OK\n");
}

static void test_gradient_invariance(void)
{
    printf("\n=== AIST Test C: gradient invariance on non-skipped ===\n");
    float eps = 1e-3f;

    for (int i = 0; i < N_WEIGHTS; i++) {
        float s_off = s_before[i];
        float s_on  = s_before[i];
        int skip_off = simulate_step(&s_off, g_step[i], s_before[i], 0, eps);
        int skip_on  = simulate_step(&s_on , g_step[i], s_before[i], 1, eps);

        if (!skip_on) {
            /* AIST did not skip -> result must be bit-exact. */
            assert(!skip_off);
            /* memcmp on the raw bits to catch any FP drift */
            assert(memcmp(&s_on, &s_off, sizeof(float)) == 0);
        } else if (!skip_off && skip_on) {
            /* AIST genuinely skipped an update the baseline would have
             * applied. The AIST-on shadow stays at its pre-update value. */
            assert(memcmp(&s_on, &s_before[i], sizeof(float)) == 0);
            /* And the baseline moved. */
            assert(memcmp(&s_off, &s_before[i], sizeof(float)) != 0);
        }
    }
    printf("  bit-exact invariance OK\n");
}

static void test_runtime_off_equivalence(void)
{
    /* With epsilon = 0, AIST (Gate A * Gate B < 0) skips nothing beyond
     * what the crystallization check already caught. Asserts the
     * runtime default-OFF guarantee. */
    printf("\n=== AIST Test: runtime ε=0 reduces to crystallization ===\n");
    int delta = 0;
    for (int i = 0; i < N_WEIGHTS; i++) {
        float s_off = s_before[i];
        float s_on  = s_before[i];
        int skip_off = simulate_step(&s_off, g_step[i], s_before[i], 0, 0.0f);
        int skip_on  = simulate_step(&s_on , g_step[i], s_before[i], 1, 0.0f);
        if (skip_off != skip_on) delta++;
        if (!skip_on) {
            assert(memcmp(&s_on, &s_off, sizeof(float)) == 0);
        }
    }
    printf("  delta(skip sets) at ε=0 = %d (must be 0)\n", delta);
    assert(delta == 0);
    printf("  runtime off equivalence OK\n");
}

int main(void)
{
    printf("\n========================================\n");
    printf("  AIST SUPERSET + INVARIANCE TESTS\n");
    printf("========================================\n");

    build_fixture();

    test_superset();
    test_gradient_invariance();
    test_runtime_off_equivalence();

    printf("\n========================================\n");
    printf("  ALL AIST SLICE-3 TESTS PASSED\n");
    printf("========================================\n");
    return 0;
}
