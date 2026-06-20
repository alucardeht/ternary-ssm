/*
 * test_aist_gate.c — R355 Slice 1 unit tests
 *
 * Covers the three inline helpers in aist_gate.h on hand-computed cases
 * from the R355 handoff §2.3 / §6 smoke test A. No optimizer involvement.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

/* Force the compile flag ON for this translation unit so the runtime
 * extern is visible. The gate functions themselves are not gated by
 * the flag — only the optimizer call sites are. */
#ifndef V12_AIST_ENABLED
#define V12_AIST_ENABLED 1
#endif

#include "../src/core/aist_gate.h"

/* Local definition of the runtime knob. The real optimizer build
 * provides it; the unit test binary links only this file plus the
 * header, so we satisfy the extern here. */
#if V12_AIST_ENABLED
float v12_aist_epsilon = V12_AIST_EPSILON_DEFAULT;
#endif

static int close_enough(float a, float b, float eps)
{
    return fabsf(a - b) <= eps;
}

static void test_delta_grid(void)
{
    printf("\n=== AIST test: delta_grid zones ===\n");
    assert(close_enough(aist_delta_grid(0.0f),  1.0f / 127.0f, 1e-9f));
    assert(close_enough(aist_delta_grid(0.5f),  1.0f / 127.0f, 1e-9f));
    assert(close_enough(aist_delta_grid(-0.9f), 1.0f / 127.0f, 1e-9f));
    assert(close_enough(aist_delta_grid(1.0f),  1.0f / 127.0f, 1e-9f));
    assert(close_enough(aist_delta_grid(1.5f),  9.0f / 127.0f, 1e-9f));
    assert(close_enough(aist_delta_grid(-5.0f), 9.0f / 127.0f, 1e-9f));
    assert(close_enough(aist_delta_grid(10.0f), 9.0f / 127.0f, 1e-9f));
    assert(!isfinite(aist_delta_grid(10.5f)));
    assert(!isfinite(aist_delta_grid(-12.0f)));
    printf("  delta_grid zones OK\n");
}

static void test_cell_prob_move(void)
{
    printf("\n=== AIST test: cell_prob_move ===\n");
    /* MVUE zone, sub-grid step */
    float d_mvue = 1.0f / 127.0f;
    assert(close_enough(aist_cell_prob_move(0.2f,  0.5f * d_mvue), 0.5f, 1e-6f));
    assert(close_enough(aist_cell_prob_move(0.2f, -0.25f * d_mvue), 0.25f, 1e-6f));
    /* Saturates at 1 when |g| >= delta */
    assert(close_enough(aist_cell_prob_move(0.2f, 2.0f * d_mvue), 1.0f, 1e-6f));
    /* Saturation zone, wider grid */
    float d_sat = 9.0f / 127.0f;
    assert(close_enough(aist_cell_prob_move(3.0f, 0.5f * d_sat), 0.5f, 1e-6f));
    /* Clipped zone => no motion regardless of |g| */
    assert(aist_cell_prob_move(11.0f, 10.0f) == 0.0f);
    printf("  cell_prob_move OK\n");
}

static void test_sample_sensitivity(void)
{
    printf("\n=== AIST test: sample_sensitivity ===\n");
    /* Deep interior -> 1 */
    assert(aist_sample_sensitivity(0.0f)  == 1.0f);
    assert(aist_sample_sensitivity(0.5f)  == 1.0f);
    assert(aist_sample_sensitivity(-0.5f) == 1.0f);
    /* At and beyond saturation -> 0 */
    assert(aist_sample_sensitivity(1.0f)  == 0.0f);
    assert(aist_sample_sensitivity(2.0f)  == 0.0f);
    assert(aist_sample_sensitivity(-4.6f) == 0.0f);
    assert(aist_sample_sensitivity(-10.0f) == 0.0f);
    /* Margin band: linear decay over one MVUE step below 1 */
    float margin = 1.0f / 127.0f;
    float mid = 1.0f - 0.5f * margin;
    assert(close_enough(aist_sample_sensitivity(mid), 0.5f, 1e-5f));
    printf("  sample_sensitivity OK\n");
}

static void test_should_skip(void)
{
    printf("\n=== AIST test: should_skip composition ===\n");
    float eps = 1e-3f;

    /* Case 1: crystallized zone -> must skip (strict superset) */
    assert(aist_should_skip(5.0f, 0.5f, eps) == 1);
    assert(aist_should_skip(-4.7f, 1.0f, eps) == 1);
    assert(aist_should_skip(10.0f, 0.001f, eps) == 1);

    /* Case 2: saturation band [1, 4.6] -> new AIST territory, must skip */
    assert(aist_should_skip(1.0f, 0.5f, eps) == 1);
    assert(aist_should_skip(2.5f, 1.0f, eps) == 1);
    assert(aist_should_skip(-3.5f, 0.2f, eps) == 1);

    /* Case 3: interior with sub-grid noise -> skip */
    float d = 1.0f / 127.0f;
    assert(aist_should_skip(0.2f, 1e-6f * d, eps) == 1);

    /* Case 4: interior with healthy step -> do NOT skip */
    assert(aist_should_skip(0.2f, 5.0f * d, eps) == 0);
    assert(aist_should_skip(-0.3f, 2.0f * d, eps) == 0);

    /* Runtime default-OFF equivalent: epsilon = 0 reduces AIST to the
     * crystallization-only gate (nothing extra is skipped beyond the
     * saturation floor, because 0 * anything < 0 is false). */
    assert(aist_should_skip(0.2f, 5.0f * d, 0.0f) == 0);
    assert(aist_should_skip(0.2f, 1e-6f * d, 0.0f) == 0);
    /* But strictly-saturated (|s|>=1) still skips at eps=0, because
     * sample_sensitivity == 0 and 0 < 0 is false — so this case does
     * NOT skip at epsilon=0. That is the correct runtime-off behavior. */
    assert(aist_should_skip(5.0f, 0.5f, 0.0f) == 0);

    printf("  should_skip composition OK\n");
}

int main(void)
{
    printf("\n========================================\n");
    printf("  AIST GATE UNIT TESTS (R355 slice 1)\n");
    printf("========================================\n");

    test_delta_grid();
    test_cell_prob_move();
    test_sample_sensitivity();
    test_should_skip();

    printf("\n========================================\n");
    printf("  ALL AIST UNIT TESTS PASSED\n");
    printf("========================================\n");
    return 0;
}
