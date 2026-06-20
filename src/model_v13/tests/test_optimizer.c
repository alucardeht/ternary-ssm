#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "../src/core/optimizer.h"
#include "../src/core/types.h"

/* =========================================================================
 * Test 1: MVUE Sampling Distribution
 *
 * Verify that samples from mvue_sample_weights follow the MVUE distribution.
 * For a shadow logit s, we expect:
 *   P(+1) ≈ max(s, 0)
 *   P(-1) ≈ max(-s, 0)
 *   P(0)  ≈ 1 - |s|
 * ========================================================================= */

void test_mvue_distribution(void) {
    printf("\n=== Test 1: MVUE Sampling Distribution ===\n");

    SSMLayer layer;
    memset(&layer, 0, sizeof(SSMLayer));

    int n_samples = V12_D_MODEL * V12_D_STATE;
    layer.shadow_w_in = (float *)malloc(n_samples * sizeof(float));
    layer.sampled_w_in = (float *)malloc(n_samples * sizeof(float));
    layer.shadow_w_out = (float *)malloc(V12_D_STATE * V12_D_MODEL * sizeof(float));
    layer.sampled_w_out = (float *)malloc(V12_D_STATE * V12_D_MODEL * sizeof(float));
    layer.shadow_w_conv = (float *)malloc(V12_CONV_KERNEL * V12_D_STATE * sizeof(float));

    for (int i = 0; i < n_samples; i++) {
        layer.shadow_w_in[i] = 0.0f;
        layer.sampled_w_in[i] = 0.0f;
    }
    for (int i = 0; i < V12_D_STATE * V12_D_MODEL; i++) {
        layer.shadow_w_out[i] = 0.0f;
        layer.sampled_w_out[i] = 0.0f;
    }
    for (int i = 0; i < V12_CONV_KERNEL * V12_D_STATE; i++) {
        layer.shadow_w_conv[i] = 0.0f;
    }

    float test_logits[] = { -0.8f, -0.5f, -0.2f, 0.0f, 0.2f, 0.5f, 0.8f };
    int n_tests = sizeof(test_logits) / sizeof(test_logits[0]);

    uint64_t rng_state = 42;

    for (int t = 0; t < n_tests; t++) {
        float s = test_logits[t];
        int n_trials = 10000;

        int count_pos = 0, count_neg = 0, count_zero = 0;

        for (int trial = 0; trial < n_trials; trial++) {
            float u = uniform01(&rng_state);
            float w = sample_ternary_from_logit(s, u);

            if (w > 0.5f) count_pos++;
            else if (w < -0.5f) count_neg++;
            else count_zero++;
        }

        float emp_p_pos = (float)count_pos / n_trials;
        float emp_p_neg = (float)count_neg / n_trials;
        float emp_p_zero = (float)count_zero / n_trials;

        float exp_p_pos = fmaxf(s, 0.0f);
        float exp_p_neg = fmaxf(-s, 0.0f);
        float exp_p_zero = 1.0f - fabsf(s);

        float err_pos = fabsf(emp_p_pos - exp_p_pos);
        float err_neg = fabsf(emp_p_neg - exp_p_neg);
        float err_zero = fabsf(emp_p_zero - exp_p_zero);

        printf("s=%.2f: P(+1) emp=%.4f exp=%.4f err=%.4f | "
               "P(-1) emp=%.4f exp=%.4f err=%.4f | "
               "P(0) emp=%.4f exp=%.4f err=%.4f\n",
               s, emp_p_pos, exp_p_pos, err_pos,
                  emp_p_neg, exp_p_neg, err_neg,
                  emp_p_zero, exp_p_zero, err_zero);

        assert(err_pos < 0.02f && "P(+1) empirical vs expected mismatch");
        assert(err_neg < 0.02f && "P(-1) empirical vs expected mismatch");
        assert(err_zero < 0.02f && "P(0) empirical vs expected mismatch");
    }

    printf("✓ MVUE distribution test passed.\n");

    free(layer.shadow_w_in);
    free(layer.sampled_w_in);
    free(layer.shadow_w_out);
    free(layer.sampled_w_out);
    free(layer.shadow_w_conv);
}

/* =========================================================================
 * Test 2: Crystallization Threshold
 *
 * Verify that update_crystallization correctly freezes weights where
 * |s| > V12_CRYSTALLIZATION_TAU.
 * ========================================================================= */

void test_crystallization_threshold(void) {
    printf("\n=== Test 2: Crystallization Threshold ===\n");

    SSMLayer layer;
    memset(&layer, 0, sizeof(SSMLayer));

    int n_in = V12_D_MODEL * V12_D_STATE;
    layer.shadow_w_in = (float *)malloc(n_in * sizeof(float));
    layer.crystal_mask_in = (uint8_t *)malloc(n_in * sizeof(uint8_t));
    layer.shadow_w_out = (float *)malloc(V12_D_STATE * V12_D_MODEL * sizeof(float));
    layer.crystal_mask_out = (uint8_t *)malloc(V12_D_STATE * V12_D_MODEL * sizeof(uint8_t));

    float below_tau = V12_CRYSTALLIZATION_TAU - 1.0f;
    float above_tau = V12_CRYSTALLIZATION_TAU + 1.0f;

    for (int i = 0; i < n_in / 2; i++) {
        layer.shadow_w_in[i] = below_tau;
    }
    for (int i = n_in / 2; i < n_in; i++) {
        layer.shadow_w_in[i] = above_tau;
    }

    for (int i = 0; i < V12_D_STATE * V12_D_MODEL; i++) {
        layer.shadow_w_out[i] = 0.0f;
    }

    TrainState state;
    memset(&state, 0, sizeof(TrainState));

    update_crystallization(&layer, 0, &state, 0);

    int frozen_below = 0, frozen_above = 0;
    for (int i = 0; i < n_in / 2; i++) {
        frozen_below += layer.crystal_mask_in[i];
    }
    for (int i = n_in / 2; i < n_in; i++) {
        frozen_above += layer.crystal_mask_in[i];
    }

    printf("Below tau (%.1f): %d frozen (expected ~0)\n", below_tau, frozen_below);
    printf("Above tau (%.1f): %d frozen (expected ~%d)\n", above_tau, frozen_above, n_in / 2);

    assert(frozen_below == 0 && "Weights below tau should not crystallize");
    assert(frozen_above == n_in / 2 && "Weights above tau should crystallize");

    printf("✓ Crystallization threshold test passed.\n");

    free(layer.shadow_w_in);
    free(layer.crystal_mask_in);
    free(layer.shadow_w_out);
    free(layer.crystal_mask_out);
}

/* =========================================================================
 * Test 3: Learning Rate Schedule
 *
 * Verify warmup and cosine decay schedule behavior.
 * ========================================================================= */

void test_lr_schedule(void) {
    printf("\n=== Test 3: Learning Rate Schedule ===\n");

    TrainState state;
    memset(&state, 0, sizeof(TrainState));
    state.lr_base = 0.1f;

    float base = state.lr_base;
    float floor = base * V12_LR_MIN_FACTOR;

    struct {
        uint32_t step;
        float expected;
        const char *desc;
    } test_cases[] = {
        { 0,                               floor,                                 "warmup start" },
        { V12_LR_WARMUP_STEPS / 2,         base * 0.5f,                           "warmup midpoint" },
        { V12_LR_WARMUP_STEPS,             base,                                  "warmup end = peak" },
        { V12_LR_TOTAL_STEPS,              floor,                                 "cosine end = floor" },
    };

    int n_tests = sizeof(test_cases) / sizeof(test_cases[0]);

    for (int i = 0; i < n_tests; i++) {
        state.step = test_cases[i].step;
        float lr = lr_schedule(&state);
        float err = fabsf(lr - test_cases[i].expected);

        printf("step=%u (%s): lr=%.6f expected≈%.6f err=%.6f\n",
               test_cases[i].step, test_cases[i].desc, lr,
               test_cases[i].expected, err);

        assert(err < 0.002f && "LR schedule value mismatch");
    }

    printf("✓ Learning rate schedule test passed.\n");
}

/* =========================================================================
 * Test 4: Soft Boundary Mask
 *
 * Verify smooth boundary mask computation.
 * ========================================================================= */

void test_soft_boundary_mask(void) {
    printf("\n=== Test 4: Soft Boundary Mask ===\n");

    struct {
        float s;
        float expected_approx;
        const char *region;
    } test_points[] = {
        { 0.0f,  1.0f, "interior" },
        { 0.5f,  1.0f, "boundary" },
        { 1.0f,  1.0f, "at boundary (d=0, mask=1)" },
        { 1.5f,  0.0f, "saturated" },
        { -0.5f, 1.0f, "interior (neg)" },
        { -1.0f, 1.0f, "at boundary neg (d=0, mask=1)" },
    };

    int n_tests = sizeof(test_points) / sizeof(test_points[0]);

    for (int i = 0; i < n_tests; i++) {
        float mask = soft_boundary_mask(test_points[i].s);
        float err = fabsf(mask - test_points[i].expected_approx);

        printf("s=%.2f (%s): mask=%.6f expected≈%.6f err=%.6f\n",
               test_points[i].s, test_points[i].region, mask,
               test_points[i].expected_approx, err);

        assert(err < 0.01f && "Soft boundary mask value mismatch");
    }

    printf("✓ Soft boundary mask test passed.\n");
}

/* =========================================================================
 * main: Run all tests
 * ========================================================================= */

int main(void) {
    printf("\n========================================\n");
    printf("  OPTIMIZER TESTS — v12\n");
    printf("========================================\n");

    test_mvue_distribution();
    test_crystallization_threshold();
    test_lr_schedule();
    test_soft_boundary_mask();

    printf("\n========================================\n");
    printf("  ALL TESTS PASSED\n");
    printf("========================================\n");

    return 0;
}
