#include <math.h>
#include <stdio.h>

#include "runtime/vulkan_backend.h"

static int almost_equal(double a, double b) {
    return fabs(a - b) < 1e-9;
}

int main(void) {
    VkRuntimeProfile profile = {0};
    VkR370Summary summary = {0};

    profile.submit_ns = 2000000ull;
    profile.wait_ns = 1000000ull;
    profile.gpu_cmd_ns = 6000000ull;
    profile.dispatch_record_ns = 1200000ull;
    profile.dispatch_submit_ns = 500000ull;
    profile.dispatch_wait_ns = 7000000ull;
    profile.dispatch_reset_ns = 300000ull;
    profile.timed_submit_count = 1;
    profile.gpu_timing_supported = 1;

    vk_profile_summarize_r370(&profile, 10.0, 0.025, &summary);

    if (!summary.gpu_timing_available) {
        fprintf(stderr, "expected GPU timing to be available\n");
        return 1;
    }
    if (summary.scope_status != VK_R370_SCOPE_COMPLETE) {
        fprintf(stderr, "expected complete scope attribution\n");
        return 1;
    }
    if (!almost_equal(summary.submit_ms, 2.0)) {
        fprintf(stderr, "submit_ms mismatch: %.12f\n", summary.submit_ms);
        return 1;
    }
    if (!almost_equal(summary.wait_ms, 1.0)) {
        fprintf(stderr, "wait_ms mismatch: %.12f\n", summary.wait_ms);
        return 1;
    }
    if (!almost_equal(summary.gpu_cmd_ms, 6.0)) {
        fprintf(stderr, "gpu_cmd_ms mismatch: %.12f\n", summary.gpu_cmd_ms);
        return 1;
    }
    if (!almost_equal(summary.host_gap_ms, 4.0)) {
        fprintf(stderr, "host_gap_ms mismatch: %.12f\n", summary.host_gap_ms);
        return 1;
    }
    if (!almost_equal(summary.dispatch_record_ms, 1.2)) {
        fprintf(stderr, "dispatch_record_ms mismatch: %.12f\n", summary.dispatch_record_ms);
        return 1;
    }
    if (!almost_equal(summary.dispatch_submit_ms, 0.5)) {
        fprintf(stderr, "dispatch_submit_ms mismatch: %.12f\n", summary.dispatch_submit_ms);
        return 1;
    }
    if (!almost_equal(summary.dispatch_wait_ms, 7.0)) {
        fprintf(stderr, "dispatch_wait_ms mismatch: %.12f\n", summary.dispatch_wait_ms);
        return 1;
    }
    if (!almost_equal(summary.dispatch_wait_non_gpu_ms, 1.0)) {
        fprintf(stderr, "dispatch_wait_non_gpu_ms mismatch: %.12f\n", summary.dispatch_wait_non_gpu_ms);
        return 1;
    }
    if (!almost_equal(summary.dispatch_reset_ms, 0.3)) {
        fprintf(stderr, "dispatch_reset_ms mismatch: %.12f\n", summary.dispatch_reset_ms);
        return 1;
    }
    if (!almost_equal(summary.profile_snapshot_ms, 0.025)) {
        fprintf(stderr, "profile_snapshot_ms mismatch: %.12f\n", summary.profile_snapshot_ms);
        return 1;
    }
    if (!almost_equal(summary.d_scope_ms, 3.0)) {
        fprintf(stderr, "d_scope_ms mismatch: %.12f\n", summary.d_scope_ms);
        return 1;
    }
    if (!almost_equal(summary.p_scope, 0.3)) {
        fprintf(stderr, "p_scope mismatch: %.12f\n", summary.p_scope);
        return 1;
    }
    if (!almost_equal(summary.other_host_ms, 1.0)) {
        fprintf(stderr, "other_host_ms mismatch: %.12f\n", summary.other_host_ms);
        return 1;
    }
    if (!almost_equal(summary.net_delta_optimistic, 0.3)) {
        fprintf(stderr, "net_delta_optimistic mismatch: %.12f\n", summary.net_delta_optimistic);
        return 1;
    }
    if (!almost_equal(summary.optimistic_speedup, 1.0 / 0.7)) {
        fprintf(stderr, "optimistic_speedup mismatch: %.12f\n", summary.optimistic_speedup);
        return 1;
    }

    profile.incomplete_timing_count = 1;
    vk_profile_summarize_r370(&profile, 10.0, 0.025, &summary);

    if (summary.gpu_timing_available) {
        fprintf(stderr, "expected GPU timing to be unavailable\n");
        return 1;
    }
    if (summary.scope_status != VK_R370_SCOPE_PARTIAL) {
        fprintf(stderr, "expected partial scope attribution\n");
        return 1;
    }
    if (!almost_equal(summary.submit_ms, 2.0) || !almost_equal(summary.wait_ms, 1.0)) {
        fprintf(stderr, "CPU-side timings should still be reported\n");
        return 1;
    }
    if (!almost_equal(summary.dispatch_record_ms, 1.2) ||
        !almost_equal(summary.dispatch_submit_ms, 0.5) ||
        !almost_equal(summary.dispatch_wait_ms, 7.0)) {
        fprintf(stderr, "scope buckets should still be reported on partial attribution\n");
        return 1;
    }

    printf("[PASS] R370 profile summary helper\n");
    return 0;
}
