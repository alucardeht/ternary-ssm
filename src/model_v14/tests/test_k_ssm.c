/*
 * test_k_ssm.c — R357 M3 slice 1 smoke tests for the K-probe / K_SSM.
 *
 * Exactly 9 hard assertions across 3 synthetic inputs (T = 65536 bytes):
 *
 *   Test A — pure random bytes (xorshift64, seed 0xA1)
 *     A.1 finite  — K_SSM(A) is a finite float
 *     A.2 near-uniform — K_SSM(A) >= 7.5 bits/byte
 *     A.3 non-zero — K_SSM(A) > 0
 *
 *   Test B — highly regular (0x41 repeated)
 *     B.1 low_ce — K_SSM(B) < 0.5 bits/byte
 *     B.2 strict_gap_vs_A — K_SSM(B) < K_SSM(A) - 5.0
 *
 *   Test C — mixed (Running-XOR)
 *     C.1 half_uniform — K_SSM(C) < 4.0 bits/byte
 *     C.2 ordering — K_SSM(B) < K_SSM(C) < K_SSM(A)
 *
 *   Global:
 *     G.1 all_finite — K_SSM(A), K_SSM(B), K_SSM(C) all finite
 *     G.2 wall_clock — total calibration wall-clock < 60s for all three
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#include "../src/eval/k_ssm.h"

#define T_LEN 65536

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(label, cond) do {                                          \
    if (cond) { printf("  [PASS] %s\n", (label)); g_pass++; }            \
    else      { printf("  [FAIL] %s\n", (label)); g_fail++; }            \
} while (0)

static double now_secs(void) {
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
    }
#endif
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static uint64_t xs64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}

static void gen_random(uint8_t *buf, size_t n, uint64_t seed) {
    uint64_t s = seed;
    for (size_t i = 0; i < n; i++) buf[i] = (uint8_t)(xs64(&s) & 0xFFu);
}

static void gen_regular(uint8_t *buf, size_t n, uint8_t byte) {
    memset(buf, byte, n);
}

static void gen_running_xor(uint8_t *buf, size_t n) {
    uint8_t x = 0x5A;
    buf[0] = x;
    for (size_t t = 1; t < n; t++) {
        x = (uint8_t)(x ^ (uint8_t)((t - 1) & 0xFFu));
        buf[t] = x;
    }
}

static double run_case(const char *label, const uint8_t *bytes, size_t n) {
    KProbeSsm *m = k_probe_alloc(K_PROBE_SEED);
    if (!m) { fprintf(stderr, "alloc failed for %s\n", label); return INFINITY; }
    int rc = k_probe_calibrate(m, bytes, n,
                               K_PROBE_CALIB_EPOCHS,
                               K_PROBE_CALIB_LR,
                               K_PROBE_WALL_SECS);
    if (rc < 0) {
        fprintf(stderr, "calibrate failed for %s\n", label);
        k_probe_free(m);
        return INFINITY;
    }
    double ce = k_probe_measure_ce_bits_per_byte(m, bytes, n);
    double k  = k_ssm_compute(ce, n);
    printf("  %s: CE=%.4f bits/byte  K_SSM=%.4f bits/byte%s\n",
           label, ce, k, rc == 1 ? "  [wall-clock abort]" : "");
    k_probe_free(m);
    return k;
}

int main(void) {
    printf("test_k_ssm — R357 M3 slice 1 smoke test (T=%d bytes)\n", T_LEN);

    uint8_t *a = (uint8_t *)malloc(T_LEN);
    uint8_t *b = (uint8_t *)malloc(T_LEN);
    uint8_t *c = (uint8_t *)malloc(T_LEN);
    if (!a || !b || !c) { fprintf(stderr, "oom\n"); return 1; }

    gen_random      (a, T_LEN, 0xA1ull);
    gen_regular     (b, T_LEN, 0x41);
    gen_running_xor (c, T_LEN);

    double t0 = now_secs();
    double kA = run_case("A pure-random",   a, T_LEN);
    double kB = run_case("B highly-regular", b, T_LEN);
    double kC = run_case("C running-xor",   c, T_LEN);
    double wall = now_secs() - t0;
    printf("  total calibration+measure wall-clock: %.2fs\n", wall);

    printf("\nassertions:\n");

    /* Test A — pure random */
    CHECK("A.1 finite",                 isfinite(kA));
    CHECK("A.2 near-uniform (>=7.5)",   isfinite(kA) && kA >= 7.5);
    CHECK("A.3 non-zero",               isfinite(kA) && kA > 0.0);

    /* Test B — highly regular */
    CHECK("B.1 low_ce (<0.5)",          isfinite(kB) && kB < 0.5);
    CHECK("B.2 strict_gap_vs_A (<A-5)", isfinite(kA) && isfinite(kB) && kB < (kA - 5.0));

    /* Test C — mixed */
    CHECK("C.1 half_uniform (<4.0)",    isfinite(kC) && kC < 4.0);
    CHECK("C.2 ordering B<C<A",         isfinite(kA) && isfinite(kB) && isfinite(kC)
                                        && kB < kC && kC < kA);

    /* Global */
    CHECK("G.1 all_finite",             isfinite(kA) && isfinite(kB) && isfinite(kC));
    CHECK("G.2 wall_clock (<60s)",      wall < 60.0);

    free(a); free(b); free(c);

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
