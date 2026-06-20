/*
 * k_ssm.h — R357 M3 slice 1 — K_SSM offline calculator (K-probe)
 *
 * Defines the tiny ternary "K-probe" SSM used to compute K_SSM(D), a
 * computable MDL upper bound on the ideal dataset complexity under the
 * ternary-SSM class:
 *
 *     K_SSM(D) = P * log2(3) / T + CE_theta_hat(D)   [bits/byte]
 *
 * where P = 4168 ternary params of the K-probe and CE_theta_hat is the
 * per-byte cross-entropy of the K-probe after a short calibration fit
 * on D. This is an honest upper bound on K_SSM_ideal because theta_hat
 * is one feasible point in {-1,0,+1}^P.
 *
 * The K-probe is INTENTIONALLY distinct from the v12 production model:
 *   - 1 layer, d_state = 8, d_inner = 8, vocab = 256 (byte-level)
 *   - Straight-through estimator ternary quantisation
 *   - Plain SGD, 3 epochs, no momentum, no schedule
 *   - Self-contained forward/backward (does NOT reuse ssm_scan_forward
 *     nor ssm_scan_backward — see r357-m3-k-ssm-handoff.md section 9)
 *
 * Seed is fixed. Any change to K_PROBE_SEED invalidates cached artifacts.
 */

#ifndef K_SSM_H
#define K_SSM_H

#include <stdint.h>
#include <stddef.h>

#define K_PROBE_LAYERS        1
#define K_PROBE_D_STATE       8
#define K_PROBE_D_INNER       8
#define K_PROBE_VOCAB         256
#define K_PROBE_P_EMBED       (K_PROBE_VOCAB * K_PROBE_D_INNER)       /* 2048 */
#define K_PROBE_P_B           (K_PROBE_D_STATE)                       /* 8    */
#define K_PROBE_P_C           (K_PROBE_D_INNER * K_PROBE_D_STATE)     /* 64   */
#define K_PROBE_P_HEAD        (K_PROBE_D_INNER * K_PROBE_VOCAB)       /* 2048 */
#define K_PROBE_P_TOTAL       (K_PROBE_P_EMBED + K_PROBE_P_B + \
                               K_PROBE_P_C + K_PROBE_P_HEAD)          /* 4168 */

#define K_PROBE_SEED          0xB35D11ull
#define K_PROBE_CALIB_SEQ_LEN 256
#define K_PROBE_CALIB_BATCH   32
#define K_PROBE_CALIB_EPOCHS  3
#define K_PROBE_CALIB_LR      0.1f
#define K_PROBE_CALIB_NMAX    8192          /* ~2 MiB cap on training chunks */
#define K_PROBE_WALL_SECS     60            /* hard wall-clock cap           */

/* log2(3) — fixed constant for the two-part MDL penalty. */
#define K_SSM_LOG2_3          1.5849625007211563

typedef struct KProbeSsm {
    /* Ternary weights stored as int8 in {-1, 0, +1}. */
    int8_t  embed[K_PROBE_P_EMBED]; /* [vocab, d_inner] */
    int8_t  B    [K_PROBE_P_B];     /* [d_state]        */
    int8_t  C    [K_PROBE_P_C];     /* [d_inner, d_state] row-major */
    int8_t  head [K_PROBE_P_HEAD];  /* [d_inner, vocab] row-major   */

    /* Real-valued auxiliaries (not counted in P). */
    float   A   [K_PROBE_D_STATE];  /* decay in (0,1)                */
    float   temperature;            /* output softmax temperature    */

    uint64_t seed;                  /* PRNG seed used for init       */
    int      calibrated;            /* 0 = raw init, 1 = post-fit    */
} KProbeSsm;

/* Allocation / lifetime. */
KProbeSsm *k_probe_alloc(uint64_t seed);
void       k_probe_free(KProbeSsm *m);

/* Persistence (little-endian raw dump, magic-tagged). */
int  k_probe_save(const KProbeSsm *m, const char *path);
int  k_probe_load(      KProbeSsm *m, const char *path);

/*
 * k_probe_calibrate
 *
 * Runs the tiny STE-SGD calibration loop on the byte stream (bytes, n).
 * Honours the 60-second wall-clock cap: if exceeded, the loop aborts
 * cleanly and the partially fitted weights are retained.
 *
 * Returns 0 on success, 1 if wall-clock cap was hit, -1 on error.
 */
int  k_probe_calibrate(KProbeSsm *m,
                       const uint8_t *bytes,
                       size_t n,
                       int    epochs,
                       float  lr,
                       int    max_seconds);

/*
 * k_probe_measure_ce_bits_per_byte
 *
 * Single forward-only pass over the full byte stream. Returns the mean
 * per-byte cross-entropy in bits/byte (equals bits/token for byte vocab).
 * Returns +INFINITY if any numerical failure is detected.
 */
double k_probe_measure_ce_bits_per_byte(const KProbeSsm *m,
                                        const uint8_t *bytes,
                                        size_t n);

/*
 * k_ssm_compute
 *
 * K_SSM(D) = P * log2(3) / T_full_bytes + ce_bits_per_byte
 *
 * T_full_bytes is the FULL dataset length (not the calibration subset).
 */
double k_ssm_compute(double ce_bits_per_byte, size_t T_full_bytes);

#endif /* K_SSM_H */
