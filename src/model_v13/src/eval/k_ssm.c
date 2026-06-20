/*
 * k_ssm.c — R357 M3 slice 1 — K_SSM offline calculator (K-probe)
 *
 * Self-contained tiny ternary SSM used only to compute an MDL upper
 * bound on dataset complexity. Does NOT reuse ssm_scan.c: the K-probe
 * is architecturally separate from the v12 production path and must
 * not drag in production quantisation / BPTT behaviour.
 */

#include "k_ssm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* =========================================================================
 * PRNG — deterministic xorshift64 for reproducible K-probe initialisation.
 * ========================================================================= */

static uint64_t xorshift64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}

/* Uniform ternary sample in {-1, 0, +1}. */
static int8_t rand_ternary(uint64_t *s) {
    uint64_t r = xorshift64(s) % 3u;
    return (int8_t)((int)r - 1);
}

/* =========================================================================
 * Monotonic clock helper — portable enough for POSIX + mingw.
 * ========================================================================= */

static double monotonic_seconds(void) {
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
    }
#endif
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

/* =========================================================================
 * Allocation / init.
 * ========================================================================= */

KProbeSsm *k_probe_alloc(uint64_t seed) {
    KProbeSsm *m = (KProbeSsm *)calloc(1, sizeof(KProbeSsm));
    if (!m) return NULL;

    m->seed = seed;
    m->calibrated = 0;
    m->temperature = 1.0f;

    for (int i = 0; i < K_PROBE_D_STATE; i++) {
        m->A[i] = 0.9f;
    }

    uint64_t s = seed ? seed : 1ull;
    for (int i = 0; i < K_PROBE_P_EMBED; i++) m->embed[i] = rand_ternary(&s);
    for (int i = 0; i < K_PROBE_P_B;     i++) m->B    [i] = rand_ternary(&s);
    for (int i = 0; i < K_PROBE_P_C;     i++) m->C    [i] = rand_ternary(&s);
    for (int i = 0; i < K_PROBE_P_HEAD;  i++) m->head [i] = rand_ternary(&s);

    return m;
}

void k_probe_free(KProbeSsm *m) { free(m); }

/* =========================================================================
 * Persistence.
 * ========================================================================= */

#define K_PROBE_MAGIC 0x4B53534Du /* "KSSM" */

int k_probe_save(const KProbeSsm *m, const char *path) {
    if (!m || !path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t magic = K_PROBE_MAGIC;
    uint32_t version = 1;
    if (fwrite(&magic,   sizeof magic,   1, f) != 1) { fclose(f); return -1; }
    if (fwrite(&version, sizeof version, 1, f) != 1) { fclose(f); return -1; }
    if (fwrite(m, sizeof *m, 1, f) != 1)             { fclose(f); return -1; }
    fclose(f);
    return 0;
}

int k_probe_load(KProbeSsm *m, const char *path) {
    if (!m || !path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t magic = 0, version = 0;
    if (fread(&magic,   sizeof magic,   1, f) != 1)  { fclose(f); return -1; }
    if (fread(&version, sizeof version, 1, f) != 1)  { fclose(f); return -1; }
    if (magic != K_PROBE_MAGIC || version != 1)      { fclose(f); return -1; }
    if (fread(m, sizeof *m, 1, f) != 1)              { fclose(f); return -1; }
    fclose(f);
    return 0;
}

/* =========================================================================
 * Numerical helpers.
 * ========================================================================= */

/* Softmax in bits: returns sum of exp for downstream gradient, writes prob. */
static void softmax(const float *logits, float *probs, int n) {
    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = expf(logits[i] - mx);
        probs[i] = e;
        sum += e;
    }
    float inv = 1.0f / (sum > 0.0f ? sum : 1e-30f);
    for (int i = 0; i < n; i++) probs[i] *= inv;
}

/* =========================================================================
 * Forward pass for one sequence of length L over bytes[0..L].
 *
 * Arrays (per call):
 *   h_cache: [(L+1) * D_STATE]  — h_cache[0..D_STATE] = zero state
 *   y_cache: [L * D_INNER]      — y[t] = C * h[t+1]
 *   e_cache: [L * D_INNER]      — embed row for input token x_t
 *   probs:   [L * VOCAB]        — softmax predictions
 *
 * Returns total cross-entropy over the L predicted positions, in nats.
 * The prediction at position t is x_{t+1}: so we use tokens[0..L] and
 * the last target is tokens[L].
 * ========================================================================= */

/* =========================================================================
 * Calibration uses a real-valued shadow of the weights so gradients flow
 * cleanly. The shadow is updated by SGD, then projected to the ternary
 * alphabet {-1, 0, +1} via sign-with-deadzone (deadzone = std/3) into the
 * KProbeSsm struct. STE interpretation: forward on continuous weights,
 * ternary projection is a separate constrainer applied at the end of
 * calibration (and recorded in m->{embed,B,C,head}).
 *
 * Only ternary weights are updated. A and temperature are held fixed.
 * ========================================================================= */

typedef struct {
    float gE[K_PROBE_P_EMBED];
    float gB[K_PROBE_P_B];
    float gC[K_PROBE_P_C];
    float gH[K_PROBE_P_HEAD];
} KProbeGrad;

static KProbeGrad *alloc_grad(void) {
    return (KProbeGrad *)calloc(1, sizeof(KProbeGrad));
}

typedef struct KProbeReal {
    float E[K_PROBE_P_EMBED];
    float B[K_PROBE_P_B];
    float C[K_PROBE_P_C];
    float H[K_PROBE_P_HEAD];
} KProbeReal;

/* Uniform(-range, +range) fill of a real weight block. */
static void fill_uniform(float *dst, int n, uint64_t *s, float range) {
    for (int i = 0; i < n; i++) {
        uint64_t r = xorshift64(s);
        float u = ((float)(r >> 11) * (1.0f / 9007199254740992.0f)); /* [0,1) */
        dst[i] = (u * 2.0f - 1.0f) * range;
    }
}

static void init_real(KProbeReal *r, uint64_t seed) {
    uint64_t s = seed ^ 0xD1B54A32D192ED03ull;
    fill_uniform(r->E, K_PROBE_P_EMBED, &s, 0.5f);
    fill_uniform(r->B, K_PROBE_P_B,     &s, 0.5f);
    fill_uniform(r->C, K_PROBE_P_C,     &s, 0.5f);
    fill_uniform(r->H, K_PROBE_P_HEAD,  &s, 0.5f);
}

/* Sign-with-deadzone quantiser. */
static void quantise_block(const float *src, int8_t *dst, int n) {
    double sq = 0.0;
    for (int i = 0; i < n; i++) sq += (double)src[i] * (double)src[i];
    float std = (float)sqrt(sq / (double)(n > 0 ? n : 1));
    float dz  = std / 3.0f;
    for (int i = 0; i < n; i++) {
        float v = src[i];
        if (v >  dz) dst[i] =  1;
        else if (v < -dz) dst[i] = -1;
        else dst[i] = 0;
    }
}

static void requantise_all(const KProbeReal *r, KProbeSsm *m) {
    quantise_block(r->E, m->embed, K_PROBE_P_EMBED);
    quantise_block(r->B, m->B,     K_PROBE_P_B);
    quantise_block(r->C, m->C,     K_PROBE_P_C);
    quantise_block(r->H, m->head,  K_PROBE_P_HEAD);
}

/* =========================================================================
 * Forward pass on the REAL (continuous) shadow weights. Caches states
 * needed for the matching backward pass.
 *
 * Forward recurrence:
 *   e_t[j]       = E[x_t, j]                        (d_inner = 8)
 *   h_curr_i     = A_i * h_prev_i + B_i * e_t_i     (d_state = d_inner = 8)
 *   y_t[j]       = sum_i C[j,i] * h_curr_i
 *   logits[v]    = (sum_j H[j,v] * y_t[j]) * inv_T
 *   p            = softmax(logits)
 *   target       = tokens[t + 1]
 *   L_t          = -log p[target]
 * ========================================================================= */

static double k_probe_forward_seq_real(const KProbeReal *r,
                                       const float *A,
                                       float temperature,
                                       const uint8_t *tokens,
                                       int L,
                                       float *h_cache,
                                       float *y_cache,
                                       float *e_cache,
                                       float *probs)
{
    const int DS = K_PROBE_D_STATE;
    const int DI = K_PROBE_D_INNER;
    const int V  = K_PROBE_VOCAB;
    float inv_T = 1.0f / (temperature > 1e-6f ? temperature : 1e-6f);

    memset(h_cache, 0, (size_t)(L + 1) * DS * sizeof(float));

    double nats_total = 0.0;

    for (int t = 0; t < L; t++) {
        uint8_t x_t = tokens[t];

        float *e_t = e_cache + t * DI;
        for (int j = 0; j < DI; j++) e_t[j] = r->E[(int)x_t * DI + j];

        const float *h_prev = h_cache +  t      * DS;
        float       *h_curr = h_cache + (t + 1) * DS;
        for (int i = 0; i < DS; i++) {
            h_curr[i] = A[i] * h_prev[i] + r->B[i] * e_t[i];
        }

        float *y_t = y_cache + t * DI;
        for (int j = 0; j < DI; j++) {
            float acc = 0.0f;
            const float *Crow = r->C + j * DS;
            for (int i = 0; i < DS; i++) acc += Crow[i] * h_curr[i];
            y_t[j] = acc;
        }

        float logits[K_PROBE_VOCAB];
        for (int v = 0; v < V; v++) {
            float acc = 0.0f;
            for (int j = 0; j < DI; j++) acc += r->H[j * V + v] * y_t[j];
            logits[v] = acc * inv_T;
        }

        float *p = probs + t * V;
        softmax(logits, p, V);

        uint8_t target = tokens[t + 1];
        float pt = p[target];
        if (pt < 1e-30f) pt = 1e-30f;
        nats_total += -logf(pt);
    }

    return nats_total;
}

/* Backward pass on the REAL shadow weights. Accumulates gradients in g. */
static void k_probe_backward_seq_real(const KProbeReal *r,
                                      const float *A,
                                      float temperature,
                                      const uint8_t *tokens,
                                      int L,
                                      const float *h_cache,
                                      const float *y_cache,
                                      const float *e_cache,
                                      const float *probs,
                                      KProbeGrad *g)
{
    const int DS = K_PROBE_D_STATE;
    const int DI = K_PROBE_D_INNER;
    const int V  = K_PROBE_VOCAB;
    float inv_T = 1.0f / (temperature > 1e-6f ? temperature : 1e-6f);

    float dy_t  [K_PROBE_D_INNER];
    float dh    [K_PROBE_D_STATE];
    float dh_acc[K_PROBE_D_STATE];
    memset(dh_acc, 0, sizeof dh_acc);

    for (int t = L - 1; t >= 0; t--) {
        const float *p = probs + t * V;
        uint8_t target = tokens[t + 1];
        const float *y_t    = y_cache + t * DI;
        const float *e_t    = e_cache + t * DI;
        const float *h_curr = h_cache + (t + 1) * DS;
        uint8_t x_t = tokens[t];

        memset(dy_t, 0, sizeof dy_t);
        for (int v = 0; v < V; v++) {
            float dl = (p[v] - (v == target ? 1.0f : 0.0f)) * inv_T;
            for (int j = 0; j < DI; j++) {
                g->gH[j * V + v] += y_t[j] * dl;
                dy_t[j] += r->H[j * V + v] * dl;
            }
        }

        for (int i = 0; i < DS; i++) dh[i] = dh_acc[i];
        for (int j = 0; j < DI; j++) {
            float dyj = dy_t[j];
            for (int i = 0; i < DS; i++) {
                g->gC[j * DS + i] += dyj * h_curr[i];
                dh[i] += r->C[j * DS + i] * dyj;
            }
        }

        for (int i = 0; i < DS; i++) {
            float dhi = dh[i];
            g->gB[i] += dhi * e_t[i];
            g->gE[(int)x_t * DI + i] += dhi * r->B[i];
            dh_acc[i] = dhi * A[i];
        }
    }
}

/* =========================================================================
 * Calibration loop.
 * ========================================================================= */

int k_probe_calibrate(KProbeSsm *m,
                      const uint8_t *bytes,
                      size_t n,
                      int    epochs,
                      float  lr,
                      int    max_seconds)
{
    if (!m || !bytes || n < (size_t)(K_PROBE_CALIB_SEQ_LEN + 2)) return -1;
    if (epochs       <= 0) epochs       = K_PROBE_CALIB_EPOCHS;
    if (lr           <= 0) lr           = K_PROBE_CALIB_LR;
    if (max_seconds  <= 0) max_seconds  = K_PROBE_WALL_SECS;

    const int L = K_PROBE_CALIB_SEQ_LEN;
    const int B = K_PROBE_CALIB_BATCH;

    /* Chunk stride: evenly spaced windows over the dataset, capped at
     * K_PROBE_CALIB_NMAX sequences (~2 MiB). This keeps calibration at
     * seconds-scale independently of dataset size. */
    size_t max_chunks = (n - 1) / (size_t)L;
    if (max_chunks == 0) return -1;
    if (max_chunks > (size_t)K_PROBE_CALIB_NMAX) max_chunks = (size_t)K_PROBE_CALIB_NMAX;

    /* Offsets evenly spaced so we cover the whole file even when we clip. */
    size_t *offsets = (size_t *)malloc(max_chunks * sizeof(size_t));
    if (!offsets) return -1;
    size_t span = n - (size_t)L - 1;
    if (max_chunks == 1) {
        offsets[0] = 0;
    } else {
        for (size_t i = 0; i < max_chunks; i++) {
            offsets[i] = (span * i) / (max_chunks - 1);
        }
    }

    KProbeReal *r = (KProbeReal *)calloc(1, sizeof(KProbeReal));
    KProbeGrad *g = alloc_grad();
    float *h_cache = (float *)malloc((size_t)(L + 1) * K_PROBE_D_STATE * sizeof(float));
    float *y_cache = (float *)malloc((size_t)L * K_PROBE_D_INNER * sizeof(float));
    float *e_cache = (float *)malloc((size_t)L * K_PROBE_D_INNER * sizeof(float));
    float *probs   = (float *)malloc((size_t)L * K_PROBE_VOCAB  * sizeof(float));
    if (!r || !g || !h_cache || !y_cache || !e_cache || !probs) {
        free(r); free(g); free(h_cache); free(y_cache); free(e_cache); free(probs);
        free(offsets);
        return -1;
    }

    init_real(r, m->seed);

    double t_start = monotonic_seconds();
    int aborted = 0;

    /* Deterministic shuffling PRNG for chunk ordering. */
    uint64_t sh = m->seed ^ 0x9E3779B97F4A7C15ull;

    for (int ep = 0; ep < epochs && !aborted; ep++) {
        /* Fisher-Yates shuffle of chunk indices. */
        for (size_t i = max_chunks - 1; i > 0; i--) {
            size_t j = (size_t)(xorshift64(&sh) % (uint64_t)(i + 1));
            size_t tmp = offsets[i]; offsets[i] = offsets[j]; offsets[j] = tmp;
        }

        size_t processed = 0;
        while (processed < max_chunks) {
            if (monotonic_seconds() - t_start > (double)max_seconds) {
                aborted = 1;
                break;
            }

            size_t batch_end = processed + (size_t)B;
            if (batch_end > max_chunks) batch_end = max_chunks;
            size_t bsz = batch_end - processed;

            memset(g, 0, sizeof *g);

            for (size_t bi = 0; bi < bsz; bi++) {
                const uint8_t *seq = bytes + offsets[processed + bi];
                (void)k_probe_forward_seq_real(r, m->A, m->temperature,
                                               seq, L,
                                               h_cache, y_cache, e_cache, probs);
                k_probe_backward_seq_real(r, m->A, m->temperature,
                                          seq, L,
                                          h_cache, y_cache, e_cache, probs, g);
            }

            float scale = lr / (float)((int)bsz * L);
            for (int i = 0; i < K_PROBE_P_EMBED; i++) r->E[i] -= scale * g->gE[i];
            for (int i = 0; i < K_PROBE_P_B;     i++) r->B[i] -= scale * g->gB[i];
            for (int i = 0; i < K_PROBE_P_C;     i++) r->C[i] -= scale * g->gC[i];
            for (int i = 0; i < K_PROBE_P_HEAD;  i++) r->H[i] -= scale * g->gH[i];

            processed = batch_end;
        }
    }

    /* Project the continuous shadow to ternary — this is the feasible
     * theta_hat in {-1,0,+1}^P that K_SSM is built on. */
    requantise_all(r, m);
    m->calibrated = 1;

    free(r); free(g); free(h_cache); free(y_cache); free(e_cache); free(probs);
    free(offsets);
    return aborted ? 1 : 0;
}

/* =========================================================================
 * Full-dataset CE measurement (forward-only) in bits/byte.
 *
 * Processes the stream as one long autoregressive sequence, carrying h
 * across the whole file (no batching). For memory, we reuse a single
 * scratch (L+1) cache block and advance h across blocks.
 * ========================================================================= */

double k_probe_measure_ce_bits_per_byte(const KProbeSsm *m,
                                        const uint8_t *bytes,
                                        size_t n)
{
    if (!m || !bytes || n < 2) return INFINITY;

    const int DS = K_PROBE_D_STATE;
    const int DI = K_PROBE_D_INNER;
    const int V  = K_PROBE_VOCAB;
    const int BLK = 1024;

    float *h_cache = (float *)malloc((size_t)(BLK + 1) * DS * sizeof(float));
    float *y_cache = (float *)malloc((size_t)BLK * DI * sizeof(float));
    float *e_cache = (float *)malloc((size_t)BLK * DI * sizeof(float));
    float *probs   = (float *)malloc((size_t)BLK * V  * sizeof(float));
    if (!h_cache || !y_cache || !e_cache || !probs) {
        free(h_cache); free(y_cache); free(e_cache); free(probs);
        return INFINITY;
    }

    /* Persistent state carried across blocks. */
    float h_state[K_PROBE_D_STATE];
    memset(h_state, 0, sizeof h_state);

    double nats_total = 0.0;
    size_t predictions = 0;

    size_t pos = 0;
    /* We need tokens[t] and tokens[t+1] for each prediction, so each
     * block of length L needs L+1 bytes from the stream. */
    while (pos + 2 <= n) {
        size_t avail = n - pos - 1;            /* predictions possible */
        int L = (int)(avail < (size_t)BLK ? avail : (size_t)BLK);
        if (L <= 0) break;

        /* Prime h_cache[0] from persistent state. */
        memcpy(h_cache, h_state, DS * sizeof(float));

        for (int t = 0; t < L; t++) {
            uint8_t x_t = bytes[pos + (size_t)t];
            float *e_t = e_cache + t * DI;
            for (int j = 0; j < DI; j++) {
                e_t[j] = (float)m->embed[(int)x_t * DI + j];
            }
            const float *h_prev = h_cache +  t      * DS;
            float       *h_curr = h_cache + (t + 1) * DS;
            for (int i = 0; i < DS; i++) {
                h_curr[i] = m->A[i] * h_prev[i] + (float)m->B[i] * e_t[i];
            }
            float *y_t = y_cache + t * DI;
            for (int j = 0; j < DI; j++) {
                float acc = 0.0f;
                const int8_t *Crow = m->C + j * DS;
                for (int i = 0; i < DS; i++) acc += (float)Crow[i] * h_curr[i];
                y_t[j] = acc;
            }
            float logits[K_PROBE_VOCAB];
            float inv_T = 1.0f / (m->temperature > 1e-6f ? m->temperature : 1e-6f);
            for (int v = 0; v < V; v++) {
                float acc = 0.0f;
                for (int j = 0; j < DI; j++) {
                    acc += (float)m->head[j * V + v] * y_t[j];
                }
                logits[v] = acc * inv_T;
            }
            float *p = probs + t * V;
            softmax(logits, p, V);

            uint8_t target = bytes[pos + (size_t)t + 1];
            float pt = p[target];
            if (pt < 1e-30f) pt = 1e-30f;
            double nll = -log((double)pt);
            if (!isfinite(nll)) {
                free(h_cache); free(y_cache); free(e_cache); free(probs);
                return INFINITY;
            }
            nats_total += nll;
            predictions++;
        }

        /* Carry final hidden state. */
        memcpy(h_state, h_cache + (size_t)L * DS, DS * sizeof(float));
        pos += (size_t)L;
    }

    free(h_cache); free(y_cache); free(e_cache); free(probs);

    if (predictions == 0) return INFINITY;
    double bits_per_byte = (nats_total / (double)predictions) / log(2.0);
    if (!isfinite(bits_per_byte)) return INFINITY;
    return bits_per_byte;
}

/* =========================================================================
 * K_SSM(D) = P * log2(3) / T + CE_theta_hat(D)          [bits/byte]
 * ========================================================================= */

double k_ssm_compute(double ce_bits_per_byte, size_t T_full_bytes) {
    if (T_full_bytes == 0) return INFINITY;
    if (!isfinite(ce_bits_per_byte)) return INFINITY;
    double model_penalty = ((double)K_PROBE_P_TOTAL * K_SSM_LOG2_3) / (double)T_full_bytes;
    return model_penalty + ce_bits_per_byte;
}
