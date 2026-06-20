/*
 * eval_k_ssm.c — R357 M3 slice 1 — offline K_SSM calculator CLI.
 *
 * Usage:
 *   eval_k_ssm <dataset_path> [--raw-bytes <path>] [--ce-theta <float>]
 *              [--kprobe-cache <path>] [--max-seconds <int>]
 *
 * Emits a single JSON line to stdout with K_SSM(D) and the inputs used
 * to compute it. No interaction with the training binary, no curriculum
 * wiring, no shader, no config. This is a pure read-only inspection tool.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "../src/eval/k_ssm.h"
#include "../src/io/dataset.h"

static void print_usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s <dataset_path> [--raw-bytes <path>] [--ce-theta <float>]\n"
        "                     [--kprobe-cache <path>] [--max-seconds <int>]\n",
        argv0);
}

static uint8_t *read_raw_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    rewind(f);
    uint8_t *buf = (uint8_t *)malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    *out_len = (size_t)sz;
    return buf;
}

/* Try to read the dataset as a byte stream. Returns NULL on vocab mismatch. */
static uint8_t *dataset_as_bytes(const char *path, size_t *out_len) {
    Dataset *ds = dataset_load(path);
    if (!ds) return NULL;

    uint16_t max_tok = 0;
    for (size_t i = 0; i < ds->n_tokens; i++) {
        if (ds->data[i] > max_tok) max_tok = ds->data[i];
    }
    if (max_tok > 255) {
        fprintf(stderr,
            "eval_k_ssm: dataset '%s' has max token %u > 255 — refusing. "
            "Supply --raw-bytes instead.\n", path, (unsigned)max_tok);
        dataset_free(ds);
        return NULL;
    }

    uint8_t *buf = (uint8_t *)malloc(ds->n_tokens);
    if (!buf) { dataset_free(ds); return NULL; }
    for (size_t i = 0; i < ds->n_tokens; i++) {
        buf[i] = (uint8_t)ds->data[i];
    }
    *out_len = ds->n_tokens;
    dataset_free(ds);
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 2) { print_usage(argv[0]); return 2; }

    const char *dataset_path  = argv[1];
    const char *raw_bytes     = NULL;
    const char *kprobe_cache  = NULL;
    int   max_seconds         = K_PROBE_WALL_SECS;
    double ce_theta_prod      = NAN; /* production model CE; optional */
    int    have_ce_theta      = 0;

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--raw-bytes") && i + 1 < argc) {
            raw_bytes = argv[++i];
        } else if (!strcmp(argv[i], "--ce-theta") && i + 1 < argc) {
            ce_theta_prod = atof(argv[++i]);
            have_ce_theta = 1;
        } else if (!strcmp(argv[i], "--kprobe-cache") && i + 1 < argc) {
            kprobe_cache = argv[++i];
        } else if (!strcmp(argv[i], "--max-seconds") && i + 1 < argc) {
            max_seconds = atoi(argv[++i]);
        } else {
            fprintf(stderr, "eval_k_ssm: unknown argument '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }

    size_t n = 0;
    uint8_t *bytes = NULL;
    if (raw_bytes) {
        bytes = read_raw_file(raw_bytes, &n);
        if (!bytes) {
            fprintf(stderr, "eval_k_ssm: failed to read raw bytes from '%s'\n", raw_bytes);
            return 1;
        }
    } else {
        bytes = dataset_as_bytes(dataset_path, &n);
        if (!bytes) return 1;
    }

    KProbeSsm *m = k_probe_alloc(K_PROBE_SEED);
    if (!m) { free(bytes); return 1; }

    int loaded_from_cache = 0;
    if (kprobe_cache && k_probe_load(m, kprobe_cache) == 0) {
        loaded_from_cache = 1;
    }

    if (!loaded_from_cache) {
        int rc = k_probe_calibrate(m, bytes, n,
                                   K_PROBE_CALIB_EPOCHS,
                                   K_PROBE_CALIB_LR,
                                   max_seconds);
        if (rc < 0) {
            fprintf(stderr, "eval_k_ssm: calibration failed\n");
            k_probe_free(m); free(bytes); return 1;
        }
        if (rc == 1) {
            fprintf(stderr, "eval_k_ssm: warning — calibration hit %d-second wall-clock cap\n",
                    max_seconds);
        }
        if (kprobe_cache) (void)k_probe_save(m, kprobe_cache);
    }

    double ce_bpb = k_probe_measure_ce_bits_per_byte(m, bytes, n);
    double k_ssm  = k_ssm_compute(ce_bpb, n);

    /* K_SSM vs uniform sanity gate (section 3.4): if we cannot beat
     * log2(256) = 8 bits/byte, emit +inf. */
    if (ce_bpb >= 8.0 || !isfinite(ce_bpb)) {
        k_ssm = INFINITY;
    }

    printf("{");
    printf("\"dataset\":\"%s\",", dataset_path);
    printf("\"T\":%zu,", n);
    printf("\"P\":%d,", K_PROBE_P_TOTAL);
    if (isfinite(ce_bpb)) {
        printf("\"CE_theta_hat_bits_per_byte\":%.6f,", ce_bpb);
    } else {
        printf("\"CE_theta_hat_bits_per_byte\":null,");
    }
    if (isfinite(k_ssm)) {
        printf("\"K_SSM\":%.6f,", k_ssm);
    } else {
        printf("\"K_SSM\":null,");
    }
    printf("\"K_gzip_reference\":null,");
    if (have_ce_theta && isfinite(k_ssm) && k_ssm > 0.0) {
        double q_ssm = (ce_theta_prod / (log(2.0) * k_ssm)) - 1.0;
        printf("\"Q_SSM\":%.6f,", q_ssm);
    } else {
        printf("\"Q_SSM\":null,");
    }
    printf("\"Q_gzip\":null");
    printf("}\n");

    /* TODO: CE_theta from live model — slice 2 hook. */

    k_probe_free(m);
    free(bytes);
    return 0;
}
