#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "ternary_types.h"
#include "ternary_gemv.h"
#include "popcount_gemv.h"
#include "simd_detect.h"

typedef struct {
    int32_t rows;
    int32_t cols;
    int32_t warmup_iters;
    int32_t bench_iters;
    const char* name;
} bench_config_t;

#ifdef _WIN32
static double get_time_ms(void) {
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER count;

    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / freq.QuadPart * 1000.0;
}
#else
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}
#endif

static void generate_test_data(
    int8_t* W_dense, int32_t rows, int32_t cols,
    float* x, int32_t x_len,
    float* bias, int32_t bias_len
) {
    for (int32_t i = 0; i < rows * cols; i++) {
        int val = rand() % 3;
        W_dense[i] = (int8_t)(val - 1);
    }

    for (int32_t i = 0; i < x_len; i++) {
        x[i] = (float)(rand() % 1000) / 1000.0f;
    }

    for (int32_t i = 0; i < bias_len; i++) {
        bias[i] = 0.0f;
    }
}

static int is_finite_vector(const float* v, int32_t n) {
    for (int32_t i = 0; i < n; i++) {
        if (!isfinite(v[i])) return 0;
    }
    return 1;
}

static void bench_ternary_scalar(
    const bench_config_t* cfg,
    double* avg_time_ms, double* gops
) {
    int32_t rows = cfg->rows;
    int32_t cols = cfg->cols;

    int8_t* W_dense = (int8_t*)malloc(rows * cols * sizeof(int8_t));
    float* x = (float*)malloc(cols * sizeof(float));
    float* bias = (float*)malloc(rows * sizeof(float));
    float* y = (float*)malloc(rows * sizeof(float));

    ternary_matrix_t* W = ternary_matrix_alloc(rows, cols);

    generate_test_data(W_dense, rows, cols, x, cols, bias, rows);
    ternary_matrix_pack(W, W_dense);

    ternary_gemv_scalar(y, W, x, bias, rows, cols);
    if (!is_finite_vector(y, rows)) {
        printf("  ERROR: Invalid output detected\n");
        *avg_time_ms = 0.0;
        *gops = 0.0;
        goto cleanup;
    }

    for (int32_t i = 0; i < cfg->warmup_iters; i++) {
        ternary_gemv_scalar(y, W, x, bias, rows, cols);
    }

    double t_start = get_time_ms();
    for (int32_t i = 0; i < cfg->bench_iters; i++) {
        ternary_gemv_scalar(y, W, x, bias, rows, cols);
        volatile float sink = y[0];
        (void)sink;
    }
    double t_end = get_time_ms();

    double elapsed = t_end - t_start;
    *avg_time_ms = elapsed / cfg->bench_iters;
    *gops = (2.0 * rows * cols) / (*avg_time_ms * 1e6);

cleanup:
    free(W_dense);
    free(x);
    free(bias);
    free(y);
    ternary_matrix_free(W);
}

#ifdef ENGINE_AVX2
static void bench_ternary_avx2(
    const bench_config_t* cfg,
    double* avg_time_ms, double* gops
) {
    int32_t rows = cfg->rows;
    int32_t cols = cfg->cols;

    int8_t* W_dense = (int8_t*)malloc(rows * cols * sizeof(int8_t));
    float* x = (float*)malloc(cols * sizeof(float));
    float* bias = (float*)malloc(rows * sizeof(float));
    float* y = (float*)malloc(rows * sizeof(float));

    ternary_matrix_t* W = ternary_matrix_alloc(rows, cols);

    generate_test_data(W_dense, rows, cols, x, cols, bias, rows);
    ternary_matrix_pack(W, W_dense);

    ternary_gemv_avx2(y, W, x, bias, rows, cols);
    if (!is_finite_vector(y, rows)) {
        printf("  ERROR: Invalid output detected\n");
        *avg_time_ms = 0.0;
        *gops = 0.0;
        goto cleanup;
    }

    for (int32_t i = 0; i < cfg->warmup_iters; i++) {
        ternary_gemv_avx2(y, W, x, bias, rows, cols);
    }

    double t_start = get_time_ms();
    for (int32_t i = 0; i < cfg->bench_iters; i++) {
        ternary_gemv_avx2(y, W, x, bias, rows, cols);
        volatile float sink = y[0];
        (void)sink;
    }
    double t_end = get_time_ms();

    double elapsed = t_end - t_start;
    *avg_time_ms = elapsed / cfg->bench_iters;
    *gops = (2.0 * rows * cols) / (*avg_time_ms * 1e6);

cleanup:
    free(W_dense);
    free(x);
    free(bias);
    free(y);
    ternary_matrix_free(W);
}
#endif

static void bench_ternary_dispatch(
    const bench_config_t* cfg,
    double* avg_time_ms, double* gops
) {
    int32_t rows = cfg->rows;
    int32_t cols = cfg->cols;

    int8_t* W_dense = (int8_t*)malloc(rows * cols * sizeof(int8_t));
    float* x = (float*)malloc(cols * sizeof(float));
    float* bias = (float*)malloc(rows * sizeof(float));
    float* y = (float*)malloc(rows * sizeof(float));

    ternary_matrix_t* W = ternary_matrix_alloc(rows, cols);

    generate_test_data(W_dense, rows, cols, x, cols, bias, rows);
    ternary_matrix_pack(W, W_dense);

    ternary_gemv(y, W, x, bias, rows, cols);
    if (!is_finite_vector(y, rows)) {
        printf("  ERROR: Invalid output detected\n");
        *avg_time_ms = 0.0;
        *gops = 0.0;
        goto cleanup;
    }

    for (int32_t i = 0; i < cfg->warmup_iters; i++) {
        ternary_gemv(y, W, x, bias, rows, cols);
    }

    double t_start = get_time_ms();
    for (int32_t i = 0; i < cfg->bench_iters; i++) {
        ternary_gemv(y, W, x, bias, rows, cols);
        volatile float sink = y[0];
        (void)sink;
    }
    double t_end = get_time_ms();

    double elapsed = t_end - t_start;
    *avg_time_ms = elapsed / cfg->bench_iters;
    *gops = (2.0 * rows * cols) / (*avg_time_ms * 1e6);

cleanup:
    free(W_dense);
    free(x);
    free(bias);
    free(y);
    ternary_matrix_free(W);
}

static void bench_popcount_compute(
    const bench_config_t* cfg,
    double* avg_time_ms, double* gops
) {
    int32_t rows = cfg->rows;
    int32_t cols = cfg->cols;

    int8_t* W_dense = (int8_t*)malloc(rows * cols * sizeof(int8_t));
    float* x = (float*)malloc(cols * sizeof(float));
    float* bias = (float*)malloc(rows * sizeof(float));
    float* y = (float*)malloc(rows * sizeof(float));

    ternary_matrix_t* W_ternary = ternary_matrix_alloc(rows, cols);
    popcount_matrix_t* W_pop = popcount_matrix_alloc(rows, cols);

    generate_test_data(W_dense, rows, cols, x, cols, bias, rows);
    ternary_matrix_pack(W_ternary, W_dense);
    popcount_matrix_from_ternary(W_pop, W_ternary);

    float threshold = 0.0f;
    popcount_gemv(W_pop, (const uint64_t*)x, threshold, y, rows);
    if (!is_finite_vector(y, rows)) {
        printf("  ERROR: Invalid output detected\n");
        *avg_time_ms = 0.0;
        *gops = 0.0;
        goto cleanup;
    }

    for (int32_t i = 0; i < cfg->warmup_iters; i++) {
        popcount_gemv(W_pop, (const uint64_t*)x, threshold, y, rows);
    }

    double t_start = get_time_ms();
    for (int32_t i = 0; i < cfg->bench_iters; i++) {
        popcount_gemv(W_pop, (const uint64_t*)x, threshold, y, rows);
        volatile float sink = y[0];
        (void)sink;
    }
    double t_end = get_time_ms();

    double elapsed = t_end - t_start;
    *avg_time_ms = elapsed / cfg->bench_iters;
    *gops = (2.0 * rows * cols) / (*avg_time_ms * 1e6);

cleanup:
    free(W_dense);
    free(x);
    free(bias);
    free(y);
    ternary_matrix_free(W_ternary);
    popcount_matrix_free(W_pop);
}

static void bench_popcount_per_token(
    const bench_config_t* cfg,
    double* avg_time_ms, double* gops
) {
    int32_t rows = cfg->rows;
    int32_t cols = cfg->cols;

    int8_t* W_dense = (int8_t*)malloc(rows * cols * sizeof(int8_t));
    float* x = (float*)malloc(cols * sizeof(float));
    float* bias = (float*)malloc(rows * sizeof(float));
    float* y = (float*)malloc(rows * sizeof(float));
    uint64_t* x_bits = (uint64_t*)malloc((cols + 63) / 64 * sizeof(uint64_t));

    ternary_matrix_t* W_ternary = ternary_matrix_alloc(rows, cols);
    popcount_matrix_t* W_pop = popcount_matrix_alloc(rows, cols);

    generate_test_data(W_dense, rows, cols, x, cols, bias, rows);
    ternary_matrix_pack(W_ternary, W_dense);
    popcount_matrix_from_ternary(W_pop, W_ternary);

    float threshold = 0.0f;
    popcount_quantize_activations(x, cols, x_bits, &threshold);
    popcount_gemv(W_pop, x_bits, threshold, y, rows);
    if (!is_finite_vector(y, rows)) {
        printf("  ERROR: Invalid output detected\n");
        *avg_time_ms = 0.0;
        *gops = 0.0;
        goto cleanup;
    }

    for (int32_t i = 0; i < cfg->warmup_iters; i++) {
        popcount_quantize_activations(x, cols, x_bits, &threshold);
        popcount_gemv(W_pop, x_bits, threshold, y, rows);
    }

    double t_start = get_time_ms();
    for (int32_t i = 0; i < cfg->bench_iters; i++) {
        popcount_quantize_activations(x, cols, x_bits, &threshold);
        popcount_gemv(W_pop, x_bits, threshold, y, rows);
        volatile float sink = y[0];
        (void)sink;
    }
    double t_end = get_time_ms();

    double elapsed = t_end - t_start;
    *avg_time_ms = elapsed / cfg->bench_iters;
    *gops = (2.0 * rows * cols) / (*avg_time_ms * 1e6);

cleanup:
    free(W_dense);
    free(x);
    free(bias);
    free(y);
    free(x_bits);
    ternary_matrix_free(W_ternary);
    popcount_matrix_free(W_pop);
}

static void bench_popcount_full_e2e(
    const bench_config_t* cfg,
    double* avg_time_ms, double* gops
) {
    int32_t rows = cfg->rows;
    int32_t cols = cfg->cols;

    int8_t* W_dense = (int8_t*)malloc(rows * cols * sizeof(int8_t));
    float* x = (float*)malloc(cols * sizeof(float));
    float* bias = (float*)malloc(rows * sizeof(float));
    float* y = (float*)malloc(rows * sizeof(float));

    ternary_matrix_t* W_ternary = ternary_matrix_alloc(rows, cols);

    generate_test_data(W_dense, rows, cols, x, cols, bias, rows);
    ternary_matrix_pack(W_ternary, W_dense);

    for (int32_t i = 0; i < cfg->warmup_iters; i++) {
        popcount_matrix_t* W_pop = popcount_matrix_alloc(rows, cols);
        popcount_matrix_from_ternary(W_pop, W_ternary);

        uint64_t* x_bits = (uint64_t*)malloc((cols + 63) / 64 * sizeof(uint64_t));
        float threshold = 0.0f;
        popcount_quantize_activations(x, cols, x_bits, &threshold);
        popcount_gemv(W_pop, x_bits, threshold, y, rows);

        free(x_bits);
        popcount_matrix_free(W_pop);
    }

    popcount_matrix_t* W_pop = popcount_matrix_alloc(rows, cols);
    popcount_matrix_from_ternary(W_pop, W_ternary);

    uint64_t* x_bits = (uint64_t*)malloc((cols + 63) / 64 * sizeof(uint64_t));
    float threshold = 0.0f;
    popcount_quantize_activations(x, cols, x_bits, &threshold);
    popcount_gemv(W_pop, x_bits, threshold, y, rows);

    if (!is_finite_vector(y, rows)) {
        printf("  ERROR: Invalid output detected\n");
        *avg_time_ms = 0.0;
        *gops = 0.0;
        free(x_bits);
        popcount_matrix_free(W_pop);
        goto cleanup;
    }

    double t_start = get_time_ms();
    for (int32_t i = 0; i < cfg->bench_iters; i++) {
        popcount_quantize_activations(x, cols, x_bits, &threshold);
        popcount_gemv(W_pop, x_bits, threshold, y, rows);
        volatile float sink = y[0];
        (void)sink;
    }
    double t_end = get_time_ms();

    double elapsed = t_end - t_start;
    *avg_time_ms = elapsed / cfg->bench_iters;
    *gops = (2.0 * rows * cols) / (*avg_time_ms * 1e6);

    free(x_bits);
    popcount_matrix_free(W_pop);

cleanup:
    free(W_dense);
    free(x);
    free(bias);
    free(y);
    ternary_matrix_free(W_ternary);
}

int main(void) {
    srand(42);

    simd_detect_init();
    ternary_gemv_init();
    popcount_gemv_init();

    printf("=== Ternary GEMV Benchmark ===\n\n");
    printf("SIMD: ");
    simd_print_caps();
    printf("\n");

    bench_config_t configs[] = {
        {256, 256, 10, 1000, "256x256"},
        {1024, 1024, 10, 500, "1024x1024"},
        {4096, 4096, 10, 100, "4096x4096"},
    };

    int num_configs = sizeof(configs) / sizeof(configs[0]);

    double results[3][6];

    for (int c = 0; c < num_configs; c++) {
        const bench_config_t* cfg = &configs[c];
        int64_t ops = (int64_t)cfg->rows * cfg->cols;

        printf("--- %s (%lld ops/call) ---\n", cfg->name, (long long)ops);
        printf("  %-25s %10s %10s %10s\n", "Kernel", "Avg (us)", "GOPS", "Mem (KB)");

        double avg_time, gops;

        bench_ternary_scalar(cfg, &avg_time, &gops);
        results[c][0] = avg_time;
        printf("  %-25s %10.2f %10.2f %10.1f\n",
               "ternary_scalar", avg_time, gops,
               (cfg->rows * cfg->cols + 3) / 4.0 / 1024.0);

#ifdef ENGINE_AVX2
        bench_ternary_avx2(cfg, &avg_time, &gops);
        results[c][1] = avg_time;
        printf("  %-25s %10.2f %10.2f %10.1f\n",
               "ternary_avx2", avg_time, gops,
               (cfg->rows * cfg->cols + 3) / 4.0 / 1024.0);
#else
        results[c][1] = -1.0;
        printf("  %-25s %10s %10s %10s\n", "ternary_avx2", "N/A", "N/A", "N/A");
#endif

        bench_ternary_dispatch(cfg, &avg_time, &gops);
        results[c][2] = avg_time;
        printf("  %-25s %10.2f %10.2f %10.1f\n",
               "ternary_dispatch", avg_time, gops,
               (cfg->rows * cfg->cols + 3) / 4.0 / 1024.0);

        bench_popcount_compute(cfg, &avg_time, &gops);
        results[c][3] = avg_time;
        printf("  %-25s %10.2f %10.2f %10.1f\n",
               "popcount_compute", avg_time, gops,
               (int64_t)(cfg->rows * ((cfg->cols + 63) / 64) * 16 + cfg->rows * 4) / 1024.0);

        bench_popcount_per_token(cfg, &avg_time, &gops);
        results[c][4] = avg_time;
        printf("  %-25s %10.2f %10.2f %10.1f\n",
               "popcount_per_token", avg_time, gops,
               (int64_t)(cfg->rows * ((cfg->cols + 63) / 64) * 16 + cfg->rows * 4 + (cfg->cols + 63) / 64 * 8) / 1024.0);

        bench_popcount_full_e2e(cfg, &avg_time, &gops);
        results[c][5] = avg_time;
        printf("  %-25s %10.2f %10.2f %10.1f\n",
               "popcount_full_e2e", avg_time, gops,
               (int64_t)(cfg->rows * ((cfg->cols + 63) / 64) * 16 + cfg->rows * 4 + (cfg->cols + 63) / 64 * 8) / 1024.0);

        printf("\n");
    }

    printf("--- Speedup Summary ---\n");
    printf("  Size         Scalar/Scalar  AVX2/Scalar  Dispatch/Scalar  Popcount/AVX2  Popcount/Scalar\n");
    for (int c = 0; c < num_configs; c++) {
        double scalar = results[c][0];
        double avx2 = results[c][1];
        double dispatch = results[c][2];
        double popcount = results[c][3];

        printf("  %-12s %13.2fx %12s %16.2fx %14.2fx %15.2fx\n",
               configs[c].name,
               1.0,
               avx2 > 0 ? "2.0x" : "N/A",
               dispatch / scalar,
               avx2 > 0 ? popcount / avx2 : popcount / scalar,
               popcount / scalar);
    }

    return 0;
}
