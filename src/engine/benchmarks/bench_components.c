#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "ternary_types.h"
#include "ternary_gemv.h"
#include "float_gemv.h"
#include "activations.h"
#include "ssm_scan.h"
#include "attention.h"
#include "model_io.h"
#include "inference.h"
#include "simd_detect.h"

#define BENCH_D_MODEL 256
#define BENCH_N_LAYER 4
#define BENCH_VOCAB 1000
#define BENCH_D_STATE 16
#define BENCH_EXPAND 2
#define BENCH_D_INNER (BENCH_D_MODEL * BENCH_EXPAND)
#define BENCH_N_HEADS 4
#define BENCH_HEAD_DIM (BENCH_D_MODEL / BENCH_N_HEADS)
#define BENCH_SSD_N_HEADS 8
#define BENCH_SSM_HEAD_DIM (BENCH_D_INNER / BENCH_SSD_N_HEADS)
#define BENCH_WINDOW_SIZE 128
#define BENCH_D_CONV 4
#define BENCH_DT_RANK 16
#define BENCH_MLP_HIDDEN ((int32_t)(BENCH_D_MODEL * 2.67f))
#define WARMUP_ITERS 100
#define BENCH_ITERS 1000

typedef struct {
    const char* name;
    double avg_us;
    double min_us;
    double max_us;
    double est_tok_s;
} bench_result_t;

#ifdef _WIN32
static double get_time_us(void) {
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER count;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / freq.QuadPart * 1000000.0;
}
#else
static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000.0 + ts.tv_nsec / 1000.0;
}
#endif

static void fill_random_float(float* arr, int32_t n) {
    for (int32_t i = 0; i < n; i++) {
        arr[i] = (float)(rand() % 2000) / 1000.0f - 1.0f;
    }
}

static void fill_random_ternary_matrix(ternary_matrix_t* mat) {
    int8_t* dense = (int8_t*)malloc(mat->rows * mat->cols * sizeof(int8_t));
    for (int64_t i = 0; i < (int64_t)mat->rows * mat->cols; i++) {
        int val = rand() % 3;
        dense[i] = (int8_t)(val - 1);
    }
    ternary_matrix_pack(mat, dense);
    for (int32_t i = 0; i < mat->rows; i++) {
        mat->scales[i] = 1.0f;
    }
    free(dense);
}

static trnm_model_t* create_bench_model(void) {
    trnm_model_t* model = (trnm_model_t*)calloc(1, sizeof(trnm_model_t));
    if (!model) return NULL;

    model->header.magic[0] = 'T';
    model->header.magic[1] = 'R';
    model->header.magic[2] = 'N';
    model->header.magic[3] = 'M';
    model->header.version = TRNM_VERSION;
    model->header.flags = TRNM_FLAG_TERNARY | TRNM_FLAG_HAS_LM_HEAD | TRNM_FLAG_LM_TERNARY | TRNM_FLAG_SSD;
    model->header.n_layers = BENCH_N_LAYER;
    model->header.vocab_size = BENCH_VOCAB;
    model->header.d_model = BENCH_D_MODEL;
    model->header.d_state = BENCH_D_STATE;
    model->header.d_conv = BENCH_D_CONV;
    model->header.expand = BENCH_EXPAND;
    model->header.dt_rank = BENCH_DT_RANK;
    model->header.n_heads = BENCH_N_HEADS;
    model->header.ssd_n_heads = BENCH_SSD_N_HEADS;
    model->header.window_size = BENCH_WINDOW_SIZE;
    model->header.max_seq_len = 2048;
    model->header.ternary_group_size = 128;
    model->header.n_swa_layers = 1;
    model->header.d_inner = BENCH_D_INNER;
    model->header.mlp_hidden = BENCH_MLP_HIDDEN;
    model->header.head_dim = BENCH_HEAD_DIM;
    model->header.rope_dim = BENCH_HEAD_DIM;

    model->header.swa_indices[0] = 3;
    for (uint32_t i = 1; i < TRNM_MAX_SWA_LAYERS; i++) {
        model->header.swa_indices[i] = 0;
    }

    model->layer_table = (trnm_layer_entry_t*)calloc(BENCH_N_LAYER, sizeof(trnm_layer_entry_t));
    model->layer_types = (uint8_t*)calloc(BENCH_N_LAYER, sizeof(uint8_t));
    model->ssm_layers = (ssm_layer_t**)calloc(BENCH_N_LAYER, sizeof(ssm_layer_t*));
    model->attn_layers = (attn_layer_t**)calloc(BENCH_N_LAYER, sizeof(attn_layer_t*));
    model->pre_norms = (float**)calloc(BENCH_N_LAYER, sizeof(float*));

    model->embedding = (float*)malloc(BENCH_VOCAB * BENCH_D_MODEL * sizeof(float));
    fill_random_float(model->embedding, BENCH_VOCAB * BENCH_D_MODEL);

    for (int32_t i = 0; i < BENCH_N_LAYER; i++) {
        if (i == 3) {
            model->layer_types[i] = TRNM_LAYER_SWA;
            model->attn_layers[i] = attn_layer_alloc(BENCH_D_MODEL, BENCH_N_HEADS, BENCH_WINDOW_SIZE);
            model->layer_table[i].layer_type = TRNM_LAYER_SWA;
        } else {
            model->layer_types[i] = TRNM_LAYER_SSM;
            model->ssm_layers[i] = ssm_layer_alloc(BENCH_D_MODEL, BENCH_D_INNER, BENCH_D_STATE,
                                                    BENCH_D_CONV, BENCH_DT_RANK, BENCH_SSD_N_HEADS, 0);
            model->layer_table[i].layer_type = TRNM_LAYER_SSM;
        }
        model->layer_table[i].layer_flags = 0;
        model->layer_table[i].data_size = 1024;
        model->layer_table[i].data_offset = 256 + i * 1024;
        model->layer_table[i].data_crc32 = 0;

        model->pre_norms[i] = (float*)malloc(BENCH_D_MODEL * sizeof(float));
        for (int32_t j = 0; j < BENCH_D_MODEL; j++) {
            model->pre_norms[i][j] = 1.0f;
        }
    }

    model->final_norm = (float*)malloc(BENCH_D_MODEL * sizeof(float));
    for (int32_t i = 0; i < BENCH_D_MODEL; i++) {
        model->final_norm[i] = 1.0f;
    }

    model->lm_head_ternary = ternary_matrix_alloc(BENCH_VOCAB, BENCH_D_MODEL);
    fill_random_ternary_matrix(model->lm_head_ternary);

    return model;
}

static void cleanup_bench_model(trnm_model_t* model) {
    if (!model) return;

    if (model->layer_table) free(model->layer_table);
    if (model->layer_types) free(model->layer_types);
    if (model->embedding) free(model->embedding);

    if (model->ssm_layers) {
        for (uint32_t i = 0; i < model->header.n_layers; i++) {
            if (model->ssm_layers[i]) ssm_layer_free(model->ssm_layers[i]);
        }
        free(model->ssm_layers);
    }

    if (model->attn_layers) {
        for (uint32_t i = 0; i < model->header.n_layers; i++) {
            if (model->attn_layers[i]) attn_layer_free(model->attn_layers[i]);
        }
        free(model->attn_layers);
    }

    if (model->pre_norms) {
        for (uint32_t i = 0; i < model->header.n_layers; i++) {
            if (model->pre_norms[i]) free(model->pre_norms[i]);
        }
        free(model->pre_norms);
    }

    if (model->final_norm) free(model->final_norm);
    if (model->lm_head_ternary) ternary_matrix_free(model->lm_head_ternary);
    if (model->lm_head_float) free(model->lm_head_float);

    free(model);
}

static bench_result_t bench_ternary_gemv(void) {
    bench_result_t result = {0};
    result.name = "Ternary GEMV 256->512";

    int32_t rows = 512, cols = 256;
    float* x = (float*)malloc(cols * sizeof(float));
    float* y = (float*)malloc(rows * sizeof(float));
    float* bias = (float*)malloc(rows * sizeof(float));
    ternary_matrix_t* W = ternary_matrix_alloc(rows, cols);

    fill_random_float(x, cols);
    fill_random_float(bias, rows);
    fill_random_ternary_matrix(W);

    for (int32_t i = 0; i < WARMUP_ITERS; i++) {
        ternary_gemv(y, W, x, bias, rows, cols);
    }

    double t_start = get_time_us();
    double min_time = 1e9, max_time = 0;
    for (int32_t i = 0; i < BENCH_ITERS; i++) {
        double iter_start = get_time_us();
        ternary_gemv(y, W, x, bias, rows, cols);
        double iter_end = get_time_us();
        double iter_time = iter_end - iter_start;
        if (iter_time < min_time) min_time = iter_time;
        if (iter_time > max_time) max_time = iter_time;
        volatile float sink = y[0];
        (void)sink;
    }
    double t_end = get_time_us();

    result.avg_us = (t_end - t_start) / BENCH_ITERS;
    result.min_us = min_time;
    result.max_us = max_time;
    result.est_tok_s = 1e6 / result.avg_us;

    free(x);
    free(y);
    free(bias);
    ternary_matrix_free(W);

    return result;
}

static bench_result_t bench_rmsnorm(void) {
    bench_result_t result = {0};
    result.name = "RMSNorm (256)";

    float* x = (float*)malloc(BENCH_D_MODEL * sizeof(float));
    float* y = (float*)malloc(BENCH_D_MODEL * sizeof(float));
    float* weight = (float*)malloc(BENCH_D_MODEL * sizeof(float));

    fill_random_float(x, BENCH_D_MODEL);
    fill_random_float(weight, BENCH_D_MODEL);

    for (int32_t i = 0; i < WARMUP_ITERS; i++) {
        act_rmsnorm(y, x, weight, BENCH_D_MODEL, 1e-5f);
    }

    double t_start = get_time_us();
    double min_time = 1e9, max_time = 0;
    for (int32_t i = 0; i < BENCH_ITERS; i++) {
        double iter_start = get_time_us();
        act_rmsnorm(y, x, weight, BENCH_D_MODEL, 1e-5f);
        double iter_end = get_time_us();
        double iter_time = iter_end - iter_start;
        if (iter_time < min_time) min_time = iter_time;
        if (iter_time > max_time) max_time = iter_time;
        volatile float sink = y[0];
        (void)sink;
    }
    double t_end = get_time_us();

    result.avg_us = (t_end - t_start) / BENCH_ITERS;
    result.min_us = min_time;
    result.max_us = max_time;
    result.est_tok_s = 1e6 / result.avg_us;

    free(x);
    free(y);
    free(weight);

    return result;
}

static bench_result_t bench_silu(void) {
    bench_result_t result = {0};
    result.name = "SiLU (512)";

    float* x = (float*)malloc(BENCH_D_INNER * sizeof(float));
    float* y = (float*)malloc(BENCH_D_INNER * sizeof(float));

    fill_random_float(x, BENCH_D_INNER);

    for (int32_t i = 0; i < WARMUP_ITERS; i++) {
        act_silu(y, x, BENCH_D_INNER);
    }

    double t_start = get_time_us();
    double min_time = 1e9, max_time = 0;
    for (int32_t i = 0; i < BENCH_ITERS; i++) {
        double iter_start = get_time_us();
        act_silu(y, x, BENCH_D_INNER);
        double iter_end = get_time_us();
        double iter_time = iter_end - iter_start;
        if (iter_time < min_time) min_time = iter_time;
        if (iter_time > max_time) max_time = iter_time;
        volatile float sink = y[0];
        (void)sink;
    }
    double t_end = get_time_us();

    result.avg_us = (t_end - t_start) / BENCH_ITERS;
    result.min_us = min_time;
    result.max_us = max_time;
    result.est_tok_s = 1e6 / result.avg_us;

    free(x);
    free(y);

    return result;
}

static bench_result_t bench_ssm_step(trnm_model_t* model) {
    bench_result_t result = {0};
    result.name = "SSM Step (SSD)";

    ssm_state_t* state = ssm_state_alloc(BENCH_SSD_N_HEADS, BENCH_D_STATE, BENCH_D_INNER / BENCH_SSD_N_HEADS, BENCH_D_CONV);
    ssm_layer_t* layer = model->ssm_layers[0];
    size_t scratch_size = ssm_scratch_size(layer);
    float* scratch = (float*)malloc(scratch_size);
    float* input = (float*)malloc(BENCH_D_MODEL * sizeof(float));
    float* output = (float*)malloc(BENCH_D_MODEL * sizeof(float));

    fill_random_float(input, BENCH_D_MODEL);

    for (int32_t i = 0; i < WARMUP_ITERS; i++) {
        ssm_step(output, state, layer, input, scratch);
    }

    double t_start = get_time_us();
    double min_time = 1e9, max_time = 0;
    for (int32_t i = 0; i < BENCH_ITERS; i++) {
        double iter_start = get_time_us();
        ssm_step(output, state, layer, input, scratch);
        double iter_end = get_time_us();
        double iter_time = iter_end - iter_start;
        if (iter_time < min_time) min_time = iter_time;
        if (iter_time > max_time) max_time = iter_time;
        volatile float sink = output[0];
        (void)sink;
    }
    double t_end = get_time_us();

    result.avg_us = (t_end - t_start) / BENCH_ITERS;
    result.min_us = min_time;
    result.max_us = max_time;
    result.est_tok_s = 1e6 / result.avg_us;

    free(scratch);
    free(input);
    free(output);
    ssm_state_free(state);

    return result;
}

static bench_result_t bench_attn_step(trnm_model_t* model) {
    bench_result_t result = {0};
    result.name = "Attention Step (SWA)";

    attn_layer_t* layer = model->attn_layers[3];
    attn_kv_cache_t* cache = attn_kv_cache_alloc(BENCH_WINDOW_SIZE, BENCH_N_HEADS, BENCH_HEAD_DIM);
    size_t scratch_size = attn_scratch_size(layer);
    float* scratch = (float*)malloc(scratch_size);
    float* input = (float*)malloc(BENCH_D_MODEL * sizeof(float));
    float* output = (float*)malloc(BENCH_D_MODEL * sizeof(float));

    fill_random_float(input, BENCH_D_MODEL);

    for (int32_t i = 0; i < 64; i++) {
        attn_step(output, cache, layer, input, scratch);
    }

    for (int32_t i = 0; i < WARMUP_ITERS; i++) {
        attn_step(output, cache, layer, input, scratch);
    }

    double t_start = get_time_us();
    double min_time = 1e9, max_time = 0;
    for (int32_t i = 0; i < BENCH_ITERS; i++) {
        double iter_start = get_time_us();
        attn_step(output, cache, layer, input, scratch);
        double iter_end = get_time_us();
        double iter_time = iter_end - iter_start;
        if (iter_time < min_time) min_time = iter_time;
        if (iter_time > max_time) max_time = iter_time;
        volatile float sink = output[0];
        (void)sink;
    }
    double t_end = get_time_us();

    result.avg_us = (t_end - t_start) / BENCH_ITERS;
    result.min_us = min_time;
    result.max_us = max_time;
    result.est_tok_s = 1e6 / result.avg_us;

    free(scratch);
    free(input);
    free(output);
    attn_kv_cache_free(cache);

    return result;
}

static bench_result_t bench_engine_forward(trnm_model_t* model) {
    bench_result_t result = {0};
    result.name = "Engine Forward (E2E)";

    engine_init();
    inference_engine_t* engine = engine_create(model);
    if (!engine) {
        result.avg_us = 0;
        result.min_us = 0;
        result.max_us = 0;
        result.est_tok_s = 0;
        return result;
    }

    for (int32_t i = 0; i < 32; i++) {
        engine_forward(engine, 42);
    }

    for (int32_t i = 0; i < WARMUP_ITERS; i++) {
        engine_forward(engine, 42);
    }

    double t_start = get_time_us();
    double min_time = 1e9, max_time = 0;
    for (int32_t i = 0; i < BENCH_ITERS; i++) {
        double iter_start = get_time_us();
        engine_forward(engine, 42);
        double iter_end = get_time_us();
        double iter_time = iter_end - iter_start;
        if (iter_time < min_time) min_time = iter_time;
        if (iter_time > max_time) max_time = iter_time;
    }
    double t_end = get_time_us();

    result.avg_us = (t_end - t_start) / BENCH_ITERS;
    result.min_us = min_time;
    result.max_us = max_time;
    result.est_tok_s = 1e6 / result.avg_us;

    engine_free(engine);

    return result;
}

int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    srand(42);

    simd_detect_init();
    ternary_gemv_init();
    act_init();
    ssm_scan_init();
    attn_init();
    engine_init();

    printf("=== Component Benchmarks (C Engine) ===\n\n");
    printf("SIMD: ");
    simd_print_caps();
    printf("\n");

    printf("Config: d_model=%d, n_layer=%d, vocab=%d, d_state=%d\n",
           BENCH_D_MODEL, BENCH_N_LAYER, BENCH_VOCAB, BENCH_D_STATE);
    printf("        expand=%d, n_heads=%d, window=%d, ssd_n_heads=%d\n",
           BENCH_EXPAND, BENCH_N_HEADS, BENCH_WINDOW_SIZE, BENCH_SSD_N_HEADS);
    printf("\n");

    printf("  %-30s %12s %12s %12s %12s\n", "Component", "Avg (us)", "Min (us)", "Max (us)", "Est tok/s");
    printf("  %s\n", "------- -------- -------- -------- --------");

    bench_result_t results[6];

    results[0] = bench_ternary_gemv();
    printf("  %-30s %12.2f %12.2f %12.2f %12.0f\n",
           results[0].name, results[0].avg_us, results[0].min_us, results[0].max_us, results[0].est_tok_s);

    results[1] = bench_rmsnorm();
    printf("  %-30s %12.2f %12.2f %12.2f %12.0f\n",
           results[1].name, results[1].avg_us, results[1].min_us, results[1].max_us, results[1].est_tok_s);

    results[2] = bench_silu();
    printf("  %-30s %12.2f %12.2f %12.2f %12.0f\n",
           results[2].name, results[2].avg_us, results[2].min_us, results[2].max_us, results[2].est_tok_s);

    trnm_model_t* model = create_bench_model();
    if (!model) {
        printf("ERROR: Failed to create benchmark model\n");
        return 1;
    }

    results[3] = bench_ssm_step(model);
    printf("  %-30s %12.2f %12.2f %12.2f %12.0f\n",
           results[3].name, results[3].avg_us, results[3].min_us, results[3].max_us, results[3].est_tok_s);

    results[4] = bench_attn_step(model);
    printf("  %-30s %12.2f %12.2f %12.2f %12.0f\n",
           results[4].name, results[4].avg_us, results[4].min_us, results[4].max_us, results[4].est_tok_s);

    results[5] = bench_engine_forward(model);
    printf("  %-30s %12.2f %12.2f %12.2f %12.0f\n",
           results[5].name, results[5].avg_us, results[5].min_us, results[5].max_us, results[5].est_tok_s);

    printf("\n");
    printf("--- Summary ---\n");
    double total_us = results[3].avg_us + results[4].avg_us;
    if (total_us > 0) {
        printf("SSM contribution:  %.1f%%\n", 100.0 * results[3].avg_us / total_us);
        printf("Attn contribution: %.1f%%\n", 100.0 * results[4].avg_us / total_us);
    }
    printf("E2E forward time:  %.2f us (%.0f tok/s)\n", results[5].avg_us, results[5].est_tok_s);

    cleanup_bench_model(model);

    return 0;
}
