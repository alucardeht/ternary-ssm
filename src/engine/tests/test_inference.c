#include "inference.h"
#include "model_io.h"
#include "ternary_types.h"
#include "ssm_scan.h"
#include "attention.h"
#include "simd_detect.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("  FAIL: %s\n", #cond); \
        return 0; \
    } \
} while(0)

#define ASSERT_APPROX(a, b, eps) do { \
    if (fabsf((a) - (b)) >= (eps)) { \
        printf("  FAIL: %f != %f (eps=%f)\n", (float)(a), (float)(b), (float)(eps)); \
        return 0; \
    } \
} while(0)

int tests_passed = 0;
int tests_total = 0;

static trnm_model_t* create_test_model(void) {
    trnm_model_t* model = (trnm_model_t*)calloc(1, sizeof(trnm_model_t));
    if (!model) return NULL;

    int32_t d_model = 32;
    int32_t n_layers = 2;
    int32_t vocab = 100;
    int32_t d_state = 4;
    int32_t d_conv = 4;
    int32_t expand = 2;
    int32_t dt_rank = 2;
    int32_t n_heads = 2;
    int32_t ssd_n_heads = 4;
    int32_t window_size = 16;
    int32_t mlp_hidden = (int32_t)(d_model * 2.67f);
    int32_t head_dim = d_model / n_heads;
    int32_t rope_dim = 8;
    int32_t d_inner = d_model * expand;

    model->header.magic[0] = 'T';
    model->header.magic[1] = 'R';
    model->header.magic[2] = 'N';
    model->header.magic[3] = 'M';
    model->header.version = TRNM_VERSION;
    model->header.flags = TRNM_FLAG_TERNARY | TRNM_FLAG_HAS_LM_HEAD | TRNM_FLAG_LM_TERNARY | TRNM_FLAG_SSD;
    model->header.n_layers = n_layers;
    model->header.vocab_size = vocab;
    model->header.d_model = d_model;
    model->header.d_state = d_state;
    model->header.d_conv = d_conv;
    model->header.expand = expand;
    model->header.dt_rank = dt_rank;
    model->header.n_heads = n_heads;
    model->header.ssd_n_heads = ssd_n_heads;
    model->header.window_size = window_size;
    model->header.max_seq_len = 2048;
    model->header.ternary_group_size = 32;
    model->header.n_swa_layers = 1;
    model->header.d_inner = d_inner;
    model->header.mlp_hidden = mlp_hidden;
    model->header.head_dim = head_dim;
    model->header.rope_dim = rope_dim;

    model->header.swa_indices[0] = 1;
    for (uint32_t i = 1; i < TRNM_MAX_SWA_LAYERS; i++) {
        model->header.swa_indices[i] = 0;
    }

    model->layer_table = (trnm_layer_entry_t*)calloc(n_layers, sizeof(trnm_layer_entry_t));
    model->layer_types = (uint8_t*)calloc(n_layers, sizeof(uint8_t));
    model->ssm_layers = (ssm_layer_t**)calloc(n_layers, sizeof(ssm_layer_t*));
    model->attn_layers = (attn_layer_t**)calloc(n_layers, sizeof(attn_layer_t*));
    model->pre_norms = (float**)calloc(n_layers, sizeof(float*));

    model->embedding = (float*)malloc(vocab * d_model * sizeof(float));
    for (int i = 0; i < vocab * d_model; i++) {
        model->embedding[i] = i * 0.01f;
    }

    model->layer_types[0] = TRNM_LAYER_SSM;
    model->layer_types[1] = TRNM_LAYER_SWA;

    model->ssm_layers[0] = ssm_layer_alloc(d_model, d_inner, d_state, d_conv, dt_rank, ssd_n_heads, 0);
    model->layer_table[0].layer_type = TRNM_LAYER_SSM;
    model->layer_table[0].layer_flags = 0;
    model->layer_table[0].data_size = 1024;
    model->layer_table[0].data_offset = 256;
    model->layer_table[0].data_crc32 = 0;

    model->attn_layers[1] = attn_layer_alloc(d_model, n_heads, window_size);
    model->layer_table[1].layer_type = TRNM_LAYER_SWA;
    model->layer_table[1].layer_flags = 0;
    model->layer_table[1].data_size = 2048;
    model->layer_table[1].data_offset = 1280;
    model->layer_table[1].data_crc32 = 0;

    model->pre_norms[0] = (float*)malloc(d_model * sizeof(float));
    model->pre_norms[1] = (float*)malloc(d_model * sizeof(float));
    for (int i = 0; i < d_model; i++) {
        model->pre_norms[0][i] = 1.0f;
        model->pre_norms[1][i] = 1.0f;
    }

    model->final_norm = (float*)malloc(d_model * sizeof(float));
    for (int i = 0; i < d_model; i++) {
        model->final_norm[i] = 1.0f;
    }

    model->lm_head_ternary = ternary_matrix_alloc(vocab, d_model);
    int8_t* lm_head_dense = (int8_t*)calloc(vocab * d_model, sizeof(int8_t));
    for (int i = 0; i < vocab * d_model; i++) {
        lm_head_dense[i] = (int8_t)((i % 3) - 1);
    }
    ternary_matrix_pack(model->lm_head_ternary, lm_head_dense);
    for (int i = 0; i < vocab; i++) {
        model->lm_head_ternary->scales[i] = 1.0f;
    }
    free(lm_head_dense);

    return model;
}

static void cleanup_test_model(trnm_model_t* model) {
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

int test_engine_create_free(void) {
    printf("test_engine_create_free...");
    tests_total++;

    trnm_model_t* model = create_test_model();
    ASSERT(model != NULL);

    inference_engine_t* engine = engine_create(model);
    ASSERT(engine != NULL);
    ASSERT(engine->model == model);
    ASSERT(engine->hidden != NULL);
    ASSERT(engine->logits != NULL);
    ASSERT(engine->ssm_states != NULL);
    ASSERT(engine->kv_caches != NULL);
    ASSERT(engine->tokens_processed == 0);

    ASSERT(engine->ssm_states[0] != NULL);
    ASSERT(engine->ssm_states[1] == NULL);

    ASSERT(engine->kv_caches[0] == NULL);
    ASSERT(engine->kv_caches[1] != NULL);

    engine_free(engine);
    engine_free(NULL);

    ASSERT(engine_create(NULL) == NULL);

    cleanup_test_model(model);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_engine_reset(void) {
    printf("test_engine_reset...");
    tests_total++;

    trnm_model_t* model = create_test_model();
    ASSERT(model != NULL);

    inference_engine_t* engine = engine_create(model);
    ASSERT(engine != NULL);

    engine_forward(engine, 1);
    ASSERT(engine->tokens_processed == 1);

    engine_reset(engine);
    ASSERT(engine->tokens_processed == 0);
    ASSERT(engine->hidden[0] == 0.0f);
    ASSERT(engine->logits[0] == 0.0f);

    engine_free(engine);
    cleanup_test_model(model);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_engine_forward(void) {
    printf("test_engine_forward...");
    tests_total++;

    trnm_model_t* model = create_test_model();
    ASSERT(model != NULL);

    inference_engine_t* engine = engine_create(model);
    ASSERT(engine != NULL);

    const float* logits = engine_forward(engine, 0);
    ASSERT(logits != NULL);
    ASSERT(engine->tokens_processed == 1);

    int finite_count = 0;
    for (int i = 0; i < 10; i++) {
        if (!isnan(logits[i]) && !isinf(logits[i])) {
            finite_count++;
        }
    }
    ASSERT(finite_count >= 10);

    int nonzero_count = 0;
    for (int i = 0; i < model->header.vocab_size; i++) {
        if (logits[i] != 0.0f) {
            nonzero_count++;
            break;
        }
    }
    ASSERT(nonzero_count > 0);

    engine_free(engine);
    cleanup_test_model(model);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_engine_step(void) {
    printf("test_engine_step...");
    tests_total++;

    trnm_model_t* model = create_test_model();
    ASSERT(model != NULL);

    inference_engine_t* engine = engine_create(model);
    ASSERT(engine != NULL);

    int32_t next = engine_step(engine, 0);
    ASSERT(next >= 0 && next < 100);
    ASSERT(engine->tokens_processed == 1);

    engine_free(engine);
    cleanup_test_model(model);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_engine_multi_step(void) {
    printf("test_engine_multi_step...");
    tests_total++;

    trnm_model_t* model = create_test_model();
    ASSERT(model != NULL);

    inference_engine_t* engine = engine_create(model);
    ASSERT(engine != NULL);

    for (int i = 0; i < 10; i++) {
        int32_t next = engine_step(engine, i % 100);
        ASSERT(next >= 0 && next < 100);
    }
    ASSERT(engine->tokens_processed == 10);

    engine_free(engine);
    cleanup_test_model(model);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_engine_greedy_deterministic(void) {
    printf("test_engine_greedy_deterministic...");
    tests_total++;

    trnm_model_t* model = create_test_model();
    ASSERT(model != NULL);

    inference_engine_t* engine1 = engine_create(model);
    ASSERT(engine1 != NULL);

    int32_t out1[3];
    out1[0] = engine_step(engine1, 5);
    out1[1] = engine_step(engine1, 10);
    out1[2] = engine_step(engine1, 15);

    inference_engine_t* engine2 = engine_create(model);
    ASSERT(engine2 != NULL);

    int32_t out2[3];
    out2[0] = engine_step(engine2, 5);
    out2[1] = engine_step(engine2, 10);
    out2[2] = engine_step(engine2, 15);

    ASSERT(out1[0] == out2[0]);
    ASSERT(out1[1] == out2[1]);
    ASSERT(out1[2] == out2[2]);

    engine_free(engine1);
    engine_free(engine2);
    cleanup_test_model(model);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_engine_reset_reproduces(void) {
    printf("test_engine_reset_reproduces...");
    tests_total++;

    trnm_model_t* model = create_test_model();
    ASSERT(model != NULL);

    inference_engine_t* engine = engine_create(model);
    ASSERT(engine != NULL);

    int32_t first[3];
    first[0] = engine_step(engine, 3);
    first[1] = engine_step(engine, 7);
    first[2] = engine_step(engine, 42);

    engine_reset(engine);
    engine->rng_state = engine->config.seed;

    int32_t second[3];
    second[0] = engine_step(engine, 3);
    second[1] = engine_step(engine, 7);
    second[2] = engine_step(engine, 42);

    ASSERT(first[0] == second[0]);
    ASSERT(first[1] == second[1]);
    ASSERT(first[2] == second[2]);

    engine_free(engine);
    cleanup_test_model(model);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_engine_prefill(void) {
    printf("test_engine_prefill...");
    tests_total++;

    trnm_model_t* model = create_test_model();
    ASSERT(model != NULL);

    inference_engine_t* engine = engine_create(model);
    ASSERT(engine != NULL);

    engine_forward(engine, 99);

    int32_t prompt[] = {1, 2, 3, 4, 5};
    int result = engine_prefill(engine, prompt, 5);
    ASSERT(result == ENGINE_OK);
    ASSERT(engine->tokens_processed == 5);

    int err_null = engine_prefill(engine, NULL, 5);
    ASSERT(err_null == ENGINE_ERR_NULL);

    int err_zero = engine_prefill(engine, prompt, 0);
    ASSERT(err_zero == ENGINE_ERR_NULL);

    engine_free(engine);
    cleanup_test_model(model);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_engine_generate(void) {
    printf("test_engine_generate...");
    tests_total++;

    trnm_model_t* model = create_test_model();
    ASSERT(model != NULL);

    inference_engine_t* engine = engine_create(model);
    ASSERT(engine != NULL);

    int32_t prompt[] = {1, 2, 3};
    int32_t output[5];
    int32_t n = engine_generate(engine, prompt, 3, output, 5);
    ASSERT(n == 5);

    for (int i = 0; i < 5; i++) {
        ASSERT(output[i] >= 0 && output[i] < 100);
    }

    engine_free(engine);
    cleanup_test_model(model);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_sample_greedy(void) {
    printf("test_sample_greedy...");
    tests_total++;

    float logits[10] = {0.1f, 0.5f, 0.3f, 0.9f, 0.2f, 0.4f, 0.8f, 0.6f, 0.7f, 0.0f};
    ASSERT(sample_greedy(logits, 10) == 3);

    float equal[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    ASSERT(sample_greedy(equal, 4) == 0);

    float single[1] = {5.0f};
    ASSERT(sample_greedy(single, 1) == 0);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_sample_temperature(void) {
    printf("test_sample_temperature...");
    tests_total++;

    float logits[10];
    for (int i = 0; i < 10; i++) {
        logits[i] = i * 0.5f;
    }

    float logits_copy1[10];
    memcpy(logits_copy1, logits, sizeof(logits));
    uint64_t rng1 = 42;
    int32_t result1 = sample_temperature(logits_copy1, 10, 1.0f, &rng1);
    ASSERT(result1 >= 0 && result1 < 10);

    float logits_copy2[10];
    memcpy(logits_copy2, logits, sizeof(logits));
    uint64_t rng2 = 42;
    int32_t result2 = sample_temperature(logits_copy2, 10, 1.0f, &rng2);
    ASSERT(result1 == result2);

    float logits_greedy[10];
    for (int i = 0; i < 10; i++) {
        logits_greedy[i] = (i == 5) ? 10.0f : 0.0f;
    }
    uint64_t rng3 = 42;
    int32_t result3 = sample_temperature(logits_greedy, 10, 0.0f, &rng3);
    ASSERT(result3 == 5);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_engine_invalid_token(void) {
    printf("test_engine_invalid_token...");
    tests_total++;

    trnm_model_t* model = create_test_model();
    ASSERT(model != NULL);

    inference_engine_t* engine = engine_create(model);
    ASSERT(engine != NULL);

    ASSERT(engine_forward(engine, -1) == NULL);
    ASSERT(engine_forward(engine, 100) == NULL);
    ASSERT(engine_forward(engine, 999) == NULL);
    ASSERT(engine->tokens_processed == 0);

    ASSERT(engine_step(engine, -1) == ENGINE_ERR_TOKEN);
    ASSERT(engine_step(engine, 100) == ENGINE_ERR_TOKEN);

    const float* logits = engine_forward(engine, 0);
    ASSERT(logits != NULL);
    ASSERT(engine->tokens_processed == 1);

    engine_free(engine);
    cleanup_test_model(model);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int main(void) {
    printf("=== Inference Engine Tests ===\n\n");

    test_engine_create_free();
    test_engine_reset();
    test_engine_forward();
    test_engine_step();
    test_engine_multi_step();
    test_engine_greedy_deterministic();
    test_engine_reset_reproduces();
    test_engine_prefill();
    test_engine_generate();
    test_sample_greedy();
    test_sample_temperature();
    test_engine_invalid_token();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
