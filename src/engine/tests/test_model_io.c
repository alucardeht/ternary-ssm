#include "model_io.h"
#include "ternary_types.h"
#include "ssm_scan.h"
#include "attention.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
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

int test_header_validation(void) {
    printf("test_header_validation...");
    tests_total++;

    trnm_header_t h = {0};

    memcpy(h.magic, "XXXX", 4);
    h.version = TRNM_VERSION;
    h.d_model = 32;
    h.n_layers = 2;
    h.vocab_size = 100;
    ASSERT(trnm_validate_header(&h) == TRNM_ERR_MAGIC);

    memcpy(h.magic, "TRNM", 4);
    h.version = 9999;
    ASSERT(trnm_validate_header(&h) == TRNM_ERR_VERSION);

    h.version = TRNM_VERSION;
    h.d_model = 0;
    ASSERT(trnm_validate_header(&h) == TRNM_ERR_INVALID);

    h.d_model = 32;
    h.n_layers = 0;
    ASSERT(trnm_validate_header(&h) == TRNM_ERR_INVALID);

    h.n_layers = 2;
    h.vocab_size = 0;
    ASSERT(trnm_validate_header(&h) == TRNM_ERR_INVALID);

    h.vocab_size = 100;
    ASSERT(trnm_validate_header(&h) == TRNM_OK);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_crc32(void) {
    printf("test_crc32...");
    tests_total++;

    uint32_t crc_empty = trnm_crc32("", 0);
    ASSERT(crc_empty == 0);

    uint32_t crc_hello = trnm_crc32("hello", 5);
    uint32_t crc_hello_again = trnm_crc32("hello", 5);
    ASSERT(crc_hello == crc_hello_again);
    ASSERT(crc_hello != 0);

    uint8_t zeros[64] = {0};
    uint32_t crc_zeros = trnm_crc32(zeros, 64);
    ASSERT(crc_zeros == trnm_crc32(zeros, 64));

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_round_trip(void) {
    printf("test_round_trip...");
    tests_total++;

    const char* test_file = "test_round_trip.trnm";

    trnm_model_t* model = create_test_model();
    ASSERT(model != NULL);

    int save_result = trnm_save(test_file, model);
    ASSERT(save_result == TRNM_OK);

    trnm_model_t* loaded = trnm_load(test_file);
    ASSERT(loaded != NULL);

    ASSERT(loaded->header.d_model == model->header.d_model);
    ASSERT(loaded->header.n_layers == model->header.n_layers);
    ASSERT(loaded->header.vocab_size == model->header.vocab_size);
    ASSERT(loaded->header.d_state == model->header.d_state);
    ASSERT(loaded->header.d_conv == model->header.d_conv);
    ASSERT(loaded->header.expand == model->header.expand);
    ASSERT(loaded->header.dt_rank == model->header.dt_rank);
    ASSERT(loaded->header.n_heads == model->header.n_heads);
    ASSERT(loaded->header.ssd_n_heads == model->header.ssd_n_heads);
    ASSERT(loaded->header.window_size == model->header.window_size);
    ASSERT(loaded->header.mlp_hidden == model->header.mlp_hidden);
    ASSERT(loaded->header.head_dim == model->header.head_dim);
    ASSERT(loaded->header.rope_dim == model->header.rope_dim);

    ASSERT(loaded->embedding != NULL);
    for (int i = 0; i < 10; i++) {
        ASSERT_APPROX(loaded->embedding[i], model->embedding[i], 1e-6f);
    }

    ASSERT(loaded->final_norm != NULL);
    for (int i = 0; i < model->header.d_model; i++) {
        ASSERT_APPROX(loaded->final_norm[i], model->final_norm[i], 1e-6f);
    }

    ASSERT(loaded->lm_head_ternary != NULL);
    ASSERT(loaded->lm_head_ternary->rows == model->lm_head_ternary->rows);
    ASSERT(loaded->lm_head_ternary->cols == model->lm_head_ternary->cols);

    trnm_free(loaded);
    cleanup_test_model(model);

    remove(test_file);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_header_read_only(void) {
    printf("test_header_read_only...");
    tests_total++;

    const char* test_file = "test_header_only.trnm";

    trnm_model_t* model = create_test_model();
    ASSERT(model != NULL);

    int save_result = trnm_save(test_file, model);
    ASSERT(save_result == TRNM_OK);

    trnm_header_t header = {0};
    int read_result = trnm_read_header(test_file, &header);
    ASSERT(read_result == TRNM_OK);

    ASSERT(header.d_model == model->header.d_model);
    ASSERT(header.n_layers == model->header.n_layers);
    ASSERT(header.vocab_size == model->header.vocab_size);
    ASSERT(header.mlp_hidden == model->header.mlp_hidden);

    cleanup_test_model(model);
    remove(test_file);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_corrupted_magic(void) {
    printf("test_corrupted_magic...");
    tests_total++;

    const char* test_file = "test_corrupted.trnm";

    trnm_model_t* model = create_test_model();
    ASSERT(model != NULL);

    int save_result = trnm_save(test_file, model);
    ASSERT(save_result == TRNM_OK);

    FILE* f = fopen(test_file, "r+b");
    if (f) {
        fseek(f, 0, SEEK_SET);
        uint8_t corrupted = 0xFF;
        fwrite(&corrupted, 1, 1, f);
        fclose(f);

        trnm_header_t header = {0};
        int read_result = trnm_read_header(test_file, &header);
        ASSERT(read_result != TRNM_OK);
    }

    cleanup_test_model(model);
    remove(test_file);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_alloc_free(void) {
    printf("test_alloc_free...");
    tests_total++;

    const char* test_file = "test_alloc_free.trnm";

    trnm_model_t* model = create_test_model();
    ASSERT(model != NULL);

    int save_result = trnm_save(test_file, model);
    ASSERT(save_result == TRNM_OK);

    trnm_model_t* loaded = trnm_load(test_file);
    ASSERT(loaded != NULL);

    trnm_free(loaded);

    trnm_free(NULL);

    cleanup_test_model(model);
    remove(test_file);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int main(void) {
    printf("=== Model I/O Tests ===\n\n");

    test_header_validation();
    test_crc32();
    test_round_trip();
    test_header_read_only();
    test_corrupted_magic();
    test_alloc_free();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
