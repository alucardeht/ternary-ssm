#ifndef INFERENCE_H
#define INFERENCE_H

#include "model_io.h"
#include "ssm_scan.h"
#include "attention.h"
#include <stdint.h>

#ifdef ENGINE_VULKAN
#include "vulkan_backend.h"
#endif

#define ENGINE_OK           0
#define ENGINE_ERR_NULL    -1
#define ENGINE_ERR_ALLOC   -2
#define ENGINE_ERR_TOKEN   -3
#define ENGINE_ERR_MODEL   -4

#define ENGINE_SAMPLE_GREEDY      0
#define ENGINE_SAMPLE_TEMPERATURE 1
#define ENGINE_SAMPLE_TOP_K       2
#define ENGINE_SAMPLE_TOP_P       3

typedef struct {
    int32_t sample_mode;
    float temperature;
    int32_t top_k;
    float top_p;
    int32_t max_tokens;
    uint64_t seed;
} inference_config_t;

typedef struct {
    const trnm_model_t* model;
    ssm_state_t** ssm_states;
    attn_kv_cache_t** kv_caches;
    float* hidden;
    float* buf_a;
    float* buf_b;
    float* logits;
    float* layer_scratch;
    size_t layer_scratch_size;
    uint64_t rng_state;
    inference_config_t config;
    int32_t tokens_processed;

#ifdef ENGINE_VULKAN
    vk_context_t* vk_ctx;
    bool use_vulkan;
#endif
} inference_engine_t;

void engine_init(void);
inference_engine_t* engine_create(const trnm_model_t* model);
void engine_free(inference_engine_t* engine);
void engine_reset(inference_engine_t* engine);
void engine_set_config(inference_engine_t* engine, const inference_config_t* config);
inference_config_t engine_default_config(void);
int32_t engine_step(inference_engine_t* engine, int32_t token_id);
const float* engine_forward(inference_engine_t* engine, int32_t token_id);
int engine_prefill(inference_engine_t* engine, const int32_t* tokens, int32_t n_tokens);
int32_t engine_generate(inference_engine_t* engine, const int32_t* prompt,
                        int32_t prompt_len, int32_t* tokens_out, int32_t max_tokens);

int32_t sample_greedy(const float* logits, int32_t vocab_size);
int32_t sample_temperature(float* logits, int32_t vocab_size, float temp, uint64_t* rng);
int32_t sample_top_k(float* logits, int32_t vocab_size, int32_t k, float temp, uint64_t* rng);
int32_t sample_top_p(float* logits, int32_t vocab_size, float p, float temp, uint64_t* rng);

uint64_t rng_splitmix64(uint64_t* state);
float rng_float(uint64_t* state);

#endif
