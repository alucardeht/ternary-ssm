#include "inference.h"
#include "ternary_gemv.h"
#include "float_gemv.h"
#include "activations.h"
#include "simd_detect.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef ENGINE_VULKAN
#include "vulkan_backend.h"
#endif

static int engine_initialized = 0;

uint64_t rng_splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

float rng_float(uint64_t* state) {
    return (float)(rng_splitmix64(state) >> 11) / (float)(1ULL << 53);
}

void engine_init(void) {
    if (engine_initialized) return;
    simd_detect_init();
    ternary_gemv_init();
    float_gemv_init();
    act_init();
    ssm_scan_init();
    attn_init();
    engine_initialized = 1;
}

inference_config_t engine_default_config(void) {
    inference_config_t cfg;
    cfg.sample_mode = ENGINE_SAMPLE_GREEDY;
    cfg.temperature = 1.0f;
    cfg.top_k = 40;
    cfg.top_p = 0.9f;
    cfg.max_tokens = 256;
    cfg.seed = 42;
    return cfg;
}

inference_engine_t* engine_create(const trnm_model_t* model) {
    if (!model) return NULL;

    engine_init();

    inference_engine_t* engine = (inference_engine_t*)calloc(1, sizeof(inference_engine_t));
    if (!engine) return NULL;

    engine->model = model;

    int32_t d_model = model->header.d_model;
    int32_t vocab_size = model->header.vocab_size;
    int32_t n_layers = model->header.n_layers;

    engine->hidden = (float*)malloc(d_model * sizeof(float));
    engine->buf_a = (float*)malloc(d_model * sizeof(float));
    engine->buf_b = (float*)malloc(d_model * sizeof(float));
    engine->logits = (float*)malloc(vocab_size * sizeof(float));

    if (!engine->hidden || !engine->buf_a || !engine->buf_b || !engine->logits) {
        engine_free(engine);
        return NULL;
    }

    engine->ssm_states = (ssm_state_t**)calloc(n_layers, sizeof(ssm_state_t*));
    engine->kv_caches = (attn_kv_cache_t**)calloc(n_layers, sizeof(attn_kv_cache_t*));

    if (!engine->ssm_states || !engine->kv_caches) {
        engine_free(engine);
        return NULL;
    }

    size_t max_scratch = 0;

    for (int32_t i = 0; i < n_layers; i++) {
        if (model->layer_types[i] == TRNM_LAYER_SSM) {
            engine->ssm_states[i] = ssm_state_alloc(
                model->header.ssd_n_heads,
                model->header.d_state,
                model->header.d_inner / model->header.ssd_n_heads,
                model->header.d_conv
            );
            if (!engine->ssm_states[i]) {
                engine_free(engine);
                return NULL;
            }

            size_t scratch_sz = ssm_scratch_size(model->ssm_layers[i]);
            if (scratch_sz > max_scratch) max_scratch = scratch_sz;
        } else {
            engine->kv_caches[i] = attn_kv_cache_alloc(
                model->header.window_size,
                model->header.n_heads,
                model->header.head_dim
            );
            if (!engine->kv_caches[i]) {
                engine_free(engine);
                return NULL;
            }

            size_t scratch_sz = attn_scratch_size(model->attn_layers[i]);
            if (scratch_sz > max_scratch) max_scratch = scratch_sz;
        }
    }

    engine->layer_scratch_size = max_scratch;
    if (max_scratch > 0) {
        engine->layer_scratch = (float*)malloc(max_scratch);
        if (!engine->layer_scratch) {
            engine_free(engine);
            return NULL;
        }
    }

    engine->config = engine_default_config();
    engine->rng_state = engine->config.seed;
    engine->tokens_processed = 0;

#ifdef ENGINE_VULKAN
    engine->use_vulkan = false;
    if (engine_has_vulkan()) {
        vk_context_t* vk_ctx = vk_init();
        if (vk_ctx != NULL) {
            if (vk_upload_weights(vk_ctx, model) == 0) {
                engine->vk_ctx = vk_ctx;
                engine->use_vulkan = true;
                fprintf(stdout, "Vulkan GPU backend initialized successfully\n");
            } else {
                fprintf(stderr, "Failed to upload weights to Vulkan GPU, falling back to CPU\n");
                vk_shutdown(vk_ctx);
            }
        } else {
            fprintf(stderr, "Failed to initialize Vulkan GPU, falling back to CPU\n");
        }
    }
#endif

    return engine;
}

void engine_free(inference_engine_t* engine) {
    if (!engine) return;

#ifdef ENGINE_VULKAN
    if (engine->vk_ctx) {
        vk_shutdown(engine->vk_ctx);
        engine->vk_ctx = NULL;
    }
#endif

    if (engine->model && engine->ssm_states) {
        for (int32_t i = 0; i < engine->model->header.n_layers; i++) {
            if (engine->ssm_states[i]) {
                ssm_state_free(engine->ssm_states[i]);
            }
        }
    }

    if (engine->model && engine->kv_caches) {
        for (int32_t i = 0; i < engine->model->header.n_layers; i++) {
            if (engine->kv_caches[i]) {
                attn_kv_cache_free(engine->kv_caches[i]);
            }
        }
    }

    free(engine->ssm_states);
    free(engine->kv_caches);
    free(engine->hidden);
    free(engine->buf_a);
    free(engine->buf_b);
    free(engine->logits);
    free(engine->layer_scratch);
    free(engine);
}

void engine_reset(inference_engine_t* engine) {
    if (!engine) return;

    memset(engine->hidden, 0, engine->model->header.d_model * sizeof(float));
    memset(engine->logits, 0, engine->model->header.vocab_size * sizeof(float));

    for (int32_t i = 0; i < engine->model->header.n_layers; i++) {
        if (engine->ssm_states[i]) {
            ssm_state_reset(engine->ssm_states[i]);
        }
        if (engine->kv_caches[i]) {
            attn_kv_cache_reset(engine->kv_caches[i]);
        }
    }

    engine->tokens_processed = 0;
}

void engine_set_config(inference_engine_t* engine, const inference_config_t* config) {
    if (!engine || !config) return;
    engine->config = *config;
    engine->rng_state = engine->config.seed;
}

static const float* engine_forward_cpu(inference_engine_t* engine, int32_t token_id) {
    if (!engine || !engine->model) return NULL;

    int32_t vocab_size = engine->model->header.vocab_size;
    int32_t d_model = engine->model->header.d_model;
    int32_t n_layers = engine->model->header.n_layers;

    if (token_id < 0 || token_id >= vocab_size) return NULL;

    memcpy(engine->hidden, &engine->model->embedding[token_id * d_model], d_model * sizeof(float));

    for (int32_t i = 0; i < n_layers; i++) {
        if (engine->model->layer_types[i] == TRNM_LAYER_SSM) {
            act_rmsnorm(engine->buf_a, engine->hidden, engine->model->pre_norms[i],
                       d_model, 1e-5f);
            ssm_step(engine->buf_b, engine->ssm_states[i], engine->model->ssm_layers[i],
                    engine->buf_a, engine->layer_scratch);

            for (int32_t j = 0; j < d_model; j++) {
                engine->hidden[j] += engine->buf_b[j];
            }
        } else {
            attn_step(engine->buf_a, engine->kv_caches[i], engine->model->attn_layers[i],
                     engine->hidden, engine->layer_scratch);
            memcpy(engine->hidden, engine->buf_a, d_model * sizeof(float));
        }
    }

    act_rmsnorm(engine->buf_a, engine->hidden, engine->model->final_norm, d_model, 1e-5f);

    if (engine->model->header.flags & TRNM_FLAG_LM_TERNARY) {
        if (engine->model->lm_head_ternary) {
            ternary_gemv(engine->logits, engine->model->lm_head_ternary, engine->buf_a,
                        NULL, vocab_size, d_model);
        }
    } else {
        if (engine->model->header.flags & TRNM_FLAG_HAS_LM_HEAD) {
            float_gemv(engine->logits, engine->model->lm_head_float, engine->buf_a,
                      NULL, vocab_size, d_model);
        } else {
            float_gemv(engine->logits, engine->model->embedding, engine->buf_a,
                      NULL, vocab_size, d_model);
        }
    }

    engine->tokens_processed++;
    return engine->logits;
}

#ifdef ENGINE_VULKAN
static const float* engine_forward_vulkan(inference_engine_t* engine, int32_t token_id) {
    if (!engine || !engine->model || !engine->vk_ctx) return NULL;

    int32_t vocab_size = engine->model->header.vocab_size;
    int32_t d_model = engine->model->header.d_model;
    int32_t n_layers = engine->model->header.n_layers;

    if (token_id < 0 || token_id >= vocab_size) return NULL;

    memcpy(engine->hidden, &engine->model->embedding[token_id * d_model], d_model * sizeof(float));
    vk_upload_to_scratch(engine->vk_ctx, 0, engine->hidden, d_model * sizeof(float));

    for (int32_t i = 0; i < n_layers; i++) {
        if (engine->model->layer_types[i] == TRNM_LAYER_SSM) {
            vk_rmsnorm(engine->vk_ctx, d_model * sizeof(float), 0,
                      engine->model->pre_norms[i] - engine->model->weight_data,
                      d_model, 1e-5f);

            vk_submit_and_wait(engine->vk_ctx);
            vk_download_from_scratch(engine->vk_ctx, d_model * sizeof(float), engine->buf_a, d_model * sizeof(float));

            act_rmsnorm(engine->buf_a, engine->hidden, engine->model->pre_norms[i], d_model, 1e-5f);
            ssm_step(engine->buf_b, engine->ssm_states[i], engine->model->ssm_layers[i],
                    engine->buf_a, engine->layer_scratch);

            vk_upload_to_scratch(engine->vk_ctx, 0, engine->buf_b, d_model * sizeof(float));

            for (int32_t j = 0; j < d_model; j++) {
                engine->hidden[j] += engine->buf_b[j];
            }
        } else {
            vk_rmsnorm(engine->vk_ctx, d_model * sizeof(float), 0,
                      engine->model->pre_norms[i] - engine->model->weight_data,
                      d_model, 1e-5f);

            vk_submit_and_wait(engine->vk_ctx);
            vk_download_from_scratch(engine->vk_ctx, d_model * sizeof(float), engine->buf_a, d_model * sizeof(float));

            attn_step(engine->buf_a, engine->kv_caches[i], engine->model->attn_layers[i],
                     engine->hidden, engine->layer_scratch);
            memcpy(engine->hidden, engine->buf_a, d_model * sizeof(float));
        }
    }

    vk_rmsnorm(engine->vk_ctx, d_model * sizeof(float), 0,
              engine->model->final_norm - engine->model->weight_data,
              d_model, 1e-5f);

    vk_submit_and_wait(engine->vk_ctx);
    vk_download_from_scratch(engine->vk_ctx, d_model * sizeof(float), engine->buf_a, d_model * sizeof(float));

    if (engine->model->header.flags & TRNM_FLAG_LM_TERNARY) {
        if (engine->model->lm_head_ternary) {
            ternary_gemv(engine->logits, engine->model->lm_head_ternary, engine->buf_a,
                        NULL, vocab_size, d_model);
        }
    } else {
        if (engine->model->header.flags & TRNM_FLAG_HAS_LM_HEAD) {
            float_gemv(engine->logits, engine->model->lm_head_float, engine->buf_a,
                      NULL, vocab_size, d_model);
        } else {
            float_gemv(engine->logits, engine->model->embedding, engine->buf_a,
                      NULL, vocab_size, d_model);
        }
    }

    engine->tokens_processed++;
    return engine->logits;
}
#endif

const float* engine_forward(inference_engine_t* engine, int32_t token_id) {
#ifdef ENGINE_VULKAN
    if (engine && engine->use_vulkan && engine->vk_ctx) {
        return engine_forward_vulkan(engine, token_id);
    }
#endif
    return engine_forward_cpu(engine, token_id);
}

int32_t sample_greedy(const float* logits, int32_t vocab_size) {
    if (!logits || vocab_size <= 0) return 0;

    int32_t best = 0;
    float max_val = logits[0];

    for (int32_t i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            best = i;
        }
    }

    return best;
}

int32_t sample_temperature(float* logits, int32_t vocab_size, float temp, uint64_t* rng) {
    if (!logits || vocab_size <= 0 || temp <= 0.0f) return sample_greedy(logits, vocab_size);

    for (int32_t i = 0; i < vocab_size; i++) {
        logits[i] /= temp;
    }

    float max_logit = logits[0];
    for (int32_t i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum_exp = 0.0f;
    for (int32_t i = 0; i < vocab_size; i++) {
        logits[i] = expf(logits[i] - max_logit);
        sum_exp += logits[i];
    }

    float u = rng_float(rng) * sum_exp;
    float acc = 0.0f;

    for (int32_t i = 0; i < vocab_size; i++) {
        acc += logits[i];
        if (u <= acc) return i;
    }

    return vocab_size - 1;
}

int32_t sample_top_k(float* logits, int32_t vocab_size, int32_t k, float temp, uint64_t* rng) {
    if (!logits || vocab_size <= 0 || k <= 0) return sample_greedy(logits, vocab_size);

    if (k > vocab_size) k = vocab_size;

    for (int32_t i = 0; i < vocab_size; i++) {
        logits[i] /= temp;
    }

    float max_logit = logits[0];
    for (int32_t i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float threshold = -INFINITY;
    int count = 0;
    for (int32_t i = 0; i < vocab_size; i++) {
        if (logits[i] > threshold) count++;
    }

    if (count > k) {
        float* vals = (float*)malloc(vocab_size * sizeof(float));
        if (!vals) return 0;

        for (int32_t i = 0; i < vocab_size; i++) {
            vals[i] = logits[i];
        }

        for (int32_t i = 0; i < k; i++) {
            float max_val = -INFINITY;
            int32_t max_idx = -1;
            for (int32_t j = 0; j < vocab_size; j++) {
                if (vals[j] > max_val) {
                    max_val = vals[j];
                    max_idx = j;
                }
            }
            if (max_idx >= 0) vals[max_idx] = -INFINITY;
            if (i == k - 1) threshold = max_val;
        }

        free(vals);
    }

    float sum_exp = 0.0f;
    for (int32_t i = 0; i < vocab_size; i++) {
        if (logits[i] >= threshold) {
            logits[i] = expf(logits[i] - max_logit);
            sum_exp += logits[i];
        } else {
            logits[i] = 0.0f;
        }
    }

    if (sum_exp <= 0.0f) return sample_greedy(logits, vocab_size);

    float u = rng_float(rng) * sum_exp;
    float acc = 0.0f;

    for (int32_t i = 0; i < vocab_size; i++) {
        acc += logits[i];
        if (u <= acc) return i;
    }

    return vocab_size - 1;
}

int32_t sample_top_p(float* logits, int32_t vocab_size, float p, float temp, uint64_t* rng) {
    if (!logits || vocab_size <= 0 || p <= 0.0f || p > 1.0f) {
        return sample_greedy(logits, vocab_size);
    }

    for (int32_t i = 0; i < vocab_size; i++) {
        logits[i] /= temp;
    }

    float max_logit = logits[0];
    for (int32_t i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    for (int32_t i = 0; i < vocab_size; i++) {
        logits[i] = expf(logits[i] - max_logit);
    }

    float sum_exp = 0.0f;
    for (int32_t i = 0; i < vocab_size; i++) {
        sum_exp += logits[i];
    }

    for (int32_t i = 0; i < vocab_size; i++) {
        logits[i] /= sum_exp;
    }

    typedef struct { float prob; int32_t idx; } prob_pair;
    prob_pair* probs = (prob_pair*)malloc(vocab_size * sizeof(prob_pair));
    if (!probs) return 0;

    for (int32_t i = 0; i < vocab_size; i++) {
        probs[i].prob = logits[i];
        probs[i].idx = i;
    }

    for (int32_t i = 0; i < vocab_size - 1; i++) {
        for (int32_t j = i + 1; j < vocab_size; j++) {
            if (probs[j].prob > probs[i].prob) {
                prob_pair tmp = probs[i];
                probs[i] = probs[j];
                probs[j] = tmp;
            }
        }
    }

    float cumsum = 0.0f;
    int32_t cutoff = vocab_size;
    for (int32_t i = 0; i < vocab_size; i++) {
        cumsum += probs[i].prob;
        if (cumsum >= p) {
            cutoff = i + 1;
            break;
        }
    }

    float u = rng_float(rng);
    float acc = 0.0f;

    for (int32_t i = 0; i < cutoff; i++) {
        acc += probs[i].prob / cumsum;
        if (u <= acc) {
            int32_t result = probs[i].idx;
            free(probs);
            return result;
        }
    }

    int32_t result = probs[0].idx;
    free(probs);
    return result;
}

int32_t engine_step(inference_engine_t* engine, int32_t token_id) {
    if (!engine) return ENGINE_ERR_NULL;

    const float* logits = engine_forward(engine, token_id);
    if (!logits) return ENGINE_ERR_TOKEN;

    int32_t vocab_size = engine->model->header.vocab_size;
    float* logits_copy = (float*)malloc(vocab_size * sizeof(float));
    if (!logits_copy) return ENGINE_ERR_ALLOC;

    memcpy(logits_copy, logits, vocab_size * sizeof(float));

    int32_t next_token;
    switch (engine->config.sample_mode) {
        case ENGINE_SAMPLE_GREEDY:
            next_token = sample_greedy(logits_copy, vocab_size);
            break;
        case ENGINE_SAMPLE_TEMPERATURE:
            next_token = sample_temperature(logits_copy, vocab_size, engine->config.temperature, &engine->rng_state);
            break;
        case ENGINE_SAMPLE_TOP_K:
            next_token = sample_top_k(logits_copy, vocab_size, engine->config.top_k, engine->config.temperature, &engine->rng_state);
            break;
        case ENGINE_SAMPLE_TOP_P:
            next_token = sample_top_p(logits_copy, vocab_size, engine->config.top_p, engine->config.temperature, &engine->rng_state);
            break;
        default:
            next_token = sample_greedy(logits_copy, vocab_size);
    }

    free(logits_copy);
    return next_token;
}

int engine_prefill(inference_engine_t* engine, const int32_t* tokens, int32_t n_tokens) {
    if (!engine || !tokens || n_tokens <= 0) return ENGINE_ERR_NULL;

    engine_reset(engine);

    for (int32_t i = 0; i < n_tokens; i++) {
        const float* logits = engine_forward(engine, tokens[i]);
        if (!logits) return ENGINE_ERR_TOKEN;
    }

    return ENGINE_OK;
}

int32_t engine_generate(inference_engine_t* engine, const int32_t* prompt,
                       int32_t prompt_len, int32_t* tokens_out, int32_t max_tokens) {
    if (!engine || !prompt || !tokens_out || prompt_len < 0 || max_tokens <= 0) {
        return ENGINE_ERR_NULL;
    }

    int result = engine_prefill(engine, prompt, prompt_len);
    if (result != ENGINE_OK) return result;

    int32_t tokens_generated = 0;
    int32_t last_token = (prompt_len > 0) ? prompt[prompt_len - 1] : 0;

    while (tokens_generated < max_tokens) {
        int32_t next_token = engine_step(engine, last_token);

        if (next_token < 0) return next_token;

        tokens_out[tokens_generated] = next_token;
        tokens_generated++;
        last_token = next_token;
    }

    return tokens_generated;
}
