#include "teaching.h"

#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEACHING_WARMUP_STEPS     8u
#define TEACHING_REPLAY_NUMER     1u
#define TEACHING_REPLAY_DENOM     4u
#define TEACHING_EVIDENCE_MIN     2u
#define TEACHING_KEY_PREFIX       16

enum {
    TEACHING_BUCKET_LANGUAGE = 0,
    TEACHING_BUCKET_NUMERIC  = 1,
    TEACHING_BUCKET_PUNCT    = 2,
    TEACHING_BUCKET_CODE     = 3,
    TEACHING_BUCKET_NEWLINE  = 4,
    TEACHING_BUCKET_MIXED    = 5
};

typedef struct {
    uint32_t key;
    float loss_ema;
    float last_loss;
    float best_loss;
    float gain_ema;
    uint32_t seen_count;
    uint32_t replay_count;
    uint32_t repair_count;
    uint32_t evidence_count;
    uint32_t deferral_count;
    uint32_t conflict_count;
    uint8_t bucket;
    uint8_t pending_evidence;
    uint8_t active;
} TeachingEpisode;

struct TeachingMemory {
    int seq_len;
    int capacity;
    int size;
    int cursor;
    uint16_t *episode_tokens;
    TeachingEpisode *episodes;
    float bucket_loss_ema[TEACHING_BUCKET_COUNT];
    float bucket_deficit[TEACHING_BUCKET_COUNT];
    float global_loss_ema;
    uint32_t fresh_events;
    uint32_t replay_events;
    uint32_t deferred_events;
    uint32_t repair_events;
    uint32_t conflict_events;
    uint64_t rng[4];
    char **vocab;
    int vocab_size;
};

static uint64_t rotl64(uint64_t x, unsigned int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t rng_next(uint64_t s[4]) {
    uint64_t result = rotl64(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl64(s[3], 45);

    return result;
}

static void rng_seed(uint64_t s[4], uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        s[i] = z ^ (z >> 31);
    }
}

static uint32_t fnv1a_u32_tokens(const uint16_t *tokens, int n) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < n; i++) {
        h ^= (uint32_t)tokens[i];
        h *= 16777619u;
    }
    return h ? h : 1u;
}

static int teaching_is_code_symbol(char c) {
    switch (c) {
        case '{': case '}': case '[': case ']':
        case '(': case ')': case ';': case ':':
        case '=': case '<': case '>': case '/':
        case '\\': case '_': case '+': case '-':
        case '*': case '%': case '&': case '|':
        case '^': case '~': case '`':
            return 1;
        default:
            return 0;
    }
}

static uint8_t teaching_classify_token(const TeachingMemory *tm, uint16_t token) {
    if (!tm || !tm->vocab || token >= (uint16_t)tm->vocab_size || tm->vocab[token] == NULL) {
        return TEACHING_BUCKET_MIXED;
    }

    const char *s = tm->vocab[token];
    if (s[0] == '\0') return TEACHING_BUCKET_NEWLINE;

    int alpha = 0;
    int digit = 0;
    int punct = 0;
    int code = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (isalpha(*p)) alpha++;
        else if (isdigit(*p)) digit++;
        else if (isspace(*p)) punct++;
        else if (teaching_is_code_symbol((char)*p)) code++;
        else punct++;
    }

    if (code > 0 && alpha > 0) return TEACHING_BUCKET_CODE;
    if (digit > 0 && alpha == 0 && punct == 0) return TEACHING_BUCKET_NUMERIC;
    if (alpha > 0 && digit == 0 && punct == 0) return TEACHING_BUCKET_LANGUAGE;
    if (punct > 0 && alpha == 0 && digit == 0) return TEACHING_BUCKET_PUNCT;
    if (code > 0) return TEACHING_BUCKET_CODE;
    if (digit > 0 && alpha > 0) return TEACHING_BUCKET_MIXED;
    return TEACHING_BUCKET_LANGUAGE;
}

static uint8_t teaching_classify_window(const TeachingMemory *tm, const uint16_t *tokens, int seq_len) {
    int counts[TEACHING_BUCKET_COUNT] = {0};
    for (int i = 0; i < seq_len; i++) {
        counts[teaching_classify_token(tm, tokens[i])]++;
    }

    int best_bucket = TEACHING_BUCKET_LANGUAGE;
    int best_count = counts[best_bucket];
    for (int b = 1; b < TEACHING_BUCKET_COUNT; b++) {
        if (counts[b] > best_count) {
            best_bucket = b;
            best_count = counts[b];
        }
    }
    return (uint8_t)best_bucket;
}

static const char *teaching_bucket_name(uint8_t bucket) {
    switch (bucket) {
        case TEACHING_BUCKET_LANGUAGE: return "language";
        case TEACHING_BUCKET_NUMERIC:  return "numeric";
        case TEACHING_BUCKET_PUNCT:    return "punct";
        case TEACHING_BUCKET_CODE:     return "code";
        case TEACHING_BUCKET_NEWLINE:  return "newline";
        case TEACHING_BUCKET_MIXED:    return "mixed";
        default:                       return "unknown";
    }
}

static float teaching_bucket_priority(const TeachingMemory *tm, uint8_t bucket) {
    float priority = tm->bucket_deficit[bucket];
    switch (bucket) {
        case TEACHING_BUCKET_CODE:
            priority += 0.35f * (tm->bucket_deficit[TEACHING_BUCKET_LANGUAGE]
                              +  tm->bucket_deficit[TEACHING_BUCKET_PUNCT]);
            break;
        case TEACHING_BUCKET_MIXED:
            priority += 0.35f * (tm->bucket_deficit[TEACHING_BUCKET_LANGUAGE]
                              +  tm->bucket_deficit[TEACHING_BUCKET_NUMERIC]);
            break;
        case TEACHING_BUCKET_NEWLINE:
            priority += 0.20f * tm->bucket_deficit[TEACHING_BUCKET_LANGUAGE];
            break;
        default:
            break;
    }
    return priority;
}

static int teaching_should_replay(const TeachingMemory *tm, uint32_t step) {
    if (!tm || tm->size < 8 || step < TEACHING_WARMUP_STEPS) return 0;
    return (uint32_t)(rng_next((uint64_t *)tm->rng) % TEACHING_REPLAY_DENOM) < TEACHING_REPLAY_NUMER;
}

static void teaching_build_boundary_mask_impl(const uint16_t *tokens, uint8_t *boundary_mask, int seq_len) {
    boundary_mask[0] = 1;
    for (int t = 1; t < seq_len; t++) {
        boundary_mask[t] = (tokens[t - 1] == '\n') ? 1 : 0;
    }
}

static void teaching_update_deficits(TeachingMemory *tm, uint8_t bucket, float loss) {
    if (!tm) return;
    if (tm->global_loss_ema <= 0.0f) tm->global_loss_ema = loss;
    tm->global_loss_ema = 0.97f * tm->global_loss_ema + 0.03f * loss;
    tm->bucket_loss_ema[bucket] = tm->bucket_loss_ema[bucket] <= 0.0f
        ? loss
        : 0.92f * tm->bucket_loss_ema[bucket] + 0.08f * loss;

    for (int b = 0; b < TEACHING_BUCKET_COUNT; b++) {
        float bucket_ema = tm->bucket_loss_ema[b] > 0.0f ? tm->bucket_loss_ema[b] : tm->global_loss_ema;
        float deficit = bucket_ema - tm->global_loss_ema;
        tm->bucket_deficit[b] = deficit > 0.0f ? deficit : 0.0f;
    }
}

static char *teaching_strdup_trimmed(const char *src) {
    size_t len = strlen(src);
    while (len > 0 && (src[len - 1] == '\n' || src[len - 1] == '\r')) len--;
    char *dst = (char *)malloc(len + 1);
    if (!dst) return NULL;
    memcpy(dst, src, len);
    dst[len] = '\0';
    return dst;
}

static char *teaching_vocab_path_from_dataset(const char *dataset_path) {
    size_t len = strlen(dataset_path);
    char *path = (char *)malloc(len + 16);
    if (!path) return NULL;
    memcpy(path, dataset_path, len + 1);

    for (size_t i = len; i > 0; i--) {
        if (path[i - 1] == '/' || path[i - 1] == '\\') {
            strcpy(path + i, "vocab.txt");
            return path;
        }
    }

    strcpy(path, "vocab.txt");
    return path;
}

static void teaching_load_vocab(TeachingMemory *tm, const char *dataset_path) {
    char *path = teaching_vocab_path_from_dataset(dataset_path);
    if (!path) return;

    FILE *fp = fopen(path, "rb");
    free(path);
    if (!fp) return;

    tm->vocab = (char **)calloc(32064, sizeof(char *));
    if (!tm->vocab) {
        fclose(fp);
        return;
    }

    char line[512];
    int idx = 0;
    while (idx < 32064 && fgets(line, sizeof(line), fp) != NULL) {
        tm->vocab[idx] = teaching_strdup_trimmed(line);
        idx++;
    }
    tm->vocab_size = idx;
    fclose(fp);
}

TeachingMemory *teaching_create(const char *dataset_path, int seq_len, int capacity, uint64_t seed) {
    TeachingMemory *tm = (TeachingMemory *)calloc(1, sizeof(TeachingMemory));
    if (!tm) return NULL;

    tm->seq_len = seq_len;
    tm->capacity = capacity;
    tm->episode_tokens = (uint16_t *)malloc((size_t)capacity * (size_t)seq_len * sizeof(uint16_t));
    tm->episodes = (TeachingEpisode *)calloc((size_t)capacity, sizeof(TeachingEpisode));
    if (!tm->episode_tokens || !tm->episodes) {
        teaching_free(tm);
        return NULL;
    }

    rng_seed(tm->rng, seed ? seed : 1u);
    teaching_load_vocab(tm, dataset_path);
    return tm;
}

void teaching_free(TeachingMemory *tm) {
    if (!tm) return;
    if (tm->vocab) {
        for (int i = 0; i < tm->vocab_size; i++) free(tm->vocab[i]);
        free(tm->vocab);
    }
    free(tm->episode_tokens);
    free(tm->episodes);
    free(tm);
}

static int teaching_find_episode_index(TeachingMemory *tm, uint32_t key) {
    for (int i = 0; i < tm->size; i++) {
        if (tm->episodes[i].active && tm->episodes[i].key == key) return i;
    }
    return -1;
}

static int teaching_pick_replay_slot(TeachingMemory *tm) {
    float best_score = -FLT_MAX;
    int best_slot = -1;

    for (int i = 0; i < tm->size; i++) {
        TeachingEpisode *ep = &tm->episodes[i];
        if (!ep->active) continue;
        if (ep->pending_evidence && ep->evidence_count < TEACHING_EVIDENCE_MIN) continue;

        float score = ep->loss_ema;
        score *= 1.0f + teaching_bucket_priority(tm, ep->bucket);
        score *= 1.0f + 0.20f * (float)ep->repair_count;
        score *= 1.0f + 0.15f * (float)ep->conflict_count;
        score *= 1.0f + fmaxf(ep->gain_ema, 0.0f);
        score /= 1.0f + 0.25f * (float)ep->replay_count;
        if (score > best_score) {
            best_score = score;
            best_slot = i;
        }
    }

    return best_slot;
}

int teaching_load_tokens(TeachingMemory *tm,
                         Dataset *ds,
                         uint16_t *tokens,
                         uint8_t *boundary_mask,
                         int seq_len,
                         int use_sequential,
                         uint32_t step,
                         int *out_seq_idx,
                         TeachingSelection *selection) {
    if (selection) {
        selection->used_replay = 0;
        selection->slot = 0;
        selection->bucket = TEACHING_BUCKET_LANGUAGE;
    }

    if (use_sequential || !tm || seq_len != tm->seq_len || !teaching_should_replay(tm, step)) {
        int seq_idx = -1;
        if (use_sequential) {
            dataset_next_sequential(ds, tokens, 1, seq_len);
            int n_seqs = (int)(ds->n_tokens / (size_t)seq_len);
            seq_idx = (int)((ds->cursor / (size_t)seq_len) % (size_t)n_seqs);
            if (seq_idx == 0 && ds->cursor > 0) seq_idx = n_seqs - 1;
            else if (seq_idx > 0) seq_idx--;
        } else {
            dataset_sample_batch(ds, tokens, 1, seq_len);
            seq_idx = 0;
        }
        teaching_build_boundary_mask_impl(tokens, boundary_mask, seq_len);
        if (out_seq_idx) *out_seq_idx = seq_idx;
        return 0;
    }

    int slot = teaching_pick_replay_slot(tm);
    if (slot < 0) {
        dataset_sample_batch(ds, tokens, 1, seq_len);
        teaching_build_boundary_mask_impl(tokens, boundary_mask, seq_len);
        if (out_seq_idx) *out_seq_idx = 0;
        return 0;
    }

    memcpy(tokens,
           tm->episode_tokens + (size_t)slot * (size_t)seq_len,
           (size_t)seq_len * sizeof(uint16_t));
    teaching_build_boundary_mask_impl(tokens, boundary_mask, seq_len);
    if (selection) {
        selection->used_replay = 1;
        selection->slot = (uint32_t)slot;
        selection->bucket = tm->episodes[slot].bucket;
    }
    if (out_seq_idx) *out_seq_idx = 0;
    tm->replay_events++;
    return 1;
}

void teaching_observe(TeachingMemory *tm,
                      const uint16_t *tokens,
                      int seq_len,
                      float loss,
                      const TeachingSelection *selection) {
    if (!tm || seq_len != tm->seq_len) return;

    if (selection && selection->used_replay && selection->slot < (uint32_t)tm->size) {
        TeachingEpisode *ep = &tm->episodes[selection->slot];
        float delta = ep->loss_ema - loss;
        ep->replay_count++;
        ep->gain_ema = ep->replay_count == 1
            ? delta
            : 0.80f * ep->gain_ema + 0.20f * delta;
        ep->last_loss = loss;
        ep->loss_ema = 0.80f * ep->loss_ema + 0.20f * loss;
        if (loss < ep->best_loss) ep->best_loss = loss;
        if (delta > 0.02f) {
            ep->repair_count++;
            tm->repair_events++;
            ep->pending_evidence = 0;
        } else if (delta < -0.02f) {
            ep->conflict_count++;
            ep->deferral_count++;
            ep->pending_evidence = 1;
            ep->evidence_count = 0;
            tm->conflict_events++;
            tm->deferred_events++;
        }
        teaching_update_deficits(tm, ep->bucket, loss);
        return;
    }

    uint8_t bucket = teaching_classify_window(tm, tokens, seq_len);
    uint32_t key = fnv1a_u32_tokens(tokens, seq_len < TEACHING_KEY_PREFIX ? seq_len : TEACHING_KEY_PREFIX);
    key ^= (uint32_t)bucket << 24;

    int slot = teaching_find_episode_index(tm, key);
    if (slot < 0) {
        slot = (tm->size < tm->capacity) ? tm->size++ : tm->cursor;
        tm->cursor = (tm->cursor + 1) % tm->capacity;
        memset(&tm->episodes[slot], 0, sizeof(TeachingEpisode));
        memcpy(tm->episode_tokens + (size_t)slot * (size_t)seq_len,
               tokens,
               (size_t)seq_len * sizeof(uint16_t));
        tm->episodes[slot].key = key;
        tm->episodes[slot].bucket = bucket;
        tm->episodes[slot].best_loss = loss;
        tm->episodes[slot].active = 1;
    }

    TeachingEpisode *ep = &tm->episodes[slot];
    ep->seen_count++;
    ep->last_loss = loss;
    ep->loss_ema = ep->seen_count == 1 ? loss : 0.85f * ep->loss_ema + 0.15f * loss;
    if (loss < ep->best_loss) ep->best_loss = loss;
    ep->bucket = bucket;
    if (ep->pending_evidence) {
        ep->evidence_count = ep->evidence_count < 255 ? ep->evidence_count + 1u : ep->evidence_count;
        if (ep->evidence_count >= TEACHING_EVIDENCE_MIN) {
            ep->pending_evidence = 0;
        } else {
            tm->deferred_events++;
        }
    } else if (ep->evidence_count == 0) {
        ep->evidence_count = 1;
    }
    tm->fresh_events++;
    teaching_update_deficits(tm, bucket, loss);
}

void teaching_log_step(TeachingMemory *tm, uint32_t step) {
    if (!tm) return;

    int eligible = 0;
    int deferred = 0;
    for (int i = 0; i < tm->size; i++) {
        TeachingEpisode *ep = &tm->episodes[i];
        if (!ep->active) continue;
        if (ep->pending_evidence && ep->evidence_count < TEACHING_EVIDENCE_MIN) {
            deferred++;
            continue;
        }
        eligible++;
    }

    int top_bucket = 0;
    float top_priority = teaching_bucket_priority(tm, 0);
    for (int b = 1; b < TEACHING_BUCKET_COUNT; b++) {
        float priority = teaching_bucket_priority(tm, (uint8_t)b);
        if (priority > top_priority) {
            top_priority = priority;
            top_bucket = b;
        }
    }

    printf("[TEACH] step=%u stored=%d eligible=%d deferred=%d fresh=%u replay=%u repairs=%u conflicts=%u top=%s deficit=%.4f\n",
           step,
           tm->size,
           eligible,
           deferred,
           tm->fresh_events,
           tm->replay_events,
           tm->repair_events,
           tm->conflict_events,
           teaching_bucket_name((uint8_t)top_bucket),
           top_priority);
}
