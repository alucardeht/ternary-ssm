#include "../src/io/dataset.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint16_t *data;
    size_t n_tokens;
    Dataset *dataset_owner;
} TokenView;

typedef struct {
    uint64_t hash;
    uint64_t offset;
    uint32_t count;
    uint8_t used;
} ContextEntry;

typedef struct {
    ContextEntry *entries;
    uint64_t capacity;
    uint64_t unique;
    uint64_t collisions;
} ContextTable;

typedef struct {
    uint64_t hash;
    uint64_t offset;
    uint16_t target;
    uint32_t count;
    uint8_t used;
} PairEntry;

typedef struct {
    PairEntry *entries;
    uint64_t capacity;
    uint64_t unique;
    uint64_t collisions;
} PairTable;

typedef struct {
    uint64_t train_terms;
    uint64_t eval_terms;
    uint64_t eval_seen_contexts;
    uint64_t eval_unseen_contexts;
    double unigram_ce;
    double scaffold_ce;
    double build_seconds;
    double eval_seconds;
} GateStats;

static uint64_t fnv1a_update_u16(uint64_t h, uint16_t v) {
    h ^= (uint64_t)(v & 0xffu);
    h *= 1099511628211ull;
    h ^= (uint64_t)((v >> 8) & 0xffu);
    h *= 1099511628211ull;
    return h;
}

static uint64_t context_hash(const uint16_t *data, uint64_t offset, uint64_t order) {
    uint64_t h = 1469598103934665603ull;
    h = fnv1a_update_u16(h, (uint16_t)(order & 0xffffu));
    h = fnv1a_update_u16(h, (uint16_t)((order >> 16) & 0xffffu));
    for (uint64_t i = 0; i < order; i++) {
        h = fnv1a_update_u16(h, data[offset + i]);
    }
    return h;
}

static int context_equal(const uint16_t *a_data,
                         uint64_t a_offset,
                         const uint16_t *b_data,
                         uint64_t b_offset,
                         uint64_t order) {
    for (uint64_t i = 0; i < order; i++) {
        if (a_data[a_offset + i] != b_data[b_offset + i]) return 0;
    }
    return 1;
}

static uint64_t next_power_of_two(uint64_t x) {
    if (x < 16) return 16;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}

static uint64_t parse_u64_arg(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr, "[R408_SCAFFOLD_GATE] invalid %s: %s\n", name, text);
        exit(2);
    }
    return (uint64_t)value;
}

static double parse_double_arg(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    double value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value)) {
        fprintf(stderr, "[R408_SCAFFOLD_GATE] invalid %s: %s\n", name, text);
        exit(2);
    }
    return value;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s <train.bin> <eval.bin> <order> [options]\n"
            "\n"
            "R408 Corpus Scaffold + Residual CE pre-gate.\n"
            "Builds a fixed corpus-owned n-gram scaffold and evaluates held-out\n"
            "next-token CE. The scaffold is an additive prior only; residual\n"
            "training remains free to represent the full original logit space.\n"
            "\n"
            "options:\n"
            "  --max-train-terms N   cap training n-gram terms\n"
            "  --max-eval-terms N    cap eval n-gram terms\n"
            "  --train-stride N      stride over training terms\n"
            "  --eval-stride N       stride over eval terms\n"
            "  --raw-u8              load train/eval as raw byte tokens\n"
            "  --vocab-size N        vocabulary size for uniform CE (default 32064)\n"
            "  --alpha X             unigram-backoff pseudocount mass (default 1.0)\n"
            "  --target-ce X         optional target CE for gamma calculation\n",
            argv0);
}

static int load_raw_u8_as_u16(const char *path, TokenView *view) {
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "[R408_SCAFFOLD_GATE] cannot open raw-u8 file: %s\n", path);
        return 0;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "[R408_SCAFFOLD_GATE] cannot seek raw-u8 file: %s\n", path);
        fclose(fp);
        return 0;
    }
    long size_long = ftell(fp);
    if (size_long <= 0) {
        fprintf(stderr, "[R408_SCAFFOLD_GATE] empty or invalid raw-u8 file: %s\n", path);
        fclose(fp);
        return 0;
    }
    rewind(fp);

    size_t n = (size_t)size_long;
    uint8_t *bytes = (uint8_t *)malloc(n);
    uint16_t *tokens = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (bytes == NULL || tokens == NULL) {
        fprintf(stderr, "[R408_SCAFFOLD_GATE] OOM loading raw-u8 file size=%zu\n", n);
        free(bytes);
        free(tokens);
        fclose(fp);
        return 0;
    }
    size_t got = fread(bytes, 1, n, fp);
    fclose(fp);
    if (got != n) {
        fprintf(stderr, "[R408_SCAFFOLD_GATE] short raw-u8 read: got=%zu expected=%zu\n", got, n);
        free(bytes);
        free(tokens);
        return 0;
    }
    for (size_t i = 0; i < n; i++) tokens[i] = (uint16_t)bytes[i];
    free(bytes);

    view->data = tokens;
    view->n_tokens = n;
    view->dataset_owner = NULL;
    return 1;
}

static int load_token_view(const char *path, int raw_u8, TokenView *view) {
    memset(view, 0, sizeof(*view));
    if (raw_u8) return load_raw_u8_as_u16(path, view);

    Dataset *ds = dataset_load(path);
    if (ds == NULL) return 0;
    view->data = ds->data;
    view->n_tokens = ds->n_tokens;
    view->dataset_owner = ds;
    return 1;
}

static void free_token_view(TokenView *view) {
    if (view->dataset_owner != NULL) {
        dataset_free(view->dataset_owner);
    } else {
        free(view->data);
    }
    memset(view, 0, sizeof(*view));
}

static int context_table_init(ContextTable *table, uint64_t expected_terms) {
    uint64_t capacity = next_power_of_two(expected_terms * 2u + 1u);
    table->entries = (ContextEntry *)calloc((size_t)capacity, sizeof(ContextEntry));
    if (table->entries == NULL) {
        fprintf(stderr, "[R408_SCAFFOLD_GATE] OOM context table capacity=%" PRIu64 "\n", capacity);
        return 0;
    }
    table->capacity = capacity;
    table->unique = 0;
    table->collisions = 0;
    return 1;
}

static void context_table_free(ContextTable *table) {
    free(table->entries);
    memset(table, 0, sizeof(*table));
}

static ContextEntry *context_table_find(ContextTable *table,
                                        const uint16_t *table_data,
                                        const uint16_t *query_data,
                                        uint64_t query_offset,
                                        uint64_t order) {
    if (order == 0) return NULL;
    uint64_t h = context_hash(query_data, query_offset, order);
    uint64_t mask = table->capacity - 1u;
    uint64_t slot = h & mask;
    for (uint64_t probe = 0; probe < table->capacity; probe++) {
        ContextEntry *entry = &table->entries[slot];
        if (!entry->used) return NULL;
        if (entry->hash == h &&
            context_equal(table_data, entry->offset, query_data, query_offset, order)) {
            return entry;
        }
        slot = (slot + 1u) & mask;
    }
    return NULL;
}

static int context_table_add(ContextTable *table,
                             const uint16_t *data,
                             uint64_t offset,
                             uint64_t order) {
    if (order == 0) return 1;
    uint64_t h = context_hash(data, offset, order);
    uint64_t mask = table->capacity - 1u;
    uint64_t slot = h & mask;
    for (uint64_t probe = 0; probe < table->capacity; probe++) {
        ContextEntry *entry = &table->entries[slot];
        if (!entry->used) {
            entry->used = 1;
            entry->hash = h;
            entry->offset = offset;
            entry->count = 1;
            table->unique++;
            return 1;
        }
        if (entry->hash == h && context_equal(data, entry->offset, data, offset, order)) {
            entry->count++;
            return 1;
        }
        table->collisions++;
        slot = (slot + 1u) & mask;
    }
    fprintf(stderr, "[R408_SCAFFOLD_GATE] context table full\n");
    return 0;
}

static int pair_table_init(PairTable *table, uint64_t expected_terms) {
    uint64_t capacity = next_power_of_two(expected_terms * 2u + 1u);
    table->entries = (PairEntry *)calloc((size_t)capacity, sizeof(PairEntry));
    if (table->entries == NULL) {
        fprintf(stderr, "[R408_SCAFFOLD_GATE] OOM pair table capacity=%" PRIu64 "\n", capacity);
        return 0;
    }
    table->capacity = capacity;
    table->unique = 0;
    table->collisions = 0;
    return 1;
}

static void pair_table_free(PairTable *table) {
    free(table->entries);
    memset(table, 0, sizeof(*table));
}

static uint64_t pair_hash(const uint16_t *data, uint64_t offset, uint64_t order, uint16_t target) {
    uint64_t h = context_hash(data, offset, order);
    h = fnv1a_update_u16(h, target);
    return h;
}

static PairEntry *pair_table_find(PairTable *table,
                                  const uint16_t *table_data,
                                  const uint16_t *query_data,
                                  uint64_t query_offset,
                                  uint64_t order,
                                  uint16_t target) {
    uint64_t h = pair_hash(query_data, query_offset, order, target);
    uint64_t mask = table->capacity - 1u;
    uint64_t slot = h & mask;
    for (uint64_t probe = 0; probe < table->capacity; probe++) {
        PairEntry *entry = &table->entries[slot];
        if (!entry->used) return NULL;
        if (entry->hash == h &&
            entry->target == target &&
            context_equal(table_data, entry->offset, query_data, query_offset, order)) {
            return entry;
        }
        slot = (slot + 1u) & mask;
    }
    return NULL;
}

static int pair_table_add(PairTable *table,
                          const uint16_t *data,
                          uint64_t offset,
                          uint64_t order,
                          uint16_t target) {
    uint64_t h = pair_hash(data, offset, order, target);
    uint64_t mask = table->capacity - 1u;
    uint64_t slot = h & mask;
    for (uint64_t probe = 0; probe < table->capacity; probe++) {
        PairEntry *entry = &table->entries[slot];
        if (!entry->used) {
            entry->used = 1;
            entry->hash = h;
            entry->offset = offset;
            entry->target = target;
            entry->count = 1;
            table->unique++;
            return 1;
        }
        if (entry->hash == h &&
            entry->target == target &&
            context_equal(data, entry->offset, data, offset, order)) {
            entry->count++;
            return 1;
        }
        table->collisions++;
        slot = (slot + 1u) & mask;
    }
    fprintf(stderr, "[R408_SCAFFOLD_GATE] pair table full\n");
    return 0;
}

static double token_unigram_probability(const uint64_t *counts,
                                        uint64_t total,
                                        uint64_t vocab_size,
                                        uint16_t token,
                                        double alpha) {
    uint64_t count = token < vocab_size ? counts[token] : 0u;
    double denom = (double)total + alpha * (double)vocab_size;
    if (denom <= 0.0) return 1.0 / (double)vocab_size;
    return ((double)count + alpha) / denom;
}

static int run_gate(const TokenView *train,
                    const TokenView *eval,
                    uint64_t order,
                    uint64_t vocab_size,
                    uint64_t max_train_terms,
                    uint64_t max_eval_terms,
                    uint64_t train_stride,
                    uint64_t eval_stride,
                    double alpha,
                    GateStats *stats) {
    memset(stats, 0, sizeof(*stats));
    if (vocab_size == 0 || vocab_size > 65536u || alpha <= 0.0) {
        fprintf(stderr, "[R408_SCAFFOLD_GATE] invalid vocab_size or alpha\n");
        return 0;
    }
    if ((uint64_t)train->n_tokens <= order || (uint64_t)eval->n_tokens <= order) {
        fprintf(stderr, "[R408_SCAFFOLD_GATE] dataset too small for order=%" PRIu64 "\n", order);
        return 0;
    }

    uint64_t possible_train = ((uint64_t)train->n_tokens - order + train_stride - 1u) / train_stride;
    uint64_t possible_eval = ((uint64_t)eval->n_tokens - order + eval_stride - 1u) / eval_stride;
    uint64_t train_terms = possible_train;
    uint64_t eval_terms = possible_eval;
    if (max_train_terms > 0 && max_train_terms < train_terms) train_terms = max_train_terms;
    if (max_eval_terms > 0 && max_eval_terms < eval_terms) eval_terms = max_eval_terms;

    uint64_t *unigram_counts = (uint64_t *)calloc((size_t)vocab_size, sizeof(uint64_t));
    if (unigram_counts == NULL) {
        fprintf(stderr, "[R408_SCAFFOLD_GATE] OOM unigram counts vocab=%" PRIu64 "\n", vocab_size);
        return 0;
    }

    ContextTable contexts;
    PairTable pairs;
    memset(&contexts, 0, sizeof(contexts));
    memset(&pairs, 0, sizeof(pairs));
    if (order > 0 && !context_table_init(&contexts, train_terms)) {
        free(unigram_counts);
        return 0;
    }
    if (!pair_table_init(&pairs, train_terms)) {
        context_table_free(&contexts);
        free(unigram_counts);
        return 0;
    }

    clock_t build_start = clock();
    uint64_t scanned_train = 0;
    for (uint64_t pos = order;
         pos < (uint64_t)train->n_tokens && scanned_train < train_terms;
         pos += train_stride) {
        uint16_t target = train->data[pos];
        if (target < vocab_size) unigram_counts[target]++;
        uint64_t context_offset = pos - order;
        if (!context_table_add(&contexts, train->data, context_offset, order) ||
            !pair_table_add(&pairs, train->data, context_offset, order, target)) {
            pair_table_free(&pairs);
            context_table_free(&contexts);
            free(unigram_counts);
            return 0;
        }
        scanned_train++;
    }
    clock_t build_end = clock();

    clock_t eval_start = clock();
    double unigram_nll = 0.0;
    double scaffold_nll = 0.0;
    uint64_t scanned_eval = 0;
    uint64_t seen_contexts = 0;
    uint64_t unseen_contexts = 0;
    for (uint64_t pos = order;
         pos < (uint64_t)eval->n_tokens && scanned_eval < eval_terms;
         pos += eval_stride) {
        uint16_t target = eval->data[pos];
        uint64_t context_offset = pos - order;
        double p_uni = token_unigram_probability(unigram_counts,
                                                 scanned_train,
                                                 vocab_size,
                                                 target,
                                                 alpha);
        double p_scaffold = p_uni;
        if (order > 0) {
            ContextEntry *ctx = context_table_find(&contexts,
                                                   train->data,
                                                   eval->data,
                                                   context_offset,
                                                   order);
            if (ctx != NULL && ctx->count > 0) {
                PairEntry *pair = pair_table_find(&pairs,
                                                  train->data,
                                                  eval->data,
                                                  context_offset,
                                                  order,
                                                  target);
                double pair_count = pair != NULL ? (double)pair->count : 0.0;
                p_scaffold = (pair_count + alpha * p_uni) / ((double)ctx->count + alpha);
                seen_contexts++;
            } else {
                unseen_contexts++;
            }
        } else {
            seen_contexts++;
        }
        if (p_uni <= 0.0) p_uni = 1.0 / (double)vocab_size;
        if (p_scaffold <= 0.0) p_scaffold = 1.0 / (double)vocab_size;
        unigram_nll -= log(p_uni);
        scaffold_nll -= log(p_scaffold);
        scanned_eval++;
    }
    clock_t eval_end = clock();

    stats->train_terms = scanned_train;
    stats->eval_terms = scanned_eval;
    stats->eval_seen_contexts = seen_contexts;
    stats->eval_unseen_contexts = unseen_contexts;
    stats->unigram_ce = scanned_eval > 0 ? unigram_nll / (double)scanned_eval : NAN;
    stats->scaffold_ce = scanned_eval > 0 ? scaffold_nll / (double)scanned_eval : NAN;
    stats->build_seconds = (double)(build_end - build_start) / (double)CLOCKS_PER_SEC;
    stats->eval_seconds = (double)(eval_end - eval_start) / (double)CLOCKS_PER_SEC;

    pair_table_free(&pairs);
    context_table_free(&contexts);
    free(unigram_counts);
    return scanned_train > 0 && scanned_eval > 0;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        usage(argv[0]);
        return 2;
    }

    const char *train_path = argv[1];
    const char *eval_path = argv[2];
    uint64_t order = parse_u64_arg(argv[3], "order");
    uint64_t max_train_terms = 0;
    uint64_t max_eval_terms = 0;
    uint64_t train_stride = 1;
    uint64_t eval_stride = 1;
    uint64_t vocab_size = 32064;
    double alpha = 1.0;
    double target_ce = NAN;
    int raw_u8 = 0;

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--max-train-terms") == 0 && i + 1 < argc) {
            max_train_terms = parse_u64_arg(argv[++i], "max_train_terms");
        } else if (strcmp(argv[i], "--max-eval-terms") == 0 && i + 1 < argc) {
            max_eval_terms = parse_u64_arg(argv[++i], "max_eval_terms");
        } else if (strcmp(argv[i], "--train-stride") == 0 && i + 1 < argc) {
            train_stride = parse_u64_arg(argv[++i], "train_stride");
        } else if (strcmp(argv[i], "--eval-stride") == 0 && i + 1 < argc) {
            eval_stride = parse_u64_arg(argv[++i], "eval_stride");
        } else if (strcmp(argv[i], "--vocab-size") == 0 && i + 1 < argc) {
            vocab_size = parse_u64_arg(argv[++i], "vocab_size");
        } else if (strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) {
            alpha = parse_double_arg(argv[++i], "alpha");
        } else if (strcmp(argv[i], "--target-ce") == 0 && i + 1 < argc) {
            target_ce = parse_double_arg(argv[++i], "target_ce");
        } else if (strcmp(argv[i], "--raw-u8") == 0) {
            raw_u8 = 1;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (train_stride == 0 || eval_stride == 0) {
        fprintf(stderr, "[R408_SCAFFOLD_GATE] strides must be > 0\n");
        return 2;
    }

    TokenView train;
    TokenView eval;
    if (!load_token_view(train_path, raw_u8, &train)) return 1;
    if (!load_token_view(eval_path, raw_u8, &eval)) {
        free_token_view(&train);
        return 1;
    }

    GateStats stats;
    int ok = run_gate(&train,
                      &eval,
                      order,
                      vocab_size,
                      max_train_terms,
                      max_eval_terms,
                      train_stride,
                      eval_stride,
                      alpha,
                      &stats);
    if (!ok) {
        free_token_view(&eval);
        free_token_view(&train);
        return 1;
    }

    double uniform_ce = log((double)vocab_size);
    double gap_removed_vs_uniform = (uniform_ce - stats.scaffold_ce) / uniform_ce;
    double scaffold_delta_vs_unigram = stats.unigram_ce - stats.scaffold_ce;
    double gamma_if_target = NAN;
    double ideal_speedup_if_target = NAN;
    double max_kappa_material_h0 = NAN;
    double max_kappa_headline_h0 = NAN;
    double min_target_ce_for_2x_ideal = 2.0 * stats.scaffold_ce - uniform_ce;
    const char *decision = "REPORT_ONLY_NEEDS_RESIDUAL_KAPPA_GATE";
    const char *headline_2x = "NOT_EVALUATED";
    if (isfinite(target_ce) && uniform_ce > target_ce) {
        gamma_if_target = (uniform_ce - stats.scaffold_ce) / (uniform_ce - target_ce);
        ideal_speedup_if_target = gamma_if_target < 1.0 ? 1.0 / (1.0 - gamma_if_target) : INFINITY;
        if (gamma_if_target < 1.0) {
            max_kappa_material_h0 = 0.90 / (1.0 - gamma_if_target);
            max_kappa_headline_h0 = 0.50 / (1.0 - gamma_if_target);
        }
        headline_2x = gamma_if_target >= 0.5 ? "PASSES_2X_HEADLINE_PRECONDITION"
                                             : "MISSES_2X_HEADLINE_PRECONDITION";
        decision = gamma_if_target >= 0.1 ? "PASSES_MATERIAL_GAIN_PRECONDITION"
                                          : "FAILS_MATERIAL_GAIN_PRECONDITION";
    }
    if (scaffold_delta_vs_unigram <= 0.0 && order > 0) {
        decision = "FAILS_HELDOUT_SCAFFOLD_BEATS_UNIGRAM";
        headline_2x = "MISSES_2X_HEADLINE_PRECONDITION";
    }

    printf("[R408_SCAFFOLD_GATE] active=1 source=corpus-owned-ngram-scaffold train=%s eval=%s "
           "order=%" PRIu64 " vocab_size=%" PRIu64 " alpha=%.6f raw_u8=%d "
           "residual_freedom=additive_logit_prior_only\n",
           train_path,
           eval_path,
           order,
           vocab_size,
           alpha,
           raw_u8);
    printf("{\"decision\":\"%s\","
           "\"order\":%" PRIu64 ","
           "\"vocab_size\":%" PRIu64 ","
           "\"train_terms\":%" PRIu64 ","
           "\"eval_terms\":%" PRIu64 ","
           "\"eval_seen_contexts\":%" PRIu64 ","
           "\"eval_unseen_contexts\":%" PRIu64 ","
           "\"uniform_ce\":%.12f,"
           "\"unigram_ce\":%.12f,"
           "\"scaffold_ce\":%.12f,"
           "\"scaffold_delta_vs_unigram\":%.12f,"
           "\"gap_removed_vs_uniform\":%.12f,"
           "\"target_ce\":%.12f,"
           "\"gamma_if_target_ce\":%.12f,"
           "\"ideal_speedup_if_target_ce\":%.12f,"
           "\"max_kappa_for_10pct_gain_h0\":%.12f,"
           "\"max_kappa_for_2x_headline_h0\":%.12f,"
           "\"headline_2x_precondition\":\"%s\","
           "\"min_target_ce_for_2x_ideal\":%.12f,"
           "\"build_seconds\":%.6f,"
           "\"eval_seconds\":%.6f,"
           "\"raw_u8\":%d,"
           "\"residual_freedom\":\"additive_logit_prior_only\"}\n",
           decision,
           order,
           vocab_size,
           stats.train_terms,
           stats.eval_terms,
           stats.eval_seen_contexts,
           stats.eval_unseen_contexts,
           uniform_ce,
           stats.unigram_ce,
           stats.scaffold_ce,
           scaffold_delta_vs_unigram,
           gap_removed_vs_uniform,
           target_ce,
           gamma_if_target,
           ideal_speedup_if_target,
           max_kappa_material_h0,
           max_kappa_headline_h0,
           headline_2x,
           min_target_ce_for_2x_ideal,
           stats.build_seconds,
           stats.eval_seconds,
           raw_u8);

    free_token_view(&eval);
    free_token_view(&train);
    return 0;
}
