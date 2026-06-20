#include "../src/io/dataset.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

typedef struct {
    uint64_t key;
    uint32_t count;
    uint8_t used;
} PairEntry64;

typedef struct {
    PairEntry64 *entries;
    uint64_t capacity;
    uint64_t unique;
    uint64_t collisions;
} PairTable64;

typedef struct {
    uint16_t *data;
    size_t n_tokens;
    Dataset *dataset_owner;
} TokenView;

typedef struct {
    uint64_t terms;
    double ce;
    double train_seconds;
    double eval_seconds;
    uint64_t residual_contexts;
    uint64_t residual_pairs;
} Checkpoint;

typedef struct {
    Checkpoint *items;
    uint64_t count;
    uint64_t capacity;
} CheckpointList;

typedef struct {
    uint64_t *context_counts;
    PairTable64 pairs;
    uint64_t unique_contexts;
} ResidualCounts;

static double now_seconds(void) {
    return omp_get_wtime();
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

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
}

static uint64_t pair_key(uint16_t context, uint16_t target) {
    return ((uint64_t)context << 32) | (uint64_t)target;
}

static uint64_t parse_u64_arg(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr, "[R408_SCAFFOLD_RESIDUAL_PILOT] invalid %s: %s\n", name, text);
        exit(2);
    }
    return (uint64_t)value;
}

static double parse_double_arg(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    double value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value)) {
        fprintf(stderr, "[R408_SCAFFOLD_RESIDUAL_PILOT] invalid %s: %s\n", name, text);
        exit(2);
    }
    return value;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s <train.bin> <eval.bin> [options]\n"
            "\n"
            "R408 corpus scaffold residual CE pilot. This is an identifying\n"
            "bigram-only native gate, not the production trainer.\n"
            "\n"
            "options:\n"
            "  --vocab-size N          vocabulary size (default 32064)\n"
            "  --alpha X               scaffold unigram-backoff mass (default 100)\n"
            "  --beta X                residual prior mass (default 1000)\n"
            "  --scaffold-terms N      corpus scaffold build terms (default 5000000)\n"
            "  --scaffold-stride N     scaffold stride (default 22)\n"
            "  --residual-terms N      residual train terms (default 200000)\n"
            "  --residual-stride N     residual stride (default 233)\n"
            "  --checkpoint-every N    residual checkpoint interval (default 25000)\n"
            "  --eval-terms N          held-out eval terms (default 20000)\n"
            "  --eval-stride N         eval stride (default 1)\n"
            "  --scaffold-reuse N      priced scaffold-build reuse count (default 1)\n"
            "  --gamma X               reference scaffold gamma (default 0.535298658010)\n",
            argv0);
}

static int load_token_view(const char *path, TokenView *view) {
    memset(view, 0, sizeof(*view));
    Dataset *ds = dataset_load(path);
    if (ds == NULL) return 0;
    view->data = ds->data;
    view->n_tokens = ds->n_tokens;
    view->dataset_owner = ds;
    return 1;
}

static void free_token_view(TokenView *view) {
    if (view->dataset_owner != NULL) dataset_free(view->dataset_owner);
    memset(view, 0, sizeof(*view));
}

static int pair_table_init(PairTable64 *table, uint64_t expected_terms) {
    uint64_t capacity = next_power_of_two(expected_terms * 2u + 1u);
    table->entries = (PairEntry64 *)calloc((size_t)capacity, sizeof(PairEntry64));
    if (table->entries == NULL) {
        fprintf(stderr, "[R408_SCAFFOLD_RESIDUAL_PILOT] OOM pair table capacity=%" PRIu64 "\n", capacity);
        return 0;
    }
    table->capacity = capacity;
    table->unique = 0;
    table->collisions = 0;
    return 1;
}

static void pair_table_free(PairTable64 *table) {
    free(table->entries);
    memset(table, 0, sizeof(*table));
}

static PairEntry64 *pair_table_find(PairTable64 *table, uint64_t key) {
    uint64_t mask = table->capacity - 1u;
    uint64_t slot = mix64(key) & mask;
    for (uint64_t probe = 0; probe < table->capacity; probe++) {
        PairEntry64 *entry = &table->entries[slot];
        if (!entry->used) return NULL;
        if (entry->key == key) return entry;
        slot = (slot + 1u) & mask;
    }
    return NULL;
}

static int pair_table_add(PairTable64 *table, uint64_t key) {
    uint64_t mask = table->capacity - 1u;
    uint64_t slot = mix64(key) & mask;
    for (uint64_t probe = 0; probe < table->capacity; probe++) {
        PairEntry64 *entry = &table->entries[slot];
        if (!entry->used) {
            entry->used = 1;
            entry->key = key;
            entry->count = 1;
            table->unique++;
            return 1;
        }
        if (entry->key == key) {
            entry->count++;
            return 1;
        }
        table->collisions++;
        slot = (slot + 1u) & mask;
    }
    fprintf(stderr, "[R408_SCAFFOLD_RESIDUAL_PILOT] pair table full\n");
    return 0;
}

static int checkpoints_init(CheckpointList *list, uint64_t residual_terms, uint64_t every) {
    uint64_t n = 2;
    if (every > 0) n += residual_terms / every + 2;
    list->items = (Checkpoint *)calloc((size_t)n, sizeof(Checkpoint));
    if (list->items == NULL) return 0;
    list->capacity = n;
    list->count = 0;
    return 1;
}

static void checkpoints_free(CheckpointList *list) {
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static int checkpoints_push(CheckpointList *list, Checkpoint cp) {
    if (list->count >= list->capacity) return 0;
    list->items[list->count++] = cp;
    return 1;
}

static double unigram_probability(const uint64_t *counts,
                                  uint64_t total,
                                  uint64_t vocab_size,
                                  uint16_t token,
                                  double alpha) {
    uint64_t count = token < vocab_size ? counts[token] : 0u;
    double denom = (double)total + alpha * (double)vocab_size;
    if (denom <= 0.0) return 1.0 / (double)vocab_size;
    return ((double)count + alpha) / denom;
}

static double scaffold_probability(uint16_t ctx,
                                   uint16_t target,
                                   const uint64_t *unigram_counts,
                                   uint64_t unigram_total,
                                   const uint64_t *scaffold_context_counts,
                                   PairTable64 *scaffold_pairs,
                                   uint64_t vocab_size,
                                   double alpha) {
    double p_uni = unigram_probability(unigram_counts, unigram_total, vocab_size, target, alpha);
    uint64_t ctx_count = ctx < vocab_size ? scaffold_context_counts[ctx] : 0u;
    if (ctx_count == 0) return p_uni;
    PairEntry64 *pair = pair_table_find(scaffold_pairs, pair_key(ctx, target));
    double pair_count = pair != NULL ? (double)pair->count : 0.0;
    return (pair_count + alpha * p_uni) / ((double)ctx_count + alpha);
}

static double eval_arm(int use_scaffold,
                       const TokenView *eval,
                       uint64_t eval_terms,
                       uint64_t eval_stride,
                       uint64_t vocab_size,
                       double alpha,
                       double beta,
                       const uint64_t *unigram_counts,
                       uint64_t unigram_total,
                       const uint64_t *scaffold_context_counts,
                       PairTable64 *scaffold_pairs,
                       ResidualCounts *residual,
                       double *eval_seconds_out,
                       uint64_t *actual_eval_terms_out) {
    double start = now_seconds();
    double nll = 0.0;
    uint64_t scanned = 0;
    for (uint64_t pos = 1; pos < (uint64_t)eval->n_tokens && scanned < eval_terms; pos += eval_stride) {
        uint16_t ctx = eval->data[pos - 1u];
        uint16_t target = eval->data[pos];
        double p_prior = use_scaffold
                             ? scaffold_probability(ctx,
                                                    target,
                                                    unigram_counts,
                                                    unigram_total,
                                                    scaffold_context_counts,
                                                    scaffold_pairs,
                                                    vocab_size,
                                                    alpha)
                             : unigram_probability(unigram_counts, unigram_total, vocab_size, target, alpha);
        uint64_t r_ctx = ctx < vocab_size ? residual->context_counts[ctx] : 0u;
        PairEntry64 *r_pair = pair_table_find(&residual->pairs, pair_key(ctx, target));
        double r_pair_count = r_pair != NULL ? (double)r_pair->count : 0.0;
        double p = (r_pair_count + beta * p_prior) / ((double)r_ctx + beta);
        if (p <= 0.0 || !isfinite(p)) p = 1.0 / (double)vocab_size;
        nll -= log(p);
        scanned++;
    }
    double end = now_seconds();
    *eval_seconds_out = end - start;
    *actual_eval_terms_out = scanned;
    return scanned > 0 ? nll / (double)scanned : NAN;
}

static int residual_init(ResidualCounts *residual, uint64_t vocab_size, uint64_t residual_terms) {
    memset(residual, 0, sizeof(*residual));
    residual->context_counts = (uint64_t *)calloc((size_t)vocab_size, sizeof(uint64_t));
    if (residual->context_counts == NULL) return 0;
    if (!pair_table_init(&residual->pairs, residual_terms)) {
        free(residual->context_counts);
        memset(residual, 0, sizeof(*residual));
        return 0;
    }
    return 1;
}

static void residual_free(ResidualCounts *residual) {
    free(residual->context_counts);
    pair_table_free(&residual->pairs);
    memset(residual, 0, sizeof(*residual));
}

static int append_eval_checkpoint(CheckpointList *checkpoints,
                                  int use_scaffold,
                                  const TokenView *eval,
                                  uint64_t terms,
                                  double train_seconds,
                                  uint64_t eval_terms,
                                  uint64_t eval_stride,
                                  uint64_t vocab_size,
                                  double alpha,
                                  double beta,
                                  const uint64_t *unigram_counts,
                                  uint64_t unigram_total,
                                  const uint64_t *scaffold_context_counts,
                                  PairTable64 *scaffold_pairs,
                                  ResidualCounts *residual) {
    double eval_seconds = 0.0;
    uint64_t actual_eval_terms = 0;
    double ce = eval_arm(use_scaffold,
                         eval,
                         eval_terms,
                         eval_stride,
                         vocab_size,
                         alpha,
                         beta,
                         unigram_counts,
                         unigram_total,
                         scaffold_context_counts,
                         scaffold_pairs,
                         residual,
                         &eval_seconds,
                         &actual_eval_terms);
    Checkpoint cp;
    cp.terms = terms;
    cp.ce = ce;
    cp.train_seconds = train_seconds;
    cp.eval_seconds = eval_seconds;
    cp.residual_contexts = residual->unique_contexts;
    cp.residual_pairs = residual->pairs.unique;
    return checkpoints_push(checkpoints, cp);
}

static int train_arm(int use_scaffold,
                     const TokenView *train,
                     const TokenView *eval,
                     uint64_t residual_terms,
                     uint64_t residual_stride,
                     uint64_t checkpoint_every,
                     uint64_t eval_terms,
                     uint64_t eval_stride,
                     uint64_t vocab_size,
                     double alpha,
                     double beta,
                     const uint64_t *unigram_counts,
                     uint64_t unigram_total,
                     const uint64_t *scaffold_context_counts,
                     PairTable64 *scaffold_pairs,
                     CheckpointList *checkpoints) {
    ResidualCounts residual;
    if (!residual_init(&residual, vocab_size, residual_terms)) return 0;
    if (!append_eval_checkpoint(checkpoints,
                                use_scaffold,
                                eval,
                                0,
                                0.0,
                                eval_terms,
                                eval_stride,
                                vocab_size,
                                alpha,
                                beta,
                                unigram_counts,
                                unigram_total,
                                scaffold_context_counts,
                                scaffold_pairs,
                                &residual)) {
        residual_free(&residual);
        return 0;
    }

    uint64_t terms = 0;
    double start = now_seconds();
    for (uint64_t pos = 1; pos < (uint64_t)train->n_tokens && terms < residual_terms; pos += residual_stride) {
        uint16_t ctx = train->data[pos - 1u];
        uint16_t target = train->data[pos];
        if (ctx < vocab_size) {
            if (residual.context_counts[ctx] == 0) residual.unique_contexts++;
            residual.context_counts[ctx]++;
        }
        if (!pair_table_add(&residual.pairs, pair_key(ctx, target))) {
            residual_free(&residual);
            return 0;
        }
        terms++;
        if (checkpoint_every > 0 && terms % checkpoint_every == 0) {
            double now = now_seconds();
            double train_seconds = now - start;
            if (!append_eval_checkpoint(checkpoints,
                                        use_scaffold,
                                        eval,
                                        terms,
                                        train_seconds,
                                        eval_terms,
                                        eval_stride,
                                        vocab_size,
                                        alpha,
                                        beta,
                                        unigram_counts,
                                        unigram_total,
                                        scaffold_context_counts,
                                        scaffold_pairs,
                                        &residual)) {
                residual_free(&residual);
                return 0;
            }
        }
    }
    if (checkpoints->items[checkpoints->count - 1u].terms != terms) {
        double now = now_seconds();
        double train_seconds = now - start;
        if (!append_eval_checkpoint(checkpoints,
                                    use_scaffold,
                                    eval,
                                    terms,
                                    train_seconds,
                                    eval_terms,
                                    eval_stride,
                                    vocab_size,
                                    alpha,
                                    beta,
                                    unigram_counts,
                                    unigram_total,
                                    scaffold_context_counts,
                                    scaffold_pairs,
                                    &residual)) {
            residual_free(&residual);
            return 0;
        }
    }
    residual_free(&residual);
    return 1;
}

static uint64_t first_reach(const CheckpointList *checkpoints, double target_ce) {
    for (uint64_t i = 0; i < checkpoints->count; i++) {
        if (checkpoints->items[i].ce <= target_ce) return i;
    }
    return UINT64_MAX;
}

static void print_checkpoint_json(const Checkpoint *cp) {
    printf("{\"terms\":%" PRIu64 ",\"ce\":%.12f,\"train_seconds\":%.6f,"
           "\"eval_seconds\":%.6f,\"residual_contexts\":%" PRIu64 ","
           "\"residual_pairs\":%" PRIu64 "}",
           cp->terms,
           cp->ce,
           cp->train_seconds,
           cp->eval_seconds,
           cp->residual_contexts,
           cp->residual_pairs);
}

static void print_checkpoint_list_json(const CheckpointList *list) {
    printf("[");
    for (uint64_t i = 0; i < list->count; i++) {
        if (i > 0) printf(",");
        print_checkpoint_json(&list->items[i]);
    }
    printf("]");
}

int main(int argc, char **argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 2;
    }

    const char *train_path = argv[1];
    const char *eval_path = argv[2];
    uint64_t vocab_size = 32064;
    double alpha = 100.0;
    double beta = 1000.0;
    uint64_t scaffold_terms = 5000000;
    uint64_t scaffold_stride = 22;
    uint64_t residual_terms = 200000;
    uint64_t residual_stride = 233;
    uint64_t checkpoint_every = 25000;
    uint64_t eval_terms = 20000;
    uint64_t eval_stride = 1;
    uint64_t scaffold_reuse = 1;
    double gamma_ref = 0.535298658010;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--vocab-size") == 0 && i + 1 < argc) {
            vocab_size = parse_u64_arg(argv[++i], "vocab_size");
        } else if (strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) {
            alpha = parse_double_arg(argv[++i], "alpha");
        } else if (strcmp(argv[i], "--beta") == 0 && i + 1 < argc) {
            beta = parse_double_arg(argv[++i], "beta");
        } else if (strcmp(argv[i], "--scaffold-terms") == 0 && i + 1 < argc) {
            scaffold_terms = parse_u64_arg(argv[++i], "scaffold_terms");
        } else if (strcmp(argv[i], "--scaffold-stride") == 0 && i + 1 < argc) {
            scaffold_stride = parse_u64_arg(argv[++i], "scaffold_stride");
        } else if (strcmp(argv[i], "--residual-terms") == 0 && i + 1 < argc) {
            residual_terms = parse_u64_arg(argv[++i], "residual_terms");
        } else if (strcmp(argv[i], "--residual-stride") == 0 && i + 1 < argc) {
            residual_stride = parse_u64_arg(argv[++i], "residual_stride");
        } else if (strcmp(argv[i], "--checkpoint-every") == 0 && i + 1 < argc) {
            checkpoint_every = parse_u64_arg(argv[++i], "checkpoint_every");
        } else if (strcmp(argv[i], "--eval-terms") == 0 && i + 1 < argc) {
            eval_terms = parse_u64_arg(argv[++i], "eval_terms");
        } else if (strcmp(argv[i], "--eval-stride") == 0 && i + 1 < argc) {
            eval_stride = parse_u64_arg(argv[++i], "eval_stride");
        } else if (strcmp(argv[i], "--scaffold-reuse") == 0 && i + 1 < argc) {
            scaffold_reuse = parse_u64_arg(argv[++i], "scaffold_reuse");
        } else if (strcmp(argv[i], "--gamma") == 0 && i + 1 < argc) {
            gamma_ref = parse_double_arg(argv[++i], "gamma");
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (vocab_size == 0 || vocab_size > 65536u || alpha <= 0.0 || beta <= 0.0 ||
        scaffold_stride == 0 || residual_stride == 0 || eval_stride == 0 ||
        scaffold_reuse == 0) {
        fprintf(stderr, "[R408_SCAFFOLD_RESIDUAL_PILOT] invalid arguments\n");
        return 2;
    }

    TokenView train;
    TokenView eval;
    if (!load_token_view(train_path, &train)) return 1;
    if (!load_token_view(eval_path, &eval)) {
        free_token_view(&train);
        return 1;
    }

    uint64_t *unigram_counts = (uint64_t *)calloc((size_t)vocab_size, sizeof(uint64_t));
    uint64_t *scaffold_context_counts = (uint64_t *)calloc((size_t)vocab_size, sizeof(uint64_t));
    if (unigram_counts == NULL || scaffold_context_counts == NULL) {
        fprintf(stderr, "[R408_SCAFFOLD_RESIDUAL_PILOT] OOM counts\n");
        free(unigram_counts);
        free(scaffold_context_counts);
        free_token_view(&eval);
        free_token_view(&train);
        return 1;
    }

    PairTable64 scaffold_pairs;
    if (!pair_table_init(&scaffold_pairs, scaffold_terms)) {
        free(unigram_counts);
        free(scaffold_context_counts);
        free_token_view(&eval);
        free_token_view(&train);
        return 1;
    }

    double unigram_start = now_seconds();
    uint64_t unigram_total = 0;
    for (uint64_t pos = 1; pos < (uint64_t)train.n_tokens && unigram_total < scaffold_terms; pos += scaffold_stride) {
        uint16_t target = train.data[pos];
        if (target < vocab_size) {
            unigram_counts[target]++;
            unigram_total++;
        }
    }
    double unigram_end = now_seconds();
    double unigram_seconds = unigram_end - unigram_start;

    double scaffold_start = now_seconds();
    uint64_t actual_scaffold_terms = 0;
    uint64_t unique_scaffold_contexts = 0;
    for (uint64_t pos = 1; pos < (uint64_t)train.n_tokens && actual_scaffold_terms < scaffold_terms; pos += scaffold_stride) {
        uint16_t ctx = train.data[pos - 1u];
        uint16_t target = train.data[pos];
        if (ctx < vocab_size) {
            if (scaffold_context_counts[ctx] == 0) unique_scaffold_contexts++;
            scaffold_context_counts[ctx]++;
        }
        if (!pair_table_add(&scaffold_pairs, pair_key(ctx, target))) {
            pair_table_free(&scaffold_pairs);
            free(unigram_counts);
            free(scaffold_context_counts);
            free_token_view(&eval);
            free_token_view(&train);
            return 1;
        }
        actual_scaffold_terms++;
    }
    double scaffold_end = now_seconds();
    double scaffold_seconds = scaffold_end - scaffold_start;

    CheckpointList baseline;
    CheckpointList scaffold;
    if (!checkpoints_init(&baseline, residual_terms, checkpoint_every) ||
        !checkpoints_init(&scaffold, residual_terms, checkpoint_every)) {
        fprintf(stderr, "[R408_SCAFFOLD_RESIDUAL_PILOT] OOM checkpoints\n");
        checkpoints_free(&baseline);
        checkpoints_free(&scaffold);
        pair_table_free(&scaffold_pairs);
        free(unigram_counts);
        free(scaffold_context_counts);
        free_token_view(&eval);
        free_token_view(&train);
        return 1;
    }

    if (!train_arm(0,
                   &train,
                   &eval,
                   residual_terms,
                   residual_stride,
                   checkpoint_every,
                   eval_terms,
                   eval_stride,
                   vocab_size,
                   alpha,
                   beta,
                   unigram_counts,
                   unigram_total,
                   scaffold_context_counts,
                   &scaffold_pairs,
                   &baseline) ||
        !train_arm(1,
                   &train,
                   &eval,
                   residual_terms,
                   residual_stride,
                   checkpoint_every,
                   eval_terms,
                   eval_stride,
                   vocab_size,
                   alpha,
                   beta,
                   unigram_counts,
                   unigram_total,
                   scaffold_context_counts,
                   &scaffold_pairs,
                   &scaffold)) {
        fprintf(stderr, "[R408_SCAFFOLD_RESIDUAL_PILOT] train arm failed\n");
        checkpoints_free(&baseline);
        checkpoints_free(&scaffold);
        pair_table_free(&scaffold_pairs);
        free(unigram_counts);
        free(scaffold_context_counts);
        free_token_view(&eval);
        free_token_view(&train);
        return 1;
    }

    double target_ce = baseline.items[baseline.count - 1u].ce;
    uint64_t baseline_idx = first_reach(&baseline, target_ce);
    if (baseline_idx == UINT64_MAX) baseline_idx = baseline.count - 1u;
    uint64_t scaffold_idx = first_reach(&scaffold, target_ce);

    Checkpoint *baseline_reach = &baseline.items[baseline_idx];
    Checkpoint *scaffold_reach = scaffold_idx != UINT64_MAX ? &scaffold.items[scaffold_idx] : NULL;
    double baseline_path_seconds = unigram_seconds + baseline_reach->train_seconds;
    double scaffold_runtime_seconds = unigram_seconds;
    if (scaffold_reach != NULL) scaffold_runtime_seconds += scaffold_reach->train_seconds;
    else scaffold_runtime_seconds += scaffold.items[scaffold.count - 1u].train_seconds;
    double scaffold_path_seconds = scaffold_runtime_seconds + scaffold_seconds;
    double scaffold_amortized_path_seconds = scaffold_runtime_seconds + scaffold_seconds / (double)scaffold_reuse;
    double scaffold_no_build_path_seconds = scaffold_runtime_seconds;

    double after_ratio = scaffold_reach != NULL && baseline_path_seconds > 0.0
                             ? scaffold_path_seconds / baseline_path_seconds
                             : INFINITY;
    double amortized_after_ratio = scaffold_reach != NULL && baseline_path_seconds > 0.0
                                       ? scaffold_amortized_path_seconds / baseline_path_seconds
                                       : INFINITY;
    double no_build_after_ratio = scaffold_reach != NULL && baseline_path_seconds > 0.0
                                      ? scaffold_no_build_path_seconds / baseline_path_seconds
                                      : INFINITY;
    double observed_speedup = isfinite(after_ratio) && after_ratio > 0.0 ? 1.0 / after_ratio : 0.0;
    double amortized_observed_speedup = isfinite(amortized_after_ratio) && amortized_after_ratio > 0.0
                                            ? 1.0 / amortized_after_ratio
                                            : 0.0;
    double no_build_observed_speedup = isfinite(no_build_after_ratio) && no_build_after_ratio > 0.0
                                           ? 1.0 / no_build_after_ratio
                                           : 0.0;
    double extra_build_ratio = baseline_path_seconds > 0.0 ? scaffold_seconds / baseline_path_seconds : INFINITY;
    double kappa_est = isfinite(after_ratio) && gamma_ref < 1.0
                           ? fmax(0.0, (after_ratio - extra_build_ratio) / (1.0 - gamma_ref))
                           : INFINITY;
    double material_reuse_den = 0.90 * baseline_path_seconds - scaffold_runtime_seconds;
    double headline_reuse_den = 0.50 * baseline_path_seconds - scaffold_runtime_seconds;
    double material_break_even_reuse = material_reuse_den > 0.0 ? ceil(scaffold_seconds / material_reuse_den) : INFINITY;
    double headline_break_even_reuse = headline_reuse_den > 0.0 ? ceil(scaffold_seconds / headline_reuse_den) : INFINITY;
    const char *decision = "FAILS_TARGET_REACH";
    if (scaffold_reach != NULL) {
        if (after_ratio <= 0.50) decision = "PASSES_2X_HEADLINE_RESIDUAL_PILOT";
        else if (after_ratio <= 0.90) decision = "PASSES_MATERIAL_RESIDUAL_PILOT";
        else decision = "FAILS_MATERIAL_RESIDUAL_PILOT";
    }

    size_t allocated_bytes = 0;
    allocated_bytes += (size_t)vocab_size * sizeof(uint64_t) * 2u;
    allocated_bytes += (size_t)scaffold_pairs.capacity * sizeof(PairEntry64);
    allocated_bytes += (size_t)next_power_of_two(residual_terms * 2u + 1u) * sizeof(PairEntry64);
    allocated_bytes += (size_t)vocab_size * sizeof(uint64_t);

    printf("[R408_SCAFFOLD_RESIDUAL_PILOT] active=1 source=native-count-residual-original-ce "
           "train=%s eval=%s order=1 residual_freedom=asymptotic_bigram_count_residual\n",
           train_path,
           eval_path);
    printf("{\"decision\":\"%s\",", decision);
    printf("\"target_policy\":\"baseline_final_ce\",");
    printf("\"target_ce\":%.12f,", target_ce);
    printf("\"baseline_path_seconds_to_target\":%.6f,", baseline_path_seconds);
    printf("\"scaffold_path_seconds_to_target\":%.6f,", scaffold_path_seconds);
    printf("\"scaffold_runtime_seconds_to_target\":%.6f,", scaffold_runtime_seconds);
    printf("\"scaffold_no_build_path_seconds_to_target\":%.6f,", scaffold_no_build_path_seconds);
    printf("\"scaffold_reuse\":%" PRIu64 ",", scaffold_reuse);
    printf("\"scaffold_amortized_path_seconds_to_target\":%.6f,", scaffold_amortized_path_seconds);
    printf("\"observed_speedup_to_same_ce\":%.12f,", observed_speedup);
    printf("\"amortized_observed_speedup_to_same_ce\":%.12f,", amortized_observed_speedup);
    printf("\"no_build_observed_speedup_to_same_ce\":%.12f,", no_build_observed_speedup);
    printf("\"after_over_before_time_ratio\":%.12f,", after_ratio);
    printf("\"amortized_after_over_before_time_ratio\":%.12f,", amortized_after_ratio);
    printf("\"no_build_after_over_before_time_ratio\":%.12f,", no_build_after_ratio);
    printf("\"material_bar_ratio\":0.90,\"headline_2x_ratio\":0.50,");
    printf("\"material_break_even_scaffold_reuse\":");
    if (isfinite(material_break_even_reuse)) printf("%.0f,", material_break_even_reuse);
    else printf("null,");
    printf("\"headline_2x_break_even_scaffold_reuse\":");
    if (isfinite(headline_break_even_reuse)) printf("%.0f,", headline_break_even_reuse);
    else printf("null,");
    printf("\"unigram_build_seconds\":%.6f,", unigram_seconds);
    printf("\"scaffold_build_seconds\":%.6f,", scaffold_seconds);
    printf("\"extra_build_ratio_vs_baseline_path\":%.12f,", extra_build_ratio);
    printf("\"gamma_reference\":%.12f,", gamma_ref);
    printf("\"kappa_estimate_from_ratio\":%.12f,", kappa_est);
    printf("\"vocab_size\":%" PRIu64 ",\"alpha\":%.6f,\"beta\":%.6f,", vocab_size, alpha, beta);
    printf("\"scaffold_terms\":%" PRIu64 ",\"residual_terms\":%" PRIu64 ",", actual_scaffold_terms, residual_terms);
    printf("\"eval_terms_requested\":%" PRIu64 ",", eval_terms);
    printf("\"unique_scaffold_contexts\":%" PRIu64 ",\"unique_scaffold_pairs\":%" PRIu64 ",",
           unique_scaffold_contexts,
           scaffold_pairs.unique);
    printf("\"allocated_bytes_estimate\":%zu,", allocated_bytes);
    printf("\"baseline_reach\":");
    print_checkpoint_json(baseline_reach);
    printf(",\"scaffold_reach\":");
    if (scaffold_reach != NULL) print_checkpoint_json(scaffold_reach);
    else printf("null");
    printf(",\"baseline_checkpoints\":");
    print_checkpoint_list_json(&baseline);
    printf(",\"scaffold_checkpoints\":");
    print_checkpoint_list_json(&scaffold);
    printf("}\n");

    checkpoints_free(&baseline);
    checkpoints_free(&scaffold);
    pair_table_free(&scaffold_pairs);
    free(unigram_counts);
    free(scaffold_context_counts);
    free_token_view(&eval);
    free_token_view(&train);
    return 0;
}
