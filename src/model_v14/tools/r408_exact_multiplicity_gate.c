#include "../src/io/dataset.h"

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint64_t hash;
    uint64_t offset;
    uint64_t count;
    uint8_t used;
} TermEntry;

typedef struct {
    TermEntry *entries;
    uint64_t capacity;
    uint64_t unique;
    uint64_t collisions;
    uint64_t exact_hits;
} TermTable;

typedef struct {
    uint16_t *data;
    size_t n_tokens;
    Dataset *dataset_owner;
} TokenView;

static uint64_t fnv1a_update_u16(uint64_t h, uint16_t v) {
    h ^= (uint64_t)(v & 0xffu);
    h *= 1099511628211ull;
    h ^= (uint64_t)((v >> 8) & 0xffu);
    h *= 1099511628211ull;
    return h;
}

static uint64_t term_hash(const uint16_t *data, uint64_t offset, uint64_t context_len) {
    uint64_t h = 1469598103934665603ull;
    h = fnv1a_update_u16(h, (uint16_t)(context_len & 0xffffu));
    h = fnv1a_update_u16(h, (uint16_t)((context_len >> 16) & 0xffffu));
    for (uint64_t i = 0; i <= context_len; i++) {
        h = fnv1a_update_u16(h, data[offset + i]);
    }
    return h;
}

static int term_equal(const uint16_t *data,
                      uint64_t a,
                      uint64_t b,
                      uint64_t context_len) {
    for (uint64_t i = 0; i <= context_len; i++) {
        if (data[a + i] != data[b + i]) return 0;
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

static int table_init(TermTable *table, uint64_t expected_terms) {
    uint64_t capacity = next_power_of_two(expected_terms * 2u + 1u);
    table->entries = (TermEntry *)calloc((size_t)capacity, sizeof(TermEntry));
    if (table->entries == NULL) {
        fprintf(stderr, "[R408_EQ_GATE] OOM allocating table capacity=%" PRIu64 "\n", capacity);
        return 0;
    }
    table->capacity = capacity;
    table->unique = 0;
    table->collisions = 0;
    table->exact_hits = 0;
    return 1;
}

static void table_free(TermTable *table) {
    free(table->entries);
    memset(table, 0, sizeof(*table));
}

static int table_add(TermTable *table,
                     const uint16_t *data,
                     uint64_t offset,
                     uint64_t context_len) {
    uint64_t h = term_hash(data, offset, context_len);
    uint64_t mask = table->capacity - 1u;
    uint64_t slot = h & mask;

    for (uint64_t probe = 0; probe < table->capacity; probe++) {
        TermEntry *entry = &table->entries[slot];
        if (!entry->used) {
            entry->used = 1;
            entry->hash = h;
            entry->offset = offset;
            entry->count = 1;
            table->unique++;
            return 1;
        }
        if (entry->hash == h && term_equal(data, entry->offset, offset, context_len)) {
            entry->count++;
            table->exact_hits++;
            return 1;
        }
        table->collisions++;
        slot = (slot + 1u) & mask;
    }

    fprintf(stderr, "[R408_EQ_GATE] table full\n");
    return 0;
}

static uint64_t parse_u64_arg(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr, "[R408_EQ_GATE] invalid %s: %s\n", name, text);
        exit(2);
    }
    return (uint64_t)value;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s <dataset.bin> <context_len> [--max-terms N] [--stride N] [--raw-u8]\n"
            "\n"
            "Counts exact next-token CE terms: fixed-length context tokens plus\n"
            "the immediately following target token. This is a strict exact\n"
            "identity gate for R408 Exact Multiplicity Quotient.\n",
            argv0);
}

static int load_raw_u8_as_u16(const char *path, TokenView *view) {
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "[R408_EQ_GATE] cannot open raw-u8 file: %s\n", path);
        return 0;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "[R408_EQ_GATE] cannot seek raw-u8 file: %s\n", path);
        fclose(fp);
        return 0;
    }
    long size_long = ftell(fp);
    if (size_long <= 0) {
        fprintf(stderr, "[R408_EQ_GATE] empty or invalid raw-u8 file: %s\n", path);
        fclose(fp);
        return 0;
    }
    rewind(fp);

    size_t n = (size_t)size_long;
    uint8_t *bytes = (uint8_t *)malloc(n);
    uint16_t *tokens = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (bytes == NULL || tokens == NULL) {
        fprintf(stderr, "[R408_EQ_GATE] OOM loading raw-u8 file size=%zu\n", n);
        free(bytes);
        free(tokens);
        fclose(fp);
        return 0;
    }
    size_t got = fread(bytes, 1, n, fp);
    fclose(fp);
    if (got != n) {
        fprintf(stderr, "[R408_EQ_GATE] short raw-u8 read: got=%zu expected=%zu\n", got, n);
        free(bytes);
        free(tokens);
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        tokens[i] = (uint16_t)bytes[i];
    }
    free(bytes);

    view->data = tokens;
    view->n_tokens = n;
    view->dataset_owner = NULL;
    return 1;
}

static int load_token_view(const char *path, int raw_u8, TokenView *view) {
    memset(view, 0, sizeof(*view));
    if (raw_u8) {
        return load_raw_u8_as_u16(path, view);
    }

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

int main(int argc, char **argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 2;
    }

    const char *dataset_path = argv[1];
    uint64_t context_len = parse_u64_arg(argv[2], "context_len");
    uint64_t max_terms = 0;
    uint64_t stride = 1;
    int raw_u8 = 0;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--max-terms") == 0 && i + 1 < argc) {
            max_terms = parse_u64_arg(argv[++i], "max_terms");
        } else if (strcmp(argv[i], "--stride") == 0 && i + 1 < argc) {
            stride = parse_u64_arg(argv[++i], "stride");
        } else if (strcmp(argv[i], "--raw-u8") == 0) {
            raw_u8 = 1;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (context_len == 0 || stride == 0) {
        fprintf(stderr, "[R408_EQ_GATE] context_len and stride must be > 0\n");
        return 2;
    }

    TokenView tokens;
    if (!load_token_view(dataset_path, raw_u8, &tokens)) return 1;

    if ((uint64_t)tokens.n_tokens <= context_len) {
        fprintf(stderr,
                "[R408_EQ_GATE] dataset too small: n_tokens=%zu context_len=%" PRIu64 "\n",
                tokens.n_tokens,
                context_len);
        free_token_view(&tokens);
        return 1;
    }

    uint64_t possible_terms = ((uint64_t)tokens.n_tokens - context_len + stride - 1u) / stride;
    uint64_t terms_to_scan = possible_terms;
    if (max_terms > 0 && max_terms < terms_to_scan) terms_to_scan = max_terms;

    TermTable table;
    if (!table_init(&table, terms_to_scan)) {
        free_token_view(&tokens);
        return 1;
    }

    uint64_t scanned = 0;
    for (uint64_t offset = 0;
         offset + context_len < (uint64_t)tokens.n_tokens && scanned < terms_to_scan;
         offset += stride) {
        if (!table_add(&table, tokens.data, offset, context_len)) {
            table_free(&table);
            free_token_view(&tokens);
            return 1;
        }
        scanned++;
    }

    double unique_ratio = scanned > 0 ? (double)table.unique / (double)scanned : 1.0;
    double duplicate_ratio = 1.0 - unique_ratio;
    double max_h_for_2x = 0.5 - unique_ratio;
    const char *decision = max_h_for_2x >= 0.0 ? "PASSES_DUPLICATE_MASS_PRECONDITION"
                                                : "FAILS_DUPLICATE_MASS_PRECONDITION";

    printf("[R408_EQ_GATE] active=1 source=exact-ce-term-identity dataset=%s context_len=%" PRIu64
           " stride=%" PRIu64 " max_terms=%" PRIu64 " raw_u8=%d\n",
           dataset_path,
           context_len,
           stride,
           max_terms,
           raw_u8);
    printf("{\"decision\":\"%s\","
           "\"n_tokens\":%zu,"
           "\"scanned_terms\":%" PRIu64 ","
           "\"unique_terms\":%" PRIu64 ","
           "\"duplicate_terms\":%" PRIu64 ","
           "\"unique_ratio\":%.12f,"
           "\"duplicate_ratio\":%.12f,"
           "\"max_overhead_h_for_2x\":%.12f,"
           "\"hash_collisions_checked\":%" PRIu64 ","
           "\"exact_hits\":%" PRIu64 ","
           "\"raw_u8\":%d}\n",
           decision,
           tokens.n_tokens,
           scanned,
           table.unique,
           scanned - table.unique,
           unique_ratio,
           duplicate_ratio,
           max_h_for_2x,
           table.collisions,
           table.exact_hits,
           raw_u8);

    table_free(&table);
    free_token_view(&tokens);
    return 0;
}
