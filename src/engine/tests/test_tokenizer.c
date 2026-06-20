#include "tokenizer.h"
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

int tests_passed = 0;
int tests_total = 0;

static int create_test_tvoc(const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return TVOC_ERR_IO;

    uint32_t magic = TVOC_MAGIC;
    uint32_t version = TVOC_VERSION;
    uint32_t flags = 0;
    uint32_t vocab_size = 16;
    int32_t pad_id = 0;
    int32_t unk_id = 1;
    int32_t bos_id = 2;
    int32_t eos_id = 3;
    int32_t n_merges = 6;

    const char* test_vocab[] = {
        "<pad>", "<unk>", "<bos>", "<eos>",
        "h", "e", "l", "o",
        "he", "ll", "lo", "hel", "hello",
        " ", "w", "r"
    };

    uint32_t vocab_offsets[16];
    uint32_t total_vocab_size = 0;

    for (int i = 0; i < 16; i++) {
        vocab_offsets[i] = total_vocab_size;
        total_vocab_size += (uint32_t)strlen(test_vocab[i]);
    }

    char* vocab_data = (char*)malloc(total_vocab_size);
    for (int i = 0; i < 16; i++) {
        memcpy(vocab_data + vocab_offsets[i], test_vocab[i], strlen(test_vocab[i]));
    }

    uint8_t header[64];
    memset(header, 0, 64);

    *(uint32_t*)(header + 0)  = magic;
    *(uint32_t*)(header + 4)  = version;
    *(uint32_t*)(header + 8)  = (uint32_t)vocab_size;
    *(uint32_t*)(header + 12) = (uint32_t)n_merges;
    *(int32_t*)(header + 16)  = pad_id;
    *(int32_t*)(header + 20)  = unk_id;
    *(int32_t*)(header + 24)  = bos_id;
    *(int32_t*)(header + 28)  = eos_id;
    *(uint32_t*)(header + 32) = flags;
    *(uint32_t*)(header + 36) = (uint32_t)total_vocab_size;

    fwrite(header, 64, 1, f);

    for (int i = 0; i < 16; i++) {
        fwrite(&vocab_offsets[i], sizeof(uint32_t), 1, f);
    }

    fwrite(vocab_data, 1, total_vocab_size, f);

    uint32_t merge_a[6] = {4, 6, 8, 11, 6, 14};
    uint32_t merge_b[6] = {5, 7, 6, 10, 6, 7};
    uint32_t merge_result[6] = {8, 10, 11, 12, 9, 15};

    for (int i = 0; i < 6; i++) {
        fwrite(&merge_a[i], sizeof(uint32_t), 1, f);
        fwrite(&merge_b[i], sizeof(uint32_t), 1, f);
        fwrite(&merge_result[i], sizeof(uint32_t), 1, f);
    }

    fclose(f);
    free(vocab_data);

    return TVOC_OK;
}

int test_load_free(void) {
    printf("test_load_free...");
    tests_total++;

    const char* test_file = "test_tokenizer_mini.tvoc";
    ASSERT(create_test_tvoc(test_file) == TVOC_OK);

    tokenizer_t* tok = tokenizer_load(test_file);
    ASSERT(tok != NULL);
    ASSERT(tok->vocab_size == 16);
    ASSERT(tok->n_merges == 6);
    ASSERT(tok->pad_id == 0);
    ASSERT(tok->unk_id == 1);
    ASSERT(tok->bos_id == 2);
    ASSERT(tok->eos_id == 3);

    tokenizer_free(tok);
    tokenizer_free(NULL);

    remove(test_file);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_decode_single(void) {
    printf("test_decode_single...");
    tests_total++;

    const char* test_file = "test_tokenizer_decode.tvoc";
    ASSERT(create_test_tvoc(test_file) == TVOC_OK);

    tokenizer_t* tok = tokenizer_load(test_file);
    ASSERT(tok != NULL);

    char buf[256];

    int n = tokenizer_decode_token(tok, 12, buf, sizeof(buf));
    ASSERT(n > 0);
    ASSERT(strcmp(buf, "hello") == 0);

    n = tokenizer_decode_token(tok, 4, buf, sizeof(buf));
    ASSERT(n > 0);
    ASSERT(strcmp(buf, "h") == 0);

    n = tokenizer_decode_token(tok, 0, buf, sizeof(buf));
    ASSERT(n > 0);
    ASSERT(strcmp(buf, "<pad>") == 0);

    tokenizer_free(tok);
    remove(test_file);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_decode_sequence(void) {
    printf("test_decode_sequence...");
    tests_total++;

    const char* test_file = "test_tokenizer_seq.tvoc";
    ASSERT(create_test_tvoc(test_file) == TVOC_OK);

    tokenizer_t* tok = tokenizer_load(test_file);
    ASSERT(tok != NULL);

    int32_t tokens[5] = {4, 5, 6, 6, 7};
    char buf[256];

    int n = tokenizer_decode(tok, tokens, 5, buf, sizeof(buf));
    ASSERT(n > 0);
    ASSERT(strcmp(buf, "hello") == 0);

    tokenizer_free(tok);
    remove(test_file);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_encode_simple(void) {
    printf("test_encode_simple...");
    tests_total++;

    const char* test_file = "test_tokenizer_encode.tvoc";
    ASSERT(create_test_tvoc(test_file) == TVOC_OK);

    tokenizer_t* tok = tokenizer_load(test_file);
    ASSERT(tok != NULL);

    int32_t out[256];
    int n = tokenizer_encode(tok, "hello", out, 256);
    ASSERT(n > 0);
    ASSERT(out[0] == 12);
    ASSERT(n == 1);

    tokenizer_free(tok);
    remove(test_file);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_encode_shorter(void) {
    printf("test_encode_shorter...");
    tests_total++;

    const char* test_file = "test_tokenizer_short.tvoc";
    ASSERT(create_test_tvoc(test_file) == TVOC_OK);

    tokenizer_t* tok = tokenizer_load(test_file);
    ASSERT(tok != NULL);

    int32_t out[256];
    int n = tokenizer_encode(tok, "he", out, 256);
    ASSERT(n > 0);
    ASSERT(out[0] == 8);

    tokenizer_free(tok);
    remove(test_file);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_roundtrip(void) {
    printf("test_roundtrip...");
    tests_total++;

    const char* test_file = "test_tokenizer_roundtrip.tvoc";
    ASSERT(create_test_tvoc(test_file) == TVOC_OK);

    tokenizer_t* tok = tokenizer_load(test_file);
    ASSERT(tok != NULL);

    const char* original = "hello";
    int32_t encoded[256];
    int n_encoded = tokenizer_encode(tok, original, encoded, 256);
    ASSERT(n_encoded > 0);

    char decoded[256];
    int n_decoded = tokenizer_decode(tok, encoded, n_encoded, decoded, sizeof(decoded));
    ASSERT(n_decoded > 0);
    ASSERT(strcmp(decoded, original) == 0);

    tokenizer_free(tok);
    remove(test_file);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_empty(void) {
    printf("test_empty...");
    tests_total++;

    const char* test_file = "test_tokenizer_empty.tvoc";
    ASSERT(create_test_tvoc(test_file) == TVOC_OK);

    tokenizer_t* tok = tokenizer_load(test_file);
    ASSERT(tok != NULL);

    int32_t out[256];
    int n = tokenizer_encode(tok, "", out, 256);
    ASSERT(n == 0);

    char buf[256];
    n = tokenizer_decode(tok, NULL, 0, buf, sizeof(buf));
    ASSERT(n == 0);

    tokenizer_free(tok);
    remove(test_file);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int test_unknown(void) {
    printf("test_unknown...");
    tests_total++;

    const char* test_file = "test_tokenizer_unk.tvoc";
    ASSERT(create_test_tvoc(test_file) == TVOC_OK);

    tokenizer_t* tok = tokenizer_load(test_file);
    ASSERT(tok != NULL);

    int32_t out[256];
    int n = tokenizer_encode(tok, "xyz", out, 256);
    ASSERT(n > 0);
    for (int i = 0; i < n; i++) {
        ASSERT(out[i] >= 0 && out[i] < tok->vocab_size);
    }

    tokenizer_free(tok);
    remove(test_file);

    printf(" PASS\n");
    tests_passed++;
    return 1;
}

int main(void) {
    printf("=== Tokenizer Tests ===\n\n");

    test_load_free();
    test_decode_single();
    test_decode_sequence();
    test_encode_simple();
    test_encode_shorter();
    test_roundtrip();
    test_empty();
    test_unknown();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
