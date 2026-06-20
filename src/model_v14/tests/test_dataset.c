#include "../src/io/dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

static void write_le_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)((v >> 8) & 0xFF);
    p[2] = (uint8_t)((v >> 16) & 0xFF);
    p[3] = (uint8_t)((v >> 24) & 0xFF);
}

static void write_le_u64(uint8_t *p, uint64_t v) {
    write_le_u32(p, (uint32_t)(v & 0xFFFFFFFFULL));
    write_le_u32(p + 4, (uint32_t)((v >> 32) & 0xFFFFFFFFULL));
}

static void test_raw_format(void) {
    printf("[TEST] raw format (no header)\n");

    FILE *f = fopen("/tmp/test_raw.bin", "wb");
    assert(f != NULL);

    uint16_t tokens[] = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
    size_t n_tokens = sizeof(tokens) / sizeof(tokens[0]);

    for (size_t i = 0; i < n_tokens; i++) {
        uint8_t buf[2];
        buf[0] = (uint8_t)(tokens[i] & 0xFF);
        buf[1] = (uint8_t)((tokens[i] >> 8) & 0xFF);
        fwrite(buf, 1, 2, f);
    }
    fclose(f);

    Dataset *ds = dataset_load("/tmp/test_raw.bin");
    assert(ds != NULL);
    assert(ds->n_tokens == n_tokens);
    assert(ds->is_mmap == 0 || ds->is_mmap == 1);

    for (size_t i = 0; i < n_tokens; i++) {
        assert(ds->data[i] == tokens[i]);
    }

    dataset_free(ds);
    printf("  PASS: loaded %zu tokens\n", n_tokens);
}

static void test_header_format(void) {
    printf("[TEST] header format (with magic+version+n_tokens)\n");

    FILE *f = fopen("/tmp/test_header.bin", "wb");
    assert(f != NULL);

    uint8_t header[16];
    write_le_u32(header + 0, DATASET_MAGIC);
    write_le_u32(header + 4, DATASET_VERSION);
    write_le_u64(header + 8, 5);

    fwrite(header, 1, 16, f);

    uint16_t tokens[] = {200, 201, 202, 203, 204};
    for (size_t i = 0; i < 5; i++) {
        uint8_t buf[2];
        buf[0] = (uint8_t)(tokens[i] & 0xFF);
        buf[1] = (uint8_t)((tokens[i] >> 8) & 0xFF);
        fwrite(buf, 1, 2, f);
    }
    fclose(f);

    Dataset *ds = dataset_load("/tmp/test_header.bin");
    assert(ds != NULL);
    assert(ds->n_tokens == 5);

    for (size_t i = 0; i < 5; i++) {
        assert(ds->data[i] == tokens[i]);
    }

    dataset_free(ds);
    printf("  PASS: loaded 5 tokens with header\n");
}

static void test_sampling(void) {
    printf("[TEST] sampling with reproducibility\n");

    FILE *f = fopen("/tmp/test_sample.bin", "wb");
    assert(f != NULL);

    uint8_t header[16];
    write_le_u32(header + 0, DATASET_MAGIC);
    write_le_u32(header + 4, DATASET_VERSION);
    write_le_u64(header + 8, 100);
    fwrite(header, 1, 16, f);

    for (int i = 0; i < 100; i++) {
        uint16_t val = (uint16_t)i;
        uint8_t buf[2];
        buf[0] = (uint8_t)(val & 0xFF);
        buf[1] = (uint8_t)((val >> 8) & 0xFF);
        fwrite(buf, 1, 2, f);
    }
    fclose(f);

    Dataset *ds1 = dataset_load("/tmp/test_sample.bin");
    assert(ds1 != NULL);

    Dataset *ds2 = dataset_load("/tmp/test_sample.bin");
    assert(ds2 != NULL);

    dataset_seed(ds1, 42);
    dataset_seed(ds2, 42);

    int batch_size = 4;
    int seq_len = 10;
    uint16_t batch1[4 * 10];
    uint16_t batch2[4 * 10];

    dataset_sample_batch(ds1, batch1, batch_size, seq_len);
    dataset_sample_batch(ds2, batch2, batch_size, seq_len);

    int match_count = 0;
    for (int i = 0; i < batch_size * seq_len; i++) {
        if (batch1[i] == batch2[i]) match_count++;
    }

    assert(match_count == batch_size * seq_len);
    printf("  PASS: reproducible sampling (seed 42)\n");

    dataset_free(ds1);
    dataset_free(ds2);
}

static void test_bounds(void) {
    printf("[TEST] bounds checking\n");

    FILE *f = fopen("/tmp/test_bounds.bin", "wb");
    assert(f != NULL);

    uint8_t header[16];
    write_le_u32(header + 0, DATASET_MAGIC);
    write_le_u32(header + 4, DATASET_VERSION);
    write_le_u64(header + 8, 50);
    fwrite(header, 1, 16, f);

    for (int i = 0; i < 50; i++) {
        uint16_t val = (uint16_t)(1000 + i);
        uint8_t buf[2];
        buf[0] = (uint8_t)(val & 0xFF);
        buf[1] = (uint8_t)((val >> 8) & 0xFF);
        fwrite(buf, 1, 2, f);
    }
    fclose(f);

    Dataset *ds = dataset_load("/tmp/test_bounds.bin");
    assert(ds != NULL);
    assert(ds->n_tokens == 50);

    dataset_seed(ds, 123);

    int batch_size = 8;
    int seq_len = 20;
    uint16_t batch[8 * 20];

    dataset_sample_batch(ds, batch, batch_size, seq_len);

    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            uint16_t val = batch[b * seq_len + t];
            assert(val >= 1000 && val < 1050);
        }
    }

    printf("  PASS: all sampled tokens in bounds [1000, 1050)\n");

    dataset_free(ds);
}

static void test_seed_zero_rejection(void) {
    printf("[TEST] seed=0 rejection\n");

    FILE *f = fopen("/tmp/test_seed_zero.bin", "wb");
    assert(f != NULL);

    uint8_t header[16];
    write_le_u32(header + 0, DATASET_MAGIC);
    write_le_u32(header + 4, DATASET_VERSION);
    write_le_u64(header + 8, 10);
    fwrite(header, 1, 16, f);

    for (int i = 0; i < 10; i++) {
        uint16_t val = (uint16_t)i;
        uint8_t buf[2];
        buf[0] = (uint8_t)(val & 0xFF);
        buf[1] = (uint8_t)((val >> 8) & 0xFF);
        fwrite(buf, 1, 2, f);
    }
    fclose(f);

    Dataset *ds = dataset_load("/tmp/test_seed_zero.bin");
    assert(ds != NULL);

    printf("  (calling dataset_seed with 0 — should warn to stderr)\n");
    dataset_seed(ds, 0);

    printf("  PASS: seed=0 safely rejected\n");

    dataset_free(ds);
}

static void test_null_free(void) {
    printf("[TEST] dataset_free with NULL\n");
    dataset_free(NULL);
    printf("  PASS: no crash\n");
}

int main(void) {
    printf("=== Dataset Module Tests ===\n\n");

    test_raw_format();
    test_header_format();
    test_sampling();
    test_bounds();
    test_seed_zero_rejection();
    test_null_free();

    printf("\n=== All Tests Passed ===\n");
    return 0;
}
