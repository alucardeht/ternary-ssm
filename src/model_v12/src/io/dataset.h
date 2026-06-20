#ifndef DATASET_H
#define DATASET_H

#include <stdint.h>
#include <stddef.h>

#define DATASET_MAGIC    0x544F4B4E
#define DATASET_VERSION  1
#define DATASET_HEADER_BYTES 16

typedef struct {
    uint16_t  *data;
    size_t     n_tokens;
    int        is_mmap;
    int        fd;
    size_t     mmap_size;
    uint64_t   rng[4];
    uint64_t   cursor;      /* sequential read position (token index) */
} Dataset;

Dataset *dataset_load(const char *path);

void dataset_sample_batch(Dataset *ds,
                          uint16_t *batch,
                          int batch_size,
                          int seq_len);

void dataset_next_sequential(Dataset *ds,
                             uint16_t *batch,
                             int batch_size,
                             int seq_len);

void dataset_reset_cursor(Dataset *ds);

void dataset_seed(Dataset *ds, uint64_t seed);

void dataset_free(Dataset *ds);

/* R493-G04 R441 Gate-to-Task Compiler (Aspect A05) */
int dataset_classify_task(const uint16_t *tokens, int seq_len);

#endif
