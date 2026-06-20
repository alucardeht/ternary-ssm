#ifndef TEACHING_H
#define TEACHING_H

#include <stdint.h>
#include "dataset.h"

#define TEACHING_BUCKET_COUNT 6

typedef struct {
    int      used_replay;
    uint32_t slot;
    uint8_t  bucket;
} TeachingSelection;

typedef struct TeachingMemory TeachingMemory;

TeachingMemory *teaching_create(const char *dataset_path, int seq_len, int capacity, uint64_t seed);
void teaching_free(TeachingMemory *tm);

int teaching_load_tokens(TeachingMemory *tm,
                         Dataset *ds,
                         uint16_t *tokens,
                         uint8_t *boundary_mask,
                         int seq_len,
                         int use_sequential,
                         uint32_t step,
                         int *out_seq_idx,
                         TeachingSelection *selection);

void teaching_observe(TeachingMemory *tm,
                      const uint16_t *tokens,
                      int seq_len,
                      float loss,
                      const TeachingSelection *selection);

void teaching_log_step(TeachingMemory *tm, uint32_t step);

#endif
