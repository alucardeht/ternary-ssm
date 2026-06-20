#ifndef DATASET_V14_H
#define DATASET_V14_H

#include <stdint.h>
#include <stddef.h>

/* The Genesis of the Continuous Dataloader */
#define AUDIO_SAMPLES_PER_STEP 256 /* How many audio frames correlate to one logic step */

typedef struct {
    uint16_t *tokens;
    float    *audio_samples; /* Raw WAV samples mapped to phase */
    size_t   n_tokens;
    size_t   n_audio_samples;
    int      is_mmap;
    int      fd_tokens;
    int      fd_audio;
    size_t   mmap_size_tokens;
    size_t   mmap_size_audio;
    uint64_t rng[4];
    uint64_t cursor; /* Shared timeline cursor */
} MultimodalDataset;

typedef struct {
    uint16_t *logic_batch; /* [batch_size * seq_len] */
    float    *audio_batch; /* [batch_size * seq_len * AUDIO_SAMPLES_PER_STEP] */
} SuperBatch;

MultimodalDataset *dataset_load_multimodal(const char *text_path, const char *audio_path);

void dataset_sample_superbatch(MultimodalDataset *ds,
                               SuperBatch *sb,
                               int batch_size,
                               int seq_len);

void dataset_free_multimodal(MultimodalDataset *ds);

#endif
