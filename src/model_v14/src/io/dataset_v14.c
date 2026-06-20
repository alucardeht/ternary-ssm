#include "dataset_v14.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Phase 2: Authentic Multimodal Dataloader (Memory/Disk Fused) */

MultimodalDataset *dataset_load_multimodal(const char *text_path, const char *audio_path) {
    MultimodalDataset *ds = calloc(1, sizeof(MultimodalDataset));
    
    printf("[v14] Loading Authentic Dataloader...\n");
    printf("  -> Logic Stream: %s\n", text_path);
    printf("  -> Audio Stream: %s\n", audio_path);
    
    /* --- 1. Load Logic (Text Tokens) --- */
    FILE *f_txt = fopen(text_path, "rb");
    if (!f_txt) {
        printf("ERROR: Could not open %s\n", text_path);
        free(ds);
        return NULL;
    }
    fseek(f_txt, 0, SEEK_END);
    long txt_sz = ftell(f_txt);
    fseek(f_txt, 0, SEEK_SET);
    
    ds->n_tokens = txt_sz / sizeof(uint16_t);
    ds->tokens = malloc(txt_sz);
    fread(ds->tokens, sizeof(uint16_t), ds->n_tokens, f_txt);
    fclose(f_txt);
    printf("  -> Logic loaded: %zu tokens (%.2f MB)\n", ds->n_tokens, (double)txt_sz/(1024*1024));
    
    /* --- 2. Load Continuous Audio (WAV float32) --- */
    FILE *f_aud = fopen(audio_path, "rb");
    if (!f_aud) {
        printf("ERROR: Could not open %s\n", audio_path);
        free(ds->tokens);
        free(ds);
        return NULL;
    }
    
    /* Skip 44-byte RIFF/WAV header */
    fseek(f_aud, 44, SEEK_SET);
    
    long current_pos = ftell(f_aud);
    fseek(f_aud, 0, SEEK_END);
    long aud_sz = ftell(f_aud) - 44;
    fseek(f_aud, 44, SEEK_SET);
    
    ds->n_audio_samples = aud_sz / sizeof(float);
    ds->audio_samples = malloc(aud_sz);
    fread(ds->audio_samples, sizeof(float), ds->n_audio_samples, f_aud);
    fclose(f_aud);
    printf("  -> Audio loaded: %zu phase samples (%.2f MB)\n", ds->n_audio_samples, (double)aud_sz/(1024*1024));
    
    /* Validate Alignment */
    size_t required_audio = ds->n_tokens * AUDIO_SAMPLES_PER_STEP;
    if (ds->n_audio_samples < required_audio) {
        printf("  [!] Warning: Audio stream is shorter than logic stream. Truncating logic.\n");
        ds->n_tokens = ds->n_audio_samples / AUDIO_SAMPLES_PER_STEP;
    }
    
    ds->cursor = 0;
    return ds;
}

void dataset_sample_superbatch(MultimodalDataset *ds,
                               SuperBatch *sb,
                               int batch_size,
                               int seq_len) {
    /* Randomly sample a unified block of space-time */
    for (int b = 0; b < batch_size; b++) {
        size_t start_idx = rand() % (ds->n_tokens - seq_len);
        
        /* Copy Logic (Grade 0) */
        for (int s = 0; s < seq_len; s++) {
            sb->logic_batch[b * seq_len + s] = ds->tokens[start_idx + s];
        }
        
        /* Copy Continuous Audio (Spinors), synchronized to logic time */
        size_t audio_start = start_idx * AUDIO_SAMPLES_PER_STEP;
        for (int s = 0; s < seq_len * AUDIO_SAMPLES_PER_STEP; s++) {
            sb->audio_batch[b * seq_len * AUDIO_SAMPLES_PER_STEP + s] = ds->audio_samples[audio_start + s];
        }
    }
}

void dataset_free_multimodal(MultimodalDataset *ds) {
    if (ds) {
        if (ds->tokens) free(ds->tokens);
        if (ds->audio_samples) free(ds->audio_samples);
        free(ds);
    }
}
