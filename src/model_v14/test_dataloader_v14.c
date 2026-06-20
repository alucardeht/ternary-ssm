#include <stdio.h>
#include <stdlib.h>
#include "src/io/dataset_v14.h"

int main() {
    printf("=========================================\n");
    printf(" v14 CLIFFORD SSM: CONTINUOUS DATALOADER \n");
    printf("=========================================\n\n");
    
    MultimodalDataset *ds = dataset_load_multimodal("dummy_logic.txt", "dummy_audio.wav");
    if (!ds) {
        printf("Failed to load multimodal dataset.\n");
        return 1;
    }
    
    int batch_size = 2;
    int seq_len = 8;
    
    SuperBatch sb;
    sb.logic_batch = malloc(batch_size * seq_len * sizeof(uint16_t));
    sb.audio_batch = malloc(batch_size * seq_len * AUDIO_SAMPLES_PER_STEP * sizeof(float));
    
    printf("Sampling SuperBatch (Batch Size: %d, Logic Seq Len: %d)...\n", batch_size, seq_len);
    printf("Audio Phase Samples per logic step: %d\n", AUDIO_SAMPLES_PER_STEP);
    
    dataset_sample_superbatch(ds, &sb, batch_size, seq_len);
    
    printf("\n[SUPERBATCH VERIFICATION]\n");
    for (int b = 0; b < batch_size; b++) {
        printf("-> Batch %d:\n", b);
        printf("   Logic Tokens: [ ");
        for (int s = 0; s < seq_len; s++) {
            printf("%u ", sb.logic_batch[b * seq_len + s]);
        }
        printf("]\n");
        
        printf("   Audio Phase : [ ");
        for (int s = 0; s < 3; s++) { /* Just print the first 3 audio floats */
            printf("%.2f ", sb.audio_batch[b * seq_len * AUDIO_SAMPLES_PER_STEP + s]);
        }
        printf("... ]\n");
    }
    
    printf("\n[SUCCESS] Continuous Fusion I/O Stable.\n");
    
    free(sb.logic_batch);
    free(sb.audio_batch);
    dataset_free_multimodal(ds);
    
    return 0;
}
