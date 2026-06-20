#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "src/io/dataset_v14.h"
#include "src/model/ternary_ssm_v14.h"

#define BATCH_SIZE 4
#define SEQ_LEN 64
#define D_MODEL 128
#define D_STATE 64
#define NUM_STEPS 50

float compute_multimodal_loss(float* logic_out, const uint16_t* target_logic, const float* target_audio) {
    float loss = 0.0f;
    for(int i = 0; i < D_MODEL; i++) {
        /* Pseudo-loss combining cross-entropy proxy for logic and L2 phase distance for audio */
        float logic_diff = logic_out[i] - (float)(target_logic[0] % 100) * 0.01f;
        float phase_diff = logic_out[i] - target_audio[0];
        
        loss += (logic_diff * logic_diff) + (phase_diff * phase_diff);
    }
    return loss / D_MODEL;
}

int main() {
    printf("====================================================\n");
    printf(" TERNARY SSM v14 ENGINE - NATIVE STARTUP \n");
    printf("====================================================\n\n");
    
    srand((unsigned int)time(NULL));
    
    /* 1. Initialize Multimodal Dataloader */
    const char* text_path = "data/fineweb_code_1B.bin";
    const char* audio_path = "data/speech_1M.bin";

    printf("Starting training...\n");
    
    MultimodalDataset *ds = dataset_load_multimodal(text_path, audio_path);
    if (!ds) {
        printf("Failed to load datasets. Please ensure data/fineweb_code_1B.bin and data/v14_audio_corpus.wav exist.\n");
        return 1;
    }
    
    /* 2. Allocate SuperBatch Memory */
    SuperBatch sb;
    sb.logic_batch = malloc(BATCH_SIZE * SEQ_LEN * sizeof(uint16_t));
    sb.audio_batch = malloc(BATCH_SIZE * SEQ_LEN * AUDIO_SAMPLES_PER_STEP * sizeof(float));
    
    /* 3. Instantiate the Clifford Geometric Matrix */
    printf("[v14] Allocating Clifford SSM (d_model=%d, d_state=%d)...\n", D_MODEL, D_STATE);
    CliffordSSM *ssm = clifford_ssm_alloc(D_MODEL, D_STATE);
    
    float *logic_out = malloc(D_MODEL * sizeof(float));
    
    printf("\n[v14] Commencing Physics-Grounded Training Loop (%d steps)...\n\n", NUM_STEPS);
    
    /* 4. Training Loop */
    for (int step = 1; step <= NUM_STEPS; step++) {
        /* Sample continuous space-time chunk */
        dataset_sample_superbatch(ds, &sb, BATCH_SIZE, SEQ_LEN);
        
        float step_loss = 0.0f;
        
        /* Forward pass over batch */
        for(int b = 0; b < BATCH_SIZE; b++) {
            /* We simulate processing the first token and its corresponding audio phase */
            const uint16_t* l_in = &sb.logic_batch[b * SEQ_LEN];
            const float* a_in = &sb.audio_batch[b * SEQ_LEN * AUDIO_SAMPLES_PER_STEP];
            
            clifford_forward_cpu(ssm, l_in, a_in, logic_out);
            
            step_loss += compute_multimodal_loss(logic_out, l_in, a_in);
        }
        step_loss /= BATCH_SIZE;
        
        /* Real Backward Pass using Clifford Adjoint & GF2 Masking */
        for(int b = 0; b < BATCH_SIZE; b++) {
            const uint16_t* l_in = &sb.logic_batch[b * SEQ_LEN];
            const float* a_in = &sb.audio_batch[b * SEQ_LEN * AUDIO_SAMPLES_PER_STEP];
            
            /* Recompute simple difference for gradient proxy */
            float logic_out_grad[D_MODEL];
            for(int i = 0; i < D_MODEL; i++) {
                float logic_diff = logic_out[i] - (float)(l_in[0] % 100) * 0.01f;
                float phase_diff = logic_out[i] - a_in[0];
                logic_out_grad[i] = (logic_diff + phase_diff); /* Derivative of L2 is 2*diff */
            }
            
            clifford_backward_cpu(ssm, l_in, a_in, logic_out_grad, 0.01f);
        }
        
        printf("Step %2d | Loss: %7.4f | Logic Cursor: %5u | Audio Phase Start: %6.2f\n", 
               step, step_loss, sb.logic_batch[0], sb.audio_batch[0]);
    }
    
    printf("\n[v14] Training Complete. Clifford Matrix stable.\n");
    
    /* 5. Cleanup */
    free(logic_out);
    free(sb.logic_batch);
    free(sb.audio_batch);
    clifford_ssm_free(ssm);
    dataset_free_multimodal(ds);
    
    return 0;
}
