#include "ternary_ssm_v14.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* Phase 2: Authentic Multimodal CPU Execution */

CliffordSSM* clifford_ssm_alloc(size_t d_model, size_t d_state) {
    CliffordSSM *ssm = calloc(1, sizeof(CliffordSSM));
    ssm->d_model = d_model;
    ssm->d_state = d_state;
    
    ssm->A_scalar = calloc(d_model, sizeof(int8_t));
    ssm->A_bivec  = calloc(d_model * 3, sizeof(int8_t));
    ssm->A_spinor = calloc(d_model, sizeof(int8_t));
    
    ssm->B_tensor = calloc(d_model, sizeof(int8_t));
    ssm->C_tensor = calloc(d_model, sizeof(int8_t));
    
    ssm->h_t = calloc(d_model, sizeof(CliffordState));
    
    /* Phase 4: Apply Hadamard-inspired Orthogonal Ternary Bases (-1, 0, 1) */
    for(size_t i = 0; i < d_model; i++) {
        /* Distribute -1, 0, 1 in a structurally orthogonal pattern */
        ssm->A_scalar[i]   = ((i % 3) == 0) ? 1 : (((i % 3) == 1) ? -1 : 0);
        ssm->A_spinor[i]   = (((i+1) % 3) == 0) ? 1 : ((((i+1) % 3) == 1) ? -1 : 0);
        
        ssm->A_bivec[i*3]   = ((i % 2) == 0) ? 1 : -1;
        ssm->A_bivec[i*3+1] = (((i/2) % 2) == 0) ? 1 : -1;
        ssm->A_bivec[i*3+2] = (((i/4) % 2) == 0) ? 1 : -1;
        
        ssm->h_t[i].spinor.r = 1.0f; /* Base phase */
        ssm->h_t[i].spinor.i = 0.0f;
    }
    
    return ssm;
}

void clifford_ssm_free(CliffordSSM* ssm) {
    if (!ssm) return;
    free(ssm->A_scalar);
    free(ssm->A_bivec);
    free(ssm->A_spinor);
    free(ssm->B_tensor);
    free(ssm->C_tensor);
    free(ssm->h_t);
    free(ssm);
}

void clifford_forward_cpu(CliffordSSM* ssm, 
                          const uint16_t* logic_in, 
                          const float* audio_in, 
                          float* logic_out) {
    
    float total_loss_delta = 0.0f;
    
    for (size_t i = 0; i < ssm->d_model; i++) {
        CliffordState *state = &ssm->h_t[i];
        
        uint16_t logic_val = logic_in[0]; 
        float audio_val = audio_in[0];
        
        /* --- PHASE 1: RAW SPEED (Bitwise Logic & XOR) --- */
        /* Rip out standard FP16 ALU multiplication, use IEEE 754 bitwise manipulation */
        union { float f; uint32_t i; } v_aud, v_xy, v_yz, v_zx, v_log;
        v_aud.f = audio_val;
        v_xy.f = state->rotor.xy;
        v_yz.f = state->rotor.yz;
        v_zx.f = state->rotor.zx;
        v_log.f = (float)logic_val;

        /* 1. Spinor rotation via XOR Phase injection */
        v_aud.i = (v_aud.i ^ ((ssm->A_spinor[i] < 0) ? 0x80000000 : 0)) & ((ssm->A_spinor[i] == 0) ? 0 : 0xFFFFFFFF);
        float phase_shift = v_aud.f;
        float cos_p = cosf(phase_shift);
        float sin_p = sinf(phase_shift);
        
        float new_sr = state->spinor.r * cos_p - state->spinor.i * sin_p;
        float new_si = state->spinor.r * sin_p + state->spinor.i * cos_p;
        
        /* 2. Bivector advection (Vision proxy) via Bitwise XOR */
        v_xy.i = (v_xy.i ^ ((ssm->A_bivec[i * 3 + 0] < 0) ? 0x80000000 : 0)) & ((ssm->A_bivec[i * 3 + 0] == 0) ? 0 : 0xFFFFFFFF);
        v_yz.i = (v_yz.i ^ ((ssm->A_bivec[i * 3 + 1] < 0) ? 0x80000000 : 0)) & ((ssm->A_bivec[i * 3 + 1] == 0) ? 0 : 0xFFFFFFFF);
        v_zx.i = (v_zx.i ^ ((ssm->A_bivec[i * 3 + 2] < 0) ? 0x80000000 : 0)) & ((ssm->A_bivec[i * 3 + 2] == 0) ? 0 : 0xFFFFFFFF);
        float adv_xy = v_xy.f;
        float adv_yz = v_yz.f;
        float adv_zx = v_zx.f;
        
        /* 3. Discrete logic updates via Bitwise XOR */
        v_log.i = (v_log.i ^ ((ssm->A_scalar[i] < 0) ? 0x80000000 : 0)) & ((ssm->A_scalar[i] == 0) ? 0 : 0xFFFFFFFF);
        float logic_delta = v_log.f;
        float new_scalar = state->scalar + logic_delta;
        
        /* 4. Commutator Bounding (Isolating modalities) */
        if (fabsf(logic_delta) > 10.0f) {
            adv_xy = -adv_xy; /* Discrete reflection */
        }
        
        /* Writeback */
        state->scalar = new_scalar;
        state->rotor.xy = adv_xy;
        state->rotor.yz = adv_yz;
        state->rotor.zx = adv_zx;
        state->spinor.r = new_sr;
        state->spinor.i = new_si;
        
        /* Phase 4: Dynamic tau (temperature) adjustments for Quality & Signal Retention */
        float bivec_norm = sqrtf(adv_xy*adv_xy + adv_yz*adv_yz + adv_zx*adv_zx) + 1e-6f;
        float tau_dynamic = 1.0f / (1.0f + 0.1f * bivec_norm);
        
        /* Project to logic out */
        logic_out[i] = (new_scalar + new_sr) * tau_dynamic;
    }
}

void clifford_backward_cpu(CliffordSSM* ssm, 
                           const uint16_t* logic_in, 
                           const float* audio_in, 
                           const float* logic_out_grad,
                           float learning_rate) {
    
    for (size_t i = 0; i < ssm->d_model; i++) {
        CliffordState *state = &ssm->h_t[i];
        
        uint16_t logic_val = logic_in[0];
        float audio_val = audio_in[0];
        
        /* Reconstruct dynamic tau */
        float adv_xy = state->rotor.xy * ssm->A_bivec[i];
        float adv_yz = state->rotor.yz * ssm->A_bivec[i];
        float adv_zx = state->rotor.zx * ssm->A_bivec[i];
        float bivec_norm = sqrtf(adv_xy*adv_xy + adv_yz*adv_yz + adv_zx*adv_zx) + 1e-6f;
        float tau_dynamic = 1.0f / (1.0f + 0.1f * bivec_norm);
        
        float grad = logic_out_grad[i] * tau_dynamic;
        
        /* --- 1. Clifford Adjoint Gradient Transport (CAGT) for Spinors (Audio) --- */
        /* Forward was: phase_shift = audio_val * A_spinor[i];
           sr = r*cos - i*sin; si = r*sin + i*cos; out = sr;
           d(out)/d(A_spinor) = d(sr)/d(phase_shift) * audio_val
           d(sr)/d(phase) = -r*sin - i*cos */
        
        /* --- PHASE 2: HYBRID ENGINE (Adjoint XOR Propagator) --- */
        /* Bypass standard multiplication using Adjoint XOR Propagator */
        union { float f; uint32_t i; } v_aud;
        v_aud.f = audio_val;
        v_aud.i = (v_aud.i ^ ((ssm->A_spinor[i] < 0) ? 0x80000000 : 0)) & ((ssm->A_spinor[i] == 0) ? 0 : 0xFFFFFFFF);
        float phase_shift = v_aud.f;

        float cos_p = cosf(phase_shift);
        float sin_p = sinf(phase_shift);
        
        float d_sr_d_phase = -state->spinor.r * sin_p - state->spinor.i * cos_p;
        float grad_spinor_continuous = grad * d_sr_d_phase * audio_val;
        
        /* Update Spinor Ternary weight using a straight-through-estimator boundary */
        if (grad_spinor_continuous > 0.1f) {
            ssm->A_spinor[i] = (ssm->A_spinor[i] > -1) ? ssm->A_spinor[i] - 1 : -1;
        } else if (grad_spinor_continuous < -0.1f) {
            ssm->A_spinor[i] = (ssm->A_spinor[i] < 1) ? ssm->A_spinor[i] + 1 : 1;
        }
        
        /* --- 2. Nilpotent Gradient Masking (GF2 Logic) --- */
        /* Phase 3: Nilpotent Gradient Masking (GF2 Isolation)
           Audio continuous gradients MUST NOT collapse GF2 logic weights. 
           We isolate GF2 backward pass based purely on Grade 0 projection. */
        
        float logic_delta_grad = grad; /* Only the grade 0 scalar piece */
        
        /* Introduce is_gf2 zero-masking for discrete pathways */
        /* Protect pure logical operations from phase contagion */
        int is_gf2 = (fabsf(grad_spinor_continuous) < 0.05f) ? 1 : 0;
        logic_delta_grad *= is_gf2; /* Zero-masking */
        
        if (logic_delta_grad > 0.1f) {
            ssm->A_scalar[i] = (ssm->A_scalar[i] > -1) ? ssm->A_scalar[i] - 1 : -1;
        } else if (logic_delta_grad < -0.1f) {
            ssm->A_scalar[i] = (ssm->A_scalar[i] < 1) ? ssm->A_scalar[i] + 1 : 1;
        }
    }
}
