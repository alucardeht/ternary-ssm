#ifndef TERNARY_SSM_V14_H
#define TERNARY_SSM_V14_H

#include <stdint.h>
#include <stddef.h>

/* 
 * The Genesis of v14: Native Multimodal Clifford SSM
 * 
 * Instead of a flat vector array, the state `h_t` is a Geometric Multivector
 * living in Cl_{3,1}(R) or a suitable subalgebra.
 * 
 * Grade 0 (Scalars)    : Logic / Abstract Reasoning
 * Grade 2 (Bivectors)  : Vision / Navier-Stokes Advection
 * Spinors (Phase)      : Audio / Doppler Geometry
 */

typedef struct {
    float r; /* Real part */
    float i; /* Imaginary part */
} ComplexPhase;

typedef struct {
    float xy;
    float yz;
    float zx;
} Bivector3D;

/* The Unified State Multivector for a single feature/channel */
typedef struct {
    float scalar;           /* Logic state (Grade 0) */
    Bivector3D rotor;       /* Vision state (Grade 2) */
    ComplexPhase spinor;    /* Audio state (Phase) */
} CliffordState;

/* 
 * The v14 Model Definition.
 * The transition matrix A is no longer a dense block. 
 * It applies the Geometric Product: A * h = (A_logic + A_vision + A_audio) * h
 */
typedef struct {
    size_t d_model;
    size_t d_state;
    
    /* Weights are natively ternary (-1, 0, 1) */
    int8_t *A_scalar;  /* Logic projections */
    int8_t *A_bivec;   /* Advection rotors */
    int8_t *A_spinor;  /* Phase oscillators */
    
    int8_t *B_tensor;  /* Control tensor projecting SuperBatch inputs into the Clifford Space */
    int8_t *C_tensor;  /* Observation tensor */
    
    /* The active state memory */
    CliffordState *h_t;
} CliffordSSM;

CliffordSSM* clifford_ssm_alloc(size_t d_model, size_t d_state);
void clifford_ssm_free(CliffordSSM* ssm);

/* CPU reference implementation of the Geometric Product forward pass */
void clifford_forward_cpu(CliffordSSM* ssm, 
                          const uint16_t* logic_in, 
                          const float* audio_in, 
                          float* logic_out);

void clifford_backward_cpu(CliffordSSM* ssm, 
                           const uint16_t* logic_in, 
                           const float* audio_in, 
                           const float* logic_out_grad,
                           float learning_rate);

#endif
