#ifndef TERNARY_GEMV_H
#define TERNARY_GEMV_H

#include <stdint.h>

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT __attribute__((visibility("default")))
#endif

EXPORT void ternary_gemv_scalar(float* output, const uint8_t* weights_packed,
    const float* activations, const float* scales, const float* bias,
    int M, int K, int group_size);

EXPORT int has_avx2(void);

EXPORT void ternary_gemv(float* output, const uint8_t* weights_packed,
    const float* activations, const float* scales, const float* bias,
    int M, int K, int group_size);

#endif
