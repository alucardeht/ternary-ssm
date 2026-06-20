#ifndef WPH_GEMV_H
#define WPH_GEMV_H

#include <stdint.h>

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT __attribute__((visibility("default")))
#endif

#define WPH_BLOCK_SIZE 4
#define WPH_NUM_PATTERNS 81

EXPORT void wph_encode(uint8_t* pattern_ids, const int8_t* ternary, int M, int K);

EXPORT void wph_gemv_scalar(float* output, const uint8_t* pattern_ids,
    const float* activations, const float* scales, const float* bias,
    int M, int K, int group_size);

EXPORT int wph_has_avx2(void);

EXPORT void wph_gemv(float* output, const uint8_t* pattern_ids,
    const float* activations, const float* scales, const float* bias,
    int M, int K, int group_size);

#endif
