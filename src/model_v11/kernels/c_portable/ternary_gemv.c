#include "ternary_gemv.h"
#include <string.h>

EXPORT void ternary_gemv_scalar(float* output, const uint8_t* weights_packed,
    const float* activations, const float* scales, const float* bias,
    int M, int K, int group_size)
{
    int K_packed = (K + 3) / 4;
    int num_groups = (K + group_size - 1) / group_size;

    for (int row = 0; row < M; row++) {
        const uint8_t* rw = weights_packed + row * K_packed;
        const float* rs = scales + row * num_groups;

        float group_accs[512];
        memset(group_accs, 0, num_groups * sizeof(float));

        for (int col = 0; col < K; col++) {
            int byte_idx = col / 4;
            int bit_offset = (col % 4) * 2;
            int w_code = (rw[byte_idx] >> bit_offset) & 0x3;

            int group_idx = col / group_size;
            if (w_code == 1) {
                group_accs[group_idx] += activations[col];
            } else if (w_code == 2) {
                group_accs[group_idx] -= activations[col];
            }
        }

        float acc = 0.0f;
        for (int g = 0; g < num_groups; g++) {
            acc += group_accs[g] * rs[g];
        }

        if (bias != NULL) {
            acc += bias[row];
        }

        output[row] = acc;
    }
}

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#ifdef _MSC_VER
#include <intrin.h>
static void do_cpuid(int info[4], int level) {
    __cpuid(info, level);
}
#else
#include <cpuid.h>
static void do_cpuid(int info[4], int level) {
    __cpuid_count(level, 0, info[0], info[1], info[2], info[3]);
}
#endif

EXPORT int has_avx2(void) {
    int info[4];
    do_cpuid(info, 0);
    if (info[0] >= 7) {
        do_cpuid(info, 7);
        return (info[1] & (1 << 5)) != 0;
    }
    return 0;
}

#else

EXPORT int has_avx2(void) {
    return 0;
}

#endif

EXPORT void ternary_gemv(float* output, const uint8_t* weights_packed,
    const float* activations, const float* scales, const float* bias,
    int M, int K, int group_size)
{
    ternary_gemv_scalar(output, weights_packed, activations, scales, bias, M, K, group_size);
}
