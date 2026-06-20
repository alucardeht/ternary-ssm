#include "wph_gemv.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int8_t digits[WPH_BLOCK_SIZE];
} WPHPattern;

static WPHPattern WPH_LOOKUP[WPH_NUM_PATTERNS];
static int _lookup_initialized = 0;

static void _init_wph_lookup(void) {
    if (_lookup_initialized) return;

    for (int p = 0; p < WPH_NUM_PATTERNS; p++) {
        int temp = p;
        for (int i = 0; i < WPH_BLOCK_SIZE; i++) {
            int digit = temp % 3;
            WPH_LOOKUP[p].digits[i] = (int8_t)(digit - 1);
            temp /= 3;
        }
    }

    _lookup_initialized = 1;
}

EXPORT void wph_encode(uint8_t* pattern_ids, const int8_t* ternary, int M, int K) {
    _init_wph_lookup();

    int num_blocks = (M + WPH_BLOCK_SIZE - 1) / WPH_BLOCK_SIZE;
    int remainder = M % WPH_BLOCK_SIZE;
    if (remainder == 0) remainder = WPH_BLOCK_SIZE;

    for (int col = 0; col < K; col++) {
        for (int b = 0; b < num_blocks; b++) {
            int pattern = 0;
            int power = 1;
            int rows_in_block = (b == num_blocks - 1) ? remainder : WPH_BLOCK_SIZE;

            for (int r = 0; r < rows_in_block; r++) {
                int row = b * WPH_BLOCK_SIZE + r;
                int8_t val = ternary[row * K + col];
                int digit = (int)(val + 1);
                pattern += digit * power;
                power *= 3;
            }

            pattern_ids[col * num_blocks + b] = (uint8_t)pattern;
        }
    }
}

EXPORT void wph_gemv_scalar(float* output, const uint8_t* pattern_ids,
    const float* activations, const float* scales, const float* bias,
    int M, int K, int group_size)
{
    _init_wph_lookup();

    int num_blocks = (M + WPH_BLOCK_SIZE - 1) / WPH_BLOCK_SIZE;
    int num_groups = (K + group_size - 1) / group_size;
    int remainder = M % WPH_BLOCK_SIZE;
    if (remainder == 0) remainder = WPH_BLOCK_SIZE;

    float* pattern_block_sums = (float*)malloc((size_t)num_blocks * WPH_NUM_PATTERNS * num_groups * sizeof(float));
    memset(pattern_block_sums, 0, (size_t)num_blocks * WPH_NUM_PATTERNS * num_groups * sizeof(float));

    float* row_group_accs = (float*)malloc(WPH_BLOCK_SIZE * num_groups * sizeof(float));

    for (int col = 0; col < K; col++) {
        int group_idx = col / group_size;
        float act = activations[col];

        for (int b = 0; b < num_blocks; b++) {
            uint8_t pattern = pattern_ids[col * num_blocks + b];
            pattern_block_sums[b * WPH_NUM_PATTERNS * num_groups + pattern * num_groups + group_idx] += act;
        }
    }

    for (int b = 0; b < num_blocks; b++) {
        memset(row_group_accs, 0, WPH_BLOCK_SIZE * num_groups * sizeof(float));

        int rows_in_block = (b == num_blocks - 1) ? remainder : WPH_BLOCK_SIZE;
        int block_offset = b * WPH_NUM_PATTERNS * num_groups;

        for (int p = 0; p < WPH_NUM_PATTERNS; p++) {
            int pat_offset = block_offset + p * num_groups;

            int has_nonzero = 0;
            for (int g = 0; g < num_groups; g++) {
                if (fabsf(pattern_block_sums[pat_offset + g]) >= 1e-12f) {
                    has_nonzero = 1;
                    break;
                }
            }
            if (!has_nonzero) continue;

            WPHPattern pat = WPH_LOOKUP[p];
            for (int r = 0; r < rows_in_block; r++) {
                for (int g = 0; g < num_groups; g++) {
                    row_group_accs[r * num_groups + g] += pat.digits[r] * pattern_block_sums[pat_offset + g];
                }
            }
        }

        for (int r = 0; r < rows_in_block; r++) {
            int row = b * WPH_BLOCK_SIZE + r;
            float acc = 0.0f;

            for (int g = 0; g < num_groups; g++) {
                acc += row_group_accs[r * num_groups + g] * scales[row * num_groups + g];
            }

            if (bias != NULL) {
                acc += bias[row];
            }

            output[row] = acc;
        }
    }

    free(pattern_block_sums);
    free(row_group_accs);
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

EXPORT int wph_has_avx2(void) {
    int info[4];
    do_cpuid(info, 0);
    if (info[0] >= 7) {
        do_cpuid(info, 7);
        return (info[1] & (1 << 5)) != 0;
    }
    return 0;
}

#else

EXPORT int wph_has_avx2(void) {
    return 0;
}

#endif

EXPORT void wph_gemv(float* output, const uint8_t* pattern_ids,
    const float* activations, const float* scales, const float* bias,
    int M, int K, int group_size)
{
    wph_gemv_scalar(output, pattern_ids, activations, scales, bias, M, K, group_size);
}
