#ifndef V12_HARDWARE_H
#define V12_HARDWARE_H

#include <stddef.h>

typedef struct {
    int cpu_cores;
    int cpu_threads;
    size_t cache_l1;
    size_t cache_l2;
    size_t cache_l3;

    int has_avx2;
    int has_avx512;
    int has_neon;

    size_t ram_total;
    size_t ram_free;

    int has_gpu;
    char gpu_name[64];
    size_t vram_total;
    size_t vram_free;

    int has_npu;
} HardwareProfile;

int hardware_detect(HardwareProfile *out);

void hardware_print(const HardwareProfile *hp);

#endif
