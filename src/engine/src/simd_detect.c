#include "simd_detect.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef ENGINE_VULKAN
#include "vulkan_backend.h"
#endif

static simd_caps_t g_caps = {0};
static bool g_initialized = false;
static engine_backend_t g_selected_backend = ENGINE_BACKEND_SCALAR;
static bool g_backend_initialized = false;
static bool g_vulkan_available = false;

#if defined(_MSC_VER)
#include <intrin.h>
static void cpuid(int info[4], int leaf) {
    __cpuid(info, leaf);
}
static void cpuidex(int info[4], int leaf, int subleaf) {
    __cpuidex(info, leaf, subleaf);
}
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
static void cpuid(int info[4], int leaf) {
    __cpuid(leaf, info[0], info[1], info[2], info[3]);
}
static void cpuidex(int info[4], int leaf, int subleaf) {
    __cpuid_count(leaf, subleaf, info[0], info[1], info[2], info[3]);
}
#endif

void simd_detect_init(void) {
    if (g_initialized) return;

    memset(&g_caps, 0, sizeof(g_caps));

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    int info[4] = {0};

    /* Check max CPUID leaf */
    cpuid(info, 0);
    int max_leaf = info[0];

    if (max_leaf >= 1) {
        cpuid(info, 1);
        /* ECX bit 23: POPCNT */
        g_caps.has_popcnt = (info[2] & (1 << 23)) != 0;
    }

    if (max_leaf >= 7) {
        cpuidex(info, 7, 0);
        /* EBX bit 5: AVX2 */
        g_caps.has_avx2 = (info[1] & (1 << 5)) != 0;
        /* EBX bit 16: AVX-512F */
        g_caps.has_avx512 = (info[1] & (1 << 16)) != 0;
    }

#elif defined(__aarch64__) || defined(_M_ARM64)
    /* ARM64: NEON is always available */
    g_caps.has_neon = true;
    g_caps.has_popcnt = true; /* ARM64 has CNT instruction */
#endif

    g_initialized = true;

    /* Backend selection: check ENV override first, then auto-select best available */
    const char* backend_env = getenv("ENGINE_BACKEND");
    if (backend_env != NULL) {
        if (strcmp(backend_env, "avx512") == 0) {
            g_selected_backend = ENGINE_BACKEND_AVX512;
        } else if (strcmp(backend_env, "avx2") == 0) {
            g_selected_backend = ENGINE_BACKEND_AVX2;
        } else if (strcmp(backend_env, "neon") == 0) {
            g_selected_backend = ENGINE_BACKEND_NEON;
        } else if (strcmp(backend_env, "wasm") == 0) {
            g_selected_backend = ENGINE_BACKEND_WASM;
        } else if (strcmp(backend_env, "scalar") == 0) {
            g_selected_backend = ENGINE_BACKEND_SCALAR;
        } else {
            fprintf(stderr, "Warning: unknown ENGINE_BACKEND='%s', using auto-select\n", backend_env);
            backend_env = NULL;
        }
    }

    if (backend_env == NULL) {
#ifdef ENGINE_AVX512
        if (g_caps.has_avx512) {
            g_selected_backend = ENGINE_BACKEND_AVX512;
        } else
#endif
#ifdef ENGINE_AVX2
        if (g_caps.has_avx2) {
            g_selected_backend = ENGINE_BACKEND_AVX2;
        } else
#endif
#ifdef ENGINE_NEON
        if (g_caps.has_neon) {
            g_selected_backend = ENGINE_BACKEND_NEON;
        } else
#endif
#ifdef ENGINE_WASM_SIMD
        if (1) {
            g_selected_backend = ENGINE_BACKEND_WASM;
        } else
#endif
        {
            g_selected_backend = ENGINE_BACKEND_SCALAR;
        }
    }

    printf("Selected SIMD backend: %s\n", simd_backend_name(g_selected_backend));

#ifdef ENGINE_VULKAN
    g_vulkan_available = vk_available();
    if (g_vulkan_available) {
        printf("Vulkan compute backend: available\n");
    } else {
        printf("Vulkan compute backend: not available\n");
    }
#endif
}

const simd_caps_t* simd_get_caps(void) {
    if (!g_initialized) simd_detect_init();
    return &g_caps;
}

void simd_print_caps(void) {
    const simd_caps_t* caps = simd_get_caps();
    printf("SIMD Capabilities:\n");
    printf("  AVX2:    %s\n", caps->has_avx2   ? "YES" : "no");
    printf("  AVX-512: %s\n", caps->has_avx512  ? "YES" : "no");
    printf("  NEON:    %s\n", caps->has_neon     ? "YES" : "no");
    printf("  POPCNT:  %s\n", caps->has_popcnt   ? "YES" : "no");
}

engine_backend_t simd_get_backend(void) {
    if (!g_backend_initialized) {
        simd_detect_init();
        g_backend_initialized = true;
    }
    return g_selected_backend;
}

const char* simd_backend_name(engine_backend_t b) {
    switch (b) {
        case ENGINE_BACKEND_SCALAR: return "SCALAR";
        case ENGINE_BACKEND_AVX2:   return "AVX2";
        case ENGINE_BACKEND_AVX512: return "AVX-512";
        case ENGINE_BACKEND_NEON:   return "NEON";
        case ENGINE_BACKEND_WASM:   return "WASM SIMD";
        default: return "UNKNOWN";
    }
}

bool engine_has_vulkan(void) {
    if (!g_initialized) simd_detect_init();
    return g_vulkan_available;
}

