#ifdef _WIN32
#  ifndef _WIN32_WINNT
#    define _WIN32_WINNT 0x0600
#  endif
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include "hardware.h"

#ifdef _WIN32
static void detect_cpu_counts(int *cores, int *threads) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    *threads = (int)si.dwNumberOfProcessors;

    DWORD len = 0;
    GetLogicalProcessorInformation(NULL, &len);
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *buf = malloc(len);
    if (!buf) {
        *cores = *threads;
        return;
    }
    if (GetLogicalProcessorInformation(buf, &len)) {
        int n = (int)(len / sizeof(*buf));
        int phys = 0;
        for (int i = 0; i < n; i++) {
            if (buf[i].Relationship == RelationProcessorCore) phys++;
        }
        *cores = phys ? phys : *threads;
    } else {
        *cores = *threads;
    }
    free(buf);
}
#else
static void detect_cpu_counts(int *cores, int *threads) {
    long t = sysconf(_SC_NPROCESSORS_ONLN);
    *threads = (t > 0) ? (int)t : 1;

    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) {
        *cores = *threads;
        return;
    }

    int max_cores = 0, sockets = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        int v;
        if (sscanf(line, "cpu cores : %d", &v) == 1 && v > max_cores) max_cores = v;
        if (sscanf(line, "physical id : %d", &v) == 1 && v + 1 > sockets) sockets = v + 1;
    }
    fclose(f);

    *cores = (max_cores > 0 && sockets > 0) ? max_cores * sockets
           : (max_cores > 0) ? max_cores
           : *threads;
}
#endif

#ifdef _WIN32
static void detect_caches_win(size_t *l1, size_t *l2, size_t *l3) {
    *l1 = *l2 = *l3 = 0;
    DWORD len = 0;
    GetLogicalProcessorInformation(NULL, &len);
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *buf = malloc(len);
    if (!buf) return;
    if (!GetLogicalProcessorInformation(buf, &len)) {
        free(buf);
        return;
    }
    int n = (int)(len / sizeof(*buf));
    for (int i = 0; i < n; i++) {
        if (buf[i].Relationship != RelationCache) continue;
        CACHE_DESCRIPTOR *c = &buf[i].Cache;
        if (c->Type == CacheInstruction) continue;
        size_t sz = c->Size;
        if (c->Level == 1 && sz > *l1) *l1 = sz;
        if (c->Level == 2 && sz > *l2) *l2 = sz;
        if (c->Level == 3 && sz > *l3) *l3 = sz;
    }
    free(buf);
}
#else
static size_t parse_cache_size_str(const char *s) {
    size_t v = 0;
    char unit = 0;
    sscanf(s, "%zu%c", &v, &unit);
    if (unit == 'K' || unit == 'k') v *= 1024;
    if (unit == 'M' || unit == 'm') v *= 1024 * 1024;
    return v;
}

static void detect_caches_linux(size_t *l1, size_t *l2, size_t *l3) {
    *l1 = *l2 = *l3 = 0;
    for (int idx = 0; idx < 8; idx++) {
        char path[128], buf[64];
        int level = 0;
        char type[32] = {0};

        snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu0/cache/index%d/level", idx);
        FILE *f = fopen(path, "r");
        if (!f) break;
        fscanf(f, "%d", &level);
        fclose(f);

        snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu0/cache/index%d/type", idx);
        f = fopen(path, "r");
        if (f) {
            fscanf(f, "%31s", type);
            fclose(f);
        }

        if (strcmp(type, "Instruction") == 0) continue;

        snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu0/cache/index%d/size", idx);
        f = fopen(path, "r");
        if (!f) continue;
        fgets(buf, sizeof(buf), f);
        fclose(f);
        size_t sz = parse_cache_size_str(buf);

        if (level == 1 && sz > *l1) *l1 = sz;
        if (level == 2 && sz > *l2) *l2 = sz;
        if (level == 3 && sz > *l3) *l3 = sz;
    }
}
#endif

static void detect_simd(int *avx2, int *avx512, int *neon) {
#ifdef __AVX2__
    *avx2 = 1;
#else
    *avx2 = 0;
#endif
#ifdef __AVX512F__
    *avx512 = 1;
#else
    *avx512 = 0;
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    *neon = 1;
#else
    *neon = 0;
#endif
}

#ifdef _WIN32
static void detect_ram_win(size_t *total, size_t *free_mem) {
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        *total    = (size_t)ms.ullTotalPhys;
        *free_mem = (size_t)ms.ullAvailPhys;
    } else {
        *total = *free_mem = 0;
    }
}
#else
static void detect_ram_linux(size_t *total, size_t *free_mem) {
    *total = *free_mem = 0;
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return;
    char key[64];
    size_t val;
    char unit[8];
    while (fscanf(f, "%63s %zu %7s\n", key, &val, unit) >= 2) {
        size_t bytes = (strncmp(unit, "kB", 2) == 0) ? val * 1024 : val;
        if (strcmp(key, "MemTotal:") == 0) *total = bytes;
        if (strcmp(key, "MemAvailable:") == 0) *free_mem = bytes;
    }
    fclose(f);
}
#endif

static void detect_gpu_stub(int *has_gpu, char *gpu_name, size_t *vram_total, size_t *vram_free) {
    *has_gpu = 0;
    gpu_name[0] = '\0';
    *vram_total = *vram_free = 0;

    const char *env = getenv("V12_GPU");
    if (env && env[0] == '1') {
        *has_gpu = 1;
        const char *name = getenv("V12_GPU_NAME");
        if (name) snprintf(gpu_name, 64, "%s", name);
        else      snprintf(gpu_name, 64, "unknown");
        const char *vram = getenv("V12_VRAM_MB");
        if (vram) {
            size_t mb = (size_t)atol(vram);
            *vram_total = mb * 1024 * 1024;
            *vram_free = *vram_total;
        }
        return;
    }

#ifdef _WIN32
    HKEY key;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                      "SOFTWARE\\Khronos\\Vulkan\\Drivers",
                      0, KEY_READ, &key) == ERROR_SUCCESS) {
        *has_gpu = 1;
        snprintf(gpu_name, 64, "unknown");
        RegCloseKey(key);
    }
#else
    FILE *f = fopen("/usr/share/vulkan/icd.d", "r");
    if (!f) f = fopen("/etc/vulkan/icd.d", "r");
    if (f) {
        fclose(f);
        *has_gpu = 1;
        snprintf(gpu_name, 64, "unknown");
    }

    if (!*has_gpu) {
        f = fopen("/proc/driver/nvidia/gpus", "r");
        if (f) {
            fclose(f);
            *has_gpu = 1;
            snprintf(gpu_name, 64, "nvidia-unknown");
        }
    }
#endif
}

static int detect_npu_stub(void) {
    const char *env = getenv("V12_NPU");
    if (env && env[0] == '1') return 1;

#ifdef _WIN32
    HKEY key;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                      "SYSTEM\\CurrentControlSet\\Services\\intel_npu",
                      0, KEY_READ, &key) == ERROR_SUCCESS) {
        RegCloseKey(key);
        return 1;
    }
    return 0;
#else
    FILE *f = fopen("/sys/class/accel/accel0", "r");
    if (f) {
        fclose(f);
        return 1;
    }
    return 0;
#endif
}

int hardware_detect(HardwareProfile *out) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));

#ifdef _WIN32
    detect_cpu_counts(&out->cpu_cores, &out->cpu_threads);
    detect_caches_win(&out->cache_l1, &out->cache_l2, &out->cache_l3);
    detect_ram_win(&out->ram_total, &out->ram_free);
#else
    detect_cpu_counts(&out->cpu_cores, &out->cpu_threads);
    detect_caches_linux(&out->cache_l1, &out->cache_l2, &out->cache_l3);
    detect_ram_linux(&out->ram_total, &out->ram_free);
#endif

    detect_simd(&out->has_avx2, &out->has_avx512, &out->has_neon);
    detect_gpu_stub(&out->has_gpu, out->gpu_name, &out->vram_total, &out->vram_free);
    out->has_npu = detect_npu_stub();

    return 0;
}

void hardware_print(const HardwareProfile *hp) {
    printf("[hardware] CPU: %d cores / %d threads\n", hp->cpu_cores, hp->cpu_threads);
    printf("[hardware] Cache: L1=%zuKB L2=%zuKB L3=%zuMB\n",
           hp->cache_l1 / 1024, hp->cache_l2 / 1024, hp->cache_l3 / (1024*1024));
    printf("[hardware] SIMD: AVX2=%d AVX512=%d NEON=%d\n",
           hp->has_avx2, hp->has_avx512, hp->has_neon);
    printf("[hardware] RAM: total=%.1fGB free=%.1fGB\n",
           (double)hp->ram_total / (1024.0*1024*1024),
           (double)hp->ram_free  / (1024.0*1024*1024));
    if (hp->has_gpu)
        printf("[hardware] GPU: %s VRAM=%.1fGB free=%.1fGB\n",
               hp->gpu_name,
               (double)hp->vram_total / (1024.0*1024*1024),
               (double)hp->vram_free  / (1024.0*1024*1024));
    else
        printf("[hardware] GPU: none detected (stub — Phase 4 will use Vulkan)\n");
    printf("[hardware] NPU: %s\n", hp->has_npu ? "present (inference-only)" : "none");
}
