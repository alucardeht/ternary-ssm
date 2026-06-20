#ifndef SIMD_DETECT_H
#define SIMD_DETECT_H

#include "ternary_types.h"

/* Detect SIMD capabilities at runtime.
 * Call once at startup. Results cached internally. */
void simd_detect_init(void);

/* Get detected capabilities */
const simd_caps_t* simd_get_caps(void);

/* Print detected capabilities to stdout */
void simd_print_caps(void);

/* Backend selection */
engine_backend_t simd_get_backend(void);
const char* simd_backend_name(engine_backend_t b);

/* Vulkan availability check */
bool engine_has_vulkan(void);

/* Top-level init — calls simd_detect_init + all module inits */
void engine_init(void);

#endif /* SIMD_DETECT_H */
