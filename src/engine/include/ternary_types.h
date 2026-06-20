#ifndef TERNARY_TYPES_H
#define TERNARY_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/*
 * Ternary weight representation: 2 bits per weight.
 *
 * Encoding:
 *   00 = 0  (zero)
 *   01 = +1 (positive)
 *   10 = -1 (negative)
 *   11 = reserved (unused)
 *
 * Packed format: 4 weights per byte, LSB first.
 *   Byte = [w3:2bit][w2:2bit][w1:2bit][w0:2bit]
 *
 * For a matrix M x N:
 *   packed_size = (M * N + 3) / 4 bytes
 *   Plus per-row scale factor (float32)
 */

/* SIMD backend selection */
typedef enum {
    ENGINE_BACKEND_SCALAR  = 0,
    ENGINE_BACKEND_AVX2    = 1,
    ENGINE_BACKEND_AVX512  = 2,
    ENGINE_BACKEND_NEON    = 3,
    ENGINE_BACKEND_WASM    = 4,
    ENGINE_BACKEND_VULKAN  = 5,
    ENGINE_BACKEND_COUNT   = 6
} engine_backend_t;

/* Packed ternary weight block */
typedef struct {
    uint8_t*  data;        /* Packed 2-bit weights */
    float*    scales;      /* Per-row scale factors */
    int32_t   rows;        /* Number of output features */
    int32_t   cols;        /* Number of input features */
    size_t    packed_size; /* Total bytes in data */
} ternary_matrix_t;

/* Decode a single ternary value from packed byte */
static inline int8_t ternary_decode(uint8_t packed, int pos) {
    uint8_t bits = (packed >> (pos * 2)) & 0x03;
    /* 00 -> 0, 01 -> +1, 10 -> -1, 11 -> 0 (treat reserved as zero) */
    static const int8_t decode_table[4] = {0, 1, -1, 0};
    return decode_table[bits];
}

/* Encode a single ternary value into packed format */
static inline uint8_t ternary_encode(int8_t value) {
    /* 0 -> 00, +1 -> 01, -1 -> 10 */
    if (value == 1)  return 0x01;
    if (value == -1) return 0x02;
    return 0x00;
}

/* Pack 4 ternary values into one byte */
static inline uint8_t ternary_pack4(int8_t w0, int8_t w1, int8_t w2, int8_t w3) {
    return (uint8_t)(
        (ternary_encode(w0) << 0) |
        (ternary_encode(w1) << 2) |
        (ternary_encode(w2) << 4) |
        (ternary_encode(w3) << 6)
    );
}

/* SIMD capability flags (set at runtime) */
typedef struct {
    bool has_avx2;
    bool has_avx512;
    bool has_neon;
    bool has_popcnt;
} simd_caps_t;

/* Matrix allocation and deallocation */
ternary_matrix_t* ternary_matrix_alloc(int32_t rows, int32_t cols);
void ternary_matrix_free(ternary_matrix_t* mat);

/* Pack dense int8 matrix into 2-bit format */
void ternary_matrix_pack(ternary_matrix_t* mat, const int8_t* dense);

/* Unpack 2-bit format back to dense int8 */
void ternary_matrix_unpack(const ternary_matrix_t* mat, int8_t* dense);

/* Get/set individual weight values */
int8_t ternary_matrix_get(const ternary_matrix_t* mat, int32_t row, int32_t col);
void ternary_matrix_set(ternary_matrix_t* mat, int32_t row, int32_t col, int8_t value);

#endif /* TERNARY_TYPES_H */
