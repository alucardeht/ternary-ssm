#include "ternary_types.h"
#include <stdlib.h>
#include <string.h>

ternary_matrix_t* ternary_matrix_alloc(int32_t rows, int32_t cols) {
    ternary_matrix_t* mat = (ternary_matrix_t*)malloc(sizeof(ternary_matrix_t));
    if (!mat) return NULL;

    mat->rows = rows;
    mat->cols = cols;
    mat->packed_size = ((size_t)rows * cols + 3) / 4;

    mat->data = (uint8_t*)calloc(mat->packed_size, 1);
    mat->scales = (float*)calloc(rows, sizeof(float));

    if (!mat->data || !mat->scales) {
        free(mat->data);
        free(mat->scales);
        free(mat);
        return NULL;
    }

    /* Default scale = 1.0 */
    for (int32_t i = 0; i < rows; i++) {
        mat->scales[i] = 1.0f;
    }

    return mat;
}

void ternary_matrix_free(ternary_matrix_t* mat) {
    if (!mat) return;
    free(mat->data);
    free(mat->scales);
    free(mat);
}

/* Pack a dense int8 matrix (values in {-1, 0, +1}) into 2-bit format */
void ternary_matrix_pack(ternary_matrix_t* mat, const int8_t* dense) {
    size_t total = (size_t)mat->rows * mat->cols;
    size_t full_groups = total / 4;
    size_t remainder = total % 4;

    for (size_t g = 0; g < full_groups; g++) {
        size_t base = g * 4;
        mat->data[g] = ternary_pack4(
            dense[base + 0], dense[base + 1],
            dense[base + 2], dense[base + 3]
        );
    }

    /* Handle remainder */
    if (remainder > 0) {
        int8_t vals[4] = {0, 0, 0, 0};
        for (size_t i = 0; i < remainder; i++) {
            vals[i] = dense[full_groups * 4 + i];
        }
        mat->data[full_groups] = ternary_pack4(vals[0], vals[1], vals[2], vals[3]);
    }
}

/* Unpack 2-bit format back to dense int8 matrix */
void ternary_matrix_unpack(const ternary_matrix_t* mat, int8_t* dense) {
    size_t total = (size_t)mat->rows * mat->cols;

    for (size_t i = 0; i < total; i++) {
        size_t byte_idx = i / 4;
        int pos = (int)(i % 4);
        dense[i] = ternary_decode(mat->data[byte_idx], pos);
    }
}

/* Get a single weight value */
int8_t ternary_matrix_get(const ternary_matrix_t* mat, int32_t row, int32_t col) {
    size_t idx = (size_t)row * mat->cols + col;
    size_t byte_idx = idx / 4;
    int pos = (int)(idx % 4);
    return ternary_decode(mat->data[byte_idx], pos);
}

/* Set a single weight value */
void ternary_matrix_set(ternary_matrix_t* mat, int32_t row, int32_t col, int8_t value) {
    size_t idx = (size_t)row * mat->cols + col;
    size_t byte_idx = idx / 4;
    int pos = (int)(idx % 4);

    /* Clear the 2 bits at position */
    uint8_t mask = ~(0x03 << (pos * 2));
    mat->data[byte_idx] = (mat->data[byte_idx] & mask) | (ternary_encode(value) << (pos * 2));
}
