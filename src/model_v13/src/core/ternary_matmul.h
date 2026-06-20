/*
 * ternary_matmul.h — Fast multiply-free matrix operations for ternary weights.
 *
 * WHY no multiplications: weights restricted to {-1, 0, +1} reduce the inner loop
 * to addition and subtraction over a sparse gather of input values.
 *
 * This header declares the public API for CSR-based ternary matrix operations.
 * The TernaryCSR struct is defined in types.h; these functions build, use, and destroy it.
 */

#ifndef V12_TERNARY_MATMUL_H
#define V12_TERNARY_MATMUL_H

#include "types.h"   /* TernaryCSR, int32_t, int8_t */
#include <stddef.h>  /* size_t — needed for memset in batch function */

/*
 * ternary_csr_init — Build CSR index arrays from a dense ternary weight matrix.
 * Allocates all internal arrays. Caller must eventually call ternary_csr_free.
 * Returns 0 on success, -1 on allocation failure.
 */
int ternary_csr_init(TernaryCSR *csr, const int8_t *w_dense,
                     int32_t rows, int32_t cols);

/*
 * ternary_matvec — Multiply-free y = W @ x for a single input vector.
 * y[i] = sum(x[pos_cols]) - sum(x[neg_cols]) for row i.
 * Returns 0 on success, -1 if csr is NULL or x/y are NULL.
 */
int ternary_matvec(const TernaryCSR *csr, const float *x, float *y);

/*
 * ternary_matvec_T — Multiply-free y = W^T @ x for a single input vector.
 *
 * Equivalent to: for each nonzero W[row,col] with sign s, y[col] += s * x[row].
 * y must be pre-zeroed by the caller (scatter accumulates into y[col]).
 * Size of x: csr->rows. Size of y: csr->cols.
 *
 * Returns 0 on success, -1 if csr is NULL or x/y are NULL.
 */
int ternary_matvec_T(const TernaryCSR *csr, const float *x, float *y);

/*
 * ternary_matmul_batch — Batched multiply-free Y = W @ X.
 * X layout: column-major in batch dim, X[j * batch + b] = input j for sample b.
 * Y layout: same, Y[i * batch + b] = output i for sample b.
 * Returns 0 on success, -1 if any pointer is NULL or batch <= 0.
 */
int ternary_matmul_batch(const TernaryCSR *csr, const float *X, float *Y,
                         int32_t batch);

/*
 * ternary_csr_free — Free all heap arrays owned by csr and zero the struct.
 * Safe to call on a zero-initialized (never init'd) struct.
 */
void ternary_csr_free(TernaryCSR *csr);

/* CSC = transposed CSR. Used for W^T @ x (backward matvec).
 * Built from CSR at init via ternary_csc_from_csr().
 * Can be uploaded to GPU as VkTernaryCSR with rows/cols swapped. */
typedef struct {
    int32_t *col_ptr_pos;   /* [cols+1] start/end of positive row indices per column */
    int32_t *pos_row_idx;   /* [total_pos] row indices where W[row,col] = +1 */
    int32_t *col_ptr_neg;   /* [cols+1] */
    int32_t *neg_row_idx;   /* [total_neg] row indices where W[row,col] = -1 */
    int32_t rows;           /* original rows (becomes "cols" when used as transposed) */
    int32_t cols;           /* original cols (becomes "rows" when used as transposed) */
    int32_t total_pos;
    int32_t total_neg;
} TernaryCSC;

/* Build CSC from existing CSR. Caller must free with ternary_csc_free(). */
int ternary_csc_from_csr(TernaryCSC *csc, const TernaryCSR *csr);
void ternary_csc_free(TernaryCSC *csc);

/*
 * ternary_dot_row — Compute dot product of a single CSR row with a dense vector.
 *
 * Returns: sum(x[j] for j in pos_cols[row]) - sum(x[j] for j in neg_cols[row])
 * This is exactly logit[row] = lm_head[row, :] @ h for one vocabulary index.
 *
 * Parameters:
 *   csr  — the TernaryCSR matrix (lm_head: rows=vocab_size, cols=d_model)
 *   row  — vocabulary index (0 <= row < csr->rows)
 *   x    — input vector of length csr->cols (norm_out[t])
 *
 * Returns 0.0f if csr or x is NULL.
 */
float ternary_dot_row(const TernaryCSR *csr, int32_t row, const float *x);

#endif /* V12_TERNARY_MATMUL_H */
