/*
 * ternary_matmul.c — Multiply-free matrix operations for ternary {-1, 0, +1} weights.
 *
 * WHY no multiplications: weights are restricted to {-1, 0, +1}, so W*x becomes
 * either +x, -x, or 0. CSR format pre-indexes the nonzero positions by sign,
 * reducing the inner loop to adds and subtracts over a gather of x values.
 *
 * The CSR struct (TernaryCSR) is defined in types.h. This file only provides
 * the functions that build, use, and destroy it.
 *
 * Integration: included by model.c (build at load), forward.c (matvec/batch),
 * and optimizer.c (free at teardown). See spec-ternary-matmul.md §6.
 */

#include "ternary_matmul.h"
#include <stdlib.h>   /* malloc, free */
#include <string.h>   /* memset */
#include <stdio.h>    /* printf */

#ifdef __AVX2__
#include <immintrin.h>

static inline float hadd8(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static int cpu_has_avx2(void) {
    static int cached = -1;
    if (cached >= 0) return cached;
    #ifdef _MSC_VER
    int info[4];
    __cpuidex(info, 7, 0);
    cached = (info[1] >> 5) & 1;
    #else
    unsigned int eax, ebx, ecx, edx;
    __asm__ volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(7), "c"(0));
    cached = (ebx >> 5) & 1;
    #endif
    return cached;
}
#endif /* __AVX2__ */

/*
 * csr_free_partial — Free whichever fields are non-NULL in a partially-initialized CSR.
 * Called only from ternary_csr_init on allocation failure.
 * Mirrors the public ternary_csr_free but safe for incomplete structs.
 */
static void csr_free_partial(TernaryCSR *csr) {
    if (!csr) return;
    free(csr->pos_indices);
    free(csr->neg_indices);
    free(csr->row_ptr_pos);
    free(csr->row_ptr_neg);
    free(csr->n_pos);
    free(csr->n_neg);
    memset(csr, 0, sizeof(TernaryCSR));
}

/*
 * ternary_csr_init — Two-pass CSR construction.
 *
 * Pass 1: Count +1 and -1 per row to compute total_pos/total_neg.
 * Pass 2: Fill index arrays by walking the dense matrix again.
 *
 * After this call, all 8 pointer fields are malloc'd and populated.
 * Columns are stored in ascending order per row (invariant for future updates).
 */
int ternary_csr_init(TernaryCSR *csr, const int8_t *w_dense,
                     int32_t rows, int32_t cols) {
    if (!csr || !w_dense) return -1;

    /* Allocate row-level arrays */
    csr->row_ptr_pos = (int32_t *)malloc((rows + 1) * sizeof(int32_t));
    if (!csr->row_ptr_pos) return -1;

    csr->row_ptr_neg = (int32_t *)malloc((rows + 1) * sizeof(int32_t));
    if (!csr->row_ptr_neg) {
        csr_free_partial(csr);
        return -1;
    }

    csr->n_pos = (int32_t *)malloc(rows * sizeof(int32_t));
    if (!csr->n_pos) {
        csr_free_partial(csr);
        return -1;
    }

    csr->n_neg = (int32_t *)malloc(rows * sizeof(int32_t));
    if (!csr->n_neg) {
        csr_free_partial(csr);
        return -1;
    }

    /* PASS 1: Count +1 and -1 per row */
    for (int32_t i = 0; i < rows; i++) {
        int32_t cnt_p = 0, cnt_n = 0;
        for (int32_t j = 0; j < cols; j++) {
            int8_t v = w_dense[i * cols + j];
            if (v == 1) cnt_p++;
            if (v == -1) cnt_n++;
        }
        csr->n_pos[i] = cnt_p;
        csr->n_neg[i] = cnt_n;
    }

    /* Build prefix-sum row pointers */
    csr->row_ptr_pos[0] = 0;
    for (int32_t i = 0; i < rows; i++) {
        csr->row_ptr_pos[i + 1] = csr->row_ptr_pos[i] + csr->n_pos[i];
    }

    csr->row_ptr_neg[0] = 0;
    for (int32_t i = 0; i < rows; i++) {
        csr->row_ptr_neg[i + 1] = csr->row_ptr_neg[i] + csr->n_neg[i];
    }

    csr->total_pos = csr->row_ptr_pos[rows];
    csr->total_neg = csr->row_ptr_neg[rows];
    csr->cap_pos = csr->total_pos;
    csr->cap_neg = csr->total_neg;

    /* Allocate flat index arrays */
    if (csr->total_pos > 0) {
        csr->pos_indices = (int32_t *)malloc(csr->total_pos * sizeof(int32_t));
        if (!csr->pos_indices) {
            csr_free_partial(csr);
            return -1;
        }
    } else {
        csr->pos_indices = NULL;
    }

    if (csr->total_neg > 0) {
        csr->neg_indices = (int32_t *)malloc(csr->total_neg * sizeof(int32_t));
        if (!csr->neg_indices) {
            csr_free_partial(csr);
            return -1;
        }
    } else {
        csr->neg_indices = NULL;
    }

    /* PASS 2: Fill index arrays with temporary cursors */
    int32_t *cursor_pos = (int32_t *)malloc(rows * sizeof(int32_t));
    if (!cursor_pos && rows > 0) {
        csr_free_partial(csr);
        return -1;
    }

    int32_t *cursor_neg = (int32_t *)malloc(rows * sizeof(int32_t));
    if (!cursor_neg && rows > 0) {
        free(cursor_pos);
        csr_free_partial(csr);
        return -1;
    }

    /* Initialize cursors */
    for (int32_t i = 0; i < rows; i++) {
        cursor_pos[i] = csr->row_ptr_pos[i];
        cursor_neg[i] = csr->row_ptr_neg[i];
    }

    /* Fill indices. Scanning j from 0 to cols-1 ensures each row's index slice
     * is sorted in ascending order — a required invariant for binary search
     * in future ternary_csr_update calls. */
    for (int32_t i = 0; i < rows; i++) {
        for (int32_t j = 0; j < cols; j++) {
            int8_t v = w_dense[i * cols + j];
            if (v == 1) {
                csr->pos_indices[cursor_pos[i]++] = j;
            } else if (v == -1) {
                csr->neg_indices[cursor_neg[i]++] = j;
            }
        }
    }

    free(cursor_pos);
    free(cursor_neg);

    csr->rows = rows;
    csr->cols = cols;
    return 0;
}

#ifdef __AVX2__
static int ternary_matvec_avx2(const TernaryCSR *csr, const float *x, float *y) {
    /* Log once (thread-safe with omp single) */
    #pragma omp single
    {
        static int logged = 0;
        if (!logged) {
            printf("[AVX2] ternary_matvec: rows=%d\n", csr->rows);
            logged = 1;
        }
    }

    /* Parallelize row loop if rows >= 256 */
    if (csr->rows >= 256) {
        #pragma omp parallel for schedule(static)
        for (int32_t i = 0; i < csr->rows; i++) {
            __m256 acc_pos = _mm256_setzero_ps();
            __m256 acc_neg = _mm256_setzero_ps();

            int32_t p_start = csr->row_ptr_pos[i];
            int32_t p_end = csr->row_ptr_pos[i + 1];
            int32_t k;

            for (k = p_start; k + 7 < p_end; k += 8) {
                __m256i idx = _mm256_loadu_si256((const __m256i *)(csr->pos_indices + k));
                __m256 gathered = _mm256_i32gather_ps(x, idx, 4);
                acc_pos = _mm256_add_ps(acc_pos, gathered);
            }

            float tail_pos = 0.0f;
            for (; k < p_end; k++) {
                tail_pos += x[csr->pos_indices[k]];
            }

            int32_t n_start = csr->row_ptr_neg[i];
            int32_t n_end = csr->row_ptr_neg[i + 1];

            for (k = n_start; k + 7 < n_end; k += 8) {
                __m256i idx = _mm256_loadu_si256((const __m256i *)(csr->neg_indices + k));
                __m256 gathered = _mm256_i32gather_ps(x, idx, 4);
                acc_neg = _mm256_add_ps(acc_neg, gathered);
            }

            float tail_neg = 0.0f;
            for (; k < n_end; k++) {
                tail_neg += x[csr->neg_indices[k]];
            }

            y[i] = hadd8(acc_pos) + tail_pos - hadd8(acc_neg) - tail_neg;
        }
    } else {
        /* Small matrices — run sequentially */
        for (int32_t i = 0; i < csr->rows; i++) {
            __m256 acc_pos = _mm256_setzero_ps();
            __m256 acc_neg = _mm256_setzero_ps();

            int32_t p_start = csr->row_ptr_pos[i];
            int32_t p_end = csr->row_ptr_pos[i + 1];
            int32_t k;

            for (k = p_start; k + 7 < p_end; k += 8) {
                __m256i idx = _mm256_loadu_si256((const __m256i *)(csr->pos_indices + k));
                __m256 gathered = _mm256_i32gather_ps(x, idx, 4);
                acc_pos = _mm256_add_ps(acc_pos, gathered);
            }

            float tail_pos = 0.0f;
            for (; k < p_end; k++) {
                tail_pos += x[csr->pos_indices[k]];
            }

            int32_t n_start = csr->row_ptr_neg[i];
            int32_t n_end = csr->row_ptr_neg[i + 1];

            for (k = n_start; k + 7 < n_end; k += 8) {
                __m256i idx = _mm256_loadu_si256((const __m256i *)(csr->neg_indices + k));
                __m256 gathered = _mm256_i32gather_ps(x, idx, 4);
                acc_neg = _mm256_add_ps(acc_neg, gathered);
            }

            float tail_neg = 0.0f;
            for (; k < n_end; k++) {
                tail_neg += x[csr->neg_indices[k]];
            }

            y[i] = hadd8(acc_pos) + tail_pos - hadd8(acc_neg) - tail_neg;
        }
    }

    return 0;
}
#endif /* __AVX2__ */

/*
 * ternary_matvec — Compute y = W @ x using CSR gather-and-accumulate.
 *
 * No multiplications. Inner loop accumulates x values at positive-weight
 * column positions and subtracts x values at negative-weight positions.
 *
 * Dispatches to AVX2 SIMD path if available at runtime, else falls back to scalar.
 */
int ternary_matvec(const TernaryCSR *csr, const float *x, float *y) {
    if (!csr || !x || !y) return -1;

#ifdef __AVX2__
    if (cpu_has_avx2()) {
        return ternary_matvec_avx2(csr, x, y);
    }
#endif

    /* Scalar fallback path — parallelize row loop if rows >= 256 */
    if (csr->rows >= 256) {
        #pragma omp parallel for schedule(static)
        for (int32_t i = 0; i < csr->rows; i++) {
            float acc = 0.0f;

            int32_t p_start = csr->row_ptr_pos[i];
            int32_t p_end = csr->row_ptr_pos[i + 1];

            int32_t n_start = csr->row_ptr_neg[i];
            int32_t n_end = csr->row_ptr_neg[i + 1];

            for (int32_t k = p_start; k < p_end; k++) {
                acc += x[csr->pos_indices[k]];
            }

            for (int32_t k = n_start; k < n_end; k++) {
                acc -= x[csr->neg_indices[k]];
            }

            y[i] = acc;
        }
    } else {
        /* Small matrices — run sequentially to avoid fork overhead */
        for (int32_t i = 0; i < csr->rows; i++) {
            float acc = 0.0f;

            int32_t p_start = csr->row_ptr_pos[i];
            int32_t p_end = csr->row_ptr_pos[i + 1];

            int32_t n_start = csr->row_ptr_neg[i];
            int32_t n_end = csr->row_ptr_neg[i + 1];

            for (int32_t k = p_start; k < p_end; k++) {
                acc += x[csr->pos_indices[k]];
            }

            for (int32_t k = n_start; k < n_end; k++) {
                acc -= x[csr->neg_indices[k]];
            }

            y[i] = acc;
        }
    }

    return 0;
}

/*
 * ternary_matvec_T — Multiply-free y = W^T @ x.
 *
 * WHY scatter instead of gather: the transpose product maps row i of W to
 * column positions in y. Walking rows of the CSR and scattering into y[col]
 * is equivalent to walking columns of W^T and gathering — but requires only
 * the row-oriented CSR that is already built for the forward pass.
 *
 * Caller must pre-zero y[0..csr->cols-1] before calling.
 * No AVX2 path: scatter writes have serial data dependencies (y[col] += ...)
 * that prevent SIMD gather-scatter without conflict resolution.
 */
int ternary_matvec_T(const TernaryCSR *csr, const float *x, float *y) {
    if (!csr || !x || !y) return -1;

    for (int32_t i = 0; i < csr->rows; i++) {
        float xi = x[i];
        if (xi == 0.0f) continue;   /* skip zero-gradient rows */

        int32_t p_start = csr->row_ptr_pos[i];
        int32_t p_end   = csr->row_ptr_pos[i + 1];
        for (int32_t k = p_start; k < p_end; k++) {
            y[csr->pos_indices[k]] += xi;
        }

        int32_t n_start = csr->row_ptr_neg[i];
        int32_t n_end   = csr->row_ptr_neg[i + 1];
        for (int32_t k = n_start; k < n_end; k++) {
            y[csr->neg_indices[k]] -= xi;
        }
    }

    return 0;
}

/*
 * ternary_matmul_batch — Batched multiply-free Y = W @ X.
 *
 * Column-major batch layout: X[j * batch + b] = input j for sample b.
 * This layout ensures that when the inner loop processes column index j,
 * it reads a contiguous slice of batch floats, enabling auto-vectorization.
 *
 * Cache behavior at V12 defaults (rows=128, cols=1024, batch=16):
 * - y_row is 64 bytes (16 floats), fits in registers.
 * - x_col is 64 bytes per index hit (one cache line).
 * - Full X is 64 KB; if > L1 (32 KB), L2 handles with low latency.
 * - The innermost b loop is a contiguous fused-add — the critical vectorizable unit.
 */
int ternary_matmul_batch(const TernaryCSR *csr, const float *X, float *Y,
                         int32_t batch) {
    if (!csr || !X || !Y || batch <= 0) return -1;

    /* Zero the output: we accumulate into Y rather than overwriting per-element,
     * because the accumulation loop has better cache behavior when y_row stays
     * in registers across multiple index hits. */
    memset(Y, 0, (size_t)csr->rows * (size_t)batch * sizeof(float));

    #pragma omp parallel for schedule(static)
    for (int32_t i = 0; i < csr->rows; i++) {
        float *y_row = Y + i * batch;

        /* Positive contributions: for each column j where W[i,j]=+1,
         * add the entire batch slice X[j*batch .. j*batch+batch-1] to y_row.
         * WHY column-major X: X[j*batch] is contiguous over b, so this inner
         * loop is a vectorizable fused-add with no stride gaps. */
        for (int32_t k = csr->row_ptr_pos[i]; k < csr->row_ptr_pos[i + 1]; k++) {
            const float *x_col = X + csr->pos_indices[k] * batch;
            for (int32_t b = 0; b < batch; b++) {
                y_row[b] += x_col[b];
            }
        }

        /* Negative contributions: same pattern, subtract instead of add */
        for (int32_t k = csr->row_ptr_neg[i]; k < csr->row_ptr_neg[i + 1]; k++) {
            const float *x_col = X + csr->neg_indices[k] * batch;
            for (int32_t b = 0; b < batch; b++) {
                y_row[b] -= x_col[b];
            }
        }
    }

    return 0;
}

/*
 * ternary_csr_free — Release all heap memory and reset the struct.
 *
 * Freeing in a defined order prevents any use-after-free. Zeroing the struct
 * after free makes double-free safe (free(NULL) is a no-op in C99) and converts
 * use-after-free from silent corruption into a detectable NULL dereference.
 * This also allows reusing the same struct after calling ternary_csr_init again.
 */
void ternary_csr_free(TernaryCSR *csr) {
    if (!csr) return;

    /* Free index arrays first (largest allocations) */
    free(csr->pos_indices);
    free(csr->neg_indices);

    /* Free row pointer arrays */
    free(csr->row_ptr_pos);
    free(csr->row_ptr_neg);

    /* Free per-row count arrays */
    free(csr->n_pos);
    free(csr->n_neg);

    /* Zero the entire struct to convert use-after-free into NULL deref */
    memset(csr, 0, sizeof(TernaryCSR));
}

int ternary_csc_from_csr(TernaryCSC *csc, const TernaryCSR *csr) {
    int R = csr->rows;
    int C = csr->cols;
    csc->rows = R;
    csc->cols = C;
    csc->total_pos = csr->total_pos;
    csc->total_neg = csr->total_neg;

    csc->col_ptr_pos = (int32_t *)calloc(C + 1, sizeof(int32_t));
    csc->pos_row_idx = (int32_t *)malloc(csr->total_pos * sizeof(int32_t));
    csc->col_ptr_neg = (int32_t *)calloc(C + 1, sizeof(int32_t));
    csc->neg_row_idx = (int32_t *)malloc(csr->total_neg * sizeof(int32_t));
    if (!csc->col_ptr_pos || !csc->pos_row_idx || !csc->col_ptr_neg || !csc->neg_row_idx) {
        ternary_csc_free(csc);
        return -1;
    }

    for (int32_t i = 0; i < csr->total_pos; i++) {
        csc->col_ptr_pos[csr->pos_indices[i] + 1]++;
    }
    for (int c = 1; c <= C; c++) {
        csc->col_ptr_pos[c] += csc->col_ptr_pos[c - 1];
    }

    int32_t *pos_count = (int32_t *)calloc(C, sizeof(int32_t));
    for (int r = 0; r < R; r++) {
        for (int32_t k = csr->row_ptr_pos[r]; k < csr->row_ptr_pos[r + 1]; k++) {
            int32_t col = csr->pos_indices[k];
            int32_t dest = csc->col_ptr_pos[col] + pos_count[col];
            csc->pos_row_idx[dest] = r;
            pos_count[col]++;
        }
    }
    free(pos_count);

    for (int32_t i = 0; i < csr->total_neg; i++) {
        csc->col_ptr_neg[csr->neg_indices[i] + 1]++;
    }
    for (int c = 1; c <= C; c++) {
        csc->col_ptr_neg[c] += csc->col_ptr_neg[c - 1];
    }
    int32_t *neg_count = (int32_t *)calloc(C, sizeof(int32_t));
    for (int r = 0; r < R; r++) {
        for (int32_t k = csr->row_ptr_neg[r]; k < csr->row_ptr_neg[r + 1]; k++) {
            int32_t col = csr->neg_indices[k];
            int32_t dest = csc->col_ptr_neg[col] + neg_count[col];
            csc->neg_row_idx[dest] = r;
            neg_count[col]++;
        }
    }
    free(neg_count);

    return 0;
}

void ternary_csc_free(TernaryCSC *csc) {
    if (csc->col_ptr_pos) { free(csc->col_ptr_pos); csc->col_ptr_pos = NULL; }
    if (csc->pos_row_idx) { free(csc->pos_row_idx); csc->pos_row_idx = NULL; }
    if (csc->col_ptr_neg) { free(csc->col_ptr_neg); csc->col_ptr_neg = NULL; }
    if (csc->neg_row_idx) { free(csc->neg_row_idx); csc->neg_row_idx = NULL; }
}

/*
 * ternary_dot_row — Compute dot product of a single CSR row with dense vector.
 *
 * Walks pos_indices[row] to gather x values at positive weight positions,
 * walks neg_indices[row] to subtract x values at negative weight positions.
 * Result: sum(x[pos]) - sum(x[neg]) = logit for one vocabulary entry.
 *
 * No SIMD optimization — used sparingly (k+1=51 times per position in sampled CE).
 * Scalar path is sufficient.
 */
float ternary_dot_row(const TernaryCSR *csr, int32_t row, const float *x) {
    if (!csr || !x) return 0.0f;

    float acc = 0.0f;

    const int32_t p0 = csr->row_ptr_pos[row];
    const int32_t p1 = csr->row_ptr_pos[row + 1];
    for (int32_t i = p0; i < p1; i++) {
        acc += x[csr->pos_indices[i]];
    }

    const int32_t n0 = csr->row_ptr_neg[row];
    const int32_t n1 = csr->row_ptr_neg[row + 1];
    for (int32_t i = n0; i < n1; i++) {
        acc -= x[csr->neg_indices[i]];
    }

    return acc;
}
