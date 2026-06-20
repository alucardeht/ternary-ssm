# Specification: ternary_matmul.c

**File:** `src/model_v12/src/core/ternary_matmul.c`
**Header:** `src/model_v12/src/core/ternary_matmul.h`
**Purpose:** Multiplication-free matrix-vector and matrix-matrix products where W ∈ {-1, 0, +1}.

---

## 1. Architectural Context

From `architecture.md` §Core Components — Ternary MatMul:

```
y[i] = sum(x[pos_indices[i]]) - sum(x[neg_indices[i]])
Cost: 2 * d_in loads + d_in adds (no multiplications)
```

This kernel is on the critical path for every SSM layer forward pass:
- `W_in`: d_model × d_state = 1024 × 128 (INT2 packed, 194 KB)
- `W_out`: d_state × d_model = 128 × 1024 (INT2 packed, 194 KB)
- `embed`: vocab × d_model = 32064 × 1024 (INT2 packed, 8.2 MB)
- `lm_head`: d_model × vocab = 1024 × 32064 (INT2 packed, 8.2 MB)

The CSR representation is the inference-time format. INT2 packed is the storage format.
CSR is built once from INT2 at load time and reused for all forward passes.

---

## 2. Data Structures

### 2.1 TernaryCSR

```c
typedef struct {
    /* Per-row index arrays */
    int32_t *pos_indices;       /* flat array: positive-weight column indices */
    int32_t *neg_indices;       /* flat array: negative-weight column indices */

    /* Row pointer arrays (CSR standard: row i spans [row_ptr[i], row_ptr[i+1])) */
    int32_t *row_ptr_pos;       /* length: rows + 1 */
    int32_t *row_ptr_neg;       /* length: rows + 1 */

    /* Per-row counts (redundant with row_ptr but cached for scalar path) */
    int32_t *n_pos;             /* length: rows — number of +1 weights per row */
    int32_t *n_neg;             /* length: rows — number of -1 weights per row */

    int32_t rows;               /* d_out */
    int32_t cols;               /* d_in  */

    int32_t total_pos;          /* total nonzero +1 entries */
    int32_t total_neg;          /* total nonzero -1 entries */
} TernaryCSR;
```

**Memory layout rationale:**
- `pos_indices` and `neg_indices` are separate flat arrays, NOT interleaved.
  Separation allows the SIMD gather path to operate on a single contiguous array per sign.
- `row_ptr_pos[i]` gives the start offset into `pos_indices` for row `i`.
  `row_ptr_pos[rows]` equals `total_pos` (sentinel).
- Same for `row_ptr_neg`.
- `n_pos[i] = row_ptr_pos[i+1] - row_ptr_pos[i]` (kept explicit to avoid the subtraction in the hot path).

**Size at model dimensions (6 layers, W_in 1024×128):**
Assuming ~50% sparsity (P(w=0) ≈ 0.5), roughly 32K nonzero entries per matrix.
- pos_indices: ~16K × 4B = 64 KB
- neg_indices: ~16K × 4B = 64 KB
- row_ptr_pos + row_ptr_neg: 2 × 129 × 4B ≈ 1 KB
- n_pos + n_neg: 2 × 128 × 4B = 1 KB
- Per-matrix CSR overhead: ~130 KB (vs 194 KB INT2 packed) — within 2× of storage size.

---

## 3. Function Specifications

### 3.1 `ternary_csr_init`

```c
void ternary_csr_init(TernaryCSR *csr, const int8_t *w_dense, int rows, int cols);
```

**Purpose:** Build the CSR structure from a dense ternary weight matrix.

**Inputs:**
- `csr`: pointer to caller-allocated `TernaryCSR` (struct fields must be zero-initialized before call)
- `w_dense`: row-major int8_t array of shape [rows × cols], values in {-1, 0, +1}
- `rows`: d_out (number of output neurons)
- `cols`: d_in  (number of input dimensions)

**Outputs:**
- All fields of `*csr` are populated and heap-allocated.
- `csr->pos_indices`, `csr->neg_indices`, `csr->row_ptr_pos`, `csr->row_ptr_neg`,
  `csr->n_pos`, `csr->n_neg` are malloc'd internally.

**Algorithm (two-pass CSR construction):**
```
Pass 1 — count:
  for i in 0..rows:
    n_pos[i] = count of w_dense[i*cols + j] == +1, j in 0..cols
    n_neg[i] = count of w_dense[i*cols + j] == -1, j in 0..cols

  Prefix-sum n_pos into row_ptr_pos (standard CSR build).
  Prefix-sum n_neg into row_ptr_neg.
  total_pos = row_ptr_pos[rows]
  total_neg = row_ptr_neg[rows]

  malloc pos_indices[total_pos], neg_indices[total_neg].

Pass 2 — fill:
  cursor_pos[i] = row_ptr_pos[i]   (temporary per-row write cursor)
  cursor_neg[i] = row_ptr_neg[i]
  for i in 0..rows:
    for j in 0..cols:
      if w_dense[i*cols + j] == +1:
        pos_indices[cursor_pos[i]++] = j
      elif w_dense[i*cols + j] == -1:
        neg_indices[cursor_neg[i]++] = j
```

**Memory allocated:** O(total_pos + total_neg + 2*(rows+1) + 2*rows) int32_t values.

**Preconditions:** Every element of w_dense is exactly -1, 0, or +1. Caller guarantees this.

**Not thread-safe.** Call once at model load. No side effects beyond writing `*csr`.

---

### 3.2 `ternary_matvec`

```c
void ternary_matvec(const TernaryCSR *csr, const float *x, float *y, int rows);
```

**Purpose:** Compute y = W @ x for a single input vector. Scalar (portable) implementation.

**Inputs:**
- `csr`: initialized TernaryCSR
- `x`: float array of length `csr->cols` (d_in), row-major, alignment not required
- `y`: float array of length `rows`, pre-allocated by caller, written in-place
- `rows`: must equal `csr->rows`

**Outputs:**
- `y[i]` = sum of `x[j]` for all j in pos_indices[i] minus sum of `x[j]` for all j in neg_indices[i]

**Algorithm:**
```c
for (int i = 0; i < rows; i++) {
    float acc = 0.0f;
    int p_start = csr->row_ptr_pos[i];
    int p_end   = csr->row_ptr_pos[i + 1];
    int n_start = csr->row_ptr_neg[i];
    int n_end   = csr->row_ptr_neg[i + 1];

    for (int k = p_start; k < p_end; k++)
        acc += x[csr->pos_indices[k]];
    for (int k = n_start; k < n_end; k++)
        acc -= x[csr->neg_indices[k]];

    y[i] = acc;
}
```

**FLOPs:** `total_pos + total_neg` floating-point add/sub operations.
At W_in (1024×128, 50% ternary density): ~32K adds. Zero multiplications.

**Memory access pattern:** `x` is accessed randomly (gather pattern). For d_in=1024 (4 KB), `x` fits in L1 cache (typically 32 KB). Cache miss risk is LOW for W_in/W_out. For lm_head (d_in=1024, d_out=32064), x fits in L1 and y is streamed sequentially — acceptable.

**No aliasing:** `x` and `y` must not overlap.

---

### 3.3 `ternary_matmul_batch`

```c
void ternary_matmul_batch(const TernaryCSR *csr, const float *X, float *Y,
                          int rows, int cols, int batch);
```

**Purpose:** Compute Y = W @ X where X is [cols × batch] (column-major batch dimension).
Each column of X is a separate input vector; each column of Y is the corresponding output.

**Inputs:**
- `csr`: initialized TernaryCSR
- `X`: float array of shape [cols × batch], layout: X[j * batch + b] = X_jb
  (column-major in the batch dimension — j is the d_in axis, b is the batch axis)
- `Y`: float array of shape [rows × batch], layout: Y[i * batch + b] = Y_ib,
  pre-allocated by caller, written in-place
- `rows`: d_out, must equal `csr->rows`
- `cols`: d_in, must equal `csr->cols`
- `batch`: number of vectors to process simultaneously

**Outputs:**
- `Y[i * batch + b]` = W[i, :] @ X[:, b] for all i, b

**Layout rationale:**
Column-major batch layout (`X[j * batch + b]`) means that for a fixed column index j,
the batch slice `X[j * batch .. j * batch + batch]` is contiguous. When the SIMD path
gathers x values for a given row, it loads `batch` contiguous floats per index hit.
This is substantially better for vectorisation than row-major batch layout.

**Algorithm (outer loop over rows, inner loop over batch):**
```c
// Zero output
memset(Y, 0, rows * batch * sizeof(float));

for (int i = 0; i < rows; i++) {
    float *y_row = Y + i * batch;

    // Positive contributions
    for (int k = csr->row_ptr_pos[i]; k < csr->row_ptr_pos[i + 1]; k++) {
        const float *x_col = X + csr->pos_indices[k] * batch;
        for (int b = 0; b < batch; b++)
            y_row[b] += x_col[b];
    }
    // Negative contributions
    for (int k = csr->row_ptr_neg[i]; k < csr->row_ptr_neg[i + 1]; k++) {
        const float *x_col = X + csr->neg_indices[k] * batch;
        for (int b = 0; b < batch; b++)
            y_row[b] -= x_col[b];
    }
}
```

**FLOPs:** `(total_pos + total_neg) * batch` float adds. No multiplications.
At W_in (1024×128), batch=16: ~512K adds.

**Cache behaviour:**
- `y_row` is batch floats = 64 bytes at batch=16. Stays in registers.
- `x_col` is batch floats = 64 bytes at batch=16. Fits in one cache line.
- For each index hit, one cache-line load from `X` plus batch fused add/sub.
- For batch >= 8 and typical L1=32 KB: working set is x[d_in=1024 floats = 4 KB] + y[rows=128 × batch=16 = 8 KB] = 12 KB. Fits in L1.

**Reuse opportunity:** The inner `b` loop over `y_row[b] += x_col[b]` is a SIMD-friendly
contiguous accumulate. The compiler should auto-vectorise this at -O2. The AVX2 path
in §3.5 handles it explicitly.

**Preconditions:** `batch` is a multiple of 8 (required for AVX2 path, recommended for scalar too).
If `batch` is not a multiple of 8, caller must pad X/Y or use scalar fallback.

---

### 3.4 `ternary_csr_update`

```c
void ternary_csr_update(TernaryCSR *csr, int row, int col, int8_t old_val, int8_t new_val);
```

**Purpose:** Incrementally update the CSR structure when a single weight flips value.
Used during training when MVUE sampling produces a weight transition, e.g. 0 → +1, +1 → 0,
0 → -1, +1 → -1, etc.

**Inputs:**
- `csr`: initialized and mutable TernaryCSR
- `row`: the output neuron index (0 <= row < csr->rows)
- `col`: the input dimension index (0 <= col < csr->cols)
- `old_val`: previous weight value (must be -1, 0, or +1)
- `new_val`: new weight value (must be -1, 0, or +1)

**Outputs:**
- `*csr` is modified in-place so that the CSR correctly reflects the new weight.
- If `old_val == new_val`, this is a no-op.

**Algorithm:**

The CSR stores indices in sorted order within each row for cache efficiency.
Updates use a sorted insert / sorted delete on the relevant row slice.

```
Step 1 — Remove old_val's contribution (if old_val != 0):
  target_array = (old_val == +1) ? pos_indices : neg_indices
  target_rptr  = (old_val == +1) ? row_ptr_pos : row_ptr_neg
  target_n     = (old_val == +1) ? n_pos       : n_neg

  Find position p in target_array[target_rptr[row] .. target_rptr[row+1])
    where target_array[p] == col.  (binary search, O(log n_pos[row]))
  Shift target_array[p+1 .. target_rptr[row+1]) left by 1.
  Decrement target_rptr[row+1 .. rows] by 1.
  Decrement total_pos or total_neg by 1.
  Decrement target_n[row] by 1.

  NOTE: This does NOT realloc. The freed slot is handled by shifting.
        Row slices remain contiguous and sorted after shift.

Step 2 — Insert new_val's contribution (if new_val != 0):
  target_array = (new_val == +1) ? pos_indices : neg_indices
  target_rptr  = (new_val == +1) ? row_ptr_pos : row_ptr_neg
  target_n     = (new_val == +1) ? n_pos       : n_neg
  total_counter = (new_val == +1) ? total_pos   : total_neg

  Find insertion position p (binary search for col in sorted row slice).
  If total_counter == allocated_capacity:
    realloc target_array to 1.5× capacity.
  Shift target_array[p .. target_rptr[row+1]) right by 1.
  Write target_array[p] = col.
  Increment target_rptr[row+1 .. rows] by 1.
  Increment total_counter by 1.
  Increment target_n[row] by 1.
```

**Complexity:**
- Delete: O(log n_pos[row] + n_pos[row]) — binary search + linear shift
- Insert: O(log n_pos[row] + n_pos[row]) — binary search + linear shift + occasional realloc
- Worst case (dense row): O(cols) = O(1024) — acceptable since updates are rare in training
  (Crystallization: after ~10K steps, 20-30% of weights no longer update)

**Thread safety:** NOT thread-safe. External lock required if called from multiple threads.

**Design note:** This function is NOT on the hot inference path. It is called only during training
when a ternary weight flips. At LR=0.1 and crystallization τ=3, flip rate drops to near zero
for crystallized weights (|s| > 3). The amortised cost is low.

**Alternative for bulk updates (e.g. repack after epoch):** Call `ternary_csr_init` again
from the updated INT2 packed weights. This is O(rows × cols) but avoids fragmentation.
Recommended when more than 5% of weights flip in a single step.

---

### 3.5 `ternary_matvec_avx2`

```c
void ternary_matvec_avx2(const TernaryCSR *csr, const float *x, float *y, int rows);
```

**Purpose:** AVX2-optimised single-vector matmul. Same semantics as `ternary_matvec` but
uses 256-bit SIMD gather instructions for throughput.

**Inputs/Outputs:** Identical to `ternary_matvec`.

**Alignment requirement:**
- `x` must be 32-byte aligned (`_mm_malloc(cols * sizeof(float), 32)` or `posix_memalign`).
- `y` must be 32-byte aligned.
- `pos_indices` and `neg_indices` must be 32-byte aligned (enforce in `ternary_csr_init`).

**SIMD strategy — gather accumulate:**

AVX2 provides `_mm256_i32gather_ps` which loads 8 floats at non-contiguous indices in one
instruction. This maps directly to the CSR access pattern.

For each row `i`, process the pos_indices slice in chunks of 8:

```
gather_and_acc(indices_ptr, count, x_base_ptr):
  __m256 acc = _mm256_setzero_ps()
  int k = 0

  // Main loop: 8 indices at a time
  while (k + 8 <= count):
    __m256i idx = _mm256_loadu_si256((__m256i*)(indices_ptr + k))
    __m256  val = _mm256_i32gather_ps(x_base_ptr, idx, 4)   // scale=4 (bytes per float)
    acc = _mm256_add_ps(acc, val)
    k += 8

  // Tail: remaining 0..7 indices, scalar
  float tail_acc = hsum_avx(acc)     // horizontal sum of 8 lanes
  while (k < count):
    tail_acc += x_base_ptr[indices_ptr[k]]
    k++

  return tail_acc
```

Horizontal sum `hsum_avx`:
```c
static inline float hsum_avx(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
```

Full row kernel:
```c
y[i] = gather_and_acc(pos_indices + row_ptr_pos[i],
                      n_pos[i], x)
     - gather_and_acc(neg_indices + row_ptr_neg[i],
                      n_neg[i], x);
```

**Throughput analysis (W_in, 1024×128, 50% density, ~128 pos + 128 neg per row average):**
- Gather throughput: `_mm256_i32gather_ps` is ~12 cycles / 8 floats on Skylake/Zen3.
- Per row: 2 × (128/8) × 12 = 384 cycles for gather.
- 128 rows: ~49K cycles.
- At 3 GHz: ~16 µs per matvec (W_in forward, single vector).
- Scalar equivalent: ~2× slower (random access, no gather pipelining).

**Latency vs throughput:** AVX2 gather has high latency (~20 cycles) but can be pipelined
by the out-of-order engine when multiple gathers are independent (consecutive rows).
Loop over rows ensures independence.

**Compiler guard:** Wrap entire function in:
```c
#ifdef __AVX2__
void ternary_matvec_avx2(...) { ... }
#else
void ternary_matvec_avx2(...) {
    ternary_matvec(csr, x, y, rows);  /* fallback */
}
#endif
```

**Dispatch:** The header exposes a function pointer `ternary_matvec_fn` that is set at
init time by `ternary_dispatch_init()`:
```c
extern void (*ternary_matvec_fn)(const TernaryCSR*, const float*, float*, int);

void ternary_dispatch_init(void) {
    if (cpu_has_avx2())
        ternary_matvec_fn = ternary_matvec_avx2;
    else
        ternary_matvec_fn = ternary_matvec;
}
```

---

## 4. Header File Structure

`src/model_v12/src/core/ternary_matmul.h`:

```c
#pragma once
#include <stdint.h>

typedef struct {
    int32_t *pos_indices;
    int32_t *neg_indices;
    int32_t *row_ptr_pos;
    int32_t *row_ptr_neg;
    int32_t *n_pos;
    int32_t *n_neg;
    int32_t  rows;
    int32_t  cols;
    int32_t  total_pos;
    int32_t  total_neg;
} TernaryCSR;

/* Lifecycle */
void ternary_csr_init(TernaryCSR *csr, const int8_t *w_dense, int rows, int cols);
void ternary_csr_free(TernaryCSR *csr);

/* Forward passes */
void ternary_matvec(const TernaryCSR *csr, const float *x, float *y, int rows);
void ternary_matmul_batch(const TernaryCSR *csr, const float *X, float *Y,
                          int rows, int cols, int batch);

/* Training update */
void ternary_csr_update(TernaryCSR *csr, int row, int col,
                        int8_t old_val, int8_t new_val);

/* AVX2 path */
void ternary_matvec_avx2(const TernaryCSR *csr, const float *x, float *y, int rows);

/* Dispatch — call once at startup */
void ternary_dispatch_init(void);
extern void (*ternary_matvec_fn)(const TernaryCSR*, const float*, float*, int);
```

---

## 5. Memory Management Rules

- `ternary_csr_init` allocates all index arrays internally with `malloc`.
- `ternary_csr_free` must free all six arrays (`pos_indices`, `neg_indices`,
  `row_ptr_pos`, `row_ptr_neg`, `n_pos`, `n_neg`) and zero the struct.
- Alignment: allocate `pos_indices` and `neg_indices` with 32-byte alignment
  (`_mm_malloc` or `aligned_alloc(32, ...)`) to allow aligned AVX2 loads.
- Caller owns `w_dense`, `x`, `y`, `X`, `Y`. This module does not take ownership.
- `ternary_csr_update` may `realloc` `pos_indices` or `neg_indices`. After realloc,
  the original pointer is invalid. All internal row_ptr offsets remain valid (they
  reference positions within the same flat array, which realloc preserves).

---

## 6. Integration Points

### 6.1 Where this module is called

| Call site | CSR matrix | rows | cols | batch | Path |
|-----------|-----------|------|------|-------|------|
| SSM in_proj | W_in per layer | 128 | 1024 | 16 | `ternary_matmul_batch` |
| SSM out_proj | W_out per layer | 1024 | 128 | 16 | `ternary_matmul_batch` |
| embed lookup | embed table | 1024 | 32064 | 1 (single token) | NOT used — embed lookup is a row select, not matmul |
| lm_head | lm_head | 32064 | 1024 | 16 | `ternary_matmul_batch` |

Note: embedding lookup (`embed[token_id]`) is a single-row read from the CSR, equivalent
to `ternary_matvec` with a one-hot x. However, since INT2 is used for storage and RNS
handles the actual embedding arithmetic, this module is NOT on the embedding path.
The embed CSR is only needed if lm_head weight tying is used.

### 6.2 Relationship to INT2 packed format

`w_dense` (int8_t, values {-1, 0, +1}) is the unpacked view of INT2 stored weights.
The caller is responsible for unpacking INT2 → int8_t before calling `ternary_csr_init`.

INT2 packing from `math-innovations.md` §13:
```c
// 4 weights per byte, values shifted: stored = w + 1 ∈ {0, 1, 2}
uint8_t pack4(int8_t w0, int8_t w1, int8_t w2, int8_t w3) {
    return (uint8_t)((w0+1) | ((w1+1)<<2) | ((w2+1)<<4) | ((w3+1)<<6));
}
int8_t unpack_nth(uint8_t byte, int n) {
    return (int8_t)((byte >> (2*n)) & 0x3) - 1;
}
```

A helper `ternary_csr_init_from_int2` (optional, not in MVP) would accept the packed
uint8_t array directly and unpack on-the-fly during the two-pass CSR build.

### 6.3 MVUE training interaction

During training, `ternary_csr_update` is called when a shadow weight `s` crosses
a quantization boundary. The boundary conditions from `architecture.md` §MVUE:

```
w_sampled = +1 if sample < max(s, 0)
w_sampled = -1 if sample < max(-s, 0)
w_sampled =  0 otherwise
```

Crystallization rule: if `|s| > 3`, skip the update entirely.
If the sampled `w_new != w_old`, call `ternary_csr_update(csr, row, col, w_old, w_new)`.

For bulk repack (every `repack_interval` steps), call `ternary_csr_init` from the
full updated int8_t weight array instead of accumulating incremental updates.

---

## 7. FLOPs Summary

| Operation | FLOPs (no multiplications) | Notes |
|-----------|---------------------------|-------|
| `ternary_matvec`, W_in (128×1024, 50% ternary) | ~32K add/sub | 0 multiplications |
| `ternary_matvec`, W_out (1024×128, 50% ternary) | ~32K add/sub | 0 multiplications |
| `ternary_matvec`, lm_head (32064×1024, 50%) | ~16M add/sub | Dominates inference |
| `ternary_matmul_batch`, W_in, batch=16 | ~512K add/sub | Main training path |
| `ternary_matmul_batch`, lm_head, batch=16 | ~256M add/sub | Bottleneck — investigate blocking |
| FP32 matmul equivalent (d_out×d_in MACs) | 2 × d_out × d_in FLOPs | Baseline reference |
| Ratio (ternary vs FP32) | ~50% × 1 add vs 1 MAC | Ternary is ~1 add per nonzero vs 1 MAC per element |

FP32 matmul for lm_head at batch=16: 2 × 32064 × 1024 × 16 = 1.05B FLOPs.
Ternary lm_head at batch=16 (50% density): 32064 × 512 × 16 = 262M ops (adds only).
Effective speedup potential: ~4× in compute, additional benefit from no-multiply pipeline.

---

## 8. Build Integration

```cmake
# In src/model_v12/CMakeLists.txt
add_library(ternary_matmul STATIC
    src/core/ternary_matmul.c
)

target_include_directories(ternary_matmul PUBLIC src/core)

# AVX2 flag — only for this translation unit
if(SIMD_BACKEND STREQUAL "AVX2")
    target_compile_options(ternary_matmul PRIVATE -mavx2)
    target_compile_definitions(ternary_matmul PRIVATE TERNARY_USE_AVX2)
endif()

if(SIMD_BACKEND STREQUAL "NEON")
    target_compile_definitions(ternary_matmul PRIVATE TERNARY_USE_NEON)
endif()
```

The scalar path compiles cleanly on any C99 target (including custom accelerator HLS flows).

---

## 9. Correctness Invariants

The following must hold after every `ternary_csr_init` and after every `ternary_csr_update`:

1. `row_ptr_pos[0] == 0` and `row_ptr_pos[rows] == total_pos`
2. `row_ptr_neg[0] == 0` and `row_ptr_neg[rows] == total_neg`
3. For each row `i`: `n_pos[i] == row_ptr_pos[i+1] - row_ptr_pos[i]`
4. For each row `i`: `n_neg[i] == row_ptr_neg[i+1] - row_ptr_neg[i]`
5. No column index appears in both `pos_indices` and `neg_indices` for the same row.
6. All column indices are in `[0, csr->cols)`.
7. Within each row slice, indices are sorted ascending (required for binary search in update).

A debug-mode `ternary_csr_validate` function should verify all seven invariants and
return a non-zero error code if any is violated.
