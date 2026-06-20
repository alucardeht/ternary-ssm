/*
 * types.h — Central type definitions for v12
 *
 * This is THE foundation header. Every other .c file includes it directly or
 * indirectly. It defines every shared struct used across the model, training,
 * and I/O layers.
 *
 * Inclusion order contract:
 *   #include "types.h"          (always first, no prerequisites)
 *   #include "ternary_matmul.h" (includes types.h, adds TernaryCSR functions)
 *   #include "ssm_scan.h"       (includes types.h, adds scan functions)
 *
 * Design principles:
 *   - C99 portable — no compiler extensions, no VLAs, no _Static_assert before C11
 *   - All array sizes are compile-time constants derived from the #defines below
 *   - Pointer fields are heap-allocated by their owning module (model.c, train.c)
 *   - NULL pointer = "not present" (inference-only checkpoints omit shadow weights)
 *   - 32-byte alignment hints via ALIGN32 for SIMD-friendly buffers
 *
 * Architecture reference: docs/architecture.md
 */

#ifndef V12_TYPES_H
#define V12_TYPES_H

#include <stdint.h>   /* uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int32_t */
#include <stddef.h>   /* size_t */
#include "r493_gates.h"

/* =========================================================================
 * R356 F2 — INT8-style shadow weights (default OFF)
 *
 * When V12_F2_INT8_SHADOWS == 0 (default) the shadow buffers stay FP32 and
 * every SHADOW_DECODE / SHADOW_STORE macro collapses to a direct float
 * read/write. The .o files must be byte-identical to the pre-R356 build.
 *
 * When V12_F2_INT8_SHADOWS == 1 the shadow buffers become int16_t cells
 * encoded by shadow_codec.h (split MVUE / saturation grids), and the
 * macros route through the codec.
 *
 * Reference: .workspace/research/r356-f2-int8-shadow-handoff.md
 * ========================================================================= */
#ifndef V12_F2_INT8_SHADOWS
#define V12_F2_INT8_SHADOWS 1
#endif

#ifndef V12_TSMSQ_ENABLED
#define V12_TSMSQ_ENABLED 1
#endif

#ifndef V12_AF8C_ENABLED
#define V12_AF8C_ENABLED 1
#endif

#ifndef V12_ZMMA_ENABLED
#define V12_ZMMA_ENABLED 1
#endif

#ifndef V12_CDSVQ_ENABLED
#define V12_CDSVQ_ENABLED 1
#endif

#ifndef V12_ASSSI_ENABLED
#define V12_ASSSI_ENABLED 1
#endif

#ifndef V12_ASSSI_VALIDATION
#define V12_ASSSI_VALIDATION 0
#endif

#ifndef V12_DADD_ENABLED
#define V12_DADD_ENABLED 1
#endif

#ifndef V12_DADD_VALIDATION
#define V12_DADD_VALIDATION 0
#endif

#ifndef V12_SISSF_ENABLED
#define V12_SISSF_ENABLED 1
#endif

#ifndef V12_PRBLT_ENABLED
#define V12_PRBLT_ENABLED 1
#endif

#ifndef V12_TOLR_ENABLED
#define V12_TOLR_ENABLED 1
#endif

#ifndef V12_MRLC_ENABLED
#define V12_MRLC_ENABLED 1
#endif

#ifndef V12_MRLC_ENERGY_LIMIT
#define V12_MRLC_ENERGY_LIMIT 10000.0f
#endif

#ifndef V12_ASPNG_ENABLED
#define V12_ASPNG_ENABLED 1
#endif


#ifndef V12_CDSVQ_KL_COEFF
#define V12_CDSVQ_KL_COEFF 0.001f
#endif

#ifndef V12_CDSVQ_DELTA
#define V12_CDSVQ_DELTA 1.0f
#endif

#if V12_F2_INT8_SHADOWS
#include "shadow_codec.h"
typedef int16_t shadow_t;
#define SHADOW_DECODE(x)            shadow_decode((int16_t)(x))
#define SHADOW_STORE(dst, s, rnd)   ((dst) = stochastic_round_shadow((float)(s), (uint32_t)(rnd)))
#define SHADOW_SIZEOF               (sizeof(int16_t))
#else
typedef float shadow_t;
#define SHADOW_DECODE(x)            ((float)(x))
#define SHADOW_STORE(dst, s, rnd)   ((void)(rnd), (dst) = (float)(s))
#define SHADOW_SIZEOF               (sizeof(float))
#endif

/* =========================================================================
 * SECTION 1 — COMPILE-TIME CONSTANTS
 *
 * Single source of truth for all model dimensions and training hyperparameters.
 * Every struct size is derivable from these constants.
 * ========================================================================= */

/* --- Model dimensions ---------------------------------------------------- */
#ifndef V12_N_LAYERS
#define V12_N_LAYERS        6       /* Production: 6 SSM layers                */
#endif
#ifndef V12_D_MODEL
#define V12_D_MODEL      1024       /* Production: model dimension              */
#endif
#ifndef V12_D_EMBED
#define V12_D_EMBED       128       /* ALBERT factored embedding dimension      */
#endif
#ifndef V12_D_STATE
#define V12_D_STATE       128       /* Production: SSM state dimension          */
#endif
#ifndef V12_N_HEADS
#define V12_N_HEADS        16       /* Production: 16 heads                    */
#endif
#ifndef V12_VOCAB_SIZE
#define V12_VOCAB_SIZE  32064       /* BPE vocabulary matching chat_combined    */
#endif
#ifndef V12_SEQ_LEN
#define V12_SEQ_LEN      1024       /* Production: sequence length              */
#endif
#ifndef V12_CONV_KERNEL
#define V12_CONV_KERNEL     4       /* conv1d kernel width in SSM layer      */
#endif
#ifndef V12_MTP_RANK
#define V12_MTP_RANK       20       /* Rank of MTP low-rank projections U, V */
#endif
#ifndef V12_MTP_K
#define V12_MTP_K           4       /* MTP lookahead horizon (steps)         */
#endif

/* --- SSM scan constants -------------------------------------------------- */
#define V12_MAX_D_STATE   256       /* Upper bound for stack allocs in scan  */
#define V12_SSM_CHUNK      64       /* Chunk size for cache-friendly scanning */

/* --- Gradient checkpointing for SSM backward ----------------------------- */
/* Every K=16 steps, one h checkpoint is saved during forward.
 * Backward re-runs forward within each segment to reconstruct h exactly.
 * Memory: (seq_len/K + 1) checkpoints instead of seq_len full h vectors.   */
#define V12_CKPT_STRIDE  16         /* Checkpoint every 16 steps            */
#define V12_CKPT_COUNT   ((V12_SEQ_LEN / V12_CKPT_STRIDE) + 1)  /* Total checkpoint count */

/* --- Ternary / INT2 packing ---------------------------------------------- */
/* 4 ternary weights packed per byte using 2-bit encoding:
 *   bits [1:0] = weight 0:  0b00=0, 0b01=+1, 0b11=-1
 *   bits [3:2] = weight 1,  bits [5:4] = weight 2,  bits [7:6] = weight 3  */
#define V12_INT2_WEIGHTS_PER_BYTE  4

/* --- Training hyperparameters -------------------------------------------- */
#define V12_LR_MAX              0.0003f  /* Adam LR (standard for transformers) */
#define V12_LR_WARMUP_STEPS     500     /* Longer warmup for stability        */
#define V12_LR_TOTAL_STEPS    18000     /* Total training steps              */
#define V12_LR_MIN_FACTOR       0.01f   /* Floor: LR never drops below 1%   */
#define V12_GP_KL_BETA          0.01f   /* GP-KL loss weight                 */
#define V12_MTP_WEIGHT          0.3f    /* Multi-step prediction loss weight */
#define V12_WCS_STRIDE          4       /* WCS-GPKL: Welford subsampling temporal stride */
#define V12_CRYSTALLIZATION_TAU 4.6f    /* |logit| > tau → weight frozen     */
#define V12_EMA_SPRT_GAMMA      0.9f    /* EMA decay for gradient accumulator */
#define V12_EMA_SPRT_THRESHOLD  0.01f   /* Threshold for update trigger       */
#define V12_REPACK_INTERVAL     500     /* Steps between shadow→INT2 repacks */
#define V12_SOFT_BOUNDARY_SCALE 10.0f   /* k in exp(-k*(|s|-1)^2) mask       */
#define V12_CONTRACTION_BETA       0.01f   /* Penalty weight for σ_min(I-A)     */
#define V12_CONTRACTION_WINDOW     V12_CKPT_STRIDE  /* Window = checkpoint stride     */
#define V12_CONTRACTION_THRESHOLD  0.5f   /* A^W must be below this             */
#define V12_BI_STE_FACTOR          0.5f   /* Gradient scale for zero-quantized weights (Bi-STE, R348 F5) */
#define V12_PHGD_SIGMA0            0.1f   /* QBS basin radius scale for PHGD Stability Optimizer (Aspect A03) */
#define V12_PHGD_TAU               0.001f /* Threshold gating parameter for coordinate descent */
#define V12_PHGD_EPS               0.0001f/* Smooth relaxation epsilon for coordinate sigmoid */
#define V12_SRE_COLLAPSE_THRESHOLD 0.35f  /* Collapse threshold coefficient eta */
#define V12_SRE_RADEMACHER_VECTORS 4      /* Number of Rademacher vectors S */
#define V12_SRE_EPSILON_0          0.0001f/* Chebyshev approximation boundary epsilon_0 */
#define V12_EVIDENCE_RATE_MODE_CURRENT 0
#define V12_EVIDENCE_RATE_MODE_TSA     1
#define V12_EVIDENCE_RATE_MODE_GBAR    2
#define V12_EVIDENCE_RATE_STATS_SLOTS  32

/* --- Checkpoint / I/O ---------------------------------------------------- */
#define V12_LOSS_HISTORY_LEN   1024     /* Ring buffer length for loss curve */
#define V12_MAX_LAYERS          32      /* Max layers in checkpoint header   */

/* --- SIMD alignment hint ------------------------------------------------- */
/* Mark heap-allocated arrays that benefit from 32-byte alignment (AVX2).
 * This is a documentation annotation only — actual alignment is the caller's
 * responsibility (use aligned_alloc(32, ...) or _mm_malloc(size, 32)).      */
#define ALIGN32  /* 32-byte aligned */

/* --- PCA-INT8 Activation Cache for Frozen FP Layers --------------------- */
#define V12_PCA_CACHE_RANK  20      /* Number of principal components per cache */

/* --- Sampled Softmax (NCE for lm_head vocabulary) ----------------------- */
#define V12_SS_K            250     /* Noise samples per target token         */
#define V12_SS_N_SAMPLES    (V12_SS_K + 1)  /* Total: target + k noise = 51 */

/* --- Hot/Cold Two-Tier lm_head ------------------------------------------ */
#define V12_K_HOT          4000      /* Top-K frequent tokens get updates      */


/* =========================================================================
 * SECTION 2 — TERNARY CSR WEIGHT MATRIX
 *
 * Inference-time representation of a ternary {-1, 0, +1} weight matrix.
 * Stored in CSR (Compressed Sparse Row) format, split by sign.
 *
 * Built once from INT2 packed weights at model load time. Rebuilt from
 * shadow weights every V12_REPACK_INTERVAL steps during training.
 *
 * Full function spec: src/core/ternary_matmul.h
 * ========================================================================= */

typedef struct {
    /* --- Index arrays (flat, contiguous per sign) ----------------------- */
    int32_t *pos_indices;   /* ALIGN32. Column indices where weight == +1.
                             * Length: total_pos. Sorted ascending per row.  */
    int32_t *neg_indices;   /* ALIGN32. Column indices where weight == -1.
                             * Length: total_neg. Sorted ascending per row.  */

    /* --- Row pointer arrays (standard CSR sentinel format) -------------- */
    int32_t *row_ptr_pos;   /* Length: rows + 1. row_ptr_pos[i] is the start
                             * index into pos_indices for row i. Sentinel:
                             * row_ptr_pos[rows] == total_pos.               */
    int32_t *row_ptr_neg;   /* Same structure for neg_indices.               */

    /* --- Per-row counts (redundant with row_ptr, cached for scalar path) */
    int32_t *n_pos;         /* Length: rows. n_pos[i] = number of +1 weights
                             * in row i. Equals row_ptr_pos[i+1]-row_ptr_pos[i]. */
    int32_t *n_neg;         /* Length: rows. Same for -1 weights.            */

    /* --- Matrix dimensions ---------------------------------------------- */
    int32_t  rows;          /* d_out: number of output neurons               */
    int32_t  cols;          /* d_in:  number of input dimensions             */

    /* --- Total nonzero counts (equals row_ptr sentinel values) ---------- */
    int32_t  total_pos;     /* Total number of +1 entries across all rows    */
    int32_t  total_neg;     /* Total number of -1 entries across all rows    */

    /* --- Capacity for incremental updates (ternary_csr_update) ---------- */
    int32_t  cap_pos;       /* Allocated capacity of pos_indices (>= total_pos) */
    int32_t  cap_neg;       /* Allocated capacity of neg_indices (>= total_neg) */
} TernaryCSR;


/* =========================================================================
 * SECTION 3 — GP-KL WELFORD MOMENT ACCUMULATORS
 *
 * Online statistics accumulated over SSM hidden states during the forward
 * pass. Used to compute the GP-KL unified loss (architecture.md §5).
 *
 * One GpKlMoments per layer per batch element. Accumulators persist across
 * training steps and converge to the true statistics over the dataset.
 *
 * Updated by: ssm_scan_forward (via gp_kl_welford_update hook)
 * Consumed by: compute_gp_kl_loss in train.c
 * ========================================================================= */

typedef struct {
    /* --- Sample count (Welford denominator) ------------------------------ */
    int32_t  n;                         /* Total state vectors processed     */

    /* --- Mean accumulator (Welford online mean) -------------------------- */
    float    mean[V12_MAX_D_STATE];     /* Running mean per state dimension  */

    /* --- Variance accumulator (Welford M2) ------------------------------- */
    float    M2[V12_MAX_D_STATE];       /* Sum of squared deviations; divide
                                         * by n to get variance.             */

    /* --- Lag-1 autocorrelation ------------------------------------------ */
    float    lag1_prev[V12_MAX_D_STATE];/* h[t-1] snapshot (updated each step)*/
    float    lag1_sum[V12_MAX_D_STATE]; /* Sum of h[t]*h[t-1] per dimension  */

    /* --- Stability: E[|h_t - h_{t-1}|^2] -------------------------------- */
    float    stability_sum;             /* Scalar accumulator across all dims */

    /* --- Orthogonality: E[<h_i, h_j>] for 32 random pairs -------------- */
    /* Pair indices are integers stored as float for alignment simplicity.
     * pair_a[k] and pair_b[k] give the two batch-item indices being compared.
     * Initialized once at model init from a fixed RNG seed.                 */
    float    orth_pair_a[32];           /* First index of each random pair   */
    float    orth_pair_b[32];           /* Second index of each random pair  */
    float    orth_sum[32];              /* Accumulated dot-product per pair  */
} GpKlMoments;


/* =========================================================================
 * SECTION 4 — CO-SPECTRAL SUBSPACE (RANK-20 PCA)
 *
 * Low-rank subspace computed from SSM output statistics after 50 warmup
 * steps. Used to constrain inter-layer gradient flow during backprop.
 *
 * One CoSpectralSubspace per layer, owned by the Layer struct.
 * ========================================================================= */

typedef struct {
    /* --- PCA basis vectors (column-major, rank-20) ----------------------- */
    float   *U;             /* ALIGN32. Shape: [d_state × 20]. Top-20
                             * eigenvectors of the hidden state covariance.  */

    /* --- Online covariance accumulator (used during warmup only) --------- */
    float   *cov_accum;     /* Shape: [d_state × d_state]. Accumulated outer
                             * product sum. Freed after calibration.         */

    /* --- Calibration state ---------------------------------------------- */
    int32_t  n_samples;     /* Step counter during warmup (0..50)            */
    int32_t  calibrated;    /* 0 = accumulating, 1 = PCA done and U is valid */
} CoSpectralSubspace;


/* =========================================================================
 * SECTION 5 — SSM LAYER
 *
 * One complete SSM layer. Contains all parameters needed for both forward
 * (inference and training) and backward (training only).
 *
 * Memory ownership:
 *   - All pointer fields are heap-allocated by model_alloc() in model.c
 *   - shadow_* and crystal_* fields are NULL in inference mode
 *   - csr_in / csr_out are rebuilt from *_int2 at load time and after repack
 *
 * Layer index k processes: in_proj → conv1d → SSM scan → out_proj → residual
 * ========================================================================= */

typedef struct {
    /* === TERNARY WEIGHTS (INT2 packed, inference + training) ============ */

    /* Input projection: d_model → d_state
     * Layout: row-major, row i = output neuron i in d_state
     * Byte count: V12_D_MODEL * V12_D_STATE / V12_INT2_WEIGHTS_PER_BYTE     */
    uint8_t *w_in_int2;         /* ALIGN32. INT2 packed W_in.                */

    /* Output projection: d_state → d_model
     * Byte count: V12_D_STATE * V12_D_MODEL / V12_INT2_WEIGHTS_PER_BYTE     */
    uint8_t *w_out_int2;        /* ALIGN32. INT2 packed W_out.               */

    /* Causal conv1d kernel: applies depthwise along d_state
     * Byte count: V12_CONV_KERNEL * V12_D_STATE / V12_INT2_WEIGHTS_PER_BYTE */
    uint8_t *w_conv_int2;       /* ALIGN32. INT2 packed conv kernel.         */

    /* === CSR INDICES (derived from INT2, rebuilt at load / repack) ====== */

    /* Pre-built CSR for fast inference-time matmul (no multiplications).
     * Heap-allocated by ternary_csr_init(). Freed by ternary_csr_free().    */
    TernaryCSR csr_in;          /* CSR for w_in: rows=d_state, cols=d_model  */
    TernaryCSR csr_out;         /* CSR for w_out: rows=d_model, cols=d_state */

    /* === SSM DYNAMICS (FP32, always active — not ternary) =============== */

    /* Diagonal SSM decay vector. Values in (0, 1) enforced by optimizer.
     * A[i] = exp(A_log[i]), stored directly as the decay coefficient.
     * Invariant: 0 < A[i] < 1 for all i (clamped after every optimizer step) */
    float   *A;                 /* ALIGN32. Shape: [d_state]                 */

    /* Input gate projection. Per-step: h[i] += B[i] * x[i]
     * Static (time-invariant) in the base architecture.                     */
    float   *B;                 /* Shape: [d_state]                          */

    /* Output projection matrix. y[t] = C @ h[t]
     * Shape: [d_state × d_state] because d_inner == d_state == 128 here.
     * (In the data flow, d_inner refers to the SSM inner dimension = d_state)*/
    float   *C;                 /* ALIGN32. Shape: [d_state × d_state]       */

    /* Output scaling vector for weight tying W_out = diag(sigma) @ W_in^T.
     * Forward: W_out @ y = W_in^T @ (sigma * y)
     * Learnable FP32, initialised to 1.0. Shape: [d_state]               */
    float   *sigma;             /* Shape: [d_state]                          */

    /* === INPUT-DEPENDENT A GATE (FP32, training) ======================== */

    /* A-gate projection matrices. Per-step: a_eff[t,i] = A[i] * tanh(W_A[i,:] @ x_conv[t])
     * W_A is FP32 [d_state × d_state], row i = projection for dimension i.
     * b_A is bias [d_state], initialized to 2.0 for near-saturated tanh gate at init. */
    float   *W_A;               /* ALIGN32. Shape: [d_state × d_state]       */
    float   *b_A;               /* Shape: [d_state], initialized to 2.0       */

    /* Shadow logits and gradient accumulators for W_A and b_A */
    shadow_t *shadow_w_A;       /* ALIGN32. Shape: [d_state × d_state]
                                 * R356 F2: shadow_t = float (flag OFF) or
                                 * int16_t (flag ON, codec via shadow_codec.h) */
    float   *dW_A;              /* Shape: [d_state × d_state]                */
    float   *db_A;              /* Shape: [d_state]                          */

    /* Adam moments for W_A and b_A */
    float   *m_W_A;             /* Shape: [d_state × d_state]                */
    float   *v_W_A;             /* Shape: [d_state × d_state]                */
    float   *m_b_A;             /* Shape: [d_state]                          */
    float   *v_b_A;             /* Shape: [d_state]                          */

    /* === MTP — MULTI-STEP PREDICTION MATRICES (FP32) =================== */

    /* Low-rank projection pair for future state estimation (innovation §4).
     * h_future = h ⊙ A^k + U @ (V^T @ h)
     * Only present in alternating layers (0, 2, 4). NULL in other layers.  */
    float   *mtp_U;             /* Shape: [d_state × V12_MTP_RANK]           */
    float   *mtp_V;             /* Shape: [d_state × V12_MTP_RANK]           */

    /* === SHADOW WEIGHTS (FP32 logits, training only — NULL in inference) = */

    /* Shadow logit s for W_in. MVUE gradient: Δs = -lr * dW * (w-s) * mask
     * These are the continuous parameters that W_in_int2 is quantized from. */
    shadow_t *shadow_w_in;      /* ALIGN32. Shape: [d_model × d_state]
                                 * R356 F2: shadow_t = float (flag OFF) /
                                 * int16_t (flag ON).                        */

    /* Shadow logit s for W_out.                                              */
    shadow_t *shadow_w_out;     /* ALIGN32. Shape: [d_state × d_model]       */

    /* Shadow logit s for conv kernel.                                        */
    shadow_t *shadow_w_conv;    /* Shape: [V12_CONV_KERNEL × d_state]        */

    /* === SAMPLED TERNARY WEIGHTS (current draw from MVUE distribution) == */

    /* The actual {-1, 0, +1} values used in the forward pass during training.
     * Resampled every V12_REPACK_INTERVAL steps from shadow logits.
     * Not needed at inference (use w_*_int2 directly via CSR).              */
    float   *sampled_w_in;      /* Shape: [d_model × d_state]                */
    float   *sampled_w_out;     /* Shape: [d_state × d_model]                */

    /* === CRYSTALLIZATION MASKS (training only — NULL in inference) ======= */

    /* 1 = crystallized (|logit| > V12_CRYSTALLIZATION_TAU), 0 = active.
     * Crystallized weights are skipped during optimizer_step.
     * After ~10K steps, 20–30% of weights crystallize (see arch §6).       */
    uint8_t *crystal_mask_in;   /* Shape: [d_model × d_state], one byte/weight */
    uint8_t *crystal_mask_out;  /* Shape: [d_state × d_model], one byte/weight */

    /* === EMA-SPRT GRADIENT ACCUMULATORS (replaces D4 sign filter) ========= */

    /* Per-weight EMA accumulator for sequential gradient evidence.
     * Each step: accum[i] = gamma*accum[i] + g_mvue_i
     * Update fires when |accum[i]| > EMA_SPRT_THRESHOLD; resets to 0 after.
     * gamma = V12_EMA_SPRT_GAMMA, threshold = V12_EMA_SPRT_THRESHOLD.        */
    float   *grad_accum_in;     /* Shape: [d_model × d_state]                */
    float   *grad_accum_out;    /* Shape: [d_state × d_model]                */

    /* === ALIVE INDEX (lazy optimizer — skip crystallized weights) ======== */

    /* Flat list of indices i where |shadow_w_in[i]| < V12_CRYSTALLIZATION_TAU.
     * Built/rebuilt every V12_REPACK_INTERVAL steps alongside crystal_mask_in.
     * Length n_alive_in <= V12_D_MODEL * V12_D_STATE.
     * NULL until first repack. Heap-allocated by optimizer_rebuild_alive_idx.   */
    int     *alive_idx_in;          /* Shape: [n_alive_in]. Flat indices into
                                      * shadow_w_in / dw_in / sampled_w_in.      */
    int      n_alive_in;            /* Current count of alive in-weights.         */

    /* Same for w_out [d_state × d_model]. */
    int     *alive_idx_out;         /* Shape: [n_alive_out].                      */
    int      n_alive_out;           /* Current count of alive out-weights.        */

    /* === GRADIENT ACCUMULATORS (zeroed after each optimizer_step) ======== */

    /* Accumulated ∂L/∂W_in over the current step. Zeroed by optimizer_step.
     * Used by compute_backward then consumed by optimizer_step.             */
    float   *dw_in;             /* Shape: [d_model × d_state]                */
    float   *dw_out;            /* Shape: [d_state × d_model]                */
    float   *dA;                /* Shape: [d_state]                          */
    float   *dB;                /* Shape: [d_state]                          */
    float   *dC;                /* Shape: [d_state × d_state]                */
    float   *d_sigma;           /* Shape: [d_state]. Gradient w.r.t. sigma.  */
    float   *d_conv_w;          /* Shape: [conv_kernel × d_state]            */
    float   *d_mtp_U;           /* Shape: [d_state × V12_MTP_RANK]. NULL on odd layers */
    float   *d_mtp_V;           /* Shape: [d_state × V12_MTP_RANK]. NULL on odd layers */

    /* Per-timestep B gradient buffer (Direct-B mode, [seq_len × d_state]).
     * ssm_scan_backward writes here when B_stride > 0.
     * Reduced to L->dB[d_state] by the Direct-B backward code in train.c.
     * NULL if Direct-B is disabled. */
    float   *dB_seq;            /* Shape: [V12_SEQ_LEN × d_state]. Training only. */

    /* === ADAM MOMENT BUFFERS (training only — NULL in inference) ======== */

    /* Factored Adam moments for w_in [d_model × d_state] */
    float   *m_w_in_row;        /* Shape: [d_model]. First moment row factor.  */
    float   *m_w_in_col;        /* Shape: [d_state]. First moment col factor.  */
    float   *v_w_in_row;        /* Shape: [d_model]. Second moment row factor. */
    float   *v_w_in_col;        /* Shape: [d_state]. Second moment col factor. */

    /* Factored Adam moments for w_out [d_state × d_model] */
    float   *m_w_out_row;       /* Shape: [d_state]. First moment row factor.  */
    float   *m_w_out_col;       /* Shape: [d_model]. First moment col factor.  */
    float   *v_w_out_row;       /* Shape: [d_state]. Second moment row factor. */
    float   *v_w_out_col;       /* Shape: [d_model]. Second moment col factor. */

    /* First and second moments for SSM dynamics A, B, C */
    float   *m_A;               /* Shape: [d_state]                          */
    float   *v_A;               /* Shape: [d_state]                          */
    float   *m_B;               /* Shape: [d_state]                          */
    float   *v_B;               /* Shape: [d_state]                          */
    float   *m_C;               /* Shape: [d_state × d_state]                */
    float   *v_C;               /* Shape: [d_state × d_state]                */

    /* Adam moments for sigma */
    float   *m_sigma;           /* Shape: [d_state]                          */
    float   *v_sigma;           /* Shape: [d_state]                          */

    /* First and second moments for conv1d kernel */
    float   *m_conv;            /* Shape: [conv_kernel × d_state]            */
    float   *v_conv;            /* Shape: [conv_kernel × d_state]            */

    /* First and second moments for MTP matrices (NULL on odd layers) */
    float   *m_mtp_U;           /* Shape: [d_state × V12_MTP_RANK]           */
    float   *v_mtp_U;           /* Shape: [d_state × V12_MTP_RANK]           */
    float   *m_mtp_V;           /* Shape: [d_state × V12_MTP_RANK]           */
    float   *v_mtp_V;           /* Shape: [d_state × V12_MTP_RANK]           */

    /* === CO-SPECTRAL SUBSPACE (rank-20 PCA, per-layer) ================== */

    CoSpectralSubspace cospec;  /* Calibrated after 50 warmup steps.         */

    /* === GP-KL MOMENT ACCUMULATORS (one per layer) ====================== */

    /* Welford accumulators updated by ssm_scan_forward on every forward call.
     * Persist across steps; used by compute_gp_kl_loss.                     */
    GpKlMoments moments;        /* Inline (not a pointer) — always present   */

    /* Bilinear recurrence gate: h[t] += C_bl[i] * h_prev[i] * x[i] (R348 F6, Volterra degree-2) */
    float   *C_bilinear;        /* [d_state] ternary gate weight                */
    shadow_t *shadow_C_bilinear; /* [d_state] shadow logits (R356 F2 quantizable) */
    float   *sampled_C_bilinear;/* [d_state] sampled ternary                    */
    float   *dC_bilinear;       /* [d_state] gradient accumulator               */
    float   *m_C_bilinear;      /* [d_state] Adam moment                        */
    float   *v_C_bilinear;      /* [d_state] Adam variance                      */

    /* Low-Rank Dyadic Ternary Perturbations (R495 Aspect A01) */
    float   *u;                 /* [d_state] ternary vector u                  */
    shadow_t *shadow_u;         /* [d_state] shadow logits for u               */
    float   *sampled_u;         /* [d_state] sampled ternary u                 */
    float   *du;                /* [d_state] gradient for u                    */
    float   *m_u;               /* [d_state] Adam moment for u                 */
    float   *v_u;               /* [d_state] Adam variance for u               */

    float   *v;                 /* [d_state] ternary vector v                  */
    shadow_t *shadow_v;         /* [d_state] shadow logits for v               */
    float   *sampled_v;         /* [d_state] sampled ternary v                 */
    float   *dv;                /* [d_state] gradient for v                    */
    float   *m_v;               /* [d_state] Adam moment for v                 */
    float   *v_v;               /* [d_state] Adam variance for v               */
    int moments_offloaded;      /* 1 = moments offloaded to host, 0 = active in GPU/VRAM */
    int escrow_bypass;          /* S-05: 1 = skip factored adam gradient updates */
} SSMLayer;


/* =========================================================================
 * SECTION 6 — MODEL
 *
 * The complete model. Owns all layers, embeddings, and norms.
 * Allocated once by model_alloc() and freed by model_free().
 *
 * In inference mode: embed_int2, lm_head_int2, layers[k].w_*_int2 are valid.
 * In training mode: additionally shadow_*, sampled_*, crystal_*, dw_* valid.
 * ========================================================================= */

typedef struct {
    /* === EMBEDDING TABLE ================================================ */

    /* Factored embedding (ALBERT-style): E @ P for memory savings.
     * Token t → E[t] (128-dim RNS) → P (128×1024 projection) → d_model vector.
     * embed_int2 holds materialized (E @ P) in INT2-packed form.
     * Byte count: V12_VOCAB_SIZE * V12_D_MODEL / V12_INT2_WEIGHTS_PER_BYTE  */
    uint8_t *embed_int2;        /* ALIGN32. [vocab_size × d_model / 4] bytes */

    /* CSR for embed lookup (used when lm_head weight-tying is active;
     * otherwise NULL — embedding lookup is a direct row select, not matmul) */
    TernaryCSR *csr_embed;      /* NULL in the default (no-tying) config     */

    /* Float embedding table for training (FP32 shadow).
     * E: RNS-initialized, shape [vocab_size × d_embed].
     * P: projection, shape [d_embed × d_model] (Xavier uniform init).
     * Forward: h[t] = E[t] @ P — no separate buffer; computed on-the-fly. */
    float   *embed_E;           /* ALIGN32. [vocab_size × d_embed]  RNS table */
    float   *embed_P;           /* ALIGN32. [d_embed × d_model]     projection */

    /* === EMBEDDING GRADIENTS AND ADAM MOMENTS (R348 F3) ================= */

    float   *d_embed_E;         /* Gradient w.r.t. embed_E [vocab_size × d_embed] */
    float   *d_embed_P;         /* Gradient w.r.t. embed_P [d_embed × d_model]    */
    float   *m_embed_E;
    int8_t  *sa_embed_E;         /* Canonical first moment for embed_E [vocab_size × d_embed] */
    float   *v_embed_E;         /* Canonical second moment for embed_E [vocab_size × d_embed] */
    float   *m_embed_E_row;     /* Legacy first moment row factor [vocab_size]              */
    float   *m_embed_E_col;     /* Legacy first moment col factor [d_embed]                 */
    float   *v_embed_E_row;     /* Legacy second moment row factor [vocab_size]             */
    float   *v_embed_E_col;     /* Legacy second moment col factor [d_embed]                */
    float   *m_embed_P;
    int8_t  *sa_embed_P;         /* First moment for embed_P [d_embed × d_model]   */
    float   *v_embed_P;         /* Second moment for embed_P [d_embed × d_model]  */

    /* === LANGUAGE MODEL HEAD ============================================= */

    /* lm_head maps d_model → vocab_size. The dominant inference cost.
     * Byte count: V12_D_MODEL * V12_VOCAB_SIZE / V12_INT2_WEIGHTS_PER_BYTE  */
    uint8_t *lm_head_int2;      /* ALIGN32. [d_model × vocab_size / 4] bytes */
    TernaryCSR csr_lm_head;     /* CSR for fast ternary matmul at lm_head    */

    /* Shadow, sampled, and gradient for lm_head (training only, NULL in inference)
     * Shape: [vocab_size × d_model] (full vocab, hot_mask guards cold tokens in optimizer) */
    shadow_t *shadow_lm_head;   /* Shape: [vocab_size × d_model] (R356 F2)   */
    float   *sampled_lm_head;   /* Shape: [vocab_size × d_model]             */
    float   *dlm_head_g_row;    /* Direct factored row gradients, [K_HOT] */
    float   *dlm_head_g_col;    /* Direct factored col gradients, [d_model] */
    float   *dlm_head_gq_row;   /* Direct factored row squared gradients, [K_HOT] */
    float   *dlm_head_gq_col;   /* Direct factored col squared gradients, [d_model] */
    int32_t *dlm_head_cnt_row;  /* Direct factored row update counts, [K_HOT] */
    int32_t *dlm_head_cnt_col;  /* Direct factored col update counts, [d_model] */

    /* Hot/cold token classification (training only, NULL in inference)      */
    uint8_t *hot_mask;          /* Shape: [vocab_size]. 1=hot, 0=cold        */
    int32_t *vocab_to_hot;      /* Shape: [vocab_size]. Hot index or -1      */
    int32_t *hot_to_vocab;      /* Shape: [K_HOT]. Vocab index for each hot  */

    /* === FINAL LAYER NORM ================================================ */

    /* RMS-norm or layer-norm applied to the final residual stream before
     * the lm_head projection. FP32, not ternary.                            */
    float   *norm_w;            /* Scale: shape [d_model]                    */
    float   *norm_b;            /* Bias:  shape [d_model]. NULL if RMS-only. */
    float   *d_norm_w;          /* Gradient accumulator for norm_w           */
    float   *d_norm_b;          /* Gradient accumulator for norm_b           */

    /* === PREDICTIVE CODING PROJECTION (pre-embedding surprise gate) ====== */

    /* W_pred: ternary rank-24 matrix for predictive coding surprise signal.
     * surprise[t] = embed[t] - W_pred @ h_prev
     * Only active at the input to layer 0.                                  */
    uint8_t *w_pred_int2;       /* Shape: [d_model × d_model / 4] bytes      */
    TernaryCSR csr_w_pred;      /* CSR for W_pred forward pass               */
    shadow_t *shadow_w_pred;    /* Training-only shadow logits (R356 F2)     */
    float   *dw_pred;           /* Gradient accumulator for W_pred           */
    uint8_t *crystal_mask_pred; /* Crystallization mask for W_pred          */

    /* Learnable MUX threshold (one per layer, scalar FP32)
     * Ternary multiplexing: W_eff = W ⊙ mask(x), mask = (|x| > tau_mux[k])
     * Initialized to 0.5, learned via SGD (not ternary — stays FP32).      */
    float    tau_mux[V12_N_LAYERS]; /* [n_layers], inline (not a pointer)    */

    /* Gradient accumulator for tau_mux. Zeroed each step.                   */
    float    d_tau_mux[V12_N_LAYERS]; /* [n_layers]                          */

    /* Sampled ternary W_pred for MVUE gradient computation.
     * Resampled every V12_REPACK_INTERVAL steps from shadow_w_pred.
     * Shape: [d_model × d_model]                                             */
    float   *sampled_w_pred;    /* Sampled ternary W_pred for MVUE gradient  */

    /* === ADAM MOMENT BUFFERS (model-level, training only) =============== */

    /* Factored Adam moments for lm_head [d_model × vocab_size] */
    float   *m_lm_head_row;     /* Shape: [K_HOT].       First moment row factor.  */
    float   *m_lm_head_col;     /* Shape: [vocab_size].  First moment col factor.  */
    float   *v_lm_head_row;     /* Shape: [K_HOT].       Second moment row factor. */
    float   *v_lm_head_col;     /* Shape: [vocab_size].  Second moment col factor. */

    /* Factored Adam moments for w_pred [d_model × d_model] */
    float   *m_w_pred_row;      /* Shape: [d_model]. First moment row factor.  */
    float   *m_w_pred_col;      /* Shape: [d_model]. First moment col factor.  */
    float   *v_w_pred_row;      /* Shape: [d_model]. Second moment row factor. */
    float   *v_w_pred_col;      /* Shape: [d_model]. Second moment col factor. */
    float   *m_norm_w;          /* Shape: [d_model]                          */
    float   *v_norm_w;          /* Shape: [d_model]                          */
    float   *m_norm_b;          /* Shape: [d_model]                          */
    float   *v_norm_b;          /* Shape: [d_model]                          */

    /* === SSM LAYERS ====================================================== */

    SSMLayer layers[V12_N_LAYERS];  /* All 6 layers, inline array            */

    /* === MODEL DIMENSIONS (runtime copy, must match compile-time consts) = */

    /* Stored explicitly so checkpoint functions can validate without
     * referencing compile-time defines from a different build.               */
    uint32_t n_layers;          /* Must equal V12_N_LAYERS (= 6)             */
    uint32_t d_model;           /* Must equal V12_D_MODEL  (= 1024)          */
    uint32_t d_state;           /* Must equal V12_D_STATE  (= 128)           */
    uint32_t vocab_size;        /* Must equal V12_VOCAB_SIZE (= 32064)       */
    uint32_t conv_kernel;       /* Must equal V12_CONV_KERNEL (= 4)          */
    uint32_t mtp_rank;          /* Must equal V12_MTP_RANK  (= 20)           */

    /* === GLOBAL STEP COUNTER ============================================= */

    uint32_t step;              /* Incremented at end of every train_step()  */

    /* === UNIGRAM PROPOSAL DISTRIBUTION (sampled softmax) ================= */

    /* Cumulative unigram frequency table for CDF-based sampling.
     * unigram_cdf[v] = sum_{i=0}^{v} freq[i] / total, normalized to [0,1].
     * Built once at model_alloc or after dataset scan via sampled_softmax_build_unigram().
     * Used by noise sampling in compute_sampled_ce_loss.
     * Size: vocab_size × 4B = 32064 × 4B = 0.128 MB. Negligible.         */
    float   *unigram_cdf;       /* Shape: [vocab_size]. For CDF-based sampling. */

    /* Log proposal probabilities: log(q[v]) for importance weight correction.
     * Computed alongside unigram_cdf. Used in sampled CE to correct bias.
     * Size: 0.128 MB. Negligible.                                           */
    float   *log_unigram;
    float   *token_freq; /* Q-06 */       /* Shape: [vocab_size]. For importance weights */

    /* === CORPUS LOGIT PRIOR ============================================== */

    /* Optional corpus bigram logit prior used as an additive sampled-CE log prior.
     * Runtime-only, explicit opt-in, and not persisted in checkpoints. The
     * prior stores sparse counts for observed (context token, target token)
     * pairs and backs off missing pairs to the corpus unigram probability. */
    int32_t  corpus_logit_prior_enabled;
    float    corpus_logit_prior_alpha;
    float    corpus_logit_prior_scale;
    uint32_t corpus_logit_prior_stride;
    uint64_t corpus_logit_prior_requested_terms;
    uint64_t corpus_logit_prior_actual_terms;
    uint32_t corpus_logit_prior_pair_capacity;
    uint32_t corpus_logit_prior_pair_unique;
    uint32_t *corpus_logit_prior_ctx_counts;  /* Shape: [vocab_size]. */
    uint32_t *corpus_logit_prior_pair_keys;   /* Open-addressed key=(ctx<<16)|target. */
    uint32_t *corpus_logit_prior_pair_counts; /* Shape: [pair_capacity]. */
    float    *corpus_logit_prior_unigram;     /* Shape: [vocab_size], raw probability. */

    /* === PROOF-CARRYING PROGRAM KERNEL ================================== */

    /* Optional deterministic local transition kernel. The table stores only
     * corpus-certified one-context/one-target transitions; conflicting or
     * under-supported contexts abstain and fall back to the neural residual. */
    int32_t  program_kernel_enabled;
    int32_t  program_kernel_gate_enabled;
    float    program_kernel_logit_boost;
    uint32_t program_kernel_min_count;
    uint32_t program_kernel_stride;
    uint64_t program_kernel_requested_terms;
    uint64_t program_kernel_actual_terms;
    uint32_t program_kernel_entries;
    uint32_t program_kernel_conflict_contexts;
    uint32_t *program_kernel_targets; /* Shape: [vocab_size], UINT32_MAX = abstain. */
    uint32_t *program_kernel_counts;  /* Shape: [vocab_size], supporting pair count. */

    float    program_kernel_gate_base_loss;
    float    program_kernel_gate_kernel_loss;
    float    program_kernel_gate_kernel_ratio;
    uint64_t program_kernel_gate_batches;
    uint64_t program_kernel_gate_routed;
    uint64_t program_kernel_gate_correct;
    uint64_t program_kernel_gate_incorrect;

    /* === CERTIFIED ABSTAINING ARTIFACT BOUNDARY ======================== */

    /* Optional R414 witness gate. The first proof-rule family is a
     * context-derived copy-current transducer: certified contexts emit the
     * current token itself only when the corpus verifier observed no
     * counterexample. It stores verifier counts and an obligation split, never
     * train-position targets. */
    int32_t  certified_artifact_boundary_gate_enabled;
    uint32_t certified_artifact_boundary_min_count;
    uint32_t certified_artifact_boundary_stride;
    uint64_t certified_artifact_boundary_requested_terms;
    uint64_t certified_artifact_boundary_actual_terms;
    uint32_t certified_artifact_boundary_candidate_contexts;
    uint32_t certified_artifact_boundary_certified_contexts;
    uint64_t certified_artifact_boundary_corpus_routed;
    uint64_t certified_artifact_boundary_corpus_abstain;
    uint64_t certified_artifact_boundary_corpus_false_routes;
    uint64_t certified_artifact_boundary_route_bytes;
    uint32_t *certified_artifact_boundary_counts;
    uint32_t *certified_artifact_boundary_false_counts;
    uint8_t  *certified_artifact_boundary_certified;
    float    certified_artifact_boundary_epsilon;
    float    certified_artifact_boundary_ce_budget;
    float    certified_artifact_boundary_overhead;
    float    certified_artifact_boundary_kappa;

    uint64_t certified_artifact_boundary_gate_batches;
    uint64_t certified_artifact_boundary_gate_positions;
    uint64_t certified_artifact_boundary_gate_routed;
    uint64_t certified_artifact_boundary_gate_correct;
    uint64_t certified_artifact_boundary_gate_incorrect;
    uint64_t certified_artifact_boundary_gate_abstain;
    float    certified_artifact_boundary_gate_b_k;
    float    certified_artifact_boundary_gate_false_route_rate;
    float    certified_artifact_boundary_gate_ce_residual;
    float    certified_artifact_boundary_gate_priced_ratio;
    int32_t  certified_artifact_boundary_obligation_audit;
    int32_t  certified_artifact_boundary_non_lookup_audit;

    /* === EXACT STATE TRANSITION ATLAS ==================================== */

    /* Optional corpus-owned exact transition atlas keyed by a bounded local
     * discrete state signature, currently (previous token, current token).
     * Conflicting or under-supported signatures abstain. */
    int32_t  state_transition_atlas_enabled;
    int32_t  state_transition_atlas_gate_enabled;
    float    state_transition_atlas_logit_boost;
    uint32_t state_transition_atlas_min_count;
    uint32_t state_transition_atlas_stride;
    uint64_t state_transition_atlas_requested_terms;
    uint64_t state_transition_atlas_actual_terms;
    uint32_t state_transition_atlas_capacity;
    uint32_t state_transition_atlas_entries;
    uint32_t state_transition_atlas_conflicts;
    uint32_t *state_transition_atlas_keys;    /* key=(prev<<16)|context. */
    uint32_t *state_transition_atlas_targets; /* UINT32_MAX = abstain/conflict. */
    uint32_t *state_transition_atlas_counts;

    float    state_transition_atlas_gate_base_loss;
    float    state_transition_atlas_gate_atlas_loss;
    float    state_transition_atlas_gate_atlas_ratio;
    uint64_t state_transition_atlas_gate_batches;
    uint64_t state_transition_atlas_gate_routed;
    uint64_t state_transition_atlas_gate_correct;
    uint64_t state_transition_atlas_gate_incorrect;

    /* === ABSORBED STATE-ATLAS RESIDUALIZATION GATE ====================== */

    /* Optional R410 pricing gate that treats the exact state atlas as an
     * absorbed transition component and measures the residual sampled-CE burden
     * left for ordinary training. It reuses the atlas real GPU path and does
     * not change the default objective unless explicitly enabled. */
    int32_t  absorbed_state_atlas_gate_enabled;
    float    absorbed_state_atlas_overhead;
    float    absorbed_state_atlas_memory_cost;
    float    absorbed_state_atlas_repair_cost;
    float    absorbed_state_atlas_kappa;
    float    absorbed_state_atlas_gate_base_loss;
    float    absorbed_state_atlas_gate_residual_loss;
    float    absorbed_state_atlas_gate_residual_ratio;
    float    absorbed_state_atlas_gate_coverage;
    float    absorbed_state_atlas_gate_route_accuracy;
    float    absorbed_state_atlas_gate_priced_ratio;
    uint64_t absorbed_state_atlas_gate_batches;
    uint64_t absorbed_state_atlas_gate_positions;
    uint64_t absorbed_state_atlas_gate_routed;
    uint64_t absorbed_state_atlas_gate_correct;
    uint64_t absorbed_state_atlas_gate_incorrect;

    /* === COUNTERFACTUAL MINIMAL-PAIR PACKETS ============================ */

    /* Optional corpus-owned local counterexample packet ledger. Each entry is
     * keyed by (previous token, current token) and stores the first two
     * distinct next-token targets observed with their support counts. The gate
     * measures paired coverage and balance on the real GPU training stream
     * without replacing original CE. */
    int32_t  counterfactual_packets_enabled;
    int32_t  counterfactual_packets_gate_enabled;
    uint32_t counterfactual_packets_min_count;
    uint32_t counterfactual_packets_stride;
    uint64_t counterfactual_packets_requested_terms;
    uint64_t counterfactual_packets_actual_terms;
    uint32_t counterfactual_packets_capacity;
    uint32_t counterfactual_packets_entries;
    uint32_t counterfactual_packets_candidate_states;
    uint64_t counterfactual_packets_other_targets;
    uint32_t *counterfactual_packets_keys;
    uint32_t *counterfactual_packets_target_a;
    uint32_t *counterfactual_packets_target_b;
    uint32_t *counterfactual_packets_count_a;
    uint32_t *counterfactual_packets_count_b;
    uint32_t *counterfactual_packets_count_other;

    uint64_t counterfactual_packets_gate_batches;
    uint64_t counterfactual_packets_gate_pairs;
    uint64_t counterfactual_packets_gate_primary;
    uint64_t counterfactual_packets_gate_secondary;
    uint64_t counterfactual_packets_gate_other;
    uint64_t counterfactual_packets_gate_abstain;
    float    counterfactual_packets_gate_coverage;
    float    counterfactual_packets_gate_balance;

    /* === CERTIFIED SILENT-SPAN TRAINING ================================= */

    /* Optional gate for span skipping. A token is silent only when original CE
     * certifies a bounded omitted logit-gradient norm. The gate records the
     * priced silent fraction but does not skip backward work until promotion. */
    int32_t  certified_silent_span_gate_enabled;
    float    certified_silent_span_max_loss;
    float    certified_silent_span_cert_cost;
    float    certified_silent_span_repair_cost;
    uint32_t certified_silent_span_min_len;
    uint64_t certified_silent_span_gate_batches;
    uint64_t certified_silent_span_gate_positions;
    uint64_t certified_silent_span_gate_tokens;
    uint64_t certified_silent_span_gate_spans;
    float    certified_silent_span_gate_fraction;
    float    certified_silent_span_gate_grad_bound;
    float    certified_silent_span_gate_priced_ratio;

    /* === ERROR-CAUSAL CURRICULUM / PREREQUISITE LATTICE ================ */

    /* Optional corpus-owned prerequisite atom ledger. The gate measures
     * whether live tokens can be ordered by observed prerequisite context
     * support while conserving target atom coverage. */
    int32_t  error_causal_lattice_enabled;
    int32_t  error_causal_lattice_gate_enabled;
    uint32_t error_causal_lattice_min_count;
    uint32_t error_causal_lattice_stride;
    uint64_t error_causal_lattice_requested_terms;
    uint64_t error_causal_lattice_actual_terms;
    uint32_t error_causal_lattice_target_atoms;
    uint32_t error_causal_lattice_prereq_atoms;
    uint32_t *error_causal_target_counts;
    uint32_t *error_causal_prereq_counts;

    uint64_t error_causal_lattice_gate_batches;
    uint64_t error_causal_lattice_gate_positions;
    uint64_t error_causal_lattice_gate_target_covered;
    uint64_t error_causal_lattice_gate_prereq_ready;
    uint64_t error_causal_lattice_gate_orderable;
    uint64_t error_causal_lattice_gate_rare_positions;
    uint64_t error_causal_lattice_gate_rare_orderable;
    float    error_causal_lattice_gate_target_coverage;
    float    error_causal_lattice_gate_prereq_readiness;
    float    error_causal_lattice_gate_orderable_fraction;
    float    error_causal_lattice_gate_rare_coverage;

    /* === UNBIASED RESIDUAL ACTIVE SAMPLING ============================== */

    /* Optional gate for loss-proportional active sampling. The gate measures
     * whether current CE residuals have enough effective sample support for an
     * unbiased importance-weighted sampler, while leaving original CE unchanged. */
    int32_t  residual_active_sampling_gate_enabled;
    float    residual_active_sampling_floor;
    float    residual_active_sampling_target_fraction;
    uint64_t residual_active_sampling_gate_batches;
    uint64_t residual_active_sampling_gate_positions;
    float    residual_active_sampling_gate_mean_loss;
    float    residual_active_sampling_gate_score_cv;
    float    residual_active_sampling_gate_ess_fraction;
    float    residual_active_sampling_gate_max_weight;
    float    residual_active_sampling_gate_variance_proxy;

    /* === EXACT LOCAL TRANSITION QUOTIENT ================================ */

    /* Optional exact multiplicity ledger for local transition micro-terms.
     * It counts exact (previous token, current token, target token) triples and
     * prices the unique/total quotient without changing original CE. */
    int32_t  exact_transition_quotient_enabled;
    int32_t  exact_transition_quotient_gate_enabled;
    uint32_t exact_transition_quotient_min_count;
    uint32_t exact_transition_quotient_stride;
    uint64_t exact_transition_quotient_requested_terms;
    uint64_t exact_transition_quotient_actual_terms;
    uint32_t exact_transition_quotient_capacity;
    uint32_t exact_transition_quotient_unique_terms;
    uint64_t exact_transition_quotient_supported_mass;
    uint64_t *exact_transition_quotient_keys;
    uint32_t *exact_transition_quotient_counts;

    uint64_t exact_transition_quotient_gate_batches;
    uint64_t exact_transition_quotient_gate_positions;
    uint64_t exact_transition_quotient_gate_supported;
    uint64_t exact_transition_quotient_gate_saved_terms;
    uint64_t exact_transition_quotient_gate_multiplicity_sum;
    float    exact_transition_quotient_gate_corpus_ratio;
    float    exact_transition_quotient_gate_live_coverage;
    float    exact_transition_quotient_gate_saved_fraction;
    float    exact_transition_quotient_gate_mean_multiplicity;

    /* === R421 MACRO CE QUOTIENT OBSERVER =============================== */

    /* Predictive quotient atoms over order-2 token contexts. Observer mode
     * only prices the quotient. Training mode applies quotient-unit gradient
     * weighting to covered CE rows in the real training path. */
    int32_t  macro_ce_quotient_observer_enabled;
    int32_t  macro_ce_quotient_training_enabled;
    uint32_t macro_ce_quotient_min_support;
    uint32_t macro_ce_quotient_stride;
    uint32_t macro_ce_quotient_signature_bins;
    uint32_t macro_ce_quotient_seed;
    uint64_t macro_ce_quotient_requested_terms;
    uint64_t macro_ce_quotient_actual_terms;
    float    macro_ce_quotient_build_fraction;
    float    macro_ce_quotient_alpha;
    float    macro_ce_quotient_residual_price_scale;
    float    macro_ce_quotient_overhead;
    float    macro_ce_quotient_training_weight;

    uint32_t macro_ce_quotient_context_capacity;
    uint32_t macro_ce_quotient_triple_capacity;
    uint32_t macro_ce_quotient_class_capacity;
    uint32_t macro_ce_quotient_class_target_capacity;
    uint32_t macro_ce_quotient_random_capacity;
    uint32_t macro_ce_quotient_random_target_capacity;
    uint32_t macro_ce_quotient_unique_contexts_raw;
    uint32_t macro_ce_quotient_unique_triples;
    uint32_t macro_ce_quotient_unique_contexts;
    uint32_t macro_ce_quotient_quotient_classes;
    uint32_t macro_ce_quotient_random_buckets;
    uint64_t *macro_ce_quotient_context_keys;
    uint32_t *macro_ce_quotient_context_totals;
    uint64_t *macro_ce_quotient_context_class_keys;
    uint16_t *macro_ce_quotient_context_top1_token;
    uint16_t *macro_ce_quotient_context_top2_token;
    uint32_t *macro_ce_quotient_context_top1_count;
    uint32_t *macro_ce_quotient_context_top2_count;
    uint64_t *macro_ce_quotient_triple_keys;
    uint32_t *macro_ce_quotient_triple_counts;
    uint64_t *macro_ce_quotient_class_keys;
    uint32_t *macro_ce_quotient_class_totals;
    uint64_t *macro_ce_quotient_class_target_keys;
    uint32_t *macro_ce_quotient_class_target_counts;
    uint64_t *macro_ce_quotient_random_keys;
    uint32_t *macro_ce_quotient_random_totals;
    uint64_t *macro_ce_quotient_random_target_keys;
    uint32_t *macro_ce_quotient_random_target_counts;

    uint64_t macro_ce_quotient_observer_batches;
    uint64_t macro_ce_quotient_observer_positions;
    uint64_t macro_ce_quotient_observer_covered;
    double   macro_ce_quotient_context_ce_sum;
    double   macro_ce_quotient_quotient_ce_sum;
    double   macro_ce_quotient_random_ce_sum;
    float    macro_ce_quotient_coverage;
    float    macro_ce_quotient_context_ce;
    float    macro_ce_quotient_quotient_ce;
    float    macro_ce_quotient_random_ce;
    float    macro_ce_quotient_ce_excess;
    float    macro_ce_quotient_random_advantage;
    float    macro_ce_quotient_ratio;
    float    macro_ce_quotient_residual_price;
    float    macro_ce_quotient_priced_work_ratio;
    float    macro_ce_quotient_net_speedup_estimate;
    uint64_t macro_ce_quotient_training_batches;
    uint64_t macro_ce_quotient_training_rows;
    uint64_t macro_ce_quotient_training_covered;
    float    macro_ce_quotient_training_coverage;
    float    macro_ce_quotient_training_effective_weight;

    /* === EXACT WINDOW-MULTISET SUFFICIENT STATISTIC ===================== */

    /* Optional R412 gate for exact model-visible window multiplicity. The
     * unit is strict: reset/position/full prefix/target must match exactly.
     * Hash hits are verified against stored corpus windows before counts are
     * accepted, so semantic or approximate duplicates never enter the gate. */
    int32_t  exact_window_multiset_gate_enabled;
    uint32_t exact_window_multiset_min_count;
    uint32_t exact_window_multiset_stride;
    uint64_t exact_window_multiset_requested_windows;
    uint64_t exact_window_multiset_actual_units;
    uint32_t exact_window_multiset_capacity;
    uint32_t exact_window_multiset_unique_units;
    uint64_t exact_window_multiset_duplicate_units;
    uint64_t exact_window_multiset_collision_count;
    uint64_t *exact_window_multiset_hashes;
    uint64_t *exact_window_multiset_starts;
    uint16_t *exact_window_multiset_positions;
    uint32_t *exact_window_multiset_counts;
    const uint16_t *exact_window_multiset_tokens;
    size_t   exact_window_multiset_n_tokens;
    float    exact_window_multiset_overhead;
    float    exact_window_multiset_noise_penalty;

    uint64_t exact_window_multiset_gate_batches;
    uint64_t exact_window_multiset_gate_positions;
    uint64_t exact_window_multiset_gate_supported;
    uint64_t exact_window_multiset_gate_saved_terms;
    uint64_t exact_window_multiset_gate_multiplicity_sum;
    float    exact_window_multiset_gate_q_ratio;
    float    exact_window_multiset_gate_live_coverage;
    float    exact_window_multiset_gate_saved_fraction;
    float    exact_window_multiset_gate_mean_multiplicity;
    float    exact_window_multiset_gate_priced_ratio;
    int32_t  exact_window_multiset_objective_audit;
    int32_t  exact_window_multiset_gradient_audit;

    /* === CE LOGIT CONTROL VARIATE GATE ================================== */

    /* Optional gate path for CE-equivalent residual-prior training. A fixed
     * corpus logit prior is added to residual logits while the shader also
     * records the baseline sampled CE, allowing H_res/H_0 pricing in the same
     * GPU pass without changing the objective family. */
    int32_t  logit_prior_gate_enabled;
    float    logit_prior_gate_base_loss;
    float    logit_prior_gate_residual_loss;
    float    logit_prior_gate_residual_ratio;
    uint64_t logit_prior_gate_batches;

    /* === RESIDUAL SHELL (R357 Alt 3) — forward-all, backward-masked ======
     *
     * RESIDUAL SHELL — OBJECTIVE IS REWEIGHTED, NOT UNCHANGED
     *
     * When use_residual_shell = 1, per-token gradient contributions are masked:
     * tokens whose CE loss falls outside (rs_eps_lo, rs_eps_hi) contribute zero
     * gradient. The surviving gradients are renormalized by 1/S where S = number
     * of surviving tokens, so the gradient magnitude matches the full-batch scale.
     *
     * This is NOT equivalent to full-gradient training. The model optimizes a
     * REWEIGHTED objective that ignores very-easy and very-hard tokens. The
     * converged optimum may differ.
     *
     * Default is OFF. Runtime-only, NOT persisted in checkpoints.
     */
    /* === PREFIX-FACTORIZED BACKWARD FRONTIER GATE ======================= */

    /* Optional R410 gate for recurrent frontier reuse. The table counts exact
     * local frontier signatures and the GPU path measures how much sampled-CE
     * adjoint energy lands on reusable frontiers. It is measurement-only until
     * the prefix backward cache is explicitly promoted. */
    int32_t  prefix_frontier_gate_enabled;
    uint32_t prefix_frontier_min_count;
    uint32_t prefix_frontier_stride;
    uint64_t prefix_frontier_requested_terms;
    uint64_t prefix_frontier_actual_terms;
    uint32_t prefix_frontier_capacity;
    uint32_t prefix_frontier_unique_terms;
    uint64_t prefix_frontier_supported_mass;
    float    prefix_frontier_overhead;
    float    prefix_frontier_repair_cost;
    uint32_t *prefix_frontier_keys;
    uint32_t *prefix_frontier_counts;

    uint64_t prefix_frontier_gate_batches;
    uint64_t prefix_frontier_gate_positions;
    uint64_t prefix_frontier_gate_supported;
    uint64_t prefix_frontier_gate_saved_terms;
    uint64_t prefix_frontier_gate_multiplicity_sum;
    double   prefix_frontier_gate_total_energy;
    double   prefix_frontier_gate_supported_energy;
    float    prefix_frontier_gate_reuse_fraction;
    float    prefix_frontier_gate_energy_fraction;
    float    prefix_frontier_gate_mean_multiplicity;
    float    prefix_frontier_gate_priced_ratio;

    /* === REVERSIBLE MACRO-TRANSCRIPT GATE ============================== */

    /* Optional R410 gate for exact reversible transcript compression. It
     * builds a corpus-owned adjacent-pair dictionary and measures the greedy
     * macro-token length ratio on the real training stream. */
    int32_t  macro_transcript_gate_enabled;
    uint32_t macro_transcript_min_count;
    uint32_t macro_transcript_stride;
    uint64_t macro_transcript_requested_terms;
    uint64_t macro_transcript_actual_terms;
    uint32_t macro_transcript_capacity;
    uint32_t macro_transcript_unique_pairs;
    uint64_t macro_transcript_supported_mass;
    uint32_t *macro_transcript_keys;
    uint32_t *macro_transcript_counts;
    float    macro_transcript_hardness;
    float    macro_transcript_codec_cost;
    float    macro_transcript_decode_cost;
    uint64_t macro_transcript_gate_batches;
    uint64_t macro_transcript_gate_positions;
    uint64_t macro_transcript_gate_ops;
    uint64_t macro_transcript_gate_pair_macros;
    uint64_t macro_transcript_gate_singletons;
    float    macro_transcript_gate_pair_fraction;
    float    macro_transcript_gate_length_ratio;
    float    macro_transcript_gate_priced_ratio;

    /* === REVERSIBLE DETERMINISTIC EXPANSION FACTORS ===================== */

    /* Optional R412 gate for fixed deterministic expansion factors. The
     * implemented transducer is intentionally narrow: repeated deterministic
     * whitespace/separator runs are represented by a seed token and length,
     * then verified by byte-for-byte inverse expansion. */
    int32_t  reversible_deterministic_expansion_gate_enabled;
    uint32_t reversible_deterministic_expansion_min_run;
    uint64_t reversible_deterministic_expansion_requested_terms;
    uint64_t reversible_deterministic_expansion_actual_terms;
    float    reversible_deterministic_expansion_overhead;
    float    reversible_deterministic_expansion_noise_penalty;
    uint64_t reversible_deterministic_expansion_corpus_tokens;
    uint64_t reversible_deterministic_expansion_corpus_covered;
    uint64_t reversible_deterministic_expansion_corpus_spans;
    uint64_t reversible_deterministic_expansion_inverse_failures;

    uint64_t reversible_deterministic_expansion_gate_batches;
    uint64_t reversible_deterministic_expansion_gate_positions;
    uint64_t reversible_deterministic_expansion_gate_covered;
    uint64_t reversible_deterministic_expansion_gate_spans;
    uint64_t reversible_deterministic_expansion_gate_coord_units;
    float    reversible_deterministic_expansion_gate_lambda;
    float    reversible_deterministic_expansion_gate_eta;
    float    reversible_deterministic_expansion_gate_priced_ratio;
    int32_t  reversible_deterministic_expansion_inverse_audit;

    /* === CE-SUFFICIENT ACTIVE STATE SUBSPACE GATE ======================= */

    /* Optional R410 gate for training only the CE-active state coordinates.
     * It measures live GPU gradient energy after RMSNorm backward and prices
     * the smallest configured top-k subspace against reversible-tail regret. */
    int32_t  active_state_subspace_gate_enabled;
    uint32_t active_state_subspace_k;
    float    active_state_subspace_overhead;
    uint64_t active_state_subspace_gate_batches;
    uint64_t active_state_subspace_gate_positions;
    double   active_state_subspace_gate_total_energy;
    double   active_state_subspace_gate_topk_energy;
    float    active_state_subspace_gate_k_over_d;
    float    active_state_subspace_gate_energy_fraction;
    float    active_state_subspace_gate_tail_regret;
    float    active_state_subspace_gate_priced_ratio;

    /* === BOUNDARY LIABILITY LEDGER GATE ================================= */

    /* Optional R417 gate. It measures original-CE gradient energy at reset
     * boundaries and records only certified gradient-null boundary rows as
     * removable candidates. The default training objective is unchanged. */
    int32_t  boundary_liability_ledger_gate_enabled;
    float    boundary_liability_null_epsilon;
    float    boundary_liability_overhead;
    uint64_t boundary_liability_gate_batches;
    uint64_t boundary_liability_gate_positions;
    uint64_t boundary_liability_gate_boundary_positions;
    uint64_t boundary_liability_gate_null_positions;
    uint64_t boundary_liability_gate_boundary_null_positions;
    double   boundary_liability_gate_total_energy;
    double   boundary_liability_gate_boundary_energy;
    double   boundary_liability_gate_null_energy;
    double   boundary_liability_gate_boundary_null_energy;
    float    boundary_liability_gate_boundary_fraction;
    float    boundary_liability_gate_boundary_energy_fraction;
    float    boundary_liability_gate_null_fraction;
    float    boundary_liability_gate_boundary_null_fraction;
    float    boundary_liability_gate_false_discharge_energy;
    float    boundary_liability_gate_priced_gain;
    int32_t  boundary_liability_gate_pass;

    /* === CE BOUNDARY-FLUX SHEAF GATE =================================== */

    /* Optional R415/R417 gate over live SSM transition-credit flux. It
     * measures whether reset-boundary endpoint rows carry enough transition
     * credit energy to justify a richer boundary-message witness. */
    int32_t  boundary_flux_sheaf_gate_enabled;
    float    boundary_flux_sheaf_overhead;
    uint64_t boundary_flux_gate_batches;
    uint64_t boundary_flux_gate_layers;
    uint64_t boundary_flux_gate_positions;
    uint64_t boundary_flux_gate_boundary_positions;
    uint64_t boundary_flux_gate_pre_boundary_positions;
    uint64_t boundary_flux_gate_endpoint_positions;
    double   boundary_flux_gate_total_energy;
    double   boundary_flux_gate_boundary_energy;
    double   boundary_flux_gate_pre_boundary_energy;
    double   boundary_flux_gate_endpoint_energy;
    float    boundary_flux_gate_boundary_fraction;
    float    boundary_flux_gate_pre_boundary_fraction;
    float    boundary_flux_gate_endpoint_fraction;
    float    boundary_flux_gate_energy_fraction;
    float    boundary_flux_gate_kappa_est;
    float    boundary_flux_gate_priced_gain;
    int32_t  boundary_flux_gate_identifying;
    int32_t  boundary_flux_gate_pass;

    /* === ONLINE TSSI LIVE GPU GATE ===================================== */

    /* Optional R420 audit over the live GPU training path. It proves whether
     * SSM hidden states and sampled CE residual scores can feed a compact
     * target-conditioned per-state accumulator without exporting a hidden
     * trace to CPU. */
    int32_t  online_tssi_live_gate_enabled;
    uint64_t online_tssi_live_gate_batches;
    uint64_t online_tssi_live_gate_state_layers;
    uint64_t online_tssi_live_gate_moment_layers;
    uint64_t online_tssi_live_gate_score_layers;
    uint64_t online_tssi_live_gate_score_updates;
    uint64_t online_tssi_live_gate_sampled_ce_buffers;
    uint64_t online_tssi_live_gate_missing_state_score_layers;
    float    online_tssi_live_gate_state_layer_fraction;
    float    online_tssi_live_gate_moment_layer_fraction;
    float    online_tssi_live_gate_score_layer_fraction;
    int32_t  online_tssi_live_gate_identifying;
    int32_t  online_tssi_live_gate_pass;

    /* === BOUNDARY-COTANGENT CREDIT GATE ================================ */

    /* Optional R416 audit over live SSM transition-credit cotangents. It
     * reconstructs exact per-step transition credit from compact boundary
     * anchors, then prices residual CE descent and verification work. */
    int32_t  boundary_cotangent_credit_gate_enabled;
    float    boundary_cotangent_overhead;
    float    boundary_cotangent_residual_budget;
    float    boundary_cotangent_verify_fraction;
    uint64_t boundary_cotangent_gate_batches;
    uint64_t boundary_cotangent_gate_layers;
    uint64_t boundary_cotangent_gate_positions;
    uint64_t boundary_cotangent_gate_anchor_positions;
    double   boundary_cotangent_gate_exact_energy;
    double   boundary_cotangent_gate_recon_energy;
    double   boundary_cotangent_gate_dot;
    double   boundary_cotangent_gate_residual_energy;
    float    boundary_cotangent_gate_anchor_fraction;
    float    boundary_cotangent_gate_saved_recurrent_fraction;
    float    boundary_cotangent_gate_cosine;
    float    boundary_cotangent_gate_residual_fraction;
    float    boundary_cotangent_gate_energy_covered;
    float    boundary_cotangent_gate_basis_bytes_per_token;
    float    boundary_cotangent_gate_verification_fraction;
    float    boundary_cotangent_gate_priced_ratio;
    float    boundary_cotangent_gate_unverified_priced_ratio;
    float    boundary_cotangent_gate_descent_per_paid_work;
    float    boundary_cotangent_gate_unverified_descent_per_paid_work;
    int32_t  boundary_cotangent_gate_identifying;
    int32_t  boundary_cotangent_gate_pass;

    /* === VARIANCE-PRICED CREDIT ALLOCATION GATE ======================== */

    /* Optional R416 audit over transition-credit items. It scores live credit
     * rows by norm, simulates an unbiased sampled update, and prices variance,
     * scoring, verification, and sampled work against exact original CE. */
    int32_t  variance_priced_credit_gate_enabled;
    float    variance_priced_sample_fraction;
    float    variance_priced_score_fraction;
    float    variance_priced_verify_fraction;
    float    variance_priced_overhead;
    uint64_t variance_priced_gate_batches;
    uint64_t variance_priced_gate_layers;
    uint64_t variance_priced_gate_items;
    uint64_t variance_priced_gate_sampled_items;
    double   variance_priced_gate_norm_sum;
    double   variance_priced_gate_norm_sq_sum;
    double   variance_priced_gate_exact_energy;
    double   variance_priced_gate_sampled_energy;
    double   variance_priced_gate_dot;
    double   variance_priced_gate_residual_energy;
    float    variance_priced_gate_sampled_fraction;
    float    variance_priced_gate_norm_ess_fraction;
    float    variance_priced_gate_mean_norm;
    float    variance_priced_gate_max_norm;
    float    variance_priced_gate_alignment;
    float    variance_priced_gate_variance_proxy;
    float    variance_priced_gate_priced_work;
    float    variance_priced_gate_descent_proxy;
    float    variance_priced_gate_descent_per_paid_work;
    int32_t  variance_priced_gate_identifying;
    int32_t  variance_priced_gate_pass;
    int32_t  equal_area_fisher_sampling_enabled;
    int32_t  equal_area_fisher_sampling_gate_enabled;
    float    equal_area_fisher_ess_floor;
    float    equal_area_fisher_bias_budget;
    float    equal_area_fisher_min_descent_per_paid_work;
    int32_t  amortized_ce_martingale_training_enabled;
    float    amortized_ce_martingale_min_descent_per_paid_work;
    uint64_t amortized_ce_martingale_training_layers;
    uint64_t amortized_ce_martingale_applied_layers;
    uint64_t amortized_ce_martingale_fallback_layers;
    float    amortized_ce_martingale_last_descent_per_paid_work;
    float    amortized_ce_martingale_last_sampled_fraction;
    int32_t  amortized_ce_martingale_last_applied;

    /* === EVENT-INNOVATION CREDIT ======================================= */

    /* Optional A10-C03 event-time credit path. Event scores are computed from
     * forward-owned state innovation and gate curvature, then audited against
     * exact dA_per_step credit. The default path is unchanged. */
    int32_t  event_innovation_credit_enabled;
    int32_t  event_innovation_credit_gate_enabled;
    float    event_innovation_fraction;
    float    event_innovation_debt_budget;
    float    event_innovation_score_cost;
    float    event_innovation_verify_fraction;
    float    event_innovation_overhead;
    float    event_innovation_low_event_scale;
    float    event_innovation_boundary_boost;
    uint64_t event_innovation_gate_batches;
    uint64_t event_innovation_gate_layers;
    uint64_t event_innovation_gate_applied_layers;
    uint64_t event_innovation_gate_positions;
    uint64_t event_innovation_gate_event_positions;
    double   event_innovation_gate_score_sum;
    float    event_innovation_gate_max_score;
    double   event_innovation_gate_exact_energy;
    double   event_innovation_gate_event_energy;
    double   event_innovation_gate_low_event_energy;
    float    event_innovation_gate_event_fraction;
    float    event_innovation_gate_low_event_debt;
    float    event_innovation_gate_energy_covered;
    float    event_innovation_gate_priced_ratio;
    float    event_innovation_gate_descent_per_paid_work;
    int32_t  event_innovation_gate_identifying;
    int32_t  event_innovation_gate_pass;

    /* === CE-DEBT SKIP LEDGER ========================================== */

    /* Optional A8-C02 CE-debt skip path. It certifies low-loss/low-gradient
     * token obligations before sampled-CE scatter, audits skipped debt, and
     * only attenuates certified rows when the original-CE debt price passes.
     * The default path is unchanged. */
    int32_t  ce_debt_skip_enabled;
    int32_t  ce_debt_skip_gate_enabled;
    float    ce_debt_max_loss;
    float    ce_debt_max_grad_norm;
    float    ce_debt_max_debt;
    float    ce_debt_audit_fraction;
    float    ce_debt_overhead;
    float    ce_debt_min_skip_fraction;
    float    ce_debt_skip_scale;
    uint64_t ce_debt_gate_batches;
    uint64_t ce_debt_gate_tokens;
    uint64_t ce_debt_gate_certified_tokens;
    uint64_t ce_debt_gate_skipped_tokens;
    uint64_t ce_debt_gate_audited_tokens;
    uint64_t ce_debt_gate_applied_batches;
    double   ce_debt_gate_total_loss;
    double   ce_debt_gate_certified_loss;
    double   ce_debt_gate_total_grad_energy;
    double   ce_debt_gate_certified_grad_energy;
    float    ce_debt_gate_certified_fraction;
    float    ce_debt_gate_skipped_fraction;
    float    ce_debt_gate_loss_debt;
    float    ce_debt_gate_grad_debt;
    float    ce_debt_gate_debt;
    float    ce_debt_gate_energy_covered;
    float    ce_debt_gate_priced_ratio;
    float    ce_debt_gate_descent_per_paid_work;
    int32_t  ce_debt_gate_identifying;
    int32_t  ce_debt_gate_pass;

    /* === GRADIENT-COHERENCE TRANSITION QUOTIENT ======================= */

    /* Optional A8-C03 transition-credit quotient path. It groups recurrent
     * dA_per_step rows by cheap transition-state signatures, audits the
     * representative/residual decomposition, and only coalesces rows when the
     * residual-priced denominator passes. The default path is unchanged. */
    int32_t  gradient_coherence_quotient_enabled;
    int32_t  gradient_coherence_quotient_gate_enabled;
    uint32_t gradient_coherence_bins;
    uint32_t gradient_coherence_probe_stride;
    float    gradient_coherence_max_residual;
    float    gradient_coherence_score_cost;
    float    gradient_coherence_verify_fraction;
    float    gradient_coherence_overhead;
    float    gradient_coherence_min_saved_fraction;
    float    gradient_coherence_residual_scale;
    uint64_t gradient_coherence_gate_batches;
    uint64_t gradient_coherence_gate_layers;
    uint64_t gradient_coherence_gate_applied_layers;
    uint64_t gradient_coherence_gate_positions;
    uint64_t gradient_coherence_gate_nonempty_clusters;
    uint64_t gradient_coherence_gate_residual_replay_positions;
    double   gradient_coherence_gate_exact_energy;
    double   gradient_coherence_gate_representative_energy;
    double   gradient_coherence_gate_residual_energy;
    float    gradient_coherence_gate_avg_cluster_size;
    float    gradient_coherence_gate_representative_fraction;
    float    gradient_coherence_gate_residual_replay_fraction;
    float    gradient_coherence_gate_residual_debt;
    float    gradient_coherence_gate_energy_covered;
    float    gradient_coherence_gate_saved_fraction;
    float    gradient_coherence_gate_priced_ratio;
    float    gradient_coherence_gate_descent_per_paid_work;
    int32_t  gradient_coherence_gate_identifying;
    int32_t  gradient_coherence_gate_pass;

    /* === PAPER 7 DF-GA REAL-BATCH AUDIT =============================== */

    /* Optional read-only audit for Direct Factored Gradient Accumulation.
     * It materializes the sampled/hot-row LM-head gradient support only for
     * measurement, then compares dense marginals against the direct streaming
     * row/column statistics used by the current v12 path. */
    int32_t  paper7_dfga_real_batch_audit_enabled;
    uint32_t paper7_dfga_real_batch_audit_max_batches;
    float    paper7_dfga_real_batch_audit_residual_budget;
    uint64_t paper7_dfga_real_batch_audit_batches;

    /* === LAGGED CE-IMPULSE ELIGIBILITY LEDGER ========================= */

    /* Optional R438 A1 real training bridge. It reconstructs recurrent
     * transition credit from lagged CE hidden impulses and local transition
     * products, compares the ledger against exact dA_per_step/dA/dbA/dWA, and
     * only blends it into the real A/bA/WA recurrent update when the live GPU
     * residual and priced denominator tests pass. */
    int32_t  lagged_ce_impulse_eligibility_enabled;
    uint32_t lagged_ce_impulse_horizon;
    float    lagged_ce_impulse_blend;
    float    lagged_ce_impulse_residual_budget;
    float    lagged_ce_impulse_local_jvp_fraction;
    float    lagged_ce_impulse_ring_fraction;
    float    lagged_ce_impulse_tail_audit_fraction;
    float    lagged_ce_impulse_memory_fraction;
    uint64_t lagged_ce_impulse_batches;
    uint64_t lagged_ce_impulse_layers;
    uint64_t lagged_ce_impulse_positions;
    double   lagged_ce_impulse_exact_transition_energy;
    double   lagged_ce_impulse_ledger_transition_energy;
    double   lagged_ce_impulse_transition_dot;
    double   lagged_ce_impulse_transition_residual_energy;
    double   lagged_ce_impulse_exact_dA_energy;
    double   lagged_ce_impulse_ledger_dA_energy;
    double   lagged_ce_impulse_dA_dot;
    double   lagged_ce_impulse_dA_residual_energy;
    double   lagged_ce_impulse_exact_dbA_energy;
    double   lagged_ce_impulse_ledger_dbA_energy;
    double   lagged_ce_impulse_dbA_dot;
    double   lagged_ce_impulse_dbA_residual_energy;
    double   lagged_ce_impulse_exact_dWA_energy;
    double   lagged_ce_impulse_ledger_dWA_energy;
    double   lagged_ce_impulse_dWA_dot;
    double   lagged_ce_impulse_dWA_residual_energy;
    uint32_t lagged_ce_impulse_dWA_sampled_columns;
    float    lagged_ce_impulse_transition_residual_fraction;
    float    lagged_ce_impulse_dA_residual_fraction;
    float    lagged_ce_impulse_dbA_residual_fraction;
    float    lagged_ce_impulse_dWA_residual_fraction;
    float    lagged_ce_impulse_cosine;
    float    lagged_ce_impulse_priced_ratio;
    float    lagged_ce_impulse_observed_rho;
    float    lagged_ce_impulse_tail_bound_fraction;
    uint64_t lagged_ce_impulse_training_layers;
    uint64_t lagged_ce_impulse_applied_layers;
    uint64_t lagged_ce_impulse_fallback_layers;
    int32_t  lagged_ce_impulse_identifying;
    int32_t  lagged_ce_impulse_pass;
    int32_t  lagged_ce_impulse_last_applied;

    /* === R438 EXECUTABLE MULTIPLICITY PACKETS ========================= */

    /* Optional B1 real GPU-path observer. It hashes live recurrent state and
     * transition signatures, measures how much training mass could be grouped
     * into executable packets, and prices representative-packet work plus
     * residual liquidation before any speed claim. */
    int32_t  executable_multiplicity_packets_enabled;
    uint32_t executable_multiplicity_min_count;
    float    executable_multiplicity_min_mass;
    float    executable_multiplicity_max_residual;
    float    executable_multiplicity_signature_fraction;
    float    executable_multiplicity_verifier_fraction;
    float    executable_multiplicity_memory_fraction;
    float    executable_multiplicity_state_scale;
    float    executable_multiplicity_transition_scale;
    uint64_t executable_multiplicity_batches;
    uint64_t executable_multiplicity_layers;
    uint64_t executable_multiplicity_positions;
    uint64_t executable_multiplicity_cells;
    uint64_t executable_multiplicity_accepted_cells;
    uint64_t executable_multiplicity_accepted_positions;
    uint64_t executable_multiplicity_rejected_positions;
    double   executable_multiplicity_exact_energy;
    double   executable_multiplicity_residual_energy;
    float    executable_multiplicity_accepted_mass;
    float    executable_multiplicity_residual_fraction;
    float    executable_multiplicity_mean_multiplicity;
    float    executable_multiplicity_packet_work_fraction;
    float    executable_multiplicity_priced_ratio;
    int32_t  executable_multiplicity_identifying;
    int32_t  executable_multiplicity_pass;

    /* === ORIGINAL-CE CREDIT-DEBT LOCAL UPDATE OBSERVER ================= */

    /* Optional R422 observer over the live v12 recurrent-credit path. It
     * compares exact transition-credit rows against the local direct
     * transition update induced by the same-step SSM output cotangent. The
     * default training objective and optimizer are unchanged. */
    int32_t  original_ce_credit_debt_observer_enabled;
    float    original_ce_credit_surface_fraction;
    float    original_ce_credit_local_work_multiplier;
    float    original_ce_credit_debt_price;
    float    original_ce_credit_verifier_fraction;
    float    original_ce_credit_overhead;
    float    original_ce_credit_recurrent_backward_share;
    float    original_ce_credit_min_progress;
    float    original_ce_credit_max_residual;
    float    original_ce_credit_min_whole_step_speedup;
    uint64_t original_ce_credit_gate_batches;
    uint64_t original_ce_credit_gate_layers;
    uint64_t original_ce_credit_gate_positions;
    uint64_t original_ce_credit_gate_selected_positions;
    double   original_ce_credit_gate_exact_energy;
    double   original_ce_credit_gate_local_energy;
    double   original_ce_credit_gate_local_dot;
    double   original_ce_credit_gate_local_residual_energy;
    double   original_ce_credit_gate_selected_exact_energy;
    double   original_ce_credit_gate_selected_local_energy;
    double   original_ce_credit_gate_selected_dot;
    double   original_ce_credit_gate_selected_residual_energy;
    float    original_ce_credit_gate_selected_fraction;
    float    original_ce_credit_gate_local_progress_ratio;
    float    original_ce_credit_gate_local_residual_fraction;
    float    original_ce_credit_gate_local_priced_ratio;
    float    original_ce_credit_gate_local_whole_step_speedup;
    float    original_ce_credit_gate_surface_progress_ratio;
    float    original_ce_credit_gate_surface_residual_fraction;
    float    original_ce_credit_gate_surface_priced_ratio;
    float    original_ce_credit_gate_surface_whole_step_speedup;
    int32_t  original_ce_credit_gate_identifying;
    int32_t  original_ce_credit_gate_local_pass;
    int32_t  original_ce_credit_gate_surface_pass;

    /* === R375 PB-G2 KRONECKER-SHAPE REAL-GRADIENT GATE ================= */

    /* Optional diagnostic for the load-bearing PB-G2 question: do real v12
     * ternary-gradient matrices contain enough non-scalar Kronecker shape for
     * a Shampoo-style left/right preconditioner to matter after MVUE shadowing?
     * This is read-only telemetry and never changes the optimizer path. */
    int32_t  r375_pbg2_gate_enabled;
    uint32_t r375_pbg2_max_matrices;
    float    r375_pbg2_min_anisotropy;
    uint64_t r375_pbg2_gate_batches;
    uint64_t r375_pbg2_gate_matrices;
    uint64_t r375_pbg2_gate_elements;
    double   r375_pbg2_gate_grad_energy;
    double   r375_pbg2_gate_row_anisotropy_sum;
    double   r375_pbg2_gate_col_anisotropy_sum;
    double   r375_pbg2_gate_shape_signal_sum;
    double   r375_pbg2_gate_mvue_active_sum;
    double   r375_pbg2_gate_grid_snr_sum;
    float    r375_pbg2_gate_row_anisotropy;
    float    r375_pbg2_gate_col_anisotropy;
    float    r375_pbg2_gate_shape_signal;
    float    r375_pbg2_gate_mvue_active_fraction;
    float    r375_pbg2_gate_grid_snr;
    float    r375_pbg2_gate_max_shape_signal;
    int32_t  r375_pbg2_gate_identifying;
    int32_t  r375_pbg2_gate_pass;

    /* Optional R375 PB-G2 prototype path. It preconditions GPU-resident
     * W_in/W_out gradients with a diagonal row/column Kronecker-RMS proxy
     * before the existing factored ternary optimizer consumes them. */
    int32_t  r375_pbg2_precondition_enabled;
    uint32_t r375_pbg2_precondition_max_matrices;
    float    r375_pbg2_precondition_eps;
    uint64_t r375_pbg2_precondition_batches;
    uint64_t r375_pbg2_precondition_matrices;
    uint64_t r375_pbg2_precondition_elements;
    double   r375_pbg2_precondition_raw_rms_sum;
    double   r375_pbg2_precondition_pre_rms_sum;
    double   r375_pbg2_precondition_scale_sum;
    double   r375_pbg2_precondition_cosine_sum;
    float    r375_pbg2_precondition_raw_rms;
    float    r375_pbg2_precondition_pre_rms;
    float    r375_pbg2_precondition_scale;
    float    r375_pbg2_precondition_cosine;
    int32_t  r375_pbg2_precondition_identifying;

    /* === R388 FLIP-EVENT SPARSE REFRESH GATE =========================== */

    /* Optional diagnostic for block-level refresh omission. The gate compares
     * deployed ternary sampled weights immediately before and after the real
     * Vulkan MVUE resample at a refresh boundary. A block is omittable only
     * when its deployed ternary code is exactly unchanged, preserving the
     * original CE path for any future sparse-refresh scheduler. */
    int32_t  r388_flip_refresh_gate_enabled;
    uint32_t r388_flip_refresh_block_elements;
    float    r388_flip_refresh_min_eta;
    uint64_t r388_flip_refresh_gate_batches;
    uint64_t r388_flip_refresh_matrices;
    uint64_t r388_flip_refresh_elements;
    uint64_t r388_flip_refresh_blocks;
    uint64_t r388_flip_refresh_omittable_blocks;
    uint64_t r388_flip_refresh_flips;
    float    r388_flip_refresh_eta;
    float    r388_flip_refresh_flip_fraction;
    float    r388_flip_refresh_projected_refresh_gain;
    int32_t  r388_flip_refresh_identifying;
    int32_t  r388_flip_refresh_pass;

    /* === CERTIFIED GRADIENT-NULL WITNESS GATE ========================== */

    /* Optional R415 gate over full original-CE parameter-gradient tensors.
     * It conservatively discharges only entire gradient groups whose RMS is
     * below a certificate epsilon, then audits omitted gradient energy. */
    int32_t  gradient_null_witness_gate_enabled;
    float    gradient_null_witness_epsilon;
    float    gradient_null_witness_omitted_budget;
    float    gradient_null_witness_overhead;
    uint64_t gradient_null_gate_batches;
    uint64_t gradient_null_gate_groups;
    uint64_t gradient_null_gate_null_groups;
    uint64_t gradient_null_gate_elements;
    uint64_t gradient_null_gate_null_elements;
    uint64_t gradient_null_gate_work_units;
    uint64_t gradient_null_gate_null_work_units;
    double   gradient_null_gate_total_energy;
    double   gradient_null_gate_null_energy;
    float    gradient_null_gate_group_fraction;
    float    gradient_null_gate_element_fraction;
    float    gradient_null_gate_saved_work_fraction;
    float    gradient_null_gate_omitted_fraction;
    float    gradient_null_gate_priced_gain;
    int32_t  gradient_null_gate_identifying;
    int32_t  gradient_null_gate_pass;

    /* === R429 DECISION-EQUIVALENCE WITNESS ============================ */

    /* Optional R429 native witness. It never removes work directly in this
     * observer form: it certifies zero-gradient ternary groups as exact
     * no-update decisions and sends every uncertified group to the full path.
     */
    int32_t  r429_decision_equivalence_witness_enabled;
    uint64_t r429_decision_witness_batches;
    uint64_t r429_decision_witness_denominator_groups;
    uint64_t r429_decision_witness_certified_groups;
    uint64_t r429_decision_witness_fallback_groups;
    uint64_t r429_decision_witness_avoided_work_units;
    uint64_t r429_decision_witness_mismatch_updates;
    float    r429_decision_witness_certified_fraction;
    float    r429_decision_witness_fallback_fraction;
    float    r429_decision_witness_avoided_work_fraction;
    int32_t  r429_decision_witness_identifying;
    int32_t  r429_decision_witness_pass;

    /* === CE-NORMAL TERNARY CHAMBER GATE =============================== */

    /* Optional R416 force-bound audit over GPU-owned ternary shadow cells.
     * It separates stochastic cells, already-crystallized cells, and
     * deterministic-but-not-crystallized chamber cells before pricing any
     * no-crossing work-removal claim. */
    int32_t  ce_normal_chamber_gate_enabled;
    float    ce_normal_chamber_sigma;
    float    ce_normal_chamber_overhead;
    uint64_t ce_normal_chamber_gate_batches;
    uint64_t ce_normal_chamber_gate_groups;
    uint64_t ce_normal_chamber_gate_elements;
    uint64_t ce_normal_chamber_gate_stochastic_elements;
    uint64_t ce_normal_chamber_gate_saturated_elements;
    uint64_t ce_normal_chamber_gate_crystallized_elements;
    uint64_t ce_normal_chamber_gate_candidate_elements;
    uint64_t ce_normal_chamber_gate_work_units;
    uint64_t ce_normal_chamber_gate_candidate_work_units;
    uint64_t ce_normal_chamber_gate_safe_work_k1;
    uint64_t ce_normal_chamber_gate_safe_work_k4;
    uint64_t ce_normal_chamber_gate_safe_work_k16;
    double   ce_normal_chamber_gate_margin_sum;
    double   ce_normal_chamber_gate_force_sum;
    float    ce_normal_chamber_gate_min_margin;
    float    ce_normal_chamber_gate_mean_margin;
    float    ce_normal_chamber_gate_mean_force;
    float    ce_normal_chamber_gate_candidate_fraction;
    float    ce_normal_chamber_gate_candidate_work_fraction;
    float    ce_normal_chamber_gate_safe_fraction_k1;
    float    ce_normal_chamber_gate_safe_fraction_k4;
    float    ce_normal_chamber_gate_safe_fraction_k16;
    float    ce_normal_chamber_gate_priced_gain_k1;
    float    ce_normal_chamber_gate_priced_gain_k4;
    float    ce_normal_chamber_gate_priced_gain_k16;
    int32_t  ce_normal_chamber_gate_identifying;
    int32_t  ce_normal_chamber_gate_pass;

    /* === SSM CREDIT-TIME EQUALIZED DYNAMICS ============================ */

    /* Optional R415 dynamics path. It uses the diagonal SSM decay horizon
     * c_j = sum_k |A_j|^(2k) to precondition exact original-CE gradients for
     * SSM state-dimension parameters. The default path is unchanged. */
    int32_t  credit_time_equalized_dynamics_enabled;
    int32_t  credit_time_equalized_dynamics_gate_enabled;
    uint32_t credit_time_horizon;
    float    credit_time_min_scale;
    float    credit_time_max_scale;
    float    credit_time_overhead;
    uint64_t credit_time_gate_batches;
    uint64_t credit_time_gate_params;
    double   credit_time_gate_base_descent;
    double   credit_time_gate_precond_descent;
    double   credit_time_gate_base_norm;
    double   credit_time_gate_precond_norm;
    float    credit_time_gate_mean_scale;
    float    credit_time_gate_min_scale_seen;
    float    credit_time_gate_max_scale_seen;
    float    credit_time_gate_descent_gain;
    float    credit_time_gate_line_shrink;
    float    credit_time_gate_net_descent_gain;
    float    credit_time_gate_priced_ratio;

    /* === CONTRACTIVE GREEN CREDIT ======================================= */

    /* Optional contractive Green adjoint tail certificate. It applies a
     * finite-horizon Green normalization only to state dimensions whose
     * recurrent tail debt is explicitly bounded. The default path is unchanged. */
    int32_t  contractive_green_credit_enabled;
    int32_t  contractive_green_credit_gate_enabled;
    uint32_t contractive_green_credit_horizon;
    float    contractive_green_credit_max_alpha;
    float    contractive_green_credit_tail_budget;
    float    contractive_green_credit_min_scale;
    float    contractive_green_credit_max_scale;
    float    contractive_green_credit_overhead;
    uint64_t contractive_green_credit_gate_batches;
    uint64_t contractive_green_credit_gate_params;
    uint64_t contractive_green_credit_gate_accepted_params;
    double   contractive_green_credit_gate_base_descent;
    double   contractive_green_credit_gate_green_descent;
    double   contractive_green_credit_gate_base_norm;
    double   contractive_green_credit_gate_green_norm;
    float    contractive_green_credit_gate_mean_alpha;
    float    contractive_green_credit_gate_max_alpha_seen;
    float    contractive_green_credit_gate_mean_tail_debt;
    float    contractive_green_credit_gate_max_tail_debt;
    float    contractive_green_credit_gate_accepted_fraction;
    float    contractive_green_credit_gate_mean_scale;
    float    contractive_green_credit_gate_min_scale_seen;
    float    contractive_green_credit_gate_max_scale_seen;
    float    contractive_green_credit_gate_descent_gain;
    float    contractive_green_credit_gate_line_shrink;
    float    contractive_green_credit_gate_net_descent_gain;
    float    contractive_green_credit_gate_priced_ratio;
    int32_t  contractive_green_credit_gate_identifying;
    int32_t  contractive_green_credit_gate_pass;

    /* === ADAPTIVE RECURRENT CREDIT HORIZON ============================= */

    /* Optional A8-C01 horizon certificate. It prices how much recurrent
     * transition credit can be omitted under an exact dA_per_step tail-debt
     * audit and a cheap recurrence-product bound. The default path is
     * unchanged. */
    int32_t  adaptive_credit_horizon_enabled;
    int32_t  adaptive_credit_horizon_gate_enabled;
    uint32_t adaptive_credit_min_horizon;
    uint32_t adaptive_credit_max_horizon;
    float    adaptive_credit_tail_budget;
    float    adaptive_credit_bound_budget;
    float    adaptive_credit_verify_fraction;
    float    adaptive_credit_overhead;
    float    adaptive_credit_min_headroom;
    float    adaptive_credit_tail_scale;
    uint64_t adaptive_credit_gate_batches;
    uint64_t adaptive_credit_gate_layers;
    uint64_t adaptive_credit_gate_applied_layers;
    uint64_t adaptive_credit_gate_positions;
    uint64_t adaptive_credit_gate_skipped_positions;
    uint64_t adaptive_credit_gate_total_backsteps;
    uint64_t adaptive_credit_gate_paid_backsteps;
    double   adaptive_credit_gate_exact_energy;
    double   adaptive_credit_gate_tail_energy;
    double   adaptive_credit_gate_bound_sum;
    float    adaptive_credit_gate_max_bound;
    float    adaptive_credit_gate_avg_horizon;
    float    adaptive_credit_gate_horizon_fraction;
    float    adaptive_credit_gate_skipped_fraction;
    float    adaptive_credit_gate_tail_debt;
    float    adaptive_credit_gate_energy_covered;
    float    adaptive_credit_gate_headroom;
    float    adaptive_credit_gate_priced_ratio;
    float    adaptive_credit_gate_descent_per_paid_work;
    int32_t  adaptive_credit_gate_identifying;
    int32_t  adaptive_credit_gate_pass;

    /* === INTERIOR ADJOINT TUBE ========================================= */

    /* Optional low-rank interior adjoint tube. It keeps high-energy state
     * coordinates as the paid tube and settles accepted residual coordinates
     * through an opt-in attenuation path. The default path is unchanged. */
    int32_t  interior_adjoint_tube_enabled;
    int32_t  interior_adjoint_tube_gate_enabled;
    uint32_t interior_adjoint_tube_rank;
    float    interior_adjoint_tube_residual_budget;
    float    interior_adjoint_tube_verify_fraction;
    float    interior_adjoint_tube_overhead;
    float    interior_adjoint_tube_residual_scale;
    float    interior_adjoint_tube_min_energy_covered;
    uint64_t interior_adjoint_tube_gate_batches;
    uint64_t interior_adjoint_tube_gate_layers;
    uint64_t interior_adjoint_tube_gate_accepted_layers;
    uint64_t interior_adjoint_tube_gate_params;
    uint64_t interior_adjoint_tube_gate_tube_params;
    double   interior_adjoint_tube_gate_exact_energy;
    double   interior_adjoint_tube_gate_tube_energy;
    double   interior_adjoint_tube_gate_residual_energy;
    double   interior_adjoint_tube_gate_base_descent;
    double   interior_adjoint_tube_gate_tube_descent;
    double   interior_adjoint_tube_gate_base_norm;
    double   interior_adjoint_tube_gate_tube_norm;
    float    interior_adjoint_tube_gate_tube_fraction;
    float    interior_adjoint_tube_gate_accepted_layer_fraction;
    float    interior_adjoint_tube_gate_residual_fraction;
    float    interior_adjoint_tube_gate_energy_covered;
    float    interior_adjoint_tube_gate_descent_gain;
    float    interior_adjoint_tube_gate_line_shrink;
    float    interior_adjoint_tube_gate_net_descent_gain;
    float    interior_adjoint_tube_gate_priced_ratio;
    float    interior_adjoint_tube_gate_descent_per_paid_work;
    int32_t  interior_adjoint_tube_gate_identifying;
    int32_t  interior_adjoint_tube_gate_pass;

    /* === OPTIMIZER INTEGRITY GATE ===================================== */

    int32_t  optimizer_integrity_gate_enabled;
    uint64_t optimizer_integrity_gate_batches;
    uint32_t optimizer_integrity_owned_ternary_groups;
    uint32_t optimizer_integrity_resident_ternary_groups;
    uint32_t optimizer_integrity_gpu_update_groups;
    uint32_t optimizer_integrity_cpu_skip_groups;
    uint32_t optimizer_integrity_missing_update_groups;
    int32_t  optimizer_integrity_gate_pass;

    /* === QUANTIZATION-CELL QUOTIENT GATE ============================== */

    int32_t  qcf_mvue_gate_enabled;
    uint32_t qcf_mvue_max_matrices;
    float    qcf_mvue_epsilon;
    float    qcf_mvue_min_rho;
    double   qcf_mvue_min_delta_energy;
    uint64_t qcf_mvue_gate_batches;
    uint64_t qcf_mvue_matrices;
    uint64_t qcf_mvue_elements;
    uint64_t qcf_mvue_active_delta_count;
    double   qcf_mvue_total_delta_energy;
    double   qcf_mvue_visible_delta_energy;
    double   qcf_mvue_invisible_delta_energy;
    float    qcf_mvue_rho_upper_bound;
    int32_t  qcf_mvue_identifying;
    int32_t  qcf_mvue_gate_pass;

    /* === EVIDENCE-RATE COMMITMENT DYNAMICS GATE ======================= */

    int32_t  evidence_rate_commitment_gate_enabled;
    int32_t  evidence_rate_commitment_mode;
    int32_t  evidence_rate_force_repack;
    float    evidence_rate_overhead;
    float    evidence_rate_boundary_eps;
    uint64_t evidence_rate_gate_batches;
    uint64_t evidence_rate_elements;
    uint64_t evidence_rate_nonzero_grads;
    uint64_t evidence_rate_active_updates;
    uint64_t evidence_rate_failed_updates;
    uint64_t evidence_rate_crystallized_skips;
    uint64_t evidence_rate_boundary_hits;
    uint64_t evidence_rate_sign_stable;
    uint64_t evidence_rate_sign_conflict;
    uint64_t evidence_rate_current_commits;
    uint64_t evidence_rate_tsa_commits;
    uint64_t evidence_rate_gbar_commits;
    uint64_t evidence_rate_applied_commits;
    double   evidence_rate_current_descent;
    double   evidence_rate_tsa_descent;
    double   evidence_rate_gbar_descent;
    double   evidence_rate_applied_descent;
    double   evidence_rate_applied_harm;
    float    evidence_rate_active_fraction;
    float    evidence_rate_commitment_fraction;
    float    evidence_rate_virtual_best_gain;
    float    evidence_rate_priced_ratio;
    int32_t  evidence_rate_gate_pass;

    /* === FP INTERFACE GAP PROBE ======================================== */

    /* Optional diagnostic-only gate for the R417/R300 stagewise-FP interface
     * question. It measures current Vulkan residual-layer effect against the
     * downstream CE gradient without changing the objective or optimizer. */
    int32_t  fp_interface_gap_probe_enabled;
    float    fp_interface_gap_effect_epsilon;
    uint64_t fp_interface_gap_probe_batches;
    uint64_t fp_interface_gap_probe_layers;
    uint64_t fp_interface_gap_near_identity_layers;
    float    fp_interface_gap_effect_ratio[V12_N_LAYERS];
    float    fp_interface_gap_input_rms[V12_N_LAYERS];
    float    fp_interface_gap_delta_rms[V12_N_LAYERS];
    float    fp_interface_gap_y_ssm_rms[V12_N_LAYERS];
    float    fp_interface_gap_grad_rms[V12_N_LAYERS];
    float    fp_interface_gap_abs_projection[V12_N_LAYERS];
    float    fp_interface_gap_descent_alignment[V12_N_LAYERS];
    uint64_t fp_interface_gap_wout_nnz[V12_N_LAYERS];
    int32_t  fp_interface_gap_near_identity[V12_N_LAYERS];
    float    fp_interface_gap_mean_effect_ratio;
    float    fp_interface_gap_mean_abs_projection;
    float    fp_interface_gap_min_effect_ratio;
    float    fp_interface_gap_max_grad_rms;
    int32_t  fp_interface_gap_probe_identifying;

    int32_t use_residual_shell; /* 0 = off (default), 1 = on                  */
    /* R501-S61 Lookahead-Certified CE Anchor Pruning (LCCAP) */
    int32_t lccap_gate_enabled;
    int32_t lccap_segment_length;
    float   lccap_target_loss;
    float   lccap_epsilon_safe;
    uint64_t lccap_segments_total;
    uint64_t lccap_segments_pruned;

    float   rs_eps_lo;          /* Lower band edge in nats (default 0.5)      */
    float   rs_eps_hi;          /* Upper band edge in nats (default 15.0)     */
    int32_t  exact_quotient_padding_enabled;
    uint32_t exact_quotient_padding_min_active_len;
    uint64_t exact_quotient_padding_seed;
    uint64_t exact_quotient_padding_batches;
    uint64_t exact_quotient_padding_total_targets;
    uint64_t exact_quotient_padding_active_targets;
    uint64_t exact_quotient_padding_inactive_targets;
    float    exact_quotient_padding_active_fraction;

    /* GPU optimizer flags — when set, CPU optimizer skips these params */
    uint32_t gpu_opt_flags;
#define GPU_OPT_NORM     (1u << 0)
#define GPU_OPT_A        (1u << 1)
#define GPU_OPT_BA       (1u << 2)
#define GPU_OPT_CONV     (1u << 3)
#define GPU_OPT_WA       (1u << 4)
#define GPU_OPT_EMBED_P  (1u << 5)
#define GPU_OPT_EMBED_E  (1u << 6)
#define GPU_OPT_W_IN     (1u << 7)
#define GPU_OPT_W_OUT    (1u << 8)
#define GPU_OPT_LM_HEAD  (1u << 9)
#define GPU_OPT_W_PRED   (1u << 10)

    // MBTS fields restored
    float mbts_alpha;
    float mbts_beta;
    float mbts_threshold;
    int mbts_enabled;
    float martingale_lambda;
    int continuous_ternary_margin_enabled;
    float continuous_ternary_margin_c;
    float last_mean_v_w_in[256];
    float last_mean_v_w_out[256];
    float last_mean_v_lm_head;
    float last_mean_v_w_pred;

    /* R504 F08 Quotient Group Memory Factorization Gate */
    int32_t  qgmf_gate_enabled;
    uint64_t qgmf_gate_batches;
    uint64_t qgmf_gate_collisions;

} Model;


/* =========================================================================
 * SECTION 6 — RESIDUAL STREAM ACTIVATION CACHE (PCA-INT8)
 *
 * Compressed cache for frozen layer outputs during Staged Forward Pipeline.
 * Each layer's output is PCA-compressed to 20 components and quantized to INT8,
 * reducing cache size by ~21× at medium config, ~684× at 7B scale.
 *
 * Built after freezing each layer, used in subsequent FP stages to skip
 * recomputing the frozen prefix.
 * ========================================================================= */

typedef struct {
    float   *basis;     /* ALIGN32. [d_model × rank]. PCA eigenvectors        */
    float   *mean;      /* [d_model]. Mean residual (subtracted at compress,
                         * added back at decompress).                         */
    int8_t  *codes;     /* [n_seqs × seq_len × rank]. INT8 PCA coefficients   */
    float   *scales;    /* [n_seqs × seq_len]. Per-position float scale       */
    int32_t  layer_idx; /* Which layer's output this caches (0-indexed)       */
    int32_t  n_seqs;    /* Number of sequences in the dataset                 */
    int32_t  seq_len;   /* Sequence length (cached for fast decompression)    */
    int32_t  d_model;   /* Model dimension (cached for decompression)         */
    int32_t  calibrated;/* 0 = not yet built, 1 = ready to use                */
} ResidualCacheEntry;

typedef struct {
    ResidualCacheEntry entries[V12_MAX_LAYERS];
    int32_t            active_entry;  /* Which entry to use in forward_pass.
                                       * -1 = cache not active (stage 0 or FP done) */
} ResidualCache;


/* =========================================================================
 * SECTION 6b — Y-SSM ACTIVATION CACHE (LOSSLESS)
 *
 * Stores y_ssm[k][seq_idx, t] (d_state per position) for frozen layers.
 * Also stores the surprise embedding r0[seq_idx, t] (d_model per position)
 * to avoid re-running the W_pred predictive coding on every forward call.
 *
 * Reconstruction:
 *   r[K][t] = r0[seq_idx, t] + sum_{k=0}^{K-1} W_out[k] @ y[k][seq_idx, t]
 *
 * Memory: n_seqs × seq_len × (d_model + K × d_state) × 4B
 * At medium config (n_seqs=128, seq_len=128, d_model=128, d_state=32, K=1):
 *   r0: 8.4 MB, y[k]: 2.1 MB — total 10.5 MB per active cache
 * ========================================================================= */

typedef struct {
    float   *y;         /* [n_seqs × seq_len × d_state]. SSM output y[k][s,t] */
    float   *r0;        /* [n_seqs × seq_len × d_model]. Surprise embedding.
                         * r0[s,t] = embed[token[s,t]] - W_pred @ h_prev[s,t-1]
                         * NULL if caller chooses to recompute W_pred on demand. */
    int32_t  layer_idx; /* Which frozen layer's y this caches (0-indexed)     */
    int32_t  n_seqs;
    int32_t  seq_len;
    int32_t  d_state;   /* Cached at entry build time for safe decompression  */
    int32_t  d_model;   /* Cached for r0 reconstruction                       */
    int32_t  calibrated;/* 0 = empty, 1 = ready                               */
} YCacheEntry;

typedef struct {
    YCacheEntry entries[V12_MAX_LAYERS];
    int32_t     active_entry;  /* -1 = not active, >=0 = highest frozen layer */
} YCache;


/* =========================================================================
 * SECTION 7 — FORWARD PASS CACHE
 *
 * Intermediate activations from the forward pass needed for backward.
 * Allocated once before training begins and reused across all steps.
 *
 * In inference mode: this struct is not needed (no backward pass).
 * Ownership: caller (train.c or main.c) allocates, model functions write.
 *
 * Size at V12 defaults (batch=1, seq=512):
 *   h_all:    6 × 512 × 128 × 4B = 1.57 MB
 *   x_proj:   6 × 512 × 128 × 4B = 1.57 MB
 *   x_conv:   same                = 1.57 MB
 *   residual: 6 × 512 × 1024 × 4B = 12.6 MB
 *   logits:   512 × 51 × 4B      = 0.1 MB (sampled softmax, was 65.7 MB)
 *   d_logits: same                = 0.1 MB (was 65.7 MB)
 *   sample_ids, log_proposal: negligible (0.2 MB combined)
 *   Total:    ~35 MB (down from ~166 MB; logits+d_logits reduced 660×)
 * ========================================================================= */

typedef struct {
    /* === FORWARD ACTIVATIONS ============================================= */

    /* SSM hidden state TRAJECTORY (full sequence, written by ssm_scan_forward).
     * Layout: h_all[k * seq_len * d_state + t * d_state + i]
     * h_all[k,t,i] = SSM hidden state h[i] at timestep t in layer k.
     * Needed for: (1) GP-KL Welford moments, (2) MTP random access.
     * Size: n_layers × seq_len × d_state.                              */
    uint8_t *h_all;         /* ALIGN32. [n_layers × seq_len × d_state] */

    /* Final SSM hidden state per layer (saved after forward pass).
     * h_ssm_final[k * d_state + i] = h after last timestep in forward pass.
     * Saved from h_init after ssm_scan_forward returns.
     * Also accessible as h_all[k, seq_len-1, :] but cached for convenience.
     * Size: n_layers × d_state = 4 × 32 × 4B = 512 B. Negligible.      */
    float   *h_ssm_final;   /* ALIGN32. [n_layers × d_state]                 */

    /* Output of W_in projection (before conv1d) per layer.
     * x_proj[k * seq_len * d_state + t * d_state + i]                      */
#if V12_AF8C_ENABLED
    uint8_t *x_proj;        /* ALIGN32. [n_layers × seq_len × d_state] FP8 E5M2 */
    uint8_t *x_conv;        /* ALIGN32. [n_layers × seq_len × d_state] FP8 E5M2 */
    uint8_t *x_gated;       /* ALIGN32. [n_layers × seq_len × d_state] FP8 E5M2 */
    uint8_t *mask_val;      /* ALIGN32. [n_layers × seq_len × d_state] FP8 E4M3 */
    uint8_t *b_eff_gate;    /* ALIGN32. [n_layers × seq_len × d_state] FP8 E4M3 */
    uint8_t *a_eff_gate;    /* ALIGN32. [n_layers × seq_len × d_state] FP8 E4M3 */
    uint8_t *a_eff_buf;     /* ALIGN32. [n_layers × seq_len × d_state] FP8 E4M3 */
    uint8_t *dA_per_step;   /* ALIGN32. [n_layers × seq_len × d_state] FP8 E5M2 */
#else
    float   *x_proj;        /* ALIGN32. [n_layers × seq_len × d_state]       */

    /* Output of conv1d (after W_in, before SSM scan) per layer.             */
    float   *x_conv;        /* ALIGN32. [n_layers × seq_len × d_state]       */

    /* Output of soft-gate mask (after W_in, before conv1d) per layer.       */
    float   *x_gated;       /* ALIGN32. [n_layers × seq_len × d_state]       */

    /* Sigmoid mask values cached for backward pass.                         */
    float   *mask_val;      /* ALIGN32. [n_layers × seq_len × d_state]       */

    /* Direct-B gate values: sigmoid(x_conv[t,i]) per layer, per timestep.
     * b_eff_gate[k * seq_len * d_state + t * d_state + i]
     * Cached for backward: dgate = dB_eff[i,t] * B[i] * sigmoid'(x_conv[i,t])  */
    float   *b_eff_gate;    /* ALIGN32. [n_layers × seq_len × d_state]       */

    /* A-gate values: sigmoid(W_A[i,:] @ x_conv[t] + b_A[i]) per layer, per timestep.
     * a_eff_gate[k * seq_len * d_state + t * d_state + i]
     * Cached for backward: d(A_eff)/d(gate) = A[i]; d(gate)/d(pre) = gate * (1 - gate) */
    float   *a_eff_gate;    /* ALIGN32. [n_layers × seq_len × d_state]       */

    /* A-eff buffer: A[i] * a_eff_gate[t,i] per layer, per timestep.
     * a_eff_buf[k * seq_len * d_state + t * d_state + i]
     * Passed directly to ssm_scan_forward as A_per_step. */
    float   *a_eff_buf;     /* ALIGN32. [n_layers × seq_len × d_state]       */

    /* dA_per_step gradient buffer written by ssm_scan_backward.
     * dA_per_step[k * seq_len * d_state + t * d_state + i] = dh[t,i] * h_prev[t,i]
     * Chained through gate and bias to accumulate dW_A and db_A. */
    float   *dA_per_step;   /* ALIGN32. [n_layers × seq_len × d_state]       */
#endif

    /* Pre-skip residual stream values (input to each SSM layer).
     * residual[k * seq_len * d_model + t * d_model + j]                    */
    float   *residual;      /* ALIGN32. [n_layers × seq_len × d_model]       */

    /* Sampled softmax logits from lm_head (k+1 = 51 samples per position).
     * Target token at index 0, noise tokens at indices 1..k.
     * Shape: [seq_len × V12_SS_N_SAMPLES]. Size: 512 × 51 × 4B = 0.1 MB
     * (vs full vocab: 512 × 32064 × 4B = 65.7 MB before change)          */
    float   *logits;        /* ALIGN32. [seq_len × V12_SS_N_SAMPLES]         */

    /* Vocabulary indices of sampled tokens (target + noise).
     * sample_ids[t * V12_SS_N_SAMPLES + s] = vocab index of sample s at position t.
     * sample_ids[t * V12_SS_N_SAMPLES + 0] = tokens[t+1] (target).
     * Size: 512 × 51 × 4B = 0.1 MB. Negligible.                           */
    int32_t *sample_ids;    /* [seq_len × V12_SS_N_SAMPLES]                  */

    /* Log proposal probabilities for importance correction (unigram).
     * log_proposal[t * V12_SS_N_SAMPLES + s] = log(q(v_s))
     * Used in sampled CE to correct bias from non-uniform sampling.
     * Size: 512 × 51 × 4B = 0.1 MB. Negligible.                           */
    float   *log_proposal;  /* [seq_len × V12_SS_N_SAMPLES]                  */

    /* === BACKWARD SCRATCH BUFFERS ======================================== */

    /* Gradient of sampled CE loss w.r.t. sampled logits.
     * Shape: [seq_len × V12_SS_N_SAMPLES]. Size: 0.1 MB (vs full: 65.7 MB) */
    float   *d_logits;      /* ALIGN32. [seq_len × V12_SS_N_SAMPLES]         */

#if V12_AF8C_ENABLED
    uint8_t *d_h;           /* ALIGN32. [n_layers × seq_len × d_state] FP8 E5M2 */
    uint8_t *d_x_conv;      /* ALIGN32. [n_layers × seq_len × d_state] FP8 E5M2 */
    uint8_t *d_x_proj;      /* ALIGN32. [n_layers × seq_len × d_state] FP8 E5M2 */
#else
    /* Gradient of loss w.r.t. hidden state h, flowing backward through BPTT.
     * d_h[k * seq_len * d_state + t * d_state + i]                         */
    float   *d_h;           /* ALIGN32. [n_layers × seq_len × d_state]       */

    /* Gradient flowing backward through conv1d.                             */
    float   *d_x_conv;      /* ALIGN32. [n_layers × seq_len × d_state]       */

    /* Gradient of loss w.r.t. x_proj (W_in backward input).               */
    float   *d_x_proj;      /* ALIGN32. [n_layers × seq_len × d_state]       */
#endif

    /* Gradient flowing backward through the skip (residual) connections.   */
    float   *d_residual;    /* ALIGN32. [n_layers × seq_len × d_model]       */

    /* === CACHED MAX LOGITS (for numerically stable CE backward) ========== */

    /* max_logit[t] = max(logits[t * vocab .. (t+1)*vocab - 1]).
     * Computed once in compute_ce_loss and reused by compute_backward to
     * avoid recomputing max in the backward softmax.                        */
    float   *max_logit;     /* Shape: [seq_len]                              */

    /* === ADVANCED CACHE OPTIMIZATIONS ===================================== */

#if V12_AF8C_ENABLED
    uint8_t *y_ssm;         /* ALIGN32. [n_layers × seq_len × d_state] FP8 E5M2 */
#else
    /* SSM output before W_out projection (y_t = C @ h_t per layer).
     * y_ssm[k * seq_len * d_state + t * d_state + i]
     * Used for analysis and loss computation hooks.                          */
    float   *y_ssm;         /* ALIGN32. [n_layers × seq_len × d_state]       */
#endif

    /* === LAYER NORM CACHE (final norm before lm_head) ===================== */

    /* Normalized residual: input to lm_head after RMS norm.
     * norm_out[t * d_model + j] = norm_w[j] * residual[t,j] / rms[t] + norm_b[j] */
    float   *norm_out;      /* ALIGN32. [seq_len × d_model]                  */

    /* Per-timestep RMS value cached for backward pass.
     * rms[t] = sqrt(mean_j(residual[t,j]^2) + 1e-6)                        */
    float   *norm_rms;      /* [seq_len]                                      */

    /* === PCA-INT8 ACTIVATION CACHE (FP stages) ============================== */

    /* PCA-INT8 activation cache for frozen FP prefixes.
     * Populated after each FP stage completes and its layer is packed.
     * Used in forward_pass to skip frozen layer computation.               */
    ResidualCache *res_cache;   /* NULL until first FP stage completes         */

    /* === Y-SSM LOSSLESS CACHE (FP stages, default) ========================= */
    YCache *y_cache;    /* NULL until first FP stage completes (lossless mode)   */

} Cache;


/* =========================================================================
 * SECTION 8 — TRAIN STATE
 *
 * Optimizer and scheduler state that persists across training steps.
 * Saved in the training checkpoint (SECTION_TRAIN_STATE in checkpoint.c).
 * Not needed for inference.
 *
 * Designed for direct binary serialization:
 *   - All fields are fixed-width integers or fp32
 *   - No pointers — TrainState is fully self-contained (no heap refs)
 *   - Natural alignment throughout: no implicit padding expected
 * ========================================================================= */

typedef struct {
    /* === GLOBAL STEP COUNTER ============================================= */

    uint32_t step;              /* Current global training step (monotone)   */

    /* === LEARNING RATE ================================================== */

    float    lr_base;           /* Base LR for MVUE ternary updates (0.1)    */

    /* Per-layer LR multipliers (eigenspectrum calibration, innovation §15).
     * lr_effective[k] = lr_base * lr_per_layer[k].
     * Initialized to 1.0f. Adjusted based on layer gradient scale.         */
    float    lr_per_layer[V12_MAX_LAYERS];

    /* === LOSS TRACKING ================================================== */

    /* Ring buffer of recent per-step CE losses (for plotting and red-flag
     * detection). head points to the next write slot.                       */
    float    loss_history[V12_LOSS_HISTORY_LEN];
    uint32_t loss_history_head; /* Next write index (mod V12_LOSS_HISTORY_LEN)*/

    /* Last measured eval–train loss gap. Red flag threshold: > 0.05.
     * NaN = not yet measured. Zeroed (with warning) on checkpoint load.    */
    float    eval_gap;

    /* === FORWARD-PASS TRAINING STATE (FP staging) ======================= */

    /* FP (staged layer training) current stage.
     * stage k = only layer k is being trained; 0..n_layers-1.
     * fp_completed = 1 after all n_layers stages have been run once.        */
    uint32_t fp_stage;
    uint8_t  fp_completed;      /* 0 = still in initial FP pass, 1 = done    */

    /* === BCD REFINEMENT STATE (post-FP) ================================== */

    uint8_t  bcd_active;        /* 1 = in BCD refinement phase               */
    uint8_t  bcd_cycle;         /* Current BCD cycle (0 or 1)                */
    uint8_t  bcd_layer;         /* Current layer being refined               */

    /* === RNG STATE FOR MVUE SAMPLING ==================================== */

    /* xorshift64 state for MVUE weight sampling. Persists across steps and
     * is checkpointed with TrainState to ensure deterministic resumption.   */
    uint64_t rng_state;

    /* Sequential dataset cursor at checkpoint boundaries. This closes the
     * resume path for deterministic sequential readers; random-sampled
     * dataset modes still rely on their own seed/state contracts.            */
    uint64_t dataset_cursor;

    /* Dataset sampler RNG state at checkpoint boundaries. When valid, this
     * closes deterministic resume for dataset_sample_batch() as well as
     * sequential cursor-driven readers.                                      */
    uint64_t dataset_rng[4];
    uint8_t  dataset_state_valid;

    /* === CRYSTALLIZATION STATISTICS (per step, informational) =========== */

    /* Fraction of weights that are crystallized at last repack.
     * Useful for progress monitoring (target: 20–30% after 10K steps).     */
    float    crystal_fraction;

    /* === RESERVED PADDING =============================================== */

    /* Explicit padding to keep the struct at a round size for serialization.
     * Adjust the count if fields are added above.                           */
    float    sre_lr_scale;      /* Dynamically scales learning rate after SRE collapse warning */
    float    cams_running_var;  /* EMA of gradient coordinate curvature variance for CAMS */

    /* === R493 TRAINING QUALITY & LINGUISTIC GATES ======================= */
    uint8_t  skip_bptt_current_step;
    uint8_t  padding_zmma[3];
    R493StageContract r493_contract;
    R493EvaluationReceipt r493_receipt;
} TrainState;


/* =========================================================================
 * SECTION 9 — SSM HIDDEN STATE (per-layer, per-batch, single timestep)
 *
 * The running state vector for one SSM layer during autoregressive inference.
 * Stored separately from the Layer struct to allow batched inference without
 * duplicating the (shared) weight parameters.
 *
 * In batch inference: allocate an array of (batch_size × n_layers) SSMState.
 * In training: the full h_cache in the Cache struct supersedes this.
 * ========================================================================= */

typedef struct {
    /* The current SSM hidden state vector for one layer at one timestep.
     * Updated in-place by ssm_scan_forward: h[i] = A[i]*h[i] + B[i]*x[i]  */
    float h[V12_MAX_D_STATE];   /* [d_state] — inline, no heap alloc needed
                                 * at MAX_D_STATE=256: 1 KB per layer        */

    /* Which layer this state belongs to (0..n_layers-1).
     * Informational: allows sanity checks in debug builds.                  */
    int32_t layer_idx;

    /* Whether h has been initialized. 0 = zeroed (start of sequence),
     * 1 = carries state from previous token (autoregressive continuation).  */
    int32_t initialized;
} SSMState;


/* =========================================================================
 * SECTION 10 — CONVENIENCE MACROS FOR BUFFER INDEXING
 *
 * These macros encode the layout contracts documented in the Cache struct.
 * Using them instead of raw arithmetic prevents off-by-one errors and makes
 * the layout convention explicit at every call site.
 * ========================================================================= */

/* h_all[layer][t][i] */
#define H_IDX(layer, t, i, d_state)  \
    ((layer) * (V12_SEQ_LEN) * (d_state) + (t) * (d_state) + (i))

/* residual[layer][t][j] */
#define R_IDX(layer, t, j, d_model)  \
    ((layer) * (V12_SEQ_LEN) * (d_model) + (t) * (d_model) + (j))

/* logits[t][v] — and d_logits, same layout */
#define LOGIT_IDX(t, v, vocab_size)  \
    ((t) * (vocab_size) + (v))

/* W_in layout: row = output neuron in d_state, col = input in d_model
 * shadow_w_in[row * d_model + col] */
#define WIN_IDX(row, col)   ((row) * V12_D_MODEL + (col))

/* W_out layout: row = output neuron in d_model, col = input in d_state
 * shadow_w_out[row * d_state + col] */
#define WOUT_IDX(row, col)  ((row) * V12_D_STATE + (col))


#endif /* V12_TYPES_H */
