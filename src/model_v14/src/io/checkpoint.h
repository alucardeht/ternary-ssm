/*
 * checkpoint.h — Binary training checkpoint save/load interface
 *
 * Checkpoints contain the complete training state needed to resume from any step:
 *   - TrainState (step counter, LR, RNG, loss history)
 *   - All shadow weights (FP32 logits)
 *   - SSM dynamics (A, B, C matrices)
 *   - Model-level parameters (factored embedding E+P, norm, lm_head shadow, tau_mux)
 *   - Adam moment buffers (m, v per param) - version 2+
 *   - Factored embedding (E, P matrices) - version 4+
 *   - Canonical embedding optimizer moments for GPU-owned resume symmetry - version 8+
 *   - Dataset cursor and sampler RNG in TrainState - version 9+
 *
 * NOT saved (runtime-derived):
 *   - CSR indices (rebuilt from sampled weights)
 *   - Sampled ternary weights (resampled from shadow logits)
 *   - Crystallization masks (recomputed from shadow logits)
 *   - Gradients (zeroed after each step)
 *   - Cache (pure scratch buffer)
 *
 * Binary format: 8-byte header + TrainState + dimensions + per-layer params + model-level params + moments (v2+).
 * All values in native byte order (portable for checkpoints on same architecture).
 * Version 1: without Adam moments.
 * Version 2: with full 2D Adam moment buffers (added 2026-03-26).
 * Version 3: with factored Adam moments for 2D matrices (added 2026-03-26).
 * Version 4: with factored embedding (E, P matrices instead of single embed table) (added 2026-03-26).
 * Version 8: with canonical embed_E/embed_P optimizer moments (added 2026-04-18).
 * Version 9: with TrainState dataset cursor/sampler RNG (added 2026-06-01).
 */

#ifndef V12_CHECKPOINT_H
#define V12_CHECKPOINT_H

#include "../core/types.h"

/* Magic number: bytes 'V','1','2','C' in little-endian = 0x43323156 */
#define V12_CHECKPOINT_MAGIC  0x43323156U

/* Checkpoint format version. Increment if format changes. */
#define V12_CHECKPOINT_VERSION 9U
#define V12_CHECKPOINT_MIN_SUPPORTED_VERSION 8U


/*
 * checkpoint_save
 *
 * Write a full training checkpoint to `path`. Includes all trained parameters
 * and training state needed to resume.
 *
 * Uses write-to-temp + rename pattern for atomicity: writes to "<path>.tmp"
 * first, then renames to final path. If the process crashes during write,
 * the checkpoint file remains intact (or absent if this is the first save).
 *
 * Parameters:
 *   path — file path for checkpoint (absolute or relative)
 *   m    — model pointer (must be fully allocated by model_alloc)
 *   s    — training state pointer
 *
 * Returns 0 on success, -1 on I/O error. Errors are printed to stderr.
 */
int checkpoint_save(const char *path, const Model *m, const TrainState *s);


/*
 * checkpoint_load
 *
 * Load a checkpoint from `path` into an already-allocated Model and TrainState.
 *
 * Caller must have called model_alloc() to set up all heap pointers in m first.
 * This function overwrites the shadow weights, SSM parameters, embeddings,
 * norms, and the entire TrainState struct with values from the checkpoint file.
 *
 * Validates:
 *   - Magic number (V12_CHECKPOINT_MAGIC)
 *   - Format version (V12_CHECKPOINT_VERSION)
 *   - Dimensions (n_layers, d_model, d_state, vocab_size, conv_kernel, mtp_rank)
 *     against compile-time constants and current Model fields.
 *   - MTP presence: if checkpoint has `has_mtp=1` for a layer but the model's
 *     mtp_U/V are NULL, returns -1 (dimension mismatch).
 *
 * After successful load, the caller MUST call checkpoint_rebuild(m, s) to:
 *   - Resample ternary weights from shadow logits (MVUE)
 *   - Rebuild all CSR indices
 *   - Recompute crystallization masks
 *
 * If the file does not exist, returns -1 silently (not an error for first run).
 *
 * Parameters:
 *   path — file path to load from
 *   m    — pointer to allocated Model (all pointer fields pre-allocated)
 *   s    — pointer to TrainState (will be overwritten with checkpoint values)
 *
 * Returns 0 on success, -1 on error (file not found, version mismatch,
 * dimension mismatch, I/O error). Errors are printed to stderr.
 */
int checkpoint_load(const char *path, Model *m, TrainState *s);


/*
 * checkpoint_rebuild
 *
 * Restore all runtime-derived state after checkpoint_load().
 *
 * After loading a checkpoint, the model's sampled weights, CSR indices, and
 * crystallization masks are stale or missing. This function rebuilds them:
 *
 *   1. Resample MVUE ternary weights from shadow logits for each layer
 *      - Populates sampled_w_in, sampled_w_out, sampled_lm_head
 *      - Uses rng_state from the loaded TrainState (deterministic)
 *   2. Rebuild CSR indices for all layers from sampled weights
 *      - Populates csr_in, csr_out for inference-time matmuls
 *   3. Rebuild CSR for lm_head
 *   4. Update crystallization masks by rescanning shadow logits
 *      - Sets crystal_mask_in, crystal_mask_out based on |logit| > tau
 *
 * After this call, the model is ready to continue training or inference.
 *
 * Parameters:
 *   m — pointer to Model (must have been just loaded by checkpoint_load)
 *   s — pointer to TrainState (must have been just loaded by checkpoint_load)
 *
 * Returns 0 on success, -1 if any CSR rebuild fails (allocation failure).
 * Errors are printed to stderr.
 */
int checkpoint_rebuild(Model *m, TrainState *s);


#endif /* V12_CHECKPOINT_H */
