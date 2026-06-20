/*
 * loss.h — Loss functions for v12 training loop
 *
 * All functions are pure (no global state). Thread-unsafe only in the sense
 * that they write into caller-supplied buffers (Cache, GpKlMoments).
 *
 * Include order: always after types.h
 */

#ifndef V12_LOSS_H
#define V12_LOSS_H

#include "types.h"  /* Model, Cache, GpKlMoments, V12_* constants */
#include <stdint.h> /* uint16_t */

/*
 * compute_ce_loss
 *
 * Cross-entropy loss over a single sequence, log-sum-exp numerically stable.
 * Covers positions t=0..seq_len-2 (predict token t+1 from logits at t).
 *
 * SIDE EFFECT: writes cache->max_logit[t] for t=0..seq_len-2.
 * These values MUST be preserved until compute_ce_backward is called.
 *
 * Returns: mean CE loss in nats (positive scalar).
 */
float compute_ce_loss(
    Cache          *cache,         /* max_logit[t] written here (side effect) */
    const float    *logits,        /* [seq_len * vocab_size]  raw pre-softmax */
    const uint16_t *targets,       /* [seq_len]  target token IDs (BPE)       */
    int             vocab_size,
    int             seq_len
);

/*
 * compute_ce_backward
 *
 * Gradient of CE loss w.r.t. logits: d_logits[t][v] = softmax[t][v] - 1[v==target].
 * Reads cache->max_logit[t] written by compute_ce_loss — must be called after it.
 * Writes cache->d_logits[t * vocab_size + v] for t=0..seq_len-2, all v.
 * Positions t=seq_len-1 are zeroed (no prediction target there).
 *
 * The gradient is already divided by (seq_len-1) to match the mean loss.
 */
void compute_ce_backward(
    Cache          *cache,         /* reads max_logit, writes d_logits        */
    const float    *logits,        /* [seq_len * vocab_size]  same as forward */
    const uint16_t *targets,       /* [seq_len]                               */
    int             vocab_size,
    int             seq_len
);

/*
 * compute_gp_kl_loss
 *
 * GP-KL unified regularization loss computed from Welford accumulators.
 * Reads model->layers[k].moments for k=0..n_layers-1.
 * Pure read — no writes to model.
 *
 * Layers without accumulated moments are skipped. This keeps checkpoint-loaded
 * staged/eval paths from aborting before their first fresh SSM scan.
 *
 * Returns: GP_KL_BETA * sum_over_layers(l_mean + l_var + l_acorr + l_stab + l_orth)
 */
float compute_gp_kl_loss(
    const Model *model
);

/*
 * compute_contraction_loss
 *
 * Penalty on the spectral radius of (I - A) for each SSM layer.
 * Encourages dissipative dynamics: σ_min(I - A) = 1 - max(A[i]).
 * Loss increases as max(A[i]) → 1 (eigenvalue of A approaches 1).
 *
 * Pseudocode:
 *   total = 0.0f
 *   for k in 0..n_layers-1:
 *       max_a = max(A[k][0..d_state-1])
 *       sigma_min = 1.0 - max_a
 *       if sigma_min > 0.0f:
 *           total -= CONTRACTION_BETA * log(sigma_min)
 *       else:
 *           total += 1e6f  // hard penalty for divergent SSM
 *   return total
 *
 * Returns: loss value (positive if any layer is too dissipative, >= 0 always).
 */
float compute_contraction_loss(
    const Model *model
);

/*
 * compute_contraction_backward
 *
 * Gradient of contraction loss w.r.t. A parameters.
 * For each layer, finds argmax(A[i]) and accumulates gradient into dA[argmax].
 * d/dA[argmax] of -beta * log(1 - max(A)) = beta / (1 - max(A)).
 *
 * Called during backward pass to update A gradients for contraction penalty.
 */
void compute_contraction_backward(
    Model *model
);

/*
 * gp_kl_welford_update
 *
 * Online Welford accumulator update for one SSM layer at one timestep.
 * Called by the SSM scan loop (ssm_scan_forward) after computing h[t].
 * NOT called from loss.c — declared here so ssm_scan.c can include loss.h.
 *
 * m    — the GpKlMoments accumulator for this layer (read+write)
 * h    — h[t], the SSM hidden state vector at this timestep [d_state floats]
 * d_state — must equal V12_D_STATE (128). Passed explicitly to avoid coupling.
 */
void gp_kl_welford_update(
    GpKlMoments *m,
    const float *h,
    int          d_state,
    int          t
);

/*
 * compute_sampled_ce_loss — Sampled softmax CE loss with NCE importance correction.
 *
 * For each position t=0..seq_len-2:
 *   - Sample k noise tokens from unigram distribution (importance sampling)
 *   - Compute logits for target + k noise tokens using ternary_dot_row
 *   - Apply importance correction: a_j = logit_j - log(k * q[v_j])
 *   - Log-sum-exp stable softmax and cross-entropy
 *   - Store sample indices, logits, and proposal probs for backward pass
 *
 * Parameters:
 *   cache      — writes logits, sample_ids, log_proposal, max_logit
 *   model      — reads csr_lm_head, unigram_cdf, log_unigram
 *   norm_out   — [seq_len × d_model] normalized residual before lm_head
 *   targets    — [seq_len] target token IDs
 *   seq_len    — sequence length
 *
 * Returns: mean CE loss over seq_len-1 positions (positive scalar in nats).
 */
float compute_sampled_ce_loss(
    Cache          *cache,
    const Model    *model,
    const float    *norm_out,
    const uint16_t *targets,
    int             seq_len
);

/*
 * compute_sampled_ce_backward — Gradient of sampled CE w.r.t. sampled logits.
 *
 * Computes softmax gradient over the k+1 sampled logits.
 * Recomputes importance-corrected scores from stored logits and proposal probs.
 * Writes d_logits[t * (k+1) + s] = softmax[s] - 1[s==0] (target only gets -1).
 *
 * Must be called after compute_sampled_ce_loss (reads sample_ids, log_proposal, max_logit).
 * Gradients are scaled by 1/(seq_len-1) to match mean loss.
 *
 * Parameters:
 *   cache   — reads logits, sample_ids, log_proposal, max_logit
 *             writes d_logits
 *   seq_len — sequence length
 */
void compute_sampled_ce_backward(
    Cache *cache,
    int    seq_len
);

/*
 * ============================================================================
 * RESIDUAL SHELL (R357 Alt 3) — OBJECTIVE IS REWEIGHTED, NOT UNCHANGED
 *
 * When enabled, per-token gradient contributions are masked: tokens whose CE
 * loss falls outside (eps_lo, eps_hi) contribute zero gradient. Surviving
 * gradients are renormalized by 1/S where S = number of surviving tokens.
 *
 * This is NOT equivalent to full-gradient training. The model optimizes a
 * REWEIGHTED objective. Default is OFF. Enabling is always explicit.
 * See types.h Model::use_residual_shell.
 * ============================================================================
 */

/*
 * compute_sampled_ce_loss_with_per_token
 *
 * Identical to compute_sampled_ce_loss but also writes the per-token NLL
 * (in nats) into per_token_loss_out[t] for t = 0..seq_len-2.
 * per_token_loss_out must point to at least (seq_len - 1) floats.
 * The returned scalar (mean CE over seq_len-1 positions) is identical to
 * compute_sampled_ce_loss.
 */
float compute_sampled_ce_loss_with_per_token(
    Cache          *cache,
    const Model    *model,
    const float    *norm_out,
    const uint16_t *targets,
    int             seq_len,
    float          *per_token_loss_out  /* [seq_len - 1] */
);

/*
 * compute_sampled_ce_backward_masked
 *
 * Masked variant of compute_sampled_ce_backward for Residual Shell.
 * For each position t in [0, seq_len-2]:
 *   if mask[t] != 0: gradient is computed with global normalizer inv_S
 *                    (replacing the unmasked 1/(seq_len-1))
 *   else           : the entire d_logits row for that position is zeroed
 *                    (both softmax row and target correction skipped)
 *
 * Caller is responsible for computing mask and inv_S = 1/S globally across
 * any batched set of sequences. The caller must guarantee inv_S > 0.
 */
void compute_sampled_ce_backward_masked(
    Cache          *cache,
    int             seq_len,
    const uint8_t  *mask,    /* [seq_len - 1] */
    float           inv_S
);

/*
 * compute_ce_loss_with_per_token
 *
 * Identical to compute_ce_loss but also writes -log_prob[t] into
 * per_token_loss_out[t] for t = 0..seq_len-2. Full-softmax variant (non-sampled).
 */
float compute_ce_loss_with_per_token(
    Cache          *cache,
    const float    *logits,
    const uint16_t *targets,
    int             vocab_size,
    int             seq_len,
    float          *per_token_loss_out  /* [seq_len - 1] */
);

/*
 * compute_ce_backward_masked
 *
 * Masked variant of compute_ce_backward. See compute_sampled_ce_backward_masked
 * for the masking semantics.
 */
void compute_ce_backward_masked(
    Cache          *cache,
    const float    *logits,
    const uint16_t *targets,
    int             vocab_size,
    int             seq_len,
    const uint8_t  *mask,    /* [seq_len - 1] */
    float           inv_S
);

/*
 * compute_sincts_loss_and_gradient
 *
 * Implements Self-Isotropic Normalized Contrastive Teaching Signals (SINCTS).
 * Matches intermediate layer hidden states with lagged embedding targets.
 * Computes isotropic dispersion (L_disp) and semantic alignment (L_align).
 * Computes exact backprop gradients directly into dh_out.
 */
float compute_sincts_loss_and_gradient(
    const float    *h_all,
    const float    *embed_E,
    const uint16_t *targets,
    float          *dh_out,
    int             batch_size,
    int             seq_len,
    int             d_state,
    int             d_embed,
    float           alpha
);

/* R493-G01 Answer Transaction CE Contrast (Aspect A01) */
float compute_answer_contrast_loss(
    const float    *logits_correct,
    const float    *logits_incorrect,
    const uint16_t *targets_correct,
    const uint16_t *targets_incorrect,
    int             vocab_size,
    int             seq_len,
    float           margin
);

#endif /* V12_LOSS_H */
