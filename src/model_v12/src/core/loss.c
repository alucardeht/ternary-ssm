#include <stdio.h>
#include "loss.h"
#include "model.h"
#include "ternary_matmul.h"  /* ternary_dot_row */
#include <math.h>    /* expf, logf, fabsf */
#include <string.h>  /* memset */
#include <stdlib.h>  /* calloc, free */
#include <assert.h>  /* assert (debug builds only) */

/*
 * ============================================================================
 * gp_kl_welford_update
 *
 * Online Welford accumulator for SSM hidden state moments.
 * Called seq_len * n_layers = 512 * 6 = 3072 times per training step.
 * Must be branch-light for performance.
 * ============================================================================
 */

void gp_kl_welford_update(GpKlMoments *m, const float *h, int d_state, int t) {
    if (t % V12_WCS_STRIDE != 0) {
        return;
    }
    m->n += 1;
    const float inv_n = 1.0f / (float)m->n;

    /* --- WELFORD MEAN AND M2 (per dimension) --- */
    for (int i = 0; i < d_state; i++) {
        const float x = h[i];
        const float delta = x - m->mean[i];
        m->mean[i] += delta * inv_n;
        const float delta2 = x - m->mean[i];
        m->M2[i] += delta * delta2;
    }

    /* --- LAG-1 AUTOCORRELATION AND STABILITY (both use old lag1_prev) --- */
    if (m->n > 1) {
        float diff_sq = 0.0f;
        for (int i = 0; i < d_state; i++) {
            m->lag1_sum[i] += h[i] * m->lag1_prev[i];
            const float diff = h[i] - m->lag1_prev[i];
            diff_sq += diff * diff;
        }
        m->stability_sum += diff_sq;
    }

    /* --- UPDATE lag1_prev FOR NEXT TIMESTEP --- */
    for (int i = 0; i < d_state; i++) {
        m->lag1_prev[i] = h[i];
    }

    /* --- ORTHOGONALITY: 32 FIXED RANDOM PAIRS --- */
    for (int p = 0; p < 32; p++) {
        const int ia = (int)m->orth_pair_a[p];
        const int ib = (int)m->orth_pair_b[p];
        const float dot = h[ia] * h[ib];
        m->orth_sum[p] += dot * dot;
    }
}


/*
 * ============================================================================
 * compute_ce_loss
 *
 * Cross-entropy loss over a single sequence using log-sum-exp trick.
 * Covers positions t=0..seq_len-2 (predict token t+1 from logits at t).
 *
 * Numerical stability: shift logits by max before exp to prevent overflow.
 * Stores max_logit[t] for reuse in backward pass to avoid recomputation.
 * ============================================================================
 */

float compute_ce_loss(Cache *cache, const float *logits,
                      const uint16_t *targets, int vocab_size, int seq_len) {
    float ce_sum = 0.0f;
    const int n_tokens = seq_len - 1;

    for (int t = 0; t < n_tokens; t++) {
        const float *row = logits + t * vocab_size;

        /* STEP 1: Find max for numerical stability (log-sum-exp trick) */
        float max_v = row[0];
        for (int v = 1; v < vocab_size; v++) {
            if (row[v] > max_v) {
                max_v = row[v];
            }
        }

        /* STEP 2: Store max for backward reuse */
        cache->max_logit[t] = max_v;

        /* STEP 3: Compute sum_exp (shifted)
         * Invariant: sum_exp >= 1.0 because the max term contributes exp(0)=1 */
        float sum_exp = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            sum_exp += expf(row[v] - max_v);
        }

        /* STEP 4: log-softmax for the target token only
         * log_prob = log(softmax[target]) = (row[target] - max) - log(sum_exp) */
        const uint16_t target_v = targets[t + 1];
        const float log_prob = (row[target_v] - max_v) - logf(sum_exp);

        /* STEP 5: Accumulate NLL (negated because log_prob <= 0) */
        ce_sum -= log_prob;
    }

    /* Zero max_logit at seq_len-1 (no backward gradient at that position) */
    cache->max_logit[seq_len - 1] = 0.0f;

    return ce_sum / (float)n_tokens;
}


/*
 * ============================================================================
 * compute_ce_backward
 *
 * Gradient of CE loss w.r.t. logits.
 * d_logits[t][v] = softmax[t][v] - 1[v == target[t+1]]
 *
 * Reuses max_logit[t] computed in compute_ce_loss to avoid recomputing max.
 * Last row (seq_len-1) is zeroed because no target exists there.
 * Gradient is scaled by 1/(seq_len-1) to match mean loss.
 * ============================================================================
 */

void compute_ce_backward(Cache *cache, const float *logits,
                         const uint16_t *targets, int vocab_size, int seq_len) {
    const int n_tokens = seq_len - 1;
    const float inv_n = 1.0f / (float)n_tokens;

    for (int t = 0; t < n_tokens; t++) {
        const float *row = logits + t * vocab_size;
        float *d_row = cache->d_logits + t * vocab_size;
        const float max_v = cache->max_logit[t];
        const uint16_t target_v = targets[t + 1];

        /* STEP 1: Recompute sum_exp using same shift as forward
         * (identical loop — compiler may fuse or vectorize)  */
        float sum_exp = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            sum_exp += expf(row[v] - max_v);
        }

        /* STEP 2: d_logits[t][v] = softmax[t][v] * inv_n
         * inv_sum = inv_n / sum_exp combines scaling with softmax denominator */
        const float inv_sum = inv_n / sum_exp;
        for (int v = 0; v < vocab_size; v++) {
            d_row[v] = expf(row[v] - max_v) * inv_sum;
        }

        /* STEP 3: Subtract 1 at target position (one-hot gradient term) */
        d_row[target_v] -= inv_n;
    }

    /* STEP 4: Zero the last position (no target exists for seq_len-1) */
    memset(cache->d_logits + (seq_len - 1) * vocab_size, 0,
           vocab_size * sizeof(float));
}


/*
 * ============================================================================
 * compute_gp_kl_loss
 *
 * GP-KL unified regularization loss computed from Welford moment accumulators.
 * Five terms per layer: mean (target=0), variance (target=1), lag-1 autocorr
 * (target=0), stability (target<=0.1, one-sided penalty), orthogonality (target=0).
 *
 * All five terms squared and summed, then scaled by V12_GP_KL_BETA = 0.01.
 * ============================================================================
 */

float compute_gp_kl_loss(const Model *model) {
    float gp_loss = 0.0f;

    for (int k = 0; k < V12_N_LAYERS; k++) {
        const GpKlMoments *m = &model->layers[k].moments;
        if (m->n == 0) {
            continue;
        }
        const float n = (float)m->n;
        const float inv_n = 1.0f / (n + 1e-8f);

        /* --- MOMENT 1: MEAN (target = 0.0) --- */
        float mu_sum = 0.0f;
        for (int i = 0; i < V12_D_STATE; i++) {
            mu_sum += m->mean[i];
        }
        const float mu = mu_sum / (float)V12_D_STATE;
        const float l_mean = mu * mu;

        /* --- MOMENT 2: VARIANCE (target = 1.0) --- */
        float var_sum = 0.0f;
        for (int i = 0; i < V12_D_STATE; i++) {
            var_sum += m->M2[i] * inv_n;
        }
        const float var_ = var_sum / (float)V12_D_STATE;
        const float l_var = (var_ - 1.0f) * (var_ - 1.0f);

        /* --- MOMENT 3: LAG-1 AUTOCORRELATION (target = 0.0) --- */
        float acorr_sum = 0.0f;
        for (int i = 0; i < V12_D_STATE; i++) {
            acorr_sum += m->lag1_sum[i] * inv_n;
        }
        const float autocr = (acorr_sum / (float)V12_D_STATE) / (var_ + 1e-8f);
        const float l_acorr = autocr * autocr;

        /* --- MOMENT 4: STABILITY (target = <= 0.1, one-sided penalty) --- */
        const float stab = m->stability_sum * inv_n;
        const float excess = stab - 0.1f;
        const float l_stab = (excess > 0.0f) ? excess * excess : 0.0f;

        /* --- MOMENT 5: ORTHOGONALITY (target = 0.0, 32 fixed pairs) --- */
        float orth_sum_total = 0.0f;
        for (int p = 0; p < 32; p++) {
            orth_sum_total += m->orth_sum[p];
        }
        const float orth = orth_sum_total / (32.0f * n + 1e-8f);
        const float l_orth = orth * orth;

        /* --- ACCUMULATE ALL FIVE TERMS FOR THIS LAYER --- */
        gp_loss += l_mean + l_var + l_acorr + l_stab + l_orth;
    }

    return V12_GP_KL_BETA * gp_loss;
}


/*
 * ============================================================================
 * compute_contraction_loss
 *
 * Penalty on the spectral radius of (I - A) for dissipative SSM dynamics.
 * A[i] values must be in (0, 1) by optimizer clamping. This loss encourages
 * smaller eigenvalues (more stable, dissipative systems).
 *
 * For diagonal A: σ_min(I - A) = 1 - max(A[i]).
 * As max(A[i]) → 1, σ_min → 0 and loss → +∞.
 * ============================================================================
 */

float compute_contraction_loss(const Model *model) {
    float total = 0.0f;
    const int W = V12_CONTRACTION_WINDOW;
    const float tau = V12_CONTRACTION_THRESHOLD;

    for (int k = 0; k < V12_N_LAYERS; k++) {
        const float *A = model->layers[k].A;

        for (int i = 0; i < V12_D_STATE; i++) {
            /* A_W = A[i]^W — computed via repeated squaring for integer W=16 */
            float a = A[i];
            float a_w = 1.0f;
            int w = W;
            while (w > 0) {
                if (w & 1) a_w *= a;
                a   *= a;
                w >>= 1;
            }

            const float excess = a_w - tau;
            if (excess > 0.0f) {
                total += excess * excess;
            }
        }
    }

    return V12_CONTRACTION_BETA * total;
}


/*
 * ============================================================================
 * compute_contraction_backward
 *
 * Gradient of contraction loss w.r.t. A parameters.
 * For each layer, identifies the dimension with max(A[i]) and accumulates
 * the gradient: d/dA[argmax] = beta / sigma_min, where sigma_min = 1 - max(A).
 * ============================================================================
 */

void compute_contraction_backward(Model *model) {
    const int W = V12_CONTRACTION_WINDOW;
    const float tau = V12_CONTRACTION_THRESHOLD;

    for (int k = 0; k < V12_N_LAYERS; k++) {
        SSMLayer *L = &model->layers[k];
        if (L->dA == NULL) continue;  /* offloaded layer */

        const float *A = L->A;

        for (int i = 0; i < V12_D_STATE; i++) {
            /* Compute A[i]^W and A[i]^(W-1) together */
            float a = A[i];
            float a_w = 1.0f;
            int w = W;
            while (w > 0) {
                if (w & 1) a_w *= a;
                a   *= a;
                w >>= 1;
            }

            const float excess = a_w - tau;
            if (excess > 0.0f) {
                /* dL/dA[i] = beta * 2 * excess * W * A[i]^(W-1)
                 * A[i]^(W-1) = a_w / A[i]  (safe: A[i] > 0 by clamping invariant) */
                const float a_w_minus1 = a_w / A[i];
                float grad = V12_CONTRACTION_BETA * 2.0f * excess
                             * (float)W * a_w_minus1;

                /* Cap gradient to prevent oscillation (matches existing guard) */
                if (grad > 1.0f) grad = 1.0f;

                L->dA[i] += grad;
            }
        }
    }
}


/*
 * ============================================================================
 * sample_noise_token (static helper)
 *
 * Draw one noise sample from unigram CDF via binary search + LCG RNG.
 * Returns token ID v in [0, vocab_size).
 * Static LCG is sufficient — no cryptographic quality needed.
 * ============================================================================
 */

static int32_t sample_noise_token(const float *cdf, int32_t vocab_size,
                                   uint32_t *rng_state) {
    *rng_state = *rng_state * 1664525u + 1013904223u;
    float u = (float)(*rng_state >> 8) * (1.0f / 16777216.0f);

    int32_t lo = 0, hi = vocab_size - 1;
    while (lo < hi) {
        int32_t mid = (lo + hi) >> 1;
        if (cdf[mid] < u) lo = mid + 1;
        else              hi = mid;
    }
    return lo;
}


/*
 * ============================================================================
 * compute_sampled_ce_loss
 *
 * Sampled softmax CE loss with importance-weighted sampling from unigram.
 * For each position t=0..seq_len-2:
 *   1. Target = tokens[t+1]
 *   2. Sample k noise tokens, avoiding target (deduplication standard practice)
 *   3. Compute logits for k+1 samples using ternary_dot_row
 *   4. Apply importance correction: a_j = logit_j - log(k * q[v_j])
 *   5. Log-sum-exp stable softmax and cross-entropy
 *   6. Store sample_ids, logits, log_proposal for backward pass
 *
 * References: Bengio & Senecal 2008, Jean et al. 2015 (NMT sampled softmax)
 * ============================================================================
 */

float compute_sampled_ce_loss(Cache *cache, const Model *model,
                              const float *norm_out,
                              const uint16_t *targets, int seq_len) {
    const int N = V12_SS_N_SAMPLES;
    const int k = V12_SS_K;
    const int d = V12_D_MODEL;
    const int V = V12_VOCAB_SIZE;
    float ce_sum = 0.0f;
    const int n_tokens = seq_len - 1;

    for (int t = 0; t < n_tokens; t++) {
        const float *h = norm_out + t * d;
        int32_t  *sids = cache->sample_ids  + t * N;
        float    *slogs = cache->log_proposal + t * N;
        float    *slogits = cache->logits + t * N;

        sids[0]  = targets[t + 1];
        slogs[0] = model->log_unigram[sids[0]];

        uint32_t rng_state = (uint32_t)(uintptr_t)sids ^ (uint32_t)t ^ 0xDEADBEEFu;
        for (int s = 1; s <= k; s++) {
            int32_t v;
            do { v = sample_noise_token(model->unigram_cdf, V, &rng_state); }
            while (v == sids[0]);
            sids[s]  = v;
            slogs[s] = model->log_unigram[v];
        }

        for (int s = 0; s < N; s++) {
            slogits[s] = ternary_dot_row(&model->csr_lm_head, sids[s], h);
            slogits[s] += corpus_logit_prior_logit(
                model, targets[t], (uint16_t)sids[s]);
            slogits[s] += program_kernel_logit(
                model, targets[t], (uint16_t)sids[s]);
            if (t > 0) {
                slogits[s] += state_transition_atlas_logit(
                    model, targets[t - 1], targets[t], (uint16_t)sids[s]);
            }
        }

        float log_k = logf((float)k);
        float a[V12_SS_N_SAMPLES];
        a[0] = slogits[0];
        for (int s = 1; s < N; s++) {
            a[s] = slogits[s] - (log_k + slogs[s]);
        }

        float max_a = a[0];
        for (int s = 1; s < N; s++) if (a[s] > max_a) max_a = a[s];
        cache->max_logit[t] = max_a;

        float sum_exp = 0.0f;
        for (int s = 0; s < N; s++) sum_exp += expf(a[s] - max_a);

        float log_prob = (a[0] - max_a) - logf(sum_exp);
        ce_sum -= log_prob;
    }

    cache->max_logit[seq_len - 1] = 0.0f;
    return ce_sum / (float)n_tokens;
}


/*
 * ============================================================================
 * compute_sampled_ce_backward
 *
 * Gradient of sampled CE w.r.t. sampled logits (before importance correction).
 *
 * Softmax gradient over importance-corrected scores:
 *   softmax_s[j] = exp(a[j] - max_a) / sum_s exp(a[s] - max_a)
 * where a[j] = logit[j] - log(k * q[v_j]) for j > 0, a[0] = logit[0].
 *
 * Because a[j] = logit[j] + const (const independent of logit[j]),
 * d(loss)/d(logit[j]) = d(loss)/d(a[j]) (chain rule: const cancels).
 *
 * CE target gradient: d_logits[0] = softmax[0] - 1,  d_logits[s] = softmax[s]
 * ============================================================================
 */

void compute_sampled_ce_backward(Cache *cache, int seq_len) {
    const int N = V12_SS_N_SAMPLES;
    const int k = V12_SS_K;
    const float inv_n = 1.0f / (float)(seq_len - 1);
    const float log_k = logf((float)k);
    const int n_tokens = seq_len - 1;

    for (int t = 0; t < n_tokens; t++) {
        const int32_t *sids  = cache->sample_ids   + t * N;
        const float   *slogs = cache->log_proposal + t * N;
        const float   *slogits = cache->logits     + t * N;
        float         *d_slogits = cache->d_logits + t * N;
        const float    max_a = cache->max_logit[t];

        float a[V12_SS_N_SAMPLES];
        a[0] = slogits[0];
        for (int s = 1; s < N; s++) {
            a[s] = slogits[s] - (log_k + slogs[s]);
        }

        float sum_exp = 0.0f;
        for (int s = 0; s < N; s++) sum_exp += expf(a[s] - max_a);
        const float inv_sum = inv_n / sum_exp;

        for (int s = 0; s < N; s++) {
            d_slogits[s] = expf(a[s] - max_a) * inv_sum;
        }
        d_slogits[0] -= inv_n;
    }

    memset(cache->d_logits + (seq_len - 1) * N, 0, N * sizeof(float));
}


/*
 * ============================================================================
 * RESIDUAL SHELL (R357 Alt 3) — OBJECTIVE IS REWEIGHTED, NOT UNCHANGED
 *
 * When use_residual_shell = 1, per-token gradient contributions are masked:
 * tokens whose CE loss falls outside (eps_lo, eps_hi) contribute zero gradient.
 * Surviving gradients are renormalized by 1/S where S = number of surviving
 * tokens, so the gradient magnitude matches the full-batch scale.
 *
 * This is NOT equivalent to full-gradient training. The model optimizes a
 * REWEIGHTED objective that ignores very-easy and very-hard tokens. The
 * converged optimum may differ.
 *
 * Default is OFF. Enabling is always explicit.
 * ============================================================================
 */

float compute_sampled_ce_loss_with_per_token(
    Cache *cache, const Model *model, const float *norm_out,
    const uint16_t *targets, int seq_len, float *per_token_loss_out) {

    const int N = V12_SS_N_SAMPLES;
    const int k = V12_SS_K;
    const int d = V12_D_MODEL;
    const int V = V12_VOCAB_SIZE;
    float ce_sum = 0.0f;
    const int n_tokens = seq_len - 1;

    for (int t = 0; t < n_tokens; t++) {
        const float *h = norm_out + t * d;
        int32_t  *sids = cache->sample_ids  + t * N;
        float    *slogs = cache->log_proposal + t * N;
        float    *slogits = cache->logits + t * N;

        sids[0]  = targets[t + 1];
        slogs[0] = model->log_unigram[sids[0]];

        uint32_t rng_state = (uint32_t)(uintptr_t)sids ^ (uint32_t)t ^ 0xDEADBEEFu;
        for (int s = 1; s <= k; s++) {
            int32_t v;
            do { v = sample_noise_token(model->unigram_cdf, V, &rng_state); }
            while (v == sids[0]);
            sids[s]  = v;
            slogs[s] = model->log_unigram[v];
        }

        for (int s = 0; s < N; s++) {
            slogits[s] = ternary_dot_row(&model->csr_lm_head, sids[s], h);
            slogits[s] += corpus_logit_prior_logit(
                model, targets[t], (uint16_t)sids[s]);
            slogits[s] += program_kernel_logit(
                model, targets[t], (uint16_t)sids[s]);
            if (t > 0) {
                slogits[s] += state_transition_atlas_logit(
                    model, targets[t - 1], targets[t], (uint16_t)sids[s]);
            }
        }

        float log_k = logf((float)k);
        float a[V12_SS_N_SAMPLES];
        a[0] = slogits[0];
        for (int s = 1; s < N; s++) {
            a[s] = slogits[s] - (log_k + slogs[s]);
        }

        float max_a = a[0];
        for (int s = 1; s < N; s++) if (a[s] > max_a) max_a = a[s];
        cache->max_logit[t] = max_a;

        float sum_exp = 0.0f;
        for (int s = 0; s < N; s++) sum_exp += expf(a[s] - max_a);

        float log_prob = (a[0] - max_a) - logf(sum_exp);
        const float nll = -log_prob;
        ce_sum += nll;
        if (per_token_loss_out) per_token_loss_out[t] = nll;
    }

    cache->max_logit[seq_len - 1] = 0.0f;
    return ce_sum / (float)n_tokens;
}


void compute_sampled_ce_backward_masked(Cache *cache, int seq_len,
                                        const uint8_t *mask, float inv_S) {
    const int N = V12_SS_N_SAMPLES;
    const int k = V12_SS_K;
    const float log_k = logf((float)k);
    const int n_tokens = seq_len - 1;

    for (int t = 0; t < n_tokens; t++) {
        float *d_slogits = cache->d_logits + t * N;

        if (!mask[t]) {
            /* Masked-out row: zero gradient, no softmax, no target correction */
            memset(d_slogits, 0, N * sizeof(float));
            continue;
        }

        const int32_t *sids    = cache->sample_ids   + t * N; (void)sids;
        const float   *slogs   = cache->log_proposal + t * N;
        const float   *slogits = cache->logits       + t * N;
        const float    max_a   = cache->max_logit[t];

        float a[V12_SS_N_SAMPLES];
        a[0] = slogits[0];
        for (int s = 1; s < N; s++) {
            a[s] = slogits[s] - (log_k + slogs[s]);
        }

        float sum_exp = 0.0f;
        for (int s = 0; s < N; s++) sum_exp += expf(a[s] - max_a);
        const float inv_sum = inv_S / sum_exp;

        for (int s = 0; s < N; s++) {
            d_slogits[s] = expf(a[s] - max_a) * inv_sum;
        }
        d_slogits[0] -= inv_S;
    }

    memset(cache->d_logits + (seq_len - 1) * N, 0, N * sizeof(float));
}


float compute_ce_loss_with_per_token(Cache *cache, const float *logits,
                                     const uint16_t *targets, int vocab_size,
                                     int seq_len, float *per_token_loss_out) {
    float ce_sum = 0.0f;
    const int n_tokens = seq_len - 1;

    for (int t = 0; t < n_tokens; t++) {
        const float *row = logits + t * vocab_size;

        float max_v = row[0];
        for (int v = 1; v < vocab_size; v++) {
            if (row[v] > max_v) max_v = row[v];
        }
        cache->max_logit[t] = max_v;

        float sum_exp = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            sum_exp += expf(row[v] - max_v);
        }

        const uint16_t target_v = targets[t + 1];
        const float log_prob = (row[target_v] - max_v) - logf(sum_exp);
        const float nll = -log_prob;
        ce_sum += nll;
        if (per_token_loss_out) per_token_loss_out[t] = nll;
    }

    cache->max_logit[seq_len - 1] = 0.0f;
    return ce_sum / (float)n_tokens;
}


void compute_ce_backward_masked(Cache *cache, const float *logits,
                                const uint16_t *targets, int vocab_size,
                                int seq_len, const uint8_t *mask, float inv_S) {
    const int n_tokens = seq_len - 1;

    for (int t = 0; t < n_tokens; t++) {
        float *d_row = cache->d_logits + t * vocab_size;

        if (!mask[t]) {
            memset(d_row, 0, vocab_size * sizeof(float));
            continue;
        }

        const float *row = logits + t * vocab_size;
        const float max_v = cache->max_logit[t];
        const uint16_t target_v = targets[t + 1];

        float sum_exp = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            sum_exp += expf(row[v] - max_v);
        }

        const float inv_sum = inv_S / sum_exp;
        for (int v = 0; v < vocab_size; v++) {
            d_row[v] = expf(row[v] - max_v) * inv_sum;
        }
        d_row[target_v] -= inv_S;
    }

    memset(cache->d_logits + (seq_len - 1) * vocab_size, 0,
           vocab_size * sizeof(float));
}

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
) {
    if (!h_all || !embed_E || !targets || !dh_out || batch_size <= 0 || seq_len <= 1 || d_state <= 0 || d_embed <= 0) {
        return 0.0f;
    }

    int dim = d_state < d_embed ? d_state : d_embed;
    int N = batch_size * (seq_len - 1);

    float *hat_h = calloc((size_t)batch_size * seq_len * d_state, sizeof(float));
    float *hat_y = calloc((size_t)batch_size * seq_len * d_state, sizeof(float));
    if (!hat_h || !hat_y) {
        free(hat_h);
        free(hat_y);
        return 0.0f;
    }

    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len - 1; t++) {
            size_t idx_h = (size_t)b * seq_len * d_state + (size_t)t * d_state;
            const float *h = h_all + idx_h;
            float *hh = hat_h + idx_h;

            float norm_h = 0.0f;
            for (int i = 0; i < d_state; i++) norm_h += h[i] * h[i];
            norm_h = sqrtf(norm_h);
            if (norm_h > 1e-12f) {
                float inv_norm = 1.0f / norm_h;
                for (int i = 0; i < d_state; i++) hh[i] = h[i] * inv_norm;
            }

            uint16_t target_tok = targets[b * seq_len + t + 1];
            const float *y = embed_E + (size_t)target_tok * d_embed;
            float *hy = hat_y + idx_h;

            float norm_y = 0.0f;
            for (int i = 0; i < dim; i++) norm_y += y[i] * y[i];
            norm_y = sqrtf(norm_y);
            if (norm_y > 1e-12f) {
                float inv_norm = 1.0f / norm_y;
                for (int i = 0; i < dim; i++) hy[i] = y[i] * inv_norm;
            }
        }
    }

    float *C = calloc((size_t)d_state * d_state, sizeof(float));
    if (!C) {
        free(hat_h);
        free(hat_y);
        return 0.0f;
    }

    float inv_N = 1.0f / (float)N;
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len - 1; t++) {
            size_t idx_h = (size_t)b * seq_len * d_state + (size_t)t * d_state;
            const float *hh = hat_h + idx_h;
            for (int i = 0; i < d_state; i++) {
                for (int j = 0; j < d_state; j++) {
                    C[i * d_state + j] += hh[i] * hh[j] * inv_N;
                }
            }
        }
    }

    float tr_C2 = 0.0f;
    for (int i = 0; i < d_state; i++) {
        for (int j = 0; j < d_state; j++) {
            float val = C[i * d_state + j];
            tr_C2 += val * val;
        }
    }
    float l_disp = tr_C2 - 1.0f / (float)d_state;
    if (l_disp < 0.0f) l_disp = 0.0f;

    float l_align = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len - 1; t++) {
            size_t idx_h = (size_t)b * seq_len * d_state + (size_t)t * d_state;
            const float *hh = hat_h + idx_h;
            const float *hy = hat_y + idx_h;
            float dot = 0.0f;
            for (int i = 0; i < d_state; i++) dot += hh[i] * hy[i];
            l_align -= dot * inv_N;
        }
    }

    float total_loss = l_disp + alpha * l_align;

    memset(dh_out, 0, (size_t)batch_size * seq_len * d_state * sizeof(float));

    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len - 1; t++) {
            size_t idx_h = (size_t)b * seq_len * d_state + (size_t)t * d_state;
            const float *h = h_all + idx_h;
            const float *hh = hat_h + idx_h;
            const float *hy = hat_y + idx_h;
            float *dh = dh_out + idx_h;

            float norm_h = 0.0f;
            for (int i = 0; i < d_state; i++) norm_h += h[i] * h[i];
            norm_h = sqrtf(norm_h);

            if (norm_h > 1e-12f) {
                float G[V12_MAX_D_STATE];
                for (int i = 0; i < d_state; i++) {
                    float C_hh = 0.0f;
                    for (int j = 0; j < d_state; j++) {
                        C_hh += C[i * d_state + j] * hh[j];
                    }
                    G[i] = 4.0f * inv_N * C_hh - alpha * inv_N * hy[i];
                }

                float hh_dot_G = 0.0f;
                for (int i = 0; i < d_state; i++) hh_dot_G += hh[i] * G[i];

                for (int i = 0; i < d_state; i++) {
                    dh[i] = (G[i] - hh_dot_G * hh[i]) / norm_h;
                }
            }
        }
    }

    free(C);
    free(hat_h);
    free(hat_y);

    return total_loss;
}

/* R493-G01 Answer Transaction CE Contrast (Aspect A01) */
float compute_answer_contrast_loss(
    const float    *logits_correct,
    const float    *logits_incorrect,
    const uint16_t *targets_correct,
    const uint16_t *targets_incorrect,
    int             vocab_size,
    int             seq_len,
    float           margin
) {
    printf("DEBUG: Entered compute_answer_contrast_loss\n"); fflush(stdout);
    if (!logits_correct || !logits_incorrect || !targets_correct || !targets_incorrect || seq_len <= 1) {
        return 0.0f;
    }

    float ce_correct = 0.0f;
    float ce_incorrect = 0.0f;
    const int n_tokens = seq_len - 1;

    for (int t = 0; t < n_tokens; t++) {
        if (t % 100 == 0) {
            printf("DEBUG: Processing token %d\n", t); fflush(stdout);
        }
        /* Correct sequence CE */
        const float *row_c = logits_correct + t * vocab_size;
        float max_c = row_c[0];
        for (int v = 1; v < vocab_size; v++) {
            if (row_c[v] > max_c) max_c = row_c[v];
        }
        float sum_exp_c = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            sum_exp_c += expf(row_c[v] - max_c);
        }
        const uint16_t target_c = targets_correct[t + 1];
        float log_prob_c = 0.0f;
        if (target_c < vocab_size) {
            log_prob_c = (row_c[target_c] - max_c) - logf(sum_exp_c);
        }
        ce_correct -= log_prob_c;

        /* Incorrect sequence CE */
        const float *row_i = logits_incorrect + t * vocab_size;
        float max_i = row_i[0];
        for (int v = 1; v < vocab_size; v++) {
            if (row_i[v] > max_i) max_i = row_i[v];
        }
        float sum_exp_i = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            sum_exp_i += expf(row_i[v] - max_i);
        }
        const uint16_t target_i = targets_incorrect[t + 1];
        float log_prob_i = 0.0f;
        if (target_i < vocab_size) {
            log_prob_i = (row_i[target_i] - max_i) - logf(sum_exp_i);
        }
        ce_incorrect -= log_prob_i;
    }

    ce_correct /= (float)n_tokens;
    ce_incorrect /= (float)n_tokens;

    printf("DEBUG: Finished loop in compute_answer_contrast_loss\n"); fflush(stdout);
    return fmaxf(0.0f, ce_correct - ce_incorrect + margin);
}
