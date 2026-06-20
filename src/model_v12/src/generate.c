/*
 * generate.c — Autoregressive text generation for v12 ternary SSM model.
 *
 * CPU-only. No GPU / Vulkan dependency.
 * Usage: ./generate <checkpoint_path> [--max-tokens N] [--temperature T]
 *                   [--top-k K] [--seed S]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#include "core/types.h"
#include "core/model.h"
#include "core/ternary_matmul.h"
#include "core/ssm_scan.h"
#include "io/checkpoint.h"

/* =========================================================================
 * INT2 decode helpers
 * Encoding: bits[1:0]=w0, bits[3:2]=w1, bits[5:4]=w2, bits[7:6]=w3
 *   0b00 =  0,  0b01 = +1,  0b11 = -1
 * ========================================================================= */

static inline float decode_int2_weight(uint8_t byte, int slot)
{
    uint8_t bits = (byte >> (slot * 2)) & 0x3;
    if (bits == 0x01) return  1.0f;
    if (bits == 0x03) return -1.0f;
    return 0.0f;
}

static void decode_conv_kernel(const uint8_t *w_conv_int2,
                                float *out,
                                int conv_kernel, int d_state)
{
    int total = conv_kernel * d_state;
    for (int idx = 0; idx < total; idx++) {
        int byte_idx = idx / V12_INT2_WEIGHTS_PER_BYTE;
        int slot     = idx % V12_INT2_WEIGHTS_PER_BYTE;
        out[idx] = decode_int2_weight(w_conv_int2[byte_idx], slot);
    }
}

/* =========================================================================
 * Per-layer autoregressive state
 * ========================================================================= */

typedef struct {
    float h[V12_D_STATE];
    float conv_buf[V12_CONV_KERNEL][V12_D_STATE];
    int   conv_pos;
} GenLayerState;

typedef struct {
    GenLayerState layers[V12_N_LAYERS];
    float         prev_residual[V12_D_MODEL];
    int           tokens_generated;
} GenState;

typedef struct {
    int enabled;
    int requested;
    int entries_per_row;
    uint16_t *cols;
    float *values;
    float *row_abs_bound;
    uint64_t selected_entries;
    uint64_t dense_residual_bytes;
    uint64_t sparse_residual_bytes;
    float resident_reduction;
} A9LogitSyndromeResidual;

typedef struct {
    int enabled;
    int requested;
    int group_size;
    int frontier_groups;
    float margin;
    int groups;
    uint16_t *group_max_pos;
    uint16_t *group_max_neg;
    uint64_t attempts;
    uint64_t accepted;
    uint64_t fallback;
    uint64_t exact_rows;
    uint64_t full_rows;
} A9ExactLogitFrontier;

typedef struct {
    A9LogitSyndromeResidual residual;
    A9ExactLogitFrontier frontier;
} A9Runtime;

/* =========================================================================
 * RNG — xorshift64 for sampling
 * ========================================================================= */

static uint64_t rng_state_g;

static uint64_t rng_next(void)
{
    rng_state_g ^= rng_state_g << 13;
    rng_state_g ^= rng_state_g >> 7;
    rng_state_g ^= rng_state_g << 17;
    return rng_state_g;
}

static float rng_uniform(void)
{
    return (float)(rng_next() >> 11) / (float)(1ULL << 53);
}

/* =========================================================================
 * Embedding: token row from embed_int2 -> d_model vector
 * embed_int2 layout: [vocab_size × d_model], packed 4 weights/byte
 * Row stride in bytes: d_model / 4
 * ========================================================================= */

static void embed_token(const Model *m, int token, float *out)
{
    int d = V12_D_MODEL;
    int row_bytes = d / V12_INT2_WEIGHTS_PER_BYTE;
    const uint8_t *row = m->embed_int2 + (size_t)token * row_bytes;
    for (int j = 0; j < d; j++) {
        int byte_idx = j / V12_INT2_WEIGHTS_PER_BYTE;
        int slot     = j % V12_INT2_WEIGHTS_PER_BYTE;
        out[j] = decode_int2_weight(row[byte_idx], slot);
    }
}

/* =========================================================================
 * Predictive coding: residual -= W_pred @ prev_residual (skip at t=0)
 * ========================================================================= */

static void apply_predictive_coding(const Model *m,
                                    float *residual,
                                    const float *prev_residual)
{
    float pred[V12_D_MODEL];
    memset(pred, 0, V12_D_MODEL * sizeof(float));
    ternary_matvec(&m->csr_w_pred, prev_residual, pred);
    for (int j = 0; j < V12_D_MODEL; j++)
        residual[j] -= pred[j];
}

/* =========================================================================
 * RMS norm: out[j] = (x[j] / rms(x)) * norm_w[j] + norm_b[j]
 * norm_b may be NULL (RMS-only mode).
 * ========================================================================= */

static void apply_rms_norm(const float *x,
                            const float *norm_w,
                            const float *norm_b,
                            float *out,
                            int d)
{
    float sum_sq = 0.0f;
    for (int j = 0; j < d; j++) sum_sq += x[j] * x[j];
    float rms = sqrtf(sum_sq / (float)d + 1e-6f);
    float inv_rms = 1.0f / rms;
    for (int j = 0; j < d; j++) {
        out[j] = x[j] * inv_rms * norm_w[j];
        if (norm_b) out[j] += norm_b[j];
    }
}

/* =========================================================================
 * Single-token forward through one SSM layer.
 * Implements: W_in CSR -> soft gate -> conv1d ring buffer ->
 *             Direct-B sigmoid -> A-gate sigmoid -> SSM step ->
 *             C @ h -> WHT -> W_out CSR -> residual add.
 * ========================================================================= */

static void forward_layer_single_token(const Model *m,
                                       const SSMLayer *L,
                                       GenLayerState *gs,
                                       const float *residual_in,
                                       float *residual_out,
                                       const float *conv_w)
{
    int d  = V12_D_MODEL;
    int ds = V12_D_STATE;
    int K  = V12_CONV_KERNEL;

    /* --- W_in projection: residual_in (d_model) -> x_proj (d_state) ------- */
    float x_proj[V12_D_STATE];
    memset(x_proj, 0, ds * sizeof(float));
    ternary_matvec(&L->csr_in, residual_in, x_proj);

    /* --- Soft gate (tau_mux applied before conv, per gpu_pipeline forward) - */
    float tau_k   = m->tau_mux[(int)(L - m->layers)];
    float scale_k = V12_SOFT_BOUNDARY_SCALE;
    float x_gated[V12_D_STATE];
    for (int i = 0; i < ds; i++) {
        float xp  = x_proj[i];
        float msk = 1.0f / (1.0f + expf(-scale_k * (fabsf(xp) - tau_k)));
        x_gated[i] = xp * msk;
    }

    /* --- Causal conv1d ring buffer ---------------------------------------- */
    int pos = gs->conv_pos;
    for (int i = 0; i < ds; i++)
        gs->conv_buf[pos % K][i] = x_gated[i];

    float x_conv[V12_D_STATE];
    memset(x_conv, 0, ds * sizeof(float));
    for (int kk = 0; kk < K; kk++) {
        int src_slot = ((pos - kk) % K + K) % K;
        for (int i = 0; i < ds; i++)
            x_conv[i] += conv_w[kk * ds + i] * gs->conv_buf[src_slot][i];
    }
    gs->conv_pos = pos + 1;

    /* --- Direct-B gate: b_eff[i] = sigmoid(x_conv[i]) * B[i] ------------- */
    float b_eff[V12_D_STATE];
    for (int i = 0; i < ds; i++) {
        float gate = 1.0f / (1.0f + expf(-x_conv[i]));
        b_eff[i] = gate * L->B[i];
    }

    /* --- A-gate: a_eff[i] = A[i] * sigmoid(W_A[i,:] @ x_conv + b_A[i]) -- */
    float a_eff[V12_D_STATE];
    for (int i = 0; i < ds; i++) {
        float dot = L->b_A[i];
        const float *wa_row = L->W_A + i * ds;
        for (int j = 0; j < ds; j++) dot += wa_row[j] * x_conv[j];
        float gate = 1.0f / (1.0f + expf(-dot));
        a_eff[i] = L->A[i] * gate;
    }

    /* --- SSM single step (inline, not ssm_scan_forward) ------------------- */
    float *h = gs->h;

    /* h[i] = a_eff[i] * h[i] + b_eff[i] * x_conv[i] */
    for (int i = 0; i < ds; i++)
        h[i] = a_eff[i] * h[i] + b_eff[i] * x_conv[i];

    /* Bilinear term: h[i] += C_bilinear[i] * h_prev[i] * x_conv[i]
     * Note: h was already updated above — to match the forward scan we apply
     * the bilinear gate on the post-update h (self-interaction).
     * If C_bilinear is NULL, skip. */
    if (L->C_bilinear) {
        for (int i = 0; i < ds; i++)
            h[i] += L->C_bilinear[i] * h[i] * x_conv[i];
    }

    /* --- Output: y = C @ h ------------------------------------------------ */
    float y_ssm[V12_D_STATE];
    memset(y_ssm, 0, ds * sizeof(float));
    for (int i = 0; i < ds; i++) {
        const float *c_row = L->C + i * ds;
        float acc = 0.0f;
        for (int j = 0; j < ds; j++) acc += c_row[j] * h[j];
        y_ssm[i] = acc;
    }

    /* --- WHT on y_ssm ----------------------------------------------------- */
    wht_inplace(y_ssm, ds);

    /* --- W_out projection: y_ssm (d_state) -> out_buf (d_model) ----------- */
    float out_buf[V12_D_MODEL];
    memset(out_buf, 0, d * sizeof(float));
    ternary_matvec(&L->csr_out, y_ssm, out_buf);

    /* --- Residual add ----------------------------------------------------- */
    for (int j = 0; j < d; j++)
        residual_out[j] = residual_in[j] + out_buf[j];
}

/* =========================================================================
 * Compute full vocab logits via ternary_dot_row for all vocab entries.
 * logits[v] = csr_lm_head[v, :] @ norm_out
 * ========================================================================= */

static void compute_full_logits(const Model *m,
                                 const float *norm_out,
                                 float *logits)
{
    int vocab = V12_VOCAB_SIZE;
    for (int v = 0; v < vocab; v++)
        logits[v] = ternary_dot_row(&m->csr_lm_head, v, norm_out);
}

static int materialize_checkpoint_host_state(Model *m)
{
    if (!m) return -1;

    if (model_materialize_lm_head_host(m) != 0) {
        fprintf(stderr, "[GENERATE] failed to materialize lm_head checkpoint buffers\n");
        return -1;
    }
    if (model_materialize_w_pred_host(m) != 0) {
        fprintf(stderr, "[GENERATE] failed to materialize w_pred checkpoint buffers\n");
        return -1;
    }
    if (model_materialize_norm_host(m) != 0) {
        fprintf(stderr, "[GENERATE] failed to materialize norm checkpoint buffers\n");
        return -1;
    }
    if (model_materialize_embed_host(m) != 0) {
        fprintf(stderr, "[GENERATE] failed to materialize embed checkpoint buffers\n");
        return -1;
    }
    for (int k = 0; k < V12_N_LAYERS; k++) {
        if (model_materialize_layer_w_io_host(&m->layers[k]) != 0) {
            fprintf(stderr,
                    "[GENERATE] failed to materialize layer %d W_in/W_out checkpoint buffers\n",
                    k);
            return -1;
        }
    }

    printf("[GENERATE_CHECKPOINT_HOST] active=1 mode=production_v12_checkpoint_load "
           "lm_head=on w_pred=on norm=on embed=on layer_w_io=on layers=%d\n",
           V12_N_LAYERS);
    return 0;
}

static float compute_residual_correction(const A9LogitSyndromeResidual *r,
                                         int token,
                                         const float *norm_out)
{
    if (!r || !r->enabled || token < 0 || token >= V12_VOCAB_SIZE)
        return 0.0f;
    float acc = 0.0f;
    size_t base = (size_t)token * (size_t)r->entries_per_row;
    for (int i = 0; i < r->entries_per_row; i++) {
        const float value = r->values[base + (size_t)i];
        if (value == 0.0f) continue;
        acc += value * norm_out[r->cols[base + (size_t)i]];
    }
    return acc;
}

static float compute_a9_logit(const Model *m,
                              const A9LogitSyndromeResidual *r,
                              int token,
                              const float *norm_out)
{
    return ternary_dot_row(&m->csr_lm_head, token, norm_out) +
           compute_residual_correction(r, token, norm_out);
}

static void compute_full_logits_a9(const Model *m,
                                   const A9LogitSyndromeResidual *r,
                                   const float *norm_out,
                                   float *logits)
{
    int vocab = V12_VOCAB_SIZE;
    for (int v = 0; v < vocab; v++)
        logits[v] = compute_a9_logit(m, r, v, norm_out);
}

static void a9_residual_free(A9LogitSyndromeResidual *r)
{
    if (!r) return;
    free(r->cols);
    free(r->values);
    free(r->row_abs_bound);
    memset(r, 0, sizeof(*r));
}

static int a9_insert_top_residual(float abs_value,
                                  float value,
                                  uint16_t col,
                                  float *top_abs,
                                  float *top_values,
                                  uint16_t *top_cols,
                                  int k)
{
    if (k <= 0 || abs_value <= top_abs[k - 1])
        return 0;
    int pos = k - 1;
    while (pos > 0 && abs_value > top_abs[pos - 1]) {
        top_abs[pos] = top_abs[pos - 1];
        top_values[pos] = top_values[pos - 1];
        top_cols[pos] = top_cols[pos - 1];
        pos--;
    }
    top_abs[pos] = abs_value;
    top_values[pos] = value;
    top_cols[pos] = col;
    return 1;
}

static int a9_residual_build(A9LogitSyndromeResidual *r, const Model *m)
{
    if (!r || !m || !r->requested) return 0;
    if (!m->shadow_lm_head || !m->sampled_lm_head) {
        fprintf(stderr,
                "[A9_LOGIT_SYNDROME_RESIDUAL] requested=1 active=0 reason=missing_lm_head_host_buffers\n");
        return 0;
    }

    int k = r->entries_per_row;
    if (k <= 0) k = 4;
    if (k > 64) k = 64;
    r->entries_per_row = k;

    const size_t rows = (size_t)V12_VOCAB_SIZE;
    const size_t cols = (size_t)V12_D_MODEL;
    const size_t entries = rows * (size_t)k;
    r->cols = (uint16_t *)calloc(entries, sizeof(uint16_t));
    r->values = (float *)calloc(entries, sizeof(float));
    r->row_abs_bound = (float *)calloc(rows, sizeof(float));
    if (!r->cols || !r->values || !r->row_abs_bound) {
        a9_residual_free(r);
        fprintf(stderr,
                "[A9_LOGIT_SYNDROME_RESIDUAL] requested=1 active=0 reason=allocation_failed\n");
        return -1;
    }

    float top_abs[64];
    float top_values[64];
    uint16_t top_cols[64];
    uint64_t selected = 0;
    double abs_bound_sum = 0.0;

    for (int v = 0; v < V12_VOCAB_SIZE; v++) {
        for (int i = 0; i < k; i++) {
            top_abs[i] = 0.0f;
            top_values[i] = 0.0f;
            top_cols[i] = 0u;
        }
        const size_t row = (size_t)v * cols;
        for (int j = 0; j < V12_D_MODEL; j++) {
            const float shadow = SHADOW_DECODE(m->shadow_lm_head[row + (size_t)j]);
            const float sampled = m->sampled_lm_head[row + (size_t)j];
            const float residual = shadow - sampled;
            a9_insert_top_residual(fabsf(residual), residual, (uint16_t)j,
                                   top_abs, top_values, top_cols, k);
        }
        float row_bound = 0.0f;
        const size_t out = (size_t)v * (size_t)k;
        for (int i = 0; i < k; i++) {
            r->cols[out + (size_t)i] = top_cols[i];
            r->values[out + (size_t)i] = top_values[i];
            row_bound += fabsf(top_values[i]);
            if (top_abs[i] > 0.0f) selected++;
        }
        r->row_abs_bound[v] = row_bound;
        abs_bound_sum += row_bound;
    }

    r->selected_entries = selected;
    r->dense_residual_bytes = (uint64_t)rows * (uint64_t)cols * (uint64_t)sizeof(float);
    r->sparse_residual_bytes =
        selected * (uint64_t)(sizeof(uint16_t) + sizeof(float)) +
        (uint64_t)rows * (uint64_t)sizeof(float);
    r->resident_reduction = r->sparse_residual_bytes > 0
        ? (float)((double)r->dense_residual_bytes / (double)r->sparse_residual_bytes)
        : 0.0f;
    r->enabled = 1;

    fprintf(stderr,
            "[A9_LOGIT_SYNDROME_RESIDUAL] active=1 mode=production_generate_lm_head_sidecar entries_per_row=%d selected_entries=%llu dense_residual_bytes=%llu sparse_residual_bytes=%llu resident_reduction=%.6f mean_abs_bound=%.9g\n",
            r->entries_per_row,
            (unsigned long long)r->selected_entries,
            (unsigned long long)r->dense_residual_bytes,
            (unsigned long long)r->sparse_residual_bytes,
            r->resident_reduction,
            (float)(abs_bound_sum / (double)V12_VOCAB_SIZE));
    return 0;
}

static void a9_frontier_free(A9ExactLogitFrontier *f)
{
    if (!f) return;
    free(f->group_max_pos);
    free(f->group_max_neg);
    memset(f, 0, sizeof(*f));
}

static int a9_frontier_build(A9ExactLogitFrontier *f, const Model *m)
{
    if (!f || !m || !f->requested) return 0;
    if (f->group_size <= 0) f->group_size = 256;
    if (f->group_size < 32) f->group_size = 32;
    if (f->frontier_groups <= 0) f->frontier_groups = 4;
    f->groups = (V12_VOCAB_SIZE + f->group_size - 1) / f->group_size;
    if (f->frontier_groups > f->groups) f->frontier_groups = f->groups;

    f->group_max_pos = (uint16_t *)calloc((size_t)f->groups, sizeof(uint16_t));
    f->group_max_neg = (uint16_t *)calloc((size_t)f->groups, sizeof(uint16_t));
    if (!f->group_max_pos || !f->group_max_neg) {
        a9_frontier_free(f);
        fprintf(stderr,
                "[A9_EXACT_LOGIT_FRONTIER] requested=1 active=0 reason=allocation_failed\n");
        return -1;
    }

    for (int row = 0; row < V12_VOCAB_SIZE; row++) {
        const int g = row / f->group_size;
        const int pos = m->csr_lm_head.n_pos ? m->csr_lm_head.n_pos[row] : 0;
        const int neg = m->csr_lm_head.n_neg ? m->csr_lm_head.n_neg[row] : 0;
        if (pos > f->group_max_pos[g]) f->group_max_pos[g] = (uint16_t)pos;
        if (neg > f->group_max_neg[g]) f->group_max_neg[g] = (uint16_t)neg;
    }
    f->enabled = 1;
    fprintf(stderr,
            "[A9_EXACT_LOGIT_FRONTIER] active=1 mode=production_generate_greedy_certificate group_size=%d groups=%d frontier_groups=%d margin=%.9g full_logits_on_fallback=1\n",
            f->group_size, f->groups, f->frontier_groups, f->margin);
    return 0;
}

static float a9_group_upper_bound(const A9ExactLogitFrontier *f,
                                  const A9LogitSyndromeResidual *r,
                                  int group,
                                  float max_pos,
                                  float max_neg_abs,
                                  float max_abs_norm)
{
    float bound = (float)f->group_max_pos[group] * max_pos +
                  (float)f->group_max_neg[group] * max_neg_abs;
    if (r && r->enabled && r->row_abs_bound) {
        const int start = group * f->group_size;
        int end = start + f->group_size;
        if (end > V12_VOCAB_SIZE) end = V12_VOCAB_SIZE;
        float max_residual = 0.0f;
        for (int row = start; row < end; row++) {
            if (r->row_abs_bound[row] > max_residual)
                max_residual = r->row_abs_bound[row];
        }
        bound += max_residual * max_abs_norm;
    }
    return bound;
}

static int a9_frontier_try_greedy(const Model *m,
                                  A9Runtime *a9,
                                  const float *norm_out,
                                  int *out_token)
{
    A9ExactLogitFrontier *f = &a9->frontier;
    A9LogitSyndromeResidual *r = &a9->residual;
    if (!f->enabled || !out_token) return 0;

    f->attempts++;
    float max_pos = 0.0f;
    float max_neg_abs = 0.0f;
    float max_abs_norm = 0.0f;
    for (int j = 0; j < V12_D_MODEL; j++) {
        const float x = norm_out[j];
        if (x > max_pos) max_pos = x;
        if (-x > max_neg_abs) max_neg_abs = -x;
        if (fabsf(x) > max_abs_norm) max_abs_norm = fabsf(x);
    }

    float *bounds = (float *)malloc((size_t)f->groups * sizeof(float));
    uint8_t *selected = (uint8_t *)calloc((size_t)f->groups, sizeof(uint8_t));
    if (!bounds || !selected) {
        free(bounds);
        free(selected);
        f->fallback++;
        return 0;
    }

    for (int g = 0; g < f->groups; g++)
        bounds[g] = a9_group_upper_bound(f, r, g, max_pos, max_neg_abs, max_abs_norm);

    for (int k = 0; k < f->frontier_groups; k++) {
        int best_g = -1;
        float best_bound = -INFINITY;
        for (int g = 0; g < f->groups; g++) {
            if (!selected[g] && bounds[g] > best_bound) {
                best_bound = bounds[g];
                best_g = g;
            }
        }
        if (best_g >= 0) selected[best_g] = 1u;
    }

    float other_upper = -INFINITY;
    for (int g = 0; g < f->groups; g++) {
        if (!selected[g] && bounds[g] > other_upper)
            other_upper = bounds[g];
    }

    int best_token = 0;
    float best_logit = -INFINITY;
    uint64_t exact_rows = 0;
    for (int g = 0; g < f->groups; g++) {
        if (!selected[g]) continue;
        const int start = g * f->group_size;
        int end = start + f->group_size;
        if (end > V12_VOCAB_SIZE) end = V12_VOCAB_SIZE;
        for (int row = start; row < end; row++) {
            const float logit = compute_a9_logit(m, r, row, norm_out);
            exact_rows++;
            if (logit > best_logit) {
                best_logit = logit;
                best_token = row;
            }
        }
    }
    f->exact_rows += exact_rows;

    const int accepted = best_logit > other_upper + f->margin;
    free(bounds);
    free(selected);
    if (accepted) {
        f->accepted++;
        *out_token = best_token;
        return 1;
    }
    f->fallback++;
    return 0;
}

static void a9_runtime_free(A9Runtime *a9)
{
    if (!a9) return;
    a9_residual_free(&a9->residual);
    a9_frontier_free(&a9->frontier);
}

static void a9_runtime_report(const A9Runtime *a9)
{
    if (!a9) return;
    if (a9->frontier.requested) {
        const double acceptance = a9->frontier.attempts > 0
            ? (double)a9->frontier.accepted / (double)a9->frontier.attempts
            : 0.0;
        fprintf(stderr,
                "[A9_EXACT_LOGIT_FRONTIER_SUMMARY] active=%d attempts=%llu accepted=%llu fallback=%llu acceptance=%.6f exact_rows=%llu full_rows=%llu\n",
                a9->frontier.enabled,
                (unsigned long long)a9->frontier.attempts,
                (unsigned long long)a9->frontier.accepted,
                (unsigned long long)a9->frontier.fallback,
                acceptance,
                (unsigned long long)a9->frontier.exact_rows,
                (unsigned long long)a9->frontier.full_rows);
    }
    if (a9->residual.requested) {
        fprintf(stderr,
                "[A9_LOGIT_SYNDROME_RESIDUAL_SUMMARY] active=%d selected_entries=%llu dense_residual_bytes=%llu sparse_residual_bytes=%llu resident_reduction=%.6f\n",
                a9->residual.enabled,
                (unsigned long long)a9->residual.selected_entries,
                (unsigned long long)a9->residual.dense_residual_bytes,
                (unsigned long long)a9->residual.sparse_residual_bytes,
                a9->residual.resident_reduction);
    }
}

/* =========================================================================
 * Sample next token from logits.
 * temperature <= 0: greedy argmax.
 * top_k == 0: full softmax sampling.
 * ========================================================================= */

static int sample_token(const float *logits, float temperature, int top_k)
{
    int vocab = V12_VOCAB_SIZE;

    if (temperature <= 0.0f) {
        int best = 0;
        for (int v = 1; v < vocab; v++)
            if (logits[v] > logits[best]) best = v;
        return best;
    }

    /* Temperature scaling */
    static float scaled[V12_VOCAB_SIZE];
    float inv_temp = 1.0f / temperature;
    float max_logit = logits[0];
    for (int v = 1; v < vocab; v++)
        if (logits[v] > max_logit) max_logit = logits[v];

    /* Softmax numerically stable */
    float sum = 0.0f;
    for (int v = 0; v < vocab; v++) {
        scaled[v] = expf((logits[v] - max_logit) * inv_temp);
        sum += scaled[v];
    }

    if (top_k > 0 && top_k < vocab) {
        /* Zero out all but top-k entries */
        static float top_vals[V12_VOCAB_SIZE];
        memcpy(top_vals, scaled, vocab * sizeof(float));

        /* Find k-th largest value via partial sort threshold */
        static float sorted_tmp[V12_VOCAB_SIZE];
        memcpy(sorted_tmp, scaled, vocab * sizeof(float));

        /* Simple selection: find the top_k-th value */
        int k = (top_k < vocab) ? top_k : vocab;
        float threshold = 0.0f;
        for (int iter = 0; iter < k; iter++) {
            float cur_max = -1.0f;
            for (int v = 0; v < vocab; v++)
                if (sorted_tmp[v] > cur_max) cur_max = sorted_tmp[v];
            threshold = cur_max;
            for (int v = 0; v < vocab; v++)
                if (sorted_tmp[v] == cur_max) { sorted_tmp[v] = -1.0f; break; }
        }

        /* Zero out entries below threshold and recompute sum */
        sum = 0.0f;
        for (int v = 0; v < vocab; v++) {
            if (scaled[v] < threshold) scaled[v] = 0.0f;
            sum += scaled[v];
        }
    }

    if (sum <= 0.0f) {
        /* Fallback to uniform if all zero */
        return (int)(rng_next() % vocab);
    }

    float r = rng_uniform() * sum;
    float cumsum = 0.0f;
    for (int v = 0; v < vocab; v++) {
        cumsum += scaled[v];
        if (cumsum >= r) return v;
    }
    return vocab - 1;
}

/* =========================================================================
 * BPE vocab table — populated by load_vocab() when --vocab is passed.
 * ========================================================================= */

static char *vocab_strings[V12_VOCAB_SIZE];
static int   vocab_loaded = 0;

static void load_vocab(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open vocab: %s\n", path); return; }
    char line[4096];
    int id = 0;
    while (fgets(line, sizeof(line), f) && id < V12_VOCAB_SIZE) {
        int len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
        vocab_strings[id] = strdup(line);
        id++;
    }
    fclose(f);
    vocab_loaded = id;
    fprintf(stderr, "[vocab] Loaded %d tokens from %s\n", id, path);
}

static void free_vocab(void)
{
    for (int i = 0; i < vocab_loaded; i++) {
        free(vocab_strings[i]);
        vocab_strings[i] = NULL;
    }
}

/* =========================================================================
 * Output one token to stdout.
 * Uses vocab table when loaded; falls back to raw byte or <T%d> placeholder.
 * ========================================================================= */

static void output_token(int token)
{
    if (vocab_loaded > 0 && token < vocab_loaded && vocab_strings[token]) {
        printf("%s", vocab_strings[token]);
    } else if (token >= 0 && token < 256) {
        putchar((unsigned char)token);
    } else {
        printf("<T%d>", token);
    }
    fflush(stdout);
}

/* =========================================================================
 * Full autoregressive forward for one token.
 * Returns the next token id.
 * ========================================================================= */

static int generate_next_token(const Model *m,
                                GenState *gs,
                                A9Runtime *a9,
                                int input_token,
                                float temperature,
                                int top_k,
                                float (*conv_kernels)[V12_CONV_KERNEL * V12_D_STATE])
{
    int d  = V12_D_MODEL;
    int ds = V12_D_STATE;

    float residual[V12_D_MODEL];
    embed_token(m, input_token, residual);

    if (gs->tokens_generated > 0)
        apply_predictive_coding(m, residual, gs->prev_residual);

    memcpy(gs->prev_residual, residual, d * sizeof(float));

    float residual_next[V12_D_MODEL];
    for (int k = 0; k < V12_N_LAYERS; k++) {
        forward_layer_single_token(m, &m->layers[k],
                                   &gs->layers[k],
                                   residual, residual_next,
                                   conv_kernels[k]);
        memcpy(residual, residual_next, d * sizeof(float));
    }

    float norm_out[V12_D_MODEL];
    apply_rms_norm(residual, m->norm_w, m->norm_b, norm_out, d);

    static float logits[V12_VOCAB_SIZE];
    int frontier_token = 0;
    if (temperature <= 0.0f &&
        a9 &&
        a9_frontier_try_greedy(m, a9, norm_out, &frontier_token)) {
        gs->tokens_generated++;
        return frontier_token;
    }

    if (a9 && a9->frontier.enabled)
        a9->frontier.full_rows += (uint64_t)V12_VOCAB_SIZE;
    if (a9 && a9->residual.enabled)
        compute_full_logits_a9(m, &a9->residual, norm_out, logits);
    else
        compute_full_logits(m, norm_out, logits);

    gs->tokens_generated++;

    return sample_token(logits, temperature, top_k);
}

/* =========================================================================
 * CLI argument parsing helpers
 * ========================================================================= */

static int parse_int_arg(int argc, char **argv, const char *flag, int def)
{
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], flag) == 0) return atoi(argv[i + 1]);
    return def;
}

static float parse_float_arg(int argc, char **argv, const char *flag, float def)
{
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], flag) == 0) return (float)atof(argv[i + 1]);
    return def;
}

/* =========================================================================
 * main
 * ========================================================================= */

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <checkpoint_path> "
                "[--max-tokens N] [--temperature T] [--top-k K] [--seed S] [--vocab PATH] "
                "[--exact-logit-frontier] [--frontier-group-size N] [--frontier-groups N] "
                "[--logit-syndrome-residual] [--residual-entries-per-row N]\n",
                argv[0]);
        return 1;
    }

    const char *ckpt_path = argv[1];
    int   max_tokens  = parse_int_arg  (argc, argv, "--max-tokens",  256);
    float temperature = parse_float_arg(argc, argv, "--temperature", 1.0f);
    int   top_k       = parse_int_arg  (argc, argv, "--top-k",       40);
    int   seed        = parse_int_arg  (argc, argv, "--seed",        (int)time(NULL));
    A9Runtime a9;
    memset(&a9, 0, sizeof(a9));
    a9.frontier.group_size = parse_int_arg(argc, argv, "--frontier-group-size", 256);
    a9.frontier.frontier_groups = parse_int_arg(argc, argv, "--frontier-groups", 4);
    a9.frontier.margin = parse_float_arg(argc, argv, "--frontier-margin", 0.0f);
    a9.residual.entries_per_row = parse_int_arg(argc, argv, "--residual-entries-per-row", 4);

    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--vocab") == 0) {
            load_vocab(argv[++i]);
        }
    }
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--exact-logit-frontier") == 0) {
            a9.frontier.requested = 1;
        } else if (strcmp(argv[i], "--logit-syndrome-residual") == 0) {
            a9.residual.requested = 1;
        }
    }

    rng_state_g = (uint64_t)seed ^ 0xdeadbeefcafe1234ULL;
    if (rng_state_g == 0) rng_state_g = 1;

    Model *m = model_alloc();
    if (!m) {
        fprintf(stderr, "model_alloc failed\n");
        return 1;
    }

    TrainState ts;
    memset(&ts, 0, sizeof(ts));

    if (materialize_checkpoint_host_state(m) != 0) {
        model_free(m);
        return 1;
    }

    if (checkpoint_load(ckpt_path, m, &ts) != 0) {
        fprintf(stderr, "Failed to load checkpoint: %s\n", ckpt_path);
        model_free(m);
        return 1;
    }

    if (checkpoint_rebuild(m, &ts) != 0) {
        fprintf(stderr, "checkpoint_rebuild failed\n");
        model_free(m);
        return 1;
    }

    if (a9_residual_build(&a9.residual, m) != 0) {
        model_free(m);
        a9_runtime_free(&a9);
        return 1;
    }
    if (a9_frontier_build(&a9.frontier, m) != 0) {
        model_free(m);
        a9_runtime_free(&a9);
        return 1;
    }

    /* Decode conv kernels for all layers once (int2 -> float) */
    static float conv_kernels[V12_N_LAYERS][V12_CONV_KERNEL * V12_D_STATE];
    for (int k = 0; k < V12_N_LAYERS; k++)
        decode_conv_kernel(m->layers[k].w_conv_int2,
                           conv_kernels[k],
                           V12_CONV_KERNEL, V12_D_STATE);

    GenState gs;
    memset(&gs, 0, sizeof(gs));

    int token = 0;

    for (int t = 0; t < max_tokens; t++) {
        token = generate_next_token(m, &gs, &a9, token, temperature, top_k, conv_kernels);
        output_token(token);
    }
    putchar('\n');

    a9_runtime_report(&a9);
    a9_runtime_free(&a9);
    model_free(m);
    free_vocab();
    return 0;
}
