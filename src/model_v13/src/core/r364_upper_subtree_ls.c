#include "r364_upper_subtree_ls.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "ternary_matmul.h"

#define R364_MAX_NEGATIVE_ROWS      256
#define R364_MAX_SUPPORT             64
#define R364_THRESHOLD_CANDIDATES     9
#define R364_RESIDUAL_PASSES          2
#define R364_ACCEPT_REL_TOL        0.01f
#define R364_ACCEPT_ABS_TOL        1e-4f
#define R364_KAPPA_REJECT          1e8f

typedef struct {
    int vocab_id;
    int pos_count;
    int neg_count;
} R364RowScore;

typedef struct {
    int changed;
    int vocab_id;
    float *candidate_row;
} R364RowSolution;

static void r364_set_reason(R364UpperSubtreeLsProbe *probe, const char *reason) {
    if (!probe) return;
    if (!reason) reason = "";
    snprintf(probe->rejection_reason, sizeof(probe->rejection_reason), "%s", reason);
}

float r364_upper_subtree_ls_measure_last_block_share_f(const Model *m) {
    const size_t lm_head_elems = (size_t)V12_VOCAB_SIZE * V12_D_MODEL;
    size_t total_elems = lm_head_elems;

    if (m && (m->gpu_opt_flags & GPU_OPT_W_PRED)) {
        total_elems += (size_t)V12_D_MODEL * V12_D_MODEL;
    }
    if (m && (m->gpu_opt_flags & GPU_OPT_NORM)) {
        total_elems += (size_t)V12_D_MODEL * 2u;
    }
    if (m && (m->gpu_opt_flags & GPU_OPT_W_IN)) {
        total_elems += (size_t)V12_N_LAYERS * V12_D_MODEL * V12_D_STATE;
    }
    if (m && (m->gpu_opt_flags & GPU_OPT_W_OUT)) {
        total_elems += (size_t)V12_N_LAYERS * V12_D_STATE * V12_D_MODEL;
    }

    if (total_elems == 0u) return 0.0f;
    return (float)((double)lm_head_elems / (double)total_elems);
}

static int r364_compare_row_score(const void *lhs, const void *rhs) {
    const R364RowScore *a = (const R364RowScore *)lhs;
    const R364RowScore *b = (const R364RowScore *)rhs;
    if (a->pos_count != b->pos_count) return b->pos_count - a->pos_count;
    if (a->neg_count != b->neg_count) return b->neg_count - a->neg_count;
    return a->vocab_id - b->vocab_id;
}

static void r364_copy_row(float *dst, const float *src) {
    memcpy(dst, src, (size_t)V12_D_MODEL * sizeof(float));
}

static int r364_rows_equal(const float *a, const float *b) {
    for (int i = 0; i < V12_D_MODEL; i++) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

static float r364_fixed_sample_loss_with_model(const Model *m,
                                               const Cache *c,
                                               const float *norm_out_seq,
                                               int seq_len) {
    const int n_tokens = seq_len - 1;
    const int n_samples = V12_SS_N_SAMPLES;
    const int k = V12_SS_K;
    const float log_k = logf((float)k);
    float ce_sum = 0.0f;

    for (int t = 0; t < n_tokens; t++) {
        const int32_t *sample_ids = c->sample_ids + (size_t)t * n_samples;
        const float *log_proposal = c->log_proposal + (size_t)t * n_samples;
        const float *h = norm_out_seq + (size_t)t * V12_D_MODEL;
        float adjusted[V12_SS_N_SAMPLES];

        adjusted[0] = ternary_dot_row(&m->csr_lm_head, sample_ids[0], h);
        for (int s = 1; s < n_samples; s++) {
            const float logit = ternary_dot_row(&m->csr_lm_head, sample_ids[s], h);
            adjusted[s] = logit - (log_k + log_proposal[s]);
        }

        float max_adjusted = adjusted[0];
        for (int s = 1; s < n_samples; s++) {
            if (adjusted[s] > max_adjusted) max_adjusted = adjusted[s];
        }

        float sum_exp = 0.0f;
        for (int s = 0; s < n_samples; s++) {
            sum_exp += expf(adjusted[s] - max_adjusted);
        }

        ce_sum -= (adjusted[0] - max_adjusted) - logf(sum_exp);
    }

    return ce_sum / (float)n_tokens;
}

static int r364_eval_sample_ids_valid(const Cache *c,
                                      int n_tokens,
                                      int n_samples,
                                      int *bad_token,
                                      int *bad_sample,
                                      int32_t *bad_id) {
    if (!c || !c->sample_ids) return 0;
    for (int t = 0; t < n_tokens; t++) {
        const int32_t *sample_ids = c->sample_ids + (size_t)t * n_samples;
        for (int s = 0; s < n_samples; s++) {
            const int32_t sample_id = sample_ids[s];
            if (sample_id < 0 || sample_id >= V12_VOCAB_SIZE) {
                if (bad_token) *bad_token = t;
                if (bad_sample) *bad_sample = s;
                if (bad_id) *bad_id = sample_id;
                return 0;
            }
        }
    }
    return 1;
}

static void r364_build_regularized_gram(const float *x_rows,
                                        int n_rows,
                                        double *gram_out,
                                        double *lambda_out) {
    const int d = V12_D_MODEL;
    const double inv_n = 1.0 / (double)n_rows;

    memset(gram_out, 0, (size_t)d * d * sizeof(double));
    for (int t = 0; t < n_rows; t++) {
        const float *x = x_rows + (size_t)t * d;
        for (int i = 0; i < d; i++) {
            const double xi = (double)x[i];
            for (int j = 0; j <= i; j++) {
                gram_out[(size_t)i * d + j] += xi * (double)x[j];
            }
        }
    }

    double diag_sum = 0.0;
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < i; j++) {
            gram_out[(size_t)i * d + j] *= inv_n;
            gram_out[(size_t)j * d + i] = gram_out[(size_t)i * d + j];
        }
        gram_out[(size_t)i * d + i] *= inv_n;
        diag_sum += gram_out[(size_t)i * d + i];
    }

    const double mean_diag = diag_sum / (double)d;
    const double lambda = fmax(1e-4, mean_diag * 1e-2);
    for (int i = 0; i < d; i++) {
        gram_out[(size_t)i * d + i] += lambda;
    }

    if (lambda_out) *lambda_out = lambda;
}

static int r364_cholesky_decompose(double *a, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = a[(size_t)i * n + j];
            for (int k = 0; k < j; k++) {
                sum -= a[(size_t)i * n + k] * a[(size_t)j * n + k];
            }
            if (i == j) {
                if (!(sum > 0.0) || !isfinite(sum)) return -1;
                a[(size_t)i * n + j] = sqrt(sum);
            } else {
                a[(size_t)i * n + j] = sum / a[(size_t)j * n + j];
            }
        }
        for (int j = i + 1; j < n; j++) {
            a[(size_t)i * n + j] = 0.0;
        }
    }
    return 0;
}

static void r364_cholesky_solve(const double *chol,
                                const double *rhs,
                                double *out,
                                int n) {
    double *tmp = (double *)malloc((size_t)n * sizeof(double));
    if (!tmp) {
        memset(out, 0, (size_t)n * sizeof(double));
        return;
    }

    for (int i = 0; i < n; i++) {
        double sum = rhs[i];
        for (int j = 0; j < i; j++) {
            sum -= chol[(size_t)i * n + j] * tmp[j];
        }
        tmp[i] = sum / chol[(size_t)i * n + i];
    }

    for (int i = n - 1; i >= 0; i--) {
        double sum = tmp[i];
        for (int j = i + 1; j < n; j++) {
            sum -= chol[(size_t)j * n + i] * out[j];
        }
        out[i] = sum / chol[(size_t)i * n + i];
    }

    free(tmp);
}

static void r364_matvec(const double *mat, const double *x, double *out, int n) {
    for (int i = 0; i < n; i++) {
        const double *row = mat + (size_t)i * n;
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
}

static double r364_dot(const double *a, const double *b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static void r364_normalize(double *x, int n) {
    const double norm = sqrt(fmax(r364_dot(x, x, n), 1e-30));
    for (int i = 0; i < n; i++) x[i] /= norm;
}

static float r364_estimate_kappa(const double *gram,
                                 const double *chol,
                                 int n) {
    double *v = (double *)malloc((size_t)n * sizeof(double));
    double *tmp = (double *)malloc((size_t)n * sizeof(double));
    double *inv = (double *)malloc((size_t)n * sizeof(double));
    if (!v || !tmp || !inv) {
        free(v);
        free(tmp);
        free(inv);
        return NAN;
    }

    for (int i = 0; i < n; i++) v[i] = 1.0;
    r364_normalize(v, n);
    for (int it = 0; it < 8; it++) {
        r364_matvec(gram, v, tmp, n);
        r364_normalize(tmp, n);
        memcpy(v, tmp, (size_t)n * sizeof(double));
    }
    r364_matvec(gram, v, tmp, n);
    const double lambda_max = r364_dot(v, tmp, n);

    for (int i = 0; i < n; i++) v[i] = 1.0;
    r364_normalize(v, n);
    for (int it = 0; it < 8; it++) {
        memset(inv, 0, (size_t)n * sizeof(double));
        r364_cholesky_solve(chol, v, inv, n);
        r364_normalize(inv, n);
        memcpy(v, inv, (size_t)n * sizeof(double));
    }
    r364_matvec(gram, v, tmp, n);
    const double lambda_min = r364_dot(v, tmp, n);

    free(v);
    free(tmp);
    free(inv);

    if (!(lambda_min > 0.0) || !isfinite(lambda_max) || !isfinite(lambda_min)) {
        return NAN;
    }
    return (float)(lambda_max / lambda_min);
}

static void r364_build_rhs(const float *x_rows,
                           const float *targets,
                           int n_rows,
                           double *rhs_out) {
    const int d = V12_D_MODEL;
    const double inv_n = 1.0 / (double)n_rows;

    memset(rhs_out, 0, (size_t)d * sizeof(double));
    for (int t = 0; t < n_rows; t++) {
        const double y = (double)targets[t];
        if (fabs(y) < 1e-12) continue;
        const float *x = x_rows + (size_t)t * d;
        for (int j = 0; j < d; j++) {
            rhs_out[j] += y * (double)x[j];
        }
    }
    for (int j = 0; j < d; j++) {
        rhs_out[j] *= inv_n;
    }
}

static float r364_candidate_objective(const float *x_rows,
                                      const float *targets,
                                      int n_rows,
                                      const int *support_idx,
                                      const float *support_sign,
                                      int support_len,
                                      double lambda) {
    const int d = V12_D_MODEL;
    float loss = 0.0f;

    for (int t = 0; t < n_rows; t++) {
        const float *x = x_rows + (size_t)t * d;
        float pred = 0.0f;
        for (int i = 0; i < support_len; i++) {
            pred += support_sign[i] * x[support_idx[i]];
        }
        const float err = pred - targets[t];
        loss += err * err;
    }

    loss /= (float)n_rows;
    loss += (float)(lambda * (double)support_len);
    return loss;
}

static int r364_support_from_threshold(const double *weights,
                                       double threshold,
                                       int *support_idx,
                                       float *support_sign,
                                       int max_support) {
    double best_abs[R364_MAX_SUPPORT];
    int best_idx[R364_MAX_SUPPORT];
    float best_sign[R364_MAX_SUPPORT];
    int chosen = 0;

    for (int i = 0; i < max_support; i++) {
        best_abs[i] = -1.0;
        best_idx[i] = -1;
        best_sign[i] = 0.0f;
    }

    for (int j = 0; j < V12_D_MODEL; j++) {
        const double magnitude = fabs(weights[j]);
        if (!(magnitude > threshold)) continue;
        if (chosen >= max_support && magnitude <= best_abs[max_support - 1]) continue;

        int insert_at = chosen;
        if (insert_at >= max_support) insert_at = max_support - 1;
        while (insert_at > 0 && magnitude > best_abs[insert_at - 1]) {
            if (insert_at < max_support) {
                best_abs[insert_at] = best_abs[insert_at - 1];
                best_idx[insert_at] = best_idx[insert_at - 1];
                best_sign[insert_at] = best_sign[insert_at - 1];
            }
            insert_at--;
        }

        if (chosen < max_support) chosen++;
        if (insert_at < max_support) {
            best_abs[insert_at] = magnitude;
            best_idx[insert_at] = j;
            best_sign[insert_at] = (weights[j] >= 0.0) ? 1.0f : -1.0f;
        }
    }

    for (int i = 0; i < chosen; i++) {
        support_idx[i] = best_idx[i];
        support_sign[i] = best_sign[i];
    }
    return chosen;
}

static void r364_support_to_dense(const int *support_idx,
                                  const float *support_sign,
                                  int support_len,
                                  float *dense_row) {
    memset(dense_row, 0, (size_t)V12_D_MODEL * sizeof(float));
    for (int i = 0; i < support_len; i++) {
        dense_row[support_idx[i]] = support_sign[i];
    }
}

static float r364_solve_row(const float *x_rows,
                            const float *targets,
                            int n_rows,
                            const double *chol,
                            double lambda,
                            float *out_row) {
    double *rhs = (double *)malloc((size_t)V12_D_MODEL * sizeof(double));
    double *solution = (double *)malloc((size_t)V12_D_MODEL * sizeof(double));
    float *best_row = (float *)malloc((size_t)V12_D_MODEL * sizeof(float));
    float *trial_row = (float *)malloc((size_t)V12_D_MODEL * sizeof(float));
    float *residual_targets = (float *)malloc((size_t)n_rows * sizeof(float));
    if (!rhs || !solution || !best_row || !trial_row || !residual_targets) {
        free(rhs);
        free(solution);
        free(best_row);
        free(trial_row);
        free(residual_targets);
        memset(out_row, 0, (size_t)V12_D_MODEL * sizeof(float));
        return INFINITY;
    }

    r364_build_rhs(x_rows, targets, n_rows, rhs);
    r364_cholesky_solve(chol, rhs, solution, V12_D_MODEL);

    float best_objective = INFINITY;
    double max_abs = 0.0;
    for (int j = 0; j < V12_D_MODEL; j++) {
        const double magnitude = fabs(solution[j]);
        if (magnitude > max_abs) max_abs = magnitude;
    }

    if (!(max_abs > 0.0) || !isfinite(max_abs)) {
        memset(out_row, 0, (size_t)V12_D_MODEL * sizeof(float));
        free(rhs);
        free(solution);
        free(best_row);
        free(trial_row);
        free(residual_targets);
        return best_objective;
    }

    for (int c = 0; c < R364_THRESHOLD_CANDIDATES; c++) {
        int support_idx[R364_MAX_SUPPORT];
        float support_sign[R364_MAX_SUPPORT];
        const double threshold = max_abs * ((double)c / (double)(R364_THRESHOLD_CANDIDATES - 1));
        const int support_len = r364_support_from_threshold(
            solution, threshold, support_idx, support_sign, R364_MAX_SUPPORT);
        const float objective = r364_candidate_objective(
            x_rows, targets, n_rows, support_idx, support_sign, support_len, lambda);
        if (objective < best_objective) {
            best_objective = objective;
            r364_support_to_dense(support_idx, support_sign, support_len, best_row);
        }
    }

    for (int pass = 0; pass < R364_RESIDUAL_PASSES; pass++) {
        for (int t = 0; t < n_rows; t++) {
            const float *x = x_rows + (size_t)t * V12_D_MODEL;
            float pred = 0.0f;
            for (int j = 0; j < V12_D_MODEL; j++) {
                if (best_row[j] != 0.0f) pred += best_row[j] * x[j];
            }
            residual_targets[t] = targets[t] - pred;
        }

        r364_build_rhs(x_rows, residual_targets, n_rows, rhs);
        r364_cholesky_solve(chol, rhs, solution, V12_D_MODEL);

        max_abs = 0.0;
        for (int j = 0; j < V12_D_MODEL; j++) {
            const double magnitude = fabs(solution[j]);
            if (magnitude > max_abs) max_abs = magnitude;
        }
        if (!(max_abs > 0.0) || !isfinite(max_abs)) break;

        for (int c = 0; c < R364_THRESHOLD_CANDIDATES; c++) {
            int support_idx[R364_MAX_SUPPORT];
            float support_sign[R364_MAX_SUPPORT];
            const double threshold = max_abs * ((double)c / (double)(R364_THRESHOLD_CANDIDATES - 1));
            const int support_len = r364_support_from_threshold(
                solution, threshold, support_idx, support_sign, R364_MAX_SUPPORT / 2);

            r364_copy_row(trial_row, best_row);
            for (int i = 0; i < support_len; i++) {
                const int idx = support_idx[i];
                trial_row[idx] += support_sign[i];
                if (trial_row[idx] > 1.0f) trial_row[idx] = 1.0f;
                if (trial_row[idx] < -1.0f) trial_row[idx] = -1.0f;
                if (fabsf(trial_row[idx]) < 0.5f) trial_row[idx] = 0.0f;
            }

            int combined_idx[R364_MAX_SUPPORT];
            float combined_sign[R364_MAX_SUPPORT];
            int combined_len = 0;
            for (int j = 0; j < V12_D_MODEL && combined_len < R364_MAX_SUPPORT; j++) {
                if (trial_row[j] != 0.0f) {
                    combined_idx[combined_len] = j;
                    combined_sign[combined_len] = trial_row[j];
                    combined_len++;
                }
            }

            const float objective = r364_candidate_objective(
                x_rows, targets, n_rows, combined_idx, combined_sign, combined_len, lambda);
            if (objective + 1e-6f < best_objective) {
                best_objective = objective;
                r364_copy_row(best_row, trial_row);
            }
        }
    }

    r364_copy_row(out_row, best_row);

    free(rhs);
    free(solution);
    free(best_row);
    free(trial_row);
    free(residual_targets);
    return best_objective;
}

static int r364_restore_rows(Model *m,
                             const int *vocab_rows,
                             int n_rows,
                             const shadow_t *shadow_backup,
                             const float *sampled_backup) {
    const size_t row_elems = (size_t)V12_D_MODEL;
    for (int i = 0; i < n_rows; i++) {
        const int vocab_id = vocab_rows[i];
        memcpy(
            m->shadow_lm_head + (size_t)vocab_id * row_elems,
            shadow_backup + (size_t)i * row_elems,
            row_elems * SHADOW_SIZEOF);
        memcpy(
            m->sampled_lm_head + (size_t)vocab_id * row_elems,
            sampled_backup + (size_t)i * row_elems,
            row_elems * sizeof(float));
    }
    return model_rebuild_lm_head_csr(m);
}

int r364_upper_subtree_ls_compile(Model *m,
                                  Cache *c,
                                  const R364UpperSubtreeLsCompileConfig *config,
                                  R364UpperSubtreeLsProbe *probe) {
    const int seq_len = V12_SEQ_LEN;
    const int n_tokens = seq_len - 1;
    const int n_samples = V12_SS_N_SAMPLES;
    const int vocab_size = V12_VOCAB_SIZE;
    const size_t row_elems = (size_t)V12_D_MODEL;
    const float negative_weight = 1.0f / (float)V12_SS_K;
    const int eval_batch = (config && config->eval_batch_index >= 0) ? config->eval_batch_index : 0;

    int *pos_counts = NULL;
    int *neg_counts = NULL;
    int *active_map = NULL;
    int *changed_vocab_rows = NULL;
    R364RowScore *active_rows = NULL;
    R364RowScore *negative_rows = NULL;
    R364RowSolution *solutions = NULL;
    float *row_targets = NULL;
    float *candidate_matrix = NULL;
    shadow_t *shadow_backup = NULL;
    float *sampled_backup = NULL;
    double *gram = NULL;
    double *chol = NULL;

    if (!probe || !config) return -1;

    memset(probe, 0, sizeof(*probe));
    probe->kappa = NAN;
    probe->pre_loss = NAN;
    probe->post_loss = NAN;
    probe->loss_delta = NAN;
    probe->compiled_block_share_f = r364_upper_subtree_ls_measure_last_block_share_f(m);
    probe->noop_hook_armed = config->enable_noop_hook ? 1 : 0;
    r364_set_reason(probe, "accepted");

    if (!m || !c
        || !m->shadow_lm_head
        || !m->sampled_lm_head
        || !m->csr_lm_head.row_ptr_pos
        || !m->csr_lm_head.row_ptr_neg
        || !m->csr_lm_head.pos_indices
        || !m->csr_lm_head.neg_indices
        || !c->norm_out
        || !c->sample_ids
        || !c->log_proposal) {
        r364_set_reason(probe, "missing-materialized-state");
        return -1;
    }

    const float *norm_out_seq = c->norm_out + (size_t)eval_batch * seq_len * V12_D_MODEL;
    {
        int bad_token = -1;
        int bad_sample = -1;
        int32_t bad_id = -1;
        if (!r364_eval_sample_ids_valid(c, n_tokens, n_samples, &bad_token, &bad_sample, &bad_id)) {
            probe->reverted_update = 1;
            snprintf(probe->rejection_reason, sizeof(probe->rejection_reason),
                     "sample-id-out-of-range t=%d s=%d id=%d",
                     bad_token, bad_sample, (int)bad_id);
            return -1;
        }
    }
    probe->pre_loss = r364_fixed_sample_loss_with_model(m, c, norm_out_seq, seq_len);
    if (!isfinite(probe->pre_loss)) {
        probe->reverted_update = 1;
        r364_set_reason(probe, "non-finite-pre-loss");
        return 0;
    }

    pos_counts = (int *)calloc((size_t)vocab_size, sizeof(int));
    neg_counts = (int *)calloc((size_t)vocab_size, sizeof(int));
    active_map = (int *)malloc((size_t)vocab_size * sizeof(int));
    active_rows = (R364RowScore *)malloc((size_t)(n_tokens + R364_MAX_NEGATIVE_ROWS) * sizeof(R364RowScore));
    negative_rows = (R364RowScore *)malloc((size_t)vocab_size * sizeof(R364RowScore));
    if (!pos_counts || !neg_counts || !active_map || !active_rows || !negative_rows) {
        r364_set_reason(probe, "oom-row-selection");
        goto fail;
    }
    for (int v = 0; v < vocab_size; v++) active_map[v] = -1;

    for (int t = 0; t < n_tokens; t++) {
        const int32_t *sample_ids = c->sample_ids + (size_t)t * n_samples;
        pos_counts[sample_ids[0]]++;
        for (int s = 1; s < n_samples; s++) {
            neg_counts[sample_ids[s]]++;
        }
    }

    int active_count = 0;
    for (int v = 0; v < vocab_size; v++) {
        if (pos_counts[v] > 0) {
            active_rows[active_count].vocab_id = v;
            active_rows[active_count].pos_count = pos_counts[v];
            active_rows[active_count].neg_count = neg_counts[v];
            active_count++;
        }
    }
    qsort(active_rows, (size_t)active_count, sizeof(R364RowScore), r364_compare_row_score);

    int negative_count = 0;
    for (int v = 0; v < vocab_size; v++) {
        if (pos_counts[v] == 0 && neg_counts[v] > 0) {
            negative_rows[negative_count].vocab_id = v;
            negative_rows[negative_count].pos_count = 0;
            negative_rows[negative_count].neg_count = neg_counts[v];
            negative_count++;
        }
    }
    qsort(negative_rows, (size_t)negative_count, sizeof(R364RowScore), r364_compare_row_score);

    {
        const int keep_negatives = (negative_count < R364_MAX_NEGATIVE_ROWS)
            ? negative_count
            : R364_MAX_NEGATIVE_ROWS;
        for (int i = 0; i < keep_negatives; i++) {
            active_rows[active_count++] = negative_rows[i];
        }
    }

    if (active_count <= 0) {
        probe->reverted_update = 1;
        r364_set_reason(probe, "no-candidate-rows");
        goto done;
    }
    probe->candidate_rows = (uint32_t)active_count;

    for (int i = 0; i < active_count; i++) {
        active_map[active_rows[i].vocab_id] = i;
    }

    row_targets = (float *)calloc((size_t)active_count * n_tokens, sizeof(float));
    gram = (double *)malloc((size_t)V12_D_MODEL * V12_D_MODEL * sizeof(double));
    chol = (double *)malloc((size_t)V12_D_MODEL * V12_D_MODEL * sizeof(double));
    solutions = (R364RowSolution *)calloc((size_t)active_count, sizeof(R364RowSolution));
    candidate_matrix = (float *)calloc((size_t)active_count * V12_D_MODEL, sizeof(float));
    changed_vocab_rows = (int *)malloc((size_t)active_count * sizeof(int));
    if (!row_targets || !gram || !chol || !solutions || !candidate_matrix || !changed_vocab_rows) {
        r364_set_reason(probe, "oom-solver-state");
        goto fail;
    }

    for (int t = 0; t < n_tokens; t++) {
        const int32_t *sample_ids = c->sample_ids + (size_t)t * n_samples;
        const int target_idx = active_map[sample_ids[0]];
        if (target_idx >= 0) {
            row_targets[(size_t)target_idx * n_tokens + t] += 1.0f;
        }
        for (int s = 1; s < n_samples; s++) {
            const int neg_idx = active_map[sample_ids[s]];
            if (neg_idx >= 0) {
                row_targets[(size_t)neg_idx * n_tokens + t] -= negative_weight;
            }
        }
    }

    double lambda = 0.0;
    r364_build_regularized_gram(norm_out_seq, n_tokens, gram, &lambda);
    memcpy(chol, gram, (size_t)V12_D_MODEL * V12_D_MODEL * sizeof(double));
    if (r364_cholesky_decompose(chol, V12_D_MODEL) != 0) {
        probe->reverted_update = 1;
        r364_set_reason(probe, "ill-conditioned-gram");
        goto done;
    }

    probe->kappa = r364_estimate_kappa(gram, chol, V12_D_MODEL);
    if (!isfinite(probe->kappa) || probe->kappa > R364_KAPPA_REJECT) {
        probe->reverted_update = 1;
        r364_set_reason(probe, "ill-conditioned-kappa");
        goto done;
    }

    int changed_count = 0;
    for (int i = 0; i < active_count; i++) {
        const int vocab_id = active_rows[i].vocab_id;
        const float *targets = row_targets + (size_t)i * n_tokens;
        float *candidate_row = candidate_matrix + (size_t)i * V12_D_MODEL;
        (void)r364_solve_row(norm_out_seq, targets, n_tokens, chol, lambda, candidate_row);

        if (!r364_rows_equal(candidate_row, m->sampled_lm_head + (size_t)vocab_id * row_elems)) {
            solutions[i].changed = 1;
            solutions[i].vocab_id = vocab_id;
            solutions[i].candidate_row = candidate_row;
            changed_vocab_rows[changed_count++] = vocab_id;
        }
    }

    if (changed_count <= 0) {
        probe->reverted_update = 1;
        r364_set_reason(probe, "no-row-updates");
        goto done;
    }

    shadow_backup = (shadow_t *)malloc((size_t)changed_count * row_elems * SHADOW_SIZEOF);
    sampled_backup = (float *)malloc((size_t)changed_count * row_elems * sizeof(float));
    if (!shadow_backup || !sampled_backup) {
        r364_set_reason(probe, "oom-backup");
        goto fail;
    }

    for (int i = 0; i < changed_count; i++) {
        const int vocab_id = changed_vocab_rows[i];
        memcpy(
            shadow_backup + (size_t)i * row_elems,
            m->shadow_lm_head + (size_t)vocab_id * row_elems,
            row_elems * SHADOW_SIZEOF);
        memcpy(
            sampled_backup + (size_t)i * row_elems,
            m->sampled_lm_head + (size_t)vocab_id * row_elems,
            row_elems * sizeof(float));
    }

    for (int i = 0; i < active_count; i++) {
        if (!solutions[i].changed) continue;
        const int vocab_id = solutions[i].vocab_id;
        const float *candidate_row = solutions[i].candidate_row;
        for (int j = 0; j < V12_D_MODEL; j++) {
            const size_t flat_idx = (size_t)vocab_id * row_elems + (size_t)j;
#if V12_F2_INT8_SHADOWS
            const uint32_t rnd = shadow_rng(
                0, V12_SHADOW_MAT_LM_HEAD, (int)flat_idx, (uint64_t)config->boundary_step);
#else
            const uint32_t rnd = 0u;
#endif
            SHADOW_STORE(m->shadow_lm_head[flat_idx], candidate_row[j], rnd);
            m->sampled_lm_head[flat_idx] = candidate_row[j];
        }
    }
    probe->updated_rows = (uint32_t)changed_count;

    if (model_rebuild_lm_head_csr(m) != 0) {
        (void)r364_restore_rows(m, changed_vocab_rows, changed_count, shadow_backup, sampled_backup);
        r364_set_reason(probe, "csr-rebuild-failed");
        goto fail;
    }

    probe->post_loss = r364_fixed_sample_loss_with_model(m, c, norm_out_seq, seq_len);
    probe->loss_delta = probe->post_loss - probe->pre_loss;
    if (!isfinite(probe->post_loss)) {
        (void)r364_restore_rows(m, changed_vocab_rows, changed_count, shadow_backup, sampled_backup);
        probe->updated_rows = 0;
        probe->reverted_update = 1;
        r364_set_reason(probe, "non-finite-post-loss");
        goto done;
    }

    {
        const float allowed_loss = probe->pre_loss * (1.0f + R364_ACCEPT_REL_TOL) + R364_ACCEPT_ABS_TOL;
        if (probe->post_loss > allowed_loss) {
            if (r364_restore_rows(m, changed_vocab_rows, changed_count, shadow_backup, sampled_backup) != 0) {
                r364_set_reason(probe, "revert-csr-failed");
                goto fail;
            }
            probe->updated_rows = 0;
            probe->reverted_update = 1;
            r364_set_reason(probe, "loss-regressed");
            goto done;
        }
    }

    probe->applied_update = 1;
    probe->accepted_update = 1;
    probe->reverted_update = 0;

done:
    free(pos_counts);
    free(neg_counts);
    free(active_map);
    free(changed_vocab_rows);
    free(active_rows);
    free(negative_rows);
    free(solutions);
    free(row_targets);
    free(candidate_matrix);
    free(shadow_backup);
    free(sampled_backup);
    free(gram);
    free(chol);
    return 0;

fail:
    free(pos_counts);
    free(neg_counts);
    free(active_map);
    free(changed_vocab_rows);
    free(active_rows);
    free(negative_rows);
    free(solutions);
    free(row_targets);
    free(candidate_matrix);
    free(shadow_backup);
    free(sampled_backup);
    free(gram);
    free(chol);
    return -1;
}
