#include "r493_gates.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float compute_stable_entropy(const uint32_t *tokens, int num_tokens) {
    if (num_tokens <= 0 || tokens == NULL) {
        return 0.0f;
    }

    /* Find max token ID to bound histogram size */
    uint32_t max_tok = 0;
    for (int i = 0; i < num_tokens; i++) {
        if (tokens[i] > max_tok) {
            max_tok = tokens[i];
        }
    }

    int hist_size = max_tok + 1;
    int *counts = (int *)calloc(hist_size, sizeof(int));
    if (counts == NULL) {
        return 0.0f;
    }

    for (int i = 0; i < num_tokens; i++) {
        counts[tokens[i]]++;
    }

    float entropy = 0.0f;
    float total = (float)num_tokens;
    for (int i = 0; i < hist_size; i++) {
        if (counts[i] > 0) {
            float p = (float)counts[i] / total;
            entropy -= p * (logf(p) / logf(2.0f));
        }
    }

    free(counts);
    return entropy;
}

R493EvaluationReceipt r493_eval_quality_harness(
    const char *checkpoint_id,
    bool base_valid,
    const float *accuracies,
    const float *false_positives,
    const float *weights,
    int num_tasks,
    const uint32_t *tokens,
    int num_tokens,
    const R493StageContract *contract
) {
    R493EvaluationReceipt receipt;
    memset(&receipt, 0, sizeof(receipt));

    strncpy(receipt.checkpoint_id, checkpoint_id ? checkpoint_id : "unknown", sizeof(receipt.checkpoint_id) - 1);
    receipt.valid = base_valid;
    
    if (num_tasks > R493_MAX_TASKS) {
        num_tasks = R493_MAX_TASKS;
    }
    receipt.num_tasks = num_tasks;

    const char *task_names[R493_MAX_TASKS] = {
        "linguistic_qa",
        "procedural_logic",
        "syntactic_code",
        "relation_transfer",
        "task_4",
        "task_5",
        "task_6",
        "task_7"
    };

    /* Dynamic task weight scaling by quality metrics (R493 Aspect A05) */
    float dynamic_weights[R493_MAX_TASKS];
    float sum_dynamic_weights = 0.0f;
    for (int i = 0; i < num_tasks; i++) {
        dynamic_weights[i] = weights[i];
        if (accuracies[i] < contract->min_quality_threshold) {
            /* Struggling tasks get their weights dynamically boosted */
            dynamic_weights[i] *= (1.5f + (contract->min_quality_threshold - accuracies[i]));
        }
        sum_dynamic_weights += dynamic_weights[i];
    }

    float sum_weighted_acc = 0.0f;
    for (int i = 0; i < num_tasks; i++) {
        strncpy(receipt.tasks[i].task_name, task_names[i], sizeof(receipt.tasks[i].task_name) - 1);
        
        float final_weight = (sum_dynamic_weights > 0.0f) ? (dynamic_weights[i] / sum_dynamic_weights) : weights[i];
        receipt.tasks[i].weight = final_weight;
        receipt.tasks[i].accuracy = accuracies[i];
        receipt.tasks[i].false_positive_rate = false_positives[i];

        /* Accumulate Q accuracy component: w_i * p_i * (1 - f_i) */
        sum_weighted_acc += final_weight * accuracies[i] * (1.0f - false_positives[i]);
    }

    /* Compute Shannon entropy over generated token sequence */
    receipt.entropy = compute_stable_entropy(tokens, num_tokens);

    /* Apply collapse penalty if entropy falls below threshold */
    receipt.collapse_penalty = 0.0f;
    if (receipt.entropy < contract->min_entropy_threshold) {
        receipt.collapse_penalty = contract->collapse_weight * (contract->min_entropy_threshold - receipt.entropy);
    }

    /* Final Staged Quality staged_quality = V(theta) * (sum_weighted_acc - collapse_penalty) */
    receipt.staged_quality = sum_weighted_acc - receipt.collapse_penalty;
    if (!receipt.valid) {
        receipt.staged_quality = 0.0f;
    }

    /* Determine if quality targets are met */
    receipt.passed = (
        receipt.valid &&
        receipt.staged_quality >= contract->min_quality_threshold &&
        receipt.collapse_penalty == 0.0f
    );

    return receipt;
}

bool r493_stage_contract_verify(
    const R493StageContract *contract,
    const R493EvaluationReceipt *receipt
) {
    if (contract == NULL || receipt == NULL) {
        return false;
    }
    return receipt->passed && contract->contract_stable;
}
