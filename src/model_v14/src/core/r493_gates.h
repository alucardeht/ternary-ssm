#ifndef R493_GATES_H
#define R493_GATES_H

#include <stdbool.h>
#include <stdint.h>

#define R493_MAX_TASKS 8

typedef struct {
    char task_name[64];
    float weight;
    float accuracy;            /* p_i(theta) */
    float false_positive_rate; /* f_i(theta) (twin false-control rate) */
} R493TaskResult;

typedef struct {
    char checkpoint_id[128];
    bool valid;                /* V(theta; S_k) */
    R493TaskResult tasks[R493_MAX_TASKS];
    int num_tasks;
    float entropy;
    float collapse_penalty;
    float staged_quality;      /* Q(theta; S_k) */
    bool passed;
} R493EvaluationReceipt;

typedef struct {
    uint32_t stage_id;
    float min_quality_threshold;
    float min_entropy_threshold;
    float collapse_weight;
    bool contract_stable;
} R493StageContract;

/* Computes staged quality Q(theta; S_k) */
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
);

/* Verifies stage contract progress and returns whether stage is complete */
bool r493_stage_contract_verify(
    const R493StageContract *contract,
    const R493EvaluationReceipt *receipt
);

#endif /* R493_GATES_H */
