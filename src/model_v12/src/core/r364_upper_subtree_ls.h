#ifndef V12_R364_UPPER_SUBTREE_LS_H
#define V12_R364_UPPER_SUBTREE_LS_H

#include <stdint.h>

#include "types.h"

typedef struct {
    uint32_t compile_id;
    uint32_t boundary_step;
    uint32_t warmup_steps;
    uint32_t compile_every_steps;
    uint32_t compiled_steps;
    int enable_noop_hook;
    int eval_batch_index;
} R364UpperSubtreeLsCompileConfig;

typedef struct {
    float kappa;
    float compiled_block_share_f;
    double compile_ms;
    float pre_loss;
    float post_loss;
    float loss_delta;
    uint32_t candidate_rows;
    uint32_t updated_rows;
    int applied_update;
    int reverted_update;
    int noop_hook_armed;
    int accepted_update;
    char rejection_reason[64];
} R364UpperSubtreeLsProbe;

float r364_upper_subtree_ls_measure_last_block_share_f(const Model *m);
int r364_upper_subtree_ls_compile(Model *m,
                                  Cache *c,
                                  const R364UpperSubtreeLsCompileConfig *config,
                                  R364UpperSubtreeLsProbe *probe);

#endif /* V12_R364_UPPER_SUBTREE_LS_H */
