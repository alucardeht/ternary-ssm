#ifndef V12_Y_CACHE_H
#define V12_Y_CACHE_H

#include "types.h"
#include "../io/dataset.h"

YCache *y_cache_alloc(void);

void y_cache_free(YCache *yc);

int y_cache_build_entry(Model *m, YCache *yc, Dataset *ds,
                        int layer_idx, int seq_len, int n_seqs);

void y_cache_load_entry(const YCache *yc, const Model *m,
                        int seq_idx, int t,
                        float *out_buf, int d_model);

#endif
