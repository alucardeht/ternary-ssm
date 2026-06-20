#ifndef V12_PCA_CACHE_H
#define V12_PCA_CACHE_H

#include "types.h"
#include "../io/dataset.h"

ResidualCache *pca_cache_alloc(void);

void pca_cache_free(ResidualCache *rc);

int pca_cache_build_entry(Model *m, ResidualCache *rc, Dataset *ds,
                          int layer_idx, int seq_len, int n_seqs);

void pca_cache_load_entry(const ResidualCacheEntry *entry,
                          int seq_idx, int t,
                          float *out_buf, int d_model, int rank);

#endif
