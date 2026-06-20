/*
 * model.h — Model and cache allocation / deallocation functions
 *
 * This module owns memory lifecycle for the Model and Cache structs.
 * Both are allocated once and reused throughout training / inference.
 *
 * Dependencies: types.h, ternary_matmul.h (for TernaryCSR functions)
 */

#ifndef V12_MODEL_H
#define V12_MODEL_H

#include "types.h"

/*
 * Allocate and zero-initialize a complete Model for training mode.
 *
 * Returns a fully-allocated Model with all pointer fields initialized
 * (either heap-allocated or NULL if not used in training). All dimension
 * fields are set to match compile-time constants.
 *
 * Returns NULL on any allocation failure (caller must check).
 */
Model *model_alloc(void);

/*
 * Free all heap allocations owned by the Model.
 *
 * Safe to call with NULL — immediately returns.
 * Frees in reverse allocation order (bottom-up through layers, then top-level).
 */
void model_free(Model *m);

/*
 * Allocate forward and backward cache buffers for one training step.
 *
 * Parameters:
 *   batch — batch size (reserved for future use; v12 uses batch=1)
 *   seq   — sequence length (must equal V12_SEQ_LEN = 512)
 *
 * Returns a fully-allocated Cache with all activation and gradient buffers.
 * Returns NULL on allocation failure.
 *
 * Total allocation at v12 defaults (batch=1, seq=512):
 *   ~35 MB (logits + d_logits reduced to 0.2 MB via sampled softmax, was 131.4 MB)
 */
Cache *cache_alloc(int batch, int seq);

/*
 * Free all heap allocations owned by the Cache.
 *
 * Safe to call with NULL — immediately returns.
 */
void cache_free(Cache *c);

/*
 * Rebuild CSR indices for one SSMLayer from its current sampled_w_in/out.
 * Call after every MVUE repack (every V12_REPACK_INTERVAL steps).
 * Returns 0 on success, -1 on allocation failure.
 */
int model_rebuild_layer_csr(SSMLayer *L);

/*
 * Rebuild CSR for lm_head from current sampled_lm_head.
 * Call together with model_rebuild_layer_csr after each repack.
 * Returns 0 on success, -1 on allocation failure.
 */
int model_rebuild_lm_head_csr(Model *m);

/*
 * Deterministically initialize the host lm_head mirrors used to seed a fresh
 * Vulkan pipeline before steady-state ownership returns to GPU-only.
 */
int model_initialize_lm_head_host(Model *m);

/*
 * Materialize or release the host lm_head mirrors used only at explicit
 * checkpoint/debug boundaries when Vulkan owns steady-state training state.
 */
int model_materialize_lm_head_host(Model *m);
void model_release_lm_head_host(Model *m);
int model_materialize_embed_host(Model *m);
void model_release_embed_host(Model *m);
int model_materialize_w_pred_host(Model *m);
void model_release_w_pred_host(Model *m);
int model_materialize_norm_host(Model *m);
void model_release_norm_host(Model *m);
/*
 * Materialize the persisted W_in/W_out host subset at an explicit boundary,
 * or fully release the host mirrors again once steady-state GPU ownership is
 * restored. grad_accum/dw remain outside this contract.
 */
int model_materialize_layer_w_io_host(SSMLayer *layer);
void model_release_layer_w_io_host(SSMLayer *layer);
int model_embed_host_is_materialized(const Model *m);
int model_w_pred_host_is_materialized(const Model *m);
int model_norm_host_is_materialized(const Model *m);
int model_layer_w_io_host_is_materialized(const SSMLayer *layer);

/*
 * sampled_softmax_build_unigram — Build unigram proposal distribution for NCE sampling.
 *
 * Scans all tokens in the dataset to compute unigram frequency distribution.
 * Applies 3/4 power smoothing (standard NCE; rare tokens sampled more often).
 * Allocates and fills m->unigram_cdf and m->log_unigram.
 *
 * Call once after dataset load, before first training step.
 * Parameters:
 *   m       — model (writes unigram_cdf and log_unigram)
 *   tokens  — flat token array from dataset
 *   n_toks  — total number of tokens
 */
void sampled_softmax_build_unigram(Model *m, const uint16_t *tokens, size_t n_toks);

/*
 * Corpus logit prior for sampled CE.
 *
 * Configure before calling corpus_logit_prior_build(). The build scans corpus
 * bigrams, stores sparse observed pair counts, and exposes an additive log prior
 * through corpus_logit_prior_logit().
 */
void corpus_logit_prior_configure(
    Model *m, int enabled, float alpha, float scale,
    uint64_t requested_terms, uint32_t stride);
int corpus_logit_prior_build(Model *m, const uint16_t *tokens, size_t n_toks);
float corpus_logit_prior_logit(const Model *m, uint16_t context, uint16_t target);

/*
 * Proof-carrying program kernel for sampled CE.
 *
 * Builds a deterministic context->target transition table from corpus-owned
 * evidence. Contexts with multiple observed targets or insufficient support
 * abstain, preserving the residual neural fallback.
 */
void program_kernel_configure(
    Model *m, int enabled, int gate_enabled, float logit_boost,
    uint32_t min_count, uint64_t requested_terms, uint32_t stride);
int program_kernel_build(Model *m, const uint16_t *tokens, size_t n_toks);
uint32_t program_kernel_predict(const Model *m, uint16_t context);
float program_kernel_logit(const Model *m, uint16_t context, uint16_t target);

/*
 * Certified abstaining artifact boundary gate.
 *
 * Builds a compact proof-mask for the context-derived copy-current
 * transducer. Certified contexts emit their own current token; every
 * conflicting or under-supported context abstains and remains an original-CE
 * residual obligation.
 */
void certified_artifact_boundary_configure(
    Model *m, int gate_enabled, uint32_t min_count,
    uint64_t requested_terms, uint32_t stride,
    float epsilon, float ce_budget, float overhead, float kappa);
int certified_artifact_boundary_build(Model *m, const uint16_t *tokens, size_t n_toks);
uint32_t certified_artifact_boundary_predict(const Model *m, uint16_t context);

/*
 * Exact state transition atlas for sampled CE.
 *
 * Stores exact/certified transitions keyed by a bounded local state signature:
 * (previous token, current token) -> target. Conflicts abstain.
 */
void state_transition_atlas_configure(
    Model *m, int enabled, int gate_enabled, float logit_boost,
    uint32_t min_count, uint64_t requested_terms, uint32_t stride);
int state_transition_atlas_build(Model *m, const uint16_t *tokens, size_t n_toks);
uint32_t state_transition_atlas_predict(const Model *m, uint16_t prev, uint16_t context);
float state_transition_atlas_logit(
    const Model *m, uint16_t prev, uint16_t context, uint16_t target);

/*
 * Counterfactual minimal-pair packet ledger.
 *
 * Stores first-two distinct targets for each local state
 * (previous token, current token). The gate exposes paired packet coverage on
 * the real training stream while preserving original CE as the objective.
 */
void counterfactual_packets_configure(
    Model *m, int enabled, int gate_enabled,
    uint32_t min_count, uint64_t requested_terms, uint32_t stride);
int counterfactual_packets_build(Model *m, const uint16_t *tokens, size_t n_toks);
uint32_t counterfactual_packets_alternative(
    const Model *m, uint16_t prev, uint16_t context, uint16_t target, uint32_t *side);

/*
 * Error-causal prerequisite lattice.
 *
 * Builds compact target/prerequisite atom support counts from the corpus. The
 * gate measures coverage-conserving orderability on the real training stream
 * without changing sampling or CE.
 */
void error_causal_lattice_configure(
    Model *m, int enabled, int gate_enabled,
    uint32_t min_count, uint64_t requested_terms, uint32_t stride);
int error_causal_lattice_build(Model *m, const uint16_t *tokens, size_t n_toks);

/*
 * Exact local transition quotient.
 *
 * Counts exact (previous token, current token, target token) local transition
 * triples. The gate prices objective-identical multiplicity collapse without
 * semantic buckets or CE approximation.
 */
void exact_transition_quotient_configure(
    Model *m, int enabled, int gate_enabled,
    uint32_t min_count, uint64_t requested_terms, uint32_t stride);
int exact_transition_quotient_build(Model *m, const uint16_t *tokens, size_t n_toks);
uint32_t exact_transition_quotient_count(
    const Model *m, uint16_t prev, uint16_t context, uint16_t target);

/*
 * R421 Macro CE quotient.
 *
 * Builds predictive quotient classes over real corpus order-2 contexts and
 * either observes the live training stream or applies quotient-unit training
 * weights to sampled CE gradients.
 */
void macro_ce_quotient_configure(
    Model *m, int observer_enabled, int training_enabled, uint32_t min_support,
    uint64_t requested_terms, uint32_t stride,
    float build_fraction, float alpha,
    float residual_price_scale, float overhead,
    uint32_t signature_bins, uint32_t seed,
    float training_weight);
int macro_ce_quotient_build(Model *m, const uint16_t *tokens, size_t n_toks);
void macro_ce_quotient_observe(
    Model *m, const uint16_t *tokens, int batch_size, int seq_len);
void macro_ce_quotient_apply_training_weights(
    Model *m, const uint16_t *tokens, float *d_logits,
    int batch_size, int seq_len, int n_samples);

void exact_window_multiset_configure(
    Model *m, int gate_enabled, uint32_t min_count,
    uint64_t requested_windows, uint32_t stride,
    float overhead, float noise_penalty);
int exact_window_multiset_build(
    Model *m, const uint16_t *tokens, size_t n_toks, int seq_len);
uint32_t exact_window_multiset_count_live(
    const Model *m, const uint16_t *window, int seq_len, int position);

/*
 * model_build_hot_mask — Classify tokens as hot (top-K) or cold by frequency.
 *
 * Scans token frequencies to mark the top K_HOT=2000 most frequent tokens as "hot".
 * Builds hot_mask[vocab], vocab_to_hot[vocab], and hot_to_vocab[K_HOT] lookup tables.
 * Cold tokens remain at -1 in vocab_to_hot and receive no gradient updates.
 *
 * Call after sampled_softmax_build_unigram during training initialization.
 * Parameters:
 *   m       — model (writes hot_mask, vocab_to_hot, hot_to_vocab)
 *   tokens  — flat token array from dataset
 *   n_toks  — total number of tokens
 */
void prefix_frontier_configure(
    Model *m, int gate_enabled, uint32_t min_count,
    uint64_t requested_terms, uint32_t stride,
    float overhead, float repair_cost);
int prefix_frontier_build(Model *m, const uint16_t *tokens, size_t n_toks);
uint32_t prefix_frontier_count(const Model *m, uint16_t prev, uint16_t context);

/*
 * Reversible macro-transcript gate.
 *
 * Builds a corpus-owned adjacent-pair dictionary and measures exact greedy
 * macro-token transcript compression. Macro tokens are reversible because each
 * dictionary entry stores its original two-token expansion.
 */
void macro_transcript_configure(
    Model *m, int gate_enabled, uint32_t min_count,
    uint64_t requested_terms, uint32_t stride,
    float hardness, float codec_cost, float decode_cost);
int macro_transcript_build(Model *m, const uint16_t *tokens, size_t n_toks);
uint32_t macro_transcript_pair_count(const Model *m, uint16_t left, uint16_t right);

void reversible_deterministic_expansion_configure(
    Model *m, int gate_enabled, uint32_t min_run,
    uint64_t requested_terms, float overhead, float noise_penalty);
int reversible_deterministic_expansion_build(
    Model *m, const uint16_t *tokens, size_t n_toks);
int reversible_deterministic_expansion_token(uint16_t token);

void model_build_hot_mask(Model *m, const uint16_t *tokens, size_t n_toks);

/*
 * pack_layer_to_int2 — Quantize and pack shadow weights of a frozen layer.
 *
 * Applies round_ternary to shadow_w_in, shadow_w_out, shadow_w_conv
 * and packs results into w_in_int2, w_out_int2, w_conv_int2 using
 * the project INT2 encoding (4 weights/byte, 0b00=0, 0b01=+1, 0b11=-1).
 *
 * Call after FP stage k completes, before advancing to stage k+1.
 * The w_*_int2 buffers are already allocated in model_alloc.
 */
void pack_layer_to_int2(SSMLayer *L);

/*
 * unpack_layer_from_int2 — Reconstruct fp32 shadow weights from INT2.
 *
 * Inverse of pack_layer_to_int2. Writes exact ternary {-1, 0, +1} values
 * into shadow_w_in, shadow_w_out, shadow_w_conv.
 *
 * Call before reactivating a frozen layer for BCD or joint training.
 * Adam moments are preserved in memory and remain valid.
 */
void unpack_layer_from_int2(SSMLayer *L);

/*
 * layer_offload_fp32 — Free fp32 training scaffolding for a frozen layer.
 *
 * Called after pack_layer_to_int2 when layer k completes its FP stage.
 * Frees all fp32 shadow weights, gradient accumulators, and Adam moments.
 * Sets all freed pointers to NULL as a sentinel.
 *
 * Layer must be packed to INT2 first. INT2 packed weights (w_in_int2, etc.),
 * CSR indices, and SSM dynamics (A, B, C, mtp_U, mtp_V) remain allocated.
 *
 * Savings at 7B scale: ~25 MB per layer (775 MB for 31 frozen layers).
 * Only meaningful for 7B+; at MEDIUM scale (d=128) benefit is negligible (~690 KB).
 */
void layer_offload_fp32(SSMLayer *L);

/*
 * layer_restore_fp32 — Restore fp32 training scaffolding for an offloaded layer.
 *
 * Re-allocates shadow weights, gradient accumulators, and Adam moments.
 * All realloc'd arrays are zero-initialized (acceptable for BCD short cycles).
 *
 * Call before reactivating an offloaded layer for BCD. Must only be called
 * on a layer that has been offloaded (dw_in == NULL).
 *
 * Sequence: pack_layer_to_int2 → layer_offload_fp32 → ... → layer_restore_fp32
 * (restore must happen before unpack_layer_from_int2 writes to shadow pointers).
 */
void layer_restore_fp32(SSMLayer *L);

#endif /* V12_MODEL_H */
