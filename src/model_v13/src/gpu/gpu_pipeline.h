#ifndef V12_GPU_PIPELINE_H
#define V12_GPU_PIPELINE_H

#include "../core/types.h"

/* Forward declarations */
typedef struct VulkanContext VulkanContext;
typedef struct GpuPipeline GpuPipeline;

typedef struct {
    uint32_t compile_id;
    uint32_t train_step_calls;
    uint32_t optimizer_step_calls;
    int compiled_interval_active;
    float compiled_block_share_f;
} GpuR364NoopProbe;

/* Create GPU pipeline: allocates all GPU buffers, uploads CSR/CSC, initial weights.
 * Returns NULL on failure (caller falls back to CPU training).
 * Model must be fully initialized before calling. */
GpuPipeline *gpu_pipeline_create(VulkanContext *ctx, Model *m, int seq_len, int batch_size);

/* Destroy pipeline and free all GPU resources. Safe with NULL. */
void gpu_pipeline_destroy(GpuPipeline *p);

/* Execute one complete training step on GPU: forward + loss + backward.
 * Returns CE loss. Gradients accumulated into Model CPU arrays.
 *
 * tokens:        [batch_size * seq_len] interleaved sequences
 * boundary_mask: [batch_size * seq_len] interleaved masks
 *
 * After return:
 *   - All gradient accumulators in Model are updated (additive)
 *   - Cache activations for GP-KL/MTP are downloaded to Cache
 *   - d_h (layer 0 input gradient) available for PC backward
 */
float gpu_train_step(GpuPipeline *p, Model *m, Cache *c,
                     const uint16_t *tokens, const uint8_t *boundary_mask,
                     int seq_len, int seq_idx, int batch_size,
                     const TrainState *state);

/* Re-upload CPU-owned live weights after optimizer_step().
 * This updates only parameters whose source of truth remains on the CPU. */
void gpu_pipeline_sync_weights(GpuPipeline *p, Model *m);

/* Re-upload a CPU-materialized model state into all live GPU views.
 * Call after checkpoint_load()/checkpoint_rebuild() before resuming training. */
void gpu_pipeline_sync_materialized_state(GpuPipeline *p, Model *m);

/* Download only lm_head GPU-resident state needed by the R364 compile boundary. */
int gpu_pipeline_sync_lm_head_to_cpu(GpuPipeline *p, Model *m);

/* Download only slice-2 GPU-resident checkpoint state. */
int gpu_pipeline_sync_slice2_to_cpu(GpuPipeline *p, Model *m);

/* Download only embedding GPU-resident checkpoint state. */
int gpu_pipeline_sync_embed_to_cpu(GpuPipeline *p, Model *m);

/* Download layer GPU-resident checkpoint state. */
int gpu_pipeline_sync_layer_state_to_cpu(GpuPipeline *p, Model *m);

/* Download one layer of GPU-resident checkpoint state. */
int gpu_pipeline_sync_layer_to_cpu(GpuPipeline *p, Model *m, int layer_idx);

/* Download all GPU-resident gradient accumulators to CPU Model. */
int gpu_pipeline_sync_gradients_to_cpu(GpuPipeline *p, Model *m);

/* Materialize the narrow batch-1 CPU cache inputs consumed by the R364 compile path. */
int gpu_pipeline_sync_r364_compile_cache_to_cpu(GpuPipeline *p,
                                                Model *m,
                                                Cache *c,
                                                int batch_size);

/* Materialize the per-layer SSM hidden trajectory into Cache for opt-in trace
 * probes. Default GPU training keeps this cache GPU-resident. */
int gpu_pipeline_sync_hidden_cache_to_cpu(GpuPipeline *p,
                                          Cache *c,
                                          int batch_size);

/* Upload only lm_head host state and refresh the GPU CE/CSR view needed by the
 * sampled loss path after an R364 compile boundary update. */
void gpu_pipeline_sync_lm_head_from_cpu(GpuPipeline *p, Model *m);

/* Rebuild CSR/CSC from the current sampled tensors after a boundary. */
void gpu_pipeline_reupload_csr(GpuPipeline *p, Model *m, uint32_t rng_seed);

/* Explicitly resample sampled tensors from shadow state, then rebuild CSR/CSC. */
void gpu_pipeline_resample_and_reupload_csr(GpuPipeline *p, Model *m, uint32_t rng_seed);

/* Re-upload single layer CSR/CSC after INT2 pack (FP stage transition). */
void gpu_pipeline_reupload_layer(GpuPipeline *p, Model *m, int layer_idx);

/* Print GPU memory stats. */
void gpu_pipeline_print_stats(GpuPipeline *p);

/* Zero all GPU gradient accumulator buffers (gpu_dw_in/out, gpu_dW_A, ce_dlm_head,
 * opt_d_embed_P, bwd_d_embed_E). Call ONCE before the gradient accumulation loop. */
void gpu_zero_grad_accumulators(GpuPipeline *p);

/* Scale all GPU gradient accumulator buffers by `scale`.
 * Call after the accumulation loop, before the GPU optimizer steps. */
void gpu_scale_gradients(GpuPipeline *p, float scale);

/* Opt-in R415 audit over GPU-resident gradient tensors. This only records
 * certificate telemetry when Model.gradient_null_witness_gate_enabled is set. */
void gpu_gradient_null_witness_gate_update(GpuPipeline *p, Model *m);

/* Opt-in R375 PB-G2 diagnostic over GPU-resident ternary gradient matrices.
 * Records Kronecker-shape telemetry only; default training is unchanged. */
void gpu_r375_pbg2_gate_update(GpuPipeline *p, Model *m, float lr);

/* Opt-in R375 PB-G2 row/column RMS preconditioner prototype for GPU-owned
 * W_in/W_out gradients. This changes training only when the runtime flag is
 * enabled. */
void gpu_r375_pbg2_precondition_update(GpuPipeline *p, Model *m);

/* Opt-in R416 force-bound audit over deterministic ternary chamber cells.
 * This records no-crossing telemetry only; the default objective is unchanged. */
void gpu_ce_normal_chamber_gate_update(GpuPipeline *p, Model *m,
                                       const TrainState *state,
                                       float lr, int step);

/* Opt-in R420 live-path audit for the online TSSI compiler. It inspects only
 * GPU buffer residency and records whether target-conditioned SSM score
 * buffers exist, without hidden-cache CPU materialization. */
void gpu_online_tssi_live_gate_update(GpuPipeline *p, Model *m);
void gpu_r423_native_growth_receipt_challenge_gate_update(GpuPipeline *p,
                                                          int step);
void gpu_r423_native_growth_interaction_release_safety_gate_update(GpuPipeline *p,
                                                                   int step);
void gpu_r423_native_growth_interaction_safe_update(GpuPipeline *p,
                                                    int step);
void gpu_r423_native_growth_genotype_exception_gate_update(GpuPipeline *p,
                                                           int step);
void gpu_r423_native_growth_genotype_compact_update(GpuPipeline *p,
                                                     int step);
int gpu_pipeline_write_online_tssi_selector(GpuPipeline *p,
                                            const char *path,
                                            float active_fraction,
                                            const char *round_label);
int gpu_pipeline_set_online_tssi_train_selector(GpuPipeline *p,
                                                const uint8_t *state_mask,
                                                uint32_t mask_len,
                                                const char *control_name,
                                                uint32_t active_dims);

/* Run Adam optimizer on GPU for norm_w and norm_b.
 * Call after scale_all_gradients, before gpu_pipeline_sync_weights.
 * step is 0-indexed (used for bias correction). */
void gpu_optimizer_step_dense(GpuPipeline *p, Model *m, float lr, int step);

/* Stub: factored Adam optimizer for ternary W_in and W_out.
 * GPU buffers (shadow logits, factored moments) are allocated and uploaded.
 * Shader dispatches are not wired yet — pending push-constant finalisation. */
void gpu_optimizer_step_ternary(GpuPipeline *p, Model *m, const TrainState *state,
                                float lr, int step);

/* Offload ternary W_in and W_out moments of layer k from GPU to Host CPU RAM */
void gpu_pipeline_offload_moments(GpuPipeline *p, Model *m, int k);

/* Download GPU-resident optimizer weights back to CPU Model struct.
 * Must be called before checkpoint_save() to ensure the checkpoint reflects
 * the latest GPU-updated shadow logits, sampled weights, and dense params. */
int gpu_sync_to_cpu(GpuPipeline *p, Model *m);

/* R364 slice A: arm a no-op compiled interval without changing GPU semantics.
 * Slice C will replace this control path with real work removal. */
void gpu_pipeline_r364_set_noop_compiled_interval(GpuPipeline *p,
                                                  int active,
                                                  uint32_t compile_id,
                                                  float compiled_block_share_f);
void gpu_pipeline_r364_get_noop_probe(const GpuPipeline *p, GpuR364NoopProbe *out);

#endif /* V12_GPU_PIPELINE_H */
