#ifndef GPU_PIPELINE_INTERNAL_H
#define GPU_PIPELINE_INTERNAL_H

#include "gpu_pipeline.h"
#include <stdio.h>
#include <string.h>
#include "../core/ternary_matmul.h"

#ifdef V12_HAS_VULKAN



#include <vulkan/vulkan.h>
#include "gpu_pipeline.h"
#include "../runtime/vulkan_backend.h"
#include "../runtime/shaders/f12_fourier_ssm_spv.h"
#include "../runtime/shaders/e13_parametric_cotangent_spv.h"
#include "../runtime/shaders/e13_lrms_spv.h"
#include "../runtime/shaders/ef03_ring_checkpoint_spv.h"
#include "../runtime/shaders/sp12_lcmbi_spv.h"
#include "../runtime/shaders/sp11_lns_tfms_spv.h"
#include "../runtime/shaders/sp10_psvcte_spv.h"
#include "../runtime/shaders/f01_rrd_involution_spv.h"
#include "../runtime/shaders/e32_aoi_ssm_spv.h"
#include "../runtime/shaders/sp13_sde_bvdj_spv.h"
#include "../runtime/shaders/sp14_gfit_spv.h"
#include "../runtime/shaders/sp15_tt_rcrb_spv.h"
#include "../runtime/shaders/sp16_null_space_prune_spv.h"
#include "../runtime/shaders/f14_ternary_activation_escrow_spv.h"
#include "../runtime/shaders/sp17_dtss_spv.h"
#include "../runtime/shaders/entangled_adam_update_spv.h"
#include "../runtime/shaders/sp19_holo_adam_reduce_spv.h"
#include "../runtime/shaders/f37_taqg_spv.h"
#include "../runtime/shaders/f38_cst_spv.h"
#include "../runtime/shaders/f39_bvgr_spv.h"
#include "../runtime/shaders/sp03_almm_pack_spv.h"
#include "../runtime/shaders/sp03_almm_scan_spv.h"
#include "../runtime/shaders/sp03_almm_unpack_spv.h"
#include "../runtime/shaders/f15_cross_layer_adjoint_spv.h"

#include "../runtime/shaders/f34_srat_forward_spv.h"
#include "../runtime/shaders/f34_srat_reconstruct_spv.h"
#include "../runtime/shaders/f35_idre_spv.h"
#include "../runtime/shaders/f36_slefc_forward_spv.h"
#include "../runtime/shaders/f36_slefc_reconstruct_spv.h"

#include <stdlib.h>
#include <stdio.h>





#include "../runtime/shaders/mprm_adam_spv.h"
extern VkPipeline mprm_adam_pipeline;



extern VkPipeline f35_idre_pipeline;



extern VkPipeline f36_slefc_forward_pipeline;
extern VkPipeline f36_slefc_reconstruct_pipeline;





extern VkPipeline f34_srat_forward_pipeline;
extern VkPipeline f34_srat_reconstruct_pipeline;





extern VkPipeline f01_rrd_involution_pipeline;
extern VkPipeline e32_aoi_ssm_pipeline;






























































































































































































extern VkPipeline ef03_pipeline;


extern VkPipeline entangled_adam_pipeline;




extern VkPipeline sp19_holo_adam_pipeline;



extern VkPipeline f37_taqg_pipeline;

extern VkPipeline f38_cst_pipeline;




extern VkPipeline e13_lrms_pipeline;



extern VkPipeline f39_bvgr_pipeline;









extern VkPipeline sp10_psvcte_pipeline;




extern VkPipeline sp11_lns_tfms_pipeline;



extern VkPipeline sp12_lcmbi_pipeline;



extern VkPipeline sp13_sde_bvdj_pipeline;



extern VkPipeline sp14_gfit_pipeline;


extern VkPipeline sp15_tt_rcrb_pipeline;




extern VkPipeline f12_fourier_pipeline;
static VkBufferHandle f12_aux_buffer = {0};



extern VkPipeline e13_parametric_cotangent_pipeline;
static VkBufferHandle e13_w_pred_buffer = {0};



extern VkPipeline sp16_nsp_pipeline;


extern VkPipeline f14_tae_pipeline;


extern VkPipeline sp17_dtss_pipeline;




extern VkPipeline f15_cross_layer_adjoint_pipeline;





/* R503 Phase 1: Memory Geometry & Allocation Modifications */
#define R504_E14_SOSE 1
#define R504_E15_HGC 1
#define R504_Q01_IGTF 1
#define R504_Q02_TCTM 1
#define R504_Q03_FLAD 1
#define R504_Q04_LGCS 1
#define R504_Q05_SHPS 1
#define R504_Q06_DSTS 1
#define R504_Q07_GAME 1
#define R504_Q08_CHST 1
#define R504_Q09_QISS 1
#define R504_Q10_TTGTA 1
#define R504_Q11_RFRSM 1
#define R504_Q12_TGF 1
#define R504_Q13_GIGN 1
#define R504_Q14_FKEP 1
#define R504_Q15_HSMT 1
#define R504_Q16_ACGM 1
#define R504_Q17_SGI 1
#define R504_Q18_AGD 1
#define R504_Q19_QPIG 1
#define R504_Q20_SGAP 1
#define R504_Q21_HGP 1
#define R504_Q22_ITEB 1
#define R504_Q23_MAM 1
#define R504_Q24_GPA 1
#define R504_Q25_LCN 1
#define R504_Q26_ECUR 1
#define R504_Q27_CSVT 1
#define R504_Q28_ABCD 1
#define R504_Q29_HFSD 1
#define R504_Q30_TGS 1
#define R504_Q31_DSGP 1
#define R504_Q32_LVP 1
#define R504_Q33_SFN 1
#define R504_Q34_KER 1
#define R504_Q35_ESS 1
#define R504_Q36_DGQ 1
#define R504_Q37_HMR 1
#define R504_Q38_SOP 1
#define R504_Q39_AWP 1
#define R504_Q40_TGE 1
#define R504_Q41_LLCN 1
#define R504_Q42_SGD 1
#define R504_Q43_OFP 1
#define R504_Q44_LES 1
#define R503_PHASE_1_EFFICIENCY 1
#define R504_S11_LNS_TFMS 1
#define S01_SUBGROUP_PACKED_TERNARY 1
#define R503_E07_PHASE_AREA_ALIASING 1
#define R503_E10_SPARSE_EMBEDDING_ALLOCATION 1
#define R503_E13_FATI_CHUNKED 1
#define R504_E10_PSCQ 1
#define R504_E11_BTRP 1
#define BTRP_R 16

#define R504_S13_SDE_BVDJ 1
#define R504_S10_PSVCTE 1
#define R504_S14_GFIT 1
#define R504_S15_TT_RCRB 1

#include "../core/model.h"
#include "../core/ternary_matmul.h"
#include "../core/ssm_scan.h"
#include "../core/loss.h"
#include "../core/optimizer.h"
#include "../core/shadow_codec.h"
#include "../io/dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#if V12_F2_INT8_SHADOWS

#endif



/* GPU moments buffer layout:
 *   [0   .. ds-1  ]  mean[ds]
 *   [ds  .. 2ds-1 ]  M2[ds]
 *   [2ds .. 3ds-1 ]  lag1_sum[ds]
 *   [3ds .. 4ds-1 ]  lag1_prev[ds]
 *   [4ds          ]  n  (int32 bit-cast to float)
 *   [4ds+1        ]  stability_sum
 *   [4ds+2..4ds+33]  orth_sum[32]
 *   [4ds+34..4ds+65] orth_pair_a[32]  (float-encoded uint indices)
 *   [4ds+66..4ds+97] orth_pair_b[32]  (float-encoded uint indices)
 * Total: 4*ds + 98 floats per layer. */
#define GPU_MOMENTS_FLOATS(ds) ((size_t)(4*(ds) + 98))
#define GPU_MOMENTS_BYTES(ds)  (GPU_MOMENTS_FLOATS(ds) * sizeof(float))
#define TSSI_SCORE_MOMENTS_FLOATS(ds) ((size_t)(2*(ds) + 4))
#define TSSI_SCORE_MOMENTS_BYTES(ds)  (TSSI_SCORE_MOMENTS_FLOATS(ds) * sizeof(float))
#define GPU_FLIP_COUNT_SLOTS   ((size_t)(V12_N_LAYERS * 2 + 2))


#define R364_CRASHMAP_LOG(...) do { \
    if (r364_crashmap_logging_enabled()) fprintf(stderr, __VA_ARGS__); \
} while (0)

enum {
    ERC_STAT_ELEMENTS = 0,
    ERC_STAT_NONZERO = 1,
    ERC_STAT_CRYSTAL = 2,
    ERC_STAT_ACTIVE = 3,
    ERC_STAT_FAILED = 4,
    ERC_STAT_SIGN_STABLE = 5,
    ERC_STAT_SIGN_CONFLICT = 6,
    ERC_STAT_BOUNDARY = 7,
    ERC_STAT_CURRENT_COMMITS = 8,
    ERC_STAT_TSA_COMMITS = 9,
    ERC_STAT_GBAR_COMMITS = 10,
    ERC_STAT_APPLIED_COMMITS = 11,
    ERC_STAT_CURRENT_DESCENT = 16,
    ERC_STAT_TSA_DESCENT = 17,
    ERC_STAT_GBAR_DESCENT = 18,
    ERC_STAT_APPLIED_DESCENT = 19,
    ERC_STAT_APPLIED_HARM = 20,
    ERC_STAT_MODE = 30,
    ERC_STAT_GATE = 31
};






static int g_r400_obs_trace_checked = 0;
static int g_r400_obs_trace_enabled = 0;

#define R400_OBS_TRACE_ENV "V12_R400_OBS_TRACE"
#define R400_OBS_TRACE_PATH_ENV "V12_R400_OBS_TRACE_PATH"
#define R400_OBS_TRACE_MAX_STEPS_ENV "V12_R400_OBS_TRACE_MAX_STEPS"
#define R400_OBS_TRACE_MAX_RECORDS_ENV "V12_R400_OBS_TRACE_MAX_RECORDS"
#define R400_OBS_TRACE_LAYER_ENV "V12_R400_OBS_TRACE_LAYER"
#define R400_OBS_TRACE_SAMPLE_ENV "V12_R400_OBS_TRACE_SAMPLE"
#define R400_OBS_TRACE_HORIZON_ENV "V12_R400_OBS_TRACE_HORIZON"
#define R400_OBS_TRACE_DEFAULT_PATH ".work/jobs/r400-observability-01/artifacts/r400_observability_trace.jsonl"
#define R400_OBS_TRACE_SCHEMA "r400.common_observability.v1"
#define R400_OBS_TRACE_GROUP_CAP 64
#define R400_OBS_TRACE_EPS 1.0e-12

typedef struct {
    char alignment_key[160];
    unsigned flags;
} R400ObsTraceGroup;

typedef struct {
    int enabled;
    int initialized;
    int meta_written;
    int summary_written;
    int activation_line_emitted;
    int config_error_emitted;
    int write_error_emitted;
    int max_steps;
    uint64_t max_records;
    int layer_filter;
    int sample_count;
    int horizon;
    int batch_size;
    int n_layers;
    uint64_t records_written;
    int group_count;
    long summary_offset;
    char path[1024];
    char run_id[64];
    R400ObsTraceGroup groups[R400_OBS_TRACE_GROUP_CAP];
} R400ObsTraceConfig;

static R400ObsTraceConfig g_r400_obs_trace = {0};






#define R400_OBS_FLAG_STATE_LIVE        0x0001u
#define R400_OBS_FLAG_TRANSITION_LIVE   0x0002u
#define R400_OBS_FLAG_OBJECTIVE_LIVE    0x0004u
#define R400_OBS_FLAG_OBJECTIVE_ABSENT  0x0008u
#define R400_OBS_FLAG_BOUNDARY_PRE      0x0010u
#define R400_OBS_FLAG_BOUNDARY_POST     0x0020u
#define R400_OBS_FLAG_BOUNDARY_SAMPLE   0x0040u
#define R400_OBS_FLAG_BOUNDARY_Q        0x0080u
#define R400_OBS_FLAG_BOUNDARY_ABSENT   0x0100u































#include "../runtime/shaders/btrp_token_gather_spv.h"

extern VkPipeline btrp_token_gather_pipeline;























typedef struct {
    uint64_t hash;
    uint32_t count;
    uint32_t rep_pos;
    double exact_energy;
    double residual_energy;
    int used;
} R438PacketCell;














struct GpuPipeline {
    VulkanContext *ctx;
    int seq_len;
    int batch_size;
    int d_model;
    int d_state;
    int n_layers;

    /* Embedding (GPU-resident, uploaded once, re-uploaded after optimizer step) */
    VkBufferHandle embed_E_gpu;     /* [vocab × d_embed]       */
    VkBufferHandle embed_P_gpu;     /* [d_embed × d_model]     */
    VkBufferHandle embed_gathered;  /* [seq_len × d_embed]     scratch */
    VkBufferHandle embed_tokens;    /* [seq_len] uint32        scratch */

    /* Shared scratch (reused across layers) */
    VkBufferHandle buf_x;    /* [seq_len × d_model] */
    VkBufferHandle buf_y;    /* [seq_len × d_model] */

    /* Residual stream (GPU-resident across layers) */
    VkBufferHandle residual[V12_N_LAYERS + 1];

    /* Predictive coding */
    VkBufferHandle pred_A;          /* sampled_w_pred [d×d] */
    VkBufferHandle shadow_pred_A;   /* shadow_w_pred [d×d] */
    VkBufferHandle pred_x;          /* [T × d] batch input */
    VkBufferHandle pred_y;          /* [T × d] batch output */

    /* GEMM scratch */
    VkBufferHandle gemm_A;          /* [seq×max(d,ds)] */
    VkBufferHandle gemm_B;          /* [seq×max(d,ds)] */
    VkBufferHandle gemm_C;          /* [max(d,ds)×max(d,ds)] */
    uint32_t q09_k;
    VkBufferHandle q09_P;
    VkBufferHandle q09_P_left;
    VkBufferHandle q09_P_right;
    VkBufferHandle q09_stats;
    VkBufferHandle q09_dw_approx;

    // Q-10 additions
    VkBufferHandle q10_w_v;         /* [ds] */
    VkBufferHandle q10_dw_v;        /* [ds] */
    VkBufferHandle q10_v_head;      /* [batch * seq] */
    VkBufferHandle q10_w_v_m;       /* [ds] */
    VkBufferHandle q10_w_v_v;       /* [ds] */

    /* SSM scan transients (reused per-layer) */
    VkBufferHandle ssm_a;           /* A_eff [seq×ds] */
    VkBufferHandle ssm_b;           /* B_eff [seq×ds] */
    VkBufferHandle ssm_x;           /* x_cv [seq×ds] */
    VkBufferHandle ssm_h;           /* h_out [seq×ds] */
    VkBufferHandle ssm_cbl;         /* C_bilinear [ds] */
    VkBufferHandle ssm_u;           /* sampled_u [ds] */
    VkBufferHandle ef03_ring_sums;
    VkBufferHandle ssm_v;           /* sampled_v [ds] */

    /* Layer weights (uploaded per-layer, reused) */
    VkBufferHandle lw_conv;         /* [K × ds] */
    VkBufferHandle lw_B;            /* [ds] */
    VkBufferHandle lw_A;            /* [ds] */
    VkBufferHandle lw_WA;           /* [ds × ds] */
    VkBufferHandle lw_bA;           /* [ds] */
    VkBufferHandle lw_C;            /* [ds × ds] */

    /* Per-layer state */
    struct {
        VkTernaryCSR gpu_csr_in;
        VkTernaryCSR gpu_csr_out;
        VkTernaryCSR gpu_csc_in;    /* W_in^T for backward */
        VkTernaryCSR gpu_csc_out;   /* W_out^T for backward */
        int uploaded;
        int csc_uploaded;

        /* Forward activations (GPU-resident for backward) */
        VkBufferHandle fwd_x_proj;
        VkBufferHandle fwd_x_gated;
        VkBufferHandle fwd_mask_val;
        VkBufferHandle fwd_x_cv;
        VkBufferHandle fwd_b_eff;
        VkBufferHandle fwd_a_buf;
        VkBufferHandle fwd_h_cache;
        VkBufferHandle f34_P_r;
        VkBufferHandle f34_z_cache;
#ifdef R504_E10_PSCQ
    VkBufferHandle fwd_h_cache_compressed;
#endif
        VkBufferHandle qrf3ac_basis;
        VkBufferHandle qrf3ac_ternary;
        VkBufferHandle htr1p_u;
        VkBufferHandle htr1p_v;
        VkBufferHandle fwd_y_ssm;   /* per-layer y_ssm result [bseq x ds] */
        VkBufferHandle gpu_moments; /* GP-KL Welford accumulators on GPU   */
        VkBufferHandle tssi_score_moments; /* R420 compact target-conditioned state scores */
        int fwd_allocated;

        /* MTP buffers — resident weights (uploaded once, reused every backward) */
        VkBufferHandle gpu_mtp_U;   /* [d_state * mtp_rank] float — NULL if layer has no MTP */
        VkBufferHandle gpu_mtp_V;   /* [d_state * mtp_rank] float */
        VkBufferHandle gpu_A_const; /* [d_state] float — A vector broadcast source */

        /* MTP scratch — zeroed before each backward MTP dispatch */
        VkBufferHandle gpu_d_mtp_U; /* [d_state * mtp_rank] uint — CAS atomic accumulator */
        VkBufferHandle gpu_d_mtp_V; /* [d_state * mtp_rank] uint — CAS atomic accumulator */
        VkBufferHandle gpu_mtp_h;   /* [bseq * d_state] float — h_mtp from MTP forward scan */
        VkBufferHandle gpu_mtp_loss;/* [1] uint — CAS atomic loss accumulator */
        int mtp_allocated;
    } layers[V12_N_LAYERS];

    /* RMS norm weights + outputs */
    VkBufferHandle norm_w;          /* [d_model] gamma */
    VkBufferHandle norm_b;          /* [d_model] beta */
    VkBufferHandle norm_out;        /* [seq_len × d_model] normed output */
    VkBufferHandle norm_rms;        /* [seq_len] per-timestep RMS values */

    /* Backward scratch */
    VkBufferHandle bwd_d_y_ssm;
    VkBufferHandle bwd_d_xp;
    VkBufferHandle bwd_d_x_cv;
    VkBufferHandle bwd_dA_ps;
    VkBufferHandle bwd_dB_seq;
    VkBufferHandle bwd_v_pre;
VkBufferHandle bwd_d_h_in;
    VkBufferHandle bwd_d_h_in_prev_layer; // Q-16
    VkBufferHandle q16_sums_buf; // Q-16      /* GPU-resident d_h */
    VkBufferHandle bwd_h_final;     /* [batch_size × d_state] recurrent backward F1 seed state */
    VkBufferHandle bwd_d_h_prev;
    VkBufferHandle bwd_d_embed_E;   /* [vocab × d_embed] float — GPU accumulator for d_embed_E grad */
#ifdef R503_PHASE_1_EFFICIENCY
    VkBufferHandle bwd_d_embed_E_sparse; /* E-10 Sparse VRAM */
    uint32_t fati_chunk_size;            /* E-13 FATI */
#endif
    VkBufferHandle bwd_d_conv_w;
    VkBufferHandle bwd_d_tau;       /* [1] float — per-layer tau gradient accumulator */
    VkBufferHandle bwd_d_dot;       /* [bseq × ds] intermediate for A-backward */
    VkBufferHandle bwd_dA_reduce;   /* [ds] reduced dA per-state */
    VkBufferHandle bwd_dbA_reduce;  /* [ds] reduced db_A per-state */
    VkBufferHandle bwd_dC_bl_accum; /* [ds] dC_bilinear GPU accumulator */
    VkBufferHandle bwd_dB_accum;    /* [ds] uint — CAS atomic accumulator for Direct-B dB gradient */
    VkBufferHandle bwd_B_vec;       /* [ds] float — current layer B vector for Direct-B shader */

    /* =========================================================================
     * GPU Optimizer State
     *
     * Mirrors every Adam / factored-Adam / EMA-SPRT buffer in SSMLayer and
     * Model so that optimizer steps can run entirely on GPU without round-
     * tripping gradients or moments through CPU.
     *
     * Naming mirrors types.h exactly:
     *   opt_layers[k].<field>  ↔  m->layers[k].<field>
     *   opt_<global>           ↔  m-><global>
     *
     * Buffer slot 0 is reserved as VK_BUFFER_NULL, so calloc leaves handles invalid.
     * Allocation happens in gpu_optimizer_alloc() (not here).
     * ========================================================================= */

    struct {
        /* --- Ternary W_in [d_model × d_state] -------------------------------- */
        VkBufferHandle shadow_w_in;      /* shadow logits        [d_model × d_state] */
        VkBufferHandle m_w_in_row;       /* factored m row       [d_model]           */
        VkBufferHandle v_w_in_row;       /* factored v row       [d_model]           */
        VkBufferHandle m_w_in_col;       /* factored m col       [d_state]           */
        VkBufferHandle v_w_in_col;       /* factored v col       [d_state]           */
        VkBufferHandle grad_accum_in;    /* EMA-SPRT accum       [d_model × d_state] */
        VkBufferHandle crystal_mask_in;  /* crystallization mask [d_model × d_state] uint8 */
        VkBufferHandle sampled_w_in;     /* current ternary samples [d_model × d_state] float */
        VkBufferHandle g_row_in;         /* gradient row stats   [d_model] float             */
        VkBufferHandle g_col_in;         /* gradient col stats   [d_state] float             */
        VkBufferHandle gq_row_in;        /* 2nd-moment row stats [d_model] float             */
        VkBufferHandle gq_col_in;        /* 2nd-moment col stats [d_state] float             */
        VkBufferHandle cnt_row_in;       /* ternary count/row    [d_model] uint32            */
        VkBufferHandle cnt_col_in;       /* ternary count/col    [d_state] uint32            */
        VkBufferHandle mean_v_in;
        VkBufferHandle sign_ema_in; // Q-13        /* scalar mean-v output [1] float                   */

        /* --- Ternary W_out [d_state × d_model] ------------------------------- */
        VkBufferHandle shadow_w_out;
        VkBufferHandle m_w_out_row;      /* factored m row       [d_state]           */
        VkBufferHandle v_w_out_row;      /* factored v row       [d_state]           */
        VkBufferHandle m_w_out_col;      /* factored m col       [d_model]           */
        VkBufferHandle v_w_out_col;      /* factored v col       [d_model]           */
        VkBufferHandle grad_accum_out;   /* EMA-SPRT accum       [d_state × d_model] */
        VkBufferHandle crystal_mask_out; /* crystallization mask [d_state × d_model] uint8 */
        VkBufferHandle sampled_w_out;    /* current ternary samples [d_state × d_model] float */
        VkBufferHandle g_row_out;        /* gradient row stats   [d_state] float             */
        VkBufferHandle g_col_out;        /* gradient col stats   [d_model] float             */
        VkBufferHandle gq_row_out;       /* 2nd-moment row stats [d_state] float             */
        VkBufferHandle gq_col_out;       /* 2nd-moment col stats [d_model] float             */
        VkBufferHandle mprm_Sigma_M;
        VkBufferHandle mprm_Sigma_V;
        VkBufferHandle cnt_row_out;      /* ternary count/row    [d_state] uint32            */
        VkBufferHandle cnt_col_out;      /* ternary count/col    [d_model] uint32            */
        VkBufferHandle mean_v_out;
        VkBufferHandle sign_ema_out; // Q-13       /* scalar mean-v output [1] float                   */

        /* --- Dense: A [d_state] ---------------------------------------------- */
        VkBufferHandle m_A;              /* Adam m               [d_state]           */
        VkBufferHandle v_A;              /* Adam v               [d_state]           */
        VkBufferHandle dA;               /* gradient accum       [d_state]           */

        /* --- Dense: B [d_state] ---------------------------------------------- */
        VkBufferHandle m_B;
        VkBufferHandle v_B;
        VkBufferHandle dB;               /* gradient accum       [d_state]           */

        /* --- Dense: C [d_state × d_state] ------------------------------------ */
        VkBufferHandle m_C;              /* Adam m               [d_state × d_state] */
        VkBufferHandle v_C;              /* Adam v               [d_state × d_state] */
        VkBufferHandle dC;               /* gradient accum       [d_state × d_state] */

        /* --- Dense: sigma [d_state] ------------------------------------------ */
        VkBufferHandle m_sigma;
        VkBufferHandle v_sigma;
        VkBufferHandle d_sigma;          /* gradient accum       [d_state]           */

        /* --- Ternary conv1d [conv_kernel × d_state] -------------------------- */
        VkBufferHandle shadow_w_conv;    /* shadow logits        [K × d_state]       */
        VkBufferHandle m_conv;           /* Adam m               [K × d_state]       */
        VkBufferHandle v_conv;           /* Adam v               [K × d_state]       */
        VkBufferHandle d_conv_w;         /* gradient accum       [K × d_state]       */

        /* --- Dense: W_A [d_state × d_state], b_A [d_state] ------------------ */
        VkBufferHandle shadow_w_A;       /* shadow logits        [d_state × d_state] */
        VkBufferHandle m_W_A;            /* Adam m               [d_state × d_state] */
        VkBufferHandle v_W_A;            /* Adam v               [d_state × d_state] */
        VkBufferHandle dW_A;             /* gradient accum       [d_state × d_state] */
        VkBufferHandle m_b_A;            /* Adam m               [d_state]           */
        VkBufferHandle v_b_A;            /* Adam v               [d_state]           */
        VkBufferHandle db_A;             /* gradient accum       [d_state]           */

        /* --- Saved gradient buffers (written during backward, read by optimizer) */
        VkBufferHandle gpu_dw_out;       /* [d_model × d_state] saved dw_out gradient  */
        VkBufferHandle gpu_dw_in;
        VkBufferHandle holo_U_in;
        VkBufferHandle holo_V_in;
        VkBufferHandle gpu_dW_A;         /* [d_state × d_state] saved dW_A gradient    */
        VkBufferHandle gpu_dA;           /* [d_state] accumulated dA gradient           */
        VkBufferHandle gpu_dbA;          /* [d_state] accumulated db_A gradient         */
        VkBufferHandle gpu_dconv;        /* [K × d_state] accumulated d_conv_w gradient */

        /* --- Dense: W_A Adam moments [d_state × d_state] -------------------- */
        VkBufferHandle m_WA;             /* Adam m               [d_state × d_state]   */
        VkBufferHandle v_WA;             /* Adam v               [d_state × d_state]   */

        /* --- Dense: MTP U,V [d_state × mtp_rank] (NULL on odd layers) ------- */
        VkBufferHandle m_mtp_U;
        VkBufferHandle v_mtp_U;
        VkBufferHandle d_mtp_U;          /* gradient accum       [d_state × mtp_rank] */
        VkBufferHandle m_mtp_V;
        VkBufferHandle v_mtp_V;
        VkBufferHandle d_mtp_V;          /* gradient accum       [d_state × mtp_rank] */

        /* --- Ternary C_bilinear [d_state] ------------------------------------ */
        VkBufferHandle shadow_C_bilinear;
        VkBufferHandle m_C_bilinear;
        VkBufferHandle v_C_bilinear;
        VkBufferHandle dC_bilinear;      /* gradient accum       [d_state]           */

        int opt_allocated;               /* 1 after gpu_optimizer_alloc fills this slot */
    } opt_layers[V12_N_LAYERS];
    VkBufferHandle tmsheaf_sim[V12_N_LAYERS];

    /* -------------------------------------------------------------------------
     * Global (model-level) optimizer buffers
     * ------------------------------------------------------------------------- */

    /* lm_head [vocab_size × d_model] — factored Adam */
    VkBufferHandle opt_shadow_lm_head;   /* shadow logits        [vocab_size × d_model] */
    VkBufferHandle opt_m_lm_head_row;    /* factored m row       [vocab_size]            */
    VkBufferHandle opt_v_lm_head_row;    /* factored v row       [vocab_size]            */
    VkBufferHandle opt_m_lm_head_col;    /* factored m col       [d_model]               */
    VkBufferHandle opt_v_lm_head_col;    /* factored v col       [d_model]               */
    VkBufferHandle opt_dlm_head;         /* gradient accum       [vocab_size × d_model]  */
    VkBufferHandle opt_crystal_lm_head;  /* hot_mask / crystal   [vocab_size] uint8      */
    /* factored-Adam scratch for lm_head */
    VkBufferHandle opt_sampled_lm_head;  /* sampled ternary      [vocab_size × d_model] */
    VkBufferHandle opt_g_row_lm;         /* grad row stats       [vocab_size]            */
    VkBufferHandle opt_g_col_lm;         /* grad col stats       [d_model]               */
    VkBufferHandle opt_gq_row_lm;        /* quant grad row       [vocab_size]            */
    VkBufferHandle opt_gq_col_lm;        /* quant grad col       [d_model]               */
    VkBufferHandle opt_cnt_row_lm;       /* nonzero count row    [vocab_size]            */
    VkBufferHandle opt_cnt_col_lm;       /* nonzero count col    [d_model]               */
    VkBufferHandle opt_mean_v_lm;
    VkBufferHandle opt_sign_ema_lm; // Q-13        /* sum(v_row), sum(v_col) scratch [2]           */
    VkBufferHandle opt_vocab_to_hot_lm;  /* vocab row -> hot row map [vocab_size]        */

    /* w_pred [d_model × d_model] — factored Adam */
    VkBufferHandle opt_shadow_w_pred;    /* shadow logits        [d_model × d_model]     */
    VkBufferHandle opt_m_w_pred_row;     /* factored m row       [d_model]               */
    VkBufferHandle opt_v_w_pred_row;     /* factored v row       [d_model]               */
    VkBufferHandle opt_m_w_pred_col;     /* factored m col       [d_model]               */
    VkBufferHandle opt_v_w_pred_col;     /* factored v col       [d_model]               */
    VkBufferHandle opt_dw_pred;          /* per-step gradient    [d_model × d_model]     */
    VkBufferHandle opt_dlm_w_pred;       /* EMA-SPRT grad accum  [d_model × d_model]     */
    VkBufferHandle opt_sampled_w_pred;   /* sampled ternary      [d_model × d_model]     */
    VkBufferHandle opt_g_row_wp;         /* grad row stats       [d_model]               */
    VkBufferHandle opt_g_col_wp;         /* grad col stats       [d_model]               */
    VkBufferHandle opt_gq_row_wp;        /* quant grad row       [d_model]               */
    VkBufferHandle opt_gq_col_wp;        /* quant grad col       [d_model]               */
    VkBufferHandle opt_cnt_row_wp;       /* nonzero count row    [d_model]               */
    VkBufferHandle opt_cnt_col_wp;       /* nonzero count col    [d_model]               */
    VkBufferHandle opt_mean_v_wp;
    VkBufferHandle opt_sign_ema_wp; // Q-13        /* sum(v_row), sum(v_col) scratch [2]           */

    /* norm [d_model] — plain Adam */
    VkBufferHandle opt_m_norm_w;         /* Adam m               [d_model]               */
    VkBufferHandle opt_v_norm_w;         /* Adam v               [d_model]               */
    VkBufferHandle opt_m_norm_b;         /* Adam m               [d_model]               */
    VkBufferHandle opt_v_norm_b;         /* Adam v               [d_model]               */
    VkBufferHandle opt_d_norm_w;         /* gradient accum       [d_model]               */
    VkBufferHandle opt_d_norm_b;         /* gradient accum       [d_model]               */

    /* embed_E [vocab_size × d_embed] — dense Adam */
    VkBufferHandle opt_m_embed_E;        /* Adam m               [vocab_size × d_embed]  */
    VkBufferHandle opt_v_embed_E;        /* Adam v               [vocab_size × d_embed]  */
    /* factored Adam stubs (unused, reserved for future use) */
    VkBufferHandle opt_m_embed_E_row;    /* factored m row       [vocab_size]            */
    VkBufferHandle opt_v_embed_E_row;    /* factored v row       [vocab_size]            */
    VkBufferHandle opt_m_embed_E_col;    /* factored m col       [d_embed]               */
    VkBufferHandle opt_v_embed_E_col;    /* factored v col       [d_embed]               */
    VkBufferHandle opt_d_embed_E;        /* gradient accum       [vocab_size × d_embed]  */

    /* embed_P [d_embed × d_model] — plain Adam */
    VkBufferHandle opt_m_embed_P;        /* Adam m               [d_embed × d_model]     */
    VkBufferHandle opt_v_embed_P;        /* Adam v               [d_embed × d_model]     */
    
    VkBufferHandle mprm_P_U;
    VkBufferHandle mprm_P_V;

    VkBufferHandle opt_d_embed_P;        /* gradient accum       [d_embed × d_model]     */

    int opt_global_allocated;            /* 1 after gpu_optimizer_alloc fills global bufs */

    /* CPU staging for gradient download */
    float *staging_dw;              /* max(d*ds, d*d) floats */

    /* Persistent CPU scratch for backward */
    float *pc_neg_buf;              /* [T × d] */
    float *pc_res_buf;              /* [T × d] */
    float *pc_dw_buf;               /* [d × d] */

    /* Sampled CE loss GPU buffers */
    VkBufferHandle ce_row_ptr_pos;  /* lm_head CSR row_ptr_pos [vocab+1] */
    VkBufferHandle ce_pos_indices;  /* lm_head CSR pos_indices */
    VkBufferHandle ce_row_ptr_neg;  /* lm_head CSR row_ptr_neg [vocab+1] */
    VkBufferHandle ce_neg_indices;  /* lm_head CSR neg_indices */
    VkBufferHandle ce_unigram_cdf;  /* [vocab] float */
    VkBufferHandle ce_log_unigram;  /* [vocab] float */
    VkBufferHandle ce_token_freq; /* Q-06 */
    VkBufferHandle ce_sample_ids;   /* [B*(T-1)*N_SAMPLES] int32 scratch */
    VkBufferHandle ce_d_logits;     /* [B*(T-1)*N_SAMPLES] float scratch */
    VkBufferHandle ce_ema_logits;   /* [B*(T-1)*N_SAMPLES] float EMA */
    VkBufferHandle ce_loss;         /* [B*(T-1)] float per-position loss */
    VkBufferHandle ce_base_loss;    /* [B*(T-1)] baseline sampled CE */
    VkBufferHandle ce_d_norm_w;     /* [d_model] float */
    VkBufferHandle ce_d_norm_b;     /* [d_model] float */
    VkBufferHandle ce_dlm_head;     /* [vocab × d_model] float — GPU accumulator for dlm_head grad */

    VkBufferHandle prior_ctx_counts;
    VkBufferHandle prior_pair_keys;
    VkBufferHandle prior_pair_counts;
    VkBufferHandle prior_unigram;
    uint32_t prior_pair_capacity;
    int corpus_prior_enabled;
    VkBufferHandle program_kernel_targets;
    VkBufferHandle program_kernel_counts;
    int program_kernel_enabled;
    VkBufferHandle state_atlas_keys;
    VkBufferHandle state_atlas_targets;
    VkBufferHandle state_atlas_counts;
    uint32_t state_atlas_capacity;
    int state_atlas_enabled;

    /* Temporary count buffers for GPU CSR rebuild (max V12_N_LAYERS * max(d,ds) rows) */
    VkBufferHandle csr_count_pos;  /* [32064] uint32 — per-row positive count scratch */
    VkBufferHandle csr_count_neg;  /* [32064] uint32 — per-row negative count scratch */
    VkBufferHandle flip_counts;    /* [GPU_FLIP_COUNT_SLOTS] uint32 — per-matrix flip counters */
    VkBufferHandle evidence_rate_stats; /* [V12_EVIDENCE_RATE_STATS_SLOTS] uint32/float-bits */

    /* R364 slice A no-op interval probes. */
    uint32_t r364_noop_compile_id;
    uint32_t r364_noop_train_step_calls;
    uint32_t r364_noop_optimizer_step_calls;
    int r364_noop_interval_active;
    float r364_noop_compiled_block_share_f;

    uint8_t *online_tssi_train_state_mask;
    uint32_t online_tssi_train_mask_len;
    uint32_t online_tssi_train_active_dims;
    uint64_t online_tssi_train_applications;
    uint64_t online_tssi_train_masked_w_in_values;
    uint64_t online_tssi_train_masked_w_out_values;
    char online_tssi_train_control_name[32];

    /* R494 CCHS + SPCS skips */
    float prev_H;
    int has_prev_H;
    int skip_bptt_current_step;

    /* PRBLT (Pipelined Ring-Buffered Layer Transfer) */
    struct {
        VkAsyncTransferHandle transfer_handle;
        VkBufferHandle staging_buf;
        char* staging_mapped;

        VkBufferHandle shadow_w_in;
        VkBufferHandle shadow_w_out;

        VkTernaryCSR gpu_csr_in;
        VkTernaryCSR gpu_csr_out;
        VkTernaryCSR gpu_csc_in;
        VkTernaryCSR gpu_csc_out;
        int active_layer_k;
    } prblt_ring[2];
};
#else // !V12_HAS_VULKAN
struct GpuPipeline { int dummy; };
typedef struct { uint32_t handle; } VkBufferHandle;
typedef struct { uint32_t dummy; } VkTernaryCSR;
typedef struct R400ObsTraceConfig R400ObsTraceConfig;
#endif // V12_HAS_VULKAN

/* Prototypes */
FILE *r400_obs_trace_open_file(R400ObsTraceConfig *cfg);
GpuPipeline *gpu_pipeline_create(VulkanContext *ctx, Model *m, int seq_len, int batch_size);
__attribute__((unused)) int vk_dispatch_f14_ternary_activation_escrow(VulkanContext *ctx, VkBufferHandle fwd_h, VkBufferHandle dh_in, int seq_len, int d_state, int batch_size);
__attribute__((unused)) int vk_dispatch_f15_cross_layer_adjoint(VulkanContext *ctx, VkBufferHandle lambda_buf, VkBufferHandle df_dh_buf, uint32_t total_elements, float dl);
__attribute__((unused)) int vk_dispatch_sp10_psvcte(VulkanContext *ctx, VkBufferHandle A_eff, VkBufferHandle B_eff, VkBufferHandle x, VkBufferHandle C_bilinear, VkBufferHandle h_out, int seq_len, int d_state, int batch_size);
__attribute__((unused)) int vk_dispatch_sp11_lns_tfms(VulkanContext *ctx, VkBufferHandle A_eff, VkBufferHandle B_eff, VkBufferHandle x, VkBufferHandle C_bilinear, VkBufferHandle h_out, int seq_len, int d_state, int batch_size);
__attribute__((unused)) int vk_dispatch_sp14_gfit(VulkanContext *ctx, VkBufferHandle A_eff, VkBufferHandle B_eff, VkBufferHandle x, VkBufferHandle C_bilinear, VkBufferHandle h_out, int seq_len, int d_state, int batch_size);
__attribute__((unused)) int vk_dispatch_sp15_tt_rcrb(VulkanContext *ctx, VkBufferHandle A_eff, VkBufferHandle B_eff, VkBufferHandle x, VkBufferHandle C_bilinear, VkBufferHandle h_out, int seq_len, int d_state, int batch_size, int tt_rank);
__attribute__((unused)) int vk_dispatch_sp16_null_space_prune(VulkanContext *ctx, VkBufferHandle dh_in, VkBufferHandle w_out, int seq_len, int d_state, int d_model, int batch_size);
__attribute__((unused)) int vk_dispatch_sp17_dtss(VulkanContext *ctx, VkBufferHandle h_cache, VkBufferHandle dh_in, int seq_len, int d_state);
__attribute__((unused)) void f15_adjoint_validate_gate(VulkanContext *ctx, VkBufferHandle lambda_buf, VkBufferHandle df_dh_buf, uint32_t total_elements);
__attribute__((unused)) void vk_dispatch_f36_slefc_forward(VulkanContext *ctx, VkBufferHandle h_in, VkBufferHandle signs_out, VkBufferHandle norm_out, int seq_len, int d_state);
__attribute__((unused)) void vk_dispatch_f36_slefc_reconstruct(VulkanContext *ctx, VkBufferHandle signs_in, VkBufferHandle norm_in, VkBufferHandle h_out, int seq_len, int d_state);
__attribute__((unused)) void vk_dispatch_f38_cst(VulkanContext *ctx, VkBufferHandle h_buf, VkBufferHandle active_flags, VkBufferHandle lambda_buf, int seq_len, int d_state, float threshold, int mode);
__attribute__((unused)) void vk_dispatch_f39_bvgr(VulkanContext *ctx, VkBufferHandle a_buf, VkBufferHandle b_buf, VkBufferHandle x_buf, VkBufferHandle h_buf, int seq_len, int d_state);
const char *erc_mode_name(int mode);
double paper7_dfga_cosine(double dot, double a_sq, double b_sq);
double paper7_dfga_rel_l2(double diff_sq, double ref_sq);
double relative_residual_or_zero(double exact_energy, double residual_energy);
double safe_f64(float v);
double variance_priced_hash01(uint32_t x);
float erc_stat_float(uint32_t bits);
float gpu_train_step(GpuPipeline *p, Model *m, Cache *c,
                     const uint16_t *tokens, const uint8_t *boundary_mask,
                     int seq_len, int seq_idx, int batch_size,
                     const TrainState *state);
int *exact_quotient_padding_build_active_lens(
    Model *m, int batch_size, int seq_len, const TrainState *state,
    int *active_targets_out, int *inactive_targets_out);
int ensure_btrp_token_gather_pipeline(VulkanContext *ctx);
int ensure_e13_lrms_pipeline(VulkanContext *ctx);
int ensure_e13_parametric_cotangent_pipeline(VulkanContext *ctx);
int ensure_e32_aoi_ssm_pipeline(VulkanContext *ctx);
int ensure_ef03_pipeline(VulkanContext *ctx);
int ensure_entangled_adam_pipeline(VulkanContext *ctx);
int ensure_f01_rrd_involution_pipeline(VulkanContext *ctx);
int ensure_f12_fourier_pipeline(VulkanContext *ctx);
int ensure_f14_pipeline(VulkanContext *ctx);
int ensure_f15_pipeline(VulkanContext *ctx);
int ensure_f34_srat_forward_pipeline(VulkanContext *ctx);
int ensure_f34_srat_reconstruct_pipeline(VulkanContext *ctx);
int ensure_f35_idre_pipeline(VulkanContext *ctx);
int ensure_f36_slefc_forward_pipeline(VulkanContext *ctx);
int ensure_f36_slefc_reconstruct_pipeline(VulkanContext *ctx);
int ensure_f37_taqg_pipeline(VulkanContext *ctx);
int ensure_f38_cst_pipeline(VulkanContext *ctx);
int ensure_f39_bvgr_pipeline(VulkanContext *ctx);
int ensure_mprm_adam_pipeline(VulkanContext *ctx);
int ensure_sp10_pipeline(VulkanContext *ctx);
int ensure_sp11_pipeline(VulkanContext *ctx);
int ensure_sp12_pipeline(VulkanContext *ctx);
int ensure_sp13_pipeline(VulkanContext *ctx);
int ensure_sp14_pipeline(VulkanContext *ctx);
int ensure_sp15_pipeline(VulkanContext *ctx);
int ensure_sp16_pipeline(VulkanContext *ctx);
int ensure_sp17_pipeline(VulkanContext *ctx);
int ensure_sp19_holo_adam_pipeline(VulkanContext *ctx);
int env_flag_enabled(const char *value);
int exact_quotient_padding_choose_active_len(
    const Model *m, int seq_len, uint32_t step, uint64_t microbatch, int batch);
int gpu_pipeline_set_online_tssi_train_selector(GpuPipeline *p,
                                                const uint8_t *state_mask,
                                                uint32_t mask_len,
                                                const char *control_name,
                                                uint32_t active_dims);
int gpu_pipeline_sync_hidden_cache_to_cpu(GpuPipeline *p,
                                          Cache *c,
                                          int batch_size);
int gpu_pipeline_sync_lm_head_to_cpu(GpuPipeline *p, Model *m);
int gpu_pipeline_sync_r364_compile_cache_to_cpu(GpuPipeline *p,
                                                Model *m,
                                                Cache *c,
                                                int batch_size);
int gpu_pipeline_write_online_tssi_selector(GpuPipeline *p, const char *path, float active_fraction, const char *round_label);
int gpu_sync_to_cpu(GpuPipeline *p, Model *m);
int lagged_ce_impulse_eligibility_update(
    Model *m, SSMLayer *L, float *transition_credit, const float *ce_impulse,
    const float *h_cache, const float *a_eff, const float *a_gate,
    const float *x_cv, const uint8_t *boundary_mask,
    int batch_size, int seq_len, int d_state);
int layer_backward_state_ready(const SSMLayer *layer,
                                      int gpu_owns_w_in,
                                      int gpu_owns_w_out);
int materialize_a_gate_from_gpu_recompute(GpuPipeline *p,
                                                 SSMLayer *L,
                                                 int layer_idx,
                                                 float *dst,
                                                 size_t ssm_bytes,
                                                 int seq_len,
                                                 int ds,
                                                 int batch_size,
                                                 uint32_t gpu_opt_flags);
int r364_crashmap_logging_enabled(void);
int r400_obs_trace_append_summary(FILE *fp, R400ObsTraceConfig *cfg);
int r400_obs_trace_capture_training(GpuPipeline *p,
                                           VulkanContext *ctx,
                                           R400ObsTraceConfig *cfg,
                                           const SSMLayer *layer_weights,
                                           int step,
                                           int layer,
                                           int microbatch,
                                           const float *dA_ps,
                                           int seq_len,
                                           int ds,
                                           int batch_size);
int r400_obs_trace_capture_training(GpuPipeline *p,
                                           VulkanContext *ctx,
                                           R400ObsTraceConfig *cfg,
                                           const SSMLayer *layer_weights,
                                           int step,
                                           int layer,
                                           int microbatch,
                                           const float *dA_ps,
                                           int seq_len,
                                           int ds,
                                           int batch_size);

static int r400_obs_trace_capture_boundary_pair(FILE *fp,
                                                R400ObsTraceConfig *cfg,
                                                int step,
                                                int layer,
                                                const char *matrix_family,
                                                int coord_index,
                                                float grad_before,
                                                float shadow_before,
                                                float sampled_before,
                                                float shadow_after,
                                                float sampled_after,
                                                float ema_threshold);
int r400_obs_trace_enabled(void);
int r400_obs_trace_ensure_parent_dirs(const char *path);
int r400_obs_trace_group_index(R400ObsTraceConfig *cfg,
                                      const char *alignment_key);
int r400_obs_trace_mkdir_single(const char *path);
int r400_obs_trace_should_capture(R400ObsTraceConfig *cfg,
                                         int step,
                                         int layer);
int r400_obs_trace_truncate_file(FILE *fp, long offset);
int r400_obs_trace_write_absence_record(FILE *fp,
                                               R400ObsTraceConfig *cfg,
                                               int step,
                                               int microbatch,
                                               int layer,
                                               int seq_index,
                                               int horizon_k,
                                               const char *matrix_family,
                                               const char *objective_kind,
                                               int coord_index,
                                               const char *alignment_key,
                                               const char *object_kind,
                                               const char *producer_stage,
                                               const char *source_name,
                                               const char *shape,
                                               const char *zero_reason);
int r400_obs_trace_write_meta(FILE *fp, const R400ObsTraceConfig *cfg);
int r400_obs_trace_write_record(FILE *fp,
                                       R400ObsTraceConfig *cfg,
                                       int step,
                                       int microbatch,
                                       int layer,
                                       int seq_index,
                                       int horizon_k,
                                       const char *matrix_family,
                                       const char *objective_kind,
                                       int coord_index,
                                       const char *alignment_key,
                                       const char *object_kind,
                                       const char *producer_stage,
                                       const char *consumer_stage,
                                       const char *source_name,
                                       int source_valid,
                                       const char *shape,
                                       int sample_count,
                                       double l2,
                                       uint64_t nonzero_count,
                                       uint64_t finite_count,
                                       const char *zero_reason);
int r400_obs_trace_write_summary(FILE *fp, R400ObsTraceConfig *cfg);
int upload_csc_to_gpu(VulkanContext *ctx, VkTernaryCSR *dst, const TernaryCSC *csc);
int upload_csr_to_gpu(VulkanContext *ctx, VkTernaryCSR *dst, const TernaryCSR *src);
int variance_priced_credit_gate_update(
    Model *m, float *transition_credit, int batch_size,
    int seq_len, int d_state);
int vk_dispatch_e13_parametric_cotangent(VulkanContext *ctx, VkBufferHandle x_proj, VkBufferHandle w_pred, VkBufferHandle lambda_out, int seq_len, int d_state, int batch_size);
int vk_dispatch_e32_aoi_ssm(VulkanContext *ctx, VkBufferHandle A_eff, VkBufferHandle B_eff, VkBufferHandle x, VkBufferHandle C_bilinear, VkBufferHandle h_out, int seq_len, int d_state, int batch_size);
int vk_dispatch_f01_rrd_involution(VulkanContext *ctx, VkBufferHandle A_eff, VkBufferHandle B_eff, VkBufferHandle x, VkBufferHandle C_bilinear, VkBufferHandle h_out, int seq_len, int d_state, int batch_size);
int vk_dispatch_f12_fourier_ssm(VulkanContext *ctx, VkBufferHandle h_cache, int seq_len, int d_state, int batch_size);
int vk_dispatch_sp12_lcmbi(VulkanContext *ctx, VkBufferHandle A_eff, VkBufferHandle B_eff, VkBufferHandle x, VkBufferHandle C_bilinear, VkBufferHandle h_out, int seq_len, int d_state, int batch_size);
int vk_dispatch_sp13_sde_bvdj(VulkanContext *ctx, VkBufferHandle A_eff, VkBufferHandle B_eff, VkBufferHandle x, VkBufferHandle C_bilinear, VkBufferHandle h_out, int seq_len, int d_state, int batch_size);
int32_t r438_quantize_signature_value(float v, float scale);
uint64_t exact_quotient_padding_mix64(uint64_t x);
uint64_t r438_mix_u64(uint64_t h, uint64_t x);
uint64_t r438_state_transition_signature_hash(
    const float *h_cache, const float *a_eff, const uint8_t *boundary_mask,
    int pos, int d_state, float state_scale, float transition_scale);
void active_state_subspace_gate_update(
    Model *m, const float *grad, int batch_size, int seq_len, int d);
void boundary_cotangent_credit_gate_update(
    Model *m, const float *transition_credit, const uint8_t *boundary_mask,
    int batch_size, int seq_len, int d_state);
void boundary_flux_sheaf_gate_update(
    Model *m, const float *transition_credit, const uint8_t *boundary_mask,
    int batch_size, int seq_len, int d_state);
void boundary_liability_ledger_gate_update(
    Model *m, const float *grad, const uint8_t *boundary_mask,
    int batch_size, int seq_len, int d);
void certified_artifact_boundary_gate_update(
    Model *m, const uint16_t *tokens, int batch_size, int seq_len);
void certified_silent_span_gate_update(
    Model *m, const float *losses, int batch_size, int seq_len);
void download_lm_hot_row_moments(VulkanContext *ctx,
                                        VkBufferHandle src_full,
                                        const int32_t *hot_to_vocab,
                                        float *dst_hot);
void download_shadow_buffer(VulkanContext *ctx, shadow_t *dst, VkBufferHandle src, size_t count, uint64_t step);
void ef03_validate_gate(VulkanContext *ctx, VkBufferHandle h_buf, VkBufferHandle sum_buf, uint32_t bseq, uint32_t ds, uint32_t K);
void exact_quotient_padding_record(
    Model *m, int total_targets, int active_targets);
void exact_window_multiset_gate_update(
    Model *m, const uint16_t *tokens, int batch_size, int seq_len);
void executable_multiplicity_packets_update(
    Model *m, const float *dA_ps, const float *h_cache, const float *a_eff,
    const uint8_t *boundary_mask, int batch_size, int seq_len, int d_state);
void fp_interface_gap_probe_capture_layer(
    GpuPipeline *p, Model *m, int step, int layer,
    int batch_size, int seq_len, int d);
void free_vk_csr(VulkanContext *ctx, VkTernaryCSR *csr);
void gpu_csr_rebuild_from_sampled(GpuPipeline *p, VkBufferHandle sampled,
                                          VkTernaryCSR *gpu_csr,
                                          uint32_t rows, uint32_t cols);

void gpu_pipeline_rebuild_lm_head_ce_csr(GpuPipeline *p, Model *m);
void gpu_download_shadows(VulkanContext *ctx, int16_t *dst_shadow, VkBufferHandle src, size_t count, uint64_t step);
void gpu_gradient_null_witness_gate_update(GpuPipeline *p, Model *m);
void gpu_online_tssi_live_gate_update(GpuPipeline *p, Model *m);
void gpu_optimizer_step_dense(GpuPipeline *p, Model *m, float lr, int step);
void gpu_pipeline_destroy(GpuPipeline *p);
void gpu_pipeline_offload_moments(GpuPipeline *p, Model *m, int k);
void gpu_pipeline_print_stats(GpuPipeline *p);
void gpu_pipeline_r364_get_noop_probe(const GpuPipeline *p, GpuR364NoopProbe *out);
void gpu_pipeline_r364_set_noop_compiled_interval(GpuPipeline *p,
                                                  int active,
                                                  uint32_t compile_id,
                                                  float compiled_block_share_f);
void gpu_pipeline_resample_and_reupload_csr(GpuPipeline *p, Model *m, uint32_t rng_seed);
void gpu_pipeline_reupload_csr(GpuPipeline *p, Model *m, uint32_t rng_seed);
void gpu_pipeline_reupload_layer(GpuPipeline *p, Model *m, int k);
void gpu_pipeline_sync_lm_head_from_cpu(GpuPipeline *p, Model *m);
void gpu_pipeline_sync_materialized_state(GpuPipeline *p, Model *m);
void gpu_pipeline_sync_weights(GpuPipeline *p, Model *m);
void gpu_r423_native_growth_genotype_compact_update(GpuPipeline *p, int step);
void gpu_r423_native_growth_genotype_exception_gate_update(GpuPipeline *p, int step);
void gpu_r423_native_growth_interaction_release_safety_gate_update(GpuPipeline *p, int step);
void gpu_r423_native_growth_interaction_safe_update(GpuPipeline *p, int step);
void gpu_r423_native_growth_receipt_challenge_gate_update(GpuPipeline *p, int step);
void gpu_scale_gradients(GpuPipeline *p, float scale);
void gpu_upload_shadows(VulkanContext *ctx, VkBufferHandle dst, const int16_t *src_shadow, size_t count);
void gpu_zero_grad_accumulators(GpuPipeline *p);
void macro_transcript_gate_update(
    Model *m, const uint16_t *tokens, int batch_size, int seq_len);
void moments_from_gpu_flat(GpKlMoments *dst, const float *src, int ds);
void moments_to_gpu_flat(const GpKlMoments *src, float *dst, int ds);
void original_ce_credit_debt_observer_update(
    Model *m, const float *transition_credit, const float *direct_state_credit,
    const float *h_cache, const uint8_t *boundary_mask,
    int batch_size, int seq_len, int d_state);
void paper7_dfga_real_batch_audit(
    Model *m, const float *d_logits_buf, const float *norm_out_buf,
    const int32_t *sample_ids_buf, int batch_size, int seq_len, int d,
    int n_samples, const TrainState *state);
void prblt_alloc_max_csr(VulkanContext *ctx, VkTernaryCSR *csr, uint32_t max_rows, uint32_t max_cols);
void prblt_free_max_csr(VulkanContext *ctx, VkTernaryCSR *csr);
void prefix_frontier_gate_update(
    Model *m, const uint16_t *tokens, const float *d_logits,
    int batch_size, int seq_len, int n_samples);
void qgmf_gate_update(Model *m, const uint16_t *tokens, int batch_size, int seq_len);
void r400_obs_trace_compute_summary(const R400ObsTraceConfig *cfg,
                                           int *aligned_groups,
                                           int *live_state_groups,
                                           int *live_transition_groups,
                                           int *live_boundary_groups,
                                           int *live_objective_grad_groups,
                                           const char **decision,
                                           const char **reason);
void r400_obs_trace_emit_non_identifying(R400ObsTraceConfig *cfg,
                                                const char *reason,
                                                int step,
                                                int layer,
                                                const char *missing_object);
void r400_obs_trace_init_config(R400ObsTraceConfig *cfg,
                                       int batch_size,
                                       int n_layers);
void r400_obs_trace_log_activation_once(R400ObsTraceConfig *cfg);
void r400_obs_trace_mark_group(R400ObsTraceConfig *cfg,
                                      const char *alignment_key,
                                      const char *object_kind,
                                      int source_valid,
                                      double l2,
                                      uint64_t nonzero_count,
                                      const char *zero_reason);
void r400_obs_trace_summarize_double_values(const double *values,
                                                   int count,
                                                   double *l2,
                                                   uint64_t *nonzero_count,
                                                   uint64_t *finite_count,
                                                   const char **zero_reason);
void r400_obs_trace_summarize_float_values(const float *values,
                                                  int count,
                                                  double *l2,
                                                  uint64_t *nonzero_count,
                                                  uint64_t *finite_count,
                                                  const char **zero_reason);
void residual_active_sampling_gate_update(
    Model *m, const float *losses, int batch_size, int seq_len);
void reversible_deterministic_expansion_gate_update(
    Model *m, const uint16_t *tokens, int batch_size, int seq_len);
void upload_lm_hot_row_moments(VulkanContext *ctx,
                                      VkBufferHandle dst_full,
                                      const int32_t *hot_to_vocab,
                                      const float *src_hot);
void upload_shadow_buffer(VulkanContext *ctx, VkBufferHandle dst, const shadow_t *src, size_t count);
void vk_dispatch_btrp_token_gather(VulkanContext *ctx, VkBufferHandle embed_E, VkBufferHandle tokens, VkBufferHandle out_rows, uint32_t seq_len, uint32_t d_embed, uint32_t batch_size, uint32_t vocab_size, uint32_t r);
void vk_dispatch_e13_lrms(VulkanContext *ctx, VkBufferHandle nodes, uint32_t total_elements, float lrms_multiplier);
void vk_dispatch_ef03_ring_checkpoint(VulkanContext *ctx, VkBufferHandle h, VkBufferHandle sum, uint32_t bseq, uint32_t ds, uint32_t K);
void vk_dispatch_entangled_adam_update(VulkanContext *ctx, VkBufferHandle packed_state, VkBufferHandle grad, uint32_t N, float lr, float b1, float b2, float eps);
void vk_dispatch_f34_srat_forward(VulkanContext *ctx, VkBufferHandle h_in, VkBufferHandle p_r, VkBufferHandle z_out, int seq_len, int d_state, int r);
void vk_dispatch_f34_srat_reconstruct(VulkanContext *ctx, VkBufferHandle z_in, VkBufferHandle p_r, VkBufferHandle h_out, int seq_len, int d_state, int r);
void vk_dispatch_f35_idre(VulkanContext *ctx, VkBufferHandle A_eff, VkBufferHandle v, VkBufferHandle dh_out, int seq_len, int d_state, int batch_size);
void vk_dispatch_f37_taqg(VulkanContext *ctx, VkBufferHandle shadow, VkBufferHandle grad, uint32_t N, float lr);
void vk_dispatch_mprm_adam(VulkanContext *ctx, VkBufferHandle param, VkBufferHandle grad,
                                  VkBufferHandle p_u, VkBufferHandle p_v, VkBufferHandle sig_m, VkBufferHandle sig_v,
                                  uint32_t rows, uint32_t cols, uint32_t r, float lr, float b1, float b2, float eps, float inv_bc1, float inv_bc2);
void vk_dispatch_sp19_holo_adam_reduce(VulkanContext *ctx, 
    VkBufferHandle grad, VkBufferHandle holo_U, VkBufferHandle holo_V, uint32_t rank,
    VkBufferHandle shadow, VkBufferHandle sampled, VkBufferHandle grad_accum,
    VkBufferHandle m_row, VkBufferHandle m_col, VkBufferHandle v_row, VkBufferHandle v_col,
    VkBufferHandle g_row, VkBufferHandle g_col, VkBufferHandle gq_row, VkBufferHandle gq_col,
    VkBufferHandle cnt_row, VkBufferHandle cnt_col, VkBufferHandle mean_v_sums, VkBufferHandle row_map,
    uint32_t matrix_rows, uint32_t moment_rows, uint32_t cols, float lr, float inv_bc1, float inv_bc2,
    uint32_t row_map_mode, uint32_t evidence_gate, uint32_t evidence_mode, float evidence_boundary_eps, float tau,
    VkBufferHandle evidence_stats);

#endif // GPU_PIPELINE_INTERNAL_H

void vk_bind_buffer_to_descriptor(VulkanContext *ctx, int slot, int binding, VkBufferHandle h);

void prblt_load_layer(GpuPipeline *p, Model *m, int k);

// R507 Masterpiece Findings - Experimental Flags
#define R507_ENABLE_SSFC 1
#define R507_ENABLE_KRYLOV 1
#define R507_ENABLE_SHEAF 1
#define R507_ENABLE_FLUX 1
#define R507_ENABLE_HAMILTONIAN 1
#define R507_ENABLE_ESCROW 1

void ssfc_sparse_cotangent_step(v12_gpu_pipeline* p);
void krylov_subspace_descent(v12_gpu_pipeline* p);
void causal_entropy_sheaf_step(v12_gpu_pipeline* p);
void discrete_area_integrator(v12_gpu_pipeline* p);
void apply_latent_energy_hamiltonian(v12_gpu_pipeline* p);
void apply_ternary_commitment_escrow(v12_gpu_pipeline* p);
