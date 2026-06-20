CC = gcc
CFLAGS = -O2 -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-missing-field-initializers -I src/ -fopenmp
LDFLAGS = -lm -fopenmp

# -----------------------------------------------------------------------------
# EXTRA_CFLAGS: load-bearing hook for feature and debug flags.
#
# WHY THIS EXISTS:
#   Passing flags via `make CFLAGS=...` REPLACES the Makefile's own CFLAGS
#   instead of appending to it. That silently clobbers the conditional
#   `CFLAGS += -DV12_HAS_VULKAN ...` block below (and any other `CFLAGS +=`
#   accumulation), because `make` treats a command-line CFLAGS assignment
#   as an override that wins over in-Makefile assignments.
#
# SYMPTOM IF IGNORED:
#   With `CFLAGS="-DV12_M1_SLICE_A"` on the command line, `V12_HAS_VULKAN`
#   is silently undefined. The build still SUCCEEDS, producing a CPU-only
#   stub binary. At runtime training dies with:
#       "GPU pipeline failed - cannot train without GPU"
#   This failure mode is hard to diagnose because the compile is clean.
#
# SUPPORTED USAGE (current R368 M1 flags):
#   mingw32-make USE_VULKAN=1 EXTRA_CFLAGS="-DV12_M1_SLICE_A" train
#   mingw32-make USE_VULKAN=1 EXTRA_CFLAGS="-DV12_M1_SLICE_A -DV12_DEBUG_M1" train
#   mingw32-make USE_VULKAN=1 EXTRA_CFLAGS="-DV12_M1_SLICE_B -DV12_DEBUG_M1" train
#
# RULE:
#   Feature toggles and debug defines MUST go through EXTRA_CFLAGS.
#   NEVER pass them via `CFLAGS=...` on the command line - it will break
#   the Vulkan build path and any other conditional CFLAGS block.
# -----------------------------------------------------------------------------
CFLAGS += $(EXTRA_CFLAGS)

# Detect Windows/MSYS
ifeq ($(OS),Windows_NT)
    USE_MMAP = 0
    EXE_EXT = .exe
else
    USE_MMAP = 1
    EXE_EXT =
endif

# Add Windows detection for MSYSTEM as well
ifdef MSYSTEM
    USE_MMAP = 0
    EXE_EXT = .exe
endif

# Add USE_MMAP to CFLAGS if Windows
ifeq ($(USE_MMAP),0)
    CFLAGS += -DUSE_MMAP=0
endif

# Support VOCAB_SIZE override
ifdef VOCAB_SIZE
    CFLAGS += -DV12_VOCAB_SIZE=$(VOCAB_SIZE)
endif

# --- Vulkan backend (opt-in: make USE_VULKAN=1) ---
USE_VULKAN=1
ifdef USE_VULKAN
    VULKAN_SDK ?= C:/VulkanSDK/1.4.341.1
    CFLAGS += -DV12_HAS_VULKAN -DV12_HAS_GPU_CE_SHADERS -DV12_PC_GPU_SHADERS -I$(VULKAN_SDK)/Include
    LDFLAGS += -L$(VULKAN_SDK)/Lib -lvulkan-1
    VULKAN_IMPL = src/runtime/vulkan_backend_impl.c
    GLSLC = $(VULKAN_SDK)/Bin/glslc.exe
    SPIRV_HDR = src/runtime/shaders/vulkan_shaders.h
endif

SSM_SCAN_SRC ?= src/core/ssm_scan.c

SRCS = src/train.c  \
    src/gpu/gpu_ops_backward.c \
    src/gpu/gpu_ops_krylov.c \
    src/gpu/gpu_ops_topological.c \
    src/gpu/gpu_ops_flux.c \
    src/gpu/gpu_ops_hamiltonian.c \
    src/gpu/gpu_pipeline_vulkan.c \
    src/gpu/gpu_pipeline_opt.c \
    src/gpu/gpu_pipeline_sync.c \
    src/gpu/gpu_pipeline_init.c \
    src/gpu/gpu_pipeline_audit.c \
    src/gpu/gpu_pipeline_train.c \
    src/gpu/gpu_pipeline_core.c src/core/model.c src/core/ternary_matmul.c $(SSM_SCAN_SRC) src/core/loss.c src/core/optimizer.c src/core/optimizer_f3_measure.c src/core/pca_cache.c src/core/r364_upper_subtree_ls.c src/core/y_cache.c src/io/dataset.c src/io/checkpoint.c src/io/teaching.c src/runtime/hardware.c src/core/r493_gates.c

# Vulkan: replace stub with real implementation
USE_VULKAN=1
ifdef USE_VULKAN
    SRCS += $(VULKAN_IMPL)
else
    SRCS += src/runtime/vulkan_backend.c
endif
OBJS = $(SRCS:.c=.o)
HEADERS = $(wildcard src/core/*.h src/gpu/*.h src/io/*.h src/runtime/*.h src/runtime/shaders/*.h src/eval/*.h)

.PHONY: all clean tests shaders generate eval_k_ssm

USE_VULKAN=1
ifdef USE_VULKAN
shaders:
	$(GLSLC) -O --target-env=vulkan1.2 -DS01_PACKED=1 src/runtime/shaders/ternary_matvec.comp -o src/runtime/shaders/spv/ternary_matvec.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/ternary_matvec.spv $(SPIRV_HDR)
	$(GLSLC) -O src/runtime/shaders/rms_norm_backward.comp -o src/runtime/shaders/spv/rms_norm_backward.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/rms_norm_backward.spv src/runtime/shaders/rms_norm_backward_spv.h
	$(GLSLC) -O src/runtime/shaders/wht_inplace.comp -o src/runtime/shaders/spv/wht_inplace.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/wht_inplace.spv src/runtime/shaders/wht_inplace_spv.h
	$(GLSLC) -O src/runtime/shaders/conv1d_bwd_input_gate.comp -o src/runtime/shaders/spv/conv1d_bwd_input_gate.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/conv1d_bwd_input_gate.spv src/runtime/shaders/conv1d_bwd_input_gate_spv.h
	$(GLSLC) -O src/runtime/shaders/direct_b_backward.comp -o src/runtime/shaders/spv/direct_b_backward.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/direct_b_backward.spv src/runtime/shaders/direct_b_backward_spv.h
	$(GLSLC) -O src/runtime/shaders/welford_update.comp -o src/runtime/shaders/spv/welford_update.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/welford_update.spv src/runtime/shaders/welford_update_spv.h
	$(GLSLC) -O src/runtime/shaders/eligibility.comp -o src/runtime/shaders/spv/eligibility.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/eligibility.spv src/runtime/shaders/eligibility_spv.h
	$(GLSLC) -O src/runtime/shaders/apply_whitening.comp -o src/runtime/shaders/spv/apply_whitening.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/apply_whitening.spv src/runtime/shaders/apply_whitening_spv.h
	$(GLSLC) -O src/runtime/shaders/online_tssi_score_update.comp -o src/runtime/shaders/spv/online_tssi_score_update.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/online_tssi_score_update.spv src/runtime/shaders/online_tssi_score_update_spv.h
	$(GLSLC) -O src/runtime/shaders/pc_gather_prev.comp -o src/runtime/shaders/spv/pc_gather_prev.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/pc_gather_prev.spv src/runtime/shaders/pc_gather_prev_spv.h
	$(GLSLC) -O src/runtime/shaders/residual_sub.comp -o src/runtime/shaders/spv/residual_sub.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/residual_sub.spv src/runtime/shaders/residual_sub_spv.h
	$(GLSLC) -O src/runtime/shaders/pc_negate_gather.comp -o src/runtime/shaders/spv/pc_negate_gather.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/pc_negate_gather.spv src/runtime/shaders/pc_negate_gather_spv.h
	$(GLSLC) -O src/runtime/shaders/residual_scatter_add.comp -o src/runtime/shaders/spv/residual_scatter_add.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/residual_scatter_add.spv src/runtime/shaders/residual_scatter_add_spv.h
	$(GLSLC) -O src/runtime/shaders/ssm_bwd_replay_f1.comp -o src/runtime/shaders/spv/ssm_bwd_replay_f1.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/ssm_bwd_replay_f1.spv src/runtime/shaders/ssm_bwd_replay_f1_spv.h
	$(GLSLC) -O src/runtime/shaders/output_project_backward.comp -o src/runtime/shaders/spv/output_project_backward.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/output_project_backward.spv src/runtime/shaders/output_project_backward_spv.h
	$(GLSLC) -O src/runtime/shaders/a_gate_recompute.comp -o src/runtime/shaders/spv/a_gate_recompute.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/a_gate_recompute.spv src/runtime/shaders/a_gate_recompute_spv.h
	$(GLSLC) -O src/runtime/shaders/sampled_ce_forward.comp -o src/runtime/shaders/spv/sampled_ce_forward.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/sampled_ce_forward.spv src/runtime/shaders/sampled_ce_forward_spv.h
	$(GLSLC) -O -DTOLR_ENABLED=1 src/runtime/shaders/sampled_ce_forward.comp -o src/runtime/shaders/spv/sampled_ce_forward_tolr.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/sampled_ce_forward_tolr.spv src/runtime/shaders/sampled_ce_forward_tolr_spv.h
	$(GLSLC) -O src/runtime/shaders/sampled_ce_scatter_dh.comp -o src/runtime/shaders/spv/sampled_ce_scatter_dh.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/sampled_ce_scatter_dh.spv src/runtime/shaders/sampled_ce_scatter_dh_spv.h
	$(GLSLC) -O -DTOLR_ENABLED=1 src/runtime/shaders/sampled_ce_scatter_dh.comp -o src/runtime/shaders/spv/sampled_ce_scatter_dh_tolr.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/sampled_ce_scatter_dh_tolr.spv src/runtime/shaders/sampled_ce_scatter_dh_tolr_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/factored_adam_reduce.comp -o src/runtime/shaders/spv/factored_adam_reduce.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/factored_adam_reduce.spv src/runtime/shaders/factored_adam_reduce_spv.h
	$(GLSLC) -O src/runtime/shaders/factored_adam_moment.comp -o src/runtime/shaders/spv/factored_adam_moment.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/factored_adam_moment.spv src/runtime/shaders/factored_adam_moment_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/factored_adam_update.comp -o src/runtime/shaders/spv/factored_adam_update.spv
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/entangled_adam_update.comp -o src/runtime/shaders/spv/entangled_adam_update.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/factored_adam_update.spv src/runtime/shaders/factored_adam_update_spv.h
	python tools/spv_to_header.py src/runtime/shaders/spv/entangled_adam_update.spv src/runtime/shaders/entangled_adam_update_spv.h
	$(GLSLC) -O src/runtime/shaders/random_projector.comp -o src/runtime/shaders/spv/random_projector.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/random_projector.spv src/runtime/shaders/random_projector_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/cos_sim_audit.comp -o src/runtime/shaders/spv/cos_sim_audit.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/cos_sim_audit.spv src/runtime/shaders/cos_sim_audit_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/forward_linear.comp -o src/runtime/shaders/spv/forward_linear.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/forward_linear.spv src/runtime/shaders/forward_linear_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/backward_linear.comp -o src/runtime/shaders/spv/backward_linear.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/backward_linear.spv src/runtime/shaders/backward_linear_spv.h
	$(GLSLC) -O src/runtime/shaders/q10_adam_update.comp -o src/runtime/shaders/spv/q10_adam_update.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/q10_adam_update.spv src/runtime/shaders/q10_adam_update_spv.h
	$(GLSLC) -O src/runtime/shaders/q10_value_forward.comp -o src/runtime/shaders/spv/q10_value_forward.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/q10_value_forward.spv src/runtime/shaders/q10_value_forward_spv.h
	$(GLSLC) -O src/runtime/shaders/q10_value_backward.comp -o src/runtime/shaders/spv/q10_value_backward.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/q10_value_backward.spv src/runtime/shaders/q10_value_backward_spv.h
	$(GLSLC) -O src/runtime/shaders/mvue_resample.comp -o src/runtime/shaders/spv/mvue_resample.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/mvue_resample.spv src/runtime/shaders/mvue_resample_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/ssm_scan.comp -o src/runtime/shaders/spv/ssm_scan.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/ssm_scan.spv src/runtime/shaders/ssm_scan_spv.h
	$(GLSLC) -O src/runtime/shaders/ssm_scan_backward.comp -o src/runtime/shaders/spv/ssm_scan_backward.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/ssm_scan_backward.spv src/runtime/shaders/ssm_scan_backward_spv.h
	$(GLSLC) -O src/runtime/shaders/ssm_scan_backward_cr.comp -o src/runtime/shaders/spv/ssm_scan_backward_cr.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/ssm_scan_backward_cr.spv src/runtime/shaders/ssm_scan_backward_cr_spv.h
	$(GLSLC) -O src/runtime/shaders/ssm_bwd_dh_norm.comp -o src/runtime/shaders/spv/ssm_bwd_dh_norm.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/ssm_bwd_dh_norm.spv src/runtime/shaders/ssm_bwd_dh_norm_spv.h
	$(GLSLC) -O src/runtime/shaders/ssm_bwd_project_dh.comp -o src/runtime/shaders/spv/ssm_bwd_project_dh.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/ssm_bwd_project_dh.spv src/runtime/shaders/ssm_bwd_project_dh_spv.h
	$(GLSLC) -O src/runtime/shaders/ssm_fused.comp -o src/runtime/shaders/spv/ssm_fused.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/ssm_fused.spv src/runtime/shaders/ssm_fused_spv.h
	$(GLSLC) -O src/runtime/shaders/ssm_scan_aas.comp -o src/runtime/shaders/spv/ssm_scan_aas.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/ssm_scan_aas.spv src/runtime/shaders/ssm_scan_aas_spv.h
	$(GLSLC) -O src/runtime/shaders/q19_kl_escrow.comp -o src/runtime/shaders/spv/q19_kl_escrow.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/q19_kl_escrow.spv src/runtime/shaders/q19_kl_escrow_spv.h
	$(GLSLC) -O src/runtime/shaders/q20_power_iteration.comp -o src/runtime/shaders/spv/q20_power_iteration.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/q20_power_iteration.spv src/runtime/shaders/q20_power_iteration_spv.h
	$(GLSLC) -O src/runtime/shaders/q22_subspace_alignment.comp -o src/runtime/shaders/spv/q22_subspace_alignment.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/q22_subspace_alignment.spv src/runtime/shaders/q22_subspace_alignment_spv.h
	$(GLSLC) -O src/runtime/shaders/q23_lsh_dedup.comp -o src/runtime/shaders/spv/q23_lsh_dedup.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/q23_lsh_dedup.spv src/runtime/shaders/q23_lsh_dedup_spv.h
	$(GLSLC) -O src/runtime/shaders/ef03_ring_checkpoint.comp -o src/runtime/shaders/spv/ef03_ring_checkpoint.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/ef03_ring_checkpoint.spv src/runtime/shaders/ef03_ring_checkpoint_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/sp04_warcp.comp -o src/runtime/shaders/spv/sp04_warcp.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/sp04_warcp.spv src/runtime/shaders/sp04_warcp_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/sp05_dfmsp.comp -o src/runtime/shaders/spv/sp05_dfmsp.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/sp05_dfmsp.spv src/runtime/shaders/sp05_dfmsp_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/sp06_bihd.comp -o src/runtime/shaders/spv/sp06_bihd.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/sp06_bihd.spv src/runtime/shaders/sp06_bihd_spv.h
	$(GLSLC) -O src/runtime/shaders/ssm_segment_backward.comp -o src/runtime/shaders/spv/ssm_segment_backward.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/ssm_segment_backward.spv src/runtime/shaders/ssm_segment_backward_spv.h
	$(GLSLC) -O src/runtime/shaders/contrastive_loss.comp -o src/runtime/shaders/spv/contrastive_loss.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/contrastive_loss.spv src/runtime/shaders/contrastive_loss_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/sp07_spbsc.comp -o src/runtime/shaders/spv/sp07_spbsc.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/sp07_spbsc.spv src/runtime/shaders/sp07_spbsc_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/sp08_qrf3ac.comp -o src/runtime/shaders/spv/sp08_qrf3ac.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/sp08_qrf3ac.spv src/runtime/shaders/sp08_qrf3ac_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/sp09_htr1p.comp -o src/runtime/shaders/spv/sp09_htr1p.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/sp09_htr1p.spv src/runtime/shaders/sp09_htr1p_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/e10_pscq_compress.comp -o src/runtime/shaders/spv/e10_pscq_compress.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/e10_pscq_compress.spv src/runtime/shaders/e10_pscq_compress_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/e10_pscq_decompress.comp -o src/runtime/shaders/spv/e10_pscq_decompress.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/e10_pscq_decompress.spv src/runtime/shaders/e10_pscq_decompress_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/sp11_tmsheaf.comp -o src/runtime/shaders/spv/sp11_tmsheaf.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/sp11_tmsheaf.spv src/runtime/shaders/sp11_tmsheaf_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/sp10_psvcte.comp -o src/runtime/shaders/spv/sp10_psvcte.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/sp10_psvcte.spv src/runtime/shaders/sp10_psvcte_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/sp14_gfit.comp -o src/runtime/shaders/spv/sp14_gfit.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/sp14_gfit.spv src/runtime/shaders/sp14_gfit_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/sp15_tt_rcrb.comp -o src/runtime/shaders/spv/sp15_tt_rcrb.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/sp15_tt_rcrb.spv src/runtime/shaders/sp15_tt_rcrb_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/sp11_lns_tfms.comp -o src/runtime/shaders/spv/sp11_lns_tfms.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/sp11_lns_tfms.spv src/runtime/shaders/sp11_lns_tfms_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/f12_fourier_ssm.comp -o src/runtime/shaders/spv/f12_fourier_ssm.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/f12_fourier_ssm.spv src/runtime/shaders/f12_fourier_ssm_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/e13_parametric_cotangent.comp -o src/runtime/shaders/spv/e13_parametric_cotangent.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/e13_parametric_cotangent.spv src/runtime/shaders/e13_parametric_cotangent_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/sp16_null_space_prune.comp -o src/runtime/shaders/spv/sp16_null_space_prune.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/sp16_null_space_prune.spv src/runtime/shaders/sp16_null_space_prune_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/f15_cross_layer_adjoint.comp -o src/runtime/shaders/spv/f15_cross_layer_adjoint.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/f15_cross_layer_adjoint.spv src/runtime/shaders/f15_cross_layer_adjoint_spv.h

	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/sp03_almm_pack.comp -o src/runtime/shaders/spv/sp03_almm_pack.spv
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/sp03_almm_unpack.comp -o src/runtime/shaders/spv/sp03_almm_unpack.spv
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/sp03_almm_scan.comp -o src/runtime/shaders/spv/sp03_almm_scan.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/sp03_almm_pack.spv src/runtime/shaders/sp03_almm_pack_spv.h
	python tools/spv_to_header.py src/runtime/shaders/spv/sp03_almm_unpack.spv src/runtime/shaders/sp03_almm_unpack_spv.h
	python tools/spv_to_header.py src/runtime/shaders/spv/sp03_almm_scan.spv src/runtime/shaders/sp03_almm_scan_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/f35_idre.comp -o src/runtime/shaders/spv/f35_idre.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/f35_idre.spv src/runtime/shaders/f35_idre_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/f36_slefc_forward.comp -o src/runtime/shaders/spv/f36_slefc_forward.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/f36_slefc_forward.spv src/runtime/shaders/f36_slefc_forward_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/f36_slefc_reconstruct.comp -o src/runtime/shaders/spv/f36_slefc_reconstruct.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/f36_slefc_reconstruct.spv src/runtime/shaders/f36_slefc_reconstruct_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/f37_taqg.comp -o src/runtime/shaders/spv/f37_taqg.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/f37_taqg.spv src/runtime/shaders/f37_taqg_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/f38_cst.comp -o src/runtime/shaders/spv/f38_cst.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/f38_cst.spv src/runtime/shaders/f38_cst_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/f39_bvgr.comp -o src/runtime/shaders/spv/f39_bvgr.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/f39_bvgr.spv src/runtime/shaders/f39_bvgr_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/e13_lrms.comp -o src/runtime/shaders/spv/e13_lrms.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/e13_lrms.spv src/runtime/shaders/e13_lrms_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/vsa_fwht_forward.comp -o src/runtime/shaders/spv/vsa_fwht_forward.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/vsa_fwht_forward.spv src/runtime/shaders/vsa_fwht_forward_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/vsa_bent_recurrence.comp -o src/runtime/shaders/spv/vsa_bent_recurrence.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/vsa_bent_recurrence.spv src/runtime/shaders/vsa_bent_recurrence_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/vsa_fractional_laplacian.comp -o src/runtime/shaders/spv/vsa_fractional_laplacian.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/vsa_fractional_laplacian.spv src/runtime/shaders/vsa_fractional_laplacian_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/vsa_dyadic_translation.comp -o src/runtime/shaders/spv/vsa_dyadic_translation.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/vsa_dyadic_translation.spv src/runtime/shaders/vsa_dyadic_translation_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/vsa_dyadic_ternary_convolution.comp -o src/runtime/shaders/spv/vsa_dyadic_ternary_convolution.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/vsa_dyadic_ternary_convolution.spv src/runtime/shaders/vsa_dyadic_ternary_convolution_spv.h
	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/vsa_dynamic_permutation_routing.comp -o src/runtime/shaders/spv/vsa_dynamic_permutation_routing.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/vsa_dynamic_permutation_routing.spv src/runtime/shaders/vsa_dynamic_permutation_routing_spv.h
endif


all: train

train: $(OBJS)
	$(CC) $(CFLAGS) -o train$(EXE_EXT) $(OBJS) $(LDFLAGS)

v13_multiscale:
	$(MAKE) train EXTRA_CFLAGS="-DARCH_MULTISCALE" SSM_SCAN_SRC="src/core/ssm_scan_multiscale.c" EXE_EXT="_v13_multiscale.exe"

v14_hadamard:
	$(MAKE) train EXTRA_CFLAGS="-DARCH_HADAMARD" SSM_SCAN_SRC="src/core/ssm_scan_hadamard.c" EXE_EXT="_v14_hadamard.exe"

v15_gf2:
	$(MAKE) train EXTRA_CFLAGS="-DARCH_GF2" SSM_SCAN_SRC="src/core/ssm_scan_gf2.c" EXE_EXT="_v15_gf2.exe"

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

test_loss: tests/test_loss.c src/core/loss.o src/core/ternary_matmul.o $(HEADERS)
	$(CC) $(CFLAGS) -o test_loss$(EXE_EXT) tests/test_loss.c src/core/loss.o src/core/ternary_matmul.o $(LDFLAGS)

test_matmul: tests/test_ternary_matmul.c src/core/ternary_matmul.o src/core/model.o src/core/optimizer.o $(HEADERS)
	$(CC) $(CFLAGS) -o test_matmul$(EXE_EXT) tests/test_ternary_matmul.c src/core/ternary_matmul.o src/core/model.o src/core/optimizer.o $(LDFLAGS)

test_ssm: tests/test_ssm_scan.c src/core/ssm_scan.o src/core/loss.o src/core/ternary_matmul.o src/core/model.o src/core/optimizer.o $(HEADERS)
	$(CC) $(CFLAGS) -o test_ssm$(EXE_EXT) tests/test_ssm_scan.c src/core/ssm_scan.o src/core/loss.o src/core/ternary_matmul.o src/core/model.o src/core/optimizer.o $(LDFLAGS)

test_optimizer: tests/test_optimizer.c src/core/optimizer.o src/core/model.o src/core/ternary_matmul.o $(HEADERS)
	$(CC) $(CFLAGS) -o test_optimizer$(EXE_EXT) tests/test_optimizer.c src/core/optimizer.o src/core/model.o src/core/ternary_matmul.o $(LDFLAGS)

test_dataset: tests/test_dataset.c src/io/dataset.o $(HEADERS)
	$(CC) $(CFLAGS) -o test_dataset$(EXE_EXT) tests/test_dataset.c src/io/dataset.o $(LDFLAGS)

test_r356_f1: tests/test_r356_f1.c src/core/ssm_scan.o src/core/loss.o src/core/ternary_matmul.o src/core/model.o src/core/optimizer.o $(HEADERS)
	$(CC) $(CFLAGS) -o test_r356_f1$(EXE_EXT) tests/test_r356_f1.c src/core/ssm_scan.o src/core/loss.o src/core/ternary_matmul.o src/core/model.o src/core/optimizer.o $(LDFLAGS)

test_f2_int8_shadow: tests/test_f2_int8_shadow.c src/core/shadow_codec.h
	$(CC) $(CFLAGS) -o test_f2_int8_shadow$(EXE_EXT) tests/test_f2_int8_shadow.c $(LDFLAGS)

test_r357_alt3: tests/test_r357_alt3_residual_shell.c src/core/loss.o src/core/ternary_matmul.o $(HEADERS)
	$(CC) $(CFLAGS) -o test_r357_alt3$(EXE_EXT) tests/test_r357_alt3_residual_shell.c src/core/loss.o src/core/ternary_matmul.o $(LDFLAGS)

test_r370_profile: tests/test_r370_profile.c $(HEADERS)
	$(CC) $(CFLAGS) -o test_r370_profile$(EXE_EXT) tests/test_r370_profile.c $(LDFLAGS)

GENERATE_SRCS = src/generate.c \
    src/core/model.c \
    src/core/ternary_matmul.c \
    src/core/ssm_scan.c \
    src/core/loss.c \
    src/core/optimizer.c \
    src/io/checkpoint.c \
    src/runtime/hardware.c \
    src/runtime/vulkan_backend.c

generate: $(GENERATE_SRCS) $(HEADERS)
	$(CC) $(CFLAGS) -o generate$(EXE_EXT) $(GENERATE_SRCS) $(LDFLAGS)

# --- R357 M3 slice 1 — offline K_SSM calculator (standalone) ---
EVAL_K_SSM_SRCS = tools/eval_k_ssm.c src/eval/k_ssm.c src/io/dataset.c

eval_k_ssm: $(EVAL_K_SSM_SRCS) src/eval/k_ssm.h src/io/dataset.h
	$(CC) $(CFLAGS) -o eval_k_ssm$(EXE_EXT) $(EVAL_K_SSM_SRCS) $(LDFLAGS)

R394_SSH_PROBE_SRCS = tools/r394_ssh_probe.c

r394_ssh_probe: $(R394_SSH_PROBE_SRCS) $(HEADERS)
	$(CC) $(CFLAGS) -o r394_ssh_probe$(EXE_EXT) $(R394_SSH_PROBE_SRCS) $(LDFLAGS)

R394_SSH_TRACE_PROBE_SRCS = tools/r394_ssh_trace_probe.c src/core/ssm_scan.c

r394_ssh_trace_probe: $(R394_SSH_TRACE_PROBE_SRCS) $(HEADERS)
	$(CC) $(CFLAGS) -o r394_ssh_trace_probe$(EXE_EXT) $(R394_SSH_TRACE_PROBE_SRCS) $(LDFLAGS)

R394_BCT_PROBE_SRCS = tools/r394_bct_probe.c

r394_bct_probe: $(R394_BCT_PROBE_SRCS) $(HEADERS)
	$(CC) $(CFLAGS) -o r394_bct_probe$(EXE_EXT) $(R394_BCT_PROBE_SRCS) $(LDFLAGS)

R399_SSQT_GATE_SRCS = tools/r399_ssqt_gate.c

r399_ssqt_gate: $(R399_SSQT_GATE_SRCS)
	$(CC) $(CFLAGS) -o r399_ssqt_gate$(EXE_EXT) $(R399_SSQT_GATE_SRCS) $(LDFLAGS)

R399_DMDC_GATE_SRCS = tools/r399_dmdc_gate.c

r399_dmdc_gate: $(R399_DMDC_GATE_SRCS)
	$(CC) $(CFLAGS) -o r399_dmdc_gate$(EXE_EXT) $(R399_DMDC_GATE_SRCS) $(LDFLAGS)

R405_TCFVL_GATE_SRCS = tools/r405_tcfvl_gate.c src/core/model.c src/core/ternary_matmul.c src/core/ssm_scan.c src/core/loss.c src/core/optimizer.c src/core/pca_cache.c src/core/y_cache.c src/io/dataset.c src/io/checkpoint.c src/runtime/hardware.c

r405_tcfvl_gate: $(R405_TCFVL_GATE_SRCS) $(HEADERS)
	$(CC) $(CFLAGS) -o r405_tcfvl_gate$(EXE_EXT) $(R405_TCFVL_GATE_SRCS) $(LDFLAGS)

R405_LVS_LEDGER_GATE_SRCS = tools/r405_lvs_ledger_gate.c src/core/model.c src/core/ternary_matmul.c src/core/ssm_scan.c src/core/loss.c src/core/optimizer.c src/core/pca_cache.c src/core/y_cache.c src/io/dataset.c src/io/checkpoint.c src/runtime/hardware.c

r405_lvs_ledger_gate: $(R405_LVS_LEDGER_GATE_SRCS) $(HEADERS)
	$(CC) $(CFLAGS) -o r405_lvs_ledger_gate$(EXE_EXT) $(R405_LVS_LEDGER_GATE_SRCS) $(LDFLAGS)

R408_EXACT_MULTIPLICITY_GATE_SRCS = tools/r408_exact_multiplicity_gate.c src/io/dataset.c

r408_exact_multiplicity_gate: $(R408_EXACT_MULTIPLICITY_GATE_SRCS) src/io/dataset.h
	$(CC) $(CFLAGS) -o r408_exact_multiplicity_gate$(EXE_EXT) $(R408_EXACT_MULTIPLICITY_GATE_SRCS) $(LDFLAGS)

R408_CORPUS_SCAFFOLD_GATE_SRCS = tools/r408_corpus_scaffold_gate.c src/io/dataset.c

r408_corpus_scaffold_gate: $(R408_CORPUS_SCAFFOLD_GATE_SRCS) src/io/dataset.h
	$(CC) $(CFLAGS) -o r408_corpus_scaffold_gate$(EXE_EXT) $(R408_CORPUS_SCAFFOLD_GATE_SRCS) $(LDFLAGS)

R408_CORPUS_SCAFFOLD_RESIDUAL_PILOT_SRCS = tools/r408_corpus_scaffold_residual_pilot.c src/io/dataset.c

r408_corpus_scaffold_residual_pilot: $(R408_CORPUS_SCAFFOLD_RESIDUAL_PILOT_SRCS) src/io/dataset.h
	$(CC) $(CFLAGS) -o r408_corpus_scaffold_residual_pilot$(EXE_EXT) $(R408_CORPUS_SCAFFOLD_RESIDUAL_PILOT_SRCS) $(LDFLAGS)

R394_RBTC_PROBE_SRCS = tools/r394_rbtc_probe.c

r394_rbtc_probe: $(R394_RBTC_PROBE_SRCS) $(HEADERS)
	$(CC) $(CFLAGS) -o r394_rbtc_probe$(EXE_EXT) $(R394_RBTC_PROBE_SRCS) $(LDFLAGS)

R375_PBG2_MVUE_PROBE_SRCS = tools/r375_pbg2_mvue_probe.c

r375_pbg2_mvue_probe: $(R375_PBG2_MVUE_PROBE_SRCS) $(HEADERS)
	$(CC) $(CFLAGS) -o r375_pbg2_mvue_probe$(EXE_EXT) $(R375_PBG2_MVUE_PROBE_SRCS) $(LDFLAGS)

test_k_ssm: tests/test_k_ssm.c src/eval/k_ssm.c src/eval/k_ssm.h
	$(CC) $(CFLAGS) -o test_k_ssm$(EXE_EXT) tests/test_k_ssm.c src/eval/k_ssm.c $(LDFLAGS)

r494_slice1_adjoint_gate: tools/r494_slice1_adjoint_gate.c
	$(CC) $(CFLAGS) -o r494_slice1_adjoint_gate$(EXE_EXT) tools/r494_slice1_adjoint_gate.c $(LDFLAGS)

r494_slice2_skip_gate: tools/r494_slice2_skip_gate.c
	$(CC) $(CFLAGS) -o r494_slice2_skip_gate$(EXE_EXT) tools/r494_slice2_skip_gate.c $(LDFLAGS)

r495_slice1_associative_gate: tools/r495_slice1_associative_gate.c
	$(CC) $(CFLAGS) -o r495_slice1_associative_gate$(EXE_EXT) tools/r495_slice1_associative_gate.c $(LDFLAGS)

tests: test_loss test_matmul test_ssm test_optimizer test_dataset test_r356_f1 test_f2_int8_shadow test_r357_alt3 test_k_ssm test_r370_profile

clean:
	rm -f $(OBJS) train$(EXE_EXT)
	rm -f src/runtime/vulkan_backend.o src/runtime/vulkan_backend_impl.o
	rm -f test_loss$(EXE_EXT) test_matmul$(EXE_EXT) test_ssm$(EXE_EXT) test_optimizer$(EXE_EXT) test_dataset$(EXE_EXT) test_r356_f1$(EXE_EXT) test_f2_int8_shadow$(EXE_EXT) test_r357_alt3$(EXE_EXT) test_k_ssm$(EXE_EXT) test_r370_profile$(EXE_EXT) eval_k_ssm$(EXE_EXT) r394_ssh_probe$(EXE_EXT) r394_ssh_trace_probe$(EXE_EXT) r394_bct_probe$(EXE_EXT) r394_rbtc_probe$(EXE_EXT) r375_pbg2_mvue_probe$(EXE_EXT) r399_ssqt_gate$(EXE_EXT) r399_dmdc_gate$(EXE_EXT) r405_tcfvl_gate$(EXE_EXT) r405_lvs_ledger_gate$(EXE_EXT) r408_exact_multiplicity_gate$(EXE_EXT) r408_corpus_scaffold_gate$(EXE_EXT) r408_corpus_scaffold_residual_pilot$(EXE_EXT) r494_slice1_adjoint_gate$(EXE_EXT) r494_slice2_skip_gate$(EXE_EXT) r495_slice1_associative_gate$(EXE_EXT)


