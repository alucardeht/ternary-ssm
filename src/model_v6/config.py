from dataclasses import dataclass, field
from typing import List
import math

@dataclass
class ModelConfig:
    d_model: int = 256
    n_layer: int = 8
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int = 16
    vocab_size: int = 8192
    seq_len: int = 512

    n_heads: int = 4
    window_size: int = 128
    swa_layer_indices: List[int] = field(default_factory=lambda: [4])

    use_head_gating: bool = False
    use_sink_tokens: bool = False
    n_sink_tokens: int = 1

    use_routing: bool = False
    routing_capacity: float = 0.5

    use_ternary: bool = False
    ternary_group_size: int = 128
    ste_type: str = 'standard'  # 'standard', 'biste', 'hestia', 'quest'
    ternary_lm_head: bool = False
    use_optrot: bool = False
    optrot_block_size: int = 64
    optrot_lambda: float = 0.01
    use_deferred_ternary: bool = False
    deferred_rank: int = 32
    deferred_merge_interval: int = 100

    use_tscs: bool = False
    tscs_osc_fraction: float = 0.375
    tscs_reset_fraction: float = 0.375
    tscs_clamp_state: float = 0.0
    use_tscs_p2_prefix_sum: bool = False  # TSCS Phase 2: parallel prefix-sum scan

    use_btg: bool = False  # BTG: Backward-Ternary GEMV (ternary backward pass)
    use_bitmask_gemv: bool = False  # Dual-bitmask GEMV: 8x less memory bandwidth via set partitioning

    use_recursive_ssm: bool = False
    recursive_ssm_iterations: int = 0  # 0 = auto (n_layer - n_swa_layers)

    # TADA (Ternary Algebraic Depth Amplification)
    use_tada: bool = False
    tada_residual_injection: bool = True     # Per-iteration learned residual mixing
    tada_adaptive_k: bool = False            # Token-dependent iteration count (training)
    tada_adaptive_k_max: int = 0             # Max K for adaptive mode (0 = use recursive_ssm_iterations)
    tada_rank_monitor_interval: int = 0      # Log effective rank every N steps (0 = disable)

    use_aga: bool = False
    aga_threshold_k: int = 32
    aga_decay_steps: int = 0
    aga_ternary_pull: float = 0.0
    aga_lr_multiplier: float = 1.0
    aga_use_int16: bool = True

    use_bbo: bool = False
    bbo_threshold_k: int = 16
    bbo_decay_steps: int = 100
    bbo_magnitude_threshold: float = 0.0
    bbo_two_step: bool = True

    use_caq: bool = False
    caq_alpha: float = 1.0

    use_moe: bool = False
    moe_num_experts: int = 4
    moe_top_k: int = 1
    moe_d_ff: int = 0
    moe_layer_indices: list = None
    moe_load_balance_coeff: float = 0.01

    use_ssd: bool = False
    ssd_n_heads: int = 16
    use_slow_heads: bool = False
    slow_head_fraction: float = 0.25
    medium_head_fraction: float = 0.25

    use_moa: bool = False
    moa_window_sizes: tuple = None

    use_rwkv7: bool = False
    rwkv7_head_size: int = 64
    rwkv7_n_heads: int = None
    rwkv7_lora_rank: int = 32
    rwkv7_chunk_size: int = 64

    use_amp: bool = False
    use_bf16: bool = False
    use_gradient_checkpointing: bool = False
    checkpoint_every_n: int = 1
    use_ssm_checkpoint: bool = False  # SSM-internal scan gradient checkpointing (Finding M1)
    ternary_aware_checkpointing: bool = False  # A2-M7: always recompute SSM layers, store SWA
    use_phase_space_attn: bool = False  # A5: Phase-Space Attention - Q/K from SSM B/C instead of residual stream
    use_sdpa: bool = True

    tbptt_k: int = 24  # Truncated BPTT for RecursiveSSM (R3-A4-M1+M3). Backprop through last K iterations only (0=disabled)

    use_activation_quant: bool = False
    activation_quant_bits: int = 4
    activation_quant_group_size: int = 128
    activation_quant_beta_start: float = 5.0
    activation_quant_beta_end: float = 50.0
    activation_quant_warmup: int = 100

    use_torch_compile: bool = False
    compile_memory_budget: float = None  # F2: Fraction of memory for activations (0.0-1.0, None=disabled)
    use_cpu_offload: bool = False
    activation_checkpoint_granularity: str = "none"  # F1: "none", "full", "selective"

    use_ags: bool = False
    ags_top_k_frac: float = 0.05
    ags_full_sync_interval: int = 50
    ags_delta: float = 0.1

    use_distillation: bool = False
    teacher_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    distill_alpha: float = 0.5
    distill_temperature: float = 2.0
    use_tsld: bool = False

    tied_embeddings: bool = True
    dropout: float = 0.0

    dataloader_num_workers: int = 4
    dataloader_prefetch_factor: int = 4

    use_cuda_prefetch: bool = False
    auto_batch_size: bool = False
    vram_budget_mb: int = 10000

    use_sequence_packing: bool = False

    # Early exit (inference - cosine similarity criterion)
    use_early_exit: bool = False
    early_exit_interval: int = 4
    early_exit_threshold: float = 0.99

    # Adaptive depth early exit (RecursiveSSM inference - per-token convergence)
    adaptive_depth_exit: bool = False
    adaptive_depth_epsilon: float = 0.01
    adaptive_depth_min_iters: int = 2

    use_eigenmode_mixed: bool = False
    eigenmode_slow_fraction: float = 0.3125

    use_ctri: bool = False
    ctri_selective_layers: List[int] = field(default_factory=lambda: [0])

    # FPIT (Full-Pipeline Integer Training)
    activation_fn: str = 'silu'          # 'silu' (default) or 'relu'
    use_bitshift_norm: bool = False      # Replace RMSNorm with BitShiftNorm

    use_margin_sparse: bool = False
    margin_delta: float = 0.1
    margin_ema_beta: float = 0.99
    margin_refresh_interval: int = 10

    use_fused_requantize: bool = False

    # INSTANT: Low-Rank Activation Compression (R2-A2-F6)
    use_instant: bool = False
    instant_rank: int = 32          # Low-rank projection dimension
    instant_update_gap: int = 50    # Recompute SVD projection every N steps
    instant_warmup: int = 100       # Steps before INSTANT activates (full precision during warmup)

    # FTEC (Factored Ternary Exponential Capacity)
    use_ftec: bool = False
    ftec_n_factors: int = 8                    # K: factorization depth
    ftec_sparsity: int = 3                     # c: non-zeros per row per factor
    ftec_pattern: str = 'random'               # Sparsity pattern: 'random', 'butterfly', 'shift'
    ftec_target_layers: str = 'out_proj'       # Which projections to factorize
    ftec_skip_connection: bool = True          # Residual skip for gradient flow

    use_anderson_acceleration: bool = False
    anderson_beta: float = 0.3

    use_geometric_extrapolation: bool = False
    geom_extrap_min_iters: int = 4
    geom_extrap_ratio_threshold: float = 0.8

    # R4-A5: Portable Kernel Algorithms (Batches 2-7)
    # Batch 2: Storage & Memory Layout
    use_quinary_packing: bool = False

    # Batch 3: Pattern LUT & SIMD
    use_block_pattern_lut: bool = False
    block_pattern_k: int = 4

    # Batch 4: Recursive Depth Optimization
    use_rtdp: bool = False
    rtdp_threshold: float = 0.01
    rtdp_diagnostics: bool = False
    use_ternary_circuit_compile: bool = False
    circuit_compile_recompile_interval: int = 500
    use_adaptive_dataflow: bool = False

    # Batch 5: SSM Scan Optimization (requires TSCS)
    use_scan_template: bool = False
    use_sparse_scan: bool = False
    use_parity_dual_stream: bool = False
    use_absorbing_pruning: bool = False

    # Batch 6: Deployment Targets
    use_ternary_binary_decomp: bool = False
    ternary_binary_block_size: int = 4

    # Batch 7: Advanced Explorations
    use_gf3_factorization: bool = False
    gf3_min_speedup: float = 1.5
    use_butterfly_factorization: bool = False
    butterfly_target_layers: str = 'all'
    use_t3m: bool = False
    t3m_activation_threshold: float = 0.5
    use_parallel_depth_scan: bool = False
    parallel_depth_warmup: int = 3
    parallel_depth_n_probes: int = 16

    use_episodic_state: bool = False
    use_episodic_gate: bool = False

    # M5: SSM-SPECTRAL-LEAKAGE-PACKING - compute spectral radius
    # (No config needed - this is infrastructure used by M7, M8, P4)

    # M7: MEMORY-ADAPTIVE-CURRICULUM-SCHEDULE
    use_adaptive_seq_len: bool = False
    adaptive_seq_min: int = 64
    adaptive_seq_max: int = 512
    adaptive_seq_interval: int = 100

    # M8: SSM-STATE-WARMUP-TAX / BATCH-GEOMETRY
    use_batch_geometry: bool = False
    batch_geometry_base_batch: int = 16
    batch_geometry_base_seq: int = 128

    # P4: STATEFUL-STREAMING-TRAINING
    use_streaming_training: bool = False
    streaming_num_streams: int = 4

    # Contractive Recurrence Regularization (R4-A3-M3)
    use_contractive_reg: bool = False
    contractive_lambda: float = 0.01
    contractive_target: float = 1.0

    # Per-Depth Learning Rate Scaling (R4-A3-F7) - RecursiveSSM only
    use_per_depth_lr: bool = False
    per_depth_lr_min: float = 0.3
    per_depth_lr_early_end: int = 3
    per_depth_lr_late_start: int = 0

    # RecursiveSSM Iteration Load-Balancing (R4-A3-F6)
    use_recursive_load_balance: bool = False
    recursive_load_balance_coeff: float = 0.01

    # UCB-Bandit Iteration Selection (R4-A3-P2)
    use_ucb_bandit: bool = False
    ucb_min_iterations: int = 4
    ucb_exploration_c: float = 1.0

    # Sparse Active Shadow Weights (R4-A2-M6/P4)
    use_sparse_shadow: bool = False
    sparse_shadow_margin: float = 0.1
    sparse_shadow_decay: float = 0.0
    sparse_shadow_interval: int = 10

    # R3-A1-M1: Vote-Buffer Convergence
    use_vote_buffer: bool = False
    vote_window: int = 50
    vote_k_sigma: float = 3.0

    # R3-A1-M3: Lyapunov-Stable Arenas
    use_arenas: bool = False
    arenas_lambda_0: float = 1.0
    arenas_epsilon: float = 0.01

    # R3-A4-M11: Ternary Pole Reassignment
    use_ternary_pole_reassignment: bool = False
    pole_reassignment_interval: int = 50

    # R3-A6-M1: Adam-Aware Flip Probability
    use_adam_flip_probability: bool = False
    adam_flip_horizon: int = 100
    adam_flip_epsilon: float = 0.01

    # FP8 Gradient Accumulation (R4-A2-F2)
    use_fp8_grad_accum: bool = False
    fp8_grad_accum_error_feedback: bool = True
    # F9: Top-K Gradient Sparsification
    use_topk_grad_sparsification: bool = False
    topk_grad_sparsity: float = 0.9  # Keep top 10% of gradients

    # Uncertainty-Weighted Multi-Task Loss (R4-A3-F5, Kendall et al.)
    use_uncertainty_loss: bool = False
    uncertainty_n_tasks: int = 4
    uncertainty_initial_logvar: float = 0.0

    # Multi-Token Prediction (F7 + M6 Ternary-Capacity-Bounded-MTP)
    mtp_num_heads: int = 0
    mtp_lambda_main: float = 0.5
    mtp_lambda_aux: list = None

    # P5: Ternary Channel-Aligned Loss
    use_gradient_sign_consistency: bool = False
    grad_sign_window: int = 8
    grad_sign_threshold: int = 6

    # A2-F6: Layerwise Gauss-Newton Preconditioning (Fisher proxy)
    use_fisher_preconditioning: bool = False
    fisher_damping: float = 1e-4

    # A2-F7: SignSGD for main training
    use_sign_sgd: bool = False
    sign_sgd_momentum: float = 0.9

    # A2-M7: Consensus-Filtered-Layer-Gradients (RecursiveSSM)
    use_consensus_gradient_filter: bool = False
    consensus_alpha: float = 0.5

    # A2-F8: Hessian-Aware Bit Allocation (adaptive quantization)
    use_adaptive_quantization: bool = False
    adaptive_quant_sensitivity_threshold: float = 0.1
    adaptive_quant_interval: int = 500

    # A2-M1: RECURSIVE-INVERTIBLE-CHECKPOINT (non-uniform checkpoint density)
    use_nonuniform_checkpointing: bool = False

    # A2-M3: VRAM-PARETO-CLOSED-FORM CONFIG GENERATOR
    use_auto_config: bool = False

    # A2-M4: SQRT(s) SSM STATE CHECKPOINT STRATEGY
    use_sqrt_ssm_checkpointing: bool = False

    # A2-M8: INFORMATION-THEORETIC TRAINING BUDGET (SNR DIAGNOSTIC)
    use_snr_diagnostic: bool = False
    snr_freeze_threshold: float = 0.1
    snr_freeze_interval: int = 100

    # A2-F4: REVERSIBLE-LAYERS-REVNET
    use_reversible_layers: bool = False

    # A2-F10: COMBINATORIAL-SUBMODULAR-TERNARY-OPTIMIZATION (post-training)
    use_combinatorial_finetune: bool = False
    combinatorial_max_sweeps: int = 3

    # A2-M5: UCB-BANDIT-TERNARY-TRAINING (Eliminate STE, direct ternary optimization)
    use_ucb_ternary_training: bool = False
    ucb_ternary_exploration_c: float = 1.0

    # A2-M6: ACTIVATION-GATED-FLIP-CRITERION (FSCR - prevent outlier-driven flips)
    use_fscr_flip_gate: bool = False
    fscr_threshold: float = 2.0

    # P1: SPECTRAL-CACHED-SSM-BACKWARD
    use_spectral_ssm: bool = False

    # P2: SPSA-TERNARY-CONCURRENT-FLIP-ESTIMATION
    use_spsa_training: bool = False
    spsa_perturbation_scale: float = 0.1
    spsa_lr: float = 0.01

    # P3: INTEGER-RANK-LOSS-TRAINING
    use_rank_margin_loss: bool = False
    rank_margin: float = 1.0
    rank_num_negatives: int = 64

    # P5: WHITTLE-INDEX-ASYNCHRONOUS-TRAINING
    use_whittle_index: bool = False
    whittle_update_fraction: float = 0.1

    # R4-A3-F4: Self-Knowledge Distillation (SSKD)
    use_sskd: bool = False
    sskd_alpha: float = 0.1
    sskd_temperature: float = 2.0

    # R4-A3-F8: Dynamic Loss Weighting
    use_dynamic_loss_weighting: bool = False
    dlw_gamma: float = 1.0

    # R4-A3-M1: Ternary Hamming Margin Loss
    use_thm_loss: bool = False
    thm_margin: float = 0.3
    thm_lambda: float = 0.01

    # R4-A3-M2: Ternary-Aligned Integer Loss
    use_tail_loss: bool = False
    tail_lambda: float = 0.01
    tail_flip_margin: float = 1.0
    tail_num_negatives: int = 5

    # R4-A3-M3: Recursive Consensus Loss (RCL)
    use_rcl_loss: bool = False
    rcl_beta: float = 1.0
    rcl_lambda: float = 0.05
    rcl_sample_every: int = 4
    collect_recursive_states: bool = False  # Shared infra for RCL/RAFU

    # R4-A3-M4: Ternary Entropy Rate Loss
    use_ter_loss: bool = False
    ter_loss_mu: float = 0.1
    ter_noise_var: float = 0.01

    # R4-A3-M5: Gradient Consensus Polarization
    use_gcp: bool = False
    gcp_alpha: float = 1.0

    # R4-A3-M6: Recurrence-Aware Flip Utility
    use_rafu: bool = False
    rafu_threshold: float = 0.1
    rafu_interval: int = 50

    # R4-A3-M7: Information-Bottleneck Layerwise Loss
    use_ibll: bool = False
    ibll_beta: float = 1.0
    ibll_gamma: float = 0.1
    ibll_eps: float = 0.01
    ibll_lambda: float = 0.02
    ibll_sample_every: int = 8

    # R4-A3-M8: Bayesian Flip Accumulator
    use_bfa: bool = False
    bfa_rho: float = 0.95
    bfa_tau: float = 1.0
    bfa_theta: float = 0.9
    bfa_interval: int = 10

    # R4-A3-P1: Minimax Depth Equilibrium
    use_mde: bool = False
    mde_eta: float = 0.1
    mde_lambda: float = 0.05
    mde_sample_every: int = 4

    # R4-A3-P2: Ternary Error-Correcting Weight Codes
    use_tecwc: bool = False
    tecwc_code_rate: float = 0.9
    tecwc_lambda: float = 0.001

    # R4-A3-P3: Spectral Surrogate Objective
    use_sso: bool = False
    sso_rho_target: float = 0.95
    sso_alpha: float = 1.0
    sso_beta: float = 0.1
    sso_interval: int = 100

    # R4-A3-P4: Spin-Glass Annealing
    use_sgapd: bool = False
    sgapd_initial_temp: float = 1.0
    sgapd_min_temp: float = 0.01
    sgapd_decay: float = 0.995
    sgapd_sweeps: int = 10
    sgapd_interval: int = 20

    # R4-A3-P5: Ergodic Trajectory Optimizer
    use_ergodic: bool = False
    ergodic_alpha: float = 1.0
    ergodic_rho_target: float = 0.9
    ergodic_flip_fraction: float = 0.01
    ergodic_interval: int = 50

    # R4-A3-P6: Minimax Depth Fairness
    use_minimax_depth: bool = False
    minimax_tau: float = 1.0
    minimax_hard: bool = False
    minimax_lambda: float = 0.05
    minimax_sample_every: int = 4

    # R4-A3-P7: Koopman Spectral Shaping
    use_koopman: bool = False
    koopman_long_target: float = 0.97
    koopman_fast_target: float = 0.5
    koopman_lambda: float = 0.01
    koopman_interval: int = 100
    koopman_max_dim: int = 64

    # R4-A3-P8: TD Depth Learning
    use_td_dl: bool = False
    td_lambda: float = 0.8
    td_dl_weight: float = 0.05
    td_sample_every: int = 2

    # ============================================================
    # R4-TIER2-BATCH4: Final 3 TIER 2 Findings
    # ============================================================

    # F12: FRAR - Fast Residual Alignment Regularization
    use_frar: bool = False
    frar_lambda_start: float = 0.01
    frar_lambda_end: float = 0.0001
    frar_warmup_steps: int = 100

    # F16: M5+M6 - Ternary Magnitude Pruning + Gradient Rescaling
    use_ternary_magnitude_pruning: bool = False
    magnitude_pruning_threshold_multiplier: float = 0.9
    use_gradient_rescaling: bool = False

    # F20: Low-Rank Momentum Factorization
    use_low_rank_momentum: bool = False
    low_rank_momentum_rank: int = 64
    low_rank_svd_interval: int = 500
    low_rank_min_dim: int = 128

    # ============================================================
    # R4-A4: Checkpoint, Resume & Fault Tolerance
    # ============================================================

    # R4-A4-F10: Gradient Health Metrics
    use_gradient_monitor: bool = False
    gradient_monitor_window: int = 100

    # R4-A4-F7: RAM Checkpointing
    use_ram_checkpoint: bool = False
    ram_checkpoint_disk_backup: bool = True

    # R4-A4-F8: Torch Deterministic Mode
    use_deterministic_mode: bool = False

    # R4-A4-F1: BitSnap Ternary Checkpoint Compression
    use_ternary_checkpoint_compression: bool = False

    # R4-A4-P8: Contractive Recurrence Guarantee (CRG)
    use_crg: bool = False
    crg_alpha_init: float = 3.0

    # R4-A4-F3: Incremental Delta Checkpoints
    use_incremental_checkpoints: bool = False
    delta_full_interval: int = 1000

    # R4-A4-M5: Tiered Checkpoint Strategy
    use_tiered_checkpoint: bool = False
    tiered_tier2_interval: int = 50
    tiered_tier3_interval: int = 500
    tiered_stability_threshold: float = 0.3

    # R4-A4-F5: Optimizer State Compression (k-means)
    use_optimizer_compression: bool = False

    # R4-A4-M1: Fiber Bundle Shadow Compression
    use_fiber_shadow_compression: bool = False

    # R4-A4-M2: TDSS Optimizer Compression
    use_tdss_compression: bool = False

    # R4-A4-F6: TRANSOM-TEE Anomaly Detection
    use_anomaly_detector: bool = False
    anomaly_z_threshold: float = 3.0
    anomaly_ring_buffer_size: int = 3

    # R4-A4-F2: Process Async Checkpoint (marker - uses RAM checkpoint infra)
    use_async_checkpoint: bool = False

    # R4-A4-P5: Quantization Margin Maximization
    use_qmm_regularizer: bool = False
    qmm_lambda: float = 0.01
    qmm_margin_target: float = 0.3

    # R4-A4-P7: Antifragile Ternary Inoculation
    use_ternary_inoculation: bool = False
    inoculation_flip_prob: float = 0.01
    inoculation_flip_fraction: float = 0.001
    inoculation_loss_weight: float = 0.1

    # R4-A4-M6: Recursive Trajectory Verification
    use_trajectory_verification: bool = False
    trajectory_verify_interval: int = 50

    # R4-A4-M7: Ternary Delta Chronicle
    use_delta_chronicle: bool = False
    delta_chronicle_max_mb: float = 500.0

    # R4-A4-M8: Flip Influence Decomposition
    use_flip_influence: bool = False
    flip_influence_top_k: int = 100

    # R4-A4-M4: Entropic Delta Encoding
    use_entropic_delta: bool = False

    # R4-A4-M3: Optimizer Reconstruction
    use_optimizer_reconstruction: bool = False
    optimizer_reconstruction_steps: int = 50

    # R4-A4-P3: Deterministic Trajectory Replay
    use_trajectory_replay: bool = False
    trajectory_anchor_interval: int = 5000

    # R4-A4-F9: Adaptive Checkpoint Interval
    use_adaptive_checkpoint: bool = False

    # R4-A4-P1: Integrate-and-Fire Optimizer
    use_iaf_optimizer: bool = False
    iaf_threshold: float = 1.0
    iaf_leak: float = 0.99

    # R4-A4-P2: Neural Checkpoint Codec
    use_neural_codec: bool = False

    # R4-A4-P4: GF(3) Algebraic Coding
    use_gf3_coding: bool = False

    # R4-A4-P6: Cross-Correlation State Recovery
    use_ccsr: bool = False

    # ================================================================
    # R4-A5: Portable Kernel Algorithms
    # ================================================================

    # Batch 2: Storage & Memory Layout
    use_quinary_packing: bool = False                       # M2: 5-in-8 byte packing (20% smaller)

    # Batch 3: Pattern LUT & SIMD
    use_block_pattern_lut: bool = False                     # M1: Amortized block pattern precomputation
    block_pattern_k: int = 4                                # Block size (4=81 patterns, 5=243)

    # Batch 4: Recursive Depth Optimization
    use_rtdp: bool = False                                  # P3: Recursive Delta Propagation (inference)
    rtdp_threshold: float = 0.01                            # Minimum activation change threshold
    rtdp_diagnostics: bool = False                          # Track per-iteration flip rates
    use_ternary_circuit_compile: bool = False               # P7: Circuit compilation (dead path elimination)
    circuit_compile_recompile_interval: int = 500           # Steps between recompilation during training
    use_adaptive_dataflow: bool = False                     # F1: AP/OP dataflow selection

    # Batch 5: SSM Scan Optimization (ALL require TSCS - ternary A matrix)
    use_scan_template: bool = False                         # M6: Cross-Layer Template Caching
    use_sparse_scan: bool = False                           # M5: Ternary Scan to Sparse Matmul
    use_parity_dual_stream: bool = False                    # M7: Parity-Partitioned Dual Stream
    use_absorbing_pruning: bool = False                     # M8: Absorbing State Tree Pruning

    # Batch 6: Deployment Targets
    use_ternary_binary_decomp: bool = False                 # F2: Ternary-to-binary decomposition
    ternary_binary_block_size: int = 4                      # Block size for binary decomposition

    # Batch 7: Advanced Explorations
    use_gf3_factorization: bool = False                     # M4: GF(3) rank factorization
    gf3_min_speedup: float = 1.5                            # Minimum speedup to apply GF(3) factorization
    use_butterfly_factorization: bool = False               # P1: Butterfly ternary factorization
    butterfly_target_layers: str = 'all'                    # Which layers to factorize: 'all', 'out_proj', 'in_proj'
    use_t3m: bool = False                                   # P2: Threshold-only TxT matmul
    t3m_activation_threshold: float = 0.5                   # Activation quantization threshold for T3M
    use_parallel_depth_scan: bool = False                   # P6: Parallel depth scan
    parallel_depth_warmup: int = 3                          # Warmup iterations before parallel solve
    parallel_depth_n_probes: int = 16                       # Random probes for Jacobian estimation

    # ---- R4-A6: Memory Layout & Tensor Format ----

    # Batch 1: Commitment Detection
    use_commitment_detector: bool = False                   # M1: Track weight commitment patterns
    commitment_threshold: float = 0.3                       # Fraction of steps before weight is "committed"
    metabolic_window: int = 100                             # Sliding window for metabolic analysis
    commitment_block_size: int = 128                        # Block size for commitment detection
    use_metabolic_probe: bool = False                       # P3: Metabolic signature analysis

    # Batch 2: Optimizer State Eviction
    use_graduated_eviction: bool = False                    # F1: Tiered eviction strategy
    eviction_tier_thresholds: str = "0.1,0.3,0.5"           # Commitment thresholds for tiers (comma-separated)
    use_param_eviction: bool = False                        # F14: Dynamic param groups eviction for committed weights
    eviction_commitment_threshold: float = 0.7              # Commitment ratio above which to evict (default 70%)
    eviction_check_interval: int = 100                      # Check every N steps for eviction candidates

    # Batch 3: Sparse Update & Drift Detection
    use_cusum_drift: bool = False                           # M5: CUSUM drift tracker
    cusum_threshold: float = 5.0                            # Sigma threshold for drift detection
    cusum_sensitivity: float = 0.5                          # Sensitivity of drift metric
    use_sparse_updates: bool = False                        # F1B: Sparse gradient updates
    sparse_update_committed_interval: int = 8               # Update non-committed weights every N steps

    # Batch 4: Compute Savings
    use_frozen_ternary_op: bool = False                     # F5A: Frozen weight operator caching
    frozen_cache_max_mb: int = 100                          # Max cache size for frozen computations
    use_recycling_pool: bool = False                        # M7: Memory recycling pool
    recycling_pool_max_mb: int = 256                        # Max recycling pool size
    use_selective_grad_subsample: bool = False              # P2: Selective gradient subsampling
    grad_subsample_committed: float = 0.1                   # Subsample ratio for committed weights

    # Batch 5: Training Paradigm Shifts
    use_bifurcated_optimizer: bool = False                  # P5: Separate optimizers for committed/uncommitted
    committed_lr_ratio: float = 0.1                         # LR ratio for committed weights (0.1 = 10% of main LR)
    use_commitment_curriculum: bool = False                 # P7: Curriculum that encourages commitment
    curriculum_target_ratio: float = 0.7                    # Target commitment ratio (default: 70%)
    curriculum_warmup_fraction: float = 0.2                 # Warmup fraction before curriculum activates
    use_thermal_memory: bool = False                        # P8: Thermal memory for weight dynamics

    # Batch 6: Explore
    use_ternary_layout_opt: bool = False                    # F4: Memory layout optimization
    layout_alignment: int = 64                              # Byte alignment for tensor layout
    use_predictive_freezer: bool = False                    # M3: Predictive weight freezing
    prediction_horizon: int = 500                           # Steps ahead to predict
    prediction_confidence: float = 0.9                      # Confidence threshold for freezing
    use_tensor_defrag: bool = False                         # F8: Tensor defragmentation
    defrag_interval: int = 1000                             # Steps between defragmentation

    # TIER 2 Batch 3: F24 Information-Theoretic Freezing
    use_info_freezing: bool = False                         # F24: Freeze converged weights
    info_freeze_threshold: float = 0.01                     # MI threshold for freezing
    info_freeze_warmup: int = 500                           # Warmup steps before freezing starts
    info_freeze_window: int = 100                           # Window for gradient variance tracking
    info_freeze_beta: float = 0.99                          # EMA decay for variance tracking

    # ================================================================
    # TIER 3 Phase 2: SSM Preservation Findings
    # ================================================================

    # R3-A1-M4: SSM Flip-Rate Bound
    use_flip_rate_limit: bool = False                       # Cap ternary weight flips per step in SSM
    ssm_max_flip_rate: float = 0.05                         # Max flip rate (5% of weights per step)

    # R3-A3-M5: Spectral-Adaptive Chunk Size
    use_spectral_adaptive_chunk: bool = False               # Adapt chunk size based on spectral radius
    chunk_epsilon: float = 1e-3                             # Target truncation error
    chunk_signal_bound: float = 10.0                        # Signal magnitude bound
    chunk_min: int = 8                                      # Minimum chunk size
    chunk_max: int = 256                                    # Maximum chunk size

    # R3-A3-M7: Krylov State Compression
    use_krylov_compression: bool = False                    # Compress SSM state via Krylov subspace
    krylov_rank: int = 16                                   # Compression rank (d_state -> k)

    # R3-A4-M3: Commitment K-Schedule
    use_commitment_k_schedule: bool = False                 # Adaptive TBPTT horizon K
    commitment_k_min: int = 10                              # Min K (high commitment, local refinement)
    commitment_k_max: int = 90                              # Max K (low commitment, exploration)
    commitment_stability_threshold: int = 50                # Steps without flip to be "committed"

    # R5-B2-C2: Commitment Tracking + Gradient Skip
    use_commitment_tracking: bool = False                   # C2: Per-weight commitment tracking
    commitment_n_stable: int = 100                          # Consecutive stable steps for commitment
    commitment_ema_decay: float = 0.99                      # σ_g EMA decay rate
    commitment_margin_multiplier: float = 4.89              # M_commit = multiplier * lr * sqrt(N) * σ_g
    use_gradient_skip: bool = False                         # C1: Skip gradients for committed weights
    grad_skip_g_max: float = 2.5                            # G_max safety bound for SSM
    commitment_log_interval: int = 100                      # Log commitment stats every N steps

    # R5-B2-C3: Progressive Layer Freezing
    use_progressive_freezing: bool = False                  # C3: Freeze committed layers
    freeze_threshold: float = 0.99                          # Commitment rate threshold for freezing
    freeze_check_interval: int = 100                        # Steps between freeze checks
    freeze_min_step: int = 500                              # Minimum step before freezing starts

    # R6-A5-M7: Commitment-Driven Precision (P2)
    use_commitment_precision: bool = False                  # P2: Reduce gradient precision based on commitment
    cdp_eviction_threshold: float = 0.9                     # Commitment above which to evict optimizer state
    cdp_freeze_threshold: float = 0.99                      # Commitment above which to freeze gradient
    cdp_min_bits: int = 4                                   # Minimum gradient precision (4-bit)
    cdp_max_bits: int = 8                                   # Maximum gradient precision (8-bit)
    cdp_log_interval: int = 100                             # Log precision stats every N steps

    # ================================================================
    # TIER 3 Phase 3: Advanced Quality (4 findings)
    # ================================================================

    # R3-A6-P5: Bifurcated Crystal-Liquid Optimizer
    use_bifurcated_optimizer: bool = False                  # P5: Split optimizers (SGD for crystal, Adam for liquid)
    bifurcated_base_lr: float = 6e-4                        # Base learning rate
    bifurcated_commitment_threshold: float = 0.01           # Threshold for commitment detection

    # R3-A6-M5: CUSUM Drift Detector
    use_cusum_drift_monitor: bool = False                   # M5: Detect uncommitting via drift
    cusum_threshold_factor: float = 3.0                     # Sigma multiplier for threshold
    cusum_alarm_level: float = 10.0                         # Alarm threshold

    # R3-A6-P3: Output Gradient Probe
    use_output_gradient_probe: bool = False                 # P3: Monitor layer health via dL/dy
    probe_ema_decay: float = 0.99                           # EMA decay for stats
    probe_var_alarm_low: float = 1e-6                       # Low variance alarm
    probe_var_alarm_high: float = 1e3                       # High variance alarm

    # R3-A3-M6: Asymmetric Chunking
    use_asymmetric_chunking: bool = False                   # M6: Different chunk sizes SSM vs SWA
    chunk_ssm_size: int = 512                               # SSM chunk size
    chunk_attn_size: int = 128                              # Attention chunk size

    # ================================================================
    # TIER 3 Phase 4: Exploratory (3 findings)
    # ================================================================

    # R3-A4-P2: Hybrid RTRL-TBPTT
    use_hybrid_rtrl: bool = False                           # P2: Exact RTRL for A_log, TBPTT for B,C,dt
    rtrl_k_truncate: int = 50                               # TBPTT truncation horizon

    # R3-A6-P6: Pattern Transplantation
    use_pattern_transplant: bool = False                    # P6: Copy ternary patterns from donors
    transplant_blend_ratio: float = 0.8                     # Blend: 80% donor, 20% random

    # R3-A6-P8: Commit-Restart
    use_commit_restart: bool = False                        # P8: Reinitialize stuck uncommitted weights
    restart_interval: int = 2000                            # Regular restart interval
    restart_stuck_threshold: int = 500                      # Steps before stuck-detection restart

    # ================================================================
    # E1: Two-Stage Flip-Triggered LR Schedule
    # ================================================================

    use_two_stage_flip_lr: bool = False                     # Enable flip-rate triggered two-stage LR schedule
    flip_lr_threshold: float = 0.05                         # Flip rate (EMA) threshold to trigger transition (5%)
    flip_lr_decay_factor: float = 0.1                       # LR multiplier in phase 2 (10% of phase 1)
    flip_lr_transition_steps: int = 100                     # Steps to decay from 1.0 to decay_factor (cosine)
    flip_lr_ema_beta: float = 0.9                           # EMA decay for flip rate smoothing
    flip_lr_min_step: int = 500                             # Minimum step before transition can occur

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    def __post_init__(self):
        if self.use_rwkv7 and self.use_ssd:
            raise ValueError("use_rwkv7 and use_ssd are mutually exclusive")
        if self.rwkv7_n_heads is None and self.use_rwkv7:
            self.rwkv7_n_heads = self.d_model // self.rwkv7_head_size
        if self.use_recursive_ssm and self.use_moe:
            raise ValueError("use_recursive_ssm and use_moe are mutually exclusive")
        if self.use_recursive_ssm and self.use_rwkv7:
            raise ValueError("use_recursive_ssm and use_rwkv7 are mutually exclusive")
        if self.use_recursive_ssm and not self.use_ssd:
            raise ValueError("use_recursive_ssm requires use_ssd=True")
        if self.use_ctri and self.use_recursive_ssm:
            raise ValueError("use_ctri and use_recursive_ssm are mutually exclusive")
        if self.use_ctri and not self.use_ssd:
            raise ValueError("use_ctri requires use_ssd=True")
        if self.use_ctri and getattr(self, 'use_eigenmode_mixed', False):
            raise ValueError("use_ctri and use_eigenmode_mixed are mutually exclusive")
        if self.use_tada and not self.use_recursive_ssm:
            raise ValueError("use_tada requires use_recursive_ssm=True (TADA extends RecursiveSSM)")
        if self.use_ucb_bandit and not self.use_recursive_ssm:
            raise ValueError("use_ucb_bandit requires use_recursive_ssm=True")
        if self.use_bbo and self.use_aga:
            raise ValueError("use_bbo and use_aga are mutually exclusive (both optimize ternary params)")
        if self.use_bbo and not self.use_ternary:
            raise ValueError("use_bbo requires use_ternary=True")
        if self.use_ftec:
            assert self.ftec_sparsity >= 2, "FTEC sparsity must be >= 2 (c=3 is optimal)"
            assert self.ftec_n_factors >= 2, "FTEC needs at least 2 factors"
            if self.ftec_pattern == 'butterfly':
                assert self.ftec_sparsity == 2, "Butterfly pattern requires sparsity=2"

        # R4-A5 validations
        if self.use_parity_dual_stream and not self.use_scan_template:
            raise ValueError("use_parity_dual_stream requires use_scan_template=True")
        if self.use_rtdp and not self.use_recursive_ssm:
            raise ValueError("use_rtdp requires use_recursive_ssm=True (RecursiveSSM only)")
        if self.use_parallel_depth_scan and not self.use_recursive_ssm:
            raise ValueError("use_parallel_depth_scan requires use_recursive_ssm=True")
        if self.use_t3m:
            import warnings
            warnings.warn("T3M requires ternary activations - quality impact unknown at 6M")

    def is_swa_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.swa_layer_indices


def config_1m() -> ModelConfig:
    return ModelConfig(
        d_model=256, n_layer=8, d_state=16, d_conv=4, expand=2,
        dt_rank=16, vocab_size=8192, seq_len=128,
        n_heads=4, window_size=128, swa_layer_indices=[4],
    )


def config_10m() -> ModelConfig:
    return ModelConfig(
        d_model=512, n_layer=16, d_state=16, d_conv=4, expand=2,
        dt_rank=32, vocab_size=16384, seq_len=1024,
        n_heads=8, window_size=128, swa_layer_indices=[5, 11],
    )


def config_100m() -> ModelConfig:
    return ModelConfig(
        d_model=768, n_layer=24, d_state=16, d_conv=4, expand=2,
        dt_rank=48, vocab_size=32000, seq_len=2048,
        n_heads=12, window_size=128, swa_layer_indices=[6, 12, 18],
    )


def config_scale_10m() -> ModelConfig:
    """~11.6M params for scale validation. seq_len=128 for SSD compatibility."""
    return ModelConfig(
        d_model=384, n_layer=8, d_state=16, d_conv=4, expand=2,
        dt_rank=24, vocab_size=8192, seq_len=128,
        n_heads=6, window_size=128, swa_layer_indices=[4],
    )


def config_scale_50m() -> ModelConfig:
    """~56.9M params for scale validation. seq_len=128 for SSD compatibility."""
    return ModelConfig(
        d_model=640, n_layer=18, d_state=16, d_conv=4, expand=2,
        dt_rank=40, vocab_size=8192, seq_len=128,
        n_heads=10, window_size=128, swa_layer_indices=[6, 12],
        use_torch_compile=True,
    )


def config_scale_100m() -> ModelConfig:
    """~112.5M params for scale validation. seq_len=128 for SSD compatibility."""
    return ModelConfig(
        d_model=768, n_layer=24, d_state=16, d_conv=4, expand=2,
        dt_rank=48, vocab_size=16384, seq_len=128,
        n_heads=12, window_size=128, swa_layer_indices=[6, 12, 18],
        use_torch_compile=True,
    )


def config_wikitext_6m() -> ModelConfig:
    """~6M params for WikiText-103. vocab=32000, seq_len=256."""
    return ModelConfig(
        d_model=256, n_layer=8, d_state=16, d_conv=4, expand=2,
        dt_rank=16, vocab_size=32000, seq_len=256,
        n_heads=4, window_size=128, swa_layer_indices=[4],
        use_ssd=True, ssd_n_heads=16,
    )


def config_wikitext_50m() -> ModelConfig:
    """~50M params for WikiText-103. vocab=32000, seq_len=256."""
    return ModelConfig(
        d_model=640, n_layer=18, d_state=16, d_conv=4, expand=2,
        dt_rank=40, vocab_size=32000, seq_len=256,
        n_heads=10, window_size=128, swa_layer_indices=[6, 12],
        use_ssd=True, ssd_n_heads=40,
        use_torch_compile=True,
        dropout=0.1,
    )


def config_wikitext_100m() -> ModelConfig:
    """~100M params for WikiText-103. vocab=32000, seq_len=256."""
    return ModelConfig(
        d_model=768, n_layer=24, d_state=16, d_conv=4, expand=2,
        dt_rank=48, vocab_size=32000, seq_len=256,
        n_heads=12, window_size=128, swa_layer_indices=[6, 12, 18],
        use_ssd=True, ssd_n_heads=48,
        use_torch_compile=True,
    )


def config_wikitext_bench() -> ModelConfig:
    """~6M params for fast A/B benchmarking. Same as wikitext_6m."""
    return ModelConfig(
        d_model=256, n_layer=8, d_state=16, d_conv=4, expand=2,
        dt_rank=16, vocab_size=32000, seq_len=256,
        n_heads=4, window_size=128, swa_layer_indices=[4],
    )


def config_ternary_6m() -> ModelConfig:
    """~6M params with ternary enabled. Same as wikitext_6m + use_ternary=True."""
    return ModelConfig(
        d_model=256, n_layer=8, d_state=16, d_conv=4, expand=2,
        dt_rank=16, vocab_size=32000, seq_len=256,
        n_heads=4, window_size=128, swa_layer_indices=[4],
        use_ssd=True, ssd_n_heads=16,
        use_ternary=True,
        ternary_lm_head=True,
    )


def config_ternary_50m() -> ModelConfig:
    """~50M params with ternary enabled. Same as wikitext_50m + use_ternary=True."""
    return ModelConfig(
        d_model=640, n_layer=18, d_state=64, d_conv=4, expand=2,
        dt_rank=40, vocab_size=32000, seq_len=256,
        n_heads=10, window_size=128, swa_layer_indices=[6, 12],
        use_ssd=True, ssd_n_heads=40,
        use_torch_compile=True,
        dropout=0.1,
        use_ternary=True,
        ternary_lm_head=True,
    )


def config_ternary_6m_moe() -> ModelConfig:
    """~6M base + ~3M expert params for MoTE testing."""
    return ModelConfig(
        d_model=256, n_layer=8, d_state=16, d_conv=4, expand=2,
        dt_rank=16, vocab_size=32000, seq_len=256,
        n_heads=4, window_size=128, swa_layer_indices=[4],
        use_ssd=True, ssd_n_heads=16,
        use_ternary=True, ternary_lm_head=True,
        use_moe=True, moe_num_experts=4, moe_top_k=1,
        moe_d_ff=256,
        moe_layer_indices=[1, 3, 5, 7],
        moe_load_balance_coeff=0.01,
    )
