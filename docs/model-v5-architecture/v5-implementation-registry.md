# v5 Implementation Registry — Techniques Already Implemented

This document lists ALL techniques, optimizations, and architectural features already present in `src/model_v5/`. Research agents MUST check this before reporting "discoveries" that we already have.

**Last updated:** 2026-03-01 (auto-generated from architecture-enforcer audit)

## Architecture
- HYBRID_SSM_SWA: ~87.5% SSM layers + ~12.5% Sliding Window Attention layers
- MAMBA1_SSM: Mamba-1 selective SSM with input-dependent B, C, delta
- MAMBA2_SSD: Mamba-2 SSD parallel scan with scalar A per head, segsum in log-space
- FLA_CHUNK_GLA: Optional Flash Linear Attention Triton kernel for parallel SSD scan
- FUSED_SEGSUM_EXP: Custom autograd fusing segsum + exp into single op
- PRE_NORM: Pre-norm (RMSNorm before each block)
- RMSNORM: Fused nn.RMSNorm with unfused fallback
- BITSHIFT_NORM: Power-of-2 approximation of RMSNorm via STE rounding
- ROPE: Rotary Position Embeddings on Q/K in SWA layers
- SWA_SLIDING_WINDOW: Configurable window size (default 128), causal masking
- SDPA: torch.nn.functional.scaled_dot_product_attention
- SWIGLU_FFN: 3-projection SwiGLU FFN in attention blocks
- CONV1D_LOCAL: Causal 1D convolution for local context in SSM blocks
- TIED_EMBEDDINGS: Shared input/output embeddings
- MOD_ROUTER: Mixture-of-Depths top-k expert-choice router
- MOE_TERNARY: Mixture of Ternary Experts (MoTE) with load-balancing loss
- RECURSIVE_SSM: Shared SSM block called N times with learned iteration embeddings
- RWKV7: RWKV-7 Goose block with data-dependent gating, LoRA gates, token shift
- EIGENMODE_MIXED: Split SSM heads into slow (FP16) and fast (ternary) sub-SSMs
- HEAD_GATING: Per-head sigmoid gating on attention output
- SINK_TOKENS: Keep first N tokens always visible in attention window
- MOA: Mixture of Attention Spans (per-head different window sizes)
- SLOW_HEADS: Slow/Medium/Fast SSM head bands by A_log initialization
- PHASE_SPACE_ATTN: Q/K derived from SSM B/C state-space
- CTRI: LTI layers with fixed B/C enabling FFT convolution training
- HIERARCHICAL_LM_HEAD: Two-pass coarse-then-fine vocabulary projection

## Quantization (STE Variants & Techniques)
- TERNARY_QUANTIZE_STE: Standard STE with per-group max scaling
- TERNARY_LINEAR: Drop-in nn.Linear replacement with shadow weights
- TERNARY_LM_HEAD: Ternary output projection
- GROUP_QUANTIZATION: Per-group quantization (default group_size=128)
- BISTE: Bilateral STE (separate paths for active/zero regions)
- HESTIA: Soft quantization with temperature tau
- HESTIA2026: Three-phase compress→anneal→solid
- QUEST: Trust-masked STE based on quantization error magnitude
- SIGMA_DELTA: Sequential noise shaping with accumulated error feedback
- BOUNDARY_AWARE: Gaussian modulation of backward gradient near threshold
- CTCE: Composite Threshold-Crossing Estimator (amplified near-threshold gradients)
- DUAL_RATE_STE: Amplify near-threshold + dampen far-from-threshold
- GASTE: Group-Aware STE correcting threshold coupling within groups
- BDT: Binary Decomposition Training (sign x activity decomposition)
- BAGR: Boundary-Attentive Gradient Reweighting with time-varying sigma
- ASGKS: Analytic Surrogate Gradient via Kernel Smoothing
- CAQ: Curvature-Aware Quantization using Adam v_hat
- TPSQ: Phase-Space Quantization with input-aware adaptive thresholds
- TCD: Ternary Critical Dynamics (learnable temperature for soft quantization)
- GUMBEL_STE: Gumbel noise injection before quantization
- TEQUILA: Deadzone reactivation regularization
- IAQD: Input-Adaptive Quantization Domain (per-feature sigmoid gating)
- STP: Stochastic Threshold Perturbation (implicit SAM)
- MDTS: Mirror Descent Ternary Solver (learnable logits per group)
- OPTROT: Optimal rotation before quantization (Cayley-parameterized)
- DEFERRED_TERNARY: Low-rank delta accumulation with periodic merge
- GRADUAL_THRESHOLD: Linear threshold coefficient interpolation
- PRESSURE_BLEND: Gradual FP→ternary transition
- PER_GROUP_ALPHA: Learnable per-group scaling factors
- LOWRANK_CORRECTION: Low-rank FP16 additive correction
- BLOCK_MIXING: Block-diagonal FP16 singular value restoration
- DENDRITIC_GATING: Pre/post multiplicative gates
- RLEC: Residual Low-Rank Error Compensation
- TSCS: Ternary State Collapse Scan (A matrix ternary, zero exp in scan)
- BTG: Backward-Ternary GEMV (add/sub backward semantics)
- STOCHASTIC_TERNARY: Probabilistic sampling for ensemble diversity

## Training Optimizations
- ADAMW: Standard decoupled weight decay
- ADAMW_BF16: BF16 optimizer states (~2x state reduction)
- ADAMW_8BIT: Block-wise INT8 optimizer states (~4x reduction)
- MUON: Momentum + Newton-Schulz orthogonalization for 2D weights
- GALORE: SVD-based low-rank gradient projection
- FIRA: SVD projection with norm-based scaling recovery
- AGA: INT16 sign accumulator with threshold-K triggering (2 bytes/weight)
- CPU_OFFLOAD_OPTIM: Pinned CPU memory for optimizer states, async transfers
- PROGRESSIVE_TRAINING: FP16→QAT→Routing three-phase scheduler
- GRADIENT_CHECKPOINTING: Configurable checkpoint_every_n layers
- SSM_CHECKPOINT: SSM-internal scan checkpointing
- AMP: Automatic Mixed Precision (FP16/BF16)
- BF16_SHADOW: BF16 shadow weights for memory savings
- TORCH_COMPILE: JIT compilation of model forward
- LIGER_CE: Fused cross-entropy from liger_kernel
- DISTILLATION: KL divergence knowledge distillation
- TSLD: Token-Scaled Logit Distillation
- RHO1: Selective Language Modeling (excess loss masking)
- SAQ: Step-Aligned Quantization (alternate FP16/STE steps)

## Inference Optimizations
- TRITON_TERNARY_GEMV: GPU add/sub/skip via 2-bit packed weights
- TRITON_FUSED_SSM: Fused SSM state update kernel
- TRITON_MEGA_KERNEL: Fused RMSNorm+proj+conv+SiLU+SSM+gate (5 launches)
- CUDA_FUSED_SSM: Custom CUDA kernel for SSM recurrence (NVRTC compiled)
- TSAR_CPU: CPU ternary GEMV via table lookup
- C_PORTABLE_GEMV: Portable C ternary GEMV with SIMD auto-detect
- WPH_GEMV: 81-pattern lookup table ternary GEMV
- POPCOUNT_GEMV: Activation binarization + popcount approximate matmul
- CUDA_GRAPH: Full CUDA graph capture for autoregressive generation
- STATEFUL_INFERENCE: Per-layer h/conv/KV-cache for single-token updates
- EARLY_EXIT: Cosine similarity layer skipping
- ENTROPY_SCREENING: L1-bound screening for vocabulary subset
- CIRCULAR_KV_CACHE: Fixed-size circular K/V buffer for SWA
- WEIGHT_PACKING: 2-bit ternary weight packing (4 weights/byte)

## Memory Optimizations
- GRADIENT_CHECKPOINTING_MEM: Activation recomputation
- SSM_SCAN_CHECKPOINT: Internal scan checkpointing
- TERNARY_WEIGHT_COMPRESSION: 2-bit weight storage (~0.2 bytes/weight)
- QWEIGHT_CACHE: Quantized weight caching across micro-batches

## Knowledge / Post-Training
- ATM_ADAPTER: Parallel ternary matrix overlay for knowledge injection
- ANTI_FORGETTING: Fisher-mask gradient projection
- DISCRETE_POLISH: Post-training ternary weight optimization
- STEWS_ENSEMBLE: Stochastic ternary ensemble inference
- MODEL_GROW: Smaller→larger checkpoint loading with noise
