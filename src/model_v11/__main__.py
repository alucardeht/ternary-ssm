"""CLI entry point for model training."""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from model_v11.config import (
    config_1m,
    config_10m,
    config_100m,
    config_scale_10m,
    config_wikitext_6m,
    config_wikitext_50m,
    config_wikitext_100m,
    config_wikitext_bench,
    config_ternary_6m,
    config_ternary_50m,
    config_ternary_6m_moe,
)
from model_v11.train import train_model


def main():
    parser = argparse.ArgumentParser(description="Train Model V7")

    # Config and data
    parser.add_argument(
        "--config",
        type=str,
        default="wikitext_6m",
        choices=[
            "1m",
            "10m",
            "100m",
            "scale_10m",
            "wikitext_6m",
            "wikitext_50m",
            "wikitext_100m",
            "wikitext_bench",
            "ternary_6m",
            "ternary_50m",
            "ternary_6m_moe",
        ],
        help="Model configuration",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="wikitext",
        choices=["synthetic", "copy", "wikitext"],
        help="Dataset type",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/wikitext103", help="Data directory"
    )

    # Training hyperparams
    parser.add_argument(
        "--max-steps", type=int, default=2000, help="Maximum training steps"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Sequence length (default: use config)",
    )
    parser.add_argument("--lr", type=float, default=6e-4, help="Base learning rate")
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=[
            "cosine",
            "cosine_restarts",
            "wsd",
            "moderate_decay",
            "inverse_sqrt",
            "two_stage",
        ],
        help="LR schedule",
    )
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument(
        "--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default 0.01, safe for ternary STE)",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="AdamW beta2 (default 0.999, safe for ternary STE)",
    )
    parser.add_argument(
        "--fused-optimizer",
        action="store_true",
        help="Enable fused AdamW (default: disabled for STE safety)",
    )

    # Device and optimization
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device"
    )
    parser.add_argument(
        "--use-amp", action="store_true", help="Use automatic mixed precision"
    )
    parser.add_argument(
        "--no-bf16",
        dest="use_bf16",
        action="store_false",
        default=True,
        help="Disable BF16 autocast (default: BF16 enabled on CUDA)",
    )
    parser.add_argument(
        "--no-compile",
        dest="use_compile",
        action="store_false",
        default=True,
        help="Disable torch.compile (default: compile enabled on CUDA)",
    )
    parser.add_argument(
        "--compile-memory-budget",
        type=float,
        default=None,
        help="F2: torch.compile activation memory budget fraction (0.0-1.0, default: None=disabled)",
    )
    parser.add_argument(
        "--activation-checkpoint-granularity",
        type=str,
        default="selective",
        choices=["none", "full", "selective"],
        help="F1: Activation checkpointing granularity (BATCH L: default selective)",
    )

    # Ternary and Rho-1
    parser.add_argument("--use-rho1", action="store_true", help="Use Rho-1 scoring")
    parser.add_argument(
        "--rho1-weighting",
        type=str,
        default="hard_mask",
        choices=["hard_mask", "relu_soft"],
        help="Rho-1 weighting mode",
    )
    parser.add_argument(
        "--rho1-keep-ratio", type=float, default=0.5, help="Rho-1 keep ratio"
    )
    parser.add_argument(
        "--rho1-ref-losses-path",
        type=str,
        default=None,
        help="Path to reference losses for Rho-1",
    )

    # Curriculum
    parser.add_argument(
        "--use-curriculum",
        action="store_true",
        help="Use curriculum learning (requires rho1-ref-losses-path)",
    )

    # F6: Sequence Packing
    parser.add_argument(
        "--use-sequence-packing",
        action="store_true",
        help="Enable F6 sequence packing (eliminates padding waste)",
    )

    # F3: CUDA-Stream-Async-Prefetch
    parser.add_argument(
        "--use-cuda-prefetch",
        action="store_true",
        help="Enable F3 CUDA-Stream async prefetch for data loading",
    )

    # M4: Ternary-Bandwidth-Saturation (auto batch size)
    parser.add_argument(
        "--auto-batch-size",
        action="store_true",
        help="Auto-compute batch_size from VRAM budget",
    )
    parser.add_argument(
        "--vram-budget-mb",
        type=int,
        default=10000,
        help="Target VRAM usage in MB for auto batch size (default: 10000)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/training",
        help="Output directory for results",
    )
    parser.add_argument("--log-every", type=int, default=50, help="Log every N steps")

    # Memory
    parser.add_argument(
        "--max-memory-mb", type=int, default=10000, help="Max memory usage in MB"
    )

    # Phase 28-B: Regularization
    parser.add_argument(
        "--angular-reg-lambda",
        type=float,
        default=0.0,
        help="Angular regularization lambda (Phase 28-B, e.g. 0.01)",
    )
    parser.add_argument(
        "--boundary-reg-lambda",
        type=float,
        default=0.0,
        help="Boundary regularization lambda (Phase 28-B, e.g. 0.001)",
    )

    # Phase 30-B: OptRot
    parser.add_argument(
        "--use-optrot",
        action="store_true",
        help="Enable OptRot rotation for ternary quantization",
    )
    parser.add_argument(
        "--optrot-lambda",
        type=float,
        default=0.01,
        help="L4 regularization weight for OptRot",
    )

    # Phase 28-C: GaLore optimizer
    parser.add_argument(
        "--use-galore",
        action="store_true",
        help="Use GaLore optimizer for memory-efficient training",
    )
    parser.add_argument("--galore-rank", type=int, default=128, help="GaLore rank")
    parser.add_argument(
        "--galore-update-proj-gap",
        type=int,
        default=200,
        help="GaLore projection update interval",
    )
    parser.add_argument(
        "--galore-scale", type=float, default=1.0, help="GaLore scale factor"
    )

    # Phase 30-C: MoTE (Mixture of Ternary Experts)
    parser.add_argument(
        "--use-moe", action="store_true", help="Enable Mixture of Ternary Experts"
    )
    parser.add_argument(
        "--moe-num-experts", type=int, default=4, help="Number of MoE experts"
    )
    parser.add_argument(
        "--moe-top-k", type=int, default=None, help="Number of experts per token"
    )
    parser.add_argument(
        "--moe-load-balance-coeff",
        type=float,
        default=None,
        help="MoE load balancing loss coefficient",
    )

    # Phase 35: Deferred Ternary (low-rank delta accumulation)
    parser.add_argument(
        "--use-deferred-ternary",
        action="store_true",
        help="Enable deferred ternary with low-rank delta accumulation",
    )
    parser.add_argument(
        "--deferred-rank",
        type=int,
        default=32,
        help="Rank for deferred ternary low-rank deltas",
    )
    parser.add_argument(
        "--deferred-merge-interval",
        type=int,
        default=100,
        help="Steps between deferred ternary delta merges",
    )

    # TSCS: Ternary State Collapse Scan
    parser.add_argument(
        "--use-tscs",
        action="store_true",
        help="Enable TSCS (Ternary State Collapse Scan)",
    )
    parser.add_argument(
        "--tscs-osc-fraction",
        type=float,
        default=0.375,
        help="Fraction of SSM heads as oscillators (A=-1) in TSCS",
    )
    parser.add_argument(
        "--tscs-reset-fraction",
        type=float,
        default=0.375,
        help="Fraction of SSM heads as resets (A=0) in TSCS",
    )
    parser.add_argument(
        "--tscs-clamp-state",
        type=float,
        default=0.0,
        help="Clamp state magnitude in TSCS (0=disabled)",
    )

    # BTG: Backward-Ternary GEMV
    parser.add_argument(
        "--use-btg",
        action="store_true",
        help="Enable BTG (Backward-Ternary GEMV) for ternary grad_input computation",
    )

    parser.add_argument("--use-tatgr", action="store_true",
                        help="TATGR: replace backward matmul with index-gather routing")

    parser.add_argument("--use-dsst", action="store_true",
                        help="Enable DSST (Decoupled Structure-Sign Training) for ternary weights")

    # AGA: Algebraic Gradient Accumulation
    parser.add_argument(
        "--use-aga",
        action="store_true",
        help="Enable AGA (Algebraic Gradient Accumulation) INT16 optimizer",
    )
    parser.add_argument(
        "--aga-threshold-k",
        type=int,
        default=32,
        help="AGA accumulator threshold K (default: 32)",
    )
    parser.add_argument(
        "--aga-decay-steps",
        type=int,
        default=0,
        help="AGA accumulator decay interval in steps (0=disabled)",
    )
    parser.add_argument(
        "--aga-ternary-pull",
        type=float,
        default=0.0,
        help="AGA pull toward ternary value (0.0=disabled, 0.1=weak)",
    )
    parser.add_argument(
        "--aga-lr-multiplier",
        type=float,
        default=1.0,
        help="AGA step size multiplier (default: 1.0 means lr=base_lr)",
    )
    parser.add_argument(
        "--use-fused-gradient-cascade",
        action="store_true",
        help="Enable fused gradient cascade (A2-M4): stream grads to INT16 accumulators via hooks",
    )

    # BBO: Ballot-Box Optimizer (A6-P5)
    parser.add_argument(
        "--use-bbo",
        action="store_true",
        help="Enable BBO (Ballot-Box Optimizer) for shadow-free ternary training",
    )
    parser.add_argument(
        "--bbo-threshold-k",
        type=int,
        default=16,
        help="BBO ballot threshold K (default: 16)",
    )
    parser.add_argument(
        "--bbo-decay-steps",
        type=int,
        default=100,
        help="BBO decay interval in steps (0=disabled, default: 100)",
    )
    parser.add_argument(
        "--bbo-magnitude-threshold",
        type=float,
        default=0.0,
        help="Minimum gradient magnitude to count as vote (default: 0.0)",
    )
    parser.add_argument(
        "--bbo-two-step",
        action="store_true",
        default=True,
        help="Require intermediate state for -1<->+1 transitions (default: enabled)",
    )

    # Q5: Context-as-Precision (Slow SSM Heads)
    parser.add_argument(
        "--use-slow-heads",
        action="store_true",
        help="Partition SSM head decay rates (fast/medium/slow)",
    )
    parser.add_argument(
        "--slow-head-fraction",
        type=float,
        default=0.25,
        help="Fraction of heads with slow decay (default 0.25)",
    )
    parser.add_argument(
        "--medium-head-fraction",
        type=float,
        default=0.25,
        help="Fraction of heads with medium decay (default 0.25)",
    )

    # A1: Mixture of Attention Spans
    parser.add_argument(
        "--use-moa",
        action="store_true",
        help="Use Mixture of Attention Spans (per-head window sizes)",
    )
    parser.add_argument(
        "--moa-window-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Window sizes for MoA (e.g. 32 64 128 256)",
    )

    # Q1: Curvature-Aware Quantization (CAQ)
    parser.add_argument(
        "--use-caq",
        action="store_true",
        help="Use Curvature-Aware Quantization (adaptive threshold from Adam v_hat)",
    )
    parser.add_argument(
        "--caq-alpha", type=float, default=1.0, help="CAQ sensitivity (default 1.0)"
    )

    # A2: Head Gating
    parser.add_argument(
        "--use-head-gating",
        action="store_true",
        help="Use per-head sigmoid gating on attention output",
    )

    # A3: Sink Tokens
    parser.add_argument(
        "--use-sink-tokens",
        action="store_true",
        help="Keep first N tokens always visible in attention window",
    )
    parser.add_argument(
        "--n-sink-tokens",
        type=int,
        default=1,
        help="Number of sink token positions (default: 1)",
    )

    # M1: SSD Gradient Checkpointing
    parser.add_argument(
        "--use-ssm-checkpoint",
        action="store_true",
        help="Enable SSM-internal scan gradient checkpointing (Finding M1)",
    )

    # A2-M7: Ternary-Aware Activation Checkpointing
    parser.add_argument(
        "--ternary-aware-checkpointing",
        action="store_true",
        help="Enable A2-M7 ternary-aware checkpointing (recompute SSM, store SWA)",
    )

    # A2-F3: Quamba-SE Activation Quantization
    parser.add_argument(
        "--no-activation-quant",
        action="store_false",
        dest="use_activation_quant",
        default=True,
        help="Disable A2-F3 Quamba-SE activation quantization",
    )
    parser.add_argument(
        "--activation-quant-bits",
        type=int,
        default=8,
        help="Bit-width for activation quantization (2 or 4, default: 8)",
    )

    # A2-F6: INSTANT Low-Rank Activation Compression
    parser.add_argument(
        "--use-instant",
        action="store_true",
        help="Enable A2-F6 INSTANT low-rank activation compression on SWA layers",
    )
    parser.add_argument(
        "--instant-rank",
        type=int,
        default=32,
        help="INSTANT projection rank (default: 32)",
    )

    # A5: Phase-Space Attention
    parser.add_argument(
        "--use-phase-space-attn",
        action="store_true",
        help="Use Phase-Space Attention (Q/K from SSM B/C state-space)",
    )

    # Q10: TPSQ
    parser.add_argument(
        "--use-tpsq",
        action="store_true",
        help="Use TPSQ (Phase-Space Quantization) for input-aware thresholds",
    )
    parser.add_argument(
        "--tpsq-beta",
        type=float,
        default=1.0,
        help="TPSQ sensitivity to input variance (default: 1.0)",
    )

    # Q17: TCD
    parser.add_argument(
        "--use-tcd",
        action="store_true",
        help="Use TCD (Ternary Critical Dynamics) learnable temperature",
    )

    # Q18: Gumbel+STE
    parser.add_argument(
        "--use-gumbel-noise",
        action="store_true",
        help="Enable Gumbel+STE noise injection during training",
    )
    parser.add_argument(
        "--gumbel-temperature",
        type=float,
        default=0.1,
        help="Gumbel noise temperature (default: 0.1)",
    )

    # Q9: Tequila (deadzone reactivation)
    parser.add_argument(
        "--tequila-lambda",
        type=float,
        default=0.0,
        help="Tequila deadzone reactivation lambda (0=disabled, 1e-3=default)",
    )

    # Q11: IAQD (Input-Adaptive Quantization Domain)
    parser.add_argument(
        "--use-iaqd",
        action="store_true",
        help="Enable IAQD input-adaptive gating on ternary layers (Q11)",
    )

    # STP: Stochastic Threshold Perturbation
    parser.add_argument(
        "--use-stp",
        action="store_true",
        help="Enable STP (Stochastic Threshold Perturbation)",
    )
    parser.add_argument(
        "--stp-sigma-ratio",
        type=float,
        default=0.1,
        help="STP noise std ratio (default: 0.1)",
    )

    # CTCE: Composite Threshold-Crossing Estimator
    parser.add_argument(
        "--use-ctce",
        action="store_true",
        help="Enable CTCE (gradient amplification near threshold)",
    )
    parser.add_argument(
        "--ctce-beta",
        type=float,
        default=1.5,
        help="CTCE amplification at threshold (default: 1.5)",
    )
    parser.add_argument(
        "--ctce-delta-ratio",
        type=float,
        default=0.5,
        help="CTCE zone width ratio (default: 0.5)",
    )

    # SAQ: Step-Aligned Quantization
    parser.add_argument(
        "--use-saq",
        action="store_true",
        help="Enable SAQ (alternate K FP16 + 1 STE steps)",
    )
    parser.add_argument(
        "--saq-k", type=int, default=3, help="SAQ FP16 steps per cycle (default: 3)"
    )

    # Dual-Rate STE
    parser.add_argument(
        "--use-dual-rate",
        action="store_true",
        help="Enable Dual-Rate STE (amplify+dampen)",
    )
    parser.add_argument(
        "--dual-rate-peak", type=float, default=2.5, help="Dual-Rate peak amplification"
    )
    parser.add_argument(
        "--dual-rate-floor", type=float, default=0.1, help="Dual-Rate floor damping"
    )

    # GASTE (Group-Aware STE)
    parser.add_argument(
        "--use-gaste", action="store_true", help="Enable GASTE (Group-Aware STE)"
    )
    parser.add_argument(
        "--gaste-sigma-g", type=float, default=0.3, help="GASTE boundary width"
    )

    # RLEC: Residual Low-Rank Error Compensation
    parser.add_argument(
        "--use-rlec",
        action="store_true",
        help="Enable RLEC low-rank error compensation",
    )
    parser.add_argument(
        "--rlec-rank", type=int, default=4, help="RLEC correction rank (default: 4)"
    )

    # AdaGC: Adaptive Gradient Clipping (R2-A3-F6)
    parser.add_argument(
        "--no-adagc",
        action="store_false",
        dest="use_adagc",
        default=True,
        help="Disable AdaGC (adaptive per-tensor gradient clipping)",
    )
    parser.add_argument(
        "--adagc-factor",
        type=float,
        default=1.5,
        help="AdaGC clipping factor (default: 1.5)",
    )
    parser.add_argument(
        "--adagc-beta",
        type=float,
        default=0.99,
        help="AdaGC EMA decay for gradient norm tracking (default: 0.99)",
    )

    # Margin-Sparse Backward + Temporal Gradient Recycling (R2-A1-M1+M4)
    parser.add_argument(
        "--use-margin-sparse",
        action="store_true",
        help="Enable Margin-Sparse Backward with gradient recycling",
    )
    parser.add_argument(
        "--margin-delta",
        type=float,
        default=0.1,
        help="Margin threshold: fraction of quantization threshold (default: 0.1)",
    )
    parser.add_argument(
        "--margin-ema-beta",
        type=float,
        default=0.99,
        help="Margin-Sparse EMA decay factor (default: 0.99)",
    )
    parser.add_argument(
        "--margin-refresh-interval",
        type=int,
        default=10,
        help="Full gradient refresh every K optimizer steps (default: 10)",
    )

    # A1-M5: Fused Backward-Update Operator
    parser.add_argument(
        "--use-fused-requantize",
        action="store_true",
        help="Enable fused optimizer.step + re-quantize (A1-M5, saves bandwidth)",
    )

    # A3-M3: Information-Theoretic Adaptive Freezing
    parser.add_argument(
        "--use-info-freeze",
        action="store_true",
        help="Enable Information-Theoretic Adaptive Freezing (R2-A3-M3)",
    )
    parser.add_argument(
        "--info-freeze-alpha",
        type=float,
        default=1e-5,
        help="SPRT significance level for info freeze (lower=more conservative, default: 1e-5)",
    )
    parser.add_argument(
        "--info-freeze-warmup",
        type=int,
        default=200,
        help="Warmup steps before info freeze begins (default: 200)",
    )

    # A3-P1: ACSI (Analytical Construction via System Identification)
    parser.add_argument(
        "--use-acsi-init",
        action="store_true",
        help="Enable ACSI (Analytical Construction via System Identification) for SSM init",
    )
    parser.add_argument(
        "--acsi-max-batches",
        type=int,
        default=4,
        help="Number of data batches for ACSI auto-correlation (default: 4)",
    )

    # A6-M14: Oscillation Dead Zone (STE Hysteresis)
    parser.add_argument(
        "--use-dead-zone",
        action="store_true",
        help="Enable Oscillation Dead Zone / STE Hysteresis (R2-A6-M14)",
    )
    parser.add_argument(
        "--dead-zone-margin",
        type=float,
        default=0.1,
        help="Fraction of threshold as hysteresis band (default: 0.1 = 10%%, range: 0.01-0.3)",
    )
    parser.add_argument(
        "--dead-zone-lr-decay",
        action="store_true",
        default=True,
        help="Scale hysteresis band by lr_ratio = current_lr/base_lr (default: True)",
    )

    # R2-A4-M3+M8: Ternary Commitment Monitor + Adaptive Training
    parser.add_argument(
        "--use-commitment-monitor",
        action="store_true",
        help="Enable ternary commitment monitoring + adaptive training (R2-A4-M3+M8)",
    )
    parser.add_argument(
        "--commitment-check-interval",
        type=int,
        default=50,
        help="Steps between commitment checks (default: 50)",
    )
    parser.add_argument(
        "--commitment-plateau-epsilon",
        type=float,
        default=0.005,
        help='Minimum |dH_T/dt| to consider "changing" (default: 0.005 bits/step)',
    )
    parser.add_argument(
        "--commitment-plateau-patience",
        type=int,
        default=3,
        help="Consecutive plateau checks before triggering action (default: 3)",
    )
    parser.add_argument(
        "--commitment-lr-boost",
        type=float,
        default=1.5,
        help="LR multiplier on plateau with high entropy (default: 1.5)",
    )
    parser.add_argument(
        "--commitment-lr-decay",
        type=float,
        default=0.5,
        help="LR multiplier on plateau with low entropy (default: 0.5)",
    )
    parser.add_argument(
        "--commitment-early-stop-threshold",
        type=float,
        default=0.98,
        help="Committed fraction threshold for early stop (default: 0.98)",
    )
    parser.add_argument(
        "--commitment-early-stop",
        action="store_true",
        default=True,
        help="Actually early-stop or just log (default: True, set to False for diagnostics-only)",
    )

    # CTRI: Calibrated Threshold Recalibration Interval
    parser.add_argument(
        "--use-ctri",
        action="store_true",
        help="Enable CTRI (Calibrated Threshold Recalibration Interval)",
    )
    parser.add_argument(
        "--ctri-selective-layers",
        type=str,
        default="0",
        help='Comma-separated layer indices for CTRI (e.g., "0,3,5" or "0" for all)',
    )

    # RecursiveSSM: shared SSM block with iterative depth
    parser.add_argument(
        "--use-recursive-ssm",
        action="store_true",
        help="Enable RecursiveSSM (shared SSM block called iteratively)",
    )
    parser.add_argument(
        "--recursive-ssm-iterations",
        type=int,
        default=0,
        help="Number of SSM iterations (0=auto: n_layer - n_swa_layers)",
    )

    # TADA: Ternary Algebraic Depth Amplification
    parser.add_argument(
        "--use-tada",
        action="store_true",
        help="Enable TADA (extends RecursiveSSM with per-iteration learned residual)",
    )
    parser.add_argument(
        "--tada-residual-injection",
        action="store_true",
        default=True,
        help="Enable per-iteration learned residual mixing (default: enabled)",
    )
    parser.add_argument(
        "--tada-no-residual-injection",
        dest="tada_residual_injection",
        action="store_false",
        help="Disable per-iteration residual injection",
    )
    parser.add_argument(
        "--tada-adaptive-k",
        action="store_true",
        help="Enable token-dependent iteration count (adaptive K during training)",
    )
    parser.add_argument(
        "--tada-adaptive-k-max",
        type=int,
        default=0,
        help="Max K for adaptive mode (0=use recursive_ssm_iterations)",
    )
    parser.add_argument(
        "--tada-rank-monitor-interval",
        type=int,
        default=0,
        help="Log effective rank of ternary matrices every N steps (0=disabled)",
    )

    # A1: Remaining findings (M5, M7, M8, P4, P8)
    parser.add_argument(
        "--use-episodic-gate",
        action="store_true",
        help="Enable P8 learned retention gate for episodic state",
    )
    parser.add_argument(
        "--use-adaptive-seq-len",
        action="store_true",
        help="Enable M7 adaptive sequence length based on spectral radius",
    )
    parser.add_argument(
        "--adaptive-seq-min",
        type=int,
        default=64,
        help="M7 minimum sequence length (default: 64)",
    )
    parser.add_argument(
        "--adaptive-seq-max",
        type=int,
        default=512,
        help="M7 maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--adaptive-seq-interval",
        type=int,
        default=100,
        help="M7 recompute interval in steps (default: 100)",
    )
    parser.add_argument(
        "--use-batch-geometry",
        action="store_true",
        help="Enable M8 batch size scaling with sequence length",
    )
    parser.add_argument(
        "--batch-geometry-base-batch",
        type=int,
        default=16,
        help="M8 reference batch size (default: 16)",
    )
    parser.add_argument(
        "--batch-geometry-base-seq",
        type=int,
        default=128,
        help="M8 reference sequence length (default: 128)",
    )
    parser.add_argument(
        "--use-streaming-training",
        action="store_true",
        help="Enable P4 stateful streaming training (N parallel streams)",
    )
    parser.add_argument(
        "--streaming-num-streams",
        type=int,
        default=4,
        help="P4 number of parallel streams (default: 4)",
    )

    # FPIT: Full-Pipeline Integer Training
    parser.add_argument(
        "--activation-fn",
        type=str,
        default="silu",
        choices=["silu", "relu"],
        help="Activation function (silu or relu)",
    )
    parser.add_argument(
        "--use-bitshift-norm",
        action="store_true",
        help="Use BitShiftNorm instead of RMSNorm",
    )

    # AdaptiveDepth: RecursiveSSM early exit (inference only)
    parser.add_argument(
        "--adaptive-depth-exit",
        action="store_true",
        help="Enable adaptive depth early exit for RecursiveSSM inference",
    )
    parser.add_argument(
        "--adaptive-depth-epsilon",
        type=float,
        default=0.01,
        help="Convergence threshold for adaptive depth (default: 0.01)",
    )
    parser.add_argument(
        "--adaptive-depth-min-iters",
        type=int,
        default=2,
        help="Minimum SSM iterations before early exit (default: 2)",
    )

    # Operational
    parser.add_argument(
        "--eval-every", type=int, default=None, help="Evaluate perplexity every N steps"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--max-nan-steps",
        type=int,
        default=0,
        help="Max consecutive NaN steps before abort (0=disabled)",
    )
    parser.add_argument(
        "--no-liger-ce", action="store_true", help="Disable Liger cross-entropy loss"
    )

    # R4-A3 Regularization: Label Smoothing, Grad Noise, Cosine Restarts, EMA
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing alpha (0.0=disabled, 0.1=recommended)",
    )
    parser.add_argument(
        "--use-grad-noise",
        action="store_true",
        help="Enable gradient noise injection during training",
    )
    parser.add_argument(
        "--grad-noise-eta",
        type=float,
        default=0.1,
        help="Gradient noise eta scale (default: 0.1)",
    )
    parser.add_argument(
        "--grad-noise-gamma",
        type=float,
        default=0.55,
        help="Gradient noise decay exponent (default: 0.55)",
    )
    parser.add_argument(
        "--cosine-restart-period",
        type=int,
        default=1000,
        help="Cosine restart period for LR schedule (default: 1000)",
    )
    parser.add_argument(
        "--cosine-restart-decay",
        type=float,
        default=0.8,
        help="Cosine restart peak decay factor (default: 0.8)",
    )
    parser.add_argument(
        "--no-ema", action="store_false", dest="use_ema", default=True,
        help="Disable EMA shadow weight tracking"
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay coefficient (default: 0.999)",
    )

    # A2-F6: Layerwise Gauss-Newton Preconditioning
    parser.add_argument(
        "--use-fisher-preconditioning",
        action="store_true",
        help="Enable A2-F6 Layerwise Gauss-Newton Preconditioning (Fisher proxy)",
    )
    parser.add_argument(
        "--fisher-damping",
        type=float,
        default=1e-4,
        help="A2-F6 Fisher preconditioning damping factor (default: 1e-4)",
    )

    # A2-F7: SignSGD for main training
    parser.add_argument(
        "--use-sign-sgd",
        action="store_true",
        help="Enable A2-F7 SignSGD with momentum for ternary weights",
    )
    parser.add_argument(
        "--sign-sgd-momentum",
        type=float,
        default=0.9,
        help="A2-F7 SignSGD momentum coefficient (default: 0.9)",
    )

    # A2-M7: Consensus-Filtered-Layer-Gradients
    parser.add_argument(
        "--use-consensus-gradient-filter",
        action="store_true",
        help="Enable A2-M7 Consensus-Filtered-Layer-Gradients (RecursiveSSM only)",
    )
    parser.add_argument(
        "--consensus-alpha",
        type=float,
        default=0.5,
        help="A2-M7 consensus filtering alpha (MAD clipping factor, default: 0.5)",
    )

    # A2-F8: Hessian-Aware Bit Allocation
    parser.add_argument(
        "--use-adaptive-quantization",
        action="store_true",
        help="Enable A2-F8 Hessian-Aware Adaptive Quantization (ternary/quinary)",
    )
    parser.add_argument(
        "--adaptive-quant-sensitivity-threshold",
        type=float,
        default=0.1,
        help="A2-F8 sensitivity threshold for 5-level quantization (default: 0.1)",
    )
    parser.add_argument(
        "--adaptive-quant-interval",
        type=int,
        default=500,
        help="A2-F8 recompute quantization levels every N steps (default: 500)",
    )

    # A2-M1: Recursive-Invertible-Checkpoint
    parser.add_argument(
        "--use-nonuniform-checkpointing",
        action="store_true",
        help="Enable A2-M1 non-uniform checkpoint density (dense early, sparse late layers)",
    )

    # A2-M3: VRAM-Pareto-Closed-Form Config Generator
    parser.add_argument(
        "--use-auto-config",
        action="store_true",
        help="Enable A2-M3 auto config: compute batch_size and grad_accum from VRAM budget",
    )

    # A2-M4: SQRT(s) SSM State Checkpoint Strategy
    parser.add_argument(
        "--use-sqrt-ssm-checkpointing",
        action="store_true",
        help="Enable A2-M4 sqrt(seq_len) SSM state checkpointing (26x activation reduction)",
    )

    # A2-M8: Information-Theoretic Training Budget (SNR Diagnostic)
    parser.add_argument(
        "--use-snr-diagnostic",
        action="store_true",
        help="Enable A2-M8 SNR diagnostic to identify bottleneck parameters",
    )
    parser.add_argument(
        "--snr-freeze-threshold",
        type=float,
        default=0.1,
        help="A2-M8 SNR threshold below which weights are frozen (default: 0.1)",
    )
    parser.add_argument(
        "--snr-freeze-interval",
        type=int,
        default=100,
        help="A2-M8 SNR computation interval in steps (default: 100)",
    )

    # A2-F4: Reversible-Layers-RevNet
    parser.add_argument(
        "--use-reversible-layers",
        action="store_true",
        help="Enable A2-F4 reversible layers for zero activation storage",
    )

    # A2-F10: Combinatorial-Submodular-Ternary-Optimization (post-training)
    parser.add_argument(
        "--use-combinatorial-finetune",
        action="store_true",
        help="Enable A2-F10 post-training combinatorial ternary optimization",
    )
    parser.add_argument(
        "--combinatorial-max-sweeps",
        type=int,
        default=3,
        help="A2-F10 maximum combinatorial optimization sweeps (default: 3)",
    )

    # Other options
    parser.add_argument(
        "--progressive", action="store_true", help="Use progressive training"
    )
    parser.add_argument(
        "--two-stage-decay", action="store_true", help="Use two-stage weight decay"
    )
    parser.add_argument(
        "--two-stage-drop-pct",
        type=float,
        default=0.5,
        help="Two-stage LR: fraction of training before LR drop (default: 0.5)",
    )
    parser.add_argument(
        "--two-stage-drop-factor",
        type=float,
        default=0.1,
        help="Two-stage LR: multiplier after drop (default: 0.1 = 10x reduction)",
    )
    parser.add_argument(
        "--two-stage-wd-start-pct",
        type=float,
        default=0.67,
        help="Two-stage WD: fraction when WD starts cosine-to-zero (default: 0.67)",
    )
    parser.add_argument(
        "--optimizer-type",
        type=str,
        default="lion",
        choices=[
            "adamw",
            "schedule_free",
            "galore",
            "muon",
            "ademamix",
            "spam",
            "fira",
            "lion",
        ],
        help="Optimizer type",
    )
    parser.add_argument(
        "--ste-type", type=str, default="standard", help="STE type for ternary"
    )
    parser.add_argument(
        "--ternary-lm-head",
        action="store_true",
        help="Use ternary quantization on LM head",
    )
    parser.add_argument("--seq-warmup", action="store_true", help="Use sequence warmup")
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of data loading workers"
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=None,
        help="Override dataloader num_workers (default: use config)",
    )
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=None,
        help="Dataloader prefetch factor per worker (default: use config)",
    )

    # A5-M2': AGS Asymmetric Gradient Sparsification
    parser.add_argument(
        "--use-ags",
        action="store_true",
        default=False,
        help="Enable R2-A5-M2' AGS gradient sparsification for CPU offload",
    )
    parser.add_argument(
        "--ags-top-k-frac",
        type=float,
        default=0.05,
        help="AGS: fraction of top-K gradients to always transfer (default: 0.05)",
    )
    parser.add_argument(
        "--ags-full-sync-interval",
        type=int,
        default=50,
        help="AGS: full gradient sync interval in steps (default: 50)",
    )
    parser.add_argument(
        "--ags-delta",
        type=float,
        default=0.1,
        help="AGS: near-threshold margin delta (default: 0.1)",
    )

    # A6-P3: FTEC (Factored Ternary Exponential Capacity)
    parser.add_argument(
        "--use-ftec",
        action="store_true",
        help="Enable FTEC (Factored Ternary Exponential Capacity)",
    )
    parser.add_argument(
        "--ftec-n-factors",
        type=int,
        default=8,
        help="FTEC factorization depth K (default: 8)",
    )
    parser.add_argument(
        "--ftec-sparsity",
        type=int,
        default=3,
        help="FTEC non-zeros per row per factor c (default: 3, optimal=3)",
    )
    parser.add_argument(
        "--ftec-pattern",
        type=str,
        default="random",
        choices=["random", "butterfly", "shift"],
        help="FTEC sparsity pattern (default: random)",
    )
    parser.add_argument(
        "--ftec-target-layers",
        type=str,
        default="out_proj",
        choices=["out_proj", "all_proj", "ssm_proj", "swa_proj"],
        help="Which projections to factorize (default: out_proj)",
    )
    parser.add_argument(
        "--ftec-no-skip",
        action="store_true",
        help="Disable FTEC skip connection (gradient flow safeguard)",
    )

    # Feature 2: Anderson Acceleration
    parser.add_argument(
        "--use-anderson-acceleration",
        action="store_true",
        help="Enable Anderson Acceleration for optimization",
    )
    parser.add_argument(
        "--anderson-beta",
        type=float,
        default=0.3,
        help="Anderson acceleration beta parameter (default: 0.3)",
    )

    # Feature 3: Geometric Extrapolation
    parser.add_argument(
        "--use-geometric-extrapolation",
        action="store_true",
        help="Enable geometric extrapolation for convergence",
    )
    parser.add_argument(
        "--geom-extrap-min-iters",
        type=int,
        default=4,
        help="Minimum iterations before geometric extrapolation (default: 4)",
    )
    parser.add_argument(
        "--geom-extrap-ratio-threshold",
        type=float,
        default=0.8,
        help="Ratio threshold for geometric extrapolation (default: 0.8)",
    )

    # ---- R4-A5: Portable Kernel Algorithms ----

    # R4-A5 Batch 2: Storage & Memory Layout
    parser.add_argument(
        "--use-quinary-packing",
        action="store_true",
        help="Use 5-in-8 quinary packing for ternary weights (20%% smaller)",
    )

    # R4-A5 Batch 3: Pattern LUT & SIMD
    parser.add_argument(
        "--use-block-pattern-lut",
        action="store_true",
        help="Use block pattern LUT for ternary matvec (1.9-2.4x speedup)",
    )
    parser.add_argument(
        "--block-pattern-k",
        type=int,
        default=4,
        help="Block size for pattern LUT (4=81 patterns, 5=243 patterns, default: 4)",
    )

    # R4-A5 Batch 4: Recursive Depth Optimization
    parser.add_argument(
        "--use-rtdp",
        action="store_true",
        help="Recursive Delta Propagation for RecursiveSSM (inference-only, 7-10x speedup)",
    )
    parser.add_argument(
        "--rtdp-threshold",
        type=float,
        default=0.01,
        help="RTDP: minimum activation change threshold (default: 0.01)",
    )
    parser.add_argument(
        "--rtdp-diagnostics",
        action="store_true",
        help="RTDP: enable per-iteration flip rate tracking",
    )
    parser.add_argument(
        "--use-ternary-circuit-compile",
        action="store_true",
        help="Enable ternary circuit compilation (15-30%% dead path elimination)",
    )
    parser.add_argument(
        "--circuit-recompile-interval",
        type=int,
        default=500,
        help="Steps between circuit recompilation during training (default: 500)",
    )
    parser.add_argument(
        "--use-adaptive-dataflow",
        action="store_true",
        help="Auto-select AP/OP dataflow per TernaryLinear layer",
    )

    # R4-A5 Batch 5: SSM Scan Optimization (BLOCKED by TSCS)
    parser.add_argument(
        "--use-scan-template",
        action="store_true",
        help="Use cross-layer template caching for SSM scan (requires TSCS)",
    )
    parser.add_argument(
        "--use-sparse-scan",
        action="store_true",
        help="Reformulate SSM scan as sparse matmul (requires TSCS)",
    )
    parser.add_argument(
        "--use-parity-dual-stream",
        action="store_true",
        help="Enable parity dual-stream scan optimization (requires scan-template)",
    )
    parser.add_argument(
        "--use-absorbing-pruning",
        action="store_true",
        help="Enable absorbing-state tree pruning for parallel scan (requires TSCS)",
    )

    # R4-A5 Batch 6: Deployment Targets
    parser.add_argument(
        "--use-ternary-binary-decomp",
        action="store_true",
        help="Use ternary-to-binary decomposition for matmul",
    )
    parser.add_argument(
        "--ternary-binary-block-size",
        type=int,
        default=4,
        help="Block size for binary decomposition (default: 4)",
    )

    # R4-A5 Batch 7: Advanced Explorations
    parser.add_argument(
        "--use-gf3-factorization",
        action="store_true",
        help="Enable GF(3) rank factorization of ternary matrices (inference optimization)",
    )
    parser.add_argument(
        "--gf3-min-speedup",
        type=float,
        default=1.5,
        help="Minimum speedup to apply GF(3) factorization (default: 1.5)",
    )
    parser.add_argument(
        "--use-butterfly-factorization",
        action="store_true",
        help="Use butterfly factorization for ternary layers (EXPERIMENTAL, high quality risk)",
    )
    parser.add_argument(
        "--butterfly-target-layers",
        type=str,
        default="all",
        choices=["all", "out_proj", "in_proj"],
        help="Which layers to factorize (default: all)",
    )
    parser.add_argument(
        "--use-t3m",
        action="store_true",
        help="Enable T3M threshold-only matmul (requires ternary activations, EXPERIMENTAL)",
    )
    parser.add_argument(
        "--t3m-activation-threshold",
        type=float,
        default=0.5,
        help="T3M: activation quantization threshold (default: 0.5)",
    )
    parser.add_argument(
        "--use-parallel-depth-scan",
        action="store_true",
        help="Enable parallel depth scan for RecursiveSSM (inference, 4x latency reduction)",
    )
    parser.add_argument(
        "--parallel-depth-warmup",
        type=int,
        default=3,
        help="Warmup iterations before parallel solve (default: 3)",
    )
    parser.add_argument(
        "--parallel-depth-n-probes",
        type=int,
        default=16,
        help="Random probes for Jacobian estimation (default: 16)",
    )

    # ---- R4-A6: Memory Layout & Tensor Format ----

    # Batch 1: Commitment Detection
    parser.add_argument(
        "--use-commitment-detector",
        action="store_true",
        help="Enable commitment detector for weight patterns",
    )
    parser.add_argument(
        "--commitment-threshold",
        type=float,
        default=0.3,
        help="Commitment threshold (default: 0.3)",
    )
    parser.add_argument(
        "--metabolic-window",
        type=int,
        default=100,
        help="Metabolic analysis window (default: 100)",
    )
    parser.add_argument(
        "--commitment-block-size",
        type=int,
        default=128,
        help="Commitment detection block size (default: 128)",
    )
    parser.add_argument(
        "--use-metabolic-probe",
        action="store_true",
        help="Enable metabolic probe analysis",
    )

    # Batch 2: Optimizer State Eviction
    parser.add_argument(
        "--use-graduated-eviction",
        action="store_true",
        help="Enable graduated eviction strategy",
    )
    parser.add_argument(
        "--eviction-tier-thresholds",
        type=str,
        default="0.1,0.3,0.5",
        help="Eviction tier thresholds (comma-separated, default: 0.1,0.3,0.5)",
    )

    # Batch 3: Sparse Update & Drift Detection
    parser.add_argument(
        "--use-cusum-drift", action="store_true", help="Enable CUSUM drift tracker"
    )
    parser.add_argument(
        "--cusum-threshold",
        type=float,
        default=5.0,
        help="CUSUM drift threshold (default: 5.0)",
    )
    parser.add_argument(
        "--cusum-sensitivity",
        type=float,
        default=0.5,
        help="CUSUM sensitivity (default: 0.5)",
    )
    parser.add_argument(
        "--use-sparse-updates",
        action="store_true",
        help="Enable sparse gradient updates",
    )
    parser.add_argument(
        "--sparse-update-committed-interval",
        type=int,
        default=8,
        help="Update interval for non-committed weights (default: 8)",
    )

    # Batch 4: Compute Savings
    parser.add_argument(
        "--use-frozen-ternary-op",
        action="store_true",
        help="Enable frozen weight operator caching",
    )
    parser.add_argument(
        "--frozen-cache-max-mb",
        type=int,
        default=100,
        help="Max frozen cache size in MB (default: 100)",
    )
    parser.add_argument(
        "--use-recycling-pool", action="store_true", help="Enable memory recycling pool"
    )
    parser.add_argument(
        "--recycling-pool-max-mb",
        type=int,
        default=256,
        help="Max recycling pool size in MB (default: 256)",
    )
    parser.add_argument(
        "--use-selective-grad-subsample",
        action="store_true",
        help="Enable selective gradient subsampling",
    )
    parser.add_argument(
        "--grad-subsample-committed",
        type=float,
        default=0.1,
        help="Subsample ratio for committed weights (default: 0.1)",
    )

    # Batch 5: Training Paradigm Shifts
    parser.add_argument(
        "--use-bifurcated-optimizer",
        action="store_true",
        help="Enable separate optimizers for committed/uncommitted weights",
    )
    parser.add_argument(
        "--committed-lr-ratio",
        type=float,
        default=0.1,
        help="LR ratio for committed weights (default: 0.1)",
    )
    parser.add_argument(
        "--use-commitment-curriculum",
        action="store_true",
        help="Enable commitment curriculum",
    )
    parser.add_argument(
        "--curriculum-target-ratio",
        type=float,
        default=0.7,
        help="Target commitment ratio (default: 0.7)",
    )
    parser.add_argument(
        "--curriculum-warmup-fraction",
        type=float,
        default=0.2,
        help="Warmup fraction for curriculum (default: 0.2)",
    )
    parser.add_argument(
        "--use-thermal-memory",
        action="store_true",
        help="Enable thermal memory for weight dynamics",
    )

    # Batch 6: Explore
    parser.add_argument(
        "--use-ternary-layout-opt",
        action="store_true",
        help="Enable memory layout optimization",
    )
    parser.add_argument(
        "--layout-alignment",
        type=int,
        default=64,
        help="Byte alignment for tensor layout (default: 64)",
    )
    parser.add_argument(
        "--use-predictive-freezer",
        action="store_true",
        help="Enable predictive weight freezing",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=500,
        help="Prediction horizon in steps (default: 500)",
    )
    parser.add_argument(
        "--prediction-confidence",
        type=float,
        default=0.9,
        help="Confidence threshold for freezing (default: 0.9)",
    )
    parser.add_argument(
        "--use-tensor-defrag", action="store_true", help="Enable tensor defragmentation"
    )
    parser.add_argument(
        "--defrag-interval",
        type=int,
        default=1000,
        help="Defragmentation interval (default: 1000)",
    )

    # Feature 4: Episodic State
    parser.add_argument(
        "--use-episodic-state",
        action="store_true",
        help="Enable episodic state snapshots in SSM",
    )

    # Feature 5: Contractive Regularization
    parser.add_argument(
        "--contractive-reg-lambda",
        type=float,
        default=0.0,
        help="Contractive regularization lambda (0.0=disabled)",
    )

    # R4-A3-F3: LET (Late-to-Early Training / Layer Distillation)
    parser.add_argument(
        "--let-teacher-path",
        type=str,
        default=None,
        help="Path to frozen teacher checkpoint for LET distillation",
    )
    parser.add_argument(
        "--let-lambda",
        type=float,
        default=0.1,
        help="LET auxiliary loss weight (default: 0.1)",
    )
    parser.add_argument(
        "--let-stop-step",
        type=int,
        default=1500,
        help="Step at which LET loss fades out (default: 1500)",
    )
    parser.add_argument(
        "--let-layer-k",
        type=int,
        default=3,
        help="Student layer index for hidden state comparison (default: 3)",
    )

    # R4-A3-F6: RecursiveSSM Iteration Load-Balancing
    parser.add_argument(
        "--use-recursive-load-balance",
        action="store_true",
        help="Enable iteration load-balancing for RecursiveSSM",
    )
    parser.add_argument(
        "--recursive-load-balance-coeff",
        type=float,
        default=0.01,
        help="Load-balancing coefficient (default: 0.01)",
    )

    # Feature 6: Per-Depth LR
    parser.add_argument(
        "--use-per-depth-lr",
        action="store_true",
        help="Enable per-depth learning rate scheduling",
    )
    parser.add_argument(
        "--per-depth-lr-min",
        type=float,
        default=0.3,
        help="Per-depth LR minimum multiplier (default: 0.3)",
    )

    # R4-A3-F5: Uncertainty-Weighted Multi-Task Loss (Kendall et al.)
    parser.add_argument(
        "--use-uncertainty-loss",
        action="store_true",
        help="Enable Kendall uncertainty-weighted multi-task loss balancing",
    )
    parser.add_argument(
        "--uncertainty-n-tasks",
        type=int,
        default=4,
        help="Number of task slots for uncertainty weighting (default: 4)",
    )
    parser.add_argument(
        "--uncertainty-initial-logvar",
        type=float,
        default=0.0,
        help="Initial log-variance for uncertainty weights (default: 0.0)",
    )

    # R4-A3-F4: Self-Knowledge Distillation (SSKD)
    parser.add_argument(
        "--use-sskd",
        action="store_true",
        help="Enable self-knowledge distillation (shadow vs quantized)",
    )
    parser.add_argument(
        "--sskd-alpha", type=float, default=0.1, help="SSKD loss weight (default: 0.1)"
    )
    parser.add_argument(
        "--sskd-temperature",
        type=float,
        default=2.0,
        help="SSKD softmax temperature (default: 2.0)",
    )

    # R4-A3-F8: Dynamic Loss Weighting
    parser.add_argument(
        "--use-dynamic-loss-weighting",
        action="store_true",
        help="Enable dynamic loss weighting (geometric mean balancing)",
    )
    parser.add_argument(
        "--dlw-gamma",
        type=float,
        default=1.0,
        help="DLW balance exponent (default: 1.0)",
    )

    # R4-A3-M1: Ternary Hamming Margin Loss
    parser.add_argument(
        "--use-thm-loss",
        action="store_true",
        help="Enable Hamming margin regularization on ternary weights",
    )
    parser.add_argument(
        "--thm-margin",
        type=float,
        default=0.3,
        help="THM target Hamming distance ratio (default: 0.3)",
    )
    parser.add_argument(
        "--thm-lambda", type=float, default=0.01, help="THM loss weight (default: 0.01)"
    )

    # R4-A3-M2: Ternary-Aligned Integer Loss
    parser.add_argument(
        "--use-tail-loss",
        action="store_true",
        help="Enable TAIL flip-distance hinge loss",
    )
    parser.add_argument(
        "--tail-lambda",
        type=float,
        default=0.01,
        help="TAIL loss weight (default: 0.01)",
    )
    parser.add_argument(
        "--tail-flip-margin",
        type=float,
        default=1.0,
        help="TAIL minimum logit margin per flip (default: 1.0)",
    )
    parser.add_argument(
        "--tail-num-negatives",
        type=int,
        default=5,
        help="TAIL number of hardest negatives (default: 5)",
    )

    # R4-A3-M3: Recursive Consensus Loss
    parser.add_argument(
        "--use-rcl-loss",
        action="store_true",
        help="Enable recursive consensus loss (per-iteration convergence weighting)",
    )
    parser.add_argument(
        "--rcl-beta",
        type=float,
        default=1.0,
        help="RCL convergence exponent (default: 1.0)",
    )
    parser.add_argument(
        "--rcl-lambda", type=float, default=0.05, help="RCL loss weight (default: 0.05)"
    )
    parser.add_argument(
        "--rcl-sample-every",
        type=int,
        default=4,
        help="RCL sample every N iterations (default: 4)",
    )

    # R4-A3-M4: Ternary Entropy Rate Loss
    parser.add_argument(
        "--use-ter-loss",
        action="store_true",
        help="Enable ternary entropy rate regularizer",
    )
    parser.add_argument(
        "--ter-loss-mu",
        type=float,
        default=0.1,
        help="TER-LOSS penalty weight (default: 0.1)",
    )
    parser.add_argument(
        "--ter-noise-var",
        type=float,
        default=0.01,
        help="TER-LOSS noise variance estimate (default: 0.01)",
    )

    # R4-A3-M5: Gradient Consensus Polarization
    parser.add_argument(
        "--use-gcp",
        action="store_true",
        help="Enable gradient consensus polarization for shared weights",
    )
    parser.add_argument(
        "--gcp-alpha",
        type=float,
        default=1.0,
        help="GCP polarization strength (default: 1.0)",
    )

    # R4-A3-M6: Recurrence-Aware Flip Utility
    parser.add_argument(
        "--use-rafu", action="store_true", help="Enable RAFU flip utility filtering"
    )
    parser.add_argument(
        "--rafu-threshold",
        type=float,
        default=0.1,
        help="RAFU utility threshold for gradient filtering (default: 0.1)",
    )
    parser.add_argument(
        "--rafu-interval",
        type=int,
        default=50,
        help="RAFU recalculation interval in steps (default: 50)",
    )

    # Feature 8: Sparse Shadow
    parser.add_argument(
        "--use-sparse-shadow",
        action="store_true",
        help="Enable sparse shadow weight accumulation",
    )
    parser.add_argument(
        "--sparse-shadow-margin",
        type=float,
        default=0.1,
        help="Sparse shadow margin threshold (default: 0.1)",
    )
    parser.add_argument(
        "--sparse-shadow-decay",
        type=float,
        default=0.0,
        help="Sparse shadow decay rate (default: 0.0)",
    )
    parser.add_argument(
        "--sparse-shadow-interval",
        type=int,
        default=10,
        help="Sparse shadow update interval in steps (default: 10)",
    )

    # F6: SCA-Sparsity-Controlled-Energy (Deng et al., arXiv:2011.00580)
    parser.add_argument(
        "--use-sca-sparsity",
        action="store_true",
        help="Enable SCA sparsity regularization for explicit sparsity control",
    )
    parser.add_argument(
        "--sca-lambda",
        type=float,
        default=0.01,
        help="SCA sparsity regularization weight (default: 0.01)",
    )
    parser.add_argument(
        "--sca-target-sparsity",
        type=float,
        default=0.5,
        help="SCA target zero ratio (default: 0.5)",
    )
    parser.add_argument(
        "--sca-warmup-steps",
        type=int,
        default=0,
        help="SCA warmup steps before activation (default: 0)",
    )

    # Feature 9: UCB-Bandit
    parser.add_argument(
        "--use-ucb-bandit",
        action="store_true",
        help="Enable UCB-Bandit optimizer selection",
    )
    parser.add_argument(
        "--ucb-min-iterations",
        type=int,
        default=4,
        help="Minimum iterations per arm before exploration (default: 4)",
    )

    # Feature 10: FP8 Grad Accumulation
    parser.add_argument(
        "--use-fp8-grad-accum",
        action="store_true",
        help="Enable FP8 gradient accumulation",
    )
    # F9: Top-K Gradient Sparsification
    parser.add_argument(
        "--use-topk-grad-sparsification",
        action="store_true",
        help="Enable F9 top-K gradient sparsification with error feedback",
    )
    parser.add_argument(
        "--topk-grad-sparsity",
        type=float,
        default=0.9,
        help="F9: Fraction of gradients to zero out (default: 0.9 = keep top 10%%)",
    )

    # F7: Multi-Token Prediction + M6 Ternary-Capacity-Bounded-MTP
    parser.add_argument(
        "--mtp-num-heads",
        type=int,
        default=0,
        help="Number of auxiliary MTP heads (0=disabled, 2=M6 default)",
    )
    parser.add_argument(
        "--mtp-lambda-main",
        type=float,
        default=0.5,
        help="Weight for main (next-token) loss in MTP (default: 0.5)",
    )
    parser.add_argument(
        "--mtp-lambda-aux",
        type=float,
        nargs="+",
        default=None,
        help="Weights for auxiliary heads (default: geometric decay [0.15, 0.05, ...])",
    )

    # P5: Ternary Channel-Aligned Loss
    parser.add_argument(
        "--use-gradient-sign-consistency",
        action="store_true",
        help="Enable P5 gradient sign consistency filter for ternary training",
    )
    parser.add_argument(
        "--grad-sign-window",
        type=int,
        default=8,
        help="Number of micro-batches to track gradient sign (default: 8)",
    )
    parser.add_argument(
        "--grad-sign-threshold",
        type=int,
        default=6,
        help="Minimum consistent signs required to update (default: 6 of 8)",
    )

    # A2-M5: UCB-Bandit-Ternary-Training (Eliminate STE, direct ternary optimization)
    parser.add_argument(
        "--use-ucb-ternary-training",
        action="store_true",
        help="Enable A2-M5 UCB-Bandit Ternary Training (eliminates STE)",
    )
    parser.add_argument(
        "--ucb-ternary-exploration-c",
        type=float,
        default=1.0,
        help="A2-M5 UCB exploration coefficient (default: 1.0)",
    )

    # A2-M6: Activation-Gated-Flip-Criterion (FSCR)
    parser.add_argument(
        "--use-fscr-flip-gate",
        action="store_true",
        help="Enable A2-M6 FSCR (Flip Safety/Confidence Ratio) to prevent outlier-driven flips",
    )
    parser.add_argument(
        "--fscr-threshold",
        type=float,
        default=2.0,
        help="A2-M6 FSCR flip safety threshold (higher=more conservative, default: 2.0)",
    )

    # P1: Spectral-Cached SSM Backward
    parser.add_argument(
        "--use-spectral-ssm",
        action="store_true",
        help="Enable P1 Spectral-Cached SSM for zero-activation backward",
    )

    # P2: SPSA Ternary Concurrent Flip Estimation
    parser.add_argument(
        "--use-spsa-training",
        action="store_true",
        help="Enable P2 SPSA ternary training (no backward pass)",
    )
    parser.add_argument(
        "--spsa-perturbation-scale",
        type=float,
        default=0.1,
        help="P2 SPSA perturbation magnitude (default: 0.1)",
    )
    parser.add_argument(
        "--spsa-lr",
        type=float,
        default=0.01,
        help="P2 SPSA learning rate (default: 0.01)",
    )

    # P3: Integer-Rank Loss Training
    parser.add_argument(
        "--use-rank-margin-loss",
        action="store_true",
        help="Enable P3 rank-margin loss (fully integer compatible)",
    )
    parser.add_argument(
        "--rank-margin", type=float, default=1.0, help="P3 margin value (default: 1.0)"
    )
    parser.add_argument(
        "--rank-num-negatives",
        type=int,
        default=64,
        help="P3 number of negative samples (default: 64)",
    )

    # P5: Whittle-Index Asynchronous Training
    parser.add_argument(
        "--use-whittle-index",
        action="store_true",
        help="Enable P5 Whittle-Index priority scheduling (5-10x FLOPS reduction)",
    )
    parser.add_argument(
        "--whittle-update-fraction",
        type=float,
        default=0.1,
        help="P5 fraction of weights to update per step (default: 0.1 = top 10%%)",
    )

    # R4-A3-M7: Information-Bottleneck Layerwise Loss
    parser.add_argument(
        "--use-ibll",
        action="store_true",
        help="Enable IB layerwise losses for intermediate supervision",
    )
    parser.add_argument("--ibll-beta", type=float, default=1.0)
    parser.add_argument("--ibll-gamma", type=float, default=0.1)
    parser.add_argument("--ibll-eps", type=float, default=0.01)
    parser.add_argument("--ibll-lambda", type=float, default=0.02)
    parser.add_argument("--ibll-sample-every", type=int, default=8)

    # R4-A3-M8: Bayesian Flip Accumulator
    parser.add_argument(
        "--use-bfa",
        action="store_true",
        help="Enable Bayesian flip accumulator (alternative to STE)",
    )
    parser.add_argument("--bfa-rho", type=float, default=0.95)
    parser.add_argument("--bfa-tau", type=float, default=1.0)
    parser.add_argument("--bfa-theta", type=float, default=0.9)
    parser.add_argument("--bfa-interval", type=int, default=10)

    # R4-A3-P1: Minimax Depth Equilibrium
    parser.add_argument(
        "--use-mde",
        action="store_true",
        help="Enable minimax depth equilibrium (adversarial depth weighting)",
    )
    parser.add_argument("--mde-eta", type=float, default=0.1)
    parser.add_argument("--mde-lambda", type=float, default=0.05)
    parser.add_argument("--mde-sample-every", type=int, default=4)

    # R4-A3-P2: Ternary Error-Correcting Weight Codes
    parser.add_argument(
        "--use-tecwc",
        action="store_true",
        help="Enable ECC regularization on ternary weight codes",
    )
    parser.add_argument("--tecwc-code-rate", type=float, default=0.9)
    parser.add_argument("--tecwc-lambda", type=float, default=0.001)

    # R4-A3-P3: Spectral Surrogate Objective
    parser.add_argument(
        "--use-sso",
        action="store_true",
        help="Enable spectral surrogate objective for SSM matrices",
    )
    parser.add_argument("--sso-rho-target", type=float, default=0.95)
    parser.add_argument("--sso-alpha", type=float, default=1.0)
    parser.add_argument("--sso-beta", type=float, default=0.1)
    parser.add_argument("--sso-interval", type=int, default=100)

    # R4-A3-P4: Spin-Glass Annealing
    parser.add_argument(
        "--use-sgapd",
        action="store_true",
        help="Enable spin-glass MC annealing for ternary weights",
    )
    parser.add_argument("--sgapd-initial-temp", type=float, default=1.0)
    parser.add_argument("--sgapd-min-temp", type=float, default=0.01)
    parser.add_argument("--sgapd-decay", type=float, default=0.995)
    parser.add_argument("--sgapd-sweeps", type=int, default=10)
    parser.add_argument("--sgapd-interval", type=int, default=20)

    # R4-A3-P5: Ergodic Trajectory Optimizer
    parser.add_argument(
        "--use-ergodic",
        action="store_true",
        help="Enable ergodic trajectory optimizer (alternative to Adam)",
    )
    parser.add_argument("--ergodic-alpha", type=float, default=1.0)
    parser.add_argument("--ergodic-rho-target", type=float, default=0.9)
    parser.add_argument("--ergodic-flip-fraction", type=float, default=0.01)
    parser.add_argument("--ergodic-interval", type=int, default=50)

    # R4-A3-P6: Minimax Depth Fairness
    parser.add_argument(
        "--use-minimax-depth",
        action="store_true",
        help="Enable minimax depth fairness (Rawlsian)",
    )
    parser.add_argument("--minimax-tau", type=float, default=1.0)
    parser.add_argument("--minimax-hard", action="store_true")
    parser.add_argument("--minimax-lambda", type=float, default=0.05)
    parser.add_argument("--minimax-sample-every", type=int, default=4)

    # R4-A3-P7: Koopman Spectral Shaping
    parser.add_argument(
        "--use-koopman",
        action="store_true",
        help="Enable Koopman spectral shaping of RecursiveSSM dynamics",
    )
    parser.add_argument("--koopman-long-target", type=float, default=0.97)
    parser.add_argument("--koopman-fast-target", type=float, default=0.5)
    parser.add_argument("--koopman-lambda", type=float, default=0.01)
    parser.add_argument("--koopman-interval", type=int, default=100)
    parser.add_argument("--koopman-max-dim", type=int, default=64)

    # R4-A3-P8: TD Depth Learning
    parser.add_argument(
        "--use-td-dl",
        action="store_true",
        help="Enable temporal difference depth learning",
    )
    parser.add_argument(
        "--td-lambda-param", type=float, default=0.8, help="TD(lambda) trace decay"
    )
    parser.add_argument("--td-dl-weight", type=float, default=0.05)
    parser.add_argument("--td-sample-every", type=int, default=2)

    # ============================================================
    # R4-A4: Checkpoint, Resume & Fault Tolerance
    # ============================================================

    # R4-A4-F10: Gradient Health Metrics
    parser.add_argument(
        "--use-gradient-monitor",
        action="store_true",
        help="Enable gradient health monitoring",
    )
    parser.add_argument(
        "--gradient-monitor-window",
        type=int,
        default=100,
        help="Rolling window for anomaly detection (default: 100)",
    )

    # R4-A4-F7: RAM Checkpointing
    parser.add_argument(
        "--use-ram-checkpoint", action="store_true", help="Enable fast RAM checkpoint"
    )
    parser.add_argument(
        "--no-ram-checkpoint-disk-backup",
        action="store_true",
        help="Disable async disk backup for RAM checkpoints",
    )

    # R4-A4-F8: Deterministic Mode
    parser.add_argument(
        "--use-deterministic-mode",
        action="store_true",
        help="Enable torch deterministic algorithms",
    )

    # R4-A4-F1: Ternary Checkpoint Compression
    parser.add_argument(
        "--use-ternary-checkpoint-compression",
        action="store_true",
        help="Enable bitmask compression for ternary checkpoints",
    )

    # R4-A4-P8: Contractive Recurrence Guarantee
    parser.add_argument(
        "--use-crg", action="store_true", help="Enable CRG alpha_decay on A matrix"
    )
    parser.add_argument(
        "--crg-alpha-init",
        type=float,
        default=3.0,
        help="CRG initial alpha (sigmoid(3.0)~0.95)",
    )

    # R4-A4-F3: Incremental Delta Checkpoints
    parser.add_argument(
        "--use-incremental-checkpoints",
        action="store_true",
        help="Enable incremental delta checkpoints for ternary weights",
    )
    parser.add_argument(
        "--delta-full-interval",
        type=int,
        default=1000,
        help="Full checkpoint interval for incremental deltas (default: 1000)",
    )

    # R4-A4-M5: Tiered Checkpoint Strategy
    parser.add_argument(
        "--use-tiered-checkpoint",
        action="store_true",
        help="Enable three-tier checkpoint strategy",
    )
    parser.add_argument(
        "--tiered-tier2-interval",
        type=int,
        default=50,
        help="Tier 2 (shadow subset) interval (default: 50)",
    )
    parser.add_argument(
        "--tiered-tier3-interval",
        type=int,
        default=500,
        help="Tier 3 (full state) interval (default: 500)",
    )

    # R4-A4-F5: Optimizer State Compression
    parser.add_argument(
        "--use-optimizer-compression",
        action="store_true",
        help="Enable k-means optimizer state compression in checkpoints",
    )

    # R4-A4-M1: Fiber Bundle Shadow Compression
    parser.add_argument(
        "--use-fiber-shadow-compression",
        action="store_true",
        help="Enable fiber bundle variable-rate shadow compression",
    )

    # R4-A4-M2: TDSS Optimizer Compression
    parser.add_argument(
        "--use-tdss-compression",
        action="store_true",
        help="Enable TDSS flip-pressure optimizer compression",
    )

    # R4-A4-F6: Anomaly Detection
    parser.add_argument(
        "--use-anomaly-detector",
        action="store_true",
        help="Enable TRANSOM-TEE anomaly detection with auto-rollback",
    )
    parser.add_argument(
        "--anomaly-z-threshold",
        type=float,
        default=3.0,
        help="Z-score threshold for anomaly detection (default: 3.0)",
    )

    # R4-A4-F2: Async Checkpoint
    parser.add_argument(
        "--use-async-checkpoint",
        action="store_true",
        help="Enable process-based async checkpoint saving",
    )

    # R4-A4-P5: QMM Regularizer
    parser.add_argument(
        "--use-qmm-regularizer",
        action="store_true",
        help="Enable quantization margin maximization regularizer",
    )
    parser.add_argument(
        "--qmm-lambda",
        type=float,
        default=0.01,
        help="QMM regularization strength (default: 0.01)",
    )
    parser.add_argument(
        "--qmm-margin-target",
        type=float,
        default=0.3,
        help="QMM target margin from boundary (default: 0.3)",
    )

    # R4-A4-P7: Ternary Inoculation
    parser.add_argument(
        "--use-ternary-inoculation",
        action="store_true",
        help="Enable antifragile ternary inoculation training",
    )
    parser.add_argument(
        "--inoculation-flip-prob",
        type=float,
        default=0.01,
        help="Probability of inoculation per step (default: 0.01)",
    )
    parser.add_argument(
        "--inoculation-flip-fraction",
        type=float,
        default=0.001,
        help="Fraction of weights to flip during inoculation (default: 0.001)",
    )

    # R4-A4-M6: Trajectory Verification
    parser.add_argument(
        "--use-trajectory-verification",
        action="store_true",
        help="Enable recursive trajectory verification",
    )
    parser.add_argument(
        "--trajectory-verify-interval",
        type=int,
        default=50,
        help="Steps between trajectory verifications (default: 50)",
    )

    # R4-A4-M7: Ternary Delta Chronicle
    parser.add_argument(
        "--use-delta-chronicle",
        action="store_true",
        help="Record every ternary weight flip",
    )
    parser.add_argument(
        "--delta-chronicle-max-mb",
        type=float,
        default=500.0,
        help="Max memory for delta history (MB, default: 500)",
    )

    # R4-A4-M8: Flip Influence Decomposition
    parser.add_argument(
        "--use-flip-influence",
        action="store_true",
        help="Enable per-flip loss influence tracking",
    )
    parser.add_argument(
        "--flip-influence-top-k",
        type=int,
        default=100,
        help="Track top-k most influential flips (default: 100)",
    )

    # R4-A4-M4: Entropic Delta Encoding
    parser.add_argument(
        "--use-entropic-delta",
        action="store_true",
        help="Enable entropic delta encoding for checkpoint compression",
    )

    # R4-A4 Batch 5: Advanced/Speculative
    parser.add_argument(
        "--use-optimizer-reconstruction",
        action="store_true",
        help="Enable optimizer state reconstruction from sketches",
    )
    parser.add_argument(
        "--optimizer-reconstruction-steps",
        type=int,
        default=50,
        help="Steps between optimizer reconstruction updates (default: 50)",
    )
    parser.add_argument(
        "--use-trajectory-replay",
        action="store_true",
        help="Enable deterministic trajectory replay for debugging",
    )
    parser.add_argument(
        "--trajectory-anchor-interval",
        type=int,
        default=5000,
        help="Steps between trajectory anchor checkpoints (default: 5000)",
    )
    parser.add_argument(
        "--use-adaptive-checkpoint",
        action="store_true",
        help="Enable adaptive checkpoint interval based on gradient variance",
    )
    parser.add_argument(
        "--use-iaf-optimizer",
        action="store_true",
        help="Enable Integrate-and-Fire optimizer variant",
    )
    parser.add_argument(
        "--iaf-threshold",
        type=float,
        default=1.0,
        help="IAF activation threshold (default: 1.0)",
    )
    parser.add_argument(
        "--iaf-leak",
        type=float,
        default=0.99,
        help="IAF memory leak factor (default: 0.99)",
    )
    parser.add_argument(
        "--use-neural-codec",
        action="store_true",
        help="Enable neural checkpoint codec for weight compression",
    )
    parser.add_argument(
        "--use-gf3-coding",
        action="store_true",
        help="Enable GF(3) algebraic error-correcting coding",
    )
    parser.add_argument(
        "--use-ccsr",
        action="store_true",
        help="Enable cross-correlation state recovery",
    )

    # R17 Batch AM: STE and SSM Optimizations
    parser.add_argument('--use-strang-ste', action='store_true', help='F1: Strang Splitting STE')
    parser.add_argument('--use-fourier-init', action='store_true', help='F8: Fourier holographic init')
    parser.add_argument('--use-gv-ste', action='store_true', help='F5: GV-STE adaptive slope')
    parser.add_argument('--use-rpe-gate', action='store_true', help='F6: RPE flip gate')
    parser.add_argument('--use-surprise-dt', action='store_true', help='F7: Surprise-driven dt gate')
    parser.add_argument('--surprise-dt-scale', type=float, default=0.5, help='F7: surprise scale')
    parser.add_argument('--use-bpnp', action='store_true', help='F2: BPNP soft SSM gate')
    parser.add_argument('--bpnp-threshold', type=float, default=0.3, help='F2: BPNP gate threshold')
    parser.add_argument('--use-dssu', action='store_true', help='F3: Dimension-Selective State Update')
    parser.add_argument('--dssu-epsilon', type=float, default=0.01, help='F3: DSSU activation threshold')
    # R17 Batch AN: R17-A findings
    parser.add_argument('--use-mdl-loss', action='store_true', help='AN-1: MDL sparsity loss')
    parser.add_argument('--mdl-alpha', type=float, default=1e-4, help='AN-1: MDL loss weight')
    parser.add_argument('--use-phase-detection', action='store_true', help='AN-3: Phase detection variance')
    parser.add_argument('--phase-window-size', type=int, default=50, help='AN-3: variance window size')
    parser.add_argument('--n-gist-tokens', type=int, default=0, help='AN-4: number of gist prefix tokens')
    parser.add_argument('--use-cohom-loss', action='store_true', help='AN-5: Cohomological loss')
    parser.add_argument('--cohom-alpha', type=float, default=1e-4, help='AN-5: cohom loss weight')
    parser.add_argument('--ms-ssm-n-scales', type=int, default=1, help='AN-6: MS-SSM number of timescale groups')
    parser.add_argument('--ms-ssm-log-min', type=float, default=-4.0, help='AN-6: MS-SSM log scale min')
    parser.add_argument('--ms-ssm-log-max', type=float, default=0.0, help='AN-6: MS-SSM log scale max')

    # R17 Batch AO: R17-B findings (Non-Textual Input, adapted for text)
    parser.add_argument('--use-wasserstein-ste', action='store_true', help='AO-1: Wasserstein-1 STE (sign grad)')
    parser.add_argument('--use-event-encoding', action='store_true', help='AO-2: Ternary diff event encoding')
    parser.add_argument('--use-ergodic-loss', action='store_true', help='AO-3: Ergodic semantic loss')
    parser.add_argument('--ergodic-loss-alpha', type=float, default=1e-3, help='AO-3: ergodic loss weight')
    parser.add_argument('--use-ssm-annealing', action='store_true', help='AO-4: SSM A_log developmental annealing')
    parser.add_argument('--ssm-anneal-warmup', type=int, default=1000, help='AO-4: steps to full A_log range')
    parser.add_argument('--ssm-anneal-log-max', type=float, default=4.0, help='AO-4: A_log clamp max at full anneal')
    parser.add_argument('--use-masked-pred', action='store_true', help='AO-5: Masked token prediction aux loss')
    parser.add_argument('--masked-pred-ratio', type=float, default=0.15, help='AO-5: fraction of tokens to mask')
    parser.add_argument('--masked-pred-lambda', type=float, default=0.1, help='AO-5: masked pred loss weight')
    parser.add_argument('--use-temporal-contrastive', action='store_true', help='AO-6: Temporal InfoNCE loss')
    parser.add_argument('--temporal-contrastive-k', type=int, default=4, help='AO-6: k step offset for positive pair')
    parser.add_argument('--temporal-contrastive-lambda', type=float, default=0.01, help='AO-6: temporal contrastive weight')

    # Batch AP: R17-C findings (Continuous Learning + Plasticity)
    parser.add_argument('--use-sigma-gate', action='store_true', help='AP-1: SSM z-branch variance novelty gate')
    parser.add_argument('--sigma-gate-threshold', type=float, default=0.5, help='AP-1: novelty threshold')
    parser.add_argument('--sigma-gate-baseline-beta', type=float, default=0.99, help='AP-1: baseline EMA decay')
    parser.add_argument('--use-etc', action='store_true', help='AP-2: Ergodic Ternary Crystallization')
    parser.add_argument('--etc-beta', type=float, default=0.9, help='AP-2: sign-EMA decay')
    parser.add_argument('--etc-tau', type=float, default=0.8, help='AP-2: crystallization threshold')
    parser.add_argument('--etc-crystallize-step', type=int, default=500, help='AP-2: earliest crystallization step')
    parser.add_argument('--use-beb', action='store_true', help='AP-3: Binary Error Backprop INT16 accumulator')
    parser.add_argument('--use-ema-replay', action='store_true', help='AP-4: EMA shadow generative replay KL loss')
    parser.add_argument('--ema-replay-lambda', type=float, default=0.05, help='AP-4: KL loss weight')
    parser.add_argument('--use-twa', action='store_true', help='AP-5: Thermodynamic Weight Annealing occupancy')
    parser.add_argument('--twa-temp', type=float, default=1.0, help='AP-5: initial temperature')
    parser.add_argument('--twa-anneal-steps', type=int, default=2000, help='AP-5: annealing steps')
    parser.add_argument('--use-scme', action='store_true', help='AP-6: Sign-Consistent Measure Evolution')
    parser.add_argument('--scme-alpha', type=float, default=0.95, help='AP-6: C accumulator decay')
    parser.add_argument('--use-si', action='store_true', help='AP-7: Synaptic Intelligence importance')
    parser.add_argument('--use-agem', action='store_true', help='AP-8: A-GEM gradient ring buffer')
    parser.add_argument('--agem-buffer-size', type=int, default=256, help='AP-8: buffer capacity')
    parser.add_argument('--use-ewc', action='store_true', help='AP-9: EWC Fisher diagonal flip protection')
    parser.add_argument('--ewc-lambda', type=float, default=0.01, help='AP-9: flip penalty weight')
    parser.add_argument('--use-deltanet-ssm', action='store_true', help='AP-10: DeltaNet SSM (stub)')
    parser.add_argument('--use-ttt-fast-weights', action='store_true', help='AP-11: TTT fast weights (stub)')
    parser.add_argument('--use-predictive-coding', action='store_true', help='AP-12: Predictive coding (stub)')
    parser.add_argument('--use-dual-hebbian', action='store_true', help='AP-13: Dual-timescale Hebbian (stub)')
    parser.add_argument('--use-fast-weights', action='store_true', help='AP-14: Fast weights outer product (stub)')
    parser.add_argument('--use-per-weight-temp', action='store_true', help='AP-15: Per-weight temperature (stub)')
    parser.add_argument('--use-lota-qaf', action='store_true', help='AP-16: LoTA-QAF INT8 adapt (stub)')
    parser.add_argument('--use-tpm', action='store_true', help='AP-17: Topological Persistence Memory (stub)')
    parser.add_argument('--use-csc', action='store_true', help='AP-18: Causal State Compression (stub)')
    parser.add_argument('--use-attractor-basin', action='store_true', help='AP-19: Attractor Basin (stub)')

    # Batch AQ — R17-D Architectural Plasticity
    parser.add_argument('--use-spectral-band-init', action='store_true', help='AQ-1: Spectral Band A_log init')
    parser.add_argument('--use-dead-dim-pruning', action='store_true', help='AQ-2: LLM-Pruner dead dims')
    parser.add_argument('--dead-dim-threshold', type=float, default=0.9, help='AQ-2: zero_frac threshold')
    parser.add_argument('--dead-dim-prune-step', type=int, default=1000, help='AQ-2: prune every N steps')
    parser.add_argument('--use-erank-fixedpoint', action='store_true', help='AQ-3: Effective rank fixed-point')
    parser.add_argument('--erank-check-interval', type=int, default=500, help='AQ-3: check every N steps')
    parser.add_argument('--erank-grow-threshold', type=float, default=0.8, help='AQ-3: grow threshold')
    parser.add_argument('--erank-prune-threshold', type=float, default=0.2, help='AQ-3: prune threshold')
    parser.add_argument('--erank-delta', type=int, default=4, help='AQ-3: dims to grow/prune')
    parser.add_argument('--use-liquid-s4', action='store_true', help='AQ-4: Liquid-S4 input-dependent A')
    parser.add_argument('--use-depth-plasticity', action='store_true', help='AQ-5: Convergence depth plasticity')
    parser.add_argument('--depth-plasticity-window', type=int, default=8, help='AQ-5: hash window size')
    parser.add_argument('--use-mod-routing', action='store_true', help='AQ-6: Mixture-of-Depths token routing')
    parser.add_argument('--mod-top-k-fraction', type=float, default=0.5, help='AQ-6: fraction of tokens processed')
    parser.add_argument('--use-expert-choice-moe', action='store_true', help='AQ-7: Expert-Choice MoE')
    parser.add_argument('--expert-choice-n-experts', type=int, default=4, help='AQ-7: N experts')
    parser.add_argument('--use-hypernetwork-thresholds', action='store_true', help='AQ-8: Hypernetwork adaptive thresholds')
    parser.add_argument('--use-cgls-stacking', action='store_true', help='AQ-9: CGLS progressive layer stacking')
    parser.add_argument('--cgls-insert-step', type=int, default=5000, help='AQ-9: insert layer at step N')
    parser.add_argument('--use-topo-insertion', action='store_true', help='AQ-10: Topological derivative layer insertion')
    parser.add_argument('--use-fisher-arch-flow', action='store_true', help='AQ-11: Fisher architecture flow')
    parser.add_argument('--fisher-arch-interval', type=int, default=1000, help='AQ-11: check every N steps')
    parser.add_argument('--use-rg-flow', action='store_true', help='AQ-12: Ternary renormalization group flow')
    parser.add_argument('--rg-flow-threshold', type=float, default=0.1, help='AQ-12: eigenvalue prune threshold')
    parser.add_argument('--use-committed-inhibition', action='store_true', help='AQ-13: Committed spatial inhibition')
    parser.add_argument('--use-ho-kalman', action='store_true', help='AQ-14: Ho-Kalman Hankel spectral realization')
    parser.add_argument('--ho-kalman-lag-max', type=int, default=64, help='AQ-14: max lag for Hankel')
    parser.add_argument('--ho-kalman-calib-tokens', type=int, default=1000, help='AQ-14: calibration tokens')
    parser.add_argument('--use-automaton-collapse', action='store_true', help='AQ-15: Automaton state collapse')
    parser.add_argument('--use-toeplitz-ca', action='store_true', help='AQ-16: Ternary Toeplitz cellular automaton')
    parser.add_argument('--toeplitz-kernel-size', type=int, default=7, help='AQ-16: kernel size')
    parser.add_argument('--use-persistent-homology', action='store_true', help='AQ-17: Persistent homology activation manifold')
    parser.add_argument('--use-supernet-nas', action='store_true', help='AQ-18: Supernet NAS for d_state')
    parser.add_argument('--use-neural-ode-weights', action='store_true', help='AQ-19: Neural ODE continuous layer weight gen')
    parser.add_argument('--use-foundation-hypernetwork', action='store_true', help='AQ-20: Foundation hypernetwork for DSST init')

    # Batch AR — R17-E Lazy Instantiation
    parser.add_argument('--use-adaponder', action='store_true', help='AR-1: AdaPonderLM halting gate')
    parser.add_argument('--adaponder-halt-threshold', type=float, default=0.5, help='AR-1: halt threshold')
    parser.add_argument('--adaponder-max-iters', type=int, default=3, help='AR-1: max iterations')
    # AR-2 --use-dssu already registered in earlier batch (F3)
    parser.add_argument('--use-sssp', action='store_true', help='AR-3: Sign-Stable State Propagation')
    parser.add_argument('--sssp-stability-threshold', type=float, default=0.8, help='AR-3: stability threshold')
    parser.add_argument('--use-scms', action='store_true', help='AR-4: Sparse Chunk Matrix Summation')
    parser.add_argument('--scms-passive-threshold', type=float, default=0.01, help='AR-4: passive token threshold')
    parser.add_argument('--use-hash-ternary-routing', action='store_true', help='AR-5: Hash Ternary Weight Selection')
    parser.add_argument('--hash-n-groups', type=int, default=8, help='AR-5: N hash groups')
    parser.add_argument('--use-input-entropy-bypass', action='store_true', help='AR-6: Input-Entropy Ternary Bypass')
    parser.add_argument('--input-entropy-lambda', type=float, default=0.001, help='AR-6: L_stability lambda')
    parser.add_argument('--use-structured-zero-cols', action='store_true', help='AR-7: Structured Zero Columns')
    parser.add_argument('--structured-zero-lambda', type=float, default=0.001, help='AR-7: L_col lambda')
    parser.add_argument('--structured-zero-threshold', type=float, default=0.02, help='AR-7: dead col threshold')
    parser.add_argument('--use-thdv', action='store_true', help='AR-8: Temporal Half-Depth Validation')
    parser.add_argument('--thdv-kl-weight', type=float, default=0.01, help='AR-8: KL weight')
    parser.add_argument('--use-layerskip', action='store_true', help='AR-9: LayerSkip (stub)')
    parser.add_argument('--use-teal-sparsity', action='store_true', help='AR-10: TEAL activation sparsity (stub)')
    parser.add_argument('--use-specee', action='store_true', help='AR-11: SpecEE (stub)')
    parser.add_argument('--use-mixture-of-recursions', action='store_true', help='AR-12: Mixture-of-Recursions (stub)')
    parser.add_argument('--use-evnn-threshold', action='store_true', help='AR-13: EVNN threshold (stub)')
    parser.add_argument('--use-ternary-speculative-state', action='store_true', help='AR-14: Ternary Speculative State (stub)')
    parser.add_argument('--use-csbc', action='store_true', help='AR-15: CSBC basis compression (stub)')

    # Batch AS — R17-F: System 1 vs System 2 Dual-Process Cognition
    parser.add_argument('--use-gvste', action='store_true', help='AS-1: Gradient Variance Adaptive STE')
    parser.add_argument('--use-ada-gate', action='store_true', help='AS-2: Adjoint-Functor Dual-Process gate')
    parser.add_argument('--ada-coherence-threshold', type=float, default=0.15, help='AS-2: ADA coherence threshold')
    parser.add_argument('--use-mi-s1', action='store_true', help='AS-3: Mutual-Information S1 Boundary (stub)')
    parser.add_argument('--mi-s1-entropy-threshold', type=float, default=1.0, help='AS-3: MI-S1 entropy threshold')
    parser.add_argument('--use-seee', action='store_true', help='AS-4: SSM-Entropy Early Exit (stub)')
    parser.add_argument('--seee-entropy-threshold', type=float, default=1.0, help='AS-4: SEEE entropy threshold')
    parser.add_argument('--use-tug', action='store_true', help='AS-5: Ternary UNKNOWN Gate')
    parser.add_argument('--tug-zero-frac-threshold', type=float, default=0.7, help='AS-5: TUG zero-frac threshold')
    parser.add_argument('--use-sgb', action='store_true', help='AS-6: Surprisal-Gated Backprop probe')
    parser.add_argument('--sgb-surprisal-threshold', type=float, default=3.0, help='AS-6: SGB surprisal threshold')
    parser.add_argument('--use-tcsm', action='store_true', help='AS-7: Ternary Crystallization Sign-Stability Mask')
    parser.add_argument('--tcsm-stable-steps', type=int, default=200, help='AS-7: TCSM stability window (steps)')
    parser.add_argument('--use-ergodic-mixing', action='store_true', help='AS-8: Ergodic Mixing Time decomposition')
    parser.add_argument('--ergodic-lambda-threshold', type=float, default=0.7, help='AS-8: ERGODIC lambda threshold')
    parser.add_argument('--use-vff-routing', action='store_true', help='AS-9: VersatileFFN Depth Routing probe')
    parser.add_argument('--use-fast-slow-ssm', action='store_true', help='AS-10: Fast-Slow SSM timescales probe')
    parser.add_argument('--fast-slow-factor', type=float, default=4.0, help='AS-10: F-S fast/slow dt factor')
    parser.add_argument('--use-lyapunov-basin', action='store_true', help='AS-11: Lyapunov Basin S1/S2 gate probe')
    parser.add_argument('--lyapunov-basin-threshold', type=float, default=1.0, help='AS-11: LYAP basin threshold')
    parser.add_argument('--use-fsw', action='store_true', help='AS-12: Fast/Slow Weights Outer Product (stub)')
    parser.add_argument('--fsw-lambda', type=float, default=0.99, help='AS-12: FSW decay lambda')
    parser.add_argument('--fsw-eta', type=float, default=0.01, help='AS-12: FSW learning rate eta')
    parser.add_argument('--use-dplr', action='store_true', help='AS-13: Decay-Partitioned Learning Rules')
    parser.add_argument('--dplr-retention-threshold', type=float, default=0.5, help='AS-13: DPLR retention threshold')
    parser.add_argument('--use-sprt', action='store_true', help='AS-14: Wald SPRT Commit (stub)')
    parser.add_argument('--sprt-log-threshold', type=float, default=2.0, help='AS-14: SPRT log-likelihood threshold')
    parser.add_argument('--use-act', action='store_true', help='AS-15: Adaptive Computation Time (stub)')
    parser.add_argument('--use-pc2', action='store_true', help='AS-16: Predictive Coding Error Correction (stub)')
    parser.add_argument('--pc2-beta', type=float, default=0.1, help='AS-16: PC2 correction beta')
    parser.add_argument('--use-bns2', action='store_true', help='AS-17: B-Projection Norm System 2 Signal')
    parser.add_argument('--bns2-norm-threshold', type=float, default=0.1, help='AS-17: BNS2 norm threshold')
    parser.add_argument('--use-tsm', action='store_true', help='AS-18: Ternary State Memoization Cache (stub)')
    parser.add_argument('--tsm-cache-k', type=int, default=8, help='AS-18: TSM cache key length')
    parser.add_argument('--use-cls-hopfield', action='store_true', help='AS-19: CLS Hopfield episodic buffer (stub)')
    parser.add_argument('--use-ponder-ssm', action='store_true', help='AS-20: PonderNet SSM Halting (stub)')
    parser.add_argument('--ponder-lambda', type=float, default=0.01, help='AS-20: PonderNet ponder cost lambda')

    # Batch AT — R17-G: Biological Memory Systems
    parser.add_argument('--use-cls-timescale', action='store_true', help='AT-1: Three-Timescale CLS EMA context')
    parser.add_argument('--cls-timescale-alpha', type=float, default=0.99, help='AT-1: CLS EMA decay alpha')
    parser.add_argument('--use-smr-memory-tokens', action='store_true', help='AT-2: SMR learnable memory tokens (stub)')
    parser.add_argument('--smr-num-tokens', type=int, default=4, help='AT-2: SMR number of memory tokens')
    parser.add_argument('--use-dopamine-rpe-gate', action='store_true', help='AT-3: Dopamine RPE Sign Gate')
    parser.add_argument('--dopamine-ema-alpha', type=float, default=0.99, help='AT-3: Dopamine EMA decay')
    parser.add_argument('--use-dt-surprise-gate', action='store_true', help='AT-4: DT Surprise Gate (prediction-error)')
    parser.add_argument('--use-holographic-init', action='store_true', help='AT-5: Holographic Fourier weight init')
    parser.add_argument('--use-nrem-rem', action='store_true', help='AT-6: NREM/REM alternation dt modulation (stub)')
    parser.add_argument('--nrem-attractor-eps', type=float, default=0.1, help='AT-6: NREM attractor stability eps')
    parser.add_argument('--use-hippocampal-sparsity', action='store_true', help='AT-7: Hippocampal indexing sparsity (stub)')
    parser.add_argument('--hippocampal-sparsity-k', type=float, default=0.05, help='AT-7: Hippocampal topk fraction')
    parser.add_argument('--use-gsemm-init', action='store_true', help='AT-8: GSEMM variable binding init (stub)')
    parser.add_argument('--use-delta-rule', action='store_true', help='AT-9: Delta rule error correction (stub)')
    parser.add_argument('--delta-rule-beta', type=float, default=0.1, help='AT-9: Delta rule learning rate')
    parser.add_argument('--use-ior-committed', action='store_true', help='AT-10: IOR committed weight decay')
    parser.add_argument('--ior-high-alpha', type=float, default=0.01, help='AT-10: IOR alpha for committed weights')
    parser.add_argument('--ior-low-alpha', type=float, default=0.1, help='AT-10: IOR alpha for uncommitted weights')
    parser.add_argument('--use-pfc-boundary-gate', action='store_true', help='AT-11: PFC top-down boundary gate (stub)')
    parser.add_argument('--use-synaptic-tagging', action='store_true', help='AT-12: Synaptic tagging and capture (stub)')
    parser.add_argument('--stc-theta-c', type=float, default=0.01, help='AT-12/18: STC grad threshold')
    parser.add_argument('--use-reconsolidation', action='store_true', help='AT-13: Reconsolidation destabilization (stub)')
    parser.add_argument('--use-offline-consolidation', action='store_true', help='AT-14: Offline consolidation pass')
    parser.add_argument('--offline-consolidation-n', type=int, default=10, help='AT-14: Offline consolidation interval (steps)')
    parser.add_argument('--offline-consolidation-magnitude-scale', type=float, default=1.1, help='AT-14: Magnitude scale for consolidation')
    parser.add_argument('--use-engram-orthogonality', action='store_true', help='AT-15: Memory engram orthogonality penalty')
    parser.add_argument('--engram-lambda', type=float, default=0.01, help='AT-15: Engram penalty weight')
    parser.add_argument('--engram-penalty-interval', type=int, default=50, help='AT-15: Engram penalty interval (steps)')
    parser.add_argument('--use-fisher-consolidation', action='store_true', help='AT-16: Fisher engram consolidation rate')
    parser.add_argument('--use-hopfield-threshold', action='store_true', help='AT-17: Hopfield capacity threshold (stub)')
    parser.add_argument('--use-stc-markov', action='store_true', help='AT-18: STC Ternary Markov absorption')
    parser.add_argument('--stc-k', type=int, default=5, help='AT-18: STC Markov absorption threshold K')
    parser.add_argument('--use-two-step-displacement', action='store_true', help='AT-19: Two-step ternary displacement via zero')
    parser.add_argument('--use-stdp-sign-conv', action='store_true', help='AT-20: STDP causal sign convolution (stub)')
    parser.add_argument('--stdp-blend-alpha', type=float, default=0.3, help='AT-20: STDP blend alpha')

    # Batch AU — R17-H: Computational Physics
    parser.add_argument('--use-slssu', action='store_true', help='AU-1: SL-SSU momentum contraction')
    parser.add_argument('--use-error-annealing', action='store_true', help='AU-2: Error annealing decay schedule')
    parser.add_argument('--error-annealing-alpha', type=float, default=0.1, help='AU-2: Error annealing alpha')
    parser.add_argument('--use-shadow-hamiltonian', action='store_true', help='AU-3: Shadow Hamiltonian conserved energy')
    parser.add_argument('--shadow-h-temperature', type=float, default=1.0, help='AU-3: Shadow Hamiltonian temperature')
    parser.add_argument('--use-maxent-gibbs', action='store_true', help='AU-4: MaxEnt Gibbs distribution sampling')
    parser.add_argument('--maxent-T', type=float, default=1.0, help='AU-4: MaxEnt temperature')
    parser.add_argument('--maxent-target-sparsity', type=float, default=0.5, help='AU-4: MaxEnt target sparsity')
    parser.add_argument('--use-kmc-firing', action='store_true', help='AU-5: KMC fired event tracking')
    parser.add_argument('--kmc-top-k', type=int, default=10, help='AU-5: KMC top-K fired events')
    parser.add_argument('--use-rg-freeze', action='store_true', help='AU-6: RG renormalization group freezing')
    parser.add_argument('--rg-freeze-threshold', type=float, default=0.1, help='AU-6: RG freeze threshold')
    parser.add_argument('--use-nose-hoover', action='store_true', help='AU-7: Nosé-Hoover thermostat coupling')
    parser.add_argument('--nose-hoover-target-flip', type=float, default=0.3, help='AU-7: Nosé-Hoover target flip probability')
    parser.add_argument('--use-rsb-monitor', action='store_true', help='AU-8: RSB monitoring (stub)')
    parser.add_argument('--use-ising-gate', action='store_true', help='AU-9: Ising model gate interaction')
    parser.add_argument('--use-fpu-recurrence', action='store_true', help='AU-10: FPU recurrence relation coupling')
    parser.add_argument('--use-percolation-mask', action='store_true', help='AU-11: Percolation network masking')
    parser.add_argument('--percolation-threshold', type=float, default=0.5, help='AU-11: Percolation threshold')
    parser.add_argument('--use-strang-splitting', action='store_true', help='AU-12: Strang splitting symplectic integrator')

    # --- Batch AV: R17-I Native Ternary Software Stack ---
    parser.add_argument('--use-av1-termac-probe', action='store_true', default=True)
    parser.add_argument('--no-av1-termac-probe', dest='use_av1_termac_probe', action='store_false')
    parser.add_argument('--use-av2-packing-probe', action='store_true', default=True)
    parser.add_argument('--no-av2-packing-probe', dest='use_av2_packing_probe', action='store_false')
    parser.add_argument('--use-av3-sparsity-probe', action='store_true', default=True)
    parser.add_argument('--use-av4-xor-probe', action='store_true', default=True)
    parser.add_argument('--use-av5-asym-threshold-probe', action='store_true', default=True)
    parser.add_argument('--use-av6-elut-probe', action='store_true', default=True)
    parser.add_argument('--use-av7-lut-layout-probe', action='store_true', default=True)
    parser.add_argument('--use-av8-sparse-nnz-probe', action='store_true', default=True)
    parser.add_argument('--use-av9-llvm-probe', action='store_true', default=True)
    parser.add_argument('--use-av10-cache-block-probe', action='store_true', default=True)
    parser.add_argument('--use-av11-reversible-ssm-probe', action='store_true', default=True)
    parser.add_argument('--use-av12-circulant-probe', action='store_true', default=True)
    parser.add_argument('--use-av13-ca-rule-probe', action='store_true', default=True)
    parser.add_argument('--use-av14-mds-probe', action='store_true', default=True)
    parser.add_argument('--use-av15-tree-reduce-probe', action='store_true', default=True)
    parser.add_argument('--use-av16-gf3-snf-probe', action='store_true', default=True)
    parser.add_argument('--use-av17-iott', action='store_true', default=True)
    parser.add_argument('--iott-use-as-threshold', action='store_true')
    parser.add_argument('--use-av18-cayley-hamilton-probe', action='store_true', default=True)
    parser.add_argument('--use-av19-coord-descent-b-probe', action='store_true', default=True)
    parser.add_argument('--use-av20-base3-probe', action='store_true', default=True)

    # --- Batch AW: R17-J Novel Mathematics ---
    parser.add_argument('--use-aw1-tropical-probe', action='store_true', default=True)
    parser.add_argument('--use-aw2-sheaf-energy-probe', action='store_true', default=True)
    parser.add_argument('--use-aw3-neural-persistence-probe', action='store_true', default=True)
    parser.add_argument('--use-aw4-dsst-functor-probe', action='store_true', default=True)
    parser.add_argument('--use-aw5-hyperoctahedral-probe', action='store_true', default=True)
    parser.add_argument('--use-aw6-fim-probe', action='store_true', default=True)
    parser.add_argument('--use-aw7-pac-bayes-probe', action='store_true', default=True)
    parser.add_argument('--use-aw8-sinkhorn-probe', action='store_true', default=True)
    parser.add_argument('--use-aw9-path-sig-init', action='store_true')
    parser.add_argument('--use-aw10-free-prob-probe', action='store_true', default=True)
    parser.add_argument('--use-aw11-bp-threshold-probe', action='store_true', default=True)
    parser.add_argument('--use-aw12-hyperreal-ste', action='store_true', default=True)
    parser.add_argument('--use-aw13-lz-probe', action='store_true', default=True)
    parser.add_argument('--use-aw14-kleene-init', action='store_true')
    parser.add_argument('--use-aw15-renyi-probe', action='store_true', default=True)
    parser.add_argument('--use-aw16-clifford-probe', action='store_true', default=True)
    parser.add_argument('--use-aw17-ncg-probe', action='store_true', default=True)
    parser.add_argument('--use-aw18-tarski-init', action='store_true')
    parser.add_argument('--use-aw19-mdl-reward-probe', action='store_true', default=True)
    parser.add_argument('--use-aw20-hott-radius-probe', action='store_true', default=True)

    # AX: Batch AX — R17-K Multi-Dimensional Evaluation
    parser.add_argument('--use-ax01-bbh-valley-probe', dest='use_ax01_bbh_valley_probe', action='store_true', default=True)
    parser.add_argument('--no-ax01-bbh-valley-probe', dest='use_ax01_bbh_valley_probe', action='store_false')
    parser.add_argument('--use-ax02-think-bench-probe', dest='use_ax02_think_bench_probe', action='store_true', default=True)
    parser.add_argument('--no-ax02-think-bench-probe', dest='use_ax02_think_bench_probe', action='store_false')
    parser.add_argument('--use-ax03-cognitive-profile-probe', dest='use_ax03_cognitive_profile_probe', action='store_true', default=True)
    parser.add_argument('--no-ax03-cognitive-profile-probe', dest='use_ax03_cognitive_profile_probe', action='store_false')
    parser.add_argument('--use-ax04-metric-pref-probe', dest='use_ax04_metric_pref_probe', action='store_true', default=True)
    parser.add_argument('--no-ax04-metric-pref-probe', dest='use_ax04_metric_pref_probe', action='store_false')
    parser.add_argument('--use-ax05-scaling-ceiling-probe', dest='use_ax05_scaling_ceiling_probe', action='store_true', default=True)
    parser.add_argument('--no-ax05-scaling-ceiling-probe', dest='use_ax05_scaling_ceiling_probe', action='store_false')
    parser.add_argument('--use-ax06-power-law-probe', dest='use_ax06_power_law_probe', action='store_true', default=True)
    parser.add_argument('--no-ax06-power-law-probe', dest='use_ax06_power_law_probe', action='store_false')
    parser.add_argument('--use-ax07-loss2loss-probe', dest='use_ax07_loss2loss_probe', action='store_true', default=True)
    parser.add_argument('--no-ax07-loss2loss-probe', dest='use_ax07_loss2loss_probe', action='store_false')
    parser.add_argument('--use-ax08-pareto-probe', dest='use_ax08_pareto_probe', action='store_true', default=True)
    parser.add_argument('--no-ax08-pareto-probe', dest='use_ax08_pareto_probe', action='store_false')
    parser.add_argument('--use-ax09-ternary-health-probe', dest='use_ax09_ternary_health_probe', action='store_true', default=True)
    parser.add_argument('--no-ax09-ternary-health-probe', dest='use_ax09_ternary_health_probe', action='store_false')
    parser.add_argument('--use-ax10-compute-efficiency-probe', dest='use_ax10_compute_efficiency_probe', action='store_true', default=True)
    parser.add_argument('--no-ax10-compute-efficiency-probe', dest='use_ax10_compute_efficiency_probe', action='store_false')
    parser.add_argument('--use-ax11-channel-capacity-probe', dest='use_ax11_channel_capacity_probe', action='store_true', default=True)
    parser.add_argument('--no-ax11-channel-capacity-probe', dest='use_ax11_channel_capacity_probe', action='store_false')
    parser.add_argument('--use-ax12-structural-win-probe', dest='use_ax12_structural_win_probe', action='store_true', default=True)
    parser.add_argument('--no-ax12-structural-win-probe', dest='use_ax12_structural_win_probe', action='store_false')
    parser.add_argument('--use-ax13-readiness-score-probe', dest='use_ax13_readiness_score_probe', action='store_true', default=True)
    parser.add_argument('--no-ax13-readiness-score-probe', dest='use_ax13_readiness_score_probe', action='store_false')
    parser.add_argument('--use-ax14-ssm-attractor-probe', dest='use_ax14_ssm_attractor_probe', action='store_true', default=True)
    parser.add_argument('--no-ax14-ssm-attractor-probe', dest='use_ax14_ssm_attractor_probe', action='store_false')
    parser.add_argument('--use-ax15-routing-entropy-probe', dest='use_ax15_routing_entropy_probe', action='store_true', default=True)
    parser.add_argument('--no-ax15-routing-entropy-probe', dest='use_ax15_routing_entropy_probe', action='store_false')
    parser.add_argument('--use-ax16-tiq-probe', dest='use_ax16_tiq_probe', action='store_true', default=True)
    parser.add_argument('--no-ax16-tiq-probe', dest='use_ax16_tiq_probe', action='store_false')
    parser.add_argument('--use-ax17-cpbtc-probe', dest='use_ax17_cpbtc_probe', action='store_true', default=True)
    parser.add_argument('--no-ax17-cpbtc-probe', dest='use_ax17_cpbtc_probe', action='store_false')
    parser.add_argument('--use-ax18-tfcd-probe', dest='use_ax18_tfcd_probe', action='store_true', default=True)
    parser.add_argument('--no-ax18-tfcd-probe', dest='use_ax18_tfcd_probe', action='store_false')
    parser.add_argument('--use-ax19-dcc-probe', dest='use_ax19_dcc_probe', action='store_true', default=True)
    parser.add_argument('--no-ax19-dcc-probe', dest='use_ax19_dcc_probe', action='store_false')
    parser.add_argument('--use-ax20-itst-probe', dest='use_ax20_itst_probe', action='store_true', default=True)
    parser.add_argument('--no-ax20-itst-probe', dest='use_ax20_itst_probe', action='store_false')

    args = parser.parse_args()

    # Load config
    config_map = {
        "1m": config_1m,
        "10m": config_10m,
        "100m": config_100m,
        "scale_10m": config_scale_10m,
        "wikitext_6m": config_wikitext_6m,
        "wikitext_50m": config_wikitext_50m,
        "wikitext_100m": config_wikitext_100m,
        "wikitext_bench": config_wikitext_bench,
        "ternary_6m": config_ternary_6m,
        "ternary_50m": config_ternary_50m,
        "ternary_6m_moe": config_ternary_6m_moe,
    }

    config = config_map[args.config]()

    # Override seq_len if specified
    if args.seq_len is not None:
        config.seq_len = args.seq_len

    # --use-galore flag overrides optimizer_type for convenience
    if args.use_galore:
        args.optimizer_type = "galore"

    if args.ternary_lm_head:
        config.ternary_lm_head = True

    if args.use_optrot:
        config.use_optrot = True
    if args.optrot_lambda != 0.01:
        config.optrot_lambda = args.optrot_lambda

    if args.use_moe:
        config.use_moe = True
    if args.moe_num_experts != 4:
        config.moe_num_experts = args.moe_num_experts
    if args.moe_top_k is not None:
        config.moe_top_k = args.moe_top_k
    if args.moe_load_balance_coeff is not None:
        config.moe_load_balance_coeff = args.moe_load_balance_coeff

    if args.use_deferred_ternary:
        config.use_deferred_ternary = True
    if args.deferred_rank != 32:
        config.deferred_rank = args.deferred_rank
    if args.deferred_merge_interval != 100:
        config.deferred_merge_interval = args.deferred_merge_interval

    if args.use_tscs:
        config.use_tscs = True
    if args.tscs_osc_fraction != 0.375:
        config.tscs_osc_fraction = args.tscs_osc_fraction
    if args.tscs_reset_fraction != 0.375:
        config.tscs_reset_fraction = args.tscs_reset_fraction
    if args.tscs_clamp_state != 0.0:
        config.tscs_clamp_state = args.tscs_clamp_state

    if args.use_btg:
        config.use_btg = True

    if args.use_tatgr:
        config.use_tatgr = True

    if args.use_dsst:
        config.use_dsst = True

    # R17-A (Batch AM) mappings
    if args.use_strang_ste:
        config.use_strang_ste = True
    if args.use_fourier_init:
        config.use_fourier_init = True
    if args.use_gv_ste:
        config.use_gv_ste = True
    if args.use_rpe_gate:
        config.use_rpe_gate = True
    if args.use_surprise_dt:
        config.use_surprise_dt = True
    if args.surprise_dt_scale != 0.5:
        config.surprise_dt_scale = args.surprise_dt_scale
    if args.use_bpnp:
        config.use_bpnp = True
    if args.bpnp_threshold != 0.3:
        config.bpnp_threshold = args.bpnp_threshold
    if args.use_dssu:
        config.use_dssu = True
    if args.dssu_epsilon != 0.01:
        config.dssu_epsilon = args.dssu_epsilon

    if args.use_aga:
        config.use_aga = True
    if args.aga_threshold_k != 32:
        config.aga_threshold_k = args.aga_threshold_k
    if args.aga_decay_steps != 0:
        config.aga_decay_steps = args.aga_decay_steps
    if args.aga_ternary_pull != 0.0:
        config.aga_ternary_pull = args.aga_ternary_pull
    if args.aga_lr_multiplier != 1.0:
        config.aga_lr_multiplier = args.aga_lr_multiplier

    if args.use_bbo:
        config.use_bbo = True
    if args.bbo_threshold_k != 16:
        config.bbo_threshold_k = args.bbo_threshold_k
    if args.bbo_decay_steps != 100:
        config.bbo_decay_steps = args.bbo_decay_steps
    if args.bbo_magnitude_threshold != 0.0:
        config.bbo_magnitude_threshold = args.bbo_magnitude_threshold
    if not args.bbo_two_step:
        config.bbo_two_step = False

    if args.use_slow_heads:
        config.use_slow_heads = True
    if args.slow_head_fraction != 0.25:
        config.slow_head_fraction = args.slow_head_fraction
    if args.medium_head_fraction != 0.25:
        config.medium_head_fraction = args.medium_head_fraction

    if args.use_moa:
        config.use_moa = True
    if args.moa_window_sizes:
        config.moa_window_sizes = tuple(args.moa_window_sizes)

    if args.use_caq:
        config.use_caq = True
    if args.caq_alpha != 1.0:
        config.caq_alpha = args.caq_alpha

    if args.use_head_gating:
        config.use_head_gating = True

    if args.use_sink_tokens:
        config.use_sink_tokens = True
    if args.n_sink_tokens != 1:
        config.n_sink_tokens = args.n_sink_tokens

    if args.use_ssm_checkpoint:
        config.use_ssm_checkpoint = True

    if args.ternary_aware_checkpointing:
        config.ternary_aware_checkpointing = True

    if args.compile_memory_budget is not None:
        config.compile_memory_budget = args.compile_memory_budget
    if args.activation_checkpoint_granularity != "none":
        config.activation_checkpoint_granularity = (
            args.activation_checkpoint_granularity
        )

    if args.use_phase_space_attn:
        config.use_phase_space_attn = True

    if args.use_ctri:
        config.use_ctri = True
    if args.ctri_selective_layers != "0":
        config.ctri_selective_layers = tuple(
            int(x.strip()) for x in args.ctri_selective_layers.split(",")
        )

    if args.use_recursive_ssm:
        config.use_recursive_ssm = True
    if args.recursive_ssm_iterations > 0:
        config.recursive_ssm_iterations = args.recursive_ssm_iterations

    # TADA: Ternary Algebraic Depth Amplification
    if args.use_tada:
        config.use_tada = True
    if args.tada_residual_injection is not None:
        config.tada_residual_injection = args.tada_residual_injection
    if args.tada_adaptive_k:
        config.tada_adaptive_k = True
    if args.tada_adaptive_k_max > 0:
        config.tada_adaptive_k_max = args.tada_adaptive_k_max
    if args.tada_rank_monitor_interval > 0:
        config.tada_rank_monitor_interval = args.tada_rank_monitor_interval

    if args.use_info_freeze:
        config.use_info_freeze = True
    if args.info_freeze_alpha != 1e-5:
        config.info_freeze_alpha = args.info_freeze_alpha
    if args.info_freeze_warmup != 200:
        config.info_freeze_warmup = args.info_freeze_warmup

    if args.activation_fn != "silu":
        config.activation_fn = args.activation_fn
    if args.use_bitshift_norm:
        config.use_bitshift_norm = True

    if args.adaptive_depth_exit:
        config.adaptive_depth_exit = True
    if args.adaptive_depth_epsilon != 0.01:
        config.adaptive_depth_epsilon = args.adaptive_depth_epsilon
    if args.adaptive_depth_min_iters != 2:
        config.adaptive_depth_min_iters = args.adaptive_depth_min_iters

    if args.use_ftec:
        config.use_ftec = True
    if args.ftec_n_factors != 8:
        config.ftec_n_factors = args.ftec_n_factors
    if args.ftec_sparsity != 3:
        config.ftec_sparsity = args.ftec_sparsity
    if args.ftec_pattern != "random":
        config.ftec_pattern = args.ftec_pattern
    if args.ftec_target_layers != "out_proj":
        config.ftec_target_layers = args.ftec_target_layers
    if args.ftec_no_skip:
        config.ftec_skip_connection = False

    # Feature 2: Anderson Acceleration (PATTERN A)
    if args.use_anderson_acceleration:
        config.use_anderson_acceleration = True
        config.anderson_beta = args.anderson_beta

    # Feature 3: Geometric Extrapolation (PATTERN A)
    if args.use_geometric_extrapolation:
        config.use_geometric_extrapolation = True
        config.geom_extrap_min_iters = args.geom_extrap_min_iters
        config.geom_extrap_ratio_threshold = args.geom_extrap_ratio_threshold

    # Feature 4: Episodic State (PATTERN A)
    if args.use_episodic_state:
        config.use_episodic_state = True

    # Feature 6: Per-Depth LR (PATTERN A)
    if args.use_per_depth_lr:
        config.use_per_depth_lr = True
        config.per_depth_lr_min = args.per_depth_lr_min

    # R4-A3-F6: RecursiveSSM Iteration Load-Balancing
    if args.use_recursive_load_balance:
        config.use_recursive_load_balance = True
    if args.recursive_load_balance_coeff != 0.01:
        config.recursive_load_balance_coeff = args.recursive_load_balance_coeff

    # R4-A3-F4: Self-Knowledge Distillation
    if args.use_sskd:
        config.use_sskd = True
        config.sskd_alpha = args.sskd_alpha
        config.sskd_temperature = args.sskd_temperature

    # R4-A3-F8: Dynamic Loss Weighting
    if args.use_dynamic_loss_weighting:
        config.use_dynamic_loss_weighting = True
        config.dlw_gamma = args.dlw_gamma

    # R4-A3-M1: Ternary Hamming Margin Loss
    if args.use_thm_loss:
        config.use_thm_loss = True
        config.thm_margin = args.thm_margin
        config.thm_lambda = args.thm_lambda

    # R4-A3-M2: TAIL Loss
    if args.use_tail_loss:
        config.use_tail_loss = True
        config.tail_lambda = args.tail_lambda
        config.tail_flip_margin = args.tail_flip_margin
        config.tail_num_negatives = args.tail_num_negatives

    # R4-A3-M3: Recursive Consensus Loss
    if args.use_rcl_loss:
        config.use_rcl_loss = True
        config.rcl_beta = args.rcl_beta
        config.rcl_lambda = args.rcl_lambda
        config.rcl_sample_every = args.rcl_sample_every
        config.collect_recursive_states = True

    # R4-A3-M4: Ternary Entropy Rate Loss
    if args.use_ter_loss:
        config.use_ter_loss = True
        config.ter_loss_mu = args.ter_loss_mu
        config.ter_noise_var = args.ter_noise_var

    # R4-A3-M5: Gradient Consensus Polarization
    if args.use_gcp:
        config.use_gcp = True
        config.gcp_alpha = args.gcp_alpha

    # R4-A3-M6: Recurrence-Aware Flip Utility
    if args.use_rafu:
        config.use_rafu = True
        config.rafu_threshold = args.rafu_threshold
        config.rafu_interval = args.rafu_interval
        config.collect_recursive_states = True

    # Feature 9: UCB-Bandit (PATTERN A)
    if args.use_ucb_bandit:
        config.use_ucb_bandit = True
        config.ucb_min_iterations = args.ucb_min_iterations

    if args.dataloader_num_workers is not None:
        config.dataloader_num_workers = args.dataloader_num_workers
    if args.dataloader_prefetch_factor is not None:
        config.dataloader_prefetch_factor = args.dataloader_prefetch_factor

    if args.use_cuda_prefetch:
        config.use_cuda_prefetch = True
    if args.auto_batch_size:
        config.auto_batch_size = True
    if args.vram_budget_mb != 10000:
        config.vram_budget_mb = args.vram_budget_mb

    if args.use_gradient_sign_consistency:
        config.use_gradient_sign_consistency = True
    if args.grad_sign_window != 8:
        config.grad_sign_window = args.grad_sign_window
    if args.grad_sign_threshold != 6:
        config.grad_sign_threshold = args.grad_sign_threshold

    # A2-F6: Layerwise Gauss-Newton Preconditioning
    if args.use_fisher_preconditioning:
        config.use_fisher_preconditioning = True
    if args.fisher_damping != 1e-4:
        config.fisher_damping = args.fisher_damping

    # A2-F7: SignSGD
    if args.use_sign_sgd:
        config.use_sign_sgd = True
    if args.sign_sgd_momentum != 0.9:
        config.sign_sgd_momentum = args.sign_sgd_momentum

    # A2-M7: Consensus-Filtered-Layer-Gradients
    if args.use_consensus_gradient_filter:
        config.use_consensus_gradient_filter = True
    if args.consensus_alpha != 0.5:
        config.consensus_alpha = args.consensus_alpha

    # A2-F8: Hessian-Aware Bit Allocation
    if args.use_adaptive_quantization:
        config.use_adaptive_quantization = True
    if args.adaptive_quant_sensitivity_threshold != 0.1:
        config.adaptive_quant_sensitivity_threshold = (
            args.adaptive_quant_sensitivity_threshold
        )
    if args.adaptive_quant_interval != 500:
        config.adaptive_quant_interval = args.adaptive_quant_interval

    # F7: Multi-Token Prediction
    if args.mtp_num_heads > 0:
        config.mtp_num_heads = args.mtp_num_heads
        config.mtp_lambda_main = args.mtp_lambda_main
        if args.mtp_lambda_aux is not None:
            config.mtp_lambda_aux = list(args.mtp_lambda_aux)

    # A2-M5: UCB-Bandit-Ternary-Training
    if args.use_ucb_ternary_training:
        config.use_ucb_ternary_training = True
    if args.ucb_ternary_exploration_c != 1.0:
        config.ucb_ternary_exploration_c = args.ucb_ternary_exploration_c

    # A2-M6: FSCR Flip Safety/Confidence Ratio
    if args.use_fscr_flip_gate:
        config.use_fscr_flip_gate = True
    if args.fscr_threshold != 2.0:
        config.fscr_threshold = args.fscr_threshold

    # P1: Spectral-Cached SSM
    if args.use_spectral_ssm:
        config.use_spectral_ssm = True

    # P2: SPSA Ternary Training
    if args.use_spsa_training:
        config.use_spsa_training = True
    if args.spsa_perturbation_scale != 0.1:
        config.spsa_perturbation_scale = args.spsa_perturbation_scale
    if args.spsa_lr != 0.01:
        config.spsa_lr = args.spsa_lr

    # P3: Integer-Rank Loss Training
    if args.use_rank_margin_loss:
        config.use_rank_margin_loss = True
    if args.rank_margin != 1.0:
        config.rank_margin = args.rank_margin
    if args.rank_num_negatives != 64:
        config.rank_num_negatives = args.rank_num_negatives

    # P5: Whittle-Index Scheduling
    if args.use_whittle_index:
        config.use_whittle_index = True
    if args.whittle_update_fraction != 0.1:
        config.whittle_update_fraction = args.whittle_update_fraction

    # R4-A3-M7: Information-Bottleneck Layerwise Loss
    if args.use_ibll:
        config.use_ibll = True
        config.ibll_beta = args.ibll_beta
        config.ibll_gamma = args.ibll_gamma
        config.ibll_eps = args.ibll_eps
        config.ibll_lambda = args.ibll_lambda
        config.ibll_sample_every = args.ibll_sample_every
        config.collect_recursive_states = True

    # R4-A3-M8: Bayesian Flip Accumulator
    if args.use_bfa:
        config.use_bfa = True
        config.bfa_rho = args.bfa_rho
        config.bfa_tau = args.bfa_tau
        config.bfa_theta = args.bfa_theta
        config.bfa_interval = args.bfa_interval

    # R4-A3-P1: Minimax Depth Equilibrium
    if args.use_mde:
        config.use_mde = True
        config.mde_eta = args.mde_eta
        config.mde_lambda = args.mde_lambda
        config.mde_sample_every = args.mde_sample_every
        config.collect_recursive_states = True

    # R4-A3-P2: Ternary Error-Correcting Weight Codes
    if args.use_tecwc:
        config.use_tecwc = True
        config.tecwc_code_rate = args.tecwc_code_rate
        config.tecwc_lambda = args.tecwc_lambda

    # R4-A3-P3: Spectral Surrogate Objective
    if args.use_sso:
        config.use_sso = True
        config.sso_rho_target = args.sso_rho_target
        config.sso_alpha = args.sso_alpha
        config.sso_beta = args.sso_beta
        config.sso_interval = args.sso_interval

    # R4-A3-P4: Spin-Glass Annealing
    if args.use_sgapd:
        config.use_sgapd = True
        config.sgapd_initial_temp = args.sgapd_initial_temp
        config.sgapd_min_temp = args.sgapd_min_temp
        config.sgapd_decay = args.sgapd_decay
        config.sgapd_sweeps = args.sgapd_sweeps
        config.sgapd_interval = args.sgapd_interval

    # R4-A3-P5: Ergodic Trajectory Optimizer
    if args.use_ergodic:
        config.use_ergodic = True
        config.ergodic_alpha = args.ergodic_alpha
        config.ergodic_rho_target = args.ergodic_rho_target
        config.ergodic_flip_fraction = args.ergodic_flip_fraction
        config.ergodic_interval = args.ergodic_interval

    # R4-A3-P6: Minimax Depth Fairness
    if args.use_minimax_depth:
        config.use_minimax_depth = True
        config.minimax_tau = args.minimax_tau
        config.minimax_hard = args.minimax_hard
        config.minimax_lambda = args.minimax_lambda
        config.minimax_sample_every = args.minimax_sample_every
        config.collect_recursive_states = True

    # R4-A3-P7: Koopman Spectral Shaping
    if args.use_koopman:
        config.use_koopman = True
        config.koopman_long_target = args.koopman_long_target
        config.koopman_fast_target = args.koopman_fast_target
        config.koopman_lambda = args.koopman_lambda
        config.koopman_interval = args.koopman_interval
        config.koopman_max_dim = args.koopman_max_dim
        config.collect_recursive_states = True

    # R4-A3-P8: TD Depth Learning
    if args.use_td_dl:
        config.use_td_dl = True
        config.td_lambda = args.td_lambda_param
        config.td_dl_weight = args.td_dl_weight
        config.td_sample_every = args.td_sample_every
        config.collect_recursive_states = True

    # R4-A4: Checkpoint & Fault Tolerance overrides
    if args.use_gradient_monitor:
        config.use_gradient_monitor = True
    if args.gradient_monitor_window != 100:
        config.gradient_monitor_window = args.gradient_monitor_window
    if args.use_ram_checkpoint:
        config.use_ram_checkpoint = True
    if args.no_ram_checkpoint_disk_backup:
        config.ram_checkpoint_disk_backup = False
    if args.use_deterministic_mode:
        config.use_deterministic_mode = True
    if args.use_ternary_checkpoint_compression:
        config.use_ternary_checkpoint_compression = True
    if args.use_crg:
        config.use_crg = True
    if args.crg_alpha_init != 3.0:
        config.crg_alpha_init = args.crg_alpha_init

    # R4-A4 Batch 2: Checkpoint compression
    if args.use_incremental_checkpoints:
        config.use_incremental_checkpoints = True
    if args.delta_full_interval != 1000:
        config.delta_full_interval = args.delta_full_interval
    if args.use_tiered_checkpoint:
        config.use_tiered_checkpoint = True
    if args.tiered_tier2_interval != 50:
        config.tiered_tier2_interval = args.tiered_tier2_interval
    if args.tiered_tier3_interval != 500:
        config.tiered_tier3_interval = args.tiered_tier3_interval
    if args.use_optimizer_compression:
        config.use_optimizer_compression = True
    if args.use_fiber_shadow_compression:
        config.use_fiber_shadow_compression = True
    if args.use_tdss_compression:
        config.use_tdss_compression = True

    # R4-A4 Batch 3: Fault Tolerance & Resilience
    if args.use_anomaly_detector:
        config.use_anomaly_detector = True
    if args.anomaly_z_threshold != 3.0:
        config.anomaly_z_threshold = args.anomaly_z_threshold
    if args.use_async_checkpoint:
        config.use_async_checkpoint = True
    if args.use_qmm_regularizer:
        config.use_qmm_regularizer = True
    if args.qmm_lambda != 0.01:
        config.qmm_lambda = args.qmm_lambda
    if args.qmm_margin_target != 0.3:
        config.qmm_margin_target = args.qmm_margin_target
    if args.use_ternary_inoculation:
        config.use_ternary_inoculation = True
    if args.inoculation_flip_prob != 0.01:
        config.inoculation_flip_prob = args.inoculation_flip_prob
    if args.inoculation_flip_fraction != 0.001:
        config.inoculation_flip_fraction = args.inoculation_flip_fraction
    if args.use_trajectory_verification:
        config.use_trajectory_verification = True
    if args.trajectory_verify_interval != 50:
        config.trajectory_verify_interval = args.trajectory_verify_interval

    # R4-A4 Batch 4: Training Archaeology
    if args.use_delta_chronicle:
        config.use_delta_chronicle = True
    if args.delta_chronicle_max_mb != 500.0:
        config.delta_chronicle_max_mb = args.delta_chronicle_max_mb
    if args.use_flip_influence:
        config.use_flip_influence = True
    if args.flip_influence_top_k != 100:
        config.flip_influence_top_k = args.flip_influence_top_k
    if args.use_entropic_delta:
        config.use_entropic_delta = True

    # R4-A4 Batch 5: Advanced/Speculative
    if args.use_optimizer_reconstruction:
        config.use_optimizer_reconstruction = True
    if args.optimizer_reconstruction_steps != 50:
        config.optimizer_reconstruction_steps = args.optimizer_reconstruction_steps
    if args.use_trajectory_replay:
        config.use_trajectory_replay = True
    if args.trajectory_anchor_interval != 5000:
        config.trajectory_anchor_interval = args.trajectory_anchor_interval
    if args.use_adaptive_checkpoint:
        config.use_adaptive_checkpoint = True
    if args.use_iaf_optimizer:
        config.use_iaf_optimizer = True
    if args.iaf_threshold != 1.0:
        config.iaf_threshold = args.iaf_threshold
    if args.iaf_leak != 0.99:
        config.iaf_leak = args.iaf_leak
    if args.use_neural_codec:
        config.use_neural_codec = True
    if args.use_gf3_coding:
        config.use_gf3_coding = True
    if args.use_ccsr:
        config.use_ccsr = True

    # ---- R4-A5: Portable Kernel Algorithms ----

    # Batch 2: Storage & Memory Layout
    if args.use_quinary_packing:
        config.use_quinary_packing = True

    # Batch 3: Pattern LUT & SIMD
    if args.use_block_pattern_lut:
        config.use_block_pattern_lut = True
    if args.block_pattern_k != 4:
        config.block_pattern_k = args.block_pattern_k

    # Batch 4: Recursive Depth Optimization
    if args.use_rtdp:
        config.use_rtdp = True
    if args.rtdp_threshold != 0.01:
        config.rtdp_threshold = args.rtdp_threshold
    if args.rtdp_diagnostics:
        config.rtdp_diagnostics = True
    if args.use_ternary_circuit_compile:
        config.use_ternary_circuit_compile = True
    if args.circuit_recompile_interval != 500:
        config.circuit_compile_recompile_interval = args.circuit_recompile_interval
    if args.use_adaptive_dataflow:
        config.use_adaptive_dataflow = True

    # Batch 5: SSM Scan Optimization (BLOCKED by TSCS)
    if args.use_scan_template:
        config.use_scan_template = True
    if args.use_sparse_scan:
        config.use_sparse_scan = True
    if args.use_parity_dual_stream:
        config.use_parity_dual_stream = True
    if args.use_absorbing_pruning:
        config.use_absorbing_pruning = True

    # Batch 6: Deployment Targets
    if args.use_ternary_binary_decomp:
        config.use_ternary_binary_decomp = True
    if args.ternary_binary_block_size != 4:
        config.ternary_binary_block_size = args.ternary_binary_block_size

    # Batch 7: Advanced Explorations
    if args.use_gf3_factorization:
        config.use_gf3_factorization = True
    if args.gf3_min_speedup != 1.5:
        config.gf3_min_speedup = args.gf3_min_speedup
    if args.use_butterfly_factorization:
        config.use_butterfly_factorization = True
    if args.butterfly_target_layers != "all":
        config.butterfly_target_layers = args.butterfly_target_layers
    if args.use_t3m:
        config.use_t3m = True
    if args.t3m_activation_threshold != 0.5:
        config.t3m_activation_threshold = args.t3m_activation_threshold
    if args.use_parallel_depth_scan:
        config.use_parallel_depth_scan = True
    if args.parallel_depth_warmup != 3:
        config.parallel_depth_warmup = args.parallel_depth_warmup
    if args.parallel_depth_n_probes != 16:
        config.parallel_depth_n_probes = args.parallel_depth_n_probes

    # ---- R4-A6: Memory Layout & Tensor Format ----

    # Batch 1: Commitment Detection
    if args.use_commitment_detector:
        config.use_commitment_detector = True
    if args.commitment_threshold != 0.3:
        config.commitment_threshold = args.commitment_threshold
    if args.metabolic_window != 100:
        config.metabolic_window = args.metabolic_window
    if args.commitment_block_size != 128:
        config.commitment_block_size = args.commitment_block_size
    if args.use_metabolic_probe:
        config.use_metabolic_probe = True

    # Batch 2: Optimizer State Eviction
    if args.use_graduated_eviction:
        config.use_graduated_eviction = True
    if args.eviction_tier_thresholds != "0.1,0.3,0.5":
        config.eviction_tier_thresholds = args.eviction_tier_thresholds

    # Batch 3: Sparse Update & Drift Detection
    if args.use_cusum_drift:
        config.use_cusum_drift = True
    if args.cusum_threshold != 5.0:
        config.cusum_threshold = args.cusum_threshold
    if args.cusum_sensitivity != 0.5:
        config.cusum_sensitivity = args.cusum_sensitivity
    if args.use_sparse_updates:
        config.use_sparse_updates = True
    if args.sparse_update_committed_interval != 8:
        config.sparse_update_committed_interval = args.sparse_update_committed_interval

    # Batch 4: Compute Savings
    if args.use_frozen_ternary_op:
        config.use_frozen_ternary_op = True
    if args.frozen_cache_max_mb != 100:
        config.frozen_cache_max_mb = args.frozen_cache_max_mb
    if args.use_recycling_pool:
        config.use_recycling_pool = True
    if args.recycling_pool_max_mb != 256:
        config.recycling_pool_max_mb = args.recycling_pool_max_mb
    if args.use_selective_grad_subsample:
        config.use_selective_grad_subsample = True
    if args.grad_subsample_committed != 0.1:
        config.grad_subsample_committed = args.grad_subsample_committed

    # Batch 5: Training Paradigm Shifts
    if args.use_bifurcated_optimizer:
        config.use_bifurcated_optimizer = True
    if args.committed_lr_ratio != 0.1:
        config.committed_lr_ratio = args.committed_lr_ratio
    if args.use_commitment_curriculum:
        config.use_commitment_curriculum = True
    if args.curriculum_target_ratio != 0.7:
        config.curriculum_target_ratio = args.curriculum_target_ratio
    if args.curriculum_warmup_fraction != 0.2:
        config.curriculum_warmup_fraction = args.curriculum_warmup_fraction
    if args.use_thermal_memory:
        config.use_thermal_memory = True

    # Batch 6: Explore
    if args.use_ternary_layout_opt:
        config.use_ternary_layout_opt = True
    if args.layout_alignment != 64:
        config.layout_alignment = args.layout_alignment
    if args.use_predictive_freezer:
        config.use_predictive_freezer = True
    if args.prediction_horizon != 500:
        config.prediction_horizon = args.prediction_horizon
    if args.prediction_confidence != 0.9:
        config.prediction_confidence = args.prediction_confidence
    if args.use_tensor_defrag:
        config.use_tensor_defrag = True
    if args.defrag_interval != 1000:
        config.defrag_interval = args.defrag_interval
    # Batch AN: R17-A findings
    if args.use_mdl_loss:
        config.use_mdl_loss = True
    if args.mdl_alpha != 1e-4:
        config.mdl_alpha = args.mdl_alpha
    if args.use_phase_detection:
        config.use_phase_detection = True
    if args.phase_window_size != 50:
        config.phase_window_size = args.phase_window_size
    if args.n_gist_tokens > 0:
        config.n_gist_tokens = args.n_gist_tokens
    if args.use_cohom_loss:
        config.use_cohom_loss = True
    if args.cohom_alpha != 1e-4:
        config.cohom_alpha = args.cohom_alpha
    if args.ms_ssm_n_scales != 1:
        config.ms_ssm_n_scales = args.ms_ssm_n_scales
    if args.ms_ssm_log_min != -4.0:
        config.ms_ssm_log_min = args.ms_ssm_log_min
    if args.ms_ssm_log_max != 0.0:
        config.ms_ssm_log_max = args.ms_ssm_log_max

    # Batch AO: R17-B findings
    if args.use_wasserstein_ste:
        config.use_wasserstein_ste = True
    if args.use_event_encoding:
        config.use_event_encoding = True
    if args.use_ergodic_loss:
        config.use_ergodic_loss = True
    if args.ergodic_loss_alpha != 1e-3:
        config.ergodic_loss_alpha = args.ergodic_loss_alpha
    if args.use_ssm_annealing:
        config.use_ssm_annealing = True
    if args.ssm_anneal_warmup != 1000:
        config.ssm_anneal_warmup = args.ssm_anneal_warmup
    if args.ssm_anneal_log_max != 4.0:
        config.ssm_anneal_log_max = args.ssm_anneal_log_max
    if args.use_masked_pred:
        config.use_masked_pred = True
    if args.masked_pred_ratio != 0.15:
        config.masked_pred_ratio = args.masked_pred_ratio
    if args.masked_pred_lambda != 0.1:
        config.masked_pred_lambda = args.masked_pred_lambda
    if args.use_temporal_contrastive:
        config.use_temporal_contrastive = True
    if args.temporal_contrastive_k != 4:
        config.temporal_contrastive_k = args.temporal_contrastive_k
    if args.temporal_contrastive_lambda != 0.01:
        config.temporal_contrastive_lambda = args.temporal_contrastive_lambda

    # Batch AP: R17-C findings
    if args.use_sigma_gate: config.use_sigma_gate = True
    if args.sigma_gate_threshold != 0.5: config.sigma_gate_threshold = args.sigma_gate_threshold
    if args.sigma_gate_baseline_beta != 0.99: config.sigma_gate_baseline_beta = args.sigma_gate_baseline_beta
    if args.use_etc: config.use_etc = True
    if args.etc_beta != 0.9: config.etc_beta = args.etc_beta
    if args.etc_tau != 0.8: config.etc_tau = args.etc_tau
    if args.etc_crystallize_step != 500: config.etc_crystallize_step = args.etc_crystallize_step
    if args.use_beb: config.use_beb = True
    if args.use_ema_replay: config.use_ema_replay = True
    if args.ema_replay_lambda != 0.05: config.ema_replay_lambda = args.ema_replay_lambda
    if args.use_twa: config.use_twa = True
    if args.twa_temp != 1.0: config.twa_temp = args.twa_temp
    if args.twa_anneal_steps != 2000: config.twa_anneal_steps = args.twa_anneal_steps
    if args.use_scme: config.use_scme = True
    if args.scme_alpha != 0.95: config.scme_alpha = args.scme_alpha
    if args.use_si: config.use_si = True
    if args.use_agem: config.use_agem = True
    if args.agem_buffer_size != 256: config.agem_buffer_size = args.agem_buffer_size
    if args.use_ewc: config.use_ewc = True
    if args.ewc_lambda != 0.01: config.ewc_lambda = args.ewc_lambda
    if args.use_deltanet_ssm: config.use_deltanet_ssm = True
    if args.use_ttt_fast_weights: config.use_ttt_fast_weights = True
    if args.use_predictive_coding: config.use_predictive_coding = True
    if args.use_dual_hebbian: config.use_dual_hebbian = True
    if args.use_fast_weights: config.use_fast_weights = True
    if args.use_per_weight_temp: config.use_per_weight_temp = True
    if args.use_lota_qaf: config.use_lota_qaf = True
    if args.use_tpm: config.use_tpm = True
    if args.use_csc: config.use_csc = True
    if args.use_attractor_basin: config.use_attractor_basin = True

    # Batch AQ: R17-D findings
    if args.use_spectral_band_init: config.use_spectral_band_init = True
    if args.use_dead_dim_pruning: config.use_dead_dim_pruning = True
    if args.dead_dim_threshold != 0.9: config.dead_dim_threshold = args.dead_dim_threshold
    if args.dead_dim_prune_step != 1000: config.dead_dim_prune_step = args.dead_dim_prune_step
    if args.use_erank_fixedpoint: config.use_erank_fixedpoint = True
    if args.erank_check_interval != 500: config.erank_check_interval = args.erank_check_interval
    if args.erank_grow_threshold != 0.8: config.erank_grow_threshold = args.erank_grow_threshold
    if args.erank_prune_threshold != 0.2: config.erank_prune_threshold = args.erank_prune_threshold
    if args.erank_delta != 4: config.erank_delta = args.erank_delta
    if args.use_liquid_s4: config.use_liquid_s4 = True
    if args.use_depth_plasticity: config.use_depth_plasticity = True
    if args.depth_plasticity_window != 8: config.depth_plasticity_window = args.depth_plasticity_window
    if args.use_mod_routing: config.use_mod_routing = True
    if args.mod_top_k_fraction != 0.5: config.mod_top_k_fraction = args.mod_top_k_fraction
    if args.use_expert_choice_moe: config.use_expert_choice_moe = True
    if args.expert_choice_n_experts != 4: config.expert_choice_n_experts = args.expert_choice_n_experts
    if args.use_hypernetwork_thresholds: config.use_hypernetwork_thresholds = True
    if args.use_cgls_stacking: config.use_cgls_stacking = True
    if args.cgls_insert_step != 5000: config.cgls_insert_step = args.cgls_insert_step
    if args.use_topo_insertion: config.use_topo_insertion = True
    if args.use_fisher_arch_flow: config.use_fisher_arch_flow = True
    if args.fisher_arch_interval != 1000: config.fisher_arch_interval = args.fisher_arch_interval
    if args.use_rg_flow: config.use_rg_flow = True
    if args.rg_flow_threshold != 0.1: config.rg_flow_threshold = args.rg_flow_threshold
    if args.use_committed_inhibition: config.use_committed_inhibition = True
    if args.use_ho_kalman: config.use_ho_kalman = True
    if args.ho_kalman_lag_max != 64: config.ho_kalman_lag_max = args.ho_kalman_lag_max
    if args.ho_kalman_calib_tokens != 1000: config.ho_kalman_calib_tokens = args.ho_kalman_calib_tokens
    if args.use_automaton_collapse: config.use_automaton_collapse = True
    if args.use_toeplitz_ca: config.use_toeplitz_ca = True
    if args.toeplitz_kernel_size != 7: config.toeplitz_kernel_size = args.toeplitz_kernel_size
    if args.use_persistent_homology: config.use_persistent_homology = True
    if args.use_supernet_nas: config.use_supernet_nas = True
    if args.use_neural_ode_weights: config.use_neural_ode_weights = True
    if args.use_foundation_hypernetwork: config.use_foundation_hypernetwork = True

    if args.use_adaponder: config.use_adaponder = True
    if args.adaponder_halt_threshold != 0.5: config.adaponder_halt_threshold = args.adaponder_halt_threshold
    if args.adaponder_max_iters != 3: config.adaponder_max_iters = args.adaponder_max_iters
    # AR-2 use_dssu already mapped in earlier batch (F3)
    if args.use_sssp: config.use_sssp = True
    if args.sssp_stability_threshold != 0.8: config.sssp_stability_threshold = args.sssp_stability_threshold
    if args.use_scms: config.use_scms = True
    if args.scms_passive_threshold != 0.01: config.scms_passive_threshold = args.scms_passive_threshold
    if args.use_hash_ternary_routing: config.use_hash_ternary_routing = True
    if args.hash_n_groups != 8: config.hash_n_groups = args.hash_n_groups
    if args.use_input_entropy_bypass: config.use_input_entropy_bypass = True
    if args.input_entropy_lambda != 0.001: config.input_entropy_lambda = args.input_entropy_lambda
    if args.use_structured_zero_cols: config.use_structured_zero_cols = True
    if args.structured_zero_lambda != 0.001: config.structured_zero_lambda = args.structured_zero_lambda
    if args.structured_zero_threshold != 0.02: config.structured_zero_threshold = args.structured_zero_threshold
    if args.use_thdv: config.use_thdv = True
    if args.thdv_kl_weight != 0.01: config.thdv_kl_weight = args.thdv_kl_weight
    if args.use_layerskip: config.use_layerskip = True
    if args.use_teal_sparsity: config.use_teal_sparsity = True
    if args.use_specee: config.use_specee = True
    if args.use_mixture_of_recursions: config.use_mixture_of_recursions = True
    if args.use_evnn_threshold: config.use_evnn_threshold = True
    if args.use_ternary_speculative_state: config.use_ternary_speculative_state = True
    if args.use_csbc: config.use_csbc = True

    # Batch AS config mappings — R17-F
    if args.use_gvste: config.use_gvste = True
    if args.use_ada_gate: config.use_ada_gate = True
    if args.ada_coherence_threshold != 0.15: config.ada_coherence_threshold = args.ada_coherence_threshold
    if args.use_mi_s1: config.use_mi_s1 = True
    if args.mi_s1_entropy_threshold != 1.0: config.mi_s1_entropy_threshold = args.mi_s1_entropy_threshold
    if args.use_seee: config.use_seee = True
    if args.seee_entropy_threshold != 1.0: config.seee_entropy_threshold = args.seee_entropy_threshold
    if args.use_tug: config.use_tug = True
    if args.tug_zero_frac_threshold != 0.7: config.tug_zero_frac_threshold = args.tug_zero_frac_threshold
    if args.use_sgb: config.use_sgb = True
    if args.sgb_surprisal_threshold != 3.0: config.sgb_surprisal_threshold = args.sgb_surprisal_threshold
    if args.use_tcsm: config.use_tcsm = True
    if args.tcsm_stable_steps != 200: config.tcsm_stable_steps = args.tcsm_stable_steps
    if args.use_ergodic_mixing: config.use_ergodic_mixing = True
    if args.ergodic_lambda_threshold != 0.7: config.ergodic_lambda_threshold = args.ergodic_lambda_threshold
    if args.use_vff_routing: config.use_vff_routing = True
    if args.use_fast_slow_ssm: config.use_fast_slow_ssm = True
    if args.fast_slow_factor != 4.0: config.fast_slow_factor = args.fast_slow_factor
    if args.use_lyapunov_basin: config.use_lyapunov_basin = True
    if args.lyapunov_basin_threshold != 1.0: config.lyapunov_basin_threshold = args.lyapunov_basin_threshold
    if args.use_fsw: config.use_fsw = True
    if args.fsw_lambda != 0.99: config.fsw_lambda = args.fsw_lambda
    if args.fsw_eta != 0.01: config.fsw_eta = args.fsw_eta
    if args.use_dplr: config.use_dplr = True
    if args.dplr_retention_threshold != 0.5: config.dplr_retention_threshold = args.dplr_retention_threshold
    if args.use_sprt: config.use_sprt = True
    if args.sprt_log_threshold != 2.0: config.sprt_log_threshold = args.sprt_log_threshold
    if args.use_act: config.use_act = True
    if args.use_pc2: config.use_pc2 = True
    if args.pc2_beta != 0.1: config.pc2_beta = args.pc2_beta
    if args.use_bns2: config.use_bns2 = True
    if args.bns2_norm_threshold != 0.1: config.bns2_norm_threshold = args.bns2_norm_threshold
    if args.use_tsm: config.use_tsm = True
    if args.tsm_cache_k != 8: config.tsm_cache_k = args.tsm_cache_k
    if args.use_cls_hopfield: config.use_cls_hopfield = True
    if args.use_ponder_ssm: config.use_ponder_ssm = True
    if args.ponder_lambda != 0.01: config.ponder_lambda = args.ponder_lambda

    # Batch AT config mappings — R17-G
    if args.use_cls_timescale: config.use_cls_timescale = True
    if args.cls_timescale_alpha != 0.99: config.cls_timescale_alpha = args.cls_timescale_alpha
    if args.use_smr_memory_tokens: config.use_smr_memory_tokens = True
    if args.smr_num_tokens != 4: config.smr_num_tokens = args.smr_num_tokens
    if args.use_dopamine_rpe_gate: config.use_dopamine_rpe_gate = True
    if args.dopamine_ema_alpha != 0.99: config.dopamine_ema_alpha = args.dopamine_ema_alpha
    if args.use_dt_surprise_gate: config.use_dt_surprise_gate = True
    if args.use_holographic_init: config.use_holographic_init = True
    if args.use_nrem_rem: config.use_nrem_rem = True
    if args.nrem_attractor_eps != 0.1: config.nrem_attractor_eps = args.nrem_attractor_eps
    if args.use_hippocampal_sparsity: config.use_hippocampal_sparsity = True
    if args.hippocampal_sparsity_k != 0.05: config.hippocampal_sparsity_k = args.hippocampal_sparsity_k
    if args.use_gsemm_init: config.use_gsemm_init = True
    if args.use_delta_rule: config.use_delta_rule = True
    if args.delta_rule_beta != 0.1: config.delta_rule_beta = args.delta_rule_beta
    if args.use_ior_committed: config.use_ior_committed = True
    if args.ior_high_alpha != 0.01: config.ior_high_alpha = args.ior_high_alpha
    if args.ior_low_alpha != 0.1: config.ior_low_alpha = args.ior_low_alpha
    if args.use_pfc_boundary_gate: config.use_pfc_boundary_gate = True
    if args.use_synaptic_tagging: config.use_synaptic_tagging = True
    if args.stc_theta_c != 0.01: config.stc_theta_c = args.stc_theta_c
    if args.use_reconsolidation: config.use_reconsolidation = True
    if args.use_offline_consolidation: config.use_offline_consolidation = True
    if args.offline_consolidation_n != 10: config.offline_consolidation_n = args.offline_consolidation_n
    if args.offline_consolidation_magnitude_scale != 1.1: config.offline_consolidation_magnitude_scale = args.offline_consolidation_magnitude_scale
    if args.use_engram_orthogonality: config.use_engram_orthogonality = True
    if args.engram_lambda != 0.01: config.engram_lambda = args.engram_lambda
    if args.engram_penalty_interval != 50: config.engram_penalty_interval = args.engram_penalty_interval
    if args.use_fisher_consolidation: config.use_fisher_consolidation = True
    if args.use_hopfield_threshold: config.use_hopfield_threshold = True
    if args.use_stc_markov: config.use_stc_markov = True
    if args.stc_k != 5: config.stc_K = args.stc_k
    if args.use_two_step_displacement: config.use_two_step_displacement = True
    if args.use_stdp_sign_conv: config.use_stdp_sign_conv = True
    if args.stdp_blend_alpha != 0.3: config.stdp_blend_alpha = args.stdp_blend_alpha

    # Batch AU config mappings — R17-H
    if args.use_slssu: config.use_slssu = True
    if args.use_error_annealing: config.use_error_annealing = True
    if args.error_annealing_alpha != 0.1: config.error_annealing_alpha = args.error_annealing_alpha
    if args.use_shadow_hamiltonian: config.use_shadow_hamiltonian = True
    if args.shadow_h_temperature != 1.0: config.shadow_h_temperature = args.shadow_h_temperature
    if args.use_maxent_gibbs: config.use_maxent_gibbs = True
    if args.maxent_T != 1.0: config.maxent_T = args.maxent_T
    if args.maxent_target_sparsity != 0.5: config.maxent_target_sparsity = args.maxent_target_sparsity
    if args.use_kmc_firing: config.use_kmc_firing = True
    if args.kmc_top_k != 10: config.kmc_top_k = args.kmc_top_k
    if args.use_rg_freeze: config.use_rg_freeze = True
    if args.rg_freeze_threshold != 0.1: config.rg_freeze_threshold = args.rg_freeze_threshold
    if args.use_nose_hoover: config.use_nose_hoover = True
    if args.nose_hoover_target_flip != 0.3: config.nose_hoover_target_flip = args.nose_hoover_target_flip
    if args.use_rsb_monitor: config.use_rsb_monitor = True
    if args.use_ising_gate: config.use_ising_gate = True
    if args.use_fpu_recurrence: config.use_fpu_recurrence = True
    if args.use_percolation_mask: config.use_percolation_mask = True
    if args.percolation_threshold != 0.5: config.percolation_threshold = args.percolation_threshold
    if args.use_strang_splitting: config.use_strang_splitting = True

    # --- Batch AV: R17-I Native Ternary Software Stack ---
    config.use_av1_termac_probe = args.use_av1_termac_probe
    config.use_av2_packing_probe = args.use_av2_packing_probe
    config.use_av3_sparsity_probe = args.use_av3_sparsity_probe
    config.use_av4_xor_probe = args.use_av4_xor_probe
    config.use_av5_asym_threshold_probe = args.use_av5_asym_threshold_probe
    config.use_av6_elut_probe = args.use_av6_elut_probe
    config.use_av7_lut_layout_probe = args.use_av7_lut_layout_probe
    config.use_av8_sparse_nnz_probe = args.use_av8_sparse_nnz_probe
    config.use_av9_llvm_probe = args.use_av9_llvm_probe
    config.use_av10_cache_block_probe = args.use_av10_cache_block_probe
    config.use_av11_reversible_ssm_probe = args.use_av11_reversible_ssm_probe
    config.use_av12_circulant_probe = args.use_av12_circulant_probe
    config.use_av13_ca_rule_probe = args.use_av13_ca_rule_probe
    config.use_av14_mds_probe = args.use_av14_mds_probe
    config.use_av15_tree_reduce_probe = args.use_av15_tree_reduce_probe
    config.use_av16_gf3_snf_probe = args.use_av16_gf3_snf_probe
    config.use_av17_iott = args.use_av17_iott
    if args.iott_use_as_threshold: config.iott_use_as_threshold = True
    config.use_av18_cayley_hamilton_probe = args.use_av18_cayley_hamilton_probe
    config.use_av19_coord_descent_b_probe = args.use_av19_coord_descent_b_probe
    config.use_av20_base3_probe = args.use_av20_base3_probe

    # --- Batch AW: R17-J Novel Mathematics ---
    config.use_aw1_tropical_probe = args.use_aw1_tropical_probe
    config.use_aw2_sheaf_energy_probe = args.use_aw2_sheaf_energy_probe
    config.use_aw3_neural_persistence_probe = args.use_aw3_neural_persistence_probe
    config.use_aw4_dsst_functor_probe = args.use_aw4_dsst_functor_probe
    config.use_aw5_hyperoctahedral_probe = args.use_aw5_hyperoctahedral_probe
    config.use_aw6_fim_probe = args.use_aw6_fim_probe
    config.use_aw7_pac_bayes_probe = args.use_aw7_pac_bayes_probe
    config.use_aw8_sinkhorn_probe = args.use_aw8_sinkhorn_probe
    if args.use_aw9_path_sig_init: config.use_aw9_path_sig_init = True
    config.use_aw10_free_prob_probe = args.use_aw10_free_prob_probe
    config.use_aw11_bp_threshold_probe = args.use_aw11_bp_threshold_probe
    config.use_aw12_hyperreal_ste = args.use_aw12_hyperreal_ste
    config.use_aw13_lz_probe = args.use_aw13_lz_probe
    if args.use_aw14_kleene_init: config.use_aw14_kleene_init = True
    config.use_aw15_renyi_probe = args.use_aw15_renyi_probe
    config.use_aw16_clifford_probe = args.use_aw16_clifford_probe
    config.use_aw17_ncg_probe = args.use_aw17_ncg_probe
    if args.use_aw18_tarski_init: config.use_aw18_tarski_init = True
    config.use_aw19_mdl_reward_probe = args.use_aw19_mdl_reward_probe
    config.use_aw20_hott_radius_probe = args.use_aw20_hott_radius_probe

    # AX: Batch AX — R17-K Multi-Dimensional Evaluation
    config.use_ax01_bbh_valley_probe = args.use_ax01_bbh_valley_probe
    config.use_ax02_think_bench_probe = args.use_ax02_think_bench_probe
    config.use_ax03_cognitive_profile_probe = args.use_ax03_cognitive_profile_probe
    config.use_ax04_metric_pref_probe = args.use_ax04_metric_pref_probe
    config.use_ax05_scaling_ceiling_probe = args.use_ax05_scaling_ceiling_probe
    config.use_ax06_power_law_probe = args.use_ax06_power_law_probe
    config.use_ax07_loss2loss_probe = args.use_ax07_loss2loss_probe
    config.use_ax08_pareto_probe = args.use_ax08_pareto_probe
    config.use_ax09_ternary_health_probe = args.use_ax09_ternary_health_probe
    config.use_ax10_compute_efficiency_probe = args.use_ax10_compute_efficiency_probe
    config.use_ax11_channel_capacity_probe = args.use_ax11_channel_capacity_probe
    config.use_ax12_structural_win_probe = args.use_ax12_structural_win_probe
    config.use_ax13_readiness_score_probe = args.use_ax13_readiness_score_probe
    config.use_ax14_ssm_attractor_probe = args.use_ax14_ssm_attractor_probe
    config.use_ax15_routing_entropy_probe = args.use_ax15_routing_entropy_probe
    config.use_ax16_tiq_probe = args.use_ax16_tiq_probe
    config.use_ax17_cpbtc_probe = args.use_ax17_cpbtc_probe
    config.use_ax18_tfcd_probe = args.use_ax18_tfcd_probe
    config.use_ax19_dcc_probe = args.use_ax19_dcc_probe
    config.use_ax20_itst_probe = args.use_ax20_itst_probe

    dataloader_num_workers = config.dataloader_num_workers
    dataloader_prefetch_factor = config.dataloader_prefetch_factor

    # Run training
    results = train_model(
        config=config,
        dataset_type=args.dataset_type,
        data_dir=args.data_dir,
        num_epochs=1,  # Use max_steps instead
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        base_lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        device=args.device,
        lr_schedule_type=args.lr_schedule,
        use_amp=args.use_amp,
        use_bf16=args.use_bf16,
        use_compile=args.use_compile,
        progressive=args.progressive,
        two_stage_decay=args.two_stage_decay,
        two_stage_drop_pct=args.two_stage_drop_pct,
        two_stage_drop_factor=args.two_stage_drop_factor,
        two_stage_wd_start_pct=args.two_stage_wd_start_pct,
        optimizer_type=args.optimizer_type,
        ste_type=args.ste_type,
        seq_warmup=args.seq_warmup,
        use_sequence_packing=args.use_sequence_packing,
        use_rho1=args.use_rho1,
        rho1_weighting=args.rho1_weighting,
        rho1_keep_ratio=args.rho1_keep_ratio,
        rho1_ref_losses_path=args.rho1_ref_losses_path,
        use_curriculum=args.use_curriculum,
        weight_decay=args.weight_decay,
        ternary_aware_checkpointing=args.ternary_aware_checkpointing,
        use_activation_quant=args.use_activation_quant,
        activation_quant_bits=args.activation_quant_bits,
        beta2=args.beta2,
        fused_optimizer=args.fused_optimizer,
        save_dir=args.output_dir,
        log_every=args.log_every,
        max_memory_mb=args.max_memory_mb,
        num_workers=dataloader_num_workers,
        prefetch_factor=dataloader_prefetch_factor,
        angular_reg_lambda=args.angular_reg_lambda,
        boundary_reg_lambda=args.boundary_reg_lambda,
        use_galore=args.use_galore,
        galore_rank=args.galore_rank,
        galore_update_proj_gap=args.galore_update_proj_gap,
        galore_scale=args.galore_scale,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume_from,
        max_nan_steps=args.max_nan_steps,
        use_liger_ce=not args.no_liger_ce,
        use_caq=args.use_caq,
        caq_alpha=args.caq_alpha,
        use_tpsq=args.use_tpsq,
        tpsq_beta=args.tpsq_beta,
        use_tcd=args.use_tcd,
        use_gumbel_noise=args.use_gumbel_noise,
        gumbel_temperature=args.gumbel_temperature,
        tequila_lambda=args.tequila_lambda,
        use_iaqd=args.use_iaqd,
        use_stp=args.use_stp,
        stp_sigma_ratio=args.stp_sigma_ratio,
        use_ctce=args.use_ctce,
        ctce_beta=args.ctce_beta,
        ctce_delta_ratio=args.ctce_delta_ratio,
        use_dual_rate=args.use_dual_rate,
        dual_rate_peak=args.dual_rate_peak,
        dual_rate_floor=args.dual_rate_floor,
        use_gaste=args.use_gaste,
        gaste_sigma_g=args.gaste_sigma_g,
        use_saq=args.use_saq,
        saq_k=args.saq_k,
        use_rlec=args.use_rlec,
        rlec_rank=args.rlec_rank,
        use_adagc=args.use_adagc,
        adagc_factor=args.adagc_factor,
        adagc_beta=args.adagc_beta,
        use_margin_sparse=args.use_margin_sparse,
        margin_delta=args.margin_delta,
        margin_ema_beta=args.margin_ema_beta,
        margin_refresh_interval=args.margin_refresh_interval,
        use_fused_requantize=args.use_fused_requantize,
        use_fused_gradient_cascade=args.use_fused_gradient_cascade,
        use_info_freeze=args.use_info_freeze,
        info_freeze_alpha=args.info_freeze_alpha,
        info_freeze_warmup=args.info_freeze_warmup,
        use_dead_zone=args.use_dead_zone,
        dead_zone_margin=args.dead_zone_margin,
        dead_zone_lr_decay=args.dead_zone_lr_decay,
        use_acsi_init=args.use_acsi_init,
        acsi_max_batches=args.acsi_max_batches,
        use_commitment_monitor=args.use_commitment_monitor,
        commitment_check_interval=args.commitment_check_interval,
        commitment_plateau_epsilon=args.commitment_plateau_epsilon,
        commitment_plateau_patience=args.commitment_plateau_patience,
        commitment_lr_boost=args.commitment_lr_boost,
        commitment_lr_decay=args.commitment_lr_decay,
        commitment_early_stop_threshold=args.commitment_early_stop_threshold,
        commitment_early_stop=args.commitment_early_stop,
        use_instant=args.use_instant,
        instant_rank=args.instant_rank,
        use_ags=args.use_ags,
        ags_top_k_frac=args.ags_top_k_frac,
        ags_full_sync_interval=args.ags_full_sync_interval,
        ags_delta=args.ags_delta,
        label_smoothing=args.label_smoothing,
        use_grad_noise=args.use_grad_noise,
        grad_noise_eta=args.grad_noise_eta,
        grad_noise_gamma=args.grad_noise_gamma,
        cosine_restart_period=args.cosine_restart_period,
        cosine_restart_decay=args.cosine_restart_decay,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        use_ema_replay=getattr(config, 'use_ema_replay', False),
        ema_replay_lambda=getattr(config, 'ema_replay_lambda', 0.05),
        contractive_reg_lambda=args.contractive_reg_lambda,
        let_teacher_path=args.let_teacher_path,
        let_lambda=args.let_lambda,
        let_stop_step=args.let_stop_step,
        let_layer_k=args.let_layer_k,
        use_uncertainty_loss=args.use_uncertainty_loss,
        uncertainty_n_tasks=args.uncertainty_n_tasks,
        uncertainty_initial_logvar=args.uncertainty_initial_logvar,
        use_sparse_shadow=args.use_sparse_shadow,
        sparse_shadow_margin=args.sparse_shadow_margin,
        sparse_shadow_decay=args.sparse_shadow_decay,
        sparse_shadow_interval=args.sparse_shadow_interval,
        use_sca_sparsity=args.use_sca_sparsity,
        sca_lambda=args.sca_lambda,
        sca_target_sparsity=args.sca_target_sparsity,
        sca_warmup_steps=args.sca_warmup_steps,
        use_fp8_grad_accum=args.use_fp8_grad_accum,
        use_topk_grad_sparsification=args.use_topk_grad_sparsification,
        topk_grad_sparsity=args.topk_grad_sparsity,
        use_cuda_prefetch=args.use_cuda_prefetch,
        auto_batch_size=args.auto_batch_size,
        vram_budget_mb=args.vram_budget_mb,
        use_gradient_sign_consistency=args.use_gradient_sign_consistency,
        grad_sign_window=args.grad_sign_window,
        grad_sign_threshold=args.grad_sign_threshold,
        use_fisher_preconditioning=args.use_fisher_preconditioning,
        fisher_damping=args.fisher_damping,
        use_sign_sgd=args.use_sign_sgd,
        sign_sgd_momentum=args.sign_sgd_momentum,
        use_consensus_gradient_filter=args.use_consensus_gradient_filter,
        consensus_alpha=args.consensus_alpha,
        use_adaptive_quantization=args.use_adaptive_quantization,
        adaptive_quant_sensitivity_threshold=args.adaptive_quant_sensitivity_threshold,
        adaptive_quant_interval=args.adaptive_quant_interval,
        use_nonuniform_checkpointing=args.use_nonuniform_checkpointing,
        use_auto_config=args.use_auto_config,
        use_sqrt_ssm_checkpointing=args.use_sqrt_ssm_checkpointing,
        use_snr_diagnostic=args.use_snr_diagnostic,
        snr_freeze_threshold=args.snr_freeze_threshold,
        snr_freeze_interval=args.snr_freeze_interval,
        use_reversible_layers=args.use_reversible_layers,
        use_combinatorial_finetune=args.use_combinatorial_finetune,
        combinatorial_max_sweeps=args.combinatorial_max_sweeps,
        use_ucb_ternary_training=args.use_ucb_ternary_training,
        ucb_ternary_exploration_c=args.ucb_ternary_exploration_c,
        use_fscr_flip_gate=args.use_fscr_flip_gate,
        fscr_threshold=args.fscr_threshold,
        use_spectral_ssm=args.use_spectral_ssm,
        use_spsa_training=args.use_spsa_training,
        spsa_perturbation_scale=args.spsa_perturbation_scale,
        spsa_lr=args.spsa_lr,
        use_rank_margin_loss=args.use_rank_margin_loss,
        rank_margin=args.rank_margin,
        rank_num_negatives=args.rank_num_negatives,
        use_whittle_index=args.use_whittle_index,
        whittle_update_fraction=args.whittle_update_fraction,
        use_sskd=args.use_sskd,
        sskd_alpha=args.sskd_alpha,
        sskd_temperature=args.sskd_temperature,
        use_dynamic_loss_weighting=args.use_dynamic_loss_weighting,
        dlw_gamma=args.dlw_gamma,
        use_thm_loss=args.use_thm_loss,
        thm_margin=args.thm_margin,
        thm_lambda=args.thm_lambda,
        use_tail_loss=args.use_tail_loss,
        tail_lambda=args.tail_lambda,
        tail_flip_margin=args.tail_flip_margin,
        tail_num_negatives=args.tail_num_negatives,
        use_rcl_loss=args.use_rcl_loss,
        rcl_beta=args.rcl_beta,
        rcl_lambda=args.rcl_lambda,
        rcl_sample_every=args.rcl_sample_every,
        use_ter_loss=args.use_ter_loss,
        ter_loss_mu=args.ter_loss_mu,
        ter_noise_var=args.ter_noise_var,
        use_gcp=args.use_gcp,
        gcp_alpha=args.gcp_alpha,
        use_rafu=args.use_rafu,
        rafu_threshold=args.rafu_threshold,
        rafu_interval=args.rafu_interval,
        use_ibll=config.use_ibll,
        ibll_beta=config.ibll_beta,
        ibll_gamma=config.ibll_gamma,
        ibll_eps=config.ibll_eps,
        ibll_lambda=config.ibll_lambda,
        ibll_sample_every=config.ibll_sample_every,
        use_bfa=config.use_bfa,
        bfa_rho=config.bfa_rho,
        bfa_tau=config.bfa_tau,
        bfa_theta=config.bfa_theta,
        bfa_interval=config.bfa_interval,
        use_mde=config.use_mde,
        mde_eta=config.mde_eta,
        mde_lambda=config.mde_lambda,
        mde_sample_every=config.mde_sample_every,
        use_tecwc=config.use_tecwc,
        tecwc_code_rate=config.tecwc_code_rate,
        tecwc_lambda=config.tecwc_lambda,
        use_sso=config.use_sso,
        sso_rho_target=config.sso_rho_target,
        sso_alpha=config.sso_alpha,
        sso_beta=config.sso_beta,
        sso_interval=config.sso_interval,
        use_sgapd=config.use_sgapd,
        sgapd_initial_temp=config.sgapd_initial_temp,
        sgapd_min_temp=config.sgapd_min_temp,
        sgapd_decay=config.sgapd_decay,
        sgapd_sweeps=config.sgapd_sweeps,
        sgapd_interval=config.sgapd_interval,
        use_ergodic=config.use_ergodic,
        ergodic_alpha=config.ergodic_alpha,
        ergodic_rho_target=config.ergodic_rho_target,
        ergodic_flip_fraction=config.ergodic_flip_fraction,
        ergodic_interval=config.ergodic_interval,
        use_minimax_depth=config.use_minimax_depth,
        minimax_tau=config.minimax_tau,
        minimax_hard=config.minimax_hard,
        minimax_lambda=config.minimax_lambda,
        minimax_sample_every=config.minimax_sample_every,
        use_koopman=config.use_koopman,
        koopman_long_target=config.koopman_long_target,
        koopman_fast_target=config.koopman_fast_target,
        koopman_lambda=config.koopman_lambda,
        koopman_interval=config.koopman_interval,
        koopman_max_dim=config.koopman_max_dim,
        use_td_dl=config.use_td_dl,
        td_lambda_val=config.td_lambda,
        td_dl_weight=config.td_dl_weight,
        td_sample_every=config.td_sample_every,
    )

    return results


if __name__ == "__main__":
    main()
