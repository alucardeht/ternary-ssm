#!/usr/bin/env python3
"""Generate figures for Paper 3: Practical Efficiency of Multiply-Free Ternary Neural Networks."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Okabe-Ito colorblind-safe palette
OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def fig1_training_waterfall():
    """Figure 1: Cumulative training throughput improvement waterfall chart (6M, RTX 5070)."""

    techniques = [
        'Baseline',
        'SSD Batched\nScan',
        'BF16\nAutocast',
        'Muon\nOptimizer',
        'Liger\nFused CE',
        'FLA Triton\nScan',
    ]

    # Cumulative multipliers (each builds on previous)
    cumulative = [1.00, 1.30, 1.64, 1.72, 1.78, 2.08]
    # Individual gains
    individual = [0, 0.30, 0.34, 0.08, 0.06, 0.30]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    colors = [OKABE_ITO[7]] + [OKABE_ITO[i] for i in range(5)]

    # Draw waterfall bars
    bottoms = [0] + cumulative[:-1]
    for i in range(len(techniques)):
        if i == 0:
            # Baseline bar from 0 to 1.0
            ax.barh(i, cumulative[i], left=0, color=colors[i], alpha=0.85, height=0.6, edgecolor='white')
            ax.text(cumulative[i] + 0.03, i, f'{cumulative[i]:.2f}x', va='center', fontsize=9, fontweight='bold')
        else:
            # Increment bar
            ax.barh(i, individual[i], left=bottoms[i], color=colors[i], alpha=0.85, height=0.6, edgecolor='white')
            # Show cumulative value
            ax.text(cumulative[i] + 0.03, i, f'{cumulative[i]:.2f}x', va='center', fontsize=9, fontweight='bold')
            # Show individual gain inside or above bar
            if individual[i] > 0.1:
                ax.text(bottoms[i] + individual[i]/2, i, f'+{individual[i]:.2f}x', va='center', ha='center', fontsize=8, color='white', fontweight='bold')

    ax.set_yticks(range(len(techniques)))
    ax.set_yticklabels(techniques, fontsize=9)
    ax.set_xlabel('Cumulative Throughput Multiplier', fontsize=10)
    ax.set_title('Training Throughput Optimization Stack (6M, RTX 5070)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 2.5)
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_training_waterfall.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated fig1_training_waterfall.pdf")


def fig2_inference_kernels():
    """Figure 2: Inference kernel comparison at 50M (RTX 5070). Dual-axis: tok/s + VRAM."""

    variants = ['Python\nEager', 'GemLite\n(W1.58)', 'Triton V1\n(7-op)', 'Triton V1+Opt\n(4-op)', 'Triton V2\nMulti-Row', 'FP16\ncuBLAS']
    toks = [46.3, 54.2, 60.0, 75.6, 55.0, 117.5]
    vram = [313, 81.5, 65.4, 65.4, 65.4, 300]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()

    x = np.arange(len(variants))
    width = 0.5

    # Bar colors: ternary variants in OKABE_ITO, FP16 cuBLAS in gray
    bar_colors = [OKABE_ITO[i] for i in range(5)] + ['#888888']

    bars = ax1.bar(x, toks, width, color=bar_colors, alpha=0.85, edgecolor='white', zorder=3)

    # VRAM line on secondary axis
    ax2.plot(x, vram, 'D-', color=OKABE_ITO[5], markersize=7, linewidth=2, label='VRAM (MB)', zorder=4)

    # Labels on bars
    for i, (t, v) in enumerate(zip(toks, vram)):
        ax1.text(i, t + 2, f'{t:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, fontsize=8)
    ax1.set_ylabel('Throughput (tok/s)', fontsize=10)
    ax1.set_ylim(0, 140)
    ax2.set_ylabel('VRAM (MB)', fontsize=10, color=OKABE_ITO[5])
    ax2.set_ylim(0, 400)
    ax2.tick_params(axis='y', labelcolor=OKABE_ITO[5])

    ax1.set_title('Inference Kernel Comparison (50M, RTX 5070)', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)

    ax1.spines['top'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_inference_kernels.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated fig2_inference_kernels.pdf")


def fig3_cpu_gpu_crossover():
    """Figure 3: CPU vs GPU throughput as function of model scale."""

    # Measured data points
    scales = [6e6, 50e6]  # 6M, 50M parameters
    cpu_toks = [342, 14.1]
    gpu_toks = [255, 47.8]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(scales, cpu_toks, 'o-', color=OKABE_ITO[0], markersize=10, linewidth=2, label='CPU (Intel 225F)', zorder=4)
    ax.plot(scales, gpu_toks, 's-', color=OKABE_ITO[4], markersize=10, linewidth=2, label='GPU (RTX 5070)', zorder=4)

    # Crossover annotation
    ax.axvline(x=25e6, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.annotate('Estimated\ncrossover\n~25M params', xy=(25e6, 150), fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    # Labels on data points
    ax.annotate(f'{cpu_toks[0]} tok/s', xy=(scales[0], cpu_toks[0]), xytext=(10, 10),
                textcoords='offset points', fontsize=8, color=OKABE_ITO[0])
    ax.annotate(f'{gpu_toks[0]} tok/s', xy=(scales[0], gpu_toks[0]), xytext=(10, -15),
                textcoords='offset points', fontsize=8, color=OKABE_ITO[4])
    ax.annotate(f'{cpu_toks[1]} tok/s', xy=(scales[1], cpu_toks[1]), xytext=(10, -10),
                textcoords='offset points', fontsize=8, color=OKABE_ITO[0])
    ax.annotate(f'{gpu_toks[1]} tok/s', xy=(scales[1], gpu_toks[1]), xytext=(10, 10),
                textcoords='offset points', fontsize=8, color=OKABE_ITO[4])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Model Parameters', fontsize=10)
    ax.set_ylabel('Throughput (tok/s)', fontsize=10)
    ax.set_title('CPU vs GPU Inference Throughput', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='center left')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M'))
    ax.grid(True, alpha=0.3, which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_cpu_gpu_crossover.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated fig3_cpu_gpu_crossover.pdf")


def fig4_state_memory():
    """Figure 4: SSM state memory vs Transformer KV-cache across context lengths."""

    context_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]

    # SSM state: constant regardless of context
    ssm_50m = [2.88] * len(context_lengths)  # MB
    ssm_7b = [8.0] * len(context_lengths)     # MB (projected)

    # Transformer KV-cache: 2 * n_layers * d_model * context * 2 bytes / 1e6
    # 50M equivalent: 18 layers, d=640
    kv_50m = [2 * 18 * 640 * ctx * 2 / 1e6 for ctx in context_lengths]
    # 7B: 32 layers, d=4096
    kv_7b = [2 * 32 * 4096 * ctx * 2 / 1e6 for ctx in context_lengths]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(context_lengths, ssm_50m, '--', color=OKABE_ITO[0], linewidth=2, marker='o', markersize=5, label='SSM State (50M)', zorder=4)
    ax.plot(context_lengths, ssm_7b, '--', color=OKABE_ITO[2], linewidth=2, marker='o', markersize=5, label='SSM State (7B, projected)', zorder=4)
    ax.plot(context_lengths, kv_50m, '-', color=OKABE_ITO[4], linewidth=2, marker='s', markersize=5, label='KV-cache (50M equiv.)', zorder=4)
    ax.plot(context_lengths, kv_7b, '-', color=OKABE_ITO[5], linewidth=2, marker='s', markersize=5, label='KV-cache (7B)', zorder=4)

    # VRAM limit reference line
    ax.axhline(y=12288, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(600, 14000, 'RTX 5070 VRAM limit (12 GB)', fontsize=8, color='red', alpha=0.8)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Context Length (tokens)', fontsize=10)
    ax.set_ylabel('Memory (MB)', fontsize=10)
    ax.set_title('State Memory: SSM (O(1)) vs Transformer KV-cache (O(n))', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.grid(True, alpha=0.3, which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_state_memory.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated fig4_state_memory.pdf")


if __name__ == '__main__':
    print("Generating Paper 3 figures...")
    fig1_training_waterfall()
    fig2_inference_kernels()
    fig3_cpu_gpu_crossover()
    fig4_state_memory()
    print("All figures generated.")
