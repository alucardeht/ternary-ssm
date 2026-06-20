import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import os
import seaborn as sns

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Palatino', 'serif']
mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['lines.linewidth'] = 1.5

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
FIGURES_DIR = os.path.join(BASE_DIR, 'paper3-efficiency', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def ema_smooth(data, alpha=0.99):
    """Exponential moving average smoothing."""
    if len(data) == 0:
        return data

    smoothed = np.zeros_like(data, dtype=np.float32)
    smoothed[0] = data[0]

    for i in range(1, len(data)):
        smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * data[i]

    return smoothed


def load_checkpoint(path):
    """Load checkpoint with error handling."""
    if not os.path.exists(path):
        return None

    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        return checkpoint
    except Exception as e:
        print(f"  Warning: Failed to load {path}: {e}")
        return None


def generate_fig1():
    """Training Loss Curves — FP16 vs Ternary 50M (5 epochs)."""
    fp16_path = os.path.join(PROJECT_DIR, 'modal-results', '50m_fp16_5e_b32.pt')
    ternary_path = os.path.join(PROJECT_DIR, 'modal-results', '50m_ternary_5e_b32.pt')

    fp16_ckpt = load_checkpoint(fp16_path)
    ternary_ckpt = load_checkpoint(ternary_path)

    if fp16_ckpt is None or ternary_ckpt is None:
        print("  Skipping Fig 1: Missing checkpoint files")
        return

    fp16_losses = np.array(fp16_ckpt['losses'], dtype=np.float32)
    ternary_losses = np.array(ternary_ckpt['losses'], dtype=np.float32)

    fp16_smooth = ema_smooth(fp16_losses, alpha=0.99)
    ternary_smooth = ema_smooth(ternary_losses, alpha=0.99)

    steps = np.arange(len(fp16_losses)) / 1000

    fig, ax = plt.subplots(figsize=(5.5, 2.8))

    ax.plot(steps, fp16_losses, color=COLORS[0], alpha=0.08, linewidth=0.5, label='_nolegend_')
    ax.plot(steps, ternary_losses, color=COLORS[1], alpha=0.08, linewidth=0.5, label='_nolegend_')

    ax.plot(steps, fp16_smooth, color=COLORS[0], linewidth=1.5, label='FP16 Baseline')
    ax.plot(steps, ternary_smooth, color=COLORS[1], linewidth=1.5, label='Ternary (Ours)')

    fp16_final = fp16_smooth[-1]
    ternary_final = ternary_smooth[-1]

    ax.text(0.98, 0.05, f'FP16: {fp16_final:.4f}\nTernary: {ternary_final:.4f}',
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Training Steps (thousands)')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, 'fig1_training_curves_50m.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def generate_fig2():
    """Validation Perplexity — FP16 vs Ternary 50M."""
    fp16_path = os.path.join(PROJECT_DIR, 'modal-results', '50m_fp16_5e_b32.pt')
    ternary_path = os.path.join(PROJECT_DIR, 'modal-results', '50m_ternary_5e_b32.pt')

    fp16_ckpt = load_checkpoint(fp16_path)
    ternary_ckpt = load_checkpoint(ternary_path)

    if fp16_ckpt is None or ternary_ckpt is None:
        print("  Skipping Fig 2: Missing checkpoint files")
        return

    fp16_ppl = fp16_ckpt.get('ppl_history', [])
    ternary_ppl = ternary_ckpt.get('ppl_history', [])

    if not fp16_ppl or not ternary_ppl:
        print("  Skipping Fig 2: Missing ppl_history in checkpoints")
        return

    fp16_steps = np.array([entry['step'] for entry in fp16_ppl]) / 1000
    fp16_vals = np.array([entry['val_ppl'] for entry in fp16_ppl])
    fp16_test = fp16_ppl[-1].get('test_ppl', fp16_vals[-1])

    ternary_steps = np.array([entry['step'] for entry in ternary_ppl]) / 1000
    ternary_vals = np.array([entry['val_ppl'] for entry in ternary_ppl])
    ternary_test = ternary_ppl[-1].get('test_ppl', ternary_vals[-1])

    fig, ax = plt.subplots(figsize=(5.5, 2.8))

    ax.plot(fp16_steps, fp16_vals, color=COLORS[0], linewidth=1.5, marker='o',
            markersize=3, label='FP16 Baseline')
    ax.plot(ternary_steps, ternary_vals, color=COLORS[1], linewidth=1.5, marker='s',
            markersize=3, label='Ternary (Ours)')

    ax.axhline(y=fp16_test, color=COLORS[0], linestyle='--', linewidth=1.0, alpha=0.7)
    ax.axhline(y=ternary_test, color=COLORS[1], linestyle='--', linewidth=1.0, alpha=0.7)

    ax.text(0.98, 0.95, f'Test PPL\nFP16: {fp16_test:.2f}\nTernary: {ternary_test:.2f}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Training Steps (thousands)')
    ax.set_ylabel('Validation Perplexity')
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, 'fig2_val_ppl_50m.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def generate_fig3():
    """Gradient Estimator Ablation — Training Loss at 6M."""
    checkpoints_dir = os.path.join(PROJECT_DIR, '.checkpoints')

    methods = [
        ('a1_ste_baseline_5000.pt', 'STE Baseline (1K)', COLORS[0]),
        ('a1_dqt_2x_5000.pt', 'DQT (2x LR)', COLORS[1]),
        ('a1_dqt_5x_5000.pt', 'DQT (5x LR)', COLORS[2]),
        ('c1_cage_conservative_5000.pt', 'CAGE', COLORS[3]),
        ('e1_qzo_full_5000.pt', 'QZO', COLORS[4]),
        ('c1_ste_baseline_5000.pt', 'STE Baseline (5K)', COLORS[0]),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    ste_5k_plotted = False

    for checkpoint_name, label, color in methods:
        path = os.path.join(checkpoints_dir, checkpoint_name)
        ckpt = load_checkpoint(path)

        if ckpt is None:
            continue

        losses = np.array(ckpt['losses'], dtype=np.float32)

        if 'STE Baseline (5K)' in label and ste_5k_plotted:
            continue
        if 'STE Baseline (5K)' in label:
            ste_5k_plotted = True

        nan_idx = np.where(np.isnan(losses))[0]
        if len(nan_idx) > 0:
            losses = losses[:nan_idx[0]]
            steps = np.arange(len(losses))
            smoothed = ema_smooth(losses, alpha=0.95)
            ax.plot(steps, smoothed, color=color, linewidth=1.5, label=label)
            ax.plot([nan_idx[0]], [smoothed[-1]], marker='x', color=color,
                   markersize=8, markeredgewidth=2)
        else:
            steps = np.arange(len(losses))
            smoothed = ema_smooth(losses, alpha=0.95)
            ax.plot(steps, smoothed, color=color, linewidth=1.5, label=label)

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', fontsize=7)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, 'fig3_gradient_estimators_6m.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def generate_fig4():
    """Direct vs Progressive Training — 6M."""
    checkpoints_dir = os.path.join(PROJECT_DIR, '.checkpoints')

    methods = [
        ('b1_direct_twn_5000.pt', 'Direct Ternary', COLORS[0]),
        ('b1_progressive_5000.pt', 'Progressive (FP16→Ternary)', COLORS[1]),
        ('b1_bitnet_5000.pt', 'BitNet b1.58 Recipe', COLORS[2]),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 2.8))

    direct_final = None
    progressive_final = None

    for checkpoint_name, label, color in methods:
        path = os.path.join(checkpoints_dir, checkpoint_name)
        ckpt = load_checkpoint(path)

        if ckpt is None:
            continue

        losses = np.array(ckpt['losses'], dtype=np.float32)
        steps = np.arange(len(losses))
        smoothed = ema_smooth(losses, alpha=0.95)

        ax.plot(steps, smoothed, color=color, linewidth=1.5, label=label)

        if 'Direct' in label:
            direct_final = smoothed[-1]
        elif 'Progressive' in label:
            progressive_final = smoothed[-1]

    if direct_final is not None and progressive_final is not None:
        diff = abs(direct_final - progressive_final)
        pct_diff = (diff / progressive_final) * 100
        ax.text(0.98, 0.05, f'Δ Loss: {diff:.4f} ({pct_diff:.2f}%)',
                transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, 'fig4_direct_vs_progressive_6m.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


if __name__ == '__main__':
    print("Generating publication-quality figures for test paper...")
    print(f"Output directory: {FIGURES_DIR}\n")

    print("Generating Figure 1: Training curves 50M...")
    generate_fig1()

    print("Generating Figure 2: Validation PPL 50M...")
    generate_fig2()

    print("Generating Figure 3: Gradient estimators 6M...")
    generate_fig3()

    print("Generating Figure 4: Direct vs Progressive 6M...")
    generate_fig4()

    print(f"\nAll figures generated and saved to {FIGURES_DIR}")
