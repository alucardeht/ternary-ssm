#!/usr/bin/env python3
"""Generate all figures for Paper 2: Gradient Estimator Ablation."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

OKABE_ITO = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']
FIGURES_DIR = Path(__file__).parent / 'figures'


def apply_clean_style(ax):
    """Apply consistent clean style to axis."""
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)


def save_figure(fig, name):
    """Save figure as PDF."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def plot_fig1_estimator_comparison():
    """Figure 1: Grouped bar chart comparing all gradient estimators at 6M scale."""
    fig, ax = plt.subplots(figsize=(6, 3.5))

    techniques = [
        'STE\n(baseline)',
        'DQT\n(LR=2x)',
        'DQT\n(LR=5x)',
        'CAGE\n(λ=0.5)',
        'CAGE\n(λ=1.5)',
        'QZO\nScale',
        'QZO\nFull',
        'Alpha-only',
        'IGA'
    ]

    losses = [4.26, 5.36, 5.21, np.nan, np.nan, 4.57, 6.12, 5.25, 4.2474]
    throughputs = [81124, 81000, 81000, 81000, 81000, 56476, 57159, 64738, 100800]

    x = np.arange(len(techniques))
    width = 0.35

    colors = []
    for i, loss in enumerate(losses):
        if techniques[i].startswith('CAGE'):
            colors.append('#D55E00')
        elif techniques[i].startswith('IGA'):
            colors.append('#009E73')
        else:
            colors.append(OKABE_ITO[0])

    bars1 = ax.bar(x - width/2, losses, width, label='Final Loss', color=colors, alpha=0.8)

    ax.axhline(y=4.26, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='STE baseline')

    for i, (bar, loss) in enumerate(zip(bars1, losses)):
        if np.isnan(loss):
            bar.set_hatch('///')
            bar.set_height(8.0)
            ax.text(bar.get_x() + bar.get_width()/2, 8.2, 'NaN', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Final Loss', fontsize=9)
    ax.set_xlabel('Technique', fontsize=9)
    ax.set_title('Gradient Estimator Comparison at 6M Scale', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(techniques, fontsize=8)
    ax.set_ylim(0, 9)

    apply_clean_style(ax)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=8)

    plt.tight_layout(pad=0.5)
    save_figure(fig, 'fig1_estimator_comparison.pdf')


def plot_fig2_training_horizon():
    """Figure 2: Line plot showing alpha saturation vs STE baseline."""
    fig, ax = plt.subplots(figsize=(6, 3.5))

    steps = np.array([500, 1000, 2000, 3000, 4000, 5000])
    ste_loss = np.array([6.39, 5.87, 5.29, 4.97, 4.78, 4.61])
    alpha_loss = np.array([6.22, 5.85, 5.57, 5.44, 5.37, 5.25])

    ax.plot(steps, ste_loss, marker='o', color=OKABE_ITO[0], label='STE baseline', linewidth=1.5, markersize=5)
    ax.plot(steps, alpha_loss, marker='s', color=OKABE_ITO[3], label='Alpha-only (random frozen)', linewidth=1.5, markersize=5)

    ax.axvline(x=1500, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(1500, 6.3, 'Crossover:\nα saturates', fontsize=8, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    gap_pct = ((5.25 - 4.61) / 4.61) * 100
    ax.text(5000, 4.9, f'+{gap_pct:.1f}%', fontsize=8, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Training Steps', fontsize=9)
    ax.set_ylabel('Loss', fontsize=9)
    ax.set_title('Training Horizon Effect: Alpha-Only vs STE', fontsize=10, fontweight='bold')

    apply_clean_style(ax)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='upper right')

    plt.tight_layout(pad=0.5)
    save_figure(fig, 'fig2_training_horizon.pdf')


def plot_fig3_scale_dependency():
    """Figure 3: Two-panel subplot showing scale-dependent reversal of HESTIA."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3.5))

    techniques_6m = ['STE baseline', 'HESTIA']
    loss_6m = [6.39, 5.78]

    x1 = np.arange(len(techniques_6m))
    width = 0.6

    bars1 = ax1.bar(x1, loss_6m, width, color=[OKABE_ITO[0], OKABE_ITO[1]], alpha=0.8)
    ax1.text(0.5, 5.9, '-9.5%', fontsize=9, ha='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax1.set_ylabel('Loss', fontsize=9)
    ax1.set_title('6M (500 steps)', fontsize=9, fontweight='bold')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(techniques_6m, fontsize=8)
    ax1.set_ylim(4, 7)
    apply_clean_style(ax1)

    techniques_50m = ['STE baseline', 'HESTIA']
    loss_50m = [4.62, 4.76]

    x2 = np.arange(len(techniques_50m))
    bars2 = ax2.bar(x2, loss_50m, width, color=[OKABE_ITO[0], OKABE_ITO[1]], alpha=0.8)
    ax2.text(0.5, 4.69, '+3.0%', fontsize=9, ha='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.set_ylabel('Cross-Entropy Loss', fontsize=9)
    ax2.set_title('50M (Full Training)', fontsize=9, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(techniques_50m, fontsize=8)
    ax2.set_ylim(4.0, 5.2)
    apply_clean_style(ax2)

    fig.suptitle('Scale-Dependent Reversal: HESTIA at 6M vs 50M', fontsize=10, fontweight='bold', y=1.02)

    plt.tight_layout(pad=0.5)
    save_figure(fig, 'fig3_scale_dependency.pdf')


def plot_fig4_iga_convergence():
    """Figure 4: Line plot comparing IGA and STE convergence trajectories."""
    fig, ax = plt.subplots(figsize=(6, 3.5))

    steps = np.array([500, 1000, 1500, 2000, 3000, 5000])
    ste_loss = np.array([5.9146, 5.4586, 5.1988, 5.0187, 4.7705, 4.3432])
    iga_loss = np.array([5.9525, 5.4682, 5.1870, 4.9843, 4.6943, 4.2474])

    ax.plot(steps, ste_loss, marker='o', color=OKABE_ITO[0], label='STE', linewidth=1.5, markersize=5)
    ax.plot(steps, iga_loss, marker='s', color=OKABE_ITO[2], label='IGA', linewidth=1.5, markersize=5)

    ax.axvline(x=1500, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(1500, 5.5, 'IGA overtakes\nSTE', fontsize=8, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.text(5000, 4.15, '$-$2.1%', fontsize=8, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Training Steps', fontsize=9)
    ax.set_ylabel('Loss', fontsize=9)
    ax.set_title('Integer Gradient Accumulation vs STE', fontsize=10, fontweight='bold')

    apply_clean_style(ax)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='upper right')

    plt.tight_layout(pad=0.5)
    save_figure(fig, 'fig4_iga_convergence.pdf')


if __name__ == '__main__':
    plot_fig1_estimator_comparison()
    plot_fig2_training_horizon()
    plot_fig3_scale_dependency()
    plot_fig4_iga_convergence()
    print("All figures generated successfully.")
