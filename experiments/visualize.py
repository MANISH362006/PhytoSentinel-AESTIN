"""
Visualization utilities for PhytoSentinel-AESTIN.
Generates plots suitable for the conference paper.

Outputs (saved to results/figures/):
  - edge_uncertainty.png   : edge weight distributions (Bayesian DAGCA)
  - epidemic_spread.png    : disease state evolution over time
  - r0_distribution.png    : R0 distribution across test graphs
  - ablation_bar.png       : ablation comparison bar chart
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS = {
    "ours":     "#2196F3",
    "det":      "#FF9800",
    "baseline": "#9E9E9E",
    "gat":      "#4CAF50",
    "infected": "#F44336",
    "susceptible": "#4CAF50",
    "exposed":  "#FFC107",
    "recovered": "#9E9E9E",
}


def plot_edge_uncertainty(means: np.ndarray, stds: np.ndarray,
                          out: str = "edge_uncertainty.png"):
    """Violin plot of edge weight distributions (mean ± std from Bayesian DAGCA)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(means, bins=40, color=COLORS["ours"], alpha=0.8, edgecolor='white')
    axes[0].set_xlabel("Edge Weight (Posterior Mean)", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Learned Dispersal Weights", fontsize=13, fontweight='bold')
    axes[0].axvline(means.mean(), color='red', linestyle='--', label=f'Mean={means.mean():.3f}')
    axes[0].legend()

    axes[1].scatter(means, stds, alpha=0.4, s=10, color=COLORS["ours"])
    axes[1].set_xlabel("Posterior Mean", fontsize=12)
    axes[1].set_ylabel("Posterior Std (Uncertainty)", fontsize=12)
    axes[1].set_title("Uncertainty vs. Weight", fontsize=13, fontweight='bold')
    high_unc = stds > stds.mean() + stds.std()
    axes[1].scatter(means[high_unc], stds[high_unc],
                    alpha=0.8, s=20, color='red', label='High uncertainty')
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, out)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Fig] Saved: {path}")


def plot_epidemic_spread(states: np.ndarray, out: str = "epidemic_spread.png"):
    """
    Stacked area chart of SEIR state counts over time.
    states: (T, N) with values 0=S,1=E,2=I,3=R
    """
    T, N = states.shape
    fracs = np.array([
        [(states[t] == k).sum() / N for k in range(4)]
        for t in range(T)
    ])  # (T, 4)

    fig, ax = plt.subplots(figsize=(10, 4))
    labels = ['Susceptible', 'Exposed', 'Infected', 'Recovered']
    colors = [COLORS['susceptible'], COLORS['exposed'],
              COLORS['infected'], COLORS['recovered']]

    ax.stackplot(range(T), fracs.T, labels=labels, colors=colors, alpha=0.85)
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Fraction of Nodes", fontsize=12)
    ax.set_title("Plant Disease Spread (SEIR Dynamics)", fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, out)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Fig] Saved: {path}")


def plot_ablation_bar(results: dict, out: str = "ablation_bar.png"):
    """
    Bar chart comparing ablation configurations on F1 and AUROC.
    results: {"Config Name": {"f1": ..., "auroc": ...}, ...}
    """
    names  = list(results.keys())
    f1s    = [results[n]["f1"]    for n in names]
    aurocs = [results[n]["auroc"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width/2, f1s,    width, label='F1 Score',
                   color=COLORS["ours"],     alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, aurocs,  width, label='AUROC',
                   color=COLORS["baseline"], alpha=0.85, edgecolor='white')

    # annotate values
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Ablation Study: PhytoSentinel-AESTIN Configurations",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, out)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Fig] Saved: {path}")


def plot_r0_distribution(r0_values: list, out: str = "r0_distribution.png"):
    """Histogram of R0 values across test graphs, with epidemic threshold line."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(r0_values, bins=30, color=COLORS["ours"], alpha=0.8, edgecolor='white')
    ax.axvline(1.0, color='red', linewidth=2, linestyle='--', label='Epidemic threshold (R₀=1)')
    ax.axvline(np.mean(r0_values), color='orange', linewidth=1.5,
               linestyle='-', label=f'Mean R₀={np.mean(r0_values):.2f}')
    ax.set_xlabel("Basic Reproduction Number (R₀)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("SENR0: Epidemic Threshold Distribution", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, out)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Fig] Saved: {path}")


def demo_all_figures():
    """Generate demo figures with synthetic data (for testing without training)."""
    print("Generating demo figures with synthetic data...")

    np.random.seed(42)
    T, N = 30, 100

    # 1. edge uncertainty
    means = np.random.beta(2, 3, 1000)
    stds  = np.random.beta(1, 5, 1000) * 0.15
    plot_edge_uncertainty(means, stds)

    # 2. epidemic spread
    states = np.zeros((T, N), dtype=int)
    states[0, :5] = 2   # initially infected
    for t in range(1, T):
        for i in range(N):
            if states[t-1, i] == 0 and np.random.random() < 0.05:
                states[t, i] = 1
            elif states[t-1, i] == 1:
                states[t, i] = 2 if np.random.random() < 0.3 else 1
            elif states[t-1, i] == 2:
                states[t, i] = 3 if np.random.random() < 0.1 else 2
            else:
                states[t, i] = states[t-1, i]
    plot_epidemic_spread(states)

    # 3. ablation bar chart
    ablation_results = {
        "BayesDAGCA+SAGE\n(Ours)": {"f1": 0.847, "auroc": 0.912},
        "DetDAGCA+SAGE":           {"f1": 0.821, "auroc": 0.887},
        "NoDAGCA+SAGE":            {"f1": 0.778, "auroc": 0.851},
        "NoDAGCA+GCN":             {"f1": 0.763, "auroc": 0.839},
        "BayesDAGCA+GAT":          {"f1": 0.838, "auroc": 0.905},
    }
    plot_ablation_bar(ablation_results)

    # 4. R0 distribution
    r0_values = np.random.lognormal(mean=0.1, sigma=0.4, size=200)
    plot_r0_distribution(r0_values.tolist())

    print("\nAll demo figures saved to results/figures/")


if __name__ == "__main__":
    demo_all_figures()
