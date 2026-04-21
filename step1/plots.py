"""
step1/plots.py
Reusable plotting functions for Step 1 tasks.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text


def plot_profit_histogram(scenario_profit, prob, title, save_path, color="#fa9537"):
    """
    Histogram of per-scenario profits with a vertical line at the expected profit.

    Parameters
    ----------
    scenario_profit : dict {omega: float}
    prob            : Series of scenario probabilities (indexed by scenario ID)
    title           : str – plot title
    save_path       : Path-like – file to save the figure to
    color           : str – bar colour
    """
    profits = list(scenario_profit.values())
    scenarios = list(scenario_profit.keys())
    expected = sum(prob[omega] * scenario_profit[omega] for omega in scenarios)

    plt.figure(figsize=(10, 5))
    plt.hist(profits, bins=50, color=color, edgecolor='white')
    plt.axvline(
        x=expected, color='red', linestyle='--', linewidth=2,
        label=f' Total expected profit: €{expected:,.0f}'
    )
    plt.legend()
    plt.xlabel("Profit (€)")
    plt.ylabel("Number of scenarios")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_cvar_frontier(frontier_df, title, save_path):
    """
    Scatter/line plot of expected profit vs CVaR for a beta sweep.

    Parameters
    ----------
    frontier_df : DataFrame with columns ['beta', 'expected_profit', 'cvar']
    title       : str – plot title
    save_path   : Path-like – file to save the figure to
    """
    plt.figure(figsize=(10, 5))
    plt.plot(frontier_df["cvar"], frontier_df["expected_profit"],
             marker="o", color="#1f77b4")
    for _, row in frontier_df.iterrows():
        plt.annotate(
            f"beta={row['beta']:.2f}",
            (row["cvar"], row["expected_profit"]),
            textcoords="offset points", xytext=(5, 5)
        )
    plt.xlabel("CVaR (alpha = 0.90)")
    plt.ylabel("Expected Profit (€)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_crossvalidation(results_df, save_path_scatter, save_path_bar):
    """
    Two plots for k-fold cross-validation results:
      1. Scatter: in-sample vs out-of-sample profit per fold
      2. Bar chart: in-sample vs out-of-sample profit per fold

    Parameters
    ----------
    results_df       : DataFrame with columns
                        ['fold', 'insample_profit', 'outsample_profit']
    save_path_scatter : Path-like
    save_path_bar     : Path-like
    """
    n_folds = len(results_df)

    # --- Scatter plot ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(results_df['insample_profit'], results_df['outsample_profit'],
               color='steelblue', s=100, zorder=5)

    min_val = min(results_df['insample_profit'].min(), results_df['outsample_profit'].min())
    max_val = max(results_df['insample_profit'].max(), results_df['outsample_profit'].max())
    margin  = (max_val - min_val) * 0.5
    ax.plot(
        [min_val - margin * 0.7, max_val + margin * 0.7],
        [min_val - margin * 0.7, max_val + margin * 0.7],
        'r--', linewidth=1.5, label='Perfect generalization'
    )

    texts = []
    for _, row in results_df.iterrows():
        texts.append(ax.annotate(
            f"Fold {int(row['fold'] + 1)}",
            (row['insample_profit'], row['outsample_profit']),
            fontsize=12
        ))
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    ax.set_xlabel("In-sample profit (€)")
    ax.set_ylabel("Out-of-sample profit (€)")
    ax.set_title("In-sample vs Out-of-sample profit - 8-fold cross-validation")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path_scatter, dpi=150)
    plt.show()

    # --- Bar plot ---
    x     = np.arange(n_folds)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, results_df['insample_profit'],  width, label='In-sample',     color='steelblue')
    ax.bar(x + width / 2, results_df['outsample_profit'], width, label='Out-of-sample', color='orange')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Profit (€)')
    ax.set_title('In-sample vs Out-of-sample profit per fold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i}' for i in range(n_folds)])
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path_bar, dpi=150)
    plt.show()
