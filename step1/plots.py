"""
step1/plots.py
Reusable plotting functions for Step 1 tasks.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text


def plot_profit_histogram(scenario_profit, prob, title=None, save_path=None, color="#FF6B35"):
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


def plot_Expected_DA_And_Balancing_Values(lambda_Balancing, lambda_One_Price_DA, hours, save_path, color="#fa9537"):
    """
    Plot of expected day-ahead value, expected balancing value and the difference for each hour to determine the decision of how much to offer in the day-ahead market.
    """
    hours = list(hours)

    plt.figure(figsize=(10, 5))
    plt.plot(hours, [lambda_One_Price_DA[h] for h in hours], marker='o', color='steelblue', label='Expected One-Price DA Value')
    plt.plot(hours, [lambda_Balancing[h] for h in hours], marker='s', color='darkgreen', label='Expected One-Price Balancing Value')
    plt.xlabel('Hour of Day')
    plt.ylabel('Value (€/MWh)')
    plt.title('Expected Day-Ahead price vs expected balancing price')
    plt.xticks(hours, [h + 1 for h in hours])
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def plot_Mean_Wind_Generation_And_DA_Price(wind_mw, lambda_One_Price_DA, lambda_Two_Price_DA, hours, save_path, color = "#fa9537"):
    """
    Plot of mean wind generation and day-ahead price across hours.

    Parameters
    ----------
    wind_mw   : DataFrame of wind power realisations (scenarios × hours)
    lambda_One_Price_DA : DataFrame of one-price day-ahead prices (scenarios × hours)
    lambda_Two_Price_DA : DataFrame of two-price day-ahead prices (scenarios × hours)
    hours     : list of hour IDs to include in the plot
    save_path : Path-like – file to save the figure to
    color     : str – line colour for wind generation
    """
    hours = list(hours)
    mean_wind = wind_mw[hours].mean()

    def _hourly_values(values):
        if isinstance(values, pd.DataFrame):
            if set(hours).issubset(values.columns):
                return values[hours].mean()
            return values.mean(axis=0).reindex(hours)
        if isinstance(values, pd.Series):
            return values.reindex(hours)
        return pd.Series(values).reindex(hours)

    price_one = _hourly_values(lambda_One_Price_DA)
    price_two = _hourly_values(lambda_Two_Price_DA)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Power (MW)', color=color)
    ax1.plot(hours, mean_wind, marker='o', color=color, label='Mean Wind Generation (MW)')
    ax1.plot(hours, price_one.values, marker='s', color='steelblue', label='One-Price Day-Ahead Price')
    ax1.plot(hours, price_two.values, marker='^', color='darkgreen', label='Two-Price Day-Ahead Price')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    plt.title('Mean Wind Generation and Day-Ahead Price Across Hours')
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def plot_hourly_offers(p_DA_values, hours, save_path, title, color="#2196F3"):
    """
    Bar plot of optimal day-ahead offers for each hour.
    """
    hours = list(hours)
    offers = [p_DA_values[h] for h in hours]

    plt.figure(figsize=(10, 5))
    plt.bar(hours, offers, color=color, edgecolor='black', linewidth=0.5)
    plt.xlabel('Hour of Day')
    plt.ylabel('Offer (MW)')
    plt.title('Optimal day-ahead offers')
    plt.title(title)
    plt.xticks(hours, [h + 1 for h in hours])
    plt.ylim(0, 600)
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


def plot_cvar_frontier_With_Both_Models(frontier_single_price_df, frontier_two_price_df, title, save_path):
    """
    Scatter/line plot of expected profit vs CVaR for a beta sweep.

    Parameters
    ----------
    frontier_single_price_df : DataFrame with columns ['beta', 'expected_profit', 'cvar']
    frontier_two_price_df    : DataFrame with columns ['beta', 'expected_profit', 'cvar']
    title                    : str – plot title
    save_path                : Path-like – file to save the figure to
    """
    plt.figure(figsize=(10, 5))
    plt.plot(frontier_single_price_df["cvar"], frontier_single_price_df["expected_profit"],
             marker="o", color="#1f77b4", label="Single-Price Model")
    plt.plot(frontier_two_price_df["cvar"], frontier_two_price_df["expected_profit"],
             marker="s", color="#ff7f0e", label="Two-Price Model")
    for _, row in frontier_single_price_df.iterrows():
        plt.annotate(
            f"beta={row['beta']:.2f}",
            (row["cvar"], row["expected_profit"]),
            textcoords="offset points", xytext=(5, 5)
        )
    offset = 25

    for _, row in frontier_two_price_df.iterrows():
        plt.annotate(
            f"beta={row['beta']:.2f}",
            (row["cvar"], row["expected_profit"]),
            textcoords="offset points", xytext=(5, offset)
        )
        offset -= 5  # adjust offset for the next annotation to avoid overlap
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

def plot_hourly_offers_and_prices(p_DA_values, lambda_Balancing, lambda_DA, hours, save_path, title=None, 
                                   color_offer="#FF6B35", color_DA="steelblue", color_balancing="#4CAF50"):
    """
    Combined plot: bar chart of optimal day-ahead offers (left axis) and
    expected DA vs balancing prices (right axis) for each hour.
    """
    hours = list(hours)
    offers = [p_DA_values[h] for h in hours]
    da_prices = [lambda_DA[h] for h in hours]
    bal_prices = [lambda_Balancing[h] for h in hours]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # left axis: hourly offers
    ax1.bar(hours, offers, color=color_offer, edgecolor='black', linewidth=0.5, alpha=0.7, label='Day-Ahead Offer (MW)')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Offer (MW)', color=color_offer)
    ax1.tick_params(axis='y', labelcolor=color_offer)
    ax1.set_ylim(0, 600)
    ax1.set_xticks(hours)
    ax1.set_xticklabels([h + 1 for h in hours])

    # right axis: expected prices
    ax2 = ax1.twinx()
    ax2.plot(hours, da_prices, marker='o', color=color_DA, label='Expected DA Price (€/MWh)')
    ax2.plot(hours, bal_prices, marker='s', color=color_balancing, label='Expected Balancing Price (€/MWh)')
    ax2.set_ylabel('Price (€/MWh)')

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    #ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.legend(lines1 + lines2, labels1 + labels2,
           bbox_to_anchor=(0.5, 1.02), loc='lower center',
           ncol=3, borderaxespad=0, fontsize=9)
    plt.title(title)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


